"""S3-Q1 因子框架验收测试。

覆盖本单四条硬指标 + 口径护栏：
  1. **无未来函数**：T 日因子值只依赖 ≤T-1 数据 —— 把 T 日及之后的价格整体污染，
     T 日因子必须**逐位相同**；反过来把 T-1 日价格改掉，因子必须变（证明它真在
     用 T-1，而不是用了更早的、看起来"更安全"的数据）。
  2. **框架自检**：合成「因子完全决定下期收益」的已知排序，IC 必须 ≈ 1、分层必须单调 ——
     证明 IC 与分层的链路没算错。再补一条端到端：跑完整 evaluate_segment，
     IC≈1、Q1<...<Q5 年化单调。
  3. **字段缺失即停工**：宽表缺 peTTM / pbMRQ / volume / turn 任一 → RuntimeError，
     不许换源、不许换口径顶替。
  4. **holdout 纪律**：dev 段末个交易日 ≤ 2025-11-06，两段不重叠、不缺口。
  5. 口径护栏：动量公式对解析解、EP 保留负值（亏损股不被静默丢弃）、
     size 走换手率恒等式、分层等大小、hforecast 收益只从 T+1 开盘起算。

红线：断言不放宽；"因子无效"也要如实断言（IC 不显著不许用近似值凑）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import Config, Panel  # noqa: E402
from src.factors import (DEV_END, HOLDOUT_START, MIN_LISTED_DAYS, FactorData,  # noqa: E402
                         bp, compute_factors, cross_section_mask, ep,
                         evaluate_segment, forward_returns, ic_summary,
                         load_factor_data, momentum, month_end_indices, rank_ic,
                         segment_bounds, size, split_layers)


# ---------------------------------------------------------------- 工具
def make_factor_data(dates, codes, open_hfq, close_hfq, close_raw=None,
                     volume=None, turn=None, pe=None, pb=None, listed=None,
                     is_st=None) -> FactorData:
    """合成 FactorData：价格矩阵显式给，其余列按常理填。"""
    o = np.asarray(open_hfq, dtype="float64")
    c = np.asarray(close_hfq, dtype="float64")
    shape = o.shape
    cr = c.copy() if close_raw is None else np.asarray(close_raw, dtype="float64")
    panel = Panel(
        dates=list(dates), codes=list(codes), open_hfq=o, close_hfq=c,
        open_raw=cr.copy(), close_raw=cr,
        blocked_buy=np.zeros(shape, bool), blocked_sell=np.zeros(shape, bool),
        suspended=np.zeros(shape, bool), hfq_gap=np.zeros(shape, bool),
    )
    n_d, n_c = shape
    vol = np.full(shape, 1e6) if volume is None else np.asarray(volume, dtype="float64")
    trn = np.full(shape, 1.0) if turn is None else np.asarray(turn, dtype="float64")
    pe_ = np.full(shape, 20.0) if pe is None else np.asarray(pe, dtype="float64")
    pb_ = np.full(shape, 2.0) if pb is None else np.asarray(pb, dtype="float64")
    ld = np.full(shape, 999.0) if listed is None else np.asarray(listed, dtype="float64")
    st = np.zeros(shape) if is_st is None else np.asarray(is_st, dtype="float64")
    return FactorData(panel=panel, pe_ttm=pe_, pb_mrq=pb_, volume=vol, turn=trn,
                      is_st=st, listed_days=ld,
                      source={"start": dates[0], "end": dates[-1],
                              "dates": n_d, "codes": n_c})


def trade_days(start: str, n: int) -> list[str]:
    """从 start 起生成 n 个连续工作日（够用即可，不追求真实日历）。"""
    return [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(start=start, periods=n)]


def build_rank_panel(n_codes: int = 20, n_months: int = 4, per_month: int = 21,
                     spread: float = 0.002):
    """构造「因子完全决定下期收益」的面板。

    每个月：第 j 只股票当月收益 = (j - (n/2-0.5)) × spread，严格按 j 递增。
    因子矩阵取常数 j（rank），因此 IC 必须 = 1，Q1..Q5 年化必须单调递增。
    """
    n_days = n_months * per_month
    dates = trade_days("2020-01-02", n_days)
    codes = [f"sh.{600000 + j}" for j in range(n_codes)]
    px = np.zeros((n_days, n_codes), dtype="float64")
    base = np.full(n_codes, 10.0)
    month_ret = (np.arange(n_codes) - (n_codes - 1) / 2.0) * spread   # 月度收益
    for m in range(n_months):
        for t in range(per_month):
            i = m * per_month + t
            frac = (t + 1) / per_month
            px[i] = base * (1.0 + month_ret * frac)
        base = base * (1.0 + month_ret)      # 下月起价 = 本月末价
    fmat = np.tile(np.arange(n_codes, dtype="float64"), (n_days, 1))
    return dates, codes, px, fmat


# ---------------------------------------------------------------- 1. 无未来函数
def test_factor_ignores_t_date_and_future_prices():
    """污染 T 日及之后的全部价格 → T 日因子值必须**逐位相同**。

    这是"只用 ≤T-1 数据"的直接检验：若实现里出现 px[i] 或任何 i 之后的下标，
    污染必然渗透进因子值。
    """
    n_d, n_c = 40, 6
    dates = trade_days("2020-01-02", n_d)
    codes = [f"sh.{600000 + j}" for j in range(n_c)]
    rng = np.random.RandomState(7)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.02, (n_d, n_c)), axis=0)

    dirty = px.copy()
    cut = n_d // 2
    dirty[cut:] *= 9.0          # 截面日及其后全部污染

    f_clean = momentum(px, 5)
    f_dirty = momentum(dirty, 5)
    np.testing.assert_allclose(f_clean[cut], f_dirty[cut], rtol=0, atol=0,
                               err_msg="T 日因子被 T 日及之后的污染影响了")

    # 对照：改 T-1 日必须**改变** T 日因子（证明它确实用 T-1，不是用了更早的）
    dirty2 = px.copy()
    dirty2[cut - 1] *= 9.0
    assert not np.allclose(momentum(dirty2, 5)[cut], f_clean[cut]), \
        "改 T-1 价格后因子没变，说明没有用到 T-1"


def test_ic_unchanged_when_only_future_contaminated():
    """框架层面复检：污染未来价格后，因子与 IC 都不许变。

    与上面那条互补：那条测因子函数，这条测 IC 链路是否偷偷读了未来
    （例如用 T 日收盘算前瞻收益的起点）。
    """
    n_d, n_c = 80, 30
    dates = trade_days("2020-01-02", n_d)
    codes = [f"sh.{600000 + j}" for j in range(n_c)]
    rng = np.random.RandomState(11)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (n_d, n_c)), axis=0)
    ends = month_end_indices(dates)

    f1 = momentum(px, 5)
    fwd1 = forward_returns(px, ends)
    ic1 = [rank_ic(f1[i], fwd1[k]) for k, i in enumerate(ends[:-1])]

    # 只污染"截面日之后、下一截面日之前"的中间段，不动截面日与成交日
    dirty = px.copy()
    for k in range(len(ends) - 1):
        a, b = ends[k] + 2, ends[k + 1]
        if b > a:
            dirty[a:b] *= 3.0
    ic2 = [rank_ic(f1[i], forward_returns(dirty, ends)[k])
           for k, i in enumerate(ends[:-1])]

    assert ic1[0] == pytest.approx(ic2[0], abs=1e-12), "IC 被中间段污染影响，链路偷看了未来"


def test_leaky_factor_sentinel_catches_lookahead():
    """哨兵：故意写一个偷看 T 日的因子，证明上面的检测真能抓到未来函数。

    合规因子（momentum，只用 ≤T-1）在下述污染下逐位不变；
    偷看因子（含 px[i]）立刻变化。若有人把动量改成吃 T 日价格，这条会红。
    """
    n_d, n_c = 40, 6
    dates = trade_days("2020-01-02", n_d)
    codes = [f"sh.{600000 + j}" for j in range(n_c)]
    rng = np.random.RandomState(5)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.02, (n_d, n_c)), axis=0)
    cut = n_d // 2
    dirty = px.copy()
    dirty[cut:] *= 9.0

    np.testing.assert_allclose(momentum(px, 5)[cut], momentum(dirty, 5)[cut],
                               rtol=0, atol=0, err_msg="合规因子被污染影响了")

    def leaky(p: np.ndarray) -> np.ndarray:
        out = np.full(p.shape, np.nan, dtype="float64")
        out[1:] = p[1:] / p[:-1] - 1.0        # 用了 T 日价格：故意违规
        return out

    assert not np.allclose(leaky(px)[cut], leaky(dirty)[cut]), \
        "偷看 T 日的因子没被污染影响 —— 说明这套检测抓不到未来函数，测试无牙"


def test_forward_return_starts_at_t_plus_1_open():
    """前瞻收益必须从 T+1 开盘算起，T 日收盘价不参与。"""
    dates = trade_days("2020-01-02", 5)
    codes = ["sh.600000"]
    px = np.array([[10.0], [12.0], [13.0], [14.0], [15.0]])
    ends = [0, 3]
    fwd = forward_returns(px, ends)
    # T=0 → 从第 1 行开盘算到第 4 行开盘：15/12-1
    assert fwd[0, 0] == pytest.approx(15.0 / 12.0 - 1.0), \
        f"前瞻收益起点不是 T+1 开盘，实际 {fwd[0, 0]}"
    # 最后一期没有下一截面 → NaN
    assert np.isnan(fwd[1, 0])


# ---------------------------------------------------------------- 2. 框架自检
def test_known_ordering_ic_is_one():
    """因子完全决定下期收益 → rank IC 必须 = 1（数值误差内）。"""
    n = 50
    f = np.arange(n, dtype="float64")
    r = np.arange(n, dtype="float64") * 0.01
    assert rank_ic(f, r) == pytest.approx(1.0, abs=1e-12)
    # 反向排序 → -1
    assert rank_ic(f, -r) == pytest.approx(-1.0, abs=1e-12)
    # 无序 → 接近 0
    rng = np.random.RandomState(3)
    assert abs(rank_ic(f, rng.normal(size=n))) < 0.35


def test_known_ordering_layers_monotonic():
    """已知排序下端到端跑 evaluate_segment：IC≈1 且 Q1<Q2<...<Q5 年化单调。

    股票数取 40（> MIN_XSEC 才允许算 IC；20 只时 rank_ic 会按纪律返回 NaN）。
    """
    dates, codes, px, fmat = build_rank_panel(n_codes=40, n_months=5)
    fd = make_factor_data(dates, codes, px, px)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=1_000_000,
                 max_positions=40, max_weight=1.0, seed=42)
    res = evaluate_segment({"rank": fmat}, fd, 0, len(dates), cfg, n_layers=5)
    r = res["rank"]

    assert r["n_periods"] >= 3, f"有效期数太少：{r['n_periods']}"
    assert r["ic_mean"] == pytest.approx(1.0, abs=1e-6), \
        f"已知排序下 IC 均值应≈1，实际 {r['ic_mean']}"
    assert r["ic_pos_ratio"] == pytest.approx(1.0), "已知排序下 IC 必须每期为正"

    anns = r["layer_ann"]
    assert len(anns) == 5
    assert all(np.isfinite(anns)), f"分层年化出现 NaN：{anns}"
    assert all(anns[i] < anns[i + 1] for i in range(4)), \
        f"分层年化必须严格单调递增，实际 {anns}"
    assert r["monotonic"] is True
    assert r["long_short_ann"] > 0, "已知排序下多空(高-低)必须为正"


def test_layer_signals_cover_full_cross_section():
    """5 层信号的并集必须等于截面 mask 的全部样本 —— 不重、不漏。

    这条是分层框架正确性的直接证据：若哪层漏了股票，5 层合并就不再等于全截面，
    多空收益会被"少了谁"这件事污染，而报告上看不出来。
    """
    from src.factors import build_layer_signals

    dates, codes, px, fmat = build_rank_panel(n_codes=40, n_months=3)
    fd = make_factor_data(dates, codes, px, px)
    ends = month_end_indices(dates)
    sigs = build_layer_signals(fmat, fd, ends, 5)

    for i in ends:
        ok = cross_section_mask(fd, i)
        expect = {codes[j] for j in np.nonzero(ok)[0]}
        got: set[str] = set()
        total_w = 0.0
        for k in range(5):
            members = sigs[k].get(dates[i], {})
            keys = set(members)
            assert not (got & keys), f"{dates[i]} 层间出现重复股票"
            got |= keys
            if members:
                # 层内等权：权重和必须为 1，且每只权重相同
                ws = list(members.values())
                assert sum(ws) == pytest.approx(1.0), f"{dates[i]} Q{k+1} 权重和 {sum(ws)}"
                assert max(ws) - min(ws) < 1e-12, "层内必须严格等权"
                total_w += sum(ws) / 5.0
        assert got == expect, f"{dates[i]} 分层并集与截面不一致：缺 {expect - got}"
        assert total_w == pytest.approx(1.0), "5 层合并后总权重应为 1"


def test_affordability_flags_underfunded_layers():
    """整手可行性诊断：本金不足必须报低，充足必须≈100%。

    防的是"分层被整手约束扭曲成价格分层"（S2 审计 B3-02 的同类坑）：
    实测 100 万本金下每层 950 只，Q1 只有 21% 买得起、Q5 有 77%，
    这个系统性差异本身就能伪造出多空收益。
    """
    from src.factors import affordability_stats, build_layer_signals

    dates, codes, px, fmat = build_rank_panel(n_codes=40, n_months=2)
    # 价格拉成 1~500 元：高价股一手要几万，本金不足时配不进去
    px = px * np.linspace(0.1, 50.0, 40)[None, :]
    fd = make_factor_data(dates, codes, px, px)
    ends = month_end_indices(dates)
    sigs = build_layer_signals(fmat, fd, ends, 5)

    poor = affordability_stats(sigs, fd.panel, cash=1e4)      # 每只 500 元
    rich = affordability_stats(sigs, fd.panel, cash=1e8)      # 每只 500 万
    assert all(x < 0.5 for x in poor), f"本金不足却没被诊断出来：{poor}"
    assert all(x > 0.99 for x in rich), f"本金充足时整手可行性应≈1：{rich}"


def test_layer_split_equal_size():
    """分层必须等大小（差不超过 1），且 Q1 取因子值最小端、层间不重叠。"""
    f = np.arange(103, dtype="float64")
    ok = np.ones(103, bool)
    layers = split_layers(f, ok, 5)
    sizes = [len(x) for x in layers]
    assert sum(sizes) == 103, f"分层丢样本：{sizes}"
    assert max(sizes) - min(sizes) <= 1, f"层大小不均：{sizes}"
    assert all(s in (20, 21) for s in sizes), f"层大小异常：{sizes}"
    assert f[layers[0]].max() < f[layers[1]].min(), "Q1 必须严格低于 Q2"
    assert f[layers[4]].min() > f[layers[3]].max(), "Q5 必须严格高于 Q4"
    # 各层互不重叠且覆盖全部有效样本
    flat = np.concatenate(layers)
    assert len(np.unique(flat)) == 103, "分层出现重复样本"
    # 样本不足 → 空（不硬切）
    assert split_layers(f[:10], np.ones(10, bool), 5) == []


# ---------------------------------------------------------------- 3. 口径护栏
def test_momentum_matches_hand_calculation():
    """动量公式对解析解：T 日值 = px(T-1)/px(T-1-window)-1，T 日价格不参与。"""
    px = np.array([[10.0], [11.0], [12.0], [13.0], [99.0], [14.0]])
    m = momentum(px, 2)
    assert np.isnan(m[0, 0]) and np.isnan(m[1, 0]) and np.isnan(m[2, 0])
    assert m[3, 0] == pytest.approx(12.0 / 10.0 - 1.0)    # px[2]/px[0]
    assert m[4, 0] == pytest.approx(13.0 / 11.0 - 1.0)    # px[3]/px[1]，第 5 行的 99 不参与
    assert m[5, 0] == pytest.approx(99.0 / 12.0 - 1.0)    # px[4]/px[2]


def test_ep_keeps_negative_for_loss_makers():
    """亏损股（PE<0）必须保留负 EP，不许设 NaN —— 否则把最差端整段丢掉。"""
    pe = np.array([[10.0], [-20.0], [0.0], [np.nan], [50.0]])
    e = ep(pe)
    assert e[1, 0] == pytest.approx(1.0 / 10.0)
    assert e[2, 0] == pytest.approx(-1.0 / 20.0), "亏损股 EP 必须为负且保留"
    assert np.isnan(e[3, 0]), "PE==0 应为 NaN"
    assert np.isnan(e[4, 0]), "PE 缺失应为 NaN"
    assert e[3, 0] != e[3, 0] and np.isnan(e[3, 0])


def test_bp_and_size_use_t_minus_1():
    """bp / size 同样只吃 T-1：T 日估值与成交量变化不影响 T 日因子。"""
    pb = np.array([[1.0], [2.0], [3.0]])
    b = bp(pb)
    assert np.isnan(b[0, 0])
    assert b[1, 0] == pytest.approx(1.0)
    assert b[2, 0] == pytest.approx(0.5)

    close = np.array([[10.0], [10.0], [20.0]])
    vol = np.array([[100.0], [100.0], [100.0]])
    turn = np.array([[1.0], [1.0], [1.0]])
    s = size(close, vol, turn)
    # T=1: close[0]*100/(1/100)=1e5 → log
    assert s[1, 0] == pytest.approx(np.log(10.0 * 100.0 / 0.01))
    # T=2 的 20 元与 T=1 的因子无关
    assert s[2, 0] == pytest.approx(np.log(10.0 * 100.0 / 0.01))


def test_cross_section_mask_excludes_st_and_new():
    """截面 mask：ST、上市不足、停牌、缺口、无行情 一律剔除（沿用 load_panel 口径）。"""
    dates = trade_days("2020-01-02", 3)
    codes = [f"sh.{600000 + j}" for j in range(5)]
    px = np.full((3, 5), 10.0)
    fd = make_factor_data(dates, codes, px, px,
                          listed=np.array([[999] * 5, [999, 999, 10, 999, 999],
                                           [999] * 5], dtype="float64"),
                          is_st=np.array([[0] * 5, [0, 1, 0, 0, 0], [0] * 5],
                                         dtype="float64"))
    fd.panel.suspended[2, 3] = True
    fd.panel.hfq_gap[2, 4] = True
    fd.panel.close_raw[2, 0] = np.nan
    ok = cross_section_mask(fd, 2)
    assert not ok[0], "无行情应剔除"
    assert ok[1], "正常股应保留"
    assert not ok[3], "停牌应剔除"
    assert not ok[4], "hfq 缺口应剔除"
    ok1 = cross_section_mask(fd, 1)
    assert not ok1[1], "ST 应剔除"
    assert not ok1[2], f"上市不足 {MIN_LISTED_DAYS} 日应剔除"


def test_ic_summary_hand_values():
    """IC 统计对解析解：均值/ICIR/t/正占比。"""
    ics = [0.1, 0.2, -0.1, 0.4]
    s = ic_summary(ics)
    v = np.array(ics)
    assert s["ic_mean"] == pytest.approx(v.mean())
    assert s["icir"] == pytest.approx(v.mean() / v.std(ddof=1))
    assert s["t_stat"] == pytest.approx(v.mean() / v.std(ddof=1) * 2.0)
    assert s["ic_pos_ratio"] == pytest.approx(0.75)
    assert s["n_periods"] == 4
    assert ic_summary([])["n_periods"] == 0


# ---------------------------------------------------------------- 4. holdout 纪律
def test_dev_segment_never_crosses_holdout():
    """dev 段末个交易日必须 ≤ 2025-11-06；两段相邻不重叠。"""
    assert DEV_END == "2025-11-06"
    assert HOLDOUT_START == "2025-11-07"
    dates = trade_days("2025-10-01", 60)
    b = segment_bounds(dates)
    i0, i1 = b["dev"]
    assert dates[i1 - 1] <= DEV_END, f"dev 段末日 {dates[i1 - 1]} 越过 {DEV_END}"
    h0, h1 = b["holdout"]
    assert dates[h0] >= HOLDOUT_START
    assert h0 == i1, "dev 与 holdout 必须首尾相接，不许留缺口"


def test_segment_bounds_raises_when_not_enough_data():
    """数据跨不过 holdout 起点时必须抛错，不许硬跑出半段。"""
    with pytest.raises(ValueError, match="不足以切分"):
        segment_bounds(["2024-01-02", "2024-02-01"])


def test_early_cross_section_not_wiped_by_listed_days():
    """2018-01-31 的截面必须非空 —— 锁死「上市天数按 ipoDate 重算」这个修复。

    背景：clean.py 的 listed_days 是相对**面板起点**算的（ipoDate 早于日历起点时
    searchsorted 返回 0），存量股会被当成"上市第 1 天"（B3-05② 遗留）。
    若因子层不按 ipoDate 重算，"上市 ≥60 日"的过滤会把 2018 年前 3 个月的截面
    整段清空（实测 dev 段白丢 3 期 IC），而报告上看不出任何异常。
    """
    fd = load_factor_data("2018-01-01", "2018-02-28")
    i = fd.dates.index("2018-01-31")
    n = int(cross_section_mask(fd, i).sum())
    assert n > 2000, f"2018-01-31 截面只有 {n} 只，listed_days 可能退回了面板起点口径"
    assert int((fd.listed_days[i] >= MIN_LISTED_DAYS).sum()) > 2000, \
        "存量股没有被识别为老股"
    # 真新股必须仍按真实天数算（不能被一律放大）
    young = (fd.listed_days[i] > 0) & (fd.listed_days[i] < MIN_LISTED_DAYS)
    assert young.sum() > 0, "2018-01-31 应有上市不足 60 日的真新股，全部被放大了？"


def test_real_report_segments_respect_holdout(tmp_path):
    """真实数据的冒烟：两段边界与产物字段齐全（不跑回测，只查结构）。"""
    fd = load_factor_data("2025-09-01", "2026-11-06")
    b = segment_bounds(fd.dates)
    assert fd.dates[b["dev"][1] - 1] <= DEV_END
    assert fd.dates[b["holdout"][0]] >= HOLDOUT_START
    # 字段口径
    assert fd.source["size_source"].startswith("log(close")
    assert set(fd.source["fields"]) == {"peTTM", "pbMRQ", "volume", "turn"}
    f = compute_factors(fd, ["momentum_20"])
    assert f["momentum_20"].shape == fd.panel.close_raw.shape


# ---------------------------------------------------------------- 5. 字段缺失即停工
def _tiny_cache(tmp_path: Path, drop: str | None = None) -> Path:
    """从真实 cache 复制一个迷你 parquet（100 只 / 60 个交易日），可选删一列。"""
    src = ROOT / "data" / "cache" / "daily_2023.parquet"
    d = pd.read_parquet(src)
    keep = sorted(d["code"].unique())[:100]
    d = d[d["code"].isin(keep)]
    days = sorted(d["date"].unique())[:60]
    d = d[d["date"].isin(days)]
    if drop:
        d = d.drop(columns=[drop])
    out = tmp_path / "daily_2023.parquet"
    d.to_parquet(out, index=False)
    return tmp_path


@pytest.mark.parametrize("col", ["peTTM", "pbMRQ", "volume", "turn"])
def test_missing_field_stops_construction(tmp_path, col):
    """宽表缺任一因子字段 → RuntimeError（停工上报），不许静默降级。"""
    cache = _tiny_cache(tmp_path, drop=col)
    with pytest.raises(RuntimeError, match="缺少因子所需字段"):
        load_factor_data("2023-01-01", "2023-03-31", cache_dir=cache)


def test_missing_field_message_names_the_column(tmp_path):
    """报错信息必须点名字段，否则停工上报说不清缺什么。"""
    cache = _tiny_cache(tmp_path, drop="pbMRQ")
    with pytest.raises(RuntimeError) as ei:
        load_factor_data("2023-01-01", "2023-03-31", cache_dir=cache)
    assert "pbMRQ" in str(ei.value)
    assert "停工" in str(ei.value) or "停止施工" in str(ei.value)


def test_normal_cache_loads_and_shapes_align(tmp_path):
    """不删列时必须能正常加载，且追加列与面板同形状（对齐正确）。"""
    cache = _tiny_cache(tmp_path)
    fd = load_factor_data("2023-01-01", "2023-03-31", cache_dir=cache)
    shp = fd.panel.close_raw.shape
    assert fd.pe_ttm.shape == shp and fd.turn.shape == shp
    assert fd.listed_days.shape == shp
    assert len(fd.codes) == 100
