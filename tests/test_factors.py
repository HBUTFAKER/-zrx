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

from src.backtest import Config, Costs, Panel  # noqa: E402
from src.factors import (DEV_END, HOLDOUT_START, MIN_LISTED_DAYS,  # noqa: E402
                         MIN_GROUP_SIZE, FactorData, _layer_shape, align_by_pubdate,
                         bp, build_report, build_validity_note, compute_factors,
                         cross_section_mask, data_quality_stats, ep,
                         evaluate_segment, forward_returns, ic_periods, ic_summary,
                         load_factor_data, momentum, month_end_indices,
                         neutralize_matrix, prepare_fundamentals, rank_ic,
                         size_cross_check,
                         segment_bounds, share_float_matrix, size, size_real,
                         split_layers)


# ---------------------------------------------------------------- 工具
def make_factor_data(dates, codes, open_hfq, close_hfq, close_raw=None,
                     volume=None, turn=None, pe=None, pb=None, listed=None,
                     is_st=None, roe_ttm=None, yoy_ni=None, debt_ratio=None,
                     float_shares=None, industry_gid=None) -> FactorData:
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
                      roe_ttm=roe_ttm, yoy_ni=yoy_ni, debt_ratio=debt_ratio,
                      float_shares=float_shares,
                      industry_gid=(None if industry_gid is None
                                    else np.asarray(industry_gid, dtype="int16")),
                      source={"start": dates[0], "end": dates[-1],
                              "dates": n_d, "codes": n_c})


def trade_days(start: str, n: int) -> list[str]:
    """从 start 起生成 n 个连续工作日（够用即可，不追求真实日历）。"""
    return [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(start=start, periods=n)]


def build_rank_panel(n_codes: int = 20, n_months: int = 4, per_month: int = 21,
                     spread: float = 0.002, start: str = "2020-01-02"):
    """构造「因子完全决定下期收益」的面板。

    每个月：第 j 只股票当月收益 = (j - (n/2-0.5)) × spread，严格按 j 递增。
    因子矩阵取常数 j（rank），因此 IC 必须 = 1，Q1..Q5 年化必须单调递增。
    """
    n_days = n_months * per_month
    dates = trade_days(start, n_days)
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


# ---------------------------------------------------------------- 6. 审计-B5 整改项
def test_ic_periods_never_cross_segment_end():
    """B5-01：dev 段任何计入 IC 的前瞻收益窗终点必须 ≤ DEV_END。

    用边界日期硬断言，不做"大概不超过"：末截面 2025-10-31 的退出日算出来是
    2025-12-01（越过 DEV_END 2025-11-06），这一期必须被丢弃并留下原因。
    """
    dates = trade_days("2025-09-01", 80)
    codes = [f"sh.{600000 + j}" for j in range(3)]
    px = np.full((len(dates), len(codes)), 10.0)
    fd = make_factor_data(dates, codes, px, px)

    dev_last = max(i for i, d in enumerate(dates) if d <= DEV_END)
    i1 = dev_last + 1
    usable, dropped = ic_periods(fd, 0, i1)

    assert usable, "dev 段一期都没有，测试前提不成立"
    for p in usable:
        assert p["exit_date"] <= DEV_END, \
            f"{p['cross_section']} 的 IC 窗口到 {p['exit_date']}，越过 {DEV_END} —— 吃了 holdout 收益"
    crossed = [d for d in dropped if d["reason"] == "exit_beyond_segment_end"]
    assert len(crossed) == 1, f"应恰好丢 1 期，实际 {dropped}"
    last_end = max(i for i in month_end_indices(dates) if i < i1)
    assert crossed[0]["cross_section"] == dates[last_end], \
        f"被丢的应是段内末截面 {dates[last_end]}，实际 {crossed[0]['cross_section']}"
    assert crossed[0]["exit_date"] > DEV_END, "既然丢了，退出日必须真的越过边界"


def test_dev_ic_period_count_real_data():
    """B5-01 真实数据：dev 段 IC 期数比截面数少 1，且末截面被丢的原因是跨段。

    这条锁死"dev 统计不消耗 holdout 收益"：截面 22 期、计入 IC 21 期。
    """
    fd = load_factor_data("2025-06-01", "2026-11-06")
    b = segment_bounds(fd.dates)
    usable, dropped = ic_periods(fd, *b["dev"])
    ends = [i for i in month_end_indices(fd.dates) if b["dev"][0] <= i < b["dev"][1]]
    assert len(usable) == len(ends) - 1, f"截面 {len(ends)} 期，计入 IC {len(usable)} 期"
    assert all(p["exit_date"] <= DEV_END for p in usable)
    reasons = [d["reason"] for d in dropped]
    assert reasons == ["exit_beyond_segment_end"], f"丢弃原因异常：{reasons}"

    res = evaluate_segment({"bp": compute_factors(fd, ["bp"])["bp"]}, fd,
                           *b["dev"], Config(start=fd.dates[0], end=fd.dates[-1],
                                             init_cash=1e6, seed=42), n_layers=5)
    assert res["_meta"]["n_ic_periods"] == len(usable)
    assert res["_meta"]["n_cross_sections"] == len(ends)
    assert len(res["_meta"]["ic_periods_dropped"]) == 1, "产物里必须留痕"


def test_layer_shape_matches_numbers():
    """B5-07：分层形状必须从数字算，不能写死形容词。"""
    s = _layer_shape([0.06, 0.05, 0.08, 0.10, 0.07])
    assert s["max_layer"] == "Q4" and s["min_layer"] == "Q2"
    assert s["max_ann"] == pytest.approx(0.10)
    # 最大相邻落差在 Q4→Q5（10%→7%），不是 Q1→Q2（1pp）
    assert s["max_adjacent_drop"]["from"] == "Q4" and s["max_adjacent_drop"]["to"] == "Q5"
    assert s["max_adjacent_drop"]["pp"] == pytest.approx(3.0)


def test_params_complete_and_reflect_cfg():
    """B5-06：lot_size / max_positions / max_weight / costs 必须入档且取自实际 cfg。

    改默认值必须让产物可见 —— 这里用 lot_size=200 + 滑点 0.2% 的模板跑一遍，
    断言产物参数跟着变、分层年化**确实**变了（不是只改了标签）。
    """
    import argparse

    dates, codes, px, fmat = build_rank_panel(n_codes=45, n_months=18, start="2025-06-02")
    fd = make_factor_data(dates, codes, px, px)
    args = argparse.Namespace(cash=1e6, seed=42, layers=5)

    tmpl = Config(start=dates[0], end=dates[-1], init_cash=1e6, seed=42,
                  lot_size=200, costs=Costs(slippage=0.002))
    rep = build_report(fd, {"rank": fmat}, args, cfg_template=tmpl)
    p = rep["params"]
    assert p["lot_size"] == 200, f"lot_size 没入档：{p.get('lot_size')}"
    assert p["max_positions"] == len(codes), "分层实际持有全部股票，须如实入档"
    assert p["max_weight"] == 1.0
    assert p["costs"]["slippage"] == pytest.approx(0.002), "costs 必须取自 cfg，不许写死"
    assert p["costs"]["commission_rate"] == pytest.approx(0.00025)

    tmpl2 = Config(start=dates[0], end=dates[-1], init_cash=1e6, seed=42,
                   lot_size=100, costs=Costs(slippage=0.002))
    rep2 = build_report(fd, {"rank": fmat}, args, cfg_template=tmpl2)
    assert rep2["params"]["lot_size"] == 100
    a1 = rep["segments"]["dev"]["rank"]["layer_ann"]
    a2 = rep2["segments"]["dev"]["rank"]["layer_ann"]
    assert not np.allclose(a1, a2, equal_nan=True), \
        f"lot_size 100→200 分层收益完全没变，参数没真正生效：{a1} vs {a2}"


def test_size_jump_stats_reproducible(tmp_path):
    """B5-03：跳变统计口径可复现（合成数据手算对拍）。

    合成 2 只股票 5 个交易日：A 在第 5 天（月末前 3 天）股本翻倍 → 记 1 跳且落进月末窗口；
    B turn 缺失（停牌）→ 只进缺失率、不进跳变（不能与上一有效观测比出一个假跳）。
    """
    rows = []
    for i, d in enumerate(["2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31"]):
        sh_a = 1e8 * (2.0 if i == 4 else 1.0)          # 第 5 天翻倍
        rows.append({"code": "sh.600000", "date": d, "close": 10.0,
                     "volume": sh_a * 0.01, "turn": 1.0, "suspended": False})
        rows.append({"code": "sh.600001", "date": d, "close": 20.0,
                     "volume": 0.0, "turn": np.nan, "suspended": True})
    pd.DataFrame(rows).to_parquet(tmp_path / "daily_2020.parquet", index=False)

    st = data_quality_stats("2020-01-01", "2020-12-31", cache_dir=tmp_path, month_end_days=7)
    sj, tm = st["size_jump"], st["turn_missing"]
    assert sj["rows"] == 1, f"应恰好 1 跳，实际 {sj['rows']}"
    assert sj["codes"] == 1
    assert sj["month_end_rows"] == 1, "2020-01-31 距月末 0 天，必须落进月末窗口"
    assert tm["missing"] == 5 and tm["rate"] == pytest.approx(0.5)
    assert tm["all_suspended"] is True, "缺失行全部是停牌行"
    assert st["scope"]["codes"] == 2


def test_validity_note_is_generated_from_stats():
    """B6-04：免责里的数字必须来自实测产物，不许写死。

    旧断言直接读模块常量 VALIDITY_NOTE，等于把 8,628 / <0.01% 这些数字钉死在代码里 ——
    改了默认值产物照样说旧话（审计-B6-01/04 的原话）。改成对构造函数断言：
    ① 传什么数字就出现什么数字；② 不传就不许出现任何具体数字；
    ③ 失效的旧表述（2~7%、0.2~0.3pp、<0.01%）永远不许再出现。
    """
    note = build_validity_note(
        size_jump={"rows": 8628, "codes": 3693, "month_end_rows": 2788,
                   "month_end_window_days": 7,
                   "examples": [{"code": "sz.001289", "date": "2025-01-24",
                                 "share_ratio": 14.588}]},
        lot={"available": True, "mean_abs_pp": 0.12, "median_abs_pp": 0.07,
             "p90_abs_pp": 0.29, "signed_ann_pp": -0.42,
             "partial_gap_min_pct": 0.70, "partial_gap_max_pct": 2.27,
             "cash_median_pct": 0.01, "cash_max_pct": 3.88},
        turn_missing={"rate": 0.007131, "missing": 68953, "worst_year": "2018",
                      "all_suspended": True, "by_year": {"2018": {"rate": 0.046586}}},
        size_cmp={"dev": {"median_abs": 0.0001, "p90_abs": 0.0003}})
    assert "8628" in note and "3693" in note and "2788" in note, "实测跳变量没进免责"
    assert "3.88" in note, "未投入现金的最大值必须按实测写（B6-01 的 0.01% 已被证伪）"
    assert "sz.001289" in note and "14.588" in note
    assert "自相关" in note, "t 值高估的免责丢了"
    assert "2~7%" not in note and "0.2~0.3pp" not in note and "<0.01%" not in note, \
        "旧的失实表述不许回流"

    bare = build_validity_note()
    for token in ("8628", "8,628", "3.88", "0.12pp"):
        assert token not in bare, f"未传统计量时不许出现具体数字：{token}"


def test_ic_summary_docstring_warns_autocorrelation():
    """B5-07③：t 值的自相关免责必须写在 ic_summary 里，不能只写在报告正文。"""
    doc = ic_summary.__doc__ or ""
    assert "自相关" in doc and "高估" in doc
    assert "Newey" in doc or "bootstrap" in doc, "免责要给出替代做法，不然只是免责不改正"


# ---------------------------------------------------------------- 7. 004B：公告日纪律
def _fin_row(code, statDate, pubDate, np_, eq, debt=np.nan, yoy=np.nan, q=None):
    return {"code": code, "statDate": statDate, "pubDate": pubDate,
            "pubDate_est": pubDate, "pubdate_src": "announce",
            "quarter": q if q is not None else float(statDate[5:7]) / 3,
            "netProfit": np_, "total_equity": eq, "debt_ratio": debt,
            "YOYNI": yoy, "epsTTM": np.nan}


def test_pubdate_rule_report_period_before_announcement():
    """核心反例：报告期早、公告晚的财报，公告日之前**不得**进入因子值。

    2019 年报（statDate 2019-12-31）到 2020-04-30 才公告。若实现拿 statDate 当时点，
    2020-01-15 就已经"知道"全年业绩 —— 这是典型未来函数。这里用日期硬断言。
    """
    # 用 debt_ratio 做时间轴断言：它不需要 4 个季度也能算，能把"时点"与"TTM 口径"
    # 两个问题分开验证（TTM 另有专门测试）
    fin = pd.DataFrame([
        _fin_row("sh.600000", "2019-09-30", "2019-10-30", np_=90.0, eq=1000.0, debt=60.0),
        _fin_row("sh.600000", "2019-12-31", "2020-04-30", np_=120.0, eq=1100.0, debt=55.0),
    ])
    rec, stats = prepare_fundamentals(fin)
    assert len(rec) == 2 and stats["pubdate_missing_rows"] == 0

    dates = ["2019-10-29", "2019-10-30", "2020-01-15", "2020-04-29", "2020-04-30"]
    m = align_by_pubdate(rec, dates, ["sh.600000"], "debt_ratio")
    assert np.isnan(m[0, 0]), "公告日之前不该有任何值"
    assert m[1, 0] == pytest.approx(60.0), "公告当日即生效（Q3）"
    assert m[2, 0] == pytest.approx(60.0), \
        "2020-01-15 必须仍是 Q3 —— 全年业绩还没公告，绝不能用 statDate 提前知道"
    assert m[3, 0] == pytest.approx(60.0), "年报公告前一天仍是 Q3"
    assert m[4, 0] == pytest.approx(55.0), "公告当日起切到年报"


def test_pubdate_falls_back_to_estimate_and_is_counted():
    """没有实际公告日时回退 pubDate_est，并在 stats 里如实计数（不许静默）。"""
    fin = pd.DataFrame([
        {"code": "sh.600000", "statDate": "2019-12-31", "pubDate": None,
         "pubDate_est": "2020-04-30", "pubdate_src": "deadline_est", "quarter": 4.0,
         "netProfit": 120.0, "total_equity": 1100.0, "debt_ratio": 60.0,
         "YOYNI": 5.0, "epsTTM": np.nan},
    ])
    rec, stats = prepare_fundamentals(fin)
    assert stats["pubdate_missing_rows"] == 1
    assert stats["pubdate_missing_rate"] == pytest.approx(1.0)
    m = align_by_pubdate(rec, ["2020-04-29", "2020-04-30"], ["sh.600000"], "roe_ttm")
    assert np.isnan(m[0, 0]) and np.isfinite(m[1, 0])


def test_ttm_net_profit_is_four_quarters_not_ytd():
    """roe_ttm 必须用 TTM 净利润，不是累计值。

    2019 Q1~Q4 累计 10/20/30/40，2020 Q1 累计 12。
    TTM(2020Q1) = 12 + 40 − 10 = 42；若错用累计值会得到 12（差 3.5 倍）。
    """
    rows = [_fin_row("sh.600000", d, d, np_=v, eq=100.0, q=q)
            for (d, v, q) in (("2019-03-31", 10.0, 1.0), ("2019-06-30", 20.0, 2.0),
                              ("2019-09-30", 30.0, 3.0), ("2019-12-31", 40.0, 4.0))]
    rows.append(_fin_row("sh.600000", "2020-03-31", "2020-04-30", np_=12.0, eq=100.0,
                         q=1.0))
    rec, _ = prepare_fundamentals(pd.DataFrame(rows))
    m = align_by_pubdate(rec, ["2020-05-06"], ["sh.600000"], "roe_ttm")
    assert m[0, 0] == pytest.approx(0.42), f"应为 TTM 42/100=0.42，实际 {m[0, 0]}"


def test_fundamentals_missing_field_stops_work():
    """缺财报字段必须抛 RuntimeError（停工上报），不许换字段顶替。"""
    bad = pd.DataFrame([{"code": "sh.600000", "statDate": "2019-12-31",
                         "pubDate": "2020-04-30", "pubDate_est": "2020-04-30",
                         "quarter": 4.0, "netProfit": 1.0, "debt_ratio": 1.0,
                         "YOYNI": 1.0}])           # 缺 total_equity
    with pytest.raises(RuntimeError, match="停工上报"):
        prepare_fundamentals(bad)


# ---------------------------------------------------------------- 8. 004B：事件填充边界
def test_share_float_matrix_event_boundaries(tmp_path):
    """事件级股本的填充边界：事件前 NaN、事件当日生效、之后沿用、次一事件当日切换。"""
    ev = pd.DataFrame({
        "code": ["sh.600000"] * 3,
        "event_date": pd.to_datetime(["2020-01-06", "2020-01-08", "2020-01-08"]),
        "float_share_a": [100.0, np.nan, np.nan],
        "float_share": [100.0, 200.0, 250.0],
        "reason": ["首发限售股份上市", "增发A股上市", "增发A股上市"],
    })
    p = tmp_path / "share_events.parquet"
    ev.to_parquet(p, index=False)
    dates = ["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
    m, st = share_float_matrix(dates, ["sh.600000"], path=p)
    col = m[:, 0]
    assert np.isnan(col[0]), "首个事件之前没有股本 → NaN，不许拿 0 或猜"
    assert col[1] == 100.0, "事件当日即生效"
    assert col[2] == 100.0, "事件之间沿用前值"
    assert col[3] == 250.0, "次日事件当日切换；同日多条取最后一条"
    assert col[4] == 250.0
    assert st["codes_coverage"] == 1.0


def test_share_float_prefers_listed_a_share(tmp_path):
    """float_share_a 优先（换手率的分母口径），缺失才回退 float_share。"""
    ev = pd.DataFrame({
        "code": ["sh.600000"] * 2,
        "event_date": pd.to_datetime(["2020-01-06", "2020-01-08"]),
        "float_share_a": [100.0, 300.0],
        "float_share": [999.0, 999.0],
        "reason": ["a", "b"],
    })
    p = tmp_path / "share_events.parquet"
    ev.to_parquet(p, index=False)
    m, _ = share_float_matrix(["2020-01-06", "2020-01-08"], ["sh.600000"], path=p)
    assert m[0, 0] == 100.0 and m[1, 0] == 300.0, "不该用已流通股份（含 B/H 股口径）"


def test_share_float_zero_a_share_falls_back(tmp_path):
    """float_share_a=0 不算"有值"：实测 1,373 行如此，回退 float_share 别整行丢掉。

    按"非缺失优先"取 0，这些事件会被 `value > 0` 过滤掉，等于该股一开始没有股本，
    size_real 白空一段 —— 而 0 的语义是"A 股还没流通"，不是"股本为零"。
    """
    ev = pd.DataFrame({
        "code": ["sh.600000"] * 2,
        "event_date": pd.to_datetime(["2020-01-06", "2020-01-08"]),
        "float_share_a": [0.0, 300.0],
        "float_share": [200.0, 999.0],
        "reason": ["仅B股", "A股上市"],
    })
    p = tmp_path / "share_events.parquet"
    ev.to_parquet(p, index=False)
    m, st = share_float_matrix(["2020-01-06", "2020-01-08"], ["sh.600000"], path=p)
    assert m[0, 0] == 200.0, "a 股流通为 0 时要用已流通股份，不能当成缺失丢掉"
    assert m[1, 0] == 300.0
    assert st["events_rows"] == 2


def test_size_real_equals_size_when_shares_consistent():
    """两口径在股本等于推算值时**逐位相同**，差别只能来自股本来源。"""
    n_d, n_c = 30, 4
    dates = trade_days("2020-01-02", n_d)
    codes = [f"sh.{600000 + j}" for j in range(n_c)]
    rng = np.random.RandomState(3)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (n_d, n_c)), axis=0)
    vol = np.tile(np.array([1e6, 2e6, 3e6, 4e6]), (n_d, 1))
    turn = vol / 1e8 * 100.0                      # 反推股本恒为 1e8
    fd = make_factor_data(dates, codes, px, px, volume=vol, turn=turn,
                          float_shares=np.full((n_d, n_c), 1e8))
    a = size(fd.panel.close_raw, fd.volume, fd.turn)
    b = size_real(fd.panel.close_raw, fd.float_shares)
    assert np.allclose(a, b, equal_nan=True), "股本一致时两口径必须完全相同"


def test_size_real_ignores_turn_rounding():
    """turn 被舍入/异常污染时，真实股本口径不受影响（这正是换口径的收益之一）。"""
    n_d, n_c = 10, 2
    dates = trade_days("2020-01-02", n_d)
    px = np.full((n_d, n_c), 10.0)
    vol = np.full((n_d, n_c), 1e6)
    turn = np.full((n_d, n_c), 1.0)
    turn[5, 0] = 1.45                            # 低换手日的舍入放大（+45%）
    fd = make_factor_data(dates, codes=["sh.600000", "sz.000001"], open_hfq=px,
                          close_hfq=px, volume=vol, turn=turn,
                          float_shares=np.full((n_d, n_c), 1e8))
    a = size(fd.panel.close_raw, fd.volume, fd.turn)[:, 0]
    b = size_real(fd.panel.close_raw, fd.float_shares)[:, 0]
    # 因子都是 T-1 口径：turn[5] 的污染在 a[6] 上体现
    assert a[6] < b[6] - 0.3, "推算口径被 turn 放大拉低了市值"
    assert b[6] == pytest.approx(b[5]), "真实股本口径不受 turn 影响"


# ---------------------------------------------------------------- 9. 004B：行业中性化
def _two_group_panel():
    """两个行业组各 6 只，因子 = 行业位移 + 组内差异。"""
    n_d, per = 45, 6
    dates = trade_days("2020-01-02", n_d)
    codes = [f"sh.{600000 + j}" for j in range(2 * per)]
    px = np.full((n_d, len(codes)), 10.0)
    f = np.zeros((n_d, len(codes)))
    f[:, :per] = 1.0 + np.arange(per) * 0.01      # 组 A 整体偏高
    f[:, per:] = 0.0 + np.arange(per) * 0.01      # 组 B
    gid = np.zeros((n_d, len(codes)), dtype="int16")
    gid[:, per:] = 1
    return dates, codes, px, f, gid


def test_neutralize_removes_industry_level():
    """demean 后每组的截面均值必须≈0，组间水平差被拿掉、组内排序保留。"""
    dates, codes, px, f, gid = _two_group_panel()
    fd = make_factor_data(dates, codes, px, px, industry_gid=gid)
    ends = month_end_indices(dates)
    out, st = neutralize_matrix(f, fd, ends, gid, method="demean")
    assert st["n_cells_neutralized"] > 0
    for i in ends:
        ok = cross_section_mask(fd, i)
        for g in (0, 1):
            sel = ok & (gid[i] == g)
            assert abs(out[i][sel].mean()) < 1e-12, "组均值没被减干净"
    # 组内排序不变（单调变换）
    i = ends[0]
    a = f[i][:6]
    b = out[i][:6]
    assert np.all(np.argsort(a) == np.argsort(b)), "中性化不能改变组内排序"


def test_neutralize_leaves_small_group_and_non_xsec_rows():
    """组内样本 <min_group 不动；非截面日的行原样复制（不逐日中性化）。"""
    dates, codes, px, f, gid = _two_group_panel()
    gid = gid.copy()
    gid[:, 11] = 7                                # 造一个只有 1 只的组
    fd = make_factor_data(dates, codes, px, px, industry_gid=gid)
    ends = month_end_indices(dates)
    out, st = neutralize_matrix(f, fd, ends, gid, method="demean")
    i = ends[0]
    assert out[i][11] == f[i][11], "小组必须原样保留（减组均值等于减噪声本身）"
    assert st["n_small_group_cells"] >= len(ends)
    mid = ends[0] + 3                             # 非截面日
    assert np.allclose(out[mid], f[mid], equal_nan=True), "非截面日不许改"


def test_neutralize_zscore_unit_std():
    """zscore 后组内标准差≈1。"""
    dates, codes, px, f, gid = _two_group_panel()
    fd = make_factor_data(dates, codes, px, px, industry_gid=gid)
    ends = month_end_indices(dates)
    out, _ = neutralize_matrix(f, fd, ends, gid, method="zscore")
    i = ends[0]
    ok = cross_section_mask(fd, i)
    for g in (0, 1):
        v = out[i][ok & (gid[i] == g)]
        assert v.std(ddof=1) == pytest.approx(1.0)


# ---------------------------------------------------------------- 10. 004B：size 新旧口径对拍
def test_size_cross_check_zero_and_double():
    """对拍函数：两口径一致时差 0；真实股本翻倍时市值差应为 +100%。"""
    n_d, n_c = 45, 6
    dates = trade_days("2020-01-02", n_d)
    codes = [f"sh.{600000 + j}" for j in range(n_c)]
    px = np.full((n_d, n_c), 10.0)
    base = np.full((n_d, n_c), 1e8)
    ends = month_end_indices(dates)
    fd_same = make_factor_data(dates, codes, px, px, float_shares=base)
    a = size_real(px, base)
    same = size_cross_check(fd_same, a, a.copy(), ends)
    assert same["available"] and same["median_abs"] == pytest.approx(0.0)

    fd2 = make_factor_data(dates, codes, px, px, float_shares=base * 2.0)
    b = size_real(px, base * 2.0)
    doubled = size_cross_check(fd2, a, b, ends)
    assert doubled["median_abs"] == pytest.approx(1.0), "股本翻倍 → 市值差正好 100%"
    assert doubled["share_gt30pct"] == pytest.approx(1.0)


# ---------------------------------------------------------------- 11. 审计-B6 四项
def test_size_jump_examples_keep_paired_reversal(tmp_path):
    """B6-03：成对的脏跳变必须成对出现，不许只留幅度最大的那一条。

    造 sz.001289 形状的数据：第 6 天股本 ×2.59、第 8 天 ×0.386（两天又变回去 ≈1）。
    旧实现每股只取 max|ratio−1|，只剩 ×2.59 一条，读起来像真实解禁。
    """
    dates = trade_days("2020-01-02", 12)
    shares = np.full(12, 1e8)
    shares[5:] = 1e8 * 2.592
    # 第二次是**相对前值**的 0.386 倍（2.592×0.386≈1），不是绝对 0.386 亿股
    shares[7:] = 1e8 * 2.592 * 0.386
    rows = [{"code": "sz.001289", "date": d, "close": 10.0,
             "volume": shares[i] * 0.01, "turn": 1.0, "suspended": False}
            for i, d in enumerate(dates)]
    pd.DataFrame(rows).to_parquet(tmp_path / "daily_2020.parquet", index=False)

    st = data_quality_stats("2020-01-01", "2020-12-31", cache_dir=tmp_path)
    ex = [e for e in st["size_jump"]["examples"] if e["code"] == "sz.001289"]
    assert len(ex) >= 2, f"双跳变被截断成 {len(ex)} 条"
    paired = [e for e in ex if e["paired_reversal"]]
    assert paired, "成对反转没被标出来"
    assert paired[0]["pair_date"] == ex[1]["date"], "配对日期要写清"
    assert abs(ex[0]["share_ratio"] * ex[1]["share_ratio"] - 1.0) < 0.05


def test_report_records_dual_commit_fields(tmp_path):
    """B6-02：产物必须同时记 code_commit / code_dirty / generated_at，落盘再补 written_*。"""
    from src.factors import write_report
    dates, codes, px, fmat = build_rank_panel(n_codes=45, n_months=6, start="2025-06-02")
    fd = make_factor_data(dates, codes, px, px)

    import argparse
    args = argparse.Namespace(cash=1e6, seed=42, layers=5, neutralize_method="demean")
    rep = build_report(fd, {"rank": fmat}, args)
    assert rep["code_commit"] and len(rep["code_commit"]) >= 7
    assert rep["git_commit"] == rep["code_commit"], "旧字段保留作别名，不许断链"
    assert isinstance(rep["code_dirty"], bool)
    assert rep["generated_at"]

    p = write_report(rep, tmp_path / "r.json")
    import json as _json
    d = _json.loads(p.read_text(encoding="utf-8"))
    assert d["written_at"] and d["written_code_commit"], "落盘要补落盘时刻的版本"
    assert d["segments"]["dev"]["_meta"]["neutralization"]["enabled"] is False, \
        "没有行业数据时必须如实标未启用，不许假装做了中性化"


def test_lot_residual_reports_cash_median_not_just_claim(tmp_path):
    """B6-01：现金结论必须来自实测字段（中位/最大都在），免责里不许再写死 0.01%。"""
    dates, codes, px, fmat = build_rank_panel(n_codes=45, n_months=18, start="2025-06-02")
    fd = make_factor_data(dates, codes, px, px)
    import argparse
    args = argparse.Namespace(cash=1e6, seed=42, layers=5, neutralize_method="demean")
    rep = build_report(fd, {"rank": fmat}, args)
    lot = rep["segments"]["dev"]["rank"]["lot_residual"]
    assert {"cash_median_pct", "cash_max_pct", "cash_p95_pct"} <= set(lot), \
        "现金指标必须同时给中位/95 分位/最大，只报 min-max 才会写出失实的「<0.01%」"
    assert lot["cash_max_pct"] >= lot["cash_median_pct"] >= 0
    assert "0.01% 以下" not in rep["validity_note"]
    assert f"{lot['cash_max_pct']}" in rep["validity_note"], \
        "免责里的现金数字必须与产物字段一致（动态生成）"

def test_jump_sentence_uses_measured_ratio_not_old_sample():
    """B6-04 续：跳变归因必须写全样本实测比例，不许再抄 004B-P 的「12/14」抽样。"""
    from src.factors import _jump_sentence
    ja = {"available": True, "jump_cells": 8628, "both_jump_cells": 7804,
          "both_jump_ratio": 0.9045, "real_coverage": 1.0}
    txt = _jump_sentence(ja)
    assert f"{ja['both_jump_ratio']:.1%}" in txt and "8628" in txt and "7804" in txt
    assert "12/14" not in txt
    # 未提供对拍结果时只指向产物字段，不编数字
    assert "jump_agreement" in _jump_sentence({"available": False})
