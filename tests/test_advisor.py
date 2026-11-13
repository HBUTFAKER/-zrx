"""005R 建议卡引擎测试（分位带分散组合）。

合成面板 + 真引擎（backtest.run_backtest），不打桩核心计算。
覆盖：等权分位带取数、档位选择只看定档段（无前视）、±1 档邻居、可投池等权基准、
月度换仓清单、预算分配与最小可行预算、分年验证窗口、门槛规则、config 解析、
以及 005 遗留的否决/过滤/评分口径回归。

合成数据的设计意图（不写清楚下次看不懂）：
  · `trap` 组（前 8 只）：pb 低（bp 高）且价格持续下跌 —— 既是"最便宜"又是"跌最多"，
    正好落在综合分的顶端。让它们**继续跌** → 所有档位都跑输 → 测 fallback；
    让它们**上涨** → 跑赢 → 测 advise。
  · 其余 32 只走平（每日 −0.2%），行业分两组各 20 只（行业内分位要求组内 ≥5 只）。
  · 所有票价格都是 `10 × (1+drift)^i`，所以"日收益 = drift"是精确值，
    可投池等权的日收益可以直接手算校验（见 test_pool_benchmark_is_equal_weight_of_pool）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.config as C
from src.advisor import (_rebalance_block, _seg_window, analyze, band_neighbors,
                         budget_plan, build_card, build_signals, build_universe,
                         composite_scores, evaluate_gates, export_daily_nav, factor_mats,
                         pool_equal_weight, run_segment, scan_bands, score_cross_sections,
                         select_band, select_top_k, side_band_signals,
                         signals_from_cache, trade_amount, yearly_validation)
from src.backtest import Panel
from src.factors import FactorData

BANDS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
N_TRAP = 8
N_TOTAL = 40


# ---------------------------------------------------------------- 合成面板
def _dates(months: int = 27, start_year: int = 2024) -> list[str]:
    """每月两个交易日（15 日 / 28 日），月末索引落在 28 日那个。"""
    out = []
    y, m = start_year, 1
    for _ in range(months):
        out.append(f"{y}-{m:02d}-15")
        out.append(f"{y}-{m:02d}-28")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def make_fd(*, trap_up: bool, months: int = 27, n_codes: int = N_TOTAL,
            roe_zero: tuple[int, ...] = (), debt_high: tuple[int, ...] = (),
            volume: float = 1e7) -> FactorData:
    """合成 FactorData。trap_up=True 时"廉价暴跌组"改为一路上涨。"""
    dates = _dates(months)
    n_d, n_c = len(dates), n_codes
    codes = [f"sh.{600000 + i}" for i in range(n_codes)]

    px = np.zeros((n_d, n_c))
    holdout_i = next((i for i, d in enumerate(dates) if d >= "2025-11-07"), n_d)
    for j in range(n_c):
        drift = 0.0
        if j < N_TRAP:
            drift = 0.02 if trap_up else -0.02
        elif trap_up:
            # 上涨版本里，非 trap 组微跌，保证基准不是纯正收益（否则门槛太好过）
            drift = -0.002
        for i in range(n_d):
            px[i, j] = 10.0 * (1.0 + drift) ** i
            if trap_up and j >= N_TRAP and i >= holdout_i + 2:
                # holdout 段（起算后第 3 行）非 trap 组集体急跌 10%：让基准有**段内**
                # 真实回撤。若从段首第一行就砸，段的第一个净值点已经在坑底，
                # 段内看不到下跌，mdd 仍是 0（踩过）。
                px[i, j] *= 0.90

    pb = np.full((n_d, n_c), 3.0)
    pb[:, :N_TRAP] = 0.8                       # bp 高 = 便宜
    pe = np.full((n_d, n_c), 20.0)
    vol = np.full((n_d, n_c), float(volume))
    turn = np.full((n_d, n_c), 1.0)
    listed = np.full((n_d, n_c), 999.0)
    is_st = np.zeros((n_d, n_c))
    roe = np.full((n_d, n_c), 0.10)
    debt = np.full((n_d, n_c), 40.0)
    for j in roe_zero:
        roe[:, j] = -0.05
    for j in debt_high:
        debt[:, j] = 95.0
    # 两组行业，各 20 只（行业内分位要求组内 ≥5 只）
    gid = np.array([[0 if j < n_codes // 2 else 1 for j in range(n_codes)]] * n_d,
                   dtype="int16")

    panel = Panel(dates=dates, codes=codes, open_hfq=px.copy(), close_hfq=px.copy(),
                  open_raw=px.copy(), close_raw=px.copy(),
                  blocked_buy=np.zeros((n_d, n_c), bool),
                  blocked_sell=np.zeros((n_d, n_c), bool),
                  suspended=np.zeros((n_d, n_c), bool),
                  hfq_gap=np.zeros((n_d, n_c), bool))
    return FactorData(panel=panel, pe_ttm=pe, pb_mrq=pb, volume=vol, turn=turn,
                      is_st=is_st, listed_days=listed, roe_ttm=roe,
                      yoy_ni=np.full((n_d, n_c), 5.0), debt_ratio=debt,
                      float_shares=np.full((n_d, n_c), 1e8),
                      industry_gid=gid,
                      source={"start": dates[0], "end": dates[-1]})


@pytest.fixture()
def cfg() -> dict:
    """合成数据只覆盖 2024-01 起，所以定档/验证年份要按数据改，不能沿用真实配置。"""
    c = C.default_config()
    c["system"]["validation"]["method"] = "yearly_freeze"
    c["system"]["validation"]["years"] = ["2022", "2023", "2024", "2025"]
    c["system"]["validation"]["band_fit_end"] = "2025-06-30"
    c["system"]["validation"]["years"] = ["2024", "2025"]
    return c


def _sys(cfg: dict) -> dict:
    return C.system_params(cfg)


def _mats(fd):
    return factor_mats(fd), trade_amount(fd)


# ================================================================ 0. 分位带取数
def test_signals_are_top_band_equal_weight(cfg):
    """信号 = 综合分前 band 的**全部**股票，每只 1/k，权重和 = 1；不在带内的不进。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    i = 23                                          # 第 12 个月末（动量需 >20 个交易日）
    cs = cache[i]
    band = 0.10
    k, sel = select_top_k(cs, band)
    assert k == int(np.ceil(band * cs["n_candidates"]))

    sig = signals_from_cache(fd, cache, band)
    w = sig[cs["date"]]
    assert len(w) == k, "分位带是「整条带全买」，不是挑几只"
    assert all(abs(v - 1.0 / k) < 1e-12 for v in w.values()), "必须严格等权"
    assert abs(sum(w.values()) - 1.0) < 1e-9
    expect = {fd.codes[int(j)] for j in cs["idx"][cs["order"][:k]]}
    assert set(w) == expect, "进组合的必须是分数最高的 k 只"
    # 带外的票一只都不许进来
    out = {fd.codes[int(j)] for j in cs["idx"][cs["order"][k:]]}
    assert not (set(w) & out)


def test_select_top_k_ceil_and_edges():
    """k = ceil(band × n)，且夹在 [1, n]；空池返回 0（不许凑数）。"""
    cs = {"n_candidates": 100, "order": np.arange(100)}
    assert select_top_k(cs, 0.03)[0] == 3
    assert select_top_k(cs, 0.025)[0] == 3, "2.5 只向上取整到 3"
    assert select_top_k(cs, 1.0)[0] == 100
    assert select_top_k(cs, 0.001)[0] == 1, "再小的带也至少 1 只"
    assert select_top_k({"n_candidates": 0, "order": np.array([])}, 0.1)[0] == 0


def test_band_neighbors_at_edges():
    """±1 档：中间档两侧都有，边界档缺的那侧如实给 None（不硬凑、不环绕）。"""
    mid = band_neighbors(0.08, BANDS)
    assert (mid["prev"], mid["selected"], mid["next"]) == (0.05, 0.08, 0.10)
    lo = band_neighbors(0.03, BANDS)
    assert lo["prev"] is None and lo["next"] == 0.05
    hi = band_neighbors(0.30, BANDS)
    assert hi["next"] is None and hi["prev"] == 0.20
    odd = band_neighbors(0.06, BANDS)               # 手改出来的档不在候选列表里
    assert odd["prev"] == 0.05 and odd["next"] == 0.08


# ================================================================ 1. 选档只看定档段
def test_band_scan_ignores_future_data(cfg):
    """**本单核心纪律**：把定档段之后的行情改成完全相反，选档结果必须一字不变。

    做法：同一份合成数据跑两遍，第二遍把定档段之后的价格整体扭曲（涨的变跌、
    跌幅放大 5 倍），再比两遍的档位表与选中档。一模一样 = 定档确实没看未来。
    """
    fd1 = make_fd(trap_up=True)
    sysd = _sys(cfg)
    mats1, amt1 = _mats(fd1)
    i0, i1, im = _seg_window(fd1, fd1.dates[0], "2025-06-30")
    rows1 = scan_bands(fd1, sysd, mats1, amt1, i0, i1, im, BANDS)
    pick1 = select_band(rows1, 0.10)

    fd2 = make_fd(trap_up=True)
    cut = next(i for i, d in enumerate(fd2.dates) if d > "2025-06-30")
    n_c = fd2.panel.close_raw.shape[1]
    for j in range(n_c):
        factor = 1.6 if j % 3 else 0.2              # 大幅扭曲，足以翻转所有排名
        for arr in (fd2.panel.close_hfq, fd2.panel.open_hfq,
                    fd2.panel.close_raw, fd2.panel.open_raw):
            arr[cut:, j] *= factor
    mats2, amt2 = _mats(fd2)
    rows2 = scan_bands(fd2, sysd, mats2, amt2, i0, i1, im, BANDS)
    pick2 = select_band(rows2, 0.10)

    assert rows1 == rows2, "定档段之后的行情变化影响了选档结果 → 存在前视"
    assert pick1["selected"] == pick2["selected"]
    assert len(rows1) == len(BANDS) >= 4, "候选档至少 4 档（业主验收）"


def test_select_band_rule_and_forced():
    """选档规则：合格 = 回撤不深于池 10pp **且超额 > 0**，在其中取超额最高。

    只有回撤合格、超额全负时也必须标 forced —— "最不差的一档"不是有证据的选档。
    """
    rows = [
        {"band": 0.03, "excess": 0.05, "strategy_mdd": -0.40, "pool_mdd": -0.20},
        {"band": 0.05, "excess": 0.03, "strategy_mdd": -0.22, "pool_mdd": -0.20},
        {"band": 0.08, "excess": 0.01, "strategy_mdd": -0.21, "pool_mdd": -0.20},
    ]
    pick = select_band(rows, 0.10)
    assert pick["selected"] == 0.05, "0.03 超额最高但回撤深 20pp，必须被排除"
    assert pick["forced"] is False
    assert pick["qualified_bands"] == [0.05, 0.08]
    assert pick["tightest_positive_band"] == 0.05, "最窄的正超额档 = 真正偏离基准的那档"
    assert "超额 > 0" in pick["rule"]

    all_neg = [dict(r, excess=-abs(r["excess"])) for r in rows]   # 全部跑输但回撤都合格
    neg = select_band(all_neg, 0.10)
    assert neg["forced"] is True
    assert neg["selected"] == 0.08, "负超额里取最接近 0 的那档（0.03 已被回撤淘汰）"
    assert "没有一档跑赢" in neg["rule"]
    assert neg["tightest_positive_band"] is None

    bad = [dict(r, strategy_mdd=-0.60) for r in rows]
    forced = select_band(bad, 0.10)
    assert forced["forced"] is True and forced["selected"] == 0.03
    assert "退化为取超额最高的一档" in forced["rule"]


# ================================================================ 2. 基准
def test_pool_benchmark_daily_mode_is_member_mean(cfg):
    """日频口径 = 池内成员日收益的简单平均。

    合成数据里价格 = 10 × (1+drift)^i，所以日收益精确等于 drift：
    trap 组 +2%/日、其余 −0.2%/日、池内共 40 只 → 池日收益 = (8×0.02 − 32×0.002)/40 = 0.24%。
    """
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    pool = pool_equal_weight(fd, cache, 0, len(fd.dates), rebalance="daily")
    r = pool.pct_change().dropna()
    # 段首两个月末之前没有截面 → 空仓计 0；从有第一个月末起，日收益恒为 0.24%
    assert np.allclose(r.iloc[2:].to_numpy(), 0.0024, atol=1e-12)
    assert abs(float(pool.iloc[0]) - 1.0) < 1e-12


def test_pool_benchmark_monthly_mode_holds_without_rebalancing(cfg):
    """月频口径（门槛用）= 期间**不再平衡**：净值 = 成员相对月末价的比值均值。

    这正是与策略同频的对照：策略一个月里也不调仓，凭什么基准天天调回等权。
    """
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    pool = pool_equal_weight(fd, cache, 0, len(fd.dates), rebalance="monthly")
    ends = sorted(cache)
    m = ends[0]
    # 月末 m 出信号，m+1 起持有：nav[m+1] = 成员 px[m+1]/px[m] 的均值
    expect = (8 * 1.02 + 32 * 0.998) / 40
    assert float(pool.iloc[m + 1]) == pytest.approx(expect, rel=1e-12)
    # 同一持仓期内不再平衡 → 越往后越偏离"每日等权"那版
    daily = pool_equal_weight(fd, cache, 0, len(fd.dates), rebalance="daily")
    assert float(pool.iloc[m + 3]) != pytest.approx(float(daily.iloc[m + 3]), rel=1e-6)
    # 下一期从新月末重新等权：nav[m2+1] = nav[m2] × 成员均值
    m2 = ends[1]
    expect2 = float(pool.iloc[m2]) * (8 * 1.02 + 32 * 0.998) / 40
    assert float(pool.iloc[m2 + 1]) == pytest.approx(expect2, rel=1e-12)


def test_pool_benchmark_differs_from_full_market(cfg):
    """池基准必须真的按 mask 缩小样本：把一只高收益票否决掉，池收益就该降。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    base = pool_equal_weight(fd, cache, 0, len(fd.dates)).iloc[-1]

    sysd2 = json.loads(json.dumps(sysd))
    sysd2["veto"]["roe_ttm_min"] = 0.2                 # 合成数据 roe 全 0.10 → 全体否决
    cache2 = score_cross_sections(fd, sysd2, mats, amt)
    empty = pool_equal_weight(fd, cache2, 0, len(fd.dates))
    assert float(empty.iloc[-1]) == pytest.approx(1.0), "池子空 → 空仓不动，不是拿全市场顶替"
    assert base > 1.0


# ================================================================ 3. 换仓清单
def test_rebalance_list_partitions_names(cfg):
    """换仓清单必须是名单的**划分**：买/卖/持 互斥，数量与上期/本期只数对得上。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    ends = sorted(cache)
    i_cur, i_prev = ends[-1], ends[-2]
    rb = _rebalance_block(fd, i_cur, i_prev, cache, 0.10, {})

    cur = set(rb["hold"]) | set(rb["buy"])
    prev = set(rb["hold"]) | set(rb["sell"])
    assert set(rb["hold"]) & set(rb["buy"]) == set()
    assert set(rb["hold"]) & set(rb["sell"]) == set()
    assert set(rb["buy"]) & set(rb["sell"]) == set()
    assert len(cur) == rb["current"] and len(prev) == rb["prev"]
    assert rb["counts"] == {"hold": len(rb["hold"]), "buy": len(rb["buy"]),
                            "sell": len(rb["sell"])}
    assert rb["counts"]["hold"] + rb["counts"]["buy"] == rb["current"]
    assert rb["counts"]["hold"] + rb["counts"]["sell"] == rb["prev"]
    assert rb["turnover"] == pytest.approx(
        (len(rb["buy"]) + len(rb["sell"])) / (2 * rb["prev"]))
    assert rb["prev_cross_section"] == cache[i_prev]["date"]


def test_rebalance_detects_full_swap(cfg):
    """上期名单被整体换掉时，清单必须给出「全卖全买」，不许静默空表。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    ends = sorted(cache)
    i_cur, i_prev = ends[-1], ends[-2]

    fake = dict(cache[i_prev])
    fake["order"] = fake["order"][::-1]              # 把上期名单翻成最差的 k 只
    cache2 = dict(cache)
    cache2[i_prev] = fake
    rb = _rebalance_block(fd, i_cur, i_prev, cache2, 0.10, {})
    assert rb["counts"]["hold"] == 0
    assert rb["counts"]["buy"] == rb["current"] == rb["counts"]["sell"]
    assert rb["turnover"] == pytest.approx(1.0)


def test_rebalance_absent_prev_cross_section(cfg):
    """数据起点附近没有上一期：如实说明，不编造换仓清单。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    first = sorted(cache)[0]
    rb = _rebalance_block(fd, first, None, cache, 0.10, {})
    assert rb["available"] is False and "上一期截面缺失" in rb["reason"]


# ================================================================ 4. 预算分配
def test_budget_plan_ok_branch():
    """预算够：每只金额 = 预算/只数，按整手向下取整，全部够一手。"""
    rows = [{"code": "a", "close": 10.0}, {"code": "b", "close": 30.0}]
    p = budget_plan(rows, 10000.0, 100)
    assert p["mode"] == "ok" and p["n_affordable"] == 2
    assert p["per_stock"] == 5000.0
    assert [x["lots"] for x in p["items"]] == [5, 1]
    assert p["est_invested"] == pytest.approx(5000 + 3000)
    assert p["min_viable_budget"] == pytest.approx(1000 + 3000)


def test_budget_plan_short_branch_gives_min_viable():
    """预算不足：如实说买不到一手，并给最小可行预算与"最多覆盖几只"。"""
    rows = [{"code": "a", "close": 10.0}, {"code": "b", "close": 30.0}]
    p = budget_plan(rows, 4000.0, 100)
    assert p["mode"] == "budget_short"
    assert p["n_affordable"] == 1 and p["max_lot_cost"] == 3000.0
    assert p["min_viable_budget"] == pytest.approx(4000.0)
    assert p["suggested_budget"] == pytest.approx(4000.0)
    assert "预算不足" in p["text"] and "攒到" in p["text"]
    assert "不是本卡的等权组合" in p["text"], "跳着买必须说明性质变了"


# ================================================================ 5. 分年验证
def test_yearly_windows_cover_dev_without_overlap(cfg):
    """分年窗口：每年只含本年交易日、止于 dev 末、年与年不重叠，预热在段前。"""
    fd = make_fd(trap_up=True)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    dev_last = max(d for d in fd.dates if d < "2025-11-07")
    sig = signals_from_cache(fd, cache, 0.10)
    res = yearly_validation(fd, cache, sig, sysd, ["2024", "2025"], dev_last, 0.10)

    rows = res["rows"]
    assert [r["period"] for r in rows] == ["2024", "2025"]
    assert rows[0]["start"] >= "2024-01-01" and rows[0]["end"] <= "2024-12-31"
    assert rows[1]["start"] >= "2025-01-01" and rows[1]["end"] <= dev_last
    assert rows[0]["end"] < rows[1]["start"], "年与年不许重叠"
    # 预热段必须严格早于当年首个交易日（否则就是把当年行情当预热）
    assert rows[1]["start"] != rows[1]["warmup_start"]
    assert rows[1]["warmup_start"] < rows[1]["start"]
    assert res["summary"]["n_years"] == 2
    assert res["summary"]["all_dd_ok"] in (True, False)


def test_yearly_skips_years_outside_data(cfg):
    """数据里没有的年份如实跳过并记账，不许静默当成"通过"。"""
    fd = make_fd(trap_up=True, months=6)             # 只有 2024 上半年
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    sig = signals_from_cache(fd, cache, 0.10)
    res = yearly_validation(fd, cache, sig, sysd, ["2023", "2024"], "2024-06-28", 0.10)
    assert [r["period"] for r in res["rows"]] == ["2024"]
    assert [s["period"] for s in res["skipped"]] == ["2023"]
    assert "没有该年交易日" in res["skipped"][0]["reason"]


# ================================================================ 6. 门槛
def test_gates_dev_fail_means_no_direction_check():
    """dev 不过 → 直接 fallback，**不看 holdout 方向**（005R 裁定②）。"""
    old_cfg = C.default_config()
    old_cfg["system"]["validation"]["method"] = "yearly_freeze"
    dev = {"excess": -0.10, "strategy": {"ann_return": -0.05}, "bench": {"ann_return": 0.05}}
    holdout = {"excess": 0.20, "strategy": {"ann_return": 0.30, "mdd": -0.10},
               "bench": {"ann_return": 0.10, "mdd": -0.20}}
    yearly = {"rows": [{"period": "2022", "dd_ok": True, "dd_worse_pp": 0.01,
                        "strategy_mdd": -0.20, "pool_mdd": -0.21, "excess": -0.02}]}
    g = evaluate_gates(dev, yearly, holdout, _sys(old_cfg))
    assert g["dev"]["pass"] is False
    assert g["holdout"]["direction_consistent"] is None, "dev 不过就不比方向"
    assert g["holdout"]["checked"] is False
    assert g["both_pass"] is False


def test_gates_period_drawdown_rule():
    """各期回撤：比池深 10pp 以内算过，超过就整条门槛不过（逐年逐期判）。"""
    old_cfg = C.default_config()
    old_cfg["system"]["validation"]["method"] = "yearly_freeze"
    dev = {"excess": 0.05, "strategy": {"ann_return": 0.12}, "bench": {"ann_return": 0.07}}
    sysd = _sys(old_cfg)
    holdout = {"excess": 0.03, "strategy": {"ann_return": 0.10, "mdd": -0.18},
               "bench": {"ann_return": 0.07, "mdd": -0.20}}
    ok = evaluate_gates(dev, {"rows": [
        {"period": "2022", "dd_ok": True, "dd_worse_pp": 0.05,
         "strategy_mdd": -0.25, "pool_mdd": -0.20, "excess": 0.01}]}, holdout, sysd)
    assert ok["periods"]["pass"] and ok["both_pass"] and ok["holdout"]["direction_consistent"]

    bad = evaluate_gates(dev, {"rows": [
        {"period": "2023", "dd_ok": False, "dd_worse_pp": 0.15,
         "strategy_mdd": -0.35, "pool_mdd": -0.20, "excess": 0.01}]}, holdout, sysd)
    assert bad["periods"]["pass"] is False
    assert bad["periods"]["failed_periods"] == ["2023"]
    assert bad["both_pass"] is False

    flipped = evaluate_gates(dev, {"rows": [
        {"period": "2022", "dd_ok": True, "dd_worse_pp": 0.0,
         "strategy_mdd": -0.20, "pool_mdd": -0.20, "excess": 0.01}]},
        {"excess": -0.02, "strategy": {"ann_return": 0.02, "mdd": -0.10},
         "bench": {"ann_return": 0.04, "mdd": -0.20}}, sysd)
    assert flipped["holdout"]["direction_consistent"] is False
    assert flipped["both_pass"] is False


def test_004c_rejects_005r_legacy_evidence():
    """005R旧产物没有rolling证据，必须被新闸拦下并给出可追溯原因。"""
    import json
    from pathlib import Path
    old_path = Path(__file__).resolve().parents[1] / "out" / "advice_card.json"
    if not old_path.exists():
        pytest.skip("仓库中没有005R旧建议卡产物")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    sysd = _sys(C.default_config())
    result = evaluate_gates(old.get("segments", {}).get("dev", {}),
                            old.get("validation", {}),
                            old.get("segments", {}).get("holdout", {}), sysd)
    assert result["both_pass"] is False
    # 005S-R2：只测「被拦 + reason 是数组且含预期闸」，不锁死具体形状
    assert isinstance(result["fallback_reason"], list)
    assert "stability" in result["fallback_reason"]
    assert result["drawdown"]["reason"] == "missing_rolling_evidence"


def test_legacy_yearly_freeze_reason_is_array():
    """005S-R2：旧 yearly_freeze 分支也返回数组，全仓库不留单值路径。"""
    sysd = _sys(C.default_config())
    sysd["validation"] = dict(sysd["validation"], method="yearly_freeze")
    dev_bad = {"excess": -0.02, "strategy": {"ann_return": 0.0},
               "bench": {"ann_return": 0.02}}
    holdout = {"excess": 0.03, "strategy": {"ann_return": 0.05, "mdd": -0.10},
               "bench": {"ann_return": 0.02, "mdd": -0.10}}

    bad = evaluate_gates(dev_bad, {"rows": []}, holdout, sysd)
    assert bad["fallback_reason"] == ["dev", "periods"]
    assert bad["both_pass"] is False

    dev_ok = {"excess": 0.02, "strategy": {"ann_return": 0.05},
              "bench": {"ann_return": 0.03}}
    rows = [{"period": "2022", "dd_ok": True, "dd_worse_pp": 0.0,
             "strategy_mdd": -0.10, "pool_mdd": -0.12, "excess": 0.01}]
    passed = evaluate_gates(dev_ok, {"rows": rows}, holdout, sysd)
    assert passed["both_pass"] is True
    assert passed["fallback_reason"] is None
    # 只挂方向一条：数组仍是数组，长度 1
    flipped = evaluate_gates(dev_ok, {"rows": rows}, dict(holdout, excess=-0.02), sysd)
    assert flipped["fallback_reason"] == ["direction"]


def test_004c_gates_require_forced_drawdown_and_stability():
    """004C三道门均用已知输入事前验证，任一失败就fallback。"""
    sysd = _sys(C.default_config())
    good = {"windows": [{"excess": 0.03}] * 5,
            "mean_excess": 0.02, "positive_windows": 4, "window_count": 5,
            "portfolio_mdd": -0.15}
    passed = evaluate_gates({}, {}, None, sysd, rolling=good)
    assert passed["both_pass"] is True
    assert passed["fallback_reason"] is None

    for field, value, reason in (("portfolio_mdd", -0.150001, "drawdown"),
                                 ("mean_excess", 0.019999, "stability"),
                                 ("positive_windows", 3, "stability")):
        bad = dict(good, **{field: value})
        result = evaluate_gates({}, {}, None, sysd, rolling=bad)
        assert result["both_pass"] is False
        assert result["fallback_reason"] == [reason]

    forced = evaluate_gates({}, {}, None, sysd, forced=True, rolling=good)
    assert forced["both_pass"] is False
    assert forced["fallback_reason"] == ["forced"]


def test_004c_fallback_reason_lists_all_triggered_gates():
    """005S-R1：三闸全挂必须全列，顺序 fixed(forced>drawdown>stability)，不许被 forced 遮蔽。"""
    sysd = _sys(C.default_config())
    bad = {"windows": [{"excess": -0.01}] * 5, "mean_excess": 0.001,
           "positive_windows": 2, "window_count": 5, "portfolio_mdd": -0.30}
    triple = evaluate_gates({}, {}, None, sysd, forced=True, rolling=bad)
    assert triple["fallback_reason"] == ["forced", "drawdown", "stability"]
    assert triple["both_pass"] is False

    # 无 forced 时只列真实失败的两闸，顺序不变
    two = evaluate_gates({}, {}, None, sysd, forced=False, rolling=bad)
    assert two["fallback_reason"] == ["drawdown", "stability"]

    # 只挂稳定性一闸：数组仍是数组（长度 1），下游不必判类型
    only_stability = evaluate_gates(
        {}, {}, None, sysd, forced=False,
        rolling=dict(bad, portfolio_mdd=-0.10, mean_excess=0.03,
                     positive_windows=3))
    assert only_stability["fallback_reason"] == ["stability"]


def test_004c_fallback_notes_cover_every_failed_gate():
    """005S-R1：fallback.reasons 必须逐闸出句子，不许只剩一句"门槛未通过"。"""
    from src.advisor import _004c_gate_failure_notes
    sysd = _sys(C.default_config())
    bad = {"windows": [{"excess": -0.01}] * 5, "mean_excess": 0.001,
           "positive_windows": 2, "window_count": 5, "portfolio_mdd": -0.30}
    gates = evaluate_gates({}, {}, None, sysd, forced=True, rolling=bad)
    rolling = {"band_selection": {"selected": 0.30},
               "portfolio_mdd_peak": "2021-12-16", "portfolio_mdd_trough": "2024-02-05"}
    notes = _004c_gate_failure_notes(gates, rolling)

    assert len(notes) == len(gates["fallback_reason"]) == 3
    assert [n.split("：")[0] for n in notes] == ["forced", "drawdown", "stability"]
    assert "-30.00%" in notes[1] and "-15%" in notes[1]
    assert "1/5" not in notes[2] and "2/5" in notes[2]
    # forced 那句话必须自带免责，避免被读成"只是选档问题"
    assert "不代表另两闸通过" in notes[0]


def test_004c_daily_nav_export_columns_order_and_mdd(tmp_path):
    """005S-R1：导出的 parquet 三列齐、日期升序、nav 列独立重算 MDD 与引擎一致。"""
    dates = pd.bdate_range("2021-01-04", periods=40)
    nav = np.concatenate([np.linspace(1.0, 1.2, 20), np.linspace(1.2, 0.9, 20)])
    bench = np.linspace(1.0, 1.05, 40)
    frame = pd.DataFrame({"date": dates, "nav": nav, "benchmark_nav": bench,
                          "junk": 0.0})
    path = tmp_path / "004c_daily_nav.parquet"
    # 故意倒序传入：导出必须自己排序，不许依赖上游顺序
    meta = export_daily_nav(frame.iloc[::-1], path)

    assert path.exists()
    back = pd.read_parquet(path)
    assert list(back.columns) == ["date", "nav", "benchmark_nav"]
    assert meta["columns"] == ["date", "nav", "benchmark_nav"]
    assert meta["rows"] == 40 == len(back)
    assert back["date"].is_monotonic_increasing
    assert meta["start"] == "2021-01-04"
    assert back["nav"].iloc[0] == pytest.approx(1.0)
    # 1.2 → 0.9 是 -25%，独立重算必须复现，否则审计对不上
    assert meta["mdd_recomputed"] == pytest.approx(-0.25, abs=1e-6)
    assert meta["benchmark_last"] == pytest.approx(float(bench[-1]))


def test_004c_composite_is_simple_equal_weight_of_selected_factors(cfg):
    """入选因子分位严格按1/N等权，低波因子与反转均参与。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    from src.factors import FACTOR_DEFS
    for name in ("lowvol_20", "reversal_20"):
        mats[name] = FACTOR_DEFS[name](fd)
    i = 23
    uni = build_universe(fd, i, _sys(cfg), mats, amt)
    cs = composite_scores(fd, i, uni, _sys(cfg), mats,
                          factor_names=["lowvol_20", "reversal_20"])
    expected = (cs["factor_pct"]["lowvol_20"] + cs["factor_pct"]["reversal_20"]) / 2
    assert np.allclose(cs["score"], expected)
    assert cs["factor_weights"] == {"lowvol_20": 0.5, "reversal_20": 0.5}


# ================================================================ 7. 端到端
def test_analyze_advises_when_gates_pass(cfg):
    """门槛全过 → verdict=advise，card 非空且是整条带（不是 5 只）。"""
    fd = make_fd(trap_up=True, n_codes=400)      # 放大到 400 只，让分散形态看得出规模
    rep = analyze(fd, cfg)
    assert rep["verdict"] == "advise", rep["fallback"]
    assert rep["gates"]["both_pass"] is True
    card = rep["card"]
    assert card is not None and card["is_recommendation"] is True
    k = card["portfolio"]["k"]
    assert k == int(np.ceil(card["portfolio"]["band"] * card["n_candidates"]))
    assert len(card["stocks"]) == k > 5, "分散形态：只数应远大于 5"
    assert all(abs(s["weight"]["point"] - 1.0 / k) < 1e-6 for s in card["stocks"])
    assert sum(s["weight"]["point"] for s in card["stocks"]) == pytest.approx(1.0)
    assert rep["schema_version"].startswith("005R")
    assert rep["band_selection"]["selected"] in BANDS
    assert len(rep["band_selection"]["table"]) == len(BANDS)


def test_analyze_outputs_fallback_without_card(cfg):
    """dev 跑不赢可投池 → verdict=fallback、card 为 null、preview 标注不构成建议。"""
    fd = make_fd(trap_up=False)
    cfg["system"]["validation"]["method"] = "yearly_freeze"
    rep = analyze(fd, cfg)
    assert rep["verdict"] == "fallback"
    assert rep["card"] is None
    fb = rep["fallback"]
    assert fb["message"] == "本次不建议" and fb["reasons"]
    assert "fallback_reason" in fb
    assert fb["numbers"]["dev"]["strategy_ann"] is not None
    assert fb["numbers"]["holdout"]["bench_ann"] is not None
    assert fb["numbers"]["yearly"], "fallback 也要带分年数字"
    assert rep["preview_card"]["is_recommendation"] is False
    assert "不构成建议" in rep["preview_card"]["label"]
    assert any("不建议" in x for x in rep["validity_note"])


# ================================================================ 8. 卡字段
def test_card_fields_complete_and_add_up(cfg):
    """每只票的字段必须齐、得分构成可对账、名次与分位自洽。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    ends = sorted(cache)
    i = ends[-1]
    card = build_card(fd, i, ends[-2], cache, sysd, C.owner_profile(cfg), mats, amt, {},
                      {"band": 0.10, "k": 4, "dev_mdd": -0.20, "dev_pool_mdd": -0.25,
                       "owner_mdd": -0.15}, band=0.10)
    k = card["portfolio"]["k"]
    assert len(card["stocks"]) == k
    need = {"code", "name", "industry", "bp_pct", "reversal_pct", "momentum_20", "score",
            "score_breakdown", "rank_in_universe", "score_pct", "position_in_band",
            "weight", "reason", "risk", "diagnostics"}
    for s in card["stocks"]:
        assert need <= set(s), f"{s['code']} 缺字段 {need - set(s)}"
        bd = s["score_breakdown"]
        assert bd["sum_check"] == pytest.approx(s["score"], abs=1e-4)
        assert bd["bp"]["contribution"] + bd["reversal"]["contribution"] == \
            pytest.approx(s["score"], abs=1e-4)
        assert bd["bp"]["weight"] == 0.6 and bd["reversal"]["weight"] == 0.4
        assert s["weight"]["point"] == pytest.approx(1.0 / k, abs=1e-6)
        assert s["reason"] and s["risk"]
        assert 0.0 < s["score_pct"] <= 1.0
    ranks = [s["rank_in_universe"] for s in card["stocks"]]
    assert ranks == list(range(1, k + 1)), "名单必须按综合分降序"
    scores = [s["score"] for s in card["stocks"]]
    assert scores == sorted(scores, reverse=True)
    assert card["risk_note"] and "等权" in card["risk_note"][0]
    assert card["rebalance"]["available"] is True
    assert "text" in card["budget"]


def test_side_reference_band_80_90_is_labelled_post_hoc(cfg):
    """80~90 分位带旁证：选的是"从第 1 名往下数 10%~20%"，且必须带事后观察标注。

    这条防止两个错：① 把 80~90 当成"最顶端 10%"（方向反了）；② 报了这个数字
    却不标"事后观察值"，等于拿它当选档依据。
    """
    fd = make_fd(trap_up=True)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    i = sorted(cache)[-1]
    cs = cache[i]
    n = cs["n_candidates"]
    sig = side_band_signals(fd, cache, 0.80, 0.90)
    picks = sig[cs["date"]]
    k = len(picks)
    order = cs["order"]
    rank_of = {fd.codes[int(cs["idx"][p])]: r + 1 for r, p in enumerate(order)}
    # 逐只核对分位落点：0.80 ≤ 1 − (名次−1)/n < 0.90
    for c in picks:
        pct = 1.0 - (rank_of[c] - 1) / n
        assert 0.80 <= pct < 0.90, f"{c} 分位 {pct:.4f} 不在 80~90 带内"
    expect_n = sum(1 for r in range(1, n + 1) if 0.80 <= 1 - (r - 1) / n < 0.90)
    assert k == expect_n
    top1 = fd.codes[int(cs["idx"][order[0]])]
    assert top1 not in picks, "80~90 带不含第 1 名（那是 top 0.1%）"
    assert all(abs(v - 1.0 / k) < 1e-12 for v in picks.values())

    rep = analyze(fd, cfg)
    sr = rep["side_reference"]
    assert sr["label"] == "事后观察值，非选择依据"
    assert "不进门槛" in sr["why"] and "事后拟合" in sr["why"]
    assert rep["band_selection"]["selected"] in [r["band"] for r in rep["band_selection"]["table"]]


# ================================================================ 9. 005 遗留口径回归
def test_score_is_weighted_sum_of_two_percentiles(cfg):
    """评分层未改：综合得分 = w_bp×bp分位 + w_rev×反转分位，反转分位 = 1 − 动量分位。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    i = 23
    uni = build_universe(fd, i, sysd, mats, amt)
    cs = composite_scores(fd, i, uni, sysd, mats)
    assert cs["n_candidates"] > 0
    expect = 0.6 * cs["bp_pct"] + 0.4 * cs["rev_pct"]
    assert cs["score"] == pytest.approx(expect)
    assert cs["weight_bp"] == 0.6 and cs["weight_reversal"] == 0.4
    top_mom = int(np.nanargmax(cs["mom_raw"]))
    assert cs["rev_pct"][top_mom] == pytest.approx(cs["rev_pct"].min())


def test_veto_excludes_loss_makers(cfg):
    """roe_ttm < 0 的票不进候选池，且剔除数如实记账。"""
    fd = make_fd(trap_up=True, months=13, roe_zero=(0, 1, 2))
    mats, amt = _mats(fd)
    uni = build_universe(fd, 23, _sys(cfg), mats, amt)
    assert uni["steps"]["vetoed_roe_negative"] == 3
    for j in (0, 1, 2):
        assert not uni["mask"][j], "亏损股必须被否决"
    assert uni["steps"]["total_vetoed"] >= 3


def test_veto_excludes_high_debt_in_industry(cfg):
    """行业内资产负债率分位 > 90% 的票被否决；组内不足 5 只时不算分位、不否决。"""
    fd = make_fd(trap_up=True, months=13, debt_high=(5, 6, 7))
    mats, amt = _mats(fd)
    uni = build_universe(fd, 23, _sys(cfg), mats, amt)
    assert uni["steps"]["vetoed_high_debt"] == 3
    for j in (5, 6, 7):
        assert not uni["mask"][j]
    assert uni["debt_stats"]["groups"] >= 1


def test_veto_skips_small_industry_groups():
    """组内 <5 只不计算分位 —— 小组分位是噪声，宁可不动手。"""
    fd = make_fd(trap_up=True, months=8)
    mats, amt = _mats(fd)
    fd.industry_gid = np.tile(np.arange(fd.panel.close_raw.shape[1], dtype="int16"),
                              (len(fd.dates), 1))
    uni = build_universe(fd, 15, _sys(C.default_config()), mats, amt)
    assert uni["debt_stats"]["groups"] == 0
    assert uni["steps"]["vetoed_high_debt"] == 0, "算不出分位就不许据此否决"


def test_filters_liquidity_st_listed_and_limit_up(cfg):
    """成交额/ST/上市天数/涨停 四个过滤器各自生效，且逐步记账能对上。"""
    fd = make_fd(trap_up=True, months=13, volume=1e7)   # 价格 10 → 成交额 1e8
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    i = 23
    base = build_universe(fd, i, sysd, mats, amt)
    n_base = base["steps"]["final"]

    s2 = json.loads(json.dumps(sysd))
    s2["filters"]["min_amount"] = 2e8
    low = build_universe(fd, i, s2, mats, amt)
    assert low["steps"]["final"] == 0
    assert low["steps"]["dropped_liquidity"] == base["steps"]["after_listed_days"]

    fd.is_st[i, 3] = 1.0
    st = build_universe(fd, i, sysd, mats, amt)
    assert not st["mask"][3]
    assert st["steps"]["final"] == n_base - 1

    s3 = json.loads(json.dumps(sysd))
    s3["filters"]["min_listed_days"] = 1000
    assert build_universe(fd, i, s3, mats, amt)["steps"]["final"] == 0

    fd.is_st[i, 3] = 0.0
    fd.panel.blocked_buy[i, 4] = True
    lu = build_universe(fd, i, sysd, mats, amt)
    assert not lu["mask"][4]
    assert lu["steps"]["dropped_limit_up"] >= 1

    fd.panel.blocked_buy[i, 4] = False
    fd.panel.blocked_buy[i + 1, 5] = True
    # 005S-R2：T+1 一字板不参与 T 日筛池 —— 拿明天的盘面筛今天的名单是未来函数。
    # 买不买得进由回测执行层在成交日判断，截面层不许预先剔除。
    assert build_universe(fd, i, sysd, mats, amt)["mask"][5]


def test_universe_ignores_next_day_limit_up(cfg):
    """005S-R2 回归（审计C B8-B）：截面池对 T+1 涨停必须完全免疫。"""
    fd = make_fd(trap_up=True, months=13)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    i = 23
    base = build_universe(fd, i, sysd, mats, amt)["mask"].copy()
    assert base.any(), "构造失效：池子本来就空，去掉 T+1 也不会变"

    fd.panel.blocked_buy[i + 1, :] = True        # 次日全市场一字板
    after = build_universe(fd, i, sysd, mats, amt)["mask"]
    assert np.array_equal(base, after), "T+1 涨停影响了 T 日池子，未来函数回来了"

    # 反向对照：T 日自己的涨停必须剔，否则上面那条断言等于空转
    fd.panel.blocked_buy[i, 4] = True
    assert not build_universe(fd, i, sysd, mats, amt)["mask"][4]


# ================================================================ 10. 段切分与信号边界
def test_segment_signals_do_not_cross_holdout(cfg):
    """dev / holdout 的信号互不越界：段内只回测段内信号。"""
    fd = make_fd(trap_up=True)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    built = build_signals(fd, sysd, mats, amt)
    sig_all = built["signals"]
    ho_start = sysd["holdout"]["start"]
    i_dev1 = max(i for i, d in enumerate(fd.dates) if d < ho_start) + 1
    i_ho0 = min(i for i, d in enumerate(fd.dates) if d >= ho_start)
    k = int(np.median([d["k"] for d in built["detail"].values() if d["k"]]))

    dev = run_segment(fd, built["cache"], 0, i_dev1, sig_all, sysd)
    ho = run_segment(fd, built["cache"], i_ho0, len(fd.dates), sig_all, sysd)
    assert dev["end"] < ho_start
    assert ho["start"] >= ho_start
    n_dev = sum(1 for d in sig_all if d < ho_start)
    n_ho = sum(1 for d in sig_all if d >= ho_start)
    assert dev["n_rebalance"] == n_dev
    assert ho["n_rebalance"] == n_ho
    assert n_dev + n_ho == len(sig_all), "每个信号日期只能归入一段"


def test_segment_metrics_start_after_warmup(cfg):
    """预热段只建仓不进指标：指标起点必须等于 i_meas 对应日期。"""
    fd = make_fd(trap_up=True)
    mats, amt = _mats(fd)
    sysd = _sys(cfg)
    cache = score_cross_sections(fd, sysd, mats, amt)
    sig = signals_from_cache(fd, cache, 0.10)
    i0, i1, im = _seg_window(fd, "2025-01-01", "2025-06-30")
    seg = run_segment(fd, cache, i0, i1, sig, sysd, i_meas=im)
    assert seg["start"] == fd.dates[im] >= "2025-01-01"
    assert seg["warmup_start"] < seg["start"], "预热段必须早于指标起点"
    assert seg["n_rebalance_warmup"] >= 0
    assert seg["bench"]["name"] == "pool_equal_weight"


# ================================================================ 11. config
def test_config_defaults_and_merge(tmp_path):
    """缺字段用默认补齐；写的字段覆盖默认；说明键不进参数。"""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"owner": {"budget": 50000, "max_drawdown": -0.10},
                             "system": {"portfolio": {"band": 0.08}}}),
                 encoding="utf-8")
    cfg = C.load_config(p)
    assert cfg["system"]["validation"]["method"] == "rolling_24_12"
    assert cfg["system"]["validation"]["years"] == ["2021", "2022", "2023", "2024", "2025"]
    assert cfg["system"]["gate"]["max_portfolio_mdd"] == 0.15
    assert cfg["owner"]["budget"] == 50000
    assert cfg["owner"]["experience"] == C.DEFAULT_OWNER["experience"], "未写的字段用默认"
    assert C.owner_profile(cfg)["max_drawdown"] == -0.10
    assert C.system_params(cfg)["portfolio"]["band"] == 0.08
    assert "_comment" not in C.owner_profile(cfg)
    assert "_comment" not in C.system_params(cfg)["portfolio"]


@pytest.mark.parametrize("bad,msg", [
    ({"owner": {"budget": -1}}, "budget"),
    ({"owner": {"max_drawdown": 0.15}}, "max_drawdown"),
    ({"system": {"weights": {"bp": 0.8, "reversal": 0.4}}}, "和为 1"),
    ({"system": {"veto": {"debt_ratio_industry_pct_max": 1.5}}}, "debt_ratio"),
    ({"system": {"portfolio": {"band": 1.5}}}, "band"),
    ({"system": {"portfolio": {"band_candidates": [0.1, 0.2]}}}, "至少 4 档"),
    ({"system": {"portfolio": {"band_candidates": [0.2, 0.1, 0.3, 0.4]}}}, "升序"),
    ({"system": {"validation": {"years": ["22"]}}}, "四位年份"),
    ({"system": {"validation": {"method": "rolling"}}}, "rolling_24_12"),
    ({"system": {"gate": {"period_max_dd_worse": -0.1}}}, "period_max_dd_worse"),
    ({"system": {"validation": {"method": "rolling_24_12"},
                   "gate": {"period_max_dd_worse": 0.1}}}, "period_max_dd_worse"),
])
def test_config_rejects_bad_values(tmp_path, bad, msg):
    """越界配置一律抛错，不做容错兜底。"""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(C.ConfigError) as e:
        C.load_config(p)
    assert msg in str(e.value)


def test_config_rejects_legacy_n_hold(tmp_path):
    """旧口径的 n_hold / max_weight 必须报错，不能静默忽略（看着生效最危险）。"""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"system": {"n_hold": 5, "max_weight": 0.3}}), encoding="utf-8")
    with pytest.raises(C.ConfigError) as e:
        C.load_config(p)
    assert "005R" in str(e.value) and "n_hold" in str(e.value)


def test_config_init_refuses_to_overwrite(tmp_path):
    """--init 不许覆盖业主手改过的配置（丢了不可逆）。"""
    p = tmp_path / "config.json"
    C.save_template(p)
    raw = p.read_text(encoding="utf-8")
    p.write_text(raw.replace('"budget": 100000.0', '"budget": 88888.0'), encoding="utf-8")
    with pytest.raises(C.ConfigError):
        C.save_template(p)
    assert "88888" in p.read_text(encoding="utf-8"), "拒绝覆盖后原内容必须完好"
    C.save_template(p, force=True)
    assert "88888" not in p.read_text(encoding="utf-8")
    assert '"portfolio"' in p.read_text(encoding="utf-8"), "模板必须含新口径"
