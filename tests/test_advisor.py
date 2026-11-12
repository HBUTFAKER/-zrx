"""005 建议卡引擎测试。

合成面板 + 真引擎（backtest.run_backtest），不打桩核心计算：
评分加权、否决项、过滤器、候选不足、fallback 分支、config 解析、得分构成、
holdout 边界共 12 条。

合成数据的设计意图（不写清楚下次看不懂）：
  · `trap` 组：pb 低（bp 高）且价格持续下跌 —— 既是"最便宜"又是"跌最多"，
    正好落在综合分的顶端。让它们**继续跌**，策略就会跑不赢等权 → 用来测 fallback；
    让它们**上涨**，策略跑赢 → 用来测 advise 分支。
  · 其余票走平，行业分两组各 20 只（满足行业内分位的最小样本要求）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import src.config as C
from src.advisor import (allocate, analyze, build_card, build_signals,
                         build_universe, composite_scores, evaluate_gates,
                         factor_mats, run_segment, trade_amount)
from src.backtest import Panel
from src.factors import FactorData

# ---------------------------------------------------------------- 合成面板
N_TRAP = 8
N_TOTAL = 40


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
            # 上涨版本里，非 trap 组微跌，保证基准不是正收益（否则门槛太好过）
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
    return C.default_config()


def _sys(cfg: dict) -> dict:
    return C.system_params(cfg)


# ---------------------------------------------------------------- 1. 仓位分配
def test_allocate_respects_cap_and_sums_to_one():
    """按得分加权、单票上限截断、截断余额摊给其余票，最后必须正好 100%。"""
    w = allocate(np.array([0.99, 0.50, 0.30, 0.20, 0.10]), cap=0.25)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w <= 0.25 + 1e-9).all(), "单票上限必须严格生效"
    assert w[0] == pytest.approx(0.25), "最高分被截到上限"
    # 平分且不超上限时，就是等权
    eq = allocate(np.array([1.0, 1.0, 1.0, 1.0]), cap=0.5)
    assert eq == pytest.approx([0.25] * 4)


# ---------------------------------------------------------------- 2. 评分加权
def test_score_is_weighted_sum_of_two_percentiles(cfg):
    """综合得分 = w_bp×bp分位 + w_rev×反转分位，且反转分位 = 1 − momentum 分位。"""
    fd = make_fd(trap_up=True, months=13)
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    sysd = _sys(cfg)
    i = 23                                        # 第 12 个月末（动量需 >20 个交易日）
    uni = build_universe(fd, i, sysd, mats, amt)
    cs = composite_scores(fd, i, uni, sysd, mats)
    assert cs["n_candidates"] > 0
    expect = 0.6 * cs["bp_pct"] + 0.4 * cs["rev_pct"]
    assert cs["score"] == pytest.approx(expect)
    assert cs["weight_bp"] == 0.6 and cs["weight_reversal"] == 0.4
    # 反转分位与动量原始值必须反向：动量最高（涨最多）的票，反转分位应最低
    top_mom = int(np.nanargmax(cs["mom_raw"]))
    assert cs["rev_pct"][top_mom] == pytest.approx(cs["rev_pct"].min())


# ---------------------------------------------------------------- 3. 否决项
def test_veto_excludes_loss_makers(cfg):
    """roe_ttm < 0 的票不进候选池，且剔除数如实记账。"""
    fd = make_fd(trap_up=True, months=13, roe_zero=(0, 1, 2))
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    i = 23
    uni = build_universe(fd, i, _sys(cfg), mats, amt)
    assert uni["steps"]["vetoed_roe_negative"] == 3
    for j in (0, 1, 2):
        assert not uni["mask"][j], "亏损股必须被否决"
    assert uni["steps"]["total_vetoed"] >= 3


def test_veto_excludes_high_debt_in_industry(cfg):
    """行业内资产负债率分位 > 90% 的票被否决；组内不足 5 只时不算分位、不否决。"""
    # 第一组 20 只里放 3 只超高杠杆 → 它们必然落在组内 >90% 分位
    fd = make_fd(trap_up=True, months=13, debt_high=(5, 6, 7))
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    i = 23
    uni = build_universe(fd, i, _sys(cfg), mats, amt)
    assert uni["steps"]["vetoed_high_debt"] == 3
    for j in (5, 6, 7):
        assert not uni["mask"][j]
    assert uni["debt_stats"]["groups"] >= 1


def test_veto_skips_small_industry_groups():
    """组内 <5 只不计算分位 —— 小组分位是噪声，宁可不动手。"""
    fd = make_fd(trap_up=True, months=8)
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    i = 15
    # 把行业改成"每只票一个组"：所有组都只有 1 只，全部跳过
    fd.industry_gid = np.tile(np.arange(fd.panel.close_raw.shape[1], dtype="int16"),
                              (len(fd.dates), 1))
    uni = build_universe(fd, i, _sys(C.default_config()), mats, amt)
    assert uni["debt_stats"]["groups"] == 0
    assert uni["steps"]["vetoed_high_debt"] == 0, "算不出分位就不许据此否决"


# ---------------------------------------------------------------- 4. 过滤器
def test_filters_liquidity_st_listed_and_limit_up(cfg):
    """成交额/ST/上市天数/涨停 四个过滤器各自生效，且逐步记账能对上。"""
    fd = make_fd(trap_up=True, months=13, volume=1e7)   # 价格 10 → 成交额 1e8
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    sysd = _sys(cfg)
    i = 23
    base = build_universe(fd, i, sysd, mats, amt)
    n_base = base["steps"]["final"]

    # ① 流动性：把阈值抬到 2 亿，全部出局
    s2 = json.loads(json.dumps(sysd))
    s2["filters"]["min_amount"] = 2e8
    low = build_universe(fd, i, s2, mats, amt)
    assert low["steps"]["final"] == 0
    assert low["steps"]["dropped_liquidity"] == base["steps"]["after_listed_days"]

    # ② ST：mark 一只 ST，它必须消失
    fd.is_st[i, 3] = 1.0
    st = build_universe(fd, i, sysd, mats, amt)
    assert not st["mask"][3]
    assert st["steps"]["final"] == n_base - 1

    # ③ 上市天数：把阈值提到 1000 天，全部出局
    s3 = json.loads(json.dumps(sysd))
    s3["filters"]["min_listed_days"] = 1000
    assert build_universe(fd, i, s3, mats, amt)["steps"]["final"] == 0

    # ④ 涨停：开盘即涨停的不可选（T 日与 T+1 日都要挡）
    fd.is_st[i, 3] = 0.0
    fd.panel.blocked_buy[i, 4] = True
    lu = build_universe(fd, i, sysd, mats, amt)
    assert not lu["mask"][4]
    assert lu["steps"]["dropped_limit_up"] >= 1

    fd.panel.blocked_buy[i, 4] = False
    fd.panel.blocked_buy[i + 1, 5] = True       # 次日一字板：卡上出得来也买不进
    lu2 = build_universe(fd, i, sysd, mats, amt)
    assert not lu2["mask"][5], "次日开盘涨停同样不可选"


# ---------------------------------------------------------------- 5. 候选不足
def test_card_reports_short_when_candidates_below_n_hold(cfg):
    """候选不足 5 只：如实少给并写明原因，不凑数、不报错。"""
    fd = make_fd(trap_up=True, months=13, n_codes=12)
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    sysd = _sys(cfg)
    sysd["n_hold"] = 30                          # 池子只有 12 只，必然凑不满
    sysd["max_weight"] = 1.0
    i = 23
    card = build_card(fd, i, sysd, C.owner_profile(cfg), mats, amt, {})
    assert 0 < len(card["stocks"]) < 30
    assert card["short_by"] == 30 - len(card["stocks"])
    assert "未凑数" in (card["reason"] or "")


# ---------------------------------------------------------------- 6. 门槛与 fallback
def test_gates_fail_when_dev_underperforms():
    """dev 超额为负 → 门槛不过；两条规则各自可判。"""
    dev = {"excess": -0.1782, "strategy": {"ann_return": -0.0979},
           "bench": {"ann_return": 0.0803}}
    holdout = {"excess": -0.05, "strategy": {"ann_return": 0.01, "mdd": -0.30},
               "bench": {"ann_return": 0.06, "mdd": -0.25}}
    g = evaluate_gates(dev, holdout, _sys(C.default_config()))
    assert g["dev"]["pass"] is False
    # 回撤 -30% 比基准 -25% 更深 → 必须判"不达标"，不能因为都是负数就放过
    assert g["holdout"]["dd_not_worse"] is False
    assert g["both_pass"] is False


def test_gate_holdout_drawdown_and_direction():
    """holdout 两项子条件：回撤更深 → 不过；超额符号与 dev 相反 → 方向不一致。"""
    sysd = _sys(C.default_config())
    dev = {"excess": 0.05, "strategy": {"ann_return": 0.12}, "bench": {"ann_return": 0.07}}
    ok = evaluate_gates(dev, {"excess": 0.02,
                              "strategy": {"ann_return": 0.05, "mdd": -0.20},
                              "bench": {"ann_return": 0.03, "mdd": -0.30}}, sysd)
    assert ok["dev"]["pass"] and ok["holdout"]["pass"] and ok["both_pass"]

    deeper = evaluate_gates(dev, {"excess": 0.02,
                                  "strategy": {"ann_return": 0.05, "mdd": -0.45},
                                  "bench": {"ann_return": 0.03, "mdd": -0.30}}, sysd)
    assert deeper["holdout"]["dd_not_worse"] is False
    assert deeper["both_pass"] is False

    flipped = evaluate_gates(dev, {"excess": -0.01,
                                   "strategy": {"ann_return": 0.02, "mdd": -0.10},
                                   "bench": {"ann_return": 0.03, "mdd": -0.30}}, sysd)
    assert flipped["holdout"]["direction_consistent"] is False
    assert flipped["both_pass"] is False


def test_analyze_outputs_fallback_without_card(cfg):
    """dev 跑不赢基准 → verdict=fallback、card 为 null、preview 标注不构成建议。"""
    fd = make_fd(trap_up=False)                  # 廉价暴跌组继续跌 → 策略必输
    rep = analyze(fd, cfg)
    assert rep["verdict"] == "fallback"
    assert rep["card"] is None, "门槛没过就不许有 card"
    fb = rep["fallback"]
    assert fb["message"] == "本次不建议"
    assert fb["reasons"], "fallback 必须说明原因"
    assert fb["numbers"]["dev"]["strategy_ann"] is not None
    assert fb["numbers"]["holdout"]["bench_ann"] is not None
    # preview 仍然给（核对字段用），但必须显式标记不是建议
    assert rep["preview_card"]["is_recommendation"] is False
    assert "不构成建议" in rep["preview_card"]["label"]
    assert any("不建议" in x for x in rep["validity_note"])


def test_analyze_advises_when_gates_pass(cfg):
    """两条门槛都过 → verdict=advise、card 非空（同一套代码的另一分支）。"""
    fd = make_fd(trap_up=True)                   # 廉价组一路涨 → 策略应跑赢
    rep = analyze(fd, cfg)
    assert rep["verdict"] == "advise", rep["fallback"]
    assert rep["gates"]["both_pass"] is True
    assert rep["card"] is not None and len(rep["card"]["stocks"]) == cfg["system"]["n_hold"]
    assert rep["card"]["is_recommendation"] is True


# ---------------------------------------------------------------- 7. 得分构成
def test_card_score_breakdown_is_complete_and_adds_up(cfg):
    """每只票的得分构成必须可解释：两项贡献之和 == 综合得分，字段一个不少。"""
    fd = make_fd(trap_up=True, months=13)
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    sysd = _sys(cfg)
    i = 23
    card = build_card(fd, i, sysd, C.owner_profile(cfg), mats, amt, {})
    assert len(card["stocks"]) == 5
    need = {"code", "name", "bp_pct", "reversal_pct", "score", "score_breakdown",
            "reason", "risk", "weight", "rank_in_universe", "diagnostics"}
    for s in card["stocks"]:
        assert need <= set(s), f"{s['code']} 缺字段 {need - set(s)}"
        bd = s["score_breakdown"]
        assert bd["sum_check"] == pytest.approx(s["score"], abs=1e-4)
        assert bd["bp"]["contribution"] + bd["reversal"]["contribution"] == \
            pytest.approx(s["score"], abs=1e-4)
        assert bd["bp"]["weight"] == 0.6 and bd["reversal"]["weight"] == 0.4
        # 仓位区间必须夹住点位权重、且不超过上限
        w = s["weight"]
        assert w["low"] <= w["point"] <= w["high"]
        assert w["high"] <= sysd["max_weight"] + 1e-9
        assert s["reason"] and s["risk"]
    total = sum(s["weight"]["point"] for s in card["stocks"])
    assert total == pytest.approx(1.0, abs=1e-6), "仓位归一到 100%"


# ---------------------------------------------------------------- 8. config
def test_config_defaults_and_merge(tmp_path):
    """缺字段用默认补齐；写的字段覆盖默认；说明键不进参数。"""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"owner": {"budget": 50000,
                                       "max_drawdown": -0.10},
                             "system": {"n_hold": 5, "max_weight": 0.25}}),
                 encoding="utf-8")
    cfg = C.load_config(p)
    assert cfg["owner"]["budget"] == 50000
    assert cfg["owner"]["experience"] == C.DEFAULT_OWNER["experience"], "未写的字段用默认"
    assert C.owner_profile(cfg)["max_drawdown"] == -0.10
    assert "_comment" not in C.owner_profile(cfg)
    assert "_comment" not in (C.system_params(cfg)["weights"])


@pytest.mark.parametrize("bad,msg", [
    ({"owner": {"budget": -1}}, "budget"),
    ({"owner": {"max_drawdown": 0.15}}, "max_drawdown"),
    ({"system": {"weights": {"bp": 0.8, "reversal": 0.4}}}, "和为 1"),
    ({"system": {"n_hold": 5, "max_weight": 0.10}}, "凑不满"),
    ({"system": {"veto": {"debt_ratio_industry_pct_max": 1.5}}}, "debt_ratio"),
])
def test_config_rejects_bad_values(tmp_path, bad, msg):
    """越界配置一律抛错，不做容错兜底。"""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(C.ConfigError) as e:
        C.load_config(p)
    assert msg in str(e.value)


def test_config_init_refuses_to_overwrite(tmp_path):
    """--init 不许覆盖业主手改过的配置（丢了不可逆）。"""
    p = tmp_path / "config.json"
    C.save_template(p)
    first = p.read_text(encoding="utf-8")
    p.write_text(first.replace('"budget": 100000.0', '"budget": 88888.0'), encoding="utf-8")
    with pytest.raises(C.ConfigError):
        C.save_template(p)
    assert "88888" in p.read_text(encoding="utf-8"), "拒绝覆盖后原内容必须完好"
    C.save_template(p, force=True)
    assert "88888" not in p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 9. holdout 边界
def test_holdout_boundary_signals_do_not_leak(cfg):
    """dev / holdout 的信号互不越界：段内只回测段内信号，跨界的期不许进 IC。"""
    fd = make_fd(trap_up=True)
    mats = factor_mats(fd)
    amt = trade_amount(fd)
    sysd = _sys(cfg)
    built = build_signals(fd, sysd, mats, amt)["signals"]
    ho_start = sysd["holdout"]["start"]
    dev_end = sysd["holdout"]["start"]

    i_dev1 = max(i for i, d in enumerate(fd.dates) if d < ho_start) + 1
    i_ho0 = min(i for i, d in enumerate(fd.dates) if d >= ho_start)

    dev = run_segment(fd, 0, i_dev1, built, sysd)
    ho = run_segment(fd, i_ho0, len(fd.dates), built, sysd)
    assert dev["end"] < ho_start, "dev 段末日必须早于 holdout 起点"
    assert ho["start"] >= ho_start
    assert dev["end"] == fd.dates[i_dev1 - 1]
    assert ho["start"] == fd.dates[i_ho0]

    n_dev_sig = sum(1 for d in built if d < ho_start)
    n_ho_sig = sum(1 for d in built if d >= ho_start)
    assert dev["n_rebalance"] == n_dev_sig
    assert ho["n_rebalance"] == n_ho_sig
    assert n_dev_sig + n_ho_sig == len(built), "每个信号日期只能归入一段"
    assert dev_end > dev["start"]
