"""建议卡生成引擎（S3-Q3 · 多因子投票版）。

组合规格（业主拍板·方案 B）：
    综合得分 = bp 分位 × w_bp + 反转分位 × w_rev
    反转     = momentum_20 的反向分位（dev 段 IC −0.057，t −4.20，方向为反转）
    否决     = roe_ttm < 阈值 / debt_ratio 行业内分位 > 阈值
    过滤     = cross_section_mask（ST/停牌/hfq 缺口/上市 60 日）+ 成交额下限 + 涨停不可选

三条口径纪律（写在这里是为了下次改代码前先看到）：
  ① **只做决策辅助，不给买卖指令。** 卡上给的是"这只票在截面里的相对位置"和
     "按得分该分多少仓位区间"，何时买卖由业主决定。代码里没有任何下单动作。
  ② **dev 定型、holdout 一次性出。** 权重/阈值/过滤器全部来自 dev 段（见
     `weight_sensitivity` 与 config 里的依据注释），定死后 holdout 只跑一次，
     不因 holdout 结果回头改参数。run() 的顺序就是 dev → 门槛 → holdout。
  ③ **不达标就不出卡。** 门槛（`evaluate_gates`）任一不过 → verdict=fallback，
     `card` 为 null，只输出"本次不建议 + 原因 + 两段数字"。宁可不给，不给凑数的票。

一个必须说清的实测事实（S3-Q3 施工时发现，直接影响本引擎的价值判断）：
    综合分的**排序能力为正**（dev 段月频 IC +0.0707、t +4.00、IC>0 占 65.6%），
    但**最顶端 5 只反而是坑**：分数 20 等分桶里，桶19 月均 +97.4bp、桶20 掉到 +44.3bp，
    直接取 top5 年化 −10.84%（纯前瞻收益、无成本口径）。这不是实现 bug（已用
    "纯前瞻收益 vs 引擎实跑"双向对拍确认，两者都 ≈ −10%），而是"最便宜的暴跌股"
    这一极端角落的非单调性。本引擎照规格取 top5，因此 dev 段过不了门槛 —— 是引擎
    如实报告结论，不是引擎坏了。
"""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as C
from src import metrics as M
from src.backtest import Config as BtConfig
from src.backtest import Costs, run_backtest
from src.factors import (MIN_GROUP_SIZE, compute_factors, cross_section_mask,
                         ic_periods, load_factor_data, month_end_indices,
                         rank_ic, slice_panel)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = ROOT / "data" / "cache.db"
OUT_DIR = ROOT / "out"
CARD_NAME = "advice_card.json"

#: 数据起点（与 factors.py 的 CLI 默认一致；holdout 端点从 config 读，不在这里写死）
DEFAULT_START = "2018-01-02"

#: 权重敏感性要报的档位（业主拍板：不许只报一档）
SENSITIVITY_WEIGHTS = (0.5, 0.6, 0.7)

#: 仓位区间半宽（±2pp，和"18%~22%"这种输出一致）
WEIGHT_BAND = 0.02


# ---------------------------------------------------------------- 小工具
def _pct_rank(x: np.ndarray) -> np.ndarray:
    """截面分位（0~1，平均秩处理并列）。NaN 原样返回 NaN。"""
    return pd.Series(x).rank(pct=True).to_numpy(dtype="float64")


def _f(x, spec: str = ".4f") -> str:
    """数字格式化：None/NaN/inf 一律打 n/a，绝不输出 `nan%`。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else format(v, spec)


def _r(x, n: int = 6):
    """四舍五入到 n 位；非有限值返回 None（JSON 里 null 比 NaN 诚实）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, n) if np.isfinite(v) else None


def _git_meta() -> dict:
    """代码溯源（与 factors._git_meta 同口径：code_dirty 只看 src/ tests/）。"""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
        out = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--porcelain"],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        paths = [ln[3:].strip().strip('"') for ln in lines]
        code_files = [p for p in paths if p.startswith(("src/", "tests/"))]
        return {"code_commit": commit, "code_dirty": bool(code_files),
                "code_dirty_files": code_files[:20],
                "worktree_dirty": bool(paths),
                "worktree_other_files": [p for p in paths
                                         if not p.startswith(("src/", "tests/"))][:20]}
    except Exception:  # noqa: BLE001 —— 取不到就写 unknown，不编造
        return {"code_commit": "unknown", "code_dirty": None,
                "code_dirty_files": [], "worktree_dirty": None,
                "worktree_other_files": []}


def load_names(path: Path | None = None) -> dict[str, str]:
    """股票简称（data/cache.db 的 stocks.code_name，只读）。

    取不到就返回空 dict，卡上只显示代码 —— **不编造名称**。名称只用于展示，
    不参与任何计算，所以缺了不算事故。
    """
    p = path or CACHE_DB
    if not p.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT code, code_name FROM stocks").fetchall()
        finally:
            con.close()
        return {str(c): str(n) for c, n in rows if c and n}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- 因子与截面
def factor_mats(fd) -> dict[str, np.ndarray]:
    """本引擎只用三个因子 + 一个否决项字段，按需算，不跑全量 9 个。"""
    return compute_factors(fd, ["bp", "momentum_20", "roe_ttm", "debt_ratio"])


SW_INDUSTRY = ROOT / "data" / "shares" / "sw_industry.parquet"
SW_INDUSTRY_NAMES = ROOT / "data" / "shares" / "sw_industry_names.parquet"


def load_industry_labels(level: int = 1) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """读申万行业归属与名称，返回 (归属表, {行业代码: 名称})。

    为什么不复用 `fd.industry_names`：load_factor_data 只把组号（int16）写进
    FactorData，没写 gid→名称的映射（组号是按"面板里出现的行业码排序"动态编的）。
    为了卡上显示"银行/钢铁"而不是"gid=13"，这里按**同一套时点规则**（in_date）
    单独查一次 —— 只查 5 只票，代价可以忽略，也不改只读模块。
    读不到就返回 (None, {})，卡上行业留空，不编造。
    """
    if not SW_INDUSTRY.exists():
        return None, {}
    try:
        sw = pd.read_parquet(SW_INDUSTRY, columns=["panel_code", "industry_code", "in_date"])
        sw = sw.dropna(subset=["in_date"])
        cut = 2 * level
        sw = sw.assign(grp=sw["industry_code"].astype(str).str[:cut])
        names: dict[str, str] = {}
        if SW_INDUSTRY_NAMES.exists():
            nm = pd.read_parquet(SW_INDUSTRY_NAMES,
                                 columns=["industry_code", "industry_name", "level"])
            nm = nm[nm["level"] == level]
            names = {str(c): str(n) for c, n in
                     zip(nm["industry_code"], nm["industry_name"])}
        return sw.sort_values("in_date"), names
    except Exception:  # noqa: BLE001 —— 展示字段，取不到就当没有
        return None, {}


def industry_of(sw: pd.DataFrame | None, names: dict[str, str], code: str,
                date: str) -> str | None:
    """某只票在 date 当天的一级行业名（in_date ≤ date 的最后一条）。"""
    if sw is None:
        return None
    try:
        sub = sw[(sw["panel_code"] == code) & (sw["in_date"] <= date)]
        if not len(sub):
            return None
        grp = str(sub.iloc[-1]["grp"])
        return names.get(grp)
    except Exception:  # noqa: BLE001
        return None


def trade_amount(fd) -> np.ndarray:
    """成交额（元）= 收盘价 × 成交量。

    口径说明：baostock 的 `amount` 是当日成交金额（VWAP 口径），本引擎用
    close_raw × volume 近似。差在"收盘价 vs 当日均价"，只用于**流动性下限**
    这种量级判断（阈值 5000 万，分位差几百万不影响取舍），不参与打分。
    不直接读 amount 列的原因：factors.load_factor_data 不带该列，为它单开一条
    数据通道等于在只读模块外另搞一套口径。
    """
    return fd.panel.close_raw * fd.volume


def industry_percentile(fd, i: int, mask: np.ndarray,
                        min_group: int = MIN_GROUP_SIZE) -> tuple[np.ndarray, dict]:
    """第 i 日 `debt_ratio` 的**行业内**分位（升序，1.0 = 行业内最高杠杆）。

    组内不足 `min_group` 只 → 该组不计算（分位是噪声，宁可不否决）；
    无行业归属（gid < 0）→ 同样不计算。返回 (pct, stats)，pct 无值处为 NaN。
    """
    d = fd.debt_ratio[i]
    n = len(d)
    pct = np.full(n, np.nan, dtype="float64")
    if fd.industry_gid is None:
        return pct, {"available": False, "reason": "no_industry",
                     "groups": 0, "skipped_groups": 0, "skipped_codes": int(mask.sum())}
    g = fd.industry_gid[i]
    okd = mask & np.isfinite(d)
    groups = 0
    skipped_g = 0
    skipped_n = 0
    for gv in np.unique(g[okd]):
        if gv < 0:
            continue
        sel = okd & (g == gv)
        if sel.sum() < min_group:
            skipped_g += 1
            skipped_n += int(sel.sum())
            continue
        pct[sel] = _pct_rank(d[sel])
        groups += 1
    return pct, {"available": groups > 0, "groups": groups,
                 "skipped_groups": skipped_g, "skipped_codes": skipped_n,
                 "min_group": min_group}


def build_universe(fd, i: int, system: dict, mats: dict[str, np.ndarray],
                   amount: np.ndarray) -> dict:
    """第 i 日的可选池：逐步过滤并**逐步记账**，每个过滤器剔掉多少只都要能说清。

    顺序是有讲究的：先走 factors 的公共口径（ST/停牌/hfq 缺口/上市天数），
    再加流动性、涨停、质量否决。顺序不影响最终集合（都是逻辑与），但影响
    "剔除计数"的可读性 —— 先干掉不可交易的，再说"因为贵/亏被否决几只"。
    """
    flt = system.get("filters") or {}
    veto = system.get("veto") or {}
    steps: dict = {}

    base = cross_section_mask(fd, i)
    steps["cross_section"] = int(base.sum())

    min_ld = int(flt.get("min_listed_days", 60))
    if min_ld > 0:
        base &= fd.listed_days[i] >= min_ld
        steps["after_listed_days"] = int(base.sum())

    thr = float(flt.get("min_amount", 0.0) or 0.0)
    liq = base & np.isfinite(amount[i]) & (amount[i] >= thr)
    steps["after_liquidity"] = int(liq.sum())
    steps["dropped_liquidity"] = int(base.sum() - liq.sum())

    if flt.get("exclude_limit_up", True):
        # blocked_buy = 开盘即涨停（买不进）。同时排除**次日**开盘涨停的票：
        # 卡是 T 日收盘出、T+1 开盘买，次日一字板根本成交不了。
        lu = liq & fd.panel.blocked_buy[i]
        if i + 1 < len(fd.dates):
            lu |= liq & fd.panel.blocked_buy[i + 1]
        liq = liq & ~lu
        steps["dropped_limit_up"] = int(lu.sum())
    else:
        steps["dropped_limit_up"] = 0
    steps["after_limit_up"] = int(liq.sum())

    # ---- 质量否决 ①：亏损企业
    roe = fd.roe_ttm[i]
    roe_min = float(veto.get("roe_ttm_min", 0.0))
    r_ok = np.isfinite(roe)
    veto_roe = liq & r_ok & (roe < roe_min)
    liq &= ~veto_roe
    steps["vetoed_roe_negative"] = int(veto_roe.sum())

    # ---- 质量否决 ②：行业内高杠杆
    pct, pstats = industry_percentile(fd, i, liq)
    dp_max = float(veto.get("debt_ratio_industry_pct_max", 0.90))
    veto_debt = liq & np.isfinite(pct) & (pct > dp_max)
    liq &= ~veto_debt
    steps["vetoed_high_debt"] = int(veto_debt.sum())

    steps["final"] = int(liq.sum())
    steps["total_vetoed"] = steps["vetoed_roe_negative"] + steps["vetoed_high_debt"]
    return {"mask": liq, "steps": steps, "debt_pct": pct, "debt_stats": pstats,
            "thresholds": {"min_amount": thr, "roe_ttm_min": roe_min,
                           "debt_ratio_industry_pct_max": dp_max,
                           "min_listed_days": min_ld}}


def composite_scores(fd, i: int, uni: dict, system: dict,
                     mats: dict[str, np.ndarray],
                     w_bp: float | None = None) -> dict:
    """综合得分 = w·bp分位 + (1−w)·反转分位（反转分位 = 1 − momentum_20 分位）。

    只在"可选池 ∩ bp/mom 都有值"的样本上算分位 —— 分位的分母必须是**同一批票**，
    否则 bp 分位和反转分位不在一个尺度上，加权没有意义。
    """
    w = system.get("weights") or {}
    wb = float(w.get("bp", 0.6)) if w_bp is None else float(w_bp)
    wr = 1.0 - wb
    mask = uni["mask"]
    b = mats["bp"][i]
    m = mats["momentum_20"][i]
    ok = mask & np.isfinite(b) & np.isfinite(m)
    idx = np.nonzero(ok)[0]
    out = {"weight_bp": wb, "weight_reversal": wr, "idx": idx,
           "n_candidates": int(len(idx)), "bp_pct": None, "rev_pct": None,
           "score": None, "mom_raw": None}
    if len(idx) == 0:
        return out
    sb = _pct_rank(b[idx])
    sr = 1.0 - _pct_rank(m[idx])
    out["bp_pct"] = sb
    out["rev_pct"] = sr
    out["score"] = wb * sb + wr * sr
    out["mom_raw"] = m[idx]
    return out


def allocate(scores: np.ndarray, cap: float) -> np.ndarray:
    """按得分加权归一到 100%，单票上限 cap（超上限的截断后把余额摊给其余票）。

    迭代到稳定：截断 → 重分 → 再截断。cap × n < 1 时无解，config.validate 已挡住。
    """
    s = np.asarray(scores, dtype="float64")
    if len(s) == 0:
        return s
    if s.sum() <= 0:
        return np.full(len(s), 1.0 / len(s))
    w = s / s.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        w[over] = cap
        rest = ~over
        budget = 1.0 - cap * float(over.sum())
        if not rest.any():
            break
        w[rest] = w[rest] / w[rest].sum() * budget
    return w


def build_signals(fd, system: dict, mats: dict[str, np.ndarray],
                  amount: np.ndarray) -> dict:
    """全部月末截面的目标权重信号：{日期: {代码: 权重}}。

    引擎在 T+1 开盘执行，信号只用 ≤T 的数据（cross_section_mask 的 all 输入
    都是 T 日收盘可知的量）。
    """
    sig: dict[str, dict[str, float]] = {}
    detail: dict[str, dict] = {}
    n_hold = int(system.get("n_hold", 5))
    cap = float(system.get("max_weight", 0.30))
    for i in month_end_indices(fd.dates):
        uni = build_universe(fd, i, system, mats, amount)
        cs = composite_scores(fd, i, uni, system, mats)
        if cs["n_candidates"] == 0:
            detail[fd.dates[i]] = {"n_candidates": 0, "picked": []}
            continue
        order = np.argsort(-cs["score"])[:n_hold]
        sel = cs["idx"][order]
        wt = allocate(cs["score"][order], cap)
        sig[fd.dates[i]] = {fd.codes[j]: float(x) for j, x in zip(sel, wt)}
        detail[fd.dates[i]] = {"n_candidates": cs["n_candidates"],
                               "picked": [fd.codes[j] for j in sel],
                               "n_hold_target": n_hold,
                               "short": bool(cs["n_candidates"] < n_hold)}
    return {"signals": sig, "detail": detail}


# ---------------------------------------------------------------- 回测与门槛
def _nav_metrics(nav: pd.DataFrame) -> dict:
    s = M.to_series(nav)
    dd = M.max_drawdown(s)
    return {"ann_return": _r(M.ann_return(s)), "mdd": _r(dd["mdd"]),
            "mdd_peak": dd["peak"], "mdd_trough": dd["trough"],
            "vol": _r(M.ann_vol(s)), "sharpe": _r(M.sharpe(s)),
            "total_return": _r(float(s.iloc[-1] / s.iloc[0] - 1.0)) if len(s) > 1 else None,
            "nav_first": str(s.index[0]), "nav_last": str(s.index[-1]),
            "n_days": int(len(s))}


def run_segment(fd, i0: int, i1: int, sig_all: dict, system: dict) -> dict:
    """在一段区间上跑组合回测，并给出两个基准的同期表现。

    基准 ①：`metrics.equal_weight_market`（S2 口径，全市场 hfq 等权、无成本）——
             这是**门槛用的**基准，与全场报告同一口径。
    基准 ②：可投池等权（同一 mask、同一月末调仓、无成本）—— 只作对照。它比 ①
             低，是因为 ① 含了小盘股（size 因子 Q1 年化 +14%），而本策略有
             5000 万成交额下限，把小微盘整体切掉了。两个都报，谁高谁低自己看。
    """
    dates = fd.dates[i0:i1]
    panel = slice_panel(fd.panel, i0, i1)
    lo, hi = dates[0], dates[-1]
    sig = {d: w for d, w in sig_all.items() if lo <= d <= hi}
    cfg = BtConfig(
        start=lo, end=hi,
        init_cash=float((system.get("backtest") or {}).get("init_cash", 1e6)),
        max_positions=int(system.get("n_hold", 5)),
        max_weight=float(system.get("max_weight", 0.30)),
        lot_size=int((system.get("backtest") or {}).get("lot_size", 100)),
        costs=_costs(system))
    res = run_backtest(panel, sig, cfg) if sig else None
    bench = M.equal_weight_market(panel)
    bench_dd = M.max_drawdown(bench)
    out = {
        "start": lo, "end": hi, "n_days": len(dates), "n_rebalance": len(sig),
        "bench": {"name": "equal_weight_market",
                  "ann_return": _r(M.ann_return(bench)), "mdd": _r(bench_dd["mdd"]),
                  "definition": "全市场 hfq 等权日收益均值（S2 基准口径，含退市股、"
                                "停牌计 0 收益、无成本）"},
    }
    if res is None:
        out["strategy"] = {"available": False, "reason": "no_signal"}
        out["excess"] = None
        return out
    m = _nav_metrics(res.nav)
    m["n_trades"] = int(len(res.trades))
    m["n_dropped"] = int(len(res.dropped)) if res.dropped is not None else 0
    if res.dropped is not None and len(res.dropped) and "reason" in res.dropped:
        m["dropped_by_reason"] = {str(k): int(v) for k, v in
                                  res.dropped["reason"].value_counts().items()}
    m["available"] = True
    out["strategy"] = m
    out["excess"] = _r(m["ann_return"] - out["bench"]["ann_return"])
    # 期末持仓（用于"卡上的票是不是真买了"的核对）
    if len(res.positions):
        last = res.positions[res.positions["date"] == res.positions["date"].max()]
        out["holdings_at_end"] = {str(r.code): _r(r.value) for r in
                                  last.itertuples()}
    return out


def _costs(system: dict) -> Costs:
    c = system.get("costs") or {}
    base = Costs()
    return Costs(commission_rate=float(c.get("commission_rate", base.commission_rate)),
                 min_commission=float(c.get("min_commission", base.min_commission)),
                 stamp_tax=float(c.get("stamp_tax", base.stamp_tax)),
                 transfer_fee=float(c.get("transfer_fee", base.transfer_fee)),
                 slippage=float(c.get("slippage", base.slippage)))


def weight_sensitivity(fd, i0: int, i1: int, system: dict,
                       mats: dict[str, np.ndarray], amount: np.ndarray,
                       fwd: np.ndarray) -> list[dict]:
    """权重 0.5 / 0.6 / 0.7 三档：组合月频 IC + 同期组合年化（业主：不许只报一档）。

    IC 用 `ic_periods` 的窗口纪律（跨出段末的期整期丢弃），只在**可投池**上算 ——
    分母必须和实际买得到的票一致。
    """
    usable, dropped = ic_periods(fd, i0, i1)
    out = []
    for w in SENSITIVITY_WEIGHTS:
        ics = []
        for p in usable:
            i, k = p["idx"], p["k"]
            uni = build_universe(fd, i, system, mats, amount)
            cs = composite_scores(fd, i, uni, system, mats, w_bp=w)
            if cs["n_candidates"] < 30:
                continue
            ics.append(rank_ic(cs["score"], fwd[k][cs["idx"]]))
        s = _ic_stats(ics)
        # 同权重的组合表现（另一条独立证据：IC 高不等于 5 只票赚钱）
        sys_w = copy.deepcopy(system)
        sys_w.setdefault("weights", {})["bp"] = w
        sigs = build_signals(fd, sys_w, mats, amount)["signals"]
        seg = run_segment(fd, i0, i1, sigs, sys_w)
        s.update({
            "weight_bp": w, "weight_reversal": round(1.0 - w, 6),
            "portfolio_ann": (seg.get("strategy") or {}).get("ann_return"),
            "portfolio_mdd": (seg.get("strategy") or {}).get("mdd"),
            "portfolio_excess": seg.get("excess"),
        })
        out.append(s)
    return out


def _ic_stats(ics: list[float]) -> dict:
    arr = np.array([x for x in ics if np.isfinite(x)], dtype="float64")
    if len(arr) == 0:
        return {"ic_mean": None, "icir": None, "t_stat": None,
                "ic_pos_ratio": None, "n_periods": 0}
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    t = float(arr.mean() / (sd / np.sqrt(len(arr)))) if sd and np.isfinite(sd) else float("nan")
    return {"ic_mean": _r(arr.mean()), "icir": _r(arr.mean() / sd) if sd else None,
            "t_stat": _r(t), "ic_pos_ratio": _r(float((arr > 0).mean())),
            "n_periods": int(len(arr)),
            "note": "月频 IC 存在自相关，t 值高估显著性，只作量级参考，不作 alpha 证据"}


def evaluate_gates(dev: dict, holdout: dict, system: dict) -> dict:
    """硬门槛（业主拍板）：两项全过才出卡。

    dev     ：组合年化 − 等权基准年化 > dev_min_excess（默认 > 0）
    holdout ：回撤不深于基准，且方向不与 dev 相反。
              方向 = 超额的符号；dev 超额为负时"不相反"意味着 holdout 也不能转正…
              这里按拍板原话实现：**符号一致才算不相反**。dev 已是负的，holdout
              若为正反而是"方向变了"，同样要业主看一眼 —— 但那种情况更可能是
              运气而非坏事，所以单独给 direction_note 说明，不静默当成崩。
    """
    g = system.get("gate") or {}
    thr = float(g.get("dev_min_excess", 0.0))
    dev_ex = dev.get("excess")
    dev_pass = dev_ex is not None and float(dev_ex) > thr

    s = holdout.get("strategy") or {}
    b = holdout.get("bench") or {}
    ho_ex = holdout.get("excess")
    mdd_s, mdd_b = s.get("mdd"), b.get("mdd")
    tol = float(g.get("holdout_max_dd_worse", 0.0))
    dd_ok = (mdd_s is not None and mdd_b is not None
             and float(mdd_s) >= float(mdd_b) - abs(tol))

    dir_ok = None
    if g.get("holdout_check_direction", True) and dev_ex is not None and ho_ex is not None:
        dir_ok = (float(dev_ex) == 0.0 or float(ho_ex) == 0.0
                  or (float(dev_ex) > 0) == (float(ho_ex) > 0))
    ho_pass = bool(dd_ok and (dir_ok is not False))

    direction_note = None
    if dir_ok is False and dev_ex is not None and ho_ex is not None:
        if float(dev_ex) < 0.0 < float(ho_ex):
            direction_note = (
                "dev 跑输、holdout 跑赢：**相对表现变号**。这不是好消息，是不稳定证据 ——"
                " 一年样本不足以证明策略在 holdout 有优势，且 dev 的门槛①本身已不过。")
        elif float(dev_ex) > 0.0 > float(ho_ex):
            direction_note = (
                "dev 跑赢、holdout 跑输：典型的过拟合信号，样本外不成立。")

    return {
        "dev": {"pass": bool(dev_pass), "excess": dev_ex,
                "strategy_ann": (dev.get("strategy") or {}).get("ann_return"),
                "bench_ann": (dev.get("bench") or {}).get("ann_return"),
                "threshold": thr,
                "rule": "组合年化 − 等权基准年化 > 门槛（默认 > 0）"},
        "holdout": {"pass": ho_pass,
                    "strategy_ann": s.get("ann_return"), "bench_ann": b.get("ann_return"),
                    "excess": ho_ex,
                    "strategy_mdd": mdd_s, "bench_mdd": mdd_b,
                    "dd_not_worse": bool(dd_ok),
                    "direction_consistent": dir_ok,
                    "direction_note": direction_note,
                    "rule": "回撤不深于基准 且 超额符号与 dev 一致"},
        "both_pass": bool(dev_pass and ho_pass),
    }


# ---------------------------------------------------------------- 建议卡
def _reason_text(name: str, sb: float, sr: float, sc: float, rank: int,
                 n_cand: int, industry: str | None) -> str:
    """入选理由一句话：全部由数字生成，不写"看好""优质"这种没有出处的词。"""
    cheap = f"bp 分位 {sb:.3f}（比池内 {sb * 100:.1f}% 的票便宜）"
    # 反转分位 0.855 的物理含义是"跌幅处在池内最深 14.5%"，不是"前 85.5%"。
    # 这张卡上最容易写错的一句，换算写死在这里。
    rev = f"20 日反转分位 {sr:.3f}（近月跌幅居池内最深 {(1 - sr) * 100:.1f}%）"
    tail = f"综合 {sc:.3f} 列第 {rank}/{n_cand}"
    ind = f"，{industry}" if industry else ""
    return f"{cheap}，{rev}{ind}；{tail}"


def _risk_text(*, roe, debt, debt_pct, amt, mom, cap: float) -> str:
    """风险提示一句话：挑最突出的两条 + 一条永远存在的仓位风险。"""
    items: list[tuple[float, str]] = []
    if roe is not None and np.isfinite(roe):
        if roe < 0.03:
            items.append((1.0 - min(max(roe, 0.0) / 0.03, 1.0),
                          f"ROE(TTM) 仅 {roe:.2%}，贴近盈亏平衡，盈利质量弱"))
    if (debt_pct is not None and np.isfinite(debt_pct) and debt_pct > 0.70
            and debt is not None and np.isfinite(debt)):
        items.append((float(debt_pct),
                      f"资产负债率 {debt:.1f}% 处行业内 {debt_pct:.0%} 分位，杠杆偏高"))
    if amt is not None and np.isfinite(amt) and amt < 1e8:
        items.append((1.0 - float(amt) / 1e8,
                      f"成交额仅 {amt / 1e8:.2f} 亿，流动性偏薄，大额进出会多付滑点"))
    if mom is not None and np.isfinite(mom) and mom < -0.15:
        items.append((min(abs(float(mom)) / 0.3, 1.0),
                      f"近 20 日已跌 {abs(float(mom)):.1%}，存在下跌中继（接飞刀）风险"))
    items.sort(key=lambda x: -x[0])
    picked = [t for _, t in items[:2]]
    picked.append(f"单票上限 {cap:.0%}，按单票极端全损 −50% 计即 {cap * 0.5:.1%} 的组合回撤")
    return "；".join(picked)


def build_card(fd, i: int, system: dict, owner: dict, mats: dict[str, np.ndarray],
               amount: np.ndarray, names: dict[str, str]) -> dict:
    """用第 i 个截面生成建议卡正文（5 只，全字段，含得分构成明细）。"""
    n_hold = int(system.get("n_hold", 5))
    cap = float(system.get("max_weight", 0.30))
    uni = build_universe(fd, i, system, mats, amount)
    cs = composite_scores(fd, i, uni, system, mats)
    steps = uni["steps"]
    head = {
        "cross_section": fd.dates[i],
        "entry_date": (fd.dates[i + 1] if i + 1 < len(fd.dates) else None),
        "candidates": cs["n_candidates"],
        "n_hold_target": n_hold,
        "weights": {"bp": cs["weight_bp"], "reversal": cs["weight_reversal"]},
        "filters": steps,
        "vetoed_total": steps["total_vetoed"],
        "veto_note": f"{steps['total_vetoed']} 只被质量否决"
                     f"（亏损 {steps['vetoed_roe_negative']} + 高杠杆 "
                     f"{steps['vetoed_high_debt']}）",
    }
    if cs["n_candidates"] == 0:
        return {**head, "stocks": [], "short_by": 0,
                "reason": "该截面可选池为空（过滤/否决后无票），不凑数"}

    order = np.argsort(-cs["score"])[:n_hold]
    sel = cs["idx"][order]
    wt = allocate(cs["score"][order], cap)
    # 排名键必须是**全局列号**：argsort 给的是"在 idx 里的位置"，直接拿它当键
    # 会在卡上打出 None 甚至错行（踩过）。
    rank_map = {int(cs["idx"][p]): r + 1
                for r, p in enumerate(np.argsort(-cs["score"]))}
    budget = float(owner.get("budget") or 0.0)
    sw, inames = load_industry_labels(1)
    stocks = []
    for pos, (jj, w, r_pos) in enumerate(zip(sel, wt, order)):
        j = int(jj)
        sb = float(cs["bp_pct"][r_pos])
        sr = float(cs["rev_pct"][r_pos])
        sc = float(cs["score"][r_pos])
        contrib_bp = cs["weight_bp"] * sb
        contrib_rev = cs["weight_reversal"] * sr
        industry = industry_of(sw, inames, fd.codes[j], fd.dates[i])
        roe = float(fd.roe_ttm[i][j]) if fd.roe_ttm is not None else None
        debt = float(fd.debt_ratio[i][j]) if fd.debt_ratio is not None else None
        dpct = uni["debt_pct"][j]
        amt = float(amount[i][j])
        mom = float(cs["mom_raw"][r_pos])
        low = max(0.01, round(w - WEIGHT_BAND, 4))
        high = min(1.0, round(w + WEIGHT_BAND, 4))
        stocks.append({
            "code": fd.codes[j],
            "name": names.get(fd.codes[j]),
            "industry": industry,
            "bp_pct": _r(sb, 4),
            "reversal_pct": _r(sr, 4),
            "momentum_20": _r(mom, 4),
            "score": _r(sc, 4),
            "score_breakdown": {
                "formula": f"{cs['weight_bp']:.2f}×{sb:.4f} + "
                           f"{cs['weight_reversal']:.2f}×{sr:.4f} = {sc:.4f}",
                "bp": {"pct": _r(sb, 4), "weight": _r(cs["weight_bp"], 4),
                       "contribution": _r(contrib_bp, 4)},
                "reversal": {"pct": _r(sr, 4), "weight": _r(cs["weight_reversal"], 4),
                             "contribution": _r(contrib_rev, 4)},
                "sum_check": _r(contrib_bp + contrib_rev, 4),
            },
            "rank_in_universe": rank_map.get(j),
            "weight": {"point": _r(w, 4), "low": _r(low, 4), "high": _r(high, 4),
                       "text": f"{low * 100:.0f}%~{high * 100:.0f}%"},
            "amount_hint": (_r(budget * w, 2) if budget else None),
            "reason": _reason_text(names.get(fd.codes[j]) or fd.codes[j], sb, sr, sc,
                                   rank_map.get(j, pos + 1), cs["n_candidates"],
                                   industry),
            "risk": _risk_text(roe=roe, debt=debt, debt_pct=dpct, amt=amt, mom=mom,
                               cap=cap),
            "diagnostics": {
                "close": _r(fd.panel.close_raw[i][j], 4),
                "amount_yi": _r(amt / 1e8, 4),
                "roe_ttm": _r(roe, 4),
                "debt_ratio": _r(debt, 4),
                "debt_ratio_industry_pct": _r(dpct, 4),
            },
        })
    return {**head, "stocks": stocks,
            "short_by": max(0, n_hold - len(stocks)),
            "reason": (f"可选池仅 {cs['n_candidates']} 只，不足 {n_hold} 只，"
                       f"如实输出 {len(stocks)} 只，未凑数"
                       if len(stocks) < n_hold else None)}


# ---------------------------------------------------------------- 主流程
def run(cfg: dict, *, start: str = DEFAULT_START, verbose: bool = True) -> dict:
    """跑完整流程：载数据 → analyze()。载数据单独一步，便于测试注入合成面板。"""
    system = C.system_params(cfg)
    holdout_end = (system.get("holdout") or {}).get("end") or datetime.now().strftime("%Y-%m-%d")
    if verbose:
        print(f"[1/6] 载入数据 {start} ~ {holdout_end} …", flush=True)
    fd = load_factor_data(start, holdout_end)
    return analyze(fd, cfg, verbose=verbose)


def analyze(fd, cfg: dict, *, verbose: bool = False) -> dict:
    """在已载入的 FactorData 上跑完整判定链（不碰 IO，测试可直接注入）。

    顺序就是纪律：dev 回测/敏感性 → 门槛① → holdout 一次性 → 门槛② → 出卡或 fallback。
    """
    system = C.system_params(cfg)
    owner = C.owner_profile(cfg)
    mats = factor_mats(fd)
    amount = trade_amount(fd)
    names = load_names()
    dates = fd.dates
    dev_end = (system.get("holdout") or {}).get("start")
    dev_stop = [i for i, d in enumerate(dates) if d < dev_end]
    ho_start = [i for i, d in enumerate(dates) if d >= dev_end]
    if not dev_stop or not ho_start:
        raise RuntimeError(f"数据不足以切分 dev/holdout（dev 末 {dev_stop[-1:]}, "
                           f"holdout 起 {ho_start[:1]}）")
    i_dev1, i_ho0 = dev_stop[-1] + 1, ho_start[0]
    if verbose:
        print(f"      面板 {len(dates)} 日 × {len(fd.codes)} 只；"
              f"dev [0,{i_dev1}) 至 {dates[i_dev1 - 1]}；"
              f"holdout [{i_ho0},{len(dates)}) 自 {dates[i_ho0]}", flush=True)

    if verbose:
        print("[2/6] 构建全部月末信号 …", flush=True)
    built = build_signals(fd, system, mats, amount)
    sig_all = built["signals"]

    if verbose:
        print("[3/6] dev 段回测 + 权重敏感性 …", flush=True)
    from src.factors import forward_returns
    fwd_all = forward_returns(fd.panel.open_hfq, month_end_indices(dates))
    dev = run_segment(fd, 0, i_dev1, sig_all, system)
    sens = weight_sensitivity(fd, 0, i_dev1, system, mats, amount, fwd_all)

    if verbose:
        print("[4/6] holdout 段一次性回测（结果出来后不回头改参数）…", flush=True)
    holdout = run_segment(fd, i_ho0, len(dates), sig_all, system)
    gates = evaluate_gates(dev, holdout, system)

    verdict = "advise" if gates["both_pass"] else "fallback"
    if verbose:
        print(f"[5/6] 门槛判定：dev {'过' if gates['dev']['pass'] else '不过'} / "
              f"holdout {'过' if gates['holdout']['pass'] else '不过'} → {verdict}", flush=True)

    # 卡：达标用最后一个**可用**截面出正式卡；不达标也出，但只放在 preview 里并标注
    last_i = None
    for i in reversed(month_end_indices(dates)):
        if i + 1 < len(dates):
            last_i = i
            break
    preview = (build_card(fd, last_i, system, owner, mats, amount, names)
               if last_i is not None else None)
    if preview is not None:
        # 先打标再分派：反过来写会让 card 指向打标前的旧对象，卡上少了
        # is_recommendation（测试抓到的），产物里两个字段对不上。
        preview = dict(preview)
        preview["is_recommendation"] = verdict == "advise"
        preview["label"] = ("正式建议卡" if verdict == "advise" else
                            "结构预览：门槛未过，此卡**不构成建议**，仅用于核对字段与口径")
    card = preview if verdict == "advise" else None

    fallback = None
    if verdict == "fallback":
        reasons = []
        if not gates["dev"]["pass"]:
            reasons.append(
                f"dev 段组合年化 {_f(gates['dev']['strategy_ann'], '.2%')} 低于等权基准 "
                f"{_f(gates['dev']['bench_ann'], '.2%')}（超额 "
                f"{_f(gates['dev']['excess'], '+.2%')}，门槛要求 > "
                f"{gates['dev']['threshold']:.0%}）")
        if not gates["holdout"]["pass"]:
            h = gates["holdout"]
            if not h["dd_not_worse"]:
                reasons.append(f"holdout 段回撤 {_f(h['strategy_mdd'], '.2%')} 深于基准 "
                               f"{_f(h['bench_mdd'], '.2%')}")
            if h["direction_consistent"] is False:
                reasons.append(f"holdout 超额 {_f(h['excess'], '+.2%')} 与 dev 方向相反"
                               + (f"（{h['direction_note']}）" if h.get("direction_note") else ""))
        fallback = {
            "message": "本次不建议",
            "reasons": reasons,
            "numbers": {
                "dev": {"strategy_ann": gates["dev"]["strategy_ann"],
                        "bench_ann": gates["dev"]["bench_ann"],
                        "excess": gates["dev"]["excess"]},
                "holdout": {"strategy_ann": gates["holdout"]["strategy_ann"],
                            "bench_ann": gates["holdout"]["bench_ann"],
                            "excess": gates["holdout"]["excess"],
                            "strategy_mdd": gates["holdout"]["strategy_mdd"],
                            "bench_mdd": gates["holdout"]["bench_mdd"]},
            },
            "what_would_change_it": "两条硬门槛全过（dev 超额 > 0 且 holdout 不崩）才会出卡；"
                                    "要改策略口径请先改 config 并在 dev 段重新定型，"
                                    "不许用 holdout 结果反推参数。",
        }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        **_git_meta(),
        "verdict": verdict,
        "gates": gates,
        "config": {"owner": owner,
                   "system": {k: v for k, v in system.items() if k != "holdout"},
                   "holdout": system.get("holdout")},
        "segments": {"dev": dev, "holdout": holdout},
        "weight_sensitivity": sens,
        "card": card,
        "preview_card": preview,
        "fallback": fallback,
        "data": {
            "start": dates[0], "end": dates[-1], "dates": len(dates),
            "codes": len(fd.codes),
            "dev_end": dates[i_dev1 - 1], "holdout_start": dates[i_ho0],
            "shares_coverage": (fd.source.get("shares") or {}).get("codes_coverage"),
            "industry_groups": (fd.source.get("industry") or {}).get("groups"),
            "fundamentals": {k: (fd.source.get("fundamentals") or {}).get(k)
                             for k in ("rows_usable", "pubdate_missing_rate",
                                       "epsTTM_coverage")},
        },
        "signal_build": {
            "n_rebalances": len(sig_all),
            "short_cross_sections": sum(1 for d in built["detail"].values()
                                        if d.get("short")),
            "note": "可选池不足 n_hold 的截面数（该期如实少买，不凑数）",
        },
        "validity_note": build_validity_note(verdict, gates, sens),
    }
    if verbose:
        print("[6/6] 完成", flush=True)
    return report


def build_validity_note(verdict: str, gates: dict, sens: list[dict]) -> list[str]:
    """免责与口径纪律。数字全部来自本次产物，不写死。"""
    notes = [
        "本卡是**决策辅助**，不构成任何买卖指令；何时买、买多少、何时卖由业主自行决定。",
        "月频 IC 存在自相关，t 值高估显著性，只作量级参考，**不作 alpha 证据**。",
        "样本区间 end 与 dev/holdout 边界沿用 factors.DEV_END / HOLDOUT_START，"
        "holdout 只跑一次、结果出来后不回头改参数。",
        "成交额用 close×volume 近似（VWAP 口径差未计），仅用于流动性下限，不参与打分。",
    ]
    if sens:
        w06 = next((x for x in sens if abs(x.get("weight_bp", 0) - 0.6) < 1e-9), None)
        if w06:
            notes.append(
                f"权重 {w06['weight_bp']}/{w06['weight_reversal']} 的 dev 月频 IC "
                f"{_f(w06['ic_mean'], '+.4f')}（t {_f(w06['t_stat'], '+.2f')}，"
                f"{w06['n_periods']} 期），但同期组合年化 "
                f"{_f(w06['portfolio_ann'], '+.2%')} —— **排序能力为正不代表 5 只票赚钱**，"
                f"两者必须分开看。")
    if verdict == "fallback":
        notes.append("本次输出为「不建议」：硬门槛未过，卡只在 preview_card 里作结构核对，"
                     "不得当作建议使用。")
    return notes


def print_summary(rep: dict) -> None:
    print("\n" + "=" * 78)
    print(f"建议卡结论：{rep['verdict']}")
    g = rep["gates"]
    print(f"  门槛① dev     ：{_f(g['dev']['strategy_ann'], '+.2%')} vs 基准 "
          f"{_f(g['dev']['bench_ann'], '+.2%')} → 超额 {_f(g['dev']['excess'], '+.2%')}"
          f"  [{'过' if g['dev']['pass'] else '不过'}]")
    print(f"  门槛② holdout ：{_f(g['holdout']['strategy_ann'], '+.2%')} vs 基准 "
          f"{_f(g['holdout']['bench_ann'], '+.2%')}，回撤 "
          f"{_f(g['holdout']['strategy_mdd'], '.2%')} vs {_f(g['holdout']['bench_mdd'], '.2%')}"
          f"  [{'过' if g['holdout']['pass'] else '不过'}]")
    print("\n权重敏感性（dev 段）：")
    print(f"  {'w_bp':>5}{'IC均值':>10}{'t':>8}{'IC>0':>8}{'期数':>6}"
          f"{'组合年化':>10}{'组合回撤':>10}{'超额':>9}")
    for x in rep["weight_sensitivity"]:
        print(f"  {x['weight_bp']:>5.1f}{_f(x['ic_mean'], '+.4f'):>10}"
              f"{_f(x['t_stat'], '+.2f'):>8}{_f(x['ic_pos_ratio'], '.1%'):>8}"
              f"{x['n_periods']:>6}{_f(x['portfolio_ann'], '+.2%'):>10}"
              f"{_f(x['portfolio_mdd'], '.2%'):>10}{_f(x['portfolio_excess'], '+.2%'):>9}")
    crd = rep.get("preview_card") or {}
    st = crd.get("stocks") or []
    if st:
        print(f"\n{'结构预览' if not crd.get('is_recommendation') else '建议卡'}"
              f"（截面 {crd['cross_section']}，候选 {crd['candidates']} 只，"
              f"{crd['veto_note']}）：")
        print(f"  {'代码':<11}{'名称':<8}{'bp分位':>7}{'反转分位':>9}{'综合':>7}"
              f"{'仓位':>11}{'成交额亿':>9}")
        for s in st:
            print(f"  {s['code']:<11}{(s['name'] or '-'):<8}{_f(s['bp_pct'], '.3f'):>7}"
                  f"{_f(s['reversal_pct'], '.3f'):>9}{_f(s['score'], '.3f'):>7}"
                  f"{s['weight']['text']:>11}{_f(s['diagnostics']['amount_yi'], '.2f'):>9}")
    if rep.get("fallback"):
        print("\n本次不建议：")
        for r in rep["fallback"]["reasons"]:
            print(f"  · {r}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="建议卡引擎（决策辅助，不构成买卖指令）")
    ap.add_argument("--init", action="store_true", help="生成 data/config.json 模板")
    ap.add_argument("--force", action="store_true", help="--init 时覆盖已有 config")
    ap.add_argument("--config", default=None, help="配置文件路径（默认 data/config.json）")
    ap.add_argument("--start", default=DEFAULT_START, help="数据起点")
    ap.add_argument("--out", default=str(OUT_DIR), help="产物目录")
    ap.add_argument("--quiet", action="store_true", help="只出产物不打印摘要")
    args = ap.parse_args(argv)

    if args.init:
        try:
            p = C.save_template(args.config, force=args.force)
        except C.ConfigError as e:
            print(f"[错误] {e}", file=sys.stderr)
            return 2
        print(f"模板已写入 {p}\n手改 owner 段（预算/回撤/周期）后直接跑：python -m src.advisor")
        return 0

    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2

    rep = run(cfg, start=args.start, verbose=not args.quiet)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / CARD_NAME
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print_summary(rep)
    print(f"\n产物：{p}")
    # fallback 是**正常结论**不是错误，但给个可区分的退出码方便脚本判断
    return 0 if rep["verdict"] == "advise" else 3


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
