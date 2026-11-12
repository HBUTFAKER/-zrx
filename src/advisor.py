"""建议卡生成引擎（S3-R · 分位带分散组合版）。

## 005 → 005R 改了什么（一句话：评分层一字未动，变现层整个换掉）

    005 ：综合分 top5 → 按分数加权 → 单票 18%~22%（集中持仓）。
    005R：综合分前 N% 全部等权 → 30~200 只（分散持仓），N 只在**定档段**扫描确定。

动因（005 已双审确认的事实）：综合分排序有效（dev 月频 IC +0.071，t +4.03），
但 top5 恰好落在"最便宜的暴跌股"这个非单调角落 —— 分数 20 等分桶里桶19 月均 +97.4bp、
桶20 掉到 +44.3bp，取 top5 年化 −10.84%。集中持仓把尾部噪声放大成了组合亏损。
业主拍板：不追求精选个股，改分散形态，预期"年化稳定跑赢基准 2~3pp + 回撤可控"。

## 组合规格（业主拍板）

    综合得分 = bp 分位 × w_bp + 反转分位 × w_rev      ← 005 原样，未改
    反转     = momentum_20 的反向分位
    否决     = roe_ttm < 阈值 / debt_ratio 行业内分位 > 阈值   ← 005 原样
    过滤     = cross_section_mask（ST/停牌/hfq 缺口/上市 60 日）
               + 成交额下限 + 当日/次日涨停不可选                ← 005 原样
    持仓     = 综合分前 band 比例的全部股票，**等权**，每只 1/k   ← 005R 新形态

## 四条口径纪律（改代码前先看这里）

  ① **只做决策辅助，不给买卖指令。** 卡上给的是"这只票在截面里的相对位置"和
     "按等权该买多少钱"，何时买卖由业主决定。代码里没有任何下单动作。
  ② **N 只许看定档段。** band 由 `validation.band_fit_end`（默认 2021-12-31）之前
     的数据扫描确定，选完之后 holdout 与后续各年只许"验证"，不许回头改 N。
     005 里那个"分位带 80~90 年化 +10.39%"是看了全期结果的**事后观察值**，
     代码里没有任何一处用它选档（见 `scan_bands` 的入参只有 i0/i1）。
  ③ **分年冻结-后验，不做全滚动。** 业主给了两个选项，本单选"分年冻结-后验"：
     2018-01-02~2021-12-31 定档，2022/2023/2024/2025 逐年做准样本外，每年独立报告。
     理由：band 是**单个离散参数**，按月重选主要增加噪声与工程量（107 个月 × 7 档
     ≈ 749 次回测），换不来额外信息；分年 4 段已经能回答"是不是只有某一年行"。
     代价：2025 段只到 dev 末（2025-11-06），是 11 个月的短年，横向比较时需注意。
  ④ **不达标就不出卡。** 门槛任一不过 → verdict=fallback，`card` 为 null，
     只在 `preview_card` 里留结构预览并标注"不构成建议"。

## 门槛（005R 更新版 · 业主裁定）

    ① dev 段超额（**对可投池等权**，不再用全市场等权）> 0
    ② 分年各期 + holdout 均不得出现"回撤深于基准 10pp 以上"的年份
    ③ 方向检查只在 dev 通过之后才看 holdout（dev 不过直接 fallback，不比方向）
"""
from __future__ import annotations

import argparse
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
                         forward_returns, ic_periods, load_factor_data,
                         month_end_indices, rank_ic, slice_panel)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = ROOT / "data" / "cache.db"
OUT_DIR = ROOT / "out"
CARD_NAME = "advice_card.json"

#: 产物 schema 版本。字段有增删就往上加，别静默改结构（下游脚本要吃这个文件）。
SCHEMA_VERSION = "005R-1"

#: 数据起点（与 factors.py 的 CLI 默认一致；holdout 端点从 config 读，不在这里写死）
DEFAULT_START = "2018-01-02"

#: 权重敏感性要报的档位（业主拍板：不许只报一档）
SENSITIVITY_WEIGHTS = (0.5, 0.6, 0.7)

#: 分位带候选档（缺省值；实际以 config.system.portfolio.band_candidates 为准）
DEFAULT_BANDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)


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
    单独查一次。读不到就返回 (None, {})，卡上行业留空，不编造。
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


def industry_map_at(sw: pd.DataFrame | None, names: dict[str, str], date: str,
                    codes: list[str]) -> dict[str, str | None]:
    """一次性取一批票在 date 当天的一级行业名（in_date ≤ date 的最后一条）。

    005 里是逐只票过滤整张表（5 只票，代价可忽略）；005R 一次要出 30~200 只，
    改成一次 groupby，避免 O(成分股 × 全表) 的重复扫描。
    """
    if sw is None or not codes:
        return {}
    try:
        sub = sw[sw["in_date"] <= date]
        if sub.empty:
            return {}
        last = sub.groupby("panel_code", sort=False).tail(1)   # 已按 in_date 升序
        g = dict(zip(last["panel_code"], last["grp"]))
        want = set(codes)
        return {c: names.get(str(v)) for c, v in g.items() if c in want}
    except Exception:  # noqa: BLE001
        return {}


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
                     w_bp: float | None = None,
                     factor_names: list[str] | None = None) -> dict:
    """按指定因子截面分位简单等权；兼容旧两因子评分调用。"""
    if factor_names is None:
        w = system.get("weights") or {}
        wb = float(w.get("bp", 0.6)) if w_bp is None else float(w_bp)
        wr = 1.0 - wb
        factor_names = ["bp", "momentum_20"]
        directions = {"bp": 1.0, "momentum_20": -1.0}
        weights = {"bp": wb, "momentum_20": wr}
    else:
        factor_names = list(factor_names)
        if not factor_names:
            raise ValueError("factor_names 不能为空")
        directions = {name: 1.0 for name in factor_names}
        weights = {name: 1.0 / len(factor_names) for name in factor_names}

    mask = uni["mask"]
    values = {name: mats[name][i] for name in factor_names}
    ok = mask.copy()
    for name in factor_names:
        ok &= np.isfinite(values[name])
    idx = np.nonzero(ok)[0]
    out = {"factor_names": factor_names, "factor_weights": weights, "idx": idx,
           "n_candidates": int(len(idx)), "score": None, "factor_pct": None,
           "mom_raw": mats.get("momentum_20", np.full(mask.shape, np.nan))[i][idx]}
    if factor_names == ["bp", "momentum_20"]:
        out.update({"weight_bp": weights["bp"], "weight_reversal": weights["momentum_20"],
                    "bp_pct": None, "rev_pct": None})
    if len(idx) == 0:
        return out
    pcts = {name: directions[name] * _pct_rank(values[name][idx])
            if directions[name] > 0 else 1.0 - _pct_rank(values[name][idx])
            for name in factor_names}
    out["factor_pct"] = pcts
    out["score"] = sum(weights[n] * pcts[n] for n in factor_names)
    if factor_names == ["bp", "momentum_20"]:
        out["bp_pct"], out["rev_pct"] = pcts["bp"], pcts["momentum_20"]
    return out


# ---------------------------------------------------------------- 截面缓存（评分层唯一出口）
def score_cross_sections(fd, system: dict, mats: dict[str, np.ndarray],
                         amount: np.ndarray, w_bp: float | None = None,
                         factor_names: list[str] | None = None) -> dict[int, dict]:
    """把**全部月末截面**的评分算一次缓存起来，供各档位复用。

    这么做有两个实在好处（不只是省时间）：
      · 所有 band 共用同一份得分，档位之间的差异**只可能**来自"取多少只"，
        不会因为重复计算引入浮点噪声，选档结论才干净；
      · 每档只做一次排序切片，回测次数从"档位 × 截面"降到"档位 × 段"。
    缓存里存 mask 是为了给"可投池等权"基准用（同一 mask、同一调仓日）。
    """
    cache: dict[int, dict] = {}
    for i in month_end_indices(fd.dates):
        uni = build_universe(fd, i, system, mats, amount)
        cs = composite_scores(fd, i, uni, system, mats, w_bp=w_bp,
                              factor_names=factor_names)
        cs = dict(cs)
        cs["i"] = i
        cs["date"] = fd.dates[i]
        cs["mask"] = uni["mask"]
        cs["steps"] = uni["steps"]
        cs["debt_pct"] = uni["debt_pct"]
        cs["order"] = (np.argsort(-cs["score"], kind="stable")
                       if cs["n_candidates"] else np.array([], dtype="int64"))
        cache[i] = cs
    return cache


def rolling_factor_mats(fd) -> dict[str, np.ndarray]:
    """004C 滚动评分所需因子及旧否决项。"""
    from src.factors import FACTOR_DEFS
    mats = factor_mats(fd)
    for name in ("lowvol_20", "lowvol_60", "lowvol_120", "turn_stab_60",
                 "reversal_20", "reversal_40", "reversal_60"):
        mats[name] = FACTOR_DEFS[name](fd)
    return mats


def rolling_factor_selection(report: dict, window: str) -> tuple[list[str], dict]:
    """从 C1 已冻结的逐窗判定中读取入选集合，不重算研究结果。"""
    selected = []
    for name, verdict in report["c4_verdict"].items():
        row = next((r for r in verdict["per_window"] if r["window"] == window), None)
        if row and row.get("pass"):
            selected.append(name)
    reversal = report["reversal_primary"][window]
    rev_name = f"reversal_{reversal['primary_window']}"
    selected = [n for n in selected if not n.startswith("reversal_")]
    if rev_name in report["c4_verdict"]:
        row = next((r for r in report["c4_verdict"][rev_name]["per_window"]
                    if r["window"] == window), None)
        if row and row.get("pass"):
            selected.append(rev_name)
    return selected, reversal


def rolling_validation(fd, system: dict, mats: dict[str, np.ndarray],
                       amount: np.ndarray, *, verbose: bool = False) -> dict:
    """24/12逐窗冻结因子与主窗，2021~2025完整年度滚动验证。"""
    from src.research_004c import build_windows
    val = system["validation"]
    report_path = ROOT / val["factor_report"]
    factor_report = json.loads(report_path.read_text(encoding="utf-8"))
    windows = build_windows(fd.dates)
    if len(windows) != 5 or any(w["i_test1"] - w["i_test0"] < 200 for w in windows):
        raise RuntimeError("rolling_24_12 需要完整的 2021~2025 五个年度窗口")

    rows, portfolios = [], []
    w1_factors, _ = rolling_factor_selection(factor_report, windows[0]["name"])
    if not w1_factors:
        raise RuntimeError("W1 没有通过 C4 的因子，不能在 2020-12-31 前定档")
    fit_cache = score_cross_sections(fd, system, mats, amount,
                                     factor_names=w1_factors)
    fit0, fit1, fit_meas = _seg_window(fd, fd.dates[0], "2020-12-31")
    band_rows = scan_bands(fd, system, mats, amount, fit0, fit1, fit_meas,
                           [float(x) for x in system["portfolio"]["band_candidates"]],
                           factor_names=w1_factors)
    pick = select_band(band_rows, 0.0)
    band = pick["selected"]
    for w in windows:
        factor_names, reversal = rolling_factor_selection(factor_report, w["name"])
        if not factor_names:
            raise RuntimeError(f"{w['name']} 没有通过 C4 的因子，不构造空综合分")
        cache = score_cross_sections(fd, system, mats, amount,
                                     factor_names=factor_names)
        sig = signals_from_cache(fd, cache, band)
        i0, i1 = w["i_test0"], w["i_test1"]
        # 用测试段前一个交易日作预热，指标从测试首日开始，保留冻结前的最后一期持仓。
        warm0 = max(0, i0 - 1)
        seg = run_segment(fd, cache, warm0, i1, sig, system, i_meas=i0)
        strategy = seg.get("strategy") or {}
        bench = seg.get("bench") or {}
        benchmark_raw = pd.read_parquet(ROOT / "data/raw/index/hs300_total_return_2026-11-09.parquet")
        benchmark_raw = benchmark_raw.set_index("date")["close"].reindex(fd.dates[i0:i1])
        if benchmark_raw.isna().any():
            raise RuntimeError(f"{w['name']} H00300 缺少检验交易日")
        benchmark_nav = benchmark_raw / float(benchmark_raw.iloc[0])
        test_nav = pd.Series((seg.get("_strategy_nav") or {}), dtype="float64").reindex(fd.dates[i0:i1])
        if test_nav.isna().any():
            raise RuntimeError(f"{w['name']} 策略净值缺少检验交易日")
        row_excess = _r(M.ann_return(test_nav) - M.ann_return(benchmark_nav))
        row_benchmark_mdd = _r(M.max_drawdown(benchmark_nav)["mdd"])
        seg["_benchmark_nav"] = {str(d): float(v) for d, v in zip(benchmark_nav.index,
                                                                     benchmark_nav.to_numpy())}
        row = {"window": w["name"], "period": str(w["test_year"]),
               "train_start": w["train_start"], "train_end": w["train_end"],
               "test_start": w["test_start"], "test_end": w["test_end"],
               "selected_factors": factor_names,
               "factor_weights": {n: 1.0 / len(factor_names) for n in factor_names},
               "reversal_primary": reversal, "band": band,
               "band_selection_forced": pick["forced"], "forced": pick["forced"],
               "excess": row_excess, "strategy_ann": _r(M.ann_return(test_nav)),
               "benchmark_ann": _r(M.ann_return(benchmark_nav)),
               "strategy_mdd": _r(M.max_drawdown(test_nav)["mdd"]),
               "benchmark_mdd": row_benchmark_mdd,
               "n_days": seg.get("n_days"), "n_rebalance": seg.get("n_rebalance"),
               "k_median": seg.get("k_median"), "avg_cash_ratio": strategy.get("avg_cash_ratio")}
        rows.append(row)
        portfolios.append({"window": w, "system": system, "cache": cache,
                           "signals": sig, "segment": seg, "row": row})
        if verbose:
            print(f"[004C] {w['name']} {factor_names} band={band:.0%} excess={seg.get('excess')}",
                  flush=True)

    excesses = [r["excess"] for r in rows if r["excess"] is not None]
    mean_excess = float(np.mean(excesses)) if excesses else None
    positive = sum(x > 0 for x in excesses)
    strategy_nav, benchmark_nav = {}, {}
    strategy_level = benchmark_level = 1.0
    for item in portfolios:
        seg = item["segment"]
        strat_part = pd.Series(seg.get("_strategy_nav") or {}, dtype="float64")
        bench_part = pd.Series(seg.get("_benchmark_nav") or {}, dtype="float64")
        if strat_part.empty or bench_part.empty:
            raise RuntimeError("滚动窗口缺少策略或 H00300 净值")
        strat_part = strat_part / float(strat_part.iloc[0]) * strategy_level
        bench_part = bench_part / float(bench_part.iloc[0]) * benchmark_level
        strategy_level = float(strat_part.iloc[-1])
        benchmark_level = float(bench_part.iloc[-1])
        strategy_nav.update(strat_part.to_dict())
        benchmark_nav.update(bench_part.to_dict())
    expected_days = sum(item["window"]["i_test1"] - item["window"]["i_test0"]
                        for item in portfolios)
    if len(strategy_nav) != expected_days:
        raise RuntimeError("滚动窗口策略净值缺失或日期重叠，不能计算全段回撤")
    strat_series = pd.Series(strategy_nav).sort_index()
    benchmark_series = pd.Series(benchmark_nav).reindex(strat_series.index)
    if benchmark_series.isna().any():
        raise RuntimeError("H00300 基准净值缺少滚动检验交易日")
    mdd = _nav_metrics(strat_series)["mdd"]
    rolling = {"method": "rolling_24_12", "benchmark": "H00300",
               "benchmark_path": "data/raw/index/hs300_total_return_2026-11-09.parquet",
               "band_selection": {"fit_end": "2020-12-31", "factors": w1_factors,
                                   "selected": band, "forced": pick["forced"],
                                   "table": band_rows},
               "band_selection_forced": pick["forced"],
               "windows": rows, "mean_excess": _r(mean_excess),
               "positive_windows": positive, "window_count": len(rows),
               "portfolio_mdd": mdd, "portfolio_mdd_peak": M.max_drawdown(strat_series)["peak"],
               "portfolio_mdd_trough": M.max_drawdown(strat_series)["trough"],
               "_portfolios": portfolios}
    return rolling


def analyze_004c(fd, cfg: dict, *, verbose: bool = False) -> dict:
    """004C 执行链：C1冻结因子 → C2滚动回测 → C5准入闸。"""
    system = C.system_params(cfg)
    mats = rolling_factor_mats(fd)
    amount = trade_amount(fd)
    rolling = rolling_validation(fd, system, mats, amount, verbose=verbose)
    forced = bool(rolling.get("band_selection_forced"))
    gates = evaluate_gates({}, {}, None, system, forced=forced, rolling=rolling)
    verdict = "advise" if gates["both_pass"] else "fallback"
    reason = gates["fallback_reason"]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": "004C-1", **_git_meta(), "verdict": verdict,
        "fallback_reason": reason,
        "validation": {k: v for k, v in rolling.items() if not k.startswith("_")},
        "gates": gates,
        "fallback": (None if verdict == "advise" else {
            "message": "不可作投资建议引用", "fallback_reason": reason,
            "reasons": [f"004C 策略准入闸未通过：{reason}"],
            "research_conclusion": "因子研究与滚动窗口结果保留，不因 fallback 丢弃。"}),
        "card": None,
        "validity_note": [
            "验证基准为沪深300全收益指数 H00300。",
            "窗口级回撤仅记录；策略准入回撤为五个滚动检验年度串接后的组合最大回撤。",
            "全段最大回撤由最深年份主导，不是五年逐年回撤考核。",
            "窗口因子与反转主窗均由对应训练段冻结，检验段只评估。"],
    }
    return report


def select_top_k(cs: dict, band: float) -> tuple[int, np.ndarray]:
    """前 band 比例：k = ceil(band × 候选数)，取综合分最高的 k 只。

    返回 (k, sel) —— sel 是**缓存数组里的位置**（已按分数降序），不是全局列号。
    上游要换成代码时用 `cs["idx"][sel]`。
    """
    n = int(cs["n_candidates"])
    if n == 0:
        return 0, np.array([], dtype="int64")
    k = int(np.ceil(float(band) * n))
    k = max(1, min(k, n))
    return k, cs["order"][:k]


def _positions_note(k_med: int, expected: list | None) -> str | None:
    """持仓只数是否落在业主预期区间。**只提示，不调档迁就**。"""
    lo, hi = (expected or [30, 200])[:2]
    if lo <= k_med <= hi:
        return None
    return (f"注意：中位只数 {k_med} 不在预期的 {lo}~{hi} 只区间内。"
            f"不为了凑进区间去改 band —— band 只由定档段的超额与回撤决定；"
            f"规模与超额在这份数据上不可兼得时，如实报告两边数字。")


def _convergence_note(tightest_positive: float | None) -> str:
    """band → 100% 时组合收敛到基准，这个几何事实必须写在明面上。"""
    tail = ("定档段没有任何一档超额为正，所以也谈不上「最窄的正超额档」。"
            if tightest_positive is None else
            f"定档段超额为正的最窄档是 {tightest_positive:.0%}，只有那一档才算真正偏离基准。")
    return ("几何事实：band → 100% 时组合收敛到池基准，超额必然趋 0。"
            "因此「最宽的档超额最高」可能只是「更接近基准」，不等于选股更好。" + tail)


def side_band_signals(fd, cache: dict[int, dict], lo: float, hi: float) -> dict:
    """分数分位落在 [lo, hi) 的那一段，等权（旁证用，**不参与选档**）。

    分位定义写死在这里：rank r（1 起）的分数分位 = 1 − (r−1)/n。
    所以 [0.80, 0.90) 就是"从第 1 名往下数 10%~20% 那一段"，不是最顶端。
    """
    sig: dict[str, dict[str, float]] = {}
    for i in sorted(cache):
        cs = cache[i]
        n = int(cs["n_candidates"])
        if n == 0:
            continue
        ranks = np.arange(1, n + 1)
        pct = 1.0 - (ranks - 1) / n
        pos = np.nonzero((pct >= lo) & (pct < hi))[0]
        if not len(pos):
            continue
        order = cs["order"][pos]                      # 保持分数降序
        sig[cs["date"]] = {fd.codes[int(j)]: 1.0 / len(order)
                           for j in cs["idx"][order]}
    return sig


def _side_reference(fd, cache: dict[int, dict], i0: int, i1: int, system: dict,
                    lo: float = 0.80, hi: float = 0.90) -> dict | None:
    """「分数分位 80~90」那一段同期表现 —— 业主指定要报的旁证。

    它在 005 的诊断里被看到过（当时年化最好），**这是事后观察值，不是选择依据**。
    本函数只在定档完成后调用一次，结果只写进产物作对照，不进门槛、不回写 band。
    """
    sig = side_band_signals(fd, cache, lo, hi)
    if not sig:
        return None
    seg = run_segment(fd, cache, i0, i1, sig, system, i_meas=i0)
    ks = [len(w) for w in sig.values() if w]
    return {
        "lo": lo, "hi": hi,
        "definition": "综合分分位落在 [0.80, 0.90) 的股票等权（分位 = 1 − (名次−1)/n，"
                      "即从第 1 名往下数 10%~20% 那一段）",
        "ann_return": (seg.get("strategy") or {}).get("ann_return"),
        "mdd": (seg.get("strategy") or {}).get("mdd"),
        "pool_ann": (seg.get("bench") or {}).get("ann_return"),
        "excess": seg.get("excess"),
        "k_median": int(np.median(ks)) if ks else 0,
        "label": "事后观察值，非选择依据",
        "why": "这是选完 band 之后才看的对照。它可能比选中的档更好，但："
               "① 这个区间是先看过全期结果才挑出来的，属事后拟合；"
               "② 它不是本单的持仓形态（本单是 top N% 等权）；"
               "③ 拿它反过来选档，正是 005 已经踩过的「看着结果改参数」。"
               "所以只作旁证，不进门槛、不影响 band。",
    }


def band_neighbors(band: float, candidates: list[float]) -> dict:
    """选中档的左右邻居（±1 档）。边界档只有一侧，缺的那侧如实给 null。

    用途：选档结论必须能说明"换一档会怎样"。业主给的容差是 ±1 档浮动，
    所以卡上固定报三行（上一档/选中/下一档），边界处不硬凑。
    """
    c = sorted({float(x) for x in candidates})
    if band not in c:
        lo = max([x for x in c if x < band], default=None)
        hi = min([x for x in c if x > band], default=None)
        return {"prev": lo, "selected": float(band), "next": hi,
                "note": "选中档不在候选档列表里（config 被手改过），邻居按插入位置推断"}
    p = c.index(band)
    return {"prev": c[p - 1] if p > 0 else None,
            "selected": float(band),
            "next": c[p + 1] if p + 1 < len(c) else None}


def signals_from_cache(fd, cache: dict[int, dict], band: float) -> dict[str, dict[str, float]]:
    """{日期: {代码: 1/k}} —— 等权，k 由该截面的候选数和 band 决定（逐截面浮动）。"""
    sig: dict[str, dict[str, float]] = {}
    for i in sorted(cache):
        cs = cache[i]
        k, sel = select_top_k(cs, band)
        if k == 0:
            continue
        w = 1.0 / k
        cols = cs["idx"][sel]
        sig[cs["date"]] = {fd.codes[int(j)]: w for j in cols}
    return sig


def build_signals(fd, system: dict, mats: dict[str, np.ndarray], amount: np.ndarray,
                  band: float | None = None) -> dict:
    """全部月末截面的目标权重信号（兼容 005 的返回结构：signals + detail）。

    band 取 `system.portfolio.band`；显式传参可覆盖（敏感性扫描用）。
    """
    port = system.get("portfolio") or {}
    b = float(port.get("band", 0.05)) if band is None else float(band)
    cache = score_cross_sections(fd, system, mats, amount)
    sig = signals_from_cache(fd, cache, b)
    detail: dict[str, dict] = {}
    for i in sorted(cache):
        cs = cache[i]
        k, sel = select_top_k(cs, b)
        detail[cs["date"]] = {"n_candidates": int(cs["n_candidates"]), "k": int(k),
                              "picked": [fd.codes[int(j)] for j in cs["idx"][sel]],
                              # k 由候选数算出，天然不会"凑不满"；只有池子空才是短
                              "short": bool(k == 0)}
    return {"signals": sig, "detail": detail, "cache": cache, "band": b}


# ---------------------------------------------------------------- 基准与回测
def pool_equal_weight(fd, cache: dict[int, dict], i0: int, i1: int, *,
                      rebalance: str = "monthly") -> pd.Series:
    """可投池等权基准：**同一 mask、同一调仓日、无成本**。

    这是 005R 门槛用的基准（业主裁定：不再用全市场等权）。
    与 `metrics.equal_weight_market` 的差别有两处，都写清楚：

    ① **分母是池子**，不是全市场。本策略有 5000 万成交额下限，全市场基准里那批
       小微盘贡献了 size 因子的 +14%/年，直接拿来比会系统性抬高门槛。
    ② **默认按持仓口径月频持有**（`rebalance="monthly"`），不是日频再平衡。
       `metrics.equal_weight_market` 是"每日成分股收益的简单平均"，等价于**每天**
       把组合调回等权；策略是月频调仓，两者不是同一件事。实测这段差距不小：
       dev 全段 日频 +2.96% / 月频 +2.35%（差 0.61pp），2024 年差 1.45pp，
       holdout 反过来日频低 2.88pp。门槛用月频（与策略同频、无成本），
       日频那版作为 `bench_daily` 一并报出，谁高谁低自己看。

    口径细节：
      · 停牌 / hfq 缺口用 `ffill` 沿用最后有效价（与回测的冻结估值一致），
        不像日频版那样把该股整天剔出分母；
      · 月末 m 收盘出信号 → m+1 起持有，所以第 i 日的持仓用"最后一个 < i 的月末"的 mask；
      · 段首无历史月末时（i0 之前没有截面）当日收益记 0，与策略同期的空仓对称。
    """
    dates = fd.dates
    px = fd.panel.close_hfq
    fpx = pd.DataFrame(px[i0:i1], copy=True).ffill().to_numpy()
    n = i1 - i0

    # 每个交易日生效的月末（严格 < i）；段首没有截面时记 -1（空仓）
    ends = sorted(cache)
    active = np.full(n, -1, dtype="int64")
    pos = 0
    for t in range(n):
        i = i0 + t
        while pos + 1 < len(ends) and ends[pos + 1] < i:
            pos += 1
        if ends and ends[pos] < i:
            active[t] = ends[pos]

    nav = np.ones(n, dtype="float64")
    if rebalance == "daily":
        prev = np.vstack([fpx[:1], fpx[:-1]])
        ok = np.isfinite(fpx) & np.isfinite(prev) & (prev > 0)
        ret = np.where(ok, fpx / np.where(ok, prev, 1.0) - 1.0, np.nan)
        for t in range(1, n):
            m = active[t]
            if m < 0:
                nav[t] = nav[t - 1]
                continue
            sel = cache[m]["mask"] & ok[t]
            r = ret[t][sel]
            r = r[np.isfinite(r)]
            nav[t] = nav[t - 1] * (1.0 + (float(r.mean()) if len(r) else 0.0))
    else:
        start = 0
        while start < n:
            m = active[start]
            t = start
            while t + 1 < n and active[t + 1] == m:
                t += 1
            if m < 0:
                lvl = nav[start - 1] if start > 0 else 1.0
                nav[start:t + 1] = lvl
            else:
                if start > 0:
                    base, lvl = fpx[start - 1], nav[start - 1]
                elif i0 > 0:
                    base, lvl = px[i0 - 1], 1.0
                else:
                    base, lvl = fpx[start], 1.0
                sel = cache[m]["mask"] & np.isfinite(base) & (base > 0)
                if not sel.any():
                    nav[start:t + 1] = lvl
                else:
                    for k in range(start, t + 1):
                        rel = fpx[k][sel] / base[sel]
                        rel = rel[np.isfinite(rel)]
                        nav[k] = lvl * (float(rel.mean()) if len(rel) else 1.0)
            start = t + 1
    return pd.Series(nav, index=[dates[i] for i in range(i0, i1)],
                     name="pool_equal_weight")


def _nav_metrics(nav: pd.DataFrame | pd.Series) -> dict:
    s = M.to_series(nav)
    dd = M.max_drawdown(s)
    return {"ann_return": _r(M.ann_return(s)), "mdd": _r(dd["mdd"]),
            "mdd_peak": dd["peak"], "mdd_trough": dd["trough"],
            "vol": _r(M.ann_vol(s)), "sharpe": _r(M.sharpe(s)),
            "total_return": _r(float(s.iloc[-1] / s.iloc[0] - 1.0)) if len(s) > 1 else None,
            "nav_first": str(s.index[0]), "nav_last": str(s.index[-1]),
            "n_days": int(len(s))}


def _costs(system: dict) -> Costs:
    c = system.get("costs") or {}
    base = Costs()
    return Costs(commission_rate=float(c.get("commission_rate", base.commission_rate)),
                 min_commission=float(c.get("min_commission", base.min_commission)),
                 stamp_tax=float(c.get("stamp_tax", base.stamp_tax)),
                 transfer_fee=float(c.get("transfer_fee", base.transfer_fee)),
                 slippage=float(c.get("slippage", base.slippage)))


def _seg_window(fd, start: str, end: str) -> tuple[int, int, int]:
    """把"日期区间"换成 (i0, i1, i_meas)。

    i_meas = 段内第一个交易日（**指标从这里开始算**）
    i0     = i_meas 之前最后一个月末（**预热**：用上一期的信号建仓，避免每段开年空仓
             一个月的假拖累。只用到段前的历史数据，不构成前视）
    i1     = 段末之后第一个交易日（切片右开）
    """
    dates = fd.dates
    i_meas = next((i for i, d in enumerate(dates) if d >= start), None)
    if i_meas is None:
        raise RuntimeError(f"数据里没有 >= {start} 的交易日")
    i1 = next((i for i, d in enumerate(dates) if d > end), len(dates))
    if i1 <= i_meas:
        raise RuntimeError(f"区间 {start}~{end} 在数据里为空")
    ends = [m for m in month_end_indices(dates) if m < i_meas]
    i0 = ends[-1] if ends else i_meas
    return i0, i1, i_meas


def run_segment(fd, cache: dict[int, dict], i0: int, i1: int, sig_all: dict,
                system: dict, *, i_meas: int | None = None) -> dict:
    """在一段区间上跑等权组合，并给出可投池等权基准的同期表现。

    指标一律从 `i_meas` 起算（预热段只用来建仓，不进年化/回撤）—— 否则每年 1 月
    的空仓会被算成"策略亏损"，逐年比较就失真了。

    **执行参数只用本段的信号推**（`max_positions` = 本段最多几只，`max_weight` =
    本段最大单票权重，由于等权它们恒不成为约束）。这一点是硬纪律：
    005R 第一版用"全样本 k 中位数"当 max_positions，等于把**未来截面**的候选数
    带进了当期执行 —— 定档段之后改行情就能改到定档段的回测结果，
    被 `test_band_scan_ignores_future_data` 当场抓住。现在本函数对段外数据零依赖。
    """
    dates = fd.dates
    i_meas = i0 if i_meas is None else i_meas
    panel = slice_panel(fd.panel, i0, i1)
    lo, hi = dates[i0], dates[i1 - 1]
    sig = {d: w for d, w in sig_all.items() if lo <= d <= hi}
    bt = system.get("backtest") or {}

    ks = [len(w) for w in sig.values() if w]
    w_max = max((max(w.values()) for w in sig.values() if w), default=0.0)
    cfg = BtConfig(
        start=lo, end=hi,
        init_cash=float(bt.get("init_cash", 5e7)),
        max_positions=max(1, max(ks) if ks else 1),
        max_weight=min(1.0, w_max) if w_max > 0 else 1.0,
        lot_size=int(bt.get("lot_size", 100)),
        costs=_costs(system))
    res = run_backtest(panel, sig, cfg) if sig else None

    pool = pool_equal_weight(fd, cache, i0, i1)
    pool_d = pool_equal_weight(fd, cache, i0, i1, rebalance="daily")
    off = i_meas - i0
    pool_m = _nav_metrics(pool.iloc[off:] / float(pool.iloc[off])) if len(pool) > off else None
    pool_dm = _nav_metrics(pool_d.iloc[off:] / float(pool_d.iloc[off])) if len(pool_d) > off else None
    market = M.equal_weight_market(panel)
    market_m = _nav_metrics(market.iloc[off:] / float(market.iloc[off])) if len(market) > off else None

    out = {
        "start": dates[i_meas], "end": hi, "warmup_start": lo,
        "n_days": len(dates[i_meas:i1]),
        "n_rebalance": sum(1 for d in sig if d >= dates[i_meas]),
        "n_rebalance_warmup": sum(1 for d in sig if d < dates[i_meas]),
        "k_median": int(np.median(ks)) if ks else 0,
        "k_min": int(min(ks)) if ks else 0, "k_max": int(max(ks)) if ks else 0,
        "bench": {"name": "pool_equal_weight",
                  "ann_return": (pool_m or {}).get("ann_return"),
                  "mdd": (pool_m or {}).get("mdd"),
                  "mdd_peak": (pool_m or {}).get("mdd_peak"),
                  "mdd_trough": (pool_m or {}).get("mdd_trough"),
                  "definition": "过滤+否决后的可投池，**月频等权持有**、无成本"
                                "（005R 门槛基准，与策略同调仓频率）"},
        "bench_daily": {"ann_return": (pool_dm or {}).get("ann_return"),
                        "mdd": (pool_dm or {}).get("mdd"),
                        "definition": "同一可投池，但每日收益简单平均 = 日频再平衡，"
                                      "含日内再平衡收益，仅作口径对照，不作门槛"},
        "bench_market": {"name": "equal_weight_market",
                         "ann_return": (market_m or {}).get("ann_return"),
                         "mdd": (market_m or {}).get("mdd"),
                         "definition": "全市场 hfq 等权（S2 口径，日频再平衡，仅作对照）"},
    }
    if res is None:
        out["strategy"] = {"available": False, "reason": "no_signal"}
        out["excess"] = None
        out["excess_market"] = None
        return out

    nav = M.to_series(res.nav)
    if len(nav) > off:
        nav = nav.iloc[off:] / float(nav.iloc[off])
    m = _nav_metrics(nav)
    m["n_trades"] = int(len(res.trades))
    m["n_dropped"] = int(len(res.dropped)) if res.dropped is not None else 0
    if res.dropped is not None and len(res.dropped) and "reason" in res.dropped:
        m["dropped_by_reason"] = {str(kk): int(v) for kk, v in
                                  res.dropped["reason"].value_counts().items()}
    # 现金占比：一手取整 + 单票上限会留下现金拖累。**必须报**，否则大 k 的档位
    # 会被"没投出去的现金"悄悄惩罚，选档结论就不成立。
    navd = res.nav.iloc[off:] if len(res.nav) > off else res.nav
    if len(navd) and float(navd["equity"].iloc[0]) > 0:
        ratio = (navd["cash"] / navd["equity"]).to_numpy(dtype="float64")
        m["avg_cash_ratio"] = _r(float(np.mean(ratio)))
        m["positions_mean"] = _r(float(navd["positions"].mean()), 1)
    m["available"] = True
    out["strategy"] = m
    out["_strategy_nav"] = {str(d): float(v) for d, v in zip(nav.index, nav.to_numpy())}
    out["excess"] = _r(m["ann_return"] - out["bench"]["ann_return"])
    out["excess_market"] = _r(m["ann_return"] - out["bench_market"]["ann_return"])
    return out


# ---------------------------------------------------------------- 选档与验证
def scan_bands(fd, system: dict, mats: dict[str, np.ndarray], amount: np.ndarray,
               i0: int, i1: int, i_meas: int, bands: list[float],
               factor_names: list[str] | None = None) -> list[dict]:
    """在**一段**区间上把各候选档位都跑一遍（选档只许调这个函数，传入的只能是定档段）。

    返回每档一行：只数、年化、回撤、对池基准超额、现金拖累。
    """
    cache = score_cross_sections(fd, system, mats, amount, factor_names=factor_names)
    rows = []
    for b in bands:
        sig = signals_from_cache(fd, cache, b)
        # 只数统计限在**本段**的信号上：段外截面一只都不许掺进来（否则就是前视）
        ks = [len(w) for d, w in sig.items() if fd.dates[i0] <= d <= fd.dates[i1 - 1] and w]
        seg = run_segment(fd, cache, i0, i1, sig, system, i_meas=i_meas)
        s = seg.get("strategy") or {}
        rows.append({
            "band": round(float(b), 4),
            "k_median": int(np.median(ks)) if ks else 0,
            "k_min": int(min(ks)) if ks else 0,
            "k_max": int(max(ks)) if ks else 0,
            "strategy_ann": s.get("ann_return"),
            "strategy_mdd": s.get("mdd"),
            "pool_ann": (seg.get("bench") or {}).get("ann_return"),
            "pool_mdd": (seg.get("bench") or {}).get("mdd"),
            "excess": seg.get("excess"),
            "excess_market": seg.get("excess_market"),
            "avg_cash_ratio": s.get("avg_cash_ratio"),
            "n_trades": s.get("n_trades"),
        })
    return rows


def select_band(rows: list[dict], max_dd_worse: float) -> dict:
    """从定档段的结果里选一档：**先过回撤约束，再取超额最高**。

    合格 = 回撤不深于池基准 `max_dd_worse`（默认 10pp）**且超额为正**。
    两条都要：只看回撤会把"哪档都跑输"的情况选成"最不差的那档"，看着像有依据，
    其实是没依据。若一档都不合格：取超额最高的一档，`forced=True`，并把
    "定档段没有一档跑赢"写进 rule —— 卡上必须能看到这句话。

    另一个必须写下来的几何事实：band → 100% 时组合**收敛到池基准**，超额必然趋 0。
    所以"最宽的档超额最高"这件事本身可能是"更接近基准"而非"选股更好"，
    本函数不替读者下结论，但 `tightest_positive` 会给出"超额为正的最窄档"，
    方便业主看：如果最窄的正超额档离基准还很远，那才是真的差异化。
    **holdout 与后续各年的数据一律不得进入本函数。**
    """
    if not rows:
        raise RuntimeError("定档段没有任何档位结果")

    def _dd_ok(r: dict) -> bool:
        return (r.get("strategy_mdd") is not None and r.get("pool_mdd") is not None
                and float(r["strategy_mdd"]) >= float(r["pool_mdd"]) - abs(max_dd_worse))

    ok = [r for r in rows if _dd_ok(r)]
    pos = [r for r in ok if r.get("excess") is not None and float(r["excess"]) > 0]
    pool = pos or ok or rows
    best = max(pool, key=lambda r: (r["excess"] if r["excess"] is not None else -9e9))
    best_ex = best.get("excess")
    tightest = min(pos, key=lambda r: r["band"])["band"] if pos else None

    if pos:
        rule = (f"合格档 = 回撤不深于池基准 {max_dd_worse:.0%} 且超额 > 0；"
                f"在其中取超额最高的一档（{best['band']:.0%}，超额 {best_ex:+.2%}）")
    elif ok:
        rule = (f"定档段**没有一档跑赢池基准**（最好一档 {best['band']:.0%} 超额 "
                f"{best_ex:+.2%}），只能在回撤合格的档位里取最接近的一档 —— "
                f"这不是有证据的选档，卡上已标注 forced")
    else:
        rule = (f"定档段没有一档同时满足「回撤不深于池 {max_dd_worse:.0%}」与「超额 > 0」，"
                f"退化为取超额最高的一档（forced）")
    return {
        "selected": float(best["band"]),
        "forced": not pos,
        "qualified_bands": [r["band"] for r in (pos or ok)],
        "positive_bands": [r["band"] for r in pos],
        "tightest_positive_band": tightest,
        "best_excess": best_ex,
        "rule": rule,
        "selected_row": best,
    }


def yearly_validation(fd, cache: dict[int, dict], sig_all: dict, system: dict,
                      years: list[str], dev_end: str, max_dd_worse: float) -> dict:
    """分年冻结-后验：band 已冻结，逐年独立跑，互不重叠、互不预热到未来。

    每年的窗口是 [当年 1/1, min(当年 12/31, dev 末)]，预热只往前取**上一年最后
    一个月末**的信号 —— 那是段前已知数据，不是前视。
    """
    rows, skipped = [], []
    for y in years:
        start, end = f"{y}-01-01", f"{y}-12-31"
        if start > dev_end:
            skipped.append({"period": y, "reason": "整年晚于 dev 末，不属于验证范围"})
            continue
        end = min(end, dev_end)
        try:
            i0, i1, i_meas = _seg_window(fd, start, end)
        except RuntimeError as e:
            skipped.append({"period": y, "reason": f"数据里没有该年交易日：{e}"})
            continue
        if i1 - i_meas < 2:
            skipped.append({"period": y, "reason": "该年交易日不足 2 天"})
            continue
        seg = run_segment(fd, cache, i0, i1, sig_all, system, i_meas=i_meas)
        s = seg.get("strategy") or {}
        b = seg.get("bench") or {}
        sm, bm = s.get("mdd"), b.get("mdd")
        worse = (_r(abs(float(sm)) - abs(float(bm))) if sm is not None and bm is not None
                 else None)
        rows.append({
            "period": y, "start": seg["start"], "end": seg["end"],
            "warmup_start": seg["warmup_start"],
            "n_days": seg["n_days"], "n_rebalance": seg["n_rebalance"],
            "strategy_ann": s.get("ann_return"), "pool_ann": b.get("ann_return"),
            "excess": seg.get("excess"),
            "strategy_mdd": sm, "pool_mdd": bm,
            "dd_worse_pp": worse,
            "dd_ok": (None if worse is None else bool(worse <= abs(max_dd_worse) + 1e-12)),
            "avg_cash_ratio": s.get("avg_cash_ratio"),
        })
    pos = [r for r in rows if r["excess"] is not None and float(r["excess"]) > 0]
    worst = [r for r in rows if r["dd_worse_pp"] is not None]
    if not rows:
        raise RuntimeError(f"分年验证一期都没跑成（配置 {years}，数据内没有对得上的年份）："
                           f"{skipped}")
    return {
        "method": "yearly_freeze",
        "method_note": "band 在定档段冻结后逐年独立验证；每年预热只取上一年最后一个"
                       "月末信号，指标从当年首个交易日起算。",
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_years": len(rows),
            "n_years_positive": len(pos),
            "years_positive": [r["period"] for r in pos],
            "all_dd_ok": bool(rows) and all(r["dd_ok"] for r in rows),
            "worst_dd_worse_pp": (max(worst, key=lambda r: r["dd_worse_pp"])["dd_worse_pp"]
                                  if worst else None),
        },
    }


def weight_sensitivity(fd, i0: int, i1: int, system: dict,
                       mats: dict[str, np.ndarray], amount: np.ndarray,
                       band: float, i_meas: int = 0) -> list[dict]:
    """权重 0.5 / 0.6 / 0.7 三档：月频 IC + 同持仓形态的组合年化（业主：不许只报一档）。

    IC 用 `ic_periods` 的窗口纪律（跨出段末的期整期丢弃），分母是**可投池**。
    组合一栏沿用冻结后的 band —— 换权重不换持仓形态，才是同一件事的对照。
    """
    usable, _ = ic_periods(fd, i0, i1)
    fwd = forward_returns(fd.panel.open_hfq, month_end_indices(fd.dates))
    out = []
    for w in SENSITIVITY_WEIGHTS:
        cache_w = score_cross_sections(fd, system, mats, amount, w_bp=w)
        ics = [rank_ic(cache_w[p["idx"]]["score"], fwd[p["k"]][cache_w[p["idx"]]["idx"]])
               for p in usable
               if p["idx"] in cache_w and cache_w[p["idx"]]["n_candidates"] >= 30]
        s = _ic_stats(ics)
        sig = signals_from_cache(fd, cache_w, band)
        seg = run_segment(fd, cache_w, i0, i1, sig, system, i_meas=i_meas)
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


def evaluate_gates(dev: dict, yearly: dict, holdout: dict | None,
                   system: dict, *, forced: bool = False,
                   rolling: dict | None = None) -> dict:
    """004C准入闸：forced、组合绝对回撤、滚动稳定性，统一 fallback。"""
    g = system.get("gate") or {}
    if rolling is None and (system.get("validation") or {}).get("method") == "yearly_freeze":
        dev_ex = dev.get("excess")
        thr = float(g.get("dev_min_excess", 0.0))
        dev_pass = dev_ex is not None and float(dev_ex) > thr
        rows = list(yearly.get("rows") or [])
        dd_limit = float(g.get("period_max_dd_worse", 0.10))
        period_rows = [{"period": r["period"], "dd_ok": r.get("dd_ok"),
                        "dd_worse_pp": r.get("dd_worse_pp"), "excess": r.get("excess"),
                        "strategy_mdd": r.get("strategy_mdd"),
                        "pool_mdd": r.get("pool_mdd")}
                       for r in rows]
        dd_fail = [r["period"] for r in period_rows if r["dd_ok"] is False]
        dd_pass = bool(period_rows) and not dd_fail
        ho_ex = holdout.get("excess") if holdout else None
        direction = None if not dev_pass or ho_ex is None else (
            float(dev_ex) == 0 or float(ho_ex) == 0 or
            ((float(dev_ex) > 0) == (float(ho_ex) > 0)))
        return {"dev": {"pass": bool(dev_pass), "excess": dev_ex, "threshold": thr,
                        "strategy_ann": (dev.get("strategy") or {}).get("ann_return"),
                        "bench_ann": (dev.get("bench") or {}).get("ann_return")},
                "periods": {"pass": dd_pass, "failed_periods": dd_fail,
                            "rows": period_rows, "max_dd_worse": dd_limit},
                "holdout": {"excess": ho_ex, "direction_consistent": direction,
                            "strategy_ann": (holdout.get("strategy") or {}).get("ann_return") if holdout else None,
                            "bench_ann": (holdout.get("bench") or {}).get("ann_return") if holdout else None,
                            "strategy_mdd": (holdout.get("strategy") or {}).get("mdd") if holdout else None,
                            "bench_mdd": (holdout.get("bench") or {}).get("mdd") if holdout else None,
                            "checked": bool(dev_pass)}, 
                "both_pass": bool(dev_pass and dd_pass and direction is not False)}
    if rolling is None:
        # 005R历史产物回归只允许拒绝，不把旧口径误读成004C证据。
        return {"forced": {"pass": not forced, "reason": "forced" if forced else None},
                "drawdown": {"pass": False, "mdd": None, "threshold": float(g.get("max_portfolio_mdd", 0.15)),
                             "reason": "missing_rolling_evidence"},
                "stability": {"pass": False, "mean_excess": None,
                              "positive_windows": 0, "window_count": 0,
                              "reason": "missing_rolling_evidence"},
                "fallback_reason": "forced" if forced else "stability",
                "both_pass": False}

    rows = rolling.get("windows") or []
    mean_excess = rolling.get("mean_excess")
    positive = int(rolling.get("positive_windows", 0))
    count = int(rolling.get("window_count", len(rows)))
    mdd = rolling.get("portfolio_mdd")
    dd_limit = float(g["max_portfolio_mdd"])
    mean_limit = float(g["min_avg_excess"])
    positive_limit = int(g["min_positive_windows"])
    forced_pass = not forced
    drawdown_pass = mdd is not None and abs(float(mdd)) <= dd_limit + 1e-12
    stability_pass = (mean_excess is not None and float(mean_excess) >= mean_limit
                      and count == 5 and positive >= positive_limit)
    reason = ("forced" if not forced_pass else
              "drawdown" if not drawdown_pass else
              "stability" if not stability_pass else None)
    rolling["fallback_reason"] = reason
    return {"forced": {"pass": forced_pass, "reason": "forced" if forced else None},
            "drawdown": {"pass": bool(drawdown_pass), "mdd": mdd,
                          "threshold": dd_limit, "scope": "rolling_full_period"},
            "stability": {"pass": bool(stability_pass), "mean_excess": mean_excess,
            "mean_threshold": mean_limit, "positive_windows": positive,
            "positive_threshold": positive_limit, "window_count": count,
            "rule": "平均超额≥2pp 且 5 窗口中至少4个为正"},
            "dev": {"strategy_ann": None, "bench_ann": None, "excess": None,
                    "pass": False, "threshold": 0.0},
            "periods": {"pass": False, "rows": [], "failed_periods": [],
                        "max_dd_worse": None},
            "holdout": {"strategy_ann": None, "bench_ann": None, "excess": None,
                        "strategy_mdd": None, "bench_mdd": None,
                        "direction_consistent": None, "checked": False},
            "fallback_reason": reason,
            "both_pass": bool(forced_pass and drawdown_pass and stability_pass)}


# ---------------------------------------------------------------- 建议卡
def _reason_text(sb: float, sr: float, sc: float, rank: int, n_cand: int,
                 industry: str | None, band: float) -> str:
    """入选理由一句话：全部由数字生成，不写"看好""优质"这种没有出处的词。"""
    cheap = f"bp 分位 {sb:.3f}（比池内 {sb * 100:.1f}% 的票便宜）"
    # 反转分位 0.855 的物理含义是"跌幅处在池内最深 14.5%"，不是"前 85.5%"。
    rev = f"20 日反转分位 {sr:.3f}（近月跌幅居池内最深 {(1 - sr) * 100:.1f}%）"
    tail = f"综合 {sc:.3f} 列第 {rank}/{n_cand}，落在前 {band:.0%} 分位带内，等权纳入"
    ind = f"，{industry}" if industry else ""
    return f"{cheap}，{rev}{ind}；{tail}"


def _risk_text(*, roe, debt, debt_pct, amt, mom, weight: float) -> str:
    """单票风险提示：挑最突出的两条 + 一条永远存在的仓位风险。"""
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
    picked.append(f"等权 {weight:.2%}，单票极端全损 −50% 即组合 −{weight * 0.5:.2%}")
    return "；".join(picked)


def budget_plan(rows: list[dict], budget: float, lot_size: int) -> dict:
    """按"预算 ÷ 成分股数"给每只票的金额，并算清"买不买得起一手"。

    口径（业主拍板二选一，本单选**给最小可行预算 + 建议攒钱后启动**）：
      · 每只金额 = 预算 / k，整手向下取整（A 股一手 100 股，不足一手买不了）；
      · `min_viable_budget` = Σ 每只一手成本 = 让这张卡**按等权形态成立**的最低预算；
      · 预算低于它时，如实写"预算不足"，并给出"最多能覆盖几只"备选 —— 但那已经不是
        等权组合，而是挑贵的跳过，性质变了，卡上明说。
    """
    k = len(rows)
    if k == 0:
        return {"available": False, "reason": "名单为空", "n": 0}
    per = float(budget) / k
    items, total_cost, min_viable, max_lot = [], 0.0, 0.0, 0.0
    for r in rows:
        close = r.get("close")
        lc = (float(close) * lot_size) if close is not None and np.isfinite(close) else None
        lots = int(per // lc) if lc and lc > 0 else 0
        amount = float(lots) * lc if lc else 0.0
        if lc:
            min_viable += lc
            max_lot = max(max_lot, lc)
            total_cost += amount
        items.append({"code": r["code"], "per_stock": _r(per, 2),
                      "lot_cost": _r(lc, 2), "lots": lots,
                      "amount": _r(amount, 2), "affordable": bool(lots >= 1)})
    n_aff = sum(1 for x in items if x["affordable"])
    if n_aff == k:
        text = (f"预算 {budget:,.0f} 元 / {k} 只 = 每只 {per:,.0f} 元，全部够一手；"
                f"按整手买入实际投入 {total_cost:,.0f} 元，余 {budget - total_cost:,.0f} 元。")
        mode = "ok"
    else:
        text = (f"预算不足：{budget:,.0f} 元 / {k} 只 = 每只 {per:,.0f} 元，"
                f"{k - n_aff} 只买不到一手（最贵一手 {max_lot:,.0f} 元）。"
                f"按当前名单建立等权组合的最小可行预算 = {min_viable:,.0f} 元"
                f"（每只各买一手）；**建议攒到该金额后启动**。"
                f"若现在就要动手，只能覆盖 {n_aff} 只，那是跳着买、不是本卡的等权组合。")
        mode = "budget_short"
    return {"available": True, "mode": mode, "n": k, "budget": _r(budget, 2),
            "per_stock": _r(per, 2), "n_affordable": n_aff,
            "min_viable_budget": _r(min_viable, 2),
            "suggested_budget": _r(float(np.ceil(min_viable / 1000.0) * 1000), 2),
            "max_lot_cost": _r(max_lot, 2), "est_invested": _r(total_cost, 2),
            "text": text, "items": items}


def build_card(fd, i: int, prev_i: int | None, cache: dict[int, dict], system: dict,
               owner: dict, mats: dict[str, np.ndarray], amount: np.ndarray,
               names: dict[str, str], ctx: dict | None = None,
               band: float | None = None) -> dict:
    """用第 i 个截面生成建议卡正文（分位带全部成分股，含得分构成、资金分配、换仓清单）。

    `band` **必须**由调用方传入本次冻结的档位。第一版这里读的是
    `system.portfolio.band`（config 里的默认 0.05），结果卡出的是 top5% 的 130 只、
    回测跑的是 top30% 的 495 只 —— 卡和回测不是同一个组合。这种"卡面与回测不一致"
    比数字难看严重得多，所以改成显式参数 + 与 `n_candidates` 对账（见下）。
    """
    cs = cache[i]
    if band is None:
        band = float((system.get("portfolio") or {}).get("band", 0.05))
    lot = int((system.get("backtest") or {}).get("lot_size", 100))
    k, sel = select_top_k(cs, band)
    if cs["n_candidates"] and k != int(np.ceil(band * cs["n_candidates"])):
        raise RuntimeError(f"卡与口径不一致：band={band} × 候选 {cs['n_candidates']} "
                           f"应得 k={int(np.ceil(band * cs['n_candidates']))}，实际 {k}")
    w_each = 1.0 / k if k else 0.0
    steps = cs["steps"]
    head = {
        "cross_section": cs["date"],
        "entry_date": (fd.dates[i + 1] if i + 1 < len(fd.dates) else None),
        "n_candidates": int(cs["n_candidates"]),
        "portfolio": {
            "mode": "percentile_band_equal_weight",
            "band": band, "k": int(k), "weight_each": _r(w_each, 6),
            "rule": f"综合分前 {band:.0%}（{k} 只）等权，每只 {w_each:.2%}",
            "k_note": ("候选池不足 2 只，只能 1 只 —— 已不是分散组合，请看 n_candidates"
                       if k <= 1 else None),
        },
        "weights": {"bp": cs["weight_bp"], "reversal": cs["weight_reversal"]},
        "filters": steps,
        "vetoed_total": steps["total_vetoed"],
        "veto_note": f"{steps['total_vetoed']} 只被质量否决"
                     f"（亏损 {steps['vetoed_roe_negative']} + 高杠杆 "
                     f"{steps['vetoed_high_debt']}）",
    }
    if cs["n_candidates"] == 0:
        return {**head, "stocks": [], "budget": {"available": False, "reason": "池为空"},
                "rebalance": {"available": False, "reason": "池为空"},
                "reason": "该截面可选池为空（过滤/否决后无票），不凑数"}

    sw, inames = load_industry_labels(1)
    codes_sel = [fd.codes[int(j)] for j in cs["idx"][sel]]
    imap = industry_map_at(sw, inames, cs["date"], codes_sel)

    stocks = []
    for r, p in enumerate(sel):
        j = int(cs["idx"][p])
        sb = float(cs["bp_pct"][p])
        sr = float(cs["rev_pct"][p])
        sc = float(cs["score"][p])
        rank = r + 1                                   # sel 已按分数降序
        contrib_bp = cs["weight_bp"] * sb
        contrib_rev = cs["weight_reversal"] * sr
        code = fd.codes[j]
        industry = imap.get(code)
        roe = float(fd.roe_ttm[i][j]) if fd.roe_ttm is not None else None
        debt = float(fd.debt_ratio[i][j]) if fd.debt_ratio is not None else None
        dpct = cs["debt_pct"][j]
        amt = float(amount[i][j])
        mom = float(cs["mom_raw"][p])
        close = float(fd.panel.close_raw[i][j])
        stocks.append({
            "code": code,
            "name": names.get(code),
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
            "rank_in_universe": rank,
            "score_pct": _r(1.0 - (rank - 1) / max(1, cs["n_candidates"]), 4),
            "position_in_band": f"{rank}/{k}",
            # 权重存 10 位：777 只时 6 位小数的舍入误差累加会让"权重和=100%"对不上账
            "weight": {"point": _r(w_each, 10), "text": f"{w_each:.2%}"},
            "reason": _reason_text(sb, sr, sc, rank, cs["n_candidates"], industry, band),
            "risk": _risk_text(roe=roe, debt=debt, debt_pct=dpct, amt=amt, mom=mom,
                               weight=w_each),
            "diagnostics": {
                "close": _r(close, 4),
                "lot_cost": _r(close * lot, 2),
                "amount_yi": _r(amt / 1e8, 4),
                "roe_ttm": _r(roe, 4),
                "debt_ratio": _r(debt, 4),
                "debt_ratio_industry_pct": _r(dpct, 4),
            },
        })

    budget = budget_plan([{"code": s["code"], "close": s["diagnostics"]["close"]}
                          for s in stocks],
                         float(owner.get("budget") or 0.0), lot)
    rebal = _rebalance_block(fd, i, prev_i, cache, band, names)

    # 风险提示里的只数/档位必须用**这张卡自己的** k（逐截面浮动），
    # 不能用 dev 的 k 中位数 —— 卡上两个数字打架比不写还糟（第一版踩过）。
    ctx = dict(ctx or {})
    ctx.update({"band": band, "k": k})
    return {**head, "stocks": stocks, "budget": budget, "rebalance": rebal,
            "risk_note": _card_risk_note(ctx, rebal), "reason": None}


def _rebalance_block(fd, i: int, prev_i: int | None, cache: dict[int, dict],
                     band: float, names: dict[str, str]) -> dict:
    """月度换仓清单：本期买入 / 卖出 / 继续持有（对照上一期同档位名单）。"""
    cs = cache[i]
    k, sel = select_top_k(cs, band)
    cur = [fd.codes[int(j)] for j in cs["idx"][sel]]
    cur_set = set(cur)
    if prev_i is None or prev_i not in cache:
        return {"available": False, "reason": "上一期截面缺失（数据起点附近）",
                "current": len(cur), "hold": [], "buy": sorted(cur_set),
                "sell": [], "counts": {"hold": 0, "buy": len(cur_set), "sell": 0},
                "turnover": None}
    pcs = cache[prev_i]
    pk, psel = select_top_k(pcs, band)
    prev = [fd.codes[int(j)] for j in pcs["idx"][psel]]
    prev_set = set(prev)
    hold = [c for c in cur if c in prev_set]           # 保持本期顺序
    buy = [c for c in cur if c not in prev_set]
    sell = [c for c in prev if c not in cur_set]
    return {
        "available": True,
        "prev_cross_section": pcs["date"], "prev_k": int(pk),
        "current": len(cur), "prev": len(prev),
        "hold": hold, "buy": buy, "sell": sell,
        "hold_names": {c: names.get(c) for c in hold},
        "buy_names": {c: names.get(c) for c in buy},
        "sell_names": {c: names.get(c) for c in sell},
        "counts": {"hold": len(hold), "buy": len(buy), "sell": len(sell)},
        "turnover": _r((len(buy) + len(sell)) / max(1, 2 * len(prev))),
        "turnover_note": "换手率 = (买入数 + 卖出数) / (2 × 上期只数)，月度双边口径",
        "action": "卖出清仓 → 买入新建 → 持有的不动（等权，不做加仓减仓）",
    }


def _card_risk_note(ctx: dict | None, rebal: dict | None = None) -> list[str]:
    """策略层风险提示（与单票风险分开写：前者是"这套机制"的，后者是"这只票"的）。"""
    if not ctx:
        return ["回测数据未注入，本卡不包含组合层风险数字。"]
    turnover = (rebal or {}).get("turnover")
    return [
        f"持仓形态：综合分前 {ctx.get('band', 0):.0%} 等权，"
        f"{ctx.get('k', 0)} 只 —— 收益接近基准，靠**偏离基准的选股打分**赚超额，"
        f"不是靠集中押注。",
        f"dev 段最大回撤 {_f(ctx.get('dev_mdd'), '.2%')}，同期可投池等权 "
        f"{_f(ctx.get('dev_pool_mdd'), '.2%')}；业主容忍回撤 "
        f"{_f(ctx.get('owner_mdd'), '.0%')}。",
        f"换手：每月按名单进出，本期换手率 {_f(turnover, '.0%')}"
        f"（无上期名单时 n/a），成本已按佣金 0.025%/印花税 0.05%/滑点 0.1% 计入回测。",
        "本卡不做止损、不做仓位择时：月末出名单，下一个交易日开盘执行，中间不动。",
    ]


# ---------------------------------------------------------------- 主流程
def run(cfg: dict, *, start: str = DEFAULT_START, verbose: bool = True) -> dict:
    """跑完整流程：载数据 → analyze()。载数据单独一步，便于测试注入合成面板。"""
    system = C.system_params(cfg)
    holdout_end = (system.get("holdout") or {}).get("end") or datetime.now().strftime("%Y-%m-%d")
    if verbose:
        print(f"[1/7] 载入数据 {start} ~ {holdout_end} …", flush=True)
    fd = load_factor_data(start, holdout_end)
    return analyze(fd, cfg, verbose=verbose)


def analyze(fd, cfg: dict, *, verbose: bool = False) -> dict:
    """在已载入的 FactorData 上跑完整判定链（不碰 IO，测试可直接注入）。

    顺序就是纪律：
        定档段扫描（只用 2018~2021）→ 冻结 band → dev 全段 → 分年验证
        → holdout 一次性 → 门槛 → 出卡或 fallback
    band 一旦冻结，后面任何一步的结果都不许回头改它。
    """
    system = C.system_params(cfg)
    if system["validation"].get("method") == "rolling_24_12":
        return analyze_004c(fd, cfg, verbose=verbose)
    owner = C.owner_profile(cfg)
    port = system.get("portfolio") or {}
    val = system.get("validation") or {}
    bands = [float(x) for x in (port.get("band_candidates") or DEFAULT_BANDS)]
    dd_lim = float((system.get("gate") or {}).get("period_max_dd_worse", 0.10))

    mats = factor_mats(fd)
    amount = trade_amount(fd)
    names = load_names()
    dates = fd.dates
    dev_end = (system.get("holdout") or {}).get("start")
    if not dev_end:
        raise RuntimeError("config 缺 system.holdout.start，无法切分 dev/holdout")
    dev_stop = [i for i, d in enumerate(dates) if d < dev_end]
    ho_start = [i for i, d in enumerate(dates) if d >= dev_end]
    if not dev_stop or not ho_start:
        raise RuntimeError(f"数据不足以切分 dev/holdout（dev 末 {dev_stop[-1:]}, "
                           f"holdout 起 {ho_start[:1]}）")
    i_ho0 = ho_start[0]
    dev_last = dates[dev_stop[-1]]
    if verbose:
        print(f"      面板 {len(dates)} 日 × {len(fd.codes)} 只；"
              f"dev 至 {dev_last}；holdout 自 {dates[i_ho0]}", flush=True)

    if verbose:
        print("[2/7] 全部月末截面打分（只算一次，各档位共用）…", flush=True)
    cache = score_cross_sections(fd, system, mats, amount)
    ns = [cache[i]["n_candidates"] for i in sorted(cache)]
    if verbose:
        print(f"      {len(cache)} 个截面，候选数中位 {int(np.median(ns))} "
              f"（{min(ns)}~{max(ns)}）", flush=True)

    # ---------- 定档（只许看定档段）
    fit_end = val.get("band_fit_end") or "2021-12-31"
    i_fit0, i_fit1, i_fit_meas = _seg_window(fd, dates[0], fit_end)
    n_fit_xs = sum(1 for i in cache if i_fit0 <= i < i_fit1)
    if n_fit_xs < 2:
        raise RuntimeError(f"定档段 {dates[0]}~{fit_end} 只有 {n_fit_xs} 个截面，"
                           f"无法定档（至少要 2 个）")
    if verbose:
        print(f"[3/7] 定档扫描（{dates[i_fit_meas]} ~ {dates[i_fit1 - 1]}，"
              f"{n_fit_xs} 个截面，{len(bands)} 档）…", flush=True)
    band_rows = scan_bands(fd, system, mats, amount, i_fit0, i_fit1, i_fit_meas, bands)
    pick = select_band(band_rows, dd_lim)
    band = float(pick["selected"])
    k_by_xs = [select_top_k(cache[i], band)[0] for i in sorted(cache) if cache[i]["n_candidates"]]
    k_med = int(np.median(k_by_xs))
    if verbose:
        print(f"      选中 top {band:.0%}（中位 {k_med} 只，"
              f"{'无档合格、强制取最优' if pick['forced'] else '合格档 ' + str(pick['qualified_bands'])}）",
              flush=True)

    sig_all = signals_from_cache(fd, cache, band)

    if verbose:
        print("[4/7] dev 全段回测 + 分年验证 …", flush=True)
    i_dev1 = dev_stop[-1] + 1
    dev = run_segment(fd, cache, 0, i_dev1, sig_all, system, i_meas=0)
    years = [str(y) for y in (val.get("years") or ["2022", "2023", "2024", "2025"])]
    # 旁证：分数分位 80~90 那一段。**只在定档之后看**，且只用于对照 —— 它不是本单
    # 的持仓形态（本单是 top N%），也不参与任何选择。业主工单要求必须标注这一条。
    side = _side_reference(fd, cache, 0, i_dev1, system)
    yearly = yearly_validation(fd, cache, sig_all, system, years, dev_last, dd_lim)
    sens = weight_sensitivity(fd, 0, i_dev1, system, mats, amount, band, i_meas=0)

    if verbose:
        print("[5/7] holdout 一次性回测（结果出来后不回头改参数）…", flush=True)
    i_ho0, i_ho1, i_ho_meas = _seg_window(fd, dates[i_ho0], dates[-1])
    holdout = run_segment(fd, cache, i_ho0, i_ho1, sig_all, system, i_meas=i_ho_meas)
    gates = evaluate_gates(dev, yearly, holdout, system)

    verdict = "advise" if gates["both_pass"] else "fallback"
    if verbose:
        print(f"[6/7] 门槛：dev {'过' if gates['dev']['pass'] else '不过'} / "
              f"各期回撤 {'过' if gates['periods']['pass'] else '不过'}"
              f"{'' if not gates['periods']['failed_periods'] else '（' + '/'.join(gates['periods']['failed_periods']) + '）'}"
              f" / 方向 {gates['holdout']['direction_consistent']} → {verdict}", flush=True)

    # 卡：达标出正式卡；不达标也出，但只放 preview 并标注"不构成建议"
    ends = sorted(cache)
    last_i = None
    for i in reversed(ends):
        if i + 1 < len(dates):
            last_i = i
            break
    prev_i = None
    if last_i is not None:
        prevs = [i for i in ends if i < last_i]
        prev_i = prevs[-1] if prevs else None
    ctx = {"band": band, "k": k_med, "dev_mdd": (dev.get("strategy") or {}).get("mdd"),
           "dev_pool_mdd": (dev.get("bench") or {}).get("mdd"),
           "owner_mdd": owner.get("max_drawdown")}
    preview = (build_card(fd, last_i, prev_i, cache, system, owner, mats, amount,
                          names, ctx=ctx, band=band) if last_i is not None else None)
    if preview is not None:
        # 先打标再分派：反过来写会让 card 指向打标前的旧对象，卡上少了
        # is_recommendation（005 测试抓到过）。
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
                f"dev 段组合年化 {_f(gates['dev']['strategy_ann'], '.2%')} 低于可投池等权 "
                f"{_f(gates['dev']['bench_ann'], '.2%')}（超额 "
                f"{_f(gates['dev']['excess'], '+.2%')}，门槛要求 > "
                f"{gates['dev']['threshold']:.0%}）")
        if not gates["periods"]["pass"]:
            for p in gates["periods"]["rows"]:
                if p["dd_ok"] is False:
                    reasons.append(
                        f"{p['period']} 段回撤 {_f(p['strategy_mdd'], '.2%')} 深于池基准 "
                        f"{_f(p['pool_mdd'], '.2%')} 超过 {dd_lim:.0%}"
                        f"（深 {_f(p['dd_worse_pp'], '.1%')}）")
        if gates["holdout"]["direction_consistent"] is False:
            reasons.append(f"holdout 超额 {_f(gates['holdout']['excess'], '+.2%')} 与 dev "
                           f"方向相反（{gates['holdout']['direction_note']}）")
        fallback = {
            "message": "本次不建议",
            "fallback_reason": ("forced" if pick["forced"] else
                               "drawdown" if not gates["periods"]["pass"] else
                               "stability"),
            "band": band,
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
                "yearly": [{"period": r["period"], "excess": r["excess"],
                            "dd_worse_pp": r["dd_worse_pp"]}
                           for r in yearly["rows"]],
            },
            "what_would_change_it": "dev 对可投池超额 > 0 且各期回撤不深于基准 10pp 才会出卡；"
                                    "要换口径请改 config 并在定档段重新定档，"
                                    "不许用 holdout 或分年结果反推 band。",
        }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        **_git_meta(),
        "verdict": verdict,
        "portfolio_mode": "percentile_band_equal_weight",
        "band_selection": {
            "fit_start": dates[i_fit_meas], "fit_end": dates[i_fit1 - 1],
            "n_cross_sections": n_fit_xs,
            "candidates": bands,
            "selected": band, "forced": pick["forced"],
            "qualified_bands": pick["qualified_bands"],
            "positive_bands": pick["positive_bands"],
            "tightest_positive_band": pick["tightest_positive_band"],
            "best_excess": pick["best_excess"],
            "rule": pick["rule"],
            "table": band_rows,
            "neighbors": band_neighbors(band, bands),
            "k_median_all": k_med, "k_min": min(k_by_xs), "k_max": max(k_by_xs),
            "expected_positions": (port.get("expected_positions") or [30, 200]),
            "positions_note": _positions_note(k_med, port.get("expected_positions")),
            "convergence_note": _convergence_note(pick["tightest_positive_band"]),
        },
        "gates": gates,
        "config": {"owner": owner,
                   "system": {kk: v for kk, v in system.items() if kk != "holdout"},
                   "holdout": system.get("holdout")},
        "segments": {"dev": dev, "holdout": holdout},
        "validation": yearly,
        "side_reference": side,
        "weight_sensitivity": sens,
        "card": card,
        "preview_card": preview,
        "fallback": fallback,
        "data": {
            "start": dates[0], "end": dates[-1], "dates": len(dates),
            "codes": len(fd.codes),
            "dev_end": dev_last, "holdout_start": dates[i_ho0],
            "shares_coverage": (fd.source.get("shares") or {}).get("codes_coverage"),
            "industry_groups": (fd.source.get("industry") or {}).get("groups"),
            "fundamentals": {kk: (fd.source.get("fundamentals") or {}).get(kk)
                             for kk in ("rows_usable", "pubdate_missing_rate",
                                        "epsTTM_coverage")},
        },
        "signal_build": {
            "n_rebalances": len(sig_all),
            "k_median": k_med, "k_min": min(k_by_xs), "k_max": max(k_by_xs),
            "note": "逐截面 k = ceil(band × 候选数)，等权 1/k；候选数不足 2 只的截面会退化为 1 只",
        },
        "validity_note": build_validity_note(verdict, gates, yearly, sens, band,
                                             band_pick=pick, k_med=k_med),
    }
    if verbose:
        print("[7/7] 完成", flush=True)
    return report


def build_validity_note(verdict: str, gates: dict, yearly: dict,
                        sens: list[dict], band: float,
                        band_pick: dict | None = None, k_med: int | None = None) -> list[str]:
    """免责与口径纪律。数字全部来自本次产物，不写死。"""
    notes = [
        "本卡是**决策辅助**，不构成任何买卖指令；何时买、买多少、何时卖由业主自行决定。",
        "月频 IC 存在自相关，t 值高估显著性，只作量级参考，**不作 alpha 证据**。",
        f"持仓形态为「综合分前 {band:.0%} 等权」。band 由定档段扫描确定，"
        f"holdout 与分年结果**未参与**选档。",
        "门槛基准是**可投池等权**（过滤+否决后的池子，月频等权持有、无成本），"
        "与策略同调仓频率；全市场等权与日频再平衡版本只作对照，不进门槛。",
        "样本切分：dev 至 holdout.start 前一日，holdout 只跑一次；分年验证逐年独立，"
        "每年指标从当年首个交易日起算（预热只用上一年最后一个月末信号）。",
        "成交额用 close×volume 近似（VWAP 口径差未计），仅用于流动性下限，不参与打分。",
        "回测按一手（100 股）取整，现金拖累已在各段 `avg_cash_ratio` 里如实给出。",
    ]
    if band_pick and band_pick.get("forced"):
        notes.append(
            "**选档证据不足**：定档段没有任何一档跑赢可投池等权"
            f"（最好一档超额 {_f(band_pick.get('best_excess'), '+.2%')}），"
            "当前 band 是在回撤合格的档位里取「最接近」的一档，属 forced 选档。"
            "看这张卡时请把 dev / 分年的超额当成独立证据，不要当成「定档段已证明」。")
    if k_med is not None and not (30 <= k_med <= 200):
        notes.append(f"中位持仓 {k_med} 只，超出业主预期的 30~200 只区间；"
                     f"band 与只数是同一件事的两面，缩到 30~200 只就必须用更窄的档，"
                     f"而更窄的档在定档段是负超额 —— 两者不可兼得，需业主拍板取舍。")
    w06 = next((x for x in sens if abs(x.get("weight_bp", 0) - 0.6) < 1e-9), None)
    if w06:
        notes.append(
            f"权重 {w06['weight_bp']}/{w06['weight_reversal']} 的 dev 月频 IC "
            f"{_f(w06['ic_mean'], '+.4f')}（t {_f(w06['t_stat'], '+.2f')}，"
            f"{w06['n_periods']} 期），同持仓形态组合年化 "
            f"{_f(w06['portfolio_ann'], '+.2%')}、对池超额 "
            f"{_f(w06['portfolio_excess'], '+.2%')}。")
    sm = (yearly.get("summary") or {})
    exs = [r["excess"] for r in (yearly.get("rows") or []) if r.get("excess") is not None]
    if sm.get("n_years"):
        notes.append(
            f"分年验证 {sm['n_years']} 期里 {sm['n_years_positive']} 期跑赢可投池等权"
            f"（{('、'.join(sm.get('years_positive')) or '无')}）；"
            f"最差回撤劣化 {_f(sm.get('worst_dd_worse_pp'), '+.1%')}。"
            "跑输的年份同样是要看的信息，不是噪声。")
    if len(exs) >= 2:
        spread = max(exs) - min(exs)
        head = (f"分年超额最高 {max(exs):+.2%}、最低 {min(exs):+.2%}，极差 {spread:.2%}。")
        if spread >= 0.10:
            notes.append(
                "**稳定性不足**：" + head +
                "业主的预期是年化稳定跑赢 2~3pp，本期实测是「有的年份大幅跑赢、"
                "有的年份大幅跑输」—— 稳定二字不成立。"
                "这一条未进门槛（业主未定），但必须让业主看到。")
        else:
            notes.append(head)
    if verdict == "fallback":
        notes.append("本次输出为「不建议」：硬门槛未过，卡只在 preview_card 里作结构核对，"
                     "不得当作建议使用。")
    return notes


def print_summary(rep: dict) -> None:
    print("\n" + "=" * 82)
    print(f"建议卡结论：{rep['verdict']}    schema {rep['schema_version']}")
    bs = rep["band_selection"]
    print(f"  定档段 {bs['fit_start']} ~ {bs['fit_end']}（{bs['n_cross_sections']} 个截面）"
          f" → 选中 top {bs['selected']:.0%}，中位 {bs['k_median_all']} 只")
    print(f"  选档依据：{bs['rule']}")
    if bs.get("positions_note"):
        print(f"  【提示】{bs['positions_note']}")
    nb = bs["neighbors"]
    prev_txt = "n/a" if nb["prev"] is None else f"{nb['prev']:.0%}"
    next_txt = "n/a" if nb["next"] is None else f"{nb['next']:.0%}"
    print(f"  ±1 档：上一档 {prev_txt} / 选中 {nb['selected']:.0%} / 下一档 {next_txt}")

    print("\n  档位对比（定档段，基准=可投池等权）：")
    print(f"  {'band':>6}{'只数中位':>9}{'年化':>9}{'池年化':>9}{'超额':>9}"
          f"{'回撤':>9}{'池回撤':>9}{'现金占比':>10}")
    for r in bs["table"]:
        mark = " ←" if abs(r["band"] - bs["selected"]) < 1e-12 else ""
        print(f"  {r['band']:>6.0%}{r['k_median']:>9}{_f(r['strategy_ann'], '+.2%'):>9}"
              f"{_f(r['pool_ann'], '+.2%'):>9}{_f(r['excess'], '+.2%'):>9}"
              f"{_f(r['strategy_mdd'], '.2%'):>9}{_f(r['pool_mdd'], '.2%'):>9}"
              f"{_f(r['avg_cash_ratio'], '.1%'):>10}{mark}")

    g = rep["gates"]
    print(f"\n  门槛① dev：{_f(g['dev']['strategy_ann'], '+.2%')} vs 池 "
          f"{_f(g['dev']['bench_ann'], '+.2%')} → 超额 {_f(g['dev']['excess'], '+.2%')}"
          f"  [{'过' if g['dev']['pass'] else '不过'}]")
    print(f"  门槛② 各期回撤（不得深于池 {g['periods']['max_dd_worse']:.0%}）："
          f"[{'过' if g['periods']['pass'] else '不过'}]"
          + (f"  失败期 {g['periods']['failed_periods']}" if g['periods']['failed_periods'] else ""))

    print("\n  分年验证（band 冻结，逐年独立）：")
    print(f"  {'年份':>6}{'区间':>25}{'年化':>9}{'池年化':>9}{'超额':>9}{'回撤':>9}"
          f"{'池回撤':>9}{'劣化pp':>9}")
    for r in rep["validation"]["rows"]:
        print(f"  {r['period']:>6}{r['start'] + '~' + r['end']:>25}"
              f"{_f(r['strategy_ann'], '+.2%'):>9}{_f(r['pool_ann'], '+.2%'):>9}"
              f"{_f(r['excess'], '+.2%'):>9}{_f(r['strategy_mdd'], '.2%'):>9}"
              f"{_f(r['pool_mdd'], '.2%'):>9}{_f(r['dd_worse_pp'], '+.1%'):>9}")

    h = rep["segments"]["holdout"]
    print(f"\n  holdout {h['start']} ~ {h['end']}：组合 "
          f"{_f((h.get('strategy') or {}).get('ann_return'), '+.2%')} vs 池 "
          f"{_f((h.get('bench') or {}).get('ann_return'), '+.2%')} → 超额 "
          f"{_f(h.get('excess'), '+.2%')}；回撤 "
          f"{_f((h.get('strategy') or {}).get('mdd'), '.2%')} vs "
          f"{_f((h.get('bench') or {}).get('mdd'), '.2%')}")

    sr = rep.get("side_reference") or {}
    if sr:
        print(f"\n  旁证（{sr['lo']:.0%}~{sr['hi']:.0%} 分位带，dev，{sr['k_median']} 只）："
              f"年化 {_f(sr['ann_return'], '+.2%')} / 回撤 {_f(sr['mdd'], '.2%')} / "
              f"对池超额 {_f(sr['excess'], '+.2%')}")
        print(f"        【{sr['label']}】{sr['why'][:58]}…\n"
              f"        不进门槛、不影响 band，理由见产物 side_reference.why")

    print("\n  权重敏感性（dev，持仓形态不变）：")
    print(f"  {'w_bp':>5}{'IC均值':>10}{'t':>8}{'期数':>6}{'年化':>10}{'超额':>9}")
    for x in rep["weight_sensitivity"]:
        print(f"  {x['weight_bp']:>5.1f}{_f(x['ic_mean'], '+.4f'):>10}"
              f"{_f(x['t_stat'], '+.2f'):>8}{x['n_periods']:>6}"
              f"{_f(x['portfolio_ann'], '+.2%'):>10}{_f(x['portfolio_excess'], '+.2%'):>9}")

    crd = rep.get("preview_card") or {}
    st = crd.get("stocks") or []
    if st:
        tag = "建议卡" if crd.get("is_recommendation") else "结构预览（不构成建议）"
        pf = crd["portfolio"]
        print(f"\n  {tag}：截面 {crd['cross_section']}，候选 {crd['n_candidates']} 只，"
              f"取 {pf['k']} 只等权（每只 {pf['weight_each']:.2%}）；{crd['veto_note']}")
        bd = crd.get("budget") or {}
        print(f"  资金：{bd.get('text')}")
        rb = crd.get("rebalance") or {}
        if rb.get("available"):
            c = rb["counts"]
            print(f"  换仓（对照 {rb['prev_cross_section']}，上期 {rb['prev']} 只）："
                  f"卖 {c['sell']} / 买 {c['buy']} / 持 {c['hold']}，"
                  f"换手 {_f(rb['turnover'], '.0%')}")
        print(f"  {'代码':<11}{'名称':<9}{'行业':<7}{'综合':>7}{'名次':>8}"
              f"{'等权':>8}{'成交额亿':>10}")
        for s in st[:10]:
            print(f"  {s['code']:<11}{(s['name'] or '-'):<9}{(s['industry'] or '-'):<7}"
                  f"{_f(s['score'], '.4f'):>7}{s['position_in_band']:>8}"
                  f"{s['weight']['text']:>8}{_f(s['diagnostics']['amount_yi'], '.2f'):>10}")
        if len(st) > 10:
            print(f"  …（全名单 {len(st)} 只，见产物 JSON 的 card.stocks）")
    if rep.get("fallback"):
        print("\n本次不建议：")
        for r in rep["fallback"]["reasons"]:
            print(f"  · {r}")
    key = [n for n in (rep.get("validity_note") or [])
           if n.startswith(("**选档证据不足**", "**稳定性不足**", "中位持仓"))]
    if key:
        print("\n必须看到的风险（不要让上面的数字盖住这几条）：")
        for n in key:
            print(f"  · {n}")
    print("=" * 82)


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
        print(f"模板已写入 {p}\n手改 owner 段（预算/回撤）后直接跑：python -m src.advisor")
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
