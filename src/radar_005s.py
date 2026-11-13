"""005S —— 降级形态：熊市防守雷达（状态灯 + 避雷名单 + 观察名单）。

产品口径（业主 2026-11-13 拍板降级，004C 判 fallback 后）：不做「每月推荐买入」。
三样输出：

  灯   ：H00300 收盘相对过去 20 日最高收盘的回撤 ≤ -10% → 防守期，否则正常期
  观察 ：低波 + 换手稳定 分位前 100 只（防守期才看，正常期不给建议）
  避雷 ：高波 + 换手不稳 + 过去 60 日跌得多 前 100 只（这些别碰）

铁律（写死在代码里，不靠报告自觉）：
  · 灯只用 **当日及以前** 的收盘价；`rolling_drawdown` 里 `rolling().max()` 是
    右对齐窗口，不可能看到未来。
  · 名单在月末 T 收盘出，只在下一个月持有和评估；检验期不参与任何选择。
  · 名单不做加权优化、不做择时微调：分数就是定向因子分位的简单等权。
  · 不达标就报失败，不改口径（调参上限沿用 004C 的 3 轮）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import metrics as M
from src.advisor import ROOT, _git_meta, build_universe, load_names, trade_amount
from src.factors import FACTOR_DEFS, month_end_indices

#: 产物 schema。字段增删要往上加版本，别静默改结构。
#: 005S-2：修 build_universe 用 T+1 涨停砍池的未来函数后重算全部统计，
#: 并新增 monthly_lists 段（逐月名单代码落盘，供审计独立复核）。
#: 005S-3：latest 段补 light_state（当日灯态快照 + 判定日期）与 watch_names
#: （code→简称，来源 data/cache.db），施工B 不再按 JSON 自行重算灯态。
SCHEMA_VERSION = "005S-3"
#: 产物路径（与 005R 的 out/advice_card.json 分开，绝不覆盖旧卡留档）。
OUT_NAME = "out/005s_validation.json"
#: 逐月名单代码明细（005S-R2，供审计逐月复核）
LISTS_NAME = "005s_monthly_lists.parquet"
#: 基准：沪深300全收益（与 004C 同一份，禁双基准）。
BENCH_PATH = "data/raw/index/hs300_total_return_2026-11-09.parquet"
#: 灯的固定参数（验收规格 §二，不许在运行时改）
DD_WINDOW = 20
DD_THRESHOLD = 0.10
#: 两张名单的规模（验收规格 §二：前 100 / worst 100）
TOP_N = 100
#: 达标线：正超额月份占比 ≥ 70%
PASS_RATE = 0.70

#: 观察名单用的因子（已定向：值越大越好），简单等权
WATCH_FACTORS = ("lowvol_20", "lowvol_60", "turn_stab_60")
#: 避雷名单：三条「别碰」特征各自的分位权重，全部换算成「越大越该躲」
AVOID_SPEC = (("lowvol_20", -1.0, 0.4),     # 波动高（lowvol 分位低）
              ("turn_stab_60", -1.0, 0.3),  # 换手不稳
              ("reversal_60", 1.0, 0.3))    # 过去 60 日跌得多

#: 产品文案里必须原样出现的限制（指令 §一）
PRODUCT_NOTE = (
    "这只雷达只在防守期有价值：2021~2023 三个熊市检验年，低波动组合对沪深300全收益"
    "分别跑出 +14.9 / +5.8 / +7.4pp 超额，且回撤年年更浅；2024、2025 反弹行情跑输。"
    "正常期不给建议，只报状态灯。")


# ---------------------------------------------------------------- 灯（S1）
def load_benchmark() -> pd.Series:
    """沪深300全收益收盘序列（date 字符串索引，升序）。"""
    df = pd.read_parquet(ROOT / BENCH_PATH)
    s = df.set_index("date")["close"].astype("float64").sort_index()
    s.index = s.index.astype(str)
    return s


def rolling_drawdown(close: pd.Series, window: int = DD_WINDOW) -> pd.Series:
    """close 相对过去 window 个交易日最高收盘的回撤（≤0）。

    `rolling(window)` 窗口右端就是当日，`max` 只含当日及之前 —— 结构上不可能用未来。
    """
    peak = close.rolling(window, min_periods=1).max()
    return close / peak - 1.0


def state_light(close: pd.Series, window: int = DD_WINDOW,
                threshold: float = DD_THRESHOLD) -> pd.Series:
    """日频灯态：True = 防守期（回撤已到 -threshold 或更深）。

    边界用 1e-12 的容差：90/100-1 在浮点上是 -0.09999999999999998，直接比 -0.1
    会把「正好 -10%」判成不亮 —— 验收线是硬的，不许被浮点噪声放宽。
    """
    dd = rolling_drawdown(close, window)
    return (dd <= -(float(threshold) - 1e-12)).rename("defensive")


def light_snapshot(light: pd.Series, dd: pd.Series, date: str,
                   window: int = DD_WINDOW,
                   threshold: float = DD_THRESHOLD) -> dict:
    """单个交易日的灯态快照（005S-R3：施工B 直接读，不再自行重算）。

    判定日期就是传入的 `date` —— 只用该日及以前的收盘价，`dd` 与 `light`
    必须已经在**同一条**基准序列上算好（右对齐窗口，结构上看不到未来）。

    缺数据时 `state=None` 并在 `note` 里写明，绝不默认成 "normal"：
    把「没算出来」当成「正常期」等于凭空放宽。
    """
    lit = light.get(date)
    d = dd.get(date)
    if lit is None or d is None or not np.isfinite(float(d)):
        return {"state": None, "as_of": str(date), "drawdown": None,
                "window": int(window), "threshold": -float(threshold),
                "note": "该交易日无基准收盘或缺前置窗口，未判定"}
    return {"state": "defensive" if bool(lit) else "normal",
            "as_of": str(date), "drawdown": _r(float(d)),
            "window": int(window), "threshold": -float(threshold),
            "rule": f"H00300 收盘相对过去 {int(window)} 日最高收盘回撤 ≤ "
                    f"-{float(threshold):.0%} → defensive，否则 normal"}


def names_for(codes: list[str], names: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """code → 简称映射；查不到的单独列出，不用「未知」之类占位符顶替。

    名称只用于展示，不参与任何计算；`load_names` 取不到 cache.db 时返回空 dict，
    于是 mapping 为空、missing 为全量 —— 仍是可解释的结果。
    """
    mapping = {c: names[c] for c in codes if names.get(c)}
    missing = [c for c in codes if not names.get(c)]
    return mapping, missing


def latest_block(latest: dict, light: pd.Series, dd: pd.Series,
                 names: dict[str, str]) -> dict:
    """产物 latest 段：名单 + 灯态快照 + 名称映射（005S-R3 字段补全）。

    两个日期**故意不同**，各自都写进产物：
      · `as_of`        —— 名单截面日（月末收盘，出名单那天）
      · `light_state.as_of` —— 基准最新交易日（灯是日频，取最新一根收盘）
    实测数据里基准比面板晚收几天（基准 2026-11-09 / 面板 2026-11-06），
    若强行统一成名单日，界面按 JSON 显示的灯会比它自己重算的旧几天。

    单独抽出来是为了可测：这段是施工B 的唯一读取口。
    """
    list_date = str(latest["date"])
    light_date = str(dd.index[-1]) if len(dd) else list_date
    mapping, missing = names_for(latest["watch_codes"], names)
    return {"as_of": latest["date"], "universe": int(latest["n_universe"]),
            "light_state": light_snapshot(light, dd, light_date),
            "watch": latest["watch_codes"], "watch_names": mapping,
            "watch_names_missing": missing,
            "avoid": latest["avoid_codes"]}


def month_flags(dates: list[str], light: pd.Series) -> dict[str, dict]:
    """把日频灯态归到自然月：**任一交易日**亮 = 该月为防守月（验收 §二口径）。

    同时记下该月最后一个交易日的灯态，供「月初就亮 vs 月末才亮」的观察。
    """
    out: dict[str, dict] = {}
    for d in dates:
        ym = str(d)[:7]
        lit = bool(light.get(str(d), False))
        row = out.setdefault(ym, {"month": ym, "defensive": False,
                                  "n_days": 0, "first_defensive": None,
                                  "month_end_state": False})
        row["n_days"] += 1
        if lit:
            row["defensive"] = True
            if row["first_defensive"] is None:
                row["first_defensive"] = str(d)
        row["month_end_state"] = lit      # 按 dates 升序遍历，最后写入的就是月末
    return out


# ---------------------------------------------------------------- 名单（S2）
def _pct_rank(x: np.ndarray) -> np.ndarray:
    """截面分位（0~1，等距）。与 advisor._pct_rank 同口径，避免跨模块私有耦合。"""
    n = len(x)
    if n == 0:
        return np.empty(0, dtype="float64")
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.arange(n, dtype="float64")
    return ranks / max(n - 1, 1)


def score_lists(mats: dict[str, np.ndarray], i: int, mask: np.ndarray,
                top_n: int = TOP_N) -> dict:
    """在一批候选（mask）里算两张名单，返回**全局列号**。

    观察：#低波 + #换手稳 的分位等权，取最高 top_n。
    避雷：波动高（lowvol 分位取反）+ 换手不稳 + 过去跌得多，取最高 top_n。
    两张名单各自用自己的有效样本（某个因子缺值就只影响它自己那张），不插补。
    """
    def _idx(names):
        ok = mask.copy()
        for n in names:
            ok &= np.isfinite(mats[n][i])
        return np.nonzero(ok)[0]

    w_names = list(WATCH_FACTORS)
    widx = _idx(w_names)
    watch = np.empty(0, dtype="int64")
    watch_score = None
    if len(widx):
        pcts = np.vstack([_pct_rank(mats[n][i][widx]) for n in w_names])
        watch_score = pcts.mean(axis=0)
        k = min(int(top_n), len(widx))
        watch = widx[np.argsort(-watch_score, kind="stable")[:k]]

    a_names = [n for n, _, _ in AVOID_SPEC]
    aidx = _idx(a_names)
    avoid = np.empty(0, dtype="int64")
    avoid_score = None
    if len(aidx):
        terms = [w * (1.0 - _pct_rank(mats[n][i][aidx]) if sign < 0
                      else _pct_rank(mats[n][i][aidx]))
                 for n, sign, w in AVOID_SPEC]
        avoid_score = np.sum(terms, axis=0)
        k = min(int(top_n), len(aidx))
        avoid = aidx[np.argsort(-avoid_score, kind="stable")[:k]]

    return {"watch": watch, "avoid": avoid,
            "n_watch_pool": int(len(widx)), "n_avoid_pool": int(len(aidx)),
            "watch_score": watch_score, "avoid_score": avoid_score}


def rank_lists(fd, i: int, system: dict, mats: dict[str, np.ndarray],
               amount: np.ndarray, top_n: int = TOP_N) -> dict:
    """第 i 日（月末收盘）出的两张名单。可投池口径沿用 advisor.build_universe。"""
    uni = build_universe(fd, i, system, mats, amount)
    out = score_lists(mats, i, uni["mask"], top_n=top_n)
    out["date"] = str(fd.dates[i])
    out["n_universe"] = int(uni["mask"].sum())
    out["watch_codes"] = [fd.codes[c] for c in out["watch"]]
    out["avoid_codes"] = [fd.codes[c] for c in out["avoid"]]
    return out


# ---------------------------------------------------------------- 单一月份的持有期表现
def basket_nav(fd, cols: np.ndarray, i0: int, i1: int) -> tuple[pd.Series, int]:
    """固定 100 只等权持有 [i0, i1] 的日净值（i0 收盘建仓，归一为 1）。

    价格用后复权（含分红再投）。停牌 / hfq 缺口按**当日 0 收益**处理（净值冻结），
    不插补价格、不猜涨跌 —— 与回测层的处置一致。
    返回 (nav, 实际持有只数)；建仓日无价格的票直接不持有。
    """
    px = np.asarray(fd.panel.close_hfq[i0:i1 + 1][:, cols], dtype="float64")
    if px.size == 0:
        return pd.Series(dtype="float64"), 0
    base = px[0]
    keep = np.isfinite(base) & (base > 0)
    px = px[:, keep]
    if px.shape[1] == 0:
        return pd.Series(dtype="float64"), 0
    prev = np.vstack([px[:1], px[:-1]])
    good = np.isfinite(px) & np.isfinite(prev)
    ret = np.where(good, px / np.where(good, prev, 1.0) - 1.0, 0.0)
    daily = ret.mean(axis=1)
    nav = np.cumprod(1.0 + daily)
    return pd.Series(nav, index=[str(d) for d in fd.dates[i0:i1 + 1]]), int(keep.sum())


def month_result(fd, cols: np.ndarray, i0: int, i1: int, bench: pd.Series) -> dict:
    """一个月持有期的收益 / 月内 MDD，策略与基准同区间同起点。"""
    nav, n_held = basket_nav(fd, cols, i0, i1)
    idx = [str(d) for d in fd.dates[i0:i1 + 1]]
    bnav = bench.reindex(idx).astype("float64")
    if bnav.isna().any() or len(nav) < 2:
        missing = bnav.index[bnav.isna()].tolist()[:3]
        raise RuntimeError(f"H00300 在 {idx[0]}~{idx[-1]} 缺交易日（{missing}），不能评估")
    bnav = bnav / float(bnav.iloc[0])
    s_ret = float(nav.iloc[-1] - 1.0)
    b_ret = float(bnav.iloc[-1] - 1.0)
    return {"start": idx[0], "end": idx[-1],
            "n_held": n_held,
            "strategy_ret": _r(s_ret), "bench_ret": _r(b_ret),
            "excess": _r(s_ret - b_ret),
            "strategy_mdd": _r(M.max_drawdown(nav)["mdd"]),
            "bench_mdd": _r(M.max_drawdown(bnav)["mdd"])}


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def summarize(rows: list[dict], *, kind: str, pass_rate: float = PASS_RATE) -> dict:
    """把逐月结果汇总成验收口径的通过率。

    watch：月超额 ≥ 0 **且** 月内 MDD 不深于基准 → 当月合格
    avoid：月均收益 ≤ 基准同月收益 → 当月合格
    """
    if kind == "watch":
        ok = [r for r in rows if (r["excess"] is not None and r["excess"] >= 0.0
                                  and r["strategy_mdd"] >= r["bench_mdd"] - 1e-12)]
        rule = "月超额≥0 且 月内MDD不深于H00300"
    elif kind == "avoid":
        ok = [r for r in rows if (r["strategy_ret"] is not None
                                  and r["bench_ret"] is not None
                                  and r["strategy_ret"] <= r["bench_ret"] + 1e-12)]
        rule = "名单月收益 ≤ H00300 同月收益"
    else:
        raise ValueError(f"未知名单类型 {kind!r}")
    n = len(rows)
    rate = (len(ok) / n) if n else None
    return {"kind": kind, "rule": rule, "months": rows, "n_months": n,
            "n_qualified": len(ok), "pass_rate": _r(rate),
            "threshold": pass_rate,
            "passed": bool(rate is not None and rate >= pass_rate)}


# ---------------------------------------------------------------- 有效性统计（S3）
def validate(fd, cfg: dict, *, verbose: bool = False) -> dict:
    """按验收 §二 跑 2021~2025：防守月统计 + 状态灯误报/漏报披露。"""
    system = C.system_params(cfg)
    val = system.get("validation") or {}
    years = [str(y) for y in (val.get("years") or ["2021", "2022", "2023", "2024", "2025"])]
    mats: dict[str, np.ndarray] = {}
    for name in ("lowvol_20", "lowvol_60", "turn_stab_60", "reversal_60"):
        mats[name] = FACTOR_DEFS[name](fd)
    amount = trade_amount(fd)
    bench = load_benchmark()
    light = state_light(bench)
    dates = [str(d) for d in fd.dates]
    flags = month_flags(dates, light)

    ends = month_end_indices(fd.dates)
    watch_rows: list[dict] = []
    avoid_rows: list[dict] = []
    month_log: list[dict] = []
    list_rows: list[dict] = []
    for a, b in zip(ends, ends[1:]):
        ym_form, ym_hold = dates[a][:7], dates[b][:7]
        if ym_hold[:4] not in years:
            continue
        flag = flags.get(ym_hold) or {}
        defensive = bool(flag.get("defensive"))
        lists = rank_lists(fd, a, system, mats, amount)
        # 逐月名单代码明细（不只防守月）：审计要能按自己的一套判定复核，
        # 所以正常月的名单也必须留档，不能只交统计结果。
        for kind, cols in (("watch", lists["watch"]), ("avoid", lists["avoid"])):
            for rk, col in enumerate(cols, start=1):
                list_rows.append({"month": ym_hold, "formed_at": dates[a],
                                  "list_type": kind, "rank": rk,
                                  "code": fd.codes[col],
                                  "n_universe": lists["n_universe"]})
        w = month_result(fd, lists["watch"], a, b, bench)
        v = month_result(fd, lists["avoid"], a, b, bench)
        row = {"month": ym_hold, "formed_at": dates[a], "defensive": defensive,
               "watch_excess": w["excess"], "watch_mdd": w["strategy_mdd"],
               "bench_mdd": w["bench_mdd"], "avoid_ret": v["strategy_ret"],
               "bench_ret": v["bench_ret"], "n_watch": w["n_held"],
               "n_avoid": v["n_held"]}
        month_log.append(row)
        if defensive:
            watch_rows.append(dict(w, month=ym_hold, formed_at=dates[a]))
            avoid_rows.append(dict(v, month=ym_hold, formed_at=dates[a]))
        if verbose:
            print(f"[005S] {ym_hold} {'防守' if defensive else '正常'} "
                  f"观察超额={w['excess']:+.4f} 避雷-基准={v['strategy_ret']-v['bench_ret']:+.4f}",
                  flush=True)

    watch = summarize(watch_rows, kind="watch")
    avoid = summarize(avoid_rows, kind="avoid")

    # ---- 误报 / 漏报披露（不设硬闸）。口径：防守月里基准当月收益 ≥ 0 = 灯亮错了边；
    # 基准当月 ≤ -5% 却没亮 = 漏报。两条都只用"已发生"的数据判定，不构成信号。
    false_pos = [r for r in month_log if r["defensive"] and r["bench_ret"] is not None
                 and r["bench_ret"] >= 0.0]
    miss = [r for r in month_log if not r["defensive"] and r["bench_ret"] is not None
            and r["bench_ret"] <= -0.05]
    n_def = sum(1 for r in month_log if r["defensive"])

    latest_end = ends[-1]
    latest = rank_lists(fd, latest_end, system, mats, amount)
    # 灯态按基准**最新交易日**给（日频），名单按月末截面给；两个日期各自写进产物。
    latest_info = latest_block(latest, light, rolling_drawdown(bench, DD_WINDOW),
                               load_names())
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": SCHEMA_VERSION, **_git_meta(),
        "mode": "radar_005s",
        "benchmark": "H00300", "benchmark_path": BENCH_PATH,
        "years": years,
        "rules": {
            "light": f"H00300 收盘相对过去 {DD_WINDOW} 日最高收盘回撤 ≤ "
                     f"-{DD_THRESHOLD:.0%} 即防守期；月内任一交易日亮 → 该月为防守月",
            "watch": "低波(lowvol_20/60)+换手稳定(turn_stab_60) 分位等权，前 100 只",
            "avoid": "高波 0.4 + 换手不稳 0.3 + 过去60日跌得多 0.3，前 100 只",
            "watch_pass": f"防守月里 月超额≥0 且 月内MDD不深于基准 的月份占比 ≥ {PASS_RATE:.0%}",
            "avoid_pass": f"防守月里 名单月收益 ≤ 基准同月收益 的月份占比 ≥ {PASS_RATE:.0%}",
        },
        "light": {
            "window": DD_WINDOW, "threshold": DD_THRESHOLD,
            "n_months": len(month_log), "n_defensive_months": n_def,
            "defensive_share": _r(n_def / len(month_log) if month_log else None),
            "defensive_months": [r["month"] for r in month_log if r["defensive"]],
            "false_positive": {"n": len(false_pos),
                               "share_of_defensive": _r(len(false_pos) / n_def if n_def else None),
                               "months": [r["month"] for r in false_pos],
                               "rule": "防守月里 H00300 当月收益 ≥ 0（灯亮在错的一边，需披露不设闸）"},
            "miss": {"n": len(miss), "months": [r["month"] for r in miss],
                     "rule": "H00300 当月 ≤ -5% 却未标防守（漏报，需披露不设闸）"},
        },
        "watch": watch, "avoid": avoid,
        "months": month_log,
        "_monthly_lists": list_rows,
        "latest": latest_info,
        "product_note": PRODUCT_NOTE,
        "validity_note": [
            "只统计防守月；正常月的名单表现不构成建议，也不参与达标统计。",
            "名单月末出、次月持有，检验期不参与任何选择；不插补、不加权、不择时。",
            "持有期收益为零成本口径（雷达不交易），实际执行成本另计。",
            "基准唯一：沪深300全收益 H00300。",
            "latest.light_state 按基准最新交易日判定（日频），可能与名单截面日 "
            "latest.as_of 不同：两个日期分别写在各自字段里，不强行对齐；"
            "state 为 null 表示该日无法判定，不默认成 normal。",
            "latest.watch_names 来自 data/cache.db 的 stocks.code_name，只用于展示；"
            "查不到的名称列在 watch_names_missing，不插补、不用占位符顶替。"],
    }


def export_monthly_lists(rows: list[dict], path: Path) -> dict:
    """逐月两张名单的代码明细落盘（长表：一行一只票）。

    审计C 要能不复用我们的月度统计、自己按名单重算。所以这里给的是**原始名单**：
    月份、形成日、名单类型、名次、代码、当月可投池只数 —— 不含任何已算好的收益。
    """
    cols = ["month", "formed_at", "list_type", "rank", "code", "n_universe"]
    frame = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    if len(back) != len(frame):
        raise RuntimeError(f"名单落盘后行数不符：写 {len(frame)} 行、读回 {len(back)} 行")
    counts = back.groupby("list_type").size().to_dict() if len(back) else {}
    return {"path": path.name, "rows": int(len(frame)), "columns": cols,
            "months": int(back["month"].nunique()) if len(back) else 0,
            "first_month": str(back["month"].min()) if len(back) else None,
            "last_month": str(back["month"].max()) if len(back) else None,
            "rows_by_type": {k: int(v) for k, v in counts.items()},
            "note": "长表：month=持有月，formed_at=名单形成日（月末收盘），list_type=watch/avoid，"
                    "rank=按分数降序的名次，n_universe=当月可投池只数。不含收益字段，"
                    "审计可自行取价复核。"}


def main(argv: list[str] | None = None) -> int:
    import argparse
    from src.advisor import load_factor_data
    ap = argparse.ArgumentParser(description="005S 防守雷达（不构成投资建议）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=OUT_NAME)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = C.load_config(args.config)
    system = C.system_params(cfg)
    holdout_end = (system.get("holdout") or {}).get("end")
    fd = load_factor_data("2018-01-02", holdout_end)
    rep = validate(fd, cfg, verbose=not args.quiet)
    p = Path(args.out)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    rep["monthly_lists"] = export_monthly_lists(
        rep.pop("_monthly_lists", []), p.parent / LISTS_NAME)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"产物：{p}")
    print(f"名单明细：{p.parent / LISTS_NAME}（{rep['monthly_lists']['rows']} 行）")
    print(f"防守月 {rep['light']['n_defensive_months']}/{rep['light']['n_months']}；"
          f"观察达标 {rep['watch']['n_qualified']}/{rep['watch']['n_months']}"
          f"（{rep['watch']['pass_rate']}）；"
          f"避雷 {rep['avoid']['n_qualified']}/{rep['avoid']['n_months']}"
          f"（{rep['avoid']['pass_rate']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
