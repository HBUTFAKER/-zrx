"""S3-Q1 因子研究框架：截面计算 + IC 分析 + 分层回测。

本模块只做**因子刻画**，不产出选股策略、不给买卖信号。框架口径全部复用 S2：

  · 数据层：`backtest.load_panel`（hfq 缺口 / 停牌 / 涨跌停 mask 一并继承，
    绝不另搞一套清洗口径）；本模块只**追加**估值、成交、股本相关列；
  · 交易层：`backtest.run_backtest`（T 日信号 → T+1 开盘成交、佣金/印花税/过户费/
    滑点/涨跌停/T+1 全部一致）；
  · 指标层：`metrics.ann_return`（252 日折年）。

时点纪律（无未来函数）
----------------------
每个因子在截面日 T 的取值**只依赖 ≤ T-1 的数据**：

    momentum(T) = px(T-1) / px(T-1-window) - 1
    ep(T)       = 1 / peTTM(T-1)
    bp(T)       = 1 / pbMRQ(T-1)
    size(T)     = log( close(T-1) × volume(T-1) / (turn(T-1)/100) )

实现上统一走 `_shift1`：先整体下移一行，再在行内做同期运算。这样"用 T 日数据算 T 日因子"
在代码里根本写不出来，不是靠约定。

前瞻收益（IC 用）
-----------------
    fwd_ret(T_i) = open_hfq(T_{i+1} + 1) / open_hfq(T_i + 1) - 1

即 T_i 收盘信号 → T_i+1 开盘买入 → 下一截面日 +1 开盘卖出，与回测的成交时点一致。
理论收益，未扣除涨跌停/停牌导致的不可成交（分层回测里由引擎真实处理），
所以 IC 与分层收益之间允许有口径差，两者不可互相校验到小数位。

**段边界纪律（审计-B5-01）**：一期的前瞻收益窗跨出本段末交易日就整期丢弃，不计 IC。
dev 末截面 2025-10-31 的窗口是 2025-11-03 open → 2025-12-01 open，越过 DEV_END
（2025-11-06），等于让 dev 统计吃掉 1 期 holdout 收益。实现见 `ic_periods()`，
被丢的期连同原因写进报告 `segments.*._meta.ic_periods_dropped`。

数据口径·市值字段（003R3 后仍缺失，总工已裁定接受，不推翻）
-----------------------------------------------------------
cache 宽表**没有**总市值/流通市值字段。`finance.parquet` 的 totalShare/liqaShare
覆盖率仅 0.15%（270 行 / 64 只，全部来自 baostock 那批），不可用；原始日线
（data/raw/daily）本身也只有 close/volume/turn，没有市值。

本模块用**换手率定义式**推算流通股本：

    turn(%) = volume / 流通股本 × 100   ⇒   流通股本 = volume / (turn/100)
    流通市值 = close × 流通股本

这是恒等式推导，不是"换一个估值口径顶替"：误差只来自 turn 的舍入。

**对拍（审计-B5-05 更正，旧描述"2 个样本、误差 7e-6/2e-6"不可复现，已作废）**：
finance.parquet 里能与面板**同日**对上的只有 1 只（sh.600896 @ 2022-06-30），
推算 9.45425e8 股 vs liqaShare 9.4544e8，相对误差 1.6e-5；放宽到 ±7 个自然日，
可对证的仍是这 1 只（8 个交易日，同一 statDate），最差 2.6e-5、中位 1.6e-5。
分母证据：同一推算值对 liqaShare 误差 2.6e-5，对 totalShare 是 7.7e-2 ——
相差三个数量级，说明 `volume/turn` 复现的是**流通股本**而不是总股本。
复现：`python -m src.factors --diag-data`（打印每个对证点与两项误差）。

风险点（B5-03 / B5-04 实测，全部可复现）：
  · turn 缺失**全样本 0.71%**（68,953 / 9,669,155 行，100% 是停牌行：
    suspended=True 且 volume=0）；**2018 单年 4.66%**，是九年里最差的一年。
    旧 docstring 写的"4.66%"是 2018 单年数字，已更正为全局口径。
  · 解禁/增发会让推算股本**真实跳变**（与真实市值同步跳变，不是误差），但跳变当天
    直接污染 size 截面：单日跳变 >30% 共 8,628 行 / 3,693 只，其中 2,788 行落在
    月末前 7 个自然日内（距当月最后交易日 ≤7 个自然日；审计口径记 2,792，差 4 行
    来自窗口边界的自然日取法，不改变结论）。实证：sz.001289 2025-01-24
    推算股本一日 ×14.59、size 分位 55.2%→96.8%；sh.603786 2022-10-17 ×9.98、
    分位 35.3%→90.6%（sh.601698 2022-06-28 ×9.81、53.0%→94.7%）。口径与复现见 `data_quality_stats()` / 产物 `size_jump_stats`。

**size 的任何结论都必须带这个前提读**：它衡量的是"换手率推算市值"，跳变日会被
当成真实的市值变化。若判定不可接受，只需改 `size()` 一个函数的数据来源，
框架其余部分不受影响（代价：size 全部结论作废）。产物里 `size_source` 字段固定写
"log(close*volume/(turn/100))" 以便事后追溯。

分层残差与"绝对水平"（审计-B5-02 更正）
--------------------------------------
旧免责里写的「5 层合并 vs 全截面等权差约 0.2~0.3pp/月」「每层留 2~7% 残差」两处都对不上
实测，且把两个不同量混为一谈：

  · **整手残差**（同引擎、同成本、同 mask，5 层各 1/5 资金合并 vs 全截面等权）：
    dev·bp 月差 mean|diff| 0.12pp、中位 0.07pp、p90 0.29pp，signed 年化 −0.42pp；
  · **每层配不满**（部分成交股的未达目标缺口）：0.70%~2.27%，不是"2~7%"；
    未投入现金 <0.01% —— 引擎按资金约束贪心成交，现金基本买空，残差体现为
    "某些股票配少了"，不是"留了一笔现金"；
  · **2pp 级别那个差不只是残差，而且窗口错配**：8.15% 是 dev 段（1903 个交易日）的
    全截面等权引擎收益，5.87% 是**全期**（2145 日）全市场 hfq 等权基准 —— 两个窗口不同，
    相减得到的 2.25pp 主要来自窗口而不是股票池。同窗口对比：8.15% vs 8.03%，差 0.13pp。

以上全部由 `evaluate_segment` 实时产出（`lot_residual` / `_meta.full_xsec` /
`_meta.market_ew`），复现命令：`python -m src.factors --out out`。

数据字段缺失即停工
------------------
`load_factor_data` 构造时硬校验 peTTM / pbMRQ / volume / turn 四列，缺任一个直接
RuntimeError 并提示"宽表缺字段，停工上报"。理由是价值/市值因子换源即换口径，
静默降级会让整份报告失效且看不出来。

复用说明：本模块不改 backtest.py / metrics.py 的现有函数，只 import 使用。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .backtest import CACHE, ROOT, Config, Panel, load_panel, run_backtest

# ---------------------------------------------------------------- 常量

DEV_END = "2025-11-06"          # 开发段最后一天（含）
HOLDOUT_START = "2025-11-07"    # 冻结样本第一天（含）—— 业主拍板，硬约束

FACTOR_COLS = ("peTTM", "pbMRQ", "volume", "turn")   # 缺任一个即停工

MIN_LISTED_DAYS = 60    # 上市不足 60 个交易日的样本不进截面（新股初期价格/市值失真）
MIN_XSEC = 30           # 截面少于 30 只不算 IC（统计上没有意义）
MIN_PER_LAYER = 5       # 每层至少 5 只，否则该期跳过

N_LAYERS = 5

VALIDITY_NOTE = (
    "本报告是因子刻画结果，不构成策略有效性或 alpha 证据。IC/分层收益受以下因素影响："
    "① 分层为层内等权、月度再平衡，未做行业/风格中性化；"
    "② IC 用理论开盘价收益，未扣除涨跌停/停牌造成的不可成交（诚实起见：IC 与分层收益之间允许有口径差）；"
    "③ 市值由换手率推算（宽表无市值字段）：单日推算股本跳变 >30% 共 8,628 行 / 3,693 只，"
    "其中 2,788 行落在月末前 7 个自然日内（即落在 size 月度截面上），"
    "如 sz.001289 2025-01-24 推算股本一日 ×14.59、size 分位 55.2%→96.8%，"
    "sh.603786 2022-10-17 ×9.98、35.3%→90.6%；"
    "**size 的全部结论必须带这个前提读**（口径与复现见产物 size_jump_stats 字段）；"
    "④ 截面已剔除 ST/停牌/缺口/上市不足 60 日；"
    "⑤ holdout 段 12 个月、11 期 IC，样本量不足以支撑统计推断，只作记录；"
    "⑥ **层间比较有效，绝对水平不可与全市场等权直接相比**。整手残差实测"
    "（5 层各 1/5 资金合并 vs 全截面等权，同引擎同成本同 mask，dev·bp）：dev 全段月差 "
    "mean|diff|=0.12pp、中位 0.07pp、p90 0.29pp，signed 年化 −0.42pp；"
    "每层部分成交缺口 0.70%~2.27%，未投入现金 <0.01%（残差是「配不满」，不是「留现金」）。"
    "旧稿把「5 层合并 vs 全截面等权差 2.25pp」记作整手残差，是**标签错误**，归因也不对："
    "2.25pp 来自**窗口错配**（dev 段引擎 8.15% 对全期基准 5.87%），"
    "同窗口下全截面引擎 8.15% vs 全市场 hfq 等权 8.03%，只差 0.13pp —— 与整手取整无关；"
    "⑦ 月频 IC 存在自相关，t 值会**高估**显著性，只作量级参考 —— 不以 |t|>2 判定因子有效，"
    "也不以「唯一 |t|>3」作为结论支点（同表里动量 |t|≈4.1 同样过线）。"
    "策略有效性判断须待后续中性化与样本外验证，不以本报告为准。"
)

SIZE_SOURCE = "log(close*volume/(turn/100))  # 流通市值，由换手率定义式推算"


# ---------------------------------------------------------------- 数据层

@dataclass
class FactorData:
    """因子计算所需的全量数据：复用 Panel + 追加估值/成交/股本列。

    panel     ：load_panel 产物（价格、mask、hfq 缺口全部沿用）
    pe_ttm    ：PE(TTM)，与 panel 同形状
    pb_mrq    ：PB(MRQ)，同上
    volume    ：成交量（股），同上
    turn      ：换手率（%），同上
    is_st     ：是否 ST（1/0），同上
    listed_days：上市交易日数，同上
    source    ：数据来源与缺口统计（写进产物）
    """
    panel: Panel
    pe_ttm: np.ndarray
    pb_mrq: np.ndarray
    volume: np.ndarray
    turn: np.ndarray
    is_st: np.ndarray
    listed_days: np.ndarray
    source: dict = field(default_factory=dict)

    @property
    def dates(self) -> list[str]:
        return self.panel.dates

    @property
    def codes(self) -> list[str]:
        return self.panel.codes

    def __post_init__(self) -> None:
        shp = self.panel.close_raw.shape
        for name in ("pe_ttm", "pb_mrq", "volume", "turn", "is_st", "listed_days"):
            arr = getattr(self, name)
            if arr.shape != shp:
                raise ValueError(f"{name} 形状 {arr.shape} 与面板 {shp} 不一致")


def _real_listed_days(dates: list[str], codes: list[str]) -> np.ndarray:
    """真实上市交易日数（相对面板日历），用于"上市不足 N 日不进截面"的过滤。

    为什么不能直接用 clean.py 的 listed_days：它是
    `searchsorted(全市场日历, ipoDate)` 算的，ipoDate 早于日历起点时返回 0，
    于是**存量股被算成"上市第 1 天"**（这就是 003R4 遗留的 B3-05②，数据层未重建）。
    实测后果：2018-01-31 全市场 listed_days 中位数 22、最大 22，
    "上市 ≥60 日"会把 2018 年前 3 个月的截面整段清空（实测 dev 段白丢 3 期 IC）。

    本函数按 ipoDate 重算：早于日历起点 → 视为已上市很久（给大数，不参与过滤）；
    真新股按日历内位置算。**只影响过滤，不改任何数据层产物。**
    """
    from .backtest import _load_ipo_dates      # 私有函数，只读复用，不修改

    ipo = _load_ipo_dates(codes)
    cal = np.array([np.datetime64(str(d)) for d in dates])
    n_d = len(dates)
    out = np.zeros((n_d, len(codes)), dtype="float64")
    for j, c in enumerate(codes):
        raw = ipo.get(c) if ipo is not None else None
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)) or str(raw) == "":
            out[:, j] = 0.0                    # 拿不到 ipoDate：当新股（不放大）
            continue
        d0 = str(raw)[:10]
        if d0 < dates[0]:
            out[:, j] = 1e6                    # 存量股：早于日历起点，视为老股
            continue
        k = int(np.searchsorted(cal, np.datetime64(d0)))
        out[:, j] = np.maximum(np.arange(n_d) - k + 1, 0)
    return out


def _parquet_columns(path: Path) -> list[str]:
    """只读 parquet 的列名（不加载数据），用于字段缺失的开工前校验。"""
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001 —— 退回 pandas，慢但不至于失败
        return list(pd.read_parquet(path).head(0).columns)


# ---------------------------------------------------------------- 数据质量（B5-03 / B5-04 / B5-05）

#: 跳变实证案例（固定代码，避免每次跑出不同样本；含"为什么它危险"）
JUMP_CASES = ("sz.001289", "sh.603786", "sh.601698")


def data_quality_stats(start: str, end: str, cache_dir: Path | None = None,
                       jump_threshold: float = 0.30,
                       month_end_days: int = 7) -> dict:
    """推算股本跳变 + turn 缺失率 + 真实股本对拍，一次性算清（B5-03/04/05）。

    口径全部写死在返回结构里，宁可啰嗦也不留"看代码猜"的空间：

    · 推算股本 `s = volume / (turn/100)`，只保留有限正值（停牌日 turn 为 NaN，跳过）；
    · 跳变 = 本日 s 相对**上一个有效观测**的变化超过 ±`jump_threshold`（默认 30%）。
      为什么不用"上一行"：停牌日 turn 缺失会让"上一行"变 NaN，等于把跨停牌的真实跳变
      漏掉（实测口径差 167 行）；
    · 月末窗口 = 距当月**最后一个交易日** ≤`month_end_days` 个自然日（含月末当日）——
      size 截面取在月末交易日，这个窗口内的跳变会直接改掉当月的因子值；
    · 对拍：finance.parquet 的 liqaShare/totalShare 与面板同日（或 ±7 日内）比对，
      同时给 totalShare 的误差作分母证据（复现表见返回结构 reproducibility）。

    返回 dict 可直接写进 factors_report.json（全为 JSON 原生类型）。
    """
    cache_dir = cache_dir or CACHE
    years = range(int(start[:4]), int(end[:4]) + 1)
    frames = []
    for y in years:
        f = cache_dir / f"daily_{y}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"缺 {f}，先跑 python -m src.clean --years ...")
        frames.append(pd.read_parquet(
            f, columns=["code", "date", "close", "volume", "turn", "suspended"]))
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["date"] >= start) & (d["date"] <= end)].sort_values(["code", "date"])
    d = d.reset_index(drop=True)

    # ---- B5-04 turn 缺失率（全局 + 分年）
    miss = d["turn"].isna().to_numpy()
    susp = d["suspended"].fillna(False).to_numpy().astype(bool)
    vol0 = (d["volume"].fillna(0) == 0).to_numpy()
    yr = d["date"].str[:4]
    by_year = {}
    for y in sorted(yr.unique()):
        m = (yr == y).to_numpy()
        by_year[y] = {"rows": int(m.sum()), "turn_missing": int((miss & m).sum()),
                      "rate": round(float((miss & m).sum() / max(m.sum(), 1)), 8)}
    turn_missing = {
        "rows": int(len(d)),
        "missing": int(miss.sum()),
        "rate": round(float(miss.sum() / max(len(d), 1)), 8),
        "all_suspended": bool(((~miss) | (susp & vol0)).all()),
        "by_year": by_year,
        "worst_year": max(by_year, key=lambda y: by_year[y]["rate"]) if by_year else None,
        "note": "100% 缺失行同时是停牌行（suspended=True 且 volume=0）；"
                "2018 单年最高，属早期停牌多，不是数据缺列",
    }

    # ---- B5-03 推算股本跳变
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = d["volume"].to_numpy("float64") / (d["turn"].to_numpy("float64") / 100.0)
    sh[~(np.isfinite(sh)) | (sh <= 0)] = np.nan
    code = d["code"]
    s = pd.Series(sh)
    prev = s.groupby(code).ffill().groupby(code).shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sh / prev
    flag = np.isfinite(ratio) & (np.abs(ratio - 1.0) > jump_threshold)

    dt = pd.to_datetime(d["date"])
    ym = dt.dt.to_period("M")
    me = dt.groupby(ym).max()                       # 当月最后一个交易日
    delta = (pd.to_datetime(ym.map(me)).to_numpy() - dt.to_numpy()).astype("timedelta64[D]").astype(int)
    in_window = flag & (delta <= month_end_days)

    # 实证案例：固定代码里取跳变最猛的一次，附截面分位（当天全样本 rank，含 ST/新股）
    size = pd.Series(np.log(np.where(np.isfinite(sh), d["close"].to_numpy("float64") * sh, np.nan)))
    pct = size.groupby(d["date"]).rank(pct=True).to_numpy()
    examples = []
    for c in JUMP_CASES:
        m = (code == c).to_numpy()
        if not m.any():
            continue
        idx = np.nonzero(m & flag)[0]
        if not len(idx):
            continue
        k = idx[np.argmax(np.abs(ratio[idx] - 1.0))]
        j = k - 1 if k - 1 >= 0 and code.iloc[k - 1] == c else None
        examples.append({
            "code": c, "date": str(d["date"].iloc[k]),
            "share_ratio": round(float(ratio[k]), 6),
            "size_pct_before": None if j is None or not np.isfinite(pct[j]) else round(float(pct[j]), 6),
            "size_pct_after": None if not np.isfinite(pct[k]) else round(float(pct[k]), 6),
        })

    size_jump = {
        "threshold": jump_threshold,
        "method": "s=volume/(turn/100)；与上一个**有效**观测比，|s_t/s_prev-1|>阈值 记一跳",
        "rows": int(flag.sum()),
        "codes": int(d.loc[flag, "code"].nunique()),
        "month_end_window_days": int(month_end_days),
        "month_end_rows": int(in_window.sum()),
        "month_end_window": "距当月最后一个交易日 ≤N 个自然日（含月末当日）",
        "examples": examples,
        "note": "解禁/增发是真实股本变化，不是数据错误；但它会在 size 截面上表现为"
                "一次无经济含义的市值跳变，故 size 结论须带此前提",
    }

    # ---- B5-05 真实股本对拍（同日 / ±7 日两组）
    fin_path = cache_dir / "finance.parquet"
    cross_check = {"available": False, "reason": "missing: finance.parquet"}
    if fin_path.exists():
        fin = pd.read_parquet(fin_path, columns=["code", "statDate", "totalShare", "liqaShare"])
        fin = fin[fin["liqaShare"].notna()].copy()
        fin["statDate"] = pd.to_datetime(fin["statDate"])
        m = d[["code", "date"]].copy()
        m["sh"] = sh
        m = m[np.isfinite(m["sh"])]
        j = fin.merge(m, on="code", how="inner")
        j["gap"] = (pd.to_datetime(j["date"]) - j["statDate"]).dt.days.abs()
        groups = {}
        for tag, win in (("same_day", 0), ("within_7d", 7)):
            g = j[j["gap"] <= win]
            if not len(g):
                groups[tag] = {"points": 0}
                continue
            e_liq = (g["sh"] / g["liqaShare"] - 1.0).abs()
            e_tot = (g["sh"] / g["totalShare"] - 1.0).abs()
            worst = g.loc[e_liq.idxmax()]
            groups[tag] = {
                "points": int(len(g)), "codes": int(g["code"].nunique()),
                "err_liqa_max": round(float(e_liq.max()), 10),
                "err_liqa_median": round(float(e_liq.median()), 10),
                "err_total_max": round(float(e_tot.max()), 10),
                "err_total_median": round(float(e_tot.median()), 10),
                "worst_point": {"code": str(worst["code"]), "date": str(worst["date"]),
                                "statDate": str(worst["statDate"])[:10]},
            }
        cross_check = {
            "available": True,
            "same_day": groups.get("same_day", {"points": 0}),
            "within_7d": groups.get("within_7d", {"points": 0}),
            "denominator_evidence": "同一推算值对 liqaShare 的误差应比 totalShare 小 3 个数量级；"
                                    "若反过来，说明 volume/turn 复现的是总股本，口径即错",
        }

    return {
        "scope": {"start": str(d["date"].iloc[0]), "end": str(d["date"].iloc[-1]),
                  "rows": int(len(d)), "codes": int(d["code"].nunique())},
        "turn_missing": turn_missing,
        "size_jump": size_jump,
        "share_cross_check": cross_check,
        "reproduce": "python -m src.factors --diag-data "
                     f"--start {start} --end {end}",
    }



def load_factor_data(start: str, end: str, codes: list[str] | None = None,
                     cache_dir: Path | None = None) -> FactorData:
    """载入因子数据。价格与 mask 完全来自 `load_panel`，本函数只追加 6 列。

    字段缺失（peTTM / pbMRQ / volume / turn 任一不存在）→ RuntimeError 停工。
    这是硬约束：换字段就是换口径，静默降级会让报告失真且无法事后发现。
    """
    cache_dir = cache_dir or CACHE
    years = range(int(start[:4]), int(end[:4]) + 1)

    need = ["code", "date", *FACTOR_COLS, "isST", "listed_days"]
    frames = []
    for y in years:
        f = cache_dir / f"daily_{y}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"缺 {f}，先跑 python -m src.clean --years ...")
        have = set(_parquet_columns(f))
        missing = [c for c in FACTOR_COLS if c not in have]
        if missing:
            raise RuntimeError(
                f"{f.name} 缺少因子所需字段 {missing}；"
                f"宽表口径不含该字段，按纪律**停止施工并上报**，不许换源或换口径顶替。")
        d = pd.read_parquet(f, columns=need)
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if codes is not None:
            d = d[d["code"].isin(codes)]
        if len(d):
            frames.append(d)
    if not frames:
        raise ValueError(f"{start}~{end} 区间内没有任何行情")

    extra = pd.concat(frames, ignore_index=True)
    del frames

    panel = load_panel(start, end, codes=codes, cache_dir=cache_dir)
    dates_all, codes_all = panel.dates, panel.codes

    def _mat(name: str) -> np.ndarray:
        m = extra.pivot(index="date", columns="code", values=name).reindex(
            index=dates_all, columns=codes_all)
        return m.to_numpy(dtype="float64", na_value=np.nan)

    fd = FactorData(
        panel=panel,
        pe_ttm=_mat("peTTM"),
        pb_mrq=_mat("pbMRQ"),
        volume=_mat("volume"),
        turn=_mat("turn"),
        is_st=np.nan_to_num(_mat("isST"), nan=1.0),
        # 上市天数按 ipoDate 重算（clean.py 的 listed_days 相对面板起点，会把
        # 存量股算成"上市第 1 天"，详见 _real_listed_days）
        listed_days=_real_listed_days(dates_all, codes_all),
        source={
            "start": dates_all[0], "end": dates_all[-1],
            "dates": len(dates_all), "codes": len(codes_all),
            "rows": int(len(extra)),
            "hfq_gap_rows": panel.source.get("hfq_gap_rows", {}),
            "size_source": SIZE_SOURCE,
            "min_listed_days": MIN_LISTED_DAYS,
            "fields": list(FACTOR_COLS),
        },
    )
    return fd


# ---------------------------------------------------------------- 因子计算（严格 T-1）

def _shift1(a: np.ndarray) -> np.ndarray:
    """整体下移一行：out[i] = a[i-1]，第 0 行为 NaN。

    所有因子都要先过这一步——这是"只用 ≤T-1 数据"的物理实现。
    """
    out = np.full(a.shape, np.nan, dtype="float64")
    out[1:] = a[:-1]
    return out


def momentum(px: np.ndarray, window: int) -> np.ndarray:
    """动量因子：T 日值 = px(T-1) / px(T-1-window) - 1。

    px 必须是**后复权价**（hfq_close）：用原始价会在除权日出现假跌幅
    （实测 hfq/close 中位 2.57 倍、最高 1 万倍）。
    窗口内任一端缺价（停牌、hfq 缺口、上市不足）→ NaN，不插补。
    """
    out = np.full(px.shape, np.nan, dtype="float64")
    if px.shape[0] > window:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[window + 1:] = px[window:-1] / px[:-(window + 1)] - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def ep(pe_ttm: np.ndarray) -> np.ndarray:
    """价值因子 EP = 1/PE(TTM)，用 T-1 日估值。

    亏损股 peTTM < 0 → EP 为负（保留，排序落在最差端，符合经济含义）；
    peTTM == 0 或非有限 → NaN（只标注不插补）。
    """
    p = _shift1(pe_ttm)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1.0 / p
    v[~np.isfinite(v)] = np.nan
    v[np.isnan(p) | (p == 0.0)] = np.nan
    return v


def bp(pb_mrq: np.ndarray) -> np.ndarray:
    """价值因子 BP = 1/PB(MRQ)，用 T-1 日估值。处理同 `ep`。"""
    p = _shift1(pb_mrq)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1.0 / p
    v[~np.isfinite(v)] = np.nan
    v[np.isnan(p) | (p == 0.0)] = np.nan
    return v


def size(close_raw: np.ndarray, volume: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """小市值因子 = log(流通市值)，用 T-1 日数据。

    流通市值 = close × volume/(turn/100)（换手率定义式，见模块 docstring）。
    用**原始价**不用 hfq：市值本身不受复权影响。
    turn ≤ 0 或 volume ≤ 0（停牌）→ NaN。
    """
    c, v, t = _shift1(close_raw), _shift1(volume), _shift1(turn)
    with np.errstate(divide="ignore", invalid="ignore"):
        fmv = c * v / (t / 100.0)
    bad = (~np.isfinite(fmv)) | (fmv <= 0) | ~(t > 0) | ~(v > 0)
    fmv = np.where(bad, np.nan, fmv)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(fmv)
    out[~np.isfinite(out)] = np.nan
    return out


FACTOR_DEFS = {
    "momentum_20": lambda fd: momentum(fd.panel.close_hfq, 20),
    "momentum_60": lambda fd: momentum(fd.panel.close_hfq, 60),
    "ep": lambda fd: ep(fd.pe_ttm),
    "bp": lambda fd: bp(fd.pb_mrq),
    "size": lambda fd: size(fd.panel.close_raw, fd.volume, fd.turn),
}


def compute_factors(fd: FactorData, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """批量计算因子矩阵，返回 {因子名: (n_dates, n_codes)}。"""
    names = names or list(FACTOR_DEFS)
    out = {}
    for n in names:
        if n not in FACTOR_DEFS:
            raise KeyError(f"未知因子 {n}，可选 {sorted(FACTOR_DEFS)}")
        out[n] = FACTOR_DEFS[n](fd)
    return out


# ---------------------------------------------------------------- 截面与 IC

def month_end_indices(dates: list[str]) -> list[int]:
    """每月最后一个交易日的行索引（升序）。"""
    last: dict[str, int] = {}
    for i, d in enumerate(dates):
        last[str(d)[:7]] = i
    return sorted(last.values())


def cross_section_mask(fd: FactorData, i: int) -> np.ndarray:
    """第 i 日可进截面的样本：有行情 + 未停牌 + 非 hfq 缺口 + 非 ST + 上市够久。

    前四条直接继承 `load_panel` 的口径（panel.suspended / hfq_gap / close_raw 是否缺失）。
    """
    p = fd.panel
    ok = (~np.isnan(p.close_raw[i])) & (~p.suspended[i]) & (~p.hfq_gap[i])
    ok &= (fd.is_st[i] == 0)
    ok &= (fd.listed_days[i] >= MIN_LISTED_DAYS)
    return ok


def forward_returns(open_hfq: np.ndarray, ends: list[int]) -> np.ndarray:
    """每期前瞻收益：T_i+1 开盘 → T_{i+1}+1 开盘。最后一期无下一期 → 全 NaN。

    返回 (len(ends), n_codes)。
    """
    n = open_hfq.shape[0]
    out = np.full((len(ends), open_hfq.shape[1]), np.nan, dtype="float64")
    for k in range(len(ends) - 1):
        a, b = ends[k] + 1, ends[k + 1] + 1
        if a < n and b < n:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[k] = open_hfq[b] / open_hfq[a] - 1.0
            out[k][~np.isfinite(out[k])] = np.nan
    return out


def ic_periods(fd: FactorData, i0: int, i1: int) -> tuple[list[dict], list[dict]]:
    """段内每一期 IC 的窗口定义；跨出段末的期直接丢弃（审计-B5-01）。

    一期的窗口是：
        截面日 T_i（收盘出信号） → T_i+1 开盘买入 → T_{i+1}+1 开盘卖出
    最后一个日期才是"收益真正兑现"的那天。若它晚于本段末交易日，这期的收益里就
    掺进了下一段（dev 掺进 holdout）的行情 —— dev 统计被 holdout 数据污染，
    holdout 也就不再是干净的样本外。

    实测：dev 末截面 2025-10-31 的退出日是 2025-12-01 open，越过 DEV_END 2025-11-06，
    正好吞掉 1 期。处理方式是**整期丢弃**（不做截断、不换更短的窗口）：截断会改变
    期与期的持有期长度，让 IC 序列口径不一致，比少一期更难解释。

    返回 (usable, dropped)。usable 每项：
        {"cross_section": 截面日, "entry_date": 买入日, "exit_date": 退出日,
         "idx": 截面日全局行号, "k": 在全局月末序列里的位置}
    dropped 每项带 reason：no_next_period（样本内最后一个截面）/ no_quote（缺报价行）/
    exit_beyond_segment_end（跨段）。
    """
    dates = fd.dates
    n = len(dates)
    seg_end = dates[i1 - 1]
    ends_global = month_end_indices(dates)
    pos = {i: k for k, i in enumerate(ends_global)}
    usable: list[dict] = []
    dropped: list[dict] = []
    for i in [x for x in ends_global if i0 <= x < i1]:
        k = pos[i]
        if k + 1 >= len(ends_global):
            dropped.append({"cross_section": dates[i], "exit_date": None,
                            "reason": "no_next_period"})
            continue
        a, b = i + 1, ends_global[k + 1] + 1
        if a >= n or b >= n:
            dropped.append({"cross_section": dates[i], "exit_date": None,
                            "reason": "no_quote"})
            continue
        exit_date = dates[b]
        if exit_date > seg_end:
            dropped.append({"cross_section": dates[i], "exit_date": exit_date,
                            "reason": "exit_beyond_segment_end"})
            continue
        usable.append({"cross_section": dates[i], "entry_date": dates[a],
                       "exit_date": exit_date, "idx": i, "k": k})
    return usable, dropped


def rank_ic(f: np.ndarray, r: np.ndarray, min_n: int = MIN_XSEC) -> float:
    """Spearman rank IC：因子秩与前瞻收益秩的 Pearson 相关。

    只用两者都有限的样本；样本数 < min_n 或任一侧无波动 → NaN。
    """
    ok = np.isfinite(f) & np.isfinite(r)
    n = int(ok.sum())
    if n < min_n:
        return float("nan")
    a = pd.Series(f[ok]).rank().to_numpy(dtype="float64")
    b = pd.Series(r[ok]).rank().to_numpy(dtype="float64")
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ic_summary(ics: list[float]) -> dict:
    """IC 统计：均值 / 标准差 / ICIR / t 值 / IC>0 占比 / 有效期数。

    ICIR = mean/std（不做年化，月频口径）。
    t = mean / (std/√n)。

    **免责（审计-B5-07③，硬性要求）**：月频 IC 存在自相关（因子暴露与持仓连续），
    上式的 t 值假设各期独立，会**高估**显著性。因此：
      · 只把 |t| 当量级参考，不用 |t|>2 判定"因子有效"；
      · 不用"唯一 t>3"这种表述下结论 —— 本报告实测动量 |t|≈4.1 同样过线，
        说明 t 值差异更多来自波动而非经济含义；
      · 真正判显著性要用 Newey-West 调整或分块 bootstrap，留到 004B。
    """
    v = np.array([x for x in ics if np.isfinite(x)], dtype="float64")
    n = int(len(v))
    if n == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"),
                "t_stat": float("nan"), "ic_pos_ratio": float("nan"), "n_periods": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if (std and np.isfinite(std) and std > 0) else float("nan")
    t = icir * np.sqrt(n) if np.isfinite(icir) else float("nan")
    return {"ic_mean": mean, "ic_std": std, "icir": float(icir),
            "t_stat": float(t), "ic_pos_ratio": float((v > 0).mean()), "n_periods": n}


# ---------------------------------------------------------------- 分层

def split_layers(f: np.ndarray, ok: np.ndarray, n_layers: int = N_LAYERS,
                 min_per_layer: int = MIN_PER_LAYER) -> list[np.ndarray]:
    """按因子值分位切层，返回 [Q1(最低), ..., Qn(最高)] 的索引数组。

    用 rank(method="first") 消除并列，保证各层样本数完全相等（差不超过 1）。
    """
    idx = np.nonzero(ok & np.isfinite(f))[0]
    if len(idx) < n_layers * min_per_layer:
        return []
    vals = f[idx]
    order = np.argsort(vals, kind="mergesort")      # 升序：Q1 = 因子值最小
    ranks = np.empty(len(idx), dtype="int64")
    ranks[order] = np.arange(len(idx))
    layer_id = (ranks * n_layers) // len(idx)        # 等分，无空层
    return [idx[layer_id == k] for k in range(n_layers)]


def build_layer_signals(fmat: np.ndarray, fd: FactorData, ends: list[int],
                        n_layers: int = N_LAYERS) -> list[dict]:
    """为每个分层生成 run_backtest 用的信号：{日期: {代码: 等权}}。

    截面日 T 的因子值只依赖 ≤T-1 数据，引擎在 T+1 开盘执行。
    返回长度 n_layers 的列表，第 k 个是第 k 层的信号。
    """
    dates, codes = fd.dates, fd.codes
    sigs: list[dict] = [dict() for _ in range(n_layers)]
    for i in ends:
        ok = cross_section_mask(fd, i)
        layers = split_layers(fmat[i], ok, n_layers)
        if not layers:
            continue
        for k, members in enumerate(layers):
            if len(members) == 0:
                continue
            w = 1.0 / len(members)
            sigs[k][dates[i]] = {codes[j]: w for j in members}
    return sigs


# ---------------------------------------------------------------- 分段

def slice_panel(panel: Panel, i0: int, i1: int) -> Panel:
    """按行切片面板（含 hfq_gap），用于把 dev / holdout 分成两段独立回测。"""
    return Panel(
        dates=panel.dates[i0:i1], codes=panel.codes,
        open_hfq=panel.open_hfq[i0:i1], close_hfq=panel.close_hfq[i0:i1],
        open_raw=panel.open_raw[i0:i1], close_raw=panel.close_raw[i0:i1],
        blocked_buy=panel.blocked_buy[i0:i1], blocked_sell=panel.blocked_sell[i0:i1],
        suspended=panel.suspended[i0:i1], hfq_gap=panel.hfq_gap[i0:i1],
        source=dict(panel.source),
    )


def segment_bounds(dates: list[str]) -> dict[str, tuple[int, int]]:
    """dev / holdout 的行区间 [i0, i1)。

    dev      ：起始 ~ DEV_END（含）
    holdout  ：HOLDOUT_START（含）~ 结束
    两段都必须非空，否则抛错（数据不足以分段时不许硬跑）。
    """
    dev_idx = [i for i, d in enumerate(dates) if d <= DEV_END]
    ho_idx = [i for i, d in enumerate(dates) if d >= HOLDOUT_START]
    if not dev_idx or not ho_idx:
        raise ValueError(
            f"数据不足以切分 dev/holdout：dev {len(dev_idx)} 行、holdout {len(ho_idx)} 行")
    return {"dev": (0, dev_idx[-1] + 1), "holdout": (ho_idx[0], len(dates))}


# ---------------------------------------------------------------- 分层回测

def affordability_stats(sigs: list[dict], panel: Panel, cash: float,
                        lot: int = 100) -> list[float]:
    """整手可行性诊断：每层"目标单只市值买不起一手"的比例（各期取中位数）。

    为什么必须看这个：分层每层约 950 只，若本金只够每只一千元，
    高价股根本配不进去，分层收益就退化成"价格分层"而**不是**因子分层。
    实测 100 万本金时 Q1 只有 21% 的股票买得起、Q5 有 77%（低价股多的层更容易配满），
    这个差异本身就会伪造出多空收益。本金必须大到一手成本可以忽略。
    """
    di = {d: i for i, d in enumerate(panel.dates)}
    ci = {c: j for j, c in enumerate(panel.codes)}
    out = []
    for sig in sigs:
        ratios = []
        for d, w in sig.items():
            i = di.get(d)
            if i is None:
                continue
            px = panel.open_raw[i]
            hit = []
            for c, wt in w.items():
                j = ci.get(c)
                if j is None:
                    continue
                cost = px[j] * lot
                if np.isfinite(cost) and cost > 0:
                    hit.append(1.0 if cost <= cash * wt else 0.0)
            if hit:
                ratios.append(float(np.mean(hit)))
        out.append(float(np.median(ratios)) if ratios else float("nan"))
    return out


def _layer_cfg(fd: FactorData, i0: int, i1: int, tmpl: Config) -> Config:
    """分层回测**实际使用**的 Config（B5-06：决定结果的参数只在这一处生成）。

    与 S2 的差别只有两处，且必须落进产物：
      · max_positions = 全部股票数（层内全持有，不截断；S2 默认 50）
      · max_weight    = 1.0（层内等权，不受 S2 的 5% 上限约束）
    lot_size / costs / seed / init_cash 一律从 tmpl 原样继承。旧实现只传了 costs/seed，
    lot_size 被悄悄用默认值 100 —— 改模板里的 lot_size 对结果毫无影响，产物也看不出来，
    这是 B5-06 要堵的洞。
    """
    return Config(start=fd.dates[i0], end=fd.dates[i1 - 1],
                  init_cash=tmpl.init_cash,
                  max_positions=len(fd.codes),
                  max_weight=1.0,
                  lot_size=tmpl.lot_size,
                  seed=tmpl.seed,
                  costs=tmpl.costs)


def _fill_stats(result, sig: dict) -> dict:
    """单层的"配不齐"实测：三个口径分开算，各期取中位数（B5-02）。

    · partial_gap = Σ_{部分成交股} max(0, 目标权重 − 实际权重)。
      只在"买到了、但没买够"的股票上求和，**排除**整只没成交的（涨停/停牌/无报价）——
      后者是"买不到"而不是"整手取整留残差"，混进来会把残差从 2% 抬到 6%（实测 Q1
      5.8%，其中大部分是这类）。这就是审计里"每层留 2~7% 残差"该被量化的那个数。
    · weight_dev  = ½Σ|实际权重 − 目标权重|（全并集，含买不到的），衡量整体偏离。
    · cash_pct    = 1 − (权益−现金)/权益，即没投出去的现金占比。
      实测这个数极小（< 0.01%）：引擎按资金约束贪心成交，现金基本买空 ——
      残差体现为"某些股票配少了"，不是"留了一笔现金"。三个数必须分开报，
      否则又会出"标签对不上数字"（B5-02 的教训）。
    """
    if result is None or not len(result.nav):
        return {"partial_gap_pct": None, "weight_dev_pct": None,
                "cash_pct": None, "n_days": 0}
    nav, pos = result.nav, result.positions
    di = {str(d): i for i, d in enumerate(nav["date"].tolist())}
    by_date = {str(k): v for k, v in pos.groupby("date")} if len(pos) else {}
    gaps, devs, cashes = [], [], []
    for d, w in sig.items():
        i = di.get(d)
        if i is None or i + 1 >= len(nav):
            continue
        row = nav.iloc[i + 1]                      # 执行日
        equity = float(row["equity"])
        if equity <= 0:
            continue
        cashes.append(1.0 - (equity - float(row["cash"])) / equity)
        held = by_date.get(str(row["date"]))
        actual = {}
        if held is not None and len(held):
            actual = dict(zip(held["code"].tolist(),
                              (held["value"] / equity).tolist()))
        names = set(w) | set(actual)
        devs.append(0.5 * sum(abs(actual.get(c, 0.0) - float(w.get(c, 0.0)))
                              for c in names))
        gaps.append(sum(max(0.0, float(w[c]) - actual.get(c, 0.0))
                        for c in w if actual.get(c, 0.0) > 0.0))
    med = lambda x: round(float(np.median(x)) * 100.0, 4) if x else None  # noqa: E731
    return {
        "partial_gap_pct": med(gaps),
        "weight_dev_pct": med(devs),
        "cash_pct": med(cashes),
        "n_days": int(len(devs)),
    }


def _run_layers_with_nav(sigs: list[dict], panel: Panel, cfg: Config
                         ) -> tuple[list[dict], list[pd.DataFrame | None], list[dict]]:
    """逐层跑 run_backtest，同时留下 nav 与配不齐统计（诊断整手残差要用）。"""
    out: list[dict] = []
    navs: list[pd.DataFrame | None] = []
    fills: list[dict] = []
    for k, sig in enumerate(sigs):
        if not sig:
            out.append({"layer": k + 1, "ann_return": float("nan"), "n_trades": 0,
                        "n_rebalance": 0})
            navs.append(None)
            fills.append({"weight_dev_pct": None, "cash_pct": None, "n_days": 0})
            continue
        r = run_backtest(panel, sig, cfg)
        s = M.to_series(r.nav)
        out.append({
            "layer": k + 1,
            "ann_return": float(M.ann_return(s)),
            "total_return": float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else float("nan"),
            "n_trades": int(len(r.trades)),
            "n_rebalance": int(len(sig)),
        })
        navs.append(r.nav)
        fills.append(_fill_stats(r, sig))
    return out, navs, fills


def run_layer_backtests(sigs: list[dict], panel: Panel, cfg: Config) -> list[dict]:
    """逐层跑 run_backtest，返回每层的指标。

    层内等权、月度再平衡；成本/滑点/涨跌停/T+1 全部由引擎负责（与 S2 一致）。
    单只权重上限设为 1.0 —— 分层研究要的是因子暴露，不该被 S2 的 5% 上限改写结构。
    """
    return _run_layers_with_nav(sigs, panel, cfg)[0]


def build_full_xsec_signal(fd: FactorData, ends: list[int]) -> dict[str, dict[str, float]]:
    """全截面等权信号（不切层）：{日期: {代码: 1/n}}。

    只为诊断"5 层合并 vs 全截面"的差从哪来 —— 它和分层走**同一个引擎、同一套成本、
    同一个截面 mask**，唯一差别是资金切成 5 份各自配一层，还是一份配全截面。
    """
    sig: dict[str, dict[str, float]] = {}
    for i in ends:
        ok = cross_section_mask(fd, i)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        w = 1.0 / len(idx)
        sig[fd.dates[i]] = {fd.codes[j]: w for j in idx}
    return sig


def _monthly_returns(nav: pd.DataFrame | pd.Series) -> pd.Series:
    """净值 → 月度收益序列（按自然月末取最后一个交易日）。"""
    s = M.to_series(nav)
    idx = pd.to_datetime(pd.Index(s.index))
    g = pd.Series(s.to_numpy(), index=idx).groupby(idx.to_period("M")).last()
    return g.pct_change().dropna()


def lot_residual_stats(layer_navs: list[pd.DataFrame | None],
                       bench_nav: pd.DataFrame, fills: list[dict]) -> dict:
    """整手残差诊断（审计-B5-02）：把「5 层合并 vs 全截面等权」的差量化到 pp。

    两块指标，口径分开写清楚，避免再出现"把股票池口径差当成整手残差"的标签错误：

    ① 组合层月差（pp）：5 层各 1/5 资金合并的净值 vs 全截面等权净值，逐月差。
       同引擎、同成本、同 mask，差的唯一来源是"资金切分 + 整手取整 + 可交易性过滤"。
       给 mean|diff| / 中位 / p90 与 signed（带符号）年化。
    ② 层内配不齐（%，来自 `_fill_stats`）：部分成交股的未达目标缺口、整体权重偏差
       ½Σ|实际−目标|、未投入现金占比，各期取中位数；5 层的最小~最大值就是"每层留多少残差"。

    注意 ① 的绝对值很小（0.1pp/月量级），**不能**拿它去解释报告里 2pp 级别的差 ——
    那个级别来自股票池口径（全截面引擎 vs 全市场 hfq 基准），与本函数无关。
    """
    ok = [x for x in layer_navs if x is not None]
    if not ok:
        return {"available": False, "reason": "no_layer_nav"}
    eq = None
    for nv in ok:
        s = pd.Series(nv["equity"].to_numpy(dtype="float64"), index=nv["date"].tolist())
        eq = s if eq is None else eq.add(s, fill_value=np.nan)
    merged = pd.DataFrame({"date": eq.index, "equity": (eq / len(ok)).to_numpy()})

    m_layer = _monthly_returns(merged)
    m_bench = _monthly_returns(bench_nav)
    j = m_layer.index.intersection(m_bench.index)
    diff = (m_layer.loc[j] - m_bench.loc[j])
    if not len(diff):
        return {"available": False, "reason": "no_overlap_month"}

    def _rng(key):
        v = [f[key] for f in fills if f.get(key) is not None]
        return (round(min(v), 4), round(max(v), 4)) if v else (None, None)

    dev_lo, dev_hi = _rng("weight_dev_pct")
    gap_lo, gap_hi = _rng("partial_gap_pct")
    cash_lo, cash_hi = _rng("cash_pct")
    return {
        "available": True,
        "n_months": int(len(diff)),
        "mean_abs_pp": round(float(diff.abs().mean() * 100.0), 4),
        "median_abs_pp": round(float(diff.abs().median() * 100.0), 4),
        "p90_abs_pp": round(float(diff.abs().quantile(0.90) * 100.0), 4),
        "mean_signed_pp": round(float(diff.mean() * 100.0), 4),
        "signed_ann_pp": round(float(diff.mean() * 12.0 * 100.0), 4),
        "merged_ann": round(float(M.ann_return(M.to_series(merged))), 6),
        "bench_ann": round(float(M.ann_return(M.to_series(bench_nav))), 6),
        "layer_partial_gap_pct": [f.get("partial_gap_pct") for f in fills],
        "partial_gap_min_pct": gap_lo,
        "partial_gap_max_pct": gap_hi,
        "layer_weight_dev_pct": [f.get("weight_dev_pct") for f in fills],
        "weight_dev_min_pct": dev_lo,
        "weight_dev_max_pct": dev_hi,
        "layer_cash_pct": [f.get("cash_pct") for f in fills],
        "cash_min_pct": cash_lo,
        "cash_max_pct": cash_hi,
        "definition": "① 5 层各 1/5 资金合并 vs 全截面等权，逐月差（pp）；"
                      "② 部分成交股未达目标缺口（执行日，各期中位数）；"
                      "③ 整体权重偏差 ½Σ|实际−目标|；④ 未投入现金占比",
    }


# ---------------------------------------------------------------- 单段评估

def evaluate_segment(factors: dict[str, np.ndarray], fd: FactorData,
                     i0: int, i1: int, cfg_template: Config,
                     n_layers: int = N_LAYERS) -> dict:
    """在一段区间上做 IC + 分层回测（因子定义不随段变化）。"""
    dates = fd.dates[i0:i1]
    panel = slice_panel(fd.panel, i0, i1)

    # 截面日一律用**全局行索引**：因子矩阵、mask、前瞻收益都按全局对齐，
    # 只有面板切片是段内的。混用相对索引会取错截面（踩过）。
    ends_all = [i for i in month_end_indices(fd.dates) if i0 <= i < i1]
    cfg = _layer_cfg(fd, i0, i1, cfg_template)

    # 前瞻收益按全样本算一次（末截面之后还要有下一期），各因子共用；
    # 但只有"退出日 ≤ 本段末交易日"的期才允许计入 IC（B5-01）。
    ends_global = month_end_indices(fd.dates)
    fwd = forward_returns(fd.panel.open_hfq, ends_global)
    usable, dropped = ic_periods(fd, i0, i1)

    # 诊断基准：全截面等权（同引擎同成本），只用于拆解整手残差
    bench_sig = build_full_xsec_signal(fd, ends_all)
    bench = run_backtest(panel, bench_sig, cfg) if bench_sig else None
    # 全市场 hfq 等权（S2 的基准口径，不做任何交易/成本假设）——
    # 它与"全截面引擎"的差就是股票池/资金切分口径差，不是整手残差（B5-02）
    market_ew = M.equal_weight_market(panel)

    res = {}
    for name, fmat in factors.items():
        ics = []
        for p in usable:
            i, k = p["idx"], p["k"]
            ok = cross_section_mask(fd, i)
            ics.append(rank_ic(fmat[i][ok], fwd[k][ok]))

        sigs = build_layer_signals(fmat, fd, ends_all, n_layers)
        layers, navs, fills = _run_layers_with_nav(sigs, panel, cfg)
        afford = affordability_stats(sigs, panel, cfg.init_cash, lot=cfg.lot_size)
        anns = [x["ann_return"] for x in layers]
        q1, q5 = anns[0], anns[-1]
        long_short = (q5 - q1) if (np.isfinite(q1) and np.isfinite(q5)) else float("nan")
        finite = [a for a in anns if np.isfinite(a)]

        res[name] = {
            **ic_summary(ics),
            "ic_series": [None if not np.isfinite(x) else round(float(x), 6) for x in ics],
            "layers": layers,
            "lot_residual": (lot_residual_stats(navs, bench.nav, fills)
                             if bench is not None else {"available": False}),
            "long_short_ann": float(long_short),
            "layer_ann": anns,
            "monotonic": bool(len(finite) == len(anns) and len(anns) >= 2 and all(
                finite[i] <= finite[i + 1] for i in range(len(finite) - 1))),
            # 整手可行性（各层中位数）：太低说明本金不足、分层已退化成价格分层
            "affordability_by_layer": [round(x, 6) if np.isfinite(x) else None
                                       for x in afford],
        }
    res["_meta"] = {"start": dates[0], "end": dates[-1], "dates": len(dates),
                    "codes": len(fd.codes), "n_cross_sections": len(ends_all),
                    # B5-01：IC 期数以"不跨段"为准，被丢的期连同原因全部留痕
                    "n_ic_periods": len(usable),
                    "ic_periods_dropped": dropped,
                    "full_xsec": ({"ann_return": round(float(M.ann_return(
                        M.to_series(bench.nav))), 6),
                        "n_trades": int(len(bench.trades)),
                        "n_rebalance": int(len(bench_sig))} if bench is not None else None),
                    # 全市场 hfq 等权（S2 基准口径）：与 full_xsec 的差 = 股票池/资金切分口径差
                    "market_ew": {"ann_return": round(float(M.ann_return(market_ew)), 6),
                                  "definition": "全市场 hfq 等权日收益均值"
                                                "（metrics.equal_weight_market，"
                                                "含退市股、停牌计 0 收益，无成本）"}}
    return res



# ---------------------------------------------------------------- 产物

def _git_commit() -> str:
    """取当前 HEAD 短哈希；取不到写 unknown（不编造）。"""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _layer_shape(layer_ann: list[float]) -> dict:
    """分层形状（只描述数字，不下结论）：最高层 / 最低层 / 最大相邻落差。"""
    finite = [(k, a) for k, a in enumerate(layer_ann) if np.isfinite(a)]
    if len(finite) < 2:
        return {}
    kmax, amax = max(finite, key=lambda x: x[1])
    kmin, amin = min(finite, key=lambda x: x[1])
    drops = [(finite[j][0], finite[j][1], finite[j + 1][1])
             for j in range(len(finite) - 1)]
    kd, a_hi, a_lo = max(drops, key=lambda x: x[1] - x[2])
    return {"max_layer": f"Q{kmax + 1}", "max_ann": float(amax),
            "min_layer": f"Q{kmin + 1}", "min_ann": float(amin),
            "max_adjacent_drop": {"from": f"Q{kd + 1}", "to": f"Q{kd + 2}",
                                  "pp": float((a_hi - a_lo) * 100.0)}}


def _readings(dev: dict, ho: dict) -> list[str]:
    """按数字自动生成的"怎么读"（B5-07：措辞随数字走，不写死的形容词）。

    审计-B5 指出三处措辞与数字不符，这里全部改成从 layer_ann 直接算：
      ① ep 不是 U 形（峰在中间偏高层、谷在 Q2）；② size 在 dev 是后段断崖、谷底在 Q5，
      不是"中间塌陷"（真中间塌陷只在 holdout 出现，11 期不足以下结论）；
      ③ bp 的 |t|>3 不是唯一，动量 |t|≈4.1 同样过线。
    """
    out = []
    ep_s = _layer_shape(dev["ep"]["layer_ann"]) if "ep" in dev else {}
    if ep_s:
        out.append(
            f"ep：峰在 {ep_s['max_layer']}（{ep_s['max_ann']:.2%}）、谷在 {ep_s['min_layer']}"
            f"（{ep_s['min_ann']:.2%}），中间层高于两端 —— 形状**不是 U 形**；"
            f"t={dev['ep']['t_stat']:.2f}（|t|<2），原「ep 不显著/无效」的结论不变，"
            f"但别再用「U 形」描述。")
    sz_s = _layer_shape(dev["size"]["layer_ann"]) if "size" in dev else {}
    sz_h = _layer_shape(ho["size"]["layer_ann"]) if "size" in ho else {}
    if sz_s:
        d = sz_s["max_adjacent_drop"]
        out.append(
            f"size：dev 最大相邻落差在 {d['from']}→{d['to']}（{d['pp']:.2f}pp），"
            f"谷底 {sz_s['min_layer']}（{sz_s['min_ann']:.2%}）—— 是「前高后低、后段断崖」，"
            f"**不是中间塌陷**" +
            (f"；holdout 谷底 {sz_h.get('min_layer')}（{sz_h.get('min_ann', float('nan')):.2%}）"
             f"才是真的中间塌陷，但 11 期 IC 不足以下结论。" if sz_h else "。") +
            "size 的全部解读须带推算市值口径前提。")
    if all(k in dev for k in ("bp", "momentum_20", "momentum_60")):
        out.append(
            f"bp：t={dev['bp']['t_stat']:.2f}"
            + ("，是价值因子里唯一 |t|>3 的" if abs(dev['bp']['t_stat']) > 3 else "，未过 |t|>3")
            + f"，但**不是全表唯一**：momentum_20 |t|={abs(dev['momentum_20']['t_stat']):.2f}、"
            f"momentum_60 |t|={abs(dev['momentum_60']['t_stat']):.2f}，"
            f"且月频 IC 自相关会高估 t 值 —— 只能说「bp 方向与量级值得跟进」，"
            f"不能说「只有 bp 有效」。")
    return out


def build_report(fd: FactorData, factors: dict[str, np.ndarray],
                 args: argparse.Namespace, cfg_template: Config | None = None,
                 data_quality: dict | None = None) -> dict:
    """一次性生成 dev + holdout 两段报告（因子定义只算一遍，两段共用）。

    cfg_template 不给时按 args.cash / args.seed 造一个默认 Config；
    测试与复算可传入自定义模板（改 lot_size / costs 必须能在产物里看见，B5-06）。
    data_quality 为 `data_quality_stats()` 的产物，落进 size_jump_stats 字段（B5-03）。
    """
    bounds = segment_bounds(fd.dates)
    tmpl = cfg_template or Config(start=fd.dates[0], end=fd.dates[-1],
                                  init_cash=args.cash, seed=args.seed)

    segs = {}
    for tag, (i0, i1) in bounds.items():
        segs[tag] = evaluate_segment(factors, fd, i0, i1, tmpl, args.layers)

    factor_keys = [k for k in segs["dev"] if not k.startswith("_")]
    assert factor_keys == [k for k in segs["holdout"] if not k.startswith("_")]

    # 参数一律取实际生效的那份 Config（B5-06：改默认值必须让产物可见）
    eff = _layer_cfg(fd, *bounds["dev"], tmpl)

    return {
        "git_commit": _git_commit(),
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "params": {
            "factors": factor_keys,
            "n_layers": args.layers,
            "rebalance": "monthly（每月最后一个交易日收盘出信号，T+1 开盘成交）",
            "layer_weight": "层内等权",
            "long_short": "Q5 年化 − Q1 年化（算术差，非几何、非可交易多空组合）",
            "init_cash_note": "每层约 950 只，本金须大到一手成本可忽略；"
                              "产物里 affordability_by_layer 是各层整手可行性中位数，应接近 1",
            "init_cash": float(eff.init_cash),
            "seed": eff.seed,
            # —— 以下四项旧版缺失或写死，改默认值会静默改变全部分层收益（B5-06）
            "lot_size": int(eff.lot_size),
            "max_positions": int(eff.max_positions),
            "max_weight": float(eff.max_weight),
            "costs": {k: float(v) for k, v in asdict(eff.costs).items()},
            "override_note": "分层回测的 max_positions=全部股票数、max_weight=1.0 "
                             "是对 S2 默认（50 / 0.10）的有意覆盖：层内要全持有、层内等权。"
                             "lot_size/costs/seed/init_cash 原样继承模板。",
            "min_listed_days": MIN_LISTED_DAYS,
            "min_xsec": MIN_XSEC,
            "dev_end": DEV_END,
            "holdout_start": HOLDOUT_START,
            "factor_defs": {
                "momentum_20": "close_hfq(T-1)/close_hfq(T-21)-1",
                "momentum_60": "close_hfq(T-1)/close_hfq(T-61)-1",
                "ep": "1/peTTM(T-1)",
                "bp": "1/pbMRQ(T-1)",
                "size": "log(close(T-1)*volume(T-1)/(turn(T-1)/100))",
            },
        },
        "data": fd.source,
        "size_jump_stats": (data_quality or {}).get("size_jump", {}),
        "data_quality": data_quality or {},
        "segments": segs,
        "layer_shape": {tag: {k: _layer_shape(v["layer_ann"])
                              for k, v in segs[tag].items() if not k.startswith("_")}
                        for tag in ("dev", "holdout")},
        "readings": _readings(segs["dev"], segs["holdout"]),
        "size_source": SIZE_SOURCE,
        "validity_note": VALIDITY_NOTE,
    }


def write_report(report: dict, path: Path) -> Path:
    """落盘 JSON。NaN 统一转 null，保证严格 JSON（下游不用猜）。"""
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float):
            return None if not np.isfinite(o) else round(o, 8)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if not np.isfinite(v) else round(v, 8)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def print_summary(report: dict) -> None:
    """控制台摘要（正负号只描述数字，不作策略判断）。"""
    print("=" * 78)
    print(f"因子报告  {report['data']['start']} ~ {report['data']['end']}  "
          f"{report['data']['codes']} 只  {report['data']['dates']} 交易日")
    print(f"dev      : {report['segments']['dev']['_meta']['start']} ~ "
          f"{report['segments']['dev']['_meta']['end']}")
    print(f"holdout  : {report['segments']['holdout']['_meta']['start']} ~ "
          f"{report['segments']['holdout']['_meta']['end']}")
    print("=" * 78)
    for tag in ("dev", "holdout"):
        seg = report["segments"][tag]
        meta = seg["_meta"]
        print(f"\n---------- {tag.upper()} ----------")
        print(f"截面 {meta['n_cross_sections']} 期 / 实际计入 IC {meta['n_ic_periods']} 期"
              f"（丢弃 {len(meta['ic_periods_dropped'])} 期，原因见 _meta.ic_periods_dropped）")
        print(f"{'因子':<14}{'IC均值':>9}{'ICIR':>8}{'t值':>8}{'IC>0':>7}{'期数':>5}"
              f"{'Q1年化':>9}{'Q5年化':>9}{'多空':>9}")
        for k, v in seg.items():
            if k.startswith("_"):
                continue
            print(f"{k:<14}{v['ic_mean']:>9.4f}{v['icir']:>8.3f}{v['t_stat']:>8.2f}"
                  f"{v['ic_pos_ratio']:>7.3f}{v['n_periods']:>5}"
                  f"{v['layer_ann'][0]:>9.2%}{v['layer_ann'][-1]:>9.2%}"
                  f"{v['long_short_ann']:>9.2%}")
        fx = meta.get("full_xsec") or {}
        if fx:
            ew = (meta.get("market_ew") or {}).get("ann_return")
            tail = f"；全市场 hfq 等权基准 {ew:.2%}" if ew is not None else ""
            print(f"全截面等权（同引擎）：年化 {fx['ann_return']:.2%}，"
                  f"{fx['n_rebalance']} 次调仓 / {fx['n_trades']} 笔{tail}")
    r = report["segments"]["dev"]["bp"].get("lot_residual", {})
    if r.get("available"):
        print(f"\n整手残差（dev·bp 为例）：月差 mean|diff|={r['mean_abs_pp']:.2f}pp / "
              f"中位 {r['median_abs_pp']:.2f}pp / p90 {r['p90_abs_pp']:.2f}pp，"
              f"signed 年化 {r['signed_ann_pp']:+.2f}pp；"
              f"每层部分成交缺口 {r['partial_gap_min_pct']:.2f}%~{r['partial_gap_max_pct']:.2f}%，"
              f"未投入现金仅 {r['cash_min_pct']:.3f}%~{r['cash_max_pct']:.3f}%")
    print("\n---------- 怎么读（措辞随数字生成） ----------")
    for line in report.get("readings", []):
        print("· " + line)
    print("\n" + VALIDITY_NOTE)



# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="S3-Q1 因子研究框架（动量/价值/小市值）")
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--factors", default=",".join(FACTOR_DEFS),
                    help=f"逗号分隔，可选 {sorted(FACTOR_DEFS)}")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--cash", type=float, default=1e8,
                    help="初始资金。每层约 950 只，本金太小会让高价股连一手都买不进，"
                         "分层退化成价格分层（1e8 时各层整手可行性 ≈100%%）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    ap.add_argument("--diag-data", action="store_true",
                    help="只跑数据质量统计（turn 缺失率 / 推算股本跳变 / 真实股本对拍）并打印，"
                         "不进回测。对应 VALIDITY_NOTE 与 size_jump_stats 的复现命令")
    args = ap.parse_args(argv)

    if args.diag_data:
        dq = data_quality_stats(args.start, args.end)
        print(json.dumps(dq, ensure_ascii=False, indent=2))
        return 0

    names = [x.strip() for x in args.factors.split(",") if x.strip()]
    print(f"[1/4] 载入数据 {args.start} ~ {args.end} ...", flush=True)
    fd = load_factor_data(args.start, args.end)
    print(f"      {len(fd.codes)} 只 / {len(fd.dates)} 交易日 / "
          f"hfq 缺口 {fd.source['hfq_gap_rows'].get('rows', 0)} 行", flush=True)

    print("[2/4] 数据质量统计（turn 缺失 / 股本跳变 / 对拍）...", flush=True)
    dq = data_quality_stats(args.start, args.end)
    sj = dq["size_jump"]
    print(f"      turn 缺失 {dq['turn_missing']['rate']:.4%}，"
          f"股本跳变 {sj['rows']} 行 / {sj['codes']} 只，"
          f"其中月末 7 日内 {sj['month_end_rows']} 行", flush=True)

    print(f"[3/4] 计算因子 {names} ...", flush=True)
    factors = compute_factors(fd, names)

    print("[4/4] IC + 分层回测（dev / holdout 两段）...", flush=True)
    report = build_report(fd, factors, args, data_quality=dq)
    out = write_report(report, (Path(args.out) / "factors_report.json").resolve())
    print_summary(report)
    print(f"\n产物：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
