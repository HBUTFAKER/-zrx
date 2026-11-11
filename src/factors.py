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

数据口径·市值字段（003R3 后仍缺失，本次上报项）
-----------------------------------------------
cache 宽表**没有**总市值/流通市值字段。`finance.parquet` 的 totalShare/liqaShare
覆盖率仅 0.15%（270 行 / 64 只，全部来自 baostock 那批），不可用；原始日线
（data/raw/daily）本身也只有 close/volume/turn，没有市值。

本模块用**换手率定义式**推算流通市值：

    turn(%) = volume / 流通股本 × 100   ⇒   流通股本 = volume / (turn/100)
    流通市值 = close × 流通股本

这是恒等式推导，不是"换一个估值口径顶替"：误差只来自 turn 的舍入。用
finance 里仅有的真实 liqaShare 做对拍（可对证 2 个样本），相对误差 7e-6 / 2e-6。
风险点如实写在这里：turn 缺失 4.66%（停牌日无成交），且解禁/增发会让推算股本真实跳变
（这与真实市值数据同步跳变，不是误差）。

**该口径已上报总工确认**；若判定不可接受，只需改 `size()` 一个函数的数据来源，
框架其余部分不受影响。产物里 `size_source` 字段固定写 "log(close*volume/(turn/100))"
以便事后追溯。

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
from dataclasses import dataclass, field
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
    "① 分层为层内等权、月度再平衡，未做行业/风格中性化；② IC 用理论开盘价收益，"
    "未扣除涨跌停/停牌造成的不可成交（诚实起见：IC 与分层收益之间允许有口径差）；"
    "③ 市值由换手率推算（宽表无市值字段），解禁/增发会造成真实跳变；"
    "④ 截面已剔除 ST/停牌/缺口/上市不足 60 日；"
    "⑤ holdout 段仅 12 个月、11 期 IC，样本量不足以支撑统计推断，只作记录；"
    "⑥ **层间比较有效，绝对水平不可与全市场等权直接相比**：整手取整会让每层留 2~7% 残差，"
    "实测 5 层净值合并与全截面等权同区间差约 0.2~0.3pp/月（下跌月里残差大的组合被动减仓、亏得少），"
    "这是执行口径差不是因子收益，故只看层间排序与多空差。"
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
    注意：月频 IC 存在自相关，t 值会**高估**显著性，只作量级参考。
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


def run_layer_backtests(sigs: list[dict], panel: Panel, cfg: Config) -> list[dict]:
    """逐层跑 run_backtest，返回每层的指标。

    层内等权、月度再平衡；成本/滑点/涨跌停/T+1 全部由引擎负责（与 S2 一致）。
    单只权重上限设为 1.0 —— 分层研究要的是因子暴露，不该被 S2 的 5% 上限改写结构。
    """
    out = []
    for k, sig in enumerate(sigs):
        if not sig:
            out.append({"layer": k + 1, "ann_return": float("nan"), "n_trades": 0,
                        "n_rebalance": 0})
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
    return out


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
    cfg = Config(start=dates[0], end=dates[-1],
                 init_cash=cfg_template.init_cash,
                 max_positions=len(fd.codes),      # 分层内全部持有，不截断
                 max_weight=1.0,                   # 层内等权，不受 S2 的 5% 上限约束
                 seed=cfg_template.seed,
                 costs=cfg_template.costs)

    # 前瞻收益按全样本算一次（末截面之后还要有下一期），各因子共用
    ends_global = month_end_indices(fd.dates)
    fwd = forward_returns(fd.panel.open_hfq, ends_global)
    pos = {i: k for k, i in enumerate(ends_global)}

    res = {}
    for name, fmat in factors.items():
        ics = []
        for i in ends_all:
            k = pos[i]
            if k >= len(ends_global) - 1:          # 最后一期没有前瞻收益
                continue
            ok = cross_section_mask(fd, i)
            ics.append(rank_ic(fmat[i][ok], fwd[k][ok]))

        sigs = build_layer_signals(fmat, fd, ends_all, n_layers)
        layers = run_layer_backtests(sigs, panel, cfg)
        afford = affordability_stats(sigs, panel, cfg.init_cash)
        anns = [x["ann_return"] for x in layers]
        q1, q5 = anns[0], anns[-1]
        long_short = (q5 - q1) if (np.isfinite(q1) and np.isfinite(q5)) else float("nan")
        finite = [a for a in anns if np.isfinite(a)]

        res[name] = {
            **ic_summary(ics),
            "ic_series": [None if not np.isfinite(x) else round(float(x), 6) for x in ics],
            "layers": layers,
            "long_short_ann": float(long_short),
            "layer_ann": anns,
            "monotonic": bool(len(finite) == len(anns) and len(anns) >= 2 and all(
                finite[i] <= finite[i + 1] for i in range(len(finite) - 1))),
            # 整手可行性（各层中位数）：太低说明本金不足、分层已退化成价格分层
            "affordability_by_layer": [round(x, 6) if np.isfinite(x) else None
                                       for x in afford],
        }
    res["_meta"] = {"start": dates[0], "end": dates[-1], "dates": len(dates),
                    "codes": len(fd.codes), "n_cross_sections": len(ends_all)}
    return res


# ---------------------------------------------------------------- 产物

def _git_commit() -> str:
    """取当前 HEAD 短哈希；取不到写 unknown（不编造）。"""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_report(fd: FactorData, factors: dict[str, np.ndarray],
                 args: argparse.Namespace) -> dict:
    """一次性生成 dev + holdout 两段报告（因子定义只算一遍，两段共用）。"""
    bounds = segment_bounds(fd.dates)
    tmpl = Config(start=fd.dates[0], end=fd.dates[-1], init_cash=args.cash, seed=args.seed)

    segs = {}
    for tag, (i0, i1) in bounds.items():
        segs[tag] = evaluate_segment(factors, fd, i0, i1, tmpl, args.layers)

    factor_keys = [k for k in segs["dev"] if not k.startswith("_")]
    assert factor_keys == [k for k in segs["holdout"] if not k.startswith("_")]

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
            "init_cash": args.cash,
            "seed": args.seed,
            "costs": {"commission_rate": 0.00025, "min_commission": 5.0,
                      "stamp_tax": 0.0005, "transfer_fee": 0.00001, "slippage": 0.001},
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
        "segments": segs,
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
        print(f"\n---------- {tag.upper()} ----------")
        print(f"{'因子':<14}{'IC均值':>9}{'ICIR':>8}{'t值':>8}{'IC>0':>7}{'期数':>5}"
              f"{'Q1年化':>9}{'Q5年化':>9}{'多空':>9}")
        for k, v in seg.items():
            if k.startswith("_"):
                continue
            print(f"{k:<14}{v['ic_mean']:>9.4f}{v['icir']:>8.3f}{v['t_stat']:>8.2f}"
                  f"{v['ic_pos_ratio']:>7.3f}{v['n_periods']:>5}"
                  f"{v['layer_ann'][0]:>9.2%}{v['layer_ann'][-1]:>9.2%}"
                  f"{v['long_short_ann']:>9.2%}")
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
    args = ap.parse_args(argv)

    names = [x.strip() for x in args.factors.split(",") if x.strip()]
    print(f"[1/3] 载入数据 {args.start} ~ {args.end} ...", flush=True)
    fd = load_factor_data(args.start, args.end)
    print(f"      {len(fd.codes)} 只 / {len(fd.dates)} 交易日 / "
          f"hfq 缺口 {fd.source['hfq_gap_rows'].get('rows', 0)} 行", flush=True)

    print(f"[2/3] 计算因子 {names} ...", flush=True)
    factors = compute_factors(fd, names)

    print("[3/3] IC + 分层回测（dev / holdout 两段）...", flush=True)
    report = build_report(fd, factors, args)
    out = write_report(report, (Path(args.out) / "factors_report.json").resolve())
    print_summary(report)
    print(f"\n产物：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
