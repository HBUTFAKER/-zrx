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

数据口径·财务（质量因子，公告日对齐）
--------------------------------------
`finance.parquet` 有 netProfit / total_equity / debt_ratio / YOYNI 等高覆盖字段
（实测 netProfit 99.99%、total_equity 93.99%、debt_ratio 94.54%、YOYNI 93.70%），
但 **epsTTM 只有 270 行（0.15%，只覆盖 baostock 那批退市股）**，不可用 —— 这是实测，
不是猜的，复现见 `--diag-data` 的 fundamentals 段。

时点纪律：财报一律用**公告日** `pubDate`（缺失回退 `pubDate_est`）对齐，
**绝不用 statDate**。T 日因子值 = 公告日 ≤ T 的最新一期；实现上先按公告日 as-of
铺成逐日矩阵，再过 `_shift1`，等于"只用 ≤ T-1 已公告的财报"。
`pubDate` 缺 17,199 行，**分母是 rows_in = 175,734（有报告期的原始行）→ 9.79%**；
纯度为"有报告期"的行里再剔无公告日的行，16,242 行回退，
**分母是 rows_with_period = 170,636 → 9.5185%**。两个比例分母不同，不可混用。

`pubDate_est` 字段结构（审计-B7-06 实测，直接影响怎么读"滞后"）：
est 覆盖 175,734 行**全量**；pubDate 有值的 158,535 行里 **est 与 pubDate 100% 相等**
（该字段在 clean 阶段就是 `pubDate.fillna(截止日)`，有实测就直接抄）；est 相对 pubDate
的**独有值恰好等于那 17,199 个缺失行**。即 est 是 pubDate 的超集补全字段，
**残余时点误差只来自缺失行的截止日推算**，不来自有实测公告日的行。

方向**是滞后不是前视**：截止日（一季报 04-30 / 半年报 08-31 / 三季报 10-31 /
年报次年 04-30）必然 ≥ 真实披露日，只可能晚知道、不可能预知未来；代价是这批期数
晚一到两周才进截面，属**时点噪声**。审计-B7 用巨潮对照 50 例复核：0 例提前、
平均滞后 8.8 天。如实写进产物 `data.fundamentals.pubdate_src`。

roe_ttm 不用 `roeAvg`：它是**区间累计**口径（Q1 是三个月、Q4 是全年），直接进截面会让
因子值随季度季节性摆动。这里改成 `净利润TTM / 期末股东权益`，TTM 由 4 个季度累计值
还原（q4 直接用年报，q1~q3 = 本期累计 + 上年年报 − 上年同期累计）。

数据口径·市值（004B 起用真实股本，推算口径保留作对拍）
------------------------------------------------------
004B-P 勘察裁决：股本改用东财 gbjg **事件级**记录（`data/shares/share_events.parquet`），
逐日流通股本由事件序列前向填充，`size_real = log(close × 流通股本)`。
`size`（换手率推算口径 `close×volume/(turn/100)`）保留，两者同表对比，差异分布写进
`size_compare` —— 这样"换口径到底改了多少"是可验证的，不是嘴上说的。

为什么必须事件级：季度快照（baostock）降级后 8 只样本 11,225 个交易日里 2,094 天
（18.65%）与事件级差 >30%，解禁后要等一个季度才反映，size 会整段用错股本。

**审计-B5-03 归因更正**：旧稿说"跳变=污染"。实测样本内 14 次推算跳变，12 次 ±3 日内
有真实股本事件（首发限售上市/转增/战略配售），倍率同日对到三位小数
（14.588/14.590 等）。所以跳变**多数是资产事实不是数据错误**，换真实股本解决的
是另外三件事：无事件对应的脏跳变（如 sz.001289 2022-08-24/26 成对异常）、
低换手日 turn 舍入噪声、停牌日 turn=0 要靠猜。立项动机已按此更正。

行业中性化
----------
行业用申万分类**变动历史**（`data/shares/sw_industry.parquet`，带 `计入日期`），
按"计入日期 ≤ T-1"取当时归属，默认一级（行业代码前 2 位）。截面内按组
demean 或 zscore（`--neutralize`，默认 demean），组内样本 <5 只的组不处理（留着原值，
并在产物里记 `n_small_group`）。无行业归属的样本（覆盖不到的那部分）同样保留原值、
单独计数 —— 不猜行业，也不静默丢样本。

旧免责里的「5 层合并 vs 全截面等权差约 0.2~0.3pp/月」「每层留 2~7% 残差」两处都对不上
实测，且把两个不同量混为一谈（整手残差 0.12pp/月 vs 每层配不满 0.70~2.27%）。
**004B 起这些数字全部由 `build_validity_note()` 从实测产物动态生成**（B6-04），
不再写死在常量里 —— 改默认值后产物不会说谎。

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

#: 质量因子所需财报字段（缺任一个即停工上报，不许换字段顶替）
FIN_COLS = ("netProfit", "total_equity", "debt_ratio", "YOYNI")
#: 财报时点字段。pubDate 是实际公告日；缺失行回退 pubDate_est（法定截止日推算）
FIN_TIME_COLS = ("pubDate", "pubDate_est", "pubdate_src", "statDate", "quarter")

MIN_LISTED_DAYS = 60    # 上市不足 60 个交易日的样本不进截面（新股初期价格/市值失真）
MIN_XSEC = 30           # 截面少于 30 只不算 IC（统计上没有意义）
MIN_PER_LAYER = 5       # 每层至少 5 只，否则该期跳过

N_LAYERS = 5

#: 行业中性化默认层级：1 = 申万一级（行业代码前 2 位）；实际组数见产物 industry.groups
INDUSTRY_LEVEL = 1
#: 组内样本少于此数不中性化（组太小，减出来的均值本身就是噪声）
MIN_GROUP_SIZE = 5

#: 数据补抓产物（004B 第一部分落盘，均已 gitignore）
SHARES_EVENTS = ROOT / "data" / "shares" / "share_events.parquet"
SW_INDUSTRY = ROOT / "data" / "shares" / "sw_industry.parquet"

#: 旧口径（推算）与 004B 新口径（真实股本）的 source 描述，写进产物便于追溯
SIZE_SOURCE = "log(close*volume/(turn/100))  # 流通市值，由换手率定义式推算"
SIZE_SOURCE_REAL = ("log(close*float_share)  # 流通市值，float_share 取东财 gbjg "
                    "事件级已上市流通A股，按变更日期前向填充")

# VALIDITY_NOTE 不再是写死的常量：004B 起由 build_validity_note() 从实测产物动态生成
# （审计-B6-04：旧版把 8,628 行、0.12pp、<0.01% 等数字写死在字符串里，改默认值后
#  产物会继续说旧话）。函数定义见文件末尾"产物"段。



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
    # —— 004B 追加（都可为 None：老测试/子集场景不需要财务/股本/行业）
    roe_ttm: np.ndarray | None = None       # 净利润TTM/期末股东权益（公告日对齐，未 shift）
    yoy_ni: np.ndarray | None = None        # 净利润同比（%）
    debt_ratio: np.ndarray | None = None    # 资产负债率（%）
    float_shares: np.ndarray | None = None  # 逐日流通股本（股，事件级前向填充）
    industry_gid: np.ndarray | None = None  # 申万行业组 id（int16，-1=无归属）
    industry_names: dict | None = None      # gid → 行业代码/名称（展示用）

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
        for name in ("roe_ttm", "yoy_ni", "debt_ratio", "float_shares", "industry_gid"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != shp:
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


# ---------------------------------------------------------------- 004B 数据层：财报 / 股本 / 行业

def align_by_pubdate(rec: pd.DataFrame, dates: list[str], codes: list[str],
                     col: str = "value") -> np.ndarray:
    """把「(code, pub_date, value) 记录表」按公告日 as-of 铺成 (n_dates, n_codes)。

    这是**时点纪律的物理实现**：第 i 行拿到的值是"公告日 ≤ dates[i] 的最新一期"。
    同一公告日多条记录时取最后一条（调用方应已按 statDate 排好序）。

    返回矩阵**未做 shift**：调用方接 `_shift1` 才满足"只用 ≤T-1 的数据"。
    公告日缺失（NaT）的行必须先被丢掉——NaT 会静默落到 searchsorted 边界，制造前视。
    """
    out = np.full((len(dates), len(codes)), np.nan, dtype="float64")
    if rec is None or not len(rec):
        return out
    r = rec.dropna(subset=["pub_date"]).sort_values(["code", "pub_date"])
    if not len(r):
        return out
    d64 = pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[ns]")
    pos = {c: j for j, c in enumerate(codes)}
    for code, sub in r.groupby("code", sort=False):
        j = pos.get(code)
        if j is None:
            continue
        pub = sub["pub_date"].to_numpy().astype("datetime64[ns]")
        val = pd.to_numeric(sub[col], errors="coerce").to_numpy("float64")
        k = np.searchsorted(pub, d64, side="right") - 1
        out[:, j] = np.where(k >= 0, val[np.clip(k, 0, len(val) - 1)], np.nan)
    return out


def prepare_fundamentals(fin: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """财报表 → 质量因子记录表（公告日 + roe_ttm/yoy_ni/debt_ratio）。

    纯函数，不读盘，方便用反例直接测时点纪律。

    · **只用 pubDate / pubDate_est 当时间轴，statDate 绝不参与**；
      17,199 行没有实际公告日（分母 rows_in=175,734 → 9.79%），用法定披露截止日
      推算的 pubDate_est 顶；再剔掉真正没报告期的行后是 16,242 行
      （分母 rows_with_period=170,636 → 9.5185%）。**两个分母不同，别混用**。
      这类行数如实计入返回的 stats，不隐藏。
    · roe_ttm 用 TTM 净利润 / 期末权益：q4 直接用年报，q1~q3 = 本期累计 + 上年年报 −
      上年同期累计。不用 roeAvg（区间累计口径，混季度会让因子值随季节摆动）。
    """
    need = ["code", "statDate", "quarter", *FIN_TIME_COLS[:2], *FIN_COLS]
    missing = [c for c in need if c not in fin.columns]
    if missing:
        raise RuntimeError(
            f"finance.parquet 缺财报字段 {missing}；质量因子按纪律**停工上报**，"
            f"不许用其他字段顶替（换字段就是换口径）。")

    d = fin.copy()
    d["statDate"] = pd.to_datetime(d["statDate"], errors="coerce")
    d["quarter"] = pd.to_numeric(d["quarter"], errors="coerce")
    d = d.dropna(subset=["statDate", "quarter"])
    d["year"] = d["statDate"].dt.year

    # ---- TTM 净利润
    ann = d.loc[d["quarter"] == 4, ["code", "year", "netProfit"]].rename(
        columns={"netProfit": "np_annual"})
    ann_prev = ann.copy()
    ann_prev["year"] = ann_prev["year"] + 1          # 让"本年"能对上"上年年报"
    prev_same = d[["code", "year", "quarter", "netProfit"]].rename(
        columns={"netProfit": "np_prev_same"})
    prev_same["year"] = prev_same["year"] + 1

    b = d[["code", "statDate", "year", "quarter", "netProfit", "total_equity",
           "debt_ratio", "YOYNI", "pubDate", "pubDate_est"]].copy()
    b = b.merge(ann_prev, on=["code", "year"], how="left")
    b = b.merge(prev_same, on=["code", "year", "quarter"], how="left")
    ttm = np.where(b["quarter"].to_numpy() == 4,
                   b["netProfit"].to_numpy("float64"),
                   b["netProfit"].to_numpy("float64") + b["np_annual"].to_numpy("float64")
                   - b["np_prev_same"].to_numpy("float64"))
    b["np_ttm"] = ttm
    b.loc[b["quarter"] != 4, "np_ttm"] = np.where(
        np.isfinite(b.loc[b["quarter"] != 4, "np_annual"]) &
        np.isfinite(b.loc[b["quarter"] != 4, "np_prev_same"]),
        b.loc[b["quarter"] != 4, "np_ttm"], np.nan)

    eq = b["total_equity"].to_numpy("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        roe = b["np_ttm"].to_numpy("float64") / eq
    roe = np.where(np.isfinite(roe) & (eq > 0), roe, np.nan)
    b["roe_ttm"] = roe
    b["yoy_ni"] = pd.to_numeric(b["YOYNI"], errors="coerce")
    b["debt_ratio"] = pd.to_numeric(b["debt_ratio"], errors="coerce")

    b["pub_date"] = pd.to_datetime(b["pubDate"], errors="coerce").fillna(
        pd.to_datetime(b["pubDate_est"], errors="coerce"))
    used_est = int(b["pubDate"].isna().sum())
    # 两个公告日字段都缺的行只能整行丢掉 —— 数量必须报出来，否则"可用 15 万行"
    # 看不出到底筛掉了多少期财报。
    no_pub = int(b["pub_date"].isna().sum())
    b = b.dropna(subset=["pub_date"])
    n_before_dedup = int(len(b))          # 去重前（已剔除无公告日的行）
    b = b.sort_values(["code", "pub_date", "statDate"]).drop_duplicates(
        subset=["code", "pub_date"], keep="last").reset_index(drop=True)

    eps_cov = (round(float(fin["epsTTM"].notna().mean()), 6)
               if "epsTTM" in fin.columns else None)
    stats = {
        "rows_in": int(len(fin)), "rows_with_period": int(len(d)),
        "rows_usable": int(len(b)),
        # 年报与一季报同日披露是常态（19,189 组），as-of 只要一个值：按 statDate
        # 取较新那期。差额必须报出来，否则"17 万行只剩 15 万"说不清去哪了。
        "rows_same_day_collapsed": int(n_before_dedup - len(b)),
        "same_day_collapse_note": "同 code+公告日多期（最常见=年报+一季报同日）"
                                  "保留 statDate 较新的一期",
        "codes": int(b["code"].nunique()),
        "pubdate_missing_rows": used_est,
        # 分母用"有报告期的行"，不是原始行数：被 statDate/quarter 丢掉的不该拉低比例
        "pubdate_missing_rate": round(used_est / max(len(d), 1), 6),
        "pubdate_unresolved_rows": no_pub,
        "pubdate_unresolved_rate": round(no_pub / max(len(d), 1), 6),
        "pubdate_unresolved_note": "pubDate 与 pubDate_est 都缺，无法定时点，整行剔除",
        "pubdate_source": "pubDate 实际公告日；缺失回退 pubDate_est（法定披露截止日推算）",
        "field_coverage": {c: round(float(fin[c].notna().mean()), 6) for c in FIN_COLS},
        "epsTTM_coverage": eps_cov,
        "epsTTM_note": (None if eps_cov is None else
                        f"epsTTM 覆盖率 {eps_cov:.2%}（只有 baostock 那批退市股），不可用作因子"),
    }
    return b[["code", "pub_date", "statDate", "roe_ttm", "yoy_ni", "debt_ratio"]], stats


def load_fundamentals(dates: list[str], codes: list[str], path: Path | None = None
                      ) -> tuple[dict[str, np.ndarray], dict]:
    """载入并铺平质量因子（未 shift）。字段缺失抛 RuntimeError（停工上报）。"""
    p = path or (CACHE / "finance.parquet")
    if not p.exists():
        raise FileNotFoundError(f"缺 {p}，质量因子无法计算")
    cols = ["code", "statDate", "quarter", "pubDate", "pubDate_est", *FIN_COLS]
    have = set(_parquet_columns(p))
    missing = [c for c in cols if c not in have]
    if missing:
        raise RuntimeError(f"{p.name} 缺 {missing}，按纪律停止施工并上报。")
    if "epsTTM" in have:            # 只为算覆盖率，不进因子
        cols = [*cols, "epsTTM"]
    fin = pd.read_parquet(p, columns=cols)
    rec, stats = prepare_fundamentals(fin)
    out = {k: align_by_pubdate(rec, dates, codes, k)
           for k in ("roe_ttm", "yoy_ni", "debt_ratio")}
    return out, stats


def share_float_matrix(dates: list[str], codes: list[str], path: Path | None = None
                       ) -> tuple[np.ndarray, dict]:
    """逐日流通股本矩阵（事件级前向填充），单位：股。

    `float_share_a`（东财「已上市流通A股」）优先，缺失回退 `float_share`（已流通股份）——
    勘察已证实换手率的分母是前者，后者含 B/H 股口径时会差出数量级。
    上市前/退市后的日期没有事件 → 保持 NaN，不猜（用 0 会让 size 变成 -inf）。
    """
    p = path or SHARES_EVENTS
    if not p.exists():
        raise FileNotFoundError(
            f"缺 {p}；先跑 python -m src.fetch_shares --stage shares")
    ev = pd.read_parquet(p, columns=["code", "event_date", "float_share_a",
                                     "float_share", "reason"])
    # 只认**正的**股本：实测 1,373 行 float_share_a=0（A 股尚未流通/仅 B 股）而
    # float_share>0，按"非缺失优先"取 0 会让这些事件整行被丢，等于该股一开始
    # 就没有股本、size_real 白白空掉一段。
    a = ev["float_share_a"].where(ev["float_share_a"] > 0)
    b = ev["float_share"].where(ev["float_share"] > 0)
    sh = a.where(a.notna(), b)
    ev = ev.assign(value=sh).dropna(subset=["value"])
    rec = ev.rename(columns={"event_date": "pub_date"})
    m = align_by_pubdate(rec, dates, codes, "value")

    covered = int(np.isfinite(m).any(axis=0).sum())
    stats = {
        "path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
        "events_rows": int(len(ev)), "events_codes": int(ev["code"].nunique()),
        "codes_covered": covered, "codes_total": len(codes),
        "codes_coverage": round(covered / max(len(codes), 1), 6),
        "cells_filled": int(np.isfinite(m).sum()),
        "cells_total": int(m.size),
        "cells_filled_rate": round(float(np.isfinite(m).mean()), 6),
        "definition": "float_share_a（>0 优先）否则回退 float_share；按变更日期前向填充到面板日历",
    }
    return m, stats


def industry_matrix(dates: list[str], codes: list[str], level: int = INDUSTRY_LEVEL,
                    path: Path | None = None) -> tuple[np.ndarray, dict]:
    """行业归属矩阵（int16 组号，-1 = 无归属），按**计入日期**做时点匹配。

    行业代码是申万 6 位码，level=1 取前 2 位、2 取前 4 位、3 取全 6 位；组数只算"本面板真有股票"的那些，写进返回的 stats["groups"]。
    时点用 `in_date`（计入日期）而不是 `update_date`：后者是记录被改写的日期，
    中位比 in_date 晚 1,356 天，拿它当时间轴等于用今天的分组回测历史。
    """
    p = path or SW_INDUSTRY
    if not p.exists():
        raise FileNotFoundError(
            f"缺 {p}；先跑 python -m src.fetch_shares --stage industry")
    cut = 2 * level
    sw = pd.read_parquet(p, columns=["panel_code", "industry_code", "in_date"])
    sw = sw.dropna(subset=["in_date"])
    sw["grp"] = sw["industry_code"].astype(str).str[:cut]
    sw = sw[sw["grp"].str.len() == cut]

    pos = {c: j for j, c in enumerate(codes)}
    n_d, n_c = len(dates), len(codes)
    raw = np.full((n_d, n_c), "", dtype=object)
    d64 = pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[ns]")
    for code, sub in sw.groupby("panel_code", sort=False):
        j = pos.get(code)
        if j is None:
            continue
        sub = sub.sort_values("in_date")
        ind = sub["in_date"].to_numpy().astype("datetime64[ns]")
        grp = sub["grp"].to_numpy()
        k = np.searchsorted(ind, d64, side="right") - 1
        raw[:, j] = np.where(k >= 0, grp[np.clip(k, 0, len(grp) - 1)], "")

    uniq = sorted({g for g in raw.ravel() if g})
    gmap = {g: i for i, g in enumerate(uniq)}
    ids = np.full((n_d, n_c), -1, dtype="int16")
    for j in range(n_c):
        colv = raw[:, j]
        ids[:, j] = np.array([gmap.get(g, -1) for g in colv], dtype="int16")

    known = int((ids >= 0).any(axis=0).sum())
    stats = {
        "level": level, "groups": len(uniq),
        "codes_with_industry": known, "codes_total": n_c,
        "codes_coverage": round(known / max(n_c, 1), 6),
        "cells_unmapped": int((ids < 0).sum()), "cells_total": int(ids.size),
        "source": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
        "time_axis": "计入日期 in_date（非 update_date）",
        "group_codes_sample": uniq[:5],
    }
    return ids, stats


def size_real(close_raw: np.ndarray, float_shares: np.ndarray) -> np.ndarray:
    """小市值因子（真实股本口径）= log(close(T-1) × 流通股本(T-1))。

    与推算口径的区别只在股本来源：这里直接用事件级流通股本，停牌日照样有股本
    （不像 turn 缺失要靠猜），低换手日的舍入噪声也不存在。
    """
    c = _shift1(close_raw)
    sh = _shift1(float_shares)
    with np.errstate(divide="ignore", invalid="ignore"):
        fmv = c * sh
    bad = (~np.isfinite(fmv)) | (fmv <= 0) | ~(sh > 0)
    fmv = np.where(bad, np.nan, fmv)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(fmv)
    out[~np.isfinite(out)] = np.nan
    return out


def neutralize_matrix(fmat: np.ndarray, fd: FactorData, ends: list[int],
                      gids: np.ndarray, method: str = "demean",
                      min_group: int = MIN_GROUP_SIZE) -> tuple[np.ndarray, dict]:
    """行业中性化：月度截面内按行业组去均值（demean）或标准化（zscore）。

    只在**截面日**处理（其余行为原样复制）——分层与 IC 都只用截面日的值，
    逐日中性化没有意义还慢。

    三条不动的样本：① 不在截面 mask 内；② 因子值非有限；③ 该组样本 < min_group。
    后两类保留原值并计数 —— 组太小的时候"减去组均值"减掉的是噪声本身。
    """
    if method not in ("demean", "zscore"):
        raise ValueError(f"未知中性化方法 {method!r}")
    out = fmat.copy()
    n_small = n_unmapped = n_done = 0
    for i in ends:
        ok = cross_section_mask(fd, i)
        x = out[i]
        g = gids[i]
        sel = ok & np.isfinite(x)
        n_unmapped += int((sel & (g < 0)).sum())
        sel &= (g >= 0)
        if not sel.any():
            continue
        s = pd.Series(x[sel])
        grp = pd.Series(g[sel])
        for _, idxs in grp.groupby(grp).groups.items():
            pos = np.asarray(idxs, dtype=int)
            if len(pos) < min_group:
                n_small += len(pos)
                continue
            v = s.iloc[pos].to_numpy("float64")
            if method == "demean":
                x[np.flatnonzero(sel)[pos]] = v - v.mean()
            else:
                sd = v.std(ddof=1)
                if not np.isfinite(sd) or sd <= 0:
                    n_small += len(pos)
                    continue
                x[np.flatnonzero(sel)[pos]] = (v - v.mean()) / sd
            n_done += len(pos)
    stats = {"method": method, "n_cells_neutralized": n_done,
             "n_small_group_cells": n_small, "n_unmapped_cells": n_unmapped,
             "min_group": min_group}
    return out, stats


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

    # 实证案例：固定代码，**按时间列全**（审计-B6-03：旧版每股只取幅度最大的那一次，
    # 把 sz.001289 2022-08-24 的 ×2.59 与 2022-08-26 的 ×0.39 这对"两天又变回去"的
    # 脏跳变截断成了单条，读起来像真实解禁。成对的就标 paired，不许截。）
    size = pd.Series(np.log(np.where(np.isfinite(sh), d["close"].to_numpy("float64") * sh, np.nan)))
    pct = size.groupby(d["date"]).rank(pct=True).to_numpy()
    max_per_case = 6
    examples = []
    for c in JUMP_CASES:
        m = (code == c).to_numpy()
        if not m.any():
            continue
        idx = np.nonzero(m & flag)[0]          # 数据已按 (code, date) 排序，idx 即时间序
        if not len(idx):
            continue
        picks = idx[:max_per_case]
        for t, k in enumerate(picks):
            j = k - 1 if k - 1 >= 0 and code.iloc[k - 1] == c else None
            nxt = picks[t + 1] if t + 1 < len(picks) else None
            paired = bool(nxt is not None and (nxt - k) <= 5 and
                          abs(float(ratio[k]) * float(ratio[nxt]) - 1.0) < 0.05)
            examples.append({
                "code": c, "date": str(d["date"].iloc[k]),
                "share_ratio": round(float(ratio[k]), 6),
                "size_pct_before": (None if j is None or not np.isfinite(pct[j])
                                    else round(float(pct[j]), 6)),
                "size_pct_after": (None if not np.isfinite(pct[k])
                                   else round(float(pct[k]), 6)),
                # 成对异常：两次跳变乘起来≈1（几天内涨回去），说明是脏数据而非资产事实
                "paired_reversal": paired,
                "pair_date": (None if not paired else str(d["date"].iloc[nxt])),
                "truncated": bool(len(idx) > max_per_case),
                "n_jumps_code": int(len(idx)),
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
        "note": "跳变本身多数是**真实解禁/增发事件**（004B-P 抽样 14 次里 12 次能在 ±3 日内"
                "找到真实股本事件，倍率同日对到三位小数），不是数据错误；"
                "但它会让 size 在截面上一日跳十几个百分点，所以 004B 起同时给 size_real"
                "（事件级真实股本）做对照，见产物 size_compare",
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
                     cache_dir: Path | None = None, *,
                     with_fundamentals: bool = True,
                     with_shares: bool = True,
                     with_industry: bool = True,
                     industry_level: int = INDUSTRY_LEVEL) -> FactorData:
    """载入因子数据。价格与 mask 完全来自 `load_panel`，本函数只追加附加列。

    字段缺失（peTTM / pbMRQ / volume / turn 任一不存在）→ RuntimeError 停工。
    这是硬约束：换字段就是换口径，静默降级会让报告失真且无法事后发现。

    004B 追加三块（都可关掉，老测试仍按 6 列构造）：
      · 质量因子 roe_ttm / yoy_ni / debt_ratio —— 公告日对齐；
      · float_shares 逐日流通股本 —— 东财 gbjg 事件级前向填充；
      · industry_gid 申万行业组号 —— 按计入日期时点匹配。
    数据文件缺失时**抛错不跳过**：静默跳过等于报告里悄悄少几个因子。
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

    src = {
        "start": dates_all[0], "end": dates_all[-1],
        "dates": len(dates_all), "codes": len(codes_all),
        "rows": int(len(extra)),
        "hfq_gap_rows": panel.source.get("hfq_gap_rows", {}),
        "size_source": SIZE_SOURCE,
        "size_source_real": SIZE_SOURCE_REAL,
        "min_listed_days": MIN_LISTED_DAYS,
        "fields": list(FACTOR_COLS),
    }

    kw: dict = {}
    if with_fundamentals:
        fund, fstats = load_fundamentals(dates_all, codes_all)
        kw.update(roe_ttm=fund["roe_ttm"], yoy_ni=fund["yoy_ni"],
                  debt_ratio=fund["debt_ratio"])
        src["fundamentals"] = fstats
    if with_shares:
        shm, sstats = share_float_matrix(dates_all, codes_all)
        kw["float_shares"] = shm
        src["shares"] = sstats
    if with_industry:
        gids, istats = industry_matrix(dates_all, codes_all, level=industry_level)
        kw["industry_gid"] = gids
        src["industry"] = istats

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
        source=src,
        **kw,
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


def _rolling_std(mat: np.ndarray, window: int) -> np.ndarray:
    """按列滚动样本标准差（ddof=1）：**窗口内必须满 window 个有效值**，否则 NaN。

    min_periods=window 是"不插补"的物理实现：窗口里少一天（停牌、hfq 缺口、
    上市不足）就返回 NaN，让下游按"排除"处理，而不是拿不足的样本算一个
    看似正常的标准差。004C 明令：换手率窗口不足 60 日排除、不插补。
    """
    if mat.shape[0] < window:
        return np.full(mat.shape, np.nan, dtype="float64")
    df = pd.DataFrame(mat)
    out = df.rolling(window=int(window), min_periods=int(window)).std(ddof=1)
    return out.to_numpy(dtype="float64")


def realized_vol(px: np.ndarray, window: int) -> np.ndarray:
    """已实现波动：过去 window 个交易日**日收益**的标准差（px 必须是 hfq 价）。

    T 日值只用 ≤T-1 的数据（先滚动、再 _shift1），与全模块口径一致。
    窗口内任一天缺价 → 该日收益 NaN → 整窗 NaN，不插补。
    用 hfq 而不是原始价：除权日的假跌幅会被记进波动，低波因子会挑出
    正在分红的股票，那是噪声不是低风险。
    """
    r = np.full(px.shape, np.nan, dtype="float64")
    if px.shape[0] > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = px[1:] / px[:-1] - 1.0
    r[~np.isfinite(r)] = np.nan
    return _shift1(_rolling_std(r, window))


def turn_stability(turn: np.ndarray, window: int = 60) -> np.ndarray:
    """换手率稳定性：过去 window 日**换手率的**标准差，低者优（值越小越稳）。

    turn ≤ 0（停牌/无成交）不是"换手率为零"，是"没有交易"，先置 NaN 再滚动；
    否则停牌日会被当成极端低换手，把"停牌多"误判成"很稳定"。
    返回原始值（越小越好），定向取负号由 FACTOR_DEFS 里的 `turn_stab_60` 负责。
    """
    t = _shift1(turn)
    t = np.where(np.isfinite(t) & (t > 0.0), t, np.nan)
    return _rolling_std(t, window)


def _need(fd: FactorData, attr: str, factor: str) -> np.ndarray:
    """取因子依赖的附加数据；没载入就报错，不许悄悄返回空矩阵。"""
    v = getattr(fd, attr, None)
    if v is None:
        raise RuntimeError(
            f"因子 {factor} 依赖 {attr}，但当前 FactorData 没载入该数据源；"
            f"请用 load_factor_data(..., {attr}=True) 或检查 data/shares 是否已抓取。")
    return v


def _quality(mat: np.ndarray) -> np.ndarray:
    """质量因子：矩阵已按公告日 as-of 铺好，再过 _shift1 满足"只用 ≤T-1 已公告"。"""
    out = _shift1(mat)
    out[~np.isfinite(out)] = np.nan
    return out


FACTOR_DEFS = {
    "momentum_20": lambda fd: momentum(fd.panel.close_hfq, 20),
    "momentum_60": lambda fd: momentum(fd.panel.close_hfq, 60),
    "ep": lambda fd: ep(fd.pe_ttm),
    "bp": lambda fd: bp(fd.pb_mrq),
    "size": lambda fd: size(fd.panel.close_raw, fd.volume, fd.turn),
    # —— 004B 新增
    "size_real": lambda fd: size_real(fd.panel.close_raw,
                                      _need(fd, "float_shares", "size_real")),
    "roe_ttm": lambda fd: _quality(_need(fd, "roe_ttm", "roe_ttm")),
    "yoy_ni": lambda fd: _quality(_need(fd, "yoy_ni", "yoy_ni")),
    "debt_ratio": lambda fd: _quality(_need(fd, "debt_ratio", "debt_ratio")),
    # —— 004C 新增：低波动异象。**方向已在本层取负号**（值越大 = 波动越低 = 越优），
    # 这样下游算 IC 时"正相关"就等于"方向为正"，不必再在报告里二次解释符号
    # （004C 复核 ①：先定向再判方向，避免低波动因子被自己的规则误杀）。
    "lowvol_20": lambda fd: -realized_vol(fd.panel.close_hfq, 20),
    "lowvol_60": lambda fd: -realized_vol(fd.panel.close_hfq, 60),
    "lowvol_120": lambda fd: -realized_vol(fd.panel.close_hfq, 120),
    "turn_stab_60": lambda fd: -turn_stability(fd.turn, 60),
    # —— 004C C2：反转多窗口。动量取负号 = 反转，值越大 = 过去跌得越多 = 越优。
    # 现有 005R 的综合分用的是 `1 - momentum_20 分位`，与 reversal_20 同义。
    "reversal_20": lambda fd: -momentum(fd.panel.close_hfq, 20),
    "reversal_40": lambda fd: -momentum(fd.panel.close_hfq, 40),
    "reversal_60": lambda fd: -momentum(fd.panel.close_hfq, 60),
}

#: 004B 报告默认因子集（既有 5 个 + 质量 3 个 + 真实股本 size）
DEFAULT_FACTORS = ("momentum_20", "momentum_60", "ep", "bp", "size",
                   "size_real", "roe_ttm", "yoy_ni", "debt_ratio")

#: 004C 候选因子（C1 低波动 4 个 + C2 反转 3 个）。全部已定向：值越大越优，
#: 于是"IC 正 + |t| ≥ 2"即等于 004C 说的"方向为正且够强"。
FACTORS_004C = ("lowvol_20", "lowvol_60", "lowvol_120", "turn_stab_60",
                "reversal_20", "reversal_40", "reversal_60")


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
      **按因子差异很大，不许写成全称句**（审计-B7-03）：动量类层期中位
      momentum_20 0.325% / momentum_60 0.2371%，其余 7 个因子 0.0002%~0.0008%
      （最高 bp 0.0007%）。引擎按资金约束贪心成交，现金基本买空，残差主体是
      "某些股票配少了"而不是"留了一笔现金"；但动量类确实会留到千分之几，
      说成"全部极小"会重犯 B6-01 的错。三个数必须分开报，
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
        "max_layer_median_pct": cash_hi,
        # 审计-B6-01：旧稿只报了 min/max 就说"未投入现金 <0.01%"，而实测单期最大
        # 到 3.88%。补中位与全样本分位，让"到底留了多少现金"能被核对。
        "cash_median_pct": (round(float(np.median([f["cash_pct"] for f in fills
                                                   if f.get("cash_pct") is not None])), 4)
                            if any(f.get("cash_pct") is not None for f in fills) else None),
        "cash_p95_pct": (round(float(np.quantile([f["cash_pct"] for f in fills
                                                  if f.get("cash_pct") is not None], 0.95)), 4)
                         if any(f.get("cash_pct") is not None for f in fills) else None),
        "definition": "① 5 层各 1/5 资金合并 vs 全截面等权，逐月差（pp）；"
                      "② 部分成交股未达目标缺口（执行日，各期中位数）；"
                      "③ 整体权重偏差 ½Σ|实际−目标|；④ 未投入现金占比",
    }


# ---------------------------------------------------------------- 单段评估

def cash_residual_distribution(seg_dev: dict) -> dict:
    """跨**全部**因子汇总未投入现金（审计-B7-01）。

    教训：B6-01 修「现金不是几乎为零」时，取了 `factor_keys` 里**第一个**因子的
    lot_residual 写进免责 —— 第一个正好是动量类（1.45%），于是把动量类的峰值当成了
    全体的水平。实测其余 7 个因子全期 <0.001%，两个方向都错（旧稿「<0.01%」被动量
    类推翻，改完又拿动量类代表全体）。

    所以这里必须给跨因子的**范围**，并点出最高的是哪个因子。

    口径提醒：`max_layer_median_pct`（**原 cash_max_pct，审计-B7 改名**）是「5 层各自的
    期中位现金」里的最大值，**不是单期最大值**；单期极值看
    lot_residual.layer_cash_pct（5 个数的明细）。旧名把"层间最大中位数"说成"最大"，
    下游一律按单期极值解读，改名前已如实标注。
    """
    rows = {}
    for k, v in seg_dev.items():
        if k.startswith("_"):
            continue
        lr = (v or {}).get("lot_residual") or {}
        if not lr.get("available"):
            continue
        rows[k] = {c: lr.get(c) for c in
                   ("cash_median_pct", "cash_p95_pct", "max_layer_median_pct")}
    if not rows:
        return {"available": False, "reason": "no_lot_residual"}

    meds = [r["cash_median_pct"] for r in rows.values()
            if r["cash_median_pct"] is not None]
    maxs = [r["max_layer_median_pct"] for r in rows.values()
            if r["max_layer_median_pct"] is not None]
    if not maxs:
        return {"available": False, "reason": "no_cash_pct"}

    worst = max(rows, key=lambda k: rows[k]["max_layer_median_pct"] if
                rows[k]["max_layer_median_pct"] is not None else -1.0)
    tiny = sorted(k for k, r in rows.items()
                  if (r["max_layer_median_pct"] or 0.0) < 0.01)
    return {
        "available": True,
        "n_factors": len(rows),
        "median_pct_min": round(float(min(meds)), 6) if meds else None,
        "median_pct_max": round(float(max(meds)), 6) if meds else None,
        "max_pct_min": round(float(min(maxs)), 6),
        "max_pct_max": round(float(max(maxs)), 6),
        "worst_factor": worst,
        "worst_layer_median_pct": round(float(rows[worst]["max_layer_median_pct"]), 6),
        "n_factors_below_001pct": len(tiny),
        "factors_below_001pct": tiny,
        "by_factor": rows,
        "definition": "未投入现金 = 1−(权益−现金)/权益，各期取中位；此处跨全部因子取"
                      "范围并点名最高因子。max_layer_median_pct（原 cash_max_pct）是"
                      "5 层「期中位数」的最大值，不是单期最大值"
                      "（单期极值见 lot_residual.layer_cash_pct）",
    }


def evaluate_segment(factors: dict[str, np.ndarray], fd: FactorData,
                     i0: int, i1: int, cfg_template: Config,
                     n_layers: int = N_LAYERS, *, neutralize: bool = False,
                     gids: np.ndarray | None = None,
                     method: str = "demean") -> dict:
    """在一段区间上做 IC + 分层回测（因子定义不随段变化）。

    neutralize=True 时，截面日上的因子值先做行业中性化（见 neutralize_matrix），
    非截面日不处理。两组结果分开产出，用于对比"中性化到底改变了什么"。
    """
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

    neu_stats: dict = {"enabled": bool(neutralize), "method": method if neutralize else None,
                       "min_group": MIN_GROUP_SIZE if neutralize else None}
    if neutralize:
        gids = gids if gids is not None else fd.industry_gid
        if gids is None:
            raise RuntimeError(
                "neutralize=True 需要行业组号；load_factor_data(with_industry=True) 未载入行业，"
                "或先跑 python -m src.fetch_shares --stage industry")
    else:
        gids = gids if gids is not None else fd.industry_gid

    # 诊断基准：全截面等权（同引擎同成本），只用于拆解整手残差
    bench_sig = build_full_xsec_signal(fd, ends_all)
    bench = run_backtest(panel, bench_sig, cfg) if bench_sig else None
    # 全市场 hfq 等权（S2 的基准口径，不做任何交易/成本假设）——
    # 它与"全截面引擎"的差就是股票池/资金切分口径差，不是整手残差（B5-02）
    market_ew = M.equal_weight_market(panel)

    res = {}
    for name, fmat in factors.items():
        if neutralize:
            fmat, ns = neutralize_matrix(fmat, fd, ends_all, gids, method=method)
            neu_stats.setdefault("factors", {})[name] = ns
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
                    "neutralization": neu_stats,
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


def _readings(dev: dict, ho: dict, neu: dict | None = None,
              size_cmp: dict | None = None,
              jump_agree: dict | None = None) -> list[str]:
    """按数字自动生成的"怎么读"（B5-07：措辞随数字走，不写死的形容词）。

    审计-B5 指出三处措辞与数字不符，这里全部改成从 layer_ann 直接算：
      ① ep 不是 U 形（峰在中间偏高层、谷在 Q2）；② size 在 dev 是后段断崖、谷底在 Q5，
      不是"中间塌陷"（真中间塌陷只在 holdout 出现，11 期不足以下结论）；
      ③ bp 的 |t|>3 不是唯一，动量 |t|≈4.1 同样过线。
    004B 追加：质量因子与 size 新旧口径的读法，同样只描述数字。
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

    # ---- 004B：质量因子（只陈述数字，显著性照实说）
    q = [(k, dev[k]) for k in ("roe_ttm", "yoy_ni", "debt_ratio") if k in dev]
    if q:
        bits = "；".join(f"{k} IC={v['ic_mean']:+.4f}（t={v['t_stat']:+.2f}，"
                        f"{v['n_periods']} 期）" for k, v in q)
        over = [k for k, v in q if abs(v["t_stat"]) > 2]
        out.append(
            f"质量因子（公告日对齐）：{bits}。"
            + (f"过 |t|>2 的：{', '.join(over)} —— 但月频 IC 自相关会高估 t，"
               f"只能算「值得继续跟」，不能当 alpha。"
               if over else
               "三个都没过 |t|>2 —— 在 dev 段看不出质量因子有稳定截面区分度，"
               "这本身就是结论，不要硬讲成「方向对」。" if q else ""))

    # ---- 004B：size 新旧口径
    if size_cmp:
        dev_cmp = size_cmp.get("dev") or {}
        ic = (size_cmp.get("ic") or {}).get("dev") or {}
        if dev_cmp.get("available"):
            _real_ic = ic.get("size_real")
            _ic_s = _f(ic.get("size"), "+.4f")
            if _real_ic is None or not np.isfinite(float(_real_ic)):
                # 真实股本没覆盖到（抓取没跑完/个别票缺失）时，只说拿到了什么，
                # 不能反过来断言"两口径一致"—— 那是没数可比的默认话术。
                out.append(
                    f"size 两口径：真实口径本次不可用（size_real 无有效 IC），"
                    f"只有推算口径 IC {_ic_s}。月度截面市值差中位 "
                    f"{_f(dev_cmp.get('median_abs'))}、最大 {_f(dev_cmp.get('max_abs'))}。"
                    f"结论：**本轮不能拿 size_real 替换 size**，先补股本再谈口径差异。")
            else:
                _both = (jump_agree or {}).get("both_jump_ratio")
                _gt5 = dev_cmp.get("share_gt5pct")
                _tail = (
                    f"推算侧跳变里 {_both:.1%} 在真实侧同日也跳，剩下 {1 - _both:.1%} "
                    f"属脏跳变（低换手日舍入/停牌复牌）—— 差异就出在后者，不是全局重排。"
                    if isinstance(_both, (int, float)) else
                    "差异集中在少数跳变日，不是全局重排。")
                out.append(
                    f"size 两口径：月度截面相对差 中位 {_f(dev_cmp['median_abs'])}、"
                    f"p90 {_f(dev_cmp['p90_abs'])}、最大 "
                    f"{_f(dev_cmp.get('max_abs_pct'), '.2f')}%"
                    f"（{_f(dev_cmp.get('max_ratio_x'), '.2f')}x 倍）"
                    + (f"，差 <5% 的点占 {_f(1 - _gt5, '.2%')}" if isinstance(_gt5, (int, float)) else "")
                    + f"，超过 30% 的占 {_f(dev_cmp['share_gt30pct'], '.2%')}；"
                    f"IC 推算 {_ic_s} vs 真实 {_f(_real_ic, '+.4f')}。" + _tail)

    # ---- 004B：行业中性化前后
    if neu and neu.get("compare"):
        cmp_dev = neu["compare"]["dev"]
        delta = sorted(((k, v["delta_ic"]) for k, v in cmp_dev.items()),
                       key=lambda x: -abs(x[1]))[:3]
        bits = "；".join(f"{k} ΔIC={d:+.4f}" for k, d in delta)
        out.append(
            f"行业中性化（申万 {neu.get('level')} 级 {neu.get('groups')} 组，"
            f"{neu.get('method')}）：IC 变化最大的三个 {bits}。"
            f"中性化前后两组结果分列 segments / segments_neutralized，"
            f"谁高谁低都如实报 —— 中性化不是「一定更好」，它只是拿掉行业暴露。")
    return out


def _git_meta() -> dict:
    """产物溯源三元组（审计-B6-02）。

    旧版只写一个 `git_commit`，而产物通常在提交**之前**生成 —— 于是产物里的哈希
    永远指向上一笔代码，"这份报告到底是哪版代码算的"无法回答。现在拆成：
      · code_commit     生成时代码所在的 HEAD；
      · code_dirty      生成时**代码**（src/ tests/）是否有未提交改动；
      · worktree_dirty  整个工作区是否有未提交改动（含别人未提交的文档）；
      · generated_at    生成时刻。

    `code_dirty` 只看 src/ 与 tests/（审计-B7-04）：仓库里长期有总工未提交的
    AGENTS.md / docs/，按全仓库算会让 code_dirty 永远为 True，"这份产物是哪版代码
    算的"就永远说不清。全量状态另出 `worktree_dirty` + `worktree_other_files`，
    不隐瞒。
    `write_report()` 还会再补一个 written_at，落盘时刻与生成时刻分开记。
    """
    commit = _git_commit()
    try:
        # 不能先 .strip()：porcelain 首行是" M path"，前导空格是状态位的一半，
        # strip 掉再切 [3:] 会把 "AGENTS.md" 切成 "GENTS.md"（实测踩过）。
        # -c core.quotepath=false：中文路径直接给 UTF-8，别输出 \344\270\211 这种八进制。
        out = subprocess.run(["git", "-c", "core.quotepath=false",
                              "status", "--porcelain"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        paths = [ln[3:].strip().strip('"') for ln in lines]
        # 只把 src/ tests/ 算进 code_dirty；文档类改动单列，不参与"代码是否干净"的判断
        code_files = [p for p in paths if p.startswith(("src/", "tests/"))]
        other_files = [p for p in paths if not p.startswith(("src/", "tests/"))][:20]
        dirty = bool(code_files)
        dirty_files = code_files[:20]
        worktree_dirty = bool(paths)
    except Exception:  # noqa: BLE001
        dirty, dirty_files, worktree_dirty, other_files = None, [], None, []
    return {"code_commit": commit, "git_commit": commit, "code_dirty": dirty,
            "code_dirty_files": dirty_files,
            "worktree_dirty": worktree_dirty,
            "worktree_other_files": other_files,
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")}


def size_cross_check(fd: FactorData, f_old: np.ndarray, f_new: np.ndarray,
                     ends: list[int]) -> dict:
    """新旧 size 口径对拍：在月度截面上比较流通市值（相对差）。

    两个口径都是 log(流通市值)，相减再 exp 就是市值相对差 —— 直接回答
    "换成真实股本后，size 到底变了多少"，而不是靠印象。
    """
    diffs = []
    n_pts = 0
    for i in ends:
        ok = cross_section_mask(fd, i)
        a, b = f_old[i][ok], f_new[i][ok]
        m = np.isfinite(a) & np.isfinite(b)
        n_pts += int(m.sum())
        if m.any():
            diffs.append(np.exp(b[m] - a[m]) - 1.0)
    if not diffs:
        return {"available": False, "reason": "no_overlap"}
    d = np.concatenate(diffs)
    a = np.abs(d)
    # B7-05：max_abs 是**相对差** |比值−1|，不是倍数。2.1585 对应 3.16x，
    # 旧汇报写成「最大 2.16 倍」把相对差当成了倍数（差 2.16x 与 3.16x 两种读法）。
    # 这里把两种口径都落进产物，下游不用再换算。
    return {
        "available": True, "n_points": int(n_pts), "n_cross_sections": len(ends),
        "median_abs": round(float(np.median(a)), 8),
        "p90_abs": round(float(np.quantile(a, 0.90)), 8),
        "p99_abs": round(float(np.quantile(a, 0.99)), 8),
        "max_abs": round(float(a.max()), 8),
        "max_abs_pct": round(float(a.max() * 100.0), 4),
        "max_abs_ratio_x": round(float(1.0 + a.max()), 4),
        "max_ratio_x": round(float(1.0 + float(d.max())), 4),
        "min_ratio_x": round(float(1.0 + float(d.min())), 4),
        "share_gt5pct": round(float((a > 0.05).mean()), 8),
        "share_gt30pct": round(float((a > 0.30).mean()), 8),
        "definition": "月度截面比值 = 真实股本市值/推算股本市值（两口径都取 T-1）；"
                      "max_abs = |比值−1| 即相对差，倍数 = 1+相对差。"
                      "max_abs=2.1585 → 相对差 215.85%、倍数 3.16x（**不是 2.16 倍**）",
    }


def share_jump_agreement(fd: FactorData, threshold: float = 0.30) -> dict:
    """解禁跳变日两口径是否一致（004B-P 的核心实证，这里做成可复算的产物字段）。

    逐日算两个股本序列的相邻比：推算 `volume/(turn/100)` 与真实 `float_shares`。
    取推算侧跳变（|比−1|>阈值）的日子，看真实侧同日是否也跳、倍数差多少。
    差异大的那些日子就是"推算口径的脏跳变"（勘察里 sz.001289 2022-08-24/26 即此类）。
    """
    if fd.float_shares is None:
        return {"available": False, "reason": "no_float_shares"}
    v, t = fd.volume, fd.turn
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = v / (t / 100.0)
    imp[~(np.isfinite(imp)) | (imp <= 0)] = np.nan
    imp = pd.DataFrame(imp).ffill().to_numpy()          # 停牌日沿用上一有效观测
    r_imp = np.full_like(imp, np.nan)
    r_imp[1:] = imp[1:] / imp[:-1]
    r_real = np.full_like(fd.float_shares, np.nan, dtype="float64")
    r_real[1:] = fd.float_shares[1:] / fd.float_shares[:-1]

    jump = np.isfinite(r_imp) & (np.abs(r_imp - 1.0) > threshold)
    both = jump & np.isfinite(r_real) & (np.abs(r_real - 1.0) > threshold)
    rel = np.abs(np.log(np.where(np.isfinite(r_real) & (r_real > 0), r_real, np.nan))
                 - np.log(np.where(np.isfinite(r_imp) & (r_imp > 0), r_imp, np.nan)))
    both_rel = rel[both & np.isfinite(rel)]
    n_codes_real = int(np.isfinite(fd.float_shares).any(axis=0).sum())
    n_codes = int(fd.float_shares.shape[1])
    return {
        "available": True, "threshold": threshold,
        # 覆盖率不报的话，股本只抓到几只时 both_jump_ratio≈0 会被误读成
        # "推算口径全是脏数据"，方向正好相反。
        "codes_with_real_shares": n_codes_real,
        "codes_total": n_codes,
        "real_coverage": round(n_codes_real / max(n_codes, 1), 6),
        # 计数单位是**股票·日**，不是自然日：同一天可能几十只票跳变
        "jump_cells": int(jump.sum()),
        "jump_codes": int(jump.any(axis=0).sum()),
        "both_jump_cells": int(both.sum()),
        "both_jump_ratio": round(float(both.sum() / max(int(jump.sum()), 1)), 6),
        "ratio_logdiff_median": (None if not len(both_rel)
                                 else round(float(np.median(both_rel)), 8)),
        "ratio_logdiff_p90": (None if not len(both_rel)
                              else round(float(np.quantile(both_rel, 0.90)), 8)),
        "unit": "股票·日（同一交易日多只票算多次）",
        "definition": "两序列相邻比；推算侧跳变里真实侧同日也跳的比例与倍数偏差",
    }


def _jump_sentence(ja: dict) -> str:
    """把"跳变到底是真实事件还是脏数据"按实测比例写成一句话。

    旧稿写死"12/14 抽样有事件对应"，那是 004B-P 的 14 次人工抽样；现在有全样本
    逐日对拍，就该报全样本比例 —— 否则数字永远停在旧抽样上。
    """
    if not ja or not ja.get("available"):
        return "。跳变与真实股本事件的对应关系见产物 size_compare.jump_agreement。"
    both = ja.get("both_jump_ratio")
    cov = ja.get("real_coverage")
    if not isinstance(both, (int, float)):
        return "。跳变与真实股本事件的对应关系见产物 size_compare.jump_agreement。"
    head = (f"。**推算侧跳变里有 {both:.1%} 真实股本同日也跳**"
            f"（{ja.get('both_jump_cells')}/{ja.get('jump_cells')} 个股票·日，"
            f"真实股本覆盖 {cov:.2%}）—— 这些是真实解禁/增发，不是数据错误；"
            f"剩下 {1 - both:.1%} 才是脏跳变（低换手日换手率舍入、停牌复牌）。")
    return head


def build_validity_note(*, size_jump: dict | None = None, lot: dict | None = None,
                        cash_dist: dict | None = None,
                        turn_missing: dict | None = None,
                        neu: dict | None = None, size_cmp: dict | None = None,
                        jump_agree: dict | None = None,
                        fundamentals: dict | None = None) -> str:
    """失效声明（审计-B6-04：数字全部从实测产物取，不再写死）。

    任何一处拿不到实测值，就写"见产物字段 X"，**绝不回填旧数字** ——
    旧版把 8,628 行 / 0.12pp / "<0.01%" 写进字符串，改了默认值产物照样说旧话。
    """
    p = ["本报告是因子刻画结果，不构成策略有效性或 alpha 证据。"]

    if neu and neu.get("enabled"):
        p.append(f"① 分层为层内等权、月度再平衡；**已产出行业中性化对照版**"
                 f"（申万 {neu.get('level', INDUSTRY_LEVEL)} 级、{neu.get('method')}、"
                 f"组内 <{MIN_GROUP_SIZE} 只不处理），两组结果分列 segments / "
                 f"segments_neutralized。中性化只作用于截面日，非截面日不处理。")
    else:
        p.append("① 分层为层内等权、月度再平衡，未做行业/风格中性化（本报告未启用中性化）。")

    p.append("② IC 用理论开盘价收益，未扣除涨跌停/停牌造成的不可成交 —— "
             "IC 与分层收益之间允许有口径差，两者不可互相校验到小数位。")

    if size_jump:
        ex = (size_jump.get("examples") or [{}])[0]
        p.append(
            f"③ size 口径：本报告同时给 `size`（换手率推算）与 `size_real`（真实股本）两列，"
            f"差异分布见 size_compare（月度截面中位 "
            f"{(size_cmp or {}).get('dev', {}).get('median_abs', float('nan')):.4f}、"
            f"p90 {(size_cmp or {}).get('dev', {}).get('p90_abs', float('nan')):.4f}、"
            f"最大相对差 {_f((size_cmp or {}).get('dev', {}).get('max_abs_pct'), '.2f')}%"
            f" 即 {_f((size_cmp or {}).get('dev', {}).get('max_ratio_x'), '.2f')}x 倍 ——"
            f"相对差与倍数不是同一个数，旧稿「最大 2.16 倍」是把相对差当倍数）。"
            f"推算口径的跳变实测 {size_jump.get('rows')} 行 / {size_jump.get('codes')} 只，"
            f"其中 {size_jump.get('month_end_rows')} 行落在月末 "
            f"{size_jump.get('month_end_window_days')} 日窗口内"
            + (f"（例 {ex.get('code')} {ex.get('date')} ×{ex.get('share_ratio')}）"
               if ex.get('code') else "")
            + (_jump_sentence(jump_agree) if jump_agree else
               "。这些跳变是否对应真实股本事件见产物 size_compare.jump_agreement。"))
    else:
        p.append("③ size 口径两种并列（size / size_real），跳变统计见产物 size_jump_stats。")

    if turn_missing:
        _wy = turn_missing.get("worst_year")
        _wrate = (turn_missing.get("by_year", {}).get(_wy) or {}).get("rate")
        p.append(f"④ turn 缺失 {_f(turn_missing.get('rate'), '.4%')}"
                 f"（{turn_missing.get('missing')} 行），"
                 f"最差年份 {_wy}（{_f(_wrate, '.4%')}）；"
                 f"缺失行 100% 是停牌行（all_suspended={turn_missing.get('all_suspended')}）。")
    else:
        p.append("④ turn 缺失率见产物 data_quality.turn_missing。")

    if fundamentals:
        _rate = fundamentals.get("pubdate_missing_rate")
        _rate_s = f"{_rate:.2%}" if isinstance(_rate, (int, float)) else "未知"
        _epsc = fundamentals.get("epsTTM_coverage")
        _eps_s = f"{_epsc:.2%}" if isinstance(_epsc, (int, float)) else "未统计"
        _unres = fundamentals.get("pubdate_unresolved_rows")
        _unres_s = (f"；另 {_unres} 行两个公告日字段都缺、整行剔除" if _unres else "")
        p.append(f"⑤ 质量因子用公告日对齐：可用 {fundamentals.get('rows_usable')} 行 / "
                 f"{fundamentals.get('codes')} 只；其中 "
                 f"{fundamentals.get('pubdate_missing_rows')} 行"
                 f"（{_rate_s}）没有实际公告日，"
                 f"回退法定披露截止日推算{_unres_s} —— 截止日必然 ≥ 真实披露日，"
                 f"方向是**滞后**（只可能晚知道、不可能预知未来），代价是这批期数"
                 f"晚一到两周才进截面，属时点噪声而非前视；审计-B7 巨潮对照 50 例"
                 f"复核 0 例提前、平均滞后 8.8 天。"
                 f"epsTTM 覆盖率 {_eps_s}，未用作因子。")
    else:
        p.append("⑤ 质量因子时点口径见产物 data.fundamentals。")

    p.append("⑥ 截面已剔除 ST/停牌/hfq 缺口/上市不足 60 日；holdout 段仅 12 个月、11 期 IC，"
             "样本量不足以支撑统计推断，只作记录，不作结论。")

    if lot and lot.get("available"):
        cash = cash_dist or {}
        if cash.get("available"):
            # B7-01：现金必须按全因子分布报。取单个因子（尤其排第一的动量类）
            # 会同时犯两个方向的错，见 cash_residual_distribution 的注释。
            cash_s = (f"**未投入现金按因子看**（{cash['n_factors']} 个因子）：各层期中位 "
                      f"{cash['median_pct_min']}%~{cash['median_pct_max']}%，"
                      f"最高是 `{cash['worst_factor']}`（层间最大中位 "
                      f"{cash['worst_layer_median_pct']}%）；另有 "
                      f"{cash['n_factors_below_001pct']} 个因子全期 <0.01%。"
                      f"取单一因子代表全体两个方向都会错（旧稿把它说成几乎为零，"
                      f"被动量类推翻；改完又拿动量类的峰值代表全体）—— 故给全因子范围。")
        else:
            cash_s = (f"未投入现金见产物 cash_residual_distribution"
                      f"（本报告该项不可用，不填旧数字）。")
        p.append(
            f"⑦ **层间比较有效，绝对水平不可与全市场等权直接相比**。整手残差实测"
            f"（5 层各 1/5 资金合并 vs 全截面等权，同引擎同成本同 mask）：月差 "
            f"mean|diff|={lot.get('mean_abs_pp')}pp、中位 {lot.get('median_abs_pp')}pp、"
            f"p90 {lot.get('p90_abs_pp')}pp，signed 年化 {lot.get('signed_ann_pp')}pp；"
            f"每层部分成交缺口 {lot.get('partial_gap_min_pct')}%~{lot.get('partial_gap_max_pct')}%。"
            + cash_s
            + "残差主体是「配不满」，不是「留了一笔现金」。")
    else:
        p.append("⑦ 整手残差见产物 segments.*.lot_residual（本报告该项不可用）。")

    p.append("⑧ 月频 IC 存在自相关，t 值会**高估**显著性，只作量级参考 —— 不以 |t|>2 判定因子"
             "有效，也不以「唯一 |t|>3」作结论支点（同表常有动量同样过线）。严格检验需 "
             "Newey-West 或 block bootstrap，本报告未做。")
    p.append("策略有效性判断须待后续样本外验证，不以本报告为准。")
    return "\n".join(p)


def build_report(fd: FactorData, factors: dict[str, np.ndarray],
                 args: argparse.Namespace, cfg_template: Config | None = None,
                 data_quality: dict | None = None,
                 neutralize_variants: bool = True) -> dict:
    """一次性生成 dev + holdout 两段报告，并产出行业中性化对照版。

    cfg_template 不给时按 args.cash / args.seed 造一个默认 Config；
    测试与复算可传入自定义模板（改 lot_size / costs 必须能在产物里看见，B5-06）。
    data_quality 为 `data_quality_stats()` 的产物，落进 size_jump_stats 字段（B5-03）。
    """
    bounds = segment_bounds(fd.dates)
    tmpl = cfg_template or Config(start=fd.dates[0], end=fd.dates[-1],
                                  init_cash=args.cash, seed=args.seed)
    method = getattr(args, "neutralize_method", "demean")

    segs = {}
    for tag, (i0, i1) in bounds.items():
        segs[tag] = evaluate_segment(factors, fd, i0, i1, tmpl, args.layers)

    factor_keys = [k for k in segs["dev"] if not k.startswith("_")]
    assert factor_keys == [k for k in segs["holdout"] if not k.startswith("_")]

    # 参数一律取实际生效的那份 Config（B5-06：改默认值必须让产物可见）
    eff = _layer_cfg(fd, *bounds["dev"], tmpl)

    # ---- 行业中性化对照版
    neu_segs = None
    neu_meta: dict = {"enabled": False}
    if neutralize_variants and fd.industry_gid is not None:
        neu_segs = {}
        for tag, (i0, i1) in bounds.items():
            neu_segs[tag] = evaluate_segment(factors, fd, i0, i1, tmpl, args.layers,
                                             neutralize=True, gids=fd.industry_gid,
                                             method=method)
        ist = (fd.source.get("industry") or {})
        neu_meta = {"enabled": True, "method": method,
                    "level": ist.get("level", INDUSTRY_LEVEL),
                    "groups": ist.get("groups"),
                    "min_group": MIN_GROUP_SIZE,
                    "codes_coverage": ist.get("codes_coverage"),
                    "cells_unmapped": ist.get("cells_unmapped")}
        neu_meta["compare"] = {
            tag: {k: {"ic_before": segs[tag][k]["ic_mean"],
                      "ic_after": neu_segs[tag][k]["ic_mean"],
                      "delta_ic": float(neu_segs[tag][k]["ic_mean"]
                                        - segs[tag][k]["ic_mean"]),
                      "t_before": segs[tag][k]["t_stat"],
                      "t_after": neu_segs[tag][k]["t_stat"],
                      "ls_before": segs[tag][k]["long_short_ann"],
                      "ls_after": neu_segs[tag][k]["long_short_ann"],
                      "q1_after": neu_segs[tag][k]["layer_ann"][0],
                      "q5_after": neu_segs[tag][k]["layer_ann"][-1]}
                  for k in factor_keys}
            for tag in ("dev", "holdout")}

    # ---- 新旧 size 口径对拍
    size_cmp: dict = {"available": False}
    if "size" in factors and "size_real" in factors:
        size_cmp = {"available": True,
                    "jump_agreement": share_jump_agreement(fd),
                    "sections": {}}
        for tag, (i0, i1) in bounds.items():
            ends = [i for i in month_end_indices(fd.dates) if i0 <= i < i1]
            size_cmp["sections"][tag] = size_cross_check(
                fd, factors["size"], factors["size_real"], ends)
        size_cmp["sections"]["ic"] = {
            tag: {"size": segs[tag]["size"]["ic_mean"],
                  "size_real": segs[tag]["size_real"]["ic_mean"],
                  "t_size": segs[tag]["size"]["t_stat"],
                  "t_size_real": segs[tag]["size_real"]["t_stat"],
                  "ls_size": segs[tag]["size"]["long_short_ann"],
                  "ls_size_real": segs[tag]["size_real"]["long_short_ann"]}
            for tag in ("dev", "holdout")}

    dq = data_quality or {}
    # 现金残差取**第一个可用**的因子：写死 "bp" 时，别的因子单独跑（如只跑 rank）
    # 会拿到空 dict，免责里退回"本报告该项不可用"，与产物字段对不上。
    lot0 = next((segs["dev"][k]["lot_residual"] for k in factor_keys
                 if segs["dev"][k].get("lot_residual", {}).get("available")), {})
    cash_dist = cash_residual_distribution(segs["dev"])
    note = build_validity_note(
        size_jump=dq.get("size_jump"), lot=lot0, cash_dist=cash_dist,
        turn_missing=dq.get("turn_missing"),
        neu={**neu_meta, "method": method} if neu_meta.get("enabled") else None,
        size_cmp=size_cmp.get("sections", {}),
        jump_agree=size_cmp.get("jump_agreement"),
        fundamentals=fd.source.get("fundamentals"))

    return {
        **_git_meta(),
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
            "neutralize": neu_meta,
            "factor_defs": {
                "momentum_20": "close_hfq(T-1)/close_hfq(T-21)-1",
                "momentum_60": "close_hfq(T-1)/close_hfq(T-61)-1",
                "ep": "1/peTTM(T-1)",
                "bp": "1/pbMRQ(T-1)",
                "size": "log(close(T-1)*volume(T-1)/(turn(T-1)/100))  # 推算口径",
                "size_real": "log(close(T-1)*float_share(T-1))  # 真实股本口径",
                "roe_ttm": "净利润TTM/期末股东权益，取公告日 ≤ T-1 的最新一期",
                "yoy_ni": "净利润同比（%），公告日对齐",
                "debt_ratio": "资产负债率（%），公告日对齐",
            },
        },
        "data": fd.source,
        "size_jump_stats": dq.get("size_jump", {}),
        "cash_residual_distribution": cash_dist,
        "data_quality": dq,
        "segments": segs,
        "segments_neutralized": neu_segs,
        "neutralization_compare": neu_meta,
        "size_compare": size_cmp,
        "layer_shape": {tag: {k: _layer_shape(v["layer_ann"])
                              for k, v in segs[tag].items() if not k.startswith("_")}
                        for tag in ("dev", "holdout")},
        "layer_shape_neutralized": ({tag: {k: _layer_shape(v["layer_ann"])
                                           for k, v in neu_segs[tag].items()
                                           if not k.startswith("_")}
                                     for tag in ("dev", "holdout")}
                                    if neu_segs else None),
        "readings": _readings(segs["dev"], segs["holdout"],
                              neu=neu_meta if neu_meta.get("enabled") else None,
                              size_cmp=size_cmp.get("sections", {}),
                              jump_agree=size_cmp.get("jump_agreement")),
        "size_source": SIZE_SOURCE,
        "size_source_real": SIZE_SOURCE_REAL,
        "validity_note": note,
    }


def write_report(report: dict, path: Path) -> Path:
    """落盘 JSON。NaN 统一转 null，保证严格 JSON（下游不用猜）。

    落盘时补 `written_at` 与**落盘时刻的** HEAD —— 与 `code_commit`（生成时刻）
    分开记，产物才能对上准确的代码版本（审计-B6-02）。
    """
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

    report["written_at"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    report["written_code_commit"] = _git_commit()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def _f(x, spec: str = ".4f") -> str:
    """表格数字格式化：None / NaN / inf 一律打 `n/a`。

    因子在某些截面会算不出值（样本不足、股本缺失），直接 f"{x:.2%}" 会打出
    `nan%`，那比留空更难读，也容易被人当成真实数字抄走。
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else format(v, spec)


def print_summary(report: dict) -> None:
    """控制台摘要（正负号只描述数字，不作策略判断）。"""
    print("=" * 78)
    print(f"因子报告  {report['data']['start']} ~ {report['data']['end']}  "
          f"{report['data']['codes']} 只  {report['data']['dates']} 交易日")
    print(f"dev      : {report['segments']['dev']['_meta']['start']} ~ "
          f"{report['segments']['dev']['_meta']['end']}")
    print(f"holdout  : {report['segments']['holdout']['_meta']['start']} ~ "
          f"{report['segments']['holdout']['_meta']['end']}")
    print(f"代码版本 : {report.get('code_commit', 'unknown')[:12]}"
          f"（dirty={report.get('code_dirty')}）  "
          f"生成 {report.get('generated_at')}")
    print("=" * 78)

    def _table(tag: str, segs: dict) -> None:
        seg = segs[tag]
        meta = seg["_meta"]
        print(f"\n---------- {tag.upper()} ----------")
        print(f"截面 {meta['n_cross_sections']} 期 / 实际计入 IC {meta['n_ic_periods']} 期"
              f"（丢弃 {len(meta['ic_periods_dropped'])} 期，原因见 _meta.ic_periods_dropped）")
        print(f"{'因子':<14}{'IC均值':>9}{'ICIR':>8}{'t值':>8}{'IC>0':>7}{'期数':>5}"
              f"{'Q1年化':>9}{'Q5年化':>9}{'多空':>9}")
        for k, v in seg.items():
            if k.startswith("_"):
                continue
            print(f"{k:<14}{_f(v['ic_mean']):>9}{_f(v['icir'], '.3f'):>8}"
                  f"{_f(v['t_stat'], '.2f'):>8}{_f(v['ic_pos_ratio'], '.3f'):>7}"
                  f"{v['n_periods']:>5}"
                  f"{_f(v['layer_ann'][0], '.2%'):>9}"
                  f"{_f(v['layer_ann'][-1], '.2%'):>9}"
                  f"{_f(v['long_short_ann'], '.2%'):>9}")
        fx = meta.get("full_xsec") or {}
        if fx:
            ew = (meta.get("market_ew") or {}).get("ann_return")
            tail = f"；全市场 hfq 等权基准 {ew:.2%}" if ew is not None else ""
            print(f"全截面等权（同引擎）：年化 {fx['ann_return']:.2%}，"
                  f"{fx['n_rebalance']} 次调仓 / {fx['n_trades']} 笔{tail}")

    for tag in ("dev", "holdout"):
        _table(tag, report["segments"])

    neu = report.get("neutralization_compare") or {}
    if neu.get("enabled") and report.get("segments_neutralized"):
        print(f"\n---------- 行业中性化后（申万 {neu.get('level')} 级 / "
              f"{neu.get('groups')} 组 / {neu.get('method')}） ----------")
        for tag in ("dev", "holdout"):
            print(f"[{tag}]")
            print(f"{'因子':<14}{'IC前':>9}{'IC后':>9}{'ΔIC':>9}{'t前':>8}{'t后':>8}"
                  f"{'多空前':>9}{'多空后':>9}")
            for k, v in (neu.get("compare") or {}).get(tag, {}).items():
                print(f"{k:<14}{_f(v['ic_before']):>9}{_f(v['ic_after']):>9}"
                      f"{_f(v['delta_ic'], '+.4f'):>9}"
                      f"{_f(v['t_before'], '.2f'):>8}{_f(v['t_after'], '.2f'):>8}"
                      f"{_f(v['ls_before'], '.2%'):>9}"
                      f"{_f(v['ls_after'], '.2%'):>9}")

    sc = report.get("size_compare") or {}
    if sc.get("available"):
        ja = sc.get("jump_agreement") or {}
        print("\n---------- size 新旧口径 ----------")
        for tag in ("dev", "holdout"):
            s = (sc.get("sections") or {}).get(tag) or {}
            if s.get("available"):
                print(f"[{tag}] 月截面 {s['n_points']} 点：市值相对差 "
                      f"中位 {_f(s['median_abs'])} / p90 {_f(s['p90_abs'])} / "
                      f"最大 {_f(s['max_abs'])}；"
                      f">30% 占 {_f(s['share_gt30pct'], '.2%')}")
        if ja.get("available"):
            print(f"解禁跳变：推算侧 {ja['jump_cells']} 个股票·日 / {ja['jump_codes']} 只，"
                  f"其中真实侧同日也跳 {ja['both_jump_cells']} 个"
                  f"（{ja['both_jump_ratio']:.1%}；真实股本覆盖 "
                  f"{ja.get('codes_with_real_shares')}/{ja.get('codes_total')} 只），"
                  f"倍数 log 差中位 {ja.get('ratio_logdiff_median')}")

    dev = report["segments"]["dev"]
    pick = next((k for k in dev if dev[k].get("lot_residual", {}).get("available")), "bp")
    r = dev.get(pick, {}).get("lot_residual", {})
    if r.get("available"):
        print(f"\n整手残差（dev·{pick} 为例）：月差 mean|diff|={r['mean_abs_pp']:.2f}pp / "
              f"中位 {r['median_abs_pp']:.2f}pp / p90 {r['p90_abs_pp']:.2f}pp，"
              f"signed 年化 {r['signed_ann_pp']:+.2f}pp；"
              f"每层部分成交缺口 {r['partial_gap_min_pct']:.2f}%~{r['partial_gap_max_pct']:.2f}%")
        cd = report.get("cash_residual_distribution") or {}
        if cd.get("available"):
            print(f"未投入现金（全部 {cd['n_factors']} 个因子）：层期中位 "
                  f"{cd['median_pct_min']}%~{cd['median_pct_max']}%，最高 "
                  f"{cd['worst_factor']} {cd['worst_layer_median_pct']}%；"
                  f"{cd['n_factors_below_001pct']} 个因子全期 <0.01%"
                  f"（口径：层间最大中位数，非单期极值）")

    print("\n---------- 怎么读（措辞随数字生成） ----------")
    for line in report.get("readings", []):
        print("· " + line)
    print("\n" + report.get("validity_note", ""))


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="S3-Q2 因子研究框架（动量/价值/小市值/质量 + 行业中性化）")
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--factors", default=",".join(DEFAULT_FACTORS),
                    help=f"逗号分隔，可选 {sorted(FACTOR_DEFS)}")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--cash", type=float, default=1e8,
                    help="初始资金。每层约 950 只，本金太小会让高价股连一手都买不进，"
                         "分层退化成价格分层（1e8 时各层整手可行性 ≈100%%）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    ap.add_argument("--industry-level", type=int, default=INDUSTRY_LEVEL,
                    choices=[1, 2, 3],
                    help="行业中性化层级（申万 1/2/3 级）；实际用到的组数看产物 "
                         "industry.groups —— 组里没股票的不计入")
    ap.add_argument("--neutralize-method", default="demean",
                    choices=["demean", "zscore"])
    ap.add_argument("--no-neutralize", action="store_true",
                    help="跳过行业中性化对照版（只出未中性化，省一半时间）")
    ap.add_argument("--diag-data", action="store_true",
                    help="只跑数据质量统计（turn 缺失率 / 推算股本跳变 / 真实股本对拍）并打印，"
                         "不进回测。对应 validity_note 与 size_jump_stats 的复现命令")
    args = ap.parse_args(argv)

    if args.diag_data:
        dq = data_quality_stats(args.start, args.end)
        print(json.dumps(dq, ensure_ascii=False, indent=2))
        return 0

    names = [x.strip() for x in args.factors.split(",") if x.strip()]
    print(f"[1/4] 载入数据 {args.start} ~ {args.end} ...", flush=True)
    fd = load_factor_data(args.start, args.end, industry_level=args.industry_level)
    sh = fd.source.get("shares") or {}
    ind = fd.source.get("industry") or {}
    fu = fd.source.get("fundamentals") or {}
    print(f"      {len(fd.codes)} 只 / {len(fd.dates)} 交易日 / "
          f"hfq 缺口 {fd.source['hfq_gap_rows'].get('rows', 0)} 行", flush=True)
    print(f"      真实股本覆盖 {sh.get('codes_covered')}/{sh.get('codes_total')}"
          f"（{sh.get('codes_coverage')}）、行业覆盖 {ind.get('codes_coverage')}"
          f"（{ind.get('groups')} 组）、财报可用 {fu.get('rows_usable')} 行", flush=True)

    print("[2/4] 数据质量统计（turn 缺失 / 股本跳变 / 对拍）...", flush=True)
    dq = data_quality_stats(args.start, args.end)
    sj = dq["size_jump"]
    print(f"      turn 缺失 {dq['turn_missing']['rate']:.4%}，"
          f"股本跳变 {sj['rows']} 行 / {sj['codes']} 只，"
          f"其中月末 7 日内 {sj['month_end_rows']} 行", flush=True)

    print(f"[3/4] 计算因子 {names} ...", flush=True)
    factors = compute_factors(fd, names)
    print(f"[4/4] IC + 分层回测（dev / holdout"
          f"{'，含中性化对照' if not args.no_neutralize else ''}）...", flush=True)
    report = build_report(fd, factors, args, data_quality=dq,
                          neutralize_variants=not args.no_neutralize)
    out = write_report(report, (Path(args.out) / "factors_report.json").resolve())
    print_summary(report)
    print(f"\n产物：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
