"""S2 指标计算：框架 5.2 清单一条不缺。

必出指标（5.2）：年化收益 / 年化波动 / 最大回撤 / 夏普（无风险 2%）/ 卡玛 /
胜率 / 盈亏比 / 年换手率 / 超额收益（vs 基准）/ 分年度收益表 / 样本外表现。

基准（5.1、5.3）：**沪深 300 全收益** 与 **等权全市场**，两个都要给，且明确标注样本外。
   · 等权全市场：从 pool_daily 派生的行情面板直接算（含退市股，无生存偏差，3.3-4），
     日收益 = 当日有前后两日有效收盘价的全部个股简单平均收益（逐日等权再平衡）。
   · 沪深 300 全收益：需要指数行情落地到 data/raw/index/。数据没落地时**不估算、不替代**，
     直接标 missing 并把原因写进产物 —— 编一条假基准比没有基准更糟（三条底线之一）。

口径注释：
  · 年化用 252 个交易日；波动用日收益标准差 × sqrt(252)；
  · 夏普 = (年化收益 − 2%) / 年化波动；卡玛 = 年化收益 / |最大回撤|；
  · 年换手率 = 年度双边成交额（买+卖） / 平均权益 / 年数，按双边口径披露；
  · 胜率 / 盈亏比只看**已平仓**交易（卖出记录），买入不开仓不计；
  · 样本内外分界默认 2021-12-31（框架 5.3：样本内 2015–2021、样本外 2022–至今；
    本项目数据自 2018 起，样本内实为 2018–2021，界线仍画在 2021-12-31）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF = 0.02
SPLIT = "2021-12-31"


# ---------------------------------------------------------------- 基础指标
def to_series(nav: pd.DataFrame | pd.Series) -> pd.Series:
    """净值序列（index 为日期字符串）。"""
    if isinstance(nav, pd.DataFrame):
        s = pd.Series(nav["equity"].to_numpy(), index=nav["date"].tolist())
    else:
        s = nav
    return s.astype("float64")


def ann_return(s: pd.Series) -> float:
    """年化收益：按交易日数折算，不用日历天数（避免周末/长假把年化拉低）。"""
    n = len(s)
    if n < 2 or s.iloc[0] <= 0:
        return float("nan")
    years = n / TRADING_DAYS
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)


def ann_vol(s: pd.Series) -> float:
    r = s.pct_change().dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(s: pd.Series) -> dict:
    """最大回撤：返回幅度与起止日期，报告里必须给出区间（只报年化不报回撤=不合格）。"""
    peak = s.cummax()
    dd = s / peak - 1.0
    if not len(dd):
        return {"mdd": float("nan"), "peak": None, "trough": None}
    i = int(dd.values.argmin())
    peak_i = int(s.iloc[: i + 1].values.argmax())
    return {"mdd": float(dd.iloc[i]), "peak": str(s.index[peak_i]),
            "trough": str(s.index[i])}


def sharpe(s: pd.Series, rf: float = RF) -> float:
    """夏普 = (年化 − rf) / 年化波动。

    护栏：波动率数值上等于 0（恒定收益序列，std 只差浮点噪声 ~1e-16）时返回 NaN。
    不设这条护栏会得到 4.4e13 这种"极优"夏普 —— 那是浮点产物，不是业绩。
    """
    v = ann_vol(s)
    if not np.isfinite(v) or v < 1e-12:
        return float("nan")
    return float((ann_return(s) - rf) / v)


def calmar(s: pd.Series) -> float:
    """卡玛 = 年化 / |最大回撤|；无回撤（含浮点噪声级回撤）时 NaN，不写 inf。"""
    m = max_drawdown(s)["mdd"]
    if not np.isfinite(m) or abs(m) < 1e-12:
        return float("nan")
    return float(ann_return(s) / abs(m))


def yearly_returns(s: pd.Series) -> dict[str, float]:
    """分年度收益：每年末净值 / 上年末净值 - 1；首年以本年首个净值为基准。"""
    idx = pd.to_datetime(pd.Index(s.index))
    out: dict[str, float] = {}
    base = float(s.iloc[0])
    for y, grp in s.groupby(idx.year):
        last = float(grp.iloc[-1])
        out[str(y)] = round(last / base - 1.0, 6)
        base = last
    return out


def _closed_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """已平仓交易 = 主动卖出 + 退市强制清算。

    清算（side='liquidate'）同样把仓位变成现金、实现盈亏，漏掉它会让胜率/盈亏比
    只统计"跑掉了的"那些股票，退市股被静默排除 —— 正是生存偏差的另一种形态。
    """
    if trades is None or trades.empty or "side" not in trades.columns:
        return pd.DataFrame()
    return trades[trades["side"].isin(("sell", "liquidate"))]


def win_rate(trades: pd.DataFrame) -> float:
    """胜率：已平仓（卖出 + 清算）交易中盈利笔数占比。"""
    closed = _closed_trades(trades)
    if closed.empty:
        return float("nan")
    return float((closed["pnl"] > 0).mean())


def profit_loss_ratio(trades: pd.DataFrame) -> float:
    """盈亏比：平均盈利 / 平均亏损（绝对值）。没有亏损笔时返回 NaN，不写 999。"""
    closed = _closed_trades(trades)
    if closed.empty:
        return float("nan")
    win = closed.loc[closed["pnl"] > 0, "pnl"]
    loss = closed.loc[closed["pnl"] < 0, "pnl"]
    if len(win) == 0 or len(loss) == 0:
        return float("nan")
    return float(win.mean() / abs(loss.mean()))


def turnover_rate(nav: pd.DataFrame, trades: pd.DataFrame) -> float:
    """年换手率（双边）：年度双边成交额 / 平均权益 / 年数。"""
    if trades is None or trades.empty or nav.empty:
        return 0.0
    amount = float(trades["amount"].sum())
    avg_equity = float(nav["equity"].mean())
    years = len(nav) / TRADING_DAYS
    if avg_equity <= 0 or years <= 0:
        return float("nan")
    return float(amount / avg_equity / years)


def excess(strategy: pd.Series, bench: pd.Series | None) -> dict:
    """超额收益：年化差 + 累计差。基准缺失时返回 NaN，不拿别的指数顶替。"""
    if bench is None or len(bench) < 2:
        return {"ann_excess": float("nan"), "cum_excess": float("nan")}
    b = bench.reindex(strategy.index).ffill()
    if b.isna().all():
        return {"ann_excess": float("nan"), "cum_excess": float("nan")}
    return {
        "ann_excess": round(ann_return(strategy) - ann_return(b.dropna()), 6),
        "cum_excess": round(float(strategy.iloc[-1] / strategy.iloc[0]
                                  - b.iloc[-1] / b.iloc[0]), 6),
    }


# ---------------------------------------------------------------- 基准
def equal_weight_market(panel) -> pd.Series:
    """等权全市场基准：每日全部个股等权，收益 = 当日个股收益的简单平均。

    为什么含退市股：面板本身来自 pool_daily 派生的宽表，退市股保留到最后交易日（3.3-4），
    用"今天的成分股"回测历史会系统性高估收益。停牌股当日无收益（价格不变），自然计 0。

    已知口径差异（003R4 / 审计 B4-06，**只记录不改逻辑**）：hfq 缺口格的收盘价是 NaN，
    该股缺口当日的收益因此被 `nanmean` **剔除**（等于当天不参与等权），而不是按 0 收益计入。
    实测影响 < 0.01pp（缺口仅 1,942 行 / 2 只 / 9 年面板）。总工裁定：S2 不动这条逻辑，
    等 004 因子框架设计时统一处理（届时要么缺口股整天剔除，要么按冻结价计 0 收益，两者择一）。
    """
    px = panel.close_hfq
    prev = np.vstack([np.full((1, px.shape[1]), np.nan), px[:-1]])
    ok = ~np.isnan(px) & ~np.isnan(prev) & (prev > 0)
    ret = np.where(ok, px / np.where(ok, prev, 1.0) - 1.0, np.nan)
    import warnings
    with warnings.catch_warnings():          # 首日无前收，整行为 NaN，属预期
        warnings.simplefilter("ignore", RuntimeWarning)
        daily = np.nanmean(np.where(ok, ret, np.nan), axis=1)
    daily = np.nan_to_num(daily, nan=0.0)
    nav = np.cumprod(1.0 + daily)
    return pd.Series(nav, index=panel.dates, name="equal_weight")


HS300_PATTERNS = ("hs300*.parquet", "index_hs300*.parquet", "hs300_total_return*.parquet")


def hs300_total_return(dates: list[str], raw_dir=None) -> tuple[pd.Series | None, str]:
    """沪深 300 全收益：从 data/raw/index/ 读取已落地的指数行情。

    数据未落地时返回 (None, 原因)，绝不拿价格指数或自造组合冒充"全收益"：
    价格指数不含分红再投资，长期会低估约 2%/年，冒充等于篡改基准。
    落地要求（供 S1 补数据用）：data/raw/index/hs300_total_return_{抓取日}.parquet，
    列至少含 date / close，且是**全收益**口径。
    """
    from pathlib import Path
    raw_dir = Path(raw_dir) if raw_dir else Path(__file__).resolve().parent.parent / "data" / "raw" / "index"
    if not raw_dir.exists():
        return None, "missing: data/raw/index/ 不存在，沪深300全收益指数行情未落地"
    files: list[Path] = []
    for pat in HS300_PATTERNS:
        files += sorted(raw_dir.glob(pat))
    if not files:
        return None, f"missing: {raw_dir} 下无沪深300指数文件（列需含 date/close）"
    d = pd.read_parquet(sorted(files)[-1])  # 文件名带抓取日期，取最新
    if not {"date", "close"}.issubset(d.columns):
        return None, f"missing: {files[-1].name} 缺 date/close 列"
    d = d[(d["date"] >= dates[0]) & (d["date"] <= dates[-1])]
    if d.empty:
        return None, "missing: 指数文件不含回测区间"
    s = pd.Series(d["close"].to_numpy(dtype="float64"), index=d["date"].tolist())
    return (s / s.iloc[0]).reindex(dates).ffill(), "ok"


def build_benchmarks(panel, dates: list[str]) -> dict:
    """两个基准一起给，缺哪个都要在产物里写明原因。"""
    ew = equal_weight_market(panel).reindex(dates).ffill()
    ew = ew / ew.iloc[0]
    hs, hs_status = hs300_total_return(dates)
    out = {
        "equal_weight": ew,
        "hs300_total_return": hs,
        "summary": {
            "equal_weight": {"status": "ok", "rows": int(len(ew))},
            "hs300_total_return": {"status": hs_status, "rows": 0 if hs is None else int(len(hs))},
        },
    }
    return out


# ---------------------------------------------------------------- 汇总
def compute_metrics(nav: pd.DataFrame, trades: pd.DataFrame, bench: dict,
                    rf: float = RF, split: str = SPLIT) -> dict:
    s = to_series(nav)
    ew = bench.get("equal_weight")
    hs = bench.get("hs300_total_return")
    oos_mask = pd.to_datetime(pd.Index(s.index)) > pd.Timestamp(split)
    oos = s[oos_mask]
    ins = s[~oos_mask]

    def _seg(x: pd.Series) -> dict:
        if len(x) < 2:
            return {"ann_return": float("nan"), "ann_vol": float("nan"),
                    "mdd": float("nan"), "sharpe": float("nan")}
        return {"ann_return": ann_return(x), "ann_vol": ann_vol(x),
                "mdd": max_drawdown(x)["mdd"], "sharpe": sharpe(x, rf),
                "days": int(len(x))}

    md = max_drawdown(s)
    m = {
        "ann_return": ann_return(s),
        "ann_vol": ann_vol(s),
        "mdd": md["mdd"],
        "mdd_peak": md["peak"],
        "mdd_trough": md["trough"],
        "sharpe": sharpe(s, rf),
        "calmar": calmar(s),
        "win_rate": win_rate(trades),
        "profit_loss_ratio": profit_loss_ratio(trades),
        "turnover_annual": turnover_rate(nav, trades),
        "years": round(len(s) / TRADING_DAYS, 4),
        "total_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
        "n_trades": int(len(trades)) if trades is not None else 0,
        "yearly_returns": yearly_returns(s),
        "in_sample": {"range": [str(s.index[0]), split], **_seg(ins)},
        "out_of_sample": {"range": [str(oos.index[0]) if len(oos) else None,
                                    str(s.index[-1])], **_seg(oos)},
        "rf": rf,
        "split": split,
        "benchmarks": {},
    }
    for name, series in (("equal_weight", ew), ("hs300_total_return", hs)):
        if series is None:
            m["benchmarks"][name] = {
                "status": bench["summary"][name]["status"],
                "ann_return": None, "mdd": None, "excess": None}
            continue
        b = series.reindex(s.index).ffill().dropna()
        b = b / b.iloc[0]
        m["benchmarks"][name] = {
            "status": "ok",
            "ann_return": ann_return(b),
            "ann_vol": ann_vol(b),
            "mdd": max_drawdown(b)["mdd"],
            "total_return": float(b.iloc[-1] / b.iloc[0] - 1.0),
            **excess(s, b),
        }
    return m


def format_card(m: dict) -> str:
    """指标卡（终端/报告通用，纯文本）。缺数一律显示"缺失"，不用 0 顶替。"""
    def pct(x, nd=2):
        return "缺失" if x is None or not np.isfinite(x) else f"{x*100:.{nd}f}%"

    def num(x, nd=2):
        return "缺失" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    lines = [
        "================ 指标卡 ================",
        f"区间        {m['years']:.2f} 年 | 累计 {pct(m['total_return'])} | 交易 {m['n_trades']} 笔",
        f"年化收益    {pct(m['ann_return'])}      年化波动 {pct(m['ann_vol'])}",
        f"最大回撤    {pct(m['mdd'])}（{m['mdd_peak']} → {m['mdd_trough']}）",
        f"夏普(2%)    {num(m['sharpe'])}         卡玛     {num(m['calmar'])}",
        f"胜率        {pct(m['win_rate'])}          盈亏比   {num(m['profit_loss_ratio'])}",
        f"年换手(双边){pct(m['turnover_annual'])}",
        "---------------- 分年度 ----------------",
    ]
    for y, r in m["yearly_returns"].items():
        lines.append(f"  {y}  {pct(r)}")
    lines.append("---------------- 基准 ----------------")
    for name, cn in (("hs300_total_return", "沪深300全收益"), ("equal_weight", "等权全市场")):
        b = m["benchmarks"][name]
        if b.get("status") != "ok":
            lines.append(f"  {cn}  {b['status']}")
        else:
            lines.append(f"  {cn}  年化 {pct(b['ann_return'])} | 回撤 {pct(b['mdd'])} | "
                         f"超额 {pct(b.get('ann_excess'))}")
    o = m["out_of_sample"]
    lines.append("------------- 样本外(2022起) -------------")
    lines.append(f"  年化 {pct(o['ann_return'])} | 波动 {pct(o['ann_vol'])} | "
                 f"回撤 {pct(o['mdd'])} | 夏普 {num(o['sharpe'])}")
    lines.append("----------------------------------------")
    # 审计 B3-02 / B4-03：指标卡最容易被截图传播，必须自带边界声明，防止被当成 alpha 证据
    lines.append("  注：自检脚手架结果，非策略有效性/alpha 证据；")
    lines.append("      seed 40~50 共 11 个：年化 5.1113%~7.0189%，级差 1.9076pp，"
                 "中位 5.9884%，")
    lines.append("      seed=42 恰在中位（升序第 6/11，54.5 分位）；"
                 "超额另有整手倾斜与可交易性过滤")
    lines.append("      口径见 result.json.extra.validity_note")
    lines.append("========================================")
    return "\n".join(lines)
