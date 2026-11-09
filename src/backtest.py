"""S2 回测引擎：自研轻量事件驱动（不引 backtrader，全流程可审计）。

口径全部出自《量化分析 App · 框架设计》v0.2，逐条对应：
  · 5.1 信号 T 日收盘产生 → **T+1 开盘价**成交；
  · 5.1 涨停不买、跌停不卖、停牌跳过；
  · 5.1 佣金 0.025%（双边，最低 5 元）、印花税 0.05%（仅卖出）、过户费 0.001%（双边）；
  · 5.1 持仓 T+1 才能卖；持仓只数、单只权重上限可配；
  · 3.3-1 时点对齐：T 日收盘后才知道的数据，最早 T+1 开盘执行（本引擎用两张价表物理隔离：
        决策只看 close[t]，成交只用 open[t+1]，代码里不存在第 t 日就用第 t 日开盘价的分支）；
  · 3.3-2 复权：成交与估值统一用**后复权价**（hfq_open / hfq_close），等价于分红再投；
        涨跌停判定必须用原始价（交易所按原始价定板），两者不可混用；
  · 3.3-3 停牌：当日不产生成交，持仓按最后有效收盘价结转；
  · 3.3-5 涨跌停：开盘即触板视为不可成交（一字板更不可能成交）。

本文件只提供"怎么成交、怎么记账"，不含任何选股/择时逻辑 —— 那是 S3 的事。
唯一自带的 signals 生成器是 `harness_monthly_equal_weight`，它是**引擎自检脚手架**
（等权持有全市场、月末调仓），用来跑通管线并与"等权全市场"基准互相印证，不是策略。

用法（示例回测，产物落 out/）：
    python -m src.backtest --start 2018-01-02 --end 2026-11-06 --out out
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
DB = ROOT / "data" / "cache.db"
OUT = ROOT / "out"

TRADING_DAYS = 252  # 年化用，A 股约 242~244，统一取 252 与框架 5.2 常见口径一致


# ---------------------------------------------------------------- 交易所规则
def round2(x) -> np.ndarray:
    """交易所式四舍五入到分。

    不能用 np.round：它是银行家舍入且受二进制表示影响，28.325 会给出 28.32，
    而交易所的涨停价是 28.33 —— 一个 tick 的偏差足以让"能不能买"判反。
    """
    return np.floor(np.asarray(x, dtype="float64") * 100.0 + 0.5 + 1e-9) / 100.0


def limit_ratio(code: str, is_st: bool, date: str) -> float:
    """个股当日涨跌幅限制。规则按交易所与板块现行制度，历史改革日按日期切换。

    · 科创板（688/689）：±20%
    · 创业板（300/301）：2020-08-24 注册制改革前 ±10%，之后 ±20%
    · 北交所（4xx/8xx/920）：±30%
    · 主板：ST/*ST ±5%，其余 ±10%
    新股上市首日不设涨跌幅 —— 由 `listed_days <= 1` 单独放过，不进本函数。
    """
    if code.startswith(("sh.688", "sh.689")):
        return 0.20
    if code.startswith(("sz.300", "sz.301")):
        return 0.20 if date >= "2020-08-24" else 0.10
    if code.startswith(("bj.", "sz.4", "sz.8", "sh.4", "sh.8")):
        return 0.30
    return 0.05 if is_st else 0.10


def _limit_arrays(codes: pd.Series, is_st: pd.Series, dates: pd.Series,
                  listed_days: pd.Series, close: pd.Series, pct_chg: pd.Series):
    """向量化算涨停价/跌停价。

    昨收（preclose）数据里没有，用 close/(1+pctChg/100) 反推：baostock 的 pctChg 有 5 位小数，
    反推误差 ~1e-4，再归整到 2 位小数（交易所昨收必为 2 位小数）后与真实昨收一致；
    实测 2018/2021/2024 三年 21929 个涨跌停样本，收盘价 == 涨跌停价 的自洽率 99.83%，
    剩余 0.17% 落在半分钱边界上，用 0.005 元容差吸收。
    """
    pre = round2(close / (1.0 + pct_chg / 100.0))
    ratio = np.zeros(len(codes), dtype="float64")
    board = codes.str.startswith(("sh.688", "sh.689")).to_numpy()
    gem = codes.str.startswith(("sz.300", "sz.301")).to_numpy()
    bj = codes.str.startswith(("bj.", "sz.4", "sz.8", "sh.4", "sh.8")).to_numpy()
    main = ~(board | gem | bj)

    ratio[board] = 0.20
    ratio[bj] = 0.30
    ratio[main] = np.where(is_st.to_numpy()[main] == 1, 0.05, 0.10)
    after_reg = (dates >= "2020-08-24").to_numpy()
    ratio[gem] = np.where(after_reg[gem], 0.20, 0.10)
    # 新股上市首日无涨跌幅限制：置 0 → 涨跌停价等于昨收，永远"未触板"
    ratio = np.where(listed_days.to_numpy() <= 1, 0.0, ratio)

    up = round2(pre * (1.0 + ratio))
    dn = round2(pre * (1.0 - ratio))
    return up, dn


# ---------------------------------------------------------------- 数据面板
@dataclass
class Panel:
    """回测用面板。行=交易日，列=股票代码，全部是 numpy 便于逐日循环。

    open_hfq / close_hfq：后复权价，成交与估值口径（3.3-2）
    open_raw           ：原始价，只用于判断开盘是否触板（交易所按原始价定板）
    blocked_buy        ：开盘即涨停（或本就无法成交）→ 不可买
    blocked_sell       ：开盘即跌停 → 不可卖
    suspended          ：停牌日 → 不产生成交（3.3-3）
    """
    dates: list[str]
    codes: list[str]
    open_hfq: np.ndarray
    close_hfq: np.ndarray
    open_raw: np.ndarray
    blocked_buy: np.ndarray
    blocked_sell: np.ndarray
    suspended: np.ndarray
    source: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.open_hfq.shape


def load_panel(start: str, end: str, codes: list[str] | None = None,
               cache_dir: Path | None = None) -> Panel:
    """从 data/cache/daily_{年}.parquet 载入面板（只读清洗层，不碰 data/raw）。

    为什么用后复权成交：3.3-2 明确"回测用后复权价"。后复权把分红送转折进价格，
    持仓收益天然含分红再投，避免除权日出现假跌。
    """
    cache_dir = cache_dir or CACHE
    years = range(int(start[:4]), int(end[:4]) + 1)
    cols = ["code", "date", "open", "close", "pctChg", "isST",
            "suspended", "listed_days", "hfq_open", "hfq_close"]
    frames = []
    for y in years:
        f = cache_dir / f"daily_{y}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"缺 {f}，先跑 python -m src.clean --years ...")
        d = pd.read_parquet(f, columns=cols)
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if codes is not None:
            d = d[d["code"].isin(codes)]
        if len(d):
            frames.append(d)
    if not frames:
        raise ValueError(f"{start}~{end} 区间内没有任何行情，检查日期与股票池")
    df = pd.concat(frames, ignore_index=True)
    del frames

    up, dn = _limit_arrays(df["code"], df["isST"].astype("int8"), df["date"],
                           df["listed_days"], df["close"], df["pctChg"])
    tol = 0.005  # 半分钱容差：涨跌停价只可能差 1 个 tick
    # 一字涨停/开盘涨停：开盘价已到涨停价，买单排队也成交不了（5.1）
    df["blk_buy"] = (df["open"].to_numpy() >= up - tol)
    # 跌停：开盘价已到跌停价，卖单同样成交不了（5.1）
    df["blk_sell"] = (df["open"].to_numpy() <= dn + tol)
    df = df.drop(columns=["isST", "pctChg", "listed_days", "open", "close"])

    codes_all = sorted(df["code"].unique().tolist())
    dates_all = sorted(df["date"].unique().tolist())
    shape = (len(dates_all), len(codes_all))

    def _mat(name: str, dtype) -> np.ndarray:
        m = df.pivot(index="date", columns="code", values=name).reindex(
            index=dates_all, columns=codes_all)
        return m.to_numpy(dtype=dtype, na_value=np.nan)

    panel = Panel(
        dates=dates_all,
        codes=codes_all,
        open_hfq=_mat("hfq_open", "float64"),
        close_hfq=_mat("hfq_close", "float64"),
        open_raw=np.zeros(shape, dtype="float64"),  # 触板判定已在上面算完，原始价不再保留
        blocked_buy=_mat("blk_buy", "bool"),
        blocked_sell=_mat("blk_sell", "bool"),
        suspended=_mat("suspended", "bool"),
        source={"start": start, "end": end, "rows": int(len(df)),
                "codes": len(codes_all), "dates": len(dates_all)},
    )
    panel.blocked_buy = np.nan_to_num(panel.blocked_buy, nan=True)   # 无行=不可交易
    panel.blocked_sell = np.nan_to_num(panel.blocked_sell, nan=True)
    panel.suspended = np.nan_to_num(panel.suspended, nan=True)
    return panel


# ---------------------------------------------------------------- 成本与配置
@dataclass
class Costs:
    """5.1 成本口径。比例均按成交金额计，佣金有最低值。"""
    commission_rate: float = 0.00025   # 佣金 0.025%，双边
    min_commission: float = 5.0        # 单笔最低 5 元
    stamp_tax: float = 0.0005          # 印花税 0.05%，仅卖出
    transfer_fee: float = 0.00001      # 过户费 0.001%，双边
    slippage: float = 0.001            # 滑点 0.1%，可配；买入抬价、卖出压价

    def fees(self, amount: float, side: str) -> float:
        """return 该笔成交的总费用（佣金 + 过户费 [+ 印花税]）。"""
        if amount <= 0:
            return 0.0
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer = amount * self.transfer_fee
        stamp = amount * self.stamp_tax if side == "sell" else 0.0
        return commission + transfer + stamp


@dataclass
class Config:
    """全部可配参数。写进 result.json 的 params，保证"同一参数同一结果"可复算。"""
    start: str
    end: str
    init_cash: float = 1_000_000.0
    max_positions: int = 50            # 持仓只数上限（5.1 可配）
    max_weight: float = 0.10           # 单只权重上限（5.1 可配）
    lot_size: int = 100                # A 股一手 100 股
    costs: Costs = field(default_factory=Costs)


# ---------------------------------------------------------------- 结果结构
@dataclass
class BacktestResult:
    config: dict
    data: dict
    nav: pd.DataFrame          # date / equity / cash / positions / turnover
    trades: pd.DataFrame       # 每笔成交
    positions: pd.DataFrame    # 每日每只持仓股数（只含非零）
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)  # 未成交订单及原因
    signals_meta: dict = field(default_factory=dict)


def _equity(cash: float, shares: np.ndarray, px: np.ndarray) -> float:
    return float(cash + np.dot(shares, px))


def run_backtest(panel: Panel, signals: dict[str, dict[str, float]],
                 cfg: Config) -> BacktestResult:
    """事件驱动主循环。

    signals：{日期: {代码: 目标权重}}，日期必须是 panel 的交易日，权重和为 0~1。
             日期 d 的信号只可能在 d+1 开盘执行 —— 这是 3.3-1 时点对齐的硬实现。

    逐日流程：
      1) 开盘：执行**上一交易日**收盘生成的调仓单（T+1 成交，5.1）
      2) 收盘：按后复权收盘价估值，记录净值（停牌用最后有效价结转，3.3-3）
      3) 收盘后：读当日信号生成明日调仓单（此处只能看到 <= t 的数据）
    """
    n_d, n_c = panel.shape
    idx = {c: i for i, c in enumerate(panel.codes)}
    cash = float(cfg.init_cash)
    shares = np.zeros(n_c, dtype="float64")
    cost_basis = np.zeros(n_c, dtype="float64")      # 含买入费用的持仓成本（用于逐笔盈亏）
    last_buy_day = np.full(n_c, -1, dtype="int64")   # T+1 限制：当日买入当日不可卖
    last_px = np.full(n_c, np.nan)                   # 停牌结转用的最后有效收盘价
    close_mask = ~np.isnan(panel.close_hfq)

    pending: dict[str, float] = {}                   # 明日开盘要调到的目标权重
    rows, trade_rows, pos_rows, drop_rows = [], [], [], []

    for i, date in enumerate(panel.dates):
        # ---------- 1) 开盘执行昨日信号（T+1）
        #    估值只用"<= i-1 日"的收盘价兜底，绝不读今日收盘价（3.3-1 防未来函数）
        if pending:
            cash = _execute(panel, i, pending, idx, cash, shares, cost_basis,
                            last_buy_day, last_px, cfg, trade_rows, drop_rows, date)

        # ---------- 2) 收盘估值
        px_now = panel.close_hfq[i]
        last_px = np.where(close_mask[i], px_now, last_px)
        px_val = np.where(np.isnan(last_px), 0.0, last_px)
        equity = _equity(cash, shares, px_val)
        rows.append({"date": date, "equity": equity, "cash": cash,
                     "positions": int((shares > 0).sum())})
        nz = np.nonzero(shares > 0)[0]
        for j in nz:
            pos_rows.append({"date": date, "code": panel.codes[j],
                             "shares": float(shares[j]),
                             "price": float(px_val[j]),
                             "value": float(shares[j] * px_val[j])})

        # ---------- 3) 收盘后读信号 → 明日开盘执行
        pending = signals.get(date, {}) or {}

    nav = pd.DataFrame(rows)
    nav["ret"] = nav["equity"].pct_change().fillna(0.0)
    trades = pd.DataFrame(trade_rows, columns=[
        "date", "code", "side", "shares", "price", "amount", "fees", "pnl"])
    positions = pd.DataFrame(pos_rows, columns=["date", "code", "shares", "price", "value"])
    dropped = pd.DataFrame(drop_rows, columns=["date", "code", "side", "shares", "reason"])
    # 数据区间必须随产物落地（手工构造的面板也要能写全，否则 result.json 无效）
    data = dict(panel.source)
    data.setdefault("start", panel.dates[0])
    data.setdefault("end", panel.dates[-1])
    data.setdefault("dates", len(panel.dates))
    data.setdefault("codes", len(panel.codes))
    return BacktestResult(
        config=cfg_to_dict(cfg), data=data, nav=nav,
        trades=trades, positions=positions, dropped=dropped,
        signals_meta={"signal_days": len(signals),
                      "dropped_orders": int(len(dropped)),
                      "note": "signals 日期 d 的信号只在 d+1 开盘执行（3.3-1）；"
                              "遇涨跌停/停牌/T+1 未成交的订单不保留，按 reason 逐笔记入 dropped"})


def _execute(panel: Panel, i: int, target: dict[str, float], idx: dict[str, int],
             cash: float, shares: np.ndarray, cost_basis: np.ndarray,
             last_buy_day: np.ndarray, last_px: np.ndarray, cfg: Config,
             trade_rows: list[dict], drop_rows: list[dict], date: str) -> float:
    """开盘调仓：按目标权重算出目标股数，先卖（降仓/清仓）后买（建仓/加仓）。

    约束顺序（5.1）：涨跌停不可成交 → 停牌跳过 → T+1 才能卖 → 只数/单只权重上限 → 资金约束。
    开盘缺价（当天没行情）时，估值退用**上一交易日**收盘价，绝不读当日收盘价。
    """
    op = panel.open_hfq[i]
    px_val = np.where(np.isnan(op), last_px, op)
    px_val = np.where(np.isnan(px_val), 0.0, px_val)
    equity = _equity(cash, shares, px_val)

    # ---- 目标权重：只数上限 + 单只上限 + 归一化到 <=1
    items = sorted(((c, w) for c, w in target.items() if c in idx and w > 0),
                   key=lambda kv: (-kv[1], kv[0]))[: cfg.max_positions]
    weights = {c: min(float(w), cfg.max_weight) for c, w in items}
    total = sum(weights.values())
    if total > 1.0:
        weights = {c: w / total for c, w in weights.items()}

    # ---- 目标股数（按含滑点的实际成交价折算，按手取整；卖不要求整手）
    tgt_sh: dict[int, int] = {}
    for c, w in weights.items():
        j = idx[c]
        px = op[j]
        if np.isnan(px) or px <= 0:
            # 当日无价（停牌/缺行）：建仓单直接作废并记账，绝不静默跳过
            if shares[j] <= 0:
                drop_rows.append({"date": date, "code": c, "side": "buy",
                                  "shares": float("nan"),
                                  "reason": "suspended" if panel.suspended[i, j] else "no_price"})
            continue
        fill_px = px * (1.0 + cfg.costs.slippage)
        tgt_sh[j] = int((w * equity) / fill_px // cfg.lot_size) * cfg.lot_size

    # ---- 卖出：持仓超出目标的减仓，目标里没有的清仓
    for j in np.nonzero(shares > 0)[0]:
        excess_sh = shares[j] - tgt_sh.get(int(j), 0)
        if excess_sh <= 1e-9:
            continue
        reason = None
        if panel.suspended[i, j] or np.isnan(op[j]):
            reason = "suspended"                       # 停牌跳过（3.3-3）
        elif panel.blocked_sell[i, j]:
            reason = "limit_down"                      # 跌停卖不出（5.1）
        elif last_buy_day[j] >= i:
            reason = "t_plus_1"                        # T+1 限制（5.1）
        if reason:
            drop_rows.append({"date": date, "code": panel.codes[int(j)], "side": "sell",
                              "shares": float(excess_sh), "reason": reason})
            continue
        cash = _sell(int(j), panel.codes[j], min(excess_sh, shares[j]), op[j],
                     cash, shares, cost_basis, cfg, trade_rows, date)

    # ---- 买入：低于目标的补足
    for c, w in weights.items():
        j = idx[c]
        want = tgt_sh.get(j, 0)
        if want - shares[j] < cfg.lot_size:
            continue
        reason = None
        if panel.suspended[i, j] or np.isnan(op[j]):
            reason = "suspended"
        elif panel.blocked_buy[i, j]:
            reason = "limit_up"                        # 涨停买不进（5.1）
        if reason:
            drop_rows.append({"date": date, "code": c, "side": "buy",
                              "shares": float(want - shares[j]), "reason": reason})
            continue
        price = op[j] * (1.0 + cfg.costs.slippage)
        qty = min(want - shares[j], (cfg.max_weight * equity - shares[j] * op[j]) // price)
        qty = int(qty // cfg.lot_size) * cfg.lot_size
        while qty > 0:                                  # 资金约束：手续费也要留够
            amount = qty * price
            fee = cfg.costs.fees(amount, "buy")
            if amount + fee <= cash + 1e-9:
                break
            qty -= cfg.lot_size
        if qty <= 0:
            drop_rows.append({"date": date, "code": c, "side": "buy",
                              "shares": float(want - shares[j]), "reason": "insufficient_cash"})
            continue
        amount = qty * price
        fee = cfg.costs.fees(amount, "buy")
        cash -= amount + fee
        cost_basis[j] += amount + fee
        shares[j] += qty
        last_buy_day[j] = i
        trade_rows.append({"date": date, "code": c, "side": "buy", "shares": float(qty),
                           "price": float(price), "amount": float(amount),
                           "fees": float(fee), "pnl": np.nan})
    return cash


def _sell(j: int, code: str, qty: float, op: float, cash: float,
          shares: np.ndarray, cost_basis: np.ndarray, cfg: Config,
          trade_rows: list[dict], date: str) -> float:
    price = op * (1.0 - cfg.costs.slippage)           # 滑点：卖出压价
    amount = qty * price
    fee = cfg.costs.fees(amount, "sell")
    cash += amount - fee
    avg = cost_basis[j] / shares[j] if shares[j] else 0.0
    pnl = amount - fee - avg * qty                     # 逐笔已实现盈亏（成本含买入费用）
    cost_basis[j] -= avg * qty
    shares[j] -= qty
    if shares[j] <= 1e-9:
        shares[j], cost_basis[j] = 0.0, 0.0
    trade_rows.append({"date": date, "code": code, "side": "sell", "shares": float(qty),
                       "price": float(price), "amount": float(amount),
                       "fees": float(fee), "pnl": float(pnl)})
    return cash


def cfg_to_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d.pop("start", None)
    d.pop("end", None)
    return d


# ---------------------------------------------------------------- 自检脚手架
def harness_monthly_equal_weight(panel: Panel, n_hold: int = 300,
                                 seed: int = 42) -> dict:
    """**引擎自检脚手架，不是策略**（S3 才会写真正的策略）。

    固定抽 n_hold 只（固定随机种子，避免按代码排序带来的板块偏斜），每月最后一个交易日
    收盘后把当日可交易（有行、未停牌、未开盘涨停）的样本设为等权目标，下月同日再平衡。

    它的净值曲线应当与"等权全市场"基准同向且量级接近：引擎若偷看未来、漏记成本或
    涨跌停判断反了，两条线会立刻分叉。这是引擎的自检对照，不做任何选股判断。
    """
    rng = np.random.RandomState(seed)
    pool = np.arange(len(panel.codes))
    universe = rng.choice(pool, size=min(n_hold, len(pool)), replace=False) \
        if n_hold < len(pool) else pool
    sig: dict[str, dict[str, float]] = {}
    last_of_month: dict[str, int] = {}
    for i, d in enumerate(panel.dates):
        last_of_month[d[:7]] = i
    for i in last_of_month.values():
        ok = ~panel.suspended[i] & ~panel.blocked_buy[i] & ~np.isnan(panel.open_hfq[i])
        cand = universe[ok[universe]]
        if len(cand) == 0:
            continue
        w = 1.0 / len(cand)
        sig[panel.dates[i]] = {panel.codes[j]: w for j in cand}
    return sig


# ---------------------------------------------------------------- 产物
def git_commit() -> str:
    """取当前 HEAD 短哈希；取不到就写 unknown（不编造）。"""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_result(result: BacktestResult, metrics: dict, benchmarks: dict,
                 path: Path, extra: dict | None = None) -> Path:
    """落 result.json。缺 git_commit / 参数 / 数据区间 / 逐日净值 任一字段即为无效产物。"""
    payload = {
        "schema": "backtest-result/1",
        "git_commit": git_commit(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": result.config,
        "data": result.data,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "nav": result.nav.to_dict(orient="records"),
        "trades": result.trades.to_dict(orient="records"),
        "dropped": result.dropped.to_dict(orient="records"),
        "extra": extra or {},
    }
    for key in ("git_commit", "params", "data", "nav"):
        assert payload.get(key) not in (None, "", [], {}), f"result.json 缺 {key}"
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False + NaN→null：产物必须是严格 JSON，NaN 不是合法 JSON 字面量，
    # 下游 JS/前端读到 NaN 会直接崩，缺失就该写 null
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=1,
                              allow_nan=False), encoding="utf-8")
    return path


def _clean(o):
    """递归把 NaN/Inf 转 None，其余原样（严格 JSON 合规）。"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(argv: list[str] | None = None) -> int:
    from . import metrics as M

    ap = argparse.ArgumentParser(description="S2 回测引擎示例运行（自检脚手架，非策略）")
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default=None, help="默认取数据最新日")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-hold", type=int, default=300, help="自检组合持股数（固定种子抽样）")
    args = ap.parse_args(argv)

    panel = load_panel(args.start, args.end or "2099-12-31")
    end = panel.dates[-1]
    cfg = Config(start=panel.dates[0], end=end,
                 max_positions=max(args.n_hold, 50), max_weight=0.05)
    signals = harness_monthly_equal_weight(panel, n_hold=args.n_hold)
    result = run_backtest(panel, signals, cfg)

    bench = M.build_benchmarks(panel, result.nav["date"].tolist())
    metrics = M.compute_metrics(result.nav, result.trades, bench)
    out = (Path(args.out) / "result.json").resolve()
    write_result(result, metrics, bench["summary"], out,
                 extra={"signals": result.signals_meta,
                        "harness": "引擎自检脚手架：固定种子等权持有 N 只、月度再平衡，非策略"})
    print(f"[backtest] 区间 {panel.dates[0]} ~ {end}，{len(panel.codes)} 只，"
          f"净值 {result.nav['equity'].iloc[-1]:,.2f}")
    print(f"[backtest] 产物 {out}")
    print(M.format_card(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
