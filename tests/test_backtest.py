"""S2 回测引擎验收测试。

覆盖框架第 7 节"验收标准"里的两条硬指标，外加成交约束与成本口径：
  1. 无未来函数：把数据截断到 2020-12-31，2020 年之前（含）逐日净值与持仓必须与全样本完全一致。
  2. 可复算：同一数据同一参数跑两次，净值逐日完全相等（浮点逐位相同，不是近似）。
  3. 涨跌停：一字涨停买不进、跌停卖不出（人工构造数据）。
  4. 停牌：停牌日不产生成交。
  5. T+1：当日买入当日不可卖。
  6. 手工对账：3 只股票 6 个交易日，逐笔手算净值与引擎输出误差 < 0.01。
  7. 产物：result.json 必含 git_commit / 全部参数 / 数据区间 / 逐日净值。

红线：本文件不许放宽断言；截断测试必须逐日相等，不是"趋势一致"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import (Costs, Config, Panel, load_panel, run_backtest,  # noqa: E402
                          write_result)
from src import metrics as M  # noqa: E402


# ---------------------------------------------------------------- 工具
def make_panel(dates, codes, open_, close, blocked_buy=None, blocked_sell=None,
               suspended=None) -> Panel:
    """人工面板：全部按后复权价给，涨跌停/停牌用布尔矩阵指定。"""
    o = np.asarray(open_, dtype="float64")
    c = np.asarray(close, dtype="float64")
    shape = o.shape
    z = np.zeros(shape, dtype=bool)
    return Panel(dates=list(dates), codes=list(codes), open_hfq=o, close_hfq=c,
                 open_raw=o,
                 blocked_buy=z.copy() if blocked_buy is None else np.asarray(blocked_buy),
                 blocked_sell=z.copy() if blocked_sell is None else np.asarray(blocked_sell),
                 suspended=z.copy() if suspended is None else np.asarray(suspended))


def momentum_signals(panel: Panel, every: int = 5, lookback: int = 20,
                     top: int = 10) -> dict[str, dict[str, float]]:
    """测试用信号：每 every 个交易日，按过去 lookback 日收益排序取前 top 只等权。

    只用 <= 当日 的收盘价（收盘后算，次日开盘执行），截断数据不影响历史信号 —— 这正是
    截断测试成立的前提。信号生成写在测试里，不进 src（策略是 S3 的事）。
    """
    px = panel.close_hfq
    n_d = px.shape[0]
    sig: dict[str, dict[str, float]] = {}
    for i in range(lookback, n_d, every):
        base = px[i - lookback]
        cur = px[i]
        ok = ~np.isnan(base) & ~np.isnan(cur) & (base > 0)
        if ok.sum() < top:
            continue
        ret = np.where(ok, cur / np.where(ok, base, 1.0) - 1.0, -np.inf)
        order = np.argsort(-ret, kind="stable")[:top]
        order = order[np.isfinite(ret[order])]
        if len(order) == 0:
            continue
        w = 1.0 / len(order)
        sig[panel.dates[i]] = {panel.codes[j]: w for j in order}
    return sig


# ---------------------------------------------------------------- 1. 无未来函数
@pytest.fixture(scope="module")
def real_panel():
    """真实数据上的固定小宇宙（60 只），保证截断测试跑得动且可复现。"""
    codes = [f"sh.6000{i:02d}" for i in range(0, 60)]
    full = load_panel("2018-01-02", "2026-11-06", codes=codes)
    return full


def test_no_lookahead_truncation(real_panel: Panel):
    """框架 7：截断到 2020-12-31，2020 年之前的逐日净值与持仓必须与全样本完全一致。

    这是整条管线唯一能证明"没有偷看未来"的测试：
    引擎若在任一处用了 t 日之后的数据（收盘价成交、未来信号、全样本归一化），
    截断后历史段立刻分叉。
    """
    cfg = Config(start="2018-01-02", end="2026-11-06", init_cash=1_000_000,
                 max_positions=10, max_weight=0.15)
    full_sig = momentum_signals(real_panel)
    full = run_backtest(real_panel, full_sig, cfg)

    cut = "2020-12-31"
    cut_idx = [i for i, d in enumerate(real_panel.dates) if d <= cut]
    trunc = Panel(dates=real_panel.dates[: cut_idx[-1] + 1],
                  codes=real_panel.codes,
                  open_hfq=real_panel.open_hfq[: cut_idx[-1] + 1],
                  close_hfq=real_panel.close_hfq[: cut_idx[-1] + 1],
                  open_raw=real_panel.open_raw[: cut_idx[-1] + 1],
                  blocked_buy=real_panel.blocked_buy[: cut_idx[-1] + 1],
                  blocked_sell=real_panel.blocked_sell[: cut_idx[-1] + 1],
                  suspended=real_panel.suspended[: cut_idx[-1] + 1],
                  source=real_panel.source)
    trunc_sig = {d: w for d, w in full_sig.items() if d <= cut}
    cfg2 = Config(start="2018-01-02", end=cut, init_cash=cfg.init_cash,
                  max_positions=cfg.max_positions, max_weight=cfg.max_weight)
    run2 = run_backtest(trunc, trunc_sig, cfg2)

    a = full.nav[full.nav["date"] <= cut].reset_index(drop=True)
    b = run2.nav.reset_index(drop=True)
    assert len(a) == len(b) > 200, "截断段长度异常"
    diff = (a["equity"] - b["equity"]).abs()
    assert diff.max() == 0.0, (
        f"截断后历史净值不一致（最大偏差 {diff.max():.6f}），说明引擎用了未来数据。"
        f"首个分叉日 {a.loc[diff.values.argmax(), 'date']}")

    pa = full.positions[full.positions["date"] <= cut]
    pb = run2.positions
    key = ["date", "code"]
    ma = pa.set_index(key)["shares"].sort_index()
    mb = pb.set_index(key)["shares"].sort_index()
    assert ma.index.equals(mb.index), "截断后持仓清单不一致（持有标的/天数不同）"
    assert (ma - mb).abs().max() == 0.0, "截断后逐日持仓股数不一致"


def test_same_params_twice_identical(real_panel: Panel):
    """框架 7 可复算：同一数据同一参数跑两次，净值必须逐位相同。"""
    cfg = Config(start="2018-01-02", end="2026-11-06", max_positions=10, max_weight=0.15)
    sig = momentum_signals(real_panel)
    r1 = run_backtest(real_panel, sig, cfg)
    r2 = run_backtest(real_panel, sig, cfg)
    assert np.array_equal(r1.nav["equity"].to_numpy(), r2.nav["equity"].to_numpy())
    assert len(r1.trades) == len(r2.trades)


# ---------------------------------------------------------------- 2. 涨跌停 / 停牌
def test_limit_up_blocks_buy():
    """一字涨停买不进：T 日收盘出信号，T+1 开盘即涨停价，买单必须作废。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    codes = ["sh.600000"]
    open_ = [[10.0], [11.0], [11.0]]     # 第二日开盘即涨停（10.0 → 11.0）
    close = [[10.0], [11.0], [11.0]]
    blocked_buy = [[False], [True], [False]]
    p = make_panel(dates, codes, open_, close, blocked_buy=blocked_buy)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)
    assert r.trades.empty, f"涨停日不该成交，却成交 {len(r.trades)} 笔"
    assert (r.nav["cash"] == 100_000).all(), "未成交时现金应保持初始值"


def test_limit_down_blocks_sell():
    """跌停卖不出：持仓遇到开盘跌停，卖单必须作废，仓位留在账上。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["sh.600000"]
    open_ = [[10.0], [10.0], [9.0], [8.1]]
    close = [[10.0], [10.0], [9.0], [8.1]]
    blocked_sell = [[False], [False], [True], [False]]
    p = make_panel(dates, codes, open_, close, blocked_sell=blocked_sell)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    # 第一日收盘建仓信号 → 第二日买入；第二日收盘清仓信号 → 第三日开盘跌停卖不出
    sig = {dates[0]: {"sh.600000": 1.0}, dates[1]: {"sh.600000": 0.0}}
    r = run_backtest(p, sig, cfg)
    sells = r.trades[r.trades["side"] == "sell"]
    assert sells.empty, "跌停日不该卖出成功"
    held = r.positions[r.positions["date"] == dates[2]]
    assert len(held) == 1 and held["shares"].iloc[0] > 0, "跌停卖不出，仓位应仍在"


def test_suspended_day_skips_trade():
    """停牌日不产生成交（3.3-3），且停牌期间净值按最后有效收盘价结转不变。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    codes = ["sh.600000"]
    open_ = [[10.0], [np.nan], [12.0]]
    close = [[10.0], [np.nan], [12.0]]
    susp = [[False], [True], [False]]
    p = make_panel(dates, codes, open_, close, suspended=susp)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    # D1 出信号 → 本该 D2 开盘买，但 D2 停牌 → 当日无成交，订单计入 dropped
    # 引擎语义：遇停牌/涨跌停未成交的订单不保留（不追封板标的），策略要在下个信号日重申目标
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)
    assert r.trades.empty, f"停牌日不该成交，实际成交 {len(r.trades)} 笔"
    assert len(r.dropped) == 1 and r.dropped.iloc[0]["reason"] == "suspended", \
        "未成交订单必须逐笔记入 dropped，不许静默跳过"
    nav = r.nav.set_index("date")["equity"]
    assert nav.loc[dates[1]] == pytest.approx(nav.loc[dates[0]]), "停牌日净值应结转不变"


def test_t_plus_1_no_same_day_sell():
    """T+1：同日买入的股票当天不能卖（本引擎物理上不可能，此处锁死契约）。"""
    dates = ["2020-01-02", "2020-01-03"]
    p = make_panel(dates, ["sh.600000"], [[10.0], [10.0]], [[10.0], [10.0]])
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    # 用同日信号反复调仓：若允许 T+0，第二日会出现买卖对撞
    sig = {dates[0]: {"sh.600000": 1.0}, dates[1]: {"sh.600000": 0.0}}
    r = run_backtest(p, sig, cfg)
    for _, t in r.trades.iterrows():
        if t["side"] == "sell":
            assert t["date"] > dates[0], "T+1 被破坏：首日买入当日即卖出"


# ---------------------------------------------------------------- 3. 成本口径
def test_costs_match_spec():
    """5.1 成本：佣金 0.025%（最低 5 元）、印花税 0.05% 仅卖出、过户费 0.001% 双边。"""
    c = Costs()
    # 小额：佣金触发最低 5 元。买入 10000 元 → 佣金 max(2.5,5)=5 + 过户费 0.1
    assert c.fees(10_000, "buy") == pytest.approx(5.0 + 0.1)
    # 卖出 10000 元 → 5 + 过户费 0.1 + 印花税 5 = 10.1
    assert c.fees(10_000, "sell") == pytest.approx(5.0 + 0.1 + 5.0)
    # 大额：佣金按比例。买入 100 万 → 250 + 10
    assert c.fees(1_000_000, "buy") == pytest.approx(250.0 + 10.0)
    assert c.fees(1_000_000, "sell") == pytest.approx(250.0 + 10.0 + 500.0)


# ---------------------------------------------------------------- 4. 手工对账
def test_hand_calculated_nav_3stocks():
    """3 只股票 6 个交易日，逐笔手算净值，与引擎输出误差 < 0.01。

    手算过程（与回测报告里的对账表同一套数，价格全程不变，只让 A 在第 4 日被卖出）：
      D1 收盘出信号（A 30%、B 20%），D2 开盘执行：
        D2 开盘 A=10.00、B=20.00，滑点 0.1% → A 买价 10.01、B 买价 20.02
        目标市值 A 30%×100000=30000 → 30000/10.01=2997.0 → 取整 2900 股 → 29029.00
              B 20%×100000=20000 → 20000/20.02= 999.0 → 取整  900 股 → 18018.00
        佣金 A=max(7.2573,5)+0.2903=7.5475；B=max(4.5045,5)+0.1802=5.1802（B 触发最低佣金）
        现金 = 100000 − 29029.00 − 7.5475 − 18018.00 − 5.1802 = 52940.2723
      D3 收盘出信号清仓 A（B 维持 20%），D4 开盘 A=10.00 → 卖出价 9.99，2900 股 = 28971.00
        费用 = 佣金 7.2428 + 过户费 0.2897 + 印花税 14.4855 = 22.0180 → 回款 28948.98
        现金 = 52940.2723 + 28948.9820 = 81889.2543
      D4 收盘净值 = 81889.2543 + 900×20.5 = 100339.2543
    """
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
             "2020-01-08", "2020-01-09"]
    codes = ["sh.600000", "sh.600001", "sh.600002"]
    # A、B 有信号；C 全程无信号（用于验证"没信号不交易"）
    open_ = np.array([[10.0, 20.0, 30.0]] * 6, dtype=float)
    close = np.array([[10.5, 20.5, 30.5]] * 6, dtype=float)
    p = make_panel(dates, codes, open_, close)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=3, max_weight=1.0)
    sig = {dates[0]: {"sh.600000": 0.30, "sh.600001": 0.20},
           dates[2]: {"sh.600000": 0.0, "sh.600001": 0.20}}
    r = run_backtest(p, sig, cfg)

    # ---- 手算 D2：建仓
    buy_a_px = 10.0 * 1.001
    buy_b_px = 20.0 * 1.001
    qty_a = int((0.30 * 100_000) / buy_a_px // 100) * 100
    qty_b = int((0.20 * 100_000) / buy_b_px // 100) * 100
    assert qty_a == 2900 and qty_b == 900, f"手算股数不符：{qty_a}/{qty_b}"
    amt_a, amt_b = qty_a * buy_a_px, qty_b * buy_b_px
    fee_a = max(amt_a * 0.00025, 5) + amt_a * 0.00001
    fee_b = max(amt_b * 0.00025, 5) + amt_b * 0.00001
    cash_hand = 100_000 - amt_a - fee_a - amt_b - fee_b
    eq_d2_hand = cash_hand + qty_a * 10.5 + qty_b * 20.5

    t2 = r.trades[r.trades["date"] == dates[1]]
    assert len(t2) == 2
    assert t2["shares"].tolist() == [qty_a, qty_b]
    assert t2["amount"].sum() == pytest.approx(amt_a + amt_b, abs=1e-6)
    assert t2["fees"].sum() == pytest.approx(fee_a + fee_b, abs=1e-6)

    nav = r.nav.set_index("date")["equity"]
    assert abs(nav.loc[dates[1]] - eq_d2_hand) < 0.01, (
        f"D2 净值 引擎 {nav.loc[dates[1]]:.4f} vs 手算 {eq_d2_hand:.4f}")

    # ---- 手算 D4：清仓 A
    sell_px = 10.0 * 0.999
    amt_s = qty_a * sell_px
    fee_s = max(amt_s * 0.00025, 5) + amt_s * 0.00001 + amt_s * 0.0005
    cash_after = cash_hand + amt_s - fee_s
    eq_d4_hand = cash_after + qty_b * 20.5   # D4 只清 A，B 继续持有

    t4 = r.trades[r.trades["date"] == dates[3]]
    assert len(t4) == 1 and t4.iloc[0]["side"] == "sell", (
        f"D4 应只有 1 笔卖出（B 的 20% 目标已满足，不该有加仓），实际 {len(t4)} 笔")
    assert t4.iloc[0]["shares"] == qty_a
    assert t4.iloc[0]["amount"] == pytest.approx(amt_s, abs=1e-6)
    assert t4.iloc[0]["fees"] == pytest.approx(fee_s, abs=1e-6)
    assert abs(nav.loc[dates[3]] - eq_d4_hand) < 0.01, (
        f"D4 净值 引擎 {nav.loc[dates[3]]:.4f} vs 手算 {eq_d4_hand:.4f}")
    assert eq_d4_hand == pytest.approx(100339.2543, abs=1e-3), "手算结果本身要与文档一致"

    # ---- 全程不碰无信号的 C
    assert set(r.trades["code"]) <= {"sh.600000", "sh.600001"}
    # ---- 逐日净值与手算累计误差 < 0.01
    assert abs(nav.iloc[-1] - (cash_after + qty_b * 20.5)) < 0.01


# ---------------------------------------------------------------- 5. 产物
def test_result_json_has_required_fields(tmp_path: Path):
    """result.json 缺 git_commit / 参数 / 数据区间 / 逐日净值 任一字段即为无效产物。"""
    dates = ["2020-01-02", "2020-01-03"]
    p = make_panel(dates, ["sh.600000"], [[10.0], [10.0]], [[10.0], [10.0]])
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000)
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)
    bench = M.build_benchmarks(p, r.nav["date"].tolist())
    m = M.compute_metrics(r.nav, r.trades, bench)
    out = tmp_path / "result.json"
    write_result(r, m, bench["summary"], out)
    d = json.loads(out.read_text(encoding="utf-8"))
    for key in ("git_commit", "params", "data", "nav", "metrics", "benchmarks"):
        assert key in d and d[key] not in (None, "", [], {}), f"result.json 缺 {key}"
    assert len(d["nav"]) == 2 and {"date", "equity"} <= set(d["nav"][0])
    assert d["data"]["start"] == dates[0] and d["data"]["end"] == dates[-1]
    assert d["params"]["max_positions"] == cfg.max_positions


# ---------------------------------------------------------------- 6. 指标口径
def test_metrics_on_known_series():
    """指标公式对着手算值验一遍，别让"能出数"冒充"算得对"。"""
    # 两年 504 个交易日，净值从 1 线性涨到 1.2（每日恒定收益）
    n = 504
    idx = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    nav = pd.DataFrame({"date": idx, "equity": 1.2 ** (np.arange(n) / (n - 1)),
                        "cash": 0.0, "positions": 1})
    s = M.to_series(nav)
    # 504 个交易日 = 2 年，累计 1.2 倍 → 年化 = 1.2^(1/2) − 1 = 9.54%
    assert M.ann_return(s) == pytest.approx(1.2 ** 0.5 - 1, abs=1e-4)
    assert M.ann_vol(s) == pytest.approx(0.0, abs=1e-9)         # 恒定收益 → 零波动
    assert M.max_drawdown(s)["mdd"] == pytest.approx(0.0, abs=1e-12)

    # 回撤：1 → 1.5 → 0.9 → 1.2，最大回撤 = 0.9/1.5-1 = -40%
    s2 = pd.Series([1.0, 1.5, 0.9, 1.2], index=["2020-01-02", "2020-01-03",
                                                "2020-01-06", "2020-01-07"])
    assert M.max_drawdown(s2)["mdd"] == pytest.approx(-0.4)
    assert M.max_drawdown(s2)["peak"] == "2020-01-03"
    assert M.max_drawdown(s2)["trough"] == "2020-01-06"


def test_metrics_marks_missing_hs300(tmp_path: Path):
    """沪深300 数据没落地时必须标 missing，不许拿别的序列顶替。"""
    s, status = M.hs300_total_return(["2020-01-02"], raw_dir=tmp_path / "nope")
    assert s is None and status.startswith("missing")
