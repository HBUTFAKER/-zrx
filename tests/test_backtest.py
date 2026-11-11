"""S2 回测引擎验收测试。

覆盖框架第 7 节"验收标准"的硬指标，外加成交约束、成本口径、记账口径与指标护栏：
  1. 无未来函数：截断到 2020-12-31 后**在截断面板上重新生成信号**，2020 年之前（含）
     逐日净值与持仓必须与全样本完全一致；另有一条**哨兵测试**用故意偷看未来的信号，
     证明这套检测真能抓到分叉（审计 B-01）。
  2. 可复算：同一数据同一参数跑两次，净值逐日完全相等（浮点逐位相同）。
  3. 涨跌停：一字涨停买不进、跌停卖不出；上市首日 / 注册制前 5 日**不判板**
     （审计 B-02；003R2 补 : 主板 2023-02-17 后前 5 日、ipoDate 早于日历起点的存量股）。
  4. 停牌：停牌日不产生成交。
  5. T+1：当日买入当日不可卖。
  6. 记账：股数/金额/费用按**原始价**记账，hfq 只用于估值；权重不失真（审计 B-03）；
     **退出按 hfq 口径变现，分红层不蒸发**（审计 B2-01，含除息后卖出与真实面板对账）。
  7. 清算：区间内数据终止的持仓按最后有效收盘价强制清算（审计 B-04）。
  8. 产物：result.json 必含 git_commit / 全部参数 / 真实数据区间 / 逐日净值 / 逐日持仓；
     hfq 缺口按原始价统一降级并计数（审计 B2-05）。
  9. 指标护栏：年化/波动/夏普/卡玛/回撤/胜率/盈亏比/换手/超额/基准对齐各自对解析解或构造值验。

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
               suspended=None, close_raw=None, open_raw=None) -> Panel:
    """人工面板：默认原始价 == 复权价（无分红送转），需要时用 *_raw 单独指定。"""
    o = np.asarray(open_, dtype="float64")
    c = np.asarray(close, dtype="float64")
    shape = o.shape
    z = np.zeros(shape, dtype=bool)
    return Panel(dates=list(dates), codes=list(codes), open_hfq=o, close_hfq=c,
                 open_raw=o.copy() if open_raw is None else np.asarray(open_raw, dtype="float64"),
                 close_raw=c.copy() if close_raw is None else np.asarray(close_raw, dtype="float64"),
                 blocked_buy=z.copy() if blocked_buy is None else np.asarray(blocked_buy, dtype=bool),
                 blocked_sell=z.copy() if blocked_sell is None else np.asarray(blocked_sell, dtype=bool),
                 suspended=z.copy() if suspended is None else np.asarray(suspended, dtype=bool))


def truncate_panel(panel: Panel, cut: str) -> Panel:
    """把面板截断到 cut（含）之前。所有价格矩阵与标记矩阵同步截断。"""
    n = max(i for i, d in enumerate(panel.dates) if d <= cut) + 1
    return Panel(dates=panel.dates[:n], codes=panel.codes,
                 open_hfq=panel.open_hfq[:n], close_hfq=panel.close_hfq[:n],
                 open_raw=panel.open_raw[:n], close_raw=panel.close_raw[:n],
                 blocked_buy=panel.blocked_buy[:n], blocked_sell=panel.blocked_sell[:n],
                 suspended=panel.suspended[:n], source=panel.source,
                 hfq_gap=panel.hfq_gap[:n])


def momentum_signals(panel: Panel, every: int = 5, lookback: int = 20,
                     top: int = 10) -> dict[str, dict[str, float]]:
    """测试用正常信号：每 every 个交易日，按过去 lookback 日收益排序取前 top 只等权。

    只读 <= 当日 的收盘价（收盘后算，次日开盘执行），因此**在截断面板上重算**
    得到的历史信号与全样本完全相同 —— 这正是截断测试成立的前提。
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


def leaky_signals(panel: Panel, every: int = 5, lookahead: int = 10,
                  top: int = 10) -> dict[str, dict[str, float]]:
    """**故意偷看未来**的信号（哨兵测试专用）：按"未来第 lookahead 天的涨幅"排序。

    这是最典型的未来函数：用 t+10 的价格决定 t 日收盘的调仓单。
    截断面板上"未来"消失，排序结果必然改变 —— 只要截断测试有牙，就一定能抓到分叉。
    """
    px = panel.close_hfq
    n_d = px.shape[0]
    sig: dict[str, dict[str, float]] = {}
    for i in range(0, n_d, every):
        j = min(i + lookahead, n_d - 1)          # 越界夹紧，截断后 j 变小
        cur, fut = px[i], px[j]
        ok = ~np.isnan(cur) & ~np.isnan(fut) & (cur > 0)
        if ok.sum() < top:
            continue
        ret = np.where(ok, fut / np.where(ok, cur, 1.0) - 1.0, -np.inf)
        order = np.argsort(-ret, kind="stable")[:top]
        order = order[np.isfinite(ret[order])]
        if len(order) == 0:
            continue
        w = 1.0 / len(order)
        sig[panel.dates[i]] = {panel.codes[j2]: w for j2 in order}
    return sig


def truncation_gap(panel: Panel, signal_fn, cfg: Config, cut: str) -> tuple[float, float, str]:
    """跑"全样本"与"截断面板"，返回历史段的 (净值最大偏差, 持仓最大偏差, 首个分叉日)。

    关键：截断后的信号**必须由 signal_fn 在截断面板上重新生成**，
    不许把全样本信号切一刀（那等于把答案抄过去，未来函数永远检不出来）。
    """
    full = run_backtest(panel, signal_fn(panel), cfg)
    trunc = truncate_panel(panel, cut)
    cfg2 = Config(**{**cfg.__dict__, "end": cut})
    part = run_backtest(trunc, signal_fn(trunc), cfg2)

    a = full.nav[full.nav["date"] <= cut].reset_index(drop=True)
    b = part.nav.reset_index(drop=True)
    if len(a) != len(b):
        return float("inf"), float("inf"), f"长度不一致 {len(a)} vs {len(b)}"
    diff = (a["equity"] - b["equity"]).abs()
    first = str(a.loc[int(diff.values.argmax()), "date"]) if diff.max() > 0 else ""

    pa = full.positions[full.positions["date"] <= cut].set_index(["date", "code"])["shares"]
    pb = part.positions.set_index(["date", "code"])["shares"]
    pa, pb = pa.sort_index(), pb.sort_index()
    if not pa.index.equals(pb.index):
        return float(diff.max()), float("inf"), "持仓清单不一致"
    return float(diff.max()), float((pa - pb).abs().max()), first


# ---------------------------------------------------------------- 1. 无未来函数
@pytest.fixture(scope="module")
def real_panel():
    """真实数据上的固定小宇宙（60 只），保证截断测试跑得动且可复现。"""
    codes = [f"sh.6000{i:02d}" for i in range(0, 60)]
    return load_panel("2018-01-02", "2026-11-06", codes=codes)


def test_no_lookahead_truncation(real_panel: Panel):
    """框架 7：截断到 2020-12-31，2020 年之前的逐日净值与持仓必须与全样本完全一致。

    这是整条管线唯一能证明"没有偷看未来"的测试：
    引擎若在任一处用了 t 日之后的数据（收盘价成交、未来信号、全样本归一化），
    截断后历史段立刻分叉。

    审计 B-01 修正：截断面板上**重新生成信号**（原实现直接切全样本信号，
    等于把未来函数生出来的答案一起抄过去，检测形同虚设）。
    """
    cfg = Config(start="2018-01-02", end="2026-11-06", init_cash=1_000_000,
                 max_positions=10, max_weight=0.15)
    cut = "2020-12-31"
    nav_gap, pos_gap, first = truncation_gap(real_panel, momentum_signals, cfg, cut)

    assert nav_gap == 0.0, (
        f"截断后历史净值不一致（最大偏差 {nav_gap:.6f}），说明引擎用了未来数据。"
        f"首个分叉日 {first}")
    assert pos_gap == 0.0, f"截断后逐日持仓股数不一致（最大偏差 {pos_gap:.6f}）"


def test_leaky_signal_breaks_truncation(real_panel: Panel):
    """哨兵测试（审计 B-01 要求）：故意偷看未来的信号，截断测试必须抓到分叉。

    如果这条测试也能"通过"（gap == 0），说明上面的截断测试是摆设 ——
    它抓不到未来函数，就没有存在的意义。这条测试是那套检测的"证伪实验"。
    """
    cfg = Config(start="2018-01-02", end="2026-11-06", init_cash=1_000_000,
                 max_positions=10, max_weight=0.15)
    cut = "2020-12-31"
    nav_gap, pos_gap, first = truncation_gap(real_panel, leaky_signals, cfg, cut)

    assert nav_gap > 0.0, (
        "leaky 信号（按 t+10 涨幅排序）竟未造成截断分叉 —— "
        "说明截断测试抓不到未来函数，检测无效")
    assert pos_gap > 0.0, "leaky 信号的截断分叉在持仓层面未被检出"
    print(f"\n[哨兵] leaky 信号被抓到：净值最大偏差 {nav_gap:.6f}，"
          f"持仓最大偏差 {pos_gap:.6f}，首个分叉 {first}")


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
    assert len(r.dropped) == 1 and r.dropped.iloc[0]["reason"] == "limit_up"


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


def test_new_stock_first_day_not_blocked():
    """审计 B-02：上市首日无涨跌幅限制，绝不能判涨停/跌停。

    构造首日开盘较昨收 +40%（远超 ±10%），若引擎按 ±10% 判板会误判为"不可买"。
    """
    from src.backtest import _blocked_arrays
    codes = pd.Series(["sh.600000", "sh.688001"])
    dates = pd.Series(["2020-01-02", "2020-01-02"])
    is_st = pd.Series([0, 0], dtype="int8")
    listed = pd.Series([1, 1])                    # 均为上市首日
    # 昨收 10 → 开盘 14（+40%），任何涨跌幅限制都会被判成触板
    open_ = np.array([14.0, 14.0])
    close = np.array([14.0, 14.0])
    pct = np.array([40.0, 40.0])
    blk_buy, blk_sell = _blocked_arrays(codes, is_st, dates, listed, open_, close, pct)
    assert not blk_buy.any(), f"上市首日被误判为涨停不可买：{blk_buy}"
    assert not blk_sell.any(), f"上市首日被误判为跌停不可卖：{blk_sell}"

    # 对照：同样的 +40% 放在上市第 100 个交易日，主板必须判成涨停不可买
    listed2 = pd.Series([100, 100])
    b2, _ = _blocked_arrays(codes, is_st, dates, listed2, open_, close, pct)
    assert b2[0], "主板上市 100 日后 +40% 应判为涨停不可买"
    # 科创板同一天 ±20%，+40% 同样超板 → 也要判成不可买
    assert b2[1], "科创板 +40% 应判为涨停不可买"


def test_registration_gem_first_5_days_not_blocked():
    """创业板注册制（2020-08-24 起）新股前 5 个交易日无涨跌幅限制。"""
    from src.backtest import _blocked_arrays
    codes = pd.Series(["sz.300001"] * 3)
    is_st = pd.Series([0, 0, 0], dtype="int8")
    dates = pd.Series(["2020-08-25"] * 3)          # 注册制改革之后
    listed = pd.Series([3, 6, 100])                # 第 3 日 / 第 6 日 / 第 100 日
    open_ = np.array([14.0, 14.0, 14.0])
    close = np.array([14.0, 14.0, 14.0])
    pct = np.array([40.0, 40.0, 40.0])
    blk, _ = _blocked_arrays(codes, is_st, dates, listed, open_, close, pct)
    assert not blk[0], "注册制新股第 3 日无涨跌幅限制，不该判板"
    assert blk[1], "注册制新股第 6 日恢复 ±20%，+40% 应判涨停"
    assert blk[2], "上市 100 日 +40% 应判涨停"

    # 改革前（2020-08-20）创业板仍是 ±10%，新股第 3 日应判板
    dates_old = pd.Series(["2020-08-20"] * 3)
    blk_old, _ = _blocked_arrays(codes, is_st, dates_old, listed, open_, close, pct)
    assert blk_old[0], "改革前创业板第 3 日照常 ±10%，+40% 应判涨停"


def test_main_board_registration_first_5_days_not_blocked():
    """审计 B2-02②：主板 2023-02-17 全面注册制后，新股**前 5 个交易日**无涨跌幅限制。

    旧实现只覆盖科创板与创业板，主板上榜新股第 2~5 日仍按 ±10% 判板，
    实测 sh.601121 等 138 只 / 551 行窗口内被误判，其中 79 行因此被判"涨停买不进"。
    本例用 601121 真实形态（上市次日开盘 +9.98%）构造：修复后不判板，第 6 日恢复 ±10%。
    """
    from src.backtest import _blocked_arrays
    codes = pd.Series(["sh.601121"] * 4)
    is_st = pd.Series([0, 0, 0, 0], dtype="int8")
    dates = pd.Series(["2023-03-13", "2023-03-16", "2023-03-17", "2023-02-16"])
    listed = pd.Series([2, 5, 6, 2])               # 第2日 / 第5日 / 第6日 / 改革前第2日
    open_ = np.array([11.0, 11.0, 11.0, 11.0])     # 昨收 10 → 开盘 11（+10%，正好触板）
    close = np.array([11.0, 11.0, 11.0, 11.0])
    pct = np.array([10.0, 10.0, 10.0, 10.0])
    blk, _ = _blocked_arrays(codes, is_st, dates, listed, open_, close, pct)
    assert not blk[0], "主板注册制新股第 2 日无涨跌幅限制，不该判涨停"
    assert not blk[1], "主板注册制新股第 5 日无涨跌幅限制，不该判涨停"
    assert blk[2], "第 6 日恢复 ±10%，开盘 +10% 应判涨停"
    assert blk[3], "2023-02-17 之前主板新股第 2 日照常 ±10%，应判涨停"


def test_main_board_ipo_real_data_601121():
    """真实数据锁：sh.601121（宝地矿业，2023-03-10 主板注册制首日）上市第 2~5 日不判板。

    合成用例能证明规则写对了，但证明不了它真的接到了数据上。这里直接载入真实面板，
    逐日核对：03-13/14/15/16（第 2~5 日，开盘 +9.98%~+10.01% 贴着 10% 线）不判涨停，
    03-17（第 6 日）恢复 ±10% 后判涨停 —— 这正是审计 B2-02② 举的那 86 行里的样本。
    """
    p = load_panel("2023-03-01", "2023-03-31", codes=["sh.601121"])
    j = 0
    blk = {d: bool(p.blocked_buy[i, j]) for i, d in enumerate(p.dates)}
    for d in ("2023-03-13", "2023-03-14", "2023-03-15", "2023-03-16"):
        assert not blk[d], f"{d}（注册制新股前 5 日）被误判为涨停不可买"
    assert blk["2023-03-17"], "第 6 日应恢复 ±10% 并判涨停，否则规则被放得太松"


def test_ipo_before_calendar_not_new_stock():
    """审计 B2-02①：ipoDate 早于交易日历起点的存量股，不得因为 listed_days 被截断成 1
    而判成"上市首日无涨跌幅"。

    成因：clean.py 用 searchsorted(全市场日历, ipoDate) 算 listed_days，ipoDate 早于
    日历起点时返回 0，于是 2018-01-02 全市场 3,470 只存量股都变成 listed_days=1。
    修复后按 ipoDate 是否落在面板日历内判定：落在起点之前 = 存量股，按普通股判板；
    真 IPO 首日（date == ipoDate）行为不变。
    """
    from src.backtest import _blocked_arrays
    codes = pd.Series(["sh.600000", "sh.600000", "sh.600000"])
    is_st = pd.Series([0, 0, 0], dtype="int8")
    dates = pd.Series(["2018-01-02"] * 3)
    listed = pd.Series([1, 1, 1])                       # 三行都是"日历首日"= 被截断的假首日
    ipo = pd.Series(["1999-11-10", "2018-01-02", ""])   # 存量股 / 真 IPO 首日 / 上市日缺失
    open_ = np.array([11.0, 11.0, 11.0])
    close = np.array([11.0, 11.0, 11.0])
    pct = np.array([10.0, 10.0, 10.0])
    blk, _ = _blocked_arrays(codes, is_st, dates, listed, open_, close, pct, ipo)
    assert blk[0], "存量股（ipo 1999-11-10）被日历截断成 listed_days=1，仍被判为无涨跌幅"
    assert not blk[1], "真 IPO 首日（date == ipoDate）必须保持无涨跌幅，不判板"
    # ipoDate 缺失 → 无从判断，按普通股保守判板（生产数据无此情形：5,565 只全部有 ipoDate）
    assert blk[2], "ipoDate 缺失时应保守按普通股判板，不许当成『新上市』放行"

    # 不传 ipo_dates 时行为完全不变（合成面板 / 老调用方兼容）
    blk_old, _ = _blocked_arrays(codes, is_st, dates, listed, open_, close, pct)
    assert not blk_old.any(), "不传 ipo_dates 时应保持旧的 listed_days 口径"


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
    """T+1：同日买入的股票当天不能卖。

    注意：当前"先卖后买 + last_buy_day 在买入后写入"的排序下，t_plus_1 拒单分支
    物理不可达（R10/B-08 已把这一点写进源码注释）。本测试锁的是**契约**：
    任何一笔卖出都不可能与其买入同日发生。
    """
    dates = ["2020-01-02", "2020-01-03"]
    p = make_panel(dates, ["sh.600000"], [[10.0], [10.0]], [[10.0], [10.0]])
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    sig = {dates[0]: {"sh.600000": 1.0}, dates[1]: {"sh.600000": 0.0}}
    r = run_backtest(p, sig, cfg)
    buys = r.trades[r.trades["side"] == "buy"]
    sells = r.trades[r.trades["side"].isin(("sell", "liquidate"))]
    for _, s in sells.iterrows():
        same_day_buy = buys[(buys["code"] == s["code"]) & (buys["date"] == s["date"])]
        assert same_day_buy.empty, f"T+1 被破坏：{s['code']} 在 {s['date']} 买入当日即卖出"


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


# ---------------------------------------------------------------- 4. 记账口径（R1）
def test_accounting_uses_raw_price_not_hfq():
    """审计 B-03：股数/金额必须按原始价记账，否则复权倍数把权重放得离谱。

    构造两只股票：A 原始价 10 元、后复权价 1000 元（因子 100）；B 原始价 2 元、复权价 2 元。
    目标各 50%。若按 hfq 价算股数：A 只能买 50000/1000≈49 股 → 取整 0 股，**完全买不到**，
    实际权重 0%，目标 50% —— 这就是审计实测"权重失真最高 1303 倍"的成因。
    """
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    codes = ["sh.600000", "sh.600001"]
    raw = [[10.0, 2.0]] * 3
    hfq = [[1000.0, 2.0]] * 3
    p = make_panel(dates, codes, hfq, hfq, open_raw=raw, close_raw=raw)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=2, max_weight=1.0)
    r = run_backtest(p, {dates[0]: {"sh.600000": 0.5, "sh.600001": 0.5}}, cfg)

    tr = r.trades.set_index("code")
    # A 必须真的买到（股数 > 0，且是 100 的整数倍）
    qa = tr.loc["sh.600000", "shares"]
    assert qa >= 100, f"按原始价应能买到 4900 股，实际 {qa} 股（说明还在用 hfq 价算股数）"
    assert qa % 100 == 0
    # 成交价必须是原始价 ± 滑点，不是复权价
    assert tr.loc["sh.600000", "price"] == pytest.approx(10.0 * 1.001)
    assert tr.loc["sh.600001", "price"] == pytest.approx(2.0 * 1.001)

    # 实际持仓市值比例应接近 1:1（目标各 50%）
    pos = r.positions[r.positions["date"] == dates[1]].set_index("code")["value"]
    ratio = pos["sh.600000"] / pos["sh.600001"]
    assert 0.9 < ratio < 1.1, f"两只股票市值比例 {ratio:.3f} 偏离 1:1，权重失真"


def test_dividend_reinvested_through_hfq():
    """3.3-2：分红再投通过 hfq 收益率体现 —— 除息日原始价下跌但净值不跌；**退出时按 hfq 变现**。

    审计 B2-01（003R2 阻断项）：旧实现只在持有段体现分红层，一卖出就按原始价回款、
    nhq 清零，累计分红层在退出瞬间蒸发。本测试同时锁住两段：
      持有时：除息日净值不变（分红再投进市值）；
      卖出时：变现额 = nhq × hfq × (1−滑点)，比"原始股数 × 原始价"多出**整笔分红层**。
    """
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["sh.600000"]
    # 第 3 天除息：原始价 10 → 9.5（每股派 0.5），后复权价恒为 1000（分红再投后总价值不变）
    raw = [[10.0], [10.0], [9.5], [9.5]]
    hfq = [[1000.0]] * 4
    p = make_panel(dates, codes, hfq, hfq, open_raw=raw, close_raw=raw)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    # D1 建仓 → D2 开盘买入；D3 收盘清仓信号 → D4 开盘卖出（除息之后）
    sig = {dates[0]: {"sh.600000": 1.0}, dates[2]: {"sh.600000": 0.0}}
    r = run_backtest(p, sig, cfg)
    nav = r.nav.set_index("date")["equity"]
    # 除息前一日与除息日：净值应只差费用，不因除权掉 5%
    assert nav.loc[dates[2]] == pytest.approx(nav.loc[dates[1]], abs=1e-6), (
        f"除息日净值从 {nav.loc[dates[1]]:.4f} 掉到 {nav.loc[dates[2]]:.4f}，分红未被再投")
    # 持仓股数（真实）不变，但原始价已除息 —— 证明估值走的是 hfq 口径而非原始价
    pos = r.positions[r.positions["date"] == dates[2]].iloc[0]
    assert pos["price_raw"] == pytest.approx(9.5)
    assert pos["value"] == pytest.approx(nav.loc[dates[2]] - r.nav.set_index("date").loc[dates[2], "cash"])

    # ---------------- 退出段（B2-01 修复后新增） ----------------
    buy = r.trades[r.trades["side"] == "buy"].iloc[0]
    sell = r.trades[r.trades["side"] == "sell"]
    assert len(sell) == 1, f"应恰好 1 笔卖出，实际 {len(sell)} 笔"
    s = sell.iloc[0]
    qty = buy["shares"]
    slip = cfg.costs.slippage

    # 记账：账面价值 = 除息日该股持仓市值（含 0.5/股 的累计分红层）
    assert s["book"] == pytest.approx(pos["value"], abs=1e-6), (
        f"卖出账面价值 {s['book']:.4f} 与除息日持仓市值 {pos['value']:.4f} 不一致")
    # 变现额 = 账面价值 × (1−滑点)：分红层随退出一起回收
    assert s["proceeds"] == pytest.approx(s["book"] * (1.0 - slip), abs=1e-9)

    # 关键对比：旧口径（原始股数 × 原始价回款）会少掉整笔分红层
    raw_cash = qty * 9.5 * (1.0 - slip)
    div_layer = s["proceeds"] - raw_cash
    assert div_layer == pytest.approx(qty * 0.5 * (1.0 - slip), abs=1e-6), (
        f"退出时分红层 {div_layer:.4f} != 每股 0.5 × {qty} 股 × (1−滑点)"
        f"，说明分红层在退出时蒸发了（B2-01 复现）")
    assert div_layer > 4000, f"分红层金额 {div_layer:.2f} 明显偏小，口径可能退回了原始价"

    # 权益守恒：卖出当日只损失 滑点 + 费用
    eq_drop = nav.loc[dates[2]] - nav.loc[dates[3]]
    assert eq_drop == pytest.approx(s["book"] * slip + s["fees"], abs=1e-6), (
        f"卖出日权益下降 {eq_drop:.6f} != 滑点 {s['book'] * slip:.6f} + 费用 {s['fees']:.6f}")


def test_cash_is_real_money():
    """现金必须是真实现金流：买入扣真实金额+费用，卖出回款真实金额-费用。"""
    dates = ["2020-01-02", "2020-01-03"]
    codes = ["sh.600000"]
    raw = [[10.0], [10.0]]
    hfq = [[1000.0], [1000.0]]
    p = make_panel(dates, codes, hfq, hfq, open_raw=raw, close_raw=raw)
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)
    t = r.trades.iloc[0]
    assert t["amount"] == pytest.approx(t["shares"] * t["price"])
    cash_after = r.nav.set_index("date").loc[dates[1], "cash"]
    assert cash_after == pytest.approx(100_000 - t["amount"] - t["fees"], abs=1e-9)
    # 而且是量级合理的小数（不是 hfq 名目下的天文数字）
    assert 0 < t["amount"] <= 100_000


# ---------------------------------------------------------------- 4b. hfq 缺口（B2-05 / B3-01）
def test_hfq_gap_marked_not_filled():
    """审计 B2-05 + B3-01：hfq 缺口必须**只标注、不插补**，且缺口行打上不可交易标记。

    003R2 曾按"hfq := 原始价（因子 1）"填平缺口，结果为 sz.001914 造出 6.41 倍幻影层。
    003R3 起缺口保持 NaN，靠 hfq_gap 标记禁买/冻结/末端清算。
    """
    from src.backtest import _mark_hfq_gap
    df = pd.DataFrame({
        "code": ["sh.600000", "sh.600000", "sh.600001"],
        "date": ["2020-01-02", "2020-01-03", "2020-01-02"],
        "open": [10.0, 10.5, 20.0],
        "close": [10.2, 10.6, 20.2],
        "hfq_open": [1000.0, np.nan, 20.0],
        "hfq_close": [1000.0, np.nan, 20.0],
    })
    out, st = _mark_hfq_gap(df)
    assert st["rows"] == 1 and st["codes"] == ["sh.600000"], f"缺口统计不对：{st}"
    assert st["open_rows"] == 1
    # 不插补：缺口行仍是 NaN，不许被填成原始价
    assert np.isnan(out.loc[1, "hfq_close"]), "缺口被填充了 —— B3-01 幻影层会回来"
    assert np.isnan(out.loc[1, "hfq_open"]), "缺口被填充了"
    assert len(out) == len(df), "缺口行不许被丢掉"
    # 标记：缺口行 True，正常行 False
    assert bool(out.loc[1, "hfq_gap"]) and not bool(out.loc[0, "hfq_gap"])
    assert not bool(out.loc[2, "hfq_gap"])
    # 无缺口的行必须原样保留
    assert out.loc[0, "hfq_close"] == pytest.approx(1000.0)
    assert "不插补" in st["policy"]


def test_hfq_gap_real_data_count_and_result_field():
    """真实数据锁：缺口 1,942 行 / 2 只（sz.001914、sz.302132），且计数必须落进 result.data。

    数量是审计复算过的口径；一旦 clean 层换了复权因子来源，这里先红，不至于让回测
    在无声降级中跑完。缺口在面板里必须仍是 NaN + 不可交易标记（B3-01）。
    """
    codes = ["sz.001914", "sz.302132"]
    p = load_panel("2018-01-02", "2026-11-06", codes=codes)
    g = p.source["hfq_gap_rows"]
    assert g["rows"] == 1942, f"缺口行数 {g['rows']} 与审计口径 1,942 不符"
    assert g["codes"] == sorted(codes), f"缺口股票清单不对：{g['codes']}"
    # 缺口行保持 NaN（不插补），且被打上 hfq_gap 标记
    gap_cells = p.hfq_gap
    assert int(gap_cells.sum()) == 1942, f"缺口标记 {int(gap_cells.sum())} 行，应为 1,942"
    assert bool(np.isnan(p.close_hfq[gap_cells]).all()), "缺口格 hfq 被填了值"
    # 反向：有原始价但既无 hfq 又没被标记的行必须为 0（否则是漏标）
    miss = (~np.isnan(p.close_raw)) & np.isnan(p.close_hfq) & ~gap_cells
    assert not miss.any(), f"有 {int(miss.sum())} 行 hfq 缺失却没有缺口标记"

    # 计数必须随产物落地（result.json 的 data.hfq_gap_rows），并附台账
    cfg = Config(start="2018-01-02", end="2026-11-06", init_cash=100_000)
    r = run_backtest(p, {}, cfg)
    assert r.data["hfq_gap_rows"]["rows"] == 1942
    assert r.data["hfq_gap_rows"]["codes"] == sorted(codes)
    led = r.data["hfq_gap_ledger"]
    assert set(led) == {"buy_blocked", "frozen_position_days",
                        "liquidated_hfq_gap", "liquidated_hfq_gap_value"}


def test_hfq_gap_no_buy_and_no_phantom_layer_real_001914():
    """审计 B3-01 (a)(b) 核心：sz.001914 的 2018 缺口期不得开仓，故不可能有幻影层。

    真实数据：2018 全年 hfq 缺失（243 行），2019-01-02 起 hfq = 原始价 × 6.4111。
    修复前缺口按因子 1 填充，缺口期买入的 nhq 在 2019-01-02 被真实因子放大 6.41 倍，
    市值凭空暴涨，退出时变成真金白银。
    修复后：缺口期买入被 hfq_gap 拦下（dropped 记录），持仓为零，净值不跳；
    缺口外（2019-01-02）正常买入时 nhq = 金额 / (hfq_open×(1+滑点))，book 与真实因子一致。
    """
    p = load_panel("2018-01-02", "2019-06-28", codes=["sz.001914"])
    j = 0
    n_gap = int(p.hfq_gap[:, j].sum())
    assert n_gap == 243, f"2018 缺口行数应 243，实际 {n_gap}"

    # 2018-06-01 信号建仓 → 次日（缺口内）必须拒单
    cfg = Config(start=p.dates[0], end=p.dates[-1], init_cash=1_000_000,
                 max_positions=1, max_weight=1.0)
    r = run_backtest(p, {"2018-06-01": {"sz.001914": 1.0}}, cfg)
    assert r.trades.empty, f"缺口期不应成交，却成交 {len(r.trades)} 笔：\n{r.trades}"
    assert not r.positions.shape[0], "缺口期不应有持仓"
    buy_drop = r.dropped[r.dropped["side"] == "buy"]
    assert len(buy_drop) == 1 and buy_drop.iloc[0]["reason"] == "hfq_gap"
    assert r.data["hfq_gap_ledger"]["buy_blocked"] >= 1
    assert r.nav["equity"].nunique() == 1, "未建仓时净值必须恒定，不许因缺口估值跳变"

    # 缺口外（2019-01-02）建仓 → 按真实因子 6.41 折算，book 不含幻影层
    d0 = "2019-01-03"
    buy_at = p.dates[p.dates.index(d0) + 1]
    r2 = run_backtest(p, {d0: {"sz.001914": 1.0}}, cfg)
    buy = r2.trades[r2.trades["side"] == "buy"].iloc[0]
    assert buy["date"] == buy_at
    i = p.dates.index(buy_at)
    raw_px = float(p.open_raw[i, j])
    hfq_px = float(p.open_hfq[i, j])
    assert hfq_px / raw_px == pytest.approx(6.4111, rel=1e-3), "缺口后因子应为 6.4111"
    # 买入当日市值 = 成交金额/(1+滑点) × (hfq_close/hfq_open)
    #   —— 只含当日涨跌，不含因子倍数；幻影层会让它凭空多出 6.41 倍
    nhq_expect = buy["amount"] / (hfq_px * 1.001)
    pos = r2.positions[r2.positions["date"] == buy_at].iloc[0]
    value_expect = nhq_expect * float(p.close_hfq[i, j])
    assert pos["value"] == pytest.approx(value_expect, rel=1e-9), (
        f"买入当日市值 {pos['value']:.2f} 应等于 nhq×hfq_close={value_expect:.2f}")
    ratio = pos["value"] / (buy["amount"] / 1.001)
    assert 0.8 < ratio < 1.2, (
        f"买入当日市值/成交金额 = {ratio:.4f}，超出当日涨跌范围 —— 幻影层又回来了")

    # 每个交易日权益都不得出现"单日 >20% 跳变"（幻影层的特征）
    nav = r2.nav.set_index("date")["equity"]
    ret = nav.pct_change().dropna().abs()
    assert ret.max() < 0.20, f"净值单日跳变 {ret.max():.2%}，疑似缺口估值断层"

    # 缺口外退出：book 必须与「真实股数 × 真实因子换算」对拍（误差 ≤ 1e-9 相对）
    d1 = "2019-05-30"
    sell_at = p.dates[p.dates.index(d1) + 1]
    r3 = run_backtest(p, {d0: {"sz.001914": 1.0}, d1: {"sz.001914": 0.0}}, cfg)
    sell = r3.trades[r3.trades["side"] == "sell"].iloc[0]
    assert sell["date"] == sell_at
    k = p.dates.index(sell_at)
    hfq_sell = float(p.open_hfq[k, j])
    factor_sell = hfq_sell / float(p.open_raw[k, j])
    assert factor_sell == pytest.approx(6.4111, rel=1e-2)
    # book = 真实股数 × 买入日原始价 × (卖出日 hfq / 买入日 hfq)，即分红再投层计入后的市值
    book_expect = buy["shares"] * raw_px * (hfq_sell / hfq_px)
    assert sell["book"] == pytest.approx(book_expect, rel=1e-9), (
        f"退出 book {sell['book']:.2f} 与『真实股数×真实因子换算』{book_expect:.2f} 不符")


def test_hfq_gap_freezes_held_position():
    """审计 B3-01 (2)：已持仓股进入 hfq 缺口段 → 禁卖、估值冻结在缺口前最后有效 hfq 价。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    codes = ["sh.600000"]
    raw = [[10.0], [10.2], [10.4], [10.6], [10.8]]
    hfq = [[1000.0], [1020.0], [np.nan], [np.nan], [1080.0]]   # 第 3~4 日缺口
    gap = [[False], [False], [True], [True], [False]]
    p = Panel(dates=dates, codes=codes,
              open_hfq=np.array(hfq), close_hfq=np.array(hfq),
              open_raw=np.array(raw), close_raw=np.array(raw),
              blocked_buy=np.zeros((5, 1), bool), blocked_sell=np.zeros((5, 1), bool),
              suspended=np.zeros((5, 1), bool), hfq_gap=np.array(gap))
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    # 第 2 日起每天收盘都给清仓信号（引擎不保留未成交订单，必须逐日重发）：
    #   第 3、4 日（缺口）→ 卖不出，记 hfq_gap；第 5 日（缺口结束）→ 成交
    sig = {dates[0]: {"sh.600000": 1.0}}
    for d in dates[1:4]:
        sig[d] = {"sh.600000": 0.0}
    r = run_backtest(p, sig, cfg)

    sells = r.trades[r.trades["side"] == "sell"]
    assert len(sells) == 1, f"应有 1 笔卖出，实际 {len(sells)} 笔：\n{r.trades}"
    assert sells.iloc[0]["date"] == dates[4], "缺口期内不许卖出，须等缺口结束后才成交"
    drops = r.dropped[(r.dropped["side"] == "sell") & (r.dropped["reason"] == "hfq_gap")]
    assert len(drops) == 2, f"缺口期应记 2 笔 hfq_gap 废单，实际 {len(drops)} 笔"

    nav = r.nav.set_index("date")["equity"]
    # 缺口两天估值冻结：净值只随现金（无变化）与冻结价变化 → 与缺口前一日完全相等
    assert nav.loc[dates[2]] == pytest.approx(nav.loc[dates[1]], abs=1e-9), (
        f"缺口日净值应冻结在 {nav.loc[dates[1]]:.6f}，实际 {nav.loc[dates[2]]:.6f}")
    assert nav.loc[dates[3]] == pytest.approx(nav.loc[dates[1]], abs=1e-9)
    # 台账：持仓冻结天数 = 2
    assert r.data["hfq_gap_ledger"]["frozen_position_days"] == 2


def test_hfq_gap_to_end_liquidated_at_last_valid_hfq():
    """审计 B3-01 (2) 后半：缺口延续到数据末端 → 按最后有效 hfq 价 ×(1−滑点) 清算并落台账。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    codes = ["sh.600000"]
    raw = [[10.0], [10.2], [10.4], [10.6]]        # 原始价一直有
    hfq = [[1000.0], [1020.0], [np.nan], [np.nan]]  # hfq 从第 3 日起彻底断供
    gap = [[False], [False], [True], [True]]
    p = Panel(dates=dates, codes=codes,
              open_hfq=np.array(hfq), close_hfq=np.array(hfq),
              open_raw=np.array(raw), close_raw=np.array(raw),
              blocked_buy=np.zeros((4, 1), bool), blocked_sell=np.zeros((4, 1), bool),
              suspended=np.zeros((4, 1), bool), hfq_gap=np.array(gap))
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)

    liq = r.trades[r.trades["side"] == "liquidate"]
    assert len(liq) == 1, f"缺口延续到末端应有 1 笔清算，实际 {len(liq)} 笔：\n{r.trades}"
    row = liq.iloc[0]
    assert row["date"] == dates[1], f"清算应在最后一个有效 hfq 日 {dates[1]}，实际 {row['date']}"
    assert row["reason"] == "hfq_gap"
    # 清算价按**最后有效 hfq** 折算（原始价 10.2 × 0.999），不是缺口期原始价
    assert row["price"] == pytest.approx(10.2 * 0.999)
    # book = nhq × hfq_last，等价于「真实股数 × 买入当日原始价」（因子被 hfq 口径抵消），
    # 即买入当日的真实市值 —— 不含任何幻影层
    assert row["book"] == pytest.approx(row["shares"] * 10.2, rel=1e-9), (
        f"清算 book 应按 hfq 价计，实际 {row['book']:.2f}")
    led = r.data["hfq_gap_ledger"]
    assert led["liquidated_hfq_gap"] == 1
    assert led["liquidated_hfq_gap_value"] == pytest.approx(row["book"])
    last = r.nav.iloc[-1]
    assert last["equity"] == pytest.approx(last["cash"]), "清算后权益必须全部是现金"


def test_load_ipo_dates_missing_db_raises(tmp_path):
    """审计 B3-01 4(c)：stocks.ipoDate 读不到必须**显式停工**，不许静默退回 listed_days。"""
    from src.backtest import _load_ipo_dates
    with pytest.raises(RuntimeError, match="stocks.ipoDate"):
        _load_ipo_dates(["sh.600000"], db=tmp_path / "missing.db")


def test_load_ipo_dates_missing_table_raises(tmp_path):
    """同上：库在但表不在（或读失败）也必须停工。"""
    from src.backtest import _load_ipo_dates
    db = tmp_path / "empty.db"
    db.write_bytes(b"")
    with pytest.raises(RuntimeError, match="stocks.ipoDate"):
        _load_ipo_dates(["sh.600000"], db=db)


# ---------------------------------------------------------------- 5. 退市清算（R4）
def test_delisted_position_force_liquidated():
    """审计 B-04：数据在区间内终止的持仓，必须在最后有效收盘价强制清算。

    原实现持仓按最后价格"结转"到回测结束，永远卖不掉，净值里挂着一笔不可变现的资产；
    同时每天产生一笔永久 suspended 的卖单废单。
    """
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
             "2020-01-08", "2020-01-09"]
    codes = ["sh.600000"]
    o = [[10.0], [10.5], [11.0], [11.5], [np.nan], [np.nan]]
    c = [[10.2], [10.7], [11.2], [11.7], [np.nan], [np.nan]]
    susp = [[False], [False], [False], [False], [True], [True]]
    p = make_panel(dates, codes, o, c, suspended=susp,
                   close_raw=[[10.2], [10.7], [11.2], [11.7], [np.nan], [np.nan]])
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000,
                 max_positions=1, max_weight=1.0)
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)

    liq = r.trades[r.trades["side"] == "liquidate"]
    assert len(liq) == 1, f"应有 1 笔清算，实际 {len(liq)} 笔：\n{r.trades}"
    row = liq.iloc[0]
    assert row["date"] == dates[3], f"清算应发生在最后有效日 {dates[3]}，实际 {row['date']}"
    assert row["price"] == pytest.approx(11.7 * 0.999), "清算价应为最后收盘价 × (1−滑点)"

    # 清算后不再有持仓行，现金=全部资产
    after = r.positions[r.positions["date"] > dates[3]]
    assert after.empty, f"清算后仍有 {len(after)} 行持仓，仓位没清干净"
    last = r.nav.iloc[-1]
    assert last["equity"] == pytest.approx(last["cash"]), "清算后权益应等于现金"

    # 不再有永久 suspended 的废单（原实现每天记一条）
    assert not (r.dropped["reason"] == "suspended").any(), \
        f"清算后仍产生 suspended 废单：{r.dropped.to_dict('records')}"


def test_exit_dividend_layer_no_evaporation_real(real_panel: Panel):
    """审计 B2-01 的真实数据锁：全样本回测里每笔退出都按 hfq 变现，分红层不蒸发。

    三件事一起验：
      1) 每笔卖出/清算：proceeds == book × (1−滑点)，且 book > 0；
      2) 分红层回收额 = Σ(proceeds − 原始股数×原始成交价) > 0（旧实现这个数恒等于 0）；
      3) 用 trades 反算现金、用 positions 反算持仓市值，必须与 nav 完全对上
         （产物自带对账能力，审计不用跑代码）。
    """
    cfg = Config(start="2018-01-02", end="2026-11-06", max_positions=10, max_weight=0.15)
    r = run_backtest(real_panel, momentum_signals(real_panel), cfg)
    slip = cfg.costs.slippage
    exits = r.trades[r.trades["side"].isin(("sell", "liquidate"))]
    assert len(exits) > 50, f"退出笔数只有 {len(exits)}，样本太小，测不出问题"

    # 1) 变现额口径
    assert np.allclose(exits["proceeds"], exits["book"] * (1.0 - slip),
                       rtol=0, atol=1e-6), "存在未按 hfq 口径变现的退出"
    assert (exits["book"] > 0).all(), "存在账面价值 <= 0 的退出"

    # 2) 分红层回收额（旧实现恒为 0）
    div = exits["proceeds"] - exits["shares"] * exits["price"]
    assert (div > -1e-6).all(), "存在变现额低于原始价回款的退出（口径写反了）"
    assert div.sum() > 0, "全样本没有任何分红层被回收 —— 退出路径又退回原始价了"
    years = len(r.nav) / 252
    avg_eq = float(r.nav["equity"].mean())
    print(f"\n[分红层] 退出 {len(exits)} 笔，回收合计 {div.sum():,.2f} 元 "
          f"（{div.sum() / avg_eq / years * 100:.2f}% / 年，占平均权益）")

    # 3) 产物对账：现金与持仓市值必须能从 trades / positions 独立重建
    buy = r.trades[r.trades["side"] == "buy"]
    cash_end = (cfg.init_cash - float((buy["amount"] + buy["fees"]).sum())
                + float((exits["proceeds"] - exits["fees"]).sum()))
    assert cash_end == pytest.approx(float(r.nav["cash"].iloc[-1]), abs=1e-6), (
        "用 trades 反算的期末现金与 nav.cash 不符")
    last = r.nav["date"].iloc[-1]
    pos_val = float(r.positions.loc[r.positions["date"] == last, "value"].sum())
    assert cash_end + pos_val == pytest.approx(float(r.nav["equity"].iloc[-1]), abs=1e-6), (
        "现金 + 持仓市值 与 nav.equity 不符")


def test_hand_calculated_nav_3stocks():
    """3 只股票 6 个交易日，逐笔手算净值，与引擎输出误差 < 0.01。

    手算过程（价格全程不变，只让 A 在第 4 日被卖出）：
      D1 收盘出信号（A 30%、B 20%），D2 开盘执行：
        D2 开盘 A=10.00、B=20.00，滑点 0.1% → A 买价 10.01、B 买价 20.02
        目标市值 A 30%×100000=30000 → 30000/10.01=2997.0 → 取整 2900 股 → 29029.00
              B 20%×100000=20000 → 20000/20.02= 999.0 → 取整  900 股 → 18018.00
        费用 A = max(7.2573,5) + 0.2903 = 7.5475；B = max(4.5045,5) + 0.1802 = 5.1802
        现金 = 100000 − 29029.00 − 7.5475 − 18018.00 − 5.1802 = 52940.2723
      D3 收盘出信号清仓 A（B 维持 20%），D4 开盘 A=10.00 → 卖出价 9.99，2900 股 = 28971.00
        费用 = 佣金 7.2428 + 过户费 0.2897 + 印花税 14.4855 = 22.0180 → 回款 28948.9820
        现金 = 52940.2723 + 28948.9820 = 81889.2543
      D4 收盘净值 = 81889.2543 + 900×20.5 = 100339.2543

    审计独立复算记录（来源：审计-B 第 B-06 条）
    ------------------------------------------------
    审计方用 Python `decimal.Decimal`（getcontext().prec=28）对上述六个环节
    独立复算一遍，与本引擎输出的误差逐项为 0.000000：
      · D2 买入 A 金额/费用       误差 0.000000
      · D2 买入 B 金额/费用       误差 0.000000
      · D2 收盘现金               误差 0.000000
      · D2 收盘净值               误差 0.000000
      · D4 卖出 A 金额/费用       误差 0.000000
      · D4 收盘净值               误差 0.000000
    即：手算—审计—引擎三方在本样本上完全一致，浮点误差未在分位上显现。
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


# ---------------------------------------------------------------- 7. 产物（R5）
def test_result_json_has_required_fields(tmp_path: Path):
    """result.json 缺 git_commit / 参数 / 数据区间 / 逐日净值 / 逐日持仓 任一字段即无效。"""
    dates = ["2020-01-02", "2020-01-03"]
    p = make_panel(dates, ["sh.600000"], [[10.0], [10.0]], [[10.0], [10.0]])
    cfg = Config(start=dates[0], end=dates[-1], init_cash=100_000, seed=7, n_hold=1)
    r = run_backtest(p, {dates[0]: {"sh.600000": 1.0}}, cfg)
    bench = M.build_benchmarks(p, r.nav["date"].tolist())
    m = M.compute_metrics(r.nav, r.trades, bench)
    out = tmp_path / "result.json"
    write_result(r, m, bench["summary"], out)
    d = json.loads(out.read_text(encoding="utf-8"))
    for key in ("git_commit", "params", "data", "nav", "positions", "metrics", "benchmarks"):
        assert key in d and d[key] not in (None, "", [], {}), f"result.json 缺 {key}"
    assert len(d["nav"]) == 2 and {"date", "equity"} <= set(d["nav"][0])
    assert d["data"]["start"] == dates[0] and d["data"]["end"] == dates[-1]
    # R5：params 必须含 start/end 与脚手架参数，否则产物无法独立复算
    for k in ("start", "end", "init_cash", "max_positions", "max_weight", "seed", "n_hold"):
        assert k in d["params"], f"params 缺 {k}"
    assert d["params"]["seed"] == 7 and d["params"]["n_hold"] == 1
    # 逐日持仓表必须真实落地且字段可用
    assert len(d["positions"]) >= 1
    assert {"date", "code", "shares", "value"} <= set(d["positions"][0])


def test_data_end_is_real_last_date_not_input():
    """R5：data.end 写真实末日，--end 入参单独存 requested_end。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    p = make_panel(dates, ["sh.600000"], [[10.0]] * 3, [[10.0]] * 3)
    p.source["requested_end"] = "2099-12-31"
    cfg = Config(start=dates[0], end="2099-12-31")
    r = run_backtest(p, {}, cfg)
    assert r.data["end"] == dates[-1], f"data.end 应为真实末日 {dates[-1]}，实际 {r.data['end']}"
    assert r.data["requested_end"] == "2099-12-31"


def test_end_clamp_upper_bound_matches_cache():
    """R5：--end 超范围要夹紧到缓存里最新年份，不能崩在缺文件上。"""
    from src.backtest import CACHE, _data_upper_bound
    years = sorted(int(f.stem.rsplit("_", 1)[-1]) for f in CACHE.glob("daily_*.parquet")
                   if f.stem.rsplit("_", 1)[-1].isdigit())
    assert years, "data/cache 下没有日线文件"
    assert _data_upper_bound() == f"{years[-1]}-12-31", (
        f"夹紧上界 {_data_upper_bound()} 与缓存实际最新年份 {years[-1]} 不一致")


# ---------------------------------------------------------------- 8. 指标护栏（R6）
def test_metrics_ann_return_vol_analytic():
    """年化与波动对解析解：504 日恒定日收益 → 年化 = 1.2^(1/2)−1，波动 = 0。"""
    n = 504
    idx = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    nav = pd.DataFrame({"date": idx, "equity": 1.2 ** (np.arange(n) / (n - 1)),
                        "cash": 0.0, "positions": 1})
    s = M.to_series(nav)
    assert M.ann_return(s) == pytest.approx(1.2 ** 0.5 - 1, abs=1e-9)
    assert M.ann_vol(s) == pytest.approx(0.0, abs=1e-12)
    assert M.max_drawdown(s)["mdd"] == pytest.approx(0.0, abs=1e-12)

    # 波动解析解：日收益在 +1% / −1% 交替，std(ddof=1) 可解析
    r = 0.01
    eq = [1.0]
    for k in range(1, 101):
        eq.append(eq[-1] * (1 + r if k % 2 else 1 - r))
    s2 = pd.Series(eq, index=pd.date_range("2020-01-01", periods=len(eq), freq="B")
                   .strftime("%Y-%m-%d"))
    daily = s2.pct_change().dropna()
    assert M.ann_vol(s2) == pytest.approx(float(daily.std(ddof=1)) * np.sqrt(252), abs=1e-15)


def test_metrics_mdd_analytic():
    """回撤解析解：1 → 1.5 → 0.9 → 1.2，最大回撤 = 0.9/1.5 − 1 = −40%。"""
    s = pd.Series([1.0, 1.5, 0.9, 1.2], index=["2020-01-02", "2020-01-03",
                                                "2020-01-06", "2020-01-07"])
    md = M.max_drawdown(s)
    assert md["mdd"] == pytest.approx(-0.4)
    assert md["peak"] == "2020-01-03"
    assert md["trough"] == "2020-01-06"


def test_metrics_sharpe_calmar_analytic():
    """夏普 = (年化 − 2%) / 波动；卡玛 = 年化 / |回撤|，都对解析解验。"""
    n = 504
    idx = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    # 有波动的序列：日收益在 +1.2% / −0.8% 间交替，但整体上行
    eq = [1.0]
    for k in range(1, n):
        eq.append(eq[-1] * (1 + (0.012 if k % 2 else -0.008)))
    nav = pd.DataFrame({"date": idx, "equity": eq, "cash": 0.0, "positions": 1})
    s = M.to_series(nav)
    ar, av = M.ann_return(s), M.ann_vol(s)
    assert M.sharpe(s, rf=0.02) == pytest.approx((ar - 0.02) / av, abs=1e-12)
    assert M.calmar(s) == pytest.approx(ar / abs(M.max_drawdown(s)["mdd"]), abs=1e-12)

    # 恒定收益 → 波动为 0 → 夏普与卡玛都必须是 NaN（不是 inf，也不是 0）
    flat = M.to_series(pd.DataFrame({"date": idx, "equity": 1.2 ** (np.arange(n) / (n - 1)),
                                     "cash": 0.0, "positions": 1}))
    assert np.isnan(M.sharpe(flat))
    assert np.isnan(M.calmar(flat))       # 无回撤 → 卡玛无定义


def test_metrics_win_rate_and_pl_ratio():
    """胜率 = 盈利笔数/已平仓笔数；盈亏比 = 平均盈利/|平均亏损|，清算计入平仓。"""
    trades = pd.DataFrame({
        "date": ["2020-01-02"] * 4,
        "code": ["a", "b", "c", "d"],
        "side": ["sell", "sell", "sell", "liquidate"],
        "pnl": [100.0, 300.0, -200.0, -100.0],
    })
    assert M.win_rate(trades) == pytest.approx(0.5)        # 2 盈 2 亏
    assert M.profit_loss_ratio(trades) == pytest.approx(200.0 / 150.0)
    # 只有买入记录 → 无已平仓交易 → NaN（不许写 0 或 999）
    only_buy = trades.assign(side="buy")
    assert np.isnan(M.win_rate(only_buy))
    assert np.isnan(M.profit_loss_ratio(only_buy))
    # 没有亏损笔 → 盈亏比 NaN
    all_win = trades[trades["pnl"] > 0]
    assert np.isnan(M.profit_loss_ratio(all_win))


def test_metrics_turnover_analytic():
    """年换手（双边）= 成交额 / 平均权益 / 年数。构造：504 日、权益恒 100、成交额 200 → 1.0。"""
    n = 504
    nav = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="B")
                        .strftime("%Y-%m-%d"), "equity": 100.0,
                        "cash": 100.0, "positions": 0})
    trades = pd.DataFrame({"date": ["2020-01-02"], "code": ["a"], "side": ["buy"],
                           "shares": [10.0], "price": [10.0],
                           "amount": [200.0], "fees": [0.0], "pnl": [np.nan]})
    assert M.turnover_rate(nav, trades) == pytest.approx(200.0 / 100.0 / (n / 252))


def test_metrics_excess_analytic():
    """超额 = 年化差 / 累计差；基准缺失必须 NaN，不许拿别的序列顶替。"""
    idx = pd.date_range("2020-01-01", periods=503, freq="B").strftime("%Y-%m-%d").tolist()
    strat = pd.Series(1.2 ** (np.arange(503) / 502), index=idx)
    bench = pd.Series(1.1 ** (np.arange(503) / 502), index=idx)
    e = M.excess(strat, bench)
    assert e["ann_excess"] == pytest.approx(M.ann_return(strat) - M.ann_return(bench), abs=1e-6)
    assert e["cum_excess"] == pytest.approx(0.2 - 0.1, abs=1e-6)
    miss = M.excess(strat, None)
    assert np.isnan(miss["ann_excess"]) and np.isnan(miss["cum_excess"])


def test_equal_weight_benchmark_calendar_aligned():
    """等权基准的日期轴必须与 panel.dates 严格对齐，不能错位或补零。"""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    codes = ["sh.600000", "sh.600001"]
    close = np.array([[10.0, 20.0], [11.0, 20.0], [12.0, 22.0]])
    p = make_panel(dates, codes, close, close)
    ew = M.equal_weight_market(p)
    assert list(ew.index) == dates, "等权基准日期轴与交易日历不一致"
    # 第 2 日：(11/10−1, 20/20−1) 均值 = 5%；第 3 日：(12/11−1, 22/20−1) 均值 = 4.545%
    assert ew.iloc[1] == pytest.approx(1.05)
    assert ew.iloc[2] == pytest.approx(1.05 * (1 + (0.0909090909 + 0.1) / 2), rel=1e-6)


def test_metrics_marks_missing_hs300(tmp_path: Path):
    """沪深300 数据没落地时必须标 missing，不许拿别的序列顶替。"""
    s, status = M.hs300_total_return(["2020-01-02"], raw_dir=tmp_path / "nope")
    assert s is None and status.startswith("missing")


# ---------------------------------------------------------------- 9. 涨跌停价自洽（R7）
def test_limit_price_selfconsistency():
    """复现注释里的统计：昨收反推 + 交易所四舍五入，收盘价==涨停价的自洽率。

    取一年（2021）数据跑，避免测试过慢；全量 2018–2026 的实测值见 src/backtest.py
    `_blocked_arrays` 的 docstring。这条测试守的是"断言不会被无声改坏"：
    自洽率跌破 99% 说明反推公式或涨跌幅规则被改错了。
    """
    from src.backtest import _ratio_array, round2

    d = pd.read_parquet(ROOT / "data" / "cache" / "daily_2021.parquet",
                        columns=["code", "date", "close", "pctChg", "isST", "listed_days"])
    d = d[d["listed_days"] > 1]                    # 新股首日无涨跌幅，不参与
    ratio = _ratio_array(d["code"], d["isST"].astype("int8"), d["date"])
    hit = d[(d["pctChg"] - ratio * 100).abs() < 0.02]
    assert len(hit) > 5000, f"涨跌停样本只有 {len(hit)} 行，口径可能错了"

    r = _ratio_array(hit["code"], hit["isST"].astype("int8"), hit["date"])
    pre = round2(hit["close"].to_numpy(dtype="float64")
                 / (1.0 + hit["pctChg"].to_numpy(dtype="float64") / 100.0))
    up = round2(pre * (1.0 + r))
    same = (np.abs(up - hit["close"].to_numpy(dtype="float64")) < 0.005).mean()
    assert same > 0.99, f"涨跌停价自洽率 {same:.4f} 低于 99%，反推公式可能被改坏"
    print(f"\n[自洽] 2021 年涨跌停样本 {len(hit)} 行，收盘==涨停价自洽率 {same*100:.3f}%")
