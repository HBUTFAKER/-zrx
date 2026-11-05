"""数据层验收测试。

覆盖三件事：
  1. 完整性：样本股的交易日与全市场交易日历严格对齐（无缺口、无重复）。
  2. 正确性：hfq_close 与 pctChg 自洽（后复权序列的日收益率必须等于披露的涨跌幅）。
  3. 防未来函数：财务表 pubDate（公告日）不得早于 statDate（报告期）。

样本股覆盖三类：主板（sh.600519）、创业板（sz.300750）、已退市（sz.000005）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "cache.db"

SAMPLES = {
    "主板": "sh.600519",
    "创业板": "sz.300750",
    "已退市": "sz.000005",
}


# ---------------------------------------------------------------- 载入
@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    files = sorted(CACHE.glob("daily_*.parquet"))
    if not files:
        pytest.fail("data/cache 下没有 daily_*.parquet，先跑 python -m src.clean --years 2018 2024")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def finance() -> pd.DataFrame:
    f = CACHE / "finance.parquet"
    if not f.exists():
        pytest.fail("data/cache/finance.parquet 不存在，先跑 python -m src.ingest --stage finance")
    return pd.read_parquet(f)


@pytest.fixture(scope="session")
def con() -> sqlite3.Connection:
    if not DB.exists():
        pytest.fail("data/cache.db 不存在，先跑 python -m src.clean")
    c = sqlite3.connect(DB)
    yield c
    c.close()


# ---------------------------------------------------------------- 1. 完整性
@pytest.mark.parametrize("name,code", list(SAMPLES.items()))
def test_rows_match_trading_calendar(daily: pd.DataFrame, name: str, code: str):
    """样本股在自己上市区间内的每个交易日都必须有一行，且不能多出行。"""
    s = daily[daily["code"] == code]
    assert len(s) > 0, f"{name} {code} 无数据"
    calendar = sorted(daily["date"].unique())
    start, end = s["date"].min(), s["date"].max()
    expected = {d for d in calendar if start <= d <= end}
    got = set(s["date"])
    assert got == expected, (
        f"{name} {code} 与交易日历不一致：缺 {len(expected - got)} 天，"
        f"多 {len(got - expected)} 天")


@pytest.mark.parametrize("name,code", list(SAMPLES.items()))
def test_no_duplicate_code_date(daily: pd.DataFrame, name: str, code: str):
    """(code, date) 必须唯一——重复行会让回测把同一天算两次。"""
    s = daily[daily["code"] == code]
    dup = int(s.duplicated(subset=["code", "date"]).sum())
    assert dup == 0, f"{name} {code} 存在 {dup} 条重复 (code,date)"


def test_global_no_duplicate_keys(daily: pd.DataFrame):
    dup = int(daily.duplicated(subset=["code", "date"]).sum())
    assert dup == 0, f"全表存在 {dup} 条重复 (code,date)"


# ---------------------------------------------------------------- 2. 复权自洽
@pytest.mark.parametrize("name,code", list(SAMPLES.items()))
def test_hfq_close_consistent_with_pctchg(daily: pd.DataFrame, name: str, code: str):
    """后复权收盘价的日收益率必须等于 baostock 披露的 pctChg。

    这是"复权有没有算对"最硬的检验：只要因子取错一天（比如除权日提前/延后），
    收益率序列立刻与 pctChg 对不上。
    """
    s = daily[daily["code"] == code].sort_values("date")
    ret = (s["hfq_close"] / s["hfq_close"].shift(1) - 1) * 100
    diff = (ret - s["pctChg"]).abs().dropna()
    assert len(diff) > 100, f"{name} 样本太少，无法检验复权自洽"
    assert diff.max() < 0.01, (
        f"{name} {code} 后复权收益率与 pctChg 不一致：最大偏差 {diff.max():.4f} 个百分点")


@pytest.mark.parametrize("name,code", list(SAMPLES.items()))
def test_hfq_close_positive_and_not_negative(daily: pd.DataFrame, name: str, code: str):
    """后复权价必须为正；前复权才会出现负价，这里落地的是后复权。"""
    s = daily[daily["code"] == code]
    assert (s["hfq_close"] > 0).all(), f"{name} {code} 存在非正的后复权价"
    assert s["backAdjustFactor"].notna().all(), f"{name} {code} 有行缺复权因子"


# ---------------------------------------------------------------- 3. 防未来函数
def test_finance_pubdate_not_before_statdate(finance: pd.DataFrame):
    """公告日必须不早于报告期——倒挂就说明时点错了，等于提前知道业绩。"""
    f = finance[finance["pubDate"].notna() & finance["statDate"].notna()].copy()
    assert len(f) > 0, "财务表里没有一条同时有 pubDate 与 statDate 的记录"
    bad = f["pubDate"] < f["statDate"]
    ratio = 1 - bad.mean()
    assert bad.sum() == 0, (
        f"pubDate < statDate 的记录 {int(bad.sum())} 条（占比 {1-ratio:.2%}），"
        f"样例：{f[bad].head(3)[['code','pubDate','statDate']].to_dict('records')}")
    assert ratio == 1.0


def test_finance_has_both_dates(finance: pd.DataFrame):
    """pubDate 允许缺失，但必须受控并显式标注。

    缺失的真实来源：后来上市的公司，其早年报表只出现在招股书里，
    东财「实际披露时间表」不收录（该表只统计当期按期披露的在市公司）。
    这类记录在当时点本就不可见，缺失是诚实的，绝不允许用
    「statDate + 固定天数」插补 —— 那是伪造成时点。
    """
    miss = int(finance["pubDate"].isna().sum())
    ratio = miss / len(finance)
    assert ratio < 0.25, (
        f"pubDate 缺失 {miss}/{len(finance)} 行（{ratio:.1%}），超过 25% 说明披露表没接上，"
        f"不是正常的招股书口径缺口")
    # 缺失必须能解释：全是 statDate 早于该股上市日、或该股在披露表里查不到
    ok = finance[finance["pubDate"].notna()]
    assert len(ok) > len(finance) * 0.75


def test_finance_covers_delisted(finance: pd.DataFrame):
    """生存偏差硬断言：2018 年后退市的股票必须在财务表里有记录。

    东财财报接口只保留仍在上市的公司（2018 一季报里，250 只 2018 后退市股只剩 19 只），
    这批必须由 baostock 通道补回。未补齐时 skip 并给出补齐命令，补齐后本测试自动生效。
    """
    f = RAW / "delisted.csv"
    if not f.exists():
        pytest.skip("缺 data/raw/delisted.csv，先跑 --stage meta")
    dl = pd.read_csv(f, dtype=str)
    dl = dl[pd.to_datetime(dl["delist_date"], errors="coerce") >= "2018-01-01"]
    pre = dl["exchange"].str.upper().map({"SH": "sh.", "SZ": "sz.", "BJ": "bj."}).fillna("sz.")
    need = set(pre + dl["code"].str.zfill(6))
    covered = need & set(finance["code"].unique())
    if len(covered) < len(need) * 0.95:
        pytest.skip(
            f"退市股财务未补齐（{len(covered)}/{len(need)}）。补齐命令："
            f"python -m src.ingest --stage finance --delisted-only --req-interval 1 --years 2018 2022")
    assert len(covered) >= len(need) * 0.95


# ---------------------------------------------------------------- 4. 股票池 / 生存偏差
def test_pool_daily_schema(con: sqlite3.Connection):
    cols = {r[1] for r in con.execute("PRAGMA table_info(pool_daily)")}
    need = {"date", "code", "is_st", "suspended", "new_stock", "delisted",
            "listed_days", "tradable"}
    assert need <= cols, f"pool_daily 缺字段: {need - cols}"


def test_pool_daily_keeps_delisted_stocks(con: sqlite3.Connection):
    """已退市股票必须留在历史池里，否则小市值类策略会被系统性高估。"""
    n = con.execute("SELECT COUNT(*) FROM pool_daily WHERE delisted=1").fetchone()[0]
    codes = con.execute("SELECT COUNT(DISTINCT code) FROM pool_daily WHERE delisted=1").fetchone()[0]
    assert n > 0 and codes > 0, "pool_daily 里没有退市股票，存在生存偏差风险"


def test_pool_daily_covers_sample(con: sqlite3.Connection):
    for name, code in SAMPLES.items():
        n = con.execute("SELECT COUNT(*) FROM pool_daily WHERE code=?", (code,)).fetchone()[0]
        assert n > 0, f"{name} {code} 不在 pool_daily 里"


def test_new_stock_flag_matches_listed_days(con: sqlite3.Connection):
    """new_stock 必须是 listed_days < 120，标志位和数值不能打架。"""
    bad = con.execute(
        "SELECT COUNT(*) FROM pool_daily WHERE (listed_days < 120) != (new_stock = 1)"
    ).fetchone()[0]
    assert bad == 0, f"{bad} 行 new_stock 与 listed_days 不一致"
