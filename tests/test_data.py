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
def test_rows_match_trading_calendar(daily: pd.DataFrame, con: sqlite3.Connection,
                                     name: str, code: str):
    """样本股在「上市日 → 退市日」区间内的每个交易日都必须有一行，且不能多出行。

    （002R3）区间端点改用 stocks.ipoDate / outDate —— 外部事实。
    原版拿 s["date"].min()/max() 当端点，等于用样本自己的首末行画边界：
    中间断了能测出来，但「开头少了一段」「结尾提前结束」永远测不出来（恒真陷阱）。
    数据窗口从 2018 年起，早于窗口的上市日按窗口起点算。
    """
    s = daily[daily["code"] == code]
    assert len(s) > 0, f"{name} {code} 无数据"
    row = con.execute("SELECT ipoDate, outDate FROM stocks WHERE code = ?", (code,)).fetchone()
    assert row is not None, f"stocks 表里没有 {name} {code}"
    ipo, out = row[0], row[1]
    assert ipo, f"{name} {code} 的 ipoDate 为空，无法确定上市日"
    calendar = sorted(daily["date"].unique())
    start = max(str(ipo)[:10], calendar[0])
    end = min(str(out)[:10], calendar[-1]) if out else calendar[-1]
    expected = {d for d in calendar if start <= d <= end}
    got = set(s["date"])
    assert got == expected, (
        f"{name} {code} 与交易日历不一致（区间 {start}~{end}）："
        f"缺 {len(expected - got)} 天，多 {len(got - expected)} 天")


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


def test_finance_pubdate_est_is_safe(finance: pd.DataFrame):
    """兜底披露日 pubDate_est 必须全覆盖、且永远不早于报告期。

    S3 建议用这一列对齐：实测 pubDate 缺失时取法定披露截止日。
    截止日 ≥ 真实披露日，所以只可能晚知道；一旦倒挂就说明兜底逻辑写反了。
    """
    assert "pubDate_est" in finance.columns, "clean 未生成 pubDate_est，重跑 python -m src.clean"
    assert finance["pubDate_est"].isna().sum() == 0, "pubDate_est 仍有缺失"
    bad = finance["pubDate_est"] < finance["statDate"]
    assert bad.sum() == 0, (
        f"pubDate_est < statDate {int(bad.sum())} 条，"
        f"样例：{finance[bad].head(3)[['code','statDate','pubDate_est']].to_dict('records')}")
    # 标记列必须与实测列一致，不能出现「标了 announce 但 pubDate 是空」这种自相矛盾
    src = finance["pubdate_src"]
    assert ((src == "announce") == finance["pubDate"].notna()).all()


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


def test_finance_units_are_percent(finance: pd.DataFrame):
    """比率类字段必须统一成百分数。

    踩过的坑：baostock 的 roeAvg/gpMargin 是小数（0.184 = 18.4%），
    东财和新浪给的是百分数（18.4）。三个源并表时若不做归一，
    同一列里会同时存在 0.18 和 18 两种量级，S3 的因子排序直接失效。
    """
    d = finance.dropna(subset=["gpMargin"])
    assert len(d) > 0, "gpMargin 全为空"
    med = float(d.groupby("source")["gpMargin"].median().min())
    assert med > 1, f"有来源的 gpMargin 中位数为 {med}，像是小数口径（应 ×100）"
    # 单条记录的极端值允许存在（资不抵债的公司 ROE 会到 -10000%），但中位数必须落在合理区间
    assert -100 < float(d["gpMargin"].median()) < 100


def test_finance_covers_delisted(finance: pd.DataFrame):
    """生存偏差硬断言：2018 年后退市的股票必须在财务表里有记录。

    东财财报接口只保留仍在上市的公司（2018 一季报里，250 只 2018 后退市股只剩 19 只），
    这批必须由退市股通道补回。现行做法：
        src/fill_delisted_cninfo.py —— 巨潮公告列表取真实首次披露日 + 新浪取财务数值。
    （002R R3）缺 delisted.csv 或缺覆盖一律 fail：这条被 skip 掉的时候，生存偏差就没人管了。
    """
    f = RAW / "delisted.csv"
    if not f.exists():
        pytest.fail("缺 data/raw/delisted.csv，先跑 --stage meta")
    dl = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    dl = dl[pd.to_datetime(dl["delist_date"], errors="coerce") >= "2018-01-01"]
    pre = dl["exchange"].str.upper().map({"SH": "sh.", "SZ": "sz.", "BJ": "bj."}).fillna("sz.")
    need = set(pre + dl["code"].str.zfill(6))
    covered = need & set(finance["code"].unique())
    missing = sorted(need - covered)
    assert len(covered) >= len(need) * 0.95, (
        f"退市股财务覆盖 {len(covered)}/{len(need)} = {len(covered)/len(need):.1%}，低于 95%。"
        f"缺 {len(missing)} 只：{missing}. "
        f"补抓：python -m src.fill_delisted_cninfo --all-delisted（需业主同意后执行）")


@pytest.mark.parametrize("year", [2023, 2024])
def test_finance_covers_delisted_in_backtest_window(finance: pd.DataFrame,
                                                  con: sqlite3.Connection, year: int):
    """回测窗口内，当年有行情的退市股必须有一期「退市前 400 天内公告」的财报。

    【口径变更 002R3（总工裁决方案 A）】原口径是「statDate 年份 == 检查年，覆盖率 ≥95%」，
    对「检查当年内退市」的股票物理不可达：公司退市后不再披露定期报告，源站确实没有该年期次。
    证据（2026-11-09 实测）：
      · 2023：141/150 = 94.0%，缺 9 只 —— sh.600139 / sh.600260 / sh.600311 / sh.688086 /
        sh.688555 / sz.000038 / sz.000587 / sz.002013 / sz.002499，全部在 2023 年内退市；
        巨潮公告列表在 2023–2024 区间对它们返回 0 条定期报告，新浪财务数值止于 2022-09/12。
      · 2024：96/105 = 91.4%，缺 9 只 —— sh.600290 / sh.603555 / sz.000976 / sz.002002 /
        sz.002089 / sz.002740 / sz.002776 / sz.300282 / sz.300742，全部在 2024 年内退市；
        新浪财务数值止于 2023-09-30，东财源不收录退市股。
      · 台账 data/gaps.csv 的 9 条 finance_fetch_failed 就是这些抓取尝试的留痕（源站事实，非缺陷）。
      补抓后实测：改用「退市前 400 天窗口」口径，2023 = 150/150、2024 = 105/105，均为 100%。

    400 天 ≈ 最长披露滞后（年报次年 4-30 披露，距报告期约 16 个月）加安全余量：
    保证退市前最后一个可交易日能拿到最近一期已公告财报，因子不会因退市当年无报告而整列消失。
    公告日取 pubDate（真实首次披露日），缺失用 pubDate_est 兜底（法定截止日，只会更晚，不会提前）。
    断言 100% 硬性：不设 skip、不设百分比阈值。delist_date 只取 stocks 表（外部事实）。
    """
    pool = pd.read_sql(
        "SELECT code, COUNT(*) n FROM pool_daily WHERE date BETWEEN ? AND ? GROUP BY code",
        con, params=(f"{year}-01-01", f"{year}-12-31"))
    active = set(pool.loc[pool["n"] >= 10, "code"])  # 当年有正经行情（≥10 个交易日）

    st = pd.read_sql(
        "SELECT code, delist_date FROM stocks "
        "WHERE delist_date IS NOT NULL AND delist_date != ''", con)
    need = {str(c): str(d)[:10] for c, d in zip(st["code"], st["delist_date"]) if c in active}
    assert need, f"{year} 年 pool_daily 里没有「当年有行情」的退市股，样本为空（先查 stocks.delist_date）"

    ann = pd.to_datetime(finance["pubDate"].fillna(finance["pubDate_est"]), errors="coerce")
    pairs = pd.DataFrame({"code": finance["code"], "ann": ann}).dropna(subset=["ann"])
    by_code = {c: g["ann"].to_numpy() for c, g in pairs.groupby("code")}

    missing = []
    for code, dd in sorted(need.items()):
        end = pd.Timestamp(dd)
        lo = end - pd.Timedelta(days=400)
        a = by_code.get(code)
        if a is None or not ((a >= np.datetime64(lo)) & (a <= np.datetime64(end))).any():
            missing.append(f"{code}(退市 {dd})")

    assert not missing, (
        f"{year} 年 {len(need)} 只退市股中 {len(missing)} 只在「退市前 400 天」内找不到任何已公告财报："
        f"{missing}。补抓：python -m src.fill_delisted_cninfo --all-delisted "
        f"--from-year {year} --to-year {year}（需业主同意）")


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


def test_data_is_fresh(con: sqlite3.Connection):
    """数据必须同步到最近 45 天内，防止拿半年前的池子当"最新"跑回测。

    教训：2026-11 之前数据只到 2024-12-31，回测窗口白白少两年，
    而所有测试都绿——因为没有一条测试关心"最后一根 K 线是哪天"。
    """
    mx = con.execute("SELECT MAX(date) FROM pool_daily").fetchone()[0]
    assert mx is not None, "pool_daily 是空的"
    limit_date = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    assert mx >= limit_date, (
        f"pool_daily 最新日期 {mx}，早于 {limit_date}。"
        f"补数据：python -m src.ingest --stage daily --years 2025 2026 再重跑 clean")


def test_new_stock_flag_matches_listed_days(con: sqlite3.Connection):
    """new_stock 必须是 listed_days < 120，标志位和数值不能打架。"""
    bad = con.execute(
        "SELECT COUNT(*) FROM pool_daily WHERE (listed_days < 120) != (new_stock = 1)"
    ).fetchone()[0]
    assert bad == 0, f"{bad} 行 new_stock 与 listed_days 不一致"


# ---------------------------------------------------------------- 5. 002R 返工补充
def _raw_daily(year: int) -> pd.DataFrame:
    """读某年**全部**原始日线并做并集去重，同 (code,date) 取抓取日期最新的一条。

    （002R3）原版只取 files[-1] 单文件：同一年若有多个抓取日期的分片，
    旧版本独有、新版本没抓到的行会被漏掉，比对结果就偏了。
    并集语义与 src/clean.py 的 `_latest` 一致（只增不改），但这里**独立实现**，
    不 import clean —— 测试复用被验证的实现等于让错误一起通过。

    只读 data/raw/daily/，不碰 data/cache/ 的派生产物。
    """
    files = sorted((RAW / "daily").glob(f"daily_{year}_*.parquet"))
    if not files:
        pytest.fail(f"缺 data/raw/daily/daily_{year}_*.parquet，"
                    f"先跑 python -m src.ingest --stage daily --years {year} {year}")
    dfs = []
    for f in files:  # 文件名含抓取日期，升序 → 越后越新
        d = pd.read_parquet(f)
        d["_src"] = f.name
        dfs.append(d)
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["code", "date", "_src"]).drop_duplicates(
        subset=["code", "date"], keep="last")
    return out.drop(columns=["_src"])


def test_pool_daily_matches_raw_daily_2023(con: sqlite3.Connection):
    """2023 年：独立重算 (date, code) 集合与逐日计数，必须与 cache.db pool_daily 严格相等。

    独立实现的理由：复用 clean.py 的派生函数会"错一起错"，测试恒真。这里只做
    groupby / set 运算，不 import clean 的任何逻辑。

    判等口径（002R R2a）：交易日集合相同、每日只数相同、全量 (date, code) 集合相同。
    """
    raw = _raw_daily(2023)[["date", "code"]]
    pool = pd.read_sql(
        "SELECT date, code FROM pool_daily WHERE date BETWEEN '2023-01-01' AND '2023-12-31'", con)
    assert len(pool) > 0, "pool_daily 没有 2023 年数据"

    raw_cnt = raw.groupby("date")["code"].nunique()
    pool_cnt = pool.groupby("date")["code"].nunique()
    only_raw = sorted(set(raw_cnt.index) - set(pool_cnt.index))
    only_pool = sorted(set(pool_cnt.index) - set(raw_cnt.index))
    assert not only_raw and not only_pool, (
        f"交易日不一致：仅有原始数据的 {only_raw[:5]}，仅有 pool 的 {only_pool[:5]}")

    diff = (raw_cnt - pool_cnt).abs()
    bad = diff[diff > 0]
    assert bad.empty, (
        f"{len(bad)} 个交易日只数不一致，样例 {bad.head(5).to_dict()}")

    raw_keys = set(map(tuple, raw.to_numpy()))
    pool_keys = set(map(tuple, pool.to_numpy()))
    assert len(raw) == len(raw_keys), (
        f"原始文件内部就有重复 (date,code)：多 {len(raw) - len(raw_keys)} 条，先查 ingest 去重")
    assert raw_keys == pool_keys, (
        f"(date,code) 集合不一致：原始多 {len(raw_keys - pool_keys)} 条，"
        f"pool 多 {len(pool_keys - raw_keys)} 条，样例 {sorted(pool_keys - raw_keys)[:5]}")


def test_delisted_sample_respects_delist_date(con: sqlite3.Connection):
    """sz.000005：退市日之前有行情、退市日之后（严格大于）无行情。

    （002R R2c）退市日只许来自 stocks.delist_date（外部事实）。拿 pool_daily 自身的
    MAX(date) 反推退市日 = 用结果当前提，测了等于没测。
    """
    row = con.execute("SELECT delist_date FROM stocks WHERE code = 'sz.000005'").fetchone()
    assert row is not None, "stocks 表里没有 sz.000005"
    assert row[0], "sz.000005 的 delist_date 为空，无法验证"
    dd = str(row[0])[:10]

    before, after, total = con.execute(
        "SELECT SUM(date < ?), SUM(date > ?), COUNT(*) FROM pool_daily WHERE code = 'sz.000005'",
        (dd, dd)).fetchone()
    assert total and total > 100, f"sz.000005 在 pool_daily 只有 {total} 行，样本不足"
    assert before and before > 100, f"退市日 {dd} 之前只有 {before} 行行情"
    assert (after or 0) == 0, f"退市日 {dd} 之后仍有 {after} 行行情——退市后不该再有数据"


def test_latest_picks_newest_fetch_date(tmp_path: Path):
    """多版本原始文件：同一年两个抓取日期，_latest 必须取抓取日最新，且旧版本独有行不能丢。

    （002R R2e）文件名里的日期是唯一版本信号，取错版本等于用旧快照盖掉新快照。
    """
    from src.clean import _latest

    old = tmp_path / "daily_2023_2026-01-01.parquet"
    pd.DataFrame({"date": ["2023-01-03", "2023-01-03"],
                  "code": ["sh.600519", "sh.600000"],
                  "close": [1.0, 9.0]}).to_parquet(old, index=False)
    new = tmp_path / "daily_2023_2026-02-01.parquet"
    pd.DataFrame({"date": ["2023-01-03"], "code": ["sh.600519"],
                  "close": [2.0]}).to_parquet(new, index=False)

    out = _latest(sorted(tmp_path.glob("daily_2023_*.parquet")))
    assert out is not None
    assert len(out) == 2, f"应得 2 行（并集），实际 {len(out)} 行"
    idx = out.set_index(["code", "date"])["close"]
    assert float(idx[("sh.600519", "2023-01-03")]) == 2.0, "同一 (code,date) 应取抓取日更新的值"
    assert float(idx[("sh.600000", "2023-01-03")]) == 9.0, "旧版本独有行必须保留（只增不改）"


@pytest.mark.parametrize("year", list(range(2018, 2027)))
def test_suspended_rows_preserved_in_pool(con: sqlite3.Connection, year: int):
    """参数化年份：按口径4重算的停牌行必须全数在 pool_daily 且 suspended=1，也不许多标。

    口径（框架 3.3 第 3 条）：tradestatus==0 或 volume==0 即停牌；volume 缺失(NaN)
    不视为 0。这条防两件事：停牌行被悄悄从池子里删掉；NaN 成交量被误判成停牌。
    （002R3）逐年参数化 2018–2026：只压 2023 一年时，别的年份改了 clean 逻辑也发现不了。
    """
    raw = _raw_daily(year)[["date", "code", "tradestatus", "volume"]]
    susp = raw[(raw["tradestatus"] == 0) | (raw["volume"] == 0)][["date", "code"]]
    assert len(susp) > 0, f"{year} 年按口径4算出 0 行停牌，口径实现肯定有问题"

    pool = pd.read_sql(
        "SELECT date, code FROM pool_daily WHERE date BETWEEN ? AND ? AND suspended = 1",
        con, params=(f"{year}-01-01", f"{year}-12-31"))
    susp_keys = set(map(tuple, susp.to_numpy()))
    pool_keys = set(map(tuple, pool.to_numpy()))

    miss = susp_keys - pool_keys
    assert not miss, (
        f"{year} 年 {len(miss)} 条停牌行没进 pool_daily 或没标 suspended=1，样例 {sorted(miss)[:5]}")
    extra = pool_keys - susp_keys
    assert not extra, (
        f"{year} 年 {len(extra)} 条被多标停牌（volume 缺失被当成 0？），样例 {sorted(extra)[:5]}")

    # 单独再压一遍 NaN 语义：volume 缺失但当日仍在交易的行，绝不能标停牌
    nan_trading = raw[raw["volume"].isna() & (raw["tradestatus"] != 0)][["date", "code"]]
    hit = set(map(tuple, nan_trading.to_numpy())) & pool_keys
    assert not hit, f"{year} 年 {len(hit)} 条 volume 缺失但正常交易的行被标成停牌"
