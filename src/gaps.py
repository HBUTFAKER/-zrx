"""缺口台账 gaps.csv：拉不到的数据必须逐只落盘，禁止静默跳过。

为什么单列一个模块：`src/ingest.py` 的三条降级路径（日线 / 复权因子 / 财务）都要
写同一张表，逻辑只留一份；gaps.csv 是"生存偏差不静默"的唯一凭据，审计靠它核账。

三条降级路径对应的 gap_type
  1. 日线查询重试耗尽        → daily_missing
  2. 复权因子查询返回空      → factor_missing（可能确实从未分红，需人工核查，不自动当缺口剔除）
  3. 财务查询为空 / statDate 为空 → finance_missing:<year>
     其中「补抓脚本试过、源站一条没给」的再加一条 finance_fetch_failed
     （凭据来自 data/raw/_state/finance_cninfo_failed.json，抓不到也要留档）

重建时还会标出两类"看着像缺口、其实不是"的记录，避免台账被噪音淹没：
  daily_missing_recovered  当时失败，后续抓取已补上（留痕可回溯）
  daily_out_of_window      退市日早于抓取窗口起点，窗口内本就没有行情
  daily_never_fetched      2018 年后退市却一行行情都没有（B 股 900 段，从未进入抓取清单）

历史缺口由 `python -m src.gaps` 从 data/raw/_state/*.json 重建：只读现有状态文件，
不联网、不重抓、不编造。重建时会用**现有产物**复核一遍，把后来补抓成功的记录改标
`*_recovered`，避免台账把已补齐的股票一直挂着。

列固定四列：code / name / delist_date / gap_type
  name、delist_date 取自 cache.db 的 stocks 表；B 股 900 段不在 stocks（baostock type≠1），
  用 data/raw/delisted.csv 补——两个来源都是已落盘的事实，不做任何推断。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
STATE = RAW / "_state"
CACHE = DATA / "cache"
DB = DATA / "cache.db"
GAPS = DATA / "gaps.csv"
FIN_FAIL_LOG = STATE / "finance_cninfo_failed.json"  # 补抓脚本留下的失败名单
COLUMNS = ["code", "name", "delist_date", "gap_type"]

_META: dict[str, tuple[str, str]] | None = None


# ---------------------------------------------------------------- 元数据
def _delisted_csv() -> pd.DataFrame:
    """退市清单 → (code 带前缀, name, delist_date)；文件缺失时返回空表。"""
    f = RAW / "delisted.csv"
    if not f.exists():
        return pd.DataFrame(columns=["code", "name", "delist_date"])
    d = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    pre = d["exchange"].str.upper().map({"SH": "sh.", "SZ": "sz.", "BJ": "bj."}).fillna("sz.")
    out = pd.DataFrame({
        "code": pre + d["code"].str.zfill(6),
        "name": d["name"].fillna(""),
        "delist_date": d["delist_date"].fillna(""),
    })
    return out.drop_duplicates(subset=["code"])


def _connect() -> sqlite3.Connection | None:
    if not DB.exists():
        return None
    return sqlite3.connect(DB)


def _meta(con: sqlite3.Connection | None) -> dict[str, tuple[str, str]]:
    """code → (名称, 退市日)。stocks 优先；stocks 里没有的（B 股）用退市清单补。"""
    frames = []
    if con is not None:
        try:
            frames.append(pd.read_sql(
                "SELECT code, code_name AS name, delist_date FROM stocks", con))
        except Exception:  # noqa: BLE001  表不存在时退化为只用退市清单
            pass
    frames.append(_delisted_csv())
    m = pd.concat(frames, ignore_index=True)
    m["name"] = m["name"].fillna("")
    m["delist_date"] = m["delist_date"].fillna("")
    m = m.drop_duplicates(subset=["code"], keep="first")  # stocks 在前，优先
    return {r.code: (r.name, r.delist_date) for r in m.itertuples(index=False)}


def _meta_cached() -> dict[str, tuple[str, str]]:
    global _META
    if _META is None:
        con = _connect()
        try:
            _META = _meta(con)
        finally:
            if con is not None:
                con.close()
    return _META


def _row(code: str, gap_type: str, meta: dict[str, tuple[str, str]]) -> dict:
    name, dd = meta.get(code, ("", ""))
    return {"code": code, "name": name or "", "delist_date": dd or "", "gap_type": gap_type}


# ---------------------------------------------------------------- 读写
def read(path: Path = GAPS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""
    return d[COLUMNS]


def _write(df: pd.DataFrame, path: Path = GAPS) -> None:
    df = df.drop_duplicates(subset=["code", "gap_type"])
    df = df.sort_values(["gap_type", "code"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def record_gaps(items: list[tuple[str, str]], path: Path = GAPS) -> int:
    """ingest 用：追加若干 (code, gap_type)，与已有台账合并去重后重写。返回总行数。"""
    if not items:
        return len(read(path))
    meta = _meta_cached()
    new = pd.DataFrame([_row(c, t, meta) for c, t in sorted(set(items))])
    return merge(new, path)


def merge(new: pd.DataFrame, path: Path = GAPS) -> int:
    """把新记录并入台账（按 code+gap_type 去重），返回合并后总行数。"""
    old = read(path)
    allrows = pd.concat([old, new[COLUMNS]], ignore_index=True)
    _write(allrows, path)
    return len(allrows.drop_duplicates(subset=["code", "gap_type"]))


# ---------------------------------------------------------------- 从状态文件重建
def _daily_failed_codes() -> set[str]:
    """data/raw/_state/daily_*.json 里的 failed 名单（剔除非股票的 chunk 级错误）。"""
    codes: set[str] = set()
    for p in sorted(STATE.glob("daily_*.json")):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 状态文件解析失败，跳过 {p.name}: {e}")
            continue
        for x in st.get("failed", []):
            if ":" not in str(x):  # "chunk12:login timeout" 这类不是股票代码
                codes.add(str(x))
    return codes


OUT_OF_WINDOW = "2018-01-01"  # 日线抓取窗口起点；早于此退市的股票窗口内本就无行情


def build_daily(con: sqlite3.Connection | None, meta: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """日线缺口：failed 名单逐只落盘 + 2018 年后退市却一行行情都没有的股票。"""
    have = set()
    if con is not None:
        try:
            have = {r[0] for r in con.execute("SELECT DISTINCT code FROM pool_daily")}
        except Exception:  # noqa: BLE001
            have = set()

    rows = []
    for c in sorted(_daily_failed_codes()):
        if c in have:
            # 已能在 pool_daily 查到 = 后续补抓成功；保留痕迹但改标 recovered，不冒充当前缺口
            gt = "daily_missing_recovered"
        else:
            dd = meta.get(c, ("", ""))[1]
            # 退市日早于抓取窗口：当年请求了也确实没有数据，属正常，别再冒充缺口
            gt = "daily_out_of_window" if (dd and dd < OUT_OF_WINDOW) else "daily_missing"
        rows.append(_row(c, gt, meta))

    # 2018 年后退市清单里，pool_daily 一行都没有的：从未进入抓取清单（B 股 900 段）
    dl = _delisted_csv()
    dl = dl[pd.to_datetime(dl["delist_date"], errors="coerce") >= "2018-01-01"]
    for c in sorted(set(dl["code"]) - have):
        rows.append(_row(c, "daily_never_fetched", meta))

    return pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)


def _finance_fetch_failed() -> set[str]:
    """data/raw/_state/finance_cninfo_failed.json 里「试过、但源站一条没取到」的股票。

    与 finance_missing:<year> 的区别：后者只说「表里没有」，前者补一句「抓过，源站没有」。
    两者并存才是完整凭据 —— 不抓就报缺，和抓了确实没有，责任完全不同。
    """
    if not FIN_FAIL_LOG.exists():
        return set()
    try:
        d = json.loads(FIN_FAIL_LOG.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 财务失败名单解析失败，跳过 {FIN_FAIL_LOG.name}: {e}")
        return set()
    return set(d.get("items", {}))


def build_finance(con: sqlite3.Connection | None, meta: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """财务缺口：当年有行情（≥10 个交易日）的退市股，当年财务表里没有记录。

    口径与 tests/test_data.py 的 in_backtest_window 完全一致，保证台账和断言说的是同一件事。
    """
    fin_f = CACHE / "finance.parquet"
    if con is None or not fin_f.exists():
        print("  [warn] 缺 cache.db / finance.parquet，跳过财务缺口统计")
        return pd.DataFrame(columns=COLUMNS)
    dl = _delisted_csv()
    dl = dl[pd.to_datetime(dl["delist_date"], errors="coerce") >= "2018-01-01"]
    if dl.empty:
        print("  [warn] 缺 data/raw/delisted.csv 或其中无 2018 年后退市股，跳过财务缺口统计")
        return pd.DataFrame(columns=COLUMNS)
    dl_codes = set(dl["code"])

    fin = pd.read_parquet(fin_f, columns=["code", "year"])
    rows = []
    tried_failed = _finance_fetch_failed()
    tried_marked: set[str] = set()
    years = pd.read_sql("SELECT DISTINCT substr(date,1,4) y FROM pool_daily ORDER BY y", con)["y"].tolist()
    for y in years:
        pool = pd.read_sql(
            "SELECT code, COUNT(*) n FROM pool_daily WHERE date BETWEEN ? AND ? GROUP BY code",
            con, params=(f"{y}-01-01", f"{y}-12-31"))
        need = dl_codes & set(pool.loc[pool["n"] >= 10, "code"])
        have = set(fin.loc[fin["year"].astype(int) == int(y), "code"])
        for c in sorted(need - have):
            rows.append(_row(c, f"finance_missing:{y}", meta))
            # 抓过、源站就是没有的，再单独记一条，避免台账把"没抓"和"没有"混为一谈
            if c in tried_failed and c not in tried_marked:
                rows.append(_row(c, "finance_fetch_failed", meta))
                tried_marked.add(c)
    return pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)


def rebuild(path: Path = GAPS) -> pd.DataFrame:
    """全量重建：daily + finance 两部分，并保留台账里已有的其他记录（如 ingest 的 factor_missing）。"""
    con = _connect()
    try:
        meta = _meta(con)
        base = pd.concat([build_daily(con, meta), build_finance(con, meta)], ignore_index=True)
    finally:
        if con is not None:
            con.close()

    old = read(path)
    # 只重建 daily_* / finance_* 两类；ingest 追加的其他类型（factor_missing 等）原样保留
    keep = old[~old["gap_type"].str.startswith(("daily_", "finance_"))]
    out = pd.concat([base, keep], ignore_index=True)
    _write(out, path)
    print(f"[gaps] {path.relative_to(ROOT)} 共 {len(out)} 行")
    for gt, n in out["gap_type"].value_counts().sort_index().items():
        print(f"  {gt}: {n}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 _state 与现有产物重建 data/gaps.csv（不联网）")
    ap.add_argument("--path", default=str(GAPS), help="输出路径，默认 data/gaps.csv")
    args = ap.parse_args(argv)
    rebuild(Path(args.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
