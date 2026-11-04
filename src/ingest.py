"""数据抓取层：baostock 日线 + 季频财务，akshare 退市清单 + 行业分类。

数据纪律（框架文档 3.3 节，写进代码）：
  1. 时点对齐：财报必须同时保留 pubDate（公告日）与 statDate（报告期）。
     下游只能用 pubDate 做可用时点，绝不能用 statDate —— 否则等于提前 1-4 个月知道业绩。
  2. 复权：这里落的是**不复权**原始价（adjustflag=3）+ 复权因子。
     后复权价在 clean.py 里算，回测用后复权；界面展示用前复权。
  3. 停牌：保留 tradestatus 原始字段，停牌日在 clean.py 里标记，回测跳过。
  4. 生存偏差：股票清单来自 baostock 全量（含已退市，outDate 非空），
     绝不用"今天的股票列表"回测历史。
  5. 涨跌停：涨跌停判定在回测层用 pctChg + 板块涨跌幅限制做，此处只落原始价与涨跌幅。

缺失值处理：**只标注，不插补**。baostock 返回空串一律落为 NaN，不做任何估算。

落地约定：data/raw/ 下文件名一律带抓取日期，已存在的文件不覆盖（只增不改）。
增量 = 每个抓取日新增一批文件，clean.py 合并时按抓取日期取最新。
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_DAILY = RAW / "daily"
RAW_FIN = RAW / "finance"
RAW_META = RAW / "meta"
RAW_TMP = RAW / "_tmp"
STATE = RAW / "_state"

for _p in (RAW_DAILY, RAW_FIN, RAW_META, RAW_TMP, STATE):
    _p.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

# ---------------------------------------------------------------- 字段定义
DAILY_FIELDS = ["date", "code", "open", "high", "low", "close", "volume",
                "amount", "turn", "pctChg", "peTTM", "pbMRQ", "isST", "tradestatus"]
DAILY_NUM = ["open", "high", "low", "close", "volume", "amount", "turn",
             "pctChg", "peTTM", "pbMRQ"]

# 季频接口 → 需要的字段（全部原样保留，缺失即 NaN）
FIN_INTERFACES = {
    "profit": ("query_profit_data",
               ["roeAvg", "npMargin", "gpMargin", "netProfit", "epsTTM",
                "MBRevenue", "totalShare", "liqaShare"]),
    "growth": ("query_growth_data",
               ["YOYEquity", "YOYAsset", "YOYNI", "YOYEPSBasic", "YOYPNI"]),
    "operation": ("query_operation_data",
                  ["NRTurnRatio", "NRTurnDays", "INVTurnRatio", "INVTurnDays",
                   "CATurnRatio", "AssetTurnRatio"]),
    "cashflow": ("query_cash_flow_data",
                 ["CAToAsset", "NCAToAsset", "CFOToNP", "CFOToGr", "CFOToOR"]),
}
# 经营现金流没有"绝对值"字段，只有 CFOToOR（经营现金流/营业收入）。
# 需要绝对额时由 clean.py 用 MBRevenue × CFOToOR 推算，并在 dict.md 标明是推算列。
FIN_NUM_COLS = sorted({c for _, cols in FIN_INTERFACES.values() for c in cols})


# ---------------------------------------------------------------- 工具
def _num(v):
    """空串/None → NaN（缺失只标注，不插补）。"""
    if v is None:
        return float("nan")
    v = str(v).strip()
    if v == "" or v.lower() == "nan":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _write_parquet_once(df: pd.DataFrame, path: Path) -> bool:
    """只增不改：目标文件已存在则跳过。"""
    if path.exists():
        print(f"  [skip] 已存在，不覆盖: {path.relative_to(ROOT)}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  [write] {path.relative_to(ROOT)}  rows={len(df)}")
    return True


def _tag(years: list[int], limit: int | None, codes: list[str] | None) -> str:
    """分片文件名标签。crc32 保证跨进程稳定（内置 hash 每个进程不同，会毁掉续跑）。"""
    import zlib
    t = f"{min(years)}_{max(years)}"
    if limit:
        t += f"_l{limit}"
    if codes:
        t += f"_c{zlib.crc32(','.join(codes).encode()) % 100000}"
    return t


def _state_path(stage: str, years: list[int], tag: str | None = None) -> Path:
    """断点状态必须按 tag 隔离：定向跑过的分片不能污染全量跑的进度。"""
    t = tag or (f"{min(years)}_{max(years)}" if years else "all")
    return STATE / f"{stage}_{t}_{TODAY}.json"


def _load_state(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"done_chunks": [], "failed": []}


def _save_state(p: Path, st: dict) -> None:
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


class BannedError(RuntimeError):
    """baostock 按 IP 拉黑（并发过高触发），只能等解封或降并发。"""


BLACKLIST_HINT = (
    "baostock 拒绝登录（10001011 黑名单用户）。原因通常是并发连接过多。"
    "处理：把 --workers 降到 1~2，等 30 分钟以上再试。"
    "这不是代码问题，换账号无效，是 IP 级封禁。"
)


def _login(retries: int = 3):
    import baostock as bs
    last = None
    for i in range(retries):
        try:
            lg = bs.login()
            if lg.error_code == "0":
                return bs
            last = f"{lg.error_code} {lg.error_msg}"
            if "黑名单" in str(lg.error_msg):
                raise BannedError(BLACKLIST_HINT)
        except BannedError:
            raise
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 + 2 * i)
    raise RuntimeError(f"baostock 登录失败: {last}")


def _query_with_retry(bs, fn, retries: int = 3, min_rows: int = 0, **kw):
    """单次查询带重试；socket 掉线时重连（baostock 偶发 10057）。

    min_rows > 0 时做完整性校验：baostock 偶发"只返回前若干行但 error_code=0"的
    静默截断，行数不足即视为失败重试，避免把残缺数据当完整数据落盘。
    """
    for i in range(retries):
        try:
            rs = fn(**kw)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code == "0" and len(rows) >= min_rows:
                return rs.fields, rows
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5 + i)
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
        bs = _login()
    return [], []


# ---------------------------------------------------------------- 元数据
def ingest_meta(workers: int = 1) -> None:
    """股票清单（baostock 全量，含退市）、退市清单（akshare）、行业分类（akshare）。"""
    print("[meta] 股票清单 / 退市清单 / 行业分类")
    bs = _login()
    # 5000+ 只股票才合理；少于 3000 视为截断，重试
    fields, rows = _query_with_retry(bs, bs.query_stock_basic, retries=4, min_rows=3000)
    bs.logout()
    if not rows:
        raise RuntimeError("query_stock_basic 返回空或截断，检查网络后重跑 --stage meta")
    basic = pd.DataFrame(rows, columns=fields)
    basic = basic[basic["type"] == "1"].copy()  # 1=股票 2=指数 3=其他
    basic["outDate"] = basic["outDate"].replace("", pd.NA)
    basic["ipoDate"] = basic["ipoDate"].replace("", pd.NA)
    basic["is_delisted"] = basic["outDate"].notna()
    _write_parquet_once(basic, RAW_META / f"stocks_{TODAY}.parquet")
    print(f"  股票 {len(basic)} 只，其中已退市 {int(basic['is_delisted'].sum())} 只")

    # --- 退市清单（akshare）：data/raw/delisted.csv
    parts = []
    try:
        import akshare as ak
        for sym, src in (("全部", "sse"), ("终止上市公司", "szse")):
            try:
                fn = ak.stock_info_sh_delist if src == "sse" else ak.stock_info_sz_delist
                d = fn(symbol=sym)
                d = d.rename(columns={"公司代码": "code", "公司简称": "name",
                                      "上市日期": "ipo_date", "暂停上市日期": "delist_date"})
                d["exchange"] = "SH" if src == "sse" else "SZ"
                parts.append(d)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] akshare 退市清单 {src} 失败: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] akshare 不可用: {e}")
    if parts:
        dl = pd.concat(parts, ignore_index=True)
        dl["code"] = dl["code"].astype(str).str.zfill(6)
        dl = dl.drop_duplicates(subset=["code"])
        # 补 baostock 侧退市股（akshare 只覆盖沪深，北交所/更早的靠 baostock）
        bs_dl = basic.loc[basic["is_delisted"], ["code", "code_name", "ipoDate", "outDate"]]
        bs_dl = bs_dl.rename(columns={"code_name": "name", "ipoDate": "ipo_date",
                                      "outDate": "delist_date"})
        bs_dl["code"] = bs_dl["code"].str.split(".").str[1]
        bs_dl["exchange"] = bs_dl["code"].map(
            lambda c: "SH" if c.startswith("6") else ("BJ" if c.startswith(("4", "8")) else "SZ"))
        dl = pd.concat([dl[["code", "name", "ipo_date", "delist_date", "exchange"]],
                        bs_dl[["code", "name", "ipo_date", "delist_date", "exchange"]]],
                       ignore_index=True).drop_duplicates(subset=["code"], keep="first")
        out = RAW / "delisted.csv"
        if out.exists():
            print(f"  [skip] 已存在: {out.relative_to(ROOT)}")
        else:
            dl.sort_values("code").to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  [write] {out.relative_to(ROOT)}  rows={len(dl)}")

    # --- 行业分类（akshare 东财行业板块成分；失败则退回 baostock 证监会行业）
    ind_rows = []
    try:
        import akshare as ak
        names = ak.stock_board_industry_name_em()
        col = "板块名称" if "板块名称" in names.columns else names.columns[1]
        boards = names[col].tolist()
        for i, b in enumerate(boards, 1):
            try:
                c = ak.stock_board_industry_cons_em(symbol=b)
                code_col = "代码" if "代码" in c.columns else c.columns[1]
                for cd in c[code_col].astype(str).str.zfill(6):
                    ind_rows.append({"code": cd, "industry": b, "source": "akshare_em"})
            except Exception:  # noqa: BLE001
                continue
            if i % 20 == 0:
                print(f"  行业板块进度 {i}/{len(boards)}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] akshare 行业失败: {e}，改用 baostock 证监会行业")
    if ind_rows:
        ind = pd.DataFrame(ind_rows).drop_duplicates(subset=["code"])
        _write_parquet_once(ind, RAW_META / f"industry_{TODAY}.parquet")
    else:
        bs = _login()
        f2, r2 = _query_with_retry(bs, bs.query_stock_industry)
        bs.logout()
        if r2:
            ind = pd.DataFrame(r2, columns=f2)[["code", "industry"]]
            ind["code"] = ind["code"].str.split(".").str[1]
            ind["source"] = "baostock_csrc"
            _write_parquet_once(ind, RAW_META / f"industry_{TODAY}.parquet")


# ---------------------------------------------------------------- 复权因子
def _fetch_adjust_factors(codes: list[str]) -> pd.DataFrame:
    bs = _login()
    out = []
    for i, c in enumerate(codes, 1):
        f, r = _query_with_retry(bs, bs.query_adjust_factor, code=c,
                                 start_date="1990-01-01", end_date=TODAY)
        for row in r:
            out.append({"code": row[0], "dividOperateDate": row[1],
                        "foreAdjustFactor": _num(row[2]), "backAdjustFactor": _num(row[3])})
        if i % 500 == 0:
            print(f"    复权因子 {i}/{len(codes)}")
    bs.logout()
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 日线
def _daily_chunk(job: tuple) -> dict:
    """子进程任务：给定一批代码，拉日线 + 复权因子，写 part 文件。"""
    idx, codes, start, end, tag = job
    try:
        bs = _login()
    except BannedError as e:
        return {"idx": idx, "ok": False, "banned": True, "err": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "ok": False, "err": f"login: {e}"}
    frames, fac_rows, bad = [], [], []
    for c in codes:
        f, r = _query_with_retry(
            bs, bs.query_history_k_data_plus, code=c,
            fields=",".join(DAILY_FIELDS), start_date=start, end_date=end,
            frequency="d", adjustflag="3")  # 3=不复权：原始价是唯一可溯源底座
        if not r:
            bad.append(c)
        else:
            frames.append(pd.DataFrame(r, columns=f))
        f2, r2 = _query_with_retry(bs, bs.query_adjust_factor, code=c,
                                   start_date="1990-01-01", end_date=end)
        for row in r2:
            fac_rows.append({"code": row[0], "dividOperateDate": row[1],
                             "foreAdjustFactor": _num(row[2]), "backAdjustFactor": _num(row[3])})
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass

    if not frames:
        return {"idx": idx, "ok": True, "rows": 0, "bad": bad}
    df = pd.concat(frames, ignore_index=True)
    for c in DAILY_NUM:
        df[c] = df[c].map(_num)
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(-1).astype("int8")
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(-1).astype("int8")
    df["year"] = df["date"].str.slice(0, 4).astype("int16")
    df["fetch_date"] = TODAY

    p = RAW_TMP / f"daily_{tag}_{idx}.parquet"
    df.to_parquet(p, index=False)
    if fac_rows:
        pd.DataFrame(fac_rows).to_parquet(RAW_TMP / f"factor_{tag}_{idx}.parquet", index=False)
    return {"idx": idx, "ok": True, "rows": len(df), "bad": bad}


def ingest_daily(years: list[int], workers: int = 3, chunk: int = 120,
                 limit: int | None = None, codes: list[str] | None = None) -> None:
    """全 A 日线，按年分 parquet 落地。"""
    stocks = _load_stocks()
    codes = codes or stocks["code"].tolist()
    if limit:
        codes = codes[:limit]
    start, end = f"{min(years)}-01-01", f"{max(years)}-12-31"
    # --limit 调试跑的产物单独打标，避免"只跑了前 N 只"的文件占住正式文件名
    tag = _tag(years, limit, codes)
    st_path = _state_path("daily", years, tag)
    st = _load_state(st_path)

    jobs = [(i, codes[i:i + chunk], start, end, tag)
            for i in range(0, len(codes), chunk)]
    jobs = [j for j in jobs if j[0] not in st["done_chunks"]]
    print(f"[daily] {len(codes)} 只 × {start}~{end}，{len(jobs)} 个分片待跑，workers={workers}")

    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for k, res in enumerate(pool.imap_unordered(_daily_chunk, jobs), 1):
                if res.get("banned"):
                    pool.terminate()
                    _save_state(st_path, st)
                    raise BannedError(BLACKLIST_HINT)
                if res.get("ok"):
                    st["done_chunks"].append(res["idx"])
                    if res.get("bad"):
                        st["failed"].extend(res["bad"])
                else:
                    st["failed"].append(f"chunk{res['idx']}:{res['err']}")
                if k % 5 == 0 or k == len(jobs):
                    _save_state(st_path, st)
                    print(f"  进度 {k}/{len(jobs)} 分片")
    else:
        for k, job in enumerate(jobs, 1):
            res = _daily_chunk(job)
            if res.get("banned"):
                _save_state(st_path, st)
                raise BannedError(BLACKLIST_HINT)
            if res.get("ok"):
                st["done_chunks"].append(res["idx"])
                st["failed"].extend(res.get("bad", []))
            _save_state(st_path, st)
            print(f"  进度 {k}/{len(jobs)} 分片")

    _merge_daily_parts(years, tag)


def _merge_daily_parts(years: list[int], tag: str) -> None:
    """合并分片 → 按年落 parquet。

    与"只增不改"的关系：raw 文件的多版本靠**文件名里的抓取日期**区分，历史版本永不删除。
    同一天内断点续跑时，用"已有当日文件 + 新分片"并集重写当日文件——这不是修改历史，
    只是把同一抓取日的数据补全（同一 (code,date) 保留最后写入的那条）。
    """
    parts = sorted(RAW_TMP.glob(f"daily_{tag}_*.parquet"))
    if not parts:
        print("  [warn] 无 part 文件可合并")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "date"], keep="last")
    for y in years:
        sub = df[df["year"] == y].drop(columns=["year"])
        if sub.empty:
            continue
        out = RAW_DAILY / f"daily_{y}_{TODAY}.parquet"
        if out.exists():
            old = pd.read_parquet(out)
            sub = pd.concat([old, sub], ignore_index=True)
            sub = sub.drop_duplicates(subset=["code", "date"], keep="last")
        sub = sub.sort_values(["code", "date"])
        sub.to_parquet(out, index=False)
        print(f"  [write] {out.relative_to(ROOT)}  rows={len(sub)}  "
              f"codes={sub['code'].nunique()}")

    # 复权因子（全量合并成一个文件）
    fps = sorted(RAW_TMP.glob(f"factor_{tag}_*.parquet"))
    if fps:
        fac = pd.concat([pd.read_parquet(p) for p in fps], ignore_index=True)
        fac = fac.drop_duplicates(subset=["code", "dividOperateDate"], keep="last")
        fo = RAW_META / f"adjust_factor_{TODAY}.parquet"
        if fo.exists():
            old = pd.read_parquet(fo)
            fac = pd.concat([old, fac], ignore_index=True).drop_duplicates(
                subset=["code", "dividOperateDate"], keep="last")
        fac.to_parquet(fo, index=False)
        print(f"  [write] {fo.relative_to(ROOT)}  rows={len(fac)}  "
              f"codes={fac['code'].nunique()}")
    for p in parts + fps:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------- 财务
def _finance_chunk(job: tuple) -> dict:
    """子进程任务：一批代码 × 年 × 季 × 接口。pubDate 与 statDate 必须都在。"""
    idx, codes, years, interfaces, tag = job
    try:
        bs = _login()
    except BannedError as e:
        return {"idx": idx, "ok": False, "banned": True, "err": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "ok": False, "err": f"login: {e}"}
    rows, bad = [], []
    for c in codes:
        for y in years:
            for q in (1, 2, 3, 4):
                rec = {"code": c, "year": y, "quarter": q,
                       "pubDate": None, "statDate": None}
                got = False
                for nm in interfaces:
                    fn_name, cols = FIN_INTERFACES[nm]
                    f, r = _query_with_retry(bs, getattr(bs, fn_name),
                                             code=c, year=y, quarter=q, retries=2)
                    if not r:
                        continue
                    got = True
                    row = r[0]
                    d = dict(zip(f, row))
                    rec["pubDate"] = rec["pubDate"] or (d.get("pubDate") or None)
                    rec["statDate"] = rec["statDate"] or (d.get("statDate") or None)
                    for col in cols:
                        rec[col] = _num(d.get(col))
                if got and rec["statDate"]:
                    rows.append(rec)
                elif not got:
                    bad.append(f"{c}:{y}Q{q}")
    try:
        bs.logout()
    except Exception:  # noqa: BLE001
        pass
    if not rows:
        return {"idx": idx, "ok": True, "rows": 0, "bad": bad}
    df = pd.DataFrame(rows)
    for c in FIN_NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["statDate"] = pd.to_datetime(df["statDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["fetch_date"] = TODAY
    p = RAW_TMP / f"fin_{tag}_{idx}.parquet"
    df.to_parquet(p, index=False)
    return {"idx": idx, "ok": True, "rows": len(df), "bad": bad}


def ingest_finance(years: list[int], workers: int = 3, chunk: int = 150,
                   interfaces: list[str] | None = None, limit: int | None = None,
                   codes: list[str] | None = None) -> None:
    interfaces = interfaces or list(FIN_INTERFACES)
    stocks = _load_stocks()
    codes = codes or stocks["code"].tolist()
    if limit:
        codes = codes[:limit]
    tag = _tag(years, limit, codes)
    st_path = _state_path("finance", years, tag)
    st = _load_state(st_path)
    jobs = [(i, codes[i:i + chunk], years, tuple(interfaces), tag)
            for i in range(0, len(codes), chunk)]
    jobs = [j for j in jobs if j[0] not in st["done_chunks"]]
    print(f"[finance] {len(codes)} 只 × {years[0]}~{years[-1]} × {interfaces}，"
          f"{len(jobs)} 个分片待跑，workers={workers}")

    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for k, res in enumerate(pool.imap_unordered(_finance_chunk, jobs), 1):
                if res.get("banned"):
                    pool.terminate()
                    _save_state(st_path, st)
                    raise BannedError(BLACKLIST_HINT)
                if res.get("ok"):
                    st["done_chunks"].append(res["idx"])
                    if res.get("bad"):
                        st["failed"].extend(res["bad"][:200])
                else:
                    st["failed"].append(f"chunk{res['idx']}:{res['err']}")
                if k % 5 == 0 or k == len(jobs):
                    _save_state(st_path, st)
                    print(f"  进度 {k}/{len(jobs)} 分片")
    else:
        for k, job in enumerate(jobs, 1):
            res = _finance_chunk(job)
            if res.get("banned"):
                _save_state(st_path, st)
                raise BannedError(BLACKLIST_HINT)
            if res.get("ok"):
                st["done_chunks"].append(res["idx"])
                st["failed"].extend(res.get("bad", [])[:200])
            _save_state(st_path, st)
            print(f"  进度 {k}/{len(jobs)} 分片")

    _merge_finance_parts(years, tag)


def _merge_finance_parts(years: list[int], tag: str) -> None:
    parts = sorted(RAW_TMP.glob(f"fin_{tag}_*.parquet"))
    if not parts:
        print("  [warn] 无 part 文件可合并")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "statDate"], keep="last")
    df["year"] = df["statDate"].str.slice(0, 4).astype("int16")
    for y in years:
        sub = df[df["year"] == y].drop(columns=["year"])
        if sub.empty:
            continue
        out = RAW_FIN / f"finance_{y}_{TODAY}.parquet"
        if out.exists():  # 同一天续跑：并集补全，不覆盖历史抓取日
            old = pd.read_parquet(out)
            sub = pd.concat([old, sub], ignore_index=True)
            sub = sub.drop_duplicates(subset=["code", "statDate"], keep="last")
        sub = sub.sort_values(["code", "statDate"])
        sub.to_parquet(out, index=False)
        print(f"  [write] {out.relative_to(ROOT)}  rows={len(sub)}  "
              f"codes={sub['code'].nunique()}")
    for p in parts:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------- 清单
def _load_stocks() -> pd.DataFrame:
    files = sorted(RAW_META.glob("stocks_*.parquet"))
    if not files:
        raise SystemExit("缺少股票清单，先跑: python -m src.ingest --stage meta")
    df = pd.read_parquet(files[-1])
    return df.sort_values("code").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="量化数据抓取（baostock + akshare）")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="年份范围，如 --years 2018 2024")
    ap.add_argument("--stage", default="all",
                    choices=["all", "meta", "daily", "finance"])
    ap.add_argument("--workers", type=int, default=3,
                    help="并发进程数；baostock 高并发会被拉黑，建议 ≤3")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 只（调试用）")
    ap.add_argument("--codes", default=None,
                    help="只跑指定代码，逗号分隔，如 sh.600519,sz.300750（调试用）")
    ap.add_argument("--interfaces", default=",".join(FIN_INTERFACES),
                    help="profit,growth,operation,cashflow 的子集")
    args = ap.parse_args(argv)

    socket.setdefaulttimeout(120)
    years = args.years or [date.today().year]
    if len(years) == 1:
        years = [years[0], years[0]]
    years = list(range(min(years), max(years) + 1))

    t0 = time.time()
    try:
        _run_stages(args, years)
    except BannedError as e:
        print(f"[STOP] {e}", file=sys.stderr)
        return 2
    print(f"[done] 用时 {time.time()-t0:.1f}s")
    return 0


def _run_stages(args, years: list[int]) -> None:
    if args.stage in ("all", "meta"):
        ingest_meta()
    only = [c.strip() for c in args.codes.split(",")] if args.codes else None
    if args.stage in ("all", "daily"):
        ingest_daily(years, workers=args.workers,
                     chunk=args.chunk or 120, limit=args.limit, codes=only)
    if args.stage in ("all", "finance"):
        ifs = [s.strip() for s in args.interfaces.split(",") if s.strip()]
        ingest_finance(years, workers=args.workers, chunk=args.chunk or 150,
                       interfaces=ifs, limit=args.limit, codes=only)


if __name__ == "__main__":
    sys.exit(main())
