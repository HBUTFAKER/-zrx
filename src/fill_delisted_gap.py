"""退市股财务缺口的精准补齐（只补缺，不重抓）。

背景与为什么要有这个脚本：
- 东财全市场报表（src/ingest_ak_fin.py）把退市股从历史报告期里剔除了 → 生存偏差。
- baostock 是全市场唯一能给「真实首次披露日(pubDate)」的源：新浪/东财的「公告日期」
  实测是最新修订日（300356 的 2018Q1 显示 2019-04-27，滞后 392 天），拿去回测等于偷看未来。
- 但 baostock 封「请求频率」：5.5 次/秒跑 25 分钟即被拉黑 IP。所以这里只查缺口，
  并且强制限速（默认 1.2 秒/次），把请求量从 44.5 万次压到几千次。

补什么：
  不按「整只股票」补，而是按 (code, statDate) 组合补 —— 已覆盖的期次绝不重查。
  应覆盖的期次定义：该报告期末前后 ±45 个交易日内该股票在 pool_daily 里有行情。

用法：
  python -m src.fill_delisted_gap --req-interval 1.2
"""

from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import pandas as pd

from .clean import DB
from .ingest import (
    BannedError,
    FIN_INTERFACES,
    RAW_FIN,
    TODAY,
    _load_state,
    _login,
    _num,
    _query_with_retry,
    _save_state,
)

STATE = Path(__file__).resolve().parents[1] / "data" / "raw" / "_state"

# 只补这两张表就够：profit 给 ROE/毛利率/营收/净利润，cashflow 给经营现金流比率
INTERFACES = ("profit", "cashflow")


def covered_pairs() -> set[tuple[str, str]]:
    """现有原始财务文件里已覆盖的 (code, statDate)。"""
    out: set[tuple[str, str]] = set()
    for f in sorted(RAW_FIN.glob("finance_*.parquet")):
        d = pd.read_parquet(f, columns=["code", "statDate"])
        d = d.dropna()
        out |= set(zip(d["code"].astype(str), d["statDate"].astype(str)))
    return out


def expected_pairs() -> set[tuple[str, str]]:
    """按股票池算「应当有财报」的 (code, statDate)。

    口径：报告期末 ±45 个交易日内，该股票在 pool_daily 里有行情 → 视为该期在市，应有财报。
    这是保守口径（只多不少），宁可多查几次，也不能漏掉退市股。
    """
    con = pd.read_sql("SELECT date, code FROM pool_daily", _db())
    days = sorted(con["date"].unique())
    by_code: dict[str, list[str]] = {}
    for code, g in con.groupby("code"):
        by_code[code] = sorted(g["date"].unique())

    out: set[tuple[str, str]] = set()
    for p in PERIODS:
        i = bisect.bisect_left(days, p)
        lo = days[max(0, i - 45)]
        hi = days[min(len(days) - 1, i + 45)]
        live = con[(con["date"] >= lo) & (con["date"] <= hi)]["code"].unique()
        for c in live:
            out.add((c, p))
    return out


def _db():
    import sqlite3

    return sqlite3.connect(DB)


PERIODS = [f"{y}-{m}" for y in range(2018, 2023) for m in ("03-31", "06-30", "09-30", "12-31")]


def missing_pairs() -> list[tuple[str, str]]:
    have = covered_pairs()
    need = expected_pairs()
    return sorted(need - have)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-interval", type=float, default=1.2,
                    help="每次请求后强制休眠秒数；baostock 按频率封 IP，别低于 1")
    ap.add_argument("--limit", type=int, default=None, help="只补前 N 个组合（调试）")
    args = ap.parse_args(argv)

    miss = missing_pairs()
    if args.limit:
        miss = miss[:args.limit]
    if not miss:
        print("[gap] 无缺口，无需补齐")
        return 0

    codes = sorted({c for c, _ in miss})
    print(f"[gap] 缺口 {len(miss)} 个(股票×报告期)，涉及 {len(codes)} 只股票")
    print(f"[gap] 限速 {args.req_interval}s/次，预计 "
          f"{len(miss) * len(INTERFACES) * (args.req_interval + 0.25) / 60:.0f} 分钟")

    st_path = STATE / f"gap_{TODAY}.json"
    st = _load_state(st_path)
    done = set(st.get("done", []))

    try:
        bs = _login()
    except BannedError as e:
        print(f"[STOP] {e}")
        return 2

    rows: list[dict] = []
    bad: list[str] = []
    t0 = time.time()
    try:
        for k, (c, stat) in enumerate(miss, 1):
            key = f"{c}|{stat}"
            if key in done:
                continue
            y = int(stat[:4])
            q = (int(stat[5:7]) + 2) // 3
            rec: dict = {"code": c, "year": y, "quarter": q,
                         "pubDate": None, "statDate": None}
            got = False
            for nm in INTERFACES:
                fn_name, cols = FIN_INTERFACES[nm]
                f, r = _query_with_retry(bs, getattr(bs, fn_name),
                                         code=c, year=y, quarter=q, retries=2)
                time.sleep(args.req_interval)
                if not r:
                    continue
                got = True
                d = dict(zip(f, r[0]))
                rec["pubDate"] = rec["pubDate"] or (d.get("pubDate") or None)
                rec["statDate"] = rec["statDate"] or (d.get("statDate") or None)
                for col in cols:
                    rec[col] = _num(d.get(col))
            if got and rec["statDate"]:
                rows.append(rec)
                done.add(key)
            else:
                bad.append(key)
            if k % 40 == 0:
                _save_state(st_path, {"done": sorted(done), "bad": bad[-500:]})
                print(f"  进度 {k}/{len(miss)}  已取 {len(rows)} 行  "
                      f"耗时 {(time.time()-t0)/60:.1f} 分", flush=True)
    except BannedError:
        print("[STOP] 中途被拉黑，已保存进度，解封后重跑本命令即可续")
        _save_state(st_path, {"done": sorted(done), "bad": bad[-500:]})
        return 2
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass

    _save_state(st_path, {"done": sorted(done), "bad": bad[-500:]})
    if not rows:
        print(f"[gap] 无新增行（全部查询失败 {len(bad)} 个）")
        return 1

    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("code", "pubDate", "statDate"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["statDate"] = pd.to_datetime(df["statDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["fetch_date"] = TODAY
    out = RAW_FIN / f"finance_bs_gap_2018_2022_{TODAY}.parquet"
    df.to_parquet(out, index=False)
    print(f"[gap] 落盘 {out.name}: {len(df)} 行 / {df['code'].nunique()} 只")
    print(f"[gap] pubDate 缺失 {int(df['pubDate'].isna().sum())} 行，"
          f"倒挂 {int((pd.to_datetime(df['pubDate'], errors='coerce') < pd.to_datetime(df['statDate'])).sum())} 行")
    print(f"[gap] 仍失败 {len(bad)} 个组合")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
