"""沪深 300 全收益指数（H00300）补抓：只抓这一个序列，落盘后由 metrics 自动生效。

为什么要单写一个模块：S2 的基准要两个（框架 5.1），全市场等权由 pool_daily 现算，
沪深 300 全收益必须来自**外部权威源**——价格指数 000300 不含分红再投，
长期低估约 2%/年，冒充全收益等于篡改基准（三条底线：不编数据）。

数据源：中证指数有限公司官网指数详情接口（akshare `stock_zh_index_hist_csindex`），
指数代码 H00300 = 沪深300全收益指数（英文名 CSI 300 Total Return Index）。

口径处理（关键）：
  1. 中证官网会在**休市日**返回占位行（成交额 0、涨跌为 0 附近的伪值），
     实测全序列 5307 行里有 1 行（2018-06-18 端午）。用新浪权威交易日历
     （akshare `tool_trade_date_hist_sina`）剔除，不做任何插补。
  2. 2004–2009 的成交额字段整体为 0（官网早期无成交额），那不是占位行——
     判据用「是否交易日」而非「成交额是否为 0」，否则会误删 1164 行真行情。
  3. 落盘列 date/close（loader 契约），另附 source/symbol/fetch_date 便于溯源。

只增不改：文件名带抓取日期，历史版本永不删除；同一抓取日重跑则重写当日文件
（与 data/raw/daily 的同日并集语义一致）。

用法：
    python -m src.fetch_index            # 抓全量并落盘
    python -m src.fetch_index --dry-run  # 只校验不落盘
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_INDEX = DATA / "raw" / "index"

SYMBOL = "H00300"          # 沪深300全收益指数
SYMBOL_NAME = "沪深300全收益指数"
START = "20041231"         # 基日：沪深300 以 2004-12-31 为 1000 点
TODAY = date.today().isoformat()


def _trade_calendar() -> set[str]:
    """新浪权威交易日历（1990 至次年末），用于剔除官网的休市占位行。"""
    import akshare as ak

    cal = ak.tool_trade_date_hist_sina()
    return set(pd.to_datetime(cal["trade_date"]).dt.strftime("%Y-%m-%d"))


def fetch_hs300_total_return() -> tuple[pd.DataFrame, dict]:
    """抓 H00300 全序列并按交易日历清洗。返回 (df, 统计信息)。"""
    import akshare as ak

    raw = ak.stock_zh_index_hist_csindex(symbol=SYMBOL, start_date=START, end_date="20991231")
    if raw is None or raw.empty:
        raise RuntimeError(f"中证接口返回空：{SYMBOL}")

    # 只认这一个序列，接口若换了代码/名称立即停手，不静默接受
    codes = set(raw["指数代码"].astype(str).unique())
    if codes != {SYMBOL}:
        raise RuntimeError(f"接口返回的指数代码不是 {SYMBOL}：{sorted(codes)}")

    d = pd.DataFrame({
        "date": pd.to_datetime(raw["日期"]).dt.strftime("%Y-%m-%d"),
        "close": pd.to_numeric(raw["收盘"], errors="coerce"),
    }).sort_values("date").reset_index(drop=True)

    if d["close"].isna().any():
        raise RuntimeError(f"收盘价存在缺失 {int(d['close'].isna().sum())} 行，不接受插补")
    if d["date"].duplicated().any():
        raise RuntimeError(f"日期重复 {int(d['date'].duplicated().sum())} 行")

    cal = _trade_calendar()
    off_cal = d[~d["date"].isin(cal)]
    kept = d[d["date"].isin(cal)].reset_index(drop=True)

    stats = {
        "rows_raw": int(len(d)),
        "rows_kept": int(len(kept)),
        "dropped_non_trading": off_cal["date"].tolist(),
        "start": kept["date"].iloc[0],
        "end": kept["date"].iloc[-1],
        "close_first": float(kept["close"].iloc[0]),
        "close_last": float(kept["close"].iloc[-1]),
    }
    return kept, stats


def write_index(df: pd.DataFrame, out_dir: Path = RAW_INDEX) -> Path:
    """落盘 data/raw/index/hs300_total_return_{抓取日}.parquet（只增不改）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hs300_total_return_{TODAY}.parquet"
    payload = df.copy()
    payload["symbol"] = SYMBOL
    payload["source"] = "csindex.com.cn via akshare.stock_zh_index_hist_csindex"
    payload["fetch_date"] = TODAY
    payload.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="补抓沪深300全收益指数（H00300）")
    ap.add_argument("--dry-run", action="store_true", help="只抓取校验，不落盘")
    args = ap.parse_args(argv)

    df, st = fetch_hs300_total_return()
    print(f"[index] {SYMBOL_NAME}（{SYMBOL}）：原始 {st['rows_raw']} 行 → 交易日 {st['rows_kept']} 行")
    print(f"[index] 区间 {st['start']} ~ {st['end']}，收盘 {st['close_first']} → {st['close_last']}")
    if st["dropped_non_trading"]:
        print(f"[index] 剔除休市占位行 {len(st['dropped_non_trading'])} 行：{st['dropped_non_trading']}")

    if args.dry_run:
        print("[index] --dry-run，未落盘")
        return 0

    path = write_index(df)
    print(f"[index] 已写入 {path.relative_to(ROOT)}（{len(df)} 行）")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"[index] 失败：{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
