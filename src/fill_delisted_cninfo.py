"""退市股财务缺口的补齐（巨潮公告日 + 新浪财务数值）。

为什么不能用东财/新浪自带的「公告日期」：
    那两家的公告日期是「最新修订日」，不是首次披露日。
    实测 sz.300356 的 2018 年一季报，其「公告日期」是 2019-04-27（滞后 392 天），
    拿它做时点对齐等于提前一年看到数据 —— 直接违反框架第 0 节「防未来函数」底线。

    真值来自巨潮资讯（cninfo）的公告原文列表：
    300356「2022年年度报告」= 2023-04-27、「2022年三季度报告」= 2022-10-26；
    002680「2018年半年度报告」= 2018-12-29（这家当年确实拖到年底才披露）。
    连「无法在法定期限内披露…」的情形都能识别 —— 那种期次本就没有报告，保持 NaN 才是对的。

数值口径：
    新浪 stock_financial_abstract 的「营业总收入 / 净利润 / 经营现金流量净额 /
    净资产收益率_平均 / 毛利率」，与东财源的 netProfit / MBRevenue 同口径（均为含少数股东损益）。

用法：
    python -m src.fill_delisted_cninfo
"""

from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd

from .fill_delisted_gap import missing_pairs
from .ingest import RAW_FIN, TODAY

STATE = RAW_FIN.parent / "_state"
FAIL_LOG = STATE / "finance_cninfo_failed.json"

# 标题里出现这些词的一律不是「首次披露的定期报告正文」
# 注意：不能排除「全文/正文」—— 大量定期报告的真实标题就是「2020年第三季度报告全文」，
# 排掉它们会让缺了公告日的那几期被误判成「没披露」。
BAD_WORDS = ("摘要", "更新后", "更正", "修订", "已取消", "英文",
             "披露提示", "提示性公告", "问询", "说明会", "业绩预告", "风险提示",
             "延期", "关于", "补充", "核查意见", "专项", "决算", "审计报告",
             "内部控制", "述职", "总经理", "董事会", "监事会", "财务报告")

# 「2022年年度报告」「2023年一季度报告」「2018年半年度报告」「2020年第三季度报告全文」……
# 用 search 而非 match：部分标题带公司名前缀（如「东方金钰股份有限公司2020年三季度报告」），
# 噪音标题已由 BAD_WORDS 挡掉（「关于…」一律排除）。
TITLE_RE = re.compile(
    r"(?P<y>20\d{2})\s*年\s*"
    r"(?P<kind>年度|半年度|中期|第一季度|一季度|第三季度|三季度)\s*报告\s*"
    r"(全文|正文)?\s*$")

KIND_TO_STAT = {
    "年度": "{y}-12-31",
    "半年度": "{y}-06-30",
    "中期": "{y}-06-30",
    "第一季度": "{y}-03-31",
    "一季度": "{y}-03-31",
    "第三季度": "{y}-09-30",
    "三季度": "{y}-09-30",
}

METRICS = {
    "营业总收入": "MBRevenue",
    "净利润": "netProfit",
    "经营现金流量净额": "ocf_abs",
    "净资产收益率_平均": "roeAvg",
    "毛利率": "gpMargin",
}


def _ex_prefix(code: str) -> str:
    """sh.600069 → sh600069（新浪/巨潮都要 6 位纯数字 + 交易所前缀）。"""
    ex, num = code.split(".")
    return f"{ex}{num}"


def disclosure_dates(code6: str, start: str = "20180101", end: str = "20231231") -> dict[str, str]:
    """从巨潮公告列表解析 {statDate: 首次披露日}。"""
    import akshare as ak

    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code6, market="沪深京", start_date=start, end_date=end)
    except Exception:  # noqa: BLE001
        return {}
    if df is None or df.empty:
        return {}
    out: dict[str, str] = {}
    for title, t in zip(df["公告标题"].astype(str), df["公告时间"]):
        s = title.strip()
        if any(w in s for w in BAD_WORDS):
            continue
        m = TITLE_RE.search(s)
        if not m:
            continue
        stat = KIND_TO_STAT[m.group("kind")].format(y=m.group("y"))
        d = pd.to_datetime(t, errors="coerce")
        if pd.isna(d):
            continue
        ds = d.strftime("%Y-%m-%d")
        # 同一报告期可能有多条（不同交易所/板块重复），取最早那条 = 首次披露
        if stat not in out or ds < out[stat]:
            out[stat] = ds
    return out


def financial_values(code6: str) -> dict[str, dict]:
    """从新浪关键指标表取 {statDate: {字段: 值}}。"""
    import akshare as ak

    try:
        df = ak.stock_financial_abstract(symbol=code6)
    except Exception:  # noqa: BLE001
        return {}
    if df is None or df.empty:
        return {}
    periods = [c for c in df.columns if re.fullmatch(r"20\d{6}", str(c))]
    if not periods:
        return {}
    # 同一指标可能出现在多个「选项」分类下，只保留第一次出现的
    rows: dict[str, dict] = {p: {} for p in periods}
    seen: set[str] = set()
    for _, r in df.iterrows():
        name = str(r["指标"]).strip()
        if name not in METRICS or name in seen:
            continue
        seen.add(name)
        col = METRICS[name]
        for p in periods:
            v = pd.to_numeric(r[p], errors="coerce")
            if pd.notna(v):
                rows[p][col] = float(v)
    out: dict[str, dict] = {}
    for p, d in rows.items():
        if not d:
            continue
        stat = f"{p[:4]}-{p[4:6]}-{p[6:]}"
        out[stat] = d
    return out


def all_delisted_pairs(from_year: int = 2018, to_year: int = 2022) -> dict[str, list[str]]:
    """全部退市股 × 其在市期间内的报告期（不做「已覆盖」扣减）。

    为什么要全量而不用 missing_pairs：
        baostock 补的那批退市股里 MBRevenue 缺失率 55%，经营现金流只有比率没有绝对值，
        留着会污染后面的估值/现金流因子。巨潮+新浪能给出完整绝对值，所以让它整批覆盖，
        baostock 只在其覆盖不到的期次兜底（文件名排序决定谁优先）。
    """
    import sqlite3

    from .clean import DB

    dl = pd.read_csv(RAW_FIN.parent / "delisted.csv", dtype=str)
    dl = dl[pd.to_datetime(dl["delist_date"], errors="coerce") >= "2018-01-01"]
    pre = dl["exchange"].str.upper().map({"SH": "sh.", "SZ": "sz.", "BJ": "bj."}).fillna("sz.")
    codes = set(pre + dl["code"].str.zfill(6))

    con = sqlite3.connect(DB)
    pool = pd.read_sql("SELECT date, code FROM pool_daily", con)
    pool = pool[pool["code"].isin(codes)]
    days = sorted(pool["date"].unique())
    import bisect

    out: dict[str, list[str]] = {}
    period_dates = [f"{y}-{m}" for y in range(from_year, to_year + 1)
                    for m in ("03-31", "06-30", "09-30", "12-31")]
    by_code = {c: sorted(g["date"].unique()) for c, g in pool.groupby("code")}
    for c, ds in by_code.items():
        for p in period_dates:
            i = bisect.bisect_left(days, p)
            lo = days[max(0, i - 45)]
            hi = days[min(len(days) - 1, i + 45)]
            # 该股票在 [lo, hi] 内是否有行情（用二分判断，不做抽样，避免漏判）
            j = bisect.bisect_left(ds, lo)
            if j < len(ds) and ds[j] <= hi:
                out.setdefault(c, []).append(p)
    return {c: sorted(set(v)) for c, v in out.items()}


def _save_fail(failed: dict[str, dict], ok: list[str]) -> None:
    """把「试过但一条没取到」的股票逐只写进 _state，供 gaps.csv 的 finance_fetch_failed 使用。

    为什么必须落盘：三条底线之一是不静默跳过。抓不到就是抓不到，得留下凭据；
    反过来，后续补抓成功的要从名单里摘掉，否则台账会一直挂着已解决的旧账。
    """
    STATE.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict] = {}
    if FAIL_LOG.exists():
        try:
            items = json.loads(FAIL_LOG.read_text(encoding="utf-8")).get("items", {})
        except Exception as e:  # noqa: BLE001  坏文件不该阻断主流程，但要吼一声
            print(f"  [warn] 失败名单解析失败，按空表重写 {FAIL_LOG.name}: {e}")
    items.update(failed)
    for c in ok:
        items.pop(c, None)
    FAIL_LOG.write_text(
        json.dumps({"updated": TODAY, "items": items}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    if failed:
        print(f"[cninfo] 失败名单 {len(failed)} 只已写入 {FAIL_LOG.relative_to(STATE.parent.parent)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.4, help="每次外部请求后休眠秒数")
    ap.add_argument("--limit", type=int, default=None, help="只补前 N 只股票（调试）")
    ap.add_argument("--codes", default=None,
                    help="只处理指定代码，逗号分隔，如 sz.300089,sz.002680")
    ap.add_argument("--all-delisted", action="store_true",
                    help="覆盖全部退市股的全部期次（不只看缺口），用于替换 baostock 的残缺数据")
    ap.add_argument("--from-year", type=int, default=2018, help="报告期起始年（含）")
    ap.add_argument("--to-year", type=int, default=2022, help="报告期结束年（含）")
    args = ap.parse_args(argv)

    if args.all_delisted:
        by_code = all_delisted_pairs(args.from_year, args.to_year)
        miss_n = sum(len(v) for v in by_code.values())
    else:
        miss = missing_pairs()
        if not miss:
            print("[cninfo] 无缺口，无需补齐")
            return 0
        by_code = {}
        for c, stat in miss:
            by_code.setdefault(c, []).append(stat)
        miss_n = len(miss)
    codes = sorted(by_code)
    if args.codes:
        want = {c.strip() for c in args.codes.split(",") if c.strip()}
        codes = [c for c in codes if c in want]
    if args.limit:
        codes = codes[:args.limit]
    scope = "全部退市股" if args.all_delisted else "缺口"
    print(f"[cninfo] {scope} {miss_n} 个组合，涉及 {len(by_code)} 只股票；本次处理 {len(codes)} 只")

    rows: list[dict] = []
    no_pub: list[str] = []
    no_val: list[str] = []
    failed: dict[str, dict] = {}
    ok: list[str] = []
    t0 = time.time()
    for k, code in enumerate(codes, 1):
        c6 = code.split(".")[1]
        try:
            pubs = disclosure_dates(c6)
            time.sleep(args.sleep)
            vals = financial_values(c6)
            time.sleep(args.sleep)
        except Exception as e:  # noqa: BLE001  单只异常不该拖垮整批，但必须记账
            print(f"  {k}/{len(codes)} {code}: 请求异常 {type(e).__name__}: {e}", flush=True)
            failed[code] = {"from_year": args.from_year, "to_year": args.to_year,
                            "fetch_date": TODAY, "reason": f"request_error:{type(e).__name__}",
                            "periods": by_code[code]}
            continue
        got = 0
        for stat in by_code[code]:
            pub = pubs.get(stat)
            v = vals.get(stat, {})
            if not v:
                no_val.append(f"{code}|{stat}")
                continue
            rec = {"code": code, "statDate": stat, "pubDate": pub,
                   "source": "cninfo+sina", "fetch_date": TODAY}
            rec.update(v)
            rows.append(rec)
            got += 1
            if not pub:
                no_pub.append(f"{code}|{stat}")
        if got:
            ok.append(code)
        else:
            # 一条都没取到：源站确实没有这些期次（退市当年通常不再出定期报告），
            # 但必须逐只留档，不能静默当作"没缺口"
            failed[code] = {"from_year": args.from_year, "to_year": args.to_year,
                            "fetch_date": TODAY, "reason": "no_data_at_source",
                            "periods": by_code[code]}
        print(f"  {k}/{len(codes)} {code}: 补 {got} 期 "
              f"(公告日命中 {sum(1 for s in by_code[code] if s in pubs)})", flush=True)

    _save_fail(failed, ok)

    if not rows:
        print("[cninfo] 没有取到任何数据")
        return 1

    df = pd.DataFrame(rows)
    df["npMargin"] = (df["netProfit"] / df["MBRevenue"].where(df["MBRevenue"] != 0) * 100).round(4)
    df = df.sort_values(["code", "statDate"])
    out = RAW_FIN / (f"finance_cninfo_delisted_{args.from_year}_{args.to_year}_"
                     f"{TODAY}.parquet")
    df.to_parquet(out, index=False)

    print(f"\n[cninfo] 落盘 {out.name}: {len(df)} 行 / {df['code'].nunique()} 只")
    print(f"[cninfo] pubDate 缺失 {int(df['pubDate'].isna().sum())} 行（该期无定期报告公告或仅业绩预告）")
    bad = int((pd.to_datetime(df["pubDate"], errors="coerce")
               < pd.to_datetime(df["statDate"])).sum())
    print(f"[cninfo] pubDate < statDate 倒挂 {bad} 行")
    print(f"[cninfo] 无财务数值的期次 {len(no_val)} 个，无公告日的期次 {len(no_pub)} 个")
    print(f"[cninfo] 用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
