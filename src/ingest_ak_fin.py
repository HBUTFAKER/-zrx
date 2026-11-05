"""akshare 全市场财务抓取（东财三表 + 实际披露日）

为什么走这条路而不是 baostock
------------------------------
baostock 的季频财务只能「逐只 × 逐季」查：5565 只 × 20 期 × 4 接口 = 44.5 万次请求。
实测 5.5 次/秒持续约 25 分钟（8000 次）即触发 IP 级拉黑（10001011），
按安全频率把 44.5 万次跑完需要几十小时，且中途还会反复被封。

东财三个报表接口是「按报告期一次返回全市场」：20 个报告期 × 3 表 ≈ 10 分钟。

⚠ 已知缺陷 —— 必须用 baostock 补齐，否则数据不可用于回测
--------------------------------------------------------
东财财报只保留当前仍在上市的公司。实测 2018 年有行情的 3575 只股票里，
2018 年一季报只能查到 3580 只，缺 221 只，而这 221 只 **100% 是 2018 年后退市的股票**
（250 只 2018 年后退市股中，只剩 19 只还在表内）。
直接使用 = 生存偏差 = 用今天的存活公司回测历史，收益系统性虚高。
→ 退市股由 src.ingest 的 baostock 通道补齐（231 只 × 20 期 × 2 接口 ≈ 9240 次，可接受）。

数据纪律（对应框架 3.3）
------------------------
1. 只增不改：产物按抓取日期命名（finance_ak_{起}_{止}_{TODAY}.parquet），从不覆盖历史文件。
2. 缺失不插补：拿不到就留 NaN，由 clean.py 标注，绝不填 0 / 前值 / 均值。
3. 防未来函数：每条记录必须同时有 statDate（报告期）和 pubDate（实际披露日），
   pubDate 取「实际披露」而非东财「公告日期」（后者是最新修正公告日，茅台 2022 年报
   显示为 2024-04-03，明显不是首次披露）。pubDate < statDate 视为数据错误。
4. 口径可溯源：推算字段一律加 _derived 后缀并在 data/dict.md 标注公式，不冒充披露原值。
5. 不用今天的股票列表回测历史：本模块不做股票池，股票池由 clean.py 的 pool_daily 负责。
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FIN = ROOT / "data" / "raw" / "finance"
TODAY = date.today().isoformat()

# 报告期：季度末日期 <-> akshare 披露表口径
QUARTER_ENDS = [(3, 31, 1), (6, 30, 2), (9, 30, 3), (12, 31, 4)]
DISCLOSURE_PERIOD = {1: "一季", 2: "半年报", 3: "三季", 4: "年报"}


def _to_bs_code(s: pd.Series) -> pd.Series:
    """6 位代码 -> baostock 风格（sh./sz./bj.），与日线表保持同一套主键。"""
    c = s.astype(str).str.zfill(6)
    prefix = np.select(
        [c.str.startswith(("60", "68", "90")), c.str.startswith(("00", "30", "20")),
         c.str.startswith(("43", "83", "87", "92"))],
        ["sh.", "sz.", "bj."], default="sz.")
    return pd.Series(prefix, index=c.index) + c


def _num(s) -> pd.Series:
    """东财字段可能带 '-' 表示缺失，统一转数值并保留 NaN（不插补）。"""
    return pd.to_numeric(s, errors="coerce")


def fetch_period(year: int, month: int, day: int, quarter: int) -> pd.DataFrame | None:
    """抓一个报告期的三表 + 实际披露日，组装成与 baostock 同 schema 的表。"""
    d = f"{year}{month:02d}{day:02d}"
    stat = f"{year}-{month:02d}-{day:02d}"
    try:
        lrb = ak.stock_lrb_em(date=d)
        zcfz = ak.stock_zcfz_em(date=d)
        xjll = ak.stock_xjll_em(date=d)
        dis = ak.stock_report_disclosure(market="沪深京",
                                         period=f"{year}{DISCLOSURE_PERIOD[quarter]}")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {stat} 抓取失败: {type(e).__name__} {e}")
        return None

    for df in (lrb, zcfz, xjll, dis):
        df["_c6"] = df["股票代码"].astype(str).str.zfill(6)

    base = lrb[["_c6", "净利润", "营业总收入", "营业总支出-营业支出",
                "净利润同比", "营业总收入同比"]].copy()
    base = base.rename(columns={"净利润": "netProfit", "营业总收入": "MBRevenue",
                                "营业总支出-营业支出": "op_cost",
                                "净利润同比": "YOYNI", "营业总收入同比": "YOYRevenue"})
    base = base.merge(zcfz[["_c6", "股东权益合计", "资产-总资产", "资产负债率", "资产-总资产同比"]]
                      .rename(columns={"股东权益合计": "total_equity", "资产-总资产": "total_asset",
                                       "资产负债率": "debt_ratio", "资产-总资产同比": "YOYAsset"}),
                      on="_c6", how="left")
    base = base.merge(xjll[["_c6", "经营性现金流-现金流量净额"]]
                      .rename(columns={"经营性现金流-现金流量净额": "ocf_abs"}),
                      on="_c6", how="left")
    # 实际披露日：这是防未来函数的命根子，取不到就留 NaN，绝不回退到「公告日期」
    base = base.merge(dis[["_c6", "实际披露"]].rename(columns={"实际披露": "pubDate"}),
                      on="_c6", how="left")

    for c in ("netProfit", "MBRevenue", "op_cost", "YOYNI", "YOYRevenue",
              "total_equity", "total_asset", "debt_ratio", "YOYAsset", "ocf_abs"):
        base[c] = _num(base[c])

    base["code"] = _to_bs_code(base["_c6"])
    base["statDate"] = stat
    base["year"] = np.int16(year)
    base["quarter"] = np.int8(quarter)
    base["pubDate"] = pd.to_datetime(base["pubDate"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ---- 推算字段（_derived）：一律标注，不冒充披露原值
    eq = base["total_equity"].where(base["total_equity"] != 0)
    rev = base["MBRevenue"].where(base["MBRevenue"] != 0)
    base["roeAvg_derived"] = base["netProfit"] / eq * 100          # 报告期 ROE（累计口径）
    base["npMargin_derived"] = base["netProfit"] / rev * 100        # 净利率
    base["gpMargin_derived"] = (base["MBRevenue"] - base["op_cost"]) / rev * 100  # 毛利率
    base["CFOToOR_derived"] = base["ocf_abs"] / rev * 100           # 经营现金流/营收
    base["CFOToNP_derived"] = base["ocf_abs"] / base["netProfit"].where(base["netProfit"] != 0) * 100

    base["source"] = "akshare_em"
    base["fetch_date"] = TODAY
    return base.drop(columns=["_c6", "op_cost", "YOYRevenue"])


def ingest_ak_finance(year_from: int = 2018, year_to: int = 2022) -> Path | None:
    """按报告期循环抓取，产物按日期命名，只增不改。"""
    RAW_FIN.mkdir(parents=True, exist_ok=True)
    out = RAW_FIN / f"finance_ak_{year_from}_{year_to}_{TODAY}.parquet"
    if out.exists():
        print(f"[ak-fin] 已存在，跳过（只增不改）：{out.name}")
        return out

    frames = []
    for y in range(year_from, year_to + 1):
        for m, d, q in QUARTER_ENDS:
            t0 = time.time()
            df = fetch_period(y, m, d, q)
            if df is None:
                continue
            # 披露日晚于报告期的记录占比：必须 100%，否则说明主键或字段对错了
            ok = df["pubDate"].notna()
            inverted = (pd.to_datetime(df["pubDate"], errors="coerce")
                        < pd.to_datetime(df["statDate"])).sum()
            print(f"  [ak-fin] {df['statDate'].iloc[0]}  行 {len(df):5d}  "
                  f"有披露日 {ok.sum():5d} ({ok.mean()*100:5.1f}%)  倒挂 {inverted}  "
                  f"{time.time()-t0:.1f}s")
            frames.append(df)
            time.sleep(0.3)  # 不轰炸东财，80 次调用不值得冒险

    if not frames:
        print("[ak-fin] 无数据产出")
        return None
    allf = pd.concat(frames, ignore_index=True)
    allf = allf.sort_values(["code", "statDate"]).reset_index(drop=True)
    allf.to_parquet(out, index=False)
    print(f"[ak-fin] 写入 {out.relative_to(ROOT)}  行 {len(allf)}  股票 {allf['code'].nunique()}  "
          f"期数 {allf['statDate'].nunique()}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="akshare 全市场财务抓取（东财三表 + 实际披露日）")
    ap.add_argument("--from-year", type=int, default=2018)
    ap.add_argument("--to-year", type=int, default=2022)
    a = ap.parse_args()
    ingest_ak_finance(a.from_year, a.to_year)
