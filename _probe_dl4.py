"""查 stock_report_disclosure 是否覆盖退市股（若覆盖，则 pubDate 有真值来源）。"""
import warnings

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")

CODES = ["300356", "002680", "600069", "300089", "300367", "600145"]

for period in ["2022年报", "2022一季", "2022半年", "2022三季"]:
    try:
        df = ak.stock_report_disclosure(market="沪深京", period=period)
    except Exception as e:  # noqa: BLE001
        print(period, "EXC", e)
        continue
    col = "股票代码" if "股票代码" in df.columns else df.columns[0]
    codes = set(df[col].astype(str).str.zfill(6))
    print(f"\n=== {period} 共 {len(df)} 行, 列={list(df.columns)} ===")
    for c in CODES:
        hit = df[df[col].astype(str).str.zfill(6) == c]
        if len(hit):
            print(f"  {c}: {hit.iloc[0].to_dict()}")
        else:
            print(f"  {c}: 不在表中")
