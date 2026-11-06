"""看巨潮公告标题的实际命名，定位「定期报告」披露日。"""
import warnings

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 60)

for sym in ["300356", "002680"]:
    df = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=sym, market="沪深京", start_date="20180101", end_date="20230630")
    t = df["公告标题"].astype(str)
    pat = "季度报告|半年度报告|年度报告|年报|第一季|第三季|中期报告"
    hit = df[t.str.contains(pat, na=False)]
    print(f"\n===== {sym} 命中 {len(hit)} / {len(df)} =====")
    print(hit.head(20).to_string(index=False))
