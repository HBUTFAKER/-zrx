"""探测巨潮公告接口能否给出退市股的「定期报告」真实披露日。"""
import warnings

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

for sym in ["300356", "002680", "600069"]:
    print(f"\n===== {sym} stock_zh_a_disclosure_report_cninfo =====", flush=True)
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=sym, market="沪深京", start_date="20180101", end_date="20230630")
        print("shape", df.shape, "cols", list(df.columns))
        key = df[df.iloc[:, 1].astype(str).str.contains("季度报告|半年度报告|年度报告|年报", na=False)]
        print(key.head(14).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print("EXC", type(e).__name__, str(e)[:150])
