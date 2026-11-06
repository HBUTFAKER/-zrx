"""验证新浪利润表的「公告日期」是首次披露日（而非最新修正日），并核对数值口径。"""
import warnings

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")

for code in ["sz300356", "sz002680", "sh600069"]:
    print(f"\n===== {code} 利润表 公告日期 =====", flush=True)
    df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
    d = df[["报告日", "营业总收入", "净利润", "归属于母公司所有者的净利润",
            "公告日期", "是否审计", "类型"]].copy()
    d = d[d["类型"].astype(str).str.contains("合并")]
    d = d[d["报告日"].astype(str).str[:4].between("2018", "2022")]
    d["滞后天数"] = (pd.to_datetime(d["公告日期"], format="%Y%m%d", errors="coerce")
                     - pd.to_datetime(d["报告日"], format="%Y%m%d", errors="coerce")).dt.days
    print(d.sort_values("报告日").to_string(index=False))

print("\n===== abstract 与 报表 数值一致性核对（300356 2022 年报）=====", flush=True)
a = ak.stock_financial_abstract(symbol="300356")
sub = a[a["指标"].isin(["营业总收入", "净利润", "经营现金流量净额", "净资产收益率_平均", "毛利率"])]
print(sub[["选项", "指标", "20221231"]].to_string(index=False))
