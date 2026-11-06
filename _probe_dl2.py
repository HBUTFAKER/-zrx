"""检查 stock_financial_abstract 的指标清单与关键字段，判断能否单次调用拿全。"""
import warnings

import akshare as ak

warnings.filterwarnings("ignore")
pd = __import__("pandas")
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

df = ak.stock_financial_abstract(symbol="300356")
print("shape", df.shape)
print("\n=== 选项 / 指标 全清单 ===")
for opt, g in df.groupby("选项", sort=False):
    print(f"[{opt}] " + " | ".join(g["指标"].tolist()))

print("\n=== 含 2018-2022 报告期的关键行 ===")
cols = [c for c in df.columns if c not in ("选项", "指标")]
want = [c for c in cols if c[:4] in {"2018", "2019", "2020", "2021", "2022"}]
key = df[df["指标"].astype(str).str.contains(
    "营业|净利润|净资产收益率|毛利率|现金流|权益|总资产", na=False)]
print(key[["选项", "指标"] + want[:6]].to_string(index=False))
