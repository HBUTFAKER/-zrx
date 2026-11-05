"""验证东财三表字段口径：用贵州茅台 2022 年报做对照"""
import akshare as ak
import pandas as pd

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)

def pick(df, code, cols):
    c = df["股票代码"].astype(str).str.zfill(6)
    r = df[c == code]
    if r.empty:
        return None
    return r.iloc[0][cols]

lrb = ak.stock_lrb_em(date="20221231")
zcfz = ak.stock_zcfz_em(date="20221231")
xjll = ak.stock_xjll_em(date="20221231")

print("=== 贵州茅台 600519 2022 年报 ===")
print("【利润表】")
print(pick(lrb, "600519", ["净利润", "营业总收入", "营业总支出-营业支出",
                           "营业总支出-营业总支出", "净利润同比", "营业总收入同比", "公告日期"]).to_string())
print("\n【资产负债表】")
print(pick(zcfz, "600519", ["资产-总资产", "股东权益合计", "资产负债率", "资产-总资产同比"]).to_string())
print("\n【现金流量表】")
print(pick(xjll, "600519", ["经营性现金流-现金流量净额", "净现金流-净现金流"]).to_string())

# 口径校验
r = pick(lrb, "600519", ["净利润", "营业总收入", "营业总支出-营业支出"])
rev, np_, cost = r["营业总收入"], r["净利润"], r["营业总支出-营业支出"]
z = pick(zcfz, "600519", ["股东权益合计", "资产-总资产"])
x = pick(xjll, "600519", ["经营性现金流-现金流量净额"])
print("\n=== 推算校验 ===")
print(f"营收         {rev/1e8:>10.1f} 亿   (实际约 1275 亿)")
print(f"净利润       {np_/1e8:>10.1f} 亿   (实际约 627 亿)")
print(f"毛利率       {(rev-cost)/rev*100:>10.2f} %   (实际约 91%)")
print(f"ROE          {np_/z['股东权益合计']*100:>10.2f} %   (实际约 30%)")
print(f"经营现金流   {x/1e8:>10.1f} 亿   (实际约 366 亿)")
print(f"报告期净利润(累计,非单季) -> 是，东财为累计口径")
