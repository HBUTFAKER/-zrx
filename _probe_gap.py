"""量化财务表对「当期在市股票」的覆盖率缺口（生存偏差体检）"""
import sqlite3
import pandas as pd

fin = pd.read_parquet("data/cache/finance.parquet")
dl = pd.read_csv("data/raw/delisted.csv", dtype=str)
dl["code"] = dl["exchange"].str.upper().map({"SH": "sh.", "SZ": "sz.", "BJ": "bj."}).fillna("sz.") + dl["code"].str.zfill(6)
dl_after = dl[pd.to_datetime(dl.delist_date, errors="coerce") >= "2018-01-01"]

con = sqlite3.connect("data/cache.db")
print(f"{'报告期':<12}{'在市股票':>9}{'财务覆盖':>9}{'缺口':>7}{'缺口=退市股':>11}")
tot_gap = 0
for stat in sorted(fin["statDate"].unique()):
    # 该报告期末前后最近的交易日，用 pool_daily 取当天在市股票
    rows = con.execute(
        "SELECT DISTINCT code FROM pool_daily WHERE date = "
        "(SELECT MAX(date) FROM pool_daily WHERE date <= ?)", (stat,)).fetchall()
    alive = {r[0] for r in rows}
    have = set(fin[fin.statDate == stat]["code"])
    gap = alive - have
    dl_hit = gap & set(dl_after["code"])
    tot_gap += len(gap)
    print(f"{stat:<12}{len(alive):>9}{len(have):>9}{len(gap):>7}{len(dl_hit):>11}")
print(f"\n累计缺口 {tot_gap} 个 (股票×期) 组合")

allmiss = set()
for stat in sorted(fin["statDate"].unique()):
    rows = con.execute("SELECT DISTINCT code FROM pool_daily WHERE date = "
                       "(SELECT MAX(date) FROM pool_daily WHERE date <= ?)", (stat,)).fetchall()
    allmiss |= ({r[0] for r in rows} - set(fin[fin.statDate == stat]["code"]))
print("完全不覆盖的股票数:", len(allmiss))
print("其中属 2018 后退市清单:", len(allmiss & set(dl_after["code"])))
print("非退市股缺口:", len(allmiss - set(dl_after["code"])), sorted(allmiss - set(dl_after["code"]))[:10])
con.close()
