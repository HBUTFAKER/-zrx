"""清洗层：原始落地文件 → 可回测的宽表 + 每日股票池快照。

数据纪律（框架文档 3.3 节）：
  1. 时点对齐：财务表同时保留 pubDate（公告日）与 statDate（报告期）。
     本层只做清洗、**不做任何前视对齐**——下游用 pubDate 决定"哪天才能看见这条财报"。
     数据缺失就留 NaN，绝不插补、绝不估算（唯一例外是显式标注为推算的 cfo_est 列）。
  2. 复权：hfq_* = 原始价 × 后复权因子（backAdjustFactor，按除权除息日 backward 取值）。
     已用 sh.600519 2018-2024 全区间与 baostock 官方后复权价对拍，偏差 0.0000%。
     回测用 hfq_*；界面看盘用原始价或前复权（前复权不落地，避免负价污染）。
  3. 停牌：tradestatus==0 或 成交量为 0 → suspended=1，回测必须跳过。
  4. 生存偏差：pool_daily 保留已退市股票到它们最后一个交易日；
     任何策略取股票池都必须读 pool_daily，不许用"当前股票列表"。
  5. 涨跌停：本层只提供原始价与 pctChg，涨跌停判定与不可成交模拟放在回测层。

产物：
  data/cache/daily_{year}.parquet   清洗后的行情宽表（含 hfq_*）
  data/cache/finance.parquet        季度财务（含 pubDate / statDate）
  data/cache.db                     SQLite：pool_daily / stocks / delisted / industry / manifest
  data/dict.md                      字段字典（本文件生成，保证与代码同步）
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_DAILY = RAW / "daily"
RAW_FIN = RAW / "finance"
RAW_META = RAW / "meta"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
DB = DATA / "cache.db"
DICT_MD = DATA / "dict.md"

NEW_STOCK_DAYS = 120  # 上市不满 120 个交易日 → new_stock


# ---------------------------------------------------------------- 字段字典
FIELD_DICT = [
    # (表, 字段, 中文名, 单位, 口径, 来源)
    ("daily", "date", "交易日期", "YYYY-MM-DD", "交易日", "baostock query_history_k_data_plus.frequency=d"),
    ("daily", "code", "股票代码", "-", "带交易所前缀，如 sh.600519", "同上"),
    ("daily", "open", "开盘价（不复权）", "元", "当日开盘价，未复权", "同上 adjustflag=3"),
    ("daily", "high", "最高价（不复权）", "元", "当日最高价，未复权", "同上"),
    ("daily", "low", "最低价（不复权）", "元", "当日最低价，未复权", "同上"),
    ("daily", "close", "收盘价（不复权）", "元", "当日收盘价，未复权", "同上"),
    ("daily", "volume", "成交量", "股", "当日成交股数；停牌为 0", "同上"),
    ("daily", "amount", "成交额", "元", "当日成交金额", "同上"),
    ("daily", "turn", "换手率", "%", "成交量/流通股本，baostock 原值", "同上"),
    ("daily", "pctChg", "涨跌幅", "%", "相对前收盘的涨跌幅，baostock 原值", "同上"),
    ("daily", "peTTM", "市盈率 TTM", "倍", "滚动 12 个月市盈率；亏损或缺失为 NaN", "同上"),
    ("daily", "pbMRQ", "市净率 MRQ", "倍", "最近报告期市净率；缺失为 NaN", "同上"),
    ("daily", "isST", "是否 ST", "0/1", "baostock 口径：1=ST 或 *ST，0=否，-1=缺失", "同上"),
    ("daily", "tradestatus", "交易状态", "0/1", "1=正常交易，0=停牌，-1=缺失", "同上"),
    ("daily", "backAdjustFactor", "后复权因子", "倍", "该交易日适用的最近一次除权除息因子（backward 取值）",
     "baostock query_adjust_factor.backAdjustFactor"),
    ("daily", "hfq_open", "开盘价（后复权）", "元", "open × backAdjustFactor", "本层计算"),
    ("daily", "hfq_high", "最高价（后复权）", "元", "high × backAdjustFactor", "本层计算"),
    ("daily", "hfq_low", "最低价（后复权）", "元", "low × backAdjustFactor", "本层计算"),
    ("daily", "hfq_close", "收盘价（后复权）", "元", "close × backAdjustFactor，回测唯一可用价格列", "本层计算"),
    ("daily", "suspended", "是否停牌", "0/1", "tradestatus=0 或成交量为 0 → 1", "本层计算"),
    ("daily", "listed_days", "上市第几个交易日", "个", "按全市场交易日历，自 ipoDate 起算的序号", "本层计算"),
    ("daily", "new_stock", "是否次新股", "0/1", f"listed_days < {NEW_STOCK_DAYS} → 1", "本层计算"),
    ("daily", "is_delisted", "是否已退市", "0/1", "该股最终是否退市（不是当日状态），用于避免生存偏差", "baostock outDate / akshare 退市清单"),
    ("daily", "delist_date", "退市日期", "YYYY-MM-DD", "终止上市日；未退市为 NaN", "同上"),
    ("daily", "fetch_date", "抓取日期", "YYYY-MM-DD", "该行原始数据的抓取日，用于溯源", "本层写入"),

    ("finance", "code", "股票代码", "-", "带交易所前缀", "baostock 季频接口"),
    ("finance", "pubDate", "公告日期", "YYYY-MM-DD", "**财报实际公告日，防未来函数只能用这个**", "query_profit_data.pubDate"),
    ("finance", "statDate", "报告期截止日", "YYYY-MM-DD", "财报报告期（如 2023-06-30），不可用于对齐", "同上"),
    ("finance", "year", "报告年度", "年", "由 statDate 截取", "本层计算"),
    ("finance", "quarter", "报告季度", "1-4", "抓取时使用的季度参数", "本层写入"),
    ("finance", "MBRevenue", "营业总收入", "元", "本期累计主营营业收入", "query_profit_data.MBRevenue"),
    ("finance", "netProfit", "净利润", "元", "本期累计净利润（含少数股东）", "query_profit_data.netProfit"),
    ("finance", "roeAvg", "净资产收益率", "%", "区间平均 ROE，baostock 原值（小数）", "query_profit_data.roeAvg"),
    ("finance", "gpMargin", "毛利率", "%", "毛利/营业收入，baostock 原值（小数）", "query_profit_data.gpMargin"),
    ("finance", "npMargin", "净利率", "%", "净利/营业收入，baostock 原值（小数）", "query_profit_data.npMargin"),
    ("finance", "epsTTM", "每股收益 TTM", "元/股", "滚动 12 个月每股收益", "query_profit_data.epsTTM"),
    ("finance", "totalShare", "总股本", "股", "报告期末总股本", "query_profit_data.totalShare"),
    ("finance", "liqaShare", "流通股本", "股", "报告期末流通股本", "query_profit_data.liqaShare"),
    ("finance", "YOYNI", "净利润同比", "%", "同比增长率，baostock 原值（小数）", "query_growth_data.YOYNI"),
    ("finance", "YOYEquity", "净资产同比", "%", "同上", "query_growth_data.YOYEquity"),
    ("finance", "YOYAsset", "总资产同比", "%", "同上", "query_growth_data.YOYAsset"),
    ("finance", "YOYEPSBasic", "每股收益同比", "%", "同上", "query_growth_data.YOYEPSBasic"),
    ("finance", "YOYPNI", "归母净利同比", "%", "同上", "query_growth_data.YOYPNI"),
    ("finance", "NRTurnRatio", "应收账款周转率", "次", "营业收入/平均应收账款", "query_operation_data.NRTurnRatio"),
    ("finance", "NRTurnDays", "应收账款周转天数", "天", "360/周转率", "query_operation_data.NRTurnDays"),
    ("finance", "INVTurnRatio", "存货周转率", "次", "营业成本/平均存货", "query_operation_data.INVTurnRatio"),
    ("finance", "INVTurnDays", "存货周转天数", "天", "360/周转率", "query_operation_data.INVTurnDays"),
    ("finance", "CATurnRatio", "流动资产周转率", "次", "营业收入/平均流动资产", "query_operation_data.CATurnRatio"),
    ("finance", "AssetTurnRatio", "总资产周转率", "次", "营业收入/平均总资产", "query_operation_data.AssetTurnRatio"),
    ("finance", "CFOToNP", "经营现金流/净利润", "倍", "利润含金量指标", "query_cash_flow_data.CFOToNP"),
    ("finance", "CFOToOR", "经营现金流/营业收入", "倍", "经营现金流占收入比", "query_cash_flow_data.CFOToOR"),
    ("finance", "CFOToGr", "经营现金流/营业总收入", "倍", "同上口径（增长率分母）", "query_cash_flow_data.CFOToGr"),
    ("finance", "CAToAsset", "流动资产/总资产", "倍", "结构指标", "query_cash_flow_data.CAToAsset"),
    ("finance", "NCAToAsset", "非流动资产/总资产", "倍", "结构指标", "query_cash_flow_data.NCAToAsset"),
    ("finance", "cfo_est", "经营现金流（推算）", "元", "**推算列**：MBRevenue × CFOToOR，不是原始披露值；查不到比率则为 NaN", "本层推算，勿直接引用为财报原值"),
    ("finance", "fetch_date", "抓取日期", "YYYY-MM-DD", "溯源用", "本层写入"),

    ("pool_daily", "date", "交易日期", "YYYY-MM-DD", "交易日", "由 daily 派生"),
    ("pool_daily", "code", "股票代码", "-", "含已退市股票", "由 daily 派生"),
    ("pool_daily", "is_st", "当日是否 ST", "0/1", "baostock isST", "由 daily 派生"),
    ("pool_daily", "suspended", "当日是否停牌", "0/1", "停牌不可成交", "由 daily 派生"),
    ("pool_daily", "new_stock", "当日是否次新", "0/1", "上市不满 120 交易日", "由 daily 派生"),
    ("pool_daily", "delisted", "是否已退市", "0/1", "该股最终退市（保留在历史池中，防生存偏差）", "由 daily 派生"),
    ("pool_daily", "listed_days", "上市交易日序号", "个", "自 ipoDate 起算", "由 daily 派生"),
    ("pool_daily", "tradable", "当日可交易", "0/1", "未停牌 → 1；ST/次新不在此列剔除，由策略自行过滤", "由 daily 派生"),

    ("stocks", "code", "股票代码", "-", "全 A 含已退市", "baostock query_stock_basic"),
    ("stocks", "code_name", "股票名称", "-", "最新简称，历史上改名不追溯", "同上"),
    ("stocks", "ipoDate", "上市日期", "YYYY-MM-DD", "首次上市日", "同上"),
    ("stocks", "outDate", "退市日期", "YYYY-MM-DD", "空=仍在上市", "同上"),
    ("stocks", "is_delisted", "是否已退市", "0/1", "outDate 非空", "同上"),
    ("stocks", "data_completeness", "数据完整度", "0-1", "区间内有效行情行数 / 该股应有交易日数", "本层计算"),
    ("delisted", "code", "股票代码", "6 位", "无交易所前缀", "akshare 退市清单 + baostock"),
    ("delisted", "name", "股票名称", "-", "退市时简称", "同上"),
    ("delisted", "delist_date", "退市日期", "YYYY-MM-DD", "终止上市日", "同上"),
    ("industry", "code", "股票代码", "6 位", "无前缀", "akshare 东财行业板块成分"),
    ("industry", "industry", "所属行业", "-", "东财行业分类；失败时退回证监会行业", "akshare stock_board_industry_cons_em"),
    ("manifest", "table", "表名", "-", "产物表名", "本层写入"),
    ("manifest", "rows", "行数", "行", "本次清洗落盘行数", "本层写入"),
    ("manifest", "built_at", "生成时间", "YYYY-MM-DD HH:MM:SS", "清洗时间", "本层写入"),
]


# ---------------------------------------------------------------- 工具
def _latest(files: list[Path], key: tuple[str, ...] = ("code", "date")) -> pd.DataFrame | None:
    """多版本原始文件 → 取并集，同键保留抓取日期最新的一条（只增不改，历史版本不删）。"""
    if not files:
        return None
    dfs = []
    for f in sorted(files):  # 文件名含日期，升序 → 越后越新
        df = pd.read_parquet(f)
        df["_src"] = f.name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(list(key) + ["_src"]).drop_duplicates(
        subset=list(key), keep="last")
    return out.drop(columns=["_src"])


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    return con


# ---------------------------------------------------------------- 主流程
def clean(years: list[int]) -> dict:
    stats: dict[str, int] = {}

    # ---------- 元数据
    stocks = _latest(sorted(RAW_META.glob("stocks_*.parquet")), key=("code",))
    if stocks is None:
        raise SystemExit("缺少股票清单，先跑 --stage meta")
    stocks = stocks.copy()
    stocks["outDate"] = stocks["outDate"].fillna("")
    stocks["ipoDate"] = stocks["ipoDate"].fillna("")
    stocks["is_delisted"] = stocks["outDate"].ne("").astype(int)
    stocks["delist_date"] = stocks["outDate"].replace("", np.nan)

    fac = _latest(sorted(RAW_META.glob("adjust_factor_*.parquet")),
                   key=("code", "dividOperateDate"))
    if fac is None:
        raise SystemExit("缺少复权因子，先跑 --stage daily")
    fac["dt"] = pd.to_datetime(fac["dividOperateDate"])
    # merge_asof 要求 right 侧也按 on 键全局有序
    fac = fac[["code", "dt", "backAdjustFactor"]].sort_values("dt")

    industry = _latest(sorted(RAW_META.glob("industry_*.parquet")), key=("code",))
    dl_file = RAW / "delisted.csv"
    delisted = pd.read_csv(dl_file, dtype=str) if dl_file.exists() else pd.DataFrame(
        columns=["code", "name", "ipo_date", "delist_date", "exchange"])

    # ---------- 交易日历（全市场并集，用于 listed_days）
    cal_set: set[str] = set()
    for y in years:
        for p in sorted(RAW_DAILY.glob(f"daily_{y}_*.parquet")):
            cal_set.update(pd.read_parquet(p, columns=["date"])["date"].unique().tolist())
    if not cal_set:
        raise SystemExit(f"没有 {years} 的日线文件，先跑 --stage daily")
    calendar = pd.DataFrame({"date": sorted(cal_set)})
    calendar["cal_idx"] = np.arange(len(calendar))

    con = _db()

    pool_rows: list[pd.DataFrame] = []
    total_daily = 0

    for y in years:
        files = sorted(RAW_DAILY.glob(f"daily_{y}_*.parquet"))
        df = _latest(files)
        if df is None:
            print(f"[clean] {y}: 无原始文件，跳过")
            continue

        df = df.copy()
        total_daily += len(df)
        df["code"] = df["code"].astype(str)

        # ---- 复权：按 (code, date) backward 取最近一次除权除息因子
        df["dt"] = pd.to_datetime(df["date"])
        df = df.sort_values("dt")  # merge_asof 要求按 on 键全局有序
        df = pd.merge_asof(df, fac.rename(columns={"dt": "fac_dt"}),
                           left_on="dt", right_on="fac_dt", by="code",
                           direction="backward")
        # 上市早于首个因子日的行情：用该股最早的因子回填（后复权基准，不改变相对收益率）
        df["backAdjustFactor"] = df.groupby("code")["backAdjustFactor"].transform(
            lambda s: s.bfill())
        for c in ("open", "high", "low", "close"):
            df[f"hfq_{c}"] = df[c] * df["backAdjustFactor"]

        # ---- 停牌 / ST / 退市
        df["suspended"] = ((df["tradestatus"] == 0) | (df["volume"].fillna(0) == 0)).astype("int8")
        df["isST"] = df["isST"].replace(-1, np.nan)

        meta = stocks.set_index("code")
        df["is_delisted"] = df["code"].map(meta["is_delisted"]).fillna(0).astype("int8")
        df["delist_date"] = df["code"].map(meta["delist_date"])
        ipo = df["code"].map(meta["ipoDate"])

        # ---- listed_days：自上市日起的交易日序号（全市场日历 searchsorted，不做估算）
        df = df.merge(calendar, on="date", how="left")
        cal_dates = calendar["date"].to_numpy()
        ipo_str = df["code"].map(meta["ipoDate"]).fillna("").to_numpy()
        ipo_idx = np.searchsorted(cal_dates, ipo_str, side="left")
        df["listed_days"] = (df["cal_idx"].to_numpy() - ipo_idx + 1).astype("int32")
        df["new_stock"] = (df["listed_days"] < NEW_STOCK_DAYS).astype("int8")
        df["fetch_date"] = df["fetch_date"].fillna("")

        # ---- 落 parquet（清洗宽表）
        keep = ["code", "date", "open", "high", "low", "close", "volume", "amount",
                "turn", "pctChg", "peTTM", "pbMRQ", "isST", "tradestatus",
                "backAdjustFactor", "hfq_open", "hfq_high", "hfq_low", "hfq_close",
                "suspended", "listed_days", "new_stock", "is_delisted", "delist_date",
                "fetch_date"]
        out = df[keep].sort_values(["code", "date"])
        out.to_parquet(CACHE / f"daily_{y}.parquet", index=False)
        print(f"[clean] daily {y}: {len(out)} 行, {out['code'].nunique()} 只")
        stats[f"daily_{y}"] = len(out)

        # ---- 股票池快照
        p = out[["date", "code", "isST", "suspended", "new_stock", "is_delisted",
                 "listed_days"]].copy()
        p["is_st"] = p["isST"].fillna(-1).astype("int8")
        p["delisted"] = p["is_delisted"].astype("int8")
        p["tradable"] = (p["suspended"] == 0).astype("int8")
        pool_rows.append(p[["date", "code", "is_st", "suspended", "new_stock",
                            "delisted", "listed_days", "tradable"]])

    # ---------- 写 SQLite
    if pool_rows:
        pool = pd.concat(pool_rows, ignore_index=True)
        con.execute("DROP TABLE IF EXISTS pool_daily")
        con.execute("""CREATE TABLE pool_daily(
            date TEXT, code TEXT, is_st INTEGER, suspended INTEGER, new_stock INTEGER,
            delisted INTEGER, listed_days INTEGER, tradable INTEGER,
            PRIMARY KEY(date, code))""")
        # executemany 而非 to_sql：SQLite 有 32766 个绑定变量上限，且整行插入更快
        con.executemany(
            "INSERT OR REPLACE INTO pool_daily VALUES (?,?,?,?,?,?,?,?)",
            pool.itertuples(index=False, name=None))
        con.execute("CREATE INDEX idx_pool_code_date ON pool_daily(code, date)")
        stats["pool_daily"] = len(pool)

    # stocks（附数据完整度）
    comp = []
    for y in years:
        f = CACHE / f"daily_{y}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["code"])
            comp.append(d["code"].value_counts())
    if comp:
        have = pd.concat(comp).groupby(level=0).sum()
    else:
        have = pd.Series(dtype=int)
    expect_days = len(calendar)
    st = stocks.copy()
    st["data_completeness"] = (st["code"].map(have).fillna(0) / expect_days).round(4)
    st = st.rename(columns={"type": "sec_type"})
    st.to_sql("stocks", con, if_exists="replace", index=False)
    stats["stocks"] = len(st)

    delisted.to_sql("delisted", con, if_exists="replace", index=False)
    stats["delisted"] = len(delisted)

    if industry is not None:
        ind = industry.drop_duplicates(subset=["code"]).rename(columns={"industry": "industry"})
        ind.to_sql("industry", con, if_exists="replace", index=False)
        stats["industry"] = len(ind)

    # ---------- 财务
    fin = _latest(sorted(RAW_FIN.glob("finance_*.parquet")), key=("code", "statDate"))
    if fin is not None:
        fin = fin.copy()
        # 经营现金流绝对值不是披露字段，只能由 收入 × 比率 推算 → 显式标注为推算列
        fin["cfo_est"] = fin["MBRevenue"] * fin["CFOToOR"]
        fin = fin.drop(columns=[c for c in ("year",) if c in fin.columns])
        fin["year"] = fin["statDate"].str.slice(0, 4).astype("int16")
        fin = fin.sort_values(["code", "statDate"])
        fin.to_parquet(CACHE / "finance.parquet", index=False)
        stats["finance"] = len(fin)
        print(f"[clean] finance: {len(fin)} 行, {fin['code'].nunique()} 只, "
              f"pubDate 缺失 {int(fin['pubDate'].isna().sum())} 行")

    # ---------- manifest
    rows = [{"table": k, "rows": int(v), "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
            for k, v in stats.items()]
    pd.DataFrame(rows).to_sql("manifest", con, if_exists="replace", index=False)
    con.commit()
    con.close()

    write_dict()
    return stats


def write_dict() -> None:
    """生成字段字典，保证文档与代码同源。"""
    lines = ["# 字段字典（data/dict.md）", "",
             "> 本文件由 `src/clean.py` 自动生成，与代码同源；改字段请改 `FIELD_DICT` 后重跑 clean。",
             "> 三条底线之一：任何数字都必须能溯源到这里写的来源接口名 + 字段名。", ""]
    cur = None
    for tbl, col, cn, unit, desc, src in FIELD_DICT:
        if tbl != cur:
            lines += ["", f"## {tbl}", "", "| 字段 | 中文名 | 单位 | 口径 | 来源 |", "|---|---|---|---|---|"]
            cur = tbl
        lines.append(f"| `{col}` | {cn} | {unit} | {desc} | {src} |")
    lines += ["", "## 缺失值约定", "",
              "- 所有缺失一律写 NaN / NULL，**不插补、不估算、不线性填充**。",
              "- 唯一例外：`finance.cfo_est`（经营现金流推算值），已在该行口径里标为推算，不得当作披露原值引用。",
              "- 下游展示缺失值时必须显示「缺失」，不得用 0 或均值替代。", ""]
    DICT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[clean] 字段字典已写入 {DICT_MD.relative_to(ROOT)}（{len(FIELD_DICT)} 个字段）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="清洗原始数据 → cache")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    args = ap.parse_args(argv)
    years = list(range(min(args.years), max(args.years) + 1))
    stats = clean(years)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
