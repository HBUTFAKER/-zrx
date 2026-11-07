# 字段字典（data/dict.md）

> 本文件由 `src/clean.py` 自动生成，与代码同源；改字段请改 `FIELD_DICT` 后重跑 clean。
> 三条底线之一：任何数字都必须能溯源到这里写的来源接口名 + 字段名。


## daily

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `date` | 交易日期 | YYYY-MM-DD | 交易日 | baostock query_history_k_data_plus.frequency=d |
| `code` | 股票代码 | - | 带交易所前缀，如 sh.600519 | 同上 |
| `open` | 开盘价（不复权） | 元 | 当日开盘价，未复权 | 同上 adjustflag=3 |
| `high` | 最高价（不复权） | 元 | 当日最高价，未复权 | 同上 |
| `low` | 最低价（不复权） | 元 | 当日最低价，未复权 | 同上 |
| `close` | 收盘价（不复权） | 元 | 当日收盘价，未复权 | 同上 |
| `volume` | 成交量 | 股 | 当日成交股数；停牌为 0 | 同上 |
| `amount` | 成交额 | 元 | 当日成交金额 | 同上 |
| `turn` | 换手率 | % | 成交量/流通股本，baostock 原值 | 同上 |
| `pctChg` | 涨跌幅 | % | 相对前收盘的涨跌幅，baostock 原值 | 同上 |
| `peTTM` | 市盈率 TTM | 倍 | 滚动 12 个月市盈率；亏损或缺失为 NaN | 同上 |
| `pbMRQ` | 市净率 MRQ | 倍 | 最近报告期市净率；缺失为 NaN | 同上 |
| `isST` | 是否 ST | 0/1 | baostock 口径：1=ST 或 *ST，0=否，-1=缺失 | 同上 |
| `tradestatus` | 交易状态 | 0/1 | 1=正常交易，0=停牌，-1=缺失 | 同上 |
| `backAdjustFactor` | 后复权因子 | 倍 | 该交易日适用的最近一次除权除息因子（backward 取值） | baostock query_adjust_factor.backAdjustFactor |
| `hfq_open` | 开盘价（后复权） | 元 | open × backAdjustFactor | 本层计算 |
| `hfq_high` | 最高价（后复权） | 元 | high × backAdjustFactor | 本层计算 |
| `hfq_low` | 最低价（后复权） | 元 | low × backAdjustFactor | 本层计算 |
| `hfq_close` | 收盘价（后复权） | 元 | close × backAdjustFactor，回测唯一可用价格列 | 本层计算 |
| `suspended` | 是否停牌 | 0/1 | tradestatus=0 或成交量为 0 → 1 | 本层计算 |
| `listed_days` | 上市第几个交易日 | 个 | 按全市场交易日历，自 ipoDate 起算的序号 | 本层计算 |
| `new_stock` | 是否次新股 | 0/1 | listed_days < 120 → 1 | 本层计算 |
| `is_delisted` | 是否已退市 | 0/1 | 该股最终是否退市（不是当日状态），用于避免生存偏差 | baostock outDate / akshare 退市清单 |
| `delist_date` | 退市日期 | YYYY-MM-DD | 终止上市日；未退市为 NaN | 同上 |
| `fetch_date` | 抓取日期 | YYYY-MM-DD | 该行原始数据的抓取日，用于溯源 | 本层写入 |

## finance

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `code` | 股票代码 | - | 带交易所前缀 | baostock 季频接口 |
| `pubDate` | 实际披露日 | YYYY-MM-DD | **防未来函数只能用这个**。东财源取「实际披露」（stock_report_disclosure），巨潮源取公告原文列表的首次披露日，baostock 源取原始 pubDate；缺失保留 NaN，绝不回退到东财/新浪的「公告日期」（那是最新修正公告日，实测会把 2018 一季报标成 2019-04-27） | stock_report_disclosure.实际披露 / cninfo 公告列表 / query_profit_data.pubDate |
| `statDate` | 报告期截止日 | YYYY-MM-DD | 财报报告期（如 2023-06-30），不可用于对齐 | 同上 |
| `year` | 报告年度 | 年 | 由 statDate 截取 | 本层计算 |
| `quarter` | 报告季度 | 1-4 | 抓取时使用的季度参数 | 本层写入 |
| `MBRevenue` | 营业总收入 | 元 | 本期累计主营营业收入 | query_profit_data.MBRevenue |
| `netProfit` | 净利润 | 元 | 本期累计净利润（含少数股东） | query_profit_data.netProfit |
| `roeAvg` | 净资产收益率 | % | 区间平均 ROE，**已统一为百分数**：baostock 原值是小数(0.184=18.4%)，入层时已 ×100；东财/新浪本就是百分数 | query_profit_data.roeAvg×100 / 新浪「净资产收益率_平均」 |
| `gpMargin` | 毛利率 | % | 毛利/营业收入，**已统一为百分数**（同 roeAvg 处理） | query_profit_data.gpMargin×100 / 新浪「毛利率」 |
| `npMargin` | 净利率 | % | 净利/营业收入，**已统一为百分数**（同 roeAvg 处理） | query_profit_data.npMargin×100 / 本层推算 |
| `epsTTM` | 每股收益 TTM | 元/股 | 滚动 12 个月每股收益 | query_profit_data.epsTTM |
| `totalShare` | 总股本 | 股 | 报告期末总股本 | query_profit_data.totalShare |
| `liqaShare` | 流通股本 | 股 | 报告期末流通股本 | query_profit_data.liqaShare |
| `YOYNI` | 净利润同比 | % | 同比增长率，baostock 原值（小数） | query_growth_data.YOYNI |
| `YOYEquity` | 净资产同比 | % | 同上 | query_growth_data.YOYEquity |
| `YOYAsset` | 总资产同比 | % | 同上 | query_growth_data.YOYAsset |
| `YOYEPSBasic` | 每股收益同比 | % | 同上 | query_growth_data.YOYEPSBasic |
| `YOYPNI` | 归母净利同比 | % | 同上 | query_growth_data.YOYPNI |
| `NRTurnRatio` | 应收账款周转率 | 次 | 营业收入/平均应收账款 | query_operation_data.NRTurnRatio |
| `NRTurnDays` | 应收账款周转天数 | 天 | 360/周转率 | query_operation_data.NRTurnDays |
| `INVTurnRatio` | 存货周转率 | 次 | 营业成本/平均存货 | query_operation_data.INVTurnRatio |
| `INVTurnDays` | 存货周转天数 | 天 | 360/周转率 | query_operation_data.INVTurnDays |
| `CATurnRatio` | 流动资产周转率 | 次 | 营业收入/平均流动资产 | query_operation_data.CATurnRatio |
| `AssetTurnRatio` | 总资产周转率 | 次 | 营业收入/平均总资产 | query_operation_data.AssetTurnRatio |
| `CFOToNP` | 经营现金流/净利润 | 倍 | 利润含金量指标 | query_cash_flow_data.CFOToNP |
| `CFOToOR` | 经营现金流/营业收入 | 倍 | 经营现金流占收入比 | query_cash_flow_data.CFOToOR |
| `CFOToGr` | 经营现金流/营业总收入 | 倍 | 同上口径（增长率分母） | query_cash_flow_data.CFOToGr |
| `CAToAsset` | 流动资产/总资产 | 倍 | 结构指标 | query_cash_flow_data.CAToAsset |
| `NCAToAsset` | 非流动资产/总资产 | 倍 | 结构指标 | query_cash_flow_data.NCAToAsset |
| `cfo_est` | 经营现金流 | 元 | 东财源为真实披露绝对值；baostock 源为推算（MBRevenue × CFOToOR），查不到比率则 NaN | ocf_abs / 推算 |
| `ocf_abs` | 经营现金流量净额 | 元 | 真实披露绝对值，非推算。东财源与「巨潮+新浪」源都有；baostock 源没有此字段（只有比率 CFOToOR） | stock_xjll_em / 新浪「经营现金流量净额」 |
| `total_equity` | 股东权益合计 | 元 | **仅东财源有**，用于推算 ROE | stock_zcfz_em.股东权益合计 |
| `total_asset` | 资产总计 | 元 | **仅东财源有** | stock_zcfz_em.资产-总资产 |
| `debt_ratio` | 资产负债率 | % | **仅东财源有** | stock_zcfz_em.资产负债率 |
| `roeAvg_derived` | 净资产收益率（推算） | % | **推算列**：netProfit / total_equity × 100，报告期累计口径，未年化 | 本层推算 |
| `npMargin_derived` | 净利率（推算） | % | **推算列**：netProfit / MBRevenue × 100 | 本层推算 |
| `gpMargin_derived` | 毛利率（推算） | % | **推算列**：(MBRevenue − 营业支出) / MBRevenue × 100 | 本层推算 |
| `CFOToOR_derived` | 经营现金流/营收（推算） | % | **推算列**：ocf_abs / MBRevenue × 100 | 本层推算 |
| `CFOToNP_derived` | 经营现金流/净利润（推算） | % | **推算列**：ocf_abs / netProfit × 100 | 本层推算 |
| `source` | 数据来源 | - | akshare_em = 东财三表（全市场，剔除退市股）；cninfo+sina = 巨潮公告日 + 新浪数值（补退市股，字段完整）；baostock = 退市股兜底（MBRevenue 缺失率 55%，仅在前两者覆盖不到的期次生效） | 本层标注 |
| `pubdate_src` | 披露日来源标记 | - | announce = pubDate 为实测公告日；deadline_est = 实测缺失，pubDate_est 用法定截止日兜底 | 本层计算 |
| `pubDate_est` | 可用披露日（含兜底） | YYYY-MM-DD | 回测对齐**推荐用这一列**：= pubDate，缺失时取交易所法定披露截止日（一季报 04-30 / 半年报 08-31 / 三季报 10-31 / 年报次年 04-30）。截止日必然 ≥ 真实披露日，只可能晚知道、不可能预知未来 | 本层计算 |
| `fetch_date` | 抓取日期 | YYYY-MM-DD | 溯源用 | 本层写入 |

## pool_daily

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `date` | 交易日期 | YYYY-MM-DD | 交易日 | 由 daily 派生 |
| `code` | 股票代码 | - | 含已退市股票 | 由 daily 派生 |
| `is_st` | 当日是否 ST | 0/1 | baostock isST | 由 daily 派生 |
| `suspended` | 当日是否停牌 | 0/1 | 停牌不可成交 | 由 daily 派生 |
| `new_stock` | 当日是否次新 | 0/1 | 上市不满 120 交易日 | 由 daily 派生 |
| `delisted` | 是否已退市 | 0/1 | 该股最终退市（保留在历史池中，防生存偏差） | 由 daily 派生 |
| `listed_days` | 上市交易日序号 | 个 | 自 ipoDate 起算 | 由 daily 派生 |
| `tradable` | 当日可交易 | 0/1 | 未停牌 → 1；ST/次新不在此列剔除，由策略自行过滤 | 由 daily 派生 |

## stocks

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `code` | 股票代码 | - | 全 A 含已退市 | baostock query_stock_basic |
| `code_name` | 股票名称 | - | 最新简称，历史上改名不追溯 | 同上 |
| `ipoDate` | 上市日期 | YYYY-MM-DD | 首次上市日 | 同上 |
| `outDate` | 退市日期 | YYYY-MM-DD | 空=仍在上市 | 同上 |
| `is_delisted` | 是否已退市 | 0/1 | outDate 非空 | 同上 |
| `data_completeness` | 数据完整度 | 0-1 | 区间内有效行情行数 / 该股应有交易日数 | 本层计算 |

## delisted

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `code` | 股票代码 | 6 位 | 无交易所前缀 | akshare 退市清单 + baostock |
| `name` | 股票名称 | - | 退市时简称 | 同上 |
| `delist_date` | 退市日期 | YYYY-MM-DD | 终止上市日 | 同上 |

## industry

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `code` | 股票代码 | 6 位 | 无前缀 | akshare 东财行业板块成分 |
| `industry` | 所属行业 | - | 东财行业分类；失败时退回证监会行业 | akshare stock_board_industry_cons_em |

## manifest

| 字段 | 中文名 | 单位 | 口径 | 来源 |
|---|---|---|---|---|
| `table` | 表名 | - | 产物表名 | 本层写入 |
| `rows` | 行数 | 行 | 本次清洗落盘行数 | 本层写入 |
| `built_at` | 生成时间 | YYYY-MM-DD HH:MM:SS | 清洗时间 | 本层写入 |

## 缺失值约定

- 所有缺失一律写 NaN / NULL，**不插补、不估算、不线性填充**。
- 唯一例外：`finance.cfo_est`（经营现金流推算值），已在该行口径里标为推算，不得当作披露原值引用。
- 下游展示缺失值时必须显示「缺失」，不得用 0 或均值替代。
