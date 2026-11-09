# 量化分析 App（quant-app）

> 定位一句话：**不荐股，只把"我是怎么分析出来的"全部摊开给你看，让你自己拍板。**
> 当前阶段：S1 数据层已完成（26 个用例、90 字段字典）；S2 回测引擎未开工，界面未开工

---

## ⚠️ 三条不可妥协的底线（原文抄录自《量化分析 App · 框架设计》第 0 节）

| # | 底线 | 怎么落实 |
|---|---|---|
| 1 | **不编造数据** | 任何数字必须能溯源到 `data/raw/` 里的原始文件 + 字段名；查不到就显示"缺失"，绝不插补、绝不估算 |
| 2 | **不承诺收益、不给买卖信号** | 输出是"证据卡 + 结论标签（观察/回避/可研究）"，不是"买入/卖出"。界面常驻免责声明 |
| 3 | **合规** | 仅个人本地研究工具，不部署公网、不对外提供、不收费、不代客理财。境内公开荐股需持牌，本工具刻意规避 |

> 这三条写进 `README.md` 顶部，任何功能与之冲突一律砍掉。
> **定位已拍板（2026-11-09，业主）：仅本地、仅本人使用。** AGENTS.md 第七节的商业化立场属跨项目通用偏好，本项目暂不执行；未接到新指令前不对外提供、不收费、不部署。

---

## 环境

- Python 3.12+（实测 3.13.14；虚拟环境 `.venv/`）。numpy 2.5.4 要求 ≥3.12，3.11 装不上。
- 依赖见 `requirements.txt`（已锁版本）：pandas 3.1.1 / numpy 2.5.4 / baostock 0.9.4 / akshare 1.20.36 / streamlit 1.67.0 / pytest 9.2.0 / pyarrow 26.0.1

```bash
# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 跑测试（data/ 不在 git，新机器无数据时是 1 passed / 25 errors，属正常）
pytest
```

## 目录结构（实际仓库，2026-11-09 核对）

```
quant-app/
├─ data/
│  ├─ raw/          # 原始落地文件，只增不改，文件名带抓取日期（git 忽略，仓库里没有）
│  ├─ cache/        # 清洗产物（git 忽略，仓库里没有）
│  └─ dict.md       # 字段字典，唯一进 git 的 data 文件，由 src/clean.py 生成
├─ src/             # 现状仅 ingest.py / ingest_ak_fin.py / clean.py / fill_delisted_cninfo.py / fill_delisted_gap.py
├─ tests/           # test_data.py（25 个数据层断言）+ test_smoke.py
└─ docs/            # 5 份设计文档 + 换机迁移指南
```

规划但仓库里尚不存在：`factors/` `backtest/` `metrics/` `evidence/` `src/api/`、`web/`、`out/`（git 不存空目录）。别说成已有。

## 施工阶段

- [x] S0：项目骨架
- [x] S1：数据层（抓取 + 清洗 + 股票池 + 字段字典 + 测试）
- [ ] S2 起：待验收后另发指令

---

## 数据层（S1）

### 怎么跑

```bash
.venv\Scripts\activate

# 1) 元数据：股票清单（含退市）、退市清单、行业分类
python -m src.ingest --stage meta

# 2) 全 A 日线（按年分 parquet）+ 复权因子
#    全量 2018-2026 约一小时（workers=3）；增量只跑缺的年份
python -m src.ingest --stage daily --years 2018 2026

# 3) 季频财务主体：东财全市场三表（32 个报告期约 10 分钟）
python -m src.ingest_ak_fin --from-year 2018 --to-year 2026

# 4) 退市股财务补齐：巨潮取首次披露日 + 新浪取数值
#    东财把退市股从历史报告期里剔除了，不补就是生存偏差
#    按年段分开跑：窗口后移就补新年段（每段 10-25 分钟）
python -m src.fill_delisted_cninfo --all-delisted --from-year 2018 --to-year 2022
python -m src.fill_delisted_cninfo --all-delisted --from-year 2023 --to-year 2024
python -m src.fill_delisted_cninfo --all-delisted --from-year 2025 --to-year 2026

# 5) 清洗入 cache.db + 生成字段字典
python -m src.clean --years 2018 2024

# 测试
pytest
```

### 落地结构

| 位置 | 内容 |
|---|---|
| `data/raw/daily/daily_{年}_{抓取日}.parquet` | 日线原始价（不复权）+ isST/tradestatus 等原始字段 |
| `data/raw/finance/finance_ak_{年段}_{抓取日}.parquet` | 财务主体：东财全市场三表（在市公司） |
| `data/raw/finance/finance_cninfo_delisted_{年段}_{抓取日}.parquet` | 财务补充：退市股（巨潮公告日 + 新浪数值） |
| `data/raw/finance/finance_bs_gap_2018_2022_{抓取日}.parquet` | 财务兜底：baostock 退市股（src/fill_delisted_gap.py 写出的实际文件名，年段写死 2018_2022） |
| `data/raw/meta/` | 股票清单、复权因子、行业分类 |
| `data/raw/delisted.csv` | 退市清单（akshare 沪深 + baostock 补全） |
| `data/cache/daily_{年}.parquet` | 清洗宽表：新增 `hfq_*`、`suspended`、`new_stock`、`listed_days` |
| `data/cache/finance.parquet` | 财务宽表（含推算列 `cfo_est`，已标注；`source` 列标来源；`pubDate_est` 为回测对齐用披露日） |
| `data/cache.db` | SQLite：`pool_daily` / `stocks` / `delisted` / `industry` / `manifest` |
| `data/dict.md` | 字段字典（由 `src/clean.py` 生成，90 个字段，与代码同源） |

### 几个必须知道的坑

1. **baostock 封的是「请求频率」，不是连接数。** 日线 0.3 次/秒跑 5 小时没事；
   财务 5.5 次/秒跑 25 分钟（约 8000 次）就被 IP 级拉黑（`10001011 黑名单用户`），
   换账号无效。默认 `--workers 3`；`--req-interval` 只存在于 `src/ingest.py`（默认 0.0，长任务建议 1.0）
   和 `src/fill_delisted_gap.py`（默认 1.2），**`src/fill_delisted_cninfo.py` 没有这个参数，用 `--sleep`（默认 0.4）**。
   实测一次封禁最长超过 13 小时。
2. **baostock 会静默截断。** 偶发"只返回前几百行但 error_code=0"。`src/ingest.py` 的
   `_query_with_retry` 支持 `min_rows`，但**只有 `query_stock_basic` 传了 3000**，其余调用均为默认 0（不校验）。
   别以为所有全量接口都防了截断。
3. **后复权算法**：`close × backAdjustFactor`（按除权日 backward 取值）。
   旧记录称与 baostock 官方后复权价在 519/300750/000005 上偏差 0.0000%，
   但**本仓库没有可复现的对拍脚本**，现有测试只验与 `pctChg` 自洽；未验证，别当已验证结论引用。
4. **「公告日期」这个字段，只有两个地方能用。** 东财财报接口和 新浪财务接口里的
   「公告日期」都是**最新修订日**——实测 sz.300356 的 2018 一季报被标成 2019-04-27
   （滞后 392 天），拿它做时点对齐等于提前一年看到数据。真值有两个来源：
   东财 `stock_report_disclosure` 的「实际披露」（只覆盖在市公司）、
   巨潮公告原文列表的首条披露日（覆盖退市股）。两者都不适用时留 NaN，绝不估算。
5. **比率字段统一百分数。** baostock 的 `roeAvg/gpMargin/npMargin` 是小数（0.184），
   东财和新浪是百分数（18.4）。三个源并表时已在入层 ×100 归一，
   `test_finance_units_are_percent` 守着这条线。
6. **东财财报剔除退市股，且按年段补。** 2018 一季报里 250 只 2018 年后退市股只剩 19 只，
   直接用就是生存偏差。必须跑 `fill_delisted_cninfo --all-delisted` 补齐；
   补完 2018-2022 不等于 2023-2024 也有——**回测窗口后移就得按新年段再补一次**。
   `test_finance_covers_delisted_in_backtest_window` 只参数化 `[2023, 2024]`，**2025/2026 无断言**
   （数据已同步到 2026-11），窗口后移要连测试一起改。旧机器实跑 2023 覆盖 94.0%、2024 覆盖 91.4%，
   本仓库无 `data/`，这两个数未验证。
7. **回测对齐用 `pubDate_est`，不要用 `pubDate`。** 退市股部分报告期在巨潮查不到披露日
   （窗口内 4.1%），`pubDate_est` 用交易所法定截止日兜底（一季报 04-30 / 半年报 08-31 /
   三季报 10-31 / 年报次年 04-30）。截止日必然 ≥ 真实披露日，只可能晚知道、不会预知；
   `pubdate_src` 列标明哪条是实测、哪条是兜底，S3 想更严格就只取 `announce`。
8. **增量 = 新抓取日新增文件**，历史版本永不覆盖。同一天断点续跑时按主键并集补全。

