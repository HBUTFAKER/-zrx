# 量化分析 App（quant-app）

> 定位一句话：**不荐股，只把"我是怎么分析出来的"全部摊开给你看，让你自己拍板。**
> 当前阶段：S0 项目骨架（无任何业务功能）

---

## ⚠️ 三条不可妥协的底线（原文抄录自《量化分析 App · 框架设计》第 0 节）

| # | 底线 | 怎么落实 |
|---|---|---|
| 1 | **不编造数据** | 任何数字必须能溯源到 `data/raw/` 里的原始文件 + 字段名；查不到就显示"缺失"，绝不插补、绝不估算 |
| 2 | **不承诺收益、不给买卖信号** | 输出是"证据卡 + 结论标签（观察/回避/可研究）"，不是"买入/卖出"。界面常驻免责声明 |
| 3 | **合规** | 仅个人本地研究工具，不部署公网、不对外提供、不收费、不代客理财。境内公开荐股需持牌，本工具刻意规避 |

> 这三条写进 `README.md` 顶部，任何功能与之冲突一律砍掉。

---

## 环境

- Python 3.11（虚拟环境 `.venv/`）
- 依赖见 `requirements.txt`：pandas / numpy / baostock / akshare / streamlit / pytest / pyarrow

```bash
# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 跑测试
pytest
```

## 目录结构（对齐框架文档 3.2 节）

```
quant-app/
├─ data/
│  └─ raw/          # 原始落地文件，只增不改，文件名带抓取日期（git 忽略）
├─ src/             # 业务代码（ingest / clean / factors / backtest / metrics / evidence / api）
├─ web/             # 前端
├─ out/             # 每次分析的结果快照（git 忽略）
├─ tests/           # 复算一致性测试
└─ docs/            # 文档
```

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
python -m src.ingest --stage daily --years 2018 2024

# 3) 季频财务（营收/净利/ROE/毛利率/经营现金流 + pubDate/statDate）
python -m src.ingest --stage finance --years 2018 2024

# 4) 清洗入 cache.db + 生成字段字典
python -m src.clean --years 2018 2024

# 测试
pytest
```

### 落地结构

| 位置 | 内容 |
|---|---|
| `data/raw/daily/daily_{年}_{抓取日}.parquet` | 日线原始价（不复权）+ isST/tradestatus 等原始字段 |
| `data/raw/finance/finance_{年}_{抓取日}.parquet` | 季频财务，含 `pubDate` / `statDate` |
| `data/raw/meta/` | 股票清单、复权因子、行业分类 |
| `data/raw/delisted.csv` | 退市清单（akshare 沪深 + baostock 补全） |
| `data/cache/daily_{年}.parquet` | 清洗宽表：新增 `hfq_*`、`suspended`、`new_stock`、`listed_days` |
| `data/cache/finance.parquet` | 财务宽表（含推算列 `cfo_est`，已标注） |
| `data/cache.db` | SQLite：`pool_daily` / `stocks` / `delisted` / `industry` / `manifest` |
| `data/dict.md` | 字段字典（由 `src/clean.py` 生成，78 个字段，与代码同源） |

### 几个必须知道的坑

1. **baostock 有并发上限。** 并发开到 8 会被 IP 级拉黑（错误码 `10001011 黑名单用户`），
   换账号无效，只能降并发等解封。默认 `--workers 3`，实测安全。
2. **baostock 会静默截断。** 偶发"只返回前几百行但 error_code=0"。代码里对所有
   全量接口加了最小行数校验，行数不足即重试，避免残缺数据落盘。
3. **后复权算法与官方对拍过。** `close × backAdjustFactor`（按除权日 backward 取值）
   与 baostock 官方后复权价在 519/300750/000005 上偏差 0.0000%。
4. **财务只能用 pubDate 对齐。** `akshare` 的业绩报表接口里"最新公告日期"是**后来更新的**，
   不是原始公告日，会污染防未来函数，所以财务一律走 baostock 季频接口。
5. **增量 = 新抓取日新增文件**，历史版本永不覆盖。同一天断点续跑时按主键并集补全。

