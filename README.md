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

- [x] S0：项目骨架（本阶段）
- [ ] S1 起：待验收后另发指令
