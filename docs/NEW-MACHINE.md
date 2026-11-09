# 换机迁移指南（2026-11-09）

## 新电脑上的恢复步骤

1. 装 Python 3.12+（2026-11-09 实测 3.13.14；numpy 2.5.4 要求 ≥3.12，旧文档写的 3.11.9 装不上已锁版本的依赖），装 git。
2. clone：
   ```
   git clone https://github.com/HBUTFAKER/-zrx.git quant-app
   cd quant-app
   ```
3. 重建虚拟环境（代码里 .venv 不进 git，属正常）：
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
   requirements.txt 已锁版本，别删版本号；首次下载约 10 分钟。
4. 验证环境：`python -m pytest --collect-only -q` 应收集 26 个用例；`python -m pytest -q` 在 data/ 为空时是
   **1 passed / 25 errors**（fixture 用 pytest.fail，不是跳过），属正常；数据就位后才谈全绿。
   另可验：`python -c "import baostock, akshare"` 与 5 个 `python -m src.* --help`。
5. 数据二选一（推荐 A：拷旧机，数据一致、不碰外部接口）：
   - **A. 旧机 → U盘 → 新机**，用 robocopy，别用资源管理器拖拽。别加 `/MIR`（会删目标文件）。
     旧机：先关掉正在跑的 Python/Streamlit，避免拷到写了一半的 `cache.db`；`E:` 换成 U盘实际盘符，建议 exFAT。
     ```
     robocopy "C:\Users\29212\quant-app\data" "E:\quant-app-data" /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /MT:8 /XF dict.md /LOG:E:\copy.log
     ```
     新机：`<仓库根目录>` = git clone 出来的那一层，`data/` 必须放在这里（代码按 `Path(__file__).parent.parent` 定位）。
     ```
     robocopy "E:\quant-app-data" "<仓库根目录>\data" /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /MT:8 /LOG:copy-in.log
     ```
     - 旧机路径按 `project-memory.md` 记录是 `C:\Users\29212\quant-app`，不一样就替换。
     - `/XF dict.md` 是故意排除：字段字典已进 git、由 `src/clean.py` 生成，别用旧机版本覆盖；两边不一致先核对代码版本。
     - robocopy 退出码 0–7 都算成功，≥8 才是失败。
     - 两端各跑一次核对文件数与字节，目标端应比源端少 1 个文件（dict.md）：
       ```
       powershell -c "$f=Get-ChildItem '<data 目录>' -Recurse -File; '{0} files, {1} bytes' -f $f.Count,($f|Measure-Object Length -Sum).Sum"
       ```
     - 验收：`.venv\Scripts\python.exe -m pytest -q` 应为 26 passed；`test_data_is_fresh` 要求 `pool_daily` 最新交易日在 45 天内。
     - 拷完只跑测试，别跑 `src.clean`（它会整表重建 `cache.db`、重写 `dict.md`，属批量写数据）。别 `git add -f data/`（1.8G 不该进仓库）。
     - 旧机若有 `data/gaps.csv`（返工补交的缺口记录），一起拷；没有就保持缺失，别自己造。
   - **B. 重抓**：按 README「数据层（S1）怎么跑」五步执行。代价：日线约 1 小时 + 财务约 10 分钟 + 退市股三段各 10–25 分钟；baostock 按请求频率封 IP（README 坑 1）；抓取日不同会新增一批 raw 文件（只增不改，属正常）。
6. 把 docs/ 里四份文档的路径告诉新会话的总工（原绝对路径已失效，
   新路径 = 本仓库 docs/ 目录）。

## 文档清单（本目录）

| 文件 | 用途 |
|---|---|
| 量化分析App-框架设计.md | 总蓝图：底线/纪律/回测规则/页面/验收 |
| 施工指令书-总工版.md | 六阶段指令词 + 口径细则 + 断言规格 |
| 三窗口合作协议.md | 组织规则：总工/施工A/B/审计C/业主 |
| project-memory.md | 项目工作记忆（决策与进度流水） |
| 炒股入门-全流程.md | 背景：业主入门笔记，非施工必需 |

## 遗留决策

1. ~~gaps.csv 缺失~~ 已闭环（2026-11-09，002R）：`data/gaps.csv` 492 行，审计-R 定点复审确认；95 只零行情退市股全部退市早于 2018 窗口，非真缺口。
2. 2010–2017 日线未补全——暂定研究区间 2018 起（待业主拍板）
3. ~~002 补审（审计C 尚未开窗）~~ 已完成（2026-11-09，审计-A + 审计-R）
4. 002 口径 1/5b 已修订（2026-11-09，业主拍板）：pool_daily 维持日线宽表派生，`query_all_stock` 逐日取不再执行，断言 (b) 作废——详见 AGENTS.md 五节与《三窗口合作协议》变更记录
5. 财务 pubDate 容缺阈值 25%（test_finance_has_both_dates）由总工收编为规格（2026-11-09，审计-R #3），缺失原因=招股书口径，不插补
6. R3 财务覆盖率 2023/2024 各缺 9 只——002R2 补抓进行中
