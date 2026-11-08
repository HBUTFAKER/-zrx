# 换机迁移指南（2026-11-09）

## 新电脑上的恢复步骤

1. 装 Python 3.11（勾选 Add to PATH），装 git。
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
4. 验证环境：`python -m pytest tests/ -q`——data/ 是空的会部分跳过或报错，
   属正常（数据不在 git 里，见下条）。
5. 数据二选一：
   - 旧电脑把 `data/` 整个目录拷到移动硬盘/U盘带过去（约 1.8G，省下载时间）；
   - 或新电脑按《施工指令书-总工版.md》量化炒股系统002 重新增量下载。
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

## 遗留决策（未拍板，见 project-memory.md 末尾）

1. gaps.csv 缺失（1.6% 退市股缺口无记录）——需施工A返工补交
2. 2010–2017 日线未补全——暂定研究区间 2018 起
3. 002 补审（审计C 尚未开窗）
