"""S2 回测引擎：自研轻量事件驱动（不引 backtrader，全流程可审计）。

口径全部出自《量化分析 App · 框架设计》v0.2，逐条对应：
  · 5.1 信号 T 日收盘产生 → **T+1 开盘价**成交；
  · 5.1 涨停不买、跌停不卖、停牌跳过；
  · 5.1 佣金 0.025%（双边，最低 5 元）、印花税 0.05%（仅卖出）、过户费 0.001%（双边）；
  · 5.1 持仓 T+1 才能卖；持仓只数、单只权重上限可配；
  · 3.3-1 时点对齐：T 日收盘后才知道的数据，最早 T+1 开盘执行（本引擎用两张价表物理隔离：
        决策只看 close[t]，成交只用 open[t+1]，代码里不存在第 t 日就用第 t 日开盘价的分支）；
  · 3.3-2 复权（**总工 2026-11-10 裁决修订**，2026-11-11 003R2 补退出路径）：成交股数、
        成交价、成交金额、费用一律用**原始价与真实金额**记账；分红再投按 hfq 收益率在
        估值环节累乘，**退出时按 hfq 口径变现**（现金为名义权益账）。理由见下方"记账模型"；
  · 3.3-3 停牌：当日不产生成交，持仓按最后有效价结转；
  · 3.3-5 涨跌停：开盘即触板视为不可成交（一字板更不可能成交）。

记账模型（总工 2026-11-10 裁决，替代原先"hfq 名目金额配仓位"；
          2026-11-11 003R2 修复退出路径分红层蒸发，审计 B2-01）
--------------------------------------------------------------
原实现用 hfq 价算股数：`qty = 目标市值 / hfq价`。hfq 价是原始价的累积复权倍数，
老股票可以差三个数量级（实测最高 1303 倍），于是：
  · 手数取整（100 股）在 hfq 价上千元的股票上等于"一手 = 十几万"，目标权重根本配不出来；
  · 最低佣金 5 元被名义金额稀释，真实成本失真；
  · 资金约束、单只权重上限全部按虚拟金额判断 —— 权重实测最高失真 1303 倍。
现改为两套账并行：
  1) **真实账**：shares（股）、成交价（原始价 ± 滑点）、成交金额（qty × 原始价）、
     交易费用（按真实原始金额计），全部是真实数字。手数、最低佣金、涨跌停、资金约束
     才有物理意义。
  2) **权益账**：每股维护 `nhq`（名义 hfq 股数）
        买入: nhq += 成交金额 / (hfq_open × (1+滑点))
        卖出: nhq *= (1 - 卖出股数 / 持仓股数)    # 按比例缩放，保证剩余仓位估值连续
        市值 = nhq × hfq_close(最后有效值)
     这个折算恰好等价于"分红全部按当日价再投"：hfq = 原始价 × 累积复权因子，
     nhq × hfq 同时吸收了价格变动与分红再投。
     买入当日市值 = 成交金额 / (1+滑点)，滑点损失自然体现为市值的即时折价。

**现金是名义权益账**（003R2 修订）：分红再投层随退出按 hfq 变现，交易费用按真实原始
金额计。卖出/清算时：
      账面价值 book = nhq_sold × hfq_price            # 含持有期内累积的分红再投层
      变现额 proceeds = book × (1 − 滑点)             # 卖出压价，与买入侧 (1+滑点) 对称
      交易费用 fee = Costs.fees(qty_raw × 原始价, 方向)  # 真实原始金额计费
      现金 += proceeds − fee
所以退出只损失「滑点 + 费用」，权益差恒等于 −(book×滑点 + fee)，分红层一分不蒸发。
旧实现（003R 及以前）按原始价回款、nhq 直接清零，把累计分红层在退出瞬间凭空抹掉
（审计 B2-01 实测 2,521/3,420 笔退出带分红层，≈2.57pp/年）。`_sell` / `_liquidate`
内置断言守这条：任何一笔退出，现金增加额与持仓市值减少额的合计必须只剩滑点与费用。

退市 / 数据终止清算（总工 2026-11-10 裁决）
--------------------------------------------
持仓股在回测区间内数据终止（退市、长期停牌至区间结束）且之后不再有行情时，
在其**最后有效收盘价**一次性强制清算回收现金，不再留下永远卖不掉的仓位。
保守假设：忽略老三板/股转系统的残余转让价值，故清算所得**低估**真实残值——
这是有意为之的保守口径，宁可少算收益也不虚构退出渠道。

hfq 缺口处理（003R3 / 审计 B3-01，替换 003R2 的"按原始价填充"）
-----------------------------------------------------------------
缺口 = 原始价有、后复权价缺。实测 1,942 行 / 2 只（sz.001914 2018 全年 243 行，
缺口后真实因子 6.4111；sz.302132 2018-01-02~2024-12-31 共 1,699 行，缺口后因子 1.0）。
003R2 曾把缺口按 `hfq := 原始价`（因子 1）填平 —— 结果是缺口期开仓的 nhq 在缺口结束后
被真实因子放大 6.41 倍，退出时按 hfq 变现成真金白银（幻影层，实测 ≈34,037 元 ≈0.31pp/年）。
现改为**只标注不插补**：
  · 缺口格 hfq 保持 NaN，同时打进 panel.hfq_gap → 当日禁买、持仓禁卖（不参与再平衡）；
  · 持仓过缺口 → 估值冻结在缺口前最后有效 hfq 价（last_hfq 不更新即自动冻结）；
  · 缺口延续到数据末端 → 按退市清算规则以**最后有效 hfq 价** ×(1−滑点) 清算；
  · 台账落 result.json 的 data.hfq_gap_rows（行数/股票）与 data.hfq_gap_ledger
    （禁买次数、持仓冻结天数、缺口清算笔数与金额），可追溯。

本文件只提供"怎么成交、怎么记账"，不含任何选股/择时逻辑 —— 那是 S3 的事。
唯一自带的 signals 生成器是 `harness_monthly_equal_weight`，它是**引擎自检脚手架**
（等权持有全市场、月末调仓），用来跑通管线并与"等权全市场"基准互相印证，不是策略。

用法（示例回测，产物落 out/）：
    python -m src.backtest --start 2018-01-02 --end 2026-11-06 --out out
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
DB = ROOT / "data" / "cache.db"
OUT = ROOT / "out"

TRADING_DAYS = 252  # 年化用，A 股约 242~244，统一取 252 与框架 5.2 常见口径一致
LIMIT_TOL = 0.005   # 涨跌停判定容差：只可能差 1 个 tick（半分钱）

# 产物里的有效性声明（审计 B3-02）：S2 全量回测**不能**当作策略有效性/alpha 证据。
# 审计 B3-02 给出的二阶拆解（施工本轮未逐项复现，仅作定性引用）：超额主要来自整手低价股
# 倾斜、可交易性过滤、seed 运气，扣掉成本与开盘滞后后所剩无几，不是选股能力。
# 施工实测（003R3 修复后，seed 40~50 共 11 个、同一面板同一参数）：
#   年化 5.1113% ~ 7.0189%，级差 1.91pp，中位 5.9884%，seed=42 处 27.3 分位（略低于中位）。
# 也就是说 seed 级差（1.91pp）远大于相对等权基准的超额（+0.12pp）—— 单看超额毫无意义。
# 有效性判断以 004 因子框架的 holdout 结果为准。
VALIDITY_NOTE = (
    "本文数字是引擎自检脚手架（固定种子等权持有 N 只、月度再平衡）的运行结果，"
    "不是策略有效性或 alpha 的证据。已知偏差：整手低价股倾斜、seed 敏感度"
    "（实测 seed 40~50 共 11 个：年化 5.1113%~7.0189%，级差 1.9076pp，"
    "中位 5.9884%，seed=42 恰好落在中位（升序第 6/11，54.5 分位））、"
    "可交易性过滤、成本与开盘滞后。策略有效性判断以 004 因子框架的 holdout 结果为准，"
    "不以 S2 全量回测收益为准。")


# ---------------------------------------------------------------- 交易所规则
def round2(x) -> np.ndarray:
    """交易所式四舍五入到分。

    不能用 np.round：它是银行家舍入且受二进制表示影响，28.325 会给出 28.32，
    而交易所的涨停价是 28.33 —— 一个 tick 的偏差足以让"能不能买"判反。
    """
    return np.floor(np.asarray(x, dtype="float64") * 100.0 + 0.5 + 1e-9) / 100.0


def limit_ratio(code: str, is_st: bool, date: str) -> float:
    """个股当日涨跌幅限制（单值版本，供文档与单元测试引用）。

    · 科创板（688/689）：±20%
    · 创业板（300/301）：2020-08-24 注册制改革前 ±10%，之后 ±20%
    · 北交所（4xx/8xx/920）：±30%
    · 主板：ST/*ST ±5%，其余 ±10%
    无涨跌幅限制的情形（上市首日、注册制新股前 5 日）不在本函数处理，
    由 `_no_limit_mask` 统一判定后直接把 blocked 置 False，见该函数说明。
    """
    if code.startswith(("sh.688", "sh.689")):
        return 0.20
    if code.startswith(("sz.300", "sz.301")):
        return 0.20 if date >= "2020-08-24" else 0.10
    if code.startswith(("bj.", "sz.4", "sz.8", "sh.4", "sh.8")):
        return 0.30
    return 0.05 if is_st else 0.10


def _ratio_array(codes: pd.Series, is_st: pd.Series, dates: pd.Series) -> np.ndarray:
    """向量化版本的涨跌幅比例（与 `limit_ratio` 同口径）。"""
    code = pd.Series(codes).reset_index(drop=True)
    st = pd.Series(is_st).reset_index(drop=True)
    dt = pd.Series(dates).reset_index(drop=True)
    board = code.str.startswith(("sh.688", "sh.689"))
    gem = code.str.startswith(("sz.300", "sz.301"))
    bj = code.str.startswith(("bj.", "sz.4", "sz.8", "sh.4", "sh.8"))
    main = ~(board | gem | bj)

    r = pd.Series(0.10, index=code.index, dtype="float64")
    r[board] = 0.20
    r[bj] = 0.30
    r[main] = np.where(st[main] == 1, 0.05, 0.10)
    r[gem] = np.where(dt[gem] >= "2020-08-24", 0.20, 0.10)
    return r.to_numpy(dtype="float64")


def _no_limit_mask(codes: pd.Series, dates: pd.Series, listed_days: pd.Series,
                   ipo_dates: pd.Series | None = None) -> np.ndarray:
    """当日是否**无涨跌幅限制**（这类交易日不判涨停/跌停，直接视为可成交）。

    规则（B-02 修正，003R2 补全 ①②）：
      · 上市首日（真实上市第 1 个交易日）：全市场无涨跌幅限制；
      · 科创板 688/689：注册制，上市前 5 个交易日无涨跌幅限制；
      · 创业板 300/301：2020-08-24 注册制改革后，新股前 5 个交易日无限制；
        （改革前创业板仍是核准制，新股首日之外照常 ±10%，故只对改革后生效）
      · **主板（沪深 600/601/603/605/000/001/002/003 等）：2023-02-17 全面注册制后，
        新股前 5 个交易日无涨跌幅限制**；该日之前主板新股仍是首日 +44%/−36% 之后 ±10%，
        即只有上市首日不受限（由"上市首日"规则覆盖）。
      · 北交所：仅上市首日无限制，之后 ±30%（已由"上市首日"规则覆盖）。

    ① 日历截断修正（审计 B2-02①）：`listed_days` 由 `searchsorted(全市场日历, ipoDate)`
       算出，ipoDate 早于交易日历起点时 searchsorted 返回 0，于是**存量股在日历首日
       被算成 listed_days=1**，误判成"上市首日无涨跌幅"（实测 2018-01-02~01-08
       共 17,350 行 / 3,470 只）。传入 `ipo_dates` 后改为按**面板日历**重算：
       ipoDate 落在日历内才算上市日，落在起点之前一律视为存量股（listed_days=∞），
       按普通股涨跌停判定；真 IPO 首日（date == ipoDate）行为不变。
       实测该口径下 2017-12 上市的 24 只在 2018-01-02 被正确归为存量股，
       而这些票在 2018 年都是主板/中小板，除首日外本就有 ±10%，不会误放行。

    原实现用 `ratio = 0` 表达"无限制"，方向是反的：ratio=0 时涨跌停价都等于昨收，
    开盘价几乎必然 >= 涨跌停价 − 容差，结果是**几乎全部判成不可成交**，
    与注释写的"永远未触板"正好相反。现改为显式掩码，ratio 只表达真实比例。
    """
    code = pd.Series(codes).reset_index(drop=True)
    dt = pd.Series(dates).reset_index(drop=True)
    ld = pd.Series(listed_days).reset_index(drop=True).to_numpy().astype("float64")

    if ipo_dates is not None:
        # 按面板自身交易日历重算上市交易日序号；ipoDate 不在日历内 → 存量股 / 未来股
        ipo = pd.Series(ipo_dates).reset_index(drop=True).fillna("").astype(str).to_numpy()
        cal = np.unique(dt.to_numpy().astype(str))
        pos = np.searchsorted(cal, ipo, side="left")
        pos_c = np.clip(pos, 0, len(cal) - 1)
        in_cal = (pos < len(cal)) & (cal[pos_c] == ipo)
        cur = np.searchsorted(cal, dt.to_numpy().astype(str), side="left")
        ld = np.where(in_cal, (cur - pos + 1).astype("float64"), 1e9)

    first_day = ld <= 1
    star = code.str.startswith(("sh.688", "sh.689")).to_numpy()
    gem = code.str.startswith(("sz.300", "sz.301")).to_numpy()
    gem_reg = gem & (dt >= "2020-08-24").to_numpy()
    # 主板全面注册制：2023-02-17 起新股前 5 个交易日无涨跌幅限制
    bj = code.str.startswith(("bj.", "sz.4", "sz.8", "sh.4", "sh.8")).to_numpy()
    main_reg = (~star) & (~gem) & (~bj) & (dt >= "2023-02-17").to_numpy()
    return first_day | ((ld <= 5) & (star | gem_reg | main_reg))


def _blocked_arrays(codes: pd.Series, is_st: pd.Series, dates: pd.Series,
                    listed_days: pd.Series, open_: np.ndarray, close: np.ndarray,
                    pct_chg: np.ndarray,
                    ipo_dates: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """算"开盘即涨停不可买 / 开盘即跌停不可卖"两个布尔向量。

    昨收（preclose）数据里没有，用 close/(1+pctChg/100) 反推：baostock 的 pctChg 有 5 位小数，
    反推误差 ~1e-4，再归整到 2 位小数（交易所昨收必为 2 位小数）后与真实昨收一致。

    自洽性实测（口径必须写全，否则数字不可复现 —— 审计 B-07）
      样本范围：data/cache/daily_2018..2026.parquet 全部 9 个文件（2018-01-02 ~ 2026-11-06）
      筛选条件：listed_days > 1（新股首日无涨跌幅，不参与）且 |pctChg − ratio×100| < 0.02
      判据：round2(round2(close/(1+pctChg/100)) × (1+ratio)) 与 close 相差 < 0.005 元
      结果：**涨跌停样本 72,693 行，自洽率 99.8652%**
      剩余 0.1348%（98 行）落在半分钱边界：昨收反推的 1e-4 级误差使涨停价差一个 tick，
      由 LIMIT_TOL=0.005 的容差吸收。

      口径敏感性（同一份数据，换个筛选条件样本量就翻倍，所以必须写清口径）：
        · |pctChg − ratio×100| < 0.02            → 72,693 行（本实现采用）
        · |pctChg| >= ratio×100 − 0.02（含超涨超跌） → 164,159 行
        · 仅按 round(pctChg,2) == ratio×100       → 29,713 行
      2021 单年的自洽率由 tests/test_backtest.py::test_limit_price_selfconsistency 守着
      （跌破 99% 即红），防止反推公式被无声改坏。
    """
    pre = round2(close / (1.0 + pct_chg / 100.0))
    ratio = _ratio_array(codes, is_st, dates)
    up = round2(pre * (1.0 + ratio))
    dn = round2(pre * (1.0 - ratio))
    no_limit = _no_limit_mask(codes, dates, listed_days, ipo_dates)

    # 无涨跌幅限制的交易日：不判板（原 ratio=0 写法方向相反，已删除）
    blk_buy = (open_ >= up - LIMIT_TOL) & ~no_limit
    blk_sell = (open_ <= dn + LIMIT_TOL) & ~no_limit
    # 无当日行情（停牌/无行）时价格是 NaN，比较结果为 False，但这类行本就不可成交，
    # 由 suspended / 缺价分支兜住（load_panel 里会把这些行显式标成不可交易）
    return blk_buy, blk_sell


# ---------------------------------------------------------------- 数据面板
@dataclass
class Panel:
    """回测用面板。行=交易日，列=股票代码，全部是 numpy 便于逐日循环。

    open_raw / close_raw：**原始价**，成交、股数、金额、费用、清算的记账口径（总工 2026-11-10 裁决）
    open_hfq / close_hfq：后复权价，用于市值折算（分红再投）与**退出变现**（003R2）；不参与成交价
                          —— hfq 缺口格**保留 NaN**，不插补（003R3 / 审计 B3-01）
    blocked_buy        ：开盘即涨停或涨跌幅不受限以外的不可成交情形 → 不可买
    blocked_sell       ：开盘即跌停 → 不可卖
    suspended          ：停牌日 → 不产生成交（3.3-3）
    hfq_gap            ：hfq **缺口格**（原始价有、后复权价缺）→ 与停牌同级不可交易，
                          持仓期间估值冻结在缺口前最后有效 hfq 价（003R3 / 审计 B3-01）
    """
    dates: list[str]
    codes: list[str]
    open_hfq: np.ndarray
    close_hfq: np.ndarray
    open_raw: np.ndarray
    blocked_buy: np.ndarray
    blocked_sell: np.ndarray
    suspended: np.ndarray
    close_raw: np.ndarray | None = None
    source: dict = field(default_factory=dict)
    hfq_gap: np.ndarray | None = None

    def __post_init__(self) -> None:
        # 人工构造的面板通常没有复权差异（原始价 == 复权价），缺省即视为相等。
        # 真实数据请在 load_panel 里填真值，不要依赖这个兜底。
        if self.close_raw is None:
            self.close_raw = self.close_hfq
        if self.open_raw is None:
            self.open_raw = self.open_hfq
        elif not np.isfinite(self.open_raw).any() or float(np.nanmax(self.open_raw)) == 0.0:
            # 全 0 / 全 NaN 视为"未提供原始价"（旧版 load_panel 就给 zeros），退回复权价
            self.open_raw = self.open_hfq
        if self.hfq_gap is None:
            # 人工面板默认无缺口；真实数据由 load_panel 显式标注
            self.hfq_gap = np.zeros(self.open_hfq.shape, dtype=bool)
        self._check_hfq_gap_consistent()

    def _check_hfq_gap_consistent(self) -> None:
        """旗标与数据一致性校验（003R4 / 审计 B4-02②）。

        `hfq_gap` 是"原始价有、复权价缺"的**唯一**判据：不可交易掩码、估值冻结、
        末端清算全靠它。一旦旗标与数据脱同步（手工面板忘标、便宜行事先填好 hfq、
        clean 层换了复权因子来源），B3-01 那类幻影层就会**静默**复发——净值悄悄变好，
        没有任何一行日志。所以这里在构造面板时就硬校验，两个方向都查：

          · 标了缺口却没缺 → 幻影层的典型形态（缺口期按因子 1 记账），抛错；
          · 缺了却没标缺口 → 该格会被当作正常可交易，卖出/估值拿 NaN 走兜底，抛错。
        """
        miss = ((~np.isnan(self.open_raw) & np.isnan(self.open_hfq))
                | (~np.isnan(self.close_raw) & np.isnan(self.close_hfq)))
        gap = np.asarray(self.hfq_gap, dtype=bool)
        if gap.shape != miss.shape:
            raise ValueError(f"hfq_gap 形状 {gap.shape} 与价格矩阵 {miss.shape} 不一致")
        stale = gap & ~miss
        if stale.any():
            r, c = np.argwhere(stale)[0]
            raise ValueError(
                f"hfq_gap 标记与数据脱同步：{self.dates[r]} {self.codes[c]} 标为缺口，"
                f"但 open/close 与 hfq 价格都在（共 {int(stale.sum())} 格）——"
                f"缺口期会被当成正常可交易，幻影层会静默复发（审计 B3-01/B4-02）")
        unflagged = miss & ~gap
        if unflagged.any():
            r, c = np.argwhere(unflagged)[0]
            raise ValueError(
                f"hfq 缺失但未标 hfq_gap：{self.dates[r]} {self.codes[c]}"
                f"有原始价却无复权价（共 {int(unflagged.sum())} 格）——"
                f"该格会被当正常可交易，估值/变现只能拿原始价顶替（审计 B4-02）")

    @property
    def shape(self) -> tuple[int, int]:
        return self.open_hfq.shape


def _liquidation_due(shares: np.ndarray, last_valid: np.ndarray,
                     last_hfq_valid: np.ndarray, i: int, n_d: int) -> dict[int, bool]:
    """当日到期的强制清算清单 `{列号: 是否因 hfq 断供}`（003R4 / 审计 B4-01、B4-07①）。

    终止点取原始价与 hfq 的**较早**者：任一侧断供都无法继续按同一口径估值。
    `from_gap` 仅在 `hfq_lv < raw_lv`（hfq 先断供）时为真 —— 原始价先终止就是退市，
    哪怕当天恰好落在缺口段里，reason 也必须记 `delisted`，不能把退市记成数据缺口。

    只对"终止日 < 回测末日"的股票返回：末日仍有效的持仓按市值留在账上，那是期末估值。
    """
    out: dict[int, bool] = {}
    for j in np.nonzero(shares > 0)[0]:
        raw_lv, hfq_lv = int(last_valid[j]), int(last_hfq_valid[j])
        if raw_lv < 0 and hfq_lv < 0:
            continue                                  # 完全没有行情，无从清算
        if raw_lv < 0:
            liq_at, from_gap = hfq_lv, False
        elif hfq_lv < 0:
            liq_at, from_gap = raw_lv, False          # hfq 整列缺失不算"缺口先断供"
        elif hfq_lv < raw_lv:
            liq_at, from_gap = hfq_lv, True
        else:
            liq_at, from_gap = raw_lv, False
        if liq_at != i or liq_at >= n_d - 1:
            continue
        out[int(j)] = from_gap
    return out


def _exit_prices(raw_px: float, hfq_px: float, last_raw: float, last_hfq: float,
                 code: str, date: str) -> tuple[float, float]:
    """清算用的两个价：原始价（费用/台账基数）与 hfq 价（变现）。

    hfq 从"当日价 → 最后有效价"依次取，**取不到就停工**（003R4 / 审计 B4-02①）。
    旧代码在这里 `else px` 拿原始价顶替：人工面板 hfq 全缺时 book 按原始价记，
    pnl 被伪造成一笔假亏损（合成实证：真值 ≈102,820 记成 10,282）。
    这与 B3-01 的红线自相矛盾（一边说"绝不拿原始价顶替"，一边留着这条兜底）。
    真实 load_panel 口径下 hfq 必有值，抛错即数据异常信号，不该静默糊过去。
    """
    px = float(raw_px)
    if not (np.isfinite(px) and px > 0):
        px = float(last_raw)
    hx = float(hfq_px)
    if not (np.isfinite(hx) and hx > 0):
        hx = float(last_hfq)
    if not (np.isfinite(hx) and hx > 0):
        raise RuntimeError(
            f"{date} {code} 清算时拿不到 hfq 价（当日 {hfq_px!r}、最后有效 {last_hfq!r}）："
            f"拒绝用原始价 {px!r} 顶替 —— 那会把账面价值和 pnl 一起记错（审计 B4-02）")
    return px, hx


def _last_valid_index(a: np.ndarray) -> np.ndarray:
    """每列最后一个非 NaN 行号；整列全 NaN 记 -1。用于退市/数据终止清算。"""
    valid = ~np.isnan(a)
    idx = np.where(valid, np.arange(a.shape[0])[:, None], -1)
    return idx.max(axis=0)


def _load_ipo_dates(codes: list[str] | None = None,
                    db: Path | None = None) -> pd.Series:
    """从 data/cache.db 的 stocks 表读 ipoDate（**只读**，不改任何数据）。

    用途：`listed_days` 由 searchsorted(全市场日历, ipoDate) 算出，ipoDate 早于日历
    起点时 searchsorted 返回 0，存量股会被算成"上市第 1 天"（审计 B2-02①）。
    有了 ipoDate 就能按面板自身日历重算，把这类行正确归为存量股。

    003R3 / 审计 B3-01 4(c)：读不到就**显式停工**，不再静默退回 listed_days 口径。
    旧兜底让"元数据缺失"和"真的没有新股"在结果里长得一模一样，
    存量股被判成上市首日 → 无涨跌幅限制 → 一字板也能成交，净值悄悄失真。
    缺依赖就报错，让人去补 stocks.ipoDate，这是数据前置条件不是可选优化。
    """
    import sqlite3

    db = db or DB
    if not Path(db).exists():
        raise RuntimeError(
            f"缺少行情元数据库 {db}：回测需要 data/cache.db 的 stocks.ipoDate "
            f"来识别上市首日与存量股，缺失时拒绝静默降级。"
            f"请先跑 python -m src.ingest 生成 stocks 表，或用 load_panel(..., db=...) 指定。")
    try:
        con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
        try:
            s = pd.read_sql("SELECT code, ipoDate FROM stocks", con).set_index("code")["ipoDate"]
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 —— 表缺失/被锁一律停工，不让回测带病跑
        raise RuntimeError(
            f"读取 {db} 的 stocks.ipoDate 失败（{type(e).__name__}: {e}）："
            f"回测依赖该表识别上市首日与存量股，拒绝静默退回 listed_days 口径。") from e
    if codes is not None:
        s = s.reindex(list(codes))
    return s


def _mark_hfq_gap(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """后复权价缺口**只标注、不插补**（003R3 修订，替换 003R2 的"按原始价降级"）。

    实测缺口 1,942 行 / 2 只：
      · sz.001914：2018-01-02 ~ 2018-12-28 共 243 行，缺口后真实因子 ≈6.4111；
      · sz.302132：2018-01-02 ~ 2024-12-31 共 1,699 行，缺口后因子 = 1.0。
    003R2 曾把缺口按 `hfq := 原始价`（因子 1）填平，理由是"避免双分支降级"。
    但 sz.001914 的真实因子是 6.41：缺口期买入的 nhq = 金额/原始价，缺口结束后
    hfq 跳回 6.41× 原始价，市值瞬间虚增 6.41 倍；退出时按 hfq 变现，
    这笔**幻影层**变成真实现金（审计 B3-01 实测前 5 笔退出 ≈34,037 元 ≈0.31pp/年）。
    "统一降级"只是把 B2-05 的静默压低净值，换成了凭空变现现金。

    现口径（总工 2026-11-11 裁定，不许自行放宽）：
      · 缺口格 hfq 保持 NaN，**不插补、不丢行**（红线：只标注不插补）；
      · 缺口格标记进 panel.hfq_gap → 当日禁买、持仓禁卖、估值冻结（见 load_panel）；
      · 缺口延续到数据末端 → 按退市清算规则以最后有效 hfq 价清算并落台账。

    return (原样返回的 df 追加 hfq_gap 标注列, 统计 dict)
    """
    stats = {"rows": 0, "n_codes": 0, "codes": [], "open_rows": 0,
             "policy": "缺口保持 NaN，不插补不丢行；缺口日禁买、持仓禁卖、估值冻结在"
                       "缺口前最后有效 hfq 价，缺口延续到末端按退市规则清算"}
    if df.empty:
        df["hfq_gap"] = False
        return df, stats

    close_gap = df["close"].notna() & df["hfq_close"].isna()
    open_gap = df["open"].notna() & df["hfq_open"].isna()
    gap = close_gap | open_gap
    df["hfq_gap"] = gap.to_numpy()
    stats["rows"] = int(close_gap.sum())
    stats["open_rows"] = int(open_gap.sum())
    if gap.any():
        stats["codes"] = sorted(df.loc[gap, "code"].unique().tolist())
        stats["n_codes"] = len(stats["codes"])
    return df, stats


def load_panel(start: str, end: str, codes: list[str] | None = None,
               cache_dir: Path | None = None) -> Panel:
    """从 data/cache/daily_{年}.parquet 载入面板（只读清洗层，不碰 data/raw）。

    同时载入原始价与后复权价：原始价负责记账（真实股数/金额/费用），
    后复权价负责估值折算（分红再投）与退出变现，两者不可混用（3.3-2）。

    hfq 缺口在载入时由 `_mark_hfq_gap` **标注**（不插补），缺口格同时并入不可交易 mask，
    统计写进 panel.source["hfq_gap_rows"]（003R3 / 审计 B3-01）。
    """
    cache_dir = cache_dir or CACHE
    years = range(int(start[:4]), int(end[:4]) + 1)
    cols = ["code", "date", "open", "close", "pctChg", "isST",
            "suspended", "listed_days", "hfq_open", "hfq_close"]
    frames = []
    for y in years:
        f = cache_dir / f"daily_{y}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"缺 {f}，先跑 python -m src.clean --years ...")
        d = pd.read_parquet(f, columns=cols)
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if codes is not None:
            d = d[d["code"].isin(codes)]
        if len(d):
            frames.append(d)
    if not frames:
        raise ValueError(f"{start}~{end} 区间内没有任何行情，检查日期与股票池")
    df = pd.concat(frames, ignore_index=True)
    del frames

    df, gap_stats = _mark_hfq_gap(df)

    ipo = _load_ipo_dates(sorted(df["code"].unique().tolist()))
    ipo_series = df["code"].map(ipo)
    blk_buy, blk_sell = _blocked_arrays(
        df["code"], df["isST"].astype("int8"), df["date"], df["listed_days"],
        df["open"].to_numpy(dtype="float64"), df["close"].to_numpy(dtype="float64"),
        df["pctChg"].to_numpy(dtype="float64"), ipo_series)
    df["blk_buy"] = blk_buy
    df["blk_sell"] = blk_sell
    # 缺口格并入不可交易 mask：与停牌同级（买入侧与卖出侧都挡）
    df["hfq_gap"] = df["hfq_gap"].to_numpy(dtype=bool)
    df = df.drop(columns=["isST", "pctChg", "listed_days"])

    codes_all = sorted(df["code"].unique().tolist())
    dates_all = sorted(df["date"].unique().tolist())
    shape = (len(dates_all), len(codes_all))

    def _mat(name: str, dtype) -> np.ndarray:
        m = df.pivot(index="date", columns="code", values=name).reindex(
            index=dates_all, columns=codes_all)
        return m.to_numpy(dtype=dtype, na_value=np.nan)

    close_raw = _mat("close", "float64")
    no_row = np.isnan(close_raw)          # 当日无该股行情
    gap = np.nan_to_num(_mat("hfq_gap", "bool"), nan=False)
    panel = Panel(
        dates=dates_all,
        codes=codes_all,
        open_hfq=_mat("hfq_open", "float64"),
        close_hfq=_mat("hfq_close", "float64"),
        open_raw=_mat("open", "float64"),
        close_raw=close_raw,
        blocked_buy=np.nan_to_num(_mat("blk_buy", "bool"), nan=True) | no_row,
        blocked_sell=np.nan_to_num(_mat("blk_sell", "bool"), nan=True) | no_row,
        suspended=np.nan_to_num(_mat("suspended", "bool"), nan=True) | no_row,
        hfq_gap=gap & ~no_row,
        source={"requested_start": start, "requested_end": end,
                "rows": int(len(df)),
                "codes": len(codes_all), "dates": len(dates_all),
                "hfq_gap_rows": gap_stats,
                "ipo_source": "cache.db/stocks.ipoDate"},
    )
    return panel


# ---------------------------------------------------------------- 成本与配置
@dataclass
class Costs:
    """5.1 成本口径。比例均按**真实成交金额**计（R1 修订后金额为原始价口径），佣金有最低值。"""
    commission_rate: float = 0.00025   # 佣金 0.025%，双边
    min_commission: float = 5.0        # 单笔最低 5 元
    stamp_tax: float = 0.0005          # 印花税 0.05%，仅卖出
    transfer_fee: float = 0.00001      # 过户费 0.001%，双边
    slippage: float = 0.001            # 滑点 0.1%，可配；买入抬价、卖出压价

    def fees(self, amount: float, side: str) -> float:
        """return 该笔成交的总费用（佣金 + 过户费 [+ 印花税]）。"""
        if amount <= 0:
            return 0.0
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer = amount * self.transfer_fee
        stamp = amount * self.stamp_tax if side == "sell" else 0.0
        return commission + transfer + stamp


@dataclass
class Config:
    """全部可配参数，原样写进 result.json 的 params，保证"同一参数同一结果"可复算（R5）。"""
    start: str
    end: str
    init_cash: float = 1_000_000.0
    max_positions: int = 50            # 持仓只数上限（5.1 可配）
    max_weight: float = 0.10           # 单只权重上限（5.1 可配）
    lot_size: int = 100                # A 股一手 100 股
    seed: int | None = None            # 脚手架抽样种子（仅 harness 用，写进产物便于复算）
    n_hold: int | None = None          # 脚手架持股数（同上）
    costs: Costs = field(default_factory=Costs)


# ---------------------------------------------------------------- 结果结构
@dataclass
class BacktestResult:
    config: dict
    data: dict
    nav: pd.DataFrame          # date / equity / cash / positions / turnover
    trades: pd.DataFrame       # 每笔成交
    positions: pd.DataFrame    # 每日每只持仓股数（只含非零）
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)  # 未成交订单及原因
    signals_meta: dict = field(default_factory=dict)


def _equity_from_nhq(cash: float, nhq: np.ndarray, px: np.ndarray) -> float:
    """总权益 = 真实现金 + Σ(名义 hfq 股数 × hfq 价)。"""
    v = np.where(np.isnan(px), 0.0, px)
    return float(cash + np.dot(nhq, v))


def run_backtest(panel: Panel, signals: dict[str, dict[str, float]],
                 cfg: Config) -> BacktestResult:
    """事件驱动主循环。

    signals：{日期: {代码: 目标权重}}，日期必须是 panel 的交易日，权重和为 0~1。
             日期 d 的信号只可能在 d+1 开盘执行 —— 这是 3.3-1 时点对齐的硬实现。

    逐日流程：
      1) 开盘：执行**上一交易日**收盘生成的调仓单（T+1 成交，5.1）
      2) 收盘前：数据在区间内终止的持仓，按最后有效收盘价强制清算（总工 2026-11-10 裁决）
      3) 收盘：按 hfq 收益率折算市值估值，记录净值（停牌用最后有效价结转，3.3-3）
      4) 收盘后：读当日信号生成明日调仓单（此处只能看到 <= t 的数据）
    """
    n_d, n_c = panel.shape
    idx = {c: i for i, c in enumerate(panel.codes)}
    cash = float(cfg.init_cash)
    shares = np.zeros(n_c, dtype="float64")       # 真实股数（记账口径）
    nhq = np.zeros(n_c, dtype="float64")          # 名义 hfq 股数（估值口径，见模块 docstring）
    cost_basis = np.zeros(n_c, dtype="float64")   # 含买入费用的真实成本（用于逐笔盈亏）
    last_buy_day = np.full(n_c, -1, dtype="int64")   # T+1 限制：当日买入当日不可卖
    last_hfq = np.full(n_c, np.nan)               # 停牌结转用的最后有效 hfq 收盘价
    last_raw = np.full(n_c, np.nan)               # 停牌结转用的最后有效原始收盘价
    last_valid = _last_valid_index(panel.close_raw)   # 每列最后有行情的一行
    # hfq 最后有效行：缺口延续到数据末端的持仓要按它清算（003R3 / B3-01）
    last_hfq_valid = _last_valid_index(panel.close_hfq)

    hfq_close = panel.close_hfq
    raw_close = panel.close_raw
    close_mask = ~np.isnan(hfq_close)
    raw_mask = ~np.isnan(raw_close)
    gap = panel.hfq_gap

    # 台账计数（随 result.data 落地，可追溯）
    ledger = {"buy_blocked": 0, "sell_blocked": 0,
              "liquidation_day_blocked": 0, "frozen_position_days": 0,
              "liquidated_hfq_gap": 0, "liquidated_hfq_gap_value": 0.0}

    pending: dict[str, float] = {}                   # 明日开盘要调到的目标权重
    rows, trade_rows, pos_rows, drop_rows = [], [], [], []

    for i, date in enumerate(panel.dates):
        # ---------- 0) 当日到期的强制清算清单（必须在开盘交易**之前**算出来）
        #    003R4 / 审计 B4-01：原先步骤 1（开盘买入）与步骤 2（清算）同一迭代，
        #    清算日同时是"最后有效 hfq 日"，当日旗标仍是 False → 可以照常买入，
        #    收盘前又被强制清算，等于同一笔钱当天买当天卖（合成复现：
        #    2020-01-03 买 9,700 股 @10.0 → 同日清算 @10.2）。这类股票当日一律禁交易。
        due = _liquidation_due(shares, last_valid, last_hfq_valid, i, n_d)
        blocked = {j: "liquidating" for j in due}

        # ---------- 1) 开盘执行昨日信号（T+1）
        #    估值只用"<= i-1 日"的收盘价兜底，绝不读今日收盘价（3.3-1 防未来函数）
        if pending:
            cash = _execute(panel, i, pending, idx, cash, shares, nhq, cost_basis,
                            last_buy_day, last_hfq, last_raw, cfg,
                            trade_rows, drop_rows, date, blocked=blocked)

        # ---------- 2) 数据终止清算（区间内退市 / 长时间停牌至结束 / hfq 缺口延续到末端）
        #    只对"最后有效日 < 回测末日"的股票执行：末日仍有效的持仓按市值留在账上，
        #    那是期末估值，不是清算。
        #    取原始价与 hfq 两者的**较早**终止点：任一侧断供都无法继续按同一口径估值。
        for j, from_gap in due.items():
            px, hx = _exit_prices(float(raw_close[i, j]), float(hfq_close[i, j]),
                                  float(last_raw[j]), float(last_hfq[j]),
                                  panel.codes[int(j)], date)
            if from_gap:
                ledger["liquidated_hfq_gap"] += 1
                ledger["liquidated_hfq_gap_value"] += float(nhq[j] * hx)
            cash = _liquidate(j, panel.codes[int(j)], float(shares[j]), px, hx,
                              cash, shares, nhq, cost_basis, cfg,
                              trade_rows, date, reason="hfq_gap" if from_gap else "delisted")

        # ---------- 3) 收盘估值
        px_now = hfq_close[i]
        last_hfq = np.where(close_mask[i], px_now, last_hfq)
        last_raw = np.where(raw_mask[i], raw_close[i], last_raw)
        # 缺口期估值冻结在缺口前最后有效 hfq 价（last_hfq 不更新即自动冻结）
        equity = _equity_from_nhq(cash, nhq, last_hfq)
        rows.append({"date": date, "equity": equity, "cash": cash,
                     "positions": int((shares > 0).sum())})
        nz = np.nonzero(shares > 0)[0]
        for j in nz:
            pos_rows.append({"date": date, "code": panel.codes[j],
                             "shares": float(shares[j]),
                             "price_raw": float(last_raw[j]) if np.isfinite(last_raw[j]) else np.nan,
                             "value": float(nhq[j] * last_hfq[j])})
        if len(nz):
            ledger["frozen_position_days"] += int(gap[i, nz].sum())

        # ---------- 4) 收盘后读信号 → 明日开盘执行
        pending = signals.get(date, {}) or {}

    nav = pd.DataFrame(rows)
    nav["ret"] = nav["equity"].pct_change().fillna(0.0)
    trades = pd.DataFrame(trade_rows, columns=[
        "date", "code", "side", "shares", "price", "amount", "fees", "pnl",
        "book", "proceeds", "reason"])
    positions = pd.DataFrame(pos_rows, columns=["date", "code", "shares", "price_raw", "value"])
    dropped = pd.DataFrame(drop_rows, columns=["date", "code", "side", "shares", "reason"])
    # hfq 缺口台账（003R3 / B3-01，003R4 补卖侧）：计数取自 dropped，同一处口径，可反查
    if len(dropped):
        for side in ("buy", "sell"):
            ledger[f"{side}_blocked"] = int(((dropped["side"] == side)
                                             & (dropped["reason"] == "hfq_gap")).sum())
        ledger["liquidation_day_blocked"] = int((dropped["reason"] == "liquidating").sum())
    # 数据区间必须随产物落地（R5：写**真实**首末日，入参另存 requested_*）
    data = dict(panel.source)
    data.update({"start": panel.dates[0], "end": panel.dates[-1],
                 "dates": len(panel.dates), "codes": len(panel.codes),
                 "hfq_gap_ledger": ledger})
    return BacktestResult(
        config=cfg_to_dict(cfg), data=data, nav=nav,
        trades=trades, positions=positions, dropped=dropped,
        signals_meta={"signal_days": len(signals),
                      "dropped_orders": int(len(dropped)),
                      "note": "signals 日期 d 的信号只在 d+1 开盘执行（3.3-1）；"
                              "遇涨跌停/停牌/hfq 缺口/当日清算/T+1 未成交的订单不保留，"
                              "按 reason 逐笔记入 dropped；"
                              "区间内数据终止的持仓按最后有效收盘价强制清算（保守口径）；"
                              "hfq 缺口期禁买禁卖、估值冻结，缺口延续到末端按最后有效 hfq 价清算"})


def _execute(panel: Panel, i: int, target: dict[str, float], idx: dict[str, int],
             cash: float, shares: np.ndarray, nhq: np.ndarray, cost_basis: np.ndarray,
             last_buy_day: np.ndarray, last_hfq: np.ndarray, last_raw: np.ndarray,
             cfg: Config, trade_rows: list[dict], drop_rows: list[dict],
             date: str, blocked: dict[int, str] | None = None) -> float:
    """开盘调仓：按目标权重算出目标股数，先卖（降仓/清仓）后买（建仓/加仓）。

    约束顺序（5.1）：**当日清算禁交易** → 涨跌停不可成交 → 停牌跳过 →
    **hfq 缺口跳过（禁买禁卖）** → T+1 才能卖 → 只数/单只权重上限 → 资金约束。
    `blocked`（列号 → reason）由主循环当日清算清单给出：这些股票当天收盘前就要被强制清算，
    开盘再买等于当天买当天卖（审计 B4-01）。
    目标股数按**原始价**折算（R1）：qty = 目标市值 / (原始开盘价 × (1+滑点))。
    卖出/清仓走 `_sell`，按 **hfq 口径变现**（含分红再投层），费用按真实原始金额计（003R2）。
    开盘缺价时估值退用**上一交易日**收盘价，绝不读当日收盘价。
    """
    op = panel.open_raw[i]                       # 原始价：成交口径
    hfq_op = panel.open_hfq[i]                   # 复权价：估值口径
    px_val = np.where(np.isnan(hfq_op), last_hfq, hfq_op)
    equity = _equity_from_nhq(cash, nhq, px_val)

    # ---- 目标权重：只数上限 + 单只上限 + 归一化到 <=1
    items = sorted(((c, w) for c, w in target.items() if c in idx and w > 0),
                   key=lambda kv: (-kv[1], kv[0]))[: cfg.max_positions]
    weights = {c: min(float(w), cfg.max_weight) for c, w in items}
    total = sum(weights.values())
    if total > 1.0:
        weights = {c: w / total for c, w in weights.items()}

    # ---- 目标股数（原始价算，按手取整；真实金额才有物理意义）
    tgt_sh: dict[int, int] = {}
    for c, w in weights.items():
        j = idx[c]
        px = op[j]
        if np.isnan(px) or px <= 0:
            # 当日无价（停牌/缺行）：建仓单直接作废并记账，绝不静默跳过
            if shares[j] <= 0:
                drop_rows.append({"date": date, "code": c, "side": "buy",
                                  "shares": float("nan"),
                                  "reason": "suspended" if panel.suspended[i, j] else "no_price"})
            continue
        fill_px = px * (1.0 + cfg.costs.slippage)
        lot = cfg.lot_size
        tgt_sh[j] = int((w * equity) / fill_px // lot) * lot

    # ---- 卖出：持仓超出目标的减仓，目标里没有的清仓
    for j in np.nonzero(shares > 0)[0]:
        excess_sh = shares[j] - tgt_sh.get(int(j), 0)
        if excess_sh <= 1e-9:
            continue
        reason = None
        if blocked and int(j) in blocked:
            # 当日收盘前就要强制清算：开盘再卖少赚一遍滑点、再买就是当天买当天卖（B4-01）
            reason = blocked[int(j)]
        elif panel.hfq_gap[i, j]:
            # hfq 缺口期：卖出会按 hfq 变现，而缺口格没有可信因子 —— 一律冻结（B3-01）
            reason = "hfq_gap"
        elif panel.suspended[i, j] or np.isnan(op[j]):
            reason = "suspended"                       # 停牌跳过（3.3-3）
        elif panel.blocked_sell[i, j]:
            reason = "limit_down"                      # 跌停卖不出（5.1）
        elif last_buy_day[j] >= i:
            # 防御性代码：主循环"先开盘执行、后记 last_buy_day"的排序下，
            # 同一交易日不可能既有买入又有卖出，此分支物理不可达（R10/B-08）。
            # 保留是为了将来若调整执行顺序（如日内多次调仓）不至于静默破坏 T+1。
            reason = "t_plus_1"
        if reason:
            drop_rows.append({"date": date, "code": panel.codes[int(j)], "side": "sell",
                              "shares": float(excess_sh), "reason": reason})
            continue
        cash = _sell(int(j), panel.codes[int(j)], float(min(excess_sh, shares[j])),
                     float(op[j]), _hfq_price(hfq_op[j]),
                     cash, shares, nhq, cost_basis, cfg, trade_rows, date)

    # ---- 买入：低于目标的补足
    for c, w in weights.items():
        j = idx[c]
        want = tgt_sh.get(j, 0)
        if want - shares[j] < cfg.lot_size:
            continue
        reason = None
        if blocked and int(j) in blocked:
            # 当日清算股禁买（审计 B4-01：否则当天买当天被清算，多付两遍滑点与费用）
            reason = blocked[int(j)]
        elif panel.hfq_gap[i, j]:
            # 缺口期开仓会在 hfq 恢复后凭空多出一层幻影市值（B3-01 实测 6.41 倍）
            reason = "hfq_gap"
        elif panel.suspended[i, j] or np.isnan(op[j]):
            reason = "suspended"
        elif panel.blocked_buy[i, j]:
            reason = "limit_up"                        # 涨停买不进（5.1）
        if reason:
            drop_rows.append({"date": date, "code": c, "side": "buy",
                              "shares": float(want - shares[j]), "reason": reason})
            continue
        price = op[j] * (1.0 + cfg.costs.slippage)
        lot = cfg.lot_size
        # 单只权重上限：持仓市值必须用与 equity **同一口径**（nhq × hfq）折算。
        # 用原始价市值会低估老股票的权重（分红层不在里面），分红层大的个股可以突破上限。
        # hfq 缺格时退**冻结价**（last_hfq），绝不用原始价顶替（B3-01）。
        # 新增买入的金额按原始价计（成交口径），故 cap_val 之后再除 price。
        cap_val = cfg.max_weight * equity - _pos_value(nhq[j], hfq_op[j], last_hfq[j])
        qty = min(want - shares[j], cap_val // price)
        qty = int(qty // lot) * lot
        while qty > 0:                                  # 资金约束：手续费也要留够
            amount = qty * price
            fee = cfg.costs.fees(amount, "buy")
            if amount + fee <= cash + 1e-9:
                break
            qty -= lot
        if qty <= 0:
            drop_rows.append({"date": date, "code": c, "side": "buy",
                              "shares": float(want - shares[j]), "reason": "insufficient_cash"})
            continue
        amount = qty * price
        fee = cfg.costs.fees(amount, "buy")
        cash -= amount + fee
        cost_basis[j] += amount + fee
        # 名义 hfq 股数：成交金额 ÷ **滑点后**的复权成交价。
        # 用当日 hfq_open 会把滑点损失吞掉（市值等于成交金额，等于没扣滑点）；
        # 复权成交价 = hfq_open × (1+滑点)，与原始成交价一一对应。
        # 缺口格在上面已被拦下，走到这里 hfq 必然有效 —— 缺失即报错，
        # 绝不按原始价折股（那会在缺口结束后变出幻影层，审计 B3-01）。
        nhq[j] += amount / (_hfq_price(hfq_op[j]) * (1.0 + cfg.costs.slippage))
        shares[j] += qty
        last_buy_day[j] = i
        trade_rows.append({"date": date, "code": c, "side": "buy", "shares": float(qty),
                           "price": float(price), "amount": float(amount),
                           "fees": float(fee), "pnl": np.nan,
                           "book": np.nan, "proceeds": np.nan, "reason": ""})
    return cash


def _pos_value(nhq: float, hfq_px: float, frozen_hfq: float) -> float:
    """单只持仓的名义市值 = nhq × hfq 价。

    hfq 缺格（缺口期）时用**冻结价**（缺口前最后有效 hfq），绝不用原始价顶替：
    原始价与 hfq 差一个复权因子（sz.001914 实测 6.41 倍），拿原始价估值等于把
    该股市值凭空压低 6.41 倍 —— 老股价差 1300 倍的个股足以把总权益打成负数。
    """
    h = float(hfq_px)
    if not (np.isfinite(h) and h > 0):
        h = float(frozen_hfq)
    if not (np.isfinite(h) and h > 0):
        return 0.0
    return float(nhq) * h


def _hfq_price(hfq_px: float) -> float:
    """取估值/变现用的 hfq 价。**缺失即报错**（003R3 / 审计 B3-01）。

    旧版在此处"缺失就退原始价"，把缺口段的因子当成 1：缺口期买入的 nhq 在
    缺口结束后被真实因子放大，退出时变现成真金白银。现要求调用方先挡缺口，
    走到这里还缺 hfq，就是逻辑漏洞，直接抛错而不是挑一个数糊过去。
    """
    h = float(hfq_px)
    if not (np.isfinite(h) and h > 0):
        raise ValueError(
            f"hfq 价缺失（{hfq_px!r}）：缺口格应在 _execute 的 hfq_gap 分支被拦下，"
            f"走到取价说明不可交易掩码没生效 —— 拒绝按原始价降级（审计 B3-01）")
    return h


def _sell(j: int, code: str, qty: float, op: float, hfq_px: float, cash: float,
          shares: np.ndarray, nhq: np.ndarray, cost_basis: np.ndarray, cfg: Config,
          trade_rows: list[dict], date: str) -> float:
    """卖出：**原始价算费用，hfq 口径变现**（003R2 / 审计 B2-01）。

    账面价值 book  = nhq_sold × hfq_px                       # 含持有期累积的分红再投层
    变现额 proceeds = book × (1 − 滑点)                      # 卖出压价，与买入侧对称
    交易费用 fee    = Costs.fees(qty × 原始成交价, "sell")    # 真实原始金额计费
    现金 += proceeds − fee

    权益差恒等于 −(book×滑点 + fee)：持仓市值减少 book、现金增加 proceeds−fee，
    两者相抵后只剩滑点与费用，**分红层一分不蒸发**。下面的断言就守这条：
    若有人再把变现额改回"原始股数 × 原始价"，断言立刻红（旧实现实测 2,521 笔蒸发）。
    """
    slip = cfg.costs.slippage
    price = op * (1.0 - slip)                        # 原始价成交（滑点压价），费用基数
    old = shares[j]
    cash_before = cash
    pos_before = nhq[j] * hfq_px
    nhq_sold = nhq[j] * (qty / old) if old > 0 else 0.0
    book = nhq_sold * hfq_px
    proceeds = book * (1.0 - slip)
    amount = qty * price                             # 真实原始成交金额（换手/费用口径）
    fee = cfg.costs.fees(amount, "sell")

    cash += proceeds - fee
    avg = cost_basis[j] / old if old else 0.0
    pnl = proceeds - fee - avg * qty                 # 逐笔已实现盈亏（成本含买入费用）
    cost_basis[j] -= avg * qty
    shares[j] = old - qty
    nhq[j] = nhq[j] - nhq_sold                       # 与股数同步缩减，剩余仓位估值连续
    if shares[j] <= 1e-9:
        shares[j], cost_basis[j], nhq[j] = 0.0, 0.0, 0.0

    _assert_exit_conserved(cash_before, cash, pos_before, nhq[j] * hfq_px,
                           book, proceeds, fee, slip, code, date, "sell")
    trade_rows.append({"date": date, "code": code, "side": "sell", "shares": float(qty),
                       "price": float(price), "amount": float(amount),
                       "fees": float(fee), "pnl": float(pnl),
                       "book": float(book), "proceeds": float(proceeds), "reason": ""})
    return cash


def _assert_exit_conserved(cash_before: float, cash_after: float,
                           pos_before: float, pos_after: float, book: float,
                           proceeds: float, fee: float, slip: float,
                           code: str, date: str, side: str) -> None:
    """退出权益守恒断言（003R2 必加项，审计 B2-01）。

    用**可观测状态**的增量做校验，任一条破即抛 AssertionError：
      1) 现金增加额 == 变现额 − 费用；
      2) 该股持仓市值减少额 == 账面价值 book（分红层被完整卖掉，不多也不少）；
      3) 两者合计的权益变化 == −(book×滑点 + 费用)，即除滑点与费用外**零蒸发**。
    旧实现（003R 及以前）满足 (2) 却把 (1) 写成"原始股数 × 原始价"，
    于是权益差里凭空多出一笔分红层损失 —— (1) 会当场红。
    """
    tol = 1e-9 * max(1.0, abs(book) + abs(proceeds) + abs(fee) + abs(cash_before))
    if abs((cash_after - cash_before) - (proceeds - fee)) > tol:
        raise AssertionError(
            f"{date} {code} {side} 现金增加额 != 变现额−费用："
            f"Δcash={cash_after - cash_before!r} proceeds={proceeds!r} fee={fee!r}")
    if abs((pos_before - pos_after) - book) > tol:
        raise AssertionError(
            f"{date} {code} {side} 持仓市值减少额 != 账面价值："
            f"Δpos={pos_before - pos_after!r} book={book!r}")
    delta = (cash_after + pos_after) - (cash_before + pos_before)
    expect = -(book * slip + fee)
    if abs(delta - expect) > tol:
        raise AssertionError(
            f"{date} {code} {side} 退出权益不守恒：权益差 {delta!r}，"
            f"应为 −(滑点+费用)={expect!r}")
    if proceeds > book + tol:
        raise AssertionError(
            f"{date} {code} {side} 变现额超过账面价值：{proceeds!r} > {book!r}")


def _liquidate(j: int, code: str, qty: float, px_last: float, hfq_last: float,
               cash: float, shares: np.ndarray, nhq: np.ndarray,
               cost_basis: np.ndarray, cfg: Config, trade_rows: list[dict],
               date: str, reason: str = "delisted") -> float:
    """强制清算：按**最后有效收盘价**一次性卖出，回收现金。

    保守假设（总工 2026-11-10 裁决，R4）：退市股理论上可去老三板转让，但流动性极差、
    价格不可得，故按 0 计残余价值并以最后收盘价清算 —— 宁可**低估**残值，
    也不虚构一条无法执行的退出渠道。滑点照收（清仓也是市价单）。

    reason（003R3 / B3-01）：
      · "delisted" —— 原始价在区间内终止（退市/长期停牌至结束）；
      · "hfq_gap"  —— hfq 缺口延续到数据末端，按最后有效 hfq 价清算（估值不能再冻结下去）。
    两种都按 hfq 口径变现（含分红再投层），费用按真实原始金额计。
    """
    slip = cfg.costs.slippage
    price = px_last * (1.0 - slip)
    amount = qty * price
    fee = cfg.costs.fees(amount, "sell")
    cash_before = cash
    book = nhq[j] * hfq_last
    pos_before = book
    proceeds = book * (1.0 - slip)
    cash += proceeds - fee
    avg = cost_basis[j] / shares[j] if shares[j] else 0.0
    pnl = proceeds - fee - avg * qty
    nhq[j] = 0.0
    _assert_exit_conserved(cash_before, cash, pos_before, 0.0, book, proceeds,
                           fee, slip, code, date, "liquidate")
    trade_rows.append({"date": date, "code": code, "side": "liquidate",
                       "shares": float(qty), "price": float(price),
                       "amount": float(amount), "fees": float(fee), "pnl": float(pnl),
                       "book": float(book), "proceeds": float(proceeds),
                       "reason": reason})
    shares[j], cost_basis[j], nhq[j] = 0.0, 0.0, 0.0
    return cash


def cfg_to_dict(cfg: Config) -> dict:
    """完整参数（R5：start/end 也要落地，否则产物无法独立复算）。"""
    return asdict(cfg)


# ---------------------------------------------------------------- 自检脚手架
def harness_monthly_equal_weight(panel: Panel, n_hold: int = 300,
                                 seed: int = 42) -> dict:
    """**引擎自检脚手架，不是策略**（S3 才会写真正的策略）。

    固定抽 n_hold 只（固定随机种子，避免按代码排序带来的板块偏斜），每月最后一个交易日
    收盘后把当日可交易（有行、未停牌、未开盘涨停）的样本设为等权目标，下月同日再平衡。

    它的净值曲线应当与"等权全市场"基准同向且量级接近：引擎若偷看未来、漏记成本或
    涨跌停判断反了，两条线会立刻分叉。这是引擎的自检对照，不做任何选股判断。
    """
    rng = np.random.RandomState(seed)
    pool = np.arange(len(panel.codes))
    universe = rng.choice(pool, size=min(n_hold, len(pool)), replace=False) \
        if n_hold < len(pool) else pool
    sig: dict[str, dict[str, float]] = {}
    last_of_month: dict[str, int] = {}
    for i, d in enumerate(panel.dates):
        last_of_month[d[:7]] = i
    for i in last_of_month.values():
        ok = (~panel.suspended[i] & ~panel.blocked_buy[i]
              & ~np.isnan(panel.open_raw[i]))
        cand = universe[ok[universe]]
        if len(cand) == 0:
            continue
        w = 1.0 / len(cand)
        sig[panel.dates[i]] = {panel.codes[j]: w for j in cand}
    return sig


# ---------------------------------------------------------------- 产物
def git_commit() -> str:
    """取当前 HEAD 短哈希；取不到就写 unknown（不编造）。"""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_result(result: BacktestResult, metrics: dict, benchmarks: dict,
                 path: Path, extra: dict | None = None) -> Path:
    """落 result.json。缺 git_commit / 参数 / 数据区间 / 逐日净值 / 逐日持仓 任一字段即无效（R5）。"""
    payload = {
        "schema": "backtest-result/1",
        "git_commit": git_commit(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": result.config,
        "data": result.data,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "nav": result.nav.to_dict(orient="records"),
        "trades": result.trades.to_dict(orient="records"),
        "positions": result.positions.to_dict(orient="records"),
        "dropped": result.dropped.to_dict(orient="records"),
        "extra": extra or {},
    }
    for key in ("git_commit", "params", "data", "nav", "positions"):
        assert payload.get(key) not in (None, "", [], {}), f"result.json 缺 {key}"
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False + NaN→null：产物必须是严格 JSON，NaN 不是合法 JSON 字面量，
    # 下游 JS/前端读到 NaN 会直接崩，缺失就该写 null
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=1,
                              allow_nan=False), encoding="utf-8")
    return path


def _clean(o):
    """递归把 NaN/Inf 转 None，其余原样（严格 JSON 合规）。"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _data_upper_bound(cache_dir: Path | None = None) -> str:
    """cache 里现有日线覆盖到的最晚年份（用于把 --end 夹紧到数据边界）。

    必须在 load_panel 之前调用：load_panel 遇到不存在的年份会直接抛
    FileNotFoundError，而 --end 写超范围应该是"警告 + 夹紧"（R5），不是崩掉。
    """
    years = []
    for f in (cache_dir or CACHE).glob("daily_*.parquet"):
        stem = f.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            years.append(int(stem))
    return f"{max(years)}-12-31" if years else "2099-12-31"


def main(argv: list[str] | None = None) -> int:
    from . import metrics as M

    ap = argparse.ArgumentParser(description="S2 回测引擎示例运行（自检脚手架，非策略）")
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default=None, help="默认取数据最新日")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-hold", type=int, default=300, help="自检组合持股数（固定种子抽样）")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子（写进产物可复算）")
    args = ap.parse_args(argv)

    # --end 超出数据范围时夹紧并告警（R5），不静默截断、也不崩在缺文件上
    upper = _data_upper_bound()
    req_end = args.end or upper
    if args.end and args.end > upper:
        print(f"[backtest] 警告：--end {args.end} 超出数据覆盖范围（最晚 {upper}），已夹紧")
        req_end = upper

    panel = load_panel(args.start, req_end)
    real_end = panel.dates[-1]
    if req_end > real_end:
        print(f"[backtest] 提示：数据实际末日为 {real_end}（{req_end} 之前无更晚行情）")

    cfg = Config(start=panel.dates[0], end=real_end,
                 max_positions=max(args.n_hold, 50), max_weight=0.05,
                 seed=args.seed, n_hold=args.n_hold)
    panel.source["requested_end"] = args.end or real_end

    signals = harness_monthly_equal_weight(panel, n_hold=args.n_hold, seed=args.seed)
    result = run_backtest(panel, signals, cfg)

    bench = M.build_benchmarks(panel, result.nav["date"].tolist())
    metrics = M.compute_metrics(result.nav, result.trades, bench)
    out = (Path(args.out) / "result.json").resolve()
    write_result(result, metrics, bench["summary"], out,
                 extra={"signals": result.signals_meta,
                        "harness": "引擎自检脚手架：固定种子等权持有 N 只、月度再平衡，非策略",
                        "accounting": "原始价记账 + hfq 收益率折算市值 + 退出按 hfq 变现"
                                      "（总工 2026-11-10 裁决 / 003R2 修复 B2-01）",
                        "validity_note": VALIDITY_NOTE})
    print(f"[backtest] 区间 {panel.dates[0]} ~ {real_end}，{len(panel.codes)} 只，"
          f"净值 {result.nav['equity'].iloc[-1]:,.2f}")
    print(f"[backtest] 产物 {out}")
    print(M.format_card(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
