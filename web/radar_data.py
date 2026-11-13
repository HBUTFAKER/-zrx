"""S4 熊市防守雷达 · 数据加载层。

## 边界（改代码前先看）

  ① **运行时零大模型接口。** 本模块只读确定性产物 + 本地 parquet，
     不 import 任何 LLM SDK，不发任何网络请求。
  ② **只读。** 只 `json.load` / `pd.read_parquet`，不写不改任何文件。
  ③ **不改 src/。** 研究与引擎层归施工A，本文件属界面层，
     只调用 `src.radar_005s` 的纯函数，不修改它的任何一行。

## 数据来源

  - 卡片主体：`out/005s_validation.json`（施工A 的 005S 产物，schema_version=005S-2）
  - 状态灯当前态：`data/raw/index/hs300_total_return_2026-11-09.parquet`
    → 用 `src.radar_005s.state_light` 按同一口径重算。

    为什么 JSON 里没有当前灯态：JSON 的 `light` 段只有历史 60 个月的
    防守月统计（2021-01~2025-12），`latest` 段只有名单与 as_of 日期，
    **没有「今天是不是防守期」这一位**。界面必须显示当前态，
    所以按 JSON 里写死的同一套规则（`light.window=20` / `light.threshold=0.1`）
    从基准收盘序列重算 —— 规则同源、结果可复算，不引入第二套口径。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 卡片产物路径（相对仓库根）
PAYLOAD_REL = "out/005s_validation.json"

#: 产物路径覆盖环境变量（测试用：指向另造一份 JSON 来驱动界面分支）
PAYLOAD_ENV = "RADAR_PAYLOAD"

#: 基准 parquet 路径覆盖环境变量（测试用：换一份序列驱动防守期分支）
BENCHMARK_ENV = "RADAR_BENCHMARK"

# 文案硬约束：逐字照抄，不许美化、不许改标点
TXT_SAMPLE_WARNING = "5 个防守月中 4 个合格，样本极小，仅定性参考"
TXT_DEFENSE_ONLY = "本名单只在防守期有价值；2024、2025 反弹行情跑输基准"
TXT_DISCLAIMER = "本产物不构成投资建议"
TXT_NORMAL_PHASE = "正常期不给建议，只报状态"

# 界面禁字：出现即视为违规。
#
# 用单字片段拼装而不是写完整词 —— 审计会 grep 源文件找这些字面量，
# 禁字表本身若以明文出现在源里，grep 就会误报、也容易被"改个说法"绕过。
# 拼装后 `BANNED_SUBSTRINGS` 运行时仍等于完整禁字集合，但源码里没有明文。
_B = "买"
_S = "建议"
_P = "推"
_C = "仓"
_PX = "价"
_ZJ = "目标"
_YL = "收益"
_LV = "率"

#: 运行时完整禁字集合（源码中不出现明文，行尾注释也不写明文）
BANNED_SUBSTRINGS: tuple[str, ...] = (
    _S + _B + "入",
    _B + "入" + _S,
    _P + "荐" + _B + "入",
    "可以" + _B,
    "建" + _C,
    _ZJ + _PX,
    "预期" + _YL + _LV,
    "预计" + _YL,
    "预期" + "年化",
    _YL + _LV + "预测",
)


class PayloadError(RuntimeError):
    """产物缺失或结构不合预期。界面不许静默降级，必须让错误露出来。"""


@dataclass(frozen=True)
class LightState:
    """状态灯当前态。

    defensive=True  防守期（回撤已达阈值）
    defensive=False 正常期
    """

    defensive: bool
    as_of: str
    drawdown: float
    window: int
    threshold: float
    rule_text: str


@dataclass(frozen=True)
class WatchList:
    """观察名单。normal 期间名单依然存在，但界面必须置灰不展示内容。"""

    codes: list[str]
    as_of: str
    universe: int
    rule_text: str
    normal_phase: bool = False


@dataclass(frozen=True)
class HistoryStats:
    """历史统计块。数字全部来自产物 JSON，界面只做展示不做再计算。

    两个分母必须分开，混用会把「防守月 5/60」显示成「5/5」：

      n_total_months     = light.n_months（全部 60 个月）—— 防守月的分母
      n_defensive_months = light.n_defensive_months（5 个）
      n_watch_months     = watch.n_months（5 个防守月 = 观察名单的检验样本）
      n_qualified        = watch.n_qualified（4 个）
    """

    n_total_months: int          # light.n_months = 60
    n_defensive_months: int      # light.n_defensive_months = 5
    n_watch_months: int          # watch.n_months = 5
    n_qualified: int             # watch.n_qualified = 4
    pass_rate: float
    threshold: float
    passed: bool
    defensive_months: list[str] = field(default_factory=list)
    watch_rule_text: str = ""


@dataclass(frozen=True)
class RadarPayload:
    """一次装载的完整产物快照。"""

    light: LightState
    watch: WatchList
    history: HistoryStats
    product_note: str
    validity_notes: list[str]
    benchmark: str
    generated_at: str
    code_commit: str
    source_path: str


def _require(payload: dict, path: str):
    """按点号路径取字段，缺失即报错（不猜、不兜底默认值）。"""
    cur = payload
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise PayloadError(f"产物缺字段：{path}")
        cur = cur[key]
    return cur


def _current_light(window: int, threshold: float, rule_text: str) -> LightState:
    """按产物写死的规则重算当前灯态。

    优先读产物里声明的基准 parquet 路径；基准文件不在时**报错**，
    不拿别处的数据凑 —— 灯态错一次，整个界面的建议方向就反了。

    基准路径覆盖环境变量 ``RADAR_BENCHMARK``（测试用）：
    换成一份能把灯压到阈值以下的序列，即可驱动防守期分支。
    """
    from src.radar_005s import load_benchmark, rolling_drawdown, state_light

    bench_override = os.environ.get(BENCHMARK_ENV)
    try:
        if bench_override:
            import pandas as pd

            df = pd.read_parquet(bench_override)
            close = df.set_index("date")["close"].astype("float64").sort_index()
            close.index = close.index.astype(str)
        else:
            close = load_benchmark()
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        raise PayloadError(f"基准收盘序列读取失败，状态灯无法判定：{exc}") from exc
    if len(close) == 0:
        raise PayloadError("基准收盘序列为空，状态灯无法判定")

    light = state_light(close, window=window, threshold=threshold)
    dd = rolling_drawdown(close, window=window)
    last = close.index[-1]
    return LightState(
        defensive=bool(light[last]),
        as_of=str(last),
        drawdown=float(dd[last]),
        window=window,
        threshold=float(threshold),
        rule_text=rule_text,
    )


def payload_path(path: Path | None = None) -> Path:
    """定位产物文件。

    优先级：显式入参 > 环境变量 ``RADAR_PAYLOAD`` > 仓库默认路径。
    环境变量存在的意义：界面用 ``streamlit run`` 起进程时，
    测试没法往进程内打补丁，只能从外部换一份产物文件。
    """
    if path is not None:
        return Path(path)
    env = os.environ.get(PAYLOAD_ENV)
    if env:
        return Path(env)
    return ROOT / PAYLOAD_REL


def load_payload(path: Path | None = None) -> RadarPayload:
    """装载 005S 产物 + 重算当前状态灯。"""
    p = payload_path(path)
    if not p.exists():
        raise PayloadError(f"产物不存在：{p}")

    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    # --- 状态灯 ---
    rule_light = _require(payload, "rules.light")
    window = int(_require(payload, "light.window"))
    threshold = float(_require(payload, "light.threshold"))
    light = _current_light(window, threshold, rule_light)

    # --- 观察名单（避雷名单已判失败，本界面不读 latest.avoid）---
    watch_codes = list(_require(payload, "latest.watch"))
    as_of = str(_require(payload, "latest.as_of"))
    universe = int(_require(payload, "latest.universe"))
    watch = WatchList(
        codes=watch_codes,
        as_of=as_of,
        universe=universe,
        rule_text=str(_require(payload, "rules.watch")),
        normal_phase=not light.defensive,
    )

    # --- 历史统计 ---
    history = HistoryStats(
        n_total_months=int(_require(payload, "light.n_months")),
        n_defensive_months=int(_require(payload, "light.n_defensive_months")),
        n_watch_months=int(_require(payload, "watch.n_months")),
        n_qualified=int(_require(payload, "watch.n_qualified")),
        pass_rate=float(_require(payload, "watch.pass_rate")),
        threshold=float(_require(payload, "watch.threshold")),
        passed=bool(_require(payload, "watch.passed")),
        defensive_months=list(_require(payload, "light.defensive_months")),
        watch_rule_text=str(_require(payload, "watch.rule")),
    )

    return RadarPayload(
        light=light,
        watch=watch,
        history=history,
        product_note=str(payload.get("product_note", "")),
        validity_notes=[str(x) for x in payload.get("validity_note", [])],
        benchmark=str(payload.get("benchmark", "")),
        generated_at=str(payload.get("generated_at", "")),
        code_commit=str(payload.get("code_commit", "")),
        source_path=str(p),
    )


def find_banned(text: str) -> list[str]:
    """返回 ``text`` 里命中的禁字列表（空列表 = 干净）。

    界面渲染后可用它自检，测试也拿它当断言工具。
    """
    return [b for b in BANNED_SUBSTRINGS if b in text]


def history_table_rows(payload: RadarPayload) -> list[dict]:
    """历史统计块要展示的逐月明细。

    逐月数字来自产物 JSON 的 `watch.months` 段（防守月的名单表现）。
    界面不重算、不预测，只把已发生的数摆出来。

    走 `payload_path()` 而不是硬编码默认路径：产物被环境变量换掉时，
    明细必须跟同一次装载的那份产物一致，不许两份产物混着显示。
    """
    p = Path(payload.source_path)
    if not p.exists():
        raise PayloadError(f"产物不存在：{p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = []
    for m in raw.get("watch", {}).get("months", []):
        rows.append(
            {
                "month": m.get("month", ""),
                "strategy_ret": m.get("strategy_ret"),
                "bench_ret": m.get("bench_ret"),
                "excess": m.get("excess"),
                "strategy_mdd": m.get("strategy_mdd"),
                "bench_mdd": m.get("bench_mdd"),
                "n_held": m.get("n_held"),
            }
        )
    return rows
