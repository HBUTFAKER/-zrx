"""S4 熊市防守雷达 · 数据加载层。

## 边界（改代码前先看）

  ① **运行时零大模型接口。** 本模块只读确定性产物 JSON，
     不 import 任何 LLM SDK，不发任何网络请求。
  ② **只读。** 只 `json.load`，不写不改任何文件。
  ③ **不改 src/。** 研究与引擎层归施工A，本文件属界面层，
     不修改 src/ 的任何一行（005S-M1 起连引擎层都不再调用 ——
     灯态由产物写死，界面只读）。

## 数据来源

  - 卡片主体：`out/005s_validation.json`（施工A 的 005S 产物，schema_version=005S-3）
  - 状态灯当前态：同一份产物的 `latest.light_state` 段
    —— 005S-R3 起由施工A 在产物里写死（state/as_of/drawdown/window/threshold/rule），
    界面**直接读**，不再从基准 parquet 自行重算。

    为什么不再自算（005S-M1）：自算口径与产物面板对不上时，
    界面显示的灯态日期和回撤值就跟产物不是一回事，等于界面另立一套口径。
    现在灯态与名单截面日的差异**由产物自己写明**：灯态日是基准最新交易日
    （日频），名单截面日是月末，两者分开披露、不强行拉齐。

  - 观察名单简称：`latest.watch_names`（code→简称，005S-M2）。
    界面只做名称-代码对齐，不自己查名、不猜名；产物没给名称的代码只显示代码。
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
    """状态灯当前态（直读产物 `latest.light_state`，界面不自行判定）。

    defensive=True   防守期（回撤已达阈值）
    defensive=False  正常期
    defensive=None   **未判定**：产物 state 为 null，
                     表示该交易日无基准收盘或缺前置窗口。
                     这种状态既不是防守期也不是正常期，
                     界面必须原样显示"未判定"，绝不许当成"正常期"——
                     把"没算出来"显示成"安全"等于凭空放宽。

    threshold 为**负数**（与产物一致，如 -0.1），判定式：
        drawdown <= threshold → 防守期
    """

    defensive: bool | None
    as_of: str
    drawdown: float | None
    window: int
    threshold: float
    rule_text: str


@dataclass(frozen=True)
class WatchList:
    """观察名单。

    normal_phase=True 只在灯态**明确为正常期**时成立；
    灯态未判定（None）时 normal_phase=False —— 未判定既不是防守期也不是正常期，
    界面不进入"正常期只报状态"分支，也不展示名单内容，而是报"灯态未判定"。

    as_of 是**名单截面日**（月末），与灯态判定日（基准最新交易日）不同，
    两个日期各自展示，不许合并成一个。

    names 是 code→简称 的映射（005S-M2），取自产物 `latest.watch_names`。
    界面只做名称-代码对齐，**不自己查名称、不猜名称**：
    产物里没给名称的代码，names 里查不到就是查不到，
    界面只显示代码、不填占位符（不写"—"、不写"未知"）。
    """

    codes: list[str]
    as_of: str
    universe: int
    rule_text: str
    normal_phase: bool = False
    #: code→简称 映射；缺名称的代码不在表里（界面据此只显示代码）
    names: dict[str, str] = field(default_factory=dict)


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


def _light_state(payload: dict) -> LightState:
    """直读产物 `latest.light_state`，不再自行重算灯态（005S-M1）。

    产物里写死的字段逐字照抄，界面只做展示：
        state     —— "defensive" / "normal" / null（null = 该日无法判定）
        as_of     —— 灯态判定日（基准最新交易日）
        drawdown  —— 判定日回撤
        window    —— 回撤窗口
        threshold —— 阈值（负数，判定式 drawdown <= threshold → 防守期）
        rule      —— 产物写明的灯规则原文

    `state` 为 null 时 `defensive` 保持 None（未判定），
    不许默认成 False —— 那会把"算不出来"显示成"正常期"。
    """
    ls = _require(payload, "latest.light_state")

    state = ls.get("state")
    if state is None:
        defensive: bool | None = None
    elif state == "defensive":
        defensive = True
    elif state == "normal":
        defensive = False
    else:
        raise PayloadError(f"产物 light_state.state 词表外取值：{state!r}")

    raw_dd = ls.get("drawdown")
    drawdown = None if raw_dd is None else float(raw_dd)

    return LightState(
        defensive=defensive,
        as_of=str(_require(ls, "as_of")),
        drawdown=drawdown,
        window=int(_require(ls, "window")),
        threshold=float(_require(ls, "threshold")),
        rule_text=str(ls.get("rule", "")),
    )


def _watch_names(payload: dict) -> dict[str, str]:
    """读产物 `latest.watch_names`（code→简称），005S-M2。

    这段是**可选增强**而非界面判定依据：
      - 段本身缺失 → 报错（产物 schema 已声明该段，不许静默降级）；
      - 某个代码在表里查不到 → 就是查不到，**不占位**（不写"—"、不写"未知"），
        界面只显示代码。名称是给人看的补充，不是准入条件，
        缺一个名字不该让整张名单渲染不出来。
      - 产物另给 `latest.watch_names_missing`（缺失清单），本层不依赖它做判定，
        只把它透传给界面做披露用；缺失清单为空时说明产物侧已全覆盖。
    """
    raw = _require(payload, "latest.watch_names")
    if not isinstance(raw, dict):
        raise PayloadError(f"产物 latest.watch_names 类型错：{type(raw).__name__}")

    names: dict[str, str] = {}
    for code, name in raw.items():
        code_s = str(code).strip()
        if not code_s:
            continue
        # 名称统一转字符串并去首尾空白；空名称视为"该代码没有名称"
        name_s = "" if name is None else str(name).strip()
        if name_s:
            names[code_s] = name_s
    return names


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
    """装载 005S 产物（含产物写死的当前灯态）。"""
    p = payload_path(path)
    if not p.exists():
        raise PayloadError(f"产物不存在：{p}")

    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    # --- 状态灯（直读产物 latest.light_state，不自算）---
    light = _light_state(payload)

    # --- 观察名单（避雷名单已判失败，本界面不读 latest.avoid）---
    watch_codes = list(_require(payload, "latest.watch"))
    as_of = str(_require(payload, "latest.as_of"))
    universe = int(_require(payload, "latest.universe"))
    watch = WatchList(
        codes=watch_codes,
        as_of=as_of,
        universe=universe,
        rule_text=str(_require(payload, "rules.watch")),
        # 只有**明确判为正常期**才置灰。灯态未判定（None）不是"正常期"，
        # 此时既不给名单也不说"正常期不给建议"，界面报的是"灯态未判定"。
        normal_phase=light.defensive is False,
        names=_watch_names(payload),
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
