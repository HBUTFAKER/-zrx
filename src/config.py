"""配置层：业主画像（owner）+ 系统参数（system），同文件两段。

设计约束（业主拍板）：
  · 一个文件 `data/config.json`，两段分离 —— 业主改 owner 段不会碰到系统口径，
    系统口径（权重/阈值/过滤器）改动必须留下痕迹，便于"dev 定型后 holdout 不许回头改"。
  · `--init` 生成带说明的模板，手改生效。JSON 本身不支持注释，所以用 `_readme` /
    `_comment` 这类下划线键写说明 —— 不引 JSON5/YAML，多一个解析器就多一处坑。
  · `disclaimer` 与 `multi_user` 只留位不实现（业主拍板：将来决定卖了再做）。
    留 None 而不是编造默认文案，避免"没实现的东西看起来像实现了"。

为什么位置上限要用回撤参数算（`position_cap`）：
  单票上限不是拍脑袋的 20%，而是从业主能忍的回撤反推 —— 假设单票极端跌幅 50%
  （连续跌停 / 退市整理），那么"单票全损正好用满回撤预算"对应的上限就是
  2 × |最大回撤|。容忍 15% → 单票上限 30%；容忍 10% → 20%。这样卡上的仓位数字
  和业主自己的问卷答案是同一件事，能当场解释，不用事后找理由。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "config.json"

#: 单票极端跌幅假设（连续跌停 / 退市整理 / 黑天鹅）。A 股个股腰斩是常态量级，
#: 取 50% 是保守值；若取 30% 会让上限虚高（3.3×回撤容忍），失去约束意义。
SINGLE_NAME_CRASH = 0.50

TARGET_TYPES = ("beat_benchmark", "annual_return")

#: 系统参数默认值。权重默认 bp 0.6 / 反转 0.4（业主拍板·方案 B）。
#: 阈值全部来自 dev 段实测分布，依据写在 `_comment` 里，改阈值请连注释一起改。
DEFAULT_SYSTEM: dict = {
    "_comment": "系统参数：dev 段定型后写死，holdout 段不许回头改。",
    "weights": {
        "_comment": "综合得分 = bp 分位 × w_bp + 反转分位 × w_rev；两项必须和为 1。"
                    "实测 dev 段 IC：0.5→+0.0752 / 0.6→+0.0707 / 0.7→+0.0645，"
                    "反转权重越高组合 IC 越高（见产物 weight_sensitivity）。",
        "bp": 0.60,
        "reversal": 0.40,
    },
    "veto": {
        "_comment": "质量否决项：命中即剔除，不计入候选。阈值依据见各项注释。",
        "roe_ttm_min": 0.0,
        "roe_ttm_comment": "亏损企业一律否决（业主拍板）。dev 段实测截面内 roe_ttm<0 "
                           "占比中位 16.1%（区间 2.9%~26.6%）。",
        "debt_ratio_industry_pct_max": 0.90,
        "debt_ratio_comment": "行业内资产负债率分位上限。实测 >90% 分位占截面 10.33%、"
                              ">95% 占 5.35%；取 90% 即否决行业内最贵的十分之一杠杆。"
                              "组内不足 5 只不计算分位、不否决（小组分位是噪声）。",
    },
    "filters": {
        "_comment": "过滤器：ST / 停牌 / hfq 缺口 / 上市天数 沿用 factors.cross_section_mask "
                    "口径，本层只追加流动性与涨停两项。",
        "min_amount": 50000000.0,
        "min_amount_comment": "成交额下限（元）。dev 段实测截面分位：10%=1800万 / "
                              "20%=2800万 / 30%=3890万 / 50%=7180万。取 5000 万（≈20~30 分位）"
                              "平均保留 61.7% 的截面样本；1 亿只保留 40.9%，会把小盘股整体切掉。",
        "min_listed_days": 60,
        "exclude_limit_up": True,
        "exclude_limit_up_comment": "截面日或次日开盘涨停的票不可选（次日买不进）。",
    },
    "n_hold": 5,
    "max_weight": 0.30,
    "max_weight_comment": "单票权重上限。默认 30% = 2 × |owner.max_drawdown|，"
                          "含义是「单票全损（-50%）正好用满业主的回撤预算」。"
                          "改 owner.max_drawdown 时请同步改这里。",
    "gate": {
        "_comment": "硬门槛：两项全过才出建议卡，任一不过输出 fallback。",
        "dev_min_excess": 0.0,
        "holdout_max_dd_worse": 0.0,
        "holdout_check_direction": True,
    },
    "holdout": {
        "_comment": "holdout 段边界，冻结不许改（factors.DEV_END / HOLDOUT_START 同源）。",
        "start": "2025-11-07",
        "end": "2026-11-06",
    },
    "costs": {
        "_comment": "成本口径沿用 backtest.Costs 默认（佣金 0.025% 双边、印花税 0.05% 卖出、"
                    "过户费 0.001%、滑点 0.1%）。改这里等于改回测结论，慎动。",
        "commission_rate": 0.00025,
        "min_commission": 5.0,
        "stamp_tax": 0.0005,
        "transfer_fee": 0.00001,
        "slippage": 0.001,
    },
    "backtest": {
        "_comment": "组合回测参数。init_cash 用 100 万：5 只票、单票 20%，一手成本可忽略。",
        "init_cash": 1000000.0,
        "lot_size": 100,
    },
}

DEFAULT_OWNER: dict = {
    "_comment": "业主画像：只影响仓位与文案，不影响选股口径。手改即可，改完重跑 CLI 生效。",
    "budget": 100000.0,
    "budget_comment": "预算金额（元）。仅用于卡上「建议投入多少钱」的换算，不回测。",
    "max_drawdown": -0.15,
    "max_drawdown_comment": "最大可接受回撤（负数）。测试期 -15%。单票上限由它反推。",
    "holding_period": "monthly",
    "holding_period_comment": "持仓周期：monthly / weekly / long_term。当前策略只有月频口径，"
                              "填别的值会照实提示「策略口径与画像不一致」，不偷偷改策略。",
    "existing_positions": [],
    "existing_positions_comment": "已有持仓，只存档不生成调仓建议（业主拍板）。"
                                  "格式 [{\"code\": \"sh.600000\", \"shares\": 100, "
                                  "\"cost\": 8.5}]",
    "target": {"type": "beat_benchmark", "value": None},
    "target_comment": "目标：{\"type\": \"beat_benchmark\"} 跑赢基准，或 "
                      "{\"type\": \"annual_return\", \"value\": 0.15} 具体年化。",
    "experience": "beginner",
    "experience_comment": "投资经验：beginner / intermediate / advanced。影响风险提示的措辞深度。",
    "disclaimer": None,
    "disclaimer_comment": "【预留位·未实现】免责声明文案。将来做多用户/售卖时填，当前恒 null。",
    "multi_user": None,
    "multi_user_comment": "【预留位·未实现】多用户字段位。当前恒 null，读到非 null 会提示未实现。",
}


class ConfigError(ValueError):
    """配置不合法。宁可停工也不许带着坏配置出建议卡。"""


def default_config() -> dict:
    """默认配置（深拷贝，改返回值不会污染模块常量）。"""
    return {"owner": copy.deepcopy(DEFAULT_OWNER),
            "system": copy.deepcopy(DEFAULT_SYSTEM)}


def _strip_comments(d: dict) -> dict:
    """去掉 `_comment` / `_readme` 这类说明键，只留真参数。"""
    out = {}
    for k, v in d.items():
        if k.endswith("_comment") or k.endswith("_readme"):
            continue
        out[k] = _strip_comments(v) if isinstance(v, dict) else v
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    """override 覆盖 base；只对同名字典递归，其余整体替换（列表不合并）。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict:
    """读 config.json，与默认值深合并后校验。

    文件不存在 → 抛 ConfigError 并提示先 `--init`。**不自动创建**：静默生成一份
    默认配置会让"业主到底填了什么"这件事失去痕迹。
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}\n先生成模板：python -m src.advisor --init")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p} 不是合法 JSON：{e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} 顶层必须是对象（含 owner / system 两段）")

    cfg = _deep_merge(default_config(), raw)
    cfg["_path"] = str(p)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    """校验关键参数。越界一律抛错，不做"容错兜底"。"""
    owner = cfg.get("owner") or {}
    system = cfg.get("system") or {}
    if not isinstance(owner, dict) or not isinstance(system, dict):
        raise ConfigError("config 必须含 owner / system 两个对象段")

    # ---------- owner ----------
    budget = owner.get("budget")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        raise ConfigError(f"owner.budget 必须是正数，当前 {budget!r}")
    mdd = owner.get("max_drawdown")
    if not isinstance(mdd, (int, float)) or isinstance(mdd, bool) or not (-1.0 < mdd < 0.0):
        raise ConfigError(f"owner.max_drawdown 必须是 (-1, 0) 的负数（如 -0.15），当前 {mdd!r}")
    target = owner.get("target") or {}
    if target.get("type") not in TARGET_TYPES:
        raise ConfigError(f"owner.target.type 必须是 {TARGET_TYPES} 之一，当前 {target.get('type')!r}")
    if target.get("type") == "annual_return":
        v = target.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ConfigError("owner.target.type=annual_return 时 value 必须是数字（如 0.15）")
    if not isinstance(owner.get("existing_positions", []), list):
        raise ConfigError("owner.existing_positions 必须是列表")

    # ---------- system ----------
    w = system.get("weights") or {}
    wb, wr = w.get("bp"), w.get("reversal")
    for name, val in (("bp", wb), ("reversal", wr)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not (0.0 <= val <= 1.0):
            raise ConfigError(f"system.weights.{name} 必须在 [0, 1]，当前 {val!r}")
    if abs(float(wb) + float(wr) - 1.0) > 1e-6:
        raise ConfigError(f"system.weights 两项必须和为 1，当前 {wb} + {wr} = {float(wb)+float(wr)}")

    n_hold = system.get("n_hold")
    if not isinstance(n_hold, int) or isinstance(n_hold, bool) or n_hold < 1:
        raise ConfigError(f"system.n_hold 必须是 >=1 的整数，当前 {n_hold!r}")
    cap = system.get("max_weight")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not (0.0 < cap <= 1.0):
        raise ConfigError(f"system.max_weight 必须在 (0, 1]，当前 {cap!r}")
    if cap * n_hold < 1.0 - 1e-9:
        raise ConfigError(
            f"system.max_weight={cap} × n_hold={n_hold} < 1：单票上限太紧，"
            f"凑不满 100% 仓位（至少要 {1.0/n_hold:.4f}）")

    veto = system.get("veto") or {}
    dp = veto.get("debt_ratio_industry_pct_max")
    if not isinstance(dp, (int, float)) or isinstance(dp, bool) or not (0.0 < dp <= 1.0):
        raise ConfigError(f"system.veto.debt_ratio_industry_pct_max 必须在 (0, 1]，当前 {dp!r}")
    if not isinstance(veto.get("roe_ttm_min"), (int, float)):
        raise ConfigError("system.veto.roe_ttm_min 必须是数字")

    flt = system.get("filters") or {}
    if not isinstance(flt.get("min_amount"), (int, float)) or flt["min_amount"] < 0:
        raise ConfigError("system.filters.min_amount 必须是非负数")
    if not isinstance(flt.get("min_listed_days", 60), int):
        raise ConfigError("system.filters.min_listed_days 必须是整数")


def position_cap(max_drawdown: float, *, crash: float = SINGLE_NAME_CRASH) -> float:
    """由回撤容忍度反推单票权重上限：cap = |max_drawdown| / crash。

    含义：单票极端跌幅 crash（默认 50%）时，这笔亏损正好等于业主能忍的回撤。
    容忍 15% → 30%。同时上封 40%（再高就谈不上分散）。
    """
    cap = abs(float(max_drawdown)) / float(crash)
    return float(min(0.40, max(0.01, round(cap, 4))))


def save_template(path: str | Path | None = None, *, force: bool = False) -> Path:
    """写一份带说明的 config.json 模板。

    已存在且 force=False 时**不覆盖** —— 业主手改过的配置被模板冲掉是不可逆的损失。
    """
    p = Path(path) if path else DEFAULT_PATH
    if p.exists() and not force:
        raise ConfigError(
            f"{p} 已存在，拒绝覆盖（业主手改的配置会被冲掉）。确需重置加 --force")
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    cfg = {"_readme": "本文件两段：owner=业主画像（可手改）/ system=系统参数（dev 定型后写死）。"
                      "下划线开头的键是说明，程序会忽略。生成：python -m src.advisor --init，"
                      "出卡：python -m src.advisor",
           **cfg}
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def system_params(cfg: dict) -> dict:
    """取出"去掉说明键"的 system 段（写进产物用，避免产物里全是注释）。"""
    return _strip_comments(cfg.get("system") or {})


def owner_profile(cfg: dict) -> dict:
    """取出"去掉说明键"的 owner 段。"""
    return _strip_comments(cfg.get("owner") or {})
