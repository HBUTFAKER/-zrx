"""配置层：业主画像（owner）+ 系统参数（system），同文件两段。

设计约束（业主拍板）：
  · 一个文件 `data/config.json`，两段分离 —— 业主改 owner 段不会碰到系统口径，
    系统口径（权重/阈值/过滤器）改动必须留下痕迹，便于"dev 定型后 holdout 不许回头改"。
  · `--init` 生成带说明的模板，手改生效。JSON 本身不支持注释，所以用 `_readme` /
    `_comment` 这类下划线键写说明 —— 不引 JSON5/YAML，多一个解析器就多一处坑。
  · `disclaimer` 与 `multi_user` 只留位不实现（业主拍板：将来决定卖了再做）。
    留 None 而不是编造默认文案，避免"没实现的东西看起来像实现了"。

005R 变更（分位带分散组合）：
  · `system.n_hold` / `system.max_weight` **删除**。持仓只数不再写死，改由
    `system.portfolio.band`（前 N% 等权）逐截面算出 k，单票权重恒为 1/k。
    老配置里还留着这两个键会直接报错 —— 静默忽略等于让人以为"按 5 只跑"，
    实际按 100 只跑，这类"看着像生效"的配置比报错危险得多。
  · 新增 `system.validation`：band 的定档段与分年验证年份。
  · `system.backtest.init_cash` 从 100 万提到 5000 万：等权 100+ 只时，
    一手（100 股）取整会吃掉 10% 以上的仓位；现金拖累会**系统性惩罚大 k 的档位**，
    让选档结论失真。这个数只影响回测模拟精度，与业主真实预算无关
    （真实预算的可行性另由卡上的 `budget` 段算）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "config.json"

TARGET_TYPES = ("beat_benchmark", "annual_return")

#: 005R 起由 portfolio.band 决定持仓只数，这两个键属于旧口径
LEGACY_SYSTEM_KEYS = ("n_hold", "max_weight")

#: 系统参数默认值。权重默认 bp 0.6 / 反转 0.4（业主拍板·方案 B）。
#: 阈值全部来自 dev 段实测分布，依据写在 `_comment` 里，改阈值请连注释一起改。
DEFAULT_SYSTEM: dict = {
    "_comment": "系统参数：dev 段定型后写死，holdout 段不许回头改。",
    "weights": {
        "_comment": "综合得分 = bp 分位 × w_bp + 反转分位 × w_rev；两项必须和为 1。"
                    "实测 dev 段 IC：0.5→+0.0752 / 0.6→+0.0707 / 0.7→+0.0645，"
                    "反转权重越高组合 IC 越高（见产物 weight_sensitivity）。"
                    "005R 未改这一层 —— 综合分排序能力为正（t≈4），问题出在持仓形态。",
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
        "exclude_limit_up_comment": "只剔截面日（T 日）开盘涨停的票。T+1 是否买得进由"
                                    "回测执行层判断，不许拿 T+1 盘面参加 T 日筛选（未来函数）。",
    },
    "portfolio": {
        "_comment": "持仓形态：综合分前 band 比例的全部股票等权持有（005R 业主拍板）。"
                    "band 由 validation.band_fit_end 之前的定档段扫描确定，"
                    "选完冻结；holdout 与分年数据不得参与选档。",
        "mode": "band",
        "band": 0.05,
        "band_comment": "选中档（定档结果写入此处）。中位只数 = ceil(band × 候选数)，"
                        "候选数中位约 2160 只 → 5% 约 110 只。"
                        "改这里等于换策略口径，必须重跑定档段扫描，不要在 holdout 上调。",
        "band_candidates": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
        "band_candidates_comment": "候选档。业主指定至少要含 top10%/15%/20%/30%，"
                                   "另加 3%/5%/8% 以便落在 30~200 只的预期规模内。",
        "expected_positions": [30, 200],
        "expected_positions_comment": "预期持仓只数区间，只用于卡上提示，不构成门槛（不调档迁就）。",
    },
    "gate": {
        "_comment": "硬门槛（004C）：forced / 全段组合绝对回撤 / 五年滚动稳定性，任一不过输出 fallback。",
        "dev_min_excess": 0.0,
        "dev_comment": "仅保留旧流程粗筛兼容项；004C 终验统一对 H00300。",
        "max_portfolio_mdd": 0.15,
        "min_avg_excess": 0.02,
        "min_positive_windows": 4,
    },
    "validation": {
        "_comment": "24个月训练 / 12个月检验；因子与反转主窗口逐窗口冻结。",
        "method": "rolling_24_12",
        "band_fit_end": "2020-12-31",
        "band_fit_end_comment": "定档段到此为止（含）。2018-01-02~2020-12-31 共 36 个截面。",
        "years": ["2021", "2022", "2023", "2024", "2025"],
        "years_comment": "滚动检验段为2021~2025，每段完整12个月；2025为完整年度。",
        "train_months": 24,
        "test_months": 12,
        "factor_report": "out/004c_c1_factors.json",
    },
    "advice_card": {
        "_comment": "产物形态（005S 降级）：radar_005s = 状态灯 + 避雷/观察名单，不做买入推荐。"
                    "005R 旧卡与 004C 验证产物都是作废留档，本产物不得写到它们头上。",
        "mode": "radar_005s",
        "output": "out/005s_validation.json",
        "legacy_outputs": ["out/advice_card.json", "out/004c_c2_validation.json"],
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
        "_comment": "组合回测参数。init_cash 用 5000 万：等权 100~200 只时一手取整的"
                    "现金拖累需压到 1% 量级，否则大 k 档位会被模拟误差误杀。"
                    "该数字与业主真实预算无关。",
        "init_cash": 50000000.0,
        "lot_size": 100,
    },
}

DEFAULT_OWNER: dict = {
    "_comment": "业主画像：只影响卡上文案与资金换算，不影响选股口径。手改即可，改完重跑 CLI 生效。",
    "budget": 100000.0,
    "budget_comment": "预算金额（元）。用于卡上「每只买多少钱 / 够不够一手 / 最小可行预算」，"
                      "不回测。",
    "max_drawdown": -0.15,
    "max_drawdown_comment": "最大可接受回撤（负数）。测试期 -15%。只在卡上作对照，"
                            "005R 起不再用它反推单票上限（分散到 100+ 只后单票恒为 1/k）。",
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

    # ---------- 旧口径拦截 ----------
    legacy = [k for k in LEGACY_SYSTEM_KEYS if k in system]
    if legacy:
        raise ConfigError(
            f"system 里还有旧口径键 {legacy}：005R 起持仓只数由 portfolio.band 决定"
            f"（前 N% 等权，单票恒为 1/k），n_hold / max_weight 已删除。"
            f"请删掉这些键后重跑 `python -m src.advisor --init` 对照新模板。"
            f"（静默忽略会让配置看起来生效、实际没生效，比报错更危险）")

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

    # ---------- system.weights ----------
    w = system.get("weights") or {}
    wb, wr = w.get("bp"), w.get("reversal")
    for name, val in (("bp", wb), ("reversal", wr)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not (0.0 <= val <= 1.0):
            raise ConfigError(f"system.weights.{name} 必须在 [0, 1]，当前 {val!r}")
    if abs(float(wb) + float(wr) - 1.0) > 1e-6:
        raise ConfigError(f"system.weights 两项必须和为 1，当前 {wb} + {wr} = {float(wb)+float(wr)}")

    # ---------- system.portfolio ----------
    port = system.get("portfolio") or {}
    if port.get("mode") != "band":
        raise ConfigError(f"system.portfolio.mode 目前只支持 'band'，当前 {port.get('mode')!r}")
    band = port.get("band")
    if not isinstance(band, (int, float)) or isinstance(band, bool) or not (0.0 < band <= 1.0):
        raise ConfigError(f"system.portfolio.band 必须在 (0, 1]，当前 {band!r}")
    cands = port.get("band_candidates")
    if not isinstance(cands, list) or len(cands) < 4:
        raise ConfigError(f"system.portfolio.band_candidates 至少 4 档（业主验收要求），"
                          f"当前 {cands!r}")
    for x in cands:
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not (0.0 < x <= 1.0):
            raise ConfigError(f"band_candidates 每档必须在 (0, 1]，当前含 {x!r}")
    if list(cands) != sorted(set(cands)):
        raise ConfigError(f"band_candidates 必须严格升序且不重复，当前 {cands!r}")

    # ---------- system.gate ----------
    gate = system.get("gate") or {}
    if not isinstance(gate.get("dev_min_excess"), (int, float)):
        raise ConfigError("system.gate.dev_min_excess 必须是数字")
    for key, lo, hi in (("max_portfolio_mdd", 0.0, 1.0),
                        ("min_avg_excess", -1.0, 1.0)):
        value = gate.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not lo <= value <= hi:
            raise ConfigError(f"system.gate.{key} 必须在 [{lo}, {hi}]，当前 {value!r}")
    positive_windows = gate.get("min_positive_windows")
    if not isinstance(positive_windows, int) or isinstance(positive_windows, bool) or positive_windows < 1:
        raise ConfigError("system.gate.min_positive_windows 必须是正整数")
    # ---------- system.validation ----------
    val = system.get("validation") or {}
    method = val.get("method")
    if method == "rolling_24_12":
        if "period_max_dd_worse" in gate or "holdout_check_direction" in gate:
            raise ConfigError("004C 已删除 period_max_dd_worse / holdout_check_direction，请按新模板更新")
        if val.get("train_months") != 24 or val.get("test_months") != 12:
            raise ConfigError("rolling_24_12 固定为 24 个月训练 / 12 个月检验")
        if val.get("band_fit_end") != "2020-12-31":
            raise ConfigError("004C 定档段必须截止 2020-12-31")
        years = val.get("years")
        if not isinstance(years, list) or any(not (isinstance(y, str) and len(y) == 4 and y.isdigit())
                                              for y in years):
            raise ConfigError("rolling_24_12 检验年份必须是四位年份字符串")
        if years != ["2021", "2022", "2023", "2024", "2025"]:
            raise ConfigError("rolling_24_12 检验年份必须是 2021~2025 五个完整年度")
        if not isinstance(val.get("factor_report"), str) or not val["factor_report"]:
            raise ConfigError("system.validation.factor_report 必须指定 C1 JSON 产物")
    elif method == "yearly_freeze":
        ddw = gate.get("period_max_dd_worse")
        if not isinstance(ddw, (int, float)) or isinstance(ddw, bool) or not (0.0 <= ddw <= 1.0):
            raise ConfigError(f"system.gate.period_max_dd_worse 必须在 [0, 1]，当前 {ddw!r}")
        years = val.get("years")
        if not isinstance(years, list) or not years:
            raise ConfigError("system.validation.years 必须是非空列表（如 ['2022','2023']）")
        for y in years:
            if not (isinstance(y, str) and len(y) == 4 and y.isdigit()):
                raise ConfigError(f"system.validation.years 每项必须是四位年份字符串，当前含 {y!r}")
    elif method == "rolling":
        raise ConfigError("system.validation.method 目前只实现 'rolling_24_12'，请使用该固定滚动口径")
    else:
        raise ConfigError(f"system.validation.method 未实现，当前 {method!r}")

    # ---------- system.advice_card（005S 产物形态） ----------
    card = system.get("advice_card") or {}
    modes = {"radar_005s", "fallback_004c", "advice_card_005r"}
    if card.get("mode") not in modes:
        raise ConfigError(f"system.advice_card.mode 必须是 {sorted(modes)} 之一，"
                          f"当前 {card.get('mode')!r}")
    out = card.get("output")
    if not isinstance(out, str) or not out.endswith(".json"):
        raise ConfigError(f"system.advice_card.output 必须是 .json 路径，当前 {out!r}")
    legacy = card.get("legacy_outputs") or []
    if not isinstance(legacy, list) or any(not isinstance(x, str) for x in legacy):
        raise ConfigError("system.advice_card.legacy_outputs 必须是字符串列表")
    if out in legacy:
        raise ConfigError(f"system.advice_card.output 不许指向作废留档 {out!r}"
                          "（005R/004C 产物只读，覆盖等于毁掉审计追溯）")

    # ---------- veto / filters ----------
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

    bt = system.get("backtest") or {}
    cash = bt.get("init_cash")
    if not isinstance(cash, (int, float)) or isinstance(cash, bool) or cash <= 0:
        raise ConfigError(f"system.backtest.init_cash 必须是正数，当前 {cash!r}")
    if not isinstance(bt.get("lot_size", 100), int) or bt.get("lot_size", 100) < 1:
        raise ConfigError("system.backtest.lot_size 必须是 >=1 的整数")


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
