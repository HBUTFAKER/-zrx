"""005S 防守雷达测试（状态灯 / 两张名单 / 达标统计 / 配置护栏）。

硬断言，不放宽：
  1. **无未来函数**：把 T 之后的基准价格整体污染，T 及之前的灯态必须逐位相同。
  2. **阈值边界**：回撤 -10.0% 亮、-9.0% 不亮（不许四舍五入放宽）。
  3. **名单方向**：观察名单取低波高分位、避雷名单取高波低分位，两张名单不许重叠。
  4. **达标线**：占比 70% 恰过、69% 不过（不四舍五入凑过）。
  5. **产物不许覆盖作废留档**：config 指向 advice_card.json 必须报错。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as C  # noqa: E402
from src.radar_005s import (AVOID_SPEC, WATCH_FACTORS, export_monthly_lists,  # noqa: E402
                            latest_block, light_snapshot, month_flags, names_for,
                            rolling_drawdown, score_lists, state_light, summarize)


# ---------------------------------------------------------------- 1. 无未来函数

def test_state_light_ignores_future_prices():
    rng = np.random.default_rng(11)
    n = 200
    px = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, n)),
                   index=[f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)])
    base = state_light(px)

    corrupted = px.copy()
    corrupted.iloc[102:] *= 0.4          # T=102 起整体砸盘（T=101 当日价不许动）
    after = state_light(corrupted)

    # 第 101 日的灯只用 ≤101 的价格，必须一模一样
    assert np.array_equal(base.iloc[:102].to_numpy(), after.iloc[:102].to_numpy())
    # 且污染确实生效（否则这条测试等于没测）
    assert after.iloc[102:].any()


def test_rolling_drawdown_uses_right_aligned_window():
    """窗口右端就是当日：当天创新高 → 回撤 0，不会因为"以后会跌"提前变负。"""
    px = pd.Series([10.0] * 25 + [9.0], index=[f"d{i}" for i in range(26)])
    dd = rolling_drawdown(px, 20)
    assert dd.iloc[24] == 0.0
    assert abs(dd.iloc[25] - (-0.10)) < 1e-12


# ---------------------------------------------------------------- 2. 阈值边界

def test_state_light_threshold_boundary():
    base = [100.0] * 25
    lit = state_light(pd.Series(base + [90.0], index=[f"d{i}" for i in range(26)]))
    assert bool(lit.iloc[-1]) is True, "-10.0% 必须亮"
    not_lit = state_light(pd.Series(base + [91.0], index=[f"d{i}" for i in range(26)]))
    assert bool(not_lit.iloc[-1]) is False, "-9.0% 不许亮"


def test_month_flags_any_single_day_lights_the_month():
    dates = ["2022-01-27", "2022-01-28", "2022-02-07", "2022-02-08"]
    light = pd.Series([False, True, False, False], index=dates)
    flags = month_flags(dates, light)
    assert flags["2022-01"]["defensive"] is True
    assert flags["2022-01"]["first_defensive"] == "2022-01-28"
    assert flags["2022-01"]["month_end_state"] is True     # 月末那天也亮
    assert flags["2022-02"]["defensive"] is False
    assert flags["2022-01"]["n_days"] == 2


# ---------------------------------------------------------------- 3. 名单方向

def _mats(n: int = 300) -> dict[str, np.ndarray]:
    idx = np.arange(n, dtype="float64")
    arr = lambda v: v.reshape(1, n)  # noqa: E731
    # 三个因子都"索引越大越好"；reversal_60 同步升序，便于断定避雷名单应落在另一端
    return {"lowvol_20": arr(idx / n), "lowvol_60": arr(idx / n),
            "turn_stab_60": arr(idx / n), "reversal_60": arr(idx / n)}


def test_score_lists_direction_and_size():
    mats = _mats(300)
    mask = np.ones(300, dtype=bool)
    mask[0] = False                      # 唯一一只被否决的票不许出现在任何名单
    out = score_lists(mats, 0, mask, top_n=100)

    assert len(out["watch"]) == 100 and len(out["avoid"]) == 100
    assert 0 not in out["watch"] and 0 not in out["avoid"]
    # 观察 = 三个"越大越好"因子的高分位 → 索引偏大；避雷 = 低分位 → 索引偏小
    assert out["watch"].min() > out["avoid"].max()
    assert not set(out["watch"].tolist()) & set(out["avoid"].tolist())
    assert out["n_watch_pool"] == 299 and out["n_avoid_pool"] == 299


def test_score_lists_skips_nan_instead_of_imputing():
    mats = _mats(300)
    mats["lowvol_60"][0, :50] = np.nan          # 50 只缺 60 日窗口
    out = score_lists(mats, 0, np.ones(300, dtype=bool), top_n=100)
    assert out["n_watch_pool"] == 250           # 缺值的不插补，直接不进候选


def test_watch_and_avoid_spec_cover_004c_factors():
    """观察用低波+换手稳；避雷必须含高波/换手不稳/过去跌得多三路。"""
    assert set(WATCH_FACTORS) == {"lowvol_20", "lowvol_60", "turn_stab_60"}
    assert {n for n, _, _ in AVOID_SPEC} == {"lowvol_20", "turn_stab_60", "reversal_60"}
    assert abs(sum(w for _, _, w in AVOID_SPEC) - 1.0) < 1e-12


# ---------------------------------------------------------------- 4. 达标线

def _watch_rows(n_ok: int, n: int = 10) -> list[dict]:
    rows = [{"excess": 0.01, "strategy_mdd": -0.10, "bench_mdd": -0.12} for _ in range(n_ok)]
    rows += [{"excess": -0.01, "strategy_mdd": -0.10, "bench_mdd": -0.12}
             for _ in range(n - n_ok)]
    return rows


def test_summarize_watch_pass_rate_boundary():
    assert summarize(_watch_rows(7), kind="watch")["passed"] is True       # 7/10 恰过
    assert summarize(_watch_rows(6), kind="watch")["passed"] is False      # 6/10 不过
    assert summarize(_watch_rows(7), kind="watch")["pass_rate"] == 0.7


def test_summarize_watch_requires_shallower_drawdown():
    """超额为正但回撤更深 → 当月不合格（两条都要）。"""
    rows = _watch_rows(7) + [{"excess": 0.02, "strategy_mdd": -0.30, "bench_mdd": -0.12}] * 3
    out = summarize(rows, kind="watch")
    assert out["n_months"] == 13 and out["n_qualified"] == 7
    assert out["passed"] is False                  # 7/13 < 70%


def test_summarize_avoid_needs_underperformance():
    rows = [{"strategy_ret": -0.05, "bench_ret": -0.02}] * 8
    rows += [{"strategy_ret": 0.03, "bench_ret": 0.01}] * 2
    out = summarize(rows, kind="avoid")
    assert out["n_qualified"] == 8 and out["pass_rate"] == 0.8 and out["passed"] is True


# ---------------------------------------------------------------- 5. 配置护栏

def _cfg_with(**card):
    c = C.default_config()
    c["system"]["advice_card"].update(card)
    return c


@pytest.mark.parametrize("card,msg", [
    ({"mode": "buy_list"}, "advice_card.mode"),
    ({"output": "out/advice_card.json"}, "不许指向作废留档"),
    ({"output": "out/004c_c2_validation.json"}, "不许指向作废留档"),
    ({"output": "out/radar.txt"}, ".json"),
])
def test_config_rejects_bad_advice_card(tmp_path, card, msg):
    p = tmp_path / "c.json"
    p.write_text(__import__("json").dumps(_cfg_with(**card)), encoding="utf-8")
    with pytest.raises(C.ConfigError) as e:
        C.load_config(p)
    assert msg in str(e.value)


def test_default_advice_card_mode_is_radar():
    s = C.default_config()["system"]["advice_card"]
    assert s["mode"] == "radar_005s"
    assert s["output"] == "out/005s_validation.json"
    assert "out/advice_card.json" in s["legacy_outputs"]


# ---------------------------------------------------------------- 6. 逐月名单明细（005S-R2）

def test_export_monthly_lists_long_table(tmp_path):
    """名单明细长表：一票一行、字段齐、名次连续、不含任何已算好的收益列。"""
    rows = [{"month": "2024-01", "formed_at": "2024-01-31", "list_type": "watch",
             "rank": 1, "code": "000001.SZ", "n_universe": 1800},
            {"month": "2024-01", "formed_at": "2024-01-31", "list_type": "watch",
             "rank": 2, "code": "000002.SZ", "n_universe": 1800},
            {"month": "2024-01", "formed_at": "2024-01-31", "list_type": "avoid",
             "rank": 1, "code": "600000.SH", "n_universe": 1800}]
    path = tmp_path / "lists.parquet"
    meta = export_monthly_lists(rows, path)

    back = pd.read_parquet(path)
    assert list(back.columns) == ["month", "formed_at", "list_type", "rank", "code", "n_universe"]
    assert len(back) == 3 and meta["rows"] == 3
    assert meta["months"] == 1 and meta["first_month"] == meta["last_month"] == "2024-01"
    assert meta["rows_by_type"] == {"avoid": 1, "watch": 2}
    # 明细里不许夹带收益字段 —— 审计要拿原始名单自己算
    assert not {"excess", "strategy_ret", "bench_ret"} & set(back.columns)
    assert back.loc[back["list_type"] == "watch", "rank"].tolist() == [1, 2]


def test_monthly_list_export_rejects_empty():
    """空输入也能落表（月份无名单时不炸），但行数为 0、不编造月份。"""
    meta = export_monthly_lists([], Path(__import__("tempfile").gettempdir()) / "empty_lists.parquet")
    assert meta["rows"] == 0 and meta["months"] == 0 and meta["first_month"] is None


# ---------------------------------------------------------------- 6. latest 段契约（005S-R3）

def test_light_snapshot_states_and_date():
    """快照的 state 必须与阈值同口径，as_of 就是判定日。"""
    idx = [f"2024-01-{i:02d}" for i in range(1, 32)]
    down = pd.Series([100.0] * 30 + [85.0], index=idx)
    snap = light_snapshot(state_light(down), rolling_drawdown(down), idx[-1])
    assert snap["state"] == "defensive"
    assert snap["as_of"] == "2024-01-31"
    assert snap["threshold"] == -0.1 and snap["window"] == 20
    assert snap["drawdown"] <= -0.1

    mild = pd.Series([100.0] * 30 + [97.0], index=idx)
    snap2 = light_snapshot(state_light(mild), rolling_drawdown(mild), idx[-1])
    assert snap2["state"] == "normal" and snap2["drawdown"] == pytest.approx(-0.03)


def test_light_snapshot_is_point_in_time():
    """T 日快照只用 ≤T 的价格：污染 T 之后的价格，快照必须一模一样。"""
    idx = [f"2024-02-{i:02d}" for i in range(1, 29)]
    px = pd.Series(100.0 * np.cumprod(1.0 + np.sin(np.arange(28)) * 0.01), index=idx)
    t = idx[20]
    before = light_snapshot(state_light(px), rolling_drawdown(px), t)

    polluted = px.copy()
    polluted.iloc[21:] *= 0.3
    after = light_snapshot(state_light(polluted), rolling_drawdown(polluted), t)
    assert before == after


def test_light_snapshot_missing_date_is_not_normal():
    """缺数据 → state=None，不许默认成 normal（那等于凭空放宽验收线）。"""
    idx = [f"2024-03-{i:02d}" for i in range(1, 11)]
    px = pd.Series([100.0] * 10, index=idx)
    snap = light_snapshot(state_light(px), rolling_drawdown(px), "2024-03-11")
    assert snap["state"] is None and snap["drawdown"] is None
    assert snap["as_of"] == "2024-03-11" and snap["note"]


def test_names_for_keeps_missing_separate():
    """名称映射只用真名；查不到的进 missing，不用占位符顶替。"""
    codes = ["sh.600000", "sz.000001", "sh.600999"]
    mapping, missing = names_for(codes, {"sh.600000": "浦发银行", "sz.000001": "平安银行"})
    assert mapping == {"sh.600000": "浦发银行", "sz.000001": "平安银行"}
    assert missing == ["sh.600999"]
    # cache.db 缺失时 load_names 返回空 dict：mapping 为空、missing 全量，仍可解释
    empty_map, empty_missing = names_for(codes, {})
    assert empty_map == {} and empty_missing == codes


@pytest.mark.skipif(not (ROOT / "out" / "005s_validation.json").exists(),
                    reason="产物尚未生成")
def test_product_latest_block_contract():
    """产物字段契约：施工B 直接读这两个字段，不许再自己重算。"""
    import json
    d = json.loads((ROOT / "out" / "005s_validation.json").read_text(encoding="utf-8"))
    if d.get("schema_version") != "005S-3":
        pytest.skip(f"磁盘上是旧产物 {d.get('schema_version')}，新字段等重出后再校验")
    lt = d["latest"]
    ls = lt["light_state"]
    assert ls["state"] in ("defensive", "normal")
    # 灯按基准最新交易日：不早于名单截面日，两个日期各自独立
    assert ls["as_of"] >= lt["as_of"]
    assert (ls["state"] == "defensive") == (ls["drawdown"] <= ls["threshold"] + 1e-12)
    assert set(lt["watch_names"]).issubset(set(lt["watch"]))
    assert set(lt["watch_names"]) | set(lt["watch_names_missing"]) == set(lt["watch"])
    assert all(isinstance(v, str) and v for v in lt["watch_names"].values())


def test_latest_block_assembles_light_and_names():
    """latest 段组装：灯按基准最新日、名单按月末日，两个日期分开写。"""
    idx = [f"2024-04-{i:02d}" for i in range(1, 29)]
    px = pd.Series([100.0] * 27 + [88.0], index=idx)
    latest = {"date": idx[-3], "n_universe": 2100,
              "watch_codes": ["sh.600000", "sz.000001", "sh.600999"],
              "avoid_codes": ["sh.600001"]}
    blk = latest_block(latest, state_light(px), rolling_drawdown(px),
                       {"sh.600000": "浦发银行", "sz.000001": "平安银行"})
    assert blk["as_of"] == "2024-04-26" and blk["universe"] == 2100
    assert blk["light_state"]["state"] == "defensive"
    # 灯取基准最新交易日，允许晚于名单截面日；不许强行拉齐
    assert blk["light_state"]["as_of"] == "2024-04-28" != blk["as_of"]
    assert blk["watch"] == latest["watch_codes"] and blk["avoid"] == ["sh.600001"]
    assert blk["watch_names"] == {"sh.600000": "浦发银行", "sz.000001": "平安银行"}
    assert blk["watch_names_missing"] == ["sh.600999"]
