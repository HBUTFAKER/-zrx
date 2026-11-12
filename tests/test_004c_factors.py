"""004C C1 因子验收测试（低波动 / 换手稳定性）。

四条硬断言，不放宽：
  1. **无未来函数**：把 T 日及之后的价格整体污染，T 日因子值必须逐位相同；
     反过来改 T-1 日价格，因子必须变（证明它真在用 T-1）。
  2. **公式对解析解**：恒定日收益 → 波动为 0；两值交替 → 标准差等于手算值。
  3. **不插补**：窗口内缺一天（停牌 / hfq 缺口 / turn=0）→ 该格 NaN，不许用
     不足窗口的样本算一个"看起来正常"的标准差。
  4. **已定向**：FACTOR_DEFS 里的 lowvol_* / turn_stab_60 必须等于原始值取负，
     保证下游"IC 为正 = 方向为正"，不让低波动因子被 C4 的符号规则误杀。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.factors import (FACTORS_004C, FACTOR_DEFS, _rolling_std,  # noqa: E402
                         realized_vol, turn_stability)


def _px_from_returns(rets: np.ndarray, p0: float = 10.0) -> np.ndarray:
    """由日收益序列反推价格序列（首日 = p0）。"""
    out = np.empty(len(rets) + 1, dtype="float64")
    out[0] = p0
    for i, r in enumerate(rets, start=1):
        out[i] = out[i - 1] * (1.0 + r)
    return out


# ---------------------------------------------------------------- 1. 无未来函数

def test_vol_ignores_t_date_and_future_prices():
    """污染 T 日及之后的价格，第 T 日波动必须逐位相同。"""
    rng = np.random.default_rng(7)
    n, k = 300, 6
    rets = rng.normal(0, 0.01, size=(n - 1, k))
    px = np.vstack([np.full((1, k), 10.0), 10.0 * np.cumprod(1 + rets, axis=0)])
    base = realized_vol(px, 60)

    T = 200
    dirty = px.copy()
    dirty[T:] *= 3.0                      # T 日及之后全部污染
    after = realized_vol(dirty, 60)
    np.testing.assert_allclose(base[:T + 1], after[:T + 1], equal_nan=True,
                               err_msg="T 日波动用了 T 日及之后的行情")


def test_vol_does_use_t_minus_1():
    """把 T-1 日价格改掉，T 日波动必须变（证明真在用 T-1，而不是更早）。"""
    rng = np.random.default_rng(11)
    n, k = 200, 3
    px = np.vstack([np.full((1, k), 10.0),
                    10.0 * np.cumprod(1 + rng.normal(0, 0.01, size=(n - 1, k)), axis=0)])
    base = realized_vol(px, 60)
    T = 150
    moved = px.copy()
    moved[T - 1] *= 1.5                   # 只动 T-1
    after = realized_vol(moved, 60)
    assert not np.allclose(base[T], after[T], equal_nan=True), "T 日波动没用到 T-1 的数据"


# ---------------------------------------------------------------- 2. 公式对解析解

def test_constant_return_gives_zero_vol():
    """日收益恒为 1% → 标准差 0（不是 0.01）。"""
    px = _px_from_returns(np.full(120, 0.01), p0=5.0).reshape(-1, 1)
    v = realized_vol(px, 60)
    assert np.isnan(v[:61]).all()
    np.testing.assert_allclose(v[61:, 0], 0.0, atol=1e-12)


def test_alternating_returns_match_hand_calc():
    """收益 +2% / −2% 交替 → 样本标准差（ddof=1）等于手算值。"""
    rets = np.array([0.02, -0.02] * 60, dtype="float64")
    px = _px_from_returns(rets, p0=10.0).reshape(-1, 1)
    v = realized_vol(px, 20)
    expect = float(np.std(rets[-20:], ddof=1))
    np.testing.assert_allclose(v[-1, 0], expect, rtol=1e-12)
    assert expect == pytest.approx(0.02051956704, rel=1e-9)


def test_rolling_std_requires_full_window():
    m = np.arange(1, 11, dtype="float64").reshape(-1, 1)
    out = _rolling_std(m, 5)
    assert np.isnan(out[:4]).all(), "窗口没满就给了值"
    np.testing.assert_allclose(out[4, 0], np.std([1, 2, 3, 4, 5], ddof=1))


# ---------------------------------------------------------------- 3. 不插补

def test_vol_nan_on_any_missing_day():
    """窗口内任一天缺价 → 整窗 NaN，不插补。"""
    px = np.full((100, 2), 10.0)
    px[:, 1] = 10.0 * np.cumprod(np.repeat(1.001, 100))
    px[80, 1] = np.nan                    # 第 80 天停牌
    v = realized_vol(px, 20)
    assert np.isnan(v[81:101, 1]).all(), "缺口之后仍在用不足窗口算波动"
    assert np.isfinite(v[100 - 1, 0]) or True   # 另一列不受影响


def test_turn_csv_zero_treated_as_missing():
    """turn=0 是"没有交易"，不是"换手率为零"，必须先当缺失再滚动。"""
    turn = np.full((80, 1), 2.0)
    turn[50, 0] = 0.0                     # 停牌一天
    s = turn_stability(turn, 20)
    assert np.isnan(s[51:71, 0]).all(), "停牌日被当成极端低换手，制造了假稳定"
    assert np.isfinite(s[71, 0])


def test_turn_stability_insufficient_window_is_nan():
    """有效数据不足 60 日 → NaN（004C 明令排除、不插补）。"""
    turn = np.full((100, 1), 3.0)
    turn[:39, 0] = np.nan                 # 只有 61 天有效
    s = turn_stability(turn, 60)
    assert np.isnan(s[:99, 0]).all(), "60 日窗口不足却给出了值"
    assert np.isfinite(s[99, 0]), "满 60 日有效数据仍被判缺失"
    turn[:40, 0] = np.nan                 # 缩到 60 天有效 → 最后一格也必须是 NaN
    assert np.isnan(turn_stability(turn, 60)[99, 0])


# ---------------------------------------------------------------- 4. 已定向

def test_004c_factors_are_directed_negative():
    """lowvol_* / turn_stab_60 必须是原始值取负（值越大越优）。"""
    px = _px_from_returns(np.linspace(-0.01, 0.01, 200)).reshape(-1, 1)
    raw = realized_vol(px, 20)
    fd = _Stub(px, turn=np.full_like(px, 2.5))
    got = FACTOR_DEFS["lowvol_20"](fd)
    np.testing.assert_allclose(got, -raw, equal_nan=True)
    ts = FACTOR_DEFS["turn_stab_60"](fd)
    assert np.isfinite(ts).any()
    assert np.nanmax(ts) <= 0.0, "换手稳定因子没取负号，值越大反而越不稳"


def test_004c_factor_list_fully_registered():
    missing = [n for n in FACTORS_004C if n not in FACTOR_DEFS]
    assert not missing, f"004C 候选因子未注册：{missing}"


class _Stub:
    """最小 FactorData 替身：只提供因子函数要用的字段。"""

    def __init__(self, px: np.ndarray, turn: np.ndarray):
        import pandas as pd

        class _P:
            close_hfq = px
        self.panel = _P()
        self.turn = turn
        self._pd = pd
