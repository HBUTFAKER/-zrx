"""004C 阶段研究：低波动因子 + 反转多窗口 + 滚动验证（只研究，不出卡）。

职责边界：
  · 不改 config、不改评分层、不产出建议卡；
  · 只做 004C 的 C1/C2/C4：因子 IC、t 值、覆盖率、主窗口选择；
  · 所有选择只在**各自 24 个月训练段**内做，检验段只评估（防泄漏铁律）。

验证口径（004C 第一节 / 第三节）：
  检验段 2021~2025 共 5 个窗口，24 训 12 验，首个训练段 2019~2020。
  2018 年数据只用于因子预热（120 日波动、60 日换手需要前置窗口）。

C4 判据（004C 第二节）：
  因子已在本模块之外定向（factors.FACTORS_004C 里全部取负号，值越大越优），
  于是"方向为正"= ic_mean > 0；"够强"= |t| >= 2.0。两条都过才准进综合分。
  t 值一律在**该窗口的训练段**上测，禁止用全样本或检验段（否则等于用检验段选因子）。
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import numpy as np

from .factors import (FACTORS_004C, ROOT, cross_section_mask, forward_returns,
                      ic_periods, ic_summary, load_factor_data,
                      month_end_indices, rank_ic)

#: 检验年份（每个窗口的检验段 = 该年整年）
TEST_YEARS = (2021, 2022, 2023, 2024, 2025)
TRAIN_MONTHS = 24
TEST_MONTHS = 12
#: C4 硬闸：定向后 |t| 阈值
T_THRESHOLD = 2.0

#: 反转候选窗口（C2：20/40/60 日多窗口，按训练段滚动 IC 选主窗口）
REVERSAL_WINDOWS = (20, 40, 60)

OUT_JSON = ROOT / "out" / "004c_c1_factors.json"
OUT_MD = ROOT / "out" / "004c_c1_factors.md"


def _idx_ge(dates: list[str], d: str) -> int:
    """第一个 ≥ d 的行索引（dates 升序，字符串可直接比较）。"""
    return bisect_left([str(x)[:10] for x in dates], d)


def _idx_gt(dates: list[str], d: str) -> int:
    """第一个 > d 的行索引。段的右边界必须用它——用 ≤ 会把下一年的
    第一个交易日吞进来（踩过：W2 检验段一度写成 2022-01-04~2023-01-03）。"""
    return bisect_right([str(x)[:10] for x in dates], d)


def build_windows(dates: list[str]) -> list[dict]:
    """按 24 训 12 验切 5 个滚动窗口。

    窗口 Y（Y ∈ TEST_YEARS）：
      训练段 = (Y-2)-01-01 ~ (Y-1)-12-31 共 24 个月
      检验段 = Y-01-01 ~ Y-12-31      共 12 个月
    返回的 i_train1 / i_test0 是同一个索引（训练段右开、检验段左闭）。
    """
    out = []
    for y in TEST_YEARS:
        tr_s, tr_e = f"{y - 2}-01-01", f"{y - 1}-12-31"
        te_s, te_e = f"{y}-01-01", f"{y}-12-31"
        i0 = _idx_ge(dates, tr_s)
        i1 = _idx_ge(dates, te_s)          # 训练段右开 = 检验段左闭
        i2 = _idx_gt(dates, te_e)          # 严格大于段末日：不吞下一年的首个交易日
        i2 = min(i2, len(dates))
        out.append({
            "name": f"W{y - 2020}", "test_year": y,
            "train_start": str(dates[i0])[:10] if i0 < len(dates) else None,
            "train_end": str(dates[max(i1 - 1, 0)])[:10] if i1 > 0 else None,
            "test_start": str(dates[i1])[:10] if i1 < len(dates) else None,
            "test_end": str(dates[i2 - 1])[:10] if i2 > 0 else None,
            "i_train0": i0, "i_train1": i1, "i_test0": i1, "i_test1": i2,
        })
    return out


def factor_ic(fd, fmat: np.ndarray, i0: int, i1: int, fwd: np.ndarray) -> dict:
    """段内逐月 IC + 覆盖率。

    覆盖率 = 该期"因子有值"只数 / 该期可投池只数。窗口不足（NaN）的股票
    已在因子层被排除，这里只统计、不补偿。
    """
    usable, dropped = ic_periods(fd, i0, i1)
    ics: list[float] = []
    rows: list[dict] = []
    for p in usable:
        i, k = p["idx"], p["k"]
        ok = cross_section_mask(fd, i)
        n_pool = int(ok.sum())
        f = fmat[i]
        n_val = int((np.isfinite(f) & ok).sum())
        ic = rank_ic(f[ok], fwd[k][ok])
        ics.append(ic)
        rows.append({"cross_section": str(fd.dates[i])[:10],
                     "n_pool": n_pool, "n_value": n_val,
                     "coverage": (n_val / n_pool) if n_pool else None,
                     "ic": None if not np.isfinite(ic) else round(float(ic), 6)})
    cov = [r["coverage"] for r in rows if r["coverage"] is not None]
    summ = ic_summary(ics)
    summ.update({
        "n_ic_periods": len(usable),
        "n_dropped": len(dropped),
        "dropped_reasons": sorted({d["reason"] for d in dropped}),
        "coverage_median": round(float(np.median(cov)), 4) if cov else None,
        "coverage_min": round(float(np.min(cov)), 4) if cov else None,
        "n_pool_median": int(np.median([r["n_pool"] for r in rows])) if rows else None,
        "n_value_median": int(np.median([r["n_value"] for r in rows])) if rows else None,
        "periods": rows,
    })
    summ["ic_mean"] = None if not np.isfinite(summ["ic_mean"]) else round(summ["ic_mean"], 6)
    summ["ic_std"] = None if not np.isfinite(summ["ic_std"]) else round(summ["ic_std"], 6)
    summ["icir"] = None if not np.isfinite(summ["icir"]) else round(summ["icir"], 4)
    summ["t_stat"] = None if not np.isfinite(summ["t_stat"]) else round(summ["t_stat"], 4)
    summ["ic_pos_ratio"] = (None if not np.isfinite(summ["ic_pos_ratio"])
                            else round(summ["ic_pos_ratio"], 4))
    return summ


def c4_verdict(stats: dict) -> dict:
    """C4 判定：方向为正（ic_mean > 0）且 |t| ≥ 2.0 才进综合分。"""
    t = stats.get("t_stat")
    ic = stats.get("ic_mean")
    if t is None or ic is None:
        return {"pass": False, "reason": "无有效 IC 期"}
    if ic <= 0:
        return {"pass": False, "reason": f"方向为负（ic_mean={ic:+.4f}）"}
    if abs(t) < T_THRESHOLD:
        return {"pass": False, "reason": f"|t|={abs(t):.2f} < {T_THRESHOLD}"}
    return {"pass": True, "reason": f"ic_mean={ic:+.4f}, t={t:+.2f}"}


def select_reversal_primary(per_window: dict[str, dict]) -> dict:
    """C2：在训练段内按 IC 均值选反转主窗口，并冻结方向。

    每个窗口独立选（不做全局单一主窗口），结果写进 by_window。
    方向冻结规则：以训练段 ic_mean 的符号为准 —— 为正说明该窗口内反转成立，
    为负则说明该窗口内动量成立，综合分按冻结方向计分。
    """
    cands = {}
    for w, fstats in per_window.items():
        row = {}
        for name in fstats:
            if not name.startswith("reversal_"):
                continue
            s = fstats[name]
            row[int(name.split("_")[1])] = s
        if not row:
            continue
        best_w = max(row, key=lambda k: (row[k].get("ic_mean") or -9e9))
        best = row[best_w]
        cands[w] = {
            "primary_window": best_w,
            "direction": 1 if (best.get("ic_mean") or 0) > 0 else -1,
            "ic_mean": best.get("ic_mean"),
            "t_stat": best.get("t_stat"),
            "by_candidate": {str(k): {"ic_mean": v.get("ic_mean"),
                                      "t_stat": v.get("t_stat")}
                             for k, v in sorted(row.items())},
        }
    return cands


def run(start: str = "2018-01-01", end: str = "2025-12-31",
        verbose: bool = True) -> dict:
    """跑 C1/C2/C4 研究，返回报告 dict 并落盘。"""
    if verbose:
        print(f"[004C] 载入因子数据 {start} ~ {end} ...", flush=True)
    fd = load_factor_data(start, end, with_fundamentals=False,
                          with_shares=False, with_industry=False)
    if verbose:
        print(f"[004C] 面板 {len(fd.dates)} 交易日 × {len(fd.codes)} 只", flush=True)

    from .factors import FACTOR_DEFS
    if verbose:
        print(f"[004C] 计算候选因子 {FACTORS_004C} ...", flush=True)
    mats = {n: FACTOR_DEFS[n](fd) for n in FACTORS_004C}

    ends_global = month_end_indices(fd.dates)
    fwd = forward_returns(fd.panel.open_hfq, ends_global)
    windows = build_windows(fd.dates)

    per_window: dict[str, dict] = {}
    detail: dict[str, dict] = {}
    for w in windows:
        if verbose:
            print(f"[004C] {w['name']} 训练段 {w['train_start']}~{w['train_end']}"
                  f" / 检验段 {w['test_start']}~{w['test_end']}", flush=True)
        tr = {}
        te = {}
        for name, mat in mats.items():
            tr[name] = factor_ic(fd, mat, w["i_train0"], w["i_train1"], fwd)
            te[name] = factor_ic(fd, mat, w["i_test0"], w["i_test1"], fwd)
        per_window[w["name"]] = tr
        detail[w["name"]] = {"train": tr, "test": te}

    # ---- C4：逐窗口判定（只看训练段）
    c4 = {}
    for name in FACTORS_004C:
        rows = []
        for w in windows:
            s = per_window[w["name"]][name]
            v = c4_verdict(s)
            rows.append({"window": w["name"], "ic_mean": s["ic_mean"],
                         "t_stat": s["t_stat"], "n_periods": s["n_ic_periods"],
                         "coverage_median": s["coverage_median"], **v})
        n_pass = sum(1 for r in rows if r["pass"])
        c4[name] = {
            "per_window": rows,
            "n_pass": n_pass,
            "n_windows": len(rows),
            "selected_any": n_pass > 0,
            # 全窗口一致入选才是"稳"；部分入选意味着因子有效性随制度变化
            "selected_all": n_pass == len(rows),
        }

    rev = select_reversal_primary(per_window)

    git = _git_meta()
    report = {
        "spec": "004C C1/C2/C4 — 低波动因子 + 反转多窗口 + 滚动验证",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code": git,
        "params": {"train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS,
                   "t_threshold": T_THRESHOLD, "test_years": list(TEST_YEARS),
                   "factors": list(FACTORS_004C),
                   "reversal_windows": list(REVERSAL_WINDOWS)},
        "panel": {"start": str(fd.dates[0])[:10], "end": str(fd.dates[-1])[:10],
                  "dates": len(fd.dates), "codes": len(fd.codes),
                  "n_month_ends": len(ends_global)},
        "windows": windows,
        "c4_verdict": c4,
        "reversal_primary": rev,
        "by_window": detail,
        "notes": [
            "全部因子已在 FACTOR_DEFS 内定向（取负号）：值越大越优，IC 为正即方向为正。",
            "t 值逐窗口在 24 个月训练段上测，检验段只评估、不参与任何选择。",
            "覆盖率 = 因子有值只数 / 可投池只数；窗口不足的股票被排除，不插补。",
            "月频 IC 存在自相关，t 值会高估显著性（ic_summary 免责条款），只作量级参考。",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    if verbose:
        print(print_summary(report))
        print(f"[004C] 产物：{OUT_JSON}\n[004C] 报告：{OUT_MD}")
    return report


def render_md(rep: dict) -> str:
    """人读版报告（施工阶段报告用）。"""
    L = ["# 004C C1/C2/C4 阶段报告", ""]
    p = rep["panel"]
    c = rep["code"]
    L += [f"- 生成：{rep['generated_at']}",
          f"- 代码：`{c.get('code_commit')}`（code_dirty={c.get('code_dirty')}）",
          f"- 面板：{p['start']} ~ {p['end']}，{p['dates']} 交易日 × {p['codes']} 只，"
          f"{p['n_month_ends']} 个月末截面",
          f"- 口径：{rep['params']['train_months']} 训 / {rep['params']['test_months']} 验，"
          f"检验段 {rep['params']['test_years'][0]}~{rep['params']['test_years'][-1]}，"
          f"|t| ≥ {rep['params']['t_threshold']}",
          ""]
    L += ["## 窗口", "", "| 窗口 | 训练段 | 检验段 | 训练期数 |", "|---|---|---|---|"]
    for w in rep["windows"]:
        n = (rep["by_window"][w["name"]]["train"][rep["params"]["factors"][0]]
             ["n_ic_periods"])
        L.append(f"| {w['name']} | {w['train_start']}~{w['train_end']} "
                 f"| {w['test_start']}~{w['test_end']} | {n} |")
    L.append("")

    L += ["## C4 判定（训练段 IC / t 值）", "",
          "| 因子 | " + " | ".join(w["name"] for w in rep["windows"])
          + " | 入选窗口 |", "|---|" + "---|" * len(rep["windows"]) + "---|"]
    for name, r in rep["c4_verdict"].items():
        cells = []
        for x in r["per_window"]:
            t = x["t_stat"]
            mk = "✅" if x["pass"] else "❌"
            cells.append(f"{mk} {x['ic_mean']:+.4f} / t={t:+.2f}" if t is not None
                         else f"{mk} n/a")
        L.append(f"| `{name}` | " + " | ".join(cells)
                 + f" | {r['n_pass']}/{r['n_windows']} |")
    L.append("")

    L += ["## 覆盖率（训练段中位）", "",
          "| 因子 | " + " | ".join(w["name"] for w in rep["windows"]) + " |",
          "|---|" + "---|" * len(rep["windows"])]
    for name in rep["params"]["factors"]:
        cells = []
        for w in rep["windows"]:
            s = rep["by_window"][w["name"]]["train"][name]
            cv = s["coverage_median"]
            cells.append(f"{cv:.1%}" if cv is not None else "n/a")
        L.append(f"| `{name}` | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## C2 反转主窗口（逐窗口重选，产物存 by_window）", "",
          "| 窗口 | 主窗口 | 冻结方向 | IC 均值 | t | 20/40/60 IC |", "|---|---|---|---|---|---|"]
    for w, r in rep["reversal_primary"].items():
        cand = " / ".join(f"{k}:{v['ic_mean']:+.4f}" if v["ic_mean"] is not None
                          else f"{k}:n/a" for k, v in r["by_candidate"].items())
        L.append(f"| {w} | {r['primary_window']} 日 | "
                 f"{'反转' if r['direction'] > 0 else '动量'} | "
                 f"{r['ic_mean']:+.4f} | {r['t_stat']:+.2f} | {cand} |")
    L.append("")
    L += ["## 说明", ""] + [f"- {x}" for x in rep["notes"]]
    return "\n".join(L) + "\n"


def print_summary(rep: dict) -> str:
    out = ["", "=" * 78, "004C C1/C2/C4 摘要", "=" * 78]
    for name, r in rep["c4_verdict"].items():
        ts = " ".join(f"{x['t_stat']:+.2f}" if x["t_stat"] is not None else "n/a"
                      for x in r["per_window"])
        out.append(f"  {name:<14} 入选 {r['n_pass']}/{r['n_windows']}   t: {ts}")
    out.append("  " + "-" * 60)
    for w, r in rep["reversal_primary"].items():
        out.append(f"  {w} 反转主窗口 {r['primary_window']} 日，"
                   f"方向{'反转' if r['direction'] > 0 else '动量'}, "
                   f"IC {r['ic_mean']:+.4f}, t {r['t_stat']:+.2f}")
    out.append("=" * 78)
    return "\n".join(out)


def _git_meta() -> dict:
    """代码溯源：沿用 advisor/factors 口径（code_dirty 只看 src/ tests/）。"""
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=10
                                ).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10).stdout
        code = [ln[3:] for ln in st.splitlines()
                if ln[3:].startswith(("src/", "tests/"))]
        return {"code_commit": commit or "unknown", "code_dirty": bool(code),
                "code_dirty_files": code[:20]}
    except Exception:  # noqa: BLE001
        return {"code_commit": "unknown", "code_dirty": None, "code_dirty_files": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="004C C1/C2/C4 研究")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.start, a.end, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
