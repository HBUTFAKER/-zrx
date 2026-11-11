"""S3-Q2 数据补抓：股本事件（东财）+ 申万行业分类变动历史。

为什么抓
--------
004B 要重建 size（真实流通市值）并做行业中性化，但现有数据两头都缺：

  · 股本：`finance.parquet` 里 totalShare/liqaShare 只有 270 行 / 64 只（0.15%），
    因为主源（东财报表摘要 / 巨潮）**不返回股本字段**，baostock 只在"退市股降级路径"
    跑过一批。004B-P 勘察结论：baostock `query_profit_data` 是**季度快照**，
    降级成"季度快照+前向填充"后 8 只样本 11,225 个交易日里 2,094 天（18.65%）
    与事件级口径差 >30%，size 会在解禁后整整一个季度用错股本 —— 不可用。
  · 行业：现有 `data/raw/meta/industry_*.parquet` 是**当前快照**（东财 180 板块 /
    证监会 84 大类），覆盖 95.5%，但 **341 只退市股一只都拿不到**，直接跑 8 年回测
    等于给 6.2% 的样本喂空行业。申万分类**变动历史**带 `计入日期`，可做时点正确归属。

数据源（均经 004B-P 勘察实测，总工已裁决）
------------------------------------------
1. 股本：东财 f10 股本结构，`ak.stock_zh_a_gbjg_em(symbol="600000.SH")`。
   返回是**事件级**（变更日期 / 总股本 / 已上市流通A股 / 变动原因），不是季度快照；
   退市股也有（实测 sh.600069 26 行、sh.600896 19 行）。
   勘察对拍：解禁日倍率与换手率推算口径同日对到三位小数（14.588/14.590 等），
   8 只样本内 14 次推算跳变有 12 次能在 ±3 日内找到真实事件。
2. 行业：申万宏源 `StockClassifyUse_stock.xls`（12,934 行 / 5,938 只 / 553 个三级代码，
   带 计入日期 / 更新日期 / 行业代码）。**该站证书链不全**，akshare 原生调用会
   SSLError，这里用 `requests(verify=False)` 直连同一官方 URL —— 已由总工追认。
3. 行业中文名：巨潮 `p_public0002`（`ak.stock_industry_category_cninfo`），
   只用于把申万 6 位代码翻译成中文名；映射不全的部分如实记覆盖率，不猜。

工程约束
--------
· **断点续传**：进度落 `data/shares/fetch_ledger.json`（每只 status/attempts/ts/rows/error），
  中断重跑只补缺；分片 checkpoint 落 `data/shares/_parts/`，避免 70 分钟白跑。
· **限流自适**：基线间隔 ≥0.5s；失败或疑似限流 → 间隔翻倍（上限 8s）；
  连续成功 20 次 → 间隔缓慢回落到基线。宁可慢，不许硬抓到被封。
· 只读网络、只写 `data/shares/`；**不碰 data/raw/ 与 data/cache/**。

用法
----
    python -m src.fetch_shares --stage shares            # 全量抓股本（约 70~90 分钟）
    python -m src.fetch_shares --stage shares --limit 5  # 小样本冒烟
    python -m src.fetch_shares --stage industry          # 申万行业 + 名称映射
    python -m src.fetch_shares --stage all
    python -m src.fetch_shares --stage check             # 只做覆盖率核对，不联网
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import ROOT

SHARES_DIR = ROOT / "data" / "shares"
PARTS_DIR = SHARES_DIR / "_parts"
LEDGER_PATH = SHARES_DIR / "fetch_ledger.json"
EVENTS_PATH = SHARES_DIR / "share_events.parquet"
SW_PATH = SHARES_DIR / "sw_industry.parquet"
SW_NAME_PATH = SHARES_DIR / "sw_industry_names.parquet"
COVERAGE_PATH = SHARES_DIR / "coverage.json"

SW_XLS_URL = ("https://www.swsresearch.com/swindex/pdf/SwClass2021/"
              "StockClassifyUse_stock.xls")

BASE_INTERVAL = 0.5     # 基线请求间隔（秒），勘察实测 0.15s 响应 + 0.5s 间隔不被限流
MAX_INTERVAL = 8.0      # 降速上限
SLOW_AFTER = 20         # 连续成功多少次后开始回落


# ---------------------------------------------------------------- 台账

def load_ledger(path: Path | None = None) -> dict:
    path = path or LEDGER_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": None, "updated_at": None, "stocks": {}}


def save_ledger(led: dict, path: Path | None = None) -> None:
    path = path or LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    led["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    led.setdefault("created_at", led["updated_at"])
    path.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 代码转换

def to_em_symbol(code: str) -> str:
    """面板 `sh.600000` → 东财 `600000.SH`。北交所（bj.）东财 f10 无此页，交给调用方判空。"""
    if "." not in code:
        raise ValueError(f"代码格式异常（应形如 sh.600000）：{code!r}")
    mk, num = code.split(".", 1)
    suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(mk.lower())
    if suffix is None:
        raise ValueError(f"未知市场前缀：{code!r}")
    return f"{num}.{suffix}"


def to_panel_code(symbol: str) -> str:
    """东财 `600000.SH` → 面板 `sh.600000`。"""
    num, _, suf = str(symbol).partition(".")
    return f"{suf.lower()}.{num}"


# ---------------------------------------------------------------- 股本单位抓取

def parse_share_frame(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    """东财股本结构原始表 → 统一事件表（纯函数，测试不联网）。

    返回列：code / event_date / total_share / float_share_a / float_share /
    restricted_share / reason / source。
    · float_share_a = 已上市流通A股（换手率的分母口径，勘察已证实），
      缺失时回退 已流通股份 —— 早期记录（上市前股本）两者都可能为空，
      此时保留 NaN 不猜（下游 as-of 时用不到，且不能凭空造股本）。
    · 原始表按时间**倒序**返回，这里统一升序。
    """
    if raw is None or not len(raw):
        return pd.DataFrame(columns=["code", "event_date", "total_share",
                                     "float_share_a", "float_share",
                                     "restricted_share", "reason", "source"])
    need = ["变更日期", "总股本"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(f"{code} 股本表缺列 {missing}，实际列 {list(raw.columns)}")

    def _num(name: str) -> pd.Series:
        """取数值列；列不存在返回全 NaN（缺列不等于错，但绝不能编出数字）。"""
        if name not in raw.columns:
            return pd.Series(np.nan, index=raw.index, dtype="float64")
        return pd.to_numeric(raw[name], errors="coerce")

    out = pd.DataFrame({
        "code": code,
        "event_date": pd.to_datetime(raw["变更日期"], errors="coerce"),
        "total_share": _num("总股本"),
        "float_share_a": _num("已上市流通A股"),
        "float_share": _num("已流通股份"),
        "restricted_share": _num("流通受限股份"),
        "reason": (raw["变动原因"] if "变动原因" in raw.columns
                   else pd.Series([""] * len(raw), index=raw.index))
                  .astype("string").fillna(""),
    })
    out = out.dropna(subset=["event_date"])
    out = out.sort_values("event_date").drop_duplicates(
        subset=["event_date"], keep="last").reset_index(drop=True)
    out["source"] = "eastmoney_gbjg"
    return out


def fetch_one_share(code: str, *, retries: int = 3, interval: float = BASE_INTERVAL,
                    sleep=time.sleep) -> tuple[pd.DataFrame | None, int, str]:
    """抓一只股票的股本事件。返回 (表|None, 尝试次数, 错误信息)。

    重试在函数内部完成（调用方只负责台账），每次重试前按当前 interval 等待，
    这样"疑似限流"的降速能立刻生效。空表（新股无记录/北交所）视为成功但 0 行。
    """
    last_err = ""
    # akshare 内部调 requests 时**没有超时**，实测会偶发整只股票卡死（首版跑到 200 只
    # 都没写出第一个 checkpoint）。这里给 socket 兜一个全局超时，卡住就抛异常走重试，
    # 否则一个坏请求能挂死整批 5,470 只。
    import socket
    socket.setdefaulttimeout(25)
    for attempt in range(1, retries + 1):
        try:
            import akshare as ak
            raw = ak.stock_zh_a_gbjg_em(symbol=to_em_symbol(code))
            df = parse_share_frame(raw, code)
            return df, attempt, ""
        except Exception as e:  # noqa: BLE001 —— 网络/限流/接口变更都走降速重试
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < retries:
                sleep(interval * attempt)
    return None, retries, last_err


def _write_part(rows: list[pd.DataFrame], idx: int) -> None:
    if not rows:
        return
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(
        PARTS_DIR / f"part_{idx:05d}.parquet", index=False)


def load_parts(parts_dir: Path | None = None) -> pd.DataFrame:
    parts_dir = parts_dir or PARTS_DIR
    files = sorted(parts_dir.glob("part_*.parquet")) if parts_dir.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def fetch_shares(codes: list[str], *, interval: float = BASE_INTERVAL,
                 retries: int = 3, checkpoint_every: int = 100,
                 limit: int | None = None, sleep=time.sleep,
                 progress=None) -> dict:
    """全量抓取（或续抓）。只补台账里没成功的代码。

    降速逻辑（自适）：
      · 失败一次 → 间隔 ×2；
      · 连续成功 SLOW_AFTER 次 → 间隔 ×0.9，但不低于基线；
      · 间隔到 MAX_INTERVAL 仍失败 3 次 → 该股记 failed，进失败清单，继续下一只
        （不整体退出：个别代码 404 不该毁掉 5,470 只的任务）。
    """
    if progress is None:
        progress = lambda m: print(m, flush=True)      # noqa: E731 —— 后台跑必须不缓冲
    SHARES_DIR.mkdir(parents=True, exist_ok=True)
    led = load_ledger()
    done = {c for c, v in led["stocks"].items() if v.get("status") == "ok"}
    todo = [c for c in codes if c not in done]
    if limit:
        todo = todo[:limit]

    cur = max(interval, BASE_INTERVAL)
    ok_streak = 0
    buf: list[pd.DataFrame] = []
    part_idx = len(list(PARTS_DIR.glob("part_*.parquet"))) if PARTS_DIR.exists() else 0
    t0 = time.time()
    n_ok = n_empty = n_fail = 0

    for k, c in enumerate(todo, 1):
        df, attempts, err = fetch_one_share(c, retries=retries, interval=cur, sleep=sleep)
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if df is None:
            n_fail += 1
            led["stocks"][c] = {"status": "failed", "attempts": attempts,
                               "ts": now, "rows": 0, "error": err}
            cur = min(cur * 2.0, MAX_INTERVAL)
            ok_streak = 0
            progress(f"  [fail] {c} ×{attempts} {err[:80]}；间隔→{cur:.2f}s")
        else:
            n_ok += 1
            if not len(df):
                n_empty += 1
            led["stocks"][c] = {"status": "ok", "attempts": attempts,
                               "ts": now, "rows": int(len(df))}
            if len(df):
                buf.append(df)
            ok_streak += 1
            if ok_streak >= SLOW_AFTER:
                cur = max(BASE_INTERVAL, cur * 0.9)
                ok_streak = 0
        sleep(cur)

        if k % 25 == 0:
            progress(f"  ... {k}/{len(todo)} ok={n_ok} fail={n_fail} 间隔={cur:.2f}s "
                     f"{time.time()-t0:.0f}s")

        if k % checkpoint_every == 0:
            _write_part(buf, part_idx)
            part_idx += 1
            buf = []
            save_ledger(led)
            el = time.time() - t0
            eta = el / k * (len(todo) - k) / 60.0
            progress(f"  {k}/{len(todo)}  ok={n_ok} empty={n_empty} fail={n_fail} "
                     f"间隔={cur:.2f}s 已用 {el/60:.1f}min 预计剩余 {eta:.0f}min")

    _write_part(buf, part_idx)
    save_ledger(led)
    return {"requested": len(todo), "ok": n_ok, "empty": n_empty, "failed": n_fail,
            "elapsed_min": round((time.time() - t0) / 60.0, 2),
            "interval_end": round(cur, 3)}


def merge_parts() -> dict:
    """分片合并成 share_events.parquet（按 code+event_date 去重，后者覆盖前者）。"""
    d = load_parts()
    if not len(d):
        return {"rows": 0, "codes": 0}
    d = d.drop_duplicates(subset=["code", "event_date"], keep="last")
    d = d.sort_values(["code", "event_date"]).reset_index(drop=True)
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(EVENTS_PATH, index=False)
    return {"rows": int(len(d)), "codes": int(d["code"].nunique()),
            "path": str(EVENTS_PATH)}


# ---------------------------------------------------------------- 申万行业

def fetch_sw_xls(url: str = SW_XLS_URL, timeout: int = 60) -> pd.DataFrame:
    """下载申万分类变动历史 xls。

    akshare 的 `stock_industry_clf_hist_sw` 在 swsresearch.com 上会 SSLError
    （证书链不全，实测 CERTIFICATE_VERIFY_FAILED），这里直连同一官方 URL 并
    关闭校验 —— 数据源没变，只是绕开证书问题；总工已追认。
    """
    import requests
    import urllib3
    urllib3.disable_warnings()
    r = requests.get(url, verify=False, timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return pd.read_excel(BytesIO(r.content))


def parse_sw_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """申万原始表 → 统一表：code / industry_code / in_date / update_date。

    · 股票代码在 xls 里是**数值**，000001 会被读成 1 —— 必须 zfill(6)，
      否则与面板 join 时整段错位（勘察踩过）。
    · in_date（计入日期）是行业归属的**生效日**，时点匹配只能用它；
      update_date 是记录被改写的日期，晚于 in_date 中位 1,356 天，说明历史归属
      被重述过（申万 2014/2021 换版），严格时点仍有残余前视 —— 报告里如实披露。
    """
    ren = {"股票代码": "code", "行业代码": "industry_code",
           "计入日期": "in_date", "更新日期": "update_date"}
    missing = [c for c in ren if c not in raw.columns]
    if missing:
        raise KeyError(f"申万表缺列 {missing}，实际列 {list(raw.columns)}")
    d = raw.rename(columns=ren)[list(ren.values())].copy()
    d["code"] = d["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    d["industry_code"] = (d["industry_code"].astype(str).str.strip()
                          .str.replace(r"\.0$", "", regex=True).str.zfill(6))
    d["in_date"] = pd.to_datetime(d["in_date"], errors="coerce")
    d["update_date"] = pd.to_datetime(d["update_date"], errors="coerce")
    d = d.dropna(subset=["in_date", "industry_code"])
    d = d.sort_values(["code", "in_date"]).drop_duplicates(
        subset=["code", "in_date"], keep="last").reset_index(drop=True)
    d["panel_code"] = np.where(d["code"].str.startswith("6"), "sh." + d["code"],
                               np.where(d["code"].str.startswith(("4", "8")),
                                        "bj." + d["code"], "sz." + d["code"]))
    return d


def fetch_sw_names() -> tuple[pd.DataFrame, dict]:
    """申万行业代码 → 中文名（巨潮 p_public0002，akshare 封装）。

    只做展示用映射，不参与计算。覆盖率按 1/2/3 级分别统计，缺的不编。
    """
    import akshare as ak
    raw = ak.stock_industry_category_cninfo(symbol="申银万国行业分类标准")
    raw = raw.copy()
    raw["num"] = (raw["类目编码"].astype(str).str.replace("^S", "", regex=True).str.strip())
    raw = raw[raw["num"].str.len().isin([2, 4, 6])]
    m = (raw[["num", "类目名称", "分级"]]
         .rename(columns={"num": "industry_code", "类目名称": "industry_name",
                          "分级": "level"})
         .drop_duplicates(subset=["industry_code"]))
    m["source"] = "cninfo_p_public0002 申银万国行业分类标准"
    return m, {"rows": int(len(m))}


def build_sw(coverage_codes: list[str] | None = None) -> dict:
    """抓 + 解析 + 落盘 + 覆盖率。返回统计（写进 coverage.json）。"""
    SHARES_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_sw_xls()
    sw = parse_sw_frame(raw)
    sw.to_parquet(SW_PATH, index=False)
    names, _ = fetch_sw_names()
    names.to_parquet(SW_NAME_PATH, index=False)

    # 覆盖率：1/2/3 级各自能翻译出中文名的比例（只报告，不猜）
    lv = {}
    leaf = set(sw["industry_code"])
    for lvl, cut in ((1, 2), (2, 4), (3, 6)):
        codes = {c[:cut] for c in leaf}
        lv[f"level{lvl}"] = {"codes": len(codes),
                             "named": int(len(codes & set(names["industry_code"])))}
    return {"rows": int(len(sw)), "codes": int(sw["code"].nunique()),
            "industry_codes": int(sw["industry_code"].nunique()),
            "names": lv, "path": str(SW_PATH)}


# ---------------------------------------------------------------- 覆盖率核对

def coverage_report(panel_codes: list[str] | None = None,
                    panel_start: str = "2018-01-02",
                    panel_end: str = "2026-11-06") -> dict:
    """股本事件 / 行业 对面板股票的覆盖率（对照勘察预期：股本 100%、行业 99.7%）。

    只用本地文件，不联网（`--stage check` 可随时复算）。
    """
    from .backtest import load_panel
    if panel_codes is None:
        panel_codes = load_panel(panel_start, panel_end).codes
    pc = set(panel_codes)

    out: dict = {"panel": {"codes": len(pc), "start": panel_start, "end": panel_end}}

    if EVENTS_PATH.exists():
        ev = pd.read_parquet(EVENTS_PATH, columns=["code", "event_date"])
        have = set(ev["code"])
        cov = pc & have
        # 事件起点晚于面板起点也算"覆盖"，但要在报告里说明有多少只晚于起点
        first = ev.groupby("code")["event_date"].min()
        late = [c for c in cov if str(first.get(c))[:10] > panel_start]
        out["shares"] = {
            "events_rows": int(len(ev)), "events_codes": int(len(have)),
            "panel_covered": len(cov),
            "panel_coverage": round(len(cov) / max(len(pc), 1), 6),
            "missing": len(pc - have),
            "first_event_after_panel_start": len(late),
            "expected": 1.0,
        }
    else:
        out["shares"] = {"available": False, "reason": f"缺 {EVENTS_PATH}"}

    if SW_PATH.exists():
        sw = pd.read_parquet(SW_PATH, columns=["panel_code", "industry_code", "in_date"])
        have = set(sw["panel_code"])
        cov = pc & have
        out["industry"] = {
            "rows": int(len(sw)), "codes": int(len(have)),
            "panel_covered": len(cov),
            "panel_coverage": round(len(cov) / max(len(pc), 1), 6),
            "missing": len(pc - have),
            "expected": 0.997,
            "note": "勘察口径：面板 5,470 只 100%、退市股 340/341。此处按当前面板复算",
        }
    else:
        out["industry"] = {"available": False, "reason": f"缺 {SW_PATH}"}

    if EVENTS_PATH.exists() or SW_PATH.exists():
        COVERAGE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    return out


# ---------------------------------------------------------------- CLI

def _panel_codes(start: str, end: str) -> list[str]:
    from .backtest import load_panel
    return load_panel(start, end).codes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S3-Q2 数据补抓（股本事件 / 申万行业）")
    ap.add_argument("--stage", default="all",
                    choices=["shares", "industry", "all", "check"])
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 只（冒烟用）")
    ap.add_argument("--codes", default=None, help="逗号分隔的代码，指定抓取")
    ap.add_argument("--interval", type=float, default=BASE_INTERVAL)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args(argv)

    if args.stage in ("shares", "all"):
        SHARES_DIR.mkdir(parents=True, exist_ok=True)
        codes = ([c.strip() for c in args.codes.split(",") if c.strip()]
                 if args.codes else _panel_codes(args.start, args.end))
        print(f"[shares] 目标 {len(codes)} 只（台账已有成功记录会跳过）", flush=True)
        st = fetch_shares(codes, interval=args.interval, retries=args.retries,
                          limit=args.limit)
        print(json.dumps(st, ensure_ascii=False))
        print("[shares] 合并分片 →", json.dumps(merge_parts(), ensure_ascii=False))

    if args.stage in ("industry", "all"):
        print("[industry] 下载申万分类变动历史 + 中文名映射 ...", flush=True)
        print(json.dumps(build_sw(), ensure_ascii=False))

    if args.stage in ("check", "all"):
        print("[check] 覆盖率核对 ...", flush=True)
        print(json.dumps(coverage_report(panel_start=args.start, panel_end=args.end),
                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
