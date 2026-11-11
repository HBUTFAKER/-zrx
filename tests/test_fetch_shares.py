"""S3-Q2 抓取层测试：全部离线，不联网。

覆盖点（对应 004B 指令第一部分）：
  1. 代码格式转换（面板 sh.600000 ↔ 东财 600000.SH），格式错误必须抛错不猜；
  2. 股本原始表 → 事件表的清洗口径（升序、同日去重、缺列不编数、坏日期丢弃）；
  3. 申万表解析（数值代码 zfill(6)、北交所映射、坏计入日期丢弃）；
  4. 断点续传（台账 ok 的跳过、failed 重抓）；
  5. 自适降速（失败翻倍、连续成功回落，且不破基线）；
  6. 分片合并去重；
  7. 覆盖率核对口径。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import fetch_shares as fs


# ---------------------------------------------------------------- 1. 代码转换

def test_symbol_roundtrip():
    assert fs.to_em_symbol("sh.600000") == "600000.SH"
    assert fs.to_em_symbol("sz.300750") == "300750.SZ"
    assert fs.to_em_symbol("bj.830799") == "830799.BJ"
    assert fs.to_panel_code("600000.SH") == "sh.600000"
    assert fs.to_panel_code("300750.SZ") == "sz.300750"


def test_symbol_rejects_bad_format():
    with pytest.raises(ValueError):
        fs.to_em_symbol("600000")          # 没有市场前缀，不许猜沪深
    with pytest.raises(ValueError):
        fs.to_em_symbol("hk.00700")        # 未知市场，不许默认


# ---------------------------------------------------------------- 2. 股本表清洗

def _raw_gbjg() -> pd.DataFrame:
    """模拟东财返回：倒序、含同日重复、含一行坏日期、含空流通股本。"""
    return pd.DataFrame({
        "变更日期": ["2025-01-24", "2025-01-22", "2025-01-22", "2024-03-11", "坏日期"],
        "总股本": [8359816164, 8359816164, 8359816164, 8359816164, 1],
        "已上市流通A股": [5041934164, 345574164, 345574165, 133336023, 2],
        "已流通股份": [5041934164, 345574164, 345574165, 133336023, 2],
        "流通受限股份": [0, 0, 0, 0, 0],
        "变动原因": ["首发限售股份上市", "股份性质变更", "股份性质变更", "回购", "x"],
    })


def test_parse_share_frame_ordering_and_dedup():
    d = fs.parse_share_frame(_raw_gbjg(), "sz.001289")
    assert list(d["event_date"].dt.strftime("%Y-%m-%d")) == [
        "2024-03-11", "2025-01-22", "2025-01-24"], "必须升序、同日只留最后一条、坏日期丢弃"
    # 同日重复保留 keep="last" → 取 345574165 那条
    assert d.loc[d["event_date"] == pd.Timestamp("2025-01-22"), "float_share_a"].iloc[0] == 345574165
    assert set(d["source"]) == {"eastmoney_gbjg"}
    assert d["code"].nunique() == 1 and d["code"].iloc[0] == "sz.001289"


def test_parse_share_frame_missing_optional_cols_is_nan_not_zero():
    raw = pd.DataFrame({"变更日期": ["2020-01-02"], "总股本": [1e9]})
    d = fs.parse_share_frame(raw, "sh.600000")
    assert d["float_share_a"].isna().all(), "缺列要留 NaN，绝不能用 0 顶替（0 会被当成真实股本）"
    assert d["float_share"].isna().all()
    assert d["reason"].iloc[0] == ""


def test_parse_share_frame_requires_key_cols():
    with pytest.raises(KeyError):
        fs.parse_share_frame(pd.DataFrame({"日期": ["2020-01-02"]}), "sh.600000")


def test_parse_share_frame_empty():
    assert len(fs.parse_share_frame(pd.DataFrame(), "sh.600000")) == 0
    assert len(fs.parse_share_frame(None, "sh.600000")) == 0


# ---------------------------------------------------------------- 3. 申万表解析

def test_parse_sw_frame_zfill_and_market():
    raw = pd.DataFrame({
        "股票代码": [1, 600000, 830799, 300750],
        "行业代码": [440101, 480101, 630305, 630701],
        "计入日期": ["1991-04-03", "1999-11-10", "2021-11-15", "2018-05-28"],
        "更新日期": ["2015-10-27", "2015-10-27", "2021-11-15", "2025-12-15"],
    })
    d = fs.parse_sw_frame(raw)
    assert sorted(d["code"]) == ["000001", "300750", "600000", "830799"], \
        "数值代码必须 zfill(6)，否则 000001 会变成 '1' 与面板 join 错位"
    m = dict(zip(d["code"], d["panel_code"]))
    assert m["000001"] == "sz.000001"
    assert m["600000"] == "sh.600000"
    assert m["830799"] == "bj.830799"
    assert set(d["industry_code"]) == {"440101", "480101", "630305", "630701"}


def test_parse_sw_frame_drops_bad_dates_and_dedups():
    raw = pd.DataFrame({
        "股票代码": ["600000", "600000", "600001"],
        "行业代码": ["480101", "480102", "360101"],
        "计入日期": ["2020-01-01", "坏", "2019-05-01"],
        "更新日期": ["2020-01-01", "坏", "2019-05-01"],
    })
    d = fs.parse_sw_frame(raw)
    assert len(d) == 2, "坏计入日期要丢，不许用 NaT 混进时点匹配"


# ---------------------------------------------------------------- 4. 断点续传 + 5. 降速

@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """把三个目录/台账路径都指到 tmp，避免测试写进真 data/。"""
    monkeypatch.setattr(fs, "SHARES_DIR", tmp_path)
    monkeypatch.setattr(fs, "PARTS_DIR", tmp_path / "_parts")
    monkeypatch.setattr(fs, "LEDGER_PATH", tmp_path / "fetch_ledger.json")
    monkeypatch.setattr(fs, "EVENTS_PATH", tmp_path / "share_events.parquet")
    monkeypatch.setattr(fs, "SW_PATH", tmp_path / "sw_industry.parquet")
    monkeypatch.setattr(fs, "COVERAGE_PATH", tmp_path / "coverage.json")
    return tmp_path


def _fake_fetch(behavior: dict):
    """behavior: code → "ok" / "empty" / "fail"。"""
    calls = []

    def _f(code, retries=3, interval=0.5, sleep=None):
        calls.append(code)
        b = behavior.get(code, "ok")
        if b == "fail":
            return None, retries, "Push2RateLimitError: 限流"
        if b == "empty":
            return fs.parse_share_frame(pd.DataFrame(), code), 1, ""
        raw = pd.DataFrame({"变更日期": ["2020-01-02", "2021-01-04"],
                            "总股本": [1e9, 1.2e9],
                            "已上市流通A股": [8e8, 9e8],
                            "已流通股份": [8e8, 9e8],
                            "流通受限股份": [0, 0],
                            "变动原因": ["首发限售股份上市", "增发A股上市"]})
        return fs.parse_share_frame(raw, code), 1, ""
    return _f, calls


def test_resume_skips_ok_and_retries_failed(sandbox, monkeypatch):
    codes = ["sh.600000", "sz.300750", "sh.600069"]
    fake, calls = _fake_fetch({"sh.600069": "fail"})
    monkeypatch.setattr(fs, "fetch_one_share", fake)

    st1 = fs.fetch_shares(codes, sleep=lambda *_: None, progress=lambda *_: None)
    assert st1["ok"] == 2 and st1["failed"] == 1
    led = fs.load_ledger()
    assert led["stocks"]["sh.600069"]["status"] == "failed"
    assert led["stocks"]["sh.600000"]["status"] == "ok"

    # 第二次：ok 的跳过，failed 重抓并成功
    fake2, calls2 = _fake_fetch({})
    monkeypatch.setattr(fs, "fetch_one_share", fake2)
    st2 = fs.fetch_shares(codes, sleep=lambda *_: None, progress=lambda *_: None)
    assert st2["requested"] == 1, f"只该重抓失败的 1 只，实际 {st2['requested']}"
    assert calls2 == ["sh.600069"]
    assert fs.load_ledger()["stocks"]["sh.600069"]["status"] == "ok"


def test_failure_doubles_interval_and_success_decays(sandbox, monkeypatch):
    fake, _ = _fake_fetch({"sh.600001": "fail", "sh.600002": "fail"})
    monkeypatch.setattr(fs, "fetch_one_share", fake)
    st = fs.fetch_shares(["sh.600001", "sh.600002"], interval=0.5,
                         sleep=lambda *_: None, progress=lambda *_: None)
    assert st["interval_end"] > fs.BASE_INTERVAL, "连续失败必须降速"

    # 连续成功 20 次后间隔回落，但不得低于基线
    fake2, _ = _fake_fetch({})
    monkeypatch.setattr(fs, "fetch_one_share", fake2)
    codes = [f"sh.{600100 + i}" for i in range(fs.SLOW_AFTER + 5)]
    st2 = fs.fetch_shares(codes, interval=2.0, sleep=lambda *_: None,
                          progress=lambda *_: None)
    assert fs.BASE_INTERVAL <= st2["interval_end"] < 2.0, \
        f"连续成功后应回落且不破基线，实际 {st2['interval_end']}"


def test_empty_result_is_ok_not_failed(sandbox, monkeypatch):
    fake, _ = _fake_fetch({"bj.830799": "empty"})
    monkeypatch.setattr(fs, "fetch_one_share", fake)
    st = fs.fetch_shares(["bj.830799"], sleep=lambda *_: None, progress=lambda *_: None)
    assert st["ok"] == 1 and st["empty"] == 1 and st["failed"] == 0
    assert fs.load_ledger()["stocks"]["bj.830799"]["status"] == "ok"


# ---------------------------------------------------------------- 6. 分片合并

def test_merge_parts_dedup(sandbox, monkeypatch):
    fs.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.DataFrame({"code": ["sh.600000"], "event_date": pd.to_datetime(["2020-01-02"]),
                      "total_share": [1.0], "reason": ["老"], "source": ["eastmoney_gbjg"]})
    b = pd.DataFrame({"code": ["sh.600000", "sh.600001"],
                      "event_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                      "total_share": [2.0, 3.0], "reason": ["新", "x"],
                      "source": ["eastmoney_gbjg", "eastmoney_gbjg"]})
    a.to_parquet(fs.PARTS_DIR / "part_00000.parquet", index=False)
    b.to_parquet(fs.PARTS_DIR / "part_00001.parquet", index=False)
    st = fs.merge_parts()
    assert st["rows"] == 2 and st["codes"] == 2
    ev = pd.read_parquet(fs.EVENTS_PATH)
    assert ev.loc[(ev.code == "sh.600000"), "total_share"].iloc[0] == 2.0, "同日重复以后写的为准"


def test_merge_parts_empty(sandbox):
    assert fs.merge_parts()["rows"] == 0


# ---------------------------------------------------------------- 7. 覆盖率

def test_coverage_report_counts_and_gaps(sandbox, monkeypatch):
    pd.DataFrame({
        "code": ["sh.600000", "sh.600000", "sz.300750"],
        "event_date": pd.to_datetime(["1999-11-10", "2020-01-02", "2017-06-11"]),
    }).to_parquet(fs.EVENTS_PATH, index=False)
    pd.DataFrame({
        "panel_code": ["sh.600000", "sz.300750"],
        "industry_code": ["480101", "630701"],
        "in_date": pd.to_datetime(["1999-11-10", "2018-05-28"]),
    }).to_parquet(fs.SW_PATH, index=False)

    cov = fs.coverage_report(panel_codes=["sh.600000", "sz.300750", "sh.600999"])
    assert cov["shares"]["panel_covered"] == 2
    assert cov["shares"]["missing"] == 1
    assert cov["shares"]["panel_coverage"] == pytest.approx(2 / 3, rel=1e-6)
    assert cov["shares"]["first_event_after_panel_start"] == 0
    assert cov["industry"]["panel_coverage"] == pytest.approx(2 / 3, rel=1e-6)
    assert json.loads(fs.COVERAGE_PATH.read_text(encoding="utf-8"))["panel"]["codes"] == 3


def test_coverage_reports_first_event_after_panel_start(sandbox, monkeypatch):
    pd.DataFrame({
        "code": ["sh.688783"],
        "event_date": pd.to_datetime(["2024-11-29"]),   # 晚于面板起点 2018-01-02
    }).to_parquet(fs.EVENTS_PATH, index=False)
    cov = fs.coverage_report(panel_codes=["sh.688783"], panel_start="2018-01-02")
    assert cov["shares"]["first_event_after_panel_start"] == 1, \
        "事件起点晚于面板起点必须单独计数（那段日子没有真实股本可用）"
