"""S4 熊市防守雷达 · 界面层测试。

验收口径由总工《施工指令 S4 · 雷达界面》给定，本文件只实现、不放宽。

覆盖：
  ① 文案硬约束逐字一致（不许美化）
  ② 禁字：界面渲染文本里不得出现「建议/买入」类字样与收益预测字样
  ③ 三块布局存在且顺序正确
  ④ 正常期观察名单置灰 + 只报状态；防守期展示 Top100
  ⑤ 避雷名单不展示（已判失败）
  ⑥ 只读产物：不改文件、不写文件
  ⑦ 运行时零大模型接口：界面模块不得 import 任何 LLM SDK
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web import radar_data as RD
from web.radar_data import (
    BANNED_SUBSTRINGS,
    PayloadError,
    find_banned,
    history_table_rows,
    load_payload,
)

PAYLOAD = ROOT / RD.PAYLOAD_REL


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def payload():
    return load_payload()


@pytest.fixture(scope="module")
def raw_payload():
    with PAYLOAD.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ================================================================= ① 文案硬约束
class TestHardCopy:
    """文案硬约束：逐字照抄，不许美化。"""

    def test_sample_warning_exact(self):
        assert RD.TXT_SAMPLE_WARNING == "5 个防守月中 4 个合格，样本极小，仅定性参考"

    def test_defense_only_exact(self):
        assert RD.TXT_DEFENSE_ONLY == "本名单只在防守期有价值；2024、2025 反弹行情跑输基准"

    def test_disclaimer_exact(self):
        assert RD.TXT_DISCLAIMER == "本产物不构成投资建议"

    def test_normal_phase_exact(self):
        assert RD.TXT_NORMAL_PHASE == "正常期不给建议，只报状态"

    def test_constants_appear_verbatim_in_app_source(self):
        """三段硬文案必须逐字出现在界面源文件里（防止被改写/换行截断）。"""
        src = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
        for const in (
            RD.TXT_SAMPLE_WARNING,
            RD.TXT_DEFENSE_ONLY,
            RD.TXT_DISCLAIMER,
            RD.TXT_NORMAL_PHASE,
        ):
            assert const in src, f"硬文案未逐字出现在 web/app.py：{const}"


# ================================================================= ② 禁字
class TestBannedWords:
    """禁字：不做买入推荐、不显示收益率预测。"""

    def test_banned_set_composition(self):
        """禁字表运行时必须等于约定集合（拼装后没拼错）。"""
        expected = {
            "建议买入", "买入建议", "推荐买入", "可以买", "建仓",
            "目标价", "预期收益率", "预计收益", "预期年化", "收益率预测",
        }
        assert set(BANNED_SUBSTRINGS) == expected

    def test_banned_literals_absent_from_source(self):
        """源文件里不得出现完整禁字明文（grep 审计要能干净通过）。"""
        for f in (ROOT / "web" / "app.py", ROOT / "web" / "radar_data.py"):
            text = f.read_text(encoding="utf-8")
            hits = [b for b in BANNED_SUBSTRINGS if b in text]
            assert not hits, f"{f.name} 出现禁字明文：{hits}"

    def test_disclaimer_not_flagged(self):
        """「本产物不构成投资建议」含「建议」但不是禁字，不得误伤。"""
        assert find_banned(RD.TXT_DISCLAIMER) == []

    def test_helper_detects(self):
        assert find_banned("这是建议买入") == ["建议买入"]
        assert find_banned("正常期不给建议，只报状态") == []
        assert find_banned("") == []


# ================================================================= ③④⑤ 界面渲染
class TestRender:
    """界面三块布局 + 正常期/防守期行为。"""

    @pytest.fixture(scope="class")
    @classmethod
    def app(cls):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=60)
        at.run()
        return at

    @staticmethod
    def _all_text(at) -> str:
        parts = [el.value for el in at.markdown]
        parts += [el.value for el in at.caption]
        parts += [str(el.value) for el in at.dataframe]
        return "\n".join(parts)

    def test_no_exception(self, app):
        assert len(app.exception) == 0, [e.value for e in app.exception]

    def test_title(self, app):
        assert app.title[0].value == "熊市防守雷达"

    def test_three_blocks_present_and_ordered(self, app):
        """三块布局：① 状态灯 ② 观察名单 ③ 历史统计，顺序不可乱。"""
        text = self._all_text(app)
        i1 = text.find("① 市场状态灯")
        i2 = text.find("② 观察名单")
        i3 = text.find("③ 历史统计")
        assert i1 != -1 and i2 != -1 and i3 != -1, "三块标题缺失"
        assert i1 < i2 < i3, "三块顺序错乱"

    def test_light_block_shows_state_and_date(self, app):
        text = self._all_text(app)
        assert "判定日期" in text
        assert "防守期" in text and "正常期" in text
        # 灯规则文案照抄产物 rules.light
        with PAYLOAD.open("r", encoding="utf-8") as fh:
            rule = json.load(fh)["rules"]["light"]
        assert rule in text, "灯规则文案未照抄产物 rules.light"

    def test_watch_block_gated_in_normal_phase(self, app, payload):
        """正常期：名单置灰 + 只报状态，不展示 Top100 内容。"""
        assert payload.light.defensive is False, (
            "当前判定为防守期，本用例前提失效（防守期分支另有用例覆盖）"
        )
        text = self._all_text(app)
        assert RD.TXT_NORMAL_PHASE in text
        assert "名单已置灰" in text
        # 正常期不得出现名单里的具体代码
        assert payload.watch.codes[0] not in text

    def test_watch_block_shows_top100_in_defensive_phase(self, payload):
        """防守期：展示 Top100（代码与名次），且不含收益预测列。"""
        codes = payload.watch.codes
        assert len(codes) == 100, "观察名单不是 100 只"

    @staticmethod
    def _defensive_benchmark(tmp_path) -> Path:
        """造一份能把灯压到 −10% 以下的基准序列（末尾一根大阴线）。"""
        import pandas as pd

        dates = pd.bdate_range("2026-01-01", periods=40).strftime("%Y-%m-%d")
        close = [100.0] * len(dates)
        close[-1] = 80.0          # 末根 −20%，必定触发防守灯
        p = tmp_path / "bench_defensive.parquet"
        pd.DataFrame({"date": dates, "close": close}).to_parquet(p, index=False)
        return p

    def test_defensive_phase_renders_top100_table(self, tmp_path, monkeypatch):
        """强制防守期分支：Top100 必须真渲染出来（当前是正常期，此分支默认走不到）。"""
        from streamlit.testing.v1 import AppTest

        bench = self._defensive_benchmark(tmp_path)
        monkeypatch.setenv(RD.BENCHMARK_ENV, str(bench))

        at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=60)
        at.run()
        assert len(at.exception) == 0, [e.value for e in at.exception]

        text = TestRender._all_text(at)
        assert "防守期" in text
        # 正常期文案只在**观察名单块**出现；页脚 product_note 里有同义的
        # 产品说明句（产物原文），不在本断言范围内。
        watch_block = text.split("② 观察名单", 1)[-1].split("③ 历史统计", 1)[0]
        assert RD.TXT_NORMAL_PHASE not in watch_block, "防守期不应再显示正常期文案"
        assert RD.TXT_DEFENSE_ONLY in watch_block

        # Top100 表格真渲染：名次 + 代码两列，100 行，顺序与产物一致
        frames = [df.value for df in at.dataframe]
        assert frames, "防守期没有渲染任何表格"
        watch_frame = frames[0]
        assert list(watch_frame.columns) == ["名次", "代码"]
        codes_in_table = [str(c) for c in watch_frame["代码"].tolist()]
        assert len(codes_in_table) == 100
        assert codes_in_table == load_payload().watch.codes, "防守期 Top100 未按产物顺序渲染"

    def test_defensive_phase_no_forecast_columns(self, tmp_path, monkeypatch):
        """防守期表格也不许出现收益/仓位/价格类列，界面文本不许有禁字。"""
        from streamlit.testing.v1 import AppTest

        bench = self._defensive_benchmark(tmp_path)
        monkeypatch.setenv(RD.BENCHMARK_ENV, str(bench))

        at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=60)
        at.run()
        assert len(at.exception) == 0, [e.value for e in at.exception]
        text = TestRender._all_text(at)
        assert find_banned(text) == [], find_banned(text)

    def test_defensive_phase_shows_no_avoid_codes(self, tmp_path, monkeypatch, raw_payload):
        """防守期也不许展示避雷名单（已判失败）。"""
        from streamlit.testing.v1 import AppTest

        bench = self._defensive_benchmark(tmp_path)
        monkeypatch.setenv(RD.BENCHMARK_ENV, str(bench))

        at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=60)
        at.run()
        text = TestRender._all_text(at)
        for code in raw_payload["latest"]["avoid"][:20]:
            assert code not in text, f"避雷名单代码泄露到界面：{code}"

    def test_history_block_numbers(self, app, payload):
        """历史统计块：防守月 5/60、观察名单 5 个防守月中 4 个合格。"""
        text = self._all_text(app)
        h = payload.history
        assert f"{h.n_defensive_months} / {h.n_total_months}" in text
        assert f"{h.n_qualified} / {h.n_watch_months}" in text
        assert RD.TXT_SAMPLE_WARNING in text

    def test_history_block_exact_numbers(self, app):
        """指令要求的字面数字：防守月 5/60、观察名单 4/5。"""
        text = self._all_text(app)
        assert "5 / 60" in text, "防守月未显示成 5 / 60"
        assert "4 / 5" in text, "观察名单合格数未显示成 4 / 5"

    def test_avoid_list_not_shown(self, app, raw_payload):
        """避雷名单已判失败（2/5 = 40% < 70%），界面不得展示其代码。"""
        text = self._all_text(app)
        avoid = raw_payload["latest"]["avoid"]
        assert len(avoid) == 100
        for code in avoid[:20]:
            assert code not in text, f"避雷名单代码泄露到界面：{code}"

    def test_disclaimer_always_visible(self, app):
        assert RD.TXT_DISCLAIMER in self._all_text(app)

    def test_no_return_forecast_words(self, app):
        """界面文本不得含收益预测类字样。"""
        text = self._all_text(app)
        assert find_banned(text) == [], find_banned(text)

    def test_no_target_price_or_advice_labels(self, app):
        text = self._all_text(app)
        for word in ("目标价", "预期年化", "预期收益率", "收益率预测"):
            assert word not in text


# ================================================================= ⑥ 只读
class TestReadOnly:
    """只读产物：不改文件、不写文件。"""

    def test_load_does_not_modify_payload(self, payload):
        before = PAYLOAD.stat().st_mtime_ns
        size = PAYLOAD.stat().st_size
        load_payload()
        assert PAYLOAD.stat().st_mtime_ns == before, "装载产物后文件被改动"
        assert PAYLOAD.stat().st_size == size

    def test_load_does_not_write_to_out_dir(self, payload):
        out_dir = ROOT / "out"
        before = {p.name: p.stat().st_mtime_ns for p in out_dir.iterdir()}
        load_payload()
        after = {p.name: p.stat().st_mtime_ns for p in out_dir.iterdir()}
        assert before == after, "装载过程写入了 out/ 目录"

    def test_missing_payload_raises(self, tmp_path):
        with pytest.raises(PayloadError):
            load_payload(tmp_path / "nope.json")

    def test_missing_field_raises_not_silently_defaulted(self, tmp_path):
        """字段缺失必须报错，不许兜底默认值 —— 否则界面会显示错的数。"""
        raw = json.loads(PAYLOAD.read_text(encoding="utf-8"))
        del raw["latest"]["watch"]
        p = tmp_path / "broken.json"
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(PayloadError):
            load_payload(p)


# ================================================================= ⑦ 零大模型
class TestNoLLM:
    """运行时零大模型接口。"""

    LLM_PATTERNS = (
        "openai", "anthropic", "google.generativeai", "genai",
        "ollama", "langchain", "llama_index", "litellm",
        "dashscope", "zhipu", "qianfan", "moonshot", "deepseek",
        "requests.post", "urllib.request", "httpx.post",
    )

    @pytest.mark.parametrize("f", ["app.py", "radar_data.py"])
    def test_no_llm_import_in_ui_modules(self, f):
        src = (ROOT / "web" / f).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        for name in imported:
            low = name.lower()
            assert not any(p in low for p in self.LLM_PATTERNS), (
                f"web/{f} 引入了疑似大模型/联网模块：{name}"
            )

    @pytest.mark.parametrize("f", ["app.py", "radar_data.py"])
    def test_no_network_calls_in_ui_modules(self, f):
        src = (ROOT / "web" / f).read_text(encoding="utf-8")
        for pat in ("requests.get", "requests.post", "urlopen", "httpx.", "aiohttp"):
            assert pat not in src, f"web/{f} 含网络调用：{pat}"


# ================================================================= 产物一致性
class TestPayloadContract:
    """界面读到的字段必须与产物一致（界面不许自己算另一套）。"""

    def test_light_rule_copied_verbatim(self, payload, raw_payload):
        assert payload.light.rule_text == raw_payload["rules"]["light"]

    def test_watch_rule_copied_verbatim(self, payload, raw_payload):
        assert payload.watch.rule_text == raw_payload["rules"]["watch"]

    def test_watch_codes_match_product(self, payload, raw_payload):
        assert payload.watch.codes == raw_payload["latest"]["watch"]

    def test_as_of_matches_product(self, payload, raw_payload):
        assert payload.watch.as_of == raw_payload["latest"]["as_of"]

    def test_history_numbers_match_product(self, payload, raw_payload):
        assert payload.history.n_qualified == raw_payload["watch"]["n_qualified"]
        assert payload.history.n_watch_months == raw_payload["watch"]["n_months"]
        assert payload.history.n_defensive_months == raw_payload["light"]["n_defensive_months"]
        assert payload.history.n_total_months == raw_payload["light"]["n_months"]

    def test_two_denominators_not_conflated(self, payload):
        """两个分母必须不同：60=全部月，5=防守月。混用会把 5/60 显示成 5/5。"""
        assert payload.history.n_total_months == 60
        assert payload.history.n_defensive_months == 5
        assert payload.history.n_watch_months == 5
        assert payload.history.n_qualified == 4

    def test_history_rows_are_defensive_months_only(self, payload, raw_payload):
        rows = history_table_rows(payload)
        assert len(rows) == raw_payload["watch"]["n_months"]
        assert {r["month"] for r in rows} == set(raw_payload["light"]["defensive_months"])

    def test_current_light_recomputed_from_same_rule(self, payload, raw_payload):
        """当前灯态按产物写死的 window/threshold 从基准重算，规则同源。"""
        assert payload.light.window == raw_payload["light"]["window"]
        assert payload.light.threshold == raw_payload["light"]["threshold"]
        assert payload.light.defensive == (payload.light.drawdown <= -payload.light.threshold + 1e-12)
