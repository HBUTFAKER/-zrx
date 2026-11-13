"""S4 熊市防守雷达 · 界面层。

## 形态（三块布局，取代原「建议卡」）

    ┌ 块一 状态灯 ─────────────┐  防守期 / 正常期 大字 + 判定日期 + 灯规则原文
    ├ 块二 观察名单 ────────────┤  防守期亮灯 → 展示 Top100（名次/代码/名称）
    │                          │  正常期     → 置灰 + 「正常期不给建议，只报状态」
    ├ 块三 历史统计 ────────────┤  防守月 5/60、观察名单 5 个防守月中 4 个合格
    └──────────────────────────┘

## 铁律（改代码前先看）

  ① **运行时零大模型接口**：本文件不 import 任何 LLM SDK，
     数据全部来自 `web/radar_data.py` 只读产物。
  ② **不做买入推荐**：界面不出现任何"建议 / 买入"组合字样、
     不出现收益类预测数值、不出现价格目标。
     避雷名单已判失败（2/5 = 40% < 70%），**不展示**。
  ③ **文案硬约束逐字照抄**，不许美化：
       「5 个防守月中 4 个合格，样本极小，仅定性参考」
       「本名单只在防守期有价值；2024、2025 反弹行情跑输基准」
       「本产物不构成投资建议」
  ④ **免责声明常驻**：不因任何交互而消失。
  ⑤ **灯态直读产物**（005S-M1）：不自行重算，不读基准 parquet。
  ⑥ **两个日期分开**（005S-M1）：灯态判定日（基准最新交易日，日频）
     与名单截面日（月末）语义不同、各自展示，**不许合并成同一个日期**。
  ⑦ **名称只取产物**（005S-M2）：Top100 表格的简称直接读
     `latest.watch_names`（code→简称），界面不查名、不猜名；
     产物没给名称的代码**只显示代码、不填占位符**。

## 启动

    .venv/Scripts/python.exe -m streamlit run web/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from web.radar_data import (
    BANNED_SUBSTRINGS,
    TXT_DEFENSE_ONLY,
    TXT_DISCLAIMER,
    TXT_NORMAL_PHASE,
    TXT_SAMPLE_WARNING,
    PayloadError,
    RadarPayload,
    history_table_rows,
    load_payload,
)

st.set_page_config(page_title="熊市防守雷达", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------ 样式
# 颜色全部走 Streamlit 主题变量（--secondary-background-color 等），
# 深浅色主题下都读得清；不写死 #94a3b8 这类只在暗色下成立的色值。
CSS = """
<style>
.block-title {font-size: 1.05rem; font-weight: 600; color: var(--text-color, #94a3b8);
               opacity: .75; letter-spacing: .08em; margin: 0 0 .4rem 0;}
.light-big   {font-size: 3.4rem; font-weight: 800; line-height: 1.1; margin: .2rem 0 .3rem 0;}
.light-date  {font-size: 1.05rem; color: var(--text-color, #cbd5e1); opacity: .85;}
.rule-text   {font-size: .88rem; color: var(--text-color, #94a3b8); opacity: .65;
               line-height: 1.6;}
.grayed      {opacity: .45; filter: grayscale(1);}
.notice      {border-left: 4px solid #f59e0b; padding: .55rem .9rem;
               background: rgba(245,158,11,.10); border-radius: 4px;
               font-size: .95rem; color: var(--text-color, #fbbf24);}
.disclaimer  {border-top: 1px solid var(--border-color, #334155); padding-top: .7rem;
               font-size: .86rem; color: var(--text-color, #94a3b8); opacity: .7;
               text-align: center;}
.stat-num    {font-size: 2.1rem; font-weight: 700; color: var(--text-color, #f1f5f9);}
.stat-lbl    {font-size: .82rem; color: var(--text-color, #94a3b8); opacity: .7;}
</style>
"""


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.2f}%"


# ------------------------------------------------------------ 块一 状态灯
def render_light(p: RadarPayload) -> None:
    st.markdown('<div class="block-title">① 市场状态灯</div>', unsafe_allow_html=True)
    light = p.light

    if light.defensive is True:
        color, label = "#ef4444", "🔴 防守期"
    elif light.defensive is False:
        color, label = "#22c55e", "🟢 正常期"
    else:
        # 产物 state 为 null：该交易日无法判定。
        # 绝不显示成"正常期"—— 把"没算出来"说成"安全"等于凭空放宽。
        color, label = "#a3a3a3", "⚪ 未判定"

    # 阈值在产物里是负数（-0.1），直接格式化会印出 "-−10%"，取绝对值再补符号。
    thr = abs(light.threshold) * 100

    dd_txt = _pct(light.drawdown) if light.drawdown is not None else "—"
    st.markdown(
        f'<div class="light-big" style="color:{color};">{label}</div>'
        f'<div class="light-date">灯态判定日：{light.as_of}'
        f'（相对过去 {light.window} 个交易日最高收盘回撤 {dd_txt}，'
        f'阈值 −{thr:.0f}%）</div>'
        f'<div class="rule-text">灯规则：{light.rule_text}</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------ 块二 观察名单
def render_watch(p: RadarPayload) -> None:
    st.markdown('<div class="block-title">② 观察名单（防守候选）</div>', unsafe_allow_html=True)
    watch = p.watch

    st.markdown(f'<div class="rule-text">名单规则：{watch.rule_text}</div>',
                unsafe_allow_html=True)

    # 名单截面日：与灯态判定日不同，两个日期分开写，不合并。
    st.markdown(
        f'<div class="rule-text">名单截面日：{watch.as_of}'
        f'（灯态判定日 {p.light.as_of}，两者不同，名单不随灯态日滚动）</div>',
        unsafe_allow_html=True,
    )

    if p.light.defensive is None:
        # 灯态未判定：既不是防守期也不是正常期。
        # 不展示名单内容，也不冒充"正常期只报状态"——那是另一回事。
        st.markdown(
            f'<div class="grayed">'
            f'<div class="light-big" style="font-size:2.2rem;">名单未展示</div>'
            f'<div class="rule-text">灯态未判定，无法确定当前是否防守期</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="notice">灯态未判定，本次不展示名单内容</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="rule-text">{TXT_DEFENSE_ONLY}</div>', unsafe_allow_html=True
        )
        return

    if watch.normal_phase:
        # 正常期：置灰 + 只报状态，不给任何名单内容
        st.markdown(
            f'<div class="grayed">'
            f'<div class="light-big" style="font-size:2.2rem;">名单已置灰</div>'
            f'<div class="rule-text">共 {len(watch.codes)} 只</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="notice">{TXT_NORMAL_PHASE}</div>', unsafe_allow_html=True
        )
        # 限制文案两期都常驻：它是"这份名单不通用"的披露，不是建议文案。
        # 防守期时随名单一起显示，正常期时单独显示，避免被当成通用名单。
        st.markdown(
            f'<div class="rule-text">{TXT_DEFENSE_ONLY}</div>', unsafe_allow_html=True
        )
        return

    st.markdown(
        f'<div class="rule-text">可投池 {watch.universe} 只｜名单 {len(watch.codes)} 只</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="notice">{TXT_DEFENSE_ONLY}</div>', unsafe_allow_html=True
    )
    # 展示为表格：只列名次、代码、名称，无收益预测、无买卖列。
    # 名称来自产物 latest.watch_names（005S-M2）：查不到名称就留空，
    # 不写占位符 —— 名称是给人看的补充，缺名字不该挡住名单本身。
    st.dataframe(
        [
            {
                "名次": i + 1,
                "代码": c,
                "名称": watch.names.get(c, ""),
            }
            for i, c in enumerate(watch.codes)
        ],
        width="stretch",
        height=420,
        hide_index=True,
    )


# ------------------------------------------------------------ 块三 历史统计
def render_history(p: RadarPayload) -> None:
    st.markdown('<div class="block-title">③ 历史统计（已发生的数，不做预测）</div>',
                unsafe_allow_html=True)
    h = p.history

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-num">{h.n_defensive_months} / {h.n_total_months}</div>'
            f'<div class="stat-lbl">防守月（{h.n_total_months} 个月里）</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-num">{h.n_qualified} / {h.n_watch_months}</div>'
            f'<div class="stat-lbl">观察名单：防守月合格数</div>',
            unsafe_allow_html=True,
        )
    with c3:
        verdict = "达标" if h.passed else "不达标"
        st.markdown(
            f'<div class="stat-num">{h.pass_rate * 100:.0f}%</div>'
            f'<div class="stat-lbl">合格占比（门槛 {h.threshold * 100:.0f}% · {verdict}）</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="notice">{TXT_SAMPLE_WARNING}</div>', unsafe_allow_html=True
    )

    with st.expander("逐月明细（防守月）", expanded=False):
        rows = history_table_rows(p)
        st.dataframe(
            [
                {
                    "月份": r["month"],
                    "名单收益": _pct(r["strategy_ret"]),
                    "H00300": _pct(r["bench_ret"]),
                    "超额": _pct(r["excess"]),
                    "名单MDD": _pct(r["strategy_mdd"]),
                    "基准MDD": _pct(r["bench_mdd"]),
                }
                for r in rows
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            f'<div class="rule-text">防守月清单：{"、".join(h.defensive_months)}'
            f'｜合格口径：{h.watch_rule_text}</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------ 免责与方法论
def render_footer(p: RadarPayload) -> None:
    with st.expander("口径与限制（产物原文）", expanded=False):
        st.markdown(f'- 产品说明：{p.product_note}')
        for n in p.validity_notes:
            st.markdown(f"- {n}")
        st.markdown(
            f'- 基准：{p.benchmark}｜产物生成时间：{p.generated_at}'
            f'｜产物版本 commit：{p.code_commit[:12]}'
        )
    st.markdown(
        f'<div class="disclaimer">{TXT_DISCLAIMER}</div>', unsafe_allow_html=True
    )


# ------------------------------------------------------------ 主流程
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("熊市防守雷达")
    st.caption(
        "只读确定性产物，运行时零大模型调用。"
        "不做买入推荐、不预测收益率；避雷名单已判失败故不展示。"
    )

    try:
        payload = load_payload()
    except PayloadError as exc:
        st.error(f"产物加载失败：{exc}")
        st.stop()
        return

    render_light(payload)
    st.divider()
    render_watch(payload)
    st.divider()
    render_history(payload)
    st.divider()
    render_footer(payload)


if __name__ == "__main__":
    main()
