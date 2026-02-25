"""
中小企業轉型貸款風險評估系統
Backend : Groq API  (免費，2–4 秒)
Hosting : Streamlit Community Cloud (免費)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────
GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
MODEL     = "llama-3.1-8b-instant"
MAX_CHARS = 3_500
TIMEOUT   = 30

HARD_DIMS = ["財務可行性", "現金流穩定性", "還款能力", "抵押擔保充足性"]
SOFT_DIMS = ["市場競爭壓力", "經營團隊能力", "轉型合理性"]
ALL_DIMS  = HARD_DIMS + SOFT_DIMS

WEIGHTS: dict[str, float] = {
    "財務可行性": 0.22, "現金流穩定性": 0.18, "還款能力": 0.20,
    "抵押擔保充足性": 0.10, "市場競爭壓力": 0.12, "經營團隊能力": 0.10, "轉型合理性": 0.08,
}

RISK_BANDS = {"低": (1.0, 2.39), "中": (2.40, 3.49), "高": (3.50, 5.00)}
RISK_COLOR = {"低": "#27ae60", "中": "#e67e22", "高": "#c0392b"}
SCORE_LABEL = {1: "優良", 2: "尚可", 3: "需關注", 4: "高風險", 5: "極高風險"}
KEYWORDS = {
    "市場競爭壓力": ["市場", "競爭", "差異", "利基", "市占", "產業"],
    "經營團隊能力": ["團隊", "管理", "經驗", "人才", "負責人", "創辦人"],
    "轉型合理性":   ["計畫", "里程碑", "KPI", "效益", "技術", "導入", "系統", "時程"],
}


# ── PDF ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def read_pdf(file_bytes: bytes) -> str:
    try:
        pdf  = PdfReader(BytesIO(file_bytes))
        text = "\n\n".join((p.extract_text() or "").strip() for p in pdf.pages).strip()
        if not text:
            raise ValueError("PDF 無法讀取文字，請確認為文字型 PDF（非掃描圖片）。")
        return text
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"PDF 解析失敗：{e}") from e


def focused_text(doc: str) -> str:
    all_kw = [kw for kws in KEYWORDS.values() for kw in kws]
    paras  = [p.strip() for p in re.split(r"\n{2,}", doc) if p.strip()]
    scored = sorted(paras, key=lambda p: sum(kw in p for kw in all_kw), reverse=True)
    buf, out = 0, []
    for p in scored:
        if buf >= MAX_CHARS:
            break
        out.append(p); buf += len(p)
    prefix = doc[:400]
    body   = "\n\n".join(out)
    return (prefix + "\n\n" + body)[:MAX_CHARS] if prefix not in body else body[:MAX_CHARS]


# ── Financials ────────────────────────────────────────────────

@dataclass
class Fin:
    annual_rev: Optional[float] = None
    monthly_rev: Optional[float] = None
    assets: Optional[float] = None
    debt: Optional[float] = None
    profit: Optional[float] = None
    cashflow: Optional[float] = None
    loan: Optional[float] = None
    collateral: Optional[float] = None
    debt_pct: Optional[float] = None
    roi_pct: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def debt_ratio(self):
        if self.debt_pct: return self.debt_pct / 100
        if self.debt and self.assets: return self.debt / self.assets
        return None

    @property
    def dscr(self):
        inc = self.cashflow or self.profit
        return (inc / (self.loan / 5)) if inc and self.loan else None

    @property
    def ltv(self):
        return (self.loan / self.collateral) if self.loan and self.collateral else None

    @property
    def rev_loan(self):
        rev = self.annual_rev or (self.monthly_rev * 12 if self.monthly_rev else None)
        return (rev / self.loan) if rev and self.loan else None

    @property
    def completeness(self):
        return sum(1 for v in [
            self.annual_rev or self.monthly_rev, self.debt or self.debt_pct,
            self.profit or self.cashflow, self.loan, self.collateral
        ] if v)


@st.cache_data(show_spinner=False)
def extract_fin(doc: str) -> Fin:
    N, U = r"[\d,]+(?:\.\d+)?", r"[億千百萬元]+"
    UNITS = {"億": 10000, "千萬": 1000, "百萬": 100, "萬": 1, "千": 0.1, "元": 0.0001}
    fin = Fin()

    def wan(ns: str, us: str) -> float:
        n = float(ns.replace(",", ""))
        for u, m in UNITS.items():
            if u in us: return n * m
        return n

    def get(pats: list[str], label: str) -> Optional[float]:
        for p in pats:
            m = re.search(p, doc)
            if m:
                try:
                    v = wan(m.group("n"), m.group("u"))
                    fin.notes.append(f"找到 {label}：{v:.0f} 萬")
                    return v
                except Exception:
                    pass
        fin.notes.append(f"未找到 {label}")
        return None

    fin.annual_rev  = get([rf"年(?:營業額|營收|收入)[^\d]*(?P<n>{N})(?P<u>{U})"], "年營收")
    if not fin.annual_rev:
        fin.monthly_rev = get([rf"月(?:均)?(?:營收|收入)[^\d]*(?P<n>{N})(?P<u>[億千百萬元]*)"], "月營收")
    fin.assets     = get([rf"總資產[^\d]*(?P<n>{N})(?P<u>{U})"], "總資產")
    fin.debt       = get([rf"(?:總)?負債[^\d]*(?P<n>{N})(?P<u>{U})"], "總負債")
    fin.profit     = get([rf"(?:稅後)?淨利[^\d]*(?P<n>{N})(?P<u>{U})", rf"年獲利[^\d]*(?P<n>{N})(?P<u>{U})"], "淨利")
    fin.cashflow   = get([rf"(?:年)?現金流[^\d]*(?P<n>{N})(?P<u>{U})", rf"EBITDA[^\d]*(?P<n>{N})(?P<u>{U})"], "現金流")
    fin.loan       = get([rf"(?:申請)?貸款(?:金額)?[^\d]*(?P<n>{N})(?P<u>{U})", rf"融資(?:金額)?[^\d]*(?P<n>{N})(?P<u>{U})"], "貸款金額")
    fin.collateral = get([rf"擔保(?:品)?(?:估)?值[^\d]*(?P<n>{N})(?P<u>{U})", rf"抵押(?:品)?(?:估)?值[^\d]*(?P<n>{N})(?P<u>{U})"], "擔保品")
    m = re.search(r"負債(?:比率?|占比)[^\d]*(?P<n>[\d.]+)\s*%", doc)
    if m: fin.debt_pct = float(m.group("n")); fin.notes.append(f"找到負債比率：{fin.debt_pct}%")
    m = re.search(r"(?:預期|投資)?報酬率[^\d]*(?P<n>[\d.]+)\s*%", doc)
    if m: fin.roi_pct = float(m.group("n")); fin.notes.append(f"找到 ROI：{fin.roi_pct}%")
    return fin


# ── Rule Engine ───────────────────────────────────────────────

@dataclass
class RS:
    score: int
    note:  str


def run_rules(fin: Fin) -> dict[str, RS]:

    def viability() -> RS:
        dr = fin.debt_ratio
        if dr is None: return RS(4, "負債比率缺失，財務結構不透明。")
        if not 0 <= dr <= 1: return RS(5, f"負債比率 {dr:.1%} 超出合理範圍。")
        s, n = ((1, f"負債比率 {dr:.1%}，財務穩健。")   if dr <= .30 else
                (2, f"負債比率 {dr:.1%}，良好。")        if dr <= .50 else
                (3, f"負債比率 {dr:.1%}，中等槓桿。")    if dr <= .65 else
                (4, f"負債比率 {dr:.1%}，高槓桿。")      if dr <= .80 else
                (5, f"負債比率 {dr:.1%}，超高槓桿。"))
        if not fin.profit and not fin.cashflow and s < 4:
            s = min(5, s + 1); n += " 未揭露獲利，加計風險。"
        return RS(s, n)

    def cashflow_s() -> RS:
        cf  = fin.cashflow
        rev = fin.annual_rev or (fin.monthly_rev * 12 if fin.monthly_rev else None)
        if not cf and not rev: return RS(5, "未揭露任何現金流或營收數據。")
        inc = cf or (rev * .15 if rev else None)
        if not inc:  return RS(4, "現金流估算困難。")
        if inc <= 0: return RS(5, f"現金流為負（{inc:.0f} 萬）。")
        if fin.loan:
            cov = inc / (fin.loan / 60)
            return RS(*((1, f"覆蓋 {cov:.1f}x，充足。") if cov >= 2.0 else
                        (2, f"覆蓋 {cov:.1f}x，尚可。") if cov >= 1.5 else
                        (3, f"覆蓋 {cov:.1f}x，勉強。") if cov >= 1.0 else
                        (4, f"覆蓋 {cov:.1f}x，不足。") if cov >= 0.7 else
                        (5, f"覆蓋 {cov:.1f}x，嚴重不足。")))
        return RS(2 if inc >= 500 else 3 if inc >= 100 else 4, f"現金流 {inc:.0f} 萬。")

    def repayment() -> RS:
        d, r = fin.dscr, fin.rev_loan
        if d is None and r is None: return RS(4, "缺乏計算 DSCR 所需數據。")
        if d is not None:
            s, n = ((1, f"DSCR={d:.2f}，還款能力強。")   if d >= 2.5 else
                    (2, f"DSCR={d:.2f}，良好。")           if d >= 1.5 else
                    (3, f"DSCR={d:.2f}，勉強，無緩衝。") if d >= 1.1 else
                    (4, f"DSCR={d:.2f}，覆蓋不足。")       if d >= 0.8 else
                    (5, f"DSCR={d:.2f}，幾乎無法還款。"))
        else:
            s, n = ((2, f"年收/貸款={r:.1f}x，充足。") if r >= 3.0 else
                    (3, f"年收/貸款={r:.1f}x，尚可。") if r >= 1.5 else
                    (4, f"年收/貸款={r:.1f}x，偏低。") if r >= 0.8 else
                    (5, f"年收/貸款={r:.1f}x，過低。"))
        if fin.roi_pct and fin.roi_pct > 300:
            s = min(5, s + 1); n += f" ROI 聲稱 {fin.roi_pct:.0f}% 不合理。"
        return RS(s, n)

    def collateral_s() -> RS:
        ltv = fin.ltv
        if ltv is None:
            return RS(4 if not fin.collateral else 3,
                      "未提及擔保品估值。" if not fin.collateral else "有擔保品但未說明估值。")
        return RS(1 if ltv <= .50 else 2 if ltv <= .70 else 3 if ltv <= .85 else 4 if ltv <= 1.0 else 5,
                  f"LTV={ltv:.1%}。")

    return {"財務可行性": viability(), "現金流穩定性": cashflow_s(),
            "還款能力": repayment(), "抵押擔保充足性": collateral_s()}


# ── Groq ─────────────────────────────────────────────────────

def call_groq(api_key: str, text: str) -> dict:
    prompt = f"""你是銀行信審主任。針對以下企劃書，評估三個定性風險維度（1=低風險，5=高風險）。

評分邏輯：
1=有具體數據且表現優良 | 2=大致良好，部分缺乏佐證 | 3=表面描述無量化依據
4=明顯缺失或不合理 | 5=完全未提及或嚴重問題
資訊缺失一律給4或5分，不得給3分。

只輸出這個JSON，不加任何說明：
{{
  "市場競爭壓力": {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句推理>", "score": <1-5>}},
  "經營團隊能力": {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句推理>", "score": <1-5>}},
  "轉型合理性":   {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句推理>", "score": <1-5>}}
}}

企劃書：
---
{text}
---"""

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1, "max_tokens": 600},
        timeout=TIMEOUT,
    )
    if resp.status_code == 401:
        raise ValueError("Groq API Key 無效，請到 console.groq.com 確認。")
    if resp.status_code == 429:
        raise ValueError("請求超過免費限額，請稍後再試。")
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    for candidate in [raw.strip(), (re.search(r'\{[\s\S]*\}', raw) or type('', (), {'group': lambda s: ''})()).group()]:
        try:
            data = json.loads(candidate)
            if all(isinstance(data.get(d), dict) and isinstance(data[d].get("score"), int)
                   and 1 <= data[d]["score"] <= 5 for d in SOFT_DIMS):
                return data
        except Exception:
            pass
    raise ValueError(f"LLM 輸出格式無法解析。原始輸出：{raw[:300]}")


# ── Scoring ───────────────────────────────────────────────────

@dataclass
class Result:
    scores: dict[str, float]
    notes:  dict[str, str]
    quotes: dict[str, str]
    src:    dict[str, str]
    wavg:   float
    risk:   str


def compute(rule_res: dict[str, RS], soft: dict) -> Result:
    sc, nt, qt, src = {}, {}, {}, {}
    for d in HARD_DIMS:
        sc[d], nt[d], qt[d], src[d] = float(rule_res[d].score), rule_res[d].note, "", "rule"
    for d in SOFT_DIMS:
        sc[d]  = float(soft[d]["score"])
        nt[d]  = soft[d].get("reason", "")
        qt[d]  = soft[d].get("quote", "")
        src[d] = "llm"
    w    = round(sum(sc[d] * WEIGHTS[d] for d in ALL_DIMS), 3)
    risk = next((k for k, (lo, hi) in RISK_BANDS.items() if lo <= w <= hi), "未知")
    return Result(sc, nt, qt, src, w, risk)


# ── Charts ────────────────────────────────────────────────────

def _c(s: float) -> str:
    return "#27ae60" if s <= 2 else ("#e67e22" if s <= 3.5 else "#c0392b")


def gauge_fig(w: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=w,
        number={"suffix": " / 5", "font": {"size": 24}},
        gauge={
            "axis":  {"range": [1, 5], "tickvals": [1, 2, 3, 4, 5]},
            "bar":   {"color": "#333", "thickness": 0.18},
            "steps": [{"range": [1.0, 2.4], "color": "#27ae60"},
                      {"range": [2.4, 3.5], "color": "#e67e22"},
                      {"range": [3.5, 5.0], "color": "#c0392b"}],
            "threshold": {"line": {"color": "#111", "width": 4}, "value": w},
        },
    ))
    fig.update_layout(height=210, margin=dict(t=10, b=5, l=20, r=20))
    return fig


def bar_fig(res: Result) -> go.Figure:
    labels = [("규칙  " if res.src[d] == "rule" else "LLM  ") + d for d in ALL_DIMS]
    vals   = [res.scores[d] for d in ALL_DIMS]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=[_c(v) for v in vals],
        text=[f"{v:.1f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 6], title="風險分數（1 = 最低，5 = 最高）", showgrid=True),
        yaxis=dict(tickfont=dict(size=12)),
        height=300, margin=dict(t=5, b=35, l=130, r=55),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
    )
    return fig


# ── UI ────────────────────────────────────────────────────────

st.set_page_config(page_title="貸款風險評估", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.5rem; font-weight: 600; }
    h2 { font-size: 1.1rem; font-weight: 600; color: #333; }
    .stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("**API 設定**")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("免費申請：console.groq.com")
    st.divider()
    st.markdown("**風險分級說明**")
    st.caption("加權平均分（1–5）")
    for lvl, (lo, hi) in RISK_BANDS.items():
        st.markdown(
            f"<span style='color:{RISK_COLOR[lvl]};font-weight:600;'>{lvl}風險</span>"
            f"　{lo:.2f} – {hi:.2f}",
            unsafe_allow_html=True,
        )

# Main
st.title("中小企業轉型貸款風險評估")
st.divider()

if not api_key:
    st.warning("請先在左側輸入 Groq API Key。")
    st.stop()

uploaded = st.file_uploader("上傳企劃書 PDF", type=["pdf"], label_visibility="collapsed")
if uploaded is None:
    st.stop()

file_bytes = uploaded.read()
with st.spinner("讀取 PDF..."):
    try:
        doc = read_pdf(file_bytes)
    except (ValueError, RuntimeError) as e:
        st.error(str(e)); st.stop()

with st.expander(f"文件預覽（共 {len(doc):,} 字）"):
    st.text(doc[:500] + ("..." if len(doc) > 500 else ""))

if len(doc) < 200:
    st.warning("文件內容過短，結果可能不準確。")

if not st.button("開始評估", type="primary"):
    st.stop()

t0 = time.time()
fin      = extract_fin(doc)
rule_res = run_rules(fin)

# Rule scores — shown immediately
st.markdown("**客觀維度（規則引擎）**")
cols = st.columns(4)
for i, dim in enumerate(HARD_DIMS):
    rs = rule_res[dim]
    with cols[i]:
        st.markdown(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:12px 14px;'>"
            f"<div style='font-size:.78rem;color:#555;margin-bottom:4px;'>{dim}</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:{_c(rs.score)};line-height:1;'>{rs.score}</div>"
            f"<div style='font-size:.72rem;color:#888;margin-top:3px;'>{SCORE_LABEL[rs.score]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("依據"):
            st.caption(rs.note)

with st.expander("財務數字詳情"):
    ca, cb = st.columns(2)
    with ca:
        rows = [("年營收（萬）", fin.annual_rev), ("月營收（萬）", fin.monthly_rev),
                ("總資產（萬）", fin.assets),     ("總負債（萬）", fin.debt),
                ("淨利（萬）",   fin.profit),      ("現金流（萬）", fin.cashflow),
                ("貸款金額（萬）",fin.loan),       ("擔保品（萬）", fin.collateral)]
        df = pd.DataFrame(rows, columns=["欄位", "數值"])
        df["數值"] = df["數值"].apply(lambda v: f"{v:.0f}" if v is not None else "—")
        st.dataframe(df.set_index("欄位"), use_container_width=True)
    with cb:
        st.metric("負債比率", f"{fin.debt_ratio:.1%}" if fin.debt_ratio else "—")
        st.metric("DSCR",     f"{fin.dscr:.2f}"        if fin.dscr      else "—")
        st.metric("LTV",      f"{fin.ltv:.1%}"          if fin.ltv       else "—")
        st.metric("資料完整度", f"{fin.completeness}/5")

st.divider()

# LLM scoring
with st.spinner("Groq 分析中..."):
    try:
        soft = call_groq(api_key, focused_text(doc))
    except ValueError as e:
        st.error(str(e)); st.stop()
    except requests.exceptions.Timeout:
        st.error("請求逾時，請重試。"); st.stop()
    except Exception as e:
        st.error(f"API 錯誤：{e}"); st.stop()

elapsed = round(time.time() - t0, 1)

st.markdown("**定性維度（LLM 分析）**")
s_cols = st.columns(3)
for i, dim in enumerate(SOFT_DIMS):
    entry = soft[dim]; sc = int(entry["score"])
    with s_cols[i]:
        st.markdown(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:12px 14px;'>"
            f"<div style='font-size:.78rem;color:#555;margin-bottom:4px;'>{dim}</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:{_c(sc)};line-height:1;'>{sc}</div>"
            f"<div style='font-size:.72rem;color:#888;margin-top:3px;'>{SCORE_LABEL[sc]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("引用 / 推理"):
            st.caption(f"引用：{entry.get('quote', '—')}")
            st.caption(f"推理：{entry.get('reason', '—')}")

st.divider()

# Final result
res = compute(rule_res, soft)

c1, c2, c3 = st.columns([1.5, 1.2, 3.3])
with c1:
    st.plotly_chart(gauge_fig(res.wavg), use_container_width=True)
with c2:
    color = RISK_COLOR.get(res.risk, "#999")
    st.markdown(
        f"<div style='background:{color};color:#fff;font-size:2rem;font-weight:700;"
        f"text-align:center;padding:.5em .8em;border-radius:10px;margin-top:1.2rem;'>"
        f"{res.risk}風險</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"加權分 {res.wavg:.3f}　耗時 {elapsed}s")
with c3:
    red   = [d for d in ALL_DIMS if res.scores[d] >= 4]
    amber = [d for d in ALL_DIMS if 3 <= res.scores[d] < 4]
    if red:   st.error(  f"高風險維度（≥4分）：{', '.join(red)}")
    if amber: st.warning(f"需關注維度（3–4分）：{', '.join(amber)}")
    if not red and not amber: st.success("所有維度低於 3 分")

st.plotly_chart(bar_fig(res), use_container_width=True)
st.divider()

# Dimension detail
st.markdown("**各維度說明**")
for dim in ALL_DIMS:
    sc  = res.scores[dim]
    src = res.src[dim]
    a, b, c = st.columns([1.8, .55, 3.65])
    with a:
        st.markdown(
            f"**{dim}**  \n"
            f"<span style='font-size:.75rem;color:#888;'>{'規則引擎' if src=='rule' else 'LLM'} · 權重 {WEIGHTS[dim]:.0%}</span>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"<div style='font-size:1.8rem;font-weight:700;color:{_c(sc)};margin-top:.2rem;'>{sc:.1f}</div>",
            unsafe_allow_html=True,
        )
    with c:
        if src == "llm":
            st.caption(f"引用：{res.quotes[dim]}")
            st.caption(f"推理：{res.notes[dim]}")
        else:
            st.caption(res.notes[dim])
    st.write("")

# Export
st.divider()
report = {
    "評估時間": time.strftime("%Y-%m-%d %H:%M:%S"), "耗時(s)": elapsed,
    "模型": MODEL, "風險等級": res.risk, "加權分": res.wavg,
    "各維度分數": res.scores,
    "財務數字": {
        "年營收_萬": fin.annual_rev, "月營收_萬": fin.monthly_rev,
        "總資產_萬": fin.assets, "總負債_萬": fin.debt,
        "淨利_萬": fin.profit, "現金流_萬": fin.cashflow,
        "貸款金額_萬": fin.loan, "擔保品_萬": fin.collateral,
        "負債比率": fin.debt_ratio, "DSCR": fin.dscr, "LTV": fin.ltv,
    },
    "規則引擎": {d: rule_res[d].note for d in HARD_DIMS},
    "LLM分析":  {d: soft[d] for d in SOFT_DIMS},
}
st.download_button(
    "下載評估報告（JSON）",
    data=json.dumps(report, ensure_ascii=False, indent=2),
    file_name="report.json",
    mime="application/json",
)
