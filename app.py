"""
中小企業轉型貸款風險評估系統 (含數位與淨零轉型)
Backend : Groq API 
Hosting : Streamlit Community Cloud
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# 載入環境變數
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Config ────────────────────────────────────────────────────
GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
MODEL     = "llama-3.1-8b-instant"
MAX_CHARS = 3_500
TIMEOUT   = 30

HARD_DIMS = ["財務可行性", "現金流穩定性", "還款能力", "抵押擔保充足性"]
SOFT_DIMS = ["市場競爭壓力", "經營團隊能力", "數位轉型合理性", "永續淨零轉型"]
ALL_DIMS  = HARD_DIMS + SOFT_DIMS

# 重新分配權重，總和為 1.0
WEIGHTS: dict[str, float] = {
    "財務可行性": 0.20, "現金流穩定性": 0.15, "還款能力": 0.20, "抵押擔保充足性": 0.10, 
    "市場競爭壓力": 0.10, "經營團隊能力": 0.08, "數位轉型合理性": 0.09, "永續淨零轉型": 0.08,
}

RISK_BANDS = {"低": (1.0, 2.39), "中": (2.40, 3.49), "高": (3.50, 5.00)}
RISK_COLOR = {"低": "#27ae60", "中": "#e67e22", "高": "#c0392b"}
SCORE_LABEL = {1: "優良", 2: "良好", 3: "需關注", 4: "高風險", 5: "極高風險"}

KEYWORDS = {
    "市場競爭壓力": ["市場", "競爭", "差異", "利基", "市占", "產業"],
    "經營團隊能力": ["團隊", "管理", "經驗", "人才", "負責人", "創辦人"],
    "數位轉型合理性": ["計畫", "數位", "KPI", "效益", "技術", "導入", "系統", "自動化", "時程"],
    "永續淨零轉型": ["碳盤查", "減碳", "ESG", "綠能", "淨零", "永續", "環保", "節能", "碳足跡", "再生能源"],
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

# ── Financials (強化版) ────────────────────────────────────────

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
        if self.debt_pct is not None: return self.debt_pct / 100
        if self.debt and self.assets: return self.debt / self.assets
        return None

    @property
    def dscr(self):
        inc = self.cashflow or self.profit
        if inc and self.loan:
            # 簡化公式：(年度現金流) / (貸款總額/5年)
            return (inc / (self.loan / 5))
        return None

    @property
    def ltv(self):
        return (self.loan / self.collateral) if self.loan and self.collateral else None

    @property
    def rev_loan(self):
        rev = self.annual_rev or (self.monthly_rev * 12 if self.monthly_rev else None)
        return (rev / self.loan) if rev and self.loan else None

@st.cache_data(show_spinner=False)
def extract_fin(raw_doc: str) -> Fin:
    # 1. 預處理：將數字中間的空格去除，並統一全形字元
    doc = raw_doc.replace(" ", "").replace("　", "")
    
    N = r"(?P<n>[\d,]+(?:\.\d+)?)" # 數字抓取
    U = r"(?P<u>[億千萬百萬元]*)"  # 單位抓取
    UNITS = {"億": 10000, "千萬": 1000, "百萬": 100, "萬": 1, "千": 0.1, "元": 0.0001}
    
    fin = Fin()

    def parse_value(ns: str, us: str) -> float:
        val = float(ns.replace(",", ""))
        if not us: return val # 預設為萬
        for u, m in UNITS.items():
            if u in us: return val * m
        return val

    def find_it(patterns: list[str], label: str) -> Optional[float]:
        for p in patterns:
            # 使用 re.IGNORECASE 且允許中間夾雜任意非數字字元 (如 : 、 是 為)
            match = re.search(p, doc, re.IGNORECASE)
            if match:
                try:
                    v = parse_value(match.group("n"), match.group("u"))
                    fin.notes.append(f"✅ {label}：偵測到 {v:.0f} 萬")
                    return v
                except: pass
        fin.notes.append(f"❌ {label}：未偵測到")
        return None

    # 擴展關鍵字組
    fin.annual_rev = find_it([
        rf"(?:年(?:營業額|營收|收入|收入淨額)|營業收入合計)[^\d\.]*{N}{U}",
        rf"Revenue[^\d\.]*{N}{U}"
    ], "年營收")
    
    if not fin.annual_rev:
        fin.monthly_rev = find_it([rf"月(?:均)?(?:營收|收入)[^\d\.]*{N}{U}"], "月營收")

    fin.assets = find_it([
        rf"(?:總資產|資產總額|資產總計)[^\d\.]*{N}{U}",
        rf"TotalAssets[^\d\.]*{N}{U}"
    ], "總資產")

    fin.debt = find_it([
        rf"(?:總負債|負債總額|負債總計)[^\d\.]*{N}{U}",
        rf"TotalLiabilities[^\d\.]*{N}{U}"
    ], "總負債")

    fin.profit = find_it([
        rf"(?:稅後(?:淨利|溢利)|本期淨利|年獲利)[^\d\.]*{N}{U}",
        rf"NetIncome[^\d\.]*{N}{U}"
    ], "年度淨利")

    fin.cashflow = find_it([
        rf"(?:現金流|EBITDA|營業活動現金)[^\d\.]*{N}{U}",
        rf"CashFlow[^\d\.]*{N}{U}"
    ], "現金流")

    fin.loan = find_it([
        rf"(?:申請|擬申貸|預計融資|貸款)(?:金額|總額)?[^\d\.]*{N}{U}",
        rf"LoanAmount[^\d\.]*{N}{U}"
    ], "貸款金額")

    fin.collateral = find_it([
        rf"(?:擔保|抵押)(?:品)?(?:估|價)?值[^\d\.]*{N}{U}",
        rf"CollateralValue[^\d\.]*{N}{U}"
    ], "擔保品估值")

    # 處理百分比
    m_debt = re.search(r"(?:負債比率?|負債比|DebtRatio)[^\d\.]*(?P<n>[\d\.]+)\s*%", doc, re.I)
    if m_debt: 
        fin.debt_pct = float(m_debt.group("n"))
        fin.notes.append(f"✅ 負債比率：{fin.debt_pct}%")

    m_roi = re.search(r"(?:預期|投資)?報酬率(?:ROI)?[^\d\.]*(?P<n>[\d\.]+)\s*%", doc, re.I)
    if m_roi: 
        fin.roi_pct = float(m_roi.group("n"))
        fin.notes.append(f"✅ ROI：{fin.roi_pct}%")

    return fin

# ── Rule Engine ───────────────────────────────────────────────

@dataclass
class RS:
    score: int
    note:  str

def run_rules(fin: Fin) -> dict[str, RS]:
    def viability() -> RS:
        dr = fin.debt_ratio
        if dr is None: return RS(4, "⚠️ 負債比率缺失，財務結構不透明。")
        if not 0 <= dr <= 1: return RS(5, f"⚠️ 負債比率 {dr:.1%} 超出合理範圍，資料可能異常。")
        s, n = ((1, f"✅ 負債比率 {dr:.1%}：自有資本充足，財務結構極為穩健。") if dr <= .30 else
                (2, f"✅ 負債比率 {dr:.1%}：財務結構良好，槓桿控制在合理範圍。") if dr <= .50 else
                (3, f"⚠️ 負債比率 {dr:.1%}：屬中度槓桿，需留意升息壓力。") if dr <= .65 else
                (4, f"❌ 負債比率 {dr:.1%}：高槓桿營運，財務風險偏高。") if dr <= .80 else
                (5, f"❌ 負債比率 {dr:.1%}：超高槓桿，資本結構脆弱。"))
        if not fin.profit and not fin.cashflow and s < 4:
            s = min(5, s + 1); n += " (註：未揭露實質獲利，加計風險)"
        return RS(s, n)

    def cashflow_s() -> RS:
        cf  = fin.cashflow
        rev = fin.annual_rev or (fin.monthly_rev * 12 if fin.monthly_rev else None)
        if not cf and not rev: return RS(5, "❌ 未揭露任何現金流或營收數據，無法評估還款來源。")
        inc = cf or (rev * .15 if rev else None)
        if not inc:  return RS(4, "⚠️ 缺乏具體數據，現金流估算困難。")
        if inc <= 0: return RS(5, f"❌ 現金流為負（{inc:.0f} 萬），無法支應日常營運及還款。")
        if fin.loan:
            cov = inc / (fin.loan / 5) # 假設5年期
            return RS(*((1, f"✅ 預估現金流為年本息的 {cov:.1f} 倍，資金極度充裕。") if cov >= 2.0 else
                        (2, f"✅ 預估現金流為年本息的 {cov:.1f} 倍，足以穩定支應還款。") if cov >= 1.5 else
                        (3, f"⚠️ 預估現金流為年本息的 {cov:.1f} 倍，無多餘緩衝資金。") if cov >= 1.0 else
                        (4, f"❌ 預估現金流僅為年本息的 {cov:.1f} 倍，覆蓋能力不足。") if cov >= 0.7 else
                        (5, f"❌ 預估現金流為年本息的 {cov:.1f} 倍，面臨嚴重違約風險。")))
        return RS(2 if inc >= 500 else 3 if inc >= 100 else 4, f"未標明貸款金額，目前預估年現金流 {inc:.0f} 萬。")

    def repayment() -> RS:
        d, r = fin.dscr, fin.rev_loan
        if d is None and r is None: return RS(4, "⚠️ 缺乏計算償債覆蓋率(DSCR)所需之核心數據。")
        if d is not None:
            s, n = ((1, f"✅ DSCR = {d:.2f}：償債能力強勁，具備良好防禦力。") if d >= 2.5 else
                    (2, f"✅ DSCR = {d:.2f}：償債能力良好。") if d >= 1.5 else
                    (3, f"⚠️ DSCR = {d:.2f}：償債能力勉強，抗風險能力較弱。") if d >= 1.1 else
                    (4, f"❌ DSCR = {d:.2f}：無法單靠營業現金流支應本息。") if d >= 0.8 else
                    (5, f"❌ DSCR = {d:.2f}：幾乎無還款能力。"))
        else:
            s, n = ((2, f"✅ 年收為貸款額 {r:.1f} 倍：營收規模足以支撐此貸款。") if r >= 3.0 else
                    (3, f"⚠️ 年收為貸款額 {r:.1f} 倍：尚可，但需檢視淨利率。") if r >= 1.5 else
                    (4, f"❌ 年收為貸款額 {r:.1f} 倍：貸款額度相對於營收規模過高。") if r >= 0.8 else
                    (5, f"❌ 年收為貸款額 {r:.1f} 倍：過度借貸。"))
        if fin.roi_pct and fin.roi_pct > 300:
            s = min(5, s + 1); n += f" (註：聲稱 ROI {fin.roi_pct:.0f}% 明顯悖離常理，有過度包裝之嫌)"
        return RS(s, n)

    def collateral_s() -> RS:
        ltv = fin.ltv
        if ltv is None:
            return RS(4 if not fin.collateral else 3,
                      "❌ 未提及任何擔保品。" if not fin.collateral else "⚠️ 有擔保品但未說明確切估值，無法計算LTV。")
        return RS(1 if ltv <= .50 else 2 if ltv <= .70 else 3 if ltv <= .85 else 4 if ltv <= 1.0 else 5,
                  f"{'✅' if ltv <= .70 else ('⚠️' if ltv <= .85 else '❌')} LTV = {ltv:.1%} (貸款約佔擔保品 {ltv*10:.1f} 成)。")

    return {"財務可行性": viability(), "現金流穩定性": cashflow_s(),
            "還款能力": repayment(), "抵押擔保充足性": collateral_s()}

# ── Groq ─────────────────────────────────────────────────────

def call_groq(api_key: str, text: str) -> dict:
    prompt = f"""你是銀行企業金融部的資深信審主任。針對以下中小企業企劃書，請評估四個定性風險維度（1=低風險，5=高風險）。

評分邏輯：
1=有具體數據/明確計畫且表現優良 | 2=大致良好，部分缺乏佐證 | 3=表面描述無量化依據
4=明顯缺失或邏輯不合理 | 5=完全未提及或存在嚴重致命問題
資訊缺失一律給4或5分，不得給3分。

只輸出以下JSON格式，不加任何說明或Markdown標籤：
{{
  "市場競爭壓力": {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句審核員視角的俐落推論>", "score": <1-5>}},
  "經營團隊能力": {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句審核員視角的俐落推論>", "score": <1-5>}},
  "數位轉型合理性": {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句審核員視角的俐落推論>", "score": <1-5>}},
  "永續淨零轉型": {{"quote": "<原文引用，無則寫未提及>", "reason": "<一句審核員視角的俐落推論>", "score": <1-5>}}
}}

企劃書：
---
{text}
---"""

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1, "max_tokens": 800},
        timeout=TIMEOUT,
    )
    if resp.status_code == 401:
        raise ValueError("Groq API Key 無效，請確認 .env 設定檔。")
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
    labels = [("規章 " if res.src[d] == "rule" else "定性 ") + d for d in ALL_DIMS]
    vals   = [res.scores[d] for d in ALL_DIMS]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=[_c(v) for v in vals],
        text=[f"{v:.1f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 6], title="風險分數（1 = 最低，5 = 最高）", showgrid=True),
        yaxis=dict(tickfont=dict(size=12)),
        height=350, margin=dict(t=5, b=35, l=130, r=55),
        plot_bgcolor="#fff", paper_bgcolor="#fff",
    )
    return fig

# ── UI ────────────────────────────────────────────────────────

st.set_page_config(page_title="貸款風險評估 (雙軸轉型版)", layout="wide")

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
    st.markdown("### 🏦 評估系統資訊")
    st.caption("支援 **數位轉型** 與 **永續淨零轉型** 雙軸評估。")
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
st.title("中小企業轉型貸款風險評估系統")
st.divider()

if not GROQ_API_KEY:
    st.error("⚠️ 未偵測到 Groq API Key。請確保根目錄有 `.env` 檔案並設置 `GROQ_API_KEY`。")
    st.stop()

uploaded = st.file_uploader("上傳企業轉型企劃書 (PDF)", type=["pdf"], label_visibility="collapsed")
if uploaded is None:
    st.stop()

file_bytes = uploaded.read()
with st.spinner("讀取並解析 PDF..."):
    try:
        doc = read_pdf(file_bytes)
    except (ValueError, RuntimeError) as e:
        st.error(str(e)); st.stop()

with st.expander(f"文件預覽（共 {len(doc):,} 字）"):
    st.text(doc[:500] + ("..." if len(doc) > 500 else ""))

if len(doc) < 200:
    st.warning("文件內容過短，系統判讀結果可能缺乏信度。")

if not st.button("開始執行信用風險評估", type="primary"):
    st.stop()

t0 = time.time()
fin      = extract_fin(doc)
rule_res = run_rules(fin)

# Rule scores — shown immediately
st.markdown("### 📊 財務與信用維度 (硬指標)")
cols = st.columns(4)
for i, dim in enumerate(HARD_DIMS):
    rs = rule_res[dim]
    with cols[i]:
        st.markdown(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:12px 14px;'>"
            f"<div style='font-size:.85rem;color:#555;margin-bottom:4px;font-weight:600;'>{dim}</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{_c(rs.score)};line-height:1;'>{rs.score}</div>"
            f"<div style='font-size:.75rem;color:#888;margin-top:3px;'>{SCORE_LABEL[rs.score]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("信審員解析"):
            st.caption(rs.note)

with st.expander("🔍 點擊查看核心財務指標算式"):
    ca, cb = st.columns(2)
    with ca:
        rows = [("年營收 (萬)", fin.annual_rev), ("月營收 (萬)", fin.monthly_rev),
                ("總資產 (萬)", fin.assets),     ("總負債 (萬)", fin.debt),
                ("淨利 (萬)",   fin.profit),      ("現金流 (萬)", fin.cashflow),
                ("申請貸款 (萬)",fin.loan),       ("擔保品估值(萬)", fin.collateral)]
        df = pd.DataFrame(rows, columns=["會計科目", "解析數值"])
        df["解析數值"] = df["解析數值"].apply(lambda v: f"{v:,.0f}" if v is not None else "—")
        st.dataframe(df.set_index("會計科目"), use_container_width=True)
    with cb:
        st.markdown("**核心審查指標**")
        st.markdown(f"**負債比率 (Debt Ratio)**: `{fin.debt_ratio:.1%}`" if fin.debt_ratio else "**負債比率**: `—` (資料不足)")
        st.caption("計算方式：總負債 ÷ 總資產。衡量企業自有資本充足度。")
        
        st.markdown(f"**償債覆蓋率 (DSCR)**: `{fin.dscr:.2f}x`" if fin.dscr else "**DSCR**: `—` (資料不足)")
        st.caption("計算方式：預估年現金流 ÷ 每年應繳本息。倍數越高代表還款能力越安全。")
        
        st.markdown(f"**貸款成數 (LTV)**: `{fin.ltv:.1%}`" if fin.ltv else "**LTV**: `—` (資料不足)")
        st.caption("計算方式：貸款金額 ÷ 擔保品估值。衡量銀行債權的保障程度。")

st.divider()

# LLM scoring
with st.spinner("AI 進行雙軸轉型與定性分析中..."):
    try:
        soft = call_groq(GROQ_API_KEY, focused_text(doc))
    except ValueError as e:
        st.error(str(e)); st.stop()
    except requests.exceptions.Timeout:
        st.error("請求逾時，請重試。"); st.stop()
    except Exception as e:
        st.error(f"API 錯誤：{e}"); st.stop()

elapsed = round(time.time() - t0, 1)

st.markdown("### 🧠 企業發展與雙軸轉型維度 (軟指標)")
s_cols = st.columns(4)
for i, dim in enumerate(SOFT_DIMS):
    entry = soft[dim]; sc = int(entry["score"])
    with s_cols[i]:
        st.markdown(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:12px 14px;'>"
            f"<div style='font-size:.85rem;color:#555;margin-bottom:4px;font-weight:600;'>{dim}</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{_c(sc)};line-height:1;'>{sc}</div>"
            f"<div style='font-size:.75rem;color:#888;margin-top:3px;'>{SCORE_LABEL[sc]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("AI 推論依據"):
            st.caption(f"**原文引用**：{entry.get('quote', '—')}")
            st.caption(f"**審核推論**：{entry.get('reason', '—')}")

st.divider()

# Final result
res = compute(rule_res, soft)

st.markdown("### 🏆 綜合風險評級")
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
    st.caption(f"加權分：**{res.wavg:.3f}** / 耗時：{elapsed}s")
with c3:
    red   = [d for d in ALL_DIMS if res.scores[d] >= 4]
    amber = [d for d in ALL_DIMS if 3 <= res.scores[d] < 4]
    if red:   st.error(f"🚨 高風險維度 (≥4分) 需強制會商：{', '.join(red)}")
    if amber: st.warning(f"⚠️ 需關注維度 (3–4分) 建議增提擔保：{', '.join(amber)}")
    if not red and not amber: st.success("✅ 所有維度皆落在安全區間 (低於3分)，建議核貸。")

st.plotly_chart(bar_fig(res), use_container_width=True)

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
    "規則引擎解析": {d: rule_res[d].note for d in HARD_DIMS},
    "定性與轉型分析":  {d: soft[d] for d in SOFT_DIMS},
}
st.download_button(
    "📥 下載完整評估報告 (JSON)",
    data=json.dumps(report, ensure_ascii=False, indent=2),
    file_name="credit_risk_report.json",
    mime="application/json",
)