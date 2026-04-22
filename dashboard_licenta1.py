import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import anthropic
import time

st.set_page_config(
    page_title="Comportament Bursier IT · S&P 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0b0f19; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }
[data-testid="stSidebar"] label {
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: rgba(99,179,237,0.15) !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    color: #63b3ed !important;
}

.kpi-row { display: flex; gap: 12px; margin: 1.5rem 0; }
.kpi-card {
    flex: 1; background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 1rem 1.25rem;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(255,255,255,0.15); }
.kpi-label {
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.08em; color: #64748b;
    margin-bottom: 6px; font-family: 'JetBrains Mono', monospace;
}
.kpi-value { font-size: 1.45rem; font-weight: 600; letter-spacing: -0.02em; }
.kpi-value.pos { color: #34d399; }
.kpi-value.neg { color: #f87171; }
.kpi-sub { font-size: 11px; color: #475569; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

.section-label {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.12em; color: #475569;
    padding: 1.8rem 0 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1rem;
}

.page-header {
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.page-title { font-size: 1.25rem; font-weight: 600; color: #f1f5f9; letter-spacing: -0.02em; margin: 0; }
.page-meta { font-size: 11px; color: #475569; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }

[data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding-bottom: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    color: #64748b !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em; padding: 6px 14px !important;
    border: none !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(99,179,237,0.1) !important;
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; overflow: hidden;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px; overflow: hidden;
}

/* ── AI TAB STYLES ─────────────────────────────────────────────── */
.ai-header-box {
    background: linear-gradient(135deg, #0f1929 0%, #0d1f2d 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.ai-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 6px #34d399;
    animation: pulse-green 2s infinite;
    flex-shrink: 0;
}
@keyframes pulse-green { 0%,100%{opacity:1} 50%{opacity:0.3} }
.ai-header-text { font-size: 12px; color: #64748b; font-family: 'JetBrains Mono', monospace; }
.ai-header-title { font-size: 15px; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em; }

.news-card {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: rgba(99,179,237,0.2); }
.news-badge {
    display: inline-block;
    font-size: 9px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    padding: 2px 8px; border-radius: 4px;
    margin-right: 8px;
}
.badge-bbc { background: rgba(220,38,38,0.15); color: #f87171; border: 1px solid rgba(220,38,38,0.2); }
.badge-cnn { background: rgba(220,38,38,0.15); color: #f87171; border: 1px solid rgba(220,38,38,0.2); }
.badge-reuters { background: rgba(249,115,22,0.15); color: #fb923c; border: 1px solid rgba(249,115,22,0.2); }
.badge-other { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.15); }
.news-title { font-size: 13px; color: #cbd5e1; line-height: 1.5; margin: 6px 0 4px; }
.news-meta { font-size: 10px; color: #334155; font-family: 'JetBrains Mono', monospace; }
.impact-pos { color: #34d399; }
.impact-neg { color: #f87171; }
.impact-neu { color: #64748b; }

.ai-reco-box {
    background: #0a1628;
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ai-reco-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace; color: #3b82f6;
    margin-bottom: 10px;
}
.ai-reco-text {
    font-size: 14px; color: #94a3b8; line-height: 1.7;
}
.signal-row {
    display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;
}
.signal-pill {
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    padding: 5px 14px; border-radius: 6px; font-weight: 500;
}
.pill-buy   { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.pill-hold  { background: rgba(251,191,36,0.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.pill-watch { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }

.portfolio-metric {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.pm-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }
.pm-val { font-size: 1.3rem; font-weight: 600; color: #e2e8f0; }
.pm-sub { font-size: 11px; color: #334155; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
COMPANII = {
    "Microsoft":  "MSFT",
    "Alphabet":   "GOOGL",
    "Oracle":     "ORCL",
    "Adobe":      "ADBE",
    "Salesforce": "CRM",
}
START, END = "2015-01-01", "2024-12-31"

PALETTE = {
    "Microsoft":  "#3b82f6",
    "Alphabet":   "#22c55e",
    "Oracle":     "#ef4444",
    "Adobe":      "#f97316",
    "Salesforce": "#a78bfa",
}

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f172a",
    font=dict(family="Inter, sans-serif", color="#64748b", size=11),
    margin=dict(l=48, r=16, t=36, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=11, color="#94a3b8"),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    ),
    hoverlabel=dict(
        bgcolor="#1e293b", bordercolor="rgba(255,255,255,0.1)",
        font=dict(color="#f1f5f9", size=12),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=False, tickfont=dict(size=11)),
)

RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "CNN": "http://rss.cnn.com/rss/edition_technology.rss",
    "Reuters": "https://feeds.reuters.com/reuters/technologyNews",
}

TICKERS_KEYWORDS = {
    "Microsoft": ["Microsoft", "MSFT", "Azure", "Copilot", "Bing"],
    "Alphabet":  ["Alphabet", "Google", "GOOGL", "Gemini", "YouTube", "DeepMind"],
    "Oracle":    ["Oracle", "ORCL", "Larry Ellison"],
    "Adobe":     ["Adobe", "ADBE", "Photoshop", "Firefly"],
    "Salesforce":["Salesforce", "CRM", "Marc Benioff", "Slack"],
}

def apply_base(fig, **kwargs):
    fig.update_layout(**{**PLOT_BASE, **kwargs})
    return fig

def col(comp):
    return PALETTE.get(comp, "#94a3b8")

# ── DATA ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    raw = yf.download(list(COMPANII.values()), start=START, end=END, auto_adjust=True, progress=False)
    df = raw["Close"].copy()
    df.columns = list(COMPANII.keys())
    df.index = pd.to_datetime(df.index)
    return df.dropna(how="all")

@st.cache_data(show_spinner=False)
def calc_annual(df):
    rand = df.groupby(df.index.year).last().pct_change() * 100
    vol  = df.groupby(df.index.year).apply(lambda x: x.pct_change().std() * np.sqrt(252) * 100)
    return rand, vol

@st.cache_data(show_spinner=False)
def calc_beta_all(df_raw):
    sp = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"]
    sp.index = pd.to_datetime(sp.index)
    sp_r = sp.pct_change().dropna()
    result = []
    for an in range(2015, 2025):
        row = {"An": an}
        sp_an = sp_r[sp_r.index.year == an]
        for comp in COMPANII.keys():
            cr = df_raw[comp].pct_change().dropna()
            cr_an = cr[cr.index.year == an]
            both = pd.concat([cr_an, sp_an], axis=1).dropna()
            both.columns = ["c", "sp"]
            if len(both) < 20:
                row[comp] = None
                continue
            row[comp] = round(both.cov().iloc[0, 1] / both["sp"].var(), 3)
        result.append(row)
    return pd.DataFrame(result).set_index("An")

# ── NEWS ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(companii_selectate):
    """Fetch RSS feeds și filtrează știrile relevante pentru companiile selectate."""
    articles = []
    keywords_active = []
    for comp in companii_selectate:
        keywords_active.extend(TICKERS_KEYWORDS.get(comp, []))

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title} {summary}".lower()
                matched = [k for k in keywords_active if k.lower() in text]
                if matched or len(articles) < 5:
                    articles.append({
                        "source": source,
                        "title": title,
                        "summary": summary[:200] if summary else "",
                        "link": entry.get("link", ""),
                        "published": entry.get("published", "recent"),
                        "matched_companies": list(set(
                            comp for comp in companii_selectate
                            if any(k.lower() in text for k in TICKERS_KEYWORDS.get(comp, []))
                        )),
                    })
                if len(articles) >= 20:
                    break
        except Exception:
            pass
    return articles[:15]

def get_impact_class(title: str) -> str:
    pos_words = ["surge", "gain", "rise", "beat", "growth", "profit", "record", "strong", "boost", "rally", "up", "positive", "creștere", "profit"]
    neg_words = ["fall", "drop", "loss", "decline", "miss", "cut", "layoff", "lawsuit", "fine", "down", "scade", "pierdere", "anchetă"]
    t = title.lower()
    if any(w in t for w in pos_words): return "pos"
    if any(w in t for w in neg_words): return "neg"
    return "neu"

def news_badge_class(source: str) -> str:
    mapping = {"BBC": "badge-bbc", "CNN": "badge-cnn", "Reuters": "badge-reuters"}
    return mapping.get(source, "badge-other")

# ── AI ANALYSIS ───────────────────────────────────────────────────────────────
def build_portfolio_context(df, df_r, df_v, df_b, companii):
    """Construiește contextul portofoliului pentru prompt-ul AI."""
    lines = []
    lines.append("DATE PORTOFOLIU (2015-2024):")
    for comp in companii:
        s_comp = df[comp].dropna()
        if s_comp.empty:
            continue
        total = (s_comp.iloc[-1] / s_comp.iloc[0] - 1) * 100
        vol_med = df_v[comp].mean() if comp in df_v.columns else 0
        beta_med = df_b[comp].mean() if comp in df_b.columns else 0
        last_rand = df_r[comp].iloc[-1] if comp in df_r.columns else 0
        lines.append(
            f"- {comp} ({COMPANII[comp]}): randament cumulat {total:+.1f}%, "
            f"volatilitate medie {vol_med:.1f}%, beta mediu {beta_med:.2f}, "
            f"ultimul randament anual {last_rand:+.1f}%"
        )
    return "\n".join(lines)

def analyze_with_claude(portfolio_context: str, news_titles: list, companii: list, api_key: str) -> str:
    """Apelează Claude API pentru analiza portofoliului."""
    client = anthropic.Anthropic(api_key=api_key)

    news_text = "\n".join(f"- [{a['source']}] {a['title']}" for a in news_titles[:10])

    prompt = f"""Ești un analist financiar specializat în companii tech din S&P 500.
Analizează portofoliul de mai jos pe baza datelor istorice și a știrilor recente.

{portfolio_context}

ȘTIRI RECENTE (RSS live):
{news_text if news_text else "Nu există știri disponibile momentan."}

COMPANII ÎN PORTOFOLIU: {', '.join(companii)}

Oferă o analiză concisă în română care să includă:
1. **Recomandări per companie** (Cumpără / Menține / Monitorizează) cu justificare scurtă
2. **Impactul știrilor recente** asupra portofoliului
3. **Risc principal** de monitorizat
4. **Sugestie de rebalansare** dacă este cazul

Fii direct, profesionist, și folosește date concrete din contextul oferit.
Răspuns maxim: 350 cuvinte."""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
with st.spinner("Se încarcă datele bursiere…"):
    df_raw = load_data()
    df_rand_all, df_vol_all = calc_annual(df_raw)
    df_beta_all = calc_beta_all(df_raw)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filtre")
    companii = st.multiselect("Companii", list(COMPANII.keys()), default=list(COMPANII.keys()))
    perioada = st.slider("Perioadă", 2015, 2024, (2015, 2024))
    st.markdown("---")

    st.markdown("""
    <div style='font-size:10px; color:#334155; font-family:"JetBrains Mono",monospace; line-height:2; text-transform:uppercase; letter-spacing:0.05em;'>
    ── AI ADVISOR ──
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Obține cheia de pe console.anthropic.com"
    )

    if api_key:
        st.markdown("""
        <div style='font-size:10px; color:#34d399; font-family:"JetBrains Mono",monospace; margin-top:4px;'>
        ✓ API KEY CONFIGURATĂ
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#334155; line-height:2;'>
    Sursă · Yahoo Finance<br>Perioadă · 2015 – 2024<br>Benchmark · S&P 500<br>Știri · BBC / CNN / Reuters
    </div>
    """, unsafe_allow_html=True)

if not companii:
    st.warning("Selectează cel puțin o companie.")
    st.stop()

# ── FILTER ─────────────────────────────────────────────────────────────────────
df = df_raw[(df_raw.index.year >= perioada[0]) & (df_raw.index.year <= perioada[1])][companii]
df_r = df_rand_all.loc[perioada[0]:perioada[1], companii]
df_v = df_vol_all.loc[perioada[0]:perioada[1], companii]
df_b = df_beta_all.loc[perioada[0]:perioada[1], companii]

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <p class="page-title">Comportamentul bursier al companiilor IT din S&P 500</p>
  <p class="page-meta">Studiu de caz · Lucrare de licență · 2015 – 2024</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ────────────────────────────────────────────────────────────────────────
kpi_html = '<div class="kpi-row">'
for comp in companii:
    s_comp = df[comp].dropna()
    if s_comp.empty:
        continue
    total = (s_comp.iloc[-1] / s_comp.iloc[0] - 1) * 100
    vm    = df_v[comp].mean()
    cls   = "pos" if total >= 0 else "neg"
    sign  = "+" if total >= 0 else ""
    kpi_html += f"""
    <div class="kpi-card">
      <div class="kpi-label">{comp}</div>
      <div class="kpi-value {cls}">{sign}{total:.1f}%</div>
      <div class="kpi-sub">vol. medie {vm:.1f}%</div>
    </div>"""
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Evoluție prețuri",
    "Randamente",
    "Volatilitate & beta",
    "Risc vs randament",
    "🤖 AI Advisor",
])

# ═══ TAB 1 ════════════════════════════════════════════════════════════════════
with t1:
    c_left, c_right = st.columns([2, 1], gap="medium")

    with c_left:
        st.markdown('<div class="section-label">Prețuri normalizate · bază 100 = ian. 2015</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for comp in companii:
            s = df[comp].dropna()
            if s.empty:
                continue
            n = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=n.index, y=n.round(2),
                name=comp, line=dict(color=col(comp), width=1.8),
                hovertemplate=f"<b>{comp}</b><br>%{{x|%d %b %Y}}<br>Indice: %{{y:.1f}}<extra></extra>",
            ))
        if perioada[0] <= 2020 <= perioada[1]:
            fig.add_vrect(x0="2020-02-20", x1="2020-04-20",
                          fillcolor="rgba(248,113,113,0.06)", line_width=0)
            fig.add_annotation(x="2020-03-15", y=0.98, yref="paper",
                               text="COVID-19", showarrow=False,
                               font=dict(size=9, color="#f87171", family="JetBrains Mono"))
        fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.1)", line_width=1)
        apply_base(fig, height=380, yaxis_title="Indice (bază 100)")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-label">Randament cumulat</div>', unsafe_allow_html=True)
        totals = sorted(
            [(c, round((df[c].dropna().iloc[-1] / df[c].dropna().iloc[0] - 1) * 100, 1))
             for c in companii if not df[c].dropna().empty],
            key=lambda x: x[1]
        )
        mx = max(abs(v) for _, v in totals)
        fig2 = go.Figure(go.Bar(
            x=[v for _, v in totals], y=[c for c, _ in totals],
            orientation="h",
            marker=dict(color=[col(c) for c, _ in totals], line_width=0),
            text=[f"{v:+.0f}%" for _, v in totals],
            textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono", color="#94a3b8"),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
        ))
        apply_base(fig2, height=380,
           xaxis_title="Randament cumulat (%)",
           xaxis_range=[-10, mx * 1.35],
           yaxis={**PLOT_BASE["yaxis"], "tickfont": dict(size=11, color="#94a3b8")})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-label">Sinteză numerică</div>', unsafe_allow_html=True)
    tabel = pd.DataFrame({
        "Preț start ($)":         [round(df[c].dropna().iloc[0], 2) if not df[c].dropna().empty else None for c in companii],
        "Preț final ($)":          [round(df[c].dropna().iloc[-1], 2) if not df[c].dropna().empty else None for c in companii],
        "Randament cumulat (%)":   [round((df[c].dropna().iloc[-1] / df[c].dropna().iloc[0] - 1) * 100, 1) if not df[c].dropna().empty else None for c in companii],
        "Volatilitate medie (%)":  [round(df_v[c].mean(), 1) for c in companii],
    }, index=companii)
    tabel.index.name = "Companie"
    st.dataframe(tabel, use_container_width=True)

# ═══ TAB 2 ════════════════════════════════════════════════════════════════════
with t2:
    c_left, c_right = st.columns(2, gap="medium")

    with c_left:
        st.markdown('<div class="section-label">Randamente anuale (%)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for comp in companii:
            fig.add_trace(go.Scatter(
                x=df_r.index.astype(str), y=df_r[comp].round(1),
                mode="lines+markers", name=comp,
                line=dict(color=col(comp), width=1.8), marker=dict(size=5),
                hovertemplate=f"<b>{comp}</b><br>An: %{{x}}<br>%{{y:.1f}}%<extra></extra>",
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.12)", line_width=1)
        apply_base(fig, height=360, yaxis_title="Randament (%)",
                   xaxis=dict(**PLOT_BASE["xaxis"], type="category"))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-label">Heatmap randamente</div>', unsafe_allow_html=True)
        z = df_r[companii].T.values.tolist()
        txt = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
        vmax = float(np.nanmax(np.abs(df_r[companii].values)))
        fig2 = go.Figure(go.Heatmap(
            z=z, x=df_r.index.astype(str).tolist(), y=companii,
            text=txt, texttemplate="%{text}",
            textfont=dict(size=10, family="JetBrains Mono"),
            colorscale=[[0, "#991b1b"], [0.42, "#1e293b"], [0.58, "#1e293b"], [1, "#065f46"]],
            zmin=-vmax, zmax=vmax, showscale=False,
            hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.2f}%<extra></extra>",
        ))
        apply_base(fig2, height=360,
                   xaxis=dict(**PLOT_BASE["xaxis"], type="category"),
                   margin=dict(l=90, r=16, t=36, b=40))
        st.plotly_chart(fig2, use_container_width=True)

# ═══ TAB 3 ════════════════════════════════════════════════════════════════════
with t3:
    c_left, c_right = st.columns(2, gap="medium")

    with c_left:
        st.markdown('<div class="section-label">Volatilitate anualizată (%)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for comp in companii:
            fig.add_trace(go.Scatter(
                x=df_v.index.astype(str), y=df_v[comp].round(2),
                mode="lines+markers", name=comp,
                line=dict(color=col(comp), width=1.8), marker=dict(size=5),
                hovertemplate=f"<b>{comp}</b><br>An: %{{x}}<br>%{{y:.2f}}%<extra></extra>",
            ))
        apply_base(fig, height=360, yaxis_title="Volatilitate (%)",
                   xaxis=dict(**PLOT_BASE["xaxis"], type="category"))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-label">Coeficient Beta față de S&P 500</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for comp in companii:
            if comp not in df_b.columns:
                continue
            fig2.add_trace(go.Scatter(
                x=df_b.index.astype(str), y=pd.to_numeric(df_b[comp], errors="coerce").round(3),
                mode="lines+markers", name=comp,
                line=dict(color=col(comp), width=1.8),
                marker=dict(size=5, symbol="square"),
                hovertemplate=f"<b>{comp}</b><br>An: %{{x}}<br>β = %{{y:.3f}}<extra></extra>",
            ))
        fig2.add_hline(y=1, line_dash="dash", line_color="rgba(255,255,255,0.18)",
                       annotation_text="β = 1  ", annotation_position="right",
                       annotation_font=dict(size=10, color="#64748b"))
        apply_base(fig2, height=360, yaxis_title="Beta (β)",
                   xaxis=dict(**PLOT_BASE["xaxis"], type="category"))
        st.plotly_chart(fig2, use_container_width=True)

# ═══ TAB 4 ════════════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="section-label">Raport risc–randament · medie perioadă selectată</div>', unsafe_allow_html=True)

    scatter = [
        (c, round(df_v[c].mean(), 2), round(df_r[c].mean(), 2))
        for c in companii
        if c in df_r.columns and c in df_v.columns
    ]

    fig = go.Figure()
    for comp, vol, rand in scatter:
        fig.add_trace(go.Scatter(
            x=[vol], y=[rand], mode="markers+text", name=comp,
            text=[comp], textposition="top center",
            textfont=dict(size=11, color=col(comp), family="Inter"),
            marker=dict(size=16, color=col(comp),
                        line=dict(width=1.5, color="rgba(255,255,255,0.12)")),
            hovertemplate=f"<b>{comp}</b><br>Vol: %{{x:.2f}}%<br>Rand: %{{y:.2f}}%<extra></extra>",
            showlegend=False,
        ))

    if scatter:
        vols  = [v for _, v, _ in scatter]
        rands = [r for _, _, r in scatter]
        fig.add_hline(y=np.mean(rands), line_dash="dot", line_color="rgba(255,255,255,0.07)")
        fig.add_vline(x=np.mean(vols),  line_dash="dot", line_color="rgba(255,255,255,0.07)")
        px = (max(vols)  - min(vols))  * 0.18 or 1
        py = (max(rands) - min(rands)) * 0.25 or 5
        fig.update_layout(
            xaxis_range=[min(vols) - px, max(vols) + px * 4],
            yaxis_range=[min(rands) - py, max(rands) + py * 2],
        )

    apply_base(fig, height=440,
               xaxis_title="Volatilitate medie anualizată (%)",
               yaxis_title="Randament mediu anual (%)")
    st.plotly_chart(fig, use_container_width=True)

    df_sc = pd.DataFrame(
        [(c, v, r, round(r / v, 3) if v else None) for c, v, r in scatter],
        columns=["Companie", "Volatilitate medie (%)", "Randament mediu (%)", "Sharpe simplificat"],
    ).set_index("Companie").sort_values("Randament mediu (%)", ascending=False)
    st.dataframe(df_sc, use_container_width=True)

# ═══ TAB 5 — AI ADVISOR ═══════════════════════════════════════════════════════
with t5:

    # Header
    st.markdown("""
    <div class="ai-header-box">
        <div class="ai-dot"></div>
        <div>
            <div class="ai-header-title">AI Portfolio Advisor</div>
            <div class="ai-header-text">Analiză AI · Știri live RSS · BBC · CNN · Reuters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div style='background:#0f172a; border:1px solid rgba(99,179,237,0.15); border-radius:10px;
                    padding:1.5rem; text-align:center; color:#475569;
                    font-family:"JetBrains Mono",monospace; font-size:12px;'>
            ── CONFIGURARE NECESARĂ ──<br><br>
            <span style='color:#64748b;'>Adaugă cheia Claude API în sidebar pentru a activa analiza AI.</span><br>
            <span style='color:#334155; font-size:11px; margin-top:8px; display:block;'>
            Obține cheia gratuit pe console.anthropic.com
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_news, col_reco = st.columns([1, 1], gap="medium")

        with col_news:
            st.markdown('<div class="section-label">Știri relevante · actualizat la 5 min</div>', unsafe_allow_html=True)

            with st.spinner("Se încarcă știrile RSS…"):
                articles = fetch_news(tuple(companii))

            if not articles:
                st.markdown("""
                <div style='color:#334155; font-size:12px; font-family:"JetBrains Mono",monospace; padding:1rem 0;'>
                Nu s-au putut prelua știri momentan. Verifică conexiunea.
                </div>
                """, unsafe_allow_html=True)
            else:
                for art in articles:
                    impact = get_impact_class(art["title"])
                    impact_icon = "▲" if impact == "pos" else ("▼" if impact == "neg" else "●")
                    impact_cls = f"impact_{impact}"
                    badge_cls = news_badge_class(art["source"])
                    matched_str = " · ".join(art["matched_companies"]) if art["matched_companies"] else "General"

                    st.markdown(f"""
                    <div class="news-card">
                        <span class="news-badge {badge_cls}">{art['source']}</span>
                        <span class="news-badge badge-other" style="font-size:9px;">{matched_str}</span>
                        <div class="news-title">{art['title']}</div>
                        <div class="news-meta">
                            <span class="{impact_cls}">{impact_icon}</span>
                            &nbsp;{art['published'][:16] if len(art.get('published','')) > 10 else 'recent'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_reco:
            st.markdown('<div class="section-label">Analiză & recomandări AI</div>', unsafe_allow_html=True)

            # Metrics rapide
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                best = max(
                    [c for c in companii if not df[c].dropna().empty],
                    key=lambda c: (df[c].dropna().iloc[-1] / df[c].dropna().iloc[0] - 1),
                    default=companii[0]
                )
                best_ret = (df[best].dropna().iloc[-1] / df[best].dropna().iloc[0] - 1) * 100
                st.markdown(f"""
                <div class="portfolio-metric">
                    <div class="pm-label">Top performer</div>
                    <div class="pm-val" style="color:{col(best)};">{best[:4]}</div>
                    <div class="pm-sub">+{best_ret:.0f}% total</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                avg_vol = np.mean([df_v[c].mean() for c in companii if c in df_v.columns])
                st.markdown(f"""
                <div class="portfolio-metric">
                    <div class="pm-label">Vol. medie</div>
                    <div class="pm-val">{avg_vol:.1f}%</div>
                    <div class="pm-sub">anualizată</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                avg_beta = np.nanmean([pd.to_numeric(df_b[c], errors="coerce").mean() for c in companii if c in df_b.columns])
                risk_level = "Ridicat" if avg_beta > 1.2 else ("Moderat" if avg_beta > 0.9 else "Scăzut")
                st.markdown(f"""
                <div class="portfolio-metric">
                    <div class="pm-label">Beta mediu</div>
                    <div class="pm-val">{avg_beta:.2f}</div>
                    <div class="pm-sub">{risk_level}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Buton analiză AI
            if st.button("⟳  Generează analiza AI", use_container_width=True, type="primary"):
                with st.spinner("Claude analizează portofoliul și știrile…"):
                    try:
                        articles_for_ai = fetch_news(tuple(companii))
                        portfolio_ctx = build_portfolio_context(df, df_r, df_v, df_b, companii)
                        analysis = analyze_with_claude(portfolio_ctx, articles_for_ai, companii, api_key)
                        st.session_state["ai_analysis"] = analysis
                        st.session_state["ai_timestamp"] = time.strftime("%H:%M:%S")
                    except Exception as e:
                        st.error(f"Eroare API: {str(e)}")

            # Afișează analiza salvată
            if "ai_analysis" in st.session_state:
                ts = st.session_state.get("ai_timestamp", "—")
                st.markdown(f"""
                <div class="ai-reco-box">
                    <div class="ai-reco-title">── ANALIZĂ AI · {ts} ──</div>
                    <div class="ai-reco-text">{st.session_state['ai_analysis'].replace(chr(10), '<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

                # Semnale rapide extrase
                signals_html = '<div class="signal-row">'
                for comp in companii:
                    analysis_text = st.session_state["ai_analysis"].lower()
                    if "cumpăr" in analysis_text and comp.lower() in analysis_text:
                        signals_html += f'<span class="signal-pill pill-buy">↑ {comp}</span>'
                    elif "monitorizea" in analysis_text and comp.lower() in analysis_text:
                        signals_html += f'<span class="signal-pill pill-watch">⚠ {comp}</span>'
                    else:
                        signals_html += f'<span class="signal-pill pill-hold">≈ {comp}</span>'
                signals_html += "</div>"
                st.markdown(signals_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='color:#334155; font-size:12px; font-family:"JetBrains Mono",monospace;
                            padding:1.5rem; text-align:center; border:1px solid rgba(255,255,255,0.04);
                            border-radius:8px; margin-top:1rem;'>
                Apasă butonul pentru a genera analiza AI bazată<br>pe datele portofoliului și știrile live.
                </div>
                """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem; padding-top:1rem;
     border-top:1px solid rgba(255,255,255,0.05);
     text-align:center; font-family:"JetBrains Mono",monospace;
     font-size:10px; color:#1e293b; letter-spacing:0.1em;'>
COMPORTAMENTUL BURSIER AL COMPANIILOR IT DIN S&P 500
&nbsp;·&nbsp; LUCRARE DE LICENȚĂ &nbsp;·&nbsp; DATE: YAHOO FINANCE &nbsp;·&nbsp; AI: ANTHROPIC CLAUDE
</div>
""", unsafe_allow_html=True)
