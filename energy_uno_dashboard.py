import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Energy Audit Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0a1628 !important;
    border-right: 1px solid #1e2d45;
    min-width: 240px !important;
    max-width: 240px !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .sidebar-logo span { color: #3fb68b !important; }
section[data-testid="stSidebar"] .sidebar-sub {
    font-size: 0.7rem;
    color: #8b949e !important;
    margin-top: -4px;
    margin-bottom: 24px;
}

/* ── Main area ── */
.main .block-container {
    background-color: #0d1117;
    padding: 1.5rem 2rem;
    max-width: 100%;
}

/* ── Metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 22px;
    height: 100%;
}
.metric-card .label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.75rem;
    margin-top: 4px;
}
.badge-orange { background: #ff8c00; border-radius: 50%; width: 32px; height: 32px;
    display:inline-flex; align-items:center; justify-content:center; font-size:0.85rem; }

/* ── Section cards ── */
.section-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px 22px;
}
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 12px;
}

/* ── Status pills ── */
.status-in-progress { color: #3fb68b; font-weight: 600; }
.status-ready { color: #3fb68b; }
.status-active { color: #3fb68b; }

/* ── Issue table ── */
.issue-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.82rem;
    color: #c9d1d9;
}
.issue-row:last-child { border-bottom: none; }
.details-btn {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #c9d1d9;
    cursor: pointer;
}

/* ── Recommendation card ── */
.rec-card {
    padding: 10px 0;
    border-bottom: 1px solid #21262d;
}
.rec-card:last-child { border-bottom: none; }
.rec-title { font-size: 0.88rem; font-weight: 600; color: #e6edf3; }
.rec-sub { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }
.rec-sub span { color: #3fb68b; font-weight: 600; }

/* ── Sidebar nav items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 0.87rem;
    cursor: pointer;
    margin-bottom: 3px;
    color: #8b949e;
    transition: background 0.15s;
}
.nav-item.active {
    background: #1f3a5c;
    color: #58a6ff !important;
    font-weight: 500;
}
.nav-item:hover { background: #161b22; color: #e6edf3; }

/* ── System status ── */
.sys-status {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.78rem;
}
.sys-status .sys-title { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #8b949e; margin-bottom: 8px; }
.dot-green { color: #3fb68b; }

/* ── Audit bar ── */
.audit-bar {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    font-size: 0.83rem;
}

/* ── AI confidence gauge ── */
.confidence-val { font-size: 1.9rem; font-weight: 700; color: #58a6ff; }
.confidence-label { font-size: 0.7rem; color: #8b949e; }

/* hide streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 14px 8px 14px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
            <span style="font-size:1.3rem;">⚡</span>
            <span class="sidebar-logo">ENERGY <span>UNO</span></span>
        </div>
        <div class="sidebar-sub">AI Energy Audit Platform</div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("📊", "Dashboard Overview", True),
        ("🗂️", "Data Ingestion", False),
        ("🧠", "AI Analysis Center", False),
        ("📋", "Audit Reports", False),
        ("⚙️", "Project Management", False),
        ("👤", "User Profile", False),
    ]
    for icon, label, active in nav_items:
        cls = "nav-item active" if active else "nav-item"
        st.markdown(f'<div class="{cls}">{icon}&nbsp;&nbsp;{label}</div>', unsafe_allow_html=True)

    st.markdown("<br>" * 6, unsafe_allow_html=True)

    st.markdown("""
    <div class="sys-status">
        <div class="sys-title">System Status</div>
        <div style="margin-bottom:5px;"><span class="dot-green">●</span>&nbsp; AI Models: <strong style="color:#3fb68b;">Ready</strong></div>
        <div><span class="dot-green">●</span>&nbsp; Data Connection: <strong style="color:#3fb68b;">Active</strong></div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family: Space Grotesk, sans-serif; font-size:1.9rem; font-weight:700;
    color:#e6edf3; margin-bottom:2px;'>Dashboard Overview</h1>
<p style='color:#8b949e; font-size:0.85rem; margin-bottom:18px;'>Welcome back, Energy Uno Team!</p>
""", unsafe_allow_html=True)

# ── Audit bar ─────────────────────────────────────────────────────────────────
audit_col1, audit_col2 = st.columns([2, 3])
with audit_col1:
    audit_site = st.selectbox(
        "Current Audit",
        ["Downtown Corporate Tower", "West Side Mall", "Harbor Office Complex"],
        label_visibility="visible",
    )
with audit_col2:
    st.markdown("""
    <div style="padding-top:28px; text-align:right; font-size:0.83rem; color:#8b949e;">
        Audit Status: <span class="status-in-progress">In Progress</span>
        &nbsp;|&nbsp; Last Updated: Oct 26, 2025
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="metric-card">
        <div class="label">Curr. Consumption</div>
        <div class="value">1,200 kWh</div>
        <div class="sub" style="color:#e05252;">↗ 5% (Oct)</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="metric-card" style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <div class="label">Predicted Inefficiency</div>
            <div class="value">18%</div>
            <div class="sub" style="color:#8b949e;">AI Estimation</div>
        </div>
        <div class="badge-orange">%</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="metric-card">
        <div class="label">Identified Savings (Est.)</div>
        <div class="value">$15,300 /yr</div>
        <div class="sub" style="color:#3fb68b;">↘ Potential</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="metric-card" style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <div class="label">AI Confidence Score</div>
            <div class="confidence-val">92%</div>
            <div class="confidence-label">High Accuracy</div>
        </div>
        <div style="font-size:2rem;">🎯</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ─── Charts Row ───────────────────────────────────────────────────────────────
chart_col, donut_col = st.columns([3, 2])

# ── Line Chart ────────────────────────────────────────────────────────────────
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
historical = [190, 145, 165, 200, 230, 250, 270, 285, 275, 250, None, None]
ai_baseline = [175, 185, 195, 210, 220, 230, 245, 255, 260, 265, 280, 300]
savings_upper = [None]*6 + [245, 255, 260, 265, 280, 300]
savings_lower = [None]*6 + [130, 120, 110, 100, 95, 90]

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=months, y=historical,
    name="Historical Usage",
    line=dict(color="#58a6ff", width=2.5),
    mode="lines",
))
fig_line.add_trace(go.Scatter(
    x=months, y=ai_baseline,
    name="AI Predicted Baseline",
    line=dict(color="#3fb68b", width=2, dash="dash"),
    mode="lines",
))
fig_line.add_trace(go.Scatter(
    x=months + months[::-1],
    y=savings_upper + savings_lower[::-1],
    fill="toself",
    fillcolor="rgba(63,182,139,0.18)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Forecasted Savings",
    showlegend=True,
))

fig_line.update_layout(
    title=dict(text="Energy Consumption Trend (kWh)", font=dict(size=13, color="#e6edf3")),
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e", size=11),
    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10),
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#21262d", showline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#21262d", showline=False, title="Energy (kWh)",
               titlefont=dict(size=10)),
    margin=dict(l=10, r=10, t=50, b=30),
    height=310,
)

with chart_col:
    st.markdown('<div class="section-card" style="padding:14px 16px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── Donut Chart ───────────────────────────────────────────────────────────────
labels = ["HVAC", "Lighting", "Equipment", "Other"]
values = [40, 25, 20, 15]
colors = ["#58a6ff", "#3fb68b", "#4a9eed", "#6e7681"]

fig_donut = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.62,
    marker=dict(colors=colors),
    textinfo="label+percent",
    textfont=dict(size=11, color="#e6edf3"),
    hovertemplate="%{label}: %{value}%<extra></extra>",
)])
fig_donut.add_annotation(
    text="Total<br><b>313%</b>",
    x=0.5, y=0.5, showarrow=False,
    font=dict(size=13, color="#e6edf3"),
    align="center",
)
fig_donut.update_layout(
    title=dict(text="Energy Breakdown by End-Use (Estimated)",
               font=dict(size=12, color="#e6edf3")),
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=10),
    height=310,
    font=dict(color="#8b949e"),
)

with donut_col:
    st.markdown('<div class="section-card" style="padding:14px 16px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ─── Bottom Row ───────────────────────────────────────────────────────────────
issues_col, recs_col = st.columns([3, 2])

# ── AI-Identified Issues ──────────────────────────────────────────────────────
issues = [
    ("HVAC running during unoccupied hours - 10/24 2AM", "⚠️"),
    ("Unusual lighting load spike - 10/25 11PM", "⚠️"),
    ("Unusual lighting load spike - 10/25 2PM", "⚠️"),
    ("HVAC running during unoccupied hours - 10/26 1PM", "⚠️"),
]

with issues_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI-Identified Potential Issues</div>', unsafe_allow_html=True)

    header_cols = st.columns([5, 1, 1])
    header_cols[0].markdown('<span style="font-size:0.72rem;color:#8b949e;">Anomalies found</span>', unsafe_allow_html=True)
    header_cols[2].markdown('<span style="font-size:0.72rem;color:#8b949e;">Validate</span>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d;margin:6px 0 4px 0;'>", unsafe_allow_html=True)

    for i, (issue, icon) in enumerate(issues):
        cols = st.columns([5, 1, 1])
        cols[0].markdown(f'<span style="font-size:0.82rem;color:#c9d1d9;">{issue}</span>', unsafe_allow_html=True)
        cols[1].button("Details", key=f"details_{i}", use_container_width=True)
        cols[2].markdown(f'<div style="text-align:center;padding-top:4px;">{icon}</div>', unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#21262d;margin:3px 0;'>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Recent Recommendations ────────────────────────────────────────────────────
recommendations = [
    ("LED Retrofit Plan", "$1.5%", "25 hrs."),
    ("Smart Thermostat Install", "$1.5%", "7 hrs."),
    ("HVAC Schedule Optimization", "$2.1%", "12 hrs."),
    ("Window Insulation Upgrade", "$0.8%", "40 hrs."),
]

with recs_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent Recommendations ✨</div>', unsafe_allow_html=True)

    for title, savings, payback in recommendations:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">{title}</div>
            <div class="rec-sub">Estimated savings: <span>{savings}</span> | Payback: {payback}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
