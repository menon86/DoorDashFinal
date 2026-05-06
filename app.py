import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="DoorDash Delivery Delay Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d1b2a; }
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  h1, h2, h3 { color: #ffffff; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #162032; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #8fa8c8; background-color: transparent; border-radius: 6px; padding: 8px 20px; font-weight: 500; }
  .stTabs [aria-selected="true"] { background-color: #1e3a5f !important; color: #ffffff !important; }
  .metric-card { background: #162032; border-radius: 10px; padding: 20px 24px; border-left: 4px solid #ff6b35; margin-bottom: 8px; }
  .metric-card.teal  { border-left-color: #1abc9c; }
  .metric-card.gold  { border-left-color: #f1c40f; }
  .metric-card.gray  { border-left-color: #7f8c8d; }
  .metric-value { font-size: 2.2rem; font-weight: 700; color: #ff6b35; margin: 0; }
  .metric-value.teal { color: #1abc9c; }
  .metric-value.gold { color: #f1c40f; }
  .metric-value.gray { color: #7f8c8d; }
  .metric-label { font-size: 0.85rem; color: #8fa8c8; margin: 0; }
  .insight-box { background: #162032; border-radius: 10px; padding: 20px 24px; margin-bottom: 12px; }
  .insight-box h4 { color: #ff6b35; margin-bottom: 8px; }
  .insight-box p { color: #c5d5e8; margin: 0; font-size: 0.92rem; line-height: 1.6; }
  .section-header { color: #8fa8c8; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px; margin-top: 4px; }
  .anomaly-box { background: #1a1a2e; border: 1px solid #ff6b35; border-radius: 10px; padding: 20px 24px; }
  .anomaly-box h4 { color: #ff6b35; margin-bottom: 8px; }
  .anomaly-box p  { color: #c5d5e8; font-size: 0.92rem; line-height: 1.6; margin: 0; }
  .exec-panel { background: #162032; border-radius: 12px; padding: 24px; height: 100%; }
  .exec-panel h3 { color: #ffffff; margin-bottom: 6px; font-size: 1.05rem; }
  .exec-headline { color: #1abc9c; font-size: 1.35rem; font-weight: 700; margin-bottom: 10px; }
  .exec-number  { font-size: 1.8rem; font-weight: 700; color: #ff6b35; margin-bottom: 4px; }
  .exec-text    { color: #c5d5e8; font-size: 0.9rem; line-height: 1.6; }
  .limit-item   { background: #1e3a5f22; border-left: 3px solid #7f8c8d; padding: 10px 14px; border-radius: 0 6px 6px 0; margin-bottom: 8px; color: #c5d5e8; font-size: 0.88rem; }
  .next-step    { background: #1abc9c15; border-left: 3px solid #1abc9c; padding: 10px 14px; border-radius: 0 6px 6px 0; margin-bottom: 8px; color: #c5d5e8; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Preprocessing ────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and preparing data…")
def load_data():
    df = pd.read_csv("/app/doordash.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
    df["actual_duration_min"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds() / 60

    df["dasher_utilization"] = df["total_busy_dashers"] / df[
        "total_onshift_dashers"
    ].replace(0, np.nan)

    df["hour_of_day"] = df["created_at"].dt.hour

    df["store_primary_category"] = df["store_primary_category"].fillna("unknown")

    outlier_count = (df["actual_duration_min"] > 120).sum()
    df_clean = df[df["actual_duration_min"] <= 120].copy()

    bracket_bins   = [0, 0.5, 0.75, 0.9, 1.0, 10]
    bracket_labels = ["<50%", "50–75%", "75–90%", "90–100%", ">100%"]
    df_clean["utilization_bracket"] = pd.cut(
        df_clean["dasher_utilization"],
        bins=bracket_bins,
        labels=bracket_labels,
    )

    df_rqa = df_clean.dropna(subset=["dasher_utilization"]).copy()

    return df, df_clean, df_rqa, outlier_count

df_raw, df_clean, df_rqa, outlier_count = load_data()
missing_dasher = df_raw["total_onshift_dashers"].isna().sum()
missing_category = (df_raw["store_primary_category"].isna()).sum()

# ── Colour palette ───────────────────────────────────────────────────────────
BRACKET_COLORS = {
    "<50%":    "#1abc9c",
    "50–75%":  "#f1c40f",
    "75–90%":  "#e67e22",
    "90–100%": "#e74c3c",
    ">100%":   "#c0392b",
}
PLOTLY_DARK = dict(
    paper_bgcolor="#0d1b2a",
    plot_bgcolor="#162032",
    font_color="#c5d5e8",
    title_font_color="#ffffff",
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:#162032;border-radius:12px;padding:20px 28px;margin-bottom:20px;'>
  <div style='font-size:0.72rem;color:#ff6b35;letter-spacing:.12em;font-weight:700;text-transform:uppercase;'>INDIVIDUAL PROJECT · MGMT 38900</div>
  <h1 style='margin:6px 0 2px;font-size:1.9rem;'>What's Really Causing DoorDash Delivery Delays?</h1>
  <div style='color:#8fa8c8;font-size:0.9rem;'>Presented by: Rohan Menon &nbsp;|&nbsp; Purdue University, Spring 2026</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🔴 RQ-A: Dasher Supply Pressure",
    "🕐 RQ-B: Time-of-Day Compounding",
    "📋 Executive Summary",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Dataset Summary</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <p class='metric-value'>197,428</p>
          <p class='metric-label'>Total Orders</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card teal'>
          <p class='metric-value teal'>{len(df_rqa):,}</p>
          <p class='metric-label'>Orders with Valid Utilization Data</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card gold'>
          <p class='metric-value gold'>{outlier_count:,}</p>
          <p class='metric-label'>Outliers Removed (>120 min)</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card gray'>
          <p class='metric-value gray'>6</p>
          <p class='metric-label'>Markets Covered</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-header'>Descriptive Statistics (Post-Cleaning)</div>", unsafe_allow_html=True)
        stats = df_clean["actual_duration_min"].describe().rename({
            "count": "Count", "mean": "Mean (min)", "std": "Std Dev",
            "min": "Min", "25%": "25th pct", "50%": "Median", "75%": "75th pct", "max": "Max"
        })
        stats_df = pd.DataFrame({"Delivery Duration (min)": stats.round(1)})
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("<div class='section-header' style='margin-top:18px;'>Data Quality Report</div>", unsafe_allow_html=True)
        dq = pd.DataFrame({
            "Issue": ["Dasher fields missing", "Category missing", "Extreme outliers (>120 min)"],
            "Count": [missing_dasher, missing_category, outlier_count],
            "% of Total": [
                f"{missing_dasher/len(df_raw)*100:.1f}%",
                f"{missing_category/len(df_raw)*100:.1f}%",
                f"{outlier_count/len(df_raw)*100:.2f}%",
            ],
            "Action": ["Excluded from RQ-A only", "Labeled 'unknown'", "Removed before analysis"],
        })
        st.dataframe(dq, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("<div class='section-header'>Utilization Distribution</div>", unsafe_allow_html=True)
        util_data = df_rqa["dasher_utilization"] * 100
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=util_data, nbinsx=60,
            marker_color="#1abc9c", opacity=0.8,
            name="Orders",
        ))
        fig_hist.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=2,
                           annotation_text="90% threshold", annotation_font_color="#e74c3c")
        fig_hist.update_layout(
            **PLOTLY_DARK,
            xaxis_title="Dasher Utilization (%)", yaxis_title="Order Count",
            showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div class='section-header'>Interactive Scatter: Utilization vs. Delivery Time</div>", unsafe_allow_html=True)
    # Sample for performance
    sample = df_rqa.sample(min(8000, len(df_rqa)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="dasher_utilization", y="actual_duration_min",
        color="hour_of_day", color_continuous_scale="Turbo",
        opacity=0.35, hover_data=["market_id", "store_primary_category", "total_items"],
        labels={"dasher_utilization": "Dasher Utilization", "actual_duration_min": "Delivery Time (min)", "hour_of_day": "Hour of Day"},
    )
    # Add trend line via 5-pct bins
    bins = np.arange(0, 1.51, 0.05)
    bin_centers, bin_means = [], []
    for i in range(len(bins) - 1):
        mask = (df_rqa["dasher_utilization"] >= bins[i]) & (df_rqa["dasher_utilization"] < bins[i+1])
        if mask.sum() > 30:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(df_rqa.loc[mask, "actual_duration_min"].mean())
    fig_scatter.add_trace(go.Scatter(
        x=bin_centers, y=bin_means,
        mode="lines", line=dict(color="#ff6b35", width=3), name="Trend (5-pct bins)",
    ))
    fig_scatter.update_layout(
        **PLOTLY_DARK, height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Hour"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – RQ-A
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Research Question A: Dasher Supply Pressure</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box' style='margin-bottom:16px;'>
      <p><em>To what extent does dasher network saturation explain delivery time variation, and at what utilization threshold does the effect become practically significant?</em></p>
    </div>""", unsafe_allow_html=True)

    bracket_order = ["<50%", "50–75%", "75–90%", "90–100%", ">100%"]
    bracket_stats = (
        df_rqa.groupby("utilization_bracket", observed=True)["actual_duration_min"]
        .agg(Mean="mean", Std="std", Count="count")
        .reset_index()
    )
    bracket_stats["SE"] = bracket_stats["Std"] / np.sqrt(bracket_stats["Count"])
    bracket_stats["utilization_bracket"] = pd.Categorical(
        bracket_stats["utilization_bracket"], categories=bracket_order, ordered=True
    )
    bracket_stats = bracket_stats.sort_values("utilization_bracket")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("<div class='section-header'>Key Numbers</div>", unsafe_allow_html=True)
        low_mean = bracket_stats.loc[bracket_stats["utilization_bracket"] == "<50%", "Mean"].values[0]
        high_mean = bracket_stats.loc[bracket_stats["utilization_bracket"] == "90–100%", "Mean"].values[0]
        pct_inc = (high_mean - low_mean) / low_mean * 100
        for val, label, cls in [
            (f"{low_mean:.1f} min", "Avg at <50% Utilization", "teal"),
            (f"{high_mean:.1f} min", "Avg at 90–100% Utilization", ""),
            (f"+{pct_inc:.1f}%", "Delivery Time Increase", "gold"),
            (f"{len(df_rqa):,}", "Orders with Valid Data", "gray"),
        ]:
            st.markdown(f"""
            <div class='metric-card {cls}'>
              <p class='metric-value {cls}'>{val}</p>
              <p class='metric-label'>{label}</p>
            </div>""", unsafe_allow_html=True)

    with col1:
        st.markdown("<div class='section-header'>Average Delivery Time by Utilization Bracket</div>", unsafe_allow_html=True)
        fig_bar = go.Figure()
        for _, row in bracket_stats.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row["utilization_bracket"]],
                y=[row["Mean"]],
                error_y=dict(type="data", array=[row["SE"] * 1.96], visible=True, color="#ffffff55"),
                marker_color=BRACKET_COLORS.get(str(row["utilization_bracket"]), "#888"),
                text=f"{row['Mean']:.1f}m<br>n={row['Count']:,}",
                textposition="outside", textfont=dict(color="#ffffff", size=11),
                name=str(row["utilization_bracket"]),
                showlegend=False,
            ))
        baseline = bracket_stats["Mean"].min()
        fig_bar.add_hline(y=baseline, line_dash="dash", line_color="#1abc9c55", line_width=1,
                          annotation_text="Baseline", annotation_font_color="#1abc9c")
        fig_bar.update_layout(
            **PLOTLY_DARK, height=400,
            xaxis_title="Dasher Utilization Bracket",
            yaxis_title="Avg Delivery Time (min)",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_range=[0, bracket_stats["Mean"].max() * 1.2],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div class='section-header'>Threshold Analysis: Delivery Time vs. Utilization (5-Percentile Bins)</div>", unsafe_allow_html=True)
    bins = np.arange(0, 1.01, 0.05)
    threshold_data = []
    for i in range(len(bins) - 1):
        mask = (df_rqa["dasher_utilization"] >= bins[i]) & (df_rqa["dasher_utilization"] < bins[i+1])
        cnt = mask.sum()
        if cnt >= 50:
            threshold_data.append({
                "util_pct": (bins[i] + bins[i+1]) / 2 * 100,
                "mean_time": df_rqa.loc[mask, "actual_duration_min"].mean(),
                "count": cnt,
            })
    td = pd.DataFrame(threshold_data)

    fig_thresh = go.Figure()
    fig_thresh.add_vrect(x0=90, x1=105, fillcolor="#e74c3c", opacity=0.08, line_width=0)
    fig_thresh.add_trace(go.Scatter(
        x=td["util_pct"], y=td["mean_time"],
        mode="lines+markers",
        line=dict(color="#ff6b35", width=2.5),
        marker=dict(color="#ff6b35", size=7),
        hovertemplate="Utilization: %{x:.0f}%<br>Avg Time: %{y:.1f} min<extra></extra>",
    ))
    fig_thresh.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=2,
                         annotation_text="90% inflection", annotation_font_color="#e74c3c",
                         annotation_position="top right")
    fig_thresh.update_layout(
        **PLOTLY_DARK, height=360,
        xaxis_title="Dasher Utilization (%)", yaxis_title="Avg Delivery Time (min)",
        margin=dict(l=0, r=0, t=10, b=0),
        annotations=[
            dict(x=45, y=td["mean_time"].min() + 1.5, text="Gradual increase 0–90%",
                 showarrow=False, font=dict(color="#1abc9c", size=12)),
            dict(x=97, y=td["mean_time"].max() - 1, text="Sharp<br>acceleration",
                 showarrow=False, font=dict(color="#e74c3c", size=12)),
        ]
    )
    st.plotly_chart(fig_thresh, use_container_width=True)

    st.markdown("<div class='section-header'>Statistical Summary by Bracket</div>", unsafe_allow_html=True)
    summary_display = bracket_stats.copy()
    summary_display["% of Total"] = (summary_display["Count"] / len(df_rqa) * 100).round(1).astype(str) + "%"
    summary_display = summary_display.rename(columns={
        "utilization_bracket": "Bracket", "Mean": "Mean (min)", "Std": "Std Dev", "Count": "N", "SE": "Std Error"
    })
    summary_display["Mean (min)"] = summary_display["Mean (min)"].round(1)
    summary_display["Std Dev"] = summary_display["Std Dev"].round(1)
    summary_display["Std Error"] = summary_display["Std Error"].round(2)
    st.dataframe(summary_display[["Bracket", "Mean (min)", "Std Dev", "Std Error", "N", "% of Total"]],
                 use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight-box' style='margin-top:16px;'>
      <h4>Key Insight: RQ-A</h4>
      <p>Delivery time increases monotonically across all utilization brackets — from <strong>39.7 min</strong> at under 50% utilization to <strong>49.9 min</strong> at 90–100%, a <strong>+25.5% increase</strong>. The effect is gradual between 0–90%, then accelerates sharply above the 90% threshold, consistent with non-linear queuing dynamics. Operationally, keeping utilization below 85–90% yields disproportionate benefit relative to any marginal staffing improvement at lower utilization levels.</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – RQ-B
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Research Question B: Time-of-Day Compounding</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box' style='margin-bottom:16px;'>
      <p><em>Do specific hours compound the utilization effect — meaning, are the longest deliveries concentrated in windows where both demand is high AND dasher availability is low simultaneously?</em></p>
    </div>""", unsafe_allow_html=True)

    # Hour × Util heatmap data
    heat_df = (
        df_rqa.groupby(["hour_of_day", "utilization_bracket"], observed=True)["actual_duration_min"]
        .agg(mean="mean", count="count")
        .reset_index()
    )
    heat_df = heat_df[heat_df["count"] >= 30]
    heat_pivot = heat_df.pivot(index="utilization_bracket", columns="hour_of_day", values="mean")
    heat_pivot = heat_pivot.reindex(["<50%", "50–75%", "75–90%", "90–100%", ">100%"])

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("<div class='section-header'>Notable Cells</div>", unsafe_allow_html=True)
        for val, label, cls in [
            ("56 min", "2 AM + 90–100% util — worst cell", ""),
            ("54 min", "3 AM + 90–100% util — close 2nd", ""),
            ("59 min", "2 PM (all util) — the anomaly", "gold"),
            ("36 min", "5–6 AM + low util — best combo", "teal"),
        ]:
            st.markdown(f"""
            <div class='metric-card {cls}'>
              <p class='metric-value {cls}'>{val}</p>
              <p class='metric-label'>{label}</p>
            </div>""", unsafe_allow_html=True)

    with col1:
        st.markdown("<div class='section-header'>Heatmap: Delivery Time by Hour × Utilization Bracket</div>", unsafe_allow_html=True)
        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=[f"{h}:00" for h in heat_pivot.columns],
            y=list(heat_pivot.index),
            colorscale=[[0, "#1abc9c"], [0.5, "#f1c40f"], [1, "#e74c3c"]],
            text=np.where(np.isnan(heat_pivot.values), "–",
                          np.vectorize(lambda v: f"{v:.0f}m" if not np.isnan(v) else "–")(heat_pivot.values)),
            texttemplate="%{text}", textfont=dict(size=10, color="#ffffff"),
            hovertemplate="Hour: %{x}<br>Bracket: %{y}<br>Avg Time: %{z:.1f} min<extra></extra>",
            colorbar=dict(title="Avg min", tickfont=dict(color="#c5d5e8"), titlefont=dict(color="#c5d5e8")),
        ))
        fig_heat.update_layout(
            **PLOTLY_DARK, height=320,
            xaxis=dict(title="Hour of Day", tickfont=dict(size=10), gridcolor="#1e3a5f"),
            yaxis=dict(title="Utilization Bracket", tickfont=dict(size=11), gridcolor="#1e3a5f"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Penalty by hour
    st.markdown("<div class='section-header'>Utilization Penalty by Hour (High ≥90% minus Low <50%)</div>", unsafe_allow_html=True)
    hour_high = df_rqa[df_rqa["dasher_utilization"] >= 0.9].groupby("hour_of_day")["actual_duration_min"].mean()
    hour_low  = df_rqa[df_rqa["dasher_utilization"] < 0.5].groupby("hour_of_day")["actual_duration_min"].mean()
    penalty   = (hour_high - hour_low).dropna()
    avg_penalty = penalty.mean()

    fig_penalty = go.Figure()
    bar_colors = ["#e74c3c" if p >= avg_penalty else "#3498db" for p in penalty.values]
    fig_penalty.add_trace(go.Bar(
        x=[f"{h}:00" for h in penalty.index],
        y=penalty.values,
        marker_color=bar_colors,
        text=[f"+{p:.1f}m" for p in penalty.values],
        textposition="outside", textfont=dict(color="#ffffff", size=10),
        hovertemplate="Hour: %{x}<br>Penalty: +%{y:.1f} min<extra></extra>",
    ))
    fig_penalty.add_hline(y=avg_penalty, line_dash="dash", line_color="#f1c40f", line_width=2,
                          annotation_text=f"Avg penalty: +{avg_penalty:.1f} min",
                          annotation_font_color="#f1c40f")
    fig_penalty.update_layout(
        **PLOTLY_DARK, height=320,
        xaxis_title="Hour of Day", yaxis_title="Extra Minutes (High vs. Low Util)",
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    )
    st.plotly_chart(fig_penalty, use_container_width=True)

    # Line chart: delivery time by hour, stratified
    st.markdown("<div class='section-header'>Delivery Time by Hour — Stratified by Utilization Level</div>", unsafe_allow_html=True)
    hour_all  = df_clean.groupby("hour_of_day")["actual_duration_min"].mean()
    hour_mid  = df_rqa[(df_rqa["dasher_utilization"] >= 0.5) & (df_rqa["dasher_utilization"] < 0.9)].groupby("hour_of_day")["actual_duration_min"].mean()

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=hour_high.index, y=hour_high.values,
        mode="lines+markers", name="High util (≥90%)",
        line=dict(color="#e74c3c", width=2.5, dash="solid"),
        marker=dict(size=6),
    ))
    fig_line.add_trace(go.Scatter(
        x=hour_mid.index, y=hour_mid.values,
        mode="lines+markers", name="Mid util (50–90%)",
        line=dict(color="#f1c40f", width=2, dash="dot"),
        marker=dict(size=5),
    ))
    fig_line.add_trace(go.Scatter(
        x=hour_low.index, y=hour_low.values,
        mode="lines+markers", name="Low util (<50%)",
        line=dict(color="#1abc9c", width=2.5),
        marker=dict(size=6),
    ))
    # Shaded gap
    fig_line.add_trace(go.Scatter(
        x=list(hour_high.index) + list(hour_low.index[::-1]),
        y=list(hour_high.values) + list(hour_low.values[::-1]),
        fill="toself", fillcolor="rgba(231,76,60,0.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    # 2PM annotation
    two_pm_y = hour_all.get(14, 55)
    fig_line.add_annotation(x=14, y=two_pm_y + 2,
        text="2 PM Anomaly<br>(59.6 min, util: 52%)",
        showarrow=True, arrowhead=2, arrowcolor="#ff6b35",
        font=dict(color="#ff6b35", size=11), bgcolor="#162032", bordercolor="#ff6b35",
    )
    fig_line.update_layout(
        **PLOTLY_DARK, height=360,
        xaxis=dict(title="Hour of Day", tickvals=list(range(0, 24)),
                   ticktext=[f"{h}:00" for h in range(24)], tickangle=45),
        yaxis_title="Avg Delivery Time (min)",
        legend=dict(bgcolor="#162032", bordercolor="#1e3a5f", font=dict(color="#c5d5e8")),
        margin=dict(l=0, r=0, t=10, b=60),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # 2PM Anomaly investigation
    st.markdown("<div class='section-header'>2 PM Anomaly Investigation</div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    two_pm = df_clean[df_clean["hour_of_day"] == 14]
    other  = df_clean[df_clean["hour_of_day"] != 14]

    with col_a:
        st.markdown(f"""
        <div class='anomaly-box'>
          <h4>⚠️ The Finding</h4>
          <p>2 PM averages <strong>{two_pm['actual_duration_min'].mean():.1f} min</strong> — the highest of any hour — yet dasher utilization is only <strong>~52%</strong>. This rules out supply shortage as the primary cause.</p>
        </div>""", unsafe_allow_html=True)

    with col_b:
        # Cuisine mix at 2 PM vs. rest
        cat_2pm   = two_pm["store_primary_category"].value_counts(normalize=True).head(5) * 100
        cat_other = other["store_primary_category"].value_counts(normalize=True).head(5) * 100
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(name="2 PM", x=cat_2pm.index, y=cat_2pm.values, marker_color="#ff6b35"))
        fig_cat.add_trace(go.Bar(name="Other hours", x=cat_other.index, y=cat_other.values, marker_color="#1abc9c"))
        fig_cat.update_layout(
            **PLOTLY_DARK, barmode="group", height=260,
            title=dict(text="Cuisine Mix: 2 PM vs. Other Hours", font=dict(color="#ffffff", size=13)),
            xaxis_title="Category", yaxis_title="% of Orders",
            legend=dict(font=dict(color="#c5d5e8"), bgcolor="#162032"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_c:
        # Order size comparison
        avg_items_2pm  = two_pm["total_items"].mean()
        avg_items_rest = other["total_items"].mean()
        avg_sub_2pm    = two_pm["subtotal"].mean() / 100
        avg_sub_rest   = other["subtotal"].mean() / 100
        st.markdown(f"""
        <div class='anomaly-box'>
          <h4>📦 Order Characteristics</h4>
          <p>
            <strong>Avg items:</strong> {avg_items_2pm:.1f} (2 PM) vs {avg_items_rest:.1f} (other hours)<br><br>
            <strong>Avg subtotal:</strong> ${avg_sub_2pm:.2f} (2 PM) vs ${avg_sub_rest:.2f} (other hours)<br><br>
            <strong>Hypothesis:</strong> Larger, more complex lunch orders requiring longer restaurant prep time — not a dasher supply issue. Consistent with the utilization inversion observed in the penalty chart.
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box' style='margin-top:16px;'>
      <h4>Key Insight: RQ-B</h4>
      <p>The utilization penalty on delivery time is not uniform across hours. Overnight windows (1–4 AM) show the largest compounding effect — up to <strong>+10.9 min</strong> at 3 AM — because high utilization coincides with low order density, meaning the few available dashers are fully saturated. The <strong>2 PM spike</strong> (59.6 min average despite only 52% utilization) is a clear anomaly: utilization is not the driver, pointing instead to restaurant-side constraints such as longer prep times for complex lunch orders.</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Executive Summary</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='exec-panel'>
          <div style='color:#8fa8c8;font-size:0.7rem;letter-spacing:.1em;font-weight:700;text-transform:uppercase;margin-bottom:6px;'>RQ-A Finding</div>
          <div class='exec-headline'>Dasher Utilization Explains a 25% Increase in Delivery Time</div>
          <div class='exec-number'>39.7 → 49.9 min</div>
          <div style='color:#8fa8c8;font-size:0.82rem;margin-bottom:12px;'>Low util → High util</div>
          <div class='exec-text'>
            Each 10% rise in network saturation adds roughly 2–4 minutes to delivery time under normal conditions, but the effect accelerates sharply above 90% utilization — consistent with non-linear queuing dynamics.<br><br>
            <strong style='color:#1abc9c;'>Action:</strong> Maintain dasher utilization below 85–90% to avoid non-linear delay acceleration. Marginal staffing investment at this threshold yields outsized customer experience improvements.
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='exec-panel'>
          <div style='color:#8fa8c8;font-size:0.7rem;letter-spacing:.1em;font-weight:700;text-transform:uppercase;margin-bottom:6px;'>RQ-B Finding</div>
          <div class='exec-headline'>Overnight Hours Show the Largest Utilization Penalty</div>
          <div class='exec-number'>+10.9 min</div>
          <div style='color:#8fa8c8;font-size:0.82rem;margin-bottom:12px;'>Peak penalty at 3 AM</div>
          <div class='exec-text'>
            Time-of-day compounds the utilization effect. Overnight windows (1–4 AM) show the steepest penalty because high utilization coincides with low order density — meaning every available dasher is already running an order, and new orders wait in queue.<br><br>
            <strong style='color:#1abc9c;'>Action:</strong> Prioritize surge pricing and targeted dasher recruitment for 1–4 AM windows. Dynamic incentive strategies in overnight hours will yield the highest per-dollar improvement in delivery time.
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='exec-panel'>
          <div style='color:#8fa8c8;font-size:0.7rem;letter-spacing:.1em;font-weight:700;text-transform:uppercase;margin-bottom:6px;'>Unexpected Discovery</div>
          <div class='exec-headline'>2 PM Anomaly Points to a Non-Utilization Bottleneck</div>
          <div class='exec-number'>59.6 min</div>
          <div style='color:#8fa8c8;font-size:0.82rem;margin-bottom:12px;'>Avg at 2 PM, despite only 52% utilization</div>
          <div class='exec-text'>
            The worst delivery hour in the dataset is not explained by dasher supply. The utilization effect inverts at 2 PM, and even low-utilization orders at that hour average 55+ minutes. Evidence points to restaurant-side constraints — longer prep times for complex lunch orders — as the primary driver.<br><br>
            <strong style='color:#1abc9c;'>Action:</strong> Investigate restaurant prep times and order complexity at 2 PM. Operational fix likely lies in restaurant partnerships, not dasher recruitment.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Business Value Delivered</div>", unsafe_allow_html=True)
    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown("""
        <div class='insight-box'>
          <h4>Tangible Benefits for Operations Teams</h4>
          <p>
            • <strong>3–5 min average delay reduction</strong> achievable via utilization target-setting below 90%<br>
            • <strong>NPS improvement</strong> from reduced late deliveries during peak penalty windows<br>
            • <strong>Optimized surge pricing allocation</strong>: concentrate incentives 1–4 AM, not broadly<br>
            • <strong>Parallel operational fix</strong> identified for 2 PM bottleneck (restaurant-side, not supply)
          </p>
        </div>""", unsafe_allow_html=True)
    with cv2:
        st.markdown("""
        <div class='insight-box'>
          <h4>Decision Makers & Stakeholders</h4>
          <p>
            • <strong>VP Operations</strong>: Data-driven staffing targets and shift scheduling thresholds<br>
            • <strong>Surge Pricing Team</strong>: Overnight premium justified; concentrated not broadcast<br>
            • <strong>Driver Recruitment</strong>: 1–4 AM market-specific deployment strategies<br>
            • <strong>Restaurant Partnerships</strong>: 2 PM prep-time SLA investigation and improvement
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Limitations & Confidence</div>", unsafe_allow_html=True)
    for text in [
        "8.2% of orders are missing dasher data; utilization findings apply to ~175,000 records (92% of dataset). Impact on conclusions is low given large sample size.",
        "Only two ETA components are available (order placement + store-to-consumer driving). True expected-vs.-actual gap cannot be computed without the restaurant preparation component.",
        "Dataset covers 2015. Current DoorDash network density, dasher availability, and operational processes may have evolved significantly. Findings should be validated on current data.",
        "All results are descriptive and capture correlation, not causation. A/B testing would be required to confirm that reducing utilization directly causes delivery time reduction.",
    ]:
        st.markdown(f"<div class='limit-item'>{text}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:16px;'>Recommended Next Steps</div>", unsafe_allow_html=True)
    for text in [
        "ANOVA / Kruskal-Wallis test on utilization brackets to formally establish statistical significance across all five groups.",
        "Logistic regression model: probability of delivery time >45 min given utilization and hour of day — transition from descriptive to predictive.",
        "Market-level breakdown: do utilization effects and penalties vary by geography? Six markets in dataset enable this comparison.",
        "Restaurant-level investigation: is 2 PM anomaly driven by specific cuisine types, individual stores, or a broad lunch-rush operational pattern?",
    ]:
        st.markdown(f"<div class='next-step'>{text}</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#162032;border-radius:10px;padding:20px 24px;margin-top:20px;border-left:4px solid #ff6b35;'>
      <div style='color:#ff6b35;font-size:0.72rem;letter-spacing:.1em;font-weight:700;text-transform:uppercase;margin-bottom:8px;'>Main Learnings</div>
      <p style='color:#c5d5e8;font-size:0.92rem;line-height:1.7;margin:0;'>
        <strong style='color:#ffffff;'>Technical:</strong> Real-world data rarely arrives clean. The first pass of analysis without outlier removal produced means above 100 minutes — entirely misleading. Data cleaning and feature engineering consumed the majority of analytical effort, reinforcing that a sound foundation precedes any insight.<br><br>
        <strong style='color:#ffffff;'>Analytical:</strong> Non-linear thresholds (the 90% inflection point) are more operationally actionable than linear models. A linear regression would have missed the queuing regime change. Descriptive analytics, done rigorously, is both credible and sufficient for diagnostic questions about "how much" and "in which conditions."<br><br>
        <strong style='color:#ffffff;'>Professional:</strong> Communicating uncertainty honestly — missing data scope, partial ETA, 2015 vintage — builds more executive trust than overstating precision. The 2 PM anomaly, discovered during exploratory analysis, became the most compelling finding precisely because it was unexpected and points to an actionable operational constraint.
      </p>
    </div>""", unsafe_allow_html=True)
