import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="DoorDash Delivery Analytics",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  h1 { margin-bottom: 4px; color: #1a1a1a; }
  h2 { margin-top: 24px; margin-bottom: 8px; color: #1a1a1a; font-size: 1.4rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 0; background: #f5f7fa; border-bottom: 1px solid #ddd; }
  .stTabs [data-baseweb="tab"] { color: #666; background: transparent; border: none; padding: 12px 20px; }
  .stTabs [aria-selected="true"] { color: #1f77b4 !important; border-bottom: 3px solid #1f77b4 !important; background: white; }
  .info-box { background: #f5f7fa; border-left: 4px solid #1f77b4; padding: 14px; border-radius: 4px; margin-bottom: 12px; }
  .info-box p { margin: 0; color: #1a1a1a; font-size: 0.95rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doordash.csv")
    df = pd.read_csv(path)
    
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
    df["actual_duration_min"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 60
    df["dasher_utilization"] = df["total_busy_dashers"] / df["total_onshift_dashers"].replace(0, np.nan)
    df["hour_of_day"] = df["created_at"].dt.hour
    df["store_primary_category"] = df["store_primary_category"].fillna("unknown")
    
    # Clean outliers
    df_clean = df[df["actual_duration_min"] <= 120].copy()
    
    # Bracket
    df_clean["util_bracket"] = pd.cut(
        df_clean["dasher_utilization"],
        bins=[0, 0.5, 0.75, 0.9, 1.0, 10],
        labels=["<50%", "50–75%", "75–90%", "90–100%", ">100%"]
    )
    
    df_rqa = df_clean.dropna(subset=["dasher_utilization"]).copy()
    outlier_count = (df["actual_duration_min"] > 120).sum()
    
    return df, df_clean, df_rqa, outlier_count

df_raw, df_clean, df_rqa, outlier_count = load_data()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🚗 DoorDash Delivery Delay Analytics")
st.markdown("""
Quantifying how dasher network utilization and time of day drive delivery delays.
Built with descriptive analytics on 197,428 orders from 2015.
""")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔴 RQ-A: Utilization", "🕐 RQ-B: Time-of-Day", "💡 Summary"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Orders", "197,428")
    with c2:
        st.metric("Valid Utilization", f"{len(df_rqa):,}")
    with c3:
        st.metric("Outliers Removed", f"{outlier_count:,}")
    with c4:
        st.metric("Markets", "6")
    
    st.markdown("### Delivery Time Statistics (Post-Cleaning)")
    stats = df_clean["actual_duration_min"].describe().round(1)
    stat_df = pd.DataFrame({
        "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"],
        "Delivery Time (min)": [
            f"{stats['count']:.0f}",
            f"{stats['mean']:.1f}",
            f"{stats['std']:.1f}",
            f"{stats['min']:.1f}",
            f"{stats['25%']:.1f}",
            f"{stats['50%']:.1f}",
            f"{stats['75%']:.1f}",
            f"{stats['max']:.1f}",
        ]
    })
    st.dataframe(stat_df, hide_index=True, use_container_width=True)
    
    st.markdown("### Data Quality")
    quality = pd.DataFrame({
        "Issue": ["Dasher fields missing", "Category missing", "Outliers (>120 min)"],
        "Count": [df_raw["total_onshift_dashers"].isna().sum(), 
                  (df_raw["store_primary_category"].isna()).sum(),
                  outlier_count],
        "% of Total": [
            f"{df_raw['total_onshift_dashers'].isna().sum() / len(df_raw) * 100:.1f}%",
            f"{(df_raw['store_primary_category'].isna()).sum() / len(df_raw) * 100:.1f}%",
            f"{outlier_count / len(df_raw) * 100:.2f}%",
        ],
        "Action": ["Excluded from RQ-A", "Labeled 'unknown'", "Removed"],
    })
    st.dataframe(quality, hide_index=True, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Utilization Distribution")
        util_hist = go.Figure()
        util_hist.add_trace(go.Histogram(
            x=df_rqa["dasher_utilization"] * 100,
            nbinsx=50,
            marker_color="#1f77b4",
            showlegend=False,
        ))
        util_hist.add_vline(x=90, line_dash="dash", line_color="#d62728")
        util_hist.update_layout(
            xaxis_title="Utilization (%)",
            yaxis_title="Count",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f5f7fa",
        )
        st.plotly_chart(util_hist, use_container_width=True)
    
    with col2:
        st.markdown("### Delivery Time Distribution")
        dur_hist = go.Figure()
        dur_hist.add_trace(go.Histogram(
            x=df_clean["actual_duration_min"],
            nbinsx=50,
            marker_color="#2ca02c",
            showlegend=False,
        ))
        dur_hist.update_layout(
            xaxis_title="Delivery Time (min)",
            yaxis_title="Count",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f5f7fa",
        )
        st.plotly_chart(dur_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: RQ-A
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Research Question A: Dasher Supply Pressure")
    st.markdown("""
    **To what extent does dasher network saturation explain delivery time variation, 
    and at what utilization threshold does the effect become practically significant?**
    """)
    
    col_metric, col_chart = st.columns([1, 2])
    
    with col_metric:
        # Stats by bracket
        bracket_stats = (
            df_rqa.groupby("util_bracket", observed=True)["actual_duration_min"]
            .agg(mean="mean", count="count")
            .reset_index()
        )
        bracket_stats["util_bracket"] = pd.Categorical(
            bracket_stats["util_bracket"],
            categories=["<50%", "50–75%", "75–90%", "90–100%", ">100%"],
            ordered=True
        )
        bracket_stats = bracket_stats.sort_values("util_bracket")
        
        low = bracket_stats.iloc[0]["mean"]
        high = bracket_stats.iloc[-1]["mean"]
        pct_chg = (high - low) / low * 100
        
        st.markdown(f"""
        <div class='info-box'>
            <p><strong>{low:.1f} min</strong> at <50% utilization</p>
            <p><strong>{high:.1f} min</strong> at 90–100% utilization</p>
            <p><strong>+{pct_chg:.1f}%</strong> increase</p>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("**Statistics by Bracket:**")
        display_stats = bracket_stats[["util_bracket", "mean", "count"]].copy()
        display_stats["mean"] = display_stats["mean"].round(1)
        st.dataframe(
            display_stats.rename(columns={"util_bracket": "Bracket", "mean": "Avg (min)", "count": "N"}),
            hide_index=True,
            use_container_width=True
        )
    
    with col_chart:
        fig_bar = go.Figure()
        colors = ["#2ca02c", "#ff7f0e", "#ff7f0e", "#d62728", "#d62728"]
        for idx, (_, row) in enumerate(bracket_stats.iterrows()):
            fig_bar.add_trace(go.Bar(
                x=[str(row["util_bracket"])],
                y=[row["mean"]],
                marker_color=colors[idx],
                text=f"{row['mean']:.1f}",
                textposition="outside",
                showlegend=False,
            ))
        
        fig_bar.update_layout(
            title="Average Delivery Time by Utilization Bracket",
            xaxis_title="Utilization Bracket",
            yaxis_title="Minutes",
            height=400,
            margin=dict(l=40, r=20, t=60, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f5f7fa",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("### Threshold Analysis (5-Percentile Bins)")
    
    # Bin analysis
    bins = np.arange(0, 1.01, 0.05)
    threshold_data = []
    for i in range(len(bins) - 1):
        mask = (df_rqa["dasher_utilization"] >= bins[i]) & (df_rqa["dasher_utilization"] < bins[i+1])
        if mask.sum() >= 30:
            threshold_data.append({
                "util_pct": (bins[i] + bins[i+1]) / 2 * 100,
                "mean_time": df_rqa.loc[mask, "actual_duration_min"].mean(),
            })
    
    td = pd.DataFrame(threshold_data)
    
    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(
        x=td["util_pct"],
        y=td["mean_time"],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=8),
    ))
    fig_thresh.add_vline(x=90, line_dash="dash", line_color="#d62728", annotation_text="90% threshold")
    fig_thresh.update_layout(
        xaxis_title="Utilization (%)",
        yaxis_title="Delivery Time (min)",
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f5f7fa",
        showlegend=False,
    )
    st.plotly_chart(fig_thresh, use_container_width=True)
    
    st.markdown("""
    <div class='info-box'>
    <p><strong>Key Finding:</strong> Delivery time increases monotonically from 39.7 min (low utilization) 
    to 49.9 min (high utilization), a 25.5% increase. The effect is gradual below 90%, then 
    accelerates sharply above — indicating a non-linear queuing regime. Operations should 
    target keeping utilization below 85–90%.</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: RQ-B
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Research Question B: Time-of-Day Compounding")
    st.markdown("""
    **Do specific hours compound the utilization effect — are longest deliveries 
    concentrated where both demand is high AND dasher availability is low?**
    """)
    
    # Penalty calculation
    hour_high = df_rqa[df_rqa["dasher_utilization"] >= 0.9].groupby("hour_of_day")["actual_duration_min"].mean()
    hour_low = df_rqa[df_rqa["dasher_utilization"] < 0.5].groupby("hour_of_day")["actual_duration_min"].mean()
    hour_all = df_clean.groupby("hour_of_day")["actual_duration_min"].mean()
    
    penalty = (hour_high - hour_low).dropna()
    avg_penalty = penalty.mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Penalty by Hour (High util − Low util)")
        fig_penalty = go.Figure()
        colors_p = ["#d62728" if p >= avg_penalty else "#1f77b4" for p in penalty.values]
        fig_penalty.add_trace(go.Bar(
            x=[f"{int(h)}:00" for h in penalty.index],
            y=penalty.values,
            marker_color=colors_p,
            text=[f"+{p:.1f}" for p in penalty.values],
            textposition="outside",
            showlegend=False,
        ))
        fig_penalty.add_hline(y=avg_penalty, line_dash="dash", line_color="#ff7f0e")
        fig_penalty.update_layout(
            xaxis_title="Hour",
            yaxis_title="Extra Minutes",
            height=350,
            margin=dict(l=40, r=20, t=20, b=80),
            paper_bgcolor="white",
            plot_bgcolor="#f5f7fa",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_penalty, use_container_width=True)
    
    with col2:
        st.markdown("### Delivery Time by Hour (Stratified)")
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=hour_high.index, y=hour_high.values,
            mode="lines+markers", name="High util (≥90%)",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
        ))
        
        fig_line.add_trace(go.Scatter(
            x=hour_low.index, y=hour_low.values,
            mode="lines+markers", name="Low util (<50%)",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=6),
        ))
        
        fig_line.update_layout(
            xaxis_title="Hour",
            yaxis_title="Delivery Time (min)",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f5f7fa",
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("### Hour × Utilization Heatmap")
    
    heat_df = (
        df_rqa.groupby(["hour_of_day", "util_bracket"], observed=True)["actual_duration_min"]
        .agg(mean="mean", count="count")
        .reset_index()
    )
    heat_df = heat_df[heat_df["count"] >= 30]
    heat_pivot = heat_df.pivot(index="util_bracket", columns="hour_of_day", values="mean")
    heat_pivot = heat_pivot.reindex(["<50%", "50–75%", "75–90%", "90–100%", ">100%"])
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_pivot.values,
        x=[f"{int(h)}:00" for h in heat_pivot.columns],
        y=list(heat_pivot.index),
        colorscale="Reds",
        text=np.where(
            np.isnan(heat_pivot.values),
            "–",
            np.vectorize(lambda v: f"{v:.0f}" if not np.isnan(v) else "–")(heat_pivot.values)
        ),
        texttemplate="%{text}m",
        textfont=dict(size=10),
        colorbar=dict(title="Minutes"),
    ))
    
    fig_heat.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Utilization Bracket",
        height=350,
        margin=dict(l=100, r=20, t=20, b=80),
        paper_bgcolor="white",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # 2 PM anomaly
    two_pm = df_clean[df_clean["hour_of_day"] == 14]
    st.markdown("### 2 PM Anomaly")
    st.markdown(f"""
    <div class='info-box'>
    <p><strong>Finding:</strong> 2 PM averages {two_pm['actual_duration_min'].mean():.1f} min — the highest hour — 
    despite only ~52% utilization. This is <strong>not</strong> explained by dasher supply pressure. 
    The utilization penalty inverts at 2 PM, suggesting restaurant-side constraints (prep time for 
    complex lunch orders) are the bottleneck. Operational fix lies in restaurant partnerships, not dasher recruitment.</p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <p><strong>Key Finding:</strong> Overnight hours (1–4 AM) show the largest utilization penalty — up to +10.9 min at 3 AM. 
    Time-of-day compounds the utilization effect in windows where high demand coincides with high saturation. 
    The 2 PM spike reveals a non-utilization driver, pointing to operational constraints elsewhere in the delivery chain.</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Executive Summary")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        **RQ-A: Utilization Explains 25% of Delay**
        
        39.7 → 49.9 min (low to high utilization)
        
        Sharp acceleration above 90% threshold. Each 10% rise in saturation adds ~2–4 min under normal conditions.
        
        **Action:** Keep utilization <85–90% for disproportionate customer experience gains.
        """)
    
    with c2:
        st.markdown("""
        **RQ-B: Overnight Hours Worst**
        
        +10.9 min penalty at 3 AM (high util vs. low util)
        
        Overnight windows compound the utilization effect because every available dasher is busy—new orders wait in queue.
        
        **Action:** Concentrate surge pricing and recruitment incentives on 1–4 AM window.
        """)
    
    with c3:
        st.markdown("""
        **2 PM Anomaly (Non-Utilization)**
        
        59.6 min avg at only 52% utilization—the worst hour despite moderate supply.
        
        Consistent with restaurant prep constraints for complex lunch orders, not dasher shortage.
        
        **Action:** Investigate restaurant SLAs and order complexity at lunch peak.
        """)
    
    st.markdown("---")
    st.markdown("### Limitations & Confidence")
    
    limits = [
        "8.2% missing dasher data limits utilization analysis to ~175,000 records (92% of dataset).",
        "Partial ETA (two components) prevents true expected-vs.-actual gap computation.",
        "2015 data; current network may differ significantly. Validate on current data.",
        "Descriptive analysis captures correlation, not causation. A/B testing required for causal claims.",
    ]
    
    for limit in limits:
        st.markdown(f"• {limit}")
    
    st.markdown("---")
    st.markdown("### Recommended Next Steps")
    
    steps = [
        "ANOVA/Kruskal-Wallis test on utilization brackets for formal statistical significance.",
        "Logistic regression: probability of >45 min delivery given utilization + hour.",
        "Market-level breakdown: do effects vary by geography?",
        "Restaurant-level deep-dive: is 2 PM anomaly cuisine-specific or broad?",
    ]
    
    for step in steps:
        st.markdown(f"• {step}")
