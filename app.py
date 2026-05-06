"""
DoorDash Delivery Delay Analytics
MGMT 38900 - Rohan Menon, Purdue University
Descriptive analytics on 197,428 orders to quantify utilization + time-of-day effects on delivery delays.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# ── Configuration ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DoorDash Delivery Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  h1 { margin-bottom: 4px; color: #1a1a1a; font-size: 2rem; }
  h2 { margin-top: 24px; margin-bottom: 8px; color: #1a1a1a; font-size: 1.35rem; font-weight: 600; }
  h3 { margin-top: 12px; margin-bottom: 6px; color: #333; font-size: 1.05rem; font-weight: 600; }
  .stTabs [data-baseweb="tab-list"] { 
    gap: 0; 
    background: #f5f7fa; 
    border-bottom: 1px solid #ddd; 
    border-radius: 0;
  }
  .stTabs [data-baseweb="tab"] { 
    color: #666; 
    background: transparent; 
    border: none; 
    padding: 12px 20px;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] { 
    color: #1f77b4 !important; 
    border-bottom: 3px solid #1f77b4 !important; 
    background: white; 
  }
  .info-box { 
    background: #f5f7fa; 
    border-left: 4px solid #1f77b4; 
    padding: 14px; 
    border-radius: 4px; 
    margin-bottom: 12px; 
  }
  .info-box p { 
    margin: 0; 
    color: #1a1a1a; 
    font-size: 0.95rem; 
    line-height: 1.6; 
  }
  .info-box strong { 
    color: #1f77b4; 
    font-weight: 600;
  }
  table { 
    font-size: 0.9rem; 
  }
</style>
""", unsafe_allow_html=True)

# ── Data Loading with Error Handling ────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    """Load and preprocess DoorDash data with full error handling."""
    try:
        # Construct path
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doordash.csv")
        
        # Load CSV
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        
        df = pd.read_csv(path)
        
        # Validate core columns exist
        required_cols = [
            "created_at", "actual_delivery_time", "total_onshift_dashers",
            "total_busy_dashers", "store_primary_category"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert timestamps
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"], errors="coerce")
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=["created_at", "actual_delivery_time"])
        
        # Compute delivery duration (minutes)
        df["actual_duration_min"] = (
            df["actual_delivery_time"] - df["created_at"]
        ).dt.total_seconds() / 60
        
        # Validate durations are positive
        df = df[df["actual_duration_min"] > 0]
        
        # Compute utilization
        df["dasher_utilization"] = (
            df["total_busy_dashers"] / df["total_onshift_dashers"].replace(0, np.nan)
        )
        
        # Extract hour of day
        df["hour_of_day"] = df["created_at"].dt.hour
        
        # Handle missing categories
        df["store_primary_category"] = df["store_primary_category"].fillna("unknown")
        
        # Count outliers before removal
        outlier_count = (df["actual_duration_min"] > 120).sum()
        
        # Remove outliers (deliveries > 120 min assumed to be data errors)
        df_clean = df[df["actual_duration_min"] <= 120].copy()
        
        # Create utilization brackets only for clean data
        df_clean["util_bracket"] = pd.cut(
            df_clean["dasher_utilization"],
            bins=[0, 0.5, 0.75, 0.9, 1.0, 10],
            labels=["<50%", "50–75%", "75–90%", "90–100%", ">100%"],
            include_lowest=True
        )
        
        # RQ-A dataset: valid utilization + valid brackets
        df_rqa = df_clean.dropna(subset=["dasher_utilization", "util_bracket"]).copy()
        
        return df, df_clean, df_rqa, outlier_count
    
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        st.stop()

# Load data
try:
    df_raw, df_clean, df_rqa, outlier_count = load_data()
except Exception as e:
    st.error(f"Fatal error during data loading: {e}")
    st.stop()

# ── Helper Functions ────────────────────────────────────────────────────────
def safe_aggregate(df, groupby_cols, agg_col, func="mean"):
    """Safely aggregate data with size validation."""
    try:
        result = df.groupby(groupby_cols, observed=True)[agg_col].agg(
            mean="mean", std="std", count="count"
        )
        return result
    except Exception as e:
        st.warning(f"Aggregation error: {e}")
        return pd.DataFrame()

def safe_plot(fig_func, *args, **kwargs):
    """Safely render plotly chart."""
    try:
        fig = fig_func(*args, **kwargs)
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="#f5f7fa",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI'", size=12, color="#1a1a1a"),
            hovermode="closest",
            margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
        return True
    except Exception as e:
        st.error(f"Chart rendering error: {e}")
        return False

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🚗 DoorDash Delivery Delay Analytics")
st.markdown("""
Quantifying how dasher network utilization and time of day drive delivery delays.
Built with descriptive analytics on **197,428 orders** from 2015.
""")

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🔴 RQ-A: Utilization",
    "🕐 RQ-B: Time-of-Day",
    "💡 Summary"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")
    
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{len(df_raw):,}")
    with col2:
        st.metric("Valid Utilization", f"{len(df_rqa):,}")
    with col3:
        st.metric("Outliers Removed", f"{outlier_count:,}")
    with col4:
        st.metric("Markets", "6")
    
    # Delivery time stats
    st.markdown("### Delivery Time Statistics (Post-Cleaning)")
    if not df_clean.empty:
        stats = df_clean["actual_duration_min"].describe().round(1)
        stat_df = pd.DataFrame({
            "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"],
            "Value (minutes)": [
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
    else:
        st.warning("No clean data available")
    
    # Data quality
    st.markdown("### Data Quality Report")
    quality_data = {
        "Issue": ["Dasher fields missing", "Category missing", "Outliers (>120 min)"],
        "Count": [
            df_raw["total_onshift_dashers"].isna().sum(),
            (df_raw["store_primary_category"].isna()).sum(),
            outlier_count
        ],
    }
    quality_data["% of Total"] = [
        f"{v / len(df_raw) * 100:.2f}%" for v in quality_data["Count"]
    ]
    quality_data["Action"] = [
        "Excluded from RQ-A only",
        "Labeled 'unknown'",
        "Removed before analysis"
    ]
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, hide_index=True, use_container_width=True)
    
    # Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Utilization Distribution")
        try:
            fig_util = go.Figure()
            fig_util.add_trace(go.Histogram(
                x=df_rqa["dasher_utilization"] * 100,
                nbinsx=50,
                marker_color="#1f77b4",
                name="Orders",
                hovertemplate="Utilization: %{x:.0f}%<br>Count: %{y}<extra></extra>"
            ))
            fig_util.add_vline(
                x=90,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="90% threshold",
                annotation_position="top right"
            )
            fig_util.update_layout(
                xaxis_title="Dasher Utilization (%)",
                yaxis_title="Number of Orders",
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_util, use_container_width=True, config={"responsive": True})
        except Exception as e:
            st.error(f"Utilization histogram error: {e}")
    
    with col2:
        st.markdown("### Delivery Time Distribution")
        try:
            fig_dur = go.Figure()
            fig_dur.add_trace(go.Histogram(
                x=df_clean["actual_duration_min"],
                nbinsx=50,
                marker_color="#2ca02c",
                name="Orders",
                hovertemplate="Duration: %{x:.0f} min<br>Count: %{y}<extra></extra>"
            ))
            fig_dur.update_layout(
                xaxis_title="Delivery Time (minutes)",
                yaxis_title="Number of Orders",
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_dur, use_container_width=True, config={"responsive": True})
        except Exception as e:
            st.error(f"Duration histogram error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: RQ-A
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Research Question A: Dasher Supply Pressure")
    st.markdown("""
    **To what extent does dasher network saturation explain delivery time variation, 
    and at what utilization threshold does the effect become practically significant?**
    """)
    
    try:
        # Stats by bracket
        bracket_stats = (
            df_rqa.groupby("util_bracket", observed=True)["actual_duration_min"]
            .agg(mean="mean", count="count")
            .reset_index()
        )
        
        # Ensure correct order
        bracket_stats["util_bracket"] = pd.Categorical(
            bracket_stats["util_bracket"],
            categories=["<50%", "50–75%", "75–90%", "90–100%", ">100%"],
            ordered=True
        )
        bracket_stats = bracket_stats.sort_values("util_bracket")
        
        if bracket_stats.empty:
            st.warning("No valid bracket data available")
        else:
            col_metric, col_chart = st.columns([1, 2])
            
            with col_metric:
                # Key metrics
                low_util = bracket_stats.iloc[0]
                high_util = bracket_stats.iloc[-1]
                pct_change = (high_util["mean"] - low_util["mean"]) / low_util["mean"] * 100
                
                st.markdown(f"""
                <div class='info-box'>
                    <p><strong>{low_util['mean']:.1f} min</strong> at &lt;50% utilization</p>
                    <p><strong>{high_util['mean']:.1f} min</strong> at 90–100% utilization</p>
                    <p><strong>+{pct_change:.1f}%</strong> increase</p>
                </div>""", unsafe_allow_html=True)
                
                st.markdown("**Statistics by Bracket:**")
                display = bracket_stats[["util_bracket", "mean", "count"]].copy()
                display["mean"] = display["mean"].round(1)
                st.dataframe(
                    display.rename(columns={
                        "util_bracket": "Bracket",
                        "mean": "Avg (min)",
                        "count": "N"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
            
            with col_chart:
                # Bar chart
                fig_bar = go.Figure()
                colors = ["#2ca02c", "#ff7f0e", "#ff7f0e", "#d62728", "#d62728"]
                
                for idx, (_, row) in enumerate(bracket_stats.iterrows()):
                    fig_bar.add_trace(go.Bar(
                        x=[str(row["util_bracket"])],
                        y=[row["mean"]],
                        marker_color=colors[idx],
                        text=f"{row['mean']:.1f}",
                        textposition="outside",
                        hovertemplate=f"Bracket: {row['util_bracket']}<br>Avg: {row['mean']:.1f} min<br>Orders: {row['count']:.0f}<extra></extra>",
                        showlegend=False,
                    ))
                
                fig_bar.update_layout(
                    title="Average Delivery Time by Utilization Bracket",
                    xaxis_title="Utilization Bracket",
                    yaxis_title="Delivery Time (minutes)",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"responsive": True})
        
        # Threshold analysis
        st.markdown("### Threshold Analysis (5-Percentile Bins)")
        
        bins = np.arange(0, 1.01, 0.05)
        threshold_data = []
        
        for i in range(len(bins) - 1):
            mask = (df_rqa["dasher_utilization"] >= bins[i]) & (df_rqa["dasher_utilization"] < bins[i+1])
            if mask.sum() >= 30:
                threshold_data.append({
                    "util_pct": (bins[i] + bins[i+1]) / 2 * 100,
                    "mean_time": df_rqa.loc[mask, "actual_duration_min"].mean(),
                    "count": mask.sum()
                })
        
        if threshold_data:
            td = pd.DataFrame(threshold_data)
            
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(
                x=td["util_pct"],
                y=td["mean_time"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=8, color="#1f77b4"),
                hovertemplate="Utilization: %{x:.0f}%<br>Avg Time: %{y:.1f} min<extra></extra>",
                name="Delivery Time"
            ))
            
            fig_thresh.add_vline(
                x=90,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="90% inflection point",
                annotation_position="top right"
            )
            
            fig_thresh.update_layout(
                xaxis_title="Dasher Utilization (%)",
                yaxis_title="Delivery Time (minutes)",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_thresh, use_container_width=True, config={"responsive": True})
        else:
            st.warning("Insufficient data for threshold analysis")
        
        # Key insight
        st.markdown("""
        <div class='info-box'>
        <p><strong>Key Finding:</strong> Delivery time increases monotonically from 39.7 min (low utilization) 
        to 49.9 min (high utilization), a 25.5% increase. The effect is gradual below 90%, then 
        accelerates sharply above—indicating a <strong>non-linear queuing regime</strong>. Operations should 
        target keeping utilization below 85–90% for disproportionate customer experience gains.</p>
        </div>""", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"RQ-A analysis error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: RQ-B
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Research Question B: Time-of-Day Compounding")
    st.markdown("""
    **Do specific hours compound the utilization effect—are longest deliveries 
    concentrated where both demand is high AND dasher availability is low?**
    """)
    
    try:
        # Compute hourly aggregates
        hour_high = df_rqa[df_rqa["dasher_utilization"] >= 0.9].groupby("hour_of_day")["actual_duration_min"].mean()
        hour_low = df_rqa[df_rqa["dasher_utilization"] < 0.5].groupby("hour_of_day")["actual_duration_min"].mean()
        hour_all = df_clean.groupby("hour_of_day")["actual_duration_min"].mean()
        
        penalty = (hour_high - hour_low).dropna()
        avg_penalty = penalty.mean() if len(penalty) > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Utilization Penalty by Hour (High − Low)")
            if len(penalty) > 0:
                fig_penalty = go.Figure()
                colors_p = ["#d62728" if p >= avg_penalty else "#1f77b4" for p in penalty.values]
                
                fig_penalty.add_trace(go.Bar(
                    x=[f"{int(h):02d}:00" for h in penalty.index],
                    y=penalty.values,
                    marker_color=colors_p,
                    text=[f"+{p:.1f}" for p in penalty.values],
                    textposition="outside",
                    hovertemplate="Hour: %{x}<br>Penalty: +%{y:.1f} min<extra></extra>",
                    showlegend=False,
                ))
                
                fig_penalty.add_hline(
                    y=avg_penalty,
                    line_dash="dash",
                    line_color="#ff7f0e",
                    annotation_text=f"Avg: +{avg_penalty:.1f} min"
                )
                
                fig_penalty.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Extra Minutes (High util vs. Low util)",
                    height=350,
                    showlegend=False,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_penalty, use_container_width=True, config={"responsive": True})
            else:
                st.warning("Insufficient data for penalty calculation")
        
        with col2:
            st.markdown("### Delivery Time by Hour (Stratified)")
            if len(hour_high) > 0 and len(hour_low) > 0:
                fig_line = go.Figure()
                
                fig_line.add_trace(go.Scatter(
                    x=hour_high.index, y=hour_high.values,
                    mode="lines+markers",
                    name="High util (≥90%)",
                    line=dict(color="#d62728", width=2),
                    marker=dict(size=6),
                    hovertemplate="Hour: %{x:02.0f}:00<br>Avg Time: %{y:.1f} min<extra></extra>"
                ))
                
                fig_line.add_trace(go.Scatter(
                    x=hour_low.index, y=hour_low.values,
                    mode="lines+markers",
                    name="Low util (<50%)",
                    line=dict(color="#2ca02c", width=2),
                    marker=dict(size=6),
                    hovertemplate="Hour: %{x:02.0f}:00<br>Avg Time: %{y:.1f} min<extra></extra>"
                ))
                
                fig_line.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Delivery Time (minutes)",
                    height=350,
                    legend=dict(x=0.7, y=0.95),
                )
                st.plotly_chart(fig_line, use_container_width=True, config={"responsive": True})
            else:
                st.warning("Insufficient hourly data")
        
        # Heatmap
        st.markdown("### Hour × Utilization Heatmap")
        
        heat_df = (
            df_rqa.groupby(["hour_of_day", "util_bracket"], observed=True)["actual_duration_min"]
            .agg(mean="mean", count="count")
            .reset_index()
        )
        heat_df = heat_df[heat_df["count"] >= 20]  # Cells with <20 orders excluded
        
        if not heat_df.empty:
            heat_pivot = heat_df.pivot(index="util_bracket", columns="hour_of_day", values="mean")
            heat_pivot = heat_pivot.reindex(["<50%", "50–75%", "75–90%", "90–100%", ">100%"])
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_pivot.values,
                x=[f"{int(h):02d}:00" for h in heat_pivot.columns],
                y=list(heat_pivot.index),
                colorscale="Reds",
                text=np.where(
                    np.isnan(heat_pivot.values),
                    "–",
                    np.vectorize(lambda v: f"{v:.0f}" if not np.isnan(v) else "–")(heat_pivot.values)
                ),
                texttemplate="%{text}m",
                textfont=dict(size=10, color="white"),
                colorbar=dict(title="Minutes"),
                hovertemplate="Hour: %{x}<br>Bracket: %{y}<br>Avg: %{z:.1f} min<extra></extra>"
            ))
            
            fig_heat.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Utilization Bracket",
                height=350,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"responsive": True})
        else:
            st.warning("Insufficient data for heatmap")
        
        # 2 PM Anomaly
        st.markdown("### 2 PM Anomaly Investigation")
        
        two_pm = df_clean[df_clean["hour_of_day"] == 14]
        if not two_pm.empty:
            two_pm_mean = two_pm["actual_duration_min"].mean()
            two_pm_util = df_rqa[df_rqa["hour_of_day"] == 14]["dasher_utilization"].mean()
            
            st.markdown(f"""
            <div class='info-box'>
            <p><strong>Finding:</strong> 2 PM averages <strong>{two_pm_mean:.1f} min</strong>—the highest hour—despite 
            only ~{two_pm_util*100:.0f}% utilization. This is <strong>not</strong> explained by dasher supply. 
            The utilization penalty inverts at 2 PM, suggesting restaurant-side constraints (longer prep time 
            for complex lunch orders) are the bottleneck. Operational fix lies in restaurant partnerships, 
            not dasher recruitment.</p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <p><strong>Key Finding:</strong> Overnight hours (1–4 AM) show the largest utilization penalty—up to +10.9 min 
        at 3 AM. Time-of-day <strong>compounds</strong> the utilization effect in windows where high demand coincides 
        with high saturation. The 2 PM spike reveals a non-utilization driver, pointing to operational constraints 
        elsewhere in the delivery chain.</p>
        </div>""", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"RQ-B analysis error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Executive Summary")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        **RQ-A: Utilization Explains 25% of Delay**
        
        **39.7 → 49.9 min** (low to high utilization)
        
        Sharp acceleration above 90% threshold. Each 10% rise in saturation adds ~2–4 min.
        
        **→ Action:** Keep utilization <85–90% for disproportionate gains.
        """)
    
    with c2:
        st.markdown("""
        **RQ-B: Overnight Hours Worst**
        
        **+10.9 min penalty** at 3 AM (high util vs. low util)
        
        Overnight windows compound the utilization effect—every dasher is busy, new orders queue.
        
        **→ Action:** Concentrate surge pricing & recruitment on 1–4 AM.
        """)
    
    with c3:
        st.markdown("""
        **2 PM Anomaly (Non-Utilization)**
        
        **59.6 min avg** at only 52% utilization—worst hour despite moderate supply.
        
        Consistent with restaurant prep constraints, not dasher shortage.
        
        **→ Action:** Investigate restaurant SLAs & order complexity.
        """)
    
    st.markdown("---")
    st.markdown("### Limitations & Confidence")
    
    limits = [
        "🔹 **8.2% missing dasher data** limits utilization analysis to ~175,000 records (92% of dataset).",
        "🔹 **Partial ETA** (two components) prevents true expected-vs.-actual gap computation.",
        "🔹 **2015 data**; current network may differ significantly. Validate on current data.",
        "🔹 **Descriptive analysis** captures correlation, not causation. A/B testing required for causal claims.",
    ]
    
    for limit in limits:
        st.markdown(limit)
    
    st.markdown("---")
    st.markdown("### Recommended Next Steps")
    
    steps = [
        "✓ ANOVA/Kruskal-Wallis test on utilization brackets for formal statistical significance.",
        "✓ Logistic regression: probability of >45 min delivery given utilization + hour.",
        "✓ Market-level breakdown: do effects vary by geography?",
        "✓ Restaurant-level deep-dive: is 2 PM anomaly cuisine-specific or broad?",
    ]
    
    for step in steps:
        st.markdown(step)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='color:#666; font-size:0.85rem; margin-top: 24px;'>
<strong>Dataset:</strong> 197,428 DoorDash orders from 2015 | 
<strong>Method:</strong> Descriptive analytics | 
<strong>Project:</strong> MGMT 38900 Individual Project, Purdue University
</p>""", unsafe_allow_html=True)
