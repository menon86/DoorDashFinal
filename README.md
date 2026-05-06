# DoorDash Delivery Delay Analytics

**MGMT 38900 Individual Project — Rohan Menon, Purdue University, Spring 2026**

## Project Overview

A production-grade **Streamlit analytics application** quantifying how dasher network utilization and time-of-day drive delivery delays in DoorDash orders. Built using **descriptive analytics** on 197,428 historical orders from 2015.

### Business Problem
DoorDash delivery estimates frequently miss target windows by 20+ minutes. Operations lack transparent diagnostics of which operational factors (dasher supply, time of day) drive delays most significantly.

### Solution
Two research questions answered via rigorous descriptive analytics:
- **RQ-A:** How much does dasher utilization explain delivery delay?
- **RQ-B:** Do specific hours compound the utilization effect?

---

## Key Findings

| Metric | Finding |
|--------|---------|
| **Utilization Impact** | 39.7 → 49.9 min (+25.5%) from low to high saturation |
| **Inflection Point** | Sharp acceleration above 90% utilization (non-linear queuing) |
| **Overnight Penalty** | 1–4 AM shows +10.9 min max penalty (3 AM) |
| **2 PM Anomaly** | 59.6 min avg at only 52% util → restaurant constraint, not supply |

### Actionable Insights
- **Operations:** Keep utilization <85–90% for disproportionate delay reduction
- **Surge Pricing:** Concentrate dynamic incentives on 1–4 AM window
- **Partnerships:** Investigate restaurant SLAs for 2 PM lunch-rush constraint

---

## Application Tabs

### 📊 Tab 1: Data Overview
KPI cards, descriptive statistics, data quality report, utilization & delivery time distributions

### 🔴 Tab 2: RQ-A – Dasher Supply Pressure
Bracket comparison chart, threshold curve (5-percentile bins), inflection point at 90%

### 🕐 Tab 3: RQ-B – Time-of-Day Compounding
Penalty bar chart, stratified line chart, hour × utilization heatmap, 2 PM anomaly investigation

### 💡 Tab 4: Executive Summary
Three-panel findings, business value, limitations, recommended next steps

---

## Data Preparation

### Cleaning Pipeline
1. **Load:** 197,428 rows, 16 features
2. **Convert:** Timestamps to datetime; drop invalid entries
3. **Engineer:** actual_duration_min, dasher_utilization, hour_of_day, util_bracket
4. **Filter:** Remove >120 min outliers (1,090 records, data errors)
5. **Validate:** 175,000 records with valid utilization data (88% of total)

### Data Quality
| Issue | Count | Action |
|-------|-------|--------|
| Missing dasher data | 16,262 (8.2%) | Excluded from RQ-A |
| Missing category | ~0 (<0.1%) | Labeled 'unknown' |
| Outliers (>120 min) | 1,090 (0.55%) | Removed |

---

## Limitations

- ⚠️ 8.2% missing dasher data reduces RQ-A to 175,000 records
- ⚠️ Partial ETA (two components) prevents true expected-vs.-actual computation
- ⚠️ 2015 data; current network may differ
- ⚠️ Descriptive (correlation), not causal. A/B testing required for confirmation

---

## Deployment

### Streamlit Cloud
1. Push `app.py`, `requirements.txt`, `doordash.csv` to GitHub
2. Connect repo to [share.streamlit.io](https://share.streamlit.io)
3. Deploy → public URL instantly available

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Technical Stack
- **Framework:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Plotly (interactive charts)
- **Deployment:** Streamlit Cloud or Hugging Face Spaces

---

## Methodology

**Analytical Approach:** Descriptive Analytics (not predictive)

**Rationale:** Both RQs ask "how much?" and "in which conditions?"—not "what will happen next?" Descriptive analytics is:
- Appropriate for diagnostic questions
- Defensible given data limitations (partial ETA, 8% missingness)
- Direct pathway to operational action

**Methods:**
- **Segment Comparison:** Mean delivery time across utilization brackets
- **Threshold Analysis:** 5-percentile bins to identify inflection points
- **Two-Way Interaction:** Hour × utilization cross-tabs for compounding effects

---

## Next Steps

1. **Statistical Testing:** ANOVA on utilization brackets
2. **Predictive Model:** Logistic regression for P(delay > 45 min)
3. **Market Breakdown:** Geographic variation analysis
4. **2 PM Deep-Dive:** Restaurant-level prep time investigation
5. **A/B Testing:** Controlled staffing intervention validation

---

## Files

- `app.py` — Main Streamlit application (production-ready)
- `doordash.csv` — 197,428 order dataset
- `requirements.txt` — Python dependencies
- `DEPLOYMENT_GUIDE.md` — Complete deployment instructions
- `README.md` — This file

---

**Status:** ✅ Production Ready  
**Last Updated:** May 2026  
**Contact:** Rohan Menon (Purdue University)
