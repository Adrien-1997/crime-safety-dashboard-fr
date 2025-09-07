# Crime & Safety Dashboard — France (2016–2024)

Turn official police & gendarmerie records into **clear, per-1,000-inhabitants** indicators for decision-makers.  
*Goal:* fast situational awareness, anomaly surfacing, and simple forecasting.

---

## (1) Problem
Public safety teams struggle to get a unified, comparable view of incidents across time and geography.  
This app provides:
- Consistent **rates per 1,000 inhabitants** (no ambiguous %)
- Spatial & temporal exploration
- Anomaly detection to flag unusual patterns
- Lightweight forecasting (baseline)

---

## (2) Data
- **Incidents:** official police/gendarmerie records (monthly granularity)  
- **Population:** latest INSEE reference to normalize rates  
- **Geographies:** commune/department codes (INSEE)

> Raw files are **not** tracked in git. See `data/README.md` for download instructions.

**Important:** All indicators are expressed as **number per 1,000 inhabitants** (no symbols, no percentages) for clarity and comparability.

---

## (3) Approach
- **Ingestion & Cleaning:** harmonize schemas, map geo codes, handle missingness  
- **Normalization:** compute rates per 1,000 inhabitants for every indicator  
- **Exploration:** spatio-temporal drill-downs, top movers, seasonality hints  
- **Anomaly detection:** IsolationForest with ROC-aware threshold selection  
- **Forecasting:** simple time-series baselines to set expectations *(WIP)*

---

## (4) App (Streamlit)
- **Filters:** period, geography, category  
- **KPIs:** level, trend, volatility  
- **Charts:** time series, YoY deltas, heatmaps, “anomaly spotlight”  
- **Download:** CSV extracts for further analysis

---

## (5) Reproduce
~~~bash
# Python 3.11+ recommended
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# Optional: fetch data (see data/README.md)
# python data/download.py

# Run the app
streamlit run app.py
~~~

---

## (6) Project structure
~~~text
.
├─ app.py
├─ src/
│  ├─ io/                 # loaders, readers, caching
│  ├─ features/           # transforms, rate computations
│  ├─ models/             # anomaly detection, baselines
│  └─ viz/                # plotting helpers
├─ data/                  # not versioned; see README for sources
│  └─ README.md
├─ assets/
│  └─ preview.png         # 1200x630 OG image
├─ requirements.txt
├─ .gitignore
└─ LICENSE
~~~

---

## (7) Evaluation notes
- Anomalies validated with **precision/recall curves** and **cost-aware** thresholds.  
- Prefer **interpretability** over complex models for operational adoption.

---

## (8) Next steps
- [ ] Weekly aggregation toggle  
- [ ] Geo tiles for communes (clustered markers)  
- [ ] Parameter panel for IsolationForest (contamination, seeds)  
- [ ] CI job to refresh cached datasets monthly

---

## (9) Credits
Author: Adrien Morel (Paris). Color palette: blue–orange for contrast and accessibility.  
License: **MIT**.

---

### Appendix A — Minimal `requirements.txt`
~~~text
streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
pyyaml
~~~

### Appendix B — Minimal `.gitignore`
~~~gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/*
!data/README.md
~~~
