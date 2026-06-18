# Dubai Commercial Office Hotspot Recommender
streamlit link: https://323groupproject-28u9xhvgfvijl2h4ksquou.streamlit.app/ 

A data-driven decision-support tool that scores and ranks Dubai districts by how attractive they are
for opening a new **commercial office**, built from public Dubai Land Department (DLD) and Ejari data
plus a stakeholder priority survey.

Built for **CSCI323 Modern Artificial Intelligence**, University of Wollongong Dubai (Spring 2026).

---

## What it does

Choosing where to open an office in Dubai is a high-stakes, multi-factor decision. This project turns
it into a transparent, reproducible score. For each of **165 districts** (those present in *both* the
transaction and rent datasets) we:

1. Engineer **8 district-level features** from millions of raw DLD transaction and Ejari rent records.
2. Combine them into a **survey-weighted composite "hotspot score" (0–100)** using client priorities.
3. Use **K-Means** for unsupervised district segmentation, and **Random Forest / XGBoost** to explain
   which features drive the score (feature importance).
4. (In progress) Serve the results through an interactive **Streamlit dashboard**.

---

## The 8 features

The client survey rated 12 candidate factors, but only **8 had reliable open data** in DLD/Ejari; the
score is renormalised over those 8. (Off-plan momentum, regulatory/tax status, Grade-A asset density
and highway access are not in the data and are not used.)

| Feature | Meaning | Source |
|---|---|---|
| `avg_sale_price` | Mean sale price per m² | DLD transactions |
| `rental_yield` | Annual office rent ÷ avg total sale value (IQR-capped, 0–14.7%) | DLD + Ejari |
| `transaction_count` | Sale-transaction volume (market liquidity) | DLD transactions |
| `contract_count` | Office rent-contract volume (absorption) | Ejari |
| `avg_rent` | Mean annual office rent | Ejari |
| `mall_score` | Share of records with a nearby mall (0–1 proximity proxy) | DLD transactions |
| `metro_score` | Share of records with a nearby metro (0–1 proximity proxy) | DLD transactions |
| `parking_score` | Share of transactions with parking | DLD transactions |

---

## Top 10 hotspots (default survey weights)

| Rank | District | Hotspot score |
|---|---|---|
| 1 | Business Bay | 100.0 |
| 2 | Marsa Dubai / Dubai Marina | 98.8 |
| 3 | Al Thanyah Fifth / JLT | 94.5 |
| 4 | Zaabeel Second | 87.5 |
| 5 | Burj Khalifa / Downtown | 86.2 |
| 6 | Zaabeel First | 85.8 |
| 7 | Al Barsha South Fourth | 82.8 |
| 8 | Al Barsha South Fifth | 81.6 |
| 9 | Al Thanyah Third | 80.6 |
| 10 | Jumeirah Second | 80.5 |

These are Dubai's established commercial cores — reasonable face validity.

---

## Project structure

```
.
├── DATA/
│   ├── FINAL_DATASET_FINAL.csv               # engineered, scored dataset (165 districts)
│   ├── FINAL_TRANSACTION1.1.csv        # DLD aggregates (price, value, volume, parking, proximity)
│   ├── rents_preprocessed_final1.0.csv # cleaned Ejari office contracts
│   ├── survey_responses.xlsx / survey.csv  # client feature-priority survey
│   ├── hotspot_rankings_with_names.csv # final ranked output (with district names)
│   └── *.zip / *b4agg.csv              # raw + intermediate data (NOT in git — see note)
├── PREPROCESSING/
│   ├── 01_rent_preprocessing.ipynb
│   ├── 02_transactions_preprocessing.ipynb   # incl. metro/mall proximity + avg_sale_value
│   ├── 03_feature_engineering.ipynb          # merge, rental_yield, survey weights, hotspot score
│   └── 03b_eda.ipynb                         # exploratory data analysis
├── MODEL_IMP/
│   └── 04_ml_model.ipynb               # K-Means + RF/XGBoost + ranked output
├── EVALUATION/
│   └── 05_xgboost_evaluation.ipynb     # CV, baselines, ROC/PR, permutation + SHAP, PCA
├── STREAMLIT/                          # dashboard (in progress)
├── requirements.txt
└── README.md
```

---

## How to run
## Dataset
The full dataset is too large for GitHub.

Download it here:
[https://drive.google.com/your-link](https://drive.google.com/drive/folders/1iYGrKL19cvJ_-lNrIPpj5EA_NK7K_VWV?usp=drive_link)
The notebooks were developed on **Google Colab** with data in Google Drive. Run in order:

```
PREPROCESSING/01_rent_preprocessing.ipynb
PREPROCESSING/02_transactions_preprocessing.ipynb
PREPROCESSING/03_feature_engineering.ipynb   ->  writes FINAL_DATASET.csv
PREPROCESSING/03b_eda.ipynb                   ->  exploratory analysis
MODEL_IMP/04_ml_model.ipynb                   ->  writes hotspot_rankings.csv, full_results.csv
EVALUATION/05_xgboost_evaluation.ipynb        ->  evaluation (run !pip install shap first)
```

Each notebook has a `DATA_PATH` / load cell near the top — point it at wherever your dataset lives
(e.g. `/content/drive/MyDrive/dataset/`). Install dependencies with `pip install -r requirements.txt`.

> **Note on data & git:** the raw DLD/Ejari exports and large intermediate files are intentionally
> **not committed** to the repository (per course guidance) — they are large and the deliverable is the
> code + engineered dataset. Raw data is available from the
> [Dubai Pulse open-data portal](https://www.dubaipulse.gov.ae/).

---

## Limitations (read before trusting the numbers)

- **The hotspot score is an index, not a prediction.** It is a transparent weighted sum of the 8
  features. The supervised models are trained to reproduce the top-30% of that score, so their labels
  are *derived from the same features* — they **explain and rank feature importance**, they do **not**
  predict an independent outcome, and high accuracy/AUC is expected, not evidence of predictive skill.
  The **final ranking comes from the survey score**, not from in-sample model predictions. K-Means is
  the only genuinely unsupervised step.
- **A truly predictive model would need an external target** (e.g. next-period rent growth or
  occupancy), which this dataset does not contain.
- **`rental_yield`** is a district-level proxy (office annual rent ÷ average total sale value),
  IQR-capped to 0–14.7%.
- **`metro_score` / `mall_score`** are proximity *proxies* (share of records flagged with a nearby
  metro/mall), and are highly correlated (~0.94) — largely the same accessibility signal.
- **Survey weights** reflect a single client stakeholder's response (averaged for robustness).

---

## Data sources

- **Dubai Land Department (DLD)** — property sale transactions (price, value, volume, parking, proximity)
- **Ejari** — office rental contracts (rent, contract volume)
- **Client survey** — feature-priority weighting (1–10 per factor)

## Team

Five students, University of Wollongong Dubai — CSCI323 Modern Artificial Intelligence, Spring 2026.
Instructors: Dr. Milan Dordevic, Dr. Abdullah El Nokiti.
