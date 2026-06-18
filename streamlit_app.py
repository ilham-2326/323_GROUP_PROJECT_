import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dubai Office Hotspot Recommender", layout="wide")
st.title("Dubai Commercial Office Hotspot Recommender")
st.markdown("*Decision-support hotspot scoring — CSCI323 Project*")

features = ['avg_sale_price', 'rental_yield', 'transaction_count', 'contract_count',
            'avg_rent', 'mall_score', 'metro_score', 'parking_score']

# ── LOCATE DATA (works whether the app is in STREAMLIT/, the repo root, or on Colab) ──
HERE = Path(__file__).resolve().parent
CANDIDATE_DIRS = [HERE / "DATA", HERE.parent / "DATA", HERE.parent, HERE,
                  Path("DATA"), Path("/content/drive/MyDrive/dataset")]

def find_file(name):
    for d in CANDIDATE_DIRS:
        p = d / name
        if p.exists():
            return p
    return None

@st.cache_data
def load_data():
    fpath = find_file("FINAL_DATASET.csv")
    if fpath is None:
        st.error("Could not find FINAL_DATASET.csv. Put it in a DATA/ folder next to the app "
                 "or set the path in CANDIDATE_DIRS.")
        st.stop()
    df = pd.read_csv(fpath)
    df['area_id'] = df['area_id'].astype(float).astype(int)

    # bring in district names + model outputs from the rankings file if available
    rpath = find_file("hotspot_rankings_with_names.csv") or find_file("hotspot_rankings.csv")
    if rpath is not None:
        r = pd.read_csv(rpath)
        r['area_id'] = r['area_id'].astype(float).astype(int)
        keep = [c for c in ['area_id', 'area_name_en', 'predicted_hotspot', 'cluster'] if c in r.columns]
        df = df.merge(r[keep], on='area_id', how='left')

    # fall back to a names file, then to "Area <id>"
    if 'area_name_en' not in df.columns:
        npath = find_file("area_names.csv")
        if npath is not None:
            n = pd.read_csv(npath); n['area_id'] = n['area_id'].astype(float).astype(int)
            df = df.merge(n.drop_duplicates('area_id')[['area_id', 'area_name_en']],
                          on='area_id', how='left')
    df['District'] = df.get('area_name_en', pd.Series(index=df.index, dtype=object)) \
                       .fillna('Area ' + df['area_id'].astype(str))
    return df

@st.cache_resource
def fit_scaler_and_model(df):
    # Self-contained: scaler + feature-importance model are derived from the dataset itself
    # (the saved artifacts are not committed to the repo). This mirrors the notebooks:
    # the label is the top-30% of the survey score, so the model reproduces that rule.
    scaler = MinMaxScaler().fit(df[features])
    Xs = scaler.transform(df[features])
    label = (df['hotspot_score'] >= df['hotspot_score'].quantile(0.70)).astype(int)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   random_state=42).fit(Xs, label)
    return scaler, model

df = load_data()
scaler, model = fit_scaler_and_model(df)
X_scaled_df = pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index)

# if the rankings file didn't supply predictions, derive them
if 'predicted_hotspot' not in df.columns:
    df['predicted_hotspot'] = model.predict(X_scaled_df)

# ── SIDEBAR: WEIGHT SLIDERS ──────────────────────────────────────────────────
st.sidebar.header("Adjust Company Priorities")
st.sidebar.markdown("Drag sliders to reflect your client's priorities. Weights auto-normalise.")
w_sale  = st.sidebar.slider("Capital Appreciation (avg_sale_price)", 0, 10, 2)
w_yield = st.sidebar.slider("Rental Yield", 0, 10, 2)
w_txn   = st.sidebar.slider("Market Liquidity (transactions)", 0, 10, 2)
w_cont  = st.sidebar.slider("Rental Activity (contracts)", 0, 10, 2)
w_rent  = st.sidebar.slider("Avg Rent Level", 0, 10, 2)
w_mall  = st.sidebar.slider("Mall Accessibility", 0, 10, 2)
w_metro = st.sidebar.slider("Metro Accessibility", 0, 10, 2)
w_park  = st.sidebar.slider("Parking Availability", 0, 10, 1)

raw_weights = [w_sale, w_yield, w_txn, w_cont, w_rent, w_mall, w_metro, w_park]
total = sum(raw_weights) or 1
norm_weights = [w / total for w in raw_weights]

# ── RECOMPUTE SCORES WITH CUSTOM WEIGHTS ─────────────────────────────────────
custom_score = sum(X_scaled_df[f] * w for f, w in zip(features, norm_weights))
df['custom_score'] = (custom_score / custom_score.max()) * 100
df['custom_rank'] = df['custom_score'].rank(ascending=False).astype(int)

# ── MAIN PANEL ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
top1 = df.nsmallest(1, 'custom_rank').iloc[0]
col1.metric("Top Hotspot Area", top1['District'], f"Score: {top1['custom_score']:.1f}/100")
col2.metric("Total Districts", str(len(df)), "DLD ∩ Ejari")
col3.metric("Hotspot Districts", str(int((df['predicted_hotspot'] == 1).sum())), "Top 30% threshold")

st.markdown("---")
st.subheader("Ranked Office Hotspot Areas")
top_n = st.slider("Show top N areas", 5, 50, 15)
display = df.nsmallest(top_n, 'custom_rank')[
    ['District', 'area_id', 'custom_score', 'custom_rank', 'predicted_hotspot']
].rename(columns={'area_id': 'Area ID', 'custom_score': 'Hotspot Score',
                  'custom_rank': 'Rank', 'predicted_hotspot': 'ML Hotspot'})
display['ML Hotspot'] = display['ML Hotspot'].map({1: 'Yes', 0: 'No'})

fig_bar = px.bar(display, x='District', y='Hotspot Score', color='ML Hotspot',
                 color_discrete_map={'Yes': '#2980b9', 'No': '#bdc3c7'},
                 title=f"Top {top_n} Districts by Hotspot Score")
fig_bar.update_xaxes(categoryorder='total descending')
st.plotly_chart(fig_bar, use_container_width=True)
st.dataframe(display, use_container_width=True)

# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("What Drives Hotspot Classification?")
st.markdown("Random Forest feature importance — which factors most influence the hotspot label. "
            "(The label is derived from the survey score, so this shows what the score is built on.)")
fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}) \
       .sort_values('Importance', ascending=True)
fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                title="Random Forest Feature Importance",
                color='Importance', color_continuous_scale='Blues')
st.plotly_chart(fig_fi, use_container_width=True)

# ── AREA DEEP DIVE ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Area Deep Dive")
name_to_id = df.sort_values('District').set_index('District')['area_id'].to_dict()
selected_name = st.selectbox("Select a district", list(name_to_id.keys()))
selected = name_to_id[selected_name]
row = df[df['area_id'] == selected].iloc[0]

cols = st.columns(4)
cols[0].metric("Hotspot Score", f"{row['custom_score']:.1f}/100")
cols[1].metric("Rank", f"#{int(row['custom_rank'])} of {len(df)}")
cols[2].metric("Avg Sale Price /m²", f"AED {row.get('avg_sale_price', 0):,.0f}")
cols[3].metric("Rental Yield", f"{row.get('rental_yield', 0):.2%}")

vals = X_scaled_df.loc[df.index[df['area_id'] == selected][0]].tolist()
fig_radar = go.Figure(go.Scatterpolar(
    r=vals + [vals[0]], theta=features + [features[0]],
    fill='toself', name=selected_name, line_color='steelblue'))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title=f"Feature Profile — {selected_name}")
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("CSCI323 Modern Artificial Intelligence — University of Wollongong Dubai, Spring 2026 | "
           "Data: Dubai Land Department & Ejari")
