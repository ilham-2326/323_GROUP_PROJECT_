import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths relative to this file so the app runs from any working directory
BASE = Path(__file__).resolve().parent.parent
MODELING = BASE / "modeling"

st.set_page_config(page_title="Dubai Office Hotspot Recommender", layout="wide")
st.title("Dubai Commercial Office Hotspot Recommender")
st.markdown("*Decision-support hotspot scoring — CSCI323 Project*")

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(MODELING / "hotspot_rankings.csv")

@st.cache_resource
def load_model():
    model  = pickle.load(open(MODELING / "best_model.pkl", "rb"))
    scaler = pickle.load(open(MODELING / "scaler.pkl", "rb"))
    return model, scaler

df = load_data()
model, scaler = load_model()

features = ['avg_sale_price', 'rental_yield', 'transaction_count',
            'contract_count', 'avg_rent', 'mall_score', 'metro_score', 'parking_score']

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
full_df = pd.read_csv(MODELING / "full_results.csv")
full_df['area_id'] = full_df['area_id'].astype(float).astype(int)

# attach district names (defensive: works even if full_results predates the names join)
if 'area_name_en' not in full_df.columns:
    _names = pd.read_csv(BASE / "data" / "area_names.csv")
    _names['area_id'] = _names['area_id'].astype(float).astype(int)
    full_df = full_df.merge(_names.drop_duplicates('area_id')[['area_id', 'area_name_en']],
                            on='area_id', how='left')
full_df['District'] = full_df['area_name_en'].fillna('Area ' + full_df['area_id'].astype(str))

X = full_df[features]
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

custom_score = sum(X_scaled_df[f] * w for f, w in zip(features, norm_weights))
full_df['custom_score'] = (custom_score / custom_score.max()) * 100
full_df['custom_rank'] = full_df['custom_score'].rank(ascending=False).astype(int)

# ── MAIN PANEL ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
top1 = full_df.nsmallest(1, 'custom_rank').iloc[0]
col1.metric("Top Hotspot Area",  top1['District'], f"Score: {top1['custom_score']:.1f}/100")
col2.metric("Total Districts",   "165", "After merging DLD + Ejari")
col3.metric("Hotspot Districts", str((full_df['predicted_hotspot'] == 1).sum()), "Top 30% threshold")

st.markdown("---")
st.subheader("Ranked Office Hotspot Areas")

top_n = st.slider("Show top N areas", 5, 50, 15)
display = full_df.nsmallest(top_n, 'custom_rank')[
    ['District', 'area_id', 'custom_score', 'custom_rank', 'predicted_hotspot']
].rename(columns={
    'area_id': 'Area ID',
    'custom_score': 'Hotspot Score',
    'custom_rank': 'Rank',
    'predicted_hotspot': 'ML Hotspot'
})
display['ML Hotspot'] = display['ML Hotspot'].map({1: 'Yes', 0: 'No'})

fig_bar = px.bar(display, x='District', y='Hotspot Score',
                 color='ML Hotspot',
                 color_discrete_map={'Yes': '#2980b9', 'No': '#bdc3c7'},
                 title=f"Top {top_n} Districts by Hotspot Score")
fig_bar.update_xaxes(categoryorder='total descending')
st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(display, use_container_width=True)

# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("What Drives Hotspot Classification?")
st.markdown("Feature importance from the selected tree model — which factors most influence the hotspot label.")
try:
    fi = pd.DataFrame({'Feature': features,
                       'Importance': model.feature_importances_}
                     ).sort_values('Importance', ascending=True)
    fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                    title="Random Forest Feature Importance",
                    color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig_fi, use_container_width=True)
except Exception:
    st.info("Feature importance is only available from the Random Forest model.")

# ── AREA DEEP DIVE ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Area Deep Dive")
name_to_id = (full_df.sort_values('District')
                     .set_index('District')['area_id'].to_dict())
selected_name = st.selectbox("Select a district", list(name_to_id.keys()))
selected = name_to_id[selected_name]
row = full_df[full_df['area_id'] == selected].iloc[0]

cols = st.columns(4)
cols[0].metric("Hotspot Score",    f"{row['custom_score']:.1f}/100")
cols[1].metric("Rank",             f"#{int(row['custom_rank'])} of 165")
cols[2].metric("Avg Sale Price /m²", f"AED {row.get('avg_sale_price', 0):,.0f}")
cols[3].metric("Rental Yield",     f"{row.get('rental_yield', 0):.2%}")

# Radar chart for this area
categories = features
vals = X_scaled_df.iloc[full_df.index[full_df['area_id'] == selected][0]].tolist()
fig_radar = go.Figure(go.Scatterpolar(
    r=vals + [vals[0]], theta=categories + [categories[0]],
    fill='toself', name=selected_name, line_color='steelblue'
))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         title=f"Feature Profile — {selected_name}")
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("CSCI323 Modern Artificial Intelligence — University of Wollongong Dubai, Spring 2026 | Data: Dubai Land Department & Ejari")
