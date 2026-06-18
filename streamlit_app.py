import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + THEME
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Dubai Office Hotspot Recommender",
                   page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

NAVY, GOLD, SKY, GREY = "#0f2a4a", "#c9a227", "#2980b9", "#e8edf3"
PLOTLY_TEMPLATE = "plotly_white"

st.markdown(f"""
<style>
.stApp {{ background: linear-gradient(180deg,#f7f9fc 0%, #eef2f7 100%); }}
.hero {{
    background: linear-gradient(120deg,{NAVY} 0%, #1c4a82 100%);
    padding: 28px 34px; border-radius: 16px; color: white;
    box-shadow: 0 6px 24px rgba(15,42,74,.25); margin-bottom: 8px;
}}
.hero h1 {{ margin:0; font-size: 2.0rem; font-weight: 800; letter-spacing:-.5px; }}
.hero p  {{ margin:.35rem 0 0; opacity:.9; font-size:1.02rem; }}
div[data-testid="stMetric"] {{
    background: #ffffff; border:1px solid {GREY}; border-left:5px solid {GOLD};
    padding:14px 18px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.04);
}}
div[data-testid="stMetricValue"] {{ color:{NAVY}; font-weight:800; }}
section[data-testid="stSidebar"] {{ background:{NAVY}; }}
section[data-testid="stSidebar"] * {{ color:#eaf0f7 !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background:#ffffff; border:1px solid {GREY}; border-radius:10px 10px 0 0; padding:8px 18px;
}}
.stTabs [aria-selected="true"] {{ background:{NAVY}; color:white; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>🏙️ Dubai Commercial Office Hotspot Recommender</h1>
  <p>Decision-support scoring across 165 Dubai districts • CSCI323 — University of Wollongong Dubai</p>
</div>
""", unsafe_allow_html=True)

features = ['avg_sale_price', 'rental_yield', 'transaction_count', 'contract_count',
            'avg_rent', 'mall_score', 'metro_score', 'parking_score']
PRETTY = {'avg_sale_price':'Sale price /m²','rental_yield':'Rental yield','transaction_count':'Market liquidity',
          'contract_count':'Rental activity','avg_rent':'Avg rent','mall_score':'Mall access',
          'metro_score':'Metro access','parking_score':'Parking'}

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (robust)
# ════════════════════════════════════════════════════════════════════════════
def _dirs():
    h = Path(__file__).resolve().parent
    return [h, h/"DATA", h.parent, h.parent/"DATA", Path.cwd(), Path.cwd()/"DATA",
            Path("/content/drive/MyDrive/dataset")]

def find_file(names):
    names = [names] if isinstance(names, str) else names
    low = {n.lower() for n in names}
    for d in _dirs():
        try:
            if d.exists():
                for p in d.iterdir():
                    if p.is_file() and p.name.lower() in low:
                        return p
        except Exception:
            pass
    for base in {Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path.cwd()}:
        try:
            for p in base.rglob("*.csv"):
                if p.name.lower() in low:
                    return p
        except Exception:
            pass
    return None

@st.cache_data
def load_data():
    fp = find_file(["dashboard_data.csv", "FINAL_DATASET.csv", "final_dataset.csv"])
    if fp is None:
        return None
    df = pd.read_csv(fp)
    df['area_id'] = df['area_id'].astype(float).astype(int)
    if 'area_name_en' not in df.columns:
        np_ = find_file("area_names.csv")
        if np_ is not None:
            n = pd.read_csv(np_); n['area_id'] = n['area_id'].astype(float).astype(int)
            df = df.merge(n.drop_duplicates('area_id')[['area_id','area_name_en']], on='area_id', how='left')
    df['District'] = df.get('area_name_en', pd.Series(dtype=object)).fillna('Area ' + df['area_id'].astype(str))
    return df

df = load_data()
if df is None:
    st.warning("Couldn't find **dashboard_data.csv**. Upload it below to continue.")
    up = st.file_uploader("Upload dashboard_data.csv (or FINAL_DATASET.csv)", type="csv")
    if up is None:
        st.stop()
    df = pd.read_csv(up); df['area_id'] = df['area_id'].astype(float).astype(int)
    df['District'] = df.get('area_name_en', pd.Series(dtype=object)).fillna('Area ' + df['area_id'].astype(str))

@st.cache_resource
def fit_model(data):
    scaler = MinMaxScaler().fit(data[features])
    Xs = scaler.transform(data[features])
    label = (data['hotspot_score'] >= data['hotspot_score'].quantile(0.70)).astype(int)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42).fit(Xs, label)
    return scaler, model

scaler, model = fit_model(df)
X_scaled = pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index)
if 'predicted_hotspot' not in df.columns:
    df['predicted_hotspot'] = model.predict(X_scaled)

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — WEIGHTS
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚖️  Company Priorities")
st.sidebar.caption("Drag to match your client's priorities. Weights auto-normalise.")
sliders = {
    'avg_sale_price': st.sidebar.slider("Capital appreciation (price/m²)", 0, 10, 2),
    'rental_yield':   st.sidebar.slider("Rental yield", 0, 10, 2),
    'transaction_count': st.sidebar.slider("Market liquidity", 0, 10, 2),
    'contract_count': st.sidebar.slider("Rental activity", 0, 10, 2),
    'avg_rent':       st.sidebar.slider("Avg rent level", 0, 10, 2),
    'mall_score':     st.sidebar.slider("Mall accessibility", 0, 10, 2),
    'metro_score':    st.sidebar.slider("Metro accessibility", 0, 10, 2),
    'parking_score':  st.sidebar.slider("Parking availability", 0, 10, 1),
}
total = sum(sliders.values()) or 1
norm_w = {k: v/total for k, v in sliders.items()}

df['custom_score'] = (sum(X_scaled[f]*norm_w[f] for f in features)).pipe(lambda s: s/s.max()*100)
df['custom_rank'] = df['custom_score'].rank(ascending=False).astype(int)

# ════════════════════════════════════════════════════════════════════════════
#  KPI ROW
# ════════════════════════════════════════════════════════════════════════════
top1 = df.nsmallest(1, 'custom_rank').iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("🥇 Top Hotspot", top1['District'], f"{top1['custom_score']:.1f} / 100")
k2.metric("📍 Districts Scored", str(len(df)))
k3.metric("🔥 Hotspot Districts", int((df['predicted_hotspot'] == 1).sum()), "top 30%")
k4.metric("📈 Median Score", f"{df['custom_score'].median():.1f}")

st.write("")
tab1, tab2, tab3 = st.tabs(["🏆  Rankings", "🧭  What drives it", "🔍  District deep-dive"])

# ── TAB 1: RANKINGS ──────────────────────────────────────────────────────────
with tab1:
    top_n = st.slider("Show top N districts", 5, 50, 15)
    d = df.nsmallest(top_n, 'custom_rank').copy()
    d['ML Hotspot'] = d['predicted_hotspot'].map({1: 'Yes', 0: 'No'})
    fig = px.bar(d, x='custom_score', y='District', orientation='h', color='ML Hotspot',
                 color_discrete_map={'Yes': NAVY, 'No': '#b9c4d0'},
                 labels={'custom_score': 'Hotspot score'},
                 title=f"Top {top_n} Districts by Hotspot Score", template=PLOTLY_TEMPLATE)
    fig.update_layout(yaxis=dict(autorange="reversed"), height=520,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    show = d[['custom_rank', 'District', 'custom_score', 'rental_yield', 'metro_score', 'ML Hotspot']] \
        .rename(columns={'custom_rank': 'Rank', 'custom_score': 'Score',
                         'rental_yield': 'Yield', 'metro_score': 'Metro'})
    st.dataframe(
        show.style
            .background_gradient(subset=['Score'], cmap='Blues')
            .format({'Score': '{:.1f}', 'Yield': '{:.1%}', 'Metro': '{:.2f}'}),
        use_container_width=True, hide_index=True)

# ── TAB 2: FEATURE IMPORTANCE ────────────────────────────────────────────────
with tab2:
    st.markdown("**What the hotspot label is built on** — Random Forest feature importance. "
                "The label is derived from the survey score, so this shows which factors carry the score.")
    fi = pd.DataFrame({'Feature': [PRETTY[f] for f in features],
                       'Importance': model.feature_importances_}).sort_values('Importance')
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Blues', template=PLOTLY_TEMPLATE)
    fig.update_layout(height=440, coloraxis_showscale=False,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.info("Note: metro and mall accessibility are highly correlated (~0.94) — largely the same signal.")

# ── TAB 3: DEEP DIVE ─────────────────────────────────────────────────────────
with tab3:
    name_to_id = df.sort_values('District').set_index('District')['area_id'].to_dict()
    sel_name = st.selectbox("Select a district", list(name_to_id.keys()))
    row = df[df['area_id'] == name_to_id[sel_name]].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hotspot Score", f"{row['custom_score']:.1f}/100")
    c2.metric("Rank", f"#{int(row['custom_rank'])} of {len(df)}")
    c3.metric("Sale Price /m²", f"AED {row.get('avg_sale_price', 0):,.0f}")
    c4.metric("Rental Yield", f"{row.get('rental_yield', 0):.2%}")

    vals = X_scaled.loc[df.index[df['area_id'] == name_to_id[sel_name]][0]].tolist()
    cats = [PRETTY[f] for f in features]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                    fill='toself', line_color=NAVY, fillcolor='rgba(15,42,74,.25)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title=f"Feature Profile — {sel_name}", height=460,
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("CSCI323 Modern Artificial Intelligence — University of Wollongong Dubai, Spring 2026 · "
           "Data: Dubai Land Department & Ejari · Scores are a decision-support index, not a prediction.")
