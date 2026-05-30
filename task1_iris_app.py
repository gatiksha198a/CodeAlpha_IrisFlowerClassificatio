import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Iris Classifier | CodeAlpha",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── BACKGROUND & CUSTOM CSS ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Lato:wght@300;400;700&display=swap');

/* Full-page floral background using CSS gradient + SVG pattern */
.stApp {
    background-color: #0d0d0d;
    background-image:
        radial-gradient(ellipse at 10% 10%, rgba(255,182,193,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 20%, rgba(147,112,219,0.15) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 80%, rgba(255,105,180,0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 90%, rgba(221,160,221,0.15) 0%, transparent 40%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Ccircle cx='30' cy='30' r='18' fill='none' stroke='rgba(255,182,193,0.08)' stroke-width='1.5'/%3E%3Cellipse cx='30' cy='18' rx='8' ry='14' fill='rgba(255,182,193,0.06)' transform='rotate(0,30,30)'/%3E%3Cellipse cx='30' cy='18' rx='8' ry='14' fill='rgba(255,182,193,0.06)' transform='rotate(72,30,30)'/%3E%3Cellipse cx='30' cy='18' rx='8' ry='14' fill='rgba(255,182,193,0.06)' transform='rotate(144,30,30)'/%3E%3Cellipse cx='30' cy='18' rx='8' ry='14' fill='rgba(255,182,193,0.06)' transform='rotate(216,30,30)'/%3E%3Cellipse cx='30' cy='18' rx='8' ry='14' fill='rgba(255,182,193,0.06)' transform='rotate(288,30,30)'/%3E%3Ccircle cx='130' cy='100' r='22' fill='none' stroke='rgba(221,160,221,0.07)' stroke-width='1'/%3E%3Cellipse cx='130' cy='86' rx='9' ry='15' fill='rgba(221,160,221,0.05)' transform='rotate(0,130,100)'/%3E%3Cellipse cx='130' cy='86' rx='9' ry='15' fill='rgba(221,160,221,0.05)' transform='rotate(60,130,100)'/%3E%3Cellipse cx='130' cy='86' rx='9' ry='15' fill='rgba(221,160,221,0.05)' transform='rotate(120,130,100)'/%3E%3Cellipse cx='130' cy='86' rx='9' ry='15' fill='rgba(221,160,221,0.05)' transform='rotate(180,130,100)'/%3E%3Cellipse cx='130' cy='86' rx='9' ry='15' fill='rgba(221,160,221,0.05)' transform='rotate(240,130,100)'/%3E%3Cellipse cx='130' cy='86' rx='9' ry='15' fill='rgba(221,160,221,0.05)' transform='rotate(300,130,100)'/%3E%3Ccircle cx='170' cy='160' r='14' fill='none' stroke='rgba(255,105,180,0.07)' stroke-width='1'/%3E%3Cellipse cx='170' cy='150' rx='6' ry='11' fill='rgba(255,105,180,0.04)' transform='rotate(0,170,160)'/%3E%3Cellipse cx='170' cy='150' rx='6' ry='11' fill='rgba(255,105,180,0.04)' transform='rotate(72,170,160)'/%3E%3Cellipse cx='170' cy='150' rx='6' ry='11' fill='rgba(255,105,180,0.04)' transform='rotate(144,170,160)'/%3E%3Cellipse cx='170' cy='150' rx='6' ry='11' fill='rgba(255,105,180,0.04)' transform='rotate(216,170,160)'/%3E%3Cellipse cx='170' cy='150' rx='6' ry='11' fill='rgba(255,105,180,0.04)' transform='rotate(288,170,160)'/%3E%3C/svg%3E");
    background-size: cover, cover, cover, cover, 200px 200px;
    font-family: 'Lato', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 5, 20, 0.92) !important;
    border-right: 1px solid rgba(255,182,193,0.2);
}
[data-testid="stSidebar"] * { color: #f0d6e8 !important; }

/* Main content glass cards */
.glass-card {
    background: rgba(20, 8, 28, 0.75);
    border: 1px solid rgba(255,182,193,0.2);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 22px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(255,105,180,0.08);
}

/* Hero header */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffb6c1 0%, #da70d6 50%, #ff69b4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.15;
    margin-bottom: 4px;
}
.hero-sub {
    font-family: 'Lato', sans-serif;
    font-size: 1rem;
    color: rgba(240,214,232,0.6);
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0;
}

/* Section headings */
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #ffb6c1;
    border-bottom: 1px solid rgba(255,182,193,0.25);
    padding-bottom: 8px;
    margin-bottom: 18px;
}

/* Metric boxes */
.metric-row { display: flex; gap: 14px; margin-bottom: 14px; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 130px;
    background: rgba(255,105,180,0.1);
    border: 1px solid rgba(255,105,180,0.3);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ff69b4;
}
.metric-label { font-size: 0.78rem; color: rgba(240,214,232,0.6); letter-spacing: 1px; text-transform: uppercase; }

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, rgba(255,105,180,0.15), rgba(147,112,219,0.15));
    border: 2px solid rgba(255,105,180,0.5);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.pred-species {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #ff69b4;
}

/* Streamlit element overrides */
h1,h2,h3,h4,h5,h6,p,li,label,span,div { color: #f0d6e8 !important; }
.stSlider > label, .stSelectbox > label { color: #ffb6c1 !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #ff69b4 !important; font-family: 'Playfair Display', serif !important; }
.stButton > button {
    background: linear-gradient(135deg, #ff69b4, #da70d6) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 10px 36px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(255,105,180,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,105,180,0.6) !important;
}
.stTabs [data-baseweb="tab"] { color: #da70d6 !important; font-family: 'Lato', sans-serif !important; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid #ff69b4 !important; color: #ff69b4 !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(255,182,193,0.2) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & TRAIN MODEL ──────────────────────────────────────
@st.cache_data
def load_and_train():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    knn.fit(X_train_s, y_train); rf.fit(X_train_s, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test_s))
    rf_acc  = accuracy_score(y_test, rf.predict(X_test_s))
    return iris, df, scaler, knn, rf, knn_acc, rf_acc, X_test_s, y_test

iris, df, scaler, knn, rf, knn_acc, rf_acc, X_test_s, y_test = load_and_train()
SPECIES_EMOJI = {'setosa': '🌸', 'versicolor': '💜', 'virginica': '🔵'}
SPECIES_COLOR = {'setosa': '#FF69B4', 'versicolor': '#DA70D6', 'virginica': '#6495ED'}

# ── HERO HEADER ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🌸 Iris Flower Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">CodeAlpha Internship · Task 1 · Gatiksha · CA/DF1/101914</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Predict Species")
    st.markdown("---")
    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.5, 0.1)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    pw = st.slider("Petal Width (cm)",  0.1, 2.5, 0.2, 0.1)
    st.markdown("---")
    model_choice = st.selectbox("Choose Model", ["Random Forest", "K-Nearest Neighbors"])
    predict_btn  = st.button("🔮 Classify Flower")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Classifies Iris flowers into 3 species using ML. Built for CodeAlpha Data Science Internship.")

# ── METRICS ROW ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("🌿 Total Samples", "150")
with col2: st.metric("🌸 Species", "3")
with col3: st.metric("🤖 RF Accuracy", f"{rf_acc*100:.1f}%")
with col4: st.metric("🔵 KNN Accuracy", f"{knn_acc*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 EDA", "📈 Model Performance", "📋 Dataset"])

# ── TAB 1: PREDICT ──────────────────────────────────────────
with tab1:
    if predict_btn:
        model  = rf if model_choice == "Random Forest" else knn
        sample = scaler.transform([[sl, sw, pl, pw]])
        pred   = model.predict(sample)[0]
        proba  = model.predict_proba(sample)[0]
        species_name = iris.target_names[pred]
        emoji = SPECIES_EMOJI[species_name]

        st.markdown(f"""
        <div class="pred-box">
            <div style="font-size:4rem;">{emoji}</div>
            <div style="color:rgba(240,214,232,0.6);font-size:0.85rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">Predicted Species</div>
            <div class="pred-species">Iris {species_name.capitalize()}</div>
            <div style="color:rgba(240,214,232,0.5);font-size:0.9rem;margin-top:8px;">Using {model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Prediction Confidence")
        prob_df = pd.DataFrame({'Species': iris.target_names, 'Confidence': proba})
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        colors = [SPECIES_COLOR[s] for s in iris.target_names]
        bars = ax.barh(prob_df['Species'], prob_df['Confidence'], color=colors, height=0.5)
        for bar, val in zip(bars, prob_df['Confidence']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', color='white', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.15); ax.set_xlabel('Confidence', color='#f0d6e8')
        ax.tick_params(colors='#f0d6e8')
        for spine in ax.spines.values(): spine.set_color((1.0, 0.714, 0.757, 0.2))
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:5rem;">🌸</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.5rem;color:#ffb6c1;margin-top:16px;">
                Adjust the sliders in the sidebar
            </div>
            <div style="color:rgba(240,214,232,0.5);margin-top:8px;">
                Then click "Classify Flower" to predict the species
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: EDA ──────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Feature Distributions by Species")
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        fig.patch.set_facecolor('#0d0d0d')
        palette = {'setosa':'#FF69B4','versicolor':'#DA70D6','virginica':'#6495ED'}
        for i, feat in enumerate(iris.feature_names):
            ax = axes[i//2][i%2]
            ax.set_facecolor('#111')
            for sp, col in palette.items():
                vals = df[df['species']==sp][feat]
                ax.hist(vals, alpha=0.7, label=sp, color=col, bins=12, edgecolor='none')
            ax.set_title(feat, color='#ffb6c1', fontsize=9)
            ax.tick_params(colors='#f0d6e8', labelsize=7)
            for spine in ax.spines.values(): spine.set_color((1.0, 0.714, 0.757, 0.15))
            if i == 0: ax.legend(fontsize=7, facecolor='#111', labelcolor='#f0d6e8')
        fig.suptitle('Feature Distributions', color='#ffb6c1', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.markdown("##### Species Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        counts = df['species'].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=counts.index,
            colors=['#FF69B4','#DA70D6','#6495ED'],
            autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor':'#0d0d0d','linewidth':2})
        for t in texts: t.set_color('#f0d6e8')
        for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
        ax.set_title('Class Balance', color='#ffb6c1', fontsize=12)
        st.pyplot(fig)

    st.markdown("##### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    corr = df[iris.feature_names].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdPu', ax=ax,
                linewidths=0.5, linecolor='#1a0a1a',
                annot_kws={'color':'white','size':10})
    ax.tick_params(colors='#f0d6e8')
    ax.set_title('Feature Correlations', color='#ffb6c1', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

# ── TAB 3: MODEL PERFORMANCE ────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)
    for col, (name, model, acc) in zip([col_a, col_b],
        [("Random Forest", rf, rf_acc), ("KNN", knn, knn_acc)]):
        with col:
            st.markdown(f"##### {name} — {acc*100:.1f}% Accuracy")
            y_pred = model.predict(X_test_s)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0d0d0d')
            ax.set_facecolor('#0d0d0d')
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax,
                        xticklabels=iris.target_names, yticklabels=iris.target_names,
                        linewidths=0.5)
            ax.tick_params(colors='#f0d6e8')
            ax.set_xlabel('Predicted', color='#f0d6e8')
            ax.set_ylabel('Actual', color='#f0d6e8')
            ax.set_title('Confusion Matrix', color='#ffb6c1')
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("##### Feature Importance (Random Forest)")
    imp = pd.DataFrame({'Feature': iris.feature_names,
                        'Importance': rf.feature_importances_}).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
    bars = ax.barh(imp['Feature'], imp['Importance'],
                   color=['#FF69B4','#DA70D6','#BA55D3','#9370DB'])
    for bar, v in zip(bars, imp['Importance']):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{v:.3f}', va='center', color='white', fontsize=10)
    ax.tick_params(colors='#f0d6e8')
    for spine in ax.spines.values(): spine.set_color((1.0, 0.714, 0.757, 0.15))
    ax.set_xlabel('Importance Score', color='#f0d6e8')
    plt.tight_layout(); st.pyplot(fig)

# ── TAB 4: DATASET ──────────────────────────────────────────
with tab4:
    st.markdown("##### Sample Dataset (First 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("##### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:20px;border-top:1px solid rgba(255,182,193,0.15);">
    <span style="color:rgba(240,214,232,0.4);font-size:0.85rem;">
        🌸 CodeAlpha Data Science Internship · Task 1 · Gatiksha · CA/DF1/101914
    </span>
</div>
""", unsafe_allow_html=True)
