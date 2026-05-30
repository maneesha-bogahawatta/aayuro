import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from lifelines import CoxPHFitter
from sklearn.ensemble import GradientBoostingRegressor
import shap

st.set_page_config(page_title="Aayuro", page_icon="🫀", layout="wide")

st.markdown("""
<style>
body, .stApp { background-color: #0F1117; color: #FAFAFA; }
section[data-testid="stSidebar"] { background-color: #1C1F2E; }
section[data-testid="stSidebar"] * { color: #FAFAFA !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label { color: #FAFAFA !important; }
h1,h2,h3,h4,p,span,div,label { color: #FAFAFA; }
.stMarkdown p { color: #FAFAFA; }
.stCaption { color: #AAAAAA !important; }

.hero {
    background: linear-gradient(135deg, #1a1a3e 0%, #0d2137 50%, #0a1628 100%);
    border-radius: 24px;
    padding: 40px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid #2a3a5c;
}
.hero-title {
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(90deg, #60A5FA, #34D399, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 17px;
    color: #94A3B8 !important;
    margin-top: 8px;
}

.score-box {
    background: linear-gradient(135deg, #1E2A4A, #0F1F38);
    border-radius: 20px;
    padding: 32px 20px;
    text-align: center;
    border: 1px solid #2a3a5c;
}
.score-number {
    font-size: 76px;
    font-weight: 900;
    line-height: 1;
}
.score-sub {
    font-size: 14px;
    color: #94A3B8 !important;
    margin-top: 6px;
}

.surv-box {
    background: #1C2333;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid #2a3a5c;
    margin-bottom: 12px;
}
.surv-number {
    font-size: 44px;
    font-weight: 900;
    color: #34D399 !important;
}
.surv-label {
    font-size: 12px;
    color: #94A3B8 !important;
    margin-top: 4px;
}

.rec-card {
    background: #1C2333;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 10px 0;
    border-left: 5px solid #F59E0B;
}
.rec-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #FBBF24 !important;
}
.rec-tip {
    font-size: 13px;
    color: #CBD5E1 !important;
    margin: 3px 0;
}

.metric-card {
    background: #1C2333;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    border: 1px solid #2a3a5c;
}
.metric-val {
    font-size: 22px;
    font-weight: 800;
    color: #F1F5F9 !important;
}
.metric-lbl {
    font-size: 12px;
    color: #94A3B8 !important;
    margin-top: 4px;
}

.disclaimer {
    background: #1C2010;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px;
    color: #D4B483 !important;
    border: 1px solid #3D3010;
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #F1F5F9 !important;
    margin-bottom: 4px;
}
.divider {
    border: none;
    border-top: 1px solid #2a3a5c;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

FEATURES = ["age","is_male","bmi","high_bp","ever_smoked","has_diabetes",
            "is_active","income_ratio","education","sedentary_hrs"]
NAME_MAP = {
    "age":"Age","is_male":"Sex (Male)","bmi":"BMI",
    "high_bp":"High Blood Pressure","ever_smoked":"Ever Smoked",
    "has_diabetes":"Diabetes","is_active":"Physically Active",
    "income_ratio":"Income Level","education":"Education Level",
    "sedentary_hrs":"Sedentary Hours/Day"
}

@st.cache_resource
def load_model():
    df_raw = pd.read_csv("nhanes_model_day4.csv")
    df = df_raw[["follow_months","died"]+FEATURES].copy()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col="follow_months", event_col="died", show_progress=False)
    surv_60 = cph.predict_survival_function(df, times=[60]).loc[60]
    df["risk_score"] = ((1 - surv_60.values)*100).clip(0,100)
    X = df[FEATURES]; y = df["risk_score"].values
    surrogate = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    surrogate.fit(X, y)
    explainer = shap.TreeExplainer(surrogate)
    return cph, surrogate, explainer

def risk_category(score):
    if score < 2:    return "Low",      "#34D399", "🟢"
    elif score < 8:  return "Moderate", "#FBBF24", "🟡"
    elif score < 20: return "High",     "#F87171", "🔴"
    else:            return "Very High","#EF4444", "🚨"

def get_survival(cph, person, months):
    return float(cph.predict_survival_function(
        pd.DataFrame([person]), times=[months]).loc[months].values[0])

RECS = {
    "high_bp":      (lambda v: v==1,  "⚠️ High Blood Pressure", "#F87171",
                     ["Reduce sodium to <2,300mg/day",
                      "Exercise 30 mins/day — lowers BP by 5–8 mmHg",
                      "Consult a doctor if BP stays ≥ 140"]),
    "has_diabetes": (lambda v: v>=1,  "⚠️ Diabetes / Pre-diabetes", "#C084FC",
                     ["Monitor blood glucose and HbA1c regularly",
                      "Reduce refined carbs and added sugars",
                      "Even 5–10% weight loss significantly improves outcomes"]),
    "is_active":    (lambda v: v==0,  "🏃 Low Physical Activity", "#FB923C",
                     ["Start with 10-min walks — build to 150 mins/week",
                      "Activity reduces mortality risk by ~24% in this dataset",
                      "Any movement counts — cycling, gardening, dancing"]),
    "sedentary_hrs":(lambda v: v>8,   "🪑 High Sedentary Time", "#A78BFA",
                     ["Stand and move for 5 mins every hour",
                      "Set a phone reminder every 60 minutes",
                      "Breaking up sitting time reduces mortality risk"]),
    "ever_smoked":  (lambda v: v==1,  "🚬 Smoking History", "#94A3B8",
                     ["Quitting at any age improves outcomes",
                      "Nicotine replacement doubles quit success rates",
                      "Talk to a doctor about cessation support"]),
    "income_ratio": (lambda v: v<1.5, "💰 Financial Stress", "#60A5FA",
                     ["Seek community health clinics for low-cost care",
                      "Ask about generic medication equivalents",
                      "Free preventive screenings are available in most areas"]),
}

# ── HERO ──────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-title'>🫀 Aayuro</div>
  <div class='hero-sub'>See your health risk clearly · Powered by real epidemiological data · NHANES 2013–2014</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
ℹ️ <strong>Educational tool only.</strong> Based on US NHANES survey data (~5,000 adults,
7-year follow-up). NOT a medical device. Does not replace professional medical advice.
Scores reflect statistical population-level risk, not individual fate.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

with st.spinner("🔬 Loading Aayuro model..."):
    cph, surrogate, explainer = load_model()

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0 20px 0;'>
  <span style='font-size:28px;'>🔬</span>
  <div style='font-size:20px; font-weight:800; color:#60A5FA;'>Aayuro</div>
  <div style='font-size:12px; color:#94A3B8;'>Your Health Profile</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### 👤 Demographics")
age      = st.sidebar.slider("Age (years)", 18, 85, 45)
sex      = st.sidebar.radio("Sex", ["Female","Male"])
is_male  = 1 if sex=="Male" else 0

st.sidebar.markdown("#### 🏃 Body & Activity")
bmi      = st.sidebar.slider("BMI", 15.0, 50.0, 25.0, step=0.5,
                              help="BMI = weight(kg) ÷ height(m)²")
active   = st.sidebar.radio("Vigorous physical activity?", ["Yes","No"])
is_active= 1 if active=="Yes" else 0
sedentary= st.sidebar.slider("Sedentary hours per day", 0, 18, 6)

st.sidebar.markdown("#### 🩺 Health Conditions")
bp_status= st.sidebar.radio("Blood pressure",
                             ["Normal (< 120)","Elevated (120–139)","High (≥ 140)"])
high_bp  = 1 if bp_status=="High (≥ 140)" else 0
diabetes = st.sidebar.selectbox("Diabetes status",
                                 ["No","Pre-diabetes / Borderline","Yes — diagnosed"])
has_diabetes = {"No":0,"Pre-diabetes / Borderline":0.5,"Yes — diagnosed":1}[diabetes]

st.sidebar.markdown("#### 🚬 Lifestyle")
smoked   = st.sidebar.radio("Ever smoked?", ["No","Yes"])
ever_smoked = 1 if smoked=="Yes" else 0

st.sidebar.markdown("#### 💼 Socioeconomic")
income_label = st.sidebar.select_slider("Income level",
    options=["Very low","Low","Middle","Comfortable","High"], value="Middle")
income_ratio = {"Very low":0.5,"Low":1.2,"Middle":2.5,"Comfortable":3.8,"High":5.0}[income_label]
edu_label= st.sidebar.selectbox("Education level",
    ["Less than high school","Some high school","High school graduate",
     "Some college","College graduate or above"])
education= {"Less than high school":1,"Some high school":2,"High school graduate":3,
            "Some college":4,"College graduate or above":5}[edu_label]

# ── PREDICT ───────────────────────────────────────────────
person = {"age":age,"is_male":is_male,"bmi":bmi,"high_bp":high_bp,
          "ever_smoked":ever_smoked,"has_diabetes":has_diabetes,
          "is_active":is_active,"income_ratio":income_ratio,
          "education":education,"sedentary_hrs":sedentary}

surv_5yr  = get_survival(cph, person, 60)
surv_10yr = get_survival(cph, person, 120)
risk_5yr  = (1-surv_5yr)*100
cat, color, emoji = risk_category(risk_5yr)

person_df = pd.DataFrame([person])[FEATURES]
shap_vals = explainer.shap_values(person_df)[0]

# ── SCORE ROW ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.markdown(f"""
    <div class='score-box'>
        <div style='font-size:13px;color:#94A3B8;letter-spacing:2px;margin-bottom:8px;'>
            5-YEAR RISK SCORE
        </div>
        <div class='score-number' style='color:{color};'>{risk_5yr:.1f}</div>
        <div class='score-sub'>out of 100</div>
        <div style='background:{color};color:#000;display:inline-block;padding:7px 24px;
                    border-radius:20px;font-size:17px;font-weight:800;margin-top:14px;'>
            {emoji} {cat} Risk
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='surv-box'>
        <div style='font-size:12px;color:#94A3B8;letter-spacing:1px;'>5-YEAR SURVIVAL</div>
        <div class='surv-number'>{surv_5yr*100:.1f}%</div>
        <div class='surv-label'>probability of surviving 5 years</div>
    </div>
    <div class='surv-box'>
        <div style='font-size:12px;color:#94A3B8;letter-spacing:1px;'>10-YEAR SURVIVAL</div>
        <div class='surv-number' style='font-size:36px;'>{surv_10yr*100:.1f}%</div>
        <div class='surv-label'>probability of surviving 10 years</div>
    </div>""", unsafe_allow_html=True)

with col3:
    fig, ax = plt.subplots(figsize=(5,2.5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    ax.barh(0, 100, height=0.4, color="#1C2333", zorder=1)
    ax.barh(0, min(risk_5yr,100), height=0.4, color=color, zorder=2, alpha=0.9)
    for x in [2,8,20]:
        ax.axvline(x, color="#0F1117", linewidth=2.5, zorder=3)
    ax.text(1,  -0.38,"Low",    fontsize=8,ha="center",color="#34D399")
    ax.text(5,  -0.38,"Mod",    fontsize=8,ha="center",color="#FBBF24")
    ax.text(14, -0.38,"High",   fontsize=8,ha="center",color="#F87171")
    ax.text(60, -0.38,"V.High", fontsize=8,ha="center",color="#EF4444")
    ax.text(min(risk_5yr+2,95), 0.28, f"{risk_5yr:.1f}",
            fontsize=13,ha="center",fontweight="bold",color=color)
    ax.set_xlim(0,100); ax.set_ylim(-0.6,0.6); ax.axis("off")
    ax.set_title("Risk Gauge", fontsize=11, pad=8, color="#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── SHAP + RECS ───────────────────────────────────────────
col_a, col_b = st.columns([1,1])

with col_a:
    st.markdown("<div class='section-title'>🔍 What's Driving Your Score?</div>", unsafe_allow_html=True)
    st.caption("Each bar shows how much that factor raises or lowers your personal risk.")

    contrib_df = pd.DataFrame({
        "label":[NAME_MAP[f] for f in FEATURES],
        "shap": shap_vals,
        "value":[person[f] for f in FEATURES]
    }).sort_values("shap", key=abs, ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    fig2.patch.set_facecolor("#0F1117")
    ax2.set_facecolor("#0F1117")
    colors_bar = ["#F87171" if s>0 else "#34D399" for s in contrib_df["shap"]]
    bars = ax2.barh(contrib_df["label"], contrib_df["shap"],
                    color=colors_bar, alpha=0.85, height=0.6)
    for bar, val in zip(bars, contrib_df["shap"]):
        x = bar.get_width()
        ax2.text(x+(0.03 if x>=0 else -0.03),
                 bar.get_y()+bar.get_height()/2,
                 f"{val:+.2f}", va="center",
                 ha="left" if x>=0 else "right",
                 fontsize=9, color="#FAFAFA")
    ax2.axvline(0, color="#FAFAFA", linewidth=1, linestyle="--", alpha=0.4)
    ax2.set_xlabel("Impact on risk score\n(+ raises risk · − lowers risk)",
                   fontsize=10, color="#94A3B8")
    ax2.tick_params(colors="#FAFAFA")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#2a3a5c")
    ax2.spines["bottom"].set_color("#2a3a5c")
    ax2.yaxis.label.set_color("#FAFAFA")
    ax2.xaxis.label.set_color("#94A3B8")
    for tick in ax2.get_yticklabels():
        tick.set_color("#FAFAFA")
    for tick in ax2.get_xticklabels():
        tick.set_color("#94A3B8")
    ax2.grid(True, alpha=0.1, linestyle="--", axis="x", color="#FAFAFA")
    ax2.legend(handles=[
        mpatches.Patch(color="#F87171",alpha=0.85,label="Raises risk"),
        mpatches.Patch(color="#34D399",alpha=0.85,label="Lowers risk")
    ], fontsize=9, loc="lower right",
       facecolor="#1C2333", edgecolor="#2a3a5c", labelcolor="#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col_b:
    st.markdown("<div class='section-title'>💡 Your Personalised Recommendations</div>", unsafe_allow_html=True)
    st.caption("Based on your modifiable risk factors — things you can actually change.")
    recs_shown = 0
    for feat, (cond, title, col_hex, tips) in RECS.items():
        if cond(person[feat]):
            tips_html = "".join([f"<div class='rec-tip'>• {t}</div>" for t in tips])
            st.markdown(f"""
            <div class='rec-card' style='border-left-color:{col_hex};'>
                <div class='rec-title' style='color:{col_hex};'>{title}</div>
                {tips_html}
            </div>""", unsafe_allow_html=True)
            recs_shown += 1
    if recs_shown == 0:
        st.success("✅ No major modifiable risk factors detected! Keep up your healthy habits.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── PROFILE SUMMARY ───────────────────────────────────────
st.markdown("<div class='section-title'>📊 Your Profile Summary</div>", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
cards = [
    ("🎂", str(age), "Age"),
    ("⚖️", str(bmi), "BMI"),
    ("🏃", "Active ✅" if is_active else "Inactive ❌",  "Activity",
     "#34D399" if is_active else "#F87171"),
    ("💓", "High BP ⚠️" if high_bp else "Normal ✅", "Blood Pressure",
     "#F87171" if high_bp else "#34D399"),
]
for col, (icon, val, lbl, *extra) in zip([c1,c2,c3,c4], cards):
    vc = extra[0] if extra else "#F1F5F9"
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:28px;'>{icon}</div>
            <div class='metric-val' style='color:{vc};'>{val}</div>
            <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div class='disclaimer'>
<strong>📌 Limitations:</strong> Trained on US adults (NHANES 2013–2014).
Risk estimates may not perfectly apply to other populations including Sri Lanka.
This predicts statistical population-level risk, not individual fate. C-index = 0.83.
Always consult a qualified healthcare professional for medical decisions.
</div>
<div style='text-align:center;padding:20px 0 10px 0;color:#4a5568;font-size:13px;'>
Aayuro · NHANES Epidemiological Data · Cox Proportional Hazards + SHAP ·
Educational purposes only
</div>
""", unsafe_allow_html=True)
