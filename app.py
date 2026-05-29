import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; font-weight: 700; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin-top: 0.5rem; }

    .risk-low    { background: linear-gradient(135deg,#0a4a2a,#1a6b3c); color:#7fff7f;
                   padding:1.5rem; border-radius:12px; text-align:center; font-size:1.5rem; font-weight:700; }
    .risk-medium { background: linear-gradient(135deg,#4a3a00,#6b5500); color:#ffd700;
                   padding:1.5rem; border-radius:12px; text-align:center; font-size:1.5rem; font-weight:700; }
    .risk-high   { background: linear-gradient(135deg,#4a0a0a,#6b1a1a); color:#ff6b6b;
                   padding:1.5rem; border-radius:12px; text-align:center; font-size:1.5rem; font-weight:700; }

    .tip-card {
        background: #1a1a2e; border-left: 4px solid #e94560;
        padding: 1rem 1.5rem; border-radius: 8px; margin: 0.5rem 0; color: #a8b2d8;
    }
    .metric-card {
        background: linear-gradient(135deg,#1a1a2e,#16213e);
        border: 1px solid #0f3460; border-radius: 12px; padding: 1.2rem;
        text-align: center; color: white;
    }
    .metric-card h3 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; font-size: 0.85rem; margin: 0; }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"]    label { color: #a8b2d8 !important; font-weight: 500; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white; border: none; padding: 0.8rem 3rem;
        border-radius: 50px; font-size: 1.1rem; font-weight: 600;
        width: 100%; cursor: pointer; transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(233,69,96,0.4); }

    .sidebar .sidebar-content { background: #0f3460; }
    .step-badge {
        background: #e94560; color: white; border-radius: 50%; width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.9rem; margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Generate Synthetic Training Data ──────────────────────────────────────────
@st.cache_data
def generate_lifestyle_data(n=2000):
    np.random.seed(42)
    data = {
        "sleep_hours":       np.random.uniform(4, 10, n),
        "physical_activity": np.random.choice([0, 1, 2, 3], n),   # 0=None,1=Light,2=Moderate,3=Heavy
        "diet_type":         np.random.choice([0, 1, 2], n),       # 0=Unhealthy,1=Average,2=Healthy
        "bmi":               np.random.uniform(16, 40, n),
        "stress_level":      np.random.randint(1, 11, n),
        "smoking":           np.random.choice([0, 1], n),
        "alcohol":           np.random.choice([0, 1, 2], n),       # 0=None,1=Occasional,2=Regular
        "family_history":    np.random.choice([0, 1], n),
        "age":               np.random.randint(18, 75, n),
        "water_intake":      np.random.uniform(0.5, 4, n),
    }
    df = pd.DataFrame(data)

    # Score-based risk label
    score = (
        (df["bmi"] > 25).astype(int) * 2 +
        (df["bmi"] > 30).astype(int) * 2 +
        (df["stress_level"] > 6).astype(int) * 2 +
        (df["sleep_hours"] < 6).astype(int) * 2 +
        df["smoking"] * 3 +
        (df["alcohol"] == 2).astype(int) * 2 +
        df["family_history"] * 2 +
        (df["physical_activity"] == 0).astype(int) * 2 +
        (df["diet_type"] == 0).astype(int) * 2 +
        (df["age"] > 50).astype(int) * 1 +
        (df["water_intake"] < 1.5).astype(int) * 1
    )
    df["risk_label"] = pd.cut(score, bins=[-1, 5, 10, 100],
                              labels=[0, 1, 2]).astype(int)   # 0=Low,1=Medium,2=High
    return df


@st.cache_data
def generate_symptom_data(n=2000):
    np.random.seed(99)
    diseases = ["Diabetes", "Hypertension", "Obesity", "Heart Disease", "Asthma"]
    templates = {
        "Diabetes":       [1,0,1,0,0,1,1,0,0,1],
        "Hypertension":   [0,1,0,1,0,0,1,1,0,0],
        "Obesity":        [0,0,1,0,1,0,1,0,1,1],
        "Heart Disease":  [1,1,0,1,0,1,0,1,0,0],
        "Asthma":         [0,0,0,0,1,0,0,0,0,0],
    }
    symptoms = ["fatigue","headache","frequent_urination","chest_pain",
                "shortness_of_breath","blurred_vision","weight_gain",
                "dizziness","joint_pain","increased_thirst"]
    rows, labels = [], []
    for _ in range(n):
        d = np.random.choice(diseases)
        base = np.array(templates[d])
        noise = np.random.randint(0, 2, len(base)) * 0.3
        row = np.clip(base + noise, 0, 1).round().astype(int)
        rows.append(row)
        labels.append(diseases.index(d))
    df = pd.DataFrame(rows, columns=symptoms)
    df["disease"] = labels
    return df, symptoms, diseases


# ─── Train Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_lifestyle_models():
    df = generate_lifestyle_data()
    features = ["sleep_hours","physical_activity","diet_type","bmi",
                "stress_level","smoking","alcohol","family_history","age","water_intake"]
    X = df[features]
    y = df["risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":     DecisionTreeClassifier(random_state=42),
        "k-NN":              KNeighborsClassifier(n_neighbors=5),
    }
    trained, accuracies = {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        accuracies[name] = round(accuracy_score(y_test, m.predict(X_test)) * 100, 2)

    best = max(accuracies, key=accuracies.get)
    return trained, accuracies, best, features


@st.cache_resource
def train_disease_models():
    df, symptoms, diseases = generate_symptom_data()
    X = df[symptoms]
    y = df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest":   RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":   DecisionTreeClassifier(random_state=42),
        "k-NN":            KNeighborsClassifier(n_neighbors=5),
    }
    trained, accuracies = {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        accuracies[name] = round(accuracy_score(y_test, m.predict(X_test)) * 100, 2)

    best = max(accuracies, key=accuracies.get)
    return trained, accuracies, best, symptoms, diseases


# ─── Recommendations ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Low": [
        "✅ Great job! Maintain your current healthy lifestyle.",
        "💧 Keep drinking 2–3 litres of water daily.",
        "🏃 Continue regular physical activity (30 min/day).",
        "😴 Maintain 7–8 hours of quality sleep.",
        "🥗 Keep eating a balanced diet with fruits & vegetables.",
    ],
    "Medium": [
        "⚠️ Your risk is moderate — lifestyle changes needed now!",
        "🏋️ Increase physical activity to at least 150 min/week.",
        "🍔 Reduce junk food, processed sugar & saturated fats.",
        "🧘 Practice stress management (yoga, meditation, deep breathing).",
        "💊 Get regular health check-ups every 6 months.",
        "🚭 If you smoke or drink, start reducing immediately.",
        "💧 Drink at least 2.5 litres of water daily.",
    ],
    "High": [
        "🚨 HIGH RISK DETECTED — Please consult a doctor immediately!",
        "🏥 Schedule a comprehensive health check-up this week.",
        "🍬 Strictly control sugar, salt, and fat intake.",
        "🚭 Stop smoking and alcohol consumption completely.",
        "💊 Follow prescribed medications if any.",
        "🏋️ Start light exercise under medical supervision.",
        "👨‍👩‍👧 Inform family members — hereditary risks may affect them too.",
        "📱 Use a health monitoring app to track vitals daily.",
    ],
}

DISEASE_TIPS = {
    "Diabetes":      ["Monitor blood sugar regularly","Reduce refined carbs & sugars","Exercise daily for 30 min","Maintain healthy BMI"],
    "Hypertension":  ["Reduce salt intake","Avoid caffeine & alcohol","Practice deep breathing exercises","Monitor BP regularly"],
    "Obesity":       ["Create a calorie deficit diet","Walk 10,000 steps daily","Avoid late-night eating","Stay hydrated"],
    "Heart Disease": ["Take prescribed medication strictly","Avoid physical overexertion","Eat heart-healthy foods (oats, fish)","Quit smoking immediately"],
    "Asthma":        ["Avoid allergens & pollutants","Always carry inhaler","Practice breathing exercises","Avoid cold air exposure"],
}


# ─── Load Models ───────────────────────────────────────────────────────────────
lf_models, lf_acc, lf_best, lf_features = train_lifestyle_models()
ds_models, ds_acc, ds_best, symptom_cols, disease_names = train_disease_models()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Navigation")
    page = st.radio("", ["🏠 Home", "📊 Phase 1: Risk Prediction",
                         "🔬 Phase 2: Disease Prediction", "📈 Model Performance"])
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <small style='color:#a8b2d8'>
    This system is for <b>educational purposes only</b>.<br>
    It does <b>NOT replace</b> professional medical advice.<br><br>
    <b>Team:</b> Agrima | Parisha | Aditi<br>
    <b>JIIT Noida | CSE 2nd Year</b>
    </small>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🩺 AI Health Predictor</h1>
        <p>AI-Powered Lifestyle-Based Disease Risk & Early Disease Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, emoji, val, label in zip(
        [c1,c2,c3,c4],
        ["🤖","📊","🏥","✅"],
        ["4","2000+","5","4"],
        ["ML Models","Training Samples","Diseases Covered","Risk Levels"]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <p style='font-size:2rem;margin:0'>{emoji}</p>
            <h3>{val}</h3>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔄 How It Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### <span class='step-badge'>1</span> Enter Lifestyle Data
        Fill in your daily habits — sleep, diet, BMI, stress, activity level etc.
        
        ### <span class='step-badge'>2</span> Get Risk Score
        Our ML models analyze your lifestyle and predict your disease risk as **Low / Medium / High**.
        
        ### <span class='step-badge'>3</span> Symptom Analysis *(if needed)*
        If your risk is Medium or High, enter your symptoms for specific disease prediction.
        
        ### <span class='step-badge'>4</span> Personalized Recommendations
        Get AI-powered health tips tailored to your risk level and predicted condition.
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        ### 🎯 Diseases Covered
        | Disease | Key Risk Factors |
        |---|---|
        | 🩸 Diabetes | Sugar, BMI, Fatigue |
        | ❤️ Heart Disease | Stress, Smoking, Chest Pain |
        | 📈 Hypertension | Salt, Alcohol, Headache |
        | ⚖️ Obesity | Diet, Activity, BMI |
        | 🌬️ Asthma | Allergens, Breathing Issues |
        
        ### 🔬 Algorithms Used
        - Random Forest ✅ *(Best Performer)*
        - Logistic Regression
        - Decision Tree
        - k-Nearest Neighbors
        """)

    st.info("👈 **Start by clicking 'Phase 1: Risk Prediction' in the sidebar!**")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PHASE 1
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Phase 1: Risk Prediction":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Phase 1: Lifestyle Risk Prediction</h1>
        <p>Enter your lifestyle details to get your disease risk score</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("lifestyle_form"):
        st.markdown("### 👤 Personal Information")
        c1, c2 = st.columns(2)
        with c1:
            age  = st.slider("🎂 Age", 18, 80, 25)
            bmi  = st.slider("⚖️ BMI (Body Mass Index)", 15.0, 45.0, 22.0, 0.1)
        with c2:
            name   = st.text_input("📝 Your Name (optional)", "")
            gender = st.selectbox("🧑 Gender", ["Male", "Female", "Other"])

        st.markdown("### 🌙 Daily Habits")
        c1, c2, c3 = st.columns(3)
        with c1:
            sleep   = st.slider("😴 Sleep (hours/day)", 3.0, 12.0, 7.0, 0.5)
            stress  = st.slider("😰 Stress Level (1–10)", 1, 10, 4)
        with c2:
            activity_lbl = st.selectbox("🏃 Physical Activity", ["None","Light","Moderate","Heavy"])
            activity     = ["None","Light","Moderate","Heavy"].index(activity_lbl)
            water = st.slider("💧 Water Intake (litres/day)", 0.5, 5.0, 2.0, 0.25)
        with c3:
            diet_lbl = st.selectbox("🥗 Diet Type", ["Unhealthy","Average","Healthy"])
            diet     = ["Unhealthy","Average","Healthy"].index(diet_lbl)

        st.markdown("### ⚠️ Risk Factors")
        c1, c2, c3 = st.columns(3)
        with c1:
            smoking = st.checkbox("🚬 Do you smoke?")
        with c2:
            alcohol_lbl = st.selectbox("🍺 Alcohol Consumption", ["None","Occasional","Regular"])
            alcohol     = ["None","Occasional","Regular"].index(alcohol_lbl)
        with c3:
            family = st.checkbox("👨‍👩‍👧 Family history of chronic disease?")

        model_choice = st.selectbox("🤖 Choose ML Algorithm",
                                    list(lf_models.keys()), index=0)
        submitted = st.form_submit_button("🔍 Predict My Risk")

    if submitted:
        inp = np.array([[sleep, activity, diet, bmi, stress,
                         int(smoking), alcohol, int(family), age, water]])
        model  = lf_models[model_choice]
        pred   = model.predict(inp)[0]
        proba  = model.predict_proba(inp)[0]
        labels = ["Low", "Medium", "High"]
        risk   = labels[pred]

        st.markdown("---")
        greeting = f"**{name},** your" if name else "Your"
        st.markdown(f"### {greeting} Result")

        colors = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}
        emojis = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}
        st.markdown(f"<div class='{colors[risk]}'>{emojis[risk]} Disease Risk: {risk.upper()}</div>",
                    unsafe_allow_html=True)

        st.markdown("#### 📊 Confidence Scores")
        cols = st.columns(3)
        for i, (lbl, prob) in enumerate(zip(labels, proba)):
            cols[i].metric(f"{lbl} Risk", f"{prob*100:.1f}%")

        # Bar chart
        fig, ax = plt.subplots(figsize=(6,3), facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        bar_colors = ["#7fff7f","#ffd700","#ff6b6b"]
        bars = ax.bar(labels, proba*100, color=bar_colors, width=0.5, edgecolor="none")
        for bar, p in zip(bars, proba*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{p:.1f}%", ha="center", color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("Confidence %", color="white")
        ax.tick_params(colors="white")
        ax.set_ylim(0, 110)
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("#### 💡 Personalized Recommendations")
        for tip in RECOMMENDATIONS[risk]:
            st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

        if risk in ["Medium", "High"]:
            st.warning("👉 **Go to Phase 2: Disease Prediction** in the sidebar for detailed symptom analysis!")

        st.session_state["risk"] = risk


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PHASE 2
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Phase 2: Disease Prediction":
    st.markdown("""
    <div class='main-header'>
        <h1>🔬 Phase 2: Disease Prediction</h1>
        <p>Select your current symptoms for specific disease prediction</p>
    </div>
    """, unsafe_allow_html=True)

    risk = st.session_state.get("risk", None)
    if risk == "Low":
        st.success("✅ Your Phase 1 risk was **Low**. Disease prediction is optional.")
    elif risk is None:
        st.info("ℹ️ Complete Phase 1 first for best results. You can still use this page independently.")

    sym_labels = {
        "fatigue":            "😴 Fatigue / Constant tiredness",
        "headache":           "🤕 Frequent Headaches",
        "frequent_urination": "🚽 Frequent Urination",
        "chest_pain":         "💔 Chest Pain / Tightness",
        "shortness_of_breath":"😮‍💨 Shortness of Breath",
        "blurred_vision":     "👁️ Blurred Vision",
        "weight_gain":        "⚖️ Unexplained Weight Gain",
        "dizziness":          "💫 Dizziness / Lightheadedness",
        "joint_pain":         "🦴 Joint Pain / Stiffness",
        "increased_thirst":   "💧 Increased Thirst",
    }

    with st.form("symptom_form"):
        st.markdown("### 🤒 Select Symptoms You Are Experiencing")
        selected = {}
        c1, c2 = st.columns(2)
        for i, (key, label) in enumerate(sym_labels.items()):
            col = c1 if i % 2 == 0 else c2
            selected[key] = int(col.checkbox(label))

        model_choice = st.selectbox("🤖 Algorithm", list(ds_models.keys()), index=0)
        submitted2 = st.form_submit_button("🔬 Predict Disease")

    if submitted2:
        inp = np.array([[selected[s] for s in symptom_cols]])
        model   = ds_models[model_choice]
        pred    = model.predict(inp)[0]
        proba   = model.predict_proba(inp)[0]
        disease = disease_names[pred]

        st.markdown("---")
        st.markdown(f"### 🏥 Predicted Condition: **{disease}**")
        st.markdown(f"**Confidence:** `{proba[pred]*100:.1f}%`")

        # Probability chart
        fig, ax = plt.subplots(figsize=(8,3), facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        y_pos  = range(len(disease_names))
        colors = ["#e94560" if i==pred else "#1a6b8a" for i in range(len(disease_names))]
        ax.barh(list(y_pos), proba*100, color=colors, edgecolor="none")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(disease_names, color="white")
        ax.set_xlabel("Confidence %", color="white")
        ax.tick_params(axis="x", colors="white")
        ax.set_xlim(0, 115)
        for i, p in enumerate(proba*100):
            ax.text(p+1, i, f"{p:.1f}%", va="center", color="white", fontsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("#### 💊 Disease-Specific Recommendations")
        for tip in DISEASE_TIPS.get(disease, []):
            st.markdown(f"<div class='tip-card'>💡 {tip}</div>", unsafe_allow_html=True)

        st.error("⚕️ **DISCLAIMER:** This is an AI prediction for awareness only. "
                 "Please consult a qualified doctor for proper diagnosis and treatment.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Model Performance Comparison</h1>
        <p>Accuracy comparison of all trained ML algorithms</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏥 Phase 1 — Lifestyle Risk Models")
    df_lf = pd.DataFrame({"Algorithm": list(lf_acc.keys()),
                           "Accuracy (%)": list(lf_acc.values())}).sort_values("Accuracy (%)", ascending=False)
    df_lf["Best?"] = df_lf["Algorithm"].apply(lambda x: "⭐ Best" if x == lf_best else "")
    st.dataframe(df_lf.reset_index(drop=True), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8,4), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    cols = ["#e94560" if a == lf_best else "#1a6b8a" for a in df_lf["Algorithm"]]
    ax.bar(df_lf["Algorithm"], df_lf["Accuracy (%)"], color=cols, edgecolor="none")
    ax.set_ylim(50, 105)
    ax.set_ylabel("Accuracy %", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_visible(False)
    for i, v in enumerate(df_lf["Accuracy (%)"]):
        ax.text(i, v+0.5, f"{v}%", ha="center", color="white", fontweight="bold")
    ax.set_title("Phase 1 — Risk Prediction Accuracy", color="white", fontsize=13)
    st.pyplot(fig); plt.close()

    st.markdown("### 🔬 Phase 2 — Disease Prediction Models")
    df_ds = pd.DataFrame({"Algorithm": list(ds_acc.keys()),
                           "Accuracy (%)": list(ds_acc.values())}).sort_values("Accuracy (%)", ascending=False)
    df_ds["Best?"] = df_ds["Algorithm"].apply(lambda x: "⭐ Best" if x == ds_best else "")
    st.dataframe(df_ds.reset_index(drop=True), use_container_width=True, hide_index=True)

    fig2, ax2 = plt.subplots(figsize=(7,4), facecolor="#0f0f1a")
    ax2.set_facecolor("#0f0f1a")
    cols2 = ["#e94560" if a == ds_best else "#1a6b8a" for a in df_ds["Algorithm"]]
    ax2.bar(df_ds["Algorithm"], df_ds["Accuracy (%)"], color=cols2, edgecolor="none")
    ax2.set_ylim(50, 105)
    ax2.set_ylabel("Accuracy %", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_visible(False)
    for i, v in enumerate(df_ds["Accuracy (%)"]):
        ax2.text(i, v+0.5, f"{v}%", ha="center", color="white", fontweight="bold")
    ax2.set_title("Phase 2 — Disease Prediction Accuracy", color="white", fontsize=13)
    st.pyplot(fig2); plt.close()

    st.markdown("### 📝 Interpretation")
    st.markdown(f"""
    - **Best Phase 1 Model:** `{lf_best}` with **{lf_acc[lf_best]}%** accuracy
    - **Best Phase 2 Model:** `{ds_best}` with **{ds_acc[ds_best]}%** accuracy
    - Random Forest generally performs best due to its ensemble nature
    - All models trained on synthetic data — real dataset will improve accuracy further
    """)
