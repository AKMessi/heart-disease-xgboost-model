import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "framingham_heart_study.csv")
FEATURES = [
    "age",
    "sex",
    "smoking_current",
    "cigs_per_day",
    "bp_meds",
    "stroke",
    "hypertension",
    "diabetes",
    "cholesterol",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "heart_rate",
    "glucose",
]
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "smoking_current": "Currently smoking",
    "cigs_per_day": "Cigarettes per day",
    "bp_meds": "On BP medication",
    "stroke": "Prior stroke",
    "hypertension": "Hypertensive",
    "diabetes": "Diabetic",
    "cholesterol": "Total cholesterol (mg/dL)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "bmi": "BMI",
    "heart_rate": "Resting heart rate",
    "glucose": "Fasting glucose (mg/dL)",
}


@st.cache_resource
def load_models():
    df = pd.read_csv(DATA_PATH)
    df["time"] = 10
    df["event"] = df["TenYearCHD"]
    df = df.rename(
        columns={
            "male": "sex",
            "currentSmoker": "smoking_current",
            "cigsPerDay": "cigs_per_day",
            "BPMeds": "bp_meds",
            "prevalentStroke": "stroke",
            "prevalentHyp": "hypertension",
            "totChol": "cholesterol",
            "sysBP": "systolic_bp",
            "diaBP": "diastolic_bp",
            "BMI": "bmi",
            "heartRate": "heart_rate",
        }
    )

    for col in FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    scaler = StandardScaler()
    cox_df = df[FEATURES + ["time", "event"]].copy()
    cox_df_scaled = cox_df.copy()
    cox_df_scaled[FEATURES] = scaler.fit_transform(cox_df[FEATURES])

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        cox_df_scaled,
        duration_col="time",
        event_col="event",
        fit_options={"step_size": 0.5},
    )

    y_xgb = np.where(df["event"] == 1, df["time"], -df["time"]).astype(float)
    dtrain = xgb.DMatrix(df[FEATURES], label=y_xgb, feature_names=FEATURES)
    xgb_params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "max_depth": 4,
        "learning_rate": 0.058,
        "min_child_weight": 33,
        "subsample": 0.70,
        "colsample_bytree": 0.61,
        "reg_lambda": 3.53,
        "reg_alpha": 1.89,
        "seed": 42,
        "verbosity": 0,
    }
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=46,
        verbose_eval=False,
    )

    pop_means = df[FEATURES].mean()

    return cph, scaler, xgb_model, FEATURES, pop_means, df


def get_cox_risk(cph, scaler, patient_dict, feature_names):
    pt_df = pd.DataFrame([patient_dict])[feature_names]
    pt_scaled = pd.DataFrame(scaler.transform(pt_df), columns=feature_names)
    pt_cox = pd.concat(
        [pt_scaled, pd.DataFrame([{"time": 10, "event": 0}])],
        axis=1,
    )
    surv_fn = cph.predict_survival_function(pt_cox)
    return float(1 - surv_fn.iloc[-1].values[0])


def get_frs_risk(age, sex, cholesterol, systolic_bp, smoking, diabetes, bp_meds):
    if sex == 1:  # Male
        if age < 35:
            age_pts = -9
        elif age < 40:
            age_pts = -4
        elif age < 45:
            age_pts = 0
        elif age < 50:
            age_pts = 3
        elif age < 55:
            age_pts = 6
        elif age < 60:
            age_pts = 8
        elif age < 65:
            age_pts = 10
        elif age < 70:
            age_pts = 11
        else:
            age_pts = 12

        if cholesterol < 160:
            chol_pts = -3
        elif cholesterol < 200:
            chol_pts = 0
        elif cholesterol < 240:
            chol_pts = 1
        elif cholesterol < 280:
            chol_pts = 2
        else:
            chol_pts = 3

        if systolic_bp < 120:
            sbp_pts = 0
        elif systolic_bp < 130:
            sbp_pts = 1 if not bp_meds else 2
        elif systolic_bp < 140:
            sbp_pts = 1 if not bp_meds else 2
        elif systolic_bp < 160:
            sbp_pts = 1 if not bp_meds else 2
        else:
            sbp_pts = 2 if not bp_meds else 3

        smk_pts = 4 if smoking else 0
        diab_pts = 3 if diabetes else 0
        total = age_pts + chol_pts + sbp_pts + smk_pts + diab_pts

        table = {
            -3: 0.01,
            -2: 0.01,
            -1: 0.01,
            0: 0.01,
            1: 0.01,
            2: 0.02,
            3: 0.02,
            4: 0.02,
            5: 0.03,
            6: 0.03,
            7: 0.04,
            8: 0.05,
            9: 0.06,
            10: 0.08,
            11: 0.10,
            12: 0.12,
            13: 0.16,
            14: 0.20,
            15: 0.25,
            16: 0.31,
        }
        return table.get(max(-3, min(16, total)), 0.31)

    if age < 35:
        age_pts = -7
    elif age < 40:
        age_pts = -3
    elif age < 45:
        age_pts = 0
    elif age < 50:
        age_pts = 3
    elif age < 55:
        age_pts = 6
    elif age < 60:
        age_pts = 8
    elif age < 65:
        age_pts = 10
    elif age < 70:
        age_pts = 12
    else:
        age_pts = 14

    if cholesterol < 160:
        chol_pts = -2
    elif cholesterol < 200:
        chol_pts = 0
    elif cholesterol < 240:
        chol_pts = 1
    elif cholesterol < 280:
        chol_pts = 1
    else:
        chol_pts = 2

    if systolic_bp < 120:
        sbp_pts = -3
    elif systolic_bp < 130:
        sbp_pts = 0 if not bp_meds else 1
    elif systolic_bp < 140:
        sbp_pts = 1 if not bp_meds else 2
    elif systolic_bp < 150:
        sbp_pts = 2 if not bp_meds else 4
    elif systolic_bp < 160:
        sbp_pts = 3 if not bp_meds else 5
    else:
        sbp_pts = 4 if not bp_meds else 6

    smk_pts = 3 if smoking else 0
    diab_pts = 4 if diabetes else 0
    total = age_pts + chol_pts + sbp_pts + smk_pts + diab_pts

    table = {
        -2: 0.01,
        -1: 0.01,
        0: 0.01,
        1: 0.01,
        2: 0.01,
        3: 0.01,
        4: 0.01,
        5: 0.02,
        6: 0.02,
        7: 0.03,
        8: 0.04,
        9: 0.05,
        10: 0.06,
        11: 0.08,
        12: 0.10,
        13: 0.12,
        14: 0.16,
        15: 0.20,
        16: 0.25,
        17: 0.30,
    }
    return table.get(max(-2, min(17, total)), 0.30)


def get_risk_level(risk):
    if risk > 0.20:
        return "HIGH", "#dc2626"
    if risk > 0.10:
        return "MODERATE", "#f97316"
    return "LOW", "#16a34a"


def build_feature_comparison(patient, pop_means):
    rows = []
    for feature in FEATURES:
        patient_value = float(patient[feature])
        population_average = float(pop_means[feature])
        rows.append(
            {
                "Feature": FEATURE_LABELS[feature],
                "Your value": round(patient_value, 1),
                "Population average": round(population_average, 1),
                "Difference": round(patient_value - population_average, 1),
            }
        )
    return pd.DataFrame(rows)


def render_metric_styles(color):
    st.markdown(
        f"""
        <style>
        div[data-testid="stMetric"] {{
            border: 1px solid #e5e7eb;
            border-radius: 0.75rem;
            padding: 0.75rem;
            background-color: #ffffff;
        }}
        div[data-testid="stMetric"]:nth-of-type(1) div[data-testid="stMetricValue"] {{
            color: {color};
        }}
        div[data-testid="stMetric"]:nth-of-type(1) div[data-testid="stMetricDelta"] {{
            color: {color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="❤️ 10-Year Cardiac Risk Calculator", layout="wide")

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

try:
    cph, scaler, xgb_model, feature_names, pop_means, df = load_models()
except Exception as exc:
    st.error(f"Failed to load models: {exc}")
    st.stop()

st.title("10-Year Cardiac Risk Calculator")
st.markdown("Based on the Framingham Heart Study (Wilson 1998)")
st.markdown(
    '<p style="color:#6b7280;font-size:0.9rem;">For research purposes only. Not validated for clinical use.</p>',
    unsafe_allow_html=True,
)

st.divider()

age = st.sidebar.slider("Age", min_value=30, max_value=70, value=50, step=1)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"], index=0)
cholesterol = st.sidebar.slider(
    "Total cholesterol (mg/dL)",
    min_value=100,
    max_value=400,
    value=200,
    step=1,
)
systolic_bp = st.sidebar.slider(
    "Systolic BP (mmHg)",
    min_value=80,
    max_value=300,
    value=120,
    step=1,
)
smoking = st.sidebar.checkbox("Currently smoking", value=False)
cigs_per_day = st.sidebar.slider(
    "Cigarettes per day",
    min_value=0,
    max_value=60,
    value=0,
    step=1,
    disabled=not smoking,
)
diabetic = st.sidebar.checkbox("Diabetic", value=False)
bp_meds = st.sidebar.checkbox("On BP medication", value=False)
hypertension = st.sidebar.checkbox("Hypertensive", value=False)
diastolic_bp = st.sidebar.slider(
    "Diastolic BP (mmHg)",
    min_value=50,
    max_value=150,
    value=80,
    step=1,
)
bmi = st.sidebar.slider(
    "BMI",
    min_value=15.0,
    max_value=50.0,
    value=25.0,
    step=0.1,
)
heart_rate = st.sidebar.slider(
    "Resting heart rate",
    min_value=40,
    max_value=120,
    value=75,
    step=1,
)
glucose = st.sidebar.slider(
    "Fasting glucose (mg/dL)",
    min_value=40,
    max_value=400,
    value=80,
    step=1,
)
calculate_clicked = st.sidebar.button("Calculate risk", use_container_width=True)

patient = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "smoking_current": 1 if smoking else 0,
    "cigs_per_day": cigs_per_day if smoking else 0,
    "bp_meds": 1 if bp_meds else 0,
    "stroke": 0,
    "hypertension": 1 if hypertension else 0,
    "diabetes": 1 if diabetic else 0,
    "cholesterol": cholesterol,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "bmi": bmi,
    "heart_rate": heart_rate,
    "glucose": glucose,
}

input_signature = tuple(patient[name] for name in feature_names)
previous_signature = st.session_state.get("last_input_signature")
inputs_changed = previous_signature != input_signature
should_calculate = (
    calculate_clicked
    or inputs_changed
    or "cox_risk" not in st.session_state
    or "frs_risk" not in st.session_state
)

if should_calculate:
    st.session_state["cox_risk"] = get_cox_risk(cph, scaler, patient, feature_names)
    st.session_state["frs_risk"] = get_frs_risk(
        age=patient["age"],
        sex=patient["sex"],
        cholesterol=patient["cholesterol"],
        systolic_bp=patient["systolic_bp"],
        smoking=bool(patient["smoking_current"]),
        diabetes=bool(patient["diabetes"]),
        bp_meds=bool(patient["bp_meds"]),
    )
    st.session_state["comparison_df"] = build_feature_comparison(patient, pop_means)
    st.session_state["patient"] = patient
    st.session_state["last_input_signature"] = input_signature

cox_risk = st.session_state["cox_risk"]
frs_risk = st.session_state["frs_risk"]
comparison_df = st.session_state["comparison_df"]
patient_df = pd.DataFrame([st.session_state["patient"]])[feature_names]
risk_level, risk_color = get_risk_level(cox_risk)

render_metric_styles(risk_color)

col1, col2 = st.columns([1, 1])

with col1:
    st.metric(
        label="Our Cox model",
        value=f"{cox_risk:.1%}",
        delta=risk_level,
        delta_color="off",
    )

with col2:
    st.metric(
        label="Published Framingham Risk Score (1998)",
        value=f"{frs_risk:.1%}",
    )

st.progress(min(max(cox_risk, 0.0), 1.0))
st.caption("LOW (0-10%) | MODERATE (10-20%) | HIGH (>20%)")

st.divider()

st.subheader("What drove this prediction")
st.caption(
    "Each bar shows how much a feature pushed the risk score up (red) or down (blue) from the average patient baseline."
)

try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer(patient_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
except Exception as exc:
    st.warning(f"SHAP computation failed: {exc}")

st.divider()

st.dataframe(comparison_df, use_container_width=True, hide_index=True)
