# ══════════════════════════════════════════════════════════════════
# HEART DISEASE — 50-PATIENT INFERENCE SCRIPT
# ══════════════════════════════════════════════════════════════════
#
# Runs the trained + calibrated model on 50 held-out test patients.
# Produces:
#   - Terminal results table with colour-coded risk scores
#   - Overall metrics (AUC, PR-AUC, accuracy at threshold)
#   - Summary grid plot — all 50 patients at a glance
#   - SHAP waterfall for every patient — saved individually
#   - Error analysis — deep dive on wrong predictions
#
# Run: python inference_50.py
# Requires (same folder): heart_unified_clean.csv
#                         heart_model_calibrated.pkl
#                         feature_cols.pkl
#                         threshold.pkl

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)

# ── Terminal colours ───────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; X = "\033[0m"

def risk_colour(s):
    return G if s < 0.15 else Y if s < 0.35 else R

def risk_label(s):
    return "LOW" if s < 0.15 else "MOD" if s < 0.35 else "HIGH"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════
# CALIBRATED MODEL CLASS — must be defined BEFORE joblib.load()
# ══════════════════════════════════════════════════════════════════
#
# WHY THIS IS HERE:
#   joblib uses pickle internally. When training saved the .pkl,
#   pickle stored a reference to CalibratedModel by class name.
#   To load it back, Python must find that class in the current
#   session BEFORE unpickling. Without this, you get:
#     AttributeError: Can't get attribute 'CalibratedModel'
#   The definition must be identical to the one in heart_disease_final.py.

class CalibratedModel:
    """Wraps XGBoost + Platt scaling into a single predict_proba interface."""
    def __init__(self, base_model, platt_model):
        self.base  = base_model
        self.platt = platt_model

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.platt.predict_proba(raw)   # shape (n, 2)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# ══════════════════════════════════════════════════════════════════
# 1 — LOAD MODEL & FEATURES
# ══════════════════════════════════════════════════════════════════

print(f"\n{B}{'='*65}{X}")
print(f"{B} 50-PATIENT INFERENCE RUN{X}")
print(f"{B}{'='*65}{X}")

calibrated  = joblib.load("heart_model_calibrated.pkl")
FEATURE_COLS= joblib.load("feature_cols.pkl")
THRESHOLD   = joblib.load("threshold.pkl")

print(f"\n✓ Model loaded")
print(f"  Features  : {len(FEATURE_COLS)}")
print(f"  Threshold : {THRESHOLD:.2f}  (F2-optimised)")

# ══════════════════════════════════════════════════════════════════
# 2 — REPRODUCE EXACT TRAIN/VAL/TEST SPLIT
# ══════════════════════════════════════════════════════════════════
# CRITICAL: must use identical random_state and split fractions as
# training script so we only ever touch patients the model never saw.

df = pd.read_csv("data/heart_unified_clean.csv")
df = df.drop(columns=["split"], errors="ignore")

train_val, test_df = train_test_split(
    df, test_size=0.15, random_state=RANDOM_SEED, stratify=df["target"])
train_df, val_df = train_test_split(
    train_val, test_size=0.176, random_state=RANDOM_SEED,
    stratify=train_val["target"])

print(f"\n✓ Split reproduced — test set: {len(test_df):,} rows")

# ══════════════════════════════════════════════════════════════════
# 3 — SELECT 50 PATIENTS FROM TEST SET
# ══════════════════════════════════════════════════════════════════
# Stratified 25 positive / 25 negative so we can properly evaluate
# both sides. All from test set — never seen during training.

ECG_SOURCES = {"heart_kaggle", "uci_multicentre", "cleveland"}

def prepare(df_split):
    """Apply same transforms as training: race encoding + is_ecg_source."""
    d = df_split.copy()
    race_dummies = pd.get_dummies(
        d["race"].fillna("Unknown"), prefix="race",
        drop_first=True, dtype=float)
    d = pd.concat([d.drop(columns=["race"]), race_dummies], axis=1)
    d["is_ecg_source"] = d["source"].isin(ECG_SOURCES).astype(float)
    return d

test_df = prepare(test_df)

pos_pool = test_df[test_df["target"] == 1]
neg_pool = test_df[test_df["target"] == 0]

sample_pos = pos_pool.sample(25, random_state=RANDOM_SEED)
sample_neg = neg_pool.sample(25, random_state=RANDOM_SEED)
patients   = pd.concat([sample_pos, sample_neg])\
               .sample(frac=1, random_state=RANDOM_SEED)\
               .reset_index(drop=True)

ground_truth = patients["target"].astype(int).tolist()

# Align columns — add any race dummies that exist in FEATURE_COLS
# but might be missing from this small sample
for col in FEATURE_COLS:
    if col not in patients.columns:
        patients[col] = 0.0

X_patients = patients[FEATURE_COLS]

print(f"✓ 50 patients selected from test set")
print(f"  Positive (disease) : 25")
print(f"  Negative (healthy) : 25")
print(f"  Age range          : {patients['age'].min():.0f}–{patients['age'].max():.0f}"
      f"  (mean {patients['age'].mean():.1f})")
print(f"  Sources            : {patients['source'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════
# 4 — PREDICT
# ══════════════════════════════════════════════════════════════════

probs       = calibrated.predict_proba(X_patients)[:, 1]
predictions = (probs >= THRESHOLD).astype(int)

# ══════════════════════════════════════════════════════════════════
# 5 — RESULTS TABLE
# ══════════════════════════════════════════════════════════════════

print(f"\n{B}{'='*65}{X}")
print(f"{B} PATIENT RESULTS  (threshold = {THRESHOLD:.2f}){X}")
print(f"{B}{'='*65}{X}")
print(f"{'#':>3}  {'Age':>3} {'Sex':>3} {'Diab':>4} {'Smk':>3} "
      f"{'Strk':>4} {'GH':>2}  {'Risk%':>6}  {'Flag':>6}  "
      f"{'Truth':>8}  {'Result'}")
print(f"{'─'*65}")

correct = 0
for i in range(len(patients)):
    pt     = patients.iloc[i]
    prob   = probs[i]
    pred   = predictions[i]
    truth  = ground_truth[i]
    match  = pred == truth
    if match: correct += 1

    col    = risk_colour(prob)
    flag   = "HIGH" if pred == 1 else "low"
    truth_s= "Disease" if truth == 1 else "Healthy"
    res    = f"{G}✓{X}" if match else f"{R}✗{X}"

    # Key features for display
    age    = f"{pt.get('age', float('nan')):.0f}"
    sex    = "M" if pt.get("sex", 0) == 1 else "F"
    diab   = "Y" if pt.get("diabetes", 0) == 1 else "n"
    smk    = "Y" if pt.get("smoking_current", 0) == 1 else "n"
    strk   = "Y" if pt.get("stroke", 0) == 1 else "n"
    gh_val = pt.get("gen_health", float("nan"))
    gh     = str(int(gh_val)) if not (isinstance(gh_val, float) and np.isnan(gh_val)) else "?"

    print(f"{i+1:>3}  {age:>3} {sex:>3} {diab:>4} {smk:>3} "
          f"{strk:>4} {gh:>2}  "
          f"{col}{prob*100:>5.1f}%{X}  "
          f"{col}{flag:>6}{X}  "
          f"{truth_s:>8}  {res}")

print(f"{'─'*65}")

# ══════════════════════════════════════════════════════════════════
# 6 — METRICS
# ══════════════════════════════════════════════════════════════════

auc_roc = roc_auc_score(ground_truth, probs)
pr_auc  = average_precision_score(ground_truth, probs)
acc     = correct / len(patients)

# Confusion matrix
cm = confusion_matrix(ground_truth, predictions)
tn, fp, fn, tp = cm.ravel()

print(f"\n{B}METRICS ON 50-PATIENT SAMPLE{X}")
print(f"  Accuracy        : {correct}/50  ({acc:.0%})")
print(f"  AUC-ROC         : {auc_roc:.3f}")
print(f"  PR-AUC          : {pr_auc:.3f}")
print(f"\n  Confusion matrix at threshold {THRESHOLD:.2f}:")
print(f"  {'':15}  Pred Healthy  Pred Disease")
print(f"  {'True Healthy':15}  {tn:>12}  {fp:>12}  ← false alarms")
print(f"  {'True Disease':15}  {fn:>12}  {tp:>12}  ← caught cases")
print(f"\n  Recall (sensitivity) : {tp/(tp+fn):.1%}  — caught {tp} of {tp+fn} disease cases")
print(f"  Precision            : {tp/(tp+fp):.1%}  — {tp} of {tp+fp} HIGH flags were real")
print(f"  Specificity          : {tn/(tn+fp):.1%}  — cleared {tn} of {tn+fp} healthy patients")

print(f"\n{classification_report(ground_truth, predictions, target_names=['Healthy','Disease'])}")

# ══════════════════════════════════════════════════════════════════
# 7 — ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════

errors = [(i, probs[i], predictions[i], ground_truth[i], patients.iloc[i])
          for i in range(50) if predictions[i] != ground_truth[i]]

print(f"\n{B}ERROR ANALYSIS — {len(errors)} wrong predictions{X}")
print(f"{'─'*65}")

for i, prob, pred, truth, pt in errors:
    etype = "FALSE POSITIVE" if pred==1 and truth==0 else "FALSE NEGATIVE"
    col   = Y if etype == "FALSE POSITIVE" else R
    print(f"\n  P{i+1:02d} — {col}{etype}{X}  (predicted {pred}, truth {truth})")
    print(f"  Risk score : {prob*100:.1f}%")
    print(f"  Age={pt.get('age',float('nan')):.0f}  "
          f"Sex={'M' if pt.get('sex',0)==1 else 'F'}  "
          f"BMI={pt.get('bmi',float('nan')):.1f}  "
          f"Diabetes={pt.get('diabetes',0):.0f}  "
          f"Stroke={pt.get('stroke',0):.0f}  "
          f"Smoking={pt.get('smoking_current',0):.0f}")
    gh = pt.get("gen_health", float("nan"))
    if not (isinstance(gh, float) and np.isnan(gh)):
        print(f"  Gen_health={gh:.0f}  Diff_walking={pt.get('diff_walking',0):.0f}  "
              f"Phys_health_days={pt.get('physical_health_days',0):.0f}")
    ecg = pt.get("is_ecg_source", 0)
    if ecg == 1:
        print(f"  ⚠ ECG-source patient — missing BRFSS lifestyle features")

# ══════════════════════════════════════════════════════════════════
# 8 — SUMMARY GRID PLOT (all 50 at a glance)
# ══════════════════════════════════════════════════════════════════

os.makedirs("inference_50", exist_ok=True)

fig = plt.figure(figsize=(24, 20))
fig.suptitle("50-Patient Risk Summary — Calibrated Scores\n"
             "Green bg = Healthy truth | Red bg = Disease truth | "
             "✓ = correct | ✗ = wrong",
             fontsize=12, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(5, 10, figure=fig, hspace=0.6, wspace=0.35)

for i in range(50):
    ax   = fig.add_subplot(gs[i // 10, i % 10])
    prob = probs[i]
    pred = predictions[i]
    truth= ground_truth[i]
    pt   = patients.iloc[i]
    match= pred == truth

    bg   = "#EAF3DE" if truth == 0 else "#FBEAEA"
    bar_c= "#E24B4A" if prob >= 0.35 else "#EF9F27" if prob >= 0.15 else "#1D9E75"
    res  = "✓" if match else "✗"

    ax.set_facecolor(bg)
    ax.barh([0], [1],    color="#E0E0E0", height=0.5, zorder=0)
    ax.barh([0], [prob], color=bar_c,     height=0.5, zorder=1)
    ax.axvline(THRESHOLD, color="#555", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1); ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=6)

    age = pt.get("age", float("nan"))
    sex = "M" if pt.get("sex", 0) == 1 else "F"
    truth_s = "Dis" if truth == 1 else "Hlthy"

    ax.set_title(
        f"P{i+1:02d} {res}  {prob*100:.0f}%\n"
        f"Age{age:.0f} {sex} | {truth_s}",
        fontsize=7, pad=2,
        color="#C0392B" if not match else "#2C3E50"
    )

plt.savefig("inference_50/summary_50_patients.png",
            dpi=130, bbox_inches="tight")
plt.close()
print(f"\n✓ Summary grid saved → inference_50/summary_50_patients.png")

# ══════════════════════════════════════════════════════════════════
# 9 — SHAP WATERFALL PLOTS (one per patient)
# ══════════════════════════════════════════════════════════════════

print(f"\n{B}Computing SHAP values for all 50 patients...{X}")
print(f"  (this takes ~60 seconds)")

# Use raw XGBoost model for SHAP — TreeExplainer is exact for trees
base_model  = calibrated.base
explainer   = shap.TreeExplainer(base_model)
shap_values = explainer(X_patients)

for i in range(50):
    truth  = ground_truth[i]
    prob   = probs[i]
    pred   = predictions[i]
    match  = pred == truth

    truth_s = "Disease" if truth == 1 else "Healthy"
    flag_s  = "HIGH RISK" if pred == 1 else "LOW RISK"
    title_c = "#E24B4A" if pred == 1 else "#1D9E75"
    res_s   = "✓ Correct" if match else "✗ Wrong"

    fig, ax = plt.subplots(figsize=(11, 6))
    shap.plots.waterfall(shap_values[i], max_display=12, show=False)
    plt.title(
        f"P{i+1:02d} — Age {patients.iloc[i].get('age',0):.0f}  "
        f"{'Male' if patients.iloc[i].get('sex',0)==1 else 'Female'}\n"
        f"Risk: {prob*100:.1f}% → {flag_s}   |   "
        f"Truth: {truth_s}   |   {res_s}",
        fontsize=9, pad=10, color=title_c
    )
    plt.tight_layout()
    fname = f"inference_50/p{i+1:02d}_waterfall.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()

    if (i+1) % 10 == 0:
        print(f"  Saved {i+1}/50 waterfall plots...")

print(f"✓ All 50 waterfall plots saved → inference_50/")

# ══════════════════════════════════════════════════════════════════
# 10 — RISK DISTRIBUTION PLOT
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Risk Score Distribution — 50 Patients", fontsize=12, fontweight="bold")

# Histogram split by truth
dis_probs = [probs[i] for i in range(50) if ground_truth[i] == 1]
hlth_probs= [probs[i] for i in range(50) if ground_truth[i] == 0]

axes[0].hist(hlth_probs, bins=12, color="#1D9E75", alpha=0.7, label="Healthy (truth)")
axes[0].hist(dis_probs,  bins=12, color="#E24B4A", alpha=0.7, label="Disease (truth)")
axes[0].axvline(THRESHOLD, color="black", lw=1.5, linestyle="--",
                label=f"Threshold {THRESHOLD:.2f}")
axes[0].set_xlabel("Calibrated risk score")
axes[0].set_ylabel("Count")
axes[0].set_title("Score distribution by truth")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Scatter: sorted by predicted risk, coloured by truth
sorted_idx = np.argsort(probs)
colours    = ["#E24B4A" if ground_truth[i]==1 else "#1D9E75" for i in sorted_idx]
axes[1].bar(range(50), probs[sorted_idx], color=colours, width=0.8)
axes[1].axhline(THRESHOLD, color="black", lw=1.5, linestyle="--",
                label=f"Threshold {THRESHOLD:.2f}")
axes[1].set_xlabel("Patients (sorted by risk)")
axes[1].set_ylabel("Risk score")
axes[1].set_title("Risk ladder — red=disease, green=healthy")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="y")

# Calibration check on the 50 patients
import numpy as np
bins = np.linspace(0, 1, 6)
bin_means, actual_rates = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (probs >= lo) & (probs < hi)
    if mask.sum() >= 3:
        bin_means.append(probs[mask].mean())
        actual_rates.append(np.array(ground_truth)[mask].mean())

if len(bin_means) >= 2:
    axes[2].plot(bin_means, actual_rates, "s-", color="#D85A30",
                 lw=2, markersize=8, label="Observed")
    axes[2].plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Actual positive rate")
    axes[2].set_title("Calibration (50 patients)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
else:
    axes[2].text(0.5, 0.5, "Too few bins\nfor calibration",
                 ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Calibration (50 patients)")

plt.tight_layout()
plt.savefig("inference_50/risk_distribution.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"\n✓ Risk distribution plot saved → inference_50/risk_distribution.png")

# ══════════════════════════════════════════════════════════════════
# 11 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

fp_count = sum(1 for i in range(50) if predictions[i]==1 and ground_truth[i]==0)
fn_count = sum(1 for i in range(50) if predictions[i]==0 and ground_truth[i]==1)

print(f"\n{B}{'='*65}{X}")
print(f"{B} FINAL SUMMARY{X}")
print(f"{B}{'='*65}{X}")
print(f"  Patients evaluated  : 50  (25 disease / 25 healthy)")
print(f"  Correct predictions : {correct}/50  ({acc:.0%})")
print(f"  AUC-ROC             : {auc_roc:.3f}")
print(f"  PR-AUC              : {pr_auc:.3f}")
print(f"\n  At threshold {THRESHOLD:.2f}:")
print(f"  True positives  (caught disease)   : {tp}/25  ({tp/25:.0%})")
print(f"  True negatives  (cleared healthy)  : {tn}/25  ({tn/25:.0%})")
print(f"  False positives (false alarms)     : {fp_count}")
print(f"  False negatives (missed disease)   : {fn_count}")
print(f"\n  Outputs saved to: inference_50/")
print(f"  ├─ summary_50_patients.png   (grid overview)")
print(f"  ├─ risk_distribution.png     (score analysis)")
print(f"  └─ p01_waterfall.png ... p50_waterfall.png")
print(f"{B}{'='*65}{X}\n")