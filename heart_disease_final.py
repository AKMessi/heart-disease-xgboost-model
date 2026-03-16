# ══════════════════════════════════════════════════════════════════
# HEART DISEASE PREDICTION — FINAL COMPLETE SCRIPT
# ══════════════════════════════════════════════════════════════════
#
# What this script does, end to end:
#   1.  Load the unified dataset
#   2.  Fix the test set — proper 70/15/15 stratified split
#       (replaces Framingham-as-test which caused distribution shift)
#   3.  One-hot encode race (new feature, was dropped before)
#   4.  Build feature matrix — 62 features total
#   5.  Hyperparameter tuning — Optuna, 100 trials, 3-fold CV
#   6.  Retrain final model on best params, early stopping on val
#   7.  Platt scaling calibration — fixes probability scores
#   8.  Full evaluation — AUC-ROC, PR-AUC, Brier, confusion matrix
#   9.  Optimal threshold selection — F1 and F2
#   10. SHAP explainability — summary + importance plots
#   11. Save all outputs
#
# Install requirements:
#   pip install xgboost shap optuna scikit-learn pandas numpy matplotlib joblib
#
# Run:
#   python heart_disease_final.py
#
# Expected runtime: ~25-45 minutes (dominated by Optuna tuning)
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import optuna
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, roc_curve, precision_recall_curve,
    fbeta_score, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Config ─────────────────────────────────────────────────────────
DATA_PATH  = "data/heart_unified_clean.csv"   # update if in a different folder
N_TRIALS   = 100                          # increase to 200 for better tuning
OUTPUT_DIR = "."                          # where plots and models are saved

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("HEART DISEASE PREDICTION — FINAL SCRIPT")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"    Sources: {df['source'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════
# 2. REBUILD THE SPLIT — PROPER 70/15/15 STRATIFIED
# ══════════════════════════════════════════════════════════════════
#
# Previously: Framingham was the test set → caused distribution
# shift (different era, different feature distributions, different
# positive rate) → test AUC crashed to 0.43.
#
# Fix: ignore the old split column. Do a fresh stratified 70/15/15
# split across the entire dataset so train/val/test all come from
# the same distribution (mostly BRFSS 2020).
#
# Framingham rows are now included in training — their longitudinal
# labels are genuinely useful signal, not a liability.

df = df.drop(columns=["split"], errors="ignore")

train_val, test = train_test_split(
    df, test_size=0.15, random_state=RANDOM_SEED, stratify=df["target"]
)
train, val = train_test_split(
    train_val, test_size=0.176,   # 0.176 × 0.85 ≈ 0.15 of total
    random_state=RANDOM_SEED, stratify=train_val["target"]
)

print(f"\n[2] Split rebuilt (70/15/15 stratified):")
for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
    print(f"    {name:<6}: {len(subset):>6} rows | "
          f"{subset.target.mean():.2%} positive")

# ══════════════════════════════════════════════════════════════════
# 3. ONE-HOT ENCODE RACE
# ══════════════════════════════════════════════════════════════════
#
# Race was dropped in earlier versions. It carries real signal:
# American Indian/Alaskan Native has 3× the positive rate of Asian.
# One-hot encoding converts the 6-category string column into 5
# binary columns (drop_first=True avoids multicollinearity).
# Rows with missing race get 0 in all race columns — a valid
# representation of "unknown".

def encode_race(df_split):
    race_dummies = pd.get_dummies(
        df_split["race"].fillna("Unknown"),
        prefix="race",
        drop_first=True,   # drops race_Asian (reference category)
        dtype=float,
    )
    return pd.concat([df_split.drop(columns=["race"]), race_dummies], axis=1)

train = encode_race(train)
val   = encode_race(val)
test  = encode_race(test)

race_cols = [c for c in train.columns if c.startswith("race_")]
print(f"\n[3] Race one-hot encoded: {race_cols}")

ECG_SOURCES = {"heart_kaggle", "uci_multicentre", "cleveland"}

def add_source_flag(df_split):
    df_split = df_split.copy()
    df_split["is_ecg_source"] = df_split["source"].isin(ECG_SOURCES).astype(float)
    return df_split

train = add_source_flag(train)
val   = add_source_flag(val)
test  = add_source_flag(test)

print(f"    is_ecg_source added — ECG rows in train: "
      f"{train['is_ecg_source'].sum():.0f} / {len(train)}")

# ══════════════════════════════════════════════════════════════════
# 4. BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════
#
# Drop metadata columns that are not clinical features.
# Everything else — including missing-indicator flags (_was_missing)
# and engineered features — is kept.
# XGBoost handles NaN natively: no imputation needed here.

DROP_COLS    = ["target", "source", "age_band"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]

X_train = train[FEATURE_COLS];  y_train = train["target"]
X_val   = val[FEATURE_COLS];    y_val   = val["target"]
X_test  = test[FEATURE_COLS];   y_test  = test["target"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
SCALE_POS_WEIGHT = neg / pos

print(f"\n[4] Features: {len(FEATURE_COLS)}")
print(f"    X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
print(f"    Positives in train: {pos:,} / {len(y_train):,} "
      f"({y_train.mean():.2%})")
print(f"    scale_pos_weight = {SCALE_POS_WEIGHT:.1f}")

# ══════════════════════════════════════════════════════════════════
# 5. OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════

def objective(trial):
    params = {
        "max_depth":          trial.suggest_int("max_depth", 3, 8),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators":             500,
        "early_stopping_rounds":     30,
        "scale_pos_weight":   SCALE_POS_WEIGHT,
        "objective":          "binary:logistic",
        "eval_metric":        "aucpr",
        "tree_method":        "hist",
        "random_state":       RANDOM_SEED,
        "verbosity":          0,
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(cv.split(X_train, y_train)):
        Xf_tr, yf_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        Xf_vl, yf_vl = X_train.iloc[vl_idx], y_train.iloc[vl_idx]

        m = xgb.XGBClassifier(**params)
        m.fit(Xf_tr, yf_tr, eval_set=[(Xf_vl, yf_vl)], verbose=False)

        score = average_precision_score(yf_vl, m.predict_proba(Xf_vl)[:, 1])
        fold_scores.append(score)

        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_scores)


print(f"\n[5] Optuna tuning — {N_TRIALS} trials × 3-fold CV")
print(f"    Estimated time: ~25-40 min\n")

def print_progress(study, trial):
    if trial.number % 10 == 0 or trial.number < 5:
        val = trial.value if trial.value is not None else "pruned"
        val_str = f"{val:.4f}" if isinstance(val, float) else val
        print(f"    Trial {trial.number:>3} | current: {val_str:<8} | "
              f"best: {study.best_value:.4f}")

study = optuna.create_study(
    direction = "maximize",
    sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner    = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1),
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_progress])

print(f"\n    Best CV PR-AUC : {study.best_value:.4f}  (trial #{study.best_trial.number})")
print(f"    Best params:")
for k, v in study.best_params.items():
    print(f"      {k:<25} {v}")

# ── Optuna history plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
trials_df  = study.trials_dataframe()
completed  = trials_df[trials_df["state"] == "COMPLETE"]

axes[0].plot(completed["number"], completed["value"],
             "o", alpha=0.35, color="#378ADD", markersize=3, label="Trial")
axes[0].plot(completed["number"], completed["value"].cummax(),
             "-", color="#E24B4A", lw=2, label="Best so far")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("CV PR-AUC")
axes[0].set_title("Optimisation history"); axes[0].legend()
axes[0].grid(True, alpha=0.3)

importances = optuna.importance.get_param_importances(study)
names  = list(importances.keys())[:10]
values = [importances[n] for n in names]
axes[1].barh(names[::-1], values[::-1], color="#1D9E75")
axes[1].set_xlabel("Importance (fANOVA)")
axes[1].set_title("Hyperparameter importance")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/optuna_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n    Saved: optuna_results.png")

# ══════════════════════════════════════════════════════════════════
# 6. RETRAIN FINAL MODEL WITH BEST PARAMS
# ══════════════════════════════════════════════════════════════════
#
# Use train set only. Watch val for early stopping.
# Val has the same distribution as train → honest early stopping.
# Test set stays completely untouched until step 8.

print(f"\n[6] Retraining final model with best params...")

best_params = study.best_params.copy()

# Floor learning_rate: very low lr (< 0.02) combined with
# early_stopping_rounds=50 can stop the model before it has
# grown enough trees to converge. 0.02 is the safe minimum.
lr = best_params.get("learning_rate", 0.05)
if lr < 0.02:
    print(f"    Flooring learning_rate {lr:.4f} → 0.02")
    best_params["learning_rate"] = 0.02

best_params.update({
    "n_estimators":          1000,
    "early_stopping_rounds": 50,
    "scale_pos_weight":      SCALE_POS_WEIGHT,
    "objective":             "binary:logistic",
    "eval_metric":           "aucpr",
    "tree_method":           "hist",
    "random_state":          RANDOM_SEED,
    "verbosity":             1,
})

model = xgb.XGBClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set = [(X_train, y_train), (X_val, y_val)],
    verbose  = 50,
)

print(f"\n    Best iteration  : {model.best_iteration}")
print(f"    Val PR-AUC      : {average_precision_score(y_val, model.predict_proba(X_val)[:,1]):.4f}")

# ══════════════════════════════════════════════════════════════════
# 7. PLATT SCALING CALIBRATION
# ══════════════════════════════════════════════════════════════════
#
# Fits a 2-parameter logistic regression on top of the model's
# raw scores using the validation set. Maps "raw score → calibrated
# probability" so that predicted 0.30 actually means ~30% of those
# patients have heart disease.
#
# This is sklearn-version-safe (works with sklearn 1.0 through 1.8+).

print(f"\n[7] Fitting Platt scaling calibration...")

raw_val_scores = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
platt = LogisticRegression(random_state=RANDOM_SEED)
platt.fit(raw_val_scores, y_val)

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

calibrated = CalibratedModel(model, platt)

# Quick sanity check
brier_raw = brier_score_loss(y_val, model.predict_proba(X_val)[:, 1])
brier_cal = brier_score_loss(y_val, calibrated.predict_proba(X_val)[:, 1])
print(f"    Brier before calibration: {brier_raw:.4f}")
print(f"    Brier after calibration : {brier_cal:.4f}  "
      f"({'better ✓' if brier_cal < brier_raw else 'worse — check val set'})")

# ══════════════════════════════════════════════════════════════════
# 8. FULL EVALUATION
# ══════════════════════════════════════════════════════════════════

cal_val_probs  = calibrated.predict_proba(X_val)[:, 1]
cal_test_probs = calibrated.predict_proba(X_test)[:, 1]
raw_val_probs  = model.predict_proba(X_val)[:, 1]

val_auc  = roc_auc_score(y_val, cal_val_probs)
val_pr   = average_precision_score(y_val, cal_val_probs)
val_brier= brier_score_loss(y_val, cal_val_probs)

test_auc  = roc_auc_score(y_test, cal_test_probs)
test_pr   = average_precision_score(y_test, cal_test_probs)
test_brier= brier_score_loss(y_test, cal_test_probs)

print(f"\n{'='*60}")
print(f"[8] EVALUATION RESULTS")
print(f"{'='*60}")
print(f"    {'Metric':<18} {'Validation':>12} {'Test':>10}")
print(f"    {'-'*42}")
print(f"    {'AUC-ROC':<18} {val_auc:>12.4f} {test_auc:>10.4f}")
print(f"    {'PR-AUC':<18} {val_pr:>12.4f} {test_pr:>10.4f}")
print(f"    {'Brier score':<18} {val_brier:>12.4f} {test_brier:>10.4f}")

# ── Evaluation plots ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Final Model — Evaluation Curves", fontsize=13, fontweight="bold")

# ROC
for probs, labels, color, label in [
    (cal_val_probs,  y_val,  "#378ADD", f"Val  AUC={val_auc:.3f}"),
    (cal_test_probs, y_test, "#1D9E75", f"Test AUC={test_auc:.3f}"),
]:
    fpr, tpr, _ = roc_curve(labels, probs)
    axes[0].plot(fpr, tpr, color=color, lw=2, label=label)
axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# PR
for probs, labels, color, label in [
    (cal_val_probs,  y_val,  "#378ADD", f"Val  PR-AUC={val_pr:.3f}"),
    (cal_test_probs, y_test, "#1D9E75", f"Test PR-AUC={test_pr:.3f}"),
]:
    prec, rec, _ = precision_recall_curve(labels, probs)
    axes[1].plot(rec, prec, color=color, lw=2, label=label)
axes[1].axhline(y=y_val.mean(), color="k", lw=1, linestyle="--",
                alpha=0.4, label=f"Random ({y_val.mean():.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Calibration
frac_raw, mean_raw = calibration_curve(y_val, raw_val_probs,  n_bins=10)
frac_cal, mean_cal = calibration_curve(y_val, cal_val_probs, n_bins=10)
axes[2].plot(mean_raw, frac_raw, "s--", color="#888780", lw=1.5,
             label="Before calibration", alpha=0.7)
axes[2].plot(mean_cal, frac_cal, "s-",  color="#D85A30", lw=2,
             label="After calibration")
axes[2].plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Perfect")
axes[2].set_xlabel("Mean predicted probability")
axes[2].set_ylabel("Fraction of positives")
axes[2].set_title("Calibration Curve (val set)")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/evaluation_final.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n    Saved: evaluation_final.png")

# ══════════════════════════════════════════════════════════════════
# 9. THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════════════
#
# F2 score weights recall 2× more than precision.
# Right choice for medical screening: missing a sick patient (false
# negative) is worse than a false alarm (false positive).

thresholds = np.arange(0.05, 0.65, 0.01)
f1_scores, f2_scores, precisions, recalls = [], [], [], []

for t in thresholds:
    preds = (cal_val_probs >= t).astype(int)
    f1_scores.append(fbeta_score(y_val, preds, beta=1, zero_division=0))
    f2_scores.append(fbeta_score(y_val, preds, beta=2, zero_division=0))
    precisions.append(precision_score(y_val, preds, zero_division=0))
    recalls.append(recall_score(y_val, preds, zero_division=0))

best_f1_thresh = thresholds[np.argmax(f1_scores)]
best_f2_thresh = thresholds[np.argmax(f2_scores)]

print(f"\n[9] THRESHOLD SELECTION")
print(f"    Best F1 threshold : {best_f1_thresh:.2f}  (balanced)")
print(f"    Best F2 threshold : {best_f2_thresh:.2f}  (recall-focused — use for screening)")

preds_f2 = (cal_val_probs >= best_f2_thresh).astype(int)
print(f"\n    Classification report at F2 threshold ({best_f2_thresh:.2f}):")
print(classification_report(y_val, preds_f2,
      target_names=["No disease", "Heart disease"],
      digits=3))

# Threshold plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precisions, "--", color="#378ADD", lw=1.5, label="Precision")
ax.plot(thresholds, recalls,    "--", color="#1D9E75", lw=1.5, label="Recall")
ax.plot(thresholds, f1_scores,  "-",  color="#534AB7", lw=2,   label="F1")
ax.plot(thresholds, f2_scores,  "-",  color="#D85A30", lw=2,   label="F2 (recall×2)")
ax.axvline(x=best_f2_thresh, color="#D85A30", linestyle=":",
           lw=1.5, label=f"Best F2 = {best_f2_thresh:.2f}")
ax.axvline(x=best_f1_thresh, color="#534AB7", linestyle=":",
           lw=1.5, label=f"Best F1 = {best_f1_thresh:.2f}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold selection"); ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/threshold_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: threshold_selection.png")

# ══════════════════════════════════════════════════════════════════
# 10. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
#
# Computed on the raw (pre-calibration) model — SHAP + tree models
# work together natively. The calibration layer is a monotonic
# transform so feature importance ranking is unchanged.
#
# Use a random 2000-row sample of val for speed. SHAP values on
# 2000 rows are statistically stable and compute in ~30 seconds.

print(f"\n[10] Computing SHAP values (~30 seconds)...")

shap_sample = X_val.sample(n=min(2000, len(X_val)), random_state=RANDOM_SEED)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(shap_sample)

# Summary plot (dot cloud — most informative)
plt.figure(figsize=(10, 9))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=FEATURE_COLS,
                  max_display=25, show=False)
plt.title("SHAP Feature Importance — Top 25 Features\n"
          "(red=high value raises risk, blue=low value lowers risk)",
          fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Bar plot (mean absolute SHAP — cleaner ranking)
plt.figure(figsize=(8, 7))
shap.summary_plot(shap_values, shap_sample,
                  feature_names=FEATURE_COLS,
                  plot_type="bar", max_display=25, show=False)
plt.title("Mean |SHAP| — Overall Feature Importance", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"    Saved: shap_summary.png")
print(f"    Saved: shap_importance.png")

# ══════════════════════════════════════════════════════════════════
# 11. SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════

model.save_model(f"{OUTPUT_DIR}/heart_xgboost_final.json")
joblib.dump(calibrated,      f"{OUTPUT_DIR}/heart_model_calibrated.pkl")
joblib.dump(FEATURE_COLS,    f"{OUTPUT_DIR}/feature_cols.pkl")
joblib.dump(best_f2_thresh,  f"{OUTPUT_DIR}/threshold.pkl")

print(f"\n{'='*60}")
print(f"COMPLETE — FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"    Val  AUC-ROC : {val_auc:.4f}  |  PR-AUC: {val_pr:.4f}  |  Brier: {val_brier:.4f}")
print(f"    Test AUC-ROC : {test_auc:.4f}  |  PR-AUC: {test_pr:.4f}  |  Brier: {test_brier:.4f}")
print(f"    Best threshold (F2) : {best_f2_thresh:.2f}")
print(f"    Best iteration      : {model.best_iteration}")

print(f"""
FILES SAVED:
  heart_xgboost_final.json      raw XGBoost model
  heart_model_calibrated.pkl    calibrated model  ← use this
  feature_cols.pkl              list of 62 feature names
  threshold.pkl                 optimal decision threshold
  evaluation_final.png          ROC + PR + calibration curves
  optuna_results.png            tuning history + param importance
  shap_summary.png              SHAP dot cloud (top 25 features)
  shap_importance.png           SHAP bar chart
  threshold_selection.png       F1/F2 threshold analysis

HOW TO PREDICT ON A NEW PATIENT:
  import joblib, pandas as pd
  model     = joblib.load('heart_model_calibrated.pkl')
  features  = joblib.load('feature_cols.pkl')
  threshold = joblib.load('threshold.pkl')

  patient = pd.DataFrame([{{
      'age': 58, 'sex': 1, 'bmi': 29.5,
      'smoking_current': 1, 'diabetes': 1,
      'gen_health': 3, 'stroke': 0,
      # fill in what you have, leave the rest as NaN
  }}])
  # add any missing columns as NaN
  for col in features:
      if col not in patient.columns:
          patient[col] = float('nan')

  risk  = model.predict_proba(patient[features])[0, 1]
  alert = risk >= threshold
  print(f'Risk score : {{risk:.1%}}')
  print(f'High risk  : {{alert}}')
""")