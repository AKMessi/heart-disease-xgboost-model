# ══════════════════════════════════════════════════════════════════
# HEART ATTACK PREDICTION — GRADIENT BOOSTING TRAINING SCRIPT
# ══════════════════════════════════════════════════════════════════
#
# WHAT THIS SCRIPT DOES (big picture):
#   1. Load the curated dataset
#   2. Prepare features and labels
#   3. Train an XGBoost gradient boosting model
#   4. Evaluate it properly (AUC-ROC, PR-AUC, calibration)
#   5. Explain every prediction with SHAP values
#
# WHY EACH LIBRARY:
#   pandas      — loads and manipulates tabular data (your CSV)
#   numpy       — fast number arrays; pandas uses it under the hood
#   xgboost     — the gradient boosting model itself
#   sklearn     — train/test splitting, metrics, calibration tools
#   shap        — explains what the model learned (non-negotiable for medical ML)
#   matplotlib  — draws the evaluation and SHAP plots
#   optuna      — finds the best hyperparameters automatically (later step)

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # saves plots to files instead of opening windows

from sklearn.metrics import (
    roc_auc_score,          # AUC-ROC: overall discrimination ability
    average_precision_score, # PR-AUC: better metric when classes are imbalanced
    roc_curve,              # the actual curve points for plotting
    precision_recall_curve, # same for PR curve
    confusion_matrix,       # TP/FP/TN/FN breakdown
    classification_report,  # precision, recall, F1 per class
    brier_score_loss,       # calibration quality (0=perfect, 1=terrible)
)
from sklearn.calibration import calibration_curve  # plots predicted vs actual %

import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────
# Setting a random seed means you get the SAME result every time you
# run the script. Without this, results differ slightly each run
# due to random sampling inside XGBoost. In research you always set this.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("✓ Imports loaded")

# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════
#
# WHY WE SPLIT INTO TRAIN/VAL/TEST:
#   Train  → the model learns from this
#   Val    → we tune hyperparameters against this (model never trains on it)
#   Test   → touched ONCE at the very end; the honest final score
#
# If you tune hyperparameters on the test set, you're cheating —
# the model indirectly "sees" the test set and scores look better
# than they'll be in real deployment. This is one of the most
# common mistakes in ML papers.

DATA_PATH = "data/heart_unified_clean.csv"  # update path if needed

df = pd.read_csv(DATA_PATH)

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

print(f"\n✓ Data loaded")
print(f"   Train : {len(train_df):>6} rows  |  {train_df.target.mean():.2%} positive")
print(f"   Val   : {len(val_df):>6} rows  |  {val_df.target.mean():.2%} positive")
print(f"   Test  : {len(test_df):>6} rows  |  {test_df.target.mean():.2%} positive")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════
#
# We drop columns that are NOT features:
#   target    → the answer we're trying to predict (never a feature)
#   split     → just a label we added to track train/val/test rows
#   source    → which dataset the row came from (not a clinical signal)
#   age_band  → we already have numeric `age`; the band is redundant
#   race      → text column; we'd need to encode it specially.
#               We leave it out for v1 — add it back with one-hot
#               encoding in v2 to enable fairness analysis.
#
# Everything else is a feature. XGBoost can handle:
#   - Numeric columns (age, BMI, cholesterol)
#   - Binary 0/1 columns (smoking, diabetes, sex)
#   - Missing values (NaN) — XGBoost has a built-in missing value
#     handler; it learns which branch to send NaN values down

DROP_COLS = ["target", "split", "source", "age_band", "race"]

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

print(f"\n✓ Feature columns selected: {len(FEATURE_COLS)} features")
print(f"   Dropped : {DROP_COLS}")

# ── Build the actual arrays ────────────────────────────────────────
# X = features (input), y = label (output we predict)
# .values converts pandas DataFrame to a numpy array
# XGBoost accepts both, but numpy is slightly faster

X_train = train_df[FEATURE_COLS]
y_train = train_df["target"]

X_val   = val_df[FEATURE_COLS]
y_val   = val_df["target"]

X_test  = test_df[FEATURE_COLS]
y_test  = test_df["target"]

print(f"\n✓ Feature matrices built")
print(f"   X_train shape: {X_train.shape}  (rows × features)")
print(f"   X_val shape  : {X_val.shape}")
print(f"   X_test shape : {X_test.shape}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — CLASS IMBALANCE: scale_pos_weight
# ══════════════════════════════════════════════════════════════════
#
# Only 9.97% of training rows are positive (heart disease = 1).
# If we train naively, the model learns "always predict 0" and gets
# 90% accuracy while being completely useless clinically.
#
# XGBoost's fix: scale_pos_weight
#   = number of negatives / number of positives
#   = tells XGBoost "treat each positive example as if it were
#     scale_pos_weight times more important than a negative"
#
# This is equivalent to oversampling the minority class but without
# actually duplicating rows — it's done mathematically during training.

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
SCALE_POS_WEIGHT = neg / pos

print(f"\n✓ Class imbalance calculated")
print(f"   Negatives    : {neg:>6}")
print(f"   Positives    : {pos:>6}")
print(f"   Ratio (neg/pos): {SCALE_POS_WEIGHT:.2f}")
print(f"   scale_pos_weight = {SCALE_POS_WEIGHT:.1f}")

print("\n" + "="*55)
print("STEPS 1 & 2 COMPLETE — data ready for model definition")
print("="*55)

# ══════════════════════════════════════════════════════════════════
# STEP 3 — DEFINE THE MODEL (baseline, before tuning)
# ══════════════════════════════════════════════════════════════════
#
# XGBoost parameters — what each one does:
#
#   n_estimators (300)
#     How many trees to build. Each tree corrects the errors of all
#     previous trees. More trees = better fit, but risks overfitting.
#     We'll use early stopping to find the right number automatically.
#
#   max_depth (4)
#     How deep each tree can grow. Depth 4 means each tree can ask
#     4 yes/no questions about your features. Shallow trees are
#     "weak learners" — that's intentional. Boosting combines many
#     weak learners into one strong one. Deep trees memorise training
#     data (overfit). For medical tabular data, 3–6 is typical.
#
#   learning_rate (0.05)
#     How much each new tree's contribution is shrunk before adding
#     it to the ensemble. Lower = more conservative = needs more trees
#     but generalises better. Think of it as step size when walking
#     downhill to the loss minimum. Too big → overshoot. Too small →
#     slow but precise. 0.05 is a safe starting point.
#
#   subsample (0.8)
#     Each tree is trained on a random 80% of the training rows.
#     The remaining 20% is ignored for that tree. This randomness
#     prevents any single tree from memorising the data — same idea
#     as dropout in neural networks. 0.8 is standard.
#
#   colsample_bytree (0.8)
#     Each tree only sees a random 80% of the features. Forces trees
#     to find different patterns, making the ensemble more robust.
#
#   min_child_weight (5)
#     A leaf node must have at least 5 samples. Prevents the tree
#     from creating splits that only apply to 1-2 unusual patients.
#     Higher = simpler trees = less overfitting.
#
#   reg_alpha (0.1) — L1 regularisation
#     Pushes small feature weights toward exactly zero. Effectively
#     does feature selection inside the model — unimportant features
#     get zeroed out. Good when you have 56 features, some of which
#     might be noise.
#
#   reg_lambda (1.0) — L2 regularisation
#     Penalises large feature weights. Smooths the model and prevents
#     any single feature from dominating excessively.
#
#   scale_pos_weight
#     The imbalance fix we calculated above. Tells XGBoost to weight
#     positive examples (heart disease) 9x heavier than negatives.
#
#   objective ('binary:logistic')
#     We're doing binary classification (disease or not). This tells
#     XGBoost to use log loss and output a probability between 0 and 1.
#
#   eval_metric ('aucpr')
#     PR-AUC is our primary training metric. We use it instead of
#     accuracy because our classes are imbalanced. A model that
#     always predicts "no disease" gets 90% accuracy but 0% PR-AUC.
#     PR-AUC only improves when you correctly identify true positives.
#
#   tree_method ('hist')
#     Uses histogram-based tree building — much faster than exact
#     method on large datasets. Standard for production XGBoost.
#
#   random_state (42)
#     Reproducibility — same result every run.

model = xgb.XGBClassifier(
    n_estimators          = 500,      # will be cut short by early stopping
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    reg_alpha             = 0.1,      # L1
    reg_lambda            = 1.0,      # L2
    scale_pos_weight      = SCALE_POS_WEIGHT,
    objective             = "binary:logistic",
    eval_metric           = "aucpr",
    tree_method           = "hist",
    random_state          = RANDOM_SEED,
    verbosity             = 1,
    early_stopping_rounds = 30,   # moved here in XGBoost 2.0+
)

print("\n✓ Model defined — baseline XGBoost")
print(f"   scale_pos_weight = {SCALE_POS_WEIGHT:.1f}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN WITH EARLY STOPPING
# ══════════════════════════════════════════════════════════════════
#
# WHAT IS EARLY STOPPING?
#   We set n_estimators=500 but we don't necessarily want 500 trees.
#   Early stopping watches the validation PR-AUC after each tree is
#   added. If it hasn't improved for 30 consecutive trees, training
#   stops automatically and rolls back to the best tree count found.
#
#   WHY THIS MATTERS:
#   Without early stopping: model trains all 500 trees, overfits
#   the training data, looks great on train, collapses on val/test.
#   With early stopping: model stops at the exact point where it
#   generalises best — not when it memorises best.
#
#   This is one of the most important techniques in gradient boosting.
#   Always use it.
#
# THE eval_set PARAMETER:
#   We pass BOTH train and val so we can watch both curves.
#   If train AUC keeps rising but val AUC flatlines → overfitting.
#   Early stopping fires when val stops improving.

print("\n  Training... (watch for early stopping)")

model.fit(
    X_train, y_train,
    eval_set = [(X_train, y_train), (X_val, y_val)],
    verbose  = 50,        # print progress every 50 trees
)

best_n = model.best_iteration
print(f"\n✓ Training complete")
print(f"   Best number of trees : {best_n}")
print(f"   (stopped early from 500 max)")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE ON VALIDATION SET
# ══════════════════════════════════════════════════════════════════
#
# WHAT EACH METRIC TELLS YOU:
#
#   predict_proba → returns probability of class 1 (heart disease)
#   This is what we want — a score from 0 to 1 for each patient.
#   The threshold (default 0.5) can be adjusted depending on whether
#   you want to prioritise catching more true positives (lower threshold)
#   or reducing false alarms (higher threshold). In medical contexts
#   you almost always lower the threshold — missing a heart attack
#   is worse than a false alarm.
#
#   AUC-ROC (area under ROC curve):
#   Probability that the model ranks a random positive patient higher
#   than a random negative patient. 0.5 = random guess. 1.0 = perfect.
#   Target: >0.80 for this problem.
#
#   PR-AUC (area under Precision-Recall curve):
#   More meaningful for imbalanced data. Measures the tradeoff between
#   precision (of all patients we flagged, how many actually have disease)
#   and recall (of all patients with disease, how many did we catch).
#   Harder to achieve than AUC-ROC when classes are imbalanced.
#   A random classifier scores ~0.10 (the positive rate). Target: >0.40.
#
#   Brier Score:
#   Mean squared error between predicted probabilities and true labels.
#   0 = perfect calibration. 0.25 = random. Lower is better.
#   Tells you if your probabilities are trustworthy — a patient with
#   predicted risk 0.7 should actually have disease 70% of the time.

val_probs  = model.predict_proba(X_val)[:, 1]   # probability of heart disease
val_preds  = (val_probs >= 0.5).astype(int)       # binary prediction at 0.5 threshold

auc_roc = roc_auc_score(y_val, val_probs)
pr_auc  = average_precision_score(y_val, val_probs)
brier   = brier_score_loss(y_val, val_probs)

print(f"\n{'='*55}")
print(f"VALIDATION SET RESULTS")
print(f"{'='*55}")
print(f"  AUC-ROC  : {auc_roc:.4f}  (target > 0.80)")
print(f"  PR-AUC   : {pr_auc:.4f}  (target > 0.40, random = {y_val.mean():.2f})")
print(f"  Brier    : {brier:.4f}  (lower is better, 0 = perfect)")
print(f"\nClassification report at 0.5 threshold:")
print(classification_report(y_val, val_preds, target_names=["No disease","Heart disease"]))

# ══════════════════════════════════════════════════════════════════
# STEP 6 — PLOT EVALUATION CURVES
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Evaluation — Validation Set", fontsize=13, fontweight="bold")

# ── ROC Curve ──────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_val, val_probs)
axes[0].plot(fpr, tpr, color="#378ADD", lw=2, label=f"AUC = {auc_roc:.3f}")
axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random (0.500)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Precision-Recall Curve ─────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_val, val_probs)
axes[1].plot(recall, precision, color="#1D9E75", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[1].axhline(y=y_val.mean(), color="k", lw=1, linestyle="--", alpha=0.4,
                label=f"Random ({y_val.mean():.2f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ── Calibration Curve ──────────────────────────────────────────
# Are predicted probabilities trustworthy?
# Perfect calibration = diagonal line (predicted 0.7 → actually 70% of cases)
frac_pos, mean_pred = calibration_curve(y_val, val_probs, n_bins=10)
axes[2].plot(mean_pred, frac_pos, "s-", color="#D85A30", lw=2, label="Model")
axes[2].plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Perfect calibration")
axes[2].set_xlabel("Mean predicted probability")
axes[2].set_ylabel("Fraction of positives")
axes[2].set_title("Calibration Curve")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("evaluation_curves.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Evaluation curves saved → evaluation_curves.png")

# ══════════════════════════════════════════════════════════════════
# STEP 7 — SHAP VALUES (explainability)
# ══════════════════════════════════════════════════════════════════
#
# WHAT IS SHAP?
#   Shapley values come from game theory. The idea: each feature is
#   a "player" in a game, and SHAP fairly distributes the model's
#   prediction among all players based on their contribution.
#
#   For a patient with predicted risk 0.73:
#     SHAP tells you: "LDL drove risk up by 0.20, age drove it up
#     by 0.15, HDL pulled it down by 0.08, BMI added 0.06..." etc.
#
#   WHY IT'S NON-NEGOTIABLE FOR MEDICAL ML:
#   - Doctors won't act on a black-box score
#   - Regulators require explainability for clinical AI
#   - It's how you catch bugs: if "source" (which dataset the row
#     came from) appears as a top SHAP feature, your model is
#     cheating — learning dataset artifacts, not clinical patterns
#
#   TreeExplainer is the fast version specifically for tree models
#   like XGBoost. It computes exact SHAP values in seconds.

print("\n  Computing SHAP values (takes ~30 seconds)...")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# ── SHAP Summary Plot (most important) ─────────────────────────
# Each dot is one patient.
# X position = SHAP value (how much this feature pushed the risk score)
# Colour     = actual feature value (red=high, blue=low)
# Y position = feature (sorted by overall importance, top = most important)
#
# How to read it:
#   If high BMI (red dots) → far right → high BMI raises heart disease risk ✓
#   If low cholesterol (blue) → far left → low cholesterol lowers risk ✓
#   If a feature's dots are all near 0 → that feature barely matters

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_val, feature_names=FEATURE_COLS,
                  show=False, max_display=20)
plt.title("SHAP Feature Importance — Top 20 Features", fontsize=12)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
print("✓ SHAP summary plot saved → shap_summary.png")

# ── SHAP Bar Plot (mean absolute impact) ───────────────────────
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_val, feature_names=FEATURE_COLS,
                  plot_type="bar", show=False, max_display=20)
plt.title("Mean |SHAP| — Overall Feature Importance", fontsize=12)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
print("✓ SHAP importance plot saved → shap_importance.png")

# ══════════════════════════════════════════════════════════════════
# STEP 8 — FINAL TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════
#
# ONLY RUN THIS ONCE — after all tuning is done.
# The test set is Framingham: a different study, different population,
# different time period. This is the honest out-of-distribution score.
# If it's much worse than validation, the model overfit to BRFSS data.

print(f"\n{'='*55}")
print(f"FINAL TEST SET RESULTS (Framingham — out-of-distribution)")
print(f"{'='*55}")

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

test_auc = roc_auc_score(y_test, test_probs)
test_pr  = average_precision_score(y_test, test_probs)
test_brier = brier_score_loss(y_test, test_probs)

print(f"  AUC-ROC  : {test_auc:.4f}")
print(f"  PR-AUC   : {test_pr:.4f}")
print(f"  Brier    : {test_brier:.4f}")
print(f"\nClassification report at 0.5 threshold:")
print(classification_report(y_test, test_preds, target_names=["No disease","Heart disease"]))

# ── Save the model ──────────────────────────────────────────────
model.save_model("heart_xgboost_v1.json")
print(f"\n✓ Model saved → heart_xgboost_v1.json")
print(f"  (reload anytime with: model = xgb.XGBClassifier(); model.load_model('heart_xgboost_v1.json'))")

print(f"\n{'='*55}")
print(f"TRAINING COMPLETE")
print(f"  Val  AUC-ROC: {auc_roc:.4f}  |  PR-AUC: {pr_auc:.4f}")
print(f"  Test AUC-ROC: {test_auc:.4f}  |  PR-AUC: {test_pr:.4f}")
print(f"{'='*55}")
print(f"\nNext step: read shap_summary.png and check which features")
print(f"the model is actually using. That tells you what to improve.")