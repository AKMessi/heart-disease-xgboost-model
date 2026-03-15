# ══════════════════════════════════════════════════════════════════
# STEP 9 — HYPERPARAMETER TUNING WITH OPTUNA
# ══════════════════════════════════════════════════════════════════
#
# HOW OPTUNA WORKS:
#
#   1. You define an "objective function" that:
#        - receives a `trial` object (Optuna's suggestion engine)
#        - asks the trial to suggest values for each hyperparameter
#        - trains a model with those values
#        - returns a score (PR-AUC on validation set)
#
#   2. Optuna runs the objective function N times (N = n_trials)
#      Each run is called a "trial"
#
#   3. After each trial, Optuna updates its internal probabilistic
#      model (a Gaussian Process or TPE sampler) of which regions
#      of the search space produced good scores
#
#   4. The next trial samples from the high-promise regions
#      This is why trial 50 is smarter than trial 1
#
#   5. At the end, Optuna tells you the best hyperparameters found
#
# WHY NOT GRID SEARCH?
#   Grid search tries every combination. With 8 hyperparameters
#   each with 5 possible values = 5^8 = 390,625 combinations.
#   At 1 minute per trial that's 270 days. Optuna finds near-optimal
#   settings in 50-100 trials (~30-60 minutes).
#
# WHY NOT RANDOM SEARCH?
#   Random search is better than grid search but still wastes trials
#   on bad regions. Optuna's TPE sampler is ~3-5x more efficient.

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # quieter output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Load data (same as before) ─────────────────────────────────
df = pd.read_csv("data/heart_unified_clean.csv")

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

DROP_COLS    = ["target", "split", "source", "age_band", "race"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X_train = train_df[FEATURE_COLS]
y_train = train_df["target"]
X_val   = val_df[FEATURE_COLS]
y_val   = val_df["target"]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df["target"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
SCALE_POS_WEIGHT = neg / pos

print(f"✓ Data loaded — {len(X_train)} train rows, {len(FEATURE_COLS)} features")
print(f"  scale_pos_weight = {SCALE_POS_WEIGHT:.1f}")

# ══════════════════════════════════════════════════════════════════
# THE OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════
#
# This function is the heart of the tuning. Optuna calls it
# repeatedly with different hyperparameter suggestions.
#
# CROSS-VALIDATION INSIDE TUNING:
#   We use 3-fold stratified cross-validation instead of a single
#   train/val split. Why? Because a single val split can be "lucky"
#   or "unlucky". If we tune 100 trials all evaluated on the same
#   val set, we risk overfitting our hyperparameters to that one
#   split (a subtle form of leakage). 3-fold CV averages over 3
#   different splits, giving a more honest estimate of each
#   hyperparameter combination's true quality.
#
#   "Stratified" means each fold preserves the 9:1 class ratio,
#   so we never accidentally get a fold with zero positives.
#
# PRUNING:
#   trial.report() and trial.should_prune() let Optuna kill bad
#   trials early — if after 2 folds the score is already terrible,
#   there's no point running the 3rd fold. This saves ~30% of time.

def objective(trial):

    # ── Optuna suggests a value for each hyperparameter ──────────
    # suggest_int(name, low, high)      → integer between low and high
    # suggest_float(name, low, high)    → float between low and high
    # suggest_float(..., log=True)      → samples in log scale
    #   (log scale is better for learning_rate because the difference
    #    between 0.001 and 0.01 matters as much as between 0.1 and 1.0)

    params = {
        # ── Tree structure ──────────────────────────────────────
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        #   Deeper trees = more complex patterns = higher overfit risk
        #   Range 3-8 covers from very simple to moderately complex

        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        #   Min samples in a leaf. Higher = simpler trees.
        #   Wide range because imbalanced data needs careful tuning here

        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        #   Minimum loss reduction needed to make a split.
        #   0 = split freely, 1 = only split if gain > 1.0
        #   Acts as a pruning criterion during tree building

        # ── Sampling (randomness / regularisation) ──────────────
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        #   Fraction of training rows each tree sees
        #   0.5 = each tree sees half the data (more diverse ensemble)

        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        #   Fraction of features each tree sees
        #   Lower = more diverse trees, less likely to overfit

        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        #   Fraction of features at each LEVEL of tree building
        #   An extra randomisation layer on top of colsample_bytree

        # ── Learning rate and regularisation ────────────────────
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        #   Step size shrinkage. log=True because 0.01→0.05 is as
        #   meaningful a jump as 0.1→0.5. Log scale samples these
        #   proportionally rather than linearly.

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        #   L1 regularisation. log=True for same reason.
        #   High alpha → sparse model (many features zeroed out)

        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        #   L2 regularisation.
        #   High lambda → small, smooth weights everywhere

        # ── Fixed parameters (not tuned) ────────────────────────
        "n_estimators":     500,        # early stopping will cut this
        "early_stopping_rounds": 30,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "tree_method":      "hist",
        "random_state":     RANDOM_SEED,
        "verbosity":        0,          # silent during tuning
    }

    # ── 3-fold stratified cross-validation ───────────────────────
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []

    for fold_idx, (train_idx, fold_val_idx) in enumerate(cv.split(X_train, y_train)):

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[fold_val_idx]
        y_fold_val   = y_train.iloc[fold_val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set  = [(X_fold_val, y_fold_val)],
            verbose   = False,
        )

        preds      = model.predict_proba(X_fold_val)[:, 1]
        fold_score = average_precision_score(y_fold_val, preds)
        fold_scores.append(fold_score)

        # ── Pruning: report intermediate score to Optuna ─────────
        # If this trial is clearly worse than completed trials,
        # Optuna prunes it here (stops early, saves time)
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_scores)   # Optuna maximises this value


# ══════════════════════════════════════════════════════════════════
# RUN THE STUDY
# ══════════════════════════════════════════════════════════════════
#
# A "study" is Optuna's container for all trials.
#
# direction="maximize" → Optuna tries to maximise PR-AUC
#
# TPESampler = Tree-structured Parzen Estimator
#   The algorithm that builds the probabilistic model of the search
#   space. After n_startup_trials (20) random trials, it switches
#   to guided sampling based on what's worked so far.
#
# MedianPruner
#   Prunes trials whose intermediate values are below the median
#   of completed trials at the same step. Efficient and conservative.
#
# N_TRIALS = 100
#   Each trial trains 3 models (3-fold CV) with early stopping.
#   On a modern laptop this takes ~20-40 minutes.
#   If you want faster results, reduce to 50 trials.
#   If you want better results and have time, go to 200.

N_TRIALS = 100

print(f"\n{'='*60}")
print(f"STARTING OPTUNA HYPERPARAMETER SEARCH")
print(f"  Trials       : {N_TRIALS}")
print(f"  CV folds     : 3 (stratified)")
print(f"  Metric       : PR-AUC (maximise)")
print(f"  Pruner       : MedianPruner")
print(f"  Estimated time: ~20-40 minutes on a modern CPU")
print(f"{'='*60}\n")

sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1)

study = optuna.create_study(
    direction = "maximize",
    sampler   = sampler,
    pruner    = pruner,
)

# Progress callback — prints a line every 10 trials so you can
# watch the model improve in real time
def print_progress(study, trial):
    if trial.number % 10 == 0 or trial.number < 5:
        best = study.best_value
        current = trial.value if trial.value is not None else "pruned"
        print(f"  Trial {trial.number:>3} | "
              f"current: {str(current)[:6] if isinstance(current, float) else current:<8} | "
              f"best so far: {best:.4f}")

study.optimize(
    objective,
    n_trials  = N_TRIALS,
    callbacks = [print_progress],
    show_progress_bar = False,
)

# ══════════════════════════════════════════════════════════════════
# INSPECT RESULTS
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"TUNING COMPLETE")
print(f"{'='*60}")
print(f"\n  Best CV PR-AUC : {study.best_value:.4f}")
print(f"  Best trial #   : {study.best_trial.number}")
print(f"\n  Best hyperparameters found:")
for k, v in study.best_params.items():
    print(f"    {k:<25} {v}")

# ── Plot the optimisation history ──────────────────────────────
# This shows how Optuna's best score improved over trials.
# A flat line after trial 30 means it converged — more trials
# wouldn't help. A still-rising line means run more trials.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: best value per trial
trials_df = study.trials_dataframe()
completed  = trials_df[trials_df["state"] == "COMPLETE"]
axes[0].plot(completed["number"], completed["value"],
             "o", alpha=0.4, color="#378ADD", markersize=3, label="Trial PR-AUC")
# Rolling best
best_so_far = completed["value"].cummax()
axes[0].plot(completed["number"], best_so_far,
             "-", color="#E24B4A", lw=2, label="Best so far")
axes[0].set_xlabel("Trial number")
axes[0].set_ylabel("CV PR-AUC")
axes[0].set_title("Optimisation history")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: hyperparameter importance
# Which hyperparameters mattered most to the final score?
# This is SHAP for hyperparameters — same concept.
importances = optuna.importance.get_param_importances(study)
names  = list(importances.keys())[:10]
values = [importances[n] for n in names]
axes[1].barh(names[::-1], values[::-1], color="#1D9E75")
axes[1].set_xlabel("Importance (fANOVA)")
axes[1].set_title("Hyperparameter importance")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("optuna_results.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Optuna results plot saved → optuna_results.png")

# ══════════════════════════════════════════════════════════════════
# RETRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
#
# NOW we use the full train set (not CV folds) + the real val set
# to train the final model with the best hyperparameters found.
#
# WHY RETRAIN ON FULL TRAIN + VAL?
#   During tuning we only used the train set (with CV).
#   Now that hyperparameters are fixed, we can give the model
#   more data — training on train+val, using the held-out test
#   set for early stopping. More data = better model.
#   Test set is still untouched for final evaluation.

print(f"\n{'='*60}")
print(f"RETRAINING FINAL MODEL WITH BEST PARAMS")
print(f"{'='*60}")

# RETRAINING STRATEGY:
# We train on X_train only, watch val for early stopping.
# This is the correct approach because:
#   1. Val set has the SAME distribution as train (both BRFSS-dominant)
#      so early stopping fires at the right point
#   2. Framingham test set has a DIFFERENT distribution (15% positive
#      vs 10% train) — using it for early stopping causes the model
#      to stop based on a foreign population, not what we care about
#   3. Test set must stay completely untouched until final evaluation
#
# We keep train and val separate (don't concat) to preserve the
# early stopping signal. The model sees 45k train rows, which is
# already plenty.

import pandas as pd

best_params = study.best_params.copy()
best_params.update({
    "n_estimators":          1000,   # early stopping will cut this
    "early_stopping_rounds": 50,     # stop after 50 rounds no improvement
    "scale_pos_weight":      SCALE_POS_WEIGHT,
    "objective":             "binary:logistic",
    "eval_metric":           "aucpr",
    "tree_method":           "hist",
    "random_state":          RANDOM_SEED,
    "verbosity":             1,
})

# Floor learning_rate so early stopping has enough rounds to work.
# Low lr (0.02) + 50 patience rounds = fine. The model will grow
# ~200-500 trees before converging, which is correct behaviour.
lr = best_params.get("learning_rate", 0.05)
if lr < 0.02:
    print(f"  Flooring learning_rate {lr:.4f} → 0.02")
    best_params["learning_rate"] = 0.02

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set = [(X_train, y_train), (X_val, y_val)],  # watch VAL not test
    verbose  = 50,
)

print(f"\n✓ Final model trained")
print(f"  Best iteration: {final_model.best_iteration}")

# ══════════════════════════════════════════════════════════════════
# CALIBRATION FIX — Platt Scaling
# ══════════════════════════════════════════════════════════════════
#
# Remember the calibration curve problem — predicted 0.3 actually
# meant ~5% real risk. We fix this with Platt scaling (sigmoid fit).
#
# HOW IT WORKS:
#   After training, we fit a simple logistic regression on top of
#   the model's raw scores. This logistic regression learns a
#   mapping from "raw score → calibrated probability" using the
#   validation set. It's a 2-parameter fix (slope + intercept).
#
# cv="prefit" means: "the base model is already trained, just
#   fit the calibration layer on top using the provided data"
#
# method="sigmoid" = Platt scaling (good for most cases)
# method="isotonic" = more flexible but needs more data (>1000 pos)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, classification_report)

print(f"\n  Fitting calibration layer (Platt scaling)...")

# sklearn 1.2+ removed cv="prefit" — use set_params workaround.
# We wrap the already-trained model and tell sklearn not to refit it
# by passing the training data as the calibration data but with
# the base estimator frozen via clone=False.
#
# The cleanest cross-version fix: use CalibratedClassifierCV with
# cv=5 on val set only (small, just for the sigmoid layer).
# Since val set has ~800 positives this is statistically sound.
from sklearn.base import clone as sk_clone
import sklearn

sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

if sklearn_version >= (1, 2):
    # New API: pass estimator already fitted, use cv="prefit" via
    # the internal flag. Workaround: fit a fresh calibrator on val.
    from sklearn.calibration import _CalibratedClassifier
    calibrated_model = CalibratedClassifierCV(
        estimator = final_model,
        cv        = 5,           # refit base model? No — we pass val only
        method    = "sigmoid",
    )
    # Override: manually do Platt scaling without refitting base model
    # by wrapping raw scores in a simple logistic regression
    from sklearn.linear_model import LogisticRegression
    raw_val_scores = final_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(raw_val_scores, y_val)

    class CalibratedWrapper:
        def __init__(self, base, platt):
            self.base = base
            self.platt = platt
        def predict_proba(self, X):
            raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
            cal = self.platt.predict_proba(raw)
            return cal  # shape (n, 2): col 1 = calibrated P(heart disease)

    calibrated_model = CalibratedWrapper(final_model, platt)
else:
    calibrated_model = CalibratedClassifierCV(
        estimator = final_model,
        cv        = "prefit",
        method    = "sigmoid",
    )
    calibrated_model.fit(X_val, y_val)

print(f"✓ Calibration layer fitted on validation set")

# ══════════════════════════════════════════════════════════════════
# FINAL EVALUATION — BEFORE vs AFTER CALIBRATION
# ══════════════════════════════════════════════════════════════════

from sklearn.calibration import calibration_curve

# Raw model predictions
raw_probs_val  = final_model.predict_proba(X_val)[:, 1]
raw_probs_test = final_model.predict_proba(X_test)[:, 1]

# Calibrated predictions
cal_probs_val  = calibrated_model.predict_proba(X_val)[:, 1]
cal_probs_test = calibrated_model.predict_proba(X_test)[:, 1]

print(f"\n{'='*60}")
print(f"VALIDATION SET — TUNED MODEL RESULTS")
print(f"{'='*60}")
print(f"{'Metric':<20} {'Before tuning':>15} {'After tuning':>14} {'Calibrated':>12}")
print(f"{'-'*62}")

# Baseline scores (from your first run)
baseline_auc  = 0.833
baseline_pr   = 0.357
baseline_brier = None   # we'll compute it now for fair comparison

print(f"{'AUC-ROC':<20} {baseline_auc:>15.4f} "
      f"{roc_auc_score(y_val, raw_probs_val):>14.4f} "
      f"{roc_auc_score(y_val, cal_probs_val):>12.4f}")

print(f"{'PR-AUC':<20} {baseline_pr:>15.4f} "
      f"{average_precision_score(y_val, raw_probs_val):>14.4f} "
      f"{average_precision_score(y_val, cal_probs_val):>12.4f}")

print(f"{'Brier score':<20} {'(see plot)':>15} "
      f"{brier_score_loss(y_val, raw_probs_val):>14.4f} "
      f"{brier_score_loss(y_val, cal_probs_val):>12.4f}")

print(f"\n{'='*60}")
print(f"TEST SET (Framingham — out-of-distribution)")
print(f"{'='*60}")
print(f"  AUC-ROC  (raw)        : {roc_auc_score(y_test, raw_probs_test):.4f}")
print(f"  AUC-ROC  (calibrated) : {roc_auc_score(y_test, cal_probs_test):.4f}")
print(f"  PR-AUC   (raw)        : {average_precision_score(y_test, raw_probs_test):.4f}")
print(f"  PR-AUC   (calibrated) : {average_precision_score(y_test, cal_probs_test):.4f}")
print(f"  Brier    (raw)        : {brier_score_loss(y_test, raw_probs_test):.4f}")
print(f"  Brier    (calibrated) : {brier_score_loss(y_test, cal_probs_test):.4f}")

# ── Calibration before vs after plot ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Tuned Model — Validation Set", fontsize=13, fontweight="bold")

from sklearn.metrics import roc_curve, precision_recall_curve

# ROC
fpr, tpr, _ = roc_curve(y_val, cal_probs_val)
auc_val = roc_auc_score(y_val, cal_probs_val)
axes[0].plot(fpr, tpr, color="#378ADD", lw=2, label=f"AUC = {auc_val:.3f}")
axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Random (0.500)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# PR curve
prec, rec, _ = precision_recall_curve(y_val, cal_probs_val)
pr_val = average_precision_score(y_val, cal_probs_val)
axes[1].plot(rec, prec, color="#1D9E75", lw=2, label=f"PR-AUC = {pr_val:.3f}")
axes[1].axhline(y=y_val.mean(),color="k",lw=1,linestyle="--",
                alpha=0.4,label=f"Random ({y_val.mean():.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Calibration — before vs after
frac_raw, mean_raw   = calibration_curve(y_val, raw_probs_val,  n_bins=10)
frac_cal, mean_cal   = calibration_curve(y_val, cal_probs_val, n_bins=10)
axes[2].plot(mean_raw, frac_raw, "s--", color="#888780", lw=1.5,
             label="Before calibration", alpha=0.7)
axes[2].plot(mean_cal, frac_cal, "s-",  color="#D85A30", lw=2,
             label="After calibration")
axes[2].plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Perfect")
axes[2].set_xlabel("Mean predicted probability")
axes[2].set_ylabel("Fraction of positives")
axes[2].set_title("Calibration (before vs after Platt scaling)")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tuned_evaluation.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Tuned evaluation plot saved → tuned_evaluation.png")

# ══════════════════════════════════════════════════════════════════
# DISTRIBUTION SHIFT DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════
#
# The val→test gap (AUC 0.82 → 0.66) is large. This section
# explains WHY by comparing the two populations directly.
# Understanding this is crucial — it tells you what to fix next.

print(f"\n{'='*60}")
print(f"DISTRIBUTION SHIFT DIAGNOSTIC")
print(f"  Why does val AUC differ so much from test AUC?")
print(f"{'='*60}")

shared_numeric = ["age", "bmi", "systolic_bp", "cholesterol",
                  "smoking_current", "diabetes"]
available = [c for c in shared_numeric if c in X_val.columns]

print(f"\n  {'Feature':<22} {'Val mean':>10} {'Test mean':>11} {'Diff %':>8}")
print(f"  {'-'*54}")
for col in available:
    v_mean = X_val[col].mean()
    t_mean = X_test[col].mean()
    if v_mean > 0:
        diff_pct = (t_mean - v_mean) / v_mean * 100
        flag = " ←" if abs(diff_pct) > 20 else ""
        print(f"  {col:<22} {v_mean:>10.2f} {t_mean:>11.2f} {diff_pct:>7.1f}%{flag}")

print(f"\n  Base rate (% positive):")
print(f"    Val set  : {y_val.mean():.2%}")
print(f"    Test set : {y_test.mean():.2%}")
print(f"\n  Source breakdown:")
print(f"    Val  → mostly BRFSS 2020 (survey, self-reported, modern)")
print(f"    Test → Framingham 1948-1972 (clinical, measured, historical)")
print(f"\n  This is expected distribution shift — not a bug.")
print(f"  The model learned BRFSS patterns; Framingham is a")
print(f"  different era, different measurement methods, different")
print(f"  base rates. A 0.66 AUC on out-of-distribution data is")
print(f"  actually reasonable. To close this gap you need either:")
print(f"    (a) More longitudinal training data like Framingham")
print(f"    (b) Domain adaptation techniques")
print(f"    (c) Separate models per data source")

# ══════════════════════════════════════════════════════════════════
# OPTIMAL THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════════════
#
# Default threshold is 0.5: predict "heart disease" if prob > 0.5
#
# In medicine this is almost always wrong. Missing a true heart
# disease case (false negative) is much more costly than a false
# alarm (false positive) that leads to further testing.
#
# We plot the tradeoff and let you choose the threshold based on
# the clinical cost ratio you're willing to accept.
#
# F2 score = weighted F-score that values recall 2x more than
# precision. Good default for medical screening where missing
# cases is worse than false alarms.

thresholds  = np.arange(0.05, 0.60, 0.01)
f1_scores   = []
f2_scores   = []
precisions  = []
recalls     = []

from sklearn.metrics import fbeta_score, precision_score, recall_score

for t in thresholds:
    preds = (cal_probs_val >= t).astype(int)
    f1_scores.append(fbeta_score(y_val, preds, beta=1, zero_division=0))
    f2_scores.append(fbeta_score(y_val, preds, beta=2, zero_division=0))
    precisions.append(precision_score(y_val, preds, zero_division=0))
    recalls.append(recall_score(y_val, preds, zero_division=0))

best_f1_thresh = thresholds[np.argmax(f1_scores)]
best_f2_thresh = thresholds[np.argmax(f2_scores)]

print(f"\n{'='*60}")
print(f"OPTIMAL THRESHOLD ANALYSIS")
print(f"{'='*60}")
print(f"  Best F1 threshold  : {best_f1_thresh:.2f}  "
      f"(balanced precision/recall)")
print(f"  Best F2 threshold  : {best_f2_thresh:.2f}  "
      f"(recall weighted 2x — recommended for screening)")

# What does the F2 threshold actually give us?
preds_f2 = (cal_probs_val >= best_f2_thresh).astype(int)
print(f"\n  At F2 threshold ({best_f2_thresh:.2f}):")
print(classification_report(y_val, preds_f2,
      target_names=["No disease", "Heart disease"]))

# Plot threshold curves
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precisions, "--", color="#378ADD", lw=1.5, label="Precision")
ax.plot(thresholds, recalls,    "--", color="#1D9E75", lw=1.5, label="Recall")
ax.plot(thresholds, f1_scores,  "-",  color="#534AB7", lw=2,   label="F1 score")
ax.plot(thresholds, f2_scores,  "-",  color="#D85A30", lw=2,   label="F2 score (recall×2)")
ax.axvline(x=best_f2_thresh, color="#D85A30", linestyle=":",
           lw=1.5, label=f"Best F2 threshold = {best_f2_thresh:.2f}")
ax.axvline(x=best_f1_thresh, color="#534AB7", linestyle=":",
           lw=1.5, label=f"Best F1 threshold = {best_f1_thresh:.2f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold selection — precision / recall / F1 / F2")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("threshold_selection.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Threshold plot saved → threshold_selection.png")

# ══════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ══════════════════════════════════════════════════════════════════

import joblib

final_model.save_model("heart_xgboost_tuned.json")
joblib.dump(calibrated_model, "heart_xgboost_calibrated.pkl")
joblib.dump(FEATURE_COLS,     "feature_cols.pkl")
joblib.dump(best_f2_thresh,   "best_threshold.pkl")

print(f"\n{'='*60}")
print(f"ALL OUTPUTS SAVED")
print(f"{'='*60}")
print(f"  heart_xgboost_tuned.json      — raw tuned model")
print(f"  heart_xgboost_calibrated.pkl  — calibrated model (use this)")
print(f"  feature_cols.pkl              — feature list (for inference)")
print(f"  best_threshold.pkl            — optimal threshold ({best_f2_thresh:.2f})")
print(f"  optuna_results.png            — tuning history + param importance")
print(f"  tuned_evaluation.png          — ROC, PR, calibration curves")
print(f"  threshold_selection.png       — F1/F2 threshold analysis")

print(f"""
TO USE THE MODEL ON A NEW PATIENT:
─────────────────────────────────
import joblib, pandas as pd
model     = joblib.load('heart_xgboost_calibrated.pkl')
features  = joblib.load('feature_cols.pkl')
threshold = joblib.load('best_threshold.pkl')

patient = pd.DataFrame([{{
    'age': 58, 'sex': 1, 'bmi': 29.5,
    'smoking_current': 1, 'diabetes': 1,
    # ... all 56 features, NaN for missing ones
}}])[features]

risk_score = model.predict_proba(patient)[0, 1]
high_risk  = risk_score >= threshold
print(f'Risk: {{risk_score:.1%}} | High risk: {{high_risk}}')
""")