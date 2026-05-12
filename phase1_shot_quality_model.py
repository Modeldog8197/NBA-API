"""
Basketball Shot Quality Model — Phase 1
Expected Field Goal Percentage (xFG%) using XGBoost

Goal: Predict the probability (0–1) that a shot will go in,
      given spatial and contextual features.
"""

# ─────────────────────────────────────────────
# STEP 0 — Install dependencies (run once)
# ─────────────────────────────────────────────
# Uncomment and run this block the first time:
#
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install",
#     "nba_api", "pandas", "xgboost", "scikit-learn"])

# ─────────────────────────────────────────────
# STEP 1 — Imports
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import players

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import time

# ─────────────────────────────────────────────
# STEP 2 — Fetch shot data
# ─────────────────────────────────────────────
# We target Stephen Curry (high volume, diverse shot selection).
# The nba_api returns ALL shots for a given player+season by default,
# so we slice to the first 5,000 rows after fetching.
#
# Season ID format:  "2023-24" for the 2023–24 season.
# Context filter:    "FGA" = all field goal attempts (made + missed).

PLAYER_NAME  = "Stephen Curry"
SEASON_ID    = "2023-24"
SEASON_TYPE  = "Regular Season"
TARGET_ROWS  = 5_000

print(f"Looking up player ID for '{PLAYER_NAME}'...")
player_info = players.find_players_by_full_name(PLAYER_NAME)
if not player_info:
    raise ValueError(f"Player '{PLAYER_NAME}' not found in nba_api static data.")

PLAYER_ID = player_info[0]["id"]
print(f"  → Player ID: {PLAYER_ID}")

print(f"\nFetching ShotChartDetail for {SEASON_ID} {SEASON_TYPE}...")
print("  (nba_api has a rate-limit; this may take ~10 seconds)")

# Small courtesy sleep to avoid hammering the NBA stats endpoint
time.sleep(1)

shot_chart = ShotChartDetail(
    team_id=0,                     # 0 = all teams (player may have been traded)
    player_id=PLAYER_ID,
    season_nullable=SEASON_ID,
    season_type_all_star=SEASON_TYPE,
    context_measure_simple="FGA",  # Field Goal Attempts
)

df_raw = shot_chart.get_data_frames()[0]
print(f"  → Fetched {len(df_raw):,} total shot records")

# Keep up to TARGET_ROWS
df = df_raw.head(TARGET_ROWS).copy()
print(f"  → Using {len(df):,} records for this phase\n")

# ─────────────────────────────────────────────
# STEP 3 — Explore available columns
# ─────────────────────────────────────────────
print("Available columns:")
print(df.columns.tolist(), "\n")

# Key columns we care about:
#   LOC_X          — horizontal distance from basket (tenths of feet, + = right)
#   LOC_Y          — vertical distance from basket   (tenths of feet, + = toward half-court)
#   SHOT_DISTANCE  — Euclidean distance to basket (feet, integer)
#   SHOT_TYPE      — "2PT Field Goal" | "3PT Field Goal"
#   SHOT_MADE_FLAG — 1 = made, 0 = missed  ← our target label

# ─────────────────────────────────────────────
# STEP 4 — Clean the data
# ─────────────────────────────────────────────
REQUIRED_COLS = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE", "SHOT_MADE_FLAG"]

# Drop rows with any NaN in the columns we need
df = df.dropna(subset=REQUIRED_COLS)

# Convert numeric columns (they come back as objects in some nba_api versions)
for col in ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_MADE_FLAG"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=REQUIRED_COLS)   # drop any rows that failed conversion
df = df.reset_index(drop=True)

print(f"After cleaning: {len(df):,} rows remain")
print(f"Made / Missed split:\n{df['SHOT_MADE_FLAG'].value_counts()}\n")

# ─────────────────────────────────────────────
# STEP 5 — Feature engineering: SHOT ANGLE
# ─────────────────────────────────────────────
#
# ┌─ MATH EXPLANATION ─────────────────────────────────────────────────────────┐
# │                                                                             │
# │  The basket is at the origin (0, 0) in the NBA coordinate system.          │
# │  A shot attempt is recorded at coordinates (LOC_X, LOC_Y).                 │
# │                                                                             │
# │  We want the ANGLE of the shot relative to the basket — specifically,      │
# │  the angle between the shot vector and the positive Y-axis (straight        │
# │  on from the basket, i.e. the center of the lane).                         │
# │                                                                             │
# │  We use the two-argument arctangent:                                        │
# │                                                                             │
# │      θ = arctan2(LOC_X, LOC_Y)       [in radians]                         │
# │                                                                             │
# │  Why arctan2 instead of arctan(LOC_X / LOC_Y)?                             │
# │    • Plain arctan(x/y) collapses the angle to the range (-π/2, π/2) and   │
# │      loses quadrant information (a shot from the left corner and the right  │
# │      corner can give the same value).                                       │
# │    • arctan2 uses BOTH the sign of LOC_X and LOC_Y to return the full      │
# │      angle in (-π, π], preserving which side of the court the shot came    │
# │      from.                                                                  │
# │                                                                             │
# │  We also keep the ABSOLUTE angle (0–π) as a separate feature:              │
# │    • A corner 3 from the right (θ ≈ +1.5 rad) and from the left           │
# │      (θ ≈ -1.5 rad) have the same shot difficulty — the abs value           │
# │      captures this symmetry.                                                │
# │    • The signed angle still goes in so the model can detect any subtle     │
# │      left/right asymmetry in this player's shot chart.                      │
# │                                                                             │
# │  Intuition:                                                                 │
# │    • θ = 0      → straight-on from the elbow / paint (easiest angle)       │
# │    • θ = ±π/2   → corner shot (most acute angle, shortest 3-point arc)     │
# │    • Angle matters because shot difficulty changes as you move away from    │
# │      the center of the basket — corner 3s go in at a different rate than    │
# │      top-of-the-key 3s at the same distance.                               │
# └─────────────────────────────────────────────────────────────────────────────┘

# arctan2(x, y) — note: arguments are (x, y), NOT (y, x),
# because we measure angle FROM the Y-axis, not from the X-axis.
df["SHOT_ANGLE"]     = np.arctan2(df["LOC_X"], df["LOC_Y"])    # signed,  (-π, π]
df["SHOT_ANGLE_ABS"] = df["SHOT_ANGLE"].abs()                   # absolute, [0, π]

print("Shot angle feature sample:")
print(df[["LOC_X", "LOC_Y", "SHOT_ANGLE", "SHOT_ANGLE_ABS"]].head(5), "\n")

# ─────────────────────────────────────────────
# STEP 6 — Encode categorical feature
# ─────────────────────────────────────────────
# SHOT_TYPE is a string: "2PT Field Goal" or "3PT Field Goal"
# XGBoost needs numerics, so we label-encode it (0 / 1).

le = LabelEncoder()
df["SHOT_TYPE_ENC"] = le.fit_transform(df["SHOT_TYPE"])
print("Shot type encoding:", dict(zip(le.classes_, le.transform(le.classes_))), "\n")

# ─────────────────────────────────────────────
# STEP 7 — Assemble feature matrix and target
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "SHOT_TYPE_ENC",
    "SHOT_ANGLE",
    "SHOT_ANGLE_ABS",
]

X = df[FEATURE_COLS].values
y = df["SHOT_MADE_FLAG"].values.astype(int)

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution — Made: {y.sum()} ({y.mean():.1%}) | Missed: {(1-y).sum()}\n")

# ─────────────────────────────────────────────
# STEP 8 — Train / test split
# ─────────────────────────────────────────────
# Stratify ensures the made/missed ratio is preserved in both splits.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train set: {X_train.shape[0]:,} samples")
print(f"Test  set: {X_test.shape[0]:,} samples\n")

# ─────────────────────────────────────────────
# STEP 9 — XGBoost training
# ─────────────────────────────────────────────
# objective="binary:logistic" → outputs probabilities (0–1) via logistic function.
# eval_metric="logloss"       → monitors log loss on the eval set during training.
# use_label_encoder=False     → suppresses a deprecation warning in newer XGBoost.

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

print("Training XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,          # print eval metric every 50 rounds
)
print()

# ─────────────────────────────────────────────
# STEP 10 — Evaluate: Log Loss and ROC-AUC
# ─────────────────────────────────────────────
#
# ┌─ WHY LOG LOSS INSTEAD OF ACCURACY? ────────────────────────────────────────┐
# │                                                                             │
# │  Our model outputs a PROBABILITY (e.g. 0.43) — not just "made" or         │
# │  "missed". The quality of that probability estimate is exactly what we     │
# │  care about for an xFG% model.                                             │
# │                                                                             │
# │  Accuracy only cares whether p > 0.5 → "made" matches the true label.     │
# │  It gives the same score to two predictions:                               │
# │    • p = 0.51 (barely confident, correctly predicted made)                 │
# │    • p = 0.99 (very confident, correctly predicted made)                   │
# │  Those two predictions are NOT equally good — 0.99 is overconfident if    │
# │  the true shot-making rate is ~45%.                                        │
# │                                                                             │
# │  Log Loss formula (per sample):                                             │
# │                                                                             │
# │    L = -(y · log(p) + (1 - y) · log(1 - p))                               │
# │                                                                             │
# │    where y ∈ {0, 1} is the true label and p ∈ (0, 1) is the predicted     │
# │    probability.                                                             │
# │                                                                             │
# │  Key properties:                                                            │
# │    • L → 0   when the model is confident AND correct     (ideal)           │
# │    • L → ∞   when the model is confident AND wrong       (catastrophic)    │
# │    • The log function means wrong high-confidence predictions are           │
# │      penalised EXPONENTIALLY harder than wrong low-confidence ones.        │
# │    • A "random" model that always predicts the base rate has                │
# │      L = -log(base_rate) ≈ 0.69 for a 50/50 split.                        │
# │                                                                             │
# │  ROC-AUC measures ranking quality — "does the model rank made shots        │
# │  higher than missed shots?" — independent of the probability scale.        │
# │  Together, Log Loss + AUC tell you both CALIBRATION and DISCRIMINATION.   │
# └─────────────────────────────────────────────────────────────────────────────┘

y_prob = model.predict_proba(X_test)[:, 1]   # probability of "made"

ll    = log_loss(y_test, y_prob)
auc   = roc_auc_score(y_test, y_prob)

# Baseline: a "dumb" model that always predicts the training-set mean
baseline_prob = np.full_like(y_prob, fill_value=y_train.mean())
baseline_ll   = log_loss(y_test, baseline_prob)

print("=" * 50)
print("  PHASE 1 EVALUATION RESULTS")
print("=" * 50)
print(f"  Log Loss  (model)    : {ll:.4f}")
print(f"  Log Loss  (baseline) : {baseline_ll:.4f}   ← always predict mean rate")
print(f"  Improvement          : {baseline_ll - ll:.4f} lower is better")
print(f"  ROC-AUC              : {auc:.4f}   ← 0.5 = random, 1.0 = perfect")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 11 — Feature importance
# ─────────────────────────────────────────────
importance = dict(zip(FEATURE_COLS, model.feature_importances_))
print("\nFeature importances (gain):")
for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 200)
    print(f"  {feat:<18} {score:.4f}  {bar}")

# ─────────────────────────────────────────────
# STEP 12 — Sample predictions
# ─────────────────────────────────────────────
print("\nSample predictions on test set:")
sample = pd.DataFrame(X_test[:8], columns=FEATURE_COLS)
sample["actual"]     = y_test[:8]
sample["xFG%"]       = y_prob[:8].round(3)
sample["shot_type"]  = le.inverse_transform(sample["SHOT_TYPE_ENC"].astype(int))
print(sample[["SHOT_DISTANCE", "shot_type", "SHOT_ANGLE_ABS", "actual", "xFG%"]].to_string(index=False))

print("\nPhase 1 complete. Ready for Phase 2 (more features, SHAP explanations, or per-zone analysis).")
