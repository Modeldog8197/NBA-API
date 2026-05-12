"""
Basketball Shot Quality Model — Model Trainer & Saver

Run this ONCE before starting the API server.
It fetches shot data, trains the XGBoost model, and saves two artefacts:

    model.json      — XGBoost native format (portable, version-safe)
    encoder.joblib  — scikit-learn LabelEncoder for SHOT_TYPE

Usage:
    python train_and_save_model.py
    python train_and_save_model.py --player "Joel Embiid" --season "2024-25"
"""

import argparse
import time
import numpy as np
import pandas as pd
import joblib

from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import players
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score

# ─────────────────────────────────────────────
# Defaults (can be overridden via CLI)
# ─────────────────────────────────────────────
DEFAULT_PLAYER  = "Stephen Curry"
DEFAULT_SEASON  = "2023-24"
DEFAULT_ROWS    = 5_000

FEATURE_COLS = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "SHOT_TYPE_ENC",
    "SHOT_ANGLE",
    "SHOT_ANGLE_ABS",
]

MODEL_PATH   = "model.json"
ENCODER_PATH = "encoder.joblib"
META_PATH    = "model_meta.joblib"     # stores training metadata for the API


def fetch_shots(player_name: str, season: str, n_rows: int) -> pd.DataFrame:
    print(f"Looking up '{player_name}'...")
    info = players.find_players_by_full_name(player_name)
    if not info:
        raise ValueError(f"Player '{player_name}' not found.")
    player_id = info[0]["id"]
    print(f"  Player ID: {player_id}")

    time.sleep(1)
    print(f"Fetching shots for {season}...")
    endpoint = ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    df = endpoint.get_data_frames()[0].head(n_rows).copy()
    print(f"  Fetched {len(df):,} records")
    return df


def clean_and_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    required = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE", "SHOT_MADE_FLAG"]
    df = df.dropna(subset=required)
    for col in ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_MADE_FLAG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)

    df["SHOT_ANGLE"]     = np.arctan2(df["LOC_X"], df["LOC_Y"])
    df["SHOT_ANGLE_ABS"] = df["SHOT_ANGLE"].abs()

    le = LabelEncoder()
    df["SHOT_TYPE_ENC"] = le.fit_transform(df["SHOT_TYPE"])
    return df, le


def train(df: pd.DataFrame) -> tuple[xgb.XGBClassifier, float, float]:
    X = df[FEATURE_COLS].values
    y = df["SHOT_MADE_FLAG"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    ll     = log_loss(y_test, y_prob)
    auc    = roc_auc_score(y_test, y_prob)
    return model, ll, auc


def main():
    parser = argparse.ArgumentParser(description="Train & save xFG% model")
    parser.add_argument("--player",  default=DEFAULT_PLAYER,  help="Full player name")
    parser.add_argument("--season",  default=DEFAULT_SEASON,  help="Season, e.g. 2023-24")
    parser.add_argument("--rows",    default=DEFAULT_ROWS,    type=int, help="Max shot rows")
    args = parser.parse_args()

    df_raw       = fetch_shots(args.player, args.season, args.rows)
    df, le       = clean_and_engineer(df_raw)
    model, ll, auc = train(df)

    # ── Save artefacts ──────────────────────────────────────────────────────
    model.save_model(MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    # Store metadata the API needs at startup (feature list, shot type classes)
    meta = {
        "feature_cols":   FEATURE_COLS,
        "shot_type_classes": list(le.classes_),   # e.g. ["2PT Field Goal", "3PT Field Goal"]
        "player":  args.player,
        "season":  args.season,
        "log_loss": ll,
        "roc_auc":  auc,
        "n_shots":  len(df),
    }
    joblib.dump(meta, META_PATH)

    print()
    print("=" * 52)
    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Encoder      → {ENCODER_PATH}")
    print(f"  Metadata     → {META_PATH}")
    print(f"  Log Loss     : {ll:.4f}")
    print(f"  ROC-AUC      : {auc:.4f}")
    print("=" * 52)
    print("\nReady — now run:  python phase3_api.py")


if __name__ == "__main__":
    main()
