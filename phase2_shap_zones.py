"""
Basketball Shot Quality Model — Phase 2
SHAP Explanations + Court Zone Analysis

Depends on: phase1_shot_quality_model.py  (must be run first in the same session,
             OR import the helpers below independently).

New in Phase 2:
  • SHAP waterfall chart  — why did the model give THIS shot a particular xFG%?
  • SHAP summary (beeswarm) — which features matter most, and how?
  • Court zone grid       — compare model xFG% vs actual FG% across 9 court zones
  • Zone calibration bar chart

Run:
    python phase2_shap_zones.py
"""

# ─────────────────────────────────────────────
# STEP 0 — Imports
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import shap
import time

from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import players

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# STEP 1 — Re-fetch & rebuild (self-contained)
# ─────────────────────────────────────────────
# All the same logic from Phase 1, consolidated into functions
# so this script can be run standalone.

PLAYER_NAME = "Stephen Curry"
SEASON_ID   = "2023-24"
TARGET_ROWS = 5_000
OUTPUT_DIR  = "."          # save charts to the same folder

FEATURE_COLS = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "SHOT_TYPE_ENC",
    "SHOT_ANGLE",
    "SHOT_ANGLE_ABS",
]

# Friendly labels for charts
FEATURE_LABELS = {
    "LOC_X":          "Horizontal Position",
    "LOC_Y":          "Vertical Position",
    "SHOT_DISTANCE":  "Shot Distance (ft)",
    "SHOT_TYPE_ENC":  "Shot Type (2PT=0 / 3PT=1)",
    "SHOT_ANGLE":     "Shot Angle (signed, rad)",
    "SHOT_ANGLE_ABS": "Shot Angle (absolute, rad)",
}

def fetch_and_prepare() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """Fetch data from nba_api, clean, engineer features, return (df, X, y, le)."""
    print(f"[Phase 2] Fetching data for {PLAYER_NAME} — {SEASON_ID}...")
    player_info = players.find_players_by_full_name(PLAYER_NAME)
    player_id   = player_info[0]["id"]
    time.sleep(1)

    shot_chart = ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=SEASON_ID,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    df = shot_chart.get_data_frames()[0].head(TARGET_ROWS).copy()

    required = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE", "SHOT_MADE_FLAG"]
    df = df.dropna(subset=required)
    for col in ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_MADE_FLAG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)

    # Shot angle features
    df["SHOT_ANGLE"]     = np.arctan2(df["LOC_X"], df["LOC_Y"])
    df["SHOT_ANGLE_ABS"] = df["SHOT_ANGLE"].abs()

    # Encode shot type
    le = LabelEncoder()
    df["SHOT_TYPE_ENC"] = le.fit_transform(df["SHOT_TYPE"])

    X = df[FEATURE_COLS].values
    y = df["SHOT_MADE_FLAG"].values.astype(int)
    print(f"  → {len(df):,} clean records | Made rate: {y.mean():.1%}\n")
    return df, X, y, le


def train_model(X, y):
    """Split data and train XGBoost. Returns model, test split, and probabilities."""
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
    print("[Phase 2] Training model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"  → Training complete. Test samples: {len(y_test):,}\n")
    return model, X_train, X_test, y_train, y_test, y_prob


# ─────────────────────────────────────────────
# STEP 2 — SHAP Setup
# ─────────────────────────────────────────────
#
# ┌─ WHAT IS SHAP? ────────────────────────────────────────────────────────────┐
# │                                                                             │
# │  SHAP (SHapley Additive exPlanations) is a game-theory-based method that  │
# │  fairly distributes a model's prediction among its input features.         │
# │                                                                             │
# │  For a single shot, the model predicts xFG% = 0.48. SHAP breaks that      │
# │  down into a sum:                                                           │
# │                                                                             │
# │    prediction = base_value                                                  │
# │               + SHAP(LOC_X)                                                 │
# │               + SHAP(LOC_Y)                                                 │
# │               + SHAP(SHOT_DISTANCE)                                         │
# │               + ...                                                          │
# │                                                                             │
# │  where base_value is the model's average prediction across the dataset.    │
# │                                                                             │
# │  Positive SHAP → this feature pushed the probability UP for this shot.     │
# │  Negative SHAP → this feature pushed the probability DOWN.                 │
# │                                                                             │
# │  Crucially, SHAP values are CONSISTENT across shots — if feature A always │
# │  contributes more than feature B across the whole dataset, its mean |SHAP| │
# │  will be larger. This is guaranteed by Shapley's fairness axioms.          │
# └─────────────────────────────────────────────────────────────────────────────┘

def compute_shap(model, X_train, X_test) -> shap.Explanation:
    """
    Compute SHAP values using the TreeExplainer (exact, fast for XGBoost).

    XGBoost binary classifiers can return SHAP values with shape
    (n_samples, n_features, 2) — one slice per class.  We always take
    index [:, :, 1] (the positive / 'made' class) so downstream charts
    receive a clean 2-D Explanation and never see a tuple-index error.
    """
    print("[Phase 2] Computing SHAP values (TreeExplainer — exact for tree models)...")
    explainer   = shap.TreeExplainer(model, data=X_train, feature_perturbation="interventional")
    shap_values = explainer(X_test, check_additivity=False)

    print(f"  → Raw SHAP shape: {shap_values.values.shape}")

    # If the output is 3-D (n_samples, n_features, n_classes) take class-1 slice
    if shap_values.values.ndim == 3:
        print("  → 3-D output detected (binary classifier); slicing to positive class (index 1)")
        shap_values = shap.Explanation(
            values       = shap_values.values[:, :, 1],
            base_values  = (
                shap_values.base_values[:, 1]
                if shap_values.base_values.ndim == 2
                else shap_values.base_values
            ),
            data         = shap_values.data,
            feature_names= FEATURE_COLS,
        )

    print(f"  → Final SHAP shape: {shap_values.values.shape}\n")
    return shap_values


# ─────────────────────────────────────────────
# STEP 3 — SHAP Chart 1: Beeswarm Summary
# ─────────────────────────────────────────────
#
# Each dot = one shot.
# X-axis = SHAP value (impact on log-odds of making the shot).
# Color   = actual feature value (red = high, blue = low).
#
# Reading the chart:
#   • Features at the TOP have the most influence overall.
#   • Red dots on the RIGHT → high feature value increases xFG%.
#   • Blue dots on the RIGHT → low feature value increases xFG%.

def plot_shap_beeswarm(shap_values, save_path: str):
    # Give SHAP the friendly names before plotting so the chart labels
    # are already correct — no need to relabel axes after the fact,
    # which is fragile across matplotlib/SHAP version combinations.
    named = shap.Explanation(
        values        = shap_values.values,
        base_values   = shap_values.base_values,
        data          = shap_values.data,
        feature_names = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
    )

    plt.close("all")
    shap.plots.beeswarm(
        named,
        max_display=len(FEATURE_COLS),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP Feature Impact — {PLAYER_NAME} {SEASON_ID}\n"
        "How each feature pushes the xFG% probability up or down",
        fontsize=12, fontweight="bold",
    )
    fig.set_size_inches(10, 6)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# STEP 4 — SHAP Chart 2: Single-Shot Waterfall
# ─────────────────────────────────────────────
#
# Pick the shot where the model is MOST CONFIDENT it will go in
# and explain why, feature by feature, starting from the base rate.

def plot_shap_waterfall(shap_values, y_prob, y_test, save_path: str):
    # Find the shot where the model is most confident it will be made
    most_confident_idx = int(np.argmax(y_prob))
    actual_outcome     = "MADE ✓" if y_test[most_confident_idx] == 1 else "MISSED ✗"

    # Build a single-row Explanation with friendly feature names so the
    # waterfall chart labels are readable without post-hoc axes surgery.
    single = shap.Explanation(
        values        = shap_values.values[most_confident_idx],
        base_values   = (
            shap_values.base_values[most_confident_idx]
            if hasattr(shap_values.base_values, "__len__")
            else shap_values.base_values
        ),
        data          = shap_values.data[most_confident_idx],
        feature_names = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
    )

    plt.close("all")
    shap.plots.waterfall(single, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP Waterfall — Most Confident 'Made' Prediction\n"
        f"xFG%: {y_prob[most_confident_idx]:.1%}   |   Actual: {actual_outcome}",
        fontsize=11, fontweight="bold",
    )
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# STEP 5 — Court Zone Grid
# ─────────────────────────────────────────────
#
# ┌─ ZONE DEFINITIONS ─────────────────────────────────────────────────────────┐
# │                                                                             │
# │  The NBA court coordinate system (nba_api units: tenths of feet):          │
# │    • LOC_X: negative = left wing, positive = right wing                    │
# │    • LOC_Y: 0 = basket, increases toward half-court                        │
# │                                                                             │
# │  We split the court into a 3×3 grid:                                       │
# │    Horizontal (X): Left (< -80), Center (-80 to 80), Right (> 80)         │
# │    Vertical   (Y): Paint (< 80), Mid-Range (80–237), 3PT+ (≥ 237)         │
# │                                                                             │
# │  For each zone we compute:                                                  │
# │    • Actual FG%  — shots made / shots attempted                            │
# │    • Model xFG%  — mean predicted probability in that zone                 │
# │    • Shot count  — volume in that zone                                      │
# │                                                                             │
# │  Good calibration → xFG% ≈ Actual FG% in every zone.                      │
# │  A large gap in a zone means the model is over/under-confident there.      │
# └─────────────────────────────────────────────────────────────────────────────┘

ZONE_X_CUTS  = [-800, -80, 80, 800]       # tenths of feet
ZONE_X_NAMES = ["Left Wing", "Center", "Right Wing"]
ZONE_Y_CUTS  = [-50, 80, 237, 900]
ZONE_Y_NAMES = ["Paint / Post", "Mid-Range", "3PT+"]


def assign_zones(df_test: pd.DataFrame) -> pd.DataFrame:
    df_test = df_test.copy()
    df_test["ZONE_X"] = pd.cut(
        df_test["LOC_X"],
        bins=ZONE_X_CUTS,
        labels=ZONE_X_NAMES,
    )
    df_test["ZONE_Y"] = pd.cut(
        df_test["LOC_Y"],
        bins=ZONE_Y_CUTS,
        labels=ZONE_Y_NAMES,
    )
    return df_test


def compute_zone_stats(df_test: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_test
        .groupby(["ZONE_Y", "ZONE_X"], observed=True)
        .agg(
            Attempts=("SHOT_MADE_FLAG", "count"),
            Actual_FG=("SHOT_MADE_FLAG", "mean"),
            Model_xFG=("y_prob", "mean"),
        )
        .reset_index()
    )
    grouped["Calibration_Gap"] = grouped["Model_xFG"] - grouped["Actual_FG"]
    return grouped


def plot_zone_heatmap(zone_stats: pd.DataFrame, save_path: str):
    """3-row × 3-col heatmap showing actual vs model xFG% in each zone."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Court Zone Analysis — {PLAYER_NAME} {SEASON_ID}",
        fontsize=13, fontweight="bold",
    )

    for ax, value_col, title, fmt, cmap in [
        (axes[0], "Actual_FG",  "Actual FG%",   ".1%", "YlGn"),
        (axes[1], "Model_xFG",  "Model xFG%",   ".1%", "YlOrRd"),
    ]:
        pivot = zone_stats.pivot(index="ZONE_Y", columns="ZONE_X", values=value_col)
        pivot = pivot.reindex(index=ZONE_Y_NAMES[::-1], columns=ZONE_X_NAMES)

        count_pivot = zone_stats.pivot(index="ZONE_Y", columns="ZONE_X", values="Attempts")
        count_pivot = count_pivot.reindex(index=ZONE_Y_NAMES[::-1], columns=ZONE_X_NAMES)

        sns.heatmap(
            pivot,
            ax=ax,
            annot=False,
            fmt=fmt,
            cmap=cmap,
            vmin=0.25, vmax=0.65,
            linewidths=2,
            linecolor="white",
            cbar_kws={"format": "{:.0%}"},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Court Horizontal Zone")
        ax.set_ylabel("Court Vertical Zone")

        # Annotate each cell with FG% and shot count
        for i, zone_y in enumerate(ZONE_Y_NAMES[::-1]):
            for j, zone_x in enumerate(ZONE_X_NAMES):
                try:
                    val   = pivot.loc[zone_y, zone_x]
                    count = int(count_pivot.loc[zone_y, zone_x])
                    ax.text(j + 0.5, i + 0.4, f"{val:.1%}",
                            ha="center", va="center", fontsize=11, fontweight="bold", color="black")
                    ax.text(j + 0.5, i + 0.65, f"n={count}",
                            ha="center", va="center", fontsize=8, color="#444444")
                except Exception:
                    pass

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_calibration_bar(zone_stats: pd.DataFrame, save_path: str):
    """Bar chart of calibration gap per zone (model xFG% − actual FG%)."""
    zone_stats = zone_stats.copy()
    zone_stats["Zone"] = zone_stats["ZONE_Y"].astype(str) + "\n" + zone_stats["ZONE_X"].astype(str)
    zone_stats = zone_stats.sort_values("Calibration_Gap")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#d73027" if v > 0 else "#1a9850" for v in zone_stats["Calibration_Gap"]]
    bars   = ax.bar(zone_stats["Zone"], zone_stats["Calibration_Gap"] * 100, color=colors, edgecolor="white")

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(
        f"Model Calibration Gap by Court Zone\n"
        f"Red = model overestimates xFG%   |   Green = model underestimates xFG%",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("xFG% − Actual FG%  (percentage points)")
    ax.set_xlabel("Zone  (Vertical × Horizontal)")

    for bar, (_, row) in zip(bars, zone_stats.iterrows()):
        height = bar.get_height()
        sign   = "+" if height >= 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.3 if height >= 0 else -0.7),
            f"{sign}{height:.1f} pp",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# STEP 6 — Shot chart scatter (actual vs model)
# ─────────────────────────────────────────────

def plot_shot_chart(df_test: pd.DataFrame, save_path: str):
    """Scatter of shot locations coloured by predicted xFG%."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Shot Chart — {PLAYER_NAME} {SEASON_ID}  (test set)",
        fontsize=13, fontweight="bold",
    )

    for ax, col, title, cmap in [
        (axes[0], "SHOT_MADE_FLAG", "Actual Outcome\n(green = made, red = missed)", None),
        (axes[1], "y_prob",         "Model xFG% Probability",                       "RdYlGn"),
    ]:
        if col == "SHOT_MADE_FLAG":
            colors = ["#d73027" if v == 0 else "#1a9850" for v in df_test[col]]
            sc = ax.scatter(df_test["LOC_X"], df_test["LOC_Y"],
                            c=colors, s=12, alpha=0.55, linewidths=0)
        else:
            sc = ax.scatter(df_test["LOC_X"], df_test["LOC_Y"],
                            c=df_test[col], cmap=cmap, vmin=0.2, vmax=0.7,
                            s=12, alpha=0.7, linewidths=0)
            fig.colorbar(sc, ax=ax, label="xFG%", format="{x:.0%}")

        # Draw the three-point arc (approximate)
        theta = np.linspace(np.radians(-68), np.radians(68), 200)
        arc_x = 237.5 * np.sin(theta)
        arc_y = 237.5 * np.cos(theta)
        ax.plot(arc_x, arc_y, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
        # Corner 3 lines
        ax.plot([-220, -220], [-47, 92.5], color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.plot([ 220,  220], [-47, 92.5], color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
        # Basket
        ax.scatter([0], [0], c="black", s=50, zorder=5)

        ax.set_xlim(-260, 260)
        ax.set_ylim(-50, 470)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("LOC_X (tenths of feet)")
        ax.set_ylabel("LOC_Y (tenths of feet)")
        ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1 — Data
    df, X, y, le = fetch_and_prepare()

    # 2 — Model
    model, X_train, X_test, y_train, y_test, y_prob = train_model(X, y)

    # 3 — Build test-set dataframe for zone / chart analysis
    df_test = pd.DataFrame(X_test, columns=FEATURE_COLS)
    df_test["SHOT_MADE_FLAG"] = y_test
    df_test["y_prob"]         = y_prob

    # 4 — SHAP
    print("[Phase 2] Generating SHAP charts...")
    shap_values = compute_shap(model, X_train, X_test)
    plot_shap_beeswarm(shap_values, f"{OUTPUT_DIR}/chart_shap_beeswarm.png")
    plot_shap_waterfall(shap_values, y_prob, y_test, f"{OUTPUT_DIR}/chart_shap_waterfall.png")

    # 5 — Zone analysis
    print("[Phase 2] Generating zone analysis charts...")
    df_test = assign_zones(df_test)
    zone_stats = compute_zone_stats(df_test)

    print("\nZone-by-zone breakdown:")
    print(
        zone_stats[["ZONE_Y", "ZONE_X", "Attempts", "Actual_FG", "Model_xFG", "Calibration_Gap"]]
        .sort_values(["ZONE_Y", "ZONE_X"])
        .to_string(index=False)
    )
    print()

    plot_zone_heatmap(zone_stats, f"{OUTPUT_DIR}/chart_zone_heatmap.png")
    plot_calibration_bar(zone_stats, f"{OUTPUT_DIR}/chart_calibration_gap.png")

    # 6 — Shot chart
    print("[Phase 2] Generating shot chart...")
    plot_shot_chart(df_test, f"{OUTPUT_DIR}/chart_shot_locations.png")

    # 7 — Summary
    print("\n" + "=" * 55)
    print("  PHASE 2 COMPLETE")
    print("=" * 55)
    print("  Charts saved:")
    print("    chart_shap_beeswarm.png    — overall feature importance")
    print("    chart_shap_waterfall.png   — single-shot explanation")
    print("    chart_zone_heatmap.png     — actual vs model FG% by zone")
    print("    chart_calibration_gap.png  — model bias per zone")
    print("    chart_shot_locations.png   — full shot chart coloured by xFG%")
    print("=" * 55)
    print("\nReady for Phase 3: multi-season stacking, defender proximity,")
    print("play-type tags (ISO, P&R, Spot-Up), or REST API serving.")
