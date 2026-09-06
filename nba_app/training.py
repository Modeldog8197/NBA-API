"""Chronological game-isolated evaluation and atomic model publication.

The original estimator's settings are retained as a reproducible comparison,
but all estimators use the corrected feature contract and honest time splits.
The historical repository's saved model has unknown provenance and is not loaded.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
import platform
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb

from .court import COURT_VERSION
from .config import validate_season
from .features import FEATURE_COLS, FEATURE_VERSION, SHOT_TYPE_ENCODING, prepare_shots
from .models import ModelStore, sigmoid

MIN_SHOTS = 150
MIN_GAMES = 15
RANDOM_SEED = 42
COMMON_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_SEED,
                     n_jobs=2, tree_method="hist")
ORIGINAL_PARAMS = {**COMMON_PARAMS, "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8}
CANDIDATE_PARAMS = {**COMMON_PARAMS, "n_estimators": 240, "max_depth": 3, "learning_rate": 0.035,
                    "min_child_weight": 12, "reg_lambda": 8.0, "reg_alpha": 0.25,
                    "subsample": 0.9, "colsample_bytree": 1.0}


class TrainingError(ValueError):
    """The requested data cannot support the documented evaluation protocol."""


def chronological_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """60/20/20 by ordered games, with validation split for calibration/selection."""
    games = df[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values(["GAME_DATE", "GAME_ID"])
    if len(df) < MIN_SHOTS or len(games) < MIN_GAMES:
        raise TrainingError(f"Need at least {MIN_SHOTS} usable shots and {MIN_GAMES} games; found {len(df)} shots in {len(games)} games.")
    n_train, n_validation = int(len(games) * 0.6), int(len(games) * 0.2)
    ordered_ids = games["GAME_ID"].tolist()
    val_start, test_start = n_train, n_train + n_validation
    calib_end = val_start + max(1, n_validation // 2)
    assignments = {"train": ordered_ids[:n_train], "calibration": ordered_ids[val_start:calib_end],
                   "validation": ordered_ids[calib_end:test_start], "final_test": ordered_ids[test_start:]}
    splits = {name: df[df["GAME_ID"].isin(ids)].copy() for name, ids in assignments.items()}
    for name, split in splits.items():
        if len(split) < 20:
            raise TrainingError(f"The chronological {name} split has only {len(split)} shots; at least 20 are required.")
        if split["SHOT_MADE_FLAG"].nunique() < 2:
            raise TrainingError(f"The chronological {name} split contains only one outcome class. More games with made and missed shots are required.")
    # Multiple players in one dataset can play different games on the same day;
    # per-player season training normally has a strict date order regardless.
    ordered_splits = list(splits.values())
    for earlier, later in zip(ordered_splits, ordered_splits[1:]):
        if earlier["GAME_DATE"].max() > later["GAME_DATE"].min():
            raise TrainingError("Chronological game boundaries overlap unexpectedly.")
    return splits


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    bins = []
    indices = np.minimum((probability * 10).astype(int), 9)
    for index in range(10):
        mask = indices == index
        if mask.any():
            bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                         "count": int(mask.sum()), "predicted": float(probability[mask].mean()),
                         "observed": float(y[mask].mean())})
    return {"n_shots": int(len(y)), "make_rate": float(np.mean(y)),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, probability)), "calibration_bins": bins}


def _calibration_svg(results: dict, selected: str, is_demo: bool) -> str:
    """Portable vector reliability plot with final-test probabilities only."""
    colors = {"constant_baseline": "#64748b", "original_xgboost": "#ea580c",
              "candidate_xgboost": "#0891b2", "original_sigmoid": "#a855f7", "candidate_sigmoid": "#059669"}
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="580" viewBox="0 0 800 580" role="img" aria-label="Final test calibration plot">',
             '<rect width="800" height="580" fill="white"/>',
             '<g font-family="sans-serif" fill="#0f172a">',
             '<text x="60" y="35" font-size="22">Final-test probability calibration</text>',
             f'<text x="60" y="58" font-size="13">{"SYNTHETIC FIXTURE — not NBA performance" if is_demo else "Held-out later games; each point is a populated probability bin"}</text>',
             '<path d="M70 480H490 M70 480V80" fill="none" stroke="#334155"/>',
             '<path d="M70 480L490 80" stroke="#94a3b8" stroke-dasharray="5 5"/>']
    for tick in range(6):
        value = tick / 5
        x, y = 70 + 420 * value, 480 - 400 * value
        lines += [f'<text x="{x}" y="503" text-anchor="middle" font-size="12">{value:.1f}</text>',
                  f'<text x="55" y="{y + 4}" text-anchor="end" font-size="12">{value:.1f}</text>']
    for row, (name, metric) in enumerate(results.items()):
        color = colors[name]
        coordinates = " ".join(f'{70+420*b["predicted"]:.2f},{480-400*b["observed"]:.2f}' for b in metric["calibration_bins"])
        lines.append(f'<polyline points="{coordinates}" stroke="{color}" fill="none" stroke-width="2"/>')
        for item in metric["calibration_bins"]:
            lines.append(f'<circle cx="{70+420*item["predicted"]:.2f}" cy="{480-400*item["observed"]:.2f}" r="4" fill="{color}"><title>n={item["count"]}</title></circle>')
        label = name.replace("_", " ") + (" (selected)" if name == selected else "")
        lines += [f'<rect x="520" y="{100+35*row}" width="10" height="10" fill="{color}"/>',
                  f'<text x="538" y="{110+35*row}" font-size="12">{label}</text>']
    lines += ['<text x="280" y="537" text-anchor="middle" font-size="15">Mean predicted make probability</text>',
              '<text transform="translate(20 280) rotate(-90)" text-anchor="middle" font-size="15">Observed make fraction</text>',
              '<text x="60" y="566" font-size="12">No error bands: uncertainty intervals have not been evaluated.</text></g></svg>']
    return "\n".join(lines)


def _split_metadata(split: pd.DataFrame) -> dict:
    return {"n_shots": int(len(split)), "n_games": int(split["GAME_ID"].nunique()),
            "start_date": split["GAME_DATE"].min().date().isoformat(), "end_date": split["GAME_DATE"].max().date().isoformat(),
            "game_ids": split["GAME_ID"].drop_duplicates().tolist(), "made": int(split["SHOT_MADE_FLAG"].sum())}


def train_bundle(df: pd.DataFrame, player_id: int, player_name: str, season: str, store: ModelStore,
                 source: str = "nba", progress=None, source_info: dict | None = None) -> dict:
    """Train, select on validation, evaluate final test once, then publish.

    ``progress`` is an optional callable receiving a human-readable string.
    No final-test observation is used in training, calibration, or selection.
    """
    def report(message):
        if progress is not None:
            progress(message)

    if not isinstance(player_id, int) or player_id < 0 or not str(player_name).strip():
        raise TrainingError("A stable player ID and player name are required.")
    try:
        validate_season(season)
    except (ValueError, TypeError) as exc:
        raise TrainingError(str(exc)) from exc
    provenance = dict(source_info or {})
    source_name = str(provenance.get("source", source))
    if isinstance(df, pd.DataFrame) and "PLAYER_ID" in df and not df.empty:
        actual_players = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
        if actual_players.isna().any() or not actual_players.eq(player_id).all():
            raise TrainingError("Shot data contains a different or missing player ID; model identity would be misleading.")
    report("Cleaning shot records and validating court geometry")
    try:
        prepared, cleaning = prepare_shots(df)
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    if prepared.empty:
        raise TrainingError("No usable shots remain after data validation.")
    # Season spans autumn through the following summer; reject mislabeled CSVs.
    season_start, season_end = pd.Timestamp(f"{season[:4]}-09-01"), pd.Timestamp(f"{int(season[:4])+1}-08-31")
    if not prepared["GAME_DATE"].between(season_start, season_end).all():
        raise TrainingError("Game dates fall outside the requested NBA season; check the data and season selection.")
    splits = chronological_split(prepared)
    matrices = {name: xgb.DMatrix(part[FEATURE_COLS], feature_names=FEATURE_COLS) for name, part in splits.items()}
    targets = {name: part["SHOT_MADE_FLAG"].to_numpy(dtype=int) for name, part in splits.items()}
    constant_probability = float((targets["train"].sum() + 1) / (len(targets["train"]) + 2))
    trained = {}
    for name, params in [("original_xgboost", ORIGINAL_PARAMS), ("candidate_xgboost", CANDIDATE_PARAMS)]:
        report(f"Fitting {name.replace('_', ' ')} on training games only")
        estimator = xgb.XGBClassifier(**params)
        estimator.fit(splits["train"][FEATURE_COLS], targets["train"], verbose=False)
        trained[name] = estimator.get_booster()
    candidates = {"constant_baseline": {"booster": None, "calibration": {"method": "none", "slope": 1.0, "intercept": 0.0}}}
    report("Fitting sigmoid calibration on the early validation games")
    calibration_notes = []
    for name, booster in trained.items():
        candidates[name] = {"booster": booster, "calibration": {"method": "none", "slope": 1.0, "intercept": 0.0}}
        raw_calibration = booster.predict(matrices["calibration"], output_margin=True).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_SEED, max_iter=1000)
        calibrator.fit(raw_calibration, targets["calibration"])
        slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
        if not math.isfinite(slope) or slope <= 0:
            calibration_notes.append(f"{name}: omitted sigmoid candidate because calibration would reverse score direction.")
            continue
        candidates[name.replace("xgboost", "sigmoid")] = {"booster": booster,
            "calibration": {"method": "sigmoid", "slope": slope, "intercept": intercept,
                            "fit_split": "calibration", "regularization_C": 1.0}}

    def probabilities(candidate, split_name):
        if candidate["booster"] is None:
            return np.full(len(targets[split_name]), constant_probability)
        raw = candidate["booster"].predict(matrices[split_name], output_margin=True)
        calibration = candidate["calibration"]
        return sigmoid(calibration["slope"] * raw + calibration["intercept"])

    report("Selecting the model by log loss on later validation games")
    validation_results = {name: evaluate(targets["validation"], probabilities(candidate, "validation")) for name, candidate in candidates.items()}
    selected = min(validation_results, key=lambda name: (validation_results[name]["log_loss"], name))
    report("Evaluating the locked selection on final test games")
    final_probabilities = {name: probabilities(candidate, "final_test") for name, candidate in candidates.items()}
    final_results = {name: evaluate(targets["final_test"], probability) for name, probability in final_probabilities.items()}
    chosen = candidates[selected]
    now = datetime.now(timezone.utc)
    model_id = f"p{player_id}-{season}-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    dependencies = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "nba_api"]:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not installed"
    is_demo = player_id == 0 or any(token in source_name.lower() for token in ["synthetic", "fixture", "demo"])
    test_predictions = splits["final_test"][["GAME_ID", "GAME_DATE", "GAME_EVENT_ID", "LOC_X", "LOC_Y", "SHOT_ZONE", "SHOT_MADE_FLAG"]].copy()
    test_predictions["GAME_DATE"] = test_predictions["GAME_DATE"].dt.strftime("%Y-%m-%d")
    for name, probability in final_probabilities.items():
        test_predictions[name] = probability
    digest_columns = ["GAME_ID", "GAME_DATE", "GAME_EVENT_ID", *FEATURE_COLS, "SHOT_MADE_FLAG"]
    data_digest = hashlib.sha256(prepared[digest_columns].to_csv(index=False, float_format="%.12g").encode()).hexdigest()
    metadata = {
        "schema_version": 1, "model_id": model_id, "created_at": now.isoformat(),
        "player_id": player_id, "player_name": player_name, "season": season, "seasons": [season],
        "n_shots": len(prepared), "n_games": int(prepared["GAME_ID"].nunique()),
        "source": source_name, "data_source": provenance or {"source": source_name}, "is_demo": is_demo,
        "data_sha256": data_digest, "cleaning": cleaning, "feature_cols": FEATURE_COLS,
        "feature_version": FEATURE_VERSION, "court_version": COURT_VERSION, "shot_type_encoding": SHOT_TYPE_ENCODING,
        "coordinate_units": "tenths of a foot; basket (0,0); x right, y toward half court",
        "selected_model": selected, "model_kind": "constant" if chosen["booster"] is None else "xgboost",
        "constant_probability": constant_probability if chosen["booster"] is None else None,
        "calibration": chosen["calibration"], "calibration_notes": calibration_notes,
        "split": {"strategy": "chronological games; 60% train, 20% validation, 20% final test by game count; early half of validation fits calibration, later half selects model",
                  "rounding": "floor train and validation game counts; remainder final test",
                  **{name: _split_metadata(part) for name, part in splits.items()}},
        "parameters": {"original_xgboost": ORIGINAL_PARAMS, "candidate_xgboost": CANDIDATE_PARAMS,
                       "constant_baseline": {"method": "Laplace-smoothed train make rate", "alpha": 1},
                       "sigmoid": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}},
        "evaluation": {"validation": validation_results, "final_test": final_results,
                       "selection_metric": "later-validation log_loss", "selected_model": selected,
                       "selected_minus_original_test_log_loss": final_results[selected]["log_loss"] - final_results["original_xgboost"]["log_loss"],
                       "selected_minus_constant_test_log_loss": final_results[selected]["log_loss"] - final_results["constant_baseline"]["log_loss"],
                       "interpretation": "Synthetic fixture engineering check; these are not NBA performance measurements." if is_demo else "A single held-out season split; loss differences are descriptive, not a statistical significance or future-performance guarantee.",
                       "original_comparison": "Original estimator settings retrained with corrected shared geometry and chronological splits; unverified legacy artifact is not evaluated."},
        "dependency_versions": dependencies, "random_seed": RANDOM_SEED,
        "limitations": ["Uses shot location only: no defender distance, fatigue, shot clock, injuries, or live game context.",
                        "Point locations are quantized and cannot establish the shooter's feet near the three-point line; conflicting source labels are excluded.",
                        "Only half-court field goals are supported; full-court heaves and free throws are excluded.",
                        "Player-season samples may be sparse in parts of the court; heatmaps extrapolate there.",
                        "No evaluated statistical uncertainty interval is available.",
                        "Models are not refit on validation or final-test games after selection; recorded train samples are the actual fit data."]
                       + (["Synthetic fixture only. Predictions are for demonstration, not an NBA player."] if is_demo else []),
    }
    report("Verifying explanations and atomically publishing the model version")
    return store.publish(metadata, chosen["booster"], test_predictions, _calibration_svg(final_results, selected, is_demo))
