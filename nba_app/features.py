"""Deterministic feature engineering; there is no fitted categorical encoder."""

import math
import numpy as np
import pandas as pd

from .court import describe_shot, normalize_shot_type

FEATURE_COLS = ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE_ENC", "SHOT_ANGLE", "SHOT_ANGLE_ABS"]
FEATURE_LABELS = {"LOC_X": "Horizontal location", "LOC_Y": "Distance toward midcourt",
                  "SHOT_DISTANCE": "Distance from basket", "SHOT_TYPE_ENC": "Two- or three-point location",
                  "SHOT_ANGLE": "Side and angle", "SHOT_ANGLE_ABS": "Angle from the center"}
FEATURE_VERSION = "geometry-six-features-v1"
SHOT_TYPE_ENCODING = {"2pt": 0, "3pt": 1}


def feature_frame(shots: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Use exactly the same geometry and feature order for every inference."""
    descriptions, rows = [], []
    for shot in shots:
        item = describe_shot(shot["loc_x"], shot["loc_y"], shot.get("shot_distance", shot.get("distance_ft")), shot.get("shot_type"))
        angle = math.atan2(item["loc_x"], item["loc_y"])
        rows.append([item["loc_x"], item["loc_y"], item["distance_ft"], SHOT_TYPE_ENCODING[item["shot_type"]], angle, abs(angle)])
        descriptions.append(item)
    return pd.DataFrame(rows, columns=FEATURE_COLS, dtype=float), descriptions


def prepare_shots(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop unusable records and conflicting event duplicates, reporting counts.

    NBA's integer SHOT_DISTANCE is replaced by precise coordinate distance.
    Conflicting official shot types are excluded because quantized locations
    cannot establish a shooter's feet near a line. Missing types are derived.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("No shot data is available for training.")
    required = {"LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "GAME_ID", "GAME_DATE", "GAME_EVENT_ID"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required NBA columns: {', '.join(missing)}.")
    work = df.copy(deep=True).reset_index(drop=True)
    report = {"input_rows": int(len(work)), "invalid_rows": 0, "duplicate_rows": 0,
              "conflicting_duplicate_rows": 0, "geometry_conflicts": 0, "out_of_court_rows": 0,
              "derived_shot_types": 0, "distance_policy": "Recompute Euclidean feet from LOC_X/LOC_Y; ignore rounded source SHOT_DISTANCE."}
    for col in ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    dates = work["GAME_DATE"].astype("string").str.replace(r"\.0$", "", regex=True)
    work["GAME_DATE"] = pd.to_datetime(dates, errors="coerce", format="mixed")
    valid = (np.isfinite(work[["LOC_X", "LOC_Y"]]).all(axis=1) & work["SHOT_MADE_FLAG"].isin([0, 1])
             & work["GAME_DATE"].notna() & work["GAME_ID"].notna() & work["GAME_EVENT_ID"].notna())
    report["invalid_rows"] += int((~valid).sum())
    work = work.loc[valid].copy()
    for key in ["GAME_ID", "GAME_EVENT_ID"]:
        work[key] = work[key].astype(str).str.strip().str.replace(r"^([0-9]+)\.0$", r"\1", regex=True)
        if key == "GAME_ID":
            # CSV readers may strip the leading zeroes from NBA game IDs.
            work[key] = work[key].map(lambda value: value.zfill(10) if value.isdigit() else value)
    valid_ids = work["GAME_ID"].ne("") & work["GAME_EVENT_ID"].ne("")
    report["invalid_rows"] += int((~valid_ids).sum())
    work = work.loc[valid_ids].copy()
    keys = ["GAME_ID", "GAME_EVENT_ID"]
    compare = ["LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "GAME_DATE"] + (["SHOT_TYPE"] if "SHOT_TYPE" in work else [])
    if not work.empty:
        conflicts = work.groupby(keys, dropna=False)[compare].transform("nunique").gt(1).any(axis=1)
        report["conflicting_duplicate_rows"] = int(conflicts.sum())
        work = work.loc[~conflicts].copy()
    before = len(work)
    work = work.drop_duplicates(keys, keep="first")
    report["duplicate_rows"] = int(before - len(work))
    if not work.empty and work.groupby("GAME_ID")["GAME_DATE"].nunique().gt(1).any():
        raise ValueError("A GAME_ID has inconsistent game dates; fix source data before training.")
    kept, engineered = [], []
    for idx, row in work.iterrows():
        try:
            shot = describe_shot(row["LOC_X"], row["LOC_Y"])
        except ValueError:
            report["out_of_court_rows"] += 1
            continue
        source_type = row.get("SHOT_TYPE")
        if pd.isna(source_type) or source_type is None:
            report["derived_shot_types"] += 1
        else:
            try:
                if normalize_shot_type(source_type) != shot["shot_type"]:
                    report["geometry_conflicts"] += 1
                    continue
            except ValueError:
                report["invalid_rows"] += 1
                continue
        angle = math.atan2(shot["loc_x"], shot["loc_y"])
        kept.append(idx)
        engineered.append({"SHOT_DISTANCE": shot["distance_ft"], "SHOT_TYPE_ENC": SHOT_TYPE_ENCODING[shot["shot_type"]],
                           "SHOT_ANGLE": angle, "SHOT_ANGLE_ABS": abs(angle), "SHOT_ZONE": shot["shot_zone"],
                           "SHOT_TYPE": "3PT Field Goal" if shot["shot_value"] == 3 else "2PT Field Goal"})
    output = work.loc[kept].copy().reset_index(drop=True)
    for col in ["SHOT_DISTANCE", "SHOT_TYPE_ENC", "SHOT_ANGLE", "SHOT_ANGLE_ABS", "SHOT_ZONE", "SHOT_TYPE"]:
        output[col] = [row[col] for row in engineered]
    output["SHOT_MADE_FLAG"] = output["SHOT_MADE_FLAG"].astype(int)
    output = output.sort_values(["GAME_DATE", "GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)
    report["output_rows"] = int(len(output))
    report["dropped_rows"] = int(report["input_rows"] - len(output))
    return output, report
