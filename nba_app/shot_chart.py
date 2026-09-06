"""Observed shooting counts and an unsmoothed, area-normalized density map.

Historical percentages use event outcomes, never predictions or averaged bin
percentages. All valid attempts count overall, including heaves outside the
displayed half court. Court zones use the same geometry as prediction inputs.
"""
import math

import pandas as pd

from . import court
from .data import DataUnavailableError

CELL_SIZE = 25.0  # NBA coordinates are tenths of feet: 2.5 ft cells.
ZONE_NAMES = ("Restricted Area", "Paint (Non-RA)", "Mid-Range", "Left Corner 3",
              "Right Corner 3", "Above the Break 3", "Outside half court", "Unlocated")


def _counts(made=0, attempts=0):
    return {"made": int(made), "attempts": int(attempts),
            "fg_pct": made / attempts if attempts else None}


def summarize_shots(frame: pd.DataFrame, player_id: int) -> dict:
    required = {"GAME_ID", "GAME_EVENT_ID", "SHOT_MADE_FLAG", "LOC_X", "LOC_Y"}
    if not required.issubset(frame.columns):
        raise DataUnavailableError("NBA shot records are missing event, outcome, or location fields.")
    work = frame.copy()
    if "PLAYER_ID" in work:
        ids = pd.to_numeric(work["PLAYER_ID"], errors="coerce")
        if not ids.eq(player_id).all():
            raise DataUnavailableError("NBA shot records do not match the selected player.")
    for key in ("GAME_ID", "GAME_EVENT_ID"):
        work[key] = work[key].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        work[key] = work[key].where(work[key].str.fullmatch(r"\d+").fillna(False))
        if key == "GAME_ID":
            work[key] = work[key].str.zfill(10)
    for key in ("SHOT_MADE_FLAG", "LOC_X", "LOC_Y"):
        work[key] = pd.to_numeric(work[key], errors="coerce")
    valid = (work["GAME_ID"].str.fullmatch(r"\d+").fillna(False)
             & work["GAME_EVENT_ID"].str.fullmatch(r"\d+").fillna(False)
             & work["SHOT_MADE_FLAG"].isin([0, 1]))
    if "SHOT_ATTEMPTED_FLAG" in work:
        valid &= pd.to_numeric(work["SHOT_ATTEMPTED_FLAG"], errors="coerce").eq(1)
    invalid_rows = int((~valid).sum())
    work = work.loc[valid].copy()
    keys = ["GAME_ID", "GAME_EVENT_ID"]
    # Conflicting copies cannot be safely counted or positioned.
    conflicts = work.groupby(keys)[["SHOT_MADE_FLAG", "LOC_X", "LOC_Y"]].transform(
        "nunique", dropna=False).gt(1).any(axis=1)
    conflicting_rows = int(conflicts.sum())
    work = work.loc[~conflicts]
    before = len(work)
    work = work.drop_duplicates(keys)
    duplicate_rows = before - len(work)
    zones = {name: [0, 0] for name in ZONE_NAMES}
    bins = {}
    for row in work.itertuples(index=False):
        x, y, made = float(row.LOC_X), float(row.LOC_Y), int(row.SHOT_MADE_FLAG)
        if not math.isfinite(x) or not math.isfinite(y):
            zone = "Unlocated"
        elif not court.X_MIN <= x <= court.X_MAX or not court.Y_MIN <= y <= court.Y_MAX:
            zone = "Outside half court"
        else:
            zone = court.describe_shot(x, y)["shot_zone"]
            col = min(int((x - court.X_MIN) // CELL_SIZE), 19)
            line = min(int((y - court.Y_MIN) // CELL_SIZE), 18)
            totals = bins.setdefault((col, line), [0, 0])
            totals[0] += made
            totals[1] += 1
        zones[zone][0] += made
        zones[zone][1] += 1
    cells = []
    for (col, line), (made, attempts) in sorted(bins.items()):
        x, y = court.X_MIN + col * CELL_SIZE, court.Y_MIN + line * CELL_SIZE
        width, height = min(CELL_SIZE, court.X_MAX - x), min(CELL_SIZE, court.Y_MAX - y)
        area_sq_ft = width * height / 100
        cells.append({"loc_x": x, "loc_y": y, "width": width, "height": height,
                      **_counts(made, attempts), "attempts_per_sq_ft": attempts / area_sq_ft})
    maximum = max((cell["attempts_per_sq_ft"] for cell in cells), default=0)
    for cell in cells:
        cell["intensity"] = cell["attempts_per_sq_ft"] / maximum
    return {"overall": _counts(int(work["SHOT_MADE_FLAG"].sum()), len(work)),
            "zones": [{"name": name, **_counts(*totals)} for name, totals in zones.items()],
            "cells": cells, "plotted": _counts(sum(c["made"] for c in cells), sum(c["attempts"] for c in cells)),
            "max_density": maximum, "cell_size_ft": CELL_SIZE / 10,
            "density_units": "attempts per square foot",
            "geometry": {"corner_x": court.CORNER_X, "arc_join_y": court.ARC_JOIN_Y,
                         "arc_radius": court.ARC_RADIUS, "restricted_radius": court.RESTRICTED_RADIUS,
                         "paint_half_width": court.PAINT_HALF_WIDTH, "paint_end_y": court.PAINT_END_Y},
            "excluded": {"invalid_rows": invalid_rows, "duplicate_rows": duplicate_rows,
                         "conflicting_rows": conflicting_rows},
            "note": "FG% = made / attempted. Overall includes valid attempts outside the plotted half court. "
                    "Zones follow this app's coordinate geometry; they may differ from NBA provider zone labels. "
                    "Red intensity shows attempt density relative to this player-season's busiest cell, not shooting accuracy."}
