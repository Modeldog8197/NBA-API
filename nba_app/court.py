"""One geometry contract for NBA shot-chart coordinates.

Coordinates are tenths of a foot, with the basket at (0, 0), x to the
viewer's right and y toward midcourt. Only the attacking half court is
supported; full-court heaves are excluded. This is a point-location model,
not a reconstruction of a shooter's feet or the physical line's width.

Source: https://official.nba.com/rule-no-1-court-dimensions-equipment/
An exact point on the three-point boundary is treated as two points.
"""

import math

X_MIN, X_MAX = -250.0, 250.0
Y_MIN, Y_MAX = -52.5, 417.5
CORNER_X = 220.0
ARC_RADIUS = 237.5
ARC_JOIN_Y = math.sqrt(ARC_RADIUS**2 - CORNER_X**2)
COURT_VERSION = "nba-halfcourt-v1"
DISTANCE_TOLERANCE_FT = 0.1


def normalize_shot_type(value: str | int) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"2", "2pt", "2pt field goal"}:
        return "2pt"
    if normalized in {"3", "3pt", "3pt field goal"}:
        return "3pt"
    raise ValueError("shot_type must be '2pt' or '3pt'.")


def describe_shot(loc_x: float, loc_y: float, shot_distance: float | None = None,
                  shot_type: str | None = None) -> dict:
    """Validate a half-court location and derive distance, value, and zone."""
    try:
        x, y = float(loc_x), float(loc_y)
    except (ValueError, TypeError) as exc:
        raise ValueError("Shot coordinates must be finite numbers.") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Shot coordinates must be finite numbers.")
    if not X_MIN <= x <= X_MAX or not Y_MIN <= y <= Y_MAX:
        raise ValueError("Location is outside the supported half court: x [-250, 250], y [-52.5, 417.5] in tenths of feet.")
    distance = math.hypot(x, y) / 10
    # Below the arc/straight-line intersection only lateral position matters.
    boundary_measure = abs(x) - CORNER_X if y <= ARC_JOIN_Y else math.hypot(x, y) - ARC_RADIUS
    is_three = boundary_measure > 1e-8
    derived_type = "3pt" if is_three else "2pt"
    if shot_type is not None and normalize_shot_type(shot_type) != derived_type:
        raise ValueError(f"shot_type conflicts with court geometry; this location is {derived_type}.")
    if shot_distance is not None:
        try:
            supplied_distance = float(shot_distance)
        except (ValueError, TypeError) as exc:
            raise ValueError("shot_distance must be a finite distance in feet.") from exc
        if not math.isfinite(supplied_distance) or abs(supplied_distance - distance) > DISTANCE_TOLERANCE_FT + 1e-9:
            raise ValueError(f"shot_distance conflicts with coordinates; expected {distance:.2f} feet (tolerance 0.1 ft).")
    if is_three:
        zone = ("Left Corner 3" if x < 0 else "Right Corner 3") if y <= ARC_JOIN_Y else "Above the Break 3"
    elif distance <= 4 and y >= 0:
        zone = "Restricted Area"
    elif abs(x) <= 80 and y <= 137.5:
        zone = "Paint (Non-RA)"
    else:
        zone = "Mid-Range"
    return {"loc_x": x, "loc_y": y, "distance_ft": distance,
            "shot_value": 3 if is_three else 2, "shot_type": derived_type,
            "shot_zone": zone}
