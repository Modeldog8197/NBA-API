import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nba_app.api import create_app
from nba_app.config import Settings
from nba_app.data import DataUnavailableError
from nba_app.shot_chart import summarize_shots


def shot(event, x=0, y=20, made=1, **extra):
    return {"GAME_ID": "0022300001", "GAME_EVENT_ID": event, "PLAYER_ID": 42,
            "LOC_X": x, "LOC_Y": y, "SHOT_MADE_FLAG": made, "SHOT_ATTEMPTED_FLAG": 1, **extra}


def test_counts_are_weighted_and_include_unplotted_attempts():
    rows = [shot(1), shot(2, made=0), shot(3, x=-230), shot(4, x=-230, made=0),
            shot(5, x=-230, made=0), shot(6, y=800, made=0), shot(7, x=math.nan)]
    chart = summarize_shots(pd.DataFrame(rows), 42)
    assert chart["overall"] == {"made": 3, "attempts": 7, "fg_pct": 3/7}
    assert chart["plotted"] == {"made": 2, "attempts": 5, "fg_pct": 2/5}
    zones = {z["name"]: z for z in chart["zones"]}
    assert zones["Restricted Area"]["fg_pct"] == 1/2
    assert zones["Left Corner 3"]["fg_pct"] == 1/3
    assert zones["Right Corner 3"]["fg_pct"] is None
    assert sum(z["made"] for z in zones.values()) == chart["overall"]["made"]
    assert sum(z["attempts"] for z in zones.values()) == chart["overall"]["attempts"]
    assert sum(c["attempts"] for c in chart["cells"]) == 5


def test_duplicates_conflicts_and_non_attempts_are_not_counted():
    rows = [shot(1), shot(1, GAME_ID=22300001.0), shot(2), shot(2, made=0),
            shot(3, made=2), shot(4, SHOT_ATTEMPTED_FLAG=0), shot(5, GAME_ID=""),
            shot(6, GAME_EVENT_ID=None)]
    chart = summarize_shots(pd.DataFrame(rows), 42)
    assert chart["overall"] == {"made": 1, "attempts": 1, "fg_pct": 1}
    assert chart["excluded"] == {"invalid_rows": 4, "duplicate_rows": 1, "conflicting_rows": 2}


@pytest.mark.parametrize("x,y,zone", [
    (-230, 0, "Left Corner 3"), (230, 0, "Right Corner 3"), (220, 0, "Mid-Range"),
    (0, 40, "Restricted Area"), (0, 41, "Paint (Non-RA)"), (80, 137.5, "Paint (Non-RA)"),
    (0, 237.5, "Mid-Range"), (0, 237.5001, "Above the Break 3"),
])
def test_zones_follow_exact_court_boundaries(x, y, zone):
    chart = summarize_shots(pd.DataFrame([shot(1, x=x, y=y)]), 42)
    assert next(z for z in chart["zones"] if z["name"] == zone)["made"] == 1


def test_density_counts_misses_and_normalizes_partial_edge_cell_area():
    chart = summarize_shots(pd.DataFrame([shot(1, x=-250, y=-52.5, made=0),
                                         shot(2, x=250, y=417.5)]), 42)
    assert len(chart["cells"]) == 2  # Empty cells are not colored.
    first, last = chart["cells"]
    assert first["attempts_per_sq_ft"] == 1/6.25
    assert last["attempts_per_sq_ft"] == 1/5
    assert last["loc_x"] + last["width"] == 250
    assert last["loc_y"] + last["height"] == 417.5
    assert first["intensity"] == pytest.approx(.8)
    assert last["intensity"] == 1


def test_empty_dataset_does_not_invent_zero_percent():
    chart = summarize_shots(pd.DataFrame([shot(1)]).iloc[:0], 42)
    assert chart["overall"]["fg_pct"] is None
    assert chart["cells"] == []
    assert chart["max_density"] == 0


def test_wrong_player_and_missing_schema_fail_explicitly():
    with pytest.raises(DataUnavailableError):
        summarize_shots(pd.DataFrame([shot(1)]), 84)
    with pytest.raises(DataUnavailableError):
        summarize_shots(pd.DataFrame([{"LOC_X": 0}]), 42)


def test_api_uses_exact_player_season_without_requiring_a_model(tmp_path):
    class DataClient:
        def resolve_player(self, player_id):
            return {"id": player_id, "name": "Test player"}

        def fetch_shots(self, player_id, season):
            assert player_id == 42
            made = int(season == "2023-24")
            return pd.DataFrame([shot(1, made=made)]), {"stale": False}

    app = create_app(Settings(model_dir=tmp_path / "models"), data_client=DataClient())
    with TestClient(app) as client:
        first = client.get("/player/shot-chart", params={"player_id": 42, "season": "2023-24"})
        second = client.get("/player/shot-chart", params={"player_id": 42, "season": "2022-23"})
        assert first.status_code == second.status_code == 200
        assert first.json()["overall"]["fg_pct"] == 1
        assert second.json()["overall"]["fg_pct"] == 0
        assert client.get("/player/shot-chart?player_id=42&season=invalid").status_code == 422
