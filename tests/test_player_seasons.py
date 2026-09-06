"""Career-season discovery uses real endpoint contracts with no network calls."""
from unittest.mock import Mock

import pandas as pd
import pytest

import nba_app.data as data_module
from nba_app.config import Settings
from nba_app.data import DataUnavailableError, NBADataClient


@pytest.fixture
def career_client(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module.players, "get_players", lambda: [
        {"id": 2544, "full_name": "LeBron James", "is_active": True},
    ])
    monkeypatch.setattr(data_module, "ShotChartDetail", Mock(side_effect=AssertionError("Unexpected network")))
    monkeypatch.setattr(data_module, "PlayerCareerStats", Mock(side_effect=AssertionError("Unexpected network")))
    return NBADataClient(Settings(cache_dir=tmp_path, retries=0))


def set_career(monkeypatch, frame):
    endpoint = Mock()
    endpoint.get_data_frames.return_value = [frame]
    upstream = Mock(return_value=endpoint)
    monkeypatch.setattr(data_module, "PlayerCareerStats", upstream)
    return upstream


def test_only_supported_played_seasons_are_returned(career_client, monkeypatch):
    upstream = set_career(monkeypatch, pd.DataFrame([
        {"SEASON_ID": "2023-24", "TEAM_ID": 0, "FGA": 500},
        {"SEASON_ID": "2023-24", "TEAM_ID": 1, "FGA": 250},
        {"SEASON_ID": "2023-24", "TEAM_ID": 2, "FGA": 250},
        {"SEASON_ID": "2024-25", "TEAM_ID": 1, "FGA": 12},
        {"SEASON_ID": "2022-23", "TEAM_ID": 1, "FGA": 0},
        {"SEASON_ID": "1996-97", "TEAM_ID": 1, "FGA": 500},
        {"SEASON_ID": "2099-00", "TEAM_ID": 1, "FGA": 500},
        {"SEASON_ID": "2023-25", "TEAM_ID": 1, "FGA": 500},
    ]))
    result = career_client.fetch_player_seasons(2544)
    assert result["seasons"] == ["2024-25", "2023-24"]
    assert result["player_id"] == 2544
    assert result["player_name"] == "LeBron James"
    assert result["provenance"]["retrieval"] == "live_nba"
    assert career_client.fetch_player_seasons(2544)["provenance"]["retrieval"] == "cache"
    assert upstream.call_count == 1
    assert upstream.call_args.kwargs == {"player_id": 2544, "timeout": 10}


def test_career_seasons_share_free_throw_cache(career_client, monkeypatch):
    upstream = set_career(monkeypatch, pd.DataFrame([
        {"SEASON_ID": "2023-24", "TEAM_ID": 0, "FGA": 500, "FTM": 100, "FTA": 125},
    ]))
    assert career_client.fetch_freethrow(2544, "2023-24")["ft_pct"] == 0.8
    assert career_client.fetch_player_seasons(2544)["seasons"] == ["2023-24"]
    assert upstream.call_count == 1


@pytest.mark.parametrize("rows", [[], [{"SEASON_ID": "1996-97", "TEAM_ID": 1, "FGA": 4}],
                                  [{"SEASON_ID": "2023-24", "TEAM_ID": 1, "FGA": 0}]])
def test_no_eligible_seasons_is_an_explicit_empty_state(career_client, monkeypatch, rows):
    upstream = set_career(monkeypatch, pd.DataFrame(rows, columns=["SEASON_ID", "TEAM_ID", "FGA"]))
    assert career_client.fetch_player_seasons(2544)["seasons"] == []
    assert career_client.fetch_player_seasons(2544)["seasons"] == []
    assert upstream.call_count == 1


def test_missing_schema_is_unavailable(career_client, monkeypatch):
    set_career(monkeypatch, pd.DataFrame([{"SEASON_ID": "2023-24"}]))
    with pytest.raises(DataUnavailableError, match="missing"):
        career_client.fetch_player_seasons(2544)


@pytest.mark.parametrize("attempts", [-1, 0.2, float("nan"), float("inf"), "unknown"])
def test_invalid_career_attempt_counts_are_rejected(career_client, monkeypatch, attempts):
    set_career(monkeypatch, pd.DataFrame([{"SEASON_ID": "2023-24", "TEAM_ID": 1, "FGA": attempts}]))
    with pytest.raises(DataUnavailableError, match="invalid"):
        career_client.fetch_player_seasons(2544)


def test_unknown_player_does_not_call_nba(career_client):
    with pytest.raises(ValueError, match="Player not found"):
        career_client.fetch_player_seasons(99999999)
    assert data_module.PlayerCareerStats.call_count == 0


def test_empty_shot_data_keeps_meaningful_message(career_client, monkeypatch):
    endpoint = Mock()
    endpoint.get_data_frames.return_value = [pd.DataFrame()]
    monkeypatch.setattr(data_module, "ShotChartDetail", Mock(return_value=endpoint))
    with pytest.raises(DataUnavailableError, match="no records for this player and season"):
        career_client.fetch_shots(2544, "2023-24")
