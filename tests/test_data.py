"""Exercise real data-client behavior while replacing every NBA endpoint."""
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pandas as pd
import pytest

import nba_app.data as data_module
from nba_app.config import Settings, validate_season
from nba_app.data import DataUnavailableError, NBADataClient


@pytest.fixture
def data_client(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module.players, "get_players", lambda: [
        {"id": 201939, "full_name": "Stephen Curry", "is_active": True},
        {"id": 1, "full_name": "Historical Curry", "is_active": False},
    ])
    monkeypatch.setattr(data_module.time, "sleep", lambda seconds: None)
    # A forgotten mock must never reach the external NBA service.
    monkeypatch.setattr(data_module, "ShotChartDetail", Mock(side_effect=AssertionError("Unexpected NBA request")))
    monkeypatch.setattr(data_module, "PlayerCareerStats", Mock(side_effect=AssertionError("Unexpected NBA request")))
    return NBADataClient(Settings(model_dir=tmp_path / "models", cache_dir=tmp_path / "cache", request_timeout=3, retries=2))


def response_with(df):
    endpoint = Mock()
    endpoint.get_data_frames.return_value = [df]
    return endpoint


def test_static_player_selection_uses_unique_ids(data_client):
    assert data_client.resolve_player("201939") == {"id": 201939, "name": "Stephen Curry"}
    assert data_client.resolve_player(" stephen curry ")["id"] == 201939
    assert data_client.search_players("curry")[0]["id"] == 201939
    assert data_client.search_players("c") == []
    with pytest.raises(ValueError, match="stable NBA ID"):
        data_client.resolve_player("Curry")


def test_fetch_passes_timeout_filters_and_uses_fresh_cache(data_client, monkeypatch):
    shots = pd.DataFrame([{"GAME_ID": "0022300001", "GAME_EVENT_ID": 1, "LOC_X": 20, "LOC_Y": 30}])
    upstream = Mock(return_value=response_with(shots))
    monkeypatch.setattr(data_module, "ShotChartDetail", upstream)
    first, live = data_client.fetch_shots(201939, "2023-24")
    second, cached = data_client.fetch_shots(201939, "2023-24")
    assert upstream.call_count == 1
    assert upstream.call_args.kwargs == {
        "team_id": 0, "player_id": 201939, "season_nullable": "2023-24",
        "season_type_all_star": "Regular Season", "context_measure_simple": "FGA", "timeout": 3,
    }
    pd.testing.assert_frame_equal(first, second)
    assert live["retrieval"] == "live_nba"
    assert cached["retrieval"] == "cache"
    assert live["sha256"] == cached["sha256"]
    assert cached["stale"] is False


def test_timeout_retries_are_bounded(data_client, monkeypatch):
    upstream = Mock(side_effect=TimeoutError("NBA is unreachable"))
    monkeypatch.setattr(data_module, "ShotChartDetail", upstream)
    with pytest.raises(DataUnavailableError, match="bounded attempts"):
        data_client.fetch_shots(201939, "2023-24")
    assert upstream.call_count == data_client.settings.retries + 1
    assert all(call.kwargs["timeout"] == 3 for call in upstream.call_args_list)


def test_transient_failure_recovers_and_caches(data_client, monkeypatch):
    shots = pd.DataFrame([{"LOC_X": 30, "LOC_Y": 40}])
    upstream = Mock(side_effect=[TimeoutError(), response_with(shots)])
    monkeypatch.setattr(data_module, "ShotChartDetail", upstream)
    actual, source = data_client.fetch_shots(201939, "2023-24")
    pd.testing.assert_frame_equal(actual, shots)
    assert upstream.call_count == 2
    assert source["retrieval"] == "live_nba"


def test_stale_cache_fallback_is_explicit_and_can_be_disabled(data_client, monkeypatch):
    key = "shots-201939-2023-24-regular-v1"
    data_client.settings.cache_dir.mkdir()
    payload = {"key": key, "fetched_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
               "records": [{"GAME_ID": "historic", "LOC_X": 10, "LOC_Y": 20}]}
    path = data_client.settings.cache_dir / f"{key}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    upstream = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(data_module, "ShotChartDetail", upstream)
    rows, info = data_client.fetch_shots(201939, "2023-24")
    assert rows.iloc[0]["GAME_ID"] == "historic"
    assert info["retrieval"] == "stale_cache"
    assert info["stale"] is True
    assert info["fetched_at"] == payload["fetched_at"]
    strict_client = NBADataClient(Settings(cache_dir=data_client.settings.cache_dir, allow_stale_cache=False, retries=0))
    with pytest.raises(DataUnavailableError):
        strict_client.fetch_shots(201939, "2023-24")


def test_corrupt_cache_is_refreshed(data_client, monkeypatch):
    data_client.settings.cache_dir.mkdir()
    path = data_client.settings.cache_dir / "shots-201939-2023-24-regular-v1.json"
    path.write_text("{broken json", encoding="utf-8")
    monkeypatch.setattr(data_module, "ShotChartDetail", Mock(return_value=response_with(pd.DataFrame([{"LOC_X": 1, "LOC_Y": 2}]))))
    _, info = data_client.fetch_shots(201939, "2023-24")
    assert info["retrieval"] == "live_nba"
    assert json.loads(path.read_text(encoding="utf-8"))["records"] == [{"LOC_X": 1, "LOC_Y": 2}]
    assert list(data_client.settings.cache_dir.glob("*.tmp")) == []


def test_empty_nba_response_is_not_cached_or_repeated(data_client, monkeypatch):
    upstream = Mock(return_value=response_with(pd.DataFrame()))
    monkeypatch.setattr(data_module, "ShotChartDetail", upstream)
    with pytest.raises(DataUnavailableError):
        data_client.fetch_shots(201939, "2023-24")
    assert upstream.call_count == 1
    assert not data_client.settings.cache_dir.exists()


def test_free_throws_match_exact_season_and_prefer_traded_player_total(data_client, monkeypatch):
    career = pd.DataFrame([
        {"SEASON_ID": "2023-24", "TEAM_ID": 0, "FTM": 90, "FTA": 100},
        {"SEASON_ID": "2023-24", "TEAM_ID": 10, "FTM": 40, "FTA": 45},
        {"SEASON_ID": "2023-24", "TEAM_ID": 11, "FTM": 50, "FTA": 55},
        {"SEASON_ID": "2024-25", "TEAM_ID": 10, "FTM": 1, "FTA": 10},
    ])
    upstream = Mock(return_value=response_with(career))
    monkeypatch.setattr(data_module, "PlayerCareerStats", upstream)
    result = data_client.fetch_freethrow(201939, "2023-24")
    assert (result["ftm"], result["fta"], result["ft_pct"]) == (90, 100, 0.9)
    assert result["season"] == "2023-24"
    assert upstream.call_args.kwargs["timeout"] == 3
    with pytest.raises(LookupError, match="selected season"):
        data_client.fetch_freethrow(201939, "2022-23")
    assert upstream.call_count == 1


def test_free_throws_sum_team_rows_without_total_and_handle_no_attempts(data_client, monkeypatch):
    career = pd.DataFrame([
        {"SEASON_ID": "2023-24", "TEAM_ID": 10, "FTM": 4, "FTA": 5},
        {"SEASON_ID": "2023-24", "TEAM_ID": 11, "FTM": 5, "FTA": 5},
        {"SEASON_ID": "2024-25", "TEAM_ID": 10, "FTM": 0, "FTA": 0},
    ])
    monkeypatch.setattr(data_module, "PlayerCareerStats", Mock(return_value=response_with(career)))
    assert data_client.fetch_freethrow(201939, "2023-24")["ft_pct"] == 0.9
    assert data_client.fetch_freethrow(201939, "2024-25")["ft_pct"] is None


@pytest.mark.parametrize("ftm,fta", [(float("nan"), 5), (-1, 5), (6, 5), (float("inf"), 5), (1.5, 5)])
def test_malformed_free_throw_counts_are_rejected(data_client, monkeypatch, ftm, fta):
    career = pd.DataFrame([{"SEASON_ID": "2023-24", "TEAM_ID": 0, "FTM": ftm, "FTA": fta}])
    monkeypatch.setattr(data_module, "PlayerCareerStats", Mock(return_value=response_with(career)))
    with pytest.raises(DataUnavailableError):
        data_client.fetch_freethrow(201939, "2023-24")


@pytest.mark.parametrize("season", ["2023", "2023-25", "1990-91", "2023-2024", "2023-24/../../other"])
def test_invalid_seasons_are_rejected_before_network(data_client, season):
    with pytest.raises(ValueError):
        data_client.fetch_shots(201939, season)
    assert data_module.ShotChartDetail.call_count == 0


@pytest.mark.parametrize("overrides", [
    {"training_enabled": True}, {"training_enabled": True, "training_api_key": "short"},
    {"request_timeout": 0}, {"request_timeout": 61}, {"retries": 4}, {"retries": -1},
    {"cache_ttl_seconds": -1}, {"cors_origins": ("*",)}, {"cors_origins": ("https://example.org/path",)},
])
def test_invalid_configuration_fails_early(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_valid_season_rollover():
    assert validate_season("1999-00") == "1999-00"


@pytest.mark.parametrize("days,stale", [(0, False), (5, True)])
def test_hosted_snapshot_preserves_age_and_never_calls_nba(tmp_path, days, stale):
    key = "shots-201939-2023-24-regular-v1"
    payload = {"key": key, "fetched_at": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
               "records": [{"PLAYER_ID": 201939, "LOC_X": 10, "LOC_Y": 20}]}
    (tmp_path / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    loader = Mock(side_effect=AssertionError("Hosted snapshots must not call NBA"))
    client = NBADataClient(Settings(cache_dir=tmp_path, snapshot_only=True))
    rows, provenance = client._cached_request(key, loader)
    assert rows.iloc[0]["PLAYER_ID"] == 201939
    assert provenance["retrieval"] == "saved_snapshot"
    assert provenance["stale"] is stale
    assert provenance["fetched_at"] == payload["fetched_at"]
    loader.assert_not_called()


@pytest.mark.parametrize("content", [None, "{broken", '{"key":"wrong","records":[]}'])
def test_hosted_missing_or_corrupt_snapshot_fails_without_network(tmp_path, content):
    if content is not None:
        (tmp_path / "test.json").write_text(content, encoding="utf-8")
    loader = Mock()
    client = NBADataClient(Settings(cache_dir=tmp_path, snapshot_only=True))
    with pytest.raises(DataUnavailableError, match="No eligible snapshot"):
        client._cached_request("test", loader)
    loader.assert_not_called()


def test_hosted_snapshot_respects_strict_staleness_setting(tmp_path):
    payload = {"key": "test", "fetched_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
               "records": [{"value": 1}]}
    (tmp_path / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    loader = Mock()
    client = NBADataClient(Settings(cache_dir=tmp_path, snapshot_only=True, allow_stale_cache=False))
    with pytest.raises(DataUnavailableError, match="No eligible snapshot"):
        client._cached_request("test", loader)
    loader.assert_not_called()
