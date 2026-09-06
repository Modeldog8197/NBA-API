"""HTTP regression tests: real model inference, synthetic data, no NBA traffic."""
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

import nba_app.api as api_module
import nba_app.data as data_module
from nba_app.api import create_app
from nba_app.config import Settings
from nba_app.data import DataUnavailableError
from nba_app.models import ModelStore
from nba_app.training import train_bundle
from scripts.generate_fixture import make_fixture

KEY = "test-secret-key-123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}
MISSING_MODEL = "p999-2023-24-20260906T000000000000Z-abcdef12"


class FixtureDataClient:
    """Explicitly synthetic source used only at the HTTP dependency boundary."""

    def resolve_player(self, player_id):
        if player_id not in {42, 84}:
            raise ValueError("Unknown fixture player ID")
        return {"id": player_id, "name": f"Synthetic player {player_id}"}

    def search_players(self, q):
        return [self.resolve_player(42)] if "synthetic" in q.lower() else []

    def fetch_shots(self, player_id, season):
        self.resolve_player(player_id)
        return make_fixture(player_id=player_id, season=season), {"source": "synthetic", "retrieval": "test_fixture"}

    def fetch_freethrow(self, player_id, season):
        player = self.resolve_player(player_id)
        if season == "2022-23":
            raise LookupError("No historical free-throw statistics for the selected season.")
        return {"player_id": player_id, "player_name": player["name"], "season": season,
                "ft_pct": 0.9, "ftm": 9, "fta": 10, "source": "synthetic historical fixture",
                "provenance": {"retrieval": "test_fixture"}}


@pytest.fixture(autouse=True)
def no_external_nba(monkeypatch):
    monkeypatch.setattr(data_module, "ShotChartDetail", Mock(side_effect=AssertionError("NBA network must be mocked")))
    monkeypatch.setattr(data_module, "PlayerCareerStats", Mock(side_effect=AssertionError("NBA network must be mocked")))


@pytest.fixture(scope="module")
def trained_store(tmp_path_factory):
    directory = tmp_path_factory.mktemp("api-models")
    store = ModelStore(directory)
    first = train_bundle(make_fixture(player_id=42), 42, "Synthetic player 42", "2023-24", store, source="synthetic")
    second = train_bundle(make_fixture(player_id=84, seed=84), 84, "Synthetic player 84", "2023-24", store, source="synthetic")
    return store, first, second


@pytest.fixture
def client(trained_store, tmp_path):
    store, _, _ = trained_store
    config = Settings(model_dir=tmp_path / "training-models", cache_dir=tmp_path / "cache",
                      training_enabled=True, training_api_key=KEY)
    with TestClient(create_app(config, data_client=FixtureDataClient(), store=store)) as http:
        yield http


def shot(model_id, **changes):
    return {"model_id": model_id, "loc_x": 230, "loc_y": 0, **changes}


def test_empty_installation_health_and_missing_model(tmp_path):
    config = Settings(model_dir=tmp_path / "models", cache_dir=tmp_path / "cache")
    with TestClient(create_app(config, data_client=FixtureDataClient())) as empty:
        response = empty.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "training_enabled": False, "models_available": 0}
        assert empty.get("/models").json() == {"models": []}
        assert empty.post("/predict", json=shot(MISSING_MODEL)).status_code == 404
        assert empty.get(f"/models/{MISSING_MODEL}").status_code == 404


def test_court_seasons_and_player_search(client):
    geometry = client.get("/court").json()
    assert geometry["coordinate_units"] == "tenths of a foot"
    assert geometry["arc_radius"] == 237.5
    assert geometry["corner_x"] == 220
    assert geometry["boundary_shot_value"] == 2
    seasons = client.get("/seasons").json()
    assert seasons["default_season"] in seasons["seasons"]
    assert "2023-24" in seasons["seasons"]
    assert client.get("/players/search", params={"q": "synthetic"}).json()[0]["id"] == 42


def test_prediction_reports_identity_geometry_expectation_and_checked_explanation(client, trained_store):
    _, first, _ = trained_store
    response = client.post("/predict", json=shot(first["model_id"]))
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["model_id"], body["player_id"], body["season"]) == (first["model_id"], 42, "2023-24")
    assert body["shot_type"] == "3pt"
    assert body["shot_zone"] == "Right Corner 3"
    assert body["distance_ft"] == 23
    assert body["expected_points"] == pytest.approx(3 * body["make_probability"])
    assert body["uncertainty"] is None
    assert "unavailable" in body["uncertainty_note"]
    assert "confidence_low" not in body and "confidence_high" not in body
    explanation = body["explanation"]
    assert explanation["units"] == "raw log-odds"
    reconstructed = explanation["base_value"] + sum(item["value"] for item in explanation["contributions"])
    assert reconstructed == pytest.approx(explanation["raw_margin"], abs=2e-5)
    assert explanation["reconstructed_probability"] == pytest.approx(body["make_probability"], abs=2e-5)


@pytest.mark.parametrize("changes", [
    {"loc_x": 251}, {"loc_y": -53}, {"loc_y": 418}, {"loc_x": "230"},
    {"loc_x": True}, {"loc_x": None}, {"shot_distance": 1}, {"shot_type": "2pt"},
    {"shot_type": "free-throw"}, {"unexpected": "field"},
])
def test_invalid_or_conflicting_shots_return_422(client, trained_store, changes):
    response = client.post("/predict", json=shot(trained_store[1]["model_id"], **changes))
    assert response.status_code == 422, response.text


def test_nonfinite_json_coordinate_is_rejected(client, trained_store):
    body = '{"model_id":"' + trained_store[1]["model_id"] + '","loc_x":NaN,"loc_y":0}'
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_model_id_is_required_and_malformed_ids_are_rejected(client):
    assert client.post("/predict", json={"loc_x": 0, "loc_y": 0}).status_code == 422
    assert client.post("/predict", json=shot("../model.json")).status_code == 422
    assert client.post("/predict", json=shot(MISSING_MODEL)).status_code == 404


def test_batch_matches_single_prediction_and_preserves_order(client, trained_store):
    model_id = trained_store[1]["model_id"]
    shots = [{"loc_x": 0, "loc_y": 30}, {"loc_x": 230, "loc_y": 0}, {"loc_x": 0, "loc_y": 250}]
    response = client.post("/predict/batch", json={"model_id": model_id, "shots": shots, "explain": False})
    assert response.status_code == 200, response.text
    batch = response.json()
    assert batch["model_id"] == model_id
    for inputs, result in zip(shots, batch["predictions"], strict=True):
        single = client.post("/predict", json={"model_id": model_id, **inputs, "explain": False}).json()
        assert result == single
        assert result["explanation"] is None


@pytest.mark.parametrize("shots", [[], [{"loc_x": 0, "loc_y": 0}] * 501,
                                     [{"loc_x": 0, "loc_y": 0}, {"loc_x": 230, "loc_y": 0, "shot_type": "2pt"}]])
def test_invalid_batches_fail_as_a_whole(client, trained_store, shots):
    response = client.post("/predict/batch", json={"model_id": trained_store[1]["model_id"], "shots": shots})
    assert response.status_code == 422
    assert "predictions" not in response.json()


def test_model_selection_is_isolated_between_clients(client, trained_store):
    store, first, second = trained_store
    before = client.post("/predict", json=shot(first["model_id"])).json()
    other = client.post("/predict", json=shot(second["model_id"])).json()
    after = client.post("/predict", json=shot(first["model_id"])).json()
    assert before == after
    assert other["player_id"] == 84
    assert other["model_id"] != before["model_id"]
    assert {item["model_id"] for item in client.get("/models").json()["models"]} == {first["model_id"], second["model_id"]}
    assert client.get(f"/models/{first['model_id']}").json()["player_id"] == 42
    assert len(store.list_models()) == 2


def test_retraining_is_disabled_by_default(tmp_path):
    with TestClient(create_app(Settings(model_dir=tmp_path / "models"), data_client=FixtureDataClient())) as disabled:
        result = disabled.post("/model/retrain", json={"player_id": 42, "season": "2023-24"}, headers=AUTH)
        assert result.status_code == 403
        assert "disabled" in result.json()["detail"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY}])
def test_retraining_requires_bearer_secret(client, headers):
    response = client.post("/model/retrain", json={"player_id": 42, "season": "2023-24"}, headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert not client.app.state.settings.model_dir.exists()


def test_failed_training_retains_old_model_and_releases_lock(client, trained_store, monkeypatch):
    old_id = trained_store[1]["model_id"]
    before = client.post("/predict", json=shot(old_id)).json()
    failure = Mock(side_effect=RuntimeError("internal error that must not appear in the response"))
    monkeypatch.setattr(api_module, "train_bundle", failure)
    response = client.post("/model/retrain", json={"player_id": 42, "season": "2023-24"}, headers=AUTH)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    status = client.get("/model/retrain/status", params={"job_id": job_id}).json()
    assert status["status"] == "error"
    assert status["model_id"] is None
    assert "internal error" not in status["error"]
    assert not (client.app.state.settings.model_dir / ".api-training.lock").exists()
    assert client.post("/predict", json=shot(old_id)).json() == before
    assert len(client.get("/models").json()["models"]) == 2
    assert failure.call_count == 1


def test_fetch_failure_has_actionable_job_error(client, monkeypatch):
    monkeypatch.setattr(client.app.state.data_client, "fetch_shots", Mock(side_effect=DataUnavailableError("NBA is unavailable; no cached records exist.")))
    response = client.post("/model/retrain", json={"player_id": 42, "season": "2023-24"}, headers=AUTH)
    status = client.get("/model/retrain/status", params={"job_id": response.json()["job_id"]}).json()
    assert status["status"] == "error"
    assert "no cached records" in status["message"]
    assert client.get("/model/retrain/status", params={"job_id": "unknown"}).status_code == 404


def test_existing_training_lock_prevents_duplicate_job(client):
    directory = client.app.state.settings.model_dir
    directory.mkdir()
    (directory / ".api-training.lock").write_text("another-process", encoding="utf-8")
    response = client.post("/model/retrain", json={"player_id": 42, "season": "2023-24"}, headers=AUTH)
    assert response.status_code == 409
    assert client.app.state.jobs == {}


def test_successful_retraining_publishes_selectable_version(tmp_path):
    config = Settings(model_dir=tmp_path / "models", cache_dir=tmp_path / "cache",
                      training_enabled=True, training_api_key=KEY)
    with TestClient(create_app(config, data_client=FixtureDataClient())) as http:
        response = http.post("/model/retrain", json={"player_id": 42, "season": "2023-24"}, headers=AUTH)
        assert response.status_code == 202, response.text
        job = http.get("/model/retrain/status", params={"job_id": response.json()["job_id"]}).json()
        assert job["status"] == "done", job
        assert job["error"] is None
        model_id = job["model_id"]
        detail = http.get(f"/models/{model_id}").json()
        assert detail["player_id"] == 42
        assert detail["season"] == "2023-24"
        assert detail["is_demo"] is True
        assert detail["data_source"]["retrieval"] == "test_fixture"
        predicted = http.post("/predict", json=shot(model_id))
        assert predicted.status_code == 200
        assert predicted.json()["model_id"] == model_id
        assert not (config.model_dir / ".api-training.lock").exists()


def test_free_throw_endpoint_requires_and_preserves_season(client):
    assert client.get("/player/freethrow", params={"player_id": 42}).status_code == 422
    assert client.get("/player/freethrow", params={"player_id": 42, "season": "2023-25"}).status_code == 422
    assert client.get("/player/freethrow", params={"player_id": 42, "season": "2022-23"}).status_code == 404
    result = client.get("/player/freethrow", params={"player_id": 42, "season": "2023-24"}).json()
    assert result["season"] == "2023-24"
    assert result["ft_pct"] == 0.9
    assert "make_probability" not in result


def test_upstream_data_outage_returns_service_unavailable(client, monkeypatch):
    monkeypatch.setattr(client.app.state.data_client, "fetch_freethrow", Mock(side_effect=DataUnavailableError("NBA unavailable; try again later.")))
    response = client.get("/player/freethrow", params={"player_id": 42, "season": "2023-24"})
    assert response.status_code == 503
    assert response.json()["detail"] == "NBA unavailable; try again later."


def test_cors_only_allows_configured_origin(tmp_path):
    config = Settings(model_dir=tmp_path / "models", cors_origins=("https://dashboard.example",))
    with TestClient(create_app(config, data_client=FixtureDataClient())) as http:
        good = http.options("/predict", headers={"Origin": "https://dashboard.example", "Access-Control-Request-Method": "POST"})
        bad = http.options("/predict", headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"})
        assert good.status_code == 200
        assert good.headers["Access-Control-Allow-Origin"] == "https://dashboard.example"
        assert bad.status_code == 400
        assert "Access-Control-Allow-Origin" not in bad.headers
