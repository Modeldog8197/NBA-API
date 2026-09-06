"""Local player preparation is convenient without enabling anonymous hosted writes."""
import pytest
from fastapi.testclient import TestClient

import nba_app.api as api_module
from nba_app.config import Settings
from nba_app.preparation import PreparationBusyError


class PreparationStub:
    def __init__(self, *args):
        self.calls = []

    def readiness(self, player_id, season):
        return {"player_id": player_id, "player_name": "Test player", "season": season,
                "status": "not_prepared", "message": "Prepare this player."}

    def prepare(self, player_id, season):
        self.calls.append((player_id, season))
        return {**self.readiness(player_id, season), "status": "queued", "job_id": "test-job"}

    def job_status(self, job_id):
        if job_id != "test-job":
            raise LookupError("Unknown job")
        return {**self.readiness(2544, "2023-24"), "status": "training", "job_id": job_id}

    def close(self, wait=False):
        pass


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "ModelPreparationService", PreparationStub)
    return api_module.create_app(Settings(model_dir=tmp_path / "models", cache_dir=tmp_path / "cache",
                                         local_player_models=True))


def browser_client(app, host="127.0.0.1", address="127.0.0.1"):
    return TestClient(app, base_url=f"http://{host}:8000", client=(address, 1234))


def test_local_session_can_prepare_and_poll_without_admin_key(local_app):
    with browser_client(local_app) as client:
        session = client.get("/session")
        token = session.json()["session_token"]
        assert token and not session.json()["requires_key"]
        assert session.headers["cache-control"] == "no-store"
        response = client.post("/models/prepare", json={"player_id": 2544, "season": "2023-24"},
                               headers={"X-Local-Session": token, "Origin": "http://127.0.0.1:8000"})
        assert response.status_code == 202
        assert response.json()["player_id"] == 2544
        assert client.get("/models/preparation/test-job").json()["status"] == "training"
        assert client.get("/models/preparation/unknown").status_code == 404
        assert client.get("/models/readiness?player_id=2544&season=2023-24").json()["status"] == "not_prepared"


@pytest.mark.parametrize("host,address,headers", [
    ("example.com", "127.0.0.1", {}),
    ("127.0.0.1", "192.0.2.10", {}),
    ("127.0.0.1", "127.0.0.1", {"Origin": "https://untrusted.example"}),
    ("localhost", "127.0.0.1", {"Sec-Fetch-Site": "cross-site"}),
])
def test_session_never_exposes_local_token_to_remote_or_cross_site(local_app, host, address, headers):
    with browser_client(local_app, host, address) as client:
        response = client.get("/session", headers=headers)
        assert response.json() == {"preparation_enabled": False, "requires_key": True, "session_token": None}
        assert client.post("/models/prepare", headers=headers,
                           json={"player_id": 2544, "season": "2023-24"}).status_code == 401
        assert not local_app.state.preparation.calls


@pytest.mark.parametrize("headers", [{}, {"X-Local-Session": "wrong-token"},
                                      {"Origin": "https://untrusted.example"}])
def test_local_post_requires_valid_same_origin_session(local_app, headers):
    with browser_client(local_app) as client:
        assert client.post("/models/prepare", headers=headers,
                           json={"player_id": 2544, "season": "2023-24"}).status_code == 401


def test_token_does_not_bypass_origin_guard(local_app):
    with browser_client(local_app) as client:
        token = client.get("/session").json()["session_token"]
        response = client.post("/models/prepare", json={"player_id": 2544, "season": "2023-24"},
                               headers={"X-Local-Session": token, "Origin": "https://untrusted.example"})
        assert response.status_code == 401


def test_hosted_preparation_requires_admin_key(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "ModelPreparationService", PreparationStub)
    key = "a-long-test-training-key"
    app = api_module.create_app(Settings(model_dir=tmp_path / "models", training_enabled=True,
                                         training_api_key=key))
    with browser_client(app, "app.example", "192.0.2.10") as client:
        assert client.get("/session").json() == {"preparation_enabled": True, "requires_key": True, "session_token": None}
        payload = {"player_id": 2544, "season": "2023-24"}
        assert client.post("/models/prepare", json=payload).status_code == 401
        assert client.post("/models/prepare", json=payload,
                           headers={"Authorization": f"Bearer {key}"}).status_code == 202


def test_queue_capacity_returns_retry_after(local_app, monkeypatch):
    def full(*args):
        raise PreparationBusyError("Queue full")
    monkeypatch.setattr(local_app.state.preparation, "prepare", full)
    with browser_client(local_app) as client:
        token = client.get("/session").json()["session_token"]
        result = client.post("/models/prepare", json={"player_id": 2544, "season": "2023-24"},
                             headers={"X-Local-Session": token})
        assert result.status_code == 429
        assert result.headers["Retry-After"] == "5"
