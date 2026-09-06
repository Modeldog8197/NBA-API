"""The public preparation flow never mixes players, seasons, or job ownership."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import nba_app.preparation as preparation_module
from nba_app.config import Settings
from nba_app.data import DataUnavailableError
from nba_app.models import ModelStore, ModelValidationError
from nba_app.preparation import ModelPreparationService, PreparationBusyError
from nba_app.training import TrainingError
from scripts.generate_fixture import make_fixture


class MemoryStore:
    """Thread-safe enough for single-worker tests; models still load explicitly."""

    def __init__(self):
        self.models = {}
        self.invalid = set()
        self.loads = []

    def list_models(self):
        snapshot = list(self.models.values())
        return sorted(snapshot, key=lambda model: model["created_at"], reverse=True)

    def load(self, model_id):
        self.loads.append(model_id)
        if model_id in self.invalid:
            raise ModelValidationError("Artifact integrity check failed")
        return SimpleNamespace(metadata=dict(self.models[model_id]))


def metadata(player_id=2544, season="2023-24", model_id=None, **extra):
    return {"model_id": model_id or f"test-{player_id}-{season}", "player_id": player_id,
            "player_name": f"Player {player_id}", "season": season, "n_shots": 500, "is_demo": False,
            "created_at": datetime.now(UTC).isoformat(), **extra}


@pytest.fixture
def service(tmp_path, monkeypatch):
    client = Mock()
    client.resolve_player.side_effect = lambda player_id: {"id": player_id, "name": f"Player {player_id}"}
    client.fetch_player_seasons.side_effect = lambda player_id: {"player_id": player_id, "seasons": ["2023-24", "2024-25"]}
    client.fetch_shots.side_effect = lambda player_id, season: (pd.DataFrame([{"PLAYER_ID": player_id}]), {"source": "nba"})
    store = MemoryStore()
    instance = ModelPreparationService(Settings(model_dir=tmp_path / "models", cache_dir=tmp_path / "cache"), client, store)

    def fake_train(shots, player_id, player_name, season, registry, source_info, progress):
        assert shots.iloc[0]["PLAYER_ID"] == player_id
        assert source_info == {"source": "nba"}
        progress("Evaluating the selected model")
        progress("Verifying and publishing model")
        saved = metadata(player_id, season)
        registry.models[saved["model_id"]] = saved
        return saved

    trainer = Mock(side_effect=fake_train)
    monkeypatch.setattr(preparation_module, "train_bundle", trainer)
    instance.trainer = trainer
    yield instance
    instance.close(wait=True)


def finished(service, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.job_status(job_id)
        if job["status"] in {"done", "unavailable", "error"}:
            return job
        time.sleep(0.005)
    raise AssertionError("Preparation did not complete within the test deadline")


def test_two_players_and_seasons_keep_explicit_identity(service):
    first = service.prepare(2544, "2023-24")
    second = service.prepare(203999, "2024-25")
    for initial in (first, second):
        job = finished(service, initial["job_id"])
        assert job["status"] == "done"
        assert (job["player_id"], job["season"]) == (initial["player_id"], initial["season"])
        saved = service.store.load(job["model_id"]).metadata
        assert saved["player_id"] == initial["player_id"]
        assert saved["season"] == initial["season"]
    assert service.trainer.call_count == 2
    assert service.readiness(2544, "2023-24")["status"] == "ready"
    assert service.readiness(2544, "2024-25")["status"] == "not_prepared"
    assert not (service.settings.model_dir / ".api-training.lock").exists()


def test_concurrent_duplicate_requests_share_one_job(service):
    entered, release = Event(), Event()

    def blocked_fetch(player_id, season):
        entered.set()
        assert release.wait(10)
        return pd.DataFrame([{"PLAYER_ID": player_id}]), {"source": "nba"}

    service.client.fetch_shots.side_effect = blocked_fetch
    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            requests = list(callers.map(lambda _: service.prepare(2544, "2023-24"), range(4)))
        assert entered.wait(5)
        assert len({job["job_id"] for job in requests}) == 1
        status = service.readiness(2544, "2023-24")
        assert status["status"] == "preparing"
        status["player_id"] = 123
        assert service.job_status(requests[0]["job_id"])["player_id"] == 2544
    finally:
        release.set()
    assert finished(service, requests[0]["job_id"])["status"] == "done"
    assert service.trainer.call_count == 1


def test_ready_model_is_loaded_and_reused_without_training(service):
    valid = metadata(model_id="valid", created_at="2024-01-01")
    invalid = metadata(model_id="invalid", created_at="2025-01-01")
    service.store.models = {"valid": valid, "invalid": invalid}
    service.store.invalid.add("invalid")
    assert service.readiness(2544, "2023-24")["model_id"] == "valid"
    ready = service.prepare(2544, "2023-24")
    assert ready["status"] == "done"
    assert ready["model_id"] == "valid"
    assert ready["job_id"] is None
    assert service.store.loads == ["invalid", "valid", "invalid", "valid"]
    service.client.fetch_shots.assert_not_called()
    service.trainer.assert_not_called()


def test_demo_and_other_player_models_never_satisfy_readiness(service):
    demo = metadata(model_id="demo", is_demo=True)
    other = metadata(201939, model_id="curry")
    service.store.models = {"demo": demo, "curry": other}
    assert service.readiness(2544, "2023-24")["status"] == "not_prepared"
    assert service.store.loads == []


def test_missing_player_season_reports_unavailable_without_training(service):
    service.client.fetch_player_seasons.side_effect = lambda player_id: {"player_id": player_id, "seasons": ["2024-25"]}
    job = finished(service, service.prepare(2544, "2023-24")["job_id"])
    assert job["status"] == "unavailable"
    assert "2023-24" in job["message"]
    assert "Choose a season" in job["message"]
    service.client.fetch_shots.assert_not_called()
    service.trainer.assert_not_called()
    assert service.readiness(2544, "2023-24")["status"] == "unavailable"


@pytest.mark.parametrize("failure", [DataUnavailableError("NBA request timed out"), TrainingError("Need at least 150 usable shots and 15 games")])
def test_unavailable_data_is_not_retried_until_requested(service, failure):
    original = service.trainer.side_effect
    service.trainer.side_effect = failure
    first = finished(service, service.prepare(2544, "2023-24")["job_id"])
    assert first["status"] == "unavailable"
    assert first["error"] == str(failure)
    assert service.readiness(2544, "2023-24")["status"] == "unavailable"
    assert service.trainer.call_count == 1
    assert not (service.settings.model_dir / ".api-training.lock").exists()
    service.trainer.side_effect = original
    retry = service.prepare(2544, "2023-24")
    assert retry["job_id"] != first["job_id"]
    assert finished(service, retry["job_id"])["status"] == "done"
    assert service.job_status(first["job_id"])["status"] == "unavailable"


def test_exception_does_not_remove_previous_models_or_expose_internals(service):
    previous = metadata(201939)
    service.store.models[previous["model_id"]] = previous
    service.trainer.side_effect = RuntimeError("private filesystem location")
    job = finished(service, service.prepare(2544, "2023-24")["job_id"])
    assert job["status"] == "error"
    assert "private filesystem" not in job["message"]
    assert service.store.models == {previous["model_id"]: previous}
    assert not (service.settings.model_dir / ".api-training.lock").exists()


def test_other_training_process_lock_is_preserved(service):
    service.settings.model_dir.mkdir()
    lock = service.settings.model_dir / ".api-training.lock"
    lock.write_text("other-process-job", encoding="utf-8")
    job = finished(service, service.prepare(2544, "2023-24")["job_id"])
    assert job["status"] == "error"
    assert "Another training job" in job["message"]
    assert lock.read_text(encoding="utf-8") == "other-process-job"
    service.client.fetch_shots.assert_not_called()


def test_queue_is_bounded_and_shutdown_cancels_queued_jobs(service, monkeypatch):
    monkeypatch.setattr(preparation_module, "MAX_PENDING", 2)
    entered, release = Event(), Event()

    def blocked_fetch(player_id, season):
        entered.set()
        assert release.wait(10)
        return pd.DataFrame([{"PLAYER_ID": player_id}]), {"source": "nba"}

    service.client.fetch_shots.side_effect = blocked_fetch
    try:
        active = service.prepare(2544, "2023-24")
        assert entered.wait(5)
        queued = service.prepare(203999, "2023-24")
        with pytest.raises(PreparationBusyError, match="queue is full"):
            service.prepare(201939, "2023-24")
        assert service.prepare(2544, "2023-24")["job_id"] == active["job_id"]
        service.close(wait=False)
        assert service.job_status(queued["job_id"])["status"] == "error"
        with pytest.raises(PreparationBusyError, match="shutting down"):
            service.prepare(201939, "2023-24")
    finally:
        release.set()
    assert finished(service, active["job_id"])["status"] == "done"


def test_terminal_history_is_bounded(service, monkeypatch):
    monkeypatch.setattr(preparation_module, "MAX_HISTORY", 2)
    service.trainer.side_effect = TrainingError("Insufficient data")
    jobs = [finished(service, service.prepare(player, "2023-24")["job_id"]) for player in (1, 2, 3)]
    with pytest.raises(LookupError, match="job not found"):
        service.job_status(jobs[0]["job_id"])
    assert service.readiness(1, "2023-24")["status"] == "not_prepared"
    assert service.job_status(jobs[1]["job_id"])["status"] == "unavailable"
    assert len(service._jobs) == 2


def test_real_training_rejects_small_sample_and_can_retry(tmp_path):
    client = Mock()
    client.resolve_player.return_value = {"id": 2544, "name": "LeBron James"}
    client.fetch_player_seasons.return_value = {"player_id": 2544, "seasons": ["2023-24"]}
    client.fetch_shots.return_value = (make_fixture(games=3, player_id=2544), {"source": "nba", "description": "Mocked test data"})
    store = ModelStore(tmp_path / "models")
    service = ModelPreparationService(Settings(model_dir=store.root, cache_dir=tmp_path / "cache"), client, store)
    try:
        insufficient = finished(service, service.prepare(2544, "2023-24")["job_id"])
        assert insufficient["status"] == "unavailable"
        assert "at least" in insufficient["message"]
        assert store.list_models() == []
        client.fetch_shots.return_value = (make_fixture(player_id=2544), {"source": "nba", "description": "Mocked test data"})
        complete = finished(service, service.prepare(2544, "2023-24")["job_id"])
        assert complete["status"] == "done"
        assert store.load(complete["model_id"]).metadata["player_id"] == 2544
        assert service.prepare(2544, "2023-24")["model_id"] == complete["model_id"]
        assert client.fetch_shots.call_count == 2
    finally:
        service.close(wait=True)


@pytest.mark.parametrize("player_id,season", [(0, "2023-24"), (True, "2023-24"), (2544, "2023-25")])
def test_invalid_identity_is_rejected_before_queuing(service, player_id, season):
    with pytest.raises(ValueError):
        service.prepare(player_id, season)
    service.client.fetch_shots.assert_not_called()
