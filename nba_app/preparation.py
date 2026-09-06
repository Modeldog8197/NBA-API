"""Bounded, deduplicated preparation of explicit NBA player-season models."""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

from .config import Settings, validate_season
from .data import DataUnavailableError, NBADataClient
from .models import ModelStore, ModelValidationError
from .training import TrainingError, train_bundle

log = logging.getLogger(__name__)
ACTIVE_STATES = frozenset({"queued", "fetching", "training", "evaluating", "publishing"})
MAX_PENDING = 8
MAX_HISTORY = 100


class PreparationBusyError(ValueError):
    """The bounded worker queue cannot accept another preparation request."""


class ModelPreparationService:
    """Prepare one model at a time without changing any caller's selected model.

    Requests for the same player and season share a job. Terminal failures are
    retained for status reads and retried only by another explicit prepare call.
    Saved bundles are validated before reuse; no other player or season is used.
    The API owns authorization. This service also works from a local batch tool.
    """

    def __init__(self, settings: Settings, client: NBADataClient, store: ModelStore):
        self.settings, self.client, self.store = settings, client, store
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._latest: dict[tuple[int, str], str] = {}
        self._futures = {}
        self._lock = threading.RLock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nba-preparation")

    def _identity(self, player_id: int, season: str) -> dict:
        validate_season(season)
        if isinstance(player_id, bool) or not isinstance(player_id, int) or player_id <= 0:
            raise ValueError("Select a valid stable NBA player ID.")
        player = self.client.resolve_player(player_id)
        if player["id"] != player_id:
            raise ValueError("Resolved player ID does not match the requested player.")
        return {"player_id": player_id, "player_name": player["name"], "season": season}

    def _saved_model(self, player_id: int, season: str) -> dict | None:
        for listed in self.store.list_models():
            if listed.get("player_id") != player_id or listed.get("season") != season or listed.get("is_demo"):
                continue
            try:
                metadata = self.store.load(listed["model_id"]).metadata
            except (FileNotFoundError, ModelValidationError) as exc:
                log.warning("Skipping unavailable saved model %s: %s", listed.get("model_id"), exc)
                continue
            if (metadata.get("player_id") == player_id and metadata.get("season") == season
                    and not metadata.get("is_demo")):
                return metadata
        return None

    def readiness(self, player_id: int, season: str) -> dict:
        identity = self._identity(player_id, season)
        saved = self._saved_model(player_id, season)
        if saved:
            return {**identity, "status": "ready", "model_id": saved["model_id"], "job_id": None,
                    "message": "A validated model is ready for this player and season."}
        with self._lock:
            job = self._jobs.get(self._latest.get((player_id, season)))
            if job is not None and job["status"] != "done":
                return {**identity, "status": "preparing" if job["status"] in ACTIVE_STATES else job["status"],
                        "job_id": job["job_id"], "model_id": None, "message": job["message"]}
        return {**identity, "status": "not_prepared", "model_id": None, "job_id": None,
                "message": "Prepare a model from this player's regular-season NBA shots."}

    def prepare(self, player_id: int, season: str) -> dict:
        identity = self._identity(player_id, season)
        key = (player_id, season)
        with self._lock:
            saved = self._saved_model(player_id, season)
            if saved:
                return {**identity, "job_id": None, "status": "done", "model_id": saved["model_id"],
                        "message": "A validated model is ready for this player and season.",
                        "error": None, "created_at": saved["created_at"]}
            previous = self._jobs.get(self._latest.get(key))
            if previous and previous["status"] in ACTIVE_STATES:
                return dict(previous)
            if self._closed:
                raise PreparationBusyError("Model preparation is shutting down. Try again after the server restarts.")
            if sum(job["status"] in ACTIVE_STATES for job in self._jobs.values()) >= MAX_PENDING:
                raise PreparationBusyError("The model preparation queue is full. Wait for a current job to finish, then retry.")
            self._trim_history()
            job_id = uuid4().hex
            job = {**identity, "job_id": job_id, "status": "queued", "model_id": None,
                   "message": "Model preparation queued.", "error": None,
                   "created_at": datetime.now(UTC).isoformat()}
            self._jobs[job_id], self._latest[key] = job, job_id
            try:
                future = self._executor.submit(self._run, job_id)
                self._futures[job_id] = future
                future.add_done_callback(lambda completed, identifier=job_id: self._finished(identifier))
            except RuntimeError as exc:
                self._update(job_id, "error", "Model preparation could not start. Retry after the server restarts.",
                             error="The preparation worker is unavailable.")
                raise PreparationBusyError(job["message"]) from exc
            return dict(job)

    def _trim_history(self):
        while len(self._jobs) >= MAX_HISTORY:
            terminal = next((key for key, value in self._jobs.items() if value["status"] not in ACTIVE_STATES), None)
            if terminal is None:
                return
            removed = self._jobs.pop(terminal)
            identity = (removed["player_id"], removed["season"])
            if self._latest.get(identity) == terminal:
                self._latest.pop(identity)

    def _finished(self, job_id):
        with self._lock:
            self._futures.pop(job_id, None)

    def job_status(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise LookupError("Preparation job not found; job history resets when the server restarts.")
            return dict(self._jobs[job_id])

    def _update(self, job_id, status, message, **extra):
        with self._lock:
            self._jobs[job_id].update(status=status, message=message, **extra)

    def _progress(self, job_id, message):
        lower = message.lower()
        status = ("publishing" if "publish" in lower else "evaluating"
                  if any(word in lower for word in ("evaluat", "calibrat", "selecting")) else "training")
        self._update(job_id, status, message)

    def _run(self, job_id):
        job = self.job_status(job_id)
        player_id, season = job["player_id"], job["season"]
        lock_path = self.settings.model_dir / ".api-training.lock"
        owns_lock = False
        completion = None
        try:
            self.settings.model_dir.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                completion = {"status": "error", "message": "Another training job is running. Wait for it to finish, then retry.",
                              "error": "Training is already running in another request or server process."}
                return
            owns_lock = True
            with os.fdopen(fd, "w") as handle:
                handle.write(job_id)
            # Another process may have prepared this version while it was queued.
            saved = self._saved_model(player_id, season)
            if saved:
                completion = {"status": "done", "message": "A validated model is ready for this player and season.",
                              "model_id": saved["model_id"]}
                return
            self._update(job_id, "fetching", f"Checking {job['player_name']}'s NBA season history.")
            career = self.client.fetch_player_seasons(player_id)
            if career.get("player_id") != player_id:
                raise DataUnavailableError("NBA career data does not match the selected player.")
            if season not in career["seasons"]:
                raise DataUnavailableError(
                    f"No supported regular-season field-goal data for {job['player_name']} in {season}. "
                    "Choose a season listed for this player.")
            self._update(job_id, "fetching", f"Loading {job['player_name']}'s {season} regular-season shots.")
            shots, provenance = self.client.fetch_shots(player_id, season)
            self._update(job_id, "training", f"Validating {len(shots):,} shots and chronological game splits.")
            metadata = train_bundle(shots, player_id, job["player_name"], season, self.store,
                                    source_info=provenance, progress=lambda message: self._progress(job_id, message))
            validated = self.store.load(metadata["model_id"]).metadata
            if (validated.get("player_id") != player_id or validated.get("season") != season
                    or validated.get("is_demo")):
                raise ModelValidationError("Prepared model does not match the selected NBA player and season.")
            completion = {"status": "done", "message": f"Ready: {validated['n_shots']:,} historical shots.",
                          "model_id": validated["model_id"]}
        except (DataUnavailableError, TrainingError) as exc:
            log.info("Preparation job %s has insufficient or unavailable NBA data: %s", job_id, exc)
            completion = {"status": "unavailable", "message": str(exc), "error": str(exc)}
        except Exception:
            log.exception("Preparation job %s failed", job_id)
            message = "Model preparation failed. Existing model versions remain available. Retry or check the server logs."
            completion = {"status": "error", "message": message, "error": message}
        finally:
            if owns_lock:
                try:
                    if lock_path.read_text(encoding="utf-8") == job_id:
                        lock_path.unlink(missing_ok=True)
                except OSError:
                    log.exception("Could not release the training lock for job %s", job_id)
            if completion is not None:
                # A terminal status guarantees that another training request can
                # acquire the lock; clients may retry immediately after failures.
                self._update(job_id, **completion)

    def close(self, wait: bool = False):
        """Stop accepting work; cancel queued jobs and let a running job finish."""
        with self._lock:
            self._closed = True
            for job_id, future in list(self._futures.items()):
                if future.cancel():
                    message = "Server stopped before model preparation began. Retry after the server restarts."
                    self._update(job_id, "error", message, error=message)
        self._executor.shutdown(wait=wait, cancel_futures=True)
