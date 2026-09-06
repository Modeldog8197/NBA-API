"""HTTP boundary; model choice is explicit and never shared between users."""
from datetime import datetime, timezone
import hmac
import logging
import os
from pathlib import Path
import threading
from typing import Any, Literal
import uuid
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import court
from .config import ROOT, Settings, available_seasons, validate_season
from .data import DataUnavailableError, NBADataClient
from .models import MODEL_ID_RE, ModelStore, ModelValidationError
from .training import train_bundle
from .preparation import ModelPreparationService, PreparationBusyError
from .shot_chart import summarize_shots

log = logging.getLogger(__name__)


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ShotInput(StrictSchema):
    loc_x: float = Field(ge=-250, le=250, strict=True, description="Tenths of a foot; basket x=0.")
    loc_y: float = Field(ge=-52.5, le=417.5, strict=True, description="Tenths of a foot toward midcourt.")
    shot_distance: float | None = Field(default=None, ge=0, strict=True, description="Optional feet; validated against geometry.")
    shot_type: str | None = None

    @model_validator(mode="after")
    def consistent_geometry(self):
        court.describe_shot(self.loc_x, self.loc_y, self.shot_distance, self.shot_type)
        return self


class PredictRequest(ShotInput):
    model_id: str = Field(min_length=1, max_length=100)
    explain: bool = True


class BatchRequest(StrictSchema):
    model_id: str = Field(min_length=1, max_length=100)
    shots: list[ShotInput] = Field(min_length=1, max_length=500)
    explain: bool = False


class RetrainRequest(StrictSchema):
    player_id: int = Field(gt=0, strict=True)
    season: str

    @field_validator("season")
    @classmethod
    def valid_season(cls, value):
        return validate_season(value)


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_id: str
    player_id: int
    player_name: str
    season: str
    n_shots: int
    created_at: str
    evaluation: dict[str, Any]


class ModelsResponse(BaseModel):
    models: list[ModelMetadata]


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    feature_value: float


class Explanation(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: str
    units: str
    base_value: float
    raw_margin: float
    contributions: list[Contribution]
    reconstruction_error: float
    calibration: dict[str, Any]
    calibrated_log_odds: float
    note: str


class PredictionResponse(BaseModel):
    model_id: str
    player_id: int
    player_name: str
    season: str
    loc_x: float
    loc_y: float
    distance_ft: float
    shot_value: Literal[2, 3]
    shot_type: Literal["2pt", "3pt"]
    shot_zone: str
    make_probability: float = Field(ge=0, le=1)
    expected_points: float = Field(ge=0, le=3)
    uncertainty: None = None
    uncertainty_note: str
    explanation: Explanation | None = None


class BatchResponse(BaseModel):
    model_id: str
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    training_enabled: bool
    models_available: int


class PlayerResponse(BaseModel):
    id: int
    name: str


class SeasonsResponse(BaseModel):
    seasons: list[str]
    default_season: str


class FreeThrowResponse(BaseModel):
    player_id: int
    player_name: str
    season: str
    ft_pct: float | None = Field(ge=0, le=1)
    ftm: int
    fta: int
    source: str
    provenance: dict[str, Any]


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "fetching", "training", "evaluating", "publishing", "done", "error"]
    message: str
    player_id: int
    player_name: str
    season: str
    model_id: str | None = None
    error: str | None = None
    created_at: str


class PreparationResponse(BaseModel):
    player_id: int
    player_name: str
    season: str
    status: Literal["ready", "not_prepared", "preparing", "unavailable", "error", "queued", "fetching", "training", "evaluating", "publishing", "done"]
    message: str
    model_id: str | None = None
    job_id: str | None = None
    error: str | None = None
    created_at: str | None = None


class SessionResponse(BaseModel):
    preparation_enabled: bool
    requires_key: bool
    session_token: str | None = None


class PlayerSeasonsResponse(BaseModel):
    player_id: int
    player_name: str
    seasons: list[str]
    provenance: dict[str, Any]


def create_app(settings: Settings | None = None, data_client=None, store=None) -> FastAPI:
    config = settings or Settings.from_env()
    client = data_client or NBADataClient(config)
    registry = store or ModelStore(config.model_dir)
    preparation = ModelPreparationService(config, client, registry)
    local_session_token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(application):
        yield
        preparation.close(wait=False)

    app = FastAPI(title="NBA API", version="4.2.0", lifespan=lifespan, description=(
        "Basketball shot probabilities with immutable player-season model versions. "
        "Coordinates use tenths of a foot, with the hoop at the origin."
    ))
    app.state.settings, app.state.store, app.state.data_client = config, registry, client
    app.state.preparation = preparation
    app.state.jobs = {}
    jobs_lock = threading.Lock()
    app.add_middleware(CORSMiddleware, allow_origins=list(config.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path == "/":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
            )
        response.headers["Cache-Control"] = "no-cache" if request.url.path.startswith("/static/") else "no-store"
        return response

    @app.exception_handler(DataUnavailableError)
    async def data_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo invalid NaN/Infinity or arbitrary input/exception objects into JSON.
        detail = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def load_bundle(model_id):
        if not MODEL_ID_RE.fullmatch(model_id):
            raise HTTPException(422, "Malformed model_id. Use an immutable ID returned by /models.")
        try:
            return registry.load(model_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Model version not found. Select or train an available model.") from exc
        except ModelValidationError as exc:
            log.warning("Unavailable model %s: %s", model_id, exc)
            raise HTTPException(503, "Model bundle failed validation. Train a new version.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return {"status": "ok", "training_enabled": config.training_enabled,
                "models_available": len(registry.list_models())}

    @app.get("/court")
    def geometry() -> dict[str, Any]:
        return {"coordinate_units": "tenths of a foot", "distance_units": "feet",
                "x_min": court.X_MIN, "x_max": court.X_MAX, "y_min": court.Y_MIN, "y_max": court.Y_MAX,
                "corner_x": court.CORNER_X, "arc_radius": court.ARC_RADIUS, "arc_join_y": court.ARC_JOIN_Y,
                "court_version": court.COURT_VERSION, "boundary_shot_value": 2}

    @app.get("/seasons", response_model=SeasonsResponse)
    def seasons():
        values = available_seasons()
        return {"seasons": values, "default_season": values[0]}

    @app.get("/players/search", response_model=list[PlayerResponse])
    def search_players(q: str = Query(default="", max_length=80)):
        return client.search_players(q)

    @app.get("/players/{player_id}/seasons", response_model=PlayerSeasonsResponse)
    def player_seasons(player_id: int):
        try:
            return client.fetch_player_seasons(player_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def is_local_dashboard(request: Request) -> bool:
        """Loopback + exact local Host + same origin: no remote/proxy/DNS-rebinding shortcut."""
        if not config.local_player_models or not request.client:
            return False
        if request.client.host not in {"127.0.0.1", "::1"}:
            return False
        host = request.headers.get("host", "").lower()
        try:
            hostname = urlsplit(f"http://{host}").hostname
        except ValueError:
            return False
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            return False
        if request.headers.get("sec-fetch-site", "same-origin") not in {"same-origin", "none"}:
            return False
        origin = request.headers.get("origin")
        return origin is None or origin in {f"http://{host}", f"https://{host}"}

    @app.get("/session", response_model=SessionResponse)
    def session(request: Request):
        local = is_local_dashboard(request)
        return {"preparation_enabled": local or config.training_enabled,
                "requires_key": not local, "session_token": local_session_token if local else None}

    @app.get("/models/readiness", response_model=PreparationResponse)
    def model_readiness(player_id: int = Query(gt=0), season: str = Query()):
        try:
            validate_season(season)
            return preparation.readiness(player_id, season)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/models/prepare", response_model=PreparationResponse, status_code=202)
    def prepare_model(req: RetrainRequest, request: Request,
                      authorization: str | None = Header(default=None),
                      x_local_session: str | None = Header(default=None)):
        local = is_local_dashboard(request) and hmac.compare_digest(x_local_session or "", local_session_token)
        token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        hosted = config.training_enabled and hmac.compare_digest(token, config.training_api_key or "")
        if not local and not hosted:
            raise HTTPException(401, "Open the local dashboard or provide the server's training access key.")
        try:
            return preparation.prepare(req.player_id, req.season)
        except PreparationBusyError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/models/preparation/{job_id}", response_model=PreparationResponse)
    def preparation_status(job_id: str):
        try:
            return preparation.job_status(job_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/models", response_model=ModelsResponse)
    def models():
        return {"models": registry.list_models()}

    @app.get("/models/{model_id}", response_model=ModelMetadata)
    def model_detail(model_id: str):
        return load_bundle(model_id).metadata

    @app.get("/player/freethrow", response_model=FreeThrowResponse)
    def free_throws(player_id: int = Query(gt=0), season: str = Query()):
        try:
            validate_season(season)
            return client.fetch_freethrow(player_id, season)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/player/shot-chart")
    def shot_chart(player_id: int = Query(gt=0), season: str = Query()):
        try:
            validate_season(season)
            player = client.resolve_player(player_id)
            frame, provenance = client.fetch_shots(player_id, season)
            return {"player_id": player_id, "player_name": player["name"], "season": season,
                    "provenance": provenance, **summarize_shots(frame, player_id)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def predictions(model_id, shots, explain):
        bundle = load_bundle(model_id)
        try:
            items = bundle.predict(shots, explain=explain)
        except ModelValidationError as exc:
            log.exception("Prediction integrity failure for %s", model_id)
            raise HTTPException(503, "Prediction failed an integrity check. Train a new model version.") from exc
        identity = {key: bundle.metadata[key] for key in ("model_id", "player_id", "player_name", "season")}
        return [{**item, **identity} for item in items]

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictRequest):
        shot = req.model_dump(exclude={"model_id", "explain"}, exclude_none=True)
        return predictions(req.model_id, [shot], req.explain)[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def batch(req: BatchRequest):
        return {"model_id": req.model_id, "predictions": predictions(
            req.model_id, [shot.model_dump(exclude_none=True) for shot in req.shots], req.explain)}

    def update_job(job_id, status, message, **extra):
        with jobs_lock:
            app.state.jobs[job_id].update(status=status, message=message, **extra)

    def run_training(job_id, player, season, lock_path):
        try:
            update_job(job_id, "fetching", f"Fetching {player['name']} · {season} regular season shots…")
            df, source_info = client.fetch_shots(player["id"], season)
            update_job(job_id, "training", f"Preparing {len(df):,} shots and chronological game splits…")
            metadata = train_bundle(df, player["id"], player["name"], season, registry,
                                    source_info=source_info,
                                    progress=lambda message: update_job(job_id, (
                                        "publishing" if "publish" in message.lower() else
                                        "evaluating" if "evaluat" in message.lower() or "calibrat" in message.lower()
                                        else "training"), message))
            update_job(job_id, "done", f"Ready: {metadata['n_shots']:,} historical shots.",
                       model_id=metadata["model_id"])
        except (ValueError, DataUnavailableError) as exc:
            log.warning("Training job %s failed: %s", job_id, exc)
            update_job(job_id, "error", str(exc), error=str(exc))
        except Exception:
            log.exception("Training job %s failed", job_id)
            message = "Training failed. Previous model versions remain available. Check the server logs."
            update_job(job_id, "error", message, error=message)
        finally:
            Path(lock_path).unlink(missing_ok=True)

    @app.post("/model/retrain", response_model=JobResponse, status_code=202)
    def retrain(req: RetrainRequest, background_tasks: BackgroundTasks,
                authorization: str | None = Header(default=None)):
        if not config.training_enabled:
            raise HTTPException(403, "API training is disabled. Use the training CLI or enable protected training.")
        token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        if not hmac.compare_digest(token, config.training_api_key or ""):
            raise HTTPException(401, "A valid training API key is required.", headers={"WWW-Authenticate": "Bearer"})
        try:
            player = client.resolve_player(req.player_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        config.model_dir.mkdir(parents=True, exist_ok=True)
        lock_path = config.model_dir / ".api-training.lock"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise HTTPException(409, "Training is already running. Wait for the current job to finish.") from exc
        job_id = uuid.uuid4().hex
        with os.fdopen(fd, "w") as handle:
            handle.write(job_id)
        job = {"job_id": job_id, "status": "queued", "message": "Training queued.",
               "player_id": player["id"], "player_name": player["name"], "season": req.season,
               "model_id": None, "error": None, "created_at": datetime.now(timezone.utc).isoformat()}
        with jobs_lock:
            if len(app.state.jobs) >= 100:
                oldest = next(iter(app.state.jobs))
                del app.state.jobs[oldest]
            app.state.jobs[job_id] = job
        background_tasks.add_task(run_training, job_id, player, req.season, lock_path)
        return dict(job)

    @app.get("/model/retrain/status", response_model=JobResponse)
    def retrain_status(job_id: str = Query(min_length=1, max_length=64)):
        with jobs_lock:
            job = app.state.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Training job not found; job history resets when the server restarts.")
            return dict(job)

    app.mount("/static", StaticFiles(directory=ROOT / "static", check_dir=False), name="static")
    return app
