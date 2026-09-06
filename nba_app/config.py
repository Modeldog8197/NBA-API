"""Validated configuration; importing this module has no side effects."""
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def validate_season(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("Season must use YYYY-YY, for example 2023-24.")
    start = int(value[:4])
    if not 1997 <= start <= date.today().year or int(value[-2:]) != (start + 1) % 100:
        raise ValueError("Use a consecutive season from 1997-98 through the current year.")
    return value


def available_seasons() -> list[str]:
    today = date.today()
    newest = today.year if today.month >= 10 else today.year - 1
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(newest, 1996, -1)]


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true or false.")
    return value in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    model_dir: Path = ROOT / "models"
    cache_dir: Path = ROOT / "data" / "cache"
    training_enabled: bool = False
    training_api_key: str | None = None
    request_timeout: float = 10
    retries: int = 2
    cache_ttl_seconds: int = 86400
    cors_origins: tuple[str, ...] = ()
    allow_stale_cache: bool = True
    local_player_models: bool = False

    def __post_init__(self):
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.training_enabled and (not self.training_api_key or len(self.training_api_key) < 16):
            raise ValueError("API training requires NBA_TRAINING_API_KEY with at least 16 characters.")
        if not 1 <= self.request_timeout <= 60:
            raise ValueError("NBA_REQUEST_TIMEOUT must be between 1 and 60 seconds.")
        if not 0 <= self.retries <= 3:
            raise ValueError("NBA_RETRIES must be between 0 and 3.")
        if not 0 <= self.cache_ttl_seconds <= 31536000:
            raise ValueError("Cache TTL must be between 0 and 31536000 seconds.")
        if any(not re.fullmatch(r"https?://[^/\s]+", origin) for origin in self.cors_origins):
            raise ValueError("CORS origins must be explicit http(s) origins without paths or wildcards.")

    @classmethod
    def from_env(cls, *, local_default: bool = False):
        return cls(
            model_dir=Path(os.getenv("NBA_MODEL_DIR", str(ROOT / "models"))),
            cache_dir=Path(os.getenv("NBA_CACHE_DIR", str(ROOT / "data/cache"))),
            training_enabled=_boolean("NBA_TRAINING_ENABLED", False),
            training_api_key=os.getenv("NBA_TRAINING_API_KEY") or None,
            request_timeout=float(os.getenv("NBA_REQUEST_TIMEOUT", "10")),
            retries=int(os.getenv("NBA_RETRIES", "2")),
            cache_ttl_seconds=int(os.getenv("NBA_CACHE_TTL_SECONDS", "86400")),
            cors_origins=tuple(v.strip() for v in os.getenv("NBA_CORS_ORIGINS", "").split(",") if v.strip()),
            allow_stale_cache=_boolean("NBA_ALLOW_STALE_CACHE", True),
            local_player_models=_boolean("NBA_LOCAL_PLAYER_MODELS", local_default),
        )
