"""NBA sources with bounded requests, JSON cache, and visible provenance."""
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import time
import uuid

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import PlayerCareerStats, ShotChartDetail
from nba_api.stats.static import players

from .config import Settings, validate_season

log = logging.getLogger(__name__)


class DataUnavailableError(RuntimeError):
    pass


class NBADataClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve_player(self, identifier: int | str) -> dict:
        roster = players.get_players()
        if isinstance(identifier, int) or str(identifier).isdigit():
            matches = [p for p in roster if p["id"] == int(identifier)]
        else:
            matches = [p for p in roster if p["full_name"].casefold() == str(identifier).strip().casefold()]
        if len(matches) != 1:
            raise ValueError("Player not found or ambiguous. Select a player using their stable NBA ID.")
        return {"id": matches[0]["id"], "name": matches[0]["full_name"]}

    def search_players(self, q: str) -> list[dict]:
        if len(q.strip()) < 2:
            return []
        matches = [p for p in players.get_players() if q.strip().casefold() in p["full_name"].casefold()]
        matches.sort(key=lambda p: (not p["is_active"], p["full_name"]))
        return [{"id": p["id"], "name": p["full_name"]} for p in matches[:20]]

    def _cached_request(self, key: str, loader) -> tuple[pd.DataFrame, dict]:
        path = self.settings.cache_dir / f"{key}.json"
        cached = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["key"] != key or not isinstance(payload["records"], list) or not payload["records"]:
                raise ValueError("Invalid cache envelope")
            fetched = datetime.fromisoformat(payload["fetched_at"])
            if fetched.tzinfo is None:
                raise ValueError("Cache timestamp must have a timezone")
            cached = payload
            age = (datetime.now(timezone.utc) - fetched).total_seconds()
            if 0 <= age <= self.settings.cache_ttl_seconds:
                return pd.DataFrame(cached["records"]), self._provenance(cached, "cache", False)
        except (OSError, ValueError, TypeError, KeyError):
            pass
        error = None
        for attempt in range(self.settings.retries + 1):
            try:
                df = loader()
                if not isinstance(df, pd.DataFrame) or df.empty:
                    raise DataUnavailableError("NBA returned no records for this player and season.")
                payload = {"key": key, "fetched_at": datetime.now(timezone.utc).isoformat(),
                           "records": json.loads(df.to_json(orient="records", date_format="iso"))}
                self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                return df, self._provenance(payload, "live_nba", False)
            except Exception as exc:
                error = exc
                log.warning("NBA request %s attempt %d failed (%s)", key, attempt + 1, type(exc).__name__)
                if isinstance(exc, DataUnavailableError):
                    break
                if attempt < self.settings.retries:
                    time.sleep(min(2 ** attempt, 4))
        if cached and self.settings.allow_stale_cache:
            return pd.DataFrame(cached["records"]), self._provenance(cached, "stale_cache", True)
        raise DataUnavailableError(
            "NBA data could not be loaded after bounded attempts. NBA may block or time out requests. "
            "Try again later or train from a local CSV; no usable cached records are available."
        ) from error

    @staticmethod
    def _provenance(payload: dict, mode: str, stale: bool) -> dict:
        encoded = json.dumps(payload["records"], sort_keys=True, separators=(",", ":")).encode()
        return {"source": "nba", "retrieval": mode, "stale": stale, "fetched_at": payload["fetched_at"],
                "sha256": hashlib.sha256(encoded).hexdigest(), "endpoint_key": payload["key"],
                "season_type": "Regular Season"}

    def fetch_shots(self, player_id: int, season: str) -> tuple[pd.DataFrame, dict]:
        validate_season(season)
        self.resolve_player(player_id)
        return self._cached_request(f"shots-{int(player_id)}-{season}-regular-v1", lambda: ShotChartDetail(
            team_id=0, player_id=player_id, season_nullable=season,
            season_type_all_star="Regular Season", context_measure_simple="FGA",
            timeout=self.settings.request_timeout,
        ).get_data_frames()[0])

    def fetch_freethrow(self, player_id: int, season: str) -> dict:
        validate_season(season)
        player = self.resolve_player(player_id)
        df, provenance = self._cached_request(f"career-{int(player_id)}-regular-v1", lambda: PlayerCareerStats(
            player_id=player_id, timeout=self.settings.request_timeout,
        ).get_data_frames()[0])
        if not {"SEASON_ID", "TEAM_ID", "FTM", "FTA"}.issubset(df.columns):
            raise DataUnavailableError("NBA free-throw response is missing required fields.")
        rows = df.loc[df["SEASON_ID"].astype(str) == season].copy()
        if rows.empty:
            raise LookupError("No historical free-throw statistics for the selected season.")
        # Traded players have total (TEAM_ID=0) and individual team rows: never double count.
        totals = rows.loc[pd.to_numeric(rows["TEAM_ID"], errors="coerce") == 0]
        rows = totals.iloc[:1] if not totals.empty else rows.drop_duplicates(subset=["TEAM_ID"])
        numeric = rows[["FTM", "FTA"]].apply(pd.to_numeric, errors="coerce")
        if (not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any()
                or (numeric % 1 != 0).any().any()):
            raise DataUnavailableError("NBA returned invalid free-throw counts.")
        ftm, fta = int(numeric["FTM"].sum()), int(numeric["FTA"].sum())
        if ftm > fta:
            raise DataUnavailableError("NBA returned inconsistent free-throw counts.")
        return {"player_id": player_id, "player_name": player["name"], "season": season,
                "ft_pct": ftm / fta if fta else None, "ftm": ftm, "fta": fta,
                "source": "NBA historical statistics", "provenance": provenance}
