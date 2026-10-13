"""Player data service: durable SQLite cache, historical game logs, and derived metrics.

This service is deliberately separate from the shot-model pipeline. NBA endpoints can
be unavailable from cloud hosts, so every route reports freshness and never fabricates data.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from nba_api.stats.endpoints import LeagueDashPlayerStats, PlayerGameLogs
from nba_api.stats.static import players as nba_players

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/data", tags=["player-data"])
DB_PATH = Path(os.getenv("NBA_PLAYER_DB", "data/nba_players.sqlite3"))
SEASON_TYPES = ("Regular Season", "Playoffs")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS player_game_logs (
        player_id INTEGER NOT NULL, season TEXT NOT NULL, season_type TEXT NOT NULL,
        game_id TEXT NOT NULL, game_date TEXT, team_id INTEGER, team_abbreviation TEXT,
        matchup TEXT, wl TEXT, min REAL, pts REAL, fgm REAL, fga REAL, fg_pct REAL,
        fg3m REAL, fg3a REAL, fg3_pct REAL, ftm REAL, fta REAL, ft_pct REAL,
        oreb REAL, dreb REAL, reb REAL, ast REAL, stl REAL, blk REAL, tov REAL, pf REAL,
        plus_minus REAL, updated_at TEXT NOT NULL,
        PRIMARY KEY(player_id, game_id)
      );
      CREATE INDEX IF NOT EXISTS idx_game_logs_player_season
        ON player_game_logs(player_id, season, game_date);
      CREATE TABLE IF NOT EXISTS season_stats (
        player_id INTEGER NOT NULL, season TEXT NOT NULL, season_type TEXT NOT NULL,
        team_id INTEGER, team_abbreviation TEXT, gp REAL, gs REAL, min REAL,
        pts REAL, fgm REAL, fga REAL, fg_pct REAL, fg3m REAL, fg3a REAL, fg3_pct REAL,
        ftm REAL, fta REAL, ft_pct REAL, oreb REAL, dreb REAL, reb REAL, ast REAL,
        stl REAL, blk REAL, tov REAL, pf REAL, plus_minus REAL, updated_at TEXT NOT NULL,
        PRIMARY KEY(player_id, season, season_type, team_id)
      );
      CREATE TABLE IF NOT EXISTS ingestion_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, dataset TEXT NOT NULL,
        season TEXT, started_at TEXT NOT NULL, finished_at TEXT,
        rows_written INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, error TEXT
      );
    """)
    return db


def seasons_to_fetch(count: int = 10, include_current: bool = True) -> list[str]:
    now = datetime.now()
    start = now.year if now.month >= 10 else now.year - 1
    years = list(range(start, start - count, -1))
    if not include_current and now.month < 10:
        years = years[1:]
    return [f"{year}-{(year + 1) % 100:02d}" for year in years]


def as_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() in ("", "nan", "None"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() in ("", "nan", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sync_player_registry() -> int:
    stamp = utc_now()
    roster = nba_players.get_players()
    with connect() as db:
        db.executemany(
            """INSERT INTO players(player_id,full_name,is_active,updated_at) VALUES(?,?,?,?)
               ON CONFLICT(player_id) DO UPDATE SET full_name=excluded.full_name,
               is_active=excluded.is_active, updated_at=excluded.updated_at""",
            [(p["id"], p["full_name"], int(p["is_active"]), stamp) for p in roster],
        )
    return len(roster)


def _df(endpoint):
    # Endpoint calls are isolated so a source failure doesn't erase already stored rows.
    return endpoint.get_data_frames()[0]


def ingest_season(season: str, season_type: str = "Regular Season") -> dict[str, Any]:
    started = utc_now()
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO ingestion_runs(dataset,season,started_at,status) VALUES(?,?,?,'running')",
            ("player_game_logs", season),
        )
        run_id = cursor.lastrowid
    try:
        frame = _df(PlayerGameLogs(season_nullable=season, season_type_nullable=season_type, timeout=45))
        if frame.empty:
            raise RuntimeError(f"NBA returned no game logs for {season} ({season_type})")
        stamp = utc_now()
        columns = {
            "player_id": "PLAYER_ID", "game_id": "GAME_ID", "game_date": "GAME_DATE",
            "team_id": "TEAM_ID", "team_abbreviation": "TEAM_ABBREVIATION", "matchup": "MATCHUP",
            "wl": "WL", "min": "MIN", "pts": "PTS", "fgm": "FGM", "fga": "FGA",
            "fg_pct": "FG_PCT", "fg3m": "FG3M", "fg3a": "FG3A", "fg3_pct": "FG3_PCT",
            "ftm": "FTM", "fta": "FTA", "ft_pct": "FT_PCT", "oreb": "OREB", "dreb": "DREB",
            "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TOV",
            "pf": "PF", "plus_minus": "PLUS_MINUS"
        }
        rows = []
        for record in frame.to_dict(orient="records"):
            row = {key: record.get(source) for key, source in columns.items()}
            row["season"], row["season_type"], row["updated_at"] = season, season_type, stamp
            row["player_id"] = as_int(row["player_id"])
            row["team_id"] = as_int(row["team_id"])
            for key in ("min", "pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
                        "ftm", "fta", "ft_pct", "oreb", "dreb", "reb", "ast", "stl", "blk",
                        "tov", "pf", "plus_minus"):
                row[key] = as_float(row[key])
            rows.append(row)
        names = list(columns.keys()) + ["season", "season_type", "updated_at"]
        sql = f"INSERT INTO player_game_logs({','.join(names)}) VALUES({','.join('?' for _ in names)}) ON CONFLICT(player_id,game_id) DO UPDATE SET " + ",".join(f"{k}=excluded.{k}" for k in names if k not in ("player_id", "game_id"))
        with connect() as db:
            db.executemany(sql, [tuple(r.get(k) for k in names) for r in rows])
            db.execute("UPDATE ingestion_runs SET finished_at=?,rows_written=?,status='ok' WHERE id=?",
                       (utc_now(), len(rows), run_id))
        return {"season": season, "season_type": season_type, "rows_written": len(rows), "updated_at": stamp}
    except Exception as exc:
        with connect() as db:
            db.execute("UPDATE ingestion_runs SET finished_at=?,status='error',error=? WHERE id=?",
                       (utc_now(), str(exc)[:500], run_id))
        log.exception("Game log ingestion failed for %s", season)
        raise


def ingest_season_totals(season: str, season_type: str = "Regular Season") -> dict[str, Any]:
    frame = _df(LeagueDashPlayerStats(
        season=season, season_type_all_star=season_type, per_mode_detailed="PerGame", timeout=45
    ))
    if frame.empty:
        raise RuntimeError(f"NBA returned no season stats for {season} ({season_type})")
    stamp = utc_now()
    fields = {
        "player_id": "PLAYER_ID", "team_id": "TEAM_ID", "team_abbreviation": "TEAM_ABBREVIATION",
        "gp": "GP", "gs": "GS", "min": "MIN", "pts": "PTS", "fgm": "FGM", "fga": "FGA",
        "fg_pct": "FG_PCT", "fg3m": "FG3M", "fg3a": "FG3A", "fg3_pct": "FG3_PCT",
        "ftm": "FTM", "fta": "FTA", "ft_pct": "FT_PCT", "oreb": "OREB", "dreb": "DREB",
        "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TOV",
        "pf": "PF", "plus_minus": "PLUS_MINUS"
    }
    rows = []
    for record in frame.to_dict(orient="records"):
        row = {k: record.get(v) for k, v in fields.items()}
        row.update(season=season, season_type=season_type, updated_at=stamp)
        row["player_id"], row["team_id"] = as_int(row["player_id"]), as_int(row["team_id"])
        for k in fields:
            if k not in ("player_id", "team_id", "team_abbreviation"):
                row[k] = as_float(row[k])
        rows.append(row)
    names = list(fields.keys()) + ["season", "season_type", "updated_at"]
    with connect() as db:
        db.executemany(
            f"INSERT INTO season_stats({','.join(names)}) VALUES({','.join('?' for _ in names)}) "
            "ON CONFLICT(player_id,season,season_type,team_id) DO UPDATE SET " +
            ",".join(f"{k}=excluded.{k}" for k in names if k not in ("player_id", "season", "season_type", "team_id")),
            [tuple(r.get(k) for k in names) for r in rows],
        )
    return {"season": season, "season_type": season_type, "rows_written": len(rows), "updated_at": stamp}


@router.get("/players")
def list_players(q: str = Query(default="", max_length=80), limit: int = Query(default=50, ge=1, le=200)):
    with connect() as db:
        count = db.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        if not count:
            sync_player_registry()
        if q.strip():
            rows = db.execute(
                "SELECT player_id,full_name,is_active,updated_at FROM players WHERE full_name LIKE ? COLLATE NOCASE ORDER BY is_active DESC,full_name LIMIT ?",
                (f"%{q.strip()}%", limit),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT player_id,full_name,is_active,updated_at FROM players ORDER BY is_active DESC,full_name LIMIT ?",
                (limit,),
            ).fetchall()
    return {"players": [dict(r) for r in rows], "count_returned": len(rows)}


@router.get("/seasons")
def list_seasons():
    with connect() as db:
        rows = db.execute("SELECT season,MAX(updated_at) AS updated_at,COUNT(*) AS games FROM player_game_logs GROUP BY season ORDER BY season DESC").fetchall()
    return {"available_seasons": seasons_to_fetch(10), "ingested_seasons": [dict(r) for r in rows]}


@router.get("/players/{player_id}/games")
def player_games(
    player_id: int, season: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    season_type: str = Query(default="Regular Season", pattern="^(Regular Season|Playoffs)$"),
    limit: int = Query(default=500, ge=1, le=2000),
):
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM player_game_logs WHERE player_id=? AND season=? AND season_type=? ORDER BY game_date DESC LIMIT ?",
            (player_id, season, season_type, limit),
        ).fetchall()
    return {"player_id": player_id, "season": season, "season_type": season_type,
            "games": [dict(r) for r in rows], "count": len(rows),
            "updated_at": max((r["updated_at"] for r in rows), default=None)}


@router.get("/players/{player_id}/stats")
def player_stats(player_id: int, season: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$")):
    with connect() as db:
        if season:
            rows = db.execute("SELECT * FROM season_stats WHERE player_id=? AND season=? ORDER BY season_type,team_id", (player_id, season)).fetchall()
        else:
            rows = db.execute("SELECT * FROM season_stats WHERE player_id=? ORDER BY season DESC,season_type,team_id", (player_id,)).fetchall()
    return {"player_id": player_id, "stats": [dict(r) for r in rows], "count": len(rows)}


@router.get("/players/{player_id}/advanced")
def player_advanced(player_id: int, season: str = Query(..., pattern=r"^\d{4}-\d{2}$")):
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM season_stats WHERE player_id=? AND season=? AND season_type='Regular Season'",
            (player_id, season),
        ).fetchall()
        games = db.execute(
            "SELECT pts,fga,fg3a,fta,fgm,fg3m,ftm,ast,tov,reb,min FROM player_game_logs WHERE player_id=? AND season=? AND season_type='Regular Season'",
            (player_id, season),
        ).fetchall()
    if not rows:
        raise HTTPException(404, "No ingested season totals for this player and season.")
    # LeagueDashPlayerStats is per-game. Use totals reconstructed from per-game values
    # multiplied by GP, clearly labeling approximate derived metrics.
    row = dict(rows[0])
    gp = row.get("gp") or 0
    fga, fg3m, fta = (row.get("fga") or 0)*gp, (row.get("fg3m") or 0)*gp, (row.get("fta") or 0)*gp
    fgm, ftm, pts = (row.get("fgm") or 0)*gp, (row.get("ftm") or 0)*gp, (row.get("pts") or 0)*gp
    ts_denom = 2 * (fga + 0.44 * fta)
    efg = ((fgm + 0.5 * fg3m) / fga) if fga else None
    ts = (pts / ts_denom) if ts_denom else None
    ast, tov = (row.get("ast") or 0)*gp, (row.get("tov") or 0)*gp
    return {"player_id": player_id, "season": season, "source": "NBA LeagueDashPlayerStats per-game",
            "metrics": {"effective_fg_pct_approx": efg, "true_shooting_pct_approx": ts,
                        "assist_turnover_ratio_approx": ast/tov if tov else None,
                        "points_per_game": row.get("pts"), "rebounds_per_game": row.get("reb"),
                        "assists_per_game": row.get("ast")},
            "limitations": ["eFG% and TS% are approximate because they are reconstructed from rounded per-game averages.",
                            "Usage rate and offensive/defensive rating are not inferred without required possession/team data."],
            "updated_at": row.get("updated_at")}


@router.get("/status")
def data_status():
    with connect() as db:
        runs = db.execute("SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 10").fetchall()
        players_count = db.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        games_count = db.execute("SELECT COUNT(*) FROM player_game_logs").fetchone()[0]
        seasons_count = db.execute("SELECT COUNT(DISTINCT season) FROM player_game_logs").fetchone()[0]
    return {"database": str(DB_PATH), "players": players_count, "player_game_logs": games_count,
            "ingested_seasons": seasons_count, "recent_runs": [dict(r) for r in runs],
            "checked_at": utc_now()}
