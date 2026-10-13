# Player data integration

This adds a separate player-data layer without changing the existing shot-chart/model pipeline.

## What it provides

- Stable NBA player IDs and an active/historical player registry.
- SQLite storage with upserts for per-game logs and per-game season totals.
- Routes under `/v1/data`: `/players`, `/seasons`, `/players/{id}/games`, `/players/{id}/stats`, `/players/{id}/advanced`, and `/status`.
- A historical ingestion command covering the latest ten seasons, optionally including playoffs.
- A best-effort scheduled refresh workflow. GitHub-hosted runners and NBA Stats are not a guaranteed real-time delivery path; refresh failures are logged and existing rows remain available.

## Enable the API routes

In `nba_app/api.py`, after the FastAPI instance is created inside `create_app`, add:

```python
from .player_data import router as player_data_router
app.include_router(player_data_router)
```

The router is intentionally isolated so it can be mounted after review without altering the shot-model API's lifecycle or data-client behavior.

## Populate historical data

```bash
export NBA_PLAYER_DB=data/nba_players.sqlite3
python -m scripts.ingest_player_data --seasons 10 --include-playoffs
```

This is a best-effort process because the NBA Stats endpoint can rate-limit or block cloud requests. Re-run safely: game rows are keyed by player and game and upserted, not duplicated. Failed endpoint calls do not clear stored data.

## Deploying

- GitHub Pages remains the static frontend; it cannot execute FastAPI or Python.
- To serve the new routes publicly, mount the router and deploy the API on a persistent Python host.
- Set `NBA_PLAYER_DB` to a persistent volume path on that host. A SQLite database in an ephemeral container will be lost at redeploy; use PostgreSQL for a multi-instance production deployment.
- Set the Pages repository variable `NBA_API_URL` to the API's HTTPS origin and configure explicit `NBA_CORS_ORIGINS` on the API.
- Do not promise second-by-second live data. This scheduled workflow runs every 15 minutes when GitHub Actions permits, but upstream availability and scheduled-workflow delays apply. For dependable in-game updates, use a licensed/stable live feed and a persistent worker.

## Metric caveats

The initial advanced route exposes approximate eFG% and TS% reconstructed from rounded per-game totals, plus per-game box-score statistics. It explicitly does not invent usage rate, offensive rating, or defensive rating without the necessary possession and team context. These approximations should be replaced with exact totals or authoritative advanced endpoints before formal analysis.
