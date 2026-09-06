# NBA API

A basketball shot probability dashboard and FastAPI service built with Python and XGBoost. Select a player, season, and immutable model version; place a shot on the court; inspect make probability, expected points, a probability heatmap, and checked feature explanations.

This project builds on [basketball-short-predictor](https://github.com/Modeldog8197/basketball-short-predictor). Its original Git history is preserved. See the [source audit and implementation plan](docs/AUDIT.md) and [evaluation methodology and evidence](docs/EVALUATION.md).

**Prediction quality must be measured on actual held-out NBA games.** The offline demo is synthetic and does not represent an NBA player's ability. No validated confidence interval is available, and the interface does not invent one.

A real Curry 2023–24 evaluation completed on 299 held-out attempts. The selected candidate's log loss improved from the original parameter reference's **0.7721 to 0.7011**, but the constant baseline was better at **0.6871**. See the [full measured comparison and calibration plot](docs/EVALUATION.md); these results do not establish reliable future performance.

## Quick start

Use Python 3.11 or 3.12. From the project directory:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install and launch the included, evaluated Curry 2023–24 reference model:

```bash
python -m pip install -r requirements.txt
python phase3_api.py
```

The reference model is included as an immutable bundle in `models/`. Its limited held-out performance is documented below and displayed in the dashboard. Newly trained bundles remain local by default. Historical free-throw data still requires an NBA request or an existing cache.

To create a separate, explicitly synthetic offline demo:

```bash
python -m pip install -r requirements.txt
python train_and_save_model.py --demo --season 2023-24
python phase3_api.py
```

For the exact verified Python 3.12 dependency snapshot, install `requirements-lock.txt` instead; it includes the development/test packages. The ordinary requirements files use compatible version bounds, and CI checks Python 3.11 and 3.12.

Open [the dashboard](http://localhost:8000) or [interactive API documentation](http://localhost:8000/docs). The application also starts without a model, showing an empty state and setup guidance. The demo is explicitly labeled **Synthetic demo** and uses player ID `0`.

To train on real regular-season shot data instead:

```bash
python train_and_save_model.py --player-id 201939 --season 2023-24
```

NBA requests may time out or be blocked by the upstream service. A successful fetch is cached; an eligible stale cache can be used when refreshing fails, and the model records that source. If no data can be retrieved, the command fails without replacing a working model. The demo remains available for evaluating the software workflow.

## Coordinates and basketball rules

- `loc_x` and `loc_y` are **tenths of a foot**. The basket is `(0, 0)`. Positive x is the viewer's right; positive y points toward midcourt.
- The supported half court is `-250 <= x <= 250`, `-52.5 <= y <= 417.5`. Full-court heaves are outside the supported domain.
- Distance in feet is `sqrt(x*x + y*y) / 10`. The app derives distance and shot type from coordinates. Optional supplied `shot_distance` must agree within 0.1 foot; optional `shot_type` must agree with geometry.
- Three-point geometry joins corner lines at `x = +/-220` to the `237.5`-unit arc at `y = sqrt(237.5^2 - 220^2)`, approximately `89.48`. A point exactly on the line is classified as a two-pointer. A location at `(230, 0)` is a corner three even though it is only 23 feet from the basket.
- Training recomputes distance from coordinates because source `SHOT_DISTANCE` is rounded. Missing shot types are derived; conflicting recorded types are excluded and counted in the cleaning report. Shot-chart locations cannot reconstruct foot placement or physical line width.
- Expected points is `shot_value * make_probability`. It excludes free throws, fouls, rebounds, turnovers, and possession context.

Court dimensions follow [NBA Rule 1](https://official.nba.com/rule-no-1-court-dimensions-equipment/). Zone names are the application's documented geometry categories, not a claim to reproduce every NBA provider zone label.

## API examples

Every prediction requires a specific `model_id`; training or selecting another player never silently changes it. First obtain a model ID:

```bash
curl http://localhost:8000/models
```

Use the returned ID in the examples below. On Windows use `curl.exe` if `curl` is a PowerShell alias. Multiline examples below use Bash line continuation; use the interactive API documentation for a shell-independent request editor.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id":"MODEL_ID_FROM_MODELS","loc_x":230,"loc_y":0}'

curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"model_id":"MODEL_ID_FROM_MODELS","shots":[{"loc_x":0,"loc_y":30},{"loc_x":230,"loc_y":0}],"explain":false}'

curl 'http://localhost:8000/players/search?q=Curry'
curl 'http://localhost:8000/player/freethrow?player_id=201939&season=2023-24'
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Application availability and training configuration; works without model artifacts. |
| `GET /court` | Shared coordinate bounds, three-point geometry, and units. |
| `GET /players/search?q=...` | Search the NBA static player catalog and retrieve stable IDs. |
| `GET /seasons` | Available season choices. |
| `GET /models` | Available versioned model metadata. |
| `GET /models/{model_id}` | Provenance, feature contract, and evaluation for one model. |
| `POST /predict` | Predict a single shot using an explicitly selected model. |
| `POST /predict/batch` | Predict 1–500 shots under one model identity; accepts `explain: false` for heatmaps. |
| `GET /player/freethrow` | Historical free-throw totals for the requested player and exact season. |
| `POST /model/retrain` | Start protected background training; returns a job ID. |
| `GET /model/retrain/status?job_id=...` | Inspect training progress or failure for that job. |

Free-throw percentages are historical made/attempted counts, not the field-goal model's prediction. A missing season or no attempts is reported explicitly rather than substituted with another season.

## Configuration and hosted retraining

Settings are validated at startup. Configure through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NBA_MODEL_DIR` | `models` | Versioned model bundles. |
| `NBA_CACHE_DIR` | `data/cache` | Cached NBA responses. |
| `NBA_TRAINING_ENABLED` | `false` | Enable the retraining HTTP endpoint. CLI training is independent. |
| `NBA_TRAINING_API_KEY` | unset | Bearer secret required when HTTP training is enabled. |
| `NBA_REQUEST_TIMEOUT` | `10` | Per-request NBA timeout in seconds. |
| `NBA_RETRIES` | `2` | Bounded retries after the initial attempt. |
| `NBA_CACHE_TTL_SECONDS` | `86400` | Freshness window for cache entries. |
| `NBA_ALLOW_STALE_CACHE` | `true` | Allow an expired cached response if refreshing fails, with visible stale provenance. |
| `NBA_CORS_ORIGINS` | empty | Comma-separated allowed origins; same-origin dashboard needs none. |

For local opt-in retraining on PowerShell, set a strong secret of your own before starting the server:

```powershell
$env:NBA_TRAINING_ENABLED = "true"
$env:NBA_TRAINING_API_KEY = "replace-with-your-own-long-random-secret"
python phase3_api.py
```

The dashboard accepts the key for training requests. Do not commit real secrets. A retrain request has the following form:

```bash
curl -X POST http://localhost:8000/model/retrain \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TRAINING_KEY" \
  -d '{"player_id":201939,"season":"2023-24"}'
```

Use one API worker for the built-in in-process training queue. Job history resets when the server restarts. A filesystem lock prevents simultaneous API training across processes; a crash can leave `models/.api-training.lock`. Remove that lock only after confirming no training process is running. For public hosting, use HTTPS and an authenticated, rate-limited reverse proxy; a shared durable job queue is needed before scaling training across processes. Model directories and caches should be writable only by the application operator. Publication is atomic within the model directory; previous versions remain available after a failed job.

## Model and explanation contract

Training and inference share the ordered features `LOC_X`, `LOC_Y`, `SHOT_DISTANCE`, `SHOT_TYPE_ENC`, `SHOT_ANGLE`, and `SHOT_ANGLE_ABS`. Encoding is fixed (`2pt=0`, `3pt=1`); angles are radians from the positive y axis. No defender proximity, fatigue, shot clock, tracking data, or play-type fields are synthesized.

Training keeps games together and separates chronological training, calibration, selection, and final testing. Reports compare a constant smoothed training-rate baseline, the original XGBoost parameter recipe under the new split/feature contract, and a regularized candidate. The held-out results, calibration plots, data-cleaning counts, source, features, parameters, and dependency versions are stored with each immutable bundle. The baseline can be selected if it has the best later-validation log loss. See [EVALUATION.md](docs/EVALUATION.md) for precise methodology and interpretation.

For XGBoost models, feature contributions are exact native TreeSHAP values in the raw **log-odds** scale. The service checks that contributions plus the base value reconstruct the raw score, then applies the model's documented probability/calibration transform. A selected constant baseline has zero feature contributions and is labeled accordingly. Contributions describe this model's associations; they do not demonstrate that moving or changing one correlated feature causes a particular change in scoring probability. No confidence interval is displayed.

Each model folder contains `metadata.json`, `test_predictions.csv`, and `calibration.svg`, plus `model.json` when an XGBoost variant is selected. File checksums and the feature contract are verified on load. To create a single-shot explanation and final-test zone/feature summaries for a model:

```bash
python phase2_shap_zones.py --model-id MODEL_ID_FROM_MODELS --x 230 --y 20
```

The `analysis` directory receives JSON, CSV, and SVG outputs; no new training or NBA request is performed. `python phase1_shot_quality_model.py --demo` is also supported as a shared training entry point. For a local data snapshot, use `python train_and_save_model.py --csv shots.csv --player-id 201939 --player "Stephen Curry" --season 2023-24`; source authenticity is recorded as user-supplied and is not independently verified.

## Project layout

```text
nba_app/
  api.py          FastAPI routes, schemas, model identity, training jobs
  config.py       Validated environment settings
  court.py        Court dimensions and shot classification
  data.py         NBA requests, exact-season statistics, disk cache
  features.py     Shared cleaning and ordered spatial features
  models.py       Versioned model loading and checked inference
  training.py     Chronological evaluation and bundle publication
static/           Dashboard HTML, CSS, and JavaScript
scripts/          Explicit synthetic fixture generator
tests/            Offline geometry, model, API, and data regression tests
docs/             Audit and evaluation notes
train_and_save_model.py  Training CLI
phase3_api.py            Local server entry point
```

The phase 1 and phase 2 entry points use shared modules rather than maintaining separate feature/training implementations. Original tracked artifacts are historical material and are not automatically treated as a current, validated model.

## Verification and limitations

Run the automated suite without live NBA access:

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
node scripts/test_dashboard.cjs
```

Tests cover court boundaries, input conflicts, feature agreement, data quality, chronological game separation, explanation reconstruction, immutable model selection, batches, missing/corrupt bundles, failed training, retraining authorization, mocked NBA failures/retries, cache behavior, and selected-season free throws. CI runs on Python 3.11 and 3.12. Dashboard regression checks use Node.js 20 or newer; Node is not required to run the app. See [verification results](docs/VERIFICATION.md).

The model learns a player's historical location/outcome associations from available regular-season field goals. Small samples are rejected; adequate volume still does not establish reliability in sparse court regions or future seasons. Heatmaps can interpolate into locations the player rarely attempts. Synthetic smoke-test metrics prove the pipeline runs, not NBA predictive performance. Always inspect the model's provenance and held-out evaluation before interpreting predictions.

Shot data comes from NBA Stats through [nba_api's ShotChartDetail endpoint](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/shotchartdetail.md). Player free throws come from its [PlayerCareerStats endpoint](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playercareerstats.md). This is an independent project, not an official NBA service. The upstream repository did not include a license; no new license is asserted here.
