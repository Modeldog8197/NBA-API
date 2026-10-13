# NBA API — Player Lab

**Website: https://modeldog8197.github.io/NBA-API/**

**Shot analysis: https://modeldog8197.github.io/NBA-API/shot-lab/**

This directory contains the player explorer and expanded shot analysis. The original Python API, models, dashboard and historical Git history remain intact in the repository root.

## Coverage and sources

The explorer covers 1,477 players and 5,492 player-seasons over the ten completed seasons 2016–17 through 2025–26. Search, season/team/position/game filters, career charts, advanced metrics, game logs and CSV exports are available.

Shot analysis includes 2,111,521 recorded attempts and 5,463 location-dependent spatial estimates. The 29 zero-attempt player-seasons retain their season and free-throw totals. All 32 original model versions remain selectable. There are 5,366 player-seasons with exactly reconciled counts and 126 with differences across sources; those differences are displayed. Missing locations are never invented. Off-court attempts count in overall totals but are excluded from the half-court drawing and spatial training.

- Season statistics: [NBA_Go](https://api.server.nbaapi.com/swagger/index.html).
- Shot outcomes and coordinates: [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data), pinned commit `e829d4678be1e075f99e5d41a1c5f97089be446b`; `datasets/shotdetail_2016.tar.xz` through `shotdetail_2025.tar.xz`. Source license is retained in `shot-data/SOURCE-LICENSE.txt`. Year in each source filename is the season's starting year.
- Game logs, scores and box scores: ESPN public JSON feeds.

## How Pages runs

GitHub Pages serves static HTML, JavaScript and versioned compressed data. The explorer and new spatial models run in the browser, without client secrets, a private service token or a server cold start. The fetch adapter presents the same application interface as the optional server implementation. Compressed shot partitions decode on demand; at most two decoded season packs are retained. The metadata and archives are loaded once per tab, with retries after failed loads. Model records can be downloaded as JSON.

Live scores and box scores use public ESPN feeds and refresh every 30 seconds while the tab is visible. A bounded per-tab cache, request coalescing and retries handle feed interruptions. Representative saved game logs provide labeled fallback; older/incomplete provider logs are flagged against archived season totals. Public feed availability is not guaranteed, and no missing games are estimated. Historical shot data is a versioned archive, not live shot tracking.

Original XGBoost/constant model inference calls the existing public Render API. Its configured CORS origin must permit `https://modeldog8197.github.io`; cold starts can take up to 85 seconds and show a retryable error. Pages training is disabled. `NBA_API_URL` is an optional public HTTPS backend URL in GitHub Actions variables, defaulting to the existing Render service; it must never contain credentials. This deployment does not modify Render settings or the existing private companion site's audience.

`server.mjs` retains the optional server-side refresh approach: fixed validated provider URLs and cache keys, 25-second live cache, deduplicated requests, bounded retries, persistent-cache support, memory fallback and stale responses labeled with timestamps. Future licensed-provider keys belong in server environment secrets. Pages cannot execute server code or unattended refresh jobs; new archive releases require a verified source refresh, rebuild and push.

## Spatial model and validation

The model combines pooled regularized logistic regression with an empirical-Bayes player-region adjustment. Eleven geometry/zone features and the regional term reconstruct its log-odds. Sparse players/regions shrink toward the league profile. Explanations are additive logistic attribution, not TreeSHAP or causal effects. Original models keep their original TreeSHAP explanations.

Complete NBA games are sorted chronologically across the league: first 60% train, next 20% select regional shrinkage, final 20% evaluate. All events from held-out games are excluded from shared-model training. Records show Brier score, log loss, ROC-AUC and a constant baseline. Some estimates underperform that baseline and the UI reports this. Small samples have no reliable individual accuracy claim; calibrated uncertainty intervals are unavailable.

## Build and test

Node 22 or later; no npm dependencies are needed for the Pages build.

```sh
cd player-lab
npm test
npm run build:pages
```

The build creates `artifacts/pages` in the repository root. Data snapshots are committed as gzip files; `scripts/restore-data.mjs` restores ignored JSON working copies before tests and build. The root `.github/workflows/pages.yml` validates data, builds, uploads and deploys on changes to `main`.

Optional DOM checks require jsdom:

```sh
node scripts/bundle.mjs
node tests/ui-check.cjs
node tests/shot-ui-check.cjs
node tests/pages-ui-check.cjs
```

To reproduce the spatial fit, first restore snapshots, then run `scripts/import-shot-archive.py`, `scripts/map-shot-identities.py` and `scripts/build-spatial-shot-data.py` with NumPy/scikit-learn. Raw source archives are retrieved from the pinned source; fitted partitions, the identity map, model metadata and reconciliation records are already committed. After a source update, re-compress changed JSON snapshots, rerun validations and push a versioned archive release. No scheduled external changes are enabled.
