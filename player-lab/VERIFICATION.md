# Player Lab validation

The published dataset contains 1,477 players, 5,492 player-seasons, 2,111,521 recorded shot attempts, 5,463 expanded spatial estimates and 32 retained original models. All ten completed seasons are present. Exactly reconciled player-seasons: 5,366; explicitly flagged cross-provider discrepancies: 126.

`npm test` checks counts, traded-player totals, representative official NBA totals, ESPN game-log reconciliation, separation of game types, unavailable-feed fallback, input validation, spatial attribution reconstruction, selection coverage and retained legacy prediction delegation. All 18 tests pass, including a Pages HTML asset and script-order regression check.

DOM verification covers player search, filters, charts/tabs, ten-season history, advanced metrics, CSV-capable logs and error/empty states. Shot verification covers search, seasons, coordinate-dependent estimates, heatmaps, metrics and discrepancy messages. Pages-specific verification checks repository-subpath assets, gzip data loading, lazy season partitions, saved log fallback and complete model records. Browser visual QA was unavailable; DOM checks use jsdom.

Representative shot charts verified: Curry 2022–23, 1,133 attempts / 559 makes; Durant 2023–24, 1,436 / 751; Durant 2016–17, 1,026 / 551. Curry 2023–24 season totals and representative Curry/Jokic/Butler logs reconcile. Older incomplete LeBron/Dirk provider logs are labeled.

New historical predictions run from committed data on Pages. Public feeds and original Render model inference remain external dependencies with loading, freshness and retry/error states. Shot archives are historical; live shot coordinates are not supplied.
