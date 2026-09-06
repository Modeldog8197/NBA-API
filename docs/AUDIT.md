# Repository audit and implementation plan

Inspected source: [Modeldog8197/basketball-short-predictor](https://github.com/Modeldog8197/basketball-short-predictor), commit `61d3b8d` (`Add files via upload`). The original history is retained in this repository.

## Existing behavior

The three training scripts fetch NBA regular-season shot attempts, derive six spatial features, and fit XGBoost classifiers. `train_and_save_model.py` saves a native model plus two joblib files. `phase3_api.py` loads those files at import time and provides single/batch prediction, player search, free-throw statistics, retraining, and an interactive SVG court embedded in Python. Phase 2 also creates SHAP and court-zone plots.

## Findings in priority order

| Priority | Finding in the original source | Consequence | Resolution direction |
| --- | --- | --- | --- |
| 1 | The API uses a 23.75-foot distance threshold for zones; the browser uses 22.5 feet for shot type; phase 2 uses horizontal bands. Distance and shot type are independently editable. | Corner threes and arc shots can disagree across training, predictions, and the court. Conflicting inputs produce plausible-looking answers. | One geometry contract, derived features, explicit conflict validation, and boundary tests. |
| 1 | Every probability is displayed with a fixed ±5 percentage point “confidence” range. | The interval has no statistical interpretation or evaluated coverage. | Remove it and explicitly state that uncertainty intervals are unavailable. |
| 1 | Train/test splitting randomly mixes attempts from the same games. The test set is passed to training as `eval_set`; no separate validation set exists. | Reported metrics do not measure a clean future-game holdout. With no early stopping, `eval_set` alone does not prove fitting leakage, but it does expose the final holdout during development. | Reproducible chronological game splits; training-only fitting; validation-only calibration; final test used for reporting. |
| 1 | Retraining overwrites three shared files separately and replaces global model state. | Partial writes and failed training can damage availability; one user's training changes another user's selected player. | Immutable player/season model IDs, staged publication, and explicit model selection in every request. |
| 2 | Feature and cleaning logic is duplicated across four files; categorical encoding is fitted per dataset. | Training/inference drift and small-sample failures are difficult to diagnose. | Shared features and stable encoding; audited row cleaning; dataset checks. |
| 2 | Phase 2 disables SHAP additivity checking and mixes log-odds and probability language. | Explanations may be misleading or fail to reconstruct the prediction. | Verify raw-margin reconstruction and any calibration transform, expose residuals, and name contribution units. |
| 2 | Training has no cache, explicit request timeout, or bounded retry policy. Free throws are taken from the last career row; the UI hardcodes a season. | NBA outages stop progress, traded-player totals can be misread, and historical data may be shown for an unintended season. | Cached requests, bounded networking, exact season selection, and clear source metadata. |
| 2 | Retraining is unprotected, CORS permits all origins, errors include uncontrolled exception text, and missing artifacts prevent application startup. | Public deployments are fragile and expensive to operate; users cannot inspect an empty installation. | Validated configuration, protected opt-in training, explicit responses, logging, and usable empty states. |
| 3 | UI assets are inside Python; no automated tests, reproducible run instructions, or compatibility bounds exist. | Changes are hard to review and verify. | Separate static assets, responsive accessible controls, meaningful offline tests, CI, and setup/evaluation documentation. |

The original phase 1 script does compare log loss with a training-set mean baseline. That comparison is useful and is preserved conceptually; it does not supply Brier score, a chronological evaluation, or a complete reproducible model report. Original saved metrics are not treated as newly verified measurements.

## Implementation sequence

1. Preserve the original source in Git history and record this audit before assessing model improvements.
2. Establish a deterministic evaluation harness, then centralize court geometry, cleaning, and features with regression tests.
3. Implement game-aware splits, a simple baseline, a documented XGBoost reference, calibration, model reports, verified explanations, and atomic versioned bundles.
4. Add an injected data client, caching and timeout/retry controls, immutable API selection, and authenticated background training.
5. Replace the embedded interface with a responsive dashboard, shared court dimensions, keyboard controls, batch heatmap, explanations, model/data provenance, and selected-season free throws.
6. Run offline tests and a clearly labeled synthetic smoke evaluation; attempt real NBA retrieval separately and disclose any unavailable real-data comparisons.

## Dashboard refinement audit — September 2026

The initial standalone dashboard used an oversized promotional heading, very small supporting labels, neon accents, decorative gradients, and repeated bordered cards. Manual training was prominent, while selecting a player without a saved model led to a dead end. The stylesheet was compressed into long lines, making component changes harder to review.

The revision establishes shared neutral/blue tokens, system typography, consistent 44-pixel controls and 6-pixel radii, and flatter sections separated by rules. Navigation wraps on mobile; evaluation tables remain keyboard-scrollable. Focus, disabled, empty, loading, and failure states are explicit. The court geometry, predictions, heatmap, explanations, historical free throws, evaluation, and advanced training controls remain available. Missing player models now prepare on demand for the exact selected season, with bounded progress tracking and clear unavailable-data messages.

## Geometry references

The official NBA rule defines a 23-foot-9-inch arc joined to straight segments three feet inside each sideline. The implementation derives the segment/arc junction rather than approximating it. See [NBA Rule 1: Court Dimensions and Equipment](https://official.nba.com/rule-no-1-court-dimensions-equipment/).

The [nba_api ShotChartDetail endpoint documentation](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/shotchartdetail.md) describes the available shot fields and request filters. Player tracking, defender positions, and other unobserved context are not invented.
