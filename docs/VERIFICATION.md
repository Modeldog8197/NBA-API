# Verification

Verified locally on Windows with Python 3.12 and the exact dependency snapshot in `requirements-lock.txt`. The automated and multi-player preparation results below were updated on 2026-09-06.

- **145 Python tests passed.** NBA requests are mocked in automated tests. The only warnings are two upstream Starlette/httpx/AnyIO deprecation warnings.
- `python -m pip check`: no broken requirements.
- Python fatal-error static checks passed.
- `node scripts/test_dashboard.cjs` passed: contribution and evaluation/calibration rendering, stale model/season/prediction responses after selection changes, hosted-key preparation, and recovery from an unavailable player model.
- Preparation tests cover simultaneous duplicate requests, separate player and season identities, validated saved-model reuse, corrupt or synthetic bundle rejection, unavailable career seasons, insufficient samples, explicit retries, an eight-job queue limit, bounded history, shutdown, and shared training-lock ownership. A test uses the actual trainer to reject a small sample and then prepare successfully after more mocked data is supplied.
- Local authorization tests cover the loopback session token and reject remote clients, nonlocal Host values, cross-origin requests, and missing or invalid local tokens. Hosted preparation remains protected by the configured Bearer key; ordinary settings and the application factory disable local preparation by default.
- Browser verified: initial empty-model state, loaded Curry model, corner-three classification and expected points, 380-location probability heatmap, TreeSHAP contributions and reconstruction indicator, keyboard movement, and responsive layout at 390 pixels.
- Revised interface verified in the browser: neutral/blue desktop layout, actual player-season choices, saved Luka predictions and free throws, and the mobile navigation, controls, and heatmap at a 390-pixel viewport with no horizontal page overflow. Asset versions prevent the previous dashboard styles and script from remaining cached after the update.
- Real NBA retrieval: 1,445 Curry shots from 2023–24; 1,443 retained after excluding two heaves. The complete split and measured comparisons are in [EVALUATION.md](EVALUATION.md).
- Real historical free throws: 299 makes / 324 attempts in 2023–24 (92.3% when rounded), retrieved through PlayerCareerStats. This statistic is separate from predictions.
- Live model training, synthetic demo CLI, and phase 2 analysis completed. Published test metrics independently recomputed from the saved CSV agree to floating-point precision (maximum difference `1.11e-16`).

## Real multi-player preparation

The shared preparation service and batch CLI fetched real NBA 2023–24 records and completed three additional player models. The four included bundles are:

| Player | NBA ID | Usable shots | Immutable model ID |
| --- | ---: | ---: | --- |
| Stephen Curry | 201939 | 1,443 | `p201939-2023-24-20260905T184018153423Z-fe3743c6` |
| LeBron James | 2544 | 1,268 | `p2544-2023-24-20260906T045023255859Z-a7eaa0de` |
| Nikola Jokić | 203999 | 1,403 | `p203999-2023-24-20260906T045025096682Z-bd1d5dfb` |
| Luka Dončić | 1629029 | 1,647 | `p1629029-2023-24-20260906T045026757177Z-25cc9b23` |

Each bundle contains its own provenance, chronological split, baseline comparisons, test predictions, and calibration plot. These runs establish that multiple player identities can fetch, train, publish, and load through the shared workflow. They do not establish that every catalog player has sufficient NBA data, or that these models have reliable future predictive performance.

The local dashboard can prepare other exact player-season combinations on demand. A player-season must have recorded field-goal attempts and satisfy the existing minimum sample and split requirements. An unavailable season or upstream failure yields a visible unavailable state; it never substitutes Curry or another player's predictions. Jobs are deduplicated and saved versions remain immutable.

The selected Curry spatial model has better final-test log loss than the original recipe but worse log loss than the simple baseline. The dashboard displays this limitation. Inspect the individual evaluation of each additional model; Curry's comparison is not a performance claim for other players. No future-season reliability, calibrated interval coverage, or statistically significant improvement is claimed. GitHub CI must run after publication; the local results do not claim a remote CI run.
