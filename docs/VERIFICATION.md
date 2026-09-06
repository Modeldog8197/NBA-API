# Verification

Verified locally on Windows with Python 3.12 and the exact dependency snapshot in `requirements-lock.txt`.

- **106 Python tests passed.** NBA requests are mocked in automated tests. The only warnings are two upstream Starlette/httpx/AnyIO deprecation warnings.
- `python -m pip check`: no broken requirements.
- Python fatal-error static checks passed.
- `node scripts/test_dashboard.cjs` passed: actual contribution-field rendering, evaluation/calibration rendering, and a delayed response from an old model cannot overwrite a new selection.
- Browser verified: initial empty-model state, loaded Curry model, corner-three classification and expected points, 380-location probability heatmap, TreeSHAP contributions and reconstruction indicator, keyboard movement, and responsive layout at 390 pixels.
- Real NBA retrieval: 1,445 Curry shots from 2023–24; 1,443 retained after excluding two heaves. The complete split and measured comparisons are in [EVALUATION.md](EVALUATION.md).
- Real historical free throws: 299 makes / 324 attempts in 2023–24 (92.3% when rounded), retrieved through PlayerCareerStats. This statistic is separate from predictions.
- Live model training, synthetic demo CLI, and phase 2 analysis completed. Published test metrics independently recomputed from the saved CSV agree to floating-point precision (maximum difference `1.11e-16`).

The selected spatial model has better final-test log loss than the original recipe but worse log loss than the simple baseline. The dashboard displays this limitation. No future-season reliability, calibrated interval coverage, or statistically significant improvement is claimed. GitHub CI must run after publication; the local results do not claim a remote CI run.
