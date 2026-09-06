"""Compatible Phase 1 entry point using the shared evaluated training pipeline.

Use the same flags as train_and_save_model.py, e.g. --demo or
--player-id 201939 --season 2023-24. Importing this module does no network work.
The published bundle contains final-test predictions and calibration.svg.
"""

from nba_app.features import FEATURE_COLS, prepare_shots
from nba_app.training import chronological_split, evaluate, train_bundle
from train_and_save_model import main


if __name__ == "__main__":
    raise SystemExit(main())
