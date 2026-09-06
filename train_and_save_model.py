"""Train and publish one immutable, evaluated player-season model.

Examples:
    python train_and_save_model.py --player-id 201939 --season 2023-24
    python train_and_save_model.py --player "Stephen Curry" --season 2023-24
    python train_and_save_model.py --demo --season 2023-24
    python train_and_save_model.py --csv shots.csv --player-id 201939 --player "Stephen Curry" --season 2023-24
"""

import argparse
from dataclasses import replace
import logging
from pathlib import Path

import pandas as pd

from nba_app.config import Settings, validate_season
from nba_app.data import NBADataClient
from nba_app.models import ModelStore
from nba_app.training import train_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player-id", type=int, help="Stable NBA player ID (preferred)")
    parser.add_argument("--player", help="Exact full player name, or local CSV display name")
    parser.add_argument("--season", help="NBA season, for example 2023-24; required for NBA or CSV data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Use an explicitly synthetic fixture, never an NBA player's identity")
    mode.add_argument("--csv", type=Path, help="Train from an NBA-shaped local CSV, with provenance recorded")
    parser.add_argument("--model-dir", type=Path, help="Override NBA_MODEL_DIR")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.demo and (args.player_id is not None or args.player):
        parser.error("--demo always uses player 0 'Synthetic demo'; omit --player and --player-id.")
    if not args.demo and not args.season:
        parser.error("--season is required for NBA and CSV training.")
    if args.csv and args.player_id is None:
        parser.error("--csv requires --player-id to preserve model identity.")
    if not args.demo and args.player_id is None and not args.player:
        parser.error("Specify --player-id, --player, or --demo.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        season = validate_season(args.season or "2023-24")
        settings = Settings.from_env()
        if args.model_dir:
            settings = replace(settings, model_dir=args.model_dir)
        if args.demo:
            from scripts.generate_fixture import make_fixture
            df = make_fixture(season=season)
            identity = {"id": 0, "name": "Synthetic demo"}
            source_info = {"source": "synthetic_fixture", "generator": "scripts.generate_fixture.make_fixture",
                           "seed": 42, "description": "Generated engineering fixture, not NBA data."}
        elif args.csv:
            df = pd.read_csv(args.csv, dtype={"GAME_ID": str, "GAME_EVENT_ID": str, "GAME_DATE": str})
            identity = {"id": args.player_id, "name": args.player or f"Player {args.player_id}"}
            source_info = {"source": "local_csv", "file_name": args.csv.name,
                           "description": "User-supplied CSV. Source authenticity has not been independently verified."}
        else:
            client = NBADataClient(settings)
            identity = client.resolve_player(args.player_id if args.player_id is not None else args.player)
            if args.player_id is not None and args.player and identity["name"].casefold() != args.player.casefold():
                raise ValueError("--player name conflicts with --player-id.")
            df, source_info = client.fetch_shots(identity["id"], season)
        metadata = train_bundle(df, identity["id"], identity["name"], season, ModelStore(settings.model_dir),
                                source_info=source_info, progress=lambda message: print(f"- {message}", flush=True))
    except Exception as exc:
        logging.error("Training failed; previous model versions remain available: %s", exc)
        return 1
    print(f"\nPublished: {metadata['model_id']}")
    print(f"Selected on later validation: {metadata['selected_model']}")
    print(metadata["evaluation"]["interpretation"])
    print("Final test comparisons:")
    for name, metrics in metadata["evaluation"]["final_test"].items():
        print(f"  {name:22s} log loss={metrics['log_loss']:.4f} Brier={metrics['brier_score']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    print(f"Bundle and calibration plot: {settings.model_dir.resolve() / metadata['model_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
