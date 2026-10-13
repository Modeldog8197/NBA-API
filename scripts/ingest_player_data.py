"""Populate the player database for the latest ten seasons.

Run: python -m scripts.ingest_player_data [--include-playoffs]
NBA endpoints can block hosted runners; failures are logged and do not erase existing data.
"""
import argparse
import logging
import time

from nba_app.player_data import ingest_season, ingest_season_totals, seasons_to_fetch, sync_player_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--seasons", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=1.5)
    args = parser.parse_args()
    print(f"Player registry: {sync_player_registry()} records")
    errors = []
    for season in seasons_to_fetch(args.seasons):
        for season_type in (("Regular Season", "Playoffs") if args.include_playoffs else ("Regular Season",)):
            for dataset, loader in (("game logs", ingest_season), ("season totals", ingest_season_totals)):
                try:
                    result = loader(season, season_type)
                    logging.info("%s %s %s: %s rows", dataset, season, season_type, result["rows_written"])
                except Exception as exc:
                    errors.append((season, season_type, dataset, str(exc)))
                    logging.error("%s %s %s failed: %s", season, season_type, dataset, exc)
                time.sleep(max(0, args.sleep))
    if errors:
        logging.warning("%s ingestion tasks failed; see logs. Existing stored data was preserved.", len(errors))
    if not errors:
        logging.info("Historical ingestion completed without errors.")


if __name__ == "__main__":
    main()
