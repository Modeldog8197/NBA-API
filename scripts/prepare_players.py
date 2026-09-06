"""Prepare reusable models from actual NBA records for multiple players.

Run from the project root with ``python -m scripts.prepare_players``.
Examples:
    python -m scripts.prepare_players --season 2023-24 --player "LeBron James" "Nikola Jokić"
    python -m scripts.prepare_players --season 2023-24 --player-id 2544 203999
    python -m scripts.prepare_players --season 2023-24 --all-active
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from nba_api.stats.static import players

from nba_app.config import Settings, validate_season
from nba_app.data import NBADataClient
from nba_app.models import ModelStore
from nba_app.preparation import ACTIVE_STATES, ModelPreparationService


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--player-id", "--player-ids", nargs="+", action="extend", type=int, help="Stable NBA player IDs")
    selected.add_argument("--player", nargs="+", action="extend", help="Quoted exact NBA player names")
    selected.add_argument("--all-active", action="store_true", help="Prepare players marked active in the NBA catalog")
    parser.add_argument("--season", required=True, help="Exact NBA season; no other season is substituted")
    parser.add_argument("--model-dir", type=Path, help="Override NBA_MODEL_DIR")
    parser.add_argument("--report", type=Path, help="Write per-player outcomes to a JSON file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        season = validate_season(args.season)
        settings = Settings.from_env()
        if args.model_dir:
            settings = replace(settings, model_dir=args.model_dir)
        client = NBADataClient(settings)
        identifiers = ([player["id"] for player in players.get_active_players()] if args.all_active
                       else args.player_id or args.player)
        identities = {player["id"]: player for player in (client.resolve_player(value) for value in identifiers)}
    except ValueError as exc:
        parser.error(str(exc))
    service = ModelPreparationService(settings, client, ModelStore(settings.model_dir))
    outcomes = []
    interrupted = False
    try:
        for player in identities.values():
            print(f"\n{player['name']} | {season}", flush=True)
            job = service.prepare(player["id"], season)
            previous = None
            while job["status"] in ACTIVE_STATES:
                if job["message"] != previous:
                    print(f"  {job['message']}", flush=True)
                    previous = job["message"]
                time.sleep(0.1)
                job = service.job_status(job["job_id"])
            print(f"  {job['status']}: {job['message']}", flush=True)
            if job["model_id"]:
                print(f"  {job['model_id']}", flush=True)
            outcomes.append(job)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped queuing players. Any running preparation will finish safely.", flush=True)
    finally:
        service.close(wait=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"season": season, "interrupted": interrupted, "players": outcomes},
                                         indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    ready = sum(job["status"] == "done" for job in outcomes)
    print(f"\n{ready}/{len(outcomes)} models ready. Unavailable seasons require more actual NBA records.")
    return 130 if interrupted else int(any(job["status"] != "done" for job in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
