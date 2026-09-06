"""Synthetic engineering fixture only; this is not historical NBA data."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nba_app.court import describe_shot


def make_fixture(player_id: int = 0, games: int = 60, shots_per_game: int = 24,
                 seed: int = 42, season: str = "2023-24") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    start = date(int(season[:4]), 10, 15)
    for game in range(games):
        for event in range(shots_per_game):
            if rng.random() < 0.35:
                x, y = rng.uniform(-65, 65), rng.uniform(-15, 90)
            else:
                x, y = rng.uniform(-242, 242), rng.uniform(-35, 325)
            x, y = round(x, 1), round(y, 1)
            shot = describe_shot(x, y)
            # Deliberately synthetic distance response; never claim NBA validity.
            log_odds = 1.1 - 0.07 * shot["distance_ft"] - 0.2 * abs(x / 250)
            probability = 1 / (1 + np.exp(-log_odds))
            records.append({"PLAYER_ID": player_id, "GAME_ID": f"fixture-{game:04d}",
                            "GAME_DATE": (start + timedelta(days=game * 3)).strftime("%Y%m%d"),
                            "GAME_EVENT_ID": event + 1, "LOC_X": x, "LOC_Y": y,
                            "SHOT_DISTANCE": int(shot["distance_ft"]),
                            "SHOT_TYPE": "3PT Field Goal" if shot["shot_value"] == 3 else "2PT Field Goal",
                            "SHOT_MADE_FLAG": int(rng.random() < probability)})
    return pd.DataFrame(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="CSV output path")
    parser.add_argument("--season", default="2023-24")
    args = parser.parse_args()
    make_fixture(season=args.season).to_csv(args.output, index=False)
    print(f"Saved explicitly synthetic fixture to {args.output}")
