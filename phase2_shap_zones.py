"""Explain an existing version without retraining or downloading new data.

python phase2_shap_zones.py --model-id MODEL_ID --x 230 --y 20

Outputs explanation.json, shot_contributions.svg, feature_importance.svg,
zone_summary.csv, and zone_comparison.svg. Zone and global summaries use the
same final-test rows used to evaluate this version, never resampled shots.
"""

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_app.config import Settings
from nba_app.features import FEATURE_COLS, FEATURE_LABELS
from nba_app.models import ModelStore


def _bar_chart(labels, values, title: str, subtitle: str, color="#0891b2") -> str:
    width, height = 860, 140 + 44 * len(labels)
    max_value = max(max(abs(float(v)) for v in values), 1e-8)
    rows = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<g fill="#0f172a" font-family="sans-serif">',
            f'<text x="25" y="33" font-size="21">{escape(title)}</text>',
            f'<text x="25" y="57" font-size="12">{escape(subtitle)}</text>']
    for index, (label, value) in enumerate(zip(labels, values)):
        value, y = float(value), 90 + index * 44
        bar_width = 220 * abs(value) / max_value
        x = 560 - bar_width if value < 0 else 560
        bar_color = "#ea580c" if value < 0 else color
        rows += [f'<text x="25" y="{y+17}" font-size="13">{escape(str(label))}</text>',
                 f'<rect x="{x}" y="{y}" width="{bar_width}" height="25" fill="{bar_color}"/>',
                 f'<text x="800" y="{y+17}" font-size="12">{value:+.3f}</text>']
    rows += [f'<path d="M560 80V{height-30}" stroke="#94a3b8"/>', '</g></svg>']
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    args = parser.parse_args(argv)
    try:
        store = ModelStore(args.model_dir or Settings.from_env().model_dir)
        bundle = store.load(args.model_id)
        prediction = bundle.predict([{"loc_x": args.x, "loc_y": args.y}], explain=True)[0]
        explanation = prediction["explanation"]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "explanation.json").write_text(json.dumps({"model_id": args.model_id, **prediction}, indent=2), encoding="utf-8")
        labels = [item["label"] for item in explanation["contributions"]]
        values = [item["value"] for item in explanation["contributions"]]
        (args.output / "shot_contributions.svg").write_text(_bar_chart(labels, values, "Shot feature contributions", "Raw model log-odds; not percentage points or causal advice. See explanation.json for calibration."), encoding="utf-8")
        test_path = store.root / args.model_id / "test_predictions.csv"
        test = pd.read_csv(test_path, dtype={"GAME_ID": str, "GAME_EVENT_ID": str})
        shots = test.rename(columns={"LOC_X": "loc_x", "LOC_Y": "loc_y"})[["loc_x", "loc_y"]].to_dict("records")
        test_explanations = []
        for start in range(0, len(shots), 1000):
            test_explanations.extend(bundle.predict(shots[start:start+1000], explain=True))
        importances = np.abs([[item["value"] for item in prediction["explanation"]["contributions"]] for prediction in test_explanations]).mean(axis=0)
        (args.output / "feature_importance.svg").write_text(_bar_chart([FEATURE_LABELS[f] for f in FEATURE_COLS], importances, "Final-test mean absolute feature contribution", "Raw log-odds; correlated location features share attribution. Not causal importance."), encoding="utf-8")
        selected = bundle.metadata["selected_model"]
        zones = test.groupby("SHOT_ZONE", sort=True).agg(n_shots=("SHOT_MADE_FLAG", "size"), observed_make_rate=("SHOT_MADE_FLAG", "mean"), predicted_make_probability=(selected, "mean")).reset_index()
        zones["prediction_minus_observed"] = zones["predicted_make_probability"] - zones["observed_make_rate"]
        zones.to_csv(args.output / "zone_summary.csv", index=False)
        zone_labels = [f"{row.SHOT_ZONE} (n={row.n_shots})" for row in zones.itertuples()]
        (args.output / "zone_comparison.svg").write_text(_bar_chart(zone_labels, zones["prediction_minus_observed"], "Final-test zone calibration gaps", "Mean predicted probability minus observed make rate; sparse zones are noisy."), encoding="utf-8")
        print(f"Analysis saved to {args.output.resolve()}")
        print(bundle.metadata["evaluation"]["interpretation"])
        print(f"Raw-margin reconstruction error: {explanation['reconstruction_error']:.8f}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
