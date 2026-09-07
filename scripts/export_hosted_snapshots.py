"""Export only public NBA snapshots that match validated local model bundles.

Run explicitly before publishing newly prepared NBA models. Never exports CSV
inputs, credentials, demo data, or arbitrary files from the local cache.
"""
import json
from pathlib import Path
import shutil

from nba_app.config import ROOT
from nba_app.data import NBADataClient
from nba_app.models import ModelStore


def export(root: Path = ROOT) -> list[str]:
    store = ModelStore(root / "models")
    destination = root / "reference_data"
    validated = []
    snapshots = {}
    for path in sorted((root / "models").glob("*/metadata.json")):
        metadata = store.load(path.parent.name).metadata
        if metadata["source"] != "nba" or metadata.get("is_demo"):
            continue
        player = metadata["player_id"]
        shot_key = f"shots-{player}-{metadata['season']}-regular-v1"
        for key in (shot_key, f"career-{player}-regular-v1"):
            source = root / "data" / "cache" / f"{key}.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload["key"] != key or not payload["records"]:
                raise ValueError(f"Invalid or empty NBA snapshot: {key}")
            if any(row.get("PLAYER_ID") != player for row in payload["records"]):
                raise ValueError(f"Wrong player in NBA snapshot: {key}")
            if key == shot_key:
                provenance = NBADataClient._provenance(payload, "saved_snapshot", False)
                if provenance["sha256"] != metadata["data_source"]["sha256"]:
                    raise ValueError(f"Snapshot differs from the evaluated model's source: {key}")
            snapshots[key] = source
        validated.append(metadata["model_id"])
    destination.mkdir(parents=True, exist_ok=True)
    for key, source in snapshots.items():
        shutil.copyfile(source, destination / f"{key}.json")
    print(f"Validated {len(validated)} NBA models; exported {len(snapshots)} public snapshots.")
    return validated


if __name__ == "__main__":
    export()
