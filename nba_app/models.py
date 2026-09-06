"""Immutable, checksummed native XGBoost bundles and shared inference."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading

import numpy as np
import xgboost as xgb

from .court import COURT_VERSION
from .features import FEATURE_COLS, FEATURE_LABELS, FEATURE_VERSION, SHOT_TYPE_ENCODING, feature_frame

logger = logging.getLogger(__name__)
MODEL_ID_RE = re.compile(r"p[0-9]+-[0-9]{4}-[0-9]{2}-[0-9]{8}T[0-9]{12}Z-[a-f0-9]{8}\Z")
UNCERTAINTY_NOTE = "An evaluated statistical uncertainty interval is unavailable. This is a model probability, not a confidence interval."


class ModelNotFoundError(FileNotFoundError):
    """Requested immutable model version does not exist."""


class ModelValidationError(ValueError):
    """Artifact integrity or feature contract is invalid."""


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return np.exp(-np.logaddexp(0, -values))


def _json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelBundle:
    def __init__(self, metadata: dict, booster: xgb.Booster | None):
        self.metadata = copy.deepcopy(metadata)
        self._booster = booster
        self._calibration = copy.deepcopy(metadata["calibration"])
        self._constant_probability = metadata.get("constant_probability")

    def predict(self, shots: list[dict], explain: bool = True) -> list[dict]:
        if not shots:
            raise ValueError("At least one shot is required.")
        if len(shots) > 2500:
            raise ValueError("A prediction batch may contain at most 2500 shots.")
        features, descriptions = feature_frame(shots)
        if self._booster is None:
            probability = float(self._constant_probability)
            raw = np.full(len(features), math.log(probability / (1 - probability)))
            contributions = np.zeros((len(features), len(FEATURE_COLS) + 1))
            contributions[:, -1] = raw
        else:
            matrix = xgb.DMatrix(features, feature_names=FEATURE_COLS)
            raw = self._booster.predict(matrix, output_margin=True).astype(float)
            contributions = self._booster.predict(matrix, pred_contribs=True, approx_contribs=False) if explain else None
        slope, intercept = self._calibration["slope"], self._calibration["intercept"]
        calibrated_logits = slope * raw + intercept
        probabilities = sigmoid(calibrated_logits)
        if not np.isfinite(probabilities).all():
            raise ModelValidationError("Model produced non-finite predictions.")
        if explain and not np.allclose(contributions.sum(axis=1), raw, atol=2e-5, rtol=2e-5):
            logger.error("SHAP additivity failed for model %s", self.metadata["model_id"])
            raise ModelValidationError("Explanation failed its raw-margin reconstruction check; prediction withheld.")
        output = []
        for index, (shot, probability) in enumerate(zip(descriptions, probabilities)):
            explanation = None
            if explain:
                explanation = {
                    "method": "TreeSHAP (XGBoost native, tree-path-dependent)" if self._booster is not None else "Train-only constant baseline",
                    "units": "raw log-odds", "base_value": float(contributions[index, -1]),
                    "raw_margin": float(raw[index]),
                    "contributions": [{"feature": feature, "label": FEATURE_LABELS[feature],
                                       "value": float(contributions[index, col]), "feature_value": float(features.iloc[index, col])}
                                      for col, feature in enumerate(FEATURE_COLS)],
                    "reconstruction_error": float(abs(contributions[index].sum() - raw[index])),
                    "calibration": copy.deepcopy(self._calibration),
                    "calibrated_log_odds": float(calibrated_logits[index]),
                    "reconstructed_probability": float(sigmoid(slope * float(contributions[index].sum()) + intercept)),
                    "note": "The base value plus feature contributions reconstructs the raw model log-odds. Probability = sigmoid(slope × raw log-odds + intercept). Contributions describe this model, are not percentage-point changes, and are not causal advice. Correlated location features can share attribution."
                }
            output.append({**shot, "make_probability": float(probability),
                           "expected_points": float(shot["shot_value"] * probability),
                           "uncertainty": None, "uncertainty_note": UNCERTAINTY_NOTE,
                           "explanation": explanation})
        return output


class ModelStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._cache: dict[str, ModelBundle] = {}
        self._lock = threading.RLock()

    def _path(self, model_id: str) -> Path:
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
            raise ModelValidationError("Invalid model ID.")
        path = (self.root / model_id).resolve()
        if path.parent != self.root:
            raise ModelValidationError("Model path escapes the configured artifact directory.")
        return path

    def _metadata(self, path: Path, model_id: str) -> dict:
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelValidationError("Model metadata is missing or unreadable.") from exc
        if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or metadata.get("model_id") != model_id
                or metadata.get("feature_cols") != FEATURE_COLS
                or metadata.get("feature_version") != FEATURE_VERSION
                or metadata.get("court_version") != COURT_VERSION
                or metadata.get("shot_type_encoding") != SHOT_TYPE_ENCODING):
            raise ModelValidationError("Model metadata does not match the supported feature and court contract.")
        for key, expected_type in {"player_id": int, "player_name": str, "season": str,
                                   "n_shots": int, "created_at": str, "evaluation": dict}.items():
            if not isinstance(metadata.get(key), expected_type):
                raise ModelValidationError(f"Required model metadata field {key} is missing or invalid.")
        calibration = metadata.get("calibration", {})
        if not isinstance(calibration, dict) or not all(isinstance(calibration.get(key), (float, int)) and math.isfinite(calibration[key]) for key in ("slope", "intercept")):
            raise ModelValidationError("Calibration parameters are invalid.")
        if calibration["slope"] <= 0:
            raise ModelValidationError("Calibration slope must preserve the direction of model scores.")
        if metadata.get("model_kind") not in {"constant", "xgboost"}:
            raise ModelValidationError("Unsupported model type.")
        if metadata["model_kind"] == "constant":
            probability = metadata.get("constant_probability")
            if not isinstance(probability, (float, int)) or not 0 < probability < 1:
                raise ModelValidationError("Invalid constant baseline probability.")
        if not isinstance(metadata.get("files"), dict):
            raise ModelValidationError("Artifact checksums are missing or invalid.")
        return metadata

    def list_models(self) -> list[dict]:
        if not self.root.exists():
            return []
        found = []
        for path in self.root.iterdir():
            if not path.is_dir() or not MODEL_ID_RE.fullmatch(path.name):
                continue
            try:
                path = self._path(path.name)
                found.append(self._metadata(path, path.name))
            except ModelValidationError:
                logger.warning("Skipping invalid model bundle %s", path.name)
        return sorted(found, key=lambda item: item["created_at"], reverse=True)

    def _load_path(self, path: Path, model_id: str) -> ModelBundle:
        metadata = self._metadata(path, model_id)
        checksums = metadata.get("files", {})
        if metadata["model_kind"] == "xgboost" and "model.json" not in checksums:
            raise ModelValidationError("Native model checksum is missing.")
        for name, checksum in checksums.items():
            if name not in {"model.json", "test_predictions.csv", "calibration.svg"}:
                raise ModelValidationError("Unexpected artifact filename.")
            artifact = (path / name).resolve()
            if artifact.parent != path.resolve() or not artifact.is_file() or _sha256(artifact) != checksum:
                raise ModelValidationError(f"Artifact integrity check failed for {name}.")
        booster = None
        if metadata["model_kind"] == "xgboost":
            try:
                booster = xgb.Booster(params={"nthread": 2})
                booster.load_model(path / "model.json")
                if booster.feature_names != FEATURE_COLS:
                    raise ModelValidationError("Native model feature order is incompatible.")
            except xgb.core.XGBoostError as exc:
                raise ModelValidationError("Native XGBoost model is unreadable.") from exc
        return ModelBundle(metadata, booster)

    def load(self, model_id: str) -> ModelBundle:
        path = self._path(model_id)
        if not path.is_dir():
            raise ModelNotFoundError("No model with this version exists. Train or select an available model.")
        with self._lock:
            if model_id not in self._cache:
                bundle = self._load_path(path, model_id)
                if len(self._cache) >= 16:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[model_id] = bundle
            return self._cache[model_id]

    def publish(self, metadata: dict, booster: xgb.Booster | None, test_predictions=None,
                calibration_svg: str | None = None) -> dict:
        """Write and validate a sibling staging directory, then rename atomically."""
        model_id = metadata["model_id"]
        destination = self._path(model_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ModelValidationError("Model versions are immutable; this version already exists.")
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            artifact_metadata = copy.deepcopy(metadata)
            artifacts = []
            if booster is not None:
                booster.save_model(stage / "model.json")
                artifacts.append("model.json")
            if test_predictions is not None:
                test_predictions.to_csv(stage / "test_predictions.csv", index=False)
                artifacts.append("test_predictions.csv")
            if calibration_svg is not None:
                (stage / "calibration.svg").write_text(calibration_svg, encoding="utf-8")
                artifacts.append("calibration.svg")
            artifact_metadata["files"] = {name: _sha256(stage / name) for name in artifacts}
            _json_write(stage / "metadata.json", artifact_metadata)
            candidate = self._load_path(stage, model_id)
            candidate.predict([{"loc_x": 0, "loc_y": 100}, {"loc_x": 230, "loc_y": 0}], explain=True)
            # No mutable "current" pointer: existing readers keep their explicit ID.
            os.rename(stage, destination)
            return artifact_metadata
        finally:
            if stage.exists() and stage.resolve().parent == self.root and stage.name.startswith(".staging-"):
                shutil.rmtree(stage)
