import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from nba_app.features import FEATURE_COLS, feature_frame, prepare_shots
from nba_app.models import ModelBundle, ModelNotFoundError, ModelStore, ModelValidationError, sigmoid
from nba_app.training import TrainingError, chronological_split, train_bundle
from scripts.generate_fixture import make_fixture


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("model-core")
    store = ModelStore(root)
    raw = make_fixture()
    metadata = train_bundle(raw, 0, "Synthetic demo", "2023-24", store, source="synthetic_fixture")
    return store, metadata, raw


def test_feature_contract_identical_for_training_and_inference():
    raw = make_fixture(games=1)
    prepared, report = prepare_shots(raw)
    infer, descriptions = feature_frame(prepared.rename(columns={"LOC_X": "loc_x", "LOC_Y": "loc_y"})[["loc_x", "loc_y"]].to_dict("records"))
    np.testing.assert_allclose(prepared[FEATURE_COLS].to_numpy(float), infer.to_numpy(float), rtol=1e-12)
    assert list(infer.columns) == FEATURE_COLS
    assert report["dropped_rows"] == 0
    assert set(prepared["SHOT_TYPE_ENC"].unique()) <= {0, 1}


def test_cleaning_reports_invalid_duplicate_and_geometry_records():
    raw = make_fixture(games=2)
    original_size = len(raw)
    duplicate = raw.iloc[[0]].copy()
    conflict = raw.iloc[[1]].copy()
    conflict["SHOT_MADE_FLAG"] = 1 - conflict["SHOT_MADE_FLAG"]
    raw.loc[2, "LOC_X"] = float("nan")
    raw.loc[3, "LOC_Y"] = 600
    raw.loc[4, "SHOT_TYPE"] = "3PT Field Goal" if raw.loc[4, "SHOT_TYPE"] == "2PT Field Goal" else "2PT Field Goal"
    raw.loc[5, "SHOT_TYPE"] = None
    # Deliberately duplicate dataframe index labels as concatenation often does.
    prepared, report = prepare_shots(pd.concat([raw, duplicate, conflict]))
    assert report["duplicate_rows"] == 1
    assert report["conflicting_duplicate_rows"] == 2
    assert report["invalid_rows"] == 1
    assert report["out_of_court_rows"] == 1
    assert report["geometry_conflicts"] == 1
    assert report["derived_shot_types"] == 1
    assert len(prepared) == original_size - 4
    assert not prepared.duplicated(["GAME_ID", "GAME_EVENT_ID"]).any()


def test_csv_numeric_and_string_ids_deduplicate_consistently():
    raw = make_fixture(games=1).iloc[[0]].copy()
    raw["GAME_ID"] = "0022300062"
    raw["GAME_EVENT_ID"] = "4"
    numeric_copy = raw.copy()
    numeric_copy["GAME_ID"] = 22300062.0
    numeric_copy["GAME_EVENT_ID"] = 4.0
    prepared, cleaning = prepare_shots(pd.concat([raw, numeric_copy]))
    assert len(prepared) == 1
    assert cleaning["duplicate_rows"] == 1
    assert prepared.iloc[0]["GAME_ID"] == "0022300062"


def test_empty_missing_small_and_single_class_data(tmp_path):
    store = ModelStore(tmp_path)
    with pytest.raises(TrainingError, match="No shot data"):
        train_bundle(pd.DataFrame(), 0, "Synthetic demo", "2023-24", store)
    with pytest.raises(ValueError, match="Missing required"):
        prepare_shots(pd.DataFrame({"LOC_X": [1]}))
    with pytest.raises(TrainingError, match="at least"):
        train_bundle(make_fixture(games=3), 0, "Synthetic demo", "2023-24", store)
    raw = make_fixture()
    raw["SHOT_MADE_FLAG"] = 1
    with pytest.raises(TrainingError, match="one outcome class"):
        train_bundle(raw, 0, "Synthetic demo", "2023-24", store)
    assert store.list_models() == []


def test_player_and_season_mismatch_rejected(tmp_path):
    store = ModelStore(tmp_path)
    with pytest.raises(TrainingError, match="different or missing player ID"):
        train_bundle(make_fixture(player_id=2), 1, "Wrong", "2023-24", store)
    with pytest.raises(TrainingError, match="outside the requested NBA season"):
        train_bundle(make_fixture(), 0, "Synthetic demo", "2024-25", store)


def test_chronological_games_are_disjoint_and_test_is_latest():
    prepared, _ = prepare_shots(make_fixture())
    splits = chronological_split(prepared)
    names = list(splits)
    for idx, name in enumerate(names):
        for later in names[idx+1:]:
            assert set(splits[name]["GAME_ID"]).isdisjoint(splits[later]["GAME_ID"])
            assert splits[name]["GAME_DATE"].max() < splits[later]["GAME_DATE"].min()
    assert len(splits["train"]) / len(prepared) == 0.6
    assert len(splits["final_test"]) / len(prepared) == 0.2


def test_batch_is_single_consistent_and_uncertainty_unavailable(trained):
    store, metadata, _ = trained
    bundle = store.load(metadata["model_id"])
    shots = [{"loc_x": 0, "loc_y": 50}, {"loc_x": 230, "loc_y": 20}, {"loc_x": -130, "loc_y": 160}]
    batch = bundle.predict(shots)
    for shot, prediction in zip(shots, batch):
        single = bundle.predict([shot])[0]
        assert prediction["make_probability"] == pytest.approx(single["make_probability"], abs=1e-7)
        assert 0 < prediction["make_probability"] < 1
        assert prediction["expected_points"] == prediction["shot_value"] * prediction["make_probability"]
        assert prediction["uncertainty"] is None
    assert bundle.predict(shots, explain=False)[0]["explanation"] is None


def test_xgboost_explanations_reconstruct_raw_and_calibrated_outputs(trained):
    _, metadata, raw = trained
    prepared, _ = prepare_shots(raw)
    estimator = xgb.XGBClassifier(n_estimators=12, max_depth=2, n_jobs=2, random_state=42)
    estimator.fit(prepared[FEATURE_COLS], prepared["SHOT_MADE_FLAG"])
    metadata = copy.deepcopy(metadata)
    metadata["calibration"] = {"method": "sigmoid", "slope": 0.7, "intercept": -0.2}
    metadata["model_kind"] = "xgboost"
    bundle = ModelBundle(metadata, estimator.get_booster())
    for result in bundle.predict([{"loc_x": 0, "loc_y": 50}, {"loc_x": 235, "loc_y": 5}]):
        explanation = result["explanation"]
        total = explanation["base_value"] + sum(item["value"] for item in explanation["contributions"])
        assert total == pytest.approx(explanation["raw_margin"], abs=2e-5)
        assert float(sigmoid(0.7 * total - 0.2)) == pytest.approx(result["make_probability"], abs=1e-6)
        assert explanation["units"] == "raw log-odds"


def test_additivity_failure_is_not_silently_suppressed(trained, monkeypatch):
    _, metadata, raw = trained
    prepared, _ = prepare_shots(raw)
    estimator = xgb.XGBClassifier(n_estimators=3, n_jobs=2)
    estimator.fit(prepared[FEATURE_COLS], prepared["SHOT_MADE_FLAG"])
    booster = estimator.get_booster()
    original = booster.predict
    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("pred_contribs"):
            result[:, 0] += 1
        return result
    monkeypatch.setattr(booster, "predict", broken)
    bundle = ModelBundle(metadata, booster)
    with pytest.raises(ModelValidationError, match="reconstruction"):
        bundle.predict([{"loc_x": 0, "loc_y": 100}])


def test_metadata_and_portable_artifacts(trained):
    store, metadata, _ = trained
    loaded = ModelStore(store.root).load(metadata["model_id"])
    assert loaded.metadata == metadata
    assert metadata["is_demo"] is True
    assert metadata["split"]["train"]["n_shots"] == 864
    assert set(metadata["evaluation"]["final_test"]) >= {"constant_baseline", "original_xgboost", "candidate_xgboost"}
    assert metadata["feature_cols"] == FEATURE_COLS
    assert metadata["dependency_versions"]["xgboost"]
    assert (store.root / metadata["model_id"] / "calibration.svg").is_file()
    assert not list(store.root.rglob("*.joblib"))


def test_missing_invalid_and_tampered_artifacts(trained, tmp_path):
    store, metadata, _ = trained
    empty_store = ModelStore(tmp_path)
    with pytest.raises(ModelNotFoundError):
        empty_store.load(metadata["model_id"])
    for bad in ["../model", "model.json", "p0-2023-24/../../x"]:
        with pytest.raises(ModelValidationError):
            empty_store.load(bad)
    import shutil
    target = tmp_path / metadata["model_id"]
    shutil.copytree(store.root / metadata["model_id"], target)
    (target / "calibration.svg").write_text("changed", encoding="utf-8")
    with pytest.raises(ModelValidationError, match="integrity"):
        empty_store.load(metadata["model_id"])


@pytest.mark.parametrize("malformed", [[], {"schema_version": 1}, "not an object"])
def test_malformed_metadata_is_skipped_and_rejected(trained, tmp_path, malformed):
    _, metadata, _ = trained
    path = tmp_path / metadata["model_id"]
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps(malformed), encoding="utf-8")
    store = ModelStore(tmp_path)
    assert store.list_models() == []
    with pytest.raises(ModelValidationError):
        store.load(metadata["model_id"])


def test_failed_retraining_retains_previous_working_model(trained):
    store, metadata, raw = trained
    expected = store.load(metadata["model_id"]).predict([{"loc_x": 0, "loc_y": 150}])
    raw = raw.copy()
    raw["SHOT_MADE_FLAG"] = 0
    before = [item["model_id"] for item in store.list_models()]
    with pytest.raises(TrainingError):
        train_bundle(raw, 0, "Synthetic demo", "2023-24", store)
    assert [item["model_id"] for item in store.list_models()] == before
    assert store.load(metadata["model_id"]).predict([{"loc_x": 0, "loc_y": 150}]) == expected


def test_atomic_publish_failure_does_not_publish_partial_model(trained, monkeypatch):
    store, metadata, raw = trained
    before = [item["model_id"] for item in store.list_models()]
    def fail(*args, **kwargs):
        raise OSError("simulated disk rename failure")
    monkeypatch.setattr("nba_app.models.os.rename", fail)
    with pytest.raises(OSError, match="simulated"):
        train_bundle(raw, 0, "Synthetic demo", "2023-24", store, source="synthetic_fixture")
    assert [item["model_id"] for item in store.list_models()] == before
    assert not list(store.root.glob(".staging-*"))


def test_final_test_labels_cannot_change_fitted_model_or_selection(trained, tmp_path):
    store, metadata, raw = trained
    raw = raw.copy()
    final_ids = metadata["split"]["final_test"]["game_ids"]
    test_mask = raw["GAME_ID"].isin(final_ids)
    raw.loc[test_mask, "SHOT_MADE_FLAG"] = 1 - raw.loc[test_mask, "SHOT_MADE_FLAG"]
    other_store = ModelStore(tmp_path)
    other = train_bundle(raw, 0, "Synthetic demo", "2023-24", other_store, source="synthetic_fixture")
    assert other["selected_model"] == metadata["selected_model"]
    assert other["calibration"] == metadata["calibration"]
    assert other["evaluation"]["validation"] == metadata["evaluation"]["validation"]
    shots = [{"loc_x": 0, "loc_y": 150}, {"loc_x": -230, "loc_y": 10}]
    assert other_store.load(other["model_id"]).predict(shots) == store.load(metadata["model_id"]).predict(shots)
    assert other["evaluation"]["final_test"] != metadata["evaluation"]["final_test"]


def test_new_version_does_not_change_existing_version(trained):
    store, metadata, raw = trained
    shot = [{"loc_x": 70, "loc_y": 175}]
    before = store.load(metadata["model_id"]).predict(shot)
    raw = raw.copy()
    raw["PLAYER_ID"] = 99
    raw["SHOT_MADE_FLAG"] = 1 - raw["SHOT_MADE_FLAG"]
    other = train_bundle(raw, 99, "Synthetic second player", "2023-24", store, source="synthetic_fixture")
    assert other["model_id"] != metadata["model_id"]
    assert other["player_id"] == 99
    assert store.load(metadata["model_id"]).predict(shot) == before
    assert store.load(other["model_id"]).predict(shot)[0]["make_probability"] != before[0]["make_probability"]


def test_published_ids_are_immutable(trained):
    store, metadata, _ = trained
    with pytest.raises(ModelValidationError, match="immutable"):
        store.publish(metadata, store.load(metadata["model_id"])._booster)
