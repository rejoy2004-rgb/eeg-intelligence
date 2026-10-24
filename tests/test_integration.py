"""End-to-end workflow on SYNTHETIC recordings (software test only - not an EEG experiment).

load/validate -> preprocess -> epoch -> features -> spatial images -> classical models -> CNN
-> evaluation protocols -> save/load -> prediction
"""
import json

import numpy as np
import pytest

from src.data.validation import DataValidationError
from src.evaluation.experiments import aggregate, run_protocol, save_experiment
from src.signal.epoching import TrialSet, build_dual_band_trials
from src.training import predict as predict_mod
from src.training.predict import load_artifact, predict_recording, predict_trials
from src.training.train import export_artifact


@pytest.fixture
def small_cfg(cfg):
    cfg["models"]["cnn"].update(max_epochs=3, patience=3, batch_size=16, n_val_subjects=1)
    return cfg


@pytest.fixture
def dataset(small_cfg, make_recording):
    bp_sets, csp_sets = [], []
    for subject in (1, 2, 3):
        for session in ("T", "E"):
            rec, label4 = make_recording(subject, session, n_trials=32, n_runs=2)
            bp, csp = build_dual_band_trials(rec, small_cfg, label4)
            bp_sets.append(bp)
            csp_sets.append(csp)
    return TrialSet.concat(bp_sets), TrialSet.concat(csp_sets)


def test_dataset_is_aligned(dataset):
    bp, csp = dataset
    assert bp.n_trials == 96 and sorted(bp.meta["subject"].unique()) == [1, 2, 3]
    assert bp.meta[["subject", "session", "trial"]].equals(csp.meta[["subject", "session", "trial"]])
    assert np.array_equal(bp.y, csp.y)


@pytest.mark.parametrize("protocol", ["cross_session", "loso"])
def test_protocol_runs_and_records_everything(dataset, small_cfg, protocol, tmp_path):
    bp, csp = dataset
    res, splits, preds = run_protocol(small_cfg, bp, csp, protocol, log=lambda *_: None)
    assert set(res["model"]) == {"majority", "logreg", "csp_lda", "cnn"}
    assert len(res) == 3 * 4 and res["accuracy"].between(0, 1).all()
    assert len(preds) == int(res["n_test"].sum())          # one prediction row per test trial and model
    # the planted (synthetic) lateralised rhythm is easy for the classical models; majority stays near chance
    assert res.query("model == 'csp_lda'")["accuracy"].mean() > 0.75
    assert res.query("model == 'majority'")["accuracy"].mean() <= 0.6
    assert len(splits) == 3
    if protocol == "loso":
        for s in splits:
            test_sub, train_sub = set(s["test"]["subjects"]), set(s["train_all"]["subjects"])
            assert test_sub.isdisjoint(train_sub)                           # no subject leakage
    agg = aggregate(res)
    assert protocol in agg and "cnn" in agg[protocol]
    out = save_experiment(small_cfg, res, splits, preds, bp, "unit_test_run", "exploratory", out_root=tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "exploratory" and summary["software"]["python"] and summary["config"]["evaluation"]["seed"] == 42
    for name in ("results_per_split.csv", "predictions.csv", "rejected_trials.csv", "splits.json"):
        assert (out / name).is_file()


def test_misaligned_sets_are_refused(dataset, small_cfg):
    bp, csp = dataset
    shuffled = csp.select(np.arange(csp.n_trials)[::-1])
    with pytest.raises(ValueError, match="not aligned"):
        run_protocol(small_cfg, bp, shuffled, "loso", log=lambda *_: None)
    with pytest.raises(ValueError, match="protocol"):
        run_protocol(small_cfg, bp, csp, "random_kfold", log=lambda *_: None)


def test_artifact_export_load_and_predict(dataset, small_cfg, tmp_path, make_recording, monkeypatch):
    bp, csp = dataset
    out = export_artifact(small_cfg, bp, csp, tmp_path / "art", heldout_subject=1)
    meta = json.loads((out / "artifact.json").read_text())
    assert meta["heldout_subject"] == 1 and 1 not in meta["train_subjects"]
    assert set(meta["file_sha256"]) == {"logreg.joblib", "csp_lda.joblib", "cnn.pt", "cnn_meta.json"}

    art = load_artifact(out)
    held = bp.meta["subject"].to_numpy() == 1
    bp1, csp1 = bp.select(held), csp.select(held)
    for model in ("logreg", "csp_lda", "cnn"):
        df = predict_trials(art, bp1, csp1, model)
        assert len(df) == bp1.n_trials
        assert set(df["predicted_class"]) <= {"left hand", "right hand"}
        assert np.isfinite(df["score_right_minus_left"]).all()
    # classical models, trained WITHOUT subject 1, still classify its synthetic trials
    df = predict_trials(art, bp1, csp1, "csp_lda")
    assert (df["predicted_class"] == df["true_class"]).mean() > 0.75

    # whole-recording path (file reading is replaced by an in-memory synthetic recording)
    rec, label4 = make_recording(1, "E", n_trials=32, n_runs=2)
    monkeypatch.setattr(predict_mod, "load_gdf_file", lambda path, cfg: rec)
    lab_path = tmp_path / "A01E.mat"
    import scipy.io as sio

    sio.savemat(lab_path, {"classlabel": label4.reshape(-1, 1)})
    df2, info = predict_recording(art, "ignored.gdf", lab_path, "csp_lda")
    assert len(df2) == 16 and info["labelled"] and info["summary"]["n_other_class"] == 16
    # unlabeled recordings: no ground truth is claimed
    raw_unl, _ = make_recording(1, "E", n_trials=32, n_runs=2, unknown_cue=True)
    monkeypatch.setattr(predict_mod, "load_gdf_file", lambda path, cfg: raw_unl)
    df3, info3 = predict_recording(art, "ignored.gdf", None, "logreg")
    assert (df3["true_class"] == "unknown").all() and not info3["labelled"] and len(df3) == 32


def test_artifact_integrity_and_input_checks(dataset, small_cfg, tmp_path):
    bp, csp = dataset
    out = export_artifact(small_cfg, bp, csp, tmp_path / "art", heldout_subject=None)
    art = load_artifact(out)
    with pytest.raises(ValueError, match="model must be"):
        predict_trials(art, bp, csp, "svm")
    with pytest.raises(DataValidationError, match="No usable trials"):
        predict_trials(art, bp.select(np.zeros(bp.n_trials, dtype=bool)), csp.select(np.zeros(bp.n_trials, dtype=bool)))
    wrong = TrialSet(bp.X, bp.y, bp.meta, bp.rejections, 128.0, bp.ch_names)
    with pytest.raises(DataValidationError, match="do not match"):
        predict_trials(art, wrong, csp)
    (out / "logreg.joblib").write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="Hash mismatch"):
        load_artifact(out)
    with pytest.raises(DataValidationError, match="No artifact.json"):
        load_artifact(tmp_path / "nowhere")
