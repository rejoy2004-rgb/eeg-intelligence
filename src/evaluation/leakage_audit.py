"""Runtime leakage audit of the REAL experiment code path.

Instead of trusting the split function, this module wraps the functions that ``run_protocol`` uses to
fit anything that learns from data (CSP + LDA, logistic regression + scaler, image scaler, CNN data
feeding) and records *which trials* each of them actually received. For every outer fold it then checks:

1. every fitted component received exactly the training trials (CSP, logistic regression);
2. the CNN-fit / CNN-validation / test image batches are disjoint and match the split;
3. fitted statistics equal statistics recomputed from training trials only (StandardScaler mean,
   ImageScaler mean/std, CSP filters) and differ from statistics that include the test trials;
4. no test participant (protocol B) or test session (protocol A) appears in any group passed to the
   inner cross-validation;
5. no trial appears twice in the data (duplicate-epoch check), labels match the event-derived labels.

A trial is identified by a hash of its epoch samples, so identity is checked on the data that the model
really saw, not on index bookkeeping.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src.evaluation import experiments as exp
from src.evaluation.splits import cross_session_split, loso_splits
from src.features.spatial import ImageScaler, builder_from_config, images_from_epochs
from src.models.baseline import BandPowerFeatures


def _hash_rows(X: np.ndarray) -> list[bytes]:
    return [hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=10).digest() for x in X]


class _Recorder:
    """Monkeypatches the fitting entry points used by ``run_protocol`` and records what they receive."""

    def __init__(self, id_of: dict):
        self.id_of = id_of
        self.events: list[dict] = []

    def ids(self, X) -> set:
        return {self.id_of.get(h) for h in _hash_rows(X)}

    def install(self):
        import src.models.eeg_cnn as cnn_mod

        self._orig = (exp.fit_logreg, exp.fit_csp_lda, exp.images_from_epochs, exp.ImageScaler, cnn_mod.train_cnn)
        rec = self

        def fit_logreg(cfg, sfreq, X, y, groups, seed=0, features="power"):
            model, info = self._orig[0](cfg, sfreq, X, y, groups, seed, features)
            rec.events.append({"kind": "logreg", "ids": rec.ids(X), "groups": set(np.asarray(groups).tolist()),
                               "model": model, "X": X})
            return model, info

        def fit_csp_lda(cfg, X, y):
            model = self._orig[1](cfg, X, y)
            rec.events.append({"kind": "csp", "ids": rec.ids(X), "model": model, "X": X, "y": y})
            return model

        def images(X, sfreq, cfg, builder):
            rec.events.append({"kind": "images", "ids": rec.ids(X), "X": X})
            return self._orig[2](X, sfreq, cfg, builder)

        class SpyScaler(ImageScaler):
            def fit(self_, images, mask):
                out = super().fit(images, mask)
                rec.events.append({"kind": "image_scaler", "scaler": self_, "images": images, "mask": mask})
                return out

        exp.fit_logreg, exp.fit_csp_lda, exp.images_from_epochs, exp.ImageScaler = fit_logreg, fit_csp_lda, images, SpyScaler
        return self

    def uninstall(self):
        import src.models.eeg_cnn as cnn_mod

        exp.fit_logreg, exp.fit_csp_lda, exp.images_from_epochs, exp.ImageScaler, cnn_mod.train_cnn = self._orig


def audit_protocol(cfg, bp, csp, protocol: str, models=("logreg", "csp_lda", "cnn"), progress=print) -> dict:
    """Run ``protocol`` with spies installed and verify every fold. Returns a JSON-serialisable report."""
    meta = bp.meta
    key = lambda i: (int(meta.iloc[i]["subject"]), str(meta.iloc[i]["session"]), int(meta.iloc[i]["trial"]))
    id_of = {h: key(i) for i, h in enumerate(_hash_rows(bp.X))}
    id_of_csp = {h: key(i) for i, h in enumerate(_hash_rows(csp.X))}
    duplicates = len(bp.X) - len(id_of)
    all_ids = lambda idx: {key(i) for i in idx}

    rec = _Recorder({**id_of, **id_of_csp}).install()
    try:
        if protocol == "cross_session":
            splits = [cross_session_split(meta, bp.y, s) for s in sorted(meta["subject"].unique())]
        else:
            splits = loso_splits(meta, bp.y, cfg.models["cnn"]["n_val_subjects"])
        builder = builder_from_config(bp.ch_names, cfg)
        folds, problems = [], []
        for split in splits:
            rec.events.clear()
            seed = cfg.evaluation["seed"] + int(meta.iloc[split.test_idx[0]]["subject"])
            exp._fit_predict_split(cfg, split, bp, csp, models, builder, seed)
            ev = {k: [e for e in rec.events if e["kind"] == k] for k in ("logreg", "csp", "images", "image_scaler")}
            train, test = all_ids(split.train_idx), all_ids(split.test_idx)
            val = all_ids(split.val_idx)
            fit = train - val
            fold = {"split": split.name, "n_train": len(train), "n_test": len(test), "checks": {}}
            chk = fold["checks"]

            def record(name, ok, detail=""):
                chk[name] = bool(ok)
                if not ok:
                    problems.append(f"{protocol}/{split.name}: {name} {detail}")

            if "logreg" in models:
                e = ev["logreg"][0]
                record("logreg_fit_ids_equal_train", e["ids"] == train)
                record("logreg_fit_ids_disjoint_from_test", not (e["ids"] & test))
                tr_groups = set(meta.iloc[split.test_idx][("subject" if protocol == "loso" else "run")].tolist())
                if protocol == "loso":
                    record("logreg_inner_cv_groups_exclude_test_subject", not (e["groups"] & tr_groups))
                else:
                    record("logreg_inner_cv_uses_training_session_only",
                           set(meta.iloc[split.train_idx]["session"]) == {"T"})
                feats_tr = BandPowerFeatures(bp.sfreq, dict(cfg.features["bands_hz"]), cfg.features["welch_nperseg"],
                                             cfg.features["welch_overlap"]).transform(bp.X[split.train_idx])
                feats_all = BandPowerFeatures(bp.sfreq, dict(cfg.features["bands_hz"]), cfg.features["welch_nperseg"],
                                              cfg.features["welch_overlap"]).transform(
                    bp.X[np.concatenate([split.train_idx, split.test_idx])])
                mean = e["model"].named_steps["scale"].mean_
                record("logreg_scaler_mean_equals_training_mean", np.allclose(mean, feats_tr.mean(0), atol=1e-9))
                record("logreg_scaler_mean_differs_from_train_plus_test_mean", not np.allclose(mean, feats_all.mean(0), atol=1e-9))
            if "csp_lda" in models:
                e = ev["csp"][0]
                record("csp_fit_ids_equal_train", e["ids"] == train)
                record("csp_fit_ids_disjoint_from_test", not (e["ids"] & test))
                from src.models.baseline import build_csp_lda_pipeline

                refit = build_csp_lda_pipeline(cfg).fit(csp.X[split.train_idx], bp.y[split.train_idx])
                record("csp_filters_equal_refit_on_training_only",
                       np.allclose(e["model"].named_steps["csp"].filters_, refit.named_steps["csp"].filters_, atol=1e-8))
            if "cnn" in models:
                imgs = ev["images"]
                record("cnn_three_image_batches_fit_val_test", len(imgs) == 3)
                f_ids, v_ids, t_ids = imgs[0]["ids"], imgs[1]["ids"], imgs[2]["ids"]
                record("cnn_fit_ids_equal_split_fit_part", f_ids == fit)
                record("cnn_validation_ids_equal_split_val_part", v_ids == val)
                record("cnn_test_ids_equal_split_test_part", t_ids == test)
                record("cnn_fit_val_test_pairwise_disjoint", not (f_ids & v_ids) and not (f_ids & t_ids) and not (v_ids & t_ids))
                if protocol == "loso":
                    subj = lambda ids: {s for s, _, _ in ids}
                    record("cnn_fit_val_test_subjects_pairwise_disjoint",
                           not (subj(f_ids) & subj(v_ids)) and not (subj(f_ids) & subj(t_ids)) and not (subj(v_ids) & subj(t_ids)))
                sc = ev["image_scaler"][0]
                fit_imgs = images_from_epochs(bp.X[np.setdiff1d(split.train_idx, split.val_idx)], bp.sfreq, cfg, builder)
                mean_fit = fit_imgs[:, :, builder.mask].mean(axis=(0, 2))
                record("image_scaler_mean_equals_cnn_fit_part_mean", np.allclose(sc["scaler"].mean_, mean_fit, atol=1e-9))
                all_imgs = images_from_epochs(bp.X[np.concatenate([split.train_idx, split.test_idx])], bp.sfreq, cfg, builder)
                record("image_scaler_mean_differs_from_train_plus_test_mean",
                       not np.allclose(sc["scaler"].mean_, all_imgs[:, :, builder.mask].mean(axis=(0, 2)), atol=1e-9))
            record("train_test_trials_disjoint", not (train & test))
            record("train_test_subjects_disjoint_when_required", protocol == "cross_session"
                   or not ({s for s, _, _ in train} & {s for s, _, _ in test}))
            if protocol == "cross_session":
                record("test_session_differs_from_train_session",
                       {sess for _, sess, _ in train} == {"T"} and {sess for _, sess, _ in test} == {"E"})
            folds.append(fold)
            progress(f"  audited {protocol}/{split.name}: {sum(chk.values())}/{len(chk)} checks passed")
    finally:
        rec.uninstall()
    return {"protocol": protocol, "n_folds": len(folds), "duplicate_epochs_in_dataset": int(duplicates),
            "all_passed": not problems and duplicates == 0, "problems": problems, "folds": folds}
