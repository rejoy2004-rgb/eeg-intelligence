"""Independent label / event / rejection integrity check over ALL 18 files (does not use the project's loader).

Reads GDF annotations directly with MNE and the official .mat label files with SciPy, then verifies:
  * 288 trial starts (768) and 288 cues per file; cue exactly 500 samples after the trial start;
  * T files: event-code labels == .mat labels (all 288 trials); E files: cue code 783 for every trial, .mat gives 72 per class;
  * every 1023 (artifact) marker coincides with a trial start;
  * the number of left/right trials with a 1023 marker equals the number of 'expert artifact' rejections in
    outputs/<run>/rejected_trials.csv, and no other rejection reason occurred.
Usage: python scripts/verify_labels_all.py [run_name] [audit_dir]
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from src.config import load_config

mne.set_log_level("ERROR")


def main(run: str = "final_v1", audit: str = "audit_v1") -> None:
    cfg = load_config()
    raw_dir, lab_dir = cfg.resolve_path(cfg.dataset["raw_dir"]), cfg.resolve_path(cfg.dataset["labels_dir"])
    out = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"]))
    rej = pd.read_csv(out / run / "rejected_trials.csv")
    report, problems = {}, []
    for s in cfg.dataset["subjects"]:
        for sess in ("T", "E"):
            name = f"A{s:02d}{sess}"
            r = mne.io.read_raw_gdf(str(raw_dir / f"{name}.gdf"), preload=False, verbose="ERROR")
            sf = r.info["sfreq"]
            ann = pd.DataFrame({"s": np.round(r.annotations.onset * sf).astype(int), "c": [int(d) for d in r.annotations.description]}).sort_values("s", kind="stable")
            starts = ann[ann.c == 768]["s"].to_numpy()
            cues = ann[ann.c.isin([769, 770, 771, 772, 783])]
            lab = sio.loadmat(str(lab_dir / f"{name}.mat"))["classlabel"].ravel().astype(int)
            art = ann[ann.c == 1023]["s"].to_numpy()
            ok = {"n_starts_288": len(starts) == 288, "n_cues_288": len(cues) == 288, "n_labels_288": len(lab) == 288,
                  "cue_500_after_start": bool(np.all(cues["s"].to_numpy() - starts == 500)),
                  "artifacts_coincide_with_trial_starts": bool(np.isin(art, starts).all()),
                  "labels_in_1_to_4": bool(set(lab) <= {1, 2, 3, 4}), "72_per_class": bool(np.all(np.bincount(lab)[1:] == 72))}
            if sess == "T":
                ok["event_codes_equal_mat_labels"] = bool(np.array_equal(cues["c"].to_numpy() - 768, lab))
            else:
                ok["all_cues_hidden_783"] = bool(np.all(cues["c"].to_numpy() == 783))
            is_lr = np.isin(lab, [1, 2])
            art_trial = np.isin(starts, art)
            n_expected_rej = int((is_lr & art_trial).sum())
            got = rej[(rej.subject == s) & (rej.session == sess)]
            ok["expert_rejections_match_independent_count"] = (len(got) == n_expected_rej and set(got["reason"]) <= {"expert artifact marker (1023)"})
            ok["rejected_trial_ids_match"] = sorted(got["trial"].tolist()) == sorted(np.flatnonzero(is_lr & art_trial).tolist())
            report[name] = {"checks": {k: bool(v) for k, v in ok.items()}, "n_left_right_marked_artifact": n_expected_rej}
            problems += [f"{name}: {k}" for k, v in ok.items() if not v]
    summary = {"files_checked": len(report), "all_checks_passed": not problems, "problems": problems,
               "total_left_right_trials": 18 * 144, "total_expert_rejected_independent": int(sum(v["n_left_right_marked_artifact"] for v in report.values())),
               "total_rejected_in_run": int(len(rej)), "per_file": report}
    d = out / audit
    d.mkdir(parents=True, exist_ok=True)
    (d / "label_integrity.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print({k: v for k, v in summary.items() if k != "per_file"})


if __name__ == "__main__":
    main(*sys.argv[1:3])
