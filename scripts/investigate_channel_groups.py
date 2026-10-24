"""Diagnostic: which scalp region carries the class information? (S08 investigation; exploratory, descriptive)

CSP + LDA (primary settings, 4 components or the number of channels if fewer) is trained and tested with the SAME
splits as the primary analysis, but only on a subset of channels:
    frontal      Fz FC3 FC1 FCz FC2 FC4          (most exposed to ocular artifacts)
    central      C5 C3 C1 Cz C2 C4 C6            (sensorimotor row)
    centroparietal CP3 CP1 CPz CP2 CP4
    sensorimotor central + centro-parietal (12)
    posterior    P1 Pz P2 POz                    (visual / attention alpha lies here, not motor cortex)
These runs are diagnostics; none is proposed as a model and none was used to choose anything.
Usage: python scripts/investigate_channel_groups.py [audit_dir]  -> outputs/<dir>/channel_groups.csv
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.experiments import load_dataset
from src.evaluation.metrics import compute_metrics
from src.models.baseline import build_csp_lda_pipeline, decision_scores

GROUPS = {
    "all 22": None,
    "frontal (6)": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4"],
    "central row (7)": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
    "centro-parietal row (5)": ["CP3", "CP1", "CPz", "CP2", "CP4"],
    "sensorimotor = central + centro-parietal (12)": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4"],
    "posterior (4)": ["P1", "Pz", "P2", "POz"],
}


def main(name: str = "audit_v1") -> None:
    cfg = load_config()
    out = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / name
    bp, csp = load_dataset(cfg, log=lambda *_: None)
    subj, sess, y = bp.meta["subject"].to_numpy(), bp.meta["session"].to_numpy(), bp.y
    rows = []
    for g, chans in GROUPS.items():
        idx = list(range(22)) if chans is None else [bp.ch_names.index(c) for c in chans]
        nc = min(cfg.models["csp_lda"]["n_components"], len(idx))
        for s in sorted(np.unique(subj)):
            m = subj == s
            for proto, tr, te in (("A", m & (sess == "T"), m & (sess == "E")), ("B", ~m, m)):
                mdl = build_csp_lda_pipeline(cfg, n_components=nc).fit(csp.X[tr][:, idx], y[tr])
                sc = decision_scores(mdl, csp.X[te][:, idx])
                rows.append({"group": g, "protocol": proto, "subject": int(s),
                             "balanced_accuracy": compute_metrics(y[te], (sc > 0).astype(int), sc)["balanced_accuracy"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "channel_groups.csv", index=False)
    pv = df.pivot_table(index="group", columns=["protocol", "subject"], values="balanced_accuracy").loc[list(GROUPS)]
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("mean over participants:\n", df.groupby(["group", "protocol"])["balanced_accuracy"].mean().unstack().loc[list(GROUPS)].round(3))
    print("\nS08 only:\n", df[df.subject == 8].pivot(index="group", columns="protocol", values="balanced_accuracy").loc[list(GROUPS)].round(3))
    print("\nParticipants (A / B) by group:")
    for g in GROUPS:
        a = df[(df.group == g) & (df.protocol == "A")].set_index("subject")["balanced_accuracy"]
        b = df[(df.group == g) & (df.protocol == "B")].set_index("subject")["balanced_accuracy"]
        print(f"  {g:48s} A: " + " ".join(f"{v:.2f}" for v in a) + "   B: " + " ".join(f"{v:.2f}" for v in b))


if __name__ == "__main__":
    main(*sys.argv[1:2])
