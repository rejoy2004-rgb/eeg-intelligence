"""Plausibility check of the positional channel names (the GDF files only name 5 of the 22 EEG channels).

If the assumed 10-20 order is right, EEG channels that are close on the scalp should be strongly
correlated and distant ones weakly correlated. We compare the Spearman correlation between
inter-electrode distance and inter-channel signal correlation for the assumed order against random
relabellings of the 17 unnamed channels.

Limits: this supports the layout *as a whole*; it cannot prove that two neighbouring unnamed channels
are not swapped. Usage:  python scripts/verify_channel_layout.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from src.config import load_config
from src.data.dataset import load_recording, recording_path
from src.features.spatial import get_electrode_xyz
from src.signal.preprocessing import preprocess_recording


def main(n_perm: int = 300) -> None:
    cfg = load_config()
    names = cfg.dataset["eeg_channels"]
    xyz = get_electrode_xyz(names, cfg.spatial["montage"])
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    iu = np.triu_indices(len(names), 1)
    anchors = {int(k) for k in cfg.dataset["anchor_channels"]}
    free = [i for i in range(len(names)) if i not in anchors]
    rng = np.random.default_rng(0)
    for s in cfg.dataset["subjects"]:
        if not recording_path(cfg, s, "T").is_file():
            continue
        x = preprocess_recording(load_recording(cfg, s, "T"), cfg)
        x = x[:, np.isfinite(x).all(0)]
        corr = np.corrcoef(x)
        observed = spearmanr(dist[iu], corr[iu])[0]
        perms = []
        for _ in range(n_perm):
            p = np.arange(len(names))
            p[free] = rng.permutation(free)
            perms.append(spearmanr(dist[np.ix_(p, p)][iu], corr[iu])[0])
        print(f"S{s:02d}: assumed order {observed:+.3f} | random relabelling mean {np.mean(perms):+.3f}, "
              f"best {np.min(perms):+.3f} | fraction of relabellings at least as good: {np.mean(np.array(perms) <= observed):.3f}")


if __name__ == "__main__":
    main()
