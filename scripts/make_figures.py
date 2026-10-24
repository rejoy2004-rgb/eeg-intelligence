"""Create the figures used in the docs from the real data (needs data/raw). Usage: python scripts/make_figures.py"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import load_external_labels, load_recording
from src.features.spatial import ImageScaler, builder_from_config, images_from_epochs
from src.signal.epoching import build_trial_set
from src.visualization import signals, spectra, topomaps

OUT = PROJECT_ROOT / "docs" / "figures"


def main(subject: int = 3) -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    rec = load_recording(cfg, subject, "T")
    ts = build_trial_set(rec, cfg, load_external_labels(cfg, subject, "T"))
    builder = builder_from_config(ts.ch_names, cfg)
    topomaps.plot_electrode_layout(builder.xy, ts.ch_names, builder).savefig(OUT / "electrode_layout.png", dpi=110)
    z = ImageScaler().fit_transform(images_from_epochs(ts.X, ts.sfreq, cfg, builder), builder.mask)
    topomaps.plot_class_difference_maps(z, ts.y, builder, list(cfg.features["bands_hz"])).savefig(
        OUT / f"class_difference_S{subject:02d}.png", dpi=110)
    spectra.plot_psd_by_class(ts.X, ts.y, ts.sfreq, ts.ch_names, "C3", dict(cfg.features["bands_hz"])).savefig(
        OUT / f"psd_S{subject:02d}_C3.png", dpi=110)
    signals.plot_traces(rec, ["C3", "Cz", "C4"], 400, 420).savefig(OUT / f"traces_S{subject:02d}.png", dpi=110)


if __name__ == "__main__":
    main()
