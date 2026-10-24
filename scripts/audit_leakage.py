"""Run the runtime leakage audit on the real dataset. Usage: python scripts/audit_leakage.py [out_dir_name]

Writes outputs/<name>/leakage_audit.json. Takes several minutes (it re-runs the protocols with spies installed).
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import load_config
from src.evaluation.experiments import load_dataset
from src.evaluation.leakage_audit import audit_protocol


def main(name: str = "audit_v1") -> None:
    cfg = load_config()
    bp, csp = load_dataset(cfg, log=lambda *_: None)
    report = {p: audit_protocol(cfg, bp, csp, p) for p in ("cross_session", "loso")}
    out = Path(cfg.resolve_path(cfg.evaluation["outputs_dir"])) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage_audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for p, r in report.items():
        print(f"{p}: folds={r['n_folds']} duplicates={r['duplicate_epochs_in_dataset']} all_passed={r['all_passed']}")
        for prob in r["problems"]:
            print("  PROBLEM:", prob)


if __name__ == "__main__":
    main(*sys.argv[1:2])
