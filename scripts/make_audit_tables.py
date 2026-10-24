"""Turn the saved audit experiments into markdown tables (so that docs/RESEARCH_AUDIT.md never contains hand-typed numbers).

Usage: python scripts/make_audit_tables.py   -> prints markdown and writes outputs/audit_v2/tables.md
Reads outputs/final_v1 (primary, untouched) and outputs/audit_v2/* (new experiments) and outputs/audit_v1/*.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / "outputs"
V2 = ROOT / "audit_v2"
PROTO = {"cross_session": "A (same participant, unseen session)", "loso": "B (unseen participant, LOSO)"}
lines = []


def w(x=""):
    lines.append(x)


def ms(v):
    v = pd.Series(v).dropna()
    return "n/a" if v.empty else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def load_run(name):
    p = V2 / name / "results_per_split.csv"
    return pd.read_csv(p) if p.exists() else None


prim = pd.read_csv(ROOT / "final_v1" / "results_per_split.csv")


def ba(df, protocol, model):
    g = df[(df.protocol == protocol) & (df.model == model)].sort_values("test_subject")
    return g.set_index("test_subject")["balanced_accuracy"]


def paired(a, b):
    d = (a - b).dropna()
    p = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
    lo, hi = boot_ci(d.to_numpy())
    return d.mean(), lo, hi, p, len(d)


# ------------------------------------------------------------------ 0. reference
w("### Primary results recomputed from `final_v1/results_per_split.csv`")
w("")
w("| Protocol | Model | Balanced accuracy (mean ± SD, n=9) | min | max |")
w("|---|---|---|---|---|")
for p in PROTO:
    for m in ("majority", "logreg", "csp_lda", "cnn"):
        v = ba(prim, p, m)
        w(f"| {PROTO[p]} | {m} | {ms(v)} | {v.min():.3f} | {v.max():.3f} |")
w("")

# ------------------------------------------------------------------ 1. nested
f = V2 / "csp_nested_selection.csv"
if f.exists():
    sel = pd.read_csv(f)
    allc = pd.read_csv(V2 / "csp_nested_all_configs.csv")
    w("### Nested selection of CSP + LDA settings (24 configurations; selection by inner CV only)")
    w("")
    for p in PROTO:
        s = sel[sel.protocol == p].set_index("test_subject").sort_index()
        d, lo, hi, pv, n = paired(s["outer_balanced_accuracy"], s["default_outer_balanced_accuracy"])
        w(f"**Protocol {PROTO[p]}**: nested (selected) {ms(s['outer_balanced_accuracy'])}; a-priori default {ms(s['default_outer_balanced_accuracy'])}; "
          f"paired difference {d:+.3f} (bootstrap 95 % CI over participants [{lo:+.3f}, {hi:+.3f}], exact Wilcoxon p = {pv:.3f}, n = {n}).")
        w("")
        w("| Held-out participant | selected configuration | inner CV BA | outer BA (selected) | outer BA (default) |")
        w("|---|---|---|---|---|")
        for sub, r in s.iterrows():
            w(f"| S{sub:02d} | {r['selected_config']} | {r['inner_balanced_accuracy']:.3f} | {r['outer_balanced_accuracy']:.3f} | {r['default_outer_balanced_accuracy']:.3f} |")
        w("")
        w(f"Selected-configuration counts: {s['selected_config'].value_counts().to_dict()}")
        w("")
        g = allc[allc.protocol == p].groupby("config")["outer_balanced_accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        w(f"*Descriptive only (outer-test scores of ALL configurations; picking the best row would be selection on test data).* Top 5 / bottom 3 of 24 for protocol {p}:")
        w("")
        w("| configuration | mean outer BA | SD |")
        w("|---|---|---|")
        for name, r in pd.concat([g.head(5), g.tail(3)]).iterrows():
            w(f"| {name} | {r['mean']:.3f} | {r['std']:.3f} |")
        w("")
    chk = allc[(allc.config == "c4_b8-30_w0.5-2.5")]
    for p in PROTO:
        a = chk[chk.protocol == p].set_index("test_subject")["outer_balanced_accuracy"].sort_index()
        b = ba(prim, p, "csp_lda")
        w(f"Consistency check: default configuration in the nested run equals final_v1 CSP+LDA for protocol {p}: max |diff| = {np.abs(a - b).max():.2e}")
    w("")

# ------------------------------------------------------------------ 2. ablation
f = V2 / "csp_ablation.csv"
if f.exists():
    ab = pd.read_csv(f)
    w("### Sensitivity of default CSP + LDA to regularisation choices (descriptive)")
    w("")
    w("| Variant | Protocol A | Protocol B |")
    w("|---|---|---|")
    for v, g in ab.groupby("variant", sort=False):
        w(f"| {v} | {ms(g[g.protocol == 'cross_session']['balanced_accuracy'])} | {ms(g[g.protocol == 'loso']['balanced_accuracy'])} |")
    w("")

# ------------------------------------------------------------------ 3. other experiments
def exp_table(title, entries, note=""):
    w(f"### {title}")
    w("")
    if note:
        w(note)
        w("")
    w("| Variant | Protocol A | Protocol B | participants above chance threshold A / B |")
    w("|---|---|---|---|")
    for label, (dfa, dfb, model) in entries.items():
        cells, above = [], []
        for df, p in ((dfa, "cross_session"), (dfb, "loso")):
            if df is None:
                cells.append("not run")
                above.append("-")
                continue
            g = df[(df.protocol == p) & (df.model == model)]
            cells.append(ms(g["balanced_accuracy"]))
            above.append(f"{int((g['accuracy'] > g['chance_threshold_p05']).sum())}/{len(g)}")
        w(f"| {label} | {cells[0]} | {cells[1]} | {above[0]} / {above[1]} |")
    w("")


def both(name):
    return load_run(f"{name}_cross_session"), load_run(f"{name}_loso")


exp_table("Reference rows (final_v1) next to the new variants", {
    "final_v1 majority": (prim, prim, "majority"), "final_v1 band-power LogReg": (prim, prim, "logreg"),
    "final_v1 CSP + LDA": (prim, prim, "csp_lda"), "final_v1 CNN (power images, 16x16)": (prim, prim, "cnn"),
    **{k: (*both(n), m) for k, (n, m) in {
        "CSP + LDA after Euclidean alignment (uses unlabeled target data!)": ("ea_csp_lda", "csp_lda"),
        "LogReg on ERD features (baseline-normalised)": ("erd", "logreg"),
        "CNN on ERD images": ("erd", "cnn"),
        "CNN, measured-only 8x8 layout grid (power)": ("cnn_layoutgrid", "cnn"),
        "CNN small (8-16-32 filters)": ("cnn_small", "cnn"),
        "CNN weight decay 1e-2": ("cnn_wd1e-2", "cnn")}.items()}})

pix = V2 / "pixel_logreg.csv"
if pix.exists():
    px = pd.read_csv(pix)
    w("### Is information lost by the interpolated image representation? (logistic regression on the 3 x 95 in-hull pixel values)")
    w("")
    w("| Model | Protocol A | Protocol B |")
    w("|---|---|---|")
    w(f"| LogReg on 66 band-power features (final_v1) | {ms(ba(prim, 'cross_session', 'logreg'))} | {ms(ba(prim, 'loso', 'logreg'))} |")
    w(f"| LogReg on image pixels (3 bands x 95 pixels) | {ms(px[px.protocol == 'cross_session']['balanced_accuracy'])} | {ms(px[px.protocol == 'loso']['balanced_accuracy'])} |")
    w("")

# Euclidean alignment per participant and pre-cue control
ea_a, ea_b = both("ea_csp_lda")
if ea_a is not None and ea_b is not None:
    w("### CSP + LDA with Euclidean alignment (UNLABELED target data used - not zero-shot) vs the primary CSP + LDA")
    w("")
    w("| Protocol | aligned | primary | paired diff. [bootstrap 95 % CI] | exact Wilcoxon p |")
    w("|---|---|---|---|---|")
    for df, p in ((ea_a, "cross_session"), (ea_b, "loso")):
        a, b = ba(df, p, "csp_lda"), ba(prim, p, "csp_lda")
        d, lo, hi, pv, n = paired(a, b)
        w(f"| {PROTO[p]} | {ms(a)} | {ms(b)} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} |")
    w("")
    w("| Participant | A primary | A aligned | B primary | B aligned |")
    w("|---|---|---|---|---|")
    pa_, pb_, ea_a_, ea_b_ = ba(prim, "cross_session", "csp_lda"), ba(prim, "loso", "csp_lda"), ba(ea_a, "cross_session", "csp_lda"), ba(ea_b, "loso", "csp_lda")
    for sub in pa_.index:
        w(f"| S{sub:02d} | {pa_[sub]:.3f} | {ea_a_[sub]:.3f} | {pb_[sub]:.3f} | {ea_b_[sub]:.3f} |")
    w("")
    w(f"Protocol B mean without S08: primary {pb_.drop(8).mean():.3f}, aligned {ea_b_.drop(8).mean():.3f}.")
    w("")

pre_a, pre_b = both("precue_control")
if pre_a is not None and pre_b is not None:
    w("### Negative control: decoding from the 1 s BEFORE the cue (cue-1.5 ... cue-0.5 s)")
    w("")
    w("| Model | Protocol A | Protocol B | participants above chance threshold A / B |")
    w("|---|---|---|---|")
    for m in ("logreg", "csp_lda"):
        ga, gb = pre_a[pre_a.model == m], pre_b[pre_b.model == m]
        w(f"| {m} | {ms(ga['balanced_accuracy'])} | {ms(gb['balanced_accuracy'])} | "
          f"{int((ga['accuracy'] > ga['chance_threshold_p05']).sum())}/9 / {int((gb['accuracy'] > gb['chance_threshold_p05']).sum())}/9 |")
    w("")
    w("Per participant (balanced accuracy, CSP + LDA, A / B): " + "; ".join(
        f"S{s:02d} {ba(pre_a, 'cross_session', 'csp_lda')[s]:.3f}/{ba(pre_b, 'loso', 'csp_lda')[s]:.3f}" for s in ba(pre_a, 'cross_session', 'csp_lda').index))
    w("")

# negative control
w("### Negative control: training labels permuted (test labels intact)")
w("")
w("| Model | Protocol A | Protocol B |")
w("|---|---|---|")
pa, pb = both("permuted_labels")
for m in ("majority", "logreg", "csp_lda", "cnn"):
    ca = ms(pa[pa.model == m]["balanced_accuracy"]) if pa is not None else "not run"
    cb = ms(pb[pb.model == m]["balanced_accuracy"]) if pb is not None else "not run"
    w(f"| {m} | {ca} | {cb} |")
w("")
if pa is not None and pb is not None:
    allp = pd.concat([pa, pb])
    for m in ("logreg", "csp_lda", "cnn"):
        v = allp[allp.model == m]["balanced_accuracy"]
        w(f"* {m}: across both protocols' {len(v)} permuted folds, mean {v.mean():.3f}, min {v.min():.3f}, max {v.max():.3f} (expected ≈ 0.5 ± sampling noise).")
    w("")

# CNN diagnostics
dg = [pd.read_csv(V2 / f"cnn_diagnostics_{p}.csv") for p in PROTO if (V2 / f"cnn_diagnostics_{p}.csv").exists()]
if dg:
    d = pd.concat(dg)
    w("### CNN diagnostics (same configuration as final_v1)")
    w("")
    w("| Protocol | training BA at selected epoch | validation BA (selected epoch) | test BA | median selected epoch / epochs run |")
    w("|---|---|---|---|---|")
    for p, g in d.groupby("protocol"):
        w(f"| {PROTO[p]} | {ms(g['info_train_balanced_accuracy'])} | {ms(g['info_best_val_balanced_accuracy'])} | {ms(g['balanced_accuracy'])} | "
          f"{g['info_best_epoch'].median():.0f} / {g['info_epochs_run'].median():.0f} |")
    w("")
    f = V2 / "cnn_learning_curves.json"
    if f.exists():
        h = json.loads(f.read_text())
        w("Mean learning curves over folds (first 10 epochs): training loss / validation loss / validation BA")
        w("")
        for p in PROTO:
            hs = [x["history"] for x in h if x["protocol"] == p]
            if not hs:
                continue
            n = min(10, min(len(x["train_loss"]) for x in hs))
            tl = np.mean([x["train_loss"][:n] for x in hs], 0)
            vl = np.mean([x["val_loss"][:n] for x in hs], 0)
            vb = np.mean([x["val_balanced_accuracy"][:n] for x in hs], 0)
            w(f"* {p}: " + "; ".join(f"e{i + 1}: {a:.3f}/{b:.3f}/{c:.3f}" for i, (a, b, c) in enumerate(zip(tl, vl, vb))))
        w("")

# ------------------------------------------------------------------ participants / S08 / ocular
A1 = ROOT / "audit_v1"
if (A1 / "participants.csv").exists():
    pt = pd.read_csv(A1 / "participants.csv")
    w("### Per-participant comparison (final_v1 results + descriptive data characteristics)")
    w("")
    w("| S | retained (L/R) | rejected % | median epoch p-p (uV) | median raw RMS (uV) | mu lateralisation d (right vs left) | A: CSP / LogReg / CNN | B: CSP / LogReg / CNN |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in pt.iterrows():
        w(f"| {int(r.subject):02d} | {int(r.n_retained)} ({int(r.n_left)}/{int(r.n_right)}) | {100 * r.rejection_rate:.1f} | {r.median_epoch_ptp_uV:.1f} | "
          f"{r.median_channel_rms_uV_raw:.1f} | {r.mu_lateralisation_d:+.2f} | {r.A_csp_lda:.3f} / {r.A_logreg:.3f} / {r.A_cnn:.3f} | "
          f"{r.B_csp_lda:.3f} / {r.B_logreg:.3f} / {r.B_cnn:.3f} |")
    w("")
    w("Mu lateralisation d = Cohen's d between right- and left-hand trials of log mu power at C4 minus C3 (positive = the expected contralateral desynchronisation). Flat channels: none in any participant.")
    w("")
if (A1 / "s08_investigation.json").exists():
    j = json.loads((A1 / "s08_investigation.json").read_text())
    w("### Exploratory Spearman correlations across the 9 participants (uncorrected; descriptive)")
    w("")
    w("| characteristic | rho vs B CSP+LDA | rho vs A CSP+LDA |")
    w("|---|---|---|")
    for k in ("rejection_rate", "mu_lateralisation_d", "median_channel_rms_uV_raw", "median_epoch_ptp_uV", "n_retained", "loso_csp_pattern_similarity_to_pooled"):
        b_, a_ = j["spearman_across_participants"][f"{k}_vs_B_csp_lda"], j["spearman_across_participants"][f"{k}_vs_A_csp_lda"]
        w(f"| {k} | {b_['rho']:+.2f} (p = {b_['p_uncorrected']:.2f}) | {a_['rho']:+.2f} (p = {a_['p_uncorrected']:.2f}) |")
    r_ = j["spearman_across_participants"]["A_csp_lda_vs_B_csp_lda"]
    w(f"| A CSP+LDA vs B CSP+LDA | {r_['rho']:+.2f} (p = {r_['p_uncorrected']:.3f}) | - |")
    w("")
    jk = j["jackknife_over_training_participants_balanced_accuracy"]
    vals = [v for k, v in jk.items() if k != "all" and v == v]
    w("### S08 stability")
    w("")
    w(f"* LOSO CSP+LDA confusion matrix (rows true left/right; columns predicted): {j['confusion_matrix_loso_csp_lda']}; per session: {{k: round(v, 3) for k, v in j['per_session_balanced_accuracy_loso_csp_lda'].items()}}".replace("{k: round(v, 3) for k, v in j['per_session_balanced_accuracy_loso_csp_lda'].items()}", str({k: round(v, 3) for k, v in j['per_session_balanced_accuracy_loso_csp_lda'].items()})))
    w(f"* Bootstrap over trials (trials treated as independent, so too narrow): 95 % CI {[round(x, 3) for x in j['bootstrap_over_trials_95ci_balanced_accuracy_(trials_treated_as_independent)']]}")
    w(f"* Dropping each single training participant: balanced accuracy range {min(vals):.3f} - {max(vals):.3f} (all training participants: {jk['all']:.3f})")
    w(f"* Train on the other participants' T sessions only, test on S08's E session: {j['train_other_T_only__test_S08_E_only']:.3f}")
    w(f"* Protocol A (own T -> own E): {j['within_participant_protocol_A_csp_lda']:.3f}")
    w("")
if (A1 / "influence_matrix.csv").exists():
    im = pd.read_csv(A1 / "influence_matrix.csv", index_col=0)
    w("### Influence of single training participants on each held-out participant (CSP+LDA, LOSO)")
    w("")
    w("| Held-out | all 8 training | min after dropping one | max after dropping one |")
    w("|---|---|---|---|")
    for sub in im.index:
        row = im.loc[sub].drop("all").dropna()
        w(f"| S{int(sub):02d} | {im.loc[sub, 'all']:.3f} | {row.min():.3f} | {row.max():.3f} |")
    w("")
if (A1 / "ocular_confound.csv").exists():
    oc = pd.read_csv(A1 / "ocular_confound.csv")
    w("### Ocular-confound diagnostic (EOG used ONLY for diagnosis, never as model input)")
    w("")
    w("| S | horizontal-EOG shift d (right vs left) | EOG-only BA A-like / LOSO | CSP+LDA without frontal channels A / B | primary CSP+LDA A / B | own CSP pattern 1: strongest channels |")
    w("|---|---|---|---|---|---|")
    for _, r in oc.iterrows():
        w(f"| {int(r.subject):02d} | {r.horizontal_eog_shift_d_right_vs_left:+.2f} | {r.eog_only_A_like_balanced_accuracy:.3f} / {r.eog_only_LOSO_balanced_accuracy:.3f} | "
          f"{r.csp_no_frontal_A_balanced_accuracy:.3f} / {r.csp_no_frontal_LOSO_balanced_accuracy:.3f} | {r.primary_csp_A:.3f} / {r.primary_csp_B:.3f} | {r.own_csp_pattern0_top_channels} |")
    w("")
    w(f"Means without frontal channels (Fz, FC1-4): A {oc.csp_no_frontal_A_balanced_accuracy.mean():.3f} vs primary {oc.primary_csp_A.mean():.3f}; "
      f"B {oc.csp_no_frontal_LOSO_balanced_accuracy.mean():.3f} vs primary {oc.primary_csp_B.mean():.3f} (descriptive diagnostic, not a selected model).")
    w("")

if (A1 / "channel_groups.csv").exists():
    cg = pd.read_csv(A1 / "channel_groups.csv")
    order = list(dict.fromkeys(cg["group"]))
    w("### Which scalp region carries the information? CSP + LDA restricted to channel groups (diagnostic only)")
    w("")
    w("| channel group | all participants A / B (mean BA) | S08 A / B | participants other than S08 A / B |")
    w("|---|---|---|---|")
    for g in order:
        x = cg[cg.group == g]
        m = lambda d, p: d[d.protocol == p]["balanced_accuracy"].mean()
        s8, ot = x[x.subject == 8], x[x.subject != 8]
        w(f"| {g} | {m(x, 'A'):.3f} / {m(x, 'B'):.3f} | {m(s8, 'A'):.3f} / {m(s8, 'B'):.3f} | {m(ot, 'A'):.3f} / {m(ot, 'B'):.3f} |")
    w("")

# ------------------------------------------------------------------ 4. paired statistics on the primary comparisons
w("### Paired participant-level comparisons on the primary results (n = 9; Holm-corrected within protocol)")
w("")
w("| Protocol | Comparison | mean diff. BA | bootstrap 95 % CI | exact Wilcoxon p | Holm-adjusted p |")
w("|---|---|---|---|---|---|")
for p in PROTO:
    pairs = [("csp_lda", "logreg"), ("csp_lda", "cnn"), ("logreg", "cnn")]
    res = [paired(ba(prim, p, a), ba(prim, p, b)) for a, b in pairs]
    adj = holm([r[3] for r in res])
    for (a, b), r, q in zip(pairs, res, adj):
        w(f"| {PROTO[p]} | {a} − {b} | {r[0]:+.3f} | [{r[1]:+.3f}, {r[2]:+.3f}] | {r[3]:.3f} | {q:.3f} |")
w("")
w("Wilcoxon exact p-values with n = 9 cannot go below 0.0039; the bootstrap interval resamples participants (not trials) and is itself coarse at n = 9.")
w("")

text = "\n".join(lines)
V2.mkdir(parents=True, exist_ok=True)
(V2 / "tables.md").write_text(text, encoding="utf-8")
if "--embed" in sys.argv:
    doc = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_AUDIT.md"
    d = doc.read_text(encoding="utf-8")
    a, b = "<!-- BEGIN TABLES -->", "<!-- END TABLES -->"
    assert a in d and b in d, "markers missing in docs/RESEARCH_AUDIT.md"
    doc.write_text(d[: d.index(a) + len(a)] + "\n\n" + text + "\n" + d[d.index(b):], encoding="utf-8")
    print("embedded tables into docs/RESEARCH_AUDIT.md")
else:
    print(text)
