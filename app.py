"""EEG Intelligence dashboard (Streamlit).

Run with:  streamlit run app.py

This file only arranges the user interface. All data handling, preprocessing, features, models and
prediction come from the ``src`` package - the same code the command line uses.
Research demo only: nothing here is a diagnosis or a reading of anyone's mental state.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import load_config
from src.data.dataset import load_external_labels, load_gdf_file, load_recording, recording_path
from src.data.validation import DataValidationError
from src.features.spatial import (ImageScaler, azimuthal_equidistant, builder_from_config, get_electrode_xyz,
                                  images_from_epochs)
from src.signal.epoching import build_trial_set
from src.signal.preprocessing import preprocess_recording
from src.signal.quality import recording_quality
from src.training.predict import load_artifact, predict_recording
from src.visualization import signals, spectra, topomaps

st.set_page_config(page_title="EEG Intelligence", page_icon="🧠", layout="wide")
MAX_UPLOAD_MB = 100

cfg = load_config()
ROOT = cfg.resolve_path(".")


# ----------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Reading recording ...")
def get_recording(subject: int, session: str):
    return load_recording(cfg, subject, session)


@st.cache_resource(show_spinner="Epoching ...")
def get_trials(subject: int, session: str):
    rec = get_recording(subject, session)
    labels = load_external_labels(cfg, subject, session)
    return build_trial_set(rec, cfg, labels)


@st.cache_resource
def get_builder():
    return builder_from_config(cfg.dataset["eeg_channels"], cfg)


@st.cache_resource
def get_artifact(path: str):
    return load_artifact(path)


def available_subjects() -> list[tuple[int, str]]:
    return [(s, ses) for s in cfg.dataset["subjects"] for ses in cfg.dataset["sessions"]
            if recording_path(cfg, s, ses).is_file()]


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("EEG Intelligence")
st.sidebar.caption("Explainable motor-imagery EEG analysis inspired by EEGLearn (Bashivan et al., 2016). "
                   "Research demo - not a medical device.")
avail = available_subjects()
if not avail:
    st.error(f"No BCI Competition IV 2a files found in `{cfg.dataset['raw_dir']}`. "
             "See docs/DATASET.md for download instructions. Only the 'Model comparison' tab can be used.")
choice = st.sidebar.selectbox("Recording", avail, format_func=lambda t: f"Subject {t[0]:02d} - session {t[1]}") if avail else None

tabs = st.tabs(["Dataset", "Signals", "Preprocessing", "Spatial EEG", "Model comparison", "Prediction"])

# ----------------------------------------------------------------------------- A. dataset
with tabs[0]:
    st.header("Dataset overview")
    st.markdown("**BCI Competition IV dataset 2a** (Graz): 9 participants, 22 EEG channels, 250 Hz, cued motor imagery "
                "(left hand, right hand, feet, tongue). This project uses **left vs right hand**.")
    if choice:
        try:
            rec, ts = get_recording(*choice), get_trials(*choice)
            c = st.columns(5)
            c[0].metric("Participant / session", f"S{choice[0]:02d} / {choice[1]}")
            c[1].metric("EEG channels", len(rec.ch_names))
            c[2].metric("Sampling rate", f"{rec.sfreq:.0f} Hz")
            c[3].metric("Duration", f"{rec.duration_s / 60:.1f} min")
            c[4].metric("Left/right trials kept", f"{ts.n_trials} / {ts.summary['n_task_trials_before_rejection']}")
            st.write(f"Channels: {', '.join(rec.ch_names)}")
            st.pyplot(signals.plot_class_counts(ts.meta, ts.y, ts.rejections))
            st.write("**Data-validation messages**")
            for w in rec.report.warnings or ["none"]:
                st.info(w)
            if ts.summary["rejection_counts"]:
                st.warning(f"Rejected trials (not silently dropped): {ts.summary['rejection_counts']}")
            st.dataframe(recording_quality(rec), width="stretch")
            st.caption("The 3 EOG channels are not loaded: the data providers state they must not be used for classification.")
        except DataValidationError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------- B. signals
with tabs[1]:
    st.header("Signal exploration")
    if choice:
        rec, ts = get_recording(*choice), get_trials(*choice)
        chans = st.multiselect("Channels", rec.ch_names, default=["C3", "Cz", "C4"], key="sig_ch")
        t0, t1 = st.slider("Time window (s)", 0.0, float(rec.duration_s), (400.0, 420.0), step=1.0, key="sig_t")
        if chans and t1 > t0:
            st.pyplot(signals.plot_traces(rec, chans, t0, t1))
            st.caption("Gaps in the trace are the documented 100-sample missing segments between runs.")
            col1, col2 = st.columns(2)
            col1.pyplot(signals.plot_amplitude_distribution(rec, chans))
            col2.pyplot(spectra.plot_recording_psd(rec.data, rec.sfreq, rec.ch_names, chans))
            ch_psd = st.selectbox("Channel for class-wise PSD", rec.ch_names, index=rec.ch_names.index("C3"))
            st.pyplot(spectra.plot_psd_by_class(ts.X, ts.y, ts.sfreq, ts.ch_names, ch_psd, dict(cfg.features["bands_hz"])))
            st.caption("A spectral difference between classes is descriptive. It does not prove a mental state, and "
                       "amplitudes vary strongly between people and sessions.")

# ----------------------------------------------------------------------------- C. preprocessing
with tabs[2]:
    st.header("Raw vs processed")
    if choice:
        rec = get_recording(*choice)
        pre = cfg.preprocessing
        st.write(f"Processing: Butterworth band-pass **{pre['bandpass_hz'][0]}-{pre['bandpass_hz'][1]} Hz**, order "
                 f"{pre['filter_order']} applied forward+backward (zero phase), re-reference: **{pre['rereference']}**, "
                 "each gap-free segment filtered separately.")
        chans = st.multiselect("Channels", rec.ch_names, default=["C3", "C4"], key="pre_ch")
        t0, t1 = st.slider("Time window (s)", 0.0, float(rec.duration_s), (400.0, 410.0), step=1.0, key="pre_t")
        if chans and t1 > t0:
            proc = preprocess_recording(rec, cfg)
            c1, c2 = st.columns(2)
            c1.pyplot(signals.plot_traces(rec, chans, t0, t1, title="Raw (as recorded; 0.5-100 Hz, 50 Hz notch by amplifier)"))
            c2.pyplot(signals.plot_traces(rec, chans, t0, t1, data=proc, title="Processed"))
            st.caption("Filtering removes slow drift and high-frequency noise but also discards signal outside the band. "
                       "A filtered trace is cleaner-looking, not necessarily more physiologically 'correct'.")

# ----------------------------------------------------------------------------- D. spatial
with tabs[3]:
    st.header("Spatial EEG representation")
    builder = get_builder()
    xy = azimuthal_equidistant(get_electrode_xyz(cfg.dataset["eeg_channels"], cfg.spatial["montage"]))
    c1, c2 = st.columns(2)
    c1.pyplot(topomaps.plot_electrode_layout(xy, cfg.dataset["eeg_channels"], builder))
    c2.markdown(
        f"- Electrode positions come from the **{cfg.spatial['montage']} template** (not measured on the participant).\n"
        f"- 3-D → 2-D: azimuthal equidistant projection (as in EEGLearn).\n"
        f"- Grid: **{builder.grid_size}×{builder.grid_size}** pixels, {cfg.spatial['interpolation']} interpolation; "
        f"{int(builder.mask.sum())} pixels lie inside the electrode hull and are *interpolated*; the rest are blank.\n"
        f"- Image channels = band power in {', '.join(cfg.features['bands_hz'])} (log µV²/Hz)."
    )
    if choice:
        ts = get_trials(*choice)
        imgs = images_from_epochs(ts.X, ts.sfreq, cfg, builder)
        z = ImageScaler().fit_transform(imgs, builder.mask)       # display-only scaling, fitted on this recording
        names = list(cfg.features["bands_hz"])
        st.pyplot(topomaps.plot_class_difference_maps(z, ts.y, builder, names))
        st.caption("Mean right-hand minus left-hand maps for this recording (z-scored for display). Descriptive only; "
                   "no significance test is applied.")
        i = st.number_input("Trial index", 0, ts.n_trials - 1, 0)
        cols = st.columns(len(names))
        for col, k, nm in zip(cols, range(len(names)), names):
            col.pyplot(topomaps.plot_spatial_map(z[int(i), k], builder, f"{nm} - trial {int(i)} ({'right' if ts.y[int(i)] else 'left'})"))

# ----------------------------------------------------------------------------- E. model comparison
with tabs[4]:
    st.header("Model comparison")
    runs = sorted(p for p in (ROOT / cfg.evaluation["outputs_dir"]).glob("*/summary.json"))
    if not runs:
        st.info("No saved experiments yet. Run `python -m src.training.train evaluate --name final_v1 --status final`.")
    else:
        run = st.selectbox("Experiment", runs, format_func=lambda p: p.parent.name)
        summary = json.loads(run.read_text(encoding="utf-8"))
        res = pd.read_csv(run.parent / "results_per_split.csv")
        st.write(f"Status: **{summary['status']}** - created {summary['created_utc']}")
        if summary.get("notes"):
            st.caption(summary["notes"])
        proto_text = {"cross_session": "Train on one participant's session T, test on that participant's session E (different day).",
                      "loso": "Leave-one-subject-out: train on 8 participants, test on the unseen 9th."}
        protocol = st.radio("Protocol", sorted(res["protocol"].unique()), horizontal=True)
        st.info(proto_text[protocol] + " The two protocols answer different questions; do not compare them as one number.")
        g = res[res["protocol"] == protocol]
        agg = pd.DataFrame({m: {"mean balanced acc.": gm["balanced_accuracy"].mean(), "std": gm["balanced_accuracy"].std(),
                                "mean accuracy": gm["accuracy"].mean(), "participants": len(gm)}
                            for m, gm in g.groupby("model")}).T
        st.dataframe(agg.style.format("{:.3f}", subset=["mean balanced acc.", "std", "mean accuracy"]), width="stretch")
        fig = px.bar(g, x="test_subject", y="balanced_accuracy", color="model", barmode="group",
                     labels={"test_subject": "Held-out participant", "balanced_accuracy": "Balanced accuracy"})
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="chance")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        model = st.selectbox("Confusion matrix (pooled over participants)", sorted(g["model"].unique()), index=0)
        cm = np.array(summary["aggregate"][protocol][model]["pooled_confusion_matrix"])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["pred. left", "pred. right"], y=["true left", "true right"],
                                  color_continuous_scale="Blues"), width="stretch")
        st.dataframe(g.drop(columns=["notes"]).round(3), width="stretch")
        st.warning("Each participant contributes ~130-290 test trials, so single-participant accuracies carry wide "
                   "uncertainty (see accuracy_ci columns). Nine participants is a small sample. See docs/LIMITATIONS.md.")

# ----------------------------------------------------------------------------- F. prediction
with tabs[5]:
    st.header("Prediction")
    st.warning("Research demo. Predicts which hand a participant was *cued to imagine moving* in a lab task. "
               "It does not diagnose anything or read a person's thoughts. Scores are not probabilities.")
    arts = sorted(p.parent for p in (ROOT / cfg.evaluation["models_dir"]).glob("*/artifact.json"))
    if not arts:
        st.info("No model artifact found. Create one with `python -m src.training.train export-demo --heldout 1`.")
    else:
        art_dir = st.selectbox("Model artifact", arts, format_func=lambda p: p.name)
        try:
            art = get_artifact(str(art_dir))
            st.caption(f"Trained on participants {art.meta['train_subjects']}; held out: {art.meta['heldout_subject']}. "
                       f"Created {art.meta['created_utc']}.")
            model = st.radio("Model", ["cnn", "csp_lda", "logreg"], horizontal=True)
            src = st.radio("Input", ["Dataset sample", "Upload GDF"], horizontal=True)
            gdf_path, label_path = None, None
            if src == "Dataset sample" and avail:
                pick = st.selectbox("Sample", avail, format_func=lambda t: f"Subject {t[0]:02d} - session {t[1]}", key="pred_pick")
                gdf_path = recording_path(cfg, *pick)
                lp = cfg.resolve_path(cfg.dataset["labels_dir"]) / f"A{pick[0]:02d}{pick[1]}.mat"
                label_path = lp if lp.is_file() else None
                if pick[0] != art.meta["heldout_subject"]:
                    st.info("This participant was part of the model's training data - results are NOT an unseen-participant test.")
            elif src == "Upload GDF":
                up = st.file_uploader("BCI IV 2a-style .gdf file", type=["gdf"])
                up_lab = st.file_uploader("Optional official label file (.mat with 'classlabel')", type=["mat"])
                if up is not None:
                    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                        st.error(f"File larger than {MAX_UPLOAD_MB} MB")
                    else:
                        tmp = Path(tempfile.mkdtemp()) / (Path(up.name).name)
                        tmp.write_bytes(up.getvalue())
                        gdf_path = tmp
                        if up_lab is not None:
                            label_path = tmp.with_suffix(".mat")
                            label_path.write_bytes(up_lab.getvalue())
            if gdf_path is not None and st.button("Run prediction"):
                with st.spinner("Validating, preprocessing and predicting ..."):
                    df, info = predict_recording(art, gdf_path, label_path, model)
                st.success(f"{len(df)} trials predicted with `{model}`")
                for w in info["warnings"]:
                    st.info(w)
                if info["summary"]["rejection_counts"]:
                    st.warning(f"Rejected trials: {info['summary']['rejection_counts']}")
                if not info["labelled"]:
                    st.warning("No labels available: trials of feet/tongue imagery (if present) are still labelled left/right by this binary model.")
                else:
                    known = df[df["true_class"] != "unknown"]
                    st.metric("Accuracy on labelled trials", f"{(known['predicted_class'] == known['true_class']).mean():.3f}", f"n = {len(known)}")
                st.dataframe(df, width="stretch")
        except (DataValidationError, FileNotFoundError, ValueError) as exc:
            st.error(f"Cannot predict: {exc}")
