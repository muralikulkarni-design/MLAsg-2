"""
Streamlit front end for ML Assignment 2.

Loads the five pre-trained pipelines from model/ and scores whatever test csv
the user uploads. Nothing is retrained here - the app only calls predict().

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

# one colour per outcome, reused in every chart so the pages feel consistent
CLASS_COLOURS = {"Dropout": "#d1495b", "Enrolled": "#edae49", "Graduate": "#00798c"}

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "model_info.json") as f:
        info = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in info["models"].items()}
    return info, models


def read_csv(uploaded):
    # sep=None lets pandas sniff the delimiter, so both our comma separated
    # test_data.csv and the original semicolon UCI file can be uploaded
    df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def score(model, X, y, classes):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    # AUC needs every class present in y, a tiny uploaded file may not have that
    try:
        metrics["AUC"] = roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                       labels=classes)
    except ValueError:
        metrics["AUC"] = float("nan")
    return y_pred, proba, metrics


def plot_confusion(y, y_pred, classes, title):
    cm = confusion_matrix(y, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"fontsize": 12})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


info, models = load_artifacts()
classes = info["classes"]
target = info["target"]
features = info["features"]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type=["csv"],
                                help=f"Needs all {len(features)} feature columns and the '{target}' column.")
    if uploaded is None:
        st.caption("No file uploaded - the bundled test_data.csv is used.")
    st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()),
                              index=list(models.keys()).index("Support Vector Machine"))
    st.caption("Tuned hyper-parameters: " +
               ", ".join(f"{k}={v}" for k, v in info["best_params"][model_name].items()))

    st.divider()
    st.caption("Models were trained offline with `train_models.py` "
               "(80/20 stratified split, 5-fold grid search). "
               "This app only loads them and predicts.")

# ---------------- header ----------------
st.title("🎓 Student Outcome Classifier")
st.markdown(
    "Predicts whether a higher-education student will **drop out**, stay **enrolled** "
    "beyond the normal duration, or **graduate**, using enrolment-time information plus "
    "1st/2nd semester academic results. Dataset: *Predict Students' Dropout and Academic "
    "Success* (UCI, 4,424 students, 36 features)."
)

if uploaded is not None:
    try:
        data = read_csv(uploaded)
        source = uploaded.name
    except Exception as e:  # bad encoding, empty file etc.
        st.error(f"Could not read that file: {e}")
        st.stop()
else:
    data = read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"

missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s):")
    st.code("\n".join(missing))
    st.stop()

data = data.dropna(subset=[target])
unknown = sorted(set(data[target]) - set(classes))
if unknown:
    st.error(f"Unexpected values in '{target}': {unknown}. Expected one of {classes}.")
    st.stop()

X, y = data[features], data[target]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(data):,}")
c2.metric("Dropout", int((y == "Dropout").sum()))
c3.metric("Enrolled", int((y == "Enrolled").sum()))
c4.metric("Graduate", int((y == "Graduate").sum()))
st.caption(f"Scoring file: **{source}**")

tab_eval, tab_compare, tab_pred, tab_data = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔎 Predictions", "📄 Data preview"])

# ---------------- single model ----------------
with tab_eval:
    y_pred, proba, m = score(models[model_name], X, y, classes)
    st.subheader(model_name)

    # two rows of three - six metrics in one row get truncated on small screens
    cols = st.columns(3) + st.columns(3)
    for col, key in zip(cols, METRIC_ORDER):
        col.metric(key, "n/a" if pd.isna(m[key]) else f"{m[key]:.4f}", border=True)
    st.caption("Precision, Recall, F1 and AUC (one-vs-rest) are macro averaged over the 3 classes.")

    left, right = st.columns([1, 1.2])
    with left:
        st.pyplot(plot_confusion(y, y_pred, classes, f"Confusion matrix - {model_name}"))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=classes, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.round(3)
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df, width="stretch")

        # per class recall is the number that matters most for an early warning tool
        recall_drop = report["Dropout"]["recall"]
        st.info(f"This model catches **{recall_drop:.0%}** of the actual dropouts in this file.")

# ---------------- all models ----------------
with tab_compare:
    st.subheader("All five models on the same file")
    rows = {}
    for name, mdl in models.items():
        rows[name] = score(mdl, X, y, classes)[2]
    comp = pd.DataFrame(rows).T[METRIC_ORDER]
    comp.index.name = "ML Model Name"

    # st.table instead of st.dataframe: the interactive grid collapsed to one
    # column inside a tab when given a Styler
    st.table(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cdeccd"))
    best = comp["F1"].idxmax()
    st.success(f"Best macro-F1 on this file: **{best}** ({comp.loc[best, 'F1']:.4f})")

    metric_to_plot = st.radio("Metric to plot", METRIC_ORDER, horizontal=True, index=4)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    order = comp[metric_to_plot].sort_values()
    ax.barh(order.index, order.values,
            color=["#00798c" if n == best else "#9fb8bf" for n in order.index])
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_to_plot)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Confusion matrices for every model"):
        grid = st.columns(3)
        for i, (name, mdl) in enumerate(models.items()):
            with grid[i % 3]:
                st.pyplot(plot_confusion(y, mdl.predict(X), classes, name))

# ---------------- row level predictions ----------------
with tab_pred:
    out = data[[target]].rename(columns={target: "Actual"}).copy()
    out["Predicted"] = y_pred
    for i, cls in enumerate(models[model_name].classes_):
        out[f"P({cls})"] = proba[:, i].round(3)
    out["Correct"] = out["Actual"] == out["Predicted"]

    only_wrong = st.toggle("Show only misclassified rows")
    view = out[~out["Correct"]] if only_wrong else out
    st.write(f"{len(view)} rows · model: {model_name}")
    st.dataframe(view, width="stretch", height=420)
    st.download_button("Download predictions as CSV", view.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv")

# ---------------- raw data ----------------
with tab_data:
    st.dataframe(data.head(50), width="stretch")
    fig, ax = plt.subplots(figsize=(5, 2.4))
    counts = y.value_counts().reindex(classes)
    ax.bar(counts.index, counts.values, color=[CLASS_COLOURS[c] for c in counts.index])
    ax.set_title("Class distribution in the uploaded file", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="content")
