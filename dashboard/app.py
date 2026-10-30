"""Streamlit dashboard for the Beam sentiment pipeline.

Run from the repo root, after training and the pipeline:
    streamlit run dashboard/app.py
"""
import json
import pickle
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  
from src.common import LABELS  # noqa: E402

OUTPUTS = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "sentiment_model.pkl"
REQUIRED = [OUTPUTS / "summary.json", OUTPUTS / "top_words.json", OUTPUTS / "predictions.jsonl", MODEL_PATH]
COLORS = {"positive": "#2e7d32", "negative": "#c62828"}

st.set_page_config(page_title="Beam sentiment pipeline", layout="wide")
st.title("Review sentiment pipeline")
st.caption(
    "UCI Sentiment Labelled Sentences (Amazon, IMDB, Yelp) → Apache Beam pipeline → "
    "RunInference with a TF-IDF + logistic regression model"
)

missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    st.error(
        "Missing: " + ", ".join(missing)
        + ". Run `python -m src.train` and then `python -m src.pipeline` from the repo root first."
    )
    st.stop()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


summary = json.loads((OUTPUTS / "summary.json").read_text())
top_words = json.loads((OUTPUTS / "top_words.json").read_text())
predictions = pd.read_json(OUTPUTS / "predictions.jsonl", lines=True)
predictions["label"] = predictions["label"].map(LABELS)
predictions["predicted"] = predictions["predicted"].map(LABELS)
model = load_model()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sentences processed", f"{summary['records']:,}")
c2.metric("Malformed lines", summary["malformed"])
c3.metric("Duplicate sentences", summary["duplicate_sentences"])
c4.metric("Test accuracy", f"{summary['test_accuracy']:.2%}", help=f"{summary['test_rows']} held-out sentences")

words_tab, sources_tab, preds_tab, try_tab = st.tabs(
    ["Top words", "Sources", "Predictions", "Try the model"]
)

with words_tab:
    st.write("Most frequent words per sentiment after lowercasing and removing stopwords (negations kept).")
    cols = st.columns(2)
    for col, sentiment in zip(cols, ["positive", "negative"]):
        df = pd.DataFrame(top_words.get(sentiment, []), columns=["word", "count"])
        chart = (
            alt.Chart(df)
            .mark_bar(color=COLORS[sentiment])
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("word:N", sort="-x", title=None),
                tooltip=["word", "count"],
            )
        )
        col.subheader(sentiment.capitalize())
        col.altair_chart(chart)

with sources_tab:
    sources = (
        pd.DataFrame.from_dict(summary["sources"], orient="index")
        .rename_axis("source")
        .reset_index()
    )
    sources["test_accuracy"] = sources["source"].map(summary["test_accuracy_by_source"])
    st.dataframe(sources, hide_index=True)
    chart = (
        alt.Chart(sources)
        .mark_bar()
        .encode(
            x=alt.X("source:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("test_accuracy:Q", title="Test accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["source", "test_accuracy"],
        )
    )
    st.altair_chart(chart)

with preds_tab:
    st.write("Confusion matrix on the held-out test split (rows = true label, columns = predicted).")
    st.dataframe(pd.crosstab(predictions["label"], predictions["predicted"]))

    f1, f2 = st.columns(2)
    source = f1.selectbox("Source", ["all"] + sorted(predictions["source"].unique()))
    result = f2.radio("Show", ["all", "correct", "wrong"], horizontal=True)
    view = predictions
    if source != "all":
        view = view[view["source"] == source]
    if result != "all":
        view = view[view["correct"] == (result == "correct")]
    st.dataframe(
        view.sort_values("confidence", ascending=False)[["source", "text", "label", "predicted", "confidence"]],
        hide_index=True,
    )

    hist = (
        alt.Chart(predictions)
        .mark_bar()
        .encode(
            x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Model confidence"),
            y=alt.Y("count():Q", title="Sentences"),
            color=alt.Color("correct:N", title="Correct"),
        )
    )
    st.altair_chart(hist)

with try_tab:
    text = st.text_input("Type a review sentence", "The food was not good at all.")
    if text.strip():
        p_negative, p_positive = model.predict_proba([text])[0]
        st.metric("Predicted sentiment", LABELS[int(p_positive >= 0.5)])
        st.progress(float(p_positive), text=f"P(positive) = {p_positive:.2f}")
