"""Tests for the helpers, the Beam transforms, and the full pipeline.

Run from the repo root:  python -m pytest -q
"""
import json
import pickle

import apache_beam as beam
import pytest
from apache_beam.ml.inference.base import KeyedModelHandler, RunInference
from apache_beam.ml.inference.sklearn_inference import SklearnModelHandlerNumpy
from apache_beam.testing.test_pipeline import TestPipeline as BeamTestPipeline
from apache_beam.testing.util import assert_that, equal_to
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.common import is_test, parse_line, tokenize
from src.pipeline import run
from src.transforms import CleanWords, ParseRecord, SourceStatsFn, predict_proba, to_prediction_row


def tiny_model(texts, labels):
    """Same structure as the real model, trained on a few test sentences."""
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(texts, labels)
    return model



def test_parse_line_repairs_ellipsis_and_strips_spaces():
    assert parse_line("Worth seeing\x85 really.  \t1") == ("Worth seeing... really.", 1)


@pytest.mark.parametrize("line", ["no tab here", "\t1", "Some text\t7"])
def test_parse_line_rejects_malformed_lines(line):
    with pytest.raises(ValueError):
        parse_line(line)


def test_tokenize_lowercases_and_strips_quotes():
    assert tokenize("I didn't LOVE it... 'Not' great!") == ["i", "didn't", "love", "it", "not", "great"]


def test_split_sends_about_20_percent_to_test():
    share = sum(is_test(f"sentence {i}") for i in range(10_000)) / 10_000
    assert 0.18 < share < 0.22



def test_parse_record_routes_bad_lines_to_malformed():
    lines = [
        ("data/raw/yelp_labelled.txt", "Great food.  \t1"),
        ("data/raw/imdb_labelled.txt", "no tab here"),
        ("data/raw/imdb_labelled.txt", "Bad label\t7"),
    ]
    expected_record = {
        "source": "yelp",
        "text": "Great food.",
        "label": 1,
        "split": "test" if is_test("Great food.") else "train",
    }
    with BeamTestPipeline() as p:
        parsed = (
            p
            | beam.Create(lines)
            | beam.ParDo(ParseRecord()).with_outputs(ParseRecord.MALFORMED, main="records")
        )
        errors = parsed[ParseRecord.MALFORMED] | beam.Map(lambda m: m["error"])
        assert_that(parsed.records, equal_to([expected_record]), label="Check records")
        assert_that(
            errors,
            equal_to(["missing tab separator", "label must be 0 or 1, got '7'"]),
            label="Check malformed",
        )


def test_clean_words_drops_stopwords_and_short_words():
    with BeamTestPipeline() as p:
        stopwords = p | "Stopwords" >> beam.Create([("the", True), ("was", True)])
        records = p | "Records" >> beam.Create([{"label": 0, "text": "The food was NOT good, a b"}])
        words = records | CleanWords(stopwords, min_len=2)
        assert_that(words, equal_to([("negative", "food"), ("negative", "not"), ("negative", "good")]))


def test_source_stats_fn():
    with BeamTestPipeline() as p:
        stats = (
            p
            | beam.Create([
                ("yelp", {"label": 1, "text": "great food"}),
                ("yelp", {"label": 0, "text": "cold and bland soup"}),
            ])
            | beam.CombinePerKey(SourceStatsFn())
        )
        assert_that(
            stats,
            equal_to([("yelp", {"sentences": 2, "positive": 1, "negative": 1, "avg_words": 3.0})]),
        )


def test_runinference_with_keyed_sklearn_model(tmp_path):
    model = tiny_model(["great product", "love it", "terrible product", "hate it"], [1, 1, 0, 0])
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(model))
    handler = KeyedModelHandler(
        SklearnModelHandlerNumpy(model_uri=str(model_path), inference_fn=predict_proba)
    )
    with BeamTestPipeline() as p:
        rows = (
            p
            | beam.Create([
                (("yelp", 1, "great product"), "great product"),
                (("yelp", 0, "terrible product"), "terrible product"),
            ])
            | RunInference(handler)
            | beam.Map(to_prediction_row)
            | beam.Map(lambda r: (r["text"], r["predicted"], r["correct"]))
        )
        assert_that(rows, equal_to([("great product", 1, True), ("terrible product", 0, True)]))



def test_full_pipeline_end_to_end(tmp_path):
    good = [f"great product number {i}" for i in range(15)]
    bad = [f"terrible product number {i}" for i in range(15)]
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    for name in ("yelp_labelled.txt", "imdb_labelled.txt"):
        lines = [f"{t}\t1" for t in good] + [f"{t}\t0" for t in bad]
        if name == "imdb_labelled.txt":
            lines.append("this line has no label")
        (data_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "stopwords.txt").write_text("the\n", encoding="utf-8")
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(tiny_model(good + bad, [1] * 15 + [0] * 15)))
    out_dir = tmp_path / "outputs"

    run([
        f"--input={data_dir}/*_labelled.txt",
        f"--stopwords={tmp_path / 'stopwords.txt'}",
        f"--model={model_path}",
        f"--output-dir={out_dir}",
    ])

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["records"] == 60
    assert summary["malformed"] == 1
    assert summary["unique_sentences"] == 30
    assert summary["train_rows"] + summary["test_rows"] == 60
    assert summary["sources"]["yelp"] == {"sentences": 30, "positive": 15, "negative": 15, "avg_words": 3.0}
    predictions = (out_dir / "predictions.jsonl").read_text().splitlines()
    assert len(predictions) == summary["test_rows"]
    if summary["test_rows"]:
        assert summary["test_accuracy"] >= 0.9
    assert "missing tab separator" in (out_dir / "malformed.jsonl").read_text()
