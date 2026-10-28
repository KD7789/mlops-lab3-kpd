"""Beam building blocks for the sentiment pipeline.

Kept in their own module so the pipeline and the tests can both import them.
"""
import json
import os

import apache_beam as beam

from src.common import LABELS, is_test, parse_line, source_from_path, tokenize


class ParseRecord(beam.DoFn):
    """Validate (filename, line) pairs.

    Good lines become record dicts on the main output; bad lines go to the
    'malformed' output (a dead-letter queue) instead of crashing the pipeline.
    """

    MALFORMED = "malformed"

    def process(self, element):
        filename, line = element
        try:
            text, label = parse_line(line)
        except ValueError as err:
            yield beam.pvalue.TaggedOutput(
                self.MALFORMED,
                {"file": os.path.basename(filename), "line": line, "error": str(err)},
            )
            return
        yield {
            "source": source_from_path(filename),
            "text": text,
            "label": label,
            "split": "test" if is_test(text) else "train",
        }


class CleanWords(beam.PTransform):
    """Composite transform: records -> (sentiment, word) pairs.

    Lowercases, drops stopwords (a side input of (word, True) pairs),
    and drops words shorter than min_len.
    """

    def __init__(self, stopwords, min_len=2):
        super().__init__()
        self.stopwords = stopwords
        self.min_len = min_len

    def expand(self, records):
        min_len = self.min_len
        return (
            records
            | "Tokenize" >> beam.FlatMap(
                lambda r: [(LABELS[r["label"]], w) for w in tokenize(r["text"])]
            )
            | "Drop stopwords" >> beam.Filter(
                lambda sw, stop: sw[1] not in stop,
                stop=beam.pvalue.AsDict(self.stopwords),
            )
            | "Drop short words" >> beam.Filter(lambda sw: len(sw[1]) >= min_len)
        )


class SourceStatsFn(beam.CombineFn):
    """Custom combiner: sentence count, positive/negative split, and average words per source."""

    def create_accumulator(self):
        return (0, 0, 0)  

    def add_input(self, acc, record):
        n, pos, words = acc
        return n + 1, pos + record["label"], words + len(tokenize(record["text"]))

    def merge_accumulators(self, accs):
        n = pos = words = 0
        for a_n, a_pos, a_words in accs:
            n, pos, words = n + a_n, pos + a_pos, words + a_words
        return n, pos, words

    def extract_output(self, acc):
        n, pos, words = acc
        return {
            "sentences": n,
            "positive": pos,
            "negative": n - pos,
            "avg_words": round(words / n, 2) if n else 0.0,
        }


def predict_proba(model, batch, inference_args=None):
    """Custom inference_fn for RunInference: class probabilities instead of hard labels."""
    return model.predict_proba(list(batch))


def to_prediction_row(keyed_result):
    """((source, label, text), PredictionResult) -> flat dict for the dashboard."""
    (source, label, text), result = keyed_result
    p_negative, p_positive = (float(p) for p in result.inference)
    predicted = int(p_positive >= 0.5)
    return {
        "source": source,
        "text": text,
        "label": label,
        "predicted": predicted,
        "confidence": round(max(p_negative, p_positive), 4),
        "correct": predicted == label,
    }


def build_summary(_, records, malformed, unique, train, test, source_stats, accuracy, accuracy_by_source):
    """Join the results of every branch into one JSON document (all arrive as side inputs)."""
    return json.dumps(
        {
            "records": records,
            "malformed": malformed,
            "unique_sentences": unique,
            "duplicate_sentences": records - unique,
            "train_rows": train,
            "test_rows": test,
            "test_accuracy": round(accuracy, 4),
            "test_accuracy_by_source": {k: round(v, 4) for k, v in sorted(accuracy_by_source.items())},
            "sources": dict(sorted(source_stats.items())),
        },
        indent=2,
    )
