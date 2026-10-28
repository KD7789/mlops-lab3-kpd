"""Sentiment analytics and batch inference pipeline built with Apache Beam.

Run from the repo root, after training the model:
    python -m src.pipeline
Extra flags (for example --runner=DirectRunner) are passed to Beam.
"""
import argparse
import functools
import json
import os
import sys

import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.io.textio import ReadFromTextWithFilename
from apache_beam.ml.inference.base import KeyedModelHandler, RunInference
from apache_beam.ml.inference.sklearn_inference import SklearnModelHandlerNumpy
from apache_beam.options.pipeline_options import PipelineOptions

from src.transforms import (
    CleanWords,
    ParseRecord,
    SourceStatsFn,
    build_summary,
    predict_proba,
    to_prediction_row,
)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/*_labelled.txt")
    parser.add_argument("--stopwords", default="data/stopwords.txt")
    parser.add_argument("--model", default="models/sentiment_model.pkl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--min-word-len", type=int, default=2)
    args, beam_args = parser.parse_known_args(argv)

    if not os.path.exists(args.model):
        sys.exit(f"Model not found at {args.model}. Run `python -m src.train` first.")
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    def write(pcoll, name, suffix, header=None):
        """Write a PCollection to a single file: <output-dir>/<name><suffix>."""
        return pcoll | f"Write {name}" >> WriteToText(
            os.path.join(out, name),
            file_name_suffix=suffix,
            shard_name_template="",
            header=header,
        )

    with beam.Pipeline(options=PipelineOptions(beam_args)) as p:
        lines = p | "Read reviews" >> ReadFromTextWithFilename(args.input)

        parsed = lines | "Parse and validate" >> beam.ParDo(ParseRecord()).with_outputs(
            ParseRecord.MALFORMED, main="records"
        )
        records, malformed = parsed.records, parsed[ParseRecord.MALFORMED]

        stopwords = (
            p
            | "Read stopwords" >> ReadFromText(args.stopwords)
            | "Key stopwords" >> beam.Map(lambda w: (w.strip(), True))
        )
        word_counts = (
            records
            | "Clean words" >> CleanWords(stopwords, args.min_word_len)
            | "Pair with 1" >> beam.Map(lambda sw: (sw, 1))
            | "Count words" >> beam.CombinePerKey(sum)
        )
        top_words = (
            word_counts
            | "Key by sentiment" >> beam.Map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
            | "Top N per sentiment" >> beam.combiners.Top.PerKey(args.top_n, key=lambda wc: wc[1])
        )

        source_stats = (
            records
            | "Key by source" >> beam.Map(lambda r: (r["source"], r))
            | "Stats per source" >> beam.CombinePerKey(SourceStatsFn())
        )

        train, test = records | "Split train/test" >> beam.Partition(
            lambda r, n: 1 if r["split"] == "test" else 0, 2
        )
        model_handler = KeyedModelHandler(
            SklearnModelHandlerNumpy(model_uri=args.model, inference_fn=predict_proba)
        )
        predictions = (
            test
            | "Key for inference" >> beam.Map(lambda r: ((r["source"], r["label"], r["text"]), r["text"]))
            | "RunInference" >> RunInference(model_handler)
            | "To prediction rows" >> beam.Map(to_prediction_row)
        )
        accuracy = (
            predictions
            | "Correct as 0/1" >> beam.Map(lambda row: int(row["correct"]))
            | "Overall accuracy" >> beam.combiners.Mean.Globally()
        )
        accuracy_by_source = (
            predictions
            | "Correct by source" >> beam.Map(lambda row: (row["source"], int(row["correct"])))
            | "Accuracy per source" >> beam.combiners.Mean.PerKey()
        )

        count = beam.combiners.Count.Globally
        unique_texts = (
            records
            | "Texts" >> beam.Map(lambda r: r["text"])
            | "Distinct texts" >> beam.Distinct()
        )
        summary = (
            p
            | "Start summary" >> beam.Create([None])
            | "Build summary" >> beam.Map(
                build_summary,
                records=beam.pvalue.AsSingleton(records | "Count records" >> count()),
                malformed=beam.pvalue.AsSingleton(malformed | "Count malformed" >> count()),
                unique=beam.pvalue.AsSingleton(unique_texts | "Count unique" >> count()),
                train=beam.pvalue.AsSingleton(train | "Count train" >> count()),
                test=beam.pvalue.AsSingleton(test | "Count test" >> count()),
                source_stats=beam.pvalue.AsDict(source_stats),
                accuracy=beam.pvalue.AsSingleton(accuracy),
                accuracy_by_source=beam.pvalue.AsDict(accuracy_by_source),
            )
        )

        write(
            word_counts | "Format CSV rows" >> beam.Map(lambda kv: f"{kv[0][0]},{kv[0][1]},{kv[1]}"),
            "word_counts", ".csv", header="sentiment,word,count",
        )
        write(
            top_words
            | "Top words to dict" >> beam.combiners.ToDict()
            | "Top words to JSON" >> beam.Map(functools.partial(json.dumps, indent=2)),
            "top_words", ".json",
        )
        write(predictions | "Predictions to JSON" >> beam.Map(json.dumps), "predictions", ".jsonl")
        write(malformed | "Malformed to JSON" >> beam.Map(json.dumps), "malformed", ".jsonl")
        write(summary, "summary", ".json")

    with open(os.path.join(out, "summary.json")) as f:
        print(f.read())
    print(f"Outputs written to {out}/")


if __name__ == "__main__":
    run()
