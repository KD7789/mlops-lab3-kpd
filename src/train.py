"""Train the sentiment model that the Beam pipeline runs with RunInference.

Run from the repo root:  python -m src.train
"""
import argparse
import json
import pickle
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from src.common import LABELS, is_test, parse_line, read_lines, source_from_path


def load_rows(data_dir):
    """Read every labelled file into (source, text, label) rows."""
    rows = []
    for path in sorted(Path(data_dir).glob("*_labelled.txt")):
        for line in read_lines(path):
            text, label = parse_line(line)
            rows.append((source_from_path(str(path)), text, label))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    rows = load_rows(args.data_dir)
    train = [r for r in rows if not is_test(r[1])]
    test = [r for r in rows if is_test(r[1])]

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    model.fit([text for _, text, _ in train], [label for _, _, label in train])

    y_true = [label for _, _, label in test]
    y_pred = model.predict([text for _, text, _ in test]).tolist()

    per_source = {}
    for source in sorted({s for s, _, _ in test}):
        idx = [i for i, (s, _, _) in enumerate(test) if s == source]
        per_source[source] = round(
            accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx]), 4
        )

    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "test_accuracy_by_source": per_source,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=[LABELS[0], LABELS[1]], output_dict=True
        ),
        "sklearn_version": sklearn.__version__,
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"train rows: {len(train)} | test rows: {len(test)}")
    print(f"test accuracy: {metrics['test_accuracy']}")
    print("by source:", per_source)
    print(f"saved {model_dir / 'sentiment_model.pkl'} and {model_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
