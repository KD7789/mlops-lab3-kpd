"""Helpers shared by the training script, the Beam pipeline, and the dashboard."""
import hashlib
import os
import re

SOURCES = {
    "amazon_cells_labelled.txt": "amazon",
    "imdb_labelled.txt": "imdb",
    "yelp_labelled.txt": "yelp",
}
LABELS = {0: "negative", 1: "positive"}
TEST_PERCENT = 20
TOKEN_RE = re.compile(r"[a-z']+")


def source_from_path(path):
    """Map a file path to its short source name."""
    return SOURCES.get(os.path.basename(path), "unknown")


def read_lines(path):
    """Split a data file on newline characters only, the same way Beam's ReadFromText does."""
    with open(path, encoding="utf-8", newline="") as f:
        return [line for line in f.read().split("\n") if line]


def parse_line(line):
    """Turn 'sentence<TAB>label' into (sentence, label). Raises ValueError if malformed.

    Also repairs a data quirk: two IMDB lines contain U+0085, a Windows-1252
    ellipsis that was decoded wrongly, so it is turned back into '...'.
    """
    if "\t" not in line:
        raise ValueError("missing tab separator")
    text, label = line.rsplit("\t", 1)
    text, label = text.replace("\x85", "...").strip(), label.strip()
    if not text:
        raise ValueError("empty sentence")
    if label not in ("0", "1"):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    return text, int(label)


def tokenize(text):
    """Same regex idea as the original lab, but lowercased and with stray quotes stripped."""
    words = (w.strip("'") for w in TOKEN_RE.findall(text.lower()))
    return [w for w in words if w]


def is_test(text):
    """Deterministic train/test split: the same sentence always lands in the same split.

    Uses md5 instead of hash(), because hash() changes between Python runs.
    """
    bucket = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 100
    return bucket < TEST_PERCENT
