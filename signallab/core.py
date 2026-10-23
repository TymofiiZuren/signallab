"""Shared contracts for the UI and command-line experiments."""
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import json
import math
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 42
PREPROCESSING = "binary-review-dedup-prefix-v1"
SPLIT_POLICY = "stratified-60-20-20-seed42-v1"
LABELS = {0: "Negative", 1: "Positive"}
BASELINE_PARAMETERS = {
    "vectorizer": {"ngram_range": (1, 2), "min_df": 1, "max_features": 30000, "sublinear_tf": True},
    "classifier": {"max_iter": 1000, "random_state": SEED, "class_weight": "balanced"},
}


def normalise(text):
    return re.sub(r"\s+", " ", text.strip()).casefold()


def canonical_review_key(text):
    """Existing text-only duplicate rule, shared with study exclusions."""
    return re.sub(r"^(?:in my experience, )+", "", normalise(text))


def validate_reviews(frame):
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("CSV needs text and label columns; label must be 0/negative or 1/positive.")
    frame = frame[["text", "label"]].copy()
    if frame.text.isna().any() or not frame.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Every review needs non-empty text. Remove or fill missing reviews.")
    if frame.text.str.len().gt(20000).any():
        raise ValueError("Reviews must be at most 20,000 characters. Shorten longer reviews.")
    if len(frame) > 50000:
        raise ValueError("Use at most 50,000 rows for this local learning app.")
    mapping = {"0": 0, "0.0": 0, "negative": 0, "1": 1, "1.0": 1, "positive": 1}
    labels = frame.label.map(lambda x: mapping.get(str(x).strip().lower()))
    if labels.isna().any():
        raise ValueError("Labels must be 0/negative or 1/positive; missing or neutral labels are unsupported.")
    frame["label"] = labels.astype(int)
    frame["text"] = frame.text.str.strip()
    frame["key"] = frame.text.map(canonical_review_key)
    if frame.groupby("key").label.nunique().gt(1).any():
        raise ValueError("Identical reviews have conflicting labels. Resolve these before training.")
    frame = frame.sort_values(["key", "text"]).drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    if any((frame.label == label).sum() < 8 for label in LABELS):
        raise ValueError("Need at least 8 distinct reviews in each class after deduplication.")
    return frame


def load_csv(source):
    try:
        return validate_reviews(pd.read_csv(source))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError("Could not read CSV. Use a UTF-8 CSV with text and label columns.") from exc


def dataset_id(frame):
    records = sorted((normalise(row.text), int(row.label)) for row in frame.itertuples())
    return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def make_splits(frame):
    clean = validate_reviews(frame)
    train, remainder = train_test_split(clean, test_size=0.4, stratify=clean.label, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=0.5, stratify=remainder.label, random_state=SEED)
    return Splits(*(part.reset_index(drop=True) for part in (train, validation, test)))


def augment_training(train, ratio):
    """A deliberately simple control, not an LLM or a claim of semantic diversity."""
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("Synthetic ratio must be between 0 and 1.")
    count = int(len(train) * ratio)
    if count == 0:
        return train.copy()
    extra = train.sample(n=count, random_state=SEED).copy()
    extra["text"] = "In my experience, " + extra.text
    return pd.concat([train, extra], ignore_index=True)


def train_baseline(train):
    model = make_pipeline(
        TfidfVectorizer(**BASELINE_PARAMETERS["vectorizer"]),
        LogisticRegression(**BASELINE_PARAMETERS["classifier"]),
    )
    return model.fit(train.text.tolist(), train.label.tolist())


def prediction_report(truth, predictions):
    truth, predictions = list(truth), [int(x) for x in predictions]
    return {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "classes": classification_report(truth, predictions, labels=[0, 1], target_names=["Negative", "Positive"], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "predictions": predictions,
    }


def review_id(text, label):
    payload = json.dumps([text, int(label)], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def exact_dataset_id(frame):
    ids = sorted(review_id(row.text, row.label) for row in frame.itertuples())
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def evidence_report(frame, predictions):
    predictions = list(predictions)
    if len(predictions) != len(frame) or any(p not in LABELS for p in predictions):
        raise ValueError('Predictions must contain one binary label per evaluated review.')
    report = prediction_report(frame.label, predictions)
    report['evidence_version'] = 1
    report['evidence'] = [
        {'review_id': review_id(row.text, row.label), 'label': int(row.label), 'prediction': int(prediction)}
        for row, prediction in zip(frame.itertuples(), predictions)
    ]
    return report


def evaluate(model, frame):
    return evidence_report(frame, model.predict(frame.text.tolist()))


def baseline_provenance():
    source_root = Path(__file__).resolve().parent
    return {
        "parameters": json.loads(json.dumps(BASELINE_PARAMETERS)),
        "preprocessing": PREPROCESSING, "split_policy": SPLIT_POLICY,
        "dependencies": {name: version(name) for name in ("pandas", "numpy", "scikit-learn")},
        "python": platform.python_version(), "git_commit": None,
        "source_hashes": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                          for name in ("core.py", "analysis.py", "runs.py")},
    }


def run_comparison(frame, ratio=0.5):
    # Capture the execution context before fitting, never when a later Save is clicked.
    provenance = baseline_provenance()
    split = make_splits(frame)
    runs = []
    models = {}
    for name, training in (
        ("TF-IDF · real only", split.train),
        ("TF-IDF · augmented", augment_training(split.train, ratio)),
    ):
        model = train_baseline(training)
        models[name] = model
        runs.append({
            "name": name, "training_rows": len(training),
            "validation": evaluate(model, split.validation),
            "test": evaluate(model, split.test),
        })
    return {
        "schema_version": 1, "dataset_id": dataset_id(frame), "input_id": exact_dataset_id(pd.concat([split.train, split.validation, split.test])),
        "seed": SEED, "provenance": provenance,
        "augmentation": {"method": "training-only prefix template", "ratio": ratio},
        "split_sizes": {key: len(getattr(split, key)) for key in ("train", "validation", "test")},
        "runs": runs,
    }, models, split
