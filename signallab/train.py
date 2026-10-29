"""A small explicit PyTorch loop: no hosted inference or Trainer dependency."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from signallab.inference import validate_batch_size, MAX_LENGTH, class_probabilities, encode_reviews
from signallab.core import SEED, augment_training, dataset_id, load_csv, make_splits, evidence_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "distilbert/distilbert-base-uncased"


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def batches(frame, tokenizer, batch_size, shuffle=False):
    rows = [{"text": row.text, "label": int(row.label)} for row in frame.itertuples()]

    def collate(examples):
        encoded = encode_reviews(tokenizer, [r["text"] for r in examples])
        encoded["labels"] = torch.tensor([r["label"] for r in examples], dtype=torch.long)
        return encoded

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(SEED))


def score(model, frame, tokenizer, batch_size):
    probabilities = class_probabilities(model, tokenizer, frame.text.tolist(), batch_size)
    return evidence_report(frame, probabilities.argmax(dim=-1).tolist())


def fit_model(train, validation, checkpoint, epochs=2, batch_size=8, learning_rate=2e-5):
    validate_batch_size(batch_size)
    # Reset before BOTH runs so their initial weights and dropout seeds match.
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}, trust_remote_code=False,
        use_safetensors=True,
    )
    model.to("cpu")
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_f1 = -1.0
    best_state = None
    history = []
    loader = batches(train, tokenizer, batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss. Check the dataset and learning rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += float(loss.detach()) * len(batch["labels"])
        validation_report = score(model, validation, tokenizer, batch_size)
        history.append({"epoch": epoch + 1, "training_loss": total_loss / len(train),
                        "validation_macro_f1": validation_report["macro_f1"]})
        print(f"epoch {epoch + 1}/{epochs}: loss={history[-1]['training_loss']:.4f}, validation F1={validation_report['macro_f1']:.4f}", flush=True)
        if validation_report["macro_f1"] > best_f1:
            best_f1 = validation_report["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, tokenizer, history


def run_transformer(frame, output, checkpoint=DEFAULT_CHECKPOINT, epochs=2, ratio=0.5, batch_size=8, learning_rate=2e-5):
    validate_batch_size(batch_size)
    if epochs < 1 or not 0 < learning_rate < 1:
        raise ValueError("Use positive epochs/batch size and a learning rate between 0 and 1.")
    split = make_splits(frame)
    augmented = augment_training(split.train, ratio)
    output = Path(output)
    # Preserve completed experiments; a failed run never replaces an existing report.
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for slug, name, training in (("real", "DistilBERT · real only", split.train),
                                 ("augmented", "DistilBERT · augmented", augmented)):
        print(f"Training {name}: {len(training)} rows on CPU", flush=True)
        model, tokenizer, history = fit_model(training, split.validation, checkpoint, epochs, batch_size, learning_rate)
        model.save_pretrained(output / slug, safe_serialization=True)
        tokenizer.save_pretrained(output / slug)
        runs.append({"name": name, "training_rows": len(training), "history": history,
                     "validation": score(model, split.validation, tokenizer, batch_size),
                     "test": score(model, split.test, tokenizer, batch_size)})
        del model
    report = {
        "schema_version": 1, "dataset_id": dataset_id(frame), "seed": SEED,
        "checkpoint": str(checkpoint), "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                     "max_length": MAX_LENGTH, "device": "cpu", "selection": "best validation macro F1"},
        "augmentation": {"method": "training-only prefix template", "ratio": ratio},
        "split_sizes": {k: len(getattr(split, k)) for k in ("train", "validation", "test")},
        "runs": runs,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/transformer")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8, help="Integer 1–128 (default: 8); checked before training/output creation.")
    args = parser.parse_args()
    try:
        report = run_transformer(load_csv(args.csv), args.output, args.checkpoint, args.epochs, args.ratio, args.batch_size)
    except ValueError as exc:
        parser.exit(1, f"Training settings/data rejected: {exc} See README: PyTorch + Hugging Face. Use a new --output for retries.\n")
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Training failed ({type(exc).__name__}). Check the CSV, checkpoint access and available memory; use a new --output directory for retries. No completed report was published.\n")
    print(f"Saved measured results to {args.output / 'report.json'} (dataset {report['dataset_id'][:8]}).")


if __name__ == "__main__":
    main()
