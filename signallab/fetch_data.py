"""Download a bounded balanced public Amazon Polarity sample, not private reviews."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pandas as pd
from datasets import load_dataset

from signallab.core import SEED, validate_reviews

ROOT = Path(__file__).resolve().parents[1]


def fetch_reviews(limit):
    if limit < 40 or limit > 10000 or limit % 2:
        raise ValueError("Choose an even --limit from 40 to 10000.")
    source = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    rows, seen, counts = [], set(), {0: 0, 1: 0}
    for row in source.shuffle(seed=SEED, buffer_size=10000):
        label = int(row["label"])
        text = f"{row['title']} {row['content']}".strip()
        key = " ".join(text.casefold().split())
        if key in seen or counts[label] >= limit // 2 or len(text) > 20000:
            continue
        rows.append({"text": text, "label": label})
        seen.add(key)
        counts[label] += 1
        if len(rows) == limit:
            break
    if len(rows) != limit:
        raise ValueError("Dataset ended before enough reviews could be collected.")
    return validate_reviews(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/reviews.csv")
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(1, "Output exists. Choose a different --output path to preserve it.\n")
    try:
        frame = fetch_reviews(args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            frame.to_csv(handle, index=False)
    except (ValueError, OSError, RuntimeError):
        parser.exit(1, "Download failed. Check network access, dataset availability and --limit; existing files were preserved.\n")
    print(f"Saved {len(frame)} reviews to {args.output}. Source: fancyzhx/amazon_polarity, train split, sampling seed {SEED}.")


if __name__ == "__main__":
    main()
