"""
Stage 1 — Download and prepare the financial_phrasebank dataset.

Output:
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl

Each line: {"text": "...", "label": 0|1|2, "label_text": "negative|neutral|positive"}
"""

import json
import logging
import random
from pathlib import Path

import yaml
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

LABEL2ID = {label: i for i, label in enumerate(CONFIG["dataset"]["labels"])}
# financial_phrasebank uses: 0=negative, 1=neutral, 2=positive
PHRASEBANK_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def format_sample(sentence: str, label_id: int) -> dict:
    """
    Format a sample as an instruction-following prompt.

    Why instruction format?
    We're fine-tuning a generative model (Llama) for classification.
    Instruction format lets it learn to produce a structured label token,
    which is more reliable than mapping logits to classes.
    """
    label_text = PHRASEBANK_LABEL_MAP[label_id]
    return {
        "text": (
            f"### Instruction:\n"
            f"Classify the sentiment of the following financial news sentence.\n"
            f"Reply with one word: negative, neutral, or positive.\n\n"
            f"### Input:\n{sentence}\n\n"
            f"### Response:\n{label_text}"
        ),
        "label": label_id,
        "label_text": label_text,
    }


def main():
    logger.info("Downloading financial_phrasebank...")
    ds = load_dataset(CONFIG["dataset"]["name"], CONFIG["dataset"]["subset"], trust_remote_code=True)

    samples = [
        format_sample(row["sentence"], row["label"])
        for row in ds["train"]
    ]

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * CONFIG["dataset"]["train_split"])
    n_val = int(n * CONFIG["dataset"]["val_split"])

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    for split_name, split_data in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(split_data)} samples → {out_path}")

    label_counts = {v: 0 for v in PHRASEBANK_LABEL_MAP.values()}
    for s in samples:
        label_counts[s["label_text"]] += 1
    logger.info(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    main()
