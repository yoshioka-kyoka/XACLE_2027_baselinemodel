"""Leave-One-Synthesizer-Out (LOSO) split generator.

XACLE Challenge 2027 trains on known audio-generation models and evaluates on
unseen ones. The official validation/test splits reuse the same generator set
as train, so they cannot measure that generalization gap on their own (see
docs/strategy.md, Stage 0). This script builds a train-internal diagnostic:
for each held-out generator, it writes

  meta_data/loso/train_minus_<generator>.csv  (train rows, generator excluded)
  meta_data/loso/heldout_<generator>.csv      (train rows, generator only)

A model trained on `train_minus_<generator>.csv` has never seen that
generator; scoring it on `heldout_<generator>.csv` approximates "accuracy on
an unseen generation model" using only data the rules allow for training.
Both CSVs keep the exact column layout of train_average.csv, so they plug
into the existing XACLEDataset/get_infdataset unchanged.

`model_ID` 0 (audiocaps) is real recorded audio, not a generation model, so
it is excluded from the default held-out set but always kept on the train
side (use --include-audiocaps to also hold it out).

Usage:
    python scripts/make_loso_splits.py
    python scripts/make_loso_splits.py --generators audiogen,tango
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
META_DIR = REPO_ROOT / "datasets" / "XACLE_Challenge_2027_development_dataset" / "meta_data"
TRAIN_AVERAGE_CSV = META_DIR / "train_average.csv"
LOSO_DIR = META_DIR / "loso"

# All generators present in the development dataset's train split.
AUDIOCAPS_MODEL_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generators",
        help="Comma-separated model_name values to hold out. Default: every "
        "synthetic generator in train_average.csv (audiocaps excluded).",
    )
    parser.add_argument(
        "--include-audiocaps",
        action="store_true",
        help="Also generate a held-out split for audiocaps (real audio, not a generator).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not TRAIN_AVERAGE_CSV.is_file():
        raise FileNotFoundError(f"train_average.csv not found at {TRAIN_AVERAGE_CSV}")

    df = pd.read_csv(TRAIN_AVERAGE_CSV)
    available = df[["model_ID", "model_name"]].drop_duplicates().sort_values("model_ID")

    if args.generators:
        requested = set(args.generators.split(","))
        available = available[available["model_name"].isin(requested)]
        missing = requested - set(available["model_name"])
        if missing:
            raise ValueError(f"Unknown generator name(s): {sorted(missing)}")
    elif not args.include_audiocaps:
        available = available[available["model_ID"] != AUDIOCAPS_MODEL_ID]

    LOSO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing LOSO splits to {LOSO_DIR}")
    for _, row in available.iterrows():
        model_id, name = int(row["model_ID"]), row["model_name"]
        heldout = df[df["model_ID"] == model_id]
        train_minus = df[df["model_ID"] != model_id]

        heldout_path = LOSO_DIR / f"heldout_{name}.csv"
        train_minus_path = LOSO_DIR / f"train_minus_{name}.csv"
        heldout.to_csv(heldout_path, index=False)
        train_minus.to_csv(train_minus_path, index=False)
        print(
            f"  {name:<16} (model_ID={model_id}): "
            f"train_minus={len(train_minus)} rows, heldout={len(heldout)} rows"
        )


if __name__ == "__main__":
    main()
