"""Run the Leave-One-Synthesizer-Out (LOSO) sweep (docs/strategy.md, Stage 0).

For each held-out generator, this trains a model on train_minus_<generator>.csv
(official validation_average.csv is still used for best-epoch selection, exactly
as the baseline already does), then scores the resulting checkpoint on
heldout_<generator>.csv -- audio from a generator the model never trained on.

Requires `python scripts/make_loso_splits.py` to have been run first.

Each generator is a full, independent training run (~minutes per run with the
frozen M2D-CLAP encoder). Train.py/inference.py/evaluate.py are invoked as
subprocesses, unmodified, so this script carries zero changes to the training
loop, loss, or dataset code -- only new data/config artifacts.

Usage:
    python scripts/run_loso_sweep.py                      # all generators
    python scripts/run_loso_sweep.py --generators tango   # smoke test, one generator
    python scripts/run_loso_sweep.py --epochs 3           # faster smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
META_DIR = REPO_ROOT / "datasets" / "XACLE_Challenge_2027_development_dataset" / "meta_data"
LOSO_META_DIR = META_DIR / "loso"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "train.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "chkpt_loso"
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generators",
        help="Comma-separated generator names to sweep. Default: every "
        "heldout_<name>.csv found under meta_data/loso/.",
    )
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, help="Override epochs (e.g. for a quick smoke test).")
    return parser.parse_args()


def discover_generators():
    names = sorted(p.stem.removeprefix("heldout_") for p in LOSO_META_DIR.glob("heldout_*.csv"))
    if not names:
        raise FileNotFoundError(
            f"No heldout_*.csv found under {LOSO_META_DIR}. "
            "Run scripts/make_loso_splits.py first."
        )
    return names


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def train_one_generator(name, base_config_path, output_root, epochs_override):
    train_minus_csv = LOSO_META_DIR / f"train_minus_{name}.csv"
    heldout_csv = LOSO_META_DIR / f"heldout_{name}.csv"
    if not train_minus_csv.is_file() or not heldout_csv.is_file():
        raise FileNotFoundError(f"Missing LOSO CSVs for generator '{name}' under {LOSO_META_DIR}")

    cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
    cfg["train_list"] = str(train_minus_csv.relative_to(REPO_ROOT))
    output_dir = output_root / name
    cfg["output_dir"] = str(output_dir.relative_to(REPO_ROOT))
    if epochs_override is not None:
        cfg["epochs"] = epochs_override

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config_path = output_dir / "loso_train_config.json"
    generated_config_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False), encoding="utf-8")

    existing_runs = {p for p in output_dir.glob("*") if p.is_dir()}
    run([PYTHON, "train.py", "--config", str(generated_config_path)], cwd=REPO_ROOT)

    new_runs = sorted(
        (p for p in output_dir.glob("*") if p.is_dir() and p not in existing_runs),
        key=lambda p: p.stat().st_mtime,
    )
    if not new_runs:
        raise RuntimeError(f"train.py did not create a new run directory under {output_dir}")
    run_dir = new_runs[-1]

    run(
        [
            PYTHON,
            "inference.py",
            str(run_dir),
            "train",
            "--metadata-list",
            str(heldout_csv.relative_to(REPO_ROOT)),
            "--result-name",
            "loso_heldout",
        ],
        cwd=REPO_ROOT,
    )

    eval_dir = run_dir / "evaluation" / "loso_heldout"
    run(
        [
            PYTHON,
            "evaluate.py",
            str(run_dir / "inference_result_for_loso_heldout.csv"),
            str(heldout_csv),
            str(eval_dir),
        ],
        cwd=REPO_ROOT,
    )

    metrics = {}
    with open(eval_dir / "evaluation_result.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics[row["metric"]] = row["value"]
    return run_dir, metrics


def main():
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    output_root = Path(args.output_root).resolve()
    names = args.generators.split(",") if args.generators else discover_generators()

    results = []
    for name in names:
        print(f"\n=== LOSO: holding out '{name}' ===")
        run_dir, metrics = train_one_generator(name, base_config_path, output_root, args.epochs)
        metrics["generator"] = name
        metrics["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
        results.append(metrics)
        print(f"  -> SRCC={metrics.get('SRCC')} LCC={metrics.get('LCC')} "
              f"KTAU={metrics.get('KTAU')} MSE={metrics.get('MSE')} N={metrics.get('N')}")

    summary_path = output_root / "loso_summary.csv"
    fieldnames = ["generator", "SRCC", "LCC", "KTAU", "MSE", "N", "run_dir"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nLOSO summary written to {summary_path}")


if __name__ == "__main__":
    main()
