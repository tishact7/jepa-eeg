import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dataset import load_train_val
from src.orchestrate import FROZEN_PLAN, run_seed
from src.selection import SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--out_dir", default="results/main")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.out_dir, "RELEASE_TRAINING.txt")):
        sys.exit("Training is not released. Create RELEASE_TRAINING.txt only after the owner's explicit release.")
    record_path = os.path.join(args.out_dir, f"selection_record_seed{args.seed}.json")
    if os.path.exists(record_path):
        sys.exit("A selection record already exists for this seed. Any retry must follow the logged infrastructure-failure rule in AMENDMENT_1.md.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, (mean, std) = load_train_val()
    stats_path = os.path.join(args.out_dir, "norm_stats.npz")
    if not os.path.exists(stats_path):
        np.savez(stats_path, mean=mean, std=std)
    record = run_seed(args.seed, data, args.out_dir, device, plan=FROZEN_PLAN)
    print("seed", args.seed, "status", record["status"], "| S8 evaluations", record["s8_evaluations"])
    if record["status"] == "SELECTED":
        for model, sel in record["selected"].items():
            print(model, {k: sel[k] for k in ("checkpoint_epoch", "learning_rate", "probe_epochs", "s8_correct", "s8_total")})
    else:
        print("failures:", record["failures"])


if __name__ == "__main__":
    main()
