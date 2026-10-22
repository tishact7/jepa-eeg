import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dataset import load_test
from src.orchestrate import build_final_report, final_counts
from src.selection import SEEDS, falsifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/main")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.out_dir, "RELEASE_S9.txt")):
        sys.exit("S9 is not released. Create RELEASE_S9.txt only after the owner's explicit release.")
    result_path = os.path.join(args.out_dir, "final_results.json")
    if os.path.exists(result_path):
        sys.exit("final_results.json already exists. S9 is evaluated once.")

    records = []
    for seed in SEEDS:
        path = os.path.join(args.out_dir, f"selection_record_seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                records.append(json.load(f))
    if len(records) != len(SEEDS) or any(r["status"] != "SELECTED" for r in records):
        report = {"falsifier": falsifier({}, {}), "s9_accessed": False,
                  "reason": "selection records incomplete or not SELECTED"}
        with open(result_path, "w") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        sys.exit("Comparison is not evaluable. S9 was not accessed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = np.load(os.path.join(args.out_dir, "norm_stats.npz"))
    Xtest, ytest = load_test(stats["mean"], stats["std"], records)
    counts = {r["seed"]: final_counts(r, Xtest, ytest, device) for r in records}
    report = build_final_report(records, counts)
    report["s9_accessed"] = True
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report["falsifier"], indent=2))


if __name__ == "__main__":
    main()
