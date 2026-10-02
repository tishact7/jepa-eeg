# jepa-eeg

JEPA-style self-supervised encoder vs a matched supervised baseline on BCI Competition IV 2a motor imagery, with cross-subject evaluation (train S1-S7, validation S8, test S9).

See `FROZEN_SPEC.md` for the full frozen design.

## Reproduce the plumbing run

    git clone https://github.com/tishact7/jepa-eeg
    cd jepa-eeg
    pip install -r requirements.txt
    python scripts/run_plumbing.py --seed 42

Outputs go to `results/plumbing/` (`summary.json`, `train_log.jsonl`, `tsne_dummy.png`). Tested on Colab T4.

## Plumbing run result (S1 only, 2 epochs, seed 42)

| Check | Result |
|---|---|
| 1. Shapes | PASS: trials (576, 22, 500), patches (576, 10, 22, 50), context (B, 6, 22, 50), target (B, 2, 22, 50), predictor out (B, 2, 128) |
| 2. Loss, first 20 steps | Strictly monotonic: False (17 of 19 transitions decreased). 5-step moving average fell 1.824 to 1.016: True. No collapse (target std about 0.93). |
| 3. Checkpoint | PASS: epoch 1 reloaded into fresh models, outputs and weights identical |
| 4. Probe | PASS: runs end to end on an 80/20 trial split of S1. JEPA eval 25.0%, random encoder eval 25.9%. This is a dummy within-subject split, not a result. |
