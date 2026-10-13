# Amendment 1 to freeze-v1

Status: append-only, adopted by owner decision. Contains no results.

Predecessor: tag `freeze-v1`, commit `8f00708c144a55cc50571a89d09cf10daa449f90`, FROZEN_SPEC.md blob `f0c6ba540f10e5899dcd7bf65142de23475513ce`. freeze-v1, its specification and the original plumbing artifacts are preserved unchanged. This file supersedes only the clauses listed in section 8.

Training, the main comparison, reruns and S9 access remain paused until an explicit implementation/run release.

## 1. Selection procedure (S8 only, finite)

- Every run trains the full 50 epochs with no early stopping and no extension. Candidate checkpoints are epochs 10, 20, 30, 40 and 50 only.
- JEPA: pretraining settings as in freeze-v1. Probe grid: lr {0.001, 0.01} x probe epochs {50, 100}, fixed epochs, no early stopping. 5 checkpoints x 4 probe configs = 20 candidates per seed.
- Baseline: same Encoder plus Linear(128, 4), AdamW, weight decay 0.05, batch 32. lr grid {0.00025, 0.0005, 0.001, 0.002} x the same five checkpoints = 20 candidates per seed. Each lr is a separate 50-epoch run. freeze-v1 did not specify these baseline settings.
- Random-encoder control: one untrained encoder per seed, the same probe grid, 4 candidates per seed. Its checkpoint value 0 is a bookkeeping sentinel only.
- Score: accuracy over all S8 trials, the same trials for every candidate. Highest wins. Ties: earlier checkpoint epoch, then smaller lr, then fewer probe epochs. Applied per model and per seed.
- The selected checkpoint and hyperparameters are recorded before final evaluation.
- S9 is evaluated once per model per seed, for the selected candidate only. Nothing from S9 informs any choice.
- The S8 t-SNE comes from each seed's epoch-50 JEPA checkpoint before any probe is trained. It is qualitative and not used for selection.
- Planned S8 candidate evaluations: 3 seeds x (20 + 20 + 4) = 132.

## 2. Falsifier

- Seeds 42, 7 and 123. Statistic: mean S9 probe accuracy over the three seeds, using each seed's selected candidate. Comparisons use exact trial-count fractions.
- F1: JEPA mean accuracy <= 27% (inclusive). F2: baseline mean minus JEPA mean > 10 percentage points (strict).
- JEPA did not add value if F1 or F2 holds. A falsifier that does not fire is not positive evidence.
- If any of the six primary S9 scores is missing or non-finite, the comparison is not evaluable. No reduced-seed average is formed.
- The random-encoder control is reported separately as JEPA mean minus random-encoder mean and is not part of the falsifier.
- Seed standard deviation is reported as sample standard deviation (ddof=1), as a display statistic only.

## 3. Resource and inference limits

- Only the encoder is matched (407,424 parameters). Trainable parameters are not: JEPA trains encoder plus predictor (674,048) and then a 516-parameter probe, and the baseline trains 407,940. Compute is not matched. No matched-compute claim is made.
- Reported per model and seed: trainable parameters, optimizer steps, epochs, wall-clock time on the Colab T4, and the number of S8 evaluations.
- Late-loss trend flag: set when the mean training loss over epochs 41-50 is more than 2% lower than over epochs 31-40, computed from exactly 50 finite non-negative epoch losses. It is a trend flag, not a convergence certificate. The verdict uses the 50-epoch results either way.
- S9 is one held-out subject. Variation across seeds is optimization variation on this single subject and is not uncertainty over unseen people. Claims are limited to this split, task and model size.

## 4. Plumbing record

Plumbing check 2 under the original strict monotonic-loss criterion remains FAIL (17 of 19 steps decreased). The `pass` field of `2_loss` in `results/plumbing/summary.json` reflects a moving-average grading. It is preserved unchanged and is superseded by this statement.

## 5. Conventions

- S8 grid completeness: the complete planned candidate grid, scored on the same S8 trials, is required before selection. If a candidate score is missing or non-finite, the surviving subset is not silently selected from and S9 is not accessed. The comparison is recorded as not evaluable, unless an infrastructure failure is resolved under the retry rule below.
- Retry rule: a retry is allowed only for an infrastructure failure, with the same code and the same seed, and every attempt is fully logged and retained. Divergence does not authorize a retry.

## 6. Implementation requirements

- Supervised training and finite selection orchestration, with the selected checkpoint and hyperparameters recorded before final evaluation.
- Training and S8 loading are separate from S9 loading. The training and selection path never loads S9.
- Each selected fitted probe and its training-feature normalization state are saved, and final evaluation uses the saved selected candidate.
- Fixture-only checks of selection, ties, failures and threshold boundaries, plus a synthetic check that selection never loads S9 and that final evaluation uses the recorded selected candidate.

## 7. Unchanged from freeze-v1

Subject split, train-only normalization, masking rule, baseline definition, the 50-epoch limit, the probe protocol, the random-encoder control's separate role, and the plumbing record.

## 8. Clauses of FROZEN_SPEC.md superseded or filled

- "Probe epochs and lr may be tuned on S8 only": replaced by the finite grids in section 1.
- The falsifier clause "within 2 points of 25%, or more than 10 points below the supervised baseline": replaced by F1 and F2 in section 2.
- "If not converged, report that": refined by the trend flag in section 3.
- Baseline training settings, unspecified in freeze-v1: specified in section 1.

## 9. Authorization status

This file records the agreed design only. Training, the main comparison, reruns and S9 access are not authorized by it.
