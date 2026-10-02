# Frozen spec v1: JEPA on EEG motor imagery

Frozen before any main run. Any later change requires a new tagged version with a written reason.

## Data
- BCI Competition IV 2a via MOABB BNCI2014_001, 9 subjects, 4 classes, 22 channels, 250 Hz
- Both sessions per subject (576 trials for S1, 144 per class)
- Train S1-S7, validation S8 only, test S9. No subject appears in two splits (asserted in code).
- Band-pass 4-40 Hz, epoch 0-2 s after cue, crop to 500 samples
- Per-channel z-score, statistics fitted on S1-S7 only
- 10 non-overlapping patches of 22 x 50

## JEPA
- Per sample: 6 random context patches, 2 target patches from the remaining 4
- Encoder: linear patch embedding (1100 to 128), learned position embedding, 2-layer transformer (4 heads), LayerNorm
- Predictor: 2-layer transformer over context latents plus 2 mask tokens carrying target position embeddings; outputs one latent per target patch
- Target encoder: EMA copy of the encoder (momentum 0.996), no gradient
- Loss: MSE in latent space, no raw signal reconstruction
- Optimizer: AdamW, lr 5e-4, weight decay 0.05, batch 32
- Collapse monitor: mean per-dimension std of target latents each epoch
- Parameters: encoder 407,424 | predictor 266,624 | target encoder 407,424 (EMA, not trained)

## Baseline
- Same Encoder class on all 10 patches, mean-pool, Linear(128, 4), cross-entropy, trained end to end on S1-S7
- Encoder parameters identical to JEPA by construction. Baseline total 407,940.

## Evaluation
- Primary metric: frozen-encoder linear probe accuracy on S9, all 10 patches, mean-pooled, no masking
- Probe: features standardized with training-feature statistics, Linear(128, 4), Adam, lr 1e-3, 100 epochs, batch 64, trained on S1-S7 labels
- Probe epochs and lr may be tuned on S8 only. S9 is evaluated once, at the end.
- Diagnostic: t-SNE of encoder outputs on S8 colored by class (perplexity 30, random_state 42, PCA init), inspected before probe training
- Control: probe on a randomly initialized frozen encoder with the same architecture

## Protocol
- Seeds 42, 7, 123, reported as mean ± std
- Budget: 50 epochs maximum on Colab T4. If not converged, report that.
- Falsifier: JEPA probe accuracy on S9 within 2 points of 25%, or more than 10 points below the supervised baseline, means JEPA added no value for this setting.

## Plumbing run
- S1 only, 2 epochs, seed 42, batch 32
- Check 2 is reported two ways: strict monotonic decrease over the first 20 steps, and decrease of the 5-step moving average. Random masking makes strict monotonic decrease unlikely.
