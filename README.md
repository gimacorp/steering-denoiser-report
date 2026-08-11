# Reducing Negative Side-Effects of Activation Steering (GPT-2 small)

Test assignment for **T-Lab (T-Bank) - AI Research / Interpretability**.

**Task:** reduce the fluency-vs-concept trade-off of activation steering. The suggested
approach was a denoiser `D(h + α·v)`. This repo implements that plus six variants and a
LoRA adapter, and reports a diagnosed negative result together with an unexpected
safety-relevant positive result.

## Key results

- **Negative (main):** no cheap activation-space correction (6 methods) nor a LoRA adapter
  (3 variants) stably improves the Pareto front. Concept and fluency are **entangled** in
  GPT-2 small's residual stream — recover one, lose the other. A single-vector diagnostic
  localizes the failure to **autoregressive error accumulation**, not per-step denoising.
- **Positive (unexpected):** a LoRA adapter trained on LM-loss under constant steering acts
  as a **"steering scrubber"** — it neutralizes the injected concept (toxic 0.31 → 0.002)
  while restoring fluency (perplexity 84 → 23). Potential defense against activation-level
  manipulation.

## Contents

- [`REPORT.md`](REPORT.md) - full report (problem, methods, diagnostics, findings, limitations).
- `notebook.ipynb` — all experiments (export from Colab).
- `MODEL_CARD.md` — card for the released LoRA adapter.
- `figures/` — Pareto plots.

## Reproduce

Runs on a free Colab T4. Open the notebook, run top-to-bottom. Dependencies:
`transformer_lens`, `transformers`, `peft`, `datasets`, `torch`.

## Method at a glance

1. Steering: `h̃ = h + α·v` after layer 6; `v` = difference-of-means "evil" direction (BOS token excluded).
2. Metrics: `toxic-bert` (concept) + GPT-2 perplexity (fluency), both API-free.
3. Corrections tried: denoiser (Gaussian / steering-noise), directional protection,
   soft mixing, orthogonal cleaning, learned translator, LoRA adapter (3 loss variants).
4. Diagnosis + honest averaged evaluation with error bars.

## Author

Karim Gimadiev.
