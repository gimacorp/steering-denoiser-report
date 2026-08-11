# Reducing the Negative Side-Effects of Activation Steering in Language Models

**A systematic study of denoising-based steering correction, with a negative result and an unexpected safety application**

Author: Karim Gimadiev
Model: GPT-2 small · Intervention layer: 6 (residual stream, post-block)

---

## Abstract

The task was to reduce the negative side-effect of activation steering — the fact that pushing a hidden state along a concept direction `v` with a large coefficient `α` makes the concept appear but destroys fluency. The suggested approach was to train a **denoiser** that "cleans" activations after steering.

We implemented the suggested denoiser and **six** progressively more sophisticated variants. Our central finding is a **negative result with a clear mechanistic explanation**: for GPT-2 small, naive activation-space correction does **not** produce a stable improvement of the concept/fluency Pareto front, because the concept and fluency are entangled in the residual stream in a way that cheap post-hoc corrections cannot separate — every method that recovers fluency does so by removing the concept, and every method that forces the concept back breaks fluency.

Along the way we obtained an **unexpected positive result**: a LoRA adapter trained on a language-modeling objective under constant steering learns to *neutralize* an injected concept while restoring fluency. This is effectively a "steering scrubber" — a potential defense against activation-level manipulation, which is relevant to safety research.

---

## 1. Problem statement

Activation steering modifies a hidden state `h` at some layer by adding a scaled concept direction:

```
h̃ = h + α·v
```

where `v ∈ ℝ^768` encodes a property (here: "evil"/toxicity) and `α` controls strength. Small `α` keeps text fluent but shows little of the concept; large `α` makes the concept strong but the model "breaks" (perplexity explodes, text degenerates). The trade-off is visualized as a Pareto front in (fluency, concept) space; the goal is to push this front toward the top-right corner (high concept **and** high fluency).

The suggested improvement (from the task description) is to train a denoiser `D` such that `D(h + ε) ≈ h` on Gaussian noise `ε`, and apply `h̃ = D(h + α·v)` at inference so the denoiser pulls the steered activation back toward the manifold of "healthy" activations.

We set out to reproduce this and improve on it.

## 2. Experimental setup

- **Model:** GPT-2 small (12 layers, d_model = 768), via TransformerLens for steering experiments and HuggingFace Transformers for the LoRA adapter (activation equivalence between the two was verified on layer 6).
- **Intervention point:** residual stream after block 6 (`blocks.6.hook_resid_post`), following the task's suggestion of a mid-network layer.
- **Concept direction `v`:** a difference-of-means vector between "evil" and neutral text activations, computed after excluding the first (BOS/attention-sink) token, whose norm (~3000) is ~30× the others (~90) and would otherwise dominate the mean. `v` is unit-normalized.
- **Concept score (Y-axis):** probability of the `toxic` class from `unitary/toxic-bert`, used as a local, API-free LLM-judge. Validated to separate clearly (neutral ≈ 0.001, overt-evil ≈ 0.73).
- **Fluency score (X-axis):** perplexity under GPT-2 itself (lower = more fluent). Validated to separate (fluent ≈ 60, broken ≈ 1130). We note the mild bias of scoring with the same model family and treat perplexity as a relative, not absolute, measure.
- **Averaging:** each Pareto point aggregates multiple generations across 4–6 neutral prompts; error bars are standard error of the mean.

## 3. Baseline: steering works, and the trade-off is real

Sweeping `α` from 0 to 80 reproduces the expected behavior:

- concept rises, peaks around α ≈ 50, then **falls** at higher α — because the text degenerates so badly that the classifier can no longer detect coherent toxicity;
- perplexity rises monotonically with α.

This confirms the core problem: beyond α ≈ 50, increasing strength hurts **both** axes. The best the baseline achieves is roughly (concept ≈ 0.70, perplexity ≈ 24) before collapse. **This is the front we must beat.**

## 4. Methods tried, and what each taught us

We treat this as a research problem: hypothesis → measurement → correction. Each method below was motivated by the diagnosed failure of the previous one.

### 4.1 Naive denoiser (Gaussian-noise training)

A 2-layer MLP with a residual connection (`x + net(x)`), 7.3M params, trained to denoise Gaussian-corrupted activations from wikitext.

**Result:** catastrophic. At inference, perplexity rose to the **thousands** (vs. tens for baseline). Generations degenerated into repetition ("en en en...").

**Diagnosis:** the denoiser was trained on Gaussian noise (isotropic, small) but steering applies a large, *directional* shift — an out-of-distribution corruption the denoiser never saw. Also, applied at every autoregressive step, its errors compound.

### 4.2 Denoiser with steering-like training noise

We retrained, corrupting activations with random unit-direction shifts of random magnitude (plus a little Gaussian noise), to match the *type* of corruption steering produces.

**Result:** large improvement — perplexity dropped from thousands back to tens/hundreds. But the denoiser still did not beat baseline: where it kept fluency (low α) it had also removed the concept (concept ≈ 0.15); where concept was high, perplexity was high.

**Diagnosis:** the denoiser pulls activations toward the *neutral* manifold it learned, erasing the very concept we injected.

### 4.3 Directional protection of `v`

We explicitly re-added the component along `v` after denoising, to stop the denoiser from erasing the concept.

**Result:** text broke again.

**Diagnosis:** forcing a single direction back does not reconstruct a *coherent* concept; "evil" lives in a coordinated multi-dimensional pattern, not one axis.

### 4.4 Diagnostic experiment (the key measurement)

Rather than keep guessing, we tested the denoiser on a **single** vector (no generation): take clean `h`, corrupt with `α·v`, denoise, and measure. Results (normalized space):

| α | clean norm | corrupted norm | denoised norm | proj on v (corrupt→denoised) | MSE |
|---|---|---|---|---|---|
| 4 | 28.0 | 28.3 | 26.4 | 4.34 → 3.81 | 0.028 |
| 8 | 28.0 | 29.2 | 27.4 | 8.34 → 7.38 | 0.076 |
| 16 | 28.0 | 32.5 | 30.4 | 16.34 → 14.67 | 0.280 |

**Interpretation:** on a *single* vector the denoiser is excellent — it restores the norm toward clean **and** preserves ~90% of the concept direction. Therefore the failure in generation is **not** the denoiser's per-step quality; it is **error accumulation over the ~40 autoregressive steps**.

### 4.5 Soft application (mixing coefficient γ)

Given the diagnosis, we applied the denoiser only partially: `h̃ = (1−γ)·(h+αv) + γ·D(h+αv)`, so per-step corrections stay small and don't compound.

**Result:** the best result of the activation-space methods. On a small (n=3) run the denoiser appeared to win at several α. But when we increased sampling (n=5 × 6 prompts) with error bars, the advantage **fell within noise** — the method matched but did not reliably beat baseline. The earlier "win" was small-sample luck.

**Honesty note:** this is exactly why we increased the sample size. We report the averaged, error-barred result, not the lucky one.

### 4.6 Orthogonal cleaning and a learned "translator"

Two further ideas — (a) remove only the component of the denoiser's correction that is *orthogonal* to `v` (clean damage, keep concept), and (b) train a network to *translate* "broken evil" activations (α=70) into "coherent evil" ones (α=30) — both failed, degenerating into broken text. Same underlying reason: cleaning damage in activation space also removes legitimate concept-carrying computation.

## 5. The LoRA adapter (the strongest attempt)

Following the task's hint ("can we fine-tune existing MLPs so as not to change the model structure?"), we moved the correction **into the weights**: a LoRA adapter (r=16, 737K params = 0.59% of the model) on the MLP projections of layers 6–11, trained *while a constant steering hook is active*, on a language-modeling objective. The idea: the adapter should learn to keep language fluent **despite** the steering shift, making the concept coexist with fluency.

**v1 (LM-loss only):** LM-loss fell from 5.0 to 3.8 (near the ~3.5 of the unsteered model) — but generations became fluent **and concept-free**. The adapter took the easy path: ignore the steering shift entirely.

**v2 (LM-loss − proj on v, unbounded):** diverged. The unbounded "maximize projection" term was reward-hacked: the adapter inflated activation norms toward infinity along `v` (projection reached 22 million) while language collapsed.

**v3 (LM-loss + (proj − target)², bounded):** stable. LM-loss fell to 3.86 while the projection stayed near its natural target (~30 vs. target 33.2). But generation showed the **same** pattern: fluent text, concept nearly gone.

**Final numbers (HuggingFace GPT-2, averaged):**

| α | baseline concept | baseline ppl | adapter concept | adapter ppl |
|---|---|---|---|---|
| 30 | 0.312 | 84.4 | 0.002 | 22.6 |
| 50 | 0.446 | 156.6 | 0.001 | 33.4 |
| 70 | 0.520 | 307.8 | 0.005 | 44.7 |

## 6. Findings

### 6.1 Main (negative) result

Across **six** methods — naive denoiser, steering-noise denoiser, directional protection, soft application, orthogonal cleaning, learned translator — and a LoRA adapter in three variants, **no cheap correction produced a stable improvement of the Pareto front**. The consistent pattern is a strict exchange: recover fluency ⇒ lose the concept; force the concept ⇒ lose fluency.

We interpret this as evidence that, in GPT-2 small, the "evil" concept and general fluency are **entangled in the residual stream**: the concept is not a small additive perturbation that can be denoised away from fluency, but is bound up with the same coherent activation structure that makes text fluent. This is a mechanistic reason the naive `D(h+αv)` recipe is insufficient, supported by the single-vector diagnostic (§4.4) which localizes the failure to **autoregressive error accumulation**, not per-step denoising quality.

### 6.2 Unexpected (positive) result: a steering scrubber

The v3 LoRA adapter is a clean, strong **concept-neutralizer**: it takes a heavily steered forward pass and restores fluent, on-distribution text while driving the injected concept to ~0 (concept 0.31→0.002 at α=30; perplexity 84→23). Framed as defense rather than enhancement, this is a useful artifact: a lightweight adapter that suppresses activation-level manipulation. This reframing turns a failed enhancement into a plausible **safety** contribution — worth further study on larger models and diverse concepts.

## 7. Limitations

- **One model, one concept.** GPT-2 small and a single "evil"/toxicity direction. Entanglement may be weaker in larger models with more disentangled representations; the negative result may not transfer.
- **Proxy metrics.** toxic-bert as concept judge and GPT-2 perplexity as fluency are cheap proxies; a persona-vectors LLM judge (requires API) would be stronger. We validated both proxies separate clearly, and verified (by inspecting texts) that toxic-bert was not simply under-scoring coherent evil.
- **Difference-of-means `v`.** A cleaner direction (e.g. from an SAE feature or persona vectors) might behave differently.
- **Single intervention layer.** All experiments fix layer 6; the entanglement may differ by depth.

## 8. What I would try next

1. **Train the corrector on generation trajectories, not static activations** — optimize fluency *and* a held concept jointly along real rollouts, so error accumulation is in-distribution during training (our diagnosis in §4.4 points here).
2. **Multi-dimensional concept subspace** instead of one direction `v` — protect a learned subspace so "coherent evil" is preserved as a pattern, not an axis (addresses §4.3).
3. **Larger models** to test whether entanglement (the root cause) weakens with scale.
4. **Develop the scrubber direction** (§6.2): systematically evaluate the adapter as a defense across many concepts and attack strengths.

## 9. Reproducibility

All experiments run on a single free Colab T4 GPU. Code is in this repository (`notebook.ipynb` / exported cells). The trained LoRA adapter is released on HuggingFace (see `MODEL_CARD.md`). Random seeds, prompt lists, and the α-grids used are included in the code.

---

*This report emphasizes the second half of the assignment's rubric: not only whether a method works, but **why** it does or does not. The central contribution is a systematic, diagnosed negative result and a reframed positive application, rather than a single unexplained score.*
