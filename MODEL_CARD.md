---
license: mit
base_model: gpt2
tags:
  - lora
  - peft
  - activation-steering
  - interpretability
  - safety
---

# LoRA "Steering Scrubber" for GPT-2 small

A small LoRA adapter (r=16, ~737K params, 0.59% of GPT-2 small) trained to **neutralize
activation steering** while preserving fluency.

## What it does

The adapter sits on the MLP projections of layers 6–11. It was trained with a constant
steering hook active at layer 6 (a strong "evil"/toxicity shift `α·v`), on a
language-modeling objective (plus a bounded term keeping the concept projection near its
natural target). As a result, at inference the adapter **absorbs** the injected steering:
generations become fluent and the injected concept is driven close to zero.

| α (steering) | concept (toxic) without adapter | with adapter | perplexity without | with |
|---|---|---|---|---|
| 30 | 0.312 | 0.002 | 84.4 | 22.6 |
| 50 | 0.446 | 0.001 | 156.6 | 33.4 |
| 70 | 0.520 | 0.005 | 307.8 | 44.7 |

## Intended use

Research only. Two framings:
- **Negative result:** demonstrates that a naive weight-space corrector, trained on LM-loss,
  suppresses an injected concept rather than making it coexist with fluency — evidence that
  concept and fluency are entangled in GPT-2 small.
- **Positive (safety):** a lightweight *defense* that scrubs activation-level manipulation.

## Limitations

GPT-2 small only; a single toxicity direction; single intervention layer; proxy metrics
(toxic-bert, GPT-2 perplexity). Not a production safety tool.

## How to load

```python
from transformers import GPT2LMHeadModel
from peft import PeftModel

base = GPT2LMHeadModel.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base, "<your-hf-username>/gpt2-steering-scrubber")
```

## Author

Karim Gimadiev - Innopolis University. Test assignment for T-Lab (T-Bank) AI Research.
