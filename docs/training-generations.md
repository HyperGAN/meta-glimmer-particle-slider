# Modern training and historical reproduction

New ParticleGAN work uses the standalone modern distribution and the separate
`glimmer_modern` namespace. The implementation is maintained in
[`HyperGAN/glimmer-particle-sliders`](https://github.com/HyperGAN/glimmer-particle-sliders/pull/1),
where it was extracted from this repository's earlier modern response work.
The modern installation below is pinned to an immutable commit rather than a moving
branch. [`training-generations.json`](training-generations.json) records the
exact source and dependency boundary.

The modern factory calls the shared `ConditionalPG09` response lifecycle for
ParticleGAN 0.9.4, with the official `release-0.9.4` source profile and
`bispf_optimized` preset, a real
shared 128 × 4 particle bank, routed HNDL attention projections and readout
critic. This repository adds no GAN implementation or substitute objective.
Only the synthetic width-12, FP32 CPU protocol model is currently admitted.
Native Glimmer training, 30B weight loading and text-generation quality remain
unqualified. A CPU fixture checkpoint is not a trained Glimmer text control.

## New training destination

Use a new Python 3.11–3.14 environment, install CPU PyTorch, then install the
standalone package. Do not install this historical distribution in that
environment: the modern product's package-isolation contract requires the
`glimmer_slider` namespace to be absent.

```sh
python -m venv .venv-modern
. .venv-modern/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
python -m pip install 'meta-glimmer-modern-response @ git+https://github.com/HyperGAN/glimmer-particle-sliders.git@0b10e8cacf6aeefc86a4106c0544d4089a576bd6'
python -m pip check
```

The exact source commit makes this CPU software cohort reproducible. Its
validation applies to the recorded source, environment and protocol.
The product checks the exact Git origin and recursive source/data contents of
its shared core. Use a fresh environment: an older installation of the same
core version must not be silently retained.

This bounded example exercises the actual shared PG0.9.4 update, continuation,
and inference APIs without loading Glimmer weights:

```python
import json
from pathlib import Path

import torch
from glimmer_modern.lifecycle import cpu_config
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

torch.set_num_threads(1)
rows = {
    "train": [
        {"neutral": [1, 3], "positive": [2, 4], "history": [6, 7]},
        {"neutral": [1, 5], "positive": [2, 8], "history": [9]},
    ],
    "evaluation": [
        {"neutral": [1, 10], "positive": [2, 11], "history": [12]},
        {"neutral": [1, 13], "positive": [2, 14], "history": [15, 16]},
    ],
}
Path("pairs.json").write_text(json.dumps(rows))
config = cpu_config("pairs.json")
with open_response(config) as run:
    run.train(stop_after=2)
    run.checkpoint("modern-training.pt")
with open_response(config, resume="modern-training.pt") as run:
    run.train()
    run.evaluate(split="evaluation", strength=0.5)
    run.export_inference("modern-ema.pt", selection="ema")
with open_inference(
    "modern-ema.pt", operation="response", native_request={"ids": [1, 3, 6, 7]}
) as run:
    response = run.response(ids=[1, 3, 6, 7], strength=0.5)
```

These IDs belong to a 32-token synthetic vocabulary. `response` contains final
hidden states, not generated text. The modern product owns namespace/API,
source integrity, exact resume, raw/EMA and wrong-cohort rejection tests.

## Historical PG82 profile

The 20 published controls at
[`ntc-ai/glimmer-particle-sliders`](https://huggingface.co/ntc-ai/glimmer-particle-sliders)
belong to the original PG82 cohort. Their RAW1024 teachers and distilled PEFT
students are not relabeled or converted by a new dependency install. Continue
using the historical loader and exact source revision for these artifacts:

```sh
python -m venv .venv-legacy
. .venv-legacy/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
python -m pip install '.[legacy,test]'
git clone https://github.com/HyperGAN/particle-sliders.git ../particle-sliders-pg82
git -C ../particle-sliders-pg82 checkout 1f5b380d317b73779a8aeab63b6bec0093c2c65c
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m pytest -q --upstream=../particle-sliders-pg82
```

The unchanged `glimmer_slider` loader checks that upstream checkout and the
PG82 Git dependency, then executes the original training API. The documented
`python -m glimmer_slider train|sample|distill` and Studio commands are all
historical. Add `native` to the legacy extras only for their original optional
host dependencies. Their existing model admission rules still apply.

Keep the packages in different environments. The modern distribution requires
ParticleGAN 0.9.4 through the shared core; this repository's `legacy` extra
requires the exact PG82 Git source. A bare installation here omits PG82 and
cannot train or load historical controls until `legacy` is selected. It does
not install the modern distribution. Existing environments created before
this split remain reproducible at their original source commit; new legacy
installs must select the `legacy` extra explicitly.

The current modern cohort refuses earlier v1/v2 modern packets and historical
PG82 packets. Resume an old job with its original source and environment.
Training equivalent controls with the current objective requires a separately
qualified new run; updating source does not update existing weights.

The prior v3 target pins corrected core `ddb41423bcefc1fe76cd94d0f9b367353b3ea137`. Its [upstream installed-package receipt](https://github.com/HyperGAN/glimmer-particle-sliders/blob/52e9accfd5e90b7717bde0a027dfe6efb3f565c3/docs/merged-base-validation.json) records 80 wheel and 80 sdist checks plus the shared CLI smoke. The earlier routing validation remains tied to its original source; it is not relabeled.

## Current learning-rate interface

The current standalone target is package 0.1.1 / protocol `glimmer-modern-final-hidden-hg-hndl-v4-lr-cpu`. Pass optional `lr`, `d_lr_mult`, and `prior_lr_mult` in its common session options or `cpu_config` API. G/base defaults to 0.012; D = `lr * d_lr_mult` (default 1.5), routed bank = `lr * prior_lr_mult` (default 2.5). Inputs must be finite positive numbers; booleans are rejected. These remain inherited defaults, not pretrained Glimmer LR calibration. Changed rates require a fresh experiment configuration; full-state resume retains its original recipe.

The [distinct LR validation receipt](https://github.com/HyperGAN/glimmer-particle-sliders/blob/0b10e8cacf6aeefc86a4106c0544d4089a576bd6/docs/lr-validation.json) records 103 installed CPU tests and 9-file source/wheel/sdist parity. Old v3 and earlier routing receipts above remain tied to their original source; historical PG82 controls/weights remain unchanged.
