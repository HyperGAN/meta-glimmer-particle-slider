# Measured Meta native PG0.9.5 preset

The [preset](../presets/meta-pg095-native-nf4-128.json) records the best tested
G rate for the pinned Meta Muse-Glimmer-30B NF4/BF16 cohort: G `0.00003`,
D `0.018000000000000002` and bank `0.029999999999999995`. The exact session
multipliers are `d_lr_mult=600.0` and `prior_lr_mult=999.9999999999998`.
Keep those literals: rounding the bank multiplier to `1000` changes the
actual binary rate. Model-weight EMA is disabled; train, select, export and
infer with RAW weights.

All initial candidates completed matched32, the initial finalists completed
matched64, and the declared finite lower-bound extension completed its own
matched32 and matched64 comparisons. The same selected C5 winner then reached
128 updates under the original common128 schedule, batch2 and seed7. Its
development RAW error was `0.17696601321263816`. The previously unused final
scored **`0.2249722810441098`** versus OFF `1.0`: all four examples valid,
none excluded. The independent CPU reduction exactly reproduced each saved
SSE, teacher energy, per-example ratio and equal-example mean. Final scoring
made zero training updates. The [sanitized measurement](pg095-native-nf4-128-measurement.json)
retains exact evidence and operative source digests without prompt text,
tokens, model weights or private paths.

This is a G screen with fixed D/bank in one native6656 FP32 response cohort,
not a global or joint optimum or evidence of generated-language quality.
The width-12 CPU fixture, standalone Glimmer's distinct data cohort and the
20 historical PG82 controls remain separate. Their defaults and weights are
not converted or relabeled.

## Fresh native factory call

Use the immutable standalone installation and caller-owned guardian/numerical
policy in the [native handoff](native-nf4-handoff.md). Do not install this
repository's historical `glimmer_slider` distribution in that environment.
The pinned standalone package has the measured 20GiB admission and no-EMA
guards; this repository does not duplicate or patch its trainer.

Prepare the original paired token data before opening the session. Preserve
the Meta message roles, `reasoning_strength="low"` and
`add_generation_prompt=True` when applying the native tokenizer's chat
template. The final neutral messages were `[user]`, positive messages were
`[system, user]`, and history was empty. The native contract accepts 1–128
tokens without truncation. Neither role flattening nor a plain-text pair
substitution reproduces that conditioning.

```python
from glimmer_native.lifecycle import config

session_config = config(
    "/path/to/a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
    "/path/to/meta-train-and-dev-tokens.json",
    "GPU-YOUR-ADMITTED-UUID",
    steps=128, batch=2, seed=7, ema_decay=None,
    lr=0.00003, d_lr_mult=600.0, prior_lr_mult=999.9999999999998,
)
```

The factory sets `pg_source_profile="release-0.9.5"` and
`penalty_input_domain="continuous"` itself; they are recorded in the preset
but are not factory keyword arguments. Snapshot/data paths and the exact
visible GPU UUID are required external inputs. This call constructs a
configuration; actual model admission still requires the native guardian,
source/model verification and strict numerical policy. Passing non-None EMA
is refused. Omitted LR options retain the inherited G `0.012`, D `0.018` and
bank `0.03`; the measured preset is an explicit override.

For exact checkpoint continuation, retain the original complete session,
cohort, source, prepared data and numerical identity. A rate preset alone
does not authorize resuming a differently bound checkpoint.

## Software checks

In the isolated modern native environment with pytest available, run:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -c /dev/null -q modern-tests
```

These checks exercise the real allocation-free factory and inspect the
preset, summary and pinned package bytes. They use no model weights,
tokenizer, GPU, training or final examples. The actual pretrained native
proof and saved-field confirmation are distinct evidence in the measurement.
The historical test suite still requires its original PG82 environment.
