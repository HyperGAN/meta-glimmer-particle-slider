# Meta/chat native PG0.9.5 learning rates

## Completed finished-output decision: six-way zero tie

All six existing saved RAW32 candidates were compared on the same two frozen
Meta/chat cases under the original native6656 NF4/BF16 source/ABI, batch2 and
horizon128. Each case has equal weight. **Every candidate's cheerful-effect
ON−OFF mean is0.0.** This is a completed comparison, not missing evidence and
not a selected beneficial rate. Keep every rate; no unique/default change,
secondary SSE tiebreak, rerating, repeated search or extra confirmation follows.

| G | d_lr_mult | prior_lr_mult | Mean effect gain |
| ---: | ---: | ---: | ---: |
| 0.0001 | 180.0 | 299.99999999999994 | 0.0 |
| 0.001 | 18.0 | 29.999999999999996 | 0.0 |
| 0.012 | 1.5 | 2.4999999999999996 | 0.0 |
| 0.036 | 0.5000000000000001 | 0.8333333333333333 | 0.0 |
| 0.000010000000000000003 | 1799.9999999999998 | 2999.9999999999986 | 0.0 |
| 0.00003 | 600.0 | 999.9999999999998 | 0.0 |

Every prospective option dictionary explicitly has `ema_decay=None`; use RAW
current weights and no model-weight EMA/averaging/comparison. Shared ParticleGAN
computes D=`lr*d_lr_mult` and bank=`lr*prior_lr_mult`. Preserve the original
literal multipliers, which produce fixed D `0.018000000000000002` and bank
`0.029999999999999995`; do not reconstruct them from rounded rates.
The [portable decision](finished-output-lr-decision.json) records all six tied
option dictionaries, retained outcomes and exact result/decision hashes, with
no private paths or weights. Its `rate_overrides` is null: it does not impose a
new default. This is documentation metadata, not a complete training session.

The only selector is `mean_cases(effect_intensity(ON)-effect_intensity(OFF))`
in finished answers. Keep negative values, zero values and all ties; technical
failures remain missing. Teacher/student SSE, preservation drift, beauty,
confidence and completion are not selectors or tiebreaks. Reuse completed
training, RAW checkpoints, outputs and first scores.

For the real standalone factory interface:

```python
from glimmer_native.lifecycle import config

# candidate_options is an explicitly chosen retained tied_rate_overrides dict.
# The zero tie recommends no new LR; this example only constructs configuration.
session_config = config(
    snapshot, meta_paired_tokens_file, admitted_gpu_uuid,
    steps=128, batch=2, seed=7, **candidate_options,
)
```

`steps=128` is the original schedule horizon; RAW32 is the compared completed
endpoint, not a32-step schedule. The original Meta `[user]` / `[system,user]`
roles, empty history, `reasoning_strength="low"` and
`add_generation_prompt=True` remain required for this scope. Do not substitute
plain-text Glimmer conditioning or pool its separate positive two-rate tie with
this six-way zero tie. The width12 CPU fixture and the **20 published PG82
controls** retain different source/loaders/recipes. No published PG82 weight is
converted or relabeled. Exact continuation always retains its complete original
session/source/data identity; these option values do not rebind a checkpoint.

## Historical teacher-response SSE screen (RAW128)

The unchanged [preset](../presets/meta-pg095-native-nf4-128.json) records the
older response-SSE-selected G0.00003 / D0.018000000000000002 /
bank0.029999999999999995, with multipliers600.0/999.9999999999998. The selected
checkpoint reached128 updates; development RAW error was0.17696601321263816
and the previously unused four-example RAW error was0.2249722810441098 versus
OFF1.0. An independent saved-field FP64 reduction reproduced those values with
zero training updates. [Original sanitized measurement](pg095-native-nf4-128-measurement.json).
This remains numerical teacher-matching history, not an intended-effect LR
winner, generated-language-quality score or instruction to repeat that work.

## Historical RAW128 factory example

Use the immutable standalone installation and caller-owned guardian/numerical
policy in the [native handoff](native-nf4-handoff.md). Do not install this
repository's historical `glimmer_slider` distribution in that environment.
The pinned standalone package retains the measured 20GiB admission and no-EMA
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
bank `0.03`; the historical preset is an explicit override, not a current
intended-effect recommendation.

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
