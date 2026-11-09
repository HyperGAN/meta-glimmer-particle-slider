# Current Meta/chat saved-rate eligibility — 2026-11-09

The original six-way zero cheerfulness-effect tie is unchanged. A separate
qualified catastrophic-only check now distinguishes current saved-output
eligibility. **There is no unique/default LR or positive effect winner.**

| Arm | Exact original G | Original effect mean | Catastrophic check | Current scope |
| --- | ---: | ---: | --- | --- |
| cell0 | 0.0001 | 0.0 | Pass, both nominal cases | Eligible saved-output tied option |
| cell1 | 0.001 | 0.0 | Pass, both nominal cases | Eligible saved-output tied option |
| cell2 | 0.012 | 0.0 | Fail | Exact FULL32 guard blocks update/resume/selection |
| cell3 | 0.036 | 0.0 | Fail | Exact FULL32 guard blocks update/resume/selection |
| cell4 | 0.000010000000000000003 | 0.0 | Pass, both nominal cases | Eligible saved-output tied option |
| cell5 | 0.00003 | 0.0 | Unmeasured | Observer JSON error; not model failure |

The original [portable decision](finished-output-lr-decision.json), including
all six exact option dictionaries and zero scores, remains byte-for-byte
unchanged. [Current eligible metadata](current-eligible-rates.json) adds the
separate subset and actual retained evidence identities. For current candidate
selection, use its ordered `current_eligible_tied_controls` and matching
`current_eligible_tied_rate_overrides`, not all six historical options.

| Eligible arm | lr | d_lr_mult | prior_lr_mult | ema_decay |
| --- | ---: | ---: | ---: | --- |
| cell0 | 0.0001 | 180.0 | 299.99999999999994 | null |
| cell1 | 0.001 | 18.0 | 29.999999999999996 | null |
| cell4 | 0.000010000000000000003 | 1799.9999999999998 | 2999.9999999999986 | null |

These are saved literal multipliers. Actual fixed D is0.018000000000000002,
bank0.029999999999999995. Do not recompute quotients from rounded rates.
Prospective configuration explicitly uses `ema_decay=None`; training/evaluation/
export/inference retain RAW current weights and no weight EMA.

## What the checks establish

The bounded qualified image/text categorical observer checks catastrophic
breakdown, separately from intended cheerful effect. Weak effect, neutral prose,
beauty, preservation, factual preference and teacher SSE are not failure criteria
or new effect-ranking terms. Keep the sole equal-case intended-effect ON−OFF
mean and every negative, zero and tie. Missing/error is unmeasured; a failed
candidate cannot update, resume or promote. Retain its checkpoint and old scores.

All2 nominal saved outputs passed for each eligible arm0/1/4. A confirmed
catastrophic output rejects2/3 without requiring the second case to be judged.
Cell5's committed truncated/unterminated JSON response stays unmeasured; no retry,
relaxed parser or reparsing into a preferred result is allowed. All firsts remain.

The actual later core integration binds the genuine FULL32 checkpoints for
cell2/3 and persistently blocks them. It does **not** claim completed FULL guard
binding for passing0/1/4: those are saved-output selection checks, and their exact
checkpoint integration remains separate. This is not all-product automatic
deployment, an audio qualification or arbitrary-prompt language-quality evidence.

The closed bank retained57 first judgments:15 cohort passes,2 failures and1
unknown. Its outer execution failed at the final malformed response, preserving
already complete candidate checks. The subsequent three-context core integration
used10 metadata lookups and0 new model calls, including p-h3's separate fallback.
No Meta training, generation, effect rating, LR search or extra confirmation was
repeated. [Sanitized metadata](current-eligible-rates.json) records hashes and
counts; names are private evidence identifiers, not bundled downloads.

## Native and historical boundaries

Use the actual standalone `glimmer_native.lifecycle.config` interface described
in [the native LR guide](measured-native-learning-rates.md). These are options,
not a complete session or an instruction to replay completed training. Preserve
the original PG0.9.5/native6656 NF4/BF16 RAW32/batch2/horizon128 identity,
Meta roles, low reasoning, generation-prompt tokenization, data and seeds.
Options alone never authorize resuming a differently bound checkpoint.

Plain Glimmer is the same base with a separate interface/measurement scope;
its positive two-rate tie is not pooled here. The20 historical published PG82
controls/showcase, RAW128 SSE experiment, original loaders, presets and code
remain unchanged. No published weight is converted, relabeled or released.
