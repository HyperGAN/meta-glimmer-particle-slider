# Current standalone native Glimmer training

The current Meta/chat PG0.9.5 finished-output comparison is complete: all six
saved RAW32 rates tie at zero cheerful-effect ON−OFF mean, with fixed D/bank,
batch2 and horizon128. **No LR or default change is selected.**
[All tied options, actual factory interface and scope](measured-native-learning-rates.md).
Reuse the existing searches, RAWs, answers and scores; no extra confirmation,
rerating or repeated training is required. The earlier G0.00003 RAW128
teacher-response score0.2249722810441098 and mechanical continuation/RAW proofs
remain historical evidence, not a finished-answer effect recommendation.

The later separate catastrophic check leaves current eligible saved arms0/1/4,
failed2/3 and unmeasured5. All six original zero effect scores remain; no unique
LR/default is chosen. The actual FULL32-bound guards block2/3; output passes for
0/1/4 do not claim completed FULL guard binding. The cell5 observer JSON error
is not model failure. [Current eligibility and options](current-eligibility-20261109.md)
preserve all checkpoints/old results without repeated training or judgments.

The native implementation is the separate [standalone package at `b621a32`](https://github.com/HyperGAN/glimmer-particle-sliders/tree/b621a32bced45c6b5466113336bc9962bfa6f18e/native-candidate),
version 0.2.0, shared core 0.13.0 at `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`
and published ParticleGAN 0.9.5. Its native package hashes exactly match the
Meta measurement's operative bytes, including the 20GiB admission profile and
strict no-EMA checks. Use a fresh Python 3.12 environment with the pinned
Torch 2.9.1+cu128 numerical stack:

```sh
python -m pip install 'glimmer-native-response-candidate @ git+https://github.com/HyperGAN/glimmer-particle-sliders.git@b621a32bced45c6b5466113336bc9962bfa6f18e#subdirectory=native-candidate'
python -m pip check
```

The namespace remains `glimmer_native`; the factory fixes `release-0.9.5` /
`continuous`. The model, native6656 geometry, strict BF16/NF4 numerics and
original Meta user/system chat roles remain fixed. A live external guardian
must enforce the 20GiB process cap and 5GiB free reserve, with 25GiB free at
admission; package installation grants no resource reservation. Exact resume
requires the original complete SessionConfig, source and data, not this preset
alone. Full [scope and evidence](pg095-native-nf4-128-measurement.json).

The [width-12 CPU route](training-generations.md), the 20 published PG82
controls and the historical PG801 proof below keep their original scopes.
The following section records the status at that earlier proof boundary;
its sealed-final and uncalibrated-rate statements are historical.

# Historical PG801 native NF4 proof

The earlier native Glimmer experiments belong to the separate standalone
[`native-candidate` subproject at `42d1262`](https://github.com/HyperGAN/glimmer-particle-sliders/tree/42d1262b2366ae9be827e7ec0de55740294fc0c9/native-candidate).
That exact installed source has passed a bounded pretrained update/resume and
numerical-response proof. It has no qualified learning rate or semantic steering
recommendation. The old prepared G32 search was superseded by the PG0.9.5 cutover.

At the time of this proof, the default CPU training guide pinned
`0b10e8cacf6aeefc86a4106c0544d4089a576bd6`, official PG 0.9.4 and core `ddb4142`.
The historical `glimmer_slider` commands and 20 published controls retain PG82
and their original loaders. Keep all three environments separate. This handoff
does not install a native extra here, change those pins, convert existing
weights, or create another LR grid.

Provision a fresh Python 3.12 environment with the locked Torch 2.9.1+cu128
runtime, then install the distinct subproject:

```sh
python -m pip install 'glimmer-native-response-candidate @ git+https://github.com/HyperGAN/glimmer-particle-sliders.git@42d1262b2366ae9be827e7ec0de55740294fc0c9#subdirectory=native-candidate'
python -m pip check
```

Its namespace is `glimmer_native`; its shared backend is `glimmer-native`.
The source binds core `5d68473a098801ee1ab3ebbb12463daef51c6346` and exact
ParticleGAN `801ae1ead8f88b88eca8f5398137e3e1bc01ad36`, selected explicitly with
`source-801ae1e-continuous-v1` / `continuous`. This upstream source reports
version 0.9.4 but differs from the official release. The official default still
refuses this 6656-wide native domain. Source and runtime locks refuse mismatched
packages, binaries, numerical settings or assets.

The base is `meta-models/Muse-Glimmer-30B` at
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. The measured cohort uses one CUDA
device, NF4 with double quantization and BF16 storage/compute, eager attention,
and profile `glimmer-native-eager-bf16-strict-v1`. Its caller owns deterministic
algorithms, strict FP32/SDPA flags and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Offload, another device layout or precision needs separate qualification.

CPU inspection can read the original local snapshot without constructing the
model:

```sh
CUDA_VISIBLE_DEVICES='' python -m glimmer_native --snapshot /path/to/a4e59da52a7bc87ae7251dd5545c0dd437c44b68 --output metadata-readiness.json
```

This is metadata inspection only. Actual loading requires verified original
assets and a caller-owned live guardian with physical leases, source checks,
device and host admission, process memory caps, and parent-lifetime guards.
Installing the package grants no GPU reservation.

The [handoff receipt](native-nf4-handoff.json) records exact source/profile and
retained campaign report hashes. The two-update native proof established:

- Fresh actual generator, critic, 128×4 bank, RNG and teacher-cache identity;
  all 512 bank values and nonbank generator/critic parameters moved.
- Exact full-state and metric continuation for two updates versus fresh 1+1,
  with the packed frozen host unchanged.
- Exact unhooked OFF and native hidden/logit/eight-greedy-token replay. The
  readout uses the native full 3D head, output multiplier and tanh softcap.
- Separate portable raw and explicitly requested diagnostic EMA numerical
  response recovery, with training constructors forbidden. EMA defaults to
  disabled; the proof's decay 0.9 was explicit.

The first readout-oracle failure and later public-render harness failure remain
preserved. Response-only recovery passed with the original successful artifacts;
it does not qualify the public render API. The eight-token check establishes numerical
replay; language quality remains unqualified. Multimodal training and
historical artifact compatibility were not established.

The superseded PG801 matched-search plan was: G 0.0001/0.001/0.012/0.036,
32 updates each on a common 128 schedule, raw selection and EMA disabled. Its
actual common D is 0.018000000000000002 and bank rate 0.029999999999999995; this
is a distinct frozen cohort from the two-update proof. That superseded historical
plan ranked and confirmed candidates using held-out raw teacher/student relative
SSE: operands were cast to FP64 before subtraction, positions were pooled within
each example, ratios used teacher-edit energy and examples had equal weight.
Its teacher-only valid/excluded set, undefined zero-energy examples and equal
completed-update/horizon comparison were part of that historical contract.
This is not a prospective LR-selection instruction.

Actual initialization, stochastic and source/model/data/numerical identity,
optimizer rates, finite native updates, exact continuation and raw-response replay
remain correctness checks. Native outputs and preservation are optional
diagnostics and never select, reject or delay a rate; endpoint net displacement
is also diagnostic after the native mechanical proof. Follow the
[common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).
Four final pairs remain sealed. There is no tuned rate, default change or
independent meta-Glimmer search to inherit yet.

The report hashes in the receipt refer to retained local campaign artifacts.
This documentation change publishes a summary, without the large checkpoints
or an independently downloadable reproduction bundle. The pinned standalone
README retains its earlier CPU-source scope; the later measured evidence and
its limits are recorded here without rewriting that older source cohort.
