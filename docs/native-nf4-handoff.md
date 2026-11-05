# Standalone native NF4 handoff

New native Glimmer experiments belong to the separate standalone
[`native-candidate` subproject at `42d1262`](https://github.com/HyperGAN/glimmer-particle-sliders/tree/42d1262b2366ae9be827e7ec0de55740294fc0c9/native-candidate).
That exact installed source has passed a bounded pretrained update/resume and
numerical-response proof. It has no qualified learning rate or semantic steering
recommendation. The prepared G32 search remains pending.

The default [CPU training guide](training-generations.md) continues to pin
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

Standalone Glimmer owns the pending matched search: G 0.0001/0.001/0.012/0.036,
32 updates each on a common 128 schedule, raw selection and EMA disabled. Its
actual common D is 0.018000000000000002 and bank rate 0.029999999999999995; this
is a distinct frozen cohort from the two-update proof. LR selection and independent
confirmation use only held-out raw teacher/student relative SSE. Cast student,
teacher and neutral to FP64 before subtraction, pool positions within each
example, divide by its teacher-edit energy, then average examples equally.
Keep the teacher-only valid/excluded set fixed; zero-energy examples remain
undefined and student failures never remove examples. Compare equal completed
updates on the same declared schedule horizon.

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
