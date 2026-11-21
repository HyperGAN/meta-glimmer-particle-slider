"""CPU release checks; no model, tokenizer, CUDA or historical engine."""
import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text())


def test_exact_rates_and_multipliers():
    preset = load("presets/meta-pg095-native-nf4-128.json")
    options = preset["session_options"]
    assert options == dict(lr=0.00003, d_lr_mult=600.0,
        prior_lr_mult=999.9999999999998, ema_decay=None,
        steps=128, batch=2, seed=7, pg_source_profile="release-0.9.5",
        penalty_input_domain="continuous")
    actual = preset["actual_optimizer_rates"]
    assert actual == dict(G=0.00003, D=0.018000000000000002,
                         bank=0.029999999999999995)
    assert (options["lr"] * options["d_lr_mult"]).hex() == actual["D"].hex()
    assert (options["lr"] * options["prior_lr_mult"]).hex() == actual["bank"].hex()
    assert options["lr"] * 1000 != actual["bank"]


def test_real_factory_accepts_preset_without_unsupported_kwargs():
    import torch
    from glimmer_native.lifecycle import config, configuration

    preset = load("presets/meta-pg095-native-nf4-128.json")
    options = preset["session_options"]
    accepted = ("lr", "d_lr_mult", "prior_lr_mult", "ema_decay", "steps", "batch", "seed")
    session = config("/tmp/native-snapshot", "/tmp/meta-paired-tokens.json",
        "GPU-72c1b506-891d-b8bc-b353-e020585e1c47",
        **{key: options[key] for key in accepted})
    for key, expected in options.items():
        assert session.options[key] == expected
    configuration(session.options)
    with pytest.raises(ValueError, match="forbids weight EMA"):
        config("/tmp/native-snapshot", "/tmp/meta-paired-tokens.json",
               "GPU-72c1b506-891d-b8bc-b353-e020585e1c47", ema_decay=0.9)
    assert not torch.cuda.is_initialized()


def test_measurement_matches_preset_and_scoped_scalar_confirmation():
    preset = load("presets/meta-pg095-native-nf4-128.json")
    summary = load("docs/pg095-native-nf4-128-measurement.json")
    assert summary["actual_optimizer_rates"] == preset["actual_optimizer_rates"]
    assert summary["session_options"] == preset["session_options"]
    final = summary["unused_final"]
    assert final["per_example"] == [0.1614707844203304, 0.2604740297395986,
                                   0.2116400860176853, 0.2663042239988249]
    assert sum(final["per_example"]) / 4 == final["primary"] == 0.2249722810441098
    assert (final["examples"], final["excluded"], final["native_training_updates"]) == (4, 0, 0)
    assert (final["OFF"], final["teacher_control"], final["reverse_control"]) == (1.0, 0.0, 4.0)
    assert not preset["scope"]["global_or_joint_optimum"]
    assert not summary["verification"]["materialized_EMA_weights"]


def test_native_conditioning_and_generation_boundaries_are_explicit():
    summary = load("docs/pg095-native-nf4-128-measurement.json")
    conditioning = summary["conditioning"]
    assert conditioning["final_neutral_roles"] == ["user"]
    assert conditioning["final_positive_roles"] == ["system", "user"]
    assert conditioning["final_history"] == []
    assert conditioning["reasoning_strength"] == "low"
    assert conditioning["add_generation_prompt"] is True
    assert conditioning["token_limit"] == 128 and conditioning["truncation"] is False
    generations = load("docs/training-generations.json")
    assert generations["native_current"]["commit"] == summary["native_implementation"]["commit"]
    assert generations["modern"]["learning_rates"]["native_calibrated"] is False
    assert generations["historical"]["published_controls"] == 20


def test_installed_native_package_is_the_measured_source():
    import glimmer_native

    package = Path(glimmer_native.__file__).parent
    summary = load("docs/pg095-native-nf4-128-measurement.json")
    for name, digest in summary["native_package_source_sha256"].items():
        assert "/" not in name and name.endswith((".py", ".json", ".hndl"))
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, name
    lock = json.loads((package / "source-lock.json").read_text())
    index = json.dumps(lock["product_sources"], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(index.encode()).hexdigest() == lock["product_sources_index_sha256"]


def test_factory_example_and_summary_remain_portable():
    import inspect
    import re
    from glimmer_native.lifecycle import config

    text = (ROOT / "docs/measured-native-learning-rates.md").read_text()
    block = re.search(r"```python\n(.*?)\n```", text, re.S).group(1)
    tree = ast.parse(block)
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "config")
    values = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    inspect.signature(config).bind(*values, **kwargs)
    assert kwargs["ema_decay"] is None
    summary_text = (ROOT / "docs/pg095-native-nf4-128-measurement.json").read_text()
    for private_prefix in ("/mnt/", "/ml2/", "/home/"):
        assert private_prefix not in summary_text
