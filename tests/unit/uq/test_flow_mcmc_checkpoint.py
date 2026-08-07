#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import io
import json
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
import phydrax.uq._flow_mcmc as flow_module


def _problem():
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(jnp.asarray(0.0), priors=phx.uq.Normal(0.0, 2.0)),
        lambda value: -0.5 * ((value - 0.4) / 0.6) ** 2,
    )


def _config(**overrides):
    settings = {
        "num_adaptation_rounds": 1,
        "num_local_adaptation_steps": 4,
        "num_global_adaptation_steps": 2,
        "num_stabilization_steps": 1,
        "num_local_steps": 1,
        "num_global_steps": 1,
        "history_capacity_per_chain": 4,
        "flow_layers": 1,
        "num_knots": 4,
        "nn_width": 8,
        "nn_depth": 1,
        "max_epochs": 1,
        "max_patience": 1,
        "batch_size": 2,
        "validation_fraction": 0.25,
    }
    settings.update(overrides)
    return phx.uq.FlowNUTSConfig(**settings)


def _assert_tree_equal(left, right):
    comparisons = jax.tree_util.tree_map(jnp.array_equal, left, right)
    assert all(jax.tree_util.tree_leaves(comparisons))


def _assert_flow_arrays_equal(left, right):
    left_arrays, _ = eqx.partition(left, eqx.is_array)
    right_arrays, _ = eqx.partition(right, eqx.is_array)
    _assert_tree_equal(left_arrays, right_arrays)


def _rewrite_checkpoint(path, mutate):
    with zipfile.ZipFile(path, mode="r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    mutate(manifest, members)
    members["manifest.json"] = json.dumps(
        manifest,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=False,
    ) as archive:
        archive.writestr("manifest.json", members.pop("manifest.json"))
        for name in sorted(members):
            archive.writestr(name, members[name])


def _replace_checkpoint_array(manifest, members, name, value):
    record = manifest["arrays"][name]
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    payload = buffer.getvalue()
    members[record["member"]] = payload
    record["shape"] = list(value.shape)
    record["dtype"] = value.dtype.str
    record["sha256"] = hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("phase", "progress_field", "progress"),
    [
        ("adaptation", "completed_rounds", 1),
        ("stabilization", "completed_stabilization", 1),
        ("production", "completed_draws", 3),
    ],
)
def test_interrupted_flow_nuts_resume_is_exact_at_every_phase_boundary(
    tmp_path,
    monkeypatch,
    phase,
    progress_field,
    progress,
):
    problem = _problem()
    common = {
        "key": jr.key(20),
        "num_chains": 2,
        "num_warmup": 16,
        "num_samples": 6,
        "max_num_doublings": 4,
        "config": _config(),
        "chain_method": "vectorized",
    }
    direct = phx.uq.sample_flow_nuts(problem, **common)
    checkpoint = tmp_path / f"flow-nuts-{phase}.phxckpt"
    original_write = flow_module._write_flow_nuts_checkpoint

    def interrupting_write(destination, **kwargs):
        original_write(destination, **kwargs)
        if kwargs["phase"] == phase and kwargs[progress_field] == progress:
            raise RuntimeError("simulated flow interruption")

    monkeypatch.setattr(
        flow_module,
        "_write_flow_nuts_checkpoint",
        interrupting_write,
    )
    with pytest.raises(RuntimeError, match="simulated flow interruption"):
        phx.uq.sample_flow_nuts(
            problem,
            checkpoint_path=checkpoint,
            checkpoint_every=3,
            checkpoint_id=f"flow-resume-{phase}",
            **common,
        )
    monkeypatch.setattr(
        flow_module,
        "_write_flow_nuts_checkpoint",
        original_write,
    )

    def adaptation_must_not_run(*args, **kwargs):
        raise AssertionError("completed flow adaptation repeated during resume")

    monkeypatch.setattr(flow_module, "_adapt_mcmc", adaptation_must_not_run)
    monkeypatch.setattr(flow_module, "_fit_flow", adaptation_must_not_run)
    resumed = phx.uq.sample_flow_nuts(
        problem,
        checkpoint_every=3,
        checkpoint_id=f"flow-resume-{phase}",
        resume_from=checkpoint,
        **common,
    )

    _assert_tree_equal(resumed.samples, direct.samples)
    _assert_tree_equal(resumed.unconstrained_samples, direct.unconstrained_samples)
    _assert_flow_arrays_equal(resumed.flow, direct.flow)
    assert jnp.array_equal(resumed.log_density, direct.log_density)
    assert jnp.array_equal(resumed.acceptance_rate, direct.acceptance_rate)
    assert jnp.array_equal(resumed.divergent, direct.divergent)
    assert jnp.array_equal(
        resumed.global_acceptance_rate,
        direct.global_acceptance_rate,
    )
    assert jnp.array_equal(
        resumed.global_accepted_count,
        direct.global_accepted_count,
    )
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            resumed.training_losses,
            direct.training_losses,
            strict=True,
        )
    )


def test_flow_nuts_checkpoint_rejects_changed_configuration(tmp_path):
    problem = _problem()
    checkpoint = tmp_path / "flow-config.phxckpt"
    common = {
        "key": jr.key(21),
        "num_chains": 2,
        "num_warmup": 12,
        "num_samples": 4,
        "max_num_doublings": 4,
        "chain_method": "vectorized",
        "checkpoint_id": "flow-config",
    }
    phx.uq.sample_flow_nuts(
        problem,
        config=_config(),
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        **common,
    )

    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.uq.sample_flow_nuts(
            problem,
            config=_config(num_local_steps=2),
            resume_from=checkpoint,
            checkpoint_every=2,
            **common,
        )
    extended_settings = common | {"num_samples": 6}
    resumed = phx.uq.sample_flow_nuts(
        problem,
        config=_config(),
        resume_from=checkpoint,
        checkpoint_every=2,
        **extended_settings,
    )
    direct = phx.uq.sample_flow_nuts(
        problem,
        config=_config(),
        **extended_settings,
    )

    _assert_tree_equal(resumed.samples, direct.samples)
    assert jnp.array_equal(
        resumed.global_acceptance_rate,
        direct.global_acceptance_rate,
    )


def test_flow_checkpoint_rejects_package_array_and_dtype_tampering(tmp_path):
    problem = _problem()
    checkpoint = tmp_path / "flow-tampering.phxckpt"
    common = {
        "key": jr.key(23),
        "num_chains": 2,
        "num_warmup": 12,
        "num_samples": 4,
        "max_num_doublings": 4,
        "chain_method": "vectorized",
        "checkpoint_id": "flow-tampering",
        "config": _config(),
    }
    phx.uq.sample_flow_nuts(
        problem,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        **common,
    )
    original = checkpoint.read_bytes()

    def resume():
        return phx.uq.sample_flow_nuts(
            problem,
            resume_from=checkpoint,
            checkpoint_every=2,
            **common,
        )

    def change_flowjax_version(manifest, members):
        manifest["compatibility"]["settings"]["flowjax_version"] = "0.0.invalid"

    _rewrite_checkpoint(checkpoint, change_flowjax_version)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        resume()
    checkpoint.write_bytes(original)

    def remove_flow_array(manifest, members):
        name = next(
            name for name in manifest["arrays"] if name.startswith("flow_parameters/")
        )
        members.pop(manifest["arrays"][name]["member"])

    _rewrite_checkpoint(checkpoint, remove_flow_array)
    with pytest.raises(phx.uq.CheckpointCorruptionError):
        resume()
    checkpoint.write_bytes(original)

    def change_global_shape(manifest, members):
        record = manifest["arrays"]["global_acceptance_rate"]
        value = np.load(io.BytesIO(members[record["member"]]), allow_pickle=False)
        _replace_checkpoint_array(
            manifest,
            members,
            "global_acceptance_rate",
            np.zeros((2, 3), dtype=value.dtype),
        )

    _rewrite_checkpoint(checkpoint, change_global_shape)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        resume()
    checkpoint.write_bytes(original)

    def change_flow_dtype(manifest, members):
        candidates = []
        for name, record in manifest["arrays"].items():
            if not name.startswith("flow_parameters/"):
                continue
            value = np.load(
                io.BytesIO(members[record["member"]]),
                allow_pickle=False,
            )
            if value.dtype.kind == "f":
                candidates.append((name, value))
        name, value = candidates[0]
        dtype = np.float32 if value.dtype != np.dtype(np.float32) else np.float64
        _replace_checkpoint_array(
            manifest,
            members,
            name,
            value.astype(dtype),
        )

    _rewrite_checkpoint(checkpoint, change_flow_dtype)
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        resume()


@pytest.fixture(scope="module")
def portable_flow_result():
    return phx.uq.sample_flow_nuts(
        _problem(),
        key=jr.key(22),
        num_chains=2,
        num_warmup=12,
        num_samples=4,
        max_num_doublings=4,
        config=_config(),
        chain_method="vectorized",
    )


def test_flow_nuts_portable_export_and_arviz_include_global_statistics(
    tmp_path,
    portable_flow_result,
):
    destination = tmp_path / "flow-result.phxuq"
    phx.uq.export_result(portable_flow_result, destination)
    archive = phx.uq.read_result_archive(destination)
    inference_data = phx.uq.to_arviz(portable_flow_result)

    assert archive.kind == "flow_nuts"
    assert archive.metadata["algorithm"] == "flow_nuts"
    assert len(archive.metadata["flow_training_duration_seconds"]) == 1
    assert "global_acceptance_rate" in archive.fields
    assert "flow_parameters" in archive.trees
    assert archive.tree("samples")["<root>"].shape == (2, 4)
    assert np.array_equal(
        inference_data.sample_stats["flow_acceptance_rate"],
        np.asarray(portable_flow_result.global_acceptance_rate),
    )
    assert inference_data.posterior.sizes["chain"] == 2
    assert inference_data.posterior.sizes["draw"] == 4
