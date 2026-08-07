#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.uq._sgmcmc as sgmcmc_module


def _problem_source(*, seed=17, batch_size=3):
    data = jnp.linspace(-1.0, 1.0, 7)
    source = phx.uq.ArrayMinibatchSource(data, batch_size=batch_size, seed=seed)
    problem = phx.uq.MinibatchPosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            priors=phx.uq.Normal(0.0, 2.0),
        ),
        lambda parameter, batch: -0.5 * (batch.data - parameter) ** 2,
        num_factors=source.num_factors,
        full_log_likelihood=lambda parameter: jnp.sum(
            -0.5 * (data - parameter) ** 2
        ),
    )
    return problem, source


def _assert_tree_equal(left, right):
    comparisons = jax.tree_util.tree_map(jnp.array_equal, left, right)
    assert all(jax.tree_util.tree_leaves(comparisons))


def _assert_exact_result(left, right):
    _assert_tree_equal(left.samples, right.samples)
    _assert_tree_equal(left.unconstrained_samples, right.unconstrained_samples)
    _assert_tree_equal(left.final_states, right.final_states)
    _assert_tree_equal(left.burnin_states, right.burnin_states)
    _assert_tree_equal(left.diagnostics.rhat, right.diagnostics.rhat)
    _assert_tree_equal(left.diagnostics.bulk_ess, right.diagnostics.bulk_ess)
    _assert_tree_equal(left.diagnostics.tail_ess, right.diagnostics.tail_ess)
    assert jnp.array_equal(left.gradient_norm, right.gradient_norm)
    assert jnp.array_equal(left.log_density, right.log_density)
    if left.thermostat is None:
        assert right.thermostat is None
        assert right.momentum_norm is None
    else:
        assert jnp.array_equal(left.thermostat, right.thermostat)
        assert jnp.array_equal(left.momentum_norm, right.momentum_norm)
    assert left.num_gradient_evaluations == right.num_gradient_evaluations + 1
    assert left.mean_update_gradient_norm == right.mean_update_gradient_norm
    assert left.max_update_gradient_norm == right.max_update_gradient_norm


@pytest.mark.parametrize("interrupt_update", [2, 4, 5, 6])
def test_sgld_resume_is_exact_across_lifecycle_boundaries(
    tmp_path,
    monkeypatch,
    interrupt_update,
):
    problem, source = _problem_source()
    common = {
        "key": jr.key(30),
        "step_size": 1.0e-4,
        "num_chains": 2,
        "num_burnin": 4,
        "num_samples": 6,
        "steps_per_sample": 2,
        "checkpoint_id": f"sgld-boundary-{interrupt_update}",
    }
    direct = phx.uq.sample_sgld(problem, source, **common)
    checkpoint = tmp_path / f"sgld-{interrupt_update}.phxckpt"
    original_write = sgmcmc_module._write_sgmcmc_checkpoint

    def interrupting_write(destination, **kwargs):
        original_write(destination, **kwargs)
        if kwargs["completed_updates"] == interrupt_update:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        sgmcmc_module,
        "_write_sgmcmc_checkpoint",
        interrupting_write,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        phx.uq.sample_sgld(
            problem,
            source,
            **common,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
        )
    monkeypatch.setattr(
        sgmcmc_module,
        "_write_sgmcmc_checkpoint",
        original_write,
    )
    resumed = phx.uq.sample_sgld(
        problem,
        source,
        **common,
        resume_from=checkpoint,
        checkpoint_every=1,
    )

    _assert_exact_result(resumed, direct)


def test_sgnht_resume_preserves_momentum_thermostat_and_samples(tmp_path, monkeypatch):
    problem, source = _problem_source(seed=18)
    common = {
        "key": jr.key(31),
        "step_size": 5.0e-5,
        "diffusion": 0.02,
        "initial_thermostat": 0.03,
        "num_chains": 2,
        "num_burnin": 3,
        "num_samples": 5,
        "steps_per_sample": 2,
        "checkpoint_id": "sgnht-exact",
    }
    direct = phx.uq.sample_sgnht(problem, source, **common)
    checkpoint = tmp_path / "sgnht.phxckpt"
    original_write = sgmcmc_module._write_sgmcmc_checkpoint

    def interrupting_write(destination, **kwargs):
        original_write(destination, **kwargs)
        if kwargs["completed_updates"] == 4:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        sgmcmc_module,
        "_write_sgmcmc_checkpoint",
        interrupting_write,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        phx.uq.sample_sgnht(
            problem,
            source,
            **common,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
        )
    monkeypatch.setattr(
        sgmcmc_module,
        "_write_sgmcmc_checkpoint",
        original_write,
    )
    resumed = phx.uq.sample_sgnht(
        problem,
        source,
        **common,
        resume_from=checkpoint,
        checkpoint_every=1,
    )

    _assert_exact_result(resumed, direct)


def test_completed_sgmcmc_checkpoint_extends_without_restarting(tmp_path):
    problem, source = _problem_source(seed=19)
    checkpoint = tmp_path / "extend.phxckpt"
    common = {
        "key": jr.key(32),
        "step_size": 1.0e-4,
        "num_chains": 2,
        "num_burnin": 3,
        "steps_per_sample": 2,
        "checkpoint_id": "extend",
    }
    first = phx.uq.sample_sgld(
        problem,
        source,
        **common,
        num_samples=4,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
    )
    extended = phx.uq.sample_sgld(
        problem,
        source,
        **common,
        num_samples=7,
        resume_from=checkpoint,
        checkpoint_every=2,
    )
    direct = phx.uq.sample_sgld(
        problem,
        source,
        **common,
        num_samples=7,
    )

    assert first.samples.shape == (2, 4)
    _assert_exact_result(extended, direct)
    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="more progress"):
        phx.uq.sample_sgld(
            problem,
            source,
            **common,
            num_samples=4,
            resume_from=checkpoint,
        )


def test_sgmcmc_checkpoint_rejects_identity_source_settings_and_corruption(tmp_path):
    problem, source = _problem_source(seed=20)
    checkpoint = tmp_path / "compatibility.phxckpt"
    common = {
        "key": jr.key(33),
        "step_size": 1.0e-4,
        "num_chains": 2,
        "num_burnin": 2,
        "num_samples": 4,
        "checkpoint_id": "compatible",
    }
    phx.uq.sample_sgld(
        problem,
        source,
        **common,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
    )

    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="checkpoint id"):
        phx.uq.sample_sgld(
            problem,
            source,
            **(common | {"checkpoint_id": "other"}),
            resume_from=checkpoint,
        )
    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="source"):
        phx.uq.sample_sgld(
            problem,
            phx.uq.ArrayMinibatchSource(source.data, batch_size=3, seed=21),
            **common,
            resume_from=checkpoint,
        )
    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="settings"):
        phx.uq.sample_sgld(
            problem,
            source,
            **(common | {"step_size": 2.0e-4}),
            resume_from=checkpoint,
        )

    checkpoint.write_bytes(checkpoint.read_bytes()[:64])
    with pytest.raises(phx.uq.CheckpointCorruptionError, match="Cannot read"):
        phx.uq.sample_sgld(
            problem,
            source,
            **common,
            resume_from=checkpoint,
        )
