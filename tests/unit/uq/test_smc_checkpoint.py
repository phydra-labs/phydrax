#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.uq._smc as smc_module


def _problem(observation=1.1):
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            priors=phx.uq.Normal(0.0, 2.0),
        ),
        lambda value: -0.5 * ((value - observation) / 0.25) ** 2,
    )


def _assert_tree_equal(left, right):
    comparisons = jax.tree_util.tree_map(jnp.array_equal, left, right)
    assert all(jax.tree_util.tree_leaves(comparisons))


@pytest.mark.parametrize("resampling_method", ["systematic", "stratified"])
def test_interrupted_smc_resume_is_exact_and_does_not_resample_prior(
    tmp_path,
    monkeypatch,
    resampling_method,
):
    problem = _problem()

    def deterministic_prior(key, count):
        return 2.0 * jr.normal(key, (count,))

    settings = {
        "key": jr.key(930),
        "num_particles": 96,
        "prior_position_sampler": deterministic_prior,
        "target_ess": 0.85,
        "num_mcmc_steps": 2,
        "step_size": 0.12,
        "num_integration_steps": 5,
        "max_tempering_steps": 20,
        "resampling_method": resampling_method,
        "checkpoint_id": f"exact-{resampling_method}",
    }
    direct = phx.uq.sample_tempered_smc(problem, **settings)
    checkpoint = tmp_path / f"{resampling_method}.phxckpt"
    original_write = smc_module._write_smc_checkpoint

    def interrupting_write(destination, **kwargs):
        original_write(destination, **kwargs)
        if kwargs["completed"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(smc_module, "_write_smc_checkpoint", interrupting_write)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        phx.uq.sample_tempered_smc(
            problem,
            **settings,
            checkpoint_path=checkpoint,
        )
    monkeypatch.setattr(smc_module, "_write_smc_checkpoint", original_write)

    def prior_must_not_run(key, count):
        raise AssertionError("prior sampler repeated during resume")

    resumed = phx.uq.sample_tempered_smc(
        problem,
        **(settings | {"prior_position_sampler": prior_must_not_run}),
        resume_from=checkpoint,
    )

    _assert_tree_equal(resumed.state, direct.state)
    _assert_tree_equal(resumed.samples, direct.samples)
    _assert_tree_equal(resumed.unconstrained_samples, direct.unconstrained_samples)
    assert jnp.array_equal(resumed.final_weights, direct.final_weights)
    assert jnp.array_equal(resumed.temperatures, direct.temperatures)
    assert jnp.array_equal(
        resumed.effective_sample_sizes,
        direct.effective_sample_sizes,
    )
    assert jnp.array_equal(resumed.acceptance_rates, direct.acceptance_rates)
    assert jnp.array_equal(resumed.divergence_rates, direct.divergence_rates)
    assert jnp.array_equal(resumed.log_evidence, direct.log_evidence)
    assert resumed.num_unique_initial_particles == direct.num_unique_initial_particles


def test_smc_checkpoint_rejects_incompatible_identity_and_corruption(tmp_path):
    problem = _problem()
    checkpoint = tmp_path / "state.phxckpt"
    settings = {
        "key": jr.key(931),
        "num_particles": 64,
        "target_ess": 0.8,
        "num_mcmc_steps": 1,
        "step_size": 0.1,
        "num_integration_steps": 4,
        "max_tempering_steps": 20,
        "checkpoint_id": "compatible-smc",
    }
    phx.uq.sample_tempered_smc(
        problem,
        **settings,
        checkpoint_path=checkpoint,
    )

    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="checkpoint id"):
        phx.uq.sample_tempered_smc(
            problem,
            **(settings | {"checkpoint_id": "different-smc"}),
            resume_from=checkpoint,
        )

    checkpoint.write_bytes(checkpoint.read_bytes()[:64])
    with pytest.raises(phx.uq.CheckpointCorruptionError, match="Cannot read"):
        phx.uq.sample_tempered_smc(
            problem,
            **settings,
            resume_from=checkpoint,
        )
