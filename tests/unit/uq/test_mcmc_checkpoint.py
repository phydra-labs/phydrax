#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.uq._mcmc as mcmc_module


def _problem(center=(0.4, -0.7)):
    center_array = jnp.asarray(center)
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.zeros(2),
            priors=phx.uq.Normal(0.0, 3.0),
        ),
        lambda value: -0.5 * jnp.sum((value - center_array) ** 2),
    )


def _assert_tree_equal(left, right):
    comparisons = jax.tree_util.tree_map(jnp.array_equal, left, right)
    assert all(jax.tree_util.tree_leaves(comparisons))


@pytest.mark.parametrize("algorithm", ["nuts", "hmc"])
def test_interrupted_mcmc_resume_is_exact_and_does_not_repeat_warmup(
    tmp_path,
    monkeypatch,
    algorithm,
):
    problem = _problem()
    common = {
        "key": jr.key(919),
        "num_chains": 2,
        "num_warmup": 30,
        "num_samples": 24,
        "initial_step_size": 0.2,
        "target_acceptance_rate": 0.9,
        "chain_method": "vectorized",
        "checkpoint_id": f"exact-{algorithm}",
    }
    if algorithm == "nuts":
        sample = phx.uq.sample_nuts
        common["max_num_doublings"] = 6
    else:
        sample = phx.uq.sample_hmc
        common["num_integration_steps"] = 5

    direct = sample(problem, **common)
    checkpoint = tmp_path / f"{algorithm}.phxckpt"
    original_write = mcmc_module._write_mcmc_checkpoint

    def interrupting_write(destination, **kwargs):
        original_write(destination, **kwargs)
        if kwargs["completed"] == 8:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(mcmc_module, "_write_mcmc_checkpoint", interrupting_write)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sample(
            problem,
            **common,
            checkpoint_path=checkpoint,
            checkpoint_every=8,
        )
    monkeypatch.setattr(mcmc_module, "_write_mcmc_checkpoint", original_write)

    def adaptation_must_not_run(*args, **kwargs):
        raise AssertionError("warmup repeated during resume")

    monkeypatch.setattr(mcmc_module, "_adapt_mcmc", adaptation_must_not_run)
    resumed = sample(
        problem,
        **common,
        resume_from=checkpoint,
        checkpoint_every=8,
    )

    _assert_tree_equal(resumed.samples, direct.samples)
    _assert_tree_equal(resumed.unconstrained_samples, direct.unconstrained_samples)
    _assert_tree_equal(resumed.final_states, direct.final_states)
    assert jnp.array_equal(resumed.log_density, direct.log_density)
    assert jnp.array_equal(resumed.acceptance_rate, direct.acceptance_rate)
    assert jnp.array_equal(resumed.divergent, direct.divergent)
    assert jnp.array_equal(resumed.energy, direct.energy)
    assert jnp.array_equal(
        resumed.num_integration_steps,
        direct.num_integration_steps,
    )
    assert jnp.array_equal(
        resumed.num_trajectory_expansions,
        direct.num_trajectory_expansions,
    )
    assert jnp.array_equal(resumed.diagnostics.rhat, direct.diagnostics.rhat)
    assert jnp.array_equal(
        resumed.diagnostics.bulk_ess,
        direct.diagnostics.bulk_ess,
    )


def test_mcmc_checkpoint_rejects_incompatible_identity_and_corruption(tmp_path):
    problem = _problem()
    checkpoint = tmp_path / "state.phxckpt"
    settings = {
        "key": jr.key(920),
        "num_chains": 2,
        "num_warmup": 25,
        "num_samples": 12,
        "initial_step_size": 0.2,
        "target_acceptance_rate": 0.9,
        "max_num_doublings": 5,
        "chain_method": "vectorized",
        "checkpoint_id": "compatible-run",
    }
    phx.uq.sample_nuts(
        problem,
        **settings,
        checkpoint_path=checkpoint,
        checkpoint_every=4,
    )

    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="checkpoint id"):
        phx.uq.sample_nuts(
            problem,
            **(settings | {"checkpoint_id": "different-run"}),
            checkpoint_path=None,
            resume_from=checkpoint,
        )

    checkpoint.write_bytes(checkpoint.read_bytes()[:64])
    with pytest.raises(phx.uq.CheckpointCorruptionError, match="Cannot read"):
        phx.uq.sample_nuts(
            problem,
            **settings,
            checkpoint_path=None,
            resume_from=checkpoint,
        )


def test_mcmc_accepts_distinct_initial_positions_for_each_chain():
    problem = _problem()
    initial_positions = jnp.asarray([[-1.5, -1.0], [1.5, 1.0]])

    result = phx.uq.sample_nuts(
        problem,
        key=jr.key(921),
        num_chains=2,
        num_warmup=12,
        num_samples=4,
        initial_positions=initial_positions,
        initial_step_size=0.2,
        max_num_doublings=4,
        chain_method="vectorized",
    )

    assert result.samples.shape == (2, 4, 2)


def test_mcmc_rejects_ambiguous_or_misshaped_initial_positions():
    problem = _problem()

    with pytest.raises(ValueError, match="cannot both"):
        phx.uq.sample_nuts(
            problem,
            key=jr.key(922),
            num_chains=2,
            num_warmup=8,
            num_samples=4,
            initial_position=jnp.zeros((2,)),
            initial_positions=jnp.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="shape"):
        phx.uq.sample_nuts(
            problem,
            key=jr.key(923),
            num_chains=2,
            num_warmup=8,
            num_samples=4,
            initial_positions=jnp.zeros((3, 2)),
        )
