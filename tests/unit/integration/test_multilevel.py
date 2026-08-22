#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _hierarchy():
    levels = []
    for index, step in enumerate((0.25, 0.125, 0.0625)):
        levels.append(
            phx.stochastic.StochasticLevelSpec(
                f"level-{index}",
                index,
                refinement_axes=("time",),
                resolutions=(step,),
                state_shape=(1,),
                problem_id="toy-sde",
                observable_id="terminal",
                solver_id="coupled-toy",
                approximation_id=f"step-{step}",
                parent_level_id=None if index == 0 else f"level-{index - 1}",
            )
        )
    return phx.stochastic.StochasticCouplingPlan(
        levels,
        hierarchy_id="toy-hierarchy",
    )


def _target(*, failures=False):
    hierarchy = _hierarchy()

    def sampler(level_index, sample_indices, key):
        keys = jax.vmap(lambda index: jr.fold_in(key, index))(sample_indices)
        normal = jax.vmap(jr.normal)(keys)
        fine_step = hierarchy.levels[level_index].resolutions[0]
        fine = normal + fine_step * normal**2
        coarse = None
        if level_index > 0:
            coarse_step = hierarchy.levels[level_index - 1].resolutions[0]
            coarse = normal + coarse_step * normal**2
        valid = (
            sample_indices % 5 != 0
            if failures
            else jnp.ones(sample_indices.shape, dtype=bool)
        )
        return phx.integration.MultilevelSampleBatch(
            fine,
            coarse,
            sample_indices,
            jnp.full(sample_indices.shape, 4.0**level_index),
            level_index=level_index,
            fine_valid=valid,
            coarse_valid=valid,
            pair_ids=sample_indices,
            provenance="toy-prefix-sampler",
        )

    return phx.integration.multilevel(
        hierarchy,
        sampler,
        sampler_id="toy-prefix-sampler",
    )


def test_fixed_mlmc_runs_through_canonical_integration_dispatch():
    target = _target()
    plan = phx.integration.MultilevelMonteCarloPlan(
        samples_per_level=(4096, 2048, 1024),
        batch_size=4096,
    )
    estimate = phx.integration.integrate(
        lambda samples, level: samples,
        target,
        plan,
        key=jr.key(20),
    )

    assert estimate.successful
    assert estimate.error_kind == "mlmc-rmse-estimate"
    assert jnp.allclose(estimate.value, 0.0625, atol=5e-2)
    assert jnp.array_equal(
        estimate.diagnostics.sample_counts,
        jnp.asarray([4096, 2048, 1024]),
    )
    assert jnp.all(estimate.diagnostics.failed_counts == 0)
    assert (
        estimate.diagnostics.correction_variance_norms[2]
        < estimate.diagnostics.correction_variance_norms[1]
    )
    assert estimate.diagnostics.mean_costs[2] > estimate.diagnostics.mean_costs[1]


def test_adaptive_mlmc_allocates_by_variance_and_cost():
    target = _target()
    plan = phx.integration.MultilevelMonteCarloPlan(
        initial_samples=32,
        target_rmse=0.15,
        max_samples_per_level=20_000,
        batch_size=4096,
        max_rounds=8,
    )
    estimate = phx.integration.integrate(
        lambda samples, level: samples,
        target,
        plan,
        key=jr.key(21),
    )

    assert estimate.successful
    assert estimate.diagnostics.rmse_estimate <= 0.15
    assert jnp.all(estimate.diagnostics.sample_counts >= 32)
    assert estimate.diagnostics.sample_counts[0] > estimate.diagnostics.sample_counts[2]
    assert estimate.diagnostics.weak_convergence_order > 0.0


def test_failed_pairs_are_masked_and_replaced_by_new_prefix_indices():
    target = _target(failures=True)
    plan = phx.integration.MultilevelMonteCarloPlan(
        samples_per_level=(64, 64, 64),
        max_samples_per_level=128,
        batch_size=32,
        max_rounds=8,
    )
    estimate = phx.integration.integrate(
        lambda samples, level: samples,
        target,
        plan,
        key=jr.key(22),
    )

    assert estimate.successful
    assert jnp.all(estimate.diagnostics.sample_counts >= 64)
    assert jnp.all(estimate.diagnostics.attempted_counts > 64)
    assert jnp.all(estimate.diagnostics.failed_counts > 0)


def test_checkpoint_resume_is_bitwise_prefix_stable(tmp_path):
    target = _target()
    plan = phx.integration.MultilevelMonteCarloPlan(
        samples_per_level=(64, 64, 64),
        batch_size=16,
        max_rounds=8,
    )
    materialized = phx.integration.materialize(target, plan, key=jr.key(23))
    realization = materialized.batch
    observable = lambda samples, level: samples
    partial = phx.integration.advance_multilevel(
        observable,
        realization,
        num_rounds=1,
    )
    path = tmp_path / "mlmc.phxckpt"

    phx.integration.write_multilevel_checkpoint(path, realization, partial)
    restored = phx.integration.read_multilevel_checkpoint(path, realization)
    resumed = phx.integration.advance_multilevel(
        observable,
        realization,
        restored,
        num_rounds=7,
    )
    uninterrupted = phx.integration.advance_multilevel(
        observable,
        realization,
        num_rounds=8,
    )
    resumed_estimate = phx.integration.finalize_multilevel(realization, resumed)
    uninterrupted_estimate = phx.integration.finalize_multilevel(
        realization,
        uninterrupted,
    )

    assert jnp.array_equal(resumed.next_indices, uninterrupted.next_indices)
    assert jnp.array_equal(resumed.attempted_counts, uninterrupted.attempted_counts)
    assert jnp.array_equal(resumed_estimate.value, uninterrupted_estimate.value)
    assert jnp.array_equal(
        resumed_estimate.diagnostics.correction_variance_norms,
        uninterrupted_estimate.diagnostics.correction_variance_norms,
    )


def test_multilevel_result_archive_is_checked_and_read_only(tmp_path):
    estimate = phx.integration.integrate(
        lambda samples, level: samples,
        _target(),
        phx.integration.MultilevelMonteCarloPlan(
            samples_per_level=(32, 32, 32),
            batch_size=32,
        ),
        key=jr.key(24),
    )
    path = tmp_path / "mlmc-result.phxresult"

    phx.integration.write_multilevel_result(path, estimate)
    archive = phx.integration.read_multilevel_result(path)

    assert archive.metadata["hierarchy_id"] == "toy-hierarchy"
    assert np.array_equal(archive.array("value"), np.asarray(estimate.value))
    assert not archive.array("value").flags.writeable
    with pytest.raises(KeyError):
        archive.array("missing")
