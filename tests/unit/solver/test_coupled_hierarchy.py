#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _hierarchy(*, changing_shape=False, independent=False):
    transfer_id = "grid-5-to-3" if changing_shape else None
    coarse = phx.stochastic.StochasticLevelSpec(
        "coarse",
        0,
        refinement_axes=("time",),
        resolutions=(0.25,),
        state_shape=(3,),
        problem_id="gbm",
        observable_id="terminal",
        solver_id="euler-maruyama",
        approximation_id="dt-1/4",
    )
    fine = phx.stochastic.StochasticLevelSpec(
        "fine",
        1,
        refinement_axes=("time",),
        resolutions=(0.125,),
        state_shape=(5,) if changing_shape else (3,),
        problem_id="gbm",
        observable_id="terminal",
        solver_id="euler-maruyama",
        approximation_id="dt-1/8",
        parent_level_id="coarse",
        state_transfer_id=transfer_id,
        noise_coupling="independent" if independent else "shared",
    )
    return phx.stochastic.StochasticHierarchy(
        (coarse, fine),
        hierarchy_id="gbm-hierarchy",
    )


def _euler_level(level, realization, parent_result, transfer):
    del parent_result, transfer
    step = level.resolutions[0]
    times = jnp.arange(0.0, 1.0, step)
    increments = realization.increments(times, times + step)[..., 0]
    factors = 1.0 + 0.1 * step + 0.4 * increments
    return jnp.prod(factors, axis=-1)


def test_coupled_hierarchy_reuses_paths_and_telescopes_exactly():
    hierarchy = _hierarchy()
    realization = phx.stochastic.WienerRealization(
        jr.key(8),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(16,),
        tolerance=1e-4,
        coupling_id="gbm-coupling",
    )
    result = phx.solver.solve_coupled_hierarchy(
        hierarchy,
        realization,
        _euler_level,
        lambda output, level: output,
        cost=lambda output, level: level.refinement_index + 1.0,
    )

    assert result.realization_id == realization.realization_id
    assert all(
        level.realization_id == realization.realization_id for level in result.levels
    )
    assert jnp.allclose(
        result.corrections[1], result.levels[1].observable - result.levels[0].observable
    )
    assert jnp.allclose(
        result.corrections[0] + result.corrections[1],
        result.finest_observable,
    )
    assert jnp.allclose(result.telescoping_mean(), jnp.mean(result.finest_observable))
    assert result.total_cost_seconds == 3.0


def test_coupled_hierarchy_tracks_failed_pairs_without_repairing_them():
    hierarchy = _hierarchy()
    realization = phx.stochastic.WienerRealization(
        jr.key(9),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(4,),
        tolerance=1e-4,
    )

    def solve(level, realization, parent_result, transfer):
        values = _euler_level(level, realization, parent_result, transfer)
        return values.at[2].set(jnp.nan) if level.refinement_index == 1 else values

    result = phx.solver.solve_coupled_hierarchy(
        hierarchy,
        realization,
        solve,
        lambda output, level: output,
    )

    assert jnp.array_equal(result.correction_valid[0], jnp.ones((4,), dtype=bool))
    assert jnp.array_equal(
        result.correction_valid[1],
        jnp.asarray([True, True, False, True]),
    )
    assert not bool(result.successful[2])


def test_coupled_hierarchy_resolves_declared_state_transfer():
    hierarchy = _hierarchy(changing_shape=True)
    transfer = phx.solver.TensorGridStateTransfer(
        (5,),
        (3,),
        transfer_id="grid-5-to-3",
    )
    received = []

    def solve(level, realization, parent_result, state_transfer):
        del realization, parent_result
        received.append(state_transfer)
        return jnp.asarray(float(level.refinement_index))

    result = phx.solver.solve_coupled_hierarchy(
        hierarchy,
        None,
        solve,
        lambda output, level: output,
        state_transfers={transfer.transfer_id: transfer},
    )

    assert received == [None, transfer]
    assert result.levels[1].state_transfer_id == transfer.transfer_id


def test_coupled_hierarchy_rejects_independent_level_noise():
    hierarchy = _hierarchy(independent=True)
    with pytest.raises(ValueError, match="independent"):
        phx.solver.solve_coupled_hierarchy(
            hierarchy,
            None,
            _euler_level,
            lambda output, level: output,
        )
