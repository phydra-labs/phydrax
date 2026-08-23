#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled(cells=16):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="distributed-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "distributed-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method
    )


def test_named_sharding_plan_validates_local_extent_and_shards_state():
    plan = phx.discretization.FiniteVolumeDecompositionPlan(
        (16,), (1,), ("x",), halo_width=2
    )
    prepared = plan.prepare((jax.devices()[0],))
    state = prepared.shard_state(jnp.ones((16, 1)))

    assert prepared.local_shape == (16,)
    assert prepared.report.device_count == 1
    assert state.sharding == prepared.cell_sharding


def test_distributed_residual_matches_local_residual_on_one_device():
    compiled = _compiled()
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (16,), (1,), ("x",), halo_width=1
    ).prepare((jax.devices()[0],))
    x = compiled.discretization.grid.structured_axes[0].interval_centers
    state = jnp.sin(2.0 * jnp.pi * x)[:, None]
    sharded = decomposition.shard_state(state)
    distributed = decomposition.residual(compiled.dynamics, 0.0, sharded)
    local = compiled(0.0, state)

    np.testing.assert_allclose(distributed, local, rtol=1e-12, atol=1e-12)


def test_periodic_halo_on_sharded_state_has_expected_wrapped_layers():
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (8,), (1,), ("x",), halo_width=2
    ).prepare((jax.devices()[0],))
    state = decomposition.shard_state(jnp.arange(8.0)[:, None])
    halo = decomposition.periodic_halo(state, 0)

    np.testing.assert_allclose(halo[:2, 0], [6.0, 7.0])
    np.testing.assert_allclose(halo[-2:, 0], [0.0, 1.0])


def test_decomposition_rejects_nondivisible_or_halo_dominated_domains():
    with pytest.raises(ValueError, match="divide exactly"):
        phx.discretization.FiniteVolumeDecompositionPlan(
            (10,), (3,), ("x",), halo_width=1
        )
    with pytest.raises(ValueError, match="smaller"):
        phx.discretization.FiniteVolumeDecompositionPlan(
            (8,), (4,), ("x",), halo_width=2
        )


@pytest.mark.skipif(
    len(jax.devices()) < 2,
    reason="requires at least two JAX devices",
)
def test_two_device_routes_and_residual_match_single_device():
    compiled = _compiled(16)
    decomposition = phx.discretization.FiniteVolumeDecompositionPlan(
        (16,), (2,), ("x",), halo_width=1
    ).prepare(jax.devices()[:2])
    x = compiled.discretization.grid.structured_axes[0].interval_centers
    state = jnp.sin(2.0 * jnp.pi * x)[:, None]
    sharded = decomposition.shard_state(state)
    distributed = decomposition.residual(compiled.dynamics, 0.0, sharded)

    assert len(decomposition.halo_routes) == 2
    assert {route.side for route in decomposition.halo_routes} == {
        "lower",
        "upper",
    }
    np.testing.assert_allclose(
        distributed, compiled(0.0, state), rtol=1e-12, atol=1e-12
    )
