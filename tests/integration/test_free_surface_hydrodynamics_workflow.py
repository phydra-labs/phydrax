#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _model(*, eta=None):
    shape = (4, 4, 3)
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (4.0, 4.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        discretization,
        jnp.full(shape[:2], -1.0),
        maximum_slope=0.5,
        maximum_iterations=150,
    )
    hydrodynamics = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        coupling_iterations=5,
        coupling_tolerance=1.0e-7,
    ).prepare()
    eta_ = jnp.zeros(shape[:2]) if eta is None else eta
    state = hydrodynamics.initial_state(eta_)
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    method = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydrodynamics)
    return hydrodynamics, method, continuation


def test_free_surface_rest_is_preserved_by_coupled_step():
    hydrodynamics, method, continuation = _model()

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.state.eta, continuation.state.eta, atol=1e-10
    )
    assert result.accepted_state.ledger.divergence_residual <= 1e-7
    assert result.accepted_state.ledger.kinematic_residual <= 1e-7


def test_small_graph_wave_advances_with_closed_volume():
    x = jnp.arange(4)[:, None]
    eta = jnp.broadcast_to(1.0e-4 * jnp.sin(2.0 * jnp.pi * x / 4), (4, 4))
    hydrodynamics, method, continuation = _model(eta=eta)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.002),
        None,
    )
    view = phx.applications.hydrodynamics.free_surface_diagnostic_view(
        hydrodynamics, result.accepted_state
    )

    assert bool(result.successful)
    assert jnp.all(jnp.isfinite(view.eta))
    assert abs(float(result.accepted_state.ledger.volume_change)) <= 1e-7


def test_uniform_scalar_content_follows_mesh_gcl():
    hydrodynamics, method, continuation = _model()
    state = continuation.state
    geometry = hydrodynamics.surface.geometry(0.0, state.eta, jnp.zeros_like(state.eta))
    scalar_content = {
        "uniform": geometry.cell_volumes * 2.0,
    }
    state = phx.applications.hydrodynamics.FreeSurfaceALEState(
        state.eta, state.momentum, scalar_content
    )
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )
    view = hydrodynamics.view(
        result.accepted_state.state,
        result.accepted_state.eta_rate,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(view.scalars["uniform"], 2.0, atol=1e-8)
