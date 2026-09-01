#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _hydrodynamics(*, eta=0.0):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (3.0, 3.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface_plan = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        discretization,
        jnp.full((3, 3), -1.0),
        maximum_slope=0.5,
        maximum_iterations=100,
    )
    hydrodynamics = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface_plan,
        coupling_iterations=4,
        coupling_tolerance=1.0e-7,
    ).prepare()
    state = hydrodynamics.initial_state(jnp.full((3, 3), eta))
    return hydrodynamics, state


def test_graph_geometry_has_exact_static_gcl_and_positive_support():
    hydrodynamics, state = _hydrodynamics()
    zero = jnp.zeros_like(state.eta)

    geometry = hydrodynamics.surface.geometry(0.0, state.eta, zero)
    evidence = hydrodynamics.surface.geometry_evidence(state.eta, zero)

    assert bool(evidence.valid)
    assert bool(geometry.passed)
    np.testing.assert_allclose(geometry.gcl_residual, 0.0, atol=1e-12)
    assert evidence.minimum_height > 0.0


def test_surface_volume_jacobian_reproduces_uniform_rise():
    hydrodynamics, state = _hydrodynamics()
    target = jnp.ones_like(state.eta) * hydrodynamics.surface.horizontal_area

    result = hydrodynamics.surface.solve_eta_rate(state.eta, target)

    assert bool(result.converged)
    np.testing.assert_allclose(result.eta_rate, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.reproduced_volume_rate, target, atol=1e-10)


def test_mapped_hodge_round_trip_and_positive_energy():
    hydrodynamics, state = _hydrodynamics()
    geometry = hydrodynamics.surface.geometry(0.0, state.eta, jnp.zeros_like(state.eta))
    velocity = tuple(0.01 * jnp.ones_like(value) for value in geometry.face_measures)
    momentum = hydrodynamics.surface.apply_hodge(geometry, velocity)

    restored = hydrodynamics.surface.inverse_hodge(geometry, momentum)

    assert bool(restored.converged)
    for actual, expected in zip(restored.velocity, velocity, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-8)
    assert hydrodynamics.surface.kinetic_energy(geometry, velocity) > 0.0


def test_mixed_projection_reduces_divergence():
    hydrodynamics, state = _hydrodynamics()
    geometry = hydrodynamics.surface.geometry(0.0, state.eta, jnp.zeros_like(state.eta))
    velocity = tuple(
        jnp.sin(jnp.arange(value.size).reshape(value.shape)) * 1.0e-3
        for value in geometry.face_measures
    )
    momentum = hydrodynamics.surface.apply_hodge(geometry, velocity)

    boundary = hydrodynamics.plan.boundary.stage(
        hydrodynamics.surface,
        geometry,
        state.eta,
        gravity=hydrodynamics.plan.gravity,
        density=hydrodynamics.plan.density,
    )
    result = hydrodynamics.projection.project(
        geometry, momentum, boundary, jnp.asarray(0.01)
    )

    assert bool(result.successful)
    before = jnp.sqrt(jnp.sum(geometry.cell_volumes * result.divergence_before**2))
    after = jnp.sqrt(jnp.sum(geometry.cell_volumes * result.divergence_after**2))
    assert after <= before


def test_free_surface_checkpoint_round_trip(tmp_path):
    hydrodynamics, state = _hydrodynamics()
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    method = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydrodynamics)
    target = tmp_path / "free-surface.chk"

    phx.applications.hydrodynamics.write_free_surface_checkpoint(
        target,
        hydrodynamics,
        method,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        continuation,
    )
    time, step, restored = phx.applications.hydrodynamics.read_free_surface_checkpoint(
        target, hydrodynamics, method, continuation
    )

    np.testing.assert_allclose(time, 0.0)
    assert int(step) == 0
    np.testing.assert_allclose(restored.state.eta, continuation.state.eta)
    for actual, expected in zip(
        restored.state.momentum, continuation.state.momentum, strict=True
    ):
        np.testing.assert_allclose(actual, expected)
