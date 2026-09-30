#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.applications.hydrodynamics import (
    FreeSurfaceBoundaryPlan,
    GraphSurfaceALEPlan,
    MappedFreeSurfaceProjectionPlan,
    PreparedGraphSurfaceALE,
)
from phydrax.linalg import LinearSolveStatus
from phydrax.solver._mac_ale import MACALEStageGeometry


def _surface(*, maximum_iterations: int = 100) -> PreparedGraphSurfaceALE:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(2, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (2.0, 2.0, 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    return GraphSurfaceALEPlan(
        reference,
        jnp.full((2, 2), -1.0),
        maximum_iterations=maximum_iterations,
        tolerance=1.0e-11,
    ).prepare()


def _geometry(surface: PreparedGraphSurfaceALE) -> MACALEStageGeometry:
    eta = jnp.asarray(((0.0, 0.04), (0.07, -0.03)))
    return surface.geometry(0.0, eta, jnp.zeros_like(eta))


def _energy_mass(
    surface: PreparedGraphSurfaceALE, geometry: MACALEStageGeometry
) -> np.ndarray:
    coordinates = surface.hodge_coordinates
    basis = jnp.eye(coordinates.size, dtype=geometry.cell_volumes.dtype)

    def reconstruct(column: Array) -> Array:
        velocity = geometry.validate_velocity(coordinates.unflatten(column))
        return geometry.reconstruct_cell_velocity(velocity).reshape((-1,))

    reconstruction = np.asarray(jax.vmap(reconstruct, in_axes=1, out_axes=1)(basis))
    volume = np.repeat(np.asarray(geometry.cell_volumes).reshape((-1,)), 3)
    diagonal = np.asarray(coordinates.flatten(geometry.face_dual_measures))
    return 0.5 * (
        np.diag(diagonal) + reconstruction.T @ (volume[:, None] * reconstruction)
    )


def test_mapped_pairing_equals_quadratic_energy_mass() -> None:
    surface = _surface()
    geometry = _geometry(surface)
    coordinates = surface.hodge_coordinates
    values = jnp.linspace(-0.8, 1.1, coordinates.size)
    velocity = geometry.validate_velocity(coordinates.unflatten(values))
    mass = _energy_mass(surface, geometry)
    space = surface.hodge_space(geometry)
    np.testing.assert_allclose(
        space.inner(velocity, velocity), values @ mass @ values, atol=1e-12
    )
    np.testing.assert_allclose(
        coordinates.flatten(space.riesz(velocity)), mass @ values, atol=1e-12
    )
    np.testing.assert_allclose(
        coordinates.flatten(space.inverse_riesz(space.riesz(velocity))), values, atol=2e-9
    )


def test_restricted_hodge_solves_principal_mass_not_masked_inverse() -> None:
    surface = _surface()
    geometry = _geometry(surface)
    coordinates = surface.hodge_coordinates
    mass = _energy_mass(surface, geometry)
    indices = np.arange(coordinates.size)
    active = indices % 3 != 0
    mask = geometry.validate_velocity(
        coordinates.unflatten(jnp.asarray(active, dtype=jnp.float64))
    )
    rhs = jnp.sin(jnp.arange(coordinates.size, dtype=jnp.float64) + 0.4)
    momentum = geometry.validate_velocity(coordinates.unflatten(rhs))
    operator = surface.hodge_operator(geometry, free_mask=mask)
    image = coordinates.flatten(operator.mv(momentum))
    oracle_operator = np.where(active[:, None] & active[None, :], mass, 0.0)
    oracle_operator += np.diag(~active)
    np.testing.assert_allclose(image, oracle_operator @ rhs, atol=1e-12)
    result = surface.inverse_hodge(geometry, momentum, free_mask=mask)
    oracle = np.zeros(coordinates.size, dtype=np.float64)
    oracle[active] = np.linalg.solve(
        mass[np.ix_(active, active)], np.asarray(rhs)[active]
    )
    assert bool(result.successful)
    assert int(result.status) == int(LinearSolveStatus.SUCCESS)
    np.testing.assert_allclose(coordinates.flatten(result.velocity), oracle, atol=2e-9)
    np.testing.assert_array_equal(
        np.asarray(coordinates.flatten(result.velocity))[~active], 0.0
    )
    masked_full_inverse = np.linalg.solve(mass, np.asarray(rhs)) * active
    assert np.linalg.norm(oracle - masked_full_inverse) > 1e-3


def test_mapped_inverse_retains_native_nonconvergence() -> None:
    surface = _surface(maximum_iterations=1)
    geometry = _geometry(surface)
    coordinates = surface.hodge_coordinates
    rhs = jnp.sin(jnp.arange(coordinates.size, dtype=jnp.float64) + 0.4)
    result = surface.inverse_hodge(
        geometry, geometry.validate_velocity(coordinates.unflatten(rhs))
    )
    assert not bool(result.successful)
    assert not bool(result.converged)
    assert bool(result.finite)
    assert int(result.status) == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert int(result.iterations) == 1
    assert float(result.residual_norm) > surface.plan.tolerance


def test_mapped_inverse_tracks_compiled_geometry_and_implicit_derivative() -> None:
    surface = _surface()
    coordinates = surface.hodge_coordinates
    rhs = jnp.sin(jnp.arange(coordinates.size, dtype=jnp.float64) + 0.4)

    def response(height: Array) -> Array:
        eta = jnp.full((2, 2), height - 1.0)
        geometry = surface.geometry(0.0, eta, jnp.zeros_like(eta))
        momentum = geometry.validate_velocity(coordinates.unflatten(rhs))
        return coordinates.flatten(surface.inverse_hodge(geometry, momentum).velocity)

    compiled = eqx.filter_jit(response)
    first = compiled(jnp.asarray(1.0))
    second = compiled(jnp.asarray(1.4))
    np.testing.assert_allclose(second, first / 1.4, atol=2e-9)
    _, tangent = jax.jvp(compiled, (jnp.asarray(1.4),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(tangent, -second / 1.4, atol=2e-8)


def test_projection_refuses_an_unconverged_inner_hodge() -> None:
    surface = _surface(maximum_iterations=1)
    geometry = _geometry(surface)
    coordinates = surface.hodge_coordinates
    rhs = jnp.sin(jnp.arange(coordinates.size, dtype=jnp.float64) + 0.4)
    momentum = geometry.validate_velocity(coordinates.unflatten(rhs))
    boundary = FreeSurfaceBoundaryPlan().stage(
        surface,
        geometry,
        jnp.zeros(surface.eta_shape, dtype=jnp.float64),
        gravity=9.81,
        density=1000.0,
    )
    result = MappedFreeSurfaceProjectionPlan(surface, maximum_iterations=2).project(
        geometry, momentum, boundary, jnp.asarray(0.01)
    )
    assert not bool(result.tentative_hodge.successful)
    assert int(result.hodge_status) == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert not bool(result.successful)
