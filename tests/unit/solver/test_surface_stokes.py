# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Closed-surface Stokes in deformation (strain) form on the unit sphere.

Independent oracles: rigid rotations ``omega x X`` are Killing fields with zero
rate of strain. For the trace-free quadratic ``Y = x^T C x`` the vector
spherical harmonics ``grad Y`` and ``n x grad Y`` have Bochner eigenvalue ``-5``
and ``div grad Y = -6 Y``; with ``K = 1`` the identity
``2 Div E(U) = Delta_B U + K U + grad div U`` gives ``2 Div E(n x grad Y) =
-4 n x grad Y`` and ``2 Div E(grad Y) = -10 grad Y``. Hence
``2 int E:E = 4 int |n x grad Y|^2 = 4 * 6 int Y^2 = 32 pi`` for
``tr(C^2) = 5/2`` (``int Y^2 = 8 pi tr(C^2) / 15``). Tolerances are stencil
errors measured at 400 nodes, 30 neighbors, degree 4.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from examples.meshfree_open_surface_diffusion import (
    chart_patch,
    LATLONG_CHART,
    PATCH_LOWER,
    PATCH_UPPER,
)
from examples.meshfree_surface_vector_pde import harmonic_fields, sphere_calculus
from phydrax.discretization.meshfree import SurfaceTangentCalculus
from phydrax.solver import MeshfreeSurfaceStokesPlan, MeshfreeSurfaceStokesResult


VISCOSITY = 1.0
REACTION = 1.0


def _bochner_policy() -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.GMRES(restart=400),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-14, max_steps=3000),
    )


@pytest.fixture(scope="module")
def calculus() -> SurfaceTangentCalculus:
    return sphere_calculus(size=400, neighbors=30, degree=4)


@pytest.fixture(scope="module")
def plan(calculus: SurfaceTangentCalculus) -> MeshfreeSurfaceStokesPlan:
    """Default linear policy: the manufactured solve exercises its convergence."""
    return MeshfreeSurfaceStokesPlan(calculus, viscosity=VISCOSITY, reaction=REACTION)


def _relative(actual: Array, expected: Array, measures: Array) -> float:
    return float(
        jnp.sqrt(
            jnp.sum(measures[:, None] * (actual - expected) ** 2)
            / jnp.sum(measures[:, None] * expected**2)
        )
    )


def _pressure_gradient(points: Array, normals: Array) -> Array:
    """Exact surface gradient of the degree-1 harmonic ``z``."""
    return jnp.asarray([0.0, 0.0, 1.0]) - points[:, 2:3] * normals


@pytest.fixture(scope="module")
def manufactured(
    calculus: SurfaceTangentCalculus, plan: MeshfreeSurfaceStokesPlan
) -> tuple[Array, Array, Array, MeshfreeSurfaceStokesResult]:
    """Exact ``U = n x grad Y``, ``p = z + 3 Y`` and the strain-form solve."""
    surface = calculus.surface
    points, normals = surface.points, surface.normals
    harmonic, gradient, rotated = harmonic_fields(points, normals)
    # -nu (Delta_B U + K U) + r U + grad p with Delta_B U = -5 U and K = 1.
    forcing = (
        (4 * VISCOSITY + REACTION) * rotated
        + _pressure_gradient(points, normals)
        + 3 * gradient
    )
    pressure = points[:, 2] + 3 * harmonic
    return forcing, rotated, pressure, plan.solve(forcing)


@pytest.mark.parametrize(
    "omega",
    [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.5, 0.8)],
    ids=["z-axis", "x-axis", "oblique"],
)
def test_rigid_rotation_has_zero_strain_and_dissipation(
    calculus: SurfaceTangentCalculus,
    plan: MeshfreeSurfaceStokesPlan,
    omega: tuple[float, float, float],
) -> None:
    surface = calculus.surface
    rotation = jnp.cross(jnp.asarray(omega)[None], surface.points)
    strain = plan.strain(rotation)
    assert strain.shape == (surface.points.shape[0], 3, 3)
    assert float(jnp.max(jnp.abs(strain))) < 5e-4
    assert float(plan.dissipation(rotation)) < 1e-6


def test_strain_divergence_obeys_sphere_curvature_identity(
    calculus: SurfaceTangentCalculus, plan: MeshfreeSurfaceStokesPlan
) -> None:
    surface = calculus.surface
    _, gradient, rotated = harmonic_fields(surface.points, surface.normals)
    divergence = calculus.tensor_divergence
    assert (
        _relative(2 * divergence.mv(plan.strain(rotated)), -4 * rotated, surface.measures)
        < 1e-2
    )
    assert (
        _relative(
            2 * divergence.mv(plan.strain(gradient)), -10 * gradient, surface.measures
        )
        < 1e-2
    )
    # A sheared tangent field does dissipate: 2 nu int E:E = 32 pi nu.
    exact = 32 * math.pi * VISCOSITY
    assert abs(float(plan.dissipation(rotated)) - exact) < 2e-2 * exact
    stress = plan.stress(rotated, surface.points[:, 2])
    np.testing.assert_allclose(jnp.swapaxes(stress, -1, -2), stress, atol=1e-12)
    np.testing.assert_allclose(
        jnp.trace(stress, axis1=-2, axis2=-1),
        2 * VISCOSITY * jnp.trace(plan.strain(rotated), axis1=-2, axis2=-1)
        - 2 * surface.points[:, 2],
        atol=1e-12,
    )


def test_manufactured_tangent_flow_is_recovered_with_constraint_evidence(
    calculus: SurfaceTangentCalculus,
    manufactured: tuple[Array, Array, Array, MeshfreeSurfaceStokesResult],
) -> None:
    measures = calculus.surface.measures
    _, velocity, pressure, result = manufactured
    assert bool(result.successful)
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert float(result.diagnostics.relative_residual) < 1e-9
    assert _relative(result.velocity, velocity, measures) < 5e-3
    assert float(result.normal_residual) < 1e-8
    assert float(result.divergence_residual) < 1e-4
    assert abs(float(result.gauge_residual)) < 1e-3
    exact = pressure - jnp.sum(measures * pressure) / jnp.sum(measures)
    error = jnp.sqrt(
        jnp.sum(measures * (result.pressure - exact) ** 2) / jnp.sum(measures * exact**2)
    )
    assert float(error) < 1e-2
    exact_dissipation = 32 * math.pi * VISCOSITY
    assert abs(float(result.dissipation) - exact_dissipation) < 2e-2 * exact_dissipation


def test_strain_form_equals_bochner_stokes_with_curvature_shifted_reaction(
    calculus: SurfaceTangentCalculus,
    manufactured: tuple[Array, Array, Array, MeshfreeSurfaceStokesResult],
) -> None:
    forcing, _, _, strain_result = manufactured
    measures = calculus.surface.measures
    # K = 1 on the unit sphere: Bochner reaction r - nu K.
    bochner = la.solve(
        calculus.stokes_system(viscosity=VISCOSITY, reaction=REACTION - VISCOSITY),
        calculus.stokes_rhs(forcing),
        policy=_bochner_policy(),
    )
    assert bool(bochner.successful)
    velocity, constraints = bochner.value
    assert _relative(strain_result.velocity, velocity, measures) < 5e-3
    difference = strain_result.pressure - constraints[:, 0]
    assert (
        float(
            jnp.sqrt(
                jnp.sum(measures * difference**2)
                / jnp.sum(measures * constraints[:, 0] ** 2)
            )
        )
        < 5e-2
    )


def test_native_failure_status_reaches_the_result(
    calculus: SurfaceTangentCalculus,
    manufactured: tuple[Array, Array, Array, MeshfreeSurfaceStokesResult],
) -> None:
    starved = MeshfreeSurfaceStokesPlan(
        calculus,
        viscosity=VISCOSITY,
        reaction=REACTION,
        linear_policy=la.LinearSolvePolicy(
            la.GMRES(restart=5),
            tolerance=la.TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=10),
        ),
    )
    result = starved.solve(manufactured[0])
    assert bool(result.finite)
    assert not bool(result.successful)
    assert int(result.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)


def test_open_surfaces_and_invalid_coefficients_are_refused(
    calculus: SurfaceTangentCalculus, plan: MeshfreeSurfaceStokesPlan
) -> None:
    patch = SurfaceTangentCalculus(
        chart_patch(LATLONG_CHART, PATCH_LOWER, PATCH_UPPER, count=8)
    )
    with pytest.raises(ValueError, match="closed surfaces"):
        MeshfreeSurfaceStokesPlan(patch, viscosity=1.0)
    for viscosity in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="viscosity"):
            MeshfreeSurfaceStokesPlan(calculus, viscosity=viscosity)
    with pytest.raises(ValueError, match="reaction"):
        MeshfreeSurfaceStokesPlan(calculus, viscosity=1.0, reaction=-1.0)
    with pytest.raises(ValueError, match="shape"):
        plan.solve(jnp.zeros((3, 3)))
