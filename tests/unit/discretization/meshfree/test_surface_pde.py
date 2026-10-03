# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Tangential vector/tensor operators on the unit sphere against analytic oracles.

For the trace-free quadratic harmonic ``Y``, ``grad_S Y`` and ``n x grad_S Y``
are vector spherical harmonics: Hodge eigenvalue ``-6``, Bochner eigenvalue
``-5`` (``Ric = g`` on the unit sphere, Weitzenboeck ``Delta_B = Delta_H + Ric``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from examples.meshfree_surface_vector_pde import harmonic_fields, sphere_calculus
from phydrax.discretization.meshfree._surface_pde import SurfaceTangentCalculus
from phydrax.ein import contract
from phydrax.metrix._tensor import TensorType


@pytest.fixture(scope="module")
def calculus() -> SurfaceTangentCalculus:
    return sphere_calculus(size=400, neighbors=30, degree=4)


def _relative(actual: Array, expected: Array, measures: Array) -> float:
    return float(
        jnp.sqrt(
            jnp.sum(measures[:, None] * (actual - expected) ** 2)
            / jnp.sum(measures[:, None] * expected**2)
        )
    )


def _policy(relative: float) -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.GMRES(restart=400),
        tolerance=la.TolerancePolicy(relative=relative, absolute=1e-14, max_steps=3000),
    )


@pytest.mark.parametrize("field", ["gradient", "rotated"])
def test_vector_spherical_harmonics_separate_bochner_and_hodge(
    calculus: SurfaceTangentCalculus, field: str
) -> None:
    surface = calculus.surface
    _, gradient, rotated = harmonic_fields(surface.points, surface.normals)
    vector = gradient if field == "gradient" else rotated
    hodge = calculus.vector_laplacian("hodge").mv(vector)
    bochner = calculus.vector_laplacian("bochner").mv(vector)
    assert _relative(hodge, -6 * vector, surface.measures) < 1e-2
    assert _relative(bochner, -5 * vector, surface.measures) < 1e-2
    # The curvature difference is the Ricci action, not a discretization artifact.
    assert _relative(bochner - hodge, vector, surface.measures) < 5e-2
    np.testing.assert_allclose(calculus.ricci, surface.geometry.projectors, atol=1e-12)
    np.testing.assert_allclose(calculus.gauss_curvature, 1, atol=1e-12)


def test_covariant_derivative_is_tangential_and_metric_compatible(
    calculus: SurfaceTangentCalculus,
) -> None:
    surface = calculus.surface
    projector = surface.geometry.projectors
    _, gradient, rotated = harmonic_fields(surface.points, surface.normals)
    first = calculus.covariant_gradient.mv(gradient)
    second = calculus.covariant_gradient.mv(rotated)
    np.testing.assert_allclose(
        contract("nab,nbe->nae", projector, first), first, atol=1e-12
    )
    np.testing.assert_allclose(
        contract("nae,neb->nab", first, projector), first, atol=1e-12
    )
    # Levi-Civita compatibility: d<U,V> = <nabla U, V> + <U, nabla V>.
    product = surface.surface_gradient.mv(jnp.sum(gradient * rotated, axis=-1))
    leibniz = contract("nae,na->ne", first, rotated) + contract(
        "nae,na->ne", second, gradient
    )
    assert float(jnp.max(jnp.abs(product - leibniz))) < 2e-2
    # Covariant Hessian of Y is the rank-one-up covariant derivative of grad Y.
    value, _, _ = harmonic_fields(surface.points, surface.normals)
    hessian = surface.surface_hessian.mv(value)
    assert float(jnp.max(jnp.abs(hessian - first))) < 5e-2
    rank_two = calculus.covariant_derivative(TensorType(("covariant", "covariant")))
    third = rank_two.mv(first)
    assert third.shape == (surface.points.shape[0], 3, 3, 3)
    np.testing.assert_allclose(
        contract("nab,nbcd->nacd", surface.geometry.projectors, third), third, atol=1e-12
    )


def test_tangency_constrained_hodge_helmholtz_saddle_solve(
    calculus: SurfaceTangentCalculus,
) -> None:
    surface = calculus.surface
    _, gradient, _ = harmonic_fields(surface.points, surface.normals)
    system = calculus.tangent_vector_system("hodge", diffusivity=1.0, reaction=1.0)
    result = la.solve(system, calculus.tangent_rhs(7 * gradient), policy=_policy(1e-10))
    assert bool(result.successful)
    assert _relative(result.value[0], gradient, surface.measures) < 1e-2
    assert float(jnp.max(jnp.abs(calculus.normal_constraint.mv(result.value[0])))) < 1e-9


def test_closed_surface_stokes_recovers_incompressible_tangent_flow(
    calculus: SurfaceTangentCalculus,
) -> None:
    surface = calculus.surface
    points, normals, measures = surface.points, surface.normals, surface.measures
    _, _, rotated = harmonic_fields(points, normals)
    pressure = points[:, 2]
    forcing = 6 * rotated + jnp.asarray([0.0, 0.0, 1.0]) - pressure[:, None] * normals
    system = calculus.stokes_system(viscosity=1.0, reaction=1.0)
    result = la.solve(system, calculus.stokes_rhs(forcing), policy=_policy(1e-8))
    assert bool(result.successful)
    velocity = result.value[0]
    assert _relative(velocity, rotated, measures) < 5e-3
    assert float(jnp.max(jnp.abs(calculus.divergence.mv(velocity)))) < 1e-3
    assert float(jnp.max(jnp.abs(calculus.normal_constraint.mv(velocity)))) < 1e-8
    exact = pressure - jnp.sum(measures * pressure) / jnp.sum(measures)
    computed = result.value[1][:, 0]
    error = jnp.sqrt(
        jnp.sum(measures * (computed - exact) ** 2) / jnp.sum(measures * exact**2)
    )
    assert float(error) < 0.15


def test_vector_laplacian_selector_and_invalid_coefficients_are_refused(
    calculus: SurfaceTangentCalculus,
) -> None:
    with pytest.raises(ValueError):
        calculus.vector_laplacian("rough")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="viscosity"):
        calculus.stokes_system(viscosity=0.0, reaction=1.0)
