# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from examples.meshfree_surface_laplace_beltrami import surface_plan
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy


def test_explicit_density_area_and_reciprocal_sampling_weights() -> None:
    prepared = surface_plan(size=64, neighbors=12).prepare()
    density = jnp.linspace(1, 2, 64)
    measures, evidence = SurfaceQuadraturePolicy(
        "normalized-density", density=density, total_area=7.5
    ).prepare(prepared.geometry, prepared.relation)
    np.testing.assert_allclose(jnp.sum(measures), 7.5, rtol=1e-12)
    np.testing.assert_allclose(
        measures * density, jnp.full(64, 7.5 / jnp.sum(1 / density)), rtol=1e-12
    )
    assert bool(evidence.positive)


def test_tangent_voronoi_sphere_area_estimate_converges() -> None:
    coarse_plan = surface_plan(size=64, neighbors=16)
    fine_plan = surface_plan(size=256, neighbors=16)
    coarse = SurfacePointCloudPlan(
        coarse_plan.points, coarse_plan.geometry, 16, quadrature=SurfaceQuadraturePolicy()
    ).prepare()
    fine = SurfacePointCloudPlan(
        fine_plan.points, fine_plan.geometry, 16, quadrature=SurfaceQuadraturePolicy()
    ).prepare()
    coarse_error = abs(float(jnp.sum(coarse.measures)) - 4 * np.pi)
    fine_error = abs(float(jnp.sum(fine.measures)) - 4 * np.pi)
    assert fine_error < coarse_error
    assert fine_error / (4 * np.pi) < 0.08
    assert bool(jnp.all(fine.measures > 0))


def test_nonpositive_measures_and_implicit_area_normalization_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        SurfaceQuadraturePolicy("supplied", measures=jnp.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="explicit positive total_area"):
        SurfaceQuadraturePolicy("normalized-density", density=jnp.ones(8))
    with pytest.raises(ValueError, match="does not accept"):
        SurfaceQuadraturePolicy("tangent-voronoi", total_area=4 * np.pi)
