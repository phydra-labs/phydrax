# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from examples.meshfree_open_surface_diffusion import (
    box_samples,
    LATLONG_CHART,
    PATCH_LOWER,
    PATCH_UPPER,
)
from examples.meshfree_surface_laplace_beltrami import surface_plan
from phydrax.discretization.meshfree._neighbors import MeshfreeNeighborhoodPlan
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import (
    ChartSurfaceGeometry,
    SurfaceGeometryEvaluation,
)
from phydrax.discretization.meshfree._surface_quadrature import (
    chart_box_atlas,
    SurfaceQuadraturePolicy,
)
from phydrax.sparse import RowRelation


def _open_patch(count: int) -> tuple[SurfaceGeometryEvaluation, RowRelation]:
    coordinates = box_samples(PATCH_LOWER, PATCH_UPPER, count, seed=7)
    geometry = ChartSurfaceGeometry(LATLONG_CHART, coordinates, closed=False)
    points = geometry.embed(geometry.chart_indices, geometry.chart_coordinates)
    relation = MeshfreeNeighborhoodPlan(points, 16).prepare().relation
    evaluation = geometry.evaluate(
        points, relation, boundary=geometry.box_boundary(PATCH_LOWER, PATCH_UPPER)
    )
    return evaluation, relation


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


def test_chart_cubature_refinement_campaign_against_analytic_integral() -> None:
    # int_patch exp(z) dA = 1.2 (exp(sin 0.6) - exp(-sin 0.6)), independent of nodes.
    exact = 1.2 * (np.exp(np.sin(0.6)) - np.exp(-np.sin(0.6)))
    area = 1.2 * 2 * np.sin(0.6)
    atlas = chart_box_atlas(LATLONG_CHART, PATCH_LOWER, PATCH_UPPER, source_id="latlong")
    errors = []
    for count in (7, 11, 15, 19):
        evaluation, relation = _open_patch(count)
        measures, evidence = SurfaceQuadraturePolicy(
            "chart-cubature", atlas=atlas, subdivisions=12
        ).prepare(evaluation, relation)
        assert evidence.authoritative and not evidence.geometric_estimate
        assert bool(evidence.positive)
        np.testing.assert_allclose(evidence.reference_area, area, rtol=1e-12)
        np.testing.assert_allclose(evidence.total_area, area, rtol=1e-10)
        assert float(evidence.transfer_residual) < 1e-3
        errors.append(
            abs(float(jnp.sum(measures * jnp.exp(evaluation.points[:, 2]))) - exact)
        )
    assert errors[-1] < errors[0] / 10
    assert errors[-1] < 1e-5


def test_tangent_voronoi_is_a_bounded_estimate_and_refuses_open_patches() -> None:
    prepared = surface_plan(size=128, neighbors=16).prepare()
    measures, evidence = SurfaceQuadraturePolicy().prepare(
        prepared.geometry, prepared.relation
    )
    assert evidence.geometric_estimate and not evidence.authoritative
    assert bool(jnp.isnan(evidence.reference_area))
    assert bool(jnp.all(measures > 0))
    evaluation, relation = _open_patch(7)
    with pytest.raises(ValueError, match="Unbounded tangent Voronoi"):
        SurfaceQuadraturePolicy().prepare(evaluation, relation)
    with pytest.raises(ValueError, match="chart-cubature integrates its atlas"):
        SurfaceQuadraturePolicy(
            "chart-cubature",
            atlas=chart_box_atlas(LATLONG_CHART, PATCH_LOWER, PATCH_UPPER, source_id="x"),
            total_area=1.0,
        )
