# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Authoritative chart sources: open patches, chart derivatives and covariance.

Oracles are analytic on the unit sphere: the patch is a latitude/longitude box,
``x = (cos mu cos lam, cos mu sin lam, sin mu)``; a second chart ``(lam, s)``
with ``s = sin mu`` parameterizes the same points through a different program.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.meshfree_open_surface_diffusion import (
    box_samples,
    LATLONG_CHART,
    PATCH_LOWER,
    PATCH_UPPER,
    run_crease,
)
from examples.meshfree_surface_laplace_beltrami import surface_plan
from phydrax.discretization.meshfree._stencils import LocalStencilPolicy
from phydrax.discretization.meshfree._surface import (
    PreparedSurfacePointCloud,
    SurfacePointCloudPlan,
)
from phydrax.discretization.meshfree._surface_geometry import ChartSurfaceGeometry
from phydrax.discretization.meshfree._surface_pde import SurfaceTangentCalculus
from phydrax.discretization.meshfree._surface_quadrature import (
    chart_box_atlas,
    SurfaceQuadraturePolicy,
)
from phydrax.metrix._chart import ChartTransition, CoordinateChart
from phydrax.metrix._embedded import EmbeddedChart
from phydrax.metrix._tensor import reexpress_tensor, TensorType


def _sine_chart(u: Array) -> Array:
    radial = jnp.sqrt(1 - u[1] ** 2)
    return jnp.stack((radial * jnp.cos(u[0]), radial * jnp.sin(u[0]), u[1]))


SINE_CHART = EmbeddedChart(
    CoordinateChart("sine-latlong", ("lambda", "s")), _sine_chart, 3
)


def _mixed_patch(count: int) -> tuple[PreparedSurfacePointCloud, np.ndarray]:
    """Box-face samples in the latlong chart; half the interior in the sine chart."""
    coordinates = box_samples(PATCH_LOWER, PATCH_UPPER, count, seed=4)
    low, high = np.asarray(PATCH_LOWER), np.asarray(PATCH_UPPER)
    interior = np.all((coordinates > low) & (coordinates < high), axis=1)
    second = interior & (np.arange(coordinates.shape[0]) % 2 == 0)
    stored = coordinates.copy()
    stored[second, 1] = np.sin(coordinates[second, 1])
    geometry = ChartSurfaceGeometry(
        (LATLONG_CHART, SINE_CHART),
        stored,
        chart_indices=second.astype(np.int32),
        closed=False,
        geometry_id="mixed-latlong-patch",
    )
    points = geometry.embed(geometry.chart_indices, geometry.chart_coordinates)
    prepared = SurfacePointCloudPlan(
        points,
        geometry,
        20,
        quadrature=SurfaceQuadraturePolicy(
            "chart-cubature",
            atlas=chart_box_atlas(LATLONG_CHART, low, high, source_id="latlong"),
            subdivisions=12,
        ),
        boundary=geometry.box_boundary(low, high),
        stencil_policy=LocalStencilPolicy(polynomial_degree=3, chunk_rows=128),
    ).prepare()
    return prepared, coordinates


@pytest.fixture(scope="module")
def mixed_patch() -> tuple[PreparedSurfacePointCloud, np.ndarray]:
    return _mixed_patch(13)


def test_open_chart_patch_geometry_boundary_and_area_are_authoritative(
    mixed_patch: tuple[PreparedSurfacePointCloud, np.ndarray],
) -> None:
    patch, coordinates = mixed_patch
    evidence = patch.geometry_evidence
    assert evidence.source == "chart" and not evidence.estimate_only
    assert bool(jnp.all(evidence.valid))
    np.testing.assert_allclose(patch.normals, patch.points, atol=1e-12)
    np.testing.assert_allclose(patch.geometry.mean_curvature, 2, atol=1e-10)
    boundary = patch.boundary
    assert boundary is not None
    perimeter = 2 * 1.2 * math.cos(0.6) + 2 * 1.2
    np.testing.assert_allclose(jnp.sum(boundary.measures), perimeter, rtol=1e-12)
    # Conormals are tangent and outward: at mu = +0.6 they point toward +mu.
    np.testing.assert_allclose(evidence.boundary_conormal_residual, 0, atol=1e-12)
    # Face-interior nodes only: corners own both faces and keep both conormals.
    lam_all, mu_all = coordinates[:, 0], coordinates[:, 1]
    face = (
        np.isclose(mu_all, PATCH_UPPER[1])
        & ~np.isclose(lam_all, PATCH_LOWER[0])
        & ~np.isclose(lam_all, PATCH_UPPER[0])
    )
    top = np.isin(np.asarray(boundary.nodes), np.flatnonzero(face))
    mu = coordinates[np.asarray(boundary.nodes)[top], 1]
    lam = coordinates[np.asarray(boundary.nodes)[top], 0]
    north = np.column_stack(
        (-np.sin(mu) * np.cos(lam), -np.sin(mu) * np.sin(lam), np.cos(mu))
    )
    assert np.all(
        np.sum(np.asarray(boundary.conormals)[top] * north, axis=-1) > 1 - 1e-12
    )
    area = 1.2 * 2 * math.sin(0.6)
    np.testing.assert_allclose(patch.quadrature_evidence.reference_area, area, rtol=1e-12)
    assert patch.quadrature_evidence.authoritative
    assert bool(jnp.all(patch.measures > 0))


def test_declared_chart_derivatives_match_analytic_and_refuse_implicit_sources(
    mixed_patch: tuple[PreparedSurfacePointCloud, np.ndarray],
) -> None:
    patch, _ = mixed_patch
    queries = np.asarray([[0.6, 0.1], [0.3, -0.4], [0.0, 0.0]])
    derivatives = patch.chart_derivatives(queries, ((1, 0), (0, 1), (2, 0), (1, 1)))
    assert bool(jnp.all(derivatives.valid))
    value = derivatives.apply(patch.points[:, 0])
    lam, mu = queries[:, 0], queries[:, 1]
    expected = np.column_stack(
        (
            -np.cos(mu) * np.sin(lam),
            -np.sin(mu) * np.cos(lam),
            -np.cos(mu) * np.cos(lam),
            np.sin(mu) * np.sin(lam),
        )
    )
    # First derivatives are fitted to degree 3; second ones lose an order and
    # are one-sided at the patch corner (0, 0).
    np.testing.assert_allclose(value[:, :2], expected[:, :2], atol=2e-3)
    np.testing.assert_allclose(value[:, 2:], expected[:, 2:], atol=5e-2)
    with pytest.raises(ValueError, match="authoritative declared chart"):
        surface_plan(size=64, neighbors=12).prepare().chart_derivatives(
            np.zeros((1, 2)), ((1, 0),)
        )


def test_tangent_tensor_components_are_chart_covariant(
    mixed_patch: tuple[PreparedSurfacePointCloud, np.ndarray],
) -> None:
    patch, coordinates = mixed_patch
    geometry = patch.plan.geometry
    assert isinstance(geometry, ChartSurfaceGeometry)
    calculus = SurfaceTangentCalculus(patch)
    # grad_S z for z = sin(mu): covariant (0, cos mu) in latlong and (0, 1) in sine.
    field = calculus.tangential(
        jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), patch.points.shape)
    )
    covariant = calculus.chart_components(field, TensorType(("covariant",)))
    contravariant = calculus.chart_components(field, TensorType(("contravariant",)))
    in_sine = np.asarray(geometry.chart_indices) == 1
    mu = coordinates[:, 1]
    expected_covariant = np.column_stack((0 * mu, np.where(in_sine, 1, np.cos(mu))))
    expected_contravariant = np.column_stack(
        (0 * mu, np.where(in_sine, np.cos(mu) ** 2, np.cos(mu)))
    )
    np.testing.assert_allclose(covariant, expected_covariant, atol=1e-12)
    np.testing.assert_allclose(contravariant, expected_contravariant, atol=1e-12)
    # The latlong components re-expressed by the native metrix transition agree
    # with the components computed directly in the sine chart.
    transition = ChartTransition(
        LATLONG_CHART.chart,
        SINE_CHART.chart,
        lambda u: jnp.stack((u[0], jnp.sin(u[1]))),
    )
    latlong_coordinates = jnp.asarray(coordinates[in_sine])
    vector = jnp.stack(
        (0 * latlong_coordinates[:, 1], jnp.cos(latlong_coordinates[:, 1])), axis=-1
    )
    np.testing.assert_allclose(
        reexpress_tensor(
            transition, vector, TensorType(("contravariant",)), latlong_coordinates
        ),
        np.asarray(contravariant)[in_sine],
        atol=1e-12,
    )


def _shifted_program(u: Array) -> Array:
    # Same embedding written through a different program (cos = shifted sin).
    radial = jnp.sin(u[1] + 0.5 * math.pi)
    return jnp.stack(
        (radial * jnp.sin(u[0] + 0.5 * math.pi), radial * jnp.sin(u[0]), jnp.sin(u[1]))
    )


def test_source_program_change_with_equal_samples_changes_identity(
    mixed_patch: tuple[PreparedSurfacePointCloud, np.ndarray],
) -> None:
    patch, _ = mixed_patch
    geometry = patch.plan.geometry
    assert isinstance(geometry, ChartSurfaceGeometry)
    shifted = ChartSurfaceGeometry(
        (EmbeddedChart(LATLONG_CHART.chart, _shifted_program, 3), SINE_CHART),
        geometry.chart_coordinates,
        chart_indices=geometry.chart_indices,
        closed=False,
        geometry_id=geometry.geometry_id,
    )
    np.testing.assert_allclose(
        shifted.embed(shifted.chart_indices, shifted.chart_coordinates),
        patch.points,
        atol=1e-14,
    )
    rebuilt = SurfacePointCloudPlan(
        patch.plan.points,
        shifted,
        patch.plan.neighbors,
        quadrature=patch.plan.quadrature,
        boundary=patch.plan.boundary,
        stencil_policy=patch.plan.stencil_policy,
    ).prepare()
    np.testing.assert_allclose(rebuilt.measures, patch.measures, atol=1e-12)
    assert rebuilt.prepared_id != patch.prepared_id


def test_crease_patches_keep_two_sided_normals_and_transmit_flux() -> None:
    result = run_crease(count=13)
    assert result["solver_successful"]
    np.testing.assert_allclose(
        result["normal_angle_across_crease"], math.pi / 3, atol=1e-12
    )
    assert float(result["seam_jump"]) < 1e-10
    assert float(result["maximum_error"]) < 2e-2


def test_closed_chart_declaration_refuses_boundary_and_open_requires_one() -> None:
    coordinates = box_samples(PATCH_LOWER, PATCH_UPPER, 7, seed=0)
    geometry = ChartSurfaceGeometry(LATLONG_CHART, coordinates, closed=False)
    points = geometry.embed(geometry.chart_indices, geometry.chart_coordinates)
    with pytest.raises(ValueError, match="Open surface sources require"):
        SurfacePointCloudPlan(points, geometry, 12, quadrature=SurfaceQuadraturePolicy())
    closed = ChartSurfaceGeometry(LATLONG_CHART, coordinates, closed=True)
    with pytest.raises(ValueError, match="closed surface source refuses"):
        SurfacePointCloudPlan(
            points,
            closed,
            12,
            quadrature=SurfaceQuadraturePolicy(),
            boundary=geometry.box_boundary(PATCH_LOWER, PATCH_UPPER),
        )
    with pytest.raises(ValueError, match="curve in R2/R3 or a sheet in R3"):
        ChartSurfaceGeometry(
            EmbeddedChart(CoordinateChart("bad", ("a", "b")), lambda u: u, 2),
            coordinates,
            closed=True,
        )
