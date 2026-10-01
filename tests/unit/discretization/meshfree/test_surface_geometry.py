# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from examples.meshfree_surface_laplace_beltrami import sphere_points, surface_plan
from phydrax.discretization.meshfree._neighbors import MeshfreeNeighborhoodPlan
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import (
    ImplicitSurfaceGeometry,
    SampledSurfaceGeometry,
)
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.metrix._ambient import RegularLevelSetManifold


def test_implicit_sphere_projection_and_curvature() -> None:
    plan = surface_plan(size=128, neighbors=16)
    neighborhood = MeshfreeNeighborhoodPlan(plan.points, 16).prepare()
    evaluation = plan.geometry.evaluate(1.01 * plan.points, neighborhood.relation)
    np.testing.assert_allclose(jnp.linalg.norm(evaluation.points, axis=-1), 1, atol=1e-10)
    np.testing.assert_allclose(evaluation.normals, evaluation.points, atol=1e-10)
    np.testing.assert_allclose(
        evaluation.curvature_tensor, evaluation.projectors, atol=1e-9
    )
    assert bool(jnp.all(evaluation.evidence.projection_valid))
    assert bool(jnp.all(evaluation.evidence.tube_valid))


def test_samples_do_not_certify_tube_or_source_projection() -> None:
    points = sphere_points(128)
    prepared = SurfacePointCloudPlan(
        points,
        SampledSurfaceGeometry(reference_normals=points, declared_tube_radius=0.1),
        16,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density", density=jnp.ones(128), total_area=4 * np.pi
        ),
    ).prepare()
    assert not prepared.geometry_evidence.tube_certified
    assert not prepared.geometry_evidence.projection_to_source
    assert bool(jnp.all(~prepared.geometry_evidence.tube_valid))
    assert float(jnp.max(jnp.linalg.norm(prepared.normals - points, axis=-1))) < 0.08
    with pytest.raises(ValueError, match="admission failed"):
        SurfacePointCloudPlan(
            points,
            prepared.plan.geometry,
            16,
            quadrature=prepared.plan.quadrature,
            require_tube=True,
        ).prepare()


def test_open_duplicate_and_unregular_samples_are_refused() -> None:
    points = sphere_points(64)
    with pytest.raises(ValueError, match="Open"):
        SampledSurfaceGeometry(closed=False)
    geometry = surface_plan(size=64).geometry
    assert isinstance(geometry, ImplicitSurfaceGeometry)
    with pytest.raises(ValueError, match="Open"):
        ImplicitSurfaceGeometry(geometry.source, closed=False)
    with pytest.raises(ValueError, match="duplicate"):
        SurfacePointCloudPlan(
            points.at[1].set(points[0]),
            SampledSurfaceGeometry(),
            12,
            quadrature=SurfaceQuadraturePolicy(),
        )
    line = jnp.column_stack((jnp.linspace(-1, 1, 32), jnp.zeros(32), jnp.zeros(32)))
    with pytest.raises(ValueError, match="admission failed"):
        SurfacePointCloudPlan(
            line,
            SampledSurfaceGeometry(
                reference_normals=jnp.tile(jnp.array([0.0, 0.0, 1.0]), (32, 1))
            ),
            12,
            quadrature=SurfaceQuadraturePolicy("supplied", measures=jnp.ones(32)),
        ).prepare()


def test_projection_failure_returns_status() -> None:
    plan = surface_plan(size=64, neighbors=12)
    relation = MeshfreeNeighborhoodPlan(plan.points, 12).prepare().relation
    evaluation = plan.geometry.evaluate(jnp.zeros_like(plan.points), relation)
    assert not bool(jnp.any(evaluation.evidence.valid))
    assert bool(jnp.all(evaluation.evidence.status != 0))


def test_native_normal_projection_preserves_retraction_and_reports_rank_loss() -> None:
    geometry = surface_plan(size=64).geometry
    assert isinstance(geometry, ImplicitSurfaceGeometry)
    source = geometry.source
    assert isinstance(source, RegularLevelSetManifold)
    point = jnp.array([0.0, 0.0, 1.0])
    tangent = jnp.array([0.2, -0.1, 0.0])
    result = source.project_normal(point + tangent)
    expected = (point + tangent) / jnp.linalg.norm(point + tangent)
    assert bool(result.valid)
    np.testing.assert_allclose(result.points, expected, atol=1e-10)
    np.testing.assert_allclose(source.retract(point, tangent), expected, atol=1e-10)
    failed = source.project_normal(jnp.zeros(3))
    assert not bool(failed.valid)
    assert int(failed.status) != 0


def test_open_planar_patch_and_inconsistent_reference_orientation_refused() -> None:
    u, v = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    points = jnp.asarray(np.column_stack((u.ravel(), v.ravel(), np.zeros(25))))
    normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (25, 1))
    with pytest.raises(ValueError, match="admission failed"):
        SurfacePointCloudPlan(
            points,
            SampledSurfaceGeometry(reference_normals=normals),
            12,
            quadrature=SurfaceQuadraturePolicy("supplied", measures=jnp.ones(25)),
        ).prepare()
    sphere = sphere_points(64)
    inconsistent = sphere * jnp.where(jnp.arange(64) % 2, 1.0, -1.0)[:, None]
    with pytest.raises(ValueError, match="admission failed"):
        SurfacePointCloudPlan(
            sphere,
            SampledSurfaceGeometry(reference_normals=inconsistent),
            12,
            quadrature=SurfaceQuadraturePolicy("supplied", measures=jnp.ones(64)),
        ).prepare()
