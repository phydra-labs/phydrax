# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.meshfree_surface_laplace_beltrami import sphere_points, surface_plan
from phydrax.discretization.meshfree._neighbors import MeshfreeNeighborhoodPlan
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import (
    ImplicitSurfaceGeometry,
    SampledSurfaceGeometry,
)
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.ein import contract
from phydrax.metrix._ambient import RegularLevelSetManifold


def _circle_points(count: int, dimension: int, rotation: np.ndarray | None) -> Array:
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, count)
    angle = (np.arange(count) + jitter) * 2 * np.pi / count
    planar = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))
    if dimension == 2:
        return jnp.asarray(planar[:, :2])
    if rotation is None:
        raise ValueError("A space curve test needs its rotation.")
    return jnp.asarray(planar @ rotation.T)


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
    geometry = surface_plan(size=64).geometry
    assert isinstance(geometry, ImplicitSurfaceGeometry)
    with pytest.raises(ValueError, match="Open surface sources require"):
        SurfacePointCloudPlan(
            points,
            ImplicitSurfaceGeometry(geometry.source, closed=False),
            12,
            quadrature=SurfaceQuadraturePolicy(),
        )
    with pytest.raises(ValueError, match="Open surface sources require"):
        SurfacePointCloudPlan(
            points,
            SampledSurfaceGeometry(reference_normals=points, closed=False),
            12,
            quadrature=SurfaceQuadraturePolicy(),
        )
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


def test_plane_and_space_curves_declare_dimension_and_curvature() -> None:
    circle = RegularLevelSetManifold(
        lambda x: jnp.asarray([x @ x - 1]),
        ambient_dimension=2,
        codimension=1,
        manifold_id="circle",
    )
    geometry = ImplicitSurfaceGeometry(circle, geometry_id="circle")
    assert (geometry.intrinsic_dimension, geometry.ambient_dimension) == (1, 2)
    points = _circle_points(40, 2, None)
    relation = MeshfreeNeighborhoodPlan(points, 6).prepare().relation
    planar = geometry.evaluate(1.02 * points, relation)
    assert bool(jnp.all(planar.evidence.valid))
    np.testing.assert_allclose(planar.normals, planar.points, atol=1e-12)
    np.testing.assert_allclose(planar.mean_curvature, 1, atol=1e-12)
    rotation = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    frame = jnp.asarray(rotation)

    def tilted(x: Array) -> Array:
        y = frame.T @ x
        return jnp.asarray([y[0] ** 2 + y[1] ** 2 - 1, y[2]])

    space = ImplicitSurfaceGeometry(
        RegularLevelSetManifold(
            tilted, ambient_dimension=3, codimension=2, manifold_id="tilted-circle"
        ),
        geometry_id="tilted-circle",
    )
    space_points = _circle_points(40, 3, rotation)
    evaluation = space.evaluate(
        space_points, MeshfreeNeighborhoodPlan(space_points, 6).prepare().relation
    )
    assert bool(jnp.all(evaluation.evidence.valid))
    assert evaluation.normal_frames.shape == (40, 3, 2)
    # Unit circle: the mean-curvature vector (Laplace-Beltrami of x) is -x.
    np.testing.assert_allclose(
        evaluation.mean_curvature_vector, -evaluation.points, atol=1e-12
    )
    with pytest.raises(ValueError, match="codimension-one"):
        _ = evaluation.curvature_tensor
    with pytest.raises(ValueError, match="curve in R2/R3 or a sheet in R3"):
        SampledSurfaceGeometry(intrinsic_dimension=2, ambient_dimension=2)


def test_sampled_fit_degree_and_oversampling_with_independent_error_estimates() -> None:
    points = sphere_points(256)
    relation = MeshfreeNeighborhoodPlan(points, 20).prepare().relation
    errors = {}
    for degree in (2, 4):
        evaluation = SampledSurfaceGeometry(
            reference_normals=points, fit_degree=degree, oversampling=1.2
        ).evaluate(points, relation, reference_orientation=points)
        evidence = evaluation.evidence
        assert bool(jnp.all(evidence.valid))
        assert evidence.estimate_only and evidence.source == "sample-estimate"
        assert not evidence.tube_certified and not evidence.projection_to_source
        normal_error = jnp.max(jnp.linalg.norm(evaluation.normals - points, axis=-1))
        curvature_error = jnp.max(jnp.abs(evaluation.mean_curvature - 2))
        # The estimates are local, finite and of the observed magnitude.
        assert bool(jnp.all(jnp.isfinite(evidence.normal_error_estimate)))
        assert bool(jnp.all(jnp.isfinite(evidence.curvature_error_estimate)))
        assert float(jnp.max(evidence.curvature_error_estimate)) < 10 * float(
            curvature_error
        )
        errors[degree] = (float(normal_error), float(curvature_error))
    assert errors[4][0] < errors[2][0] / 3
    assert errors[4][1] < errors[2][1] / 5
    with pytest.raises(ValueError, match="fit_degree"):
        SampledSurfaceGeometry(fit_degree=5)
    with pytest.raises(ValueError, match="oversampling"):
        SampledSurfaceGeometry(oversampling=0.5)


def test_sampled_geometry_outputs_are_gauge_invariant_under_rotation() -> None:
    points = sphere_points(256)
    relation = MeshfreeNeighborhoodPlan(points, 20).prepare().relation
    rotation = jnp.asarray(np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0])
    geometry = SampledSurfaceGeometry(reference_normals=points, fit_degree=3)
    original = geometry.evaluate(points, relation, reference_orientation=points)
    rotated_points = points @ rotation.T
    rotated = SampledSurfaceGeometry(
        reference_normals=rotated_points, fit_degree=3
    ).evaluate(rotated_points, relation, reference_orientation=rotated_points)
    # Discrete frames may switch, but projector and second form are covariant.
    np.testing.assert_allclose(
        rotated.projectors,
        contract("ai,nij,bj->nab", rotation, original.projectors, rotation),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rotated.second_fundamental_form,
        contract(
            "ai,bj,ck,nijk->nabc",
            rotation,
            rotation,
            rotation,
            original.second_fundamental_form,
        ),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        rotated.mean_curvature, original.mean_curvature, atol=1e-10
    )
    assert bool(jnp.all(rotated.evidence.gauge_margin > 0))
