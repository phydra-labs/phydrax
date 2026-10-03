# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.meshfree_open_surface_diffusion import (
    chart_patch,
    LATLONG_CHART,
    PATCH_LOWER,
    PATCH_UPPER,
    run_open_patch,
)
from examples.meshfree_surface_laplace_beltrami import run_workflow, surface_plan
from phydrax.discretization.meshfree._stencils import LocalStencilPolicy
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import ImplicitSurfaceGeometry
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.geometry.analytic import Sphere
from phydrax.metrix._ambient import RegularLevelSetManifold


def _numeric_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise TypeError(f"{name} must contain a measured real numerical error.")
    return float(value)


@pytest.mark.parametrize("approximation", ["gmls", "phs-rbf-fd"])
def test_sphere_eigenfunction_and_ambient_tangent_divergence(approximation: str) -> None:
    plan = surface_plan(size=128, neighbors=20)
    from phydrax.discretization.meshfree._types import MeshfreeApproximation
    from phydrax.typing import parse

    policy = LocalStencilPolicy(
        approximation=parse(approximation, MeshfreeApproximation, "approximation")
    )
    prepared = SurfacePointCloudPlan(
        plan.points, plan.geometry, 20, quadrature=plan.quadrature, stencil_policy=policy
    ).prepare()
    field = prepared.points[:, 2]
    laplace = prepared.laplace_beltrami.mv(field)
    assert float(jnp.linalg.norm(laplace + 2 * field) / jnp.linalg.norm(2 * field)) < 0.2
    np.testing.assert_allclose(prepared.laplace_beltrami.mv(jnp.ones(128)), 0, atol=1e-9)
    expected_gradient = (
        jnp.array([0.0, 0.0, 1.0])[None, :] - field[:, None] * prepared.normals
    )
    assert (
        float(
            jnp.linalg.norm(prepared.surface_gradient.mv(field) - expected_gradient)
            / jnp.linalg.norm(expected_gradient)
        )
        < 0.08
    )
    # Ambient position restricted to S2 has surface divergence 2. This checks
    # the geometry terms, rather than comparing an implementation to itself.
    np.testing.assert_allclose(
        prepared.strong_surface_divergence.mv(prepared.points), 2, atol=0.15
    )


def test_measure_paired_divergence_has_closed_surface_green_identity() -> None:
    prepared = surface_plan(size=96, neighbors=16).prepare()
    scalar = prepared.points[:, 0] * prepared.points[:, 1] + 0.2 * prepared.points[:, 2]
    vector = jnp.stack(
        (prepared.points[:, 2], jnp.sin(prepared.points[:, 0]), prepared.points[:, 1]),
        axis=-1,
    )
    left = jnp.sum(prepared.measures * scalar * prepared.surface_divergence.mv(vector))
    right = -jnp.sum(
        prepared.measures[:, None] * prepared.surface_gradient.mv(scalar) * vector
    )
    np.testing.assert_allclose(left, right, atol=1e-11)
    np.testing.assert_allclose(
        jnp.sum(prepared.measures * prepared.surface_divergence.mv(vector)), 0, atol=1e-11
    )


def test_fixed_support_refresh_jit_projection_jvp_and_refusal() -> None:
    prepared = surface_plan(size=64, neighbors=12).prepare()
    refreshed = eqx.filter_jit(prepared.refresh)(prepared.points)
    assert bool(refreshed.accepted)
    np.testing.assert_allclose(refreshed.measures, prepared.measures, atol=1e-10)
    repeated = refreshed.prepared.refresh(refreshed.points)
    assert bool(repeated.accepted)
    np.testing.assert_allclose(repeated.measures, prepared.measures, atol=1e-10)
    tangent = jnp.broadcast_to(jnp.array([0.03, -0.02, 0.01]), prepared.points.shape)
    _, derivative = jax.jvp(
        lambda points: prepared.refresh(points).points, (prepared.points,), (tangent,)
    )
    expected = (
        tangent - jnp.sum(tangent * prepared.normals, axis=-1)[:, None] * prepared.normals
    )
    np.testing.assert_allclose(derivative, expected, atol=1e-8)
    failed = prepared.refresh(jnp.roll(prepared.points, 1, axis=0))
    assert not bool(failed.accepted)
    assert bool(jnp.any(failed.status != 0))


def test_native_surface_values_refuse_off_surface_inside_region() -> None:
    prepared = surface_plan(size=96, neighbors=16).prepare()
    envelope = Sphere(jnp.zeros(3), 2.0).compile()
    reconstruction = prepared.prepare_field_reconstruction(support_geometry=envelope)
    route, on_surface = reconstruction.kernel.locate(prepared.points, (0, 0, 0), None)
    _, off_surface = reconstruction.kernel.locate(0.8 * prepared.points, (0, 0, 0), None)
    assert bool(jnp.all(on_surface.valid))
    assert not bool(jnp.any(off_surface.valid))
    value = reconstruction.kernel.apply(route, jnp.ones(96))
    np.testing.assert_allclose(value, 1, atol=1e-10)


def test_analytic_sphere_torus_and_sampled_intrinsic_behavior() -> None:
    result = run_workflow(size=512)
    assert result["geometry_valid"]
    for name in (
        "sphere_exact_error",
        "sphere_sampled_error",
        "torus_exact_error",
        "torus_sampled_error",
    ):
        assert _numeric_metric(result, name) < 0.3
    assert _numeric_metric(result, "sphere_normal_error") < 0.08
    assert _numeric_metric(result, "torus_normal_error") < 0.2


def test_underresolved_torus_workflow_refuses_without_synthetic_metrics() -> None:
    with pytest.raises(ValueError):
        run_workflow(size=128)


@pytest.mark.parametrize("sampled", [False, True])
def test_expanding_sphere_reference_area_ratio_without_refresh_drift(
    sampled: bool,
) -> None:
    from phydrax.discretization.meshfree._surface_geometry import (
        ImplicitSurfaceGeometry,
        SampledSurfaceGeometry,
    )
    from phydrax.metrix._ambient import RegularLevelSetManifold

    plan = surface_plan(size=96, neighbors=16)
    perturbation = jnp.asarray(
        np.random.default_rng(17).normal(scale=0.002, size=(96, 3))
    )
    points = plan.points + perturbation
    points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
    geometry = (
        SampledSurfaceGeometry(reference_normals=points) if sampled else plan.geometry
    )
    prepared = SurfacePointCloudPlan(
        points, geometry, 16, quadrature=plan.quadrature
    ).prepare()
    trust = float(jnp.min(prepared.neighborhood.trust_margin))
    assert trust > 0
    scale = 1 + 0.1 * trust
    if sampled:
        expanded = prepared.refresh(scale * prepared.points)
    else:

        def constraint(point: Array) -> Array:
            return jnp.asarray([jnp.dot(point, point) - scale**2])

        source = RegularLevelSetManifold(
            constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere"
        )
        geometry = ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.5, geometry_id=plan.geometry.geometry_id
        )
        expanded = prepared.refresh(scale * prepared.points, geometry=geometry)
    assert bool(expanded.accepted)
    np.testing.assert_allclose(
        expanded.measures, scale**2 * prepared.measures, rtol=1e-10, atol=1e-12
    )
    repeated = expanded.prepared.refresh(expanded.points)
    assert bool(repeated.accepted)
    np.testing.assert_allclose(
        repeated.measures, expanded.measures, rtol=1e-12, atol=1e-12
    )


def test_open_patch_green_identity_closes_with_declared_boundary_quadrature() -> None:
    patch = chart_patch(LATLONG_CHART, PATCH_LOWER, PATCH_UPPER, count=13)
    boundary = patch.boundary
    assert boundary is not None
    x = patch.points
    scalar = x[:, 0] * x[:, 1]
    # x y is a degree-2 spherical harmonic: Lap_S(x y) = -6 x y.
    laplace = patch.laplace_beltrami.mv(scalar)
    assert (
        float(jnp.linalg.norm(laplace + 6 * scalar) / jnp.linalg.norm(6 * scalar)) < 2e-2
    )
    vector = jnp.stack((x[:, 2], jnp.sin(x[:, 0]), x[:, 1]), axis=-1)
    tangent = vector - jnp.sum(vector * patch.normals, -1, keepdims=True) * patch.normals
    left = jnp.sum(patch.measures * scalar * patch.surface_divergence.mv(tangent))
    flux = jnp.sum(
        scalar[boundary.nodes]
        * jnp.sum(boundary.weighted_conormals * tangent[boundary.nodes], axis=-1)
    )
    right = -jnp.sum(
        patch.measures[:, None] * patch.surface_gradient.mv(scalar) * tangent
    )
    np.testing.assert_allclose(left, right + flux, atol=1e-12)
    # The pointwise divergence obeys the continuous divergence theorem.
    gradient_z = jnp.asarray([0.0, 0.0, 1.0]) - x[:, 2:3] * x
    np.testing.assert_allclose(
        jnp.sum(patch.measures * patch.strong_surface_divergence.mv(gradient_z)),
        jnp.sum(jnp.sum(boundary.weighted_conormals * gradient_z[boundary.nodes], -1)),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="no conormal"):
        _ = surface_plan(size=64, neighbors=12).prepare().conormal_derivative


def test_open_patch_transient_diffusion_matches_spherical_harmonic_decay() -> None:
    result = run_open_patch(count=11, steps=10)
    assert result["solver_successful"]
    assert float(result["final_relative_error"]) < 2e-2
    assert float(result["perimeter_error"]) < 1e-12
    assert float(result["maximum_balance_defect"]) < 0.1


def test_circle_curve_operators_eigenfunction_length_and_refresh() -> None:
    circle = RegularLevelSetManifold(
        lambda point: jnp.asarray([point @ point - 1]),
        ambient_dimension=2,
        codimension=1,
        manifold_id="unit-circle",
    )
    angle = np.sort(np.random.default_rng(2).uniform(0, 2 * np.pi, 64))
    points = jnp.asarray(np.column_stack((np.cos(angle), np.sin(angle))))
    prepared = SurfacePointCloudPlan(
        points,
        ImplicitSurfaceGeometry(circle, geometry_id="unit-circle"),
        8,
        quadrature=SurfaceQuadraturePolicy(),
        stencil_policy=LocalStencilPolicy(polynomial_degree=4, chunk_rows=32),
    ).prepare()
    x = prepared.points[:, 0]
    assert (
        float(jnp.linalg.norm(prepared.laplace_beltrami.mv(x) + x) / jnp.linalg.norm(x))
        < 1e-2
    )
    np.testing.assert_allclose(prepared.geometry.mean_curvature, 1, atol=1e-12)
    assert abs(float(prepared.quadrature_evidence.total_area) - 2 * np.pi) < 0.1
    refreshed = prepared.refresh(prepared.points)
    assert bool(refreshed.accepted)
    np.testing.assert_allclose(refreshed.measures, prepared.measures, atol=1e-12)
