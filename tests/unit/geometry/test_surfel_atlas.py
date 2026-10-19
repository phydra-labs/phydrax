from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _atlas_plan() -> tuple[
    phx.geometry.BoundaryAtlas,
    phx.geometry.ImmersedMarkerQuadraturePlan,
    phx.geometry.BoundaryAtlasSurfelPlan,
]:
    atlas = phx.geometry.circle_boundary_atlas(
        jnp.asarray((0.5, 0.5)),
        jnp.asarray(0.2),
        source_id="surfel-circle",
    )
    quadrature = phx.geometry.ImmersedMarkerQuadraturePlan(
        jnp.arange(4),
        jnp.arange(4),
        jnp.full((4, 1), 0.5),
        jnp.ones((4,)),
    )
    plan = phx.geometry.BoundaryAtlasSurfelPlan(quadrature, footprint_area_ratio=1.5)
    return atlas, quadrature, plan


# The existing 1e-12 circle-oracle allowance exceeds 4096 binary64 eps:
# it covers trigonometric chart evaluation, norms, and four-chart summation.
# Both relative and absolute bounds are retained from the analytic circle fixture.
def test_surfel_atlas_scenario_1() -> None:
    atlas, _, plan = _atlas_plan()
    prepared = plan.prepare(atlas, 0.0)
    materialized = prepared.materialize(atlas, 0.0)
    geometry = materialized.geometry
    assert bool(materialized.successful)
    assert geometry.ambient_dimension == 2
    np.testing.assert_allclose(
        geometry.footprint_measure,
        1.5 * geometry.physical_surface_weight,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    radial = geometry.position - jnp.asarray((0.5, 0.5))
    alignment = jnp.abs(jnp.sum(radial * geometry.normal, axis=-1))
    np.testing.assert_allclose(alignment, 0.2, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        jnp.sum(geometry.physical_surface_weight),
        2.0 * jnp.pi * 0.2,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    atlas, quadrature, plan = _atlas_plan()
    prepared = plan.prepare(atlas, 0.0)
    materialized = eqx.filter_jit(prepared.materialize)(
        atlas,
        jnp.asarray(0.0),
        velocity=jnp.tile(jnp.asarray((0.25, -0.1)), (4, 1)),
    )
    marker_materialization = quadrature.materialize(
        atlas,
        0.0,
        velocity=jnp.tile(jnp.asarray((0.25, -0.1)), (4, 1)),
    )
    markers = quadrature.marker_plan(marker_materialization).prepare()
    kinematics = materialized.marker_kinematics(markers)
    np.testing.assert_allclose(kinematics.position, materialized.geometry.position)
    np.testing.assert_allclose(kinematics.velocity, materialized.velocity)
    atlas, _, _ = _atlas_plan()
    invalid = phx.geometry.ImmersedMarkerQuadraturePlan(
        jnp.asarray((0,)),
        jnp.asarray((atlas.num_charts,)),
        jnp.asarray(((0.5,),)),
        jnp.asarray((1.0,)),
    )

    with pytest.raises(RuntimeError, match="ImmersedMarkerQuadraturePlan.chart_indices"):
        invalid.materialize(atlas, 0.0)


def test_marker_plan_refuses_wrong_materialization_kind() -> None:
    _, quadrature, _ = _atlas_plan()
    with pytest.raises(TypeError, match="materialization"):
        quadrature.marker_plan(None)  # ty: ignore[invalid-argument-type]


def test_marker_velocity_provider_matches_array_velocity() -> None:
    atlas, quadrature, _ = _atlas_plan()

    def velocity(time: Array, positions: Array) -> Array:
        return jnp.ones_like(positions) * time

    provided = quadrature.materialize(atlas, 0.25, velocity=velocity)
    explicit = quadrature.materialize(atlas, 0.25, velocity=jnp.full((4, 2), 0.25))
    np.testing.assert_array_equal(provided.velocity, explicit.velocity)
    np.testing.assert_array_equal(provided.position, explicit.position)
    assert bool(provided.finite)
