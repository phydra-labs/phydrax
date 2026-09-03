from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _atlas_plan():
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


def test_boundary_atlas_materializes_oriented_weighted_surfels() -> None:
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


def test_atlas_surfel_materialization_adapts_to_marker_kinematics() -> None:
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
