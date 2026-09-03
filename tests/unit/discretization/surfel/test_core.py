from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _prepared(active_mask=None):
    ids = jnp.asarray((7, 3, 11), dtype=jnp.int64)
    position = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    weights = jnp.asarray((2.0, 3.0, 4.0))
    return phx.discretization.SurfelSetPlan(
        ids,
        position,
        weights,
        active_mask=active_mask,
        source_entity_ids=jnp.asarray((20, 21, 22)),
    ).prepare()


def _geometry(active_mask=None):
    prepared = _prepared(active_mask)
    normals = jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (3, 1))
    first = jnp.tile(jnp.asarray((0.5, 0.0, 0.0)), (3, 1))
    second = jnp.tile(jnp.asarray((0.0, 0.25, 0.0)), (3, 1))
    axes = jnp.stack((first, second), axis=-1)
    return phx.discretization.SurfelGeometryPlan(prepared).materialize(
        prepared.reference_position,
        normals,
        axes,
    )


def test_surfel_discretization_uses_point_ownership_and_surface_measure() -> None:
    prepared = _prepared()
    assert isinstance(prepared.support.topology, phx.discretization.PointTopology)
    assert prepared.capacity == 3
    assert prepared.active_count == 3
    assert prepared.field_spaces == (
        prepared.position_space,
        prepared.normal_space,
        prepared.tangent_axes_space,
    )
    np.testing.assert_allclose(prepared.measures[0].weights, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(prepared.source_entity_ids, [20, 21, 22])


def test_surfel_geometry_validates_anisotropic_footprints() -> None:
    geometry = _geometry()
    assert bool(geometry.evidence.successful)
    np.testing.assert_allclose(
        geometry.footprint_measure,
        np.pi * 0.5 * 0.25,
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(geometry.footprint_half_width, [[0.5, 0.25, 0.0]] * 3)
    np.testing.assert_allclose(geometry.evidence.maximum_tangency_defect, 0.0)
    np.testing.assert_allclose(geometry.evidence.minimum_orientation_cosine, 1.0)


def test_surfel_geometry_fails_closed_for_invalid_active_normal() -> None:
    prepared = _prepared()
    normals = (
        jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (3, 1))
        .at[1]
        .set(jnp.asarray((0.0, 0.0, 2.0)))
    )
    axes = jnp.tile(
        jnp.asarray(((0.5, 0.0), (0.0, 0.25), (0.0, 0.0)))[None, ...],
        (3, 1, 1),
    )
    geometry = phx.discretization.SurfelGeometryPlan(prepared).materialize(
        prepared.reference_position,
        normals,
        axes,
    )
    assert not bool(geometry.evidence.successful)
    assert geometry.evidence.maximum_normal_norm_defect == 1.0
    assert bool(jnp.all(jnp.isfinite(geometry.normal)))


def test_inactive_nonfinite_surfel_geometry_is_sanitized() -> None:
    prepared = _prepared(jnp.asarray((True, True, False)))
    position = prepared.reference_position.at[2].set(jnp.asarray((jnp.nan,) * 3))
    normal = (
        jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (3, 1))
        .at[2]
        .set(jnp.asarray((jnp.nan,) * 3))
    )
    axes = (
        jnp.tile(
            jnp.asarray(((0.5, 0.0), (0.0, 0.25), (0.0, 0.0)))[None, ...],
            (3, 1, 1),
        )
        .at[2]
        .set(jnp.nan)
    )
    geometry = phx.discretization.SurfelGeometryPlan(prepared).materialize(
        position, normal, axes
    )
    assert bool(geometry.evidence.successful)
    np.testing.assert_array_equal(geometry.position[2], 0.0)
    np.testing.assert_array_equal(geometry.normal[2], 0.0)


def test_surfel_geometry_materialization_jits() -> None:
    prepared = _prepared()
    normals = jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (3, 1))
    axes = jnp.tile(
        jnp.asarray(((0.5, 0.0), (0.0, 0.25), (0.0, 0.0)))[None, ...],
        (3, 1, 1),
    )
    materialize = eqx.filter_jit(
        phx.discretization.SurfelGeometryPlan(prepared).materialize
    )
    geometry = materialize(prepared.reference_position, normals, axes)
    assert bool(geometry.evidence.successful)
