import jax
import jax.numpy as jnp
import pytest

from phydrax.metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection


_IDS = {
    "chart_id": "cartesian",
    "convention_id": "relativity-convention",
    "scale_id": "relativity-scale",
    "topology_id": "grid-topology",
    "geometry_lineage_id": "adm-geometry",
}


def _geometry(*, alpha=None, active=None, valid=None, snapshot_token=0):
    dtype = jnp.float32
    identity = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
    return ADMGridGeometry(
        jnp.ones(2, dtype=dtype) if alpha is None else alpha,
        jnp.zeros((2, 3), dtype=dtype),
        identity,
        identity,
        jnp.ones(2, dtype=dtype),
        jnp.zeros((2, 3, 3), dtype=dtype),
        jnp.asarray([True, False]) if active is None else active,
        jnp.asarray([True, False]) if valid is None else valid,
        snapshot_token=jnp.asarray(snapshot_token, dtype=jnp.int32),
        **_IDS,
    )


def test_adm_grid_geometry_validates_each_active_lane_under_jit():
    geometry = _geometry()

    finite, physical, inverse_defect, determinant_defect, all_valid = jax.jit(
        lambda value: (
            value.finite,
            value.physically_valid,
            value.inverse_defect,
            value.determinant_defect,
            value.all_active_valid,
        )
    )(geometry)

    assert geometry.leading_shape == (2,)
    assert geometry.snapshot_token.dtype == jnp.int32
    assert geometry.snapshot_token.shape == ()
    assert jnp.array_equal(finite, jnp.asarray([True, True]))
    assert jnp.array_equal(physical, jnp.asarray([True, False]))
    assert jnp.array_equal(inverse_defect, jnp.zeros(2))
    assert jnp.array_equal(determinant_defect, jnp.zeros(2))
    assert bool(all_valid)


def test_adm_grid_geometry_reports_invalid_lapse_without_clipping_it():
    alpha = jnp.asarray([1.0, -0.25], dtype=jnp.float32)
    geometry = _geometry(
        alpha=alpha,
        active=jnp.asarray([True, True]),
        valid=jnp.asarray([True, True]),
    )

    assert jnp.array_equal(geometry.alpha, alpha)
    assert jnp.array_equal(geometry.lapse_positive, jnp.asarray([True, False]))
    assert jnp.array_equal(
        geometry.physically_valid,
        jnp.asarray([True, False]),
    )
    assert not bool(geometry.all_active_valid)


def test_adm_grid_geometry_detects_non_spatial_metric_and_inverse_defects_per_lane():
    dtype = jnp.float32
    spatial = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
    spatial = spatial.at[1, 0, 0].set(-1.0)
    inverse = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(2, dtype=dtype),
        jnp.zeros((2, 3), dtype=dtype),
        spatial,
        inverse,
        jnp.ones(2, dtype=dtype),
        jnp.zeros((2, 3, 3), dtype=dtype),
        jnp.ones(2, dtype="bool"),
        jnp.ones(2, dtype="bool"),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        **_IDS,
    )

    assert jnp.array_equal(
        geometry.spatial_positive_definite,
        jnp.asarray([True, False]),
    )
    assert jnp.array_equal(
        geometry.inverse_consistent,
        jnp.asarray([True, False]),
    )
    assert jnp.array_equal(
        geometry.physically_valid,
        jnp.asarray([True, False]),
    )


def test_stress_energy_projection_preserves_defects_and_geometry_binding():
    geometry = _geometry()
    dtype = jnp.float32
    projection = StressEnergyProjection(
        jnp.asarray([2.0, jnp.nan], dtype=dtype),
        jnp.zeros((2, 3), dtype=dtype),
        jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
        jnp.asarray([True, False]),
        jnp.asarray([True, False]),
        jnp.asarray([1.0e-6, jnp.nan], dtype=dtype),
        jnp.asarray([2.0e-6, jnp.nan], dtype=dtype),
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id="stress-energy",
    )

    finite, physically_valid, all_valid = jax.jit(
        lambda value: (value.finite, value.physically_valid, value.all_active_valid)
    )(projection)

    assert bool(projection.compatible_with(geometry))
    assert jnp.array_equal(projection.projection_defect[0], jnp.asarray(1.0e-6))
    assert jnp.array_equal(projection.conservation_defect[0], jnp.asarray(2.0e-6))
    assert jnp.array_equal(finite, jnp.asarray([True, False]))
    assert jnp.array_equal(physically_valid, jnp.asarray([True, False]))
    assert bool(all_valid)


def test_stress_energy_compatibility_rejects_changed_geometry_lineage():
    geometry = _geometry()
    dtype = jnp.float32
    projection = StressEnergyProjection(
        jnp.ones(2, dtype=dtype),
        jnp.zeros((2, 3), dtype=dtype),
        jnp.zeros((2, 3, 3), dtype=dtype),
        jnp.ones(2, dtype="bool"),
        jnp.ones(2, dtype="bool"),
        jnp.zeros(2, dtype=dtype),
        jnp.zeros(2, dtype=dtype),
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id="different-geometry",
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id="stress-energy",
    )

    assert not bool(projection.compatible_with(geometry))


def test_stress_energy_compatibility_rejects_stale_dynamic_stage_under_jit():
    geometry = _geometry(snapshot_token=7)
    dtype = jnp.float32
    projection = StressEnergyProjection(
        jnp.ones(2, dtype=dtype),
        jnp.zeros((2, 3), dtype=dtype),
        jnp.zeros((2, 3, 3), dtype=dtype),
        jnp.ones(2, dtype="bool"),
        jnp.ones(2, dtype="bool"),
        jnp.zeros(2, dtype=dtype),
        jnp.zeros(2, dtype=dtype),
        snapshot_token=jnp.asarray(6, dtype=jnp.int32),
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id="stress-energy",
    )

    compatible = jax.jit(
        lambda source, current_geometry: source.compatible_with(current_geometry)
    )(projection, geometry)

    assert not bool(compatible)


def test_exchange_records_reject_misaligned_shapes_and_invalid_identities():
    dtype = jnp.float32
    identity = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
    with pytest.raises(ValueError, match="beta_contravariant must have shape"):
        ADMGridGeometry(
            jnp.ones(2, dtype=dtype),
            jnp.zeros((3, 3), dtype=dtype),
            identity,
            identity,
            jnp.ones(2, dtype=dtype),
            jnp.zeros((2, 3, 3), dtype=dtype),
            jnp.ones(2, dtype="bool"),
            jnp.ones(2, dtype="bool"),
            snapshot_token=jnp.asarray(0, dtype=jnp.int32),
            **_IDS,
        )
    with pytest.raises(ValueError, match="snapshot_token must be scalar"):
        ADMGridGeometry(
            jnp.ones(2, dtype=dtype),
            jnp.zeros((2, 3), dtype=dtype),
            identity,
            identity,
            jnp.ones(2, dtype=dtype),
            jnp.zeros((2, 3, 3), dtype=dtype),
            jnp.ones(2, dtype="bool"),
            jnp.ones(2, dtype="bool"),
            snapshot_token=jnp.zeros(2, dtype=jnp.int32),
            **_IDS,
        )
    with pytest.raises(ValueError, match="projection_id"):
        StressEnergyProjection(
            jnp.ones(2, dtype=dtype),
            jnp.zeros((2, 3), dtype=dtype),
            identity,
            jnp.ones(2, dtype="bool"),
            jnp.ones(2, dtype="bool"),
            jnp.zeros(2, dtype=dtype),
            jnp.zeros(2, dtype=dtype),
            snapshot_token=jnp.asarray(0, dtype=jnp.int32),
            geometry_lineage_id="geometry",
            convention_id="convention",
            scale_id="scale",
            topology_id="topology",
            projection_id="",
        )
