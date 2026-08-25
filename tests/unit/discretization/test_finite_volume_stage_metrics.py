#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._embedded_dynamics import (
    lower_embedded_stage_metrics,
    UnstructuredEmbeddedBoundarySet,
)
from phydrax.discretization.finite_volume._geometry_protocol import (
    ALEGeometryConsistencyPolicy,
    ExplicitFaceBlockGeometry,
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageFaceBlock,
    FiniteVolumeStageFaceLayout,
    FiniteVolumeStageGeometryEvidence,
    FiniteVolumeStageMetrics,
    lower_static_unstructured_stage_metrics,
    PreparedFiniteVolumeGeometry,
)
from phydrax.discretization.finite_volume._physical_boundaries import SlipWallBoundary
from phydrax.discretization.finite_volume._unstructured_motion import (
    FixedConnectivityMotionPlan,
)


def _unstructured(vertices=None):
    points = (
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        if vertices is None
        else np.asarray(vertices)
    )
    return phx.discretization.UnstructuredFiniteVolumePlan(
        points,
        triangles=np.asarray(((0, 1, 2), (0, 2, 3))),
    ).prepare()


def _quadrilateral_strip():
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        )
    )
    quadrilaterals = np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
    ).prepare()


def _face_layout(**overrides):
    values = {
        "face_ids": np.asarray((10, 11), dtype=np.int32),
        "owner_cells": np.asarray((0, 1), dtype=np.int32),
        "neighbour_cells": np.asarray((1, -1), dtype=np.int32),
        "active_mask": np.asarray((True, True)),
        "boundary_policy_count": 1,
        "spatial_shape": (2, 2),
        "quadrature_shape": (2, 2),
        "block_id": "face-block",
    }
    values.update(overrides)
    if "boundary_policy_ids" not in overrides:
        values["boundary_policy_ids"] = np.where(
            np.asarray(values["neighbour_cells"]) < 0,
            0,
            -1,
        )
    return FiniteVolumeStageFaceLayout(**values)


def _face_block(**overrides):
    layout_fields = {
        "face_ids",
        "owner_cells",
        "neighbour_cells",
        "active_mask",
        "spatial_shape",
        "boundary_policy_ids",
        "boundary_policy_count",
        "quadrature_shape",
        "block_id",
    }
    layout_overrides = {
        key: overrides.pop(key) for key in tuple(overrides) if key in layout_fields
    }
    layout = overrides.pop("layout", None)
    if layout is not None and layout_overrides:
        raise ValueError("Provide either layout or layout field overrides.")
    values = {
        "layout": _face_layout(**layout_overrides) if layout is None else layout,
        "face_centers": np.asarray(((0.5, 0.5), (0.0, 0.5))),
        "area_vectors": np.asarray(((1.0, 0.0), (-1.0, 0.0))),
        "face_measures": np.asarray((1.0, 1.0)),
        "quadrature_points": np.asarray(
            (
                ((0.5, 0.25), (0.5, 0.75)),
                ((0.0, 0.25), (0.0, 0.75)),
            )
        ),
        "quadrature_weights": np.full((2, 2), 0.5),
        "quadrature_grid_normal_velocity": np.zeros((2, 2)),
    }
    values.update(overrides)
    return FiniteVolumeStageFaceBlock(**values)


def _geometry_evidence(**overrides):
    values = {
        "coordinate_effective_volume_defect": np.zeros((2,)),
        "coordinate_effective_volume_tolerance": np.full((2,), 1.0e-10),
        "face_closure_defect": np.zeros((2,)),
        "face_closure_tolerance": np.full((2,), 1.0e-10),
        "gcl_identity_defect": np.zeros((2,)),
        "gcl_identity_tolerance": np.full((2,), 1.0e-10),
        "expected_order": 2,
        "proposed_reduction_factor": 1.0,
        "passed": True,
        "status": FiniteVolumeGeometryStatus.SUCCESS,
        "evidence_version": 5,
        "policy_id": "stage-geometry-policy",
    }
    values.update(overrides)
    return FiniteVolumeStageGeometryEvidence(**values)


def _stage_metric_values():
    return {
        "topology_epoch_id": "epoch-0",
        "geometry_family_id": "geometry-family-0",
        "geometry_layout_id": "geometry-layout-0",
        "geometry_version": 3,
        "time": 0.25,
        "effective_cell_volumes": np.asarray((0.5, 0.5)),
        "coordinate_effective_cell_volumes": np.asarray((0.5, 0.5)),
        "mesh_volume_rate": np.zeros((2,)),
        "cell_centers": np.asarray(((2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0))),
        "active_cell_mask": np.asarray((True, True)),
        "face_blocks": (_face_block(),),
        "evidence": _geometry_evidence(),
    }


def _stage_metrics(**overrides):
    values = _stage_metric_values()
    values.update(overrides)
    return FiniteVolumeStageMetrics(**values)


def test_static_unstructured_lowering_has_stationary_success_evidence():
    discretization = _unstructured()
    metrics = lower_static_unstructured_stage_metrics(
        discretization,
        time=jnp.asarray(0.375),
        topology_epoch_id="accepted-topology-3",
    )

    assert metrics.topology_epoch_id == "accepted-topology-3"
    assert metrics.geometry_family_id == discretization.geometry_id
    assert metrics.geometry_layout_id
    assert metrics.geometry_version.shape == ()
    assert int(metrics.geometry_version) == 0
    assert metrics.time.shape == ()
    np.testing.assert_allclose(metrics.time, 0.375)
    np.testing.assert_allclose(
        metrics.effective_cell_volumes, discretization.cell_volumes
    )
    np.testing.assert_allclose(
        metrics.coordinate_effective_cell_volumes,
        metrics.effective_cell_volumes,
    )
    np.testing.assert_array_equal(
        metrics.active_cell_mask, np.ones((discretization.cell_count,), dtype=bool)
    )
    np.testing.assert_allclose(metrics.mesh_volume_rate, 0.0)
    np.testing.assert_allclose(metrics.cell_centers, discretization.cell_centers)
    assert isinstance(metrics.evidence, FiniteVolumeStageGeometryEvidence)
    assert metrics.evidence.policy_id
    assert bool(metrics.evidence.passed)
    assert int(metrics.evidence.status) == int(FiniteVolumeGeometryStatus.SUCCESS)
    assert int(metrics.evidence.evidence_version) == 0
    assert int(metrics.evidence.expected_order) == 0
    assert float(metrics.evidence.proposed_reduction_factor) == 1.0
    np.testing.assert_allclose(metrics.evidence.coordinate_effective_volume_defect, 0.0)
    np.testing.assert_allclose(metrics.evidence.face_closure_defect, 0.0)
    np.testing.assert_allclose(metrics.evidence.gcl_identity_defect, 0.0)
    for tolerance in (
        metrics.evidence.coordinate_effective_volume_tolerance,
        metrics.evidence.face_closure_tolerance,
        metrics.evidence.gcl_identity_tolerance,
    ):
        assert np.all(np.asarray(tolerance) > 0.0)
    assert len(metrics.face_blocks) == 1
    block = metrics.face_blocks[0]
    np.testing.assert_allclose(block.quadrature_grid_normal_velocity, 0.0)
    np.testing.assert_allclose(block.grid_normal_velocity, 0.0)
    np.testing.assert_allclose(
        jnp.sum(block.quadrature_weights, axis=1), block.face_measures
    )
    source_block = discretization.face_blocks[0]
    np.testing.assert_array_equal(block.layout.face_ids, source_block.face_ids)
    np.testing.assert_array_equal(block.layout.owner_cells, source_block.owner_cells)
    np.testing.assert_array_equal(
        block.layout.neighbour_cells, source_block.neighbour_cells
    )
    assert block.layout.boundary_policy_count == len(discretization.boundary_patch_names)
    policy_ids = np.asarray(block.layout.boundary_policy_ids)
    neighbours = np.asarray(block.layout.neighbour_cells)
    active = np.asarray(block.layout.active_mask)
    assert np.all(policy_ids[active & (neighbours < 0)] >= 0)
    assert np.all(
        policy_ids[active & (neighbours < 0)] < block.layout.boundary_policy_count
    )
    assert np.all(policy_ids[neighbours >= 0] == -1)
    np.testing.assert_array_equal(block.layout.active_mask, source_block.active_mask)
    np.testing.assert_allclose(block.face_centers, source_block.face_centers)
    np.testing.assert_allclose(block.area_vectors, source_block.area_vectors)
    np.testing.assert_allclose(block.face_measures, source_block.face_measures)
    np.testing.assert_allclose(
        block.quadrature_points,
        discretization.face_quadrature_points[source_block.face_ids],
    )
    np.testing.assert_allclose(
        block.quadrature_weights,
        discretization.face_quadrature_weights[source_block.face_ids],
    )


def test_moving_layout_preserves_exact_physical_boundary_routes():
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    base_plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        triangles=np.asarray(((0, 1, 2), (0, 2, 3))),
    )
    prepared = base_plan.prepare()
    motion = FixedConnectivityMotionPlan(
        base_plan,
        lambda time, points, args: points,
        mapping_id="boundary-route-motion",
    )

    layout = motion.face_layout
    assert layout.boundary_policy_count == len(prepared.boundary_patch_names)
    np.testing.assert_array_equal(
        layout.boundary_policy_ids,
        prepared.face_blocks[0].boundary_patch_ids,
    )
    layout.validate_boundary_policy_count(len(prepared.boundary_patch_names))
    geometry = motion.prepare_ssprk33_step(
        0.0,
        0.1,
        "moving-epoch",
        0,
        0,
        prior_effective_cell_volumes=prepared.cell_volumes,
    )
    stages = (
        geometry.stage_1,
        geometry.stage_2,
        geometry.stage_3,
        geometry.accepted_geometry,
    )
    assert all(stage.geometry_family_id == motion.plan_id for stage in stages)
    assert tuple(int(stage.geometry_version) for stage in stages) == (0, 1, 2, 3)


def test_embedded_lowering_binds_exact_physical_and_cut_policy_counts():
    discretization = _quadrilateral_strip()
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        field_id="stage-route-counts",
        body_tag=7,
    ).prepare()
    embedded_boundaries = UnstructuredEmbeddedBoundarySet({7: SlipWallBoundary()})

    metrics = lower_embedded_stage_metrics(
        discretization,
        embedded,
        embedded_boundaries,
        "epoch:stage-route-counts",
        0,
        0,
    )

    assert {block.layout.block_kind for block in metrics.face_blocks} == {
        "physical",
        "cut",
    }
    for block in metrics.face_blocks:
        layout = block.layout
        expected_count = (
            len(discretization.boundary_patch_names)
            if layout.block_kind == "physical"
            else len(embedded_boundaries.boundaries)
        )
        assert layout.boundary_policy_count == expected_count
        layout.validate_boundary_policy_count(expected_count)
        policies = np.asarray(layout.boundary_policy_ids)
        neighbours = np.asarray(layout.neighbour_cells)
        active = np.asarray(layout.active_mask)
        assert np.all(policies[active & (neighbours < 0)] >= 0)
        assert np.all(policies[active & (neighbours < 0)] < expected_count)
        assert np.all(policies[active & (neighbours >= 0)] == -1)


def test_closed_inactive_face_has_safe_zero_grid_velocity_average():
    block = _face_block(
        active_mask=np.asarray((True, False)),
        area_vectors=np.asarray(((1.0, 0.0), (0.0, 0.0))),
        face_measures=np.asarray((1.0, 0.0)),
        quadrature_weights=np.asarray(((0.5, 0.5), (0.0, 0.0))),
        quadrature_grid_normal_velocity=np.asarray(((2.0, 4.0), (3.0, -5.0))),
    )

    np.testing.assert_allclose(block.face_measures, (1.0, 0.0))
    np.testing.assert_allclose(block.quadrature_weights[1], 0.0)
    np.testing.assert_allclose(
        block.quadrature_grid_normal_velocity,
        ((2.0, 4.0), (3.0, -5.0)),
    )
    np.testing.assert_allclose(
        jnp.sum(block.quadrature_weights, axis=1), block.face_measures
    )
    np.testing.assert_allclose(block.grid_normal_velocity, (3.0, 0.0))
    assert np.all(np.isfinite(block.grid_normal_velocity))


def test_stage_face_block_rejects_zero_active_measure_and_invalid_quadrature():
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Active face measures"
    ):
        _face_block(
            face_measures=np.asarray((1.0, 0.0)),
            quadrature_weights=np.asarray(((0.5, 0.5), (0.0, 0.0))),
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonnegative"):
        _face_block(quadrature_weights=np.asarray(((-0.5, 1.5), (0.5, 0.5))))
    with pytest.raises(ValueError, match="grid-normal velocity"):
        _face_block(quadrature_grid_normal_velocity=np.zeros((2, 1)))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="sum"):
        _face_block(quadrature_weights=np.full((2, 2), 0.25))
    with pytest.raises(ValueError, match="same shape"):
        _face_block(area_vectors=np.ones((2, 3)))
    with pytest.raises(ValueError, match="cell routes"):
        _face_block(owner_cells=np.asarray((0.0, 1.0)))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Inactive face measures"
    ):
        _face_block(active_mask=np.asarray((True, False)))


@pytest.mark.parametrize("route", ("face_ids", "owner_cells", "neighbour_cells"))
def test_stage_face_routes_reject_int32_overflow(route):
    values = np.asarray((0, np.iinfo(np.int32).max + 1), dtype=np.int64)
    with pytest.raises(ValueError, match="representable as int32"):
        _face_block(**{route: values})


def test_stage_face_routes_are_normalized_to_int32():
    block = _face_block(
        face_ids=np.asarray((10, 11), dtype=np.int64),
        owner_cells=np.asarray((0, 1), dtype=np.int64),
        neighbour_cells=np.asarray((1, -1), dtype=np.int64),
    )

    assert np.asarray(block.layout.face_ids).dtype == np.int32
    assert np.asarray(block.layout.owner_cells).dtype == np.int32
    assert np.asarray(block.layout.neighbour_cells).dtype == np.int32


@pytest.mark.parametrize(
    ("boundary_policy_ids", "message"),
    (
        (None, "Active boundary face routes"),
        (np.asarray((-1, 1), dtype=np.int32), "in-range nonnegative"),
        (np.asarray((0, 0), dtype=np.int32), "Active interior face routes"),
    ),
)
def test_active_face_routes_require_valid_boundary_policy_ownership(
    boundary_policy_ids, message
):
    with pytest.raises(ValueError, match=message):
        _face_layout(boundary_policy_ids=boundary_policy_ids)


def test_inactive_face_routes_normalize_boundary_policy_ids_to_minus_one():
    layout = _face_layout(
        active_mask=np.asarray((False, True)),
        boundary_policy_ids=np.asarray(
            (np.iinfo(np.int64).max, 0),
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(layout.boundary_policy_ids, (-1, 0))
    assert np.asarray(layout.boundary_policy_ids).dtype == np.int32


def test_stage_face_layout_revalidates_exact_bound_policy_count():
    layout = _face_layout()

    layout.validate_boundary_policy_count(1)
    with pytest.raises(ValueError, match="does not match the bound boundary set"):
        layout.validate_boundary_policy_count(2)


def test_stage_face_layout_owns_static_routes_and_exact_geometry_shapes():
    layout = _face_layout()
    block = _face_block(layout=layout)

    assert block.layout is layout
    assert layout.spatial_shape == block.face_centers.shape
    assert layout.quadrature_shape == block.quadrature_weights.shape
    assert layout.face_count == 2
    assert layout.spatial_dimension == 2
    assert layout.quadrature_count == 2
    with pytest.raises(AttributeError, match="cannot assign"):
        layout.block_id = "changed"
    with pytest.raises(ValueError, match="leading entry per face"):
        _face_layout(spatial_shape=(1, 2))
    with pytest.raises(ValueError, match="spatial_shape"):
        _face_block(spatial_shape=(2, 3))


def test_stage_face_layout_routes_are_array_leaves_with_bounded_static_metadata():
    def make_layout(face_count):
        return FiniteVolumeStageFaceLayout(
            face_ids=np.arange(face_count, dtype=np.int32),
            owner_cells=np.zeros((face_count,), dtype=np.int32),
            neighbour_cells=np.full((face_count,), -1, dtype=np.int32),
            active_mask=np.ones((face_count,), dtype=bool),
            boundary_policy_ids=np.zeros((face_count,), dtype=np.int32),
            boundary_policy_count=1,
            spatial_shape=(face_count, 2),
            quadrature_shape=(face_count, 1),
            block_id="scalable-face-layout",
        )

    small = make_layout(2)
    large = make_layout(1024)

    @eqx.filter_jit
    def jit_identity(layout):
        return layout

    jitted = jit_identity(small)
    small_leaves, small_definition = jax.tree_util.tree_flatten(small)
    large_leaves, large_definition = jax.tree_util.tree_flatten(large)

    assert len(small_leaves) == len(large_leaves) == 5
    assert all(
        leaf is route
        for leaf, route in zip(
            large_leaves,
            (
                large.face_ids,
                large.owner_cells,
                large.neighbour_cells,
                large.active_mask,
                large.boundary_policy_ids,
            ),
            strict=True,
        )
    )
    assert large_definition.num_leaves == small_definition.num_leaves == 5
    assert small.boundary_policy_count == large.boundary_policy_count == 1
    assert jitted.boundary_policy_count == 1
    np.testing.assert_array_equal(
        jitted.boundary_policy_ids,
        small.boundary_policy_ids,
    )
    assert len(repr(large_definition)) <= len(repr(small_definition)) + 32


def test_inactive_solid_cell_may_have_zero_effective_and_coordinate_volume():
    block = _face_block(
        owner_cells=np.asarray((0, 1), dtype=np.int32),
        neighbour_cells=np.asarray((-1, -1), dtype=np.int32),
        active_mask=np.asarray((True, False)),
        area_vectors=np.asarray(((1.0, 0.0), (0.0, 0.0))),
        face_measures=np.asarray((1.0, 0.0)),
        quadrature_weights=np.asarray(((0.5, 0.5), (0.0, 0.0))),
    )
    metrics = _stage_metrics(
        effective_cell_volumes=np.asarray((0.5, 0.0)),
        coordinate_effective_cell_volumes=np.asarray((0.5, 0.0)),
        active_cell_mask=np.asarray((True, False)),
        face_blocks=(block,),
    )

    np.testing.assert_allclose(metrics.effective_cell_volumes, (0.5, 0.0))
    np.testing.assert_allclose(metrics.coordinate_effective_cell_volumes, (0.5, 0.0))
    np.testing.assert_array_equal(metrics.active_cell_mask, (True, False))
    assert metrics.cell_count == 2
    with pytest.raises(AttributeError, match="cannot assign"):
        metrics.geometry_version = jnp.asarray(4, dtype=jnp.int32)


@pytest.mark.parametrize(
    "volume_field",
    ("effective_cell_volumes", "coordinate_effective_cell_volumes"),
)
def test_stage_metrics_reject_nonzero_inactive_volume(volume_field):
    values = {
        "effective_cell_volumes": np.asarray((0.5, 0.0)),
        "coordinate_effective_cell_volumes": np.asarray((0.5, 0.0)),
        "active_cell_mask": np.asarray((True, False)),
        "face_blocks": (),
    }
    values[volume_field] = np.asarray((0.5, 1.0e-12))

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Inactive .* exactly zero"
    ):
        _stage_metrics(**values)


def test_stage_metrics_reject_zero_active_volume_and_misaligned_cell_data():
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Active effective cell volumes"
    ):
        _stage_metrics(effective_cell_volumes=np.asarray((0.5, 0.0)))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonnegative"):
        _stage_metrics(
            effective_cell_volumes=np.asarray((0.5, -0.5)),
            active_cell_mask=np.asarray((True, False)),
        )
    with pytest.raises(ValueError, match="cell_centers"):
        _stage_metrics(cell_centers=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="active_cell_mask"):
        _stage_metrics(active_cell_mask=np.asarray((True,)))
    with pytest.raises(ValueError, match="coordinate_effective_cell_volumes"):
        _stage_metrics(coordinate_effective_cell_volumes=np.asarray((0.5,)))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="Active coordinate-effective"
    ):
        _stage_metrics(coordinate_effective_cell_volumes=np.asarray((0.5, 0.0)))
    with pytest.raises(ValueError, match="mesh_volume_rate"):
        _stage_metrics(mesh_volume_rate=np.zeros((2, 1)))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="outside"):
        _stage_metrics(
            face_blocks=(_face_block(owner_cells=np.asarray((0, 2), dtype=np.int32)),)
        )


def test_stage_metrics_require_exact_zero_mesh_volume_rate_on_inactive_cells():
    evidence = _geometry_evidence()

    @eqx.filter_jit
    def construct(volume_rate):
        return FiniteVolumeStageMetrics(
            topology_epoch_id="epoch-0",
            geometry_family_id="geometry-family-0",
            geometry_layout_id="geometry-layout-0",
            geometry_version=jnp.asarray(3, dtype=jnp.int32),
            time=jnp.asarray(0.25),
            effective_cell_volumes=jnp.asarray((0.5, 0.0)),
            coordinate_effective_cell_volumes=jnp.asarray((0.5, 0.0)),
            mesh_volume_rate=volume_rate,
            cell_centers=jnp.asarray(((2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0))),
            active_cell_mask=jnp.asarray((True, False)),
            face_blocks=(),
            evidence=evidence,
        )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="Inactive mesh_volume_rate entries must be exactly zero",
    ):
        metrics = construct(jnp.asarray((0.0, 1.0e-12)))
        jax.block_until_ready(metrics.mesh_volume_rate)


@pytest.mark.parametrize(
    "missing",
    (
        "geometry_family_id",
        "coordinate_effective_cell_volumes",
        "mesh_volume_rate",
        "evidence",
    ),
)
def test_stage_metrics_require_geometry_data_and_typed_evidence(missing):
    values = _stage_metric_values()
    values.pop(missing)
    with pytest.raises(TypeError):
        FiniteVolumeStageMetrics(**values)

    if missing == "evidence":
        values["evidence"] = "successful"
        with pytest.raises(TypeError, match="FiniteVolumeStageGeometryEvidence"):
            FiniteVolumeStageMetrics(**values)


@pytest.mark.parametrize(
    "tolerance_field",
    (
        "coordinate_effective_volume_tolerance",
        "face_closure_tolerance",
        "gcl_identity_tolerance",
    ),
)
def test_geometry_evidence_tolerances_are_nonnegative_per_cell(tolerance_field):
    with pytest.raises(ValueError, match="one entry per cell"):
        _geometry_evidence(**{tolerance_field: np.zeros((1,))})
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="nonnegative"):
        _geometry_evidence(**{tolerance_field: np.asarray((0.0, -1.0e-10))})


def test_zero_mixed_tolerances_produce_safe_finite_retry_reductions():
    policy = ALEGeometryConsistencyPolicy(
        absolute_tolerance=0.0,
        relative_tolerance=0.5,
        reduction_safety_factor=0.8,
        minimum_reduction_factor=0.1,
    )
    evidence = policy.evidence(
        coordinate_effective_volume_defect=np.asarray((0.0, 0.0)),
        coordinate_effective_volume_reference=np.asarray((0.0, 0.0)),
        face_closure_defect=np.asarray((0.0, 2.0)),
        face_closure_reference=np.asarray((0.0, 2.0)),
        gcl_identity_defect=np.asarray((0.0, 0.0)),
        gcl_identity_reference=np.asarray((0.0, 0.0)),
        expected_order=2,
        evidence_version=0,
    )

    assert not bool(evidence.passed)
    assert np.isfinite(float(evidence.proposed_reduction_factor))
    assert float(evidence.proposed_reduction_factor) == pytest.approx(0.8 / np.sqrt(2.0))


@pytest.mark.parametrize(
    "defect_field",
    (
        "coordinate_effective_volume_defect",
        "face_closure_defect",
        "gcl_identity_defect",
    ),
)
def test_positive_defect_over_zero_tolerance_uses_infinite_ratio_and_finite_floor(
    defect_field,
):
    policy = ALEGeometryConsistencyPolicy(
        absolute_tolerance=0.0,
        relative_tolerance=1.0,
        reduction_safety_factor=0.9,
        minimum_reduction_factor=0.125,
    )

    @jax.jit
    def certify(defect):
        zero_defect = jnp.zeros((2,))
        defects = {
            "coordinate_effective_volume_defect": zero_defect,
            "face_closure_defect": zero_defect,
            "gcl_identity_defect": zero_defect,
        }
        defects[defect_field] = jnp.stack((jnp.zeros_like(defect), defect))
        evidence = policy.evidence(
            **defects,
            coordinate_effective_volume_reference=jnp.zeros((2,)),
            face_closure_reference=jnp.zeros((2,)),
            gcl_identity_reference=jnp.zeros((2,)),
            expected_order=2,
            evidence_version=jnp.asarray(3, dtype=jnp.int32),
        )
        return evidence.passed, evidence.proposed_reduction_factor

    exact_passed, exact_reduction = certify(jnp.asarray(0.0))
    failed, failed_reduction = certify(jnp.asarray(1.0))
    assert bool(exact_passed)
    assert float(exact_reduction) == 1.0
    assert not bool(failed)
    assert np.isfinite(float(failed_reduction))
    assert float(failed_reduction) == pytest.approx(0.125)


@pytest.mark.parametrize(
    "defect_field",
    (
        "coordinate_effective_volume_defect",
        "face_closure_defect",
        "gcl_identity_defect",
    ),
)
def test_each_geometry_defect_threshold_mechanically_fails_evidence(defect_field):
    evidence = _geometry_evidence(
        **{
            defect_field: np.asarray((0.0, 1.0e-9)),
            "proposed_reduction_factor": 0.5,
            "passed": False,
            "status": FiniteVolumeGeometryStatus.FAILED,
        }
    )

    assert not bool(evidence.passed)
    assert int(evidence.status) == int(FiniteVolumeGeometryStatus.FAILED)
    assert int(evidence.expected_order) == 2
    assert float(evidence.proposed_reduction_factor) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "overrides, message",
    (
        (
            {
                "coordinate_effective_volume_defect": np.asarray((0.0, 1.0e-9)),
                "passed": True,
                "status": FiniteVolumeGeometryStatus.SUCCESS,
            },
            "passed must equal",
        ),
        (
            {
                "coordinate_effective_volume_defect": np.asarray((0.0, 1.0e-9)),
                "passed": False,
                "status": FiniteVolumeGeometryStatus.SUCCESS,
            },
            "status must equal",
        ),
        (
            {
                "passed": False,
                "status": FiniteVolumeGeometryStatus.FAILED,
            },
            "passed must equal",
        ),
        (
            {
                "passed": True,
                "status": FiniteVolumeGeometryStatus.FAILED,
            },
            "status must equal",
        ),
    ),
)
def test_geometry_evidence_rejects_pass_or_status_inconsistent_with_thresholds(
    overrides, message
):
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match=message):
        _geometry_evidence(**overrides)


def test_geometry_evidence_accepts_defects_exactly_at_every_threshold():
    coordinate_tolerance = np.asarray((2.0e-12, 3.0e-12))
    closure_tolerance = np.asarray((4.0e-12, 5.0e-12))
    gcl_tolerance = np.asarray((6.0e-12, 7.0e-12))

    evidence = _geometry_evidence(
        coordinate_effective_volume_defect=coordinate_tolerance,
        coordinate_effective_volume_tolerance=coordinate_tolerance,
        face_closure_defect=closure_tolerance,
        face_closure_tolerance=closure_tolerance,
        gcl_identity_defect=gcl_tolerance,
        gcl_identity_tolerance=gcl_tolerance,
    )

    assert bool(evidence.passed)
    assert int(evidence.status) == int(FiniteVolumeGeometryStatus.SUCCESS)


def test_stage_metrics_carry_typed_failed_dynamic_evidence():
    failed = _geometry_evidence(
        coordinate_effective_volume_defect=np.asarray((0.0, 2.5e-3)),
        gcl_identity_defect=np.asarray((0.0, 1.0e-3)),
        proposed_reduction_factor=0.5,
        passed=False,
        status=FiniteVolumeGeometryStatus.FAILED,
    )
    metrics = _stage_metrics(evidence=failed)

    assert not bool(metrics.evidence.passed)
    assert int(metrics.evidence.status) == int(FiniteVolumeGeometryStatus.FAILED)
    np.testing.assert_allclose(
        metrics.evidence.coordinate_effective_volume_defect,
        (0.0, 2.5e-3),
    )
    assert float(metrics.evidence.proposed_reduction_factor) == pytest.approx(0.5)
    successful = _stage_metrics()
    assert jax.tree_util.tree_structure(metrics) == jax.tree_util.tree_structure(
        successful
    )


def test_filter_jit_constructs_and_certifies_dynamic_stage_metrics():
    layout = _face_layout()

    @eqx.filter_jit
    def certify(time, geometry_version, evidence_version, defect, active_cell_mask):
        face_block = FiniteVolumeStageFaceBlock(
            layout=layout,
            face_centers=jnp.asarray(((0.5, 0.5), (0.0, 0.5))) + 0.0 * time,
            area_vectors=jnp.asarray(((1.0, 0.0), (-1.0, 0.0))),
            face_measures=jnp.ones((2,)),
            quadrature_points=jnp.asarray(
                (
                    ((0.5, 0.25), (0.5, 0.75)),
                    ((0.0, 0.25), (0.0, 0.75)),
                )
            )
            + 0.0 * time,
            quadrature_weights=jnp.full((2, 2), 0.5),
            quadrature_grid_normal_velocity=jnp.full((2, 2), time),
        )
        defect_values = jnp.stack((jnp.zeros_like(defect), defect))
        tolerances = jnp.full((2,), 1.0e-3)
        passed = jnp.all(defect_values <= tolerances)
        status = jnp.where(
            passed,
            int(FiniteVolumeGeometryStatus.SUCCESS),
            int(FiniteVolumeGeometryStatus.FAILED),
        )
        evidence = FiniteVolumeStageGeometryEvidence(
            coordinate_effective_volume_defect=defect_values,
            coordinate_effective_volume_tolerance=tolerances,
            face_closure_defect=jnp.zeros((2,)),
            face_closure_tolerance=tolerances,
            gcl_identity_defect=jnp.zeros((2,)),
            gcl_identity_tolerance=tolerances,
            expected_order=jnp.asarray(2, dtype=jnp.int32),
            proposed_reduction_factor=jnp.where(passed, 1.0, 0.5),
            passed=passed,
            status=status,
            evidence_version=evidence_version,
            policy_id="stage-geometry-policy",
        )
        return FiniteVolumeStageMetrics(
            topology_epoch_id="epoch-0",
            geometry_family_id="geometry-family-0",
            geometry_layout_id="geometry-layout-0",
            geometry_version=geometry_version,
            time=time,
            effective_cell_volumes=jnp.asarray((0.5, 0.5)),
            coordinate_effective_cell_volumes=jnp.asarray((0.5, 0.5)),
            mesh_volume_rate=jnp.zeros((2,)),
            cell_centers=jnp.asarray(((2.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0)))
            + 0.0 * time,
            active_cell_mask=active_cell_mask,
            face_blocks=(face_block,),
            evidence=evidence,
        )

    successful = certify(
        jnp.asarray(0.25),
        jnp.asarray(7, dtype=jnp.int32),
        jnp.asarray(11, dtype=jnp.int32),
        jnp.asarray(5.0e-4),
        jnp.asarray((True, True)),
    )
    failed = certify(
        jnp.asarray(0.5),
        jnp.asarray(8, dtype=jnp.int32),
        jnp.asarray(12, dtype=jnp.int32),
        jnp.asarray(2.0e-3),
        jnp.asarray((True, True)),
    )

    assert jax.tree_util.tree_structure(successful) == jax.tree_util.tree_structure(
        failed
    )
    assert successful.geometry_family_id == failed.geometry_family_id
    assert successful.geometry_layout_id == failed.geometry_layout_id
    assert (
        successful.face_blocks[0].layout.block_id == failed.face_blocks[0].layout.block_id
    )
    np.testing.assert_array_equal(
        successful.face_blocks[0].layout.face_ids,
        failed.face_blocks[0].layout.face_ids,
    )
    assert successful.evidence.policy_id == failed.evidence.policy_id
    assert float(successful.time) == pytest.approx(0.25)
    assert float(failed.time) == pytest.approx(0.5)
    assert int(successful.geometry_version) == 7
    assert int(failed.geometry_version) == 8
    assert int(successful.evidence.evidence_version) == 11
    assert int(failed.evidence.evidence_version) == 12
    assert bool(successful.evidence.passed)
    assert not bool(failed.evidence.passed)
    assert int(failed.evidence.status) == int(FiniteVolumeGeometryStatus.FAILED)
    np.testing.assert_allclose(
        failed.evidence.coordinate_effective_volume_defect,
        (0.0, 2.0e-3),
    )
    assert float(failed.evidence.proposed_reduction_factor) == pytest.approx(0.5)


def test_dynamic_geometry_versions_do_not_change_static_tree_metadata():
    first = _stage_metrics(
        geometry_version=7,
        evidence=_geometry_evidence(evidence_version=17),
    )
    second = _stage_metrics(
        geometry_version=8,
        time=0.5,
        effective_cell_volumes=np.asarray((0.6, 0.4)),
        coordinate_effective_cell_volumes=np.asarray((0.6, 0.4)),
        evidence=_geometry_evidence(
            face_closure_tolerance=np.full((2,), 2.0e-10),
            expected_order=4,
            proposed_reduction_factor=0.75,
            evidence_version=18,
        ),
    )

    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)
    assert first.geometry_family_id == second.geometry_family_id
    assert first.geometry_layout_id == second.geometry_layout_id
    assert first.evidence.policy_id == second.evidence.policy_id
    assert int(first.geometry_version) == 7
    assert int(second.geometry_version) == 8
    assert int(first.evidence.evidence_version) == 17
    assert int(second.evidence.evidence_version) == 18
    assert int(first.evidence.expected_order) == 2
    assert int(second.evidence.expected_order) == 4
    assert float(second.evidence.proposed_reduction_factor) == pytest.approx(0.75)


def test_active_face_routes_must_own_only_active_cells():
    inactive_face = {
        "active_mask": np.asarray((True, False)),
        "area_vectors": np.asarray(((1.0, 0.0), (0.0, 0.0))),
        "face_measures": np.asarray((1.0, 0.0)),
        "quadrature_weights": np.asarray(((0.5, 0.5), (0.0, 0.0))),
        "neighbour_cells": np.asarray((-1, -1), dtype=np.int32),
    }
    inactive_cells = {
        "effective_cell_volumes": np.asarray((0.5, 0.0)),
        "coordinate_effective_cell_volumes": np.asarray((0.5, 0.0)),
        "active_cell_mask": np.asarray((True, False)),
    }
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="active face owner"):
        _stage_metrics(
            **inactive_cells,
            face_blocks=(
                _face_block(
                    **inactive_face,
                    owner_cells=np.asarray((1, 0), dtype=np.int32),
                ),
            ),
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="internal face neighbour"
    ):
        _stage_metrics(
            **inactive_cells,
            face_blocks=(
                _face_block(
                    **{
                        **inactive_face,
                        "neighbour_cells": np.asarray((1, -1), dtype=np.int32),
                    },
                    owner_cells=np.asarray((0, 1), dtype=np.int32),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("topology_epoch_id", " epoch"),
        ("geometry_family_id", "family "),
        ("geometry_layout_id", "layout "),
        ("topology_epoch_id", object()),
    ),
)
def test_stage_metrics_reject_noncanonical_static_ids(field, value):
    with pytest.raises(ValueError, match="canonical stripped string"):
        _stage_metrics(**{field: value})

    with pytest.raises(ValueError, match="canonical stripped string"):
        _geometry_evidence(policy_id=" policy")
    with pytest.raises(ValueError, match="canonical stripped string"):
        _face_block(block_id="face-block ")
    with pytest.raises(ValueError, match="canonical stripped string"):
        lower_static_unstructured_stage_metrics(
            _unstructured(),
            topology_epoch_id=" epoch",
        )


@pytest.mark.parametrize(
    "vertices",
    (
        ((2.0, -3.0), (3.0, -3.0), (3.0, -2.0), (2.0, -2.0)),
        ((0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)),
    ),
)
def test_static_stage_family_binds_equal_volume_geometry_not_route_layout(vertices):
    base = _unstructured()
    changed = _unstructured(vertices)
    first = lower_static_unstructured_stage_metrics(base, topology_epoch_id="epoch-a")
    next_epoch = lower_static_unstructured_stage_metrics(
        base, topology_epoch_id="epoch-b"
    )
    moved = lower_static_unstructured_stage_metrics(
        changed,
        topology_epoch_id="epoch-a",
    )

    np.testing.assert_allclose(first.effective_cell_volumes, moved.effective_cell_volumes)
    assert first.geometry_family_id == base.geometry_id
    assert moved.geometry_family_id == changed.geometry_id
    assert first.geometry_family_id == next_epoch.geometry_family_id
    assert first.geometry_family_id != moved.geometry_family_id
    assert first.geometry_layout_id == next_epoch.geometry_layout_id
    assert first.geometry_layout_id == moved.geometry_layout_id
    assert (
        first.face_blocks[0].layout.block_id != next_epoch.face_blocks[0].layout.block_id
    )
    assert first.face_blocks[0].layout.block_id == moved.face_blocks[0].layout.block_id
    assert first.evidence.policy_id == next_epoch.evidence.policy_id
    assert first.evidence.policy_id == moved.evidence.policy_id
    assert int(first.geometry_version) == 0
    assert int(next_epoch.geometry_version) == 0
    assert int(moved.geometry_version) == 0


def test_stage_contracts_do_not_change_structured_or_mapped_protocol_conformance():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(3),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    structured = phx.discretization.FiniteVolumePlan(grid).prepare()
    mapped = phx.discretization.MappedFiniteVolumePlan(
        structured,
        lambda point: point,
        mapping_id="stage-metrics-protocol-identity",
    ).prepare()
    explicit = _unstructured()

    assert isinstance(structured, PreparedFiniteVolumeGeometry)
    assert isinstance(mapped, PreparedFiniteVolumeGeometry)
    assert not isinstance(structured, ExplicitFaceBlockGeometry)
    assert not isinstance(mapped, ExplicitFaceBlockGeometry)
    assert isinstance(explicit, ExplicitFaceBlockGeometry)
