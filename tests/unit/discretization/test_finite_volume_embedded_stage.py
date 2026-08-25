#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.discretization.finite_volume._unstructured_embedded_boundary as embedded_boundary_module
from phydrax.discretization.finite_volume._boundary import ExtrapolationBoundary
from phydrax.discretization.finite_volume._embedded_dynamics import (
    lower_embedded_stage_metrics,
    UnstructuredEmbeddedBoundarySet,
)
from phydrax.discretization.finite_volume._geometry_protocol import (
    lower_static_unstructured_stage_metrics,
)
from phydrax.discretization.finite_volume._physical_boundaries import SlipWallBoundary


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


def _embedded(discretization, level_set, field_id, *, body_tag=7):
    return phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set,
        field_id=field_id,
        body_tag=body_tag,
    ).prepare()


def _lower(
    discretization,
    metrics,
    *,
    time=0.0,
    geometry_version=4,
    evidence_version=9,
    topology_epoch_id="epoch:stationary-eb",
):
    return lower_embedded_stage_metrics(
        discretization,
        metrics,
        UnstructuredEmbeddedBoundarySet({7: SlipWallBoundary()}),
        topology_epoch_id,
        time=time,
        geometry_version=geometry_version,
        evidence_version=evidence_version,
    )


def _owner_oriented_closure(stage):
    closure = np.zeros_like(np.asarray(stage.cell_centers))
    for block in stage.face_blocks:
        area = np.asarray(block.area_vectors)
        owner = np.asarray(block.layout.owner_cells)
        neighbour = np.asarray(block.layout.neighbour_cells)
        np.add.at(closure, owner, area)
        internal = neighbour >= 0
        np.add.at(closure, neighbour[internal], -area[internal])
    return closure


def test_embedded_stage_compacts_open_physical_and_cut_blocks_with_closure():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        "vertical-stationary-cut",
    )
    stage = _lower(discretization, metrics)

    assert tuple(block.layout.block_kind for block in stage.face_blocks) == (
        "physical",
        "cut",
    )
    physical, cut = stage.face_blocks
    expected_cut_cells = np.flatnonzero(
        np.asarray(metrics.cut_face_active)
        & np.asarray(metrics.active_fluid_cells)
        & (np.asarray(metrics.cut_face_measures) > 0.0)
    ).astype(np.int32)
    assert cut.layout.face_count == expected_cut_cells.size
    np.testing.assert_array_equal(cut.layout.owner_cells, expected_cut_cells)
    np.testing.assert_array_equal(
        cut.layout.face_ids,
        discretization.face_measures.size + expected_cut_cells,
    )
    np.testing.assert_array_equal(
        cut.layout.neighbour_cells,
        np.full(expected_cut_cells.shape, -1, dtype=np.int32),
    )
    assert all(
        np.all(np.asarray(block.layout.active_mask)) for block in stage.face_blocks
    )
    assert all(
        np.all(np.asarray(block.face_measures) > 0.0) for block in stage.face_blocks
    )
    np.testing.assert_allclose(_owner_oriented_closure(stage), 0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        jnp.sum(physical.quadrature_weights, axis=1),
        physical.face_measures,
    )
    np.testing.assert_allclose(
        jnp.sum(cut.quadrature_weights, axis=1),
        cut.face_measures,
    )
    np.testing.assert_array_equal(
        cut.quadrature_grid_normal_velocity,
        jnp.zeros_like(cut.quadrature_grid_normal_velocity),
    )
    assert stage.evidence.passed
    assert stage.evidence.evidence_version == 9


def test_embedded_body_policies_are_complete_typed_and_deterministically_routed():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        "tagged-cut",
    )
    wall = SlipWallBoundary()
    first = UnstructuredEmbeddedBoundarySet({7: wall})
    second = UnstructuredEmbeddedBoundarySet({np.int32(7): SlipWallBoundary()})
    stage = lower_embedded_stage_metrics(
        discretization,
        metrics,
        first,
        "epoch:tagged-cut",
        time=0.0,
        geometry_version=0,
        evidence_version=0,
    )

    assert first.body_tags == (7,)
    assert first.boundary_set_id == second.boundary_set_id
    assert isinstance(first.boundaries[0], SlipWallBoundary)
    cut = stage.face_blocks[-1]
    np.testing.assert_array_equal(
        cut.layout.boundary_policy_ids,
        np.zeros(cut.layout.owner_cells.shape, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        metrics.body_tags[cut.layout.owner_cells],
        np.full(cut.layout.owner_cells.shape, 7, dtype=np.int32),
    )
    with pytest.raises(TypeError, match="SlipWallBoundary"):
        UnstructuredEmbeddedBoundarySet({7: ExtrapolationBoundary()})
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="cover every metric body tag exactly",
    ):
        lower_embedded_stage_metrics(
            discretization,
            metrics,
            UnstructuredEmbeddedBoundarySet({8: SlipWallBoundary()}),
            "epoch:tagged-cut",
            time=0.0,
            geometry_version=0,
            evidence_version=0,
        )


def test_embedded_stage_realizes_clipped_fluid_centroids_in_float32_evidence():
    with jax.enable_x64(False):
        discretization = _quadrilateral_strip()
        metrics = _embedded(
            discretization,
            lambda points, args: points[:, 0] + 0.25 * points[:, 1] - 0.3,
            "asymmetric-fluid-centroid",
        )
        stage = _lower(discretization, metrics)

    expected_cut_centroid = jnp.asarray((257.0 / 780.0, 22.0 / 39.0))
    assert metrics.fluid_cell_centers.dtype == jnp.float32
    assert stage.cell_centers.dtype == jnp.float32
    for value in (
        metrics.evidence.volume_closure_defect,
        metrics.evidence.volume_closure_tolerance,
        metrics.evidence.aperture_closure_defect,
        metrics.evidence.aperture_closure_tolerance,
        metrics.evidence.cut_face_closure_defect,
        metrics.evidence.cut_face_closure_tolerance,
        metrics.evidence.minimum_nonzero_volume_fraction,
    ):
        assert value.dtype == jnp.float32
    assert stage.geometry_family_id == discretization.geometry_id
    assert bool(metrics.cut_cells[0])
    assert bool(metrics.active_fluid_cells[0])
    np.testing.assert_allclose(
        metrics.fluid_cell_centers[0],
        expected_cut_centroid,
        rtol=8.0e-7,
        atol=2.0e-7,
    )
    np.testing.assert_array_equal(stage.cell_centers[0], metrics.fluid_cell_centers[0])
    np.testing.assert_array_equal(
        stage.cell_centers[1],
        jnp.asarray(
            discretization.cell_centers[1],
            dtype=stage.cell_centers.dtype,
        ),
    )


def test_embedded_metrics_identity_binds_realized_fluid_centroid_payload(monkeypatch):
    discretization = _quadrilateral_strip()

    def level_set(points, args):
        del args
        return points[:, 0] - 0.75

    baseline = _embedded(
        discretization,
        level_set,
        "centroid-identity",
    )
    polygon_measure_centroid = embedded_boundary_module._polygon_measure_centroid

    def shifted_polygon_measure_centroid(vertices):
        measure, centroid = polygon_measure_centroid(vertices)
        if measure > 0.0:
            centroid = centroid + np.asarray((0.0, 1.0 / 32.0))
        return measure, centroid

    monkeypatch.setattr(
        embedded_boundary_module,
        "_polygon_measure_centroid",
        shifted_polygon_measure_centroid,
    )
    shifted = _embedded(
        discretization,
        level_set,
        "centroid-identity",
    )

    np.testing.assert_array_equal(
        shifted.volume_fraction,
        baseline.volume_fraction,
    )
    np.testing.assert_array_equal(
        shifted.cut_face_centers,
        baseline.cut_face_centers,
    )
    np.testing.assert_array_equal(
        shifted.cut_face_normals,
        baseline.cut_face_normals,
    )
    assert not jnp.array_equal(
        shifted.fluid_cell_centers,
        baseline.fluid_cell_centers,
    )
    assert shifted.metrics_id != baseline.metrics_id


def test_full_fluid_embedded_stage_preserves_static_physical_geometry():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: jnp.ones((points.shape[0],)),
        "full-fluid",
    )
    embedded = _lower(discretization, metrics)
    static = lower_static_unstructured_stage_metrics(
        discretization,
        topology_epoch_id="epoch:stationary-eb",
    )

    np.testing.assert_array_equal(
        embedded.effective_cell_volumes,
        static.effective_cell_volumes,
    )
    np.testing.assert_array_equal(embedded.active_cell_mask, static.active_cell_mask)
    np.testing.assert_array_equal(embedded.cell_centers, static.cell_centers)
    assert embedded.geometry_family_id == discretization.geometry_id
    assert embedded.geometry_family_id == static.geometry_family_id
    assert len(embedded.face_blocks) == len(static.face_blocks)
    assert all(block.layout.block_kind == "physical" for block in embedded.face_blocks)
    for embedded_block, static_block in zip(
        embedded.face_blocks, static.face_blocks, strict=True
    ):
        np.testing.assert_array_equal(
            embedded_block.layout.face_ids,
            static_block.layout.face_ids,
        )
        np.testing.assert_array_equal(
            embedded_block.layout.owner_cells,
            static_block.layout.owner_cells,
        )
        np.testing.assert_array_equal(
            embedded_block.layout.neighbour_cells,
            static_block.layout.neighbour_cells,
        )
        np.testing.assert_array_equal(
            embedded_block.layout.boundary_policy_ids,
            discretization.face_block.boundary_patch_ids,
        )
        np.testing.assert_array_equal(
            embedded_block.face_centers,
            static_block.face_centers,
        )
        np.testing.assert_array_equal(
            embedded_block.area_vectors,
            static_block.area_vectors,
        )
        np.testing.assert_array_equal(
            embedded_block.face_measures,
            static_block.face_measures,
        )
        np.testing.assert_array_equal(
            embedded_block.quadrature_points,
            static_block.quadrature_points,
        )
        np.testing.assert_array_equal(
            embedded_block.quadrature_weights,
            static_block.quadrature_weights,
        )


def test_full_solid_embedded_stage_has_no_face_blocks_or_routed_physics():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: -jnp.ones((points.shape[0],)),
        "full-solid",
    )
    stage = _lower(discretization, metrics)

    assert not jnp.any(stage.active_cell_mask)
    assert jnp.all(jnp.isfinite(stage.cell_centers))
    np.testing.assert_array_equal(
        stage.cell_centers,
        jnp.zeros_like(stage.cell_centers),
    )
    np.testing.assert_array_equal(
        stage.effective_cell_volumes,
        jnp.zeros_like(stage.effective_cell_volumes),
    )
    np.testing.assert_array_equal(
        stage.coordinate_effective_cell_volumes,
        jnp.zeros_like(stage.coordinate_effective_cell_volumes),
    )
    np.testing.assert_array_equal(
        stage.mesh_volume_rate,
        jnp.zeros_like(stage.mesh_volume_rate),
    )
    assert stage.face_blocks == ()
    assert sum(block.layout.face_count for block in stage.face_blocks) == 0


def test_embedded_stage_routes_and_ids_are_stable_across_dynamic_versions():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        "stable-route-cut",
    )
    first = _lower(discretization, metrics, geometry_version=3, evidence_version=5)
    second = _lower(discretization, metrics, geometry_version=4, evidence_version=6)
    other_epoch = _lower(
        discretization,
        metrics,
        geometry_version=4,
        evidence_version=6,
        topology_epoch_id="epoch:stationary-eb:next",
    )

    assert first.topology_epoch_id == "epoch:stationary-eb"
    assert other_epoch.topology_epoch_id == "epoch:stationary-eb:next"
    assert first.geometry_layout_id == second.geometry_layout_id
    assert first.geometry_layout_id == other_epoch.geometry_layout_id
    assert first.geometry_version == 3
    assert second.geometry_version == 4
    assert first.evidence.evidence_version == 5
    assert second.evidence.evidence_version == 6
    for first_block, second_block, other_block in zip(
        first.face_blocks,
        second.face_blocks,
        other_epoch.face_blocks,
        strict=True,
    ):
        assert first_block.layout.block_id == second_block.layout.block_id
        assert first_block.layout.block_id != other_block.layout.block_id
        assert first_block.layout.block_kind == second_block.layout.block_kind
        np.testing.assert_array_equal(
            first_block.layout.face_ids,
            second_block.layout.face_ids,
        )
        np.testing.assert_array_equal(
            first_block.layout.face_ids,
            other_block.layout.face_ids,
        )
        np.testing.assert_array_equal(
            first_block.layout.owner_cells,
            second_block.layout.owner_cells,
        )
        np.testing.assert_array_equal(
            first_block.layout.neighbour_cells,
            second_block.layout.neighbour_cells,
        )
        np.testing.assert_array_equal(
            first_block.layout.boundary_policy_ids,
            second_block.layout.boundary_policy_ids,
        )


def test_mixed_solid_blocks_contain_only_remapped_positive_measure_routes():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        "compacted-route-cut",
    )
    physical, cut = _lower(discretization, metrics).face_blocks

    for block in (physical, cut):
        assert np.all(np.asarray(block.layout.active_mask))
        assert np.all(np.asarray(block.face_measures) > 0.0)
        assert block.layout.face_count == block.face_measures.size

    source = discretization.face_block
    source_ids = np.asarray(source.face_ids)
    owners = np.asarray(source.owner_cells)
    neighbours = np.asarray(source.neighbour_cells)
    internal = neighbours >= 0
    safe_neighbours = np.where(internal, neighbours, 0)
    active_cells = np.asarray(metrics.active_fluid_cells)
    expected_physical = (
        np.asarray(source.active_mask)
        & (np.asarray(metrics.face_open_fraction)[source_ids] > 0.0)
        & active_cells[owners]
        & (~internal | active_cells[safe_neighbours])
    )
    expected_face_ids = source_ids[expected_physical]
    np.testing.assert_array_equal(physical.layout.face_ids, expected_face_ids)
    np.testing.assert_array_equal(
        physical.layout.boundary_policy_ids,
        np.asarray(source.boundary_patch_ids)[expected_physical],
    )
    expected_cut_cells = np.flatnonzero(
        np.asarray(metrics.cut_face_active) & np.asarray(metrics.active_fluid_cells)
    ).astype(np.int32)
    np.testing.assert_array_equal(cut.layout.owner_cells, expected_cut_cells)
    np.testing.assert_array_equal(
        cut.layout.face_ids,
        discretization.face_measures.size + expected_cut_cells,
    )
    np.testing.assert_array_equal(
        cut.layout.boundary_policy_ids,
        np.zeros(expected_cut_cells.shape, dtype=np.int32),
    )


def test_partial_physical_face_quadrature_integrates_over_actual_open_segment():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        "open-segment-quadrature",
    )
    physical = _lower(discretization, metrics).face_blocks[0]

    face_ids = np.asarray(physical.layout.face_ids)
    fractions = np.asarray(metrics.face_open_fraction)[face_ids]
    partial_boundary = (
        (fractions > 0.0)
        & (fractions < 1.0)
        & (np.asarray(physical.layout.neighbour_cells) < 0)
    )
    assert np.any(partial_boundary)
    partial_ids = face_ids[partial_boundary]
    edge_vertex_ids = np.asarray(discretization.connectivity.edges)[partial_ids]
    edge_points = np.asarray(discretization.vertices)[edge_vertex_ids]
    edge_values = np.asarray(metrics.vertex_values)[edge_vertex_ids]
    partial_fractions = fractions[partial_boundary]
    tangent = edge_points[:, 1] - edge_points[:, 0]
    open_start = edge_points[:, 0].copy()
    open_stop = edge_points[:, 1].copy()
    start_is_fluid = edge_values[:, 0] >= 0.0
    open_stop[start_is_fluid] = (
        edge_points[start_is_fluid, 0]
        + partial_fractions[start_is_fluid, None] * tangent[start_is_fluid]
    )
    open_start[~start_is_fluid] = (
        edge_points[~start_is_fluid, 1]
        - partial_fractions[~start_is_fluid, None] * tangent[~start_is_fluid]
    )
    open_centers = 0.5 * (open_start + open_stop)
    open_measures = np.linalg.norm(open_stop - open_start, axis=1)
    quadrature_points = np.asarray(physical.quadrature_points)[partial_boundary]
    quadrature_weights = np.asarray(physical.quadrature_weights)[partial_boundary]
    integrand = 1.25 + 2.0 * quadrature_points[..., 0] - 0.75 * quadrature_points[..., 1]
    exact = open_measures * (1.25 + 2.0 * open_centers[:, 0] - 0.75 * open_centers[:, 1])

    np.testing.assert_allclose(
        np.asarray(physical.face_centers)[partial_boundary],
        open_centers,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(quadrature_weights * integrand, axis=1),
        exact,
        atol=2.0e-12,
    )


def test_cut_stage_area_vectors_preserve_metric_outward_normals():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] + 0.25 * points[:, 1] - 0.8,
        "oblique-cut-normal",
    )
    cut = _lower(discretization, metrics).face_blocks[-1]
    active = cut.layout.active_mask

    stage_normals = cut.area_vectors[active] / cut.face_measures[active, None]
    np.testing.assert_allclose(
        stage_normals,
        metrics.cut_face_normals[metrics.cut_face_active],
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(stage_normals, axis=-1),
        1.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        cut.quadrature_points[active, 0],
        metrics.cut_face_centers[metrics.cut_face_active],
    )


def test_embedded_stage_policy_routes_remain_jax_leaves_under_jit():
    discretization = _quadrilateral_strip()
    metrics = _embedded(
        discretization,
        lambda points, args: points[:, 0] - 0.75,
        "jit-stage-leaves",
    )
    boundary_set = UnstructuredEmbeddedBoundarySet({7: SlipWallBoundary()})
    stage = lower_embedded_stage_metrics(
        discretization,
        metrics,
        boundary_set,
        "epoch:stationary-eb",
        jnp.asarray(11, dtype=jnp.int32),
        jnp.asarray(13, dtype=jnp.int32),
        time=jnp.asarray(0.375),
    )

    with pytest.raises(AttributeError, match="cannot assign"):
        stage.face_blocks[-1].layout.boundary_policy_ids = jnp.ones_like(
            stage.face_blocks[-1].layout.boundary_policy_ids
        )
    lowered = jax.jit(
        lambda value: (
            value.geometry_version,
            value.evidence.evidence_version,
            value.face_blocks[0].layout.boundary_policy_ids,
            value.face_blocks[-1].layout.boundary_policy_ids,
            value.face_blocks[-1].quadrature_weights,
        )
    )(stage)
    assert lowered[0] == 11
    assert lowered[1] == 13
    assert stage.time == pytest.approx(0.375)
    np.testing.assert_array_equal(
        lowered[2], stage.face_blocks[0].layout.boundary_policy_ids
    )
    np.testing.assert_array_equal(
        lowered[3], stage.face_blocks[-1].layout.boundary_policy_ids
    )
    np.testing.assert_array_equal(lowered[4], stage.face_blocks[-1].quadrature_weights)
    leaves = jax.tree_util.tree_leaves(stage)
    assert any(
        leaf is stage.face_blocks[-1].layout.boundary_policy_ids for leaf in leaves
    )
