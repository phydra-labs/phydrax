from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.discretization.finite_volume import UnstructuredFiniteVolumePlan
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._piecewise_linear_boundary import (
    PiecewiseLinearBoundarySource,
    PiecewiseLinearDomainBoundaryOwner,
)
from phydrax.geometry._planar_embedding import PlanarEmbedding
from phydrax.geometry.brep._patches import PlanePatch
from phydrax.meshing import NativeMeshingProvider
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    MeshQualityTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    BlockInterfaceControl,
    LayerSchedule,
    TransfiniteCurveControl,
    TransfiniteSurfaceControl,
)
from phydrax.meshing._multiblock import glue_structured_blocks
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._structured import (
    generate_structured_block,
    logical_cell_bank,
    TransfiniteBlock,
)
from phydrax.meshing._sweep import (
    column_connectivity,
    generate_sweep,
    SweepControl,
    SweepMapKind,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import NativeStructuredSchedule
from phydrax.meshing.providers._native_sources import (
    NativeStructuredSource,
    NativeSweepSource,
)
from phydrax.meshing.providers._native_structured import (
    _size_compliance,
    prepare_native_structured,
    sweep_support_issues,
)


def _curve(
    points: tuple[tuple[float, ...], ...],
    /,
    *,
    source_id: str,
    intervals: int,
) -> TransfiniteCurveControl:
    degree = len(points) - 1
    knots = (0.0,) * (degree + 1) + (1.0,) * (degree + 1)
    curve = phx.geometry.BSplineCurve(
        np.asarray(points, dtype=np.float64),
        np.ones((len(points),), dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        degree,
    )
    return TransfiniteCurveControl(curve, source_id, (0.0, 1.0), intervals)


def _rectangle(
    name: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    /,
    *,
    x_intervals: int = 2,
    y_intervals: int = 2,
    curved_bottom: bool = False,
) -> TransfiniteBlock:
    bottom_points = (
        ((x0, y0), ((x0 + x1) / 2.0, y0 - 0.2), (x1, y0))
        if curved_bottom
        else ((x0, y0), (x1, y0))
    )
    return TransfiniteBlock(
        name,
        (
            _curve(((x0, y0), (x0, y1)), source_id=f"{name}:left", intervals=y_intervals),
            _curve(
                ((x1, y0), (x1, y1)), source_id=f"{name}:right", intervals=y_intervals
            ),
            _curve(bottom_points, source_id=f"{name}:bottom", intervals=x_intervals),
            _curve(((x0, y1), (x1, y1)), source_id=f"{name}:top", intervals=x_intervals),
        ),
    )


def _box(
    name: str,
    origin: tuple[float, float, float],
    axes: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    /,
    *,
    intervals: tuple[int, int, int] = (2, 2, 2),
) -> TransfiniteBlock:
    faces = []
    for face in range(6):
        normal, side = divmod(face, 2)
        free = tuple(axis for axis in range(3) if axis != normal)
        start = np.asarray(origin, dtype=np.float64) + side * np.asarray(axes[normal])
        surface = PlanePatch(start, axes[free[0]], axes[free[1]])
        faces.append(
            TransfiniteSurfaceControl(
                surface,
                f"{name}:face:{face}",
                np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
                (intervals[free[0]], intervals[free[1]]),
            )
        )
    return TransfiniteBlock(name, tuple(faces))


def test_logical_cell_bank_uses_c_order_quad_and_hex_reference_vertices() -> None:
    np.testing.assert_array_equal(
        logical_cell_bank((2, 2)),
        np.asarray(((0, 2, 3, 1),), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        logical_cell_bank((2, 2, 2)),
        np.asarray(((0, 4, 6, 2, 1, 5, 7, 3),), dtype=np.int32),
    )


@pytest.mark.parametrize("shape", ((), (2,), (2, 2, 2, 2), (0, 2), (2, 1)))
def test_logical_cell_bank_rejects_invalid_logical_dimensions(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="logical_shape"):
        logical_cell_bank(shape)


def test_logical_cell_bank_rejects_int32_identity_overflow_before_allocation() -> None:
    with pytest.raises(OverflowError, match="int32"):
        logical_cell_bank((2**31 + 1, 2))


def test_opposite_curve_count_mismatch_is_rejected() -> None:
    block = _rectangle("counts", 0.0, 1.0, 0.0, 1.0)
    boundaries = list(block.boundaries)
    boundaries[1] = _curve(
        ((1.0, 0.0), (1.0, 1.0)), source_id="different-count", intervals=3
    )
    with pytest.raises(ValueError, match="opposing curve interval counts"):
        TransfiniteBlock("mismatch", tuple(boundaries))


def test_curved_transfinite_boundary_is_realized_in_the_high_order_map() -> None:
    block = _rectangle("curved", 0.0, 1.0, 0.0, 1.0, curved_bottom=True)
    construction = generate_structured_block(block, geometry_degree=2)
    assert construction.mesh.blocks[0].cell_kind == "quadrilateral"
    elements, routes, coordinates = construction.geometry.resolve(construction.mesh)
    element = elements[0]
    assert isinstance(element, FiniteElementSpec)
    assert element.degree == 2
    assert routes[0].shape == (4, element.local_dof_count)
    grid = np.asarray(coordinates).reshape((5, 5, 2))
    bottom = block.boundaries[2]
    assert isinstance(bottom, TransfiniteCurveControl)
    expected = np.asarray(
        bottom.curve.evaluate(jnp.asarray(np.linspace(0.0, 1.0, 5), dtype=jnp.float64))
    )
    np.testing.assert_allclose(grid[:, 0], expected, rtol=0.0, atol=2e-15)


def test_transfinite_source_charts_must_agree_at_shared_vertices() -> None:
    block = _rectangle("cracked", 0.0, 1.0, 0.0, 1.0)
    boundaries = list(block.boundaries)
    boundaries[1] = _curve(
        ((1.0, 0.1), (1.0, 1.0)), source_id="cracked-right", intervals=2
    )
    with pytest.raises(ValueError, match="boundaries disagree"):
        generate_structured_block(TransfiniteBlock("cracked", tuple(boundaries)))


def test_elliptic_improvement_keeps_every_sampled_boundary_node_fixed() -> None:
    block = _rectangle("elliptic", 0.0, 1.0, 0.0, 1.0, x_intervals=3, y_intervals=2)
    initial = generate_structured_block(block)
    optimized = generate_structured_block(block, optimization_steps=1)
    initial_points = np.asarray(initial.mesh.coordinates).reshape((4, 3, 2))
    optimized_points = np.asarray(optimized.mesh.coordinates).reshape((4, 3, 2))
    boundary = np.zeros((4, 3), dtype=np.bool_)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    np.testing.assert_array_equal(initial_points[boundary], optimized_points[boundary])
    assert optimized.optimization is not None


def test_transfinite_surface_faces_construct_a_positive_structured_hex() -> None:
    block = _box(
        "box",
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    construction = generate_structured_block(block, geometry_degree=2)
    assert construction.mesh.blocks[0].cell_kind == "hexahedron"
    assert construction.mesh.blocks[0].cell_count == 8
    elements, _, _ = construction.geometry.resolve(construction.mesh)
    element = elements[0]
    assert isinstance(element, FiniteElementSpec)
    values, gradients = element.tabulate(
        jnp.asarray(((0.17, 0.31, 0.43),), dtype=jnp.float64)
    )
    np.testing.assert_allclose(np.sum(values, axis=1), 1.0, atol=2e-14)
    np.testing.assert_allclose(np.sum(gradients, axis=1), 0.0, atol=2e-13)


def test_triangle_and_quad_sweeps_preserve_family_and_source_cell_ids() -> None:
    schedule = LayerSchedule((0.25, 0.75))
    control = SweepControl(SweepMapKind.EXTRUSION, schedule, "z-axis")
    quad = phx.discretization.CellMesh(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        (
            phx.discretization.CellBlock(
                "quad", "quadrilateral", ((0, 1, 2, 3),), global_ids=(91,)
            ),
        ),
    )
    triangle = phx.discretization.CellMesh.from_triangles(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),), cell_global_ids=(37,)
    )
    swept_quad = generate_sweep(quad, control)
    swept_triangle = generate_sweep(triangle, control)
    assert swept_quad.mesh.blocks[0].cell_kind == "hexahedron"
    assert swept_triangle.mesh.blocks[0].cell_kind == "prism"
    np.testing.assert_array_equal(swept_quad.cell_parent_ids[0][1], (91, 91))
    np.testing.assert_array_equal(swept_triangle.cell_parent_ids[0][1], (37, 37))
    triangle_map = swept_triangle.reference_map[0]
    assert triangle_map.target_cell_kind == "prism"
    assert triangle_map.source_cell_kind == "triangle"
    np.testing.assert_array_equal(triangle_map.source_cell_ids, (37, 37))
    np.testing.assert_array_equal(
        triangle_map.source_reference_vertices[:, :3],
        triangle_map.source_reference_vertices[:, 3:],
    )
    np.testing.assert_allclose(swept_quad.stations, (0.0, 0.25, 1.0))
    np.testing.assert_allclose(
        swept_quad.measured_layer_thicknesses, ((0.25,) * 4, (0.75,) * 4)
    )


def test_periodic_profile_extrusion_preserves_quotient_and_physical_stations() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (1.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    )
    rows = ((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4))
    lifted = phx.discretization.CellMesh.from_triangles(points, rows)
    generator = np.eye(4)[None]
    generator[0, 0, 3] = 1.0
    group = PeriodicIsometryGroup(generator)
    representatives = np.asarray((0, 0, 2, 2, 4, 4))
    shifts = np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)))
    topology = PeriodicMeshTopology(lifted, group, representatives, shifts)
    profile = phx.discretization.CellMesh(
        points, lifted.blocks, periodic_topology=topology
    )
    swept = generate_sweep(
        profile, SweepControl(SweepMapKind.EXTRUSION, LayerSchedule((0.01, 0.01)), "z")
    )
    actual = swept.mesh.periodic_topology
    assert actual is not None
    assert isinstance(actual.cell, PeriodicIsometryGroup)
    assert actual.cell.group_id == group.group_id
    np.testing.assert_array_equal(
        actual.vertex_representatives,
        np.concatenate([representatives + 6 * station for station in range(3)]),
    )
    np.testing.assert_array_equal(actual.vertex_shifts, np.tile(shifts, (3, 1)))
    np.testing.assert_allclose(swept.stations, (0.0, 0.01, 0.02), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        swept.measured_layer_thicknesses, 0.01, rtol=0.0, atol=1e-15
    )
    assert swept.mesh.blocks[0].cell_count == 8
    boundaries = [item.scipy_boundary().toarray() for item in actual.quotient.incidences]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        np.testing.assert_array_equal(left @ right, 0)


def test_periodic_extrusion_rejects_axis_not_preserved_by_group() -> None:
    points = np.asarray(
        ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0))
    )
    lifted = phx.discretization.CellMesh.from_triangles(points, ((0, 1, 2), (0, 2, 3)))
    generator = np.asarray(
        (
            (
                (0.0, -1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
    )
    group = PeriodicIsometryGroup(generator)
    topology = PeriodicMeshTopology(
        lifted, group, np.asarray((0, 1, 1, 0)), np.asarray(((0,), (0,), (1,), (1,)))
    )
    profile = phx.discretization.CellMesh(
        points, lifted.blocks, periodic_topology=topology
    )
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        LayerSchedule((0.01,)),
        "x",
        axis=np.asarray((1.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="does not preserve the extrusion direction"):
        generate_sweep(profile, control)


def test_closed_revolution_has_an_explicit_periodic_seam_map() -> None:
    profile = phx.discretization.CellMesh.from_triangles(
        ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
        ((0, 1, 2),),
    )
    control = SweepControl(
        SweepMapKind.REVOLUTION,
        LayerSchedule((np.pi / 2.0,) * 4),
        "full-turn",
        closed=True,
    )
    swept = generate_sweep(profile, control)
    assert swept.mesh.blocks[0].cell_kind == "prism"
    np.testing.assert_array_equal(swept.layer_vertex_ids[0], swept.layer_vertex_ids[-1])
    np.testing.assert_array_equal(
        swept.mesh.blocks[0].vertices[-1, 3:], swept.layer_vertex_ids[0]
    )


def test_column_connectivity_closes_only_the_declared_last_station() -> None:
    profile = np.asarray(((0, 2, 3, 1),), dtype=np.int32)
    columns = column_connectivity(profile, 3, 4, True)
    np.testing.assert_array_equal(columns[-1, 4:], profile[0])
    assert int(np.max(columns)) == 11


def test_two_dimensional_blocks_glue_only_declared_oriented_edges() -> None:
    left = generate_structured_block(_rectangle("left", 0.0, 0.5, 0.0, 1.0))
    right = generate_structured_block(_rectangle("right", 0.5, 1.0, 0.0, 1.0))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    glued = glue_structured_blocks((left, right), (interface,))
    assert glued.mesh.coordinates.shape[0] == 15
    assert glued.maximum_gluing_residual == 0.0
    left_map = dict(glued.block_vertex_ids)["left"]
    right_map = dict(glued.block_vertex_ids)["right"]
    np.testing.assert_array_equal(left_map[[6, 7, 8]], right_map[[0, 1, 2]])
    assert dict(glued.block_cell_parent_ids)["left"].shape == (
        left.mesh.blocks[0].cell_count,
    )


def test_coincident_blocks_are_not_welded_without_an_interface() -> None:
    first = generate_structured_block(_rectangle("first", 0.0, 1.0, 0.0, 1.0))
    second = generate_structured_block(_rectangle("second", 0.0, 1.0, 0.0, 1.0))
    construction = glue_structured_blocks((first, second), ())
    assert construction.mesh.coordinates.shape[0] == 18


def test_wrong_interface_permutation_and_nonconforming_counts_are_rejected() -> None:
    left = generate_structured_block(_rectangle("left", 0.0, 0.5, 0.0, 1.0))
    right = generate_structured_block(_rectangle("right", 0.5, 1.0, 0.0, 1.0))
    wrong = BlockInterfaceControl("left", 1, "right", 0, (0,), (True,))
    with pytest.raises(ValueError, match="residual"):
        glue_structured_blocks((left, right), (wrong,))
    nonconforming_right = generate_structured_block(
        _rectangle("other", 0.5, 1.0, 0.0, 1.0, y_intervals=3)
    )
    mismatch = BlockInterfaceControl("left", 1, "other", 0, (0,), (False,))
    with pytest.raises(ValueError, match="counts differ"):
        glue_structured_blocks((left, nonconforming_right), (mismatch,))


def test_three_dimensional_face_permutation_and_flip_are_applied_explicitly() -> None:
    left = generate_structured_block(
        _box(
            "left3d", (0.0, 0.0, 0.0), ((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    )
    right = generate_structured_block(
        _box(
            "right3d",
            (0.5, 0.0, 1.0),
            ((0.5, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        )
    )
    interface = BlockInterfaceControl("left3d", 1, "right3d", 0, (1, 0), (False, True))
    glued = glue_structured_blocks((left, right), (interface,))
    assert glued.mesh.blocks[0].cell_kind == "hexahedron"
    assert glued.maximum_gluing_residual == 0.0


def test_sweep_connectivity_carries_high_order_geometry_rows_and_refuses_wraparound() -> (
    None
):
    profile_geometry_rows = np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32)
    swept = column_connectivity(profile_geometry_rows, 2, 6, False)
    np.testing.assert_array_equal(swept[0], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
    with pytest.raises(OverflowError, match="int32"):
        column_connectivity(np.asarray(((0, 2**31, 2),), dtype=np.int64), 1, 3, False)


def test_closed_frame_sweep_retains_the_declared_periodic_seam() -> None:
    profile = phx.discretization.CellMesh.from_triangles(
        ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
        ((0, 1, 2),),
    )
    identity = np.eye(3, dtype=np.float64)
    quarter_turn = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    half_turn = np.asarray(
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    three_quarter_turn = np.asarray(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    frames = np.stack((identity, quarter_turn, half_turn, three_quarter_turn, identity))
    control = SweepControl(
        SweepMapKind.FRAMES,
        LayerSchedule((0.25, 0.25, 0.25, 0.25)),
        "closed-frame-cycle",
        frames=frames,
        translations=np.zeros((5, 3), dtype=np.float64),
        closed=True,
    )
    swept = generate_sweep(profile, control)
    assert swept.mesh.blocks[0].cell_kind == "prism"
    np.testing.assert_array_equal(swept.layer_vertex_ids[0], swept.layer_vertex_ids[-1])
    np.testing.assert_array_equal(
        swept.mesh.blocks[0].vertices[-1, 3:], swept.layer_vertex_ids[0]
    )


def _native_square_request(
    required_families: tuple[str, ...],
    /,
    *,
    outside_source_bank: bool = False,
) -> tuple[
    NativeStructuredSource,
    SurfaceMeshingSpec,
    NativeMeshingOptions,
    SpatialCoordinateContract,
]:
    source_id = "square-source"
    domain = PiecewiseLinearDomain(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("square",),
        source_id=source_id,
    )
    revision = domain.source_revision
    block = _rectangle("square-block", 0.0, 1.0, 0.0, 1.0)
    curves = cast(tuple[TransfiniteCurveControl, ...], block.boundaries)
    boundary = ParametricCurveBoundarySource(
        tuple(control.curve for control in curves),
        tuple(control.parameter_range for control in curves),
        source_id=source_id,
        source_revision=revision,
        curve_entity_ids=tuple(control.source_id for control in block.boundaries),
        occurrence_paths=((),) * len(block.boundaries),
        chart_ids=tuple(control.source_id for control in block.boundaries),
    )
    source = NativeStructuredSource(
        (block,),
        (),
        source_id,
        revision,
        fidelity_source=boundary,
        maximum_deviation=1e-12,
        domain=domain,
    )
    scope = MeshingScope.whole(source, 2)
    if outside_source_bank:
        bank_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
        bank_id_set = {int(identifier) for identifier in bank_ids}
        invalid_id = 0
        while invalid_id in bank_id_set:
            invalid_id += 1
        scope = MeshingScope(
            source_id,
            revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            np.asarray((invalid_id,), dtype=np.int64),
        )
    target = CellMeshingTarget(
        2,
        2,
        CellFamilyPolicy(required=required_families),
        geometry_order=1,
    )
    specification = SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(),
        quality_target=MeshQualityTarget(minimum_angle=0.2),
        planar_embedding=PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    options = NativeMeshingOptions(
        "structured_transfinite",
        structured_schedule=NativeStructuredSchedule(),
    )
    return source, specification, options, SpatialCoordinateContract.si()


def test_structured_source_archive_replays_and_rejects_changed_identity(
    tmp_path: Path,
) -> None:
    import copy

    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    source, _, _, _ = _native_square_request(("quadrilateral",))
    path = tmp_path / "structured.phx"
    write_model_artifact(path, source, limits=None)
    restored = read_model_artifact(
        path,
        expected_type=NativeStructuredSource,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored.binding_id == source.binding_id
    assert restored.domain is not None and source.domain is not None
    assert restored.domain.domain_id == source.domain.domain_id
    for actual, expected in zip(
        restored.scope_entity_id_banks, source.scope_entity_id_banks, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    altered = copy.copy(source)
    object.__setattr__(altered, "binding_id", "tampered")
    write_model_artifact(path, altered, limits=None)
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            path,
            expected_type=NativeStructuredSource,
            restore=restore_model_archive_value,
            limits=None,
        )


def test_structured_solid_controls_archive_replay(tmp_path: Path) -> None:
    import copy

    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    block = _box(
        "solid",
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    interface = BlockInterfaceControl("solid", 1, "other", 0, (1, 0), (False, True))
    for value, identity in ((block, "block_id"), (interface, "control_id")):
        path = tmp_path / f"{identity}.phx"
        write_model_artifact(path, value, limits=None)
        restored = read_model_artifact(
            path,
            expected_type=type(value),
            restore=restore_model_archive_value,
            limits=None,
        )
        assert getattr(restored, identity) == getattr(value, identity)
        altered = copy.copy(value)
        object.__setattr__(altered, identity, "tampered")
        write_model_artifact(path, altered, limits=None)
        with pytest.raises(ArrayArchiveCorruptionError):
            read_model_artifact(
                path,
                expected_type=type(value),
                restore=restore_model_archive_value,
                limits=None,
            )


def test_sweep_source_archive_replays_and_rejects_changed_scope(tmp_path: Path) -> None:
    import copy

    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from tools.meshing_qualification.scenarios.native_family_matrix import _make_profile

    source = _make_profile("native-family-lifecycle:sweep-hex").source
    path = tmp_path / "sweep.phx"
    write_model_artifact(path, source, limits=None)
    restored = read_model_artifact(
        path,
        expected_type=NativeSweepSource,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored.binding_id == source.binding_id
    assert (
        restored.fidelity_source.source_query_id == source.fidelity_source.source_query_id
    )
    altered = copy.copy(source)
    object.__setattr__(altered, "scope_entity_set_ids", ("tampered",) * 4)
    write_model_artifact(path, altered, limits=None)
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            path,
            expected_type=NativeSweepSource,
            restore=restore_model_archive_value,
            limits=None,
        )
    owner = copy.copy(source.fidelity_source.owner)
    object.__setattr__(owner, "_scope_entity_set_ids", ("tampered",) * 4)
    write_model_artifact(path, owner, limits=None)
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            path,
            expected_type=type(owner),
            restore=restore_model_archive_value,
            limits=None,
        )


@pytest.mark.parametrize(
    "profile",
    (
        "transfinite-quad",
        "multiblock-quad",
        "sweep-hex",
        "curved-frame-sweep-hex",
        "multiblock-reversed-hex",
        "multiblock-material-hex",
    ),
)
def test_structured_family_cold_fe_continuation(profile: str) -> None:
    from tools.meshing_qualification.scenarios.native_family_cold import (
        native_family_cold,
    )

    evidence = native_family_cold(f"native-family-lifecycle:{profile}")
    producer, consumer = evidence["producer"], evidence["consumer"]
    assert isinstance(producer, dict) and isinstance(consumer, dict)
    assert producer["pid"] != consumer["pid"]
    assert producer["accepted_id"] == consumer["accepted_id"]
    assert consumer["next_accepted_id"] != consumer["accepted_id"]


def test_native_structured_provider_returns_canonical_construction_data() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("quadrilateral",)
    )
    prepared = prepare_native_structured(
        source, specification, options, coordinate_contract
    )
    construction = prepared.execute()
    assert construction.mesh.blocks[0].cell_kind == "quadrilateral"
    elements, _, _ = construction.geometry.resolve(construction.mesh)
    assert isinstance(elements[0], FiniteElementSpec)
    assert elements[0].degree == 1
    assert construction.compliance.passed
    assert dict(construction.compliance.requested)["minimum_angle"] == 0.2
    assert dict(construction.compliance.achieved)["minimum_angle"] == pytest.approx(
        math.pi / 2.0
    )
    assert construction.certification.schedule.route == "planar"
    assert construction.certification.domain is source.domain
    np.testing.assert_array_equal(
        construction.certification.cell_regions,
        np.zeros(construction.mesh.connectivity.cell_count, dtype=np.int64),
    )
    assert construction.certification.fidelity_tolerance == 1e-12
    assert any(
        label.name.startswith("structured-boundary:")
        for label in construction.organization.labels
    )


def test_embedded_structured_surface_preserves_its_surface_query() -> None:
    domain = PiecewiseLinearDomain(
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))),
        np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("solid",),
        source_id="embedded-structured-source",
    )
    query = PiecewiseLinearBoundarySource(
        PiecewiseLinearDomainBoundaryOwner(domain),
        2,
        entity_ids=np.asarray((0,), dtype=np.int64),
    )
    edges = (
        ((0.0, 0.0, 0.0), (0.0, 0.5, 0.0)),
        ((0.5, 0.0, 0.0), (0.5, 0.5, 0.0)),
        ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ((0.0, 0.5, 0.0), (0.5, 0.5, 0.0)),
    )
    block = TransfiniteBlock(
        "embedded",
        tuple(
            _curve(points, source_id=f"embedded:edge:{row}", intervals=2)
            for row, points in enumerate(edges)
        ),
    )
    source = NativeStructuredSource(
        (block,),
        (),
        domain.source_id,
        domain.source_revision,
        fidelity_source=query,
        maximum_deviation=1e-12,
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("quadrilateral",))),
        MeshingScope.whole(source, 2),
        size_controls=(),
        quality_target=MeshQualityTarget(minimum_angle=0.2),
    )
    construction = prepare_native_structured(
        source,
        specification,
        NativeMeshingOptions(
            "structured_transfinite", structured_schedule=NativeStructuredSchedule()
        ),
        SpatialCoordinateContract.si(),
    ).execute()
    assert construction.mesh.ambient_dimension == 3
    assert construction.mesh.blocks[0].cell_kind == "quadrilateral"
    assert construction.compliance.passed
    assert construction.certification.schedule.route == "surface"
    assert construction.certification.fidelity_source is query
    assert construction.certification.fidelity_witness is None
    np.testing.assert_array_equal(np.asarray(construction.mesh.coordinates)[:, 2], 0.0)


@pytest.mark.parametrize("hard", (True, False))
def test_structured_surface_quality_records_unmet_physical_corner_angle(
    hard: bool,
) -> None:
    _, baseline, _, _ = _native_square_request(("quadrilateral",))
    specification = SurfaceMeshingSpec(
        baseline.target,
        baseline.scope,
        size_controls=(),
        quality_target=MeshQualityTarget(minimum_angle=0.2, hard=hard),
    )
    block = TransfiniteBlock(
        "skew",
        tuple(
            _curve(points, source_id=f"skew:{side}", intervals=1)
            for side, points in enumerate(
                (
                    ((0.0, 0.0), (1.0, 0.1)),
                    ((1.0, 0.0), (2.0, 0.1)),
                    ((0.0, 0.0), (1.0, 0.0)),
                    ((1.0, 0.1), (2.0, 0.1)),
                )
            )
        ),
    )
    construction = generate_structured_block(block)
    compliance = _size_compliance(construction.mesh, construction.geometry, specification)

    assert compliance.passed is not hard
    assert ("minimum_angle" in compliance.issues) is hard
    assert dict(compliance.requested)["minimum_angle"] == 0.2
    assert dict(compliance.achieved)["minimum_angle"] == pytest.approx(math.atan(0.1))


def test_structured_surface_quality_uses_curved_edge_tangents_not_chords() -> None:
    _, specification, _, _ = _native_square_request(("quadrilateral",))
    rectangle = _rectangle(
        "curved-quality", 0.0, 1.0, 0.0, 1.0, x_intervals=1, y_intervals=1
    )
    boundaries = list(rectangle.boundaries)
    boundaries[2] = _curve(
        ((0.0, 0.0), (0.5, 0.2), (1.0, 0.0)),
        source_id="curved-quality:bottom",
        intervals=1,
    )
    construction = generate_structured_block(
        TransfiniteBlock(rectangle.name, tuple(boundaries)), geometry_degree=2
    )
    compliance = _size_compliance(construction.mesh, construction.geometry, specification)

    assert compliance.passed
    assert dict(compliance.achieved)["minimum_angle"] == pytest.approx(
        math.atan2(1.0, 0.4)
    )


def test_native_sweep_provider_preserves_domain_and_profile_field_ancestry() -> None:
    source_id = "sweep-prism"
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    facets = np.asarray(
        (
            (0, 2, 1),
            (3, 4, 5),
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (2, 0, 3),
            (2, 3, 5),
        ),
        dtype=np.int64,
    )
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(
            np.asarray(((0, -1),), dtype=np.int64),
            (facets.shape[0], 1),
        ),
        ("solid",),
        source_id=source_id,
    )
    profile = phx.discretization.CellMesh.from_triangles(
        vertices[:3],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        cell_global_ids=(37,),
    )
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        LayerSchedule((1.0,)),
        "unit-prism-extrusion",
    )
    source = NativeSweepSource(
        profile,
        control,
        source_id,
        domain.source_revision,
        fidelity_source=PiecewiseLinearBoundarySource(
            PiecewiseLinearDomainBoundaryOwner(domain), 2
        ),
        maximum_deviation=0.0,
        domain=domain,
        domain_vertex_map=np.asarray(
            ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
            dtype=np.int64,
        ),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("prism",)),
            geometry_order=1,
        ),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.SWEEP,
        size_controls=(),
    )

    unmapped_source = NativeSweepSource(
        profile,
        control,
        source_id,
        domain.source_revision,
        fidelity_source=source.fidelity_source,
        maximum_deviation=0.0,
        domain=domain,
    )
    unmapped_specification = VolumeMeshingSpec(
        specification.target,
        MeshingScope.whole(unmapped_source, 2),
        specification.fill_strategy,
        size_controls=(),
    )
    with pytest.raises(MeshingFailure, match="explicit source-domain vertex"):
        NativeMeshingProvider(NativeMeshingOptions("sweep")).plan(
            unmapped_source,
            unmapped_specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )

    result = (
        NativeMeshingProvider(NativeMeshingOptions("sweep"))
        .plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute()
    )

    assert result.mesh.blocks[0].cell_kind == "prism"
    assert result.certification is not None and result.certification.passed
    fidelity = result.certification.fidelity
    assert fidelity is not None
    assert fidelity.mesh_to_source_upper == 0.0
    assert fidelity.source_to_mesh_upper == 0.0
    assert result.audit.passed
    assert result.compliance.passed
    assert result.certification.inputs.domain is domain
    assert result.certification.inputs.source is source.fidelity_source
    np.testing.assert_array_equal(
        result.mesh.coordinates,
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
            ),
            dtype=np.float64,
        ),
    )
    attributes = {attribute.name: attribute for attribute in result.attributes}
    np.testing.assert_array_equal(
        attributes[f"{result.mesh.blocks[0].name}:source-profile-cell-id"].values,
        (37,),
    )
    assert {zone.name for zone in result.zones} == {"solid"}

    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    cell_volumes = np.asarray(finite_volume.cell_volumes, dtype=np.float64)
    np.testing.assert_allclose(cell_volumes, (0.5,), rtol=0.0, atol=1.0e-12)
    source_field = {37: 2.0}
    parent_ids = np.asarray(
        attributes[f"{result.mesh.blocks[0].name}:source-profile-cell-id"].values,
        dtype=np.int64,
    )
    transported_field = np.asarray(
        [source_field[int(parent_id)] for parent_id in parent_ids],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.dot(transported_field, cell_volumes),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_native_graded_whole_slab_sweep_keeps_the_physical_layer_schedule() -> None:
    # Native counterpart of the CAD whole-solid layer sweep: the full slab is the
    # layer domain, with no core fill and an exact first-cell height.
    source_id = "graded-slab"
    thicknesses = (0.08, 0.10, 0.12)
    stations = np.concatenate(((0.0,), np.cumsum(thicknesses)))
    triangle = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    # The PWL boundary carries every station so each wall facet is one swept face.
    vertices = np.asarray(
        [(x, y, z) for z in stations for x, y in triangle], dtype=np.float64
    )
    top = 3 * (len(thicknesses))
    facet_rows = [(0, 2, 1), (top, top + 1, top + 2)]
    for layer in range(len(thicknesses)):
        base = 3 * layer
        for first, second in ((0, 1), (1, 2), (2, 0)):
            facet_rows.append((base + first, base + second, base + second + 3))
            facet_rows.append((base + first, base + second + 3, base + first + 3))
    facets = np.asarray(facet_rows, dtype=np.int64)
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (facets.shape[0], 1)),
        ("slab",),
        source_id=source_id,
    )
    owner = PiecewiseLinearDomainBoundaryOwner(domain)
    profile = phx.discretization.CellMesh.from_triangles(
        vertices[:3],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        cell_global_ids=(5,),
    )
    source = NativeSweepSource(
        profile,
        SweepControl(
            SweepMapKind.EXTRUSION,
            LayerSchedule(thicknesses),
            "graded-slab-extrusion",
        ),
        source_id,
        domain.source_revision,
        fidelity_source=PiecewiseLinearBoundarySource(owner, 2),
        maximum_deviation=0.0,
        domain=domain,
        domain_vertex_map=np.asarray(
            [
                (profile_row, station)
                for station in range(stations.size)
                for profile_row in range(3)
            ],
            dtype=np.int64,
        ),
    )
    assert isinstance(source.fidelity_source, PiecewiseLinearBoundarySource)
    assert source.fidelity_source.owner is owner
    wall_scope = MeshingScope(
        source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        owner.scope_entity_set_id(2),
        np.asarray((0,), dtype=np.int64),
    )
    cap_scope = MeshingScope(
        source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        owner.scope_entity_set_id(2),
        np.asarray((1,), dtype=np.int64),
    )
    volume_scope = MeshingScope(
        source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        owner.scope_entity_set_id(3),
        np.asarray((0,), dtype=np.int64),
    )
    control = phx.meshing.BoundaryLayerControl(
        wall_scope,
        source.control.schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        cap_scope=cap_scope,
        volume_scope=volume_scope,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("prism",))),
        MeshingScope.whole(owner, 2),
        VolumeFillStrategy.SWEEP,
        size_controls=(),
        layer_controls=(control,),
    )
    assert not sweep_support_issues(source, specification)
    wrong_schedule = phx.meshing.BoundaryLayerControl(
        wall_scope,
        LayerSchedule((0.08, 0.10, 0.11)),
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        cap_scope=cap_scope,
        volume_scope=volume_scope,
    )
    bad_specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("prism",))),
        MeshingScope.whole(owner, 2),
        VolumeFillStrategy.SWEEP,
        size_controls=(),
        layer_controls=(wrong_schedule,),
    )
    assert sweep_support_issues(source, bad_specification)
    side_scope = MeshingScope(
        source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        owner.scope_entity_set_id(2),
        np.asarray((2,), dtype=np.int64),
    )
    wrong_cap = phx.meshing.BoundaryLayerControl(
        wall_scope,
        source.control.schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        cap_scope=side_scope,
        volume_scope=volume_scope,
    )
    wrong_cap_specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("prism",))),
        MeshingScope.whole(owner, 2),
        VolumeFillStrategy.SWEEP,
        size_controls=(),
        layer_controls=(wrong_cap,),
    )
    assert sweep_support_issues(source, wrong_cap_specification)

    result = (
        NativeMeshingProvider(NativeMeshingOptions("sweep"))
        .plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute()
    )

    assert {block.cell_kind for block in result.mesh.blocks} == {"prism"}
    assert sum(block.cell_count for block in result.mesh.blocks) == len(thicknesses)
    assert result.certification is not None and result.certification.passed
    assert result.audit.passed and result.compliance.passed
    stations = np.unique(np.asarray(result.mesh.coordinates, dtype=np.float64)[:, 2])
    np.testing.assert_allclose(
        stations, np.concatenate(((0.0,), np.cumsum(thicknesses))), rtol=0.0, atol=1e-15
    )
    assert stations[1] - stations[0] == pytest.approx(thicknesses[0], rel=0.0, abs=1e-15)
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    cell_volumes = np.sort(np.asarray(finite_volume.cell_volumes, dtype=np.float64))
    # Independent oracle: each layer is a right prism over the unit right triangle.
    np.testing.assert_allclose(
        cell_volumes, np.sort(0.5 * np.asarray(thicknesses)), rtol=0.0, atol=1e-14
    )


def test_native_structured_provider_refuses_a_required_nonquad_target() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("triangle",)
    )
    with pytest.raises(MeshingFailure, match="required_cell_families"):
        prepare_native_structured(source, specification, options, coordinate_contract)


def test_native_structured_provider_refuses_scope_ids_outside_the_source_bank() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("quadrilateral",), outside_source_bank=True
    )
    with pytest.raises(MeshingFailure, match="source-owned entity bank"):
        prepare_native_structured(source, specification, options, coordinate_contract)


def test_native_structured_planar_provider_requires_independent_coverage_domain() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("quadrilateral",)
    )
    unbound = NativeStructuredSource(
        source.blocks,
        source.interfaces,
        source.source_id,
        source.source_revision,
        fidelity_source=source.fidelity_source,
        maximum_deviation=source.maximum_deviation,
    )
    with pytest.raises(MeshingFailure, match="authoritative planar source domain"):
        prepare_native_structured(unbound, specification, options, coordinate_contract)


@pytest.mark.parametrize(
    "profile", ("sweep-prism", "multiblock-reversed-hex", "multiblock-material-hex")
)
def test_split_face_associations_reject_forged_scope_side_and_coverage(
    profile: str,
) -> None:
    from dataclasses import fields

    from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
    from phydrax.meshing._audit import _validated_partial_face_coverage
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _CONTRACT,
        _make_profile,
    )

    case = _make_profile(f"native-family-lifecycle:{profile}")
    result = (
        NativeMeshingProvider(case.options)
        .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    assert result.certification is not None
    inputs = result.certification.inputs
    assert inputs.fidelity_witness is not None
    token = validate_source_fidelity_witness(
        result.mesh, result.geometry, case.source.fidelity_source, inputs.fidelity_witness
    )

    def audit(associations: tuple, source: object = case.source) -> tuple:
        return _validated_partial_face_coverage(
            result.mesh,
            result.geometry,
            associations,
            (token,),
            source,
            None,
            inputs,
            None,
            None,
        )

    covered, issues, _wall_ids = audit(result.associations)
    assert covered and not issues
    assert "association_source_witness" in audit(result.associations, object())[1]
    original = result.associations[0]
    original_fields = {
        field.name: getattr(original, field.name) for field in fields(original)
    }
    for name, replacement in (
        ("source_occurrence_paths", (("forged",), *original.source_occurrence_paths[1:])),
        (
            "source_region_sides",
            (
                tuple(reversed(original.source_region_sides[0])),
                *original.source_region_sides[1:],
            ),
        ),
        ("target_reference_pieces", (None, *original.target_reference_pieces[1:])),
    ):
        values = dict(original_fields)
        values[name] = replacement
        forged = object.__new__(type(original))
        for field_name, value in values.items():
            object.__setattr__(forged, field_name, value)
        associations = (forged, *result.associations[1:])
        issues = audit(associations)[1]
        assert "association_source_witness" in issues
        assert any(
            detail.startswith("association_source_witness_detail:")
            and "ValueError:" in detail
            and '"target_scopes":' in detail
            for detail in issues
        )
        from phydrax.meshing._certification import _audit_findings

        forged_audit = object.__new__(type(result.audit))
        for field in fields(result.audit):
            object.__setattr__(
                forged_audit,
                field.name,
                issues if field.name == "issues" else getattr(result.audit, field.name),
            )
        findings = _audit_findings(inputs, forged_audit, None)
        assert any(
            finding.check == "association_source_witness"
            and finding.entity_ids
            and "ValueError:" in finding.detail
            and "target_scopes=" in finding.detail
            for finding in findings
        )
    if profile == "multiblock-material-hex":
        from phydrax.meshing._controls import RegionSeed
        from phydrax.meshing.providers._native_structured import (
            _validate_structured_region_seeds,
        )

        assert isinstance(case.specification, VolumeMeshingSpec)
        original_seed = case.specification.region_seeds[0]
        for point in ((1.5, 0.5, 0.5), (1.0, 0.5, 0.5)):
            altered = object.__new__(type(case.specification))
            for field in fields(case.specification):
                value = getattr(case.specification, field.name)
                if field.name == "region_seeds":
                    value = (
                        RegionSeed(
                            np.asarray(point),
                            original_seed.region_name,
                            original_seed.material_id,
                            original_seed.role,
                        ),
                    )
                object.__setattr__(altered, field.name, value)
            with pytest.raises(ValueError, match="source region|source boundary"):
                _validate_structured_region_seeds(
                    case.source, altered, result.mesh, result.geometry
                )


def test_annulus_source_anchor_refuses_ambiguity_occurrence_and_side_forgery() -> None:
    from dataclasses import fields, replace

    from phydrax.meshing.providers._native_structured import (
        _anchor_proved_planar_boundary,
        _structured_curve_witness,
        PreparedStructured,
    )
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _CONTRACT,
        _make_profile,
    )

    case = _make_profile("native-family-lifecycle:multiblock-cycle-quad")
    assert isinstance(case.specification, SurfaceMeshingSpec)
    assert case.source.domain is not None
    prepared = (
        NativeMeshingProvider(case.options)
        .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
        .prepared
    )
    assert isinstance(prepared, PreparedStructured)
    constructions = tuple(
        generate_structured_block(
            block,
            geometry_degree=case.specification.target.geometry_order,
            optimization_steps=prepared.schedule.optimization_steps,
        )
        for block in case.source.blocks
    )
    combined = glue_structured_blocks(constructions, case.source.interfaces)
    witness = _structured_curve_witness(case.source, constructions, combined)
    assert witness is not None
    original_domain_vertices = np.array(case.source.domain.vertices)
    anchored = _anchor_proved_planar_boundary(
        case.source, case.specification, combined, witness
    )
    assert anchored is not combined
    np.testing.assert_array_equal(case.source.domain.vertices, original_domain_vertices)
    assert all(
        any(
            np.array_equal(point, target)
            for target in np.asarray(anchored.mesh.coordinates)
        )
        for point in original_domain_vertices
    )
    offset = np.asarray((1e-8, 2e-8))
    shifted_mesh = combined.mesh.with_coordinates(
        np.asarray(combined.mesh.coordinates) + offset, numeric_version="unanchored"
    )
    shifted_geometry = phx.discretization.CellGeometrySpec(
        dict(zip(combined.geometry.block_names, combined.geometry.elements, strict=True)),
        dict(
            zip(
                combined.geometry.block_names,
                combined.geometry.geometry_dofs,
                strict=True,
            )
        ),
        np.asarray(combined.geometry.coordinates) + offset,
    )
    shifted = replace(combined, mesh=shifted_mesh, geometry=shifted_geometry)
    shifted_witness = _structured_curve_witness(case.source, constructions, shifted)
    assert shifted_witness is not None
    with pytest.raises(ValueError, match="exact common endpoint anchor"):
        _anchor_proved_planar_boundary(
            case.source, case.specification, shifted, shifted_witness
        )
    ambiguous = NativeStructuredSource(
        case.source.blocks,
        case.source.interfaces,
        case.source.source_id,
        case.source.source_revision,
        fidelity_source=case.source.fidelity_source,
        maximum_deviation=2.0,
        domain=case.source.domain,
        block_regions=case.source.block_regions,
    )
    with pytest.raises(ValueError, match="unique certified"):
        _anchor_proved_planar_boundary(ambiguous, case.specification, combined, witness)
    for field_name in ("region_occurrence_paths", "facet_regions"):
        domain = object.__new__(type(case.source.domain))
        for field in fields(case.source.domain):
            value = getattr(case.source.domain, field.name)
            if field.name == field_name == "region_occurrence_paths":
                value = (("forged",),)
            elif field.name == field_name == "facet_regions":
                value = np.array(value)
                value[0] = value[0, ::-1]
            object.__setattr__(domain, field.name, value)
        forged_source = object.__new__(type(case.source))
        for field in fields(case.source):
            object.__setattr__(
                forged_source,
                field.name,
                domain if field.name == "domain" else getattr(case.source, field.name),
            )
        with pytest.raises(ValueError, match="occurrence|oriented PWL"):
            _anchor_proved_planar_boundary(
                forged_source, case.specification, combined, witness
            )


def test_structured_material_interface_patch_control_publishes_and_refuses_partial() -> (
    None
):
    from phydrax.meshing._controls import PatchControl
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _CONTRACT,
        _make_profile,
        _volume_specification,
    )

    case = _make_profile("native-family-lifecycle:multiblock-material-hex")
    assert isinstance(case.specification, VolumeMeshingSpec)
    (control,) = case.specification.patch_controls
    result = (
        NativeMeshingProvider(case.options)
        .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    patches = {patch.name: patch for patch in result.patches}
    interface = patches["material-interface"]
    zones = {zone.name: zone for zone in result.zones}
    assert interface.adjacent_zone_ids == tuple(
        sorted((zones["left"].zone_id, zones["right"].zone_id))
    )
    assert interface.connected
    # resolution r: the reversed blocks meet on one r x r grid of hexahedron faces.
    assert np.asarray(interface.scope.entity_ids).size == 4 * 4
    left = set(np.asarray(zones["left"].scope.entity_ids).tolist())
    right = set(np.asarray(zones["right"].scope.entity_ids).tolist())
    assert left and right and not left & right

    scope = control.scope
    partial = PatchControl(
        control.name,
        MeshingScope(
            scope.source_id,
            scope.source_revision,
            scope.entity_kind,
            2,
            scope.entity_set_id,
            np.asarray(scope.entity_ids)[:1],
        ),
        control.adjacent_region_names,
    )
    mislabeled = PatchControl(control.name, scope, ("left",))
    for forged in (partial, mislabeled):
        specification = _volume_specification(
            case.source,
            4,
            "hexahedron",
            region_controls=case.specification.region_controls,
            patch_controls=(forged,),
        )
        with pytest.raises(MeshingFailure, match="complete declared source material"):
            NativeMeshingProvider(case.options).plan(
                case.source, specification, coordinate_contract=_CONTRACT
            )
