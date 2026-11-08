from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate
from phydrax.meshing._hex_frame_map import (
    generate_frame_hexes,
    prepare_frame_hex_grid,
    prepare_source_hex_frame,
)
from phydrax.meshing._hex_frame_records import FrameFieldRefusal, HexFrameIntegerLayout
from phydrax.meshing._hex_schedule import NativeHexGridSchedule
from phydrax.meshing._plc_source import PiecewiseLinearComplex


def _cube_plc_and_fill() -> tuple[PiecewiseLinearComplex, CellMesh]:
    vertices = np.asarray(
        tuple(
            (float(x), float(y), float(z))
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ),
        dtype=np.float64,
    )
    row = {(int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(vertices)}
    polygons = (
        (row[(0, 0, 0)], row[(0, 1, 0)], row[(1, 1, 0)], row[(1, 0, 0)]),
        (row[(0, 0, 1)], row[(1, 0, 1)], row[(1, 1, 1)], row[(0, 1, 1)]),
        (row[(0, 0, 0)], row[(0, 0, 1)], row[(0, 1, 1)], row[(0, 1, 0)]),
        (row[(1, 0, 0)], row[(1, 1, 0)], row[(1, 1, 1)], row[(1, 0, 1)]),
        (row[(0, 0, 0)], row[(1, 0, 0)], row[(1, 0, 1)], row[(0, 0, 1)]),
        (row[(0, 1, 0)], row[(0, 1, 1)], row[(1, 1, 1)], row[(1, 1, 0)]),
    )
    segments = tuple(
        tuple(sorted((row[first], row[second])))
        for first, second in (
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 1)),
            ((1, 1, 1), (0, 1, 1)),
            ((1, 1, 1), (1, 0, 1)),
            ((1, 1, 1), (1, 1, 0)),
            ((1, 0, 0), (1, 1, 0)),
            ((1, 0, 0), (1, 0, 1)),
            ((0, 1, 0), (1, 1, 0)),
            ((0, 1, 0), (0, 1, 1)),
            ((0, 0, 1), (1, 0, 1)),
            ((0, 0, 1), (0, 1, 1)),
        )
    )
    source = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(face, dtype=np.int64) for face in polygons),
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(polygons), 1)),
        ("solid",),
        segments=np.asarray(segments, dtype=np.int64),
        vertex_ids=tuple(f"v{i}" for i in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"f{i}" for i in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=tuple(f"e{i}" for i in range(len(segments))),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
    )
    tetrahedra = []
    for order in permutations(range(3)):
        point = [0, 0, 0]
        tet = [row[(point[0], point[1], point[2])]]
        for axis in order:
            point[axis] += 1
            tet.append(row[(point[0], point[1], point[2])])
        local = vertices[np.asarray(tet)]
        determinant = np.linalg.det(
            np.stack(
                (local[1] - local[0], local[2] - local[0], local[3] - local[0]), axis=-1
            )
        )
        if determinant < 0.0:
            tet[1], tet[2] = tet[2], tet[1]
        tetrahedra.append(tet)
    fill = CellMesh.from_tetrahedra(vertices, np.asarray(tetrahedra, dtype=np.int64))
    return source, fill


def _two_disconnected_tetrahedra() -> tuple[PiecewiseLinearComplex, CellMesh, np.ndarray]:
    base = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    angle = 0.41
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    stretch = np.diag((1.2, 0.8, 1.1))
    second = (base - np.mean(base, axis=0)) @ stretch @ rotation.T
    second += np.mean(base, axis=0) + np.asarray((3.0, 0.25, 0.1))
    vertices = np.concatenate((base, second), axis=0)
    facet_loops = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    polygons = tuple(
        np.asarray(loop, dtype=np.int64)
        for loop in (
            *facet_loops,
            *(tuple(value + 4 for value in loop) for loop in facet_loops),
        )
    )
    source = PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(len(polygons), dtype=np.int64),
        np.concatenate(
            (
                np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
                np.tile(np.asarray((-1, 1), dtype=np.int64), (4, 1)),
            ),
            axis=0,
        ),
        ("material-left", "material-right"),
        segments=np.zeros((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"left-v{row}" for row in range(4))
        + tuple(f"right-v{row}" for row in range(4)),
        vertex_occurrence_paths=((),) * 8,
        facet_ids=tuple(f"left-f{row}" for row in range(4))
        + tuple(f"right-f{row}" for row in range(4)),
        facet_occurrence_paths=((),) * 8,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
    )
    fill = CellMesh.from_tetrahedra(
        vertices,
        np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int64),
    )
    centers = np.mean(
        np.asarray(fill.coordinates)[np.asarray(fill.blocks[0].vertices, dtype=np.int64)],
        axis=1,
    )
    regions = (centers[:, 0] > 2.0).astype(np.int64)
    return source, fill, regions


def _adjacent_skew_tetrahedra() -> tuple[PiecewiseLinearComplex, CellMesh]:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.1, 0.1, 1.0),
            (2.0, -1.2, -1.5),
        ),
        dtype=np.float64,
    )
    polygons = tuple(
        np.asarray(loop, dtype=np.int64)
        for loop in (
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (2, 1, 4),
            (0, 4, 1),
            (0, 2, 4),
        )
    )
    source = PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(6, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("bipyramid-solid",),
        segments=np.zeros((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"bipyramid-v{row}" for row in range(5)),
        vertex_occurrence_paths=((),) * 5,
        facet_ids=tuple(f"bipyramid-f{row}" for row in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    fill = CellMesh.from_tetrahedra(
        vertices,
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64),
    )
    return source, fill


def _hex_volume(mesh: CellMesh) -> float:
    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    samples = 0.5 + np.asarray((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))
    coordinates = np.asarray(mesh.coordinates)
    total = 0.0
    for cell in np.asarray(mesh.blocks[0].vertices, dtype=np.int64):
        nodes = coordinates[cell]
        for point in product(samples, repeat=3):
            point_array = np.asarray(point, dtype=np.float64)
            factors = np.where(reference == 1.0, point_array, 1.0 - point_array)
            derivative = np.empty((8, 3), dtype=np.float64)
            for axis in range(3):
                other = [index for index in range(3) if index != axis]
                derivative[:, axis] = np.where(
                    reference[:, axis] == 1.0, 1.0, -1.0
                ) * np.prod(factors[:, other], axis=1)
            determinant = float(np.linalg.det(nodes.T @ derivative))
            assert determinant > 0.0
            total += determinant / 8.0
    return total


def test_source_frame_axis_bound_encloses_the_exact_jacobian_column_norm() -> None:
    # WIN-5 C5: one ``nextafter`` after ``sqrt(sum of squares)`` gave
    # 0x1.70ce94b6672e0p+2, whose square is below the exact column norm square.
    column = tuple(
        float.fromhex(value)
        for value in (
            "0x1.43d55519530b8p-6",
            "0x1.70abf9fc8c4c0p+2",
            "0x1.3cecce40c6800p-3",
        )
    )
    matrix = np.asarray(
        ((column[0], 0.0, 0.0), (column[1], 1.0, 0.0), (column[2], 0.0, 1.0)),
        dtype=np.float64,
    )
    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    mesh = CellMesh(reference, (CellBlock("cells", "hexahedron", np.arange(8)[None, :]),))
    element = lagrange_element("hexahedron", 1)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(nodes.shape[0], dtype=np.int64)[None, :]},
        nodes @ matrix.T,
    )
    frame = prepare_source_hex_frame(mesh, geometry)
    exact = sum((Fraction(value) ** 2 for value in column), Fraction(0))
    bound = Fraction(float(frame.axis_jacobian_bounds[0, 0]))
    assert exact <= bound * bound <= exact * (1 + Fraction(1, 2**47))


def test_integrable_per_cell_frame_layout_preserves_hex_quality_and_map() -> None:
    source, fill = _cube_plc_and_fill()
    schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0e-6,
        minimum_mean_ratio=1.0e-6,
        maximum_aspect_ratio=1.0e6,
    )
    prepared = prepare_frame_hex_grid(
        source,
        fill,
        schedule,
        0.6,
        source_id="frame-cube",
        source_revision="r1",
    )
    assert bool(prepared.frame.optimization.successful)
    assert prepared.frame.cell_frames.shape == (fill.connectivity.cell_count, 3, 3)
    assert prepared.frame.cell_global_ids.shape == (fill.connectivity.cell_count,)
    assert prepared.frame.source_complex_id == source.complex_id
    assert prepared.layout is not None
    assert prepared.grid is not None
    assert prepared.obstruction is None
    assert prepared.grid.frame_layout_id == prepared.layout.layout_id
    assert prepared.layout.cell_block_indices.tolist() == list(
        range(fill.connectivity.cell_count)
    )

    result = generate_frame_hexes(
        source,
        fill,
        np.zeros((fill.connectivity.cell_count,), dtype=np.int64),
        prepared,
        schedule,
    )

    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert result.validity.all_certified
    assert set(np.asarray(result.integer_blocks).tolist()) == set(
        range(fill.connectivity.cell_count)
    )
    assert _hex_volume(result.mesh) == pytest.approx(1.0, rel=1.0e-12)
    assert result.geometry.source_map is not None

    assert result.global_embedding.status == "certified"
    chain_witnesses = tuple(
        witness
        for witness in result.global_embedding.witnesses
        if witness.theorem == "forward_reference_chain_injectivity"
    )
    assert len(chain_witnesses) == 1
    chain_witness = chain_witnesses[0]
    assert chain_witness.premise_ids[:2] == (
        result.global_embedding.binding.binding_id,
        result.validity.certificate_id,
    )
    np.testing.assert_array_equal(
        np.asarray(chain_witness.entity_ids, dtype=np.int64),
        np.asarray(result.mesh.blocks[0].global_ids, dtype=np.int64),
    )
    rebuilt_embedding = GlobalEmbeddingCertificate.from_stored_fields(
        result.global_embedding
    )
    assert rebuilt_embedding.certificate_id == result.global_embedding.certificate_id


def test_connected_varying_field_returns_a_typed_integrability_obstruction() -> None:
    source, fill = _adjacent_skew_tetrahedra()
    schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0e-12,
        minimum_mean_ratio=1.0e-12,
        maximum_aspect_ratio=1.0e12,
    )
    prepared = prepare_frame_hex_grid(
        source,
        fill,
        schedule,
        0.4,
        source_id="skew-connected-tets",
        source_revision="r1",
    )

    assert prepared.frame.cell_frames.shape == (2, 3, 3)
    assert prepared.obstruction is not None
    assert prepared.layout is None
    assert prepared.grid is None
    assert prepared.obstruction.kind in {
        "boundary-parity",
        "nonintegrable-face-transition",
        "nonintegrable-block-spacing",
        "singular-frame-transition",
    }
    assert prepared.obstruction.entity_ids
    with pytest.raises(FrameFieldRefusal) as refusal:
        generate_frame_hexes(
            source,
            fill,
            np.zeros((2,), dtype=np.int64),
            prepared,
            schedule,
        )
    assert refusal.value.kind == prepared.obstruction.kind


def test_varying_multiblock_field_generates_certified_hexes() -> None:
    source, fill, regions = _two_disconnected_tetrahedra()
    schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0e-12,
        minimum_mean_ratio=1.0e-12,
        maximum_aspect_ratio=1.0e12,
    )
    prepared = prepare_frame_hex_grid(
        source,
        fill,
        schedule,
        0.55,
        source_id="two-tetra-blocks",
        source_revision="r1",
    )
    assert prepared.layout is not None
    assert prepared.grid is not None
    assert prepared.layout.cell_block_indices.tolist() == [0, 1]
    assert len(prepared.layout.block_axes) == 2
    assert not np.allclose(
        prepared.frame.cell_frames[0],
        prepared.frame.cell_frames[1],
        rtol=0.0,
        atol=1.0e-4,
    )
    assert prepared.frame.source_region_identities == (
        ("material-left", ()),
        ("material-right", ()),
    )
    assert prepared.frame.source_vertex_identities == tuple(
        zip(source.vertex_ids, source.vertex_occurrence_paths, strict=True)
    )
    assert prepared.frame.source_facet_identities == tuple(
        zip(source.facet_ids, source.facet_occurrence_paths, strict=True)
    )
    assert prepared.frame.source_feature_identities == tuple(
        zip(source.plc_edge_ids, source.plc_edge_occurrence_paths, strict=True)
    )
    result = generate_frame_hexes(source, fill, regions, prepared, schedule)

    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert result.validity.all_certified
    assert set(np.asarray(result.cell_regions).tolist()) == {0, 1}
    assert set(np.asarray(result.integer_blocks).tolist()) == {0, 1}
    assert _hex_volume(result.mesh) == pytest.approx((1.0 + 1.2 * 0.8 * 1.1) / 6.0)
    assert result.geometry.source_map is not None


def test_exact_rotated_block_transition_retains_integer_offset() -> None:
    identity = np.eye(3, dtype=np.int64)
    quarter_turn = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.int64)
    axis = (Fraction(-1), Fraction(0), Fraction(1))
    chart = (axis, axis, axis)
    layout = HexFrameIntegerLayout(
        np.asarray((201, 202), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int64),
        np.zeros((2, 3), dtype=np.int64),
        np.stack((identity, quarter_turn)),
        np.zeros((2, 3), dtype=np.float64),
        (chart, chart),
        np.asarray(((0, 0, 1, 1),), dtype=np.int64),
        quarter_turn[None, :, :],
        np.asarray(((2, 0, 0),), dtype=np.int64),
        np.ones((2,), dtype=np.int8),
        "complex",
        "mesh",
        "topology",
        "schedule",
        "frame",
    )

    assert np.array_equal(layout.face_transitions[0], quarter_turn)
    assert np.array_equal(layout.face_offsets[0], np.asarray((2, 0, 0)))
    assert layout.layout_id


def test_frame_integer_layout_refuses_odd_boundary_parity() -> None:
    axis = (Fraction(-1), Fraction(0), Fraction(1))
    chart = (axis, axis, axis)
    with pytest.raises(FrameFieldRefusal) as refusal:
        HexFrameIntegerLayout(
            np.asarray((301,), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.zeros((1, 3), dtype=np.int64),
            np.eye(3, dtype=np.float64)[None, :, :],
            np.zeros((1, 3), dtype=np.float64),
            (chart,),
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0, 3, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.asarray((-1,), dtype=np.int8),
            "complex",
            "mesh",
            "topology",
            "schedule",
            "frame",
        )
    assert refusal.value.kind == "boundary-parity"


def test_frame_integer_layout_refuses_nontrivial_cycle_holonomy() -> None:
    identity = np.eye(3, dtype=np.int64)
    quarter_turn = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.int64)
    axis = (Fraction(0), Fraction(1))
    chart = (axis, axis, axis)
    with pytest.raises(FrameFieldRefusal) as refusal:
        HexFrameIntegerLayout(
            np.asarray((101, 102, 103), dtype=np.int64),
            np.asarray((0, 1, 2), dtype=np.int64),
            np.zeros((3, 3), dtype=np.int64),
            np.repeat(identity[None, :, :], 3, axis=0),
            np.zeros((3, 3), dtype=np.float64),
            (chart, chart, chart),
            np.asarray(((0, 0, 1, 0), (1, 1, 2, 0), (2, 1, 0, 1)), dtype=np.int64),
            np.stack((identity, identity, quarter_turn)),
            np.zeros((3, 3), dtype=np.int64),
            np.ones((3,), dtype=np.int8),
            "complex",
            "mesh",
            "topology",
            "schedule",
            "frame",
        )
    assert refusal.value.kind == "cross-block-holonomy"
