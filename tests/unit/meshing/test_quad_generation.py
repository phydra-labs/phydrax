import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import final

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_geometry import _coordinate_coefficient_identity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.discretization.fem._reference import (
    FiniteElementSpec,
    lagrange_reference_node_data,
)
from phydrax.meshing._dual_extraction import (
    DualExtraction,
    restore_surface_quad_field_event,
    SurfaceCrossField,
    SurfaceQuadFieldEvent,
)
from phydrax.meshing._quad_generation import (
    _maximum_partial_pairing,
    _PairCandidate,
    _split_quad_templates_valid,
    extract_surface_quads,
    prepare_surface_cross_field,
    prepare_surface_quad_layout,
    QuadLayoutObstruction,
)


def test_partial_pairing_maximizes_cardinality_before_alignment() -> None:
    # A long alternating path has a unique perfect matching. Its expensive
    # edges must win over the one-shorter matching of well-aligned edges.
    pairs = 502
    candidates = [
        _PairCandidate(
            edge_row=i,
            faces=(i, i + 1),
            vertices=(0, 1, 2, 3),
            alignment_error=2.0 if i % 2 == 0 else 0.0,
            quad_quality=1.0,
        )
        for i in range(2 * pairs - 1)
    ]
    assert _maximum_partial_pairing(candidates, 2 * pairs) == list(
        range(0, 2 * pairs, 2)
    )


def test_partial_pairing_rejects_rounded_degenerate_split_templates() -> None:
    # Actual rejected holed-plate pair: the represented parent is nearly
    # collinear at one corner; its rounded subdivision cannot meet the same
    # determinant floor required of every published cell.
    points = np.asarray((
        (0.8, 0.8),
        (0.7250000000000001, 0.6000000000000001),
        (0.775, 0.4),
        (0.7875000000000001, 0.6000000000000001),
    ))
    assert not _split_quad_templates_valid((0, 1, 2, 3), points)
    square = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    assert _split_quad_templates_valid((0, 1, 2, 3), square)


def _disconnected_square_pairs() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (2.0, 1.0),
        )
    )
    return CellMesh.from_triangles(
        points,
        np.asarray(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7))),
    )


def _annulus() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 2.0),
            (1.0, 2.0),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        )
    )
    return CellMesh.from_triangles(points, triangles)


def _triangular_annulus() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (3.0, 0.0),
            (1.5, 3.0),
            (1.0, 0.8),
            (2.0, 0.8),
            (1.5, 1.8),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (2, 0, 3),
            (2, 3, 5),
        )
    )
    return CellMesh.from_triangles(points, triangles)


def _interior_vertex_grid() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
            (2.0, 2.0),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (3, 4, 7),
            (3, 7, 6),
            (4, 5, 8),
            (4, 8, 7),
        )
    )
    return CellMesh.from_triangles(points, triangles)


def _concave_l() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        )
    )
    triangles = np.asarray(((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5)))
    return CellMesh.from_triangles(points, triangles)


def _area(mesh: CellMesh) -> float:
    points = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices)
    return float(
        np.sum(
            np.abs(
                np.sum(
                    points[cells, 0] * np.roll(points[cells, 1], -1, axis=1)
                    - points[cells, 1] * np.roll(points[cells, 0], -1, axis=1),
                    axis=1,
                )
            )
            / 2.0
        )
    )


def _boundary_edge_count(mesh: CellMesh) -> int:
    counts: dict[tuple[int, int], int] = {}
    for block in mesh.blocks:
        for cell in np.asarray(block.vertices).tolist():
            for a, b in zip(cell, (*cell[1:], cell[0]), strict=True):
                edge = (min(a, b), max(a, b))
                counts[edge] = counts.get(edge, 0) + 1
    return sum(count == 1 for count in counts.values())


def test_cross_field_recombination_changes_topology_and_retains_exact_ancestry() -> None:
    source = _disconnected_square_pairs()
    connectivity = source.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    edges = np.asarray(connectivity.edges)
    first_diagonal = next(
        row for row, edge in enumerate(edges.tolist()) if tuple(sorted(edge)) == (0, 2)
    )

    cross_field = prepare_surface_cross_field(source)
    protected = prepare_surface_cross_field(source, feature_edges=[first_diagonal])
    layout = prepare_surface_quad_layout(source, cross_field)

    assert layout.paired_edge_rows.size == 2
    with pytest.raises(QuadLayoutObstruction) as obstruction:
        prepare_surface_quad_layout(source, protected)
    assert obstruction.value.category == "patch_parity"
    assert obstruction.value.source_cell_rows

    dual = extract_surface_quads(source)
    aligned = extract_surface_quads(source, cross_field=cross_field)
    source_edge_ids = np.asarray(dual.source_edge_ids, dtype=np.int64)
    mapped_source_edges = source_edge_ids >= 0
    assert np.any(mapped_source_edges)
    assert np.all(
        np.isin(
            source_edge_ids[mapped_source_edges],
            np.asarray(source.entity_set(1).entity_ids, dtype=np.int64),
        )
    )
    fractions = np.asarray(dual.source_edge_parameter_numerators, dtype=np.int64)[
        mapped_source_edges
    ]
    denominators = np.asarray(dual.source_edge_parameter_denominators, dtype=np.int64)[
        mapped_source_edges
    ]
    assert np.all(denominators > 0)
    assert np.all(fractions <= denominators)
    assert np.any(denominators == 2)
    assert dual.mesh.connectivity.cell_count == 12
    assert aligned.mesh.connectivity.cell_count == 2
    assert aligned.validity.all_certified
    assert aligned.parent_offsets.shape == (3,)
    np.testing.assert_array_equal(np.diff(aligned.parent_offsets), np.full(2, 2))
    np.testing.assert_array_equal(
        np.asarray(aligned.vertex_supports)[:, 0],
        np.arange(source.coordinates.shape[0]),
    )
    target_source_edges = np.asarray(aligned.source_edge_ids, dtype=np.int64)
    assert np.all(target_source_edges >= 0)
    source_edge_entity_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    feature_edge_ids = source_edge_entity_ids[
        np.asarray(cross_field.feature_edges, dtype=np.int64)
    ]
    assert np.all(np.isin(feature_edge_ids, target_source_edges))
    assert np.all(dual.valences[dual.singular_vertices] != 4)
    assert not np.array_equal(dual.mesh.topology_id, aligned.mesh.topology_id)
    assert float(cross_field.field_solve_relative_residual) <= 1.0e-10
    assert cross_field.field_solver == "scipy.sparse.linalg.spsolve"
    assert cross_field.singular_vertices.shape == cross_field.singularity_indices.shape
    assert float(cross_field.field_energy) >= 0.0
    assert _area(dual.mesh) == pytest.approx(2.0)
    assert _area(aligned.mesh) == pytest.approx(2.0)


def test_simplex2_lowering_keeps_source_family_during_quad_extraction() -> None:
    baseline = _disconnected_square_pairs()
    baseline_block = baseline.blocks[0]
    source = CellMesh(
        np.asarray(baseline.coordinates),
        (
            CellBlock(
                baseline_block.name,
                "simplex:2",
                np.asarray(baseline_block.vertices, dtype=np.int64),
                global_ids=np.asarray(baseline_block.global_ids, dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(baseline.vertex_global_ids, dtype=np.int64),
    )
    connectivity = source.connectivity
    assert isinstance(connectivity, SimplicialConnectivity)
    assert connectivity.dimension == 2

    cross_field = prepare_surface_cross_field(source)
    layout = prepare_surface_quad_layout(source, cross_field)
    result = extract_surface_quads(source, cross_field=cross_field)

    assert layout.paired_edge_rows.size == 2
    assert source.blocks[0].cell_kind == "simplex:2"
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.validity.all_certified
    assert result.mesh.connectivity.cell_count == 2
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells, dtype=np.int64)),
        np.arange(connectivity.cell_count, dtype=np.int64),
    )
    assert _area(result.mesh) == pytest.approx(2.0)


def test_independent_singular_fields_select_distinct_quad_topologies(tmp_path: Path) -> None:
    from copy import copy

    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _solve_family_affine_state,
    )

    # One source and one boundary-feature set; the two prescribed fourfold
    # directions differ only in phase around the same index-one singularity.
    ring_angles = 2.0 * np.pi * np.arange(6) / 6
    points = np.vstack(
        (np.zeros((1, 2)), np.column_stack((np.cos(ring_angles), np.sin(ring_angles))))
    )
    triangles = np.asarray([(0, i + 1, (i + 1) % 6 + 1) for i in range(6)])
    source = CellMesh.from_triangles(points, triangles)
    frames = np.asarray(prepare_surface_cross_field(source).frames)
    centers = points[triangles].mean(axis=1)
    polar = np.arctan2(centers[:, 1], centers[:, 0])
    fields = []
    layouts = []
    results = []
    for phase in (0.0, np.pi / 8):
        direction = polar / 4 + phase
        physical = np.column_stack((np.cos(direction), np.sin(direction), np.zeros(6)))
        local = np.einsum("fij,fj->fi", frames, physical)
        guide = np.arctan2(local[:, 1], local[:, 0])
        field = prepare_surface_cross_field(source, guide_angles=guide)
        layout = prepare_surface_quad_layout(source, field)
        result = extract_surface_quads(source, cross_field=field)
        fields.append(field)
        layouts.append(layout)
        results.append(result)
        np.testing.assert_array_equal(field.singular_vertices, [0])
        np.testing.assert_array_equal(field.singularity_indices, [1])
        assert int(np.asarray(result.valences)[0]) == 3
        np.testing.assert_array_equal(
            np.sort(np.asarray(result.parent_cells)), np.arange(6)
        )
        assert result.mesh.blocks[0].cell_kind == "quadrilateral"
        assert result.mesh.connectivity.cell_count == 3
        assert result.validity.all_certified
        assert _area(result.mesh) == pytest.approx(3 * np.sqrt(3) / 2)
        assert _boundary_edge_count(result.mesh) == 6
        target_cells = np.asarray(result.mesh.blocks[0].vertices)
        offsets = np.asarray(result.parent_offsets)
        parents = np.asarray(result.parent_cells)
        for row, quad in enumerate(target_cells):
            # The quad boundary must equal the symmetric difference of its
            # two source-triangle boundaries, not just have the same area.
            source_boundary = set()
            for parent in parents[offsets[row] : offsets[row + 1]]:
                triangle = triangles[parent]
                for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
                    edge = tuple(sorted((int(a), int(b))))
                    source_boundary.symmetric_difference_update((edge,))
            target_boundary = {
                tuple(sorted((int(a), int(b))))
                for a, b in zip(quad, np.roll(quad, -1), strict=True)
            }
            assert target_boundary == source_boundary
            q = points[quad]
            sides = np.roll(q, -1, axis=0) - q
            turns = sides[:, 0] * np.roll(sides[:, 1], -1) - sides[:, 1] * np.roll(
                sides[:, 0], -1
            )
            assert np.min(turns) > 0.1
        record, _, _ = _solve_family_affine_state(result)
        assert record["cell_kind"] == "quadrilateral"
        assert isinstance(record["maximum_error"], float)
        assert isinstance(record["normalized_residual"], float)
        assert record["maximum_error"] < 2.0e-9
        assert record["normalized_residual"] < 2.0e-9
    np.testing.assert_array_equal(fields[0].feature_edges, fields[1].feature_edges)
    assert fields[0].field_id != fields[1].field_id
    assert layouts[0].layout_id != layouts[1].layout_id
    assert not np.array_equal(layouts[0].paired_edge_rows, layouts[1].paired_edge_rows)
    assert results[0].mesh.topology_id != results[1].mesh.topology_id
    geometry = CellGeometrySpec.affine(source)
    events = []
    for result in results:
        assert isinstance(result.field_event, SurfaceQuadFieldEvent)
        events.append(result.field_event)
    assert events[0].event_id != events[1].event_id
    for index, event in enumerate(events):
        recipe = event.archive_recipe()
        path = tmp_path / f"field-event-{index}.phx"
        composition = Composition(
            (
                CompositionEntry(
                    event,
                    entry_id="quad/event",
                    role="model-state",
                    owner_id=source.mesh_id,
                    structure_id=event.event_id,
                    revision_id=source.geometry_id,
                    semantics_id="surface-quad-field-event",
                ),
            ),
            boundary_id="surface-quad-field-event",
        )
        write_model_artifact(path, composition, limits=None)
        restored_composition = read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=None,
        )
        archive_event = restored_composition.value("quad/event")
        assert archive_event.event_id == event.event_id
        assert archive_event.source.mesh_id == source.mesh_id
        assert archive_event.field.field_id == event.field.field_id
        assert archive_event.layout.layout_id == event.layout.layout_id
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from phydrax.lifecycle import Composition; "
                "from phydrax.meshing._model_codec import read_model_artifact, "
                "restore_model_archive_value; "
                "c=read_model_artifact(sys.argv[1],expected_type=Composition,"
                "restore=restore_model_archive_value,limits=None); "
                "e=c.value('quad/event'); "
                "assert (e.event_id,e.source.mesh_id,e.field.field_id,e.layout.layout_id)"
                " == tuple(sys.argv[2:])",
                str(path),
                event.event_id,
                source.mesh_id,
                event.field.field_id,
                event.layout.layout_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        restored = restore_surface_quad_field_event(recipe, source, geometry)
        assert restored.event_id == event.event_id
        assert restored.field.field_id == event.field.field_id
        assert restored.layout.layout_id == event.layout.layout_id
        np.testing.assert_array_equal(restored.field.angles, event.field.angles)
        for name in (
            "frames",
            "transport_edges",
            "transport_angles",
            "feature_rows",
            "feature_angles",
            "feature_residuals",
            "field_solve_residual",
            "field_solve_relative_residual",
            "field_energy",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(restored.field, name)),
                np.asarray(getattr(event.field, name)),
            )
        np.testing.assert_array_equal(
            restored.layout.paired_edge_rows, event.layout.paired_edge_rows
        )
        with pytest.raises(ValueError, match="cross field"):
            restore_surface_quad_field_event(
                recipe._replace(field_id="altered"), source, geometry
            )
        tampered = eqx.tree_at(
            lambda value: value.field_energy, event.field, event.field.field_energy + 1.0
        )
        with pytest.raises(ValueError, match="cross field does not reproduce"):
            restore_surface_quad_field_event(
                recipe._replace(field=tampered), source, geometry
            )

    def alignment(result: DualExtraction, field: SurfaceCrossField) -> float:
        cells = np.asarray(result.mesh.blocks[0].vertices)
        offsets = np.asarray(result.parent_offsets)
        parents = np.asarray(result.parent_cells)
        field_frames = np.asarray(field.frames)
        field_angles = np.asarray(field.angles)
        error = 0.0
        for row, cell in enumerate(cells):
            weighted = length_sum = 0.0
            for a, b in zip(cell, np.roll(cell, -1), strict=True):
                face = next(
                    int(f)
                    for f in parents[offsets[row] : offsets[row + 1]]
                    if a in triangles[f] and b in triangles[f]
                )
                vector = np.append(points[b] - points[a], 0.0)
                local = field_frames[face] @ vector
                angle = np.arctan2(local[1], local[0])
                length = float(np.linalg.norm(vector))
                weighted += length * (1 - np.cos(4 * (angle - field_angles[face])))
                length_sum += length
            error += weighted / length_sum
        return error

    assert alignment(results[0], fields[0]) < alignment(results[1], fields[0])
    assert alignment(results[1], fields[1]) < alignment(results[0], fields[1])
    altered = eqx.tree_at(lambda value: value.angles, fields[0], fields[0].angles + 0.1)
    with pytest.raises(QuadLayoutObstruction, match="scientific identity"):
        prepare_surface_quad_layout(source, altered)
    same_id_source = eqx.tree_at(
        lambda value: value.vertex_global_ids,
        source,
        source.vertex_global_ids + 100,
    )
    assert same_id_source.mesh_id == source.mesh_id
    with pytest.raises(ValueError, match="source mesh topology and geometry"):
        prepare_surface_quad_layout(same_id_source, fields[0])
    with pytest.raises(ValueError, match="cross field does not reproduce"):
        restore_surface_quad_field_event(
            events[0].archive_recipe(), same_id_source, geometry
        )
    same_id_target = eqx.tree_at(
        lambda value: value.vertex_global_ids,
        results[0].mesh,
        results[0].mesh.vertex_global_ids + 100,
    )
    assert same_id_target.mesh_id == results[0].mesh.mesh_id
    assert results[0].geometry is not None
    with pytest.raises(ValueError, match="target mesh and geometry"):
        events[0].require_target(same_id_target, results[0].geometry)
    same_id_geometry = copy(results[0].geometry)
    object.__setattr__(same_id_geometry, "block_names", ("forged",))
    assert cell_geometry_id(same_id_geometry) == cell_geometry_id(results[0].geometry)
    with pytest.raises(ValueError, match="target mesh and geometry"):
        events[0].require_target(results[0].mesh, same_id_geometry)
    shifted = CellMesh.from_triangles(points + (0.1, 0.0), triangles)
    forged_recipe = events[0].archive_recipe()._replace(
        source=shifted, source_geometry=CellGeometrySpec.affine(shifted)
    )
    forged = Composition(
        (
            CompositionEntry(
                forged_recipe,
                entry_id="quad/event",
                role="model-state",
                owner_id=shifted.mesh_id,
                structure_id=events[0].event_id,
                revision_id=shifted.geometry_id,
                semantics_id="surface-quad-field-event",
            ),
        ),
        boundary_id="surface-quad-field-event",
    )
    forged_path = tmp_path / "forged-source.phx"
    write_model_artifact(forged_path, forged, limits=None)
    rejected = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from phydrax.lifecycle import Composition; "
            "from phydrax.meshing._model_codec import read_model_artifact, "
            "restore_model_archive_value; "
            "read_model_artifact(sys.argv[1],expected_type=Composition,"
            "restore=restore_model_archive_value,limits=None)",
            str(forged_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "cross field does not reproduce" in rejected.stderr
    corrupt_field = eqx.tree_at(
        lambda value: value.feature_residuals,
        fields[0],
        fields[0].feature_residuals + 0.1,
    )
    corrupt_recipe = events[0].archive_recipe()._replace(field=corrupt_field)
    corrupt = Composition(
        (
            CompositionEntry(
                corrupt_recipe,
                entry_id="quad/event",
                role="model-state",
                owner_id=source.mesh_id,
                structure_id=events[0].event_id,
                revision_id=source.geometry_id,
                semantics_id="surface-quad-field-event",
            ),
        ),
        boundary_id="surface-quad-field-event",
    )
    corrupt_path = tmp_path / "forged-field-evidence.phx"
    write_model_artifact(corrupt_path, corrupt, limits=None)
    rejected_evidence = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from phydrax.lifecycle import Composition; "
            "from phydrax.meshing._model_codec import read_model_artifact, "
            "restore_model_archive_value; "
            "read_model_artifact(sys.argv[1],expected_type=Composition,"
            "restore=restore_model_archive_value,limits=None)",
            str(corrupt_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected_evidence.returncode != 0
    assert "cross field does not reproduce" in rejected_evidence.stderr
    with pytest.raises(ValueError, match="source mesh topology and geometry"):
        prepare_surface_quad_layout(shifted, fields[0])
    with pytest.raises(ValueError, match="one finite angle"):
        prepare_surface_cross_field(source, guide_angles=[float("nan")] * 6)


def test_public_native_quad_fields_change_topology_on_same_source(tmp_path: Path) -> None:
    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.simplicial import PlanarMeshRegion
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshingFailure,
        MeshingScope,
        MeshQualityTarget,
        NativeMeshingOptions,
        NativeMeshingProvider,
        SizeControlStrength,
        SurfaceMeshingSpec,
        UniformSizeControl,
    )
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from phydrax.meshing._result import CellMeshingResult
    from phydrax.meshing.providers._native_sources import NativePlanarSource
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _solve_family_affine_state,
    )

    vertices = np.asarray(
        ((2.0, 0.0), (1.0, 1.0), (-1.0, 1.0), (-2.0, 0.0), (-1.0, -1.0), (1.0, -1.0))
    )
    region = PlanarMeshRegion(vertices, ((0, 1, 2, 3, 4, 5),), feature_id="field-hex")
    source = NativePlanarSource(
        region,
        "r1",
        region_scope_entity_ids=np.asarray([501]),
        vertex_entity_ids=np.arange(101, 107),
        edge_entity_ids=np.arange(201, 207),
    )
    scope = MeshingScope.whole(source, 2)
    controls = (UniformSizeControl(scope, 0.75, strength=SizeControlStrength.SOFT),)
    quality = MeshQualityTarget(minimum_angle=0.01, hard=True)
    coordinates = SpatialCoordinateContract.si()
    triangle_spec = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=controls,
        quality_target=quality,
    )
    triangle = NativeMeshingProvider(
        NativeMeshingOptions("planar_constrained_delaunay")
    ).plan(source, triangle_spec, coordinate_contract=coordinates).execute()
    assert triangle.surface_quad_field_event is None
    default_result_path = tmp_path / "default-surface-result.phx"
    write_model_artifact(default_result_path, triangle, limits=None)
    restored_triangle = read_model_artifact(
        default_result_path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored_triangle.result_id == triangle.result_id
    assert restored_triangle.surface_quad_field_event is None
    scaffold = triangle.mesh
    triangles = np.asarray(scaffold.blocks[0].vertices)
    points = np.asarray(scaffold.coordinates)
    frames = np.asarray(prepare_surface_cross_field(scaffold).frames)
    centers = points[triangles].mean(axis=1)
    polar = np.arctan2(centers[:, 1], centers[:, 0])
    quad_target = CellMeshingTarget(2, 2, CellFamilyPolicy(required=("quadrilateral",)))
    requests = []
    results = []
    for phase in (0.0, np.pi / 8):
        direction = polar / 4 + phase
        physical = np.column_stack(
            (np.cos(direction), np.sin(direction), np.zeros(len(triangles)))
        )
        local = np.einsum("fij,fj->fi", frames, physical)
        field = prepare_surface_cross_field(
            scaffold, guide_angles=np.arctan2(local[:, 1], local[:, 0])
        )
        specification = SurfaceMeshingSpec(
            quad_target,
            scope,
            size_controls=controls,
            quality_target=quality,
            cross_field=field,
        )
        result = NativeMeshingProvider(NativeMeshingOptions("planar_dual_quad")).plan(
            source, specification, coordinate_contract=coordinates
        ).execute()
        requests.append(specification)
        results.append(result)
        assert result.certification is not None and result.certification.passed
        assert result.certification.inputs.source_id == source.source_id
        assert result.certification.inputs.source_revision == source.source_revision
        assert result.audit.passed and result.compliance.passed
        assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
        assert any(association.complete for association in result.associations)
        assert _area(result.mesh) == pytest.approx(6.0)
        event = result.surface_quad_field_event
        assert event is not None and event.field.field_id == field.field_id
        assert event.source.mesh_id == scaffold.mesh_id
        np.testing.assert_array_equal(field.singularity_indices, [1])
        event.require_target(result.mesh, result.geometry)
        record, _, _ = _solve_family_affine_state(result)
        assert isinstance(record["maximum_error"], float)
        assert isinstance(record["normalized_residual"], float)
        assert record["maximum_error"] < 2.0e-9
        assert record["normalized_residual"] < 2.0e-9

        # Independent boundary coverage and source fidelity: every output
        # boundary edge lies on an authored segment, and the projected
        # intervals partition each segment without gaps or overlap.
        cells = np.asarray(result.mesh.blocks[0].vertices)
        coordinates_out = np.asarray(result.mesh.coordinates)
        for cell in cells:
            q = coordinates_out[cell]
            incoming = np.roll(q, 1, axis=0) - q
            outgoing = np.roll(q, -1, axis=0) - q
            cosine = np.sum(incoming * outgoing, axis=1) / (
                np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
            )
            assert np.min(np.arccos(np.clip(cosine, -1.0, 1.0))) >= 0.01
        counts = {}
        for cell in cells:
            for a, b in zip(cell, np.roll(cell, -1), strict=True):
                key = tuple(sorted((int(a), int(b))))
                counts[key] = counts.get(key, 0) + 1
        intervals = [[] for _ in range(6)]
        for edge, count in counts.items():
            if count != 1:
                continue
            a, b = coordinates_out[list(edge)]
            for side in range(6):
                start = vertices[side]
                axis = vertices[(side + 1) % 6] - start
                denominator = float(axis @ axis)
                ta, tb = float((a - start) @ axis / denominator), float(
                    (b - start) @ axis / denominator
                )
                if (
                    np.linalg.norm(a - (start + ta * axis)) < 1.0e-10
                    and np.linalg.norm(b - (start + tb * axis)) < 1.0e-10
                    and -1.0e-10 <= ta <= 1.0 + 1.0e-10
                    and -1.0e-10 <= tb <= 1.0 + 1.0e-10
                ):
                    intervals[side].append(tuple(sorted((ta, tb))))
                    break
            else:
                raise AssertionError("Target boundary edge is not on the source.")
        for pieces in intervals:
            ordered = sorted(pieces)
            assert ordered and ordered[0][0] == pytest.approx(0.0, abs=1.0e-10)
            assert ordered[-1][1] == pytest.approx(1.0, abs=1.0e-10)
            for previous, current in zip(ordered, ordered[1:]):
                assert previous[1] == pytest.approx(current[0], abs=1.0e-10)

        path = tmp_path / f"field-request-{len(requests)}.phx"
        write_model_artifact(path, specification, limits=None)
        restored = read_model_artifact(
            path,
            expected_type=SurfaceMeshingSpec,
            restore=restore_model_archive_value,
            limits=None,
        )
        assert restored.specification_id == specification.specification_id
        assert restored.cross_field.field_id == field.field_id
        result_path = tmp_path / f"field-result-{len(results)}.phx"
        write_model_artifact(result_path, result, limits=None)
        restored_result = read_model_artifact(
            result_path,
            expected_type=CellMeshingResult,
            restore=restore_model_archive_value,
            limits=None,
        )
        assert restored_result.result_id == result.result_id
        assert restored_result.surface_quad_field_event.event_id == event.event_id

        request_entry = CompositionEntry(
            specification,
            entry_id="quad/request",
            role="model-state",
            owner_id=source.source_id,
            structure_id=specification.specification_id,
            revision_id=source.source_revision,
            semantics_id="surface-quad-request",
        )
        result_entry = CompositionEntry(
            result,
            entry_id="quad/result",
            role="model-state",
            owner_id=source.source_id,
            structure_id=result.result_id,
            revision_id=source.source_revision,
            semantics_id="surface-quad-result",
            dependencies=(request_entry.binding("structure"),),
        )
        composition = Composition(
            (request_entry, result_entry), boundary_id="surface-quad-accepted"
        )
        composition_path = tmp_path / f"field-composition-{len(results)}.phx"
        write_model_artifact(composition_path, composition, limits=None)
        restored_composition = read_model_artifact(
            composition_path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=None,
        )
        assert (
            restored_composition.value("quad/request").specification_id
            == specification.specification_id
        )
        assert restored_composition.value("quad/result").result_id == result.result_id
        reader = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from phydrax.lifecycle import Composition; "
                "from phydrax.meshing._model_codec import read_model_artifact, "
                "restore_model_archive_value; "
                "c=read_model_artifact(sys.argv[1],expected_type=Composition,"
                "restore=restore_model_archive_value,limits=None); "
                "s=c.value('quad/request'); r=c.value('quad/result'); "
                "assert (s.specification_id,r.result_id,"
                "r.surface_quad_field_event.event_id)==tuple(sys.argv[2:])",
                str(composition_path),
                specification.specification_id,
                result.result_id,
                event.event_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert reader.returncode == 0, reader.stderr
    assert requests[0].specification_id != requests[1].specification_id
    assert requests[0].cross_field is not None
    assert requests[1].cross_field is not None
    assert results[0].surface_quad_field_event is not None
    assert results[1].surface_quad_field_event is not None
    np.testing.assert_array_equal(
        requests[0].cross_field.feature_edges, requests[1].cross_field.feature_edges
    )
    first_angles = np.asarray(requests[0].cross_field.angles)
    second_angles = np.asarray(requests[1].cross_field.angles)
    assert np.max(np.abs(np.sin(4 * (first_angles - second_angles)))) > 0.1
    assert results[0].mesh.topology_id != results[1].mesh.topology_id
    assert (
        results[0].surface_quad_field_event.event_id
        != results[1].surface_quad_field_event.event_id
    )
    with pytest.raises(MeshingFailure, match="source-bound cross fields"):
        NativeMeshingProvider(NativeMeshingOptions("planar_constrained_delaunay")).plan(
            source, requests[0], coordinate_contract=coordinates
        )
    shifted = CellMesh.from_triangles(points + (0.1, 0.0), triangles)
    wrong_field = prepare_surface_cross_field(
        shifted, guide_angles=requests[0].cross_field.guide_angles
    )
    wrong_request = SurfaceMeshingSpec(
        quad_target,
        scope,
        size_controls=controls,
        quality_target=quality,
        cross_field=wrong_field,
    )
    with pytest.raises(QuadLayoutObstruction, match="actual triangle scaffold") as mismatch:
        NativeMeshingProvider(NativeMeshingOptions("planar_dual_quad")).plan(
            source, wrong_request, coordinate_contract=coordinates
        ).execute()
    assert mismatch.value.category == "source_identity"
    assert isinstance(scaffold.connectivity, PolygonalConnectivity)
    interior_edge = int(
        np.flatnonzero(~np.asarray(scaffold.connectivity.boundary_edges))[0]
    )
    wrong_features = prepare_surface_cross_field(
        scaffold,
        feature_edges=[interior_edge],
        guide_angles=requests[0].cross_field.guide_angles,
    )
    wrong_feature_request = SurfaceMeshingSpec(
        quad_target,
        scope,
        size_controls=controls,
        quality_target=quality,
        cross_field=wrong_features,
    )
    with pytest.raises(
        QuadLayoutObstruction, match="exactly the scaffold's physical features"
    ) as feature_mismatch:
        NativeMeshingProvider(NativeMeshingOptions("planar_dual_quad")).plan(
            source, wrong_feature_request, coordinate_contract=coordinates
        ).execute()
    assert feature_mismatch.value.category == "field_feature_integrity"
    assert interior_edge in feature_mismatch.value.source_edge_rows
    forged_field = eqx.tree_at(
        lambda value: value.angles,
        requests[0].cross_field,
        requests[0].cross_field.angles + 0.1,
    )
    forged_request = eqx.tree_at(
        lambda value: value.cross_field, requests[0], forged_field
    )
    forged_request_path = tmp_path / "forged-field-request.phx"
    write_model_artifact(forged_request_path, forged_request, limits=None)
    with pytest.raises((ValueError, ArrayArchiveCorruptionError)):
        read_model_artifact(
            forged_request_path,
            expected_type=SurfaceMeshingSpec,
            restore=restore_model_archive_value,
            limits=None,
        )
    forged_event = eqx.tree_at(
        lambda value: value.field, results[0].surface_quad_field_event, forged_field
    )
    forged_result = eqx.tree_at(
        lambda value: value.surface_quad_field_event, results[0], forged_event
    )
    with pytest.raises((ValueError, QuadLayoutObstruction)):
        write_model_artifact(tmp_path / "forged-event-result.phx", forged_result, limits=None)
    default_path = tmp_path / "default-surface-request.phx"
    write_model_artifact(default_path, triangle_spec, limits=None)
    restored_default = read_model_artifact(
        default_path,
        expected_type=SurfaceMeshingSpec,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored_default.cross_field is None
    assert restored_default.specification_id == triangle_spec.specification_id
    default_request_entry = CompositionEntry(
        triangle_spec,
        entry_id="surface/request",
        role="model-state",
        owner_id=source.source_id,
        structure_id=triangle_spec.specification_id,
        revision_id=source.source_revision,
        semantics_id="surface-request",
    )
    default_result_entry = CompositionEntry(
        triangle,
        entry_id="surface/result",
        role="model-state",
        owner_id=source.source_id,
        structure_id=triangle.result_id,
        revision_id=source.source_revision,
        semantics_id="surface-result",
        dependencies=(default_request_entry.binding("structure"),),
    )
    default_composition = Composition(
        (default_request_entry, default_result_entry), boundary_id="default-surface"
    )
    default_composition_path = tmp_path / "default-surface-composition.phx"
    write_model_artifact(default_composition_path, default_composition, limits=None)
    restored_composition = read_model_artifact(
        default_composition_path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored_composition.value("surface/request").cross_field is None
    assert restored_composition.value("surface/result").surface_quad_field_event is None
    assert restored_composition.value("surface/result").result_id == triangle.result_id


def test_distinct_feature_driven_fields_change_admissible_annulus_layout() -> None:
    original = _annulus()
    points = np.asarray(original.coordinates).copy()
    points[:, 0] += 0.13 * points[:, 1] ** 2
    points[:, 1] += 0.07 * points[:, 0] ** 2
    triangles = np.asarray(original.blocks[0].vertices)
    source = CellMesh.from_triangles(points, triangles)
    assert isinstance(source.connectivity, PolygonalConnectivity)
    edges = np.asarray(source.connectivity.edges)
    guide = int(np.flatnonzero(np.all(edges == (0, 4), axis=1))[0])
    baseline = prepare_surface_cross_field(source)
    guided = prepare_surface_cross_field(source, feature_edges=[guide])
    assert np.max(
        np.abs(np.sin(4 * (np.asarray(baseline.angles) - np.asarray(guided.angles))))
    ) > 0.5
    assert guided.source_topology_id == source.topology_id
    assert guided.source_geometry_id == source.geometry_id
    baseline_layout = prepare_surface_quad_layout(source, baseline)
    guided_layout = prepare_surface_quad_layout(source, guided)
    np.testing.assert_array_equal(
        baseline_layout.paired_edge_rows, guided_layout.paired_edge_rows
    )
    assert baseline_layout.layout_id != guided_layout.layout_id

    # The guide is a retained quad edge, not one of the deleted diagonals.
    layout = prepare_surface_quad_layout(source, guided, maximum_field_error=0.7)
    assert guide not in np.asarray(layout.paired_edge_rows)
    with pytest.raises(QuadLayoutObstruction):
        prepare_surface_quad_layout(source, baseline, maximum_field_error=0.7)
    result = extract_surface_quads(source, cross_field=guided, maximum_field_error=0.7)
    assert result.validity.all_certified
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells)), np.arange(len(triangles))
    )
    quads = np.asarray(result.mesh.blocks[0].vertices)
    assert len(quads) == 4
    target_edges = {
        tuple(sorted((int(a), int(b))))
        for cell in quads
        for a, b in zip(cell, np.roll(cell, -1), strict=True)
    }
    assert tuple(edges[guide]) in target_edges
    assert all(
        tuple(edges[row]) in target_edges for row in np.asarray(guided.feature_edges)
    )

    # Independently integrate the fourfold error over each quad boundary,
    # using the incident original triangle and the physical edge length.
    def pair_errors(field: SurfaceCrossField) -> list[float]:
        offsets = np.asarray(result.parent_offsets)
        parents = np.asarray(result.parent_cells)
        frames = np.asarray(field.frames)
        angles = np.asarray(field.angles)
        errors = []
        for row, cell in enumerate(quads):
            faces = parents[offsets[row] : offsets[row + 1]]
            weighted = total = 0.0
            for a, b in zip(cell, np.roll(cell, -1), strict=True):
                face = next(
                    int(f) for f in faces if a in triangles[f] and b in triangles[f]
                )
                direction = np.append(points[b] - points[a], 0.0)
                local = frames[face] @ direction
                theta = np.arctan2(local[1], local[0])
                length = float(np.linalg.norm(direction))
                weighted += length * (1.0 - np.cos(4.0 * (theta - angles[face])))
                total += length
            errors.append(weighted / total)
        return errors

    assert max(pair_errors(guided)) < 0.7
    assert max(pair_errors(baseline)) > 0.7
    source_area = sum(
        (
            (points[t[1], 0] - points[t[0], 0]) * (points[t[2], 1] - points[t[0], 1])
            - (points[t[1], 1] - points[t[0], 1]) * (points[t[2], 0] - points[t[0], 0])
        )
        / 2
        for t in triangles
    )
    assert _area(result.mesh) == pytest.approx(source_area)
    assert _boundary_edge_count(result.mesh) == _boundary_edge_count(source)


def test_feature_guided_quads_preserve_a_multiply_connected_domain() -> None:
    source = _annulus()
    field = prepare_surface_cross_field(source)
    result = extract_surface_quads(source, cross_field=field)
    topology = result.mesh.topology

    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.mesh.connectivity.cell_count == 4
    assert result.validity.all_certified
    assert _area(result.mesh) == pytest.approx(8.0)
    assert _boundary_edge_count(result.mesh) == _boundary_edge_count(source)
    assert (
        topology.entities(0).count
        - topology.entities(1).count
        + topology.entities(2).count
    ) == 0
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells)),
        np.arange(source.connectivity.cell_count),
    )
    assert result.parent_offsets.shape == (result.mesh.connectivity.cell_count + 1,)
    np.testing.assert_array_equal(np.diff(result.parent_offsets), np.full(4, 2))
    assert result.boundary_vertices.sum() == 8


def test_feature_guided_layout_enforces_interior_regular_valence() -> None:
    source = _interior_vertex_grid()
    field = prepare_surface_cross_field(source)
    result = extract_surface_quads(source, cross_field=field)

    assert result.mesh.connectivity.cell_count == 4
    assert result.validity.all_certified
    assert not bool(np.asarray(result.boundary_vertices)[4])
    assert int(np.asarray(result.valences)[4]) == 4
    assert 4 not in set(np.asarray(result.singular_vertices).tolist())
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.parent_cells)),
        np.arange(source.connectivity.cell_count),
    )


def test_feature_guided_quads_preserve_a_concave_source_domain() -> None:
    source = _concave_l()
    field = prepare_surface_cross_field(source)
    result = extract_surface_quads(source, cross_field=field, maximum_field_error=1.5)
    topology = result.mesh.topology

    assert result.mesh.connectivity.cell_count == 2
    assert result.validity.all_certified
    assert _area(result.mesh) == pytest.approx(3.0)
    assert _boundary_edge_count(result.mesh) == _boundary_edge_count(source)
    assert (
        topology.entities(0).count
        - topology.entities(1).count
        + topology.entities(2).count
    ) == 1


def test_cross_field_route_reports_unsatisfied_patch_parity() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),)),
    )
    field = prepare_surface_cross_field(source)

    with pytest.raises(QuadLayoutObstruction) as obstruction:
        extract_surface_quads(source, cross_field=field)
    assert obstruction.value.category == "patch_parity"
    assert obstruction.value.source_cell_rows == (0,)


def test_unpaired_triangles_subdivide_with_strip_propagated_midpoints() -> None:
    # A square pair plus one triangle whose only pair would be degenerate.
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0))),
        np.asarray(((0, 1, 2), (0, 2, 3), (1, 4, 2))),
    )
    field = prepare_surface_cross_field(source)
    with pytest.raises(QuadLayoutObstruction) as obstruction:
        extract_surface_quads(source, cross_field=field)
    assert obstruction.value.category == "patch_parity"

    result = extract_surface_quads(
        source, cross_field=field, unpaired_triangles="subdivide"
    )

    # The triangle splits in three; its split edge (1, 2) crosses the square
    # strip to the boundary edge (3, 0), so the square splits in two.
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.mesh.connectivity.cell_count == 5
    assert result.validity.all_certified
    assert _area(result.mesh) == pytest.approx(1.5)
    offsets = np.asarray(result.parent_offsets, dtype=np.int64)
    assert sorted(np.diff(offsets).tolist()) == [1, 1, 1, 2, 2]
    supports = np.asarray(result.vertex_supports, dtype=np.int64)
    midpoint_supports = {
        tuple(int(value) for value in row if value >= 0)
        for row in supports.tolist()
        if np.count_nonzero(np.asarray(row) >= 0) == 2
    }
    assert {(0, 3), (1, 2)} <= midpoint_supports
    assert (0, 2) not in midpoint_supports
    assert _boundary_edge_count(result.mesh) == _boundary_edge_count(source) + 3


def test_four_triangle_star_subdivides_without_a_positive_pairing() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
    )
    field = prepare_surface_cross_field(source)

    result = extract_surface_quads(
        source, cross_field=field, unpaired_triangles="subdivide"
    )

    assert result.mesh.connectivity.cell_count == 12
    assert result.validity.all_certified
    assert _area(result.mesh) == pytest.approx(1.0)


def test_cross_field_route_reports_odd_boundary_cycle() -> None:
    source = _triangular_annulus()
    field = prepare_surface_cross_field(source)

    with pytest.raises(QuadLayoutObstruction) as obstruction:
        prepare_surface_quad_layout(source, field)
    assert obstruction.value.category == "boundary_parity"
    assert len(obstruction.value.source_edge_rows) == 3


def test_curved_dual_quads_restrict_exact_source_charts() -> None:
    parameter_vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    source_points = np.column_stack(
        (
            parameter_vertices,
            parameter_vertices[:, 0] * parameter_vertices[:, 1] / 8.0,
        )
    )
    source = CellMesh.from_triangles(source_points, triangles)
    element = lagrange_element("triangle", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinate_rows: dict[tuple[Fraction, Fraction], int] = {}
    coordinate_bank: list[tuple[float, float, float]] = []
    geometry_routes = []
    exact_parameters = tuple(
        tuple(Fraction(float(value)) for value in point) for point in reference_nodes
    )
    for triangle in triangles:
        vertices = parameter_vertices[triangle]
        route = []
        for reference in exact_parameters:
            u = (
                Fraction.from_float(float(vertices[0, 0]))
                + reference[0]
                * Fraction.from_float(float(vertices[1, 0] - vertices[0, 0]))
                + reference[1]
                * Fraction.from_float(float(vertices[2, 0] - vertices[0, 0]))
            )
            v = (
                Fraction.from_float(float(vertices[0, 1]))
                + reference[0]
                * Fraction.from_float(float(vertices[1, 1] - vertices[0, 1]))
                + reference[1]
                * Fraction.from_float(float(vertices[2, 1] - vertices[0, 1]))
            )
            key = (u, v)
            if key not in coordinate_rows:
                coordinate_rows[key] = len(coordinate_bank)
                coordinate_bank.append((float(u), float(v), float(u * v / 8)))
            route.append(coordinate_rows[key])
        geometry_routes.append(route)
    block_name = source.blocks[0].name
    source_geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.asarray(geometry_routes, dtype=np.int32)},
        np.asarray(coordinate_bank, dtype=np.float64),
    )
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    assert source_validity.invalid_count == 0
    assert source_validity.unresolved_count == 0
    expected_source_status = np.full(
        (2,), CellValidityStatus.CERTIFIED_VALID, dtype=np.int32
    )
    np.testing.assert_array_equal(source_validity.source_status, expected_source_status)
    np.testing.assert_array_equal(
        source_validity.execution_status, expected_source_status
    )

    result = extract_surface_quads(source, source_geometry=source_geometry)

    coordinates = np.asarray(result.mesh.coordinates)
    np.testing.assert_allclose(
        coordinates[:, 2],
        coordinates[:, 0] * coordinates[:, 1] / 8.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result.validity.all_certified
    geometry = result.geometry
    assert geometry is not None
    assert geometry.restriction_source is not None
    for block in result.mesh.blocks:
        reference = np.asarray(reference_cell_topology("quadrilateral").vertices)
        reference_points = np.broadcast_to(
            reference, (block.cell_count, *reference.shape)
        )
        mapped = geometry.evaluate(result.mesh, block.global_ids, reference_points)
        np.testing.assert_allclose(
            mapped,
            coordinates[np.asarray(block.vertices)],
            rtol=0.0,
            atol=1.0e-15,
        )


def test_folded_lagrange2_source_map_is_invalid() -> None:
    element = lagrange_element("triangle", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    folded_controls = np.column_stack(
        (
            reference_nodes[:, 0],
            reference_nodes[:, 1] * (1.0 - 2.0 * reference_nodes[:, 0]),
        )
    )
    mesh = CellMesh.from_triangles(
        folded_controls[:3],
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    block_name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.arange(reference_nodes.shape[0], dtype=np.int32)[None, :]},
        folded_controls,
    )

    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    assert validity.invalid_count == 1
    assert validity.unresolved_count == 0
    np.testing.assert_array_equal(
        validity.status,
        np.asarray((CellValidityStatus.INVALID,), dtype=np.int32),
    )


def test_unsupported_coordinate_family_remains_unresolved_with_block_identity() -> None:
    canonical = lagrange_element("triangle", 2)
    element = FiniteElementSpec(
        "OpaqueLagrange",
        canonical.cell_kind,
        canonical.degree,
        np.asarray(canonical.reference_nodes),
        canonical.entity_dofs,
        value_spec=canonical.value_spec,
        continuity=canonical.continuity,
        representation=canonical.representation,
    )
    coordinates = np.asarray(canonical.reference_nodes, dtype=np.float64)
    mesh = CellMesh.from_triangles(
        coordinates[:3],
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    block_name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.arange(element.local_dof_count, dtype=np.int32)[None, :]},
        coordinates,
    )

    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    assert validity.invalid_count == 0
    assert validity.unresolved_count == 1
    assert validity.unsupported_block_names == (block_name,)
    np.testing.assert_array_equal(
        validity.status,
        np.asarray((CellValidityStatus.UNRESOLVED,), dtype=np.int32),
    )


@final
class _ExactCoordinateBankSourceMap(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)
    coordinate_bank: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        source: CellMesh,
        coordinate_bank: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ...],
        /,
    ) -> None:
        self.source_id = "curved-cross-field-source"
        self.source_revision = source.topology_id
        self.coordinate_bank = coordinate_bank
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-coordinate-bank-source",
                "source": self.source_id,
                "revision": self.source_revision,
                "coordinates": tuple(
                    tuple(_coordinate_coefficient_identity(value) for value in row)
                    for row in coordinate_bank
                ),
            }
        )

    def source_coordinates(
        self, /
    ) -> tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ...]:
        return self.coordinate_bank


def test_curved_cross_field_layout_preserves_one_exact_shared_patch_map() -> None:
    parameters = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    source = CellMesh.from_triangles(
        np.column_stack((parameters, np.zeros((4,), dtype=np.float64))),
        triangles,
    )
    element = lagrange_element("triangle", 4)
    nodal_points = tuple(
        node.exact_point for node in lagrange_reference_node_data(element)
    )
    if any(point is None for point in nodal_points):
        raise ValueError("The curved source requires exact Lagrange node identities.")
    reference_nodes = tuple(point for point in nodal_points if point is not None)
    if len(reference_nodes) != element.local_dof_count:
        raise ValueError("Exact Lagrange node identities must cover every local DOF.")
    coordinate_rows: dict[tuple[ExactScalar, ExactScalar], int] = {}
    coordinate_bank: list[tuple[ExactScalar, ExactScalar, ExactScalar]] = []
    routes = []
    for triangle in triangles:
        vertices = parameters[triangle]
        route = []
        for first, second in reference_nodes:
            u = (
                Fraction.from_float(float(vertices[0, 0]))
                + first * Fraction.from_float(float(vertices[1, 0] - vertices[0, 0]))
                + second * Fraction.from_float(float(vertices[2, 0] - vertices[0, 0]))
            )
            v = (
                Fraction.from_float(float(vertices[0, 1]))
                + first * Fraction.from_float(float(vertices[1, 1] - vertices[0, 1]))
                + second * Fraction.from_float(float(vertices[2, 1] - vertices[0, 1]))
            )
            key = (u, v)
            if key not in coordinate_rows:
                coordinate_rows[key] = len(coordinate_bank)
                height = u * (1 - u) * v * (1 - v) / 4
                coordinate_bank.append((u, v, height))
            route.append(coordinate_rows[key])
        routes.append(route)
    block_name = source.blocks[0].name
    exact_coordinate_bank = tuple(coordinate_bank)
    source_map = _ExactCoordinateBankSourceMap(source, exact_coordinate_bank)
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.asarray(routes, dtype=np.int32)},
        np.asarray(
            tuple(tuple(float(value) for value in row) for row in coordinate_bank),
            dtype=np.float64,
        ),
        source_map=source_map,
        source_mesh=source,
    )
    cross_field = prepare_surface_cross_field(source)
    dual = extract_surface_quads(source, source_geometry=geometry)

    result = extract_surface_quads(
        source,
        cross_field=cross_field,
        source_geometry=geometry,
    )

    assert dual.mesh.connectivity.cell_count == 6
    assert result.mesh.connectivity.cell_count == 1
    assert result.validity.all_certified
    coordinates = np.asarray(result.mesh.coordinates)
    supports = np.asarray(result.vertex_supports)
    carrier = np.asarray(source.coordinates)
    parameter_points = np.asarray(
        [np.mean(carrier[row[row >= 0], :2], axis=0) for row in supports]
    )
    expected_height = (
        parameter_points[:, 0]
        * (1.0 - parameter_points[:, 0])
        * parameter_points[:, 1]
        * (1.0 - parameter_points[:, 1])
        / 4.0
    )
    np.testing.assert_allclose(coordinates[:, 2], expected_height, rtol=0.0, atol=1.0e-14)
    geometry = result.geometry
    assert geometry is not None
    assert result.field_event is not None
    result.field_event.require_target(result.mesh, geometry)
    reference = np.asarray(reference_cell_topology("quadrilateral").vertices)
    block = result.mesh.blocks[0]
    mapped = geometry.evaluate(
        result.mesh,
        block.global_ids,
        np.broadcast_to(reference, (block.cell_count, *reference.shape)),
    )
    np.testing.assert_allclose(
        mapped,
        coordinates[np.asarray(block.vertices)],
        rtol=0.0,
        atol=1.0e-14,
    )
