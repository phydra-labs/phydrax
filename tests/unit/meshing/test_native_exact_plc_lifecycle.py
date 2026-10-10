"""Ancestry-backed decimal PLC publication, refinement, restart and adaptation."""

import subprocess
import sys
import textwrap
from collections import Counter
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreStatus,
    TetMesh3D,
    TetMeshBoundaryPolicy,
    TetMeshSourceComplex,
)
from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization import CellGeometrySpec, CellMesh, ExactPlcCellGeometrySource
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._tetra_metric import execute_tetra_metric_adaptation
from phydrax.meshing._topology_edit import TopologyEditBlock


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore unavailable"
)
_POINTS = np.asarray(
    ((0.1, 0.2, 0.3), (1.7, 0.4, 0.9), (0.3, 1.9, 0.7), (0.6, 0.5, 2.3)), dtype=np.float64
)
_FACES = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)), dtype=np.int32)
_SEGMENTS = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32)
_BOUND = 1e-12


def _determinant(corners: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    origin = corners[0]
    u, v, w = tuple(
        tuple(point[axis] - origin[axis] for axis in range(3)) for point in corners[1:]
    )
    return (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )


def _cell_determinants(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[Fraction, ...]:
    elements, routes, _ = geometry.resolve(mesh)
    source = geometry.source_coordinates()
    determinants = []
    for element, block_routes in zip(elements, routes, strict=True):
        for route in np.asarray(block_routes):
            corners = coordinate_corner_images(
                element, tuple(source[int(index)] for index in route)
            )
            if corners is None:
                raise AssertionError("PLC cell map has no exact reference-corner images.")
            determinants.append(_determinant(corners))
    return tuple(determinants)


def _declared(
    strata: np.ndarray,
    rows: np.ndarray,
    parameters: np.ndarray,
    bound: float = _BOUND,
    *,
    face_ids: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
) -> TetMeshSourceComplex:
    return TetMeshSourceComplex(
        _POINTS,
        _FACES,
        np.arange(4, dtype=np.int32),
        np.full(4, bound, dtype=np.float64),
        _SEGMENTS,
        np.arange(6, dtype=np.int32),
        np.full(6, bound, dtype=np.float64),
        strata,
        rows,
        parameters,
        np.arange(4, dtype=np.int64) if face_ids is None else face_ids,
        np.arange(6, dtype=np.int64) if segment_ids is None else segment_ids,
    )


def _published(
    *,
    face_ids: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
) -> tuple[CellMesh, CellGeometrySpec]:
    register_meshing_source_artifacts()
    original = tuple(
        tuple(Fraction(float(value)) for value in point) for point in _POINTS
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    if _determinant(original) < 0:
        cells[0, 0], cells[0, 1] = cells[0, 1], cells[0, 0]
    declared = _declared(
        np.zeros(4, dtype=np.int8),
        np.full(4, -1, dtype=np.int32),
        np.zeros((4, 2), dtype=np.float64),
        face_ids=face_ids,
        segment_ids=segment_ids,
    )
    native = TetMesh3D(
        _POINTS,
        cells,
        np.zeros(1, dtype=np.int32),
        _FACES,
        declared.face_ids[declared.face_group_rows],
        _SEGMENTS,
        declared.segment_ids[declared.segment_group_rows],
        boundary_policy="conforming",
        source=declared,
        max_vertices=256,
        max_tetrahedra=4096,
        max_scratch_bytes=1 << 26,
    )
    try:
        run = native.refine(
            sizes=np.full(4, 0.8, dtype=np.float64),
            max_insertions=200,
            work_limit=1 << 26,
        )
        assert run.status in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT)
        arrays, evidence = native.arrays(), native.source_evidence()
        assert evidence.achieved_bound > 0 and evidence.achieved_bound <= _BOUND
        live = np.flatnonzero(arrays.vertex_dimension >= 0)
        inverse = np.full(arrays.points.shape[0], -1, dtype=np.int32)
        inverse[live] = np.arange(live.size, dtype=np.int32)
        mesh = CellMesh.from_tetrahedra(arrays.points[live], inverse[arrays.tetrahedra])
        source = ExactPlcCellGeometrySource(
            _POINTS,
            _FACES,
            _SEGMENTS,
            evidence.witness_strata[live],
            evidence.witness_entities[live],
            evidence.witness_parameters[live],
            domain_source_id="tilted-decimal-plc",
            domain_source_revision="tilted-decimal-plc-authored",
            source_triangle_ids=declared.face_ids[declared.face_group_rows],
            source_triangle_bounds=declared.face_tolerances,
            source_segment_ids=declared.segment_ids[declared.segment_group_rows],
            source_segment_bounds=declared.segment_tolerances,
        )
    finally:
        native.close()
    return mesh, CellGeometrySpec.plc(mesh, source)


def _assert_original_domain(mesh: CellMesh, geometry: CellGeometrySpec) -> None:
    determinants = _cell_determinants(mesh, geometry)
    assert min(determinants) > 0
    original = tuple(
        tuple(Fraction(float(value)) for value in point) for point in _POINTS
    )
    assert sum(determinants, Fraction(0)) == abs(_determinant(original))
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"


def test_decimal_plc_source_publication_and_restart_keep_exact_rows_and_parameters() -> (
    None
):
    mesh, geometry = _published()
    _assert_original_domain(mesh, geometry)
    assert any(
        tuple(Fraction(float(value)) for value in carrier) != point
        for carrier, point in zip(
            np.asarray(mesh.coordinates), geometry.source_coordinates(), strict=True
        )
    )
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(geometry, recipe, prefix="decimal"),
        prefix="decimal",
    )
    validate_meshing_source_closure(restored)
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    assert restored.source_coordinates() == geometry.source_coordinates()
    _assert_original_domain(mesh, restored)


def test_decimal_plc_metric_adaptation_retains_original_source_and_publishes_real_witnesses() -> (
    None
):
    mesh, geometry = _published()
    outcome = execute_tetra_metric_adaptation(
        mesh,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) * 16, (mesh.coordinates.shape[0], 3, 3)
        ),
        source_geometry=geometry,
        maximum_passes=1,
        relocation=False,
        maximum_vertices=256,
        maximum_cells=4096,
        maximum_operations=200,
        maximum_work_units=1 << 26,
        maximum_scratch_bytes=1 << 26,
    )
    assert outcome.evidence.splits > 0
    assert outcome.exact_source is not None
    block = outcome.edit.blocks[0]
    if not isinstance(block, TopologyEditBlock):
        raise AssertionError(
            "Tetrahedral adaptation published a non-cell topology edit block."
        )
    target = CellMesh.from_tetrahedra(outcome.edit.coordinates, block.cells)
    target_geometry = CellGeometrySpec.plc(target, outcome.exact_source)
    _assert_original_domain(target, target_geometry)
    original = geometry.exact_source
    if not isinstance(original, ExactPlcCellGeometrySource):
        raise AssertionError("Decimal publication lost its original PLC source.")
    assert np.array_equal(
        np.asarray(outcome.exact_source.source_points).view(np.uint64),
        np.asarray(original.source_points).view(np.uint64),
    )
    assert np.array_equal(
        outcome.exact_source.source_triangles, original.source_triangles
    )
    assert np.array_equal(outcome.exact_source.source_segments, original.source_segments)
    assert np.array_equal(
        outcome.exact_source.source_triangle_ids, original.source_triangle_ids
    )
    assert np.array_equal(
        outcome.exact_source.source_segment_ids, original.source_segment_ids
    )


def test_decimal_plc_durable_archive_preserves_independent_source_banks_and_aliases(
    tmp_path: Path,
) -> None:
    mesh, geometry = _published()
    source = geometry.exact_source
    if not isinstance(source, ExactPlcCellGeometrySource):
        raise AssertionError("Decimal publication lost its exact PLC source.")
    records = (mesh, geometry, source)
    receipt = write_meshing_source_closure(tmp_path / "decimal-plc", records)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    restored_mesh, reopened, reopened_source = restored
    assert type(reopened_source) is ExactPlcCellGeometrySource
    assert reopened.exact_source is reopened_source
    assert cell_geometry_id(reopened) == cell_geometry_id(geometry)
    assert reopened_source.source_id == source.source_id
    assert reopened_source.domain_source_id == "tilted-decimal-plc"
    assert reopened_source.domain_source_revision == "tilted-decimal-plc-authored"
    assert reopened.source_coordinates() == geometry.source_coordinates()
    for name in (
        "source_points",
        "source_triangles",
        "source_segments",
        "source_triangle_ids",
        "source_triangle_bounds",
        "source_segment_ids",
        "source_segment_bounds",
        "vertex_strata",
        "vertex_rows",
        "vertex_parameters",
    ):
        original = object.__getattribute__(source, name)
        recovered = object.__getattribute__(reopened_source, name)
        assert original.dtype == recovered.dtype
        np.testing.assert_array_equal(original, recovered)
    leaves = jax.tree_util.tree_leaves(reopened_source)
    assert len(leaves) == 10 and all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert reopened_source.source_points.shape[0] == _POINTS.shape[0]
    assert restored_mesh.coordinates.shape[0] > reopened_source.source_points.shape[0]
    reopened_source.require_domain(
        _POINTS, _FACES, "tilted-decimal-plc", "tilted-decimal-plc-authored"
    )
    with pytest.raises(ValueError, match="declared domain"):
        reopened_source.require_domain(
            _POINTS, _FACES, "tilted-decimal-plc", "different-authored-revision"
        )
    _assert_original_domain(restored_mesh, reopened)


@pytest.mark.parametrize(
    "malformation", ("undeclared-row", "outside-parameters", "wrong-source-kind")
)
def test_restored_decimal_plc_source_refuses_malformed_witness_authority(
    malformation: str,
) -> None:
    _, geometry = _published()
    source = geometry.exact_source
    if not isinstance(source, ExactPlcCellGeometrySource):
        raise AssertionError("Decimal publication lost its exact PLC source.")
    match malformation:
        case "undeclared-row":
            rows = np.array(source.vertex_rows, copy=True)
            constrained = np.flatnonzero(np.asarray(source.vertex_strata) != 0)
            assert constrained.size > 0
            rows[constrained[0]] = max(
                source.source_triangles.shape[0], source.source_segments.shape[0]
            )
            corrupt = eqx.tree_at(
                lambda value: value.vertex_rows, source, jax.numpy.asarray(rows)
            )
            expected = "undeclared source rows"
        case "outside-parameters":
            parameters = np.array(source.vertex_parameters, copy=True)
            constrained = np.flatnonzero(np.asarray(source.vertex_strata) != 0)
            assert constrained.size > 0
            parameters[constrained[0], 0] = 2.0
            corrupt = eqx.tree_at(
                lambda value: value.vertex_parameters,
                source,
                jax.numpy.asarray(parameters),
            )
            expected = "closed source entity"
        case "wrong-source-kind":
            corrupt = eqx.tree_at(
                lambda value: value.source_points, source, "not-source-coordinates"
            )
            expected = None
        case _:
            raise AssertionError(f"Uncovered source malformation {malformation!r}.")
    recipe = model_structure_recipe(corrupt)
    with pytest.raises((TypeError, ValueError), match=expected):
        restored = model_from_array_recipe(
            recipe,
            model_recipe_array_values(corrupt, recipe, prefix="invalid-plc"),
            prefix="invalid-plc",
        )
        validate_meshing_source_closure(restored)


def test_decimal_plc_transient_preparation_cannot_be_archived(tmp_path: Path) -> None:
    mesh, geometry = _published()
    source = geometry.exact_source
    if not isinstance(source, ExactPlcCellGeometrySource):
        raise AssertionError("Decimal publication lost its exact PLC source.")
    proof = source.prepare(mesh.coordinates)
    with pytest.raises(TypeError, match="nonowning source type PreparedExactPlcGeometry"):
        write_meshing_source_closure(tmp_path / "transient-plc-proof", proof)


@pytest.mark.parametrize(
    "route",
    (
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
    ),
)
def test_decimal_plc_nested_adaptation_and_coarsening_preserve_original_physical_source(
    route: phx.meshing.MeshAdaptationRoute,
    tmp_path: Path,
) -> None:
    mesh, geometry = _published()
    source = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        route,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    identifier = np.asarray(mesh.blocks[0].global_ids)[0]
    refinement = phx.meshing.MarkedMeshAdaptation(
        np.asarray([identifier], dtype=np.int64)
    )
    refined = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(source, refinement, policy=policy),
    )
    assert refined.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    _assert_original_domain(refined.target.mesh, refined.target.geometry)
    refined_source = refined.target.geometry.exact_source
    if not isinstance(refined_source, ExactPlcCellGeometrySource):
        raise AssertionError("PLC refinement lost its exact source.")
    original_source = geometry.exact_source
    if not isinstance(original_source, ExactPlcCellGeometrySource):
        raise AssertionError("The source geometry lost its exact PLC owner.")
    assert refined_source.domain_source_id == original_source.domain_source_id
    refined_coordinates = refined.target.geometry.source_coordinates()
    assert min(_cell_determinants(refined.target.mesh, refined.target.geometry)) > 0
    receipt = write_meshing_source_closure(
        tmp_path / "refined-plc",
        (refined.target, refined.hierarchy, refined_source),
    )
    reopened_target, reopened_hierarchy, reopened_source = read_meshing_source_closure(
        receipt.path,
        expected_content_id=receipt.content_id,
    )
    assert reopened_target.geometry.exact_source is reopened_source
    assert reopened_source.source_id == refined_source.source_id
    assert reopened_target.geometry.source_coordinates() == refined_coordinates
    cold_receipt = write_meshing_source_closure(
        tmp_path / "cold-original-plc",
        (source, refined.target, refined.hierarchy),
    )
    cold_script = textwrap.dedent("""
        import sys
        import numpy as np
        import phydrax as phx
        from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
        original, fine, hierarchy = read_meshing_source_closure(
            sys.argv[1], expected_content_id=sys.argv[2])
        children = np.concatenate([np.asarray(block.global_ids) for block in fine.mesh.blocks])
        original_ids = np.concatenate([np.asarray(block.global_ids) for block in original.mesh.blocks])
        marks = children[~np.isin(children, original_ids)]
        policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute(sys.argv[3]),
            compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT)
        result = phx.meshing.execute_mesh_adaptation(phx.meshing.prepare_mesh_adaptation(
            fine, phx.meshing.MarkedMeshAdaptation((), marks, hierarchy=hierarchy), policy=policy))
        assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
        np.testing.assert_array_equal(result.target.mesh.vertex_global_ids, original.mesh.vertex_global_ids)
        np.testing.assert_array_equal(result.target.mesh.coordinates, original.mesh.coordinates)
        for before, after in zip(original.mesh.blocks, result.target.mesh.blocks, strict=True):
            np.testing.assert_array_equal(after.global_ids, before.global_ids)
            np.testing.assert_array_equal(after.vertices, before.vertices)
        assert result.target.geometry.source_coordinates() == original.geometry.source_coordinates()
        for name in ('source_points', 'source_triangles', 'source_segments',
                     'source_triangle_ids', 'source_triangle_bounds', 'source_segment_ids',
                     'source_segment_bounds', 'vertex_strata', 'vertex_rows', 'vertex_parameters'):
            before = getattr(original.geometry.exact_source, name)
            after = getattr(result.target.geometry.exact_source, name)
            assert after.dtype == before.dtype
            np.testing.assert_array_equal(after, before)
    """)
    subprocess.run(
        (
            sys.executable,
            "-c",
            cold_script,
            str(cold_receipt.path),
            cold_receipt.content_id,
            route.value,
        ),
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )
    children = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in refined.target.mesh.blocks
        ]
    )
    original = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    new_children = children[~np.isin(children, original)]
    coarsening = phx.meshing.MarkedMeshAdaptation(
        np.empty((0,), dtype=np.int64),
        new_children,
        hierarchy=reopened_hierarchy,
    )
    coarsened = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(reopened_target, coarsening, policy=policy),
    )
    assert coarsened.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert sum(block.cell_count for block in coarsened.target.mesh.blocks) < sum(
        block.cell_count for block in refined.target.mesh.blocks
    )
    coarsening_witnesses = coarsened.coarsening_witnesses
    if coarsening_witnesses is None:
        raise AssertionError("PLC coarsening published no inverse reference witnesses.")
    # Exact coefficient-action publication carries coarsening charts directly
    # even when no separately reconstructed geometry transition is necessary.
    assert set(np.asarray(coarsening_witnesses.fine_cell_ids).tolist()) == set(
        children.tolist()
    )
    assert set(np.asarray(coarsening_witnesses.coarse_cell_ids).tolist()) == set(
        np.concatenate(
            [np.asarray(block.global_ids) for block in coarsened.target.mesh.blocks]
        ).tolist()
    )
    coarsened_transition = coarsened.geometry_transition
    if coarsened_transition is not None:
        assert coarsened_transition.evidence.measure_exact
        assert coarsened_transition.evidence.coverage_defect == 0.0
    coarsened_source = coarsened.target.geometry.exact_source
    if not isinstance(coarsened_source, ExactPlcCellGeometrySource):
        raise AssertionError("PLC coarsening lost its exact source.")
    np.testing.assert_array_equal(
        coarsened_source.source_points, original_source.source_points
    )
    np.testing.assert_array_equal(
        coarsened_source.source_triangles, original_source.source_triangles
    )
    coarsened_coordinates = coarsened.target.geometry.source_coordinates()
    assert min(_cell_determinants(coarsened.target.mesh, coarsened.target.geometry)) > 0
    np.testing.assert_array_equal(
        coarsened.target.mesh.vertex_global_ids, mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(coarsened.target.mesh.coordinates, mesh.coordinates)
    for original_block, restored_block in zip(
        mesh.blocks, coarsened.target.mesh.blocks, strict=True
    ):
        np.testing.assert_array_equal(
            restored_block.global_ids, original_block.global_ids
        )
        np.testing.assert_array_equal(restored_block.vertices, original_block.vertices)
    assert coarsened_coordinates == geometry.source_coordinates()
    for name in (
        "source_points",
        "source_triangles",
        "source_segments",
        "source_triangle_ids",
        "source_triangle_bounds",
        "source_segment_ids",
        "source_segment_bounds",
        "vertex_strata",
        "vertex_rows",
        "vertex_parameters",
    ):
        original_bank = object.__getattribute__(original_source, name)
        restored_bank = object.__getattribute__(coarsened_source, name)
        assert restored_bank.dtype == original_bank.dtype
        np.testing.assert_array_equal(restored_bank, original_bank)


def _original_native(
    bound: float, boundary_policy: TetMeshBoundaryPolicy = "conforming"
) -> TetMesh3D:
    original = tuple(
        tuple(Fraction(float(value)) for value in point) for point in _POINTS
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    if _determinant(original) < 0:
        cells[0, 0], cells[0, 1] = cells[0, 1], cells[0, 0]
    declared = _declared(
        np.zeros(4, dtype=np.int8),
        np.full(4, -1, dtype=np.int32),
        np.zeros((4, 2), dtype=np.float64),
        bound,
    )
    return TetMesh3D(
        _POINTS,
        cells,
        np.zeros(1, dtype=np.int32),
        _FACES,
        declared.face_ids[declared.face_group_rows],
        _SEGMENTS,
        declared.segment_ids[declared.segment_group_rows],
        source=declared,
        boundary_policy=boundary_policy,
        max_vertices=32,
        max_tetrahedra=128,
        max_scratch_bytes=1 << 24,
    )


def test_public_decimal_edge_split_commits_its_real_nearest_float_witness() -> None:
    native = _original_native(_BOUND)
    try:
        initial = native.source_evidence()
        assert np.all(initial.witness_strata > 0)
        assert np.all(initial.witness_entities >= 0)
        assert np.all(initial.witness_deviations == 0.0)
        constructed = native.edge_split_point(0, 1, work_limit=1 << 20)
        assert constructed is not None
        position, parameter = constructed.position, constructed.parameter
        inserted = native.split_edge(
            0, 1, position, source_fraction=parameter, work_limit=1 << 20
        )
        assert inserted is not None
        arrays, witnesses = native.arrays(), native.source_evidence()
        assert witnesses.witness_strata[inserted] == 1
        row = witnesses.witness_entities[inserted]
        first, second = (
            tuple(Fraction(float(value)) for value in _POINTS[index])
            for index in _SEGMENTS[row]
        )
        t = Fraction(float(witnesses.witness_parameters[inserted, 0]))
        exact = tuple(a + t * (b - a) for a, b in zip(first, second, strict=True))
        carrier = tuple(Fraction(float(value)) for value in arrays.points[inserted])
        assert 0 < t < 1 and exact != carrier
        assert tuple(float(value) for value in exact) == tuple(arrays.points[inserted])
        squared = sum(
            ((a - b) ** 2 for a, b in zip(exact, carrier, strict=True)), Fraction(0)
        )
        achieved = Fraction(float(witnesses.witness_deviations[inserted]))
        assert 0 < squared <= achieved**2 <= Fraction(_BOUND) ** 2
        assert witnesses.achieved_bound == witnesses.witness_deviations[inserted]
        assert np.all(arrays.tetrahedron_regions == 0)
    finally:
        native.close()


@pytest.mark.parametrize(
    ("bound", "boundary_policy"), ((0.0, "conforming"), (_BOUND, "fixed"))
)
def test_public_decimal_edge_split_refuses_zero_bound_and_fixed_source_without_mutation(
    bound: float,
    boundary_policy: TetMeshBoundaryPolicy,
) -> None:
    native = _original_native(bound, boundary_policy)
    try:
        before = native.arrays()
        assert native.edge_split_point(0, 1, work_limit=1 << 20) is None
        after = native.arrays()
        for first, second in (
            (before.points, after.points),
            (before.tetrahedra, after.tetrahedra),
            (before.faces, after.faces),
            (before.segments, after.segments),
            (before.tetrahedron_regions, after.tetrahedron_regions),
        ):
            np.testing.assert_array_equal(first, second)
    finally:
        native.close()


def test_prepared_decimal_split_publishes_only_its_inspected_source_cavity() -> None:
    native = _original_native(_BOUND)
    try:
        before = native.arrays()
        construction = native.edge_split_point(0, 1, work_limit=1 << 20)
        assert construction is not None
        position, parameter = construction.position, construction.parameter
        exact = tuple(
            Fraction(float(a))
            + Fraction(parameter) * (Fraction(float(b)) - Fraction(float(a)))
            for a, b in zip(_POINTS[0], _POINTS[1], strict=True)
        )
        assert exact != tuple(Fraction(float(value)) for value in position)
        plan = native.inspect_edge_insertion(
            0,
            1,
            position,
            position,
            source_fraction=parameter,
            maximum_cavity_cells=128,
            work_limit=1 << 20,
        )
        assert plan is not None
        inspected = native.arrays()
        np.testing.assert_array_equal(inspected.points, before.points)
        np.testing.assert_array_equal(inspected.tetrahedra, before.tetrahedra)
        inserted = native.commit_edge_insertion(plan, work_limit=1 << 20)
        assert inserted is not None
        after, evidence = native.arrays(), native.source_evidence()
        remaining = Counter(tuple(sorted(row)) for row in before.tetrahedra.tolist())
        remaining.subtract(tuple(sorted(row)) for row in plan.removed.tolist())
        assert all(count >= 0 for count in remaining.values())
        proposed = np.where(plan.proposed < 0, inserted, plan.proposed)
        remaining.update(tuple(sorted(row)) for row in proposed.tolist())
        assert +remaining == Counter(
            tuple(sorted(row)) for row in after.tetrahedra.tolist()
        )
        np.testing.assert_array_equal(after.points[: len(_POINTS)], _POINTS)
        np.testing.assert_array_equal(after.points[inserted], position)
        assert evidence.witness_strata[inserted] == 1
        assert evidence.witness_entities[inserted] == 0
        assert evidence.witness_parameters[inserted, 0] == parameter
        squared = sum(
            ((a - Fraction(float(b))) ** 2 for a, b in zip(exact, position, strict=True)),
            Fraction(0),
        )
        achieved = Fraction(float(evidence.witness_deviations[inserted]))
        assert 0 < squared <= achieved**2 <= Fraction(_BOUND) ** 2
        assert np.all(after.tetrahedron_regions == 0)
    finally:
        native.close()


@pytest.mark.parametrize(
    "refusal",
    ("missing-witness", "wrong-carrier", "zero-bound", "fixed-source"),
)
def test_prepared_decimal_split_refuses_unauthorized_source_carriers_atomically(
    refusal: str,
) -> None:
    author = _original_native(_BOUND)
    try:
        construction = author.edge_split_point(0, 1, work_limit=1 << 20)
        assert construction is not None
        position, parameter = construction.position.copy(), construction.parameter
    finally:
        author.close()
    native = _original_native(
        0.0 if refusal == "zero-bound" else _BOUND,
        "fixed" if refusal == "fixed-source" else "conforming",
    )
    try:
        before = native.arrays()
        if refusal == "wrong-carrier":
            position[0] = np.nextafter(position[0], np.inf)
        assert (
            native.inspect_edge_insertion(
                0,
                1,
                position,
                position,
                source_fraction=None if refusal == "missing-witness" else parameter,
                maximum_cavity_cells=128,
                work_limit=1 << 20,
            )
            is None
        )
        after = native.arrays()
        for old, current in (
            (before.points, after.points),
            (before.tetrahedra, after.tetrahedra),
            (before.faces, after.faces),
            (before.segments, after.segments),
            (before.tetrahedron_regions, after.tetrahedron_regions),
        ):
            np.testing.assert_array_equal(current, old)
    finally:
        native.close()


def test_native_provider_generates_decimal_plc_on_original_exact_source_geometry() -> (
    None
):
    meshing = phx.meshing
    original = tuple(
        tuple(Fraction(float(value)) for value in point) for point in _POINTS
    )
    regions = np.empty((4, 2), dtype=np.int32)
    for row, face in enumerate(_FACES):
        opposite = next(index for index in range(4) if index not in face)
        corners = tuple(original[index] for index in (*face, opposite))
        regions[row] = (0, -1) if _determinant(corners) > 0 else (-1, 0)
    complex_ = meshing.PiecewiseLinearComplex(
        _POINTS,
        tuple(tuple(int(index) for index in face) for face in _FACES),
        np.arange(4, dtype=np.int32),
        regions,
        ("solid",),
        segments=_SEGMENTS,
    )
    source = meshing.NativePlcSource(
        complex_, "decimal-provider-plc", "decimal-provider-authored"
    )
    facets = meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        "decimal-provider-facets",
        np.arange(4, dtype=np.int64),
    )
    edges = meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        1,
        "decimal-provider-segments",
        np.arange(6, dtype=np.int64),
    )
    specification = meshing.VolumeMeshingSpec(
        meshing.CellMeshingTarget(
            3, 3, meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        facets,
        meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            meshing.UniformSizeControl(
                facets, 0.8, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            meshing.ProtectedFeature(
                facets, meshing.FeatureKind.SURFACE, maximum_deviation=_BOUND
            ),
            meshing.ProtectedFeature(
                edges, meshing.FeatureKind.CURVE, maximum_deviation=_BOUND
            ),
        ),
        limits=meshing.MeshingLimits(
            maximum_vertices=256, maximum_cells=4096, maximum_work_units=1 << 26
        ),
    )
    schedule = meshing.NativeVolumeSchedule(
        improvement_passes=1, minimum_dihedral_degrees=0.0
    )
    produced = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions("plc_tetrahedral", volume_schedule=schedule),
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    assert produced.audit.passed
    assert produced.certification is not None and produced.certification.passed
    exact_source = produced.geometry.exact_source
    assert isinstance(exact_source, ExactPlcCellGeometrySource)
    assert exact_source.domain_source_id == source.source_id
    assert exact_source.domain_source_revision == source.source_revision
    assert exact_source.prepare(produced.mesh.coordinates).maximum_rounding_error > 0
    _assert_original_domain(produced.mesh, produced.geometry)


def test_sparse_int64_source_ids_survive_real_bounded_refinement_and_metric_adaptation() -> (
    None
):
    face_ids = np.asarray(
        (2**61 + 7, 2**61 + 37, 2**61 + 101, 2**61 + 1009), dtype=np.int64
    )
    segment_ids = np.asarray(
        (2**62 + 3, 2**62 + 19, 2**62 + 71, 2**62 + 307, 2**62 + 401, 2**62 + 10007),
        dtype=np.int64,
    )
    mesh, geometry = _published(face_ids=face_ids, segment_ids=segment_ids)
    original = geometry.exact_source
    assert isinstance(original, ExactPlcCellGeometrySource)
    assert original.prepare(mesh.coordinates).maximum_rounding_error > 0
    _assert_original_domain(mesh, geometry)
    np.testing.assert_array_equal(original.source_triangle_ids, face_ids)
    np.testing.assert_array_equal(original.source_segment_ids, segment_ids)
    outcome = execute_tetra_metric_adaptation(
        mesh,
        np.broadcast_to(
            np.eye(3, dtype=np.float64) * 16, (mesh.coordinates.shape[0], 3, 3)
        ),
        source_geometry=geometry,
        maximum_passes=1,
        relocation=False,
        maximum_vertices=256,
        maximum_cells=4096,
        maximum_operations=200,
        maximum_work_units=1 << 26,
        maximum_scratch_bytes=1 << 26,
    )
    assert outcome.evidence.splits > 0
    assert isinstance(outcome.exact_source, ExactPlcCellGeometrySource)
    np.testing.assert_array_equal(outcome.exact_source.source_triangle_ids, face_ids)
    np.testing.assert_array_equal(outcome.exact_source.source_segment_ids, segment_ids)
    assert outcome.exact_source.source_triangle_ids.dtype == np.dtype(np.int64)
    assert outcome.exact_source.source_segment_ids.dtype == np.dtype(np.int64)
    block = outcome.edit.blocks[0]
    if not isinstance(block, TopologyEditBlock):
        raise AssertionError(
            "Tetrahedral adaptation published a non-cell topology edit block."
        )
    target = CellMesh.from_tetrahedra(outcome.edit.coordinates, block.cells)
    _assert_original_domain(target, CellGeometrySpec.plc(target, outcome.exact_source))


@pytest.mark.parametrize("ambient", (False, True), ids=("standalone", "ambient"))
def test_exact_plc_metric_tiny_original_scratch_refuses_without_changing_live_source(
    ambient: bool,
) -> None:
    from contextlib import nullcontext

    from phydrax._meshcore import MeshcoreError, NativeExecutionBudget
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory

    mesh, geometry = _published()
    source = geometry.exact_source
    if not isinstance(source, ExactPlcCellGeometrySource):
        raise AssertionError("Decimal publication lost its canonical PLC authority.")
    identity = source.source_id
    bank = geometry.source_coordinates()
    coordinates = np.asarray(mesh.coordinates).copy()
    cells = tuple(np.asarray(block.vertices).copy() for block in mesh.blocks)
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64) * 16, (mesh.coordinates.shape[0], 3, 3)
    )
    native = (
        NativeExecutionBudget(
            max_work=1 << 26,
            max_geometry_queries=1 << 26,
            max_cavity_cells=1 << 20,
            max_scratch_bytes=32,
            max_wall_seconds=np.inf,
        )
        if ambient
        else None
    )
    with (
        pytest.raises(MeshcoreError) if native is not None else nullcontext(),
        nullcontext() if native is None else native,
    ):
        with pytest.raises(MeshingFailure) as refusal:
            execute_tetra_metric_adaptation(
                mesh,
                metric,
                source_geometry=geometry,
                maximum_passes=1,
                relocation=False,
                maximum_vertices=256,
                maximum_cells=4096,
                maximum_operations=200,
                maximum_work_units=1 << 26,
                maximum_location_pairs=1 << 26,
                maximum_cavity_cells=1 << 20,
                maximum_cavity_work=1 << 26,
                maximum_scratch_bytes=(1 << 26) if ambient else 32,
            )
        assert refusal.value.category == MeshingFailureCategory.RESOURCE_EXHAUSTED
    if native is not None:
        assert native.evidence is not None
        assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
        assert native.evidence.host_storage_live_bytes_upper == 0
        assert native.evidence.host_storage_peak_bytes_upper == 0
    assert source.source_id == identity
    assert geometry.source_coordinates() == bank
    np.testing.assert_array_equal(mesh.coordinates, coordinates)
    for block, original in zip(mesh.blocks, cells, strict=True):
        np.testing.assert_array_equal(block.vertices, original)
