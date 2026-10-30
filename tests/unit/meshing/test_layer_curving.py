from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._coordinate_enclosure import evaluate_expression
from phydrax.discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
)
from phydrax.discretization._nested_reference import nested_cell_reference_maps
from phydrax.discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.discretization.fem._reference import lagrange_reference_node_data
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._source_identity import source_entity_id
from phydrax.meshing._adaptation import (
    _identity_lineage,
    _identity_reference_maps_for_transition,
)
from phydrax.meshing._curving import _straight_geometry, HighOrderCurvingPolicy
from phydrax.meshing._layer_curving import (
    _compose_mapped_reference_maps,
    _implicit_source_faces,
    _source_parameter,
    _SourceFace,
    curve_layer_core_construction,
    LayerCurvingFailure,
    LayerCurvingStatus,
    renew_layer_curving_source_evidence,
    require_layer_core_construction,
)
from phydrax.meshing._layer_curving_records import (
    _fraction3,
    _prepare_barycentric_solver_3d,
    _reference_face_dofs,
)
from phydrax.meshing._mixed_adaptation import MixedCoarseningMap
from tests._support.implicit_layer_curving_fixtures import (
    make_implicit_layer_curving_archive_case,
)
from tests._support.layer_curving_fixtures import (
    _layer_mesh,
    _mapped_cube_source_for_layer,
    _mapped_cube_surface_charts,
    _mapped_layer_core_construction_for_fixture,
    make_layer_curving_archive_case,
    make_mapped_layer_curving_archive_case,
    make_restricted_layer_curving_archive_case,
)
from tests.unit.geometry.test_implicit_source_charts import _adaptive_source


def test_exact_layer_reference_map_uses_noncorner_local_coordinates() -> None:
    face = _SourceFace(
        0,
        50,
        0,
        "surface-face",
        (),
        900,
        (0, 900),
        0,
        0,
        50,
        0,
        12,
        (0, 1, 2),
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
        ),
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    face_solver = _prepare_barycentric_solver_3d(face.target_reference_corners)

    assert _source_parameter(
        (Fraction(1, 3), Fraction(1, 3), Fraction(0)), face, face_solver
    ) == (Fraction(1, 3), Fraction(1, 3))

    tetrahedron = lagrange_element("tetrahedron", 3)
    nodal_data = lagrange_reference_node_data(tetrahedron)
    face_node_dofs = _reference_face_dofs(tetrahedron, face.target_reference_corners)
    algebraic_points: list[tuple[ExactScalar, ExactScalar, ExactScalar]] = []
    for local_dof, _ in face_node_dofs:
        point = nodal_data[local_dof].exact_point
        if point is None:
            continue
        if len(point) != 3:
            raise AssertionError(
                "A tetrahedral Lagrange reference point has wrong dimension."
            )
        candidate = (point[0], point[1], point[2])
        if any(isinstance(value, RadicalElement) for value in candidate):
            algebraic_points.append(candidate)
    algebraic_point = next(iter(algebraic_points), None)
    if algebraic_point is None:
        raise AssertionError("Expected an algebraic tetrahedral face node.")
    assert _source_parameter(algebraic_point, face, face_solver) == (
        algebraic_point[0],
        algebraic_point[1],
    )


def test_general_simplex_lagrange_preserves_qualified_cell_kind_and_nodes() -> None:
    topology = reference_cell_topology("simplex:3")
    linear = lagrange_element("simplex:3", 1)
    assert linear.cell_kind == "simplex:3"
    np.testing.assert_array_equal(
        np.asarray(linear.reference_nodes),
        np.asarray(topology.vertices, dtype=np.float64),
    )

    quadratic = lagrange_element("simplex:3", 2)
    values, _ = quadratic.tabulate(quadratic.reference_nodes)
    np.testing.assert_allclose(
        np.asarray(values),
        np.eye(quadratic.local_dof_count),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_high_order_layer_face_nodes_include_vertex_and_edge_dofs() -> None:
    element = lagrange_element("prism", 2)
    node_data = lagrange_reference_node_data(element)
    face_vertices = reference_cell_topology("prism").entities[2][0]
    corner_dofs = tuple(
        element.entity_dofs[0][int(vertex)][0] for vertex in face_vertices
    )
    face_corners_list = []
    for dof in corner_dofs:
        exact_point = node_data[dof].exact_point
        assert exact_point is not None
        face_corners_list.append(_fraction3(exact_point))
    face_corners = tuple(face_corners_list)
    face_nodes = _reference_face_dofs(element, face_corners)
    face_dofs = {local_dof for local_dof, _ in face_nodes}

    assert len(face_dofs) == 6
    assert set(corner_dofs) <= face_dofs
    assert len(face_dofs) > len(element.entity_dofs[2][0])


@pytest.mark.parametrize("cell_kind", ("tetrahedron", "prism"))
def test_layer_face_nodes_use_exact_lagrange_points_on_tet_and_prism_facets(
    cell_kind: str,
) -> None:
    element = lagrange_element(cell_kind, 3)
    node_data = lagrange_reference_node_data(element)
    linear_node_data = lagrange_reference_node_data(lagrange_element(cell_kind, 1))
    topology = reference_cell_topology(cell_kind)
    gll_lower = RadicalElement(Fraction(1, 2)) - RadicalElement.sqrt(Fraction(5)) / 10
    gll_upper = RadicalElement(Fraction(1, 2)) + RadicalElement.sqrt(Fraction(5)) / 10
    found_one_third_face_node = False
    found_gll_face_node = False
    for face_index, face in enumerate(topology.entities[2]):
        if len(face) != 3:
            continue
        face_corners_list = []
        for vertex_index in face:
            exact_point = linear_node_data[vertex_index].exact_point
            assert exact_point is not None
            face_corners_list.append(_fraction3(exact_point))
        face_corners = tuple(face_corners_list)
        face_nodes = dict(_reference_face_dofs(element, face_corners))
        expected = {
            int(local_dof)
            for vertex in face
            for local_dof in element.entity_dofs[0][vertex]
        }
        face_vertex_set = set(face)
        for edge_index, edge in enumerate(topology.entities[1]):
            if set(edge) <= face_vertex_set:
                expected.update(element.entity_dofs[1][edge_index])
        expected.update(element.entity_dofs[2][face_index])

        assert set(face_nodes) == expected
        assert (
            dict(_reference_face_dofs(element, tuple(reversed(face_corners))))
            == face_nodes
        )
        for local_dof, reference in face_nodes.items():
            exact_point = node_data[local_dof].exact_point
            assert exact_point is not None
            assert reference == exact_point
            if Fraction(1, 3) in reference:
                found_one_third_face_node = True
            if any(
                isinstance(value, RadicalElement) and value in (gll_lower, gll_upper)
                for value in reference
            ):
                found_gll_face_node = True
    assert found_one_third_face_node
    assert found_gll_face_node


def test_mapped_root_face_uv_preserves_algebraic_reference_points() -> None:
    source, domain, _ = _mapped_cube_source_for_layer()
    query = source.boundary_query()
    row = 0
    root_face = int(query.polygon_root_face_global_ids[row])
    root_cell = int(query.polygon_root_cell_global_ids[row])
    local_face = int(query.polygon_root_local_face_indices[row])
    topology = reference_cell_topology(domain.reference_mesh.blocks[0].cell_kind)
    linear_node_data = lagrange_reference_node_data(
        lagrange_element(domain.reference_mesh.blocks[0].cell_kind, 1)
    )
    face_vertices = topology.entities[2][local_face]
    corners_list = []
    for vertex_index in face_vertices[:3]:
        exact_point = linear_node_data[vertex_index].exact_point
        assert exact_point is not None
        corners_list.append(_fraction3(exact_point))
    corners = tuple(corners_list)
    weight_0 = RadicalElement(Fraction(1, 2)) - RadicalElement.sqrt(Fraction(5)) / 10
    weight_1 = RadicalElement(Fraction(1, 2)) + RadicalElement.sqrt(Fraction(5)) / 10
    weights = (weight_0, weight_1, Fraction(0))
    root_point = tuple(
        sum((weights[index] * corners[index][axis] for index in range(3)), Fraction(0))
        for axis in range(3)
    )
    corner_uv = query.root_face_uv(root_face, root_cell, local_face, corners)
    expected_uv = tuple(
        sum((weights[index] * corner_uv[index][axis] for index in range(3)), Fraction(0))
        for axis in range(2)
    )
    actual_uv = query.root_face_uv(root_face, root_cell, local_face, (root_point,))[0]

    assert any(isinstance(value, RadicalElement) for value in root_point)
    assert actual_uv == expected_uv

    float_point = tuple(float(value) for value in root_point)
    float_uv = query.root_face_uv(
        root_face,
        root_cell,
        local_face,
        np.asarray((float_point,), dtype=np.float64),
    )[0]
    exact_float_point = tuple(Fraction.from_float(value) for value in float_point)
    expected_float_uv = query.root_face_uv(
        root_face, root_cell, local_face, (exact_float_point,)
    )[0]
    assert float_uv == expected_float_uv


def test_mapped_layer_transition_composes_exact_target_to_root_maps() -> None:
    linear_node_data = lagrange_reference_node_data(lagrange_element("hexahedron", 1))
    corners_list = []
    for node in linear_node_data:
        assert node.exact_point is not None
        corners_list.append(_fraction3(node.exact_point))
    corners = tuple(corners_list)
    exact_zero = (0, 0, 0)
    identity: tuple[Mapping[tuple[int, ...], Fraction], ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    child_to_parent: tuple[Mapping[tuple[int, ...], Fraction], ...] = (
        {exact_zero: Fraction(1, 4), (1, 0, 0): Fraction(1, 2)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )

    def bank(
        target_cell_id: int,
        source_cell_id: int,
        arguments: tuple[Mapping[tuple[int, ...], Fraction], ...],
    ) -> ReferenceCellMapWitnesses:
        def exact_fraction(value: object) -> Fraction:
            if not isinstance(value, Fraction):
                raise ValueError(
                    "Affine test reference maps must retain rational values."
                )
            return value

        exact_vertices = tuple(
            tuple(
                exact_fraction(evaluate_expression(dict(expression), point))
                for expression in arguments
            )
            for point in corners
        )
        return ReferenceCellMapWitnesses(
            np.asarray([target_cell_id], dtype=np.int64),
            np.asarray([source_cell_id], dtype=np.int64),
            "hexahedron",
            "hexahedron",
            np.asarray(
                [[[float(value) for value in point] for point in exact_vertices]],
                dtype=np.float64,
            ),
            (exact_vertices,),
            freeze_reference_chart_arguments((arguments,)),
        )

    composed = _compose_mapped_reference_maps(
        (bank(30, 20, child_to_parent),),
        (bank(20, 10, identity),),
    )
    assert len(composed) == 1
    (mapping,) = nested_cell_reference_maps(composed[0])
    assert (mapping.target_cell_id, mapping.source_cell_id) == (30, 10)
    assert mapping.evaluate((Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))) == (
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
    )
    mixed_coarsening = MixedCoarseningMap(
        30,
        "cells",
        "hexahedron",
        np.asarray((20,), dtype=np.int64),
        (corners,),
    )
    mixed_composed = _compose_mapped_reference_maps(
        (),
        (bank(20, 10, identity),),
        mixed_coarsening_maps=(mixed_coarsening,),
    )
    assert len(mixed_composed) == 1
    (mixed_mapping,) = nested_cell_reference_maps(mixed_composed[0])
    assert (mixed_mapping.target_cell_id, mixed_mapping.source_cell_id) == (30, 10)
    assert mixed_mapping.evaluate((Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))) == (
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
    )


@pytest.mark.parametrize("degree", (2, 3))
def test_parametric_layer_core_certifies_mixed_prism_tet_source_maps(
    degree: int,
) -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    mapped_source, _, _, _, wall, layers, _, construction = (
        _mapped_layer_core_construction_for_fixture()
    )
    domain, charts = _mapped_cube_surface_charts(
        wall, mapped_source.source_id, mapped_source.source_revision
    )
    result = curve_layer_core_construction(
        construction,
        layers,
        charts,
        policy=HighOrderCurvingPolicy(degree=degree, relaxation_rounds=0),
        fidelity_tolerance=0.0,
    )

    assert result.status is LayerCurvingStatus.CURVED, result.failure
    assert require_layer_core_construction(result) is result.construction
    assert result.candidate_geometry is not None
    assert result.validity is not None and result.validity.all_certified
    assert result.embedding is not None and result.embedding.status == "certified"
    recomputed_embedding = certify_global_embedding(
        result.construction.mesh, result.candidate_geometry, result.validity
    )
    assert recomputed_embedding.status == "certified", recomputed_embedding.findings
    assert recomputed_embedding.certificate_id == result.embedding.certificate_id
    assert tuple(
        (witness.theorem, witness.entity_ids, witness.exact_values)
        for witness in result.embedding.witnesses
    ) == tuple(
        (witness.theorem, witness.entity_ids, witness.exact_values)
        for witness in recomputed_embedding.witnesses
    )
    assert any(
        witness.theorem == "global_degree_one_injectivity"
        for witness in recomputed_embedding.witnesses
    )
    assert result.chart_witness is not None
    assert (
        result.source_fidelity is not None
        and result.source_fidelity.status == "certified"
    )
    assert result.source_evidence is not None
    evidence = result.source_evidence
    assert evidence.chart_column_identity.row_count == len(
        result.chart_witness.source_chart_ids
    )
    assert result.projection_trace is None
    assert evidence.source_surface_charts is charts
    cell_families = {
        (topology.family, topology.dimension)
        for block in result.construction.mesh.blocks
        for topology in (reference_cell_topology(block.cell_kind),)
    }
    assert {("prism", 3), ("tetrahedron", 3)} <= cell_families
    assert len(evidence.source_piece_ids) == evidence.chart_column_identity.row_count
    assert all(len(piece) == 2 for piece in evidence.source_piece_ids)
    assert set(evidence.source_entity_ids) == {
        domain.entity_id(2, patch) for patch in range(6)
    }
    column_identity = evidence.column_identity
    assert set(column_identity.source_wall_cell_ids) <= set(
        np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    )
    assert all(value >= 0 for value in np.asarray(column_identity.front_row_ids))
    assert all(column_identity.front_row_identities)
    assert all(column_identity.source_column_ids)
    assert all(
        len(columns) == len(names) == len(roots)
        for columns, names, roots in zip(
            column_identity.source_column_ids,
            column_identity.source_column_identities,
            column_identity.source_column_root_vertex_ids,
            strict=True,
        )
    )
    np.testing.assert_array_equal(
        np.asarray(result.candidate_geometry.coordinates)[
            : construction.mesh.coordinates.shape[0]
        ],
        np.asarray(construction.mesh.coordinates),
    )
    straight = _straight_geometry(construction.mesh, degree)
    np.testing.assert_array_equal(
        np.asarray(result.candidate_geometry.coordinates),
        np.asarray(straight.coordinates),
    )
    assert evidence.layer_result_id == layers.result_id
    assert evidence.layer_evidence.evidence_id == layers.evidence.evidence_id
    evidence.require_current(result.construction.mesh, result.construction.geometry)


def test_adaptive_implicit_layer_builds_and_revalidates_source_evidence() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    result = make_implicit_layer_curving_archive_case()
    assert result.status is LayerCurvingStatus.CURVED
    assert require_layer_core_construction(result) is result.construction
    assert result.candidate_geometry is not None
    assert result.validity is not None and result.validity.all_certified
    assert result.embedding is not None and result.embedding.status == "certified"
    evidence = result.source_evidence
    assert evidence is not None and evidence.implicit_root_trace is not None
    assert evidence.implicit_root_trace.resolved
    assert evidence.projection_trace is None
    assert evidence.fidelity.status == "certified"
    assert set(evidence.source_entity_ids) == {
        source_entity_id(
            evidence.source_id,
            evidence.source_revision,
            "implicit-surface",
            "zero-set",
        )
    }
    np.testing.assert_array_equal(
        np.asarray(result.candidate_geometry.coordinates)[
            : result.construction.mesh.coordinates.shape[0]
        ],
        np.asarray(result.construction.mesh.coordinates),
    )
    evidence.require_current(result.construction.mesh, result.candidate_geometry)


def test_mapped_root_layer_builds_and_revalidates_exact_source_evidence() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    result = make_mapped_layer_curving_archive_case()
    assert result.status is LayerCurvingStatus.CURVED
    assert require_layer_core_construction(result) is result.construction
    assert result.candidate_geometry is not None
    assert result.validity is not None and result.validity.all_certified
    assert result.embedding is not None and result.embedding.status == "certified"
    evidence = result.source_evidence
    assert evidence is not None and evidence.mapped_chart_trace is not None
    assert evidence.mapped_chart_trace.resolved
    assert evidence.source_surface_charts is None
    assert evidence.projection_trace is None
    assert evidence.mapped_reference_maps
    assert evidence.fidelity.status == "certified"
    np.testing.assert_array_equal(
        np.asarray(result.candidate_geometry.coordinates)[
            : result.construction.mesh.coordinates.shape[0]
        ],
        np.asarray(result.construction.mesh.coordinates),
    )
    evidence.require_current(result.construction.mesh, result.candidate_geometry)


def test_layer_curving_source_revision_mismatch_rolls_back_unchanged() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    mapped_source, _, _, _, wall, layers, _, construction = (
        _mapped_layer_core_construction_for_fixture()
    )
    _, other_charts = _mapped_cube_surface_charts(
        wall, "other-curved-layer-source", mapped_source.source_revision
    )

    result = curve_layer_core_construction(
        construction,
        layers,
        other_charts,
        policy=HighOrderCurvingPolicy(degree=2, relaxation_rounds=0),
        fidelity_tolerance=0.0,
    )

    assert result.status is LayerCurvingStatus.UNRESOLVED_SOURCE_MAP
    assert result.construction is construction
    assert result.candidate_geometry is None
    assert result.failure is not None
    np.testing.assert_array_equal(
        np.asarray(result.construction.geometry.coordinates),
        np.asarray(construction.geometry.coordinates),
    )

    with pytest.raises(LayerCurvingFailure) as failure:
        require_layer_core_construction(result)
    assert failure.value.result is result


def test_implicit_layer_root_chart_transfer_preserves_exact_cell_and_column_ancestry() -> (
    None
):
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    _surface, source, _contract, wall, _wall_geometry, root_witness = _adaptive_source()
    layers = _layer_mesh(wall, None)
    mesh = layers.mesh
    faces = _implicit_source_faces(mesh, layers, source, root_witness)
    implicit_entity = source_entity_id(
        source.source_id,
        source.source_revision,
        "implicit-surface",
        "zero-set",
    )
    implicit_cell_keys = np.asarray(root_witness.implicit_cell_keys, dtype=np.int64)
    root_by_chart = {
        chart_id: (
            int(cell_id),
            tuple(int(implicit_cell_keys[row, axis]) for axis in range(3)),
        )
        for row, (chart_id, cell_id) in enumerate(
            zip(
                root_witness.source_chart_ids,
                np.asarray(root_witness.cell_global_ids, dtype=np.int64),
                strict=True,
            )
        )
    }
    assert root_by_chart
    assert root_witness.source_entity_ids == tuple(
        implicit_entity for _ in root_witness.source_chart_ids
    )
    assert tuple(face.source_entity_id for face in faces) == tuple(
        implicit_entity for _ in faces
    )
    assert {face.implicit_chart_id for face in faces} == set(root_by_chart)
    for face in faces:
        root_cell_id, implicit_cell_key = root_by_chart[face.implicit_chart_id]
        assert face.source_cell_id == root_cell_id
        assert face.implicit_cell_key == implicit_cell_key
    source_cells: dict[int, tuple[int, int, int]] = {}
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Implicit source ancestry requires triangular source cells.")
        cell_ids = np.asarray(block.global_ids, dtype=np.int64).reshape((-1,))
        cell_vertices = np.asarray(block.vertices, dtype=np.int64).reshape((-1, 3))
        for row in range(cell_ids.size):
            source_cells[int(cell_ids[row])] = (
                int(cell_vertices[row, 0]),
                int(cell_vertices[row, 1]),
                int(cell_vertices[row, 2]),
            )
    source_vertex_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    for face in faces:
        root_vertices = tuple(
            int(source_vertex_ids[row]) for row in source_cells[face.source_cell_id]
        )
        assert face.implicit_root_cell_vertex_global_ids == root_vertices
        assert set(face.source_column_root_vertex_ids) <= set(source_vertex_ids.tolist())
        assert face.target_cell_id >= 0
        assert face.layer_index >= 0
        assert face.layer_front_row_identity


def test_fixed_topology_layer_evidence_renews_from_exact_identity_maps() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    result = make_layer_curving_archive_case()
    evidence = result.source_evidence
    assert evidence is not None
    mesh = result.construction.mesh
    geometry = result.construction.geometry
    lineage = _identity_lineage(mesh, mesh)
    reference_maps = _identity_reference_maps_for_transition(mesh, mesh)

    renewed = renew_layer_curving_source_evidence(
        evidence,
        mesh,
        geometry,
        mesh,
        geometry,
        evidence.chart_witness,
        lineage,
        reference_maps,
    )

    assert (
        renewed.chart_column_identity.identity_id
        == evidence.chart_column_identity.identity_id
    )
    renewed.require_current(mesh, geometry)


def test_mapped_layer_transition_rebuilds_root_maps_and_lift_evidence() -> None:
    if not meshcore_available():
        pytest.skip("native mapped layer/core generation requires meshcore")
    result = make_mapped_layer_curving_archive_case()
    evidence = result.source_evidence
    geometry = result.candidate_geometry
    assert evidence is not None and geometry is not None
    assert evidence.mapped_chart_trace is not None
    mesh = result.construction.mesh
    renewed = renew_layer_curving_source_evidence(
        evidence,
        mesh,
        geometry,
        mesh,
        geometry,
        evidence.chart_witness,
        _identity_lineage(mesh, mesh),
        _identity_reference_maps_for_transition(mesh, mesh),
    )
    assert renewed.mapped_chart_trace is not None
    assert renewed.mapped_reference_maps
    np.testing.assert_array_equal(
        renewed.mapped_chart_trace.root_cell_global_ids,
        evidence.mapped_chart_trace.root_cell_global_ids,
    )
    renewed.require_current(mesh, geometry)


def test_restricted_layer_transition_replays_full_map_without_node_rows() -> None:
    if not meshcore_available():
        pytest.skip("native boundary-layer generation requires meshcore")
    result = make_restricted_layer_curving_archive_case()
    assert result.status is LayerCurvingStatus.CURVED
    evidence = result.source_evidence
    assert evidence is not None and evidence.source_surface_charts is not None
    assert evidence.projection_trace is None
    assert evidence.chart_witness.mesh_id == result.construction.mesh.mesh_id
    evidence.require_current(result.construction.mesh, result.construction.geometry)
