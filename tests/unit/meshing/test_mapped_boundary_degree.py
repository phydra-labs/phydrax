#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Boundary-degree embedding of curved (mapped) cell meshes.

Oracles are independent of the certificate: polynomial diffeomorphisms
(sheared cubes and squares), explicit nested or self-overlapping constructions,
and deliberately nonconforming shared nodes.
"""

from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import Expression, Polynomial
from phydrax.geometry import GlobalEmbeddingCertificate
from phydrax.geometry._embedding_cells import Cell
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from tests._support.cell_meshes import kuhn_cube_arrays
from tests._support.mapped_embedding import (
    certify_mapped_fixture,
    p2_geometry,
    shear,
    square_with_hole,
)
from tests._support.meshing_fixtures import finding_checks


def test_curved_p2_cube_face_certifies_by_boundary_degree() -> None:
    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(2, 0.0, 1.0))
    embedding = certify_mapped_fixture(mesh, p2_geometry(mesh, shear))
    evidence = embedding.boundary_degree
    assert embedding.status == "certified" and not embedding.findings
    assert evidence is not None and evidence.status == "embedded"
    assert evidence.failed_premise is None
    assert (evidence.shell_count, evidence.shell_exterior_degrees) == (1, (0,))
    assert evidence.maximum_degree == 1
    assert evidence.oriented_cell_count == 48
    assert evidence.boundary_facet_count == embedding.boundary_facet_count == 48
    assert evidence.tangent_cone_contact_count > 0
    assert evidence.conforming_entity_count > 0
    assert {
        "mapped_closed_cell_orientation",
        "mapped_shared_entity_conformity",
        "mapped_boundary_injectivity",
        "mapped_shell_exterior_degree",
        "mapped_boundary_degree",
    } <= set(embedding.evaluated_checks)


def test_straight_p2_tetrahedra_use_the_exact_affine_contact_route() -> None:
    # Dyadic straight edge nodes make every quadratic coefficient action
    # affine; the rational affine boundary-contact and exterior-degree proof runs.
    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(2, 0.0, 1.0))
    embedding = certify_mapped_fixture(mesh, p2_geometry(mesh, np.copy))
    assert embedding.status == "certified"
    assert embedding.binding.coordinate_scope == "mapped"
    assert {"mapped_affine_source_maps", "boundary_contact", "exterior_degree"} <= set(
        embedding.evaluated_checks
    )
    assert embedding.boundary_degree is None


def test_ulp_curved_edge_adjacent_p2_tetrahedra_certify_without_deep_subdivision() -> (
    None
):
    # Kuhn tetrahedra share the cube diagonal and other edges without sharing
    # facets. One-ulp edge-node offsets make every map quadratic.
    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    vertices = mesh.coordinates.shape[0]

    def nudge(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[vertices:] = np.nextafter(result[vertices:], np.inf)
        return result

    embedding = certify_mapped_fixture(mesh, p2_geometry(mesh, nudge))
    evidence = embedding.boundary_degree
    assert embedding.status == "certified"
    assert evidence is not None and evidence.status == "embedded"
    assert evidence.tangent_cone_contact_count > 0
    # Nearly affine contacts close on their root pieces.
    assert embedding.subdivision_piece_count <= 4 * evidence.boundary_pair_count + 64


def test_ulp_curved_p2_surface_adjacent_cells_certify_by_tangent_cones() -> None:
    # The outward boundary triangles of a Kuhn cube: adjacent cells meet at
    # 90 and 180 degree dihedral angles along shared edges and at vertices.
    points, tetrahedra = kuhn_cube_arrays(1, 0.0, 1.0)
    faces = phx.discretization.reference_cell_topology("tetrahedron").entities[2]
    rows = np.asarray([cell[list(face)] for cell in tetrahedra for face in faces])
    _, inverse, counts = np.unique(
        np.sort(rows, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    surface = phx.discretization.CellMesh.from_triangles(
        points, rows[counts[inverse.reshape(-1)] == 1]
    )

    def nudge(values: np.ndarray) -> np.ndarray:
        result = values.copy()
        result[points.shape[0] :] = np.nextafter(result[points.shape[0] :], -np.inf)
        return result

    embedding = certify_mapped_fixture(surface, p2_geometry(surface, nudge))
    assert embedding.status == "certified"
    assert embedding.boundary_degree is None


def test_folded_adjacent_p2_surface_cells_are_never_certified() -> None:
    # The second triangle folds back onto the first across their shared edge:
    # identical tangent cones admit no separating direction.
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.25, 0.25, 0.0))
    )
    surface = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (1, 0, 3)))
    )

    def nudge(values: np.ndarray) -> np.ndarray:
        result = values.copy()
        result[points.shape[0] :, 2] = np.nextafter(0.0, 1.0)
        return result

    geometry = p2_geometry(surface, nudge)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=surface)
    assert validity.all_certified
    # A folded contact stays undecided at every depth; a shallow budget keeps
    # the refusal fast without changing its meaning.
    embedding = phx.geometry.certify_global_embedding(
        surface,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_subdivision_depth=4),
    )
    assert embedding.status == "unresolved"
    assert "mapped_intersection_subdivision_depth" in finding_checks(
        embedding, "unresolved"
    )


def test_curved_p2_square_with_hole_certifies_two_shells() -> None:
    def bend(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[:, 0] += points[:, 1] * (3.0 - points[:, 1]) / 16.0
        return result

    embedding = certify_mapped_fixture(
        square_with_hole(), p2_geometry(square_with_hole(), bend)
    )
    evidence = embedding.boundary_degree
    assert embedding.status == "certified"
    assert evidence is not None and evidence.status == "embedded"
    # The outer shell is positively oriented, the hole is a cavity inside it.
    assert (evidence.shell_count, evidence.shell_exterior_degrees) == (2, (0, 0))
    assert embedding.ray_test_count > 0


def test_nested_p2_components_have_degree_two_and_are_refused() -> None:
    outer, outer_cells = kuhn_cube_arrays(1, 0.0, 3.0)
    inner, inner_cells = kuhn_cube_arrays(1, 1.0, 2.0)
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.concatenate((outer, inner)),
        np.concatenate((outer_cells, inner_cells + outer.shape[0])),
    )

    def bend(points: np.ndarray) -> np.ndarray:
        result = points.copy()
        result[:, 0] += points[:, 1] * (3.0 - points[:, 1]) / 64.0
        return result

    embedding = certify_mapped_fixture(mesh, p2_geometry(mesh, bend))
    evidence = embedding.boundary_degree
    assert embedding.status == "violated"
    assert "mapped_boundary_degree" in finding_checks(embedding, "violated")
    assert evidence is not None and evidence.status == "violated"
    assert evidence.failed_premise == "mapped_shell_exterior_degree"
    assert evidence.shell_count == 2
    assert sorted(evidence.shell_exterior_degrees) == [0, 1]
    assert evidence.maximum_degree == 2


def test_folded_boundary_with_valid_cells_is_never_accepted() -> None:
    # A strip wound one and a quarter times around the origin on an outward
    # spiral: every cell is positively oriented while the boundary crosses
    # itself and the last quarter turn covers the first.
    count = 20
    radius = np.linspace(1.0, 2.0, 2)
    angle = np.linspace(0.0, 2.5 * np.pi, count + 1)
    parameters = np.stack(np.meshgrid(radius, angle, indexing="ij"), -1).reshape(-1, 2)
    index = np.arange(parameters.shape[0]).reshape(2, count + 1)
    triangles = []
    for j in range(count):
        a, b = int(index[0, j]), int(index[1, j])
        c, d = int(index[1, j + 1]), int(index[0, j + 1])
        triangles.extend(((a, b, c), (a, c, d)))
    mesh = phx.discretization.CellMesh.from_triangles(parameters, np.asarray(triangles))

    def wind(points: np.ndarray) -> np.ndarray:
        r = points[:, 0] + 0.3 * points[:, 1] / (2.0 * np.pi)
        return np.stack((r * np.cos(points[:, 1]), r * np.sin(points[:, 1])), axis=1)

    geometry = p2_geometry(mesh, wind)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    embedding = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "violated"
    assert "mapped_cell_overlap" in finding_checks(embedding, "violated")
    assert embedding.boundary_degree is not None
    assert embedding.boundary_degree.status == "violated"
    assert embedding.boundary_degree.failed_premise == "mapped_boundary_injectivity"


def test_nonconforming_shared_edge_midpoint_is_not_conforming() -> None:
    # Two tetrahedra share only the edge 0-1; each owns its quadratic nodes and
    # the second moves its copy of the shared edge midpoint off the edge.
    corners = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    cells = np.asarray(((0, 1, 2, 3), (0, 1, 4, 5)), dtype=np.int64)
    mesh = phx.discretization.CellMesh.from_tetrahedra(corners, cells)
    element = coordinate_lagrange_element("tetrahedron", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    nodes = []
    for row in cells:
        origin = corners[row[0]]
        nodes.append(origin + reference @ (corners[row[1:]] - origin))
    midpoint = np.flatnonzero(
        np.all(reference == np.asarray((0.5, 0.0, 0.0), dtype=np.float64), axis=1)
    ).item()
    nodes[1][midpoint] += (0.0, -0.001, -0.001)
    name = mesh.blocks[0].name
    geometry = phx.discretization.CellGeometrySpec(
        {name: element},
        {name: np.arange(2 * reference.shape[0], dtype=np.int64).reshape(2, -1)},
        np.concatenate(nodes),
    )
    embedding = certify_mapped_fixture(mesh, geometry)
    evidence = embedding.boundary_degree
    assert embedding.status == "violated"
    assert "mapped_trace_mismatch" in finding_checks(embedding, "violated")
    assert evidence is not None and evidence.status == "violated"
    assert evidence.failed_premise == "mapped_shared_entity_conformity"


def test_certified_curved_embedding_round_trips_through_source_closure(
    tmp_path: Path,
) -> None:

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    embedding = certify_mapped_fixture(mesh, geometry)
    assert embedding.boundary_degree is not None
    receipt = write_meshing_source_closure(
        tmp_path / "curved-embedding", (mesh, geometry, embedding)
    )
    _, _, restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert type(restored) is type(embedding)
    assert isinstance(restored, GlobalEmbeddingCertificate)
    assert restored.boundary_degree is not None
    assert restored.certificate_id == embedding.certificate_id
    assert restored.boundary_degree.evidence_id == embedding.boundary_degree.evidence_id
    assert eqx.tree_equal(restored, embedding)


@pytest.mark.parametrize(
    ("shell_degrees", "maximum_degree"),
    (
        pytest.param((0,), 2, id="embedded-degree-two"),
        pytest.param((1,), 1, id="embedded-nonzero-shell"),
    ),
)
def test_boundary_degree_evidence_refuses_inconsistent_embedded_records(
    shell_degrees: tuple[int, ...], maximum_degree: int
) -> None:
    with pytest.raises(
        ValueError,
        match=r"MappedBoundaryDegreeEvidence\.shell_exterior_degrees must be zero on every embedded shell with maximum degree one",
    ):
        phx.geometry.MappedBoundaryDegreeEvidence(
            "embedded",
            None,
            oriented_cell_count=1,
            conforming_entity_count=0,
            boundary_facet_count=4,
            boundary_pair_count=0,
            tangent_cone_contact_count=0,
            shell_count=1,
            shell_exterior_degrees=shell_degrees,
            maximum_degree=maximum_degree,
        )


def test_boundary_degree_evidence_requires_a_deciding_premise_for_violation() -> None:
    with pytest.raises(
        ValueError,
        match=r"MappedBoundaryDegreeEvidence\.failed_premise must be absent exactly for embedded status",
    ):
        phx.geometry.MappedBoundaryDegreeEvidence(
            "violated",
            None,
            oriented_cell_count=1,
            conforming_entity_count=0,
            boundary_facet_count=4,
            boundary_pair_count=0,
            tangent_cone_contact_count=0,
            shell_count=1,
            shell_exterior_degrees=(0,),
            maximum_degree=1,
        )


def _polynomial(value: Expression) -> Polynomial:
    """Narrow an actual P2 trace coordinate, which is always a polynomial."""
    assert isinstance(value, dict)
    return value


def _actual_tet_trace_cells(
    mesh: phx.discretization.CellMesh,
    geometry: phx.discretization.CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
) -> list[Cell]:
    from phydrax.discretization._coordinate_enclosure import (
        coordinate_expressions,
        prepared_coordinate_source_bank,
    )
    from phydrax.geometry._embedding_cells import vertices

    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    bank = prepared_coordinate_source_bank(geometry, budget=budget)
    cells: list[Cell] = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for row, dofs in zip(np.asarray(block.vertices), np.asarray(route), strict=True):
            coordinates = coordinate_expressions(
                element, tuple(bank[int(dof)] for dof in dofs), budget=budget,
            )
            assert coordinates is not None
            cells.append(Cell(
                coordinates, "simplex", "tetrahedron", 3,
                tuple(int(vertex) for vertex in row), vertices("tetrahedron"),
            ))
    return cells


def test_actual_facet_proof_transports_connected_vertex_and_edge_stars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        legacy = owner._shared_entity_conformity(cells, budget=budget)
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert state.clean and witness is not None
        assert witness.evidence_id and witness.pairs
        restrictions = []
        original = owner.restrict_chart_expressions

        def observed(
            coordinates: tuple[Expression, ...], source_kind: str, target_kind: str,
            origin: np.ndarray, matrix: np.ndarray, *, budget: CoordinateEnclosureBudget,
        ) -> tuple[Expression, ...]:
            restrictions.append((coordinates, source_kind, target_kind, origin, matrix))
            return original(coordinates, source_kind, target_kind, origin, matrix, budget=budget)

        monkeypatch.setattr(owner, "restrict_chart_expressions", observed)
        actual = owner._shared_entity_conformity_from_facets(state, mesh, cells, witness, geometry=geometry, budget=budget)
        assert actual.mismatch is None and legacy.mismatch is None
        assert actual.count == legacy.count > 0
        assert actual.images == legacy.images
        assert restrictions == []


@pytest.mark.parametrize("fault", ("orientation", "vertex", "mid-edge", "face"))
def test_facet_transport_requires_real_current_trace_identities(fault: str) -> None:
    from dataclasses import replace
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import add, multiply, Polynomial
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState, mesh_facets
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        facets = mesh_facets(mesh, geometry=geometry, budget=budget)
        group = int(np.flatnonzero(facets.counts == 2)[0])
        occurrences = np.flatnonzero(facets.group == group)
        index = int(facets.cells[occurrences[1]])
        cell = cells[index]
        shared = tuple(int(vertex) for vertex in facets.rows[occurrences[0]] if vertex >= 0)
        local = tuple(cell.vertices.index(vertex) for vertex in shared)
        if fault == "orientation":
            row = list(cell.vertices)
            row[local[0]], row[local[1]] = row[local[1]], row[local[0]]
            cells[index] = replace(cell, vertices=tuple(row))
        else:
            barycentric: tuple[Polynomial, ...] = (
                {(0, 0, 0): Fraction(1), (1, 0, 0): Fraction(-1),
                 (0, 1, 0): Fraction(-1), (0, 0, 1): Fraction(-1)},
                {(1, 0, 0): Fraction(1)},
                {(0, 1, 0): Fraction(1)},
                {(0, 0, 1): Fraction(1)},
            )
            perturbation: Polynomial = {(0, 0, 0): Fraction(1, 8)}
            if fault != "vertex":
                for vertex in local[:2 if fault == "mid-edge" else 3]:
                    perturbation = multiply(
                        perturbation, barycentric[vertex], budget=budget,
                    )
            changed = add(_polynomial(cell.coordinates[0]), perturbation, budget=budget)
            cells[index] = replace(cell, coordinates=(changed, *cell.coordinates[1:]))
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert not state.clean and witness is None
        assert any(finding.check == "mapped_trace_mismatch" for finding in state.findings)
        if fault == "face":
            # A face bubble vanishes on every edge and vertex: their agreement
            # cannot substitute for the independently required facet theorem.
            assert owner._shared_entity_conformity(cells, budget=budget).mismatch is None


@pytest.mark.parametrize("fault", ("none", "vertex", "mid-edge"))
def test_disconnected_entity_stars_keep_full_exact_component_comparisons(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from dataclasses import replace
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import add, Polynomial
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                         (0, -1, 0), (0, 0, -1)), dtype=np.float64)
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3), (0, 1, 4, 5)), dtype=np.int64)
    )
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        if fault != "none":
            perturbation: Polynomial = (
                {(0, 0, 0): Fraction(1, 8)} if fault == "vertex" else
                {(1, 0, 0): Fraction(1, 8), (2, 0, 0): Fraction(-1, 8)}
            )
            changed = add(_polynomial(cells[1].coordinates[0]), perturbation, budget=budget)
            cells[1] = replace(cells[1], coordinates=(changed, *cells[1].coordinates[1:]))
        legacy = owner._shared_entity_conformity(cells, budget=budget)
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert state.clean and witness is not None and not witness.pairs
        observed = []
        original = owner.restrict_chart_expressions

        def restriction(
            coordinates: tuple[Expression, ...], source_kind: str, target_kind: str,
            origin: np.ndarray, matrix: np.ndarray, *, budget: CoordinateEnclosureBudget,
        ) -> tuple[Expression, ...]:
            observed.append((coordinates, source_kind, target_kind, origin, matrix))
            return original(coordinates, source_kind, target_kind, origin, matrix, budget=budget)

        monkeypatch.setattr(owner, "restrict_chart_expressions", restriction)
        actual = owner._shared_entity_conformity_from_facets(state, mesh, cells, witness, geometry=geometry, budget=budget)
        assert (actual.mismatch is None) == (legacy.mismatch is None) == (fault == "none")
        assert actual.count == legacy.count and actual.images == legacy.images
        if fault != "vertex":
            assert observed


def test_facet_proof_rejects_copies_changed_source_and_foreign_proof_context() -> None:
    import pickle
    from dataclasses import replace
    from fractions import Fraction

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert witness is not None
        with pytest.raises(TypeError, match="archive or proof"):
            pickle.dumps(witness)
        with pytest.raises(ValueError, match="actual current proof"):
            owner._shared_entity_conformity_from_facets(state, mesh, cells, replace(witness), geometry=geometry, budget=budget)
        with pytest.raises(ValueError, match="actual current proof"):
            owner._shared_entity_conformity_from_facets(EmbeddingState([], []), mesh, cells, witness, geometry=geometry, budget=budget)
        polynomial = cells[0].coordinates[0]
        assert isinstance(polynomial, dict)
        old = dict(polynomial)
        polynomial[(0, 0, 0)] = polynomial.get((0, 0, 0), Fraction(0)) + Fraction(1, 8)
        try:
            with pytest.raises(ValueError, match="actual current proof"):
                owner._shared_entity_conformity_from_facets(state, mesh, cells, witness, geometry=geometry, budget=budget)
        finally:
            polynomial.clear()
            polynomial.update(old)
        with coordinate_reference_budget() as foreign:
            with pytest.raises(ValueError, match="actual current proof"):
                owner._shared_entity_conformity_from_facets(state, mesh, cells, witness, geometry=geometry, budget=foreign)
    foreign_ticket = HostBatchTicket(
        budget.maximum_work_units, bytes_upper=budget.maximum_memory_bytes,
    )
    with budget.bind_ticket(foreign_ticket):
        with pytest.raises(ValueError, match="actual current proof"):
            owner._shared_entity_conformity_from_facets(state, mesh, cells, witness, geometry=geometry, budget=budget)
    assert foreign_ticket.work_units == 0


def test_facet_trace_witness_does_not_keep_its_proof_or_cells_alive() -> None:
    import gc
    import weakref

    from phydrax.geometry import _mapped_embedding as mapped
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert witness is not None
        first = weakref.ref(cells[0])
        state_ref = weakref.ref(state)
        del cells, state
        gc.collect()
        assert first() is None and state_ref() is None


@pytest.mark.parametrize("kind", ("rational", "nonstandard"))
def test_unsupported_facet_transport_retains_full_polynomial_fallback(kind: str) -> None:
    from dataclasses import replace
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import (
        expression_reference_evaluate,
        RationalPolynomial,
    )
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        if kind == "rational":
            cells = [
                replace(cell, coordinates=tuple(
                    RationalPolynomial(_polynomial(value), {(0, 0, 0): Fraction(1)})
                    for value in cell.coordinates
                ))
                for cell in cells
            ]
        else:
            cells = [
                replace(cell, corner_images=tuple(tuple(
                    expression_reference_evaluate(value, point, cell.domain, budget=budget)
                    for value in cell.coordinates
                ) for point in cell.reference_vertices))
                for cell in cells
            ]
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert state.clean and witness is None
        assert owner._shared_entity_conformity(cells, budget=budget).mismatch is None


def test_constructive_entity_proof_still_requires_actual_admission() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        cells = _actual_tet_trace_cells(mesh, geometry, budget)
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=geometry, budget=budget,
            retain_facet_traces=True,
        )
        assert witness is not None
        before = budget.work_units
        with pytest.raises(CoordinateEnclosureResourceError):
            with budget.bound_stage(0, budget.maximum_memory_bytes):
                owner._shared_entity_conformity_from_facets(state, mesh, cells, witness, geometry=geometry, budget=budget)
        assert budget.work_units == before


def test_materialized_composed_geometry_keeps_full_entity_comparisons() -> None:
    from phydrax.discretization._cell_geometry import (
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization.fem._reference_types import FiniteElementSpec
    from phydrax.geometry import (
        _boundary_degree_embedding as owner,
        _mapped_embedding as mapped,
    )
    from phydrax.geometry._embedding_state import EmbeddingState
    from tests._support.coordinate_references import coordinate_reference_budget

    mesh = phx.discretization.CellMesh.from_tetrahedra(*kuhn_cube_arrays(1, 0.0, 1.0))
    geometry = p2_geometry(mesh, shear)
    with coordinate_reference_budget() as budget:
        source = geometry.elements[0]
        assert isinstance(source, FiniteElementSpec)
        chart = coordinate_lagrange_element("tetrahedron", 1)
        chart_points = np.asarray(chart.reference_nodes, dtype=np.int64)
        composed = PolynomialComposedCellGeometryElement(
            source, chart, chart_points, np.ones_like(chart_points), budget=budget,
        )
        pulled = phx.discretization.CellGeometrySpec(
            {geometry.block_names[0]: composed},
            dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
            geometry.coordinates,
        )
        cells = _actual_tet_trace_cells(mesh, pulled, budget)
        assert all(type(value) is dict for cell in cells for value in cell.coordinates)
        state = EmbeddingState([], [])
        witness = mapped.trace_continuity(
            state, mesh, cells, geometry=pulled, budget=budget,
            retain_facet_traces=True,
        )
        assert state.clean and witness is None
        assert owner._shared_entity_conformity(cells, budget=budget).mismatch is None
