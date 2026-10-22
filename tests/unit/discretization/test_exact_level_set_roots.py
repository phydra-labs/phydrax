# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Source-owned exact P1 roots: rounded carriers never stand in for exact geometry."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    barycentric_homogeneous_preparation_bounds,
    BarycentricCellGeometryElement,
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    exact_barycentric_action,
    homogeneous_barycentric_element,
    PolynomialComposedCellGeometryElement,
    prepare_barycentric_source_snapshot,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    affine_geometry_is_carrier,
    affine_vertex_positions,
    exact_root_points,
    is_affine_cell_geometry,
    transition_nested_cell_geometry,
    vertex_positions_are_carrier,
)
from phydrax.discretization._cell_geometry_transition_types import (
    exact_reference_corners,
    NestedReferenceWitnesses,
    require_reference_homogeneous,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import coordinate_corner_images
from phydrax.discretization._exact_algebraic_geometry import (
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
)
from phydrax.discretization._exact_rational_geometry import rational_source_layouts
from phydrax.discretization._mapped_geometry_measure import (
    certified_cell_measures,
    certify_nested_geometry_pairs,
)
from phydrax.discretization._nested_reference import nested_reference_pairs
from phydrax.discretization._nested_reference_types import rational_pair_affine
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from tests._support.coordinate_references import coordinate_reference_budget


# A P1 level set with values 1 at vertex 0 and -2 at vertex 1 has its rational root
# at homogeneous weights (|-2|, |1|) = (2, 1): reference x = 1/3, physical 2/3.
_PARENT = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)))
_THIRD = float(Fraction(1, 3))
_ROOT = float(Fraction(2, 3))
_REFERENCE = np.asarray(
    (
        ((0.0, 0.0, 0.0), (_THIRD, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((_THIRD, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
)
_HOMOGENEOUS = np.asarray(
    (
        ((1.0, 0, 0, 0), (2.0, 1.0, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, 1.0)),
        ((2.0, 1.0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, 1.0)),
    )
)


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


_PARENT_ROW = np.asarray(((0, 1, 2, 3),), dtype=np.int32)


def _source_mesh() -> CellMesh:
    return CellMesh(
        _PARENT,
        (
            CellBlock(
                "solid",
                "tetrahedron",
                _PARENT_ROW,
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(4),
    )


def _target_mesh(*, split_blocks: bool) -> CellMesh:
    coordinates = np.concatenate((_PARENT, ((_ROOT, 0.0, 0.0),)))
    rows = np.asarray(((0, 4, 2, 3), (4, 1, 2, 3)), dtype=np.int32)
    ids = np.asarray((10, 11), dtype=np.int64)
    blocks = (
        (
            CellBlock("first", "tetrahedron", rows[:1], global_ids=ids[:1]),
            CellBlock("second", "tetrahedron", rows[1:], global_ids=ids[1:]),
        )
        if split_blocks
        else (CellBlock("solid", "tetrahedron", rows, global_ids=ids),)
    )
    return CellMesh(coordinates, blocks, vertex_global_ids=np.arange(5))


def _witnesses(*, exact: bool) -> NestedReferenceWitnesses:
    return NestedReferenceWitnesses(
        np.asarray((10, 11), dtype=np.int64),
        np.asarray((0, 0), dtype=np.int64),
        _REFERENCE.copy(),
        _HOMOGENEOUS.copy() if exact else None,
    )


def _restricted_parent_geometry(budget: CoordinateEnclosureBudget) -> CellGeometrySpec:
    """Identity restriction of the P1 parent, with its scientific ancestry."""
    p1 = coordinate_lagrange_element("tetrahedron", 1)
    root = CellGeometrySpec({"solid": p1}, {"solid": _PARENT_ROW}, _PARENT, budget=budget)
    return CellGeometrySpec(
        {
            "solid": RestrictedCellGeometryElement(
                p1, "tetrahedron", np.eye(3), np.zeros(3)
            )
        },
        {"solid": _PARENT_ROW},
        _PARENT,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(root),
            _source_mesh().topology_id,
            {"solid": np.asarray((0,), dtype=np.int64)},
            {"solid": np.asarray(_PARENT_ROW, dtype=np.int64)},
        ),
        budget=budget,
    )


def test_homogeneous_bank_authenticates_its_rounded_corners() -> None:
    bank = require_reference_homogeneous(_REFERENCE, _HOMOGENEOUS, "roots")
    assert bank is not None and not bank.flags.writeable
    corners = exact_reference_corners(_REFERENCE[0], bank[0])
    assert corners[1] == (Fraction(1, 3), Fraction(0), Fraction(0))
    assert Fraction(_REFERENCE[0, 1, 0]) != Fraction(1, 3)
    shifted = _REFERENCE.copy()
    shifted[0, 1, 0] = np.nextafter(_THIRD, 1.0)
    with pytest.raises(ValueError, match="round-to-nearest view"):
        require_reference_homogeneous(shifted, _HOMOGENEOUS, "roots")
    signed = _HOMOGENEOUS.copy()
    signed[0, 0, 1] = -0.0
    with pytest.raises(ValueError, match="positive-signed"):
        require_reference_homogeneous(_REFERENCE, signed, "roots")


def test_exact_pairs_certify_the_rounded_carrier_deviation(
    budget: CoordinateEnclosureBudget,
) -> None:
    source, target = _source_mesh(), _target_mesh(split_blocks=False)
    source_geometry = CellGeometrySpec.affine(source, budget=budget)
    target_geometry = CellGeometrySpec.affine(target, budget=budget)
    parents = np.zeros(2, dtype=np.int64)
    exact = nested_reference_pairs(
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells=parents,
        parent_reference_vertices=_REFERENCE,
        parent_reference_homogeneous=_HOMOGENEOUS,
        budget=budget,
    )
    matrix, offset = rational_pair_affine(exact[0])
    assert matrix[0][0] == Fraction(1, 3) and offset == (Fraction(0),) * 3
    matrix, offset = rational_pair_affine(exact[1])
    assert offset[0] == Fraction(1, 3) and matrix[0][0] == Fraction(2, 3)
    # The exact source law differs from the RNE carrier by its rounding only.
    deviation = certify_nested_geometry_pairs(
        source, source_geometry, target, target_geometry, exact, budget=budget
    )
    assert 0.0 < deviation <= float(np.spacing(_ROOT))
    # Without the bank the rounded corners are only float reference maps; their
    # rational images are not a provable partition of the parent, so they are
    # refused rather than certified as a rational restriction.
    with pytest.raises(ValueError, match="nested_reference"):
        nested_reference_pairs(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parents,
            parent_reference_vertices=_REFERENCE,
            budget=budget,
        )


def test_homogeneous_action_retains_exact_corner_images_and_execution_rows(
    budget: CoordinateEnclosureBudget,
) -> None:
    p1 = coordinate_lagrange_element("tetrahedron", 1)
    element = homogeneous_barycentric_element(p1, _HOMOGENEOUS[0])
    assert exact_barycentric_action(element)[1] == (
        Fraction(2, 3),
        Fraction(1, 3),
        Fraction(0),
        Fraction(0),
    )
    dyadic = BarycentricCellGeometryElement(p1, np.asarray(element.barycentric_weights))
    assert dyadic.element_id != element.element_id
    local = tuple(tuple(Fraction(value) for value in row) for row in _PARENT)
    images = coordinate_corner_images(element, local, budget=budget)
    assert images is not None
    assert images[1] == (Fraction(2, 3), Fraction(0), Fraction(0))
    wrong = np.asarray(element.barycentric_weights).copy()
    wrong[1, 0] = np.nextafter(wrong[1, 0], 0.0)
    with pytest.raises(ValueError, match="round-to-nearest view"):
        BarycentricCellGeometryElement(p1, wrong, _HOMOGENEOUS[0])
    # The binary64 rows remain the dynamic execution leaf under JIT.
    points = np.asarray(((0.25, 0.25, 0.25),))
    values, _ = element.tabulate(points)
    jitted, _ = jax.jit(lambda owner: owner.tabulate(points))(element)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(jitted))


def test_exact_root_children_restrict_nest_conserve_and_restore(
    budget: CoordinateEnclosureBudget,
) -> None:
    source = _source_mesh()
    p1 = coordinate_lagrange_element("tetrahedron", 1)
    identity = BarycentricCellGeometryElement(p1, np.eye(4))
    source_geometry = CellGeometrySpec(
        {"solid": identity}, {"solid": _PARENT_ROW}, _PARENT, budget=budget
    )
    target = _target_mesh(split_blocks=True)
    transition = transition_nested_cell_geometry(
        source,
        source_geometry,
        target,
        CellGeometrySpec.affine(target, budget=budget),
        refinement=_witnesses(exact=True),
        budget=budget,
    )
    assert transition.parent_reference_homogeneous is not None
    np.testing.assert_array_equal(
        np.asarray(transition.parent_reference_homogeneous)[:, :4], _HOMOGENEOUS
    )
    evidence = transition.evidence
    assert evidence.containment_defect == 0.0
    assert evidence.measure_exact
    source_error = evidence.source_measure_error_bound
    target_error = evidence.target_measure_error_bound
    assert source_error is not None and target_error is not None
    assert abs(evidence.target_measure - evidence.source_measure) <= (
        source_error + target_error
    )
    assert float(np.asarray(transition.vertex_coordinates)[4, 0]) == _ROOT
    local = tuple(tuple(Fraction(value) for value in row) for row in _PARENT)
    first = transition.geometry.elements[0]
    images = coordinate_corner_images(first, local, budget=budget)
    assert images is not None and images[1][0] == Fraction(2, 3)
    carrier = target.with_coordinates(
        transition.vertex_coordinates, numeric_version=target.numeric_version
    )
    pairs = nested_reference_pairs(
        source,
        source_geometry,
        carrier,
        transition.geometry,
        geometry_transition=transition,
        budget=budget,
    )
    assert all(pair.exact_matrix is not None for pair in pairs)
    assert (
        certify_nested_geometry_pairs(
            source, source_geometry, carrier, transition.geometry, pairs, budget=budget
        )
        == 0.0
    )
    values, errors, exact = certified_cell_measures(
        carrier, transition.geometry, budget=budget
    )
    assert exact
    assert abs(Fraction(sum(values)) - Fraction(4, 3)) <= Fraction(
        sum(errors)
    ) + Fraction(float(np.spacing(4 / 3)))
    register_meshing_source_artifacts()
    restored = reconstruct_stored_fields(transition)
    np.testing.assert_array_equal(
        np.asarray(restored.parent_reference_homogeneous),
        np.asarray(transition.parent_reference_homogeneous),
    )
    assert restored.transition_id == transition.transition_id
    back = transition_nested_cell_geometry(
        carrier,
        transition.geometry,
        source,
        CellGeometrySpec.affine(source, budget=budget),
        refinement=None,
        coarsening=_witnesses(exact=True),
        budget=budget,
    )
    assert back.geometry.elements[0].element_id == identity.element_id
    assert back.coarsened_reference_homogeneous is not None


def test_restricted_parents_keep_exact_composed_root_charts(
    budget: CoordinateEnclosureBudget,
) -> None:
    # An admitted affine parent that is not a full P1 coefficient owner is
    # composed with the exact rational child chart, never a rounded restriction.
    source = _source_mesh()
    source_geometry = _restricted_parent_geometry(budget)
    parent = source_geometry.elements[0]
    target = _target_mesh(split_blocks=True)
    transition = transition_nested_cell_geometry(
        source,
        source_geometry,
        target,
        CellGeometrySpec.affine(target, budget=budget),
        refinement=_witnesses(exact=True),
        budget=budget,
    )
    first = transition.geometry.elements[0]
    assert isinstance(first, PolynomialComposedCellGeometryElement)
    assert first.chart_coefficients[1][0] == (1, 3)
    local = tuple(tuple(Fraction(value) for value in row) for row in _PARENT)
    images = coordinate_corner_images(first, local, budget=budget)
    assert images is not None and images[1][0] == Fraction(2, 3)
    carrier = target.with_coordinates(
        transition.vertex_coordinates, numeric_version=target.numeric_version
    )
    back = transition_nested_cell_geometry(
        carrier,
        transition.geometry,
        source,
        CellGeometrySpec.affine(source, budget=budget),
        refinement=None,
        coarsening=_witnesses(exact=True),
        budget=budget,
    )
    assert back.geometry.elements[0].element_id == parent.element_id


def test_mixed_route_refuses_corners_that_are_not_the_root_rounding(
    budget: CoordinateEnclosureBudget,
) -> None:
    source = _source_mesh()
    p1 = coordinate_lagrange_element("tetrahedron", 1)
    source_geometry = CellGeometrySpec(
        {"solid": BarycentricCellGeometryElement(p1, np.eye(4))},
        {"solid": _PARENT_ROW},
        _PARENT,
        budget=budget,
    )
    shifted = _witnesses(exact=True)
    vertices = shifted.fine_reference_vertices.copy()
    vertices[0, 1, 0] = 0.25
    with pytest.raises(ValueError, match="round-to-nearest view"):
        transition_nested_cell_geometry(
            source,
            source_geometry,
            _target_mesh(split_blocks=True),
            CellGeometrySpec.affine(_target_mesh(split_blocks=True), budget=budget),
            refinement=shifted._replace(fine_reference_vertices=vertices),
            budget=budget,
        )


def test_replay_refuses_rescaled_bank_under_stale_identity(
    budget: CoordinateEnclosureBudget,
) -> None:
    # Doubling a homogeneous row keeps its rational point and rounded vertex,
    # so only the identity binding of the whole bank can refuse it.
    source = _source_mesh()
    p1 = coordinate_lagrange_element("tetrahedron", 1)
    source_geometry = CellGeometrySpec(
        {"solid": BarycentricCellGeometryElement(p1, np.eye(4))},
        {"solid": _PARENT_ROW},
        _PARENT,
        budget=budget,
    )
    target = _target_mesh(split_blocks=True)
    transition = transition_nested_cell_geometry(
        source,
        source_geometry,
        target,
        CellGeometrySpec.affine(target, budget=budget),
        refinement=_witnesses(exact=True),
        budget=budget,
    )
    bank = transition.parent_reference_homogeneous
    assert bank is not None
    register_meshing_source_artifacts()
    assert reconstruct_stored_fields(transition).transition_id == (
        transition.transition_id
    )
    scaled = np.asarray(bank).copy()
    scaled[0, 1] *= 2.0
    forged = eqx.tree_at(
        lambda value: value.parent_reference_homogeneous, transition, scaled
    )
    with pytest.raises(ValueError, match="transition_id must bind"):
        reconstruct_stored_fields(forged)


def test_exact_snapshot_normalization_is_admitted_before_it_runs() -> None:
    p1 = coordinate_lagrange_element("tetrahedron", 1)
    element = homogeneous_barycentric_element(p1, _HOMOGENEOUS[0])
    required, _ = barycentric_homogeneous_preparation_bounds(4)
    with HostBatchTicket(required - 1, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=required - 1, maximum_memory_bytes=1 << 24
        ).activate() as ledger:
            with pytest.raises(CoordinateEnclosureResourceError):
                prepare_barycentric_source_snapshot(element, budget=ledger)
            assert not ledger.barycentric_source_snapshot_cache


def test_exact_root_points_use_exact_source_not_rounded_endpoints(
    budget: CoordinateEnclosureBudget,
) -> None:
    # Rational endpoints 1/3 and 1 with homogeneous weights (3, 8): the root is
    # 9/11, whose rounding differs from the root of the rounded endpoint 1/3.
    third_carrier = float(Fraction(1, 3))
    mesh = CellMesh(
        np.asarray(((third_carrier, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (
            CellBlock(
                "solid",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(3),
    )
    p1 = coordinate_lagrange_element("triangle", 1)
    # Corner 0 is the rational parent point 1/3 of the source edge [0, 1].
    third = homogeneous_barycentric_element(
        p1, np.asarray(((2.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    )
    geometry = CellGeometrySpec(
        {"solid": third},
        {"solid": np.asarray(((0, 1, 2),), dtype=np.int32)},
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        budget=budget,
    )
    assert not affine_geometry_is_carrier(mesh, geometry)
    positions = affine_vertex_positions(mesh, geometry, budget=budget)
    assert positions[0] == (Fraction(1, 3), Fraction(0))
    assert not vertex_positions_are_carrier(positions, mesh)
    carriers, deviations = exact_root_points(
        positions,
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray(((3.0, 8.0),)),
        budget=budget,
    )
    assert float(carriers[0, 0]) == float(Fraction(9, 11)) == 0.8181818181818182
    rounded_endpoint = float((3 * Fraction(third_carrier) + 8) / 11)
    assert rounded_endpoint == 0.8181818181818181
    assert 0.0 < deviations[0] <= float(np.spacing(0.8181818181818182))
    # A carrier that is not the rounding of the exact source vertex refuses.
    shifted = mesh.with_coordinates(
        np.asarray(((0.25, 0.0), (1.0, 0.0), (0.0, 1.0))),
        numeric_version=mesh.numeric_version,
    )
    with pytest.raises(ValueError, match="round-to-nearest view of the exact source"):
        affine_vertex_positions(shifted, geometry, budget=budget)
    plain = CellGeometrySpec.affine(mesh, budget=budget)
    assert affine_geometry_is_carrier(mesh, plain)


@pytest.mark.parametrize("family", ("p1", "restricted"))
def test_repeated_nondyadic_roots_compose_exact_source_geometry(
    family: str, budget: CoordinateEnclosureBudget
) -> None:
    # First split publishes the root 2/3 under each admitted parent family; a
    # second split between that root and vertex 1 (at 2) with weights (3, 8)
    # must use the unrounded 2/3, giving 18/11, never the rounded first root.
    source = _source_mesh()
    source_geometry = (
        CellGeometrySpec(
            {
                "solid": BarycentricCellGeometryElement(
                    coordinate_lagrange_element("tetrahedron", 1), np.eye(4)
                )
            },
            {"solid": _PARENT_ROW},
            _PARENT,
            budget=budget,
        )
        if family == "p1"
        else _restricted_parent_geometry(budget)
    )
    target = _target_mesh(split_blocks=True)
    transition = transition_nested_cell_geometry(
        source,
        source_geometry,
        target,
        CellGeometrySpec.affine(target, budget=budget),
        refinement=_witnesses(exact=True),
        budget=budget,
    )
    carrier = target.with_coordinates(
        transition.vertex_coordinates, numeric_version=target.numeric_version
    )
    first = transition.geometry.elements[0]
    assert isinstance(
        first,
        BarycentricCellGeometryElement
        if family == "p1"
        else PolynomialComposedCellGeometryElement,
    )
    assert not affine_geometry_is_carrier(carrier, transition.geometry)
    positions = affine_vertex_positions(carrier, transition.geometry, budget=budget)
    assert positions[4] == (Fraction(2, 3), Fraction(0), Fraction(0))
    assert not vertex_positions_are_carrier(positions, carrier)
    carriers, deviations = exact_root_points(
        positions,
        np.asarray(((4, 1),), dtype=np.int64),
        np.asarray(((3.0, 8.0),)),
        budget=budget,
    )
    assert float(carriers[0, 0]) == float(Fraction(18, 11))
    assert tuple(carriers[0, 1:].tolist()) == (0.0, 0.0)
    exact_deviation = abs(Fraction(float(carriers[0, 0])) - Fraction(18, 11))
    assert Fraction(float(deviations[0])) >= exact_deviation
    assert deviations[0] <= float(np.spacing(float(Fraction(18, 11))))


def test_admitted_radical_affine_source_roots_round_from_field_values(
    budget: CoordinateEnclosureBudget,
) -> None:
    # An algebraic affine tetrahedron with a sqrt(2) corner is admitted by
    # `is_affine_cell_geometry`; its roots stay in the field, never Fractions.
    bank = (
        (RadicalElement(), RadicalElement(), RadicalElement()),
        (RadicalElement.sqrt(2), RadicalElement(), RadicalElement()),
        (RadicalElement(), RadicalElement(1), RadicalElement()),
        (RadicalElement(), RadicalElement(), RadicalElement(1)),
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    elements = {"solid": coordinate_lagrange_element("tetrahedron", 1)}
    routes = {"solid": cells}
    source = ExactAlgebraicCellGeometrySource(
        bank,
        rational_source_layouts(elements, routes),
        source_identity="level-set-radical-affine-tetrahedron",
        maximum_work=budget.maximum_work_units,
        maximum_bits=1024,
        budget=budget,
    )
    points = source.representatives(budget=budget)
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "solid",
                "tetrahedron",
                cells,
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(4),
    )
    geometry = CellGeometrySpec(
        elements, routes, points, exact_source=source, budget=budget
    )
    assert is_affine_cell_geometry(mesh, geometry, budget=budget)
    assert not affine_geometry_is_carrier(mesh, geometry)
    positions = affine_vertex_positions(mesh, geometry, budget=budget)
    assert positions[1][0] == RadicalElement.sqrt(2)
    assert not vertex_positions_are_carrier(positions, mesh)
    # Root at sqrt(2)/3 between vertex 0 and vertex 1 with weights (2, 1).
    carriers, deviations = exact_root_points(
        positions,
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray(((2.0, 1.0),)),
        budget=budget,
    )
    exact = RadicalElement.sqrt(2) / 3
    assert carriers[0, 0] == float(exact)
    assert tuple(carriers[0, 1:].tolist()) == (0.0, 0.0)
    assert 0.0 < deviations[0] <= float(np.spacing(float(exact)))
