# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational child correspondence reuses source evaluations at shared corners."""

from __future__ import annotations

from fractions import Fraction
from typing import NoReturn

import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    _cell_geometry_transfer as transfer_owner,
    _mapped_geometry_measure as measure_owner,
    CellBlock,
    CellMesh,
)
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from phydrax.discretization._cell_geometry_transfer import (
    _plc_affine_point,
    _plc_nested_bound,
    _plc_node_reference_point,
    _plc_parent_polynomials,
    _plc_reference_node_weights,
    _require_plc_successor_corners,
    SimplexGeometryBlock,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransitionError,
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import (
    CoordinateCoefficients,
    CoordinatePreparationKey,
    CoordinateSourceBank,
    evaluate,
    ExactScalar,
    Polynomial,
)
from phydrax.discretization._reference_cell_types import CellGeometryElement
from tests._support.cell_meshes import kuhn_cube, unit_tetrahedron


@pytest.mark.parametrize(
    "offset", (Fraction(0), Fraction(1, 8)), ids=("exact", "changed-shared-corner")
)
def test_child_correspondence_reuses_exact_source_corners_under_original_work(
    offset: Fraction,
) -> None:
    root = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    midpoint = (Fraction(1, 2), Fraction(0), Fraction(0))
    first = (root[0], midpoint, root[2], root[3])
    second = (midpoint, root[1], (offset, Fraction(1), Fraction(0)), root[3])
    references = np.asarray(
        ((root[0], midpoint, root[2], root[3]), (midpoint, root[1], root[2], root[3])),
        dtype=np.float64,
    )
    witnesses = NestedReferenceWitnesses(
        np.asarray((11, 12), dtype=np.int64),
        np.asarray((7, 7), dtype=np.int64),
        references,
    )
    # Four root corners cost 9 each and the midpoint costs 12. Re-evaluating
    # shared corners costs 78, which cannot pass this unchanged arithmetic cap.
    with HostBatchTicket(48, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=48, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            bound = _plc_nested_bound(
                witnesses, {11: first, 12: second}, {7: root}, budget=budget
            )
            assert budget.work_units == 48
            assert budget.temporary_bytes_upper == 0
    assert bound == offset


def test_children_reuse_parent_polynomials_without_recharging_preparation() -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    roots = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    shifted = tuple((point[0] + 2, *point[1:]) for point in roots)
    bank = (*roots, *shifted)
    route = np.arange(4, dtype=np.int64)
    memo: dict[int, tuple[Polynomial, ...]] = {}
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            first = _plc_parent_polynomials(0, element, bank, route, memo, budget=budget)
            prepared_work = budget.work_units
            repeated = _plc_parent_polynomials(
                0, element, bank, route, memo, budget=budget
            )
            assert budget.work_units == prepared_work
            other = _plc_parent_polynomials(
                1, element, bank, route + 4, memo, budget=budget
            )
            point = (Fraction(1, 4),) * 3
            assert evaluate(first[0], point, budget=budget) == Fraction(1, 4)
            assert evaluate(repeated[0], point, budget=budget) == Fraction(1, 4)
            assert evaluate(other[0], point, budget=budget) == Fraction(9, 4)


@pytest.mark.parametrize("degree", (1, 2, 3))
def test_node_weight_plan_preserves_exact_homogeneous_corners(degree: int) -> None:
    element = coordinate_lagrange_element("tetrahedron", degree)
    homogeneous = np.asarray(
        (((1, 0, 0, 0), (1, 2, 0, 0), (1, 0, 2, 0), (1, 0, 0, 2)),),
        dtype=np.float64,
    )
    reference = homogeneous[..., 1:] / np.sum(homogeneous, axis=-1, keepdims=True)
    witness = NestedReferenceWitnesses(
        np.asarray((11,), dtype=np.int64),
        np.asarray((7,), dtype=np.int64),
        reference,
        fine_reference_homogeneous=homogeneous,
    )
    corners = transfer_owner._exact_witness_corners(witness, 0)
    assert corners[1][0] == Fraction(2, 3)
    assert corners[1][0] != Fraction(float(reference[0, 1, 0]))
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            plan = _plc_reference_node_weights(element, budget=budget)
            prepared_work = budget.work_units
            nodes = np.asarray(element.reference_nodes)
            original_weights = np.concatenate(
                (1 - np.sum(nodes, axis=-1, keepdims=True), nodes), axis=-1
            )
            for weights, original in zip(plan, original_weights, strict=True):
                point = _plc_node_reference_point(weights, corners)
                assert point == tuple(
                    sum(
                        (
                            Fraction(float(weight)) * corner[axis]
                            for weight, corner in zip(original, corners, strict=True)
                            if weight
                        ),
                        Fraction(0),
                    )
                    for axis in range(3)
                )
                if len(weights) == 1 and weights[0][1] == 1:
                    assert point is corners[weights[0][0]]
            assert budget.work_units == prepared_work


@pytest.mark.parametrize("cardinal", (True, False), ids=("cardinal", "general"))
@pytest.mark.parametrize("corrupt", (False, True), ids=("exact", "changed-carrier"))
def test_successor_corners_round_distinct_authorities_once(
    monkeypatch: pytest.MonkeyPatch, cardinal: bool, corrupt: bool
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) + Fraction(1, 3) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    coordinates = np.asarray(bank, dtype=np.float64)
    if corrupt:
        coordinates[-1, 0] = np.nextafter(coordinates[-1, 0], np.inf)
    block = SimplexGeometryBlock(
        element,
        np.tile(np.arange(4, dtype=np.int64), (8, 1)),
        np.arange(8, dtype=np.int64),
        3,
        1,
    )
    rounded = transfer_owner.imported_coordinate_enclosure.rounded_point
    calls: list[tuple[Fraction, ...]] = []

    def counted(point: tuple[Fraction, ...]) -> np.ndarray:
        calls.append(point)
        return rounded(point)

    monkeypatch.setattr(
        transfer_owner.imported_coordinate_enclosure, "rounded_point", counted
    )
    if not cardinal:
        # Disable only the optimization; the real general exact evaluator runs.
        monkeypatch.setattr(
            transfer_owner.imported_cell_geometry,
            "_cardinal_corner_nodes",
            lambda element, *, budget: None,
        )
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            if corrupt:
                with pytest.raises(ValueError, match="correctly rounded map images"):
                    _require_plc_successor_corners(
                        (block,), bank, coordinates, budget=budget
                    )
            else:
                _require_plc_successor_corners(
                    (block,), bank, coordinates, budget=budget
                )
    assert calls == list(bank)


def test_affine_corner_evaluation_matches_exact_polynomials_and_primitive_work() -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = (
        (Fraction(1, 3), Fraction(2, 7), Fraction(-1, 11)),
        (Fraction(5, 3), Fraction(-1, 7), Fraction(3, 11)),
        (Fraction(2, 3), Fraction(13, 7), Fraction(5, 11)),
        (Fraction(4, 3), Fraction(3, 7), Fraction(17, 11)),
    )
    tail = (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            nodes = transfer_owner.imported_cell_geometry._cardinal_corner_nodes(
                element, budget=budget
            )
            assert nodes is not None
            assert transfer_owner.imported_coordinate_enclosure.affine_simplex_source(
                element, budget=budget
            )
            polynomials = _plc_parent_polynomials(
                0, element, bank, np.arange(4, dtype=np.int64), {}, budget=budget
            )
            expected = tuple(evaluate(value, tail, budget=budget) for value in polynomials)
            before = budget.work_units
            actual = _plc_affine_point(
                tuple(bank[node] for node in nodes), tail, budget=budget
            )
            assert actual == expected
            assert budget.work_units - before == 6 + 3 * 4


def test_affine_measure_selects_exact_cardinal_authority_without_image_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    shift = (Fraction(1, 3), Fraction(2, 7), Fraction(-1, 11))
    bank = tuple(
        tuple(Fraction(float(value)) + shift[axis] for axis, value in enumerate(point))
        for point in np.asarray(element.reference_nodes)
    )

    def forbidden_image_preparation(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Cardinal affine measures must select their source authority")

    monkeypatch.setattr(
        measure_owner, "coordinate_corner_images", forbidden_image_preparation
    )
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            measured = measure_owner._exact_affine_simplex_measure(
                element, bank, budget=budget
            )
            assert measured is not None
            value, error, exact = measured
            assert exact
            assert abs(Fraction(value) - Fraction(1, 6)) <= Fraction(error)


def test_public_affine_measures_do_not_retain_per_cell_cache_key_graphs() -> None:
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            values, errors, exact = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
            assert exact
            assert abs(Fraction(float(values[0])) - Fraction(1, 6)) <= Fraction(
                float(errors[0])
            )
            assert not budget.coefficient_preparation_cache
            assert len(budget.exact_cell_measure_cache) == 1
            key = next(iter(budget.exact_cell_measure_cache))
            assert len(key) == 4 and key[2] == ()
            assert key[0].startswith("affine-rational-measure:")
            assert type(key[3]) is bytes and len(key[3]) < 1024
            work = budget.work_units
    with HostBatchTicket(work, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            admitted = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            np.testing.assert_array_equal(admitted[0], values)
            np.testing.assert_array_equal(admitted[1], errors)
    with HostBatchTicket(work, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=work - 1, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            with pytest.raises(CellGeometryTransitionError) as refusal:
                measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert refusal.value.reason == "resource_limit"
            assert refusal.value.limit == work - 1


@pytest.mark.parametrize("corner", range(4))
def test_affine_cardinal_point_retains_exact_source_authority(corner: int) -> None:
    corners = tuple(
        tuple(Fraction(1 + row, 3 + axis) for axis in range(3)) for row in range(4)
    )
    tail = tuple(Fraction(int(corner == axis + 1)) for axis in range(3))
    with HostBatchTicket(9, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=9, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            actual = _plc_affine_point(corners, tail, budget=budget)
            assert actual is corners[corner]
            assert budget.work_units == 9
            assert budget.temporary_bytes_upper == 0


def test_affine_corner_evaluation_admits_storage_before_rational_arithmetic() -> None:
    corners = ((Fraction(1, 3), Fraction(2, 7), Fraction(3, 11)),) * 4
    with HostBatchTicket(100, bytes_upper=1024, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100, maximum_memory_bytes=1024
        ).activate() as budget:
            with pytest.raises(CoordinateEnclosureResourceError):
                _plc_affine_point(corners, (Fraction(1, 4),) * 3, budget=budget)
            assert budget.work_units == 0


@pytest.mark.parametrize("cardinal", (True, False), ids=("selected", "general"))
def test_nested_pair_corner_selection_preserves_exact_values_and_charges(
    monkeypatch: pytest.MonkeyPatch, cardinal: bool
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(value) + Fraction(1, 7 + axis) for axis, value in enumerate(point))
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    calls = []
    original = measure_owner.imported_coordinate_enclosure.coordinate_preparation_key

    def observed(
        element: CellGeometryElement,
        local: CoordinateCoefficients,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[CoordinateSourceBank, CoordinatePreparationKey]:
        calls.append(1)
        return original(element, local, budget=budget)

    monkeypatch.setattr(
        measure_owner.imported_coordinate_enclosure, "coordinate_preparation_key", observed
    )
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            assert measure_owner.imported_coordinate_enclosure.affine_simplex_source(
                element, budget=budget
            )
            expected = measure_owner.imported_coordinate_enclosure.coordinate_corner_images(
                element, bank, budget=budget
            )
            nodes = measure_owner._cardinal_corner_nodes(element, budget=budget)
            assert nodes is not None
            calls.clear()
            if not cardinal:
                monkeypatch.setattr(
                    measure_owner, "_cardinal_corner_nodes", lambda *a, **k: None
                )
            before = budget.work_units
            proofs: dict = {}
            with budget.live_storage() as storage:
                actual = measure_owner._nested_affine_corner_images(
                    element, bank, proofs=proofs, storage=storage, budget=budget
                )
                assert actual == expected
                if cardinal:
                    assert not calls
                    # The first proof still authenticates every current owner;
                    # only later pure-batch cells skip that already executed scan.
                    assert budget.work_units - before > len(nodes) * (1 + len(bank[0]))
                else:
                    assert calls
                # A second cell of the same element reuses its completed verdicts
                # while still paying its own canonical corner selection.
                identities = []
                original_identity = measure_owner.imported_coordinate_enclosure.element_identity

                def counted(
                    element: CellGeometryElement,
                    /,
                    *,
                    budget: CoordinateEnclosureBudget | None = None,
                ) -> tuple[str, str]:
                    identities.append(1)
                    return original_identity(element, budget=budget)

                monkeypatch.setattr(
                    measure_owner.imported_coordinate_enclosure, "element_identity", counted
                )
                between = budget.work_units
                again = measure_owner._nested_affine_corner_images(
                    element, bank, proofs=proofs, storage=storage, budget=budget
                )
                assert again == expected
                if cardinal:
                    assert not identities
                    assert budget.work_units - between == len(nodes) * (1 + len(bank[0]))


def test_nested_pair_cardinality_does_not_replace_full_affine_basis_proof() -> None:
    element = coordinate_lagrange_element("tetrahedron", 2)
    bank = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            assert measure_owner._cardinal_corner_nodes(element, budget=budget) is not None
            with budget.live_storage() as storage:
                assert (
                    measure_owner._nested_affine_corner_images(
                        element, bank, proofs={}, storage=storage, budget=budget
                    )
                    is None
                )


def test_public_affine_batch_reauthenticates_source_and_reuses_only_exact_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = kuhn_cube(1)
    geometry = CellGeometrySpec.affine(mesh)
    source_basis = measure_owner.source_basis
    cardinal_nodes = measure_owner._cardinal_corner_nodes
    corner_measure = measure_owner._exact_affine_simplex_measure_from_corners
    sources: list[object] = []
    cardinals: list[object] = []
    determinants: list[object] = []

    def counted_source(
        element: CellGeometryElement, *, budget: CoordinateEnclosureBudget,
    ) -> tuple[Polynomial, ...] | None:
        sources.append(element)
        return source_basis(element, budget=budget)

    def counted_cardinal(
        element: CellGeometryElement, *, budget: CoordinateEnclosureBudget,
    ) -> tuple[int, ...] | None:
        cardinals.append(element)
        return cardinal_nodes(element, budget=budget)

    def counted_measure(
        corners: tuple[tuple[ExactScalar, ...], ...],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[float, float, bool]:
        determinants.append(corners)
        return corner_measure(corners, budget=budget)

    monkeypatch.setattr(measure_owner, "source_basis", counted_source)
    monkeypatch.setattr(measure_owner, "_cardinal_corner_nodes", counted_cardinal)
    monkeypatch.setattr(
        measure_owner, "_exact_affine_simplex_measure_from_corners", counted_measure
    )
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            executed_work = []
            first_storage = None
            first_publication = None
            for operation in range(2):
                before = budget.work_units
                values, errors, exact = measure_owner.certified_cell_measures(
                    mesh, geometry, budget=budget
                )
                assert exact and len(values) == 6
                assert all(
                    abs(Fraction(float(value)) - Fraction(1, 6)) <= Fraction(float(error))
                    for value, error in zip(values, errors, strict=True)
                )
                assert len(sources) == len(cardinals) == operation + 1
                assert sources[-1] is cardinals[-1]
                assert len(determinants) == (6 if operation == 0 else 7)
                assert not budget.coefficient_preparation_cache
                assert len(budget.exact_cell_measure_cache) == 6
                executed_work.append(budget.work_units - before)
                if first_publication is None:
                    first_publication = (values.tobytes(), errors.tobytes())
                    first_storage = budget.retained_basis_bytes
                else:
                    assert (values.tobytes(), errors.tobytes()) == first_publication
                    assert budget.retained_basis_bytes == first_storage
            assert 0 < executed_work[1] < executed_work[0]


def test_public_highorder_simplex_keeps_full_nonaffine_source_despite_cardinal_corners() -> None:
    element = coordinate_lagrange_element("tetrahedron", 2)
    nodes = np.asarray(element.reference_nodes)
    coordinates = nodes.copy()
    coordinates[:, 0] += nodes[:, 0] * nodes[:, 1] / 4
    mesh = CellMesh(
        np.asarray(unit_tetrahedron().coordinates),
        (CellBlock("cells", "tetrahedron", np.arange(4, dtype=np.int64)[None]),),
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(element.local_dof_count, dtype=np.int64)[None]},
        coordinates,
    )
    bank = tuple(tuple(Fraction(float(value)) for value in row) for row in coordinates)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            assert measure_owner._cardinal_corner_nodes(element, budget=budget) is not None
            with budget.live_storage() as storage:
                proofs: measure_owner._NestedCornerProofs = {}
                assert measure_owner._exact_affine_simplex_measure(
                    element, bank, preparation=(proofs, storage), budget=budget
                ) is None
                assert not proofs
            values, errors, exact = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
    # X=x+xy/4 has det J=1+y/4, while all four corner images are unchanged.
    assert exact
    assert abs(Fraction(float(values[0])) - Fraction(17, 96)) <= Fraction(float(errors[0]))
    assert abs(Fraction(float(values[0])) - Fraction(1, 6)) > Fraction(float(errors[0]))


@pytest.mark.parametrize("bank_kind", ("coefficients", "source"))
def test_public_affine_batch_reauthenticates_readonly_numpy_owners_between_calls(
    bank_kind: str,
) -> None:
    root = coordinate_lagrange_element("tetrahedron", 1)
    element = BarycentricCellGeometryElement(root, np.eye(4))
    mesh = CellMesh(
        np.asarray(unit_tetrahedron().coordinates),
        (CellBlock("cells", "tetrahedron", np.arange(4, dtype=np.int64)[None]),),
    )
    coordinates = np.asarray(mesh.coordinates).copy()
    geometry = CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(4, dtype=np.int64)[None]}, coordinates
    )
    original = coordinates if bank_kind == "coefficients" else np.eye(4)
    backing = np.repeat(original, 2, axis=1)
    readonly = backing[:, ::2]
    readonly.setflags(write=False)
    if bank_kind == "coefficients":
        object.__setattr__(geometry, "coordinates", readonly)
    else:
        object.__setattr__(element, "barycentric_weights", readonly)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            first, first_error, first_exact = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
            if bank_kind == "coefficients":
                backing[:, 0] *= 2
                expected = Fraction(1, 3)
            else:
                backing[1, ::2] = (0.5, 0.5, 0.0, 0.0)
                expected = Fraction(1, 12)
            second, second_error, second_exact = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
    assert first_exact and second_exact
    assert abs(Fraction(float(first[0])) - Fraction(1, 6)) <= Fraction(float(first_error[0]))
    assert abs(Fraction(float(second[0])) - expected) <= Fraction(float(second_error[0]))
    assert second[0] != first[0]
    assert not readonly.flags.writeable


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_public_affine_batch_obeys_original_work_and_storage_caps(resource: str) -> None:
    mesh = kuhn_cube(1)
    geometry = CellGeometrySpec.affine(mesh)
    before = np.asarray(geometry.coordinates).copy()
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            work, storage = budget.work_units, budget.peak_bytes_upper
    for deficit in (0, 1):
        work_limit = work - deficit if resource == "work" else 1_000_000
        storage_limit = storage - deficit if resource == "storage" else 1 << 24
        with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
            with CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=work_limit,
                maximum_memory_bytes=storage_limit,
            ).activate() as budget:
                if deficit:
                    with pytest.raises(CellGeometryTransitionError) as refusal:
                        measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
                    assert refusal.value.reason == "resource_limit"
                    assert refusal.value.measured > refusal.value.limit
                    np.testing.assert_array_equal(geometry.coordinates, before)
                else:
                    values, _, exact = measure_owner.certified_cell_measures(
                        mesh, geometry, budget=budget
                    )
                    assert exact and len(values) == 6


def test_affine_measure_preadmits_operation_proof_metadata_before_source_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )

    def forbidden_source(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Operation proof metadata must be admitted before source work")

    monkeypatch.setattr(measure_owner, "source_basis", forbidden_source)
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=128
        ).activate() as budget:
            with budget.live_storage() as storage:
                proofs: measure_owner._NestedCornerProofs = {}
                with pytest.raises(CoordinateEnclosureResourceError):
                    measure_owner._exact_affine_simplex_measure(
                        element, bank, preparation=(proofs, storage), budget=budget
                    )
                assert not proofs and storage.bound == 0
                assert budget.work_units == 0


def test_affine_batch_reuse_keeps_each_actual_bank_determinant_and_original_charges() -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    with HostBatchTicket(100_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            with budget.live_storage() as storage:
                proofs: measure_owner._NestedCornerProofs = {}
                first = measure_owner._exact_affine_simplex_measure(
                    element, bank, preparation=(proofs, storage), budget=budget
                )
                assert first is not None and id(element) in proofs
                assert proofs[id(element)][0] is element
                before = budget.work_units
                stretched = tuple((2 * point[0], *point[1:]) for point in bank)
                second = measure_owner._exact_affine_simplex_measure(
                    element, stretched, preparation=(proofs, storage), budget=budget
                )
                assert second is not None
                assert abs(Fraction(second[0]) - Fraction(1, 3)) <= Fraction(second[1])
                # Four corner+component selections, twelve actual corner values,
                # and the unchanged 26-operation three-dimensional determinant.
                assert budget.work_units - before == 4 * (1 + 3) + 12 + 26
                before = budget.work_units
                inverted = tuple((-point[0], *point[1:]) for point in bank)
                with pytest.raises(ValueError, match="nonpositive Jacobian bound"):
                    measure_owner._exact_affine_simplex_measure(
                        element, inverted, preparation=(proofs, storage), budget=budget
                    )
                assert budget.work_units - before == 4 * (1 + 3) + 12 + 26


@pytest.mark.parametrize("field", ("_numerator", "_denominator"))
def test_affine_measure_cache_encodes_current_fraction_values_not_owner_hashes(
    monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    first_bank = tuple((3 * point[0], *point[1:]) for point in bank)
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    monkeypatch.setattr(
        measure_owner, "mapped_geometry_cells",
        lambda *args, **kwargs: ((element, first_bank), (element, bank)),
    )
    current = next(point[0] for point in bank if point[0] == 1)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            original, _, _ = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
            old_keys = tuple(budget.exact_cell_measure_cache)
            old_hash = hash(current)
            object.__setattr__(current, field, 2)
            changed, errors, exact = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
            assert len(budget.exact_cell_measure_cache) == 3
            assert all(key in budget.exact_cell_measure_cache for key in old_keys)
            assert hash(current) != old_hash
    expected = Fraction(1, 3) if field == "_numerator" else Fraction(1, 12)
    assert exact and changed[0] == original[0]
    assert abs(Fraction(float(changed[1])) - expected) <= Fraction(float(errors[1]))
    assert changed[1] != original[1]


def test_affine_measure_cache_distinguishes_current_basis_with_unchanged_declared_id() -> None:
    root = coordinate_lagrange_element("tetrahedron", 1)
    element = BarycentricCellGeometryElement(root, np.eye(4))
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec(
        {mesh.blocks[0].name: element},
        {mesh.blocks[0].name: np.arange(4, dtype=np.int64)[None]},
        np.asarray(mesh.coordinates),
    )
    declared_id = element.element_id
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            first = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            first_key = next(iter(budget.exact_cell_measure_cache))
            # A three-cycle preserves orientation but changes the actual basis.
            object.__setattr__(
                element, "barycentric_weights", np.eye(4)[[0, 2, 3, 1]]
            )
            second = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert len(budget.exact_cell_measure_cache) == 2
            second_key = next(key for key in budget.exact_cell_measure_cache if key != first_key)
            assert first_key[0] == second_key[0]
            assert first_key[1] != second_key[1]
            assert first_key[2:] == second_key[2:]
    assert element.element_id == declared_id
    assert first[2] and second[2]
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1].tobytes() == second[1].tobytes()


def test_affine_measure_cache_rechecks_current_tolerance_on_numerical_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    exact_bank = tuple((3 * point[0], *point[1:]) for point in bank)
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    monkeypatch.setattr(
        measure_owner, "mapped_geometry_cells",
        lambda *args, **kwargs: ((element, exact_bank), (element, bank)),
    )
    original = measure_owner._exact_affine_simplex_measure_from_corners
    determinants = []

    def counted(
        corners: tuple[tuple[ExactScalar, ...], ...],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[float, float, bool]:
        determinants.append(corners)
        return original(corners, budget=budget)

    monkeypatch.setattr(measure_owner, "_exact_affine_simplex_measure_from_corners", counted)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            _, errors, exact = measure_owner.certified_cell_measures(
                mesh, geometry, budget=budget
            )
            assert exact and errors[0] == 0 and errors[1] > 0
            with pytest.raises(CellGeometryTransitionError, match="publication_error"):
                measure_owner.certified_cell_measures(
                    mesh, geometry, absolute_tolerance=float(errors[1]),
                    relative_tolerance=0, budget=budget,
                )
            # First-cell proof is fresh; the second-cell tolerance refusal
            # occurs on import, without executing another determinant.
            assert len(determinants) == 3


@pytest.mark.parametrize("scale", (0, -1), ids=("degenerate", "negative"))
def test_affine_measure_cache_never_imports_a_changed_nonpositive_map(
    monkeypatch: pytest.MonkeyPatch, scale: int,
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    cells = [(element, bank), (element, bank)]
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    monkeypatch.setattr(
        measure_owner, "mapped_geometry_cells", lambda *args, **kwargs: tuple(cells)
    )
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert len(budget.exact_cell_measure_cache) == 1
            cells[1] = (element, tuple((scale * point[0], *point[1:]) for point in bank))
            with pytest.raises(ValueError, match="nonpositive Jacobian bound"):
                measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert len(budget.exact_cell_measure_cache) == 1


def test_affine_measure_cache_requires_a_strictly_positive_publication_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operand_owner

    def forbidden_issue(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Unproved positive publication intervals cannot issue operands")

    monkeypatch.setattr(
        operand_owner, "_remember_exact_cell_measure_operand", forbidden_issue
    )
    element = coordinate_lagrange_element("tetrahedron", 1)
    tiny = Fraction(1, 1 << 1100)
    bank = tuple(
        (Fraction(float(point[0])) * tiny,
         Fraction(float(point[1])), Fraction(float(point[2])))
        for point in np.asarray(element.reference_nodes)
    )
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    monkeypatch.setattr(
        measure_owner, "mapped_geometry_cells",
        lambda *args, **kwargs: ((element, bank), (element, bank)),
    )
    original = measure_owner._exact_affine_simplex_measure_from_corners
    determinants = []

    def counted(
        corners: tuple[tuple[ExactScalar, ...], ...],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[float, float, bool]:
        determinants.append(corners)
        return original(corners, budget=budget)

    monkeypatch.setattr(measure_owner, "_exact_affine_simplex_measure_from_corners", counted)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            budget.capture_measure_operands = True
            budget.capture_coordinate_operands = True
            publications = [
                measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
                for _ in range(2)
            ]
            assert not budget.exact_cell_measure_cache
            assert len(determinants) == 4
    assert all(result[2] and np.all(result[0] == 0) for result in publications)
    assert publications[0][0].tobytes() == publications[1][0].tobytes()
    assert publications[0][1].tobytes() == publications[1][1].tobytes()


def test_embedded_irrational_affine_measure_is_never_cached_as_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _coordinate_operands as operand_owner

    def forbidden_issue(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Irrational affine measures cannot issue exact operands")

    monkeypatch.setattr(
        operand_owner, "_remember_exact_cell_measure_operand", forbidden_issue
    )
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 1)), dtype=np.float64)
    mesh = CellMesh(
        np.concatenate((points, points + (2, 0, 0))),
        (CellBlock("cells", "triangle", np.asarray(((0, 1, 2), (3, 4, 5)))),),
    )
    geometry = CellGeometrySpec.affine(mesh)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            budget.capture_measure_operands = True
            budget.capture_coordinate_operands = True
            first = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            second = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert not budget.exact_cell_measure_cache
    assert not first[2] and not second[2]
    assert np.all(first[1] > 0)
    np.testing.assert_allclose(first[0], np.sqrt(2) / 2, atol=0, rtol=1e-15)
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1].tobytes() == second[1].tobytes()


@pytest.mark.parametrize("scalar_kind", ("fraction-subclass", "algebraic"))
def test_affine_measure_cache_leaves_nonplain_scalars_on_original_route(
    monkeypatch: pytest.MonkeyPatch, scalar_kind: str,
) -> None:
    from phydrax.discretization import _coordinate_operands as operand_owner

    def forbidden_issue(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Nonplain scalar banks cannot issue rational measure operands")

    monkeypatch.setattr(
        operand_owner, "_remember_exact_cell_measure_operand", forbidden_issue
    )

    class DerivedFraction(Fraction):
        pass

    scalar = (
        DerivedFraction if scalar_kind == "fraction-subclass"
        else measure_owner.exact_algebraic_geometry_owner.RadicalElement
    )
    element = coordinate_lagrange_element("tetrahedron", 1)
    bank = tuple(
        tuple(scalar(Fraction(float(value))) for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    monkeypatch.setattr(
        measure_owner, "mapped_geometry_cells",
        lambda *args, **kwargs: ((element, bank), (element, bank)),
    )
    original = measure_owner._exact_affine_simplex_measure_from_corners
    determinants = []

    def counted(
        corners: tuple[tuple[ExactScalar, ...], ...],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[float, float, bool]:
        determinants.append(corners)
        return original(corners, budget=budget)

    monkeypatch.setattr(measure_owner, "_exact_affine_simplex_measure_from_corners", counted)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            budget.capture_measure_operands = True
            budget.capture_coordinate_operands = True
            for _ in range(2):
                values, errors, exact = measure_owner.certified_cell_measures(
                    mesh, geometry, budget=budget
                )
                assert exact
                assert all(
                    abs(Fraction(float(value)) - Fraction(1, 6)) <= Fraction(float(error))
                    for value, error in zip(values, errors, strict=True)
                )
            assert not budget.exact_cell_measure_cache
            assert len(determinants) == 4


def test_affine_measure_cache_accounts_complete_large_keys_without_repeat_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    element = coordinate_lagrange_element("tetrahedron", 1)
    translation = Fraction(1 << 8192)
    bank = tuple(
        tuple(Fraction(float(value)) + translation for value in point)
        for point in np.asarray(element.reference_nodes)
    )
    mesh = unit_tetrahedron()
    geometry = CellGeometrySpec.affine(mesh)
    monkeypatch.setattr(
        measure_owner, "mapped_geometry_cells",
        lambda *args, **kwargs: ((element, bank), (element, bank)),
    )
    with HostBatchTicket(1_000_000, bytes_upper=1 << 20, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 20
        ).activate() as budget:
            first = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert len(budget.exact_cell_measure_cache) == 1
            key = next(iter(budget.exact_cell_measure_cache))
            assert len(key) == 4 and key[2] == () and len(key[3]) > 12_000
            retained = budget.retained_basis_bytes
            assert retained >= len(key[3])
            for _ in range(3):
                second = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
                assert second[2]
                assert second[0].tobytes() == first[0].tobytes()
                assert second[1].tobytes() == first[1].tobytes()
                assert budget.retained_basis_bytes == retained
            assert budget.peak_bytes_upper <= budget.maximum_memory_bytes
            assert not budget.coefficient_preparation_cache


@pytest.mark.parametrize("capture", (False, True), ids=("default", "issued"))
def test_affine_measure_operand_issue_requires_fresh_computation_not_cache_hit(
    monkeypatch: pytest.MonkeyPatch, capture: bool,
) -> None:
    from phydrax.discretization import _coordinate_operands as operand_owner

    mesh = kuhn_cube(1)
    geometry = CellGeometrySpec.affine(mesh)
    original = operand_owner._remember_exact_cell_measure_operand
    issued = []

    def remember(
        key: CoordinatePreparationKey,
        integral: tuple[float, float],
        budget: CoordinateEnclosureBudget,
    ) -> None:
        issued.append((key, integral))
        assert integral[0] > integral[1] >= 0
        assert budget.exact_cell_measure_cache[key] == integral
        return original(key, integral, budget)

    monkeypatch.setattr(operand_owner, "_remember_exact_cell_measure_operand", remember)
    with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
        with CoordinateEnclosureBudget(
            ticket, maximum_work_units=1_000_000, maximum_memory_bytes=1 << 24
        ).activate() as budget:
            if capture:
                budget.capture_measure_operands = True
                budget.capture_coordinate_operands = True
            first = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert len(issued) == (6 if capture else 0)
            second = measure_owner.certified_cell_measures(mesh, geometry, budget=budget)
            assert len(issued) == (7 if capture else 0)
            assert first[2] and second[2]
            assert first[0].tobytes() == second[0].tobytes()
            assert first[1].tobytes() == second[1].tobytes()
