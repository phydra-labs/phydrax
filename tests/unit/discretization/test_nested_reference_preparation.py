# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bound nested-reference preparations shared by field transfers of one transition.

Oracles are independent of the shared path: affine bisection children tile their
parents exactly (zero physical-map bound), refused tilings stay refused when an
equal valid pattern was proved first, and a curved P2 shear restricted by native
bisection carries the coarse coordinate field onto the fine one. Only the record
`prepare_nested_reference` issued is admitted: rebuilt, replaced, copied or
lookalike records and changed pieces are refused before any consumer reads them.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import pickle
import weakref
from collections.abc import Iterator
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import phydrax as phx
import phydrax.discretization._nested_reference as nested_reference_owner
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_transition_types import (
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._nested_reference import (
    _NESTED_REFERENCE_ISSUER,
    nested_reference_binding,
    prepare_nested_reference,
    PreparedNestedReference,
    require_nested_reference,
)
from phydrax.discretization._nested_reference_types import PolynomialReferencePair
from phydrax.discretization.fem._topology_transfer import (
    prepare_shared_nested_field_transfer,
)
from phydrax.meshing import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.geometry_transfer_cases import (
    curved_shear_source,
    shear_term,
    transfer_triangles,
)
from tests._support.rational_mapped_cells import cylinder_source, six_quad_source


# Unit-coordinate restriction allocates 512 binary64 ulps (see the curved
# nested-transfer contract in test_cell_geometry_transfer.py).
_RESTRICTION_ROUNDING = 1e-13
_CHILD_A = ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5))
_CHILD_B = ((0.0, 0.0), (0.5, 0.5), (0.0, 1.0))
_PARENTS = (0, 0, 1, 1)
_TILING = (_CHILD_A, _CHILD_B, _CHILD_A, _CHILD_B)


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


def _bisected_square() -> tuple[CellMesh, CellGeometrySpec, CellMesh, CellGeometrySpec]:
    """Two parents, each split from its first vertex to the opposite midpoint."""
    corners = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    source = CellMesh.from_triangles(corners, np.asarray(((0, 1, 2), (0, 2, 3))))
    fine = np.concatenate((corners, ((1.0, 0.5), (0.5, 1.0))))
    target = CellMesh.from_triangles(
        fine, np.asarray(((0, 1, 4), (0, 4, 2), (0, 2, 5), (0, 5, 3)))
    )
    return (
        source,
        CellGeometrySpec.affine(source),
        target,
        CellGeometrySpec.affine(target),
    )


def _prepare(
    witnesses: tuple[tuple[tuple[float, float], ...], ...],
    budget: CoordinateEnclosureBudget,
) -> Any:
    source, source_geometry, target, target_geometry = _bisected_square()
    return prepare_nested_reference(
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells=np.asarray(_PARENTS, dtype=np.int64),
        parent_reference_vertices=np.asarray(witnesses, dtype=np.float64),
        rooted=False,
        budget=budget,
    )


def _admit(prepared: Any, budget: CoordinateEnclosureBudget) -> Any:
    """Admit a record for the valid bisected-square tiling on ``budget``."""
    source, source_geometry, target, target_geometry = _bisected_square()
    return require_nested_reference(
        prepared,
        source,
        source_geometry,
        target,
        target_geometry,
        parent_cells=np.asarray(_PARENTS, dtype=np.int64),
        parent_reference_vertices=np.asarray(_TILING, dtype=np.float64),
        rooted=False,
        certified=True,
        budget=budget,
    )


def _centroid_split(budget: CoordinateEnclosureBudget) -> Any:
    """One triangle split at its centroid: nondyadic rational child maps."""
    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    source = CellMesh.from_triangles(np.asarray(corners), np.asarray(((0, 1, 2),)))
    third = 1.0 / 3.0
    target = CellMesh.from_triangles(
        np.asarray((*corners, (third, third))),
        np.asarray(((0, 1, 3), (1, 2, 3), (2, 0, 3))),
    )
    vertices = np.asarray(
        [(corners[a], corners[b], (third, third)) for a, b in ((0, 1), (1, 2), (2, 0))],
        dtype=np.float64,
    )
    # The centroid is (1, 1, 1) / 3 in homogeneous parent corners.
    homogeneous = np.zeros((3, 3, 3), dtype=np.float64)
    homogeneous[:, 2] = 1.0
    return prepare_nested_reference(
        source,
        CellGeometrySpec.affine(source),
        target,
        CellGeometrySpec.affine(target),
        parent_cells=np.zeros(3, dtype=np.int64),
        parent_reference_vertices=vertices,
        parent_reference_homogeneous=homogeneous,
        rooted=False,
        budget=budget,
    )


def _array_owners(array: Any) -> Iterator[np.ndarray]:
    while isinstance(array, np.ndarray):
        yield array
        array = array.base


def test_repeated_child_patterns_certify_exact_nesting(
    budget: CoordinateEnclosureBudget,
) -> None:
    prepared = _prepare(_TILING, budget)
    assert prepared.affine and not prepared.rooted
    assert prepared.certified_bound() == 0.0
    assert len(prepared.affine_pieces()) == 4
    assert prepared.work_units > 0
    for pair in prepared.affine_pieces():
        assert not pair.matrix.flags.writeable


def test_proved_pattern_never_admits_a_duplicated_child(
    budget: CoordinateEnclosureBudget,
) -> None:
    # Parent 0 proves {A, B}; parent 1 repeats A twice with equal total measure.
    with pytest.raises(ValueError, match=r"nested_reference\."):
        _prepare((_CHILD_A, _CHILD_B, _CHILD_A, _CHILD_A), budget)


def test_proved_pattern_never_admits_a_reversed_child(
    budget: CoordinateEnclosureBudget,
) -> None:
    reversed_a = (_CHILD_A[0], _CHILD_A[2], _CHILD_A[1])
    with pytest.raises(ValueError, match=r"nested_reference\.jacobian"):
        _prepare((_CHILD_A, _CHILD_B, reversed_a, _CHILD_B), budget)


def test_issued_record_refuses_rebuilding_copies_and_array_writes(
    budget: CoordinateEnclosureBudget,
) -> None:
    prepared = _centroid_split(budget)
    assert any(pair.exact_matrix is not None for pair in prepared.affine_pieces())
    fields = tuple(getattr(prepared, item.name) for item in dataclasses.fields(prepared))
    with pytest.raises(TypeError, match=r"PreparedNestedReference\.issuer"):
        PreparedNestedReference(*fields, object())
    with pytest.raises(ValueError, match="issuer"):
        dataclasses.replace(prepared, geometry_bound=0.0)
    for duplicate in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match=r"PreparedNestedReference\.record"):
            duplicate(prepared)
    for pair in prepared.affine_pieces():
        for view in (pair.matrix, pair.offset):
            with pytest.raises(ValueError):
                view[...] = 0.0
            for owner in _array_owners(view):
                with pytest.raises(ValueError):
                    owner.setflags(write=True)


def test_rebuilt_records_are_refused_even_with_equal_content(
    budget: CoordinateEnclosureBudget,
) -> None:
    prepared = _prepare(_TILING, budget)
    assert _admit(prepared, budget) is prepared
    shifted = tuple(
        pair._replace(offset=pair.offset + 0.25) for pair in prepared.affine_pieces()
    )
    for changes in (
        {"pairs": shifted},
        {"geometry_bound": 0.0, "work_units": 0, "storage_bytes": 0},
        {},
    ):
        rebuilt = dataclasses.replace(
            prepared, issuer=_NESTED_REFERENCE_ISSUER, **changes
        )
        with pytest.raises(ValueError, match=r"PreparedNestedReference\.provenance"):
            _admit(rebuilt, budget)
    lookalike = SimpleNamespace(
        **{
            item.name: getattr(prepared, item.name)
            for item in dataclasses.fields(prepared)
        }
    )
    with pytest.raises(TypeError):
        _admit(lookalike, budget)


def test_each_other_ledger_pays_the_completed_preparation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with coordinate_reference_budget() as preparing:
        prepared = _prepare(_TILING, preparing)
        work = preparing.work_units
        assert _admit(prepared, preparing) is prepared
        assert preparing.work_units == work
    # Admission never reruns the proof: count every proving entry point.
    proofs: list[str] = []
    for owner, name in (
        (nested_reference_owner, "_reference_records"),
        (nested_reference_owner, "_require_reference_partition"),
        (nested_reference_owner.mapped_measure_owner, "certify_nested_geometry_pairs"),
    ):
        original = getattr(owner, name)

        def counted(
            *args: Any, _name: str = name, _call: Any = original, **kw: Any
        ) -> Any:
            proofs.append(_name)
            return _call(*args, **kw)

        monkeypatch.setattr(owner, name, counted)
    for _ in range(2):
        with coordinate_reference_budget() as ledger:
            work = ledger.work_units
            storage = ledger.retained_basis_bytes + ledger.temporary_bytes_upper
            queries = ledger.ticket.geometry_queries
            for _ in range(2):
                assert _admit(prepared, ledger) is prepared
            assert ledger.work_units - work == prepared.work_units > 0
            assert (
                ledger.retained_basis_bytes + ledger.temporary_bytes_upper - storage
                == prepared.storage_bytes
            )
            assert ledger.ticket.geometry_queries - queries == prepared.geometry_queries
    assert proofs == []
    # The same counters observe a cold preparation's proof.
    with coordinate_reference_budget() as ledger:
        _prepare(_TILING, ledger)
    assert set(proofs) == {
        "_reference_records",
        "_require_reference_partition",
        "certify_nested_geometry_pairs",
    }


def test_issue_provenance_retains_no_record() -> None:
    with coordinate_reference_budget() as ledger:
        issued = weakref.ref(_prepare(_TILING, ledger))
    gc.collect()
    assert issued() is None


def test_changed_polynomial_coefficients_are_refused(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    parents = np.zeros(6, dtype=np.int64)
    prepared = prepare_nested_reference(
        mesh,
        geometry,
        fine_mesh,
        fine_geometry,
        parent_cells=parents,
        rooted=True,
        budget=budget,
    )

    def admit() -> Any:
        return require_nested_reference(
            prepared,
            mesh,
            geometry,
            fine_mesh,
            fine_geometry,
            parent_cells=parents,
            rooted=True,
            certified=True,
            budget=budget,
        )

    assert admit() is prepared
    pair = prepared.pairs[0]
    assert isinstance(pair, PolynomialReferencePair)
    argument = pair.arguments[0]
    exponents, coefficient = next(iter(argument.items()))
    argument[exponents] = coefficient + Fraction(1, 7)
    try:
        with pytest.raises(ValueError, match=r"PreparedNestedReference\.pairs"):
            admit()
    finally:
        argument[exponents] = coefficient
    assert admit() is prepared


def test_reference_identity_never_rounds_supplied_values(
    budget: CoordinateEnclosureBudget,
) -> None:
    source, source_geometry, target, target_geometry = _bisected_square()
    parents = np.asarray(_PARENTS, dtype=np.int64)
    vertices = np.asarray(_TILING, dtype=np.float64)

    def identity(**inputs: Any) -> str:
        return nested_reference_binding(
            source, source_geometry, target, target_geometry, **inputs
        ).witness_id

    def prepare(**inputs: Any) -> Any:
        return prepare_nested_reference(
            source,
            source_geometry,
            target,
            target_geometry,
            parent_cells=parents,
            rooted=False,
            budget=budget,
            **inputs,
        )

    expected = identity(parent_cells=parents, parent_reference_vertices=vertices)
    # Equal values share one identity whatever dtype carries them.
    assert expected == identity(
        parent_cells=parents.astype(np.uint64),
        parent_reference_vertices=vertices.astype(np.float32),
    )
    assert expected == identity(
        parent_cells=parents, parent_reference_vertices=vertices.astype(object)
    )
    # A value binary64 would round keeps its own identity and is refused.
    third = vertices.astype(object)
    third[0, 2, 0] = Fraction(1, 3)
    assert identity(parent_cells=parents, parent_reference_vertices=third) != identity(
        parent_cells=parents, parent_reference_vertices=third.astype(np.float64)
    )
    with pytest.raises(
        ValueError, match=r"parent_reference_vertices must hold binary64 values"
    ):
        prepare(parent_reference_vertices=third)
    bank = np.full((4, 3, 3), Fraction(0), dtype=object)
    bank[0, 2] = (Fraction(1, 3),) * 3
    assert identity(
        parent_cells=parents,
        parent_reference_vertices=vertices,
        parent_reference_homogeneous=bank,
    ) != identity(
        parent_cells=parents,
        parent_reference_vertices=vertices,
        parent_reference_homogeneous=bank.astype(np.float64),
    )
    with pytest.raises(
        ValueError, match=r"reference_homogeneous must hold binary64 values"
    ):
        prepare(parent_reference_vertices=vertices, parent_reference_homogeneous=bank)
    # Identifiers beyond int64 keep a lossless identity instead of failing.
    assert expected != identity(
        parent_cells=np.asarray([0, 0, 1, 1 << 70], dtype=object),
        parent_reference_vertices=vertices,
    )


def test_coarsening_identifiers_are_never_truncated(
    budget: CoordinateEnclosureBudget,
) -> None:
    coarse_mesh, coarse_geometry, fine_mesh, fine_geometry = _bisected_square()
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in fine_mesh.blocks]
    )
    coarse_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in coarse_mesh.blocks]
    )[np.asarray(_PARENTS)]
    vertices = np.asarray(_TILING, dtype=np.float64)

    def prepare(fine: np.ndarray, coarse: np.ndarray) -> Any:
        return prepare_nested_reference(
            fine_mesh,
            fine_geometry,
            coarse_mesh,
            coarse_geometry,
            coarsening_witnesses=NestedReferenceWitnesses(fine, coarse, vertices),
            rooted=False,
            budget=budget,
        )

    expected = prepare(fine_ids, coarse_ids)
    assert all(pair.fine_is_source for pair in expected.affine_pieces())
    # Narrower integer dtypes carry the same identifiers and the same pieces.
    narrow = prepare(fine_ids.astype(np.uint16), coarse_ids.astype(np.int32))
    assert narrow.binding == expected.binding
    for fine, coarse, name in (
        (fine_ids + 0.5, coarse_ids, "fine_cell_ids"),
        (fine_ids, coarse_ids.astype(np.float64), "coarse_cell_ids"),
    ):
        with pytest.raises(
            ValueError,
            match=rf"coarsening\.{name} must hold integer identifiers within int64",
        ):
            prepare(fine, coarse)


def _refined(marked: tuple[int, ...]) -> tuple[Any, Any]:
    source = curved_shear_source(transfer_triangles(2), 2, shear_term(2, 2))
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION)
    request = MarkedMeshAdaptation(
        np.asarray(marked, dtype=np.int64), np.zeros(0, dtype=np.int64)
    )
    return source, execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _space(
    result: Any, *, budget: CoordinateEnclosureBudget | None = None
) -> Any:
    fields = (
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 2)
        ),
        phx.discretization.FiniteElementFieldSpec(
            "w", phx.discretization.lagrange_element("triangle", 1)
        ),
    )
    return phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry, budget=budget
    ).prepare(budget=budget)


def _witnesses(source: Any, refined: Any) -> dict[str, Any]:
    return {
        "parent_reference_vertices": refined.parent_reference_vertices,
        "parent_reference_homogeneous": refined.parent_reference_homogeneous,
        "source_geometry": source.geometry,
        "target_geometry": refined.target.geometry,
        "geometry_transition": refined.geometry_transition,
    }


def _coordinate_field(space: Any, result: Any) -> np.ndarray:
    """Physical coordinates at the P2 field's own nodal functionals."""
    routes = np.asarray(space.dof_maps[0].cell_dofs[0])
    nodes = np.asarray(result.geometry.geometry_dofs[0])
    values = np.zeros((space.dof_maps[0].global_dof_count, 2))
    weights, _ = result.geometry.elements[0].tabulate(
        space.elements[0][0].reference_nodes
    )
    local = np.asarray(weights) @ np.asarray(result.geometry.coordinates)[nodes]
    values[routes.reshape(-1)] = local.reshape((-1, 2))
    return values


def test_shared_preparation_matches_independent_field_transfers() -> None:
    source, refined = _refined((0, 4))
    coarse, fine = _space(source), _space(refined.target)
    witnesses = _witnesses(source, refined)
    first, reference = prepare_shared_nested_field_transfer(
        coarse, fine, refined.parent_cells, field_name="u", **witnesses
    )
    second, reused = prepare_shared_nested_field_transfer(
        coarse,
        fine,
        refined.parent_cells,
        field_name="w",
        prepared=() if reference is None else (reference,),
        **witnesses,
    )
    assert reference is not None and reused is reference
    for name, shared in (("u", first), ("w", second)):
        independent = phx.discretization.prepare_nested_field_transfer(
            coarse, fine, refined.parent_cells, field_name=name, **witnesses
        )
        assert shared.transfer_id == independent.transfer_id
        assert shared.evidence.evidence_id == independent.evidence.evidence_id
        assert shared.evidence.passed
    np.testing.assert_allclose(
        np.asarray(first.transfer.apply(_coordinate_field(coarse, source))),
        _coordinate_field(fine, refined.target),
        rtol=0.0,
        atol=_RESTRICTION_ROUNDING,
    )


def test_shared_nested_transfer_uses_live_budget_without_a_fresh_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization.fem._topology_transfer as transfer_owner

    source, refined = _refined((0, 4))
    coarse, fine = _space(source), _space(refined.target)
    witnesses = _witnesses(source, refined)
    independent, _ = prepare_shared_nested_field_transfer(
        coarse, fine, refined.parent_cells, field_name="u", **witnesses
    )

    def refuse_fresh_ticket(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("a supplied live request must not open an owned ticket")

    with coordinate_reference_budget() as ledger:
        before = ledger.work_units
        with monkeypatch.context() as patch:
            patch.setattr(transfer_owner, "_owned_transfer_budget", refuse_fresh_ticket)
            actual, reference = prepare_shared_nested_field_transfer(
                coarse,
                fine,
                refined.parent_cells,
                field_name="u",
                budget=ledger,
                **witnesses,
            )
        assert ledger.work_units > before
        assert reference is not None
    assert actual.transfer_id == independent.transfer_id
    assert actual.evidence.evidence_id == independent.evidence.evidence_id
    np.testing.assert_allclose(
        np.asarray(actual.transfer.apply(_coordinate_field(coarse, source))),
        _coordinate_field(fine, refined.target),
        rtol=0.0,
        atol=_RESTRICTION_ROUNDING,
    )


def test_prepared_transition_uses_live_budget_for_structural_endpoint_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.solver import (
        FiniteElementAcceptedState,
        FiniteElementTopologyTransaction,
    )
    from phydrax.solver._finite_element_adaptivity import (
        PreparedFiniteElementTopologyTransition,
    )

    source, refined = _refined((0, 4))
    with coordinate_reference_budget() as ledger:
        coarse = _space(source, budget=ledger)
        fine = _space(refined.target, budget=ledger)
        fields = coarse.construction_plan.fields
        values = tuple(
            np.ones(dofs.global_dof_count, dtype=np.float64)
            for dofs in coarse.dof_maps
        )
        accepted = FiniteElementAcceptedState(
            values,
            0.0,
            0,
            source.mesh.topology_id,
            coarse.prepared_id,
            "authored-structural-endpoint-admission-test",
        )

        def finite_fields(
            mesh: Any, arrays: Any, materials: Any, lineage: Any, context: Any
        ) -> bool:
            del mesh, materials, lineage, context
            return all(np.all(np.isfinite(array)) for array in arrays)

        transaction = FiniteElementTopologyTransaction(finite_fields, fields=fields)
        original_resolve = CellGeometrySpec.resolve
        resolved: list[CellGeometrySpec] = []

        def resolve_with_actual_budget(
            geometry: CellGeometrySpec, mesh: CellMesh, /, **kwargs: Any
        ) -> Any:
            assert kwargs.get("budget") is ledger
            resolved.append(geometry)
            return original_resolve(geometry, mesh, **kwargs)

        def refuse_fresh_ticket(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("live structural admission must not open a host ticket")

        before = ledger.work_units
        with monkeypatch.context() as patch:
            patch.setattr(CellGeometrySpec, "resolve", resolve_with_actual_budget)
            patch.setattr(HostBatchTicket, "__init__", refuse_fresh_ticket)
            prepared = transaction.prepare_transition(
                accepted,
                source.mesh,
                refined,
                source=coarse,
                target=fine,
                budget=ledger,
            )
        assert ledger.work_units > before
        assert any(geometry is source.geometry for geometry in resolved)
        assert any(geometry is refined.target.geometry for geometry in resolved)
        assert prepared.source is coarse and prepared.target is fine
        assert all(transfer.evidence.passed for transfer in prepared.transfers)
        assert "budget" not in {
            field.name for field in dataclasses.fields(PreparedFiniteElementTopologyTransition)
        }


def test_foreign_preparation_imports_and_refuses_rebuilt_or_stale_records() -> None:
    source, refined = _refined((0, 4))
    coarse, fine = _space(source), _space(refined.target)
    witnesses = _witnesses(source, refined)
    _, reference = prepare_shared_nested_field_transfer(
        coarse, fine, refined.parent_cells, field_name="u", **witnesses
    )
    assert reference is not None
    with coordinate_reference_budget() as ledger:
        before = ledger.work_units
        phx.discretization.prepare_nested_field_transfer(
            coarse,
            fine,
            refined.parent_cells,
            field_name="w",
            nested_reference=reference,
            budget=ledger,
            **witnesses,
        )
        # Sharing removes repeated work but never hides its completed cost.
        assert ledger.work_units - before >= reference.work_units
    # Rebuilt maps, bounds or charges never reach a consumer route.
    shifted = tuple(
        pair._replace(offset=pair.offset + 0.25) for pair in reference.affine_pieces()
    )
    for changes in ({"pairs": shifted}, {"work_units": 0, "storage_bytes": 0}):
        rebuilt = dataclasses.replace(
            reference, issuer=_NESTED_REFERENCE_ISSUER, **changes
        )
        with pytest.raises(ValueError, match=r"PreparedNestedReference\.provenance"):
            phx.discretization.prepare_nested_field_transfer(
                coarse,
                fine,
                refined.parent_cells,
                field_name="w",
                nested_reference=rebuilt,
                **witnesses,
            )
    lookalike = SimpleNamespace(serves=lambda *_, **__: True)
    with pytest.raises(TypeError):
        prepare_shared_nested_field_transfer(
            coarse,
            fine,
            refined.parent_cells,
            field_name="w",
            prepared=(lookalike,),
            **witnesses,
        )
    other_source, other = _refined((1,))
    with pytest.raises(ValueError, match=r"PreparedNestedReference\.binding"):
        phx.discretization.prepare_nested_field_transfer(
            _space(other_source),
            _space(other.target),
            other.parent_cells,
            field_name="u",
            nested_reference=reference,
            **_witnesses(other_source, other),
        )
