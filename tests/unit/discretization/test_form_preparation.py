"""White-box reference preparation reuse, factor admission, and singular solves."""

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh, form_element, FormBasis
from phydrax.discretization.fem._assembly import (
    canonical_form_entity_bases,
    topology_vertex_sets,
)
from phydrax.discretization.fem._form_elements import HybridFactors
from phydrax.discretization.fem._hybrid_forms import solve_hybrid_dual


def test_prism_entity_owners_reuse_reference_trace_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 2.0),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "wedges",
                "prism",
                np.asarray(((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)), dtype=np.int32),
            ),
        ),
    )
    element = form_element("prism", 1, 2, proxy="circulation", twist="untwisted")
    basis = element.form_basis
    assert isinstance(basis, FormBasis)
    levels = topology_vertex_sets(mesh)
    lookups = tuple(
        {vertices: index for index, vertices in enumerate(level)} for level in levels
    )
    original = FormBasis.entity_basis
    calls = []

    def counted(self: FormBasis, face: tuple[int, ...], /) -> FormBasis:
        calls.append((self.basis_id, face))
        return original(self, face)

    monkeypatch.setattr(FormBasis, "entity_basis", counted)
    owners = canonical_form_entity_bases(mesh, (element,), lookups)
    expected_kinds = {
        basis.entity_kind(face)
        for dimension in (1, 2)
        for face, dofs in zip(
            basis.entity_vertices[dimension],
            element.entity_dofs[dimension],
            strict=True,
        )
        if dofs
    }
    # Immutable reference traces are prepared once per distinct reference kind,
    # then reused while each mesh entity retains its reference basis.
    assert len(calls) == len(expected_kinds)
    assert {basis.entity_kind(face) for _, face in calls} == expected_kinds
    expected = [[None] * len(level) for level in levels]
    for dimension in (1, 2):
        for cell in np.asarray(mesh.blocks[0].vertices):
            for face, dofs in zip(
                basis.entity_vertices[dimension],
                element.entity_dofs[dimension],
                strict=True,
            ):
                if not dofs:
                    continue
                trace = original(basis, face)
                row = lookups[dimension][tuple(sorted(cell[list(face)]))]
                previous = expected[dimension][row]
                if previous is None or trace.basis_id < previous:
                    expected[dimension][row] = trace.basis_id
    assert tuple(
        tuple(None if owner is None else owner.basis_id for owner in level)
        for level in owners
    ) == tuple(tuple(level) for level in expected)


def test_hybrid_dual_refuses_singular_functional_matrix() -> None:
    with pytest.raises(ValueError, match="Singular matrix"):
        solve_hybrid_dual(np.asarray(((1.0, 2.0), (2.0, 4.0)), dtype=np.float64))


def test_hybrid_factor_refuses_excessive_condition() -> None:
    basis = form_element("pyramid", 1, 1, proxy="circulation").form_basis
    assert basis is not None and basis.hybrid_factors is not None
    factors = basis.hybrid_factors
    with pytest.raises(ValueError, match="hybrid_factors.source_bank"):
        HybridFactors(
            factors.source_bank,
            factors.generators,
            1e13,
            0.0,
            factors.dual_status,
            factors.body_test_exponents,
            factors.body_test_source_bank,
            factors.body_test_coefficients,
        )
