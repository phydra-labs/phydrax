"""Consumer contracts for periodic reconstruction and keyed identity admission."""

import equinox as eqx
import numpy as np
import pytest

from phydrax.discretization import CellMesh, PeriodicCell, PeriodicMeshTopology
from phydrax.meshing import PeriodicQuotientEvidence
from phydrax.meshing._periodic import require_periodic_identity_bank
from phydrax.meshing._topology_edit import PeriodicEntityIdentityBank
from tests._support.cell_meshes import unit_square


def _periodic_axis(mesh: CellMesh, axis: int) -> CellMesh:
    """Identify the two opposite boundaries of the selected unit-square axis."""
    points = np.asarray(mesh.coordinates)
    exponents = np.zeros(points.shape, dtype=np.int64)
    exponents[:, axis] = np.floor(points[:, axis]).astype(np.int64)
    reduced = points - exponents
    rows = {tuple(point): row for row, point in enumerate(points)}
    representatives = np.asarray(
        [rows[tuple(point)] for point in reduced], dtype=np.int64
    )
    cell = PeriodicCell(np.eye(2, dtype=np.float64), periodic_axes=(axis == 0, axis == 1))
    topology = PeriodicMeshTopology(mesh, cell, representatives, exponents)
    return CellMesh(mesh.coordinates, mesh.blocks, periodic_topology=topology)


@pytest.fixture
def periodic_mesh() -> CellMesh:
    return _periodic_axis(unit_square(2), 0)


def test_quotient_evidence_reconstruction_preserves_all_fields(
    periodic_mesh: CellMesh,
) -> None:
    evidence = PeriodicQuotientEvidence(periodic_mesh)
    restored = PeriodicQuotientEvidence.from_stored_fields(evidence, mesh=periodic_mesh)
    assert restored is not evidence
    assert eqx.tree_equal(restored, evidence, typematch=True)


def test_quotient_evidence_reconstruction_refuses_changed_group_binding(
    periodic_mesh: CellMesh,
) -> None:
    evidence = PeriodicQuotientEvidence(periodic_mesh)
    different_group = _periodic_axis(unit_square(2), 1)
    with pytest.raises(ValueError, match="PeriodicQuotientEvidence.periodic_topology_id"):
        PeriodicQuotientEvidence.from_stored_fields(evidence, mesh=different_group)


def test_quotient_evidence_reconstruction_refuses_other_mesh_with_equal_quotient(
    periodic_mesh: CellMesh,
) -> None:
    # Moving an interior vertex keeps the quotient group, counts and measure,
    # so only the bound mesh identity can reject this foreign carrier.
    base = unit_square(2)
    points = np.asarray(base.coordinates).copy()
    points[4, 0] = 0.375
    moved = _periodic_axis(CellMesh(points, base.blocks), 0)
    evidence = PeriodicQuotientEvidence(periodic_mesh)
    assert PeriodicQuotientEvidence(moved).total_measure == pytest.approx(
        evidence.total_measure
    )
    with pytest.raises(ValueError, match="PeriodicQuotientEvidence.mesh_id"):
        PeriodicQuotientEvidence.from_stored_fields(evidence, mesh=moved)


def test_periodic_identity_admission_preserves_variable_arity_keys() -> None:
    keys = ((10, 20), (30, 40, 50))
    identifiers = (5, 6)
    bank = PeriodicEntityIdentityBank(1, keys, identifiers, 7)
    require_periodic_identity_bank(bank)
    assert bank.entity_keys is keys
    assert bank.entity_global_ids is identifiers
    assert bank.allocator_next_id == 7


@pytest.mark.parametrize(
    "bank",
    (
        PeriodicEntityIdentityBank(1, ((10, 20), (10, 20)), (5, 6), 7),
        PeriodicEntityIdentityBank(1, ((10, 20), (30, 40)), (5, 5), 7),
        PeriodicEntityIdentityBank(1, ((10, 20), (30, 40)), (5, 7), 7),
    ),
    ids=("duplicate-key", "reused-id", "cursor-collision"),
)
def test_periodic_identity_admission_refuses_invalid_allocation(
    bank: PeriodicEntityIdentityBank,
) -> None:
    with pytest.raises(ValueError, match=r"require_periodic_identity_bank\.bank"):
        require_periodic_identity_bank(bank)
