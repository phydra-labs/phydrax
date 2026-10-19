# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Replay transfer evidence without changing quantities or acceptance."""

from dataclasses import fields

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._model._structure import (
    _restore_dataclass,
    model_from_logical_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization.fem import (
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
    vertex_interpolation_transfer,
)
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts


def _stored_evidence(
    value: FiniteElementTransferEvidence,
    *,
    stored_id: str,
    stored_passed: bool,
) -> FiniteElementTransferEvidence:
    """Replay the registered dataclass fields through the artifact boundary."""
    stored = {
        "defects": value.defects,
        "estimates": value.estimates,
        "bounds": value.bounds,
        "tolerance": value.tolerance,
        "passed": stored_passed,
        "evidence_id": stored_id,
        "execution_evidence": value.execution_evidence,
    }
    ordered = [stored[field.name] for field in fields(FiniteElementTransferEvidence)]
    return _restore_dataclass(
        FiniteElementTransferEvidence, list(range(len(ordered))), ordered
    )


@pytest.fixture(scope="module")
def evidence() -> FiniteElementTransferEvidence:
    register_meshing_source_artifacts()
    # Dyadic quantities make the identity checks independent of rounding tolerances.
    return FiniteElementTransferEvidence(
        {"constants": 0.0, "content": 0.125},
        0.25,
        estimates={"mass_condition": 2.0},
        bounds={"content": 0.0625},
    )


def test_transfer_evidence_restore_replays_quantities(
    evidence: FiniteElementTransferEvidence,
) -> None:
    restored = FiniteElementTransferEvidence.from_stored_fields(evidence)
    assert restored is not evidence
    assert restored.evidence_id == evidence.evidence_id
    assert restored.passed == evidence.passed
    assert restored.defects == evidence.defects
    assert restored.estimates == evidence.estimates
    assert restored.bounds == evidence.bounds
    assert restored.tolerance == evidence.tolerance
    assert restored.execution_evidence is evidence.execution_evidence


def test_transfer_evidence_restore_refuses_changed_identity(
    evidence: FiniteElementTransferEvidence,
) -> None:
    corrupted = _stored_evidence(
        evidence, stored_id="changed", stored_passed=evidence.passed
    )
    with pytest.raises(ValueError, match="FiniteElementTransferEvidence.stored_fields"):
        FiniteElementTransferEvidence.from_stored_fields(corrupted)


def test_transfer_evidence_restore_refuses_changed_acceptance(
    evidence: FiniteElementTransferEvidence,
) -> None:
    corrupted = _stored_evidence(
        evidence, stored_id=evidence.evidence_id, stored_passed=False
    )
    with pytest.raises(ValueError, match="FiniteElementTransferEvidence.stored_fields"):
        FiniteElementTransferEvidence.from_stored_fields(corrupted)


@pytest.fixture(scope="module")
def claim_transfer() -> FiniteElementTopologyTransfer:
    register_meshing_source_artifacts()
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.5)), dtype=np.float64)
    measures = np.asarray((0.25, 0.75), dtype=np.float64)
    return vertex_interpolation_transfer(
        np.asarray(((0,), (1,)), dtype=np.int32),
        np.ones((2, 1), dtype=np.float64),
        np.ones((2, 1), dtype=np.bool_),
        source_size=2,
        source_topology_id="source",
        target_topology_id="target",
        preserves_linear=True,
        conservative=True,
        source_coordinates=coordinates,
        target_coordinates=coordinates,
        source_measures=measures,
        target_measures=measures,
    )


def test_topology_transfer_restore_recertifies_retained_claim_inputs(
    claim_transfer: FiniteElementTopologyTransfer,
) -> None:
    recipe = model_structure_recipe(claim_transfer)
    arrays = model_recipe_array_values(claim_transfer, recipe, prefix="transfer")
    archived = model_from_logical_array_recipe(recipe, arrays, prefix="transfer")
    assert isinstance(archived, FiniteElementTopologyTransfer)
    restored = reconstruct_stored_fields(archived)
    assert tree_content_equal(restored, claim_transfer)
    assert restored.transfer_id == claim_transfer.transfer_id
    assert restored.preserves_linear and restored.conservative
    np.testing.assert_array_equal(
        restored.apply(restored.source_coordinates), restored.target_coordinates
    )


@pytest.mark.parametrize(
    "field",
    ("source_coordinates", "target_coordinates", "source_measures", "target_measures"),
)
def test_topology_transfer_restore_refuses_tampered_claim_inputs(
    claim_transfer: FiniteElementTopologyTransfer, field: str
) -> None:
    value = getattr(claim_transfer, field)
    altered = eqx.tree_at(
        lambda transfer: getattr(transfer, field), claim_transfer, value + 0.25
    )
    with pytest.raises(ValueError, match="certify_claims.configuration"):
        reconstruct_stored_fields(altered)


@pytest.mark.parametrize("claim", ("preserves_linear", "conservative"))
def test_topology_transfer_claim_requires_certification_inputs(
    claim_transfer: FiniteElementTopologyTransfer, claim: str
) -> None:
    with pytest.raises(ValueError, match="certify_claims.configuration must provide"):
        FiniteElementTopologyTransfer(
            claim_transfer.primal,
            "source",
            "target",
            preserves_linear=claim == "preserves_linear",
            conservative=claim == "conservative",
        )


def test_topology_transfer_without_claims_retains_no_certification_arrays(
    claim_transfer: FiniteElementTopologyTransfer,
) -> None:
    transfer = FiniteElementTopologyTransfer(
        claim_transfer.primal,
        "source",
        "target",
        source_coordinates=claim_transfer.source_coordinates,
        target_coordinates=claim_transfer.target_coordinates,
        source_measures=claim_transfer.source_measures,
        target_measures=claim_transfer.target_measures,
    )
    assert transfer.source_coordinates is None
    assert transfer.target_coordinates is None
    assert transfer.source_measures is None
    assert transfer.target_measures is None
    assert tree_content_equal(reconstruct_stored_fields(transfer), transfer)
