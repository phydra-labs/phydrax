#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.relativistic_scattering._matrix_element_revision import (
    MatrixElementAdaptationEvidence,
    MatrixElementAdaptationProposal,
    MatrixElementRevision,
    MatrixElementWeightSnapshot,
    resolve_matrix_element_adaptation,
)


def _digest(name):
    return canonical_fingerprint({"test": name})


def _revision(parameter, *, training="training", rights="native-commercial"):
    return MatrixElementRevision(
        "dark-fermion-annihilation",
        "vector-mediator-tree",
        "phydrax-native",
        rights,
        "covariant-2E",
        "two-to-two-massive",
        "multi-channel",
        "held-out-gated",
        training,
        "adam",
        "relative-cross-section-error",
        _digest(parameter),
        differentiation_id="analytic-native",
    )


def test_revision_identity_binds_every_scientific_and_provenance_contract():
    base = _revision("parameters-a")
    repeated = MatrixElementRevision.from_record(base.to_record())
    assert repeated.revision_id == base.revision_id
    assert _revision("parameters-b").revision_id != base.revision_id
    assert (
        _revision("parameters-a", rights="research-only").revision_id != base.revision_id
    )
    assert not base.is_adapted


def test_adaptation_accept_reject_and_replay_are_epoch_bound_and_exact():
    boundary = _digest("committed-epoch")
    base = _revision("parameters-a")
    candidate = _revision("parameters-b")
    proposal = MatrixElementAdaptationProposal(
        base.revision_id,
        candidate.revision_id,
        candidate.training_data_id,
        candidate.optimizer_id,
        candidate.adaptation_id,
        boundary,
    )
    accepted_evidence = MatrixElementAdaptationEvidence(
        proposal.proposal_record_id,
        "held-out",
        "relative-error",
        0.20,
        0.12,
        0.01,
        boundary,
    )
    accepted = resolve_matrix_element_adaptation(
        base,
        candidate,
        proposal,
        accepted_evidence,
        after_epoch_manifest_id=boundary,
    )
    replayed = resolve_matrix_element_adaptation(
        base,
        candidate,
        proposal,
        accepted_evidence,
        after_epoch_manifest_id=boundary,
    )
    proposal_replay = MatrixElementAdaptationProposal.from_record(proposal.to_record())
    evidence_replay = MatrixElementAdaptationEvidence.from_record(
        accepted_evidence.to_record()
    )
    decision_replay = type(accepted).from_record(accepted.to_record())
    assert proposal_replay.proposal_record_id == proposal.proposal_record_id
    assert evidence_replay.evidence_id == accepted_evidence.evidence_id
    assert decision_replay.decision_id == accepted.decision_id
    assert accepted.accepted
    assert accepted.decision_id == replayed.decision_id
    assert accepted.active_revision.parent_revision_id == base.revision_id
    assert accepted.active_revision.held_out_evidence_id == accepted_evidence.evidence_id
    assert accepted.active_revision.valid_from_epoch_manifest_id == boundary

    rejected_evidence = MatrixElementAdaptationEvidence(
        proposal.proposal_record_id,
        "held-out",
        "relative-error",
        0.20,
        0.21,
        0.01,
        boundary,
    )
    rejected = resolve_matrix_element_adaptation(
        base,
        candidate,
        proposal,
        rejected_evidence,
        after_epoch_manifest_id=boundary,
    )
    assert not rejected.accepted
    assert rejected.active_revision is base
    with pytest.raises(ValueError, match="disjoint"):
        bad_evidence = MatrixElementAdaptationEvidence(
            proposal.proposal_record_id,
            candidate.training_data_id,
            "relative-error",
            0.20,
            0.12,
            0.01,
            boundary,
        )
        resolve_matrix_element_adaptation(
            base,
            candidate,
            proposal,
            bad_evidence,
            after_epoch_manifest_id=boundary,
        )


def test_prior_event_weights_remain_revision_bound_after_adaptation():
    base = _revision("parameters-a")
    snapshot = MatrixElementWeightSnapshot(
        (_digest("event-a"), _digest("event-b")),
        (1.0, -0.25),
        base.revision_id,
        _digest("epoch-a"),
    )
    restored = MatrixElementWeightSnapshot.from_record(snapshot.to_record())
    assert restored.weight_snapshot_id == snapshot.weight_snapshot_id
    before = snapshot.weight_snapshot_id
    candidate = _revision("parameters-b")
    assert snapshot.weights == (1.0, -0.25)
    assert snapshot.matrix_element_revision_id == base.revision_id
    assert snapshot.weight_snapshot_id == before
    assert snapshot.matrix_element_revision_id != candidate.revision_id
