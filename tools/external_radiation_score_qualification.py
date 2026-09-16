#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Offline audit records for already imported external radiation scores.

This module neither launches providers nor upgrades imported scores to clinical or
scientific qualification. It records whether the governed import boundary retained
its exact profile, artifacts, geometry, semantics, losses, and estimator evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from phydrax.applications.radiation_biophysics._scores import (
    ExternalRadiationScoreResult,
)


def external_radiation_score_qualification(
    scores: Sequence[ExternalRadiationScoreResult], /
) -> dict[str, object]:
    """Return deterministic research-only audit evidence for imported scores."""

    if isinstance(scores, ExternalRadiationScoreResult):
        raise TypeError("scores must be a sequence, not one result.")
    values = tuple(scores)
    if not values or any(
        not isinstance(value, ExternalRadiationScoreResult) for value in values
    ):
        raise TypeError("scores must contain at least one ExternalRadiationScoreResult.")
    identifiers = tuple(value.score_id for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Qualification inputs must have unique score identities.")
    records: list[dict[str, object]] = []
    for value in values:
        value.require_rights()
        records.append(
            {
                "score_id": value.score_id,
                "profile_id": value.profile_id,
                "report_id": value.report.report_id,
                "run_identity_id": value.run_identity.identity_id,
                "artifact_manifest_ids": [
                    reference.manifest_id for reference in value.artifact_references
                ],
                "governing_manifest_ids": [
                    reference.manifest_id for reference in value.references
                ],
                "quantity_kind": value.definition.quantity_kind.value,
                "quantity_id": value.definition.quantity.quantity_id,
                "unit_id": value.definition.unit.unit_id,
                "normalization": value.definition.normalization,
                "representation": value.definition.representation,
                "shape": list(value.definition.shape),
                "dtype": value.definition.dtype,
                "grid_affine_id": value.definition.grid_affine.affine_id,
                "estimator_evidence_id": value.estimator_evidence.evidence_id,
                "uncertainty_kind": value.estimator_evidence.uncertainty_kind,
                "correlation_model": value.estimator_evidence.correlation_model,
                "declared_loss_ids": [loss.loss_id for loss in value.declared_losses],
                "research_only": value.research_only,
                "adapter_valid": value.report.valid,
            }
        )
    checks = {
        "checksums_and_rights_admitted": True,
        "exact_profiles_pinned": all(record["profile_id"] for record in records),
        "grid_geometry_pinned": all(record["grid_affine_id"] for record in records),
        "score_semantics_explicit": all(
            record["quantity_kind"]
            and record["unit_id"]
            and record["normalization"]
            and record["representation"]
            for record in records
        ),
        "uncertainty_correlation_not_fabricated": all(
            record["uncertainty_kind"] == "unreported"
            or record["correlation_model"] not in ("unreported", "assumed-independent")
            for record in records
        ),
        "research_only_scope": all(record["research_only"] for record in records),
    }
    return {
        "kind": "external-radiation-score-import-qualification",
        "scope": "research-only",
        "clinical_use_permitted": False,
        "provider_execution_performed": False,
        "scientifically_qualified": False,
        "checks": checks,
        "successful": all(checks.values()),
        "scores": records,
    }


__all__ = ["external_radiation_score_qualification"]
