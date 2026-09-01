#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.operators.integral.layer_potential._qualification import (
    BoundaryOperationalEvidence,
    BoundaryProductProvenance,
    BoundaryQualificationEvidence,
    BoundarySupportEnvelope,
)


def _support(
    *, unsupported_claims: dict[str, str] | None = None
) -> BoundarySupportEnvelope:
    return BoundarySupportEnvelope(
        geometry_id="closed-oriented-triangle-mesh",
        trace_id="triangle-dp0",
        pde_formulation_id="laplace-single-layer-dirichlet",
        provider_id="direct-blocked-reference",
        precision_id="float64-accumulate-float64",
        differentiation_id="none",
        platform_id="cpu-posix",
        claims=("operator-action", "finite-execution", "continuum-error"),
        unsupported_claims={} if unsupported_claims is None else unsupported_claims,
        stop_ship_conditions=("resource-preflight-failed",),
    )


def _operational(**overrides: object) -> BoundaryOperationalEvidence:
    arguments: dict[str, object] = {
        "plan_id": "plan:assemble",
        "result_id": "result:assemble",
        "parent_plan_ids": ("plan:geometry",),
        "parent_result_ids": ("result:geometry",),
        "provider_id": "direct-blocked-reference",
        "provider_deterministic": True,
        "security_preflight_passed": True,
        "security_preflight_id": "security:local-native-provider",
        "resource_preflight_passed": True,
        "resource_preflight_id": "resource:bounded-block-action",
        "resource_limit_bytes": 4096,
        "forecast_bytes": 2048,
        "observed_bytes": 2304,
    }
    arguments.update(overrides)
    return BoundaryOperationalEvidence(**arguments)  # type: ignore[arg-type]


def _provenance(**overrides: object) -> BoundaryProductProvenance:
    arguments: dict[str, object] = {
        "product_id": "product:weak-operator",
        "producer_id": "phydrax.boundary",
        "producer_version": "native",
        "provider_id": "direct-blocked-reference",
        "source_kind": "adapted",
        "source_id": "source:singular-transform",
        "source_content_id": "sha256:source-content",
        "license_id": "MIT",
        "clean_room_record_id": "record:reviewed-origin-and-adaptation",
        "parent_product_ids": ("product:surface",),
        "parent_plan_ids": ("plan:assemble",),
        "parent_result_ids": ("result:assemble",),
    }
    arguments.update(overrides)
    return BoundaryProductProvenance(**arguments)  # type: ignore[arg-type]


def test_continuum_certification_requires_prerequisite_evidence() -> None:
    support = _support()
    provenance = _provenance()
    operational = _operational()

    with pytest.raises(ValueError, match="requires prerequisite evidence"):
        BoundaryQualificationEvidence(
            support,
            claim="continuum-error",
            level="continuum-certified",
            maturity="Q1",
            supported=True,
            error_bound=0.0,
            error_metric="relative-capacitance-error",
            provenance_id=provenance.provenance_id,
            operational_evidence_id=operational.operational_id,
            evidence_artifact_ids=("artifact:continuum-proof",),
        )

    evidence = BoundaryQualificationEvidence(
        support,
        claim="continuum-error",
        level="continuum-certified",
        maturity="Q1",
        supported=True,
        error_bound=0.0,
        error_metric="relative-capacitance-error",
        provenance_id=provenance.provenance_id,
        operational_evidence_id=operational.operational_id,
        evidence_artifact_ids=("artifact:continuum-proof",),
        prerequisite_evidence_ids=("evidence:continuum-qualified",),
    )
    assert evidence.level == "continuum-certified"
    assert evidence.maturity == "Q1"


def test_missing_provenance_and_unknown_claims_are_rejected() -> None:
    with pytest.raises(ValueError, match="license_id must be non-empty"):
        _provenance(license_id="   ")

    support = _support()
    with pytest.raises(ValueError, match="Unknown boundary qualification claim"):
        BoundaryQualificationEvidence(
            support,
            claim="unregistered-capability",
            level="computed",
            maturity="Q0",
            supported=True,
            provenance_id="provenance:fixture",
            operational_evidence_id="operation:fixture",
            evidence_artifact_ids=("artifact:fixture",),
        )

    with pytest.raises(ValueError, match="provenance_id must be non-empty"):
        BoundaryQualificationEvidence(
            support,
            claim="finite-execution",
            level="computed",
            maturity="Q0",
            supported=True,
            provenance_id="",
            operational_evidence_id="operation:fixture",
            evidence_artifact_ids=("artifact:fixture",),
        )


def test_unsupported_is_explicit_and_not_zero_error() -> None:
    support = _support(
        unsupported_claims={
            "continuum-error": "No continuum discretization estimator is implemented."
        }
    )
    evidence = BoundaryQualificationEvidence(
        support,
        claim="continuum-error",
        level="computed",
        maturity="Q0",
        supported=False,
        unsupported_reason="Provider reports this capability as unsupported.",
        provenance_id="provenance:fixture",
        operational_evidence_id="operation:fixture",
        evidence_artifact_ids=("artifact:unsupported-declaration",),
    )

    assert evidence.supported is False
    assert evidence.error_bound is None
    assert evidence.unsupported_reason is not None

    with pytest.raises(ValueError, match="not a numerical zero"):
        BoundaryQualificationEvidence(
            support,
            claim="continuum-error",
            level="computed",
            maturity="Q0",
            supported=False,
            error_bound=0.0,
            error_metric="relative-error",
            unsupported_reason="Provider reports this capability as unsupported.",
            provenance_id="provenance:fixture",
            operational_evidence_id="operation:fixture",
            evidence_artifact_ids=("artifact:unsupported-declaration",),
        )


def test_fingerprints_are_deterministic_under_set_like_input_order() -> None:
    first = BoundarySupportEnvelope(
        geometry_id="geometry",
        trace_id="trace",
        pde_formulation_id="pde/formulation",
        provider_id="provider",
        precision_id="precision",
        differentiation_id="differentiation",
        platform_id="platform",
        claims=("b", "a"),
        unsupported_claims=(("b", "not provided"),),
        stop_ship_conditions=("security", "resource"),
    )
    second = BoundarySupportEnvelope(
        geometry_id="geometry",
        trace_id="trace",
        pde_formulation_id="pde/formulation",
        provider_id="provider",
        precision_id="precision",
        differentiation_id="differentiation",
        platform_id="platform",
        claims=("a", "b"),
        unsupported_claims={"b": "not provided"},
        stop_ship_conditions=("resource", "security"),
    )
    assert first.envelope_id == second.envelope_id
    with pytest.raises(AttributeError):
        first.platform_id = "other"  # type: ignore[misc]

    provenance_first = _provenance(
        parent_product_ids=("product:b", "product:a"),
        parent_plan_ids=("plan:b", "plan:a"),
        parent_result_ids=("result:b", "result:a"),
    )
    provenance_second = _provenance(
        parent_product_ids=("product:a", "product:b"),
        parent_plan_ids=("plan:a", "plan:b"),
        parent_result_ids=("result:a", "result:b"),
    )
    assert provenance_first.provenance_id == provenance_second.provenance_id

    operational = _operational()
    evidence_first = BoundaryQualificationEvidence(
        first,
        claim="a",
        level="checked-discrete",
        maturity="Q1",
        supported=True,
        error_bound=0.0,
        error_metric="action-parity",
        provenance_id=provenance_first.provenance_id,
        operational_evidence_id=operational.operational_id,
        evidence_artifact_ids=("artifact:b", "artifact:a"),
        prerequisite_evidence_ids=("evidence:b", "evidence:a"),
    )
    evidence_second = BoundaryQualificationEvidence(
        second,
        claim="a",
        level="checked-discrete",
        maturity="Q1",
        supported=True,
        error_bound=0.0,
        error_metric="action-parity",
        provenance_id=provenance_second.provenance_id,
        operational_evidence_id=operational.operational_id,
        evidence_artifact_ids=("artifact:a", "artifact:b"),
        prerequisite_evidence_ids=("evidence:a", "evidence:b"),
    )
    assert evidence_first.evidence_id == evidence_second.evidence_id


def test_resource_observation_requires_a_result_and_valid_byte_counts() -> None:
    with pytest.raises(ValueError, match="observed_bytes requires a result_id"):
        _operational(result_id=None, observed_bytes=1)

    with pytest.raises(ValueError, match="nonnegative"):
        _operational(observed_bytes=-1)

    with pytest.raises(ValueError, match="limit violation"):
        _operational(observed_bytes=4097)

    exceeded = _operational(
        observed_bytes=4097,
        stop_ship_reasons=("observed-resource-limit-exceeded",),
    )
    assert exceeded.observed_bytes == 4097
    assert exceeded.stop_ship_reasons == ("observed-resource-limit-exceeded",)

    with pytest.raises(ValueError, match="failed preflight requires a stop-ship reason"):
        _operational(
            result_id=None,
            observed_bytes=None,
            security_preflight_passed=False,
        )
