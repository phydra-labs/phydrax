#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import equinox as eqx

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


BoundaryEvidenceLevel: TypeAlias = Literal[
    "computed",
    "checked-discrete",
    "quadrature-supported",
    "continuum-qualified",
    "continuum-certified",
]
BoundaryMaturity: TypeAlias = Literal["Q0", "Q1", "Q2", "Q3"]
BoundaryProvenanceSource: TypeAlias = Literal[
    "native", "clean-room", "adapted", "external"
]

_EVIDENCE_LEVELS = (
    "computed",
    "checked-discrete",
    "quadrature-supported",
    "continuum-qualified",
    "continuum-certified",
)
_MATURITY_LEVELS = ("Q0", "Q1", "Q2", "Q3")
_PROVENANCE_SOURCES = ("native", "clean-room", "adapted", "external")
_MAX_ITEMS = 256
_MAX_TEXT_LENGTH = 4096
_MAX_BYTES = (1 << 63) - 1


def _required_text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    if len(resolved) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{name} exceeds the {_MAX_TEXT_LENGTH}-character limit.")
    return resolved


def _optional_text(value: str | None, name: str, /) -> str | None:
    return None if value is None else _required_text(value, name)


def _canonical_strings(
    values: Sequence[str], name: str, /, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not one string.")
    if len(values) > _MAX_ITEMS:
        raise ValueError(f"{name} cannot contain more than {_MAX_ITEMS} values.")
    resolved = tuple(_required_text(value, name) for value in values)
    if not allow_empty and not resolved:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate values.")
    return tuple(sorted(resolved))


def _parent_ids(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    return _canonical_strings(values, name)


def _bounded_bytes(value: int, name: str, /, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count.")
    lower = 1 if positive else 0
    if value < lower or value > _MAX_BYTES:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} signed 64-bit byte count.")
    return value


def _strict_bool(value: bool, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool.")
    return value


class BoundarySupportEnvelope(StrictModule, NonTrainableState):
    """One exact, bounded boundary-platform support tuple and its declared claims."""

    geometry_id: str = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)
    pde_formulation_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    differentiation_id: str = eqx.field(static=True)
    platform_id: str = eqx.field(static=True)
    claims: tuple[str, ...] = eqx.field(static=True)
    unsupported_claims: tuple[tuple[str, str], ...] = eqx.field(static=True)
    stop_ship_conditions: tuple[str, ...] = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_id: str,
        trace_id: str,
        pde_formulation_id: str,
        provider_id: str,
        precision_id: str,
        differentiation_id: str,
        platform_id: str,
        claims: Sequence[str],
        unsupported_claims: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        stop_ship_conditions: Sequence[str] = (),
    ):
        dimensions = tuple(
            _required_text(value, name)
            for value, name in (
                (geometry_id, "geometry_id"),
                (trace_id, "trace_id"),
                (pde_formulation_id, "pde_formulation_id"),
                (provider_id, "provider_id"),
                (precision_id, "precision_id"),
                (differentiation_id, "differentiation_id"),
                (platform_id, "platform_id"),
            )
        )
        claims_ = _canonical_strings(claims, "claims", allow_empty=False)
        if len(unsupported_claims) > _MAX_ITEMS:
            raise ValueError(
                f"unsupported_claims cannot contain more than {_MAX_ITEMS} values."
            )
        entries = (
            tuple(unsupported_claims.items())
            if isinstance(unsupported_claims, Mapping)
            else tuple(unsupported_claims)
        )
        normalized_entries: list[tuple[str, str]] = []
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("unsupported_claims must contain (claim, reason) pairs.")
            claim = _required_text(entry[0], "unsupported claim")
            reason = _required_text(entry[1], "unsupported reason")
            if claim not in claims_:
                raise ValueError(
                    f"Unsupported claim {claim!r} is not declared by this envelope."
                )
            normalized_entries.append((claim, reason))
        unsupported_ = tuple(sorted(normalized_entries))
        if len({claim for claim, _ in unsupported_}) != len(unsupported_):
            raise ValueError("unsupported_claims must not repeat a claim.")
        stop_ship_ = _canonical_strings(stop_ship_conditions, "stop_ship_conditions")
        (
            self.geometry_id,
            self.trace_id,
            self.pde_formulation_id,
            self.provider_id,
            self.precision_id,
            self.differentiation_id,
            self.platform_id,
        ) = dimensions
        self.claims = claims_
        self.unsupported_claims = unsupported_
        self.stop_ship_conditions = stop_ship_
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "boundary-support-envelope",
                "geometry": dimensions[0],
                "trace": dimensions[1],
                "pde_formulation": dimensions[2],
                "provider": dimensions[3],
                "precision": dimensions[4],
                "differentiation": dimensions[5],
                "platform": dimensions[6],
                "claims": list(claims_),
                "unsupported_claims": [
                    {"claim": claim, "reason": reason} for claim, reason in unsupported_
                ],
                "stop_ship_conditions": list(stop_ship_),
            }
        )

    def supports(self, claim: str, /) -> bool:
        """Return declared support, rejecting claims outside this exact envelope."""
        claim_ = _required_text(claim, "claim")
        if claim_ not in self.claims:
            raise ValueError(f"Unknown boundary qualification claim {claim_!r}.")
        return claim_ not in {name for name, _ in self.unsupported_claims}


class BoundaryQualificationEvidence(StrictModule, NonTrainableState):
    """Content-addressed evidence for one claim in one exact support envelope."""

    support: BoundarySupportEnvelope
    claim: str = eqx.field(static=True)
    level: BoundaryEvidenceLevel = eqx.field(static=True)
    maturity: BoundaryMaturity = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    error_bound: float | None = eqx.field(static=True)
    error_metric: str | None = eqx.field(static=True)
    unsupported_reason: str | None = eqx.field(static=True)
    evidence_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    prerequisite_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    operational_evidence_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: BoundarySupportEnvelope,
        /,
        *,
        claim: str,
        level: BoundaryEvidenceLevel,
        maturity: BoundaryMaturity,
        supported: bool,
        provenance_id: str,
        operational_evidence_id: str,
        evidence_artifact_ids: Sequence[str],
        prerequisite_evidence_ids: Sequence[str] = (),
        error_bound: float | None = None,
        error_metric: str | None = None,
        unsupported_reason: str | None = None,
    ):
        if not isinstance(support, BoundarySupportEnvelope):
            raise TypeError("support must be BoundarySupportEnvelope.")
        claim_ = _required_text(claim, "claim")
        if claim_ not in support.claims:
            raise ValueError(f"Unknown boundary qualification claim {claim_!r}.")
        if level not in _EVIDENCE_LEVELS:
            raise ValueError(f"Unknown boundary evidence level {level!r}.")
        if maturity not in _MATURITY_LEVELS:
            raise ValueError(f"Unknown boundary maturity {maturity!r}.")
        supported_ = _strict_bool(supported, "supported")
        if supported_ != support.supports(claim_):
            raise ValueError(
                "Evidence support status must match the support envelope declaration."
            )
        artifacts = _canonical_strings(
            evidence_artifact_ids, "evidence_artifact_ids", allow_empty=False
        )
        prerequisites = _canonical_strings(
            prerequisite_evidence_ids, "prerequisite_evidence_ids"
        )
        if level != "computed" and not prerequisites:
            raise ValueError(f"{level} evidence requires prerequisite evidence IDs.")
        provenance_id_ = _required_text(provenance_id, "provenance_id")
        operational_id_ = _required_text(
            operational_evidence_id, "operational_evidence_id"
        )
        metric = _optional_text(error_metric, "error_metric")
        reason = _optional_text(unsupported_reason, "unsupported_reason")
        if error_bound is None:
            bound = None
        else:
            if isinstance(error_bound, bool):
                raise TypeError("error_bound must be a real number or None.")
            bound = float(error_bound)
            if not math.isfinite(bound) or bound < 0.0:
                raise ValueError("error_bound must be finite and nonnegative.")
        if supported_:
            if reason is not None:
                raise ValueError("Supported evidence cannot have an unsupported reason.")
            if level != "computed" and bound is None:
                raise ValueError(f"{level} evidence requires an error bound.")
            if (bound is None) != (metric is None):
                raise ValueError(
                    "error_bound and error_metric must either both be present or both be absent."
                )
        else:
            if level != "computed":
                raise ValueError("Unsupported evidence must use the computed level.")
            if reason is None:
                raise ValueError("Unsupported evidence requires an explicit reason.")
            if bound is not None or metric is not None:
                raise ValueError(
                    "Unsupported is not a numerical zero: error fields must be absent."
                )
        self.support = support
        self.claim = claim_
        self.level = level
        self.maturity = maturity
        self.supported = supported_
        self.error_bound = bound
        self.error_metric = metric
        self.unsupported_reason = reason
        self.evidence_artifact_ids = artifacts
        self.prerequisite_evidence_ids = prerequisites
        self.provenance_id = provenance_id_
        self.operational_evidence_id = operational_id_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "boundary-qualification-evidence",
                "support": support.envelope_id,
                "claim": claim_,
                "level": level,
                "maturity": maturity,
                "supported": supported_,
                "error_bound": bound,
                "error_metric": metric,
                "unsupported_reason": reason,
                "evidence_artifact_ids": list(artifacts),
                "prerequisite_evidence_ids": list(prerequisites),
                "provenance": provenance_id_,
                "operational_evidence": operational_id_,
            }
        )


class BoundaryProductProvenance(StrictModule, NonTrainableState):
    """License-aware product lineage without implying legal approval."""

    product_id: str = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    source_kind: BoundaryProvenanceSource = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_content_id: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    clean_room_record_id: str = eqx.field(static=True)
    parent_product_ids: tuple[str, ...] = eqx.field(static=True)
    parent_plan_ids: tuple[str, ...] = eqx.field(static=True)
    parent_result_ids: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        product_id: str,
        producer_id: str,
        producer_version: str,
        provider_id: str,
        source_kind: BoundaryProvenanceSource,
        source_id: str,
        source_content_id: str,
        license_id: str,
        clean_room_record_id: str,
        parent_product_ids: Sequence[str] = (),
        parent_plan_ids: Sequence[str] = (),
        parent_result_ids: Sequence[str] = (),
    ):
        values = tuple(
            _required_text(value, name)
            for value, name in (
                (product_id, "product_id"),
                (producer_id, "producer_id"),
                (producer_version, "producer_version"),
                (provider_id, "provider_id"),
                (source_id, "source_id"),
                (source_content_id, "source_content_id"),
                (license_id, "license_id"),
                (clean_room_record_id, "clean_room_record_id"),
            )
        )
        if source_kind not in _PROVENANCE_SOURCES:
            raise ValueError(f"Unknown boundary provenance source {source_kind!r}.")
        product_parents = _parent_ids(parent_product_ids, "parent_product_ids")
        plan_parents = _parent_ids(parent_plan_ids, "parent_plan_ids")
        result_parents = _parent_ids(parent_result_ids, "parent_result_ids")
        if values[0] in product_parents:
            raise ValueError("product_id cannot be its own parent product ID.")
        (
            self.product_id,
            self.producer_id,
            self.producer_version,
            self.provider_id,
            self.source_id,
            self.source_content_id,
            self.license_id,
            self.clean_room_record_id,
        ) = values
        self.source_kind = source_kind
        self.parent_product_ids = product_parents
        self.parent_plan_ids = plan_parents
        self.parent_result_ids = result_parents
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "boundary-product-provenance",
                "product": values[0],
                "producer": values[1],
                "producer_version": values[2],
                "provider": values[3],
                "source_kind": source_kind,
                "source": values[4],
                "source_content": values[5],
                "license": values[6],
                "clean_room_record": values[7],
                "parent_product_ids": list(product_parents),
                "parent_plan_ids": list(plan_parents),
                "parent_result_ids": list(result_parents),
            }
        )


class BoundaryOperationalEvidence(StrictModule, NonTrainableState):
    """Fail-closed provider preflight, lineage, determinism, and byte evidence."""

    plan_id: str = eqx.field(static=True)
    result_id: str | None = eqx.field(static=True)
    parent_plan_ids: tuple[str, ...] = eqx.field(static=True)
    parent_result_ids: tuple[str, ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    provider_deterministic: bool = eqx.field(static=True)
    nondeterminism_reason: str | None = eqx.field(static=True)
    security_preflight_passed: bool = eqx.field(static=True)
    security_preflight_id: str = eqx.field(static=True)
    resource_preflight_passed: bool = eqx.field(static=True)
    resource_preflight_id: str = eqx.field(static=True)
    resource_limit_bytes: int = eqx.field(static=True)
    forecast_bytes: int = eqx.field(static=True)
    observed_bytes: int | None = eqx.field(static=True)
    stop_ship_reasons: tuple[str, ...] = eqx.field(static=True)
    operational_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan_id: str,
        result_id: str | None,
        parent_plan_ids: Sequence[str] = (),
        parent_result_ids: Sequence[str] = (),
        provider_id: str,
        provider_deterministic: bool,
        nondeterminism_reason: str | None = None,
        security_preflight_passed: bool,
        security_preflight_id: str,
        resource_preflight_passed: bool,
        resource_preflight_id: str,
        resource_limit_bytes: int,
        forecast_bytes: int,
        observed_bytes: int | None,
        stop_ship_reasons: Sequence[str] = (),
    ):
        plan = _required_text(plan_id, "plan_id")
        result = _optional_text(result_id, "result_id")
        plan_parents = _parent_ids(parent_plan_ids, "parent_plan_ids")
        result_parents = _parent_ids(parent_result_ids, "parent_result_ids")
        if plan in plan_parents:
            raise ValueError("plan_id cannot be its own parent plan ID.")
        if result is not None and result in result_parents:
            raise ValueError("result_id cannot be its own parent result ID.")
        provider = _required_text(provider_id, "provider_id")
        deterministic = _strict_bool(provider_deterministic, "provider_deterministic")
        nondeterminism = _optional_text(nondeterminism_reason, "nondeterminism_reason")
        if deterministic == (nondeterminism is not None):
            raise ValueError(
                "Deterministic providers omit nondeterminism_reason; other providers require it."
            )
        security_passed = _strict_bool(
            security_preflight_passed, "security_preflight_passed"
        )
        resource_passed = _strict_bool(
            resource_preflight_passed, "resource_preflight_passed"
        )
        security_id = _required_text(security_preflight_id, "security_preflight_id")
        resource_id = _required_text(resource_preflight_id, "resource_preflight_id")
        limit = _bounded_bytes(
            resource_limit_bytes, "resource_limit_bytes", positive=True
        )
        forecast = _bounded_bytes(forecast_bytes, "forecast_bytes")
        observed = (
            None
            if observed_bytes is None
            else _bounded_bytes(observed_bytes, "observed_bytes")
        )
        reasons = _canonical_strings(stop_ship_reasons, "stop_ship_reasons")
        if resource_passed and forecast > limit:
            raise ValueError(
                "resource_preflight_passed cannot be true when forecast_bytes exceeds the limit."
            )
        if (not security_passed or not resource_passed) and not reasons:
            raise ValueError("A failed preflight requires a stop-ship reason.")
        if result is not None and (not security_passed or not resource_passed):
            raise ValueError("A result cannot be recorded after a failed preflight.")
        if result is None and observed is not None:
            raise ValueError("observed_bytes requires a result_id.")
        if result is not None and observed is None:
            raise ValueError("A result_id requires observed_bytes.")
        if observed is not None and observed > limit and not reasons:
            raise ValueError(
                "An observed resource-limit violation requires a stop-ship reason."
            )
        self.plan_id = plan
        self.result_id = result
        self.parent_plan_ids = plan_parents
        self.parent_result_ids = result_parents
        self.provider_id = provider
        self.provider_deterministic = deterministic
        self.nondeterminism_reason = nondeterminism
        self.security_preflight_passed = security_passed
        self.security_preflight_id = security_id
        self.resource_preflight_passed = resource_passed
        self.resource_preflight_id = resource_id
        self.resource_limit_bytes = limit
        self.forecast_bytes = forecast
        self.observed_bytes = observed
        self.stop_ship_reasons = reasons
        self.operational_id = canonical_fingerprint(
            {
                "kind": "boundary-operational-evidence",
                "plan": plan,
                "result": result,
                "parent_plan_ids": list(plan_parents),
                "parent_result_ids": list(result_parents),
                "provider": provider,
                "provider_deterministic": deterministic,
                "nondeterminism_reason": nondeterminism,
                "security_preflight_passed": security_passed,
                "security_preflight": security_id,
                "resource_preflight_passed": resource_passed,
                "resource_preflight": resource_id,
                "resource_limit_bytes": limit,
                "forecast_bytes": forecast,
                "observed_bytes": observed,
                "stop_ship_reasons": list(reasons),
            }
        )


__all__ = [
    "BoundaryEvidenceLevel",
    "BoundaryMaturity",
    "BoundaryOperationalEvidence",
    "BoundaryProductProvenance",
    "BoundaryProvenanceSource",
    "BoundaryQualificationEvidence",
    "BoundarySupportEnvelope",
]
