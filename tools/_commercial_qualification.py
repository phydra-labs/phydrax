#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, fail-closed producers for commercial qualification candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import (
    ForecastResourceRecord,
    ObservedResourceRecord,
    QualificationEvidence,
    SupportDependency,
)
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.qualification._registry import CapabilityProfile, SupportTuple


GATE_CATEGORIES = ("scientific", "performance", "operational", "security")
_OUTCOMES = ("passed", "failed", "inconclusive")
_TIMING_TOKENS = (
    "elapsed",
    "latency",
    "runtime",
    "seconds",
    "throughput",
    "timing",
    "wall-clock",
    "wall_clock",
)


def _identifier(value: object, name: str, /) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _mapping(value: object, name: str, /) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, name: str, /) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _canonical_copy(value: object, /) -> object:
    return json.loads(canonical_json(value))


def _identified(
    kind: str, fields: Mapping[str, object], id_key: str, /
) -> dict[str, object]:
    core = {"kind": _identifier(kind, "record kind"), **dict(fields)}
    normalized = _canonical_copy(core)
    if not isinstance(normalized, dict):
        raise TypeError("Content-addressed record must serialize to an object.")
    return {**normalized, id_key: canonical_fingerprint(normalized)}


def _verify_address(record: Mapping[str, object], id_key: str, label: str, /) -> None:
    identifier = record.get(id_key)
    core = {name: value for name, value in record.items() if name != id_key}
    if type(identifier) is not str or canonical_fingerprint(core) != identifier:
        raise ValueError(f"{label} has an invalid content address.")


@dataclass(frozen=True, slots=True)
class GateDefinition:
    """One producer-owned gate identity and evidence category."""

    name: str
    category: str
    description: str

    def __post_init__(self) -> None:
        _identifier(self.name, "gate name")
        _identifier(self.description, "gate description")
        if self.category not in GATE_CATEGORIES:
            raise ValueError(
                "Gate category must be scientific, performance, operational, or security."
            )
        if self.category == "scientific" and _is_timing_name(
            f"{self.name} {self.description}"
        ):
            raise ValueError("Timing measurements cannot be scientific gates.")


@dataclass(frozen=True, slots=True)
class RouteDefinition:
    """Exact private API and evidence requirements for one qualification route."""

    route: str
    gates: tuple[GateDefinition, ...]
    private_api_symbols: tuple[str, ...]
    dependency_scope: str = "scientific"

    def __post_init__(self) -> None:
        _identifier(self.route, "route")
        if not self.gates or any(
            not isinstance(value, GateDefinition) for value in self.gates
        ):
            raise TypeError("Route gates must contain GateDefinition values.")
        names = tuple(value.name for value in self.gates)
        if len(set(names)) != len(names):
            raise ValueError("Route gate names must be unique.")
        required_categories = {"scientific", "performance", "operational"}
        if not required_categories.issubset(value.category for value in self.gates):
            raise ValueError(
                "Every route must separately declare scientific, performance, and operational gates."
            )
        if not self.private_api_symbols:
            raise ValueError(
                "A qualification route must bind at least one private API symbol."
            )
        for symbol in self.private_api_symbols:
            module, separator, name = symbol.partition(":")
            if not separator or not module or not name:
                raise ValueError("Private API symbols must use 'module:qualified-name'.")
        if self.dependency_scope not in ("scientific", "deployment"):
            raise ValueError("Dependency scope must be scientific or deployment.")


def _is_timing_name(value: str, /) -> bool:
    normalized = value.lower().replace("_", "-")
    return any(token.replace("_", "-") in normalized for token in _TIMING_TOKENS)


def private_api_contract(symbols: Sequence[str], /) -> tuple[dict[str, object], ...]:
    """Resolve exact private symbols and return their deterministic identities."""

    records: list[dict[str, object]] = []
    for symbol in sorted(_identifier(value, "private API symbol") for value in symbols):
        module_name, qualified_name = symbol.split(":", 1)
        value: object = importlib.import_module(module_name)
        for component in qualified_name.split("."):
            value = getattr(value, component)
        canonical_module = _identifier(value.__module__, "private API module")
        canonical_name = _identifier(value.__qualname__, "private API qualified name")
        records.append(
            _identified(
                "private-api-contract",
                {"symbol": f"{canonical_module}:{canonical_name}"},
                "api_id",
            )
        )
    return tuple(records)


def _coerce_support_tuple(value: object, /) -> SupportTuple:
    if isinstance(value, SupportTuple):
        return value
    return SupportTuple.from_record(_mapping(value, "support_tuple"))


def _coerce_dependency(value: object, /) -> SupportDependency:
    if isinstance(value, SupportDependency):
        return value
    return SupportDependency.from_record(_mapping(value, "support_dependency"))


def _coerce_run_spec(value: object, /) -> ResolvedRunSpec:
    if isinstance(value, ResolvedRunSpec):
        return value
    return ResolvedRunSpec.from_record(_mapping(value, "resolved_run_spec"))


def _normalize_resources(
    request: Mapping[str, object],
    evidence_context: Mapping[str, object],
    /,
) -> tuple[tuple[ObservedResourceRecord, ...], tuple[ForecastResourceRecord, ...]]:
    observed_values = _sequence(
        request.get("observed_resource_records", ()), "observed_resource_records"
    )
    forecast_values = _sequence(
        request.get("forecast_resource_records", ()), "forecast_resource_records"
    )
    observed = tuple(
        value
        if isinstance(value, ObservedResourceRecord)
        else ObservedResourceRecord.from_record(
            _mapping(value, "observed resource record")
        )
        for value in observed_values
    )
    forecasts = tuple(
        value
        if isinstance(value, ForecastResourceRecord)
        else ForecastResourceRecord.from_record(
            _mapping(value, "forecast resource record")
        )
        for value in forecast_values
    )
    observed_ids = tuple(value.record_id for value in observed)
    forecast_ids = tuple(value.record_id for value in forecasts)
    if len(set(observed_ids)) != len(observed_ids) or len(set(forecast_ids)) != len(
        forecast_ids
    ):
        raise ValueError("Resource records must have unique content addresses.")
    exact_fields = ("build_id", "environment_id", "backend", "topology")
    for record in (*observed, *forecasts):
        for field in exact_fields:
            if getattr(record, field) != evidence_context[field]:
                raise ValueError(
                    f"Resource record {record.record_id} does not match evidence {field}."
                )
    observed_id_set = set(observed_ids)
    for forecast in forecasts:
        if not set(forecast.source_record_ids).issubset(observed_id_set):
            raise ValueError(
                "Every forecast source record must be included as exact observed evidence."
            )
    return (
        tuple(sorted(observed, key=lambda value: value.record_id)),
        tuple(sorted(forecasts, key=lambda value: value.record_id)),
    )


def _reference_envelope(
    manifest_value: ReferenceArtifactManifest | Mapping[str, object] | None,
    payload: bytes | None,
    /,
) -> dict[str, object] | None:
    if manifest_value is None and payload is None:
        return None
    if manifest_value is None or payload is None:
        raise ValueError(
            "External reference manifest and checksum payload are an all-or-none hook."
        )
    manifest = (
        manifest_value
        if isinstance(manifest_value, ReferenceArtifactManifest)
        else ReferenceArtifactManifest.from_record(manifest_value)
    )
    manifest.require_rights(commercial_use=True)
    if not isinstance(payload, bytes):
        raise TypeError("External reference payload must be bytes.")
    if len(payload) != manifest.size_bytes:
        raise ValueError("External reference payload size does not match its manifest.")
    digest = hashlib.new(manifest.checksum_algorithm, payload).hexdigest()
    if digest != manifest.checksum:
        raise ValueError(
            "External reference payload checksum does not match its manifest."
        )
    return _identified(
        "governed-reference-input",
        {
            "manifest": manifest.to_record(),
            "checksum_verified": True,
            "data_embedded": False,
        },
        "reference_id",
    )


def _criterion_and_metric(
    definition: GateDefinition,
    observation: object,
    /,
    *,
    subject_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    unavailable_reason: str | None = None
    source_artifact_id: str | None = None
    quantity = definition.name
    comparison: str
    target: object
    value: object | None

    if observation is None:
        unavailable_reason = "required-observation-missing"
        comparison = "is-true"
        target = True
        value = None
    elif isinstance(observation, Mapping):
        allowed = {
            "value",
            "comparison",
            "target",
            "unavailable_reason",
            "source_artifact_id",
            "quantity",
        }
        extras = set(observation) - allowed
        if extras:
            raise ValueError(
                f"Observation {definition.name!r} has unsupported fields: {sorted(extras)}."
            )
        if "quantity" in observation:
            quantity = _identifier(observation["quantity"], "observation quantity")
        if "source_artifact_id" in observation:
            source_artifact_id = _identifier(
                observation["source_artifact_id"], "source artifact ID"
            )
        if "unavailable_reason" in observation:
            unavailable_reason = _identifier(
                observation["unavailable_reason"], "unavailable reason"
            )
            if "value" in observation:
                raise ValueError(
                    "An unavailable observation cannot also contain a value."
                )
            value = None
            comparison = str(observation.get("comparison", "is-true"))
            target = observation.get("target", True)
        else:
            if "value" not in observation:
                raise ValueError(
                    "An observation mapping needs value or unavailable_reason."
                )
            value = observation["value"]
            if isinstance(value, bool):
                comparison = str(observation.get("comparison", "is-true"))
                target = observation.get("target", True)
            else:
                if "comparison" not in observation or "target" not in observation:
                    raise ValueError(
                        "A non-boolean observation needs an exact comparison and target."
                    )
                comparison = str(observation["comparison"])
                target = observation["target"]
    elif isinstance(observation, bool):
        value = observation
        comparison = "is-true"
        target = True
    else:
        raise TypeError(
            "Gate observations must be booleans, mappings, or None for unavailable evidence."
        )

    if definition.category == "scientific" and _is_timing_name(quantity):
        raise ValueError("Timing measurements cannot be used as scientific evidence.")
    comparison = _identifier(comparison, "criterion comparison")
    if comparison not in (
        "is-true",
        "equal",
        "less-than-or-equal",
        "greater-than-or-equal",
    ):
        raise ValueError("Unsupported qualification comparison.")
    if comparison == "is-true" and target is not True:
        raise ValueError("is-true criteria must have the exact target true.")
    if comparison in ("less-than-or-equal", "greater-than-or-equal"):
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("Ordered criterion targets must be real numbers.")
        if not math.isfinite(float(target)):
            raise ValueError("Ordered criterion targets must be finite.")
        target = float(target)

    metric_fields: dict[str, object] = {
        "name": definition.name,
        "category": definition.category,
        "quantity": quantity,
        "subject_id": _identifier(subject_id, "observation subject ID"),
    }
    if unavailable_reason is None:
        if isinstance(value, float) and not math.isfinite(value):
            outcome = "failed"
            reason = f"{definition.name}: observed metric is non-finite."
        else:
            if isinstance(value, (dict, list, tuple)):
                raise TypeError("Qualification gate values must be scalar.")
            metric_fields["value"] = value
            if comparison == "is-true":
                passed = value is True
            elif comparison == "equal":
                passed = value == target
            elif comparison == "less-than-or-equal":
                passed = bool(value <= target)
            else:
                passed = bool(value >= target)
            outcome = "passed" if passed else "failed"
            reason = (
                f"{definition.name}: observed metric satisfied the exact criterion."
                if passed
                else f"{definition.name}: observed metric did not satisfy the exact criterion."
            )
    else:
        metric_fields["unavailable_reason"] = unavailable_reason
        outcome = "inconclusive"
        reason = f"{definition.name}: {unavailable_reason}."
    if source_artifact_id is not None:
        metric_fields["source_artifact_id"] = source_artifact_id

    metric = _identified("qualification-observation", metric_fields, "metric_id")
    criterion = _identified(
        "qualification-criterion",
        {
            "name": definition.name,
            "category": definition.category,
            "description": definition.description,
            "comparison": comparison,
            "target": target,
            "subject_id": subject_id,
        },
        "criterion_id",
    )
    gate = _identified(
        "qualification-gate-outcome",
        {
            "name": definition.name,
            "category": definition.category,
            "metric_id": metric["metric_id"],
            "criterion_id": criterion["criterion_id"],
            "outcome": outcome,
            "reason": reason,
        },
        "gate_id",
    )
    return metric, criterion, gate


def _evidence_context(request: Mapping[str, object], /) -> Mapping[str, object]:
    context = _mapping(request.get("evidence_context"), "evidence_context")
    identifiers = (
        "build_id",
        "environment_id",
        "backend",
        "topology",
        "precision",
        "reduction",
        "replay_id",
        "reviewer_id",
    )
    for name in identifiers:
        _identifier(context.get(name), f"evidence {name}")
    issued = context.get("issued_at")
    expires = context.get("expires_at")
    if type(issued) is not int or type(expires) is not int:
        raise TypeError("Evidence issued_at and expires_at must be integers.")
    if issued < 0 or expires <= issued:
        raise ValueError("Evidence validity must have positive non-negative duration.")
    triggers = context.get(
        "requalification_triggers",
        ("build-changed", "private-api-changed", "support-tuple-changed"),
    )
    for value in _sequence(triggers, "requalification_triggers"):
        _identifier(value, "requalification trigger")
    return context


def make_candidate_artifact(
    capability: str,
    definition: RouteDefinition,
    request: Mapping[str, object],
    /,
    *,
    reference_manifest: ReferenceArtifactManifest | Mapping[str, object] | None = None,
    reference_payload: bytes | None = None,
    extra_record: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create one exact, unsigned candidate from governed typed admissions."""

    capability_ = _identifier(capability, "capability")
    if not isinstance(definition, RouteDefinition):
        raise TypeError("definition must be a RouteDefinition.")
    request_ = _mapping(request, "qualification request")
    if (
        request_.get("dns_claimed", False) is not False
        or request_.get("inherits_dns", False) is not False
    ):
        raise ValueError("Candidate qualification cannot claim or inherit DNS support.")

    support = _coerce_support_tuple(request_.get("support_tuple"))
    dependency = _coerce_dependency(request_.get("support_dependency"))
    run_spec = _coerce_run_spec(request_.get("resolved_run_spec"))
    attributes = dict(support.attributes)
    if support.capability != capability_ or attributes.get("route") != definition.route:
        raise ValueError("SupportTuple must exactly identify the capability and route.")
    if dependency.support_tuple_id != support.support_tuple_id:
        raise ValueError("SupportDependency does not bind the exact SupportTuple.")
    dependencies = (
        run_spec.scientific_dependencies
        if definition.dependency_scope == "scientific"
        else run_spec.deployment_dependencies
    )
    if (
        tuple(value.dependency_id for value in dependencies).count(
            dependency.dependency_id
        )
        != 1
    ):
        raise ValueError(
            "ResolvedRunSpec does not admit the exact SupportDependency in the required scope."
        )
    if dependency.profile_id not in run_spec.profile_ids:
        raise ValueError("ResolvedRunSpec does not bind the dependency profile.")

    context = _evidence_context(request_)
    if not int(context["issued_at"]) <= run_spec.valid_at <= int(context["expires_at"]):
        raise ValueError(
            "ResolvedRunSpec valid_at must lie within the qualification evidence window."
        )
    observed_resources, forecast_resources = _normalize_resources(request_, context)
    external_reference = _reference_envelope(reference_manifest, reference_payload)
    observations = _mapping(request_.get("observations", {}), "observations")
    api_records = private_api_contract(definition.private_api_symbols)

    definitions = list(definition.gates)
    definitions.append(
        GateDefinition(
            "typed-admission-integrity",
            "security",
            "SupportTuple, SupportDependency, ResolvedRunSpec, and private API identities verify exactly.",
        )
    )
    automatic: dict[str, object] = {"typed-admission-integrity": True}
    if external_reference is not None:
        definitions.append(
            GateDefinition(
                "external-reference-governance",
                "security",
                "Commercial rights, checksum, size, and uncertainty manifest are verified.",
            )
        )
        automatic["external-reference-governance"] = {
            "value": True,
            "source_artifact_id": external_reference["reference_id"],
        }

    metrics: dict[str, dict[str, object]] = {}
    criteria: dict[str, dict[str, object]] = {}
    gates_by_category: dict[str, list[dict[str, object]]] = {
        category: [] for category in GATE_CATEGORIES
    }
    for gate_definition in definitions:
        observation = automatic.get(
            gate_definition.name, observations.get(gate_definition.name)
        )
        metric, criterion, gate = _criterion_and_metric(
            gate_definition,
            observation,
            subject_id=support.support_tuple_id,
        )
        metrics[gate_definition.name] = metric
        criteria[gate_definition.name] = criterion
        gates_by_category[gate_definition.category].append(gate)
    for values in gates_by_category.values():
        values.sort(key=lambda value: str(value["gate_id"]))

    evidence_records: dict[str, dict[str, object]] = {}
    subject_ids = (
        support.support_tuple_id,
        dependency.dependency_id,
        run_spec.spec_id,
    )
    triggers = tuple(
        _identifier(value, "requalification trigger")
        for value in _sequence(
            context.get(
                "requalification_triggers",
                ("build-changed", "private-api-changed", "support-tuple-changed"),
            ),
            "requalification_triggers",
        )
    )
    for category in GATE_CATEGORIES:
        category_gates = gates_by_category[category]
        outcomes = tuple(str(value["outcome"]) for value in category_gates)
        outcome = (
            "failed"
            if "failed" in outcomes
            else "inconclusive"
            if "inconclusive" in outcomes
            else "passed"
        )
        reason = {
            "passed": f"All {category} route criteria passed.",
            "failed": f"At least one observed {category} route criterion failed.",
            "inconclusive": f"Required {category} route observations were unavailable.",
        }[outcome]
        category_names = tuple(str(value["name"]) for value in category_gates)
        evidence = QualificationEvidence(
            category,
            outcome,
            subject_ids,
            build_id=str(context["build_id"]),
            environment_id=str(context["environment_id"]),
            backend=str(context["backend"]),
            topology=str(context["topology"]),
            precision=str(context["precision"]),
            reduction=str(context["reduction"]),
            replay_id=str(context["replay_id"]),
            criteria_ids=tuple(
                str(criteria[name]["criterion_id"]) for name in category_names
            ),
            raw_artifact_ids=tuple(
                str(metrics[name]["metric_id"]) for name in category_names
            ),
            reviewer_id=str(context["reviewer_id"]),
            issued_at=int(context["issued_at"]),
            expires_at=int(context["expires_at"]),
            reason=reason,
            requalification_triggers=triggers,
            observed_resource_record_ids=(
                tuple(value.record_id for value in observed_resources)
                if category == "performance"
                else ()
            ),
            forecast_resource_record_ids=(
                tuple(value.record_id for value in forecast_resources)
                if category == "performance"
                else ()
            ),
        )
        evidence_records[category] = evidence.to_record()

    all_gates = tuple(
        gate for category in GATE_CATEGORIES for gate in gates_by_category[category]
    )
    failed = sorted(
        str(value["reason"]) for value in all_gates if value["outcome"] == "failed"
    )
    inconclusive = sorted(
        str(value["reason"]) for value in all_gates if value["outcome"] == "inconclusive"
    )
    status = "failed" if failed else "inconclusive" if inconclusive else "passed"
    normalized_extra = _canonical_copy({} if extra_record is None else dict(extra_record))
    if not isinstance(normalized_extra, dict):
        raise TypeError("extra_record must serialize to an object.")
    core = {
        "kind": "commercial-qualification-candidate-artifact",
        "capability": capability_,
        "route": definition.route,
        "dependency_scope": definition.dependency_scope,
        "private_api_contracts": list(api_records),
        "support_tuple": support.to_record(),
        "support_dependency": dependency.to_record(),
        "resolved_run_spec": run_spec.to_record(),
        "qualification_evidence": evidence_records,
        "metrics": dict(sorted(metrics.items())),
        "criteria": dict(sorted(criteria.items())),
        "gates": {category: gates_by_category[category] for category in GATE_CATEGORIES},
        "resource_records": {
            "observed": [value.to_record() for value in observed_resources],
            "forecast": [value.to_record() for value in forecast_resources],
        },
        "external_reference": external_reference,
        "extra": normalized_extra,
        "status": status,
        "failed_reasons": failed,
        "inconclusive_reasons": inconclusive,
        "dns_claimed": False,
        "inherits_dns": False,
        "signed": False,
        "release_ready": False,
    }
    return {**core, "artifact_id": canonical_fingerprint(core)}


def verify_candidate_artifact(record: Mapping[str, object], /) -> None:
    """Content-verify a candidate and every exact typed record it binds."""

    value = _mapping(record, "qualification artifact")
    if value.get("kind") != "commercial-qualification-candidate-artifact":
        raise ValueError("Unsupported qualification candidate kind.")
    if (
        value.get("release_ready") is not False
        or value.get("signed") is not False
        or value.get("dns_claimed") is not False
        or value.get("inherits_dns") is not False
    ):
        raise ValueError("Qualification candidate must remain unsigned and unreleased.")
    if any(name in value for name in ("schema", "schema_version", "signature")):
        raise ValueError("Qualification candidates cannot carry schemas or signatures.")
    _verify_address(value, "artifact_id", "Qualification artifact")

    support = SupportTuple.from_record(
        _mapping(value.get("support_tuple"), "support_tuple")
    )
    dependency = SupportDependency.from_record(
        _mapping(value.get("support_dependency"), "support_dependency")
    )
    run_spec = ResolvedRunSpec.from_record(
        _mapping(value.get("resolved_run_spec"), "resolved_run_spec")
    )
    if support.capability != value.get("capability") or dict(support.attributes).get(
        "route"
    ) != value.get("route"):
        raise ValueError("Candidate SupportTuple is not route exact.")
    if dependency.support_tuple_id != support.support_tuple_id:
        raise ValueError("Candidate SupportDependency is not tuple exact.")
    scope = value.get("dependency_scope")
    dependencies = (
        run_spec.scientific_dependencies
        if scope == "scientific"
        else run_spec.deployment_dependencies
        if scope == "deployment"
        else ()
    )
    if (
        tuple(item.dependency_id for item in dependencies).count(dependency.dependency_id)
        != 1
    ):
        raise ValueError(
            "Candidate ResolvedRunSpec does not exactly admit its dependency."
        )

    api_records = _sequence(value.get("private_api_contracts"), "private_api_contracts")
    if not api_records:
        raise ValueError("Candidate must bind private API contracts.")
    for item in api_records:
        record_ = _mapping(item, "private API contract")
        _verify_address(record_, "api_id", "Private API contract")
        resolved = private_api_contract((str(record_.get("symbol")),))
        if len(resolved) != 1 or resolved[0] != record_:
            raise ValueError(
                "Private API contract does not match the available private symbol."
            )

    metric_values = _mapping(value.get("metrics"), "metrics")
    criterion_values = _mapping(value.get("criteria"), "criteria")
    gates = _mapping(value.get("gates"), "gates")
    evidence_values = _mapping(
        value.get("qualification_evidence"), "qualification_evidence"
    )
    if tuple(gates) != GATE_CATEGORIES or tuple(evidence_values) != GATE_CATEGORIES:
        raise ValueError(
            "Candidate must separate all four qualification gate categories."
        )
    if set(metric_values) != set(criterion_values):
        raise ValueError("Qualification metrics and criteria must cover the same gates.")
    metric_ids: dict[str, str] = {}
    criterion_ids: dict[str, str] = {}
    for name, item in metric_values.items():
        metric = _mapping(item, "metric")
        _verify_address(metric, "metric_id", "Qualification metric")
        metric_ids[str(name)] = str(metric["metric_id"])
        if metric.get("name") != name or metric.get("category") not in GATE_CATEGORIES:
            raise ValueError(
                "Qualification metric identity does not match its record key."
            )
        if metric.get("subject_id") != support.support_tuple_id:
            raise ValueError("Qualification metric does not bind the exact SupportTuple.")
        if metric["category"] == "scientific" and _is_timing_name(
            str(metric.get("quantity"))
        ):
            raise ValueError("Timing measurements cannot be scientific evidence.")
    for name, item in criterion_values.items():
        criterion = _mapping(item, "criterion")
        _verify_address(criterion, "criterion_id", "Qualification criterion")
        criterion_ids[str(name)] = str(criterion["criterion_id"])
        if (
            criterion.get("name") != name
            or criterion.get("category") not in GATE_CATEGORIES
        ):
            raise ValueError(
                "Qualification criterion identity does not match its record key."
            )
        if criterion.get("subject_id") != support.support_tuple_id:
            raise ValueError(
                "Qualification criterion does not bind the exact SupportTuple."
            )

    all_outcomes: list[str] = []
    failed_reasons: list[str] = []
    inconclusive_reasons: list[str] = []
    evidence_by_category: dict[str, QualificationEvidence] = {}
    for category in GATE_CATEGORIES:
        category_gates = _sequence(gates[category], f"{category} gates")
        names: list[str] = []
        category_outcomes: list[str] = []
        for item in category_gates:
            gate = _mapping(item, "gate")
            _verify_address(gate, "gate_id", "Qualification gate")
            name = str(gate["name"])
            if gate.get("category") != category:
                raise ValueError("Qualification gate appears in the wrong category.")
            if gate.get("metric_id") != metric_ids.get(name) or gate.get(
                "criterion_id"
            ) != criterion_ids.get(name):
                raise ValueError(
                    "Qualification gate does not bind its exact metric and criterion."
                )
            outcome = str(gate.get("outcome"))
            if outcome not in _OUTCOMES:
                raise ValueError("Qualification gate has an invalid outcome.")
            reason = _identifier(gate.get("reason"), "qualification gate reason")
            if outcome == "failed":
                failed_reasons.append(reason)
            elif outcome == "inconclusive":
                inconclusive_reasons.append(reason)
            names.append(name)
            category_outcomes.append(outcome)
            all_outcomes.append(outcome)
        evidence = QualificationEvidence.from_record(
            _mapping(evidence_values[category], f"{category} evidence")
        )
        evidence_by_category[category] = evidence
        expected_outcome = (
            "failed"
            if "failed" in category_outcomes
            else "inconclusive"
            if "inconclusive" in category_outcomes
            else "passed"
        )
        if evidence.evidence_kind != category or evidence.outcome != expected_outcome:
            raise ValueError("QualificationEvidence does not match its gate category.")
        if set(evidence.criteria_ids) != {criterion_ids[name] for name in names}:
            raise ValueError("QualificationEvidence does not bind the exact criteria.")
        if set(evidence.raw_artifact_ids) != {metric_ids[name] for name in names}:
            raise ValueError(
                "QualificationEvidence does not bind the exact observations."
            )
        if not {
            support.support_tuple_id,
            dependency.dependency_id,
            run_spec.spec_id,
        }.issubset(evidence.subject_ids):
            raise ValueError(
                "QualificationEvidence does not bind the exact admission subjects."
            )

    expected_status = (
        "failed"
        if "failed" in all_outcomes
        else "inconclusive"
        if "inconclusive" in all_outcomes
        else "passed"
    )
    if value.get("status") != expected_status:
        raise ValueError("Qualification candidate has an invalid aggregate status.")
    if value.get("failed_reasons") != sorted(failed_reasons) or value.get(
        "inconclusive_reasons"
    ) != sorted(inconclusive_reasons):
        raise ValueError(
            "Qualification candidate reasons do not match its gate outcomes."
        )

    resources = _mapping(value.get("resource_records"), "resource_records")
    observed_records = tuple(
        ObservedResourceRecord.from_record(_mapping(item, "observed resource record"))
        for item in _sequence(resources.get("observed"), "observed resource records")
    )
    forecast_records = tuple(
        ForecastResourceRecord.from_record(_mapping(item, "forecast resource record"))
        for item in _sequence(resources.get("forecast"), "forecast resource records")
    )
    performance_evidence = evidence_by_category["performance"]
    if set(performance_evidence.observed_resource_record_ids) != {
        item.record_id for item in observed_records
    } or set(performance_evidence.forecast_resource_record_ids) != {
        item.record_id for item in forecast_records
    }:
        raise ValueError("Performance evidence does not bind the exact resource records.")

    reference = value.get("external_reference")
    if reference is not None:
        envelope = _mapping(reference, "external_reference")
        _verify_address(envelope, "reference_id", "External reference")
        if (
            envelope.get("checksum_verified") is not True
            or envelope.get("data_embedded") is not False
        ):
            raise ValueError("External reference envelope is not checksum governed.")
        manifest = ReferenceArtifactManifest.from_record(
            _mapping(envelope.get("manifest"), "reference manifest")
        )
        manifest.require_rights(commercial_use=True)


def assemble_candidate_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str,
    provider: str = "phydrax",
) -> dict[str, object]:
    """Assemble passed candidates without signing or releasing their profile."""

    records = tuple(_mapping(value, "qualification artifact") for value in artifacts)
    if not records:
        raise ValueError("At least one qualification artifact is required.")
    supports: list[SupportTuple] = []
    dependencies: list[SupportDependency] = []
    artifact_ids: list[str] = []
    routes: list[str] = []
    criterion_ids: set[str] = set()
    capabilities: set[str] = set()
    for record in records:
        verify_candidate_artifact(record)
        if record.get("status") != "passed":
            raise ValueError(
                f"Qualification artifact {record['artifact_id']} is not passed."
            )
        supports.append(
            SupportTuple.from_record(_mapping(record["support_tuple"], "support_tuple"))
        )
        dependencies.append(
            SupportDependency.from_record(
                _mapping(record["support_dependency"], "support_dependency")
            )
        )
        artifact_ids.append(str(record["artifact_id"]))
        routes.append(str(record["route"]))
        capabilities.add(str(record["capability"]))
        for criterion in _mapping(record["criteria"], "criteria").values():
            criterion_ids.add(str(_mapping(criterion, "criterion")["criterion_id"]))
    if len(capabilities) != 1:
        raise ValueError("A candidate profile can contain only one capability family.")
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("Candidate profile received duplicate artifacts.")
    dependency_by_id = {value.dependency_id: value for value in dependencies}
    profile = CapabilityProfile(
        _identifier(name, "profile name"),
        _identifier(provider, "profile provider"),
        "candidate",
        tuple(supports),
        dependencies=tuple(dependency_by_id.values()),
        required_gates=tuple(sorted(criterion_ids)),
        release_evidence=(),
        released=False,
    )
    core = {
        "kind": "commercial-capability-profile-candidate",
        "capability": next(iter(capabilities)),
        "qualification_artifact_ids": sorted(artifact_ids),
        "qualified_routes": sorted(routes),
        "profile": profile.to_record(),
        "signed": False,
        "release_ready": False,
    }
    return {**core, "candidate_id": canonical_fingerprint(core)}


def availability_observation(
    availability: Mapping[str, object] | None,
    /,
    *,
    provider: str,
    minimum_devices: int = 1,
    require_hardware: bool = True,
) -> bool | dict[str, object]:
    """Convert observed provider availability into a fail-closed gate value."""

    provider_ = _identifier(provider, "provider")
    if minimum_devices < 1:
        raise ValueError("minimum_devices must be positive.")
    if availability is None:
        return {"unavailable_reason": f"{provider_}-availability-not-recorded"}
    value = _mapping(availability, "availability")
    if value.get("simulated", False) is True:
        return {"unavailable_reason": f"{provider_}-simulation-is-not-qualification"}
    if value.get("observed") is not True:
        return {"unavailable_reason": f"{provider_}-availability-not-observed"}
    if value.get("provider_available") is not True:
        reason = value.get("reason", f"{provider_}-provider-absent")
        return {"unavailable_reason": _identifier(reason, "availability reason")}
    if require_hardware:
        device_count = value.get("device_count")
        if type(device_count) is not int or device_count < minimum_devices:
            return {
                "unavailable_reason": (
                    f"{provider_}-requires-{minimum_devices}-physical-devices"
                )
            }
    return True


def with_observation(
    request: Mapping[str, object], name: str, observation: object, /
) -> dict[str, object]:
    """Return a request with one producer-owned observation overwritten."""

    result = dict(_mapping(request, "qualification request"))
    values = dict(_mapping(result.get("observations", {}), "observations"))
    values[_identifier(name, "observation name")] = observation
    result["observations"] = values
    return result


def _reject_nonfinite(value: str, /) -> None:
    raise ValueError(f"Non-finite JSON value {value!r} is not permitted.")


def read_json_object(path: Path, /) -> Mapping[str, object]:
    value = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
    return _mapping(value, f"JSON object {path}")


def write_json(payload: Mapping[str, object], output: Path, /) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def build_cli_parser(
    description: str,
    routes: Mapping[str, RouteDefinition],
    /,
    *,
    profile_name: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    commands = parser.add_subparsers(dest="command", required=True)
    for route in sorted(routes):
        command = commands.add_parser(route)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--reference-manifest", type=Path)
        command.add_argument("--reference-data", type=Path)
        command.add_argument("--output", type=Path, required=True)
    assemble = commands.add_parser("assemble-profile")
    assemble.add_argument("artifacts", nargs="+", type=Path)
    assemble.add_argument("--name", default=profile_name)
    assemble.add_argument("--provider", default="phydrax")
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def run_cli(
    parser: argparse.ArgumentParser,
    routes: Mapping[str, RouteDefinition],
    capability: str,
    argv: Sequence[str] | None,
    /,
    *,
    producer,
) -> None:
    arguments = parser.parse_args(argv)
    if arguments.command == "assemble-profile":
        payload = assemble_candidate_profile(
            tuple(read_json_object(path) for path in arguments.artifacts),
            name=arguments.name,
            provider=arguments.provider,
        )
    else:
        manifest_path = arguments.reference_manifest
        data_path = arguments.reference_data
        if (manifest_path is None) != (data_path is None):
            parser.error("--reference-manifest and --reference-data are all-or-none")
        manifest = (
            None
            if manifest_path is None
            else ReferenceArtifactManifest.from_record(read_json_object(manifest_path))
        )
        reference_payload = None if data_path is None else data_path.read_bytes()
        payload = producer(
            arguments.command,
            read_json_object(arguments.input),
            reference_manifest=manifest,
            reference_payload=reference_payload,
        )
    write_json(payload, arguments.output)


__all__ = [
    "GATE_CATEGORIES",
    "GateDefinition",
    "RouteDefinition",
    "assemble_candidate_profile",
    "availability_observation",
    "build_cli_parser",
    "canonical_json",
    "make_candidate_artifact",
    "private_api_contract",
    "read_json_object",
    "run_cli",
    "verify_candidate_artifact",
    "with_observation",
    "write_json",
]
