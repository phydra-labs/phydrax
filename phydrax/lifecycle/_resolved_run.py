#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification._evidence import SupportDependency


_MAX_TIMESTAMP = 2**63 - 1
_DEPENDENCY_FIELDS = frozenset(
    {"kind", "profile_id", "support_tuple_id", "dependency_id"}
)
_RUN_SPEC_FIELDS = frozenset(
    {
        "kind",
        "scientific_dependencies",
        "deployment_dependencies",
        "release_index_id",
        "profile_ids",
        "trust_policy_id",
        "valid_at",
        "valid_from",
        "valid_until",
        "prepared_configuration_id",
        "precision_policy_id",
        "resource_policy_id",
        "checkpoint_policy_id",
        "output_policy_id",
        "repository_id",
        "scheduler_id",
        "auth_policy_id",
        "spec_id",
    }
)


def _identifier(value: object, name: str, /) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _timestamp(value: object, name: str, /) -> int:
    if type(value) is not int or value < 0 or value > _MAX_TIMESTAMP:
        raise ValueError(f"{name} must be a non-negative signed 64-bit timestamp.")
    return value


def _exact_fields(
    record: Mapping[str, object], expected: frozenset[str], label: str, /
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    keys = set(record)
    if any(type(key) is not str for key in keys):
        raise TypeError(f"{label} field names must be strings.")
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{label} has {'; '.join(details)}.")


def _dependency_record(value: SupportDependency, /) -> dict[str, object]:
    if not isinstance(value, SupportDependency):
        raise TypeError("Run dependencies must contain SupportDependency values.")
    return value.to_record()


def _dependency_from_record(record: object, /) -> SupportDependency:
    if not isinstance(record, Mapping):
        raise TypeError("Serialized support dependencies must be mappings.")
    _exact_fields(record, _DEPENDENCY_FIELDS, "Support-dependency record")
    if record["kind"] != "exact-support-dependency":
        raise ValueError("Support-dependency record has an unsupported kind.")
    dependency = SupportDependency(
        _identifier(record["profile_id"], "profile ID"),
        _identifier(record["support_tuple_id"], "support-tuple ID"),
    )
    recorded_id = _identifier(record["dependency_id"], "dependency ID")
    if recorded_id != dependency.dependency_id:
        raise ValueError("Serialized support dependency has an invalid content address.")
    return dependency


def _dependencies(
    values: Sequence[SupportDependency], name: str, /
) -> tuple[SupportDependency, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of SupportDependency values.")
    dependencies = tuple(values)
    if any(not isinstance(value, SupportDependency) for value in dependencies):
        raise TypeError(f"{name} must contain only SupportDependency values.")
    ids = tuple(value.dependency_id for value in dependencies)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate support dependencies.")
    return tuple(sorted(dependencies, key=lambda value: value.dependency_id))


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    identifiers = tuple(_identifier(value, name) for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique.")
    return tuple(sorted(identifiers))


def _reject_nonfinite(value: str, /) -> None:
    raise ValueError(f"Non-finite JSON value {value!r} is not permitted.")


def _load_json_object(payload: str, label: str, /) -> Mapping[str, object]:
    if type(payload) is not str:
        raise TypeError(f"{label} JSON payload must be a string.")
    value = json.loads(payload, parse_constant=_reject_nonfinite)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} JSON root must be an object.")
    return value


class ResolvedRunSpec(StrictModule, NonTrainableState):
    """Fully bound, time-scoped identity for one prepared execution."""

    scientific_dependencies: tuple[SupportDependency, ...]
    deployment_dependencies: tuple[SupportDependency, ...]
    release_index_id: str = eqx.field(static=True)
    profile_ids: tuple[str, ...] = eqx.field(static=True)
    trust_policy_id: str = eqx.field(static=True)
    valid_at: int = eqx.field(static=True)
    valid_from: int = eqx.field(static=True)
    valid_until: int = eqx.field(static=True)
    prepared_configuration_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    resource_policy_id: str = eqx.field(static=True)
    checkpoint_policy_id: str = eqx.field(static=True)
    output_policy_id: str = eqx.field(static=True)
    repository_id: str = eqx.field(static=True)
    scheduler_id: str = eqx.field(static=True)
    auth_policy_id: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        scientific_dependencies: Sequence[SupportDependency],
        deployment_dependencies: Sequence[SupportDependency],
        /,
        *,
        release_index_id: str,
        profile_ids: Sequence[str],
        trust_policy_id: str,
        valid_at: int,
        valid_from: int,
        valid_until: int,
        prepared_configuration_id: str,
        precision_policy_id: str,
        resource_policy_id: str,
        checkpoint_policy_id: str,
        output_policy_id: str,
        repository_id: str,
        scheduler_id: str,
        auth_policy_id: str,
    ):
        scientific = _dependencies(scientific_dependencies, "scientific_dependencies")
        deployment = _dependencies(deployment_dependencies, "deployment_dependencies")
        all_dependencies = scientific + deployment
        dependency_ids = tuple(value.dependency_id for value in all_dependencies)
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError(
                "A support dependency cannot be both scientific and deployment scope."
            )
        tuple_by_profile: dict[str, str] = {}
        for dependency in all_dependencies:
            previous = tuple_by_profile.setdefault(
                dependency.profile_id, dependency.support_tuple_id
            )
            if previous != dependency.support_tuple_id:
                raise ValueError(
                    "A profile cannot resolve to conflicting support-tuple dependencies."
                )
        profiles = _identifiers(profile_ids, "profile IDs")
        resolved_profiles = tuple(sorted(tuple_by_profile))
        if profiles != resolved_profiles:
            raise ValueError(
                "profile_ids must exactly match the profiles bound by run dependencies."
            )
        at = _timestamp(valid_at, "valid_at")
        start = _timestamp(valid_from, "valid_from")
        end = _timestamp(valid_until, "valid_until")
        if end <= start:
            raise ValueError("The validity window must have positive duration.")
        if not start <= at <= end:
            raise ValueError("valid_at must lie within the inclusive validity window.")
        self.scientific_dependencies = scientific
        self.deployment_dependencies = deployment
        self.release_index_id = _identifier(release_index_id, "release-index ID")
        self.profile_ids = profiles
        self.trust_policy_id = _identifier(trust_policy_id, "trust-policy ID")
        self.valid_at = at
        self.valid_from = start
        self.valid_until = end
        self.prepared_configuration_id = _identifier(
            prepared_configuration_id, "prepared-configuration ID"
        )
        self.precision_policy_id = _identifier(precision_policy_id, "precision-policy ID")
        self.resource_policy_id = _identifier(resource_policy_id, "resource-policy ID")
        self.checkpoint_policy_id = _identifier(
            checkpoint_policy_id, "checkpoint-policy ID"
        )
        self.output_policy_id = _identifier(output_policy_id, "output-policy ID")
        self.repository_id = _identifier(repository_id, "repository ID")
        self.scheduler_id = _identifier(scheduler_id, "scheduler ID")
        self.auth_policy_id = _identifier(auth_policy_id, "auth-policy ID")
        self.spec_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "resolved-run-spec",
            "scientific_dependencies": [
                _dependency_record(value) for value in self.scientific_dependencies
            ],
            "deployment_dependencies": [
                _dependency_record(value) for value in self.deployment_dependencies
            ],
            "release_index_id": self.release_index_id,
            "profile_ids": list(self.profile_ids),
            "trust_policy_id": self.trust_policy_id,
            "valid_at": self.valid_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "prepared_configuration_id": self.prepared_configuration_id,
            "precision_policy_id": self.precision_policy_id,
            "resource_policy_id": self.resource_policy_id,
            "checkpoint_policy_id": self.checkpoint_policy_id,
            "output_policy_id": self.output_policy_id,
            "repository_id": self.repository_id,
            "scheduler_id": self.scheduler_id,
            "auth_policy_id": self.auth_policy_id,
        }

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic JSON-ready run specification."""
        return {**self._content_record(), "spec_id": self.spec_id}

    def to_json(self) -> str:
        """Serialize this specification to canonical JSON."""
        return canonical_json(self.to_record())

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ResolvedRunSpec:
        """Strictly reconstruct and content-verify a run specification."""
        _exact_fields(record, _RUN_SPEC_FIELDS, "Resolved-run record")
        if record["kind"] != "resolved-run-spec":
            raise ValueError("Resolved-run record has an unsupported kind.")
        scientific_records = record["scientific_dependencies"]
        deployment_records = record["deployment_dependencies"]
        profile_ids = record["profile_ids"]
        for name, values in (
            ("scientific_dependencies", scientific_records),
            ("deployment_dependencies", deployment_records),
            ("profile_ids", profile_ids),
        ):
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError(f"Serialized {name} must be a sequence.")
        value = cls(
            tuple(_dependency_from_record(item) for item in scientific_records),
            tuple(_dependency_from_record(item) for item in deployment_records),
            release_index_id=_identifier(record["release_index_id"], "release-index ID"),
            profile_ids=tuple(_identifier(item, "profile ID") for item in profile_ids),
            trust_policy_id=_identifier(record["trust_policy_id"], "trust-policy ID"),
            valid_at=_timestamp(record["valid_at"], "valid_at"),
            valid_from=_timestamp(record["valid_from"], "valid_from"),
            valid_until=_timestamp(record["valid_until"], "valid_until"),
            prepared_configuration_id=_identifier(
                record["prepared_configuration_id"], "prepared-configuration ID"
            ),
            precision_policy_id=_identifier(
                record["precision_policy_id"], "precision-policy ID"
            ),
            resource_policy_id=_identifier(
                record["resource_policy_id"], "resource-policy ID"
            ),
            checkpoint_policy_id=_identifier(
                record["checkpoint_policy_id"], "checkpoint-policy ID"
            ),
            output_policy_id=_identifier(record["output_policy_id"], "output-policy ID"),
            repository_id=_identifier(record["repository_id"], "repository ID"),
            scheduler_id=_identifier(record["scheduler_id"], "scheduler ID"),
            auth_policy_id=_identifier(record["auth_policy_id"], "auth-policy ID"),
        )
        recorded_id = _identifier(record["spec_id"], "specification ID")
        if recorded_id != value.spec_id:
            raise ValueError(
                "Serialized run specification has an invalid content address."
            )
        return value

    @classmethod
    def from_json(cls, payload: str, /) -> ResolvedRunSpec:
        """Load strict canonical run-specification JSON."""
        return cls.from_record(_load_json_object(payload, "Resolved-run"))


def resolve_run_spec(
    scientific_dependencies: Sequence[SupportDependency],
    deployment_dependencies: Sequence[SupportDependency],
    /,
    *,
    release_index_id: str,
    profile_ids: Sequence[str],
    trust_policy_id: str,
    valid_at: int,
    valid_from: int,
    valid_until: int,
    prepared_configuration_id: str,
    precision_policy_id: str,
    resource_policy_id: str,
    checkpoint_policy_id: str,
    output_policy_id: str,
    repository_id: str,
    scheduler_id: str,
    auth_policy_id: str,
) -> ResolvedRunSpec:
    """Resolve explicit, already-qualified run bindings without defaults or aliases."""
    return ResolvedRunSpec(
        scientific_dependencies,
        deployment_dependencies,
        release_index_id=release_index_id,
        profile_ids=profile_ids,
        trust_policy_id=trust_policy_id,
        valid_at=valid_at,
        valid_from=valid_from,
        valid_until=valid_until,
        prepared_configuration_id=prepared_configuration_id,
        precision_policy_id=precision_policy_id,
        resource_policy_id=resource_policy_id,
        checkpoint_policy_id=checkpoint_policy_id,
        output_policy_id=output_policy_id,
        repository_id=repository_id,
        scheduler_id=scheduler_id,
        auth_policy_id=auth_policy_id,
    )


def load_resolved_run_spec(payload: str, /) -> ResolvedRunSpec:
    """Load and content-verify canonical resolved-run JSON."""
    return ResolvedRunSpec.from_json(payload)


__all__ = [
    "ResolvedRunSpec",
    "load_resolved_run_spec",
    "resolve_run_spec",
]
