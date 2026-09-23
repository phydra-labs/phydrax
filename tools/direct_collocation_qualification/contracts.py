#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from phydrax._fingerprint import canonical_fingerprint


def _finite_json(value: Any, owner: str, /) -> None:
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{owner} contains a nonfinite value.")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{owner}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_json(item, f"{owner}[{index}]")


def _validate_metadata(metadata: dict[str, Any], /) -> None:
    required = (
        "qualification",
        "source_id",
        "package_fingerprint",
        "platform",
        "python",
        "jax",
        "numpy",
        "dtype",
        "backends",
    )
    missing = tuple(name for name in required if name not in metadata)
    if missing:
        raise ValueError(f"Qualification metadata is missing fields: {missing}.")
    for name in required[:-1]:
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"Qualification metadata {name!r} must be non-empty.")
    if "runtime" in metadata:
        runtime = metadata["runtime"]
        if not isinstance(runtime, dict) or not isinstance(
            runtime.get("fingerprint"), str
        ):
            raise ValueError("Qualification runtime metadata requires a fingerprint.")
    backends = metadata["backends"]
    if (
        not isinstance(backends, list)
        or not backends
        or any(not isinstance(value, str) or not value for value in backends)
        or len(set(backends)) != len(backends)
    ):
        raise ValueError("Qualification backends must be unique non-empty strings.")
    _finite_json(metadata, "qualification metadata")


@dataclass(frozen=True)
class DirectCollocationQualificationCase:
    case_id: str
    family: str
    reference_kind: str
    expected_feasible: bool
    replay_required: bool

    def identity_payload(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        for value, owner in (
            (self.case_id, "case_id"),
            (self.family, "family"),
            (self.reference_kind, "reference_kind"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{owner} must be a non-empty string.")


@dataclass(frozen=True)
class DirectCollocationQualificationRecord:
    case_id: str
    backend: str
    method_id: str
    successful: bool
    backend_status: int
    public_status: int
    false_success: bool
    false_failure: bool
    objective: float
    reference_error: float
    maximum_defect: float
    maximum_constraint_violation: float
    maximum_off_grid_defect: float
    replay_error: float
    derivative_action_error: float
    variables: int
    constraints: int
    jacobian_nonzeros: int
    dense_materialized: bool
    materialization_verified: bool
    elapsed_seconds: float
    certified: bool
    jit_verified: bool
    vmap_verified: bool
    refresh_verified: bool
    record_id: str

    @classmethod
    def create(cls, **values):
        payload = dict(values)
        payload.pop("record_id", None)
        _finite_json(payload, "qualification record")
        return cls(**payload, record_id=canonical_fingerprint(payload))

    def __post_init__(self) -> None:
        if not self.case_id or not self.backend or not self.method_id:
            raise ValueError("Qualification record identities must be non-empty.")
        for name in ("variables", "constraints", "jacobian_nonzeros"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.false_success and self.false_failure:
            raise ValueError("A record cannot be both a false success and false failure.")
        if self.false_success and not self.successful:
            raise ValueError("false_success requires a successful provider result.")
        if self.false_failure and self.successful:
            raise ValueError("false_failure requires an unsuccessful provider result.")
        if self.certified and (
            not self.successful or self.false_success or self.false_failure
        ):
            raise ValueError("Certified records require unambiguous provider success.")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be non-negative.")

    def verify(self) -> None:
        payload = asdict(self)
        observed = payload.pop("record_id")
        _finite_json(payload, f"record {self.case_id}")
        if canonical_fingerprint(payload) != observed:
            raise ValueError(
                f"Qualification record {self.case_id!r} fingerprint mismatch."
            )


@dataclass(frozen=True)
class DirectCollocationQualificationArtifact:
    metadata: dict[str, Any]
    cases: tuple[DirectCollocationQualificationCase, ...]
    records: tuple[DirectCollocationQualificationRecord, ...]
    graduation: dict[str, Any]
    artifact_id: str

    @classmethod
    def create(
        cls,
        *,
        metadata: dict[str, Any],
        cases: tuple[DirectCollocationQualificationCase, ...],
        records: tuple[DirectCollocationQualificationRecord, ...],
        graduation: dict[str, Any],
    ):
        _validate_artifact_evidence(metadata, cases, records, graduation)
        payload = {
            "metadata": metadata,
            "cases": [asdict(case) for case in cases],
            "records": [asdict(record) for record in records],
            "graduation": graduation,
        }
        _finite_json(payload, "qualification artifact")
        return cls(
            metadata,
            cases,
            records,
            graduation,
            canonical_fingerprint(payload),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any], /):
        expected = {"metadata", "cases", "records", "graduation", "artifact_id"}
        if set(value) != expected:
            raise ValueError(f"Qualification artifact keys must be {sorted(expected)}.")
        cases = tuple(
            DirectCollocationQualificationCase(**case) for case in value["cases"]
        )
        records = tuple(
            DirectCollocationQualificationRecord(**record) for record in value["records"]
        )
        artifact = cls(
            dict(value["metadata"]),
            cases,
            records,
            dict(value["graduation"]),
            str(value["artifact_id"]),
        )
        artifact.verify(required_case_ids=tuple(case.case_id for case in cases))
        return artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "cases": [asdict(case) for case in self.cases],
            "records": [asdict(record) for record in self.records],
            "graduation": self.graduation,
            "artifact_id": self.artifact_id,
        }

    def verify(self, *, required_case_ids: tuple[str, ...]) -> None:
        _validate_metadata(self.metadata)
        identifiers = [case.case_id for case in self.cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Qualification artifact contains duplicate cases.")
        if tuple(sorted(identifiers)) != tuple(sorted(required_case_ids)):
            raise ValueError("Qualification artifact case coverage is incomplete.")
        record_keys = [(record.case_id, record.backend) for record in self.records]
        if len(set(record_keys)) != len(record_keys):
            raise ValueError(
                "Qualification artifact contains duplicate case/backend records."
            )
        unknown = sorted({record.case_id for record in self.records} - set(identifiers))
        if unknown:
            raise ValueError(f"Qualification records reference unknown cases: {unknown}.")
        expected_record_keys = {
            (case_id, backend)
            for case_id in identifiers
            for backend in self.metadata["backends"]
        }
        if set(record_keys) != expected_record_keys:
            missing = sorted(expected_record_keys - set(record_keys))
            extra = sorted(set(record_keys) - expected_record_keys)
            raise ValueError(
                "Qualification artifact case/backend coverage is incomplete: "
                f"missing={missing}, extra={extra}."
            )
        for record in self.records:
            record.verify()
        _validate_artifact_evidence(
            self.metadata, self.cases, self.records, self.graduation
        )
        payload = self.to_dict()
        observed = payload.pop("artifact_id")
        _finite_json(payload, "qualification artifact")
        if canonical_fingerprint(payload) != observed:
            raise ValueError("Qualification artifact fingerprint mismatch.")


def _validate_artifact_evidence(
    metadata: dict[str, Any],
    cases: tuple[DirectCollocationQualificationCase, ...],
    records: tuple[DirectCollocationQualificationRecord, ...],
    graduation: dict[str, Any],
) -> None:
    setups = metadata.get("setups")
    if not isinstance(setups, list) or len(setups) != len(cases):
        raise ValueError("Qualification metadata requires one setup identity per case.")
    setup_case_ids = tuple(
        item.get("case_id") for item in setups if isinstance(item, dict)
    )
    if setup_case_ids != tuple(case.case_id for case in cases):
        raise ValueError("Qualification setup identities do not match case order.")
    for item in setups:
        if set(item) != {"case_id", "problem_id", "plan_id"} or any(
            not isinstance(item[name], str) or not item[name]
            for name in ("case_id", "problem_id", "plan_id")
        ):
            raise ValueError("Qualification setup identities are incomplete.")
    case_by_id = {case.case_id: case for case in cases}
    for record in records:
        case = case_by_id.get(record.case_id)
        if case is None:
            continue
        if record.false_success != (
            record.successful and (not case.expected_feasible or not record.certified)
        ):
            raise ValueError(
                f"Record {record.case_id!r} false_success is inconsistent with evidence."
            )
        if record.false_failure != (case.expected_feasible and not record.successful):
            raise ValueError(
                f"Record {record.case_id!r} false_failure is inconsistent with evidence."
            )
    from .graduation import evaluate_direct_collocation_graduation

    expected = evaluate_direct_collocation_graduation(
        records,
        documentation_complete=bool(graduation.get("documentation_complete", False)),
        artifact_present=bool(graduation.get("artifact_present", False)),
    )
    if graduation != expected:
        raise ValueError("Qualification graduation does not match record evidence.")


__all__ = [
    "DirectCollocationQualificationArtifact",
    "DirectCollocationQualificationCase",
    "DirectCollocationQualificationRecord",
]
