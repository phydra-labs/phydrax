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
    elapsed_seconds: float
    record_id: str

    @classmethod
    def create(cls, **values):
        payload = dict(values)
        payload.pop("record_id", None)
        _finite_json(payload, "qualification record")
        return cls(**payload, record_id=canonical_fingerprint(payload))

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
        _validate_metadata(metadata)
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
        for record in self.records:
            record.verify()
        payload = self.to_dict()
        observed = payload.pop("artifact_id")
        _finite_json(payload, "qualification artifact")
        if canonical_fingerprint(payload) != observed:
            raise ValueError("Qualification artifact fingerprint mismatch.")


__all__ = [
    "DirectCollocationQualificationArtifact",
    "DirectCollocationQualificationCase",
    "DirectCollocationQualificationRecord",
]
