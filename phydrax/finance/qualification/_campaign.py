#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification._evidence import (
    QualificationCoverageReport,
    QualificationEvidence,
    QualificationMatrix,
)
from ...qualification._registry import SupportTuple


_FINANCE_CAPABILITIES = frozenset(
    {
        "finance.market-resolution",
        "finance.curve-calibration",
        "finance.valuation",
        "finance.econometrics",
        "finance.portfolio",
        "finance.exposure-xva",
        "finance.execution",
        "finance.advanced",
    }
)
_REQUIRED_EVIDENCE = (
    ("data", "reference"),
    ("model", "scientific"),
    ("numerical", "unit"),
    ("use", "operational"),
)


def _support_tuples(values: Sequence[SupportTuple], /) -> tuple[SupportTuple, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError("support_tuples must be a sequence of SupportTuple values.")
    supports = tuple(values)
    if not supports or any(not isinstance(item, SupportTuple) for item in supports):
        raise TypeError("support_tuples must contain at least one SupportTuple.")
    if any(item.capability not in _FINANCE_CAPABILITIES for item in supports):
        raise ValueError("Finance qualification accepts only finance support tuples.")
    supports = tuple(sorted(supports, key=lambda item: item.support_tuple_id))
    if len({item.support_tuple_id for item in supports}) != len(supports):
        raise ValueError("Finance qualification support tuples must be unique.")
    return supports


class FinanceQualificationMatrix(StrictModule, NonTrainableState):
    """Exact finance support tuples and their four independent evidence planes."""

    support_tuples: tuple[SupportTuple, ...]
    qualification_matrix: QualificationMatrix
    matrix_id: str = eqx.field(static=True)

    def __init__(
        self,
        support_tuples: Sequence[SupportTuple],
        qualification_matrix: QualificationMatrix,
        /,
    ):
        supports = _support_tuples(support_tuples)
        if not isinstance(qualification_matrix, QualificationMatrix):
            raise TypeError("qualification_matrix must be a QualificationMatrix.")
        expected = _predicates(supports)
        if qualification_matrix.predicates != QualificationMatrix(expected).predicates:
            raise ValueError(
                "Finance qualification matrices must cover data, model, numerical, "
                "and use evidence for every exact support tuple."
            )
        self.support_tuples = supports
        self.qualification_matrix = qualification_matrix
        self.matrix_id = qualification_matrix.matrix_id

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "finance-qualification-matrix",
            "support_tuples": [item.to_record() for item in self.support_tuples],
            "qualification_matrix": self.qualification_matrix.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        """Return the deterministic matrix and exact qualified support coordinates."""
        return {**self._content_record(), "matrix_id": self.matrix_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> FinanceQualificationMatrix:
        """Reconstruct and verify a serialized finance qualification matrix."""
        if not isinstance(record, Mapping):
            raise TypeError("Finance qualification matrix record must be a mapping.")
        expected_fields = {
            "kind",
            "support_tuples",
            "qualification_matrix",
            "matrix_id",
        }
        if set(record) != expected_fields:
            raise ValueError("Finance qualification matrix fields are not canonical.")
        if record.get("kind") != "finance-qualification-matrix":
            raise ValueError("Record is not a finance qualification matrix.")
        supports_record = record["support_tuples"]
        matrix_record = record["qualification_matrix"]
        if not isinstance(supports_record, Sequence) or isinstance(supports_record, str):
            raise TypeError("Serialized support_tuples must be a sequence.")
        if not isinstance(matrix_record, Mapping):
            raise TypeError("Serialized qualification_matrix must be a mapping.")
        supports = []
        for item in supports_record:
            if not isinstance(item, Mapping):
                raise TypeError("Serialized support tuples must be mappings.")
            supports.append(SupportTuple.from_record(item))
        value = cls(supports, QualificationMatrix.from_record(matrix_record))
        if record.get("matrix_id") != value.matrix_id:
            raise ValueError("Serialized finance matrix has an invalid content address.")
        return value


class FinanceQualificationCampaign(StrictModule, NonTrainableState):
    """One immutable evaluation of existing reviewed qualification evidence."""

    matrix: FinanceQualificationMatrix
    evidence: tuple[QualificationEvidence, ...]
    coverage: QualificationCoverageReport
    evaluated_at: int = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: FinanceQualificationMatrix,
        evidence: Sequence[QualificationEvidence],
        coverage: QualificationCoverageReport,
        /,
        *,
        evaluated_at: int,
    ):
        if not isinstance(matrix, FinanceQualificationMatrix):
            raise TypeError("matrix must be a FinanceQualificationMatrix.")
        if not isinstance(evidence, Sequence) or isinstance(evidence, str):
            raise TypeError("evidence must be a sequence of QualificationEvidence.")
        records = tuple(evidence)
        if any(not isinstance(item, QualificationEvidence) for item in records):
            raise TypeError("evidence must contain QualificationEvidence values.")
        records = tuple(sorted(records, key=lambda item: item.evidence_id))
        if len({item.evidence_id for item in records}) != len(records):
            raise ValueError("Finance campaign evidence IDs must be unique.")
        if not isinstance(coverage, QualificationCoverageReport):
            raise TypeError("coverage must be a QualificationCoverageReport.")
        if coverage.matrix_id != matrix.qualification_matrix.matrix_id:
            raise ValueError("Coverage report belongs to another qualification matrix.")
        if isinstance(evaluated_at, bool) or not isinstance(evaluated_at, int):
            raise TypeError("evaluated_at must be an integer timestamp.")
        if evaluated_at < 0 or coverage.evaluated_at != evaluated_at:
            raise ValueError(
                "evaluated_at must match the non-negative coverage timestamp."
            )
        expected = matrix.qualification_matrix.evaluate(records, at_time=evaluated_at)
        if expected.report_id != coverage.report_id:
            raise ValueError("Coverage report does not match the supplied evidence.")
        self.matrix = matrix
        self.evidence = records
        self.coverage = coverage
        self.evaluated_at = evaluated_at
        self.campaign_id = canonical_fingerprint(self._content_record())

    @property
    def passed(self) -> bool:
        """Whether every required evidence plane passed for every support tuple."""
        return self.coverage.passed

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "finance-qualification-campaign",
            "matrix": self.matrix.to_record(),
            "evidence": [item.to_record() for item in self.evidence],
            "coverage": self.coverage.to_record(),
            "evaluated_at": self.evaluated_at,
        }

    def to_record(self) -> dict[str, object]:
        """Return complete, deterministic campaign evidence and its coverage result."""
        return {**self._content_record(), "campaign_id": self.campaign_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> FinanceQualificationCampaign:
        """Reconstruct and independently re-evaluate a serialized campaign."""
        if not isinstance(record, Mapping):
            raise TypeError("Finance qualification campaign record must be a mapping.")
        expected_fields = {
            "kind",
            "matrix",
            "evidence",
            "coverage",
            "evaluated_at",
            "campaign_id",
        }
        if set(record) != expected_fields:
            raise ValueError("Finance qualification campaign fields are not canonical.")
        if record.get("kind") != "finance-qualification-campaign":
            raise ValueError("Record is not a finance qualification campaign.")
        matrix_record = record["matrix"]
        evidence_record = record["evidence"]
        coverage_record = record["coverage"]
        if not isinstance(matrix_record, Mapping) or not isinstance(
            coverage_record, Mapping
        ):
            raise TypeError("Serialized matrix and coverage must be mappings.")
        if not isinstance(evidence_record, Sequence) or isinstance(evidence_record, str):
            raise TypeError("Serialized evidence must be a sequence.")
        evaluated_at = record["evaluated_at"]
        if isinstance(evaluated_at, bool) or not isinstance(evaluated_at, int):
            raise TypeError("Serialized evaluated_at must be an integer.")
        evidence = []
        for item in evidence_record:
            if not isinstance(item, Mapping):
                raise TypeError("Serialized evidence entries must be mappings.")
            evidence.append(QualificationEvidence.from_record(item))
        value = cls(
            FinanceQualificationMatrix.from_record(matrix_record),
            evidence,
            QualificationCoverageReport.from_record(coverage_record),
            evaluated_at=evaluated_at,
        )
        if record.get("campaign_id") != value.campaign_id:
            raise ValueError(
                "Serialized finance campaign has an invalid content address."
            )
        return value


def _predicates(support_tuples: Sequence[SupportTuple], /) -> dict[str, dict[str, str]]:
    predicates: dict[str, dict[str, str]] = {}
    for support in support_tuples:
        prefix = f"{support.capability}:{support.support_tuple_id}"
        for plane, evidence_kind in _REQUIRED_EVIDENCE:
            predicates[f"{prefix}:{plane}"] = {
                "evidence_kind": evidence_kind,
                "subject_id": support.support_tuple_id,
            }
    return predicates


def build_finance_qualification_matrix(
    support_tuples: Sequence[SupportTuple], /
) -> FinanceQualificationMatrix:
    """Build a fail-closed four-plane matrix for exact finance support tuples."""
    supports = _support_tuples(support_tuples)
    qualification = QualificationMatrix(_predicates(supports))
    return FinanceQualificationMatrix(supports, qualification)


def evaluate_finance_campaign(
    matrix: FinanceQualificationMatrix,
    evidence: Sequence[QualificationEvidence],
    /,
    *,
    at_time: int,
) -> FinanceQualificationCampaign:
    """Evaluate reviewed records; absent evidence remains explicitly inconclusive."""
    if not isinstance(matrix, FinanceQualificationMatrix):
        raise TypeError("matrix must be a FinanceQualificationMatrix.")
    coverage = matrix.qualification_matrix.evaluate(evidence, at_time=at_time)
    return FinanceQualificationCampaign(
        matrix,
        evidence,
        coverage,
        evaluated_at=at_time,
    )


__all__ = [
    "FinanceQualificationCampaign",
    "FinanceQualificationMatrix",
    "build_finance_qualification_matrix",
    "evaluate_finance_campaign",
]
