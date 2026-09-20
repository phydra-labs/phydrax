#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Closure classification, implementation, and release states."""

from __future__ import annotations

from dataclasses import dataclass

from .._fingerprint import canonical_fingerprint
from ._closure_requirement import CapabilityClosureRequirement, CapabilityGapResolution
from ._closure_taxonomy import ClosureDisposition, ClosureState
from ._registry import _capability_name


@dataclass(frozen=True, slots=True)
class CapabilityClosureMatrix:
    family: str
    requirements: tuple[CapabilityClosureRequirement, ...]
    resolutions: tuple[CapabilityGapResolution, ...]

    @classmethod
    def create(cls, family, requirements, resolutions, /):
        return cls(
            _capability_name(family, "family"),
            tuple(sorted(requirements, key=lambda value: value.requirement_id)),
            tuple(sorted(resolutions, key=lambda value: value.requirement_id)),
        )

    def __post_init__(self) -> None:
        requirement_ids = tuple(value.requirement_id for value in self.requirements)
        resolution_ids = tuple(value.requirement_id for value in self.resolutions)
        if not requirement_ids or len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("Closure requirements must be nonempty and unique.")
        if len(set(resolution_ids)) != len(resolution_ids) or set(
            resolution_ids
        ).difference(requirement_ids):
            raise ValueError(
                "Closure resolutions must be unique and reference known requirements."
            )

    @property
    def unclassified_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {value.requirement_id for value in self.requirements}.difference(
                    value.requirement_id for value in self.resolutions
                )
            )
        )

    @property
    def classified(self) -> bool:
        return not self.unclassified_requirement_ids

    @property
    def implementation_closed(self) -> bool:
        if not self.classified:
            return False
        requirement_by_id = {value.requirement_id: value for value in self.requirements}
        return all(
            value.disposition
            in (
                ClosureDisposition.IMPLEMENTED,
                ClosureDisposition.PROVIDER,
                ClosureDisposition.REJECTED,
            )
            and value.actual_depth
            >= requirement_by_id[value.requirement_id].minimum_depth
            for value in self.resolutions
        )

    @property
    def release_closed(self) -> bool:
        return self.implementation_closed and all(
            value.disposition
            in (ClosureDisposition.IMPLEMENTED, ClosureDisposition.REJECTED)
            and value.release_authorized
            and (
                value.disposition is ClosureDisposition.REJECTED
                or bool(value.evidence_ids)
            )
            for value in self.resolutions
        )

    @property
    def state(self) -> ClosureState:
        if self.release_closed:
            return ClosureState.RELEASE_CLOSED
        if self.implementation_closed:
            return ClosureState.IMPLEMENTATION_CLOSED
        if self.classified:
            return ClosureState.CLASSIFIED
        return ClosureState.UNCLASSIFIED

    @property
    def matrix_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "capability-closure-matrix",
            "family": self.family,
            "requirements": [value.to_record() for value in self.requirements],
            "resolutions": [value.to_record() for value in self.resolutions],
            "unclassified_requirement_ids": list(self.unclassified_requirement_ids),
            "classified": self.classified,
            "implementation_closed": self.implementation_closed,
            "release_closed": self.release_closed,
            "state": self.state.value,
        }
        return {**record, "matrix_id": self.matrix_id} if include_id else record


__all__ = ["CapabilityClosureMatrix"]
