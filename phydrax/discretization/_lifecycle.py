#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

from .._precision import PrecisionEvidenceEnvelope
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    normalized_capabilities,
    PreparationReport,
)
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace
from ._support import DiscreteSupport


class AbstractDiscretizationPlan(StrictModule, NonTrainableState):
    """Symbolic, identity-bearing plan for one numerical discretization."""

    key: AbstractAttribute[DiscretizationKey]
    capabilities: AbstractAttribute[tuple[DiscretizationCapability, ...]]
    plan_id: AbstractAttribute[str]


class AbstractPreparedDiscretization(StrictModule, NonTrainableState):
    """Prepared finite support, spaces, measures, and method-specific state."""

    key: AbstractAttribute[DiscretizationKey]
    support: AbstractAttribute[DiscreteSupport]
    field_spaces: AbstractAttribute[tuple[DiscreteFieldSpace, ...]]
    measures: AbstractAttribute[tuple[DiscreteMeasure, ...]]
    capabilities: AbstractAttribute[tuple[DiscretizationCapability, ...]]
    plan_id: AbstractAttribute[str]
    prepared_id: AbstractAttribute[str]
    numeric_version: AbstractAttribute[str]
    preparation: AbstractAttribute[PreparationReport]

    @property
    def precision_evidence(self) -> PrecisionEvidenceEnvelope | None:
        """Observed execution precision, when the method provides it."""
        return None

    @property
    def precision_evidence_id(self) -> str | None:
        """Identity of observed execution precision, when the method provides it."""
        evidence = self.precision_evidence
        return None if evidence is None else evidence.evidence_id

    @property
    def resource_evidence_id(self) -> str | None:
        """Identity of resource assumptions or measurements, when available."""
        return None


def validate_prepared_metadata(
    *,
    key: DiscretizationKey,
    support: DiscreteSupport,
    field_spaces: Sequence[DiscreteFieldSpace],
    measures: Sequence[DiscreteMeasure],
    capabilities: Sequence[DiscretizationCapability],
    preparation: PreparationReport,
) -> tuple[
    tuple[DiscreteFieldSpace, ...],
    tuple[DiscreteMeasure, ...],
    tuple[DiscretizationCapability, ...],
]:
    """Validate common prepared-discretization metadata."""
    if not isinstance(key, DiscretizationKey):
        raise TypeError("key must be a DiscretizationKey.")
    if not isinstance(support, DiscreteSupport):
        raise TypeError("support must be a DiscreteSupport.")
    spaces = tuple(field_spaces)
    if not spaces or not all(isinstance(space, DiscreteFieldSpace) for space in spaces):
        raise TypeError(
            "field_spaces must contain one or more DiscreteFieldSpace values."
        )
    names = tuple(space.name for space in spaces)
    identifiers = tuple(space.field_space_id for space in spaces)
    if len(set(names)) != len(names) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Prepared field-space names and IDs must be unique.")
    if any(space.support_id != support.support_id for space in spaces):
        raise ValueError(
            "Every prepared field space must belong to the prepared support."
        )
    measures_ = tuple(measures)
    if not all(isinstance(measure, DiscreteMeasure) for measure in measures_):
        raise TypeError("measures must contain DiscreteMeasure values.")
    measure_ids = tuple(measure.measure_id for measure in measures_)
    if len(set(measure_ids)) != len(measure_ids):
        raise ValueError("Prepared measure IDs must be unique.")
    if any(measure.support_id != support.support_id for measure in measures_):
        raise ValueError("Every prepared measure must belong to the prepared support.")
    capabilities_ = normalized_capabilities(tuple(capabilities))
    if not isinstance(preparation, PreparationReport):
        raise TypeError("preparation must be a PreparationReport.")
    if preparation.capabilities != capabilities_:
        raise ValueError("Preparation capabilities must match prepared capabilities.")
    return spaces, measures_, capabilities_


def require_capabilities(
    prepared: AbstractPreparedDiscretization,
    required: Sequence[DiscretizationCapability],
    /,
) -> None:
    """Reject an operation when its structural capabilities are unavailable."""
    if not isinstance(prepared, AbstractPreparedDiscretization):
        raise TypeError("prepared must be an AbstractPreparedDiscretization.")
    required_ = normalized_capabilities(tuple(required))
    available = set(prepared.capabilities)
    missing = tuple(value for value in required_ if value not in available)
    if missing:
        names = ", ".join(str(value) for value in missing)
        raise ValueError(
            f"Prepared discretization {prepared.prepared_id!r} lacks capabilities: "
            f"{names}."
        )


def field_space(
    prepared: AbstractPreparedDiscretization,
    name: str,
    /,
) -> DiscreteFieldSpace:
    """Resolve one named field space without a dynamic mapping allocation."""
    for space in prepared.field_spaces:
        if space.name == name:
            return space
    raise KeyError(f"Unknown discrete field space {name!r}.")


__all__ = [
    "AbstractDiscretizationPlan",
    "AbstractPreparedDiscretization",
    "field_space",
    "require_capabilities",
]
