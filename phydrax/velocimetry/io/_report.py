#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Literal

import equinox as eqx

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


AdapterDirection = Literal["import", "export"]
AdapterLossCategory = Literal["dropped", "synthesized", "transformed", "unsupported"]


class AdapterStatus(IntEnum):
    """Outcome of an external-format conversion."""

    LOSSLESS = 0
    DECLARED_LOSS = 1
    UNSUPPORTED_REQUIRED_SEMANTIC = 2
    MALFORMED_SOURCE = 3
    OPTIONAL_DEPENDENCY_UNAVAILABLE = 4
    INCONSISTENT_SOURCE = 5


class AdapterLoss(StrictModule, NonTrainableState):
    """One exact semantic change made by an external-format adapter."""

    path: str = eqx.field(static=True)
    direction: AdapterDirection = eqx.field(static=True)
    category: AdapterLossCategory = eqx.field(static=True)
    rationale: str = eqx.field(static=True)
    changes_interpretation: bool = eqx.field(static=True)

    def __init__(
        self,
        path: str,
        direction: AdapterDirection,
        category: AdapterLossCategory,
        rationale: str,
        /,
        *,
        changes_interpretation: bool,
    ):
        path_ = str(path).strip()
        rationale_ = str(rationale).strip()
        if not path_ or not rationale_:
            raise ValueError("Adapter loss paths and rationales must be non-empty.")
        if direction not in ("import", "export"):
            raise ValueError("Adapter loss direction must be 'import' or 'export'.")
        if category not in ("dropped", "synthesized", "transformed", "unsupported"):
            raise ValueError("Unknown adapter loss category.")
        self.path = path_
        self.direction = direction
        self.category = category
        self.rationale = rationale_
        self.changes_interpretation = bool(changes_interpretation)


class AdapterReport(StrictModule, NonTrainableState):
    """Auditable status and semantic accounting for one conversion."""

    valid: bool = eqx.field(static=True)
    status: AdapterStatus = eqx.field(static=True)
    source_format: str = eqx.field(static=True)
    target_format: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    coordinate_mapping: tuple[str, ...] = eqx.field(static=True)
    preserved_fields: tuple[str, ...] = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    losses: tuple[AdapterLoss, ...] = eqx.field(static=True)

    def __init__(
        self,
        status: AdapterStatus,
        source_format: str,
        target_format: str,
        /,
        *,
        source_id: str,
        target_id: str,
        coordinate_mapping: Sequence[str] = (),
        preserved_fields: Sequence[str] = (),
        assumptions: Sequence[str] = (),
        losses: Sequence[AdapterLoss] = (),
    ):
        status_ = AdapterStatus(status)
        source_format_ = str(source_format).strip()
        target_format_ = str(target_format).strip()
        source_id_ = str(source_id).strip()
        target_id_ = str(target_id).strip()
        if not source_format_ or not target_format_ or not source_id_ or not target_id_:
            raise ValueError("Adapter formats and identities must be non-empty.")
        losses_ = tuple(losses)
        if not all(isinstance(item, AdapterLoss) for item in losses_):
            raise TypeError("losses must contain AdapterLoss values.")
        if status_ == AdapterStatus.LOSSLESS and losses_:
            raise ValueError("A lossless report cannot contain semantic losses.")
        if status_ == AdapterStatus.DECLARED_LOSS and not losses_:
            raise ValueError("A declared-loss report must enumerate its losses.")
        self.valid = status_ in (AdapterStatus.LOSSLESS, AdapterStatus.DECLARED_LOSS)
        self.status = status_
        self.source_format = source_format_
        self.target_format = target_format_
        self.source_id = source_id_
        self.target_id = target_id_
        self.coordinate_mapping = _strings(coordinate_mapping, "coordinate_mapping")
        self.preserved_fields = _strings(preserved_fields, "preserved_fields")
        self.assumptions = _strings(assumptions, "assumptions")
        self.losses = losses_


class AdapterError(ValueError):
    """Conversion failure with a machine-readable adapter status."""

    status: AdapterStatus

    def __init__(self, status: AdapterStatus, message: str, /):
        self.status = AdapterStatus(status)
        super().__init__(str(message))


def require_lossless(report: AdapterReport, /) -> None:
    """Reject a conversion result whose report declares semantic loss."""
    if report.status != AdapterStatus.LOSSLESS:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "The requested lossless conversion cannot represent every native semantic.",
        )


def _strings(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{owner} must contain unique non-empty strings.")
    return result


__all__ = [
    "AdapterDirection",
    "AdapterError",
    "AdapterLoss",
    "AdapterLossCategory",
    "AdapterReport",
    "AdapterStatus",
    "require_lossless",
]
