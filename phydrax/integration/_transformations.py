#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx

from .._strict import StrictModule


class MeasureTransformationRecord(StrictModule):
    """Ordered evidence for one finite-measure transformation."""

    diagnostics: Any
    kind: str = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)
    target_provenance: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        diagnostics: Any,
        /,
        *,
        source_provenance: str,
        target_provenance: str,
    ):
        kind_ = str(kind)
        source = str(source_provenance)
        target = str(target_provenance)
        if not kind_ or not source or not target:
            raise ValueError("Transformation identities and provenance must be non-empty.")
        self.kind = kind_
        self.diagnostics = diagnostics
        self.source_provenance = source
        self.target_provenance = target


class TransformedIntegrationDiagnostics(StrictModule):
    """Ordered measure transformations paired with downstream reduction evidence."""

    transformations: tuple[MeasureTransformationRecord, ...]
    reduction: Any

    def __init__(
        self,
        transformations: tuple[MeasureTransformationRecord, ...],
        reduction: Any,
        /,
    ):
        transformations_ = tuple(transformations)
        if any(
            not isinstance(item, MeasureTransformationRecord)
            for item in transformations_
        ):
            raise TypeError(
                "transformations must contain MeasureTransformationRecord values."
            )
        self.transformations = transformations_
        self.reduction = reduction


__all__ = ["MeasureTransformationRecord", "TransformedIntegrationDiagnostics"]
