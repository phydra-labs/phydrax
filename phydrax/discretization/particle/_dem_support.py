#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class DEMSupportStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    QUALIFIED = "qualified"
    PRODUCTION = "production"
    UNSUPPORTED = "unsupported"


class DEMSupportMatrixEntry(StrictModule, NonTrainableState):
    dimension: int = eqx.field(static=True)
    normal_law: str = eqx.field(static=True)
    tangential_law: str = eqx.field(static=True)
    neighborhood: str = eqx.field(static=True)
    kernel_backend: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    sensitivity: str = eqx.field(static=True)
    barrier: str = eqx.field(static=True)
    status: DEMSupportStatus = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    entry_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        dimension: int,
        normal_law: str,
        tangential_law: str = "none",
        neighborhood: str,
        kernel_backend: str = "reference",
        precision: str = "float64",
        sensitivity: str = "forward",
        barrier: str = "none",
        status: DEMSupportStatus = DEMSupportStatus.EXPERIMENTAL,
        evidence_ids: Sequence[str] = (),
    ):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("DEM support entries require dimension 2 or 3.")
        if not isinstance(status, DEMSupportStatus):
            raise TypeError("status must be a DEMSupportStatus.")
        values = tuple(
            str(value)
            for value in (
                normal_law,
                tangential_law,
                neighborhood,
                kernel_backend,
                precision,
                sensitivity,
                barrier,
            )
        )
        if any(not value for value in values):
            raise ValueError("DEM support matrix labels must be nonempty.")
        evidence = tuple(str(value) for value in evidence_ids)
        if any(not value for value in evidence) or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids must be unique nonempty strings.")
        if (
            status in (DEMSupportStatus.QUALIFIED, DEMSupportStatus.PRODUCTION)
            and not evidence
        ):
            raise ValueError("Qualified and production support entries require evidence.")
        (
            normal,
            tangential,
            neighborhood_,
            backend,
            precision_,
            sensitivity_,
            barrier_,
        ) = values
        self.dimension = dimension_
        self.normal_law = normal
        self.tangential_law = tangential
        self.neighborhood = neighborhood_
        self.kernel_backend = backend
        self.precision = precision_
        self.sensitivity = sensitivity_
        self.barrier = barrier_
        self.status = status
        self.evidence_ids = evidence
        self.entry_id = canonical_fingerprint(
            {
                "kind": "dem-support-matrix-entry",
                "dimension": dimension_,
                "normal": normal,
                "tangential": tangential,
                "neighborhood": neighborhood_,
                "backend": backend,
                "precision": precision_,
                "sensitivity": sensitivity_,
                "barrier": barrier_,
                "status": status.value,
                "evidence": list(evidence),
            }
        )


class DEMSupportMatrix(StrictModule, NonTrainableState):
    entries: tuple[DEMSupportMatrixEntry, ...]
    matrix_id: str = eqx.field(static=True)

    def __init__(self, entries: Sequence[DEMSupportMatrixEntry], /):
        values = tuple(entries)
        if not values or any(
            not isinstance(value, DEMSupportMatrixEntry) for value in values
        ):
            raise TypeError("entries must contain DEMSupportMatrixEntry values.")
        configurations = tuple(
            (
                value.dimension,
                value.normal_law,
                value.tangential_law,
                value.neighborhood,
                value.kernel_backend,
                value.precision,
                value.sensitivity,
                value.barrier,
            )
            for value in values
        )
        if len(set(configurations)) != len(configurations):
            raise ValueError("DEM support matrix configurations must be unique.")
        self.entries = values
        self.matrix_id = canonical_fingerprint(
            {
                "kind": "dem-support-matrix",
                "entries": [value.entry_id for value in values],
            }
        )

    @property
    def production_ready(self) -> bool:
        return all(value.status is DEMSupportStatus.PRODUCTION for value in self.entries)

    def entries_with_status(
        self, status: DEMSupportStatus, /
    ) -> tuple[DEMSupportMatrixEntry, ...]:
        if not isinstance(status, DEMSupportStatus):
            raise TypeError("status must be a DEMSupportStatus.")
        return tuple(value for value in self.entries if value.status is status)


__all__ = [
    "DEMSupportMatrix",
    "DEMSupportMatrixEntry",
    "DEMSupportStatus",
]
