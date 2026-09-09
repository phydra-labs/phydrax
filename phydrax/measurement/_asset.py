#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Acquisition, derivation, and governed measurement assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .._fingerprint import canonical_fingerprint, canonical_mapping
from ..artifacts import DifferentiationContract
from ..qualification import ReferenceArtifactManifest
from ._field import QuantityField
from ._quantity import canonical_quantity_text


class DataOrigin(StrEnum):
    EXTERNAL = "external"
    SYNTHETIC = "synthetic"


class DataStage(StrEnum):
    RAW = "raw"
    CALIBRATED = "calibrated"
    DERIVED = "derived"
    RECONSTRUCTED = "reconstructed"
    INFERRED = "inferred"


@dataclass(frozen=True, slots=True)
class AcquisitionIdentity:
    acquisition_id: str
    series_id: str
    modality: str
    protocol_id: str
    instrument_id: str | None = None
    calibration_ids: tuple[str, ...] = ()
    clock_id: str | None = None
    identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        acquisition = canonical_quantity_text(self.acquisition_id, "acquisition_id")
        series = canonical_quantity_text(self.series_id, "series_id")
        modality = canonical_quantity_text(self.modality, "modality")
        protocol = canonical_quantity_text(self.protocol_id, "protocol_id")
        instrument = (
            None
            if self.instrument_id is None
            else canonical_quantity_text(self.instrument_id, "instrument_id")
        )
        calibrations = tuple(
            canonical_quantity_text(identifier, "calibration_id")
            for identifier in self.calibration_ids
        )
        if len(calibrations) != len(set(calibrations)):
            raise ValueError("calibration_ids must be unique.")
        clock = (
            None
            if self.clock_id is None
            else canonical_quantity_text(self.clock_id, "clock_id")
        )
        object.__setattr__(self, "acquisition_id", acquisition)
        object.__setattr__(self, "series_id", series)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "instrument_id", instrument)
        object.__setattr__(self, "calibration_ids", calibrations)
        object.__setattr__(self, "clock_id", clock)
        object.__setattr__(
            self,
            "identity_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-acquisition",
                    "acquisition": acquisition,
                    "series": series,
                    "modality": modality,
                    "protocol": protocol,
                    "instrument": instrument,
                    "calibrations": list(calibrations),
                    "clock": clock,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DerivationRecord:
    origin: DataOrigin
    stage: DataStage
    parent_ids: tuple[str, ...] = ()
    transformation_id: str | None = None
    adapter_report_ids: tuple[str, ...] = ()
    calibration_ids: tuple[str, ...] = ()
    differentiation: DifferentiationContract | None = None
    derivation_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.origin, DataOrigin):
            raise TypeError("origin must be DataOrigin.")
        if not isinstance(self.stage, DataStage):
            raise TypeError("stage must be DataStage.")
        parents = tuple(
            canonical_quantity_text(identifier, "parent_id")
            for identifier in self.parent_ids
        )
        reports = tuple(
            canonical_quantity_text(identifier, "adapter_report_id")
            for identifier in self.adapter_report_ids
        )
        calibrations = tuple(
            canonical_quantity_text(identifier, "calibration_id")
            for identifier in self.calibration_ids
        )
        for values, name in (
            (parents, "parent_ids"),
            (reports, "adapter_report_ids"),
            (calibrations, "calibration_ids"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique.")
        transformation = (
            None
            if self.transformation_id is None
            else canonical_quantity_text(self.transformation_id, "transformation_id")
        )
        if self.origin is DataOrigin.SYNTHETIC and transformation is None:
            raise ValueError("Synthetic data requires a generator transformation_id.")
        if parents and transformation is None:
            raise ValueError("Derived parent links require transformation_id.")
        if self.differentiation is not None and not isinstance(
            self.differentiation, DifferentiationContract
        ):
            raise TypeError("differentiation must be DifferentiationContract or None.")
        object.__setattr__(self, "parent_ids", parents)
        object.__setattr__(self, "adapter_report_ids", reports)
        object.__setattr__(self, "calibration_ids", calibrations)
        object.__setattr__(self, "transformation_id", transformation)
        object.__setattr__(
            self,
            "derivation_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-derivation",
                    "origin": self.origin.value,
                    "stage": self.stage.value,
                    "parents": list(parents),
                    "transformation": transformation,
                    "adapter_reports": list(reports),
                    "calibrations": list(calibrations),
                    "differentiation": None
                    if self.differentiation is None
                    else self.differentiation.contract_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MeasurementAsset:
    """One governed externally sourced or synthetic quantity field."""

    asset_id: str
    field: QuantityField
    acquisition: AcquisitionIdentity | None
    references: tuple[ReferenceArtifactManifest, ...]
    derivation: DerivationRecord
    intended_use: str = "research"
    metadata: Mapping[str, Any] | None = None
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        asset_id = canonical_quantity_text(self.asset_id, "asset_id")
        if not isinstance(self.field, QuantityField):
            raise TypeError("field must be QuantityField.")
        if self.acquisition is not None and not isinstance(
            self.acquisition, AcquisitionIdentity
        ):
            raise TypeError("acquisition must be AcquisitionIdentity or None.")
        references = tuple(self.references)
        if not references or any(
            not isinstance(reference, ReferenceArtifactManifest)
            for reference in references
        ):
            raise TypeError(
                "references must contain at least one ReferenceArtifactManifest."
            )
        if len({reference.manifest_id for reference in references}) != len(references):
            raise ValueError("references must be unique.")
        if not isinstance(self.derivation, DerivationRecord):
            raise TypeError("derivation must be DerivationRecord.")
        intended_use = canonical_quantity_text(self.intended_use, "intended_use")
        requested = {
            "research": {},
            "commercial": {"commercial_use": True},
            "training": {"training_use": True},
            "redistribution": {"redistribution": True},
            "export": {"export": True},
        }
        if intended_use not in requested:
            raise ValueError(
                "intended_use must be research, commercial, training, redistribution, or export."
            )
        for reference in references:
            reference.require_rights(**requested[intended_use])
        metadata = canonical_mapping({} if self.metadata is None else self.metadata)
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "intended_use", intended_use)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(
            self,
            "content_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-asset",
                    "asset": asset_id,
                    "field": self.field.content_id,
                    "acquisition": None
                    if self.acquisition is None
                    else self.acquisition.identity_id,
                    "references": [reference.manifest_id for reference in references],
                    "derivation": self.derivation.derivation_id,
                    "intended_use": intended_use,
                    "metadata": metadata,
                }
            ),
        )

    @classmethod
    def from_single_reference(
        cls,
        asset_id: str,
        field: QuantityField,
        reference: ReferenceArtifactManifest,
        derivation: DerivationRecord,
        /,
        *,
        acquisition: AcquisitionIdentity | None = None,
        intended_use: str = "research",
        metadata: Mapping[str, Any] | None = None,
    ) -> MeasurementAsset:
        return cls(
            asset_id,
            field,
            acquisition,
            (reference,),
            derivation,
            intended_use,
            metadata,
        )


__all__ = [
    "AcquisitionIdentity",
    "DataOrigin",
    "DataStage",
    "DerivationRecord",
    "MeasurementAsset",
]
