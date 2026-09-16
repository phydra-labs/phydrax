#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...particle_physics import EventWeightSet, PreparedParticleEvents
from .._report import AdapterFormatProfile, AdapterReport, AdapterStatus


class HEPOptionalDependencyError(ImportError):
    """A requested HEP format requires an unavailable optional dependency."""


class HEPColumnProfile(StrictModule, NonTrainableState):
    """Exact field mapping for one concrete columnar event representation."""

    format_name: str = eqx.field(static=True)
    field_mapping: tuple[tuple[str, str], ...] = eqx.field(static=True)
    unit_qualifiers: tuple[tuple[str, str], ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        format_name: str,
        field_mapping: Mapping[str, str],
        /,
        *,
        unit_qualifiers: Mapping[str, str],
    ):
        format_name_ = str(format_name).strip()
        fields = tuple(
            sorted(
                (str(key).strip(), str(value).strip())
                for key, value in field_mapping.items()
            )
        )
        units = tuple(
            sorted(
                (str(key).strip(), str(value).strip())
                for key, value in unit_qualifiers.items()
            )
        )
        required = {
            "event_ids",
            "subevent_ids",
            "event_active",
            "pdg_ids",
            "roles",
            "provider_status",
            "momenta",
            "rest_energies",
            "particle_active",
            "mother_indices",
            "production_vertex_indices",
            "end_vertex_indices",
            "color_flow",
            "production_vertices",
            "vertex_active",
            "overflow",
        }
        if not format_name_ or any(not key or not value for key, value in fields + units):
            raise ValueError("Column profile names and mappings must be non-empty.")
        if {key for key, _ in fields} != required:
            raise ValueError(
                "field_mapping must contain exactly the canonical event fields."
            )
        self.format_name = format_name_
        self.field_mapping = fields
        self.unit_qualifiers = units
        self.profile_id = canonical_fingerprint(
            {
                "kind": "hep-column-profile",
                "format": format_name_,
                "fields": [list(item) for item in fields],
                "units": [list(item) for item in units],
            }
        )

    @property
    def fields(self) -> dict[str, str]:
        return dict(self.field_mapping)


class HEPImportResult(StrictModule, NonTrainableState):
    events: object
    report: AdapterReport
    source_record_count: int = eqx.field(static=True)
    admitted_record_count: int = eqx.field(static=True)
    overflow_count: int = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return bool(self.report.valid) and self.overflow_count == 0


class HEPExportResult(StrictModule, NonTrainableState):
    payload: str
    report: AdapterReport
    exported_record_count: int = eqx.field(static=True)


def import_event_columns(
    columns: Mapping[str, ArrayLike],
    prepared: PreparedParticleEvents,
    profile: HEPColumnProfile,
    weights: EventWeightSet,
    /,
    *,
    source_id: str,
) -> HEPImportResult:
    """Admit one exact fixed-capacity column profile without implicit reshaping."""
    if not isinstance(columns, Mapping):
        raise TypeError("columns must be a mapping.")
    if not isinstance(prepared, PreparedParticleEvents):
        raise TypeError("prepared must be PreparedParticleEvents.")
    if not isinstance(profile, HEPColumnProfile):
        raise TypeError("profile must be HEPColumnProfile.")
    mapping = profile.fields
    missing = tuple(field for field, source in mapping.items() if source not in columns)
    if missing:
        raise ValueError(f"Missing required source columns: {missing}.")
    kwargs = {field: columns[source] for field, source in mapping.items()}
    events = prepared.admit(weights=weights, source_id=source_id, **kwargs)
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        profile.format_name,
        "phydrax-particle-event",
        source_id=source_id,
        target_id=prepared.prepared_id,
        preserved_fields=tuple(sorted(mapping)),
        source_profile=AdapterFormatProfile(
            profile.format_name,
            qualifiers=dict(profile.unit_qualifiers),
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-particle-event",
            qualifiers={
                "momentum_unit": prepared.plan.momentum_unit.symbol,
                "length_unit": prepared.plan.length_unit.symbol,
                "time_unit": prepared.plan.time_unit.symbol,
            },
        ),
        stage="hep-column-import",
    )
    return HEPImportResult(
        events,
        report,
        prepared.plan.event_capacity,
        prepared.plan.event_capacity,
        0,
    )


__all__ = [
    "HEPColumnProfile",
    "HEPExportResult",
    "HEPImportResult",
    "HEPOptionalDependencyError",
    "import_event_columns",
]
