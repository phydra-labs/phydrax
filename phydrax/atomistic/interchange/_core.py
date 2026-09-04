#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import AMOUNT, ENERGY, UnitDefinition
from .._force_field import AtomisticForceFieldPlan
from .._units import AtomisticUnitSystem


class UnsupportedAtomisticContentError(ValueError):
    pass


class AtomisticInterchangeReport(StrictModule, NonTrainableState):
    source_kind: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    source_energy_unit: UnitDefinition | None
    avogadro_constant_set_id: str | None = eqx.field(static=True)
    supported_terms: tuple[str, ...] = eqx.field(static=True)
    unsupported_terms: tuple[str, ...] = eqx.field(static=True)
    warnings: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_kind: str,
        units: AtomisticUnitSystem,
        supported_terms=(),
        unsupported_terms=(),
        warnings=(),
        /,
        *,
        source_energy_unit: UnitDefinition | None = None,
        avogadro_constant_set_id: str | None = None,
    ):
        source = str(source_kind).strip()
        supported = tuple(str(value) for value in supported_terms)
        unsupported = tuple(str(value) for value in unsupported_terms)
        warnings_ = tuple(str(value) for value in warnings)
        if not source:
            raise ValueError("source_kind must be non-empty.")
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("Interchange report units must be an AtomisticUnitSystem.")
        if (source_energy_unit is None) != (avogadro_constant_set_id is None):
            raise ValueError(
                "Molar source unit and Avogadro provenance must be provided together."
            )
        if source_energy_unit is not None:
            if (
                not isinstance(source_energy_unit, UnitDefinition)
                or source_energy_unit.dimension != ENERGY / AMOUNT
                or source_energy_unit.reference_system_id
                != units.scale.energy_unit.reference_system_id
            ):
                raise ValueError(
                    "Interchange source energy must have ENERGY / AMOUNT in the "
                    "unit system's reference system."
                )
            if avogadro_constant_set_id != units.constant_set_id:
                raise ValueError(
                    "Interchange Avogadro provenance must match the unit system."
                )
        self.source_kind = source
        self.units = units
        self.supported_terms = supported
        self.source_energy_unit = source_energy_unit
        self.avogadro_constant_set_id = avogadro_constant_set_id
        self.unsupported_terms = unsupported
        self.warnings = warnings_
        self.report_id = canonical_fingerprint(
            {
                "kind": "atomistic-interchange-report",
                "source": source,
                "unit_system": units.unit_system_id,
                "supported": list(supported),
                "unsupported": list(unsupported),
                "source_energy_unit": None
                if source_energy_unit is None
                else source_energy_unit.unit_id,
                "avogadro_constant_set": avogadro_constant_set_id,
                "warnings": list(warnings_),
            }
        )

    def require_complete(self) -> None:
        if self.unsupported_terms:
            raise UnsupportedAtomisticContentError(
                "Unsupported atomistic content: " + ", ".join(self.unsupported_terms)
            )


class AtomisticInterchangeBundle(StrictModule):
    force_field: AtomisticForceFieldPlan
    report: AtomisticInterchangeReport
    source_id: str = eqx.field(static=True)

    def __init__(
        self, force_field: AtomisticForceFieldPlan, report: AtomisticInterchangeReport, /
    ):
        if not isinstance(force_field, AtomisticForceFieldPlan) or not isinstance(
            report, AtomisticInterchangeReport
        ):
            raise TypeError("Interchange bundle requires force-field plan and report.")
        report.require_complete()
        if report.units.unit_system_id != force_field.system.units.unit_system_id:
            raise ValueError(
                "Interchange report and force field have incompatible unit systems."
            )
        self.force_field = force_field
        self.report = report
        self.source_id = canonical_fingerprint(
            {
                "kind": "atomistic-interchange-bundle",
                "force_field": force_field.plan_id,
                "report": report.report_id,
            }
        )


def require_mapping_fields(value: Mapping, fields: tuple[str, ...], /) -> None:
    missing = tuple(field for field in fields if field not in value)
    if missing:
        raise ValueError("Missing interchange fields: " + ", ".join(missing))


def canonical_source_digest(value: Mapping, /) -> str:
    def normalize(content):
        if isinstance(content, np.ndarray):
            return {
                "shape": list(content.shape),
                "dtype": content.dtype.name,
                "data": content.tolist(),
            }
        if isinstance(content, Mapping):
            return {str(key): normalize(item) for key, item in content.items()}
        if isinstance(content, (list, tuple)):
            return [normalize(item) for item in content]
        if isinstance(content, (str, int, float, bool, type(None))):
            return content
        return repr(content)

    return canonical_fingerprint(
        {"kind": "atomistic-interchange-source", "content": normalize(value)}
    )


__all__ = [
    "AtomisticInterchangeBundle",
    "AtomisticInterchangeReport",
    "UnsupportedAtomisticContentError",
    "canonical_source_digest",
    "require_mapping_fields",
]
