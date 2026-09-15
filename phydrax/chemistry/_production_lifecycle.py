#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed lifecycle archives for production chemistry result payloads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle import (
    create,
    LifecycleArchive,
    open as open_lifecycle,
    payload_digest,
    ResultManifest,
)
from ..units import UnitDefinition


_KIND_KEY = "production_chemistry_kind"
_PLAN_KEY = "production_chemistry_plan"


class ProductionChemistryArchivePlan(StrictModule, NonTrainableState):
    """Exact field/unit layout for one production chemistry result family."""

    result_kind: str = eqx.field(static=True)
    field_units: tuple[tuple[str, str], ...] = eqx.field(static=True)
    scientific_plan_id: str = eqx.field(static=True)
    archive_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        result_kind: str,
        field_units: Mapping[str, UnitDefinition | str],
        scientific_plan_id: str,
        /,
    ):
        kind = str(result_kind).strip()
        scientific_plan = str(scientific_plan_id).strip()
        if not kind or not scientific_plan:
            raise ValueError("Result kind and scientific plan ID must be non-empty.")
        units = []
        for field, unit in field_units.items():
            name = str(field).strip()
            unit_id = (
                unit.unit_id if isinstance(unit, UnitDefinition) else str(unit).strip()
            )
            if not name or not unit_id:
                raise ValueError("Production archive fields and units must be non-empty.")
            units.append((name, unit_id))
        normalized = tuple(sorted(units))
        if not normalized or len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("Production archive fields must be non-empty and unique.")
        self.result_kind = kind
        self.field_units = normalized
        self.scientific_plan_id = scientific_plan
        self.archive_plan_id = canonical_fingerprint(
            {
                "kind": "production-chemistry-archive-plan",
                "result_kind": kind,
                "field_units": [list(value) for value in normalized],
                "scientific_plan": scientific_plan,
            }
        )

    def write(
        self,
        path: str | Path,
        result_id: str,
        run_id: str,
        arrays: Mapping[str, object],
        /,
        *,
        evidence_ids: tuple[str, ...] = (),
        diagnostic_ids: tuple[str, ...] = (),
    ) -> LifecycleArchive:
        result = str(result_id).strip()
        run = str(run_id).strip()
        if not result or not run:
            raise ValueError("Production result and run IDs must be non-empty.")
        expected = {name for name, _ in self.field_units}
        if set(arrays) != expected:
            raise ValueError("Production archive payload fields do not match the plan.")
        payloads = {name: np.asarray(value) for name, value in arrays.items()}
        if any(value.dtype == object for value in payloads.values()):
            raise TypeError(
                "Production chemistry archive arrays cannot use object dtype."
            )
        manifest = ResultManifest(
            result,
            run,
            dict(self.field_units),
            {name: payload_digest(value) for name, value in payloads.items()},
            evidence_ids=evidence_ids,
            diagnostic_ids=diagnostic_ids,
            sampled_semantics={
                _KIND_KEY: self.result_kind,
                _PLAN_KEY: self.scientific_plan_id,
            },
        )
        return create(path, manifest=manifest, arrays=payloads)

    def open(
        self,
        path: str | Path,
        /,
        *,
        expected_result_id: str | None = None,
    ) -> LifecycleArchive:
        archive = open_lifecycle(path)
        if not isinstance(archive.manifest, ResultManifest):
            raise TypeError("Production chemistry archives require ResultManifest.")
        semantics = dict(archive.manifest.sampled_semantics)
        if semantics.get(_KIND_KEY) != self.result_kind or (
            semantics.get(_PLAN_KEY) != self.scientific_plan_id
        ):
            raise ValueError(
                "Production chemistry archive semantics differ from the plan."
            )
        if expected_result_id is not None and archive.manifest.result_id != str(
            expected_result_id
        ):
            raise ValueError("Production chemistry archive result identity differs.")
        observed_fields = tuple(
            sorted((name, unit) for name, _, unit in archive.manifest.fields)
        )
        if observed_fields != self.field_units:
            raise ValueError("Production chemistry archive field/unit layout differs.")
        return archive


__all__ = ["ProductionChemistryArchivePlan"]
