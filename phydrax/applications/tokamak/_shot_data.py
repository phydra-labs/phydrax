#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Machine- and shot-aware governed tokamak measurement collections."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._fingerprint import canonical_fingerprint
from ...measurement import IndexSampleSupport, MeasurementAsset, SampleTimeAxis


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


@dataclass(frozen=True, slots=True)
class TokamakShotRecord:
    machine_id: str
    shot_id: str
    time_axis: SampleTimeAxis
    assets: tuple[MeasurementAsset, ...]
    equilibrium_ids: tuple[str, ...]
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        machine = _text(self.machine_id, "machine_id")
        shot = _text(self.shot_id, "shot_id")
        if not isinstance(self.time_axis, SampleTimeAxis):
            raise TypeError("time_axis must be SampleTimeAxis.")
        assets = tuple(self.assets)
        if not assets or any(not isinstance(value, MeasurementAsset) for value in assets):
            raise TypeError("assets must contain MeasurementAsset values.")
        if len({value.asset_id for value in assets}) != len(assets):
            raise ValueError("Tokamak shot asset IDs must be unique.")
        for asset in assets:
            if asset.acquisition is None:
                raise ValueError("Tokamak shot assets require acquisition identity.")
            if asset.acquisition.series_id != shot:
                raise ValueError("Tokamak shot asset series_id must equal shot_id.")
            support = asset.field.support
            if not isinstance(support, IndexSampleSupport):
                raise TypeError("Initial tokamak shot assets require IndexSampleSupport.")
            if (
                support.time_axis is None
                or support.time_axis.time_axis_id != self.time_axis.time_axis_id
            ):
                raise ValueError(
                    "Every tokamak shot asset must share the exact sample-time axis."
                )
        equilibria = tuple(
            _text(value, "equilibrium_id") for value in self.equilibrium_ids
        )
        if len(set(equilibria)) != len(equilibria):
            raise ValueError("equilibrium_ids must be unique.")
        object.__setattr__(self, "machine_id", machine)
        object.__setattr__(self, "shot_id", shot)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "equilibrium_ids", equilibria)
        object.__setattr__(
            self,
            "record_id",
            canonical_fingerprint(
                {
                    "kind": "tokamak-shot-record",
                    "machine": machine,
                    "shot": shot,
                    "time_axis": self.time_axis.time_axis_id,
                    "assets": [value.content_id for value in assets],
                    "equilibria": list(equilibria),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TokamakShotSplit:
    training_record_ids: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    locked_evaluation_record_ids: tuple[str, ...]
    split_id: str = field(init=False)

    def __post_init__(self) -> None:
        groups = tuple(
            tuple(_text(value, "record_id") for value in group)
            for group in (
                self.training_record_ids,
                self.validation_record_ids,
                self.locked_evaluation_record_ids,
            )
        )
        if not groups[0] or not groups[2]:
            raise ValueError(
                "Shot splits require training and locked evaluation records."
            )
        flattened = groups[0] + groups[1] + groups[2]
        if len(flattened) != len(set(flattened)):
            raise ValueError("A tokamak shot record cannot cross data-split roles.")
        object.__setattr__(self, "training_record_ids", groups[0])
        object.__setattr__(self, "validation_record_ids", groups[1])
        object.__setattr__(self, "locked_evaluation_record_ids", groups[2])
        object.__setattr__(
            self,
            "split_id",
            canonical_fingerprint(
                {
                    "kind": "tokamak-shot-split",
                    "training": list(groups[0]),
                    "validation": list(groups[1]),
                    "locked_evaluation": list(groups[2]),
                }
            ),
        )


__all__ = ["TokamakShotRecord", "TokamakShotSplit"]
