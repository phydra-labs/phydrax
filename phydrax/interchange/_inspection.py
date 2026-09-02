#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._report import AdapterReport


InspectionLocation = Literal[
    "cell", "face", "vertex", "particle", "marker", "point", "global"
]
InspectionStateKind = Literal["candidate", "accepted"]


def _identifier(value: str, owner: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{owner} must be a non-empty string.")
    return identifier


def _readonly_array(value: ArrayLike, /, *, dtype=None) -> NDArray:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("Host inspection arrays cannot use object dtype.")
    original_shape = array.shape
    contiguous = np.ascontiguousarray(array)
    readonly = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return readonly.reshape(original_shape)


@dataclass(frozen=True, slots=True)
class HostInspectionField:
    """One immutable host array with its exact native support semantics."""

    name: str
    values: NDArray
    valid: NDArray[np.bool_]
    location: InspectionLocation
    support_id: str
    layout_id: str
    representation: str
    unit_id: str | None = None
    component_labels: tuple[str, ...] = ()
    provenance_id: str = ""

    def __post_init__(self) -> None:
        name = _identifier(self.name, "Host inspection field name")
        if self.location not in (
            "cell",
            "face",
            "vertex",
            "particle",
            "marker",
            "point",
            "global",
        ):
            raise ValueError("Unknown host inspection field location.")
        support_id = _identifier(self.support_id, "Host inspection support_id")
        layout_id = _identifier(self.layout_id, "Host inspection layout_id")
        representation = _identifier(
            self.representation, "Host inspection representation"
        )
        provenance_id = _identifier(self.provenance_id, "Host inspection provenance_id")
        unit_id = (
            None
            if self.unit_id is None
            else _identifier(self.unit_id, "Host inspection unit_id")
        )
        labels = tuple(
            _identifier(label, "Host inspection component label")
            for label in self.component_labels
        )
        if len(set(labels)) != len(labels):
            raise ValueError("Host inspection component labels must be unique.")

        values = _readonly_array(self.values)
        valid = _readonly_array(self.valid, dtype=bool)
        if labels and (values.ndim == 0 or values.shape[-1] != len(labels)):
            raise ValueError(
                "Host inspection component labels must match the final value axis."
            )
        support_shape = values.shape[:-1] if labels else values.shape
        if valid.shape not in ((), support_shape):
            raise ValueError(
                "Host inspection validity must be scalar or match the field support shape."
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "support_id", support_id)
        object.__setattr__(self, "layout_id", layout_id)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "component_labels", labels)
        object.__setattr__(self, "provenance_id", provenance_id)


@dataclass(frozen=True, slots=True)
class HostInspectionFrame:
    """Host-only inspection snapshot; deliberately not a PyTree or archive schema."""

    time: float
    step: int
    state_kind: InspectionStateKind
    successful: bool
    status: int
    fields: tuple[HostInspectionField, ...]
    producer_id: str
    result_id: str

    def __post_init__(self) -> None:
        time = float(self.time)
        if not np.isfinite(time):
            raise ValueError("Host inspection frame time must be finite.")
        step = index(self.step)
        if step < 0:
            raise ValueError("Host inspection frame step must be nonnegative.")
        if self.state_kind not in ("candidate", "accepted"):
            raise ValueError("Host inspection state_kind must be candidate or accepted.")
        successful = np.asarray(self.successful, dtype=bool)
        if successful.shape != ():
            raise ValueError("Host inspection successful must be scalar.")
        status = index(self.status)
        fields = tuple(self.fields)
        if not fields or not all(
            isinstance(field, HostInspectionField) for field in fields
        ):
            raise TypeError(
                "Host inspection frames require at least one HostInspectionField."
            )
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("Host inspection field names must be unique within a frame.")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "successful", bool(successful))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "producer_id",
            _identifier(self.producer_id, "Host inspection producer_id"),
        )
        object.__setattr__(
            self, "result_id", _identifier(self.result_id, "Host inspection result_id")
        )


@dataclass(frozen=True, slots=True)
class HostInspectionConversion:
    """One host inspection frame together with its auditable adapter report."""

    frame: HostInspectionFrame
    report: AdapterReport

    def __post_init__(self) -> None:
        if not isinstance(self.frame, HostInspectionFrame):
            raise TypeError("frame must be a HostInspectionFrame.")
        if not isinstance(self.report, AdapterReport):
            raise TypeError("report must be an AdapterReport.")
        if self.report.target_id != self.frame.result_id:
            raise ValueError(
                "Adapter report target_id must equal the inspection result_id."
            )


__all__ = [
    "HostInspectionConversion",
    "HostInspectionField",
    "HostInspectionFrame",
    "InspectionLocation",
    "InspectionStateKind",
]
