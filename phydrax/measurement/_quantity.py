#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical quantity and component-layout contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

from .._fingerprint import canonical_fingerprint
from ..units import conversion_factor, UnitDefinition


_TOKEN = re.compile(r"[a-z][a-z0-9_.-]*\Z")


def canonical_quantity_text(
    value: str,
    role: str,
    /,
    *,
    allow_empty: bool = False,
) -> str:
    """Validate canonical identity-bearing quantity text."""
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string.")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{role} must be canonical text without surrounding whitespace.")
    if not value and not allow_empty:
        raise ValueError(f"{role} must be non-empty.")
    return value


def _token(value: str, role: str, /) -> str:
    result = canonical_quantity_text(value, role)
    if _TOKEN.fullmatch(result) is None:
        raise ValueError(f"{role} must be a stable lower-case token.")
    return result


def _axes(axes: Sequence[str], /) -> tuple[str, ...]:
    if isinstance(axes, str):
        raise TypeError("axes must be a sequence of axis labels, not one string.")
    labels = tuple(canonical_quantity_text(axis, "axis") for axis in axes)
    if len(labels) != len(set(labels)):
        raise ValueError("Quantity axis labels must be unique.")
    return labels


class ValueKind(StrEnum):
    REAL_SCALAR = "real_scalar"
    COMPLEX_SCALAR = "complex_scalar"
    VECTOR = "vector"
    COVECTOR = "covector"
    SYMMETRIC_TENSOR = "symmetric_tensor"
    GENERAL_TENSOR = "general_tensor"
    CATEGORICAL = "categorical"
    PROBABILITY = "probability"
    COUNT = "count"


@dataclass(frozen=True, slots=True)
class ValueLayout:
    """Algebraic component shape independent of physical quantity meaning."""

    kind: ValueKind
    component_shape: tuple[int, ...] = ()
    component_labels: tuple[str, ...] = ()
    component_frame_id: str | None = None
    layout_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ValueKind):
            raise TypeError("kind must be ValueKind.")
        if isinstance(self.component_shape, Integral):
            raise TypeError("component_shape must be a sequence of positive integers.")
        shape = tuple(self.component_shape)
        if any(
            isinstance(size, bool) or not isinstance(size, Integral) for size in shape
        ):
            raise TypeError("component_shape entries must be integers.")
        shape = tuple(int(size) for size in shape)
        if any(size < 1 for size in shape):
            raise ValueError("component_shape entries must be positive.")
        labels = _axes(self.component_labels)
        scalar_kinds = {
            ValueKind.REAL_SCALAR,
            ValueKind.COMPLEX_SCALAR,
            ValueKind.CATEGORICAL,
            ValueKind.COUNT,
        }
        if self.kind in scalar_kinds and shape:
            raise ValueError(
                f"{self.kind.value} values require scalar component shape ()."
            )
        if self.kind in {ValueKind.VECTOR, ValueKind.COVECTOR, ValueKind.PROBABILITY}:
            if len(shape) != 1:
                raise ValueError(f"{self.kind.value} values require rank-one components.")
        if self.kind is ValueKind.SYMMETRIC_TENSOR:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(
                    "symmetric_tensor values require square rank-two components."
                )
        if self.kind is ValueKind.GENERAL_TENSOR and len(shape) < 2:
            raise ValueError(
                "general_tensor values require rank-two-or-higher components."
            )
        if labels and (len(shape) != 1 or len(labels) != shape[0]):
            raise ValueError("component_labels require one label per rank-one component.")
        frame = (
            None
            if self.component_frame_id is None
            else canonical_quantity_text(self.component_frame_id, "component_frame_id")
        )
        if (
            self.kind
            in {
                ValueKind.VECTOR,
                ValueKind.COVECTOR,
                ValueKind.SYMMETRIC_TENSOR,
                ValueKind.GENERAL_TENSOR,
            }
            and frame is None
        ):
            raise ValueError(f"{self.kind.value} values require component_frame_id.")
        if self.kind is ValueKind.PROBABILITY and frame is not None:
            raise ValueError("probability components cannot carry a spatial frame.")
        object.__setattr__(self, "component_shape", shape)
        object.__setattr__(self, "component_labels", labels)
        object.__setattr__(self, "component_frame_id", frame)
        object.__setattr__(
            self,
            "layout_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-value-layout",
                    "value_kind": self.kind.value,
                    "component_shape": list(shape),
                    "component_labels": list(labels),
                    "component_frame": frame,
                }
            ),
        )

    @classmethod
    def scalar(cls) -> ValueLayout:
        return cls(ValueKind.REAL_SCALAR)


@dataclass(frozen=True, slots=True)
class QuantitySpec:
    """Canonical physical meaning and compatibility identity for one quantity."""

    namespace: str
    name: str
    quantity_kind: str
    unit: UnitDefinition
    compatibility_key: str
    axes: tuple[str, ...] = ()
    sign_convention: str = ""
    support_association: str = ""
    reference_configuration: str = ""
    quantity_id: str = field(init=False)
    compatibility_id: str = field(init=False)

    def __post_init__(self) -> None:
        namespace = _token(self.namespace, "namespace")
        name = _token(self.name, "name")
        kind = _token(self.quantity_kind, "quantity_kind")
        key = _token(self.compatibility_key, "compatibility_key")
        if not isinstance(self.unit, UnitDefinition):
            raise TypeError("unit must be a UnitDefinition.")
        axes = _axes(self.axes)
        sign = canonical_quantity_text(
            self.sign_convention, "sign_convention", allow_empty=True
        )
        support = canonical_quantity_text(
            self.support_association, "support_association", allow_empty=True
        )
        reference = canonical_quantity_text(
            self.reference_configuration,
            "reference_configuration",
            allow_empty=True,
        )
        compatibility = canonical_fingerprint(
            {
                "kind": "quantity-compatibility",
                "key": key,
                "dimension": self.unit.dimension.dimension_id,
                "reference_system": self.unit.reference_system_id,
                "axes": list(axes),
                "sign_convention": sign,
                "support_association": support,
                "reference_configuration": reference,
            }
        )
        identity = canonical_fingerprint(
            {
                "kind": "quantity-spec",
                "namespace": namespace,
                "name": name,
                "quantity_kind": kind,
                "unit": self.unit.unit_id,
                "compatibility_key": key,
                "compatibility": compatibility,
            }
        )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "quantity_kind", kind)
        object.__setattr__(self, "compatibility_key", key)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "sign_convention", sign)
        object.__setattr__(self, "support_association", support)
        object.__setattr__(self, "reference_configuration", reference)
        object.__setattr__(self, "quantity_id", identity)
        object.__setattr__(self, "compatibility_id", compatibility)

    def compatible_with(self, other: QuantitySpec, /) -> bool:
        return (
            isinstance(other, QuantitySpec)
            and self.compatibility_id == other.compatibility_id
        )


def resolve_quantity(
    *,
    domain: str,
    reference_units: Mapping[str, UnitDefinition],
    name: str,
    quantity_kind: str,
    unit: UnitDefinition,
    axes: tuple[str, ...] = (),
    sign_convention: str = "",
    support_association: str = "",
    reference_configuration: str = "",
    compatibility_key: str | None = None,
) -> QuantitySpec:
    """Resolve one domain quantity against its canonical reference unit."""
    domain_ = _token(domain, "domain")
    kind = _token(quantity_kind, "quantity_kind")
    if kind not in reference_units:
        raise ValueError(f"Unsupported {domain_} quantity kind {kind!r}.")
    conversion_factor(unit, reference_units[kind])
    return QuantitySpec(
        domain_,
        name,
        kind,
        unit,
        f"{domain_}.{kind}" if compatibility_key is None else compatibility_key,
        axes,
        sign_convention,
        support_association,
        reference_configuration,
    )


__all__ = [
    "QuantitySpec",
    "ValueKind",
    "ValueLayout",
    "canonical_quantity_text",
    "resolve_quantity",
]
