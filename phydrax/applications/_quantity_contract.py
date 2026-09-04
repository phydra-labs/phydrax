#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .._fingerprint import canonical_fingerprint
from ..units import conversion_factor, UnitDefinition


_TOKEN = re.compile(r"[a-z][a-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class ResolvedApplicationQuantity:
    """Canonical fields shared by domain-owned physical quantity specs."""

    name: str
    quantity_kind: str
    unit: UnitDefinition
    axes: tuple[str, ...]
    sign_convention: str
    support_association: str
    reference_configuration: str
    quantity_id: str


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


def _canonical_axes(axes: tuple[str, ...], /) -> tuple[str, ...]:
    if isinstance(axes, str):
        raise TypeError("axes must be a sequence of axis labels, not one string.")
    labels = tuple(canonical_quantity_text(axis, "axis") for axis in axes)
    if len(labels) != len(set(labels)):
        raise ValueError("Quantity axis labels must be unique.")
    return labels


def resolve_application_quantity(
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
) -> ResolvedApplicationQuantity:
    """Resolve one domain quantity against its canonical reference unit."""
    domain_ = canonical_quantity_text(domain, "domain")
    name_ = canonical_quantity_text(name, "name")
    kind = canonical_quantity_text(quantity_kind, "quantity_kind")
    if _TOKEN.fullmatch(name_) is None or _TOKEN.fullmatch(kind) is None:
        raise ValueError("Quantity names and kinds must be stable tokens.")
    if not isinstance(unit, UnitDefinition):
        raise TypeError("unit must be a UnitDefinition.")
    if kind not in reference_units:
        raise ValueError(f"Unsupported {domain_} quantity kind {kind!r}.")
    conversion_factor(unit, reference_units[kind])
    axes_ = _canonical_axes(axes)
    sign = canonical_quantity_text(sign_convention, "sign_convention", allow_empty=True)
    support = canonical_quantity_text(
        support_association, "support_association", allow_empty=True
    )
    reference = canonical_quantity_text(
        reference_configuration, "reference_configuration", allow_empty=True
    )
    identity = canonical_fingerprint(
        {
            "kind": f"{domain_}-quantity-spec",
            "name": name_,
            "quantity_kind": kind,
            "unit_id": unit.unit_id,
            "axes": list(axes_),
            "sign_convention": sign,
            "support_association": support,
            "reference_configuration": reference,
        }
    )
    return ResolvedApplicationQuantity(
        name_,
        kind,
        unit,
        axes_,
        sign,
        support,
        reference,
        identity,
    )


__all__ = [
    "ResolvedApplicationQuantity",
    "canonical_quantity_text",
    "resolve_application_quantity",
]
