#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .._fingerprint import canonical_fingerprint


_TOKEN = re.compile(r"[a-z][a-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class ResolvedApplicationQuantity:
    """Canonical fields shared by domain-owned physical quantity specs."""

    name: str
    physical_dimension: str
    kernel_unit: str
    si_unit: str
    si_factor: Fraction
    axes: tuple[str, ...]
    sign_convention: str
    support_association: str
    reference_configuration: str
    quantity_id: str


def exact_positive_factor(value: Any, /) -> Fraction:
    """Return one positive finite factor without losing exact decimal scale."""
    if isinstance(value, bool):
        raise TypeError("si_factor must be a positive finite real number.")
    if isinstance(value, Fraction):
        factor = value
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("si_factor must be finite.")
        factor = Fraction(value)
    elif isinstance(value, numbers.Integral):
        factor = Fraction(int(value), 1)
    elif isinstance(value, numbers.Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("si_factor must be finite.")
        factor = Fraction(str(scalar))
    elif isinstance(value, str):
        factor = Fraction(value)
    else:
        raise TypeError("si_factor must be a positive finite real number.")
    if factor <= 0:
        raise ValueError("si_factor must be positive.")
    return factor


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
    supported_conversions: Mapping[tuple[str, str, str], Fraction],
    name: str,
    physical_dimension: str,
    kernel_unit: str,
    si_unit: str,
    si_factor: Any,
    axes: tuple[str, ...] = (),
    sign_convention: str = "",
    support_association: str = "",
    reference_configuration: str = "",
) -> ResolvedApplicationQuantity:
    """Resolve one domain quantity against its explicit supported unit routes."""
    domain_ = canonical_quantity_text(domain, "domain")
    name_ = canonical_quantity_text(name, "name")
    dimension = canonical_quantity_text(physical_dimension, "physical_dimension")
    if _TOKEN.fullmatch(name_) is None or _TOKEN.fullmatch(dimension) is None:
        raise ValueError("Quantity names and physical dimensions must be stable tokens.")
    kernel = canonical_quantity_text(kernel_unit, "kernel_unit")
    si = canonical_quantity_text(si_unit, "si_unit")
    factor = exact_positive_factor(si_factor)
    route = (dimension, kernel, si)
    if route not in supported_conversions:
        raise ValueError(
            f"Unsupported or ambiguous {domain_} kernel-to-SI unit route."
        )
    if factor != supported_conversions[route]:
        raise ValueError("si_factor does not match the declared exact unit route.")
    axes_ = _canonical_axes(axes)
    sign = canonical_quantity_text(
        sign_convention, "sign_convention", allow_empty=True
    )
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
            "physical_dimension": dimension,
            "kernel_unit": kernel,
            "si_unit": si,
            "si_factor": [factor.numerator, factor.denominator],
            "axes": list(axes_),
            "sign_convention": sign,
            "support_association": support,
            "reference_configuration": reference,
        }
    )
    return ResolvedApplicationQuantity(
        name_,
        dimension,
        kernel,
        si,
        factor,
        axes_,
        sign,
        support,
        reference,
        identity,
    )


__all__ = [
    "ResolvedApplicationQuantity",
    "canonical_quantity_text",
    "exact_positive_factor",
    "resolve_application_quantity",
]
