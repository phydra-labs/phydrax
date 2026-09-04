from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from numbers import Integral
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.units._dimension import DimensionSignature, Exponent


Scale: TypeAlias = int | Fraction | str | tuple[int, int]
UnitComponent: TypeAlias = tuple["UnitDefinition", Exponent]


def _scale(value: Scale) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("unit scales must be exact positive rational values")
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("unit scale tuples must contain numerator and denominator")
        numerator, denominator = value
        if (
            isinstance(numerator, bool)
            or isinstance(denominator, bool)
            or not isinstance(numerator, Integral)
            or not isinstance(denominator, Integral)
        ):
            raise TypeError("unit scale tuple entries must be integers")
        result = Fraction(int(numerator), int(denominator))
    elif isinstance(value, (Integral, Fraction, str)):
        result = Fraction(value)
    else:
        raise TypeError("unit scales must be exact positive rational values")
    if result <= 0:
        raise ValueError("unit scales must be positive")
    return result


def _exact_nth_root(value: int, degree: int) -> int:
    if degree <= 0:
        raise ValueError("root degree must be positive")
    if value < 0:
        raise ValueError("exact roots require non-negative integers")
    if value in (0, 1) or degree == 1:
        return value
    low, high = 1, 1
    while high**degree < value:
        high *= 2
    while low <= high:
        middle = (low + high) // 2
        powered = middle**degree
        if powered == value:
            return middle
        if powered < value:
            low = middle + 1
        else:
            high = middle - 1
    raise ValueError("unit scale does not have an exact rational root")


def _scale_power(scale: Fraction, exponent: Fraction) -> Fraction:
    numerator_root = _exact_nth_root(scale.numerator, exponent.denominator)
    denominator_root = _exact_nth_root(scale.denominator, exponent.denominator)
    rooted = Fraction(numerator_root, denominator_root)
    return rooted**exponent.numerator


class UnitDefinition(StrictModule, NonTrainableState):
    """Immutable multiplicative unit connected to an explicit reference system."""

    symbol: str = eqx.field(static=True)
    dimension: DimensionSignature
    reference_system_id: str = eqx.field(static=True)
    scale_numerator: int = eqx.field(static=True)
    scale_denominator: int = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        symbol: str,
        dimension: DimensionSignature,
        reference_system_id: str,
        scale_to_reference: Scale = 1,
    ) -> None:
        if not isinstance(symbol, str) or not symbol or symbol.strip() != symbol:
            raise ValueError("unit symbols must be non-empty stripped strings")
        if not isinstance(dimension, DimensionSignature):
            raise TypeError("unit dimensions must be DimensionSignature values")
        if (
            not isinstance(reference_system_id, str)
            or not reference_system_id
            or reference_system_id.strip() != reference_system_id
        ):
            raise ValueError("reference_system_id must be a non-empty stripped string")
        scale = _scale(scale_to_reference)
        self.symbol = symbol
        self.dimension = dimension
        self.reference_system_id = reference_system_id
        self.scale_numerator = scale.numerator
        self.scale_denominator = scale.denominator
        self.unit_id = canonical_fingerprint(
            {
                "kind": "unit_definition",
                "symbol": symbol,
                "dimension_id": dimension.dimension_id,
                "reference_system_id": reference_system_id,
                "scale": [scale.numerator, scale.denominator],
            }
        )

    @property
    def scale_to_reference(self) -> Fraction:
        return Fraction(self.scale_numerator, self.scale_denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "dimension": self.dimension.to_dict(),
            "reference_system_id": self.reference_system_id,
            "scale_numerator": self.scale_numerator,
            "scale_denominator": self.scale_denominator,
            "unit_id": self.unit_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnitDefinition:
        if not isinstance(payload, Mapping):
            raise TypeError("unit payload must be a mapping")
        expected = {
            "symbol",
            "dimension",
            "reference_system_id",
            "scale_numerator",
            "scale_denominator",
            "unit_id",
        }
        if set(payload) != expected:
            raise ValueError("unit payload must use the canonical fields")
        symbol = payload.get("symbol")
        dimension_payload = payload.get("dimension")
        reference_system_id = payload.get("reference_system_id")
        numerator = payload.get("scale_numerator")
        denominator = payload.get("scale_denominator")
        if not isinstance(symbol, str):
            raise TypeError("unit payload symbol must be a string")
        if not isinstance(dimension_payload, Mapping):
            raise TypeError("unit payload dimension must be a mapping")
        if not isinstance(reference_system_id, str):
            raise TypeError("unit payload reference_system_id must be a string")
        if isinstance(numerator, bool) or not isinstance(numerator, Integral):
            raise TypeError("unit payload scale numerator must be an integer")
        if isinstance(denominator, bool) or not isinstance(denominator, Integral):
            raise TypeError("unit payload scale denominator must be an integer")
        unit = cls(
            symbol,
            DimensionSignature.from_dict(dimension_payload),
            reference_system_id,
            (int(numerator), int(denominator)),
        )
        claimed_id = payload.get("unit_id")
        if not isinstance(claimed_id, str):
            raise TypeError("unit payload unit_id must be a string")
        if claimed_id != unit.unit_id:
            raise ValueError("unit payload fingerprint does not match its content")
        return unit

    def __hash__(self) -> int:
        return hash(self.unit_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnitDefinition) and self.unit_id == other.unit_id


def conversion_factor(source: UnitDefinition, target: UnitDefinition) -> Fraction:
    """Return the exact multiplicative factor from ``source`` to ``target``."""
    if not isinstance(source, UnitDefinition) or not isinstance(target, UnitDefinition):
        raise TypeError("unit conversion requires UnitDefinition values")
    if source.dimension != target.dimension:
        raise ValueError("unit conversion requires exactly matching dimensions")
    if source.reference_system_id != target.reference_system_id:
        raise ValueError("unit conversion requires a shared reference system")
    return source.scale_to_reference / target.scale_to_reference


def convert_value(value: Any, *, source: UnitDefinition, target: UnitDefinition):
    """Convert a scalar or array through an exact, statically resolved factor."""
    factor = conversion_factor(source, target)
    array = jnp.asarray(value)
    if factor == 1:
        return array
    if jnp.issubdtype(array.dtype, jnp.inexact):
        scalar = jnp.asarray(float(factor), dtype=array.dtype)
    else:
        dtype = jnp.result_type(array, jnp.asarray(1.0))
        array = array.astype(dtype)
        scalar = jnp.asarray(float(factor), dtype=dtype)
    return array * scalar


def derived_unit(
    symbol: str,
    components: Iterable[UnitComponent],
) -> UnitDefinition:
    """Construct an exactly scaled product of powers of compatible units."""
    entries = tuple(components)
    if not entries:
        raise ValueError("derived units require at least one component")
    first_unit = entries[0][0]
    if not isinstance(first_unit, UnitDefinition):
        raise TypeError("derived unit components must contain UnitDefinition values")
    reference_system_id = first_unit.reference_system_id
    dimension = DimensionSignature()
    scale = Fraction(1)
    for unit, raw_exponent in entries:
        if not isinstance(unit, UnitDefinition):
            raise TypeError("derived unit components must contain UnitDefinition values")
        if unit.reference_system_id != reference_system_id:
            raise ValueError("derived unit components must share a reference system")
        if isinstance(raw_exponent, bool) or not isinstance(
            raw_exponent, (Integral, Fraction)
        ):
            raise TypeError("derived unit exponents must be integers or fractions")
        exponent = Fraction(raw_exponent)
        dimension = dimension * unit.dimension.power(exponent)
        scale *= _scale_power(unit.scale_to_reference, exponent)
    return UnitDefinition(symbol, dimension, reference_system_id, scale)
