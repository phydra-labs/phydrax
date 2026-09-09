#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class MonetaryRounding(str, Enum):
    """Explicit floating-major-unit to exact-minor-atom rounding rule."""

    HALF_EVEN = "half_even"
    HALF_AWAY_FROM_ZERO = "half_away_from_zero"
    TOWARD_ZERO = "toward_zero"
    FLOOR = "floor"
    CEILING = "ceiling"


def _int64_scalar(value: Any, name: str, /) -> Array:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer scalar.")
    if isinstance(value, Integral):
        integer = int(value)
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise OverflowError(f"{name} must fit in signed int64.")
        array = jnp.asarray(integer, dtype=jnp.int64)
    else:
        raw = jnp.asarray(value)
        if raw.shape != () or not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError(f"{name} must be an integer scalar.")
        if raw.dtype != jnp.dtype(jnp.int64):
            raise TypeError(f"{name} array must have signed int64 dtype.")
        array = raw
    if array.dtype != jnp.dtype(jnp.int64):
        raise RuntimeError("Exact monetary atoms require JAX 64-bit mode.")
    return array


def _currency_tuple(values: Any, /) -> tuple[Currency, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("currencies must be a sequence of Currency values.")
    currencies = tuple(values)
    if not currencies or any(not isinstance(value, Currency) for value in currencies):
        raise TypeError("currencies must be a non-empty sequence of Currency values.")
    identifiers = tuple(value.currency_id for value in currencies)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("currencies must not contain duplicate currency identities.")
    return currencies


class Currency(StrictModule):
    """Currency identity and its number of decimal minor-unit places."""

    code: str = eqx.field(static=True)
    minor_unit: int = eqx.field(static=True)
    currency_id: str = eqx.field(static=True)

    def __init__(self, code: str, minor_unit: int, /):
        if not isinstance(code, str):
            raise TypeError("currency code must be a string.")
        if (
            len(code) != 3
            or not code.isascii()
            or not code.isalpha()
            or not code.isupper()
        ):
            raise ValueError(
                "currency code must be exactly three uppercase ASCII letters."
            )
        if isinstance(minor_unit, bool) or not isinstance(minor_unit, Integral):
            raise TypeError(
                "currency minor_unit must be an integer number of decimal places."
            )
        places = int(minor_unit)
        if places < 0 or places > 9:
            raise ValueError("currency minor_unit must be between zero and nine.")
        self.code = code
        self.minor_unit = places
        self.currency_id = canonical_fingerprint(
            {"kind": "currency", "code": code, "minor_unit": places}
        )

    @property
    def atoms_per_unit(self) -> int:
        return 10**self.minor_unit


class CurrencyAmount(StrictModule):
    """One exact signed-int64 quantity of a currency's minor atoms."""

    currency: Currency = eqx.field(static=True)
    atoms: Array

    def __init__(self, currency: Currency, atoms: Any, /):
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        self.currency = currency
        self.atoms = _int64_scalar(atoms, "currency atoms")


class MonetaryArray(StrictModule):
    """Floating major-unit values with explicit currency and validity maps.

    Invalid slots use the unique neutral representation value=0 and currency_index=0.
    Exact settlement amounts arise only at :func:`round_to_minor_atoms`.
    """

    values: Array
    currencies: tuple[Currency, ...] = eqx.field(static=True)
    currency_index: Array
    valid: Array

    def __init__(
        self,
        values: ArrayLike,
        currencies: tuple[Currency, ...],
        currency_index: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        values_ = jnp.asarray(values)
        if not jnp.issubdtype(values_.dtype, jnp.floating):
            raise TypeError("monetary values must have a floating dtype.")
        if values_.ndim < 1 or any(int(size) <= 0 for size in values_.shape):
            raise ValueError("monetary values must have a non-empty fixed shape.")
        currencies_ = _currency_tuple(currencies)
        raw_index = jnp.asarray(currency_index)
        if raw_index.shape != values_.shape or not jnp.issubdtype(
            raw_index.dtype, jnp.integer
        ):
            raise TypeError(
                "currency_index must be an integer array matching values shape."
            )
        mask = jnp.asarray(valid)
        if mask.shape != values_.shape or mask.dtype != jnp.dtype(bool):
            raise TypeError("valid must be a boolean array matching values shape.")
        values_ = eqx.error_if(
            values_,
            jnp.any(mask & ~jnp.isfinite(values_)),
            "active monetary values must be finite.",
        )
        raw_index = eqx.error_if(
            raw_index,
            jnp.any(mask & ((raw_index < 0) | (raw_index >= len(currencies_)))),
            "active currency indices are out of bounds.",
        )
        values_ = eqx.error_if(
            values_,
            jnp.any(~mask & (values_ != 0)),
            "inactive monetary values must use neutral zero padding.",
        )
        raw_index = eqx.error_if(
            raw_index,
            jnp.any(~mask & (raw_index != 0)),
            "inactive currency indices must use neutral zero padding.",
        )
        index = raw_index.astype(jnp.int32)
        self.values = values_
        self.currencies = currencies_
        self.currency_index = index
        self.valid = mask


def _same_currency(left: CurrencyAmount, right: CurrencyAmount, /) -> None:
    if not isinstance(left, CurrencyAmount) or not isinstance(right, CurrencyAmount):
        raise TypeError("currency arithmetic requires CurrencyAmount values.")
    if left.currency.currency_id != right.currency.currency_id:
        raise ValueError("currency arithmetic requires exactly matching currencies.")


def add_currency_amounts(
    left: CurrencyAmount, right: CurrencyAmount, /
) -> CurrencyAmount:
    """Add exact same-currency atoms and fail on signed-int64 overflow."""
    _same_currency(left, right)
    result = left.atoms + right.atoms
    overflow = ((right.atoms > 0) & (result < left.atoms)) | (
        (right.atoms < 0) & (result > left.atoms)
    )
    result = eqx.error_if(result, overflow, "currency addition overflowed signed int64.")
    return CurrencyAmount(left.currency, result)


def subtract_currency_amounts(
    left: CurrencyAmount, right: CurrencyAmount, /
) -> CurrencyAmount:
    """Subtract exact same-currency atoms and fail on signed-int64 overflow."""
    _same_currency(left, right)
    result = left.atoms - right.atoms
    overflow = ((right.atoms < 0) & (result < left.atoms)) | (
        (right.atoms > 0) & (result > left.atoms)
    )
    result = eqx.error_if(
        result, overflow, "currency subtraction overflowed signed int64."
    )
    return CurrencyAmount(left.currency, result)


def _round(values: Array, rule: MonetaryRounding, /) -> Array:
    if rule is MonetaryRounding.HALF_EVEN:
        return jnp.round(values)
    if rule is MonetaryRounding.HALF_AWAY_FROM_ZERO:
        return jnp.sign(values) * jnp.floor(jnp.abs(values) + 0.5)
    if rule is MonetaryRounding.TOWARD_ZERO:
        return jnp.trunc(values)
    if rule is MonetaryRounding.FLOOR:
        return jnp.floor(values)
    return jnp.ceil(values)


def round_to_minor_atoms(
    monetary: MonetaryArray,
    rounding: MonetaryRounding,
    /,
) -> Array:
    """Cross the explicit floating-to-exact boundary into signed minor atoms."""
    if not isinstance(monetary, MonetaryArray):
        raise TypeError("monetary must be a MonetaryArray.")
    if not isinstance(rounding, MonetaryRounding):
        raise TypeError("rounding must be a MonetaryRounding.")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("Exact monetary atoms require JAX 64-bit mode.")
    scales = jnp.asarray(
        tuple(currency.atoms_per_unit for currency in monetary.currencies),
        dtype=monetary.values.dtype,
    )
    scaled = monetary.values * scales[monetary.currency_index]
    rounded = _round(scaled, rounding)
    lower = jnp.asarray(_INT64_MIN, dtype=monetary.values.dtype)
    upper_exclusive = jnp.asarray(2**63, dtype=monetary.values.dtype)
    rounded = eqx.error_if(
        rounded,
        jnp.any(
            monetary.valid
            & (~jnp.isfinite(rounded) | (rounded < lower) | (rounded >= upper_exclusive))
        ),
        "rounded monetary values do not fit in signed int64.",
    )
    return jnp.where(monetary.valid, rounded, 0).astype(jnp.int64)


class FXPair(StrictModule):
    """Ordered base/quote currency identity; it deliberately carries no rate."""

    base: Currency = eqx.field(static=True)
    quote: Currency = eqx.field(static=True)
    pair_id: str = eqx.field(static=True)

    def __init__(self, base: Currency, quote: Currency, /):
        if not isinstance(base, Currency) or not isinstance(quote, Currency):
            raise TypeError("FX pair endpoints must be Currency values.")
        if base.currency_id == quote.currency_id:
            raise ValueError("FX pair base and quote currencies must be distinct.")
        self.base = base
        self.quote = quote
        self.pair_id = canonical_fingerprint(
            {
                "kind": "fx_pair",
                "base_currency_id": base.currency_id,
                "quote_currency_id": quote.currency_id,
            }
        )


__all__ = [
    "add_currency_amounts",
    "Currency",
    "CurrencyAmount",
    "FXPair",
    "MonetaryArray",
    "MonetaryRounding",
    "round_to_minor_atoms",
    "subtract_currency_amounts",
]
