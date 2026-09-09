#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from collections.abc import Sequence

import equinox as eqx

from phydrax._strict import StrictModule

from ._currency import Currency


_SCHEME_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_VALUE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*\Z")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*\Z")


def _canonical_text(value: str, name: str, /, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _token(value: str, name: str, /) -> str:
    token = _canonical_text(value, name)
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(f"{name} must be a canonical identifier token.")
    return token


def _description(value: str, name: str, /) -> str:
    return _canonical_text(value, name, allow_empty=True)


class FinancialIdentifier(StrictModule):
    """A namespaced financial identifier without implied reference-data meaning."""

    scheme: str = eqx.field(static=True)
    value: str = eqx.field(static=True)
    canonical: str = eqx.field(static=True)

    def __init__(self, scheme: str, value: str, /):
        scheme_ = _canonical_text(scheme, "identifier scheme")
        value_ = _canonical_text(value, "identifier value")
        if _SCHEME_PATTERN.fullmatch(scheme_) is None:
            raise ValueError(
                "identifier scheme must start with a lowercase letter and contain "
                "only lowercase letters, digits, '.', '_' or '-'."
            )
        if _VALUE_PATTERN.fullmatch(value_) is None:
            raise ValueError("identifier value must be a canonical identifier token.")
        self.scheme = scheme_
        self.value = value_
        self.canonical = f"{scheme_}:{value_}"

    @property
    def identifier_id(self) -> str:
        return self.canonical


class AssetReference(StrictModule):
    """Static identity and denomination of one referenced financial asset."""

    identifier: FinancialIdentifier = eqx.field(static=True)
    asset_class: str = eqx.field(static=True)
    currency: Currency = eqx.field(static=True)
    description: str = eqx.field(static=True)

    def __init__(
        self,
        identifier: FinancialIdentifier,
        asset_class: str,
        currency: Currency,
        description: str,
        /,
    ):
        if not isinstance(identifier, FinancialIdentifier):
            raise TypeError("identifier must be a FinancialIdentifier.")
        if not isinstance(currency, Currency):
            raise TypeError("currency must be a Currency.")
        self.identifier = identifier
        self.asset_class = _token(asset_class, "asset_class")
        self.currency = currency
        self.description = _description(description, "asset description")

    @property
    def asset_id(self) -> str:
        return self.identifier.canonical


class InstrumentReference(StrictModule):
    """Static instrument identity with explicit underlyings and quote semantics."""

    identifier: FinancialIdentifier = eqx.field(static=True)
    underlying_ids: tuple[FinancialIdentifier, ...] = eqx.field(static=True)
    settlement_currency: Currency = eqx.field(static=True)
    quote_unit: str = eqx.field(static=True)
    description: str = eqx.field(static=True)

    def __init__(
        self,
        identifier: FinancialIdentifier,
        underlying_ids: Sequence[FinancialIdentifier],
        settlement_currency: Currency,
        quote_unit: str,
        description: str,
        /,
    ):
        if not isinstance(identifier, FinancialIdentifier):
            raise TypeError("identifier must be a FinancialIdentifier.")
        if isinstance(underlying_ids, (str, bytes)) or not isinstance(
            underlying_ids, Sequence
        ):
            raise TypeError(
                "underlying_ids must be a sequence of FinancialIdentifier values."
            )
        underlyings = tuple(underlying_ids)
        if any(not isinstance(item, FinancialIdentifier) for item in underlyings):
            raise TypeError("underlying_ids must contain FinancialIdentifier values.")
        canonical = tuple(item.canonical for item in underlyings)
        if len(set(canonical)) != len(canonical):
            raise ValueError("underlying_ids must be unique.")
        if not isinstance(settlement_currency, Currency):
            raise TypeError("settlement_currency must be a Currency.")
        self.identifier = identifier
        self.underlying_ids = underlyings
        self.settlement_currency = settlement_currency
        self.quote_unit = _token(quote_unit, "quote_unit")
        self.description = _description(description, "instrument description")

    @property
    def instrument_id(self) -> str:
        return self.identifier.canonical


__all__ = [
    "AssetReference",
    "FinancialIdentifier",
    "InstrumentReference",
]
