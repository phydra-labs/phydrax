#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag


class MarketStatus(IntFlag):
    """Composable fail-closed status bits for market-data preparation."""

    SUCCESS = 0
    DUPLICATE = 1
    STALE = 2
    MISSING_FACTOR = 4
    CAPACITY_EXCEEDED = 8
    FX_AMBIGUOUS = 16
    CROSSED_QUOTE = 32
    CAUSAL_TIME_VIOLATION = 64
    NONFINITE = 128
    OUTSIDE_WINDOW = 256
    CURRENCY_MISMATCH = 512
    INVALID_DOMAIN = 1024


_STATUS_MESSAGES = {
    MarketStatus.DUPLICATE: "duplicate observations require an explicit tie policy",
    MarketStatus.STALE: "the selected observation is older than the admissible age",
    MarketStatus.MISSING_FACTOR: "a required market factor is missing",
    MarketStatus.CAPACITY_EXCEEDED: "fixed market-data capacity was exceeded",
    MarketStatus.FX_AMBIGUOUS: "more than one equally short FX conversion path is available",
    MarketStatus.CROSSED_QUOTE: "the selected bid exceeds the selected ask",
    MarketStatus.CAUSAL_TIME_VIOLATION: "data availability is later than the decision time",
    MarketStatus.NONFINITE: "market data contains a non-finite value",
    MarketStatus.OUTSIDE_WINDOW: "no observation exists in the requested half-open window",
    MarketStatus.CURRENCY_MISMATCH: "the FX path endpoints do not match the requested currencies",
    MarketStatus.INVALID_DOMAIN: "the market-data request is outside its declared domain",
}


def market_status_message(status: int | MarketStatus, /) -> str:
    """Return a stable host-readable description of one or more status bits."""

    value = MarketStatus(int(status))
    if value == MarketStatus.SUCCESS:
        return "successful"
    known = MarketStatus.SUCCESS
    messages: list[str] = []
    for flag, message in _STATUS_MESSAGES.items():
        known |= flag
        if value & flag:
            messages.append(message)
    unknown = int(value) & ~int(known)
    if unknown:
        messages.append(f"unknown market status bits: {unknown}")
    return "; ".join(messages)


__all__ = ["MarketStatus", "market_status_message"]
