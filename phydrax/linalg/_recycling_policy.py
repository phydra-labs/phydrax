#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx

from .._strict import StrictModule


RecyclingExtraction: TypeAlias = Literal["harmonic-ritz"]
RecyclingRefresh: TypeAlias = Literal["reuse-source", "rebuild"]


class RecyclingPolicy(StrictModule):
    """Fixed-capacity GCRO-DR extraction and numerical-refresh policy."""

    capacity: int = eqx.field(static=True)
    extraction: RecyclingExtraction = eqx.field(static=True)
    refresh: RecyclingRefresh = eqx.field(static=True)

    def __init__(
        self,
        *,
        capacity: int = 20,
        extraction: RecyclingExtraction = "harmonic-ritz",
        refresh: RecyclingRefresh = "reuse-source",
    ):
        capacity_ = int(capacity)
        if capacity_ < 1:
            raise ValueError("Recycling capacity must be positive.")
        if extraction != "harmonic-ritz":
            raise ValueError("Only harmonic-ritz recycling extraction is supported.")
        if refresh not in ("reuse-source", "rebuild"):
            raise ValueError("Unknown recycling refresh policy.")
        self.capacity = capacity_
        self.extraction = extraction
        self.refresh = refresh


__all__ = ["RecyclingExtraction", "RecyclingPolicy", "RecyclingRefresh"]
