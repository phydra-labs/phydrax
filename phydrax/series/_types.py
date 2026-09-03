#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, NamedTuple, TypeAlias

from jaxtyping import Array


CoordinateKind: TypeAlias = Literal["continuous", "discrete"]
SeriesAlignment: TypeAlias = Literal["node", "edge"]
SeriesInterpolation: TypeAlias = Literal[
    "nearest",
    "previous",
    "linear",
    "cubic_hermite",
    "interval_hold",
]


class SeriesEvaluation(NamedTuple):
    """A reconstructed numerical PyTree and its query-support mask."""

    values: Any
    support: Array


class SeriesReconstructionCapabilities(NamedTuple):
    """Static numerical properties of one sampled-series reconstruction."""

    alignment: SeriesAlignment
    maximum_explicit_derivative_order: int
    causal: bool
    continuous: bool


__all__ = [
    "CoordinateKind",
    "SeriesAlignment",
    "SeriesEvaluation",
    "SeriesInterpolation",
    "SeriesReconstructionCapabilities",
]
