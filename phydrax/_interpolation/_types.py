#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
from jaxtyping import Array

from .._strict import StrictModule


BoundsMode: TypeAlias = Literal["clip", "error", "extrapolate", "fill"]
MaskMode: TypeAlias = Literal["reject", "renormalize", "strict"]
NearestTiePolicy: TypeAlias = Literal["lower", "round_even", "upper"]


class InterpolationResourcePolicy(StrictModule):
    """Static bounds for prepared interpolation routes and storage."""

    maximum_routes: int = eqx.field(static=True)
    maximum_index_bytes: int = eqx.field(static=True)
    maximum_weight_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_routes: int = 20_000_000,
        maximum_index_bytes: int = 1 << 30,
        maximum_weight_bytes: int = 1 << 30,
    ):
        values = (maximum_routes, maximum_index_bytes, maximum_weight_bytes)
        if any(value <= 0 for value in values):
            raise ValueError("Interpolation resource limits must be positive.")
        self.maximum_routes = maximum_routes
        self.maximum_index_bytes = maximum_index_bytes
        self.maximum_weight_bytes = maximum_weight_bytes


class InterpolationCapabilities(NamedTuple):
    """Static numerical properties of one interpolation family."""

    partition_of_unity: bool
    nonnegative_value_weights: bool
    local_support: bool
    mask_renormalizable: bool
    tensor_product_composable: bool
    maximum_explicit_derivative_order: int | None


class InterpolationResult(NamedTuple):
    """An interpolated value array and its query support mask."""

    values: Array
    support: Array


__all__ = [
    "BoundsMode",
    "InterpolationCapabilities",
    "InterpolationResourcePolicy",
    "InterpolationResult",
    "MaskMode",
    "NearestTiePolicy",
]
