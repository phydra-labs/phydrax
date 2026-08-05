#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

from jaxtyping import Array


BoundsMode: TypeAlias = Literal["clip", "error", "extrapolate", "fill"]
MaskMode: TypeAlias = Literal["reject", "renormalize", "strict"]
NearestTiePolicy: TypeAlias = Literal["lower", "round_even", "upper"]


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
    "InterpolationResult",
    "MaskMode",
    "NearestTiePolicy",
]
