#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class KFACAffineBlock:
    """One affine parameter block exposed to a KFAC layout adapter."""

    name: str
    weight: Any
    bias: Any | None
    parameterization: Literal["direct", "low_rank_update", "rwf", "transformed"] = (
        "direct"
    )
    block_kind: Literal["dense-affine", "convolution", "tensor-contraction"] = (
        "dense-affine"
    )
    input_axes: tuple[int, ...] = ()
    output_axes: tuple[int, ...] = (0,)
    coordinate_mode: Literal["real", "complex-cartesian"] = "real"
    sharing_group: str | None = None
    reshape: tuple[int, ...] = ()
    permutation: tuple[int, ...] = ()
    coordinate_pullback: Any | None = None


class KFACLayoutProvider(abc.ABC):
    """Model that explicitly exposes affine blocks supported by KFAC."""

    @abc.abstractmethod
    def kfac_affine_blocks(self) -> tuple[KFACAffineBlock, ...]:
        raise NotImplementedError

    def kfac_validation_errors(self) -> tuple[str, ...]:
        return ()


__all__ = ["KFACAffineBlock", "KFACLayoutProvider"]
