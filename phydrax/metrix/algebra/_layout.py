#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import AbstractFiniteRealAlgebraSpec


class AlgebraElementLayout(StrictModule, NonTrainableState):
    """Complete ordered coordinate layout for one finite real algebra."""

    algebra: AbstractFiniteRealAlgebraSpec
    basis_indices: tuple[int, ...] = eqx.field(static=True)
    basis_ids: tuple[str, ...] = eqx.field(static=True)
    algebra_axis: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: AbstractFiniteRealAlgebraSpec,
        /,
        *,
        algebra_axis: int = -1,
    ):
        if not isinstance(algebra, AbstractFiniteRealAlgebraSpec):
            raise TypeError("algebra must implement AbstractFiniteRealAlgebraSpec.")
        axis = int(algebra_axis)
        self.algebra = algebra
        self.basis_indices = tuple(range(algebra.coordinate_dimension))
        self.basis_ids = algebra.basis_ids
        self.algebra_axis = axis
        self.layout_id = canonical_fingerprint(
            {
                "kind": "algebra-element-layout-v1",
                "algebra": algebra.algebra_id,
                "basis": list(self.basis_indices),
                "axis": axis,
            }
        )

    @property
    def coordinate_dimension(self) -> int:
        return self.algebra.coordinate_dimension


__all__ = ["AlgebraElementLayout"]
