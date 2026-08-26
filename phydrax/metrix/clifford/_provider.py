#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._blades import CliffordBladeLayout
from ._involutions import clifford_conjugate
from ._product import CliffordProductPlan
from ._spec import CliffordAlgebraSpec


class CliffordFiniteAlgebraProvider(StrictModule, NonTrainableState):
    """Expose one full Clifford blade layout through the finite-algebra provider API."""

    algebra: CliffordAlgebraSpec
    layout: CliffordBladeLayout
    basis_ids: tuple[str, ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, algebra: CliffordAlgebraSpec, /):
        if not isinstance(algebra, CliffordAlgebraSpec):
            raise TypeError("algebra must be CliffordAlgebraSpec.")
        layout = CliffordBladeLayout.full(algebra)
        labels = tuple(
            "1" if not axes else "e" + "".join(str(axis) for axis in axes)
            for axes in layout.axes
        )
        self.algebra = algebra
        self.layout = layout
        self.basis_ids = labels
        self.provider_id = canonical_fingerprint(
            {
                "kind": "clifford-finite-algebra-provider-v1",
                "algebra": algebra.algebra_id,
                "layout": layout.layout_id,
                "basis": list(labels),
            }
        )

    @property
    def coordinate_dimension(self) -> int:
        return self.layout.blade_count

    @property
    def algebra_id(self) -> str:
        return self.provider_id

    def conjugate(self, value: ArrayLike, /):
        return clifford_conjugate(value, self.layout)

    def prepare_product(self, **kwargs) -> CliffordProductPlan:
        return CliffordProductPlan(self.algebra, self.layout, self.layout, **kwargs)


__all__ = ["CliffordFiniteAlgebraProvider"]
