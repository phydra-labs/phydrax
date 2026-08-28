#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._high_order import ReferenceNodalFamily


class PTransferData(StrictModule, NonTrainableState):
    prolongation: Array
    restriction: Array
    coarse_order: int = eqx.field(static=True)
    fine_order: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        prolongation: ArrayLike,
        restriction: ArrayLike,
        coarse_order: int,
        fine_order: int,
        /,
    ):
        prolongation_ = jnp.asarray(prolongation)
        restriction_ = jnp.asarray(restriction)
        coarse = int(coarse_order)
        fine = int(fine_order)
        if prolongation_.ndim != 2 or restriction_.shape != prolongation_.T.shape:
            raise ValueError("p-transfer operators have incompatible shapes.")
        if coarse < 1 or fine <= coarse:
            raise ValueError("p-transfer orders must satisfy 1 <= coarse < fine.")
        self.prolongation = prolongation_
        self.restriction = restriction_
        self.coarse_order = coarse
        self.fine_order = fine
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-p-transfer",
                "coarse_order": coarse,
                "fine_order": fine,
                "shape": list(prolongation_.shape),
            }
        )

    def prolong(self, value: ArrayLike, /) -> Array:
        return self.prolongation @ jnp.asarray(value)

    def restrict(self, value: ArrayLike, /) -> Array:
        return self.restriction @ jnp.asarray(value)

    def galerkin(self, fine_operator: ArrayLike, /) -> Array:
        operator = jnp.asarray(fine_operator)
        return oe.contract(
            "ai,ab,bj->ij",
            self.prolongation,
            operator,
            self.prolongation,
        )


def quadrilateral_p_transfer(
    coarse: ReferenceNodalFamily,
    fine: ReferenceNodalFamily,
    /,
) -> PTransferData:
    if not isinstance(coarse, ReferenceNodalFamily) or not isinstance(
        fine, ReferenceNodalFamily
    ):
        raise TypeError("p-transfer requires ReferenceNodalFamily values.")
    if coarse.cell_kind != fine.cell_kind or fine.order <= coarse.order:
        raise ValueError("p-transfer families require one cell and increasing order.")
    fine_nodes = jnp.stack(
        jnp.meshgrid(fine.axis_nodes, fine.axis_nodes, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    prolongation, _ = coarse.tabulate(fine_nodes)
    mass = prolongation.T @ prolongation
    restriction = jnp.linalg.solve(mass, prolongation.T)
    return PTransferData(
        prolongation,
        restriction,
        coarse.order,
        fine.order,
    )


__all__ = ["PTransferData", "quadrilateral_p_transfer"]
