#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
)
from ._high_order import ReferenceNodalFamily
from ._reference import FiniteElementSpec


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
    transfer = finite_element_p_transfer(
        coarse.finite_element(),
        fine.finite_element(),
    )
    return PTransferData(
        transfer.primal_prolongation,
        transfer.mass_projection,
        coarse.order,
        fine.order,
    )


PTransferRole = Literal[
    "primal-prolongation",
    "dual-pullback",
    "pairing-adjoint",
    "mass-projection",
]


class FiniteElementPTransfer(StrictModule, NonTrainableState):
    """Local fixed-mesh p-transfer with explicit primal and dual roles."""

    primal_prolongation: Array
    dual_pullback: Array
    pairing_adjoint: Array
    mass_projection: Array
    coarse_element_id: str = eqx.field(static=True)
    fine_element_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal_prolongation: ArrayLike,
        pairing_adjoint: ArrayLike,
        coarse_element_id: str,
        fine_element_id: str,
        /,
    ):
        prolongation = jnp.asarray(primal_prolongation)
        adjoint = jnp.asarray(pairing_adjoint)
        if prolongation.ndim != 2 or adjoint.shape != prolongation.T.shape:
            raise ValueError("Finite-element p-transfer shapes are incompatible.")
        identifiers = (str(coarse_element_id), str(fine_element_id))
        if any(not value for value in identifiers):
            raise ValueError("Finite-element p-transfer identities must be non-empty.")
        self.primal_prolongation = prolongation
        self.dual_pullback = prolongation.T
        self.pairing_adjoint = adjoint
        self.mass_projection = adjoint
        self.coarse_element_id, self.fine_element_id = identifiers
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-p-transfer-roles",
                "coarse": identifiers[0],
                "fine": identifiers[1],
                "shape": list(prolongation.shape),
            }
        )

    def apply(self, role: PTransferRole, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if role == "primal-prolongation":
            return self.primal_prolongation @ value_
        if role == "dual-pullback":
            return self.dual_pullback @ value_
        if role == "pairing-adjoint":
            return self.pairing_adjoint @ value_
        if role == "mass-projection":
            return self.mass_projection @ value_
        raise ValueError("Unknown p-transfer role.")


def finite_element_p_transfer(
    coarse: FiniteElementSpec,
    fine: FiniteElementSpec,
    /,
    *,
    coarse_mass: ArrayLike | None = None,
    fine_mass: ArrayLike | None = None,
) -> FiniteElementPTransfer:
    if not isinstance(coarse, FiniteElementSpec) or not isinstance(
        fine, FiniteElementSpec
    ):
        raise TypeError("p-transfer requires FiniteElementSpec values.")
    if (
        coarse.cell_kind != fine.cell_kind
        or coarse.conformity != fine.conformity
        or fine.degree <= coarse.degree
    ):
        raise ValueError(
            "p-transfer elements require one cell/conformity and increasing degree."
        )
    prolongation, _ = coarse.tabulate(fine.reference_nodes)
    if coarse_mass is None and fine_mass is None:
        pairing_adjoint = prolongation.T
    elif coarse_mass is None or fine_mass is None:
        raise ValueError("Pairing-aware p-transfer requires both mass matrices.")
    else:
        coarse_mass_ = jnp.asarray(coarse_mass)
        fine_mass_ = jnp.asarray(fine_mass)
        if coarse_mass_.shape != (
            coarse.local_dof_count,
            coarse.local_dof_count,
        ) or fine_mass_.shape != (fine.local_dof_count, fine.local_dof_count):
            raise ValueError("p-transfer mass matrix shapes are incompatible.")
        coarse_space = ArraySpace((coarse.local_dof_count,), dtype=coarse_mass_.dtype)
        coarse_operator = DenseLinearOperator(
            coarse_mass_,
            source=coarse_space,
            target=coarse_space,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "supplied mass matrix",
                    "positive_definite": "supplied mass matrix",
                },
            ),
        )
        prepared = factorize(
            coarse_operator,
            FactorizationPolicy(kind="cholesky"),
        )
        right_hand_side = prolongation.T @ fine_mass_
        result = prepared.solve(right_hand_side)
        pairing_adjoint = result.value
        pairing_adjoint = eqx.error_if(
            pairing_adjoint,
            ~jnp.all(result.successful),
            "Pairing-aware p-transfer mass solve failed.",
        )
    return FiniteElementPTransfer(
        prolongation,
        pairing_adjoint,
        coarse.element_id,
        fine.element_id,
    )


__all__ = [
    "FiniteElementPTransfer",
    "PTransferData",
    "PTransferRole",
    "finite_element_p_transfer",
    "quadrilateral_p_transfer",
]
