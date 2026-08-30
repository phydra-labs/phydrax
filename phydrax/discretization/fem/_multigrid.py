#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
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


def quadrilateral_p_transfer(
    coarse: ReferenceNodalFamily,
    fine: ReferenceNodalFamily,
    /,
) -> FiniteElementPTransfer:
    if not isinstance(coarse, ReferenceNodalFamily) or not isinstance(
        fine, ReferenceNodalFamily
    ):
        raise TypeError("p-transfer requires ReferenceNodalFamily values.")
    if (
        coarse.cell_kind != "quadrilateral"
        or fine.cell_kind != "quadrilateral"
        or len(coarse.orders) != len(fine.orders)
        or not all(
            fine_order >= coarse_order
            for coarse_order, fine_order in zip(
                coarse.orders,
                fine.orders,
                strict=True,
            )
        )
        or not any(
            fine_order > coarse_order
            for coarse_order, fine_order in zip(
                coarse.orders,
                fine.orders,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "Quadrilateral p-transfer requires nested axis orders with at least "
            "one strict increase."
        )
    return finite_element_p_transfer(
        coarse.finite_element(),
        fine.finite_element(),
    )


PTransferRole = Literal[
    "primal-prolongation",
    "dual-pullback",
    "pairing-adjoint",
    "mass-projection",
]


class FiniteElementPTransfer(StrictModule, NonTrainableState):
    """Fixed-mesh p-transfer whose four mathematical roles remain explicit."""

    primal_prolongation: Array
    dual_pullback: Array
    pairing_adjoint: Array
    mass_projection: Array | None
    coarse_element_id: str = eqx.field(static=True)
    fine_element_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal_prolongation: ArrayLike,
        pairing_adjoint: ArrayLike,
        mass_projection: ArrayLike | None,
        coarse_element_id: str,
        fine_element_id: str,
        /,
    ):
        prolongation = jnp.asarray(primal_prolongation)
        adjoint = jnp.asarray(pairing_adjoint)
        projection = None if mass_projection is None else jnp.asarray(mass_projection)
        if prolongation.ndim != 2 or adjoint.shape != prolongation.T.shape:
            raise ValueError("Finite-element p-transfer shapes are incompatible.")
        if projection is not None and projection.shape != prolongation.T.shape:
            raise ValueError("Finite-element mass projection has an incompatible shape.")
        if not all(
            jnp.issubdtype(value.dtype, jnp.inexact) for value in (prolongation, adjoint)
        ) or (
            projection is not None and not jnp.issubdtype(projection.dtype, jnp.inexact)
        ):
            raise TypeError("Finite-element p-transfer matrices must be inexact.")
        identifiers = (str(coarse_element_id), str(fine_element_id))
        if any(not value for value in identifiers):
            raise ValueError("Finite-element p-transfer identities must be non-empty.")
        dual = prolongation.T
        self.primal_prolongation = prolongation
        self.dual_pullback = dual
        self.pairing_adjoint = adjoint
        self.mass_projection = projection
        self.coarse_element_id, self.fine_element_id = identifiers
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-p-transfer-roles",
                "coarse": identifiers[0],
                "fine": identifiers[1],
                "primal": array_tree_fingerprint(np.asarray(prolongation)),
                "pairing_adjoint": array_tree_fingerprint(np.asarray(adjoint)),
                "mass_projection": (
                    None
                    if projection is None
                    else array_tree_fingerprint(np.asarray(projection))
                ),
            }
        )

    def prolong(self, value: ArrayLike, /) -> Array:
        return self.primal_prolongation @ jnp.asarray(value)

    def apply(self, role: PTransferRole, value: ArrayLike, /) -> Array:
        value_ = jnp.asarray(value)
        if role == "primal-prolongation":
            return self.primal_prolongation @ value_
        if role == "dual-pullback":
            return self.dual_pullback @ value_
        if role == "pairing-adjoint":
            return self.pairing_adjoint @ value_
        if role == "mass-projection":
            if self.mass_projection is None:
                raise ValueError(
                    "Mass projection requires coarse_mass and fine_mass at construction."
                )
            return self.mass_projection @ value_
        raise ValueError("Unknown p-transfer role.")


def _tensor_orders(element: FiniteElementSpec, /) -> tuple[int, ...] | None:
    if element.cell_kind not in ("quadrilateral", "hexahedron") or element.family not in (
        "Lagrange",
        "SimplexLagrange",
        "TensorProductLagrange",
    ):
        return None
    nodes = np.asarray(element.reference_nodes)
    return tuple(np.unique(nodes[:, axis]).size - 1 for axis in range(nodes.shape[1]))


def finite_element_p_transfer(
    coarse: FiniteElementSpec,
    fine: FiniteElementSpec,
    /,
    *,
    coarse_pairing: ArrayLike | None = None,
    fine_pairing: ArrayLike | None = None,
    coarse_mass: ArrayLike | None = None,
    fine_mass: ArrayLike | None = None,
) -> FiniteElementPTransfer:
    if not isinstance(coarse, FiniteElementSpec) or not isinstance(
        fine, FiniteElementSpec
    ):
        raise TypeError("p-transfer requires FiniteElementSpec values.")
    lagrange_families = {
        "Lagrange",
        "SimplexLagrange",
        "TensorProductLagrange",
    }
    compatible_family = coarse.family == fine.family or (
        coarse.family in lagrange_families and fine.family in lagrange_families
    )
    coarse_orders = _tensor_orders(coarse)
    fine_orders = _tensor_orders(fine)
    if coarse_orders is None and fine_orders is None:
        increasing_degree = fine.degree > coarse.degree
    elif coarse_orders is not None and fine_orders is not None:
        increasing_degree = (
            len(coarse_orders) == len(fine_orders)
            and all(
                fine_order >= coarse_order
                for coarse_order, fine_order in zip(
                    coarse_orders, fine_orders, strict=True
                )
            )
            and any(
                fine_order > coarse_order
                for coarse_order, fine_order in zip(
                    coarse_orders, fine_orders, strict=True
                )
            )
        )
    else:
        increasing_degree = False
    if (
        coarse.cell_kind != fine.cell_kind
        or not compatible_family
        or coarse.conformity != fine.conformity
        or coarse.representation != fine.representation
        or coarse.mapping != fine.mapping
        or coarse.value_shape != fine.value_shape
        or not increasing_degree
    ):
        raise ValueError(
            "p-transfer requires one compatible finite-element family, "
            "representation, mapping, cell, and conformity with increasing degree."
        )
    prolongation, _ = coarse.tabulate(fine.reference_nodes)
    if coarse_pairing is None and fine_pairing is None:
        pairing_adjoint = prolongation.T
    elif coarse_pairing is None or fine_pairing is None:
        raise ValueError(
            "Pairing-aware p-transfer requires both coarse and fine pairings."
        )
    else:
        pairing_adjoint = _weighted_adjoint(
            prolongation,
            coarse_pairing,
            fine_pairing,
            coarse.local_dof_count,
            fine.local_dof_count,
            role="pairing adjoint",
        )
    if coarse_mass is None and fine_mass is None:
        mass_projection = None
    elif coarse_mass is None or fine_mass is None:
        raise ValueError(
            "Finite-element mass projection requires both physical mass matrices."
        )
    else:
        mass_projection = _weighted_adjoint(
            prolongation,
            coarse_mass,
            fine_mass,
            coarse.local_dof_count,
            fine.local_dof_count,
            role="mass projection",
        )
    return FiniteElementPTransfer(
        prolongation,
        pairing_adjoint,
        mass_projection,
        coarse.element_id,
        fine.element_id,
    )


def _weighted_adjoint(
    prolongation: Array,
    coarse_metric: ArrayLike,
    fine_metric: ArrayLike,
    coarse_size: int,
    fine_size: int,
    /,
    *,
    role: str,
) -> Array:
    coarse_matrix = jnp.asarray(coarse_metric)
    fine_matrix = jnp.asarray(fine_metric)
    if coarse_matrix.shape != (coarse_size, coarse_size) or fine_matrix.shape != (
        fine_size,
        fine_size,
    ):
        raise ValueError(f"Finite-element {role} matrix shapes are incompatible.")
    coarse_space = ArraySpace((coarse_size,), dtype=coarse_matrix.dtype)
    coarse_operator = DenseLinearOperator(
        coarse_matrix,
        source=coarse_space,
        target=coarse_space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_definite": "asserted",
            },
        ),
    )
    prepared = factorize(
        coarse_operator,
        FactorizationPolicy("cholesky"),
    )
    result = prepared.solve(prolongation.T @ fine_matrix)
    return eqx.error_if(
        result.value,
        ~jnp.all(result.successful),
        f"Finite-element {role} metric solve failed.",
    )


__all__ = [
    "FiniteElementPTransfer",
    "PTransferRole",
    "finite_element_p_transfer",
    "quadrilateral_p_transfer",
]
