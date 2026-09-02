#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator
from ._spec import VirtualElementFieldSpec


class CurvedVirtualElementEdge(StrictModule, NonTrainableState):
    points: Array
    tangents: Array
    arc_weights: Array
    chart_id: str = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart_id: str,
        points: ArrayLike,
        tangents: ArrayLike,
        reference_weights: ArrayLike,
        /,
    ):
        p = np.asarray(points, dtype=float)
        t = np.asarray(tangents, dtype=float)
        w = np.asarray(reference_weights, dtype=float)
        if (
            not str(chart_id)
            or p.ndim != 2
            or t.shape != p.shape
            or w.shape != (p.shape[0],)
        ):
            raise ValueError("Curved VEM edge chart arrays are incompatible.")
        jac = np.linalg.norm(t, axis=1)
        if (
            np.any(~np.isfinite(p))
            or np.any(~np.isfinite(t))
            or np.any(jac <= 0)
            or np.any(w <= 0)
        ):
            raise ValueError(
                "Curved VEM edge requires finite positive chart Jacobians/weights."
            )
        self.points = jnp.asarray(p)
        self.tangents = jnp.asarray(t)
        self.arc_weights = jnp.asarray(w * jac)
        self.chart_id = str(chart_id)
        self.minimum_jacobian = float(np.min(jac))
        self.edge_id = canonical_fingerprint(
            {
                "kind": "curved-vem-edge",
                "chart": chart_id,
                "points": array_tree_fingerprint(p),
                "tangents": array_tree_fingerprint(t),
            }
        )


class VirtualElementProductPlan(StrictModule, NonTrainableState):
    fields: tuple[VirtualElementFieldSpec, ...]
    block_matrix: Array
    operator: DenseLinearOperator
    field_offsets: tuple[int, ...] = eqx.field(static=True)
    inf_sup_margin: float = eqx.field(static=True)
    commuting_defect: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: tuple[VirtualElementFieldSpec, ...],
        field_sizes: tuple[int, ...],
        block_matrix: ArrayLike,
        /,
        *,
        inf_sup_margin: float,
        commuting_defect: float,
        maximum_commuting_defect: float = 1e-8,
    ):
        if (
            len(fields) < 2
            or len(fields) != len(field_sizes)
            or not all(isinstance(v, VirtualElementFieldSpec) for v in fields)
        ):
            raise ValueError(
                "Mixed VEM products require aligned field specifications/sizes."
            )
        offsets = np.cumsum((0, *field_sizes))
        matrix = np.asarray(block_matrix)
        if (
            matrix.shape != (offsets[-1], offsets[-1])
            or float(inf_sup_margin) <= 0
            or float(commuting_defect) > float(maximum_commuting_defect)
        ):
            raise ValueError(
                "Mixed VEM block/rank/commuting evidence is outside its envelope."
            )
        self.fields = fields
        self.block_matrix = jnp.asarray(matrix)
        self.operator = DenseLinearOperator(
            self.block_matrix,
            operator_id=canonical_fingerprint(
                {"kind": "vem-product-operator", "matrix": array_tree_fingerprint(matrix)}
            ),
        )
        self.field_offsets = tuple(int(v) for v in offsets)
        self.inf_sup_margin = float(inf_sup_margin)
        self.commuting_defect = float(commuting_defect)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vem-product-plan",
                "fields": [v.field_spec_id for v in fields],
                "sizes": field_sizes,
                "matrix": array_tree_fingerprint(matrix),
            }
        )


class VirtualElementAdaptivityPolicy(StrictModule, NonTrainableState):
    fraction: float = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        fraction: float = 0.5,
        maximum_degree: int = 8,
        maximum_cells: int = 1_000_000,
    ):
        if (
            not 0 < float(fraction) <= 1
            or int(maximum_degree) < 1
            or int(maximum_cells) < 1
        ):
            raise ValueError("VEM adaptivity bounds are invalid.")
        self.fraction = float(fraction)
        self.maximum_degree = int(maximum_degree)
        self.maximum_cells = int(maximum_cells)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "vem-adaptivity-policy",
                "fraction": fraction,
                "maximum_degree": maximum_degree,
                "maximum_cells": maximum_cells,
            }
        )


class VirtualElementEpoch(StrictModule, NonTrainableState):
    cell_global_ids: Array
    degrees: Array
    generation: int = eqx.field(static=True)
    parent_epoch_id: str | None = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_global_ids: ArrayLike,
        degrees: ArrayLike,
        /,
        *,
        generation: int = 0,
        parent_epoch_id: str | None = None,
    ):
        ids = np.asarray(cell_global_ids, dtype=np.int64)
        degree = np.asarray(degrees, dtype=np.int32)
        if (
            ids.ndim != 1
            or degree.shape != ids.shape
            or np.unique(ids).size != ids.size
            or np.any(degree < 1)
        ):
            raise ValueError("VEM epoch IDs/degrees are invalid.")
        self.cell_global_ids = jnp.asarray(ids)
        self.degrees = jnp.asarray(degree)
        self.generation = int(generation)
        self.parent_epoch_id = parent_epoch_id
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "vem-epoch",
                "ids": array_tree_fingerprint(ids),
                "degrees": array_tree_fingerprint(degree),
                "generation": generation,
                "parent": parent_epoch_id,
            }
        )


class VirtualElementAdaptationResult(StrictModule, NonTrainableState):
    source: VirtualElementEpoch
    target: VirtualElementEpoch
    marked: Array
    transfer: Array
    conservation_defect: float = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def adapt_virtual_element_p(
    epoch: VirtualElementEpoch,
    indicators: ArrayLike,
    transfer: ArrayLike,
    policy: VirtualElementAdaptivityPolicy,
    /,
) -> VirtualElementAdaptationResult:
    values = np.asarray(indicators, dtype=float)
    matrix = np.asarray(transfer)
    if values.shape != epoch.degrees.shape or np.any(values < 0) or matrix.ndim != 2:
        raise ValueError("VEM adaptation indicators/transfer are invalid.")
    order = np.lexsort((np.asarray(epoch.cell_global_ids), -values))
    total = np.sum(values**2)
    cumulative = np.cumsum(values[order] ** 2)
    count = (
        1 if total == 0 else int(np.searchsorted(cumulative, policy.fraction * total) + 1)
    )
    marked = np.sort(order[:count])
    degree = np.asarray(epoch.degrees).copy()
    degree[marked] += 1
    if np.any(degree > policy.maximum_degree):
        raise ValueError("VEM p-adaptation exceeds maximum_degree.")
    target = VirtualElementEpoch(
        epoch.cell_global_ids,
        degree,
        generation=epoch.generation + 1,
        parent_epoch_id=epoch.epoch_id,
    )
    constant = np.ones((matrix.shape[1],))
    defect = float(
        np.linalg.norm(matrix @ constant - np.ones((matrix.shape[0],)), ord=np.inf)
    )
    return VirtualElementAdaptationResult(
        epoch,
        target,
        jnp.asarray(marked),
        jnp.asarray(matrix),
        defect,
        canonical_fingerprint(
            {
                "kind": "vem-adaptation-result",
                "source": epoch.epoch_id,
                "target": target.epoch_id,
                "marked": marked.tolist(),
                "transfer": array_tree_fingerprint(matrix),
            }
        ),
    )


class PreparedPolyhedralPolynomialVEM3D(StrictModule, NonTrainableState):
    """Bounded higher-degree polyhedral VEM from exact moment/projector data."""

    operator: DenseLinearOperator
    polynomial_projector: Array
    polynomial_reproduction_defect: float = eqx.field(static=True)
    projector_rank_margin: float = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def prepare_polyhedral_polynomial_vem_3d(
    consistent_matrix: ArrayLike,
    stabilization_matrix: ArrayLike,
    polynomial_projector: ArrayLike,
    polynomial_dof_values: ArrayLike,
    /,
    *,
    degree: int,
    maximum_dofs: int = 4096,
    minimum_rank_margin: float = 1e-10,
) -> PreparedPolyhedralPolynomialVEM3D:
    """Prepare a degree-k 3-D moment VEM after root-topology cubature assembly."""

    consistent = np.asarray(consistent_matrix)
    stabilization = np.asarray(stabilization_matrix)
    projector = np.asarray(polynomial_projector)
    polynomial_values = np.asarray(polynomial_dof_values)
    degree_ = int(degree)
    if degree_ < 1 or consistent.ndim != 2 or consistent.shape[0] != consistent.shape[1]:
        raise ValueError(
            "Polyhedral polynomial VEM requires degree >= 1 and square local action."
        )
    if (
        consistent.shape != stabilization.shape
        or projector.shape[1] != consistent.shape[0]
    ):
        raise ValueError("Polyhedral VEM action/projector shapes are incompatible.")
    if (
        consistent.shape[0] > int(maximum_dofs)
        or polynomial_values.shape[0] != consistent.shape[0]
    ):
        raise ValueError(
            "Polyhedral VEM DOF capacity or polynomial moment shape is invalid."
        )
    singular = np.linalg.svd(projector, compute_uv=False)
    rank_margin = float(singular[-1])
    if rank_margin < float(minimum_rank_margin):
        raise ValueError("Polyhedral VEM projector rank margin is exhausted.")
    reproduction = float(
        np.linalg.norm(
            projector @ polynomial_values - np.eye(projector.shape[0]), ord=np.inf
        )
    )
    if reproduction > 1e-8:
        raise ValueError("Polyhedral VEM moment projector does not reproduce P_k.")
    matrix = consistent + stabilization
    operator = DenseLinearOperator(
        jnp.asarray(matrix),
        operator_id=canonical_fingerprint(
            {
                "kind": "polyhedral-polynomial-vem-3d",
                "degree": degree_,
                "matrix": array_tree_fingerprint(matrix),
            }
        ),
    )
    return PreparedPolyhedralPolynomialVEM3D(
        operator,
        jnp.asarray(projector),
        reproduction,
        rank_margin,
        degree_,
        canonical_fingerprint(
            {
                "kind": "prepared-polyhedral-polynomial-vem-3d",
                "operator": operator.operator_id,
                "degree": degree_,
            }
        ),
    )


def adapt_virtual_element_hp(
    epoch: VirtualElementEpoch,
    target_cell_global_ids: ArrayLike,
    target_degrees: ArrayLike,
    transfer: ArrayLike,
    policy: VirtualElementAdaptivityPolicy,
    /,
) -> VirtualElementAdaptationResult:
    """Commit one explicit conforming h/p/hp topology transaction."""

    ids = np.asarray(target_cell_global_ids, dtype=np.int64)
    degrees = np.asarray(target_degrees, dtype=np.int32)
    matrix = np.asarray(transfer)
    if ids.ndim != 1 or ids.size > policy.maximum_cells or degrees.shape != ids.shape:
        raise ValueError("Target VEM hp epoch exceeds cell/degree shape bounds.")
    if np.any(degrees < 1) or np.any(degrees > policy.maximum_degree):
        raise ValueError("Target VEM hp degrees exceed the declared envelope.")
    if matrix.ndim != 2 or matrix.shape[0] < ids.size:
        raise ValueError("VEM hp transfer does not cover the target epoch.")
    constant = np.ones((matrix.shape[1],))
    defect = float(
        np.linalg.norm(matrix @ constant - np.ones((matrix.shape[0],)), ord=np.inf)
    )
    if defect > 1e-10:
        raise ValueError("VEM hp transfer does not preserve constants.")
    target = VirtualElementEpoch(
        ids,
        degrees,
        generation=epoch.generation + 1,
        parent_epoch_id=epoch.epoch_id,
    )
    marked = np.arange(ids.size, dtype=np.int32)
    return VirtualElementAdaptationResult(
        epoch,
        target,
        jnp.asarray(marked),
        jnp.asarray(matrix),
        defect,
        canonical_fingerprint(
            {
                "kind": "vem-hp-adaptation-result",
                "source": epoch.epoch_id,
                "target": target.epoch_id,
                "transfer": array_tree_fingerprint(matrix),
            }
        ),
    )


__all__ = [
    "CurvedVirtualElementEdge",
    "PreparedPolyhedralPolynomialVEM3D",
    "VirtualElementAdaptationResult",
    "VirtualElementAdaptivityPolicy",
    "VirtualElementEpoch",
    "VirtualElementProductPlan",
    "adapt_virtual_element_hp",
    "adapt_virtual_element_p",
    "prepare_polyhedral_polynomial_vem_3d",
]
