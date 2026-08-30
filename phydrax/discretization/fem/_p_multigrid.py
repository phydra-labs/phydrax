#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    GalerkinHierarchyBuilder,
    MultigridCyclePolicy,
    MultigridHierarchyBuilder,
    MultigridLevelBuilder,
    PreconditionerProperties,
)


PDegreeCoarsening = Literal[
    "all-degrees",
    "half-degrees",
    "half-dofs",
    "explicit",
]

PLevelOrder: TypeAlias = int | tuple[int, ...]
PCoarseOperatorSource = Literal["direct", "galerkin"]


def _order_axes(cell_kind: str, order: PLevelOrder, /) -> tuple[int, ...]:
    cell = str(cell_kind)
    if isinstance(order, tuple):
        dimension = {"quadrilateral": 2, "hexahedron": 3}.get(cell)
        if dimension is None:
            raise ValueError(
                "Anisotropic p-level orders are supported only for quadrilaterals "
                "and hexahedra."
            )
        axes = tuple(int(value) for value in order)
        if len(axes) != dimension or any(value < 1 for value in axes):
            raise ValueError(
                "Anisotropic p-level order must have one positive degree per axis."
            )
        return axes
    value = int(order)
    if value < 1:
        raise ValueError("Polynomial orders must be positive.")
    if cell not in ("triangle", "quadrilateral", "tetrahedron", "hexahedron"):
        raise ValueError("Unknown finite-element cell kind.")
    return (value,)


def _local_dof_count(cell_kind: str, order: PLevelOrder, /) -> int:
    axes = _order_axes(cell_kind, order)
    if isinstance(order, tuple):
        return prod(value + 1 for value in axes)
    value = axes[0]
    if cell_kind == "triangle":
        return (value + 1) * (value + 2) // 2
    if cell_kind == "quadrilateral":
        return (value + 1) ** 2
    if cell_kind == "tetrahedron":
        return (value + 1) * (value + 2) * (value + 3) // 6
    return (value + 1) ** 3


def _restore_order(axes: tuple[int, ...], anisotropic: bool, /) -> PLevelOrder:
    return axes if anisotropic else axes[0]


def _decremented_order(
    cell_kind: str,
    current: PLevelOrder,
    coarsening: PDegreeCoarsening,
    /,
) -> PLevelOrder:
    anisotropic = isinstance(current, tuple)
    axes = _order_axes(cell_kind, current)
    if coarsening == "all-degrees":
        coarse_axes = tuple(max(1, value - 1) for value in axes)
    elif coarsening == "half-degrees":
        coarse_axes = tuple(max(1, (value + 1) // 2) for value in axes)
    else:
        target = _local_dof_count(cell_kind, current) / 2
        candidate = list(axes)
        while _local_dof_count(
            cell_kind,
            _restore_order(tuple(candidate), anisotropic),
        ) > target and any(value > 1 for value in candidate):
            largest = max(candidate)
            axis = candidate.index(largest)
            candidate[axis] -= 1
        coarse_axes = tuple(candidate)
    return _restore_order(coarse_axes, anisotropic)


class FiniteElementPMultigridPolicy(StrictModule, NonTrainableState):
    degree_coarsening: PDegreeCoarsening = eqx.field(static=True)
    explicit_orders: tuple[PLevelOrder, ...] = eqx.field(static=True)
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)
    cycle: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree_coarsening: PDegreeCoarsening = "half-dofs",
        /,
        *,
        explicit_orders: tuple[PLevelOrder, ...] = (),
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
        cycle: str = "v",
    ):
        coarsening = str(degree_coarsening)
        orders = tuple(
            tuple(int(axis) for axis in value) if isinstance(value, tuple) else int(value)
            for value in explicit_orders
        )
        pre = int(pre_smoothing)
        post = int(post_smoothing)
        cycle_ = str(cycle)
        if coarsening not in (
            "all-degrees",
            "half-degrees",
            "half-dofs",
            "explicit",
        ):
            raise ValueError("Unknown p-multigrid degree coarsening policy.")
        if coarsening == "explicit":
            if not orders:
                raise ValueError("Explicit p-level orders must be non-empty.")
            anisotropic = isinstance(orders[0], tuple)
            if any(isinstance(value, tuple) != anisotropic for value in orders):
                raise ValueError(
                    "Explicit p-level orders cannot mix isotropic and anisotropic forms."
                )
            if anisotropic:
                widths = {len(value) for value in orders if isinstance(value, tuple)}
                terminal = orders[-1]
                if (
                    len(widths) != 1
                    or any(
                        any(axis < 1 for axis in value)
                        for value in orders
                        if isinstance(value, tuple)
                    )
                    or not isinstance(terminal, tuple)
                    or any(axis != 1 for axis in terminal)
                    or any(
                        not all(
                            left_axis >= right_axis
                            for left_axis, right_axis in zip(left, right, strict=True)
                        )
                        or left == right
                        for left, right in zip(orders, orders[1:], strict=True)
                        if isinstance(left, tuple) and isinstance(right, tuple)
                    )
                ):
                    raise ValueError(
                        "Explicit anisotropic p-level orders must decrease "
                        "componentwise and terminate at all ones."
                    )
            else:
                isotropic_orders = []
                for value in orders:
                    if isinstance(value, tuple):
                        raise RuntimeError(
                            "Validated isotropic p-level orders became anisotropic."
                        )
                    isotropic_orders.append(value)
                if isotropic_orders[-1] != 1 or any(
                    left <= right
                    for left, right in zip(
                        isotropic_orders,
                        isotropic_orders[1:],
                        strict=True,
                    )
                ):
                    raise ValueError(
                        "Explicit p-level orders must decrease strictly to one."
                    )
        if coarsening != "explicit" and orders:
            raise ValueError("explicit_orders require explicit degree coarsening.")
        if pre < 0 or post < 0 or cycle_ not in ("v", "w", "f", "full"):
            raise ValueError("p-multigrid smoothing/cycle policy is invalid.")
        self.degree_coarsening = coarsening
        self.explicit_orders = orders
        self.pre_smoothing = pre
        self.post_smoothing = post
        self.cycle = cycle_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-p-multigrid-policy",
                "degree_coarsening": coarsening,
                "explicit_orders": list(orders),
                "pre_smoothing": pre,
                "post_smoothing": post,
                "cycle": cycle_,
            }
        )

    def degree_sequence(
        self,
        cell_kind: str,
        fine_order: PLevelOrder,
        /,
    ) -> tuple[PLevelOrder, ...]:
        fine_axes = _order_axes(cell_kind, fine_order)
        if self.degree_coarsening == "explicit":
            explicit_axes = tuple(
                _order_axes(cell_kind, value) for value in self.explicit_orders
            )
            if isinstance(fine_order, tuple) != isinstance(
                self.explicit_orders[0], tuple
            ):
                raise ValueError(
                    "Explicit p-level sequence must preserve the fine-order form."
                )
            if explicit_axes[0] != fine_axes:
                raise ValueError("Explicit p-level sequence must start at fine_order.")
            return self.explicit_orders
        orders: list[PLevelOrder] = [fine_order]
        current = fine_order
        while any(value > 1 for value in _order_axes(cell_kind, current)):
            coarse = _decremented_order(
                cell_kind,
                current,
                self.degree_coarsening,
            )
            if _order_axes(cell_kind, coarse) == _order_axes(cell_kind, current):
                raise ValueError("p-level coarsening failed to reduce polynomial order.")
            orders.append(coarse)
            current = coarse
        return tuple(orders)


class FiniteElementPMultigridPlan(StrictModule, NonTrainableState):
    level_orders: tuple[PLevelOrder, ...] = eqx.field(static=True)
    hierarchy_builder: AbstractPreconditionerBuilder
    policy: FiniteElementPMultigridPolicy
    coarse_operator_source: PCoarseOperatorSource = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level_orders: tuple[PLevelOrder, ...],
        hierarchy_builder: AbstractPreconditionerBuilder,
        policy: FiniteElementPMultigridPolicy,
        coarse_operator_source: PCoarseOperatorSource,
        /,
    ):
        if not isinstance(
            hierarchy_builder, AbstractPreconditionerBuilder
        ) or not isinstance(policy, FiniteElementPMultigridPolicy):
            raise TypeError("p-multigrid plan requires hierarchy builder and policy.")
        source = str(coarse_operator_source)
        if source not in ("direct", "galerkin"):
            raise ValueError("Unknown p-multigrid coarse-operator source.")
        orders = tuple(level_orders)
        if len(orders) < 2:
            raise ValueError("A p-multigrid plan requires at least two p-levels.")
        self.level_orders = orders
        self.hierarchy_builder = hierarchy_builder
        self.policy = policy
        self.coarse_operator_source = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-p-multigrid-plan",
                "orders": list(orders),
                "hierarchy": hierarchy_builder.builder_id,
                "policy": policy.policy_id,
                "coarse_operator_source": source,
            }
        )


def finite_element_p_multigrid_plan(
    cell_kind: str,
    fine_order: PLevelOrder,
    operators: tuple[AbstractLinearOperator, ...],
    smoothers: tuple[AbstractPreconditioner | AbstractPreconditionerBuilder, ...],
    restrictions: tuple[AbstractLinearOperator, ...],
    prolongations: tuple[AbstractLinearOperator, ...],
    /,
    *,
    policy: FiniteElementPMultigridPolicy | None = None,
    properties: PreconditionerProperties | None = None,
    coarse_operator_source: PCoarseOperatorSource = "direct",
) -> FiniteElementPMultigridPlan:
    selected = FiniteElementPMultigridPolicy() if policy is None else policy
    if not isinstance(selected, FiniteElementPMultigridPolicy):
        raise TypeError("policy must be FiniteElementPMultigridPolicy or None.")
    orders = selected.degree_sequence(cell_kind, fine_order)
    source = str(coarse_operator_source)
    if source not in ("direct", "galerkin"):
        raise ValueError("Unknown p-multigrid coarse-operator source.")
    expected_operators = len(orders) if source == "direct" else 1
    if (
        len(operators) != expected_operators
        or len(smoothers) != len(orders)
        or len(restrictions) != len(orders) - 1
        or len(prolongations) != len(orders) - 1
    ):
        raise ValueError(
            "p-multigrid operators, smoothers, and transfers are incomplete."
        )
    cycle_policy = MultigridCyclePolicy(selected.cycle)
    if source == "galerkin":
        builder: AbstractPreconditionerBuilder = GalerkinHierarchyBuilder(
            tuple(zip(restrictions, prolongations, strict=True)),
            smoothers[:-1],
            smoothers[-1],
            properties=properties,
            cycle_policy=cycle_policy,
            pre_smoothing=selected.pre_smoothing,
            post_smoothing=selected.post_smoothing,
        )
    else:
        levels = []
        for index, (operator, smoother) in enumerate(
            zip(operators, smoothers, strict=True)
        ):
            if index == len(orders) - 1:
                levels.append(MultigridLevelBuilder(operator, smoother))
            else:
                levels.append(
                    MultigridLevelBuilder(
                        operator,
                        smoother,
                        restriction=restrictions[index],
                        prolongation=prolongations[index],
                        pre_smoothing=selected.pre_smoothing,
                        post_smoothing=selected.post_smoothing,
                    )
                )
        builder = MultigridHierarchyBuilder(
            tuple(levels),
            properties=properties,
            cycle_policy=cycle_policy,
        )
    return FiniteElementPMultigridPlan(orders, builder, selected, source)


__all__ = [
    "FiniteElementPMultigridPlan",
    "FiniteElementPMultigridPolicy",
    "PCoarseOperatorSource",
    "PDegreeCoarsening",
    "PLevelOrder",
    "finite_element_p_multigrid_plan",
]
