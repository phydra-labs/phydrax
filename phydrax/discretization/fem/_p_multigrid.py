#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
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


def _local_dof_count(cell_kind: str, order: int) -> int:
    if cell_kind == "triangle":
        return (order + 1) * (order + 2) // 2
    if cell_kind == "quadrilateral":
        return (order + 1) ** 2
    if cell_kind == "tetrahedron":
        return (order + 1) * (order + 2) * (order + 3) // 6
    if cell_kind == "hexahedron":
        return (order + 1) ** 3
    raise ValueError("Unknown finite-element cell kind.")


class FiniteElementPMultigridPolicy(StrictModule, NonTrainableState):
    degree_coarsening: PDegreeCoarsening = eqx.field(static=True)
    explicit_orders: tuple[int, ...] = eqx.field(static=True)
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)
    cycle: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree_coarsening: PDegreeCoarsening = "half-dofs",
        /,
        *,
        explicit_orders: tuple[int, ...] = (),
        pre_smoothing: int = 1,
        post_smoothing: int = 1,
        cycle: str = "v",
    ):
        coarsening = str(degree_coarsening)
        orders = tuple(int(value) for value in explicit_orders)
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
        if coarsening == "explicit" and (
            not orders
            or orders[-1] != 1
            or any(left <= right for left, right in zip(orders, orders[1:], strict=True))
        ):
            raise ValueError("Explicit p-level orders must decrease strictly to one.")
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

    def degree_sequence(self, cell_kind: str, fine_order: int, /) -> tuple[int, ...]:
        fine = int(fine_order)
        if fine < 1:
            raise ValueError("Fine polynomial order must be positive.")
        if self.degree_coarsening == "explicit":
            if self.explicit_orders[0] != fine:
                raise ValueError("Explicit p-level sequence must start at fine_order.")
            return self.explicit_orders
        orders = [fine]
        current = fine
        while current > 1:
            if self.degree_coarsening == "all-degrees":
                coarse = current - 1
            elif self.degree_coarsening == "half-degrees":
                coarse = max(1, (current + 1) // 2)
            else:
                fine_count = _local_dof_count(cell_kind, current)
                coarse = current - 1
                while coarse > 1 and _local_dof_count(cell_kind, coarse) > fine_count / 2:
                    coarse -= 1
            if coarse >= current:
                raise ValueError("p-level coarsening failed to reduce polynomial order.")
            orders.append(coarse)
            current = coarse
        return tuple(orders)


class FiniteElementPMultigridPlan(StrictModule, NonTrainableState):
    level_orders: tuple[int, ...] = eqx.field(static=True)
    hierarchy_builder: MultigridHierarchyBuilder
    policy: FiniteElementPMultigridPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level_orders: tuple[int, ...],
        hierarchy_builder: MultigridHierarchyBuilder,
        policy: FiniteElementPMultigridPolicy,
        /,
    ):
        if not isinstance(hierarchy_builder, MultigridHierarchyBuilder) or not isinstance(
            policy, FiniteElementPMultigridPolicy
        ):
            raise TypeError("p-multigrid plan requires hierarchy builder and policy.")
        orders = tuple(int(value) for value in level_orders)
        if len(orders) != len(hierarchy_builder.levels):
            raise ValueError("p-level orders must match hierarchy levels.")
        self.level_orders = orders
        self.hierarchy_builder = hierarchy_builder
        self.policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-p-multigrid-plan",
                "orders": list(orders),
                "hierarchy": hierarchy_builder.builder_id,
                "policy": policy.policy_id,
            }
        )


def finite_element_p_multigrid_plan(
    cell_kind: str,
    fine_order: int,
    operators: tuple[AbstractLinearOperator, ...],
    smoothers: tuple[AbstractPreconditioner | AbstractPreconditionerBuilder, ...],
    restrictions: tuple[AbstractLinearOperator, ...],
    prolongations: tuple[AbstractLinearOperator, ...],
    /,
    *,
    policy: FiniteElementPMultigridPolicy | None = None,
    properties: PreconditionerProperties | None = None,
) -> FiniteElementPMultigridPlan:
    selected = FiniteElementPMultigridPolicy() if policy is None else policy
    if not isinstance(selected, FiniteElementPMultigridPolicy):
        raise TypeError("policy must be FiniteElementPMultigridPolicy or None.")
    orders = selected.degree_sequence(cell_kind, fine_order)
    if (
        len(operators) != len(orders)
        or len(smoothers) != len(orders)
        or len(restrictions) != len(orders) - 1
        or len(prolongations) != len(orders) - 1
    ):
        raise ValueError(
            "p-multigrid operators, smoothers, and transfers are incomplete."
        )
    levels = []
    for index, (operator, smoother) in enumerate(zip(operators, smoothers, strict=True)):
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
        cycle_policy=MultigridCyclePolicy(selected.cycle),
    )
    return FiniteElementPMultigridPlan(orders, builder, selected)


__all__ = [
    "FiniteElementPMultigridPlan",
    "FiniteElementPMultigridPolicy",
    "PDegreeCoarsening",
    "finite_element_p_multigrid_plan",
]
