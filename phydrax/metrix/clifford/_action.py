#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg._compound import compound_matrix
from ._blades import CliffordBladeLayout
from ._isometries import MetricIsometryAction, MetricIsometryAuditSet
from ._product import prepare_product


class CliffordOutermorphismPlan(StrictModule, NonTrainableState):
    """Grade-preserving outermorphism induced by one metric isometry."""

    __hash__ = object.__hash__

    action: MetricIsometryAction
    layout: CliffordBladeLayout
    representation: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: MetricIsometryAction,
        layout: CliffordBladeLayout,
        /,
    ) -> None:
        if not isinstance(action, MetricIsometryAction):
            raise TypeError("action must be a MetricIsometryAction.")
        if not isinstance(layout, CliffordBladeLayout):
            raise TypeError("layout must be a CliffordBladeLayout.")
        action.algebra.require_compatible(layout.algebra)
        if not layout.complete_grades:
            raise ValueError(
                "Clifford outermorphism requires a union of complete grade supports."
            )
        representation = jnp.zeros(
            (layout.blade_count, layout.blade_count), dtype=action.matrix.dtype
        )
        for grade in sorted(set(layout.grades)):
            positions = jnp.asarray(layout.grade_positions(grade), dtype=jnp.int32)
            grade_action = compound_matrix(action.matrix, grade)
            representation = representation.at[
                positions[:, None], positions[None, :]
            ].set(grade_action)
        self.action = action
        self.layout = layout
        self.representation = representation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "clifford-outermorphism-plan",
                "action": action.action_id,
                "layout": layout.layout_id,
            }
        )

    def __call__(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.layout.blade_count:
            raise ValueError(
                "Clifford action input must end in the prepared blade count."
            )
        return ein.contract(
            "oi,...i->...o",
            self.representation.astype(array.dtype),
            array,
            backend="jax",
        )


class CliffordActionAuditReport(StrictModule, NonTrainableState):
    """Numerical evidence that one outermorphism preserves Clifford products."""

    finite: Array
    valid: Array
    metric_defect: Array
    automorphism_defect: Array
    tolerance: Array
    action_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        finite: ArrayLike,
        metric_defect: ArrayLike,
        automorphism_defect: ArrayLike,
        tolerance: ArrayLike,
        action_id: str,
        layout_id: str,
    ) -> None:
        finite_ = jnp.asarray(finite, dtype=jnp.bool_)
        metric = jnp.asarray(metric_defect)
        automorphism = jnp.asarray(automorphism_defect)
        tolerance_ = jnp.asarray(tolerance)
        for value, name in (
            (metric, "metric_defect"),
            (automorphism, "automorphism_defect"),
            (tolerance_, "tolerance"),
        ):
            if value.shape != ():
                raise ValueError(f"{name} must be scalar.")
        self.finite = finite_
        self.metric_defect = metric
        self.automorphism_defect = automorphism
        self.tolerance = tolerance_
        self.valid = finite_ & (metric <= tolerance_) & (automorphism <= tolerance_)
        self.action_id = str(action_id)
        self.layout_id = str(layout_id)
        self.report_id = canonical_fingerprint(
            {
                "kind": "clifford-action-audit",
                "action": self.action_id,
                "layout": self.layout_id,
                "metric_defect": float(metric),
                "automorphism_defect": float(automorphism),
                "tolerance": float(tolerance_),
            }
        )


def audit_clifford_action(
    action: MetricIsometryAction,
    /,
    *,
    tolerance: float | None = None,
) -> CliffordActionAuditReport:
    """Exhaustively audit one action over all basis-blade products."""
    layout = CliffordBladeLayout.full(action.algebra)
    plan = CliffordOutermorphismPlan(action, layout)
    product = prepare_product(
        action.algebra,
        layout,
        layout,
        output_layout=layout,
        backend="sparse",
    )
    basis = jnp.eye(layout.blade_count, dtype=action.matrix.dtype)
    left = basis[:, None, :]
    right = basis[None, :, :]
    transformed_product = plan(product(left, right))
    product_of_transformed = product(plan(left), plan(right))
    defect = jnp.max(jnp.abs(transformed_product - product_of_transformed))
    tolerance_ = action.tolerance if tolerance is None else float(tolerance)
    finite = (
        jnp.all(jnp.isfinite(plan.representation))
        & jnp.isfinite(defect)
        & jnp.isfinite(action.metric_defect)
    )
    return CliffordActionAuditReport(
        finite=finite,
        metric_defect=action.metric_defect,
        automorphism_defect=defect,
        tolerance=tolerance_,
        action_id=action.action_id,
        layout_id=layout.layout_id,
    )


def audit_clifford_actions(
    actions: MetricIsometryAuditSet,
    /,
    *,
    tolerance: float | None = None,
) -> tuple[CliffordActionAuditReport, ...]:
    """Audit independent isometries without asserting finite-group closure."""
    if not isinstance(actions, MetricIsometryAuditSet):
        raise TypeError("actions must be a MetricIsometryAuditSet.")
    return tuple(
        audit_clifford_action(action, tolerance=tolerance) for action in actions.actions
    )


__all__ = [
    "audit_clifford_action",
    "audit_clifford_actions",
    "CliffordActionAuditReport",
    "CliffordOutermorphismPlan",
]
