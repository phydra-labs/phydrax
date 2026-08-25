#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....integration import AdaptiveQuadraturePlan, IntegrationStatus
from ._core import LayerPotentialTargetReport


class LayerEvaluationPlan2D(StrictModule, NonTrainableState):
    """Explicit evaluator policy for a two-dimensional layer representation."""

    method: Literal["direct", "adaptive", "qbx"] = eqx.field(static=True)
    accuracy_clearance: float = eqx.field(static=True)
    near_ratio: float = eqx.field(static=True)
    qbx_order: int = eqx.field(static=True)
    qbx_radius_factor: float = eqx.field(static=True)
    adaptive_plan: AdaptiveQuadraturePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["direct", "adaptive", "qbx"] = "direct",
        /,
        *,
        accuracy_clearance: float = 0.0,
        near_ratio: float = 4.0,
        qbx_order: int = 6,
        qbx_radius_factor: float = 0.5,
        adaptive_plan: AdaptiveQuadraturePlan | None = None,
    ):
        if method not in ("direct", "adaptive", "qbx"):
            raise ValueError("Unknown layer evaluator method.")
        clearance = float(accuracy_clearance)
        ratio = float(near_ratio)
        order = int(qbx_order)
        radius_factor = float(qbx_radius_factor)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("accuracy_clearance must be finite and nonnegative.")
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("near_ratio must be finite and positive.")
        if order < 1:
            raise ValueError("qbx_order must be positive.")
        if not math.isfinite(radius_factor) or radius_factor <= 0.0:
            raise ValueError("qbx_radius_factor must be finite and positive.")
        adaptive_plan_ = (
            AdaptiveQuadraturePlan() if adaptive_plan is None else adaptive_plan
        )
        if not isinstance(adaptive_plan_, AdaptiveQuadraturePlan):
            raise TypeError("adaptive_plan must be an AdaptiveQuadraturePlan.")
        self.method = method
        self.accuracy_clearance = clearance
        self.near_ratio = ratio
        self.qbx_order = order
        self.qbx_radius_factor = radius_factor
        self.adaptive_plan = adaptive_plan_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "layer-evaluation-plan-2d-v3",
                "method": method,
                "accuracy_clearance": clearance,
                "near_ratio": ratio,
                "qbx_order": order,
                "qbx_radius_factor": radius_factor,
                "adaptive_rule": type(adaptive_plan_.rule).__name__,
                "adaptive_max_intervals": adaptive_plan_.max_intervals,
                "adaptive_max_evaluations": adaptive_plan_.max_evaluations,
                "adaptive_absolute_tolerance": adaptive_plan_.absolute_tolerance,
                "adaptive_relative_tolerance": adaptive_plan_.relative_tolerance,
            }
        )


class LayerEvaluationReport(StrictModule, NonTrainableState):
    """Evaluator evidence kept separate from representation and discretization."""

    method: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    target_fingerprint: str = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    num_evaluations: Array
    error_estimate: Array
    error_kind: str = eqx.field(static=True)
    status: Array
    finite: Array
    accuracy_supported: Array
    near_panel_count: int = eqx.field(static=True)
    far_panel_count: int = eqx.field(static=True)
    failed_panel_count: int = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: LayerEvaluationPlan2D,
        representation_id: str,
        target_report: LayerPotentialTargetReport,
        num_evaluations: ArrayLike,
        error_estimate: ArrayLike,
        error_kind: str,
        finite: ArrayLike,
        accuracy_supported: ArrayLike,
        status: ArrayLike = 0,
        near_panel_count: int = 0,
        far_panel_count: int = 0,
        failed_panel_count: int = 0,
    ):
        if not isinstance(plan, LayerEvaluationPlan2D):
            raise TypeError("plan must be a LayerEvaluationPlan2D.")
        if not representation_id or not error_kind:
            raise ValueError("Layer evaluation identifiers must be nonempty.")
        self.method = plan.method
        self.plan_id = plan.plan_id
        self.representation_id = str(representation_id)
        self.target_fingerprint = target_report.target_fingerprint
        self.target_count = target_report.target_count
        self.num_evaluations = jnp.asarray(num_evaluations, dtype=jnp.int32)
        self.error_estimate = jnp.asarray(error_estimate)
        self.error_kind = str(error_kind)
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.accuracy_supported = jnp.asarray(
            accuracy_supported, dtype=bool
        ).reshape(())
        self.near_panel_count = int(near_panel_count)
        self.far_panel_count = int(far_panel_count)
        self.failed_panel_count = int(failed_panel_count)
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "layer-evaluation-report-v2",
                "plan_id": self.plan_id,
                "representation_id": self.representation_id,
                "target_fingerprint": self.target_fingerprint,
                "target_count": self.target_count,
                "num_evaluations": int(self.num_evaluations),
                "error_kind": self.error_kind,
                "status": int(self.status),
                "near_panel_count": self.near_panel_count,
                "far_panel_count": self.far_panel_count,
                "failed_panel_count": self.failed_panel_count,
            }
        )



class LayerEvaluationResult(StrictModule, NonTrainableState):
    """Values plus target and evaluator evidence for one explicit evaluation."""

    values: Array
    target_report: LayerPotentialTargetReport
    evaluation_report: LayerEvaluationReport

    def __init__(
        self,
        *,
        values: Array,
        target_report: LayerPotentialTargetReport,
        evaluation_report: LayerEvaluationReport,
    ):
        if not isinstance(target_report, LayerPotentialTargetReport):
            raise TypeError("target_report must be a LayerPotentialTargetReport.")
        if not isinstance(evaluation_report, LayerEvaluationReport):
            raise TypeError("evaluation_report must be a LayerEvaluationReport.")
        if (
            evaluation_report.target_fingerprint != target_report.target_fingerprint
            or evaluation_report.target_count != target_report.target_count
        ):
            raise ValueError("Evaluation and target reports describe different targets.")
        self.values = jnp.asarray(values)
        self.target_report = target_report
        self.evaluation_report = evaluation_report


def evaluate_layer_potential(
    potential: object,
    targets: ArrayLike,
    plan: LayerEvaluationPlan2D,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
) -> LayerEvaluationResult:
    """Evaluate a supported layer representation under an explicit plan."""
    from ._helmholtz2d import (
        HelmholtzCombinedField2D,
        HelmholtzLayerPotential2D,
    )
    from ._laplace2d import LaplaceLayerPotential2D

    supported = (
        LaplaceLayerPotential2D,
        HelmholtzLayerPotential2D,
        HelmholtzCombinedField2D,
    )
    if not isinstance(potential, supported):
        raise TypeError("Layer evaluation requires a supported 2D layer representation.")
    if not isinstance(plan, LayerEvaluationPlan2D):
        raise TypeError("plan must be a LayerEvaluationPlan2D.")
    targets_ = jnp.asarray(targets, dtype=float)
    single = targets_.ndim == 1
    if single:
        targets_ = targets_[None, :]
    if targets_.ndim != 2 or targets_.shape[1] != 2 or targets_.shape[0] == 0:
        raise ValueError("Layer targets must have shape (target_count, 2).")
    target_report = LayerPotentialTargetReport(
        targets_,
        potential.panelization,
        target_side=target_side,
        accuracy_clearance=plan.accuracy_clearance,
    )
    if bool(target_report.intersects_singular_support):
        node_collision = jnp.all(
            targets_[:, None, :] == potential.panelization.points[None, :, :],
            axis=-1,
        )
        if plan.method == "qbx":
            pass
        elif not (
            plan.method == "adaptive"
            and target_side == "boundary"
            and potential.kind == "single"
            and bool(jnp.all(jnp.any(node_collision, axis=1)))
        ):
            raise ValueError(
                "Boundary targets require QBX or adaptive single-layer "
                "self correction at source nodes."
            )
    elif not bool(target_report.pde_membership_valid):
        raise ValueError(
            "Layer evaluators require targets in the declared target side."
        )
    if plan.method == "direct":
        values = potential._evaluate_direct(targets_)
        finite = jnp.all(jnp.isfinite(values))
        status = jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32)
        error_estimate = jnp.asarray(jnp.inf)
        error_kind = "unestimated-direct"
        num_evaluations = (
            targets_.shape[0] * potential.panelization.node_count
        )
        near_panel_count = 0
        far_panel_count = targets_.shape[0] * potential.panelization.panel_count
        failed_panel_count = 0
        accuracy_supported = False
    elif plan.method == "adaptive":
        if not isinstance(potential, LaplaceLayerPotential2D):
            raise TypeError("Adaptive B1 evaluation currently supports Laplace layers.")
        from ._quadrature2d import (
            classify_panel_interactions_2d,
            evaluate_laplace_adaptive_2d,
        )

        interactions = classify_panel_interactions_2d(
            potential.panelization,
            targets_,
            near_ratio=plan.near_ratio,
        )
        adaptive = evaluate_laplace_adaptive_2d(
            potential,
            targets_,
            plan.adaptive_plan,
            interactions,
        )
        values = adaptive.values
        finite = jnp.all(jnp.isfinite(values))
        status = adaptive.status
        error_estimate = adaptive.error_estimate
        error_kind = "adaptive-embedded-rule"
        num_evaluations = adaptive.num_evaluations
        near_panel_count = adaptive.near_panel_count
        far_panel_count = adaptive.far_panel_count
        failed_panel_count = adaptive.failed_panel_count
        accuracy_supported = (
            finite & adaptive.accuracy_supported & target_report.accuracy_supported
        )
    else:
        from ._qbx2d import evaluate_qbx_2d

        qbx = evaluate_qbx_2d(
            potential,
            targets_,
            target_side=target_side,
            order=plan.qbx_order,
            radius_factor=plan.qbx_radius_factor,
            adaptive_plan=plan.adaptive_plan,
        )
        values = qbx.values
        finite = jnp.all(jnp.isfinite(values))
        status = qbx.status
        error_estimate = qbx.error_estimate
        error_kind = "qbx-coefficient-quadrature-and-truncation"
        num_evaluations = qbx.num_evaluations
        near_panel_count = targets_.shape[0]
        far_panel_count = 0
        failed_panel_count = int(not bool(qbx.accuracy_supported))
        accuracy_supported = (
            finite & qbx.accuracy_supported & target_report.accuracy_supported
        )
    evaluation_report = LayerEvaluationReport(
        plan=plan,
        representation_id=potential.representation_id,
        target_report=target_report,
        num_evaluations=num_evaluations,
        error_estimate=error_estimate,
        error_kind=error_kind,
        status=status,
        finite=finite,
        accuracy_supported=accuracy_supported,
        near_panel_count=near_panel_count,
        far_panel_count=far_panel_count,
        failed_panel_count=failed_panel_count,
    )
    result = LayerEvaluationResult(
        values=values,
        target_report=target_report,
        evaluation_report=evaluation_report,
    )
    if single:
        result = eqx.tree_at(lambda output: output.values, result, values[0])
    return result


__all__ = [
    "LayerEvaluationPlan2D",
    "LayerEvaluationReport",
    "LayerEvaluationResult",
    "evaluate_layer_potential",
]
