#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DifferentiationPolicy,
    estimate_operator_action_cost,
    FailurePolicy,
    FGMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare as prepare_linear,
    PreparedLinearSolve,
    solve as solve_linear,
)
from ..operators.integral.layer_potential._scalar_interfaces3d import (
    scalar_transmission_formulation_3d,
    ScalarCauchyTraceBundle3D,
    ScalarTransmissionAssemblyReport3D,
    ScalarTransmissionData3D,
    ScalarTransmissionFormulation3D,
    ScalarTransmissionMaterial3D,
    ScalarTransmissionSideConvention3D,
)


class PreparedScalarTransmission3D(StrictModule, NonTrainableState):
    """Prepared bounded two-domain scalar transmission solve.

    The exact four-by-four V/K block and its algebraic transpose live in
    ``formulation.operator``. The linear plan is matrix-free and uses no W,
    dense fallback, hidden resonance suppression, or unreported gauge.
    """

    formulation: ScalarTransmissionFormulation3D
    prepared_linear: PreparedLinearSolve
    linear_policy: LinearSolvePolicy
    resource_evidence: tuple[tuple[str, int], ...] = eqx.field(static=True)
    accuracy_evidence: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def assembly_report(self) -> ScalarTransmissionAssemblyReport3D:
        return self.formulation.report

    def right_hand_side(
        self,
        data: ScalarTransmissionData3D
        | tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
        /,
    ) -> tuple[Array, Array, Array, Array]:
        return self.formulation.right_hand_side(data)

    def manufactured_data(
        self,
        minus: ScalarCauchyTraceBundle3D,
        plus: ScalarCauchyTraceBundle3D,
        /,
    ) -> ScalarTransmissionData3D:
        """Apply the block to exact bundles and return manufactured RHS data."""

        if not isinstance(minus, ScalarCauchyTraceBundle3D) or not isinstance(
            plus, ScalarCauchyTraceBundle3D
        ):
            raise TypeError("minus and plus must be ScalarCauchyTraceBundle3D values.")
        if (
            minus.side != "minus"
            or plus.side != "plus"
            or minus.material_id != self.formulation.minus.material_id
            or plus.material_id != self.formulation.plus.material_id
        ):
            raise ValueError("Manufactured bundles do not match the prepared sides.")
        blocks = self.formulation.operator.mv(
            (
                minus.dirichlet,
                minus.normal_derivative,
                plus.dirichlet,
                plus.normal_derivative,
            )
        )
        return ScalarTransmissionData3D(*blocks)

    def solve(
        self,
        data: ScalarTransmissionData3D
        | tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
        /,
    ) -> ScalarTransmissionResult3D:
        return solve_scalar_transmission_3d(self, data)


class ScalarTransmissionResult3D(StrictModule, NonTrainableState):
    """Solved discrete Cauchy bundles and auditable block/continuity defects."""

    minus: ScalarCauchyTraceBundle3D
    plus: ScalarCauchyTraceBundle3D
    right_hand_side: tuple[Array, Array, Array, Array]
    residual_blocks: tuple[Array, Array, Array, Array]
    linear_result: LinearSolveResult
    relative_block_residual: Array
    dirichlet_continuity_defect: Array
    weighted_flux_continuity_defect: Array
    finite: Array
    valid: Array
    assembly_report: ScalarTransmissionAssemblyReport3D
    prepared_id: str = eqx.field(static=True)


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def prepare_scalar_transmission_3d(
    minus: ScalarTransmissionMaterial3D,
    plus: ScalarTransmissionMaterial3D,
    /,
    *,
    convention: ScalarTransmissionSideConvention3D | None = None,
    formulation: str = "two-sided-direct-Calderon-multitrace",
    linear: LinearSolvePolicy | None = None,
) -> PreparedScalarTransmission3D:
    """Prepare a direct two-sided Laplace/Yukawa/Helmholtz transmission solve."""

    transmission = scalar_transmission_formulation_3d(
        minus,
        plus,
        convention=convention,
        formulation=formulation,
    )
    policy = _default_linear_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError(
            "Scalar transmission currently requires differentiation mode 'none'."
        )
    if not bool(transmission.report.accuracy_supported):
        raise ValueError("Scalar transmission quadrature evidence is unsupported.")
    problem_id = canonical_fingerprint(
        {
            "kind": "scalar-transmission-linear-system-3d-v1",
            "formulation": transmission.formulation_id,
        }
    )
    prepared_linear = prepare_linear(
        LinearSystem(transmission.operator, problem_id=problem_id), policy
    )
    cost = estimate_operator_action_cost(transmission.operator)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-scalar-transmission-3d-v1",
            "formulation": transmission.formulation_id,
            "linear_plan": prepared_linear.plan.plan_id,
        }
    )
    resources = (
        ("block_unknowns", transmission.operator.source.size),
        (
            "calderon_preparation_workspace_bytes",
            transmission.report.preparation_workspace_bytes,
        ),
        ("resident_bytes", max(transmission.report.resident_bytes, cost.storage_bytes)),
        ("operator_action_workspace_bytes_per_rhs", cost.apply_workspace_bytes_per_rhs),
    )
    accuracy = (
        f"minus scalar Calderon report {transmission.report.assembly_report_ids[0]}",
        f"plus scalar Calderon report {transmission.report.assembly_report_ids[1]}",
        f"operator action cost exact={cost.exact}: {cost.reason}",
        transmission.report.resonance_evidence,
        "linear diagnostics and four block residuals are returned per solve",
        "continuum discretization error is not estimated",
    )
    return PreparedScalarTransmission3D(
        formulation=transmission,
        prepared_linear=prepared_linear,
        linear_policy=policy,
        resource_evidence=resources,
        accuracy_evidence=accuracy,
        prepared_id=prepared_id,
    )


def _squared_norm(blocks: tuple[Array, ...], /) -> Array:
    return sum(
        (jnp.vdot(block, block).real for block in blocks),
        start=jnp.asarray(0.0),
    )


def solve_scalar_transmission_3d(
    prepared: PreparedScalarTransmission3D,
    data: ScalarTransmissionData3D | tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    /,
) -> ScalarTransmissionResult3D:
    """Solve the prepared multitrace system and report physical continuity."""

    if not isinstance(prepared, PreparedScalarTransmission3D):
        raise TypeError("prepared must be PreparedScalarTransmission3D.")
    right_hand_side = prepared.right_hand_side(data)
    linear_result = solve_linear(prepared.prepared_linear, right_hand_side)
    solution = prepared.formulation.operator.source.validate(linear_result.value)
    minus, plus = prepared.formulation.bundles(solution)
    image = prepared.formulation.operator.mv(solution)
    residual = tuple(
        left - right for left, right in zip(image, right_hand_side, strict=True)
    )
    residual_norm = jnp.sqrt(_squared_norm(residual))
    right_norm = jnp.sqrt(_squared_norm(right_hand_side))
    relative = residual_norm / jnp.maximum(right_norm, jnp.asarray(1.0))
    dirichlet_defect = plus.dirichlet - minus.dirichlet - right_hand_side[2]
    flux_defect = (
        prepared.formulation.plus.flux_coefficient * plus.normal_derivative
        - prepared.formulation.minus.flux_coefficient * minus.normal_derivative
        - right_hand_side[3]
    )
    finite = (
        all(jnp.all(jnp.isfinite(block)) for block in solution)
        & all(jnp.all(jnp.isfinite(block)) for block in residual)
        & jnp.all(jnp.isfinite(dirichlet_defect))
        & jnp.all(jnp.isfinite(flux_defect))
        & jnp.isfinite(relative)
    )
    valid = (
        prepared.formulation.report.accuracy_supported
        & linear_result.successful
        & linear_result.diagnostics.finite
        & finite
    )
    return ScalarTransmissionResult3D(
        minus=minus,
        plus=plus,
        right_hand_side=right_hand_side,
        residual_blocks=residual,
        linear_result=linear_result,
        relative_block_residual=relative,
        dirichlet_continuity_defect=dirichlet_defect,
        weighted_flux_continuity_defect=flux_defect,
        finite=finite,
        valid=valid,
        assembly_report=prepared.formulation.report,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "PreparedScalarTransmission3D",
    "ScalarTransmissionResult3D",
    "prepare_scalar_transmission_3d",
    "solve_scalar_transmission_3d",
]
