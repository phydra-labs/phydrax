#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseSVD,
    DifferentiationPolicy,
    GMRES,
    JacobianLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    MaterializationPolicy,
    prepare as prepare_linear,
    prepare_linearization,
    PreparedLinearSolve,
    RankPolicy,
    RHSLayout,
    solve as solve_linear,
    TolerancePolicy,
)
from ...nonlinear import (
    implicit_root_output,
    SensitivityEvidence,
    SensitivityPolicy,
    SensitivityStatus,
)
from ...optim import (
    AbstractLeastSquaresTrialPolicy,
    least_squares,
    LeastSquaresResult,
    LeastSquaresTrialResult,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)
from ...typing import checked


class DesignTrialPolicy(AbstractLeastSquaresTrialPolicy):
    """Reject inadmissible trials and require a bounded weighted residual.

    The native LM direction and Jacobian image are unchanged. Bounds describe
    admissibility, not a projection or a repair of the proposed geometry.
    """

    lower: Array
    upper: Array
    strict_lower: Array
    tolerance: float = eqx.field(static=True)
    validity: Callable[[Array], Array] | None = None

    def admissible(self, vector: Array, /) -> Array:
        lower_valid = jnp.where(
            self.strict_lower, vector > self.lower, vector >= self.lower
        )
        bounds_valid = jnp.all(
            jnp.isfinite(vector) & lower_valid & (vector <= self.upper)
        )
        if self.validity is None:
            return bounds_valid
        dynamic_validity, static_validity = eqx.partition(self.validity, eqx.is_array)

        def evaluate(
            operand: tuple[Array, Callable[[Array], Array] | None],
        ) -> Array:
            current, dynamic_callback = operand
            callback = eqx.combine(dynamic_callback, static_validity)
            return (
                jnp.asarray(True, dtype=jnp.bool_)
                if callback is None
                else callback(current)
            )

        # Callback arrays are conditional operands; only metadata is captured.
        return jax.lax.cond(
            bounds_valid,
            evaluate,
            lambda _: jnp.asarray(False, dtype=jnp.bool_),
            (vector, dynamic_validity),
        )

    def __call__(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: JacobianLinearOperator,
        remaining_model_work: Array,
        args: None,
    ) -> LeastSquaresTrialResult:
        del residual, remaining_model_work, args
        return LeastSquaresTrialResult(
            direction=direction,
            scale=jnp.asarray(1.0, dtype=parameters.dtype),
            model_image=jacobian.mv(direction),
            valid=self.admissible(parameters + direction),
            jvp_actions=jnp.asarray(1, dtype=jnp.int32),
            vjp_actions=jnp.asarray(0, dtype=jnp.int32),
            model_work_units=jnp.asarray(0, dtype=jnp.int64),
            model_visits=jnp.asarray(0, dtype=jnp.int64),
            resource_refused=jnp.asarray(False, dtype=jnp.bool_),
        )

    def termination_valid(self, residual: Array, args: None) -> Array:
        del args
        return jnp.all(jnp.isfinite(residual)) & (
            jnp.linalg.norm(residual) <= self.tolerance
        )


@checked
def solve_design_residual(
    residual: Callable[[Array, None], Array],
    initial: Array,
    policy: DesignTrialPolicy,
    /,
    *,
    max_iterations: int,
    damping: float,
    maximum_trials: int,
    problem_id: str,
) -> LeastSquaresResult:
    """Solve an admissible design residual with native LM and linear solves.

    Seed and conditioned residual data carry no iteration derivatives. Sketch
    sensitivities attach separately to the accepted least-squares endpoint.
    """
    initial = eqx.error_if(
        initial,
        ~policy.admissible(initial),
        "design.solve.initial must satisfy the declared admissibility policy; got inadmissible state",
    )
    result = least_squares(
        NonlinearLeastSquaresProblem(
            _conditioned_inputs(residual),
            problem_id=problem_id,
            trial_policy=_conditioned_inputs(policy),
            trial_policy_id=f"{problem_id}/admissibility",
            trial_model_work_limit=jnp.asarray(0, dtype=jnp.int64),
        ),
        jax.lax.stop_gradient(initial),
        method=LevenbergMarquardt(
            initial_damping=damping,
            minimum_damping=min(damping, 1e-12),
            maximum_damping=max(damping, 1e12),
            maximum_trials=maximum_trials,
        ),
        termination=OptimizationTermination(
            absolute_optimality=policy.tolerance,
            relative_optimality=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=max_iterations,
        ),
    )
    return result


def _conditioned_inputs[T](tree: T, /) -> T:
    """Keep numerical inputs visible while excluding optimizer-path tangents."""
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


class _ResidualCoordinates(StrictModule):
    residual: Callable[[Array, None], Array]

    def __call__(self, point: Array, /) -> Array:
        return self.residual(point, None)


class _LeastSquaresStationarity(StrictModule):
    residual: Callable[[Array, None], Array]

    def __call__(self, point: Array, /) -> Array:
        linearization = prepare_linearization(_ResidualCoordinates(self.residual), point)
        values = jnp.asarray(linearization.primal, dtype=point.dtype)
        return jnp.asarray(linearization.vjp(values), dtype=point.dtype)


def _stationarity_output(
    point: Array, residual: Callable[[Array, None], Array], /
) -> tuple[Array, None]:
    """Return square stationarity equations for the retained LS endpoint."""
    return _LeastSquaresStationarity(residual)(point), None


def _prepare_endpoint_derivative(
    operator: AbstractLinearOperator, /
) -> PreparedLinearSolve:
    """Prepare matrix-free tangent and adjoint actions at an admitted endpoint."""
    return prepare_linear(
        LinearSystem(operator),
        LinearSolvePolicy(
            GMRES(restart=min(64, operator.source.size)),
            tolerance=TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=512),
            differentiation=DifferentiationPolicy("none"),
        ),
        rhs_layout=RHSLayout((1,)),
    )


class _SketchDerivativeAdmission(StrictModule):
    evidence: SensitivityEvidence
    rank: Array


def _sketch_derivative_admission(
    residual: Callable[[Array, None], Array],
    point: Array,
    receipt: LeastSquaresResult,
    tolerance: float,
    /,
) -> _SketchDerivativeAdmission:
    """Evaluate native rank and conditioning without changing the primal result.

    The endpoint is the gradient of one half of the weighted residual squares.
    Dense rank preparation is bounded to 64 coordinates and one MiB workspace;
    the derivative actions themselves remain matrix-free.
    """
    primal_norm = jnp.linalg.norm(jnp.asarray(receipt.residual, dtype=point.dtype))
    primal_valid = (
        receipt.successful & jnp.isfinite(primal_norm) & (primal_norm <= tolerance)
    )
    materialization = MaterializationPolicy(
        max_entries=64 * 64,
        max_bytes=64 * 64 * jnp.dtype(point.dtype).itemsize,
        max_workspace_bytes=1024 * 1024,
    )
    policy = SensitivityPolicy(
        "implicit-forward",
        primal_residual_tolerance=tolerance,
        linear=LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(require_full_rank=True),
            materialization=materialization,
            differentiation=DifferentiationPolicy("none"),
        ),
    )
    if point.size > 64:
        status = jnp.where(
            primal_valid,
            int(SensitivityStatus.UNSUPPORTED),
            int(SensitivityStatus.PRIMAL_FAILED),
        ).astype(jnp.int32)
        rank = jnp.asarray(-1, dtype=jnp.int32)
        condition = jnp.asarray(jnp.nan, dtype=point.dtype)
        linear_status = jnp.asarray(
            int(LinearSolveStatus.CAPABILITY_REJECTED), dtype=jnp.int32
        )
        finite = jnp.all(jnp.isfinite(point))
        linear_residual = jnp.asarray(jnp.nan, dtype=point.dtype)
    else:

        def evaluate(_: None) -> tuple[Array, Array, Array, Array, Array, Array]:
            linearization = prepare_linearization(
                _LeastSquaresStationarity(_conditioned_inputs(residual)), point
            )
            operator = JacobianLinearOperator(linearization)
            prepared = prepare_linear(
                LeastSquaresProblem(operator), policy.linear, rhs_layout=RHSLayout((1,))
            )
            proof = solve_linear(prepared, jnp.zeros((point.size, 1), dtype=point.dtype))
            rank = jnp.min(proof.diagnostics.rank)
            condition = jnp.max(proof.diagnostics.condition_estimate)
            finite = jnp.all(proof.diagnostics.finite)
            regular = (
                jnp.asarray(False, dtype=jnp.bool_)
                if proof.derivative_regular is None
                else jnp.all(proof.derivative_regular)
            )
            status = jnp.where(
                (rank >= 0) & (rank < point.size),
                int(SensitivityStatus.SINGULAR),
                jnp.where(
                    rank < 0,
                    int(SensitivityStatus.UNSUPPORTED),
                    jnp.where(
                        ~finite,
                        int(SensitivityStatus.NONFINITE),
                        jnp.where(
                            ~regular,
                            int(SensitivityStatus.UNSUPPORTED),
                            jnp.where(
                                ~jnp.isfinite(condition)
                                | (condition > policy.condition_limit),
                                int(SensitivityStatus.CONDITION_LIMIT),
                                jnp.where(
                                    jnp.all(proof.successful),
                                    int(SensitivityStatus.SUCCESS),
                                    int(SensitivityStatus.LINEAR_FAILED),
                                ),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            return (
                status,
                rank,
                condition,
                jnp.max(proof.status),
                finite,
                jnp.max(proof.diagnostics.residual_norm),
            )

        def refused(_: None) -> tuple[Array, Array, Array, Array, Array, Array]:
            return (
                jnp.asarray(int(SensitivityStatus.PRIMAL_FAILED), dtype=jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=point.dtype),
                jnp.asarray(int(LinearSolveStatus.CAPABILITY_REJECTED), dtype=jnp.int32),
                jnp.all(jnp.isfinite(point)),
                jnp.asarray(jnp.nan, dtype=point.dtype),
            )

        status, rank, condition, linear_status, finite, linear_residual = jax.lax.cond(
            primal_valid, evaluate, refused, operand=None
        )
    evidence = SensitivityEvidence(
        status,
        condition,
        linear_residual,
        finite,
        mode="implicit-forward",
        primal_residual_norm=primal_norm,
        primal_residual_tolerance=tolerance,
        primal_valid=primal_valid,
        linear_status=linear_status,
    )
    return _SketchDerivativeAdmission(_conditioned_inputs(evidence), rank)


@checked
def attach_sketch_sensitivity(
    residual: Callable[[Array, None], Array],
    receipt: LeastSquaresResult,
    tolerance: float,
    /,
) -> tuple[Array, SensitivityEvidence, Array]:
    """Attach native IFT actions to an existing weighted LS solution.

    Rank-deficient and uncertified endpoints keep their primal values and report
    invalid derivative evidence; native derivative actions return NaN.
    """
    point = jax.lax.stop_gradient(jnp.asarray(receipt.parameters, dtype=jnp.float64))
    admission = _sketch_derivative_admission(residual, point, receipt, tolerance)
    attached, _ = implicit_root_output(
        point,
        residual,
        None,
        admission.evidence.successful,
        residual_and_output=_stationarity_output,
        root_space=ArraySpace(
            point.shape,
            dtype=point.dtype,
            space_id=f"{receipt.provenance.problem_id}/stationarity",
        ),
        prepare_linear_solve=_prepare_endpoint_derivative,
        derivative_failure="status",
    )
    return attached, admission.evidence, admission.rank
