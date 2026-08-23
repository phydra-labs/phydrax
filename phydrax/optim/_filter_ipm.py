#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._tree_math import validate_real_inexact_tree
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve as solve_linear,
)
from ._certificates import (
    certify_constrained_physical,
    reconcile_optimization_status,
)
from ._constrained_model import prepare_constrained_model
from ._iterative import (
    AbstractMinimizationMethod,
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._kkt import (
    factor_kkt,
    KKTFactorization,
    KKTSolveResult,
    plan_kkt,
    solve_factored_kkt,
)


def _max_abs(value):
    return jnp.max(jnp.abs(value), initial=0.0)


def _fraction_to_boundary(value, direction, fraction):
    ratios = jnp.where(direction < 0.0, -value / direction, jnp.inf)
    return jnp.minimum(1.0, fraction * jnp.min(ratios, initial=jnp.inf))


class _IPMDirection(eqx.Module):
    primal: jax.Array
    equality_dual: jax.Array
    inequality_dual: jax.Array
    slack: jax.Array
    kkt: KKTSolveResult


class FilterInteriorPointEvidence(eqx.Module):
    """KKT reuse and restoration evidence for one interior-point solve."""

    kkt_plan_id: str = eqx.field(static=True)
    kkt_form: str = eqx.field(static=True)
    kkt_factorizations: jax.Array
    kkt_rhs_solves: jax.Array
    kkt_factorization_reuses: jax.Array
    restoration_steps: jax.Array
    final_barrier: jax.Array


def _condensed_hessian(
    hessian,
    inequality_jacobian,
    slack,
    dual,
):
    inverse_slack = 1.0 / jnp.maximum(slack, 1e-30)
    diagonal = dual * inverse_slack
    return hessian + jnp.conj(inequality_jacobian.T) @ (
        diagonal[:, None] * inequality_jacobian
    )


def _condensed_kkt_direction(
    factorization: KKTFactorization,
    inequality_jacobian,
    slack,
    dual,
    dual_residual,
    equality_residual,
    inequality_residual,
    complementarity_residual,
):
    inverse_slack = 1.0 / jnp.maximum(slack, 1e-30)
    correction = inverse_slack * (complementarity_residual + dual * inequality_residual)
    adjusted_dual = dual_residual + jnp.conj(inequality_jacobian.T) @ correction
    kkt = solve_factored_kkt(
        factorization,
        adjusted_dual,
        equality_residual,
    )
    primal = kkt.primal_step
    equality_dual = kkt.dual_step
    slack_direction = inequality_jacobian @ primal + inequality_residual
    inequality_dual = -inverse_slack * (complementarity_residual + dual * slack_direction)
    return _IPMDirection(
        primal,
        equality_dual,
        inequality_dual,
        slack_direction,
        kkt,
    )


class FilterInteriorPoint(AbstractMinimizationMethod):
    """Dense primal-dual filter interior-point method with restoration."""

    fraction_to_boundary: float = eqx.field(static=True)
    minimum_barrier: float = eqx.field(static=True)
    filter_margin: float = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    maximum_restoration_steps: int = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        fraction_to_boundary: float = 0.995,
        minimum_barrier: float = 1e-10,
        filter_margin: float = 1e-4,
        maximum_line_search_steps: int = 24,
        maximum_restoration_steps: int = 3,
        max_dense_dimension: int = 512,
        linear: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        values = tuple(
            float(value)
            for value in (fraction_to_boundary, minimum_barrier, filter_margin)
        )
        search = int(maximum_line_search_steps)
        restoration = int(maximum_restoration_steps)
        dimension = int(max_dense_dimension)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Interior-point controls must be finite and positive.")
        if values[0] >= 1.0 or values[2] >= 1.0:
            raise ValueError("Fraction-to-boundary and filter margin must be below one.")
        if search < 1 or restoration < 1 or dimension < 1:
            raise ValueError("Interior-point step and dimension limits must be positive.")
        linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.fraction_to_boundary, self.minimum_barrier, self.filter_margin = values
        self.maximum_line_search_steps = search
        self.maximum_restoration_steps = restoration
        self.max_dense_dimension = dimension
        self.linear = linear_
        self.precision = precision_

    @property
    def method_id(self):
        return "filter-interior-point"

    @property
    def capabilities(self):
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=True,
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        if not isinstance(problem, MinimizationProblem):
            raise TypeError("problem must be MinimizationProblem.")
        self.precision.validate_tolerance(termination.absolute_optimality)
        parameters = self.precision.state(
            validate_real_inexact_tree(initial_parameters, name="parameters")
        )
        model = prepare_constrained_model(problem, parameters, args=args)
        if model.template_coordinates.size > self.max_dense_dimension:
            raise ValueError("FilterInteriorPoint exceeds max_dense_dimension.")
        evaluation = model.evaluate(parameters, args)
        me = evaluation.equalities.size
        mi = evaluation.inequality_slacks.size
        initial_equality_jacobian = evaluation.constraint_jacobian[model.equality_indices]
        jacobian_density = (
            0.0
            if initial_equality_jacobian.size == 0
            else float(
                jnp.count_nonzero(initial_equality_jacobian)
                / initial_equality_jacobian.size
            )
        )
        kkt_plan = plan_kkt(
            evaluation.coordinates.size,
            evaluation.equalities.size,
            jacobian_density=jacobian_density,
        )
        equality_dual = jnp.zeros((me,), dtype=evaluation.coordinates.dtype)
        slack = jnp.maximum(evaluation.inequality_slacks, 1.0)
        inequality_dual = jnp.ones((mi,), dtype=evaluation.coordinates.dtype)
        barrier = jnp.maximum(
            self.minimum_barrier,
            jnp.vdot(slack, inequality_dual).real / max(mi, 1),
        )
        filter_pairs = []
        accepted = rejected = evaluations = gradients = constraints = linear_solves = 0
        iterations = restorations = factorizations = factorization_reuses = 0
        step_norm = 0.0
        status = int(OptimizationStatus.ITERATING)
        initial_optimality = None
        auxiliary = problem.value(parameters, args)[1]
        while (
            status == int(OptimizationStatus.ITERATING)
            and iterations < termination.maximum_steps
        ):
            evaluation = model.evaluate(parameters, args)
            raw_jacobian = evaluation.constraint_jacobian
            equality_jacobian = raw_jacobian[model.equality_indices]
            lower_jacobian = raw_jacobian[model.lower_indices]
            upper_jacobian = -raw_jacobian[model.upper_indices]
            inequality_jacobian = jnp.concatenate(
                [lower_jacobian, upper_jacobian], axis=0
            )
            lower_count = evaluation.lower_slacks.size
            lower_dual = inequality_dual[:lower_count]
            upper_dual = inequality_dual[lower_count:]
            hessian = model.lagrangian_hessian(
                parameters,
                equality_dual,
                lower_dual,
                upper_dual,
                args,
            )
            dual_residual = (
                evaluation.gradient
                + jnp.conj(equality_jacobian.T) @ equality_dual
                - jnp.conj(inequality_jacobian.T) @ inequality_dual
            )
            equality_residual = evaluation.equalities
            inequality_residual = evaluation.inequality_slacks - slack
            complementarity_residual = slack * inequality_dual
            primal = jnp.maximum(
                _max_abs(equality_residual), _max_abs(inequality_residual)
            )
            dual = _max_abs(dual_residual)
            complementarity = _max_abs(complementarity_residual)
            optimality = jnp.maximum(primal, jnp.maximum(dual, complementarity))
            if initial_optimality is None:
                initial_optimality = optimality
            if float(optimality) <= float(
                termination.optimality_threshold(initial_optimality)
            ):
                status = int(OptimizationStatus.SUCCESS)
                break
            condensed_hessian = _condensed_hessian(
                hessian,
                inequality_jacobian,
                slack,
                inequality_dual,
            )
            kkt_factorization = factor_kkt(
                condensed_hessian,
                equality_jacobian,
                kkt_plan,
                precision=self.precision,
            )
            factorizations += 1
            affine = _condensed_kkt_direction(
                kkt_factorization,
                inequality_jacobian,
                slack,
                inequality_dual,
                dual_residual,
                equality_residual,
                inequality_residual,
                complementarity_residual,
            )
            linear_solves += 1
            if not bool(affine.kkt.finite & affine.kkt.inertia_matches):
                status = int(OptimizationStatus.LINEAR_SOLVE_FAILED)
                break
            alpha_affine = jnp.minimum(
                _fraction_to_boundary(slack, affine.slack, 1.0),
                _fraction_to_boundary(
                    inequality_dual,
                    affine.inequality_dual,
                    1.0,
                ),
            )
            affine_mu = jnp.vdot(
                slack + alpha_affine * affine.slack,
                inequality_dual + alpha_affine * affine.inequality_dual,
            ).real / max(mi, 1)
            sigma = jnp.clip(
                (affine_mu / jnp.maximum(barrier, 1e-30)) ** 3,
                0.0,
                1.0,
            )
            corrected_center = (
                complementarity_residual
                + affine.slack * affine.inequality_dual
                - sigma * barrier
            )
            direction = _condensed_kkt_direction(
                kkt_factorization,
                inequality_jacobian,
                slack,
                inequality_dual,
                dual_residual,
                equality_residual,
                inequality_residual,
                corrected_center,
            )
            linear_solves += 1
            factorization_reuses += 1
            if not bool(direction.kkt.finite & direction.kkt.inertia_matches):
                status = int(OptimizationStatus.LINEAR_SOLVE_FAILED)
                break
            alpha = jnp.minimum(
                _fraction_to_boundary(
                    slack,
                    direction.slack,
                    self.fraction_to_boundary,
                ),
                _fraction_to_boundary(
                    inequality_dual,
                    direction.inequality_dual,
                    self.fraction_to_boundary,
                ),
            )
            current_pair = (float(evaluation.objective), float(primal))
            filter_pairs.append(current_pair)
            accepted_trial = False
            for _ in range(self.maximum_line_search_steps):
                candidate_coordinates = evaluation.coordinates + alpha * direction.primal
                candidate = model.unflatten(candidate_coordinates)
                candidate_evaluation = model.evaluate(candidate, args)
                candidate_slack = slack + alpha * direction.slack
                candidate_dual = inequality_dual + alpha * direction.inequality_dual
                candidate_equality_dual = equality_dual + alpha * direction.equality_dual
                candidate_ineq_residual = (
                    candidate_evaluation.inequality_slacks - candidate_slack
                )
                candidate_primal = jnp.maximum(
                    _max_abs(candidate_evaluation.equalities),
                    _max_abs(candidate_ineq_residual),
                )
                dominated = any(
                    float(candidate_evaluation.objective)
                    >= objective_value - self.filter_margin * violation
                    and float(candidate_primal) >= (1.0 - self.filter_margin) * violation
                    for objective_value, violation in filter_pairs
                )
                finite = bool(
                    candidate_evaluation.finite
                    & jnp.all(candidate_slack > 0.0)
                    & jnp.all(candidate_dual > 0.0)
                )
                if finite and not dominated:
                    parameters = candidate
                    slack = candidate_slack
                    inequality_dual = candidate_dual
                    equality_dual = candidate_equality_dual
                    evaluation = candidate_evaluation
                    accepted_trial = True
                    accepted += 1
                    break
                alpha *= 0.5
                rejected += 1
            evaluations += 1
            gradients += 1
            constraints += 1
            iterations += 1
            step_norm = float(jnp.linalg.norm(alpha * direction.primal))
            if accepted_trial:
                barrier = jnp.maximum(
                    self.minimum_barrier,
                    jnp.vdot(slack, inequality_dual).real / max(mi, 1),
                )
                filter_pairs = [
                    pair
                    for pair in filter_pairs
                    if not (
                        pair[0] >= float(evaluation.objective)
                        and pair[1] >= float(evaluation.primal_feasibility)
                    )
                ]
            else:
                restorations += 1
                constraint_matrix = jnp.concatenate(
                    [equality_jacobian, inequality_jacobian], axis=0
                )
                restoration_rhs = -jnp.concatenate(
                    [equality_residual, inequality_residual]
                )
                restoration_direction = self.precision.direction(
                    solve_linear(
                        LeastSquaresProblem(
                            DenseLinearOperator(
                                self.precision.accumulation(constraint_matrix)
                            )
                        ),
                        self.precision.accumulation(restoration_rhs),
                        policy=self.precision.bind_linear(self.linear),
                    ).value
                )
                parameters = model.unflatten(
                    jnp.asarray(
                        evaluation.coordinates + 0.5 * restoration_direction,
                        dtype=evaluation.coordinates.dtype,
                    )
                )
                restored = model.evaluate(parameters, args)
                slack = jnp.maximum(restored.inequality_slacks, 1e-8)
                if restorations >= self.maximum_restoration_steps:
                    status = int(OptimizationStatus.RESTORATION_FAILED)
            if step_norm <= float(
                termination.step_threshold(jnp.linalg.norm(evaluation.coordinates))
            ):
                status = int(OptimizationStatus.STAGNATION)
        if status == int(OptimizationStatus.ITERATING):
            status = int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
        final = model.evaluate(parameters, args)
        raw_jacobian = final.constraint_jacobian
        equality_jacobian = raw_jacobian[model.equality_indices]
        inequality_jacobian = jnp.concatenate(
            [raw_jacobian[model.lower_indices], -raw_jacobian[model.upper_indices]],
            axis=0,
        )
        dual_residual = (
            final.gradient
            + jnp.conj(equality_jacobian.T) @ equality_dual
            - jnp.conj(inequality_jacobian.T) @ inequality_dual
        )
        multiplier_violation = jnp.max(
            jnp.maximum(-inequality_dual, 0.0),
            initial=0.0,
        )
        final_dual = jnp.maximum(
            _max_abs(dual_residual),
            multiplier_violation,
        )
        physical_slacks = jnp.maximum(final.inequality_slacks, 0.0)
        final_complementarity = _max_abs(physical_slacks * inequality_dual)
        active_tolerance = jnp.sqrt(
            jnp.asarray(
                termination.absolute_optimality,
                dtype=final.objective.dtype,
            )
        )
        canonical = ConstrainedOptimalityCertificate(
            equality_multipliers=equality_dual,
            inequality_multipliers=inequality_dual,
            slacks=physical_slacks,
            active_mask=final.inequality_slacks <= active_tolerance,
            stationarity_residual=model.unflatten(dual_residual),
            primal_feasibility=final.primal_feasibility,
            dual_feasibility=final_dual,
            complementarity=final_complementarity,
            equality_sources=model.equality_sources,
            inequality_sources=model.inequality_sources,
            precision_evidence=self.precision.evidence_for(
                parameters,
                model.unflatten(dual_residual),
            ),
        )
        certificate = certify_constrained_physical(
            model,
            parameters,
            canonical,
            termination.absolute_optimality,
            kind="active-kkt",
            args=args,
            linear=self.linear,
            precision=self.precision,
        )
        status_evidence = reconcile_optimization_status(
            status,
            certificate,
            allow_certificate_promotion=True,
        )
        evidence = FilterInteriorPointEvidence(
            kkt_plan.plan_id,
            kkt_plan.form,
            jnp.asarray(factorizations, dtype=jnp.int32),
            jnp.asarray(linear_solves, dtype=jnp.int32),
            jnp.asarray(factorization_reuses, dtype=jnp.int32),
            jnp.asarray(restorations, dtype=jnp.int32),
            jnp.asarray(barrier),
        )
        diagnostics = OptimizationDiagnostics(
            iterations=iterations,
            accepted_steps=accepted,
            rejected_steps=rejected,
            objective_evaluations=evaluations + 2 + certificate.evaluation_work,
            gradient_evaluations=gradients + 1 + certificate.evaluation_work,
            constraint_evaluations=constraints + 1 + certificate.evaluation_work,
            linear_solves=linear_solves,
            linear_iterations=linear_solves,
            globalization_evaluations=accepted + rejected,
            initial_optimality_norm=(
                certificate.optimality_norm
                if initial_optimality is None
                else initial_optimality
            ),
            final_optimality_norm=certificate.optimality_norm,
            final_step_norm=step_norm,
            accepted_step_size=1.0 if accepted else 0.0,
            damping=barrier,
            primal_feasibility=certificate.primal_feasibility,
            dual_feasibility=certificate.dual_feasibility,
            complementarity=certificate.complementarity,
            active_constraints=jnp.sum(canonical.active_mask),
        )
        provenance = OptimizationProvenance(
            problem_id=problem.problem_id,
            method=self.method_id,
            backend="phydrax-native",
            globalization="objective-feasibility-filter",
            matrix_free=False,
            implicit_differentiation=True,
            precision_policy_id=self.precision.policy_id,
            notes=(
                f"restorations={restorations};"
                f"kkt-plan={kkt_plan.plan_id};"
                f"internal-status={status}"
            ),
        )
        objective, auxiliary = problem.value(parameters, args)
        output_parameters = jax.tree.map(self.precision.output, parameters)
        precision_evidence = self.precision.evidence_for(
            parameters,
            model.unflatten(dual_residual),
            children={
                "canonical-kkt": canonical.precision_evidence,
                "physical-certificate": certificate.precision_evidence,
            },
            output_value=output_parameters,
        )
        return MinimizationResult(
            output_parameters,
            self.precision.output(objective),
            auxiliary,
            status_evidence.public_status,
            diagnostics,
            provenance,
            certificate=canonical,
            optimality_certificate=certificate,
            status_evidence=status_evidence,
            method_evidence=evidence,
            precision_evidence=precision_evidence,
        )


__all__ = ["FilterInteriorPoint", "FilterInteriorPointEvidence"]
