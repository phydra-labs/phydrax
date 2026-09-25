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

from phydrax._strict import StrictModule

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
    OptimizationCertificate,
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


class _IPMDirection(StrictModule):
    primal: jax.Array
    equality_dual: jax.Array
    inequality_dual: jax.Array
    slack: jax.Array
    kkt: KKTSolveResult


class FilterInteriorPointEvidence(StrictModule):
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
        accepted = rejected = linear_solves = 0
        evaluations = gradients = constraints = 1
        globalization_evaluations = 0
        iterations = restorations = factorizations = factorization_reuses = 0
        step_norm = 0.0
        status = int(OptimizationStatus.ITERATING)
        initial_optimality = None
        while (
            status == int(OptimizationStatus.ITERATING)
            and iterations < termination.maximum_steps
            and (
                termination.maximum_evaluations is None
                or evaluations + self.maximum_line_search_steps
                <= termination.maximum_evaluations
            )
        ):
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
            line_search_alphas = alpha * 0.5 ** jnp.arange(
                self.maximum_line_search_steps,
                dtype=alpha.dtype,
            )

            def evaluate_trial(step_size):
                candidate_coordinates = (
                    evaluation.coordinates + step_size * direction.primal
                )
                candidate = model.unflatten(candidate_coordinates)
                candidate_evaluation = model.evaluate(candidate, args)
                candidate_slack = slack + step_size * direction.slack
                candidate_dual = inequality_dual + step_size * direction.inequality_dual
                candidate_equality_dual = (
                    equality_dual + step_size * direction.equality_dual
                )
                candidate_ineq_residual = (
                    candidate_evaluation.inequality_slacks - candidate_slack
                )
                candidate_primal = jnp.maximum(
                    _max_abs(candidate_evaluation.equalities),
                    _max_abs(candidate_ineq_residual),
                )
                return (
                    candidate,
                    candidate_evaluation,
                    candidate_slack,
                    candidate_dual,
                    candidate_equality_dual,
                    candidate_primal,
                )

            (
                candidate_parameters,
                candidate_evaluations,
                candidate_slacks,
                candidate_duals,
                candidate_equality_duals,
                candidate_primals,
            ) = eqx.filter_vmap(evaluate_trial)(line_search_alphas)
            filter_objectives = jnp.asarray(
                tuple(value for value, _ in filter_pairs),
                dtype=evaluation.objective.dtype,
            )
            filter_violations = jnp.asarray(
                tuple(value for _, value in filter_pairs),
                dtype=candidate_primals.dtype,
            )
            dominated = jnp.any(
                (
                    candidate_evaluations.objective[:, None]
                    >= filter_objectives[None, :]
                    - self.filter_margin * filter_violations[None, :]
                )
                & (
                    candidate_primals[:, None]
                    >= (1.0 - self.filter_margin) * filter_violations[None, :]
                ),
                axis=-1,
            )
            # A trial that leaves the primal iterate in place only recenters the
            # slacks and multipliers: its (objective, violation) pair equals the
            # current filter entry, so the filter cannot rank it. Admit it when it
            # does not raise the primal violation.
            centering = (
                line_search_alphas * jnp.linalg.norm(direction.primal)
                <= termination.step_threshold(jnp.linalg.norm(evaluation.coordinates))
            ) & (
                candidate_primals
                <= jnp.maximum(
                    primal,
                    termination.optimality_threshold(initial_optimality),
                )
            )
            acceptable = (
                candidate_evaluations.finite
                & jnp.all(candidate_slacks > 0.0, axis=-1)
                & jnp.all(candidate_duals > 0.0, axis=-1)
                & (~dominated | centering)
            )
            accepted_trial = bool(jnp.any(acceptable))
            selected = int(jnp.argmax(acceptable))
            selected_alpha = jnp.where(
                jnp.any(acceptable),
                line_search_alphas[selected],
                line_search_alphas[-1] * 0.5,
            )
            rejected += selected if accepted_trial else self.maximum_line_search_steps
            if accepted_trial:
                parameters = jax.tree.map(
                    lambda value: value[selected],
                    candidate_parameters,
                )
                evaluation = jax.tree.map(
                    lambda value: value[selected],
                    candidate_evaluations,
                )
                slack = candidate_slacks[selected]
                inequality_dual = candidate_duals[selected]
                equality_dual = candidate_equality_duals[selected]
                accepted += 1
            alpha = selected_alpha
            trial_evaluations = self.maximum_line_search_steps
            evaluations += trial_evaluations
            gradients += trial_evaluations
            constraints += trial_evaluations
            globalization_evaluations += trial_evaluations
            iterations += 1
            step_norm = float(jnp.linalg.norm(alpha * direction.primal))
            iterate_step_norm = jnp.linalg.norm(
                alpha
                * jnp.concatenate(
                    [
                        direction.primal,
                        direction.slack,
                        direction.inequality_dual,
                        direction.equality_dual,
                    ]
                )
            )
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
                if (
                    termination.maximum_evaluations is not None
                    and evaluations >= termination.maximum_evaluations
                ):
                    status = int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
                else:
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
                    evaluation = restored
                    evaluations += 1
                    gradients += 1
                    constraints += 1
                    slack = jnp.maximum(restored.inequality_slacks, 1e-8)
                    if restorations >= self.maximum_restoration_steps:
                        status = int(OptimizationStatus.RESTORATION_FAILED)
            if status == int(OptimizationStatus.ITERATING) and float(
                iterate_step_norm
            ) <= float(
                termination.step_threshold(
                    jnp.linalg.norm(
                        jnp.concatenate(
                            [
                                evaluation.coordinates,
                                slack,
                                inequality_dual,
                                equality_dual,
                            ]
                        )
                    )
                )
            ):
                status = int(OptimizationStatus.STAGNATION)
        if status == int(OptimizationStatus.ITERATING):
            status = (
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
                if (
                    termination.maximum_evaluations is not None
                    and evaluations + self.maximum_line_search_steps
                    > termination.maximum_evaluations
                )
                else int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
            )
        final = evaluation
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
        budget_reached = status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
        if not budget_reached and (
            termination.maximum_evaluations is None
            or evaluations < termination.maximum_evaluations
        ):
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
        else:
            canonical_optimality = jnp.maximum(
                canonical.primal_feasibility,
                jnp.maximum(
                    canonical.dual_feasibility,
                    canonical.complementarity,
                ),
            )
            finite = (
                final.finite
                & jnp.isfinite(canonical_optimality)
                & jnp.isfinite(final.objective)
            )
            certificate = OptimizationCertificate(
                kind="active-kkt",
                tolerance=termination.absolute_optimality,
                optimality_norm=canonical_optimality,
                primal_feasibility=canonical.primal_feasibility,
                dual_feasibility=canonical.dual_feasibility,
                complementarity=canonical.complementarity,
                projected_stationarity=canonical.dual_feasibility,
                finite=finite,
                regular=True,
                certified=finite
                & (canonical_optimality <= termination.absolute_optimality),
                evaluation_work=0,
                certificate_id=f"{problem.problem_id}/budgeted-active-kkt",
                precision_evidence=canonical.precision_evidence,
            )
        status_evidence = reconcile_optimization_status(
            status,
            certificate,
            allow_certificate_promotion=False,
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
        if not budget_reached and (
            termination.maximum_evaluations is None
            or evaluations + certificate.evaluation_work < termination.maximum_evaluations
        ):
            objective, auxiliary = problem.value(parameters, args)
            evaluations += 1
        else:
            objective, auxiliary = final.objective, None
        diagnostics = OptimizationDiagnostics(
            iterations=iterations,
            accepted_steps=accepted,
            rejected_steps=rejected,
            objective_evaluations=evaluations + certificate.evaluation_work,
            gradient_evaluations=gradients + certificate.evaluation_work,
            constraint_evaluations=constraints + certificate.evaluation_work,
            linear_solves=linear_solves,
            linear_iterations=linear_solves,
            globalization_evaluations=globalization_evaluations,
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
                f"restorations={restorations};kkt-plan={kkt_plan.plan_id};internal-status={status}"
            ),
        )
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
