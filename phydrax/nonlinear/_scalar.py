#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Callable, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._precision import NonlinearPrecisionPolicy
from ._types import (
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearTermination,
)


BracketMethod: TypeAlias = Literal["bisection", "brent", "ridder", "toms748"]


def _scalar_abs(value: Any, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.abs(precision.accumulation(value)))


def _sign_product(
    left: Any,
    right: Any,
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    return precision.decision(
        precision.accumulation(left) * precision.accumulation(right)
    )


class ScalarRootProblem(StrictModule):
    """One real scalar equation with optional bracket and derivatives."""

    function: Callable[[Array, Any], Any]
    derivative: Callable[[Array, Any], Any] | None
    second_derivative: Callable[[Array, Any], Any] | None
    validity: Callable[[Array, Array, Any], Any]
    lower: Array | None
    upper: Array | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array, Any], Any],
        /,
        *,
        bracket: tuple[Any, Any] | None = None,
        derivative: Callable[[Array, Any], Any] | None = None,
        second_derivative: Callable[[Array, Any], Any] | None = None,
        validity: Callable[[Array, Array, Any], Any] | None = None,
        problem_id: str = "scalar-root",
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if derivative is not None and not callable(derivative):
            raise TypeError("derivative must be callable or None.")
        if second_derivative is not None and not callable(second_derivative):
            raise TypeError("second_derivative must be callable or None.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        lower = upper = None
        if bracket is not None:
            if len(bracket) != 2:
                raise ValueError("bracket must contain two endpoints.")
            lower = jnp.asarray(bracket[0])
            upper = jnp.asarray(bracket[1])
            if lower.shape != () or upper.shape != ():
                raise ValueError("Scalar root bracket endpoints must be scalar.")
            lower = eqx.error_if(
                lower,
                ~jnp.isfinite(lower) | ~jnp.isfinite(upper) | (lower >= upper),
                "Scalar root bracket must be finite and ordered.",
            )
        self.function = function
        self.derivative = derivative
        self.second_derivative = second_derivative
        self.validity = (
            (lambda state, value, args: jnp.asarray(True))
            if validity is None
            else validity
        )
        self.lower = lower
        self.upper = upper
        self.problem_id = identifier

    def evaluate(self, state: Any, args: Any = None, /) -> Array:
        state_ = jnp.asarray(state)
        value = jnp.asarray(self.function(state_, args))
        if state_.shape != () or value.shape != ():
            raise ValueError("Scalar root state and value must be scalar.")
        return value

    def valid(self, state: Any, value: Any, args: Any = None, /) -> Array:
        return jnp.asarray(self.validity(jnp.asarray(state), jnp.asarray(value), args))


class ScalarRootResult(StrictModule):
    nonlinear_result: NonlinearResult
    lower: Array
    upper: Array
    lower_value: Array
    upper_value: Array
    bracket_valid: Array

    @property
    def root(self) -> Array:
        return self.nonlinear_result.state

    @property
    def value(self) -> Array:
        return self.nonlinear_result.residual

    @property
    def status(self) -> Array:
        return self.nonlinear_result.status

    @property
    def successful(self) -> Array:
        return self.nonlinear_result.successful


class AbstractScalarRootMethod(StrictModule):
    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        problem: ScalarRootProblem,
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ) -> ScalarRootResult:
        raise NotImplementedError


class _BracketRun(StrictModule):
    lower: Array
    upper: Array
    lower_value: Array
    upper_value: Array
    previous: Array
    previous_value: Array
    best: Array
    best_value: Array
    initial_residual: Array
    iterations: Array
    evaluations: Array
    nonfinite: Array
    domain_failures: Array
    status: Array


def _inverse_quadratic(a, fa, b, fb, c, fc):
    def safe(value):
        return jnp.where(
            jnp.abs(value) < 1e-30,
            jnp.where(value < 0.0, -1e-30, 1e-30),
            value,
        )

    first = a * fb * fc / safe((fa - fb) * (fa - fc))
    second = b * fa * fc / safe((fb - fa) * (fb - fc))
    third = c * fa * fb / safe((fc - fa) * (fc - fb))
    return first + second + third


def _safe_candidate(candidate, lower, upper):
    width = upper - lower
    interior_lower = lower + 0.05 * width
    interior_upper = upper - 0.05 * width
    midpoint = 0.5 * (lower + upper)
    return jnp.where(
        jnp.isfinite(candidate)
        & (candidate > interior_lower)
        & (candidate < interior_upper),
        candidate,
        midpoint,
    )


def _bracket_candidate(kind, run):
    midpoint = 0.5 * (run.lower + run.upper)
    if kind == "bisection":
        return midpoint
    distinct = (
        (run.lower_value != run.upper_value)
        & (run.lower_value != run.previous_value)
        & (run.upper_value != run.previous_value)
    )
    inverse = _inverse_quadratic(
        run.lower,
        run.lower_value,
        run.upper,
        run.upper_value,
        run.previous,
        run.previous_value,
    )
    secant = run.upper - run.upper_value * (run.upper - run.lower) / jnp.where(
        run.upper_value == run.lower_value,
        1.0,
        run.upper_value - run.lower_value,
    )
    candidate = jnp.where(distinct, inverse, secant)
    return _safe_candidate(candidate, run.lower, run.upper)


def _update_bracket(run, candidate, value, precision, /):
    replace_upper = _sign_product(run.lower_value, value, precision) <= 0.0
    lower = jnp.where(replace_upper, run.lower, candidate)
    lower_value = jnp.where(replace_upper, run.lower_value, value)
    upper = jnp.where(replace_upper, candidate, run.upper)
    upper_value = jnp.where(replace_upper, value, run.upper_value)
    choose_candidate = _scalar_abs(value, precision) < _scalar_abs(
        run.best_value,
        precision,
    )
    return (
        lower,
        upper,
        lower_value,
        upper_value,
        jnp.where(choose_candidate, candidate, run.best),
        jnp.where(choose_candidate, value, run.best_value),
    )


def _solve_bracketed(
    problem: ScalarRootProblem,
    method_id: str,
    kind: BracketMethod,
    termination: NonlinearTermination,
    args: Any,
    precision: NonlinearPrecisionPolicy,
    /,
) -> ScalarRootResult:
    precision.validate_tolerance(termination.absolute_residual)
    if problem.lower is None or problem.upper is None:
        raise ValueError(f"{method_id} requires a bracket.")
    lower = precision.state(problem.lower)
    upper = precision.state(problem.upper)
    lower_value = precision.residual(problem.evaluate(lower, args))
    upper_value = precision.residual(problem.evaluate(upper, args))
    precision.validate_trees(lower, lower_value)
    lower_valid = problem.valid(lower, lower_value, args)
    upper_valid = problem.valid(upper, upper_value, args)
    finite = (
        jnp.isfinite(lower_value) & jnp.isfinite(upper_value) & lower_valid & upper_valid
    )
    bracket_valid = finite & (_sign_product(lower_value, upper_value, precision) <= 0.0)
    lower_best = _scalar_abs(lower_value, precision) <= _scalar_abs(
        upper_value,
        precision,
    )
    best = jnp.where(lower_best, lower, upper)
    best_value = jnp.where(lower_best, lower_value, upper_value)
    initial_residual = jnp.maximum(
        jnp.minimum(
            _scalar_abs(lower_value, precision),
            _scalar_abs(upper_value, precision),
        ),
        precision.decision(1e-30),
    )
    endpoint_success = bracket_valid & (
        _scalar_abs(best_value, precision)
        <= termination.residual_threshold(initial_residual)
    )
    run = _BracketRun(
        lower=lower,
        upper=upper,
        lower_value=lower_value,
        upper_value=upper_value,
        previous=lower,
        previous_value=lower_value,
        best=best,
        best_value=best_value,
        initial_residual=initial_residual,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        evaluations=jnp.asarray(2, dtype=jnp.int32),
        nonfinite=(~finite).astype(jnp.int32),
        domain_failures=(finite & ~(lower_valid & upper_valid)).astype(jnp.int32),
        status=jnp.where(
            endpoint_success,
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                bracket_valid,
                int(NonlinearStatus.ITERATING),
                int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            ),
        ).astype(jnp.int32),
    )

    def condition(current):
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current.evaluations
            + (3 if kind == "ridder" else 2 if kind == "toms748" else 1)
            <= termination.maximum_evaluations
        )
        return (
            (current.status == int(NonlinearStatus.ITERATING))
            & (current.iterations < termination.maximum_steps)
            & within_evaluations
        )

    def one_candidate(current, candidate):
        candidate = precision.state(candidate)
        value = precision.residual(problem.evaluate(candidate, args))
        valid = problem.valid(candidate, value, args)
        finite_value = jnp.isfinite(value)
        usable = finite_value & valid
        (
            next_lower,
            next_upper,
            next_lower_value,
            next_upper_value,
            next_best,
            next_best_value,
        ) = _update_bracket(current, candidate, value, precision)
        return _BracketRun(
            lower=jnp.where(usable, next_lower, current.lower),
            upper=jnp.where(usable, next_upper, current.upper),
            lower_value=jnp.where(usable, next_lower_value, current.lower_value),
            upper_value=jnp.where(usable, next_upper_value, current.upper_value),
            previous=current.best,
            previous_value=current.best_value,
            best=jnp.where(usable, next_best, current.best),
            best_value=jnp.where(usable, next_best_value, current.best_value),
            initial_residual=current.initial_residual,
            iterations=current.iterations,
            evaluations=current.evaluations + 1,
            nonfinite=current.nonfinite + (~finite_value).astype(jnp.int32),
            domain_failures=current.domain_failures
            + (finite_value & ~valid).astype(jnp.int32),
            status=jnp.where(
                usable,
                int(NonlinearStatus.ITERATING),
                jnp.where(
                    finite_value,
                    int(NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE),
                    int(NonlinearStatus.NONFINITE_EVALUATION),
                ),
            ).astype(jnp.int32),
        )

    def body(current):
        if kind == "ridder":
            midpoint = 0.5 * (current.lower + current.upper)
            midpoint_value = precision.residual(problem.evaluate(midpoint, args))
            discriminant = jnp.maximum(
                precision.accumulation(midpoint_value) ** 2
                - _sign_product(
                    current.lower_value,
                    current.upper_value,
                    precision,
                ),
                0.0,
            )
            candidate = midpoint + (
                (midpoint - current.lower)
                * jnp.sign(current.lower_value - current.upper_value)
                * midpoint_value
                / jnp.maximum(jnp.sqrt(discriminant), 1e-30)
            )
            candidate = _safe_candidate(
                candidate,
                current.lower,
                current.upper,
            )
            middle = one_candidate(current, midpoint)
            updated = one_candidate(middle, candidate)
            updated = eqx.tree_at(
                lambda value: value.evaluations,
                updated,
                updated.evaluations + 1,
            )
        elif kind == "toms748":
            first = _bracket_candidate("brent", current)
            intermediate = one_candidate(current, first)
            second = _bracket_candidate("brent", intermediate)
            updated = one_candidate(intermediate, second)
        else:
            candidate = _bracket_candidate(kind, current)
            updated = one_candidate(current, candidate)
        residual_converged = _scalar_abs(
            updated.best_value,
            precision,
        ) <= termination.residual_threshold(updated.initial_residual)
        bracket_width = updated.upper - updated.lower
        bracket_stagnated = bracket_width <= termination.step_threshold(
            jnp.maximum(jnp.abs(updated.lower), jnp.abs(updated.upper))
        )
        status = jnp.where(
            residual_converged,
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                bracket_stagnated,
                int(NonlinearStatus.RESIDUAL_STAGNATION),
                updated.status,
            ),
        ).astype(jnp.int32)
        return eqx.tree_at(
            lambda value: (value.iterations, value.status),
            updated,
            (updated.iterations + 1, status),
        )

    run = jax.lax.while_loop(condition, body, run)
    exhausted = (
        jnp.asarray(False)
        if termination.maximum_evaluations is None
        else run.evaluations >= termination.maximum_evaluations
    )
    status = jnp.where(
        run.status == int(NonlinearStatus.ITERATING),
        jnp.where(
            exhausted,
            int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
        ),
        run.status,
    ).astype(jnp.int32)
    certificate_state = precision.certificate(run.best)
    certificate_value = precision.certificate(problem.evaluate(certificate_state, args))
    certificate_norm = _scalar_abs(certificate_value, precision)
    threshold = precision.decision(termination.residual_threshold(run.initial_residual))
    certified = (
        jnp.isfinite(certificate_state)
        & jnp.isfinite(certificate_value)
        & problem.valid(certificate_state, certificate_value, args)
        & (certificate_norm <= threshold)
    )
    status = jnp.where(
        (status == int(NonlinearStatus.SUCCESS)) & ~certified,
        int(NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED),
        status,
    ).astype(jnp.int32)
    diagnostics = NonlinearDiagnostics(
        initial_residual_norm=run.initial_residual,
        final_residual_norm=certificate_norm,
        final_step_norm=precision.decision(run.upper - run.lower),
        iterations=run.iterations,
        residual_evaluations=run.evaluations + 1,
        accepted_steps=run.iterations,
        domain_failures=run.domain_failures,
        nonfinite_trials=run.nonfinite,
    )
    output_state = precision.output(certificate_state)
    nonlinear = NonlinearResult(
        state=output_state,
        residual=certificate_value,
        auxiliary=None,
        status=status,
        diagnostics=diagnostics,
        provenance=NonlinearProvenance(
            problem_id=problem.problem_id,
            method_id=method_id,
            derivative_id="function-values",
            globalization_id="certified-bracket",
            precision_policy_id=precision.policy_id,
        ),
        precision_evidence=precision.evidence_for(
            run.best,
            run.best_value,
            output_value=output_state,
        ),
    )
    return ScalarRootResult(
        nonlinear,
        run.lower,
        run.upper,
        run.lower_value,
        run.upper_value,
        _sign_product(run.lower_value, run.upper_value, precision) <= 0.0,
    )


class AbstractBracketedScalarRoot(AbstractScalarRootMethod):
    @property
    @abc.abstractmethod
    def kind(self) -> BracketMethod:
        raise NotImplementedError

    def solve(self, problem, /, *, termination, args=None, precision=None):
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        return _solve_bracketed(
            problem,
            self.method_id,
            self.kind,
            termination,
            args,
            precision_,
        )


class Bisection(AbstractBracketedScalarRoot):
    @property
    def method_id(self):
        return "bisection"

    @property
    def kind(self):
        return "bisection"


class Brent(AbstractBracketedScalarRoot):
    @property
    def method_id(self):
        return "brent-dekker"

    @property
    def kind(self):
        return "brent"


class Ridder(AbstractBracketedScalarRoot):
    @property
    def method_id(self):
        return "ridder"

    @property
    def kind(self):
        return "ridder"


class TOMS748(AbstractBracketedScalarRoot):
    @property
    def method_id(self):
        return "toms748"

    @property
    def kind(self):
        return "toms748"


class _OpenRun(StrictModule):
    current: Array
    current_value: Array
    previous: Array
    previous_value: Array
    lower: Array
    upper: Array
    lower_value: Array
    upper_value: Array
    initial_residual: Array
    iterations: Array
    evaluations: Array
    derivative_evaluations: Array
    status: Array


class AbstractSafeguardedDerivativeRoot(AbstractScalarRootMethod):
    order: int = eqx.field(static=True)

    def __init__(self):
        self.order = self.derivative_order

    @property
    @abc.abstractmethod
    def derivative_order(self) -> int:
        raise NotImplementedError

    @property
    def method_id(self):
        return "safeguarded-halley" if self.order == 2 else "safeguarded-newton"

    def solve(self, problem, /, *, termination, args=None, precision=None):
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        precision_.validate_tolerance(termination.absolute_residual)
        if problem.lower is None or problem.upper is None:
            raise ValueError("Safeguarded derivative roots require a bracket.")
        derivative = (
            problem.derivative
            if problem.derivative is not None
            else lambda value, current_args: jax.grad(
                lambda point: problem.evaluate(point, current_args)
            )(value)
        )
        second = (
            problem.second_derivative
            if problem.second_derivative is not None
            else lambda value, current_args: jax.grad(
                lambda point: derivative(point, current_args)
            )(value)
        )
        lower = precision_.state(problem.lower)
        upper = precision_.state(problem.upper)
        lower_value = precision_.residual(problem.evaluate(lower, args))
        upper_value = precision_.residual(problem.evaluate(upper, args))
        precision_.validate_trees(lower, lower_value)
        bracket_valid = (
            jnp.isfinite(lower_value)
            & jnp.isfinite(upper_value)
            & (_sign_product(lower_value, upper_value, precision_) <= 0.0)
        )
        choose_lower = _scalar_abs(lower_value, precision_) <= _scalar_abs(
            upper_value,
            precision_,
        )
        current = jnp.where(choose_lower, lower, upper)
        current_value = jnp.where(choose_lower, lower_value, upper_value)
        initial = jnp.maximum(
            _scalar_abs(current_value, precision_),
            precision_.decision(1e-30),
        )
        run = _OpenRun(
            current=current,
            current_value=current_value,
            previous=current,
            previous_value=current_value,
            lower=lower,
            upper=upper,
            lower_value=lower_value,
            upper_value=upper_value,
            initial_residual=initial,
            iterations=jnp.asarray(0, dtype=jnp.int32),
            evaluations=jnp.asarray(2, dtype=jnp.int32),
            derivative_evaluations=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.where(
                bracket_valid,
                int(NonlinearStatus.ITERATING),
                int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            ).astype(jnp.int32),
        )

        def condition(value):
            return (value.status == int(NonlinearStatus.ITERATING)) & (
                value.iterations < termination.maximum_steps
            )

        def body(value):
            first = precision_.direction(derivative(value.current, args))
            first_ = precision_.accumulation(first)
            residual_ = precision_.accumulation(value.current_value)
            if self.order == 2:
                second_value = precision_.direction(second(value.current, args))
                second_ = precision_.accumulation(second_value)
                denominator = 2.0 * first_ * first_ - residual_ * second_
                raw_step = (
                    2.0
                    * residual_
                    * first_
                    / jnp.where(denominator == 0.0, 1.0, denominator)
                )
                derivative_count = 2
            else:
                raw_step = residual_ / jnp.where(first_ == 0.0, 1.0, first_)
                derivative_count = 1
            candidate = precision_.state(
                _safe_candidate(
                    value.current - raw_step,
                    value.lower,
                    value.upper,
                )
            )
            candidate_value = precision_.residual(problem.evaluate(candidate, args))
            valid = problem.valid(candidate, candidate_value, args)
            finite_value = jnp.isfinite(candidate_value) & valid
            replace_upper = (
                _sign_product(value.lower_value, candidate_value, precision_) <= 0.0
            )
            lower = jnp.where(replace_upper, value.lower, candidate)
            lower_value = jnp.where(replace_upper, value.lower_value, candidate_value)
            upper = jnp.where(replace_upper, candidate, value.upper)
            upper_value = jnp.where(replace_upper, candidate_value, value.upper_value)
            residual_converged = _scalar_abs(
                candidate_value,
                precision_,
            ) <= termination.residual_threshold(value.initial_residual)
            step_stagnated = _scalar_abs(
                candidate - value.current,
                precision_,
            ) <= termination.step_threshold(_scalar_abs(value.current, precision_))
            status = jnp.where(
                ~finite_value,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    residual_converged,
                    int(NonlinearStatus.SUCCESS),
                    jnp.where(
                        step_stagnated,
                        int(NonlinearStatus.RESIDUAL_STAGNATION),
                        int(NonlinearStatus.ITERATING),
                    ),
                ),
            ).astype(jnp.int32)
            return _OpenRun(
                current=jnp.where(finite_value, candidate, value.current),
                current_value=jnp.where(
                    finite_value, candidate_value, value.current_value
                ),
                previous=value.current,
                previous_value=value.current_value,
                lower=jnp.where(finite_value, lower, value.lower),
                upper=jnp.where(finite_value, upper, value.upper),
                lower_value=jnp.where(finite_value, lower_value, value.lower_value),
                upper_value=jnp.where(finite_value, upper_value, value.upper_value),
                initial_residual=value.initial_residual,
                iterations=value.iterations + 1,
                evaluations=value.evaluations + 1,
                derivative_evaluations=value.derivative_evaluations + derivative_count,
                status=status,
            )

        run = jax.lax.while_loop(condition, body, run)
        status = jnp.where(
            run.status == int(NonlinearStatus.ITERATING),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            run.status,
        ).astype(jnp.int32)
        certificate_state = precision_.certificate(run.current)
        certificate_value = precision_.certificate(
            problem.evaluate(certificate_state, args)
        )
        certificate_norm = _scalar_abs(certificate_value, precision_)
        certified = (
            jnp.isfinite(certificate_state)
            & jnp.isfinite(certificate_value)
            & problem.valid(certificate_state, certificate_value, args)
            & (certificate_norm <= termination.residual_threshold(run.initial_residual))
        )
        status = jnp.where(
            (status == int(NonlinearStatus.SUCCESS)) & ~certified,
            int(NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED),
            status,
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=run.initial_residual,
            final_residual_norm=certificate_norm,
            final_step_norm=_scalar_abs(
                run.current - run.previous,
                precision_,
            ),
            iterations=run.iterations,
            residual_evaluations=run.evaluations + 1,
            jvp_evaluations=run.derivative_evaluations,
            accepted_steps=run.iterations,
        )
        output_state = precision_.output(certificate_state)
        nonlinear = NonlinearResult(
            state=output_state,
            residual=certificate_value,
            auxiliary=None,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="first-and-second" if self.order == 2 else "first",
                globalization_id="certified-bracket",
                precision_policy_id=precision_.policy_id,
            ),
            precision_evidence=precision_.evidence_for(
                run.current,
                run.current_value,
                output_value=output_state,
            ),
        )
        return ScalarRootResult(
            nonlinear,
            run.lower,
            run.upper,
            run.lower_value,
            run.upper_value,
            _sign_product(run.lower_value, run.upper_value, precision_) <= 0.0,
        )


class SafeguardedNewton(AbstractSafeguardedDerivativeRoot):
    @property
    def derivative_order(self) -> int:
        return 1


class SafeguardedHalley(AbstractSafeguardedDerivativeRoot):
    @property
    def derivative_order(self) -> int:
        return 2


def scalar_root(
    problem: ScalarRootProblem,
    /,
    *,
    method: AbstractScalarRootMethod | None = None,
    termination: NonlinearTermination | None = None,
    args: Any = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> ScalarRootResult:
    if not isinstance(problem, ScalarRootProblem):
        raise TypeError("problem must be ScalarRootProblem.")
    method_ = TOMS748() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(method_, AbstractScalarRootMethod):
        raise TypeError("method must be AbstractScalarRootMethod or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    return method_.solve(
        problem,
        termination=termination_,
        args=args,
        precision=precision_,
    )


__all__ = [
    "AbstractScalarRootMethod",
    "Bisection",
    "Brent",
    "Ridder",
    "SafeguardedHalley",
    "SafeguardedNewton",
    "ScalarRootProblem",
    "ScalarRootResult",
    "TOMS748",
    "scalar_root",
]
