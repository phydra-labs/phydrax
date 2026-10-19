#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Callable, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._iteration import IterationPlan
from .._strict import StrictModule
from .._validation import positive_integer, positive_real_scalar
from ..typing import checked
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
    ) -> None:
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


def quadratic_event_bound(
    quadratic: Array, linear: Array, constant: Array, valid: Array, /
) -> tuple[Array, Array]:
    """Bound a trial by the first simple positive quadratic event in (0, 1].

    This is finite floating algebra, not a certified source predicate or an
    iterative scalar solve. Zero roots are already represented events;
    persistent equal polynomials and double roots do not change ordering.
    Invalid active arithmetic refuses the bound rather than inventing a root.
    """
    active = jnp.asarray(valid, dtype=jnp.bool_)
    finite = jnp.isfinite(quadratic) & jnp.isfinite(linear) & jnp.isfinite(constant)
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(quadratic), jnp.abs(linear)), jnp.abs(constant)
    )
    denominator = jnp.where(scale > 0.0, scale, 1.0)
    a = jnp.where(active & finite, quadratic / denominator, 0.0)
    b = jnp.where(active & finite, linear / denominator, 0.0)
    c = jnp.where(active & finite, constant / denominator, 0.0)
    discriminant = b * b - 4.0 * a * c
    radical = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    q = -0.5 * (b + jnp.where(b >= 0.0, radical, -radical))
    is_quadratic = a != 0.0
    roots = jnp.stack(
        (
            q / jnp.where(is_quadratic, a, 1.0),
            c / jnp.where(q != 0.0, q, 1.0),
            -c / jnp.where(b != 0.0, b, 1.0),
        )
    )
    simple = jnp.stack(
        (
            is_quadratic & (discriminant > 0.0),
            is_quadratic & (discriminant > 0.0) & (q != 0.0),
            (~is_quadratic) & (b != 0.0),
        )
    )
    candidates = jnp.where(
        active[None, ...] & simple & (roots > 0.0) & (roots <= 1.0),
        roots,
        jnp.inf,
    )
    first = jnp.min(candidates)
    admissible = jnp.all((~active) | finite)
    return jnp.where(admissible, jnp.minimum(first, 1.0), 0.0), admissible


def _inverse_quadratic(
    a: Array, fa: Array, b: Array, fb: Array, c: Array, fc: Array
) -> Array:
    def safe(value: Array) -> Array:
        return jnp.where(
            jnp.abs(value) < 1e-30,
            jnp.where(value < 0.0, -1e-30, 1e-30),
            value,
        )

    first = a * fb * fc / safe((fa - fb) * (fa - fc))
    second = b * fa * fc / safe((fb - fa) * (fb - fc))
    third = c * fa * fb / safe((fc - fa) * (fc - fb))
    return first + second + third


def _safe_candidate(candidate: Array, lower: Array, upper: Array) -> Array:
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


def _bracket_candidate(kind: BracketMethod, run: _BracketRun) -> Array:
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


def _update_bracket(
    run: _BracketRun,
    candidate: Array,
    value: Array,
    precision: NonlinearPrecisionPolicy,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
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

    def condition(current: _BracketRun) -> Array:
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

    def one_candidate(current: _BracketRun, candidate: Array) -> _BracketRun:
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

    def body(current: _BracketRun) -> _BracketRun:
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

    def solve(
        self,
        problem: ScalarRootProblem,
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ) -> ScalarRootResult:
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
    def method_id(self) -> str:
        return "bisection"

    @property
    def kind(self) -> BracketMethod:
        return "bisection"


class Brent(AbstractBracketedScalarRoot):
    @property
    def method_id(self) -> str:
        return "brent-dekker"

    @property
    def kind(self) -> BracketMethod:
        return "brent"


class Ridder(AbstractBracketedScalarRoot):
    @property
    def method_id(self) -> str:
        return "ridder"

    @property
    def kind(self) -> BracketMethod:
        return "ridder"


class TOMS748(AbstractBracketedScalarRoot):
    @property
    def method_id(self) -> str:
        return "toms748"

    @property
    def kind(self) -> BracketMethod:
        return "toms748"


type _ITPCarry = tuple[Array, Array, Array, Array, Array, Array, Array]


def _itp_run(
    function: Callable[[Array], Array],
    lower_values: Array,
    upper_values: Array,
    tolerance: Array,
    template: Array,
    bracket_tolerance: float,
    maximum_steps: int,
    truncation_factor: float,
    truncation_exponent: float,
    /,
) -> _ITPCarry:
    """Isolate unit-interval roots with a shared bounded batch traversal.

    The seven-field carry freezes a lane after its residual meets tolerance.
    Interpolation, truncation, projection, and bracket updates retain their
    arithmetic order; all callback evaluations remain inside one JAX loop.
    """
    count = lower_values.shape[0]
    initial: _ITPCarry = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((count,), dtype=template.dtype),
        jnp.ones((count,), dtype=template.dtype),
        lower_values,
        upper_values,
        jnp.zeros((count,), dtype=template.dtype),
        jnp.zeros((count,), dtype=jnp.bool_),
    )

    def active(carry: _ITPCarry) -> Array:
        _, left, right, _, _, _, converged = carry
        return ~converged & (right - left > 2.0 * bracket_tolerance)

    def condition(carry: _ITPCarry) -> Array:
        return (carry[0] < maximum_steps) & jnp.any(active(carry))

    def body(carry: _ITPCarry) -> _ITPCarry:
        iteration, left, right, left_value, right_value, root, converged = carry
        running = active(carry)
        width = right - left
        middle = 0.5 * (left + right)
        radius = jnp.maximum(
            bracket_tolerance
            * jnp.exp2((maximum_steps - 1 - iteration).astype(template.dtype))
            - 0.5 * width,
            0.0,
        )
        truncation = truncation_factor * width**truncation_exponent
        falsi = (right_value * left - left_value * right) / (right_value - left_value)
        falsi = jnp.where(jnp.isfinite(falsi), falsi, middle)
        sigma = jnp.sign(middle - falsi)
        truncated = jnp.where(
            truncation <= jnp.abs(middle - falsi), falsi + sigma * truncation, middle
        )
        candidate = jnp.where(
            jnp.abs(truncated - middle) <= radius, truncated, middle - sigma * radius
        )
        value = function(candidate)
        same_as_left = (value < 0.0) == (left_value < 0.0)
        move_left = running & same_as_left
        move_right = running & ~same_as_left
        hit = running & (jnp.abs(value) <= tolerance)
        return (
            iteration + 1,
            jnp.where(move_left, candidate, left),
            jnp.where(move_right, candidate, right),
            jnp.where(move_left, value, left_value),
            jnp.where(move_right, value, right_value),
            jnp.where(hit, candidate, root),
            converged | hit,
        )

    return jax.lax.while_loop(condition, body, initial)


def _itp_result(
    function: Callable[[Array], Array],
    run: _ITPCarry,
    initial_lower_values: Array,
    initial_upper_values: Array,
    tolerance: Array,
    precision: NonlinearPrecisionPolicy,
    bracket_tolerance: float,
    problem_id: str,
    /,
    *,
    evaluation_limited: bool = False,
) -> ScalarRootResult:
    """Publish lane statuses and batch-global iteration/evaluation evidence."""
    iterations, left, right, left_value, right_value, root, converged = run
    parameter = jnp.where(converged, root, 0.5 * (left + right))
    certificate_state = precision.certificate(parameter)
    value = precision.certificate(function(certificate_state))
    residual = _scalar_abs(value, precision)
    initial = jnp.minimum(
        _scalar_abs(initial_lower_values, precision),
        _scalar_abs(initial_upper_values, precision),
    )
    finite_inputs = jnp.isfinite(initial_lower_values) & jnp.isfinite(
        initial_upper_values
    )
    bracket_valid = finite_inputs & (
        _sign_product(initial_lower_values, initial_upper_values, precision) <= 0.0
    )
    finite_result = jnp.isfinite(certificate_state) & jnp.isfinite(value)
    exhausted = (
        NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED
        if evaluation_limited
        else NonlinearStatus.MAXIMUM_STEPS_REACHED
    )
    status = jnp.where(
        ~finite_inputs,
        int(NonlinearStatus.NONFINITE_INPUT),
        jnp.where(
            ~bracket_valid,
            int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            jnp.where(
                ~finite_result,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    residual <= tolerance,
                    int(NonlinearStatus.SUCCESS),
                    jnp.where(
                        right - left <= 2.0 * bracket_tolerance,
                        int(NonlinearStatus.RESIDUAL_STAGNATION),
                        int(exhausted),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    output = precision.output(certificate_state)
    nonlinear = NonlinearResult(
        state=output,
        residual=value,
        auxiliary=None,
        status=status,
        diagnostics=NonlinearDiagnostics(
            initial_residual_norm=initial,
            final_residual_norm=residual,
            final_step_norm=precision.decision(right - left),
            iterations=iterations,
            residual_evaluations=iterations + 3,
            accepted_steps=iterations,
            domain_failures=(finite_inputs & ~bracket_valid).astype(jnp.int32),
            nonfinite_trials=(~finite_result).astype(jnp.int32),
        ),
        provenance=NonlinearProvenance(
            problem_id=problem_id,
            method_id="itp",
            derivative_id="function-values",
            globalization_id="certified-bracket",
            precision_policy_id=precision.policy_id,
        ),
        precision_evidence=precision.evidence_for(parameter, value, output_value=output),
    )
    return ScalarRootResult(
        nonlinear, left, right, left_value, right_value, bracket_valid
    )


@final
class ITP(AbstractScalarRootMethod):
    """Interpolate, truncate, and project bracketed scalar roots.

    Oliveira and Takahashi, ACM TOMS 47(1), Article 5 (2021),
    doi:10.1145/3423597. Unit-interval batches use the same seven-field carry
    as scalar solves; residual and bracket failures remain visible in status.
    """

    bracket_tolerance: float = eqx.field(static=True)
    truncation_factor: float = eqx.field(static=True)
    truncation_exponent: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        bracket_tolerance: float = 2.0**-50,
        truncation_factor: float = 0.2,
        truncation_exponent: float = 2.0,
    ) -> None:
        tolerance = positive_real_scalar(bracket_tolerance, "ITP.bracket_tolerance")
        factor = positive_real_scalar(truncation_factor, "ITP.truncation_factor")
        exponent = positive_real_scalar(truncation_exponent, "ITP.truncation_exponent")
        if tolerance >= 0.5:
            raise ValueError(
                f"ITP.bracket_tolerance must be below one half; got {tolerance}"
            )
        if not 1.0 <= exponent < 1.0 + (1.0 + 5.0**0.5) / 2.0:
            raise ValueError(
                "ITP.truncation_exponent must lie in [1, 1 + golden ratio); "
                f"got {exponent}"
            )
        self.bracket_tolerance = tolerance
        self.truncation_factor = factor
        self.truncation_exponent = exponent

    @property
    def method_id(self) -> str:
        return "itp"

    @checked
    def solve_batched(
        self,
        function: Callable[[Array], Array],
        lower_values: Array,
        upper_values: Array,
        tolerance: Array,
        template: Array,
        /,
        *,
        maximum_steps: int = 51,
        precision: NonlinearPrecisionPolicy | None = None,
        problem_id: str = "batched-unit-interval-root",
    ) -> ScalarRootResult:
        """Solve independent roots on [0, 1] with supplied endpoint evaluations.

        ``template`` selects parameter precision. Endpoint values and residual
        tolerances are lane data; scalar counters measure the shared batch.
        The callback preserves lane shape, and each lane must have a finite
        sign-changing bracket for successful status.
        """
        steps = positive_integer(maximum_steps, "ITP.maximum_steps")
        if lower_values.ndim != 1 or upper_values.shape != lower_values.shape:
            raise ValueError(
                "ITP.endpoint_values must be matching rank-one arrays; "
                f"got {lower_values.shape}, {upper_values.shape}"
            )
        if tolerance.shape not in ((), lower_values.shape):
            raise ValueError(
                f"ITP.tolerance must be scalar or match the lanes; got {tolerance.shape}"
            )
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        run = _itp_run(
            function,
            lower_values,
            upper_values,
            tolerance,
            template,
            self.bracket_tolerance,
            steps,
            self.truncation_factor,
            self.truncation_exponent,
        )
        return _itp_result(
            function,
            run,
            lower_values,
            upper_values,
            tolerance,
            precision_,
            self.bracket_tolerance,
            problem_id,
        )

    @checked
    def solve(
        self,
        problem: ScalarRootProblem,
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ) -> ScalarRootResult:
        """Solve a scalar bracket with native residual and work-limit evidence."""
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        precision_.validate_tolerance(termination.absolute_residual)
        if problem.lower is None or problem.upper is None:
            raise ValueError(
                f"ITP.problem.bracket must contain lower and upper bounds; got {(problem.lower, problem.upper)!r}."
            )
        lower, upper = precision_.state(problem.lower), precision_.state(problem.upper)
        width = upper - lower

        def function(parameter: Array) -> Array:
            return precision_.residual(
                problem.evaluate(lower + parameter[0] * width, args)
            )[None]

        lower_value = precision_.residual(problem.evaluate(lower, args))[None]
        upper_value = precision_.residual(problem.evaluate(upper, args))[None]
        initial = jnp.maximum(
            jnp.minimum(
                _scalar_abs(lower_value, precision_), _scalar_abs(upper_value, precision_)
            ),
            precision_.decision(1e-30),
        )
        tolerance = termination.residual_threshold(initial)
        steps = termination.maximum_steps
        evaluation_limited = termination.maximum_evaluations is not None and (
            termination.maximum_evaluations - 3 < steps
        )
        if termination.maximum_evaluations is not None:
            steps = max(0, min(steps, termination.maximum_evaluations - 3))
        lower_best = _scalar_abs(lower_value, precision_) <= _scalar_abs(
            upper_value, precision_
        )
        endpoint_success = (
            jnp.minimum(
                _scalar_abs(lower_value, precision_), _scalar_abs(upper_value, precision_)
            )
            <= tolerance
        )
        endpoint_run: _ITPCarry = (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros_like(lower_value),
            jnp.ones_like(upper_value),
            lower_value,
            upper_value,
            jnp.where(
                lower_best, jnp.zeros_like(lower_value), jnp.ones_like(upper_value)
            ),
            endpoint_success,
        )
        run = jax.lax.cond(
            jnp.all(endpoint_success),
            lambda: endpoint_run,
            lambda: _itp_run(
                function,
                lower_value,
                upper_value,
                tolerance,
                lower,
                self.bracket_tolerance,
                steps,
                self.truncation_factor,
                self.truncation_exponent,
            ),
        )
        normalized = _itp_result(
            function,
            run,
            lower_value,
            upper_value,
            tolerance,
            precision_,
            self.bracket_tolerance,
            problem.problem_id,
            evaluation_limited=evaluation_limited,
        )
        parameter = jnp.where(run[-1], run[-2], 0.5 * (run[1] + run[2]))[0]
        certificate_root = precision_.certificate(lower + parameter * width)
        root = precision_.output(certificate_root)
        value = normalized.value[0]
        status = normalized.status[0]
        valid = (
            problem.valid(lower, lower_value[0], args)
            & problem.valid(upper, upper_value[0], args)
            & problem.valid(root, value, args)
        )
        status = jnp.where(
            valid, status, int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)
        ).astype(jnp.int32)
        diagnostics = eqx.tree_at(
            lambda d: (
                d.initial_residual_norm,
                d.final_residual_norm,
                d.final_step_norm,
                d.domain_failures,
                d.nonfinite_trials,
            ),
            normalized.nonlinear_result.diagnostics,
            (
                initial[0],
                _scalar_abs(value, precision_),
                precision_.decision((normalized.upper[0] - normalized.lower[0]) * width),
                normalized.nonlinear_result.diagnostics.domain_failures[0],
                normalized.nonlinear_result.diagnostics.nonfinite_trials[0],
            ),
        )
        nonlinear = NonlinearResult(
            state=root,
            residual=value,
            auxiliary=None,
            status=status,
            diagnostics=diagnostics,
            provenance=normalized.nonlinear_result.provenance,
            precision_evidence=precision_.evidence_for(
                certificate_root, value, output_value=root
            ),
        )
        return ScalarRootResult(
            nonlinear,
            lower + normalized.lower[0] * width,
            lower + normalized.upper[0] * width,
            normalized.lower_value[0],
            normalized.upper_value[0],
            normalized.bracket_valid[0] & valid,
        )


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

    def __init__(self) -> None:
        self.order = self.derivative_order

    @property
    @abc.abstractmethod
    def derivative_order(self) -> int:
        raise NotImplementedError

    @property
    def method_id(self) -> str:
        return "safeguarded-halley" if self.order == 2 else "safeguarded-newton"

    def solve(
        self,
        problem: ScalarRootProblem,
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ) -> ScalarRootResult:
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        precision_.validate_tolerance(termination.absolute_residual)
        if (
            termination.maximum_evaluations is not None
            and termination.maximum_evaluations < 3
        ):
            raise ValueError(
                "Safeguarded derivative roots require at least three residual "
                "evaluations to establish and certify a bracket."
            )
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

        def condition(value: _OpenRun) -> Array:
            within_evaluations = (
                jnp.asarray(True)
                if termination.maximum_evaluations is None
                else value.evaluations < termination.maximum_evaluations - 1
            )
            return (
                (value.status == int(NonlinearStatus.ITERATING))
                & (value.iterations < termination.maximum_steps)
                & within_evaluations
            )

        def body(value: _OpenRun) -> _OpenRun:
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
        evaluation_exhausted = (
            jnp.asarray(False)
            if termination.maximum_evaluations is None
            else run.evaluations >= termination.maximum_evaluations - 1
        )
        status = jnp.where(
            run.status == int(NonlinearStatus.ITERATING),
            jnp.where(
                evaluation_exhausted,
                int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
                int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            ),
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
    if TYPE_CHECKING:
        __init__ = AbstractSafeguardedDerivativeRoot.__init__

    @property
    def derivative_order(self) -> int:
        return 1


class SafeguardedHalley(AbstractSafeguardedDerivativeRoot):
    if TYPE_CHECKING:
        __init__ = AbstractSafeguardedDerivativeRoot.__init__

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
    iteration: IterationPlan | None = None,
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
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    if iteration is not None and iteration.granularity != "terminal":
        raise ValueError("Scalar roots currently support terminal iteration evidence.")
    result = method_.solve(
        problem,
        termination=termination_,
        args=args,
        precision=precision_,
    )
    from ._newton import _attach_terminal_nonlinear_iteration

    nonlinear = _attach_terminal_nonlinear_iteration(
        result.nonlinear_result,
        iteration,
        method_.method_id,
    )
    return eqx.tree_at(lambda value: value.nonlinear_result, result, nonlinear)


__all__ = [
    "AbstractScalarRootMethod",
    "Bisection",
    "Brent",
    "ITP",
    "Ridder",
    "SafeguardedHalley",
    "SafeguardedNewton",
    "ScalarRootProblem",
    "ScalarRootResult",
    "TOMS748",
    "scalar_root",
    "quadratic_event_bound",
]
