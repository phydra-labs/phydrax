#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import core
from jaxtyping import Array

from .._bounds import Bounds
from .._strict import StrictModule
from ._iterative._types import (
    MinimizationProblem,
    NonlinearConstraint,
    NonlinearLeastSquaresProblem,
    OptimizationStatus,
    OptimizationTermination,
)
from ._least_squares import (
    AbstractBoundedLeastSquaresMethod,
    BoundedGaussNewton,
    least_squares,
)
from ._nonlinear_constraints import _canonical_constraints, _constraint_layout


# Evidence is a fixed-structure JAX PyTree, not a host execution record.
_Response = Callable[[Array, Any], tuple[Array, Array, Any]]


def _positive(value: Any, name: str) -> Array:
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.asarray(1.0).dtype)
    return eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value) | (value <= 0)),
        f"{name} must be finite and strictly positive.",
    )


def _evaluate(callback: _Response, design: Array, args: Any, shape: tuple[int, ...]):
    output = callback(design, args)
    if not isinstance(output, tuple) or len(output) != 3:
        raise TypeError("Response callbacks must return (values, valid, evidence).")
    values, valid, evidence = output
    values, valid = jnp.asarray(values), jnp.asarray(valid)
    if values.shape != shape or not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError(
            "Response values must be a real floating vector matching targets."
        )
    if valid.shape != () or valid.dtype != jnp.bool_:
        raise ValueError("Response validity must be one boolean scalar.")
    return values, valid & jnp.all(jnp.isfinite(values)), evidence


class AnchoredResponseModel(StrictModule):
    """Accepted differentiable predictor with an explicit anchor correction.

    ``predict(design, args)`` returns ``(values, valid, evidence)``. An implicit
    predictor may use native ``implicit_least_squares`` or an accepted state
    solve with implicit derivatives; ``valid`` must include its physical state
    acceptance. Evidence must have a fixed JAX PyTree structure. A failed
    prediction is rejected, never replaced by a physical penalty response.

    Additive correction is ``yk + (m(x) - m(xk))``; multiplicative correction is
    ``yk * (m(x) / m(xk))``. Multiplicative users must supply a strictly positive
    minimum denominator in response units (scalar or response vector). Unsafe
    anchors are rejected; no epsilon is added and the correction never switches.
    """

    predict: _Response
    minimum_denominator: Array | None
    correction: Literal["additive", "multiplicative"] = eqx.field(static=True)

    def __init__(
        self,
        predict: _Response,
        /,
        *,
        correction: Literal["additive", "multiplicative"] = "additive",
        minimum_denominator: Any = None,
    ):
        if not callable(predict):
            raise TypeError("predict must be callable.")
        if correction not in ("additive", "multiplicative"):
            raise ValueError("correction must be additive or multiplicative.")
        if correction == "multiplicative" and minimum_denominator is None:
            raise ValueError("Multiplicative correction requires minimum_denominator.")
        if correction == "additive" and minimum_denominator is not None:
            raise ValueError(
                "minimum_denominator applies only to multiplicative correction."
            )
        self.predict = predict
        self.correction = correction
        self.minimum_denominator = (
            None
            if minimum_denominator is None
            else _positive(minimum_denominator, "minimum_denominator")
        )

    def anchor_valid(self, anchor_prediction: Array, /) -> Array:
        finite = jnp.all(jnp.isfinite(anchor_prediction))
        if self.correction == "additive":
            return finite
        if self.minimum_denominator is None:
            raise RuntimeError(
                "Multiplicative response model lost its denominator bound."
            )
        threshold = jnp.broadcast_to(self.minimum_denominator, anchor_prediction.shape)
        return finite & jnp.all(jnp.abs(anchor_prediction) > threshold)

    def correct(
        self, prediction: Array, anchor_prediction: Array, anchor_values: Array, /
    ) -> Array:
        """Interpolate the anchor exactly; reject unsafe multiplicative anchors."""
        anchor_prediction = eqx.error_if(
            anchor_prediction,
            ~self.anchor_valid(anchor_prediction),
            "Unsafe anchored-response denominator or nonfinite anchor prediction.",
        )
        if self.correction == "additive":
            return anchor_values + (prediction - anchor_prediction)
        return anchor_values * (prediction / anchor_prediction)


class AnchoredTargetProblem(StrictModule):
    """Deterministic callable physical target matching on a bounded design vector.

    ``evaluate(design, args)`` returns ``(responses, valid, evidence)`` in the
    declared ``response_names`` order. Targets and strictly positive physical
    scales are fixed throughout a solve; success uses the maximum absolute
    scaled target error, not inner optimizer stationarity.

    Native ``NonlinearConstraint`` functions receive ``(design, responses)`` as
    their parameter PyTree. They therefore see fine responses for actual merit
    and corrected responses for predicted merit. Canonical rows are equalities,
    then all lower inequalities, then all upper inequalities, with design bounds
    appended after the declared constraints before canonicalization. Inequalities
    are feasible at <= 0. ``constraint_scales`` is a positive scalar or a vector
    in this canonical row order, including the design-bound rows.

    Callbacks and ``args`` must be deterministic or use an explicitly frozen
    realization for the entire solve. Native execution requires pure JAX
    callbacks; the explicit host execution profile permits a host-only physical
    evaluator while keeping the predictor and inner solver native. Stochastic
    resampling and differentiation of the outer trajectory are unsupported.
    """

    evaluate: _Response
    model: AnchoredResponseModel
    targets: Array
    scales: Array
    bounds: Bounds
    constraints: tuple[NonlinearConstraint, ...]
    constraint_scales: Array
    response_names: tuple[str, ...] = eqx.field(static=True)
    realization: Literal["deterministic", "frozen"] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluate: _Response,
        model: AnchoredResponseModel,
        /,
        *,
        targets: Any,
        response_names: Sequence[str],
        scales: Any,
        bounds: Bounds,
        constraints: Sequence[NonlinearConstraint] = (),
        constraint_scales: Any = 1.0,
        realization: Literal["deterministic", "frozen"] = "deterministic",
        problem_id: str = "anchored-target",
    ):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable.")
        if not isinstance(model, AnchoredResponseModel):
            raise TypeError("model must be AnchoredResponseModel.")
        if not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds.")
        targets_ = jnp.asarray(targets)
        if targets_.ndim != 1 or not targets_.size:
            raise ValueError("targets must be a nonempty vector.")
        if not jnp.issubdtype(targets_.dtype, jnp.floating):
            targets_ = targets_.astype(jnp.asarray(1.0).dtype)
        self.targets = eqx.error_if(
            targets_, ~jnp.all(jnp.isfinite(targets_)), "targets must be finite."
        )
        names = tuple(response_names)
        if len(names) != targets_.size or any(
            not isinstance(n, str) or not n for n in names
        ):
            raise ValueError("Every target must have a nonempty response name.")
        if len(set(names)) != len(names):
            raise ValueError("response_names must be unique.")
        constraints_ = tuple(constraints)
        if any(not isinstance(c, NonlinearConstraint) for c in constraints_):
            raise TypeError("constraints must contain NonlinearConstraint values.")
        if realization not in ("deterministic", "frozen"):
            raise ValueError(
                "The realization must be deterministic or explicitly frozen."
            )
        if not problem_id:
            raise ValueError("problem_id must be nonempty.")
        self.evaluate = evaluate
        self.model = model
        self.scales = jnp.broadcast_to(_positive(scales, "scales"), targets_.shape)
        self.bounds = bounds
        self.constraints = constraints_
        self.constraint_scales = _positive(constraint_scales, "constraint_scales")
        self.response_names = names
        self.realization = realization
        self.problem_id = str(problem_id)


class AnchoredTargetMethod(StrictModule):
    """Fixed-anchor, box-trust-region outer policy using native bounded NLS.

    ``maximum_evaluations`` counts physical callback evaluations, including the
    initial point and every rejected or invalid physical trial. Predictor work
    is governed by ``inner_termination`` and reported separately. Trial bounds
    intersect physical bounds with ``anchor +/- radius * design_scales``.
    """

    inner_method: AbstractBoundedLeastSquaresMethod
    inner_termination: OptimizationTermination
    design_scales: Array
    target_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    constraint_weight: float = eqx.field(static=True)
    initial_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    shrink: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    step_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_evaluations: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        inner_method: AbstractBoundedLeastSquaresMethod | None = None,
        inner_termination: OptimizationTermination | None = None,
        design_scales: Any = 1.0,
        target_tolerance: float = 1e-6,
        constraint_tolerance: float = 1e-6,
        constraint_weight: float = 10.0,
        initial_radius: float = 1.0,
        minimum_radius: float = 1e-8,
        maximum_radius: float = 4.0,
        acceptance_ratio: float = 0.1,
        shrink: float = 0.5,
        growth: float = 2.0,
        step_tolerance: float = 1e-10,
        maximum_steps: int = 64,
        maximum_evaluations: int = 32,
    ):
        inner = BoundedGaussNewton() if inner_method is None else inner_method
        termination = (
            OptimizationTermination(
                absolute_optimality=1e-9,
                relative_optimality=0.0,
                maximum_steps=64,
            )
            if inner_termination is None
            else inner_termination
        )
        if not isinstance(inner, AbstractBoundedLeastSquaresMethod):
            raise TypeError("inner_method must be a native bounded least-squares method.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("inner_termination must be OptimizationTermination.")
        positive = (
            target_tolerance,
            constraint_tolerance,
            constraint_weight,
            initial_radius,
            minimum_radius,
            maximum_radius,
            step_tolerance,
        )
        if any(not isfinite(v) or v <= 0 for v in positive):
            raise ValueError(
                "Tolerances, constraint weight and radii must be positive and finite."
            )
        if not minimum_radius <= initial_radius <= maximum_radius:
            raise ValueError("Radii must satisfy minimum <= initial <= maximum.")
        if not 0 < acceptance_ratio < 1 or not 0 < shrink < 1:
            raise ValueError("acceptance_ratio and shrink must lie in (0, 1).")
        if not isfinite(growth) or growth <= 1:
            raise ValueError("growth must be finite and greater than one.")
        if maximum_steps < 1 or maximum_evaluations < 1:
            raise ValueError("Evaluation and step budgets must be positive.")
        self.inner_method = inner
        self.inner_termination = termination
        self.design_scales = _positive(design_scales, "design_scales")
        self.target_tolerance = float(target_tolerance)
        self.constraint_tolerance = float(constraint_tolerance)
        self.constraint_weight = float(constraint_weight)
        self.initial_radius = float(initial_radius)
        self.minimum_radius = float(minimum_radius)
        self.maximum_radius = float(maximum_radius)
        self.acceptance_ratio = float(acceptance_ratio)
        self.shrink = float(shrink)
        self.growth = float(growth)
        self.step_tolerance = float(step_tolerance)
        self.maximum_steps = int(maximum_steps)
        self.maximum_evaluations = int(maximum_evaluations)


class AnchoredTargetResult(StrictModule):
    """Last accepted physical anchor, never an unaccepted inner/trial solution.

    ``accepted`` certifies physical evaluation validity, not target achievement.
    Only ``successful`` means both target tolerance and feasibility were met.
    History has fixed capacity; only its first ``iterations`` rows are populated.
    Rejected trial evidence is kept separately from accepted-anchor evidence.
    ``inner_residual_evaluations`` uses native NLS diagnostics; it does not count
    JVP/VJP applications as additional physical evaluations.
    """

    design: Array
    values: Array
    scaled_residual: Array
    equality: Array
    inequality: Array
    merit: Array
    target_error: Array
    constraint_violation: Array
    accepted: Array
    status: Array
    iterations: Array
    evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    inner_residual_evaluations: Array
    radius: Array
    fine_evidence: Any
    model_evidence: Any
    last_trial_evidence: Any
    last_trial_model_evidence: Any
    history: dict[str, Array]
    response_names: tuple[str, ...] = eqx.field(static=True)
    equality_sources: tuple[str, ...] = eqx.field(static=True)
    inequality_sources: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted & (self.status == int(OptimizationStatus.SUCCESS))


class _Run(NamedTuple):
    design: Array
    values: Array
    prediction: Array
    valid: Array
    predictor_valid: Array
    evidence: Any
    model_evidence: Any
    trial_evidence: Any
    trial_model_evidence: Any
    merit: Array
    radius: Array
    status: Array
    iterations: Array
    evaluations: Array
    accepted_steps: Array
    inner_evaluations: Array
    history: dict[str, Array]


def solve_anchored_target(
    problem: AnchoredTargetProblem,
    initial_design: Any,
    /,
    *,
    method: AnchoredTargetMethod | None = None,
    args: Any = None,
    execution: Literal["native", "host"] = "native",
) -> AnchoredTargetResult:
    """Match callable physical targets without differentiating the outer solve.

    Each inner solve freezes its accepted fine/model anchor. Actual and predicted
    reduction use exactly the same fixed scaled least-squares merit, including
    physical constraints. Invalid evaluations and non-improving trials shrink
    the trust box and retain every accepted-anchor value and evidence leaf.

    ``execution="native"`` uses a JAX outer loop and can be JIT compiled.
    ``execution="host"`` uses the same acceptance policy with host control flow:
    the physical evaluator is called outside tracing, while the differentiable
    predictor and bounded inner solve remain native. Host exceptions propagate;
    expected physical solve failures must be returned as ``valid=False``.
    """
    if not isinstance(problem, AnchoredTargetProblem):
        raise TypeError("problem must be AnchoredTargetProblem.")
    method_ = AnchoredTargetMethod() if method is None else method
    if not isinstance(method_, AnchoredTargetMethod):
        raise TypeError("method must be AnchoredTargetMethod.")
    if execution not in ("native", "host"):
        raise ValueError("execution must be native or host.")
    design = jnp.asarray(initial_design)
    if (
        design.ndim != 1
        or not design.size
        or not jnp.issubdtype(design.dtype, jnp.floating)
    ):
        raise ValueError("initial_design must be a nonempty real floating vector.")
    if execution == "host" and isinstance(design, core.Tracer):
        raise ValueError("Host anchored-target execution cannot be JIT transformed.")
    design = eqx.error_if(
        design,
        ~jnp.all(jnp.isfinite(design)) | ~problem.bounds.contains(design),
        "initial_design must be finite and satisfy its physical bounds.",
    )
    design = jax.lax.stop_gradient(design)
    lower, upper = problem.bounds.materialize(design)
    design_scales = jnp.broadcast_to(method_.design_scales, design.shape)
    targets = jax.lax.stop_gradient(problem.targets)
    scales = jax.lax.stop_gradient(problem.scales)
    values, valid, evidence = _evaluate(problem.evaluate, design, args, targets.shape)
    prediction, predictor_valid, model_evidence = _evaluate(
        problem.model.predict, design, args, targets.shape
    )
    constraints = MinimizationProblem(
        lambda point, args: jnp.asarray(0.0, dtype=design.dtype),
        constraints=problem.constraints
        + (
            NonlinearConstraint(
                lambda point, args: point[0],
                lower=problem.bounds.lower,
                upper=problem.bounds.upper,
                constraint_id="design-bounds",
            ),
        ),
    )
    layout = _constraint_layout(constraints, (design, values), args)
    n_equal = layout.equality_indices.size
    n_inequal = layout.lower_indices.size + layout.upper_indices.size
    constraint_scales = jax.lax.stop_gradient(
        jnp.broadcast_to(problem.constraint_scales, (n_equal + n_inequal,))
    )

    def physical(candidate, responses):
        equality, inequality = _canonical_constraints(
            constraints, layout, (candidate, responses), args
        )
        scaled_equality = equality / constraint_scales[:n_equal]
        scaled_inequality = inequality / constraint_scales[n_equal:]
        violation = jnp.max(
            jnp.concatenate(
                (
                    jnp.zeros((1,), dtype=responses.dtype),
                    jnp.abs(scaled_equality),
                    jnp.maximum(scaled_inequality, 0.0),
                )
            )
        )
        residual = jnp.concatenate(
            (
                (responses - targets) / scales,
                jnp.sqrt(method_.constraint_weight) * scaled_equality,
                jnp.sqrt(method_.constraint_weight) * jnp.maximum(scaled_inequality, 0.0),
            )
        )
        finite = jnp.all(jnp.isfinite(equality)) & jnp.all(jnp.isfinite(inequality))
        return residual, equality, inequality, violation, finite

    def merit(candidate, responses):
        residual, _, _, _, _ = physical(candidate, responses)
        return 0.5 * jnp.sum(jnp.square(residual))

    def target_met(candidate, responses):
        _, _, _, violation, finite = physical(candidate, responses)
        return (
            finite
            & (violation <= method_.constraint_tolerance)
            & (
                jnp.max(jnp.abs((responses - targets) / scales))
                <= method_.target_tolerance
            )
        )

    valid = valid & physical(design, values)[4]
    initial_status = jnp.where(
        ~valid,
        int(OptimizationStatus.NONFINITE_EVALUATION),
        jnp.where(
            target_met(design, values),
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.ITERATING),
        ),
    ).astype(jnp.int32)
    capacity = method_.maximum_steps
    initial_merit = merit(design, values)
    nan = jnp.asarray(jnp.nan, dtype=initial_merit.dtype)
    history = {
        "design": jnp.full((capacity, design.size), jnp.nan, dtype=design.dtype),
        "anchor_design": jnp.full((capacity, design.size), jnp.nan, dtype=design.dtype),
        "radius": jnp.full((capacity,), jnp.nan, dtype=design.dtype),
        "predicted_reduction": jnp.full((capacity,), nan),
        "actual_reduction": jnp.full((capacity,), nan),
        "ratio": jnp.full((capacity,), nan),
        "fine_evaluated": jnp.zeros((capacity,), dtype=jnp.bool_),
        "fine_valid": jnp.zeros((capacity,), dtype=jnp.bool_),
        "accepted": jnp.zeros((capacity,), dtype=jnp.bool_),
        "inner_status": jnp.full((capacity,), -1, dtype=jnp.int32),
    }
    zero = jnp.asarray(0, dtype=jnp.int32)
    run = _Run(
        design,
        values,
        prediction,
        valid,
        predictor_valid,
        evidence,
        model_evidence,
        evidence,
        model_evidence,
        initial_merit,
        jnp.asarray(method_.initial_radius, dtype=design.dtype),
        initial_status,
        zero,
        jnp.asarray(1, dtype=jnp.int32),
        zero,
        zero,
        history,
    )

    def branch(predicate, true, false, operand=None):
        if execution == "host":
            return true(operand) if bool(predicate) else false(operand)
        return jax.lax.cond(predicate, true, false, operand)

    def condition(current):
        return (
            (current.status == int(OptimizationStatus.ITERATING))
            & (current.iterations < method_.maximum_steps)
            & (current.evaluations < method_.maximum_evaluations)
        )

    def body(current):
        anchor_safe = current.predictor_valid & problem.model.anchor_valid(
            current.prediction
        )

        def attempt(_):
            anchor_prediction = jax.lax.stop_gradient(current.prediction)
            anchor_values = jax.lax.stop_gradient(current.values)

            def residual(candidate, unused):
                raw, model_valid, model_aux = _evaluate(
                    problem.model.predict, candidate, args, targets.shape
                )
                corrected = problem.model.correct(raw, anchor_prediction, anchor_values)
                residual_, _, _, _, constraints_finite = physical(candidate, corrected)
                model_valid = (
                    model_valid & constraints_finite & jnp.all(jnp.isfinite(corrected))
                )
                # NaN signals an unavailable residual to native globalization; it
                # is not a fabricated physical value or a merit penalty fallback.
                return jnp.where(model_valid, residual_, jnp.nan), (
                    raw,
                    model_valid,
                    model_aux,
                )

            inner = least_squares(
                NonlinearLeastSquaresProblem(
                    residual,
                    has_aux=True,
                    bounds=Bounds(
                        jnp.maximum(
                            lower, current.design - current.radius * design_scales
                        ),
                        jnp.minimum(
                            upper, current.design + current.radius * design_scales
                        ),
                    ),
                    problem_id=f"{problem.problem_id}/fixed-anchor",
                ),
                current.design,
                method=method_.inner_method,
                termination=method_.inner_termination,
            )
            candidate = inner.parameters
            raw, model_valid, trial_model_evidence = inner.auxiliary
            predicted = current.merit - inner.objective
            step = jnp.max(jnp.abs((candidate - current.design) / design_scales))
            evaluate_fine = (
                inner.successful
                & model_valid
                & jnp.all(jnp.isfinite(candidate))
                & jnp.isfinite(predicted)
                & (predicted > 0)
                & (step > method_.step_tolerance)
            )
            trial_values, trial_valid, trial_evidence = branch(
                evaluate_fine,
                lambda _: _evaluate(problem.evaluate, candidate, args, targets.shape),
                lambda _: (current.values, jnp.asarray(False), current.trial_evidence),
                operand=None,
            )
            trial_valid = trial_valid & physical(candidate, trial_values)[4]
            trial_merit = merit(candidate, trial_values)
            actual = jnp.where(evaluate_fine, current.merit - trial_merit, nan)
            ratio = branch(
                evaluate_fine, lambda _: actual / predicted, lambda _: nan, None
            )
            accepted = (
                evaluate_fine
                & trial_valid
                & jnp.isfinite(actual)
                & jnp.isfinite(ratio)
                & (actual > 0)
                & (ratio >= method_.acceptance_ratio)
            )
            select = lambda old, new: jax.tree.map(
                lambda a, b: jnp.where(accepted, b, a), old, new
            )
            next_design = select(current.design, candidate)
            next_values = select(current.values, trial_values)
            next_radius = jnp.where(
                accepted,
                jnp.where(
                    (ratio > 0.75) & (step >= 0.8 * current.radius),
                    jnp.minimum(method_.maximum_radius, method_.growth * current.radius),
                    current.radius,
                ),
                method_.shrink * current.radius,
            )
            stalled = inner.successful & (step <= method_.step_tolerance)
            next_status = jnp.where(
                accepted & target_met(next_design, next_values),
                int(OptimizationStatus.SUCCESS),
                jnp.where(
                    stalled,
                    int(OptimizationStatus.STAGNATION),
                    jnp.where(
                        next_radius < method_.minimum_radius,
                        int(OptimizationStatus.TRUST_REGION_FAILED),
                        int(OptimizationStatus.ITERATING),
                    ),
                ),
            ).astype(jnp.int32)
            row = {
                "design": candidate,
                "anchor_design": current.design,
                "radius": current.radius,
                "predicted_reduction": predicted,
                "actual_reduction": actual,
                "ratio": ratio,
                "fine_evaluated": evaluate_fine,
                "fine_valid": trial_valid,
                "accepted": accepted,
                "inner_status": inner.status,
            }
            next_history = {
                name: data.at[current.iterations].set(row[name])
                for name, data in current.history.items()
            }
            return _Run(
                next_design,
                next_values,
                select(current.prediction, raw),
                current.valid,
                select(current.predictor_valid, model_valid),
                select(current.evidence, trial_evidence),
                select(current.model_evidence, trial_model_evidence),
                trial_evidence,
                trial_model_evidence,
                jnp.where(accepted, trial_merit, current.merit),
                next_radius,
                next_status,
                current.iterations + 1,
                current.evaluations + evaluate_fine.astype(jnp.int32),
                current.accepted_steps + accepted.astype(jnp.int32),
                current.inner_evaluations + inner.diagnostics.residual_evaluations,
                next_history,
            )

        return branch(
            anchor_safe,
            attempt,
            lambda _: current._replace(
                status=jnp.asarray(
                    int(OptimizationStatus.CERTIFICATION_FAILED), dtype=jnp.int32
                )
            ),
            operand=None,
        )

    if execution == "host":
        while bool(condition(run)):
            run = body(run)
    else:
        run = jax.lax.while_loop(condition, body, run)
    status = jnp.where(
        run.status == int(OptimizationStatus.ITERATING),
        jnp.where(
            run.evaluations >= method_.maximum_evaluations,
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        ),
        run.status,
    ).astype(jnp.int32)
    _, equality, inequality, violation, _ = physical(run.design, run.values)
    result = AnchoredTargetResult(
        design=run.design,
        values=run.values,
        scaled_residual=(run.values - targets) / scales,
        equality=equality,
        inequality=inequality,
        merit=run.merit,
        target_error=jnp.max(jnp.abs((run.values - targets) / scales)),
        constraint_violation=violation,
        accepted=run.valid,
        status=status,
        iterations=run.iterations,
        evaluations=run.evaluations,
        accepted_steps=run.accepted_steps,
        rejected_steps=run.iterations - run.accepted_steps,
        inner_residual_evaluations=run.inner_evaluations,
        radius=run.radius,
        fine_evidence=run.evidence,
        model_evidence=run.model_evidence,
        last_trial_evidence=run.trial_evidence,
        last_trial_model_evidence=run.trial_model_evidence,
        history=run.history,
        response_names=problem.response_names,
        equality_sources=layout.equality_sources,
        inequality_sources=layout.inequality_sources,
    )
    return jax.tree.map(jax.lax.stop_gradient, result)


__all__ = [
    "AnchoredResponseModel",
    "AnchoredTargetMethod",
    "AnchoredTargetProblem",
    "AnchoredTargetResult",
    "solve_anchored_target",
]
