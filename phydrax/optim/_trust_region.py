#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import AbstractLinearOperator, AbstractPreconditioner


class TrustRegionSubproblemStatus(IntEnum):
    """Terminal status of an operator trust-region quadratic solve."""

    CONVERGED = 0
    BOUNDARY_REACHED = 1
    NEGATIVE_CURVATURE = 2
    MAXIMUM_STEPS_REACHED = 3
    NONFINITE_EVALUATION = 4
    INVALID_MODEL = 5


class TrustRegionQuadraticProblem(StrictModule):
    """Self-adjoint quadratic model with one trust-region radius."""

    hessian: AbstractLinearOperator
    gradient: PyTree[Array]
    radius: Array

    def __init__(
        self,
        hessian: AbstractLinearOperator,
        gradient: PyTree[Any],
        radius: Any,
        /,
    ):
        if not isinstance(hessian, AbstractLinearOperator):
            raise TypeError("hessian must be AbstractLinearOperator.")
        if hessian.batch_shape:
            raise ValueError("Trust-region Hessians must be unbatched.")
        if not hessian.source.compatible(hessian.target):
            raise ValueError("Trust-region Hessians must be endomorphisms.")
        if not hessian.properties.certifies("self_adjoint"):
            raise ValueError(
                "Trust-region Hessians require certified self-adjoint structure."
            )
        gradient_ = hessian.source.validate(gradient)
        radius_ = jnp.asarray(radius)
        if radius_.shape != ():
            raise ValueError("Trust-region radius must be scalar.")
        radius_ = eqx.error_if(
            radius_,
            ~jnp.isfinite(radius_) | (radius_ <= 0.0),
            "Trust-region radius must be finite and positive.",
        )
        self.hessian = hessian
        self.gradient = gradient_
        self.radius = radius_


class SteihaugToint(StrictModule):
    """Preconditioned truncated CG for a self-adjoint trust-region model."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    curvature_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1e-7,
        absolute_tolerance: float = 1e-10,
        maximum_steps: int = 200,
        curvature_tolerance: float = 0.0,
    ):
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        steps = int(maximum_steps)
        curvature = float(curvature_tolerance)
        if any(
            not isfinite(value) or value < 0.0
            for value in (relative, absolute, curvature)
        ):
            raise ValueError("Trust-region tolerances must be finite and non-negative.")
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.maximum_steps = steps
        self.curvature_tolerance = curvature


class TrustRegionSubproblemDiagnostics(StrictModule):
    """Operator work and model evidence from a trust-region solve."""

    iterations: Array
    hessian_actions: Array
    preconditioner_applications: Array
    initial_residual_norm: Array
    final_residual_norm: Array
    step_norm: Array
    predicted_reduction: Array
    minimum_curvature: Array
    boundary_hit: Array
    negative_curvature: Array


class TrustRegionSubproblemResult(StrictModule):
    """Trust-region step with explicit termination evidence."""

    step: PyTree[Array]
    status: Array
    diagnostics: TrustRegionSubproblemDiagnostics
    precision_evidence: PrecisionEvidenceEnvelope

    @property
    def successful(self) -> Array:
        return (
            (self.status == int(TrustRegionSubproblemStatus.CONVERGED))
            | (self.status == int(TrustRegionSubproblemStatus.BOUNDARY_REACHED))
            | (self.status == int(TrustRegionSubproblemStatus.NEGATIVE_CURVATURE))
        )


class _TrustRegionRun(StrictModule):
    step: PyTree[Array]
    residual: PyTree[Array]
    preconditioned_residual: PyTree[Array]
    direction: PyTree[Array]
    residual_pairing: Array
    residual_norm: Array
    iteration: Array
    hessian_actions: Array
    preconditioner_applications: Array
    minimum_curvature: Array
    status: Array


def _inner(space, left, right, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.real(precision.inner(space, left, right)))


def _norm(space, value, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.norm(space, value)


def _add_scaled(base, direction, scale, /):
    return jax.tree.map(lambda x, d: x + scale * d, base, direction)


def _negative(value, /):
    return jax.tree.map(jnp.negative, value)


def _apply_preconditioner(preconditioner, residual, /):
    return residual if preconditioner is None else preconditioner.apply(residual)


def _boundary_rate(
    space,
    step,
    direction,
    radius,
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    a = _inner(space, direction, direction, precision)
    b = 2.0 * _inner(space, step, direction, precision)
    c = _inner(space, step, step, precision) - radius * radius
    discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    tiny = jnp.finfo(a.dtype).tiny
    return (-b + jnp.sqrt(discriminant)) / jnp.maximum(2.0 * a, tiny)


def solve_trust_region_subproblem(
    problem: TrustRegionQuadraticProblem,
    /,
    *,
    method: SteihaugToint | None = None,
    preconditioner: AbstractPreconditioner | None = None,
    precision: NonlinearPrecisionPolicy | None = None,
) -> TrustRegionSubproblemResult:
    """Solve one matrix-free quadratic trust-region subproblem."""
    if not isinstance(problem, TrustRegionQuadraticProblem):
        raise TypeError("problem must be TrustRegionQuadraticProblem.")
    method_ = SteihaugToint() if method is None else method
    if not isinstance(method_, SteihaugToint):
        raise TypeError("method must be SteihaugToint or None.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be a NonlinearPrecisionPolicy or None.")
    space = problem.hessian.source
    if preconditioner is not None:
        if not isinstance(preconditioner, AbstractPreconditioner):
            raise TypeError("preconditioner must be AbstractPreconditioner or None.")
        if not preconditioner.space.compatible(space):
            raise ValueError("Trust-region preconditioner space is incompatible.")
        properties = preconditioner.properties
        if not (
            properties.certifies("linear")
            and properties.certifies("stationary")
            and properties.certifies("self_adjoint")
            and properties.certifies("positive_definite")
        ):
            raise ValueError(
                "Steihaug-Toint preconditioning requires fixed linear SPD evidence."
            )
    gradient = space.validate(problem.gradient)
    precision_.validate_trees(gradient, gradient)
    precision_.validate_accumulation_space(space)
    initial_norm = _norm(space, gradient, precision_)
    threshold = precision_.decision(
        method_.absolute_tolerance + method_.relative_tolerance * initial_norm
    )
    radius = precision_.decision(problem.radius)
    preconditioned = space.validate(_apply_preconditioner(preconditioner, gradient))
    pairing = _inner(space, gradient, preconditioned, precision_)
    finite = (
        tree_allfinite(gradient)
        & tree_allfinite(preconditioned)
        & jnp.isfinite(initial_norm)
        & jnp.isfinite(pairing)
    )
    initially_converged = finite & (initial_norm <= threshold)
    initial_status = jnp.where(
        ~finite,
        int(TrustRegionSubproblemStatus.NONFINITE_EVALUATION),
        jnp.where(
            initially_converged,
            int(TrustRegionSubproblemStatus.CONVERGED),
            int(TrustRegionSubproblemStatus.INVALID_MODEL)
            if preconditioner is not None
            else int(TrustRegionSubproblemStatus.MAXIMUM_STEPS_REACHED),
        ),
    ).astype(jnp.int32)
    initial_status = jnp.where(
        finite & ~initially_converged & (pairing > 0.0),
        -1,
        initial_status,
    ).astype(jnp.int32)
    run = _TrustRegionRun(
        step=space.zeros(),
        residual=gradient,
        preconditioned_residual=preconditioned,
        direction=_negative(preconditioned),
        residual_pairing=pairing,
        residual_norm=initial_norm,
        iteration=jnp.asarray(0, dtype=jnp.int32),
        hessian_actions=jnp.asarray(0, dtype=jnp.int32),
        preconditioner_applications=jnp.asarray(
            0 if preconditioner is None else 1,
            dtype=jnp.int32,
        ),
        minimum_curvature=jnp.asarray(jnp.inf, dtype=initial_norm.dtype),
        status=initial_status,
    )

    def condition(current):
        return (current.status == -1) & (current.iteration < method_.maximum_steps)

    def body(current):
        hessian_direction = space.validate(problem.hessian.mv(current.direction))
        curvature = _inner(
            space,
            current.direction,
            hessian_direction,
            precision_,
        )
        direction_norm_squared = _inner(
            space,
            current.direction,
            current.direction,
            precision_,
        )
        tiny = jnp.finfo(curvature.dtype).tiny
        normalized_curvature = curvature / jnp.maximum(direction_norm_squared, tiny)
        nonfinite = (
            ~tree_allfinite(hessian_direction)
            | ~jnp.isfinite(curvature)
            | ~jnp.isfinite(normalized_curvature)
        )
        negative = normalized_curvature <= method_.curvature_tolerance
        alpha = jnp.where(
            negative | nonfinite,
            0.0,
            current.residual_pairing / jnp.maximum(curvature, tiny),
        )
        unconstrained = _add_scaled(current.step, current.direction, alpha)
        crosses = _norm(space, unconstrained, precision_) >= radius
        boundary = negative | crosses
        rate = _boundary_rate(
            space,
            current.step,
            current.direction,
            radius,
            precision_,
        )
        boundary_step = _add_scaled(current.step, current.direction, rate)
        trial_step = jax.tree.map(
            lambda bounded, free: jnp.where(boundary, bounded, free),
            boundary_step,
            unconstrained,
        )
        trial_residual = _add_scaled(
            current.residual,
            hessian_direction,
            alpha,
        )
        trial_preconditioned = space.validate(
            _apply_preconditioner(preconditioner, trial_residual)
        )
        trial_pairing = _inner(
            space,
            trial_residual,
            trial_preconditioned,
            precision_,
        )
        trial_norm = _norm(space, trial_residual, precision_)
        converged = (~boundary) & (trial_norm <= threshold)
        usable_pairing = jnp.isfinite(trial_pairing) & (trial_pairing > 0.0)
        beta = trial_pairing / jnp.maximum(current.residual_pairing, tiny)
        next_direction = jax.tree.map(
            lambda z, d: -z + beta * d,
            trial_preconditioned,
            current.direction,
        )
        status = jnp.where(
            nonfinite,
            int(TrustRegionSubproblemStatus.NONFINITE_EVALUATION),
            jnp.where(
                negative,
                int(TrustRegionSubproblemStatus.NEGATIVE_CURVATURE),
                jnp.where(
                    crosses,
                    int(TrustRegionSubproblemStatus.BOUNDARY_REACHED),
                    jnp.where(
                        converged,
                        int(TrustRegionSubproblemStatus.CONVERGED),
                        jnp.where(
                            usable_pairing,
                            -1,
                            int(TrustRegionSubproblemStatus.INVALID_MODEL),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return _TrustRegionRun(
            step=jax.tree.map(
                lambda new, old: jnp.where(nonfinite, old, new),
                trial_step,
                current.step,
            ),
            residual=jax.tree.map(
                lambda new, old: jnp.where(boundary | nonfinite, old, new),
                trial_residual,
                current.residual,
            ),
            preconditioned_residual=jax.tree.map(
                lambda new, old: jnp.where(boundary | nonfinite, old, new),
                trial_preconditioned,
                current.preconditioned_residual,
            ),
            direction=jax.tree.map(
                lambda new, old: jnp.where(boundary | nonfinite, old, new),
                next_direction,
                current.direction,
            ),
            residual_pairing=jnp.where(
                boundary | nonfinite,
                current.residual_pairing,
                trial_pairing,
            ),
            residual_norm=jnp.where(
                boundary | nonfinite,
                current.residual_norm,
                trial_norm,
            ),
            iteration=current.iteration + 1,
            hessian_actions=current.hessian_actions + 1,
            preconditioner_applications=current.preconditioner_applications
            + (0 if preconditioner is None else 1),
            minimum_curvature=jnp.minimum(
                current.minimum_curvature,
                normalized_curvature,
            ),
            status=status,
        )

    run = jax.lax.while_loop(condition, body, run)
    status = jnp.where(
        run.status == -1,
        int(TrustRegionSubproblemStatus.MAXIMUM_STEPS_REACHED),
        run.status,
    ).astype(jnp.int32)
    hessian_step = space.validate(problem.hessian.mv(run.step))
    predicted_reduction = -(
        _inner(space, gradient, run.step, precision_)
        + 0.5 * _inner(space, run.step, hessian_step, precision_)
    )
    final_finite = tree_allfinite(run.step) & jnp.isfinite(predicted_reduction)
    status = jnp.where(
        ~final_finite,
        int(TrustRegionSubproblemStatus.NONFINITE_EVALUATION),
        status,
    ).astype(jnp.int32)
    diagnostics = TrustRegionSubproblemDiagnostics(
        iterations=run.iteration,
        hessian_actions=run.hessian_actions + 1,
        preconditioner_applications=run.preconditioner_applications,
        initial_residual_norm=initial_norm,
        final_residual_norm=run.residual_norm,
        step_norm=_norm(space, run.step, precision_),
        predicted_reduction=predicted_reduction,
        minimum_curvature=run.minimum_curvature,
        boundary_hit=(
            (status == int(TrustRegionSubproblemStatus.BOUNDARY_REACHED))
            | (status == int(TrustRegionSubproblemStatus.NEGATIVE_CURVATURE))
        ),
        negative_curvature=(
            status == int(TrustRegionSubproblemStatus.NEGATIVE_CURVATURE)
        ),
    )
    output_step = jax.tree.map(precision_.output, run.step)
    precision_evidence = precision_.evidence_for(run.step, gradient)
    return TrustRegionSubproblemResult(
        step=output_step,
        status=status,
        diagnostics=diagnostics,
        precision_evidence=precision_evidence,
    )


__all__ = [
    "SteihaugToint",
    "TrustRegionQuadraticProblem",
    "TrustRegionSubproblemDiagnostics",
    "TrustRegionSubproblemResult",
    "TrustRegionSubproblemStatus",
    "solve_trust_region_subproblem",
]
