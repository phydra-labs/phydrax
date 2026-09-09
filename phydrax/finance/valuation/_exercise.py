#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Regression Monte Carlo and reflected-BSDE early-exercise routes."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, LinearSystem, solve
from ...stochastic._bsde import BSDEPathBatch
from ...stochastic._path_dependent_bsde import ReflectedPathDependentBSDEProblem
from ..contracts._options import BermudanPayoff, OptionType
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw
from ._monte_carlo import MonteCarloPathBatch
from ._types import ValuationEvidence, ValuationResult


class LSMPlan(StrictModule):
    polynomial_degree: int = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    minimum_in_the_money_paths: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        polynomial_degree: int = 3,
        ridge: float = 1.0e-10,
        minimum_in_the_money_paths: int = 16,
    ):
        if (
            isinstance(polynomial_degree, bool)
            or not isinstance(polynomial_degree, int)
            or polynomial_degree < 1
        ):
            raise ValueError("polynomial_degree must be a positive integer.")
        if (
            isinstance(minimum_in_the_money_paths, bool)
            or not isinstance(minimum_in_the_money_paths, int)
            or minimum_in_the_money_paths < polynomial_degree + 1
        ):
            raise ValueError(
                "minimum_in_the_money_paths must exceed the regression basis size."
            )
        ridge_ = float(ridge)
        if not isfinite(ridge_) or ridge_ <= 0.0:
            raise ValueError("ridge must be finite and strictly positive.")
        self.polynomial_degree = polynomial_degree
        self.ridge = ridge_
        self.minimum_in_the_money_paths = minimum_in_the_money_paths


class LSMProblem(StrictModule):
    paths: MonteCarloPathBatch
    exercise_mask: Array
    payoff: BermudanPayoff
    rate: Array
    currency: Currency | None = eqx.field(static=True)
    evidence_binding: FinanceEvidenceBinding | None
    pricing_law: PricingLaw | None

    def __init__(
        self,
        paths: MonteCarloPathBatch,
        exercise_mask: ArrayLike,
        payoff: BermudanPayoff,
        rate: ArrayLike,
        /,
        *,
        currency: Currency | None = None,
        evidence_binding: FinanceEvidenceBinding | None = None,
        pricing_law: PricingLaw | None = None,
    ):
        if not isinstance(paths, MonteCarloPathBatch) or paths.asset_count != 1:
            raise ValueError("LSM requires a one-asset MonteCarloPathBatch.")
        if not isinstance(payoff, BermudanPayoff):
            raise TypeError("payoff must be BermudanPayoff.")
        mask = jnp.asarray(exercise_mask, dtype=bool)
        if mask.shape != (paths.time_count,):
            raise ValueError("exercise_mask must contain one flag per path time.")
        mask = eqx.error_if(
            mask,
            ~mask[-1] | (jnp.sum(mask) < 2),
            "Bermudan LSM requires maturity and at least one earlier exercise date.",
        )
        rate_ = jnp.asarray(rate, dtype=float)
        if rate_.shape != ():
            raise ValueError("rate must be scalar.")
        rate_ = eqx.error_if(rate_, ~jnp.isfinite(rate_), "rate must be finite.")
        self.paths, self.exercise_mask, self.payoff, self.rate = (
            paths,
            mask,
            payoff,
            rate_,
        )
        self.currency, self.evidence_binding, self.pricing_law = (
            currency,
            evidence_binding,
            pricing_law,
        )


class LSMDiagnostics(StrictModule):
    exercise_time_index: Array
    exercise_counts: Array
    regression_successful: Array
    standard_error: Array
    basis_size: int = eqx.field(static=True)


def _intrinsic(payoff: BermudanPayoff, spots: Array) -> Array:
    sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
    return payoff.notional * jnp.maximum(sign * (spots - payoff.strike), 0.0)


def _polynomial_basis(spots: Array, strike: Array, degree: int) -> Array:
    scaled = spots / strike
    powers = jnp.arange(degree + 1, dtype=spots.dtype)
    return scaled[:, None] ** powers[None, :]


def _masked_regression(features: Array, targets: Array, mask: Array, ridge: float):
    weights = mask.astype(features.dtype)
    gram = features.T @ (weights[:, None] * features) + ridge * jnp.eye(
        features.shape[1], dtype=features.dtype
    )
    rhs = features.T @ (weights * targets)
    result = solve(LinearSystem(DenseLinearOperator(gram)), rhs)
    return features @ result.value, result.successful


def evaluate_bermudan_lsm(problem: LSMProblem, plan: LSMPlan, /) -> ValuationResult:
    """Longstaff-Schwartz stopping with native dense regression solves."""

    if not isinstance(problem, LSMProblem) or not isinstance(plan, LSMPlan):
        raise TypeError("problem and plan must be LSM records.")
    spots = problem.paths.values[:, :, 0]
    valid_paths = problem.paths.path_valid
    maturity_index = problem.paths.time_count - 1
    cashflow = _intrinsic(problem.payoff, spots[:, maturity_index])
    exercise_index = jnp.full(
        (problem.paths.path_count,), maturity_index, dtype=jnp.int32
    )
    exercise_counts = (
        jnp.zeros((problem.paths.time_count,), dtype=jnp.int32)
        .at[maturity_index]
        .set(jnp.sum(valid_paths))
    )
    regression_successful = jnp.asarray(True)
    for time_index in range(maturity_index - 1, -1, -1):
        intrinsic = _intrinsic(problem.payoff, spots[:, time_index])
        itm = valid_paths & (intrinsic > 0.0)
        discount_to_time = jnp.exp(
            -problem.rate
            * (problem.paths.times[exercise_index] - problem.paths.times[time_index])
        )
        target = cashflow * discount_to_time
        features = _polynomial_basis(
            spots[:, time_index], problem.payoff.strike, plan.polynomial_degree
        )
        continuation, solved = _masked_regression(features, target, itm, plan.ridge)
        itm_count = jnp.sum(itm)
        has_itm = itm_count > 0
        enough = itm_count >= plan.minimum_in_the_money_paths
        regression_ok = ~has_itm | (enough & solved)
        allowed = problem.exercise_mask[time_index] & enough & solved
        exercise = allowed & itm & (intrinsic > continuation)
        cashflow = jnp.where(exercise, intrinsic, cashflow)
        exercise_index = jnp.where(exercise, time_index, exercise_index)
        exercise_counts = exercise_counts.at[time_index].set(jnp.sum(exercise))
        regression_successful = regression_successful & (
            ~problem.exercise_mask[time_index] | regression_ok
        )
    discounted = cashflow * jnp.exp(
        -problem.rate * (problem.paths.times[exercise_index] - problem.paths.times[0])
    )
    valid_count = jnp.sum(valid_paths)
    value = jnp.sum(jnp.where(valid_paths, discounted, 0.0)) / valid_count
    centered = jnp.where(valid_paths, discounted - value, 0.0)
    sample_variance = jnp.sum(centered**2) / jnp.maximum(valid_count - 1, 1)
    standard_error = jnp.sqrt(sample_variance / valid_count)
    value = eqx.error_if(
        value,
        ~regression_successful | ~jnp.isfinite(value) | ~jnp.isfinite(standard_error),
        "Bermudan LSM regression was singular or undersampled.",
    )
    evidence = ValuationEvidence(
        route="bermudan-longstaff-schwartz",
        finite=jnp.isfinite(value),
        converged=regression_successful,
        exercise_consistent=jnp.all(problem.exercise_mask[exercise_index]),
        error_estimate=standard_error,
        binding=problem.evidence_binding,
        pricing_law=problem.pricing_law,
    )
    diagnostics = LSMDiagnostics(
        exercise_index,
        exercise_counts,
        regression_successful,
        standard_error,
        plan.polynomial_degree + 1,
    )
    return ValuationResult(
        value,
        evidence,
        standard_error=standard_error,
        currency=problem.currency,
        diagnostics=diagnostics,
    )


class ReflectedBSDEPlan(StrictModule):
    ridge: float = eqx.field(static=True)
    minimum_paths: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        ridge: float = 1.0e-8,
        minimum_paths: int = 32,
        residual_tolerance: float = 1.0e-4,
    ):
        ridge_, tolerance = float(ridge), float(residual_tolerance)
        if (
            not isfinite(ridge_)
            or ridge_ <= 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "reflected-BSDE ridge and residual_tolerance must be finite and positive."
            )
        if (
            isinstance(minimum_paths, bool)
            or not isinstance(minimum_paths, int)
            or minimum_paths < 2
        ):
            raise ValueError("minimum_paths must be an integer at least two.")
        self.ridge, self.minimum_paths, self.residual_tolerance = (
            ridge_,
            minimum_paths,
            tolerance,
        )


class PreparedReflectedBSDE(StrictModule):
    problem: ReflectedPathDependentBSDEProblem
    paths: BSDEPathBatch
    plan: ReflectedBSDEPlan
    currency: Currency | None = eqx.field(static=True)
    evidence_binding: FinanceEvidenceBinding | None
    pricing_law: PricingLaw | None


class ReflectedBSDEDiagnostics(StrictModule):
    values: Array
    controls: Array
    lower_reflections: Array
    upper_reflections: Array
    maximum_dynamic_residual: Array
    regression_successful: Array


def prepare_reflected_bsde(
    problem: ReflectedPathDependentBSDEProblem,
    paths: BSDEPathBatch,
    plan: ReflectedBSDEPlan,
    /,
    *,
    currency: Currency | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
    pricing_law: PricingLaw | None = None,
) -> PreparedReflectedBSDE:
    if (
        not isinstance(problem, ReflectedPathDependentBSDEProblem)
        or not isinstance(paths, BSDEPathBatch)
        or not isinstance(plan, ReflectedBSDEPlan)
    ):
        raise TypeError("problem, paths and plan must be reflected-BSDE records.")
    if (
        paths.process_id != problem.process_id
        or paths.state_shape != problem.state_shape
        or paths.noise_shape != problem.noise_shape
    ):
        raise ValueError("reflected-BSDE problem and paths are incompatible.")
    if (
        len(paths.sample_shape) != 1
        or problem.output_shape != (1,)
        or problem.noise_shape != (1,)
    ):
        raise ValueError(
            "this reflected-BSDE regression route supports one path axis and scalar output/noise only."
        )
    if paths.num_paths < plan.minimum_paths:
        raise ValueError("reflected-BSDE path batch is undersampled for the plan.")
    return PreparedReflectedBSDE(
        problem, paths, plan, currency, evidence_binding, pricing_law
    )


def _feature_regression(features: Array, targets: Array, valid: Array, ridge: float):
    design = jnp.concatenate(
        (jnp.ones((features.shape[0], 1), dtype=features.dtype), features), axis=1
    )
    prediction, successful = _masked_regression(design, targets, valid, ridge)
    return prediction, successful


def evaluate_reflected_bsde(prepared: PreparedReflectedBSDE, /) -> ValuationResult:
    """Backward Euler/regression solution of a scalar reflected path-dependent BSDE."""

    if not isinstance(prepared, PreparedReflectedBSDE):
        raise TypeError("evaluate_reflected_bsde requires PreparedReflectedBSDE.")
    problem, paths, plan = prepared.problem, prepared.paths, prepared.plan
    states = paths.states
    times = paths.times
    valid_paths = paths.path_valid
    terminal_time = times[-1]
    terminal = jax.vmap(
        lambda path: jnp.asarray(
            problem.terminal(terminal_time, path, problem.args)
        ).reshape(())
    )(states)
    y = jnp.where(valid_paths, terminal, 0.0)
    value_history = (
        jnp.zeros((paths.num_paths, paths.num_steps + 1), dtype=y.dtype).at[:, -1].set(y)
    )
    control_history = jnp.zeros((paths.num_paths, paths.num_steps, 1), dtype=y.dtype)
    lower_history = jnp.zeros_like(value_history)
    upper_history = jnp.zeros_like(value_history)
    maximum_residual = jnp.asarray(0.0, dtype=y.dtype)
    regression_successful = jnp.asarray(True)
    for time_index in range(paths.num_steps - 1, -1, -1):
        time = times[time_index]
        dt = times[time_index + 1] - time
        prefixes = states[:, : time_index + 1]
        features = jax.vmap(
            lambda prefix: jnp.asarray(
                problem.path_features(time, prefix, problem.args)
            ).reshape((-1,))
        )(prefixes)
        conditional_y, solved_y = _feature_regression(
            features, y, valid_paths, plan.ridge
        )
        dw = paths.wiener_increments[:, time_index, 0]
        conditional_z, solved_z = _feature_regression(
            features, y * dw / dt, valid_paths, plan.ridge
        )
        generators = jax.vmap(
            lambda prefix, feature, y_value, z_value: jnp.asarray(
                problem.generator(
                    time, prefix, feature, y_value[None], z_value[None], problem.args
                )
            ).reshape(())
        )(prefixes, features, conditional_y, conditional_z)
        unreflected = conditional_y + dt * generators
        lower = (
            jnp.full_like(unreflected, -jnp.inf)
            if problem.lower_obstacle is None
            else jax.vmap(
                lambda prefix, feature: jnp.asarray(
                    problem.lower_obstacle(time, prefix, feature, problem.args)
                ).reshape(())
            )(prefixes, features)
        )
        upper = (
            jnp.full_like(unreflected, jnp.inf)
            if problem.upper_obstacle is None
            else jax.vmap(
                lambda prefix, feature: jnp.asarray(
                    problem.upper_obstacle(time, prefix, feature, problem.args)
                ).reshape(())
            )(prefixes, features)
        )
        lower_reflection = jnp.maximum(lower - unreflected, 0.0)
        after_lower = jnp.maximum(unreflected, lower)
        upper_reflection = jnp.maximum(after_lower - upper, 0.0)
        next_y = jnp.minimum(after_lower, upper)
        dynamic_residual = next_y - unreflected - lower_reflection + upper_reflection
        maximum_residual = jnp.maximum(
            maximum_residual,
            jnp.max(jnp.abs(jnp.where(valid_paths, dynamic_residual, 0.0))),
        )
        regression_successful = (
            regression_successful & solved_y & solved_z & jnp.all(lower <= upper)
        )
        y = jnp.where(valid_paths, next_y, 0.0)
        value_history = value_history.at[:, time_index].set(y)
        control_history = control_history.at[:, time_index, 0].set(conditional_z)
        lower_history = lower_history.at[:, time_index].set(lower_reflection)
        upper_history = upper_history.at[:, time_index].set(upper_reflection)
    valid_count = jnp.sum(valid_paths)
    value = jnp.sum(jnp.where(valid_paths, y, 0.0)) / valid_count
    centered = jnp.where(valid_paths, y - value, 0.0)
    sample_variance = jnp.sum(centered**2) / jnp.maximum(valid_count - 1, 1)
    standard_error = jnp.sqrt(sample_variance / valid_count)
    converged = regression_successful & (maximum_residual <= plan.residual_tolerance)
    value = eqx.error_if(
        value,
        ~converged | ~jnp.isfinite(value),
        "reflected-BSDE route did not satisfy regression and reflection residual tolerances.",
    )
    evidence = ValuationEvidence(
        route="reflected-path-dependent-bsde",
        finite=jnp.isfinite(value),
        converged=converged,
        exercise_consistent=jnp.all(lower_history >= 0.0) & jnp.all(upper_history >= 0.0),
        error_estimate=standard_error,
        residual_norm=maximum_residual,
        binding=prepared.evidence_binding,
        pricing_law=prepared.pricing_law,
    )
    diagnostics = ReflectedBSDEDiagnostics(
        value_history,
        control_history,
        lower_history,
        upper_history,
        maximum_residual,
        regression_successful,
    )
    return ValuationResult(
        value,
        evidence,
        standard_error=standard_error,
        currency=prepared.currency,
        diagnostics=diagnostics,
    )


__all__ = [
    "LSMDiagnostics",
    "LSMPlan",
    "LSMProblem",
    "PreparedReflectedBSDE",
    "ReflectedBSDEDiagnostics",
    "ReflectedBSDEPlan",
    "evaluate_bermudan_lsm",
    "evaluate_reflected_bsde",
    "prepare_reflected_bsde",
]
