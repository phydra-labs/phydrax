#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Theta finite-difference PDE and finite-activity PIDE valuation routes."""

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._numerics._quadrature_rules import gauss_legendre_data
from ..._strict import StrictModule
from ...linalg._tridiagonal_lines import solve_tridiagonal_lines
from ..contracts._options import OptionType, VanillaPayoff
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw
from ..models._diffusion import BlackScholesModel, LocalVolatilityModel
from ..models._jump import KouJumpDiffusionModel, MertonJumpDiffusionModel
from ._types import ValuationEvidence, ValuationResult


ExerciseRoute = Literal["european", "american"]


class FiniteDifferencePlan(StrictModule):
    space_steps: int = eqx.field(static=True)
    time_steps: int = eqx.field(static=True)
    theta: float = eqx.field(static=True)
    spot_minimum: float = eqx.field(static=True)
    spot_maximum: float = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        space_steps: int = 400,
        time_steps: int = 400,
        theta: float = 0.5,
        spot_minimum: float = 0.0,
        spot_maximum: float,
        pivot_tolerance: float = 1.0e-13,
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 4
            for value in (space_steps, time_steps)
        ):
            raise ValueError(
                "finite-difference step counts must be integers at least four."
            )
        theta_, minimum, maximum, tolerance = map(
            float, (theta, spot_minimum, spot_maximum, pivot_tolerance)
        )
        if not all(isfinite(value) for value in (theta_, minimum, maximum, tolerance)):
            raise ValueError("finite-difference scalar plan values must be finite.")
        if not 0.5 <= theta_ <= 1.0:
            raise ValueError("theta must lie in [0.5, 1].")
        if minimum < 0.0 or maximum <= minimum or tolerance <= 0.0:
            raise ValueError(
                "spot bounds must be ordered/non-negative and pivot_tolerance positive."
            )
        self.space_steps = space_steps
        self.time_steps = time_steps
        self.theta = theta_
        self.spot_minimum = minimum
        self.spot_maximum = maximum
        self.pivot_tolerance = tolerance


class PDEProblem(StrictModule):
    model: BlackScholesModel | LocalVolatilityModel
    payoff: VanillaPayoff
    spot: Array
    maturity: Array
    rate: Array
    dividend_yield: Array
    exercise_route: ExerciseRoute = eqx.field(static=True)
    currency: Currency | None = eqx.field(static=True)
    evidence_binding: FinanceEvidenceBinding | None
    pricing_law: PricingLaw | None

    def __init__(
        self,
        model: BlackScholesModel | LocalVolatilityModel,
        payoff: VanillaPayoff,
        spot: ArrayLike,
        maturity: ArrayLike,
        rate: ArrayLike,
        /,
        *,
        dividend_yield: ArrayLike = 0.0,
        exercise_route: ExerciseRoute = "european",
        currency: Currency | None = None,
        evidence_binding: FinanceEvidenceBinding | None = None,
        pricing_law: PricingLaw | None = None,
    ):
        if not isinstance(model, (BlackScholesModel, LocalVolatilityModel)):
            raise TypeError(
                "PDE model must be BlackScholesModel or LocalVolatilityModel."
            )
        if not isinstance(payoff, VanillaPayoff):
            raise TypeError("payoff must be VanillaPayoff.")
        if exercise_route not in ("european", "american"):
            raise ValueError("PDE exercise_route must be european or american.")
        if currency is not None and not isinstance(currency, Currency):
            raise TypeError("currency must be Currency or None.")
        if evidence_binding is not None and not isinstance(
            evidence_binding, FinanceEvidenceBinding
        ):
            raise TypeError("evidence_binding must be FinanceEvidenceBinding or None.")
        if pricing_law is not None and not isinstance(pricing_law, PricingLaw):
            raise TypeError("pricing_law must be PricingLaw or None.")
        spot_, maturity_, rate_, dividend = tuple(
            jnp.asarray(value, dtype=float)
            for value in (spot, maturity, rate, dividend_yield)
        )
        if any(value.shape != () for value in (spot_, maturity_, rate_, dividend)):
            raise ValueError("PDE market inputs must be scalar.")
        spot_ = eqx.error_if(
            spot_,
            ~jnp.isfinite(spot_)
            | (spot_ <= 0.0)
            | ~jnp.isfinite(maturity_)
            | (maturity_ <= 0.0)
            | ~jnp.isfinite(rate_)
            | ~jnp.isfinite(dividend),
            "PDE market inputs are invalid.",
        )
        self.model, self.payoff = model, payoff
        self.spot, self.maturity, self.rate, self.dividend_yield = (
            spot_,
            maturity_,
            rate_,
            dividend,
        )
        self.exercise_route, self.currency = exercise_route, currency
        self.evidence_binding, self.pricing_law = evidence_binding, pricing_law


class PreparedPDE(StrictModule):
    problem: PDEProblem
    plan: FiniteDifferencePlan
    spot_grid: Array
    terminal_values: Array


class PIDEPlan(StrictModule):
    finite_difference: FiniteDifferencePlan
    jump_quadrature_nodes: int = eqx.field(static=True)
    jump_truncation: float = eqx.field(static=True)

    def __init__(
        self,
        finite_difference: FiniteDifferencePlan,
        /,
        *,
        jump_quadrature_nodes: int = 48,
        jump_truncation: float = 10.0,
    ):
        if not isinstance(finite_difference, FiniteDifferencePlan):
            raise TypeError("finite_difference must be a FiniteDifferencePlan.")
        if (
            isinstance(jump_quadrature_nodes, bool)
            or not isinstance(jump_quadrature_nodes, int)
            or jump_quadrature_nodes < 8
        ):
            raise ValueError("jump_quadrature_nodes must be an integer at least eight.")
        truncation = float(jump_truncation)
        if not isfinite(truncation) or truncation <= 2.0:
            raise ValueError("jump_truncation must be finite and greater than two.")
        self.finite_difference = finite_difference
        self.jump_quadrature_nodes = jump_quadrature_nodes
        self.jump_truncation = truncation


class PIDEProblem(StrictModule):
    diffusion_problem: PDEProblem
    jump_model: MertonJumpDiffusionModel | KouJumpDiffusionModel

    def __init__(
        self,
        diffusion_problem: PDEProblem,
        jump_model: MertonJumpDiffusionModel | KouJumpDiffusionModel,
        /,
    ):
        if not isinstance(diffusion_problem, PDEProblem):
            raise TypeError("diffusion_problem must be a PDEProblem.")
        if not isinstance(diffusion_problem.model, BlackScholesModel):
            raise ValueError(
                "finite-activity PIDE currently requires constant diffusion."
            )
        if not isinstance(
            jump_model, (MertonJumpDiffusionModel, KouJumpDiffusionModel)
        ):
            raise TypeError(
                "PIDE supports finite-activity Merton or Kou jumps only; "
                "infinite-activity routes require a different discretization."
            )
        self.diffusion_problem = diffusion_problem
        self.jump_model = jump_model


class PreparedPIDE(StrictModule):
    problem: PIDEProblem
    plan: PIDEPlan
    prepared_pde: PreparedPDE
    jump_nodes: Array
    jump_weights: Array


class FiniteDifferenceDiagnostics(StrictModule):
    maximum_linear_residual: Array
    minimum_pivot: Array
    all_linear_solves_successful: Array
    space_step: Array
    time_step: Array
    space_steps: int = eqx.field(static=True)
    time_steps: int = eqx.field(static=True)
    route: str = eqx.field(static=True)


def _intrinsic(payoff: VanillaPayoff, spot: Array) -> Array:
    sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
    return payoff.notional * jnp.maximum(sign * (spot - payoff.strike), 0.0)


def prepare_pde(problem: PDEProblem, plan: FiniteDifferencePlan, /) -> PreparedPDE:
    if not isinstance(problem, PDEProblem) or not isinstance(plan, FiniteDifferencePlan):
        raise TypeError("problem and plan must be PDE records.")
    grid = jnp.linspace(plan.spot_minimum, plan.spot_maximum, plan.space_steps + 1)
    grid = eqx.error_if(
        grid,
        (problem.spot <= plan.spot_minimum)
        | (problem.spot >= plan.spot_maximum)
        | (problem.payoff.strike >= plan.spot_maximum),
        "valuation spot must lie inside the PDE grid and spot_maximum must exceed strike.",
    )
    return PreparedPDE(problem, plan, grid, _intrinsic(problem.payoff, grid))


def _boundaries(problem: PDEProblem, tau: Array, grid: Array) -> tuple[Array, Array]:
    if problem.payoff.option_type is OptionType.CALL:
        left = jnp.asarray(0.0, dtype=grid.dtype)
        right = problem.payoff.notional * jnp.maximum(
            grid[-1] * jnp.exp(-problem.dividend_yield * tau)
            - problem.payoff.strike * jnp.exp(-problem.rate * tau),
            0.0,
        )
    else:
        left = (
            problem.payoff.notional * problem.payoff.strike * jnp.exp(-problem.rate * tau)
        )
        right = jnp.asarray(0.0, dtype=grid.dtype)
    if problem.exercise_route == "american":
        left = jnp.maximum(left, _intrinsic(problem.payoff, grid[0]))
        right = jnp.maximum(right, _intrinsic(problem.payoff, grid[-1]))
    return left, right


def _local_variance(problem: PDEProblem, tau: Array, interior_spots: Array) -> Array:
    if isinstance(problem.model, BlackScholesModel):
        return jnp.full_like(interior_spots, problem.model.volatility**2)
    log_moneyness = jnp.log(interior_spots / problem.spot)
    variance = problem.model.variance(tau, log_moneyness)
    return eqx.error_if(
        variance,
        jnp.any(~jnp.isfinite(variance)) | jnp.any(variance <= 0.0),
        "local variance became non-positive on the PDE grid.",
    )


def _theta_step(problem, plan, grid, current, tau, drift, explicit_source):
    dt = problem.maturity / plan.time_steps
    ds = grid[1] - grid[0]
    spots = grid[1:-1]
    variance = _local_variance(problem, tau, spots)
    a = 0.5 * variance * spots**2 / ds**2 - 0.5 * drift * spots / ds
    b = -variance * spots**2 / ds**2 - problem.rate
    c = 0.5 * variance * spots**2 / ds**2 + 0.5 * drift * spots / ds
    old = current[1:-1]
    applied = a * current[:-2] + b * old + c * current[2:]
    rhs = old + (1.0 - plan.theta) * dt * applied + dt * explicit_source
    lower_boundary, upper_boundary = _boundaries(problem, tau, grid)
    lhs_lower_raw = -plan.theta * dt * a
    lhs_diagonal = 1.0 - plan.theta * dt * b
    lhs_upper_raw = -plan.theta * dt * c
    rhs = rhs.at[0].add(-lhs_lower_raw[0] * lower_boundary)
    rhs = rhs.at[-1].add(-lhs_upper_raw[-1] * upper_boundary)
    lhs_lower = lhs_lower_raw.at[0].set(0.0)
    lhs_upper = lhs_upper_raw.at[-1].set(0.0)
    solve = solve_tridiagonal_lines(
        lhs_lower, lhs_diagonal, lhs_upper, rhs, -1, pivot_tolerance=plan.pivot_tolerance
    )
    updated = jnp.concatenate((lower_boundary[None], solve.value, upper_boundary[None]))
    if problem.exercise_route == "american":
        updated = jnp.maximum(updated, _intrinsic(problem.payoff, grid))
    return updated, solve


def evaluate_pde(
    problem_or_prepared: PDEProblem | PreparedPDE,
    plan: FiniteDifferencePlan | None = None,
    /,
) -> ValuationResult:
    if isinstance(problem_or_prepared, PreparedPDE):
        if plan is not None:
            raise ValueError("plan must be omitted with PreparedPDE.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, PDEProblem) and isinstance(
        plan, FiniteDifferencePlan
    ):
        prepared = prepare_pde(problem_or_prepared, plan)
    else:
        raise TypeError(
            "evaluate_pde requires PreparedPDE or PDEProblem plus FiniteDifferencePlan."
        )
    problem, plan_ = prepared.problem, prepared.plan
    initial = (
        prepared.terminal_values,
        jnp.asarray(0.0),
        jnp.asarray(jnp.inf),
        jnp.asarray(True),
    )

    def body(iteration, carry):
        current, maximum_residual, minimum_pivot, successful = carry
        tau = (iteration + 1) * problem.maturity / plan_.time_steps
        updated, solve = _theta_step(
            problem,
            plan_,
            prepared.spot_grid,
            current,
            tau,
            problem.rate - problem.dividend_yield,
            jnp.zeros((plan_.space_steps - 1,), dtype=current.dtype),
        )
        return (
            updated,
            jnp.maximum(maximum_residual, solve.residual_norm),
            jnp.minimum(minimum_pivot, solve.minimum_pivot),
            successful & solve.successful,
        )

    values, residual, pivot, successful = jax.lax.fori_loop(
        0, plan_.time_steps, body, initial
    )
    value = jnp.interp(problem.spot, prepared.spot_grid, values)
    lower, upper = _valuation_bounds(problem)
    evidence = ValuationEvidence(
        route=f"finite-difference-pde:{problem.exercise_route}",
        finite=jnp.isfinite(value),
        converged=successful,
        residual_norm=residual,
        binding=problem.evidence_binding,
        pricing_law=problem.pricing_law,
    )
    diagnostics = FiniteDifferenceDiagnostics(
        residual,
        pivot,
        successful,
        prepared.spot_grid[1] - prepared.spot_grid[0],
        problem.maturity / plan_.time_steps,
        plan_.space_steps,
        plan_.time_steps,
        "pde",
    )
    return ValuationResult(
        value,
        evidence,
        lower_bound=lower,
        upper_bound=upper,
        currency=problem.currency,
        diagnostics=diagnostics,
    )


def _valuation_bounds(problem: PDEProblem) -> tuple[Array, Array]:
    discount = jnp.exp(-problem.rate * problem.maturity)
    carry = jnp.exp(-problem.dividend_yield * problem.maturity)
    if problem.payoff.option_type is OptionType.CALL:
        european = problem.payoff.notional * jnp.maximum(
            problem.spot * carry - problem.payoff.strike * discount, 0.0
        )
        lower = (
            _intrinsic(problem.payoff, problem.spot)
            if problem.exercise_route == "american"
            else european
        )
        upper = problem.payoff.notional * problem.spot
    else:
        european = problem.payoff.notional * jnp.maximum(
            problem.payoff.strike * discount - problem.spot * carry, 0.0
        )
        lower = (
            _intrinsic(problem.payoff, problem.spot)
            if problem.exercise_route == "american"
            else european
        )
        upper = problem.payoff.notional * problem.payoff.strike
    return lower, upper


def prepare_pide(problem: PIDEProblem, plan: PIDEPlan, /) -> PreparedPIDE:
    if not isinstance(problem, PIDEProblem) or not isinstance(plan, PIDEPlan):
        raise TypeError("problem and plan must be PIDE records.")
    prepared = prepare_pde(problem.diffusion_problem, plan.finite_difference)
    rule = gauss_legendre_data(plan.jump_quadrature_nodes)
    base_nodes = jnp.asarray(rule.nodes)
    base_weights = jnp.asarray(rule.weights)
    jump = problem.jump_model
    if isinstance(jump, MertonJumpDiffusionModel):
        nodes = jump.jump_mean + plan.jump_truncation * jump.jump_volatility * base_nodes
        density = jnp.exp(
            -0.5 * ((nodes - jump.jump_mean) / jump.jump_volatility) ** 2
        ) / (jnp.sqrt(2.0 * jnp.pi) * jump.jump_volatility)
        weights = base_weights * plan.jump_truncation * jump.jump_volatility * density
    else:
        nodes = plan.jump_truncation * base_nodes
        density = jnp.where(
            nodes >= 0.0,
            jump.upward_probability
            * jump.upward_rate
            * jnp.exp(-jump.upward_rate * nodes),
            (1.0 - jump.upward_probability)
            * jump.downward_rate
            * jnp.exp(jump.downward_rate * nodes),
        )
        weights = base_weights * plan.jump_truncation * density
    weights = weights / jnp.sum(weights)
    return PreparedPIDE(problem, plan, prepared, nodes, weights)


def evaluate_pide(
    problem_or_prepared: PIDEProblem | PreparedPIDE,
    plan: PIDEPlan | None = None,
    /,
) -> ValuationResult:
    if isinstance(problem_or_prepared, PreparedPIDE):
        if plan is not None:
            raise ValueError("plan must be omitted with PreparedPIDE.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, PIDEProblem) and isinstance(plan, PIDEPlan):
        prepared = prepare_pide(problem_or_prepared, plan)
    else:
        raise TypeError(
            "evaluate_pide requires PreparedPIDE or PIDEProblem plus PIDEPlan."
        )
    problem = prepared.problem.diffusion_problem
    jump = prepared.problem.jump_model
    fd = prepared.plan.finite_difference
    grid = prepared.prepared_pde.spot_grid
    intensity = jump.jump_intensity
    compensator = jump.exponential_compensator
    initial = (
        prepared.prepared_pde.terminal_values,
        jnp.asarray(0.0),
        jnp.asarray(jnp.inf),
        jnp.asarray(True),
    )

    def body(iteration, carry):
        current, maximum_residual, minimum_pivot, successful = carry
        tau = (iteration + 1) * problem.maturity / fd.time_steps
        shifted_spots = grid[1:-1, None] * jnp.exp(prepared.jump_nodes[None, :])
        shifted_values = jax.vmap(lambda row: jnp.interp(row, grid, current))(
            shifted_spots
        )
        expectation = jnp.sum(shifted_values * prepared.jump_weights[None, :], axis=1)
        source = intensity * (expectation - current[1:-1])
        updated, solve = _theta_step(
            problem,
            fd,
            grid,
            current,
            tau,
            problem.rate - problem.dividend_yield - compensator,
            source,
        )
        return (
            updated,
            jnp.maximum(maximum_residual, solve.residual_norm),
            jnp.minimum(minimum_pivot, solve.minimum_pivot),
            successful & solve.successful,
        )

    values, residual, pivot, successful = jax.lax.fori_loop(
        0, fd.time_steps, body, initial
    )
    value = jnp.interp(problem.spot, grid, values)
    lower, upper = _valuation_bounds(problem)
    evidence = ValuationEvidence(
        route=f"finite-activity-pide:{problem.exercise_route}",
        finite=jnp.isfinite(value),
        converged=successful,
        residual_norm=residual,
        binding=problem.evidence_binding,
        pricing_law=problem.pricing_law,
    )
    diagnostics = FiniteDifferenceDiagnostics(
        residual,
        pivot,
        successful,
        grid[1] - grid[0],
        problem.maturity / fd.time_steps,
        fd.space_steps,
        fd.time_steps,
        "pide",
    )
    return ValuationResult(
        value,
        evidence,
        lower_bound=lower,
        upper_bound=upper,
        currency=problem.currency,
        diagnostics=diagnostics,
    )


__all__ = [
    "FiniteDifferenceDiagnostics",
    "FiniteDifferencePlan",
    "PDEProblem",
    "PIDEPlan",
    "PIDEProblem",
    "PreparedPDE",
    "PreparedPIDE",
    "evaluate_pde",
    "evaluate_pide",
    "prepare_pde",
    "prepare_pide",
]
