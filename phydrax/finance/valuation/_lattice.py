#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Recombining lattice valuation for European and American vanilla exercise."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..contracts._options import OptionType, VanillaPayoff
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw
from ..models._diffusion import BlackScholesModel
from ._types import ValuationEvidence, ValuationResult


ExerciseRoute = Literal["european", "american"]
LatticeScheme = Literal["cox-ross-rubinstein", "jarrow-rudd"]


class LatticePlan(StrictModule):
    steps: int = eqx.field(static=True)
    scheme: LatticeScheme = eqx.field(static=True)

    def __init__(self, steps: int, /, *, scheme: LatticeScheme = "cox-ross-rubinstein"):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("steps must be an integer at least two.")
        if scheme not in ("cox-ross-rubinstein", "jarrow-rudd"):
            raise ValueError("unsupported lattice scheme.")
        self.steps = steps
        self.scheme = scheme


class LatticeProblem(StrictModule):
    model: BlackScholesModel
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
        model: BlackScholesModel,
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
        if not isinstance(model, BlackScholesModel) or not isinstance(
            payoff, VanillaPayoff
        ):
            raise TypeError("model/payoff must be BlackScholesModel and VanillaPayoff.")
        if exercise_route not in ("european", "american"):
            raise ValueError("lattice exercise_route must be european or american.")
        if currency is not None and not isinstance(currency, Currency):
            raise TypeError("currency must be Currency or None.")
        if evidence_binding is not None and not isinstance(
            evidence_binding, FinanceEvidenceBinding
        ):
            raise TypeError("evidence_binding must be FinanceEvidenceBinding or None.")
        if pricing_law is not None and not isinstance(pricing_law, PricingLaw):
            raise TypeError("pricing_law must be PricingLaw or None.")
        values = tuple(
            jnp.asarray(value, dtype=float)
            for value in (spot, maturity, rate, dividend_yield)
        )
        if any(value.shape != () for value in values):
            raise ValueError("lattice market inputs must be scalar.")
        spot_, maturity_, rate_, dividend = values
        spot_ = eqx.error_if(
            spot_,
            ~jnp.isfinite(spot_)
            | (spot_ <= 0.0)
            | ~jnp.isfinite(maturity_)
            | (maturity_ <= 0.0)
            | ~jnp.isfinite(rate_)
            | ~jnp.isfinite(dividend),
            "lattice market inputs are invalid.",
        )
        self.model = model
        self.payoff = payoff
        self.spot = spot_
        self.maturity = maturity_
        self.rate = rate_
        self.dividend_yield = dividend
        self.exercise_route = exercise_route
        self.currency = currency
        self.evidence_binding = evidence_binding
        self.pricing_law = pricing_law


class PreparedLattice(StrictModule):
    problem: LatticeProblem
    plan: LatticePlan
    up: Array
    down: Array
    probability: Array
    discount: Array


class LatticeDiagnostics(StrictModule):
    up: Array
    down: Array
    probability: Array
    steps: int = eqx.field(static=True)
    scheme: LatticeScheme = eqx.field(static=True)


def prepare_lattice(problem: LatticeProblem, plan: LatticePlan, /) -> PreparedLattice:
    if not isinstance(problem, LatticeProblem) or not isinstance(plan, LatticePlan):
        raise TypeError("problem and plan must be lattice records.")
    dt = problem.maturity / plan.steps
    sigma = problem.model.volatility
    if plan.scheme == "cox-ross-rubinstein":
        up = jnp.exp(sigma * jnp.sqrt(dt))
        down = 1.0 / up
        probability = (jnp.exp((problem.rate - problem.dividend_yield) * dt) - down) / (
            up - down
        )
    else:
        drift = (problem.rate - problem.dividend_yield - 0.5 * sigma**2) * dt
        up = jnp.exp(drift + sigma * jnp.sqrt(dt))
        down = jnp.exp(drift - sigma * jnp.sqrt(dt))
        probability = jnp.asarray(0.5, dtype=up.dtype)
    probability = eqx.error_if(
        probability,
        ~jnp.isfinite(probability) | (probability < 0.0) | (probability > 1.0),
        "lattice step has an inadmissible risk-neutral transition probability; refine the grid.",
    )
    return PreparedLattice(
        problem, plan, up, down, probability, jnp.exp(-problem.rate * dt)
    )


def _intrinsic(payoff: VanillaPayoff, spot: Array) -> Array:
    sign = 1.0 if payoff.option_type is OptionType.CALL else -1.0
    return payoff.notional * jnp.maximum(sign * (spot - payoff.strike), 0.0)


def evaluate_lattice(
    problem_or_prepared: LatticeProblem | PreparedLattice,
    plan: LatticePlan | None = None,
    /,
) -> ValuationResult:
    """Execute the prepared recombining lattice; no route dispatch is inferred."""

    if isinstance(problem_or_prepared, PreparedLattice):
        if plan is not None:
            raise ValueError("plan must be omitted when evaluating PreparedLattice.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, LatticeProblem) and isinstance(
        plan, LatticePlan
    ):
        prepared = prepare_lattice(problem_or_prepared, plan)
    else:
        raise TypeError(
            "evaluate_lattice requires PreparedLattice or LatticeProblem plus LatticePlan."
        )
    problem, plan_ = prepared.problem, prepared.plan
    indices = jnp.arange(plan_.steps + 1, dtype=problem.spot.dtype)
    terminal_spots = (
        problem.spot * prepared.down ** (plan_.steps - indices) * prepared.up**indices
    )
    values = _intrinsic(problem.payoff, terminal_spots)

    def backward(iteration, current):
        step = plan_.steps - 1 - iteration
        active = indices <= step
        continuation = prepared.discount * (
            (1.0 - prepared.probability) * current[:-1]
            + prepared.probability * current[1:]
        )
        padded = jnp.concatenate((continuation, jnp.zeros((1,), dtype=current.dtype)))
        if problem.exercise_route == "american":
            step_spots = (
                problem.spot * prepared.down ** (step - indices) * prepared.up**indices
            )
            padded = jnp.maximum(padded, _intrinsic(problem.payoff, step_spots))
        return jnp.where(active, padded, 0.0)

    values = jax.lax.fori_loop(0, plan_.steps, backward, values)
    value = values[0]
    discount = jnp.exp(-problem.rate * problem.maturity)
    carry = jnp.exp(-problem.dividend_yield * problem.maturity)
    if problem.payoff.option_type is OptionType.CALL:
        european_lower = jnp.maximum(
            problem.spot * carry - problem.payoff.strike * discount, 0.0
        )
        lower = (
            _intrinsic(problem.payoff, problem.spot)
            if problem.exercise_route == "american"
            else problem.payoff.notional * european_lower
        )
        upper = problem.payoff.notional * problem.spot
    else:
        european_lower = jnp.maximum(
            problem.payoff.strike * discount - problem.spot * carry, 0.0
        )
        lower = (
            _intrinsic(problem.payoff, problem.spot)
            if problem.exercise_route == "american"
            else problem.payoff.notional * european_lower
        )
        upper = problem.payoff.notional * problem.payoff.strike
    evidence = ValuationEvidence(
        route=f"lattice:{problem.exercise_route}:{plan_.scheme}",
        finite=jnp.isfinite(value),
        binding=problem.evidence_binding,
        pricing_law=problem.pricing_law,
    )
    diagnostics = LatticeDiagnostics(
        prepared.up, prepared.down, prepared.probability, plan_.steps, plan_.scheme
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
    "LatticeDiagnostics",
    "LatticePlan",
    "LatticeProblem",
    "PreparedLattice",
    "evaluate_lattice",
    "prepare_lattice",
]
