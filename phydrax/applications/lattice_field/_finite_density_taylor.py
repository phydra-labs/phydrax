#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._finite_density import (
    FiniteDensityStatus,
    SusceptibilityEstimate,
)


class PreparedTaylorEOS(StrictModule, NonTrainableState):
    estimate: SusceptibilityEstimate
    exponents: Array
    inverse_factorials: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, estimate: SusceptibilityEstimate, /):
        if not isinstance(estimate, SusceptibilityEstimate):
            raise TypeError("estimate must be SusceptibilityEstimate.")
        exponents = np.asarray(
            [value.orders for value in estimate.indices], dtype=np.int32
        )
        inverse_factorials = np.asarray(
            [1.0 / np.prod([factorial(int(order)) for order in row]) for row in exponents]
        )
        self.estimate = estimate
        self.exponents = jnp.asarray(exponents)
        self.inverse_factorials = jnp.asarray(inverse_factorials)
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-taylor-bqs-eos", "estimate": estimate.estimate_id}
        )


class TaylorEOSResult(StrictModule, NonTrainableState):
    pressure_over_temperature4: Array
    densities_over_temperature3: Array
    entropy_over_temperature3: Array
    energy_over_temperature4: Array
    susceptibility_matrix: Array
    thermodynamic_identity_residual: Array
    in_domain: Array
    derivative_valid: Array
    status: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(FiniteDensityStatus.SUCCESS)


def prepare_taylor_eos(estimate: SusceptibilityEstimate, /) -> PreparedTaylorEOS:
    return PreparedTaylorEOS(estimate)


def _monomial_derivative(exponents, ratios, axis):
    order = exponents[:, axis]
    reduced = exponents.at[:, axis].set(jnp.maximum(order - 1, 0))
    powers = jnp.prod(jnp.power(ratios[None, :], reduced), axis=1)
    return jnp.where(order > 0, order * powers, 0.0)


def _monomial_second_derivative(exponents, ratios, first, second):
    first_order = exponents[:, first]
    after_first = exponents.at[:, first].set(jnp.maximum(first_order - 1, 0))
    second_order = after_first[:, second]
    reduced = after_first.at[:, second].set(jnp.maximum(second_order - 1, 0))
    powers = jnp.prod(jnp.power(ratios[None, :], reduced), axis=1)
    return jnp.where(
        (first_order > 0) & (second_order > 0), first_order * second_order * powers, 0.0
    )


def evaluate_taylor_eos(
    prepared: PreparedTaylorEOS,
    temperature: ArrayLike,
    chemical_potentials: ArrayLike,
    /,
) -> TaylorEOSResult:
    """Evaluate one B/Q/S state from a single Taylor pressure potential."""
    if not isinstance(prepared, PreparedTaylorEOS):
        raise TypeError("prepared must be PreparedTaylorEOS.")
    temperature_ = jnp.asarray(temperature, dtype=prepared.estimate.values.dtype).reshape(
        ()
    )
    chemical = jnp.asarray(chemical_potentials, dtype=temperature_.dtype)
    if chemical.shape != (3,):
        raise ValueError("chemical_potentials must contain B/Q/S values.")
    ratios = chemical / temperature_
    coefficients = linear_interpolate(
        prepared.estimate.temperatures,
        prepared.estimate.values,
        temperature_,
        axis=0,
        bounds="fill",
        fill_value=jnp.nan,
    )
    coefficient_derivative = linear_interpolate(
        prepared.estimate.temperatures,
        prepared.estimate.values,
        temperature_,
        axis=0,
        derivative_order=1,
        bounds="fill",
        fill_value=jnp.nan,
    )
    monomials = jnp.prod(jnp.power(ratios[None, :], prepared.exponents), axis=1)
    weighted_coefficients = coefficients.values * prepared.inverse_factorials
    pressure = jnp.sum(weighted_coefficients * monomials)
    densities = jnp.stack(
        tuple(
            jnp.sum(
                weighted_coefficients
                * _monomial_derivative(prepared.exponents, ratios, axis)
            )
            for axis in range(3)
        )
    )
    hessian = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    jnp.sum(
                        weighted_coefficients
                        * _monomial_second_derivative(
                            prepared.exponents, ratios, first, second
                        )
                    )
                    for second in range(3)
                )
            )
            for first in range(3)
        )
    )
    explicit_temperature_derivative = jnp.sum(
        coefficient_derivative.values * prepared.inverse_factorials * monomials
    )
    entropy = (
        4.0 * pressure
        + temperature_ * explicit_temperature_derivative
        - jnp.dot(ratios, densities)
    )
    energy = -pressure + entropy + jnp.dot(ratios, densities)
    residual = energy + pressure - entropy - jnp.dot(ratios, densities)
    in_domain = prepared.estimate.domain.contains(temperature_, ratios)
    interpolation_valid = coefficients.support & coefficient_derivative.support
    source_valid = jnp.all(prepared.estimate.valid)
    finite = jnp.all(
        jnp.isfinite(
            jnp.concatenate(
                (
                    jnp.asarray([pressure, entropy, energy, residual]),
                    densities,
                    hessian.reshape((-1,)),
                )
            )
        )
    )
    derivative_valid = in_domain & interpolation_valid & source_valid & finite
    status = jnp.where(
        ~in_domain,
        int(FiniteDensityStatus.OUTSIDE_DOMAIN),
        jnp.where(
            ~source_valid,
            int(FiniteDensityStatus.UNQUALIFIED_SOURCE),
            jnp.where(
                ~finite,
                int(FiniteDensityStatus.NONFINITE),
                int(FiniteDensityStatus.SUCCESS),
            ),
        ),
    )
    return TaylorEOSResult(
        jnp.where(in_domain, pressure, jnp.nan),
        jnp.where(in_domain, densities, jnp.nan),
        jnp.where(in_domain, entropy, jnp.nan),
        jnp.where(in_domain, energy, jnp.nan),
        jnp.where(in_domain, hessian, jnp.nan),
        jnp.where(in_domain, residual, jnp.nan),
        in_domain,
        derivative_valid,
        status.astype(jnp.int32),
        prepared.prepared_id,
    )


class HeavyIonConstraintPlan(StrictModule, NonTrainableState):
    charge_to_baryon_ratio: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        charge_to_baryon_ratio: float,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 32,
    ):
        ratio = float(charge_to_baryon_ratio)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if (
            not math.isfinite(ratio)
            or not math.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations < 1
        ):
            raise ValueError("Heavy-ion constraint ratio and solver policy are invalid.")
        self.charge_to_baryon_ratio = ratio
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "heavy-ion-bqs-constraint",
                "charge_to_baryon_ratio": ratio,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
            }
        )


class HeavyIonConstraintResult(StrictModule, NonTrainableState):
    chemical_potentials: Array
    residual: Array
    iterations: Array
    converged: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def solve_heavy_ion_path(
    prepared: PreparedTaylorEOS,
    plan: HeavyIonConstraintPlan,
    temperature: ArrayLike,
    baryon_chemical_potential: ArrayLike,
    /,
    *,
    initial_charge_strangeness: ArrayLike = (0.0, 0.0),
) -> HeavyIonConstraintResult:
    """Solve n_S=0 and n_Q=r n_B under a bounded fixed Newton budget."""
    if not isinstance(prepared, PreparedTaylorEOS) or not isinstance(
        plan, HeavyIonConstraintPlan
    ):
        raise TypeError("prepared and plan must use finite-density Taylor types.")
    temperature_ = jnp.asarray(temperature, dtype=prepared.estimate.values.dtype).reshape(
        ()
    )
    baryon = jnp.asarray(baryon_chemical_potential, dtype=temperature_.dtype).reshape(())
    initial = jnp.asarray(initial_charge_strangeness, dtype=temperature_.dtype)
    if initial.shape != (2,):
        raise ValueError("initial_charge_strangeness must contain Q/S values.")

    def iteration(_, state):
        values, active = state
        chemical = jnp.asarray([baryon, values[0], values[1]])
        evaluated = evaluate_taylor_eos(prepared, temperature_, chemical)
        densities = evaluated.densities_over_temperature3
        residual = jnp.asarray(
            [densities[2], densities[1] - plan.charge_to_baryon_ratio * densities[0]]
        )
        hessian = evaluated.susceptibility_matrix
        jacobian = (
            jnp.asarray(
                [
                    [hessian[2, 1], hessian[2, 2]],
                    [
                        hessian[1, 1] - plan.charge_to_baryon_ratio * hessian[0, 1],
                        hessian[1, 2] - plan.charge_to_baryon_ratio * hessian[0, 2],
                    ],
                ]
            )
            / temperature_
        )
        solved = solve(
            LinearSystem(DenseLinearOperator(jacobian)),
            -residual,
            policy=LinearSolvePolicy(DenseLU()),
        )
        candidate = values + solved.value
        converged = jnp.linalg.norm(residual) <= plan.tolerance
        valid = (
            evaluated.derivative_valid
            & jnp.all(solved.status == 0)
            & jnp.all(jnp.isfinite(candidate))
        )
        update = active & ~converged & valid
        return jnp.where(update, candidate, values), update

    final, active = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        iteration,
        (initial, jnp.asarray(True)),
    )
    chemical = jnp.asarray([baryon, final[0], final[1]])
    evaluated = evaluate_taylor_eos(prepared, temperature_, chemical)
    residual = jnp.asarray(
        [
            evaluated.densities_over_temperature3[2],
            evaluated.densities_over_temperature3[1]
            - plan.charge_to_baryon_ratio * evaluated.densities_over_temperature3[0],
        ]
    )
    converged = jnp.linalg.norm(residual) <= plan.tolerance
    valid = converged & evaluated.derivative_valid & ~active
    return HeavyIonConstraintResult(
        chemical,
        residual,
        jnp.asarray(plan.maximum_iterations, dtype=jnp.int32),
        converged,
        valid,
        jnp.asarray(
            jnp.where(
                valid,
                int(FiniteDensityStatus.SUCCESS),
                int(FiniteDensityStatus.CONSTRAINT_FAILURE),
            ),
            dtype=jnp.int32,
        ),
        plan.plan_id,
        prepared.prepared_id,
    )


__all__ = [
    "HeavyIonConstraintPlan",
    "HeavyIonConstraintResult",
    "PreparedTaylorEOS",
    "TaylorEOSResult",
    "evaluate_taylor_eos",
    "prepare_taylor_eos",
    "solve_heavy_ion_path",
]
