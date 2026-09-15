#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.linalg import SmallLinearSolvePlan, solve_small_linear

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._regulators import FunctionalRGStatus, Regulator, ThresholdQuadraturePlan
from ._wetterich import _volume_factor


class PolynomialONFlowPlan(StrictModule, NonTrainableState):
    """O(N) potential projected onto mass, quartic, and optional sextic terms."""

    __hash__ = object.__hash__

    regulator: Regulator
    threshold: ThresholdQuadraturePlan
    component_count: int = eqx.field(static=True)
    dimension: float = eqx.field(static=True)
    coupling_count: int = eqx.field(static=True)
    lpa_prime: bool = eqx.field(static=True)
    volume_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_count: int,
        dimension: float,
        regulator: Regulator,
        threshold: ThresholdQuadraturePlan,
        /,
        *,
        coupling_count: int = 2,
        lpa_prime: bool = False,
    ):
        components = int(component_count)
        dimension_ = float(dimension)
        count = int(coupling_count)
        if not isinstance(regulator, Regulator) or not isinstance(
            threshold, ThresholdQuadraturePlan
        ):
            raise TypeError(
                "Polynomial O(N) flow requires regulator and threshold plans."
            )
        if (
            components <= 0
            or not 1.0 < dimension_ < 6.0
            or threshold.dimension != dimension_
            or count not in (2, 3)
        ):
            raise ValueError("Polynomial O(N) truncation is invalid.")
        self.regulator = regulator
        self.threshold = threshold
        self.component_count = components
        self.dimension = dimension_
        self.coupling_count = count
        self.lpa_prime = bool(lpa_prime)
        self.volume_factor = _volume_factor(dimension_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polynomial-on-wetterich-flow",
                "components": components,
                "dimension": dimension_,
                "regulator": regulator.regulator_id,
                "threshold": threshold.plan_id,
                "coupling_count": count,
                "lpa_prime": bool(lpa_prime),
                "normalization": "u=m2-rho+lambda-rho2-over-2+g6-rho3-over-6",
            }
        )

    def anomalous_dimension(self, couplings: ArrayLike, /) -> Array:
        value = jnp.asarray(couplings)
        mass, quartic = value[0], value[1]
        safe_quartic = jnp.where(jnp.abs(quartic) > 1.0e-15, quartic, 1.0)
        kappa = jnp.maximum(-mass / safe_quartic, 0.0)
        denominator = 1.0 + 2.0 * kappa * quartic
        eta = (
            16.0
            * self.volume_factor
            / self.dimension
            * kappa
            * quartic**2
            / denominator**2
        )
        return jnp.where(self.lpa_prime, eta, jnp.zeros_like(eta))

    def beta(self, couplings: ArrayLike, /) -> Array:
        value = jnp.asarray(couplings)
        if value.shape != (self.coupling_count,):
            raise ValueError("Couplings must match the polynomial truncation size.")
        mass, quartic = value[0], value[1]
        sextic = value[2] if self.coupling_count == 3 else jnp.zeros_like(mass)
        eta = self.anomalous_dimension(value)

        def threshold(mass_squared):
            return self.threshold.evaluate(self.regulator, mass_squared, eta).value

        first = jax.grad(threshold)(mass)
        second = jax.grad(jax.grad(threshold))(mass)
        loop_scale = 2.0 * self.volume_factor
        beta_mass = (-2.0 + eta) * mass + loop_scale * (
            self.component_count + 2.0
        ) * quartic * first
        beta_quartic = (self.dimension - 4.0 + 2.0 * eta) * quartic + loop_scale * (
            (self.component_count + 8.0) * quartic**2 * second
            + (self.component_count + 4.0) * sextic * first
        )
        if self.coupling_count == 2:
            return jnp.stack((beta_mass, beta_quartic))
        third = jax.grad(jax.grad(jax.grad(threshold)))(mass)
        beta_sextic = (2.0 * self.dimension - 6.0 + 3.0 * eta) * sextic + loop_scale * (
            (self.component_count + 26.0) * quartic**3 * third
            + 3.0 * (self.component_count + 14.0) * quartic * sextic * second
        )
        return jnp.stack((beta_mass, beta_quartic, beta_sextic))


class FixedPointResult(StrictModule):
    couplings: Array
    beta: Array
    stability_matrix: Array
    critical_exponents: Array
    eigenvectors: Array
    anomalous_dimension: Array
    residual_norm: Array
    iterations: Array
    finite: Array
    converged: Array
    status: Array
    flow_id: str = eqx.field(static=True)
    search_id: str = eqx.field(static=True)


class FixedPointSearchPlan(StrictModule, NonTrainableState):
    """Bounded Newton search and linearized critical-exponent extraction."""

    linear_solve: SmallLinearSolvePlan
    maximum_iterations: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    search_id: str = eqx.field(static=True)

    def __init__(
        self,
        coupling_count: int,
        /,
        *,
        maximum_iterations: int = 32,
        absolute_tolerance: float = 1.0e-9,
        relative_tolerance: float = 1.0e-8,
        damping: float = 1.0,
    ):
        count = int(coupling_count)
        iterations = int(maximum_iterations)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        damping_ = float(damping)
        if (
            count not in (2, 3)
            or iterations <= 0
            or not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute <= 0.0
            or relative <= 0.0
            or not np.isfinite(damping_)
            or not 0.0 < damping_ <= 1.0
        ):
            raise ValueError("Fixed-point work budget or tolerances are invalid.")
        self.linear_solve = SmallLinearSolvePlan(
            count,
            singular_tolerance=min(absolute, 1.0e-10),
            maximum_condition=1.0e12,
            refinement_iterations=2,
        )
        self.maximum_iterations = iterations
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.damping = damping_
        self.search_id = canonical_fingerprint(
            {
                "kind": "bounded-frg-fixed-point-search",
                "coupling_count": count,
                "maximum_iterations": iterations,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "damping": damping_,
                "linear_solve": self.linear_solve.plan_id,
            }
        )

    def search(
        self, flow: PolynomialONFlowPlan, initial_couplings: ArrayLike, /
    ) -> FixedPointResult:
        if not isinstance(flow, PolynomialONFlowPlan):
            raise TypeError("flow must be PolynomialONFlowPlan.")
        if flow.coupling_count != self.linear_solve.dimension:
            raise ValueError("Fixed-point search and flow coupling counts differ.")
        initial = jnp.asarray(initial_couplings)
        if initial.shape != (flow.coupling_count,):
            raise ValueError("Initial couplings have the wrong fixed shape.")
        initial_scale = jnp.maximum(jnp.linalg.norm(initial), 1.0)

        def body(index, carry):
            current, active, iterations, linear_success = carry
            residual = flow.beta(current)
            norm = jnp.linalg.norm(residual)
            converged = norm <= (
                self.absolute_tolerance + self.relative_tolerance * initial_scale
            )
            jacobian = jax.jacfwd(flow.beta)(current)
            solve = solve_small_linear(self.linear_solve, jacobian, -residual)
            candidate = current + self.damping * solve.value
            finite_candidate = jnp.all(jnp.isfinite(candidate))
            update = active & ~converged & solve.successful & finite_candidate
            current = jnp.where(update, candidate, current)
            iterations = jnp.where(active & ~converged, index + 1, iterations)
            active = update
            linear_success = linear_success & jnp.where(
                active | converged, solve.successful, linear_success
            )
            return current, active, iterations, linear_success

        seed = (
            initial,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True),
        )
        couplings, _, iterations, linear_success = jax.lax.fori_loop(
            0, self.maximum_iterations, body, seed
        )
        beta = flow.beta(couplings)
        residual_norm = jnp.linalg.norm(beta)
        stability = jax.jacfwd(flow.beta)(couplings)
        eigenvalues, eigenvectors = jnp.linalg.eig(stability)
        exponents = jnp.sort(-jnp.real(eigenvalues))[::-1]
        finite = (
            jnp.all(jnp.isfinite(couplings))
            & jnp.all(jnp.isfinite(beta))
            & jnp.all(jnp.isfinite(stability))
            & jnp.all(jnp.isfinite(exponents))
        )
        threshold = self.absolute_tolerance + self.relative_tolerance * initial_scale
        converged = finite & linear_success & (residual_norm <= threshold)
        status = jnp.where(
            converged,
            int(FunctionalRGStatus.SUCCESS),
            jnp.where(
                finite,
                int(FunctionalRGStatus.NOT_CONVERGED),
                int(FunctionalRGStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return FixedPointResult(
            couplings,
            beta,
            stability,
            exponents,
            eigenvectors,
            flow.anomalous_dimension(couplings),
            residual_norm,
            iterations,
            finite,
            converged,
            status,
            flow.plan_id,
            self.search_id,
        )


__all__ = [
    "FixedPointResult",
    "FixedPointSearchPlan",
    "PolynomialONFlowPlan",
]
