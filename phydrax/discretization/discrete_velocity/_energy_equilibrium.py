#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
from phydrax.linalg import SmallLinearSolvePlan, solve_small_linear

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._quadrature import CertifiedDiscreteVelocityQuadrature


class EnergyEquilibriumStatus(IntEnum):
    """Outcome of a positive total-energy equilibrium evaluation."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONPOSITIVE_TOTAL_ENERGY = 2
    INFEASIBLE_TARGET = 3
    LINEAR_SOLVE_FAILED = 4
    MAXIMUM_ITERATIONS = 5
    NONFINITE_OUTPUT = 6
    NONPOSITIVE_POPULATIONS = 7


class EnergyEquilibriumEvidence(StrictModule):
    """Small diagnostics for one or more total-energy equilibria.

    Population-sized values deliberately live only on
    :class:`EnergyEquilibriumResult`.
    """

    target_total_energy: Array
    recovered_total_energy: Array
    total_energy_residual: Array
    target_flux: Array
    recovered_flux: Array
    flux_residual: Array
    normalized_target_flux: Array
    interior_margin: Array
    residual_norm: Array
    flux_error_norm: Array
    minimum_population: Array
    iterations: Array
    status: Array
    finite: Array
    positive: Array
    converged: Array
    quadrature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(EnergyEquilibriumStatus.SUCCESS)


class EnergyEquilibriumResult(StrictModule):
    """Positive energy populations, their dual, and compact evidence."""

    populations: Array
    dual: Array
    evidence: EnergyEquilibriumEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


def _cross_2d(origin: np.ndarray, first: np.ndarray, second: np.ndarray, /) -> float:
    left = first - origin
    right = second - origin
    return float(left[0] * right[1] - left[1] * right[0])


def _convex_hull_halfspaces(
    velocities: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(velocities, axis=0)
    if unique.shape[0] < 3:
        raise ValueError("Energy-equilibrium velocity support must span two dimensions.")
    scale = max(float(np.max(np.abs(unique))), 1.0)
    tolerance = 128.0 * np.finfo(unique.dtype).eps * scale * scale

    lower: list[np.ndarray] = []
    for point in unique:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in unique[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)

    hull = np.stack(lower[:-1] + upper[:-1], axis=0)
    if hull.shape[0] < 3:
        raise ValueError("Energy-equilibrium velocity support must span two dimensions.")
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.sqrt(np.sum(edges * edges, axis=-1))
    signed_area_twice = float(
        np.sum(
            hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1]
        )
    )
    if (
        signed_area_twice <= tolerance
        or np.any(~np.isfinite(lengths))
        or np.any(lengths <= tolerance)
    ):
        raise ValueError("Energy-equilibrium velocity support must span two dimensions.")
    normals = np.stack((edges[:, 1], -edges[:, 0]), axis=-1) / lengths[:, None]
    offsets = np.sum(normals * hull, axis=-1)
    return unique, normals, offsets


class PositiveEnergyEquilibriumPlan(StrictModule, NonTrainableState):
    """Prepared positive total-energy equilibrium on a two-dimensional rule.

    The native population convention is weight absorbed: ``sum(g)`` is the
    total-energy density and ``sum(g[..., i] * c[i])`` is its flux. The dual
    parametrizes a quadrature-weighted exponential family. ``solve`` obtains
    that dual with a fixed-count masked Newton method, while ``evaluate``
    evaluates a supplied (for example learned) dual without claiming that its
    constitutive flux error vanishes.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    unique_velocities: Array
    halfspace_normals: Array
    halfspace_offsets: Array
    linear_solve: SmallLinearSolvePlan
    maximum_iterations: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    interior_tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    population_convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        /,
        *,
        maximum_iterations: int = 32,
        residual_tolerance: float = 1.0e-10,
        interior_tolerance: float = 1.0e-10,
        damping: float = 0.9,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if quadrature.dimension != 2:
            raise ValueError("Positive energy equilibrium requires dimension two.")
        if not jnp.issubdtype(quadrature.velocities.dtype, jnp.floating):
            raise TypeError(
                "Energy-equilibrium quadrature data must be real floating point."
            )
        iterations = int(maximum_iterations)
        residual = float(residual_tolerance)
        interior = float(interior_tolerance)
        damping_ = float(damping)
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if not isfinite(residual) or residual <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        if not isfinite(interior) or interior <= 0.0:
            raise ValueError("interior_tolerance must be finite and positive.")
        if not isfinite(damping_) or not 0.0 < damping_ <= 1.0:
            raise ValueError("damping must lie in (0, 1].")

        velocity_values = np.asarray(quadrature.velocities)
        unique, normals, offsets = _convex_hull_halfspaces(velocity_values)
        linear_solve = SmallLinearSolvePlan(2)
        self.quadrature = quadrature
        self.unique_velocities = jnp.asarray(unique, dtype=quadrature.velocities.dtype)
        self.halfspace_normals = jnp.asarray(normals, dtype=quadrature.velocities.dtype)
        self.halfspace_offsets = jnp.asarray(offsets, dtype=quadrature.velocities.dtype)
        self.linear_solve = linear_solve
        self.maximum_iterations = iterations
        self.residual_tolerance = residual
        self.interior_tolerance = interior
        self.damping = damping_
        self.dimension = 2
        self.population_convention = "weight_absorbed_total_energy_sum"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "positive-energy-equilibrium-plan",
                "quadrature": quadrature.quadrature_id,
                "unique_velocities": array_tree_fingerprint(unique),
                "halfspace_normals": array_tree_fingerprint(normals),
                "halfspace_offsets": array_tree_fingerprint(offsets),
                "linear_solve": linear_solve.plan_id,
                "maximum_iterations": iterations,
                "residual_tolerance": residual,
                "interior_tolerance": interior,
                "damping": damping_,
                "dimension": 2,
                "population_convention": "weight_absorbed_total_energy_sum",
            }
        )

    def _physical_inputs(
        self, total_energy: ArrayLike, target_flux: ArrayLike, /
    ) -> tuple[Array, Array]:
        energy = jnp.asarray(total_energy)
        flux = jnp.asarray(target_flux)
        if not jnp.issubdtype(energy.dtype, jnp.number) or jnp.issubdtype(
            energy.dtype, jnp.complexfloating
        ):
            raise TypeError("total_energy must be real numeric data.")
        if not jnp.issubdtype(flux.dtype, jnp.number) or jnp.issubdtype(
            flux.dtype, jnp.complexfloating
        ):
            raise TypeError("target_flux must be real numeric data.")
        if flux.ndim == 0 or flux.shape[-1] != self.dimension:
            raise ValueError("target_flux must have trailing shape (2,).")
        if flux.shape[:-1] != energy.shape:
            raise ValueError(
                "total_energy and target_flux must have identical leading batch axes."
            )
        dtype = self.quadrature.velocities.dtype
        return jnp.asarray(energy, dtype=dtype), jnp.asarray(flux, dtype=dtype)

    def _target_state(
        self, total_energy: Array, target_flux: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        finite = jnp.isfinite(total_energy) & jnp.all(jnp.isfinite(target_flux), axis=-1)
        positive_energy = total_energy > 0.0
        safe_energy = jnp.where(finite & positive_energy, total_energy, 1.0)
        safe_flux = jnp.where(jnp.isfinite(target_flux), target_flux, 0.0)
        normalized_target = safe_flux / safe_energy[..., None]
        halfspace_slack = self.halfspace_offsets - ein.contract(
            "hd,...d->...h", self.halfspace_normals, normalized_target
        )
        raw_margin = jnp.min(halfspace_slack, axis=-1)
        margin = jnp.where(finite & positive_energy, raw_margin, -jnp.inf)
        interior = finite & positive_energy & (margin > self.interior_tolerance)
        return (
            finite,
            positive_energy,
            normalized_target,
            margin,
            interior,
            safe_energy,
        )

    def _distribution(self, dual: Array, /) -> Array:
        logits = ein.contract("...d,qd->...q", dual, self.quadrature.velocities)
        shifted_logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        weighted = self.quadrature.weights * jnp.exp(shifted_logits)
        return weighted / jnp.sum(weighted, axis=-1, keepdims=True)

    def _statistics(self, dual: Array, /) -> tuple[Array, Array, Array]:
        distribution = self._distribution(dual)
        mean = ein.contract("...q,qd->...d", distribution, self.quadrature.velocities)
        centered = self.quadrature.velocities - mean[..., None, :]
        covariance = ein.contract(
            "...q,...qi,...qj->...ij", distribution, centered, centered
        )
        return distribution, mean, covariance

    def _base_status(
        self, finite: Array, positive_energy: Array, interior: Array, /
    ) -> Array:
        status = jnp.full(
            finite.shape,
            int(EnergyEquilibriumStatus.MAXIMUM_ITERATIONS),
            dtype=jnp.int32,
        )
        status = jnp.where(
            finite & ~positive_energy,
            int(EnergyEquilibriumStatus.NONPOSITIVE_TOTAL_ENERGY),
            status,
        )
        status = jnp.where(
            finite & positive_energy & ~interior,
            int(EnergyEquilibriumStatus.INFEASIBLE_TARGET),
            status,
        )
        return jnp.where(
            ~finite,
            int(EnergyEquilibriumStatus.NONFINITE_INPUT),
            status,
        ).astype(jnp.int32)

    def _result(
        self,
        total_energy: Array,
        target_flux: Array,
        normalized_target: Array,
        interior_margin: Array,
        populations: Array,
        dual: Array,
        residual_norm: Array,
        iterations: Array,
        status: Array,
        converged: Array,
        computation_finite: Array,
        /,
    ) -> EnergyEquilibriumResult:
        recovered_energy = jnp.sum(populations, axis=-1)
        recovered_flux = ein.contract(
            "...q,qd->...d", populations, self.quadrature.velocities
        )
        flux_residual = recovered_flux - target_flux
        flux_error_norm = jnp.sqrt(jnp.sum(flux_residual * flux_residual, axis=-1))
        output_finite = (
            jnp.all(jnp.isfinite(populations), axis=-1)
            & jnp.all(jnp.isfinite(dual), axis=-1)
            & jnp.isfinite(recovered_energy)
            & jnp.all(jnp.isfinite(recovered_flux), axis=-1)
        )
        positive = jnp.all(populations > 0.0, axis=-1)
        evidence = EnergyEquilibriumEvidence(
            target_total_energy=total_energy,
            recovered_total_energy=recovered_energy,
            total_energy_residual=recovered_energy - total_energy,
            target_flux=target_flux,
            recovered_flux=recovered_flux,
            flux_residual=flux_residual,
            normalized_target_flux=normalized_target,
            interior_margin=interior_margin,
            residual_norm=residual_norm,
            flux_error_norm=flux_error_norm,
            minimum_population=jnp.min(populations, axis=-1),
            iterations=iterations,
            status=status,
            finite=computation_finite & output_finite,
            positive=positive,
            converged=converged,
            quadrature_id=self.quadrature.quadrature_id,
            plan_id=self.plan_id,
        )
        return EnergyEquilibriumResult(populations, dual, evidence, self.plan_id)

    def solve(
        self, total_energy: ArrayLike, target_flux: ArrayLike, /
    ) -> EnergyEquilibriumResult:
        """Solve for the positive exponential-family equilibrium dual."""

        energy, flux = self._physical_inputs(total_energy, target_flux)
        (
            input_finite,
            positive_energy,
            normalized_target,
            margin,
            interior,
            safe_energy,
        ) = self._target_state(energy, flux)
        status = self._base_status(input_finite, positive_energy, interior)
        eligible = input_finite & positive_energy & interior

        dual = jnp.zeros(flux.shape, dtype=flux.dtype)
        distribution, mean, covariance = self._statistics(dual)
        residual = mean - normalized_target
        residual_norm = jnp.sqrt(jnp.sum(residual * residual, axis=-1))
        state_finite = (
            jnp.all(jnp.isfinite(distribution), axis=-1)
            & jnp.all(jnp.isfinite(mean), axis=-1)
            & jnp.all(jnp.isfinite(covariance), axis=(-2, -1))
            & jnp.isfinite(residual_norm)
        )
        state_positive = jnp.all(distribution > 0.0, axis=-1)
        converged = (
            eligible
            & state_finite
            & state_positive
            & (residual_norm <= self.residual_tolerance)
        )
        active = eligible & ~converged
        iterations = jnp.zeros(energy.shape, dtype=jnp.int32)
        linear_failed = jnp.zeros(energy.shape, dtype=bool)
        nonfinite_failed = eligible & ~state_finite
        nonpositive_failed = eligible & ~state_positive
        active = active & state_finite & state_positive

        def newton_step(_, carry):
            (
                dual_,
                distribution_,
                mean_,
                covariance_,
                residual_,
                residual_norm_,
                active_,
                converged_,
                iterations_,
                linear_failed_,
                nonfinite_failed_,
                nonpositive_failed_,
            ) = carry
            del distribution_, mean_, residual_norm_
            linear_result = solve_small_linear(self.linear_solve, covariance_, residual_)
            direction = linear_result.value
            solver_ok = linear_result.successful & jnp.all(
                jnp.isfinite(direction), axis=-1
            )
            attempted = active_
            iterations_ = iterations_ + attempted.astype(jnp.int32)
            linear_failed_ = linear_failed_ | (attempted & ~solver_ok)

            candidate_dual = dual_ - self.damping * direction
            candidate_finite = jnp.all(jnp.isfinite(candidate_dual), axis=-1)
            apply_candidate = attempted & solver_ok & candidate_finite
            nonfinite_failed_ = nonfinite_failed_ | (
                attempted & solver_ok & ~candidate_finite
            )
            dual_ = jnp.where(apply_candidate[..., None], candidate_dual, dual_)

            distribution_, mean_, covariance_ = self._statistics(dual_)
            residual_ = mean_ - normalized_target
            residual_norm_ = jnp.sqrt(jnp.sum(residual_ * residual_, axis=-1))
            state_finite_ = (
                jnp.all(jnp.isfinite(distribution_), axis=-1)
                & jnp.all(jnp.isfinite(mean_), axis=-1)
                & jnp.all(jnp.isfinite(covariance_), axis=(-2, -1))
                & jnp.isfinite(residual_norm_)
            )
            state_positive_ = jnp.all(distribution_ > 0.0, axis=-1)
            nonfinite_failed_ = nonfinite_failed_ | (apply_candidate & ~state_finite_)
            nonpositive_failed_ = nonpositive_failed_ | (
                apply_candidate & state_finite_ & ~state_positive_
            )
            candidate_converged = (
                apply_candidate
                & state_finite_
                & state_positive_
                & (residual_norm_ <= self.residual_tolerance)
            )
            converged_ = converged_ | candidate_converged
            active_ = (
                apply_candidate & state_finite_ & state_positive_ & ~candidate_converged
            )
            return (
                dual_,
                distribution_,
                mean_,
                covariance_,
                residual_,
                residual_norm_,
                active_,
                converged_,
                iterations_,
                linear_failed_,
                nonfinite_failed_,
                nonpositive_failed_,
            )

        (
            dual,
            distribution,
            mean,
            covariance,
            residual,
            residual_norm,
            active,
            converged,
            iterations,
            linear_failed,
            nonfinite_failed,
            nonpositive_failed,
        ) = lax.fori_loop(
            0,
            self.maximum_iterations,
            newton_step,
            (
                dual,
                distribution,
                mean,
                covariance,
                residual,
                residual_norm,
                active,
                converged,
                iterations,
                linear_failed,
                nonfinite_failed,
                nonpositive_failed,
            ),
        )

        raw_populations = safe_energy[..., None] * distribution
        raw_finite = jnp.all(jnp.isfinite(raw_populations), axis=-1)
        raw_positive = jnp.all(raw_populations > 0.0, axis=-1)
        nonfinite_failed = nonfinite_failed | (eligible & ~raw_finite)
        nonpositive_failed = nonpositive_failed | (eligible & raw_finite & ~raw_positive)
        successful = eligible & converged & raw_finite & raw_positive
        status = jnp.where(
            eligible & linear_failed,
            int(EnergyEquilibriumStatus.LINEAR_SOLVE_FAILED),
            status,
        )
        status = jnp.where(
            eligible & ~linear_failed & nonfinite_failed,
            int(EnergyEquilibriumStatus.NONFINITE_OUTPUT),
            status,
        )
        status = jnp.where(
            eligible & ~linear_failed & ~nonfinite_failed & nonpositive_failed,
            int(EnergyEquilibriumStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            successful,
            int(EnergyEquilibriumStatus.SUCCESS),
            status,
        ).astype(jnp.int32)
        populations = jnp.where(successful[..., None], raw_populations, 0.0)
        accepted_dual = jnp.where(successful[..., None], dual, 0.0)
        return self._result(
            energy,
            flux,
            normalized_target,
            margin,
            populations,
            accepted_dual,
            residual_norm,
            iterations,
            status,
            converged & successful,
            input_finite & state_finite & raw_finite,
        )

    def evaluate(
        self,
        total_energy: ArrayLike,
        target_flux: ArrayLike,
        dual: ArrayLike,
        /,
    ) -> EnergyEquilibriumResult:
        """Evaluate a supplied dual and expose its constitutive flux error."""

        energy, flux = self._physical_inputs(total_energy, target_flux)
        dual_value = jnp.asarray(dual)
        if not jnp.issubdtype(dual_value.dtype, jnp.number) or jnp.issubdtype(
            dual_value.dtype, jnp.complexfloating
        ):
            raise TypeError("dual must be real numeric data.")
        if dual_value.shape != flux.shape:
            raise ValueError("dual must have the same shape as target_flux.")
        dual_value = jnp.asarray(dual_value, dtype=flux.dtype)
        (
            physical_finite,
            positive_energy,
            normalized_target,
            margin,
            interior,
            safe_energy,
        ) = self._target_state(energy, flux)
        dual_finite = jnp.all(jnp.isfinite(dual_value), axis=-1)
        input_finite = physical_finite & dual_finite
        status = self._base_status(input_finite, positive_energy, interior)
        eligible = input_finite & positive_energy & interior

        safe_dual = jnp.where(dual_finite[..., None], dual_value, 0.0)
        distribution, mean, _ = self._statistics(safe_dual)
        residual = mean - normalized_target
        residual_norm = jnp.sqrt(jnp.sum(residual * residual, axis=-1))
        raw_populations = safe_energy[..., None] * distribution
        output_finite = (
            jnp.all(jnp.isfinite(distribution), axis=-1)
            & jnp.all(jnp.isfinite(raw_populations), axis=-1)
            & jnp.all(jnp.isfinite(mean), axis=-1)
            & jnp.isfinite(residual_norm)
        )
        output_positive = jnp.all(raw_populations > 0.0, axis=-1)
        successful = eligible & output_finite & output_positive
        status = jnp.where(
            eligible & ~output_finite,
            int(EnergyEquilibriumStatus.NONFINITE_OUTPUT),
            status,
        )
        status = jnp.where(
            eligible & output_finite & ~output_positive,
            int(EnergyEquilibriumStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            successful,
            int(EnergyEquilibriumStatus.SUCCESS),
            status,
        ).astype(jnp.int32)
        populations = jnp.where(successful[..., None], raw_populations, 0.0)
        accepted_dual = jnp.where(successful[..., None], dual_value, 0.0)
        converged = successful & (residual_norm <= self.residual_tolerance)
        return self._result(
            energy,
            flux,
            normalized_target,
            margin,
            populations,
            accepted_dual,
            residual_norm,
            jnp.zeros(energy.shape, dtype=jnp.int32),
            status,
            converged,
            input_finite & output_finite,
        )


__all__ = [
    "EnergyEquilibriumEvidence",
    "EnergyEquilibriumResult",
    "EnergyEquilibriumStatus",
    "PositiveEnergyEquilibriumPlan",
]
