#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..equations._homogeneous_thermodynamics import HomogeneousHelmholtzPlan
from ..equations._peng_robinson import (
    peng_robinson_roots,
    PengRobinsonResidualHelmholtzTerm,
)
from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS


class PhaseEquilibriumStatus(IntEnum):
    SUCCESS_SINGLE_PHASE = 0
    SUCCESS_TWO_PHASE = 1
    NO_INSTABILITY_FOUND = 2
    INDETERMINATE = 3
    PURE_COEXISTENCE_UNDERDETERMINED = 4
    CRITICAL_DEGENERACY = 5
    SOLVER_FAILED = 6


class TPDStabilityResult(StrictModule):
    minimum_tpd: Array
    trial_composition: Array
    trial_root_index: Array
    reference_root_index: Array
    unstable: Array
    successful: Array
    status: Array
    stationarity_norm: Array
    stability_margin: Array
    model_id: str = eqx.field(static=True)


class TPDSearchPlan(StrictModule):
    thermodynamics: HomogeneousHelmholtzPlan
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        /,
        *,
        tolerance: float = 1.0e-8,
        maximum_steps: int = 100,
    ) -> None:
        if not isinstance(thermodynamics.residual, PengRobinsonResidualHelmholtzTerm):
            raise TypeError(
                "TPD stability currently requires Peng-Robinson thermodynamics."
            )
        tolerance_value = float(tolerance)
        steps = int(maximum_steps)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if steps <= 0:
            raise ValueError("maximum_steps must be positive.")
        self.thermodynamics = thermodynamics
        self.tolerance = tolerance_value
        self.maximum_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tpd-search",
                "thermodynamics": thermodynamics.model_id,
                "tolerance": tolerance_value,
                "maximum_steps": steps,
            }
        )

    def solve(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        feed_composition: ArrayLike,
        /,
    ) -> TPDStabilityResult:
        temperature_value = jnp.asarray(temperature)
        pressure_value = jnp.asarray(pressure)
        feed = jnp.asarray(feed_composition)
        count = self.thermodynamics.schema.component_count
        if temperature_value.shape != () or pressure_value.shape != ():
            raise ValueError("TPD search currently accepts one state at a time.")
        if feed.shape != (count,):
            raise ValueError("feed_composition must have shape (component_count,).")
        if bool(np.any(np.asarray(feed) < 0.0)) or not np.isclose(
            float(jnp.sum(feed)), 1.0, rtol=0.0, atol=self.tolerance
        ):
            raise ValueError("feed_composition must be nonnegative and normalized.")
        roots = peng_robinson_roots(
            self.thermodynamics, temperature_value, pressure_value, feed
        )
        reference_index = roots.minimum_gibbs_index
        reference = _chemical_at_root(
            self.thermodynamics,
            temperature_value,
            pressure_value,
            feed,
            reference_index,
        )
        feed_log_activity = jnp.log(jnp.maximum(feed, jnp.finfo(feed.dtype).tiny)) + (
            reference.log_fugacity_coefficient
        )
        starts = _tpd_starts(
            self.thermodynamics.residual,
            temperature_value,
            pressure_value,
            feed,
        )
        best_value = jnp.asarray(0.0, dtype=feed.dtype)
        best_composition = feed
        best_root = reference_index
        best_stationarity = jnp.asarray(0.0, dtype=feed.dtype)

        start_array = jnp.stack(starts)
        for dense in (True, False):

            def objective(logits, _, dense=dense):
                composition = jax.nn.softmax(logits)
                root_set = peng_robinson_roots(
                    self.thermodynamics,
                    temperature_value,
                    pressure_value,
                    composition,
                )
                root_index = _root_index(root_set.stable, dense=dense)
                chemical = _chemical_at_root(
                    self.thermodynamics,
                    temperature_value,
                    pressure_value,
                    composition,
                    root_index,
                )
                trial_log_activity = (
                    jnp.log(jnp.maximum(composition, jnp.finfo(composition.dtype).tiny))
                    + chemical.log_fugacity_coefficient
                )
                tpd = jnp.sum(composition * (trial_log_activity - feed_log_activity))
                valid = root_set.stable[root_index] & chemical.successful
                return jnp.where(
                    valid,
                    tpd,
                    jnp.asarray(1.0e6, dtype=tpd.dtype),
                )

            direct_values = jax.vmap(lambda start: objective(start, None))(start_array)
            direct_index = jnp.argmin(direct_values)
            direct_composition = jax.nn.softmax(start_array[direct_index])
            direct_root_set = peng_robinson_roots(
                self.thermodynamics,
                temperature_value,
                pressure_value,
                direct_composition,
            )
            direct_root = _root_index(direct_root_set.stable, dense=dense)
            direct_select = direct_values[direct_index] < best_value
            best_value = jnp.where(direct_select, direct_values[direct_index], best_value)
            best_composition = jnp.where(
                direct_select, direct_composition, best_composition
            )
            best_root = jnp.where(direct_select, direct_root, best_root)
            best_stationarity = jnp.where(
                direct_select, jnp.asarray(jnp.inf, dtype=feed.dtype), best_stationarity
            )

            def solve_start(start):
                return minimize(
                    objective,
                    start,
                    bounds=Bounds(
                        jnp.full_like(start, -80.0),
                        jnp.full_like(start, 80.0),
                    ),
                    method=ProjectedLBFGS(),
                    termination=OptimizationTermination(
                        absolute_optimality=self.tolerance,
                        relative_optimality=0.0,
                        maximum_steps=self.maximum_steps,
                    ),
                )

            solved = jax.vmap(solve_start)(start_array)
            values = jnp.where(solved.successful, solved.objective, jnp.inf)
            selected_index = jnp.argmin(values)
            composition = jax.nn.softmax(solved.parameters[selected_index])
            value = values[selected_index]
            root_set = peng_robinson_roots(
                self.thermodynamics,
                temperature_value,
                pressure_value,
                composition,
            )
            root_index = _root_index(root_set.stable, dense=dense)
            select = value < best_value
            best_value = jnp.where(select, value, best_value)
            best_composition = jnp.where(select, composition, best_composition)
            best_root = jnp.where(select, root_index, best_root)
            best_stationarity = jnp.where(
                select,
                solved.diagnostics.final_optimality_norm[selected_index],
                best_stationarity,
            )
        unstable = best_value < -self.tolerance
        successful = roots.successful & jnp.isfinite(best_value)
        status = jnp.where(
            successful,
            jnp.where(
                unstable,
                int(PhaseEquilibriumStatus.SUCCESS_TWO_PHASE),
                int(PhaseEquilibriumStatus.NO_INSTABILITY_FOUND),
            ),
            int(PhaseEquilibriumStatus.INDETERMINATE),
        ).astype(jnp.int32)
        return TPDStabilityResult(
            best_value,
            best_composition,
            best_root,
            reference_index,
            unstable,
            successful,
            status,
            best_stationarity,
            jnp.abs(best_value),
            self.plan_id,
        )


class FixedTwoPhaseTPFlashResult(StrictModule):
    phase_fraction: Array
    phase_composition: Array
    phase_molar_density: Array
    phase_root_index: Array
    active_phase: Array
    material_residual: Array
    fugacity_residual: Array
    stability: TPDStabilityResult
    iteration_count: Array
    status: Array
    derivative_valid: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class FixedTwoPhaseTPFlashPlan(StrictModule):
    thermodynamics: HomogeneousHelmholtzPlan
    stability: TPDSearchPlan
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        /,
        *,
        tolerance: float = 1.0e-8,
        maximum_steps: int = 80,
        damping: float = 0.5,
    ) -> None:
        if not isinstance(thermodynamics.residual, PengRobinsonResidualHelmholtzTerm):
            raise TypeError("TP flash currently requires Peng-Robinson thermodynamics.")
        tolerance_value = float(tolerance)
        steps = int(maximum_steps)
        damping_value = float(damping)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if steps <= 0 or not 0.0 < damping_value <= 1.0:
            raise ValueError("Flash iteration controls are invalid.")
        self.thermodynamics = thermodynamics
        self.stability = TPDSearchPlan(
            thermodynamics, tolerance=tolerance_value, maximum_steps=steps
        )
        self.tolerance = tolerance_value
        self.maximum_steps = steps
        self.damping = damping_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-two-phase-tp-flash",
                "thermodynamics": thermodynamics.model_id,
                "stability": self.stability.plan_id,
                "tolerance": tolerance_value,
                "maximum_steps": steps,
                "damping": damping_value,
            }
        )

    def solve(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        feed_composition: ArrayLike,
        /,
    ) -> FixedTwoPhaseTPFlashResult:
        temperature_value = jnp.asarray(temperature)
        pressure_value = jnp.asarray(pressure)
        feed = jnp.asarray(feed_composition)
        stability = self.stability.solve(temperature_value, pressure_value, feed)
        roots = peng_robinson_roots(
            self.thermodynamics, temperature_value, pressure_value, feed
        )
        if not bool(np.asarray(stability.successful)):
            return self._failure(feed, stability, PhaseEquilibriumStatus.INDETERMINATE)
        if not bool(np.asarray(stability.unstable)):
            reference = int(np.asarray(roots.minimum_gibbs_index))
            dense = reference == int(np.asarray(_root_index(roots.stable, dense=True)))
            fraction = jnp.asarray((1.0, 0.0) if dense else (0.0, 1.0))
            composition = jnp.stack((feed, feed))
            density = jnp.asarray(
                (roots.molar_density[reference], roots.molar_density[reference])
            )
            root_index = jnp.asarray((reference, reference), dtype=jnp.int32)
            active = fraction > 0.0
            return FixedTwoPhaseTPFlashResult(
                fraction,
                composition,
                density,
                root_index,
                active,
                jnp.zeros_like(feed),
                jnp.zeros_like(feed),
                stability,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(
                    int(PhaseEquilibriumStatus.SUCCESS_SINGLE_PHASE), dtype=jnp.int32
                ),
                jnp.asarray(True),
                jnp.asarray(True),
                self.plan_id,
            )

        log_k = jnp.log(
            _wilson_k(
                self.thermodynamics.residual,
                temperature_value,
                pressure_value,
            )
        )
        phase_fraction = jnp.asarray(0.5, dtype=feed.dtype)
        dense_composition = feed
        dilute_composition = feed
        dense_root = jnp.asarray(0, dtype=jnp.int32)
        dilute_root = jnp.asarray(0, dtype=jnp.int32)

        for _ in range(self.maximum_steps):
            k_value = jnp.exp(jnp.clip(log_k, -80.0, 80.0))
            phase_fraction = _rachford_rice(feed, k_value)
            denominator = 1.0 + phase_fraction * (k_value - 1.0)
            dense_composition = feed / denominator
            dilute_composition = k_value * dense_composition
            dense_composition = dense_composition / jnp.sum(dense_composition)
            dilute_composition = dilute_composition / jnp.sum(dilute_composition)
            dense_roots = peng_robinson_roots(
                self.thermodynamics,
                temperature_value,
                pressure_value,
                dense_composition,
            )
            dilute_roots = peng_robinson_roots(
                self.thermodynamics,
                temperature_value,
                pressure_value,
                dilute_composition,
            )
            dense_root = _root_index(dense_roots.stable, dense=True)
            dilute_root = _root_index(dilute_roots.stable, dense=False)
            dense_chemical = _chemical_at_root(
                self.thermodynamics,
                temperature_value,
                pressure_value,
                dense_composition,
                dense_root,
            )
            dilute_chemical = _chemical_at_root(
                self.thermodynamics,
                temperature_value,
                pressure_value,
                dilute_composition,
                dilute_root,
            )
            target = (
                dense_chemical.log_fugacity_coefficient
                - dilute_chemical.log_fugacity_coefficient
            )
            log_k = (1.0 - self.damping) * log_k + self.damping * target

        k_value = jnp.exp(jnp.clip(log_k, -80.0, 80.0))
        denominator = 1.0 + phase_fraction * (k_value - 1.0)
        material = (
            (1.0 - phase_fraction) * dense_composition
            + phase_fraction * dilute_composition
            - feed
        )
        dense_chemical = _chemical_at_root(
            self.thermodynamics,
            temperature_value,
            pressure_value,
            dense_composition,
            dense_root,
        )
        dilute_chemical = _chemical_at_root(
            self.thermodynamics,
            temperature_value,
            pressure_value,
            dilute_composition,
            dilute_root,
        )
        fugacity = log_k - (
            dense_chemical.log_fugacity_coefficient
            - dilute_chemical.log_fugacity_coefficient
        )
        dense_roots = peng_robinson_roots(
            self.thermodynamics,
            temperature_value,
            pressure_value,
            dense_composition,
        )
        dilute_roots = peng_robinson_roots(
            self.thermodynamics,
            temperature_value,
            pressure_value,
            dilute_composition,
        )
        successful = (
            jnp.all(denominator > 0.0)
            & dense_roots.stable[dense_root]
            & dilute_roots.stable[dilute_root]
            & jnp.all(jnp.abs(material) <= self.tolerance)
            & jnp.all(jnp.abs(fugacity) <= 10.0 * self.tolerance)
            & (phase_fraction > 0.0)
            & (phase_fraction < 1.0)
        )
        status = jnp.where(
            successful,
            int(PhaseEquilibriumStatus.SUCCESS_TWO_PHASE),
            int(PhaseEquilibriumStatus.SOLVER_FAILED),
        ).astype(jnp.int32)
        return FixedTwoPhaseTPFlashResult(
            jnp.asarray((1.0 - phase_fraction, phase_fraction)),
            jnp.stack((dense_composition, dilute_composition)),
            jnp.asarray(
                (
                    dense_roots.molar_density[dense_root],
                    dilute_roots.molar_density[dilute_root],
                )
            ),
            jnp.asarray((dense_root, dilute_root), dtype=jnp.int32),
            jnp.asarray((True, True)),
            material,
            fugacity,
            stability,
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            status,
            successful,
            successful,
            self.plan_id,
        )

    def _failure(self, feed, stability, status):
        count = feed.shape[0]
        nan = jnp.asarray(jnp.nan, dtype=feed.dtype)
        return FixedTwoPhaseTPFlashResult(
            jnp.full((2,), nan),
            jnp.full((2, count), nan),
            jnp.full((2,), nan),
            jnp.full((2,), -1, dtype=jnp.int32),
            jnp.zeros((2,), dtype=bool),
            jnp.full((count,), nan),
            jnp.full((count,), nan),
            stability,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(status), dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
            self.plan_id,
        )


def _chemical_at_root(thermodynamics, temperature, pressure, composition, root_index):
    roots = peng_robinson_roots(thermodynamics, temperature, pressure, composition)
    density = roots.molar_density[root_index]
    return thermodynamics.evaluate_chemical(temperature, density, composition)


def _root_index(stable, *, dense: bool):
    if dense:
        return jnp.argmax(stable).astype(jnp.int32)
    return (stable.shape[0] - 1 - jnp.argmax(stable[::-1])).astype(jnp.int32)


def _wilson_k(residual, temperature, pressure):
    parameters = residual.parameters
    return (
        parameters.critical_pressure.astype(temperature.dtype)
        / pressure
        * jnp.exp(
            5.373
            * (1.0 + parameters.acentric_factor.astype(temperature.dtype))
            * (
                1.0
                - parameters.critical_temperature.astype(temperature.dtype) / temperature
            )
        )
    )


def _tpd_starts(residual, temperature, pressure, feed):
    tiny = jnp.finfo(feed.dtype).tiny
    wilson = _wilson_k(residual, temperature, pressure)
    starts = [jnp.log(jnp.maximum(feed, tiny))]
    starts.append(jnp.log(jnp.maximum(feed * wilson, tiny)))
    starts.append(jnp.log(jnp.maximum(feed / wilson, tiny)))
    for index in range(feed.shape[0]):
        enriched = jnp.full_like(feed, 1.0e-6)
        enriched = enriched.at[index].set(1.0)
        starts.append(jnp.log(enriched / jnp.sum(enriched)))
    return tuple(starts)


def _rachford_rice(feed, equilibrium_ratio):
    def residual(beta):
        return jnp.sum(
            feed * (equilibrium_ratio - 1.0) / (1.0 + beta * (equilibrium_ratio - 1.0))
        )

    def body(_, bounds):
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        value = residual(midpoint)
        return (
            jnp.where(value > 0.0, midpoint, lower),
            jnp.where(value > 0.0, upper, midpoint),
        )

    lower, upper = jax.lax.fori_loop(
        0,
        80,
        body,
        (jnp.asarray(0.0, dtype=feed.dtype), jnp.asarray(1.0, dtype=feed.dtype)),
    )
    return 0.5 * (lower + upper)


__all__ = [
    "FixedTwoPhaseTPFlashPlan",
    "FixedTwoPhaseTPFlashResult",
    "PhaseEquilibriumStatus",
    "TPDSearchPlan",
    "TPDStabilityResult",
]
