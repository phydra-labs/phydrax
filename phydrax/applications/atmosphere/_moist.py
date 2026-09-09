#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constant-caloric moist air with stable liquid/ice phase equilibrium.

All fractions are per total moist mass, including condensate. Condensed phases
have negligible volume. Mixed liquid/ice exists only at the reference freezing
temperature; no metastability, ice nucleation, or empirical mixed-phase band is
implied. Saturation follows the integrated Clausius--Clapeyron relation using
exactly the enthalpy differences of this caloric model.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import SmallRootKernel


class MoistAdjustmentResult(StrictModule):
    temperature: Array
    vapor: Array
    liquid: Array
    ice: Array
    density: Array
    pressure: Array
    successful: Array
    residual: Array
    derivative_valid: Array


class MoistThermodynamicPlan(StrictModule, NonTrainableState):
    """An ideal dry/vapor gas and incompressible liquid/ice caloric closure.

    ``adjust`` conserves specific internal energy at fixed total density;
    ``adjust_isobaric`` conserves specific enthalpy at fixed total pressure.
    Residuals are signed J/kg. Certification and derivative validity are
    elementwise; callers must reject failed cells, never silently commit them.
    """

    dry_gas_constant: float = eqx.field(static=True)
    vapor_gas_constant: float = eqx.field(static=True)
    dry_cv: float = eqx.field(static=True)
    vapor_cv: float = eqx.field(static=True)
    liquid_heat_capacity: float = eqx.field(static=True)
    ice_heat_capacity: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    reference_saturation_pressure: float = eqx.field(static=True)
    latent_vaporization: float = eqx.field(static=True)
    latent_fusion: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        dry_gas_constant: float = 287.05,
        vapor_gas_constant: float = 461.5,
        dry_cv: float = 717.5,
        vapor_cv: float = 1410.0,
        liquid_heat_capacity: float = 4186.0,
        ice_heat_capacity: float = 2106.0,
        reference_temperature: float = 273.16,
        reference_saturation_pressure: float = 611.657,
        latent_vaporization: float = 2500800.0,
        latent_fusion: float = 333700.0,
        minimum_temperature: float = 150.0,
        maximum_temperature: float = 400.0,
        maximum_steps: int = 24,
        energy_tolerance: float = 1e-6,
    ):
        constants = dict(
            dry_gas_constant=float(dry_gas_constant),
            vapor_gas_constant=float(vapor_gas_constant),
            dry_cv=float(dry_cv),
            vapor_cv=float(vapor_cv),
            liquid_heat_capacity=float(liquid_heat_capacity),
            ice_heat_capacity=float(ice_heat_capacity),
            reference_temperature=float(reference_temperature),
            reference_saturation_pressure=float(reference_saturation_pressure),
            latent_vaporization=float(latent_vaporization),
            latent_fusion=float(latent_fusion),
            minimum_temperature=float(minimum_temperature),
            maximum_temperature=float(maximum_temperature),
            energy_tolerance=float(energy_tolerance),
        )
        if any(not math.isfinite(x) or x <= 0 for x in constants.values()):
            raise ValueError(
                "Moist caloric constants and tolerances must be positive and finite."
            )
        if not minimum_temperature < reference_temperature < maximum_temperature:
            raise ValueError(
                "Temperature domain must contain the reference freezing point."
            )
        if int(maximum_steps) != maximum_steps or maximum_steps < 1:
            raise ValueError("maximum_steps must be a positive integer.")
        cp_v = vapor_cv + vapor_gas_constant
        # Positive latent internal energies make each stable caloric branch monotone.
        for t in (minimum_temperature, maximum_temperature):
            lv = latent_vaporization + (cp_v - liquid_heat_capacity) * (
                t - reference_temperature
            )
            ls = (
                latent_vaporization
                + latent_fusion
                + (cp_v - ice_heat_capacity) * (t - reference_temperature)
            )
            lf = latent_fusion + (liquid_heat_capacity - ice_heat_capacity) * (
                t - reference_temperature
            )
            if min(lv - vapor_gas_constant * t, ls - vapor_gas_constant * t, lf) <= 0:
                raise ValueError(
                    "Caloric parameters require positive latent energies over the domain."
                )
        self.dry_gas_constant = constants["dry_gas_constant"]
        self.vapor_gas_constant = constants["vapor_gas_constant"]
        self.dry_cv = constants["dry_cv"]
        self.vapor_cv = constants["vapor_cv"]
        self.liquid_heat_capacity = constants["liquid_heat_capacity"]
        self.ice_heat_capacity = constants["ice_heat_capacity"]
        self.reference_temperature = constants["reference_temperature"]
        self.reference_saturation_pressure = constants["reference_saturation_pressure"]
        self.latent_vaporization = constants["latent_vaporization"]
        self.latent_fusion = constants["latent_fusion"]
        self.minimum_temperature = constants["minimum_temperature"]
        self.maximum_temperature = constants["maximum_temperature"]
        self.energy_tolerance = constants["energy_tolerance"]
        self.maximum_steps = int(maximum_steps)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stable-moist-caloric-equilibrium",
                **constants,
                "maximum_steps": self.maximum_steps,
                "mixed_phase": "freezing-coexistence-only",
                "condensate_volume": "zero",
            }
        )

    def phase_energies(self, temperature: ArrayLike) -> tuple[Array, Array, Array, Array]:
        t = jnp.asarray(temperature)
        delta = t - self.reference_temperature
        return (
            self.dry_cv * delta,
            self.vapor_cv * delta
            + self.latent_vaporization
            - self.vapor_gas_constant * self.reference_temperature,
            self.liquid_heat_capacity * delta,
            self.ice_heat_capacity * delta - self.latent_fusion,
        )

    def phase_enthalpies(
        self, temperature: ArrayLike
    ) -> tuple[Array, Array, Array, Array]:
        t = jnp.asarray(temperature)
        dry, vapor, liquid, ice = self.phase_energies(t)
        return (
            dry + self.dry_gas_constant * t,
            vapor + self.vapor_gas_constant * t,
            liquid,
            ice,
        )

    def gas_constant(self, vapor: ArrayLike, liquid: ArrayLike, ice: ArrayLike) -> Array:
        qv, ql, qi = jnp.asarray(vapor), jnp.asarray(liquid), jnp.asarray(ice)
        return (1.0 - qv - ql - qi) * self.dry_gas_constant + qv * self.vapor_gas_constant

    def heat_capacity(
        self,
        vapor: ArrayLike,
        liquid: ArrayLike,
        ice: ArrayLike,
        *,
        at_constant_pressure: bool = False,
    ) -> Array:
        qv, ql, qi = jnp.asarray(vapor), jnp.asarray(liquid), jnp.asarray(ice)
        cv = (
            (1.0 - qv - ql - qi) * self.dry_cv
            + qv * self.vapor_cv
            + ql * self.liquid_heat_capacity
            + qi * self.ice_heat_capacity
        )
        return cv + self.gas_constant(qv, ql, qi) if at_constant_pressure else cv

    def energy(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        vapor: ArrayLike,
        liquid: ArrayLike,
        ice: ArrayLike,
    ) -> Array:
        """Specific internal energy (J/kg total moist mass); density-independent here."""
        qv, ql, qi = jnp.asarray(vapor), jnp.asarray(liquid), jnp.asarray(ice)
        ed, ev, el, ei = self.phase_energies(temperature)
        value = (1.0 - qv - ql - qi) * ed + qv * ev + ql * el + qi * ei
        return jnp.broadcast_arrays(value, jnp.asarray(density))[0]

    def enthalpy(
        self, temperature: ArrayLike, vapor: ArrayLike, liquid: ArrayLike, ice: ArrayLike
    ) -> Array:
        return self.energy(1.0, temperature, vapor, liquid, ice) + self.gas_constant(
            vapor, liquid, ice
        ) * jnp.asarray(temperature)

    def pressure(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        vapor: ArrayLike,
        liquid: ArrayLike,
        ice: ArrayLike,
    ) -> Array:
        return (
            jnp.asarray(density)
            * self.gas_constant(vapor, liquid, ice)
            * jnp.asarray(temperature)
        )

    def saturation_pressure(
        self, temperature: ArrayLike, *, phase: str = "liquid"
    ) -> Array:
        """Pa, with d(log e_s)/dT = latent_enthalpy(T)/(Rv*T**2)."""
        if phase not in ("liquid", "ice"):
            raise ValueError("Saturation phase must be liquid or ice.")
        t = jnp.asarray(temperature)
        cp_v = self.vapor_cv + self.vapor_gas_constant
        cp_c = self.liquid_heat_capacity if phase == "liquid" else self.ice_heat_capacity
        latent = self.latent_vaporization + (
            0.0 if phase == "liquid" else self.latent_fusion
        )
        dcp = cp_v - cp_c
        exponent = (
            (latent - dcp * self.reference_temperature)
            * (1.0 / self.reference_temperature - 1.0 / t)
            + dcp * jnp.log(t / self.reference_temperature)
        ) / self.vapor_gas_constant
        return self.reference_saturation_pressure * jnp.exp(exponent)

    def _partition(self, constraint, temperature, total_water, frozen, isobaric):
        es = jnp.where(
            frozen,
            self.saturation_pressure(temperature, phase="ice"),
            self.saturation_pressure(temperature),
        )
        if isobaric:
            # Above the boiling-pressure limit all available water is vapor.
            denominator = jnp.where(constraint > es, constraint - es, 1.0)
            saturated = (
                (1.0 - total_water)
                * self.dry_gas_constant
                * es
                / (self.vapor_gas_constant * denominator)
            )
            saturated = jnp.where(es >= constraint, total_water, saturated)
        else:
            saturated = es / (constraint * self.vapor_gas_constant * temperature)
        vapor = jnp.minimum(total_water, saturated)
        condensate = total_water - vapor
        return (
            vapor,
            jnp.where(frozen, 0.0, condensate),
            jnp.where(frozen, condensate, 0.0),
        )

    def equilibrium(
        self, density: ArrayLike, temperature: ArrayLike, total_water: ArrayLike
    ) -> MoistAdjustmentResult:
        rho, t, qt = self._arrays(density, temperature, total_water)
        qv, ql, qi = self._partition(rho, t, qt, t < self.reference_temperature, False)
        p = self.pressure(rho, t, qv, ql, qi)
        valid = self._valid(rho, t, qt) & jnp.isfinite(p) & (p > 0)
        regular = self._regular(rho, t, qt, qv, ql, qi, False)
        return MoistAdjustmentResult(
            t, qv, ql, qi, rho, p, valid, jnp.zeros_like(t), valid & regular
        )

    def adjust(
        self,
        density: ArrayLike,
        total_water: ArrayLike,
        specific_internal_energy: ArrayLike,
    ) -> MoistAdjustmentResult:
        return self._adjust(density, total_water, specific_internal_energy, False)

    def adjust_isobaric(
        self, pressure: ArrayLike, total_water: ArrayLike, specific_enthalpy: ArrayLike
    ) -> MoistAdjustmentResult:
        return self._adjust(pressure, total_water, specific_enthalpy, True)

    def _arrays(self, a, b, c):
        a, b, c = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)
        dtype = jnp.result_type(a.dtype, b.dtype, c.dtype, jnp.float32)
        return jnp.broadcast_arrays(
            jnp.asarray(a, dtype=dtype),
            jnp.asarray(b, dtype=dtype),
            jnp.asarray(c, dtype=dtype),
        )

    def _valid(self, constraint, temperature, qt):
        return (
            jnp.isfinite(constraint)
            & (constraint > 0)
            & jnp.isfinite(temperature)
            & (temperature >= self.minimum_temperature)
            & (temperature <= self.maximum_temperature)
            & jnp.isfinite(qt)
            & (qt >= 0)
            & (qt < 1)
        )

    def _caloric(self, constraint, temperature, qt, frozen, isobaric):
        qv, ql, qi = self._partition(constraint, temperature, qt, frozen, isobaric)
        return (
            self.enthalpy(temperature, qv, ql, qi)
            if isobaric
            else self.energy(constraint, temperature, qv, ql, qi)
        )

    def _regular(self, constraint, t, qt, qv, ql, qi, isobaric):
        es = jnp.where(
            t < self.reference_temperature,
            self.saturation_pressure(t, phase="ice"),
            self.saturation_pressure(t),
        )
        rho = constraint / (self.gas_constant(qv, ql, qi) * t) if isobaric else constraint
        vapor_pressure = rho * qv * self.vapor_gas_constant * t
        # Saturated interiors are regular; the onset and freezing endpoints are not.
        near_onset = (jnp.abs(vapor_pressure - es) <= 1e-5 * es) & ((ql + qi) <= 1e-7)
        near_freezing = (jnp.abs(t - self.reference_temperature) <= 1e-5) & (qt > 0)
        return (
            ~near_onset
            & ~near_freezing
            & (t > self.minimum_temperature)
            & (t < self.maximum_temperature)
        )

    def _adjust(self, constraint, total_water, specific_energy, isobaric):
        c, qt, target = self._arrays(constraint, total_water, specific_energy)
        shape = c.shape
        freezing = jnp.full_like(c, self.reference_temperature)
        ice_limit = self._caloric(c, freezing, qt, True, isobaric)
        liquid_limit = self._caloric(c, freezing, qt, False, isobaric)
        coexistence = (
            (liquid_limit > ice_limit) & (target >= ice_limit) & (target <= liquid_limit)
        )
        frozen = target < ice_limit
        lower = jnp.where(frozen, self.minimum_temperature, self.reference_temperature)
        upper = jnp.where(frozen, self.reference_temperature, self.maximum_temperature)
        lower_energy = self._caloric(c, lower, qt, frozen, isobaric)
        upper_energy = self._caloric(c, upper, qt, frozen, isobaric)
        bracketed = coexistence | ((target >= lower_energy) & (target <= upper_energy))
        initial = jnp.clip(
            lower
            + (upper - lower)
            * (target - lower_energy)
            / jnp.maximum(upper_energy - lower_energy, 1.0),
            lower,
            upper,
        )
        initial = jnp.where(coexistence, freezing, initial)
        scale = self.dry_cv
        arguments = tuple(x.reshape(-1) for x in (c, qt, target, frozen, coexistence))

        def residual_one(t_vector, args):
            constraint_, water_, energy_, frozen_, coexistence_ = args
            t = t_vector[0]
            residual = (
                self._caloric(constraint_, t, water_, frozen_, isobaric) - energy_
            ) / scale
            return jnp.reshape(
                jnp.where(coexistence_, t - self.reference_temperature, residual), (1,)
            )

        def residual_all(temperatures):
            return jax.vmap(residual_one)(temperatures[:, None], arguments)[:, 0]

        tolerance = max(
            self.energy_tolerance / scale,
            8.0 * float(jnp.finfo(c.dtype).eps) * self.maximum_temperature,
        )
        kernel = SmallRootKernel(
            residual_one,
            maximum_dimension=1,
            maximum_steps=self.maximum_steps,
            absolute_tolerance=tolerance,
            relative_tolerance=0.0,
        )

        def solve(function, guess):
            del function
            return kernel.solve(guess[:, None], arguments).state[:, 0]

        def tangent_solve(linearized, right_hand_side):
            return right_hand_side / linearized(jnp.ones_like(right_hand_side))

        t = jax.lax.custom_root(
            residual_all, initial.reshape(-1), solve, tangent_solve
        ).reshape(shape)
        qv, ql, qi = self._partition(c, t, qt, frozen, isobaric)
        coexist_vapor, _, coexist_ice = self._partition(c, freezing, qt, True, isobaric)
        coexist_liquid = jnp.clip(
            (target - ice_limit) / self.latent_fusion, 0.0, coexist_ice
        )
        qv = jnp.where(coexistence, coexist_vapor, qv)
        ql = jnp.where(coexistence, coexist_liquid, ql)
        qi = jnp.where(coexistence, coexist_ice - coexist_liquid, qi)
        t = jnp.where(coexistence, freezing, t)
        rho = c / (self.gas_constant(qv, ql, qi) * t) if isobaric else c
        p = c if isobaric else self.pressure(rho, t, qv, ql, qi)
        actual = (
            self.enthalpy(t, qv, ql, qi) if isobaric else self.energy(rho, t, qv, ql, qi)
        )
        residual = actual - target
        threshold = jnp.maximum(
            self.energy_tolerance,
            16.0 * jnp.finfo(c.dtype).eps * jnp.maximum(jnp.abs(target), scale * t),
        )
        valid = (
            self._valid(c, t, qt)
            & bracketed
            & jnp.isfinite(target)
            & jnp.isfinite(residual)
            & jnp.isfinite(rho)
            & (rho > 0)
            & (p > 0)
            & (qv >= 0)
            & (ql >= 0)
            & (qi >= 0)
            & (jnp.abs(residual) <= threshold)
        )
        regular = self._regular(c, t, qt, qv, ql, qi, isobaric)
        return MoistAdjustmentResult(
            t, qv, ql, qi, rho, p, valid, residual, valid & regular
        )


__all__ = ["MoistAdjustmentResult", "MoistThermodynamicPlan"]
