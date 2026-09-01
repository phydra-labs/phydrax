#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class HomogenizedRoughContactPlan(StrictModule, NonTrainableState):
    pressure_scale: float = eqx.field(static=True)
    separation_scale: float = eqx.field(static=True)
    rms_roughness: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pressure_scale: float,
        separation_scale: float,
        rms_roughness: float,
    ):
        values = tuple(
            float(value)
            for value in (
                pressure_scale,
                separation_scale,
                rms_roughness,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Rough-contact scales must be finite and positive.")
        self.pressure_scale, self.separation_scale, self.rms_roughness = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "homogenized-rough-contact-plan",
                "parameters": tuple(value.hex() for value in values),
            }
        )


class HomogenizedRoughContactResponse(StrictModule):
    pressure: Array
    tangent: Array
    area_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_homogenized_rough_contact(
    plan: HomogenizedRoughContactPlan,
    mean_separation: ArrayLike,
    /,
) -> HomogenizedRoughContactResponse:
    if not isinstance(plan, HomogenizedRoughContactPlan):
        raise TypeError("plan must be HomogenizedRoughContactPlan.")
    separation = jnp.asarray(mean_separation)
    pressure = plan.pressure_scale * jnp.exp(-separation / plan.separation_scale)
    tangent = -pressure / plan.separation_scale
    normalized = separation / (jnp.sqrt(2.0) * plan.rms_roughness)
    area = 0.5 * jax.scipy.special.erfc(normalized)
    finite = (
        jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(tangent))
        & jnp.all(jnp.isfinite(area))
    )
    return HomogenizedRoughContactResponse(
        pressure,
        tangent,
        jnp.clip(area, 0.0, 1.0),
        finite,
        finite & jnp.all(pressure >= 0.0),
        plan.plan_id,
    )


class PeriodicRoughContactPlan(StrictModule, NonTrainableState):
    compliance_spectrum: Array
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        compliance_spectrum: ArrayLike,
        /,
        *,
        maximum_iterations: int = 1000,
        tolerance: float = 1.0e-10,
        relaxation: float = 0.9,
    ):
        spectrum = np.asarray(compliance_spectrum, dtype=float)
        if spectrum.ndim != 2 or np.any(~np.isfinite(spectrum)) or np.any(spectrum < 0.0):
            raise ValueError(
                "Rough-contact compliance spectrum must be a nonnegative matrix."
            )
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        relaxation_ = float(relaxation)
        if iterations <= 0 or tolerance_ <= 0.0 or not 0.0 < relaxation_ <= 1.0:
            raise ValueError("Rough-contact solver controls are invalid.")
        self.compliance_spectrum = jnp.asarray(spectrum)
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.relaxation = relaxation_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-rough-contact-plan",
                "spectrum": array_tree_fingerprint(spectrum),
                "maximum_iterations": iterations,
                "tolerance": tolerance_.hex(),
                "relaxation": relaxation_.hex(),
            }
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.compliance_spectrum.shape)

    def displacement(self, pressure: ArrayLike, /) -> Array:
        pressure_ = jnp.asarray(pressure, dtype=self.compliance_spectrum.dtype)
        if pressure_.shape != self.shape:
            raise ValueError("Periodic rough pressure has invalid shape.")
        transformed = jnp.fft.fft2(pressure_)
        return jnp.real(jnp.fft.ifft2(self.compliance_spectrum * transformed))


class PeriodicRoughContactEvidence(StrictModule):
    converged: Array
    iterations: Array
    projected_residual: Array
    complementarity_defect: Array
    minimum_gap: Array
    total_load: Array
    contact_area_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PeriodicRoughContactResult(StrictModule):
    pressure: Array
    displacement: Array
    gap: Array
    evidence: PeriodicRoughContactEvidence


def solve_periodic_rough_contact(
    plan: PeriodicRoughContactPlan,
    undeformed_gap: ArrayLike,
    /,
    *,
    initial_pressure: ArrayLike | None = None,
) -> PeriodicRoughContactResult:
    if not isinstance(plan, PeriodicRoughContactPlan):
        raise TypeError("plan must be PeriodicRoughContactPlan.")
    gap0 = jnp.asarray(undeformed_gap, dtype=plan.compliance_spectrum.dtype)
    if gap0.shape != plan.shape:
        raise ValueError("Periodic rough gap has invalid shape.")
    pressure = (
        jnp.zeros(plan.shape, dtype=gap0.dtype)
        if initial_pressure is None
        else jnp.asarray(initial_pressure, dtype=gap0.dtype)
    )
    if pressure.shape != plan.shape:
        raise ValueError("Periodic rough initial pressure has invalid shape.")
    pressure = jnp.maximum(pressure, 0.0)
    maximum_compliance = jnp.max(plan.compliance_spectrum, initial=0.0)
    step = plan.relaxation / jnp.maximum(
        maximum_compliance,
        jnp.finfo(gap0.dtype).eps,
    )
    scale = jnp.maximum(1.0, jnp.sqrt(jnp.sum(gap0 * gap0)))
    tolerance = plan.tolerance * scale

    def body(index, state):
        value, converged, first_converged, residual_norm = state
        displacement = plan.displacement(value)
        gap = gap0 + displacement
        projected = jnp.maximum(0.0, value - step * gap)
        residual = value - projected
        norm = jnp.sqrt(jnp.sum(residual * residual))
        now = norm <= tolerance
        first = jnp.where((~converged) & now, index + 1, first_converged)
        return (
            jnp.where(converged, value, projected),
            converged | now,
            first,
            norm,
        )

    pressure, converged, iterations, residual = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        body,
        (
            pressure,
            jnp.asarray(False),
            jnp.asarray(plan.maximum_iterations, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=gap0.dtype),
        ),
    )
    displacement = plan.displacement(pressure)
    gap = gap0 + displacement
    complementarity = jnp.max(jnp.abs(pressure * jnp.maximum(gap, 0.0)), initial=0.0)
    finite = (
        jnp.all(jnp.isfinite(pressure))
        & jnp.all(jnp.isfinite(displacement))
        & jnp.all(jnp.isfinite(gap))
    )
    evidence = PeriodicRoughContactEvidence(
        converged,
        iterations,
        residual,
        complementarity,
        jnp.min(gap, initial=jnp.inf),
        jnp.sum(pressure),
        jnp.mean(pressure > plan.tolerance),
        finite,
        converged & finite & jnp.all(pressure >= 0.0),
        plan.plan_id,
    )
    return PeriodicRoughContactResult(pressure, displacement, gap, evidence)


class HertzContactReference(StrictModule):
    contact_radius: Array
    maximum_pressure: Array
    indentation: Array
    force: Array


def hertz_sphere_half_space(
    radius: ArrayLike,
    effective_modulus: ArrayLike,
    force: ArrayLike,
    /,
) -> HertzContactReference:
    radius_ = jnp.asarray(radius)
    modulus = jnp.asarray(effective_modulus, dtype=radius_.dtype)
    force_ = jnp.asarray(force, dtype=radius_.dtype)
    contact_radius = (3.0 * force_ * radius_ / (4.0 * modulus)) ** (1.0 / 3.0)
    maximum_pressure = 3.0 * force_ / (2.0 * jnp.pi * contact_radius * contact_radius)
    indentation = contact_radius * contact_radius / radius_
    return HertzContactReference(contact_radius, maximum_pressure, indentation, force_)


__all__ = [
    "HertzContactReference",
    "HomogenizedRoughContactPlan",
    "HomogenizedRoughContactResponse",
    "PeriodicRoughContactEvidence",
    "PeriodicRoughContactPlan",
    "PeriodicRoughContactResult",
    "evaluate_homogenized_rough_contact",
    "hertz_sphere_half_space",
    "solve_periodic_rough_contact",
]
