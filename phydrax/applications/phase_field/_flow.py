#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


CapillaryForceRepresentation = Literal["mu-grad-phi", "minus-phi-grad-mu"]


class PhaseFluidMaterial(StrictModule, NonTrainableState):
    phase_densities: Array
    phase_viscosities: Array
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        phase_densities: ArrayLike,
        phase_viscosities: ArrayLike,
        /,
        *,
        material_id: str,
    ):
        densities = np.asarray(phase_densities)
        viscosities = np.asarray(phase_viscosities)
        identifier = str(material_id)
        if (
            densities.ndim != 1
            or densities.size < 2
            or viscosities.shape != densities.shape
            or np.any(~np.isfinite(densities))
            or np.any(~np.isfinite(viscosities))
            or np.any(densities <= 0.0)
            or np.any(viscosities <= 0.0)
            or not identifier
        ):
            raise ValueError("Phase-fluid material data are invalid.")
        self.phase_densities = jnp.asarray(densities)
        self.phase_viscosities = jnp.asarray(viscosities)
        self.material_id = canonical_fingerprint(
            {
                "kind": "phase-fluid-material",
                "declared_id": identifier,
                "phase_densities": densities.tolist(),
                "phase_viscosities": viscosities.tolist(),
            }
        )

    def mixture(self, phase_weights: ArrayLike, /) -> tuple[Array, Array]:
        weights = jnp.asarray(phase_weights)
        if weights.shape[-1] != self.phase_densities.size:
            raise ValueError("Phase-fluid weights have incompatible phase count.")
        density = ein.contract(
            "...p,p->...", weights, self.phase_densities.astype(weights.dtype)
        )
        viscosity = ein.contract(
            "...p,p->...", weights, self.phase_viscosities.astype(weights.dtype)
        )
        return density, viscosity


class ModelHCouplingEvaluation(StrictModule):
    density: Array
    viscosity: Array
    kinetic_energy: Array
    viscous_dissipation: Array
    phase_advection_rate: Array
    capillary_force: Array
    flow_capillary_power: Array
    phase_advection_power: Array
    exchange_defect: Array
    incompressibility_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ModelHCouplingPlan(StrictModule, NonTrainableState):
    material: PhaseFluidMaterial
    force_representation: CapillaryForceRepresentation = eqx.field(static=True)
    incompressibility_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: PhaseFluidMaterial,
        /,
        *,
        force_representation: CapillaryForceRepresentation = "mu-grad-phi",
        incompressibility_tolerance: float = 1.0e-10,
    ):
        if not isinstance(material, PhaseFluidMaterial):
            raise TypeError("material must be PhaseFluidMaterial.")
        if force_representation not in ("mu-grad-phi", "minus-phi-grad-mu"):
            raise ValueError("Unknown capillary force representation.")
        tolerance = float(incompressibility_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Incompressibility tolerance must be nonnegative.")
        self.material = material
        self.force_representation = force_representation
        self.incompressibility_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "model-h-coupling-plan",
                "material": material.material_id,
                "force_representation": force_representation,
                "incompressibility_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        phase_weights: ArrayLike,
        phase_value: ArrayLike,
        phase_gradient: ArrayLike,
        chemical_potential: ArrayLike,
        chemical_gradient: ArrayLike,
        velocity: ArrayLike,
        velocity_gradient: ArrayLike,
        /,
    ) -> ModelHCouplingEvaluation:
        weights = jnp.asarray(phase_weights)
        phase = jnp.asarray(phase_value, dtype=weights.dtype)
        phase_grad = jnp.asarray(phase_gradient, dtype=weights.dtype)
        chemical = jnp.asarray(chemical_potential, dtype=weights.dtype)
        chemical_grad = jnp.asarray(chemical_gradient, dtype=weights.dtype)
        flow = jnp.asarray(velocity, dtype=weights.dtype)
        flow_grad = jnp.asarray(velocity_gradient, dtype=weights.dtype)
        if (
            phase.shape != weights.shape[:-1]
            or chemical.shape != phase.shape
            or phase_grad.shape[:-1] != phase.shape
            or chemical_grad.shape != phase_grad.shape
            or flow.shape != phase_grad.shape
            or flow_grad.shape != phase.shape + (flow.shape[-1], flow.shape[-1])
        ):
            raise ValueError("Model-H coupling fields have incompatible shapes.")
        density, viscosity = self.material.mixture(weights)
        strain_rate = 0.5 * (flow_grad + jnp.swapaxes(flow_grad, -1, -2))
        kinetic = 0.5 * density * ein.contract("...d,...d->...", flow, flow)
        viscous = (
            2.0 * viscosity * ein.contract("...ij,...ij->...", strain_rate, strain_rate)
        )
        advection = ein.contract("...d,...d->...", flow, phase_grad)
        if self.force_representation == "mu-grad-phi":
            force = chemical[..., None] * phase_grad
            flow_power = ein.contract("...d,...d->...", flow, force)
            phase_power = -chemical * advection
            exchange = flow_power + phase_power
        else:
            force = -phase[..., None] * chemical_grad
            flow_power = ein.contract("...d,...d->...", flow, force)
            phase_power = -chemical * advection
            pressure_shift_rate = ein.contract(
                "...d,...d->...", flow, chemical * phase_grad + phase * chemical_grad
            )
            exchange = flow_power + phase_power + pressure_shift_rate
        divergence = jnp.trace(flow_grad, axis1=-2, axis2=-1)
        incompressibility = jnp.max(jnp.abs(divergence))
        scale = jnp.maximum(
            jnp.max(jnp.abs(flow_power)) + jnp.max(jnp.abs(phase_power)), 1.0
        )
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(viscous))
            & jnp.all(jnp.isfinite(exchange))
        )
        successful = (
            finite
            & jnp.all(density > 0.0)
            & jnp.all(viscosity > 0.0)
            & (jnp.max(jnp.abs(exchange)) <= 128.0 * jnp.finfo(flow.dtype).eps * scale)
            & (incompressibility <= self.incompressibility_tolerance)
        )
        return ModelHCouplingEvaluation(
            density,
            viscosity,
            kinetic,
            viscous,
            advection,
            force,
            flow_power,
            phase_power,
            exchange,
            incompressibility,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "CapillaryForceRepresentation",
    "ModelHCouplingEvaluation",
    "ModelHCouplingPlan",
    "PhaseFluidMaterial",
]
