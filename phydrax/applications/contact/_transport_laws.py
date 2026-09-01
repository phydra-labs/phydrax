#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.contact._kinematics import ContactKinematicsBatch
from ._closure import (
    AbstractContactTransportLaw,
    ContactClosureCapability,
    ContactTransportResponse,
    NormalContactResponse,
    TangentialContactResponse,
)
from ._materials import ContactPairParameters
from ._route_state import ContactRouteState


class CoupledContactTransportLaw(AbstractContactTransportLaw):
    """Pressure/gap-dependent thermal, electrical, and hydraulic transfer.

    The driving jump columns are temperature, electric potential, and chemical
    or hydraulic potential, oriented from the plus participant to the minus
    participant.
    """

    pressure_exponent: float = eqx.field(static=True)
    gap_decay: float = eqx.field(static=True)
    friction_heat_fraction: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pressure_exponent: float = 1.0,
        gap_decay: float,
        friction_heat_fraction: float = 1.0,
    ):
        exponent = float(pressure_exponent)
        decay = float(gap_decay)
        heat_fraction = float(friction_heat_fraction)
        if not np.isfinite(exponent) or exponent < 0.0:
            raise ValueError("pressure_exponent must be finite and nonnegative.")
        if not np.isfinite(decay) or decay <= 0.0:
            raise ValueError("gap_decay must be finite and positive.")
        if not np.isfinite(heat_fraction) or not 0.0 <= heat_fraction <= 1.0:
            raise ValueError("friction_heat_fraction must lie in [0, 1].")
        self.pressure_exponent = exponent
        self.gap_decay = decay
        self.friction_heat_fraction = heat_fraction
        self._law_id = canonical_fingerprint(
            {
                "kind": "coupled-contact-transport-law",
                "pressure_exponent": exponent.hex(),
                "gap_decay": decay.hex(),
                "friction_heat_fraction": heat_fraction.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.TRANSPORT | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        tangential: TangentialContactResponse,
        state: ContactRouteState,
        driving_jump,
        /,
    ) -> ContactTransportResponse:
        del state
        jump = jnp.asarray(driving_jump, dtype=kinematics.gap.dtype)
        if jump.shape != (kinematics.capacity, 3):
            raise ValueError(
                "Coupled contact transport expects temperature, electric, and mass jumps."
            )
        pressure = jnp.maximum(normal.traction, 0.0)
        pressure_factor = jnp.where(
            pressure > 0.0,
            pressure**self.pressure_exponent,
            jnp.where(self.pressure_exponent == 0.0, 1.0, 0.0),
        )
        positive_gap = jnp.maximum(kinematics.gap, 0.0)
        gap_factor = jnp.exp(-positive_gap / self.gap_decay)
        contact_factor = pressure_factor * gap_factor
        active = kinematics.valid & parameters.transport_available
        heat = parameters.thermal_conductance * contact_factor * jump[:, 0]
        electrical = parameters.electrical_conductance * contact_factor * jump[:, 1]
        mass = contact_factor * jump[:, 2]
        frictional_heat = self.friction_heat_fraction * tangential.dissipated_power
        heat = jnp.where(active, heat, 0.0)
        electrical = jnp.where(active, electrical, 0.0)
        mass = jnp.where(active, mass, 0.0)
        frictional_heat = jnp.where(active, frictional_heat, 0.0)
        finite = (
            jnp.all(jnp.isfinite(heat))
            & jnp.all(jnp.isfinite(electrical))
            & jnp.all(jnp.isfinite(mass))
            & jnp.all(jnp.isfinite(frictional_heat))
            & jnp.all(frictional_heat >= 0.0)
        )
        return ContactTransportResponse(
            heat,
            electrical,
            mass,
            frictional_heat,
            finite,
        )


class ContactFluxAssembly(StrictModule):
    plus_heat: Array
    minus_heat: Array
    plus_electrical: Array
    minus_electrical: Array
    plus_mass: Array
    minus_mass: Array
    generated_heat: Array
    conservation_defect: Array
    finite: Array
    successful: Array


def assemble_contact_fluxes(
    response: ContactTransportResponse,
    quadrature_weight,
    /,
    *,
    heat_partition: float = 0.5,
) -> ContactFluxAssembly:
    partition = float(heat_partition)
    if not 0.0 <= partition <= 1.0:
        raise ValueError("heat_partition must lie in [0, 1].")
    weight = jnp.asarray(quadrature_weight, dtype=response.heat_flux.dtype)
    if weight.shape != response.heat_flux.shape:
        raise ValueError("Contact flux quadrature weights have invalid shape.")
    conductive_heat = weight * response.heat_flux
    electrical = weight * response.electrical_current
    mass = weight * response.mass_flux
    generated = weight * response.frictional_heat
    plus_heat = -conductive_heat + partition * generated
    minus_heat = conductive_heat + (1.0 - partition) * generated
    plus_electrical = -electrical
    minus_electrical = electrical
    plus_mass = -mass
    minus_mass = mass
    conservation = jnp.stack(
        (
            jnp.sum(plus_electrical + minus_electrical),
            jnp.sum(plus_mass + minus_mass),
            jnp.sum(plus_heat + minus_heat - generated),
        )
    )
    finite = jnp.all(
        jnp.isfinite(
            jnp.concatenate(
                (
                    plus_heat,
                    minus_heat,
                    plus_electrical,
                    minus_electrical,
                    plus_mass,
                    minus_mass,
                    generated,
                    conservation,
                )
            )
        )
    )
    tolerance = (
        512.0
        * jnp.finfo(weight.dtype).eps
        * jnp.maximum(1.0, jnp.sum(jnp.abs(generated)))
    )
    return ContactFluxAssembly(
        plus_heat,
        minus_heat,
        plus_electrical,
        minus_electrical,
        plus_mass,
        minus_mass,
        generated,
        conservation,
        finite,
        finite & jnp.all(jnp.abs(conservation) <= tolerance),
    )


__all__ = [
    "ContactFluxAssembly",
    "CoupledContactTransportLaw",
    "assemble_contact_fluxes",
]
