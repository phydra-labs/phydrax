#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization.contact._kinematics import ContactKinematicsBatch
from ._closure import (
    AbstractInterfaceEvolutionLaw,
    ContactClosureCapability,
    ContactEvolutionResponse,
    NormalContactResponse,
    TangentialContactResponse,
)
from ._materials import ContactPairParameters
from ._route_state import ContactRouteMode, ContactRouteState


class FrictionWearEvolutionLaw(AbstractInterfaceEvolutionLaw):
    critical_slip_distance: float = eqx.field(static=True)
    damage_onset: float = eqx.field(static=True)
    damage_completion: float = eqx.field(static=True)
    minimum_film_thickness: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        critical_slip_distance: float,
        damage_onset: float,
        damage_completion: float,
        minimum_film_thickness: float = 0.0,
    ):
        critical = float(critical_slip_distance)
        onset = float(damage_onset)
        completion = float(damage_completion)
        minimum_film = float(minimum_film_thickness)
        if not np.isfinite(critical) or critical <= 0.0:
            raise ValueError("critical_slip_distance must be finite and positive.")
        if (
            not np.isfinite(onset)
            or not np.isfinite(completion)
            or onset < 0.0
            or completion <= onset
        ):
            raise ValueError("Cohesive damage opening interval is invalid.")
        if not np.isfinite(minimum_film) or minimum_film < 0.0:
            raise ValueError("minimum_film_thickness must be finite and nonnegative.")
        self.critical_slip_distance = critical
        self.damage_onset = onset
        self.damage_completion = completion
        self.minimum_film_thickness = minimum_film
        self._law_id = canonical_fingerprint(
            {
                "kind": "friction-wear-evolution-law",
                "critical_slip_distance": critical.hex(),
                "damage_onset": onset.hex(),
                "damage_completion": completion.hex(),
                "minimum_film_thickness": minimum_film.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.STATEFUL
            | ContactClosureCapability.WEAR
            | ContactClosureCapability.ADHESION
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        tangential: TangentialContactResponse,
        state: ContactRouteState,
        /,
    ) -> ContactEvolutionResponse:
        dtype = kinematics.gap.dtype
        slip_increment = kinematics.tangential_slip_increment
        slip_magnitude = jnp.sqrt(
            jnp.sum(slip_increment * slip_increment, axis=-1) + jnp.finfo(dtype).tiny
        )
        accumulated = state.accumulated_slip + jnp.where(
            kinematics.valid[..., None], slip_increment, 0.0
        )
        wear_increment = (
            parameters.wear_coefficient.astype(dtype)
            / parameters.hardness.astype(dtype)
            * jnp.maximum(normal.traction, 0.0)
            * slip_magnitude
        )
        wear = state.wear_depth + jnp.where(kinematics.valid, wear_increment, 0.0)
        damage_trial = (jnp.maximum(kinematics.gap, 0.0) - self.damage_onset) / (
            self.damage_completion - self.damage_onset
        )
        damage = jnp.maximum(
            state.adhesion_damage,
            jnp.clip(damage_trial, 0.0, 1.0),
        )
        critical = jnp.asarray(self.critical_slip_distance, dtype=dtype)
        rate_state = jnp.maximum(
            state.rate_state * jnp.exp(-slip_magnitude / critical),
            jnp.finfo(dtype).eps,
        )
        film = jnp.maximum(
            self.minimum_film_thickness,
            state.film_thickness - wear_increment,
        )
        mode = jnp.where(
            damage >= 1.0,
            int(ContactRouteMode.DEBONDED),
            jnp.where(
                tangential.slip,
                int(ContactRouteMode.SLIP),
                jnp.where(
                    tangential.stick,
                    int(ContactRouteMode.STICK),
                    jnp.where(
                        parameters.adhesion_energy > 0.0,
                        int(ContactRouteMode.ADHERED),
                        int(ContactRouteMode.OPEN),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        finite = (
            jnp.all(jnp.isfinite(accumulated))
            & jnp.all(jnp.isfinite(wear))
            & jnp.all(jnp.isfinite(damage))
            & jnp.all(jnp.isfinite(rate_state))
            & jnp.all(jnp.isfinite(film))
            & jnp.all(wear >= state.wear_depth)
            & jnp.all(damage >= state.adhesion_damage)
        )
        return ContactEvolutionResponse(
            mode,
            accumulated,
            damage,
            wear,
            rate_state,
            film,
            finite,
        )


__all__ = ["FrictionWearEvolutionLaw"]
