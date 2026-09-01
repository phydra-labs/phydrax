#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LubricationContactPlan(StrictModule, NonTrainableState):
    viscosity: float = eqx.field(static=True)
    minimum_film_thickness: float = eqx.field(static=True)
    cavitation_pressure: float = eqx.field(static=True)
    asperity_transition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        viscosity: float,
        minimum_film_thickness: float,
        cavitation_pressure: float = 0.0,
        asperity_transition: float,
    ):
        viscosity_ = float(viscosity)
        minimum = float(minimum_film_thickness)
        cavitation = float(cavitation_pressure)
        transition = float(asperity_transition)
        if (
            not np.isfinite(viscosity_)
            or viscosity_ <= 0.0
            or not np.isfinite(minimum)
            or minimum <= 0.0
            or not np.isfinite(cavitation)
            or not np.isfinite(transition)
            or transition <= minimum
        ):
            raise ValueError("Lubrication contact parameters are invalid.")
        self.viscosity = viscosity_
        self.minimum_film_thickness = minimum
        self.cavitation_pressure = cavitation
        self.asperity_transition = transition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lubrication-contact-plan",
                "viscosity": viscosity_.hex(),
                "minimum_film_thickness": minimum.hex(),
                "cavitation_pressure": cavitation.hex(),
                "asperity_transition": transition.hex(),
            }
        )


class LubricationContactResponse(StrictModule):
    film_thickness: Array
    fluid_pressure: Array
    asperity_fraction: Array
    normal_traction: Array
    tangential_traction: Array
    dissipated_power: Array
    cavitated: Array
    finite: Array
    dissipative: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def evaluate_lubrication_contact(
    plan: LubricationContactPlan,
    gap: ArrayLike,
    normal_velocity: ArrayLike,
    tangential_velocity: ArrayLike,
    effective_radius: ArrayLike,
    /,
    *,
    asperity_pressure: ArrayLike = 0.0,
) -> LubricationContactResponse:
    if not isinstance(plan, LubricationContactPlan):
        raise TypeError("plan must be LubricationContactPlan.")
    gap_ = jnp.asarray(gap)
    normal_velocity_ = jnp.asarray(normal_velocity, dtype=gap_.dtype)
    tangential = jnp.asarray(tangential_velocity, dtype=gap_.dtype)
    radius = jnp.asarray(effective_radius, dtype=gap_.dtype)
    asperity = jnp.asarray(asperity_pressure, dtype=gap_.dtype)
    if normal_velocity_.shape != gap_.shape or tangential.shape[:-1] != gap_.shape:
        raise ValueError("Lubrication contact kinematic shapes are invalid.")
    film = jnp.maximum(gap_, plan.minimum_film_thickness)
    closing = jnp.maximum(-normal_velocity_, 0.0)
    squeeze_pressure = 3.0 * plan.viscosity * radius * radius * closing / (2.0 * film**3)
    fluid_pressure = jnp.maximum(plan.cavitation_pressure, squeeze_pressure)
    cavitated = squeeze_pressure < plan.cavitation_pressure
    asperity_fraction = jnp.clip(
        (plan.asperity_transition - film)
        / (plan.asperity_transition - plan.minimum_film_thickness),
        0.0,
        1.0,
    )
    normal_traction = (
        1.0 - asperity_fraction
    ) * fluid_pressure + asperity_fraction * jnp.maximum(asperity, 0.0)
    fluid_shear = -(plan.viscosity / film)[..., None] * tangential
    tangential_traction = (1.0 - asperity_fraction)[..., None] * fluid_shear
    dissipated = (
        -jnp.sum(tangential_traction * tangential, axis=-1) + fluid_pressure * closing
    )
    finite = (
        jnp.all(jnp.isfinite(film))
        & jnp.all(jnp.isfinite(fluid_pressure))
        & jnp.all(jnp.isfinite(normal_traction))
        & jnp.all(jnp.isfinite(tangential_traction))
        & jnp.all(jnp.isfinite(dissipated))
    )
    dissipative = jnp.all(dissipated >= -64.0 * jnp.finfo(gap_.dtype).eps)
    return LubricationContactResponse(
        film,
        fluid_pressure,
        asperity_fraction,
        normal_traction,
        tangential_traction,
        jnp.maximum(dissipated, 0.0),
        cavitated,
        finite,
        dissipative,
        finite & dissipative,
        plan.plan_id,
    )


__all__ = [
    "LubricationContactPlan",
    "LubricationContactResponse",
    "evaluate_lubrication_contact",
]
