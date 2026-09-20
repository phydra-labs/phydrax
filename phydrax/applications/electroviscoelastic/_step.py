#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...rheology import ViscoelasticLaw


@dataclass(frozen=True, slots=True)
class ElectroViscoelasticState:
    conformation_minus: Array
    conformation_plus: Array
    surface_charge_c_m2: Array


def advance_constitutive_state(
    state: ElectroViscoelasticState,
    law_minus: ViscoelasticLaw | None,
    law_plus: ViscoelasticLaw | None,
    velocity_gradient_minus: ArrayLike,
    velocity_gradient_plus: ArrayLike,
    surface_charge_rate: ArrayLike,
    step_size_s: ArrayLike,
    /,
):
    dt = jnp.asarray(step_size_s)
    minus = (
        state.conformation_minus
        if law_minus is None
        else state.conformation_minus
        + dt
        * law_minus.upper_convected_rate(
            state.conformation_minus, velocity_gradient_minus
        )
    )
    plus = (
        state.conformation_plus
        if law_plus is None
        else state.conformation_plus
        + dt
        * law_plus.upper_convected_rate(state.conformation_plus, velocity_gradient_plus)
    )
    charge = state.surface_charge_c_m2 + dt * jnp.asarray(surface_charge_rate)
    return ElectroViscoelasticState(minus, plus, charge)


__all__ = ["ElectroViscoelasticState", "advance_constitutive_state"]
