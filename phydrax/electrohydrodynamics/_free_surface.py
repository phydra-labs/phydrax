#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def total_interface_traction(
    hydrodynamic_stress_minus: ArrayLike,
    hydrodynamic_stress_plus: ArrayLike,
    electric_stress_minus: ArrayLike,
    electric_stress_plus: ArrayLike,
    normal: ArrayLike,
    surface_tension: float,
    curvature: ArrayLike,
    surface_tension_gradient: ArrayLike,
    /,
):
    n = jnp.asarray(normal)
    jump = (
        jnp.asarray(hydrodynamic_stress_plus)
        + jnp.asarray(electric_stress_plus)
        - jnp.asarray(hydrodynamic_stress_minus)
        - jnp.asarray(electric_stress_minus)
    ) @ n
    return (
        jump
        - float(surface_tension) * jnp.asarray(curvature)[..., None] * n
        - jnp.asarray(surface_tension_gradient)
    )


__all__ = ["total_interface_traction"]
