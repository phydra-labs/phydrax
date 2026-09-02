#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax.ein as ein

from ._entropy_pair import ConvexEntropyPair
from ._hyperbolic_systems import IdealMHDSystem, ShallowWaterSystem


def _pointwise_gradient(function, state):
    value = jnp.asarray(state)
    flat = value.reshape((-1, value.shape[-1]))
    gradient = jax.vmap(jax.grad(function))(flat)
    return gradient.reshape(value.shape)


def ideal_mhd_entropy_pair(system: IdealMHDSystem, /) -> ConvexEntropyPair:
    if not isinstance(system, IdealMHDSystem):
        raise TypeError("system must be IdealMHDSystem.")
    gamma = system.material.gamma

    def entropy(state):
        density = state[..., 0]
        pressure = system.pressure(state)
        physical_entropy = jnp.log(pressure) - gamma * jnp.log(density)
        return -density * physical_entropy / (gamma - 1.0)

    def variables(state):
        return _pointwise_gradient(lambda point: entropy(point), state)

    def entropy_flux(state, axis, args):
        del args
        velocity = state[..., 1 + int(axis)] / state[..., 0]
        return entropy(state) * velocity

    return ConvexEntropyPair(
        system,
        entropy,
        variables,
        entropy_flux,
        system.admissible,
        entropy_id="ideal-mhd-thermodynamic-entropy-divergence-free",
    )


def shallow_water_energy_pair(system: ShallowWaterSystem, /) -> ConvexEntropyPair:
    if not isinstance(system, ShallowWaterSystem):
        raise TypeError("system must be ShallowWaterSystem.")

    def entropy(state):
        depth = state[..., 0]
        discharge = state[..., 1:]
        kinetic = (
            0.5
            * ein.contract("...d,...d->...", discharge, discharge, backend="jax")
            / depth
        )
        return kinetic + 0.5 * system.gravity * depth**2

    def variables(state):
        return _pointwise_gradient(lambda point: entropy(point), state)

    def entropy_flux(state, axis, args):
        del args
        depth = state[..., 0]
        velocity = state[..., 1 + int(axis)] / depth
        return (entropy(state) + 0.5 * system.gravity * depth**2) * velocity

    return ConvexEntropyPair(
        system,
        entropy,
        variables,
        entropy_flux,
        system.admissible,
        entropy_id="shallow-water-total-energy",
    )


__all__ = ["ideal_mhd_entropy_pair", "shallow_water_energy_pair"]
