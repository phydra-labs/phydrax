#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _spherical_bessel(maximum_order: int, argument: Array, /) -> Array:
    safe = jnp.where(jnp.abs(argument) > 1.0e-8, argument, 1.0)
    j0 = jnp.where(
        jnp.abs(argument) > 1.0e-8, jnp.sin(argument) / safe, 1.0 - argument**2 / 6.0
    )
    if maximum_order == 0:
        return j0[None]
    j1 = jnp.where(
        jnp.abs(argument) > 1.0e-6,
        jnp.sin(argument) / safe**2 - jnp.cos(argument) / safe,
        argument / 3.0,
    )
    values = [j0, j1]
    for order in range(1, maximum_order):
        values.append((2 * order + 1) / safe * values[-1] - values[-2])
    return jnp.stack(values)


class NativeBoltzmannResult(StrictModule):
    conformal_times: Array
    wavenumbers: Array
    states: Array
    temperature_transfer: Array
    temperature_cl: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class EinsteinBoltzmannPlan(StrictModule, NonTrainableState):
    conformal_times: Array
    wavenumbers: Array
    maximum_multipole: int = eqx.field(static=True)
    hubble_conformal: Callable
    opacity_derivative: Callable
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        conformal_times,
        wavenumbers,
        hubble_conformal,
        opacity_derivative,
        /,
        *,
        maximum_multipole=16,
        plan_id="native-einstein-boltzmann",
    ):
        if not callable(hubble_conformal) or not callable(opacity_derivative):
            raise TypeError("Boltzmann background functions must be callable.")
        self.conformal_times = jnp.asarray(conformal_times)
        self.wavenumbers = jnp.asarray(wavenumbers)
        self.maximum_multipole = int(maximum_multipole)
        self.hubble_conformal = hubble_conformal
        self.opacity_derivative = opacity_derivative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-einstein-boltzmann",
                "times": int(self.conformal_times.size),
                "wavenumbers": int(self.wavenumbers.size),
                "ell_max": int(maximum_multipole),
            }
        )

    @property
    def state_dimension(self) -> int:
        return 5 + 2 * (self.maximum_multipole + 1)

    def _derivative(self, time: Array, state: Array, k: Array, args: Any, /) -> Array:
        del args
        ell_count = self.maximum_multipole + 1
        delta_c, theta_c, delta_b, theta_b, potential = state[:5]
        photons = state[5 : 5 + ell_count]
        neutrinos = state[5 + ell_count :]
        hubble = self.hubble_conformal(time)
        opacity = self.opacity_derivative(time)
        photon_rate = jnp.zeros_like(photons)
        neutrino_rate = jnp.zeros_like(neutrinos)
        photon_rate = photon_rate.at[0].set(-k * photons[1])
        photon_rate = photon_rate.at[1].set(
            k / 3.0 * (photons[0] + potential - 2.0 * photons[2])
            - opacity * (photons[1] - theta_b / (3.0 * k))
        )
        neutrino_rate = neutrino_rate.at[0].set(-k * neutrinos[1])
        neutrino_rate = neutrino_rate.at[1].set(
            k / 3.0 * (neutrinos[0] + potential - 2.0 * neutrinos[2])
        )
        for ell in range(2, self.maximum_multipole):
            photon_rate = photon_rate.at[ell].set(
                k
                / (2 * ell + 1)
                * (ell * photons[ell - 1] - (ell + 1) * photons[ell + 1])
                - opacity * photons[ell]
            )
            neutrino_rate = neutrino_rate.at[ell].set(
                k
                / (2 * ell + 1)
                * (ell * neutrinos[ell - 1] - (ell + 1) * neutrinos[ell + 1])
            )
        photon_rate = photon_rate.at[-1].set(
            k * photons[-2]
            - (self.maximum_multipole + 1) / jnp.maximum(time, 1.0e-30) * photons[-1]
            - opacity * photons[-1]
        )
        neutrino_rate = neutrino_rate.at[-1].set(
            k * neutrinos[-2]
            - (self.maximum_multipole + 1) / jnp.maximum(time, 1.0e-30) * neutrinos[-1]
        )
        matter = jnp.asarray(
            (
                -theta_c,
                -hubble * theta_c + k**2 * potential,
                -theta_b,
                -hubble * theta_b
                + k**2 * potential
                + 3.0 * opacity * k * (photons[1] - theta_b / (3.0 * k)),
                0.0,
            )
        )
        return jnp.concatenate((matter, photon_rate, neutrino_rate))

    def solve(
        self, initial_states: ArrayLike, primordial_power: ArrayLike, args: Any = None, /
    ) -> NativeBoltzmannResult:
        initial = jnp.asarray(initial_states)
        primordial = jnp.asarray(primordial_power)
        expected = (int(self.wavenumbers.size), self.state_dimension)
        if initial.shape != expected or primordial.shape != self.wavenumbers.shape:
            raise ValueError("Boltzmann initial state/power shapes are incompatible.")

        def solve_mode(k, state0):
            def step(state, interval):
                start, end = interval
                dt = end - start
                k1 = self._derivative(start, state, k, args)
                k2 = self._derivative(start + 0.5 * dt, state + 0.5 * dt * k1, k, args)
                k3 = self._derivative(start + 0.5 * dt, state + 0.5 * dt * k2, k, args)
                k4 = self._derivative(end, state + dt * k3, k, args)
                next_state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                return next_state, next_state

            intervals = jnp.stack(
                (self.conformal_times[:-1], self.conformal_times[1:]), axis=-1
            )
            _, tail = jax.lax.scan(step, state0, intervals)
            return jnp.concatenate((state0[None], tail), axis=0)

        states = jax.vmap(solve_mode)(self.wavenumbers, initial)
        ell_count = self.maximum_multipole + 1
        photon_monopole = states[:, :, 5]
        visibility = -self.opacity_derivative(self.conformal_times) * jnp.exp(
            -jnp.cumsum(
                -self.opacity_derivative(self.conformal_times)
                * jnp.diff(self.conformal_times, prepend=self.conformal_times[0])
            )
        )
        source = photon_monopole * visibility[None, :]
        distance = self.conformal_times[-1] - self.conformal_times

        def transfer_mode(k, source_mode):
            bessel = _spherical_bessel(self.maximum_multipole, k * distance)
            return jnp.trapezoid(
                bessel * source_mode[None, :], self.conformal_times, axis=-1
            )

        transfer = jax.vmap(transfer_mode)(self.wavenumbers, source)
        cl = (
            4.0
            * jnp.pi
            * jnp.trapezoid(
                primordial[:, None] * transfer**2 / self.wavenumbers[:, None],
                self.wavenumbers,
                axis=0,
            )
        )
        valid = jnp.all(jnp.isfinite(states)) & jnp.all(jnp.isfinite(cl))
        return NativeBoltzmannResult(
            self.conformal_times,
            self.wavenumbers,
            states,
            transfer,
            cl,
            valid,
            self.plan_id,
        )


__all__ = ["EinsteinBoltzmannPlan", "NativeBoltzmannResult"]
