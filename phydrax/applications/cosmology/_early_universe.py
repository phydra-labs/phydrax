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
from ._products import (
    ThermodynamicsHistory,
)


class RelicBackgroundResult(StrictModule):
    scale_factor: Array
    photon_density: Array
    neutrino_density: Array
    total_radiation_density: Array
    plan_id: str = eqx.field(static=True)


class RelicBackgroundPlan(StrictModule, NonTrainableState):
    photon_density_today: Array
    effective_neutrinos: Array
    neutrino_temperature_ratio: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        photon_density_today,
        effective_neutrinos=3.046,
        neutrino_temperature_ratio=(4.0 / 11.0) ** (1.0 / 3.0),
        /,
    ):
        self.photon_density_today = jnp.asarray(photon_density_today).reshape(())
        self.effective_neutrinos = jnp.asarray(effective_neutrinos).reshape(())
        self.neutrino_temperature_ratio = jnp.asarray(neutrino_temperature_ratio).reshape(
            ()
        )
        self.plan_id = canonical_fingerprint(
            {"kind": "relic-background", "neff": float(self.effective_neutrinos)}
        )

    def evaluate(self, scale_factor: ArrayLike, /) -> RelicBackgroundResult:
        a = jnp.asarray(scale_factor)
        photon = self.photon_density_today / a**4
        neutrino = (
            photon
            * self.effective_neutrinos
            * (7.0 / 8.0)
            * self.neutrino_temperature_ratio**4
        )
        return RelicBackgroundResult(a, photon, neutrino, photon + neutrino, self.plan_id)


class BbnResult(StrictModule):
    times: Array
    abundances: Array
    baryon_conservation_error: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class BbnReactionNetworkPlan(StrictModule, NonTrainableState):
    stoichiometry: Array
    baryon_numbers: Array
    rate_model: Callable
    times: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stoichiometry,
        baryon_numbers,
        rate_model,
        times,
        /,
        *,
        network_id="bbn-network",
    ):
        self.stoichiometry = jnp.asarray(stoichiometry)
        self.baryon_numbers = jnp.asarray(baryon_numbers)
        self.rate_model = rate_model
        self.times = jnp.asarray(times)
        if (
            self.stoichiometry.ndim != 2
            or self.stoichiometry.shape[0] != self.baryon_numbers.size
            or not callable(rate_model)
        ):
            raise ValueError("BBN reaction network arrays are inconsistent.")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bbn-reaction-network",
                "network_id": str(network_id),
                "species": int(self.baryon_numbers.size),
                "reactions": int(self.stoichiometry.shape[1]),
            }
        )

    def solve(self, initial_abundances: ArrayLike, args: Any = None, /) -> BbnResult:
        initial = jnp.asarray(initial_abundances)
        baryons0 = jnp.sum(self.baryon_numbers * initial)

        def derivative(time, abundance):
            rates = jnp.asarray(self.rate_model(time, abundance, args))
            return self.stoichiometry @ rates

        def step(abundance, interval):
            start, end = interval
            dt = end - start
            k1 = derivative(start, abundance)
            k2 = derivative(start + 0.5 * dt, abundance + 0.5 * dt * k1)
            k3 = derivative(start + 0.5 * dt, abundance + 0.5 * dt * k2)
            k4 = derivative(end, abundance + dt * k3)
            candidate = abundance + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            candidate = jnp.maximum(candidate, 0.0)
            valid = jnp.all(jnp.isfinite(candidate))
            accepted = jnp.where(valid, candidate, abundance)
            return accepted, (accepted, valid)

        intervals = jnp.stack((self.times[:-1], self.times[1:]), axis=-1)
        _, outputs = jax.lax.scan(step, initial, intervals)
        abundances = jnp.concatenate((initial[None], outputs[0]), axis=0)
        conservation = (
            jnp.sum(abundances * self.baryon_numbers[None, :], axis=1) - baryons0
        )
        valid = jnp.concatenate((jnp.asarray(True)[None], outputs[1]))
        return BbnResult(self.times, abundances, conservation, valid, self.plan_id)


class RecombinationPlan(StrictModule, NonTrainableState):
    scale_factors: Array
    cmb_temperature_today: Array
    residual_ionization: Array
    recombination_scale_factor: Array
    width: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors,
        /,
        *,
        cmb_temperature_today=2.7255,
        residual_ionization=2.0e-4,
        recombination_redshift=1089.0,
        width=0.08,
    ):
        self.scale_factors = jnp.asarray(scale_factors)
        self.cmb_temperature_today = jnp.asarray(cmb_temperature_today).reshape(())
        self.residual_ionization = jnp.asarray(residual_ionization).reshape(())
        self.recombination_scale_factor = jnp.asarray(
            1.0 / (1.0 + recombination_redshift)
        )
        self.width = jnp.asarray(width).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-recombination",
                "nodes": int(self.scale_factors.size),
                "redshift": float(recombination_redshift),
            }
        )

    def build(self, scale, provenance, realization, /) -> ThermodynamicsHistory:
        a = self.scale_factors
        log_ratio = jnp.log(a / self.recombination_scale_factor)
        ionization = self.residual_ionization + (1.0 - self.residual_ionization) * 0.5 * (
            1.0 - jnp.tanh(log_ratio / self.width)
        )
        temperature = self.cmb_temperature_today / a
        opacity = ionization / a**2
        da = jnp.diff(a, prepend=a[0])
        optical_depth = jnp.flip(jnp.cumsum(jnp.flip(opacity * da)))
        visibility = opacity * jnp.exp(-optical_depth)
        normalization = jnp.trapezoid(visibility, a)
        visibility = visibility / jnp.where(normalization > 0.0, normalization, 1.0)
        return ThermodynamicsHistory(
            a,
            ionization,
            temperature,
            opacity,
            visibility,
            scale,
            provenance,
            realization,
        )


__all__ = [
    "BbnReactionNetworkPlan",
    "BbnResult",
    "RecombinationPlan",
    "RelicBackgroundPlan",
    "RelicBackgroundResult",
]
