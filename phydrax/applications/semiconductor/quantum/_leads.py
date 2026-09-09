# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Analytic retarded semi-infinite leads, not finite absorbing reservoirs."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from .._quantities import _text
from ._basis import _array, KB


def scalar_embedding(energy, surface_green, coupling_h, coupling_s=0.0):
    """Scalar Schur complement with the full nonorthogonal ES-H coupling.

    The reverse analytic block is z*S_cd-H_cd, NOT the complex conjugate of
    z*S_dc-H_dc. Conjugating z violates retarded analyticity. This primitive
    admits cross overlap; the high-level cell-basis backend exposes only S=I
    and zero cross overlap, so it cannot silently apply orthogonal charge
    reconstruction to a nonorthogonal partition.
    """
    forward = energy * coupling_s - coupling_h
    reverse = energy * jnp.conj(coupling_s) - jnp.conj(coupling_h)
    return forward * surface_green * reverse


class SemiInfiniteLead(StrictModule):
    """Orthogonal scalar principal layer repeated to infinity.

    onsite/hopping/coupling/chemical_potential are J; temperature is K.
    hopping and contact coupling must be nonzero. Numerical eta is supplied
    separately to surface_green and is never interpreted as scattering.
    """

    onsite: Array
    hopping: Array
    coupling: Array
    chemical_potential: Array
    temperature: Array
    energy_reference: str = eqx.field(static=True)

    def __init__(
        self,
        onsite,
        hopping,
        coupling,
        chemical_potential,
        temperature,
        *,
        energy_reference,
    ):
        values = [
            _array(v, n)
            for v, n in zip(
                (onsite, hopping, coupling, chemical_potential, temperature),
                (
                    "lead onsite",
                    "lead hopping",
                    "contact coupling",
                    "chemical potential",
                    "temperature",
                ),
                strict=True,
            )
        ]
        if (
            any(v.shape != () for v in values)
            or float(values[1]) == 0
            or float(values[2]) == 0
            or float(values[4]) <= 0
        ):
            raise ValueError(
                "Lead data must be scalars with nonzero hoppings and positive temperature."
            )
        (
            self.onsite,
            self.hopping,
            self.coupling,
            self.chemical_potential,
            self.temperature,
        ) = values
        self.energy_reference = _text(energy_reference, "electronic energy reference")

    def band(self):
        width = 2 * jnp.abs(self.hopping)
        return jnp.stack((self.onsite - width, self.onsite + width))

    def surface_green(self, energy, *, eta=0.0):
        z = jnp.asarray(energy) + 1j * jnp.asarray(eta)
        delta = z - self.onsite
        width = 2 * jnp.abs(self.hopping)
        root = jnp.sqrt(delta - width + 0j) * jnp.sqrt(delta + width + 0j)
        return 2.0 / (delta + root)

    def self_energy(self, energy, *, eta=0.0):
        z = jnp.asarray(energy) + 1j * jnp.asarray(eta)
        return scalar_embedding(z, self.surface_green(energy, eta=eta), self.coupling)

    def broadening(self, energy, *, eta=0.0):
        return -2 * jnp.imag(self.self_energy(energy, eta=eta))

    def occupation(self, energy):
        return jax.nn.sigmoid(
            (self.chemical_potential - energy) / (KB * self.temperature)
        )

    def shifted(self, energy):
        return eqx.tree_at(
            lambda lead: (lead.onsite, lead.chemical_potential),
            self,
            (self.onsite + energy, self.chemical_potential + energy),
        )


class BoundStateOccupation(StrictModule):
    """Explicit preparation of states not thermalized by coherent injection.

    A specified common preparation reservoir is a physical initial-state
    assumption even under bias. It is not inferred from left/right contacts.
    ``equilibrium`` checks contact compatibility before selecting this law.
    """

    chemical_potential: Array
    temperature: Array
    preparation: str = eqx.field(static=True)

    def __init__(self, chemical_potential, temperature, *, preparation):
        self.chemical_potential = _array(
            chemical_potential, "bound-state chemical potential"
        )
        self.temperature = _array(temperature, "bound-state temperature", positive=True)
        if self.chemical_potential.shape != () or self.temperature.shape != ():
            raise ValueError("Bound-state preparation data must be scalar.")
        self.preparation = _text(preparation, "bound-state preparation")

    @classmethod
    def equilibrium(cls, left, right):
        if not np.array_equal(
            np.asarray(left.chemical_potential), np.asarray(right.chemical_potential)
        ) or not np.array_equal(
            np.asarray(left.temperature), np.asarray(right.temperature)
        ):
            raise ValueError(
                "Equilibrium bound occupation requires common contact mu and temperature."
            )
        return cls(
            left.chemical_potential,
            left.temperature,
            preparation="common-reservoir equilibrium",
        )
