#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._structured_cochain import StructuredCochainBridge


def stable_bernoulli(value: ArrayLike, /) -> Array:
    """Return ``B(x) = x/expm1(x)`` with finite inactive AD branches."""

    x = jnp.asarray(value)
    x = x.astype(jnp.result_type(x, 1.0))
    magnitude = jnp.abs(x)
    small = magnitude < 1.0e-4
    # Mask arguments before evaluating: even a dormant 0/0 or overflowing
    # polynomial poisons reverse AD. Exponentials only see negative arguments.
    series_x = jnp.where(small, x, 0.0)
    series = 1.0 - series_x / 2.0 + series_x**2 / 12.0 - series_x**4 / 720.0
    regular_magnitude = jnp.where(small, 1.0, magnitude)
    positive = (
        regular_magnitude
        * jnp.exp(-regular_magnitude)
        / -jnp.expm1(-regular_magnitude)
    )
    # B(-x) = B(x) + x; this also avoids overflow for large negative x.
    regular = jnp.where(x < 0.0, positive + regular_magnitude, positive)
    return jnp.where(small, series, regular)


def scharfetter_gummel_flux(
    tail_density: ArrayLike,
    head_density: ArrayLike,
    drift_difference: ArrayLike,
    diffusivity: ArrayLike = 1.0,
) -> Array:
    """Return oriented tail-to-head ``D * (B(delta)*tail - B(-delta)*head)``.

    ``delta`` is the head-minus-tail dimensionless *drift* potential,
    excluding the ideal chemical term ``log(density)``. Positive delta drives
    particles toward the tail; zero flux requires ``head/tail = exp(-delta)``.
    The result is not yet multiplied by edge transmissibility. All arguments
    broadcast, so the same kernel serves cochain and finite-volume transport.
    """

    difference = jnp.asarray(drift_difference)
    return jnp.asarray(diffusivity) * (
        stable_bernoulli(difference) * jnp.asarray(tail_density)
        - stable_bernoulli(-difference) * jnp.asarray(head_density)
    )


class CochainElectrochemicalFluxEvaluation(StrictModule):
    edge_flux: Array
    concentration_rate: Array
    species_mass_defect: Array
    free_energy_dissipation: Array
    explicit_step_restriction: Array
    finite: Array
    conservative: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PreparedCochainElectrochemicalFlux(StrictModule, NonTrainableState):
    """Closed-edge SG transport with the positive Hodge-adjoint codifferential.

    ``edge_flux`` is oriented tail-to-head before the degree-one Hodge factor.
    Consequently ``concentration_rate = delta(edge_flux)`` removes material
    from the tail and adds it to the head. Dissipation is dimensionless
    free-energy decay; multiply by the species thermal energy for physical units.
    """

    bridge: StructuredCochainBridge
    diffusivities: Array
    tail_indices: Array
    head_indices: Array
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        diffusivities: ArrayLike,
        /,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        values = np.asarray(diffusivities, dtype=float)
        if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                "diffusivities must be one-dimensional, finite, and positive."
            )
        incidence = bridge.cochain.topology.incidences[0]
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        source = np.asarray(incidence.relation.source_indices)[valid]
        target = np.asarray(incidence.relation.target_indices)[valid]
        signs = np.asarray(incidence.signs)[valid]
        edge_count = bridge.cochain.cell_counts[1]
        tail = np.full(edge_count, -1, dtype=np.int32)
        head = np.full(edge_count, -1, dtype=np.int32)
        tail[target[signs < 0.0]] = source[signs < 0.0]
        head[target[signs > 0.0]] = source[signs > 0.0]
        if np.any(tail < 0) or np.any(head < 0):
            raise ValueError("Every oriented edge must have one tail and one head node.")
        self.bridge = bridge
        self.diffusivities = jnp.asarray(values)
        self.tail_indices = jnp.asarray(tail)
        self.head_indices = jnp.asarray(head)
        self.species_count = values.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cochain-electrochemical-flux",
                "bridge": bridge.bridge_id,
                "diffusivities": array_tree_fingerprint(values),
                "tail": array_tree_fingerprint(tail),
                "head": array_tree_fingerprint(head),
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        dimensionless_drift_potential: ArrayLike,
        /,
        *,
        dimensionless_electrochemical_potential: ArrayLike | None = None,
    ) -> CochainElectrochemicalFluxEvaluation:
        """Evaluate drift flux and entropy using distinct thermodynamic inputs.

        The full dimensionless electrochemical potential defaults to
        ``log(concentrations) + dimensionless_drift_potential``. A closure may
        supply it explicitly, with any spatially constant species reference.
        """

        concentration = jnp.asarray(concentrations)
        potential = jnp.asarray(
            dimensionless_drift_potential, dtype=concentration.dtype
        )
        node_count = self.bridge.cochain.cell_counts[0]
        if concentration.shape != (node_count, self.species_count) or (
            potential.shape != concentration.shape
        ):
            raise ValueError(
                "concentrations and drift potential must have node/species shape."
            )
        electrochemical = (
            jnp.log(jnp.where(concentration > 0.0, concentration, 1.0)) + potential
            if dimensionless_electrochemical_potential is None
            else jnp.asarray(
                dimensionless_electrochemical_potential, dtype=concentration.dtype
            )
        )
        if electrochemical.shape != concentration.shape:
            raise ValueError("Electrochemical potential must have node/species shape.")
        tail_concentration = concentration[self.tail_indices]
        head_concentration = concentration[self.head_indices]
        difference = potential[self.head_indices] - potential[self.tail_indices]
        flux = scharfetter_gummel_flux(
            tail_concentration, head_concentration, difference, self.diffusivities
        )
        rates = []
        for species in range(self.species_count):
            rates.append(self.bridge.cochain.codifferential(1, flux[:, species]))
        rate = jnp.stack(rates, axis=-1)
        weights = self.bridge.cochain.hodge_stars[0].astype(concentration.dtype)
        mass_defect = jnp.sum(weights[:, None] * rate, axis=0)
        consuming = rate < 0.0
        restriction = jnp.min(
            jnp.where(
                consuming,
                concentration / jnp.maximum(-rate, jnp.finfo(rate.dtype).tiny),
                jnp.inf,
            )
        )
        electrochemical_difference = (
            electrochemical[self.head_indices] - electrochemical[self.tail_indices]
        )
        edge_weights = self.bridge.cochain.hodge_stars[1].astype(concentration.dtype)
        dissipation = -jnp.sum(
            edge_weights[:, None] * flux * electrochemical_difference
        )
        scale = jnp.maximum(
            jnp.sum(weights[:, None] * jnp.abs(rate), axis=0),
            1.0,
        )
        tolerance = 256.0 * jnp.finfo(rate.dtype).eps * scale
        finite = jnp.all(jnp.isfinite(flux)) & jnp.all(jnp.isfinite(rate))
        conservative = jnp.all(jnp.abs(mass_defect) <= tolerance)
        successful = (
            finite
            & conservative
            & jnp.all(concentration > 0.0)
            & jnp.isfinite(dissipation)
            & (dissipation >= -256.0 * jnp.finfo(rate.dtype).eps)
        )
        return CochainElectrochemicalFluxEvaluation(
            flux,
            rate,
            mass_defect,
            dissipation,
            restriction,
            finite,
            conservative,
            successful,
            self.plan_id,
        )


__all__ = [
    "CochainElectrochemicalFluxEvaluation",
    "PreparedCochainElectrochemicalFlux",
    "scharfetter_gummel_flux",
    "stable_bernoulli",
]
