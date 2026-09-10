#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermal fusion reactivity and branch-resolved product sources."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._interpolation import linear_interpolate

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._identity import NuclearSpeciesTable
from ._reaction import NuclearReactionChannel


class ReactivityEvaluation(StrictModule):
    reactivity_m3_s: Array
    valid: Array
    finite: Array
    successful: Array


class TabulatedMaxwellianReactivity(StrictModule, NonTrainableState):
    """Piecewise-linear Maxwellian reactivity without extrapolation authority."""

    thermal_energy_j: Array
    reactivity_m3_s: Array
    channel_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermal_energy_j: ArrayLike,
        reactivity_m3_s: ArrayLike,
        channel: NuclearReactionChannel,
        /,
    ):
        if not isinstance(channel, NuclearReactionChannel):
            raise TypeError("channel must be NuclearReactionChannel.")
        energy_host = np.asarray(thermal_energy_j, dtype=np.float64)
        reactivity_host = np.asarray(reactivity_m3_s, dtype=np.float64)
        if (
            energy_host.ndim != 1
            or energy_host.size < 2
            or reactivity_host.shape != energy_host.shape
        ):
            raise ValueError(
                "Reactivity tables require matching nontrivial rank-one arrays."
            )
        if (
            np.any(~np.isfinite(energy_host))
            or np.any(~np.isfinite(reactivity_host))
            or np.any(energy_host <= 0.0)
            or np.any(np.diff(energy_host) <= 0.0)
            or np.any(reactivity_host < 0.0)
        ):
            raise ValueError(
                "Reactivity coordinates and values violate physical support."
            )
        energy = jax.lax.stop_gradient(jnp.asarray(energy_host))
        reactivity = jax.lax.stop_gradient(jnp.asarray(reactivity_host))
        self.thermal_energy_j = energy
        self.reactivity_m3_s = reactivity
        self.channel_id = channel.channel_id
        self.data_id = channel.data.provenance_id
        self.model_id = canonical_fingerprint(
            {
                "kind": "tabulated-maxwellian-reactivity",
                "thermal_energy_j": array_tree_fingerprint(energy_host),
                "reactivity_m3_s": array_tree_fingerprint(reactivity_host),
                "channel": channel.channel_id,
                "data": channel.data.provenance_id,
                "interpolation": "piecewise-linear-no-extrapolation",
            }
        )

    def evaluate(self, thermal_energy_j: ArrayLike, /) -> ReactivityEvaluation:
        temperature = jnp.asarray(thermal_energy_j, dtype=self.thermal_energy_j.dtype)
        valid = (
            jnp.isfinite(temperature)
            & (temperature >= self.thermal_energy_j[0])
            & (temperature <= self.thermal_energy_j[-1])
        )
        value = linear_interpolate(
            self.thermal_energy_j, self.reactivity_m3_s, temperature
        ).values
        finite = jnp.all(jnp.isfinite(value))
        successful = finite & jnp.all(valid)
        return ReactivityEvaluation(value, valid, finite, successful)


class FusionProductSource(StrictModule):
    product_rate_density_m3_s: Array
    product_power_density_w_m3: Array
    kinetic_energy_j: Array
    product_species_ids: tuple[str, str] = eqx.field(static=True)
    isotropic: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


class ThermalFusionReactionResult(StrictModule):
    reaction_rate_density_m3_s: Array
    reactant_sink_density_m3_s: Array
    products: FusionProductSource
    total_power_density_w_m3: Array
    support_valid: Array
    finite: Array
    conservation_valid: Array
    sensitivity_valid: Array
    successful: Array


class ThermalFusionReactionPlan(StrictModule, NonTrainableState):
    """One two-reactant, two-product Maxwellian fusion branch."""

    reactivity: TabulatedMaxwellianReactivity
    product_kinetic_energy_j: Array
    reactant_species_ids: tuple[str, str] = eqx.field(static=True)
    product_species_ids: tuple[str, str] = eqx.field(static=True)
    identical_reactants: bool = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        channel: NuclearReactionChannel,
        reactivity: TabulatedMaxwellianReactivity,
        species: NuclearSpeciesTable,
        /,
    ):
        if not isinstance(channel, NuclearReactionChannel):
            raise TypeError("channel must be NuclearReactionChannel.")
        if not isinstance(reactivity, TabulatedMaxwellianReactivity):
            raise TypeError("reactivity must be TabulatedMaxwellianReactivity.")
        if not isinstance(species, NuclearSpeciesTable):
            raise TypeError("species must be NuclearSpeciesTable.")
        if reactivity.channel_id != channel.channel_id:
            raise ValueError("Reactivity and reaction channel identities disagree.")
        if (
            len(channel.reactants) != 2
            or len(channel.products) != 2
            or any(
                item.multiplicity != 1 for item in (*channel.reactants, *channel.products)
            )
        ):
            raise ValueError(
                "Thermal fusion initially requires two unit-multiplicity reactants and products."
            )
        if channel.q_value_j <= 0.0:
            raise ValueError("Thermal fusion branch requires positive released energy.")
        product_masses = jnp.asarray(
            [
                species.rest_masses_kg[species.index(item.species)]
                for item in channel.products
            ]
        )
        total_product_mass = jnp.sum(product_masses)
        kinetic = channel.q_value_j * product_masses[::-1] / total_product_mass
        reactant_ids = tuple(item.species.species_id for item in channel.reactants)
        product_ids = tuple(item.species.species_id for item in channel.products)
        self.reactivity = reactivity
        self.product_kinetic_energy_j = kinetic
        self.reactant_species_ids = reactant_ids
        self.product_species_ids = product_ids
        self.identical_reactants = reactant_ids[0] == reactant_ids[1]
        self.channel_id = channel.channel_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thermal-fusion-reaction-plan",
                "channel": channel.channel_id,
                "reactivity": reactivity.model_id,
                "mass_table": species.table_id,
                "products": list(product_ids),
            }
        )

    def evaluate(
        self,
        reactant_a_density_m3: ArrayLike,
        reactant_b_density_m3: ArrayLike,
        thermal_energy_j: ArrayLike,
        /,
    ) -> ThermalFusionReactionResult:
        density_a = jnp.asarray(reactant_a_density_m3)
        density_b = jnp.asarray(reactant_b_density_m3, dtype=density_a.dtype)
        temperature = jnp.asarray(thermal_energy_j, dtype=density_a.dtype)
        if density_a.shape != density_b.shape or density_a.shape != temperature.shape:
            raise ValueError(
                "Fusion densities and thermal energy must have identical shapes."
            )
        evaluation = self.reactivity.evaluate(temperature)
        symmetry = 0.5 if self.identical_reactants else 1.0
        rate = symmetry * density_a * density_b * evaluation.reactivity_m3_s
        reactant_sink = jnp.stack((rate, rate), axis=-1)
        product_rate = jnp.stack((rate, rate), axis=-1)
        product_power = product_rate * self.product_kinetic_energy_j
        total_power = jnp.sum(product_power, axis=-1)
        finite = (
            jnp.all(jnp.isfinite(density_a))
            & jnp.all(jnp.isfinite(density_b))
            & jnp.all(jnp.isfinite(rate))
            & jnp.all(jnp.isfinite(product_power))
        )
        nonnegative = (
            jnp.all(density_a >= 0.0) & jnp.all(density_b >= 0.0) & jnp.all(rate >= 0.0)
        )
        energy_residual = total_power - rate * jnp.sum(self.product_kinetic_energy_j)
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(total_power)))
        conservation = jnp.all(
            jnp.abs(energy_residual) <= 256.0 * jnp.finfo(rate.dtype).eps * scale
        )
        source_id = canonical_fingerprint(
            {
                "kind": "fusion-product-source",
                "plan": self.plan_id,
                "products": list(self.product_species_ids),
                "angular_distribution": "isotropic",
            }
        )
        products = FusionProductSource(
            product_rate,
            product_power,
            self.product_kinetic_energy_j,
            self.product_species_ids,
            True,
            source_id,
        )
        successful = evaluation.successful & finite & nonnegative & conservation
        return ThermalFusionReactionResult(
            rate,
            reactant_sink,
            products,
            total_power,
            evaluation.valid,
            finite,
            conservation,
            successful,
            successful,
        )


__all__ = [
    "FusionProductSource",
    "ReactivityEvaluation",
    "TabulatedMaxwellianReactivity",
    "ThermalFusionReactionPlan",
    "ThermalFusionReactionResult",
]
