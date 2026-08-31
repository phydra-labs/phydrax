#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations import MultispeciesEulerSystem
from ._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ._balance_law_transport import (
    AbstractPreparedBalanceLawTransport,
    BalanceLawSourceView,
)


class StoichiometricReactionNetwork(StrictModule, NonTrainableState):
    stoichiometry: Array
    reactant_orders: Array
    rate_constants: Array
    energy_releases: Array
    invariant_matrix: Array
    species_names: tuple[str, ...] = eqx.field(static=True)
    reaction_names: tuple[str, ...] = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        stoichiometry: ArrayLike,
        reactant_orders: ArrayLike,
        rate_constants: ArrayLike,
        /,
        *,
        energy_releases: ArrayLike | None = None,
        invariant_matrix: ArrayLike | None = None,
        species_names: tuple[str, ...] | None = None,
        reaction_names: tuple[str, ...] | None = None,
    ):
        stoichiometry_ = np.asarray(stoichiometry, dtype=float)
        orders = np.asarray(reactant_orders, dtype=float)
        rates = np.asarray(rate_constants, dtype=float)
        if stoichiometry_.ndim != 2:
            raise ValueError("Stoichiometry must have shape (reactions, species).")
        reactions, species = stoichiometry_.shape
        releases = (
            np.zeros((reactions,), dtype=float)
            if energy_releases is None
            else np.asarray(energy_releases, dtype=float)
        )
        invariants = (
            np.zeros((0, species), dtype=float)
            if invariant_matrix is None
            else np.asarray(invariant_matrix, dtype=float)
        )
        species_ids = (
            tuple(f"species-{index}" for index in range(species))
            if species_names is None
            else tuple(species_names)
        )
        reaction_ids = (
            tuple(f"reaction-{index}" for index in range(reactions))
            if reaction_names is None
            else tuple(reaction_names)
        )
        if (
            orders.shape != stoichiometry_.shape
            or rates.shape != (reactions,)
            or releases.shape != (reactions,)
            or invariants.ndim != 2
            or invariants.shape[1] != species
            or np.any(~np.isfinite(stoichiometry_))
            or np.any(~np.isfinite(orders))
            or np.any(orders < 0.0)
            or np.any(~np.isfinite(rates))
            or np.any(rates < 0.0)
            or np.any(~np.isfinite(releases))
            or np.any(~np.isfinite(invariants))
            or len(species_ids) != species
            or len(reaction_ids) != reactions
            or np.max(np.abs(invariants @ stoichiometry_.T), initial=0.0) > 1e-12
        ):
            raise ValueError(
                "Reaction network structure or conservation invariants are invalid."
            )
        self.stoichiometry = jnp.asarray(stoichiometry_)
        self.reactant_orders = jnp.asarray(orders)
        self.rate_constants = jnp.asarray(rates)
        self.energy_releases = jnp.asarray(releases)
        self.invariant_matrix = jnp.asarray(invariants)
        self.species_names = species_ids
        self.reaction_names = reaction_ids
        self.network_id = canonical_fingerprint(
            {
                "kind": "stoichiometric-reaction-network",
                "stoichiometry": array_tree_fingerprint(stoichiometry_),
                "orders": array_tree_fingerprint(orders),
                "rate_constants": array_tree_fingerprint(rates),
                "energy_releases": array_tree_fingerprint(releases),
                "invariants": array_tree_fingerprint(invariants),
                "species_names": list(species_ids),
                "reaction_names": list(reaction_ids),
            }
        )

    @property
    def species_count(self) -> int:
        return int(self.stoichiometry.shape[1])

    def reaction_rates(self, species_density: Array, /) -> Array:
        safe = jnp.maximum(species_density, jnp.finfo(species_density.dtype).tiny)
        logarithmic = oe.contract("...s,rs->...r", jnp.log(safe), self.reactant_orders)
        return self.rate_constants * jnp.exp(logarithmic)


class ThermochemistryDiagnostics(StrictModule):
    species_before: Array
    species_after: Array
    invariant_defect: Array
    energy_change: Array
    successful: Array


class ThermochemistryProcessPlan(AbstractBalanceLawProcessPlan):
    network: StoichiometricReactionNetwork
    subcycles: int = eqx.field(static=True)
    safety_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        network: StoichiometricReactionNetwork,
        /,
        *,
        subcycles: int = 8,
        safety_fraction: float = 0.25,
    ):
        if not isinstance(network, StoichiometricReactionNetwork):
            raise TypeError("network must be StoichiometricReactionNetwork.")
        count = int(subcycles)
        fraction = float(safety_fraction)
        if count <= 0 or not 0.0 < fraction <= 1.0:
            raise ValueError("Thermochemistry integration controls are invalid.")
        self.network = network
        self.subcycles = count
        self.safety_fraction = fraction
        self.process_id = canonical_fingerprint(
            {
                "kind": "thermochemistry-process",
                "network": network.network_id,
                "subcycles": count,
                "safety_fraction": fraction,
            }
        )

    def prepare(
        self, transport: AbstractPreparedBalanceLawTransport, /
    ) -> PreparedThermochemistryProcess:
        return PreparedThermochemistryProcess(self, transport)


class PreparedThermochemistryProcess(AbstractPreparedBalanceLawProcess):
    plan: ThermochemistryProcessPlan
    transport: AbstractPreparedBalanceLawTransport
    species_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)

    def __init__(
        self,
        plan: ThermochemistryProcessPlan,
        transport: AbstractPreparedBalanceLawTransport,
        /,
    ):
        if not isinstance(transport.dynamics.system, MultispeciesEulerSystem):
            raise TypeError("Continuum thermochemistry requires MultispeciesEulerSystem.")
        system = transport.dynamics.system
        if system.species_count != plan.network.species_count:
            raise ValueError(
                "Reaction network species do not match the transport system."
            )
        self.plan = plan
        self.transport = transport
        self.species_indices = tuple(range(system.species_count))
        self.energy_index = len(system.component_names) - 1
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-thermochemistry-process",
                "plan": plan.process_id,
                "transport": transport.transport_id,
            }
        )
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = "branchwise-explicit-subcycled"
        self.modified_components = tuple(
            system.component_names[index] for index in self.species_indices
        ) + ("total_energy",)

    def initialize(
        self, source_view: BalanceLawSourceView, args: Any = None, /
    ) -> BalanceLawProcessState:
        del source_view, args
        return BalanceLawProcessState.empty(self.process_id)

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, process_state, args
        species = cell_average[..., self.species_indices]
        rates = self.plan.network.reaction_rates(species)
        consumption = -jnp.minimum(self.plan.network.stoichiometry, 0.0)
        consumption_rate = oe.contract("...r,rs->...s", rates, consumption)
        local = jnp.where(consumption_rate > 0.0, species / consumption_rate, jnp.inf)
        return self.plan.safety_fraction * jnp.min(local)

    def advance(
        self,
        start_time: Array,
        end_time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        realization: Any = None,
        args: Any = None,
        /,
    ) -> BalanceLawProcessAdvance:
        del realization, args
        incoming = jnp.asarray(cell_average)
        species_before = incoming[..., self.species_indices]
        energy_before = incoming[..., self.energy_index]
        substep = (end_time - start_time) / self.plan.subcycles

        def body(_, carry):
            species, energy = carry
            rates = self.plan.network.reaction_rates(species)
            reaction_amount = substep * rates
            species_next = species + oe.contract(
                "...r,rs->...s", reaction_amount, self.plan.network.stoichiometry
            )
            energy_next = energy + oe.contract(
                "...r,r->...", reaction_amount, self.plan.network.energy_releases
            )
            return species_next, energy_next

        species_after, energy_after = jax.lax.fori_loop(
            0,
            self.plan.subcycles,
            body,
            (species_before, energy_before),
        )
        candidate = incoming.at[..., self.species_indices].set(species_after)
        candidate = candidate.at[..., self.energy_index].set(energy_after)
        invariant_before = oe.contract(
            "...s,is->...i", species_before, self.plan.network.invariant_matrix
        )
        invariant_after = oe.contract(
            "...s,is->...i", species_after, self.plan.network.invariant_matrix
        )
        invariant_defect = invariant_after - invariant_before
        successful = (
            jnp.all(species_after >= 0.0)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.abs(invariant_defect) <= 1e-10)
            & jnp.all(self.transport.dynamics.system.admissible(candidate))
        )
        accepted = jnp.where(successful, candidate, incoming)
        diagnostics = ThermochemistryDiagnostics(
            species_before=species_before,
            species_after=accepted[..., self.species_indices],
            invariant_defect=invariant_defect,
            energy_change=accepted[..., self.energy_index] - energy_before,
            successful=successful,
        )
        return BalanceLawProcessAdvance(
            cell_average=accepted,
            process_state=process_state,
            successful=successful,
            source_change=accepted - incoming,
            diagnostics=diagnostics,
        )


__all__ = [
    "PreparedThermochemistryProcess",
    "StoichiometricReactionNetwork",
    "ThermochemistryDiagnostics",
    "ThermochemistryProcessPlan",
]
