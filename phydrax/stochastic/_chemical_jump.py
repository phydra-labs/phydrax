#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..equations._chemical_mechanism import PreparedChemicalMechanism
from ..equations._chemical_rates import ChemicalRateKind, ChemicalRateRuntime
from ._jump import AbstractJumpProcess


class ChemicalJumpRuntime(StrictModule):
    temperature: Array
    pressure: Array
    rate_runtime: ChemicalRateRuntime

    def __init__(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        rate_runtime: ChemicalRateRuntime | None = None,
    ):
        temperature_value = jnp.asarray(temperature)
        pressure_value = jnp.asarray(pressure, dtype=temperature_value.dtype)
        if temperature_value.shape != () or pressure_value.shape != ():
            raise ValueError("Chemical jump temperature and pressure must be scalar.")
        self.temperature = temperature_value
        self.pressure = pressure_value
        self.rate_runtime = (
            ChemicalRateRuntime(jnp.zeros((0,), dtype=temperature_value.dtype), 0.0)
            if rate_runtime is None
            else rate_runtime
        )


class ChemicalJumpProcess(AbstractJumpProcess):
    """Exact-count jump process compiled from a deterministic chemical mechanism."""

    mechanism: PreparedChemicalMechanism
    system_measure: Array
    channel_reaction: Array
    channel_direction: Array
    channel_stoichiometry: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    channel_orders: Array
    channel_normalization: Array
    maximum_order: int = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        system_measure: ArrayLike,
        /,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        measure = jnp.asarray(system_measure)
        if measure.shape != ():
            raise ValueError("system_measure must be scalar.")
        reactant = np.asarray(mechanism.reactant_stoichiometry)
        product = np.asarray(mechanism.product_stoichiometry)
        orders = np.asarray(mechanism.forward_orders)
        if (
            np.any(reactant != np.round(reactant))
            or np.any(product != np.round(product))
            or np.any(orders != reactant)
        ):
            raise ValueError(
                "Chemical jump mechanisms require integral stoichiometric mass action."
            )
        allowed = {ChemicalRateKind.ARRHENIUS, ChemicalRateKind.PHOTOLYSIS}
        for reaction in mechanism.reactions:
            if reaction.forward_rate.kind not in allowed or (
                reaction.reverse_rate is not None
                and reaction.reverse_rate.kind not in allowed
            ):
                raise ValueError(
                    "Chemical jump mechanism contains a rate without discrete semantics."
                )
        reaction_indices = []
        directions = []
        channel_stoichiometry = []
        channel_orders = []
        for reaction_index, reaction in enumerate(mechanism.reactions):
            reaction_indices.append(reaction_index)
            directions.append(1)
            channel_stoichiometry.append(
                product[reaction_index] - reactant[reaction_index]
            )
            channel_orders.append(reactant[reaction_index])
            if reaction.reverse_rate is not None or reaction.thermodynamic_reversible:
                reaction_indices.append(reaction_index)
                directions.append(-1)
                channel_stoichiometry.append(
                    reactant[reaction_index] - product[reaction_index]
                )
                channel_orders.append(product[reaction_index])
        channel_stoichiometry_ = np.asarray(channel_stoichiometry, dtype=np.int32)
        channel_orders_ = np.asarray(channel_orders, dtype=np.int32)
        normalization = np.vectorize(factorial, otypes=[float])(channel_orders_)
        self.mechanism = mechanism
        self.system_measure = measure
        self.channel_reaction = jnp.asarray(reaction_indices, dtype=jnp.int32)
        self.channel_direction = jnp.asarray(directions, dtype=jnp.int32)
        self.channel_stoichiometry = jnp.asarray(channel_stoichiometry_)
        self.channel_orders = jnp.asarray(channel_orders_)
        self.channel_normalization = jnp.asarray(normalization)
        self.maximum_order = int(np.max(channel_orders_, initial=0))
        self.channel_count = len(reaction_indices)
        self.process_id = canonical_fingerprint(
            {
                "kind": "chemical-jump-process",
                "mechanism": mechanism.mechanism_id,
                "measure": array_tree_fingerprint(np.asarray(measure)),
                "channels": array_tree_fingerprint(channel_stoichiometry_),
            }
        )
        self.state_shape = (mechanism.schema.species_count,)
        self.num_channels = len(reaction_indices)
        self.mark_shape = ()

    def intensities(self, time, state, args=None, /):
        del time
        counts = jnp.asarray(state)
        if counts.shape[-1] != self.mechanism.schema.species_count:
            raise ValueError("Chemical jump state must end in species axis.")
        if not isinstance(args, ChemicalJumpRuntime):
            raise TypeError("Chemical jump process requires ChemicalJumpRuntime args.")
        concentration = counts / self.system_measure
        evaluation = self.mechanism.evaluate(
            concentration,
            args.temperature,
            args.pressure,
            runtime=args.rate_runtime,
        )
        forward = evaluation.forward_rate_constants[self.channel_reaction]
        reverse = evaluation.reverse_rate_constants[self.channel_reaction]
        constants = jnp.where(self.channel_direction > 0, forward, reverse)
        combinatorial = _falling_factorial_mass_action(
            counts,
            self.channel_orders,
            self.channel_normalization,
            self.maximum_order,
        )
        total_order = jnp.sum(self.channel_orders, axis=-1)
        scaling = self.system_measure ** (1.0 - total_order)
        intensity = constants * scaling * combinatorial
        valid = (
            evaluation.successful
            & jnp.all(counts >= 0.0)
            & jnp.all(counts == jnp.floor(counts))
            & jnp.all(jnp.isfinite(intensity) & (intensity >= 0.0))
            & jnp.isfinite(self.system_measure)
            & (self.system_measure > 0.0)
        )
        return jnp.where(valid, intensity, jnp.nan)

    def jump(self, state, channel, mark, args=None, /):
        del mark, args
        return (
            jnp.asarray(state)
            + self.channel_stoichiometry[jnp.asarray(channel, dtype=jnp.int32)]
        )

    def sample_mark(self, key, time, state, channel, args=None, /):
        del key, time, channel, args
        return jnp.asarray(0, dtype=jnp.asarray(state).dtype)

    def conservation_residual(self, state, reference_invariant, /):
        values = jnp.asarray(state)
        elements = contract(
            "...s,es->...e",
            values,
            self.mechanism.schema.element_composition,
        )
        charge = contract("...s,s->...", values, self.mechanism.schema.charges)
        invariant = jnp.concatenate((elements, charge[..., None]), axis=-1)
        return invariant - jnp.asarray(reference_invariant)


def _falling_factorial_mass_action(counts, orders, normalization, maximum_order):
    terms = []
    for species in range(orders.shape[-1]):
        order = orders[:, species]
        value = counts[..., species, None]
        factor = jnp.ones(value.shape[:-1] + (orders.shape[0],), dtype=counts.dtype)
        for count in range(maximum_order):
            factor = factor * jnp.where(order > count, value - float(count), 1.0)
        terms.append(factor / normalization[:, species])
    return jnp.prod(jnp.stack(terms, axis=-1), axis=-1)


__all__ = ["ChemicalJumpProcess", "ChemicalJumpRuntime"]
