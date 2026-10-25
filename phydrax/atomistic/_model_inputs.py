#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import Bool, checked, Float, Integer, parse, PRNGKey, Scope
from ._conditioning import AtomisticConditioning, AtomisticConditionStatus
from ._dimensions import AtomisticNodeDim
from ._graph import AtomisticGraph
from ._types import AtomisticScaleContract, AtomisticSpeciesKind


@final
class AtomisticGraphInputs(StrictModule, NonTrainableState):
    """One bound geometry and its declared species, units and conditioning.

    Positions and current node metadata belong to ``graph``. They are not
    separately supplied or duplicated in this PyTree: a model cannot receive
    current coordinates alongside edges bound to an older coordinate argument.
    Conditioning is a dynamic fixed-role input, not trainable model parameters.
    """

    __strict_contract__ = True

    graph: AtomisticGraph
    scale: AtomisticScaleContract
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    conditioning: AtomisticConditioning | None
    key: PRNGKey | None

    @checked
    def __init__(
        self,
        graph: AtomisticGraph,
        scale: AtomisticScaleContract,
        /,
        *,
        species_kind: AtomisticSpeciesKind,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
    ) -> None:
        kind = parse(species_kind, AtomisticSpeciesKind, "species_kind")
        scope = Scope()
        positions = parse(
            graph.positions, Float[AtomisticNodeDim, Literal[3]], "positions", scope=scope
        )
        parse(graph.topology.atom_mask, Bool[AtomisticNodeDim], "atom_mask", scope=scope)
        parse(
            graph.topology.atom_cases,
            Integer[AtomisticNodeDim],
            "atom_cases",
            scope=scope,
        )
        parse(
            graph.graph.nodes["atomic_numbers"],
            Integer[AtomisticNodeDim],
            "atomic_numbers",
            scope=scope,
        )
        parse(
            graph.graph.nodes["atom_type_ids"],
            Integer[AtomisticNodeDim],
            "atom_type_ids",
            scope=scope,
        )
        if positions.shape[0] != graph.topology.case_count * graph.topology.atom_capacity:
            raise ValueError("Graph node extent differs from its declared case layout.")
        if conditioning is not None and (
            conditioning.case_count != graph.topology.case_count
            or conditioning.atom_capacity != graph.topology.atom_capacity
        ):
            raise ValueError("Conditioning belongs to another case/atom layout.")
        self.graph = graph
        self.scale = scale
        self.species_kind = kind
        self.conditioning = conditioning
        self.key = key

    @property
    def positions(self) -> Array:
        return self.graph.positions

    @property
    def atom_mask(self) -> Array:
        return self.graph.topology.atom_mask

    @property
    def atom_cases(self) -> Array:
        return self.graph.topology.atom_cases

    @property
    def case_count(self) -> int:
        return self.graph.topology.case_count

    @property
    def atom_capacity(self) -> int:
        return self.graph.topology.atom_capacity

    @property
    def species_ids(self) -> Array:
        match self.species_kind:
            case AtomisticSpeciesKind.ATOMIC_NUMBER:
                return jnp.asarray(self.graph.graph.nodes["atomic_numbers"])
            case AtomisticSpeciesKind.ATOM_TYPE_ID:
                return jnp.asarray(self.graph.graph.nodes["atom_type_ids"])
            case _:
                assert_never(self.species_kind)

    @property
    def route_identity_fields(self) -> tuple[Array, ...]:
        """Semantic addresses, independent of route storage and particle ordering."""
        topology = self.graph.topology
        words = topology.case_identity_words[topology.edge_cases]
        fields = (
            *(words[:, column] for column in range(words.shape[1])),
            topology.particle_ids[topology.senders],
            topology.particle_ids[topology.receivers],
        )
        shifts = topology.image_shifts
        counts = self.graph.image_counts
        if counts is not None:
            # x_material = x_represented + image_count @ H.
            shifts = (
                shifts.astype(jnp.int64)
                + counts[topology.senders].astype(jnp.int64)
                - counts[topology.receivers].astype(jnp.int64)
            )
        return (*fields, *(shifts[:, axis] for axis in range(shifts.shape[1])))

    @property
    def successful(self) -> Array:
        match self.species_kind:
            case AtomisticSpeciesKind.ATOMIC_NUMBER:
                admitted = self.species_ids > 0
            case AtomisticSpeciesKind.ATOM_TYPE_ID:
                admitted = self.species_ids >= 0
            case _:
                assert_never(self.species_kind)
        valid = self.graph.valid & jnp.all(
            (~self.atom_mask | admitted).reshape((self.case_count, self.atom_capacity)),
            axis=1,
        )
        if self.conditioning is None:
            return valid
        return valid & (
            self.conditioning.case_status(self.atom_mask)
            == int(AtomisticConditionStatus.SUCCESS)
        )

    @checked
    def require_compatible(
        self,
        scale: AtomisticScaleContract,
        species_kind: AtomisticSpeciesKind,
        /,
    ) -> None:
        kind = parse(species_kind, AtomisticSpeciesKind, "species_kind")
        if self.scale.scale_id != scale.scale_id:
            raise ValueError("Model and graph input scale contracts differ.")
        if self.species_kind is not kind:
            raise ValueError("Model and graph input species identities differ.")


__all__ = ["AtomisticGraphInputs"]
