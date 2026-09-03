#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax.ein import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._graph import (
    AtomisticGraph,
    AtomisticGraphExecutionPlan,
    realize_atomistic_graph,
)
from ...atomistic._potential import (
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
    AtomisticSpeciesKind,
    initialize_atomistic_potential_identity,
)
from ...atomistic._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
)
from ..layers import Linear
from ..parameters import IdentityTransform


class _PaiNNConfiguration(StrictModule, NonTrainableState):
    radial_frequencies: Array
    cutoff: float = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    maximum_species_id: int = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)


class _PaiNNInteraction(StrictModule):
    filter_in: Linear
    filter_out: Linear
    message_in: Linear
    message_out: Linear
    vector_u: Linear
    vector_v: Linear
    update_in: Linear
    update_out: Linear

    def __init__(self, feature_count: int, radial_basis_count: int, key: Key[Array, ""]):
        keys = jr.split(key, 8)
        identity = IdentityTransform()
        self.filter_in = Linear(
            in_size=radial_basis_count,
            out_size=feature_count,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=identity,
            key=keys[0],
        )
        self.filter_out = Linear(
            in_size=feature_count,
            out_size=3 * feature_count,
            rwf=False,
            weight_transform=identity,
            key=keys[1],
        )
        self.message_in = Linear(
            in_size=feature_count,
            out_size=feature_count,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=identity,
            key=keys[2],
        )
        self.message_out = Linear(
            in_size=feature_count,
            out_size=3 * feature_count,
            rwf=False,
            weight_transform=identity,
            key=keys[3],
        )
        self.vector_u = Linear(
            in_size=feature_count,
            out_size=feature_count,
            rwf=False,
            use_bias=False,
            weight_transform=identity,
            key=keys[4],
        )
        self.vector_v = Linear(
            in_size=feature_count,
            out_size=feature_count,
            rwf=False,
            use_bias=False,
            weight_transform=identity,
            key=keys[5],
        )
        self.update_in = Linear(
            in_size=2 * feature_count,
            out_size=feature_count,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=identity,
            key=keys[6],
        )
        self.update_out = Linear(
            in_size=feature_count,
            out_size=3 * feature_count,
            rwf=False,
            weight_transform=identity,
            key=keys[7],
        )

    def atomwise_update(self, scalar: Array, vector: Array, /) -> tuple[Array, Array]:
        """Apply the norm-conditioned canonical three-head PaiNN atomwise update."""

        vector_u = self.vector_u(vector)
        vector_v = self.vector_v(vector)
        squared_norm = contract("ndf,ndf->nf", vector_v, vector_v)
        tiny = jnp.asarray(jnp.finfo(vector.dtype).tiny, dtype=vector.dtype)
        vector_norm = jnp.where(
            squared_norm > 0.0,
            jnp.sqrt(jnp.maximum(squared_norm, tiny)),
            0.0,
        )
        invariant = contract("ndf,ndf->nf", vector_u, vector_v)
        update = self.update_out(
            self.update_in(jnp.concatenate((scalar, vector_norm), axis=-1))
        )
        scalar_scalar, scalar_vector, vector_vector = jnp.split(update, 3, axis=-1)
        return (
            scalar + scalar_scalar + scalar_vector * invariant,
            vector + vector_vector[:, None, :] * vector_u,
        )

    def __call__(
        self,
        scalar: Array,
        vector: Array,
        graph: AtomisticGraph,
        radial: Array,
        cutoff_weight: Array,
        /,
    ) -> tuple[Array, Array]:
        ir = graph.graph
        if ir.senders is None or ir.receivers is None or ir.edge_mask is None:
            raise ValueError("PaiNN requires an explicit masked edge relation.")
        feature_count = int(scalar.shape[-1])
        edge_mask = ir.edge_mask[:, None]
        safe_radial = jnp.where(edge_mask, radial, 0.0)
        safe_cutoff = jnp.where(edge_mask, cutoff_weight, 0.0)
        filtered = self.filter_out(self.filter_in(safe_radial)) * safe_cutoff
        sender_scalar = jnp.where(edge_mask, scalar[ir.senders], 0.0)
        message = self.message_out(self.message_in(sender_scalar)) * filtered
        message = jnp.where(edge_mask, message, 0.0)
        scalar_coefficient, vector_coefficient, direction_coefficient = jnp.split(
            message, 3, axis=-1
        )
        direction = jnp.where(
            edge_mask,
            jnp.asarray(ir.edges["direction"], dtype=scalar.dtype),
            0.0,
        )
        sender_vector = jnp.where(edge_mask[:, :, None], vector[ir.senders], 0.0)
        vector_message = (
            vector_coefficient[:, None, :] * sender_vector
            + direction_coefficient[:, None, :] * direction[:, :, None]
        )
        scalar_delta = jnp.zeros_like(scalar).at[ir.receivers].add(scalar_coefficient)
        vector_delta = jnp.zeros_like(vector).at[ir.receivers].add(vector_message)
        scalar, vector = self.atomwise_update(
            scalar + scalar_delta, vector + vector_delta
        )
        if int(scalar.shape[-1]) != feature_count:
            raise RuntimeError("PaiNN interaction changed its scalar feature width.")
        return scalar, vector


class PaiNNPotential(AbstractAtomisticPotential):
    """Finite nonperiodic molecular PaiNN scalar energy potential.

    The candidate topology is fixed by ``AtomisticBatch``. Geometry only changes
    differentiable displacement payloads and smooth cutoff weights; no edge is
    truncated or rebuilt inside the force derivative.
    """

    embedding: Array
    interactions: tuple[_PaiNNInteraction, ...]
    readout_hidden: Linear
    readout_energy: Linear
    configuration: _PaiNNConfiguration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    parameter_state_id: str = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        /,
        *,
        cutoff: float,
        feature_count: int = 64,
        interaction_count: int = 3,
        radial_basis_count: int = 20,
        maximum_species_id: int = 118,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
        precision: AtomisticPrecisionPolicy | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        cutoff_value = float(cutoff)
        features = int(feature_count)
        interactions = int(interaction_count)
        radial_count = int(radial_basis_count)
        maximum_z = int(maximum_species_id)
        if not math.isfinite(cutoff_value) or cutoff_value <= 0.0:
            raise ValueError("cutoff must be finite and positive.")
        if features <= 0 or interactions <= 0 or radial_count <= 0:
            raise ValueError(
                "PaiNN feature, interaction, and radial counts must be positive."
            )
        if maximum_z <= 0:
            raise ValueError("maximum_species_id must be positive.")
        if not isinstance(species_kind, AtomisticSpeciesKind):
            raise TypeError("species_kind must be AtomisticSpeciesKind.")
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
        keys = jr.split(key, interactions + 3)
        embedding = jr.normal(
            keys[0], (maximum_z + 1, features), dtype=jnp.dtype(precision_.compute_dtype)
        ) / jnp.sqrt(jnp.asarray(features, dtype=precision_.compute_dtype))
        interaction_modules = tuple(
            _PaiNNInteraction(features, radial_count, keys[index + 1])
            for index in range(interactions)
        )
        readout_hidden = Linear(
            in_size=features,
            out_size=features,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=IdentityTransform(),
            key=keys[-2],
        )
        readout_energy = Linear(
            in_size=features,
            out_size="scalar",
            rwf=False,
            weight_transform=IdentityTransform(),
            key=keys[-1],
        )
        compute_dtype = jnp.dtype(precision_.compute_dtype)

        def cast_inexact(value: Any) -> Any:
            if eqx.is_inexact_array(value):
                return value.astype(compute_dtype)
            return value

        self.embedding = embedding
        self.interactions = jax.tree_util.tree_map(cast_inexact, interaction_modules)
        self.readout_hidden = jax.tree_util.tree_map(cast_inexact, readout_hidden)
        self.readout_energy = jax.tree_util.tree_map(cast_inexact, readout_energy)
        self.configuration = _PaiNNConfiguration(
            radial_frequencies=jnp.arange(1, radial_count + 1, dtype=compute_dtype),
            cutoff=cutoff_value,
            feature_count=features,
            interaction_count=interactions,
            radial_basis_count=radial_count,
            maximum_species_id=maximum_z,
            species_kind=species_kind,
        )
        self.scale = scale
        self.precision = precision_
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "painn-architecture",
                "scale": scale.scale_id,
                "precision": precision_.policy_id,
                "cutoff": cutoff_value,
                "feature_count": features,
                "interaction_count": interactions,
                "radial_basis_count": radial_count,
                "maximum_species_id": maximum_z,
                "species_kind": species_kind.value,
            }
        )
        self.method_id = "negative-position-gradient-of-total-painn-energy"
        (
            self.parameter_state_id,
            self.potential_id,
        ) = initialize_atomistic_potential_identity(self)

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities(
            species_kind=self.configuration.species_kind
        )

    def parameter_state_tree(self, /) -> Any:
        return {
            "embedding": self.embedding,
            "interactions": self.interactions,
            "readout_hidden": self.readout_hidden,
            "readout_energy": self.readout_energy,
        }

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if not isinstance(batch, AtomisticBatch):
            raise TypeError("batch must be an AtomisticBatch.")
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError(
                "Potential and structure must share one exact scale contract."
            )
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the PaiNN precision contract."
            )
        if batch.has_periodic_metadata:
            raise ValueError(
                "PaiNNPotential supports finite nonperiodic molecules only and rejects "
                "preserved cell or periodic metadata."
            )

    def _radial_basis(self, distance: Array, /) -> tuple[Array, Array]:
        dtype = jnp.dtype(self.precision.compute_dtype)
        radius = jnp.asarray(distance, dtype=dtype)
        cutoff = jnp.asarray(self.configuration.cutoff, dtype=dtype)
        scaled = radius / cutoff
        frequencies = self.configuration.radial_frequencies
        basis = (jnp.pi * frequencies / cutoff) * jnp.sinc(
            scaled[:, None] * frequencies[None, :]
        )
        envelope = jnp.where(
            scaled < 1.0,
            0.5 * (jnp.cos(jnp.pi * scaled) + 1.0),
            0.0,
        )
        return basis, envelope[:, None]

    def graph_energy(
        self,
        species_ids: Array,
        atom_mask: Array,
        atom_cases: Array,
        case_count: int,
        atom_capacity: int,
        graph: AtomisticGraph,
        /,
    ) -> tuple[Array, Array]:
        numbers = jnp.asarray(species_ids).reshape((-1,))
        numbers = eqx.error_if(
            numbers,
            jnp.any(numbers > self.configuration.maximum_species_id),
            "Species ID exceeds PaiNNPotential.maximum_species_id.",
        )
        mask = jnp.asarray(atom_mask, dtype=bool).reshape((-1,))
        scalar = self.embedding[numbers].astype(self.precision.compute_dtype)
        vector = jnp.zeros(
            (scalar.shape[0], 3, self.configuration.feature_count),
            dtype=self.precision.compute_dtype,
        )
        scalar = scalar * mask[:, None]
        distance = jnp.asarray(graph.graph.edges["distance"])[:, 0]
        radial, cutoff_weight = self._radial_basis(distance)
        for interaction in self.interactions:
            scalar, vector = interaction(scalar, vector, graph, radial, cutoff_weight)
            scalar = scalar * mask[:, None]
            vector = vector * mask[:, None, None]
        atom_energy = self.readout_energy(self.readout_hidden(scalar))
        atom_energy = atom_energy * mask.astype(atom_energy.dtype)
        total_energy = (
            jnp.zeros((case_count,), dtype=self.precision.reduction_dtype)
            .at[jnp.asarray(atom_cases).reshape((-1,))]
            .add(atom_energy.astype(self.precision.reduction_dtype))
        )
        return (
            total_energy.astype(self.precision.output_dtype),
            atom_energy.reshape((case_count, atom_capacity)).astype(
                self.precision.output_dtype
            ),
        )

    def _energy_unchecked(
        self,
        batch: AtomisticBatch,
        positions: Array,
        execution: AtomisticGraphExecutionPlan,
        /,
    ) -> tuple[Array, Array, AtomisticGraph]:
        coordinate = jnp.asarray(positions, dtype=self.precision.coordinate_dtype)
        if self.configuration.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER:
            coordinate = eqx.error_if(
                coordinate,
                jnp.any(batch.atom_mask & ~batch.element_mask),
                "Atomic-number PaiNN cannot evaluate non-element particles.",
            )
        graph = realize_atomistic_graph(
            batch,
            execution,
            cutoff=self.configuration.cutoff,
            positions=coordinate,
        )
        species = (
            batch.atomic_numbers
            if self.configuration.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER
            else batch.atom_type_ids
        )
        energy, atom_energy = self.graph_energy(
            species,
            batch.atom_mask,
            batch.atom_cases,
            batch.case_count,
            batch.atom_capacity,
            graph,
        )
        return energy, atom_energy, graph

    def energy(
        self,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        positions: Array | None = None,
    ) -> Array:
        """Evaluate typed-scale total energies, failing closed on graph overflow."""

        self._validate_batch(batch)
        coordinate = batch.positions if positions is None else positions
        energy, _, graph = self._energy_unchecked(batch, coordinate, execution)
        return graph.require_success(energy)

    def __call__(
        self,
        structure: AtomicStructure | AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
    ) -> Array:
        if isinstance(structure, AtomicStructure):
            batch = AtomisticBatch.from_structure(structure)
            return self.energy(batch, execution)[0]
        if isinstance(structure, AtomisticBatch):
            return self.energy(structure, execution)
        raise TypeError("PaiNNPotential expects AtomicStructure or AtomisticBatch.")


__all__ = ["PaiNNPotential"]
