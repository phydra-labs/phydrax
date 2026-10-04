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
from jax import Array

from phydrax.ein import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._graph import (
    AtomisticGraph,
    AtomisticGraphExecutionPlan,
    AtomisticGraphTopology,
    realize_atomistic_graph,
)
from ...atomistic._potential import (
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
    AtomisticSpeciesKind,
)
from ...atomistic._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
)
from ...sparse._streamed import PreparedStreamedRelation, StreamedPayloadSpec
from ...typing import checked, PRNGKey
from ..layers import Linear
from ..operator.layers import o3_gated_activation, O3TensorProduct, O3TensorProductPlan
from ..operator.representations import O3Features, O3Representation
from ..parameters import IdentityTransform


class _NequIPConfiguration(StrictModule, NonTrainableState):
    radial_frequencies: Array
    hidden_representation: O3Representation
    edge_representation: O3Representation
    tensor_product_plan_ids: tuple[str, ...] = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    maximum_species_id: int = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    maximum_tensor_product_parameters: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)


class _SpeciesSelfConnection(StrictModule):
    representation: O3Representation
    weights: tuple[Array, ...]

    def __init__(
        self,
        representation: O3Representation,
        maximum_species_id: int,
        /,
        *,
        dtype: jnp.dtype,
        key: PRNGKey,
    ) -> None:
        counts = (
            representation.scalars,
            representation.pseudoscalars,
            representation.vectors,
            representation.pseudovectors,
            representation.tensors,
            representation.pseudotensors,
        )
        keys = jr.split(key, 6)
        weights = []
        for count, block_key in zip(counts, keys, strict=True):
            scale = 1.0 / math.sqrt(float(count))
            value = scale * jr.normal(
                block_key,
                (maximum_species_id + 1, count, count),
                dtype=dtype,
            )
            weights.append(value)
        self.representation = representation
        self.weights = tuple(weights)

    def __call__(self, values: Array, species_ids: Array, /) -> Array:
        features = self.representation.split(values)
        selected = tuple(weight[species_ids] for weight in self.weights)
        return self.representation.join(
            O3Features(
                scalars=contract("noi,ni->no", selected[0], features.scalars),
                pseudoscalars=contract("noi,ni->no", selected[1], features.pseudoscalars),
                vectors=contract("noi,nic->noc", selected[2], features.vectors),
                pseudovectors=contract(
                    "noi,nic->noc", selected[3], features.pseudovectors
                ),
                tensors=contract("noi,nicd->nocd", selected[4], features.tensors),
                pseudotensors=contract(
                    "noi,nicd->nocd", selected[5], features.pseudotensors
                ),
            )
        )


class _NequIPInteraction(StrictModule):
    tensor_product: O3TensorProduct
    radial_in: Linear
    radial_out: Linear
    self_connection: _SpeciesSelfConnection
    representation: O3Representation

    def __init__(
        self,
        representation: O3Representation,
        edge_representation: O3Representation,
        radial_basis_count: int,
        maximum_species_id: int,
        maximum_tensor_product_parameters: int,
        /,
        *,
        dtype: jnp.dtype,
        key: PRNGKey,
    ) -> None:
        plan = O3TensorProductPlan(
            representation,
            edge_representation,
            representation,
            maximum_parameters=maximum_tensor_product_parameters,
        )
        radial_in_key, radial_out_key, self_key = jr.split(key, 3)
        self.tensor_product = O3TensorProduct(plan, internal_weights=False, dtype=dtype)
        self.radial_in = Linear(
            in_size=radial_basis_count,
            out_size=representation.scalars,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=IdentityTransform(),
            key=radial_in_key,
        )
        self.radial_out = Linear(
            in_size=representation.scalars,
            out_size=plan.parameter_count,
            rwf=False,
            weight_transform=IdentityTransform(),
            key=radial_out_key,
        )
        self.self_connection = _SpeciesSelfConnection(
            representation,
            maximum_species_id,
            dtype=dtype,
            key=self_key,
        )
        self.representation = representation

    def __call__(
        self,
        potential: NequIPPotential,
        values: Array,
        species_ids: Array,
        edges: PreparedStreamedRelation,
        edge_data: tuple[Array, Array],
        edge_active: Array,
        node_mask: Array,
        /,
    ) -> Array:
        node = jax.ShapeDtypeStruct(values.shape[1:], values.dtype)
        activated = edges.evaluate(
            StreamedPayloadSpec(message=node, output=node),
            _nequip_message,
            _nequip_update,
            (potential, self),
            values,
            (values, species_ids),
            edge_data,
            edge_active=edge_active,
        ).receiver_outputs
        return activated * node_mask[:, None]


def _nequip_message(
    parameters: tuple[NequIPPotential, _NequIPInteraction],
    sender: Array,
    receiver: tuple[Array, Array],
    edge: tuple[Array, Array],
    /,
) -> Array:
    """Radially weighted tensor-product message of one directed edge."""
    potential, interaction = parameters
    del receiver
    distance, direction = edge
    radial, cutoff_envelope = potential._radial_basis(distance)
    path_weights = interaction.radial_out(interaction.radial_in(radial))
    path_weights = path_weights * cutoff_envelope
    return interaction.tensor_product(
        sender, potential._edge_features(direction[None])[0], path_weights
    )


def _nequip_update(
    parameters: tuple[NequIPPotential, _NequIPInteraction],
    receiver: tuple[Array, Array],
    aggregate: Array,
    /,
) -> Array:
    """Species self-connection plus message sum, then gated activation, per receiver."""
    _potential, interaction = parameters
    values, species_id = receiver
    connected = interaction.self_connection(values[None], species_id[None])[0] + aggregate
    return o3_gated_activation(connected[None], interaction.representation)[0]


class NequIPPotential(AbstractAtomisticPotential):
    """Low-degree NequIP scalar energy potential on finite or periodic-image graphs.

    This is a Cartesian O(3) implementation with degrees zero through two. It
    consumes the same prepared graph topology as PaiNN: finite dense all-pairs
    routes, or explicit integer periodic image routes for orthorhombic,
    triclinic and partially periodic cells. Geometry and cell vectors only
    change differentiable edge payloads and smooth weights, so forces and the
    first strain derivative (stress) are exact; the cosine envelope is only C1
    at the cutoff, so coordinate second derivatives jump where an edge crosses
    it. Every interaction evaluates each directed edge message once and each
    receiver update once on the streamed schedule the graph topology prepared
    for its execution plan.
    """

    embedding: Array
    interactions: tuple[_NequIPInteraction, ...]
    readout_hidden: Linear
    readout_energy: Linear
    configuration: _NequIPConfiguration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        /,
        *,
        cutoff: float,
        feature_count: int = 32,
        interaction_count: int = 3,
        radial_basis_count: int = 20,
        maximum_species_id: int = 118,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
        maximum_tensor_product_parameters: int = 10_000_000,
        precision: AtomisticPrecisionPolicy | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        cutoff_value = float(cutoff)
        features = int(feature_count)
        interactions = int(interaction_count)
        radial_count = int(radial_basis_count)
        maximum_z = int(maximum_species_id)
        tensor_product_limit = int(maximum_tensor_product_parameters)
        if not math.isfinite(cutoff_value) or cutoff_value <= 0.0:
            raise ValueError("cutoff must be finite and positive.")
        if features <= 0 or interactions <= 0 or radial_count <= 0:
            raise ValueError(
                "NequIP feature, interaction, and radial counts must be positive."
            )
        if maximum_z <= 0:
            raise ValueError("maximum_species_id must be positive.")
        if not isinstance(species_kind, AtomisticSpeciesKind):
            raise TypeError("species_kind must be AtomisticSpeciesKind.")
        if tensor_product_limit < 0:
            raise ValueError("maximum_tensor_product_parameters must be non-negative.")
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
        compute_dtype = jnp.dtype(precision_.compute_dtype)
        hidden_representation = O3Representation(
            scalars=features,
            pseudoscalars=features,
            vectors=features,
            pseudovectors=features,
            tensors=features,
            pseudotensors=features,
        )
        edge_representation = O3Representation(scalars=1, vectors=1, tensors=1)
        keys = jr.split(key, interactions + 3)
        embedding = jr.normal(
            keys[0], (maximum_z + 1, features), dtype=compute_dtype
        ) / jnp.sqrt(jnp.asarray(features, dtype=compute_dtype))
        interaction_modules = tuple(
            _NequIPInteraction(
                hidden_representation,
                edge_representation,
                radial_count,
                maximum_z,
                tensor_product_limit,
                dtype=compute_dtype,
                key=keys[index + 1],
            )
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

        def cast_inexact(value: Any) -> Any:
            if eqx.is_inexact_array(value):
                return value.astype(compute_dtype)
            return value

        self.embedding = embedding
        self.interactions = jax.tree_util.tree_map(cast_inexact, interaction_modules)
        self.readout_hidden = jax.tree_util.tree_map(cast_inexact, readout_hidden)
        self.readout_energy = jax.tree_util.tree_map(cast_inexact, readout_energy)
        plan_ids = tuple(
            interaction.tensor_product.plan.plan_id for interaction in interaction_modules
        )
        self.configuration = _NequIPConfiguration(
            radial_frequencies=jnp.arange(1, radial_count + 1, dtype=compute_dtype),
            hidden_representation=hidden_representation,
            edge_representation=edge_representation,
            tensor_product_plan_ids=plan_ids,
            cutoff=cutoff_value,
            feature_count=features,
            interaction_count=interactions,
            radial_basis_count=radial_count,
            maximum_species_id=maximum_z,
            species_kind=species_kind,
            maximum_tensor_product_parameters=tensor_product_limit,
            maximum_degree=2,
        )
        self.scale = scale
        self.precision = precision_
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "nequip-architecture",
                "scope": "cartesian-degree-at-most-two",
                "scale": scale.scale_id,
                "precision": precision_.policy_id,
                "cutoff": cutoff_value,
                "feature_count": features,
                "interaction_count": interactions,
                "radial_basis_count": radial_count,
                "maximum_species_id": maximum_z,
                "species_kind": species_kind.value,
                "maximum_tensor_product_parameters": tensor_product_limit,
                "tensor_product_plans": plan_ids,
            }
        )
        self.method_id = "negative-position-gradient-of-total-nequip-energy"

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        # The radius-local energy consumes explicit image routes; strain enters
        # only through cell-dependent image displacements (first derivative).
        return AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            species_kind=self.configuration.species_kind,
        )

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError(
                "Potential and structure must share one exact scale contract."
            )
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the NequIP precision contract."
            )

    def _radial_basis(self, distance: Array, /) -> tuple[Array, Array]:
        """Enveloped sinc radial basis and cosine envelope, broadcast over distances."""
        dtype = jnp.dtype(self.precision.compute_dtype)
        radius = jnp.asarray(distance, dtype=dtype)
        cutoff = jnp.asarray(self.configuration.cutoff, dtype=dtype)
        scaled = radius / cutoff
        frequencies = self.configuration.radial_frequencies
        basis = (jnp.pi * frequencies / cutoff) * jnp.sinc(
            scaled[..., None] * frequencies
        )
        envelope = jnp.where(
            scaled < 1.0,
            0.5 * (jnp.cos(jnp.pi * scaled) + 1.0),
            0.0,
        )
        return basis * envelope[..., None], envelope

    def _edge_features(self, direction: Array, /) -> Array:
        dtype = jnp.dtype(self.precision.compute_dtype)
        unit = jnp.asarray(direction, dtype=dtype)
        identity = jnp.eye(3, dtype=dtype)
        outer = contract("ei,ej->eij", unit, unit)
        tensor = jnp.sqrt(jnp.asarray(1.5, dtype=dtype)) * (
            outer - identity[None, :, :] / 3.0
        )
        edge_count = unit.shape[0]
        empty_scalar = jnp.zeros((edge_count, 0), dtype=dtype)
        empty_vector = jnp.zeros((edge_count, 0, 3), dtype=dtype)
        empty_tensor = jnp.zeros((edge_count, 0, 3, 3), dtype=dtype)
        return self.configuration.edge_representation.join(
            O3Features(
                scalars=jnp.ones((edge_count, 1), dtype=dtype),
                pseudoscalars=empty_scalar,
                vectors=unit[:, None, :],
                pseudovectors=empty_vector,
                tensors=tensor[:, None, :, :],
                pseudotensors=empty_tensor,
            )
        )

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
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("NequIP requires explicit edge masks.")
        numbers = jnp.asarray(species_ids).reshape((-1,))
        numbers = eqx.error_if(
            numbers,
            jnp.any(numbers > self.configuration.maximum_species_id),
            "Species ID exceeds NequIPPotential.maximum_species_id.",
        )
        node_mask = (
            jnp.asarray(atom_mask, dtype=jnp.bool_)
            .reshape((-1,))
            .astype(self.precision.compute_dtype)
        )
        node_count = numbers.shape[0]
        packed_size = self.configuration.hidden_representation.packed_size
        values = jnp.zeros((node_count, packed_size), dtype=self.precision.compute_dtype)
        scalar = self.embedding[numbers].astype(self.precision.compute_dtype)
        values = values.at[:, : self.configuration.feature_count].set(scalar)
        values = values * node_mask[:, None]
        # One prepared candidate schedule serves every interaction; the cutoff
        # mask is runtime route activity, never a schedule change.
        edges = graph.topology.streamed
        edge_data = (
            jnp.asarray(ir.edges["distance"])[:, 0],
            jnp.asarray(ir.edges["direction"]),
        )
        for interaction in self.interactions:
            values = interaction(
                self,
                values,
                numbers,
                edges,
                edge_data,
                ir.edge_mask,
                node_mask,
            )
        invariant_scalar = self.configuration.hidden_representation.split(values).scalars
        atom_energy = self.readout_energy(self.readout_hidden(invariant_scalar))
        atom_energy = atom_energy * node_mask.astype(atom_energy.dtype)
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
        *,
        topology: AtomisticGraphTopology | None = None,
        cell_vectors: Array | None = None,
    ) -> tuple[Array, Array, AtomisticGraph]:
        coordinate = jnp.asarray(positions, dtype=self.precision.coordinate_dtype)
        if self.configuration.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER:
            coordinate = eqx.error_if(
                coordinate,
                jnp.any(batch.atom_mask & ~batch.element_mask),
                "Atomic-number NequIP cannot evaluate non-element particles.",
            )
        if coordinate.shape != batch.positions.shape:
            raise ValueError("positions must have the batch position shape.")
        coordinate = jnp.where(batch.atom_mask[:, :, None], coordinate, 0.0)
        graph = realize_atomistic_graph(
            batch,
            execution,
            cutoff=self.configuration.cutoff,
            positions=coordinate,
            topology=topology,
            cell_vectors=cell_vectors,
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
        topology: AtomisticGraphTopology | None = None,
        cell_vectors: Array | None = None,
    ) -> Array:
        """Evaluate typed-scale total energies, failing closed on graph overflow.

        Periodic batches require ``topology`` from
        ``prepare_atomistic_graph_topology``, prepared on the host before any
        transformed call; ``cell_vectors`` default to the batch cells.
        """

        self._validate_batch(batch)
        coordinate = batch.positions if positions is None else positions
        energy, _, graph = self._energy_unchecked(
            batch, coordinate, execution, topology=topology, cell_vectors=cell_vectors
        )
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
        raise TypeError("NequIPPotential expects AtomicStructure or AtomisticBatch.")


__all__ = ["NequIPPotential"]
