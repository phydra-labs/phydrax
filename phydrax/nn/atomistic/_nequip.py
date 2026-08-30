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
from opt_einsum import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._graph import AtomisticGraph, realize_atomistic_graph
from ...atomistic._types import (
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomicStructure,
)
from ..layers import Linear
from ..operator.layers import O3TensorProduct, O3TensorProductPlan, o3_gated_activation
from ..operator.representations import O3Features, O3Representation
from ..parameters import IdentityTransform
from ._state import AbstractAtomisticPotential, initialize_atomistic_potential_identity




class _NequIPConfiguration(StrictModule, NonTrainableState):
    radial_frequencies: Array
    hidden_representation: O3Representation
    edge_representation: O3Representation
    tensor_product_plan_ids: tuple[str, ...] = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    maximum_dense_atoms: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    maximum_atomic_number: int = eqx.field(static=True)
    maximum_tensor_product_parameters: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)


class _SpeciesSelfConnection(StrictModule):
    representation: O3Representation
    weights: tuple[Array, ...]

    def __init__(
        self,
        representation: O3Representation,
        maximum_atomic_number: int,
        /,
        *,
        dtype: jnp.dtype,
        key: Key[Array, ""],
    ):
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
                (maximum_atomic_number + 1, count, count),
                dtype=dtype,
            )
            weights.append(value.at[0].set(0.0))
        self.representation = representation
        self.weights = tuple(weights)

    def __call__(self, values: Array, atomic_numbers: Array, /) -> Array:
        features = self.representation.split(values)
        selected = tuple(weight[atomic_numbers] for weight in self.weights)
        return self.representation.join(
            O3Features(
                scalars=contract("noi,ni->no", selected[0], features.scalars),
                pseudoscalars=contract(
                    "noi,ni->no", selected[1], features.pseudoscalars
                ),
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
        maximum_atomic_number: int,
        maximum_tensor_product_parameters: int,
        /,
        *,
        dtype: jnp.dtype,
        key: Key[Array, ""],
    ):
        plan = O3TensorProductPlan(
            representation,
            edge_representation,
            representation,
            maximum_parameters=maximum_tensor_product_parameters,
        )
        radial_in_key, radial_out_key, self_key = jr.split(key, 3)
        self.tensor_product = O3TensorProduct(
            plan, internal_weights=False, dtype=dtype
        )
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
            maximum_atomic_number,
            dtype=dtype,
            key=self_key,
        )
        self.representation = representation

    def __call__(
        self,
        values: Array,
        edge_features: Array,
        atomic_numbers: Array,
        graph: AtomisticGraph,
        radial: Array,
        cutoff_envelope: Array,
        node_mask: Array,
        /,
    ) -> Array:
        ir = graph.graph
        if ir.senders is None or ir.receivers is None or ir.edge_mask is None:
            raise ValueError("NequIP requires an explicit masked edge relation.")
        edge_mask = ir.edge_mask.astype(values.dtype)
        path_weights = self.radial_out(self.radial_in(radial))
        path_weights = path_weights * cutoff_envelope[:, None] * edge_mask[:, None]
        messages = self.tensor_product(
            values[ir.senders], edge_features, path_weights
        )
        messages = messages * edge_mask[:, None]
        aggregate = jnp.zeros_like(values).at[ir.receivers].add(messages)
        connected = self.self_connection(values, atomic_numbers) + aggregate
        activated = o3_gated_activation(connected, self.representation)
        return activated * node_mask[:, None]




class NequIPPotential(AbstractAtomisticPotential):
    """Low-degree finite nonperiodic NequIP scalar energy potential.

    This is a Cartesian O(3) implementation with degrees zero through two. It
    consumes the same fixed candidate ``AtomisticBatch`` topology as PaiNN;
    geometry only changes differentiable edge payloads and smooth weights inside
    conservative force derivatives.
    """

    embedding: Array
    interactions: tuple[_NequIPInteraction, ...]
    readout_hidden: Linear
    readout_energy: Linear
    configuration: _NequIPConfiguration
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
        maximum_neighbors: int,
        maximum_dense_atoms: int,
        feature_count: int = 32,
        interaction_count: int = 3,
        radial_basis_count: int = 20,
        maximum_atomic_number: int = 118,
        maximum_tensor_product_parameters: int = 10_000_000,
        precision: AtomisticPrecisionPolicy | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        cutoff_value = float(cutoff)
        neighbor_limit = int(maximum_neighbors)
        dense_limit = int(maximum_dense_atoms)
        features = int(feature_count)
        interactions = int(interaction_count)
        radial_count = int(radial_basis_count)
        maximum_z = int(maximum_atomic_number)
        tensor_product_limit = int(maximum_tensor_product_parameters)
        if not math.isfinite(cutoff_value) or cutoff_value <= 0.0:
            raise ValueError("cutoff must be finite and positive.")
        if neighbor_limit < 0 or dense_limit <= 0:
            raise ValueError(
                "Neighbor capacity must be non-negative and dense guard positive."
            )
        if features <= 0 or interactions <= 0 or radial_count <= 0:
            raise ValueError(
                "NequIP feature, interaction, and radial counts must be positive."
            )
        if maximum_z <= 0:
            raise ValueError("maximum_atomic_number must be positive.")
        if tensor_product_limit < 0:
            raise ValueError(
                "maximum_tensor_product_parameters must be non-negative."
            )
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
        embedding = embedding.at[0].set(0.0)
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
            interaction.tensor_product.plan.plan_id
            for interaction in interaction_modules
        )
        self.configuration = _NequIPConfiguration(
            radial_frequencies=jnp.arange(1, radial_count + 1, dtype=compute_dtype),
            hidden_representation=hidden_representation,
            edge_representation=edge_representation,
            tensor_product_plan_ids=plan_ids,
            cutoff=cutoff_value,
            maximum_neighbors=neighbor_limit,
            maximum_dense_atoms=dense_limit,
            feature_count=features,
            interaction_count=interactions,
            radial_basis_count=radial_count,
            maximum_atomic_number=maximum_z,
            maximum_tensor_product_parameters=tensor_product_limit,
            maximum_degree=2,
        )
        self.scale = scale
        self.precision = precision_
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "nequip-architecture",
                "scope": "finite-nonperiodic-degree-at-most-two",
                "scale": scale.scale_id,
                "precision": precision_.policy_id,
                "cutoff": cutoff_value,
                "maximum_neighbors": neighbor_limit,
                "maximum_dense_atoms": dense_limit,
                "feature_count": features,
                "interaction_count": interactions,
                "radial_basis_count": radial_count,
                "maximum_atomic_number": maximum_z,
                "maximum_tensor_product_parameters": tensor_product_limit,
                "tensor_product_plans": plan_ids,
            }
        )
        self.method_id = "negative-position-gradient-of-total-nequip-energy"
        (
            self.parameter_state_id,
            self.potential_id,
        ) = initialize_atomistic_potential_identity(self)

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
            raise ValueError("Potential and structure must share one exact scale contract.")
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the NequIP precision contract."
            )
        if batch.has_periodic_metadata:
            raise ValueError(
                "NequIPPotential supports finite nonperiodic molecules only and rejects "
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
        return basis * envelope[:, None], envelope

    def _edge_features(self, direction: Array, edge_mask: Array, /) -> Array:
        dtype = jnp.dtype(self.precision.compute_dtype)
        unit = jnp.asarray(direction, dtype=dtype)
        identity = jnp.eye(3, dtype=dtype)
        outer = contract("ei,ej->eij", unit, unit)
        tensor = jnp.sqrt(jnp.asarray(1.5, dtype=dtype)) * (
            outer - identity[None, :, :] / 3.0
        )
        edge_count = int(unit.shape[0])
        empty_scalar = jnp.zeros((edge_count, 0), dtype=dtype)
        empty_vector = jnp.zeros((edge_count, 0, 3), dtype=dtype)
        empty_tensor = jnp.zeros((edge_count, 0, 3, 3), dtype=dtype)
        packed = self.configuration.edge_representation.join(
            O3Features(
                scalars=jnp.ones((edge_count, 1), dtype=dtype),
                pseudoscalars=empty_scalar,
                vectors=unit[:, None, :],
                pseudovectors=empty_vector,
                tensors=tensor[:, None, :, :],
                pseudotensors=empty_tensor,
            )
        )
        return packed * edge_mask.astype(dtype)[:, None]

    def _energy_unchecked(
        self,
        batch: AtomisticBatch,
        positions: Array,
        /,
    ) -> tuple[Array, Array, AtomisticGraph]:
        coordinate = jnp.asarray(positions, dtype=self.precision.coordinate_dtype)
        if coordinate.shape != batch.positions.shape:
            raise ValueError("positions must have the batch position shape.")
        coordinate = jnp.where(batch.atom_mask[:, :, None], coordinate, 0.0)
        graph = realize_atomistic_graph(
            batch,
            cutoff=self.configuration.cutoff,
            maximum_neighbors=self.configuration.maximum_neighbors,
            maximum_dense_atoms=self.configuration.maximum_dense_atoms,
            positions=coordinate,
        )
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("NequIP requires explicit edge masks.")
        numbers = batch.atomic_numbers.reshape((-1,))
        numbers = eqx.error_if(
            numbers,
            jnp.any(numbers > self.configuration.maximum_atomic_number),
            "Atomic number exceeds NequIPPotential.maximum_atomic_number.",
        )
        node_mask = batch.atom_mask.reshape((-1,)).astype(self.precision.compute_dtype)
        node_count = int(numbers.shape[0])
        packed_size = self.configuration.hidden_representation.packed_size
        values = jnp.zeros(
            (node_count, packed_size), dtype=self.precision.compute_dtype
        )
        scalar = self.embedding[numbers].astype(self.precision.compute_dtype)
        values = values.at[:, : self.configuration.feature_count].set(scalar)
        values = values * node_mask[:, None]
        distance = jnp.where(
            ir.edge_mask, jnp.asarray(ir.edges["distance"])[:, 0], 0.0
        )
        direction = jnp.where(
            ir.edge_mask[:, None], jnp.asarray(ir.edges["direction"]), 0.0
        )
        radial, cutoff_envelope = self._radial_basis(distance)
        edge_features = self._edge_features(direction, ir.edge_mask)
        for interaction in self.interactions:
            values = interaction(
                values,
                edge_features,
                numbers,
                graph,
                radial,
                cutoff_envelope,
                node_mask,
            )
        invariant_scalar = self.configuration.hidden_representation.split(values).scalars
        atom_energy = self.readout_energy(self.readout_hidden(invariant_scalar))
        atom_energy = atom_energy * node_mask.astype(atom_energy.dtype)
        total_energy = jnp.zeros(
            (batch.case_count,), dtype=self.precision.reduction_dtype
        ).at[batch.atom_cases].add(atom_energy.astype(self.precision.reduction_dtype))
        return (
            total_energy.astype(self.precision.output_dtype),
            atom_energy.reshape((batch.case_count, batch.atom_capacity)).astype(
                self.precision.output_dtype
            ),
            graph,
        )

    def energy(
        self,
        batch: AtomisticBatch,
        /,
        *,
        positions: Array | None = None,
    ) -> Array:
        """Evaluate typed-scale total energies, failing closed on graph overflow."""

        self._validate_batch(batch)
        coordinate = batch.positions if positions is None else positions
        energy, _, graph = self._energy_unchecked(batch, coordinate)
        return graph.require_success(energy)

    def __call__(self, structure: AtomicStructure | AtomisticBatch, /) -> Array:
        if isinstance(structure, AtomicStructure):
            batch = AtomisticBatch.from_structure(structure)
            return self.energy(batch)[0]
        if isinstance(structure, AtomisticBatch):
            return self.energy(structure)
        raise TypeError("NequIPPotential expects AtomicStructure or AtomisticBatch.")


__all__ = ["NequIPPotential"]
