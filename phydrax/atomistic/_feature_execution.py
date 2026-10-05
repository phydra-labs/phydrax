#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local layer schedule of learned atomistic message passing.

This module binds a layered learned potential to one owner region of the
canonical `DistributedHaloPlan`. It owns only the domain schedule: which node
payload crosses the halo before each interaction, the order of interactions,
and the reverse layer order. Halo gathers and exactly-once reverse returns are
the halo owner's `local_gather` and `local_transpose`; edge execution is the
model's own prepared streamed relation.

Each owner holds ``R = local_capacity`` receiver rows and ``C`` source columns
(its owned rows followed by deduplicated halo columns). Owner-local edges carry
``(receiver row, source column, n)`` with displacement
``d = x_receiver - x_source + n @ H`` relative to stored coordinates, so a
periodic image alias reuses the physical atom's column and repeated images of
one source keep their multiplicity as distinct edges.

Forward execution prepares sources on owned rows, gathers them into columns,
and updates owned receivers, one interaction after another. The explicit reverse
replays each interaction in reverse order from its retained inter-layer node
state, applies the model's ordinary-JAX local VJP, and returns source-column
cotangents to their owners once through `local_transpose`. Geometry cotangents
return to receiver rows directly and to source rows through the same transpose.
Everything is ordinary differentiable JAX, so forces may themselves be
differentiated (parameter gradients of force losses, coordinate actions).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree

from .._trainable import combine_parameters, partition_parameters
from ..discretization.spatial._distributed_relations import DistributedHaloPlan
from ..ein import contract
from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationEvidence


type NodeState = PyTree[Array]
"""PyTree of node arrays whose leaves all lead with the node axis."""


class AtomisticLayerUpdate(Protocol):
    """Updated receiver states, per-receiver layer energies, streamed evidence."""

    @property
    def features(self) -> NodeState: ...

    @property
    def energy(self) -> Array: ...

    @property
    def evidence(self) -> StreamedRelationEvidence: ...


class AtomisticLayeredModel(Protocol):
    """Interaction-layer contract of a learned message-passing potential.

    Every method is pure ordinary JAX, padding safe, and makes no global-node
    assumption: receivers ``R`` and source columns ``C`` may differ, and the
    relation may be rectangular. ``index`` is a static interaction index.
    """

    @property
    def interaction_count(self) -> int: ...

    def init_features(self, species: Array, atom_mask: Array, /) -> NodeState: ...

    def prepare_sources(
        self, index: int, features: NodeState, species: Array, atom_mask: Array, /
    ) -> NodeState: ...

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: NodeState,
        features: NodeState,
        species: Array,
        atom_mask: Array,
        /,
        *,
        edge_active: Array | None = None,
    ) -> AtomisticLayerUpdate: ...

    def atom_energies(
        self, species: Array, atom_mask: Array, layer_energies: Array, /
    ) -> Array: ...


class OwnerLayerTopology(NamedTuple):
    """One owner's prepared edges inside a mapped owner region."""

    relation: PreparedStreamedRelation
    receivers: Array
    columns: Array
    shifts: Array
    valid: Array
    send_slots: Array
    send_valid: Array


class OwnerEdgeGeometry(NamedTuple):
    """Owner-local edge displacements and their cutoff activity."""

    displacements: Array
    vectors: Array
    active: Array


class OwnerLayerGradients(NamedTuple):
    """Owner contributions of one explicit layered reverse sweep.

    ``gradient`` is ``dE/dx`` of the owned rows after every halo contribution
    has returned exactly once. ``strain_gradient`` is this owner's partial of
    ``dE/d strain`` from its own edges; the shared cell cotangent is the
    owner-ordered sum of these partials. ``parameter_gradient`` is this owner's
    partial of the PARAMETER-lane gradient, or ``None`` when not requested.
    ``relation_successful`` conjoins every streamed layer evidence of the
    forward and replayed reverse sweeps.
    """

    atom_energies: Array
    gradient: Array
    strain_gradient: Array
    parameter_gradient: PyTree[Array] | None
    relation_successful: Array


def _inexact(value: Array) -> bool:
    return jnp.issubdtype(value.dtype, jnp.inexact)


def _zero_cotangent(primal: Array) -> Array | np.ndarray:
    if _inexact(primal):
        return jnp.zeros_like(primal)
    return np.zeros(primal.shape, dtype=jax.dtypes.float0)


def _add_cotangents(first: NodeState, second: NodeState) -> NodeState:
    def add(left: Array, right: Array) -> Array:
        if left.dtype == jax.dtypes.float0:
            return left
        return left + right

    return jax.tree.map(add, first, second)


def _gather_columns(
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    owned: NodeState,
    message_capacity_bytes: int,
) -> NodeState:
    """Gather owned node payload into columns under the declared message bound.

    One interaction's halo message of one owner is ``owner_count *
    halo_capacity`` padded rows of the payload; the static byte count is
    refused at trace time when it exceeds the declared bound.
    """
    rows = halo.ownership.owner_count * halo.halo_capacity
    message = sum(
        rows * (leaf.size // leaf.shape[0]) * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(owned)
    )
    if message > message_capacity_bytes:
        raise ValueError(
            f"One interaction's halo message needs {message} bytes, exceeding "
            f"message_capacity_bytes={message_capacity_bytes}."
        )
    return jax.tree.map(
        lambda leaf: halo.local_gather(leaf, topology.send_slots, topology.send_valid),
        owned,
    )


def _return_columns(
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    cotangents: NodeState,
    owned: NodeState,
) -> NodeState:
    """Return column cotangents to owned rows exactly once (halo transpose)."""

    def transpose(cotangent: Array, primal: Array) -> Array | np.ndarray:
        if not _inexact(primal):
            return _zero_cotangent(primal)
        return halo.local_transpose(cotangent, topology.send_slots, topology.send_valid)

    return jax.tree.map(transpose, cotangents, owned)


def owner_edge_geometry(
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    positions: Array,
    cell_vectors: Array,
    cutoff: float,
) -> OwnerEdgeGeometry:
    """Gather source coordinates and form ``d = x_r - x_s + n @ H`` per edge.

    Padded edges evaluate a fixed nonzero in-domain sentinel displacement so
    directional and radial functions stay finite; their activity is false and
    their recorded displacement is zero, so padding rows never reach the
    strain partial.
    """
    columns = halo.local_gather(positions, topology.send_slots, topology.send_valid)
    translations = contract(
        "ea,ad->ed", topology.shifts.astype(positions.dtype), cell_vectors
    )
    raw = positions[topology.receivers] - columns[topology.columns] + translations
    sentinel = (
        jnp.zeros((positions.shape[-1],), dtype=positions.dtype).at[0].set(0.5 * cutoff)
    )
    valid = topology.valid[:, None]
    vectors = jnp.where(valid, raw, sentinel)
    squared = jnp.sum(vectors * vectors, axis=-1)
    active = topology.valid & (squared < cutoff * cutoff)
    return OwnerEdgeGeometry(jnp.where(valid, raw, 0.0), vectors, active)


class _ForwardSweep(NamedTuple):
    atom_energies: Array
    boundaries: tuple[NodeState, ...]
    layer_energies: Array
    relation_successful: Array


def _forward(
    model: AtomisticLayeredModel,
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    geometry: OwnerEdgeGeometry,
    species: Array,
    atom_mask: Array,
    message_capacity_bytes: int,
) -> _ForwardSweep:
    """Run every interaction; retain only inter-layer owned node states."""
    features = model.init_features(species, atom_mask)
    boundaries: list[NodeState] = []
    energies: list[Array] = []
    successful = jnp.asarray(True)
    for index in range(model.interaction_count):
        boundaries.append(features)
        sources = model.prepare_sources(index, features, species, atom_mask)
        columns = _gather_columns(halo, topology, sources, message_capacity_bytes)
        update = model.update_receivers(
            index,
            topology.relation,
            geometry.vectors,
            columns,
            features,
            species,
            atom_mask,
            edge_active=geometry.active,
        )
        features = update.features
        energies.append(update.energy)
        successful = successful & update.evidence.successful
    layer_energies = jnp.stack(energies)
    return _ForwardSweep(
        model.atom_energies(species, atom_mask, layer_energies),
        tuple(boundaries),
        layer_energies,
        successful,
    )


def owner_atom_energies(
    model: AtomisticLayeredModel,
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    positions: Array,
    species: Array,
    atom_mask: Array,
    cell_vectors: Array,
    cutoff: float,
    *,
    message_capacity_bytes: int,
) -> tuple[Array, Array]:
    """Owned atom energies of one owner region and its streamed success."""
    geometry = owner_edge_geometry(halo, topology, positions, cell_vectors, cutoff)
    sweep = _forward(
        model, halo, topology, geometry, species, atom_mask, message_capacity_bytes
    )
    return sweep.atom_energies, sweep.relation_successful


def owner_layer_gradients(
    model: AtomisticLayeredModel,
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    positions: Array,
    species: Array,
    atom_mask: Array,
    cell_vectors: Array,
    cutoff: float,
    *,
    message_capacity_bytes: int,
    parameters: bool,
) -> OwnerLayerGradients:
    """Owned energies plus the explicit reverse layer sweep of one owner.

    The seed is ``dE/d atom_energy = 1`` on every active owned atom, so the
    sum over owners of each partial is the gradient of the global energy.
    """
    lanes = partition_parameters(model)
    parameter_lane = lanes[0]

    def bind(lane: PyTree[Array]) -> AtomisticLayeredModel:
        return combine_parameters(lane, lanes[1], lanes[2])

    geometry = owner_edge_geometry(halo, topology, positions, cell_vectors, cutoff)
    sweep = _forward(
        model, halo, topology, geometry, species, atom_mask, message_capacity_bytes
    )
    successful = sweep.relation_successful
    seed = atom_mask.astype(sweep.atom_energies.dtype)
    _, readout_vjp = jax.vjp(
        lambda lane, values: bind(lane).atom_energies(species, atom_mask, values),
        parameter_lane,
        sweep.layer_energies,
    )
    parameter_cotangent, layer_cotangent = readout_vjp(seed)
    feature_cotangent: NodeState | None = None
    vector_cotangent = jnp.zeros_like(geometry.vectors)
    for index in reversed(range(model.interaction_count)):
        features = sweep.boundaries[index]
        sources, source_vjp = jax.vjp(
            _source_function(bind, index, species, atom_mask), parameter_lane, features
        )
        columns = _gather_columns(halo, topology, sources, message_capacity_bytes)
        outputs, receiver_vjp, replayed = jax.vjp(
            _receiver_function(bind, index, topology, geometry, species, atom_mask),
            parameter_lane,
            geometry.vectors,
            columns,
            features,
            has_aux=True,
        )
        successful = successful & replayed
        output_cotangent = (
            jax.tree.map(_zero_cotangent, outputs[0])
            if feature_cotangent is None
            else feature_cotangent,
            layer_cotangent[index].astype(outputs[1].dtype),
        )
        (
            receiver_parameters,
            vectors_cotangent,
            columns_cotangent,
            direct_cotangent,
        ) = receiver_vjp(output_cotangent)
        owned_cotangent = _return_columns(halo, topology, columns_cotangent, sources)
        source_parameters, source_features = source_vjp(owned_cotangent)
        feature_cotangent = _add_cotangents(direct_cotangent, source_features)
        parameter_cotangent = jax.tree.map(
            jnp.add,
            parameter_cotangent,
            jax.tree.map(jnp.add, receiver_parameters, source_parameters),
        )
        vector_cotangent = vector_cotangent + vectors_cotangent
    _, initial_vjp = jax.vjp(
        lambda lane: bind(lane).init_features(species, atom_mask), parameter_lane
    )
    if feature_cotangent is not None:
        (initial_parameters,) = initial_vjp(feature_cotangent)
        parameter_cotangent = jax.tree.map(
            jnp.add, parameter_cotangent, initial_parameters
        )
    edge_cotangent = jnp.where(topology.valid[:, None], vector_cotangent, 0.0)
    gradient = _edge_position_gradient(halo, topology, edge_cotangent, positions)
    strain_gradient = contract("ea,eb->ab", edge_cotangent, geometry.displacements)
    return OwnerLayerGradients(
        atom_energies=sweep.atom_energies,
        gradient=jnp.where(atom_mask[:, None], gradient, 0.0),
        strain_gradient=strain_gradient,
        parameter_gradient=parameter_cotangent if parameters else None,
        relation_successful=successful,
    )


def _source_function(
    bind: Callable[[PyTree[Array]], AtomisticLayeredModel],
    index: int,
    species: Array,
    atom_mask: Array,
) -> Callable[[PyTree[Array], NodeState], NodeState]:
    def source(lane: PyTree[Array], features: NodeState) -> NodeState:
        return bind(lane).prepare_sources(index, features, species, atom_mask)

    return source


def _receiver_function(
    bind: Callable[[PyTree[Array]], AtomisticLayeredModel],
    index: int,
    topology: OwnerLayerTopology,
    geometry: OwnerEdgeGeometry,
    species: Array,
    atom_mask: Array,
) -> Callable[
    [PyTree[Array], Array, NodeState, NodeState], tuple[tuple[NodeState, Array], Array]
]:
    """Differentiable receiver update; streamed success is auxiliary evidence."""

    def receiver(
        lane: PyTree[Array], vectors: Array, columns: NodeState, features: NodeState
    ) -> tuple[tuple[NodeState, Array], Array]:
        update = bind(lane).update_receivers(
            index,
            topology.relation,
            vectors,
            columns,
            features,
            species,
            atom_mask,
            edge_active=geometry.active,
        )
        return (update.features, update.energy), update.evidence.successful

    return receiver


def _edge_position_gradient(
    halo: DistributedHaloPlan,
    topology: OwnerLayerTopology,
    edge_cotangent: Array,
    positions: Array,
) -> Array:
    """Route ``dE/dd`` to receivers and, through the halo transpose, sources.

    The receiver and column accumulations are incidental single indexed
    updates over this owner's fixed edge capacity; the cross-owner return is
    the halo owner's exactly-once transpose.
    """
    receivers = jnp.zeros_like(positions).at[topology.receivers].add(edge_cotangent)
    column_count = halo.column_count
    columns = (
        jnp.zeros((column_count,) + positions.shape[1:], dtype=positions.dtype)
        .at[topology.columns]
        .add(-edge_cotangent)
    )
    return receivers + halo.local_transpose(
        columns, topology.send_slots, topology.send_valid
    )


__all__ = [
    "AtomisticLayeredModel",
    "AtomisticLayerUpdate",
    "OwnerEdgeGeometry",
    "OwnerLayerGradients",
    "OwnerLayerTopology",
    "owner_atom_energies",
    "owner_edge_geometry",
    "owner_layer_gradients",
]
