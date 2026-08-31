#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh, PolygonalConnectivity
from ...graph import GraphIR
from ...sparse import EdgeRelation


def _host_integer_vector(name: str, value: Any, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a rank-1 integer array.")
    return array.astype(np.int32, copy=False)


def _constraint_mask(
    node_count: int,
    dimension: int,
    constrained_dofs: ArrayLike | None,
    fixed_nodes: Sequence[int] | ArrayLike | None,
    /,
) -> np.ndarray:
    if constrained_dofs is not None and fixed_nodes is not None:
        raise ValueError("Supply constrained_dofs or fixed_nodes, not both.")
    if constrained_dofs is not None:
        mask = np.asarray(constrained_dofs, dtype=bool)
        if mask.shape != (node_count, dimension):
            raise ValueError(
                "constrained_dofs must have shape "
                f"({node_count}, {dimension}); got {mask.shape}."
            )
        return mask
    mask = np.zeros((node_count, dimension), dtype=bool)
    if fixed_nodes is None:
        return mask
    indices = _host_integer_vector("fixed_nodes", fixed_nodes)
    if indices.size and (
        np.any(indices < 0)
        or np.any(indices >= node_count)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError("fixed_nodes must contain unique in-range node indices.")
    mask[indices, :] = True
    return mask


def _active_graph_masks(graph: GraphIR, /) -> tuple[np.ndarray, np.ndarray]:
    node_count = graph.num_nodes
    member_count = graph.num_edges
    graph_active = (
        np.ones((graph.num_graphs,), dtype=bool)
        if graph.graph_mask is None
        else np.asarray(graph.graph_mask, dtype=bool)
    )
    node_graph = np.repeat(
        np.arange(graph.num_graphs, dtype=np.int32), np.asarray(graph.n_node)
    )
    edge_graph = np.repeat(
        np.arange(graph.num_graphs, dtype=np.int32), np.asarray(graph.n_edge)
    )
    node_valid = graph_active[node_graph]
    member_valid = graph_active[edge_graph]
    if graph.node_mask is not None:
        node_valid &= np.asarray(graph.node_mask, dtype=bool)
    if graph.edge_mask is not None:
        member_valid &= np.asarray(graph.edge_mask, dtype=bool)
    if node_valid.shape != (node_count,) or member_valid.shape != (member_count,):
        raise ValueError("Graph masks do not match graph counts.")
    return node_valid, member_valid


def _components(
    node_count: int,
    senders: np.ndarray,
    receivers: np.ndarray,
    node_valid: np.ndarray,
    member_valid: np.ndarray,
    /,
) -> tuple[np.ndarray, int]:
    parent = np.arange(node_count, dtype=np.int32)

    def root(index: int) -> int:
        current = int(index)
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for sender, receiver, valid in zip(senders, receivers, member_valid, strict=True):
        if valid:
            union(int(sender), int(receiver))

    labels = np.full((node_count,), -1, dtype=np.int32)
    roots = sorted({root(index) for index in np.flatnonzero(node_valid)})
    root_to_label = {root_value: label for label, root_value in enumerate(roots)}
    for index in np.flatnonzero(node_valid):
        labels[index] = root_to_label[root(int(index))]
    return labels, len(roots)


def _equilibrium_routes(
    senders: np.ndarray,
    receivers: np.ndarray,
    member_valid: np.ndarray,
    full_to_free: np.ndarray,
    dimension: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    members: list[int] = []
    signs: list[float] = []

    def append(source: int, target: int, member: int, sign: float) -> None:
        sources.append(source)
        targets.append(target)
        members.append(member)
        signs.append(sign)

    for member, (sender, receiver, valid) in enumerate(
        zip(senders, receivers, member_valid, strict=True)
    ):
        if not valid:
            continue
        for coordinate in range(dimension):
            sender_full = int(sender) * dimension + coordinate
            receiver_full = int(receiver) * dimension + coordinate
            sender_free = int(full_to_free[sender_full])
            receiver_free = int(full_to_free[receiver_full])
            if sender_free >= 0:
                append(sender_free, sender_free, member, 1.0)
            if receiver_free >= 0:
                append(receiver_free, receiver_free, member, 1.0)
            if sender_free >= 0 and receiver_free >= 0:
                append(receiver_free, sender_free, member, -1.0)
                append(sender_free, receiver_free, member, -1.0)

    return (
        np.asarray(sources, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        np.asarray(members, dtype=np.int32),
        np.asarray(signs, dtype=float),
    )


class ForceDensityStructure(StrictModule, NonTrainableState):
    """Immutable member topology and positional-constraint layout."""

    graph: GraphIR
    dimension: int = eqx.field(static=True)
    senders: Array
    receivers: Array
    node_valid: Array
    member_valid: Array
    constrained_dofs: Array
    free_dof_indices: Array
    constrained_dof_indices: Array
    full_to_free: Array
    component_labels: Array
    component_count: int = eqx.field(static=True)
    equilibrium_relation: EdgeRelation
    route_members: Array
    route_signs: Array
    surface_connectivity: PolygonalConnectivity | None
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        graph: GraphIR,
        dimension: int,
        /,
        *,
        constrained_dofs: ArrayLike | None = None,
        fixed_nodes: Sequence[int] | ArrayLike | None = None,
        surface_connectivity: PolygonalConnectivity | None = None,
    ):
        if not isinstance(graph, GraphIR):
            raise TypeError("graph must be a GraphIR.")
        graph.validate()
        resolved_dimension = int(dimension)
        if resolved_dimension <= 0:
            raise ValueError("dimension must be positive.")
        if graph.senders is None or graph.receivers is None:
            raise ValueError("Force-density structures require explicit members.")

        node_count = graph.num_nodes
        member_count = graph.num_edges
        if node_count <= 0 or member_count <= 0:
            raise ValueError("Force-density structures require nodes and members.")
        senders = np.asarray(graph.senders, dtype=np.int32)
        receivers = np.asarray(graph.receivers, dtype=np.int32)
        node_valid, member_valid = _active_graph_masks(graph)
        member_valid &= node_valid[senders] & node_valid[receivers]
        if not np.any(node_valid) or not np.any(member_valid):
            raise ValueError("Force-density structures require active nodes and members.")
        if np.any(member_valid & (senders == receivers)):
            raise ValueError("Active force-density members may not be self-loops.")

        constraints = _constraint_mask(
            node_count,
            resolved_dimension,
            constrained_dofs,
            fixed_nodes,
        ).copy()
        constraints[~node_valid, :] = True
        flat_constraints = constraints.reshape((-1,))
        free = np.flatnonzero(~flat_constraints).astype(np.int32)
        constrained = np.flatnonzero(flat_constraints).astype(np.int32)
        full_to_free = np.full((node_count * resolved_dimension,), -1, dtype=np.int32)
        full_to_free[free] = np.arange(free.size, dtype=np.int32)

        component_labels, component_count = _components(
            node_count,
            senders,
            receivers,
            node_valid,
            member_valid,
        )
        missing: list[tuple[int, int]] = []
        for component in range(component_count):
            component_nodes = component_labels == component
            for coordinate in range(resolved_dimension):
                if not np.any(constraints[component_nodes, coordinate]):
                    missing.append((component, coordinate))
        if missing:
            raise ValueError(
                "Every active connected component must constrain every translation "
                f"coordinate; missing (component, coordinate) pairs: {tuple(missing)}."
            )

        if surface_connectivity is not None:
            if not isinstance(surface_connectivity, PolygonalConnectivity):
                raise TypeError(
                    "surface_connectivity must be PolygonalConnectivity or None."
                )
            if surface_connectivity.vertex_count != node_count:
                raise ValueError(
                    "Surface connectivity vertex count must match graph node count."
                )
            if np.any(~node_valid):
                raise ValueError(
                    "Surface-connected force-density structures do not support "
                    "inactive padded nodes."
                )

        source, target, route_members, route_signs = _equilibrium_routes(
            senders,
            receivers,
            member_valid,
            full_to_free,
            resolved_dimension,
        )
        relation = EdgeRelation(
            source,
            target,
            source_size=int(free.size),
            target_size=int(free.size),
        )
        identifier = canonical_fingerprint(
            {
                "kind": "force-density-structure",
                "dimension": resolved_dimension,
                "graph_counts": {
                    "nodes": np.asarray(graph.n_node).tolist(),
                    "members": np.asarray(graph.n_edge).tolist(),
                },
                "topology": array_tree_fingerprint(
                    (
                        senders,
                        receivers,
                        node_valid,
                        member_valid,
                        constraints,
                    )
                ),
                "surface": (
                    None
                    if surface_connectivity is None
                    else array_tree_fingerprint(
                        (
                            surface_connectivity.cell_vertices,
                            surface_connectivity.cell_vertex_valid,
                        )
                    )
                ),
            }
        )

        self.graph = graph
        self.dimension = resolved_dimension
        self.senders = jnp.asarray(senders)
        self.receivers = jnp.asarray(receivers)
        self.node_valid = jnp.asarray(node_valid)
        self.member_valid = jnp.asarray(member_valid)
        self.constrained_dofs = jnp.asarray(constraints)
        self.free_dof_indices = jnp.asarray(free)
        self.constrained_dof_indices = jnp.asarray(constrained)
        self.full_to_free = jnp.asarray(full_to_free)
        self.component_labels = jnp.asarray(component_labels)
        self.component_count = component_count
        self.equilibrium_relation = relation
        self.route_members = jnp.asarray(route_members)
        self.route_signs = jnp.asarray(route_signs)
        self.surface_connectivity = surface_connectivity
        self.structure_id = identifier

    @classmethod
    def from_graph(
        cls,
        graph: GraphIR,
        dimension: int,
        /,
        *,
        constrained_dofs: ArrayLike | None = None,
        fixed_nodes: Sequence[int] | ArrayLike | None = None,
        surface_connectivity: PolygonalConnectivity | None = None,
    ) -> ForceDensityStructure:
        """Construct from a graph with one directed route per physical member."""
        return cls(
            graph,
            dimension,
            constrained_dofs=constrained_dofs,
            fixed_nodes=fixed_nodes,
            surface_connectivity=surface_connectivity,
        )

    @classmethod
    def from_edges(
        cls,
        edges: ArrayLike,
        node_count: int,
        dimension: int,
        /,
        *,
        constrained_dofs: ArrayLike | None = None,
        fixed_nodes: Sequence[int] | ArrayLike | None = None,
        node_mask: ArrayLike | None = None,
        edge_mask: ArrayLike | None = None,
        surface_connectivity: PolygonalConnectivity | None = None,
    ) -> ForceDensityStructure:
        endpoints = np.asarray(edges)
        if endpoints.ndim != 2 or endpoints.shape[1] != 2:
            raise ValueError("edges must have shape (members, 2).")
        if not np.issubdtype(endpoints.dtype, np.integer):
            raise TypeError("edges must have integer dtype.")
        endpoints = endpoints.astype(np.int32, copy=False)
        nodes = int(node_count)
        if nodes <= 0 or (
            endpoints.size and (np.any(endpoints < 0) or np.any(endpoints >= nodes))
        ):
            raise ValueError("edge endpoints must lie in [0, node_count).")
        graph = GraphIR(
            senders=endpoints[:, 0],
            receivers=endpoints[:, 1],
            n_node=jnp.asarray((nodes,), dtype=jnp.int32),
            n_edge=jnp.asarray((endpoints.shape[0],), dtype=jnp.int32),
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
        return cls(
            graph,
            dimension,
            constrained_dofs=constrained_dofs,
            fixed_nodes=fixed_nodes,
            surface_connectivity=surface_connectivity,
        )

    @classmethod
    def from_cell_mesh(
        cls,
        mesh: CellMesh,
        /,
        *,
        constrained_dofs: ArrayLike | None = None,
        fixed_nodes: Sequence[int] | ArrayLike | None = None,
    ) -> ForceDensityStructure:
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Force-density cell meshes must have polygonal connectivity.")
        return cls.from_edges(
            mesh.connectivity.edges,
            int(mesh.coordinates.shape[0]),
            int(mesh.coordinates.shape[1]),
            constrained_dofs=constrained_dofs,
            fixed_nodes=fixed_nodes,
            surface_connectivity=mesh.connectivity,
        )

    @property
    def node_count(self) -> int:
        return int(self.node_valid.shape[0])

    @property
    def member_count(self) -> int:
        return int(self.member_valid.shape[0])

    @property
    def full_dof_count(self) -> int:
        return self.node_count * self.dimension

    @property
    def free_dof_count(self) -> int:
        return int(self.free_dof_indices.shape[0])

    @property
    def constrained_dof_count(self) -> int:
        return int(self.constrained_dof_indices.shape[0])

    def prescribed_values(self, positions: ArrayLike, /) -> Array:
        values = jnp.asarray(positions)
        if values.shape != (self.node_count, self.dimension):
            raise ValueError(
                "positions must have shape "
                f"({self.node_count}, {self.dimension}); got {values.shape}."
            )
        return values.reshape((-1,))[self.constrained_dof_indices]

    def lift(self, prescribed_values: ArrayLike, /) -> Array:
        values = jnp.asarray(prescribed_values)
        if values.shape != (self.constrained_dof_count,):
            raise ValueError(
                "prescribed_values must have shape "
                f"({self.constrained_dof_count},); got {values.shape}."
            )
        return (
            jnp.zeros((self.full_dof_count,), dtype=values.dtype)
            .at[self.constrained_dof_indices]
            .set(values, unique_indices=True)
            .reshape((self.node_count, self.dimension))
        )

    def expand(self, reduced: ArrayLike, prescribed_values: ArrayLike, /) -> Array:
        reduced_values = jnp.asarray(reduced)
        if reduced_values.shape != (self.free_dof_count,):
            raise ValueError(
                f"reduced must have shape ({self.free_dof_count},); "
                f"got {reduced_values.shape}."
            )
        lift = self.lift(prescribed_values).reshape((-1,))
        return (
            lift.at[self.free_dof_indices]
            .set(reduced_values, unique_indices=True)
            .reshape((self.node_count, self.dimension))
        )

    def reduce(self, full: ArrayLike, /) -> Array:
        values = jnp.asarray(full)
        if values.shape != (self.node_count, self.dimension):
            raise ValueError(
                "full must have shape "
                f"({self.node_count}, {self.dimension}); got {values.shape}."
            )
        return values.reshape((-1,))[self.free_dof_indices]


__all__ = ["ForceDensityStructure"]
