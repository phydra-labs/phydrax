#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Physical interval-mesh networks with conservative cochain actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from ._cell_complex import IntervalConnectivity
from ._cell_mesh import CellBlock, CellMesh


class NetworkEvidence(StrictModule):
    connected_component_count: Array
    root_count: Array
    tip_count: Array
    minimum_length: Array
    minimum_area: Array
    finite: Array
    successful: Array


class PreparedMetricNetwork(StrictModule):
    mesh: CellMesh
    coordinate_contract: SpatialCoordinateContract
    senders: Array
    receivers: Array
    lengths: Array
    tangents: Array
    areas: Array
    perimeters: Array
    node_measures: Array
    root_mask: Array
    tip_mask: Array
    evidence: NetworkEvidence
    network_id: str = eqx.field(static=True)

    def gradient(self, node_values: ArrayLike, /) -> Array:
        values = jnp.asarray(node_values)
        if values.shape[:1] != (self.mesh.coordinates.shape[0],):
            raise ValueError("node_values must begin with the network vertex count.")
        scale = self.lengths.reshape(self.lengths.shape + (1,) * (values.ndim - 1))
        return (values[self.receivers] - values[self.senders]) / scale

    def divergence(self, oriented_edge_fluxes: ArrayLike, /) -> Array:
        flux = jnp.asarray(oriented_edge_fluxes)
        if flux.shape[:1] != (self.senders.shape[0],):
            raise ValueError(
                "oriented_edge_fluxes must begin with the network edge count."
            )
        result = jnp.zeros(
            (self.mesh.coordinates.shape[0],) + flux.shape[1:], dtype=flux.dtype
        )
        result = result.at[self.senders].add(-flux)
        result = result.at[self.receivers].add(flux)
        return result

    def mass(self, node_values: ArrayLike, /) -> Array:
        values = jnp.asarray(node_values)
        if values.shape[:1] != (self.mesh.coordinates.shape[0],):
            raise ValueError("node_values must begin with the network vertex count.")
        weights = self.node_measures.reshape(
            self.node_measures.shape + (1,) * (values.ndim - 1)
        )
        return jnp.sum(weights * values, axis=0)

    def diffusion(self, node_values: ArrayLike, diffusivity: ArrayLike, /) -> Array:
        gradient = self.gradient(node_values)
        coefficient = jnp.asarray(diffusivity)
        if coefficient.shape == ():
            coefficient = jnp.broadcast_to(coefficient, self.lengths.shape)
        if coefficient.shape != self.lengths.shape:
            raise ValueError("diffusivity must be scalar or one value per network edge.")
        scale = (self.areas * coefficient).reshape(
            self.areas.shape + (1,) * (gradient.ndim - 1)
        )
        return -self.divergence(scale * gradient)

    def advective_flux(self, node_values: ArrayLike, volume_flow: ArrayLike, /) -> Array:
        values = jnp.asarray(node_values)
        flow = jnp.asarray(volume_flow)
        if flow.shape != self.lengths.shape:
            raise ValueError("volume_flow must contain one oriented value per edge.")
        scale = flow.reshape(flow.shape + (1,) * (values.ndim - 1))
        upwind = jnp.where(scale >= 0.0, values[self.senders], values[self.receivers])
        return scale * upwind


@dataclass(frozen=True, slots=True)
class MetricNetworkPlan:
    mesh: CellMesh
    coordinate_contract: SpatialCoordinateContract
    areas: np.ndarray
    perimeters: np.ndarray
    root_vertex_ids: np.ndarray
    tip_vertex_ids: np.ndarray
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, CellMesh) or not isinstance(
            self.mesh.connectivity, IntervalConnectivity
        ):
            raise TypeError("Metric networks require an interval CellMesh.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        areas = np.asarray(self.areas, dtype=float)
        perimeters = np.asarray(self.perimeters, dtype=float)
        edges = np.concatenate([np.asarray(block.vertices) for block in self.mesh.blocks])
        if areas.shape != (len(edges),) or perimeters.shape != (len(edges),):
            raise ValueError("areas and perimeters must contain one value per edge.")
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("areas must be finite and positive.")
        if np.any(~np.isfinite(perimeters)) or np.any(perimeters <= 0.0):
            raise ValueError("perimeters must be finite and positive.")
        vertex_ids = np.asarray(self.mesh.vertex_global_ids)
        roots = np.asarray(self.root_vertex_ids)
        tips = np.asarray(self.tip_vertex_ids)
        if not np.issubdtype(roots.dtype, np.integer) or not np.issubdtype(
            tips.dtype, np.integer
        ):
            raise TypeError("root and tip IDs must be integer arrays.")
        if roots.ndim != 1 or tips.ndim != 1 or roots.size == 0 or tips.size == 0:
            raise ValueError("root and tip IDs must be non-empty rank-one arrays.")
        if np.intersect1d(roots, tips).size:
            raise ValueError("Root and tip vertices must be disjoint.")
        if np.setdiff1d(np.concatenate((roots, tips)), vertex_ids).size:
            raise ValueError("Root and tip IDs must belong to the network mesh.")
        for name, value in (
            ("areas", areas),
            ("perimeters", perimeters),
            ("root_vertex_ids", roots.astype(np.int64)),
            ("tip_vertex_ids", tips.astype(np.int64)),
        ):
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "metric-network-plan",
                    "mesh": self.mesh.mesh_id,
                    "coordinate_contract": self.coordinate_contract.spatial_id,
                    "areas": array_tree_fingerprint(areas),
                    "perimeters": array_tree_fingerprint(perimeters),
                    "roots": array_tree_fingerprint(roots),
                    "tips": array_tree_fingerprint(tips),
                }
            ),
        )

    @classmethod
    def from_arrays(
        cls,
        coordinates: ArrayLike,
        edges: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        areas: ArrayLike,
        perimeters: ArrayLike,
        root_vertex_ids: ArrayLike,
        tip_vertex_ids: ArrayLike,
        vertex_global_ids: ArrayLike | None = None,
        edge_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
    ) -> MetricNetworkPlan:
        edge_array = np.asarray(edges)
        block = CellBlock(
            "network-edges", "interval", edge_array, global_ids=edge_global_ids
        )
        mesh = CellMesh(
            coordinates,
            (block,),
            vertex_global_ids=vertex_global_ids,
            numeric_version=numeric_version,
        )
        return cls(
            mesh,
            coordinate_contract,
            np.asarray(areas),
            np.asarray(perimeters),
            np.asarray(root_vertex_ids),
            np.asarray(tip_vertex_ids),
        )

    def prepare(self) -> PreparedMetricNetwork:
        edges = np.concatenate([np.asarray(block.vertices) for block in self.mesh.blocks])
        points = np.asarray(self.mesh.coordinates)
        difference = points[edges[:, 1]] - points[edges[:, 0]]
        lengths = np.linalg.norm(difference, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
            raise ValueError("Network edges must have finite positive length.")
        tangents = difference / lengths[:, None]
        node_measures = np.zeros((len(points),), dtype=float)
        half_volume = 0.5 * self.areas * lengths
        np.add.at(node_measures, edges[:, 0], half_volume)
        np.add.at(node_measures, edges[:, 1], half_volume)
        vertex_ids = np.asarray(self.mesh.vertex_global_ids)
        root_mask = np.isin(vertex_ids, self.root_vertex_ids)
        tip_mask = np.isin(vertex_ids, self.tip_vertex_ids)
        adjacency = [[] for _ in points]
        for first, second in edges:
            adjacency[int(first)].append(int(second))
            adjacency[int(second)].append(int(first))
        visited = np.zeros((len(points),), dtype=bool)
        component_count = 0
        for start in range(len(points)):
            if visited[start]:
                continue
            component_count += 1
            stack = [start]
            visited[start] = True
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
        finite = bool(
            np.all(np.isfinite(tangents))
            and np.all(np.isfinite(node_measures))
            and np.all(node_measures > 0.0)
        )
        successful = finite and component_count == 1
        evidence = NetworkEvidence(
            jnp.asarray(component_count, dtype=jnp.int32),
            jnp.asarray(np.count_nonzero(root_mask), dtype=jnp.int32),
            jnp.asarray(np.count_nonzero(tip_mask), dtype=jnp.int32),
            jnp.asarray(lengths.min()),
            jnp.asarray(self.areas.min()),
            jnp.asarray(finite),
            jnp.asarray(successful),
        )
        network_id = canonical_fingerprint(
            {
                "kind": "prepared-metric-network",
                "plan": self.plan_id,
                "lengths": array_tree_fingerprint(lengths),
                "node_measures": array_tree_fingerprint(node_measures),
            }
        )
        return PreparedMetricNetwork(
            self.mesh,
            self.coordinate_contract,
            jnp.asarray(edges[:, 0], dtype=jnp.int32),
            jnp.asarray(edges[:, 1], dtype=jnp.int32),
            jnp.asarray(lengths),
            jnp.asarray(tangents),
            jnp.asarray(self.areas),
            jnp.asarray(self.perimeters),
            jnp.asarray(node_measures),
            jnp.asarray(root_mask),
            jnp.asarray(tip_mask),
            evidence,
            network_id,
        )


__all__ = ["MetricNetworkPlan", "NetworkEvidence", "PreparedMetricNetwork"]
