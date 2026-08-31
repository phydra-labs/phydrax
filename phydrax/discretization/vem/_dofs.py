#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import RowRelation
from .._cell_complex import PolygonalConnectivity
from .._cell_mesh import CellMesh
from ._spec import VirtualElementSpec


class VirtualElementDofMap(StrictModule, NonTrainableState):
    """Entity-functional DOFs and per-block local gathers."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    cell_dofs: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    vertex_dof_count: int = eqx.field(static=True)
    edge_dof_count: int = eqx.field(static=True)
    cell_dof_count: int = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    boundary_dof_mask: Array
    point_dof_valid: Array
    default_dof_points: Array
    dof_map_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, element: VirtualElementSpec, /):
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError(
                "Virtual elements require two-dimensional polygon connectivity."
            )
        if not isinstance(element, VirtualElementSpec):
            raise TypeError("element must be VirtualElementSpec.")
        connectivity = mesh.connectivity
        vertex_count = int(mesh.coordinates.shape[0])
        edge_count = int(connectivity.edges.shape[0])
        cell_count = connectivity.cell_count
        edge_width = element.edge_interior_dof_count
        cell_width = element.cell_moment_count
        edge_dof_count = edge_count * edge_width
        cell_dof_count = cell_count * cell_width
        global_count = vertex_count + edge_dof_count + cell_dof_count
        cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_edge_signs, dtype=float)
        block_dofs = []
        relations = []
        cell_offset = 0
        for block in mesh.blocks:
            width = element.local_dof_count(block.arity)
            local = np.empty((block.cell_count, width), dtype=np.int32)
            vertices = np.asarray(block.vertices, dtype=np.int32)
            local[:, : block.arity] = vertices
            cursor = block.arity
            if edge_width:
                positions = np.arange(edge_width, dtype=np.int32)
                for local_edge in range(block.arity):
                    edges = cell_edges[
                        cell_offset : cell_offset + block.cell_count, local_edge
                    ]
                    signs = cell_signs[
                        cell_offset : cell_offset + block.cell_count, local_edge
                    ]
                    oriented = np.where(
                        signs[:, None] > 0.0,
                        positions[None, :],
                        positions[::-1][None, :],
                    )
                    local[:, cursor : cursor + edge_width] = (
                        vertex_count + edges[:, None] * edge_width + oriented
                    )
                    cursor += edge_width
            if cell_width:
                cells = np.arange(
                    cell_offset,
                    cell_offset + block.cell_count,
                    dtype=np.int32,
                )
                local[:, cursor : cursor + cell_width] = (
                    vertex_count
                    + edge_dof_count
                    + cells[:, None] * cell_width
                    + np.arange(cell_width, dtype=np.int32)[None, :]
                )
            block_dofs.append(jnp.asarray(local))
            relations.append(RowRelation(local, source_size=global_count))
            cell_offset += block.cell_count
        boundary = np.zeros((global_count,), dtype=bool)
        boundary[:vertex_count] = np.asarray(connectivity.boundary_vertices, dtype=bool)
        if edge_width:
            boundary[vertex_count : vertex_count + edge_dof_count] = np.repeat(
                np.asarray(connectivity.boundary_edges, dtype=bool), edge_width
            )
        point_valid = np.ones((global_count,), dtype=bool)
        if cell_width:
            point_valid[vertex_count + edge_dof_count :] = False
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.cell_dofs = tuple(block_dofs)
        self.relations = tuple(relations)
        self.vertex_dof_count = vertex_count
        self.edge_dof_count = edge_dof_count
        self.cell_dof_count = cell_dof_count
        self.global_dof_count = global_count
        self.boundary_dof_mask = jnp.asarray(boundary)
        self.point_dof_valid = jnp.asarray(point_valid)
        self.default_dof_points = self.evaluate_point_coordinates(mesh.coordinates, mesh)
        self.dof_map_id = canonical_fingerprint(
            {
                "kind": "virtual-element-dof-map",
                "mesh": mesh.topology_id,
                "element": element.element_id,
                "routes": [array_tree_fingerprint(value) for value in block_dofs],
                "boundary": array_tree_fingerprint(boundary),
                "point_valid": array_tree_fingerprint(point_valid),
            }
        )

    def evaluate_point_coordinates(
        self,
        coordinates: ArrayLike,
        mesh: CellMesh,
        /,
    ) -> Array:
        points = jnp.asarray(coordinates)
        if points.shape != mesh.coordinates.shape:
            raise ValueError(
                "VEM coordinate refresh must preserve the mesh coordinate shape."
            )
        result = jnp.zeros((self.global_dof_count, points.shape[1]), dtype=points.dtype)
        result = result.at[: self.vertex_dof_count].set(points)
        edge_width = (
            self.edge_dof_count // int(mesh.connectivity.edges.shape[0])
            if self.edge_dof_count
            else 0
        )
        if edge_width:
            from ...integration import GaussLobattoLegendreRule, interval_rule_data

            data = interval_rule_data(GaussLobattoLegendreRule(edge_width + 2))
            nodes = 0.5 * (jnp.asarray(data.nodes)[1:-1] + 1.0)
            edges = jnp.asarray(mesh.connectivity.edges, dtype=jnp.int32)
            start = points[edges[:, 0]]
            stop = points[edges[:, 1]]
            edge_points = (1.0 - nodes[None, :, None]) * start[:, None, :] + nodes[
                None, :, None
            ] * stop[:, None, :]
            result = result.at[
                self.vertex_dof_count : self.vertex_dof_count + self.edge_dof_count
            ].set(edge_points.reshape((-1, points.shape[1])))
        return result


__all__ = ["VirtualElementDofMap"]
