#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_mesh import CellMesh


@runtime_checkable
class CellGeometryElement(Protocol):
    """Reference element contract required by a cell geometry layout."""

    cell_kind: str
    conformity: str
    local_dof_count: int
    element_id: str


class CellVertexGeometryElement(StrictModule, NonTrainableState):
    """Vertex-coordinate geometry descriptor for a variable-topology cell block."""

    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(self, cell_kind: str, local_dof_count: int, /):
        kind = str(cell_kind)
        count = int(local_dof_count)
        if kind not in ("polygon", "polyhedron"):
            raise ValueError(
                "CellVertexGeometryElement supports polygon or polyhedron blocks."
            )
        if count < (3 if kind == "polygon" else 4):
            raise ValueError("Variable-topology geometry requires enough vertices.")
        self.cell_kind = kind
        self.conformity = "H1"
        self.local_dof_count = count
        self.element_id = canonical_fingerprint(
            {
                "kind": "cell-vertex-geometry-element",
                "cell_kind": kind,
                "local_dof_count": count,
            }
        )


class CellGeometrySpec(StrictModule, NonTrainableState):
    """Per-block coordinate elements, geometry routes, and coordinate values."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    elements: tuple[CellGeometryElement, ...]
    geometry_dofs: tuple[Array, ...]
    coordinates: Array
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        elements: Mapping[str, CellGeometryElement],
        geometry_dofs: Mapping[str, ArrayLike],
        coordinates: ArrayLike,
        /,
    ):
        items = tuple(sorted((str(name), element) for name, element in elements.items()))
        routes = {
            str(name): np.asarray(value, dtype=np.int32)
            for name, value in geometry_dofs.items()
        }
        points = np.asarray(coordinates, dtype=float)
        if not items or any(not name for name, _ in items):
            raise ValueError("Coordinate element mapping must be non-empty.")
        if not all(isinstance(element, CellGeometryElement) for _, element in items):
            raise TypeError("Coordinate elements must satisfy CellGeometryElement.")
        if any(element.conformity != "H1" for _, element in items):
            raise ValueError("Coordinate elements must be H1-conforming.")
        if set(routes) != {name for name, _ in items}:
            raise ValueError(
                "Coordinate DOF routes must match coordinate element blocks."
            )
        if points.ndim != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Coordinate values must be one finite rank-2 array.")
        normalized_routes = []
        for name, element in items:
            route = routes[name]
            if route.ndim != 2 or route.shape[1] != element.local_dof_count:
                raise ValueError(
                    "Coordinate DOF route width must match its coordinate element."
                )
            if np.any(route < 0) or np.any(route >= points.shape[0]):
                raise ValueError("Coordinate DOF routes index undeclared coordinates.")
            normalized_routes.append(jnp.asarray(route))
        self.block_names = tuple(name for name, _ in items)
        self.elements = tuple(element for _, element in items)
        self.geometry_dofs = tuple(normalized_routes)
        self.coordinates = jnp.asarray(points)
        self.geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-spec",
                "blocks": [[name, element.element_id] for name, element in items],
                "geometry_dofs": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in normalized_routes
                ],
                "coordinate_shape": list(points.shape),
            }
        )

    @classmethod
    def affine(cls, mesh: CellMesh, /) -> CellGeometrySpec:
        from .fem._reference import lagrange_element

        elements = {}
        for block in mesh.blocks:
            elements[block.name] = (
                CellVertexGeometryElement(block.cell_kind, block.arity)
                if block.cell_kind == "polyhedron"
                else lagrange_element(block.cell_kind, 1)
            )
        return cls(
            elements,
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
        )

    def resolve(
        self,
        mesh: CellMesh,
        /,
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        mapping = dict(zip(self.block_names, self.elements, strict=True))
        routes = dict(zip(self.block_names, self.geometry_dofs, strict=True))
        if set(mapping) != {block.name for block in mesh.blocks}:
            raise ValueError("Coordinate element assignments must match mesh blocks.")
        resolved = tuple(mapping[block.name] for block in mesh.blocks)
        resolved_routes = tuple(routes[block.name] for block in mesh.blocks)
        for block, element, route in zip(
            mesh.blocks,
            resolved,
            resolved_routes,
            strict=True,
        ):
            if block.cell_kind != element.cell_kind:
                raise ValueError("Coordinate element cell kind does not match its block.")
            if route.shape[0] != block.cell_count:
                raise ValueError("Coordinate DOF routes require one row per cell.")
        return resolved, resolved_routes, self.coordinates


__all__ = [
    "CellGeometryElement",
    "CellGeometrySpec",
    "CellVertexGeometryElement",
]
