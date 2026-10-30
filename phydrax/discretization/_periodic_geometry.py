#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._periodic_exact import exact_periodic_element, ExactIsometry, identification_orders
from ._periodic_topology import PeriodicMeshTopology


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh
    from .fem._generic import FiniteElementDofMap


def _readonly_indices(value: ArrayLike | None, count: int, /) -> np.ndarray:
    if value is None:
        result = np.arange(count, dtype=np.int64)
    else:
        result = np.asarray(value)
        if result.ndim != 1 or result.dtype.kind not in "iu":
            raise TypeError("active_indices must be an integer vector.")
        if result.dtype.kind == "u" and result.size and np.any(result > np.iinfo(np.int64).max):
            raise ValueError("active_indices must fit int64.")
        result = result.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= count) or np.unique(result).size != result.size:
        raise ValueError("active_indices must be unique in-range source DOF rows.")
    result = result.copy()
    result.flags.writeable = False
    return result


def _exact_point(matrix: ExactIsometry, point: np.ndarray, /) -> tuple[Fraction, ...]:
    vector = tuple(Fraction.from_float(float(item)) for item in point) + (Fraction(1),)
    dimension = len(point)
    return tuple(
        sum(
            (matrix[row][column] * vector[column] for column in range(dimension + 1)),
            Fraction(),
        )
        for row in range(dimension)
    )


@final
class PeriodicCellGeometrySource(StrictModule, NonTrainableState):
    """Bind lifted P1 coordinate DOFs to authored periodic vertex representatives."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    numbering: FiniteElementDofMap
    representative_coordinates: Array
    representative_nodes: tuple[int, ...] = eqx.field(static=True)
    node_orbits: tuple[int, ...] = eqx.field(static=True)
    node_exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    active_indices: tuple[int, ...] = eqx.field(static=True)
    exact_isometries: tuple[ExactIsometry, ...] = eqx.field(static=True)
    numerical_isometries: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(static=True)
    node_isometries: tuple[int, ...] = eqx.field(static=True)
    source_layout_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        active_indices: ArrayLike | None = None,
        representative_coordinates: ArrayLike | None = None,
        numbering: FiniteElementDofMap | None = None,
    ) -> None:
        from ._cell_geometry import CellGeometrySpec
        from ._cell_mesh import CellMesh
        from .fem._generic import FiniteElementDofMap
        from .fem._reference import FiniteElementSpec
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError("Periodic geometry source requires a CellMesh and CellGeometrySpec.")
        periodic = mesh.periodic_topology
        if not isinstance(periodic, PeriodicMeshTopology):
            raise ValueError("source_mesh must retain its PeriodicMeshTopology.")
        periodic.require_lifted(mesh.topology, mesh.topology_id, mesh.blocks)
        periodic.require_coordinates(geometry.coordinates)
        if (
            geometry.source_map is not None
            or geometry.restriction_source is not None
        ):
            raise ValueError(
                "Periodic geometry source must bind an independent nodal coordinate carrier."
            )
        elements, routes, _ = geometry.resolve(mesh)
        finite_elements = tuple(
            element for element in elements if isinstance(element, FiniteElementSpec)
        )
        if len(finite_elements) != len(elements):
            raise ValueError(
                "Periodic geometry requires FiniteElementSpec coordinate elements."
            )
        for element in finite_elements:
            if (
                element.representation != "point_value"
                or element.mapping != "identity"
                or element.value_shape
            ):
                raise ValueError("Periodic geometry supports scalar identity-mapped P1 point-value coordinates only.")
        dofs = (
            FiniteElementDofMap(mesh, finite_elements)
            if numbering is None
            else numbering
        )
        if not isinstance(dofs, FiniteElementDofMap):
            raise TypeError("numbering must be a FiniteElementDofMap.")
        count = int(dofs.global_dof_count)
        vertex_count = int(mesh.coordinates.shape[0])
        if dofs.association != "vertex" or count != vertex_count:
            raise ValueError("Periodic geometry requires one lifted H1 DOF per source vertex.")
        vertex_for_dof = np.full((count,), -1, dtype=np.int64)
        dof_for_vertex = np.full((vertex_count,), -1, dtype=np.int64)
        for block, route, dof_rows in zip(mesh.blocks, routes, dofs.cell_dofs, strict=True):
            local_vertices = np.asarray(block.vertices, dtype=np.int64)
            geometry_rows = np.asarray(route, dtype=np.int64)
            global_dofs = np.asarray(dof_rows, dtype=np.int64)
            if (
                geometry_rows.shape != local_vertices.shape
                or global_dofs.shape != local_vertices.shape
                or not np.array_equal(geometry_rows, local_vertices)
            ):
                raise ValueError("Periodic geometry coordinates must be vertex-indexed P1 rows.")
            for vertex, dof in zip(geometry_rows.reshape(-1), global_dofs.reshape(-1), strict=True):
                if vertex_for_dof[dof] not in (-1, vertex) or dof_for_vertex[vertex] not in (-1, dof):
                    raise ValueError("FE numbering does not preserve one-to-one lifted vertex identity.")
                vertex_for_dof[dof] = vertex
                dof_for_vertex[vertex] = dof
        if np.any(vertex_for_dof < 0) or np.any(dof_for_vertex < 0):
            raise ValueError("Every periodic coordinate DOF must correspond to one source vertex.")
        quotient_begin, quotient_end = map(int, np.asarray(periodic.quotient_offsets)[:2])
        quotient_count = quotient_end - quotient_begin
        quotient_representatives = np.asarray(periodic.representatives, dtype=np.int64)[
            quotient_begin:quotient_end
        ]
        representative_nodes = tuple(
            int(dof_for_vertex[row]) for row in quotient_representatives
        )
        vertex_orbits = np.asarray(periodic.orbit_indices, dtype=np.int64)[
            np.asarray(periodic.lifted_offsets, dtype=np.int64)[0]
            : np.asarray(periodic.lifted_offsets, dtype=np.int64)[1]
        ] - quotient_begin
        node_orbits = tuple(int(vertex_orbits[vertex]) for vertex in vertex_for_dof)
        vertex_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
        node_exponents = tuple(
            tuple(map(int, vertex_shifts[vertex])) for vertex in vertex_for_dof
        )
        actions: list[ExactIsometry] = []
        action_to_index: dict[ExactIsometry, int] = {}
        node_isometries = []
        zero = np.zeros((len(identification_orders(periodic.cell)),), dtype=np.int64)
        identity = exact_periodic_element(periodic.cell, zero)
        actions.append(identity)
        action_to_index[identity] = 0
        for exponents in node_exponents:
            action = exact_periodic_element(periodic.cell, np.asarray(exponents, dtype=np.int64))
            index = action_to_index.get(action)
            if index is None:
                index = len(actions)
                actions.append(action)
                action_to_index[action] = index
            node_isometries.append(index)
        values = (
            np.asarray(geometry.coordinates, dtype=np.float64)[
                np.asarray([vertex_for_dof[node] for node in representative_nodes], dtype=np.int64)
            ]
            if representative_coordinates is None
            else np.asarray(representative_coordinates, dtype=np.float64)
        )
        if values.shape != (quotient_count, mesh.ambient_dimension) or np.any(~np.isfinite(values)):
            raise ValueError("representative_coordinates must cover every quotient coordinate DOF.")
        expanded = np.asarray(
            [
                [float(item) for item in _exact_point(actions[action], values[orbit])]
                for orbit, action in zip(node_orbits, node_isometries, strict=True)
            ],
            dtype=np.float64,
        )
        source_points = np.asarray(geometry.coordinates, dtype=np.float64)
        if not np.array_equal(expanded, source_points[vertex_for_dof]):
            raise ValueError(
                "Periodic coordinate actions do not reproduce the authored "
                "source geometry."
            )
        active = _readonly_indices(active_indices, count)
        numerical = tuple(
            tuple(tuple(float(value) for value in row) for row in matrix)
            for matrix in actions
        )
        self.source_mesh = mesh
        self.source_geometry = geometry
        self.numbering = dofs
        self.representative_coordinates = jnp.asarray(values)
        self.representative_nodes = representative_nodes
        self.node_orbits = node_orbits
        self.node_exponents = node_exponents
        self.active_indices = tuple(int(value) for value in active)
        self.exact_isometries = tuple(actions)
        self.numerical_isometries = numerical
        self.node_isometries = tuple(node_isometries)
        self.source_layout_id = canonical_fingerprint(
            {
                "kind": "periodic-coordinate-source-layout",
                "topology": periodic.periodic_topology_id,
                "geometry_layout": geometry.geometry_layout_id,
                "numbering": dofs.dof_map_id,
                "orbits": list(node_orbits),
                "exponents": [list(row) for row in node_exponents],
                "actions": [
                    [[(value.numerator, value.denominator) for value in row] for row in matrix]
                    for matrix in actions
                ],
                "active_indices": list(self.active_indices),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicCellGeometrySource],
        value: PeriodicCellGeometrySource,
        /,
    ) -> PeriodicCellGeometrySource:
        renewed = cls(
            value.source_mesh,
            value.source_geometry,
            active_indices=np.asarray(value.active_indices, dtype=np.int64),
            representative_coordinates=value.representative_coordinates,
            numbering=value.numbering,
        )
        if renewed.source_layout_id != value.source_layout_id:
            raise ValueError("Stored periodic coordinate source no longer matches its quotient layout.")
        if (
            renewed.representative_nodes != value.representative_nodes
            or renewed.node_orbits != value.node_orbits
            or renewed.node_exponents != value.node_exponents
            or renewed.exact_isometries != value.exact_isometries
            or renewed.node_isometries != value.node_isometries
        ):
            raise ValueError("Stored periodic coordinate actions changed.")
        return renewed

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "periodic-coordinate-source",
                "layout": self.source_layout_id,
                "representatives": array_tree_fingerprint(self.representative_coordinates),
                "source_elements": array_tree_fingerprint(self.source_geometry.elements),
            }
        )

    def runtime_coordinates(self, /) -> Array:
        indices = np.asarray(self.active_indices, dtype=np.int64)
        orbits = jnp.asarray(np.asarray(self.node_orbits, dtype=np.int32)[indices])
        actions = jnp.asarray(np.asarray(self.node_isometries, dtype=np.int32)[indices])
        matrices = jnp.asarray(self.numerical_isometries, dtype=self.representative_coordinates.dtype)[actions]
        values = self.representative_coordinates[orbits]
        mapped = matrices[:, :-1, :-1] @ values[..., None]
        return mapped[..., 0] + matrices[:, :-1, -1]

    def source_coordinates(self, /) -> tuple[tuple[Fraction, ...], ...]:
        if isinstance(self.representative_coordinates, jax_core.Tracer):
            raise TypeError("source_coordinates requires host representative coefficients.")
        values = np.asarray(self.representative_coordinates, dtype=np.float64)
        return tuple(
            _exact_point(
                self.exact_isometries[self.node_isometries[node]],
                values[self.node_orbits[node]],
            )
            for node in self.active_indices
        )

    def reindexed(self, indices: ArrayLike, /) -> PeriodicCellGeometrySource:
        positions = _readonly_indices(indices, len(self.active_indices))
        selected = np.asarray(self.active_indices, dtype=np.int64)[positions]
        return PeriodicCellGeometrySource(
            self.source_mesh,
            self.source_geometry,
            active_indices=selected,
            representative_coordinates=self.representative_coordinates,
            numbering=self.numbering,
        )

    def rebound(self, coordinates: ArrayLike, /) -> PeriodicCellGeometrySource:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (len(self.active_indices), self.representative_coordinates.shape[1]):
            raise ValueError("rebound coordinates must preserve the active source row shape.")
        representatives = np.asarray(self.representative_coordinates, dtype=np.float64).copy()
        positions = {node: position for position, node in enumerate(self.active_indices)}
        for orbit, node in enumerate(self.representative_nodes):
            if node in positions:
                representatives[orbit] = values[positions[node]]
        return PeriodicCellGeometrySource(
            self.source_mesh,
            self.source_geometry,
            active_indices=np.asarray(self.active_indices, dtype=np.int64),
            representative_coordinates=representatives,
            numbering=self.numbering,
        )


__all__ = ["PeriodicCellGeometrySource"]
