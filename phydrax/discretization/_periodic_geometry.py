#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific quotient coefficients and their separate RNE execution carrier."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ._cell_mesh import CellMesh
from ._periodic_topology import _exact_periodic_element, _exact_periodic_generators


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._coordinate_enclosure import CoordinateSourceBank
    from .fem._generic import FiniteElementDofMap


class PeriodicCellGeometrySource(StrictModule, NonTrainableState):
    """Existing FE nodal numbering bound to exact authored group expressions.

    Numerical image coefficients are not independent scientific parameters.
    Only representative coefficients are dynamic parameters of this source.
    Exact coefficients are prepared on the host; execution remains ordinary
    binary64 JAX operations with a separately bounded coefficient error.
    """

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    numbering: FiniteElementDofMap
    representative_coordinates: Array
    representative_nodes: tuple[int, ...] = eqx.field(static=True)
    node_orbits: tuple[int, ...] = eqx.field(static=True)
    node_exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    active_indices: tuple[int, ...] = eqx.field(static=True)
    exact_isometries: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = eqx.field(
        static=True
    )
    numerical_isometries: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(
        static=True
    )
    node_isometries: tuple[int, ...] = eqx.field(static=True)
    source_layout_id: str = eqx.field(static=True)

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
        from .fem._generic import (
            _build_finite_element_dof_routes,
            _lifted_dof_entities,
            _lifted_finite_element_dof_layout,
            FiniteElementDofMap,
        )
        from .fem._reference import FiniteElementSpec

        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "A quotient coordinate source requires canonical mesh and geometry owners."
            )
        periodic = mesh.periodic_topology
        if (
            periodic is None
            or geometry.periodic_source is not None
            or geometry.exact_source is not None
        ):
            raise ValueError(
                "A quotient coordinate source must compile an independent nodal carrier on its authored periodic topology."
            )
        elements, routes, _ = geometry.resolve(mesh)
        if any(
            not isinstance(element, FiniteElementSpec)
            or element.representation != "point_value"
            or element.mapping != "identity"
            or element.value_shape
            for element in elements
        ):
            raise ValueError(
                "A quotient coordinate source requires canonical scalar nodal coordinate elements."
            )
        resolved = tuple(
            element for element in elements if isinstance(element, FiniteElementSpec)
        )
        dofs = (
            FiniteElementDofMap(mesh, resolved, coordinate_spec=geometry)
            if numbering is None
            else numbering
        )
        if not isinstance(dofs, FiniteElementDofMap) or dofs.coordinate_gather is None:
            raise TypeError(
                "The quotient coordinate source requires the actual FE nodal numbering owner."
            )
        flattened = np.concatenate(
            [np.asarray(route, dtype=np.int64).reshape(-1) for route in routes]
        )
        representatives = flattened[np.asarray(dofs.coordinate_gather, dtype=np.int64)]
        count = geometry.coordinates.shape[0]
        orbits = np.full(count, -1, dtype=np.int64)
        lifted = _lifted_finite_element_dof_layout(mesh, resolved, ())
        lifted_routes, _, _ = _build_finite_element_dof_routes(mesh, resolved, lifted)
        degrees, entities, _, _ = _lifted_dof_entities(mesh, lifted)
        owner_degree = np.full(count, -1, dtype=np.int64)
        owner_entity = np.full(count, -1, dtype=np.int64)
        for route, quotient_route, lifted_route in zip(
            routes, dofs.cell_dofs, lifted_routes, strict=True
        ):
            for node, orbit, lifted_dof in zip(
                np.asarray(route).reshape(-1),
                np.asarray(quotient_route).reshape(-1),
                np.asarray(lifted_route).reshape(-1),
                strict=True,
            ):
                node, orbit, lifted_dof = int(node), int(orbit), int(lifted_dof)
                if orbits[node] >= 0 and orbits[node] != orbit:
                    raise ValueError(
                        "Shared coordinate coefficients disagree on their authored quotient identity."
                    )
                orbits[node] = orbit
                owner_degree[node], owner_entity[node] = (
                    degrees[lifted_dof],
                    entities[lifted_dof],
                )
        if np.any(orbits < 0) or np.any(owner_degree < 0):
            raise ValueError(
                "Every coordinate coefficient requires its actual FE quotient owner."
            )
        generators, orders = _exact_periodic_generators(periodic.cell)
        exponent_rows = []
        exact = []
        numerical = []
        indices = []
        unique: dict[tuple[int, ...], int] = {}
        for node, orbit in enumerate(orbits):
            root = int(representatives[orbit])
            degree = int(owner_degree[node])
            if degree != int(owner_degree[root]):
                raise ValueError(
                    "A quotient coefficient changes its owning entity degree."
                )
            quotient, _, anchors = (
                np.asarray(value) for value in periodic.orbits(degree)
            )
            copy, base = int(owner_entity[node]), int(owner_entity[root])
            if quotient[copy] != quotient[base]:
                raise ValueError(
                    "A quotient coefficient crosses unrelated winding entities."
                )
            exponent = tuple(
                (int(a) - int(b)) % order if order else int(a) - int(b)
                for a, b, order in zip(anchors[copy], anchors[base], orders, strict=True)
            )
            exponent_rows.append(exponent)
            if exponent not in unique:
                unique[exponent] = len(exact)
                matrix = _exact_periodic_element(generators, orders, exponent)
                exact.append(
                    tuple(
                        tuple((value.numerator, value.denominator) for value in row)
                        for row in matrix
                    )
                )
                numerical.append(
                    tuple(tuple(float(value) for value in row) for row in matrix)
                )
            indices.append(unique[exponent])
        active = (
            np.arange(count, dtype=np.int64)
            if active_indices is None
            else np.asarray(active_indices)
        )
        if (
            active.ndim != 1
            or not np.issubdtype(active.dtype, np.integer)
            or np.any(active < 0)
            or np.any(active >= count)
        ):
            raise ValueError(
                "Quotient coefficient views require valid original source indices."
            )
        values = (
            geometry.coordinates[representatives]
            if representative_coordinates is None
            else jnp.asarray(representative_coordinates, dtype=jnp.float64)
        )
        if values.shape != (dofs.global_dof_count, mesh.ambient_dimension) or not np.all(
            np.isfinite(np.asarray(values))
        ):
            raise ValueError(
                "Representative coefficients must cover the actual finite-element quotient numbering."
            )
        self.source_mesh, self.source_geometry, self.numbering = mesh, geometry, dofs
        self.representative_coordinates = values
        self.representative_nodes = tuple(int(value) for value in representatives)
        self.node_orbits = tuple(int(value) for value in orbits)
        self.node_exponents = tuple(exponent_rows)
        self.active_indices = tuple(int(value) for value in active)
        self.exact_isometries, self.numerical_isometries = tuple(exact), tuple(numerical)
        self.node_isometries = tuple(indices)
        self.source_layout_id = canonical_fingerprint(
            {
                "kind": "periodic-coordinate-source-layout",
                "topology": periodic.periodic_topology_id,
                "geometry_layout": geometry.geometry_layout_id,
                "numbering": dofs.dof_map_id,
                "orbits": self.node_orbits,
                "exponents": self.node_exponents,
                "active_indices": self.active_indices,
            }
        )

    @property
    def source_id(self) -> str:
        from ._cell_geometry import CellGeometrySpec

        if not isinstance(self.source_geometry, CellGeometrySpec):
            raise TypeError("The quotient source lost its actual coordinate basis owner.")
        return canonical_fingerprint(
            {
                "kind": "periodic-coordinate-source",
                "layout": self.source_layout_id,
                "representatives": array_tree_fingerprint(
                    self.representative_coordinates
                ),
                "source_elements": array_tree_fingerprint(self.source_geometry.elements),
                "exact_isometries": self.exact_isometries,
            }
        )

    def runtime_coordinates(self) -> Array:
        orbits = jnp.asarray(
            tuple(self.node_orbits[index] for index in self.active_indices),
            dtype=jnp.int32,
        )
        image_indices = jnp.asarray(
            tuple(self.node_isometries[index] for index in self.active_indices),
            dtype=jnp.int32,
        )
        matrices = jnp.asarray(self.numerical_isometries, dtype=jnp.float64)[
            image_indices
        ]
        values = self.representative_coordinates[orbits]
        return (
            contract("nij,nj->ni", matrices[:, :-1, :-1], values) + matrices[:, :-1, -1]
        )

    def source_coordinates(self) -> CoordinateSourceBank:
        if isinstance(self.representative_coordinates, jax_core.Tracer):
            raise TypeError("Exact quotient coefficient preparation is host-only.")
        values = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(self.representative_coordinates)
        )
        result = []
        for node in self.active_indices:
            point = values[self.node_orbits[node]]
            matrix = self.exact_isometries[self.node_isometries[node]]
            result.append(
                tuple(
                    sum(
                        (
                            Fraction(*entry) * value
                            for entry, value in zip(row[:-1], point, strict=True)
                        ),
                        Fraction(*row[-1]),
                    )
                    for row in matrix[:-1]
                )
            )
        return tuple(result)

    def reindexed(self, indices: ArrayLike, /) -> PeriodicCellGeometrySource:
        from ._cell_geometry import CellGeometrySpec

        if not isinstance(self.source_geometry, CellGeometrySpec):
            raise TypeError("The periodic source lost its canonical geometry owner.")
        indices_ = np.asarray(indices)
        if (
            indices_.ndim != 1
            or not np.issubdtype(indices_.dtype, np.integer)
            or np.any(indices_ < 0)
            or np.any(indices_ >= len(self.active_indices))
        ):
            raise ValueError(
                "Periodic coefficient reindexing requires valid current source rows."
            )
        return PeriodicCellGeometrySource(
            self.source_mesh,
            self.source_geometry,
            active_indices=np.asarray(
                [self.active_indices[int(index)] for index in indices_], dtype=np.int64
            ),
            representative_coordinates=self.representative_coordinates,
            numbering=self.numbering,
        )

    def rebound(self, coordinates: ArrayLike, /) -> PeriodicCellGeometrySource:
        from ._cell_geometry import CellGeometrySpec

        if not isinstance(self.source_geometry, CellGeometrySpec):
            raise TypeError("The periodic source lost its canonical geometry owner.")
        values = jnp.asarray(coordinates, dtype=jnp.float64)
        if values.shape != (
            len(self.active_indices),
            self.representative_coordinates.shape[1],
        ):
            raise ValueError(
                "Periodic rebinding must preserve the actual coefficient view."
            )
        positions = {node: position for position, node in enumerate(self.active_indices)}
        roots = [
            orbit
            for orbit, node in enumerate(self.representative_nodes)
            if node in positions
        ]
        rows = [positions[self.representative_nodes[orbit]] for orbit in roots]
        representatives = self.representative_coordinates.at[
            jnp.asarray(roots, dtype=jnp.int32)
        ].set(values[jnp.asarray(rows, dtype=jnp.int32)])
        return PeriodicCellGeometrySource(
            self.source_mesh,
            self.source_geometry,
            active_indices=np.asarray(self.active_indices, dtype=np.int64),
            representative_coordinates=representatives,
            numbering=self.numbering,
        )
