#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific quotient coefficients and their separate RNE execution carrier."""

from __future__ import annotations

from contextlib import ExitStack
from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core, device_get, device_put
from jax.typing import ArrayLike

import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    exact_periodic_generators,
    owned_periodic_preparation,
)

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..geometry._rational import rational_vector
from ..typing import checked
from . import (
    _cell_geometry as imported_cell_geometry,
    _cell_mesh as cell_mesh_owner,
    _coordinate_enclosure as coordinate_enclosure_owner,
    _periodic_topology as periodic_topology_owner,
    fem as fem_owner,
)
from ._coordinate_budget import CoordinateEnclosureBudget


if TYPE_CHECKING:
    from ._coordinate_enclosure import CoordinateSourceBank


class _PeriodicNodeOwners(StrictModule):
    """Host correspondence between coordinate nodes and quotient entities."""

    representatives: np.ndarray
    orbits: np.ndarray
    degree: np.ndarray
    entity: np.ndarray


class _PeriodicNodeActions(StrictModule):
    """Prepared group actions in first-occurrence node order."""

    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    exact: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = eqx.field(static=True)
    numerical: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(static=True)
    indices: tuple[int, ...] = eqx.field(static=True)


def _prepare_periodic_node_owners(
    mesh: cell_mesh_owner.CellMesh,
    geometry: imported_cell_geometry.CellGeometrySpec,
    resolved: tuple[reference_types_owner.FiniteElementSpec, ...],
    routes: tuple[Array, ...],
    dofs: fem_owner.FiniteElementDofMap,
    /,
) -> _PeriodicNodeOwners:
    # Cell geometry → periodic geometry → FEM assembly → cell geometry.
    from .fem import _assembly as assembly_owner

    if dofs.coordinate_gather is None:
        raise TypeError(
            "PeriodicCellGeometrySource.numbering.coordinate_gather must retain nodal coordinate numbering; got None"
        )
    host_routes, gather = device_get((routes, dofs.coordinate_gather))
    flattened = np.concatenate(
        [np.asarray(route, dtype=np.int64).reshape(-1) for route in host_routes]
    )
    representatives = flattened[np.asarray(gather, dtype=np.int64)]
    count = geometry.coordinates.shape[0]
    orbits = np.full(count, -1, dtype=np.int64)
    lifted = assembly_owner.lifted_finite_element_dof_layout(mesh, resolved, ())
    lifted_routes, _, _ = assembly_owner.build_finite_element_dof_routes(
        mesh, resolved, lifted
    )
    degrees, entities, _, _ = assembly_owner.lifted_dof_entities(mesh, lifted)
    owner_degree = np.full(count, -1, dtype=np.int64)
    owner_entity = np.full(count, -1, dtype=np.int64)
    host_quotient, host_lifted = device_get((dofs.cell_dofs, lifted_routes))
    for route, quotient_route, lifted_route in zip(
        host_routes, host_quotient, host_lifted, strict=True
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
                    "PeriodicCellGeometrySource.node_orbits must agree at shared coordinate nodes; got conflicting quotient identities"
                )
            orbits[node] = orbit
            owner_degree[node], owner_entity[node] = (
                degrees[lifted_dof],
                entities[lifted_dof],
            )
    if np.any(orbits < 0) or np.any(owner_degree < 0):
        raise ValueError(
            "PeriodicCellGeometrySource.node_orbits must cover every coordinate coefficient; got unbound quotient entities"
        )
    return _PeriodicNodeOwners(representatives, orbits, owner_degree, owner_entity)


def _prepare_periodic_node_actions(
    periodic: periodic_topology_owner.PeriodicMeshTopology,
    owners: _PeriodicNodeOwners,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PeriodicNodeActions:
    generators, orders = exact_periodic_generators(periodic.cell, budget=budget)
    # Packed topology fields cross to the host once, before coefficient loops.
    offsets, quotient_indices, orbit_shifts = device_get(
        (periodic.lifted_offsets, periodic.orbit_indices, periodic.orbit_shifts)
    )
    exponent_rows = []
    exact = []
    numerical = []
    indices = []
    unique: dict[tuple[int, ...], int] = {}
    for node, orbit in enumerate(owners.orbits):
        root = int(owners.representatives[orbit])
        degree = int(owners.degree[node])
        if degree != int(owners.degree[root]):
            raise ValueError(
                "PeriodicCellGeometrySource.node_orbits must preserve entity degree; got a different representative degree"
            )
        start, stop = int(offsets[degree]), int(offsets[degree + 1])
        quotient = quotient_indices[start:stop]
        anchors = orbit_shifts[start:stop]
        copy, base = int(owners.entity[node]), int(owners.entity[root])
        if quotient[copy] != quotient[base]:
            raise ValueError(
                "PeriodicCellGeometrySource.node_orbits must preserve winding entity identity; got unrelated entities"
            )
        exponent = tuple(
            (int(a) - int(b)) % order if order else int(a) - int(b)
            for a, b, order in zip(anchors[copy], anchors[base], orders, strict=True)
        )
        exponent_rows.append(exponent)
        if exponent not in unique:
            unique[exponent] = len(exact)
            matrix = exact_periodic_element(generators, orders, exponent, budget=budget)
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
    return _PeriodicNodeActions(
        tuple(exponent_rows), tuple(exact), tuple(numerical), tuple(indices)
    )


class PeriodicCellGeometrySource(StrictModule, NonTrainableState):
    """Bind FE nodal numbering to authored group expressions.

    Numerical image coefficients are not independent parameters. Representative
    coefficients remain dynamic, rational coefficients are prepared on the host,
    and binary64 execution retains a bounded coefficient error.
    """

    source_mesh: cell_mesh_owner.CellMesh
    source_geometry: imported_cell_geometry.CellGeometrySpec
    numbering: fem_owner.FiniteElementDofMap
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

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        geometry: imported_cell_geometry.CellGeometrySpec,
        /,
        *,
        active_indices: ArrayLike | None = None,
        representative_coordinates: ArrayLike | None = None,
        numbering: fem_owner.FiniteElementDofMap | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        with ExitStack() as stack:
            periodic = mesh.periodic_topology
            if (
                periodic is None
                or geometry.periodic_source is not None
                or geometry.exact_source is not None
            ):
                raise ValueError(
                    "PeriodicCellGeometrySource.source_geometry must be an independent nodal carrier on periodic_topology; got missing topology or a dependent source"
                )
            if budget is None:
                work_upper, bytes_upper, queries_upper = (
                    periodic_topology_owner.periodic_preparation_bounds(
                        periodic.cell,
                        geometry.coordinates.shape[0],
                        np.asarray(periodic.orbit_shifts, dtype=np.int64),
                    )
                )
                budget = stack.enter_context(
                    owned_periodic_preparation(
                        work_upper, bytes_upper, queries_upper=queries_upper
                    )
                )
            elements, routes, _ = geometry.resolve(mesh, budget=budget)
            resolved_elements: list[reference_types_owner.FiniteElementSpec] = []
            for element in elements:
                match element:
                    case reference_types_owner.FiniteElementSpec(
                        representation="point_value", mapping="identity", value_shape=()
                    ):
                        resolved_elements.append(element)
                    case _:
                        raise ValueError(
                            "PeriodicCellGeometrySource.source_geometry.elements must be scalar identity-mapped point-value elements; got another coordinate element"
                        )
            resolved = tuple(resolved_elements)
            dofs = (
                fem_owner.FiniteElementDofMap(mesh, resolved, coordinate_spec=geometry)
                if numbering is None
                else numbering
            )
            owners = _prepare_periodic_node_owners(mesh, geometry, resolved, routes, dofs)
            representatives, orbits = owners.representatives, owners.orbits
            count = geometry.coordinates.shape[0]
            actions = _prepare_periodic_node_actions(periodic, owners, budget=budget)
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
                    "PeriodicCellGeometrySource.active_indices must index source coordinate rows; got invalid source indices"
                )
            values = (
                np.asarray(geometry.coordinates)[representatives]
                if representative_coordinates is None
                else np.asarray(representative_coordinates, dtype=np.float64)
            )
            if values.shape != (
                dofs.global_dof_count,
                mesh.ambient_dimension,
            ) or not np.all(np.isfinite(np.asarray(values))):
                raise ValueError(
                    f"PeriodicCellGeometrySource.representative_coordinates must cover finite quotient DOFs; got {values.shape}"
                )
            self.source_mesh, self.source_geometry, self.numbering = mesh, geometry, dofs
            self.representative_coordinates = device_put(values)
            self.representative_nodes = tuple(int(value) for value in representatives)
            self.node_orbits = tuple(int(value) for value in orbits)
            self.node_exponents = actions.exponents
            self.active_indices = tuple(int(value) for value in active)
            self.exact_isometries, self.numerical_isometries = (
                actions.exact,
                actions.numerical,
            )
            self.node_isometries = actions.indices
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicCellGeometrySource],
        value: PeriodicCellGeometrySource,
        /,
    ) -> PeriodicCellGeometrySource:
        """Rebuild numbering and authenticate the stored quotient coordinate law."""
        renewed = cls(
            value.source_mesh,
            value.source_geometry,
            active_indices=np.asarray(value.active_indices, dtype=np.int64),
            representative_coordinates=value.representative_coordinates,
            numbering=None,
        )
        if renewed.source_layout_id != value.source_layout_id:
            raise ValueError(
                "PeriodicCellGeometrySource.source_layout_id must bind restored numbering and topology; "
                f"got {value.source_layout_id!r}"
            )
        if (
            value.representative_nodes,
            value.node_orbits,
            value.node_exponents,
            value.exact_isometries,
            value.numerical_isometries,
            value.node_isometries,
        ) != (
            renewed.representative_nodes,
            renewed.node_orbits,
            renewed.node_exponents,
            renewed.exact_isometries,
            renewed.numerical_isometries,
            renewed.node_isometries,
        ):
            raise ValueError(
                "PeriodicCellGeometrySource.node_exponents must retain the restored quotient coordinate law; "
                "got different node or action rows"
            )
        if (
            value.numbering.dof_map_id != renewed.numbering.dof_map_id
            or array_tree_fingerprint(value.numbering)
            != array_tree_fingerprint(renewed.numbering)
        ):
            raise ValueError(
                "PeriodicCellGeometrySource.numbering must match restored FE numbering; "
                "got different numbering content"
            )
        return renewed

    @property
    def source_id(self) -> str:

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

    def preparation_bounds(self) -> tuple[int, int, int]:
        """Bound rebinding plus two rational source-bank preparation stages.

        Matrix actions use the admitted identification and retained exponent
        rows. Coefficient bits follow the stored rational matrices and the
        binary64 representative contract; no numerical values are transferred.
        """
        periodic = self.source_mesh.periodic_topology
        if periodic is None:
            raise ValueError(
                "PeriodicCellGeometrySource.source_mesh.periodic_topology must retain its identification; got None"
            )
        work, storage, _ = periodic_topology_owner.periodic_preparation_bounds(
            periodic.cell,
            len(self.node_orbits),
            np.asarray(self.node_exponents, dtype=np.int64),
        )
        dimension = self.representative_coordinates.shape[1]
        binary64_bits = (
            float(np.finfo(np.float64).smallest_subnormal)
            .as_integer_ratio()[1]
            .bit_length()
        )
        matrix_bits = max(
            abs(numerator).bit_length() + denominator.bit_length()
            for matrix in self.exact_isometries
            for row in matrix
            for numerator, denominator in row
        )
        coefficient_bits = (dimension + 1) * (
            binary64_bits + matrix_bits
        ) + dimension.bit_length()
        representatives = self.representative_coordinates.shape[0] * dimension
        values = len(self.active_indices) * dimension
        bank = coordinate_enclosure_owner.coordinate_polynomial_estimate(
            terms=representatives + 2 * values,
            dimension=dimension,
            degree=0,
            numerator_bits=coefficient_bits,
            denominator_bits=(representatives + 2 * values) * coefficient_bits,
        )
        bank_work = representatives + 2 * values * (dimension + 1)
        return (
            work + bank_work + bank.work_upper,
            storage + bank.bytes_upper,
            coefficient_bits,
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

    @checked
    def source_coordinates(
        self, *, budget: CoordinateEnclosureBudget
    ) -> CoordinateSourceBank:
        if isinstance(self.representative_coordinates, jax_core.Tracer):
            raise TypeError(
                "PeriodicCellGeometrySource.source_coordinates must receive host coefficients; got a JAX tracer"
            )
        values = tuple(
            rational_vector(row) for row in np.asarray(self.representative_coordinates)
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

    @checked
    def reindexed(
        self, indices: ArrayLike, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> PeriodicCellGeometrySource:

        indices_ = np.asarray(indices)
        if (
            indices_.ndim != 1
            or not np.issubdtype(indices_.dtype, np.integer)
            or np.any(indices_ < 0)
            or np.any(indices_ >= len(self.active_indices))
        ):
            raise ValueError(
                "PeriodicCellGeometrySource.active_indices must index current source rows; got invalid reindexing"
            )
        return PeriodicCellGeometrySource(
            self.source_mesh,
            self.source_geometry,
            active_indices=np.asarray(
                [self.active_indices[int(index)] for index in indices_], dtype=np.int64
            ),
            representative_coordinates=self.representative_coordinates,
            numbering=self.numbering,
            budget=budget,
        )

    @checked
    def rebound(
        self,
        coordinates: ArrayLike,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PeriodicCellGeometrySource:

        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (
            len(self.active_indices),
            self.representative_coordinates.shape[1],
        ):
            raise ValueError(
                f"PeriodicCellGeometrySource.rebound.coordinates must preserve coefficient view shape; got {values.shape}"
            )
        positions = {node: position for position, node in enumerate(self.active_indices)}
        roots = [
            orbit
            for orbit, node in enumerate(self.representative_nodes)
            if node in positions
        ]
        rows = [positions[self.representative_nodes[orbit]] for orbit in roots]
        representatives = np.asarray(self.representative_coordinates).copy()
        representatives[np.asarray(roots, dtype=np.int64)] = values[
            np.asarray(rows, dtype=np.int64)
        ]
        return PeriodicCellGeometrySource(
            self.source_mesh,
            self.source_geometry,
            active_indices=np.asarray(self.active_indices, dtype=np.int64),
            representative_coordinates=representatives,
            numbering=self.numbering,
            budget=budget,
        )
