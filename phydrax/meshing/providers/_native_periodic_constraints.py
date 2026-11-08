#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ...discretization._cell_mesh import CellMesh
from ...discretization._periodic_exact import (
    exact_periodic_element,
    identification_orders,
    reduce_periodic_exponents,
)
from ...discretization._periodic_topology import _entity_vertex_rows
from ...meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
from ...meshing._controls import PeriodicConstraint
from ...meshing._periodic_constraint import bind_periodic_entity_pairs
from ...meshing._sizing import UniformSizeControl
from ...typing import checked


def _entity_centroids(mesh: CellMesh, degree: int, /) -> np.ndarray:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    rows = _entity_vertex_rows(mesh)[degree]
    return np.stack(
        tuple(
            np.mean(coordinates[np.asarray(row, dtype=np.int64)], axis=0) for row in rows
        )
    )


@checked
def validate_periodic_mesh_constraints(
    mesh: CellMesh,
    source_id: str,
    source_revision: str,
    constraints: tuple[PeriodicConstraint, ...],
    /,
) -> None:
    """Validate authored pair scopes from integer quotient IDs and windings."""
    if mesh.periodic_topology is None:
        raise ValueError(
            "Periodic source constraints require an attached quotient descriptor."
        )
    topology = mesh.periodic_topology
    topology.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    orbit_indices = np.asarray(topology.orbit_indices, dtype=np.int64)
    orbit_orientations = np.asarray(topology.orbit_orientations, dtype=np.int64)
    orbit_shifts = np.asarray(topology.orbit_shifts, dtype=np.int64)
    orders = identification_orders(topology.cell)
    for constraint in constraints:
        if not isinstance(constraint, PeriodicConstraint):
            raise TypeError("Periodic constraints must be PeriodicConstraint values.")
        source_scope, target_scope = constraint.source_scope, constraint.target_scope
        degree = int(source_scope.entity_dimension)
        if (
            (source_scope.source_id, source_scope.source_revision)
            != (source_id, source_revision)
            or (target_scope.source_id, target_scope.source_revision)
            != (source_id, source_revision)
            or target_scope.entity_dimension != degree
            or not 0 <= degree <= mesh.topological_dimension
            or source_scope.entity_set_id != mesh.entity_set(degree).entity_set_id
            or target_scope.entity_set_id != mesh.entity_set(degree).entity_set_id
        ):
            raise ValueError(
                "Periodic constraint scopes must select the bound source entity set and revision."
            )
        lifted_begin, lifted_end = map(int, lifted_offsets[degree : degree + 2])
        quotient_begin, quotient_end = map(int, quotient_offsets[degree : degree + 2])
        lifted_ids = topology.lifted_entity_ids[degree]
        if lifted_ids is None:
            raise ValueError(
                "Periodic source entity IDs are absent from the quotient descriptor."
            )
        ids = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
        if not np.array_equal(ids, np.asarray(lifted_ids, dtype=np.int64)):
            raise ValueError(
                "Periodic source entity IDs do not match the quotient descriptor."
            )
        quotient_ids = np.asarray(
            topology.quotient.entity_sets[degree].entity_ids, dtype=np.int64
        )
        quotient_rows = orbit_indices[lifted_begin:lifted_end] - quotient_begin
        entity_quotient_ids = quotient_ids[quotient_rows]
        entity_windings = orbit_shifts[lifted_begin:lifted_end]
        entity_orientations = orbit_orientations[lifted_begin:lifted_end]
        source_ids = np.asarray(source_scope.entity_ids, dtype=np.int64)
        target_ids = np.asarray(target_scope.entity_ids, dtype=np.int64)
        row_by_id = {int(identifier): row for row, identifier in enumerate(ids)}
        if any(
            int(identifier) not in row_by_id for identifier in (*source_ids, *target_ids)
        ):
            raise ValueError(
                "Periodic constraint scope references an absent lifted entity."
            )
        source_rows = np.asarray(
            [row_by_id[int(identifier)] for identifier in source_ids], dtype=np.int64
        )
        target_rows = np.asarray(
            [row_by_id[int(identifier)] for identifier in target_ids], dtype=np.int64
        )
        source_quotient = entity_quotient_ids[source_rows]
        target_to_source = np.empty((target_ids.size,), dtype=np.int64)
        source_signs = np.empty((target_ids.size,), dtype=np.int32)
        transform = np.asarray(constraint.transform, dtype=np.float64)
        for position, target_row in enumerate(target_rows):
            candidates = source_rows[source_quotient == entity_quotient_ids[target_row]]
            matches = []
            for source_row in candidates:
                exponents = reduce_periodic_exponents(
                    entity_windings[target_row] - entity_windings[source_row], orders
                )
                element = np.asarray(
                    exact_periodic_element(topology.cell, exponents), dtype=np.float64
                )
                if element.shape == transform.shape and np.allclose(
                    element, transform, rtol=0.0, atol=constraint.tolerance
                ):
                    matches.append(int(source_row))
            if len(matches) != 1:
                raise ValueError(
                    "Periodic entity pairing is not uniquely determined by quotient ID and winding."
                )
            source_row = matches[0]
            target_to_source[position] = ids[source_row]
            source_signs[position] = int(entity_orientations[target_row]) * int(
                entity_orientations[source_row]
            )
        coordinates = _entity_centroids(mesh, degree)
        bind_periodic_entity_pairs(
            constraint,
            source_ids,
            target_ids,
            target_to_source,
            source_coordinates=coordinates[source_rows],
            target_coordinates=coordinates[target_rows],
            orientations=None if degree == 0 else source_signs,
        )


@checked
def validate_periodic_source_controls(
    source_id: str,
    source_revision: str,
    source_entity_ids: np.ndarray,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> None:
    """Refuse controls whose geometric/source authority is absent on this route."""
    if (
        specification.protected_features
        or specification.region_controls
        or specification.patch_controls
        or specification.periodic_constraints
        or specification.layer_controls
        or (
            isinstance(specification, VolumeMeshingSpec)
            and (specification.region_seeds or specification.hole_seeds)
        )
    ):
        raise ValueError(
            "Periodic Delaunay point-orbit generation does not yet own protected, material, patch, layered, or additional source-pair controls."
        )
    domain_ids = np.asarray(source_entity_ids, dtype=np.int64)
    if (
        domain_ids.ndim != 1
        or np.any(domain_ids < 0)
        or np.unique(domain_ids).size != domain_ids.size
    ):
        raise ValueError("Periodic source entity IDs must be unique nonnegative rows.")
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if len(controls) != len(specification.size_controls):
        raise ValueError(
            "Periodic Delaunay point-source adaptation supports only uniform controls."
        )
    for control in controls:
        scope = control.scope
        if (
            scope.source_id != source_id
            or scope.source_revision != source_revision
            or scope.entity_dimension != 0
        ):
            raise ValueError(
                "Periodic size controls must select source vertices at the bound revision."
            )
        ids = np.asarray(scope.entity_ids, dtype=np.int64)
        if np.setdiff1d(ids, domain_ids).size:
            raise ValueError(
                "Periodic size control selects an entity outside its authored point source."
            )


__all__ = ["validate_periodic_mesh_constraints", "validate_periodic_source_controls"]
