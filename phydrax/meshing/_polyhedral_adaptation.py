#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native polyhedral edits with rational source geometry and periodic quotients.

Splitting/agglomeration retain source restrictions and scientific orbits.
Regeneration admits a new power source through full common-refinement coverage
and unchanged domain/material/periodic controls. Scalar nodal
stencils are restricted to known source-vertex/edge functionals; geometric convex
actions do not evaluate a general virtual-element face or interior field.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from enum import StrEnum
from fractions import Fraction
from typing import assert_never, NamedTuple

import equinox as eqx
import jax
import numpy as np
from jax import Array

import phydrax._meshcore as _owner_phydrax__meshcore
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._exact_power_geometry as _dependency_phydrax_discretization__exact_power_geometry
import phydrax.discretization._periodic_topology as _owner_phydrax_discretization__periodic_topology
import phydrax.geometry._periodic_power_preparation as periodic_power_preparation_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.geometry._polygon_triangulation as _owner_phydrax_geometry__mesh_certificates
import phydrax.geometry._supermesh as _dependency_phydrax_geometry__supermesh
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._contracts as contracts_owner
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._host_banks as host_banks_owner
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage
import phydrax.meshing._periodic as _owner_phydrax_meshing__periodic
import phydrax.meshing._periodic_constraint as periodic_constraint_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._polyhedral_adaptation_shared as polyhedral_shared_owner
import phydrax.meshing._polyhedral_generation as _dependency_phydrax_meshing__polyhedral_generation
import phydrax.meshing._polyhedral_generation as _owner_phydrax_meshing__polyhedral_generation
import phydrax.meshing._polyhedral_plane_restriction as plane_restriction_owner
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit
import phydrax.meshing._topology_edit as _owner_phydrax_meshing__topology_edit
import phydrax.meshing.providers._native_polyhedral as _owner_phydrax_meshing_providers__native_polyhedral
import phydrax.meshing.providers._native_sources as _owner_phydrax_meshing_providers__native_sources
from phydrax.meshing._failures import MeshingFailureCategory

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    clip_box_halfspaces,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import polyhedral_connectivity
from ..geometry._rational import rational_vector
from ..typing import checked, ConvertibleToArray, parse


class PolyhedralAdaptationOperation(StrEnum):
    PLANE_SPLIT = "plane_split"
    AGGLOMERATE = "agglomerate"
    REGENERATE = "regenerate"


class _PolyhedralRequestPreparation(StrictModule):
    planes_: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    sites_: np.ndarray | None
    weights_: np.ndarray | None
    schedule_: _dependency_phydrax_meshing__polyhedral_generation.NativePolyhedralSchedule
    constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...]


def _prepare_polyhedral_request(
    operation: PolyhedralAdaptationOperation,
    planes: ConvertibleToArray,
    agglomerations: tuple[tuple[int, ...], ...],
    sites: ConvertibleToArray | None,
    weights: ConvertibleToArray | None,
    schedule: _dependency_phydrax_meshing__polyhedral_generation.NativePolyhedralSchedule
    | None,
    coordinate_tolerance: float,
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
) -> _PolyhedralRequestPreparation:
    constraints = tuple(periodic_constraints)
    for value in constraints:
        match value:
            case periodic_constraint_owner.PeriodicConstraint():
                pass
            case _:
                raise TypeError(
                    "PolyhedralMeshAdaptation.periodic_constraints must contain PeriodicConstraint values; got another member kind"
                )
    if constraints and operation is not PolyhedralAdaptationOperation.REGENERATE:
        raise ValueError(
            "PolyhedralMeshAdaptation.periodic_constraints must accompany site regeneration; got another operation"
        )
    planes_ = np.asarray(planes, dtype=np.float64).reshape((-1, 4))
    groups = tuple(tuple(sorted(group)) for group in agglomerations)
    if any(
        len(group) < 2
        or len(set(group)) != len(group)
        or any(identifier < 0 for identifier in group)
        for group in groups
    ):
        raise ValueError(
            "PolyhedralMeshAdaptation.agglomerations must contain at least two distinct nonnegative IDs per group; got invalid groups"
        )
    if len({identifier for group in groups for identifier in group}) != sum(
        map(len, groups)
    ):
        raise ValueError(
            "PolyhedralMeshAdaptation.agglomerations must be disjoint; got overlapping groups"
        )
    if not np.all(np.isfinite(planes_)) or np.any(
        np.linalg.norm(planes_[:, :3], axis=1) == 0
    ):
        raise ValueError(
            "PolyhedralMeshAdaptation.planes must contain finite nonzero normals and finite offsets; got invalid planes"
        )
    if not np.isfinite(coordinate_tolerance) or coordinate_tolerance < 0:
        raise ValueError(
            "PolyhedralMeshAdaptation.coordinate_tolerance must be finite and nonnegative; got an invalid tolerance"
        )
    sites_ = None if sites is None else np.asarray(sites, dtype=np.float64)
    weights_ = None if weights is None else np.asarray(weights, dtype=np.float64)
    if sites_ is not None and (
        sites_.ndim != 2
        or sites_.shape[1] != 3
        or sites_.shape[0] == 0
        or not np.all(np.isfinite(sites_))
    ):
        raise ValueError(
            "PolyhedralMeshAdaptation.sites must have finite nonempty (N, 3) coordinates; got invalid sites"
        )
    if weights_ is not None and (
        sites_ is None
        or weights_.shape != (sites_.shape[0],)
        or not np.all(np.isfinite(weights_))
    ):
        raise ValueError(
            "PolyhedralMeshAdaptation.weights must contain one finite value per site; got invalid weights"
        )
    schedule_ = (
        _dependency_phydrax_meshing__polyhedral_generation.NativePolyhedralSchedule()
        if schedule is None
        else schedule
    )
    return _PolyhedralRequestPreparation(
        planes_, groups, sites_, weights_, schedule_, constraints
    )


class PolyhedralMeshAdaptation(StrictModule, NonTrainableState):
    """Request plane splitting, connected-cell agglomeration, or site regeneration.

    Plane subdivision propagates each plane through the source, so shared faces
    have matching subdivisions. Site insertion/removal uses successor site and
    weight arrays; site identity is not a nodal stencil.
    """

    operation: PolyhedralAdaptationOperation = eqx.field(static=True)
    planes: Array
    agglomerations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    complex_: plc_source_owner.PiecewiseLinearComplex | None
    sites: Array | None
    weights: Array | None
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...]
    schedule: _dependency_phydrax_meshing__polyhedral_generation.NativePolyhedralSchedule = eqx.field(
        static=True
    )
    coordinate_tolerance: float = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        operation: PolyhedralAdaptationOperation,
        *,
        planes: ConvertibleToArray = (),
        agglomerations: tuple[tuple[int, ...], ...] = (),
        complex_: plc_source_owner.PiecewiseLinearComplex | None = None,
        sites: ConvertibleToArray | None = None,
        weights: ConvertibleToArray | None = None,
        schedule: _dependency_phydrax_meshing__polyhedral_generation.NativePolyhedralSchedule
        | None = None,
        coordinate_tolerance: float = 0.0,
        periodic_constraints: tuple[
            periodic_constraint_owner.PeriodicConstraint, ...
        ] = (),
    ) -> None:
        operation = parse(operation, PolyhedralAdaptationOperation, "operation")
        prepared = _prepare_polyhedral_request(
            operation,
            planes,
            agglomerations,
            sites,
            weights,
            schedule,
            coordinate_tolerance,
            periodic_constraints,
        )
        planes_ = prepared.planes_
        groups = prepared.groups
        sites_ = prepared.sites_
        weights_ = prepared.weights_
        schedule_ = prepared.schedule_
        constraints = prepared.constraints
        match operation:
            case PolyhedralAdaptationOperation.PLANE_SPLIT:
                valid = (
                    planes_.shape[0] > 0
                    and not groups
                    and complex_ is None
                    and sites_ is None
                )
            case PolyhedralAdaptationOperation.AGGLOMERATE:
                valid = (
                    bool(groups)
                    and planes_.shape[0] == 0
                    and complex_ is None
                    and sites_ is None
                )
            case PolyhedralAdaptationOperation.REGENERATE:
                valid = (
                    complex_ is not None
                    and sites_ is not None
                    and not groups
                    and planes_.shape[0] == 0
                )
            case invalid:
                assert_never(invalid)
        if not valid:
            raise ValueError(
                "PolyhedralMeshAdaptation.operation must have only its supported controls; got missing or conflicting controls"
            )
        self.operation = operation
        _host_planes = np.asarray(planes_)
        self.agglomerations = tuple(sorted(groups))
        self.complex_ = complex_
        _host_sites = None if sites_ is None else np.asarray(sites_)
        _host_weights = None if weights_ is None else np.asarray(weights_)
        (
            self.planes,
            self.sites,
            self.weights,
        ) = jax.device_put(
            (
                _host_planes,
                _host_sites,
                _host_weights,
            )
        )
        self.periodic_constraints = constraints
        self.schedule = schedule_
        self.coordinate_tolerance = float(coordinate_tolerance)
        self.request_id = canonical_fingerprint(
            {
                "kind": "polyhedral-adaptation-request",
                "operation": operation.value,
                "planes": array_tree_fingerprint(planes_),
                "agglomerations": self.agglomerations,
                "source": None if complex_ is None else complex_.complex_id,
                "sites": None if sites_ is None else array_tree_fingerprint(sites_),
                "weights": None if weights_ is None else array_tree_fingerprint(weights_),
                "schedule": asdict(schedule_),
                "coordinate_tolerance": self.coordinate_tolerance,
                "periodic_constraints": tuple(
                    value.constraint_id for value in constraints
                ),
            }
        )


class PolyhedralAdaptationEvidence(NamedTuple):
    operation: PolyhedralAdaptationOperation
    source_cells: int
    target_cells: int
    plane_construction_residual: float
    common_refinement_id: str
    changed_cell_ids: np.ndarray
    construction_id: str | None
    site_points: np.ndarray | None
    site_weights: np.ndarray | None
    site_parents: np.ndarray | None
    cell_sites: np.ndarray | None
    cell_components: np.ndarray | None
    cell_regions: np.ndarray | None
    face_facets: np.ndarray | None
    feature_edge_sources: np.ndarray | None

    @property
    def identity_work(self) -> int:
        """Host reads and streamed hashes of every bank ``evidence_id`` visits."""
        banks = (
            self.changed_cell_ids,
            self.site_points,
            self.site_weights,
            self.site_parents,
            self.cell_sites,
            self.cell_components,
            self.cell_regions,
            self.face_facets,
            self.feature_edge_sources,
        )
        return 256 + 2 * sum(bank.size for bank in banks if bank is not None)

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "polyhedral-adaptation-evidence",
                "operation": self.operation.value,
                "source_cells": self.source_cells,
                "target_cells": self.target_cells,
                "plane_construction_residual": self.plane_construction_residual,
                "common_refinement": self.common_refinement_id,
                "changed_cells": array_tree_fingerprint(self.changed_cell_ids),
                "construction": self.construction_id,
                "ancestry": array_tree_fingerprint(
                    (
                        self.site_points,
                        self.site_weights,
                        self.site_parents,
                        self.cell_sites,
                        self.cell_components,
                        self.cell_regions,
                        self.face_facets,
                        self.feature_edge_sources,
                    )
                ),
            }
        )


class PolyhedralAdaptationOutcome(NamedTuple):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    common_refinement: _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement
    evidence: PolyhedralAdaptationEvidence
    construction: (
        _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction | None
    )
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec


@checked
def regenerated_polyhedral_associations(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    request: PolyhedralMeshAdaptation,
    construction: _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction,
    target: CellMesh,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    """Reissue source strata from restricted-diagram carrier evidence."""

    if request.complex_ is None or not source.associations:
        raise ValueError(
            "polyhedral_adaptation.source_associations must bind the request PLC; got missing PLC or associations"
        )
    certificate = source.certification
    if (
        certificate is None
        or not certificate.passed
        or certificate.request.domain is None
    ):
        raise ValueError(
            "polyhedral_adaptation.source.certification must retain current PLC coverage; got missing or stale coverage"
        )
    if certificate.request.domain.domain_id != construction.domain.domain_id:
        raise ValueError(
            "polyhedral_adaptation.complex_ must retain the accepted PLC domain; got a changed domain"
        )
    first = source.associations[0]
    for association in source.associations:
        if (
            association.association_kind
            is not association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != first.source_id
            or association.source_revision != first.source_revision
        ):
            raise ValueError(
                "polyhedral_adaptation.source_associations must share one represented PLC revision; got mismatched bindings"
            )
        dimensions = [
            dimension
            for dimension in range(4)
            if source.mesh.entity_set(dimension).entity_set_id
            == association.target_entity_set_id
        ]
        if len(dimensions) != 1:
            raise ValueError(
                "polyhedral_adaptation.source_associations must bind mesh entities; got an unbound association"
            )
        # Four entity-set comparisons, then one visit per association row for
        # target membership and each of the resolved/ambiguous scans.
        budget.reserve(4 + 3 * np.asarray(association.resolved).size)
        association.validate_target(source.mesh.entity_set(dimensions[0]))
        if not np.all(np.asarray(association.resolved)) or np.any(
            np.asarray(association.ambiguous)
        ):
            raise ValueError(
                "polyhedral_adaptation.source_associations must resolve unambiguous strata; got unresolved or ambiguous rows"
            )
    authority = _owner_phydrax_meshing_providers__native_sources.NativePlcSource(
        request.complex_, first.source_id, first.source_revision
    )
    associations = (
        _owner_phydrax_meshing_providers__native_polyhedral.organization(
            authority,
            None,
            replace(construction, mesh=target),
            request.schedule.feature_tolerance,
            budget=budget,
        )
    ).associations
    return associations


def _stratum_overlap(
    source_points: np.ndarray,
    target_points: np.ndarray,
    dimension: int,
    work: list[int],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    # Only the Boolean decision escapes. Coordinate gathers, triangulation
    # lists and exact predicate intermediates end at this boundary.
    with budget.temporary_scope():
        budget.reserve(
            0, 16 * 256 + 512 * (len(source_points) + len(target_points))
        )
        # One logical visit per profiled coordinate scalar.
        budget.reserve(source_points.size + target_points.size)
        bits = max(
            (value.numerator.bit_length() + value.denominator.bit_length()
             if isinstance(value, Fraction)
             else planar_primitives_owner.binary64_fraction_bits(float(value))
             for points in (source_points, target_points)
             for row in points for value in row),
            default=1,
        )
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            0, 16, 0, 8 * bits + 4, budget=budget
        )
        return _decide_stratum_overlap(
            source_points, target_points, dimension, work, bits=bits, budget=budget
        )


def _decide_stratum_overlap(
    source_points: np.ndarray,
    target_points: np.ndarray,
    dimension: int,
    work: list[int],
    /,
    *,
    bits: int,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    """Deciding surface/curve ancestry requires positive physical overlap."""
    if dimension == 0:
        if source_points.shape == target_points.shape:
            budget.reserve(2 * source_points.size)
        return np.array_equal(source_points, target_points)
    # polygon_bounds charges two coefficient units per scalar comparison.
    # These are the actual two min/max reductions for each input and the
    # three componentwise comparisons of their resulting bounding boxes; the
    # binary64 reductions visit the same comparisons.
    comparisons = source_points.shape[1] * (
        2 * (len(source_points) - 1) + 2 * (len(target_points) - 1) + 3
    )
    budget.reserve(2 * comparisons)
    if np.any(
        np.maximum(np.min(source_points, axis=0), np.min(target_points, axis=0))
        > np.minimum(np.max(source_points, axis=0), np.max(target_points, axis=0))
    ):
        return False
    if dimension == 1:
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            0, 48, 0, 16 * bits + 5, budget=budget
        )
        a, b = source_points
        # rational_points' existing traversal/lift rate is two per coefficient;
        # each direction/delta difference adds one coefficient operation.
        budget.reserve(2 * len(a))
        origin = tuple(
            value if isinstance(value, Fraction) else Fraction(float(value))
            for value in a
        )
        budget.reserve(3 * len(b))
        direction = tuple(
            (value if isinstance(value, Fraction) else Fraction(float(value))) - previous
            for value, previous in zip(b, origin, strict=True)
        )
        for axis, component in enumerate(direction):
            budget.reserve(1)  # The existing exact scalar zero-test unit.
            if component != 0:
                break
        else:
            raise StopIteration
        parameters = []
        for point in target_points:
            budget.reserve(3 * len(point))
            delta = tuple(
                (value if isinstance(value, Fraction) else Fraction(float(value)))
                - previous
                for value, previous in zip(point, origin, strict=True)
            )
            budget.reserve(1)
            parameter = delta[axis] / direction[axis]
            for value, component in zip(delta, direction, strict=True):
                # One product and the canonical two-unit exact comparison.
                budget.reserve(3)
                if value != parameter * component:
                    return False
            parameters.append(parameter)
        # Each parameter min/max visits N-1 comparisons, followed by the two
        # endpoint comparisons and the final positive-overlap comparison.
        budget.reserve(2 * (2 * (len(parameters) - 1) + 3))
        return max(Fraction(0), min(parameters)) < min(Fraction(1), max(parameters))

    triangles = []
    for points in (source_points, target_points):
        loops = np.arange(points.shape[0], dtype=np.int64)[None, :]
        indices, _, nonplanar, undecided = (
            _owner_phydrax_geometry__mesh_certificates.loop_triangles(
                points, loops, work[0], budget=budget
            )
        )
        if np.any(nonplanar | undecided):
            raise polyhedral_shared_owner.PolyhedralAdaptationError(
                "polyhedral_adaptation.source_stratum must contain decided planar faces; got nonplanar or unresolved faces"
            )
        triangles.append(points[indices])
    pairs = triangles[0].shape[0] * triangles[1].shape[0]
    if pairs > work[0]:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.maximum_geometry_queries must admit stratum overlap; got a query budget refusal",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    work[0] -= pairs
    if source_points.dtype == object or target_points.dtype == object:
        for first in triangles[0]:
            for second in triangles[1]:
                budget.reserve_geometry_queries(1)
                a, b = (
                    planar_primitives_owner.rational_points(first, budget=budget),
                    planar_primitives_owner.rational_points(second, budget=budget),
                )
                key = planar_primitives_owner.plane_key(a, budget=budget)
                other = planar_primitives_owner.plane_key(b, budget=budget)
                if (
                    key is not None
                    and other is not None
                    and key[0] == other[0]
                    and planar_primitives_owner.intersection(
                        planar_primitives_owner.project(a, key[1], budget=budget),
                        planar_primitives_owner.project(b, key[1], budget=budget),
                        budget=budget,
                    )
                    > 0
                ):
                    return True
        return False
    for first in triangles[0]:
        for second in triangles[1]:
            # One native triangle-pair predicate: one query and one visit,
            # matching the exact branch's per-pair query ticket.
            budget.reserve_geometry_queries(1)
            budget.reserve(1)
            native = _owner_phydrax__meshcore.triangle_intersections(
                first[None, :], second[None, :]
            )
            counts = native.point_counts
            status = native.status
            if np.any(status != int(_owner_phydrax__meshcore.MeshcoreStatus.OK)):
                raise polyhedral_shared_owner.PolyhedralAdaptationError(
                    "polyhedral_adaptation.source_stratum must have resolved overlap predicates; got unresolved status"
                )
            if counts[0] >= 3:
                return True
    return False


def _stratum_pass_work(mesh: CellMesh, dimension: int, /) -> int:
    """Logical visits of one sorted key or packed-row pass over one stratum.

    Each row and each of its connectivity entries is visited once per
    comparison level of the sort/search, matching ``key_rows``' sorted lookup.
    """
    c = mesh.connectivity
    rows = int(np.asarray(mesh.entity_set(dimension).entity_ids).size)
    match dimension:
        case 0 | 3:
            entries = rows
        case 1:
            entries = 2 * rows
        case _:
            entries = (
                int(np.asarray(c.face_vertex_values).size)
                if isinstance(c, PolyhedralConnectivity)
                else 3 * rows
            )
    return (rows + entries) * max(1, (rows + 1).bit_length())

def _stratum_vertex_rows(mesh: CellMesh, dimension: int, /) -> tuple[np.ndarray, ...]:
    """Keep packed cyclic face order; lineage keys deliberately discard it."""
    c = mesh.connectivity
    match c:
        case PolyhedralConnectivity():
            pass
        case _:
            raise TypeError(
                "polyhedral_adaptation.connectivity must use packed strata; got another connectivity kind"
            )
    identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if dimension == 0:
        rows = entity_keys_owner.key_rows(
            np.asarray(mesh.vertex_global_ids)[:, None], identifiers[:, None]
        )
        return tuple(row[None] for row in rows)
    if dimension == 1:
        rows = entity_keys_owner.key_rows(
            np.asarray(c.edge_global_ids)[:, None], identifiers[:, None]
        )
        edges = np.asarray(c.edges, dtype=np.int64)
        return tuple(edges[row] for row in rows)
    rows = entity_keys_owner.key_rows(
        np.asarray(c.face_global_ids)[:, None], identifiers[:, None]
    )
    offsets, vertices = (
        np.asarray(c.face_vertex_offsets),
        np.asarray(c.face_vertex_values),
    )
    return tuple(vertices[offsets[row] : offsets[row + 1]] for row in rows)


@checked
def regenerated_polyhedral_relations(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    target: CellMesh,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    associations: tuple[association_state_owner.GeometryAssociation, ...],
    maximum_work_units: int,
    /,
    *,
    maximum_geometry_queries: int,
    target_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__topology_edit.CellTopologyEdit:
    with _polyhedral_exact_phase(
        scope, budget, maximum_work_units, budget.maximum_memory_bytes
    ):
        with _polyhedral_topology_phase(
            scope, target, periodic_source=source.mesh
        ) as (storage, bound):
            source_bound = _polyhedral_adaptation_storage_bound(
                _polyhedral_incidence(source.mesh, scope=scope)._replace(rank=0)
            )
            base = bound.construction + source_bound.construction
            storage.workspace.set_bound(base)
            result, relation_bytes = _build_regenerated_relations(
                source, target, edit, associations, maximum_work_units,
                maximum_geometry_queries=maximum_geometry_queries,
                target_geometry=target_geometry, scope=scope, budget=budget,
                storage=storage.workspace, base=base,
            )
            storage.retain(
                relation_bytes,
                *(leaf for leaf in jax.tree_util.tree_leaves(result)
                  if eqx.is_array(leaf)),
                record_count_upper=bound.records + source_bound.records,
            )
            return result


def _build_regenerated_relations(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    target: CellMesh,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    associations: tuple[association_state_owner.GeometryAssociation, ...],
    maximum_work_units: int,
    /,
    *,
    maximum_geometry_queries: int,
    target_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    storage: NativeHostStorageWorkspace,
    base: int,
) -> tuple[_dependency_phydrax_meshing__topology_edit.CellTopologyEdit, int]:
    """Carry stratum membership and refuse inconsistent partial scopes.

    Regenerated cells use measured physical overlap. Lower strata use source-entity
    identity rather than nearest coordinates or display names. Membership must
    agree among every deciding predecessor of a remeshed stratum.
    """
    with _polyhedral_exact_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        relations = list(edit.relations)
        work = 0
        relation_bytes = 3 * 4 * 256
        relation_storage = 0
        queries = [min(maximum_work_units, maximum_geometry_queries)]
        with _polyhedral_exact_phase(
            scope, budget, maximum_work_units, budget.maximum_memory_bytes
        ):
            source.geometry.resolve(source.mesh, budget=budget)
            target_geometry.resolve(target, budget=budget)
            source_coordinates = np.asarray(
                source.geometry.source_coordinates(budget=budget), dtype=object
            )
            target_coordinates = np.asarray(
                target_geometry.source_coordinates(budget=budget), dtype=object
            )
        for dimension in range(3):
            # entity_keys, _stratum_vertex_rows and the position key_rows
            # below are three sorted passes over each stratum.
            budget.reserve(
                3
                * (
                    _stratum_pass_work(source.mesh, dimension)
                    + _stratum_pass_work(target, dimension)
                )
            )
            old_keys, new_keys = (
                entity_keys_owner.entity_keys(source.mesh, dimension),
                entity_keys_owner.entity_keys(target, dimension),
            )
            old_ids = np.asarray(
                source.mesh.entity_set(dimension).entity_ids, dtype=np.int64
            )
            new_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
            source_vertices = _stratum_vertex_rows(source.mesh, dimension)
            target_vertices = _stratum_vertex_rows(target, dimension)
            previous = [
                value
                for value in source.associations
                if value.target_entity_set_id
                == source.mesh.entity_set(dimension).entity_set_id
            ]
            current = [
                value
                for value in associations
                if value.target_entity_set_id
                == target.entity_set(dimension).entity_set_id
            ]
            if not previous or not current:
                continue
            if len(previous) != 1 or len(current) != 1:
                raise ValueError(
                    "polyhedral_adaptation.source_associations must provide one association per mesh dimension; got conflicting dimensions"
                )
            old, new = previous[0], current[0]
            parent_rows, child_rows = [], []
            bank: dict[str, list[int]] = {}
            old_positions = entity_keys_owner.key_rows(
                old_ids[:, None], np.asarray(old.target_global_ids)[:, None]
            )
            new_positions = entity_keys_owner.key_rows(
                new_ids[:, None], np.asarray(new.target_global_ids)[:, None]
            )
            if np.any(old_positions < 0) or np.any(new_positions < 0):
                raise ValueError(
                    "polyhedral_adaptation.source_associations must name declared mesh entities; got unknown IDs"
                )
            # Bank insertion, the matched-multiplicity scan and one visit per
            # target identity row (including unmatched rows) below.
            budget.reserve(old_positions.size + 3 * new_positions.size)
            for identity, position in zip(
                old.source_entities.entity_ids(), old_positions.tolist(), strict=True
            ):
                bank.setdefault(identity, []).append(position)
            new_identities = new.source_entities.entity_ids()
            matched = sum(len(bank.get(identity, ())) for identity in new_identities)
            # Exact same-stratum identity multiplicities bound every possible
            # accepted relation before overlap predicates or list growth.
            relation_storage += matched * (
                512 + 16 * (old_keys.shape[1] + new_keys.shape[1])
            )
            storage.set_bound(base + relation_storage)
            work += old_positions.size + new_positions.size
            for row, identity in enumerate(new_identities):
                parents = bank.get(identity, [])
                work += len(parents)
                budget.reserve(len(parents))
                if work > maximum_work_units:
                    raise polyhedral_shared_owner.PolyhedralAdaptationError(
                        "polyhedral_adaptation.maximum_work_units must admit stratum lineage; got a work budget refusal",
                        category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    )
                child = int(new_positions[row])
                target_points = target_coordinates[target_vertices[child]]
                for parent in parents:
                    if queries[0] < 1:
                        raise polyhedral_shared_owner.PolyhedralAdaptationError(
                            "polyhedral_adaptation.maximum_geometry_queries must admit stratum lineage; got a query budget refusal",
                            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        )
                    queries[0] -= 1
                    budget.reserve_geometry_queries(1)
                    if _stratum_overlap(
                        source_coordinates[source_vertices[parent]],
                        target_points,
                        dimension,
                        queries,
                        budget=budget,
                    ):
                        parent_rows.append(parent)
                        child_rows.append(child)
            relation_bytes += len(parent_rows) * (
                8 * (old_keys.shape[1] + new_keys.shape[1]) + 4
            )
            relations[dimension] = (
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    dimension,
                    old_keys[np.asarray(parent_rows, dtype=np.int64)],
                    new_keys[np.asarray(child_rows, dtype=np.int64)],
                    np.full(
                        len(parent_rows),
                        int(
                            _dependency_phydrax_meshing__lineage.EntityLineageKind.SWAPPED_FROM
                        ),
                        dtype=np.int32,
                    ),
                )
            )
        return edit._replace(relations=tuple(relations)), relation_bytes


class _PolyhedralIncidence(NamedTuple):
    vertices: int
    cells: int
    loops: int
    corners: int
    squared_arities: int
    edges: int
    rank: int
    face_key_slots: int
    maximum_face_arity: int
    maximum_cell_corners: int
    split_squared_arities: int


def _polyhedral_incidence(
    mesh: CellMesh, /, *, scope: NativeExecutionScope
) -> _PolyhedralIncidence:
    """Read concrete packed incidence only after admitting its host copies."""
    c = _dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(mesh)
    with NativeHostStorageWorkspace(scope) as storage:
        storage.set_bound(
            4 * 256 + 8 * (
                c.cell_face_offsets.size + c.cell_face_values.size
                + c.face_vertex_offsets.size
                + c.cell_vertex_offsets.size
            )
        )
        scope.charge(work=c.cell_count + c.cell_face_values.size + c.face_count)
        co = np.asarray(c.cell_face_offsets)
        cf = np.asarray(c.cell_face_values)
        fo = np.asarray(c.face_vertex_offsets)
        vo = np.asarray(c.cell_vertex_offsets)
        corners = 0
        squares = 4 * c.edges.shape[0]
        maximum_face_arity = maximum_cell_corners = 0
        split_squares = 0
        for cell, (begin, end) in enumerate(zip(co[:-1], co[1:], strict=True)):
            cell_corners = 0
            for face in cf[int(begin):int(end)]:
                arity = int(fo[int(face) + 1]) - int(fo[int(face)])
                corners += arity
                cell_corners += arity
                split_squares += 8 * arity * arity
                maximum_face_arity = max(maximum_face_arity, arity)
            vertices = int(vo[cell + 1]) - int(vo[cell])
            squares += vertices * vertices
            cell_edges_upper = cell_corners // 2
            split_squares += 2 * (
                (vertices + cell_edges_upper) ** 2 + cell_edges_upper ** 2
            )
            maximum_cell_corners = max(maximum_cell_corners, cell_corners)
        for begin, end in zip(fo[:-1], fo[1:], strict=True):
            squares += (int(end) - int(begin)) ** 2
        split_squares += 24 * corners
        periodic = mesh.periodic_topology
        rank = 0 if periodic is None else periodic.vertex_shifts.shape[1]
        return _PolyhedralIncidence(
            mesh.coordinates.shape[0], c.cell_count, cf.size, corners,
            squares, c.edges.shape[0], rank,
            c.face_count * maximum_face_arity,
            maximum_face_arity, maximum_cell_corners,
            split_squares,
        )


class _PolyhedralStorageBound(NamedTuple):
    construction: int
    retained: int
    records: int


def _polyhedral_adaptation_storage_bound(
    incidence: _PolyhedralIncidence, /, *, split: bool = False
) -> _PolyhedralStorageBound:
    """Bound one concrete edit, not schedule-wide maximum-arity padding.

    A plane adds at most one vertex per source edge, two children per cell,
    two clipped copies per face occurrence and two section loops per cell.
    Clipping a loop has at most twice its original corners; both section loops
    together have at most twice the cell's original corner occurrences.
    """
    (v, c, f, k, q, e, rank, key_slots, face_arity, cell_corners,
     split_squares) = incidence
    if split:
        v, c, f, k, q = v + e, 2 * c, 2 * f + 2 * c, 6 * k, split_squares
        key_slots = f * max(2 * face_arity, cell_corners)
    construction, retained = (
        _dependency_phydrax_meshing__polyhedral_generation._polyhedral_mesh_storage_bound(
            v, c, f, k
        )
    )
    records = 512 + 64 * c
    if rank:
        periodic, periodic_retained = (
            _dependency_phydrax_meshing__polyhedral_generation._periodic_topology_storage_bound(
                v, c, f, k, q, rank
            )
        )
        construction += periodic
        retained += periodic_retained
        records += 16 * (v + k + f + c)
    # Source topology copies, loop tuples/views, identity/support mappings,
    # normal/offset rows, remapping banks, and per-loop assembly lists/sets.
    # Python index payload permits full signed-int64 scientific identities.
    construction += 256 * (32 + 8 * c + 4 * f) + 320 * (v + c + f + k)
    # entity_keys materializes padded, sorted and selected face-key banks.
    # Use the real face count and width, not the sum of corners as a proxy.
    construction += 8 * 8 * key_slots
    return _PolyhedralStorageBound(construction, retained, records)


def _polyhedral_geometry_aliases(
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
) -> tuple[object, ...]:
    """Retain newly authored geometry banks, not an entire provenance ancestry.

    Power witnesses own their exact vertices and coefficients through the same
    coordinate ledger. Parent recipe banks already retain their producer
    storage; only a new restriction head adds arrays on this edit.
    """
    source = polyhedral_shared_owner.bound_power_source(geometry)
    banks: tuple[object, ...] = (geometry.coordinates, *geometry.geometry_dofs)
    if isinstance(
        source,
        _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource,
    ):
        banks += (source.planes, source.vertex_parents, source.vertex_plane_ids)
    return banks


def _polyhedral_cross_key_storage(source: CellMesh, target: CellMesh, /) -> int:
    """Bound source/target face keys after their actual widths are unified.

    ID assignment and lineage concatenate, pad, unique-sort and select these
    banks. Sixteen simultaneous scalar banks cover both input sorts, both pads,
    combined rows, structured lexical sorting, uniqueness and lookup operands.
    """
    first = _dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(source)
    second = _dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(target)
    return (
        8 * 16 * max(first.maximum_face_arity, second.maximum_face_arity)
        * (first.face_count + second.face_count)
    )


@contextmanager
def _polyhedral_topology_phase(
    scope: NativeExecutionScope,
    mesh: CellMesh,
    /,
    *,
    split: bool = False,
    periodic_source: CellMesh | None = None,
    periodic_topology: bool = True,
    limits: contracts_owner.MeshingLimits | None = None,
) -> Iterator[tuple[
    _dependency_phydrax_meshing__polyhedral_generation._PolyhedralTopologyStorage,
    _PolyhedralStorageBound,
]]:
    incidence = _polyhedral_incidence(mesh, scope=scope)
    if periodic_source is not None and periodic_source.periodic_topology is not None:
        incidence = incidence._replace(
            rank=periodic_source.periodic_topology.vertex_shifts.shape[1]
        )
    if not periodic_topology:
        incidence = incidence._replace(rank=0)
    bound = _polyhedral_adaptation_storage_bound(incidence, split=split)
    # A periodic descriptor retains its exact source ancestry. Each restriction
    # adds three array leaves; a linear-action node plus its preparation has
    # fewer than 64 schema leaves. Admit finalizer/traversal records from the
    # actual chain depth rather than assuming one generation epoch.
    depth = 0
    for owner in (mesh, periodic_source):
        if owner is None or owner.periodic_topology is None:
            continue
        geometry = owner.periodic_topology._actual_geometry
        if geometry is None:
            continue
        node = polyhedral_shared_owner.bound_power_source(geometry)
        current_depth = 0
        while isinstance(node, (
            _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource,
            _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource,
        )):
            scope.charge(work=1)
            current_depth += 1
            node = node.parent
        depth = max(depth, current_depth + int(split))
    extra_records = 64 * depth
    bound = bound._replace(
        construction=bound.construction + 512 * extra_records,
        records=bound.records + extra_records,
    )
    with _dependency_phydrax_meshing__polyhedral_generation._PolyhedralTopologyStorage(scope) as storage:
        storage.workspace.set_bound(bound.construction)
        vertices, cells, loops, corners = (
            incidence.vertices, incidence.cells, incidence.loops, incidence.corners
        )
        if split:
            vertices += incidence.edges
            cells, loops, corners = 2 * cells, 2 * loops + 2 * cells, 6 * corners
        if limits is not None:
            _dependency_phydrax_meshing__polyhedral_generation._admit_polyhedral_mesh_capacity(
                vertices, cells, loops, corners, limits
            )
        scope.charge(
            work=_dependency_phydrax_meshing__polyhedral_generation._polyhedral_mesh_work_bound(
                vertices, cells, loops, corners
            )
        )
        yield storage, bound


class _PlaneSplitResult(StrictModule):
    mesh: CellMesh
    residual: float


class _PlaneSplitPreparation(StrictModule):
    loops: tuple[tuple[np.ndarray, ...], ...]
    coordinates: np.ndarray
    vertex_ids: np.ndarray
    cell_ids: np.ndarray
    cut: list[int]
    normals: list[np.ndarray]
    offsets: list[np.ndarray]
    counts: list[int]


def _prepare_plane_split(
    mesh: CellMesh,
    plane: np.ndarray,
    protected: set[int],
    tolerance: float,
    maximum_cells: int,
    *,
    scope: NativeExecutionScope,
) -> _PlaneSplitPreparation:
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            pass
        case _:
            raise ValueError(
                "polyhedral_adaptation.connectivity must be polyhedral for plane subdivision; got another connectivity kind"
            )
    loops = polyhedral_shared_owner.polyhedral_face_loops(mesh)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signed = coordinates @ plane[:3] - plane[3]
    cut = [
        index
        for index, cell in enumerate(loops)
        if np.min(signed[np.unique(np.concatenate(cell))])
        < 0
        < np.max(signed[np.unique(np.concatenate(cell))])
    ]
    if len(loops) + len(cut) > maximum_cells:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.maximum_cells must admit every plane-split child; got a cell budget refusal",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    # Each plane acts on the current candidate, not the original source. Later
    # planes can touch more descendants than the source-wide touched-cell union.
    # Refuse before constructing halfspaces or invoking the clipping kernel.
    with scope.batch(0, cavity_upper=len(cut)) as ticket:
        ticket.consume(cavity_cells=len(cut))
    for index in cut:
        for local, face in enumerate(loops[index]):
            if (
                np.min(signed[face]) < 0 < np.max(signed[face])
                and int(face_ids[cf[co[index] + local]]) in protected
            ):
                raise polyhedral_shared_owner.PolyhedralAdaptationError(
                    "polyhedral_adaptation.protected_face_ids must survive plane closure; got a subdivided protected face"
                )
    normals, offsets, counts = [], [], []
    for index in cut:
        cell_normals, cell_offsets = [], []
        corners = sum(len(face) for face in loops[index])
        # Sorting the cell's corners, then per face one cross product/offset
        # and one signed residual visit per unique cell vertex.
        scope.charge(
            work=corners * max(1, corners.bit_length())
            + len(loops[index]) * (8 + corners)
        )
        vertices = np.unique(np.concatenate(loops[index]))
        for face in loops[index]:
            points = coordinates[face]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            offset = float(normal @ points[0])
            residual = coordinates[vertices] @ normal - offset
            if np.any(residual > tolerance * np.linalg.norm(normal)):
                raise polyhedral_shared_owner.PolyhedralAdaptationError(
                    "polyhedral_adaptation.source_cells must be convex and planar; got a failed halfspace enclosure"
                )
            cell_normals.append(normal)
            cell_offsets.append(offset)
        for sign in (1.0, -1.0):
            normals.append(
                np.concatenate((np.asarray(cell_normals), plane[None, :3] * sign))
            )
            offsets.append(np.asarray([*cell_offsets, plane[3] * sign], dtype=np.float64))
            counts.append(len(cell_normals) + 1)
    return _PlaneSplitPreparation(
        loops, coordinates, vertex_ids, cell_ids, cut, normals, offsets, counts
    )


def _plane_clip_shape(planes: int, /) -> tuple[int, int, int]:
    """Return clip3d.cpp's convex face, vertex and loop-entry capacities."""
    faces = planes + 6
    vertices = 2 * faces - 4
    return faces, vertices, 2 * (vertices + faces - 2)


def _plane_clip_work(counts: list[int], width: int, /) -> int:
    """Bound native box/halfspace clipping visits before the kernel runs.

    The host pads every item's plane rows to ``width``. clip3d.cpp then clips
    each item's box by its planes in turn; each cut visits at most every current
    vertex, face and loop entry of the convex polytope. Output compaction sorts
    the vertices and faces and rotates every loop entry; moments revisit them.
    """
    work = 4 * len(counts) * width
    for planes in counts:
        faces, vertices, loops = _plane_clip_shape(planes)
        work += (
            3 * planes
            + planes * (vertices + faces + loops)
            + vertices * (vertices.bit_length() + 1)
            + faces * (faces.bit_length() + 1)
            + 2 * loops
        )
    return work


def _clip_plane_split(
    preparation: _PlaneSplitPreparation,
    storage: NativeHostStorageWorkspace,
    *,
    scope: NativeExecutionScope,
) -> _owner_phydrax__meshcore.BoxHalfspaceClip:
    count, width = len(preparation.counts), max(preparation.counts)
    faces, vertices, loops = _plane_clip_shape(width)
    # The implementation pads the *actual* cut-cell halfspaces. This term
    # cannot be replaced by an incidence sum for mixed cell arities. The native
    # call returns padded vertex, face-offset, label and loop banks plus counts,
    # volumes, moments and status per item; they stay live through assembly.
    storage.set_bound(
        12 * 256 + 32 * count * width + 8 * count + 256
        + count * (24 * vertices + 4 * (2 * faces + 1) + 4 * loops + 48)
    )
    scope.charge(work=_plane_clip_work(preparation.counts, width))
    # Native clipping workspace belongs to the request's managed pool.
    with scope.ticket_view() as ticket, ticket.native_target():
        return _build_plane_clip(preparation)


def _build_plane_clip(
    preparation: _PlaneSplitPreparation,
) -> _owner_phydrax__meshcore.BoxHalfspaceClip:
    normals, offsets, counts = (
        preparation.normals,
        preparation.offsets,
        preparation.counts,
    )
    coordinates = preparation.coordinates
    width = max(counts)
    n = np.zeros((len(normals), width, 3), dtype=np.float64)
    h = np.zeros((len(normals), width), dtype=np.float64)
    for index, (normal, offset) in enumerate(zip(normals, offsets, strict=True)):
        n[index, : normal.shape[0]], h[index, : offset.size] = normal, offset
    lower, upper = np.min(coordinates, axis=0), np.max(coordinates, axis=0)
    extent = np.max(upper - lower)
    native = clip_box_halfspaces(
        lower - extent, upper + extent, n, h, np.asarray(counts, dtype=np.int32)
    )
    if np.any(native.status != 0):
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            f"polyhedral_adaptation.clip.status must report success; got {tuple(int(value) for value in native.status)}"
        )
    return native


class _PlaneSplitAssembly(StrictModule):
    global_points: list[np.ndarray]
    global_ids: list[np.int64 | int]
    point_keys: dict[tuple[int, ...], int]


class _PlaneCutPoint(StrictModule):
    global_vertex: int
    error: float
    next_vertex: int


def _map_plane_cut_vertex(
    original: tuple[np.ndarray, ...],
    labels: list[int],
    point: np.ndarray,
    plane: np.ndarray,
    preparation: _PlaneSplitPreparation,
    assembly: _PlaneSplitAssembly,
    vertex_support: dict[int, set[int]],
    tolerance: float,
    maximum_vertices: int,
    next_vertex: int,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PlaneCutPoint:
    coordinates, vertex_ids = preparation.coordinates, preparation.vertex_ids
    global_points, global_ids, point_keys = (
        assembly.global_points,
        assembly.global_ids,
        assembly.point_keys,
    )
    # This source-edge Fraction kernel preserves construction and outward-rounding
    # order before deduplicating a shared cut point by scientific edge identity.
    source_faces = [original[label] for label in labels if 0 <= label < len(original)]
    if not source_faces:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.clip_vertex must retain source-face ancestry; got no incident source face"
        )
    support = set(int(value) for value in source_faces[0])
    for face in source_faces[1:]:
        support.intersection_update(int(value) for value in face)
    if len(support) == 1:
        global_vertex = next(iter(support))
        error = float(np.max(np.abs(point - coordinates[global_vertex])))
    elif len(support) == 2 and len(original) in labels:
        key = tuple(sorted(int(vertex_ids[value]) for value in support))
        first, second = sorted(support)
        bits = max(
            planar_primitives_owner.binary64_fraction_bits(float(value))
            for row in (coordinates[first], coordinates[second], plane)
            for value in row
        )
        budget.reserve(0, 32 * (128 + 2 * ((bits + 7) // 8)))
        # rational_points' canonical conversion/traversal rate: two visits
        # for each of the nine binary64 coefficients, with no helper copy.
        budget.reserve(18)
        origin = rational_vector(coordinates[first])
        endpoint = rational_vector(coordinates[second])
        normal = rational_vector(plane[:3])
        plane_restriction_owner._reserve_fraction_storage(
            (origin, endpoint), 3, 2, 1, budget=budget
        )
        # Coefficient-operation units match plane_key's explicit difference /
        # product circuit and _line's one unit per normalization division.
        budget.reserve(3)  # Three direction differences.
        direction = tuple(
            value - start
            for value, start in zip(endpoint, origin, strict=True)
        )
        plane_restriction_owner._reserve_fraction_storage(
            (normal, direction), 1, 6, 3, budget=budget
        )
        budget.reserve(6)  # Three products and three sum additions.
        denominator = sum(
            value * delta for value, delta in zip(normal, direction, strict=True)
        )
        if denominator == 0:
            raise polyhedral_shared_owner.PolyhedralAdaptationError(
                "polyhedral_adaptation.clip_edge must intersect its plane; got an exactly parallel edge"
            )
        budget.reserve(2)  # The storage-bound operand's binary64 lift.
        plane_restriction_owner._reserve_fraction_storage(
            (normal, origin, (Fraction(float(plane[3])), denominator)),
            1, 8, 4, budget=budget,
        )
        # Repeat the original offset conversion, then evaluate exactly the
        # original dot/subtraction/division circuit (3 + 3 + 1 + 1).
        budget.reserve(2 + 8)
        parameter = (
            Fraction(float(plane[3]))
            - sum(value * start for value, start in zip(normal, origin, strict=True))
        ) / denominator
        plane_restriction_owner._reserve_fraction_storage(
            (origin, direction, (parameter,)), 3, 3, 2, budget=budget
        )
        budget.reserve(6)  # Three products and three coordinate additions.
        exact = tuple(
            start + parameter * delta
            for start, delta in zip(origin, direction, strict=True)
        )
        budget.reserve(6)  # Three binary64 lifts for the storage operands.
        plane_restriction_owner._reserve_fraction_storage(
            (tuple(Fraction(float(value)) for value in point), exact),
            3, 2, 1, budget=budget,
        )
        # Three repeated binary64 lifts, three differences and three magnitudes.
        # _exact_polyhedral_geometry._polyhedron_measure's 12+14 circuit
        # prices nine differences, the 14-operation determinant, magnitude,
        # division and zero test in the same coefficient-operation units.
        budget.reserve(6 + 3 + 3)
        construction_error = max(
            abs(Fraction(float(value)) - expected)
            for value, expected in zip(point, exact, strict=True)
        )
        budget.reserve(2)  # Original tolerance conversion.
        if construction_error > Fraction(tolerance):
            raise polyhedral_shared_owner.PolyhedralAdaptationError(
                "PolyhedralMeshAdaptation.coordinate_tolerance must bound source-edge construction; got excess rational construction error"
            )
        # binary64_polygon_bounds pays one directed-rounding primitive per
        # coefficient, covering float conversion and its outward nextafter.
        # Preserve this kernel's original conditional-nextafter numerical order.
        budget.reserve(1)
        construction_bound = float(construction_error)
        budget.reserve(2)  # Original rounded-bound conversion.
        if Fraction(construction_bound) < construction_error:
            construction_bound = float(np.nextafter(construction_bound, np.inf))
        if key not in point_keys:
            if len(global_points) >= maximum_vertices:
                raise polyhedral_shared_owner.PolyhedralAdaptationError(
                    "polyhedral_adaptation.maximum_vertices must admit every cut vertex; got a vertex budget refusal",
                    category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                )
            point_keys[key] = len(global_points)
            global_points.append(point)
            global_ids.append(next_vertex)
            vertex_support[next_vertex] = set().union(
                *(vertex_support[int(vertex_ids[value])] for value in support)
            )
            next_vertex += 1
        global_vertex = point_keys[key]
        error = max(
            construction_bound,
            float(np.max(np.abs(point - global_points[global_vertex]))),
        )
    else:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.clip_vertex must have unique source vertex or edge ancestry; got ambiguous support"
        )
    return _PlaneCutPoint(global_vertex, error, next_vertex)


def _split(
    mesh: CellMesh,
    plane: np.ndarray,
    protected: set[int],
    tolerance: float,
    maximum_cells: int,
    maximum_vertices: int,
    vertex_support: dict[int, set[int]],
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PlaneSplitResult:
    with _polyhedral_topology_phase(
        scope, mesh, split=True, periodic_topology=False
    ) as (storage, bound):
        with _polyhedral_exact_phase(
            scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
        ):
            result = _build_plane_split(
                mesh, plane, protected, tolerance, maximum_cells, maximum_vertices,
                vertex_support, scope=scope, budget=budget,
            )
        if result.mesh is not mesh:
            storage.retain(
                bound.retained, result.mesh, record_count_upper=bound.records
            )
        return result


def _build_plane_split(
    mesh: CellMesh,
    plane: np.ndarray,
    protected: set[int],
    tolerance: float,
    maximum_cells: int,
    maximum_vertices: int,
    vertex_support: dict[int, set[int]],
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PlaneSplitResult:
    preparation = _prepare_plane_split(
        mesh, plane, protected, tolerance, maximum_cells, scope=scope
    )
    if not preparation.cut:
        return _PlaneSplitResult(mesh, 0.0)
    with NativeHostStorageWorkspace(scope) as clip_storage:
        native = _clip_plane_split(preparation, clip_storage, scope=scope)
        points, point_counts = native.vertices, native.vertex_counts
        face_offsets, face_labels = native.face_offsets, native.face_labels
        face_vertices, face_counts = native.face_vertices, native.face_counts
        loops, cell_ids = preparation.loops, preparation.cell_ids
        vertex_ids, coordinates, cut = (
            preparation.vertex_ids,
            preparation.coordinates,
            preparation.cut,
        )
        # Uncut loop/id copies; per cut child, the incident label lists and
        # the remapped face loops each visit every used loop entry, and every
        # output vertex is mapped once (its exact construction is budgeted).
        used = face_offsets[np.arange(face_counts.shape[0]), face_counts]
        scope.charge(
            work=2 * len(loops) + 2 * coordinates.shape[0]
            + 3 * int(np.sum(used)) + 2 * int(np.sum(point_counts))
        )
        global_points = list(coordinates)
        global_ids = list(vertex_ids)
        point_keys: dict[tuple[int, ...], int] = {}
        assembly = _PlaneSplitAssembly(global_points, global_ids, point_keys)
        next_vertex = int(np.max(vertex_ids)) + 1
        next_cell = int(np.max(cell_ids)) + 1
        new_loops, new_ids = [], []
        residual = 0.0
        cut_positions = {cell: index for index, cell in enumerate(cut)}
        for cell, original in enumerate(loops):
            if cell not in cut_positions:
                new_loops.append(original)
                new_ids.append(int(cell_ids[cell]))
                continue
            for side in range(2):
                item = 2 * cut_positions[cell] + side
                incident: list[list[int]] = [[] for _ in range(point_counts[item])]
                for face in range(face_counts[item]):
                    label = int(face_labels[item, face])
                    for vertex in face_vertices[
                        item, face_offsets[item, face] : face_offsets[item, face + 1]
                    ]:
                        incident[int(vertex)].append(label)
                local_to_global = []
                for vertex, labels in enumerate(incident):
                    mapped = _map_plane_cut_vertex(
                        original,
                        labels,
                        points[item, vertex],
                        plane,
                        preparation,
                        assembly,
                        vertex_support,
                        tolerance,
                        maximum_vertices,
                        next_vertex,
                        budget=budget,
                    )
                    next_vertex = mapped.next_vertex
                    residual = max(residual, mapped.error)
                    local_to_global.append(mapped.global_vertex)
                faces = tuple(
                    np.asarray(
                        [
                            local_to_global[int(value)]
                            for value in face_vertices[
                                item, face_offsets[item, face] : face_offsets[item, face + 1]
                            ]
                        ],
                        dtype=np.int32,
                    )
                    for face in range(face_counts[item])
                )
                new_loops.append(faces)
                new_ids.append(next_cell)
                next_cell += 1
        if residual > tolerance:
            raise polyhedral_shared_owner.PolyhedralAdaptationError(
                "PolyhedralMeshAdaptation.coordinate_tolerance must bound shared-edge constructions; got excess construction error"
            )
        target = CellMesh.from_polyhedra(
            np.asarray(global_points, dtype=np.float64),
            new_loops,
            vertex_global_ids=np.asarray(global_ids, dtype=np.int64),
            cell_global_ids=np.asarray(new_ids, dtype=np.int64),
        )
        return _PlaneSplitResult(target, residual)


class _AgglomerationPreparation(StrictModule):
    loops: tuple[tuple[np.ndarray, ...], ...]
    ids: np.ndarray
    index: dict[int, int]
    co: np.ndarray
    cf: np.ndarray
    face_ids: np.ndarray
    owner: np.ndarray
    neighbor: np.ndarray
    cells: list[tuple[np.ndarray, ...]]
    cell_ids: list[int]
    next_cell: int
    classes: np.ndarray
    protected: set[int]


def _prepare_agglomeration(
    mesh: CellMesh,
    groups: tuple[tuple[int, ...], ...],
    classes: np.ndarray,
    protected: set[int],
) -> _AgglomerationPreparation:
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            pass
        case _:
            raise ValueError(
                "polyhedral_adaptation.connectivity must use packed oriented incidence; got another connectivity kind"
            )
    loops = polyhedral_shared_owner.polyhedral_face_loops(mesh)
    ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    index = {int(identifier): row for row, identifier in enumerate(ids)}
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    owner, neighbor = (
        np.asarray(connectivity.face_owner),
        np.asarray(connectivity.face_neighbor),
    )
    removed = set(identifier for group in groups for identifier in group)
    if not removed.issubset(index):
        raise ValueError(
            "PolyhedralMeshAdaptation.agglomerations must name current cells; got unknown IDs"
        )
    cells, cell_ids = (
        [cell for row, cell in enumerate(loops) if int(ids[row]) not in removed],
        [int(identifier) for identifier in ids if int(identifier) not in removed],
    )
    next_cell = int(np.max(ids)) + 1
    return _AgglomerationPreparation(
        loops,
        ids,
        index,
        co,
        cf,
        face_ids,
        owner,
        neighbor,
        cells,
        cell_ids,
        next_cell,
        classes,
        protected,
    )


class _AgglomerationBoundary(StrictModule):
    faces: tuple[np.ndarray, ...]


def _agglomeration_boundary(
    preparation: _AgglomerationPreparation,
    group: tuple[int, ...],
) -> _AgglomerationBoundary:
    loops = preparation.loops
    index = preparation.index
    co = preparation.co
    cf = preparation.cf
    face_ids = preparation.face_ids
    owner = preparation.owner
    neighbor = preparation.neighbor
    classes = preparation.classes
    protected = preparation.protected
    rows = {index[identifier] for identifier in group}
    if np.unique(classes[list(rows)]).size != 1:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.cell_classes must agree within each agglomeration; got mixed classes"
        )
    visited, frontier = set(), [min(rows)]
    while frontier:
        row = frontier.pop()
        if row in visited:
            continue
        visited.add(row)
        for face in cf[co[row] : co[row + 1]]:
            adjacent = int(neighbor[face]) if owner[face] == row else int(owner[face])
            if adjacent in rows and adjacent not in visited:
                frontier.append(adjacent)
    if visited != rows:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "PolyhedralMeshAdaptation.agglomerations must be face-connected; got a disconnected group"
        )
    boundary = []
    for row in sorted(rows):
        for local, face in enumerate(cf[co[row] : co[row + 1]]):
            if int(owner[face]) in rows and int(neighbor[face]) in rows:
                if int(face_ids[face]) in protected:
                    raise polyhedral_shared_owner.PolyhedralAdaptationError(
                        "polyhedral_adaptation.protected_face_ids must survive agglomeration; got a removed protected face"
                    )
            else:
                boundary.append(loops[row][local])
    return _AgglomerationBoundary(tuple(boundary))


def _agglomerate(
    mesh: CellMesh,
    groups: tuple[tuple[int, ...], ...],
    classes: np.ndarray,
    protected: set[int],
    *,
    scope: NativeExecutionScope,
) -> CellMesh:
    # Groups are disjoint by request validation; all replacements coexist in
    # this candidate. Admission precedes topology planning and allocation.
    with scope.batch(0, cavity_upper=sum(len(group) for group in groups)) as ticket:
        ticket.consume(cavity_cells=ticket.cavity_upper)
    with _polyhedral_topology_phase(
        scope, mesh, periodic_topology=False
    ) as (storage, bound):
        target = _build_agglomeration(mesh, groups, classes, protected)
        storage.retain(bound.retained, target, record_count_upper=bound.records)
        return target


def _build_agglomeration(
    mesh: CellMesh,
    groups: tuple[tuple[int, ...], ...],
    classes: np.ndarray,
    protected: set[int],
) -> CellMesh:
    preparation = _prepare_agglomeration(mesh, groups, classes, protected)
    cells, cell_ids, next_cell = (
        preparation.cells,
        preparation.cell_ids,
        preparation.next_cell,
    )
    for group in groups:
        boundary = _agglomeration_boundary(preparation, group)
        cells.append(boundary.faces)
        cell_ids.append(next_cell)
        next_cell += 1
    used = np.unique(np.concatenate([face for cell in cells for face in cell]))
    remap = np.full(mesh.coordinates.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return CellMesh.from_polyhedra(
        np.asarray(mesh.coordinates)[used],
        tuple(tuple(remap[face] for face in cell) for cell in cells),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids)[used],
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
    )


def _topology_edit(
    source: CellMesh,
    target: CellMesh,
    refinement: _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement | None,
    vertex_support: dict[int, set[int]],
    *,
    scope: NativeExecutionScope,
) -> _dependency_phydrax_meshing__topology_edit.CellTopologyEdit:
    with _polyhedral_topology_phase(scope, target) as (storage, bound):
        source_bound = _polyhedral_adaptation_storage_bound(
            _polyhedral_incidence(source, scope=scope)._replace(rank=0)
        )
        base = bound.construction + source_bound.construction
        overlap_rows = 0 if refinement is None else refinement.source_cells.size
        base += 256 + 64 * overlap_rows
        storage.workspace.set_bound(base)
        edit, relation_bytes = _build_topology_edit(
            source, target, refinement, vertex_support,
            storage=storage.workspace, base=base, scope=scope,
        )
        storage.retain(
            bound.retained + relation_bytes + 64 * overlap_rows,
            *(leaf for leaf in jax.tree_util.tree_leaves(edit) if eqx.is_array(leaf)),
            record_count_upper=bound.records,
        )
        return edit


def _build_topology_edit(
    source: CellMesh,
    target: CellMesh,
    refinement: _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement | None,
    vertex_support: dict[int, set[int]],
    *,
    storage: NativeHostStorageWorkspace,
    base: int,
    scope: NativeExecutionScope,
) -> tuple[_dependency_phydrax_meshing__topology_edit.CellTopologyEdit, int]:
    loops = polyhedral_shared_owner.polyhedral_face_loops(target)
    blocks = []
    offset = 0
    source_kinds = {block.name: block.cell_kind for block in source.blocks}
    for block in target.blocks:
        connectivity = polyhedral_connectivity(
            loops[offset : offset + block.cell_count],
            target.coordinates.shape[0],
            vertex_global_ids=target.vertex_global_ids,
            cell_global_ids=block.global_ids,
        )
        blocks.append(
            _dependency_phydrax_meshing__topology_edit.PolyhedralTopologyEditBlock(
                block.name,
                source_kinds.get(block.name),
                connectivity,
                np.asarray(block.global_ids, dtype=np.int64),
            )
        )
        offset += block.cell_count
    source_ids = (
        np.zeros(0, dtype=np.int64)
        if refinement is None
        else np.asarray(refinement.source_cell_global_ids, dtype=np.int64)
    )
    target_ids = (
        np.zeros(0, dtype=np.int64)
        if refinement is None
        else np.asarray(refinement.target_cell_global_ids, dtype=np.int64)
    )
    relations = []
    relation_bytes = 0
    for dimension in range(4):
        # Sorted source and target entity-key extraction before matching; the
        # cell relation gathers visit each refinement row once.
        scope.charge(
            work=_stratum_pass_work(source, dimension)
            + _stratum_pass_work(target, dimension)
            + (0 if refinement is None else 2 * int(np.asarray(refinement.source_cells).size))
        )
        keys = entity_keys_owner.entity_keys(source, dimension)
        if dimension == 3:
            source_cells = (
                np.zeros(0, dtype=np.int64)
                if refinement is None
                else np.asarray(refinement.source_cells)
            )
            target_cells = (
                np.zeros(0, dtype=np.int64)
                if refinement is None
                else np.asarray(refinement.target_cells)
            )
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    3,
                    source_ids[source_cells][:, None],
                    target_ids[target_cells][:, None],
                    np.full(
                        source_cells.size,
                        int(
                            _dependency_phydrax_meshing__lineage.EntityLineageKind.SWAPPED_FROM
                        ),
                        dtype=np.int32,
                    ),
                )
            )
        else:
            target_keys = entity_keys_owner.entity_keys(target, dimension)
            parents, children = [], []
            if dimension in (1, 2):
                for target_key in target_keys:
                    # Key filtering and the support union over its vertices.
                    scope.charge(work=2 + 2 * target_keys.shape[1])
                    values = [int(value) for value in target_key if value >= 0]
                    support = set().union(
                        *(vertex_support.get(value, set()) for value in values)
                    )
                    if not support:
                        continue
                    # Admit this key's full scan of every source key before it:
                    # each comparison builds one source set and two subset tests.
                    scope.charge(
                        work=len(keys) * (2 * keys.shape[1] + len(support) + len(values) + 2)
                    )
                    for source_key in keys:
                        if support.issubset(
                            set(int(value) for value in source_key if value >= 0)
                        ) and set(values) != set(
                            int(value) for value in source_key if value >= 0
                        ):
                            # Admit each matched relation before its two retained
                            # NumPy views/list slots and the eventual key banks.
                            relation_bytes += (
                                2 * 256 + 64
                                + 16 * (keys.shape[1] + target_keys.shape[1])
                            )
                            storage.set_bound(base + relation_bytes)
                            parents.append(source_key)
                            children.append(target_key)
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    dimension,
                    np.asarray(parents, dtype=np.int64).reshape((-1, keys.shape[1])),
                    np.asarray(children, dtype=np.int64).reshape(
                        (-1, target_keys.shape[1])
                    ),
                    np.full(
                        len(parents),
                        int(
                            _dependency_phydrax_meshing__lineage.EntityLineageKind.REFINED_FROM
                        ),
                        dtype=np.int32,
                    ),
                )
            )
    # Source vertex-set build, then one membership visit per target vertex
    # and one relation-array row write.
    scope.charge(
        work=int(np.asarray(source.vertex_global_ids).size)
        + 2 * int(np.asarray(target.vertex_global_ids).size)
    )
    source_vertices = set(int(value) for value in np.asarray(source.vertex_global_ids))
    vertices = np.asarray(target.vertex_global_ids, dtype=np.int64)
    valid = np.asarray(
        [int(value) in source_vertices for value in vertices], dtype=np.bool_
    )[:, None]
    return _dependency_phydrax_meshing__topology_edit.CellTopologyEdit(
        "local_reconnection",
        np.asarray(target.coordinates),
        vertices,
        tuple(blocks),
        np.where(valid, vertices[:, None], -1),
        valid.astype(np.float64),
        valid,
        tuple(relations),
        target_periodic_topology=target.periodic_topology,
    ), relation_bytes


class _PolyhedralAllowance(NamedTuple):
    maximum_work_units: int
    maximum_scratch_bytes: int

    def native_limits(self) -> NativeExecutionLimits:
        """Lower existing work/storage limits without adding query, cavity or wall controls."""
        # These axes were unrestricted on the standalone API. Preserve that
        # contract using the native-addressable maxima, not a new shape policy.
        native_addressable_maximum = (1 << 64) - 1
        return NativeExecutionLimits(
            self.maximum_work_units,
            native_addressable_maximum,
            native_addressable_maximum,
            self.maximum_scratch_bytes,
            math.inf,
        )


@contextmanager
def _polyhedral_exact_phase(
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    maximum_work: int,
    maximum_bytes: int,
    /,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    if not budget.ticket.settled:
        with budget.bound_stage(maximum_work, maximum_bytes), budget.temporary_scope():
            yield budget
        return
    remaining = scope.remaining()
    ticket = scope.batch(
        min(maximum_work, remaining.remaining_work_units),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(maximum_bytes, remaining.remaining_managed_bytes),
        cavity_upper=remaining.remaining_cavity_cells,
    )
    with budget.bind_ticket(ticket), ticket.native_target(), budget.temporary_scope():
        yield budget


class _EdgeTraceActions(StrictModule):
    source_ids: np.ndarray
    actions: list[dict[int, Fraction]] | None


def _compose_edge_trace_actions(
    source: CellMesh,
    original: _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource
    | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource
    | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource,
    chain: list[
        _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource
    ],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _EdgeTraceActions:
    source_ids = np.asarray(source.vertex_global_ids)
    budget.reserve(len(source_ids), 256 + 256 * len(source_ids))
    actions = [{row: Fraction(1)} for row in range(len(source_ids))]
    for node in reversed(chain):
        prepared = node.prepare(budget=budget)
        parents = np.asarray(node.vertex_parents)
        budget.reserve(len(parents), 256 + 384 * len(parents))
        successor = []
        for indices, coefficients in zip(parents, prepared.barycentric, strict=True):
            row = {}
            for parent, coefficient in zip(
                indices[indices >= 0], coefficients, strict=True
            ):
                for origin, weight in actions[int(parent)].items():
                    if coefficient == 0 or weight == 0:
                        continue
                    if origin not in row and len(row) == 2:
                        return _EdgeTraceActions(source_ids, None)
                    accumulated = row.get(origin, Fraction(0))
                    planar_primitives_owner.reserve_fraction_work(
                        ((coefficient, weight, accumulated),), 2, 1, 2, budget=budget
                    )
                    value = accumulated + coefficient * weight
                    if (
                        max(
                            value.numerator.bit_length(),
                            value.denominator.bit_length(),
                        )
                        > original.maximum_bits
                    ):
                        raise polyhedral_shared_owner.PolyhedralAdaptationError(
                            "polyhedral_adaptation.edge_trace must fit source integer-bit capacity; got an oversized rational action",
                            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        )
                    row[origin] = value
            successor.append({origin: value for origin, value in row.items() if value})
        actions = successor
    return _EdgeTraceActions(source_ids, actions)


class _EdgeTraceStencil(StrictModule):
    sources: np.ndarray | None
    weights: np.ndarray | None


def _prepare_edge_trace_stencil(
    source: CellMesh,
    target: CellMesh,
    source_ids: np.ndarray,
    actions: list[dict[int, Fraction]],
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _EdgeTraceStencil:
    if len(actions) != target.coordinates.shape[0]:
        raise ValueError(
            "polyhedral_adaptation.edge_trace.actions must cover target coordinates; got missing rows"
        )
    edge_rows = np.asarray(
        _dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(
            source
        ).edges
    )
    budget.reserve(edge_rows.size, 256 + 192 * len(edge_rows))
    edges = {frozenset(int(value) for value in edge) for edge in edge_rows}
    # One membership visit per action row for the unsupported-edge scan.
    budget.reserve(2 * len(actions))
    if any(len(row) != 1 and frozenset(row) not in edges for row in actions):
        return _EdgeTraceStencil(None, None)
    budget.reserve(len(actions))
    sources = host_banks_owner.family_host_array(
        (len(actions), 2), np.int64, scope=scope
    )
    weights = host_banks_owner.family_host_array(
        (len(actions), 2), np.float64, scope=scope
    )
    sources.fill(-1)
    weights.fill(0)
    for target_row, row in enumerate(actions):
        with budget.temporary_scope():
            budget.reserve(0, 64 + 16 * len(row))
            coefficients = tuple(row.values())
            # Existing coefficient arithmetic and exact scalar comparisons:
            # N sum additions, one unit-value test, then min and zero tests.
            planar_primitives_owner.reserve_fraction_work(
                (coefficients,), len(row) + 1, 2, max(len(row), 1), budget=budget
            )
            if sum(coefficients, Fraction(0)) != 1:
                raise ValueError(
                    "polyhedral_adaptation.edge_trace.actions must be convex unit functionals; got invalid coefficients"
                )
            budget.reserve(2 * (len(row) - 1) + 1)
            if min(coefficients) < 0:
                raise ValueError(
                    "polyhedral_adaptation.edge_trace.actions must be convex unit functionals; got invalid coefficients"
                )
            for slot, (origin, coefficient) in enumerate(sorted(row.items())):
                sources[target_row, slot] = source_ids[origin]
                # The existing binary64 export primitive costs one unit.
                budget.reserve(1)
                weights[target_row, slot] = float(coefficient)
            del coefficients
    return _EdgeTraceStencil(sources, weights)


def _edge_trace_stencil(
    source: CellMesh,
    target: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    target_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    *,
    maximum_work: int,
    maximum_bytes: int,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__topology_edit.CellTopologyEdit:
    """Compose degree-one functionals on retained vertices and source edges.

    Geometric face/interior actions do not evaluate a virtual-element field.
    Site regeneration retains unknown rows when it lacks source restrictions.
    """
    original, current = (
        polyhedral_shared_owner.bound_power_source(source_geometry),
        polyhedral_shared_owner.bound_power_source(target_geometry),
    )
    if original is None or current is None:
        return edit
    chain = []
    visited = set()
    while current.source_id != original.source_id:
        # One cycle test, one set insertion and one parent step per link.
        scope.charge(work=3)
        if id(current) in visited:
            raise ValueError(
                "polyhedral_adaptation.edge_trace.ancestry must be acyclic; got a source cycle"
            )
        visited.add(id(current))
        match current:
            case _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource():
                pass
            case _:
                return edit
        chain.append(current)
        current = current.parent
    with _polyhedral_exact_phase(scope, budget, maximum_work, maximum_bytes):
        composed = _compose_edge_trace_actions(source, original, chain, budget=budget)
        if composed.actions is None:
            return edit
        stencil = _prepare_edge_trace_stencil(
            source,
            target,
            composed.source_ids,
            composed.actions,
            scope=scope,
            budget=budget,
        )
        if stencil.sources is None or stencil.weights is None:
            return edit
        valid = host_banks_owner.family_host_array(
            stencil.sources.shape, np.bool_, scope=scope
        )
        np.greater_equal(stencil.sources, 0, out=valid)
        return edit._replace(
            stencil_sources=stencil.sources,
            stencil_weights=stencil.weights,
            stencil_valid=valid,
        )


def _periodic_source_target(
    source: CellMesh,
    target: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    *,
    maximum_work: int,
    maximum_bytes: int,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellMesh:
    """Bind source-backed split/agglomeration through W11's rational seam orbits."""

    periodic = source.periodic_topology
    if periodic is None:
        return target
    with _polyhedral_exact_phase(scope, budget, maximum_work, maximum_bytes):
        root = _owner_phydrax_meshing__topology_edit.power_domain_root(
            geometry, budget=budget
        )
        preparation = root.periodic_preparation
        match preparation:
            case periodic_power_preparation_owner.PeriodicPowerPreparation():
                pass
            case _:
                raise ValueError(
                    "polyhedral_adaptation.periodic_source must retain image preparation; got missing periodic source data"
                )
        points = geometry.rational_source_coordinates(budget=budget)
        # One insertion per point plus one hash visit per rational coordinate.
        budget.reserve(4 * len(points), 256 + 128 * len(points))
        lookup = {point: row for row, point in enumerate(points)}
        generator_bits = max(
            (planar_primitives_owner.binary64_fraction_bits(float(value))
             for value in preparation.generators.flat),
            default=1,
        )
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            2 * preparation.generators.size, preparation.generators.size, 0,
            generator_bits, budget=budget
        )
        matrices = tuple(
            tuple(rational_vector(row) for row in matrix)
            for matrix in preparation.generators
        )
        pairs = []
        for generator, matrix in enumerate(matrices):
            for row, point in enumerate(points):
                planar_primitives_owner.reserve_fraction_work(
                    (point, *matrix), 36, 3, 8, budget=budget
                )
                image = tuple(
                    sum(matrix[axis][column] * point[column] for column in range(3))
                    + matrix[axis][3]
                    for axis in range(3)
                )
                other = lookup.get(image)
                if other is not None:
                    budget.reserve(1, 128)
                    pairs.append((row, other, generator))
        representatives, shifts = (
            _owner_phydrax_meshing__polyhedral_generation.seam_vertex_orbits(
                points,
                np.asarray(pairs, dtype=np.int64).reshape((-1, 3)),
                np.asarray(target.vertex_global_ids),
                preparation,
                scope=scope,
                budget=budget,
            )
        )
        descriptor = (
            _owner_phydrax_discretization__periodic_topology.PeriodicMeshTopology(
                target,
                periodic.cell,
                representatives,
                shifts,
                actual_geometry=geometry,
                budget=budget,
            )
        )

    return CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        polyhedral_connectivity=_dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(
            target
        ),
        periodic_topology=descriptor,
        numeric_version=target.numeric_version,
        budget=budget,
    )


def _stage_polyhedral_target(
    source: CellMesh,
    target: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    support: dict[int, set[int]],
    *,
    numeric_version: str,
    maximum_work: int,
    maximum_bytes: int,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellMesh:
    with _polyhedral_topology_phase(
        scope, target, periodic_source=source
    ) as (storage, bound):
        source_bound = _polyhedral_adaptation_storage_bound(
            _polyhedral_incidence(source, scope=scope)._replace(rank=0)
        )
        storage.workspace.set_bound(
            bound.construction + source_bound.construction
            + _polyhedral_cross_key_storage(source, target)
        )
        staged = _build_polyhedral_target(
            source, target, geometry, support,
            numeric_version=numeric_version, maximum_work=maximum_work,
            maximum_bytes=maximum_bytes, scope=scope, budget=budget,
        )
        storage.retain(bound.retained, staged, record_count_upper=bound.records)
        return staged


def _build_polyhedral_target(
    source: CellMesh,
    target: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    support: dict[int, set[int]],
    *,
    numeric_version: str,
    maximum_work: int,
    maximum_bytes: int,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellMesh:
    with _polyhedral_exact_phase(scope, budget, maximum_work, maximum_bytes):
        if source.periodic_topology is not None and target.periodic_topology is None:
            target = _periodic_source_target(
                source,
                target,
                geometry,
                maximum_work=maximum_work,
                maximum_bytes=maximum_bytes,
                scope=scope,
                budget=budget,
            )
        target = _owner_phydrax_meshing__topology_edit.prepare_topology_edit_target(
            source,
            _topology_edit(source, target, None, support, scope=scope),
            numeric_version=numeric_version,
            budget=budget,
        )
        if source.periodic_topology is not None:
            with _polyhedral_exact_phase(scope, budget, maximum_work, maximum_bytes):
                target = _owner_phydrax_meshing__periodic.prepare_periodic_edit_target(
                    source, target, budget=budget
                )
        elif target.periodic_topology is not None:
            raise ValueError(
                "polyhedral_adaptation.construction must preserve nonperiodic source semantics; got a periodic source law"
            )
        return target


def _periodic_polyhedral_edit(
    source: CellMesh,
    target: CellMesh,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    authority: _dependency_phydrax_meshing__topology_edit.PeriodicNonnestedGeometryAuthority
    | None = None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__topology_edit.CellTopologyEdit:
    with _polyhedral_exact_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        descriptor = target.periodic_topology
        if source.periodic_topology is None:
            return edit
        if descriptor is None:
            raise ValueError(
                "polyhedral_adaptation.target.periodic_topology must retain its quotient source; got missing source data"
            )
        witness = _owner_phydrax_meshing__periodic.periodic_vertex_orbit_witness(
            source,
            target,
            np.asarray(target.vertex_global_ids)[
                np.asarray(descriptor.vertex_representatives)
            ],
            np.asarray(descriptor.vertex_shifts),
            None,
            nonnested_geometry=authority,
            budget=budget,
        )
        return edit._replace(periodic_orbits=witness)


def _regenerated_target(
    source: CellMesh,
    construction: _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[CellMesh, _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec]:
    with _polyhedral_topology_phase(scope, construction.mesh) as (storage, bound):
        storage.workspace.set_bound(
            bound.construction + 2 * 256
            + 8 * (source.coordinates.shape[0] + source.connectivity.cell_count)
        )
        target, geometry = _build_regenerated_target(
            source, construction, scope=scope, budget=budget
        )
        storage.retain(
            bound.retained, target, *_polyhedral_geometry_aliases(geometry),
            record_count_upper=bound.records,
        )
        return target, geometry


def _build_regenerated_target(
    source: CellMesh,
    construction: _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[CellMesh, _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec]:
    """Rebind scientific IDs while retaining the producer's source and actions."""
    with _polyhedral_exact_phase(
        scope,
        budget,
        budget.maximum_work_units - budget.work_units,
        budget.maximum_memory_bytes,
    ):
        generated = construction.mesh
        source_connectivity = source.connectivity
        match source_connectivity:
            case PolyhedralConnectivity():
                pass
            case _:
                raise TypeError(
                    "polyhedral_adaptation.source.connectivity must retain packed identities; got another connectivity kind"
                )
        vertex_count = generated.coordinates.shape[0]
        cell_count = generated.connectivity.cell_count
        generated_connectivity = (
            _dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(
                generated
            )
        )
        # Source identity maxima, the generated face-loop extraction (each face
        # bounds at most two cell loops) and both identity ranges. The topology
        # phase charged one construction; a periodic target constructs twice.
        loops = generated_connectivity.cell_face_values.size
        corners = 2 * generated_connectivity.face_vertex_values.size
        scope.charge(
            work=source.coordinates.shape[0] + source_connectivity.cell_count
            + loops + corners + vertex_count + cell_count
            + (
                0
                if generated.periodic_topology is None
                else _dependency_phydrax_meshing__polyhedral_generation._polyhedral_mesh_work_bound(
                    vertex_count, cell_count, loops, corners
                )
            )
        )
        first_vertex = int(np.max(np.asarray(source.vertex_global_ids))) + 1
        first_cell = int(np.max(np.asarray(source_connectivity.cell_global_ids))) + 1
        if (
            max(first_vertex + vertex_count - 1, first_cell + cell_count - 1)
            > np.iinfo(np.int64).max
        ):
            raise polyhedral_shared_owner.PolyhedralAdaptationError(
                "polyhedral_adaptation.target_ids must fit scientific identity capacity; got an allocator overflow",
                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
            )
        target = CellMesh.from_polyhedra(
            generated.coordinates,
            polyhedral_shared_owner.polyhedral_face_loops(generated),
            vertex_global_ids=np.arange(
                first_vertex, first_vertex + vertex_count, dtype=np.int64
            ),
            cell_global_ids=np.arange(
                first_cell, first_cell + cell_count, dtype=np.int64
            ),
        )
        exact = polyhedral_shared_owner.bound_power_source(construction.geometry)
        if exact is None:
            raise ValueError(
                "polyhedral_adaptation.construction.geometry must retain its power source; got missing source construction"
            )
        geometry = (
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
                target, exact, budget=budget
            )
        )
        periodic = generated.periodic_topology
        if periodic is not None:
            descriptor = (
                _owner_phydrax_discretization__periodic_topology.PeriodicMeshTopology(
                    target,
                    periodic.cell,
                    periodic.vertex_representatives,
                    periodic.vertex_shifts,
                    actual_geometry=geometry,
                    budget=budget,
                )
            )
            target = CellMesh(
                target.coordinates,
                target.blocks,
                vertex_global_ids=target.vertex_global_ids,
                polyhedral_connectivity=_dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(
                    target
                ),
                periodic_topology=descriptor,
                budget=budget,
            )
            geometry = (
                _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
                    target, exact, budget=budget
                )
            )
        return target, geometry


@checked
def adapt_polyhedral_mesh(
    mesh: CellMesh,
    request: PolyhedralMeshAdaptation,
    *,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    cell_classes: ConvertibleToArray | None = None,
    protected_face_ids: ConvertibleToArray = (),
    maximum_cells: int = 1 << 24,
    maximum_vertices: int = 1 << 22,
    maximum_work_units: int = 1 << 26,
    maximum_scratch_bytes: int = 8000000000,
    source_id: str = "native-polyhedral",
    common_refinement_policy: _dependency_phydrax_geometry__supermesh.CommonRefinementPolicy
    | None = None,
    numeric_version: str = "0",
    scope: NativeExecutionScope | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
    limits: contracts_owner.MeshingLimits | None = None,
) -> PolyhedralAdaptationOutcome:
    """Prepare polyhedral source, orbit, and overlap data on one coordinate ledger."""
    try:
        allowance = _PolyhedralAllowance(maximum_work_units, maximum_scratch_bytes)
        context = (
            NativeExecutionScope(allowance.native_limits())
            if scope is None
            else nullcontext(scope)
        )
        with context as execution:
            owns_budget = budget is None
            if budget is None:
                initial = execution.batch(0)
                budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
                    initial,
                    maximum_work_units=maximum_work_units,
                    maximum_memory_bytes=maximum_scratch_bytes,
                )
                initial.settle()
            retention = (
                budget.retained_storage(
                    lambda amount: execution.batch(0, bytes_upper=amount)
                )
                if owns_budget
                else nullcontext()
            )
            with budget.measurement(), retention:
                return _adapt_polyhedral_mesh(
                    mesh,
                    request,
                    source_geometry=source_geometry,
                    cell_classes=cell_classes,
                    protected_face_ids=protected_face_ids,
                    maximum_cells=maximum_cells,
                    maximum_vertices=maximum_vertices,
                    maximum_work_units=maximum_work_units,
                    maximum_scratch_bytes=maximum_scratch_bytes,
                    source_id=source_id,
                    common_refinement_policy=common_refinement_policy,
                    numeric_version=numeric_version,
                    scope=execution,
                    budget=budget,
                    limits=limits,
                )
    except (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
        MeshcoreError,
    ) as error:
        if isinstance(error, MeshcoreError) and error.status not in (
            MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT
        ):
            raise
        failure = _owner_phydrax_meshing_providers__native_polyhedral.construction_failure(
            error, "polyhedral_adaptation"
        )
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            str(error), category=failure.category,
            requested=failure.evidence.requested,
            achieved=failure.evidence.achieved,
        ) from error


class _PolyhedralExecution(StrictModule):
    mesh: CellMesh
    request: PolyhedralMeshAdaptation
    maximum_cells: int
    maximum_vertices: int
    maximum_work_units: int
    maximum_scratch_bytes: int
    source_id: str
    common_refinement_policy: (
        _dependency_phydrax_geometry__supermesh.CommonRefinementPolicy | None
    )
    numeric_version: str
    scope: NativeExecutionScope
    budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )
    limits: contracts_owner.MeshingLimits | None


class _PolyhedralSource(StrictModule):
    connectivity: PolyhedralConnectivity
    count: int
    classes: np.ndarray
    protected: set[int]
    vertex_support: dict[int, set[int]]
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    source_power: (
        _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource
        | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource
        | _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource
        | None
    )


class _PolyhedralCandidate(StrictModule):
    target: CellMesh
    target_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    residual: float
    construction: (
        _dependency_phydrax_meshing__polyhedral_generation.PolyhedralConstruction | None
    )


class _PolyhedralCoverage(StrictModule):
    target: CellMesh
    target_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    common: _dependency_phydrax_geometry__supermesh.PreparedCommonRefinement


def _admit_polyhedral_source(
    execution: _PolyhedralExecution,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None,
    cell_classes: ConvertibleToArray | None,
    protected_face_ids: ConvertibleToArray,
) -> _PolyhedralSource:
    mesh = execution.mesh
    maximum_work_units = execution.maximum_work_units
    maximum_scratch_bytes = execution.maximum_scratch_bytes
    scope = execution.scope
    budget = execution.budget
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            pass
        case _:
            raise TypeError(
                "polyhedral_adaptation.mesh must have packed polyhedral connectivity; got another connectivity kind"
            )
    count = connectivity.cell_count
    classes = (
        np.zeros(count, dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes, dtype=np.int64)
    )
    if classes.shape != (count,):
        raise ValueError(
            "polyhedral_adaptation.cell_classes must align with polyhedral cells; got incompatible rows"
        )
    # Protected-ID set insertion and one singleton support set per vertex.
    scope.charge(
        work=int(np.asarray(protected_face_ids).size)
        + 2 * int(np.asarray(mesh.vertex_global_ids).size)
    )
    protected = set(
        int(value) for value in np.asarray(protected_face_ids, dtype=np.int64).reshape(-1)
    )
    vertex_support = {
        int(identifier): {int(identifier)}
        for identifier in np.asarray(mesh.vertex_global_ids)
    }
    source_geometry = (
        _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(mesh)
        if source_geometry is None
        else source_geometry
    )
    with _polyhedral_exact_phase(
        scope, budget, maximum_work_units, maximum_scratch_bytes
    ):
        source_geometry.resolve(mesh, budget=budget)
    source_power = polyhedral_shared_owner.bound_power_source(source_geometry)
    return _PolyhedralSource(
        connectivity=connectivity,
        count=count,
        classes=classes,
        protected=protected,
        vertex_support=vertex_support,
        source_geometry=source_geometry,
        source_power=source_power,
    )


def _execute_polyhedral_plane(
    execution: _PolyhedralExecution,
    source: _PolyhedralSource,
    state: _PolyhedralCandidate,
    plane: np.ndarray,
) -> _PolyhedralCandidate:
    with _polyhedral_topology_phase(
        execution.scope, state.target, split=True,
        periodic_topology=False,
        limits=execution.limits,
    ) as (storage, bound):
        ids = np.asarray(state.target.vertex_global_ids)
        edges = np.asarray(
            _dependency_phydrax_meshing__polyhedral_generation.require_polyhedral_connectivity(
                state.target
            ).edges
        )
        # One new support set per intersected edge; the union has no more
        # entries than its two actual predecessor sets together.
        # One visit per support set, then two lookups and one sum per edge.
        execution.scope.charge(
            work=len(source.vertex_support) + 3 * int(edges.shape[0])
        )
        support_slots = sum(len(values) for values in source.vertex_support.values())
        for first, second in edges:
            support_slots += (
                len(source.vertex_support[int(ids[first])])
                + len(source.vertex_support[int(ids[second])])
            )
        support_bytes = 256 * support_slots
        storage.workspace.set_bound(bound.construction + support_bytes)
        candidate = _build_polyhedral_plane(execution, source, state, plane)
        storage.retain(
            bound.retained + support_bytes, candidate.target,
            *_polyhedral_geometry_aliases(candidate.target_geometry),
            record_count_upper=bound.records,
        )
        return candidate


def _build_polyhedral_plane(
    execution: _PolyhedralExecution,
    source: _PolyhedralSource,
    state: _PolyhedralCandidate,
    plane: np.ndarray,
) -> _PolyhedralCandidate:
    request = execution.request
    maximum_cells = execution.maximum_cells
    maximum_vertices = execution.maximum_vertices
    maximum_work_units = execution.maximum_work_units
    maximum_scratch_bytes = execution.maximum_scratch_bytes
    scope = execution.scope
    budget = execution.budget
    protected = source.protected
    vertex_support = source.vertex_support
    target = state.target
    target_geometry = state.target_geometry
    residual = state.residual
    construction = state.construction
    if polyhedral_shared_owner.bound_power_source(target_geometry) is None:
        split = _split(
            target,
            plane,
            protected,
            request.coordinate_tolerance,
            maximum_cells,
            maximum_vertices,
            vertex_support,
            scope=scope,
            budget=budget,
        )
        candidate = split.mesh
        target_geometry = (
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(
                candidate
            )
        )
        residual = max(residual, split.residual)
    else:
        with _polyhedral_exact_phase(
            scope, budget, maximum_work_units, maximum_scratch_bytes
        ):
            candidate, target_geometry = (
                plane_restriction_owner.restrict_polyhedral_plane(
                    target,
                    target_geometry,
                    plane,
                    protected,
                    vertex_support,
                    maximum_cells=maximum_cells,
                    maximum_vertices=maximum_vertices,
                    maximum_work=maximum_work_units,
                    maximum_bytes=maximum_scratch_bytes,
                    budget=budget,
                )
            )
    # One singleton support set per candidate vertex.
    execution.scope.charge(work=2 * int(np.asarray(target.vertex_global_ids).size))
    support = {
        int(identifier): {int(identifier)}
        for identifier in np.asarray(target.vertex_global_ids)
    }
    staged = _stage_polyhedral_target(
        target,
        candidate,
        target_geometry,
        support,
        numeric_version="polyhedral-plane-candidate",
        maximum_work=maximum_work_units,
        maximum_bytes=maximum_scratch_bytes,
        scope=scope,
        budget=budget,
    )
    candidate_edit = _periodic_polyhedral_edit(
        target,
        staged,
        _topology_edit(target, staged, None, support, scope=scope),
        scope=scope,
        budget=budget,
    )
    target = _assemble_polyhedral_candidate(
        target, staged, candidate_edit,
        numeric_version="polyhedral-plane-candidate",
        scope=scope, budget=budget,
    )
    return _PolyhedralCandidate(
        target=target,
        target_geometry=target_geometry,
        residual=residual,
        construction=construction,
    )


def _assemble_polyhedral_candidate(
    source: CellMesh,
    staged: CellMesh,
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit,
    *,
    numeric_version: str,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellMesh:
    with _polyhedral_topology_phase(
        scope, staged, periodic_source=source
    ) as (storage, bound):
        source_bound = _polyhedral_adaptation_storage_bound(
            _polyhedral_incidence(source, scope=scope)._replace(rank=0)
        )
        # Lineage may have many more overlap relations than target cells.
        # Bound its merged keys, sort/gather/mask banks and Python row records
        # from the actual input edit, not from the mesh's cell count.
        relation_storage = sum(
            24 * 8 * (row.source_keys.size + row.target_keys.size + row.kinds.size)
            + 512 * row.kinds.size
            for row in edit.relations
        )
        storage.workspace.set_bound(
            bound.construction + source_bound.construction + relation_storage
            + _polyhedral_cross_key_storage(source, staged)
        )
        with _polyhedral_exact_phase(
            scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
        ):
            target, _, _ = _dependency_phydrax_meshing__topology_edit.assemble_topology_edit(
                source, edit, numeric_version=numeric_version, budget=budget
            )
        storage.retain(bound.retained, target, record_count_upper=bound.records)
        return target


def _execute_polyhedral_agglomeration(
    execution: _PolyhedralExecution,
    prepared_source: _PolyhedralSource,
) -> _PolyhedralCandidate:
    with _polyhedral_topology_phase(
        execution.scope, execution.mesh, periodic_topology=False,
        limits=execution.limits,
    ) as (storage, bound):
        candidate = _build_polyhedral_agglomeration(execution, prepared_source)
        storage.retain(
            bound.retained, candidate.target,
            *_polyhedral_geometry_aliases(candidate.target_geometry),
            record_count_upper=bound.records,
        )
        return candidate


def _build_polyhedral_agglomeration(
    execution: _PolyhedralExecution,
    prepared_source: _PolyhedralSource,
) -> _PolyhedralCandidate:
    mesh = execution.mesh
    request = execution.request
    maximum_work_units = execution.maximum_work_units
    maximum_scratch_bytes = execution.maximum_scratch_bytes
    scope = execution.scope
    budget = execution.budget
    classes = prepared_source.classes
    protected = prepared_source.protected
    source_power = prepared_source.source_power
    target = _agglomerate(
        mesh, request.agglomerations, classes, protected, scope=scope
    )
    if source_power is None:
        target_geometry = (
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(
                target
            )
        )
    else:
        target_connectivity = target.connectivity
        match target_connectivity:
            case PolyhedralConnectivity():
                pass
            case _:
                raise TypeError(
                    "polyhedral_adaptation.target.connectivity must retain packed incidence for source restriction; got another connectivity kind"
                )
        used = np.unique(np.asarray(target_connectivity.cell_vertex_values))
        lowering = np.full(target.coordinates.shape[0], -1, dtype=np.int32)
        lowering[used] = np.arange(used.size, dtype=np.int32)
        # Agglomeration compacts away interior vertices. The rational geometry
        # still uses source rows, not the target's compact rows.
        source_rows = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids))
        }
        retained_source_rows = np.asarray(
            [
                source_rows[int(identifier)]
                for identifier in np.asarray(target.vertex_global_ids)[used]
            ],
            dtype=np.int64,
        )
        source = _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource(
            source_power,
            np.empty((0, 4), dtype=np.float64),
            np.column_stack(
                (retained_source_rows, np.full(used.size, -1, dtype=np.int64))
            ),
            np.full(used.size, -1, dtype=np.int64),
        )
        with _polyhedral_exact_phase(
            scope, budget, maximum_work_units, maximum_scratch_bytes
        ):
            target = CellMesh.from_polyhedra(
                source.prepare(budget=budget).rounded_vertices,
                tuple(
                    tuple(lowering[row] for row in cell)
                    for cell in polyhedral_shared_owner.polyhedral_face_loops(target)
                ),
                vertex_global_ids=np.asarray(target.vertex_global_ids)[used],
                cell_global_ids=np.asarray(target_connectivity.cell_global_ids),
            )
            target_geometry = (
                _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
                    target, source, budget=budget
                )
            )
    residual = 0.0
    construction = None
    return _PolyhedralCandidate(
        target=target,
        target_geometry=target_geometry,
        residual=residual,
        construction=construction,
    )


def _execute_polyhedral_regeneration(
    execution: _PolyhedralExecution,
    source: _PolyhedralSource,
) -> _PolyhedralCandidate:
    mesh = execution.mesh
    request = execution.request
    maximum_cells = execution.maximum_cells
    maximum_vertices = execution.maximum_vertices
    maximum_work_units = execution.maximum_work_units
    maximum_scratch_bytes = execution.maximum_scratch_bytes
    source_id = execution.source_id
    scope = execution.scope
    budget = execution.budget
    protected = source.protected
    if request.complex_ is None or request.sites is None:
        raise ValueError(
            "PolyhedralMeshAdaptation.sites must accompany a PLC for regeneration; got missing PLC or sites"
        )
    if protected:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.protected_face_ids must be empty for site regeneration; got protected faces"
        )
    # Regeneration replaces the complete source, even when its sites happen to
    # reproduce the same topology. It is not a zero-cell local edit.
    with scope.batch(0, cavity_upper=source.count) as ticket:
        ticket.consume(cavity_cells=source.count)
    schedule = replace(
        request.schedule,
        maximum_cells=min(maximum_cells, request.schedule.maximum_cells),
        maximum_vertices=min(maximum_vertices, request.schedule.maximum_vertices),
        maximum_work_units=min(maximum_work_units, request.schedule.maximum_work_units),
        maximum_scratch_bytes=min(
            maximum_scratch_bytes, request.schedule.maximum_scratch_bytes
        ),
    )

    try:
        with budget.measurement():
            construction = _dependency_phydrax_meshing__polyhedral_generation.generate_polyhedral_volume(
                request.complex_,
                sites=request.sites,
                weights=request.weights,
                schedule=schedule,
                source_id=source_id,
                periodic_constraints=request.periodic_constraints,
                scope=scope,
                budget=budget,
                limits=execution.limits,
            )
    except (
        MeshcoreError,
        _dependency_phydrax_meshing__polyhedral_generation.PolyhedralGenerationError,
    ) as error:
        raise _owner_phydrax_meshing_providers__native_polyhedral.construction_failure(
            error, "polyhedral_adaptation"
        ) from error
    target, target_geometry = _regenerated_target(
        mesh, construction, scope=scope, budget=budget
    )
    residual = 0.0
    return _PolyhedralCandidate(
        target=target,
        target_geometry=target_geometry,
        residual=residual,
        construction=construction,
    )


def _certify_polyhedral_candidate(
    execution: _PolyhedralExecution,
    source: _PolyhedralSource,
    candidate: _PolyhedralCandidate,
) -> _PolyhedralCoverage:
    mesh = execution.mesh
    maximum_cells = execution.maximum_cells
    maximum_vertices = execution.maximum_vertices
    maximum_work_units = execution.maximum_work_units
    maximum_scratch_bytes = execution.maximum_scratch_bytes
    common_refinement_policy = execution.common_refinement_policy
    numeric_version = execution.numeric_version
    scope = execution.scope
    budget = execution.budget
    classes = source.classes
    vertex_support = source.vertex_support
    source_geometry = source.source_geometry
    source_power = source.source_power
    target = candidate.target
    target_geometry = candidate.target_geometry
    construction = candidate.construction
    if target.connectivity.cell_count > maximum_cells:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.maximum_cells must admit the candidate; got a cell budget refusal",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    if target.coordinates.shape[0] > maximum_vertices:
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.maximum_vertices must admit the candidate; got a vertex budget refusal",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    policy = (
        _dependency_phydrax_geometry__supermesh.CommonRefinementPolicy(
            overlap_simplices=True
        )
        if common_refinement_policy is None
        else common_refinement_policy
    )
    if (
        policy.coverage
        is not _dependency_phydrax_geometry__supermesh.CommonRefinementCoverage.COMPLETE
    ):
        raise ValueError(
            "polyhedral_adaptation.common_refinement_policy.coverage must cover source and target; got partial coverage"
        )
    target = _stage_polyhedral_target(
        mesh,
        target,
        target_geometry,
        vertex_support,
        numeric_version=numeric_version,
        maximum_work=maximum_work_units,
        maximum_bytes=maximum_scratch_bytes,
        scope=scope,
        budget=budget,
    )
    if construction is not None:
        power_source = polyhedral_shared_owner.bound_power_source(construction.geometry)
        if power_source is None:
            raise ValueError(
                "polyhedral_adaptation.construction.geometry must retain its power source; got missing source construction"
            )
    else:
        power_source = polyhedral_shared_owner.bound_power_source(target_geometry)
    with _polyhedral_exact_phase(
        scope, budget, maximum_work_units, maximum_scratch_bytes
    ):
        target_geometry = (
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(
                target
            )
            if power_source is None
            else _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
                target, power_source, budget=budget
            )
        )
    with _polyhedral_exact_phase(
        scope, budget, maximum_work_units, maximum_scratch_bytes
    ):
        common = _dependency_phydrax_geometry__supermesh.prepare_common_refinement(
            mesh,
            target,
            policy=policy,
            source_geometry=source_geometry if source_power is not None else None,
            target_geometry=target_geometry if power_source is not None else None,
            scope=scope,
            budget=budget,
        )
    if (
        common.status
        != _dependency_phydrax_geometry__supermesh.CommonRefinementStatus.SUCCESS
    ):
        raise polyhedral_shared_owner.PolyhedralAdaptationError(
            "polyhedral_adaptation.common_refinement.status must report success; "
            f"got {common.status.name}; reason={common.evidence.reason}; "
            f"retained_bytes={common.evidence.retained_bytes}, "
            f"working_bytes={common.evidence.working_bytes}",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED
            if common.status
            is _dependency_phydrax_geometry__supermesh.CommonRefinementStatus.RESOURCE_LIMIT
            else MeshingFailureCategory.AUDIT_FAILED,
        )
    with NativeHostStorageWorkspace(scope) as class_storage:
        # Imported overlap indices, class gather, unique-sort banks and masks.
        # The actual accepted overlap count bounds every target row.
        class_storage.set_bound(
            12 * 256 + 8 * (
                common.target_offsets.size + 6 * common.source_cells.size
            )
        )
        offsets = np.asarray(common.target_offsets)
        source_cells = np.asarray(common.source_cells)
        # Per target row: offset reads, a gather and a sort-based unique over
        # its accepted overlap owners; summed over every accepted overlap row.
        rows = int(source_cells.size)
        scope.charge(
            work=4 * common.target_cell_count + rows * (2 + max(1, rows).bit_length())
        )
        for row in range(common.target_cell_count):
            begin, end = offsets[row:row + 2]
            owners = source_cells[begin:end]
            if np.unique(classes[owners]).size != 1:
                raise polyhedral_shared_owner.PolyhedralAdaptationError(
                    "polyhedral_adaptation.cell_classes must agree on each target overlap; got mixed predecessor classes"
                )
    return _PolyhedralCoverage(
        target=target,
        target_geometry=target_geometry,
        common=common,
    )


def _publish_polyhedral_candidate(
    execution: _PolyhedralExecution,
    source: _PolyhedralSource,
    candidate: _PolyhedralCandidate,
    coverage: _PolyhedralCoverage,
) -> PolyhedralAdaptationOutcome:
    with _polyhedral_topology_phase(
        execution.scope, coverage.target, periodic_source=execution.mesh
    ) as (storage, bound):
        # Accepted periodic assembly, identity differences and orbit/authority
        # records coexist with the certified target, but never mutate it.
        source_count = execution.mesh.connectivity.cell_count
        source_bound = _polyhedral_adaptation_storage_bound(
            _polyhedral_incidence(execution.mesh, scope=execution.scope)._replace(rank=0)
        )
        storage.workspace.set_bound(
            bound.construction + source_bound.construction + 128 * source_count + 256
        )
        outcome = _build_polyhedral_publication(
            execution, source, candidate, coverage
        )
        storage.retain(
            bound.retained + 8 * source_count,
            *(leaf for leaf in jax.tree_util.tree_leaves((outcome.edit, outcome.evidence))
              if eqx.is_array(leaf)),
            record_count_upper=bound.records,
        )
        return outcome


def _build_polyhedral_publication(
    execution: _PolyhedralExecution,
    source: _PolyhedralSource,
    candidate: _PolyhedralCandidate,
    coverage: _PolyhedralCoverage,
) -> PolyhedralAdaptationOutcome:
    mesh = execution.mesh
    request = execution.request
    maximum_work_units = execution.maximum_work_units
    maximum_scratch_bytes = execution.maximum_scratch_bytes
    numeric_version = execution.numeric_version
    scope = execution.scope
    budget = execution.budget
    connectivity = source.connectivity
    count = source.count
    vertex_support = source.vertex_support
    source_geometry = source.source_geometry
    target = coverage.target
    target_geometry = coverage.target_geometry
    residual = candidate.residual
    construction = candidate.construction
    common = coverage.common
    edit = _topology_edit(mesh, target, common, vertex_support, scope=scope)
    edit = _edge_trace_stencil(
        mesh,
        target,
        source_geometry,
        target_geometry,
        edit,
        maximum_work=maximum_work_units,
        maximum_bytes=maximum_scratch_bytes,
        scope=scope,
        budget=budget,
    )
    if mesh.periodic_topology is not None:
        with _polyhedral_exact_phase(
            scope, budget, maximum_work_units, maximum_scratch_bytes
        ):
            authority = (
                None
                if construction is None
                else _owner_phydrax_meshing__topology_edit.prepare_periodic_nonnested_geometry(
                    mesh,
                    target,
                    source_geometry,
                    target_geometry,
                    common,
                    budget=budget,
                )
            )
        edit = _periodic_polyhedral_edit(
            mesh, target, edit, authority, scope=scope, budget=budget
        )
        accepted = _assemble_polyhedral_candidate(
            mesh, target, edit, numeric_version=numeric_version,
            scope=scope, budget=budget,
        )
        if accepted.mesh_id != target.mesh_id:
            raise ValueError(
                "polyhedral_adaptation.target.mesh_id must retain the certified overlap identity; got a changed publication"
            )
    target_connectivity = target.connectivity
    match target_connectivity:
        case PolyhedralConnectivity():
            pass
        case _:
            raise TypeError(
                "polyhedral_adaptation.target.connectivity must be packed polyhedral incidence; got another connectivity kind"
            )
    source_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target_connectivity.cell_global_ids, dtype=np.int64)
    # setdiff1d sorts both ID arrays, then merges them once.
    total = source_ids.size + target_ids.size
    scope.charge(work=total * max(1, (total + 1).bit_length()))
    evidence = PolyhedralAdaptationEvidence(
        request.operation,
        count,
        target_connectivity.cell_count,
        residual,
        common.refinement_id,
        np.setdiff1d(source_ids, target_ids),
        None if construction is None else construction.construction_id,
        None if construction is None else construction.diagram.points,
        None if construction is None else construction.diagram.weights,
        None if construction is None else construction.site_parents,
        None if construction is None else construction.cell_sites,
        None if construction is None else construction.component_ids,
        None if construction is None else construction.cell_regions,
        None if construction is None else construction.face_facets,
        None if construction is None else construction.feature_edge_sources,
    )
    return PolyhedralAdaptationOutcome(
        edit, common, evidence, construction, target_geometry
    )


def _adapt_polyhedral_mesh(
    mesh: CellMesh,
    request: PolyhedralMeshAdaptation,
    *,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None,
    cell_classes: ConvertibleToArray | None,
    protected_face_ids: ConvertibleToArray,
    maximum_cells: int,
    maximum_vertices: int,
    maximum_work_units: int,
    maximum_scratch_bytes: int,
    source_id: str,
    common_refinement_policy: _dependency_phydrax_geometry__supermesh.CommonRefinementPolicy
    | None,
    numeric_version: str,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    limits: contracts_owner.MeshingLimits | None,
) -> PolyhedralAdaptationOutcome:
    """Prepare, execute, certify, and publish one bounded polyhedral candidate."""
    execution = _PolyhedralExecution(
        mesh=mesh,
        request=request,
        maximum_cells=maximum_cells,
        maximum_vertices=maximum_vertices,
        maximum_work_units=maximum_work_units,
        maximum_scratch_bytes=maximum_scratch_bytes,
        source_id=source_id,
        common_refinement_policy=common_refinement_policy,
        numeric_version=numeric_version,
        scope=scope,
        budget=budget,
        limits=limits,
    )
    with NativeHostStorageWorkspace(scope) as source_storage:
        # Imported class/identity copies and Python source-control banks remain
        # live through every candidate, but do not escape in the published edit.
        def input_count(value: object) -> int:
            if isinstance(value, (tuple, list)):
                return sum(input_count(item) for item in value)
            shape = getattr(value, "shape", ())
            return math.prod(shape)

        source_storage.set_bound(
            16 * 256 + 768 * mesh.coordinates.shape[0]
            + 256 * input_count(protected_face_ids)
            + 64 * mesh.connectivity.cell_count + 8 * request.planes.size
        )
        return _run_polyhedral_execution(
            execution, source_geometry, cell_classes, protected_face_ids
        )


def _run_polyhedral_execution(
    execution: _PolyhedralExecution,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec | None,
    cell_classes: ConvertibleToArray | None,
    protected_face_ids: ConvertibleToArray,
) -> PolyhedralAdaptationOutcome:
    source = _admit_polyhedral_source(
        execution, source_geometry, cell_classes, protected_face_ids
    )
    match execution.request.operation:
        case PolyhedralAdaptationOperation.PLANE_SPLIT:
            candidate = _PolyhedralCandidate(execution.mesh, source.source_geometry, 0.0, None)
            for plane in np.asarray(execution.request.planes):
                candidate = _execute_polyhedral_plane(execution, source, candidate, plane)
        case PolyhedralAdaptationOperation.AGGLOMERATE:
            candidate = _execute_polyhedral_agglomeration(execution, source)
        case PolyhedralAdaptationOperation.REGENERATE:
            candidate = _execute_polyhedral_regeneration(execution, source)
        case invalid:
            assert_never(invalid)
    coverage = _certify_polyhedral_candidate(execution, source, candidate)
    return _publish_polyhedral_candidate(execution, source, candidate, coverage)
