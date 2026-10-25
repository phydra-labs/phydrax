#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Domain-restricted native power-cell generation on recovered PLC tetrahedra.

The tetrahedral carrier is constrained recovery, never an unconstrained convex
hull filtered by centroids. Cells join only within a material across unconstrained
faces. Non-star-visible components are decomposed into convex restricted pieces
for the existing degree-one VEM/FV consumers; their site/component lineage is
retained, not confused with a nodal interpolation map. Acceptance is independent
of this construction and belongs to ``providers._native_polyhedral``.

Native callers pass their request scope. Rational host phases reserve local
tickets, retain one coordinate view across source/certificate preparation, and
keep its live cache bytes reserved between native construction phases.
"""

from __future__ import annotations

import fractions as _owner_fractions
import math
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import assert_never, final, SupportsFloat, SupportsIndex

import equinox as eqx
import jax
import numpy as np

import phydrax._bvh as _owner_phydrax__bvh
import phydrax._meshcore as _owner_phydrax__meshcore
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._cell_mesh as _dependency_phydrax_discretization__cell_mesh
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._exact_power_geometry as _dependency_phydrax_discretization__exact_power_geometry
import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.discretization._periodic_topology as _dependency_phydrax_discretization__periodic_topology
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._exact_polyhedral_geometry as _owner_phydrax_geometry__exact_polyhedral_geometry
import phydrax.geometry._periodic_power_preparation as periodic_power_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.geometry._triangulation as _dependency_phydrax_geometry__triangulation
import phydrax.linalg._hermitian_spectral as feature_fraction_owner
import phydrax.linalg._small_batched as _owner_phydrax_linalg__small_batched
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._periodic_constraint as periodic_constraint_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_storage import (
    declared_object_storage_bytes,
    retained_host_storage_fanout_upper,
    retained_host_storage_profile,
)
from .._meshcore import (
    NativeBatchTicket,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    NativeScopeTicket,
    PlcRecovery3D,
    recover_plc_3d,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_finite_float, positive_integer
from ..discretization._cell_complex import PolyhedralConnectivity
from ..geometry._rational import rational_vector
from ..typing import checked, ConvertibleToArray, parse


@final
class NativePolyhedralSchedule(StrictModule, NonTrainableState):
    """Bound site refinement, weight optimization, and construction resources.

    ``maximum_cell_diameter`` bounds restricted cell vertices.
    ``target_site_volumes`` sums disconnected components of each site.
    Protected sites remain fixed during relocation; constrained curves retain
    recovered feature edges through cell decomposition.
    """

    maximum_vertices: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    maximum_refinement_steps: int = eqx.field(static=True)
    lloyd_steps: int = eqx.field(static=True)
    maximum_cell_diameter: float | None = eqx.field(static=True)
    target_site_volumes: tuple[float, ...] = eqx.field(static=True)
    relative_volume_tolerance: float = eqx.field(static=True)
    feature_tolerance: float = eqx.field(static=True)
    protected_sites: tuple[int, ...] = eqx.field(static=True)
    require_every_site: bool = eqx.field(static=True)

    def __init__(
        self,
        maximum_vertices: int = 1 << 22,
        maximum_cells: int = 1 << 20,
        maximum_work_units: int = 1 << 26,
        maximum_scratch_bytes: int = 8_000_000_000,
        maximum_refinement_steps: int = 8,
        lloyd_steps: int = 0,
        maximum_cell_diameter: float | None = None,
        target_site_volumes: Sequence[float] = (),
        relative_volume_tolerance: float = 1e-6,
        feature_tolerance: float = 1e-10,
        protected_sites: Sequence[int] = (),
        require_every_site: bool = False,
    ) -> None:
        vertices = positive_integer(maximum_vertices, "maximum_vertices")
        cells = positive_integer(maximum_cells, "maximum_cells")
        work_units = positive_integer(maximum_work_units, "maximum_work_units")
        scratch_bytes = positive_integer(maximum_scratch_bytes, "maximum_scratch_bytes")
        refinement_steps = nonnegative_integer(
            maximum_refinement_steps, "maximum_refinement_steps"
        )
        relocation_steps = nonnegative_integer(lloyd_steps, "lloyd_steps")
        volume_tolerance = positive_finite_float(
            relative_volume_tolerance,
            "NativePolyhedralSchedule.relative_volume_tolerance",
        )
        distance_tolerance = positive_finite_float(
            feature_tolerance, "NativePolyhedralSchedule.feature_tolerance"
        )
        diameter = (
            None
            if maximum_cell_diameter is None
            else positive_finite_float(
                maximum_cell_diameter, "NativePolyhedralSchedule.maximum_cell_diameter"
            )
        )
        if any(not math.isfinite(x) or x <= 0 for x in target_site_volumes):
            raise ValueError(
                "NativePolyhedralSchedule.target_site_volumes must be finite and positive; "
                f"got {target_site_volumes!r}"
            )
        volumes = tuple(float(value) for value in target_site_volumes)
        protected = tuple(protected_sites)
        self.maximum_vertices = vertices
        self.maximum_cells = cells
        self.maximum_work_units = work_units
        self.maximum_scratch_bytes = scratch_bytes
        self.maximum_refinement_steps = refinement_steps
        self.lloyd_steps = relocation_steps
        self.maximum_cell_diameter = diameter
        self.target_site_volumes = volumes
        self.relative_volume_tolerance = volume_tolerance
        self.feature_tolerance = distance_tolerance
        self.protected_sites = protected
        self.require_every_site = require_every_site

    @property
    def schedule_id(self) -> str:
        """Return the canonical fingerprint of the construction controls."""
        return canonical_fingerprint(
            {
                "maximum_vertices": self.maximum_vertices,
                "maximum_cells": self.maximum_cells,
                "maximum_work_units": self.maximum_work_units,
                "maximum_scratch_bytes": self.maximum_scratch_bytes,
                "maximum_refinement_steps": self.maximum_refinement_steps,
                "lloyd_steps": self.lloyd_steps,
                "maximum_cell_diameter": self.maximum_cell_diameter,
                "target_site_volumes": self.target_site_volumes,
                "relative_volume_tolerance": self.relative_volume_tolerance,
                "feature_tolerance": self.feature_tolerance,
                "protected_sites": self.protected_sites,
                "require_every_site": self.require_every_site,
            }
        )

    def native_limits(self) -> NativeExecutionLimits:
        """Return native execution bounds for construction."""
        return _dependency_phydrax_meshing__contracts.MeshingLimits(
            maximum_vertices=self.maximum_vertices,
            maximum_cells=self.maximum_cells,
            maximum_work_units=self.maximum_work_units,
            maximum_scratch_bytes=self.maximum_scratch_bytes,
        ).native_limits()


@dataclass(frozen=True, slots=True)
class PolyhedralConstruction:
    """Retain the mesh, source geometry, and construction ancestry.

    Cell and face banks follow mesh entity order. Site volumes sum disconnected
    components, while component IDs distinguish their connectivity.
    """

    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    domain: domain_coverage_owner.PiecewiseLinearDomain
    recovery: PlcRecovery3D
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram
    cell_regions: np.ndarray
    cell_sites: np.ndarray
    face_facets: np.ndarray
    face_recovery_triangles: np.ndarray
    interface_faces: np.ndarray
    sheet_faces: np.ndarray
    feature_edge_sources: np.ndarray
    feature_source_edges: np.ndarray
    feature_maximum_distance: np.ndarray
    feature_coverage_gap: np.ndarray
    site_parents: np.ndarray
    component_ids: np.ndarray
    split_sites: np.ndarray
    requested_site_volumes: np.ndarray
    achieved_site_volumes: np.ndarray
    refinement_steps: int
    lloyd_steps: int
    optimization_relative_residual: float
    maximum_cell_diameter: float
    construction_id: str
    work_units: int
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...] = ()
    seam_fragments: tuple[PeriodicPolyhedralSeamFragment, ...] = ()
    seam_face_pairs: np.ndarray | None = None
    seam_vertex_pairs: np.ndarray | None = None


@checked
def polyhedral_periodic_group(
    constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PeriodicIsometryGroup | None:
    """Return the periodic group defined by authored transforms and tolerances."""
    if not constraints:
        return None
    if not all(
        isinstance(value, periodic_constraint_owner.PeriodicConstraint)
        for value in constraints
    ):
        raise TypeError(
            "polyhedral.periodic_constraints must contain PeriodicConstraint values; "
            f"got {tuple(type(value).__name__ for value in constraints)!r}"
        )
    matrices = np.stack(
        [np.asarray(value.transform, dtype=np.float64) for value in constraints]
    )
    # The group's numerical classification bound leaves each authored action
    # unchanged for source preparation and seam coverage.
    return PeriodicIsometryGroup(
        matrices, tolerance=max(value.tolerance for value in constraints), budget=budget
    )


class PolyhedralGenerationReason(StrEnum):
    """Closed scientific and resource refusal reasons for polyhedral generation."""

    AMBIGUOUS_FEATURE_ANCESTRY = "ambiguous_feature_ancestry"
    COINCIDENT_POSITIVE_VOLUME_TARGETS = "coincident_positive_volume_targets"
    FEATURE_CANDIDATE_BUDGET = "feature_candidate_budget"
    FEATURE_COVERAGE_UNRESOLVED = "feature_coverage_unresolved"
    FEATURE_MEASURE_UNREPRESENTABLE = "feature_measure_unrepresentable"
    FIXED_SITE_DIAMETER_UNRESOLVED = "fixed_site_diameter_unresolved"
    MISSING_FEATURE_EDGE = "missing_feature_edge"
    PERIODIC_PIECE_GEOMETRY_UNRESOLVED = "periodic_piece_geometry_unresolved"
    PERIODIC_SEAM_DEGENERATE_EDGE = "periodic_seam_degenerate_edge"
    PERIODIC_SEAM_DEGENERATE_TRIANGLE = "periodic_seam_degenerate_triangle"
    PERIODIC_SEAM_DOUBLE_COVERAGE = "periodic_seam_double_coverage"
    PERIODIC_SEAM_EDGE_RANK = "periodic_seam_edge_rank"
    PERIODIC_SEAM_EXACT_COVERAGE = "periodic_seam_exact_coverage"
    PERIODIC_SEAM_FACET_SCOPE = "periodic_seam_facet_scope"
    PERIODIC_SEAM_MISSING_FACET = "periodic_seam_missing_facet"
    PERIODIC_SEAM_ORBIT_CYCLE = "periodic_seam_orbit_cycle"
    PERIODIC_SEAM_SOURCE_ACTION = "periodic_seam_source_action"
    PERIODIC_SEAM_SOURCE_BINDING = "periodic_seam_source_binding"
    PERIODIC_SEAM_SOURCE_CONTAINMENT = "periodic_seam_source_containment"
    PERIODIC_SEAM_SOURCE_RANK = "periodic_seam_source_rank"
    PERIODIC_SEAM_TRANSFORM_DIMENSION = "periodic_seam_transform_dimension"
    PERIODIC_SEAM_WINDING_BUDGET = "periodic_seam_winding_budget"
    PERIODIC_VERTEX_BUDGET = "periodic_vertex_budget"
    PERIODIC_VERTEX_IDENTITY_BUDGET = "periodic_vertex_identity_budget"
    PERIODIC_VOLUME_OPTIMIZATION_UNRESOLVED = "periodic_volume_optimization_unresolved"
    PIECE_GEOMETRY_UNRESOLVED = "piece_geometry_unresolved"
    REFINEMENT_UNRESOLVED = "refinement_unresolved"
    REQUIRED_EMPTY_SITES = "required_empty_sites"
    SITE_INSERTION_STALLED = "site_insertion_stalled"
    SOURCE_VERTEX_BUDGET = "source_vertex_budget"
    WEIGHT_JACOBIAN_UNRESOLVED = "weight_jacobian_unresolved"
    WORK_BUDGET = "work_budget"

    @property
    def failure_category(self) -> _leaf_phydrax_meshing__failures.MeshingFailureCategory:
        match self:
            case (
                PolyhedralGenerationReason.FEATURE_CANDIDATE_BUDGET
                | PolyhedralGenerationReason.PERIODIC_SEAM_WINDING_BUDGET
                | PolyhedralGenerationReason.PERIODIC_VERTEX_BUDGET
                | PolyhedralGenerationReason.PERIODIC_VERTEX_IDENTITY_BUDGET
                | PolyhedralGenerationReason.SOURCE_VERTEX_BUDGET
                | PolyhedralGenerationReason.WORK_BUDGET
            ):
                return _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            case (
                PolyhedralGenerationReason.AMBIGUOUS_FEATURE_ANCESTRY
                | PolyhedralGenerationReason.COINCIDENT_POSITIVE_VOLUME_TARGETS
                | PolyhedralGenerationReason.FEATURE_COVERAGE_UNRESOLVED
                | PolyhedralGenerationReason.FEATURE_MEASURE_UNREPRESENTABLE
                | PolyhedralGenerationReason.FIXED_SITE_DIAMETER_UNRESOLVED
                | PolyhedralGenerationReason.MISSING_FEATURE_EDGE
                | PolyhedralGenerationReason.PERIODIC_PIECE_GEOMETRY_UNRESOLVED
                | PolyhedralGenerationReason.PERIODIC_SEAM_DEGENERATE_EDGE
                | PolyhedralGenerationReason.PERIODIC_SEAM_DEGENERATE_TRIANGLE
                | PolyhedralGenerationReason.PERIODIC_SEAM_DOUBLE_COVERAGE
                | PolyhedralGenerationReason.PERIODIC_SEAM_EDGE_RANK
                | PolyhedralGenerationReason.PERIODIC_SEAM_EXACT_COVERAGE
                | PolyhedralGenerationReason.PERIODIC_SEAM_FACET_SCOPE
                | PolyhedralGenerationReason.PERIODIC_SEAM_MISSING_FACET
                | PolyhedralGenerationReason.PERIODIC_SEAM_ORBIT_CYCLE
                | PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_ACTION
                | PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_BINDING
                | PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_CONTAINMENT
                | PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_RANK
                | PolyhedralGenerationReason.PERIODIC_SEAM_TRANSFORM_DIMENSION
                | PolyhedralGenerationReason.PERIODIC_VOLUME_OPTIMIZATION_UNRESOLVED
                | PolyhedralGenerationReason.PIECE_GEOMETRY_UNRESOLVED
                | PolyhedralGenerationReason.REFINEMENT_UNRESOLVED
                | PolyhedralGenerationReason.REQUIRED_EMPTY_SITES
                | PolyhedralGenerationReason.SITE_INSERTION_STALLED
                | PolyhedralGenerationReason.WEIGHT_JACOBIAN_UNRESOLVED
            ):
                return _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED
            case _:
                assert_never(self)


class PolyhedralGenerationError(RuntimeError):
    """Report unmet construction bounds with entity and quantity evidence."""

    def __init__(
        self,
        reason: PolyhedralGenerationReason,
        *,
        entities: tuple[int, ...] = (),
        achieved: int | float | np.ndarray | dict[str, float] | None = None,
    ) -> None:
        self.reason = parse(
            reason, PolyhedralGenerationReason, "PolyhedralGenerationError.reason"
        )
        self.entities, self.achieved = entities, achieved
        super().__init__(
            "polyhedral.generation must satisfy the construction request; "
            f"got reason={self.reason}, entities={entities}, achieved={achieved}"
        )


@dataclass(frozen=True, slots=True)
class PeriodicPolyhedralSeamFragment:
    """Retain a common-refinement polygon and both source parent actions.

    ``vertices`` are in the target image. Source coefficients act on the
    source triangle BEFORE the authored transform; target coefficients act
    directly on the target triangle. Neither action uses rounded vertices.
    """

    constraint_id: str
    source_face: int
    target_face: int
    source_triangle: tuple[int, int, int]
    target_triangle: tuple[int, int, int]
    vertices: tuple[tuple[Fraction, ...], ...]
    source_coefficients: tuple[tuple[Fraction, ...], ...]
    target_coefficients: tuple[tuple[Fraction, ...], ...]
    projected_measure: Fraction
    projection_axes: tuple[int, ...]
    relative_orientation: int


def _seam_triangle_actions(
    triangle: tuple[tuple[Fraction, ...], ...],
    vertices: tuple[tuple[Fraction, ...], ...],
    axes: tuple[int, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:

    budget.reserve(0, 512 + 48 * len(vertices))
    matrix = (
        (Fraction(1),) * 3,
        tuple(point[axes[0]] for point in triangle),
        tuple(point[axes[1]] for point in triangle),
    )
    right = (
        (Fraction(1),) * len(vertices),
        tuple(point[axes[0]] for point in vertices),
        tuple(point[axes[1]] for point in vertices),
    )
    solved = _owner_phydrax_linalg__small_batched.prepare_exact_small_linear_actions(
        matrix, right, budget=budget
    )
    if solved.actions is None:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_RANK
        )
    actions = tuple(
        tuple(row[column] for row in solved.actions) for column in range(len(vertices))
    )
    planar_primitives_owner.reserve_fraction_work(
        (*triangle, *vertices), 40 * len(vertices), len(vertices), 6, budget=budget
    )
    for coefficients, point in zip(actions, vertices, strict=True):
        if any(value < 0 for value in coefficients) or sum(coefficients) != 1:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_CONTAINMENT
            )
        if any(
            sum(coefficients[index] * triangle[index][axis] for index in range(3))
            != point[axis]
            for axis in range(3)
        ):
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_ACTION
            )
    return actions


@dataclass(frozen=True, slots=True)
class _PeriodicSeamTriangle:
    face: int
    corners: tuple[int, int, int]
    points: tuple[tuple[Fraction, ...], ...]
    plane: tuple[Fraction, ...]
    axes: tuple[int, ...]
    projected: tuple[tuple[Fraction, ...], ...]
    measure: Fraction


def _seam_parent_triangles(
    coordinates: tuple[tuple[Fraction, ...], ...],
    point_array: np.ndarray,
    faces: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    matrix: tuple[tuple[Fraction, ...], ...] | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_PeriodicSeamTriangle, ...]:

    result = []
    for face in faces:
        a, b = offsets[face : face + 2]
        loop = tuple(map(int, rows[a:b]))
        work, storage = (
            _owner_phydrax_geometry__exact_polyhedral_geometry.triangulation_charge(
                point_array, loop, budget=budget
            )
        )
        # The loop proof charges its performed operations; admit only the bound.
        budget.admit_work_bound(work, storage_upper=storage)
        budget.reserve(0, storage)
        for (
            triangle
        ) in _owner_phydrax_geometry__exact_polyhedral_geometry.triangulate_loop(
            point_array, loop, budget=budget
        ):
            points = tuple(coordinates[index] for index in triangle)
            if matrix is not None:
                planar_primitives_owner.reserve_fraction_work(
                    points, 36, 9, 8, budget=budget
                )
                points = tuple(
                    tuple(
                        sum(matrix[axis][column] * point[column] for column in range(3))
                        + matrix[axis][3]
                        for axis in range(3)
                    )
                    for point in points
                )
            key = planar_primitives_owner.plane_key(points, budget=budget)
            if key is None:
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.PERIODIC_SEAM_DEGENERATE_TRIANGLE,
                    entities=(int(face),),
                )
            projected = planar_primitives_owner.project(points, key[1], budget=budget)
            budget.reserve(1, 512)
            result.append(
                _PeriodicSeamTriangle(
                    int(face),
                    triangle,
                    points,
                    key[0],
                    key[1],
                    projected,
                    abs(planar_primitives_owner.signed_measure(projected, budget=budget)),
                )
            )
    return tuple(result)


def _seam_pair_fragment(
    source: _PeriodicSeamTriangle,
    target: _PeriodicSeamTriangle,
    constraint_id: str,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PeriodicPolyhedralSeamFragment | None:

    budget.reserve(1)
    if source.plane != target.plane:
        return None
    # One exact source/target seam-triangle overlap query.
    budget.reserve_geometry_queries(1)
    raw_polygon = planar_primitives_owner.intersection_polygon(
        target.projected, source.projected, budget=budget
    )
    # Compact equal constructed intersections while retaining distinct source
    # coordinates and oriented clipping order.
    budget.reserve(len(raw_polygon), 256 + 16 * len(raw_polygon))
    polygon = tuple(
        point
        for index, point in enumerate(raw_polygon)
        if point != raw_polygon[index - 1]
    )
    measure = abs(planar_primitives_owner.signed_measure(polygon, budget=budget))
    if not measure:
        return None
    axes, plane = target.axes, target.plane
    missing_axis = next(axis for axis in range(3) if axis not in axes)
    planar_primitives_owner.reserve_fraction_work(
        (*source.points, *target.points, *polygon),
        6 * len(polygon),
        3 * len(polygon),
        12,
        budget=budget,
    )
    vertices = []
    for point in polygon:
        row = [Fraction(0)] * 3
        row[axes[0]], row[axes[1]] = point
        row[missing_axis] = (
            -(plane[3] + sum(plane[axis] * row[axis] for axis in axes))
            / plane[missing_axis]
        )
        vertices.append(tuple(row))
    bank = tuple(vertices)
    source_actions = _seam_triangle_actions(source.points, bank, axes, budget=budget)
    target_actions = _seam_triangle_actions(target.points, bank, axes, budget=budget)
    orientation = (
        1
        if planar_primitives_owner.signed_measure(source.projected, budget=budget)
        * planar_primitives_owner.signed_measure(target.projected, budget=budget)
        > 0
        else -1
    )
    fragment = PeriodicPolyhedralSeamFragment(
        constraint_id,
        source.face,
        target.face,
        source.corners,
        target.corners,
        bank,
        source_actions,
        target_actions,
        measure,
        axes,
        orientation,
    )
    budget.retain_basis(
        (
            fragment,
            constraint_id,
            source.face,
            target.face,
            source.corners,
            target.corners,
            bank,
            source_actions,
            target_actions,
            measure,
            axes,
            orientation,
        )
    )
    return fragment


def _seam_group_fragments(
    source: tuple[_PeriodicSeamTriangle, ...],
    target: tuple[_PeriodicSeamTriangle, ...],
    constraint_id: str,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[PeriodicPolyhedralSeamFragment, ...]:

    for triangles in (source, target):
        for index, first in enumerate(triangles):
            for second_index in range(index + 1, len(triangles)):
                second = triangles[second_index]
                budget.reserve(1)
                if first.plane != second.plane:
                    continue
                # One exact same-side seam-triangle overlap query.
                budget.reserve_geometry_queries(1)
                if (
                    planar_primitives_owner.intersection(
                        first.projected, second.projected, budget=budget
                    )
                    > 0
                ):
                    raise PolyhedralGenerationError(
                        PolyhedralGenerationReason.PERIODIC_SEAM_DOUBLE_COVERAGE,
                        entities=(first.face, second.face),
                    )
    budget.reserve(0, 256 + 16 * (len(source) + len(target)))
    covered_source = [Fraction(0) for _ in source]
    covered_target = [Fraction(0) for _ in target]
    result = []
    for source_index, first in enumerate(source):
        for target_index, second in enumerate(target):
            fragment = _seam_pair_fragment(first, second, constraint_id, budget=budget)
            if fragment is not None:
                budget.reserve(1, 16)
                planar_primitives_owner.reserve_fraction_work(
                    (
                        (
                            covered_source[source_index],
                            covered_target[target_index],
                            fragment.projected_measure,
                        ),
                    ),
                    2,
                    2,
                    2,
                    budget=budget,
                )
                result.append(fragment)
                covered_source[source_index] += fragment.projected_measure
                covered_target[target_index] += fragment.projected_measure
    for triangles, covered in ((source, covered_source), (target, covered_target)):
        bad = tuple(
            triangle.face
            for triangle, measure in zip(triangles, covered, strict=True)
            if triangle.measure != measure
        )
        if bad:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_SEAM_EXACT_COVERAGE, entities=bad
            )
    budget.reserve(0, 256 + 8 * len(result))
    return tuple(result)


@checked
def prepare_periodic_polyhedral_seam_fragments(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    face_facets: np.ndarray,
    constraints: Sequence[periodic_constraint_owner.PeriodicConstraint],
    /,
    *,
    maximum_work: int,
    maximum_scratch_bytes: int,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[PeriodicPolyhedralSeamFragment, ...]:
    """Return reciprocal seam common refinement of the source geometry.

    Scientific facet scopes select parent faces. Source triangulation and
    clipping handle unequal subdivisions and concave parent loops. Every
    selected triangle must have single coverage on both sides.
    """

    c = require_polyhedral_connectivity(mesh)
    facets = np.asarray(face_facets)
    if facets.shape != (c.face_count,) or not np.issubdtype(facets.dtype, np.integer):
        raise ValueError(
            "polyhedral.face_facets must name every face's source facet; "
            f"got shape={facets.shape}, dtype={facets.dtype}, faces={c.face_count}"
        )
    values = tuple(constraints)
    if not all(
        isinstance(value, periodic_constraint_owner.PeriodicConstraint)
        for value in values
    ):
        raise TypeError(
            "polyhedral.seam.constraints must contain PeriodicConstraint values; "
            f"got {tuple(type(value).__name__ for value in values)!r}"
        )
    offsets, rows = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    fragments: list[PeriodicPolyhedralSeamFragment] = []
    with budget.bound_stage(maximum_work, maximum_scratch_bytes):
        coordinates = geometry.rational_source_coordinates(budget=budget)
        geometry.resolve(mesh, exact_source_prepared=True, budget=budget)
        budget.reserve(0, 256 + 24 * len(coordinates))
        point_array = np.asarray(coordinates, dtype=object)
        for constraint in values:
            if (
                constraint.source_scope.entity_dimension != 2
                or constraint.target_scope.entity_dimension != 2
            ):
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.PERIODIC_SEAM_FACET_SCOPE
                )
            source_facets = tuple(
                map(int, np.asarray(constraint.source_scope.entity_ids))
            )
            target_facets = tuple(
                map(int, np.asarray(constraint.target_scope.entity_ids))
            )
            if constraint.source_entity_ids is None:
                groups = ((source_facets, target_facets),)
            else:
                groups = tuple(
                    ((int(source),), (target,))
                    for source, target in zip(
                        np.asarray(constraint.source_entity_ids),
                        target_facets,
                        strict=True,
                    )
                )
            matrix = tuple(
                rational_vector(row) for row in np.asarray(constraint.transform)
            )
            if len(matrix) != 4:
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.PERIODIC_SEAM_TRANSFORM_DIMENSION
                )
            for source_group, target_group in groups:
                # np.isin sorts each facet group and searches every face facet.
                selected = len(source_group) + len(target_group)
                budget.reserve(
                    (2 * len(facets) + selected)
                    * max(1, (selected + 1).bit_length())
                )
                source_faces = np.flatnonzero(np.isin(facets, source_group))
                target_faces = np.flatnonzero(np.isin(facets, target_group))
                if not source_faces.size or not target_faces.size:
                    raise PolyhedralGenerationError(
                        PolyhedralGenerationReason.PERIODIC_SEAM_MISSING_FACET,
                        entities=(*source_group, *target_group),
                    )
                source = _seam_parent_triangles(
                    coordinates,
                    point_array,
                    source_faces,
                    offsets,
                    rows,
                    matrix,
                    budget=budget,
                )
                target = _seam_parent_triangles(
                    coordinates,
                    point_array,
                    target_faces,
                    offsets,
                    rows,
                    None,
                    budget=budget,
                )
                group = _seam_group_fragments(
                    source, target, constraint.constraint_id, budget=budget
                )
                budget.reserve(len(group), 256 + 16 * len(group))
                fragments.extend(group)
        budget.reserve(0, 256 + 8 * len(fragments))
        result = tuple(fragments)
        budget.retain_basis((result,))
    return result


@dataclass(frozen=True, slots=True)
class _PeriodicSeamBank:
    points: tuple[tuple[Fraction, ...], ...]
    parents: np.ndarray
    coefficients: tuple[tuple[Fraction, ...], ...]
    actions: np.ndarray
    carriers: np.ndarray
    face_parts: dict[int, tuple[tuple[int, ...], ...]]
    paired_loops: tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]
    vertex_pairs: np.ndarray


def _periodic_seam_bank(
    source: _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource,
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    fragments: tuple[PeriodicPolyhedralSeamFragment, ...],
    constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    maximum_vertices: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PeriodicSeamBank:
    """Assemble seam incidence in the caller's admitted ``bank_bound``.

    _reconcile_periodic_polyhedral owns every Python loop/lookup bank here
    through the expanded-cell consumer. Native array payload owns its pool
    capacity independently; exact point/coefficient work uses ``budget``.
    """
    preparation = diagram.periodic_preparation
    if preparation is None:
        raise ValueError(
            "polyhedral.diagram.periodic_preparation must be retained for seam assembly; got None"
        )
    prepared = source.prepare(budget=budget)
    vertex_count = len(prepared.vertices)
    carrier_rows = diagram.construction.vertex_carriers
    # Per prepared vertex: the list copy, parent/coefficient/action rows, the
    # point lookup insertion and hash over three rational coordinates, plus
    # two visits per carrier column; one visit per constraint index.
    budget.reserve(
        vertex_count * (8 + 2 * int(carrier_rows.shape[1])) + len(constraints),
        512 + 256 * vertex_count,
    )
    points: list[tuple[Fraction, ...]] = list(prepared.vertices)
    rank = len(preparation.orders)
    parents = [(index, -1, -1) for index in range(len(points))]
    coefficients: list[tuple[Fraction, ...]] = [
        (Fraction(1), Fraction(0), Fraction(0))
    ] * len(points)
    actions = [(0,) * rank] * len(points)
    carriers = []
    for vertex_index, weights in enumerate(prepared.barycentric):
        indices = tuple(
            int(carrier_rows[vertex_index, column])
            for column in range(carrier_rows.shape[1])
            if carrier_rows[vertex_index, column] >= 0
        )
        carriers.append(
            tuple(
                index
                for index, weight in zip(indices, weights, strict=True)
                if weight > 0
            )
        )
    lookup = {}
    for index, point in enumerate(points):
        lookup.setdefault(point, index)
    constraint_indices = {
        value.constraint_id: index for index, value in enumerate(constraints)
    }
    parts: dict[int, list[tuple[int, ...]]] = {}
    paired_loops = []
    vertex_pairs = []

    def vertex(
        point: tuple[Fraction, ...],
        row: tuple[int, int, int],
        weights: tuple[Fraction, ...],
        exponents: tuple[int, ...],
        physical_row: tuple[int, int, int],
        physical_weights: tuple[Fraction, ...],
    ) -> int:
        if point in lookup:
            return lookup[point]
        if len(points) >= maximum_vertices:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_VERTEX_BUDGET,
                achieved=len(points) + 1,
            )
        budget.reserve(1, 512)
        index = len(points)
        points.append(point)
        parents.append(row)
        coefficients.append(weights)
        actions.append(exponents)
        support = {
            carrier
            for parent, weight in zip(physical_row, physical_weights, strict=True)
            if weight
            for carrier in carriers[parent]
        }
        carriers.append(tuple(sorted(support)))
        lookup[point] = index
        budget.retain_basis((point, row, weights, exponents, carriers[-1]))
        return index

    for fragment in fragments:
        generator = constraint_indices[fragment.constraint_id]
        exponents = tuple(int(axis == generator) for axis in range(rank))
        if array_tree_fingerprint(
            preparation.generators[generator]
        ) != array_tree_fingerprint(np.asarray(constraints[generator].transform)):
            raise ValueError(
                "polyhedral.seam.transform must match the retained periodic generator; "
                f"got constraint index={generator}"
            )
        left, right = [], []
        for point, source_weights, target_weights in zip(
            fragment.vertices,
            fragment.source_coefficients,
            fragment.target_coefficients,
            strict=True,
        ):
            triangle = tuple(
                prepared.vertices[index] for index in fragment.source_triangle
            )
            planar_primitives_owner.reserve_fraction_work(
                triangle, 18, 3, 6, budget=budget
            )
            original = tuple(
                sum(
                    (
                        weight * parent[axis]
                        for parent, weight in zip(triangle, source_weights, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            source_vertex = vertex(
                original,
                fragment.source_triangle,
                source_weights,
                (0,) * rank,
                fragment.source_triangle,
                source_weights,
            )
            target_vertex = vertex(
                point,
                fragment.source_triangle,
                source_weights,
                exponents,
                fragment.target_triangle,
                target_weights,
            )
            left.append(source_vertex)
            right.append(target_vertex)
            budget.reserve(1, 128)
            vertex_pairs.append((source_vertex, target_vertex, generator))
        source_loop = tuple(left if fragment.relative_orientation > 0 else reversed(left))
        target_loop = tuple(right)
        parts.setdefault(fragment.source_face, []).append(source_loop)
        parts.setdefault(fragment.target_face, []).append(target_loop)
        paired_loops.append((source_loop, target_loop, generator))
    count = len(points)
    parent_bank = _dependency_phydrax_meshing__host_banks.family_host_array(
        (count, 3), np.int64, scope=scope
    )
    action_bank = _dependency_phydrax_meshing__host_banks.family_host_array(
        (count, rank), np.int64, scope=scope
    )
    carrier_bank = _dependency_phydrax_meshing__host_banks.family_host_array(
        (count, max(4, max(map(len, carriers), default=0))), np.int32, scope=scope
    )
    carrier_bank.fill(-1)
    for index in range(count):
        parent_bank[index] = parents[index]
        action_bank[index] = actions[index]
        carrier_bank[index, : len(carriers[index])] = carriers[index]
    pairs = _dependency_phydrax_meshing__host_banks.family_host_array(
        (len(vertex_pairs), 3), np.int64, scope=scope
    )
    for index, pair in enumerate(vertex_pairs):
        pairs[index] = pair
    return _PeriodicSeamBank(
        tuple(points),
        parent_bank,
        tuple(coefficients),
        action_bank,
        carrier_bank,
        {face: tuple(loops) for face, loops in parts.items()},
        tuple(paired_loops),
        pairs,
    )


def _seam_segment_subdivision(
    points: tuple[tuple[Fraction, ...], ...],
    loops: Sequence[tuple[int, ...]],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Subdivide in the caller's admitted edge-by-point structural capacity.

    The private caller retains that capacity through both expanded-cell
    construction and final paired-face lookup, not merely this function exit.
    """
    # Two visits per loop edge occurrence (pair sort and set insertion), then
    # the sort of at most that many unique seam edges.
    occurrences = sum(len(loop) for loop in loops)
    budget.reserve(occurrences * (2 + max(1, (occurrences + 1).bit_length())))
    edges = {
        tuple(sorted((first, second)))
        for loop in loops
        for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
    }
    result = {}
    for first, second in sorted(edges):
        start, stop = points[first], points[second]
        planar_primitives_owner.reserve_fraction_work(
            (start, stop), 3, 3, 2, budget=budget
        )
        direction = tuple(b - a for a, b in zip(start, stop, strict=True))
        pivot = next((axis for axis, value in enumerate(direction) if value), None)
        if pivot is None:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_SEAM_DEGENERATE_EDGE,
                entities=(first, second),
            )
        interior = []
        for index, point in enumerate(points):
            if index in (first, second):
                continue
            with budget.temporary_scope():
                planar_primitives_owner.reserve_fraction_work(
                    (point, start), 1, 1, 2, budget=budget
                )
                solve = _owner_phydrax_linalg__small_batched.prepare_exact_small_linear_actions(
                    ((direction[pivot],),),
                    ((point[pivot] - start[pivot],),),
                    budget=budget,
                )
                if solve.actions is None:
                    raise PolyhedralGenerationError(
                        PolyhedralGenerationReason.PERIODIC_SEAM_EDGE_RANK
                    )
                parameter = solve.actions[0][0]
                planar_primitives_owner.reserve_fraction_work(
                    (start, direction, point), 9, 3, 4, budget=budget
                )
                inside = 0 < parameter < 1 and all(
                    start[axis] + parameter * direction[axis] == point[axis]
                    for axis in range(3)
                )
            if inside:
                budget.retain_basis((parameter, index))
                budget.reserve(1, 128)
                interior.append((parameter, index))
        budget.reserve(
            len(interior) * max(1, len(interior).bit_length()), 256 + 16 * len(interior)
        )
        result[(first, second)] = (
            first,
            *(index for _, index in sorted(interior)),
            second,
        )
    return result


def _expanded_seam_loop(
    loop: tuple[int, ...],
    edges: dict[tuple[int, int], tuple[int, ...]],
    /,
) -> tuple[int, ...]:
    """Expand within the reconciliation owner's admitted occurrence bank."""
    result = []
    for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
        edge = edges[(min(first, second), max(first, second))]
        result.extend(edge[:-1] if first < second else edge[:0:-1])
    return tuple(result)


@checked
def seam_vertex_orbits(
    points: tuple[tuple[Fraction, ...], ...],
    pairs: np.ndarray,
    identifiers: np.ndarray,
    preparation: periodic_power_owner.PeriodicPowerPreparation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertex orbit representatives and group-action exponents."""
    with NativeHostStorageWorkspace(scope) as orbit_storage:
        # Parent indices, winding tuples and three simultaneous root/action tuples.
        # Each int is <=144 bytes plus a 16-byte sequence slot; headers <=256.
        orbit_storage.set_bound(
            1024 + (len(points) + 6) * (256 + 160 * (1 + len(preparation.orders)))
        )
        # Binary64 numerators/denominators have <=1075 bits, including
        # subnormals. Admit 16 Fraction owners per homogeneous generator,
        # each with two declared integer owners, before rational_vector.
        integer_bytes = 128 + 4 * ((1075 + 7) // 8)
        fraction_bytes = 256 + 2 * integer_bytes
        rank = len(preparation.orders)
        budget.reserve(
            16 * rank,
            256 + rank * (256 + 16 * 4 + 4 * (256 + 16 * 4 + 4 * fraction_bytes)),
        )
        matrices = tuple(
            tuple(rational_vector(row) for row in matrix) for matrix in preparation.generators
        )
        orders = preparation.orders
        rank = len(orders)
        # Parent/winding initialization, then per pair the root calls'
        # root tests, difference normalization and union, and per vertex the
        # final root call's test plus the winding range check and row writes.
        budget.reserve(
            len(points) * (1 + rank)
            + len(pairs) * (4 + 4 * rank)
            + len(points) * (3 + 2 * rank)
        )
        parent = list(range(len(points)))
        winding = [(0,) * rank for _ in points]
    
        def normalized(values: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(
                value % order if order else value
                for value, order in zip(values, orders, strict=True)
            )
    
        def root(vertex: int) -> tuple[int, tuple[int, ...]]:
            total = (0,) * rank
            while parent[vertex] != vertex:
                budget.reserve(rank)
                total = normalized(
                    tuple(a + b for a, b in zip(total, winding[vertex], strict=True))
                )
                vertex = parent[vertex]
            return vertex, total
    
        for source, target, generator in pairs:
            source_root, source_shift = root(int(source))
            target_root, target_shift = root(int(target))
            difference = normalized(
                tuple(
                    int(axis == generator) + a - b
                    for axis, (a, b) in enumerate(
                        zip(source_shift, target_shift, strict=True)
                    )
                )
            )
            if source_root == target_root:
                matrix = periodic_exact_owner.exact_periodic_element(
                    matrices, orders, difference, budget=budget
                )
                point = points[source_root]
                planar_primitives_owner.reserve_fraction_work(
                    (point, *matrix), 36, 3, 8, budget=budget
                )
                image = tuple(
                    sum(matrix[axis][column] * point[column] for column in range(3))
                    + matrix[axis][3]
                    for axis in range(3)
                )
                if image != point:
                    raise PolyhedralGenerationError(
                        PolyhedralGenerationReason.PERIODIC_SEAM_ORBIT_CYCLE,
                        entities=(int(source), int(target)),
                    )
            elif identifiers[source_root] < identifiers[target_root]:
                parent[target_root], winding[target_root] = source_root, difference
            else:
                parent[source_root] = target_root
                winding[source_root] = normalized(tuple(-value for value in difference))
        representatives = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(points),), np.int64, scope=scope
        )
        shifts = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(points), rank), np.int64, scope=scope
        )
        for index in range(len(points)):
            representative, values = root(index)
            if any(
                value < np.iinfo(np.int32).min or value > np.iinfo(np.int32).max
                for value in values
            ):
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.PERIODIC_SEAM_WINDING_BUDGET, entities=(index,)
                )
            representatives[index], shifts[index] = representative, values
        parent.clear()
        winding.clear()
        del matrices
        return representatives, shifts


@dataclass(frozen=True, slots=True)
class _PeriodicPolyhedralAssembly:
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    parent_faces: np.ndarray
    vertex_carriers: np.ndarray
    face_pairs: np.ndarray
    vertex_pairs: np.ndarray


def _reconcile_periodic_polyhedral(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    fragments: tuple[PeriodicPolyhedralSeamFragment, ...],
    constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    group: PeriodicIsometryGroup,
    maximum_vertices: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None = None,
) -> _PeriodicPolyhedralAssembly:
    with _PolyhedralTopologyStorage(scope) as storage:
        source = geometry.exact_source
        match source:
            case _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource():
                if diagram.periodic_preparation is None:
                    raise ValueError(
                        "polyhedral.geometry.exact_source must retain a power source and periodic preparation; "
                        "got preparation=None"
                    )
            case invalid:
                raise ValueError(
                    "polyhedral.geometry.exact_source must retain a power source and periodic preparation; "
                    f"got source={type(invalid).__name__}"
                )
        old = require_polyhedral_connectivity(mesh)
        fragment_corners = sum(len(fragment.vertices) for fragment in fragments)
        possible_vertices = old.vertex_count + 2 * fragment_corners
        rank = len(diagram.periodic_preparation.orders)
        carrier_width = diagram.construction.vertex_carriers.shape[1]
        # _periodic_seam_bank creates at most two vertices per fragment corner.
        # Per vertex: point/coefficient references, parent and action tuples,
        # carrier support (a union of three original carrier rows), lookup slot.
        # Per fragment: two loops, paired-loop record and face-parts slots;
        # per corner: pair tuple and the two live loop-list slots. Fraction
        # payload/arithmetic remains with CoordinateEnclosureBudget.
        bank_bound = (
            256 * (16 + 12 * possible_vertices + 8 * len(fragments))
            + 160 * (
                (11 + rank + 3 * carrier_width) * possible_vertices
                + 6 * fragment_corners
            )
            + 64 * (possible_vertices + len(constraints) + 2 * len(fragments))
            + 256 * (4 * old.face_count + 4 * len(constraints))
            + 160 * len(old.face_vertex_values)
        )
        storage.workspace.set_bound(bank_bound)
        bank = _periodic_seam_bank(
            source,
            diagram,
            fragments,
            constraints,
            maximum_vertices,
            scope=scope,
            budget=budget,
        )
        offsets, vertices = (
            np.asarray(old.face_vertex_offsets),
            np.asarray(old.face_vertex_values),
        )
        # Face-part lookup over every face and its packed corners.
        budget.reserve(old.face_count + len(vertices))
        parts = {
            face: bank.face_parts.get(face, (tuple(map(int, vertices[a:b])),))
            for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
        }
        # all_loops flattening and the corner count visit each loop corner.
        budget.reserve(2 * sum(len(loop) for loops in parts.values() for loop in loops))
        all_loops = tuple(loop for loops in parts.values() for loop in loops)
        loop_corners = sum(len(loop) for loop in all_loops)
        # _seam_segment_subdivision has <=loop_corners distinct edges, each
        # containing <=len(points) vertices. One interior list and its sorted
        # copy coexist. _expanded_seam_loop and origins each retain at most
        # loop_corners*(len(points)-1) slots, plus current sorted key copies.
        expanded_corners_upper = loop_corners * max(1, len(bank.points) - 1)
        # This is temporary subdivision capacity, not retained connectivity.
        # Admit the realized expanded incidence below, before output banks.
        subdivision_bound = (
            256 * (16 + 4 * loop_corners + 4 * len(all_loops) + 4 * old.face_count)
            + 160 * (4 * loop_corners + loop_corners * len(bank.points))
            + (256 + 2 * 160) * len(bank.points)
            + 16 * 6 * expanded_corners_upper
        )
        storage.workspace.set_bound(bank_bound + subdivision_bound)
        subdivision = _seam_segment_subdivision(bank.points, all_loops, budget=budget)
        # Realized expanded corners: one subdivision lookup per loop corner.
        budget.reserve(loop_corners)
        # _expanded_seam_loop emits len(edge) - 1 corners per occurrence.
        expanded_corners = sum(
            len(subdivision[(min(first, second), max(first, second))]) - 1
            for loop in all_loops
            for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
        )
        # Expansion emits each realized corner once, origins sort each expanded
        # loop, and the cell-face count visits each face's loops (each face
        # bounds at most two cells).
        budget.reserve(
            expanded_corners * (2 + max(1, (expanded_corners + 1).bit_length()))
            + len(old.cell_face_values)
            + 2 * len(all_loops)
        )
        expanded = {
            face: tuple(_expanded_seam_loop(loop, subdivision) for loop in loops)
            for face, loops in parts.items()
        }
        origins = {
            tuple(sorted(loop)): face for face, loops in expanded.items() for loop in loops
        }
        cell_offsets, cell_faces = (
            np.asarray(old.cell_face_offsets),
            np.asarray(old.cell_face_values),
        )
        signs = np.asarray(old.cell_face_sign_values)
        loop_count = corner_count = 0
        for face in cell_faces:
            for loop in expanded[int(face)]:
                loop_count += 1
                corner_count += len(loop)
        if limits is not None:
            _admit_polyhedral_mesh_capacity(
                len(bank.points), old.cell_count, loop_count, corner_count, limits
            )
        budget.reserve(
            _polyhedral_mesh_work_bound(
                len(bank.points), old.cell_count, loop_count, corner_count
            )
        )
        mesh_bound, retained_mesh_bound = _polyhedral_mesh_storage_bound(
            len(bank.points), old.cell_count, loop_count, corner_count
        )
        # cells owns one int64 copy per expanded occurrence; the canonical
        # builder additionally adapts int32 loops. Both coexist until return.
        cell_bank_bound = 256 * (old.cell_count + loop_count + 1) + 8 * corner_count
        storage.workspace.set_bound(
            bank_bound + subdivision_bound + cell_bank_bound + mesh_bound
        )
        cells = []
        for a, b in zip(cell_offsets[:-1], cell_offsets[1:], strict=True):
            cells.append(
                [
                    np.asarray(loop if sign > 0 else loop[::-1], dtype=np.int64)
                    for face, sign in zip(cell_faces[a:b], signs[a:b], strict=True)
                    for loop in expanded[int(face)]
                ]
            )
        old_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        identifiers = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(bank.points),), np.int64, scope=scope
        )
        identifiers[: len(old_ids)] = old_ids
        first = int(np.max(old_ids)) + 1
        if first + len(bank.points) - len(old_ids) > np.iinfo(np.int64).max:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_VERTEX_IDENTITY_BUDGET
            )
        for index in range(len(old_ids), len(bank.points)):
            identifiers[index] = first + index - len(old_ids)
        linear = _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource(
            source,
            bank.parents,
            bank.coefficients,
            bank.actions,
            periodic_preparation=diagram.periodic_preparation,
        )
        prepared = linear.prepare(budget=budget)
        if prepared.vertices != bank.points:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.PERIODIC_SEAM_SOURCE_BINDING
            )
        lifted = _dependency_phydrax_discretization__cell_mesh.CellMesh.from_polyhedra(
            prepared.rounded_vertices,
            cells,
            block_name="restricted-power",
            numeric_version=mesh.numeric_version,
            vertex_global_ids=identifiers,
            cell_global_ids=np.asarray(old.cell_global_ids),
        )
        lifted_geometry = (
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
                lifted, linear, budget=budget
            )
        )
        representatives, shifts = seam_vertex_orbits(
            bank.points,
            bank.vertex_pairs,
            identifiers,
            diagram.periodic_preparation,
            scope=scope,
            budget=budget,
        )
        lifted_connectivity = require_polyhedral_connectivity(lifted)
        face_offsets = np.asarray(lifted_connectivity.face_vertex_offsets)
        squared_arities = 4 * lifted_connectivity.edge_count
        for a, b in zip(face_offsets[:-1], face_offsets[1:], strict=True):
            squared_arities += int(b - a) ** 2
        for block in lifted.blocks:
            squared_arities += block.cell_count * block.vertices.shape[1] ** 2
        topology_bound, retained_topology_bound = _periodic_topology_storage_bound(
            len(bank.points), old.cell_count, loop_count, corner_count,
            squared_arities, rank,
        )
        # The bound mesh constructor revalidates/rebuilds its three lifted
        # incidences while lifted and quotient topology remain live.
        storage.workspace.set_bound(
            bank_bound + subdivision_bound + cell_bank_bound
            + 2 * mesh_bound + topology_bound
        )
        topology = _dependency_phydrax_discretization__periodic_topology.PeriodicMeshTopology(
            lifted,
            group,
            representatives,
            shifts,
            actual_geometry=lifted_geometry,
            budget=budget,
        )
        bound = _dependency_phydrax_discretization__cell_mesh.CellMesh(
            lifted.coordinates,
            lifted.blocks,
            vertex_global_ids=lifted.vertex_global_ids,
            polyhedral_connectivity=require_polyhedral_connectivity(lifted),
            periodic_topology=topology,
            numeric_version=lifted.numeric_version,
            budget=budget,
        )
        bound_geometry = (
            _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
                bound, linear, budget=budget
            )
        )
        connectivity = require_polyhedral_connectivity(bound)
        new_offsets, new_vertices = (
            np.asarray(connectivity.face_vertex_offsets),
            np.asarray(connectivity.face_vertex_values),
        )
        lookup = {
            tuple(sorted(map(int, new_vertices[a:b]))): face
            for face, (a, b) in enumerate(zip(new_offsets[:-1], new_offsets[1:], strict=True))
        }
        parent_faces = _dependency_phydrax_meshing__host_banks.family_host_array(
            (connectivity.face_count,), np.int64, scope=scope
        )
        for key, face in lookup.items():
            parent_faces[face] = origins[key]
        face_pairs = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(bank.paired_loops), 3), np.int64, scope=scope
        )
        for index, (left, right, generator) in enumerate(bank.paired_loops):
            face_pairs[index] = (
                lookup[tuple(sorted(_expanded_seam_loop(left, subdivision)))],
                lookup[tuple(sorted(_expanded_seam_loop(right, subdivision)))],
                generator,
            )
        result = _PeriodicPolyhedralAssembly(
            bound, bound_geometry, parent_faces, bank.carriers, face_pairs, bank.vertex_pairs
        )
        record_count_upper = (
            512 + 64 * old.cell_count
            + 16 * (len(bank.points) + corner_count + loop_count)
        )
        # Release topology assembly banks before replacing their construction
        # reservation by the retained mesh/quotient reservation.
        del cells, parts, all_loops, subdivision, expanded, origins, lookup
        del cell_offsets, cell_faces, signs, offsets, vertices, face_offsets
        del new_offsets, new_vertices, loop, key, bank, lifted, lifted_geometry
        storage.retain(
            2 * retained_mesh_bound + retained_topology_bound,
            bound, topology,
            record_count_upper=record_count_upper,
        )
        return result


def _freeze(value: np.ndarray, /, *, scope: NativeExecutionScope) -> np.ndarray:
    result = _dependency_phydrax_meshing__host_banks.family_host_array(
        value.shape, value.dtype, scope=scope
    )
    scope.charge(work=1 + value.size)
    np.copyto(result, value)
    result.setflags(write=False)
    return result


def _geometry_identity_work(
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec, /
) -> int:
    """Host work of ``cell_geometry_id``: streamed reads and hashes of its banks."""
    if geometry.storage is not None:
        # Owner-local storage publishes its checked logical identity.
        return 256
    leaves = jax.tree_util.tree_leaves(
        (geometry.elements, geometry.geometry_dofs, geometry.coordinates)
    )
    return 2 * sum(int(np.size(leaf)) for leaf in leaves) + 256 * (
        1 + len(geometry.elements)
    )


@checked
def require_polyhedral_connectivity(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh, /
) -> PolyhedralConnectivity:
    """Return the mesh's packed polyhedral connectivity."""
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            return connectivity
        case invalid:
            raise TypeError(
                "polyhedral.mesh.connectivity must be PolyhedralConnectivity; "
                f"got {type(invalid).__name__}"
            )


def _tet_facets(
    recovery: PlcRecovery3D, /, *, scope: NativeExecutionScope
) -> np.ndarray:
    # Facet ids are semantic groups and may span noncoplanar polygons. Merge
    # constrained fragments only on the same planar recovered subtriangle.
    # Each dictionary entry owns one three-integer tuple and one row integer.
    # The declared structural convention is 256 + 16*n per tuple, 64 per
    # mapping slot, and <=144 bytes per signed int32 integer. The extra two
    # entries cover the current sorted-list/tuple construction and lookup key.
    face_entry = 64 + 256 + 3 * 16 + 4 * 144
    with NativeHostStorageWorkspace(scope) as scratch:
        scratch.set_bound(256 + (len(recovery.faces) + 2) * face_entry)
        faces: dict[tuple[int, ...], int] = {}
        for index, face in enumerate(recovery.faces):
            # One actual key visit, before sorting or allocating that key.
            scope.charge(work=1)
            faces[tuple(sorted(map(int, face)))] = index
        result = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(recovery.tetrahedra), 4), np.int32, scope=scope
        )
        for row, tet in enumerate(recovery.tetrahedra):
            for opposite in range(4):
                scope.charge(work=1)
                key = tuple(sorted(int(tet[v]) for v in range(4) if v != opposite))
                result[row, opposite] = faces.get(key, -1)
        return result


class _PolyhedralTopologyStorage:
    """Pool-bound host topology storage, including logical JAX output banks.

    This is declared host capacity, not device, compiler, or process RSS
    measurement. Native-backed input/output payloads have their own allocation
    lifetime and are deliberately absent from these bounds.
    """

    def __init__(self, scope: NativeExecutionScope, /) -> None:
        self._scope = scope
        self.workspace = NativeHostStorageWorkspace(scope)
        self._remaining = 0

    def __enter__(self) -> _PolyhedralTopologyStorage:
        self.workspace.open()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._remaining:
            self.workspace.close()

    def _release(self) -> None:
        self._remaining -= 1
        if not self._remaining:
            self.workspace.close()

    def retain(
        self, bytes_upper: int, *owners: object, record_count_upper: int
    ) -> None:
        # Only explicit roots and array leaves keep the token alive. The shared
        # profiler visits/deduplicates the full graph once; its owner map is
        # temporary, never captured by a finalizer.
        metadata = 1024 + 512 * record_count_upper
        initial = max(self.workspace.bound, bytes_upper + metadata)
        selected: list[object] = []
        finalizers: list[weakref.finalize] = []
        owner_indices = pending_slots = largest_fanout = 0
        pending_row = declared_object_storage_bytes((None, False, False, False)) + 16

        def admit_owner(value: object) -> None:
            nonlocal owner_indices, pending_slots, largest_fanout
            self._scope.charge(work=1)
            keep = eqx.is_array(value) or id(value) in roots
            if keep and len(selected) >= record_count_upper:
                raise RuntimeError("polyhedral.retention_records must cover the output tree")
            children = retained_host_storage_fanout_upper(value)
            owner_indices += 128 + declared_object_storage_bytes(id(value))
            pending_slots += children + int(children != 0)
            largest_fanout = max(largest_fanout, children)
            # Admit every duplicate edge/leaving row before profile expansion,
            # plus the identity indices and current child-tuple staging bank.
            traversal = (
                1024 + owner_indices + pending_row * pending_slots + 512 * largest_fanout
            )
            self.workspace.set_bound(
                max(initial, bytes_upper + traversal + 512 * (len(selected) + int(keep)))
            )
            if keep:
                selected.append(value)

        try:
            self.workspace.set_bound(initial)
            if len(owners) > record_count_upper:
                raise RuntimeError("polyhedral.retention_records must cover the output tree")
            roots = {id(value) for value in owners}
            retained_host_storage_profile(
                owners, {}, _owner_phydrax__meshcore.is_native_host_buffer,
                admit_owner=admit_owner,
            )
            # Selected owners stay strong until all attachments and the final
            # reservation succeed. A refusal detaches every partial attachment.
            for value in selected:
                finalizers.append(weakref.finalize(value, self._release))
                self._remaining += 1
            self.workspace.set_bound(bytes_upper + metadata)
        except BaseException:
            for finalizer in finalizers:
                finalizer.detach()
            self._remaining = 0
            self.workspace.close()
            raise


def _polyhedral_mesh_storage_bound(
    vertices: int, cells: int, loops: int, corners: int, /
) -> tuple[int, int]:
    """Bound the explicit-loop CellMesh builder from concrete incidences.

    ``loops`` counts face *occurrences*, ``corners`` their vertex occurrences.
    Distinct faces/edges and cell vertices are all bounded by these counts.
    The following envelopes count eight-byte scalar slots (also covering bool,
    int32 and float64) and 256-byte array/container headers, not native payload.

    Owners: CellMesh.from_polyhedra's normalization, arity grouping and sets;
    _cell_complex.polyhedral_connectivity's canonicalization/defect/shell
    scans, row grouping and identity sorts; polyhedral_cell_complex's three
    incidence relations and CellComplexTopology's two sparse chain products.
    A chain product has at most 2*corners vertex-face paths or corners
    edge-cell paths; no dense vertex-by-cell matrix is assumed.
    """
    # Retained connectivity, blocks, entity sets, signed relations, coordinates
    # and identity banks: <=64 slots per incidence, <=16 per vertex/cell.
    # At most one block per cell; each owns <=32 structural/header records.
    retained = (
        256 * (128 + 32 * cells)
        + 8 * (64 * (corners + loops) + 16 * (vertices + cells + 1))
    )
    # Simultaneously live local banks in the canonical builder and its nested
    # sort/validation helpers: 32 corner, 24 loop, 12 cell, 4 vertex slots;
    # identity/grouping sorts add 24 corner, 16 loop, 8 cell/vertex slots.
    canonical = 8 * (
        56 * corners + 40 * loops + 20 * cells + 12 * vertices + 64
    )
    # Explicit-loop adaptation: converted int32 loops, NumPy view headers,
    # normalized cell tuples, and the per-cell distinct-vertex set. The latter
    # has <=corners entries, each a mapping/set slot plus an int64 Python int.
    adaptation = (
        256 * (16 + 8 * cells + 4 * loops)
        + (64 + 144 + 16) * corners
        + 8 * (16 * cells + 8 * loops + 4 * corners)
    )
    # Relation validation/CSR conversions and sparse chain multiplication:
    # 3*corners+loops relation entries, <=3*corners product paths, row pointers
    # and sorting scratch. Both products are bounded even before cancellation.
    relations = 8 * (
        32 * (3 * corners + loops) + 16 * (vertices + cells + loops + 1)
    )
    retention_metadata = 1024 + 512 * (512 + 64 * cells)
    return retained + canonical + adaptation + relations + retention_metadata, retained


def _polyhedral_numeric_publication_bounds(
    vertices: int, cells: int, loops: int, corners: int, /
) -> tuple[int, int]:
    """Canonical topology connectivity slots and retained numeric-bank bytes."""
    slots = 64 * (corners + loops) + 16 * (vertices + cells + 1)
    return slots, 8 * slots


def _validate_polyhedral_preflight(
    data: _owner_phydrax__meshcore.RestrictedPowerCells,
    loops: int,
    corners: int,
    /,
) -> None:
    """Bind native counts to exported banks and the Python publication owner.

    The caller already meters its sided-incidence scan. This constant-size
    comparison neither repeats native clipping work nor reserves its payload:
    native export and diagram storage retain those allocations independently.
    """
    header = data.preflight
    slots, numeric_bytes = _polyhedral_numeric_publication_bounds(
        len(data.vertices), len(data.cell_sites), loops, corners
    )
    counts = (
        ("vertices", len(data.vertices)),
        ("faces", len(data.face_cells)),
        ("face_corners", data.face_vertices.size),
        ("cells", len(data.cell_sites)),
        ("pieces", len(data.piece_sites)),
        ("vertex_site_witnesses", data.vertex_sites.size),
        ("native_output_payload_bytes", sum(
            value.nbytes for value in vars(data).values() if isinstance(value, np.ndarray)
        )),
        ("sided_loop_occurrences", loops),
        ("sided_corner_occurrences", corners),
        ("publication_connectivity_entries_upper", slots),
        ("publication_numeric_data_bytes_upper", numeric_bytes),
    )
    for name, expected in counts:
        actual = getattr(header, name)
        if actual != expected:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
                f"polyhedral.preflight.{name} must match its Python owner; "
                f"got {actual} != {expected}",
                stage="polyhedral_candidate",
                achieved=((f"native_{name}", actual), (f"owner_{name}", expected)),
            )


def _admit_polyhedral_mesh_capacity(
    vertices: int,
    cells: int,
    loops: int,
    corners: int,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
) -> None:
    """Admit the topology builder's output banks before constructing them.

    Face/edge occurrences bound distinct entities. The numeric-bank envelope
    is the retained portion of ``_polyhedral_mesh_storage_bound`` without its
    Python/header metadata. It is not a process-memory or whole-result claim.
    """
    slots, data_bytes = _polyhedral_numeric_publication_bounds(
        vertices, cells, loops, corners
    )
    checks = (
        ("vertices", vertices, limits.maximum_vertices),
        ("edges", corners, limits.maximum_edges),
        ("faces", loops, limits.maximum_faces),
        ("cells", cells, limits.maximum_cells),
        ("connectivity_entries", slots, limits.maximum_connectivity_entries),
        ("data_bytes", data_bytes, limits.maximum_data_bytes),
    )
    for name, required, maximum in checks:
        if required > maximum:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"polyhedral.candidate.{name} must admit its output bank upper; "
                f"got {required} > {maximum}",
                stage="polyhedral_candidate",
                requested=((f"maximum_{name}", maximum),),
                achieved=(
                    (f"required_{name}", required),
                    (f"remaining_{name}", maximum),
                ),
            )


def _polyhedral_mesh_work_bound(
    vertices: int, cells: int, loops: int, corners: int, /
) -> int:
    """Host scalar visits in canonicalization, identity sorts and incidence.

    Explicit-loop normalization visits corners and loops. Canonical row/key
    sorting visits at most the binary merge depth per scalar; signed incidence
    and sparse chain-product paths are bounded by three corner occurrences.
    This excludes native construction and all coordinate/source predicates.
    """
    rows = vertices + cells + loops + corners
    return 12 * rows + 8 * (corners + loops) * max(1, rows.bit_length())


def _periodic_topology_storage_bound(
    vertices: int,
    cells: int,
    loops: int,
    corners: int,
    squared_arities: int,
    rank: int,
    /,
) -> tuple[int, int]:
    """Bound _periodic_topology's concrete entity-key/quotient callgraph.

    Each oriented n-loop has 2*n variants of n*(rank+1) key scalars;
    each n-corner cell has n variants. Distinct-corner validation has n*n*rank
    scalars. ``squared_arities`` sums n*n over actual face and cell loops plus
    four per edge. No schedule maximum or maximum-arity Cartesian padding is
    used. Twenty-four variant-sized banks cover anchored variants, reduction,
    comparisons, lexical sorts and their operands, including sort workspace.
    """
    entities = vertices + corners + loops + cells
    key_slots = (vertices + 3 * corners) * (rank + 1)
    retained = (
        256 * (128 + 16 * entities)
        + 8 * (key_slots + (4 + rank) * entities + rank * corners + 32)
    )
    variants = 8 * 24 * 2 * squared_arities * (rank + 1)
    # _Degree keys/lookup lists, quotient incidence grouping, chain products,
    # boundary masks, and concatenation retain no more than lifted incidence.
    assembly = (
        256 * (128 + 8 * entities)
        + 8 * (64 * (3 * corners + loops + entities) + 8 * key_slots)
    )
    retention_metadata = 1024 + 512 * (512 + 64 * cells + 16 * entities)
    return retained + variants + assembly + retention_metadata, retained


def _publish(
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    recovery: PlcRecovery3D,
    revision: str,
    maximum_work: int,
    *,
    scope: NativeExecutionScope,
    domain_source: plc_source_owner.PiecewiseLinearComplex,
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None = None,
) -> tuple[
    _dependency_phydrax_discretization__cell_mesh.CellMesh,
    _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    np.ndarray,
    np.ndarray,
]:
    with _PolyhedralTopologyStorage(scope) as storage:
        data = diagram.construction
        # Count occurrences without constructing a boolean/index array first.
        loop_count = corner_count = 0
        budget.reserve(len(data.face_cells))
        for face in range(len(data.face_cells)):
            right = data.face_cells[face, 1]
            copies = 1 + int(right >= 0)
            loop_count += copies
            corner_count += copies * int(data.face_offsets[face + 1] - data.face_offsets[face])
        _validate_polyhedral_preflight(data, loop_count, corner_count)
        if limits is not None:
            _admit_polyhedral_mesh_capacity(
                len(data.vertex_carriers), len(data.cell_sites),
                loop_count, corner_count, limits,
            )
        source = _dependency_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource(
            diagram.points,
            diagram.weights,
            recovery.points,
            recovery.tetrahedra,
            data.vertex_site_offsets,
            data.vertex_sites,
            data.vertex_carriers,
            maximum_work=maximum_work,
            periodic_preparation=diagram.periodic_preparation,
            domain_source=domain_source,
            periodic_constraints=periodic_constraints,
        )
        exact = source.prepare(budget=budget)
        construction_bound, retained_bound = _polyhedral_mesh_storage_bound(
            len(exact.rounded_vertices), len(data.cell_sites), loop_count, corner_count
        )
        # Native-face dictionary, sorted lookup keys, and current key/sort lists.
        lookup_bound = (
            256 * (4 + len(data.face_cells))
            + (64 + 2 * 144) * len(data.face_cells)
            + (2 * 16 + 144) * (len(data.face_vertices) + corner_count)
        )
        storage.workspace.set_bound(construction_bound + lookup_bound)
        budget.reserve(
            _polyhedral_mesh_work_bound(
                len(exact.rounded_vertices), len(data.cell_sites),
                loop_count, corner_count,
            )
            + len(data.face_cells) + corner_count
        )
        cells: list[list[np.ndarray]] = [[] for _ in data.cell_sites]
        for face, (left, right) in enumerate(
            zip(data.face_cells[:, 0], data.face_cells[:, 1], strict=True)
        ):
            start, stop = data.face_offsets[face : face + 2]
            loop = data.face_vertices[start:stop]
            cells[int(left)].append(loop)
            if right >= 0:
                cells[int(right)].append(loop[::-1])
        mesh = _dependency_phydrax_discretization__cell_mesh.CellMesh.from_polyhedra(
            exact.rounded_vertices,
            cells,
            block_name="restricted-power",
            numeric_version=revision,
        )
        geometry = _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.power(
            mesh, source, budget=budget
        )
        # CellMesh groups arities, retaining input cell ids. Reorder all ancestry by
        # those ids rather than assuming publication preserves native row order.
        order = np.asarray(
            require_polyhedral_connectivity(mesh).cell_global_ids, dtype=np.int64
        )
        native_faces = {
            tuple(sorted(map(int, data.face_vertices[a:b]))): index
            for index, (a, b) in enumerate(
                zip(data.face_offsets[:-1], data.face_offsets[1:], strict=True)
            )
        }
        connectivity = require_polyhedral_connectivity(mesh)
        if connectivity.edge_count != data.preflight.edges:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
                "polyhedral.preflight.edges must match canonical topology; "
                f"got {data.preflight.edges} != {connectivity.edge_count}",
                stage="polyhedral_candidate",
                achieved=(
                    ("native_edges", data.preflight.edges),
                    ("owner_edges", connectivity.edge_count),
                ),
            )
        offsets, values = (
            np.asarray(connectivity.face_vertex_offsets),
            np.asarray(connectivity.face_vertex_values),
        )
        face_order = _dependency_phydrax_meshing__host_banks.family_host_array(
            (connectivity.face_count,), np.int64, scope=scope
        )
        for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            face_order[face] = native_faces[tuple(sorted(map(int, values[a:b])))]
        del cells, native_faces, offsets, values
        storage.retain(
            retained_bound, mesh, order, record_count_upper=512 + 64 * len(data.cell_sites)
        )
        return mesh, geometry, order, face_order


def _feature_fraction_bits(
    rows: Iterable[Iterable[SupportsFloat | SupportsIndex]],
) -> int:
    """Profile operands without first allocating their rational lifts."""
    # sqrt_upper also compares the square of a lifted binary64: its numerator
    # and denominator each have at most 1075 bits, including subnormals.
    bits = 1075
    for row in rows:
        for value in row:
            if isinstance(value, Fraction):
                bits = max(
                    bits, abs(value.numerator).bit_length(),
                    value.denominator.bit_length(),
                )
            else:
                bits = max(
                    bits, planar_primitives_owner.binary64_fraction_bits(float(value))
                )
    return bits


def _feature_entry_storage(count: int) -> int:
    """Bound integer-pair dictionary/list/set entries on the supported host.

    Each entry allows a 64-byte pair tuple, three 36-byte index integers,
    128-byte sparse hash-table slot, a 64-byte list header, four 16-byte
    overallocated reference slots and a 64-byte sorting decoration.
    """
    return count * (64 + 3 * 36 + 128 + 64 + 4 * 16 + 64)


def _feature_bvh_storage(count: int) -> int:
    """Median BVH build plus host one-item-query storage, before either runs.

    _bvh._build_split_levels has at most 2*n-1 nodes; _median_split visits
    at most n items per level; _pack has at most 16*n padded leaf slots.
    Sum its existing array-admission formulas over all levels, rather than
    replacing the caller's scientific work charges with BVH build charges.
    Host traversal against one leaf visits at most all nodes and 16*n padded
    pairs. Its arrays allow 20 index/mask words and four 3-vector box banks
    per pair, plus 32 words per node for pending/frontier banks.
    Device buffers/compiler storage are not measured or enforced here.
    """
    nodes, levels, payload = max(1, 2 * count - 1), count.bit_length() + 1, 16 * count
    build = (
        96 * count + 48 * count + 8 * (count + 2) + 64 * nodes
        + 8 * (8 * count + 12 * count + 24 * count) * levels
        + 8 * (20 * nodes + 18 * count + 6 * payload + 12 * nodes)
    )
    traversal = 8 * ((20 + 4 * 3) * payload + 32 * nodes)
    # Array headers and per-level Python list/reference owners. The build has
    # fewer than 20 arrays per node/level; traversal fewer than 128 fixed banks.
    containers = 256 * (20 * nodes + 20 * levels + 128)
    return build + traversal + containers


def _admitted_feature_bvh_storage(_bytes_upper: int, /) -> None:
    """Accept metered BVH build arrays already covered by ``_feature_bvh_storage``.

    Feature callers set that bound on their host workspace before building, so
    the per-pass build admissions must not be charged a second time.
    """


def _diameters(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    # Twice the radius about the rational vertex mean bounds the cell diameter.
    c = require_polyhedral_connectivity(mesh)
    with NativeHostStorageWorkspace(scope) as storage:
        # Source object-array slots, topology host copies, and one cell's
        # tuple-of-triples references. Fraction payloads are admitted separately.
        storage.set_bound(
            8 * (3 * mesh.coordinates.shape[0] + c.cell_vertex_offsets.size
                 + c.cell_vertex_values.size)
            + 256 * (8 + c.cell_vertex_values.size)
        )
        points = np.asarray(geometry.source_coordinates(budget=budget), dtype=object)
        offsets, values = np.asarray(c.cell_vertex_offsets), np.asarray(c.cell_vertex_values)
        bounds = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(offsets) - 1,), np.float64, scope=scope
        )
        bits = _feature_fraction_bits(points)
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            count = int(b - a)
            # A mean expands n coordinate operands, then division by n.
            # Each squared difference expands 2*(n+1) operands; summing three
            # squares and cross-multiplying comparisons needs <=12*(n+1).
            # 16 also covers division/count carry and sqrt_upper's float lift.
            with budget.temporary_scope():
                feature_fraction_owner.reserve_fraction_work(
                    budget, 0, 3 * count + 32,
                    16 * (count + 1) * bits + 64 * (count + 1).bit_length(),
                )
                vertices = tuple(tuple(points[index]) for index in values[a:b])
                center = tuple(
                    sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
                    for axis in range(3)
                )
                squared_radius = max(
                    sum(((point[axis] - center[axis]) ** 2 for axis in range(3)), Fraction(0))
                    for point in vertices
                )
                bounds[cell] = sqrt_upper(4 * squared_radius)
                del vertices, center, squared_radius
        return bounds


def _centroid_in_kernel(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    cell: int,
    point: np.ndarray,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:

    c = require_polyhedral_connectivity(mesh)
    with NativeHostStorageWorkspace(scope) as storage:
        storage.set_bound(
            8 * (3 * mesh.coordinates.shape[0] + c.cell_face_offsets.size
                 + 2 * c.cell_face_values.size + c.face_vertex_offsets.size
                 + c.face_vertex_values.size)
            + 256 * (12 + c.face_vertex_values.size)
            # The query remains live while plane_key admits its own scratch.
            + feature_fraction_owner.fraction_storage_upper(
                3, _feature_fraction_bits((point,))
            )
        )
        face_offsets, faces = np.asarray(c.cell_face_offsets), np.asarray(c.cell_face_values)
        signs = np.asarray(c.cell_face_sign_values)
        offsets, vertices = (
            np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values),
        )
        coordinates = np.asarray(geometry.source_coordinates(budget=budget), dtype=object)
        feature_fraction_owner.reserve_fraction_work(
            budget, 0, 8, 2 * _feature_fraction_bits((point,)) + 16
        )
        query = rational_vector(point)
        start, stop = face_offsets[cell : cell + 2]
        for face, sign in zip(faces[start:stop], signs[start:stop], strict=True):
            a, b = offsets[face : face + 2]
            # Gather one face's exact corners, then one source containment
            # query against its supporting halfspace.
            budget.reserve(1 + 3 * int(b - a))
            budget.reserve_geometry_queries(1)
            loop = vertices[a:b]
            if sign < 0:
                loop = loop[::-1]
            polygon = tuple(tuple(coordinates[index]) for index in loop)
            key = next(
                (
                    planar_primitives_owner.plane_key(
                        (polygon[0], polygon[index], polygon[index + 1]), budget=budget
                    )
                    for index in range(1, len(polygon) - 1)
                    if planar_primitives_owner.plane_key(
                        (polygon[0], polygon[index], polygon[index + 1]), budget=budget
                    )
                    is not None
                ),
                None,
            )
            if key is None:
                return False
            # Four affine terms and the orientation product; exact comparison
            # needs at most sixteen input occurrences and eight live results.
            feature_fraction_owner.reserve_fraction_work(
                budget, 0, 8, 16 * _feature_fraction_bits((key[0], query)) + 64
            )
            if key[3] * sum(
                (key[0][axis] * query[axis] for axis in range(3)), key[0][3]
            ) >= 0:
                return False
        return True


@checked
def sqrt_upper(value: _owner_fractions.Fraction) -> float:
    """Return a binary64 upper bound for the square root."""

    if value == 0:
        return 0.0
    if float(value) == 0:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.FEATURE_MEASURE_UNREPRESENTABLE
        )
    result = float(np.nextafter(math.sqrt(float(value)), np.inf))
    if not math.isfinite(result):
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.FEATURE_MEASURE_UNREPRESENTABLE
        )
    while _owner_fractions.Fraction(result) ** 2 < value:
        result = float(np.nextafter(result, np.inf))
    return result


def _feature_coordinate_fraction(
    value: Fraction | SupportsFloat | SupportsIndex | str, /
) -> Fraction:
    """Retain rational coordinates or lift a binary64 coordinate component."""
    match value:
        case Fraction():
            return value
        case scalar:
            return Fraction(float(scalar))


def _segment_evidence(
    ends: np.ndarray, a: np.ndarray, b: np.ndarray, *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_owner_fractions.Fraction, _owner_fractions.Fraction, float, float]:

    bits = max(_feature_fraction_bits(ends), _feature_fraction_bits((a, b)))
    # Differences use two operands; a 3D dot uses twelve; its quotient
    # uses twenty-four. A clamped projection residual uses 28, its squared
    # norm 168, and exact comparison cross-products 336. 384 includes carries.
    # Three origin/direction/delta vectors, endpoint results and expression
    # temporaries occupy at most 32 + 2*len(ends) rational slots. Work visits
    # each rational operation: 18 for the origin, direction and squared length,
    # then at most 30 per end (lift, delta, dot, quotient, clamp, residual)
    # plus its share of the final extrema and two square-root bounds.
    feature_fraction_owner.reserve_fraction_work(
        budget, 24 + 32 * len(ends), 32 + 2 * len(ends), 384 * bits + 512
    )
    origin = tuple(_feature_coordinate_fraction(x) for x in a)
    direction = tuple(
        _feature_coordinate_fraction(y) - x for x, y in zip(origin, b, strict=True)
    )
    squared_length = sum((x * x for x in direction), _owner_fractions.Fraction(0))
    parameters: list[_owner_fractions.Fraction] = []
    squared_distances: list[_owner_fractions.Fraction] = []
    for end in ends:
        delta = tuple(
            _feature_coordinate_fraction(x) - y for x, y in zip(end, origin, strict=True)
        )
        parameter = (
            sum(
                (x * y for x, y in zip(delta, direction, strict=True)),
                _owner_fractions.Fraction(0),
            )
            / squared_length
        )
        parameters.append(parameter)
        nearest = min(
            _owner_fractions.Fraction(1), max(_owner_fractions.Fraction(0), parameter)
        )
        squared_distances.append(
            sum(
                ((x - nearest * y) ** 2 for x, y in zip(delta, direction, strict=True)),
                _owner_fractions.Fraction(0),
            )
        )
    return (
        max(_owner_fractions.Fraction(0), min(parameters)),
        min(_owner_fractions.Fraction(1), max(parameters)),
        sqrt_upper(max(squared_distances)),
        sqrt_upper(squared_length),
    )


def _source_feature_edges(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    recovery: PlcRecovery3D,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    """Return semantic curve edges and geometric creases within facet groups."""

    occurrences = complex_.polygon_vertices.size
    triangle_count = len(recovery.input_triangles)
    semantic_count = len(recovery.plc_edges)
    with NativeHostStorageWorkspace(scope) as storage:
        # At most one incidence entry per polygon edge occurrence, one retained
        # triangle per input triangle, and semantic+incidence output pairs.
        # Account sorting, facet sets and known-pair sets as separate entries.
        storage.set_bound(
            _feature_entry_storage(4 * occurrences + 2 * semantic_count + triangle_count)
            + 24 * len(complex_.vertices) + 72 * triangle_count
            + 8 * occurrences + 256 * (triangle_count + 16)
        )
        # Host visits: the vertex bank copy, one gathered triangle per input
        # triangle, each polygon edge occurrence's sorted key and append, the
        # semantic edge list and set, and the unknown-entry query count below.
        scope.charge(
            work=3 * len(complex_.vertices) + 10 * triangle_count + 7 * occurrences
            + 3 * semantic_count
        )
        source_vertices = np.asarray(complex_.vertices, dtype=np.float64)
        triangles: dict[int, np.ndarray] = {}
        for triangle, polygon in zip(
            recovery.input_triangles, recovery.input_polygons, strict=True
        ):
            triangles.setdefault(int(polygon), source_vertices[triangle])
        incidence: dict[tuple[int, ...], list[int]] = {}
        offsets = complex_.polygon_offsets
        for polygon, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            loop = complex_.polygon_vertices[a:b]
            for first, second in zip(loop, np.roll(loop, -1), strict=True):
                incidence.setdefault(tuple(sorted((int(first), int(second)))), []).append(
                    polygon
                )
        edges = [tuple(map(int, pair)) for pair in recovery.plc_edges]
        known = {tuple(sorted(pair)) for pair in edges}
        # Every unknown entry's occurrence after its first is one coplanarity
        # query of an incident polygon against the anchor (three exact
        # orientations, their broadcast operands and sign test). Admit the
        # entry sort, the facet sets over all occurrences and the output rows.
        entries = len(incidence)
        pairs = sum(
            len(polygons) - 1 for edge, polygons in incidence.items() if edge not in known
        )
        scope.charge(
            work=entries * (entries.bit_length() + 1) + 2 * occurrences + 36 * pairs
            + 2 * (semantic_count + entries),
            geometry_queries=pairs,
        )
        for edge, polygons in sorted(incidence.items()):
            if edge in known:
                continue
            facets = {int(complex_.polygon_facets[polygon]) for polygon in polygons}
            sharp = len(facets) > 1 or len(polygons) == 1
            anchor = triangles[polygons[0]]
            for polygon in polygons[1:]:
                other = triangles[polygon]
                signs = _owner_phydrax__meshcore.exact_orient3d(
                    np.broadcast_to(anchor[0], other.shape),
                    np.broadcast_to(anchor[1], other.shape),
                    np.broadcast_to(anchor[2], other.shape),
                    other,
                )
                sharp = sharp or bool(np.any(signs != 0))
            if sharp:
                edges.append(edge)
        output = _dependency_phydrax_meshing__host_banks.family_host_array(
            (len(edges), 2), np.int32, scope=scope
        )
        for row, edge in enumerate(edges):
            output[row] = edge
        return output


@final
class _FeaturePreparation(StrictModule):
    carrier_sources: dict[tuple[int, ...], int]
    source_points: np.ndarray
    tree: _owner_phydrax__bvh.PackedBVH
    lengths: tuple[float, ...]


@final
class _FeatureMatch(StrictModule):
    source: int
    work_units: int


@final
class _FeatureCoverage(StrictModule):
    gaps: np.ndarray


@final
class _FeatureAncestry(StrictModule):
    sources: np.ndarray
    distances: np.ndarray
    gaps: np.ndarray
    work_units: int


def _prepare_feature_ancestry(
    recovery: PlcRecovery3D, source_edges: np.ndarray, *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _FeaturePreparation:
    count = len(source_edges)
    storage = NativeHostStorageWorkspace(scope).open()
    try:
        storage.set_bound(
            _feature_entry_storage(count + len(recovery.segments))
            + _feature_bvh_storage(count) + 48 * count + 32 * count + 256
        )
        # Carrier-key map over source edges and recovered segments, the gathered
        # endpoint bank and its per-edge box extrema. The metered median build
        # charges its own split/pack visits before each pass.
        budget.reserve(2 * (count + len(recovery.segments)) + 18 * count)
        carrier_sources = {
            tuple(sorted(map(int, pair))): source for source, pair in enumerate(source_edges)
        }
        for pair, source in zip(recovery.segments, recovery.segment_sources, strict=True):
            carrier_sources[tuple(sorted(map(int, pair)))] = int(source)
        source_points = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count, 2, 3), np.float64, scope=scope
        )
        np.take(recovery.points, source_edges, axis=0, out=source_points)
        tree = _owner_phydrax__bvh.prepare_bvh(
            np.min(source_points, axis=1), np.max(source_points, axis=1),
            dtype=np.float64,
            _charge_work=budget.reserve,
            _reserve_storage=_admitted_feature_bvh_storage,
        )
        def segment_length(pair: np.ndarray) -> float:
            # Only the binary64 length survives; exact projection/square-root
            # scratch belongs to this one source segment, not every segment.
            with budget.temporary_scope():
                return _segment_evidence(pair, pair[0], pair[1], budget=budget)[3]

        lengths = tuple(segment_length(pair) for pair in source_points)
        result = _FeaturePreparation(
            carrier_sources=carrier_sources, source_points=source_points,
            tree=tree, lengths=lengths,
        )
        # The dictionary, tree banks and lengths remain live after this helper
        # returns. Release their admission only when their actual owner dies.
        weakref.finalize(result, storage.close)
        return result
    except BaseException:
        storage.close()
        raise


def _match_feature_carrier(
    preparation: _FeaturePreparation,
    recovery: PlcRecovery3D,
    key: tuple[int, ...],
    edge: int,
    work_units: int,
    work_limit: int,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _FeatureMatch:
    tree, source_points = preparation.tree, preparation.source_points
    with NativeHostStorageWorkspace(scope) as storage:
        storage.set_bound(
            _feature_bvh_storage(1) + _feature_bvh_storage(len(source_points))
            + _feature_entry_storage(len(source_points)) + 48 + 256
        )
        carrier_points = recovery.points[list(key)]
        budget.reserve(12)
        carrier_tree = _owner_phydrax__bvh.prepare_bvh(
            np.min(carrier_points, axis=0)[None, :],
            np.max(carrier_points, axis=0)[None, :],
            dtype=np.float64,
            _charge_work=budget.reserve,
            _reserve_storage=_admitted_feature_bvh_storage,
        )
        matched = []
        # Each evaluated node/item box test is charged before it runs.
        for candidates, _ in _owner_phydrax__bvh.bvh_overlap_pair_blocks(
            tree, carrier_tree, include_touching=True, visit=budget.reserve
        ):
            for source in candidates:
                work_units += 1
                if work_units > work_limit:
                    raise PolyhedralGenerationError(
                        PolyhedralGenerationReason.FEATURE_CANDIDATE_BUDGET,
                        entities=(edge,),
                        achieved=work_units,
                    )
                budget.reserve(1)
                # One exact carrier/source-segment overlap query.
                budget.reserve_geometry_queries(1)
                a, b = source_points[source]
                with budget.temporary_scope():
                    start, stop, residual, _ = _segment_evidence(
                        carrier_points, a, b, budget=budget
                    )
                    if residual == 0.0 and stop > start:
                        matched.append(int(source))
                    del start, stop
        if len(matched) > 1:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.AMBIGUOUS_FEATURE_ANCESTRY,
                entities=(edge,),
            )
        return _FeatureMatch(source=matched[0] if matched else -1, work_units=work_units)


def _feature_coverage(
    intervals: list[list[tuple[Fraction, Fraction]]],
    lengths: tuple[float, ...],
    gaps: np.ndarray,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _FeatureCoverage:
    missing = tuple(source for source, pieces in enumerate(intervals) if not pieces)
    if missing:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.MISSING_FEATURE_EDGE,
            entities=missing,
            achieved={"covered_segments": 0.0},
        )
    # Coverage subtracts two retained parameters then multiplies one lifted
    # binary64 length; sorting comparisons cross-multiply two parameters.
    # Work: per piece its two-parameter bit profile, sort visit and two fold
    # comparisons/subtractions; per source its sort levels and gap rounding.
    sizes = tuple(len(pieces) for pieces in intervals)
    budget.reserve(
        len(sizes) + sum(size * (size.bit_length() + 7) + 8 for size in sizes)
    )
    bits = max(
        (_feature_fraction_bits(piece) for piece in intervals), default=1075
    )
    feature_fraction_owner.reserve_fraction_work(budget, 0, 12, 8 * bits + 64)
    for source, pieces in enumerate(intervals):
        cursor, gap = _owner_fractions.Fraction(0), _owner_fractions.Fraction(0)
        for start, stop in sorted(pieces):
            gap = max(gap, start - cursor)
            cursor = max(cursor, stop)
        gap = max(gap, _owner_fractions.Fraction(1) - cursor) * _owner_fractions.Fraction(
            lengths[source]
        )
        upper_gap = float(gap)
        if _owner_fractions.Fraction(upper_gap) < gap:
            upper_gap = float(np.nextafter(upper_gap, np.inf))
        gaps[source] = upper_gap
    return _FeatureCoverage(gaps=gaps)


def _feature_ancestry(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    recovery: PlcRecovery3D,
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    source_edges: np.ndarray,
    work_limit: int,
    *,
    scope: NativeExecutionScope,
    vertex_carriers: np.ndarray | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _FeatureAncestry:
    """Return carrier ancestry and affine feature-distance/coverage bounds.

    Recovered carrier ancestry distinguishes nearby thin features. Unregistered
    crease subedges must lie on a source segment. Source coefficients supply
    ideal coordinates independently of the rounded carrier coordinates.
    """

    connectivity = require_polyhedral_connectivity(mesh)
    edge_count, source_count = connectivity.edges.shape[0], len(source_edges)
    carriers = (
        diagram.construction.vertex_carriers
        if vertex_carriers is None
        else vertex_carriers
    )
    with NativeHostStorageWorkspace(scope) as storage:
        # One int64 edge copy, one object coordinate view, and one selected
        # two-vertex carrier bank. Include ndarray headers before np.asarray.
        arrays = (
            16 * edge_count + 24 * mesh.coordinates.shape[0]
            + 16 * carriers.shape[1] + 256 * 12
        )
        storage.set_bound(arrays)
        # The edge copy, the three output fills and both coordinate bit
        # profiles (three components per point).
        budget.reserve(
            3 * edge_count + 2 * source_count
            + 3 * (mesh.coordinates.shape[0] + len(recovery.points))
        )
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        points = np.asarray(geometry.source_coordinates(budget=budget), dtype=object)
        sources = _dependency_phydrax_meshing__host_banks.family_host_array(
            (edge_count,), np.int64, scope=scope
        )
        distances = _dependency_phydrax_meshing__host_banks.family_host_array(
            (source_count,), np.float64, scope=scope
        )
        gaps = _dependency_phydrax_meshing__host_banks.family_host_array(
            (source_count,), np.float64, scope=scope
        )
        sources.fill(-1)
        distances.fill(0.0)
        gaps.fill(0.0)
        if source_count == 0:
            return _FeatureAncestry(
                sources=sources, distances=distances, gaps=gaps, work_units=0
            )
        bits = max(_feature_fraction_bits(points), _feature_fraction_bits(recovery.points))
        # Each output edge contributes at most one interval (two exact
        # parameters) and one previously unknown carrier-cache entry.
        # _segment_evidence's expression expansion bounds each retained
        # parameter too; sorting uses at most one extra reference per interval.
        storage.set_bound(
            arrays + _feature_entry_storage(2 * edge_count + source_count)
            + 16 * edge_count
            + feature_fraction_owner.fraction_storage_upper(
                2 * edge_count, 384 * bits + 512
            )
            + _feature_entry_storage(2 * carriers.shape[1])
        )
        preparation = _prepare_feature_ancestry(
            recovery, source_edges, scope=scope, budget=budget
        )
        carrier_sources, source_points = (
            preparation.carrier_sources, preparation.source_points,
        )
        intervals: list[list[tuple[Fraction, Fraction]]] = [[] for _ in source_edges]
        work_units = 0
        for edge, pair in enumerate(edges):
            work_units += 1
            if work_units > work_limit:
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.FEATURE_CANDIDATE_BUDGET,
                    entities=(edge,),
                    achieved=work_units,
                )
            # The two endpoints' carrier rows, their set and sorted key.
            budget.reserve(1 + 3 * carriers.shape[1])
            key = tuple(
                sorted({int(point) for point in carriers[pair].reshape(-1) if point >= 0})
            )
            if len(key) != 2:
                continue
            if key not in carrier_sources:
                matched = _match_feature_carrier(
                    preparation, recovery, key, edge, work_units, work_limit,
                    scope=scope, budget=budget,
                )
                work_units = matched.work_units
                carrier_sources[key] = matched.source
            source = carrier_sources[key]
            if source < 0:
                continue
            sources[edge] = source
            a, b = source_points[source]
            # One exact mesh-edge/source-segment containment query.
            budget.reserve_geometry_queries(1)
            with budget.temporary_scope():
                start, stop, residual, _ = _segment_evidence(
                    points[pair], a, b, budget=budget
                )
                distances[source] = max(distances[source], residual)
                if stop > start:
                    # The retained parameters' full payload was already
                    # admitted in storage before the edge loop.
                    intervals[source].append((start, stop))
                del start, stop
        with budget.temporary_scope():
            coverage = _feature_coverage(
                intervals, preparation.lengths, gaps, budget=budget
            )
        result = _FeatureAncestry(
            sources=sources, distances=distances, gaps=coverage.gaps, work_units=work_units
        )
        del intervals, carrier_sources, preparation
        return result


def _numerical_power_sites(
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rounded optimization coordinates for retained periodic images."""

    preparation = diagram.periodic_preparation
    if preparation is None:
        return diagram.points, np.arange(len(diagram.points), dtype=np.int64)
    points = _dependency_phydrax_meshing__host_banks.family_host_array(
        (len(preparation.image_sites), 3), np.float64, scope=scope
    )
    offsets, components = (
        preparation.image_coordinate_offsets,
        preparation.image_coordinate_components,
    )
    for coordinate, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        with budget.temporary_scope():
            bank = planar_primitives_owner.rational_points(
                components[a:b].reshape((-1, 1)), budget=budget
            )
            planar_primitives_owner.reserve_fraction_work(
                bank, len(bank) + 1, 1, max(1, len(bank)), budget=budget
            )
            value = sum((row[0] for row in bank), Fraction(0))
            points[coordinate // 3, coordinate % 3] = float(value)
    return points, preparation.image_sites


def _polyhedral_coordinate_ticket(
    maximum_work: int,
    maximum_scratch_bytes: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> NativeBatchTicket:
    """Reserve a source-coordinate phase before host preparation."""
    remaining = scope.remaining()
    return scope.batch(
        min(maximum_work, remaining.remaining_work_units),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(maximum_scratch_bytes, remaining.remaining_managed_bytes),
    )


@contextmanager
def _polyhedral_coordinate_phase(
    maximum_work: int,
    maximum_scratch_bytes: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Bind source-coordinate work and native allocations to one admitted phase."""
    ticket = _polyhedral_coordinate_ticket(
        maximum_work, maximum_scratch_bytes, scope=scope
    )
    with (
        budget.bind_ticket(ticket),
        ticket.native_target(),
        budget.bound_stage(maximum_work, maximum_scratch_bytes),
    ):
        yield


@final
class _PolyhedralSites(StrictModule):
    points: np.ndarray
    weights: np.ndarray
    protected: frozenset[int]
    targets: np.ndarray
    split: np.ndarray
    parents: np.ndarray
    preserve_vertices: bool


@final
class _PolyhedralCarrier(StrictModule):
    recovery: PlcRecovery3D
    domain: domain_coverage_owner.PiecewiseLinearDomain
    facets: np.ndarray
    sites: _PolyhedralSites
    work_units: int


@final
class _PolyhedralPublication(StrictModule):
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    order: np.ndarray
    face_order: np.ndarray
    unresolved: np.ndarray
    work_units: int


@final
class _PolyhedralMeasures(StrictModule):
    achieved: np.ndarray
    empty: np.ndarray
    diameters: np.ndarray
    large: np.ndarray
    residual: float
    work_units: int


@final
class _PolyhedralWeightUpdate(StrictModule):
    values: np.ndarray
    work_units: int


@final
class _PolyhedralRelocation(StrictModule):
    points: np.ndarray
    work_units: int


@final
class _PolyhedralFinalization(StrictModule):
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    order: np.ndarray
    regions: np.ndarray
    face_recovery_triangles: np.ndarray
    face_facets: np.ndarray
    interface_faces: np.ndarray
    sheet_faces: np.ndarray
    diameters: np.ndarray
    achieved: np.ndarray
    residual: float
    seam_fragments: tuple[PeriodicPolyhedralSeamFragment, ...]
    assembly: _PeriodicPolyhedralAssembly | None
    work_units: int


@final
class _PolyhedralFeatures(StrictModule):
    sources: np.ndarray
    source_edges: np.ndarray
    distances: np.ndarray
    gaps: np.ndarray
    work_units: int


@final
class _PolyhedralLineage(StrictModule):
    components: np.ndarray


@final
class _PolyhedralRefinement(StrictModule):
    finished: bool
    update_weights: bool
    insert_sites: bool
    relocate_sites: bool


def _plan_polyhedral_refinement(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    schedule_: NativePolyhedralSchedule,
    refinement_steps: int,
    lloyd_steps: int,
    *,
    scope: NativeExecutionScope,
) -> _PolyhedralRefinement:
    """Choose the next refinement actions or refuse unmet bounded requests."""
    needs_weights = (
        bool(state.targets.size)
        and measures.residual > schedule_.relative_volume_tolerance
    )
    if (
        not measures.large.size
        and not needs_weights
        and lloyd_steps >= schedule_.lloyd_steps
    ):
        if schedule_.require_every_site and measures.empty.size:
            scope.charge(work=measures.empty.size)
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.REQUIRED_EMPTY_SITES,
                entities=tuple(map(int, measures.empty)),
            )
        return _PolyhedralRefinement(
            finished=True, update_weights=False, insert_sites=False, relocate_sites=False
        )
    if refinement_steps >= schedule_.maximum_refinement_steps:
        scope.charge(work=2 * measures.large.size + measures.diameters.size)
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.REFINEMENT_UNRESOLVED,
            entities=tuple(map(int, publication.order[measures.large])),
            achieved={
                "diameter_upper": float(np.max(measures.diameters)),
                "relative_volume_residual": measures.residual,
            },
        )
    if (
        measures.large.size
        and state.targets.size
        and not needs_weights
        and lloyd_steps >= schedule_.lloyd_steps
    ):
        scope.charge(work=2 * measures.large.size + measures.diameters.size)
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.FIXED_SITE_DIAMETER_UNRESOLVED,
            entities=tuple(map(int, publication.order[measures.large])),
            achieved={"cell_diameter_upper": float(np.max(measures.diameters))},
        )
    return _PolyhedralRefinement(
        finished=False,
        update_weights=needs_weights,
        insert_sites=bool(measures.large.size) and not bool(state.targets.size),
        relocate_sites=lloyd_steps < schedule_.lloyd_steps,
    )


def _polyhedral_input_scalar_count(
    value: ConvertibleToArray | None, /, *, scope: NativeExecutionScope
) -> int:
    """Admit the actual input traversal before normalization allocates a bank."""
    with NativeHostStorageWorkspace(scope) as storage:
        storage.set_bound(1024)
        scope.charge(work=1)
        pending: list[tuple[object, bool]] = [(value, False)]
        visiting: set[int] = set()
        count = 0
        while pending:
            item, leaving = pending.pop()
            if leaving:
                visiting.remove(id(item))
                continue
            if item is None:
                continue
            shape = getattr(item, "shape", None)
            if shape is not None:
                scope.charge(work=len(shape))
                count += math.prod(shape)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if id(item) in visiting:
                    raise ValueError("Polyhedral input sequences must be acyclic.")
                scope.charge(work=len(item) + 1)
                storage.set_bound(
                    1024 + 512 * (len(pending) + len(item) + len(visiting) + 2)
                )
                visiting.add(id(item))
                pending.append((item, True))
                pending.extend((child, False) for child in item)
            else:
                count += 1
        return count


def _recover_polyhedral_source(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    seeds: ConvertibleToArray | None,
    seed_regions: ConvertibleToArray | None,
    schedule_: NativePolyhedralSchedule,
    *,
    scope: NativeExecutionScope,
    completed_work: int,
) -> PlcRecovery3D:
    # The typed authored PLC already owns its complete input vertex prefix.
    # Native recovery cannot accept an output capacity smaller than that input;
    # this is a resource shortfall, not malformed recovery geometry.
    required_vertices = complex_.vertices.shape[0]
    if schedule_.maximum_vertices < required_vertices:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.SOURCE_VERTEX_BUDGET,
            achieved={
                "required_vertices": required_vertices,
                "maximum_vertices": schedule_.maximum_vertices,
            },
        )
    # _plc_host_arrays may hold an int64 validation conversion and an int32
    # execution conversion plus two bool comparison masks per index scalar.
    # _points holds one binary64 conversion and one finite mask per scalar.
    index_scalars = sum(
        value.size
        for value in (
            complex_.polygon_offsets,
            complex_.polygon_vertices,
            complex_.polygon_facets,
            complex_.facet_regions,
            complex_.segments,
        )
    ) + _polyhedral_input_scalar_count(seed_regions, scope=scope)
    point_scalars = complex_.vertices.size + _polyhedral_input_scalar_count(seeds, scope=scope)
    # Counter/status query banks (including refusal evidence) are fixed-size;
    # their wrapper and scalar metadata are admitted before the native call.
    input_bound = (
        14 * index_scalars
        + 9 * point_scalars
        + 12 * 256
        + 32 * 160
        + 8 * (6 + 14 + 6 + 5 + 5 + 2)
        + 2 * 4
    )
    output_bound = 0
    with (
        scope.ticket_view() as native_view,
        native_view.native_target(),
        _PolyhedralTopologyStorage(scope) as storage,
    ):
        storage.workspace.set_bound(input_bound)
        # Host casts/range/finite scans precede the native recovery handle.
        scope.charge(work=64 + 4 * point_scalars + 6 * index_scalars)

        def admit_host_output(bytes_upper: int) -> None:
            nonlocal output_bound
            # Returned counter and allocator records contain 14+6 signed/uint64
            # integer owners in addition to the sixteen exported array banks.
            output_bound = bytes_upper + 2 * 256 + 20 * 160
            storage.workspace.set_bound(input_bound + output_bound)
            # The ABI's sixteen banks occupy >=4 bytes per logical export
            # word, including the packed per-vertex int8 banks. Zeroing and
            # native export copying are separate passes over that payload.
            payload = bytes_upper - 17 * 256
            scope.charge(work=2 * ((payload + 3) // 4))

        recovery = recover_plc_3d(
            complex_.vertices,
            complex_.polygon_offsets,
            complex_.polygon_vertices,
            complex_.polygon_facets,
            complex_.facet_regions,
            segments=complex_.segments,
            seeds=seeds,
            seed_regions=seed_regions,
            boundary_policy=complex_.boundary,
            max_vertices=schedule_.maximum_vertices,
            max_tetrahedra=schedule_.maximum_cells,
            work_limit=schedule_.maximum_work_units - completed_work,
            max_scratch_bytes=schedule_.maximum_scratch_bytes,
            admit_host_output=admit_host_output,
        )
        storage.retain(output_bound, recovery, record_count_upper=128)
        return recovery


def _prepare_polyhedral_carrier(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    sites: ConvertibleToArray | None,
    weights: ConvertibleToArray | None,
    seeds: ConvertibleToArray | None,
    seed_regions: ConvertibleToArray | None,
    protected_vertices: tuple[int, ...],
    source_id: str,
    schedule_: NativePolyhedralSchedule,
    scope: NativeExecutionScope,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralCarrier:
    recovery = _recover_polyhedral_source(
        complex_,
        seeds,
        seed_regions,
        schedule_,
        scope=scope,
        completed_work=budget.work_units,
    )
    triangles = len(recovery.input_triangles)
    site_scalars = 3 if sites is None else _polyhedral_input_scalar_count(sites, scope=scope)
    site_rows = max(1, site_scalars)
    weight_scalars = site_rows if weights is None else _polyhedral_input_scalar_count(weights, scope=scope)
    target_scalars = len(schedule_.target_site_volumes)
    protected_count = len(schedule_.protected_sites)
    # Domain output stores triangle indices (3*i64) and region pairs (2*i64).
    # State output owns site/weight/target f64 banks, parent i64 and split i8,
    # plus a protected-site set (64-byte slots + <=160-byte int64 atoms).
    retained = (
        40 * triangles
        + 8 * (site_scalars + weight_scalars + target_scalars)
        + 9 * site_rows
        + 224 * protected_count
        + 256 * (16 + len(complex_.region_ids))
        + 4 * (len(source_id) + sum(map(len, complex_.region_ids)))
    )
    # Domain filtering and exact-degeneracy input adaptation coexist: gathered
    # 3x3 f64 corners, current/previous projected 3x2 banks, three contiguous
    # predicate inputs, index conversions, incidence banks and bool/sign masks.
    # Their widths total <=256 bytes/triangle. Site validation/unique owns at
    # most two site copies, a sorted record bank, permutation/sort scratch and
    # finite/distinct masks, bounded by 12 f64 words per supplied scalar.
    temporary = (
        256 * triangles
        + 96 * site_scalars
        + 16 * weight_scalars
        + 224 * protected_count
        + 32 * 256
        # Source-coordinate finite mask and streaming noncontiguous hashes.
        + complex_.vertices.size + 2 * 65536
    )
    with _PolyhedralTopologyStorage(scope) as storage:
        storage.workspace.set_bound(retained + temporary)
        # Input conversion/copies are distinct from the later finite and
        # duplicate scans. Their scalar counts were admitted without copying.
        scope.charge(
            work=(
                0 if sites is None else
                1 if isinstance(sites, np.ndarray) and sites.dtype == np.float64 else site_scalars
            ) + (
                0 if weights is None else
                1 if isinstance(weights, np.ndarray) and weights.dtype == np.float64 else weight_scalars
            ) + target_scalars
        )
        result = _build_polyhedral_carrier(
            complex_, sites, weights, protected_vertices, source_id, schedule_,
            recovery, scope=scope, budget=budget,
        )
        storage.retain(
            retained, result.domain, result.sites,
            record_count_upper=128 + len(complex_.region_ids),
        )
        return result


def _build_polyhedral_carrier(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    sites: ConvertibleToArray | None,
    weights: ConvertibleToArray | None,
    protected_vertices: tuple[int, ...],
    source_id: str,
    schedule_: NativePolyhedralSchedule,
    recovery: PlcRecovery3D,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralCarrier:
    domain = _dependency_phydrax_meshing__volume_generation.domain(
        complex_, recovery.input_triangles, recovery.input_polygons, source_id
    )
    facets = _tet_facets(recovery, scope=scope)
    # The default singleton diagram is subsequently refined under an explicit
    # size request; it is not a disguised dense background sampler.
    if sites is None:
        # Gather four 3-vectors, reduce them and divide the three coordinates.
        scope.charge(work=27)
    site_array = (
        np.mean(recovery.points[recovery.tetrahedra[0]], axis=0)[None, :]
        if sites is None
        else np.asarray(sites, dtype=np.float64)
    )
    scope.charge(work=2 * site_array.size)
    finite_sites = bool(np.all(np.isfinite(site_array)))
    if (
        site_array.ndim != 2
        or site_array.shape[1] != 3
        or site_array.shape[0] < 1
        or not finite_sites
    ):
        raise ValueError(
            "polyhedral.sites must be a finite nonempty (n, 3) array; "
            f"got shape={site_array.shape}, finite={finite_sites}"
        )
    scope.charge(work=site_array.size)
    site_array = site_array.copy()
    if weights is None:
        scope.charge(work=site_array.shape[0])
        values = np.zeros(site_array.shape[0], dtype=np.float64)
    else:
        values = np.asarray(weights, dtype=np.float64)
        scope.charge(work=values.size)
        values = values.copy()
    scope.charge(work=2 * values.size)
    if values.shape != (site_array.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError(
            "polyhedral.weights must be a finite (site_count,) array; "
            f"got shape={values.shape}, site_count={site_array.shape[0]}"
        )
    scope.charge(work=len(schedule_.protected_sites))
    protected = set(schedule_.protected_sites)
    scope.charge(work=len(protected))
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= site_array.shape[0]
        for x in protected
    ):
        raise ValueError(
            "polyhedral.protected_sites must name existing sites; "
            f"got {schedule_.protected_sites!r}, site_count={site_array.shape[0]}"
        )
    targets = np.asarray(schedule_.target_site_volumes, dtype=np.float64)
    scope.charge(work=targets.size)
    if targets.size and targets.shape != values.shape:
        raise ValueError(
            "polyhedral.target_site_volumes must match the supplied sites; "
            f"got shape={targets.shape}, site_count={values.size}"
        )
    if targets.size:
        # Unique sorts lexicographic 3-coordinate rows before testing equality.
        count = site_array.shape[0]
        scope.charge(work=3 * count * (count.bit_length() + 1))
    if targets.size and np.unique(site_array, axis=0).shape[0] != site_array.shape[0]:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.COINCIDENT_POSITIVE_VOLUME_TARGETS
        )
    preserve_vertices = bool(protected_vertices)
    scope.charge(work=len(protected_vertices))
    if any(
        vertex < 0 or vertex >= complex_.vertices.shape[0]
        for vertex in protected_vertices
    ):
        raise ValueError(
            "polyhedral.protected_vertices must name PLC input vertices; "
            f"got {protected_vertices!r}, vertex_count={complex_.vertices.shape[0]}"
        )
    scope.charge(work=2 * site_array.shape[0])
    split = np.full(
        site_array.shape[0],
        bool(complex_.segments.shape[0]) or preserve_vertices,
        dtype=np.int8,
    )
    parents = np.arange(site_array.shape[0], dtype=np.int64)
    return _PolyhedralCarrier(
        recovery=recovery,
        domain=domain,
        facets=facets,
        sites=_PolyhedralSites(
            points=site_array,
            weights=values,
            protected=frozenset(protected),
            targets=targets,
            split=split,
            parents=parents,
            preserve_vertices=preserve_vertices,
        ),
        work_units=budget.work_units + recovery.counters.work_units,
    )


def _construct_polyhedral_publication(
    carrier: _PolyhedralCarrier,
    state: _PolyhedralSites,
    schedule_: NativePolyhedralSchedule,
    work_units: int,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    periodic_values: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    periodic_group: PeriodicIsometryGroup | None,
    record_native_phase: Callable[[str, float, int | None, int], None] | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None,
) -> _PolyhedralPublication:
    recovery, domain, facets = carrier.recovery, carrier.domain, carrier.facets
    site_array, values, split = state.points, state.weights, state.split
    if work_units >= schedule_.maximum_work_units:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.WORK_BUDGET, achieved=work_units
        )
    diagram = _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram(
        site_array,
        values,
        recovery.points,
        recovery.tetrahedra,
        recovery.tetrahedron_regions,
        facets,
        split_sites=split,
        max_pieces=schedule_.maximum_cells,
        max_vertices=(
            schedule_.maximum_vertices if limits is None
            else min(schedule_.maximum_vertices, limits.maximum_vertices)
        ),
        work_limit=schedule_.maximum_work_units - work_units,
        record_native_phase=record_native_phase,
        periodic_group=periodic_group,
        maximum_images=schedule_.maximum_work_units - work_units
        if periodic_group is not None
        else None,
        scope=scope,
        budget=budget,
        output_limits=None if limits is None else (
            _owner_phydrax__meshcore.RestrictedPowerOutputLimits(
                max_edges=limits.maximum_edges,
                max_faces=limits.maximum_faces,
                max_face_entries=limits.maximum_connectivity_entries,
                max_cells=limits.maximum_cells,
                max_site_witnesses=limits.maximum_connectivity_entries,
                # The existing topology owner uses these occurrences as its
                # conservative face/edge bounds, not unique native entities.
                max_sided_loop_occurrences=min(
                    limits.maximum_faces, limits.maximum_connectivity_entries
                ),
                max_sided_corner_occurrences=min(
                    limits.maximum_edges, limits.maximum_connectivity_entries
                ),
                max_connectivity_entries=limits.maximum_connectivity_entries,
                max_data_bytes=limits.maximum_data_bytes,
            )
        ),
    )
    preflight = diagram.construction.preflight
    native_envelope = (
        preflight.publication_connectivity_entries_upper,
        preflight.publication_numeric_data_bytes_upper,
    )
    if native_envelope != _polyhedral_numeric_publication_bounds(
        preflight.vertices, preflight.cells, preflight.sided_loop_occurrences,
        preflight.sided_corner_occurrences,
    ):
        raise RuntimeError(
            "polyhedral.preflight must match its independent publication owner"
        )
    work_units += (
        diagram.construction.counters.clips + diagram.construction.counters.walk_steps
    )
    work_units += diagram.preparation_work_units
    work_units += diagram.adjacency_work_units
    work_units += diagram.construction.source_work_units
    if work_units >= schedule_.maximum_work_units:
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.WORK_BUDGET, achieved=work_units
        )
    remaining_work = schedule_.maximum_work_units - work_units
    before = budget.work_units
    with _polyhedral_coordinate_phase(
        remaining_work, schedule_.maximum_scratch_bytes, scope=scope, budget=budget
    ):
        mesh, geometry, order, face_order = _publish(
            diagram,
            recovery,
            domain.source_revision,
            schedule_.maximum_work_units - work_units,
            domain_source=complex_,
            periodic_constraints=periodic_values,
            scope=scope,
            budget=budget,
            limits=limits,
        )
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            geometry, mesh=mesh, budget=budget
        )
    work_units += budget.work_units - before
    with _PolyhedralTopologyStorage(scope) as status_storage:
        # A copied int64-width status view and its bool comparison result.
        status_storage.workspace.set_bound(4 * 256 + 9 * len(order))
        unresolved = np.asarray(validity.status) != int(
            _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
        )
        status_storage.retain(
            256 + len(order), unresolved, record_count_upper=2
        )
    return _PolyhedralPublication(
        diagram=diagram,
        mesh=mesh,
        geometry=geometry,
        order=order,
        face_order=face_order,
        unresolved=unresolved,
        work_units=work_units,
    )


def _measure_polyhedral_publication(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    schedule_: NativePolyhedralSchedule,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralMeasures:
    cells, sites = len(publication.order), len(state.points)
    # Retained bincount, empty-site and large-cell rows. Diameter payloads are
    # native-owned. Temporary index/weight copies, comparison masks and the
    # three residual arithmetic banks coexist with these rows.
    retained = 3 * 256 + 16 * sites + 8 * cells
    temporary = 12 * 256 + 24 * cells + 32 * sites
    with _PolyhedralTopologyStorage(scope) as storage:
        storage.workspace.set_bound(retained + temporary)
        result = _measure_polyhedral_publication_active(
            publication, state, schedule_, scope=scope, budget=budget
        )
        storage.retain(
            retained, result.achieved, result.empty, result.large,
            record_count_upper=8,
        )
        return result


def _measure_polyhedral_publication_active(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    schedule_: NativePolyhedralSchedule,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralMeasures:
    diagram, mesh, geometry, order = (
        publication.diagram,
        publication.mesh,
        publication.geometry,
        publication.order,
    )
    site_array, targets = state.points, state.targets
    work_units = publication.work_units
    remaining_work = schedule_.maximum_work_units - work_units
    before = budget.work_units
    with _polyhedral_coordinate_phase(
        remaining_work, schedule_.maximum_scratch_bytes, scope=scope, budget=budget
    ):
        exact_volumes, _, _ = mapped_measure_owner.certified_cell_measures(
            mesh, geometry, maximum_work=remaining_work, budget=budget
        )
    work_units += budget.work_units - before
    # Gather and accumulate each cell; initialize site bins, compare them and
    # scan the empty-site mask. These are host passes, not geometry queries.
    scope.charge(work=2 * len(order) + 3 * len(site_array))
    achieved = np.bincount(
        diagram.cell_original_sites[order],
        weights=exact_volumes,
        minlength=site_array.shape[0],
    )
    empty = np.flatnonzero(achieved == 0)
    if schedule_.require_every_site and empty.size and not targets.size:
        scope.charge(work=empty.size)
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.REQUIRED_EMPTY_SITES,
            entities=tuple(map(int, empty)),
        )
    remaining_work = schedule_.maximum_work_units - work_units
    before = budget.work_units
    with _polyhedral_coordinate_phase(
        remaining_work, schedule_.maximum_scratch_bytes, scope=scope, budget=budget
    ):
        diameters = _diameters(mesh, geometry, scope=scope, budget=budget)
    work_units += budget.work_units - before
    if schedule_.maximum_cell_diameter is not None:
        scope.charge(work=2 * len(diameters))
    large = (
        np.flatnonzero(diameters > schedule_.maximum_cell_diameter)
        if schedule_.maximum_cell_diameter is not None
        else np.zeros(0, dtype=np.int64)
    )
    if targets.size:
        scope.charge(work=4 * targets.size)
    residual = (
        float(np.max(np.abs(achieved - targets) / targets)) if targets.size else 0.0
    )
    return _PolyhedralMeasures(
        achieved=achieved,
        empty=empty,
        diameters=diameters,
        large=large,
        residual=residual,
        work_units=work_units,
    )


def _revive_hidden_polyhedral_sites(
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    recovery: PlcRecovery3D,
    values: np.ndarray,
    targets: np.ndarray,
    empty: np.ndarray,
    metric_sites: np.ndarray,
    metric_owners: np.ndarray,
    *,
    scope: NativeExecutionScope,
) -> _PolyhedralWeightUpdate:
    data = diagram.construction
    # Revive hidden weighted sites by a sparse power-distance walk
    # at an interior carrier point, not an all-site distance table.
    scope.charge(work=48)
    point = np.mean(recovery.points[recovery.tetrahedra[0]], axis=0)
    owner = int(data.cell_sites[0])
    best = float(
        np.sum((point - metric_sites[owner]) ** 2) - values[metric_owners[owner]]
    )
    while True:
        next_owner, next_best = owner, best
        a, b = diagram.neighbor_offsets[owner : owner + 2]
        # Admit the whole sparse neighborhood before evaluating any distance.
        scope.charge(work=4 + 16 * int(b - a))
        for neighbor in diagram.neighbors[a:b]:
            if neighbor >= len(metric_sites):
                continue
            candidate = float(
                np.sum((point - metric_sites[neighbor]) ** 2)
                - values[metric_owners[neighbor]]
            )
            if candidate < next_best:
                next_owner, next_best = int(neighbor), candidate
        if next_owner == owner:
            break
        owner, best = next_owner, next_best
    for site in empty:
        scope.charge(work=2 * len(metric_owners))
        images = np.flatnonzero(metric_owners == site)
        scope.charge(work=16 * len(images) + 8)
        distance = min(
            float(np.sum((point - metric_sites[image]) ** 2)) for image in images
        )
        values[site] = float(distance - best + 0.1 * targets[site] ** (2.0 / 3.0))
    return _PolyhedralWeightUpdate(values=values, work_units=0)


def _update_polyhedral_weights(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    carrier: _PolyhedralCarrier,
    schedule_: NativePolyhedralSchedule,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralWeightUpdate:
    data = publication.diagram.construction
    # One face polygon, rolled/subtracted polygons, cross-product operands and
    # results fit twelve 3-vector f64 rows per face-vertex occurrence. Hidden
    # revival forms one metric-site distance vector and source-weight vectors.
    rows = len(data.face_vertices)
    images = len(publication.diagram.points)
    if publication.diagram.periodic_preparation is not None:
        images = len(publication.diagram.periodic_preparation.image_sites)
    with NativeHostStorageWorkspace(scope) as scratch:
        scratch.set_bound(
            24 * 256 + 288 * rows + 64 * images + 64 * len(state.points)
        )
        return _update_polyhedral_weights_active(
            publication, state, measures, carrier, schedule_,
            scope=scope, budget=budget,
        )


def _update_polyhedral_weights_active(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    carrier: _PolyhedralCarrier,
    schedule_: NativePolyhedralSchedule,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralWeightUpdate:
    diagram, recovery = publication.diagram, carrier.recovery
    values, targets = state.weights, state.targets
    achieved, empty, work_units = measures.achieved, measures.empty, measures.work_units
    # Sparse diagonal Jacobian: dV_i/dw_i = sum A_ij/(2|s_i-s_j|).
    scope.charge(work=values.size)
    diagonal = np.zeros_like(values)
    data = diagram.construction
    remaining_work = schedule_.maximum_work_units - work_units
    before = budget.work_units
    with _polyhedral_coordinate_phase(
        remaining_work,
        schedule_.maximum_scratch_bytes,
        scope=scope,
        budget=budget,
    ):
        metric_sites, metric_owners = _numerical_power_sites(
            diagram, scope=scope, budget=budget
        )
    work_units += budget.work_units - before
    scope.charge(work=4 * len(data.face_cells))
    for f, (left, right) in enumerate(
        zip(data.face_cells[:, 0], data.face_cells[:, 1], strict=True)
    ):
        if right < 0:
            continue
        i, j = (
            int(diagram.cell_original_sites[left]),
            int(diagram.cell_original_sites[right]),
        )
        if i == j:
            continue
        a, b = data.face_offsets[f : f + 2]
        # Gather/roll/subtract/cross/reduce plus the two vector norms and
        # diagonal updates. Admission precedes all polygon temporaries.
        scope.charge(work=32 * int(b - a) + 32)
        polygon = data.vertices[data.face_vertices[a:b]]
        area_vector = (
            np.sum(
                np.cross(
                    polygon - polygon[0],
                    np.roll(polygon, -1, axis=0) - polygon[0],
                ),
                axis=0,
            )
            / 2
        )
        derivative = np.linalg.norm(area_vector) / (
            2
            * np.linalg.norm(
                metric_sites[data.cell_sites[left]] - metric_sites[data.cell_sites[right]]
            )
        )
        diagonal[i] += derivative
        diagonal[j] += derivative
    if empty.size:
        revived = _revive_hidden_polyhedral_sites(
            diagram, recovery, values, targets, empty, metric_sites, metric_owners,
            scope=scope,
        )
        return _PolyhedralWeightUpdate(values=revived.values, work_units=work_units)
    scope.charge(work=2 * diagonal.size)
    if np.any(diagonal <= 0):
        scope.charge(work=3 * diagonal.size)
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.WEIGHT_JACOBIAN_UNRESOLVED,
            entities=tuple(map(int, np.flatnonzero(diagonal <= 0))),
        )
    scope.charge(work=5 * values.size)
    values += 0.5 * (targets - achieved) / diagonal
    values -= values[0]
    return _PolyhedralWeightUpdate(values=values, work_units=work_units)


def _insert_polyhedral_sites(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    *,
    scope: NativeExecutionScope,
) -> _PolyhedralSites:
    added, total = len(measures.large), len(state.points) + len(measures.large)
    connectivity = require_polyhedral_connectivity(publication.mesh)
    corners = connectivity.cell_vertex_values.size
    vertices = publication.mesh.coordinates.size
    # Four output banks: xyz f64 sites, f64 weights, i64 parents, i8 splits.
    retained = 4 * 256 + 41 * total
    # Vertex/CSR host views, one gathered cell bank, centers, selected source
    # rows, concatenate inputs and np.unique(axis=0)'s record/sort/mask banks.
    temporary = 20 * 256 + 8 * vertices + 16 * corners + 96 * total + 40 * added
    with _PolyhedralTopologyStorage(scope) as storage:
        storage.workspace.set_bound(retained + temporary)
        # CSR/coordinate copies, each selected corner's gather/reduction,
        # concatenation, row sorting and ancestry gather/output passes.
        scope.charge(
            work=vertices + 7 * corners + len(connectivity.cell_vertex_offsets)
            + 3 * total * (total.bit_length() + 2) + 12 * total + 4 * added
        )
        result = _insert_polyhedral_sites_active(publication, state, measures, complex_)
        storage.retain(
            retained, result.points, result.weights, result.parents, result.split,
            record_count_upper=8,
        )
        return result


def _insert_polyhedral_sites_active(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    complex_: plc_source_owner.PiecewiseLinearComplex,
) -> _PolyhedralSites:
    mesh, diagram, order = publication.mesh, publication.diagram, publication.order
    site_array, values, parents, split = (
        state.points,
        state.weights,
        state.parents,
        state.split,
    )
    preserve_vertices, large = state.preserve_vertices, measures.large
    c = require_polyhedral_connectivity(mesh)
    offsets, vertices = (
        np.asarray(c.cell_vertex_offsets),
        np.asarray(c.cell_vertex_values),
    )
    coordinates = np.asarray(mesh.coordinates, dtype=mesh.coordinates.dtype)
    centers = np.asarray(
        [
            np.mean(
                coordinates[vertices[offsets[k] : offsets[k + 1]]],
                axis=0,
            )
            for k in large
        ],
        dtype=np.float64,
    )
    # A certified vertex-mean star center lies inside the restricted cell.
    new_sites = np.concatenate((site_array, centers))
    if np.unique(new_sites, axis=0).shape[0] != new_sites.shape[0]:
        # Failure evidence uses only the already admitted selected-order bank.
        raise PolyhedralGenerationError(
            PolyhedralGenerationReason.SITE_INSERTION_STALLED,
            entities=tuple(map(int, order[large])),
        )
    source_sites = diagram.cell_original_sites[order[large]]
    site_array = new_sites
    values = np.concatenate((values, values[source_sites]))
    parents = np.concatenate((parents, source_sites))
    split = np.concatenate(
        (
            split,
            np.full(
                large.size,
                bool(complex_.segments.shape[0]) or preserve_vertices,
                dtype=np.int8,
            ),
        )
    )
    return _PolyhedralSites(
        points=site_array,
        weights=values,
        protected=state.protected,
        targets=state.targets,
        split=split,
        parents=parents,
        preserve_vertices=preserve_vertices,
    )


def _relocate_polyhedral_sites(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    schedule_: NativePolyhedralSchedule,
    work_units: int,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralRelocation:
    # multiplicity, inverse permutation and its arange input, plus a 3-vector
    # centroid. The returned coordinates alias the existing admitted state.
    with NativeHostStorageWorkspace(scope) as scratch:
        scratch.set_bound(
            4 * 256 + 8 * len(state.points) + 16 * len(publication.order) + 24
        )
        return _relocate_polyhedral_sites_active(
            publication, state, measures, schedule_, work_units,
            scope=scope, budget=budget,
        )


def _relocate_polyhedral_sites_active(
    publication: _PolyhedralPublication,
    state: _PolyhedralSites,
    measures: _PolyhedralMeasures,
    schedule_: NativePolyhedralSchedule,
    work_units: int,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralRelocation:
    diagram, mesh, geometry, order = (
        publication.diagram,
        publication.mesh,
        publication.geometry,
        publication.order,
    )
    site_array, protected, split, achieved = (
        state.points,
        state.protected,
        state.split,
        measures.achieved,
    )
    # Restrict relocation to single unsplit star-visible components;
    # averaging disconnected components can move a site into a hole.
    data = diagram.construction
    # Bincount zeroing/accumulation, inverse permutation initialization and
    # scatter, then each eligibility check and possible centroid/write. Keep
    # this host admission outside the coordinate ticket's reserved allowance.
    scope.charge(
        work=achieved.size + 10 * len(diagram.cell_original_sites) + 2 * order.size
    )
    multiplicity = np.bincount(diagram.cell_original_sites, minlength=achieved.size)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size, dtype=np.int64)
    remaining_work = schedule_.maximum_work_units - work_units
    before = budget.work_units
    with _polyhedral_coordinate_phase(
        remaining_work,
        schedule_.maximum_scratch_bytes,
        scope=scope,
        budget=budget,
    ):
        for cell, site in enumerate(diagram.cell_original_sites):
            if int(site) not in protected and multiplicity[site] == 1 and not split[site]:
                centroid = data.cell_moments[cell] / data.cell_volumes[cell]
                if _centroid_in_kernel(
                    mesh,
                    geometry,
                    int(inverse_order[cell]),
                    centroid,
                    scope=scope,
                    budget=budget,
                ):
                    site_array[site] = centroid
    work_units += budget.work_units - before
    return _PolyhedralRelocation(points=site_array, work_units=work_units)


def _polyhedral_finalization_storage(
    faces: int, cells: int, sites: int, /
) -> tuple[int, int]:
    # Retained rows: face ancestry/interfaces/sheets (<=24F), cell
    # order/region/diameter (20C), and site achieved volumes (8S).
    retained = 8 * 256 + 24 * faces + 20 * cells + 8 * sites
    # Region pairs, constraint/comparison masks and advanced-index inputs fit
    # the remaining per-face rows. Residual evaluation owns three f64 site
    # banks. The old-diameter lookup has a <=160-byte int64 key, <=264-byte
    # NumPy scalar and 64-byte mapping slot per cell (488 <= 512).
    peak = 32 * 256 + 64 * faces + 544 * cells + 32 * sites
    return peak, retained


def _finalize_polyhedral_periodicity(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    publication: _PolyhedralPublication,
    measures: _PolyhedralMeasures,
    state: _PolyhedralSites,
    carrier: _PolyhedralCarrier,
    schedule_: NativePolyhedralSchedule,
    periodic_values: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    periodic_group: PeriodicIsometryGroup | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None,
) -> _PolyhedralFinalization:
    connectivity = require_polyhedral_connectivity(publication.mesh)
    peak, _ = _polyhedral_finalization_storage(
        connectivity.face_count, connectivity.cell_count, len(state.points)
    )
    with _PolyhedralTopologyStorage(scope) as storage:
        storage.workspace.set_bound(peak)
        result = _finalize_polyhedral_periodicity_active(
            complex_, publication, measures, state, carrier, schedule_,
            periodic_values, periodic_group, scope=scope, budget=budget,
            host_storage=storage,
            limits=limits,
        )
        connectivity = require_polyhedral_connectivity(result.mesh)
        _, retained = _polyhedral_finalization_storage(
            connectivity.face_count, connectivity.cell_count, len(state.points)
        )
        storage.retain(
            retained, result.order, result.regions, result.face_recovery_triangles,
            result.face_facets, result.interface_faces, result.sheet_faces,
            result.diameters, result.achieved, record_count_upper=16,
        )
        return result


def _finalize_polyhedral_periodicity_active(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    publication: _PolyhedralPublication,
    measures: _PolyhedralMeasures,
    state: _PolyhedralSites,
    carrier: _PolyhedralCarrier,
    schedule_: NativePolyhedralSchedule,
    periodic_values: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    periodic_group: PeriodicIsometryGroup | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    host_storage: _PolyhedralTopologyStorage,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None,
) -> _PolyhedralFinalization:
    diagram, mesh, geometry, order, face_order = (
        publication.diagram,
        publication.mesh,
        publication.geometry,
        publication.order,
        publication.face_order,
    )
    recovery, site_array, targets = carrier.recovery, state.points, state.targets
    diameters, achieved, residual, work_units = (
        measures.diameters,
        measures.achieved,
        measures.residual,
        measures.work_units,
    )
    data = diagram.construction
    scope.charge(work=len(order) + 7 * len(face_order))
    regions = data.cell_regions[order]
    face_recovery_triangles = data.face_facets[face_order]
    face_facets = np.full(face_recovery_triangles.size, -1, dtype=np.int32)
    constrained_triangles = face_recovery_triangles >= 0
    face_facets[constrained_triangles] = recovery.face_sources[
        face_recovery_triangles[constrained_triangles]
    ]
    seam_fragments: tuple[PeriodicPolyhedralSeamFragment, ...] = ()
    assembly = None
    if periodic_group is not None:
        scope.charge(work=2 * len(order))
        old_bounds = dict(zip(map(int, order), diameters, strict=True))
        remaining = schedule_.maximum_work_units - work_units
        before = budget.work_units
        with _polyhedral_coordinate_phase(
            remaining, schedule_.maximum_scratch_bytes, scope=scope, budget=budget
        ):
            seam_fragments = prepare_periodic_polyhedral_seam_fragments(
                mesh,
                geometry,
                face_facets,
                periodic_values,
                maximum_work=remaining,
                maximum_scratch_bytes=schedule_.maximum_scratch_bytes,
                budget=budget,
            )
            assembly = _reconcile_periodic_polyhedral(
                mesh,
                geometry,
                diagram,
                seam_fragments,
                periodic_values,
                periodic_group,
                schedule_.maximum_vertices,
                scope=scope,
                budget=budget,
                limits=limits,
            )
        connectivity = require_polyhedral_connectivity(assembly.mesh)
        expanded_peak, _ = _polyhedral_finalization_storage(
            connectivity.face_count, connectivity.cell_count, len(site_array)
        )
        # Resize the outer token between native targets, never under a foreign
        # batch owner. Authored ancestry and old diameters remain live.
        host_storage.workspace.set_bound(
            host_storage.workspace.bound + expanded_peak
        )
        with _polyhedral_coordinate_phase(
            remaining - (budget.work_units - before),
            schedule_.maximum_scratch_bytes,
            scope=scope,
            budget=budget,
        ):
            validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                assembly.geometry, mesh=assembly.mesh, budget=budget
            )
            budget.reserve(3 * len(validity.status))
            bad = np.flatnonzero(
                np.asarray(validity.status)
                != int(
                    _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
                )
            )
            if bad.size:
                budget.reserve(bad.size)
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.PERIODIC_PIECE_GEOMETRY_UNRESOLVED,
                    entities=tuple(map(int, bad)),
                )
            mesh, geometry = assembly.mesh, assembly.geometry
            budget.reserve(
                2 * len(assembly.parent_faces)
                + 4 * require_polyhedral_connectivity(mesh).cell_count
            )
            face_facets = face_facets[assembly.parent_faces]
            face_recovery_triangles = face_recovery_triangles[assembly.parent_faces]
            order = np.asarray(
                require_polyhedral_connectivity(mesh).cell_global_ids, dtype=np.int64
            )
            regions = data.cell_regions[order]
            diameters = np.asarray(
                [old_bounds[int(cell)] for cell in order], dtype=np.float64
            )
            exact_volumes, _, _ = mapped_measure_owner.certified_cell_measures(
                mesh, geometry, maximum_work=remaining, budget=budget
            )
            budget.reserve(2 * len(order) + len(site_array) + 4 * targets.size)
            achieved = np.bincount(
                diagram.cell_original_sites[order],
                weights=exact_volumes,
                minlength=site_array.shape[0],
            )
            residual = (
                float(np.max(np.abs(achieved - targets) / targets))
                if targets.size
                else 0.0
            )
            if targets.size and residual > schedule_.relative_volume_tolerance:
                raise PolyhedralGenerationError(
                    PolyhedralGenerationReason.PERIODIC_VOLUME_OPTIMIZATION_UNRESOLVED,
                    achieved={"relative_volume_residual": residual},
                )
        work_units += budget.work_units - before
    scope.charge(work=20 * face_facets.size)
    pairs = np.full((face_facets.size, 2), -1, dtype=np.int64)
    constrained = face_facets >= 0
    pairs[constrained] = complex_.facet_regions[face_facets[constrained]]
    interfaces = np.flatnonzero(
        constrained
        & (pairs[:, 0] >= 0)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 0] != pairs[:, 1])
    )
    sheets = np.flatnonzero(constrained & (pairs[:, 0] == pairs[:, 1]))
    return _PolyhedralFinalization(
        mesh=mesh,
        geometry=geometry,
        order=order,
        regions=regions,
        face_recovery_triangles=face_recovery_triangles,
        face_facets=face_facets,
        interface_faces=interfaces,
        sheet_faces=sheets,
        diameters=diameters,
        achieved=achieved,
        residual=residual,
        seam_fragments=seam_fragments,
        assembly=assembly,
        work_units=work_units,
    )


def _certify_polyhedral_features(
    finalization: _PolyhedralFinalization,
    carrier: _PolyhedralCarrier,
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    schedule_: NativePolyhedralSchedule,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PolyhedralFeatures:
    mesh, geometry, assembly = (
        finalization.mesh,
        finalization.geometry,
        finalization.assembly,
    )
    recovery, work_units = carrier.recovery, finalization.work_units
    with scope.ticket_view() as native_view, native_view.native_target():
        feature_source_edges = _source_feature_edges(complex_, recovery, scope=scope)
    remaining_work = schedule_.maximum_work_units - work_units
    before = budget.work_units
    with _polyhedral_coordinate_phase(
        remaining_work, schedule_.maximum_scratch_bytes, scope=scope, budget=budget
    ):
        ancestry = _feature_ancestry(
            mesh,
            geometry,
            recovery,
            diagram,
            feature_source_edges,
            remaining_work,
            scope=scope,
            vertex_carriers=None if assembly is None else assembly.vertex_carriers,
            budget=budget,
        )
    feature_sources, feature_distances, feature_gaps = (
        ancestry.sources,
        ancestry.distances,
        ancestry.gaps,
    )
    work_units += budget.work_units - before
    with NativeHostStorageWorkspace(scope) as storage:
        # Sum, outward-rounded sum and where result can coexist; allow eight
        # float/index banks, two masks and a Python (int, reference) per source
        # for the failure entities. All disappear before the output returns.
        storage.set_bound(
            (8 * 8 + 2 + 36 + 8) * len(feature_source_edges) + 256 * 16
        )
        # Sum, zero test, outward rounding, where, tolerance test and the
        # nonzero scan visit each source feature edge once.
        scope.charge(work=6 * len(feature_source_edges))
        combined = feature_distances + feature_gaps
        combined = np.where(combined == 0, 0.0, np.nextafter(combined, np.inf))
        bad_features = np.flatnonzero(combined > schedule_.feature_tolerance)
        if bad_features.size:
            raise PolyhedralGenerationError(
                PolyhedralGenerationReason.FEATURE_COVERAGE_UNRESOLVED,
                entities=tuple(map(int, bad_features)),
                achieved=feature_gaps[bad_features],
            )
        del combined, bad_features
    return _PolyhedralFeatures(
        sources=feature_sources,
        source_edges=feature_source_edges,
        distances=feature_distances,
        gaps=feature_gaps,
        work_units=work_units,
    )


def _polyhedral_component_lineage(
    finalization: _PolyhedralFinalization,
    diagram: _dependency_phydrax_geometry__triangulation.RestrictedPowerDiagram,
    *,
    scope: NativeExecutionScope,
) -> _PolyhedralLineage:
    mesh, order, assembly = finalization.mesh, finalization.order, finalization.assembly
    data = diagram.construction
    count = data.cell_sites.size
    # NumPy unique/inverse and selection coexist with the preceding iteration's
    # selected/inverse banks. Sixteen count-sized int64 banks cover those
    # inputs, sorting permutations/scratch, masks, packed unique values and
    # inverse results; bool/int32 banks are bounded at the int64 width. Face
    # owners require at most one int32 bank when copied from the mesh device.
    with NativeHostStorageWorkspace(scope) as scratch:
        scratch.set_bound(
            16 * (256 + 8 * count) + 256 + 4 * data.face_cells.shape[0]
        )
        # Consumer splitting is not a new connected site component. Rejoin
        # through reciprocal unconstrained same-site faces and periodic seams.
        parent = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count,), np.int64, scope=scope
        )
        for cell in range(count):
            scope.charge(work=1)
            parent[cell] = cell

        def root(cell: int) -> int:
            while parent[cell] != cell:
                scope.charge(work=1)
                parent[cell] = parent[parent[cell]]
                cell = parent[cell]
            return int(cell)

        for face in range(data.face_cells.shape[0]):
            scope.charge(work=1)
            left, right = int(data.face_cells[face, 0]), int(data.face_cells[face, 1])
            if (
                right >= 0
                and data.face_facets[face] < 0
                and diagram.cell_original_sites[left] == diagram.cell_original_sites[right]
            ):
                a, b = root(left), root(right)
                parent[max(a, b)] = min(a, b)
        if assembly is not None:
            connectivity = require_polyhedral_connectivity(mesh)
            owners = np.asarray(connectivity.face_owner)
            for source_face, target_face, _ in assembly.face_pairs:
                scope.charge(work=1)
                left, right = (
                    int(order[owners[source_face]]),
                    int(order[owners[target_face]]),
                )
                if diagram.cell_original_sites[left] == diagram.cell_original_sites[right]:
                    a, b = root(left), root(right)
                    parent[max(a, b)] = min(a, b)
        roots = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count,), np.int64, scope=scope
        )
        for cell in range(count):
            scope.charge(work=1)
            roots[cell] = root(cell)
        components = _dependency_phydrax_meshing__host_banks.family_host_array(
            diagram.cell_original_sites.shape,
            diagram.cell_original_sites.dtype,
            scope=scope,
        )
        scope.charge(work=len(diagram.cell_original_sites))
        components.fill(0)
        site_count = len(diagram.cell_original_sites)
        scope.charge(work=site_count * (site_count.bit_length() + 1))
        for site in np.unique(diagram.cell_original_sites):
            scope.charge(work=2 * site_count)
            selected = np.flatnonzero(diagram.cell_original_sites == site)
            selected_count = len(selected)
            # Gather, sorted unique/inverse construction, and scatter all
            # precede the next site's mask while both banks remain admitted.
            scope.charge(work=selected_count * (selected_count.bit_length() + 3))
            _, inverse = np.unique(roots[selected], return_inverse=True)
            components[selected] = inverse
        return _PolyhedralLineage(components=components)


def _assemble_polyhedral_construction(
    carrier: _PolyhedralCarrier,
    state: _PolyhedralSites,
    publication: _PolyhedralPublication,
    finalization: _PolyhedralFinalization,
    features: _PolyhedralFeatures,
    periodic_values: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    refinement_steps: int,
    lloyd_steps: int,
    *,
    scope: NativeExecutionScope,
) -> PolyhedralConstruction:
    # Frozen ancestry payloads use family_host_array and are charged by their
    # own native allocation lifetime. Admit their Python wrappers/record and
    # the two cell-order advanced-index inputs before any freeze. Fingerprint
    # streaming uses at most two 64-KiB noncontiguous byte buffers.
    retained = 24 * 256 + 8 * 160 + 512
    temporary = 2 * 65536 + 64 * 256 + 16 * len(finalization.order)
    with _PolyhedralTopologyStorage(scope) as storage:
        storage.workspace.set_bound(retained + temporary)
        result = _assemble_polyhedral_construction_active(
            carrier, state, publication, finalization, features,
            periodic_values, refinement_steps, lloyd_steps, scope=scope,
        )
        storage.retain(
            retained, result.cell_regions, result.cell_sites, result.face_facets,
            result.face_recovery_triangles, result.interface_faces, result.sheet_faces,
            result.feature_edge_sources, result.feature_source_edges,
            result.feature_maximum_distance, result.feature_coverage_gap,
            result.site_parents, result.component_ids, result.split_sites,
            result.requested_site_volumes, result.achieved_site_volumes,
            *((result.seam_face_pairs,) if result.seam_face_pairs is not None else ()),
            *((result.seam_vertex_pairs,) if result.seam_vertex_pairs is not None else ()),
            record_count_upper=2 * (
                15 + int(result.seam_face_pairs is not None)
                + int(result.seam_vertex_pairs is not None)
            ),
        )
        return result


def _assemble_polyhedral_construction_active(
    carrier: _PolyhedralCarrier,
    state: _PolyhedralSites,
    publication: _PolyhedralPublication,
    finalization: _PolyhedralFinalization,
    features: _PolyhedralFeatures,
    periodic_values: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
    refinement_steps: int,
    lloyd_steps: int,
    *,
    scope: NativeExecutionScope,
) -> PolyhedralConstruction:
    diagram, domain, recovery = publication.diagram, carrier.domain, carrier.recovery
    data = diagram.construction
    mesh, geometry, order = finalization.mesh, finalization.geometry, finalization.order
    face_facets, face_recovery_triangles, regions = (
        finalization.face_facets,
        finalization.face_recovery_triangles,
        finalization.regions,
    )
    diameters, achieved, residual = (
        finalization.diameters,
        finalization.achieved,
        finalization.residual,
    )
    assembly, seam_fragments = finalization.assembly, finalization.seam_fragments
    site_array, values, parents, split, targets = (
        state.points,
        state.weights,
        state.parents,
        state.split,
        state.targets,
    )
    feature_sources, feature_source_edges, feature_distances, feature_gaps = (
        features.sources,
        features.source_edges,
        features.distances,
        features.gaps,
    )
    work_units = features.work_units
    components = _polyhedral_component_lineage(
        finalization, diagram, scope=scope
    ).components
    interfaces, sheets = finalization.interface_faces, finalization.sheet_faces
    # Streamed host reads and hashes of every identity bank, as in
    # _cell_mesh._hash_work: the geometry identity's element/route/coordinate
    # arrays, the site banks and the four piece banks; then the two cell-order
    # gathers frozen below (each frozen copy charges itself).
    scope.charge(
        work=_geometry_identity_work(geometry)
        + 2 * (
            site_array.size + values.size + data.piece_sites.size
            + data.piece_tets.size + data.piece_cells.size + data.piece_volumes.size
            + order.size
        )
    )
    identity = canonical_fingerprint(
        {
            "kind": "native-polyhedral-construction",
            "source": domain.domain_id,
            "mesh": mesh.mesh_id,
            "geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                geometry
            ),
            "sites": array_tree_fingerprint((site_array, values)),
            "pieces": array_tree_fingerprint(
                (
                    data.piece_sites,
                    data.piece_tets,
                    data.piece_cells,
                    data.piece_volumes,
                )
            ),
        }
    )
    return PolyhedralConstruction(
        mesh=mesh,
        geometry=geometry,
        domain=domain,
        recovery=recovery,
        diagram=diagram,
        cell_regions=_freeze(regions, scope=scope),
        cell_sites=_freeze(diagram.cell_original_sites[order], scope=scope),
        face_facets=_freeze(face_facets, scope=scope),
        face_recovery_triangles=_freeze(face_recovery_triangles, scope=scope),
        interface_faces=_freeze(interfaces, scope=scope),
        sheet_faces=_freeze(sheets, scope=scope),
        feature_edge_sources=_freeze(feature_sources, scope=scope),
        feature_source_edges=_freeze(feature_source_edges, scope=scope),
        feature_maximum_distance=_freeze(feature_distances, scope=scope),
        feature_coverage_gap=_freeze(feature_gaps, scope=scope),
        site_parents=_freeze(parents, scope=scope),
        component_ids=_freeze(components[order], scope=scope),
        split_sites=_freeze(split, scope=scope),
        requested_site_volumes=_freeze(targets, scope=scope),
        achieved_site_volumes=_freeze(achieved, scope=scope),
        refinement_steps=refinement_steps,
        lloyd_steps=lloyd_steps,
        optimization_relative_residual=residual,
        maximum_cell_diameter=float(np.max(diameters)),
        construction_id=identity,
        work_units=work_units,
        periodic_constraints=periodic_values,
        seam_fragments=seam_fragments,
        seam_face_pairs=None
        if assembly is None
        else _freeze(assembly.face_pairs, scope=scope),
        seam_vertex_pairs=None
        if assembly is None
        else _freeze(assembly.vertex_pairs, scope=scope),
    )


@checked
def generate_polyhedral_volume(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    /,
    *,
    sites: ConvertibleToArray | None = None,
    weights: ConvertibleToArray | None = None,
    schedule: NativePolyhedralSchedule | None = None,
    source_id: str = "native-polyhedral",
    seeds: ConvertibleToArray | None = None,
    seed_regions: ConvertibleToArray | None = None,
    protected_vertices: tuple[int, ...] = (),
    periodic_constraints: Sequence[periodic_constraint_owner.PeriodicConstraint] = (),
    record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None = None,
) -> PolyhedralConstruction:
    """Generate arbitrary nonconvex/material PLC-restricted power polyhedra.

    The provider separately certifies source coordinates, global embedding,
    and declared material-domain coverage before assigning accepted status.
    A supplied ``budget`` is the caller's request ledger: construction stays
    within the schedule's allowance on it, and its retained witnesses remain
    available to later phases that share the ledger.
    """
    with ExitStack() as request:
        schedule_ = NativePolyhedralSchedule() if schedule is None else schedule

        def reserve_retained_storage(_bytes_upper: int) -> NativeScopeTicket:
            return scope.ticket_view()

        periodic_storage = request.enter_context(NativeHostStorageWorkspace(scope))
        periodic_storage.set_bound(256 + 16 * len(periodic_constraints))
        scope.charge(work=len(periodic_constraints))
        periodic_values = tuple(periodic_constraints)
        if budget is not None:
            # The schedule allowance is relative to the ledger's completed work
            # and live storage, exactly as on a construction-owned ledger.
            request.enter_context(
                budget.bound_stage(
                    schedule_.maximum_work_units,
                    budget.retained_basis_bytes
                    + budget.temporary_bytes_upper
                    + schedule_.maximum_scratch_bytes,
                )
            )
        ticket = _polyhedral_coordinate_ticket(
            schedule_.maximum_work_units if periodic_values else 0,
            schedule_.maximum_scratch_bytes,
            scope=scope,
        )
        if budget is None:
            budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=schedule_.maximum_work_units,
                maximum_memory_bytes=schedule_.maximum_scratch_bytes,
            )
            with ticket, budget.activate():
                periodic_group = polyhedral_periodic_group(periodic_values, budget=budget)
        else:
            with budget.bind_ticket(ticket):
                periodic_group = polyhedral_periodic_group(periodic_values, budget=budget)
        if budget.storage_ticket is None:
            request.enter_context(budget.retained_storage(reserve_retained_storage))
        carrier = _prepare_polyhedral_carrier(
            complex_,
            sites,
            weights,
            seeds,
            seed_regions,
            protected_vertices,
            source_id,
            schedule_,
            scope=scope,
            budget=budget,
        )
        state = carrier.sites
        refinement_steps = 0
        lloyd_steps = 0
        work_units = carrier.work_units
        loop_storage = request.enter_context(NativeHostStorageWorkspace(scope))
        offending = None
        while True:
            # Discard the previous loop's selection before releasing its
            # temporary bound; retained publication/state banks own themselves.
            offending = None
            loop_storage.set_bound(0)
            publication = _construct_polyhedral_publication(
                carrier,
                state,
                schedule_,
                work_units,
                complex_,
                periodic_values,
                periodic_group,
                record_native_phase,
                scope=scope,
                budget=budget,
                limits=limits,
            )
            work_units = publication.work_units
            # np.unique's selected/site/sort rows and unresolved/split masks.
            loop_storage.set_bound(16 * 256 + 64 * len(publication.order))
            scope.charge(work=len(publication.unresolved))
            if np.any(publication.unresolved):
                scope.charge(work=len(publication.order))
                selected = publication.order[publication.unresolved]
                scope.charge(work=len(selected))
                selected_sites = publication.diagram.cell_original_sites[selected]
                scope.charge(work=len(selected) * (1 + len(selected).bit_length()))
                offending = np.unique(selected_sites)
                del selected_sites
                scope.charge(work=2 * len(offending))
                if np.any(state.split[offending]):
                    scope.charge(work=len(selected))
                    raise PolyhedralGenerationError(
                        PolyhedralGenerationReason.PIECE_GEOMETRY_UNRESOLVED,
                        entities=tuple(map(int, selected)),
                    )
                del selected
                scope.charge(work=len(offending))
                state.split[offending] = 1
                continue
            measures = _measure_polyhedral_publication(
                publication,
                state,
                schedule_,
                scope=scope,
                budget=budget,
            )
            work_units = measures.work_units
            refinement = _plan_polyhedral_refinement(
                publication, state, measures, schedule_, refinement_steps, lloyd_steps,
                scope=scope,
            )
            if refinement.finished:
                break
            if refinement.update_weights:
                update = _update_polyhedral_weights(
                    publication,
                    state,
                    measures,
                    carrier,
                    schedule_,
                    scope=scope,
                    budget=budget,
                )
                work_units = update.work_units
                if measures.empty.size:
                    refinement_steps += 1
                    continue
            if refinement.insert_sites:
                state = _insert_polyhedral_sites(
                    publication, state, measures, complex_, scope=scope
                )
            if refinement.relocate_sites:
                relocation = _relocate_polyhedral_sites(
                    publication,
                    state,
                    measures,
                    schedule_,
                    work_units,
                    scope=scope,
                    budget=budget,
                )
                work_units = relocation.work_units
                lloyd_steps += 1
            refinement_steps += 1
        finalization = _finalize_polyhedral_periodicity(
            complex_,
            publication,
            measures,
            state,
            carrier,
            schedule_,
            periodic_values,
            periodic_group,
            scope=scope,
            budget=budget,
            limits=limits,
        )
        features = _certify_polyhedral_features(
            finalization,
            carrier,
            publication.diagram,
            complex_,
            schedule_,
            scope=scope,
            budget=budget,
        )
        return _assemble_polyhedral_construction(
            carrier,
            state,
            publication,
            finalization,
            features,
            periodic_values,
            refinement_steps,
            lloyd_steps,
            scope=scope,
        )


__all__ = [
    "NativePolyhedralSchedule",
    "PolyhedralConstruction",
    "PolyhedralGenerationError",
    "PeriodicPolyhedralSeamFragment",
    "prepare_periodic_polyhedral_seam_fragments",
    "generate_polyhedral_volume",
]
