#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native periodic construction and publication on the canonical quotient carrier.

Point-orbit sources use certified translational Delaunay construction. Explicit
lifted simplex domains additionally admit finite rotational boundary isometries;
they are refined by orbit bisection, not advertised as rotational Delaunay.
Material regions are represented source-cell assignments, never centroid guesses.
"""

from __future__ import annotations

from itertools import combinations
from time import monotonic
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import current_native_execution_budget, MeshcoreError, MeshcoreStatus
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import (
    CellGeometrySpec,
    CellMesh,
    periodic_orbit_measures,
    PeriodicCell,
)
from ...discretization._cell_geometry_transfer import is_affine_cell_geometry
from ...discretization._periodic_topology import _identification_id, PeriodicIsometryGroup
from ...geometry._mesh_certificates import _dyadic_integers
from ...geometry._triangulation import (
    PeriodicDelaunayTriangulation,
    PeriodicImageBudgetError,
)
from ...typing import checked
from .._association import (
    _target_dimension,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    PlcEntityClasses,
)
from .._audit import CellMeshAuditPolicy
from .._certification import MeshCertificationSchedule
from .._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from .._controls import FeatureKind, PeriodicConstraint
from .._lineage import MeshLineage
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._organization import MeshLabel, MeshZone, MeshZoneRole, RegionRole
from .._periodic import (
    _affine_periodic_source_support_boxes,
    _interiors_overlap,
    _periodic_overlap_image_exponents,
    _PeriodicEmbeddingResourceError,
    _PeriodicImageFrame,
    _prepare_periodic_image_frame,
    _require_periodic_topology,
    _simplex_entity_corners,
    certify_periodic_embedding,
    periodic_delaunay_mesh,
    periodic_edge_size_evidence,
    PeriodicPointOrbits,
    PeriodicQuotientEvidence,
    publish_periodic_simplices,
    refine_periodic_mesh,
)
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import SizeCompliancePolicy, SizeControlStrength, UniformSizeControl
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from .._volume_generation import _entity
from ._native_periodic_constraints import (
    bind_periodic_source_strata,
    periodic_source_seed_points,
    PeriodicConstructionImageBudgetError,
    PeriodicSourceStrata,
    PeriodicStratifiedTriangulation,
    prepare_periodic_source_strata,
    triangulate_periodic_source_strata,
)
from ._native_periodic_size import (
    periodic_reference_lifts,
    PERIODIC_SIZE_EVIDENCE_FIELDS,
    PeriodicSizeRecord,
    propose_periodic_site_relocation,
    quotient_edge_lengths,
)
from ._native_publication import (
    check_deadline,
    edge_growth_evidence,
    NativeCertificationRequest,
    publish_native_result,
    simplex_entity_limits,
    uniform_size_compliance,
)


@final
class NativePeriodicSource(StrictModule, NonTrainableState):
    """Revision-bound torus seed or represented periodic simplex domain.

    Source region IDs are the nonnegative entries in ``cell_regions`` (region
    zero for a point-orbit source). Edge and vertex IDs of a represented domain
    are its quotient entity positions, including distinct winding entities.
    The supplied domain is an actual scientific source, not a construction ghost
    tiling. Its material interface complex survives every refinement.
    """

    domain: PeriodicPointOrbits | CellMesh
    cell_regions: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: PeriodicPointOrbits | CellMesh,
        source_id: str,
        source_revision: str,
        /,
        *,
        cell_regions: ArrayLike | None = None,
    ) -> None:
        if isinstance(domain, PeriodicPointOrbits):
            dimension = domain.cell.ambient_dimension
            if cell_regions is not None:
                raise ValueError(
                    "Seed orbits have no represented material-cell partition."
                )
            regions = np.zeros(1, dtype=np.int32)
            identity = domain.orbits_id
        elif isinstance(domain, CellMesh):
            if domain.periodic_topology is None or len(domain.blocks) != 1:
                raise ValueError(
                    "A represented periodic domain needs one bound simplex block."
                )
            dimension = domain.ambient_dimension
            if domain.blocks[0].cell_kind != {2: "triangle", 3: "tetrahedron"}.get(
                dimension
            ):
                raise ValueError(
                    "Periodic source domains require full-dimensional simplices."
                )
            count = domain.blocks[0].cell_count
            regions = (
                np.zeros(count, dtype=np.int32)
                if cell_regions is None
                else np.asarray(cell_regions)
            )
            if (
                regions.shape != (count,)
                or not np.issubdtype(regions.dtype, np.integer)
                or np.any(regions < 0)
            ):
                raise ValueError(
                    "cell_regions must assign one nonnegative region ID per source cell."
                )
            identity = domain.mesh_id
        else:
            raise TypeError("domain must be PeriodicPointOrbits or a periodic CellMesh.")
        source, revision = str(source_id).strip(), str(source_revision).strip()
        if not source or not revision:
            raise ValueError("Source and revision identities must be nonempty.")
        if np.any(regions > np.iinfo(np.int64).max):
            raise ValueError("Periodic region IDs exceed the canonical integer domain.")
        regions = np.ascontiguousarray(regions, dtype=np.int64)
        regions.setflags(write=False)
        self.domain, self.cell_regions = domain, regions
        self.source_id, self.source_revision = source, revision
        self.ambient_dimension = dimension
        self.binding_id = canonical_fingerprint(
            {
                "kind": "native-periodic-source",
                "domain": identity,
                "source": source,
                "revision": revision,
                "regions": array_tree_fingerprint(regions),
            }
        )


def _scope(specification: SurfaceMeshingSpec | VolumeMeshingSpec) -> MeshingScope:
    return (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )


def periodic_support_issues(
    source: NativePeriodicSource, specification: SurfaceMeshingSpec | VolumeMeshingSpec, /
) -> list[str]:
    issues = []
    dimension = source.ambient_dimension
    family = {2: "triangle", 3: "tetrahedron"}[dimension]
    target = specification.target
    if (
        target.topological_dimension != dimension
        or target.ambient_dimension != dimension
        or target.geometry_order != 1
    ):
        issues.append("full-dimensional affine periodic simplices")
    if (
        set((*target.cell_families.required, *target.cell_families.preferred)) != {family}
        or target.cell_families.allow_mixed
    ):
        issues.append("the periodic simplex cell family")
    if not specification.size_controls or any(
        not isinstance(control, UniformSizeControl)
        for control in specification.size_controls
    ):
        issues.append("uniform periodic size controls")
    regions = np.unique(source.cell_regions)
    scope = _scope(specification)
    if (
        scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
    ):
        issues.append("the bound periodic source revision")
    if isinstance(specification, SurfaceMeshingSpec) and (
        scope.entity_dimension != dimension
        or not np.array_equal(np.asarray(scope.entity_ids), regions)
    ):
        issues.append("the complete represented periodic region scope")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        if control.scope.entity_dimension != dimension or not np.array_equal(
            np.asarray(control.scope.entity_ids), regions
        ):
            issues.append("size controls over all represented periodic regions")
    for feature in specification.protected_features:
        if not isinstance(source.domain, CellMesh) or feature.feature_kind not in (
            FeatureKind.CORNER,
            FeatureKind.CURVE,
            FeatureKind.SURFACE,
            FeatureKind.MATERIAL_INTERFACE,
        ):
            issues.append("represented periodic feature orbits")
        elif feature.scope.entity_dimension >= dimension or np.any(
            np.asarray(feature.scope.entity_ids)
            >= _require_periodic_topology(source.domain)
            .quotient.entities(feature.scope.entity_dimension)
            .count
        ):
            issues.append("feature scopes naming source quotient entities")
    for control in specification.region_controls:
        if not control.meshing_enabled or not np.all(
            np.isin(np.asarray(control.scope.entity_ids), regions)
        ):
            issues.append("enabled represented material regions")
    if isinstance(specification, VolumeMeshingSpec) and (
        specification.region_seeds or specification.hole_seeds
    ):
        issues.append(
            "regions and holes represented by the source quotient complex, not unbound seeds"
        )
    if specification.patch_controls or specification.layer_controls:
        issues.append("periodic patch/layer construction")
    # Identification is already part of the source topology. Additional boundary
    # constraints must not silently overwrite that scientific identity.
    if specification.periodic_constraints and (
        not isinstance(source.domain, CellMesh)
        or any(
            constraint.source_entity_ids is None
            for constraint in specification.periodic_constraints
        )
    ):
        issues.append(
            "explicit boundary-entity orbits of the represented periodic source"
        )
    return issues


def _bound_constraint_evidence(
    source: NativePeriodicSource, constraints: tuple[PeriodicConstraint, ...]
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Check explicit source-geometry seam pairs against the bound quotient."""

    requested, achieved = [], []
    if not constraints:
        return requested, achieved
    mesh = source.domain
    if not isinstance(mesh, CellMesh):
        raise ValueError("Bound seam constraints require a represented lifted source.")
    topology = _require_periodic_topology(mesh)
    for constraint in constraints:
        degree = constraint.source_scope.entity_dimension
        if degree >= mesh.topological_dimension:
            raise ValueError(
                "Periodic boundary constraints must name vertices, edges or faces."
            )
        entities = mesh.topology.entities(degree)
        identifiers = np.asarray(entities.entity_ids)
        lookup = {int(identifier): row for row, identifier in enumerate(identifiers)}
        try:
            first = np.asarray(
                [
                    lookup[int(value)]
                    for value in np.asarray(constraint.source_entity_ids)
                ],
                dtype=np.int64,
            )
            second = np.asarray(
                [
                    lookup[int(value)]
                    for value in np.asarray(constraint.target_scope.entity_ids)
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "A bound periodic constraint names an unknown lifted entity."
            ) from error
        orbit, orientations, _ = (np.asarray(value) for value in topology.orbits(degree))
        if np.any(orbit[first] != orbit[second]):
            raise ValueError(
                "A periodic constraint pairs different quotient entity orbits."
            )
        if degree and np.any(
            np.asarray(constraint.orientations)
            != orientations[first] * orientations[second]
        ):
            raise ValueError(
                "A periodic constraint contradicts the bound entity orientation."
            )
        maps = topology.orbit_isometries(degree)
        expected = maps[second] @ np.linalg.inv(maps[first])
        transform = np.asarray(constraint.transform)
        if transform.shape != expected.shape[1:]:
            raise ValueError("A periodic constraint has the wrong ambient dimension.")
        residual = float(np.max(np.abs(expected - transform)))
        if residual > constraint.tolerance:
            raise ValueError(
                "A periodic constraint contradicts the bound isometry cycle."
            )
        requested.append(
            (f"periodic:{constraint.constraint_id}:tolerance", constraint.tolerance)
        )
        achieved.extend(
            (
                (f"periodic:{constraint.constraint_id}:transform_residual", residual),
                (f"periodic:{constraint.constraint_id}:entity_pairs", float(first.size)),
            )
        )
    return requested, achieved


@final
class PreparedPeriodicDomain(StrictModule, NonTrainableState):
    source: NativePeriodicSource
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativePeriodicSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> None:
        issues = periodic_support_issues(source, specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION, "; ".join(issues)
            )
        self.source = source
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-domain",
                "source": source.binding_id,
                "specification": specification.specification_id,
            }
        )


def _feature_entity_mask(
    source: CellMesh, target: CellMesh, identifiers: np.ndarray, degree: int
) -> np.ndarray:
    """Recognize exact affine source-edge/face subdivisions on every lifted copy."""
    original = _require_periodic_topology(source)
    orbit = np.asarray(original.orbits(degree)[0])
    source_rows = _simplex_entity_corners(source, degree)
    target_rows = _simplex_entity_corners(target, degree)
    segments = np.asarray(source.coordinates)[source_rows[np.isin(orbit, identifiers)]]
    corners = np.asarray(target.coordinates)[target_rows]
    result = np.zeros(corners.shape[0], dtype=np.bool_)
    for segment in segments:
        # Refinement retains each parent's continuous lift. Check every displayed
        # source copy directly; nearest-image shifts lose long winding features.
        candidate = corners
        # A zero-deviation protected segment is an exact represented constraint.
        packed = np.concatenate(
            (candidate.reshape((-1, source.ambient_dimension)), segment)
        )
        integers, _ = _dyadic_integers(packed)
        corner_count = candidate.shape[0] * candidate.shape[1]
        test = integers[:corner_count].reshape(candidate.shape)
        original_corners = integers[corner_count : corner_count + segment.shape[0]]
        result |= _source_entity_contains(test, original_corners)
    return result


def _source_entity_contains(corners: np.ndarray, source: np.ndarray, /) -> np.ndarray:
    """Exact affine support predicate shared by feature and association publication."""
    if source.shape[0] == 1:
        return np.all(corners == source[0], axis=(1, 2))
    start, stop = source[:2]
    direction, delta = stop - start, corners - start
    if source.shape[0] == 2:
        squared = direction @ direction
        if squared == 0:
            return np.zeros(corners.shape[0], dtype=np.bool_)
        parameter = delta @ direction
        if source.shape[1] == 2:
            collinear = delta[..., 0] * direction[1] == delta[..., 1] * direction[0]
        else:
            collinear = np.all(
                np.stack(
                    (
                        delta[..., 1] * direction[2] - delta[..., 2] * direction[1],
                        delta[..., 2] * direction[0] - delta[..., 0] * direction[2],
                        delta[..., 0] * direction[1] - delta[..., 1] * direction[0],
                    ),
                    axis=-1,
                )
                == 0,
                axis=-1,
            )
        return np.all(collinear & (parameter >= 0) & (parameter <= squared), axis=1)
    second = source[2] - start
    normal = np.asarray(
        (
            direction[1] * second[2] - direction[2] * second[1],
            direction[2] * second[0] - direction[0] * second[2],
            direction[0] * second[1] - direction[1] * second[0],
        ),
        dtype=object,
    )
    aa, ab, bb = direction @ direction, direction @ second, second @ second
    denominator = aa * bb - ab * ab
    if denominator == 0:
        return np.zeros(corners.shape[0], dtype=np.bool_)
    pa, pb = delta @ direction, delta @ second
    first_parameter, second_parameter = pa * bb - pb * ab, pb * aa - pa * ab
    return np.all(
        (delta @ normal == 0)
        & (first_parameter >= 0)
        & (second_parameter >= 0)
        & (first_parameter + second_parameter <= denominator),
        axis=1,
    )


def _periodic_entity_rows(mesh: CellMesh, degree: int, /) -> np.ndarray:
    if degree == 0:
        return np.arange(mesh.coordinates.shape[0], dtype=np.int64)[:, None]
    if degree == mesh.topological_dimension:
        positions = {}
        for block in mesh.blocks:
            vertices = np.asarray(block.vertices, dtype=np.int64)
            for identifier, row in zip(
                np.asarray(block.global_ids), vertices, strict=True
            ):
                positions[int(identifier)] = row
        return np.asarray(
            [
                positions[int(identifier)]
                for identifier in np.asarray(mesh.entity_set(degree).entity_ids)
            ],
            dtype=np.int64,
        )
    return _simplex_entity_corners(mesh, degree)


def _periodic_region_incidence(
    mesh: CellMesh,
    regions: np.ndarray,
    /,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Material incidence of every actual quotient entity, including seam copies."""
    topology = _require_periodic_topology(mesh)
    cells = _periodic_entity_rows(mesh, mesh.topological_dimension)
    result: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(mesh.topological_dimension):
        rows = _periodic_entity_rows(mesh, degree)
        lookup = {tuple(sorted(row.tolist())): index for index, row in enumerate(rows)}
        orbit = np.asarray(topology.orbits(degree)[0])
        members: list[set[int]] = [
            set() for _ in range(topology.quotient.entities(degree).count)
        ]
        for corners, region in zip(cells, regions, strict=True):
            for subset in combinations(corners.tolist(), degree + 1):
                row = lookup[tuple(sorted(subset))]
                members[int(orbit[row])].add(int(region))
        result.append(tuple(tuple(sorted(members[int(index)])) for index in orbit))
    result.append(tuple((int(region),) for region in regions))
    return tuple(result)


def _periodic_source_strata(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> tuple[tuple[int, ...], ...]:
    domain = source.domain
    if not isinstance(domain, CellMesh):
        raise TypeError("Represented source strata require the authored CellMesh.")
    dimension = domain.topological_dimension
    topology = _require_periodic_topology(domain)
    block_ids = np.asarray(domain.blocks[0].global_ids)
    positions = {int(identifier): row for row, identifier in enumerate(block_ids)}
    regions = np.asarray(source.cell_regions)[
        [
            positions[int(identifier)]
            for identifier in np.asarray(domain.entity_set(dimension).entity_ids)
        ]
    ]
    incidence = _periodic_region_incidence(domain, regions)
    selected: list[set[int]] = [set() for _ in range(dimension)]
    interface = np.flatnonzero(
        np.asarray([len(values) > 1 for values in incidence[dimension - 1]])
    )
    selected[dimension - 1].update(
        np.asarray(topology.orbits(dimension - 1)[0])[interface].tolist()
    )
    boundary = topology.quotient.entities(dimension - 1).subset("boundary")
    selected[dimension - 1].update(np.flatnonzero(np.asarray(boundary.mask)).tolist())
    for feature in specification.protected_features:
        selected[feature.scope.entity_dimension].update(
            np.asarray(feature.scope.entity_ids, dtype=np.int64).tolist()
        )
    for degree in range(dimension - 1, 0, -1):
        orbit = np.asarray(topology.orbits(degree)[0])
        rows = _periodic_entity_rows(domain, degree)[
            np.isin(orbit, sorted(selected[degree]))
        ]
        for lower in range(degree):
            lookup = {
                tuple(sorted(row.tolist())): index
                for index, row in enumerate(_periodic_entity_rows(domain, lower))
            }
            lower_orbit = np.asarray(topology.orbits(lower)[0])
            for corners in rows:
                for subset in combinations(corners.tolist(), lower + 1):
                    selected[lower].add(int(lower_orbit[lookup[tuple(sorted(subset))]]))
    return tuple(tuple(sorted(values)) for values in selected)


class _PeriodicSupportAllowance:
    """One support phase borrows the original native ledger without renewing it."""

    def __init__(self, limits: MeshingLimits, /) -> None:
        self.limits, self.work = limits, 0
        self.budget = current_native_execution_budget()
        self.started = monotonic()

    def spend(self, count: int, /) -> None:
        if self.work + count > self.limits.maximum_work_units:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Exact periodic source support exceeds its original work allowance.",
                requested=(("maximum_work_units", self.limits.maximum_work_units),),
                achieved=(
                    ("source_support_work", self.work),
                    ("source_support_request", count),
                ),
            )
        if self.work + count > self.limits.maximum_geometry_queries:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Exact periodic source support exceeds its original geometry-query allowance.",
                requested=(
                    ("maximum_geometry_queries", self.limits.maximum_geometry_queries),
                ),
                achieved=(("source_support_query_bound", self.work + count),),
            )
        check_deadline(self.started, self.limits, MeshingStageKind.GEOMETRY_ASSOCIATION)
        if self.budget is not None:
            self.budget.admit_work_bound(count)
            from ...discretization._coordinate_enclosure import _COORDINATE_BUDGET

            ledger = _COORDINATE_BUDGET.get()
            if ledger is not None:
                ledger.admit_work_bound(count)
            self.budget.charge(work=count, geometry_queries=count)
        self.work += count

    def scratch(self, bound: int, /) -> None:
        available = self.limits.maximum_scratch_bytes
        if self.budget is not None:
            available = min(available, self.budget.remaining().remaining_scratch_bytes)
        if bound > available:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Exact periodic source support exceeds its original scratch allowance.",
                requested=(
                    ("maximum_scratch_bytes", self.limits.maximum_scratch_bytes),
                    ("remaining_scratch_bytes", available),
                ),
                achieved=(("source_support_scratch_bound", bound),),
            )


def _periodic_source_frame(
    source: CellMesh,
    target: CellMesh,
    allowance: _PeriodicSupportAllowance,
    /,
) -> _PeriodicImageFrame:
    original, successor = (
        _require_periodic_topology(source),
        _require_periodic_topology(target),
    )
    if _identification_id(original.cell) != _identification_id(successor.cell):
        raise ValueError(
            "Periodic source support must retain its actual identification group."
        )
    dimension = source.ambient_dimension
    group_images = (
        int(np.prod(original.cell.linear_orders, dtype=object))
        if isinstance(original.cell, PeriodicIsometryGroup)
        else 0
    )
    point_count = source.coordinates.shape[0] + target.coordinates.shape[0]
    entity_count = sum(
        mesh.entity_set(degree).count
        for mesh in (source, target)
        for degree in range(dimension + 1)
    )
    # Binary64 coordinates and two affine group products fit in 6400 integer
    # bits. Include Python integer digits/pointers and bounded SAT/support
    # temporaries before preparing the unbounded-precision host bank.
    scratch = (
        1024
        * (
            8 * point_count * dimension
            + (group_images + 2 * point_count) * (dimension + 1) ** 2
            + 4 * entity_count * (dimension + 1) * dimension
        )
        + 2**23
    )
    allowance.scratch(scratch)
    from ...discretization._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(0, scratch)
    allowance.spend(point_count + group_images)
    coordinates = np.concatenate(
        (np.asarray(source.coordinates), np.asarray(target.coordinates))
    )
    roots = np.concatenate(
        (
            np.asarray(original.vertex_representatives),
            source.coordinates.shape[0] + np.asarray(successor.vertex_representatives),
        )
    )
    shifts = np.concatenate(
        (np.asarray(original.vertex_shifts), np.asarray(successor.vertex_shifts))
    )
    try:
        image_exponents = None
        if isinstance(original.cell, PeriodicIsometryGroup) and any(
            order == 0 for order in original.cell.orders
        ):
            boxes = _affine_periodic_source_support_boxes(
                coordinates, roots, shifts, original.cell
            )
            image_exponents = _periodic_overlap_image_exponents(
                original.cell,
                boxes,
                allowance.limits.maximum_vertices,
            )
            actual_images = image_exponents.shape[0]
            additional_storage = (
                1024 * (actual_images - group_images) * (dimension + 1) ** 2
            )
            allowance.scratch(scratch + additional_storage)
            if ledger is not None:
                ledger.reserve(0, additional_storage)
            allowance.spend(actual_images - group_images)
        return _prepare_periodic_image_frame(
            coordinates,
            roots,
            shifts,
            original.cell,
            allowance.limits.maximum_vertices,
            image_exponents=image_exponents,
        )
    except _PeriodicEmbeddingResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        ) from error


def _periodic_source_regions(
    source: NativePeriodicSource,
    target: CellMesh,
    frame: _PeriodicImageFrame,
    allowance: _PeriodicSupportAllowance,
    /,
) -> np.ndarray:
    domain = source.domain
    if not isinstance(domain, CellMesh):
        raise TypeError(
            "Periodic material support requires the original represented source."
        )
    source_rows = _periodic_entity_rows(domain, domain.topological_dimension)
    target_rows = _periodic_entity_rows(target, target.topological_dimension)
    count = domain.coordinates.shape[0]
    target_corners = frame.points[count:][target_rows]
    target_lower, target_upper = (
        np.min(target_corners, axis=1),
        np.max(target_corners, axis=1),
    )
    block_positions = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(domain.blocks[0].global_ids))
    }
    source_regions = np.asarray(source.cell_regions)[
        [
            block_positions[int(identifier)]
            for identifier in np.asarray(
                domain.entity_set(domain.topological_dimension).entity_ids
            )
        ]
    ]
    members: list[set[int]] = [set() for _ in target_rows]
    for image in frame.images():
        allowance.spend(source_rows.shape[0] * target_rows.shape[0])
        corners = frame.image_points(image, slice(0, count))[source_rows]
        lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
        for row, triangle in enumerate(target_corners):
            possible = np.flatnonzero(
                np.all(
                    (target_upper[row] > lower) & (upper > target_lower[row]),
                    axis=1,
                )
            )
            for parent in possible:
                if _interiors_overlap(triangle, corners[parent]):
                    members[row].add(int(source_regions[parent]))
    unresolved = tuple(
        int(identifier)
        for identifier, material in zip(
            np.asarray(target.entity_set(target.topological_dimension).entity_ids),
            members,
            strict=True,
        )
        if len(material) != 1
    )
    if unresolved:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A periodic target cell crosses or loses an actual authored material stratum.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=unresolved,
        )
    return np.asarray([next(iter(material)) for material in members], dtype=np.int64)


def _periodic_seed_associations(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    target: CellMesh,
    /,
) -> tuple[GeometryAssociation, ...]:
    domain = source.domain
    if not isinstance(domain, PeriodicPointOrbits):
        raise TypeError(
            "Lattice-region authority requires the original periodic point-orbit source."
        )
    topology = _require_periodic_topology(target)
    if _identification_id(topology.cell) != _identification_id(domain.cell):
        raise ValueError(
            "Periodic lattice-region authority must retain its original identification."
        )
    quotient = PeriodicQuotientEvidence(target)
    if (
        not quotient.all_cells_valid
        or quotient.boundary_facet_count
        or quotient.relative_coverage_defect is None
        or quotient.relative_coverage_defect > 1.0e-10
    ):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "The periodic target does not cover its actual authored lattice region once.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        )
    allowance = _PeriodicSupportAllowance(specification.limits)
    frame = _periodic_source_frame(target, target, allowance)
    allowance.spend(target.blocks[0].cell_count ** 2 * frame.image_count)
    certify_periodic_embedding(
        target,
        maximum_images=specification.limits.maximum_vertices,
        maximum_pairs=specification.limits.maximum_work_units,
    )
    result: list[GeometryAssociation] = []
    for degree in range(target.topological_dimension + 1):
        entities = target.entity_set(degree)
        count = entities.count
        allowance.spend(count)
        result.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                entities.entity_set_id,
                entities.entity_ids,
                (
                    _entity(
                        source.source_revision, GeometrySourceEntityRole.REGION.value, 0
                    ),
                )
                * count,
                np.zeros(count, dtype=np.float64),
                exact=True,
                source_dimensions=np.full(
                    count, target.topological_dimension, dtype=np.int8
                ),
                source_indices=np.zeros(count, dtype=np.int64),
                source_entity_roles=(GeometrySourceEntityRole.REGION,) * count,
            )
        )
    return tuple(result)


def _periodic_source_associations(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    target: CellMesh,
    /,
) -> tuple[GeometryAssociation, ...]:
    """Keep real source support storage alive through the complete membership phase."""
    from ...discretization._coordinate_enclosure import (
        coordinate_enclosure_budget,
        CoordinateEnclosureResourceError,
    )

    limits = specification.limits
    ledger = coordinate_enclosure_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    try:
        with (
            ledger.activate(),
            ledger.bound_stage(
                limits.maximum_work_units,
                limits.maximum_scratch_bytes,
                starting_work_units=ledger.work_units,
            ),
            ledger.temporary_scope(),
        ):
            try:
                return _periodic_source_membership(source, specification, target)
            finally:
                ledger.charge_native_work(
                    ledger.work_units - ledger.native_charged_work_units
                )
    except CoordinateEnclosureResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Original periodic source support exhausted its exact work/storage allowance.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            requested=((str(error.resource), error.requested), ("limit", error.limit)),
            achieved=(("completed", error.completed),),
        ) from error


def _periodic_source_membership(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    target: CellMesh,
    /,
) -> tuple[GeometryAssociation, ...]:
    domain = source.domain
    if isinstance(domain, PeriodicPointOrbits):
        return _periodic_seed_associations(source, specification, target)
    allowance = _PeriodicSupportAllowance(specification.limits)
    original = PeriodicQuotientEvidence(domain)
    if not original.all_cells_valid or (
        original.relative_coverage_defect is not None
        and (original.boundary_facet_count or original.relative_coverage_defect > 1.0e-10)
    ):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "The original periodic geometry authority fails its positive quotient coverage contract.",
        )
    frame = _periodic_source_frame(domain, target, allowance)
    allowance.spend(domain.blocks[0].cell_count ** 2 * frame.image_count)
    certify_periodic_embedding(
        domain,
        maximum_images=specification.limits.maximum_vertices,
        maximum_pairs=specification.limits.maximum_work_units,
    )
    regions = _periodic_source_regions(source, target, frame, allowance)
    incidence = _periodic_region_incidence(target, regions)
    strata = _periodic_source_strata(source, specification)
    topology = _require_periodic_topology(domain)
    count, dimension = domain.coordinates.shape[0], domain.topological_dimension
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
    )
    associations: list[GeometryAssociation] = []
    for degree in range(dimension + 1):
        rows = _periodic_entity_rows(target, degree)
        corners = frame.points[count:][rows]
        classes = np.full(rows.shape[0], dimension, dtype=np.int8)
        indices = np.asarray(
            [material[0] if len(material) == 1 else -1 for material in incidence[degree]],
            dtype=np.int64,
        )
        entity_roles = [GeometrySourceEntityRole.REGION for _ in rows]
        ambiguous = np.zeros(rows.shape[0], dtype=np.bool_)
        for source_degree in range(degree, dimension):
            source_rows = _periodic_entity_rows(domain, source_degree)
            source_orbits = np.asarray(topology.orbits(source_degree)[0])
            selected = np.flatnonzero(np.isin(source_orbits, strata[source_degree]))
            for image in frame.images():
                allowance.spend(rows.shape[0] * selected.size)
                imaged = frame.image_points(image, slice(0, count))
                for parent in selected:
                    mask = np.empty(rows.shape[0], dtype=np.bool_)
                    for begin in range(0, rows.shape[0], 32):
                        end = min(begin + 32, rows.shape[0])
                        mask[begin:end] = _source_entity_contains(
                            corners[begin:end], imaged[source_rows[parent]]
                        )
                    ambiguous |= (
                        mask
                        & (classes == source_degree)
                        & (indices != source_orbits[parent])
                    )
                    replace = mask & (classes > source_degree)
                    classes[replace], indices[replace] = (
                        source_degree,
                        source_orbits[parent],
                    )
                    ambiguous[replace] = False
                    for row in np.flatnonzero(replace):
                        entity_roles[int(row)] = roles[source_degree]
        bad = (indices < 0) | ambiguous
        if np.any(bad):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Periodic geometry membership has an unresolved actual source stratum.",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                entity_ids=tuple(
                    int(value)
                    for value in np.asarray(target.entity_set(degree).entity_ids)[bad]
                ),
            )
        source_ids = tuple(
            _entity(source.source_revision, role.value, int(index))
            for role, index in zip(entity_roles, indices, strict=True)
        )
        associations.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                target.entity_set(degree).entity_set_id,
                target.entity_set(degree).entity_ids,
                source_ids,
                np.zeros(rows.shape[0], dtype=np.float64),
                exact=True,
                source_dimensions=classes,
                source_indices=indices,
                source_entity_roles=tuple(entity_roles),
            )
        )
    return tuple(associations)


@final
class PeriodicAssociationTransfer(StrictModule, NonTrainableState):
    """Renew complete membership in the unchanged authored periodic geometry."""

    source: NativePeriodicSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativePeriodicSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> None:
        issues = periodic_support_issues(source, specification)
        if issues:
            raise ValueError("; ".join(issues))
        self.source, self.specification = source, specification
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "periodic-association-transfer",
                "source": source.binding_id,
                "specification": specification.specification_id,
            }
        )

    @checked
    def source_associations(
        self,
        source: CellMeshingResult,
        /,
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        if not is_affine_cell_geometry(source.mesh, source.geometry):
            raise ValueError(
                "Periodic source authority requires its actual affine quotient map."
            )
        dimension = source.mesh.topological_dimension
        by_dimension = {
            _target_dimension(source.mesh, item): item for item in source.associations
        }
        if len(by_dimension) != len(source.associations) or set(by_dimension) != set(
            range(dimension + 1)
        ):
            raise ValueError(
                "Periodic source requires one complete original association table per dimension."
            )
        expected = _periodic_source_associations(
            self.source, self.specification, source.mesh
        )
        for degree, proof in enumerate(expected):
            item = by_dimension[degree]
            item.validate_target(source.mesh.entity_set(degree))
            rows = item.target_rows(np.asarray(proof.target_global_ids))
            if (
                item.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
                or item.source_id != self.source.source_id
                or item.source_revision != self.source.source_revision
                or not item.exact
                or not item.complete
                or tuple(item.source_entity_ids[row] for row in rows)
                != proof.source_entity_ids
                or not np.array_equal(
                    np.asarray(item.source_dimensions)[rows],
                    np.asarray(proof.source_dimensions),
                )
                or not np.array_equal(
                    np.asarray(item.source_indices)[rows],
                    np.asarray(proof.source_indices),
                )
                or item.source_entity_roles is None
                or tuple(item.source_entity_roles[row] for row in rows)
                != proof.source_entity_roles
                or np.any(np.asarray(item.residuals)[rows] != 0.0)
            ):
                raise ValueError(
                    "Periodic associations disagree with their unchanged exact original strata."
                )
        return by_dimension[0], tuple(range(1, dimension + 1))

    @checked
    def classes(self, source: CellMeshingResult, /) -> tuple[PlcEntityClasses, ...]:
        self.source_associations(source)
        domain = self.source.domain
        dimension = self.source.ambient_dimension
        namespaces: list[tuple[int, ...]]
        if isinstance(domain, CellMesh):
            topology = _require_periodic_topology(domain)
            namespaces = [
                tuple(range(topology.quotient.entities(degree).count))
                for degree in range(dimension)
            ]
        else:
            namespaces = [() for _ in range(dimension)]
        namespaces.append(
            tuple(int(region) for region in np.unique(self.source.cell_regions))
        )
        codes = {
            (degree, identifier): code
            for code, (degree, identifier) in enumerate(
                (degree, identifier)
                for degree, bank in enumerate(namespaces)
                for identifier in bank
            )
        }
        result: list[PlcEntityClasses] = []
        for degree in range(dimension + 1):
            item = next(
                item
                for item in source.associations
                if _target_dimension(source.mesh, item) == degree
            )
            rows = item.target_rows(np.asarray(source.mesh.entity_set(degree).entity_ids))
            dimensions = np.asarray(item.source_dimensions, dtype=np.int64)[rows]
            indices = np.asarray(item.source_indices, dtype=np.int64)[rows]
            namespace = np.asarray(
                [
                    codes[int(kind), int(identifier)]
                    for kind, identifier in zip(dimensions, indices, strict=True)
                ],
                dtype=np.int64,
            )
            result.append(
                PlcEntityClasses(
                    dimensions, indices, np.ones(rows.shape, dtype=np.bool_), namespace
                )
            )
        return tuple(result)

    @checked
    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
    ) -> tuple[GeometryAssociation, ...]:
        if (lineage.source_topology_id, lineage.target_topology_id) != (
            source.mesh.topology_id,
            target.topology_id,
        ):
            raise ValueError(
                "Periodic source renewal requires its actual mesh transition lineage."
            )
        if not is_affine_cell_geometry(target, geometry):
            raise ValueError(
                "Represented periodic source renewal requires its admitted affine geometry."
            )
        self.source_associations(source)
        return _periodic_source_associations(self.source, self.specification, target)


class _SiteMesh:
    """Quotient edge bank of one periodic Delaunay triangulation of orbit sites."""

    def __init__(
        self,
        points: np.ndarray,
        vectors: np.ndarray,
        triangulation: PeriodicDelaunayTriangulation | PeriodicStratifiedTriangulation,
        /,
    ) -> None:
        self.triangulation = triangulation
        self.sizing_evidence: tuple[PeriodicSizeRecord, ...] = ()
        self.image_refusals = 0
        simplices = np.asarray(triangulation.simplices, dtype=np.int64)
        shifts = np.asarray(triangulation.simplex_shifts, dtype=np.int64)
        self.simplices = simplices
        self.simplex_shifts = shifts
        rows = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            head = np.concatenate((simplices[:, first, None], shifts[:, first]), axis=1)
            tail = np.concatenate((simplices[:, second, None], shifts[:, second]), axis=1)
            difference = tail - head
            nonzero = difference != 0
            leading = np.argmax(nonzero, axis=1)
            leads = ~np.any(nonzero, axis=1) | (
                difference[np.arange(difference.shape[0]), leading] > 0
            )
            # The translation-invariant key of `_edge_key`: leader, follower, relative shift.
            rows.append(
                np.where(
                    leads[:, None],
                    np.concatenate(
                        (head[:, :1], tail[:, :1], tail[:, 1:] - head[:, 1:]), axis=1
                    ),
                    np.concatenate(
                        (tail[:, :1], head[:, :1], head[:, 1:] - tail[:, 1:]), axis=1
                    ),
                )
            )
        self.keys, inverse = np.unique(np.concatenate(rows), axis=0, return_inverse=True)
        # Quotient edge rows of each simplex, one row per local edge.
        self.simplex_edges = inverse.reshape((3, simplices.shape[0]))
        self.vectors = vectors
        self.points = points
        # Hard statistics are judged on published coordinates, so measure the
        # same reference lifts and shift arithmetic as the compliance owner.
        self.lengths = quotient_edge_lengths(
            points,
            self.keys,
            periodic_reference_lifts(simplices, shifts, points.shape[0]),
            vectors,
        )

    def merit(
        self,
        controls: list[UniformSizeControl],
        policy: SizeCompliancePolicy,
        /,
        *,
        quadratic: bool = False,
    ) -> tuple[int, int, float, int]:
        """Hard issues, over-target edges, statistic deviation, and target-length edges."""
        growth = edge_growth_evidence(
            self.lengths, self.keys[:, :2], self.points.shape[0]
        )
        issues, longer, deviation, matched = 0, 0, 0.0, 0
        for control in controls:
            _, achieved, failed = uniform_size_compliance(
                control, policy, self.lengths, growth
            )
            tolerance = policy.tolerance(control.target_size)
            issues += len(failed)
            longer += int(
                np.count_nonzero(self.lengths > control.target_size + tolerance)
            )
            matched += int(
                np.count_nonzero(np.abs(self.lengths - control.target_size) <= tolerance)
            )
            values = dict(achieved)
            prefix = f"size:{control.control_id}"
            deviation += sum(
                (
                    abs(values[f"{prefix}:{name}_edge"] - control.target_size)
                    / control.target_size
                )
                ** (2 if quadratic else 1)
                for name in policy.target_statistics
            )
            if control.maximum_size is not None:
                deviation += (
                    max(0.0, values[f"{prefix}:maximum_edge"] - control.maximum_size)
                    / control.maximum_size
                )
            if control.minimum_size is not None:
                deviation += (
                    max(0.0, control.minimum_size - values[f"{prefix}:minimum_edge"])
                    / control.minimum_size
                )
            if control.maximum_growth_rate is not None:
                deviation += (
                    max(0.0, growth - control.maximum_growth_rate)
                    / control.maximum_growth_rate
                )
        return issues, longer, deviation, -matched

    def frontal_edges(self, size: float, /) -> np.ndarray:
        """Edges of every triangle that still holds an edge longer than ``size``."""
        open_cells = np.any(self.lengths[self.simplex_edges] > size, axis=0)
        return np.unique(self.simplex_edges[:, open_cells])


# Frontal apexes closer than this fraction of the target to an existing orbit
# would only create sliver edges; they are not candidates.
_MINIMUM_SITE_SEPARATION = 0.25


def _frontal_site(
    state: _SiteMesh, edge: int, side: float, size: float, /
) -> np.ndarray | None:
    """Fundamental representative of the frontal ideal point of one quotient edge.

    This is the apex of the isosceles triangle with legs ``size`` over a
    quotient edge no longer than ``2 size``, on the requested side.
    """
    vectors, key = state.vectors, state.keys[edge]
    first = state.points[key[0]]
    second = state.points[key[1]] + key[2:] @ vectors
    direction = second - first
    length = float(np.linalg.norm(direction))
    if not 0.0 < length <= 2.0 * size:
        return None
    normal = side * np.asarray((-direction[1], direction[0])) / length
    apex = 0.5 * (first + second) + normal * np.sqrt(size * size - 0.25 * length * length)
    image = np.floor(np.linalg.solve(vectors.T, apex))
    site = apex - image @ vectors
    if side == 0.0:
        return site
    # Choose the representable frontal construction, not a relaxed comparison.
    # Wrapping and subtraction can round an ideal target-length leg upward.
    # Adjacent coordinate values are alternative constructions evaluated with
    # precisely the quotient metric used by the compliance owner.
    neighbors = [site]
    lower, upper = site.copy(), site.copy()
    for _ in range(2):
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
        neighbors.extend((lower, upper))
    candidates = np.asarray(
        [(x[0], y[1]) for x in neighbors for y in neighbors], dtype=np.float64
    )
    legs = np.stack(
        (
            candidates - first + image @ vectors,
            candidates - state.points[key[1]] + (image - key[2:]) @ vectors,
        ),
        axis=1,
    )
    lengths = np.linalg.norm(legs, axis=2)
    over = np.count_nonzero(lengths > size, axis=1)
    exact = np.count_nonzero(lengths == size, axis=1)
    deviation = np.sum(np.abs(lengths - size), axis=1)
    selected = np.lexsort((deviation, -exact, over))[0]
    return candidates[selected]


def _source_growth_site(
    state: _SiteMesh, controls: list[UniformSizeControl], policy: SizeCompliancePolicy, /
) -> np.ndarray | None:
    """Remove unprotected fixed long edges, then make immutable quantiles feasible."""
    construction = state.triangulation
    if not isinstance(construction, PeriodicStratifiedTriangulation):
        raise TypeError(
            "Source growth requires native constrained construction witnesses."
        )
    protected_edges = set(construction.protected_edges)
    threshold = min(
        control.target_size + policy.tolerance(control.target_size)
        for control in controls
    )
    fixed = np.all(construction.protected_vertices[state.keys[:, :2]], axis=1)
    for edge in np.argsort(-state.lengths, kind="stable"):
        key = tuple(int(value) for value in state.keys[edge])
        if fixed[edge] and state.lengths[edge] > threshold and key not in protected_edges:
            return 0.5 * (
                state.points[key[0]]
                + state.points[key[1]]
                + np.asarray(key[2:], dtype=np.int64) @ state.vectors
            )
    immutable_over = sum(
        state.lengths[index] > threshold
        for index, row in enumerate(state.keys)
        if tuple(int(value) for value in row) in protected_edges
    )
    quantile = max({"p50": 0.5, "p95": 0.95}[name] for name in policy.target_statistics)
    required_edges = int(np.ceil(1.0 + immutable_over / (1.0 - quantile)))
    if state.keys.shape[0] >= required_edges:
        return None
    corners = state.points[state.simplices] + state.simplex_shifts @ state.vectors
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    for cell in np.argsort(-area, kind="stable"):
        candidate = np.mean(corners[cell], axis=0)
        gap = np.linalg.solve(state.vectors.T, (state.points - candidate).T).T
        if np.min(np.linalg.norm((gap - np.rint(gap)) @ state.vectors, axis=1)) > 0.0:
            return candidate
    raise MeshingFailure(
        MeshingFailureCategory.COMPLIANCE_FAILED,
        "No source-interior cavity can satisfy the immutable periodic rank budget.",
    )


def _realize_target_size_sites(
    points: np.ndarray,
    cell: PeriodicCell,
    controls: list[UniformSizeControl],
    policy: SizeCompliancePolicy,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    image_budget: int,
    /,
    *,
    source_binding_id: str,
    source_strata: PeriodicSourceStrata | None = None,
) -> tuple[_SiteMesh, int]:
    """Insert frontal orbit sites until every hard uniform size statistic is met.

    Each step tries the ideal frontal apex of every edge on the front (edges of
    triangles that still hold an over-target edge), on both sides, triangulates
    the actual periodic Delaunay candidate and accepts the one that most
    reduces (hard issues, over-target edges, statistic deviation) while
    maximizing target-length edges (within the policy's comparison
    tolerance). Inserted sites are lattice orbits, so the quotient is
    preserved by construction. Source-stratified construction uses the first
    improving candidate after protected-orbit subdivision; this bounds native
    constrained retriangulation work. A step without improvement stops the
    realization; the unchanged hard compliance check then reports it.
    """
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    size = min(control.target_size for control in controls)
    work = 0

    def triangulate(points: np.ndarray, /) -> _SiteMesh:
        nonlocal work
        if source_strata is not None:
            constrained = triangulate_periodic_source_strata(
                points,
                source_strata,
                limits,
                image_budget,
                limits.maximum_work_units - work,
            )
            work += constrained.work_units
            return _SiteMesh(points, vectors, constrained)
        # Native scratch is first sized to this candidate's site count; only a
        # refusal at that size escalates to the request's enforced limits.
        images = min(image_budget, 27 * points.shape[0] + 4096)
        for maximum_images, maximum_cells in (
            (images, min(limits.maximum_cells, 64 * images)),
            (image_budget, limits.maximum_cells),
        ):
            try:
                triangulation = PeriodicDelaunayTriangulation(
                    points,
                    cell,
                    maximum_images=maximum_images,
                    maximum_simplices=maximum_cells,
                )
            except PeriodicImageBudgetError as error:
                if (maximum_images, maximum_cells) == (
                    image_budget,
                    limits.maximum_cells,
                ):
                    raise MeshingFailure(
                        MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        str(error),
                        stage=stage.value,
                    ) from error
                continue
            work += triangulation.simplices.shape[0]
            if work > limits.maximum_work_units:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Periodic target-size site insertion exceeds its work budget.",
                    stage=stage.value,
                )
            return _SiteMesh(points, vectors, triangulation)
        raise RuntimeError(
            "Periodic site triangulation exhausted its escalation without a result."
        )

    state = triangulate(points)
    image_refusals = 0
    current = state.merit(controls, policy, quadratic=source_strata is not None)
    while current[0]:
        if state.points.shape[0] + 1 > limits.maximum_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic target-size site insertion exceeds the vertex budget.",
                stage=stage.value,
            )
        current_score = (
            (current[2], current[0], float(current[1]), current[3])
            if source_strata is not None
            else (float(current[0]), current[1], current[2], current[3])
        )
        best: (
            tuple[tuple[float, int, float, int], tuple[int, int, float, int], _SiteMesh]
            | None
        ) = None
        front_edges = state.frontal_edges(size) if source_strata is None else ()
        for edge in front_edges:
            protected = source_strata is not None and any(
                np.array_equal(state.keys[edge], key) for key in source_strata.edges
            )
            fronts = (
                ((1.0, -1.0), (1.0,), (-1.0,))
                if source_strata is not None
                else ((1.0,), (-1.0,))
            )
            if protected:
                fronts = ((0.0,), *fronts)
            for sides in fronts:
                check_deadline(started, limits, stage)
                if state.points.shape[0] + len(sides) > limits.maximum_vertices:
                    continue
                sites: list[np.ndarray] = []
                for side in sides:
                    site = _frontal_site(state, int(edge), side, size)
                    if site is None:
                        break
                    existing = (
                        state.points
                        if not sites
                        else np.concatenate((state.points, np.stack(sites)))
                    )
                    gap = np.linalg.solve(vectors.T, (existing - site).T).T
                    if (
                        np.min(np.linalg.norm((gap - np.rint(gap)) @ vectors, axis=1))
                        < _MINIMUM_SITE_SEPARATION * size
                    ):
                        break
                    sites.append(site)
                if len(sites) != len(sides):
                    continue
                try:
                    candidate = triangulate(
                        np.concatenate((state.points, np.stack(sites)))
                    )
                except PeriodicConstructionImageBudgetError as error:
                    work += error.work_units
                    image_refusals += 1
                    continue
                candidate.sizing_evidence = state.sizing_evidence
                merit = candidate.merit(
                    controls, policy, quadratic=source_strata is not None
                )
                score = (
                    (merit[2], merit[0], float(merit[1]), merit[3])
                    if source_strata is not None
                    else (float(merit[0]), merit[1], merit[2], merit[3])
                )
                if best is None or score < best[0]:
                    best = (score, merit, candidate)
                if merit[0] == 0 or (source_strata is not None and score < current_score):
                    break
            if best is not None and (
                best[1][0] == 0 or (source_strata is not None and best[0] < current_score)
            ):
                break
        if best is None or best[0] >= current_score:
            if source_strata is not None:
                growth = _source_growth_site(state, controls, policy)
                if growth is not None:
                    enlarged = triangulate(np.concatenate((state.points, growth[None])))
                    enlarged.sizing_evidence = state.sizing_evidence
                    current, state = (
                        enlarged.merit(controls, policy, quadratic=True),
                        enlarged,
                    )
                    continue
            if source_strata is not None and isinstance(
                state.triangulation, PeriodicStratifiedTriangulation
            ):
                try:
                    proposal = propose_periodic_site_relocation(
                        state.points,
                        state.keys,
                        state.simplices,
                        state.simplex_shifts,
                        vectors,
                        state.triangulation.protected_vertices,
                        size,
                        policy.target_statistics,
                        limits.maximum_work_units - work,
                        limits.maximum_scratch_bytes,
                        lambda: check_deadline(started, limits, stage),
                        source_binding_id=source_binding_id,
                        maximum_size=min(
                            (
                                control.maximum_size
                                for control in controls
                                if control.maximum_size is not None
                            ),
                            default=np.inf,
                        ),
                        minimum_size=max(
                            (
                                control.minimum_size
                                for control in controls
                                if control.minimum_size is not None
                            ),
                            default=0.0,
                        ),
                    )
                except MeshingFailure as error:
                    if error.category is not MeshingFailureCategory.RESOURCE_EXHAUSTED:
                        raise
                    measured = (
                        ("candidate:p50_edge", float(np.quantile(state.lengths, 0.5))),
                        ("candidate:p95_edge", float(np.quantile(state.lengths, 0.95))),
                        ("candidate:maximum_edge", float(np.max(state.lengths))),
                        ("candidate:optimizer_calls", len(state.sizing_evidence)),
                    )
                    raise MeshingFailure(
                        error.category,
                        str(error),
                        requested=error.evidence.requested,
                        achieved=(*error.evidence.achieved, *measured),
                    ) from error
                work += proposal.work_units
                state.sizing_evidence = (*state.sizing_evidence, proposal.record)
                check_deadline(started, limits, stage)
                if proposal.points is not None:
                    corrected = _SiteMesh(proposal.points, vectors, state.triangulation)
                    measured = corrected.merit(controls, policy, quadratic=True)
                    if (
                        measured[2],
                        measured[0],
                        float(measured[1]),
                        measured[3],
                    ) < current_score:
                        corrected.sizing_evidence = state.sizing_evidence
                        current, state = measured, corrected
                        continue
            break
        current, state = best[1], best[2]
    state.image_refusals = image_refusals
    return state, work


def execute_periodic_route(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    prepared: PreparedPeriodicDomain,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    if (
        prepared.source.binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError("The prepared periodic route is bound to a different request.")
    started = monotonic()
    try:
        constraint_requested, constraint_achieved = _bound_constraint_evidence(
            source, specification.periodic_constraints
        )
    except ValueError as error:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED, str(error)
        ) from error
    limits = specification.limits
    stage = (
        MeshingStageKind.SURFACE_MESHING
        if source.ambient_dimension == 2
        else MeshingStageKind.VOLUME_FILL
    )
    image_budget = min(
        limits.maximum_vertices,
        limits.maximum_scratch_bytes // (8 * (source.ambient_dimension + 8)),
    )
    if image_budget <= 0:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Periodic construction has no image storage budget.",
        )
    controls: list[UniformSizeControl] = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError("The prepared periodic route requires uniform size controls.")
        controls.append(control)
    work = 0
    inserted_sites = 0
    sizing_evidence: tuple[PeriodicSizeRecord, ...] = ()
    image_refusals = 0
    hard = [
        control for control in controls if control.strength is SizeControlStrength.HARD
    ]
    stratified = False
    ancestry: tuple[tuple[int, ...], ...]
    if isinstance(source.domain, PeriodicPointOrbits):
        orbits = source.domain
        with measure_phase(record_phase, "periodic_construction"):
            if hard and source.ambient_dimension == 2:
                site_mesh, work = _realize_target_size_sites(
                    np.asarray(orbits.representatives),
                    orbits.cell,
                    hard,
                    specification.size_compliance,
                    limits,
                    started,
                    stage,
                    image_budget,
                    source_binding_id=source.binding_id,
                )
                inserted_sites = site_mesh.points.shape[0] - source.domain.orbit_count
                mesh = publish_periodic_simplices(
                    site_mesh.points,
                    site_mesh.simplices,
                    site_mesh.simplex_shifts,
                    orbits.cell,
                )
            else:
                try:
                    construction = periodic_delaunay_mesh(
                        orbits,
                        maximum_images=image_budget,
                        maximum_simplices=limits.maximum_cells,
                    )
                except PeriodicImageBudgetError as error:
                    evidence = error.evidence
                    raise MeshingFailure(
                        MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        str(error),
                        stage=stage.value,
                        requested=(
                            ("maximum_images", evidence.maximum_images),
                            ("maximum_cells", evidence.maximum_cells),
                        ),
                        achieved=(
                            ("image_count", evidence.image_count),
                            ("finite_cell_slots", evidence.finite_cell_slots),
                        ),
                    ) from error
                mesh = construction.mesh
        regions = np.zeros(mesh.blocks[0].cell_count, dtype=np.int32)
        original_mesh = mesh
        original_regions = regions.copy()
        ancestry = tuple((index,) for index in range(mesh.blocks[0].cell_count))
    else:
        mesh = source.domain
        original_mesh = mesh
        original_regions = np.asarray(source.cell_regions).copy()
        regions = original_regions.copy()
        ancestry = tuple((index,) for index in range(mesh.blocks[0].cell_count))
        if (
            hard
            and source.ambient_dimension == 2
            and isinstance(_require_periodic_topology(mesh).cell, PeriodicCell)
        ):
            with measure_phase(record_phase, "periodic_construction"):
                strata = prepare_periodic_source_strata(
                    mesh,
                    regions,
                    specification.protected_features,
                )
                minimum = max(
                    control.minimum_size
                    - specification.size_compliance.tolerance(control.minimum_size)
                    if control.minimum_size is not None
                    else 0.0
                    for control in hard
                )
                seeds = periodic_source_seed_points(
                    strata,
                    min(
                        (
                            control.maximum_size
                            for control in hard
                            if control.maximum_size is not None
                        ),
                        default=np.inf,
                    ),
                    minimum,
                    limits.maximum_vertices,
                )
                site_mesh, work = _realize_target_size_sites(
                    seeds,
                    strata.cell,
                    hard,
                    specification.size_compliance,
                    limits,
                    started,
                    stage,
                    image_budget,
                    source_strata=strata,
                    source_binding_id=source.binding_id,
                )
                sizing_evidence = site_mesh.sizing_evidence
                image_refusals = site_mesh.image_refusals
                regions, ancestry, certification_work = bind_periodic_source_strata(
                    site_mesh.points,
                    site_mesh.simplices,
                    site_mesh.simplex_shifts,
                    strata,
                    limits.maximum_work_units - work,
                )
                work += certification_work
                inserted_sites = site_mesh.points.shape[0] - strata.points.shape[0]
                mesh = publish_periodic_simplices(
                    site_mesh.points,
                    site_mesh.simplices,
                    site_mesh.simplex_shifts,
                    strata.cell,
                    block_name=strata.block_name,
                )
                stratified = True
    initial = PeriodicQuotientEvidence(original_mesh)
    initial_measures = np.asarray(periodic_orbit_measures(original_mesh).orbit_measures)
    threshold = min(
        control.maximum_size
        if control.maximum_size is not None
        else control.target_size
        * (1.0 + specification.size_compliance.relative_tolerance)
        + specification.size_compliance.absolute_tolerance
        for control in controls
    )
    while True:
        check_deadline(started, limits, stage)
        edges = _simplex_entity_corners(mesh, 1)
        lengths, growth = periodic_edge_size_evidence(mesh)
        if stratified or float(np.max(lengths)) <= threshold:
            break
        # Bound the entire orbit closure before allocating a successor.
        count = mesh.blocks[0].cell_count
        upper_cells = count * (4 if source.ambient_dimension == 2 else 64)
        upper_vertices = mesh.coordinates.shape[0] + edges.shape[0] * (
            source.ambient_dimension + 1
        )
        if (
            upper_cells > limits.maximum_cells
            or upper_vertices > limits.maximum_vertices
            or upper_cells * (source.ambient_dimension + 1)
            > limits.maximum_connectivity_entries
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic refinement exceeds its closure budget.",
            )
        work += upper_cells
        if (
            work > limits.maximum_work_units
            or upper_cells * 128 + upper_vertices * 64 > limits.maximum_scratch_bytes
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic refinement exceeds its work/scratch budget.",
            )
        with measure_phase(record_phase, "refinement"):
            refinement = refine_periodic_mesh(
                mesh, source_geometry=CellGeometrySpec.affine(mesh)
            )
        regions = regions[np.asarray(refinement.parent_cells)]
        ancestry = tuple(
            ancestry[parent] for parent in np.asarray(refinement.parent_cells)
        )
        mesh = refinement.mesh
    simplex_entity_limits(
        np.asarray(mesh.coordinates),
        np.asarray(mesh.blocks[0].vertices),
        limits,
        stage,
        cell_kind=mesh.blocks[0].cell_kind,
    )
    quotient = PeriodicQuotientEvidence(mesh)
    if (
        not quotient.all_cells_valid
        or abs(quotient.total_measure - initial.total_measure)
        > 1.0e-10 * initial.total_measure
    ):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Periodic orbit subdivision changed source coverage.",
        )
    if quotient.relative_coverage_defect is not None and (
        quotient.relative_coverage_defect > 1.0e-10 or quotient.boundary_facet_count
    ):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "The translational quotient is not a closed degree-one domain cover.",
        )
    embedding_work: list[tuple[int, int, int]] = []
    embedding_unmanaged: list[tuple[str, float]] = []

    def record_embedding_work(
        setup_bound: int, aabb_tests: int, exact_tests: int
    ) -> None:
        embedding_work[:] = [(setup_bound, aabb_tests, exact_tests)]

    def record_embedding_unmanaged(peak: int) -> None:
        embedding_unmanaged[:] = [
            ("periodic:embedding_logical_unmanaged_peak_bytes_upper", float(peak)),
        ]

    def embedding_work_evidence() -> tuple[tuple[str, float], ...]:
        return tuple(
            item
            for setup_bound, aabb_tests, exact_tests in embedding_work
            for item in (
                ("periodic:embedding_setup_work_bound", float(setup_bound)),
                ("periodic:embedding_aabb_tests", float(aabb_tests)),
                ("periodic:embedding_exact_interior_tests", float(exact_tests)),
                # SAT big-integer inner arithmetic is not instrumented as work.
                ("periodic:embedding_inner_predicate_work_complete", 0.0),
            )
        ) + tuple(embedding_unmanaged)

    with measure_phase(record_phase, "periodic_embedding"):
        try:
            # Actual bounded BVH visits now debit the original native owner;
            # candidate count is not a proxy for a fictitious all-cell scan.
            embedding_budget = image_budget // mesh.coordinates.shape[0]
            if embedding_budget <= 0 or limits.maximum_work_units <= work:
                raise _PeriodicEmbeddingResourceError(
                    "Periodic embedding has no remaining original image/work allowance."
                )
            embedding = certify_periodic_embedding(
                mesh,
                maximum_images=embedding_budget,
                maximum_pairs=limits.maximum_work_units - work,
                record_work=record_embedding_work,
                record_unmanaged=record_embedding_unmanaged,
            )
        except _PeriodicEmbeddingResourceError as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=MeshingStageKind.CERTIFICATION.value,
                achieved=embedding_work_evidence(),
            ) from error
        except MeshcoreError as error:
            if error.status not in (
                MeshcoreStatus.CAPACITY_EXCEEDED,
                MeshcoreStatus.TIMEOUT,
            ):
                raise
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT
                if error.status is MeshcoreStatus.TIMEOUT
                else MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=MeshingStageKind.CERTIFICATION.value,
                achieved=embedding_work_evidence(),
            ) from error
        except ValueError as error:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                str(error),
                stage=MeshingStageKind.CERTIFICATION.value,
                achieved=embedding_work_evidence(),
            ) from error
    requested, achieved, issues = constraint_requested, constraint_achieved, []
    with measure_phase(record_phase, "compliance"):
        for control in controls:
            expected, measured, failures = uniform_size_compliance(
                control, specification.size_compliance, lengths, growth
            )
            requested.extend(expected)
            achieved.extend(measured)
            issues.extend(failures)
    achieved.extend(
        (
            ("periodic:quotient_measure", quotient.total_measure),
            ("periodic:embedding_images", float(embedding.image_count)),
            ("periodic:embedding_pairs", float(embedding.candidate_pair_count)),
            ("periodic:target_size_orbit_sites", float(inserted_sites)),
            ("periodic:proposal_image_refusals", float(image_refusals)),
        )
    )
    achieved.extend(embedding_work_evidence())
    achieved.extend(
        (
            ("periodic:coordinate_lift_residual", embedding.maximum_coordinate_residual),
            (
                "periodic:coordinate_lift_residual_bound",
                embedding.coordinate_residual_bound,
            ),
        )
    )
    for index, record in enumerate(sizing_evidence):
        achieved.extend(
            (f"periodic:size_optimizer:{index}:{name}", float(value))
            for name, value in zip(
                PERIODIC_SIZE_EVIDENCE_FIELDS, record[1:-1], strict=True
            )
        )
        achieved.append((f"periodic:size_optimizer:{index}:counts_complete", 0.0))
        if record.polish is not None:
            for name, value in record.polish._asdict().items():
                if name == "status" or isinstance(value, str):
                    continue
                key = f"periodic:size_optimizer:{index}:polish_{name}"
                if value is not None and np.isfinite(value):
                    achieved.append((key, float(value)))
                else:
                    # The raw nonfinite value remains in the source record;
                    # a numeric compliance quantity cannot pretend it is known.
                    achieved.append((f"{key}_available", 0.0))
    successor_measures = np.asarray(periodic_orbit_measures(mesh).orbit_measures)
    with measure_phase(record_phase, "compliance"):
        for region in np.unique(original_regions):
            expected = float(np.sum(initial_measures[original_regions == region]))
            measured = float(np.sum(successor_measures[regions == region]))
            requested.append((f"periodic:region:{region}:measure", expected))
            achieved.append((f"periodic:region:{region}:measure", measured))
            if abs(measured - expected) > 1.0e-10 * expected:
                issues.append(f"periodic_material_coverage:{region}")
    if (
        isinstance(specification, SurfaceMeshingSpec)
        and specification.quality_target is not None
    ):
        corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
        angles = []
        for vertex in range(3):
            first = corners[:, (vertex + 1) % 3] - corners[:, vertex]
            second = corners[:, (vertex + 2) % 3] - corners[:, vertex]
            cosine = np.sum(first * second, axis=1) / (
                np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
            )
            angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        minimum = float(np.min(angles))
        requested.append(
            ("quality:minimum_angle", specification.quality_target.minimum_angle)
        )
        achieved.append(("quality:minimum_angle", minimum))
        if (
            specification.quality_target.hard
            and minimum < specification.quality_target.minimum_angle
        ):
            issues.append("quality:minimum_angle")
    labels = []
    if specification.protected_features:
        with measure_phase(record_phase, "organization"):
            for feature in specification.protected_features:
                ids = np.asarray(feature.scope.entity_ids)
                degree = feature.scope.entity_dimension
                if degree > 0:
                    mask = _feature_entity_mask(original_mesh, mesh, ids, degree)
                    source_rows = _simplex_entity_corners(original_mesh, degree)
                    source_corners = np.asarray(original_mesh.coordinates)[source_rows]
                    target_rows = _simplex_entity_corners(mesh, degree)
                    target_corners = np.asarray(mesh.coordinates)[target_rows]
                    if degree == 1:
                        source_lengths, _ = periodic_edge_size_evidence(original_mesh)
                        source_measures = source_lengths[
                            np.asarray(
                                _require_periodic_topology(original_mesh).orbits(1)[0]
                            )
                        ]
                        target_measures = lengths[
                            np.asarray(_require_periodic_topology(mesh).orbits(1)[0])
                        ]
                    else:
                        source_measures = 0.5 * np.linalg.norm(
                            np.cross(
                                source_corners[:, 1] - source_corners[:, 0],
                                source_corners[:, 2] - source_corners[:, 0],
                            ),
                            axis=1,
                        )
                        target_measures = 0.5 * np.linalg.norm(
                            np.cross(
                                target_corners[:, 1] - target_corners[:, 0],
                                target_corners[:, 2] - target_corners[:, 0],
                            ),
                            axis=1,
                        )
                    leaders = np.asarray(
                        _require_periodic_topology(original_mesh).orbit_representatives(
                            degree
                        )
                    )[ids]
                    target_leaders = np.asarray(
                        _require_periodic_topology(mesh).orbit_representatives(degree)
                    )
                    covered = float(
                        np.sum(target_measures[target_leaders[mask[target_leaders]]])
                    )
                    expected = float(np.sum(source_measures[leaders]))
                    if abs(covered - expected) > 1.0e-10 * max(1.0, expected):
                        issues.append(f"feature_orbit_coverage:{feature.feature_id}")
                else:
                    source_vertices = np.asarray(
                        _require_periodic_topology(original_mesh).orbit_representatives(0)
                    )[ids]
                    source_points = np.asarray(original_mesh.coordinates)[source_vertices]
                    points = np.asarray(mesh.coordinates)
                    mask = np.zeros(points.shape[0], dtype=np.bool_)
                    vertex_orbits = np.asarray(
                        _require_periodic_topology(mesh).orbits(0)[0]
                    )
                    leaders = np.asarray(
                        _require_periodic_topology(mesh).orbit_representatives(0)
                    )
                    for point in source_points:
                        matches = np.flatnonzero(np.all(points[leaders] == point, axis=1))
                        if matches.size != 1:
                            raise MeshingFailure(
                                MeshingFailureCategory.COMPLIANCE_FAILED,
                                "A protected quotient corner has no unique retained representative.",
                            )
                        mask |= vertex_orbits == matches[0]
                if not np.any(mask):
                    raise MeshingFailure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "A protected periodic feature orbit is absent.",
                    )
                entity_set = mesh.topology.entities(degree)
                labels.append(
                    MeshLabel(
                        f"periodic-feature:{feature.feature_id}",
                        MeshingScope(
                            mesh.mesh_id,
                            mesh.numeric_version,
                            MeshingEntityKind.MESH,
                            degree,
                            entity_set.entity_set_id,
                            np.asarray(entity_set.entity_ids)[mask],
                        ),
                    )
                )
    zones = []
    cell_set = mesh.topology.entities(mesh.topological_dimension)
    with measure_phase(record_phase, "organization"):
        for region in np.unique(regions):
            matching = [
                control
                for control in specification.region_controls
                if region in np.asarray(control.scope.entity_ids)
            ]
            materials = {control.material_id for control in matching}
            if len(materials) > 1:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "One periodic region has conflicting material assignments.",
                )
            control = matching[0] if matching else None
            zones.append(
                MeshZone(
                    f"periodic-region:{region}"
                    if control is None
                    else control.region_name,
                    MeshZoneRole.REGION,
                    MeshingScope(
                        mesh.mesh_id,
                        mesh.numeric_version,
                        MeshingEntityKind.MESH,
                        mesh.topological_dimension,
                        cell_set.entity_set_id,
                        np.asarray(cell_set.entity_ids)[regions == region],
                    ),
                    material_id=str(region) if control is None else control.material_id,
                    region_role=RegionRole.USER if control is None else control.role,
                )
            )
    with measure_phase(record_phase, "geometry_association"):
        associations = _periodic_source_associations(source, specification, mesh)
    cell_association = associations[-1]
    associated_regions = np.asarray(cell_association.source_indices)[
        cell_association.target_rows(np.asarray(mesh.blocks[0].global_ids))
    ]
    if not np.array_equal(associated_regions, regions):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Renewed original source membership changes the published material inventory.",
        )
    compliance = MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
        issues=tuple(issues),
    )
    certification = NativeCertificationRequest(
        MeshCertificationSchedule("periodic"),
        source.source_id,
        source.source_revision,
        limits,
    )
    check_deadline(started, limits, MeshingStageKind.CERTIFICATION)
    return publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        (
            MeshingStageReport(
                stage,
                MeshingStageStatus.PASSED,
                input_ids=(source.binding_id, prepared.prepared_id),
                output_ids=(mesh.mesh_id, embedding.certificate_id),
            ),
        ),
        provider,
        {
            "route": "periodic_delaunay",
            "plan": plan_id,
            "periodic_embedding": embedding.certificate_id,
            "periodic_coordinate_scope": embedding.coordinate_scope,
            "source_cell_ancestry": [list(parents) for parents in ancestry],
            "cell_regions": regions.tolist(),
            "size_optimizer_evidence": [
                {
                    **dict(zip(record._fields[:-1], record[:-1], strict=True)),
                    "polish": None
                    if record.polish is None
                    else {
                        name: value.hex()
                        if isinstance(value, float) and not np.isfinite(value)
                        else value
                        for name, value in zip(
                            record.polish._fields, record.polish, strict=True
                        )
                    },
                }
                for record in sizing_evidence
            ],
            "construction": "translational_delaunay"
            if isinstance(source.domain, PeriodicPointOrbits)
            else (
                "source_stratified_periodic_cdt"
                if stratified
                else "represented_domain_orbit_bisection"
            ),
        },
        certification,
        audit_policy=CellMeshAuditPolicy(),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "cells",
            "edges",
            "faces",
            "connectivity_entries",
            "data_bytes",
            "scratch_bytes",
            "work_units",
            "wall_seconds",
        )
        + (("cavity_cells",) if stratified else ())
        + ("geometry_queries",),
        unenforced_limits=(() if stratified else ("cavity_cells",)),
        labels=tuple(labels),
        zones=tuple(zones),
        associations=associations,
        record_phase=record_phase,
    )


__all__ = [
    "NativePeriodicSource",
    "PreparedPeriodicDomain",
    "PeriodicAssociationTransfer",
    "periodic_support_issues",
    "execute_periodic_route",
]
