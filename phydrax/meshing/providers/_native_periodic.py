#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native periodic construction and publication on the quotient carrier.

Point-orbit sources use certified translational Delaunay construction. Explicit
lifted simplex domains additionally admit finite rotational boundary isometries;
they are refined by orbit bisection, not advertised as rotational Delaunay.
Material regions are represented source-cell assignments, never centroid guesses.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, nullcontext
from fractions import Fraction
from itertools import combinations
from typing import assert_never, final, Literal

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._periodic_cell as periodic_cell_owner
import phydrax.discretization._periodic_measures as periodic_measures_owner
import phydrax.geometry._affine_mapped_contact as affine_mapped_contact_owner
from phydrax.discretization._periodic_exact import (
    identification_id,
    PeriodicIsometryGroup,
)
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    target_dimension,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingLimits,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._host_banks import entity_key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeResourceFailure,
    periodic_refine_2d,
    PeriodicRefinement,
)
from ..._physical import SpatialCoordinateContract
from ..._resource_ticket import BatchTicket
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import normalized_identifier
from ...discretization._cell_geometry_transfer import is_affine_cell_geometry
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ...discretization._exact_algebraic_geometry import (
    algebraic_source_limits,
    coefficient_endpoint,
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
)
from ...discretization._exact_rational_geometry import rational_source_layouts
from ...geometry._periodic_image_frame import (
    periodic_overlap_image_exponents,
    PeriodicEmbeddingResourceError,
    PeriodicImageFrame,
    prepare_periodic_image_frame,
)
from ...geometry._triangulation import (
    PeriodicDelaunayTriangulation,
    PeriodicImageBudgetError,
)
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...typing import checked, parse
from .. import _periodic as periodic_owner
from .._association_routes import PlcEntityClasses
from .._controls import FeatureKind
from .._lineage_records import MeshLineage
from .._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
)
from .._native_preparation import native_preparation_ledger
from .._organization_records import MeshLabel, MeshZone, MeshZoneRole, RegionRole
from .._periodic_constraint import PeriodicConstraint
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    size_statistic_quantile,
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from . import _native_periodic_constraints as constraints_owner
from ._native_periodic_size import (
    periodic_reference_lifts,
    PERIODIC_SIZE_EVIDENCE_FIELDS,
    PeriodicSizeRecord,
    propose_periodic_rank_placement,
    propose_periodic_site_relocation,
    quotient_edge_lengths,
)
from ._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import (
    check_deadline,
    edge_growth_evidence,
    simplex_entity_limits,
    uniform_size_compliance,
)
from ._native_route_metadata import specification_scope
from ._native_route_policy import publication_policy
from ._native_support import format_support_issues, SupportIssue


@final
class NativePeriodicSource(StrictModule, NonTrainableState):
    """Bind a torus seed or represented periodic simplex domain to its revision.

    ``cell_regions`` contains nonnegative source-region IDs (region zero for a
    point-orbit seed). A represented domain uses quotient entity positions,
    including distinct winding entities, for edge and vertex IDs. Refinement
    retains its material-interface complex.
    """

    domain: periodic_owner.PeriodicPointOrbits | cell_mesh_owner.CellMesh
    cell_regions: np.ndarray
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: periodic_owner.PeriodicPointOrbits | cell_mesh_owner.CellMesh,
        source_id: str,
        source_revision: str,
        /,
        *,
        cell_regions: ArrayLike | None = None,
    ) -> None:
        match domain:
            case periodic_owner.PeriodicPointOrbits():
                dimension = domain.cell.ambient_dimension
                if cell_regions is not None:
                    raise ValueError(
                        "NativePeriodicSource.cell_regions must be absent for point-orbit seeds; got a represented partition"
                    )
                regions = np.zeros(1, dtype=np.int32)
                identity = domain.orbits_id
            case cell_mesh_owner.CellMesh():
                if domain.periodic_topology is None or len(domain.blocks) != 1:
                    raise ValueError(
                        "NativePeriodicSource.domain must contain one bound periodic simplex block; got an unbound or multi-block mesh"
                    )
                dimension = domain.ambient_dimension
                if domain.blocks[0].cell_kind != {2: "triangle", 3: "tetrahedron"}.get(
                    dimension
                ):
                    raise ValueError(
                        "NativePeriodicSource.domain must contain full-dimensional simplices; got an incompatible cell family"
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
                        "NativePeriodicSource.cell_regions must assign one nonnegative integer per source cell; got an incompatible region array"
                    )
                identity = domain.mesh_id
            case _ as invalid:
                assert_never(invalid)
        source = normalized_identifier(str(source_id), "NativePeriodicSource.source_id")
        revision = normalized_identifier(
            str(source_revision), "NativePeriodicSource.source_revision"
        )
        if np.any(regions > np.iinfo(np.int64).max):
            raise ValueError(
                "NativePeriodicSource.cell_regions must fit in int64; got a region ID outside int64"
            )
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[NativePeriodicSource], value: NativePeriodicSource, /
    ) -> NativePeriodicSource:
        """Revalidate a stored periodic seed or represented domain."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.stored_fields must have the registered type; got {type(value).__name__}"
            )
        match value.domain:
            case periodic_owner.PeriodicPointOrbits():
                regions = None
            case cell_mesh_owner.CellMesh():
                regions = value.cell_regions
            case _ as invalid:
                assert_never(invalid)
        fresh = cls(
            value.domain,
            value.source_id,
            value.source_revision,
            cell_regions=regions,
        )
        if (
            fresh.binding_id != value.binding_id
            or fresh.ambient_dimension != value.ambient_dimension
            or not np.array_equal(fresh.cell_regions, value.cell_regions)
        ):
            raise ValueError(
                f"{cls.__name__}.stored_fields must match the periodic source binding; got changed regions or identity"
            )
        return fresh


@checked
def periodic_support_issues(
    source: NativePeriodicSource, specification: SurfaceMeshingSpec | VolumeMeshingSpec, /
) -> tuple[SupportIssue, ...]:
    issues: list[SupportIssue] = []
    dimension = source.ambient_dimension
    family = {2: "triangle", 3: "tetrahedron"}[dimension]
    target = specification.target
    if (
        target.topological_dimension != dimension
        or target.ambient_dimension != dimension
        or target.geometry_order != 1
    ):
        issues.append(
            SupportIssue("geometry", "full-dimensional affine periodic simplices")
        )
    if (
        set((*target.cell_families.required, *target.cell_families.preferred)) != {family}
        or target.cell_families.allow_mixed
    ):
        issues.append(SupportIssue("cell-families", "the periodic simplex cell family"))
    if not specification.size_controls or any(
        not isinstance(control, UniformSizeControl)
        for control in specification.size_controls
    ):
        issues.append(SupportIssue("size-control", "uniform periodic size controls"))
    regions = np.unique(source.cell_regions)
    scope = specification_scope(specification)
    if (
        scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
    ):
        issues.append(
            SupportIssue("region-identity", "the bound periodic source revision")
        )
    if isinstance(specification, SurfaceMeshingSpec) and (
        scope.entity_dimension != dimension
        or not np.array_equal(np.asarray(scope.entity_ids), regions)
    ):
        issues.append(
            SupportIssue("region-scope", "the complete represented periodic region scope")
        )
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        if control.scope.entity_dimension != dimension or not np.array_equal(
            np.asarray(control.scope.entity_ids), regions
        ):
            issues.append(
                SupportIssue(
                    "size-control", "size controls over all represented periodic regions"
                )
            )
    for feature in specification.protected_features:
        match source.domain:
            case periodic_owner.PeriodicPointOrbits():
                issues.append(
                    SupportIssue(
                        "feature-identity", "represented periodic feature orbits"
                    )
                )
            case cell_mesh_owner.CellMesh() as domain:
                match feature.feature_kind:
                    case (
                        FeatureKind.CORNER
                        | FeatureKind.CURVE
                        | FeatureKind.SURFACE
                        | FeatureKind.MATERIAL_INTERFACE
                    ):
                        if feature.scope.entity_dimension >= dimension or np.any(
                            np.asarray(feature.scope.entity_ids)
                            >= periodic_owner.require_periodic_topology(domain)
                            .quotient.entities(feature.scope.entity_dimension)
                            .count
                        ):
                            issues.append(
                                SupportIssue(
                                    "feature-scope",
                                    "feature scopes naming source quotient entities",
                                )
                            )
                    case _:
                        issues.append(
                            SupportIssue(
                                "feature-identity", "represented periodic feature orbits"
                            )
                        )
            case _ as invalid:
                assert_never(invalid)
    for control in specification.region_controls:
        if not control.meshing_enabled or not np.all(
            np.isin(np.asarray(control.scope.entity_ids), regions)
        ):
            issues.append(
                SupportIssue("disabled-region", "enabled represented material regions")
            )
    if isinstance(specification, VolumeMeshingSpec) and (
        specification.region_seeds or specification.hole_seeds
    ):
        issues.append(
            SupportIssue(
                "region-semantics",
                "regions and holes represented by the source quotient complex, not unbound seeds",
            )
        )
    if specification.patch_controls or specification.layer_controls:
        issues.append(SupportIssue("layer-route", "periodic patch/layer construction"))
    # Identification is already part of the source topology. Additional boundary
    # constraints must not silently overwrite that scientific identity.
    if specification.periodic_constraints and (
        not isinstance(source.domain, cell_mesh_owner.CellMesh)
        or any(
            constraint.source_entity_ids is None
            for constraint in specification.periodic_constraints
        )
    ):
        issues.append(
            SupportIssue(
                "periodic-recovery",
                "explicit boundary-entity orbits of the represented periodic source",
            )
        )
    return tuple(issues)


def _bound_constraint_evidence(
    source: NativePeriodicSource, constraints: tuple[PeriodicConstraint, ...]
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Check explicit source-geometry seam pairs against the bound quotient.

    The relative transform is a native right solve, with failure diagnostics
    reaching the provider refusal instead of applying an explicit inverse.
    """

    requested, achieved = [], []
    if not constraints:
        return requested, achieved
    mesh = source.domain
    match mesh:
        case cell_mesh_owner.CellMesh():
            pass
        case periodic_owner.PeriodicPointOrbits():
            raise ValueError(
                "periodic.constraints.source must be a represented lifted CellMesh; got a point-orbit seed"
            )
        case _ as invalid:
            assert_never(invalid)
    topology = periodic_owner.require_periodic_topology(mesh)
    solve_plan = SmallLinearSolvePlan(mesh.ambient_dimension + 1)
    for constraint in constraints:
        degree = constraint.source_scope.entity_dimension
        if degree >= mesh.topological_dimension:
            raise ValueError(
                "periodic.constraints.scope must name boundary vertices, edges, or faces; got a full-dimensional scope"
            )
        entities = mesh.topology.entities(degree)
        identifiers = np.asarray(entities.entity_ids)
        lookup = {int(identifier): row for row, identifier in enumerate(identifiers)}
        first = np.asarray(
            [
                lookup.get(int(value), -1)
                for value in np.asarray(constraint.source_entity_ids)
            ],
            dtype=np.int64,
        )
        if np.any(first < 0):
            raise ValueError(
                "periodic.constraints.entity_ids must name retained lifted entities; got an unknown entity ID"
            )
        second = np.asarray(
            [
                lookup.get(int(value), -1)
                for value in np.asarray(constraint.target_scope.entity_ids)
            ],
            dtype=np.int64,
        )
        if np.any(second < 0):
            raise ValueError(
                "periodic.constraints.entity_ids must name retained lifted entities; got an unknown entity ID"
            )
        orbit, orientations, _ = (np.asarray(value) for value in topology.orbits(degree))
        if np.any(orbit[first] != orbit[second]):
            raise ValueError(
                "periodic.constraints.entity_pairs must share quotient entity orbits; got different orbit IDs"
            )
        if degree and np.any(
            np.asarray(constraint.orientations)
            != orientations[first] * orientations[second]
        ):
            raise ValueError(
                "periodic.constraints.orientations must agree with the bound entity orientations; got a conflicting orientation"
            )
        maps = topology.orbit_isometries(degree)
        solved = solve_small_linear(
            solve_plan,
            np.swapaxes(maps[first], -1, -2),
            np.swapaxes(maps[second], -1, -2),
        )
        if not np.all(np.asarray(solved.successful)):
            raise ValueError(
                "periodic.constraints.transform must yield a successful native relative-isometry solve; got "
                f"status={np.asarray(solved.status)}, "
                f"condition={np.asarray(solved.condition_estimate)}, "
                f"residual={np.asarray(solved.residual_norm)}."
            )
        expected = np.swapaxes(np.asarray(solved.value), -1, -2)
        transform = np.asarray(constraint.transform)
        if transform.shape != expected.shape[1:]:
            raise ValueError(
                "periodic.constraints.transform must match the source ambient dimension; got an incompatible transform shape"
            )
        residual = float(np.max(np.abs(expected - transform)))
        if residual > constraint.tolerance:
            raise ValueError(
                "periodic.constraints.transform must agree with the bound isometry cycle within tolerance; got an excessive residual"
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


type PeriodicConstructionMethod = Literal[
    "orbit_lattice_algebraic_bands",
    "source_action_algebraic_bands",
    "orbit_sites",
    "represented_source",
]


@final
class PreparedPeriodicDomain(StrictModule, NonTrainableState):
    """Admit one periodic construction from exact source and request premises.

    The algebraic constructions are selected only when their capability proofs
    pass; proof resource refusals propagate. Execution dispatches on the bound
    selection and refuses a changed premise instead of choosing another route.
    """

    source: NativePeriodicSource
    specification_id: str = eqx.field(static=True)
    construction: PeriodicConstructionMethod = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: NativePeriodicSource,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        /,
    ) -> None:
        from ._native_periodic_algebraic import (
            prove_orbit_lattice_bands,
            prove_source_action_bands,
        )

        issues = periodic_support_issues(source, specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                format_support_issues(issues),
            )
        match source.domain:
            case periodic_owner.PeriodicPointOrbits():
                construction = (
                    "orbit_lattice_algebraic_bands"
                    if prove_orbit_lattice_bands(source, specification) is not None
                    else "orbit_sites"
                )
            case cell_mesh_owner.CellMesh():
                construction = (
                    "source_action_algebraic_bands"
                    if prove_source_action_bands(source, specification)
                    else "represented_source"
                )
            case _ as invalid:
                assert_never(invalid)
        self.source = source
        self.specification_id = specification.specification_id
        self.construction = parse(
            construction,
            PeriodicConstructionMethod,
            "PreparedPeriodicDomain.construction",
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-domain",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "construction": self.construction,
            }
        )


def _periodic_entity_rows(mesh: cell_mesh_owner.CellMesh, degree: int, /) -> np.ndarray:
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
    return periodic_owner.simplex_entity_corners(mesh, degree)


def _periodic_region_incidence(
    mesh: cell_mesh_owner.CellMesh,
    regions: np.ndarray,
    /,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Material incidence of every quotient entity, including seam copies."""
    topology = periodic_owner.require_periodic_topology(mesh)
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
    match domain:
        case cell_mesh_owner.CellMesh():
            pass
        case periodic_owner.PeriodicPointOrbits():
            raise TypeError(
                "periodic.source_strata.domain must be a represented CellMesh; got a point-orbit seed"
            )
        case _ as invalid:
            assert_never(invalid)
    dimension = domain.topological_dimension
    topology = periodic_owner.require_periodic_topology(domain)
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
    """One support phase borrows the native ledger without renewing it."""

    def __init__(
        self,
        limits: MeshingLimits,
        /,
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        self.limits, self.work = limits, 0
        self.scope = scope
        self.budget = budget
        self.ticket: BatchTicket | None = budget.ticket
        self.started = scope.operation_started

    def spend(self, count: int, /) -> None:
        if self.work + count > self.limits.maximum_work_units:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.source_support.work must fit the declared work allowance; got a request beyond the allowance",
                requested=(("maximum_work_units", self.limits.maximum_work_units),),
                achieved=(
                    ("source_support_work", self.work),
                    ("source_support_request", count),
                ),
            )
        if self.work + count > self.limits.maximum_geometry_queries:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.source_support.queries must fit the declared query allowance; got a request beyond the allowance",
                requested=(
                    ("maximum_geometry_queries", self.limits.maximum_geometry_queries),
                ),
                achieved=(("source_support_query_bound", self.work + count),),
            )
        check_deadline(self.started, self.limits, MeshingStageKind.GEOMETRY_ASSOCIATION)
        if self.ticket is None:
            raise RuntimeError(
                "periodic.source_support.ticket must be admitted before consumption; got no phase ticket"
            )
        if (
            count > self.ticket.remaining_work_units
            or count > self.ticket.remaining_geometry_queries
        ):
            self.ticket.refuse(count, queries_upper=count)
        self.ticket.consume(count, geometry_queries=count)
        self.work += count

    def scratch(self, bound: int, /) -> None:
        ticket = self.ticket
        managed = (
            ticket.bytes_upper
            if ticket is not None and not ticket.settled
            else self.scope.remaining().remaining_managed_bytes
        )
        available = min(self.limits.maximum_scratch_bytes, managed)
        if (
            bound <= self.limits.maximum_scratch_bytes
            and bound > available
            and ticket is not None
            and not ticket.settled
        ):
            ticket.refuse(0, bytes_upper=bound)
        if bound > available:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.source_support.storage must fit the remaining scratch allowance; got a request beyond the allowance",
                requested=(
                    ("maximum_scratch_bytes", self.limits.maximum_scratch_bytes),
                    ("remaining_scratch_bytes", available),
                ),
                achieved=(("source_support_scratch_bound", bound),),
            )

    def batch(self, upper: int, /) -> BatchTicket:
        if not self.budget.ticket.settled:
            parent = self.budget.ticket
            if (
                upper > parent.remaining_work_units
                or upper > parent.remaining_geometry_queries
            ):
                parent.refuse(upper, queries_upper=upper)
        owner = self.scope if self.budget.ticket.settled else self.budget.ticket
        self.ticket = owner.batch(upper, queries_upper=upper)
        return self.ticket

    def exact_phase(self) -> AbstractContextManager[CoordinateEnclosureBudget]:
        if not self.budget.ticket.settled:
            self.ticket = self.budget.ticket
            return nullcontext(self.budget)
        remaining = self.scope.remaining()
        self.ticket = self.scope.batch(
            min(self.limits.maximum_work_units, remaining.remaining_work_units),
            queries_upper=min(
                self.limits.maximum_geometry_queries,
                remaining.remaining_geometry_queries,
            ),
            bytes_upper=min(
                self.limits.maximum_scratch_bytes, remaining.remaining_managed_bytes
            ),
        )
        return self.budget.bind_ticket(self.ticket)


def _periodic_source_frame(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    allowance: _PeriodicSupportAllowance,
    /,
    *,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
) -> PeriodicImageFrame:
    with ExitStack() as stack:
        stack.enter_context(allowance.budget.measurement())
        original, successor = (
            periodic_owner.require_periodic_topology(source),
            periodic_owner.require_periodic_topology(target),
        )
        if identification_id(original.cell) != identification_id(successor.cell):
            raise ValueError(
                "periodic.source_support.identification must match the source identification group; got a changed identification"
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
        with allowance.batch(point_count + group_images):
            allowance.spend(point_count + group_images)
        stack.enter_context(allowance.exact_phase())
        allowance.budget.reserve(0, scratch)
        coordinates = np.concatenate(
            (np.asarray(source.coordinates), np.asarray(target.coordinates))
        )
        roots = np.concatenate(
            (
                np.asarray(original.vertex_representatives),
                source.coordinates.shape[0]
                + np.asarray(successor.vertex_representatives),
            )
        )
        shifts = np.concatenate(
            (np.asarray(original.vertex_shifts), np.asarray(successor.vertex_shifts))
        )
        exact_coordinates = (
            None
            if geometry is None
            else periodic_owner.periodic_exact_coordinate_bank(
                source, None, budget=allowance.budget
            )
            + periodic_owner.periodic_exact_coordinate_bank(
                target, geometry, budget=allowance.budget
            )
        )
        try:
            image_exponents = None
            if isinstance(original.cell, PeriodicIsometryGroup) and any(
                order == 0 for order in original.cell.orders
            ):
                boxes = periodic_owner.affine_periodic_source_support_boxes(
                    coordinates,
                    roots,
                    shifts,
                    original.cell,
                    budget=allowance.budget,
                    exact_coordinates=exact_coordinates,
                )
                image_exponents = periodic_overlap_image_exponents(
                    original.cell,
                    boxes,
                    allowance.limits.maximum_vertices,
                    budget=allowance.budget,
                )
                actual_images = image_exponents.shape[0]
                additional_storage = (
                    1024 * (actual_images - group_images) * (dimension + 1) ** 2
                )
                allowance.scratch(scratch + additional_storage)
                allowance.budget.reserve(0, additional_storage)
                # Image storage is bounded by the explicit image-preparation owner.
                allowance.spend(actual_images - group_images)
            return prepare_periodic_image_frame(
                coordinates,
                roots,
                shifts,
                original.cell,
                allowance.limits.maximum_vertices,
                image_exponents=image_exponents,
                exact_coordinates=exact_coordinates,
                budget=allowance.budget,
            )
        except PeriodicEmbeddingResourceError as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            ) from error


def _periodic_row_enclosures(
    corners: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose each field-valued corner row with directed binary64 coordinate bounds."""
    budget.reserve(corners.size, 16 * corners.shape[0] * corners.shape[2])

    def endpoint(value: Fraction | RadicalElement, upper: bool) -> float:
        # Only the binary64 result survives each scalar enclosure. The interval
        # refinement fractions must not accumulate across unrelated rows.
        with budget.temporary_scope():
            return coefficient_endpoint(value, upper=upper, saturate=True, budget=budget)

    bounds = [
        np.asarray(
            [
                [endpoint(value, upper) for value in row]
                for row in (np.max(corners, axis=1) if upper else np.min(corners, axis=1))
            ],
            dtype=np.float64,
        )
        for upper in (False, True)
    ]
    return bounds[0], bounds[1]


def _periodic_source_regions(
    source: NativePeriodicSource,
    target: cell_mesh_owner.CellMesh,
    frame: PeriodicImageFrame,
    allowance: _PeriodicSupportAllowance,
    /,
) -> np.ndarray:
    domain = source.domain
    match domain:
        case cell_mesh_owner.CellMesh():
            pass
        case periodic_owner.PeriodicPointOrbits():
            raise TypeError(
                "periodic.material_support.domain must be a represented CellMesh; got a point-orbit seed"
            )
        case _ as invalid:
            assert_never(invalid)
    source_rows = _periodic_entity_rows(domain, domain.topological_dimension)
    target_rows = _periodic_entity_rows(target, target.topological_dimension)
    count = domain.coordinates.shape[0]
    target_corners = frame.points[count:][target_rows]
    algebraic = any(isinstance(value, RadicalElement) for value in target_corners.flat)
    target_lower, target_upper = (
        _periodic_row_enclosures(target_corners, budget=allowance.budget)
        if algebraic
        else (np.min(target_corners, axis=1), np.max(target_corners, axis=1))
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
    # _periodic_source_frame reserves the SAT/support workspace for this phase.
    # Borrow that storage while charging each prepared action's field work.
    target_sat = (
        tuple(
            affine_mapped_contact_owner.prepare_triangle_sat(
                triangle, budget=allowance.budget, storage_precharged=True
            )
            for triangle in target_corners
        )
        if target.topological_dimension == 2
        else ()
    )
    for image in frame.images():
        with allowance.budget.temporary_scope():
            allowance.spend(source_rows.shape[0] * target_rows.shape[0])
            corners = frame.image_points(image, slice(0, count), budget=allowance.budget)[
                source_rows
            ]
            lower, upper = (
                _periodic_row_enclosures(corners, budget=allowance.budget)
                if algebraic
                else (np.min(corners, axis=1), np.max(corners, axis=1))
            )
            source_sat = (
                tuple(
                    affine_mapped_contact_owner.prepare_triangle_sat(
                        triangle, budget=allowance.budget, storage_precharged=True
                    )
                    for triangle in corners
                )
                if target_sat
                else ()
            )
            for row, triangle in enumerate(target_corners):
                possible = np.flatnonzero(
                    np.all(
                        (target_upper[row] > lower) & (upper > target_lower[row]),
                        axis=1,
                    )
                )
                for parent in possible:
                    overlap = (
                        affine_mapped_contact_owner.prepared_triangle_interiors_overlap(
                            target_sat[row],
                            source_sat[parent],
                            budget=allowance.budget,
                            storage_precharged=True,
                        )
                        if target_sat
                        else affine_mapped_contact_owner.interiors_overlap(
                            triangle, corners[parent]
                        )
                    )
                    if overlap:
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
            "periodic.material_support.cells must retain one authored material stratum per cell; got unresolved material membership",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=unresolved,
        )
    return np.asarray([next(iter(material)) for material in members], dtype=np.int64)


def _periodic_seed_associations(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
) -> tuple[GeometryAssociation, ...]:
    with ExitStack() as stack:
        domain = source.domain
        match domain:
            case periodic_owner.PeriodicPointOrbits():
                pass
            case cell_mesh_owner.CellMesh():
                raise TypeError(
                    "periodic.seed_support.domain must be PeriodicPointOrbits; got a represented CellMesh"
                )
            case _ as invalid:
                assert_never(invalid)
        topology = periodic_owner.require_periodic_topology(target)
        if identification_id(topology.cell) != identification_id(domain.cell):
            raise ValueError(
                "periodic.seed_support.identification must match the source identification; got a changed identification"
            )
        allowance = _PeriodicSupportAllowance(
            specification.limits, scope=scope, budget=budget
        )
        # A declared field-valued geometry integrates its exact orbit measures on
        # the request ledger; binary64 carriers need no exact phase. Only static
        # counts and measures survive, so the proof's temporaries end here.
        with (
            allowance.exact_phase() if geometry is not None else nullcontext(),
            budget.temporary_scope(),
        ):
            quotient = periodic_owner.PeriodicQuotientEvidence(
                target, geometry=geometry, budget=budget
            )
        if (
            not quotient.all_cells_valid
            or quotient.boundary_facet_count
            or quotient.relative_coverage_defect is None
            or quotient.relative_coverage_defect > 1.0e-10
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "periodic.seed_support.coverage must cover the source lattice region once with positive cells; got an invalid quotient cover",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            )
        periodic_owner.certify_periodic_embedding(
            target,
            maximum_images=specification.limits.maximum_vertices,
            maximum_pairs=specification.limits.maximum_work_units,
            scope=scope,
            budget=allowance.budget,
            geometry=geometry,
        )
        stack.enter_context(
            allowance.batch(
                sum(
                    target.entity_set(degree).count
                    for degree in range(target.topological_dimension + 1)
                )
            )
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
                        entity_key(
                            source.source_revision,
                            GeometrySourceEntityRole.REGION.value,
                            0,
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


@final
class _PeriodicSourceAssociationPreparation(StrictModule):
    """Retain source-membership proof inputs and associated vertex strata."""

    transfer_source: NativePeriodicSource
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    result: CellMeshingResult
    association: GeometryAssociation
    associated_degrees: tuple[int, ...]

    @classmethod
    def from_stored_fields(
        cls, value: _PeriodicSourceAssociationPreparation, /
    ) -> _PeriodicSourceAssociationPreparation:
        """Restore membership data without registering a live proof identity."""
        return cls(
            value.transfer_source,
            value.specification,
            value.result,
            value.association,
            value.associated_degrees,
        )

    @property
    def prepared_id(self) -> str:
        return self.result.result_id


@checked
def periodic_source_associations(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    _prepared_cover: _PeriodicSourceAssociationPreparation | None = None,
) -> tuple[GeometryAssociation, ...]:
    """Keep source support storage alive through the membership phase."""

    try:
        with budget.measurement(), budget.temporary_scope():
            return _periodic_source_membership(
                source,
                specification,
                target,
                scope=scope,
                budget=budget,
                geometry=geometry,
                _prepared_cover=_prepared_cover,
            )
    except CoordinateEnclosureResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.source_support.resources must fit the remaining work and storage allowances; got a resource refusal",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            requested=(
                (str(error.resource), float(error.requested)),
                ("limit", float(error.limit)),
            ),
            achieved=(("completed", float(error.completed)),),
        ) from error


def _periodic_source_membership(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    _prepared_cover: _PeriodicSourceAssociationPreparation | None = None,
) -> tuple[GeometryAssociation, ...]:
    with ExitStack() as stack:
        domain = source.domain
        match domain:
            case periodic_owner.PeriodicPointOrbits():
                return _periodic_seed_associations(
                    source,
                    specification,
                    target,
                    scope=scope,
                    budget=budget,
                    geometry=geometry,
                )
            case cell_mesh_owner.CellMesh():
                pass
            case _ as invalid:
                assert_never(invalid)
        allowance = _PeriodicSupportAllowance(
            specification.limits, scope=scope, budget=budget
        )
        covered = (
            _prepared_cover is not None
            and _prepared_cover.transfer_source is source
            and _prepared_cover.specification is specification
            and native_preparation_ledger(scope).live_contains(_prepared_cover)
        )
        if not covered:
            original = periodic_owner.PeriodicQuotientEvidence(domain)
            if not original.all_cells_valid or (
                original.relative_coverage_defect is not None
                and (
                    original.boundary_facet_count
                    or original.relative_coverage_defect > 1.0e-10
                )
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.AUDIT_FAILED,
                    "periodic.source_support.coverage must have positive cells and closed quotient coverage; got an invalid source cover",
                )
        frame = _periodic_source_frame(domain, target, allowance, geometry=geometry)
        if not covered:
            with allowance.batch(domain.blocks[0].cell_count ** 2 * frame.image_count):
                allowance.spend(domain.blocks[0].cell_count ** 2 * frame.image_count)
            periodic_owner.certify_periodic_embedding(
                domain,
                maximum_images=specification.limits.maximum_vertices,
                maximum_pairs=specification.limits.maximum_work_units,
                scope=scope,
                budget=allowance.budget,
            )
        stack.enter_context(allowance.exact_phase())
        regions = _periodic_source_regions(source, target, frame, allowance)
        incidence = _periodic_region_incidence(target, regions)
        strata = _periodic_source_strata(source, specification)
        topology = periodic_owner.require_periodic_topology(domain)
        count, dimension = domain.coordinates.shape[0], domain.topological_dimension
        roles = (
            GeometrySourceEntityRole.VERTEX,
            GeometrySourceEntityRole.EDGE,
            GeometrySourceEntityRole.FACET,
        )
        associations: list[GeometryAssociation] = []
        # Keys index this frame's fixed source strata and target point bank.
        # Entity support is the conjunction of its vertices' source support.
        point_masks: dict[tuple[int, int, int], np.ndarray] = {}
        points = frame.points[count:]
        for degree in range(dimension + 1):
            rows = _periodic_entity_rows(target, degree)
            classes = np.full(rows.shape[0], dimension, dtype=np.int8)
            indices = np.asarray(
                [
                    material[0] if len(material) == 1 else -1
                    for material in incidence[degree]
                ],
                dtype=np.int64,
            )
            entity_roles = [GeometrySourceEntityRole.REGION for _ in rows]
            ambiguous = np.zeros(rows.shape[0], dtype=np.bool_)
            for source_degree in range(degree, dimension):
                source_rows = _periodic_entity_rows(domain, source_degree)
                source_orbits = np.asarray(topology.orbits(source_degree)[0])
                selected = np.flatnonzero(np.isin(source_orbits, strata[source_degree]))
                for image_index, image in enumerate(frame.images()):
                    allowance.spend(rows.shape[0] * selected.size)
                    imaged = frame.image_points(
                        image, slice(0, count), budget=allowance.budget
                    )
                    for parent in selected:
                        key = source_degree, image_index, int(parent)
                        point_mask = point_masks.get(key)
                        if point_mask is None:
                            budget.reserve(0, 512 + points.shape[0])
                            point_mask = np.empty(points.shape[0], dtype=np.bool_)
                            for begin in range(0, points.shape[0], 32):
                                end = min(begin + 32, points.shape[0])
                                point_mask[begin:end] = (
                                    periodic_owner.source_entity_contains(
                                        points[begin:end, None, :],
                                        imaged[source_rows[parent]],
                                    )
                                )
                            point_masks[key] = point_mask
                        mask = np.all(point_mask[rows], axis=1)
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
                    "periodic.source_support.membership must resolve each target entity to one source stratum; got ambiguous or absent source membership",
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                    entity_ids=tuple(
                        int(value)
                        for value in np.asarray(target.entity_set(degree).entity_ids)[bad]
                    ),
                )
            source_ids = tuple(
                entity_key(source.source_revision, role.value, int(index))
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
class _PeriodicSourceAssociationReceipt(StrictModule):
    """Bind ended preparation charges to retained source proof identities."""

    proofs: tuple[_PeriodicSourceAssociationPreparation, ...]
    evidence: NativeExecutionRecord
    expires_at: float = eqx.field(static=True)

    @classmethod
    def from_stored_fields(
        cls, value: _PeriodicSourceAssociationReceipt, /
    ) -> _PeriodicSourceAssociationReceipt:
        """Restore charges with proof copies detached from live authentication."""
        return cls(
            tuple(
                _PeriodicSourceAssociationPreparation(
                    proof.transfer_source,
                    proof.specification,
                    proof.result,
                    proof.association,
                    proof.associated_degrees,
                )
                for proof in value.proofs
            ),
            value.evidence,
            value.expires_at,
        )


@final
class PeriodicAssociationTransfer(StrictModule, NonTrainableState):
    """Renew membership in the unchanged authored periodic geometry."""

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
            raise ValueError(format_support_issues(issues))
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
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        _prepared_source: list[_PeriodicSourceAssociationPreparation] | None = None,
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        allowance = _PeriodicSupportAllowance(
            self.specification.limits, scope=scope, budget=budget
        )
        for prepared in () if _prepared_source is None else _prepared_source:
            if (
                prepared.transfer_source is self.source
                and prepared.specification is self.specification
                and prepared.result is source
                and native_preparation_ledger(scope).live_contains(prepared)
            ):
                with allowance.exact_phase():
                    budget.ticket.checkpoint()
                return prepared.association, prepared.associated_degrees
        # The classifier returns only a verdict. Its per-cell corner evaluation
        # scratch must not overlap the subsequent full source-support proof;
        # coordinate/basis caches keep their separately retained ownership.
        with allowance.exact_phase(), budget.temporary_scope():
            affine = is_affine_cell_geometry(source.mesh, source.geometry, budget=budget)
        if not affine:
            raise ValueError(
                "periodic.transfer.source_geometry must be the affine quotient map; got nonaffine geometry"
            )
        dimension = source.mesh.topological_dimension
        by_dimension = {
            target_dimension(source.mesh, item): item for item in source.associations
        }
        if len(by_dimension) != len(source.associations) or set(by_dimension) != set(
            range(dimension + 1)
        ):
            raise ValueError(
                "periodic.transfer.associations must have one source association table per dimension; got missing or duplicate dimensions"
            )
        expected = periodic_source_associations(
            self.source,
            self.specification,
            source.mesh,
            scope=scope,
            budget=budget,
            geometry=source.geometry,
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
                or tuple(item.source_entities.entity_id(row) for row in rows)
                != proof.source_entities.entity_ids()
                or not np.array_equal(
                    np.asarray(item.source_dimensions)[rows],
                    np.asarray(proof.source_dimensions),
                )
                or not np.array_equal(
                    np.asarray(item.source_indices)[rows],
                    np.asarray(proof.source_indices),
                )
                or not item.source_entities.indexed_roles
                or tuple(item.source_entities.role(row) for row in rows)
                != proof.source_entities.roles()
                or np.any(np.asarray(item.residuals)[rows] != 0.0)
            ):
                raise ValueError(
                    "periodic.transfer.associations must agree with the unchanged source strata; got inconsistent source membership"
                )
        association, degrees = by_dimension[0], tuple(range(1, dimension + 1))
        if _prepared_source is not None:
            with allowance.exact_phase():
                # Retain proof identity in the request ledger; execution may
                # borrow it only while the charged root request remains live.
                budget.reserve(1, 512)
                prepared = _PeriodicSourceAssociationPreparation(
                    self.source,
                    self.specification,
                    source,
                    association,
                    degrees,
                )
            native_preparation_ledger(scope).remember_live(scope, prepared)
            _prepared_source.append(prepared)
        return association, degrees

    @checked
    def classes(
        self,
        source: CellMeshingResult,
        /,
        *,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        _prepared_source: list[_PeriodicSourceAssociationPreparation] | None = None,
    ) -> tuple[PlcEntityClasses, ...]:
        self.source_associations(
            source, scope=scope, budget=budget, _prepared_source=_prepared_source
        )
        domain = self.source.domain
        dimension = self.source.ambient_dimension
        namespaces: list[tuple[int, ...]]
        match domain:
            case cell_mesh_owner.CellMesh():
                topology = periodic_owner.require_periodic_topology(domain)
                namespaces = [
                    tuple(range(topology.quotient.entities(degree).count))
                    for degree in range(dimension)
                ]
            case periodic_owner.PeriodicPointOrbits():
                namespaces = [() for _ in range(dimension)]
            case _ as invalid:
                assert_never(invalid)
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
                if target_dimension(source.mesh, item) == degree
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
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: cell_geometry_owner.CellGeometrySpec,
        scope: NativeExecutionScope,
        budget: CoordinateEnclosureBudget,
        _prepared_source: tuple[_PeriodicSourceAssociationPreparation, ...] = (),
    ) -> tuple[GeometryAssociation, ...]:
        if (lineage.source_topology_id, lineage.target_topology_id) != (
            source.mesh.topology_id,
            target.topology_id,
        ):
            raise ValueError(
                "periodic.transfer.lineage must bind the source and target topology IDs; got a different mesh transition"
            )
        allowance = _PeriodicSupportAllowance(
            self.specification.limits, scope=scope, budget=budget
        )
        with allowance.exact_phase():
            affine = is_affine_cell_geometry(target, geometry, budget=budget)
        if not affine:
            raise ValueError(
                "periodic.transfer.target_geometry must be the admitted affine geometry; got nonaffine geometry"
            )
        self.source_associations(
            source, scope=scope, budget=budget, _prepared_source=list(_prepared_source)
        )
        prepared_cover = next(
            (proof for proof in _prepared_source if proof.result is source), None
        )
        return periodic_source_associations(
            self.source,
            self.specification,
            target,
            scope=scope,
            budget=budget,
            geometry=geometry,
            _prepared_cover=prepared_cover,
        )


class _SiteMesh:
    """Quotient edge bank of one periodic Delaunay triangulation of orbit sites."""

    def __init__(
        self,
        points: np.ndarray,
        vectors: np.ndarray,
        triangulation: PeriodicDelaunayTriangulation
        | constraints_owner.PeriodicStratifiedTriangulation
        | PeriodicRefinement,
        /,
        *,
        source_support: constraints_owner.PeriodicStratifiedTriangulation | None = None,
    ) -> None:
        self.triangulation = triangulation
        self.source_support = (
            triangulation
            if isinstance(
                triangulation, constraints_owner.PeriodicStratifiedTriangulation
            )
            else source_support
        )
        self.protected_vertices = np.zeros(points.shape[0], dtype=np.bool_)
        self.protected_edges: tuple[tuple[int, ...], ...] = ()
        if self.source_support is not None:
            protected = self.source_support.protected_vertices
            self.protected_vertices[: protected.shape[0]] = protected
            self.protected_edges = self.source_support.protected_edges
        self.sizing_evidence: tuple[PeriodicSizeRecord, ...] = ()
        self.image_refusals = 0
        simplices = np.asarray(
            triangulation.vertices
            if isinstance(triangulation, PeriodicRefinement)
            else triangulation.simplices,
            dtype=np.int64,
        )
        shifts = np.asarray(
            triangulation.shifts
            if isinstance(triangulation, PeriodicRefinement)
            else triangulation.simplex_shifts,
            dtype=np.int64,
        )
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
            # The translation-invariant key of `edge_key`: leader, follower, relative shift.
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
        self.reference_lifts = periodic_reference_lifts(
            simplices, shifts, points.shape[0]
        )
        self.lengths = quotient_edge_lengths(
            points, self.keys, self.reference_lifts, vectors
        )

    def merit(
        self,
        controls: list[UniformSizeControl],
        policy: SizeCompliancePolicy,
        /,
        *,
        allowance: _PeriodicSupportAllowance,
        quadratic: bool = False,
    ) -> tuple[int, int, float, int]:
        """Hard issues, over-target edges, statistic deviation, and target-length edges.

        Hard issues are the compliance owner's unrounded verdict on the lengths these
        sites would publish; the remaining terms rank binary64 representatives.
        """
        growth = edge_growth_evidence(
            self.lengths, self.keys[:, :2], self.points.shape[0]
        )
        budget = allowance.budget
        with allowance.exact_phase(), budget.temporary_scope():
            squared = periodic_owner.periodic_site_squared_lengths(
                self.points, self.keys, self.reference_lifts, self.vectors, budget=budget
            )
            issues = sum(
                len(
                    periodic_owner.periodic_size_control_issues(
                        squared, self.keys[:, :2], control, policy, budget=budget
                    )
                )
                for control in controls
            )
        longer, deviation, matched = 0, 0.0, 0
        for control in controls:
            _, achieved, _ = uniform_size_compliance(
                control, policy, self.lengths, growth
            )
            tolerance = policy.tolerance(control.target_size)
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


def _source_growth_site(
    state: _SiteMesh,
    controls: list[UniformSizeControl],
    policy: SizeCompliancePolicy,
    /,
    *,
    allowance: _PeriodicSupportAllowance,
) -> np.ndarray | None:
    """Remove unprotected fixed long edges, then make immutable quantiles feasible.

    Edges exceed the target bound by their unrounded published lengths; among
    unprotected fixed candidates the exactly longest, then lowest row, splits.
    """
    if state.source_support is None:
        raise TypeError(
            "periodic.site_growth.source_support must retain source strata; got no constrained construction"
        )
    protected_edges = set(state.protected_edges)
    threshold = min(
        control.target_size + policy.tolerance(control.target_size)
        for control in controls
    )
    fixed = np.all(state.protected_vertices[state.keys[:, :2]], axis=1)
    protected = np.asarray(
        [tuple(int(value) for value in row) in protected_edges for row in state.keys],
        dtype=np.bool_,
    )
    budget = allowance.budget
    with allowance.exact_phase(), budget.temporary_scope():
        squared = periodic_owner.periodic_site_squared_lengths(
            state.points, state.keys, state.reference_lifts, state.vectors, budget=budget
        )
        over = periodic_owner.periodic_lengths_exceeding(
            squared, threshold, budget=budget
        )
        candidates = np.flatnonzero(fixed & over & ~protected)
        (longest,) = periodic_owner.periodic_longest_positions(
            [[squared[index] for index in candidates]], budget=budget
        )
    if candidates.size:
        key = tuple(int(value) for value in state.keys[candidates[longest]])
        return 0.5 * (
            state.points[key[0]]
            + state.points[key[1]]
            + np.asarray(key[2:], dtype=np.int64) @ state.vectors
        )
    immutable_over = int(np.count_nonzero(over & protected))
    quantile = max(size_statistic_quantile(name) for name in policy.target_statistics)
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
        "periodic.site_growth.cavity must satisfy the immutable rank budget; got no admissible interior cavity",
    )


type _SiteMerit = tuple[int, int, float, int]
type _SiteScore = tuple[float, int, float, int]


def _site_merit_score(merit: _SiteMerit, /) -> _SiteScore:
    """Prioritize statistic deviation for protected source-stratum proposals."""
    return merit[2], merit[0], float(merit[1]), merit[3]


def _triangulate_target_size_sites(
    points: np.ndarray,
    vectors: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell,
    limits: MeshingLimits,
    stage: MeshingStageKind,
    image_budget: int,
    work: int,
    /,
    *,
    scope: NativeExecutionScope,
    source_strata: constraints_owner.PeriodicSourceStrata | None,
) -> tuple[_SiteMesh, int]:
    """Construct one candidate and settle its work before the next candidate."""
    if source_strata is not None:
        constrained = constraints_owner.triangulate_periodic_source_strata(
            points,
            source_strata,
            limits,
            image_budget,
            limits.maximum_work_units - work,
            scope=scope,
        )
        work += constrained.work_units
        return _SiteMesh(points, vectors, constrained), work
    # Size native scratch to this candidate before escalating a refusal to limits.
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
                "periodic.site_insertion.work must fit the declared work budget; got a request beyond the budget",
                stage=stage.value,
            )
        return _SiteMesh(points, vectors, triangulation), work
    raise RuntimeError(
        "periodic.site_insertion.triangulation must produce a result within the admitted escalation; got no result"
    )


def _relocate_target_size_sites(
    state: _SiteMesh,
    controls: list[UniformSizeControl],
    policy: SizeCompliancePolicy,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    size: float,
    work: int,
    current_score: _SiteScore,
    /,
    *,
    source_binding_id: str,
    scope: NativeExecutionScope,
    allowance: _PeriodicSupportAllowance,
) -> tuple[tuple[_SiteMerit, _SiteMesh] | None, int]:
    """Retain proposal/refusal evidence before accepting an improving relocation."""
    try:
        proposal = propose_periodic_site_relocation(
            state.points,
            state.keys,
            state.simplices,
            state.simplex_shifts,
            state.vectors,
            state.protected_vertices,
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
            scope=scope,
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
        corrected = _SiteMesh(
            proposal.points,
            state.vectors,
            state.triangulation,
            source_support=state.source_support,
        )
        measured = corrected.merit(controls, policy, allowance=allowance, quadratic=True)
        if _site_merit_score(measured) < current_score:
            corrected.sizing_evidence = state.sizing_evidence
            corrected.image_refusals = state.image_refusals
            return (measured, corrected), work
    return None, work


def _refine_target_size_sites(
    state: _SiteMesh,
    cell: periodic_cell_owner.PeriodicCell,
    controls: list[UniformSizeControl],
    policy: SizeCompliancePolicy,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    image_budget: int,
    work: int,
    /,
    *,
    scope: NativeExecutionScope,
    allowance: _PeriodicSupportAllowance,
) -> tuple[_SiteMesh, int]:
    """Keep one native quotient owner through all read-only candidate batches."""
    control_rows = np.asarray(
        [
            (
                control.target_size,
                np.nan if control.minimum_size is None else control.minimum_size,
                np.nan if control.maximum_size is None else control.maximum_size,
                np.nan
                if control.maximum_growth_rate is None
                else control.maximum_growth_rate,
                float(control.strength is SizeControlStrength.HARD),
            )
            for control in controls
        ],
        dtype=np.float64,
    )
    check_deadline(started, limits, stage)
    remaining = scope.remaining()
    with (
        scope.batch(
            min(limits.maximum_work_units - work, remaining.remaining_work_units),
            cavity_upper=min(
                limits.maximum_cavity_cells, remaining.remaining_cavity_cells
            ),
            bytes_upper=min(
                limits.maximum_scratch_bytes, remaining.remaining_managed_bytes
            ),
        ) as ticket,
        ticket.native_target(),
    ):
        result = periodic_refine_2d(
            state.points,
            state.vectors,
            np.asarray(cell.inverse_vectors, dtype=np.float64),
            state.simplices,
            state.simplex_shifts,
            control_rows,
            np.asarray(
                [size_statistic_quantile(name) for name in policy.target_statistics],
                dtype=np.float64,
            ),
            protected_edges=np.asarray(state.protected_edges, dtype=np.int32).reshape(
                (-1, 4)
            ),
            absolute_tolerance=policy.absolute_tolerance,
            relative_tolerance=policy.relative_tolerance,
            max_vertices=limits.maximum_vertices,
            max_cells=limits.maximum_cells,
            max_images=image_budget,
            max_work_units=limits.maximum_work_units - work,
            max_cavity_cells=limits.maximum_cavity_cells,
            max_scratch_bytes=limits.maximum_scratch_bytes,
        )
    work += result.counters.work_units
    if result.status != MeshcoreStatus.OK:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED
            if result.status is MeshcoreStatus.CAPACITY_EXCEEDED
            else MeshingFailureCategory.INVALID_SOURCE,
            "periodic.site_insertion.native must admit the quotient cavity within the declared controls and limits; got "
            + result.status.name,
            stage=stage.value,
            achieved=(
                ("site_work", work),
                ("site_candidates", result.counters.candidates),
                ("site_commits", result.counters.commits),
                ("site_image_refusals", result.counters.image_refusals),
            ),
        )
    check_deadline(started, limits, stage)
    if not np.array_equal(result.points[: state.points.shape[0]], state.points):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "periodic.site_insertion.points must retain the input root prefix; got changed source roots",
            stage=stage.value,
        )
    refined = _SiteMesh(
        result.points,
        state.vectors,
        result,
        source_support=state.source_support,
    )
    refined.sizing_evidence = state.sizing_evidence
    refined.image_refusals = state.image_refusals + result.counters.image_refusals
    if (
        result.counters.stop_reason == 2
        and refined.merit(controls, policy, allowance=allowance)[0]
    ):
        target = min(control.target_size for control in controls)
        proposal = propose_periodic_rank_placement(
            refined.points,
            refined.keys,
            refined.simplices,
            refined.simplex_shifts,
            refined.vectors,
            refined.protected_vertices,
            target,
            policy.target_statistics,
            limits.maximum_work_units - work,
            limits.maximum_scratch_bytes,
            lambda: check_deadline(started, limits, stage),
            scope=scope,
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
        work += proposal.work_units
        if proposal.points is not None:
            corrected = _SiteMesh(
                proposal.points,
                state.vectors,
                result,
                source_support=refined.source_support,
            )
            corrected.sizing_evidence = refined.sizing_evidence
            corrected.image_refusals = refined.image_refusals
            refined = corrected
        refined.sizing_evidence = (*refined.sizing_evidence, proposal.record)
    return refined, work


def _realize_target_size_sites(
    points: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell,
    controls: list[UniformSizeControl],
    policy: SizeCompliancePolicy,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    image_budget: int,
    /,
    *,
    source_binding_id: str,
    scope: NativeExecutionScope,
    allowance: _PeriodicSupportAllowance,
    source_strata: constraints_owner.PeriodicSourceStrata | None = None,
) -> tuple[_SiteMesh, int]:
    """Realize hard statistics while retaining protected source-support evidence."""
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    state, work = _triangulate_target_size_sites(
        points,
        vectors,
        cell,
        limits,
        stage,
        image_budget,
        0,
        scope=scope,
        source_strata=source_strata,
    )
    state, work = _refine_target_size_sites(
        state,
        cell,
        controls,
        policy,
        limits,
        started,
        stage,
        image_budget,
        work,
        scope=scope,
        allowance=allowance,
    )
    if source_strata is None:
        return state, work
    size = min(control.target_size for control in controls)
    current = state.merit(controls, policy, allowance=allowance, quadratic=True)
    while current[0]:
        if state.points.shape[0] + 1 > limits.maximum_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.site_insertion.vertices must fit the declared vertex budget; got a request beyond the budget",
                stage=stage.value,
            )
        growth = _source_growth_site(state, controls, policy, allowance=allowance)
        if growth is not None:
            enlarged, work = _triangulate_target_size_sites(
                np.concatenate((state.points, growth[None])),
                vectors,
                cell,
                limits,
                stage,
                image_budget,
                work,
                scope=scope,
                source_strata=source_strata,
            )
            enlarged.sizing_evidence = state.sizing_evidence
            enlarged.image_refusals = state.image_refusals
            enlarged, work = _refine_target_size_sites(
                enlarged,
                cell,
                controls,
                policy,
                limits,
                started,
                stage,
                image_budget,
                work,
                scope=scope,
                allowance=allowance,
            )
            current, state = (
                enlarged.merit(controls, policy, allowance=allowance, quadratic=True),
                enlarged,
            )
            continue
        relocated, work = _relocate_target_size_sites(
            state,
            controls,
            policy,
            limits,
            started,
            stage,
            size,
            work,
            _site_merit_score(current),
            source_binding_id=source_binding_id,
            scope=scope,
            allowance=allowance,
        )
        if relocated is None:
            break
        current, state = relocated
    return state, work


@final
class _PeriodicConstruction(StrictModule):
    """Retain constructed cells and their source inventory through certification."""

    mesh: cell_mesh_owner.CellMesh
    original_mesh: cell_mesh_owner.CellMesh
    regions: np.ndarray
    original_regions: np.ndarray
    ancestry: tuple[tuple[int, ...], ...]
    work: int
    inserted_sites: int
    sizing_evidence: tuple[PeriodicSizeRecord, ...]
    image_refusals: int
    stratified: bool
    source_construction_images: int | None
    geometry: cell_geometry_owner.CellGeometrySpec | None = None
    construction_evidence: tuple[tuple[str, int], ...] = eqx.field(
        static=True, default=()
    )


@final
class _PeriodicCertification(StrictModule):
    """Carry quotient and embedding evidence with phase-local work quantities."""

    quotient: periodic_owner.PeriodicQuotientEvidence
    embedding: periodic_owner.PeriodicEmbeddingEvidence
    work_evidence: tuple[tuple[str, float], ...]


@final
class _PeriodicCompliance(StrictModule):
    """Accumulate ordered requested quantities, measurements, and issues."""

    requested: list[tuple[str, float]]
    achieved: list[tuple[str, float]]
    issues: list[str]


def _construct_periodic_seed(
    orbits: periodic_owner.PeriodicPointOrbits,
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    image_budget: int,
    hard: list[UniformSizeControl],
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _PeriodicConstruction:
    work = 0
    inserted_sites = 0
    sizing_evidence: tuple[PeriodicSizeRecord, ...] = ()
    image_refusals = 0
    stratified = False
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
                scope=scope,
                allowance=_PeriodicSupportAllowance(limits, scope=scope, budget=budget),
            )
            inserted_sites = site_mesh.points.shape[0] - orbits.orbit_count
            sizing_evidence = site_mesh.sizing_evidence
            image_refusals = site_mesh.image_refusals
            mesh = periodic_owner.publish_periodic_simplices(
                site_mesh.points,
                site_mesh.simplices,
                site_mesh.simplex_shifts,
                orbits.cell,
                scope=scope,
                budget=budget,
            )
        else:
            try:
                with budget.measurement():
                    construction = periodic_owner.periodic_delaunay_mesh(
                        orbits,
                        maximum_images=image_budget,
                        maximum_simplices=limits.maximum_cells,
                        scope=scope,
                        budget=budget,
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
    return _PeriodicConstruction(
        mesh,
        original_mesh,
        regions,
        original_regions,
        ancestry,
        work,
        inserted_sites,
        sizing_evidence,
        image_refusals,
        stratified,
        None,
    )


def _construct_algebraic_periodic_domain(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    construction: Literal[
        "orbit_lattice_algebraic_bands", "source_action_algebraic_bands"
    ],
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _PeriodicConstruction:
    """Construct and declare the field-valued geometry of an admitted algebraic carrier."""
    with measure_phase(record_phase, "periodic_construction"), budget.temporary_scope():
        starting_work = budget.work_units
        from ._native_periodic_algebraic import (
            construct_exact_algebraic_carrier,
            construct_orbit_lattice_band_carrier,
        )

        try:
            match construction:
                case "orbit_lattice_algebraic_bands":
                    carrier = construct_orbit_lattice_band_carrier(
                        source, specification, scope=scope, budget=budget
                    )
                case "source_action_algebraic_bands":
                    carrier = construct_exact_algebraic_carrier(
                        source, specification, scope=scope, budget=budget
                    )
                case _ as invalid:
                    assert_never(invalid)
        except NativeResourceFailure as error:
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT
                if error.status is MeshcoreStatus.TIMEOUT
                else MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=MeshingStageKind.SURFACE_MESHING.value,
                achieved=(
                    ("native_status", float(error.status)),
                    *tuple(
                        (f"native_execution:{index}", float(value))
                        for index, value in enumerate(error.work_evidence)
                    ),
                    *tuple(
                        (f"native_memory:{name}", float(value))
                        for name, value in error.memory_counters.items()
                    ),
                ),
            ) from error
        except CoordinateEnclosureResourceError as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.algebraic.resources must fit the remaining work and storage allowances; got a resource refusal",
                stage=MeshingStageKind.SURFACE_MESHING.value,
                requested=(
                    (str(error.resource), float(error.requested)),
                    ("limit", float(error.limit)),
                ),
                achieved=(("completed", float(error.completed)),),
            ) from error
        # Seed orbits have no source cells: the carrier is its own region source.
        original: cell_mesh_owner.CellMesh | None
        match source.domain:
            case periodic_owner.PeriodicPointOrbits() as orbits:
                cell = orbits.cell
                block_name = "periodic"
                source_vertices = orbits.orbit_count
                original = None
            case cell_mesh_owner.CellMesh() as domain:
                topology = periodic_owner.require_periodic_topology(domain)
                if not isinstance(topology.cell, periodic_cell_owner.PeriodicCell):
                    raise TypeError(
                        "periodic.algebraic.cell must be a translational periodic cell; got isometries"
                    )
                cell = topology.cell
                block_name = domain.blocks[0].name
                source_vertices = topology.quotient.entities(0).count
                original = domain
            case _ as invalid:
                assert_never(invalid)
        mesh = periodic_owner.publish_periodic_simplices(
            carrier.points,
            carrier.simplices,
            carrier.shifts,
            cell,
            block_name=block_name,
            exact_coordinates=carrier.exact_coordinates,
            scope=scope,
            budget=budget,
        )
        allowance = _PeriodicSupportAllowance(
            specification.limits, scope=scope, budget=budget
        )
        with allowance.exact_phase():
            bank = periodic_owner.periodic_lifted_exact_coordinates(
                carrier.exact_coordinates,
                carrier.simplices,
                carrier.shifts,
                cell,
                budget=budget,
            )
            elements = {
                block.name: cell_geometry_owner.coordinate_lagrange_element(
                    block.cell_kind, 1
                )
                for block in mesh.blocks
            }
            routes = {block.name: block.vertices for block in mesh.blocks}
            source_work, bits = algebraic_source_limits(bank)
            exact = ExactAlgebraicCellGeometrySource(
                bank,
                rational_source_layouts(elements, routes),
                source_identity=carrier.source_identity,
                maximum_work=source_work,
                maximum_bits=bits,
                budget=budget,
            )
            geometry = cell_geometry_owner.CellGeometrySpec(
                elements, routes, mesh.coordinates, exact_source=exact, budget=budget
            )
            budget.retain_basis((mesh, geometry))
        return _PeriodicConstruction(
            mesh,
            mesh if original is None else original,
            carrier.regions,
            carrier.regions.copy()
            if original is None
            else np.asarray(source.cell_regions).copy(),
            carrier.ancestry,
            budget.work_units - starting_work,
            carrier.points.shape[0] - source_vertices,
            (),
            0,
            True,
            carrier.image_count,
            geometry,
            carrier.work_evidence,
        )


def _construct_represented_periodic_domain(
    mesh: cell_mesh_owner.CellMesh,
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    image_budget: int,
    hard: list[UniformSizeControl],
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _PeriodicConstruction:
    work = 0
    inserted_sites = 0
    sizing_evidence: tuple[PeriodicSizeRecord, ...] = ()
    image_refusals = 0
    stratified = False
    source_construction_images: int | None = None
    original_mesh = mesh
    original_regions = np.asarray(source.cell_regions).copy()
    regions = original_regions.copy()
    ancestry = tuple((index,) for index in range(mesh.blocks[0].cell_count))
    if (
        hard
        and source.ambient_dimension == 2
        and isinstance(
            periodic_owner.require_periodic_topology(mesh).cell,
            periodic_cell_owner.PeriodicCell,
        )
    ):
        with measure_phase(record_phase, "periodic_construction"):
            strata = constraints_owner.prepare_periodic_source_strata(
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
            seeds = constraints_owner.periodic_source_seed_points(
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
                scope=scope,
                allowance=_PeriodicSupportAllowance(limits, scope=scope, budget=budget),
            )
            sizing_evidence = site_mesh.sizing_evidence
            image_refusals = site_mesh.image_refusals
            if site_mesh.source_support is None:
                raise MeshingFailure(
                    MeshingFailureCategory.AUDIT_FAILED,
                    "periodic.source_strata.support must retain constrained construction evidence; got no source support",
                    stage=stage.value,
                )
            source_construction_images = site_mesh.source_support.image_count
            regions, ancestry, certification_work = (
                constraints_owner.bind_periodic_source_strata(
                    site_mesh.points,
                    site_mesh.simplices,
                    site_mesh.simplex_shifts,
                    strata,
                    limits.maximum_work_units - work,
                    scope=scope,
                )
            )
            work += certification_work
            inserted_sites = site_mesh.points.shape[0] - strata.points.shape[0]
            mesh = periodic_owner.publish_periodic_simplices(
                site_mesh.points,
                site_mesh.simplices,
                site_mesh.simplex_shifts,
                strata.cell,
                block_name=strata.block_name,
                scope=scope,
                budget=budget,
            )
            stratified = True
    return _PeriodicConstruction(
        mesh,
        original_mesh,
        regions,
        original_regions,
        ancestry,
        work,
        inserted_sites,
        sizing_evidence,
        image_refusals,
        stratified,
        source_construction_images,
    )


def _construct_periodic_domain(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    construction: PeriodicConstructionMethod,
    limits: MeshingLimits,
    started: float,
    stage: MeshingStageKind,
    image_budget: int,
    hard: list[UniformSizeControl],
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _PeriodicConstruction:
    match construction:
        case "orbit_lattice_algebraic_bands" | "source_action_algebraic_bands":
            return _construct_algebraic_periodic_domain(
                source,
                specification,
                construction,
                scope=scope,
                budget=budget,
                record_phase=record_phase,
            )
        case "orbit_sites":
            orbits = source.domain
            if not isinstance(orbits, periodic_owner.PeriodicPointOrbits):
                raise TypeError(
                    "periodic.construction must match the admitted seed-orbit domain; got represented source cells"
                )
            return _construct_periodic_seed(
                orbits,
                source,
                specification,
                limits,
                started,
                stage,
                image_budget,
                hard,
                scope=scope,
                budget=budget,
                record_phase=record_phase,
            )
        case "represented_source":
            mesh = source.domain
            if not isinstance(mesh, cell_mesh_owner.CellMesh):
                raise TypeError(
                    "periodic.construction must match the admitted represented domain; got seed orbits"
                )
            return _construct_represented_periodic_domain(
                mesh,
                source,
                specification,
                limits,
                started,
                stage,
                image_budget,
                hard,
                scope=scope,
                budget=budget,
                record_phase=record_phase,
            )
        case _ as invalid:
            assert_never(invalid)


def _certify_periodic_construction(
    mesh: cell_mesh_owner.CellMesh,
    initial: periodic_owner.PeriodicQuotientEvidence,
    limits: MeshingLimits,
    work: int,
    image_budget: int,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    record_phase: NativeMeshingPhaseRecorder | None,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
) -> _PeriodicCertification:
    quotient = periodic_owner.PeriodicQuotientEvidence(
        mesh, geometry=geometry, budget=budget
    )
    if (
        not quotient.all_cells_valid
        or abs(quotient.total_measure - initial.total_measure)
        > 1.0e-10 * initial.total_measure
    ):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "periodic.refinement.coverage must preserve positive source coverage within tolerance; got invalid cells or changed measure",
        )
    if quotient.relative_coverage_defect is not None and (
        quotient.relative_coverage_defect > 1.0e-10 or quotient.boundary_facet_count
    ):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "periodic.refinement.quotient must be a closed degree-one domain cover; got boundary facets or a coverage defect",
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
            with budget.measurement():
                # Measured bounded BVH visits now debit the native owner;
                # candidate count is not a proxy for a fictitious all-cell scan.
                embedding_budget = image_budget // mesh.coordinates.shape[0]
                if embedding_budget <= 0 or limits.maximum_work_units <= work:
                    raise PeriodicEmbeddingResourceError(
                        "periodic.embedding.resources must have positive remaining image and work allowances; got an exhausted allowance"
                    )
                embedding = periodic_owner.certify_periodic_embedding(
                    mesh,
                    maximum_images=embedding_budget,
                    maximum_pairs=limits.maximum_work_units - work,
                    record_work=record_embedding_work,
                    record_unmanaged=record_embedding_unmanaged,
                    scope=scope,
                    budget=budget,
                    geometry=geometry,
                )
        except PeriodicEmbeddingResourceError as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=MeshingStageKind.CERTIFICATION.value,
                achieved=embedding_work_evidence(),
            ) from error
        except MeshcoreError as error:
            match error.status:
                case MeshcoreStatus.CAPACITY_EXCEEDED:
                    category = MeshingFailureCategory.RESOURCE_EXHAUSTED
                case MeshcoreStatus.TIMEOUT:
                    category = MeshingFailureCategory.TIMED_OUT
                case _:
                    raise
            raise MeshingFailure(
                category,
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
    return _PeriodicCertification(quotient, embedding, embedding_work_evidence())


def _periodic_size_compliance(
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec | None,
    control: UniformSizeControl,
    policy: SizeCompliancePolicy,
    lengths: np.ndarray,
    growth: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Display binary64 length evidence; decide hard issues on exact quotient lengths.

    Without a separately declared geometry the published binary64
    representatives and unrounded group actions are the source. In both cases the
    verdict compares field-valued squares, never rounded Euclidean roots.
    """
    requested, achieved, _ = uniform_size_compliance(control, policy, lengths, growth)
    if geometry is not None:
        prefix = f"size:{control.control_id}"
        values = periodic_owner.periodic_edge_size_quantile(
            mesh,
            np.asarray(
                [size_statistic_quantile(name) for name in policy.target_statistics],
                dtype=np.float64,
            ),
            geometry=geometry,
            budget=budget,
        )
        replacements = {
            f"{prefix}:{name}_edge": float(value)
            for name, value in zip(policy.target_statistics, values, strict=True)
        }
        achieved = [(name, replacements.get(name, value)) for name, value in achieved]
    if control.strength is not SizeControlStrength.HARD:
        return requested, achieved, []
    topology = periodic_owner.require_periodic_topology(mesh)
    edges = periodic_owner.simplex_entity_corners(mesh, 1)[
        np.asarray(topology.orbit_representatives(1))
    ]
    with budget.temporary_scope():
        issues = periodic_owner.periodic_size_control_issues(
            periodic_owner.periodic_edge_squared_lengths(
                mesh, geometry=geometry, budget=budget
            ),
            np.asarray(topology.orbits(0)[0])[edges],
            control,
            policy,
            budget=budget,
        )
    return requested, achieved, issues


def _periodic_compliance(
    construction: _PeriodicConstruction,
    mesh: cell_mesh_owner.CellMesh,
    certification: _PeriodicCertification,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    controls: list[UniformSizeControl],
    constraints: _PeriodicCompliance,
    initial_measures: np.ndarray,
    lengths: np.ndarray,
    growth: float,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    budget: CoordinateEnclosureBudget,
) -> _PeriodicCompliance:
    quotient, embedding = certification.quotient, certification.embedding
    inserted_sites, image_refusals = (
        construction.inserted_sites,
        construction.image_refusals,
    )
    sizing_evidence = construction.sizing_evidence
    original_regions, regions = construction.original_regions, construction.regions
    requested, achieved, issues = (
        constraints.requested,
        constraints.achieved,
        constraints.issues,
    )
    with measure_phase(record_phase, "compliance"):
        for control in controls:
            expected, measured, failures = _periodic_size_compliance(
                mesh,
                construction.geometry,
                control,
                specification.size_compliance,
                lengths,
                growth,
                budget=budget,
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
    if construction.source_construction_images is not None:
        achieved.append(
            (
                "periodic:source_construction_images",
                float(construction.source_construction_images),
            )
        )
    achieved.extend(certification.work_evidence)
    achieved.extend(
        (
            (
                "periodic:coordinate_lift_residual",
                embedding.maximum_coordinate_residual,
            ),
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
    successor_measures = np.asarray(
        periodic_measures_owner.periodic_orbit_measures(mesh).orbit_measures
    )
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
    return _PeriodicCompliance(requested, achieved, issues)


def _periodic_feature_labels(
    original_mesh: cell_mesh_owner.CellMesh,
    mesh: cell_mesh_owner.CellMesh,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    lengths: np.ndarray,
    compliance: _PeriodicCompliance,
    associations: tuple[GeometryAssociation, ...],
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> NativePublicationOrganization:
    issues = compliance.issues
    labels = []
    if specification.protected_features:
        with measure_phase(record_phase, "organization"):
            for feature in specification.protected_features:
                ids = np.asarray(feature.scope.entity_ids)
                degree = feature.scope.entity_dimension
                if degree > 0:
                    association = associations[degree]
                    rows = association.target_rows(
                        np.asarray(mesh.entity_set(degree).entity_ids)
                    )
                    mask = (
                        np.asarray(association.source_dimensions)[rows] == degree
                    ) & np.isin(np.asarray(association.source_indices)[rows], ids)
                    source_rows = periodic_owner.simplex_entity_corners(
                        original_mesh, degree
                    )
                    source_corners = np.asarray(original_mesh.coordinates)[source_rows]
                    target_rows = periodic_owner.simplex_entity_corners(mesh, degree)
                    target_corners = np.asarray(mesh.coordinates)[target_rows]
                    if degree == 1:
                        source_lengths, _ = periodic_owner.periodic_edge_size_evidence(
                            original_mesh
                        )
                        source_measures = source_lengths[
                            np.asarray(
                                periodic_owner.require_periodic_topology(
                                    original_mesh
                                ).orbits(1)[0]
                            )
                        ]
                        target_measures = lengths[
                            np.asarray(
                                periodic_owner.require_periodic_topology(mesh).orbits(1)[
                                    0
                                ]
                            )
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
                        periodic_owner.require_periodic_topology(
                            original_mesh
                        ).orbit_representatives(degree)
                    )[ids]
                    target_leaders = np.asarray(
                        periodic_owner.require_periodic_topology(
                            mesh
                        ).orbit_representatives(degree)
                    )
                    covered = float(
                        np.sum(target_measures[target_leaders[mask[target_leaders]]])
                    )
                    expected = float(np.sum(source_measures[leaders]))
                    if abs(covered - expected) > 1.0e-10 * max(1.0, expected):
                        issues.append(f"feature_orbit_coverage:{feature.feature_id}")
                else:
                    source_vertices = np.asarray(
                        periodic_owner.require_periodic_topology(
                            original_mesh
                        ).orbit_representatives(0)
                    )[ids]
                    source_points = np.asarray(original_mesh.coordinates)[source_vertices]
                    points = np.asarray(mesh.coordinates)
                    mask = np.zeros(points.shape[0], dtype=np.bool_)
                    vertex_orbits = np.asarray(
                        periodic_owner.require_periodic_topology(mesh).orbits(0)[0]
                    )
                    leaders = np.asarray(
                        periodic_owner.require_periodic_topology(
                            mesh
                        ).orbit_representatives(0)
                    )
                    for point in source_points:
                        matches = np.flatnonzero(np.all(points[leaders] == point, axis=1))
                        if matches.size != 1:
                            raise MeshingFailure(
                                MeshingFailureCategory.COMPLIANCE_FAILED,
                                "periodic.features.corner must have one retained quotient representative; got absent or ambiguous representatives",
                            )
                        mask |= vertex_orbits == matches[0]
                if not np.any(mask):
                    raise MeshingFailure(
                        MeshingFailureCategory.COMPLIANCE_FAILED,
                        "periodic.features.orbit must contain a retained target entity; got an empty feature orbit",
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
    return NativePublicationOrganization(labels=tuple(labels))


def _organize_periodic_domain(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    mesh: cell_mesh_owner.CellMesh,
    regions: np.ndarray,
    features: NativePublicationOrganization,
    associations: tuple[GeometryAssociation, ...],
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> NativePublicationOrganization:
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
                    "periodic.regions.material_id must be unique within each region; got conflicting material assignments",
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
    cell_association = associations[-1]
    associated_regions = np.asarray(cell_association.source_indices)[
        cell_association.target_rows(np.asarray(mesh.blocks[0].global_ids))
    ]
    if not np.array_equal(associated_regions, regions):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "periodic.associations.cell_regions must agree with the published material inventory; got changed source membership",
        )
    return NativePublicationOrganization(
        labels=features.labels, zones=tuple(zones), associations=associations
    )


def _periodic_construction_label(
    method: PeriodicConstructionMethod, stratified: bool, /
) -> str:
    match method:
        case "orbit_lattice_algebraic_bands":
            return "orbit_lattice_algebraic_equilateral_bands"
        case "source_action_algebraic_bands":
            return "source_identity_algebraic_circle_bands"
        case "orbit_sites":
            return "translational_delaunay"
        case "represented_source":
            return (
                "source_stratified_periodic_cdt"
                if stratified
                else "represented_domain_orbit_bisection"
            )
        case _ as invalid:
            assert_never(invalid)


def _publish_periodic_domain(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    prepared: PreparedPeriodicDomain,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    construction: _PeriodicConstruction,
    mesh: cell_mesh_owner.CellMesh,
    certification_data: _PeriodicCertification,
    compliance_data: _PeriodicCompliance,
    organization: NativePublicationOrganization,
    stage: MeshingStageKind,
    *,
    execution: NativePublicationExecution,
) -> CellMeshingResult:
    limits, started = specification.limits, execution.scope.operation_started
    embedding = certification_data.embedding
    ancestry, regions = construction.ancestry, construction.regions
    sizing_evidence, stratified = construction.sizing_evidence, construction.stratified
    compliance = MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(compliance_data.requested),
        achieved=tuple(compliance_data.achieved),
        issues=tuple(compliance_data.issues),
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
            **(
                {
                    "declared_geometry": cell_geometry_id(construction.geometry),
                    "exact_construction_work": dict(construction.construction_evidence),
                    "exact_construction_inner_predicate_work_complete": False,
                }
                if construction.geometry is not None
                else {}
            ),
            "route": "periodic_exact_source"
            if construction.geometry is not None
            else "periodic_delaunay",
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
            "construction": _periodic_construction_label(
                prepared.construction, stratified
            ),
        },
        certification,
        policy=publication_policy("periodic_stratified" if stratified else "periodic"),
        organization=organization,
        execution=execution,
        geometry=construction.geometry,
    )


@final
class _PeriodicRoutePreparation(StrictModule):
    """Carry seam compliance and the admitted construction request."""

    stage: MeshingStageKind
    image_budget: int
    controls: list[UniformSizeControl]
    hard: list[UniformSizeControl]
    constraints: _PeriodicCompliance


def _prepare_periodic_route_request(
    source: NativePeriodicSource, specification: SurfaceMeshingSpec | VolumeMeshingSpec, /
) -> _PeriodicRoutePreparation:
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
            "periodic.construction.maximum_images must be positive; got no image storage allowance",
        )
    controls: list[UniformSizeControl] = []
    for control in specification.size_controls:
        match control:
            case UniformSizeControl():
                controls.append(control)
            case _:
                raise TypeError(
                    "periodic.size_controls must be uniform; got a nonuniform control"
                )
    hard = [
        control for control in controls if control.strength is SizeControlStrength.HARD
    ]
    return _PeriodicRoutePreparation(
        stage,
        image_budget,
        controls,
        hard,
        _PeriodicCompliance(constraint_requested, constraint_achieved, []),
    )


@checked
def execute_periodic_route(
    source: NativePeriodicSource,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    prepared: PreparedPeriodicDomain,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    with ExitStack() as stack:
        if (
            prepared.source.binding_id != source.binding_id
            or prepared.specification_id != specification.specification_id
        ):
            raise ValueError(
                "periodic.prepared must bind this source and specification; got a different request"
            )
        started = scope.operation_started
        initial = scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=specification.limits.maximum_work_units,
            maximum_memory_bytes=specification.limits.maximum_scratch_bytes,
        )
        initial.settle()
        stack.enter_context(
            budget.retained_storage(lambda bound: scope.batch(0, bytes_upper=bound))
        )
        workspace = stack.enter_context(scope.host_workspace())
        request = _prepare_periodic_route_request(source, specification)
        limits, stage = specification.limits, request.stage
        image_budget, controls, hard = (
            request.image_budget,
            request.controls,
            request.hard,
        )
        construction = _construct_periodic_domain(
            source,
            specification,
            prepared.construction,
            limits,
            started,
            stage,
            image_budget,
            hard,
            scope=scope,
            budget=budget,
            record_phase=record_phase,
        )
        mesh, original_mesh = construction.mesh, construction.original_mesh
        regions, ancestry = construction.regions, construction.ancestry
        work, stratified = construction.work, construction.stratified
        allowance = _PeriodicSupportAllowance(limits, scope=scope, budget=budget)
        initial = periodic_owner.PeriodicQuotientEvidence(original_mesh)
        initial_measures = np.asarray(
            periodic_measures_owner.periodic_orbit_measures(original_mesh).orbit_measures
        )
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
            edges = periodic_owner.simplex_entity_corners(mesh, 1)
            with (
                allowance.exact_phase()
                if construction.geometry is not None
                else nullcontext(),
                budget.temporary_scope()
                if construction.geometry is not None
                else nullcontext(),
            ):
                if construction.geometry is None:
                    lengths, growth = periodic_owner.periodic_edge_size_evidence(mesh)
                else:
                    size_evidence = periodic_owner.periodic_exact_edge_size_evidence(
                        mesh, construction.geometry, budget=budget
                    )
                    budget.retain_basis((size_evidence,))
                    lengths = size_evidence.representative_lengths
                    growth = size_evidence.maximum_local_edge_ratio
            # Only the stratified algebraic construction carries a separate
            # declared geometry. Otherwise the constructed binary64 carrier is
            # the declared geometry: its field-valued longest quotient edge, not
            # a rounded norm, decides refinement and then orders the bisection.
            if stratified:
                break
            geometry = cell_geometry_owner.CellGeometrySpec.affine(mesh)
            with allowance.exact_phase(), budget.temporary_scope():
                exceeds = (
                    periodic_owner.periodic_edge_size_quantile_compare(
                        mesh, 1.0, threshold, geometry=geometry, budget=budget
                    )
                    > 0
                )
            if not exceeds:
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
                    "periodic.refinement.closure must fit the entity and connectivity budgets; got a request beyond the budgets",
                )
            work += upper_cells
            if (
                work > limits.maximum_work_units
                or upper_cells * 128 + upper_vertices * 64 > limits.maximum_scratch_bytes
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "periodic.refinement.resources must fit the work and scratch budgets; got a request beyond the budgets",
                )
            with measure_phase(record_phase, "refinement"):
                refinement = periodic_owner.refine_periodic_mesh(
                    mesh, source_geometry=geometry, scope=scope, budget=budget
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
        construction = eqx.tree_at(
            lambda item: (item.mesh, item.regions, item.ancestry, item.work),
            construction,
            (mesh, regions, ancestry, work),
        )
        with (
            allowance.exact_phase()
            if construction.geometry is not None
            else nullcontext(),
            budget.temporary_scope()
            if construction.geometry is not None
            else nullcontext(),
        ):
            certification = _certify_periodic_construction(
                mesh,
                initial,
                limits,
                work,
                image_budget,
                scope=scope,
                budget=budget,
                record_phase=record_phase,
                geometry=construction.geometry,
            )
            if construction.geometry is not None:
                budget.retain_basis((certification,))
        # Hard verdicts compare unrounded lengths for every source, so they borrow
        # the native ledger even without a separately declared geometry.
        with (
            allowance.exact_phase()
            if construction.geometry is not None or hard
            else nullcontext()
        ):
            compliance = _periodic_compliance(
                construction,
                mesh,
                certification,
                specification,
                controls,
                request.constraints,
                initial_measures,
                lengths,
                growth,
                record_phase=record_phase,
                budget=budget,
            )
        with measure_phase(record_phase, "geometry_association"):
            associations = periodic_source_associations(
                source,
                specification,
                mesh,
                scope=scope,
                budget=budget,
                geometry=construction.geometry,
            )
        features = _periodic_feature_labels(
            original_mesh,
            mesh,
            specification,
            lengths,
            compliance,
            associations,
            record_phase=record_phase,
        )
        organization = _organize_periodic_domain(
            source,
            specification,
            mesh,
            regions,
            features,
            associations,
            record_phase=record_phase,
        )
        return _publish_periodic_domain(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            construction,
            mesh,
            certification,
            compliance,
            organization,
            stage,
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
        )


__all__ = [
    "NativePeriodicSource",
    "PreparedPeriodicDomain",
    "PeriodicAssociationTransfer",
    "periodic_support_issues",
    "execute_periodic_route",
]
