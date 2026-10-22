#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared structured/swept routes using the common native publication owner."""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from math import prod
from typing import assert_never, final

import equinox as eqx
import jax
import numpy as np
from jax import Array

from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry._domain_coverage import (
    DomainCoverageCertificate,
    PiecewiseLinearDomain,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryQuery
from phydrax.meshing._certification import MeshCertificationReport
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._host_banks import entities, family_host_array
from phydrax.typing import checked

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_complex import IntervalConnectivity
from ...discretization._cell_geometry import (
    affine_geometry_construction_bounds,
    CellGeometryRestrictionSource,
    require_scalar_coordinate_element,
)
from ...discretization._cell_geometry_transition_types import CellGeometryTransitionError
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ...discretization._mapped_geometry_measure import prepare_mapped_edge_arc_length
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from .._audit import (
    audit_cell_mesh,
    CellMeshAuditDisposition,
    CellMeshAuditPolicy,
)
from .._controls import BlockInterfaceControl, RegionControl
from .._family_execution import count_geometry_queries, family_route_refusals
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._multiblock import construct_glued_structured_blocks
from .._organization_records import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionBoundaryEvidence,
)
from .._quality import evaluate_cell_quality, summarize_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._structured import construct_structured_block, TransfiniteBlock
from .._sweep import construct_sweep, SweepControl
from .._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from ._native_publication import (
    coordinate_budget,
    coordinate_phase,
    derived_certificate_limits,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import (
    check_deadline,
    edge_size_evidence,
    uniform_size_compliance,
)
from ._native_route_metadata import mesh_entity_scope, specification_scope
from ._native_route_policy import publication_policy


def _family_issues(
    actual: set[str],
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> list[str]:
    policy = specification.target.cell_families
    issues = []
    if not set(policy.required).issubset(actual):
        issues.append("required_cell_families")
    if not actual.issubset(
        set((*policy.required, *policy.preferred, *policy.allowed_transitions))
    ):
        issues.append("undeclared_cell_families")
    if not policy.allow_mixed and len(actual) != 1:
        issues.append("pure_family_policy")
    return issues


@checked
def structured_support_issues(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
    *,
    sweep: bool = False,
) -> list[str]:
    issues = []
    if isinstance(specification, VolumeMeshingSpec):
        expected_fill = (
            VolumeFillStrategy.SWEEP if sweep else VolumeFillStrategy.MULTIZONE
        )
        if specification.fill_strategy is not expected_fill:
            issues.append(f"the declared {expected_fill.value} volume fill strategy")
    if not sweep:
        family = (
            "quadrilateral"
            if specification.target.topological_dimension == 2
            else "hexahedron"
        )
        issues.extend(_family_issues({family}, specification))
    if not sweep and specification.target.geometry_order != 1:
        issues.append("vertex-level mapped tensor/prism geometry")
    if any(
        not isinstance(control, UniformSizeControl)
        for control in specification.size_controls
    ):
        issues.append("uniform size controls on the declared structured scope")
    scope = specification_scope(specification)
    if any(
        isinstance(control, UniformSizeControl)
        and control.scope.scope_id != scope.scope_id
        for control in specification.size_controls
    ):
        issues.append("size controls outside the complete declared structured scope")
    if isinstance(specification, SurfaceMeshingSpec) and (
        specification.region_controls or specification.patch_controls
    ):
        issues.append("volume region/facet controls on a surface block declaration")
    if any(
        control.scope.entity_dimension != 3 for control in specification.region_controls
    ):
        issues.append("region controls outside declared volume regions")
    if any(
        control.scope.entity_dimension != 2 for control in specification.patch_controls
    ):
        issues.append("patch controls outside declared source facets")
    if specification.periodic_constraints or specification.layer_controls:
        issues.append(
            "source-specific periodic/layer controls outside the explicit block/sweep declaration"
        )
    if isinstance(specification, VolumeMeshingSpec) and (
        specification.region_seeds or specification.hole_seeds
    ):
        issues.append("region/hole decomposition through explicitly declared blocks")
    return issues


@final
class PreparedStructured(StrictModule, NonTrainableState):
    blocks: tuple[TransfiniteBlock, ...]
    interfaces: tuple[BlockInterfaceControl, ...]
    optimization_steps: int = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        blocks: tuple[TransfiniteBlock, ...],
        interfaces: tuple[BlockInterfaceControl, ...],
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        source_binding_id: str,
        /,
        *,
        optimization_steps: int = 0,
    ) -> None:
        if not blocks or not all(isinstance(block, TransfiniteBlock) for block in blocks):
            raise TypeError(
                "PreparedStructured.blocks must contain nonempty TransfiniteBlock values; got an empty or invalid block sequence"
            )
        if len({block.name for block in blocks}) != len(blocks):
            raise ValueError(
                "PreparedStructured.blocks.name must be unique; got duplicate names"
            )
        if any(
            len(block.intervals) != specification.target.topological_dimension
            for block in blocks
        ):
            raise ValueError(
                "PreparedStructured.blocks.dimensions must match the physical target; got mismatched dimensions"
            )
        if any(
            not isinstance(i, BlockInterfaceControl) or not i.conforming
            for i in interfaces
        ):
            raise ValueError(
                "PreparedStructured.interfaces must be conforming for one CellMeshingResult; got an independent or nonconforming interface"
            )
        if (
            isinstance(optimization_steps, bool)
            or not isinstance(optimization_steps, int)
            or optimization_steps < 0
        ):
            raise ValueError(
                "PreparedStructured.optimization_steps must be a nonnegative integer; got an invalid count"
            )
        if not source_binding_id:
            raise ValueError(
                "native.prepared.source_binding_id must be nonempty; got no source binding identity"
            )
        issues = structured_support_issues(specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY, "; ".join(issues)
            )
        self.blocks = tuple(sorted(blocks, key=lambda block: block.name))
        self.interfaces = tuple(
            sorted(interfaces, key=lambda interface: interface.control_id)
        )
        self.optimization_steps = optimization_steps
        self.specification_id = specification.specification_id
        self.source_binding_id = source_binding_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-structured",
                "source": source_binding_id,
                "specification": self.specification_id,
                "blocks": tuple(b.block_id for b in self.blocks),
                "interfaces": tuple(i.control_id for i in self.interfaces),
                "optimization_steps": optimization_steps,
            }
        )


@final
class PreparedSweep(StrictModule, NonTrainableState):
    profile: CellMesh
    profile_geometry: CellGeometrySpec
    control: SweepControl
    specification_id: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        profile: CellMesh,
        control: SweepControl,
        specification: VolumeMeshingSpec,
        source_binding_id: str,
        /,
        *,
        profile_geometry: CellGeometrySpec | None = None,
        scope: NativeExecutionScope,
    ) -> None:
        if profile.topological_dimension != 2 or profile.ambient_dimension not in (2, 3):
            raise ValueError(
                "PreparedSweep.profile must have two-dimensional cells in 2D or 3D; got incompatible dimensions"
            )
        if any(
            block.cell_kind not in ("triangle", "quadrilateral")
            for block in profile.blocks
        ):
            raise ValueError(
                "PreparedSweep.profile.blocks.cell_kind must be triangle or quadrilateral; got an incompatible source cell family"
            )
        if profile_geometry is None:
            profile_shapes = tuple(
                (block.cell_kind, block.cell_count, block.arity)
                for block in profile.blocks
            )
            with scope.host_workspace() as workspace:
                workspace.set_bound(
                    affine_geometry_construction_bounds(
                        profile.coordinates.shape[0],
                        profile.ambient_dimension,
                        profile_shapes,
                    )
                )
                geometry = CellGeometrySpec.affine(profile)
        else:
            geometry = profile_geometry
        geometry.resolve(profile)

        if any(
            not isinstance(element, FiniteElementSpec) for element in geometry.elements
        ):
            raise ValueError(
                "PreparedSweep.profile_geometry.elements must be unrestricted polynomial maps; got a restricted or nonpolynomial map"
            )
        degrees = tuple(
            element.degree
            for element in geometry.elements
            if isinstance(element, FiniteElementSpec)
        )
        if max(degrees) != specification.target.geometry_order:
            raise ValueError(
                "PreparedSweep.profile_geometry.degree must equal target.geometry_order; got mismatched geometry order"
            )
        actual = {
            "prism" if block.cell_kind == "triangle" else "hexahedron"
            for block in profile.blocks
        }
        issues = structured_support_issues(specification, sweep=True)
        issues.extend(_family_issues(actual, specification))
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY, "; ".join(issues)
            )
        if not source_binding_id:
            raise ValueError(
                "native.prepared.source_binding_id must be nonempty; got no source binding identity"
            )
        self.profile = profile
        self.profile_geometry = geometry
        self.control = control
        self.specification_id = specification.specification_id
        self.source_binding_id = source_binding_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-sweep",
                "source": source_binding_id,
                "specification": self.specification_id,
                "profile": profile.mesh_id,
                "control": control.control_id,
                "profile_geometry": cell_geometry_id(geometry),
            }
        )


def _preflight(
    vertices: int,
    cells: int,
    connectivity_entries: int,
    ambient: int,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
    *,
    attribute_bytes: int = 0,
) -> None:
    limits = specification.limits
    estimates = (
        (vertices, limits.maximum_vertices, "vertices"),
        (cells, limits.maximum_cells, "cells"),
        (connectivity_entries, limits.maximum_connectivity_entries, "connectivity"),
        (
            vertices * ambient * 8 + connectivity_entries * 4 + attribute_bytes,
            limits.maximum_data_bytes,
            "data",
        ),
    )
    for actual, maximum, name in estimates:
        if actual > maximum:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"structured.entity_capacity must not exceed maximum_{name}={maximum}; got {actual}",
                stage="structured_preflight",
            )


@checked
def mapped_edge_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical source-edge arc lengths and their certified bounds."""

    limits = specification.limits
    edge_count = int(mesh.entity_set(1).entity_ids.shape[0])
    remaining = scope.remaining()
    # Source preparation and intrinsic predicates share the edge-query phase;
    # every physical edge needs one query within the request's remaining queries.
    queries = min(limits.maximum_geometry_queries, remaining.remaining_geometry_queries)
    if edge_count > queries:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "structured.edge_arc_queries must fit the remaining geometry queries; got more physical edges",
            stage="structured_edge_arcs",
            requested=(("remaining_geometry_queries", queries),),
            achieved=(("physical_edges", edge_count),),
        )
    with scope.host_workspace() as workspace:
        # Host edge rows, CPython key tuples, integers and dictionary slots of at
        # most 256 bytes per edge, and the coverage mask stay live through the
        # edge loop.
        workspace.set_bound(273 * edge_count + 4096)
        lengths = family_host_array((edge_count,), np.float64, scope=scope)
        lower = family_host_array(lengths.shape, np.float64, scope=scope)
        upper = family_host_array(lengths.shape, np.float64, scope=scope)
        lengths.fill(np.nan)
        remaining = scope.remaining()
        work = min(limits.maximum_work_units, remaining.remaining_work_units)
        scratch = min(limits.maximum_scratch_bytes, remaining.remaining_managed_bytes)
        ticket = (
            scope.batch(work, queries_upper=queries, bytes_upper=scratch)
            if budget is None
            else None
        )
        ledger = (
            CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            if ticket is not None
            else budget
        )
        if ledger is None:
            raise RuntimeError("Mapped edge evidence requires its actual request ledger.")
        try:
            with (
                ticket if ticket is not None else nullcontext(),
                ticket.native_target() if ticket is not None else nullcontext(),
                ledger.activate() if ticket is not None else nullcontext(),
                ledger.bound_stage(work, scratch)
                if ticket is not None
                else nullcontext(),
            ):
                edges = entities(mesh, 1)
                lookup = {
                    tuple(sorted(pair.tolist())): row for row, pair in enumerate(edges)
                }
                elements, routes, _ = geometry.resolve(mesh, budget=ledger)
                coefficients = geometry.source_coordinates(budget=ledger)
                for block, element, route in zip(
                    mesh.blocks, elements, routes, strict=True
                ):
                    topology = reference_cell_topology(block.cell_kind)
                    points = tuple(
                        tuple(Fraction(float(value)) for value in point)
                        for point in topology.vertices
                    )
                    for vertex_row, dof_row in zip(
                        np.asarray(block.vertices), np.asarray(route), strict=True
                    ):
                        vertices = vertex_row.tolist()
                        local = tuple(coefficients[index] for index in dof_row.tolist())
                        for first, last in topology.entities[1]:
                            row = lookup[
                                tuple(sorted((vertices[first], vertices[last])))
                            ]
                            if np.isfinite(lengths[row]):
                                continue
                            count_geometry_queries(1, work_units=0, budget=ledger)
                            # Each edge's restriction snapshots and controls
                            # are consumed by its own enclosure.
                            with ledger.polynomial_preparation_scope():
                                enclosure = prepare_mapped_edge_arc_length(
                                    element,
                                    local,
                                    points[first],
                                    points[last],
                                    budget=ledger,
                                ).integrate(
                                    absolute_tolerance=1e-12,
                                    relative_tolerance=1e-12,
                                    maximum_work=limits.maximum_work_units,
                                    maximum_subcells=min(limits.maximum_cells, 10000),
                                    budget=ledger,
                                )
                            lengths[row], lower[row], upper[row] = (
                                enclosure.value,
                                enclosure.lower,
                                enclosure.upper,
                            )
                if not np.all(np.isfinite(lengths)):
                    raise MeshingFailure(
                        MeshingFailureCategory.AUDIT_FAILED,
                        "structured.edge_arc_coverage must cover every physical edge; got missing edges",
                        stage="structured_edge_arcs",
                    )
        except CoordinateEnclosureResourceError as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage="structured_edge_arcs",
            ) from error
        except CellGeometryTransitionError as error:
            category = (
                MeshingFailureCategory.RESOURCE_EXHAUSTED
                if error.reason == "resource_limit"
                else MeshingFailureCategory.AUDIT_FAILED
            )
            raise MeshingFailure(
                category, str(error), stage="structured_edge_arcs"
            ) from error
    return edges, lengths, lower, upper


def _compliance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> MeshingComplianceReport:
    target = specification.target
    actual = {block.cell_kind for block in mesh.blocks}
    issues = []
    if (
        mesh.topological_dimension != target.topological_dimension
        or mesh.ambient_dimension != target.ambient_dimension
    ):
        issues.append("target_dimensions")
    issues.extend(_family_issues(actual, specification))
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = [
        (
            f"family:{kind}:cells",
            sum(b.cell_count for b in mesh.blocks if b.cell_kind == kind),
        )
        for kind in sorted(actual)
    ]
    if mesh.topological_dimension not in (2, 3):
        raise ValueError(
            "structured.mesh.connectivity must represent surface or volume cells; got a different connectivity family"
        )
    connectivity = mesh.connectivity
    if isinstance(connectivity, IntervalConnectivity):
        raise TypeError(
            "structured.mesh.connectivity must represent surface or volume cells; got interval connectivity"
        )
    vertex_count = mesh.coordinates.shape[0]
    with scope.host_workspace() as workspace:
        if any(
            require_scalar_coordinate_element(
                element, "Structured physical sizing"
            ).degree
            > 1
            for element in geometry.elements
        ):
            edges, lengths, lower, upper = mapped_edge_evidence(
                mesh, geometry, specification, scope=scope
            )
            edge_count = edges.shape[0]
            # The returned host edge rows stay live; size statistics own the
            # vertex ratio temporary.
            retained = 16 * edge_count
            workspace.set_bound(retained + 8 * vertex_count + 4096)
            minimum = family_host_array((vertex_count,), np.float64, scope=scope)
            maximum = family_host_array(minimum.shape, np.float64, scope=scope)
            minimum.fill(np.inf)
            maximum.fill(0.0)
            for endpoint in range(2):
                np.minimum.at(minimum, edges[:, endpoint], lower)
                np.maximum.at(maximum, edges[:, endpoint], upper)
            growth = float(np.max(maximum / minimum, initial=1.0))
            achieved.extend(
                (
                    ("physical_edge_arc_minimum_lower", float(np.min(lower))),
                    ("physical_edge_arc_maximum_upper", float(np.max(upper))),
                )
            )
        else:
            # Linear tensor/simplex coordinate bases map reference edges to
            # straight chords between source corners.
            source_edges = (
                connectivity.entities[1]
                if isinstance(connectivity, SimplicialConnectivity)
                else connectivity.edges
            )
            edge_count = source_edges.shape[0]
            ambient = mesh.coordinates.shape[1]
            # Edge rows and the retained length bank, the two chord gathers and
            # their difference, norm passes, and incident vertex extrema, masks,
            # gathers and ratios.
            retained = 24 * edge_count
            workspace.set_bound(
                retained + 32 * edge_count * ambient + 16 * edge_count
                + 48 * vertex_count + 4096
            )
            edges = np.asarray(source_edges, dtype=np.int64)
            lengths, growth = edge_size_evidence(
                np.asarray(mesh.coordinates, dtype=np.float64), edges
            )
            lower, upper = lengths, lengths
        workspace.set_bound(retained + 16 * edge_count + 4096)
        for control in specification.size_controls:
            match control:
                case UniformSizeControl():
                    pass
                case _:
                    raise TypeError(
                        "structured.size_controls must contain UniformSizeControl values; got a different control kind"
                    )
            wanted, measured, failed = uniform_size_compliance(
                control, specification.size_compliance, lengths, growth
            )
            requested.extend(wanted)
            achieved.extend(measured)
            issues.extend(failed)
            if lower is not lengths:
                for bound in (lower, upper):
                    _, _, bounded_failures = uniform_size_compliance(
                        control, specification.size_compliance, bound, growth
                    )
                    issues.extend(
                        name for name in bounded_failures if name not in issues
                    )
    if (
        isinstance(specification, SurfaceMeshingSpec)
        and specification.quality_target is not None
    ):
        with coordinate_phase(scope, specification.limits, budget):
            quality = summarize_cell_quality(
                evaluate_cell_quality(mesh, geometry=geometry, budget=budget)
            )
        quality_target = specification.quality_target
        requested.append(("minimum_angle", quality_target.minimum_angle))
        achieved.append(("minimum_angle", quality.minimum_angle))
        if quality_target.hard and (
            not np.isfinite(quality.minimum_angle)
            or quality.minimum_angle < quality_target.minimum_angle
        ):
            issues.append("minimum_angle")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _require_fidelity_source(
    source: SourceBoundaryQuery,
    tolerance: float,
    source_id: str,
    source_revision: str,
    /,
) -> None:
    if source.source_id != source_id or source.source_revision != source_revision:
        raise ValueError(
            "structured.fidelity_source must bind source_id and source_revision; got mismatched source identity"
        )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "structured.fidelity_tolerance must be finite and positive; got an invalid tolerance"
        )


def _require_source_scopes(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    source_id: str,
    source_revision: str,
    /,
) -> None:
    domain_scope = specification_scope(specification)
    scopes = (
        domain_scope,
        *(feature.scope for feature in specification.protected_features),
        *(control.scope for control in specification.region_controls),
        *(control.scope for control in specification.patch_controls),
    )
    if any(
        scope.source_id != source_id or scope.source_revision != source_revision
        for scope in scopes
    ):
        raise ValueError(
            "structured.specification.scopes must bind the declared source identity; got mismatched source ID or revision"
        )


def _cell_scope(mesh: CellMesh, block_index: int, /) -> MeshingScope:
    cell_entities = mesh.entity_set(mesh.topological_dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        mesh.topological_dimension,
        cell_entities.entity_set_id,
        mesh.blocks[block_index].global_ids,
    )


def _require_approximation_domain(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    /,
) -> None:
    match specification:
        case VolumeMeshingSpec():
            match domain:
                case PiecewiseLinearDomain() | MappedReferenceDomain():
                    if domain.ambient_dimension != 3:
                        raise ValueError(
                            "structured.domain.ambient_dimension must equal 3; got a different dimension"
                        )
                case None:
                    raise TypeError(
                        "structured.domain must declare the represented volume; got no volume domain"
                    )
                case invalid:
                    assert_never(invalid)
        case SurfaceMeshingSpec():
            if domain is not None:
                raise ValueError(
                    "structured.domain must be absent for a surface block request; got a volume domain"
                )
        case invalid:
            assert_never(invalid)


def _cell_regions(
    mesh: CellMesh,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    block_regions: tuple[tuple[str, str], ...],
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray | None:
    if domain is None:
        if block_regions:
            raise ValueError(
                "structured.block_regions must accompany a declared volume domain; got assignments without a domain"
            )
        return None
    names = tuple(block.name for block in mesh.blocks)
    if not block_regions:
        if len(domain.region_ids) != 1:
            raise ValueError(
                "structured.block_regions must identify every block when multiple regions are declared; got no assignments"
            )
        assignments = dict.fromkeys(names, domain.region_ids[0])
    else:
        assignments = dict(block_regions)
        if len(assignments) != len(block_regions) or set(assignments) != set(names):
            raise ValueError(
                "structured.block_regions must identify every generated block once; got missing or duplicate assignments"
            )
    if any(region not in domain.region_ids for region in assignments.values()):
        raise ValueError(
            "structured.block_regions.region_name must identify a declared domain region; got an undeclared region"
        )
    regions = {name: index for index, name in enumerate(domain.region_ids)}
    cells = sum(block.cell_count for block in mesh.blocks)
    with scope.host_workspace() as workspace:
        # Per-block region rows coexist until their concatenation.
        workspace.set_bound(8 * cells + 4096)
        return np.concatenate(
            tuple(
                np.full(
                    (block.cell_count,), regions[assignments[block.name]], dtype=np.int64
                )
                for block in mesh.blocks
            )
        )


def _controlled_organization(
    result: CellMeshingResult,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    regions: np.ndarray | None,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> CellMeshingResult:
    "Bind material/facet controls from the completed coverage relations."
    if not specification.region_controls and not specification.patch_controls:
        return result
    if not isinstance(domain, PiecewiseLinearDomain) or regions is None:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "structured.domain must declare piecewise-linear regions and facets for controls; got missing domain or cell-region rows",
            stage="structured_organization",
        )
    mesh = result.mesh
    cells = regions.size
    with scope.host_workspace() as workspace:
        # Converted cell IDs, one region mask and every zone's selected rows.
        workspace.set_bound(17 * cells + 4096)
        phase = _structured_region_zones(mesh, specification, domain, regions)
    controls = phase.controls
    zones = phase.zones
    zone_ids = phase.zone_ids
    coverage = None if result.certification is None else result.certification.coverage
    overlaps = (
        coverage.facet_source_overlaps.count
        if isinstance(coverage, DomainCoverageCertificate)
        else 0
    )
    sources = tuple(
        np.asarray(control.scope.entity_ids).size
        for control in specification.patch_controls
    )
    patch_rows = sum(
        np.asarray(patch.scope.global_entity_ids).size for patch in result.patches
    ) + overlaps * len(sources)
    with scope.host_workspace() as workspace:
        # CPython face groups of at most 400 bytes per overlap row, source sets
        # of at most 72 bytes per source row, selected face rows, and one label
        # union with its sort temporaries.
        workspace.set_bound(
            416 * overlaps
            + 72 * max(sources, default=0)
            + 64 * patch_rows
            + 4096
        )
        phase = _structured_patch_organization(
            result, specification, domain, controls, zone_ids
        )
    patches = phase.patches
    labels = phase.labels
    boundary_evidence = phase.boundary_evidence
    with coordinate_phase(scope, specification.limits, budget):
        audit = audit_cell_mesh(
            mesh,
            result.geometry,
            result.quality.evaluation,
            policy=CellMeshAuditPolicy(
                watertight_boundary=CellMeshAuditDisposition.REJECT
                if mesh.topological_dimension == 3
                else CellMeshAuditDisposition.SKIP
            ),
            prepared_validity=result.audit.validity,
            boundary=result.boundary,
            patches=tuple(patches),
            zones=tuple(zones),
            labels=tuple(labels),
            attributes=result.attributes,
            associations=result.associations,
            budget=budget,
        )
    audit.require_passed()
    certification = result.certification
    if certification is None:
        raise RuntimeError(
            "structured.result.certification must retain source certification; got no report"
        )
    certification = MeshCertificationReport(
        certification.schedule,
        mesh,
        result.geometry,
        audit,
        certification.outcomes,
        embedding=certification.embedding,
        coverage=certification.coverage,
        fidelity=certification.fidelity,
        request=certification.request,
        scoped_fidelity=certification.scoped_fidelity,
    )
    return CellMeshingResult(
        result.mesh,
        result.geometry,
        result.coordinate_contract,
        audit,
        audit.quality,
        result.compliance,
        result.trace,
        result.provider,
        result.runtime,
        result.derivative_mode,
        result.provenance,
        boundary=result.boundary,
        patches=tuple(patches),
        zones=tuple(zones),
        labels=tuple(labels),
        attributes=result.attributes,
        associations=result.associations,
        adapter_reports=result.adapter_reports,
        certification=certification,
        region_evidence=result.region_evidence,
        region_boundary_evidence=tuple(boundary_evidence),
        collective_evidence=result.collective_evidence,
        storage_binding=result.storage_binding,
        collective_certificates=result.collective_certificates,
        execution_evidence=result.execution_evidence,
        surface_source=result.surface_source,
    )


def _publish(
    mesh: CellMesh,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    publication: _StructuredPublicationSource,
    geometry: CellGeometrySpec,
    execution: _StructuredPublicationRun,
    /,
) -> CellMeshingResult:
    source_id = publication.source_id
    source_revision = publication.source_revision
    prepared_id = publication.prepared_id
    started = publication.started
    quantities = publication.quantities
    fidelity_source = publication.fidelity_source
    fidelity_tolerance = publication.fidelity_tolerance
    route_id = publication.route_id
    attributes = publication.attributes
    source_geometry_ids = publication.source_geometry_ids
    domain = publication.domain
    block_regions = publication.block_regions
    record_phase = execution.record_phase
    scope = execution.scope
    workspace = execution.workspace
    budget = execution.budget
    check_deadline(started, specification.limits, MeshingStageKind.GEOMETRY_AUDIT)
    actual_order = max(
        require_scalar_coordinate_element(element, "Structured publication").degree
        for element in geometry.elements
    )
    if actual_order != specification.target.geometry_order:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "structured.geometry.degree must equal target.geometry_order; got a mismatched coordinate basis degree",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            requested=(("geometry_order", specification.target.geometry_order),),
            achieved=(("geometry_order", actual_order),),
        )
    for dimension, maximum in (
        (1, specification.limits.maximum_edges),
        (2, specification.limits.maximum_faces),
    ):
        if mesh.entity_set(dimension).entity_ids.shape[0] > maximum:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "structured.mesh.entity_count must fit maximum_edges and maximum_faces; got an exceeded capacity",
                stage="structured_publication",
            )
    tolerances = [
        feature.maximum_deviation
        for feature in specification.protected_features
        if feature.hard
    ]
    tolerance = min((*tolerances, fidelity_tolerance))
    schedule = MeshCertificationSchedule(
        "surface"
        if mesh.topological_dimension == 2
        else "mapped_volume"
        if isinstance(domain, MappedReferenceDomain)
        else "volume_implicit"
    )
    certification = NativeCertificationRequest(
        schedule,
        source_id,
        source_revision,
        specification.limits,
        fidelity_source=fidelity_source,
        fidelity_tolerance=tolerance,
        domain=domain,
        cell_regions=_cell_regions(mesh, domain, block_regions, scope=scope),
    )
    stage = MeshingStageReport(
        MeshingStageKind.SURFACE_MESHING
        if mesh.topological_dimension == 2
        else MeshingStageKind.VOLUME_FILL,
        MeshingStageStatus.PASSED,
        input_ids=(prepared_id,),
        output_ids=(mesh.mesh_id,),
        diagnostics=(
            MeshingDiagnostic(
                MeshingDiagnosticSeverity.INFO,
                "Measured structured construction quantities.",
                provider_code="structured_construction",
                quantities=tuple((name, value, value) for name, value in quantities),
            ),
        ),
    )
    with measure_phase(record_phase, "compliance"):
        compliance = _compliance(
            mesh, geometry, specification, scope=scope, budget=budget
        )
    result = publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        (stage,),
        provider,
        {
            "route": route_id,
            "plan": plan_id,
            "specification": specification.specification_id,
            "prepared": prepared_id,
            "source_geometry_ids": source_geometry_ids,
        },
        certification,
        policy=publication_policy(
            "structured_volume"
            if mesh.topological_dimension == 3
            else "structured_surface"
        ),
        organization=NativePublicationOrganization(attributes=attributes),
        execution=NativePublicationExecution(
            scope=scope, workspace=workspace, record_phase=record_phase, budget=budget
        ),
        geometry=geometry,
    )
    result = _controlled_organization(
        result,
        specification,
        domain,
        certification.cell_regions,
        budget=budget,
        scope=scope,
    )
    data_bytes = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(result)
        if isinstance(leaf, (Array, np.ndarray))
    )
    if data_bytes > specification.limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "structured.result.data_bytes must not exceed maximum_data_bytes; got an oversized result",
            stage="structured_publication",
            requested=(("maximum_data_bytes", specification.limits.maximum_data_bytes),),
            achieved=(("data_bytes", data_bytes),),
        )
    check_deadline(
        started, specification.limits, MeshingStageKind.SPECIFICATION_COMPLIANCE
    )
    return result


@checked
def execute_structured_route(
    prepared: PreparedStructured,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    source_id: str,
    source_revision: str,
    source_binding_id: str,
    /,
    *,
    fidelity_source: SourceBoundaryQuery,
    fidelity_tolerance: float,
    domain: PiecewiseLinearDomain | None = None,
    block_regions: tuple[tuple[str, str], ...] = (),
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "structured"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            prepared.source_binding_id != source_binding_id
            or prepared.specification_id != specification.specification_id
        ):
            raise ValueError(
                "PreparedStructured.binding must match the source and specification IDs; got mismatched IDs"
            )
        _require_fidelity_source(
            fidelity_source, fidelity_tolerance, source_id, source_revision
        )
        if fidelity_source.ambient_dimension != specification.target.ambient_dimension:
            raise ValueError(
                "structured.fidelity_source.ambient_dimension must equal the target dimension; got mismatched dimensions"
            )
        _require_source_scopes(specification, source_id, source_revision)
        _require_approximation_domain(specification, domain)
        started = scope.operation_started
        # Face gluing preserves cell count. Admit aggregate cell topology before
        # construction; admit vertices per block and after gluing to account for
        # vertices shared across blocks. Each cell corner retains one int32
        # logical index per block axis as a published attribute.
        total_cells = sum(prod(block.intervals) for block in prepared.blocks)
        arity = 4 if specification.target.topological_dimension == 2 else 8
        logical_bytes = tuple(
            prod(block.intervals) * 2 ** len(block.intervals) * len(block.intervals) * 4
            for block in prepared.blocks
        )
        _preflight(
            0,
            total_cells,
            total_cells * arity,
            specification.target.ambient_dimension,
            specification,
            attribute_bytes=sum(logical_bytes),
        )
        for block, attribute_bytes in zip(prepared.blocks, logical_bytes, strict=True):
            shape = tuple(n + 1 for n in block.intervals)
            ambient = specification.target.ambient_dimension
            cells = prod(block.intervals)
            _preflight(
                prod(shape),
                cells,
                cells * arity,
                ambient,
                specification,
                attribute_bytes=attribute_bytes,
            )
        with measure_phase(record_phase, "construction"):
            constructions = tuple(
                (
                    construct_structured_block(
                        block,
                        optimization_steps=prepared.optimization_steps,
                        scope=scope,
                        budget=budget,
                    )
                    for block in prepared.blocks
                )
            )
        with measure_phase(record_phase, "topology_construction"):
            combined = construct_glued_structured_blocks(
                constructions, prepared.interfaces, scope=scope, budget=budget
            )
        rank = combined.mesh.topological_dimension
        entries = sum(b.vertices.size for b in combined.mesh.blocks)
        _preflight(
            combined.mesh.coordinates.shape[0],
            sum(b.cell_count for b in combined.mesh.blocks),
            entries,
            combined.mesh.ambient_dimension,
            specification,
            attribute_bytes=entries * rank * 4,
        )
        quantities = [("maximum_gluing_residual", combined.maximum_gluing_residual)]
        for block, construction in zip(prepared.blocks, constructions, strict=True):
            if construction.optimization is not None:
                quantities.extend(
                    (
                        (
                            f"{block.name}:optimization_status",
                            int(construction.optimization.status),
                        ),
                        (
                            f"{block.name}:optimization_iterations",
                            int(construction.optimization.diagnostics.iterations),
                        ),
                    )
                )
        attributes = []
        by_name = {
            construction.mesh.blocks[0].name: construction
            for construction in constructions
        }
        largest = max(b.vertices.size for b in combined.mesh.blocks)
        with scope.host_workspace() as attribute_workspace:
            # Converted corners, unravelled axes and their stacked bank coexist
            # per block; the narrowed attribute is admitted by the preflight.
            attribute_workspace.set_bound(8 * largest * (1 + 2 * rank) + 4096)
            for block_index, cell_block in enumerate(combined.mesh.blocks):
                construction = by_name[cell_block.name]
                local_vertices = np.asarray(
                    construction.mesh.blocks[0].vertices, dtype=np.int64
                )
                logical_indices = np.stack(
                    np.unravel_index(local_vertices, construction.logical_shape),
                    axis=-1,
                ).astype(np.int32)
                attributes.append(
                    MeshAttribute(
                        f"block:{cell_block.name}:logical_vertices",
                        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                        _cell_scope(combined.mesh, block_index),
                        logical_indices,
                    )
                )
        return _publish(
            combined.mesh,
            specification,
            coordinate_contract,
            provider,
            plan_id,
            _StructuredPublicationSource(
                source_id=source_id,
                source_revision=source_revision,
                prepared_id=prepared.prepared_id,
                started=started,
                quantities=tuple(quantities),
                fidelity_source=fidelity_source,
                fidelity_tolerance=fidelity_tolerance,
                route_id="structured_transfinite",
                attributes=tuple(attributes),
                source_geometry_ids=tuple((block.block_id for block in prepared.blocks)),
                domain=domain,
                block_regions=block_regions,
            ),
            combined.geometry,
            _StructuredPublicationRun(
                record_phase=record_phase, scope=scope, workspace=workspace, budget=budget
            ),
        )


@checked
def execute_sweep_route(
    prepared: PreparedSweep,
    specification: VolumeMeshingSpec,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    source_id: str,
    source_revision: str,
    source_binding_id: str,
    /,
    *,
    fidelity_source: SourceBoundaryQuery,
    fidelity_tolerance: float,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None = None,
    block_regions: tuple[tuple[str, str], ...] = (),
    root_correspondence: CellGeometryRestrictionSource | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "sweep"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            prepared.source_binding_id != source_binding_id
            or prepared.specification_id != specification.specification_id
        ):
            raise ValueError(
                "PreparedSweep.binding must match the source and specification IDs; got mismatched IDs"
            )
        _require_fidelity_source(
            fidelity_source, fidelity_tolerance, source_id, source_revision
        )
        if fidelity_source.ambient_dimension != specification.target.ambient_dimension:
            raise ValueError(
                "structured.fidelity_source.ambient_dimension must equal the target dimension; got mismatched dimensions"
            )
        _require_approximation_domain(specification, domain)
        _require_source_scopes(specification, source_id, source_revision)
        started = scope.operation_started
        layers = prepared.control.schedule.layer_count
        node_layers = layers if prepared.control.closed else layers + 1
        # Each column cell retains both layers' profile corner IDs, its profile
        # cell ID, its layer index and its two stations as published attributes.
        _preflight(
            node_layers * prepared.profile.coordinates.shape[0],
            layers * sum(b.cell_count for b in prepared.profile.blocks),
            layers
            * sum(
                b.cell_count * (6 if b.cell_kind == "triangle" else 8)
                for b in prepared.profile.blocks
            ),
            3,
            specification,
            attribute_bytes=layers
            * sum(
                b.cell_count * (16 * b.vertices.shape[1] + 28)
                for b in prepared.profile.blocks
            ),
        )
        # Every node layer copies the profile edges, and every profile cell and
        # edge sweeps one cap or side face per layer: a lower edge bound and the
        # exact face count, refused before construction.
        profile_edges = prepared.profile.entity_set(1).entity_ids.shape[0]
        profile_cells = sum(b.cell_count for b in prepared.profile.blocks)
        for count, maximum in (
            (node_layers * profile_edges, specification.limits.maximum_edges),
            (
                node_layers * profile_cells + layers * profile_edges,
                specification.limits.maximum_faces,
            ),
        ):
            if count > maximum:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "structured.mesh.entity_count must fit maximum_edges and maximum_faces; got an exceeded capacity",
                    stage="structured_preflight",
                )
        root_domain: MappedReferenceDomain | None = None
        match domain:
            case MappedReferenceDomain():
                match root_correspondence:
                    case CellGeometryRestrictionSource():
                        root_domain = domain
                    case None:
                        raise TypeError(
                            "sweep.root_correspondence must identify the mapped source roots; got no correspondence"
                        )
                    case invalid:
                        assert_never(invalid)
            case PiecewiseLinearDomain() | None:
                if root_correspondence is not None:
                    raise ValueError(
                        "sweep.domain must be mapped when root_correspondence is provided; got a different domain kind"
                    )
            case invalid:
                assert_never(invalid)
        with measure_phase(record_phase, "construction"):
            # Construction certifies the published geometry under publication's
            # caps, so publication reuses those request-earned certificates.
            construction = construct_sweep(
                prepared.profile,
                prepared.control,
                certificate_limits=derived_certificate_limits(specification.limits),
                profile_geometry=prepared.profile_geometry,
                root_domain=root_domain,
                root_correspondence=root_correspondence,
                scope=scope,
                budget=budget,
            )
        geometry = construction.geometry
        attributes = []
        profile_vertices_count = prepared.profile.coordinates.shape[0]
        measured = construction.measured_layer_thicknesses
        transient = 0
        for profile_block in prepared.profile.blocks:
            cells = profile_block.cell_count
            corners = profile_block.vertices.size
            transient = max(transient, 16 * corners + 20 * layers + 8 * cells)
        with scope.host_workspace() as attribute_workspace:
            # Converted profile vertex IDs stay live; each block's converted and
            # gathered corners, layer range, station pairs and converted cell IDs
            # coexist. Layered banks are retained data admitted by the preflight.
            attribute_workspace.set_bound(
                8 * profile_vertices_count + transient + 4096
            )
            source_ids = np.asarray(prepared.profile.vertex_global_ids, dtype=np.int64)
            for block_index, profile_block in enumerate(prepared.profile.blocks):
                cells = profile_block.cell_count
                attribute_scope = _cell_scope(construction.mesh, block_index)
                profile_vertices = source_ids[
                    np.asarray(profile_block.vertices, dtype=np.int64)
                ]
                corner_ids = np.tile(profile_vertices, (layers, 2))
                layer_indices = np.repeat(np.arange(layers, dtype=np.int32), cells)
                profile_cell_ids = np.tile(
                    np.asarray(profile_block.global_ids, dtype=np.int64), layers
                )
                stations = np.repeat(
                    np.stack(
                        (construction.stations[:-1], construction.stations[1:]),
                        axis=-1,
                    ),
                    cells,
                    axis=0,
                )
                attributes.extend(
                    (
                        MeshAttribute(
                            f"{profile_block.name}:source_profile_vertex_id",
                            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                            attribute_scope,
                            corner_ids,
                        ),
                        MeshAttribute(
                            f"{profile_block.name}:source_profile_cell_id",
                            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                            attribute_scope,
                            profile_cell_ids,
                        ),
                        MeshAttribute(
                            f"{profile_block.name}:sweep_layer_index",
                            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                            attribute_scope,
                            layer_indices,
                        ),
                        MeshAttribute(
                            f"{profile_block.name}:sweep_stations",
                            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                            attribute_scope,
                            stations,
                        ),
                    )
                )
            minimum_column_step = float(np.min(measured))
        return _publish(
            construction.mesh,
            specification,
            coordinate_contract,
            provider,
            plan_id,
            _StructuredPublicationSource(
                source_id=source_id,
                source_revision=source_revision,
                prepared_id=prepared.prepared_id,
                started=started,
                quantities=(
                    ("layers", layers),
                    ("minimum_column_step", minimum_column_step),
                ),
                fidelity_source=fidelity_source,
                fidelity_tolerance=fidelity_tolerance,
                route_id="sweep",
                attributes=tuple(attributes),
                source_geometry_ids=(
                    prepared.profile.mesh_id,
                    cell_geometry_id(prepared.profile_geometry),
                    prepared.control.control_id,
                ),
                domain=domain,
                block_regions=tuple(
                    ((f"sweep:{name}", region) for name, region in block_regions)
                ),
            ),
            geometry,
            _StructuredPublicationRun(
                record_phase=record_phase, scope=scope, workspace=workspace, budget=budget
            ),
        )


__all__ = [
    "PreparedStructured",
    "PreparedSweep",
    "structured_support_issues",
    "execute_structured_route",
    "execute_sweep_route",
]


@final
class _StructuredRegionPhase(StrictModule):
    controls: dict[str, RegionControl]
    zones: list[MeshZone]
    zone_ids: dict[str, str] = eqx.field(static=True)


def _structured_region_zones(
    mesh: CellMesh,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: PiecewiseLinearDomain,
    regions: np.ndarray,
) -> _StructuredRegionPhase:
    """Structured region zones."""
    controls = {control.region_name: control for control in specification.region_controls}
    if len(controls) != len(specification.region_controls):
        raise ValueError(
            "structured.region_controls must contain at most one control per region; got duplicate region names"
        )
    if any(name not in domain.region_ids for name in controls):
        raise ValueError(
            "structured.region_controls.region_name must identify a declared region; got an undeclared name"
        )

    zones = []
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    for index, name in enumerate(domain.region_ids):
        control = controls.get(name)
        selected = cell_ids[regions == index]
        if control is not None:
            if (
                control.scope.entity_kind is not MeshingEntityKind.GEOMETRY
                or not np.array_equal(
                    np.asarray(control.scope.entity_ids),
                    np.asarray((index,), dtype=np.int64),
                )
            ):
                raise ValueError(
                    "structured.region_controls.scope must identify the declared geometry region row; got a mismatched region scope"
                )
            if selected.size and not control.meshing_enabled:
                raise MeshingFailure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "structured.region_controls.meshing_enabled must allow every populated region; got blocks in a disabled region",
                    stage="structured_organization",
                )
        if selected.size:
            zones.append(
                MeshZone(
                    name,
                    MeshZoneRole.REGION,
                    mesh_entity_scope(mesh, 3, selected),
                    material_id=None if control is None else control.material_id,
                    region_role=None if control is None else control.role,
                )
            )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    return _StructuredRegionPhase(controls, zones, zone_ids)


@final
class _StructuredPatchPhase(StrictModule):
    patches: list[MeshPatch]
    labels: list[MeshLabel]
    boundary_evidence: list[RegionBoundaryEvidence]


def _structured_patch_organization(
    result: CellMeshingResult,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: PiecewiseLinearDomain,
    controls: dict[str, RegionControl],
    zone_ids: dict[str, str],
) -> _StructuredPatchPhase:
    """Structured patch organization."""
    mesh = result.mesh
    report = result.certification
    coverage = None if report is None else report.coverage
    if (
        not isinstance(coverage, DomainCoverageCertificate)
        or coverage.status != "certified"
        or coverage.domain_id != domain.domain_id
    ):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "structured.result.certification.coverage must certify the declared domain; got missing, uncertified, or mismatched coverage",
            stage="structured_organization",
        )
    overlap_rows = coverage.facet_source_overlaps
    positive = overlap_rows.upper > 0.0
    if np.any(
        positive
        & ~(overlap_rows.semantic_mask("exact") & overlap_rows.space_mask("physical"))
    ):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "structured.coverage.facet_source_overlaps must use physical source-certified overlap semantics; got candidate overlap evidence",
            stage="structured_organization",
        )
    face_sources: dict[int, set[int]] = {}
    for face, source in zip(
        overlap_rows.facets[positive].tolist(),
        overlap_rows.sources[positive].tolist(),
        strict=True,
    ):
        face_sources.setdefault(face, set()).add(source)
    patches = list(result.patches)
    labels = list(result.labels)
    boundary_evidence: list[RegionBoundaryEvidence] = []
    for control in specification.patch_controls:
        source_rows = np.asarray(control.scope.entity_ids)
        selected_sources = set(map(int, source_rows))
        if control.scope.entity_kind is not MeshingEntityKind.GEOMETRY or any(
            source < 0 or source >= domain.facets.shape[0] for source in selected_sources
        ):
            raise ValueError(
                "structured.patch_controls.scope must identify declared geometry facet rows; got an invalid facet scope"
            )
        adjacent = tuple(sorted(control.adjacent_region_names))
        oriented_pairs = {
            tuple(
                None if index < 0 else domain.region_ids[index]
                for index in domain.facet_regions[source]
            )
            for source in selected_sources
        }
        if len(oriented_pairs) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "structured.patch_controls.source_incidence must identify one oriented region pair; got mixed source incidences",
                stage="structured_organization",
            )
        oriented = next(iter(oriented_pairs))
        source_pair: tuple[str | None, str | None] = (
            oriented[0],
            oriented[1],
        )
        for source in selected_sources:
            actual = tuple(
                sorted(
                    domain.region_ids[index]
                    for index in domain.facet_regions[source]
                    if index >= 0
                )
            )
            if actual != adjacent:
                raise ValueError(
                    "structured.patch_controls.adjacent_region_names must equal source facet incidence; got mismatched adjacency"
                )
        selected_faces = []
        for face, sources in face_sources.items():
            if sources & selected_sources:
                if not sources <= selected_sources:
                    raise MeshingFailure(
                        MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "structured.patch_controls.source_facets must cover each selected generated face; got a partially selected face",
                        stage="structured_organization",
                    )
                selected_faces.append(face)
        if control.required and not selected_faces:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "structured.patch_controls must publish child faces for every required patch; got an empty patch",
                stage="structured_organization",
            )
        if selected_faces:
            patches.append(
                MeshPatch(
                    control.name,
                    mesh_entity_scope(
                        mesh, 2, np.asarray(sorted(selected_faces), dtype=np.int64)
                    ),
                    adjacent_zone_ids=tuple(zone_ids[name] for name in adjacent),
                    source_adjacent_region_ids=source_pair,
                )
            )
    phase = _structured_region_boundaries(
        mesh, domain, controls, patches, labels, boundary_evidence
    )
    labels = phase.labels
    boundary_evidence = phase.boundary_evidence
    return _StructuredPatchPhase(patches, labels, boundary_evidence)


@final
class _StructuredBoundaryPhase(StrictModule):
    labels: list[MeshLabel]
    boundary_evidence: list[RegionBoundaryEvidence]


def _structured_region_boundaries(
    mesh: CellMesh,
    domain: PiecewiseLinearDomain,
    controls: dict[str, RegionControl],
    patches: list[MeshPatch],
    labels: list[MeshLabel],
    boundary_evidence: list[RegionBoundaryEvidence],
) -> _StructuredBoundaryPhase:
    """Structured region boundaries."""
    for index, name in enumerate(domain.region_ids):
        sides = tuple(
            (
                patch.patch_id,
                1 if patch.source_adjacent_region_ids[0] == name else -1,
            )
            for patch in patches
            if patch.source_adjacent_region_ids is not None
            and name in patch.source_adjacent_region_ids
        )
        if not sides:
            continue
        control = controls.get(name)
        source_scope = (
            control.scope
            if control is not None
            else MeshingScope(
                domain.source_id,
                domain.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                canonical_fingerprint(
                    {
                        "kind": "piecewise-linear-domain-region-set",
                        "domain": domain.domain_id,
                        "regions": domain.region_ids,
                    }
                ),
                np.asarray((index,), dtype=np.int64),
            )
        )
        side_ids = {identifier for identifier, _ in sides}
        rows = tuple(
            np.asarray(patch.scope.global_entity_ids, dtype=np.int64)
            for patch in patches
            if patch.patch_id in side_ids
        )
        selected = np.unique(np.concatenate(rows))
        label = MeshLabel(
            f"source-region-boundary:{name}", mesh_entity_scope(mesh, 2, selected)
        )
        labels.append(label)
        boundary_evidence.append(
            RegionBoundaryEvidence(
                mesh,
                source_scope,
                name,
                domain.domain_id if control is None else control.control_id,
                None if control is None else control.material_id,
                None if control is None else control.role,
                label,
                sides,
            )
        )
    return _StructuredBoundaryPhase(labels, boundary_evidence)


@final
class _StructuredPublicationSource(StrictModule):
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    started: float = eqx.field(static=True)
    quantities: tuple[tuple[str, float], ...]
    fidelity_source: SourceBoundaryQuery
    fidelity_tolerance: float = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    attributes: tuple[MeshAttribute, ...]
    source_geometry_ids: tuple[str, ...]
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None
    block_regions: tuple[tuple[str, str], ...]


@final
class _StructuredPublicationRun(StrictModule):
    record_phase: NativeMeshingPhaseRecorder | None
    scope: NativeExecutionScope
    workspace: NativeHostStorageWorkspace
    budget: CoordinateEnclosureBudget
