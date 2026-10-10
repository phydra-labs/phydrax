#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared structured/swept routes using the common native publication owner."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from time import monotonic
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import charge_native_geometry_queries, current_native_execution_budget
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._cell_complex import IntervalConnectivity
from ...discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometryElement,
    CellGeometryRestrictionSource,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import SimplicialConnectivity
from ...discretization._coordinate_enclosure import coordinate_polynomials
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._mesh_certificates import PiecewiseLinearDomain, SourceBoundaryQuery
from .._audit import (
    audit_cell_mesh,
    CellMeshAuditDisposition,
    CellMeshAuditPolicy,
)
from .._certification import MeshCertificationReport, MeshCertificationSchedule
from .._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._controls import BlockInterfaceControl
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._multiblock import glue_structured_blocks
from .._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionBoundaryEvidence,
)
from .._quad_generation import _entities, _family_host_array
from .._quality import evaluate_cell_quality, summarize_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._structured import generate_structured_block, TransfiniteBlock
from .._sweep import generate_sweep, SweepControl
from .._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from ._native_publication import (
    check_deadline,
    edge_size_evidence,
    NativeCertificationRequest,
    publish_native_result,
    uniform_size_compliance,
)


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
    scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
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
            raise TypeError("Structured preparation requires TransfiniteBlock values.")
        if len({block.name for block in blocks}) != len(blocks):
            raise ValueError("Structured block names must be unique.")
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError(
                "Structured preparation requires a surface or volume specification."
            )
        if any(
            len(block.intervals) != specification.target.topological_dimension
            for block in blocks
        ):
            raise ValueError("Declared block dimensions must match the physical target.")
        if any(
            not isinstance(i, BlockInterfaceControl) or not i.conforming
            for i in interfaces
        ):
            raise ValueError(
                "One CellMeshingResult requires conforming block interfaces; independent parts use MeshAssembly."
            )
        if (
            isinstance(optimization_steps, bool)
            or not isinstance(optimization_steps, int)
            or optimization_steps < 0
        ):
            raise ValueError("optimization_steps must be a nonnegative integer.")
        if not source_binding_id:
            raise ValueError("Prepared routes require explicit source binding identity.")
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

    def __init__(
        self,
        profile: CellMesh,
        control: SweepControl,
        specification: VolumeMeshingSpec,
        source_binding_id: str,
        /,
        *,
        profile_geometry: CellGeometrySpec | None = None,
    ) -> None:
        if not isinstance(profile, CellMesh) or not isinstance(control, SweepControl):
            raise TypeError("Sweep preparation requires CellMesh and SweepControl.")
        if not isinstance(specification, VolumeMeshingSpec):
            raise TypeError("Sweep preparation requires VolumeMeshingSpec.")
        if profile.topological_dimension != 2 or profile.ambient_dimension not in (2, 3):
            raise ValueError("Sweep profiles require two-dimensional cells in 2D or 3D.")
        if any(
            block.cell_kind not in ("triangle", "quadrilateral")
            for block in profile.blocks
        ):
            raise ValueError(
                "Prism/hex sweeps require triangle or quadrilateral source cells."
            )
        geometry = (
            CellGeometrySpec.affine(profile)
            if profile_geometry is None
            else profile_geometry
        )
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("profile_geometry must be CellGeometrySpec or None.")
        geometry.resolve(profile)
        from ...discretization.fem._reference import FiniteElementSpec

        if any(
            not isinstance(element, FiniteElementSpec) for element in geometry.elements
        ):
            raise ValueError(
                "Sweeps require unrestricted canonical polynomial profile maps."
            )
        degrees = tuple(
            element.degree
            for element in geometry.elements
            if isinstance(element, FiniteElementSpec)
        )
        if max(degrees) != specification.target.geometry_order:
            raise ValueError(
                "The requested geometry order must match the actual source profile order."
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
            raise ValueError("Prepared routes require explicit source binding identity.")
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
) -> None:
    limits = specification.limits
    estimates = (
        (vertices, limits.maximum_vertices, "vertices"),
        (cells, limits.maximum_cells, "cells"),
        (connectivity_entries, limits.maximum_connectivity_entries, "connectivity"),
        (
            vertices * ambient * 8 + connectivity_entries * 4,
            limits.maximum_data_bytes,
            "data",
        ),
    )
    for actual, maximum, name in estimates:
        if actual > maximum:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"Structured {name} capacity {actual} exceeds {maximum}.",
                stage="structured_preflight",
            )


def _mapped_edge_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Certified complete-edge source arc lengths, never curved corner chords."""
    from ...discretization._cell_geometry_transfer import (
        _prepare_mapped_edge_arc_length,
        CellGeometryTransitionError,
    )
    from ...discretization._coordinate_enclosure import (
        coordinate_enclosure_budget,
        CoordinateEnclosureResourceError,
    )
    from ...discretization._reference_cell import reference_cell_topology

    edges = _entities(mesh, 1)
    lookup = {tuple(sorted(pair)): row for row, pair in enumerate(edges.tolist())}
    lengths = _family_host_array((edges.shape[0],), np.float64)
    lower = _family_host_array(lengths.shape, np.float64)
    upper = _family_host_array(lengths.shape, np.float64)
    lengths.fill(np.nan)
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    ledger = coordinate_enclosure_budget(
        specification.limits.maximum_work_units,
        specification.limits.maximum_scratch_bytes,
    )
    budget = current_native_execution_budget()
    work = specification.limits.maximum_work_units
    if budget is not None:
        work = min(
            work,
            max(
                0,
                budget.remaining().remaining_work_units
                - (ledger.work_units - ledger.native_charged_work_units),
            ),
        )
    try:
        with (
            ledger.activate(),
            ledger.bound_stage(work, specification.limits.maximum_scratch_bytes),
        ):
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
                topology = reference_cell_topology(block.cell_kind)
                points = tuple(
                    tuple(Fraction(float(value)) for value in point)
                    for point in topology.vertices
                )
                for vertices, dofs in zip(
                    np.asarray(block.vertices).tolist(),
                    np.asarray(route).tolist(),
                    strict=True,
                ):
                    local = tuple(coefficients[index] for index in dofs)
                    for first, last in topology.entities[1]:
                        row = lookup[tuple(sorted((vertices[first], vertices[last])))]
                        if np.isfinite(lengths[row]):
                            continue
                        charge_native_geometry_queries(1)
                        enclosure = _prepare_mapped_edge_arc_length(
                            element, local, points[first], points[last]
                        ).integrate(
                            absolute_tolerance=1e-12,
                            relative_tolerance=1e-12,
                            maximum_work=specification.limits.maximum_work_units,
                            maximum_subcells=min(
                                specification.limits.maximum_cells, 10000
                            ),
                        )
                        lengths[row], lower[row], upper[row] = (
                            enclosure.value,
                            enclosure.lower,
                            enclosure.upper,
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
    finally:
        ledger.charge_native_work(ledger.work_units - ledger.native_charged_work_units)
    if not np.all(np.isfinite(lengths)):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Structured physical edge arc coverage is incomplete.",
            stage="structured_edge_arcs",
        )
    return edges, lengths, lower, upper


def _compliance(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    /,
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
            "Structured publication requires surface or volume cell connectivity."
        )
    connectivity = mesh.connectivity
    if isinstance(connectivity, IntervalConnectivity):
        raise TypeError(
            "Structured surface/volume publication cannot use interval connectivity."
        )
    edges = np.asarray(
        connectivity.entities[1]
        if isinstance(connectivity, SimplicialConnectivity)
        else connectivity.edges,
        dtype=np.int64,
    )
    if any(
        _require_scalar_coordinate_element(element, "Structured physical sizing").degree
        > 1
        for element in geometry.elements
    ):
        edges, lengths, lower, upper = _mapped_edge_evidence(
            mesh, geometry, specification
        )
        minimum = _family_host_array((mesh.coordinates.shape[0],), np.float64)
        maximum = _family_host_array(minimum.shape, np.float64)
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
        # The owning unrestricted linear tensor/simplex basis has straight
        # reference edges, so correctly bound source corners define exact chords.
        lengths, growth = edge_size_evidence(
            np.asarray(mesh.coordinates, dtype=np.float64), edges
        )
        lower, upper = lengths, lengths
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError("Admitted structured sizing must be UniformSizeControl.")
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
                issues.extend(name for name in bounded_failures if name not in issues)
    if (
        isinstance(specification, SurfaceMeshingSpec)
        and specification.quality_target is not None
    ):
        quality = summarize_cell_quality(evaluate_cell_quality(mesh, geometry=geometry))
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
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError(
            "Structured publication requires an authoritative SourceBoundaryQuery."
        )
    if source.source_id != source_id or source.source_revision != source_revision:
        raise ValueError(
            "The fidelity query must bind the exact declared source revision."
        )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("fidelity_tolerance must be finite and positive.")


def _require_source_scopes(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    source_id: str,
    source_revision: str,
    /,
) -> None:
    domain_scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
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
            "Structured physical scopes must bind the exact authoritative source revision."
        )


def _cell_scope(mesh: CellMesh, block_index: int, /) -> MeshingScope:
    entities = mesh.entity_set(mesh.topological_dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        mesh.topological_dimension,
        entities.entity_set_id,
        mesh.blocks[block_index].global_ids,
    )


def _require_approximation_domain(
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    /,
) -> None:
    if isinstance(specification, VolumeMeshingSpec):
        if not isinstance(domain, (PiecewiseLinearDomain, MappedReferenceDomain)):
            raise TypeError(
                "Volume publication requires an independently declared represented approximation domain."
            )
        if domain.ambient_dimension != 3:
            raise ValueError(
                "The represented volume domain must use three-dimensional coordinates."
            )
    elif domain is not None:
        raise ValueError(
            "Surface block publication does not consume a volume approximation domain."
        )


def _cell_regions(
    mesh: CellMesh,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    block_regions: tuple[tuple[str, str], ...],
    /,
) -> np.ndarray | None:
    if domain is None:
        if block_regions:
            raise ValueError(
                "Block regions require a declared represented volume domain."
            )
        return None
    names = tuple(block.name for block in mesh.blocks)
    if not block_regions:
        if len(domain.region_ids) != 1:
            raise ValueError(
                "Multiple source regions require explicit region identity for every named block."
            )
        assignments = dict.fromkeys(names, domain.region_ids[0])
    else:
        assignments = dict(block_regions)
        if len(assignments) != len(block_regions) or set(assignments) != set(names):
            raise ValueError(
                "Region assignments must identify every exact generated block once."
            )
    if any(region not in domain.region_ids for region in assignments.values()):
        raise ValueError("A block region is absent from the declared represented domain.")
    regions = {name: index for index, name in enumerate(domain.region_ids)}
    return np.concatenate(
        tuple(
            np.full((block.cell_count,), regions[assignments[block.name]], dtype=np.int64)
            for block in mesh.blocks
        )
    )


def _controlled_organization(
    result: CellMeshingResult,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    regions: np.ndarray | None,
    /,
) -> CellMeshingResult:
    """Bind original material/facet controls from the completed exact coverage relations."""
    if not specification.region_controls and not specification.patch_controls:
        return result
    if not isinstance(domain, PiecewiseLinearDomain) or regions is None:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Controlled structured strata require declared piecewise-linear source region/facet authority.",
            stage="structured_organization",
        )
    controls = {control.region_name: control for control in specification.region_controls}
    if len(controls) != len(specification.region_controls):
        raise ValueError("A structured source region may have only one material control.")
    if any(name not in domain.region_ids for name in controls):
        raise ValueError(
            "A structured material control names an undeclared source region."
        )
    mesh = result.mesh

    def scope(dimension: int, identifiers: np.ndarray) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            dimension,
            mesh.entity_set(dimension).entity_set_id,
            identifiers,
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
                    "Structured material controls must name the exact declared source region row."
                )
            if selected.size and not control.meshing_enabled:
                raise MeshingFailure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "A disabled source region still owns declared structured blocks.",
                    stage="structured_organization",
                )
        if selected.size:
            zones.append(
                MeshZone(
                    name,
                    MeshZoneRole.REGION,
                    scope(3, selected),
                    material_id=None if control is None else control.material_id,
                    region_role=None if control is None else control.role,
                )
            )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    report = result.certification
    coverage = None if report is None else report.coverage
    if (
        coverage is None
        or coverage.status != "certified"
        or coverage.domain_id != domain.domain_id
    ):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Structured patch controls require the original completed exact source coverage.",
            stage="structured_organization",
        )
    face_sources: dict[int, set[int]] = {}
    for (
        face,
        source,
        axes,
        lower,
        upper,
        semantics,
        space,
    ) in coverage.facet_source_overlaps:
        if upper <= 0.0:
            continue
        if semantics != "exact" or space != "physical":
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "Candidate source overlap cannot assign a controlled structured facet.",
                stage="structured_organization",
            )
        face_sources.setdefault(face, set()).add(source)
    patches = list(result.patches)
    labels = list(result.labels)
    boundary_evidence: list[RegionBoundaryEvidence] = []
    for control in specification.patch_controls:
        selected_sources = set(map(int, np.asarray(control.scope.entity_ids)))
        if control.scope.entity_kind is not MeshingEntityKind.GEOMETRY or any(
            source < 0 or source >= domain.facets.shape[0] for source in selected_sources
        ):
            raise ValueError(
                "Structured patch controls must name exact declared source facet rows."
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
                "A controlled source patch needs one exact oriented region incidence.",
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
                    "Structured patch adjacency differs from the original source facet incidence."
                )
        selected_faces = []
        for face, sources in face_sources.items():
            if sources & selected_sources:
                if not sources <= selected_sources:
                    raise MeshingFailure(
                        MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "A controlled source patch cuts through a generated scientific face.",
                        stage="structured_organization",
                    )
                selected_faces.append(face)
        if control.required and not selected_faces:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "A required structured source patch has no published child faces.",
                stage="structured_organization",
            )
        if selected_faces:
            patches.append(
                MeshPatch(
                    control.name,
                    scope(2, np.asarray(sorted(selected_faces), dtype=np.int64)),
                    adjacent_zone_ids=tuple(zone_ids[name] for name in adjacent),
                    source_adjacent_region_ids=source_pair,
                )
            )
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
        selected = np.unique(
            np.concatenate(
                tuple(
                    np.asarray(patch.scope.global_entity_ids, dtype=np.int64)
                    for patch in patches
                    if patch.patch_id in {identifier for identifier, _ in sides}
                )
            )
        )
        label = MeshLabel(f"source-region-boundary:{name}", scope(2, selected))
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
    audit = audit_cell_mesh(
        mesh,
        result.geometry,
        result.quality.evaluation,
        policy=CellMeshAuditPolicy(
            watertight_boundary=(
                CellMeshAuditDisposition.REJECT
                if mesh.topological_dimension == 3
                else CellMeshAuditDisposition.SKIP
            )
        ),
        prepared_validity=result.audit.validity,
        boundary=result.boundary,
        patches=tuple(patches),
        zones=tuple(zones),
        labels=tuple(labels),
        attributes=result.attributes,
        associations=result.associations,
    )
    audit.require_passed()
    certification = result.certification
    if certification is None:
        raise RuntimeError(
            "Controlled structured organization lost its completed source certification."
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
    source_id: str,
    source_revision: str,
    prepared_id: str,
    started: float,
    quantities: tuple[tuple[str, float], ...],
    /,
    *,
    fidelity_source: SourceBoundaryQuery,
    fidelity_tolerance: float,
    route_id: str,
    attributes: tuple[MeshAttribute, ...],
    source_geometry_ids: tuple[str, ...],
    domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    block_regions: tuple[tuple[str, str], ...],
    geometry: CellGeometrySpec,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> CellMeshingResult:
    check_deadline(started, specification.limits, MeshingStageKind.GEOMETRY_AUDIT)
    actual_order = max(
        _require_scalar_coordinate_element(element, "Structured publication").degree
        for element in geometry.elements
    )
    if actual_order != specification.target.geometry_order:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "The authoritative coordinate basis does not meet the requested geometry order.",
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
                "Structured entity capacity exceeded.",
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
        cell_regions=_cell_regions(mesh, domain, block_regions),
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
        compliance = _compliance(mesh, geometry, specification)
    result = publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        (stage,),
        provider,
        {
            "route": route_id,
            "plan": plan_id,
            "prepared": prepared_id,
            "source_geometry_ids": source_geometry_ids,
        },
        certification,
        audit_policy=CellMeshAuditPolicy(
            watertight_boundary=CellMeshAuditDisposition.REJECT
            if mesh.topological_dimension == 3
            else CellMeshAuditDisposition.SKIP
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        attributes=attributes,
        geometry=geometry,
        record_phase=record_phase,
        enforced_limits=(
            "maximum_vertices",
            "maximum_cells",
            "maximum_edges",
            "maximum_faces",
            "maximum_connectivity_entries",
            "maximum_data_bytes",
            "maximum_wall_seconds",
        ),
        unenforced_limits=(
            "maximum_work_units",
            "maximum_cavity_cells",
            "maximum_geometry_queries",
            "maximum_scratch_bytes",
        ),
    )
    result = _controlled_organization(
        result, specification, domain, certification.cell_regions
    )
    data_bytes = sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(result)
        if isinstance(leaf, (Array, np.ndarray))
    )
    if data_bytes > specification.limits.maximum_data_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Published structured numerical data exceed the declared byte limit.",
            stage="structured_publication",
            requested=(("maximum_data_bytes", specification.limits.maximum_data_bytes),),
            achieved=(("data_bytes", data_bytes),),
        )
    check_deadline(
        started, specification.limits, MeshingStageKind.SPECIFICATION_COMPLIANCE
    )
    return result


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
) -> CellMeshingResult:
    if (
        prepared.source_binding_id != source_binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "The prepared structured route binds another source or specification."
        )
    _require_fidelity_source(
        fidelity_source, fidelity_tolerance, source_id, source_revision
    )
    if fidelity_source.ambient_dimension != specification.target.ambient_dimension:
        raise ValueError(
            "The authoritative boundary query must use the target coordinate dimension."
        )
    _require_source_scopes(specification, source_id, source_revision)
    _require_approximation_domain(specification, domain)
    started = monotonic()
    # Cells are never merged by face gluing. Admit the complete topology before
    # constructing any constituent; vertex admission remains exact per block
    # and after the declared gluing rather than assuming no shared vertices.
    total_cells = sum(prod(block.intervals) for block in prepared.blocks)
    arity = 4 if specification.target.topological_dimension == 2 else 8
    _preflight(
        0,
        total_cells,
        total_cells * arity,
        specification.target.ambient_dimension,
        specification,
    )
    for block in prepared.blocks:
        shape = tuple(n + 1 for n in block.intervals)
        ambient = specification.target.ambient_dimension
        cells = prod(block.intervals)
        _preflight(prod(shape), cells, cells * arity, ambient, specification)
    with measure_phase(record_phase, "construction"):
        constructions = tuple(
            generate_structured_block(
                block, optimization_steps=prepared.optimization_steps
            )
            for block in prepared.blocks
        )
    with measure_phase(record_phase, "topology_construction"):
        combined = glue_structured_blocks(constructions, prepared.interfaces)
    _preflight(
        combined.mesh.coordinates.shape[0],
        sum(b.cell_count for b in combined.mesh.blocks),
        sum(b.vertices.size for b in combined.mesh.blocks),
        combined.mesh.ambient_dimension,
        specification,
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
        construction.mesh.blocks[0].name: construction for construction in constructions
    }
    for block_index, cell_block in enumerate(combined.mesh.blocks):
        construction = by_name[cell_block.name]
        local_vertices = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
        logical_indices = np.stack(
            np.unravel_index(local_vertices, construction.logical_shape), axis=-1
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
        source_id,
        source_revision,
        prepared.prepared_id,
        started,
        tuple(quantities),
        fidelity_source=fidelity_source,
        fidelity_tolerance=fidelity_tolerance,
        route_id="structured_transfinite",
        attributes=tuple(attributes),
        source_geometry_ids=tuple(block.block_id for block in prepared.blocks),
        geometry=combined.geometry,
        domain=domain,
        block_regions=block_regions,
        record_phase=record_phase,
    )


def _sweep_root_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MappedReferenceDomain,
    correspondence: CellGeometryRestrictionSource,
    /,
) -> CellGeometrySpec:
    """Retain an authored root basis only after exact source-expression equality."""
    root_elements, root_routes, root_coordinates = domain.source_geometry.resolve(
        domain.reference_mesh
    )
    root_values = np.asarray(root_coordinates, dtype=np.float64)
    roots = {
        int(identifier): (element, np.asarray(route, dtype=np.int32), block.cell_kind)
        for block, element, routes in zip(
            domain.reference_mesh.blocks, root_elements, root_routes, strict=True
        )
        for identifier, route in zip(
            np.asarray(block.global_ids), np.asarray(routes), strict=True
        )
    }
    elements, routes, coordinates = geometry.resolve(mesh)
    values = np.asarray(coordinates, dtype=np.float64)
    bound_elements: dict[str, CellGeometryElement] = {}
    bound_routes: dict[str, np.ndarray] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        parents = correspondence.block_parent_cell_ids[block.name]
        selected = []
        for cell_id, local_route, parent in zip(
            np.asarray(block.global_ids),
            np.asarray(route),
            np.asarray(parents),
            strict=True,
        ):
            if int(parent) not in roots:
                raise ValueError(
                    "Sweep correspondence names an absent authoritative source root."
                )
            root_element, root_route, root_kind = roots[int(parent)]
            if root_kind != block.cell_kind:
                raise ValueError(
                    "Swept cells require the declared source root reference family."
                )
            generated = coordinate_polynomials(element, values[local_route])
            authored = coordinate_polynomials(root_element, root_values[root_route])
            if generated is None or authored is None or generated != authored:
                raise MeshingFailure(
                    MeshingFailureCategory.AUDIT_FAILED,
                    "Swept coordinate map differs from its explicitly declared authoritative source root.",
                    stage="sweep_source_map",
                    entity_ids=(int(cell_id),),
                )
            previous = bound_elements.get(block.name)
            if previous is not None and previous.element_id != root_element.element_id:
                raise ValueError(
                    "One swept block requires one authoritative root coordinate basis."
                )
            bound_elements[block.name] = root_element
            selected.append(root_route)
        bound_routes[block.name] = np.stack(selected)
    return CellGeometrySpec(
        bound_elements, bound_routes, root_coordinates, restriction_source=correspondence
    )


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
) -> CellMeshingResult:
    if (
        prepared.source_binding_id != source_binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "The prepared sweep route binds another source or specification."
        )
    _require_fidelity_source(
        fidelity_source, fidelity_tolerance, source_id, source_revision
    )
    if fidelity_source.ambient_dimension != specification.target.ambient_dimension:
        raise ValueError(
            "The authoritative boundary query must use the target coordinate dimension."
        )
    _require_approximation_domain(specification, domain)
    _require_source_scopes(specification, source_id, source_revision)
    started = monotonic()
    layers = prepared.control.schedule.layer_count
    node_layers = layers if prepared.control.closed else layers + 1
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
    )
    with measure_phase(record_phase, "construction"):
        construction = generate_sweep(
            prepared.profile, prepared.control, profile_geometry=prepared.profile_geometry
        )
    geometry = construction.geometry
    if isinstance(domain, MappedReferenceDomain):
        if not isinstance(root_correspondence, CellGeometryRestrictionSource):
            raise TypeError(
                "Mapped sweep publication requires its explicit authored root correspondence."
            )
        geometry = _sweep_root_geometry(
            construction.mesh, geometry, domain, root_correspondence
        )
    elif root_correspondence is not None:
        raise ValueError(
            "Sweep root correspondence requires an independently authored mapped domain."
        )
    attributes = []
    source_ids = np.asarray(prepared.profile.vertex_global_ids, dtype=np.int64)
    for block_index, profile_block in enumerate(prepared.profile.blocks):
        scope = _cell_scope(construction.mesh, block_index)
        profile_vertices = source_ids[np.asarray(profile_block.vertices, dtype=np.int64)]
        corner_ids = np.tile(profile_vertices, (layers, 2))
        layer_indices = np.repeat(
            np.arange(layers, dtype=np.int32), profile_block.cell_count
        )
        profile_cell_ids = np.tile(
            np.asarray(profile_block.global_ids, dtype=np.int64), layers
        )
        stations = np.repeat(
            np.stack((construction.stations[:-1], construction.stations[1:]), axis=-1),
            profile_block.cell_count,
            axis=0,
        )
        attributes.extend(
            (
                MeshAttribute(
                    f"{profile_block.name}:source_profile_vertex_id",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    scope,
                    corner_ids,
                ),
                MeshAttribute(
                    f"{profile_block.name}:source_profile_cell_id",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    scope,
                    profile_cell_ids,
                ),
                MeshAttribute(
                    f"{profile_block.name}:sweep_layer_index",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    scope,
                    layer_indices,
                ),
                MeshAttribute(
                    f"{profile_block.name}:sweep_stations",
                    MeshAttributeRole.GEOMETRY_CLASSIFICATION,
                    scope,
                    stations,
                ),
            )
        )
    return _publish(
        construction.mesh,
        specification,
        coordinate_contract,
        provider,
        plan_id,
        source_id,
        source_revision,
        prepared.prepared_id,
        started,
        (
            ("layers", layers),
            (
                "minimum_column_step",
                float(np.min(construction.measured_layer_thicknesses)),
            ),
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
        geometry=geometry,
        domain=domain,
        block_regions=tuple((f"sweep:{name}", region) for name, region in block_regions),
        record_phase=record_phase,
    )


__all__ = [
    "PreparedStructured",
    "PreparedSweep",
    "structured_support_issues",
    "execute_structured_route",
    "execute_sweep_route",
]
