from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    expression_equal,
    linear_combinations,
    RationalPolynomial,
)
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked
from ._common_root_volume_overlap import (
    apply_physical_overlap_extensive,
    apply_physical_overlap_intensive,
    CommonRootOverlapError,
    CommonRootVolumeWitness,
    CurvedOverlapIntegrationEvidence,
    CurvedOverlapIntegrationPolicy,
    MaterialReferenceVolumeOverlap,
    prepare_common_root_volume_refinement,
    prepare_material_reference_volume_overlap,
    transpose_physical_overlap_extensive,
    transpose_physical_overlap_intensive,
)
from ._domain_certification import DomainCoverageCertificate
from ._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
    PreparedCommonRefinement,
)
from ._volume_material_deformation import (
    _require_coverage,
    declare_material_reference,
    MaterialReferenceDeclaration,
    MaterialReferencePolicy,
    VolumeMaterialDeformation,
)


def require_polyhedral_carrier_geometry(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> None:
    """Require coordinate maps to realize the exact straight cell carriers."""
    elements, routes_by_block, _ = geometry.resolve(mesh)
    mesh_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    coordinate_bank = np.asarray(geometry.coordinates, dtype=np.float64)
    if geometry.source_map is not None:
        exact_coordinates = geometry.source_coordinates()
        rounded_exact = np.asarray(
            [[float(value) for value in row] for row in exact_coordinates],
            dtype=np.float64,
        )
        if rounded_exact.shape != coordinate_bank.shape or not np.array_equal(
            rounded_exact, coordinate_bank
        ):
            raise VolumeMaterialMapError(
                "Exact source-coordinate leaves must round to the executed geometry coefficient bank."
            )
    for block, element, routes in zip(
        mesh.blocks, elements, routes_by_block, strict=True
    ):
        connectivity = np.asarray(block.vertices, dtype=np.int64)
        route_rows = np.asarray(routes, dtype=np.int64)
        if isinstance(element, CellVertexGeometryElement):
            if block.cell_kind not in ("polygon", "polyhedron"):
                raise VolumeMaterialMapError(
                    "Variable-topology overlap requires a vertex-defined physical carrier."
                )
            mask = np.asarray(block.vertex_valid, dtype=np.bool_)
            actual = coordinate_bank[route_rows]
            expected = mesh_coordinates[connectivity]
            if not np.array_equal(actual[mask], expected[mask]):
                raise VolumeMaterialMapError(
                    "Vertex-defined coordinate maps must match their physical cell carriers."
                )
            continue
        if not isinstance(element, ScalarCoordinateElement):
            raise VolumeMaterialMapError(
                "Physical overlap requires an owning scalar coordinate map."
            )

        try:
            topology = reference_cell_topology(block.cell_kind)
        except (TypeError, ValueError) as error:
            raise VolumeMaterialMapError(
                "Material overlap requires explicit physical cell geometry."
            ) from error
        reference_corners = np.asarray(topology.vertices, dtype=np.float64)
        if connectivity.shape[1] != reference_corners.shape[0] or route_rows.shape[
            1
        ] != getattr(element, "local_dof_count", -1):
            raise VolumeMaterialMapError(
                "Coordinate routes do not realize the declared cell-corner carriers."
            )
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        mapped_corners = geometry.evaluate(
            mesh,
            identifiers,
            np.broadcast_to(
                reference_corners, (block.cell_count, *reference_corners.shape)
            ),
        )
        expected_corners = mesh_coordinates[connectivity]
        scale = max(1.0, float(np.max(np.abs(expected_corners))))
        if (
            not np.all(np.isfinite(mapped_corners))
            or float(np.max(np.abs(mapped_corners - expected_corners)))
            > 32 * np.finfo(np.float64).eps * scale
        ):
            raise VolumeMaterialMapError(
                "Coordinate maps must reproduce the physical cell carrier corners."
            )

        for cell in range(block.cell_count):
            basis = coordinate_effective_basis(element, cell_row=cell)
            route = route_rows[cell]
            coefficients = geometry.source_coordinates(route)
            if len(basis) != len(coefficients):
                raise VolumeMaterialMapError(
                    "Exact coordinate maps do not align with their physical carrier DOFs."
                )
            coordinate_expressions = linear_combinations(
                basis,
                tuple(
                    tuple(coefficients[dof][axis] for dof in range(len(basis)))
                    for axis in range(mesh.ambient_dimension)
                ),
            )
            for expression in coordinate_expressions:
                terms = (
                    expression.numerator
                    if isinstance(expression, RationalPolynomial)
                    else expression
                )
                affine = {
                    exponent: coefficient
                    for exponent, coefficient in terms.items()
                    if sum(exponent) <= 1
                }
                if not expression_equal(expression, affine):
                    raise VolumeMaterialMapError(
                        "Physical overlap currently requires exact affine cell maps."
                    )


class VolumeMaterialMapError(ValueError):
    """A material map cannot bind its endpoints or cover their physical cells."""

    def __init__(
        self,
        message: str,
        /,
        *,
        status: CommonRefinementStatus | str | None = None,
        refinement: PreparedCommonRefinement | None = None,
        reference_source_coverage_defects: Sequence[str] = (),
        reference_target_coverage_defects: Sequence[str] = (),
        unmatched_source_rows: Sequence[int] = (),
        unmatched_target_rows: Sequence[int] = (),
        overlapping_source_rows: Sequence[int] = (),
        overlapping_target_rows: Sequence[int] = (),
        integration_error: float | None = None,
        integration_tolerance: float | None = None,
        unresolved_reference_tetrahedron: Sequence[Sequence[str]] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.refinement = refinement
        self.reference_source_coverage_defects = tuple(reference_source_coverage_defects)
        self.reference_target_coverage_defects = tuple(reference_target_coverage_defects)
        self.overlapping_source_rows = jnp.asarray(
            overlapping_source_rows, dtype=jnp.int64
        )
        self.overlapping_target_rows = jnp.asarray(
            overlapping_target_rows, dtype=jnp.int64
        )
        self.integration_error = integration_error
        self.integration_tolerance = integration_tolerance
        self.unresolved_reference_tetrahedron = tuple(
            tuple(str(value) for value in point)
            for point in unresolved_reference_tetrahedron
        )
        if refinement is None:
            self.source_coverage_defects = None
            self.target_coverage_defects = None
            self.unmatched_source_rows = jnp.asarray(
                unmatched_source_rows, dtype=jnp.int64
            )
            self.unmatched_target_rows = jnp.asarray(
                unmatched_target_rows, dtype=jnp.int64
            )
        else:
            self.source_coverage_defects = refinement.evidence.source_coverage_defects
            self.target_coverage_defects = refinement.evidence.target_coverage_defects
            self.unmatched_source_rows = jnp.flatnonzero(
                jnp.bincount(
                    refinement.source_cells,
                    weights=refinement.volumes,
                    length=refinement.source_cell_count,
                )
                == 0.0
            )
            self.unmatched_target_rows = jnp.flatnonzero(
                jnp.bincount(
                    refinement.target_cells,
                    weights=refinement.volumes,
                    length=refinement.target_cell_count,
                )
                == 0.0
            )


class MaterialMapActionEvidence(NamedTuple):
    """Observed componentwise conservation defect of one extensive action."""

    values: Array
    before_total: Array
    after_total: Array
    absolute_defect: Array
    source_coverage_defects: tuple[Array, ...]
    target_coverage_defects: tuple[Array, ...]


class MaterialSourceMapCapability(StrictModule, NonTrainableState):
    """Stable source/material identity shared by geometric cell realizations."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        domain_id: str,
        cell_entity_ids: Sequence[str],
        region_ids: Sequence[str],
        /,
        *,
        occurrence_paths: Sequence[tuple[str, ...]] | None = None,
    ) -> None:
        identity = tuple(
            str(value).strip() for value in (source_id, source_revision, domain_id)
        )
        if any(not value for value in identity):
            raise ValueError("Material source identity/revision/domain must be nonempty.")
        entities = tuple(str(value).strip() for value in cell_entity_ids)
        regions = tuple(str(value).strip() for value in region_ids)
        paths = (
            tuple(() for _ in entities)
            if occurrence_paths is None
            else tuple(tuple(path) for path in occurrence_paths)
        )
        if (
            not entities
            or len(regions) != len(entities)
            or len(paths) != len(entities)
            or any(not value for value in entities)
            or any(not value for value in regions)
            or any(
                any(
                    not isinstance(part, str) or not part or part != part.strip()
                    for part in path
                )
                for path in paths
            )
        ):
            raise ValueError(
                "Material source cell entities, regions, and paths must align."
            )
        self.source_id, self.source_revision, self.domain_id = identity
        self.cell_entity_ids = entities
        self.region_ids = regions
        self.occurrence_paths = paths
        self.map_id = canonical_fingerprint(
            {
                "kind": "material-source-map-capability",
                "source": identity[0],
                "revision": identity[1],
                "domain": identity[2],
                "cell_entities": entities,
                "regions": regions,
                "occurrence_paths": paths,
            }
        )

    @classmethod
    def revalidate(
        cls, value: MaterialSourceMapCapability, /
    ) -> MaterialSourceMapCapability:
        if not isinstance(value, MaterialSourceMapCapability):
            raise TypeError("value must be MaterialSourceMapCapability.")
        rebuilt = cls(
            value.source_id,
            value.source_revision,
            value.domain_id,
            value.cell_entity_ids,
            value.region_ids,
            occurrence_paths=value.occurrence_paths,
        )
        if rebuilt.map_id != value.map_id:
            raise ValueError(
                "Stored material source capability identity does not revalidate."
            )
        return rebuilt


def _material_occurrence_paths(
    source_map: MaterialSourceMapCapability,
    entities: tuple[str, ...],
    regions: tuple[str, ...],
    supplied: Sequence[Sequence[str]] | None,
    role: str,
    /,
) -> tuple[tuple[str, ...], ...]:
    if supplied is None:
        available: dict[tuple[str, str], set[tuple[str, ...]]] = {}
        for entity, region, path in zip(
            source_map.cell_entity_ids,
            source_map.region_ids,
            source_map.occurrence_paths,
            strict=True,
        ):
            available.setdefault((entity, region), set()).add(path)
        resolved = []
        for entity, region in zip(entities, regions, strict=True):
            paths = available.get((entity, region), set())
            if len(paths) != 1:
                raise VolumeMaterialMapError(
                    f"{role} material occurrence side must be supplied when its source entity is ambiguous."
                )
            resolved.append(next(iter(paths)))
        paths = tuple(resolved)
    else:
        paths = tuple(tuple(path) for path in supplied)
    if len(paths) != len(entities) or any(
        any(
            not isinstance(part, str) or not part or part != part.strip() for part in path
        )
        for path in paths
    ):
        raise VolumeMaterialMapError(
            f"{role} occurrence paths must align with endpoint material rows."
        )
    known = set(
        zip(
            source_map.cell_entity_ids,
            source_map.region_ids,
            source_map.occurrence_paths,
            strict=True,
        )
    )
    if any(
        (entity, region, path) not in known
        for entity, region, path in zip(entities, regions, paths, strict=True)
    ):
        raise VolumeMaterialMapError(
            f"{role} material occurrence/region sides do not match the retained source map."
        )
    return paths


def _require_endpoint_source_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source_map: MaterialSourceMapCapability,
    region_ids: tuple[str, ...],
    coverage: DomainCoverageCertificate,
    role: str,
    /,
    *,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
) -> DomainCoverageCertificate:
    if not isinstance(coverage, DomainCoverageCertificate):
        raise VolumeMaterialMapError(
            f"{role} endpoint requires a DomainCoverageCertificate."
        )
    if coverage.status != "certified":
        raise VolumeMaterialMapError(
            f"{role} source-domain coverage is not certified ({coverage.status!r})."
        )
    if (
        coverage.source_id != source_map.source_id
        or coverage.source_revision != source_map.source_revision
        or coverage.domain_id != source_map.domain_id
    ):
        raise VolumeMaterialMapError(
            f"{role} source coverage does not match the retained source/revision/domain map."
        )
    region_rows_by_name = {name: row for row, name in enumerate(coverage.region_ids)}
    try:
        region_rows = np.asarray(
            [region_rows_by_name[name] for name in region_ids], dtype=np.int64
        )
    except KeyError as error:
        raise VolumeMaterialMapError(
            f"{role} material regions are absent from certified source coverage."
        ) from error
    if region_rows.shape != coverage.cell_regions.shape or not np.array_equal(
        region_rows, np.asarray(coverage.cell_regions, dtype=np.int64)
    ):
        raise VolumeMaterialMapError(
            f"{role} material region rows differ from certified dense source coverage."
        )
    try:
        coverage.require_current(
            mesh,
            geometry,
            coverage.domain,
            region_rows,
            embedding=coverage.embedding,
            source=coverage.source,
            fidelity=coverage.fidelity,
            source_coverage_witness=coverage.source_coverage_witness,
            chart_parent=chart_parent,
        )
    except (TypeError, ValueError) as error:
        raise VolumeMaterialMapError(
            f"{role} source coverage is stale or does not revalidate against endpoint geometry."
        ) from error
    return coverage


class EulerianVolumeOverlapMap(StrictModule, NonTrainableState):
    """Eulerian physical overlap between exact mesh/coordinate epochs."""

    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_map: MaterialSourceMapCapability
    source_coverage: DomainCoverageCertificate
    target_coverage: DomainCoverageCertificate
    source_cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    target_cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_region_ids: tuple[str, ...] = eqx.field(static=True)
    target_region_ids: tuple[str, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    target_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    refinement: PreparedCommonRefinement
    common_root: CommonRootVolumeWitness | None
    integration_evidence: CurvedOverlapIntegrationEvidence | None
    source_overlap_measures: Array
    target_overlap_measures: Array
    source_coverage_defect: Array
    target_coverage_defect: Array
    map_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        source_map: MaterialSourceMapCapability,
        source_cell_entity_ids: Sequence[str],
        target_cell_entity_ids: Sequence[str],
        source_region_ids: Sequence[str],
        target_region_ids: Sequence[str],
        /,
        *,
        source_coverage: DomainCoverageCertificate,
        target_coverage: DomainCoverageCertificate,
        policy: CommonRefinementPolicy | None = None,
        source_occurrence_paths: Sequence[Sequence[str]] | None = None,
        target_occurrence_paths: Sequence[Sequence[str]] | None = None,
        common_root: CommonRootVolumeWitness | None = None,
        integration_policy: CurvedOverlapIntegrationPolicy | None = None,
        source_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
        target_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> None:
        if not all(
            isinstance(value, CellMesh) for value in (source_mesh, target_mesh)
        ) or not all(
            isinstance(value, CellGeometrySpec)
            for value in (source_geometry, target_geometry)
        ):
            raise TypeError(
                "Material-map endpoints require canonical mesh and geometry owners."
            )
        policy_ = (
            CommonRefinementPolicy(coverage=CommonRefinementCoverage.COMPLETE)
            if policy is None
            else policy
        )
        if common_root is None:
            if integration_policy is not None:
                raise ValueError(
                    "integration_policy requires an explicit curved common-root witness."
                )
            require_polyhedral_carrier_geometry(source_mesh, source_geometry)
            require_polyhedral_carrier_geometry(target_mesh, target_geometry)
            refinement = prepare_common_refinement(
                source_mesh, target_mesh, policy=policy_
            )
            integration_evidence = None
        else:
            if not isinstance(common_root, CommonRootVolumeWitness):
                raise TypeError("common_root must be CommonRootVolumeWitness or None.")
            common_root = common_root.revalidate(
                common_root.root_mesh,
                common_root.root_geometry,
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
            )
            integration_policy_ = (
                CurvedOverlapIntegrationPolicy()
                if integration_policy is None
                else integration_policy
            )
            try:
                refinement, integration_evidence = prepare_common_root_volume_refinement(
                    source_mesh,
                    source_geometry,
                    target_mesh,
                    target_geometry,
                    common_root,
                    policy=policy_,
                    integration_policy=integration_policy_,
                )
            except CommonRootOverlapError as error:
                raise VolumeMaterialMapError(
                    str(error),
                    status=error.status,
                    reference_source_coverage_defects=tuple(
                        str(value) for value in error.source_reference_defects
                    ),
                    reference_target_coverage_defects=tuple(
                        str(value) for value in error.target_reference_defects
                    ),
                    overlapping_source_rows=error.overlapping_source_rows,
                    overlapping_target_rows=error.overlapping_target_rows,
                    integration_error=error.integration_error,
                    integration_tolerance=error.integration_tolerance,
                    unresolved_reference_tetrahedron=error.unresolved_reference_tetrahedron,
                    unmatched_source_rows=error.unmatched_source_rows,
                    unmatched_target_rows=error.unmatched_target_rows,
                ) from error
        source_map = MaterialSourceMapCapability.revalidate(source_map)
        if not isinstance(refinement, PreparedCommonRefinement):
            raise TypeError("refinement must be PreparedCommonRefinement.")
        if (
            refinement.status is not CommonRefinementStatus.SUCCESS
            or refinement.policy.coverage is not CommonRefinementCoverage.COMPLETE
            or any(
                (
                    refinement.evidence.source_gap_count,
                    refinement.evidence.target_gap_count,
                    refinement.evidence.source_double_count,
                    refinement.evidence.target_double_count,
                    refinement.evidence.uncertain_predicate_count,
                    refinement.evidence.invalid_cell_count,
                    refinement.evidence.intersection_failure_count,
                )
            )
        ):
            raise VolumeMaterialMapError(
                "Physical material overlap must cover both cell domains completely.",
                status=refinement.status,
                refinement=refinement,
            )
        if (
            refinement.source_mesh_id != source_mesh.mesh_id
            or refinement.target_mesh_id != target_mesh.mesh_id
            or refinement.source_topology_id != source_mesh.topology_id
            or refinement.target_topology_id != target_mesh.topology_id
        ):
            raise VolumeMaterialMapError(
                "Physical overlap is stale for the requested endpoint meshes."
            )
        source_entities = tuple(str(value).strip() for value in source_cell_entity_ids)
        target_entities = tuple(str(value).strip() for value in target_cell_entity_ids)
        source_regions = tuple(str(value).strip() for value in source_region_ids)
        target_regions = tuple(str(value).strip() for value in target_region_ids)
        source_count = sum(block.cell_count for block in source_mesh.blocks)
        target_count = sum(block.cell_count for block in target_mesh.blocks)
        if (
            len(source_entities) != source_count
            or len(source_regions) != source_count
            or len(target_entities) != target_count
            or len(target_regions) != target_count
            or any(
                not value
                for value in (
                    *source_entities,
                    *target_entities,
                    *source_regions,
                    *target_regions,
                )
            )
        ):
            raise VolumeMaterialMapError(
                "Material cell identities/regions must align with endpoint cells."
            )
        source_paths = _material_occurrence_paths(
            source_map, source_entities, source_regions, source_occurrence_paths, "source"
        )
        target_paths = _material_occurrence_paths(
            source_map, target_entities, target_regions, target_occurrence_paths, "target"
        )
        chart_parent = (
            None
            if common_root is None
            else (common_root.root_mesh, common_root.root_geometry)
        )
        source_coverage = _require_endpoint_source_coverage(
            source_mesh,
            source_geometry,
            source_map,
            source_regions,
            source_coverage,
            "source",
            chart_parent=chart_parent
            if source_chart_parent is None
            else source_chart_parent,
        )
        target_coverage = _require_endpoint_source_coverage(
            target_mesh,
            target_geometry,
            source_map,
            target_regions,
            target_coverage,
            "target",
            chart_parent=chart_parent
            if target_chart_parent is None
            else target_chart_parent,
        )
        overlap_source = np.asarray(refinement.source_cells, dtype=np.int64)
        overlap_target = np.asarray(refinement.target_cells, dtype=np.int64)
        if any(
            source_regions[int(left)] != target_regions[int(right)]
            or source_paths[int(left)] != target_paths[int(right)]
            for left, right in zip(overlap_source, overlap_target, strict=True)
        ):
            raise VolumeMaterialMapError(
                "Physical overlap crosses unmatched material region or occurrence sides."
            )
        overlap_volumes = np.asarray(refinement.volumes, dtype=np.float64)
        source_overlap_measures = np.bincount(
            overlap_source, weights=overlap_volumes, minlength=source_count
        )
        target_overlap_measures = np.bincount(
            overlap_target, weights=overlap_volumes, minlength=target_count
        )
        source_measures = np.asarray(refinement.source_measures, dtype=np.float64)
        target_measures = np.asarray(refinement.target_measures, dtype=np.float64)
        source_coverage_defect = source_overlap_measures - source_measures
        target_coverage_defect = target_overlap_measures - target_measures
        source_evidence_defect = np.asarray(
            refinement.evidence.source_coverage_defects, dtype=np.float64
        )
        target_evidence_defect = np.asarray(
            refinement.evidence.target_coverage_defects, dtype=np.float64
        )
        source_tolerance = np.asarray(
            refinement.evidence.source_coverage_tolerances, dtype=np.float64
        )
        target_tolerance = np.asarray(
            refinement.evidence.target_coverage_tolerances, dtype=np.float64
        )
        if (
            source_evidence_defect.shape != (source_count,)
            or target_evidence_defect.shape != (target_count,)
            or source_tolerance.shape != (source_count,)
            or target_tolerance.shape != (target_count,)
            or not np.array_equal(source_coverage_defect, source_evidence_defect)
            or not np.array_equal(target_coverage_defect, target_evidence_defect)
            or np.any(np.abs(source_coverage_defect) > source_tolerance)
            or np.any(np.abs(target_coverage_defect) > target_tolerance)
        ):
            raise VolumeMaterialMapError(
                "Physical material overlap lacks source/target coverage proof within its measure error."
            )
        if (
            np.any(source_overlap_measures <= 0.0)
            or np.any(target_overlap_measures <= 0.0)
            or source_measures.shape != (source_count,)
            or target_measures.shape != (target_count,)
        ):
            raise VolumeMaterialMapError(
                "Physical overlap must measure every material endpoint cell."
            )
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.source_geometry = source_geometry
        self.target_geometry = target_geometry
        self.source_map = source_map
        self.source_coverage = source_coverage
        self.target_coverage = target_coverage
        self.source_cell_entity_ids = source_entities
        self.target_cell_entity_ids = target_entities
        self.source_region_ids = source_regions
        self.target_region_ids = target_regions
        self.source_occurrence_paths = source_paths
        self.target_occurrence_paths = target_paths
        self.common_root = common_root
        self.integration_evidence = integration_evidence
        self.source_overlap_measures = jnp.asarray(source_overlap_measures)
        self.target_overlap_measures = jnp.asarray(target_overlap_measures)
        self.source_coverage_defect = jnp.asarray(
            source_overlap_measures - source_measures
        )
        self.target_coverage_defect = jnp.asarray(
            target_overlap_measures - target_measures
        )
        self.refinement = refinement
        self.map_id = canonical_fingerprint(
            {
                "kind": "eulerian-volume-overlap-map",
                "source_map": source_map.map_id,
                "source_coverage": source_coverage.certificate_id,
                "target_coverage": target_coverage.certificate_id,
                "source_mesh": source_mesh.mesh_id,
                "target_mesh": target_mesh.mesh_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "target_geometry": cell_geometry_id(target_geometry),
                "source_entities": source_entities,
                "target_entities": target_entities,
                "source_regions": source_regions,
                "target_regions": target_regions,
                "source_occurrences": source_paths,
                "target_occurrences": target_paths,
                "overlap": refinement.refinement_id,
                "common_root": None if common_root is None else common_root.witness_id,
                "integration_evidence": (
                    None
                    if integration_evidence is None
                    else integration_evidence.evidence_id
                ),
                "source_coverage_defect": array_tree_fingerprint(
                    source_overlap_measures - source_measures
                ),
                "target_coverage_defect": array_tree_fingerprint(
                    target_overlap_measures - target_measures
                ),
            }
        )

    @classmethod
    def revalidate(
        cls,
        value: EulerianVolumeOverlapMap,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        root_mesh: CellMesh | None = None,
        root_geometry: CellGeometrySpec | None = None,
        source_coverage: DomainCoverageCertificate | None = None,
        target_coverage: DomainCoverageCertificate | None = None,
        source_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
        target_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> EulerianVolumeOverlapMap:
        """Rebuild Eulerian overlap and coverage from restored endpoint owners."""
        if not isinstance(value, EulerianVolumeOverlapMap):
            raise TypeError("value must be EulerianVolumeOverlapMap.")
        source_coverage_ = (
            value.source_coverage if source_coverage is None else source_coverage
        )
        target_coverage_ = (
            value.target_coverage if target_coverage is None else target_coverage
        )
        common_root = None
        if value.common_root is None:
            if root_mesh is not None or root_geometry is not None:
                raise ValueError(
                    "Affine material-map restore does not accept a common root."
                )
        else:
            if not isinstance(root_mesh, CellMesh) or not isinstance(
                root_geometry, CellGeometrySpec
            ):
                raise ValueError(
                    "Curved material-map restore requires independently restored root owners."
                )
            common_root = value.common_root.revalidate(
                root_mesh,
                root_geometry,
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
            )
        policy = CommonRefinementPolicy(
            predicate_mode=value.refinement.policy.predicate_mode,
            coverage=value.refinement.policy.coverage,
            coverage_tolerance=value.refinement.policy.coverage_tolerance,
            second_moments=value.refinement.policy.second_moments,
            overlap_simplices=value.refinement.policy.overlap_simplices,
        )
        integration_policy = (
            None
            if value.integration_evidence is None
            else CurvedOverlapIntegrationPolicy(
                relative_tolerance=value.integration_evidence.integration_policy.relative_tolerance,
                absolute_tolerance=value.integration_evidence.integration_policy.absolute_tolerance,
            )
        )
        rebuilt = prepare_eulerian_volume_overlap(
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            MaterialSourceMapCapability.revalidate(value.source_map),
            value.source_cell_entity_ids,
            value.target_cell_entity_ids,
            value.source_region_ids,
            value.target_region_ids,
            source_coverage=source_coverage_,
            target_coverage=target_coverage_,
            policy=policy,
            source_occurrence_paths=value.source_occurrence_paths,
            target_occurrence_paths=value.target_occurrence_paths,
            common_root=common_root,
            integration_policy=integration_policy,
            source_chart_parent=source_chart_parent,
            target_chart_parent=target_chart_parent,
        )
        if rebuilt.map_id != value.map_id:
            raise ValueError(
                "Stored material-map state does not match recomputed overlap evidence."
            )
        return rebuilt

    def apply_extensive(self, content: Any, /) -> Array:
        return self.apply_extensive_with_evidence(content).values

    def apply_extensive_with_evidence(self, content: Any, /) -> MaterialMapActionEvidence:
        result = apply_physical_overlap_extensive(self.refinement, content)
        source_values = jnp.asarray(content).astype(result.dtype)
        before = jnp.sum(source_values, axis=0)
        after = jnp.sum(result, axis=0)
        return MaterialMapActionEvidence(
            result,
            before,
            after,
            jnp.abs(after - before),
            (self.source_coverage_defect,),
            (self.target_coverage_defect,),
        )

    def apply_extensive_transpose(self, cotangent: Any, /) -> Array:
        """Euclidean transpose of the extensive content action."""
        return transpose_physical_overlap_extensive(self.refinement, cotangent)

    def apply_intensive(self, average: Any, /) -> Array:
        return apply_physical_overlap_intensive(self.refinement, average)

    def apply_intensive_transpose(self, cotangent: Any, /) -> Array:
        """Euclidean transpose of the intensive field action."""
        return transpose_physical_overlap_intensive(self.refinement, cotangent)


def _tree_content_equal(first: Any, second: Any, /) -> bool:
    equal = eqx.tree_equal(first, second, typematch=True)
    return equal is True or bool(np.asarray(equal))


class EulerianVolumeOverlapChain(StrictModule, NonTrainableState):
    """Compact chronological composition of validated Eulerian volume overlaps."""

    maps: tuple[EulerianVolumeOverlapMap, ...]
    source_map: MaterialSourceMapCapability
    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    chain_id: str = eqx.field(static=True)

    def __init__(self, maps: Sequence[EulerianVolumeOverlapMap], /) -> None:
        records = tuple(maps)
        if not records or not all(
            isinstance(value, EulerianVolumeOverlapMap) for value in records
        ):
            raise TypeError(
                "Eulerian overlap chain requires one or more EulerianVolumeOverlapMap owners."
            )
        for previous, following in zip(records, records[1:]):
            if (
                previous.target_mesh.mesh_id != following.source_mesh.mesh_id
                or not _tree_content_equal(previous.target_mesh, following.source_mesh)
                or not _tree_content_equal(
                    previous.target_geometry, following.source_geometry
                )
                or not _tree_content_equal(previous.source_map, following.source_map)
            ):
                raise VolumeMaterialMapError(
                    "Eulerian overlap maps must join identical intermediate mesh, coordinate, and source-map states."
                )
        self.maps = records
        self.source_map = records[0].source_map
        self.source_mesh_id = records[0].source_mesh.mesh_id
        self.target_mesh_id = records[-1].target_mesh.mesh_id
        self.source_geometry_id = cell_geometry_id(records[0].source_geometry)
        self.target_geometry_id = cell_geometry_id(records[-1].target_geometry)
        self.chain_id = canonical_fingerprint(
            {
                "kind": "eulerian-volume-overlap-chain",
                "source_geometry": self.source_geometry_id,
                "target_geometry": self.target_geometry_id,
                "maps": tuple(value.map_id for value in records),
            }
        )

    @classmethod
    def revalidate(
        cls,
        value: EulerianVolumeOverlapChain,
        maps: Sequence[EulerianVolumeOverlapMap],
        /,
    ) -> EulerianVolumeOverlapChain:
        if not isinstance(value, EulerianVolumeOverlapChain):
            raise TypeError("value must be EulerianVolumeOverlapChain.")
        rebuilt = cls(maps)
        if rebuilt.chain_id != value.chain_id:
            raise ValueError(
                "Stored Eulerian overlap chain does not match recomputed maps."
            )
        return rebuilt

    def apply_extensive(self, content: Any, /) -> Array:
        return self.apply_extensive_with_evidence(content).values

    def apply_extensive_with_evidence(self, content: Any, /) -> MaterialMapActionEvidence:
        initial = jnp.asarray(content)
        result = initial
        source_defects = []
        target_defects = []
        for material_map in self.maps:
            action = material_map.apply_extensive_with_evidence(result)
            result = action.values
            source_defects.extend(action.source_coverage_defects)
            target_defects.extend(action.target_coverage_defects)
        before = jnp.sum(initial, axis=0)
        after = jnp.sum(result, axis=0)
        return MaterialMapActionEvidence(
            result,
            before,
            after,
            jnp.abs(after - before),
            tuple(source_defects),
            tuple(target_defects),
        )

    def apply_extensive_transpose(self, cotangent: Any, /) -> Array:
        result = jnp.asarray(cotangent)
        for material_map in reversed(self.maps):
            result = material_map.apply_extensive_transpose(result)
        return result

    def apply_intensive(self, average: Any, /) -> Array:
        result = jnp.asarray(average)
        for material_map in self.maps:
            result = material_map.apply_intensive(result)
        return result

    def apply_intensive_transpose(self, cotangent: Any, /) -> Array:
        result = jnp.asarray(cotangent)
        for material_map in reversed(self.maps):
            result = material_map.apply_intensive_transpose(result)
        return result


def prepare_eulerian_volume_overlap(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_map: MaterialSourceMapCapability,
    source_cell_entity_ids: Sequence[str],
    target_cell_entity_ids: Sequence[str],
    source_region_ids: Sequence[str],
    target_region_ids: Sequence[str],
    /,
    *,
    source_coverage: DomainCoverageCertificate,
    target_coverage: DomainCoverageCertificate,
    policy: CommonRefinementPolicy | None = None,
    source_occurrence_paths: Sequence[Sequence[str]] | None = None,
    target_occurrence_paths: Sequence[Sequence[str]] | None = None,
    common_root: CommonRootVolumeWitness | None = None,
    integration_policy: CurvedOverlapIntegrationPolicy | None = None,
    source_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    target_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
) -> EulerianVolumeOverlapMap:
    return EulerianVolumeOverlapMap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        source_map,
        source_cell_entity_ids,
        target_cell_entity_ids,
        source_region_ids,
        target_region_ids,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        policy=policy,
        source_occurrence_paths=source_occurrence_paths,
        target_occurrence_paths=target_occurrence_paths,
        common_root=common_root,
        integration_policy=integration_policy,
        source_chart_parent=source_chart_parent,
        target_chart_parent=target_chart_parent,
    )


def _rows_from_target_offsets(
    offsets: np.ndarray, source_rows: np.ndarray, /
) -> np.ndarray:
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != source_rows.size
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise VolumeMaterialMapError(
            "Material-reference overlap has malformed target-major CSR offsets."
        )
    return np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))


def _sparse_material_apply(
    values: Any,
    source_rows: Array,
    target_rows: Array,
    weights: Array,
    source_count: int,
    target_count: int,
    /,
) -> Array:
    source = jnp.asarray(values)
    if source.ndim == 0 or source.shape[0] != source_count:
        raise ValueError("Material action input must have one row per source cell.")
    scale = weights.reshape((weights.shape[0],) + (1,) * (source.ndim - 1))
    contributions = source[source_rows] * scale
    result = jnp.zeros((target_count, *source.shape[1:]), dtype=contributions.dtype)
    return result.at[target_rows].add(contributions)


def _sparse_material_transpose(
    values: Any,
    source_rows: Array,
    target_rows: Array,
    weights: Array,
    source_count: int,
    target_count: int,
    /,
) -> Array:
    target = jnp.asarray(values)
    if target.ndim == 0 or target.shape[0] != target_count:
        raise ValueError("Material transpose input must have one row per target cell.")
    scale = weights.reshape((weights.shape[0],) + (1,) * (target.ndim - 1))
    contributions = target[target_rows] * scale
    result = jnp.zeros((source_count, *target.shape[1:]), dtype=contributions.dtype)
    return result.at[source_rows].add(contributions)


class VolumeMaterialMap(StrictModule, NonTrainableState):
    """Lagrangian volume transfer over one explicitly declared material reference."""

    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    deformation: VolumeMaterialDeformation
    reference: MaterialReferenceDeclaration
    source_map: MaterialSourceMapCapability
    source_coverage: DomainCoverageCertificate
    target_coverage: DomainCoverageCertificate
    source_cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    target_cell_entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_region_ids: tuple[str, ...] = eqx.field(static=True)
    target_region_ids: tuple[str, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    target_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    policy: CommonRefinementPolicy
    integration_policy: CurvedOverlapIntegrationPolicy
    material_overlap: MaterialReferenceVolumeOverlap
    target_offsets: Array
    source_rows: Array
    target_rows: Array
    reference_overlap_measures: Array
    source_physical_overlap_measures: Array
    target_physical_overlap_measures: Array
    source_reference_measures: Array
    target_reference_measures: Array
    source_physical_measures: Array
    target_physical_measures: Array
    source_reference_coverage_defect: Array
    target_reference_coverage_defect: Array
    source_physical_coverage_defect: Array
    target_physical_coverage_defect: Array
    extensive_weights: Array
    density_weights: Array
    scalar_weights: Array
    map_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        deformation: VolumeMaterialDeformation,
        source_cell_entity_ids: Sequence[str],
        target_cell_entity_ids: Sequence[str],
        source_region_ids: Sequence[str],
        target_region_ids: Sequence[str],
        /,
        *,
        source_coverage: DomainCoverageCertificate,
        target_coverage: DomainCoverageCertificate,
        policy: CommonRefinementPolicy | None = None,
        source_occurrence_paths: Sequence[Sequence[str]] | None = None,
        target_occurrence_paths: Sequence[Sequence[str]] | None = None,
        integration_policy: CurvedOverlapIntegrationPolicy | None = None,
        source_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
        target_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> None:
        if not all(
            isinstance(value, CellMesh) for value in (source_mesh, target_mesh)
        ) or not all(
            isinstance(value, CellGeometrySpec)
            for value in (source_geometry, target_geometry)
        ):
            raise TypeError(
                "Material map endpoints require canonical mesh and geometry owners."
            )
        deformation.require_bound(
            source_mesh, source_geometry, target_mesh, target_geometry
        )
        reference = deformation.reference
        source_map = reference.source_map
        policy_ = (
            CommonRefinementPolicy(coverage=CommonRefinementCoverage.COMPLETE)
            if policy is None
            else policy
        )
        integration_policy_ = (
            CurvedOverlapIntegrationPolicy()
            if integration_policy is None
            else integration_policy
        )
        if (
            not isinstance(policy_, CommonRefinementPolicy)
            or policy_.coverage is not CommonRefinementCoverage.COMPLETE
        ):
            raise ValueError(
                "Material-reference transfer requires complete reference and endpoint coverage."
            )
        source_coverage = _require_coverage(
            source_mesh,
            source_geometry,
            source_map,
            source_coverage,
            "source",
            chart_parent=(
                (reference.root_mesh, reference.root_geometry)
                if source_chart_parent is None
                else source_chart_parent
            ),
        )
        target_coverage = _require_coverage(
            target_mesh,
            target_geometry,
            source_map,
            target_coverage,
            "target",
            chart_parent=(
                (reference.root_mesh, reference.root_geometry)
                if target_chart_parent is None
                else target_chart_parent
            ),
        )
        if (
            source_coverage.certificate_id != deformation.source_coverage_id
            or target_coverage.certificate_id != deformation.target_coverage_id
        ):
            raise VolumeMaterialMapError(
                "Material deformation does not bind these endpoint coverage certificates."
            )
        source_entities = tuple(str(value).strip() for value in source_cell_entity_ids)
        target_entities = tuple(str(value).strip() for value in target_cell_entity_ids)
        source_regions = tuple(str(value).strip() for value in source_region_ids)
        target_regions = tuple(str(value).strip() for value in target_region_ids)
        source_count = sum(block.cell_count for block in source_mesh.blocks)
        target_count = sum(block.cell_count for block in target_mesh.blocks)
        if (
            len(source_entities) != source_count
            or len(source_regions) != source_count
            or len(target_entities) != target_count
            or len(target_regions) != target_count
            or any(
                not value
                for value in (
                    *source_entities,
                    *target_entities,
                    *source_regions,
                    *target_regions,
                )
            )
        ):
            raise VolumeMaterialMapError(
                "Material cell identities/regions must align with endpoint cells."
            )
        source_paths = _material_occurrence_paths(
            source_map, source_entities, source_regions, source_occurrence_paths, "source"
        )
        target_paths = _material_occurrence_paths(
            source_map, target_entities, target_regions, target_occurrence_paths, "target"
        )
        source_coverage_regions = tuple(
            source_coverage.region_ids[int(row)]
            for row in np.asarray(source_coverage.cell_regions, dtype=np.int64)
        )
        target_coverage_regions = tuple(
            target_coverage.region_ids[int(row)]
            for row in np.asarray(target_coverage.cell_regions, dtype=np.int64)
        )
        if (
            source_coverage_regions != source_regions
            or target_coverage_regions != target_regions
        ):
            raise VolumeMaterialMapError(
                "Material region rows must match their independently certified endpoint coverage."
            )
        try:
            material_overlap = prepare_material_reference_volume_overlap(
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
                deformation.reference_witness,
                policy=policy_,
                integration_policy=integration_policy_,
            )
        except CommonRootOverlapError as error:
            raise VolumeMaterialMapError(
                str(error),
                status=error.status,
                reference_source_coverage_defects=tuple(
                    str(value) for value in error.source_reference_defects
                ),
                reference_target_coverage_defects=tuple(
                    str(value) for value in error.target_reference_defects
                ),
                overlapping_source_rows=error.overlapping_source_rows,
                overlapping_target_rows=error.overlapping_target_rows,
                integration_error=error.integration_error,
                integration_tolerance=error.integration_tolerance,
                unresolved_reference_tetrahedron=error.unresolved_reference_tetrahedron,
                unmatched_source_rows=error.unmatched_source_rows,
                unmatched_target_rows=error.unmatched_target_rows,
            ) from error
        if (
            material_overlap.status is not CommonRefinementStatus.SUCCESS
            or not material_overlap.succeeded
        ):
            raise VolumeMaterialMapError(
                "Material-reference overlap is not a complete certified result.",
                status=material_overlap.status,
            )
        offsets = np.asarray(material_overlap.target_offsets, dtype=np.int64)
        source_rows = np.asarray(material_overlap.source_rows, dtype=np.int64)
        target_rows = _rows_from_target_offsets(offsets, source_rows)
        reference_overlap = np.asarray(
            material_overlap.reference_overlap_measures, dtype=np.float64
        )
        reference_error = np.asarray(
            material_overlap.reference_overlap_errors, dtype=np.float64
        )
        source_overlap = np.asarray(
            material_overlap.source_overlap_measures, dtype=np.float64
        )
        source_overlap_error = np.asarray(
            material_overlap.source_overlap_errors, dtype=np.float64
        )
        target_overlap = np.asarray(
            material_overlap.target_overlap_measures, dtype=np.float64
        )
        target_overlap_error = np.asarray(
            material_overlap.target_overlap_errors, dtype=np.float64
        )
        source_reference = np.asarray(
            material_overlap.source_reference_measures, dtype=np.float64
        )
        source_reference_error = np.asarray(
            material_overlap.source_reference_errors, dtype=np.float64
        )
        target_reference = np.asarray(
            material_overlap.target_reference_measures, dtype=np.float64
        )
        target_reference_error = np.asarray(
            material_overlap.target_reference_errors, dtype=np.float64
        )
        source_physical = np.asarray(
            material_overlap.source_physical_measures, dtype=np.float64
        )
        source_physical_error = np.asarray(
            material_overlap.source_physical_errors, dtype=np.float64
        )
        target_physical = np.asarray(
            material_overlap.target_physical_measures, dtype=np.float64
        )
        target_physical_error = np.asarray(
            material_overlap.target_physical_errors, dtype=np.float64
        )
        arrays = (
            reference_overlap,
            reference_error,
            source_overlap,
            source_overlap_error,
            target_overlap,
            target_overlap_error,
        )
        if any(
            value.shape != source_rows.shape
            or not np.all(np.isfinite(value))
            or np.any(value < 0.0)
            for value in arrays
        ) or any(
            np.any(value <= 0.0)
            for value in (reference_overlap, source_overlap, target_overlap)
        ):
            raise VolumeMaterialMapError(
                "Material-reference overlap pair measures/errors are malformed."
            )
        if any(
            value.shape != (count,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0.0)
            for value, count in (
                (source_reference, source_count),
                (source_reference_error, source_count),
                (source_physical, source_count),
                (source_physical_error, source_count),
                (target_reference, target_count),
                (target_reference_error, target_count),
                (target_physical, target_count),
                (target_physical_error, target_count),
            )
        ) or any(
            np.any(value <= 0.0)
            for value in (
                source_reference,
                source_physical,
                target_reference,
                target_physical,
            )
        ):
            raise VolumeMaterialMapError(
                "Material-reference endpoint measure/error arrays are malformed."
            )
        if (
            source_rows.size == 0
            or np.any(source_rows < 0)
            or np.any(source_rows >= source_count)
            or np.any(target_rows < 0)
            or np.any(target_rows >= target_count)
            or any(
                source_regions[int(left)] != target_regions[int(right)]
                or source_paths[int(left)] != target_paths[int(right)]
                for left, right in zip(source_rows, target_rows, strict=True)
            )
        ):
            raise VolumeMaterialMapError(
                "Material-reference overlap crosses unmatched material region or occurrence sides."
            )
        source_reference_covered = np.bincount(
            source_rows, weights=reference_overlap, minlength=source_count
        )
        target_reference_covered = np.bincount(
            target_rows, weights=reference_overlap, minlength=target_count
        )
        source_physical_covered = np.bincount(
            source_rows, weights=source_overlap, minlength=source_count
        )
        target_physical_covered = np.bincount(
            target_rows, weights=target_overlap, minlength=target_count
        )
        source_reference_defect = source_reference_covered - source_reference
        target_reference_defect = target_reference_covered - target_reference
        source_physical_defect = source_physical_covered - source_physical
        target_physical_defect = target_physical_covered - target_physical
        source_reference_error = source_reference_error + np.bincount(
            source_rows, weights=reference_error, minlength=source_count
        )
        target_reference_error = target_reference_error + np.bincount(
            target_rows, weights=reference_error, minlength=target_count
        )
        source_physical_error = source_physical_error + np.bincount(
            source_rows, weights=source_overlap_error, minlength=source_count
        )
        target_physical_error = target_physical_error + np.bincount(
            target_rows, weights=target_overlap_error, minlength=target_count
        )
        eps = np.finfo(np.float64).eps
        source_reference_tolerance = (
            policy_.coverage_tolerance * source_reference
            + source_reference_error
            + 256.0 * eps * np.maximum(1.0, source_reference)
        )
        target_reference_tolerance = (
            policy_.coverage_tolerance * target_reference
            + target_reference_error
            + 256.0 * eps * np.maximum(1.0, target_reference)
        )
        source_physical_tolerance = (
            policy_.coverage_tolerance * source_physical
            + source_physical_error
            + 256.0 * eps * np.maximum(1.0, source_physical)
        )
        target_physical_tolerance = (
            policy_.coverage_tolerance * target_physical
            + target_physical_error
            + 256.0 * eps * np.maximum(1.0, target_physical)
        )
        source_reference_evidence_defect = np.asarray(
            material_overlap.source_reference_coverage_defects, dtype=np.float64
        )
        target_reference_evidence_defect = np.asarray(
            material_overlap.target_reference_coverage_defects, dtype=np.float64
        )
        source_reference_evidence_tolerance = np.asarray(
            material_overlap.source_reference_coverage_tolerances, dtype=np.float64
        )
        target_reference_evidence_tolerance = np.asarray(
            material_overlap.target_reference_coverage_tolerances, dtype=np.float64
        )
        source_physical_evidence_defect = np.asarray(
            material_overlap.source_coverage_defects, dtype=np.float64
        )
        target_physical_evidence_defect = np.asarray(
            material_overlap.target_coverage_defects, dtype=np.float64
        )
        source_physical_evidence_tolerance = np.asarray(
            material_overlap.source_coverage_tolerances, dtype=np.float64
        )
        target_physical_evidence_tolerance = np.asarray(
            material_overlap.target_coverage_tolerances, dtype=np.float64
        )
        coverage_evidence_arrays = (
            (source_reference_evidence_defect, source_count),
            (source_reference_evidence_tolerance, source_count),
            (target_reference_evidence_defect, target_count),
            (target_reference_evidence_tolerance, target_count),
            (source_physical_evidence_defect, source_count),
            (source_physical_evidence_tolerance, source_count),
            (target_physical_evidence_defect, target_count),
            (target_physical_evidence_tolerance, target_count),
        )
        coverage_evidence_valid = all(
            value.shape == (count,)
            and np.all(np.isfinite(value))
            and (index % 2 == 0 or np.all(value >= 0.0))
            for index, (value, count) in enumerate(coverage_evidence_arrays)
        )
        exact_reference_defects = tuple(
            Fraction(str(value))
            for value in (
                *material_overlap.source_reference_defects,
                *material_overlap.target_reference_defects,
            )
        )
        if (
            not coverage_evidence_valid
            or any(value != 0 for value in exact_reference_defects)
            or np.any(
                np.abs(source_reference_evidence_defect - source_reference_defect)
                > source_reference_error + 256.0 * eps * np.maximum(1.0, source_reference)
            )
            or np.any(
                np.abs(target_reference_evidence_defect - target_reference_defect)
                > target_reference_error + 256.0 * eps * np.maximum(1.0, target_reference)
            )
            or np.any(
                np.abs(source_reference_evidence_defect)
                > source_reference_evidence_tolerance
            )
            or np.any(
                np.abs(target_reference_evidence_defect)
                > target_reference_evidence_tolerance
            )
            or np.any(
                np.abs(source_physical_evidence_defect - source_physical_defect)
                > source_physical_error + 256.0 * eps * np.maximum(1.0, source_physical)
            )
            or np.any(
                np.abs(target_physical_evidence_defect - target_physical_defect)
                > target_physical_error + 256.0 * eps * np.maximum(1.0, target_physical)
            )
            or np.any(
                np.abs(source_physical_evidence_defect)
                > source_physical_evidence_tolerance
            )
            or np.any(
                np.abs(target_physical_evidence_defect)
                > target_physical_evidence_tolerance
            )
            or np.any(np.abs(source_reference_defect) > source_reference_tolerance)
            or np.any(np.abs(target_reference_defect) > target_reference_tolerance)
            or np.any(np.abs(source_physical_defect) > source_physical_tolerance)
            or np.any(np.abs(target_physical_defect) > target_physical_tolerance)
        ):
            raise VolumeMaterialMapError(
                "Material-reference or endpoint physical coverage does not close within its certified measure errors.",
                status=material_overlap.status,
                reference_source_coverage_defects=tuple(
                    str(value) for value in source_reference_defect
                ),
                reference_target_coverage_defects=tuple(
                    str(value) for value in target_reference_defect
                ),
            )
        source_rows_array = jnp.asarray(source_rows, dtype=jnp.int64)
        target_rows_array = jnp.asarray(target_rows, dtype=jnp.int64)
        ref_overlap_array = jnp.asarray(reference_overlap)
        source_overlap_array = jnp.asarray(source_overlap)
        target_overlap_array = jnp.asarray(target_overlap)
        source_reference_array = jnp.asarray(source_reference)
        target_reference_array = jnp.asarray(target_reference)
        source_physical_array = jnp.asarray(source_physical)
        target_physical_array = jnp.asarray(target_physical)
        extensive_weights = ref_overlap_array / source_reference_array[source_rows_array]
        density_weights = source_overlap_array / target_physical_array[target_rows_array]
        scalar_weights = ref_overlap_array / target_reference_array[target_rows_array]
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.source_geometry = source_geometry
        self.target_geometry = target_geometry
        self.deformation = deformation
        self.reference = reference
        self.source_map = source_map
        self.source_coverage = source_coverage
        self.target_coverage = target_coverage
        self.source_cell_entity_ids = source_entities
        self.target_cell_entity_ids = target_entities
        self.source_region_ids = source_regions
        self.target_region_ids = target_regions
        self.source_occurrence_paths = source_paths
        self.target_occurrence_paths = target_paths
        self.policy = policy_
        self.integration_policy = integration_policy_
        self.material_overlap = material_overlap
        self.target_offsets = jnp.asarray(offsets)
        self.source_rows = source_rows_array
        self.target_rows = target_rows_array
        self.reference_overlap_measures = ref_overlap_array
        self.source_physical_overlap_measures = source_overlap_array
        self.target_physical_overlap_measures = target_overlap_array
        self.source_reference_measures = source_reference_array
        self.target_reference_measures = target_reference_array
        self.source_physical_measures = source_physical_array
        self.target_physical_measures = target_physical_array
        self.source_reference_coverage_defect = jnp.asarray(source_reference_defect)
        self.target_reference_coverage_defect = jnp.asarray(target_reference_defect)
        self.source_physical_coverage_defect = jnp.asarray(source_physical_defect)
        self.target_physical_coverage_defect = jnp.asarray(target_physical_defect)
        self.extensive_weights = extensive_weights
        self.density_weights = density_weights
        self.scalar_weights = scalar_weights
        self.map_id = canonical_fingerprint(
            {
                "kind": "lagrangian-volume-material-map",
                "reference": reference.declaration_id,
                "deformation": deformation.deformation_id,
                "source_map": source_map.map_id,
                "source_coverage": source_coverage.certificate_id,
                "target_coverage": target_coverage.certificate_id,
                "source_mesh": source_mesh.mesh_id,
                "target_mesh": target_mesh.mesh_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "target_geometry": cell_geometry_id(target_geometry),
                "source_entities": source_entities,
                "target_entities": target_entities,
                "source_regions": source_regions,
                "target_regions": target_regions,
                "source_occurrences": source_paths,
                "target_occurrences": target_paths,
                "frame": deformation.frame_id,
                "target_offsets": array_tree_fingerprint(offsets),
                "source_rows": array_tree_fingerprint(source_rows),
                "reference_overlap": array_tree_fingerprint(reference_overlap),
                "reference_overlap_error": array_tree_fingerprint(reference_error),
                "source_overlap": array_tree_fingerprint(source_overlap),
                "source_overlap_error": array_tree_fingerprint(source_overlap_error),
                "target_overlap": array_tree_fingerprint(target_overlap),
                "target_overlap_error": array_tree_fingerprint(target_overlap_error),
                "source_reference_measures": array_tree_fingerprint(source_reference),
                "target_reference_measures": array_tree_fingerprint(target_reference),
                "source_physical_measures": array_tree_fingerprint(source_physical),
                "target_physical_measures": array_tree_fingerprint(target_physical),
                "source_reference_coverage_defect": array_tree_fingerprint(
                    source_reference_defect
                ),
                "target_reference_coverage_defect": array_tree_fingerprint(
                    target_reference_defect
                ),
                "source_physical_coverage_defect": array_tree_fingerprint(
                    source_physical_defect
                ),
                "target_physical_coverage_defect": array_tree_fingerprint(
                    target_physical_defect
                ),
                "source_reference_evidence_defect": array_tree_fingerprint(
                    source_reference_evidence_defect
                ),
                "target_reference_evidence_defect": array_tree_fingerprint(
                    target_reference_evidence_defect
                ),
                "source_reference_evidence_tolerance": array_tree_fingerprint(
                    source_reference_evidence_tolerance
                ),
                "target_reference_evidence_tolerance": array_tree_fingerprint(
                    target_reference_evidence_tolerance
                ),
                "source_physical_evidence_defect": array_tree_fingerprint(
                    source_physical_evidence_defect
                ),
                "target_physical_evidence_defect": array_tree_fingerprint(
                    target_physical_evidence_defect
                ),
                "source_physical_evidence_tolerance": array_tree_fingerprint(
                    source_physical_evidence_tolerance
                ),
                "target_physical_evidence_tolerance": array_tree_fingerprint(
                    target_physical_evidence_tolerance
                ),
                "policy": policy_.policy_id,
                "integration_policy": integration_policy_.policy_id,
            }
        )

    @classmethod
    def revalidate(
        cls,
        value: VolumeMaterialMap,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        root_coverage: DomainCoverageCertificate,
        source_coverage: DomainCoverageCertificate | None = None,
        target_coverage: DomainCoverageCertificate | None = None,
        coordinate_transition: Any = None,
        root_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
        source_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
        target_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> VolumeMaterialMap:
        if not isinstance(value, VolumeMaterialMap):
            raise TypeError("value must be VolumeMaterialMap.")
        source_map = MaterialSourceMapCapability.revalidate(value.source_map)
        reference = MaterialReferenceDeclaration.revalidate(
            value.reference,
            root_mesh,
            root_geometry,
            source_map,
            root_coverage,
            chart_parent=root_chart_parent,
        )
        deformation = VolumeMaterialDeformation.revalidate(
            value.deformation,
            reference,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            source_coverage=value.source_coverage
            if source_coverage is None
            else source_coverage,
            target_coverage=value.target_coverage
            if target_coverage is None
            else target_coverage,
            coordinate_transition=coordinate_transition,
            source_chart_parent=source_chart_parent,
            target_chart_parent=target_chart_parent,
        )
        rebuilt = prepare_volume_material_map(
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            deformation,
            value.source_cell_entity_ids,
            value.target_cell_entity_ids,
            value.source_region_ids,
            value.target_region_ids,
            source_coverage=deformation.source_coverage,
            target_coverage=deformation.target_coverage,
            policy=value.policy,
            source_occurrence_paths=value.source_occurrence_paths,
            target_occurrence_paths=value.target_occurrence_paths,
            integration_policy=value.integration_policy,
            source_chart_parent=source_chart_parent,
            target_chart_parent=target_chart_parent,
        )
        if rebuilt.map_id != value.map_id:
            raise ValueError(
                "Stored material map does not match recomputed reference overlap and Jacobian evidence."
            )
        return rebuilt

    def apply_material_extensive(self, content: Any, /) -> Array:
        return _sparse_material_apply(
            content,
            self.source_rows,
            self.target_rows,
            self.extensive_weights,
            self.source_reference_measures.shape[0],
            self.target_reference_measures.shape[0],
        )

    def apply_material_extensive_with_evidence(
        self, content: Any, /
    ) -> MaterialMapActionEvidence:
        result = self.apply_material_extensive(content)
        values = jnp.asarray(content).astype(result.dtype)
        before = jnp.sum(values, axis=0)
        after = jnp.sum(result, axis=0)
        return MaterialMapActionEvidence(
            result,
            before,
            after,
            jnp.abs(after - before),
            (self.source_reference_coverage_defect,),
            (self.target_reference_coverage_defect,),
        )

    def apply_material_extensive_jvp(self, tangent: Any, /) -> Array:
        return self.apply_material_extensive(tangent)

    def apply_material_extensive_transpose(self, cotangent: Any, /) -> Array:
        return _sparse_material_transpose(
            cotangent,
            self.source_rows,
            self.target_rows,
            self.extensive_weights,
            self.source_reference_measures.shape[0],
            self.target_reference_measures.shape[0],
        )

    def apply_material_density(self, density: Any, /) -> Array:
        return _sparse_material_apply(
            density,
            self.source_rows,
            self.target_rows,
            self.density_weights,
            self.source_physical_measures.shape[0],
            self.target_physical_measures.shape[0],
        )

    def apply_material_density_jvp(self, tangent: Any, /) -> Array:
        return self.apply_material_density(tangent)

    def apply_material_density_transpose(self, cotangent: Any, /) -> Array:
        return _sparse_material_transpose(
            cotangent,
            self.source_rows,
            self.target_rows,
            self.density_weights,
            self.source_physical_measures.shape[0],
            self.target_physical_measures.shape[0],
        )

    def apply_material_density_with_evidence(
        self, density: Any, /
    ) -> MaterialMapActionEvidence:
        result = self.apply_material_density(density)
        source = jnp.asarray(density).astype(result.dtype)
        expand = (1,) * (source.ndim - 1)
        before = jnp.sum(
            source * self.source_physical_measures.reshape((-1, *expand)), axis=0
        )
        after = jnp.sum(
            result * self.target_physical_measures.reshape((-1, *expand)), axis=0
        )
        return MaterialMapActionEvidence(
            result,
            before,
            after,
            jnp.abs(after - before),
            (self.source_physical_coverage_defect,),
            (self.target_physical_coverage_defect,),
        )

    def apply_material_scalar(self, scalar: Any, /) -> Array:
        return _sparse_material_apply(
            scalar,
            self.source_rows,
            self.target_rows,
            self.scalar_weights,
            self.source_reference_measures.shape[0],
            self.target_reference_measures.shape[0],
        )

    def apply_material_scalar_jvp(self, tangent: Any, /) -> Array:
        return self.apply_material_scalar(tangent)

    def apply_material_scalar_transpose(self, cotangent: Any, /) -> Array:
        return _sparse_material_transpose(
            cotangent,
            self.source_rows,
            self.target_rows,
            self.scalar_weights,
            self.source_reference_measures.shape[0],
            self.target_reference_measures.shape[0],
        )


class VolumeMaterialChain(StrictModule, NonTrainableState):
    """Chronological composition of validated maps over one material reference."""

    maps: tuple[VolumeMaterialMap, ...]
    reference: MaterialReferenceDeclaration
    source_map: MaterialSourceMapCapability
    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    chain_id: str = eqx.field(static=True)

    def __init__(self, maps: Sequence[VolumeMaterialMap], /) -> None:
        records = tuple(maps)
        if not records or not all(
            isinstance(value, VolumeMaterialMap) for value in records
        ):
            raise TypeError(
                "Material chain requires one or more Lagrangian VolumeMaterialMap owners."
            )
        reference_id = records[0].reference.declaration_id
        source_map_id = records[0].source_map.map_id
        frame = records[0].deformation.frame_id
        for previous, following in zip(records, records[1:]):
            if (
                previous.target_mesh.mesh_id != following.source_mesh.mesh_id
                or not _tree_content_equal(previous.target_mesh, following.source_mesh)
                or not _tree_content_equal(
                    previous.target_geometry, following.source_geometry
                )
                or not _tree_content_equal(
                    previous.target_coverage, following.source_coverage
                )
                or previous.target_cell_entity_ids != following.source_cell_entity_ids
                or previous.target_region_ids != following.source_region_ids
                or previous.target_occurrence_paths != following.source_occurrence_paths
                or previous.reference.declaration_id != following.reference.declaration_id
                or following.reference.declaration_id != reference_id
                or following.source_map.map_id != source_map_id
                or following.deformation.frame_id != frame
            ):
                raise VolumeMaterialMapError(
                    "Material maps must join identical endpoint states under one declared reference and frame."
                )
        self.maps = records
        self.reference = records[0].reference
        self.source_map = records[0].source_map
        self.source_mesh_id = records[0].source_mesh.mesh_id
        self.target_mesh_id = records[-1].target_mesh.mesh_id
        self.source_geometry_id = cell_geometry_id(records[0].source_geometry)
        self.target_geometry_id = cell_geometry_id(records[-1].target_geometry)
        self.frame_id = frame
        self.chain_id = canonical_fingerprint(
            {
                "kind": "lagrangian-volume-material-chain",
                "reference": reference_id,
                "source_map": source_map_id,
                "frame": frame,
                "source_geometry": self.source_geometry_id,
                "target_geometry": self.target_geometry_id,
                "maps": tuple(value.map_id for value in records),
            }
        )

    @classmethod
    def revalidate(
        cls, value: VolumeMaterialChain, maps: Sequence[VolumeMaterialMap], /
    ) -> VolumeMaterialChain:
        if not isinstance(value, VolumeMaterialChain):
            raise TypeError("value must be VolumeMaterialChain.")
        rebuilt = cls(maps)
        if rebuilt.chain_id != value.chain_id:
            raise ValueError("Stored material chain does not match recomputed maps.")
        return rebuilt

    def apply_material_extensive(self, content: Any, /) -> Array:
        result = jnp.asarray(content)
        for material_map in self.maps:
            result = material_map.apply_material_extensive(result)
        return result

    def apply_material_extensive_jvp(self, tangent: Any, /) -> Array:
        result = jnp.asarray(tangent)
        for material_map in self.maps:
            result = material_map.apply_material_extensive_jvp(result)
        return result

    def apply_material_extensive_transpose(self, cotangent: Any, /) -> Array:
        result = jnp.asarray(cotangent)
        for material_map in reversed(self.maps):
            result = material_map.apply_material_extensive_transpose(result)
        return result

    def apply_material_density(self, density: Any, /) -> Array:
        result = jnp.asarray(density)
        for material_map in self.maps:
            result = material_map.apply_material_density(result)
        return result

    def apply_material_density_jvp(self, tangent: Any, /) -> Array:
        result = jnp.asarray(tangent)
        for material_map in self.maps:
            result = material_map.apply_material_density_jvp(result)
        return result

    def apply_material_density_transpose(self, cotangent: Any, /) -> Array:
        result = jnp.asarray(cotangent)
        for material_map in reversed(self.maps):
            result = material_map.apply_material_density_transpose(result)
        return result

    def apply_material_scalar(self, scalar: Any, /) -> Array:
        result = jnp.asarray(scalar)
        for material_map in self.maps:
            result = material_map.apply_material_scalar(result)
        return result

    def apply_material_scalar_jvp(self, tangent: Any, /) -> Array:
        result = jnp.asarray(tangent)
        for material_map in self.maps:
            result = material_map.apply_material_scalar_jvp(result)
        return result

    def apply_material_scalar_transpose(self, cotangent: Any, /) -> Array:
        result = jnp.asarray(cotangent)
        for material_map in reversed(self.maps):
            result = material_map.apply_material_scalar_transpose(result)
        return result


def prepare_volume_material_map(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    deformation: VolumeMaterialDeformation,
    source_cell_entity_ids: Sequence[str],
    target_cell_entity_ids: Sequence[str],
    source_region_ids: Sequence[str],
    target_region_ids: Sequence[str],
    /,
    *,
    source_coverage: DomainCoverageCertificate,
    target_coverage: DomainCoverageCertificate,
    policy: CommonRefinementPolicy | None = None,
    source_occurrence_paths: Sequence[Sequence[str]] | None = None,
    target_occurrence_paths: Sequence[Sequence[str]] | None = None,
    integration_policy: CurvedOverlapIntegrationPolicy | None = None,
    source_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    target_chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
) -> VolumeMaterialMap:
    return VolumeMaterialMap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        deformation,
        source_cell_entity_ids,
        target_cell_entity_ids,
        source_region_ids,
        target_region_ids,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        policy=policy,
        source_occurrence_paths=source_occurrence_paths,
        target_occurrence_paths=target_occurrence_paths,
        integration_policy=integration_policy,
        source_chart_parent=source_chart_parent,
        target_chart_parent=target_chart_parent,
    )


__all__ = [
    "CommonRootVolumeWitness",
    "CurvedOverlapIntegrationEvidence",
    "CurvedOverlapIntegrationPolicy",
    "MaterialSourceMapCapability",
    "MaterialMapActionEvidence",
    "EulerianVolumeOverlapChain",
    "EulerianVolumeOverlapMap",
    "MaterialReferenceDeclaration",
    "MaterialReferencePolicy",
    "VolumeMaterialDeformation",
    "VolumeMaterialChain",
    "VolumeMaterialMap",
    "declare_material_reference",
    "VolumeMaterialMapError",
    "require_polyhedral_carrier_geometry",
    "prepare_eulerian_volume_overlap",
    "prepare_volume_material_map",
]
