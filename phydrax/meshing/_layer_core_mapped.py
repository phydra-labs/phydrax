#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lift an actual reference partition through independently authored source roots."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from .._meshcore import charge_native_geometry_queries
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometryElement,
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    add,
    coordinate_corner_images,
    coordinate_polynomials,
    derivative,
    rounded_point,
    scale,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ._boundary_layer import BoundaryLayerMesh
from ._layer_core_resources import LayerCoreSourceWork
from ._quad_generation import _family_host_array
from ._reference_root_composition import exact_reference_chart_controls


if TYPE_CHECKING:
    from .._physical import SpatialCoordinateContract
    from ._contracts import MeshingProviderInfo, VolumeMeshingSpec
    from ._layer_core import LayerCoreConstruction
    from ._measurements import NativeMeshingPhaseRecorder
    from ._organization import MeshAttribute, MeshLabel, MeshPatch, MeshZone
    from ._result import CellMeshingResult
    from .providers._native_layer import PreparedLayerCore
    from .providers._native_sources import NativeLayerCoreSource


class LayerReferenceComposition(NamedTuple):
    mesh: CellMesh
    geometry: CellGeometrySpec
    reference_mesh: CellMesh
    cell_order: np.ndarray
    cell_regions: np.ndarray


def validate_layer_reference_columns(
    layers: BoundaryLayerMesh,
    reference_layers: BoundaryLayerMesh,
    domain: MappedReferenceDomain,
    work: LayerCoreSourceWork,
    /,
) -> None:
    """Prove the entire original physical interval vectors, not sampled heights."""
    if not np.array_equal(
        layers.mesh.vertex_global_ids, reference_layers.mesh.vertex_global_ids
    ):
        raise ValueError(
            "Reference layer columns must retain the actual physical source vertex registry."
        )
    if not np.array_equal(
        layers.layer_index, reference_layers.layer_index
    ) or not np.array_equal(
        layers.column_index,
        reference_layers.column_index,
    ):
        raise ValueError(
            "Reference layer columns must retain every actual physical interval and column identity."
        )
    reference_cells = {
        int(identifier): tuple(
            np.asarray(reference_layers.mesh.vertex_global_ids)[vertices].tolist()
        )
        for block in reference_layers.mesh.blocks
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    elements, routes, _ = domain.source_geometry.resolve(domain.reference_mesh)
    values = domain.source_geometry.source_coordinates()
    roots = {
        int(identifier): (
            block,
            element,
            np.asarray(route, dtype=np.int64)[row],
            vertices,
        )
        for block, element, route in zip(
            domain.reference_mesh.blocks, elements, routes, strict=True
        )
        for row, (identifier, vertices) in enumerate(
            zip(np.asarray(block.global_ids), np.asarray(block.vertices), strict=True)
        )
    }
    physical_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    root_vertex_ids = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    affine = coordinate_lagrange_element("prism", 1)
    for block in layers.mesh.blocks:
        if block.cell_kind != "prism":
            raise ValueError(
                "Source-controlled layer composition requires actual prepared prism columns."
            )
        for identifier, vertices in zip(
            np.asarray(block.global_ids).tolist(), np.asarray(block.vertices), strict=True
        ):
            work.charge(1)
            if identifier not in roots or identifier not in reference_cells:
                raise ValueError(
                    "Every physical source layer cell must own its independently declared reference root."
                )
            root_block, element, route, root_vertices = roots[identifier]
            expected = tuple(physical_ids[vertices].tolist())
            if (
                root_block.cell_kind != "prism"
                or reference_cells[identifier] != expected
                or tuple(root_vertex_ids[root_vertices].tolist()) != expected
            ):
                raise ValueError(
                    "Reference roots changed an actual physical column's directed source corners."
                )
            if not np.array_equal(
                np.asarray(domain.reference_mesh.coordinates)[root_vertices].view(
                    np.uint64
                ),
                np.asarray(reference_layers.mesh.coordinates)[vertices].view(np.uint64),
            ):
                raise ValueError(
                    "Original layer roots must retain their actual authored reference column coordinates."
                )
            local = tuple(values[index] for index in route)
            physical = np.asarray(layers.mesh.coordinates, dtype=np.float64)[vertices]
            images = coordinate_corner_images(element, local)
            if images != tuple(
                tuple(Fraction(float(value)) for value in point) for point in physical
            ):
                raise ValueError(
                    "Reference source roots changed the original physical column corners."
                )
            source = coordinate_polynomials(element, local)
            original = coordinate_polynomials(affine, physical)
            if source is None or original is None:
                raise ValueError(
                    "Actual layer columns require their original exact polynomial coordinate expressions."
                )
            work.charge(sum(len(value) for value in (*source, *original)))
            if any(
                derivative(add(current, scale(initial, -1)), 2)
                for current, initial in zip(source, original, strict=True)
            ):
                raise ValueError(
                    "Mapped layer curvature changed an entire original physical interval vector."
                )
    if set(reference_cells) != set(
        np.asarray(layers.mesh.entity_set(3).entity_ids).tolist()
    ):
        raise ValueError(
            "The reference layer source added or omitted an original physical column cell."
        )


def require_reference_root_sheets(
    reference: NativeLayerCoreSource,
    domain: MappedReferenceDomain,
    work: LayerCoreSourceWork,
    /,
) -> None:
    """Require actual core-root incidence sheets, even within one material."""
    from ._layer_core import _faces, _prepare_identity, _triangles

    mapping, _, polygons, points = _prepare_identity(
        reference.layers,
        reference.complex,
        reference.vertex_layer_ids,
        reference.cap_polygon_ids,
        work=work,
    )
    layer_ids = np.asarray(reference.layers.mesh.vertex_global_ids, dtype=np.int64)
    extra = points.shape[0] - layer_ids.size
    next_vertex = int(np.max(layer_ids)) + 1
    if extra and next_vertex + extra - 1 > np.iinfo(np.int64).max:
        from ._contracts import MeshingFailure, MeshingFailureCategory

        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The reference layer vertex registry has no room for its original core source identities.",
        )
    combined_ids = np.concatenate(
        (layer_ids, next_vertex + np.arange(extra, dtype=np.int64))
    )
    constrained = {tuple(sorted(row.tolist())) for row in combined_ids[mapping[polygons]]}
    cells = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in domain.reference_mesh.blocks
        ]
    )
    sorted_cells = np.sort(cells)
    regions = np.empty(cells.size, dtype=np.int64)
    regions[np.searchsorted(sorted_cells, cells)] = domain.cell_regions
    layer_cells = set(np.asarray(reference.layers.mesh.entity_set(3).entity_ids).tolist())
    vertices = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    for incidents in _faces(domain.reference_mesh, regions, work=work).values():
        if len(incidents) != 2 or any(
            int(sorted_cells[face.cell]) in layer_cells for face in incidents
        ):
            continue
        for triangle in _triangles(incidents[0].vertices):
            work.charge(1)
            key = tuple(sorted(vertices[np.asarray(triangle, dtype=np.int64)].tolist()))
            if key not in constrained:
                raise ValueError(
                    "Native reference fill must retain every original internal core-root incidence sheet, including equal-material sheets."
                )


def compose_layer_reference_mesh(
    reference_mesh: CellMesh,
    domain: MappedReferenceDomain,
    cell_regions: np.ndarray,
    work: LayerCoreSourceWork,
    /,
) -> LayerReferenceComposition:
    """Bind every actual cell to its unique entire exact reference-root chart.

    Native fill must retain every internal source-root incidence sheet, including
    sheets whose two material labels agree. A target crossing a sheet has no
    containing root and is refused; no physical inverse or nearest-root choice
    can conceal missing reference partition constraints.
    """
    if reference_mesh.topological_dimension != 3 or reference_mesh.ambient_dimension != 3:
        raise ValueError(
            "Layer reference composition requires a three-dimensional reference partition."
        )
    count = sum(block.cell_count for block in reference_mesh.blocks)
    regions = np.asarray(cell_regions)
    if regions.shape != (count,) or regions.dtype.kind not in "iu":
        raise ValueError(
            "Layer reference regions must bind every actual cell in block order."
        )
    source_elements, source_routes, _ = domain.source_geometry.resolve(
        domain.reference_mesh
    )
    source_coordinates = domain.source_geometry.source_coordinates()
    reference_points = np.asarray(domain.reference_mesh.coordinates, dtype=np.float64)
    source_ids = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    roots: list[tuple[CellBlock, CellGeometryElement, np.ndarray, int, int]] = []
    source_offset = 0
    for block, element, route in zip(
        domain.reference_mesh.blocks, source_elements, source_routes, strict=True
    ):
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Mapped layer roots require canonical fixed-arity CellBlock reference cells."
            )
        for row in range(block.cell_count):
            roots.append(
                (
                    block,
                    element,
                    np.asarray(route, dtype=np.int64),
                    row,
                    int(domain.cell_regions[source_offset + row]),
                )
            )
        source_offset += block.cell_count
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    groups: dict[str, list[tuple[int, np.ndarray, int, int, np.ndarray, np.ndarray]]] = {}
    elements: dict[str, PolynomialComposedCellGeometryElement] = {}
    exact_points: dict[int, tuple[Fraction, ...]] = {}
    physical_points = _family_host_array(points.shape, np.float64)
    position = 0
    for block in reference_mesh.blocks:
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Mapped layer targets require canonical fixed-arity CellBlock reference cells."
            )
        chart = coordinate_lagrange_element(block.cell_kind, 1)
        for target_row in range(block.cell_count):
            identifier = int(np.asarray(block.global_ids)[target_row])
            vertices = np.asarray(block.vertices[target_row], dtype=np.int64)
            candidates = []
            target = points[vertices]
            for parent, (
                root_block,
                source_element,
                route,
                root_row,
                root_region,
            ) in enumerate(roots):
                work.charge(1)
                if regions[position] != root_region:
                    continue
                root_vertices = np.asarray(root_block.vertices[root_row], dtype=np.int64)
                corners = reference_points[root_vertices]
                charge_native_geometry_queries(1)
                if np.any(np.min(target, axis=0) < np.min(corners, axis=0)) or np.any(
                    np.max(target, axis=0) > np.max(corners, axis=0)
                ):
                    continue
                controls = exact_reference_chart_controls(
                    target, corners, root_block.cell_kind, chart
                )
                if controls is not None:
                    candidates.append((parent, controls))
            if len(candidates) != 1:
                raise ValueError(
                    "An actual reference cell has no unique entire source-root chart; preserve all internal root incidence sheets during fill."
                )
            parent, (numerators, denominators) = candidates[0]
            root_block, source_element, route, root_row, _ = roots[parent]
            source_element = _require_scalar_coordinate_element(
                source_element, "Layer source root"
            )
            element = PolynomialComposedCellGeometryElement(
                source_element, chart, numerators, denominators
            )
            name = f"root-chart:{element.element_id}"
            elements[name] = element
            source_route = np.asarray(route[root_row], dtype=np.int64)
            root_vertices = np.asarray(root_block.vertices[root_row], dtype=np.int64)
            groups.setdefault(name, []).append(
                (
                    position,
                    vertices,
                    identifier,
                    int(np.asarray(root_block.global_ids)[root_row]),
                    source_ids[root_vertices],
                    source_route,
                )
            )
            images = coordinate_corner_images(
                element,
                tuple(source_coordinates[int(index)] for index in source_route.tolist()),
            )
            if images is None:
                raise ValueError(
                    "A layer root composition lacks its original exact coordinate expression."
                )
            work.charge(len(images))
            for vertex, image in zip(vertices.tolist(), images, strict=True):
                previous = exact_points.get(vertex)
                if previous is not None and previous != image:
                    raise ValueError(
                        "Independent layer source charts disagree exactly at a shared reference vertex."
                    )
                exact_points[vertex] = image
                physical_points[vertex] = rounded_point(image)
            position += 1
    if len(exact_points) != points.shape[0]:
        raise ValueError(
            "The reference partition has a vertex without an original source-root chart."
        )
    blocks, routes, parents, parent_vertices, order = [], {}, {}, {}, []
    for name in sorted(groups):
        cells = sorted(groups[name], key=lambda cell: cell[2])
        blocks.append(
            CellBlock(
                name,
                elements[name].cell_kind,
                np.stack([cell[1] for cell in cells]),
                global_ids=np.asarray([cell[2] for cell in cells], dtype=np.int64),
            )
        )
        routes[name] = np.stack([cell[5] for cell in cells])
        parents[name] = np.asarray([cell[3] for cell in cells], dtype=np.int64)
        parent_vertices[name] = np.stack([cell[4] for cell in cells])
        order.extend(cell[0] for cell in cells)
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(domain.source_geometry),
        domain.reference_mesh.topology_id,
        parents,
        parent_vertices,
    )
    geometry = CellGeometrySpec(
        elements,
        routes,
        domain.source_geometry.coordinates,
        restriction_source=ancestry,
        periodic_source=domain.source_geometry.periodic_source,
    )
    reference = CellMesh(
        points,
        tuple(blocks),
        vertex_global_ids=reference_mesh.vertex_global_ids,
        numeric_version=reference_mesh.numeric_version,
    )
    mesh = reference.with_coordinates(
        physical_points, numeric_version=domain.source_revision
    )
    cell_order = np.asarray(order, dtype=np.int64)
    return LayerReferenceComposition(
        mesh, geometry, reference, cell_order, regions[cell_order]
    )


def _mapped_layer_organization(
    construction: LayerCoreConstruction,
    mesh: CellMesh,
    /,
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[MeshAttribute, ...],
]:
    from ._canonical import _entity_vertex_keys
    from ._layer_core import _scope
    from ._organization import MeshAttribute, MeshLabel, MeshPatch, MeshZone
    from ._scope import MeshingScope

    correspondences = {}
    for degree in range(4):
        old = np.asarray(construction.mesh.entity_set(degree).entity_ids, dtype=np.int64)
        new = np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
        if degree in (0, 3):
            # Composition retains explicit vertex/cell identities; only edge
            # and face registries are rebuilt from directed incidence.
            if not np.array_equal(np.sort(old), np.sort(new)):
                raise ValueError(
                    "Mapped organization must retain every explicit source vertex and cell identity."
                )
            correspondences[degree] = dict(zip(old.tolist(), old.tolist(), strict=True))
            continue
        targets = dict(zip(_entity_vertex_keys(mesh, degree), new.tolist(), strict=True))
        correspondences[degree] = {
            identifier: targets[key]
            for identifier, key in zip(
                old.tolist(), _entity_vertex_keys(construction.mesh, degree), strict=True
            )
        }

    def scope(original: MeshingScope) -> MeshingScope:
        degree = original.entity_dimension
        return _scope(
            mesh,
            degree,
            np.asarray(
                [
                    correspondences[degree][identifier]
                    for identifier in np.asarray(original.entity_ids).tolist()
                ],
                dtype=np.int64,
            ),
        )

    zones = tuple(
        MeshZone(
            zone.name,
            zone.role,
            scope(zone.scope),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for zone in construction.zones
    )
    zone_ids = {
        old.zone_id: new.zone_id
        for old, new in zip(construction.zones, zones, strict=True)
    }
    patches = tuple(
        MeshPatch(
            patch.name,
            scope(patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(
                zone_ids[identifier] for identifier in patch.adjacent_zone_ids
            ),
            source_adjacent_region_ids=patch.source_adjacent_region_ids,
        )
        for patch in construction.patches
    )
    labels = tuple(
        MeshLabel(label.name, scope(label.scope)) for label in construction.labels
    )
    attributes = []
    for attribute in construction.attributes:
        renamed = np.asarray(
            [
                correspondences[attribute.scope.entity_dimension][identifier]
                for identifier in np.asarray(attribute.scope.entity_ids).tolist()
            ],
            dtype=np.int64,
        )
        attributes.append(
            MeshAttribute(
                attribute.name,
                attribute.role,
                scope(attribute.scope),
                np.asarray(attribute.values)[np.argsort(renamed, kind="stable")],
                unit=attribute.unit,
            )
        )
    return zones, patches, labels, tuple(attributes)


def execute_mapped_layer_core_route(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedLayerCore,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Native reference fill followed by original-root, column-preserving curving."""
    from dataclasses import replace
    from time import monotonic

    from ..discretization._periodic_topology import PeriodicMeshTopology
    from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
    from ._certification import MeshCertificationSchedule
    from ._contracts import MeshingDerivativeMode
    from ._layer_core import generate_layer_core
    from ._layer_core_controls import compose_layer_controls
    from ._layer_core_periodic import core_periodic_constraint_evidence
    from ._layer_core_resources import _row_entity_vertex_keys
    from ._layer_source_certification import layer_source_fidelity_requests
    from ._mapped_reference_association import mapped_reference_associations
    from ._measurements import measure_phase
    from ._result import MeshingComplianceReport
    from ._sizing import UniformSizeControl
    from ._trace import MeshingStageKind
    from .providers._native_publication import (
        check_deadline,
        edge_size_evidence,
        NativeCertificationRequest,
        publish_native_result,
        uniform_size_compliance,
    )

    reference, domain = source.reference_source, source.mapped_domain
    reference_prepared = prepared.reference_prepared
    if reference is None or domain is None or reference_prepared is None:
        raise ValueError(
            "Mapped layer/core execution requires the exact independently prepared source roots and reference partition."
        )
    started = monotonic()
    limits = specification.limits
    work = LayerCoreSourceWork(limits.maximum_work_units, prepared.source_work_units)
    periodic_requested, periodic_achieved = core_periodic_constraint_evidence(
        source, specification, work=work
    )
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    construction = generate_layer_core(
        reference.layers,
        reference.complex,
        reference_prepared.core_specification,
        prepared.schedule,
        validity_policy=audit_policy.validity_policy,
        vertex_layer_ids=reference.vertex_layer_ids,
        cap_polygon_ids=reference.cap_polygon_ids,
        layer_regions=reference.layer_regions,
        source_id=reference.source_id,
        source_revision=reference.source_revision,
        input_id=plan_id,
        source_binding=reference,
        operation_started=started,
        source_work_units=work.work_units,
        record_phase=record_phase,
    )
    work.work_units = construction.work_units
    with measure_phase(record_phase, "curving"):
        composed = compose_layer_reference_mesh(
            construction.mesh, domain, construction.cell_regions, work
        )
        mesh = composed.mesh
        periodic = source.layers.mesh.periodic_topology
        reference_periodic = construction.mesh.periodic_topology
        if (periodic is None) != (reference_periodic is None):
            raise ValueError(
                "Physical and reference layer sources must retain the same declared periodic column ancestry."
            )
        if periodic is not None and reference_periodic is not None:
            topology = PeriodicMeshTopology(
                mesh,
                periodic.cell,
                reference_periodic.vertex_representatives,
                reference_periodic.vertex_shifts,
            )
            mesh = CellMesh(
                mesh.coordinates,
                mesh.blocks,
                vertex_global_ids=mesh.vertex_global_ids,
                numeric_version=mesh.numeric_version,
                periodic_topology=topology,
            )
        from ._layer_curving import _source_layer_cells

        _source_layer_cells(mesh, source.layers)
    with measure_phase(record_phase, "organization"):
        zones, patches, labels, attributes = _mapped_layer_organization(
            construction, mesh
        )
        physical = replace(
            construction,
            mesh=mesh,
            domain=domain,
            cell_regions=composed.cell_regions,
            zones=zones,
            patches=patches,
            labels=labels,
            attributes=attributes,
        )
        zones, patches = compose_layer_controls(
            source, specification, physical, work=work
        )
    with measure_phase(record_phase, "geometry_association"):
        associations = mapped_reference_associations(
            domain,
            mesh,
            composed.geometry,
            maximum_support_queries=min(
                limits.maximum_geometry_queries, limits.maximum_work_units
            ),
        )
    with measure_phase(record_phase, "compliance"):
        size = specification.size_controls[0]
        if not isinstance(size, UniformSizeControl):
            raise TypeError(
                "Mapped layer/core publication requires its original uniform physical size control."
            )
        lengths, growth = edge_size_evidence(
            np.asarray(mesh.coordinates, dtype=np.float64),
            np.asarray(_row_entity_vertex_keys(mesh, 1), dtype=np.int64),
        )
        requested, achieved, issues = uniform_size_compliance(
            size, specification.size_compliance, lengths, growth
        )
        bounds = tuple(
            control.core_maximum_size
            for control in (*specification.layer_controls, source.layers.control)
            if control.core_maximum_size is not None
        )
        if bounds:
            cap = min(bounds)
            vertices = np.asarray(mesh.coordinates, dtype=np.float64)
            core_cells = np.concatenate(
                [
                    np.asarray(block.vertices, dtype=np.int64)
                    for block in mesh.blocks
                    if block.cell_kind == "tetrahedron"
                ]
            )
            from .providers._native_publication import unique_edges

            core_edges = unique_edges(core_cells, "tetrahedron")
            maximum = float(
                np.max(
                    np.linalg.norm(
                        vertices[core_edges[:, 1]] - vertices[core_edges[:, 0]], axis=1
                    )
                )
            )
            requested.append(("core_maximum_size", cap))
            achieved.append(("core_maximum_edge", maximum))
            if (
                maximum
                > cap
                + specification.size_compliance.absolute_tolerance
                + specification.size_compliance.relative_tolerance * cap
            ):
                issues.append("core_maximum_size")
        achieved.extend(
            (f"reference:construction:{name}", value)
            for name, value in construction.core.construction_counters
        )
        achieved.extend(
            (
                ("reference:native:work_units", construction.core.work_units),
                ("reference:native:steiner_points", construction.core.steiner_points),
            )
        )
        evidence = source.layers.evidence
        compliance = MeshingComplianceReport(
            specification.specification_id,
            requested=(
                *requested,
                *periodic_requested,
                *(
                    (f"layer:{index}:thickness", value)
                    for index, value in enumerate(evidence.requested_thicknesses)
                ),
            ),
            achieved=(
                *achieved,
                *periodic_achieved,
                ("original_polynomial_column_vectors", 1),
                ("fixed_cap_vertices_bitwise", 1),
                ("fixed_reference_root_sheets", 1),
                ("layer_core_prepublication_work_units", work.work_units),
                *(
                    (f"layer:{index}:mean_thickness", float(value))
                    for index, value in enumerate(
                        np.asarray(evidence.achieved_thicknesses)
                    )
                    if np.isfinite(value)
                ),
            ),
            issues=tuple(issues),
        )
    scoped = layer_source_fidelity_requests(source, specification, physical, work=work)
    fidelity = source.fidelity_source
    if fidelity is None:
        raise ValueError(
            "Mapped layer/core publication must retain its independent original physical geometry query."
        )
    from ..geometry._mesh_certificates import MappedDomainBoundarySource

    if isinstance(fidelity, MappedDomainBoundarySource):
        # The source owns immutable root-region facts; certification binds the
        # same exact mapped boundary to this publication's actual cell inventory.
        fidelity = fidelity.for_target_regions(composed.cell_regions)
    from ._controls import FeatureKind

    tolerance = min(
        (
            feature.maximum_deviation
            for feature in specification.protected_features
            if feature.feature_kind is FeatureKind.SURFACE
            and feature.scope.scope_id == specification.boundary_scope.scope_id
        ),
        default=size.target_size,
    )
    request = NativeCertificationRequest(
        MeshCertificationSchedule("mapped_volume"),
        source.source_id,
        source.source_revision,
        limits,
        domain=domain,
        cell_regions=composed.cell_regions,
        fidelity_source=fidelity,
        fidelity_tolerance=tolerance,
        scoped_fidelity=scoped,
    )
    check_deadline(started, limits, MeshingStageKind.CERTIFICATION)
    result = publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        construction.stages,
        provider,
        {
            "kind": "native-mapped-layer-core",
            "plan": plan_id,
            "specification": specification.specification_id,
            "source": source.binding_id,
            "reference": reference.binding_id,
            "mapped_domain": domain.domain_id,
        },
        request,
        geometry=composed.geometry,
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "wall_time",
        ),
        unenforced_limits=("native_workspace",),
        zones=zones,
        patches=patches,
        labels=labels,
        attributes=attributes,
        associations=associations,
        record_phase=record_phase,
    )
    check_deadline(started, limits, MeshingStageKind.CERTIFICATION)
    return result
