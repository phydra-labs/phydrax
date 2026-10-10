#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit native-family owners admitted by the canonical scientific archive.

This is a constructor/admission boundary, not an identity registry or serializer.
Execution worksets are rebuilt by their providers from these authored values.
"""

from __future__ import annotations

import json
from dataclasses import fields
from inspect import Parameter, signature
from typing import Any, TypeGuard

import numpy as np

from .._array_archive import ArrayArchiveLimits
from ..discretization import (
    _cell_complex,
    _cell_geometry,
    _cell_mesh,
    _hexahedral,
    _periodic_topology,
)
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._periodic_geometry import PeriodicCellGeometrySource
from ..discretization.fem._reference import FiniteElementSpec
from ..geometry import _mapped_reference_domain, _mesh_certificates, _meshing_domain
from ..geometry._compartments import CompartmentMeshingSource
from ..geometry._triangulation import PeriodicPowerPreparation
from ..geometry.brep import _intersection
from ..geometry.surface import _contracts as _surface_contracts, _model as _surface_model
from ..meshing import (
    _audit,
    _boundary_layer,
    _certification,
    _certification_inputs,
    _contracts,
    _controls,
    _hex_generation,
    _periodic,
    _polyhedral_generation,
    _result,
    _scope,
    _structured,
    _surface_envelope,
    _sweep,
    _volume_generation,
)
from ..meshing._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from ..meshing.providers import _native_periodic, _native_sources
from ..meshing.providers._native_options import NativeMeshingOptions


type NativeGenerationSource = (
    _native_sources.NativePlanarSource
    | _native_sources.NativePlcSource
    | _native_sources.NativeStructuredSource
    | _native_sources.NativeMappedHexSource
    | _native_sources.NativePolyhedralSource
    | _native_sources.NativeSurfaceSource
    | _native_sources.NativeSweepSource
    | _native_periodic.NativePeriodicSource
    | _native_sources.NativeLayerCoreSource
    | _native_sources.NativeSurfaceEnvelopeSource
    | CompartmentMeshingSource
)


def native_generation_source_types() -> tuple[type, ...]:
    return (
        _native_sources.NativePlanarSource,
        _native_sources.NativePlcSource,
        _native_sources.NativeStructuredSource,
        _native_sources.NativeMappedHexSource,
        _native_sources.NativePolyhedralSource,
        _native_sources.NativeSurfaceSource,
        _native_sources.NativeSweepSource,
        _native_periodic.NativePeriodicSource,
        _native_sources.NativeLayerCoreSource,
        _native_sources.NativeSurfaceEnvelopeSource,
        CompartmentMeshingSource,
    )


def is_native_generation_source(value: object, /) -> TypeGuard[NativeGenerationSource]:
    """Admit only exact closed native authored-source owners."""
    return type(value) in native_generation_source_types()


def native_family_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    """Finite, reviewed owner list; canonical registration remains in _artifacts."""
    return (
        (
            "phydrax.meshing",
            (
                _native_sources.NativePlcSource,
                _native_sources.NativeStructuredSource,
                _native_sources.NativeMappedHexSource,
                _native_sources.NativePolyhedralSource,
                _native_sources.NativeSurfaceSource,
                _volume_generation.PiecewiseLinearComplex,
                _volume_generation.NativeVolumeSchedule,
                _polyhedral_generation.NativePolyhedralSchedule,
                _hex_generation.NativeHexGridSchedule,
                _structured.TransfiniteBlock,
                _contracts.VolumeMeshingSpec,
                _contracts.VolumeFillStrategy,
                _controls.RegionSeed,
                _controls.HoleSeed,
                _controls.TransfiniteCurveControl,
                _controls.TransfiniteSurfaceControl,
                _controls.BlockInterfaceControl,
                _native_periodic.NativePeriodicSource,
                _periodic.PeriodicPointOrbits,
                _native_periodic.PeriodicAssociationTransfer,
                _native_sources.NativeSweepSource,
                _sweep.SweepControl,
                _sweep.SweepMapKind,
                _controls.LayerSchedule,
                _native_sources.NativeLayerCoreSource,
                _boundary_layer.BoundaryLayerMesh,
                _boundary_layer.BoundaryLayerEvidence,
                _boundary_layer.BoundaryLayerPolicy,
                _controls.BoundaryLayerControl,
                _controls.BoundaryLayerCollisionPolicy,
                _controls.BoundaryLayerCornerPolicy,
                _controls.BoundaryLayerRoute,
                _controls.PeriodicConstraint,
                _native_sources.NativeSurfaceEnvelopeSource,
                _surface_envelope.RawTriangleSoup,
                _surface_envelope.SurfaceEnvelopePolicy,
                _surface_envelope.SurfaceEnvelope,
                _surface_envelope.SurfaceEnvelopeEvidence,
                _surface_envelope.EnvelopeTopology,
                _surface_envelope.EnvelopeFeaturePermission,
                _surface_envelope.EnvelopeTopologyPermission,
            ),
        ),
        (
            "phydrax.discretization",
            (
                _cell_mesh.PolyhedralBlock,
                _cell_complex.PolyhedralConnectivity,
                _hexahedral.HexahedralConnectivity,
                _cell_geometry.RestrictedCellGeometryElement,
                _cell_geometry.PolynomialComposedCellGeometryElement,
                _cell_geometry.RationalComposedCellGeometryElement,
                _cell_geometry.BarycentricCellGeometryElement,
                _cell_geometry.SplineCellGeometryElement,
                _cell_geometry.LayerColumnCellGeometryElement,
                _cell_geometry.CellVertexGeometryElement,
                _cell_geometry.CellGeometryRestrictionSource,
                ExactPlcCellGeometrySource,
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPlcCellGeometryConvexSource,
                ExactPowerCellGeometryLinearActionSource,
                PeriodicCellGeometrySource,
            ),
        ),
        (
            "phydrax.geometry",
            (
                _mapped_reference_domain.MappedReferenceDomain,
                _mesh_certificates.MappedDomainBoundarySource,
                PeriodicPowerPreparation,
            ),
        ),
        (
            "phydrax.geometry.surface",
            (
                _surface_contracts.SurfaceMetadata,
                _surface_contracts.SurfaceSelection,
                _surface_contracts.SurfaceInterface,
                _surface_contracts.SurfaceAuditPolicy,
                _surface_contracts.SurfaceOrientationRepair,
                _surface_contracts.SurfaceChartMappingEvidence,
                _surface_contracts.SurfaceAuditReport,
                _surface_contracts.SurfaceValidityCertificate,
                _surface_model.SurfaceModel,
                _surface_model.SurfaceRealization,
            ),
        ),
        ("phydrax.geometry.brep", (_intersection.BranchRootEndpoint,)),
    )


_CONSTRUCTOR_TYPES = (
    _native_sources.NativeSurfaceSource,
    _native_sources.NativeStructuredSource,
    _native_sources.NativeMappedHexSource,
    _structured.TransfiniteBlock,
    _native_sources.NativeSweepSource,
    _controls.LayerSchedule,
    _contracts.VolumeMeshingSpec,
    _controls.RegionSeed,
    _controls.HoleSeed,
    _controls.TransfiniteCurveControl,
    _controls.TransfiniteSurfaceControl,
    _controls.BlockInterfaceControl,
    _volume_generation.NativeVolumeSchedule,
    _boundary_layer.BoundaryLayerPolicy,
    _controls.BoundaryLayerControl,
    _controls.PeriodicConstraint,
    _surface_envelope.RawTriangleSoup,
    _surface_envelope.SurfaceEnvelopePolicy,
    _surface_contracts.SurfaceMetadata,
    _surface_contracts.SurfaceSelection,
    _surface_contracts.SurfaceInterface,
    _surface_contracts.SurfaceAuditPolicy,
    _surface_contracts.SurfaceOrientationRepair,
    _surface_contracts.SurfaceChartMappingEvidence,
    _surface_model.SurfaceModel,
    _polyhedral_generation.NativePolyhedralSchedule,
    _hex_generation.NativeHexGridSchedule,
    _cell_mesh.PolyhedralBlock,
    _cell_geometry.RestrictedCellGeometryElement,
    _cell_mesh.CellBlock,
    _cell_geometry.BarycentricCellGeometryElement,
    _cell_geometry.SplineCellGeometryElement,
    _cell_geometry.CellVertexGeometryElement,
    _mapped_reference_domain.MappedReferenceDomain,
    _mesh_certificates.MappedDomainBoundarySource,
    _intersection.BranchRootEndpoint,
    _periodic.PeriodicPointOrbits,
    _native_periodic.PeriodicAssociationTransfer,
    _periodic_topology.PeriodicIsometryGroup,
    ExactPlcCellGeometrySource,
    ExactPowerCellGeometrySource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPlcCellGeometryConvexSource,
)


def rebuild_native_family_value(node: Any, /) -> Any | None:
    """Reestablish real owner constructor checks without rewriting source facts."""
    kind = type(node)
    if kind is PeriodicPowerPreparation:
        node.validate_restored()
        return None
    if kind is ExactPowerCellGeometryLinearActionSource:
        return kind(
            node.parent,
            node.vertex_parents,
            node.vertex_coefficients,
            node.vertex_actions,
            periodic_preparation=node.periodic_preparation,
        )
    if kind is _cell_geometry.LayerColumnCellGeometryElement:
        from ..discretization._coordinate_enclosure import coordinate_source_signature

        fresh = kind(node.wall_element, fiber_graph=node.fiber_graph)
        for name in (
            "cell_kind",
            "conformity",
            "local_dof_count",
            "degree",
            "topological_dimension",
            "station_axis",
            "fiber_graph",
            "element_id",
        ):
            actual = object.__getattribute__(node, name)
            expected = object.__getattribute__(fresh, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"Restored layer column {name} differs from its authored profile owner."
                )
        if type(node.corner_element) is not type(
            fresh.corner_element
        ) or coordinate_source_signature(
            node.corner_element
        ) != coordinate_source_signature(fresh.corner_element):
            raise ValueError("Restored layer column changed its canonical corner source.")
        return fresh
    if kind is PeriodicCellGeometrySource:
        return kind(
            node.source_mesh,
            node.source_geometry,
            active_indices=node.active_indices,
            representative_coordinates=node.representative_coordinates,
            numbering=None,
        )
    if kind is _cell_geometry._CoordinateTabulator:
        return kind(node.cell_kind, node.degree)
    if kind is _cell_geometry._SweptCoordinateTabulator:
        return kind(node.source)
    if (
        kind is FiniteElementSpec
        and type(node.tabulator) is _cell_geometry._SweptCoordinateTabulator
    ):
        return _cell_geometry.swept_coordinate_element(node.tabulator.source)
    if kind in (
        _cell_geometry.PolynomialComposedCellGeometryElement,
        _cell_geometry.RationalComposedCellGeometryElement,
    ):
        numerators = np.asarray(
            [[a for a, _ in row] for row in node.chart_coefficients], dtype=object
        )
        denominators = np.asarray(
            [[b for _, b in row] for row in node.chart_coefficients], dtype=object
        )
        return kind(node.source_element, node.chart_element, numerators, denominators)
    if kind is _sweep.SweepControl:
        frames = (
            {}
            if node.kind is not _sweep.SweepMapKind.FRAMES
            else {
                "frames": node.frames,
                "translations": node.translations,
            }
        )
        return kind(
            node.kind,
            node.schedule,
            node.map_id,
            origin=node.origin,
            axis=node.axis,
            closed=node.closed,
            **frames,
        )
    if (
        kind is FiniteElementSpec
        and type(node.tabulator) is _cell_geometry._CoordinateTabulator
    ):
        return _cell_geometry.coordinate_lagrange_element(node.cell_kind, node.degree)
    if kind is _scope.MeshingScope and node._local_entity_universe is None:
        return kind(
            node.source_id,
            node.source_revision,
            node.entity_kind,
            node.entity_dimension,
            node.entity_set_id,
            node.global_entity_ids,
        )
    if kind is _volume_generation.PiecewiseLinearComplex:
        loops = tuple(
            node.polygon_vertices[first:last]
            for first, last in zip(
                node.polygon_offsets[:-1],
                node.polygon_offsets[1:],
                strict=True,
            )
        )
        return kind(
            node.vertices,
            loops,
            node.polygon_facets,
            node.facet_regions,
            node.region_ids,
            segments=node.segments,
            boundary=node.boundary,
        )
    if kind is _surface_model.SurfaceRealization:
        return kind._prepare(node.model, node.mesh, node.policy)
    if kind is _surface_envelope.SurfaceEnvelope:
        # The repaired carrier and directed bounds need their original source
        # theorem, not merely constructor checks on self-reported evidence.
        fresh = _surface_envelope.wrap_surface_envelope(node.source, node.policy)
        return kind(
            fresh.source,
            fresh.policy,
            fresh.repaired,
            fresh.evidence,
            fresh.carrier_vertices,
            fresh.carrier_tetrahedra,
            execution_evidence=node.execution_evidence,
        )
    if kind is _native_sources.NativeSurfaceEnvelopeSource:
        regions = node.plc_source.complex.region_ids
        if len(regions) != 1:
            raise ValueError(
                "A restored envelope source requires its single authored solid region."
            )
        return kind(node.envelope, regions[0])
    if kind is _native_sources.NativeLayerCoreSource:
        return kind(
            node.layers,
            node.complex,
            node.source_id,
            node.source_revision,
            vertex_layer_ids=node.vertex_layer_ids,
            cap_polygon_ids=node.cap_polygon_ids,
            layer_regions=node.layer_regions,
            region_ids=node.region_ids,
            core_region_map=node.core_region_map,
            source_boundary_scope=node.source_boundary_scope,
            source_domain=node.source_domain,
            fidelity_source=node.fidelity_source,
            core_facet_source_ids=node.core_facet_source_ids,
            core_vertex_representatives=node.core_vertex_representatives,
            core_vertex_shifts=node.core_vertex_shifts,
            core_seam_polygon_pairs=node.core_seam_polygon_pairs,
            reference_source=node.reference_source,
            mapped_domain=node.mapped_domain,
        )
    if kind is _boundary_layer.BoundaryLayerEvidence:
        return kind(
            **{
                member.name: object.__getattribute__(node, member.name)
                for member in fields(node)
                if member.name not in ("evidence_id", "active_column_counts")
            }
        )
    if kind is _boundary_layer.BoundaryLayerMesh:
        return kind(
            node.mesh,
            node.cap,
            node.wall_vertices,
            node.cap_vertices,
            node.layer_index,
            node.validity,
            node.evidence,
            column_index=np.asarray(node.column_index),
            control=node.control,
            source_wall=node.source_wall,
            wall_association=node.wall_association,
            source_domain=node.source_domain,
            policy_id=node.policy_id,
        )
    if kind is _native_sources.NativePlcSource:
        return kind(node.complex, node.source_id, node.source_revision)
    if kind is _native_sources.NativePolyhedralSource:
        return kind(
            node.complex,
            node.source_id,
            node.source_revision,
            sites=node.sites,
            weights=node.weights,
        )
    if kind is _cell_geometry.CellGeometryRestrictionSource:
        return kind(
            node.source_geometry_id,
            node.source_topology_id,
            node.block_parent_cell_ids,
            node.block_parent_vertex_ids,
            block_source_blocks=node.block_source_blocks,
        )
    if kind is _native_periodic.NativePeriodicSource:
        # Seed orbits carry their implicit single region, never an authored partition.
        regions = (
            None
            if isinstance(node.domain, _periodic.PeriodicPointOrbits)
            else node.cell_regions
        )
        return kind(
            node.domain, node.source_id, node.source_revision, cell_regions=regions
        )
    if kind is _cell_mesh.CellMesh and node.storage is None:
        # Face loops, their orientations, and arbitrary-width cell incidence are
        # authored polyhedral facts. Vertex rows alone cannot reconstruct them.
        def carrier(
            periodic: _periodic_topology.PeriodicMeshTopology | None, /
        ) -> _cell_mesh.CellMesh:
            return kind(
                node.coordinates,
                node.blocks,
                vertex_global_ids=node.vertex_global_ids,
                entity_global_ids={
                    dimension: node.entity_set(dimension).entity_ids
                    for dimension in range(node.topological_dimension + 1)
                },
                polyhedral_connectivity=(
                    node.connectivity
                    if type(node.connectivity) is _cell_complex.PolyhedralConnectivity
                    else None
                ),
                periodic_topology=periodic,
                numeric_version=node.numeric_version,
            )

        # The quotient descriptor is revalidated on the plain lifted carrier,
        # retaining its quotient IDs and allocation cursors.
        periodic = node.periodic_topology
        return carrier(None if periodic is None else periodic.rebuilt(carrier(None)))
    if kind not in _CONSTRUCTOR_TYPES:
        return None
    declared = {member.name for member in fields(node)}
    positional, keywords = [], {}
    for name, parameter in signature(kind).parameters.items():
        if (
            parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            or name not in declared
        ):
            raise TypeError(
                f"{kind.__name__} lacks its authored constructor input {name!r}."
            )
        value = object.__getattribute__(node, name)
        if parameter.kind is Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[name] = value
    return kind(*positional, **keywords)


def native_family_audit_policy(
    source: Any, specification: Any, mesh: Any, /
) -> _audit.CellMeshAuditPolicy:
    """The actual provider-owned policy, not planar-policy substitution."""
    disposition = _audit.CellMeshAuditDisposition
    if type(source) in (
        _native_sources.NativeStructuredSource,
        _native_sources.NativeSweepSource,
    ):
        return _audit.CellMeshAuditPolicy(
            watertight_boundary=disposition.REJECT
            if mesh.topological_dimension == 3
            else disposition.SKIP,
        )
    if type(source) is _native_sources.NativeSurfaceSource:
        from ..meshing._domain import compile_surface_domain
        from ..meshing.providers._native_surface import _closed

        closed = _closed(compile_surface_domain(source.domain, specification))
        return _audit.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=disposition.REJECT if closed else disposition.SKIP,
        )
    if type(source) is _native_sources.NativeLayerCoreSource:
        return _audit.CellMeshAuditPolicy(
            require_complete_association=source.mapped_domain is not None,
            watertight_boundary=disposition.REJECT,
        )
    if type(source) in (
        _native_sources.NativePlanarSource,
        _native_sources.NativePlcSource,
        _native_sources.NativeMappedHexSource,
        _native_sources.NativePolyhedralSource,
        _native_sources.NativeSurfaceEnvelopeSource,
        CompartmentMeshingSource,
    ):
        return _audit.CellMeshAuditPolicy(
            require_complete_association=True, watertight_boundary=disposition.REJECT
        )
    if type(source) is _native_periodic.NativePeriodicSource:
        # The closed quotient has no physical boundary; the provider audits by default policy.
        return _audit.CellMeshAuditPolicy()
    raise TypeError(
        "A generation registration requires an explicitly admitted native family."
    )


def _require_registered_compartment(
    source: CompartmentMeshingSource,
    specification: _contracts.VolumeMeshingSpec,
    result: _result.CellMeshingResult,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    """Replay the actual occupied-image/material and independently authored outer source."""
    from ..meshing._compartments import revalidate_region_evidence
    from ._meshing_sources import _native_authority_equal

    source.validate_source_integrity()
    evidence = result.region_evidence
    if evidence is None or result.certification is None:
        raise ValueError(
            "Image registration requires the complete original material/source theorem."
        )
    evidence.require_source(source.compartments)
    evidence.require_current(
        result.mesh, result.zones, result.patches, geometry=result.geometry
    )
    request = result.certification.request
    if not _native_authority_equal(request.domain, evidence.domain, limits=limits):
        raise ValueError(
            "Image certificate lost its exact authoritative material domain."
        )
    regions = np.asarray(
        [evidence.domain.region_ids.index(region) for region in evidence.cell_region_ids],
        dtype=np.int64,
    )
    if request.cell_regions is None or not np.array_equal(request.cell_regions, regions):
        raise ValueError(
            "Image certificate lost its actual cell-to-material assignments."
        )
    renewal = revalidate_region_evidence(
        result,
        result.mesh,
        result.geometry,
        result.zones,
        result.patches,
        compartment_source=source,
        certificate_limits=request.limits,
        limits=specification.limits,
    )
    if not _native_authority_equal(evidence, renewal.region_evidence, limits=limits):
        raise ValueError(
            "Image material/interface/outer-source renewal differs from its original scientific authority."
        )
    if not _native_authority_equal(
        result.zones, renewal.zones, limits=limits
    ) or not _native_authority_equal(result.patches, renewal.patches, limits=limits):
        raise ValueError(
            "Image registration lost its exact material-zone and interface organization."
        )


def _require_registered_plc_geometry(
    source: _native_sources.NativePlcSource,
    specification: _contracts.VolumeMeshingSpec,
    result: _result.CellMeshingResult,
    /,
) -> None:
    """Bind exact coordinates to independent original source rows and their bounds."""
    exact = result.geometry.exact_source
    if not isinstance(exact, ExactPlcCellGeometrySource):
        return
    if (exact.domain_source_id, exact.domain_source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError(
            "Registered exact PLC geometry must retain its original source identity and revision."
        )
    prepared = _volume_generation.prepare_plc_source(
        source.complex,
        source.source_id,
        source.source_revision,
        result.coordinate_contract,
        limits=specification.limits,
    )
    transfer = prepared.association_transfer
    triangles, edges = (
        np.asarray(transfer.triangle_vertices),
        np.asarray(transfer.edge_vertices),
    )
    diagonals = (
        _volume_generation._facet_diagonals(triangles, prepared.input_polygons, edges)[0]
        if source.complex.boundary == "fixed"
        else np.empty((0, 2), dtype=np.int32)
    )
    expected = _volume_generation._plc_source_rows(
        source.complex,
        specification,
        triangles,
        prepared.input_polygons,
        edges,
        diagonals,
    )
    exact.require_domain(
        expected[0], expected[1], source.source_id, source.source_revision
    )
    for name, declared in zip(
        (
            "source_points",
            "source_triangles",
            "source_triangle_ids",
            "source_triangle_bounds",
            "source_segments",
            "source_segment_ids",
            "source_segment_bounds",
        ),
        expected,
        strict=True,
    ):
        actual = np.asarray(object.__getattribute__(exact, name))
        same = actual.shape == declared.shape and (
            np.array_equal(actual.view(np.uint64), declared.view(np.uint64))
            if declared.dtype == np.dtype(np.float64)
            else np.array_equal(actual, declared)
        )
        if not same:
            raise ValueError(
                f"Registered exact PLC geometry changed its original {name} bank."
            )


def _require_registered_layer_authority(
    request: _certification_inputs.MeshCertificationInputs,
    source: _native_sources.NativeLayerCoreSource,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    """Keep original fidelity separate from its declared approximation coverage."""
    from ._meshing_sources import _native_authority_equal

    if not _native_authority_equal(request.source, source.fidelity_source, limits=limits):
        raise ValueError(
            "Layer certificate must retain its independent original fidelity source."
        )
    original = source.source_domain
    if isinstance(original, _mesh_certificates.PiecewiseLinearDomain):
        if not _native_authority_equal(request.domain, original, limits=limits):
            raise ValueError(
                "Layer certificate must retain its independently authored represented domain."
            )
    elif isinstance(original, _meshing_domain.MeshingDomain):
        query = request.source
        if type(query) is not _meshing_domain.MeshingDomainBoundarySource:
            raise ValueError(
                "Layer certificate requires its original parametric domain query."
            )
        if not _native_authority_equal(query.domain, original, limits=limits):
            raise ValueError(
                "Layer certificate query must retain its independently authored parametric domain."
            )
    elif original is not None:
        raise TypeError(
            "Layer source domain must retain its canonical represented or parametric owner."
        )


def _require_registered_envelope(
    source: _native_sources.NativeSurfaceEnvelopeSource,
    specification: _contracts.VolumeMeshingSpec,
    result: _result.CellMeshingResult,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> None:
    from ._meshing_sources import _native_authority_equal

    report = result.certification
    if report is None:
        raise ValueError(
            "Registered envelope volume requires its complete source certificate."
        )
    original = source.plc_source
    declared = _volume_generation.declared_plc_domain(
        original.complex, original.source_id
    )
    if not _native_authority_equal(report.request.domain, declared, limits=limits):
        raise ValueError(
            "Envelope volume differs from its independently authored repaired PLC domain."
        )
    _require_registered_plc_geometry(original, specification, result)
    evidence = source.envelope.evidence
    stage = MeshingStageReport(
        MeshingStageKind.TOPOLOGY_REPAIR,
        MeshingStageStatus.PASSED,
        input_ids=(
            evidence.source_id,
            evidence.source_revision,
            evidence.source_geometry_id,
            evidence.policy_id,
        ),
        output_ids=(
            source.source_id,
            source.source_revision,
            evidence.repaired_model_id,
            evidence.evidence_id,
        ),
        created_count=evidence.repaired_topology.faces,
        deleted_count=evidence.source_topology.faces,
    )
    if result.trace.stages[:1] != (stage,):
        raise ValueError("Envelope volume lost its original source repair theorem.")
    achieved = dict(result.compliance.achieved)
    for name in (
        "source_to_repaired_upper",
        "repaired_to_source_upper",
        "source_inclusion_margin",
        "interpolation_error",
    ):
        if achieved.get(f"envelope:{name}") != object.__getattribute__(evidence, name):
            raise ValueError(f"Envelope volume changed its original {name} evidence.")


def require_same_native_scientific_theorem(
    original: Any,
    renewed: Any,
    mesh: Any,
    geometry: Any,
    audit: Any,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> tuple[Any, Any]:
    """Authenticate both genuine proofs without equating two execution ledgers.

    Exact-expression work/storage and identities of freshly renewed dependent
    proofs are historical execution receipts, not scientific premises. Every
    original and renewed owner is reconstructed before those receipt fields are
    excluded from the semantic comparison; neither report is rewritten.
    """
    from ._meshing_sources import (
        _native_authority_equal,
        register_meshing_source_artifacts,
    )

    register_meshing_source_artifacts()

    def equal(left: Any, right: Any) -> bool:
        return _native_authority_equal(left, right, limits=limits)

    def reconstruct(value: Any, **context: Any) -> None:
        positional, keywords = [], {}
        for parameter in signature(type(value)).parameters.values():
            if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                raise TypeError(
                    "Scientific proof constructors require explicit current inputs."
                )
            argument = (
                context[parameter.name]
                if parameter.name in context
                else getattr(value, parameter.name)
            )
            if parameter.kind is Parameter.POSITIONAL_ONLY:
                positional.append(argument)
            else:
                keywords[parameter.name] = argument
        rebuilt = type(value)(*positional, **keywords)
        if not equal(value, rebuilt):
            raise ValueError(
                "Scientific proof differs from its complete current constructor and identity."
            )

    for report in (original, renewed):
        if type(report) is not _certification.MeshCertificationReport:
            raise TypeError(
                "Scientific renewal requires exact registered certification reports."
            )
        report.request.validate_source_integrity()
        if type(report.request.limits) is not _mesh_certificates.MeshCertificateLimits:
            raise TypeError(
                "Scientific renewal requires its exact original certificate controls."
            )
        reconstruct(report.request.limits)
        reconstruct(report.schedule)
        report.require_passed()
        embedding, coverage = report.embedding, report.coverage
        if embedding is not None:
            if type(embedding) is not _mesh_certificates.GlobalEmbeddingCertificate:
                raise TypeError(
                    "Scientific renewal requires the exact embedding proof owner."
                )
            embedding.binding.require(mesh, geometry)
            reconstruct(
                embedding.binding,
                mesh=mesh,
                geometry=geometry,
                limits=report.request.limits,
            )
            for finding in embedding.findings:
                reconstruct(finding)
            if embedding.binding.limits_id != report.request.limits.limits_id:
                raise ValueError(
                    "Historical embedding receipt changes its actual certificate controls."
                )
            for name, maximum in (
                (
                    "source_expression_work_units",
                    report.request.limits.maximum_work_units,
                ),
                (
                    "source_expression_peak_bytes",
                    report.request.limits.maximum_scratch_bytes,
                ),
            ):
                quantity = getattr(embedding, name)
                if type(quantity) is not int or not 0 <= quantity <= maximum:
                    raise ValueError(
                        "Historical embedding execution receipt exceeds its actual controls."
                    )
            reconstruct(embedding)
        if coverage is not None:
            if (
                type(coverage) is not _mesh_certificates.DomainCoverageCertificate
                or embedding is None
            ):
                raise TypeError(
                    "Coverage renewal requires its actual embedding and domain proof owners."
                )
            coverage.binding.require(mesh, geometry)
            reconstruct(
                coverage.binding,
                mesh=mesh,
                geometry=geometry,
                limits=report.request.limits,
            )
            for finding in coverage.findings:
                reconstruct(finding)
            if coverage.embedding_certificate_id != embedding.certificate_id:
                raise ValueError(
                    "Coverage changed its actual embedding evidence dependency."
                )
            reconstruct(coverage, domain=report.request.domain)
        for outcome in report.outcomes:
            if type(outcome) is not _certification.MeshCertificationOutcome:
                raise TypeError("Scientific renewal requires exact outcome owners.")
            reconstruct(outcome)
            dependency = {"global_embedding": embedding, "domain_coverage": coverage}.get(
                outcome.check
            )
            if (
                dependency is not None
                and outcome.evidence_id != dependency.certificate_id
            ):
                raise ValueError(
                    "Scientific outcome changed its actual theorem dependency."
                )
        reconstruct(report, mesh=mesh, geometry=geometry, audit=audit)

    def compare_fields(left: Any, right: Any, omitted: frozenset[str]) -> None:
        if type(left) is not type(right):
            raise TypeError("Scientific renewal changes its exact proof owner.")
        for field in fields(left):
            if field.name not in omitted and not equal(
                getattr(left, field.name), getattr(right, field.name)
            ):
                raise ValueError(
                    f"Renewed scientific theorem changed {type(left).__name__}.{field.name}."
                )

    def compare_coverage(left: Any, right: Any) -> None:
        if len(left.premise_certificate_ids) != len(right.premise_certificate_ids):
            raise ValueError(
                "Renewed domain coverage changed its complete premise owner count."
            )
        compare_fields(
            left,
            right,
            frozenset(
                {
                    "embedding_certificate_id",
                    "premise_certificate_ids",
                    "source_expression_work_units",
                    "source_expression_peak_bytes",
                    "certificate_id",
                }
            ),
        )

    compare_fields(
        original,
        renewed,
        frozenset({"embedding", "coverage", "fidelity", "outcomes", "report_id"}),
    )
    if (original.embedding is None) != (renewed.embedding is None) or (
        original.coverage is None
    ) != (renewed.coverage is None):
        raise ValueError("Scientific renewal changes its required theorem owners.")
    if original.embedding is not None:
        compare_fields(
            original.embedding,
            renewed.embedding,
            frozenset(
                {
                    "source_expression_work_units",
                    "source_expression_peak_bytes",
                    "certificate_id",
                }
            ),
        )
    if original.coverage is not None:
        compare_coverage(original.coverage, renewed.coverage)
    if (original.fidelity is None) != (renewed.fidelity is None):
        raise ValueError("Scientific renewal changes its required fidelity owner.")
    if original.fidelity is not None:
        compare_fields(
            original.fidelity,
            renewed.fidelity,
            frozenset({"domain_coverage", "certificate_id"}),
        )
        old_domain = original.fidelity.domain_coverage
        new_domain = renewed.fidelity.domain_coverage
        if (old_domain is None) != (new_domain is None):
            raise ValueError(
                "Scientific fidelity renewal changes its mapped domain proof owner."
            )
        if old_domain is not None:
            compare_coverage(old_domain, new_domain)
    if len(original.outcomes) != len(renewed.outcomes):
        raise ValueError("Scientific renewal changes its complete scheduled outcomes.")
    for old, new in zip(original.outcomes, renewed.outcomes, strict=True):
        omitted = frozenset({"outcome_id"})
        if old.check in ("global_embedding", "domain_coverage", "source_fidelity"):
            omitted |= {"evidence_id"}
        compare_fields(old, new, omitted)
    return original, renewed


def validate_registered_native_part(
    part: Any,
    source: Any,
    specification: Any,
    options: NativeMeshingOptions | None = None,
    /,
    *,
    limits: ArrayArchiveLimits,
) -> tuple[Any, Any]:
    """Bind original family declaration to an actual fresh audit and source theorem."""
    from ..meshing._result import CellMeshingResult
    from ._meshing_sources import (
        _native_authority_equal,
        _validate_association_binding,
        _validate_layer_core_association,
        recertify_restored_meshing_source,
        validate_restored_meshing_source_bindings,
    )

    if type(source) not in native_generation_source_types():
        raise TypeError("Generation source must retain its exact owning native family.")
    volume = type(source) in (
        _native_sources.NativePlcSource,
        _native_sources.NativePolyhedralSource,
        _native_sources.NativeMappedHexSource,
        _native_sources.NativeSweepSource,
        _native_sources.NativeLayerCoreSource,
        _native_sources.NativeSurfaceEnvelopeSource,
        CompartmentMeshingSource,
    )
    if type(specification) not in (
        (_contracts.VolumeMeshingSpec,)
        if volume
        else (_contracts.SurfaceMeshingSpec, _contracts.VolumeMeshingSpec)
    ):
        raise TypeError(
            "Generation hard request must retain its owning native family contract."
        )
    if (
        type(source)
        in (_native_sources.NativePlanarSource, _native_sources.NativeSurfaceSource)
        and type(specification) is not _contracts.SurfaceMeshingSpec
    ):
        raise TypeError("Native surface generation requires SurfaceMeshingSpec.")
    scope = (
        specification.boundary_scope
        if type(specification) is _contracts.VolumeMeshingSpec
        else specification.scope
    )
    if (scope.source_id, scope.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError(
            "Original generation request must bind its actual authored source revision."
        )
    result = part.carrier
    if type(result) is not CellMeshingResult or result.certification is None:
        raise ValueError(
            "Native generation requires the whole source-certified CellMeshingResult."
        )
    if result.compliance.specification_id != specification.specification_id:
        raise ValueError("Registered carrier lost its original hard generation request.")
    if (result.trace.binding.source_id, result.trace.binding.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError(
            "Registered native result must retain its actual generation source revision."
        )
    if options is not None:
        if type(options) is not NativeMeshingOptions:
            raise TypeError(
                "Native generation options require their exact algorithm declaration."
            )
        from ..meshing.providers._native import NativeMeshingProvider

        # This prepares real source-specific controls and rejects unsupported
        # family/target/option combinations. No live plan enters the archive.
        plan = NativeMeshingProvider(options).plan(
            source,
            specification,
            coordinate_contract=result.coordinate_contract,
        )
        provenance = json.loads(result.provenance.content_json)
        if (
            provenance.get("kind") != "mapping"
            or dict(provenance["items"]).get("plan") != plan.plan_id
        ):
            raise ValueError(
                "Native generation options differ from the original source-bound generation plan."
            )
    elif type(source) is not _native_sources.NativePlanarSource:
        raise ValueError(
            "Native family registration must retain its original generation options."
        )
    request = result.certification.request
    validate_restored_meshing_source_bindings(
        request.source, request, result.certification, limits=limits
    )
    for association in result.associations:
        if type(source) is _native_sources.NativeLayerCoreSource:
            _validate_layer_core_association(
                request, association, source, mesh=result.mesh
            )
        else:
            _validate_association_binding(request, association, generation_source=source)
    if type(source) in (
        _native_sources.NativeStructuredSource,
        _native_sources.NativeSweepSource,
    ):
        if not _native_authority_equal(
            request.source, source.fidelity_source, limits=limits
        ):
            raise ValueError(
                "Structured certificate must retain its independent authored boundary query."
            )
        if not _native_authority_equal(request.domain, source.domain, limits=limits):
            raise ValueError(
                "Structured certificate must bind its independently declared represented domain."
            )
    elif type(source) is _native_sources.NativeMappedHexSource:
        if not _native_authority_equal(request.domain, source.domain, limits=limits):
            raise ValueError(
                "Mapped certificate must bind its complete independent source coordinate maps."
            )
        if (
            type(request.source) is not _mesh_certificates.MappedDomainBoundarySource
            or not _native_authority_equal(
                request.source.domain, source.domain, limits=limits
            )
        ):
            raise ValueError(
                "Mapped certificate must retain its actual mapped boundary authority."
            )
    elif (
        type(source) is _native_sources.NativeLayerCoreSource
        and source.mapped_domain is not None
    ):
        from ..meshing._layer_core_association import (
            prepare_native_layer_association_transfer,
        )

        if not _native_authority_equal(
            request.domain, source.mapped_domain, limits=limits
        ):
            raise ValueError(
                "Mapped layer certificate must retain its complete independent source roots."
            )
        fidelity = source.fidelity_source
        if (
            type(fidelity) is not _mesh_certificates.MappedDomainBoundarySource
            or type(request.source) is not _mesh_certificates.MappedDomainBoundarySource
            or not _native_authority_equal(
                request.source.domain, fidelity.domain, limits=limits
            )
            or not _native_authority_equal(
                request.source.limits, fidelity.limits, limits=limits
            )
            or request.source.covering_radius != fidelity.covering_radius
            or not np.array_equal(
                np.asarray(fidelity.cell_regions),
                np.asarray(source.mapped_domain.cell_regions),
            )
            or request.cell_regions is None
            or not np.array_equal(
                np.asarray(request.source.cell_regions),
                np.asarray(request.cell_regions),
            )
        ):
            raise ValueError(
                "Mapped layer certificate must retain its independent original "
                "fidelity source and explicit target-region renewal."
            )
        transfer = prepare_native_layer_association_transfer(
            source, specification, result
        )
        transfer.source_associations(result)
    elif type(source) is _native_sources.NativeLayerCoreSource:
        _require_registered_layer_authority(request, source, limits=limits)
    elif type(source) is _native_sources.NativeSurfaceSource:
        from ..geometry._meshing_domain import MeshingDomainBoundarySource

        if type(
            request.source
        ) is not MeshingDomainBoundarySource or not _native_authority_equal(
            request.source.domain, source.domain, limits=limits
        ):
            raise ValueError(
                "Surface certificate must retain its original complete source chart domain."
            )
    elif type(source) is _native_sources.NativeSurfaceEnvelopeSource:
        _require_registered_envelope(source, specification, result, limits=limits)
    elif type(source) is CompartmentMeshingSource:
        _require_registered_compartment(source, specification, result, limits=limits)
    elif type(source) in (
        _native_sources.NativePlcSource,
        _native_sources.NativePolyhedralSource,
    ):
        if (
            request.domain is None
            or request.domain.source_id != source.source_id
            or request.domain.region_ids != source.complex.region_ids
        ):
            raise ValueError(
                "PLC certificate must bind its actual authored source domain and regions."
            )
        # Independently triangulate only source polygons; a carrier-selected
        # smaller domain is never an acceptance oracle.
        declared = _volume_generation.declared_plc_domain(
            source.complex, source.source_id
        )
        if not _native_authority_equal(request.domain, declared, limits=limits):
            raise ValueError(
                "PLC certificate differs from its independent original source polygon domain."
            )
        if type(source) is _native_sources.NativePlcSource:
            _require_registered_plc_geometry(source, specification, result)
    policy = native_family_audit_policy(source, specification, result.mesh)
    if policy.policy_id != result.audit.policy_id:
        raise ValueError("Native carrier lost its owning source-family audit policy.")
    audit = _audit.audit_cell_mesh(
        result.mesh,
        result.geometry,
        policy=policy,
        boundary=result.boundary,
        patches=result.patches,
        associations=result.associations,
        attributes=result.attributes,
        zones=result.zones,
        labels=result.labels,
    )
    audit.require_passed()
    audit.require_decided()
    if not _native_authority_equal(result.audit, audit, limits=limits):
        raise ValueError(
            "Stored native audit differs from its actual fresh family audit."
        )
    # Authenticate the historical receipt before executing another proof ledger.
    require_same_native_scientific_theorem(
        result.certification,
        result.certification,
        result.mesh,
        result.geometry,
        audit,
        limits=limits,
    )
    fresh = recertify_restored_meshing_source(
        request, result.mesh, result.geometry, audit, archive_limits=limits
    )
    return require_same_native_scientific_theorem(
        result.certification,
        fresh,
        result.mesh,
        result.geometry,
        audit,
        limits=limits,
    )
