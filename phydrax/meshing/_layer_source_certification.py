#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-source acceptance for immutable layer/core composition."""

from __future__ import annotations

import numpy as np

import phydrax.geometry._domain_boundary as domain_boundary
import phydrax.geometry._domain_coverage as _dependency_phydrax_geometry__mesh_certificates
import phydrax.geometry._meshing_domain as _dependency_phydrax_geometry__meshing_domain
import phydrax.geometry._source_boundary_records as source_boundary_records_owner
import phydrax.meshing._certification_schedule as _leaf_phydrax_meshing__certification_schedule
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._layer_core_controls as _dependency_phydrax_meshing__layer_core_controls
import phydrax.meshing._layer_core_resources as _dependency_phydrax_meshing__layer_core_resources
import phydrax.meshing._layer_core_topology as _dependency_phydrax_meshing__layer_core
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing.providers._native_publication as _dependency_phydrax_meshing_providers__native_publication
import phydrax.meshing.providers._native_sources as _dependency_phydrax_meshing_providers__native_sources

from ..typing import checked


@checked
def layer_source_certification(
    source: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    construction: _dependency_phydrax_meshing__layer_core.LayerCoreConstruction,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> _dependency_phydrax_meshing_providers__native_publication.NativeCertificationRequest:
    """Compose source represented coverage and continuous source fidelity.

    A curved source's PLC coverage remains explicitly an approximation-domain
    certificate; continuous fidelity is independently bound to the source
    source query. It is not relabeled as curved-domain coverage.
    """
    original = source.source_domain
    query = source.fidelity_source
    scope = source.source_boundary_scope
    if original is None:
        return _dependency_phydrax_meshing_providers__native_publication.NativeCertificationRequest(
            _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
                "volume_plc"
            ),
            source.source_id,
            source.source_revision,
            specification.limits,
            domain=construction.domain,
            cell_regions=construction.cell_regions,
        )
    if query is None or scope is None:
        raise ValueError(
            "NativeLayerCoreSource.source_domain must provide its full boundary scope and continuous query; got missing or incompatible its full boundary scope and continuous query"
        )
    if (query.source_id, query.source_revision) != (
        original.source_id,
        original.source_revision,
    ):
        raise ValueError(
            "NativeLayerCoreSource.source_domain must bind the declared source geometry revision; got incompatible source fidelity query"
        )
    whole = tuple(
        feature
        for feature in specification.protected_features
        if feature.feature_kind
        is _dependency_phydrax_meshing__controls.FeatureKind.SURFACE
        and feature.scope.scope_id == scope.scope_id
    )
    size = specification.size_controls[0]
    match size:
        case _dependency_phydrax_meshing__sizing.UniformSizeControl():
            pass
        case _:
            raise ValueError(
                "NativeLayerCoreSource.source_domain must declare its whole-source resolution bound; got incompatible layer/core request"
            )
    # The size request supplies an explicit geometric resolution bound when
    # no tighter whole-boundary deviation was requested. The consumer reports
    # this resolution semantics rather than inventing a zero-distance claim.
    tolerance = min(
        (feature.maximum_deviation for feature in whole), default=size.target_size
    )
    match original:
        case (
            _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain() as original
        ):
            indices = {
                identifier: index for index, identifier in enumerate(original.region_ids)
            }
            if any(identifier not in indices for identifier in source.region_ids):
                raise ValueError(
                    "NativeLayerCoreSource.source_domain must include every declared layer/core material identity; got incompatible Source represented regions"
                )
            regions = np.asarray(
                [
                    indices[source.region_ids[int(index)]]
                    for index in construction.cell_regions
                ],
                dtype=np.int64,
            )
            domain = original
        case _dependency_phydrax_geometry__meshing_domain.MeshingDomain() as original:
            regions = construction.cell_regions
            domain = construction.domain
        case _:
            raise TypeError(
                "NativeLayerCoreSource.source_domain must be MeshingDomain or PiecewiseLinearDomain; got incompatible Source layer geometry"
            )
    return _dependency_phydrax_meshing_providers__native_publication.NativeCertificationRequest(
        _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
            "volume_layer"
        ),
        original.source_id,
        original.source_revision,
        specification.limits,
        domain=domain,
        cell_regions=regions,
        fidelity_source=query,
        fidelity_tolerance=tolerance,
        scoped_fidelity=layer_source_fidelity_requests(
            source, specification, construction, work=work
        ),
    )


@checked
def layer_source_fidelity_requests(
    source: _dependency_phydrax_meshing_providers__native_sources.NativeLayerCoreSource,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    construction: _dependency_phydrax_meshing__layer_core.LayerCoreConstruction,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> tuple[
    tuple[
        source_boundary_records_owner.SourceBoundaryQuery,
        np.ndarray,
        float,
    ],
    ...,
]:
    """Bind each source face to its full authored target trace and query."""
    original, query, scope = (
        source.source_domain,
        source.fidelity_source,
        source.source_boundary_scope,
    )
    if original is None:
        return ()
    match (original, query):
        case (
            _dependency_phydrax_geometry__meshing_domain.MeshingDomain() as original,
            domain_boundary.MeshingDomainBoundarySource() as query,
        ) if scope is not None:
            pass
        case _:
            if any(
                feature.scope.scope_id != specification.boundary_scope.scope_id
                or feature.feature_kind
                is _dependency_phydrax_meshing__controls.FeatureKind.MATERIAL_INTERFACE
                for feature in specification.protected_features
            ):
                raise ValueError(
                    "NativeLayerCoreSource.fidelity_source must provide a parametric face query for scoped features; got missing or incompatible face support"
                )
            return ()
    if query.domain.domain_id != original.domain_id:
        raise ValueError(
            "NativeLayerCoreSource.fidelity_source must retain the realized source parametric domain; got incompatible Scoped fidelity"
        )
    authored = (
        _dependency_phydrax_meshing__layer_core_controls.layer_source_face_ancestry(
            source, work=work
        )
    )
    mesh = construction.mesh
    work.charge(mesh.coordinates.shape[0] + mesh.entity_set(2).count)
    face_ids = dict(
        zip(
            _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                mesh, 2
            ),
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    targets: dict[int, list[int]] = {}
    for key, owner in authored.items():
        work.charge(1)
        if key not in face_ids:
            raise ValueError(
                "NativeLayerCoreSource.fidelity_source must retain source patch identity and the requested combined-volume face trace; got an authored source face is absent from the combined volume trace"
            )
        targets.setdefault(owner, []).append(face_ids[key])
    patches = dict(
        zip(
            np.asarray(original.scope_indices(2), dtype=np.int64).tolist(),
            range(len(original.patches)),
            strict=True,
        )
    )
    size = specification.size_controls[0]
    match size:
        case _dependency_phydrax_meshing__sizing.UniformSizeControl():
            pass
        case _:
            raise ValueError(
                "NativeLayerCoreSource.fidelity_source must provide its declared whole-source resolution; got missing or incompatible its declared whole-source resolution"
            )
    whole = tuple(
        feature.maximum_deviation
        for feature in specification.protected_features
        if feature.feature_kind
        is _dependency_phydrax_meshing__controls.FeatureKind.SURFACE
        and feature.scope.scope_id == scope.scope_id
    )
    default = min(whole, default=size.target_size)
    requests: list[
        tuple[
            source_boundary_records_owner.SourceBoundaryQuery,
            np.ndarray,
            float,
        ]
    ] = []
    for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist():
        work.charge(1)
        if identifier not in patches or identifier not in targets:
            raise ValueError(
                "NativeLayerCoreSource.fidelity_source must retain source patch identity and the requested combined-volume face trace; got every source requested face needs its realized source patch and full target trace"
            )
        patch = patches[identifier]
        tolerances = [
            feature.maximum_deviation
            for feature in specification.protected_features
            if identifier in np.asarray(feature.scope.entity_ids)
        ]
        tolerance = min((default, *tolerances))
        charts = tuple(
            record for record in query.chart_triangulations if record[0] == patch
        )
        scoped = domain_boundary.MeshingDomainBoundarySource(
            original, (patch,), resolution=query.resolution, chart_triangulations=charts
        )
        identifiers = np.asarray(sorted(set(targets[identifier])), dtype=np.int64)
        identifiers.setflags(write=False)
        requests.append((scoped, identifiers, tolerance))
    return tuple(requests)


__all__ = ["layer_source_certification"]
