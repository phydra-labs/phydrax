#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Original-source acceptance for immutable layer/core composition."""

from __future__ import annotations

import numpy as np

from ..geometry._mesh_certificates import PiecewiseLinearDomain, SourceBoundaryQuery
from ..geometry._meshing_domain import MeshingDomain, MeshingDomainBoundarySource
from ._certification import MeshCertificationSchedule
from ._contracts import VolumeMeshingSpec
from ._controls import FeatureKind
from ._layer_core import LayerCoreConstruction
from ._layer_core_controls import layer_source_face_ancestry
from ._layer_core_resources import _row_entity_vertex_keys, LayerCoreSourceWork
from ._sizing import UniformSizeControl
from .providers._native_publication import NativeCertificationRequest
from .providers._native_sources import NativeLayerCoreSource


def layer_source_certification(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    construction: LayerCoreConstruction,
    /,
    *,
    work: LayerCoreSourceWork,
) -> NativeCertificationRequest:
    """Compose original represented coverage and continuous original fidelity.

    A curved source's PLC coverage remains explicitly an approximation-domain
    certificate; continuous fidelity is independently bound to the original
    source query. It is not relabeled as exact curved-domain coverage.
    """
    original = source.source_domain
    query = source.fidelity_source
    scope = source.source_boundary_scope
    if original is None:
        return NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source.source_id,
            source.source_revision,
            specification.limits,
            domain=construction.domain,
            cell_regions=construction.cell_regions,
        )
    if query is None or scope is None:
        raise ValueError(
            "Original layer geometry requires its complete boundary scope and continuous query."
        )
    if (query.source_id, query.source_revision) != (
        original.source_id,
        original.source_revision,
    ):
        raise ValueError(
            "The original fidelity query must bind the declared original geometry revision."
        )
    whole = tuple(
        feature
        for feature in specification.protected_features
        if feature.feature_kind is FeatureKind.SURFACE
        and feature.scope.scope_id == scope.scope_id
    )
    size = specification.size_controls[0]
    if not isinstance(size, UniformSizeControl):
        raise ValueError(
            "A layer/core request must declare its whole-source resolution bound."
        )
    # The size request supplies an explicit geometric resolution bound when
    # no tighter whole-boundary deviation was requested. The consumer reports
    # this resolution semantics rather than inventing a zero-distance claim.
    tolerance = min(
        (feature.maximum_deviation for feature in whole), default=size.target_size
    )
    if isinstance(original, PiecewiseLinearDomain):
        indices = {
            identifier: index for index, identifier in enumerate(original.region_ids)
        }
        if any(identifier not in indices for identifier in source.region_ids):
            raise ValueError(
                "Original represented regions must include every declared layer/core material identity."
            )
        regions = np.asarray(
            [
                indices[source.region_ids[int(index)]]
                for index in construction.cell_regions
            ],
            dtype=np.int64,
        )
        domain = original
    elif isinstance(original, MeshingDomain):
        regions = construction.cell_regions
        domain = construction.domain
    else:
        raise TypeError(
            "Original layer geometry must be MeshingDomain or PiecewiseLinearDomain."
        )
    return NativeCertificationRequest(
        MeshCertificationSchedule("volume_layer"),
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


def layer_source_fidelity_requests(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    construction: LayerCoreConstruction,
    /,
    *,
    work: LayerCoreSourceWork,
) -> tuple[tuple[SourceBoundaryQuery, np.ndarray, float], ...]:
    """Bind each original face to its complete authored target trace and query."""
    original, query, scope = (
        source.source_domain,
        source.fidelity_source,
        source.source_boundary_scope,
    )
    if original is None:
        return ()
    if (
        not isinstance(original, MeshingDomain)
        or not isinstance(query, MeshingDomainBoundarySource)
        or scope is None
    ):
        if any(
            feature.scope.scope_id != specification.boundary_scope.scope_id
            or feature.feature_kind is FeatureKind.MATERIAL_INTERFACE
            for feature in specification.protected_features
        ):
            raise ValueError(
                "Scoped original-source fidelity requires its nominal parametric face query."
            )
        return ()
    if query.domain.domain_id != original.domain_id:
        raise ValueError(
            "Scoped fidelity must retain the actual original parametric domain."
        )
    authored = layer_source_face_ancestry(source, work=work)
    mesh = construction.mesh
    face_ids = dict(
        zip(
            _row_entity_vertex_keys(mesh, 2),
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    targets: dict[int, list[int]] = {}
    for key, owner in authored.items():
        work.charge(1)
        if key not in face_ids:
            raise ValueError(
                "An authored original face is absent from the combined volume trace."
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
    if not isinstance(size, UniformSizeControl):
        raise ValueError(
            "Scoped layer fidelity requires its declared whole-source resolution."
        )
    whole = tuple(
        feature.maximum_deviation
        for feature in specification.protected_features
        if feature.feature_kind is FeatureKind.SURFACE
        and feature.scope.scope_id == scope.scope_id
    )
    default = min(whole, default=size.target_size)
    requests: list[tuple[SourceBoundaryQuery, np.ndarray, float]] = []
    for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist():
        work.charge(1)
        if identifier not in patches or identifier not in targets:
            raise ValueError(
                "Every original requested face needs its actual source patch and complete target trace."
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
        scoped = MeshingDomainBoundarySource(
            original, (patch,), resolution=query.resolution, chart_triangulations=charts
        )
        identifiers = np.asarray(sorted(set(targets[identifier])), dtype=np.int64)
        identifiers.setflags(write=False)
        requests.append((scoped, identifiers, tolerance))
    return tuple(requests)


__all__ = ["layer_source_certification"]
