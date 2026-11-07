#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Input-only reproof of an original mapped layer/core source."""

from __future__ import annotations

import numpy as np

from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._surface_source_archive import _mapped_reference
from ._layer_core_mapped import validate_layer_reference_columns
from ._layer_core_periodic import prepare_core_periodic
from ._layer_core_topology import prepare_identity
from .providers._native_sources import (
    _mapped_layer_core_facet_relation,
    NativeLayerCoreSource,
    NativeMappedHexSource,
    NativePlcSource,
)


def validate_mapped_layer_core_source(source: NativeLayerCoreSource, /) -> None:
    """Reprove the original reference, mapped root, and full PLC source bindings.

    All arguments are restored authoring inputs, not a generated mesh or a
    certification cache. The root geometry is reconstructed by its registered
    coordinate-element and source-map constructors before it is used as proof.
    """
    if type(source) is not NativeLayerCoreSource:
        raise TypeError("Expected the canonical mapped layer/core source.")
    reference, domain, query = (
        source.reference_source,
        source.mapped_domain,
        source.fidelity_source,
    )
    if (
        type(reference) is not NativeLayerCoreSource
        or reference.reference_source is not None
        or reference.mapped_domain is not None
        or domain is None
        or type(query) is not MappedReferenceBoundarySource
        or not isinstance(query.authority, (NativePlcSource, NativeMappedHexSource))
        or type(query.authority) not in (NativePlcSource, NativeMappedHexSource)
    ):
        raise ValueError(
            "Mapped layer/core reproof requires its original reference, root and full source query."
        )
    if (source.source_id, source.source_revision) != (
        reference.source_id,
        reference.source_revision,
    ) or (source.source_id, source.source_revision) != (
        domain.source_id,
        domain.source_revision,
    ):
        raise ValueError("Mapped layer/core and reference source identities differ.")

    # Rebuild every root cell's exact coordinate program, not a sampled map.
    restored_domain = _mapped_reference(domain)
    if restored_domain.domain_id != domain.domain_id:
        raise ValueError("Mapped layer/core root identity changed.")
    reference.validate_restored()
    source.validate_restored()
    query.require_current()

    for name in (
        "vertex_layer_ids",
        "cap_polygon_ids",
        "layer_regions",
        "region_entity_ids",
        "core_region_map",
        "core_vertex_representatives",
        "core_vertex_shifts",
        "core_seam_polygon_pairs",
    ):
        left, right = getattr(source, name), getattr(reference, name)
        if (left is None) != (right is None) or (
            left is not None and not np.array_equal(left, right)
        ):
            raise ValueError(f"Mapped layer/core reference {name} changed.")
    for name in (
        "region_ids",
        "region_occurrence_paths",
        "core_facet_source_ids",
        "core_facet_source_occurrence_paths",
        "cap_refinement",
        "noncap_refinement",
    ):
        if getattr(source, name) != getattr(reference, name):
            raise ValueError(f"Mapped layer/core reference {name} changed.")
    if domain.region_ids != source.region_ids:
        raise ValueError("Mapped root and layer/core material identities differ.")
    for name in (
        "vertex_ids",
        "vertex_occurrence_paths",
        "vertex_entity_ids",
        "segment_ids",
        "segment_occurrence_paths",
        "segment_entity_ids",
        "facet_ids",
        "facet_occurrence_paths",
        "facet_entity_ids",
        "region_ids",
        "region_occurrence_paths",
        "region_entity_ids",
        "plc_edge_vertices",
        "plc_edge_facets",
        "plc_edge_ids",
        "plc_edge_occurrence_paths",
        "plc_edge_entity_ids",
        "boundary",
    ):
        if getattr(source.complex, name) != getattr(reference.complex, name):
            raise ValueError(f"Mapped core/reference PLC {name} changed.")
    for name in (
        "polygon_offsets",
        "polygon_vertices",
        "polygon_facets",
        "facet_regions",
        "segments",
    ):
        if not np.array_equal(
            getattr(source.complex, name), getattr(reference.complex, name)
        ):
            raise ValueError(f"Mapped core/reference PLC {name} changed.")

    validate_layer_reference_columns(source, reference, restored_domain)
    # Reconstruct the independently declared combined layer/core domain and
    # prove each non-internal core polygon's original full-PLC identity/sides.
    _mapped_layer_core_facet_relation(source, query.authority)
    mapping, _, polygons, points = prepare_identity(
        reference.layers,
        reference.complex,
        reference.vertex_layer_ids,
        reference.cap_polygon_ids,
    )
    prepare_core_periodic(reference, mapping, polygons, points)
