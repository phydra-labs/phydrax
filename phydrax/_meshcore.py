#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host binding of the native Phydrax meshcore library.

The optional ``phydrax-meshcore`` distribution ships a C++ shared library with a
plain C ABI (``native/meshcore/include/phydrax_meshcore.h``) that is loaded with
``ctypes``.  It owns exact geometric predicates (Shewchuk adaptive expansion
arithmetic with index-ordered symbolic perturbation), exactly classified convex
clipping moments, and exact-predicate Delaunay/regular/constrained
triangulations.  Every function here is host-only NumPy interchange: inputs are
validated and converted to contiguous float64/int32 arrays, results are fresh
arrays, and native statuses reach the caller (per-item status arrays for batched
clipping, exceptions for rejected calls).

Library lookup order: the native ``ProviderConfig.library_paths`` override,
then ``PHYDRAX_MESHCORE_LIBRARY``, then ``phydrax_meshcore.library_path()``. A
library is usable only when it exports every C ABI symbol bound here and its
version equals the installed ``phydrax`` release; anything else is reported as
:class:`MeshcoreUnavailableError` with the reason.
"""

from __future__ import annotations

import ctypes
import functools
import importlib
import importlib.metadata
import importlib.util
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, final, NamedTuple

import numpy as np

from ._config_environment import LIBRARY_ENVIRONMENT
from ._config_state import library_input
from ._validation import canonical_identifier, finite_real_scalar


class MeshcoreUnavailableError(ImportError):
    """The native meshcore library is not installed or cannot be loaded."""


class MeshcoreError(RuntimeError):
    """The native meshcore library refused a call for resource or internal reasons."""

    def __init__(self, status: MeshcoreStatus, message: str, /) -> None:
        super().__init__(f"{message} (meshcore status {status.name})")
        self.status = status


class MeshcoreStatus(IntEnum):
    """Call and item status codes of the meshcore C ABI."""

    OK = 0
    INVALID_ARGUMENT = 1
    NONFINITE_INPUT = 2
    RANGE_ERROR = 3
    DEGENERATE_INPUT = 4
    INVALID_INPUT = 5
    CAPACITY_EXCEEDED = 6
    CONSTRAINT_INTERSECTION = 7
    UNRESOLVED = 8
    INTERNAL_ERROR = 9
    ALLOCATION_FAILURE = 10
    INDEX_RANGE_ERROR = 11


class QuadraticValidityDecision(IntEnum):
    """Per-cell continuous quadratic-map validity decision."""

    VALID = 0
    INVALID = 1
    UNRESOLVED = 2


class IntersectionClass(IntEnum):
    """Exact contact class of two native triangle or segment features."""

    DISJOINT = 0
    SHARED_VERTEX = 1
    SHARED_EDGE = 2
    TOUCHING = 3
    CROSSING = 4
    COPLANAR_OVERLAP = 5
    COINCIDENT = 6


class MixedElementKind(IntEnum):
    TETRAHEDRON = 0
    HEXAHEDRON = 1
    PRISM = 2
    PYRAMID = 3
    QUADRILATERAL = 4


class MixedTemplateCounts(NamedTuple):
    template_count: int
    child_count: int
    corner_count: int


class PowerCellFailureReason(IntEnum):
    """Scientific failure location category for native restricted power cells."""

    NONE = 0
    DEGENERATE_VERTEX = 1
    MISSING_PIECE = 2
    INCONSISTENT_FACE = 3
    INVALID_DOMAIN = 4
    CLIP_FAILURE = 5
    SEED_FAILURE = 6
    IMAGE_SOURCE_DOMAIN = 7
    OUTPUT_COUNT_OVERFLOW = 8


class PlcBoundaryPolicy(IntEnum):
    FIXED = 0
    CONFORMING = 1


class PlcFailureReason(IntEnum):
    NONE = 0
    DUPLICATE_POINT = 1
    INVALID_POLYGON = 2
    INVALID_FACET = 3
    INTERSECTING = 4
    OPEN_BOUNDARY = 5
    INCONSISTENT_REGIONS = 6
    REGION_LEAK = 7
    FIXED_SEGMENT = 8
    FIXED_FACET = 9
    NONREPRESENTABLE = 10
    INVALID_SEED = 14
    OUTSIDE_DOMAIN = 15
    SOURCE_DEVIATION = 17
    STEINER_SUPPRESSION = 18


class PlcEntityKind(IntEnum):
    NONE = 0
    POINT = 1
    POLYGON = 2
    FACET = 3
    EDGE = 4
    REGION = 5
    SEED = 6


class PlcSourceFailure(MeshcoreError):
    """Structured source-preparation refusal with original PLC entities.

    ``suppression_search`` carries the Steiner-suppression evidence up to the
    refusal (for a suppression refusal: the exhausted merged regions).
    """

    def __init__(
        self,
        status: MeshcoreStatus,
        reason: PlcFailureReason,
        first_kind: PlcEntityKind,
        first: int,
        second_kind: PlcEntityKind,
        second: int,
        certified_deviation: float,
        declared_bound: float,
        /,
        *,
        suppression_search: "PlcSuppressionSearch | None" = None,
    ) -> None:
        super().__init__(
            status,
            f"PLC source preparation failed ({reason.name}; "
            f"first={first_kind.name}:{first}; second={second_kind.name}:{second})",
        )
        self.reason = reason
        self.first_kind = first_kind
        self.first = first
        self.second_kind = second_kind
        self.second = second
        self.certified_deviation = certified_deviation
        self.declared_bound = declared_bound
        self.suppression_search = suppression_search


class RestrictedPowerFailure(MeshcoreError):
    """Scientific refusal of restricted power-cell construction."""

    def __init__(
        self,
        status: MeshcoreStatus,
        reason: PowerCellFailureReason,
        site: int,
        tet: int,
        clip_status: MeshcoreStatus,
        /,
        *,
        failed_image_coordinate: int = -1,
        failed_component: int = -1,
    ) -> None:
        super().__init__(
            status,
            "restricted power-cell construction failed "
            f"({reason.name}; site={site}; tetrahedron={tet}; clip={clip_status.name})",
        )
        self.reason = reason
        self.site = site
        self.tet = tet
        self.clip_status = clip_status
        self.failed_image_coordinate = failed_image_coordinate
        self.failed_component = failed_component


_P = ctypes.c_void_p
_I32 = ctypes.c_int32
_I64 = ctypes.c_int64
_U64 = ctypes.c_uint64
_F64 = ctypes.c_double
_ParallelChunkCallback = ctypes.CFUNCTYPE(_I32, _P, _I64, _I64, _I64)
PHX_MC_TET_MESH_SNAPSHOT_SIZES = 19


class _RestrictedPowerCellsRequest(ctypes.Structure):
    _fields_ = (
        ("site_count", _I64),
        ("sites", _P),
        ("weights", _P),
        ("neighbor_offsets", _P),
        ("neighbors", _P),
        ("split_sites", _P),
        ("point_count", _I64),
        ("points", _P),
        ("tet_count", _I64),
        ("tets", _P),
        ("tet_regions", _P),
        ("tet_face_facets", _P),
        ("result", _P),
    )


class _RestrictedPowerCellsExactRequest(ctypes.Structure):
    _fields_ = (
        ("original_site_count", _I64),
        ("original_sites", _P),
        ("original_weights", _P),
        ("image_count", _I64),
        ("image_site_owners", _P),
        ("image_coordinate_offsets", _P),
        ("component_count", _I64),
        ("image_coordinate_components", _P),
        ("neighbor_offsets", _P),
        ("neighbors", _P),
        ("split_sites", _P),
        ("point_count", _I64),
        ("points", _P),
        ("tet_count", _I64),
        ("tets", _P),
        ("tet_regions", _P),
        ("tet_face_facets", _P),
        ("result", _P),
        ("failed_image_coordinate", _P),
        ("failed_component", _P),
    )


class _ReferenceTetrahedronClipRequest(ctypes.Structure):
    _fields_ = (
        ("words", _P),
        ("word_count", _I64),
        ("offsets", _P),
        ("signs", _P),
        ("plane_count", _I32),
        ("result", _P),
    )


class _ArrangeTetrahedraExactRequest(ctypes.Structure):
    _fields_ = (
        ("point_count", _I64),
        ("plane_count", _I64),
        ("tetrahedron_count", _I64),
        ("job_count", _I64),
        ("cut_count", _I64),
        ("point_word_count", _I64),
        ("plane_word_count", _I64),
        ("point_words", _P),
        ("point_offsets", _P),
        ("point_signs", _P),
        ("plane_words", _P),
        ("plane_offsets", _P),
        ("plane_signs", _P),
        ("tetrahedra", _P),
        ("tetrahedron_planes", _P),
        ("parents", _P),
        ("cut_offsets", _P),
        ("cut_planes", _P),
        ("result", _P),
    )


class _PeriodicRefinementInput(ctypes.Structure):
    _fields_ = (
        ("point_count", _I64),
        ("points", _P),
        ("lattice", _P),
        ("inverse", _P),
        ("cell_count", _I64),
        ("vertices", _P),
        ("shifts", _P),
        ("control_count", _I64),
        ("controls", _P),
        ("statistic_count", _I64),
        ("quantiles", _P),
        ("absolute_tolerance", _F64),
        ("relative_tolerance", _F64),
        ("protected_edge_count", _I64),
        ("protected_edges", _P),
    )


class _PeriodicLocalDelaunayRequest(ctypes.Structure):
    _fields_ = (
        ("dimension", _I32),
        ("point_count", _I64),
        ("points", _P),
        ("fractional", _P),
        ("point_global_ids", _P),
        ("point_owner_ranks", _P),
        ("local_owner_rank", _I32),
        ("lattice", _P),
        ("inverse", _P),
        ("initial_margin", _F64),
        ("result", _P),
    )


class _PeriodicSupportQueryRequest(ctypes.Structure):
    _fields_ = (
        ("dimension", _I32),
        ("cell_count", _I64),
        ("cell_vertex_points", _P),
        ("cell_vertex_global_ids", _P),
        ("cell_vertex_shifts", _P),
        ("point_count", _I64),
        ("points", _P),
        ("point_global_ids", _P),
        ("lattice", _P),
        ("candidate_revision", _U64),
        ("result", _P),
    )


class _SurfaceReconnectRequest(ctypes.Structure):
    _fields_ = (
        ("vertex_count", _I64),
        ("charts", _P),
        ("points", _P),
        ("normals", _P),
        ("vertex_metrics", _P),
        ("vertex_pole_ids", _P),
        ("triangle_count", _I64),
        ("triangles", _P),
        ("constrained", _P),
        ("insert_count", _I64),
        ("insert_charts", _P),
        ("insert_points", _P),
        ("insert_normals", _P),
        ("insert_metrics", _P),
        ("insert_pole_ids", _P),
        ("insert_spacing", _P),
        ("insert_hints", _P),
        ("insert_edges", _P),
        ("sweep", _I32),
        ("triangle_capacity", _I64),
        ("triangles_out", _P),
        ("constrained_out", _P),
        ("triangle_count_out", _P),
        ("vertex_ids", _P),
        ("item_status", _P),
    )


class _SurfaceEnvelopeCreateRequest(ctypes.Structure):
    _fields_ = (
        ("vertices", _P),
        ("vertex_count", _I64),
        ("faces", _P),
        ("face_count", _I64),
        ("offset", _F64),
        ("spacing", _F64),
        ("certificate_spacing", _F64),
        ("result", _P),
        ("bounds", _P),
    )


class _LevelSetSplit3DRequest(ctypes.Structure):
    _fields_ = (
        ("point_count", _I64),
        ("points", _P),
        ("values", _P),
        ("cell_count", _I64),
        ("cells", _P),
        ("face_count", _I64),
        ("faces", _P),
        ("edge_count", _I64),
        ("edges", _P),
        ("protected_edges", _P),
        ("output_mode", _I32),
        ("out_point_capacity", _I64),
        ("out_cell_capacity", _I64),
        ("out_points", _P),
        ("out_cells", _P),
        ("out_parents", _P),
        ("out_sources", _P),
        ("out_weights", _P),
        ("out_deviations", _P),
        ("out_point_count", _P),
        ("out_cell_count", _P),
        ("failed_edge", _P),
    )


class _Plc3DPrepareSourceRequest(ctypes.Structure):
    _fields_ = (
        ("point_count", _I64),
        ("points", _P),
        ("polygon_count", _I64),
        ("polygon_vertex_count", _I64),
        ("polygon_offsets", _P),
        ("polygon_vertices", _P),
        ("polygon_facets", _P),
        ("facet_count", _I64),
        ("facet_regions", _P),
        ("segment_count", _I64),
        ("segments", _P),
        ("seed_count", _I64),
        ("seeds", _P),
        ("seed_regions", _P),
        ("boundary_policy", _I32),
        ("facet_tolerances", _P),
        ("segment_tolerances", _P),
        ("facet_planes", _P),
        ("facet_plane_tolerances", _P),
        ("result", _P),
    )


class _TetPeriodicVertexMember(ctypes.Structure):
    _fields_ = (
        ("vertex_id", _U64),
        ("winding", _I64 * 3),
        ("representative_to_member", _F64 * 12),
    )


class _TetPeriodicOrbit(ctypes.Structure):
    _fields_ = (
        ("id", _U64),
        ("representative_vertex_id", _U64),
        ("member_offset", _I64),
        ("member_count", _I64),
    )


class _TetPeriodicQuotientCell(ctypes.Structure):
    _fields_ = (
        ("orbit_id", _U64),
        ("id", _U64),
        ("representative_vertices", _U64 * 4),
        ("winding", _I64 * 12),
        ("region", _I32),
        ("lifted_cell_offset", _I64),
        ("lifted_cell_count", _I64),
    )


class _TetPeriodicEntityOrbitMap(ctypes.Structure):
    _fields_ = (
        ("dimension", _I32),
        ("arity", _I32),
        ("rank", _I32),
        ("lifted_offset", _I64),
        ("lifted_count", _I64),
        ("quotient_offset", _I64),
        ("quotient_count", _I64),
        ("winding_offset", _I64),
        ("corner_permutation_offset", _I64),
        ("representative_offset", _I64),
    )


class _TetPeriodicIdentityMapView(ctypes.Structure):
    _fields_ = (
        ("identity_bound", _I32),
        ("source_mesh_id", _P),
        ("source_mesh_id_length", _I64),
        ("source_topology_id", _P),
        ("source_topology_id_length", _I64),
        ("lifted_topology_id", _P),
        ("lifted_topology_id_length", _I64),
        ("periodic_topology_id", _P),
        ("periodic_topology_id_length", _I64),
        ("cell_entity_set_id", _P),
        ("cell_entity_set_id_length", _I64),
        ("edge_entity_set_id", _P),
        ("edge_entity_set_id_length", _I64),
        ("entity_map_count", _I64),
        ("entity_maps", _P),
        ("entity_lifted_count", _I64),
        ("entity_lifted_ids", _P),
        ("entity_lifted_to_quotient", _P),
        ("entity_orientation_signs", _P),
        ("entity_quotient_count", _I64),
        ("entity_quotient_ids", _P),
        ("entity_winding_count", _I64),
        ("entity_winding_shifts", _P),
        ("entity_corner_count", _I64),
        ("entity_corner_permutations", _P),
        ("entity_representative_count", _I64),
        ("entity_representative_rows", _P),
    )


class _TetPeriodicStateView(ctypes.Structure):
    _fields_ = (
        ("identity_map", _TetPeriodicIdentityMapView),
        ("periodic_orbit_count", _I64),
        ("periodic_orbits", _P),
        ("periodic_member_count", _I64),
        ("periodic_members", _P),
        ("periodic_quotient_cell_count", _I64),
        ("periodic_quotient_cells", _P),
        ("periodic_quotient_lifted_count", _I64),
        ("periodic_quotient_lifted_cell_ids", _P),
    )


class _TetPeriodicEditView(ctypes.Structure):
    _fields_ = (
        ("periodic_member_count", _I64),
        ("periodic_members", _P),
        ("removed_periodic_orbit_count", _I64),
        ("removed_periodic_orbit_ids", _P),
        ("replacement_periodic_orbit_count", _I64),
        ("replacement_periodic_orbits", _P),
        ("replacement_periodic_member_count", _I64),
        ("replacement_periodic_members", _P),
        ("removed_quotient_cell_count", _I64),
        ("removed_quotient_cell_ids", _P),
        ("replacement_quotient_cell_count", _I64),
        ("replacement_quotient_cells", _P),
        ("replacement_quotient_lifted_count", _I64),
        ("replacement_quotient_lifted_cell_ids", _P),
        ("target_identity_map", _P),
    )


class _TetPlaneRefusal(ctypes.Structure):
    _fields_ = (
        ("refused", _I32),
        ("face_row", _I32),
        ("vertex", _I64),
        ("parameters", ctypes.c_double * 2),
        ("deviation", ctypes.c_double),
        ("tolerance", ctypes.c_double),
    )


class _TetMeshCreateRequest(ctypes.Structure):
    _fields_ = (
        ("point_count", _I64),
        ("points", _P),
        ("vertex_ids", _P),
        ("tetrahedron_count", _I64),
        ("tetrahedra", _P),
        ("tetrahedron_regions", _P),
        ("cell_ids", _P),
        ("face_count", _I64),
        ("faces", _P),
        ("face_sources", _P),
        ("face_orientations", _P),
        ("facet_ids", _P),
        ("segment_count", _I64),
        ("segments", _P),
        ("segment_sources", _P),
        ("segment_ids", _P),
        ("protection_radii", _P),
        ("protected_vertices", _P),
        ("source_point_count", _I64),
        ("source_points", _P),
        ("source_face_count", _I64),
        ("source_faces", _P),
        ("source_face_groups", _P),
        ("source_face_ids", _P),
        ("source_face_tolerances", _P),
        ("source_segment_count", _I64),
        ("source_segments", _P),
        ("source_segment_groups", _P),
        ("source_segment_ids", _P),
        ("source_segment_tolerances", _P),
        ("source_face_planes", _P),
        ("source_face_plane_tolerances", _P),
        ("witness_strata", _P),
        ("witness_entities", _P),
        ("witness_parameters", _P),
        ("witness_deviations", _P),
        ("boundary_policy", _I32),
        ("identity_bound", _I32),
        ("source_id", _P),
        ("source_id_length", _I64),
        ("source_revision", _P),
        ("source_revision_length", _I64),
        ("source_map_id", _P),
        ("source_map_id_length", _I64),
        ("frame_id", _P),
        ("frame_id_length", _I64),
        ("periodic_state", _P),
        ("plane_refusal", _P),
        ("result", _P),
    )


class _TetMeshSnapshotRequest(ctypes.Structure):
    _fields_ = (
        ("vertex_capacity", _I64),
        ("cell_capacity", _I64),
        ("facet_capacity", _I64),
        ("segment_capacity", _I64),
        ("source_point_capacity", _I64),
        ("source_face_capacity", _I64),
        ("source_segment_capacity", _I64),
        ("periodic_orbit_capacity", _I64),
        ("periodic_member_capacity", _I64),
        ("periodic_quotient_cell_capacity", _I64),
        ("periodic_quotient_lifted_capacity", _I64),
        ("periodic_entity_map_capacity", _I64),
        ("periodic_entity_lifted_capacity", _I64),
        ("periodic_entity_quotient_capacity", _I64),
        ("periodic_entity_winding_capacity", _I64),
        ("periodic_entity_corner_capacity", _I64),
        ("periodic_entity_representative_capacity", _I64),
        ("vertex_ancestry_capacity", _I64),
        ("cell_ancestor_capacity", _I64),
        ("vertex_ids", _P),
        ("points", _P),
        ("witness_strata", _P),
        ("witness_entities", _P),
        ("witness_parameters", _P),
        ("witness_deviations", _P),
        ("vertex_dimensions", _P),
        ("protection_radii", _P),
        ("protected_vertices", _P),
        ("vertex_edge_witness_present", _P),
        ("vertex_edge_witness_endpoints", _P),
        ("vertex_edge_parameters", _P),
        ("vertex_ancestry_offsets", _P),
        ("vertex_ancestor_ids", _P),
        ("vertex_ancestor_winding", _P),
        ("vertex_ancestry_weights", _P),
        ("periodic_orbits", _P),
        ("periodic_members", _P),
        ("periodic_quotient_cells", _P),
        ("periodic_quotient_lifted_cell_ids", _P),
        ("periodic_entity_maps", _P),
        ("periodic_entity_lifted_ids", _P),
        ("periodic_entity_quotient_ids", _P),
        ("periodic_entity_lifted_to_quotient", _P),
        ("periodic_entity_orientation_signs", _P),
        ("periodic_entity_winding_shifts", _P),
        ("periodic_entity_corner_permutations", _P),
        ("periodic_entity_representative_rows", _P),
        ("cell_ancestry_offsets", _P),
        ("cell_ancestor_ids", _P),
        ("cell_ancestry_edit_kinds", _P),
        ("cell_ids", _P),
        ("cell_vertices", _P),
        ("cell_regions", _P),
        ("facet_ids", _P),
        ("facet_vertices", _P),
        ("facet_sources", _P),
        ("facet_orientations", _P),
        ("protected_facets", _P),
        ("segment_ids", _P),
        ("segment_vertices", _P),
        ("segment_sources", _P),
        ("protected_segments", _P),
        ("source_points", _P),
        ("source_faces", _P),
        ("source_face_groups", _P),
        ("source_face_ids", _P),
        ("source_face_tolerances", _P),
        ("source_segments", _P),
        ("source_segment_groups", _P),
        ("source_segment_tolerances", _P),
        ("source_segment_ids", _P),
        ("source_face_planes", _P),
        ("source_face_plane_tolerances", _P),
        ("generation", _P),
    )


class _TetMeshEditRequest(ctypes.Structure):
    _fields_ = (
        ("expected_generation", _U64),
        ("identity_bound", _I32),
        ("source_id", _P),
        ("source_id_length", _I64),
        ("source_revision", _P),
        ("source_revision_length", _I64),
        ("source_map_id", _P),
        ("source_map_id_length", _I64),
        ("frame_id", _P),
        ("frame_id_length", _I64),
        ("periodic", _P),
        ("kind", _I32),
        ("collapse_pair_count", _I64),
        ("collapse_removed_ids", _P),
        ("collapse_survivor_ids", _P),
        ("split_edge_count", _I64),
        ("split_edges", _P),
        ("removed_vertex_count", _I64),
        ("removed_vertex_ids", _P),
        ("removed_cell_count", _I64),
        ("removed_cell_ids", _P),
        ("changed_vertex_count", _I64),
        ("changed_vertex_ids", _P),
        ("changed_vertex_points", _P),
        ("changed_witness_strata", _P),
        ("changed_witness_entities", _P),
        ("changed_witness_parameters", _P),
        ("changed_witness_deviations", _P),
        ("changed_protection_radii", _P),
        ("changed_protected_vertices", _P),
        ("changed_edge_witness_present", _P),
        ("changed_edge_witness_endpoints", _P),
        ("changed_edge_parameters", _P),
        ("changed_vertex_ancestry_count", _I64),
        ("changed_vertex_ancestry_offsets", _P),
        ("changed_vertex_ancestor_ids", _P),
        ("changed_vertex_ancestor_winding", _P),
        ("changed_vertex_ancestry_weights", _P),
        ("replacement_cell_count", _I64),
        ("replacement_cell_ids", _P),
        ("replacement_cell_vertices", _P),
        ("replacement_cell_facet_sources", _P),
        ("replacement_regions", _P),
        ("removed_facet_count", _I64),
        ("replacement_cell_ancestry_count", _I64),
        ("replacement_cell_ancestry_offsets", _P),
        ("replacement_cell_ancestor_ids", _P),
        ("removed_facet_ids", _P),
        ("replacement_facet_count", _I64),
        ("replacement_facet_ids", _P),
        ("replacement_facet_vertices", _P),
        ("replacement_facet_sources", _P),
        ("replacement_facet_orientations", _P),
        ("replacement_protected_facets", _P),
        ("removed_segment_count", _I64),
        ("removed_segment_ids", _P),
        ("replacement_segment_count", _I64),
        ("replacement_segment_ids", _P),
        ("replacement_segment_vertices", _P),
        ("replacement_segment_sources", _P),
        ("replacement_protected_segments", _P),
    )


_SIGNATURES: dict[str, tuple[object, tuple[object, ...]]] = {
    "phx_mc_version": (ctypes.c_char_p, ()),
    "phx_mc_build_hash": (ctypes.c_char_p, ()),
    "phx_mc_exact_domain": (None, (_P, _P)),
    "phx_mc_orient2d": (_I32, (_I64, _P, _P, _P, _P)),
    "phx_mc_orient3d": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_incircle": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_insphere": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_orient2d_sos": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_orient3d_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_incircle_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_insphere_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_polygon_intersection_moments": (
        _I32,
        (_I64, _I32, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedron_intersection_moments": (
        _I32,
        (_I64, _P, _P, _P, _P, _P),
    ),
    "phx_mc_polygon_intersection_simplices": (
        _I32,
        (_I64, _I32, _P, _P, _I32, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedron_intersection_simplices": (
        _I32,
        (_I64, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_polyhedron_clip_moments": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_clip_box_halfplanes": (
        _I32,
        (_I64, _P, _P, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_clip_box_halfspaces": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _I32,
            _P,
            _P,
            _P,
            _I32,
            _I32,
            _I32,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_interval_program_execute": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _P, _I64),
    ),
    "phx_mc_simplex_quadratic_bounds": (
        _I32,
        (_I64, _I32, _I32, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_quadratic_tetrahedron_validity": (
        _I32,
        (_I64, _P, _P, _F64, _P, _P, _P),
    ),
    "phx_mc_graph_partition": (
        _I32,
        (_I64, _I64, _P, _P, _P, _P, _I32, _P, _P, _I32, _I32, _P),
    ),
    "phx_mc_parallel_for": (_I32, (_I64, _I64, _I32, _ParallelChunkCallback, _P)),
    "phx_mc_tetrahedron_dual_hexes": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_triangle_dual_quads": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_restricted_power_cells": (
        _I32,
        (ctypes.POINTER(_RestrictedPowerCellsRequest),),
    ),
    "phx_mc_restricted_power_cells_exact": (
        _I32,
        (ctypes.POINTER(_RestrictedPowerCellsExactRequest),),
    ),
    "phx_mc_power_cells_sizes": (_I32, (_P, _P)),
    "phx_mc_power_cells_export": (
        _I32,
        (_P, _I64, _I64, _I64, _I64, _I64, _I64, *(_P,) * 15),
    ),
    "phx_mc_power_cells_site_witness_size": (_I64, (_P,)),
    "phx_mc_power_cells_export_vertex_sites": (
        _I32,
        (_P, _I64, _I64, _I64, _P, _P),
    ),
    "phx_mc_power_cells_export_vertex_carriers": (_I32, (_P, _I64, _P)),
    "phx_mc_power_cells_failure": (_I32, (_P, _P, _P, _P, _P)),
    "phx_mc_power_cells_free": (None, (_P,)),
    "phx_mc_triangle_intersections": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_segment_triangle_intersections": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_point_triangle_locations": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_arrange_triangles": (_I32, (_I64, _P, _I64, _P, _P, _I64, _P, _P, _P)),
    "phx_mc_arrangement_sizes": (_I32, (_P, _P)),
    "phx_mc_arrangement_export": (
        _I32,
        (_P, _I64, _I64, _I64, _I64, _I64, *(_P,) * 9),
    ),
    "phx_mc_arrangement_classify": (_I32, (_P, _I64, _P)),
    "phx_mc_arrangement_classification_sizes": (_I32, (_P, _P)),
    "phx_mc_arrangement_export_classification": (
        _I32,
        (_P, _I64, _I64, _I64, _P, _P),
    ),
    "phx_mc_arrangement_free": (None, (_P,)),
    "phx_mc_clip_reference_triangle_exact": (
        _I32,
        (_P, _I64, _P, _P, _I32, _P, _P, _P),
    ),
    "phx_mc_clip_reference_tetrahedron_exact": (
        _I32,
        (ctypes.POINTER(_ReferenceTetrahedronClipRequest),),
    ),
    "phx_mc_reference_tetrahedron_clip_sizes": (_I32, (_P, _P)),
    "phx_mc_reference_tetrahedron_clip_export": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _I64, _P, _I64, _P, _I64, _P),
    ),
    "phx_mc_reference_tetrahedron_clip_free": (None, (_P,)),
    "phx_mc_arrange_tetrahedra_exact": (
        _I32,
        (ctypes.POINTER(_ArrangeTetrahedraExactRequest),),
    ),
    "phx_mc_tetrahedra_arrangement_sizes": (_I32, (_P, _P)),
    "phx_mc_tetrahedra_arrangement_export": (
        _I32,
        (_P, _I64, _I64, _I64, _I64, _I64, _I64, *(_P,) * 8),
    ),
    "phx_mc_tetrahedra_arrangement_free": (None, (_P,)),
    "phx_mc_periodic_delaunay": (_I32, (_I32, _I64, _P, _P, _P, _P, _F64, _P)),
    "phx_mc_periodic_cell_count": (_I64, (_P,)),
    "phx_mc_periodic_copy_cells": (_I32, (_P, _I64, _P, _P)),
    "phx_mc_periodic_free": (None, (_P,)),
    "phx_mc_periodic_delaunay_local": (
        _I32,
        (ctypes.POINTER(_PeriodicLocalDelaunayRequest),),
    ),
    "phx_mc_periodic_local_delaunay_sizes": (_I32, (_P, _P, _P)),
    "phx_mc_periodic_local_delaunay_copy": (
        _I32,
        (_P, _I64, *(_P,) * 9),
    ),
    "phx_mc_periodic_local_delaunay_owner_bounds": (
        _I32,
        (_P, _P, _P, _P),
    ),
    "phx_mc_periodic_local_delaunay_free": (None, (_P,)),
    "phx_mc_periodic_owner_support_query": (
        _I32,
        (ctypes.POINTER(_PeriodicSupportQueryRequest),),
    ),
    "phx_mc_periodic_support_query_sizes": (_I32, (_P, _P, _P)),
    "phx_mc_periodic_support_query_copy_hits": (
        _I32,
        (_P, _I64, _P, _P, _P, _P),
    ),
    "phx_mc_periodic_support_query_free": (None, (_P,)),
    "phx_mc_periodic_refine_2d": (
        _I32,
        (ctypes.POINTER(_PeriodicRefinementInput), _P),
    ),
    "phx_mc_periodic_refinement_sizes": (_I32, (_P, _P)),
    "phx_mc_periodic_refinement_copy_points": (_I32, (_P, _I64, _P)),
    "phx_mc_periodic_refinement_copy_cells": (_I32, (_P, _I64, _P, _P)),
    "phx_mc_periodic_refinement_free": (None, (_P,)),
    "phx_mc_restricted_dual_3d": (
        _I32,
        (_I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_restricted_centers_3d": (_I32, (_I64, _P, _I64, _P, _P, _P)),
    "phx_mc_restricted_rays_3d": (
        _I32,
        (_I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_surface_reconnect": (
        _I32,
        (ctypes.POINTER(_SurfaceReconnectRequest),),
    ),
    "phx_mc_surface_envelope_create": (
        _I32,
        (ctypes.POINTER(_SurfaceEnvelopeCreateRequest),),
    ),
    "phx_mc_surface_envelope_sizes": (_I32, (_P, _P)),
    "phx_mc_surface_envelope_arrays": (
        _I32,
        (_P, _I64, _I64, _I64, _I64, _P, _P, _P, _P),
    ),
    "phx_mc_surface_envelope_free": (None, (_P,)),
    "phx_mc_level_set_split_3d": (_I32, (ctypes.POINTER(_LevelSetSplit3DRequest),)),
    "phx_mc_mixed_template_counts": (_I32, (_I32, _I32, _I32, _P)),
    "phx_mc_mixed_template": (_I32, (_I32, _I32, _I32, _I64, _P)),
    "phx_mc_plc3d_prepare_source": (
        _I32,
        (ctypes.POINTER(_Plc3DPrepareSourceRequest),),
    ),
    "phx_mc_plc3d_recover": (
        _I32,
        (ctypes.POINTER(_Plc3DPrepareSourceRequest),),
    ),
    "phx_mc_plc3d_sizes": (_I32, (_P, _P)),
    "phx_mc_plc3d_export": (
        _I32,
        (_P, _I64, _I64, _I64, _I64, _I64, _I64, *(_P,) * 13),
    ),
    "phx_mc_plc3d_source_witnesses": (
        _I32,
        (_P, _I64, _P, _P, _P, _P),
    ),
    "phx_mc_plc3d_failure": (_I32, (_P, *(_P,) * 7)),
    "phx_mc_plc3d_suppression_search": (_I32, (_P, _P)),
    "phx_mc_plc3d_free": (None, (_P,)),
    "phx_mc_tet_mesh_create": (
        _I32,
        (ctypes.POINTER(_TetMeshCreateRequest),),
    ),
    "phx_mc_tet_mesh_snapshot_sizes": (_I32, (_P, _P)),
    "phx_mc_tet_mesh_identity_sizes": (_I32, (_P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_periodic_identity_sizes": (_I32, (_P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_periodic_identity": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _I64, _P, _I64, _P, _I64, _P),
    ),
    "phx_mc_tet_mesh_identity": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _I64, _P),
    ),
    "phx_mc_tet_mesh_unmet_count": (_I32, (_P, _P)),
    "phx_mc_tet_mesh_unmet": (_I32, (_P, _I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_plane_record_count": (_I32, (_P, _P)),
    "phx_mc_tet_mesh_plane_records": (_I32, (_P, _I64, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_snapshot": (
        _I32,
        (_P, ctypes.POINTER(_TetMeshSnapshotRequest)),
    ),
    "phx_mc_tet_mesh_apply_edit": (
        _I32,
        (_P, ctypes.POINTER(_TetMeshEditRequest), _P),
    ),
    "phx_mc_tet_mesh_refine": (
        _I32,
        (
            _P,
            _U64,
            _I32,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _F64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_tet_mesh_improve_quality": (
        _I32,
        (
            _P,
            _U64,
            _I32,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _F64,
            _F64,
            _F64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_tet_mesh_free": (None, (_P,)),
    "phx_mc_delaunay_2d": (_I32, (_I64, _P, _P)),
    "phx_mc_regular_2d": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_delaunay_3d": (_I32, (_I64, _P, _P)),
    "phx_mc_regular_3d": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_constrained_delaunay_2d": (
        _I32,
        (_I64, _P, _I64, _P, _I64, _P, _I32, _F64, _F64, _P),
    ),
    "phx_mc_mesh_dimension": (_I32, (_P,)),
    "phx_mc_mesh_point_count": (_I64, (_P,)),
    "phx_mc_mesh_cell_count": (_I64, (_P,)),
    "phx_mc_mesh_input_point_count": (_I64, (_P,)),
    "phx_mc_mesh_copy_points": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_copy_cells": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_copy_vertex_map": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_copy_cell_segments": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_copy_constraint_sources": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_copy_constraint_parents": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_copy_constraint_ratios": (_I32, (_P, _I64, _P)),
    "phx_mc_mesh_free": (None, (_P,)),
}


def _library_location(configured: str, /) -> Path | str:
    """Return the library path, or the reason it is unavailable."""

    if configured:
        path = Path(configured)
        if not path.is_file():
            return f"{LIBRARY_ENVIRONMENT['meshcore']} names a missing meshcore library: {configured!r}."
        return path
    if importlib.util.find_spec("phydrax_meshcore") is None:
        return (
            "The meshcore library is unavailable: install the optional "
            "'phydrax[meshcore]' extra or set "
            f"{LIBRARY_ENVIRONMENT['meshcore']} to libphydrax_meshcore."
        )
    return Path(importlib.import_module("phydrax_meshcore").library_path())


@final
class MeshcoreLibrary:
    """Loaded meshcore shared library with typed C entry points."""

    __slots__ = (
        "_functions",
        "_interval_capability_error",
        "_interval_capability_lock",
        "_interval_program_opcodes",
        "_library",
        "build_hash",
        "path",
        "version",
    )

    def __init__(
        self,
        library: ctypes.CDLL,
        functions: Mapping[str, Callable[..., object]],
        path: Path,
        version: str,
        build_hash: str,
        /,
    ) -> None:
        self._library = library
        self._functions = functions
        self.path = path
        self.version = version
        self.build_hash = build_hash
        self._interval_program_opcodes: frozenset[int] | None = None
        self._interval_capability_error: str | None = None
        self._interval_capability_lock = threading.Lock()

    def __getitem__(self, name: str, /) -> Any:
        return self._functions[name]

    def interval_program_capabilities(self, /) -> frozenset[int]:
        """Read the required interval opcode capability mask once on first use."""
        if (cached := self._interval_program_opcodes) is not None:
            return cached
        with self._interval_capability_lock:
            if (cached := self._interval_program_opcodes) is not None:
                return cached
            if (error := self._interval_capability_error) is not None:
                raise MeshcoreUnavailableError(error)
            try:
                function = self._library["phx_mc_interval_program_capabilities"]
            except AttributeError as error:
                message = (
                    f"meshcore library {str(self.path)!r} lacks the required "
                    "phx_mc_interval_program_capabilities C ABI symbol."
                )
                self._interval_capability_error = message
                raise MeshcoreUnavailableError(message) from error
            function.restype = _I32
            function.argtypes = (_P,)
            mask_array = np.zeros((1,), dtype=np.uint64)
            status = function(mask_array.ctypes.data)
            _raise_for_call(status, "interval_program_capabilities")
            mask = int(mask_array[0])
            opcodes = frozenset(opcode for opcode in range(1, 64) if mask & (1 << opcode))
            self._interval_program_opcodes = opcodes
            return opcodes


def _identity_text(function: Callable[[], object], symbol: str, path: Path, /) -> Any:
    """Decode one required non-null ASCII identity string, or return its error."""

    raw = function()
    if not isinstance(raw, bytes):
        return None, f"meshcore library {str(path)!r} returned null from {symbol}."
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        return None, f"meshcore library {str(path)!r} returned non-ASCII {symbol}."
    if not value:
        return None, f"meshcore library {str(path)!r} returned an empty {symbol}."
    return value, None


def _bind(library: ctypes.CDLL, path: Path, /) -> MeshcoreLibrary | str:
    """Bind every C ABI entry point and verify the release, or return the reason."""

    functions = {}
    missing = []
    for name, (restype, argtypes) in _SIGNATURES.items():
        # ctypes reports an unresolved symbol only as AttributeError.
        try:
            function = library[name]
        except AttributeError:
            missing.append(name)
            continue
        # ty: ignore[invalid-assignment]
        function.restype = restype
        # ty: ignore[invalid-assignment]
        function.argtypes = argtypes
        functions[name] = function
    if missing:
        return (
            f"meshcore library {str(path)!r} lacks C ABI symbols "
            f"{', '.join(missing)}; install the phydrax-meshcore release of this phydrax."
        )
    version, error = _identity_text(functions["phx_mc_version"], "phx_mc_version", path)
    if error is not None:
        return error
    build_hash, error = _identity_text(
        functions["phx_mc_build_hash"], "phx_mc_build_hash", path
    )
    if error is not None:
        return error
    if len(build_hash) != 64 or any(
        character not in "0123456789abcdef" for character in build_hash
    ):
        return f"meshcore library {str(path)!r} returned an invalid phx_mc_build_hash."
    try:
        required = importlib.metadata.version("phydrax")
    except importlib.metadata.PackageNotFoundError:
        return (
            "The meshcore release cannot be verified: phydrax distribution "
            "metadata is not installed."
        )
    if version != required:
        return (
            f"meshcore library {str(path)!r} is release {version}, but phydrax "
            f"{required} requires phydrax-meshcore=={required}."
        )
    return MeshcoreLibrary(library, functions, path, version, build_hash)


@functools.cache
def _resolve(configured: str, /) -> MeshcoreLibrary | str:
    location = _library_location(configured)
    if isinstance(location, str):
        return location
    # ctypes reports every loader failure as OSError; it is the only signal.
    try:
        library = ctypes.CDLL(str(location))
    except OSError as error:
        return f"meshcore library {str(location)!r} cannot be loaded: {error}"
    return _bind(library, location)


def _current() -> MeshcoreLibrary | str:
    return _resolve((library_input("meshcore") or "").strip())


def meshcore_available() -> bool:
    """Whether the native meshcore library can be loaded."""

    return not isinstance(_current(), str)


def load_meshcore() -> MeshcoreLibrary:
    """Return the loaded meshcore library or raise :class:`MeshcoreUnavailableError`."""

    resolved = _current()
    if isinstance(resolved, str):
        raise MeshcoreUnavailableError(resolved)
    return resolved


def meshcore_identity() -> str:
    """Library release and source build hash: ``phydrax-meshcore <release> <sha256>``."""

    library = load_meshcore()
    return f"phydrax-meshcore {library.version} {library.build_hash}"


def meshcore_exact_domain() -> tuple[int, int]:
    """Binary exponent range ``(minimum, maximum)`` of exactly handled coordinates.

    Coordinates must be zero or have magnitude in ``[2**minimum, 2**maximum]``;
    weights use the doubled range.
    """

    library = load_meshcore()
    minimum = np.zeros((1,), dtype=np.int32)
    maximum = np.zeros((1,), dtype=np.int32)
    library["phx_mc_exact_domain"](minimum.ctypes.data, maximum.ctypes.data)
    return int(minimum[0]), int(maximum[0])


# ---------------------------------------------------------------- validation


def _real_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_) or not (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be a real floating or integer array.")
    if np.issubdtype(array.dtype, np.integer) and np.any(
        (array > 2**53) | (array < -(2**53))
    ):
        raise ValueError(f"{name} integer values must be exactly representable.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _index_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array.")
    return array


def _points(value: object, name: str, width: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    return array


def _batched_points(
    arrays: tuple[object, ...], names: tuple[str, ...], width: int, /
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    converted = tuple(
        _real_array(value, name) for value, name in zip(arrays, names, strict=True)
    )
    for array, name in zip(converted, names, strict=True):
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(f"{name} must have shape (..., {width}).")
    leading = np.broadcast_shapes(*(array.shape[:-1] for array in converted))
    flat = tuple(
        np.ascontiguousarray(
            np.broadcast_to(array, leading + (width,)).reshape((-1, width)),
            dtype=np.float64,
        )
        for array in converted
    )
    return flat, leading


# Statuses that reject the caller's input; they raise ``ValueError``.
_CALLER_INPUT_STATUSES = frozenset(
    (
        MeshcoreStatus.NONFINITE_INPUT,
        MeshcoreStatus.RANGE_ERROR,
        MeshcoreStatus.DEGENERATE_INPUT,
        MeshcoreStatus.INVALID_ARGUMENT,
        MeshcoreStatus.INVALID_INPUT,
        MeshcoreStatus.CONSTRAINT_INTERSECTION,
    )
)


def _raise_for_call(
    status: int,
    operation: str,
    /,
    *,
    contracts: Mapping[MeshcoreStatus, str] | None = None,
) -> None:
    """Raise the error a native call status reports.

    The default texts state each status's library-wide contract. A call site
    whose kernel returns a status for a different contract (for example a
    certified error bound in place of the coordinate domain) names it through
    ``contracts``; the exception type stays the status's.
    """
    code = MeshcoreStatus(status)
    contract = None if contracts is None else contracts.get(code)
    match code:
        case MeshcoreStatus.OK:
            return
        case _ if contract is not None:
            message = f"{operation}: {contract}"
            if code in _CALLER_INPUT_STATUSES:
                raise ValueError(message)
            if code is MeshcoreStatus.INDEX_RANGE_ERROR:
                raise OverflowError(message)
            raise MeshcoreError(code, message)
        case MeshcoreStatus.NONFINITE_INPUT:
            raise ValueError(f"{operation}: inputs must be finite.")
        case MeshcoreStatus.RANGE_ERROR:
            raise ValueError(
                f"{operation}: coordinates must be zero or have magnitude in "
                "[2**-120, 2**120] (weights [2**-240, 2**240]) for exact evaluation."
            )
        case MeshcoreStatus.DEGENERATE_INPUT:
            raise ValueError(f"{operation}: the distinct points do not span the space.")
        case MeshcoreStatus.INVALID_ARGUMENT | MeshcoreStatus.INVALID_INPUT:
            raise ValueError(f"{operation}: invalid input ({code.name}).")
        case MeshcoreStatus.CONSTRAINT_INTERSECTION:
            raise ValueError(
                f"{operation}: constraint segments cross in their interiors."
            )
        case MeshcoreStatus.CAPACITY_EXCEEDED:
            raise MeshcoreError(code, f"{operation}: caller output buffer is too small")
        case MeshcoreStatus.INDEX_RANGE_ERROR:
            raise OverflowError(
                f"{operation}: native index representation is insufficient."
            )
        case (
            MeshcoreStatus.UNRESOLVED
            | MeshcoreStatus.ALLOCATION_FAILURE
            | MeshcoreStatus.INTERNAL_ERROR
        ):
            raise MeshcoreError(code, f"{operation}: native operation did not complete")
        case _:
            raise ValueError(f"Unknown meshcore status {status}.")


def _native_integer_array(value: object, name: str, dtype: np.dtype, /) -> np.ndarray:
    array = _index_array(value, name)
    bounds = np.iinfo(dtype)
    if array.size:
        if array.dtype.kind == "u":
            if np.any(array > bounds.max):
                raise OverflowError(f"{name} exceeds the native integer range.")
        elif np.any(array < bounds.min) or np.any(array > bounds.max):
            raise OverflowError(f"{name} exceeds the native integer range.")
    return np.ascontiguousarray(array, dtype=dtype)


def interval_program_execute(
    instructions: object,
    indices: object,
    parameters: object,
    lower: object,
    upper: object,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute an interval program in caller-owned mutable binary64 workspaces.

    ``instructions`` has shape ``(rows, 8)``; lower and upper are equal-shaped
    one-dimensional workspaces whose length is their actual native output extent.
    """

    instruction_array = _native_integer_array(
        instructions, "instructions", np.dtype(np.int64)
    )
    index_array = _native_integer_array(indices, "indices", np.dtype(np.int64))
    parameter_array = _real_array(parameters, "parameters")
    lower_array = _real_array(lower, "lower")
    upper_array = _real_array(upper, "upper")
    if instruction_array.ndim != 2 or instruction_array.shape[1] != 8:
        raise ValueError("instructions must have shape (rows, 8).")
    if index_array.ndim != 1 or parameter_array.ndim != 1:
        raise ValueError("indices and parameters must be one-dimensional.")
    if lower_array.ndim != 1 or upper_array.shape != lower_array.shape:
        raise ValueError("lower and upper must be equal-shaped one-dimensional arrays.")
    if not lower_array.flags.writeable or not upper_array.flags.writeable:
        raise ValueError("lower and upper workspaces must be writable.")
    if np.shares_memory(lower_array, upper_array):
        raise ValueError("lower and upper workspaces must not overlap.")
    for input_array in (instruction_array, index_array, parameter_array):
        if np.shares_memory(input_array, lower_array) or np.shares_memory(
            input_array, upper_array
        ):
            raise ValueError(
                "interval program inputs must not overlap output workspaces."
            )
    library = load_meshcore()
    status = library["phx_mc_interval_program_execute"](
        instruction_array.ctypes.data,
        instruction_array.shape[0],
        index_array.ctypes.data,
        index_array.shape[0],
        parameter_array.ctypes.data,
        parameter_array.shape[0],
        lower_array.ctypes.data,
        upper_array.ctypes.data,
        lower_array.size,
    )
    _raise_for_call(status, "interval_program_execute")
    return lower_array, upper_array


def simplex_quadratic_bounds(
    q_lower: object,
    q_upper: object,
    barycentric_lower: object,
    barycentric_upper: object,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outward Bernstein coefficient bounds for simplex quadratic maps."""

    lower_coefficients = _real_array(q_lower, "q_lower")
    upper_coefficients = _real_array(q_upper, "q_upper")
    if (
        lower_coefficients.ndim != 4
        or upper_coefficients.shape != lower_coefficients.shape
        or lower_coefficients.shape[2] != lower_coefficients.shape[3]
    ):
        raise ValueError("quadratic coefficients must have aligned shape (n, c, m, m).")
    count, components, vertex_count, _ = lower_coefficients.shape
    if not 1 <= components <= 3 or not 2 <= vertex_count <= 4:
        raise ValueError(
            "quadratic maps require 1-3 components and simplex dimension 1-3."
        )
    bary_lower = _real_array(barycentric_lower, "barycentric_lower")
    bary_upper = _real_array(barycentric_upper, "barycentric_upper")
    expected = (count, vertex_count, vertex_count)
    if bary_lower.shape != expected or bary_upper.shape != expected:
        raise ValueError(f"barycentric bounds must have shape {expected}.")
    lower = np.empty((count, components), dtype=np.float64)
    upper = np.empty((count, components), dtype=np.float64)
    library = load_meshcore()
    status = library["phx_mc_simplex_quadratic_bounds"](
        count,
        vertex_count - 1,
        components,
        lower_coefficients.ctypes.data,
        upper_coefficients.ctypes.data,
        bary_lower.ctypes.data,
        bary_upper.ctypes.data,
        lower.ctypes.data,
        upper.ctypes.data,
    )
    _raise_for_call(status, "simplex_quadratic_bounds")
    return lower, upper


def quadratic_tetrahedron_validity(
    nodes: object,
    derivative_weights: object,
    relative_floor: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Certify quadratic tetrahedron validity or return per-cell UNRESOLVED."""

    node_array = _real_array(nodes, "nodes")
    if node_array.ndim != 3 or node_array.shape[1:] != (10, 3):
        raise ValueError("nodes must have shape (n, 10, 3).")
    weight_array = _native_integer_array(
        derivative_weights, "derivative_weights", np.dtype(np.int32)
    )
    if weight_array.shape != (3, 4, 10):
        raise ValueError("derivative_weights must have shape (3, 4, 10).")
    floor = float(relative_floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError("relative_floor must lie in [0, 1).")
    count = node_array.shape[0]
    decisions = np.empty((count,), dtype=np.int32)
    lower = np.empty((count,), dtype=np.float64)
    upper = np.empty((count,), dtype=np.float64)
    library = load_meshcore()
    status = library["phx_mc_quadratic_tetrahedron_validity"](
        count,
        node_array.ctypes.data,
        weight_array.ctypes.data,
        floor,
        decisions.ctypes.data,
        lower.ctypes.data,
        upper.ctypes.data,
    )
    _raise_for_call(status, "quadratic_tetrahedron_validity")
    return decisions, lower, upper


def graph_partition(
    offsets: object,
    neighbors: object,
    edge_weights: object,
    vertex_weights: object,
    part_targets: object,
    part_capacities: object,
    /,
    *,
    require_nonempty: bool,
    refinement_passes: int,
) -> np.ndarray:
    """Partition a symmetric weighted CSR graph without work quotas."""

    csr_offsets = _native_integer_array(offsets, "offsets", np.dtype(np.int64))
    adjacent = _native_integer_array(neighbors, "neighbors", np.dtype(np.int32))
    edges = _native_integer_array(edge_weights, "edge_weights", np.dtype(np.int64))
    vertices = _native_integer_array(vertex_weights, "vertex_weights", np.dtype(np.int64))
    targets = _native_integer_array(part_targets, "part_targets", np.dtype(np.int64))
    capacities = _native_integer_array(
        part_capacities, "part_capacities", np.dtype(np.int64)
    )
    if vertices.ndim != 1:
        raise ValueError("vertex_weights must have shape (n,).")
    vertex_count = vertices.shape[0]
    if (
        csr_offsets.shape != (vertex_count + 1,)
        or adjacent.ndim != 1
        or edges.shape != adjacent.shape
        or targets.ndim != 1
        or capacities.shape != targets.shape
    ):
        raise ValueError("CSR and partition weight arrays have incompatible shapes.")
    if not isinstance(require_nonempty, bool):
        raise TypeError("require_nonempty must be a bool.")
    if isinstance(refinement_passes, bool) or not isinstance(
        refinement_passes, (int, np.integer)
    ):
        raise TypeError("refinement_passes must be an integer.")
    passes = int(refinement_passes)
    if passes < 0 or passes > np.iinfo(np.int32).max:
        raise ValueError("refinement_passes must lie in the native int32 range.")
    parts = np.empty((vertex_count,), dtype=np.int32)
    library = load_meshcore()
    status = library["phx_mc_graph_partition"](
        vertex_count,
        adjacent.size,
        csr_offsets.ctypes.data,
        adjacent.ctypes.data,
        edges.ctypes.data,
        vertices.ctypes.data,
        targets.size,
        targets.ctypes.data,
        capacities.ctypes.data,
        1 if require_nonempty else 0,
        passes,
        parts.ctypes.data,
    )
    _raise_for_call(status, "graph_partition")
    return parts


def _int32_argument(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    bounds = np.iinfo(np.int32)
    if result < bounds.min or result > bounds.max:
        raise OverflowError(f"{name} exceeds the native int32 range.")
    return result


def parallel_for(
    count: int,
    body: Callable[[int, int, int], MeshcoreStatus | int | None],
    /,
    *,
    chunk_size: int = 64,
    parallelism: int = 1,
) -> MeshcoreStatus:
    """Run disjoint indexed chunks through the native process-wide worker pool.

    ``body(begin, end, chunk_index)`` may return ``None`` or a meshcore status.
    All chunks finish before the first non-OK status in chunk order is returned.
    Python exceptions raised by the callback are re-raised after native workers
    have joined.
    """

    if not callable(body):
        raise TypeError("body must be callable.")
    item_count = _nonnegative_capacity(count, "count", np.iinfo(np.int64).max)
    chunk_extent = _nonnegative_capacity(chunk_size, "chunk_size", np.iinfo(np.int64).max)
    if chunk_extent == 0:
        raise ValueError("chunk_size must be positive.")
    workers = _int32_argument(parallelism, "parallelism")
    if workers <= 0:
        raise ValueError("parallelism must be positive.")

    callback_errors: dict[int, BaseException] = {}

    def dispatch(_context: object, begin: int, end: int, chunk_index: int) -> int:
        try:
            result = body(begin, end, chunk_index)
            if result is None:
                return int(MeshcoreStatus.OK)
            if isinstance(result, bool) or not isinstance(result, (int, np.integer)):
                raise TypeError(
                    "parallel callback must return None or a meshcore status."
                )
            return int(MeshcoreStatus(int(result)))
        except BaseException as exc:
            callback_errors.setdefault(chunk_index, exc)
            return int(MeshcoreStatus.INTERNAL_ERROR)

    native_callback = _ParallelChunkCallback(dispatch)
    status = MeshcoreStatus(
        load_meshcore()["phx_mc_parallel_for"](
            item_count, chunk_extent, workers, native_callback, None
        )
    )
    if callback_errors:
        raise callback_errors[min(callback_errors)]
    return status


def _mixed_template_counts(
    library: MeshcoreLibrary, kind: int, axial: int, index: int, /
) -> MixedTemplateCounts:
    counts = np.empty((3,), dtype=np.int32)
    _raise_for_call(
        library["phx_mc_mixed_template_counts"](kind, axial, index, counts.ctypes.data),
        "mixed_template_counts",
    )
    if np.any(counts <= 0):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "mixed_template_counts: invalid successful result",
        )
    return MixedTemplateCounts(*(int(value) for value in counts))


def mixed_template_counts(
    kind: int | MixedElementKind, axial: int | bool, index: int, /
) -> MixedTemplateCounts:
    """Return the template-family, child, and corner counts for one topology."""

    kind_value = _int32_argument(kind, "kind")
    axial_value = (
        int(axial) if isinstance(axial, bool) else _int32_argument(axial, "axial")
    )
    index_value = _int32_argument(index, "index")
    return _mixed_template_counts(load_meshcore(), kind_value, axial_value, index_value)


def mixed_template(
    kind: int | MixedElementKind, axial: int | bool, index: int, /
) -> np.ndarray:
    """Return an orientation-preserving reference-cell partition."""

    kind_value = _int32_argument(kind, "kind")
    axial_value = (
        int(axial) if isinstance(axial, bool) else _int32_argument(axial, "axial")
    )
    index_value = _int32_argument(index, "index")
    library = load_meshcore()
    counts = _mixed_template_counts(library, kind_value, axial_value, index_value)
    references = np.empty((counts.child_count, counts.corner_count, 3), dtype=np.float64)
    _raise_for_call(
        library["phx_mc_mixed_template"](
            kind_value, axial_value, index_value, references.size, references.ctypes.data
        ),
        "mixed_template",
    )
    return references


def tetrahedron_dual_hexes(nodes: object, vertex_count: int, /) -> np.ndarray:
    """Return the four barycentric dual hexahedra of each tetrahedron.

    ``nodes`` has shape ``(tetrahedra, 15)`` and contains each positive
    tetrahedron's four vertices, six shared edge-centroid IDs, four shared
    face-centroid IDs, and cell-centroid ID. IDs index the complete coordinate
    bank, whose row count is ``vertex_count``.
    """

    node_array = _native_integer_array(nodes, "nodes", np.dtype(np.int64))
    if node_array.ndim != 2 or node_array.shape[1] != 15:
        raise ValueError("nodes must have shape (tetrahedra, 15).")
    count = int(node_array.shape[0])
    if count > np.iinfo(np.int64).max // 4:
        raise OverflowError("tetrahedron_dual_hexes output row count overflows int64.")
    vertex_extent = _nonnegative_capacity(
        vertex_count, "vertex_count", int(np.iinfo(np.int64).max)
    )
    if node_array.size and (
        np.any(node_array < 0) or np.any(node_array >= vertex_extent)
    ):
        raise ValueError("nodes contain a vertex ID outside vertex_count.")
    hexes = np.empty((4 * count, 8), dtype=np.int64)
    library = load_meshcore()
    _raise_for_call(
        library["phx_mc_tetrahedron_dual_hexes"](
            count,
            vertex_extent,
            node_array.ctypes.data,
            4 * count,
            hexes.ctypes.data,
        ),
        "tetrahedron_dual_hexes",
    )
    return hexes


def triangle_dual_quads(nodes: object, vertex_count: int, /) -> np.ndarray:
    """Return the three barycentric dual quadrilaterals of each triangle.

    ``nodes`` has shape ``(triangles, 7)`` and contains the triangle vertices,
    shared edge-centroid IDs in (01, 12, 20) order, and its center ID. All IDs
    index the complete coordinate bank, whose row count is ``vertex_count``.
    """

    node_array = _native_integer_array(nodes, "nodes", np.dtype(np.int64))
    if node_array.ndim != 2 or node_array.shape[1] != 7:
        raise ValueError("nodes must have shape (triangles, 7).")
    count = int(node_array.shape[0])
    max_triangles = np.iinfo(np.intp).max // (12 * np.dtype(np.int64).itemsize)
    if count > max_triangles:
        raise OverflowError(
            "triangle_dual_quads output extent exceeds addressable storage."
        )
    vertex_extent = _nonnegative_capacity(
        vertex_count, "vertex_count", int(np.iinfo(np.int64).max)
    )
    if node_array.size and (
        np.any(node_array < 0) or np.any(node_array >= vertex_extent)
    ):
        raise ValueError("nodes contain a vertex ID outside vertex_count.")
    quads = np.empty((3 * count, 4), dtype=np.int64)
    library = load_meshcore()
    _raise_for_call(
        library["phx_mc_triangle_dual_quads"](
            count,
            vertex_extent,
            node_array.ctypes.data,
            3 * count,
            quads.ctypes.data,
        ),
        "triangle_dual_quads",
    )
    return quads


# ---------------------------------------------------------------- predicates


def _predicate(
    name: str, arrays: tuple[object, ...], width: int, /, ids: object = None
) -> np.ndarray:
    library = load_meshcore()
    names = tuple(f"point {index}" for index in range(len(arrays)))
    flat, leading = _batched_points(arrays, names, width)
    count = flat[0].shape[0]
    signs = np.zeros((count,), dtype=np.int8)
    pointers = [array.ctypes.data for array in flat]
    if ids is None:
        status = library[name](count, *pointers, signs.ctypes.data)
    else:
        index = _index_array(ids, "ids")
        if index.shape[-1:] != (len(arrays),):
            raise ValueError(f"ids must have shape (..., {len(arrays)}).")
        index_flat = np.ascontiguousarray(
            np.broadcast_to(index, leading + (len(arrays),)).reshape((-1, len(arrays))),
            dtype=np.int64,
        )
        status = library[name](
            count, *pointers, index_flat.ctypes.data, signs.ctypes.data
        )
    _raise_for_call(status, name.removeprefix("phx_mc_"))
    return signs.reshape(leading)


def exact_orient2d(a: object, b: object, c: object, /) -> np.ndarray:
    """Exact ``sign det[b - a, c - a]`` for ``(..., 2)`` arrays (int8 in {-1, 0, 1})."""

    return _predicate("phx_mc_orient2d", (a, b, c), 2)


def exact_orient3d(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """Exact ``sign det[b - a, c - a, d - a]`` for ``(..., 3)`` arrays."""

    return _predicate("phx_mc_orient3d", (a, b, c, d), 3)


def exact_incircle(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """Exact in-circle sign: positive when ``d`` is inside the circle of CCW ``(a, b, c)``."""

    return _predicate("phx_mc_incircle", (a, b, c, d), 2)


def exact_insphere(
    a: object, b: object, c: object, d: object, e: object, /
) -> np.ndarray:
    """Exact in-sphere sign: positive when ``e`` is inside the sphere of positive ``(a..d)``."""

    return _predicate("phx_mc_insphere", (a, b, c, d, e), 3)


def symbolic_orient2d(a: object, b: object, c: object, ids: object, /) -> np.ndarray:
    """Index-ordered Simulation-of-Simplicity ``orient2d``; never zero for distinct ids."""

    return _predicate("phx_mc_orient2d_sos", (a, b, c), 2, ids)


def symbolic_orient3d(
    a: object, b: object, c: object, d: object, ids: object, /
) -> np.ndarray:
    """Index-ordered Simulation-of-Simplicity ``orient3d``; never zero for distinct ids."""

    return _predicate("phx_mc_orient3d_sos", (a, b, c, d), 3, ids)


def symbolic_incircle(
    a: object, b: object, c: object, d: object, ids: object, /
) -> np.ndarray:
    """Lifted-perturbation ``incircle``; zero only for four collinear points."""

    return _predicate("phx_mc_incircle_sos", (a, b, c, d), 2, ids)


def symbolic_insphere(
    a: object, b: object, c: object, d: object, e: object, ids: object, /
) -> np.ndarray:
    """Lifted-perturbation ``insphere``; zero only for five coplanar points."""

    return _predicate("phx_mc_insphere_sos", (a, b, c, d, e), 3, ids)


# ---------------------------------------------------------------- clipping


def _counted_polygons(
    vertices: object, counts: object, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    array = _real_array(vertices, name)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"{name} must have shape (n, k, 2).")
    if array.shape[1] > np.iinfo(np.int32).max:
        raise OverflowError(f"{name} vertex stride exceeds the native int32 extent.")
    count_array = _index_array(counts, f"{name} counts")
    if count_array.shape != array.shape[:1]:
        raise ValueError(f"{name} counts must have shape (n,).")
    if np.any(count_array < 0) or np.any(count_array > array.shape[1]):
        raise ValueError(f"{name} counts must lie in [0, {array.shape[1]}].")
    return array, np.ascontiguousarray(count_array, dtype=np.int32)


def polygon_intersection_moments(
    first_vertices: object,
    first_counts: object,
    second_vertices: object,
    second_counts: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of convex polygon pairs.

    Polygons are ``(n, k, 2)`` padded vertex arrays with per-row counts, in either
    orientation.  Returns ``(area (n,), first_moment (n, 2), status (n,))`` where
    ``first_moment`` is the integral of ``x`` over the intersection and ``status``
    holds :class:`MeshcoreStatus` item codes (empty and contact overlaps are ``OK``
    with zero area).
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError("first and second polygon batches must have equal length.")
    count = first.shape[0]
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polygon_intersection_moments"](
        count,
        first.shape[1],
        first.ctypes.data,
        first_count.ctypes.data,
        second.shape[1],
        second.ctypes.data,
        second_count.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polygon_intersection_moments")
    return area, moment, status


def polygon_intersection_simplices(
    first_vertices: object,
    first_counts: object,
    second_vertices: object,
    second_counts: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convex polygon-pair intersections as triangle partitions plus moments.

    The returned simplex storage uses the proven ``k_first + k_second`` upper
    bound; ``simplex_counts`` selects each item's occupied prefix.
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError("first and second polygon batches must have equal length.")
    capacity = first.shape[1] + second.shape[1]
    if capacity > np.iinfo(np.int32).max:
        raise OverflowError(
            "polygon intersection output extent exceeds native int32 range."
        )
    count = first.shape[0]
    simplices = np.zeros((count, capacity, 3, 2), dtype=np.float64)
    simplex_counts = np.zeros((count,), dtype=np.int32)
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polygon_intersection_simplices"](
        count,
        first.shape[1],
        first.ctypes.data,
        first_count.ctypes.data,
        second.shape[1],
        second.ctypes.data,
        second_count.ctypes.data,
        capacity,
        simplices.ctypes.data,
        simplex_counts.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polygon_intersection_simplices")
    return simplices, simplex_counts, area, moment, status


def _tetrahedra(value: object, name: str, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (4, 3):
        raise ValueError(f"{name} must have shape (n, 4, 3).")
    return array


def tetrahedron_intersection_moments(
    first: object, second: object, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of tetrahedron pairs ``(n, 4, 3)``."""

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
    count = first_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_moments"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_moments")
    return volume, moment, status


def tetrahedron_intersection_simplices(
    first: object, second: object, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tetrahedron-pair intersections as simplex partitions plus their moments."""

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
    capacity = 20
    count = first_array.shape[0]
    simplices = np.zeros((count, capacity, 4, 3), dtype=np.float64)
    simplex_counts = np.zeros((count,), dtype=np.int32)
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_simplices"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        capacity,
        simplices.ctypes.data,
        simplex_counts.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_simplices")
    return simplices, simplex_counts, volume, moment, status


def _halfspaces(
    normals: object, offsets: object, plane_counts: object, width: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_array = _real_array(normals, "normals")
    if normal_array.ndim != 3 or normal_array.shape[2] != width:
        raise ValueError(f"normals must have shape (n, m, {width}).")
    if normal_array.shape[1] > np.iinfo(np.int32).max:
        raise OverflowError("plane_capacity exceeds the native int32 input stride.")
    offset_array = _real_array(offsets, "offsets")
    if offset_array.shape != normal_array.shape[:2]:
        raise ValueError("offsets must have shape (n, m).")
    count_array = _index_array(plane_counts, "plane_counts")
    if count_array.shape != normal_array.shape[:1]:
        raise ValueError("plane_counts must have shape (n,).")
    if np.any(count_array < 0) or np.any(count_array > normal_array.shape[1]):
        raise ValueError(f"plane_counts must lie in [0, {normal_array.shape[1]}].")
    return normal_array, offset_array, np.ascontiguousarray(count_array, dtype=np.int32)


def polyhedron_clip_moments(
    normals: object,
    offsets: object,
    plane_counts: object,
    tetrahedra: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moments of convex halfspace polyhedra intersected with tetrahedra."""

    library = load_meshcore()
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    tetrahedron_array = _tetrahedra(tetrahedra, "tetrahedra")
    if tetrahedron_array.shape[0] != normal_array.shape[0]:
        raise ValueError("halfspace and tetrahedron batches must have equal length.")
    count = normal_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polyhedron_clip_moments"](
        count,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        tetrahedron_array.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polyhedron_clip_moments")
    return volume, moment, status


def _box(lower: object, upper: object, width: int, /) -> tuple[np.ndarray, np.ndarray]:
    lower_array = _real_array(lower, "box_lower")
    upper_array = _real_array(upper, "box_upper")
    if lower_array.shape != (width,) or upper_array.shape != (width,):
        raise ValueError(f"box bounds must have shape ({width},).")
    if not np.all(lower_array < upper_array):
        raise ValueError("box_lower must be strictly below box_upper.")
    return lower_array, upper_array


def clip_box_halfplanes(
    box_lower: object,
    box_upper: object,
    normals: object,
    offsets: object,
    plane_counts: object,
    /,
) -> tuple[np.ndarray, ...]:
    """Clip a 2D box by each item's convex halfplane set."""

    library = load_meshcore()
    lower, upper = _box(box_lower, box_upper, 2)
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 2
    )
    capacity = normal_array.shape[1] + 4
    if capacity > np.iinfo(np.int32).max:
        raise OverflowError("clipped polygon extent exceeds native int32 range.")
    count = normal_array.shape[0]
    vertices = np.zeros((count, capacity, 2), dtype=np.float64)
    labels = np.zeros((count, capacity), dtype=np.int32)
    vertex_counts = np.zeros((count,), dtype=np.int32)
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_clip_box_halfplanes"](
        count,
        lower.ctypes.data,
        upper.ctypes.data,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        capacity,
        vertices.ctypes.data,
        labels.ctypes.data,
        vertex_counts.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "clip_box_halfplanes")
    return vertices, labels, vertex_counts, area, moment, status


def clip_box_halfspaces(
    box_lower: object,
    box_upper: object,
    normals: object,
    offsets: object,
    plane_counts: object,
    /,
) -> tuple[np.ndarray, ...]:
    """Clip a 3D box by each item's convex halfspace set."""

    library = load_meshcore()
    lower, upper = _box(box_lower, box_upper, 3)
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    faces_bound = normal_array.shape[1] + 6
    vertices_bound = 2 * faces_bound - 4
    face_vertex_bound = 2 * (vertices_bound + faces_bound - 2)
    maximum = np.iinfo(np.int32).max
    if max(faces_bound, vertices_bound, face_vertex_bound) > maximum:
        raise OverflowError("clipped polyhedron extent exceeds native int32 range.")
    count = normal_array.shape[0]
    vertices = np.zeros((count, vertices_bound, 3), dtype=np.float64)
    vertex_counts = np.zeros((count,), dtype=np.int32)
    face_offsets = np.zeros((count, faces_bound + 1), dtype=np.int32)
    face_labels = np.zeros((count, faces_bound), dtype=np.int32)
    face_vertices = np.zeros((count, face_vertex_bound), dtype=np.int32)
    face_counts = np.zeros((count,), dtype=np.int32)
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_clip_box_halfspaces"](
        count,
        lower.ctypes.data,
        upper.ctypes.data,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        vertices_bound,
        faces_bound,
        face_vertex_bound,
        vertices.ctypes.data,
        vertex_counts.ctypes.data,
        face_offsets.ctypes.data,
        face_labels.ctypes.data,
        face_vertices.ctypes.data,
        face_counts.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "clip_box_halfspaces")
    return (
        vertices,
        vertex_counts,
        face_offsets,
        face_labels,
        face_vertices,
        face_counts,
        volume,
        moment,
        status,
    )


# ---------------------------------------------------------------- triangulations


def _read_mesh(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # The native handle is released even when a copy fails.
    try:
        dimension = library["phx_mc_mesh_dimension"](handle)
        point_count = library["phx_mc_mesh_point_count"](handle)
        cell_count = library["phx_mc_mesh_cell_count"](handle)
        input_count = library["phx_mc_mesh_input_point_count"](handle)
        if (
            dimension not in (2, 3)
            or point_count < 0
            or point_count >= np.iinfo(np.int32).max
            or cell_count < 0
            or input_count < 0
            or input_count >= np.iinfo(np.int32).max
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "mesh handle reported invalid output sizes"
            )
        points = np.zeros((point_count, dimension), dtype=np.float64)
        cells = np.zeros((cell_count, dimension + 1), dtype=np.int32)
        vertex_map = np.zeros((input_count,), dtype=np.int32)
        cell_segments = np.zeros((cell_count, 3), dtype=np.int32)
        constraint_sources = np.empty((point_count,), dtype=np.int32)
        constraint_parents = np.empty((point_count, 2), dtype=np.int32)
        constraint_ratios = np.empty((point_count,), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_mesh_copy_points"](handle, point_count, points.ctypes.data),
            "mesh_copy_points",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_cells"](handle, cell_count, cells.ctypes.data),
            "mesh_copy_cells",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_vertex_map"](
                handle, input_count, vertex_map.ctypes.data
            ),
            "mesh_copy_vertex_map",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_cell_segments"](
                handle, cell_count, cell_segments.ctypes.data
            ),
            "mesh_copy_cell_segments",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_constraint_sources"](
                handle, point_count, constraint_sources.ctypes.data
            ),
            "mesh_copy_constraint_sources",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_constraint_parents"](
                handle, point_count, constraint_parents.ctypes.data
            ),
            "mesh_copy_constraint_parents",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_constraint_ratios"](
                handle, point_count, constraint_ratios.ctypes.data
            ),
            "mesh_copy_constraint_ratios",
        )
    finally:
        library["phx_mc_mesh_free"](handle)
    return (
        points,
        cells,
        vertex_map,
        cell_segments,
        constraint_sources,
        constraint_parents,
        constraint_ratios,
    )


def _triangulate(
    name: str,
    points: object,
    weights: object | None,
    width: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    library = load_meshcore()
    point_array = _points(points, "points", width)
    count = point_array.shape[0]
    handle = ctypes.c_void_p()
    if weights is None:
        status = library[name](count, point_array.ctypes.data, ctypes.addressof(handle))
    else:
        weight_array = _real_array(weights, "weights")
        if weight_array.shape != (count,):
            raise ValueError("weights must have shape (n,).")
        status = library[name](
            count,
            point_array.ctypes.data,
            weight_array.ctypes.data,
            ctypes.addressof(handle),
        )
    operation = name.removeprefix("phx_mc_")
    if status != MeshcoreStatus.OK:
        _raise_for_call(status, operation)
    if not handle.value:
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, f"{operation}: missing mesh")
    _, cells, vertex_map, _, _, _, _ = _read_mesh(library, handle)
    return cells, vertex_map


def delaunay_2d(points: object, /) -> tuple[np.ndarray, np.ndarray]:
    """Exact Delaunay triangulation of ``(n, 2)`` points.

    Returns ``(triangles (T, 3) int32, vertex_map (n,) int32)``; triangles are
    counterclockwise, canonically ordered, and cocircular ties are resolved by
    index-ordered symbolic perturbation. ``vertex_map`` sends duplicates to
    their smallest identical index.
    """

    return _triangulate("phx_mc_delaunay_2d", points, None, 2)


def delaunay_3d(points: object, /) -> tuple[np.ndarray, np.ndarray]:
    """Exact Delaunay tetrahedralization of ``(n, 3)`` points.

    Returns positively oriented, canonically ordered tetrahedra and the input
    vertex map.
    """

    return _triangulate("phx_mc_delaunay_3d", points, None, 3)


def regular_2d(points: object, weights: object, /) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) triangulation; redundant points map to -1."""

    return _triangulate("phx_mc_regular_2d", points, weights, 2)


def regular_3d(points: object, weights: object, /) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) tetrahedralization; redundant points map to -1."""

    return _triangulate("phx_mc_regular_3d", points, weights, 3)


class RestrictedPowerCells(NamedTuple):
    """Restricted power cells, piece ancestry, and vertex construction witnesses."""

    vertices: np.ndarray
    face_offsets: np.ndarray
    face_vertices: np.ndarray
    face_cells: np.ndarray
    face_facets: np.ndarray
    cell_sites: np.ndarray
    cell_regions: np.ndarray
    cell_volumes: np.ndarray
    cell_moments: np.ndarray
    cell_second_moments: np.ndarray
    piece_sites: np.ndarray
    piece_tets: np.ndarray
    piece_cells: np.ndarray
    piece_volumes: np.ndarray
    piece_moments: np.ndarray
    vertex_carriers: np.ndarray
    vertex_site_offsets: np.ndarray
    vertex_sites: np.ndarray


class ContactIntersections(NamedTuple):
    classes: np.ndarray
    point_counts: np.ndarray
    points: np.ndarray
    point_bounds: np.ndarray
    point_features: np.ndarray
    status: np.ndarray


# The native arrangement ABI leaves own/coincident-operand windings unclassified.
ARRANGEMENT_UNCLASSIFIED = np.iinfo(np.int32).min


class ArrangementClassification(NamedTuple):
    status: MeshcoreStatus
    fragment_components: np.ndarray
    component_windings: np.ndarray


class NativeArrangement(NamedTuple):
    vertices: np.ndarray
    bounds: np.ndarray
    constructions: np.ndarray
    origins: np.ndarray
    fragments: np.ndarray
    fragment_faces: np.ndarray
    contact_edges: np.ndarray
    coplanar: np.ndarray
    locations: np.ndarray
    classification: ArrangementClassification | None


class NativeReferenceTriangleClip(NamedTuple):
    vertex_count: int
    supporting_planes: np.ndarray
    edge_labels: np.ndarray


class NativeReferenceTetrahedronClip(NamedTuple):
    supporting_planes: np.ndarray
    vertex_incident_offsets: np.ndarray
    vertex_incident_planes: np.ndarray
    face_offsets: np.ndarray
    face_planes: np.ndarray
    face_vertices: np.ndarray


class NativeTetrahedraArrangement(NamedTuple):
    vertex_words: np.ndarray
    vertex_word_offsets: np.ndarray
    vertex_signs: np.ndarray
    vertex_input_index: np.ndarray
    job_face_offsets: np.ndarray
    face_offsets: np.ndarray
    face_vertices: np.ndarray
    face_planes: np.ndarray


class PlcSuppressionSearch(NamedTuple):
    """Evidence of the exhaustive Steiner-free suppression search.

    The search runs only for wedges that no polynomial fill handles:
    wedges searched and filled, states expanded in total and at most per
    wedge, and the largest searched wedge in cells. Wedges with no fill of
    their own are merged with neighbouring cells layer by layer: wedges
    filled that way and the most layers one needed, and wedges whose
    constraint-delimited region was exhausted without a fill with the
    largest such region in cells. Last, the gift-wrapped constrained
    Delaunay fills: how many ran and the cells they laid in total and at
    most in one fill (one step per cell), a deterministic work measure.
    """

    searches: int
    fills: int
    nodes: int
    max_nodes: int
    max_wedge_cells: int
    merges: int
    max_merge_layers: int
    exhausted: int
    max_exhausted_cells: int
    wraps: int
    wrap_cells: int
    max_wrap_cells: int


class PlcSourceData(NamedTuple):
    """PLC result; nonempty witnesses index raw carrier rows, not source banks.

    Stratum 1 entities index ``plc_edges``; stratum 2 entities index
    ``input_triangles``. Stratum 0 uses entity ``-1``.
    """

    status: MeshcoreStatus
    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    face_orientations: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    protection: np.ndarray
    vertex_dimensions: np.ndarray
    plc_edges: np.ndarray
    input_triangles: np.ndarray
    input_polygons: np.ndarray
    source_strata: np.ndarray
    source_entities: np.ndarray
    source_parameters: np.ndarray
    source_deviations: np.ndarray
    suppression_search: PlcSuppressionSearch


class TetMeshEditKind(IntEnum):
    SPLIT = 0
    COLLAPSE = 1
    FLIP23 = 2
    FLIP32 = 3
    RECONNECT = 4
    RELOCATE = 5
    REFINE_CELL = 6
    RETRIANGULATE = 7


class TetMeshEditStatus(IntEnum):
    APPLIED = 0
    REJECTED_TOPOLOGY = 1
    REJECTED_PROTECTED = 2
    INVALID_GEOMETRY = 3
    UNCERTAIN_PREDICATE = 4
    STALE_GENERATION = 5
    ALLOCATION_FAILURE = 6
    INDEX_OVERFLOW = 7


class PlcRecovery3D(NamedTuple):
    """Recovered PLC; witness entities index raw carrier rows by stratum.

    Stratum 1 indexes ``plc_edges`` and stratum 2 indexes ``input_triangles``.
    """

    status: MeshcoreStatus
    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    face_orientations: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    protection_radii: np.ndarray
    vertex_dimensions: np.ndarray
    plc_edges: np.ndarray
    input_triangles: np.ndarray
    input_polygons: np.ndarray
    witness_strata: np.ndarray
    witness_entities: np.ndarray
    witness_parameters: np.ndarray
    witness_deviations: np.ndarray
    suppression_search: PlcSuppressionSearch


class TetMeshSourceComplex(NamedTuple):
    """Persistent source banks whose witnesses index canonical rows.

    Stratum 1 entities index ``segments``; strata 2 and 3 entities index
    ``faces``. ``face_planes`` declares an exact plane per face row (origin,
    first axis, second axis; an all-NaN row is undeclared; every row of a face
    group declares the same plane or none) with ``face_plane_tolerances``. A
    stratum-3 witness is the point ``origin + s * first_axis + t * second_axis``
    of its row's declared plane, ``(s, t)`` its parameters.
    """

    points: np.ndarray
    faces: np.ndarray
    face_group_rows: np.ndarray
    face_tolerances: np.ndarray
    segments: np.ndarray
    segment_group_rows: np.ndarray
    segment_tolerances: np.ndarray
    witness_strata: np.ndarray
    witness_entities: np.ndarray
    witness_parameters: np.ndarray
    face_ids: np.ndarray
    segment_ids: np.ndarray
    face_planes: np.ndarray
    face_plane_tolerances: np.ndarray


class TetMeshIdentity(NamedTuple):
    generation: int
    identity_bound: bool
    source_id: str
    source_revision: str
    source_map_id: str
    frame_id: str


class TetMeshPeriodicEntityOrbitMap(NamedTuple):
    dimension: int
    arity: int
    rank: int
    lifted_entity_ids: np.ndarray
    quotient_entity_ids: np.ndarray
    lifted_to_quotient: np.ndarray
    orientation_signs: np.ndarray
    winding_shifts: np.ndarray
    corner_permutations: np.ndarray
    representative_lifted_rows: np.ndarray


class TetMeshPeriodicIdentityMap(NamedTuple):
    source_mesh_id: str
    source_topology_id: str
    lifted_topology_id: str
    periodic_topology_id: str
    cell_entity_set_id: str
    edge_entity_set_id: str
    entity_maps: tuple[TetMeshPeriodicEntityOrbitMap, ...]


class TetMeshPeriodicMember(NamedTuple):
    vertex_id: int
    winding: np.ndarray
    representative_to_member: np.ndarray


class TetMeshPeriodicOrbit(NamedTuple):
    orbit_id: int
    representative_vertex_id: int
    members: tuple[TetMeshPeriodicMember, ...]


class TetMeshPeriodicQuotientCell(NamedTuple):
    orbit_id: int
    cell_id: int
    representative_vertices: np.ndarray
    winding: np.ndarray
    region: int
    lifted_cell_ids: np.ndarray


class TetMeshPeriodicState(NamedTuple):
    identity_map: TetMeshPeriodicIdentityMap
    orbits: tuple[TetMeshPeriodicOrbit, ...]
    quotient_cells: tuple[TetMeshPeriodicQuotientCell, ...]


@dataclass(frozen=True, slots=True)
class TetMeshPeriodicEdit:
    periodic_members: tuple[TetMeshPeriodicMember, ...]
    removed_periodic_orbit_ids: object
    replacement_periodic_orbits: tuple[TetMeshPeriodicOrbit, ...]
    removed_quotient_cell_ids: object
    replacement_quotient_cells: tuple[TetMeshPeriodicQuotientCell, ...]
    target_identity_map: TetMeshPeriodicIdentityMap


class TetMeshPeriodicSnapshot(NamedTuple):
    identity_map: TetMeshPeriodicIdentityMap
    orbits: tuple[TetMeshPeriodicOrbit, ...]
    quotient_cells: tuple[TetMeshPeriodicQuotientCell, ...]


class TetMeshArrays(NamedTuple):
    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    vertex_dimensions: np.ndarray
    protection_radii: np.ndarray
    protected_vertices: np.ndarray
    vertex_edge_witness_present: np.ndarray
    vertex_edge_witness_endpoints: np.ndarray
    vertex_edge_parameters: np.ndarray
    vertex_ids: np.ndarray
    cell_ids: np.ndarray
    facet_ids: np.ndarray
    facet_orientations: np.ndarray
    protected_facets: np.ndarray
    segment_ids: np.ndarray
    protected_segments: np.ndarray
    source_points: np.ndarray
    source_faces: np.ndarray
    source_face_group_rows: np.ndarray
    source_face_ids: np.ndarray
    source_face_tolerances: np.ndarray
    source_segments: np.ndarray
    source_segment_group_rows: np.ndarray
    source_segment_ids: np.ndarray
    source_segment_tolerances: np.ndarray
    source_face_planes: np.ndarray
    source_face_plane_tolerances: np.ndarray
    witness_strata: np.ndarray
    witness_entities: np.ndarray
    witness_parameters: np.ndarray
    witness_deviations: np.ndarray
    generation: int
    identity_bound: bool
    source_id: str
    source_revision: str
    source_map_id: str
    frame_id: str
    vertex_ancestry_offsets: np.ndarray
    vertex_ancestor_ids: np.ndarray
    vertex_ancestor_winding: np.ndarray
    vertex_ancestry_weights: np.ndarray
    cell_ancestry_offsets: np.ndarray
    cell_ancestor_ids: np.ndarray
    cell_ancestry_edit_kinds: np.ndarray
    periodic: TetMeshPeriodicSnapshot | None


class TetMeshSourceEvidence(NamedTuple):
    requested_bound: float
    achieved_bound: float
    witness_strata: np.ndarray
    witness_entities: np.ndarray
    witness_parameters: np.ndarray
    witness_deviations: np.ndarray


class TetMeshPlaneRecords(NamedTuple):
    """One row per (vertex, declared face group) incidence of the constrained facets.

    ``face_rows`` is the group's first source face row, ``parameters`` the
    correctly rounded parameters of the carrier's orthogonal projection onto
    the declared plane, ``deviations`` an outward bound of its distance to it.
    """

    vertex_ids: np.ndarray
    face_rows: np.ndarray
    parameters: np.ndarray
    deviations: np.ndarray


class TetMeshPlaneRefusal(MeshcoreError):
    """A boundary vertex refused its facet's declared exact plane.

    The plane neither meets the closed rounding box of the vertex carrier nor
    lies within the declared tolerance of it: ``vertex`` is the input vertex
    row, ``face_row`` the plane's source face row, ``parameters`` and
    ``deviation`` the carrier's record on the plane.
    """

    def __init__(
        self,
        status: MeshcoreStatus,
        vertex: int,
        face_row: int,
        parameters: tuple[float, float],
        deviation: float,
        tolerance: float,
        /,
    ) -> None:
        super().__init__(
            status,
            f"declared plane of source face row {face_row} refused vertex {vertex} "
            f"(deviation {deviation!r} > tolerance {tolerance!r})",
        )
        self.vertex = vertex
        self.face_row = face_row
        self.parameters = parameters
        self.deviation = deviation
        self.tolerance = tolerance


class TetMeshUnmetReason(IntEnum):
    FIXED_BOUNDARY = 0
    PROTECTED = 1
    NONREPRESENTABLE = 2
    REFUSED = 3
    NO_IMPROVEMENT = 4
    # Conforming refinement left the cell: the boundary split it required lies
    # in the protected region of an acute input feature (segments, or a segment
    # and a facet, meeting below 90 degrees, or facets meeting at a dihedral
    # angle below 90 degrees through the domain).
    INPUT_ANGLE = 5


# Apex vertex ID of an unmet row without INPUT_ANGLE evidence.
TET_MESH_UNMET_NO_APEX = np.iinfo(np.uint64).max


class TetMeshUnmet(NamedTuple):
    cell_ids: np.ndarray
    criteria: np.ndarray
    reasons: np.ndarray
    values: np.ndarray
    # INPUT_ANGLE rows: the acute feature's apex vertex ID (a ridge's first
    # endpoint) and angle in degrees; TET_MESH_UNMET_NO_APEX and NaN otherwise.
    apex_vertex_ids: np.ndarray
    apex_angles: np.ndarray


class TetMeshRun(NamedTuple):
    edit_status: TetMeshEditStatus
    status: MeshcoreStatus
    inserted_vertices: int
    collapsed_vertices: int
    relocated_vertices: int
    unmet_cells: int
    # Improvement scores are floored to multiples of this quantum (2^-20), the
    # acceptance grid that bounds accepted edits; None for refinement runs.
    score_quantum: float | None = None


@dataclass(frozen=True, slots=True)
class TetMeshEdit:
    kind: TetMeshEditKind
    expected_generation: int
    source_id: str | None = None
    source_revision: str | None = None
    source_map_id: str | None = None
    frame_id: str | None = None
    collapse_pairs: object | None = None
    split_edges: object | None = None
    removed_vertex_ids: object | None = None
    removed_cell_ids: object | None = None
    changed_vertex_ids: object | None = None
    changed_vertex_points: object | None = None
    changed_witness_strata: object | None = None
    changed_witness_entities: object | None = None
    changed_witness_parameters: object | None = None
    changed_witness_deviations: object | None = None
    changed_protection_radii: object | None = None
    changed_protected_vertices: object | None = None
    changed_edge_witness_present: object | None = None
    changed_edge_witness_endpoints: object | None = None
    changed_edge_parameters: object | None = None
    changed_vertex_ancestry_offsets: object | None = None
    changed_vertex_ancestor_ids: object | None = None
    changed_vertex_ancestor_winding: object | None = None
    changed_vertex_ancestry_weights: object | None = None
    replacement_cell_ancestry_offsets: object | None = None
    replacement_cell_ancestor_ids: object | None = None
    replacement_cell_ids: object | None = None
    replacement_cell_vertices: object | None = None
    replacement_cell_facet_sources: object | None = None
    replacement_regions: object | None = None
    removed_facet_ids: object | None = None
    replacement_facet_ids: object | None = None
    replacement_facet_vertices: object | None = None
    replacement_facet_sources: object | None = None
    replacement_facet_orientations: object | None = None
    replacement_protected_facets: object | None = None
    removed_segment_ids: object | None = None
    replacement_segment_ids: object | None = None
    replacement_segment_vertices: object | None = None
    replacement_segment_sources: object | None = None
    replacement_protected_segments: object | None = None
    periodic: TetMeshPeriodicEdit | None = None


class ArrangementFailure(MeshcoreError):
    def __init__(self, status: MeshcoreStatus, failed_face: int, /) -> None:
        super().__init__(status, f"arrange_triangles: failed face {failed_face}")
        self.failed_face = failed_face


class PeriodicTriangulation(NamedTuple):
    cells: np.ndarray
    shifts: np.ndarray


class PeriodicLocalTriangulation(NamedTuple):
    """Owner-local periodic cells, exact support bounds, and stable identities."""

    vertices: np.ndarray
    shifts: np.ndarray
    cell_ids: np.ndarray
    cell_key_ids: np.ndarray
    cell_key_relative_shifts: np.ndarray
    cell_owner_ranks: np.ndarray
    support_cell_mask: np.ndarray
    support_fractional_lower: np.ndarray
    support_fractional_upper: np.ndarray
    certified_margin: float
    has_owned_sites: bool
    owner_fractional_lower: np.ndarray | None
    owner_fractional_upper: np.ndarray | None


class PeriodicSupportHits(NamedTuple):
    """Exact remote point images inside or cospherical with requested cells."""

    cell_rows: np.ndarray
    point_global_ids: np.ndarray
    image_shifts: np.ndarray
    predicate_signs: np.ndarray
    candidate_revision: int


class PeriodicRefinementResult(NamedTuple):
    status: MeshcoreStatus
    points: np.ndarray
    cells: np.ndarray
    shifts: np.ndarray


class RestrictedDual3D(NamedTuple):
    facets: np.ndarray
    cells: np.ndarray
    endpoints: np.ndarray
    kinds: np.ndarray
    status: np.ndarray


def _restricted_tetrahedra(
    points: object,
    tetrahedra: object,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _points(points, "points", 3)
    cells = _native_integer_array(tetrahedra, "tetrahedra", np.dtype(np.int32))
    if (
        cells.ndim != 2
        or cells.shape[1] != 4
        or np.any(cells < 0)
        or np.any(cells >= coordinates.shape[0])
    ):
        raise ValueError("tetrahedra must contain valid four-vertex point indices.")
    return coordinates, cells


def _restricted_domain(
    carrier_lower: object,
    carrier_upper: object,
    /,
) -> np.ndarray:
    lower = _real_array(carrier_lower, "carrier_lower")
    upper = _real_array(carrier_upper, "carrier_upper")
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("carrier_lower and carrier_upper must have shape (3,).")
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("carrier bounds must be finite and strictly ordered.")
    return np.ascontiguousarray(np.concatenate((lower, upper)), dtype=np.float64)


def restricted_dual_3d(
    points: object,
    tetrahedra: object,
    carrier_lower: object,
    carrier_upper: object,
    /,
) -> RestrictedDual3D:
    """Return numerical restricted dual candidates; statuses are not certificates."""
    coordinates, cells = _restricted_tetrahedra(points, tetrahedra)
    bounds = _restricted_domain(carrier_lower, carrier_upper)
    count = cells.shape[0]
    capacity = 4 * count
    if count == 0:
        return RestrictedDual3D(
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    if capacity > np.iinfo(np.int64).max:
        raise OverflowError("The restricted dual facet extent exceeds int64.")
    facets = np.empty((capacity, 3), dtype=np.int32)
    incident = np.empty((capacity, 2), dtype=np.int32)
    endpoints = np.empty((capacity, 2, 3), dtype=np.float64)
    kinds = np.empty((capacity,), dtype=np.int32)
    item_status = np.empty((capacity,), dtype=np.int32)
    facet_count = np.zeros((1,), dtype=np.int64)
    status = load_meshcore()["phx_mc_restricted_dual_3d"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        count,
        cells.ctypes.data,
        bounds.ctypes.data,
        capacity,
        facets.ctypes.data,
        incident.ctypes.data,
        endpoints.ctypes.data,
        kinds.ctypes.data,
        item_status.ctypes.data,
        facet_count.ctypes.data,
    )
    _raise_for_call(status, "restricted_dual_3d")
    used = int(facet_count[0])
    if not 0 <= used <= capacity:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "restricted_dual_3d returned an invalid facet count",
        )
    return RestrictedDual3D(
        facets[:used],
        incident[:used],
        endpoints[:used],
        kinds[:used],
        item_status[:used],
    )


def restricted_centers_3d(
    points: object,
    tetrahedra: object,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outward circumcenter intervals and per-tetrahedron statuses."""
    coordinates, cells = _restricted_tetrahedra(points, tetrahedra)
    bounds = np.empty((cells.shape[0], 2, 3), dtype=np.float64)
    item_status = np.empty((cells.shape[0],), dtype=np.int32)
    if cells.shape[0] == 0:
        return bounds, item_status
    status = load_meshcore()["phx_mc_restricted_centers_3d"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        cells.shape[0],
        cells.ctypes.data,
        bounds.ctypes.data,
        item_status.ctypes.data,
    )
    _raise_for_call(status, "restricted_centers_3d")
    return bounds, item_status


def restricted_rays_3d(
    points: object,
    tetrahedra: object,
    center_bounds: object,
    facets: object,
    incident_cells: object,
    carrier_lower: object,
    carrier_upper: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound hull dual rays; each incident-cell row retains both cell slots."""
    coordinates, cells = _restricted_tetrahedra(points, tetrahedra)
    centers = _real_array(center_bounds, "center_bounds")
    faces = _native_integer_array(facets, "facets", np.dtype(np.int32))
    owners = _native_integer_array(incident_cells, "incident_cells", np.dtype(np.int32))
    bounds = _restricted_domain(carrier_lower, carrier_upper)
    if (
        centers.shape != (cells.shape[0], 2, 3)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or owners.shape != (faces.shape[0], 2)
        or np.any(faces < 0)
        or np.any(faces >= coordinates.shape[0])
        or np.any(owners[:, 0] < 0)
        or np.any(owners[:, 0] >= cells.shape[0])
    ):
        raise ValueError("restricted ray facets, incident cells, and centers must align.")
    ray_count = faces.shape[0]
    endpoints = np.empty((ray_count, 2, 2, 3), dtype=np.float64)
    kinds = np.empty((ray_count,), dtype=np.int32)
    item_status = np.empty((ray_count,), dtype=np.int32)
    if ray_count == 0:
        return endpoints, kinds, item_status
    status = load_meshcore()["phx_mc_restricted_rays_3d"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        cells.shape[0],
        cells.ctypes.data,
        centers.ctypes.data,
        ray_count,
        faces.ctypes.data,
        owners.ctypes.data,
        bounds.ctypes.data,
        endpoints.ctypes.data,
        kinds.ctypes.data,
        item_status.ctypes.data,
    )
    _raise_for_call(status, "restricted_rays_3d")
    return endpoints, kinds, item_status


class SurfaceReconnectResult(NamedTuple):
    triangles: np.ndarray
    constrained: np.ndarray
    vertex_ids: np.ndarray
    item_status: np.ndarray
    triangle_count: int


class SurfaceEnvelopeResult(NamedTuple):
    bounds: np.ndarray
    surface_vertices: np.ndarray
    surface_triangles: np.ndarray
    volume_vertices: np.ndarray
    tetrahedra: np.ndarray


# Periodic kernels also bound lattice (fractional) coordinates by 2**30 so that
# integer image shifts stay exact.
_PERIODIC_RANGE_CONTRACT = (
    "points and lattice entries must be zero or have magnitude in [2**-120, 2**120], "
    "and every fractional (lattice) coordinate must have magnitude at most 2**30."
)


def _surface_envelope_range_contract(
    points: np.ndarray, offset: float, spacing: float, certificate_spacing: float, /
) -> str:
    """State the surface-envelope contract behind a RANGE_ERROR status.

    The kernel certifies the envelope only when ``offset`` strictly exceeds its
    certified error: the grid-cell diameter ``sqrt(3) * spacing``, the source
    covering radius (at most ``certificate_spacing`` plus roundoff) and the
    arithmetic error ``1024 * eps * scale``, where ``scale`` bounds the source
    coordinates and ``offset + spacing``. This recomputes that requirement in
    binary64 to report it; an offset that clears it failed instead because the
    error bound, the distance field or an envelope coordinate left the binary64
    exact-evaluation domain.
    """
    scale = max(offset + spacing, float(np.max(np.abs(points))))
    arithmetic = 1024.0 * float(np.finfo(np.float64).eps) * scale
    required = float(np.sqrt(3.0)) * spacing + certificate_spacing + arithmetic
    if offset <= required:
        return (
            f"offset {offset:.17g} does not dominate the certified envelope error "
            f"~{required:.17g} (sqrt(3) * spacing {np.sqrt(3.0) * spacing:.17g} + "
            f"certificate_spacing {certificate_spacing:.17g} + arithmetic error "
            f"{arithmetic:.3g}); increase the offset or decrease spacing and "
            "certificate_spacing."
        )
    return (
        f"offset {offset:.17g} dominates the nominal envelope error ~{required:.17g}, "
        "but the certified error bound, the sampled distance field or an envelope "
        "coordinate left the exact-evaluation domain (finite, zero or magnitude in "
        "[2**-120, 2**120]); the offset may also fall short of the error once "
        "roundoff is bounded."
    )


def surface_envelope(
    vertices: object,
    triangles: object,
    offset: float,
    spacing: float,
    certificate_spacing: float,
    /,
) -> SurfaceEnvelopeResult:
    """Tetrahedralize an explicit unsigned-distance envelope around a triangle soup.

    ``offset`` is the requested envelope radius. The native kernel returns its
    independent source/envelope deviation, envelope-distance interval, and
    interpolation error in ``bounds``; it refuses when the offset cannot
    dominate those certified errors. This does not infer an interior from an
    unsigned zero set.
    """
    points = np.ascontiguousarray(_points(vertices, "vertices", 3), dtype=np.float64)
    faces = np.ascontiguousarray(
        _native_integer_array(triangles, "triangles", np.dtype(np.int64)),
        dtype=np.int64,
    )
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("triangles must be a nonempty (N, 3) integer array.")
    if points.shape[0] == 0 or np.any(faces < 0) or np.any(faces >= points.shape[0]):
        raise ValueError("triangles must index the supplied source vertices.")
    values = tuple(float(value) for value in (offset, spacing, certificate_spacing))
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "offset, spacing, and certificate_spacing must be finite and positive."
        )

    bounds = np.empty((4,), dtype=np.float64)
    handle = ctypes.c_void_p()
    request = _SurfaceEnvelopeCreateRequest(
        points.ctypes.data,
        points.shape[0],
        faces.ctypes.data,
        faces.shape[0],
        values[0],
        values[1],
        values[2],
        ctypes.addressof(handle),
        bounds.ctypes.data,
    )
    library = load_meshcore()
    status = library["phx_mc_surface_envelope_create"](ctypes.byref(request))
    _raise_for_call(
        status,
        "surface_envelope_create",
        # The certified-error contract is formatted only when it failed.
        contracts={
            MeshcoreStatus.RANGE_ERROR: _surface_envelope_range_contract(points, *values)
        }
        if status == MeshcoreStatus.RANGE_ERROR
        else None,
    )
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "surface_envelope_create returned no owned result handle.",
        )
    try:
        sizes = np.empty((4,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_surface_envelope_sizes"](handle, sizes.ctypes.data),
            "surface_envelope_sizes",
        )
        if np.any(sizes < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "surface_envelope_sizes returned a negative array extent.",
            )
        surface_vertex_count, triangle_count, volume_vertex_count, tetrahedron_count = (
            int(value) for value in sizes.tolist()
        )
        index_limit = int(np.iinfo(np.intp).max)
        if (
            surface_vertex_count > index_limit // 3
            or triangle_count > index_limit // 3
            or volume_vertex_count > index_limit // 3
            or tetrahedron_count > index_limit // 4
        ):
            raise MeshcoreError(
                MeshcoreStatus.RANGE_ERROR,
                "surface_envelope output extents exceed addressable NumPy arrays.",
            )
        surface_vertices = np.empty((surface_vertex_count, 3), dtype=np.float64)
        surface_triangles = np.empty((triangle_count, 3), dtype=np.int64)
        volume_vertices = np.empty((volume_vertex_count, 3), dtype=np.float64)
        tetrahedra = np.empty((tetrahedron_count, 4), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_surface_envelope_arrays"](
                handle,
                surface_vertex_count,
                triangle_count,
                volume_vertex_count,
                tetrahedron_count,
                surface_vertices.ctypes.data,
                surface_triangles.ctypes.data,
                volume_vertices.ctypes.data,
                tetrahedra.ctypes.data,
            ),
            "surface_envelope_arrays",
        )
        if (
            surface_vertices.shape[0] == 0
            or surface_triangles.shape[0] == 0
            or volume_vertices.shape[0] == 0
            or tetrahedra.shape[0] == 0
            or np.any(~np.isfinite(bounds))
            or np.any(~np.isfinite(surface_vertices))
            or np.any(~np.isfinite(volume_vertices))
            or np.any(surface_triangles < 0)
            or np.any(surface_triangles >= surface_vertices.shape[0])
            or np.any(tetrahedra < 0)
            or np.any(tetrahedra >= volume_vertices.shape[0])
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "surface_envelope returned invalid geometry or error evidence.",
            )
        for output in (
            bounds,
            surface_vertices,
            surface_triangles,
            volume_vertices,
            tetrahedra,
        ):
            output.setflags(write=False)
        return SurfaceEnvelopeResult(
            bounds,
            surface_vertices,
            surface_triangles,
            volume_vertices,
            tetrahedra,
        )
    finally:
        library["phx_mc_surface_envelope_free"](handle)


class LevelSetSplitResult(NamedTuple):
    status: MeshcoreStatus
    points: np.ndarray
    cells: np.ndarray
    parents: np.ndarray
    sources: np.ndarray
    weights: np.ndarray
    deviations: np.ndarray | None
    failed_edge: int


def periodic_delaunay(
    points: object,
    fractional: object,
    lattice: object,
    inverse: object,
    /,
    *,
    initial_margin: float = 1.0,
) -> PeriodicTriangulation:
    """Build a complete native periodic Delaunay quotient and its image shifts."""

    point_array = _real_array(points, "points")
    if point_array.ndim != 2 or point_array.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3).")
    dimension = point_array.shape[1]
    fractional_array = _real_array(fractional, "fractional")
    lattice_array = _real_array(lattice, "lattice")
    inverse_array = _real_array(inverse, "inverse")
    if (
        fractional_array.shape != point_array.shape
        or lattice_array.shape != (dimension, dimension)
        or inverse_array.shape != (dimension, dimension)
    ):
        raise ValueError(
            "periodic points, fractional coordinates, and lattice arrays are misaligned."
        )
    margin = float(initial_margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("initial_margin must be finite and positive.")
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = library["phx_mc_periodic_delaunay"](
        dimension,
        point_array.shape[0],
        point_array.ctypes.data,
        fractional_array.ctypes.data,
        lattice_array.ctypes.data,
        inverse_array.ctypes.data,
        margin,
        ctypes.addressof(handle),
    )
    if status != MeshcoreStatus.OK:
        _raise_for_call(
            status,
            "periodic_delaunay",
            contracts={MeshcoreStatus.RANGE_ERROR: _PERIODIC_RANGE_CONTRACT},
        )
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "periodic_delaunay: missing handle"
        )
    try:
        cell_count = library["phx_mc_periodic_cell_count"](handle)
        if cell_count < 0:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "negative periodic cell count"
            )
        cells = np.empty((cell_count, dimension + 1), dtype=np.int32)
        shifts = np.empty((cell_count, dimension + 1, dimension), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_periodic_copy_cells"](
                handle, cell_count, cells.ctypes.data, shifts.ctypes.data
            ),
            "periodic_copy_cells",
        )
        return PeriodicTriangulation(cells, shifts)
    finally:
        library["phx_mc_periodic_free"](handle)


def periodic_delaunay_local(
    points: object,
    fractional: object,
    point_global_ids: object,
    point_owner_ranks: object,
    local_owner_rank: int,
    lattice: object,
    inverse: object,
    /,
    *,
    initial_margin: float = 1.0,
) -> PeriodicLocalTriangulation:
    """Triangulate an owner's source/ghost bank with canonical cell identities."""

    point_array = _real_array(points, "points")
    if point_array.ndim != 2 or point_array.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3).")
    dimension = point_array.shape[1]
    fractional_array = _points(fractional, "fractional", dimension)
    lattice_array = _real_array(lattice, "lattice")
    inverse_array = _real_array(inverse, "inverse")
    if (
        fractional_array.shape != point_array.shape
        or lattice_array.shape != (dimension, dimension)
        or inverse_array.shape != (dimension, dimension)
    ):
        raise ValueError(
            "periodic local point, fractional, and lattice arrays are misaligned."
        )
    point_ids = _native_integer_array(
        point_global_ids, "point_global_ids", np.dtype(np.int64)
    )
    point_owners = _native_integer_array(
        point_owner_ranks, "point_owner_ranks", np.dtype(np.int32)
    )
    if point_ids.shape != (point_array.shape[0],) or point_owners.shape != (
        point_array.shape[0],
    ):
        raise ValueError(
            "periodic local point identities and owners must align with points."
        )
    if np.any(point_ids < 0) or np.any(point_owners < 0):
        raise ValueError(
            "periodic local point identities and owners must be nonnegative."
        )
    if isinstance(local_owner_rank, (bool, np.bool_)) or not isinstance(
        local_owner_rank, (int, np.integer)
    ):
        raise TypeError("local_owner_rank must be an integer excluding bool.")
    owner_rank = int(local_owner_rank)
    if owner_rank < 0:
        raise ValueError("local_owner_rank must be nonnegative.")
    if owner_rank > np.iinfo(np.int32).max:
        raise OverflowError("local_owner_rank exceeds the native int32 range.")
    margin = float(initial_margin)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("initial_margin must be finite and positive.")

    library = load_meshcore()
    handle = ctypes.c_void_p()
    request = _PeriodicLocalDelaunayRequest(
        dimension=dimension,
        point_count=point_array.shape[0],
        points=point_array.ctypes.data,
        fractional=fractional_array.ctypes.data,
        point_global_ids=point_ids.ctypes.data,
        point_owner_ranks=point_owners.ctypes.data,
        local_owner_rank=owner_rank,
        lattice=lattice_array.ctypes.data,
        inverse=inverse_array.ctypes.data,
        initial_margin=margin,
        result=ctypes.addressof(handle),
    )
    status = library["phx_mc_periodic_delaunay_local"](ctypes.byref(request))
    if status != MeshcoreStatus.OK:
        _raise_for_call(status, "periodic_delaunay_local")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "periodic_delaunay_local: missing handle"
        )
    try:
        cell_count = np.zeros((1,), dtype=np.int64)
        certified_margin = np.zeros((1,), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_periodic_local_delaunay_sizes"](
                handle, cell_count.ctypes.data, certified_margin.ctypes.data
            ),
            "periodic_local_delaunay_sizes",
        )
        count = int(cell_count[0])
        if count < 0:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "negative periodic local cell count"
            )
        has_owned_sites = np.zeros((1,), dtype=np.int8)
        owner_lower = np.empty((dimension,), dtype=np.float64)
        owner_upper = np.empty((dimension,), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_periodic_local_delaunay_owner_bounds"](
                handle,
                has_owned_sites.ctypes.data,
                owner_lower.ctypes.data,
                owner_upper.ctypes.data,
            ),
            "periodic_local_delaunay_owner_bounds",
        )
        if has_owned_sites[0] not in (0, 1):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "periodic_local_delaunay_owner_bounds: invalid ownership flag",
            )
        arity = dimension + 1
        vertices = np.empty((count, arity), dtype=np.int32)
        shifts = np.empty((count, arity, dimension), dtype=np.int32)
        cell_ids = np.empty((count,), dtype=np.int64)
        cell_key_ids = np.empty((count, arity), dtype=np.int64)
        cell_key_relative_shifts = np.empty((count, arity, dimension), dtype=np.int32)
        cell_owner_ranks = np.empty((count,), dtype=np.int32)
        support_cell_mask = np.empty((count,), dtype=np.int8)
        support_lower = np.empty((count, dimension), dtype=np.float64)
        support_upper = np.empty((count, dimension), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_periodic_local_delaunay_copy"](
                handle,
                count,
                vertices.ctypes.data,
                shifts.ctypes.data,
                cell_ids.ctypes.data,
                cell_key_ids.ctypes.data,
                cell_key_relative_shifts.ctypes.data,
                cell_owner_ranks.ctypes.data,
                support_cell_mask.ctypes.data,
                support_lower.ctypes.data,
                support_upper.ctypes.data,
            ),
            "periodic_local_delaunay_copy",
        )
        has_owned = bool(has_owned_sites[0])
        return PeriodicLocalTriangulation(
            vertices,
            shifts,
            cell_ids,
            cell_key_ids,
            cell_key_relative_shifts,
            cell_owner_ranks,
            support_cell_mask,
            support_lower,
            support_upper,
            float(certified_margin[0]),
            has_owned,
            owner_lower if has_owned else None,
            owner_upper if has_owned else None,
        )
    finally:
        library["phx_mc_periodic_local_delaunay_free"](handle)


def periodic_owner_support_query(
    cell_vertex_points: object,
    cell_vertex_global_ids: object,
    cell_vertex_shifts: object,
    remote_owned_points: object,
    remote_owned_ids: object,
    lattice: object,
    candidate_revision: int,
    /,
) -> PeriodicSupportHits:
    """Return every exact in-sphere remote image for owner-local candidate cells."""

    cell_points = _real_array(cell_vertex_points, "cell_vertex_points")
    if (
        cell_points.ndim != 3
        or cell_points.shape[2] not in (2, 3)
        or cell_points.shape[1] != cell_points.shape[2] + 1
    ):
        raise ValueError(
            "cell_vertex_points must have shape (c, d + 1, d) for d in (2, 3)."
        )
    dimension = cell_points.shape[2]
    cell_ids = _native_integer_array(
        cell_vertex_global_ids, "cell_vertex_global_ids", np.dtype(np.int64)
    )
    cell_shifts = _native_integer_array(
        cell_vertex_shifts, "cell_vertex_shifts", np.dtype(np.int32)
    )
    if cell_ids.shape != cell_points.shape[:2] or cell_shifts.shape != cell_points.shape:
        raise ValueError("candidate identities and image shifts must align with cells.")
    if np.any(cell_ids < 0):
        raise ValueError("cell_vertex_global_ids must be nonnegative.")
    remote_points = _points(remote_owned_points, "remote_owned_points", dimension)
    remote_ids = _native_integer_array(
        remote_owned_ids, "remote_owned_ids", np.dtype(np.int64)
    )
    if remote_ids.shape != (remote_points.shape[0],):
        raise ValueError("remote_owned_ids must align with remote_owned_points.")
    if np.any(remote_ids < 0):
        raise ValueError("remote_owned_ids must be nonnegative.")
    lattice_array = _real_array(lattice, "lattice")
    if lattice_array.shape != (dimension, dimension):
        raise ValueError("lattice must have shape (d, d).")
    if isinstance(candidate_revision, (bool, np.bool_)) or not isinstance(
        candidate_revision, (int, np.integer)
    ):
        raise TypeError("candidate_revision must be an integer excluding bool.")
    revision = int(candidate_revision)
    if revision < 0 or revision > np.iinfo(np.uint64).max:
        raise OverflowError("candidate_revision must fit in uint64.")

    library = load_meshcore()
    handle = ctypes.c_void_p()
    request = _PeriodicSupportQueryRequest(
        dimension=dimension,
        cell_count=cell_points.shape[0],
        cell_vertex_points=cell_points.ctypes.data,
        cell_vertex_global_ids=cell_ids.ctypes.data,
        cell_vertex_shifts=cell_shifts.ctypes.data,
        point_count=remote_points.shape[0],
        points=remote_points.ctypes.data,
        point_global_ids=remote_ids.ctypes.data,
        lattice=lattice_array.ctypes.data,
        candidate_revision=revision,
        result=ctypes.addressof(handle),
    )
    status = library["phx_mc_periodic_owner_support_query"](ctypes.byref(request))
    if status != MeshcoreStatus.OK:
        _raise_for_call(status, "periodic_owner_support_query")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "periodic_owner_support_query: missing handle"
        )
    try:
        hit_count = np.zeros((1,), dtype=np.int64)
        echoed_revision = np.zeros((1,), dtype=np.uint64)
        _raise_for_call(
            library["phx_mc_periodic_support_query_sizes"](
                handle, hit_count.ctypes.data, echoed_revision.ctypes.data
            ),
            "periodic_support_query_sizes",
        )
        count = int(hit_count[0])
        if count < 0:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "negative periodic support-hit count"
            )
        if int(echoed_revision[0]) != revision:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "periodic_owner_support_query: candidate revision mismatch",
            )
        cell_rows = np.empty((count,), dtype=np.int64)
        point_ids = np.empty((count,), dtype=np.int64)
        image_shifts = np.empty((count, dimension), dtype=np.int32)
        predicate_signs = np.empty((count,), dtype=np.int8)
        _raise_for_call(
            library["phx_mc_periodic_support_query_copy_hits"](
                handle,
                count,
                cell_rows.ctypes.data,
                point_ids.ctypes.data,
                image_shifts.ctypes.data,
                predicate_signs.ctypes.data,
            ),
            "periodic_support_query_copy_hits",
        )
        return PeriodicSupportHits(
            cell_rows, point_ids, image_shifts, predicate_signs, revision
        )
    finally:
        library["phx_mc_periodic_support_query_free"](handle)


def periodic_refine_2d(
    points: object,
    lattice: object,
    inverse: object,
    vertices: object,
    shifts: object,
    controls: object,
    quantiles: object,
    protected_edges: object,
    absolute_tolerance: float,
    relative_tolerance: float,
    /,
) -> PeriodicRefinementResult:
    """Refine a 2D periodic quotient to its declared metric and rank criteria."""

    point_array = _points(points, "points", 2)
    lattice_array = _real_array(lattice, "lattice")
    inverse_array = _real_array(inverse, "inverse")
    if lattice_array.shape != (2, 2) or inverse_array.shape != (2, 2):
        raise ValueError("lattice and inverse must have shape (2, 2).")
    cell_vertices = _native_integer_array(vertices, "vertices", np.dtype(np.int32))
    if cell_vertices.ndim != 2 or cell_vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n_cells, 3).")
    cell_shifts = _native_integer_array(shifts, "shifts", np.dtype(np.int32))
    if cell_shifts.shape != (cell_vertices.shape[0], 3, 2):
        raise ValueError("shifts must have shape (n_cells, 3, 2).")
    control_array = _real_array(controls, "controls")
    if control_array.ndim != 2 or control_array.shape[1] != 5:
        raise ValueError("controls must have shape (n_controls, 5).")
    quantile_array = _real_array(quantiles, "quantiles")
    if quantile_array.ndim != 1:
        raise ValueError("quantiles must be one-dimensional.")
    protected = _native_integer_array(
        protected_edges, "protected_edges", np.dtype(np.int32)
    )
    if protected.ndim != 2 or protected.shape[1] != 4:
        raise ValueError("protected_edges must have shape (n_edges, 4).")
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    if not (
        np.isfinite(absolute)
        and np.isfinite(relative)
        and absolute >= 0.0
        and relative >= 0.0
    ):
        raise ValueError("periodic tolerances must be finite and nonnegative.")
    request = _PeriodicRefinementInput(
        point_count=point_array.shape[0],
        points=point_array.ctypes.data,
        lattice=lattice_array.ctypes.data,
        inverse=inverse_array.ctypes.data,
        cell_count=cell_vertices.shape[0],
        vertices=cell_vertices.ctypes.data,
        shifts=cell_shifts.ctypes.data,
        control_count=control_array.shape[0],
        controls=control_array.ctypes.data,
        statistic_count=quantile_array.shape[0],
        quantiles=quantile_array.ctypes.data,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        protected_edge_count=protected.shape[0],
        protected_edges=protected.ctypes.data,
    )
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_periodic_refine_2d"](
            ctypes.byref(request), ctypes.addressof(handle)
        )
    )
    if status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
        _raise_for_call(
            status,
            "periodic_refine_2d",
            contracts={MeshcoreStatus.RANGE_ERROR: _PERIODIC_RANGE_CONTRACT},
        )
    if not handle.value:
        raise MeshcoreError(
            status if status != MeshcoreStatus.OK else MeshcoreStatus.INTERNAL_ERROR,
            "periodic_refine_2d: missing result handle",
        )
    try:
        sizes = np.empty((2,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_periodic_refinement_sizes"](handle, sizes.ctypes.data),
            "periodic_refinement_sizes",
        )
        if np.any(sizes < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "negative periodic refinement size"
            )
        point_count, cell_count = sizes
        out_points = np.empty((point_count, 2), dtype=np.float64)
        out_cells = np.empty((cell_count, 3), dtype=np.int32)
        out_shifts = np.empty((cell_count, 3, 2), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_periodic_refinement_copy_points"](
                handle, point_count, out_points.ctypes.data
            ),
            "periodic_refinement_copy_points",
        )
        _raise_for_call(
            library["phx_mc_periodic_refinement_copy_cells"](
                handle, cell_count, out_cells.ctypes.data, out_shifts.ctypes.data
            ),
            "periodic_refinement_copy_cells",
        )
        return PeriodicRefinementResult(status, out_points, out_cells, out_shifts)
    finally:
        library["phx_mc_periodic_refinement_free"](handle)


class ConstrainedDelaunay2D(NamedTuple):
    """Constrained triangulation and per-vertex source-edge construction witnesses."""

    points: np.ndarray
    triangles: np.ndarray
    cell_segments: np.ndarray
    constraint_sources: np.ndarray
    constraint_parents: np.ndarray
    constraint_ratios: np.ndarray
    status: MeshcoreStatus


def constrained_delaunay_2d(
    points: object,
    segments: object,
    /,
    *,
    holes: object | None = None,
    keep_convex_hull: bool = False,
    min_angle: float = 0.0,
    max_area: float = float("inf"),
) -> ConstrainedDelaunay2D:
    """Constrained Delaunay triangulation with source-edge ancestry.

    ``segments`` are ``(m, 2)`` point indices. ``min_angle`` (degrees,
    ``[0, 60)``) and ``max_area`` are physical mesh criteria. A finite
    numerical obstruction returns ``UNRESOLVED`` with a valid conforming mesh.
    Every constrained Steiner point carries its input source segment, parent
    edge endpoints, and split ratio.
    """

    library = load_meshcore()
    point_array = _points(points, "points", 2)
    segment_array = _index_array(segments, "segments")
    if segment_array.ndim != 2 or segment_array.shape[1] != 2:
        raise ValueError("segments must have shape (m, 2).")
    if np.any(segment_array < 0) or np.any(segment_array >= point_array.shape[0]):
        raise ValueError("segments reference a missing point.")
    segment_int = np.ascontiguousarray(segment_array, dtype=np.int32)
    hole_array = (
        np.zeros((0, 2), dtype=np.float64)
        if holes is None
        else _points(holes, "holes", 2)
    )
    if not isinstance(keep_convex_hull, bool):
        raise TypeError("keep_convex_hull must be a bool.")
    angle = float(min_angle)
    area = float(max_area)
    if not (0.0 <= angle < 60.0):
        raise ValueError("min_angle must lie in [0, 60) degrees.")
    if not area > 0.0:
        raise ValueError("max_area must be positive (inf disables the bound).")
    handle = ctypes.c_void_p()
    status = library["phx_mc_constrained_delaunay_2d"](
        point_array.shape[0],
        point_array.ctypes.data,
        segment_int.shape[0],
        segment_int.ctypes.data,
        hole_array.shape[0],
        hole_array.ctypes.data,
        1 if keep_convex_hull else 0,
        angle,
        area,
        ctypes.addressof(handle),
    )
    code = MeshcoreStatus(status)
    if code not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
        _raise_for_call(status, "constrained_delaunay_2d")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "constrained_delaunay_2d: missing mesh"
        )
    (
        mesh_points,
        cells,
        _,
        cell_segments,
        constraint_sources,
        constraint_parents,
        constraint_ratios,
    ) = _read_mesh(library, handle)
    return ConstrainedDelaunay2D(
        mesh_points,
        cells,
        cell_segments,
        constraint_sources,
        constraint_parents,
        constraint_ratios,
        code,
    )


def _power_domain_arrays(
    points: object,
    tets: object,
    tet_regions: object,
    tet_face_facets: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    domain = _points(points, "points", 3)
    tetrahedra = _native_integer_array(tets, "tets", np.dtype(np.int32))
    regions = _native_integer_array(tet_regions, "tet_regions", np.dtype(np.int32))
    facets = _native_integer_array(tet_face_facets, "tet_face_facets", np.dtype(np.int32))
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or regions.shape != (tetrahedra.shape[0],)
        or facets.shape != tetrahedra.shape
        or np.any(tetrahedra < 0)
        or np.any(tetrahedra >= domain.shape[0])
    ):
        raise ValueError(
            "tets, tet_regions, and tet_face_facets must align with domain points."
        )
    return domain, tetrahedra, regions, facets


def _power_graph_arrays(
    site_count: int,
    neighbor_offsets: object,
    neighbors: object,
    split_sites: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = _native_integer_array(
        neighbor_offsets, "neighbor_offsets", np.dtype(np.int64)
    )
    adjacent = _native_integer_array(neighbors, "neighbors", np.dtype(np.int32))
    split_values = _index_array(split_sites, "split_sites")
    if (
        offsets.shape != (site_count + 1,)
        or adjacent.ndim != 1
        or offsets[0] != 0
        or offsets[-1] != adjacent.size
        or np.any(offsets[1:] < offsets[:-1])
        or np.any(adjacent < 0)
        or np.any(adjacent >= site_count)
    ):
        raise ValueError("neighbor_offsets and neighbors must define site-indexed CSR.")
    if split_values.shape != (site_count,) or np.any(
        (split_values != 0) & (split_values != 1)
    ):
        raise ValueError("split_sites must contain one zero-or-one flag per site.")
    return offsets, adjacent, np.ascontiguousarray(split_values, dtype=np.int8)


def _copy_restricted_power_cells(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> RestrictedPowerCells:
    sizes = np.empty((5,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_power_cells_sizes"](handle, sizes.ctypes.data),
        "power_cells_sizes",
    )
    if np.any(sizes < 0):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "negative power-cell output size"
        )
    vertices_count, faces_count, entries_count, cells_count, pieces_count = sizes
    witness_count = library["phx_mc_power_cells_site_witness_size"](handle)
    if witness_count < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "negative power-cell witness size"
        )
    result = RestrictedPowerCells(
        np.empty((vertices_count, 3), dtype=np.float64),
        np.empty((faces_count + 1,), dtype=np.int64),
        np.empty((entries_count,), dtype=np.int32),
        np.empty((faces_count, 2), dtype=np.int32),
        np.empty((faces_count,), dtype=np.int32),
        np.empty((cells_count,), dtype=np.int32),
        np.empty((cells_count,), dtype=np.int32),
        np.empty((cells_count,), dtype=np.float64),
        np.empty((cells_count, 3), dtype=np.float64),
        np.empty((cells_count,), dtype=np.float64),
        np.empty((pieces_count,), dtype=np.int32),
        np.empty((pieces_count,), dtype=np.int32),
        np.empty((pieces_count,), dtype=np.int32),
        np.empty((pieces_count,), dtype=np.float64),
        np.empty((pieces_count, 3), dtype=np.float64),
        np.empty((vertices_count, 4), dtype=np.int32),
        np.empty((vertices_count + 1,), dtype=np.int64),
        np.empty((witness_count,), dtype=np.int32),
    )
    _raise_for_call(
        library["phx_mc_power_cells_export"](
            handle,
            vertices_count,
            faces_count + 1,
            faces_count,
            entries_count,
            cells_count,
            pieces_count,
            result.vertices.ctypes.data,
            result.face_offsets.ctypes.data,
            result.face_vertices.ctypes.data,
            result.face_cells.ctypes.data,
            result.face_facets.ctypes.data,
            result.cell_sites.ctypes.data,
            result.cell_regions.ctypes.data,
            result.cell_volumes.ctypes.data,
            result.cell_moments.ctypes.data,
            result.cell_second_moments.ctypes.data,
            result.piece_sites.ctypes.data,
            result.piece_tets.ctypes.data,
            result.piece_cells.ctypes.data,
            result.piece_volumes.ctypes.data,
            result.piece_moments.ctypes.data,
        ),
        "power_cells_export",
    )
    _raise_for_call(
        library["phx_mc_power_cells_export_vertex_carriers"](
            handle, vertices_count, result.vertex_carriers.ctypes.data
        ),
        "power_cells_export_vertex_carriers",
    )
    _raise_for_call(
        library["phx_mc_power_cells_export_vertex_sites"](
            handle,
            vertices_count + 1,
            vertices_count,
            witness_count,
            result.vertex_site_offsets.ctypes.data,
            result.vertex_sites.ctypes.data,
        ),
        "power_cells_export_vertex_sites",
    )
    return result


def _finish_restricted_power_cells(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    status: int,
    /,
    *,
    failed_image_coordinate: int = -1,
    failed_component: int = -1,
) -> RestrictedPowerCells:
    code = MeshcoreStatus(status)
    if not handle.value:
        if failed_image_coordinate >= 0 or failed_component >= 0:
            raise RestrictedPowerFailure(
                code,
                PowerCellFailureReason.IMAGE_SOURCE_DOMAIN,
                -1,
                -1,
                code,
                failed_image_coordinate=failed_image_coordinate,
                failed_component=failed_component,
            )
        _raise_for_call(status, "restricted_power_cells")
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "missing power-cell result")
    try:
        if code != MeshcoreStatus.OK:
            reason = np.zeros((1,), dtype=np.int64)
            site = np.zeros((1,), dtype=np.int64)
            tet = np.zeros((1,), dtype=np.int64)
            clip_status = np.zeros((1,), dtype=np.int32)
            _raise_for_call(
                library["phx_mc_power_cells_failure"](
                    handle,
                    reason.ctypes.data,
                    site.ctypes.data,
                    tet.ctypes.data,
                    clip_status.ctypes.data,
                ),
                "power_cells_failure",
            )
            failure_reason = PowerCellFailureReason(int(reason[0]))
            if failure_reason is PowerCellFailureReason.NONE:
                _raise_for_call(status, "restricted_power_cells")
            raise RestrictedPowerFailure(
                code,
                failure_reason,
                int(site[0]),
                int(tet[0]),
                MeshcoreStatus(int(clip_status[0])),
                failed_image_coordinate=failed_image_coordinate,
                failed_component=failed_component,
            )
        return _copy_restricted_power_cells(library, handle)
    finally:
        library["phx_mc_power_cells_free"](handle)


def restricted_power_cells(
    sites: object,
    weights: object,
    neighbor_offsets: object,
    neighbors: object,
    split_sites: object,
    points: object,
    tets: object,
    tet_regions: object,
    tet_face_facets: object,
    /,
) -> RestrictedPowerCells:
    """Restrict weighted cells to a conforming tetrahedral domain."""

    site_array = _points(sites, "sites", 3)
    weight_array = _real_array(weights, "weights")
    if (
        site_array.shape[0] == 0
        or weight_array.shape != (site_array.shape[0],)
        or not np.all(np.isfinite(weight_array))
    ):
        raise ValueError("weights must be a finite vector with one value per site.")
    offsets, adjacent, flags = _power_graph_arrays(
        site_array.shape[0], neighbor_offsets, neighbors, split_sites
    )
    domain, tetrahedra, regions, facets = _power_domain_arrays(
        points, tets, tet_regions, tet_face_facets
    )
    library = load_meshcore()
    handle = ctypes.c_void_p()
    request = _RestrictedPowerCellsRequest(
        site_count=site_array.shape[0],
        sites=site_array.ctypes.data,
        weights=weight_array.ctypes.data,
        neighbor_offsets=offsets.ctypes.data,
        neighbors=adjacent.ctypes.data,
        split_sites=flags.ctypes.data,
        point_count=domain.shape[0],
        points=domain.ctypes.data,
        tet_count=tetrahedra.shape[0],
        tets=tetrahedra.ctypes.data,
        tet_regions=regions.ctypes.data,
        tet_face_facets=facets.ctypes.data,
        result=ctypes.addressof(handle),
    )
    status = library["phx_mc_restricted_power_cells"](ctypes.byref(request))
    return _finish_restricted_power_cells(library, handle, status)


def restricted_power_cells_exact(
    sites: object,
    weights: object,
    image_site_owners: object,
    image_coordinate_offsets: object,
    image_coordinate_components: object,
    neighbor_offsets: object,
    neighbors: object,
    split_sites: object,
    points: object,
    tets: object,
    tet_regions: object,
    tet_face_facets: object,
    /,
) -> RestrictedPowerCells:
    """Restrict exact prepared power images without publishing rounded images."""

    site_array = _points(sites, "sites", 3)
    weight_array = _real_array(weights, "weights")
    owners = _native_integer_array(
        image_site_owners, "image_site_owners", np.dtype(np.int32)
    )
    coordinate_offsets = _native_integer_array(
        image_coordinate_offsets, "image_coordinate_offsets", np.dtype(np.int64)
    )
    components = _real_array(image_coordinate_components, "image_coordinate_components")
    if (
        site_array.shape[0] == 0
        or weight_array.shape != (site_array.shape[0],)
        or not np.all(np.isfinite(weight_array))
        or owners.ndim != 1
        or owners.size == 0
        or np.any(owners < 0)
        or np.any(owners >= site_array.shape[0])
        or components.ndim != 1
        or not np.all(np.isfinite(components))
        or coordinate_offsets.shape != (3 * owners.size + 1,)
        or coordinate_offsets[0] != 0
        or coordinate_offsets[-1] != components.size
        or np.any(coordinate_offsets[1:] < coordinate_offsets[:-1])
    ):
        raise ValueError(
            "exact power image owners and coordinate expansion arrays are misaligned."
        )
    offsets, adjacent, flags = _power_graph_arrays(
        owners.size, neighbor_offsets, neighbors, split_sites
    )
    domain, tetrahedra, regions, facets = _power_domain_arrays(
        points, tets, tet_regions, tet_face_facets
    )
    failed_coordinate = np.full((1,), -1, dtype=np.int64)
    failed_component = np.full((1,), -1, dtype=np.int64)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    request = _RestrictedPowerCellsExactRequest(
        original_site_count=site_array.shape[0],
        original_sites=site_array.ctypes.data,
        original_weights=weight_array.ctypes.data,
        image_count=owners.size,
        image_site_owners=owners.ctypes.data,
        image_coordinate_offsets=coordinate_offsets.ctypes.data,
        component_count=components.size,
        image_coordinate_components=components.ctypes.data,
        neighbor_offsets=offsets.ctypes.data,
        neighbors=adjacent.ctypes.data,
        split_sites=flags.ctypes.data,
        point_count=domain.shape[0],
        points=domain.ctypes.data,
        tet_count=tetrahedra.shape[0],
        tets=tetrahedra.ctypes.data,
        tet_regions=regions.ctypes.data,
        tet_face_facets=facets.ctypes.data,
        result=ctypes.addressof(handle),
        failed_image_coordinate=failed_coordinate.ctypes.data,
        failed_component=failed_component.ctypes.data,
    )
    status = library["phx_mc_restricted_power_cells_exact"](ctypes.byref(request))
    return _finish_restricted_power_cells(
        library,
        handle,
        status,
        failed_image_coordinate=int(failed_coordinate[0]),
        failed_component=int(failed_component[0]),
    )


def _contact_simplices(value: object, name: str, width: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (width, 3):
        raise ValueError(f"{name} must have shape (n, {width}, 3).")
    return array


def _contact_ids(
    value: object | None, name: str, count: int, width: int, /
) -> np.ndarray | None:
    if value is None:
        return None
    identifiers = _native_integer_array(value, name, np.dtype(np.int64))
    if identifiers.shape != (count, width):
        raise ValueError(f"{name} must have shape ({count}, {width}).")
    ordered = np.sort(identifiers, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError(f"{name} must contain distinct IDs within each simplex.")
    return identifiers


def triangle_intersection_classes(
    first: object,
    second: object,
    /,
    *,
    first_ids: object | None = None,
    second_ids: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify triangle contacts without constructing intersection vertices."""

    first_array = _contact_simplices(first, "first", 3)
    second_array = _contact_simplices(second, "second", 3)
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second triangle batches must have equal length.")
    count = first_array.shape[0]
    first_identifiers = _contact_ids(first_ids, "first_ids", count, 3)
    second_identifiers = _contact_ids(second_ids, "second_ids", count, 3)
    if (first_identifiers is None) != (second_identifiers is None):
        raise ValueError("first_ids and second_ids must be supplied together.")
    classes = np.full((count,), -1, dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    library = load_meshcore()
    call = library["phx_mc_triangle_intersections"](
        count,
        0,
        first_array.ctypes.data,
        second_array.ctypes.data,
        None if first_identifiers is None else first_identifiers.ctypes.data,
        None if second_identifiers is None else second_identifiers.ctypes.data,
        classes.ctypes.data,
        None,
        None,
        None,
        None,
        status.ctypes.data,
    )
    _raise_for_call(call, "triangle_intersections")
    return classes, status


def triangle_intersections(
    first: object,
    second: object,
    /,
    *,
    first_ids: object | None = None,
    second_ids: object | None = None,
) -> ContactIntersections:
    """Return triangle contact classes and bounded construction witnesses."""

    first_array = _contact_simplices(first, "first", 3)
    second_array = _contact_simplices(second, "second", 3)
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second triangle batches must have equal length.")
    count = first_array.shape[0]
    first_identifiers = _contact_ids(first_ids, "first_ids", count, 3)
    second_identifiers = _contact_ids(second_ids, "second_ids", count, 3)
    if (first_identifiers is None) != (second_identifiers is None):
        raise ValueError("first_ids and second_ids must be supplied together.")
    classes = np.full((count,), -1, dtype=np.int8)
    point_counts = np.zeros((count,), dtype=np.int32)
    points = np.zeros((count, 6, 3), dtype=np.float64)
    bounds = np.zeros((count, 6), dtype=np.float64)
    features = np.full((count, 6, 2), -1, dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    library = load_meshcore()
    call = library["phx_mc_triangle_intersections"](
        count,
        6,
        first_array.ctypes.data,
        second_array.ctypes.data,
        None if first_identifiers is None else first_identifiers.ctypes.data,
        None if second_identifiers is None else second_identifiers.ctypes.data,
        classes.ctypes.data,
        point_counts.ctypes.data,
        points.ctypes.data,
        bounds.ctypes.data,
        features.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "triangle_intersections")
    return ContactIntersections(classes, point_counts, points, bounds, features, status)


def segment_triangle_intersections(
    segments: object,
    triangles: object,
    /,
    *,
    segment_ids: object | None = None,
    triangle_ids: object | None = None,
) -> ContactIntersections:
    """Return segment/triangle contacts and construction witnesses."""

    segment_array = _contact_simplices(segments, "segments", 2)
    triangle_array = _contact_simplices(triangles, "triangles", 3)
    if segment_array.shape[0] != triangle_array.shape[0]:
        raise ValueError("segment and triangle batches must have equal length.")
    count = segment_array.shape[0]
    segment_identifiers = _contact_ids(segment_ids, "segment_ids", count, 2)
    triangle_identifiers = _contact_ids(triangle_ids, "triangle_ids", count, 3)
    if (segment_identifiers is None) != (triangle_identifiers is None):
        raise ValueError("segment_ids and triangle_ids must be supplied together.")
    classes = np.full((count,), -1, dtype=np.int8)
    point_counts = np.zeros((count,), dtype=np.int32)
    points = np.zeros((count, 2, 3), dtype=np.float64)
    bounds = np.zeros((count, 2), dtype=np.float64)
    features = np.full((count, 2, 2), -1, dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    library = load_meshcore()
    call = library["phx_mc_segment_triangle_intersections"](
        count,
        2,
        segment_array.ctypes.data,
        triangle_array.ctypes.data,
        None if segment_identifiers is None else segment_identifiers.ctypes.data,
        None if triangle_identifiers is None else triangle_identifiers.ctypes.data,
        classes.ctypes.data,
        point_counts.ctypes.data,
        points.ctypes.data,
        bounds.ctypes.data,
        features.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "segment_triangle_intersections")
    return ContactIntersections(classes, point_counts, points, bounds, features, status)


def point_triangle_locations(
    points: object, triangles: object, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify points relative to closed triangles in 3D."""

    point_array = _points(points, "points", 3)
    triangle_array = _contact_simplices(triangles, "triangles", 3)
    if point_array.shape[0] != triangle_array.shape[0]:
        raise ValueError("points and triangles must have equal length.")
    count = point_array.shape[0]
    sides = np.zeros((count,), dtype=np.int8)
    features = np.full((count,), -1, dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    library = load_meshcore()
    call = library["phx_mc_point_triangle_locations"](
        count,
        point_array.ctypes.data,
        triangle_array.ctypes.data,
        sides.ctypes.data,
        features.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "point_triangle_locations")
    return sides, features, status


def arrange_triangles(
    vertices: object,
    triangles: object,
    surfaces: object,
    pairs: object,
    /,
    *,
    classification_operands: int | None = None,
) -> NativeArrangement:
    """Arrange complete broad-phase triangle candidates with exact construction witnesses."""

    points = _points(vertices, "vertices", 3)
    faces = _native_integer_array(triangles, "triangles", np.dtype(np.int64))
    owners = _native_integer_array(surfaces, "surfaces", np.dtype(np.int32))
    candidates = _native_integer_array(pairs, "pairs", np.dtype(np.int64))
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or owners.shape != (faces.shape[0],)
        or candidates.ndim != 2
        or candidates.shape[1] != 2
        or np.any(faces < 0)
        or np.any(faces >= points.shape[0])
        or np.any(candidates < 0)
        or np.any(candidates >= faces.shape[0])
    ):
        raise ValueError("triangle, surface, and pair arrays are misaligned.")
    if classification_operands is not None and (
        isinstance(classification_operands, bool)
        or not isinstance(classification_operands, (int, np.integer))
        or classification_operands <= 0
        or classification_operands > np.iinfo(np.int64).max
    ):
        raise ValueError("classification_operands must be a positive int64 count.")
    library = load_meshcore()
    handle = ctypes.c_void_p()
    failed_face = np.full((1,), -1, dtype=np.int64)
    status = MeshcoreStatus(
        library["phx_mc_arrange_triangles"](
            points.shape[0],
            points.ctypes.data,
            faces.shape[0],
            faces.ctypes.data,
            owners.ctypes.data,
            candidates.shape[0],
            candidates.ctypes.data,
            failed_face.ctypes.data,
            ctypes.addressof(handle),
        )
    )
    if not handle.value:
        if status != MeshcoreStatus.OK:
            raise ArrangementFailure(status, int(failed_face[0]))
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "arrange_triangles: missing result"
        )
    try:
        if status != MeshcoreStatus.OK:
            raise ArrangementFailure(status, int(failed_face[0]))
        sizes = np.empty((5,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_arrangement_sizes"](handle, sizes.ctypes.data),
            "arrangement_sizes",
        )
        if np.any(sizes < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "negative arrangement size"
            )
        vertex_count, fragment_count, edge_count, coplanar_count, location_count = sizes
        result = NativeArrangement(
            np.empty((vertex_count, 3), dtype=np.float64),
            np.empty((vertex_count,), dtype=np.float64),
            np.empty((vertex_count,), dtype=np.int8),
            np.empty((vertex_count,), dtype=np.int64),
            np.empty((fragment_count, 3), dtype=np.int64),
            np.empty((fragment_count,), dtype=np.int64),
            np.empty((edge_count, 2), dtype=np.int64),
            np.empty((coplanar_count, 3), dtype=np.int64),
            np.empty((location_count, 3), dtype=np.int64),
            None,
        )
        _raise_for_call(
            library["phx_mc_arrangement_export"](
                handle,
                vertex_count,
                fragment_count,
                edge_count,
                coplanar_count,
                location_count,
                result.vertices.ctypes.data,
                result.bounds.ctypes.data,
                result.constructions.ctypes.data,
                result.origins.ctypes.data,
                result.fragments.ctypes.data,
                result.fragment_faces.ctypes.data,
                result.contact_edges.ctypes.data,
                result.coplanar.ctypes.data,
                result.locations.ctypes.data,
            ),
            "arrangement_export",
        )
        if classification_operands is None:
            return result
        failed_face[0] = -1
        classification_status = MeshcoreStatus(
            library["phx_mc_arrangement_classify"](
                handle, classification_operands, failed_face.ctypes.data
            )
        )
        match classification_status:
            case MeshcoreStatus.UNRESOLVED:
                classification = ArrangementClassification(
                    classification_status,
                    np.empty((0,), dtype=np.int64),
                    np.empty((0, classification_operands), dtype=np.int32),
                )
            case MeshcoreStatus.OK:
                classification_sizes = np.empty((2,), dtype=np.int64)
                _raise_for_call(
                    library["phx_mc_arrangement_classification_sizes"](
                        handle, classification_sizes.ctypes.data
                    ),
                    "arrangement_classification_sizes",
                )
                components, operands = classification_sizes
                if components < 0 or operands != classification_operands:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "arrangement classification reported invalid sizes",
                    )
                fragment_components = np.empty((fragment_count,), dtype=np.int64)
                component_windings = np.empty((components, operands), dtype=np.int32)
                _raise_for_call(
                    library["phx_mc_arrangement_export_classification"](
                        handle,
                        fragment_count,
                        components,
                        operands,
                        fragment_components.ctypes.data,
                        component_windings.ctypes.data,
                    ),
                    "arrangement_export_classification",
                )
                classification = ArrangementClassification(
                    classification_status, fragment_components, component_windings
                )
            case _:
                raise ArrangementFailure(classification_status, int(failed_face[0]))
        return result._replace(classification=classification)
    finally:
        library["phx_mc_arrangement_free"](handle)


def _exact_scalar_bank(
    words: object, offsets: object, signs: object, scalar_count: int, name: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    word_bank = _native_integer_array(words, f"{name}_words", np.dtype(np.uint32))
    offset_bank = _native_integer_array(offsets, f"{name}_offsets", np.dtype(np.int64))
    sign_bank = _native_integer_array(signs, f"{name}_signs", np.dtype(np.int8))
    if (
        word_bank.ndim != 1
        or offset_bank.shape != (scalar_count + 1,)
        or sign_bank.shape != (scalar_count,)
    ):
        raise ValueError(f"{name} exact scalar banks have inconsistent extents.")
    return word_bank, offset_bank, sign_bank


def _nonnegative_capacity(value: object, name: str, maximum: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    capacity = int(value)
    if capacity < 0:
        raise ValueError(f"{name} must be nonnegative.")
    if capacity > maximum:
        raise OverflowError(f"{name} exceeds the native index range.")
    return capacity


def clip_reference_triangle_exact(
    words: object, offsets: object, signs: object, vertex_capacity: int, /
) -> NativeReferenceTriangleClip:
    """Clip the exact reference triangle and retain source-plane witnesses."""

    word_bank, offset_bank, sign_bank = _exact_scalar_bank(
        words, offsets, signs, 18, "triangle_planes"
    )
    capacity = _nonnegative_capacity(
        vertex_capacity, "vertex_capacity", int(np.iinfo(np.int32).max)
    )
    supporting_planes = np.empty((capacity, 2), dtype=np.int32)
    edge_labels = np.empty((capacity,), dtype=np.int32)
    vertex_count = np.zeros((1,), dtype=np.int32)
    library = load_meshcore()
    status = library["phx_mc_clip_reference_triangle_exact"](
        word_bank.ctypes.data,
        word_bank.size,
        offset_bank.ctypes.data,
        sign_bank.ctypes.data,
        capacity,
        supporting_planes.ctypes.data,
        edge_labels.ctypes.data,
        vertex_count.ctypes.data,
    )
    _raise_for_call(status, "clip_reference_triangle_exact")
    count = int(vertex_count[0])
    if count < 0 or count > capacity:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "clip_reference_triangle_exact: invalid vertex count",
        )
    return NativeReferenceTriangleClip(
        count, supporting_planes[:count], edge_labels[:count]
    )


def clip_reference_tetrahedron_exact(
    words: object, offsets: object, signs: object, /
) -> NativeReferenceTetrahedronClip:
    """Clip an exact reference tetrahedron, returning plane and face witnesses."""

    sign_rows = _native_integer_array(signs, "plane_signs", np.dtype(np.int8))
    if sign_rows.ndim != 2 or sign_rows.shape[1] != 4:
        raise ValueError("plane_signs must have shape (plane_count, 4).")
    plane_count = sign_rows.shape[0]
    if plane_count > np.iinfo(np.int32).max:
        raise OverflowError("plane_count exceeds the native index range.")
    word_bank, offset_bank, sign_bank = _exact_scalar_bank(
        words, offsets, sign_rows.reshape(-1), 4 * plane_count, "plane"
    )
    library = load_meshcore()
    handle = ctypes.c_void_p()
    request = _ReferenceTetrahedronClipRequest(
        word_bank.ctypes.data,
        word_bank.size,
        offset_bank.ctypes.data,
        sign_bank.ctypes.data,
        plane_count,
        ctypes.addressof(handle),
    )
    status = library["phx_mc_clip_reference_tetrahedron_exact"](ctypes.byref(request))
    if status != MeshcoreStatus.OK:
        if handle.value:
            library["phx_mc_reference_tetrahedron_clip_free"](handle)
        _raise_for_call(status, "clip_reference_tetrahedron_exact")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "clip_reference_tetrahedron_exact: missing result",
        )
    try:
        sizes = np.empty((4,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_reference_tetrahedron_clip_sizes"](handle, sizes.ctypes.data),
            "reference_tetrahedron_clip_sizes",
        )
        if np.any(sizes < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "reference_tetrahedron_clip_sizes: negative result extent",
            )
        vertex_count, face_count, face_entry_count, incident_count = map(int, sizes)
        result = NativeReferenceTetrahedronClip(
            np.empty((vertex_count, 3), dtype=np.int32),
            np.empty((vertex_count + 1,), dtype=np.int32),
            np.empty((incident_count,), dtype=np.int32),
            np.empty((face_count + 1,), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
            np.empty((face_entry_count,), dtype=np.int32),
        )
        _raise_for_call(
            library["phx_mc_reference_tetrahedron_clip_export"](
                handle,
                vertex_count,
                result.supporting_planes.ctypes.data,
                vertex_count + 1,
                result.vertex_incident_offsets.ctypes.data,
                incident_count,
                result.vertex_incident_planes.ctypes.data,
                face_count + 1,
                result.face_offsets.ctypes.data,
                face_count,
                result.face_planes.ctypes.data,
                face_entry_count,
                result.face_vertices.ctypes.data,
            ),
            "reference_tetrahedron_clip_export",
        )
        return result
    finally:
        library["phx_mc_reference_tetrahedron_clip_free"](handle)


def arrange_tetrahedra_exact(
    point_words: object,
    point_offsets: object,
    point_signs: object,
    plane_words: object,
    plane_offsets: object,
    plane_signs: object,
    tetrahedra: object,
    tetrahedron_planes: object,
    parents: object,
    cut_offsets: object,
    cut_planes: object,
    /,
) -> NativeTetrahedraArrangement:
    """Arrange exact tetrahedra using signed one-based plane references.

    Plane rows have a positive first nonzero spatial coefficient. Positive
    references retain the row's ``<= 0`` halfspace; negative references reverse
    it. Negative parents encode source tetrahedron ``i`` as ``-i-1``.
    """

    point_sign_rows = _native_integer_array(point_signs, "point_signs", np.dtype(np.int8))
    plane_sign_rows = _native_integer_array(plane_signs, "plane_signs", np.dtype(np.int8))
    if point_sign_rows.ndim != 2 or point_sign_rows.shape[1] != 4:
        raise ValueError("point_signs must have shape (point_count, 4).")
    if plane_sign_rows.ndim != 2 or plane_sign_rows.shape[1] != 4:
        raise ValueError("plane_signs must have shape (plane_count, 4).")
    point_count, plane_count = point_sign_rows.shape[0], plane_sign_rows.shape[0]
    point_word_bank, point_offset_bank, point_sign_bank = _exact_scalar_bank(
        point_words,
        point_offsets,
        point_sign_rows.reshape(-1),
        4 * point_count,
        "point",
    )
    plane_word_bank, plane_offset_bank, plane_sign_bank = _exact_scalar_bank(
        plane_words,
        plane_offsets,
        plane_sign_rows.reshape(-1),
        4 * plane_count,
        "plane",
    )
    tet_bank = _native_integer_array(tetrahedra, "tetrahedra", np.dtype(np.int32))
    tet_plane_bank = _native_integer_array(
        tetrahedron_planes, "tetrahedron_planes", np.dtype(np.int32)
    )
    parent_bank = _native_integer_array(parents, "parents", np.dtype(np.int32))
    cut_offset_bank = _native_integer_array(
        cut_offsets, "cut_offsets", np.dtype(np.int64)
    )
    cut_plane_bank = _native_integer_array(cut_planes, "cut_planes", np.dtype(np.int32))
    if (
        tet_bank.ndim != 2
        or tet_bank.shape[1] != 4
        or tet_plane_bank.shape != tet_bank.shape
        or parent_bank.ndim != 1
        or cut_offset_bank.shape != (parent_bank.size + 1,)
        or cut_plane_bank.ndim != 1
    ):
        raise ValueError("tetrahedron jobs have inconsistent topology extents.")
    request = _ArrangeTetrahedraExactRequest(
        point_count,
        plane_count,
        tet_bank.shape[0],
        parent_bank.size,
        cut_plane_bank.size,
        point_word_bank.size,
        plane_word_bank.size,
        point_word_bank.ctypes.data,
        point_offset_bank.ctypes.data,
        point_sign_bank.ctypes.data,
        plane_word_bank.ctypes.data,
        plane_offset_bank.ctypes.data,
        plane_sign_bank.ctypes.data,
        tet_bank.ctypes.data,
        tet_plane_bank.ctypes.data,
        parent_bank.ctypes.data,
        cut_offset_bank.ctypes.data,
        cut_plane_bank.ctypes.data,
        0,
    )
    library = load_meshcore()
    handle = ctypes.c_void_p()
    request.result = ctypes.addressof(handle)
    status = library["phx_mc_arrange_tetrahedra_exact"](ctypes.byref(request))
    if status != MeshcoreStatus.OK:
        if handle.value:
            library["phx_mc_tetrahedra_arrangement_free"](handle)
        _raise_for_call(status, "arrange_tetrahedra_exact")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "arrange_tetrahedra_exact: missing result",
        )
    try:
        sizes = np.empty((5,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_tetrahedra_arrangement_sizes"](handle, sizes.ctypes.data),
            "tetrahedra_arrangement_sizes",
        )
        if np.any(sizes < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "tetrahedra_arrangement_sizes: negative result extent",
            )
        vertex_count, word_count, job_count, face_count, face_entry_count = map(
            int, sizes
        )
        result = NativeTetrahedraArrangement(
            np.empty((word_count,), dtype=np.uint32),
            np.empty((4 * vertex_count + 1,), dtype=np.int64),
            np.empty((vertex_count, 4), dtype=np.int8),
            np.empty((vertex_count,), dtype=np.int32),
            np.empty((job_count + 1,), dtype=np.int64),
            np.empty((face_count + 1,), dtype=np.int64),
            np.empty((face_entry_count,), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
        )
        _raise_for_call(
            library["phx_mc_tetrahedra_arrangement_export"](
                handle,
                vertex_count,
                word_count,
                job_count + 1,
                face_count + 1,
                face_count,
                face_entry_count,
                result.vertex_words.ctypes.data,
                result.vertex_word_offsets.ctypes.data,
                result.vertex_signs.ctypes.data,
                result.vertex_input_index.ctypes.data,
                result.job_face_offsets.ctypes.data,
                result.face_offsets.ctypes.data,
                result.face_vertices.ctypes.data,
                result.face_planes.ctypes.data,
            ),
            "tetrahedra_arrangement_export",
        )
        return result
    finally:
        library["phx_mc_tetrahedra_arrangement_free"](handle)


def _plc3d_build(
    operation: str,
    points: object,
    polygon_offsets: object,
    polygon_vertices: object,
    polygon_facets: object,
    facet_regions: object,
    segments: object,
    /,
    *,
    seeds: object | None = None,
    seed_regions: object | None = None,
    boundary_policy: PlcBoundaryPolicy = PlcBoundaryPolicy.CONFORMING,
    facet_tolerances: object | None = None,
    segment_tolerances: object | None = None,
    facet_planes: object | None = None,
    facet_plane_tolerances: object | None = None,
) -> PlcSourceData:
    """Run native PLC source preparation or full protected tetrahedral recovery."""

    operation_label = (
        "prepare_plc_source"
        if operation == "phx_mc_plc3d_prepare_source"
        else "recover_plc"
    )
    point_array = _points(points, "points", 3)
    offsets = _native_integer_array(
        polygon_offsets, "polygon_offsets", np.dtype(np.int64)
    )
    polygon_vertex_array = _native_integer_array(
        polygon_vertices, "polygon_vertices", np.dtype(np.int32)
    )
    polygon_facet_array = _native_integer_array(
        polygon_facets, "polygon_facets", np.dtype(np.int32)
    )
    region_array = _native_integer_array(
        facet_regions, "facet_regions", np.dtype(np.int32)
    )
    segment_array = _native_integer_array(segments, "segments", np.dtype(np.int32))
    if (
        offsets.ndim != 1
        or polygon_facet_array.ndim != 1
        or offsets.shape != (polygon_facet_array.size + 1,)
        or polygon_vertex_array.ndim != 1
        or region_array.ndim != 2
        or region_array.shape[1] != 2
        or segment_array.ndim != 2
        or segment_array.shape[1] != 2
    ):
        raise ValueError(
            "PLC polygon, facet, and segment banks have inconsistent extents."
        )
    seed_array = (
        np.empty((0, 3), dtype=np.float64)
        if seeds is None
        else _points(seeds, "seeds", 3)
    )
    if seeds is None:
        if seed_regions is not None:
            raise ValueError("seed_regions require seeds.")
        seed_region_array = np.empty((0,), dtype=np.int32)
    else:
        if seed_regions is None:
            raise ValueError("seed_regions are required when seeds are supplied.")
        seed_region_array = _native_integer_array(
            seed_regions, "seed_regions", np.dtype(np.int32)
        )
        if seed_region_array.shape != (seed_array.shape[0],):
            raise ValueError("seed_regions must have shape (seed_count,).")
    if not isinstance(boundary_policy, (int, np.integer)) or isinstance(
        boundary_policy, bool
    ):
        raise TypeError("boundary_policy must be a PlcBoundaryPolicy or integer.")
    policy = int(boundary_policy)
    if policy < np.iinfo(np.int32).min or policy > np.iinfo(np.int32).max:
        raise OverflowError("boundary_policy exceeds the native int32 range.")
    facet_tolerance_array = (
        None
        if facet_tolerances is None
        else _real_array(facet_tolerances, "facet_tolerances")
    )
    if facet_tolerance_array is not None and facet_tolerance_array.shape != (
        region_array.shape[0],
    ):
        raise ValueError("facet_tolerances must have shape (facet_count,).")
    segment_tolerance_array = (
        None
        if segment_tolerances is None
        else _real_array(segment_tolerances, "segment_tolerances")
    )
    if segment_tolerance_array is not None and segment_tolerance_array.shape != (
        segment_array.shape[0],
    ):
        raise ValueError("segment_tolerances must have shape (segment_count,).")
    facet_count = region_array.shape[0]
    plane_array = (
        None if facet_planes is None else _real_array(facet_planes, "facet_planes")
    )
    if plane_array is not None and plane_array.shape != (facet_count, 3, 3):
        raise ValueError("facet_planes must have shape (facet_count, 3, 3).")
    if facet_plane_tolerances is not None and plane_array is None:
        raise ValueError("facet_plane_tolerances require facet_planes.")
    plane_tolerance_array = (
        None
        if facet_plane_tolerances is None
        else _real_array(facet_plane_tolerances, "facet_plane_tolerances")
    )
    if plane_tolerance_array is not None and plane_tolerance_array.shape != (
        facet_count,
    ):
        raise ValueError("facet_plane_tolerances must have shape (facet_count,).")

    library = load_meshcore()
    handle = ctypes.c_void_p()
    request = _Plc3DPrepareSourceRequest(
        point_array.shape[0],
        point_array.ctypes.data,
        polygon_facet_array.size,
        polygon_vertex_array.size,
        offsets.ctypes.data,
        polygon_vertex_array.ctypes.data,
        polygon_facet_array.ctypes.data,
        region_array.shape[0],
        region_array.ctypes.data,
        segment_array.shape[0],
        segment_array.ctypes.data,
        seed_array.shape[0],
        seed_array.ctypes.data,
        seed_region_array.ctypes.data,
        policy,
        None if facet_tolerance_array is None else facet_tolerance_array.ctypes.data,
        None if segment_tolerance_array is None else segment_tolerance_array.ctypes.data,
        None if plane_array is None else plane_array.ctypes.data,
        None if plane_tolerance_array is None else plane_tolerance_array.ctypes.data,
        ctypes.addressof(handle),
    )
    code = MeshcoreStatus(library[operation](ctypes.byref(request)))
    if not handle.value:
        _raise_for_call(code, operation_label)
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"{operation_label}: missing result",
        )
    try:
        failure_reason = np.zeros((1,), dtype=np.int32)
        first_kind = np.zeros((1,), dtype=np.int32)
        first = np.zeros((1,), dtype=np.int64)
        second_kind = np.zeros((1,), dtype=np.int32)
        second = np.zeros((1,), dtype=np.int64)
        certified_deviation = np.zeros((1,), dtype=np.float64)
        declared_bound = np.zeros((1,), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_plc3d_failure"](
                handle,
                failure_reason.ctypes.data,
                first_kind.ctypes.data,
                first.ctypes.data,
                second_kind.ctypes.data,
                second.ctypes.data,
                certified_deviation.ctypes.data,
                declared_bound.ctypes.data,
            ),
            "plc3d_failure",
        )
        search = np.zeros((len(PlcSuppressionSearch._fields),), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_plc3d_suppression_search"](handle, search.ctypes.data),
            "plc3d_suppression_search",
        )
        suppression_search = PlcSuppressionSearch(*(int(value) for value in search))
        if code != MeshcoreStatus.OK:
            if failure_reason[0] != PlcFailureReason.NONE:
                raise PlcSourceFailure(
                    code,
                    PlcFailureReason(int(failure_reason[0])),
                    PlcEntityKind(int(first_kind[0])),
                    int(first[0]),
                    PlcEntityKind(int(second_kind[0])),
                    int(second[0]),
                    float(certified_deviation[0]),
                    float(declared_bound[0]),
                    suppression_search=suppression_search,
                )
            _raise_for_call(code, operation_label)
        sizes = np.empty((6,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_plc3d_sizes"](handle, sizes.ctypes.data),
            "plc3d_sizes",
        )
        if np.any(sizes < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "plc3d_sizes: negative result extent",
            )
        point_count, tet_count, face_count, segment_count, edge_count, triangle_count = (
            map(int, sizes)
        )
        result = PlcSourceData(
            code,
            np.empty((point_count, 3), dtype=np.float64),
            np.empty((tet_count, 4), dtype=np.int32),
            np.empty((tet_count,), dtype=np.int32),
            np.empty((face_count, 3), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
            np.empty((face_count,), dtype=np.int8),
            np.empty((segment_count, 2), dtype=np.int32),
            np.empty((segment_count,), dtype=np.int32),
            np.empty((point_count,), dtype=np.float64),
            np.empty((point_count,), dtype=np.int8),
            np.empty((edge_count, 2), dtype=np.int32),
            np.empty((triangle_count, 3), dtype=np.int32),
            np.empty((triangle_count,), dtype=np.int32),
            np.empty((point_count,), dtype=np.int8),
            np.empty((point_count,), dtype=np.int32),
            np.empty((point_count, 2), dtype=np.float64),
            np.empty((point_count,), dtype=np.float64),
            suppression_search,
        )
        _raise_for_call(
            library["phx_mc_plc3d_export"](
                handle,
                point_count,
                tet_count,
                face_count,
                segment_count,
                edge_count,
                triangle_count,
                result.points.ctypes.data,
                result.tetrahedra.ctypes.data,
                result.tetrahedron_regions.ctypes.data,
                result.faces.ctypes.data,
                result.face_sources.ctypes.data,
                result.face_orientations.ctypes.data,
                result.segments.ctypes.data,
                result.segment_sources.ctypes.data,
                result.protection.ctypes.data,
                result.vertex_dimensions.ctypes.data,
                result.plc_edges.ctypes.data,
                result.input_triangles.ctypes.data,
                result.input_polygons.ctypes.data,
            ),
            "plc3d_export",
        )
        _raise_for_call(
            library["phx_mc_plc3d_source_witnesses"](
                handle,
                point_count,
                result.source_strata.ctypes.data,
                result.source_entities.ctypes.data,
                result.source_parameters.ctypes.data,
                result.source_deviations.ctypes.data,
            ),
            "plc3d_source_witnesses",
        )
        return result
    finally:
        library["phx_mc_plc3d_free"](handle)


def prepare_plc_source(
    points: object,
    polygon_offsets: object,
    polygon_vertices: object,
    polygon_facets: object,
    facet_regions: object,
    segments: object,
    /,
    *,
    seeds: object | None = None,
    seed_regions: object | None = None,
    boundary_policy: PlcBoundaryPolicy = PlcBoundaryPolicy.CONFORMING,
    facet_tolerances: object | None = None,
    segment_tolerances: object | None = None,
    facet_planes: object | None = None,
    facet_plane_tolerances: object | None = None,
) -> PlcSourceData:
    """Prepare authored PLC topology without tetrahedral recovery.

    ``facet_planes`` (``(facet_count, 3, 3)``: origin, first axis, second
    axis; an all-NaN row declares none) declares exact facet planes: a
    declared facet's triangles form one group whatever their binary64
    rounding, so its internal edges are not PLC edges, once every vertex is
    admitted (the plane meets the vertex's rounding box or lies within the
    facet's ``facet_plane_tolerances`` entry, default 0); otherwise the facet
    refuses with ``SOURCE_DEVIATION`` naming the vertex and its certified
    distance.
    """

    return _plc3d_build(
        "phx_mc_plc3d_prepare_source",
        points,
        polygon_offsets,
        polygon_vertices,
        polygon_facets,
        facet_regions,
        segments,
        seeds=seeds,
        seed_regions=seed_regions,
        boundary_policy=boundary_policy,
        facet_tolerances=facet_tolerances,
        segment_tolerances=segment_tolerances,
        facet_planes=facet_planes,
        facet_plane_tolerances=facet_plane_tolerances,
    )


def recover_plc(
    points: object,
    polygon_offsets: object,
    polygon_vertices: object,
    polygon_facets: object,
    facet_regions: object,
    segments: object,
    /,
    *,
    seeds: object | None = None,
    seed_regions: object | None = None,
    boundary_policy: PlcBoundaryPolicy = PlcBoundaryPolicy.CONFORMING,
    facet_tolerances: object | None = None,
    segment_tolerances: object | None = None,
    facet_planes: object | None = None,
    facet_plane_tolerances: object | None = None,
) -> PlcSourceData:
    """Recover a PLC with witnesses indexed by exported raw carrier banks.

    Stratum 1 entities index ``plc_edges`` and stratum 2 entities index
    ``input_triangles``. Python consumers must resolve these rows through exact
    source identity before constructing a persistent source bank. Declared
    facet planes are as for ``prepare_plc_source``.
    """

    return _plc3d_build(
        "phx_mc_plc3d_recover",
        points,
        polygon_offsets,
        polygon_vertices,
        polygon_facets,
        facet_regions,
        segments,
        seeds=seeds,
        seed_regions=seed_regions,
        boundary_policy=boundary_policy,
        facet_tolerances=facet_tolerances,
        segment_tolerances=segment_tolerances,
        facet_planes=facet_planes,
        facet_plane_tolerances=facet_plane_tolerances,
    )


def recover_plc_3d(
    points: object,
    polygon_offsets: object,
    polygon_vertices: object,
    polygon_facets: object,
    facet_regions: object,
    /,
    *,
    segments: object | None = None,
    seeds: object | None = None,
    seed_regions: object | None = None,
    boundary_policy: PlcBoundaryPolicy = PlcBoundaryPolicy.CONFORMING,
    facet_tolerances: object | None = None,
    segment_tolerances: object | None = None,
    facet_planes: object | None = None,
    facet_plane_tolerances: object | None = None,
) -> PlcRecovery3D:
    """Recover a PLC with raw carrier-row witnesses and region labels.

    Stratum 1 entities index ``plc_edges``; stratum 2 entities index
    ``input_triangles``. These are not persistent canonical source-bank rows.
    Declared facet planes are as for ``prepare_plc_source``.
    """

    segment_bank = np.empty((0, 2), dtype=np.int32) if segments is None else segments
    recovered = recover_plc(
        points,
        polygon_offsets,
        polygon_vertices,
        polygon_facets,
        facet_regions,
        segment_bank,
        seeds=seeds,
        seed_regions=seed_regions,
        boundary_policy=boundary_policy,
        facet_tolerances=facet_tolerances,
        segment_tolerances=segment_tolerances,
        facet_planes=facet_planes,
        facet_plane_tolerances=facet_plane_tolerances,
    )
    return PlcRecovery3D(
        recovered.status,
        recovered.points,
        recovered.tetrahedra,
        recovered.tetrahedron_regions,
        recovered.faces,
        recovered.face_sources,
        recovered.face_orientations,
        recovered.segments,
        recovered.segment_sources,
        recovered.protection,
        recovered.vertex_dimensions,
        recovered.plc_edges,
        recovered.input_triangles,
        recovered.input_polygons,
        recovered.source_strata,
        recovered.source_entities,
        recovered.source_parameters,
        recovered.source_deviations,
        recovered.suppression_search,
    )


def _tet_mesh_rows(
    value: object, name: str, width: int, dtype: np.dtype, /
) -> np.ndarray:
    array = _native_integer_array(value, name, dtype)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    return array


def _tet_mesh_vector(value: object, name: str, dtype: np.dtype, /) -> np.ndarray:
    array = _native_integer_array(value, name, dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _tet_mesh_identity_payload(
    source_id: str | None,
    source_revision: str | None,
    source_map_id: str | None,
    frame_id: str | None,
    /,
) -> tuple[
    int,
    tuple[bytes, bytes, bytes, bytes],
    tuple[
        ctypes.Array[ctypes.c_char] | None,
        ctypes.Array[ctypes.c_char] | None,
        ctypes.Array[ctypes.c_char] | None,
        ctypes.Array[ctypes.c_char] | None,
    ],
]:
    values = (source_id, source_revision, source_map_id, frame_id)
    if all(value is None for value in values):
        return 0, (b"", b"", b"", b""), (None, None, None, None)
    if any(value is None for value in values):
        raise ValueError("TetMesh identity fields must be supplied together.")
    encoded: list[bytes] = []
    for value, name in zip(
        values,
        ("source_id", "source_revision", "source_map_id", "frame_id"),
        strict=True,
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        try:
            data = value.encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            raise ValueError(f"{name} must contain valid UTF-8 scalar values.") from error
        if len(data) > np.iinfo(np.int64).max:
            raise OverflowError(f"{name} exceeds the native UTF-8 length range.")
        encoded.append(data)
    if not any(encoded):
        return 0, (b"", b"", b"", b""), (None, None, None, None)
    if not encoded[1] or not encoded[3]:
        raise ValueError(
            "Bound TetMesh identity requires nonempty source_revision and frame_id."
        )
    payload = (encoded[0], encoded[1], encoded[2], encoded[3])
    buffers = (
        ctypes.create_string_buffer(payload[0]),
        ctypes.create_string_buffer(payload[1]),
        ctypes.create_string_buffer(payload[2]),
        ctypes.create_string_buffer(payload[3]),
    )
    return 1, payload, buffers


def _tet_mesh_identity_pointer(
    identity_bound: int, buffer: ctypes.Array[ctypes.c_char] | None, /
) -> _P | None:
    if not identity_bound:
        return None
    if buffer is None:
        raise RuntimeError("Bound TetMesh identity lost its UTF-8 buffer.")
    return ctypes.cast(buffer, _P)


def _tet_mesh_identity_arguments(
    identity_bound: int,
    encoded: tuple[bytes, bytes, bytes, bytes],
    buffers: tuple[
        ctypes.Array[ctypes.c_char] | None,
        ctypes.Array[ctypes.c_char] | None,
        ctypes.Array[ctypes.c_char] | None,
        ctypes.Array[ctypes.c_char] | None,
    ],
    /,
) -> tuple[object, ...]:
    return tuple(
        argument
        for data, buffer in zip(encoded, buffers, strict=True)
        for argument in (_tet_mesh_identity_pointer(identity_bound, buffer), len(data))
    )


def _tet_mesh_uint64(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint64).max:
        raise OverflowError(f"{name} exceeds uint64.")
    return result


def _tet_mesh_global_ids(
    values: object | None, count: int, name: str, /
) -> np.ndarray | None:
    if values is None:
        return None
    raw = np.asarray(values)
    if raw.shape != (count,) or raw.dtype.kind not in "iu":
        raise TypeError(
            f"{name} must be a one-dimensional integer identity bank of length {count}."
        )
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError(f"{name} must contain nonnegative global identities.")
    ids = np.ascontiguousarray(raw, dtype=np.uint64)
    if np.any(ids == np.iinfo(np.uint64).max) or np.unique(ids).size != count:
        raise ValueError(
            f"{name} must contain unique IDs other than reserved UINT64_MAX."
        )
    ids.setflags(write=False)
    return ids


def _tet_mesh_protected_mask(values: object | None, count: int, /) -> np.ndarray | None:
    if values is None:
        return None
    raw = np.asarray(values)
    if raw.shape != (count,) or raw.dtype.kind not in "biu":
        raise TypeError(
            "protected_vertices must be a boolean or integer vector matching points."
        )
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("protected_vertices must contain only 0 or 1.")
    result = np.ascontiguousarray(raw, dtype=np.int8)
    result.setflags(write=False)
    return result


class _TetPeriodicPayload(NamedTuple):
    view: _TetPeriodicStateView | _TetPeriodicEditView
    keepalive: tuple[object, ...]


def _periodic_integer_bank(
    values: object, name: str, dtype: np.dtype, shape: tuple[int, ...], /
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != shape or raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array with shape {shape}.")
    if dtype.kind == "i" and raw.dtype.kind == "u" and raw.size:
        if np.any(raw > np.iinfo(dtype).max):
            raise OverflowError(f"{name} exceeds {dtype}.")
    if dtype.kind == "i" and raw.dtype.kind == "i" and raw.size:
        if np.any(raw < np.iinfo(dtype).min) or np.any(raw > np.iinfo(dtype).max):
            raise OverflowError(f"{name} exceeds {dtype}.")
    result = np.ascontiguousarray(raw, dtype=dtype)
    result.setflags(write=False)
    return result


def _periodic_float_bank(
    values: object, name: str, shape: tuple[int, ...], /
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a real array with shape {shape}.")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result.setflags(write=False)
    return result


def _periodic_identity_view(
    identity_map: TetMeshPeriodicIdentityMap, /
) -> tuple[_TetPeriodicIdentityMapView, tuple[object, ...]]:
    if not isinstance(identity_map, TetMeshPeriodicIdentityMap):
        raise TypeError("identity_map must be a TetMeshPeriodicIdentityMap.")
    identity_names = (
        "source_mesh_id",
        "source_topology_id",
        "lifted_topology_id",
        "periodic_topology_id",
        "cell_entity_set_id",
        "edge_entity_set_id",
    )
    texts = tuple(
        canonical_identifier(getattr(identity_map, name), name) for name in identity_names
    )
    encoded = tuple(text.encode("utf-8", "strict") for text in texts)
    buffers = tuple(ctypes.create_string_buffer(value) for value in encoded)
    maps = identity_map.entity_maps
    if not isinstance(maps, tuple) or len(maps) != 4:
        raise ValueError(
            "Periodic identity maps must contain exactly dimensions 0 through 3."
        )
    descriptors = (_TetPeriodicEntityOrbitMap * 4)()
    lifted_ids_rows: list[np.ndarray] = []
    quotient_ids_rows: list[np.ndarray] = []
    lifted_to_quotient_rows: list[np.ndarray] = []
    orientation_rows: list[np.ndarray] = []
    winding_rows: list[np.ndarray] = []
    corner_rows: list[np.ndarray] = []
    representative_rows: list[np.ndarray] = []
    lifted_offset = quotient_offset = winding_offset = corner_offset = (
        representative_offset
    ) = 0
    for dimension, mapping in enumerate(maps):
        if not isinstance(mapping, TetMeshPeriodicEntityOrbitMap):
            raise TypeError(
                "entity_maps must contain TetMeshPeriodicEntityOrbitMap records."
            )
        if mapping.dimension != dimension:
            raise ValueError(
                "Periodic entity maps must be ordered by intrinsic dimension."
            )
        arity = _int32_argument(mapping.arity, "periodic entity arity")
        rank = _int32_argument(mapping.rank, "periodic rank")
        if arity <= 0 or rank < 0 or rank > 3:
            raise ValueError(
                "Periodic entity arity and rank are outside their native range."
            )
        lifted_raw = np.asarray(mapping.lifted_entity_ids)
        quotient_raw = np.asarray(mapping.quotient_entity_ids)
        if lifted_raw.ndim != 1 or quotient_raw.ndim != 1:
            raise ValueError("Periodic entity global IDs must be one-dimensional banks.")
        lifted_ids = _tet_mesh_global_ids(
            lifted_raw, lifted_raw.size, f"entity_maps[{dimension}].lifted_entity_ids"
        )
        quotient_ids = _tet_mesh_global_ids(
            quotient_raw,
            quotient_raw.size,
            f"entity_maps[{dimension}].quotient_entity_ids",
        )
        assert lifted_ids is not None and quotient_ids is not None
        lifted_count, quotient_count = lifted_ids.size, quotient_ids.size
        lifted_to = _periodic_integer_bank(
            mapping.lifted_to_quotient,
            f"entity_maps[{dimension}].lifted_to_quotient",
            np.dtype(np.int32),
            (lifted_count,),
        )
        orientations = _periodic_integer_bank(
            mapping.orientation_signs,
            f"entity_maps[{dimension}].orientation_signs",
            np.dtype(np.int32),
            (lifted_count,),
        )
        winding = _periodic_integer_bank(
            mapping.winding_shifts,
            f"entity_maps[{dimension}].winding_shifts",
            np.dtype(np.int64),
            (lifted_count, rank),
        )
        corners = _periodic_integer_bank(
            mapping.corner_permutations,
            f"entity_maps[{dimension}].corner_permutations",
            np.dtype(np.int32),
            (lifted_count, arity),
        )
        representatives = _periodic_integer_bank(
            mapping.representative_lifted_rows,
            f"entity_maps[{dimension}].representative_lifted_rows",
            np.dtype(np.int32),
            (quotient_count,),
        )
        if (
            (
                lifted_count
                and (np.any(lifted_to < 0) or np.any(lifted_to >= quotient_count))
            )
            or np.any(~np.isin(orientations, (-1, 1)))
            or np.any(representatives < 0)
            or np.any(representatives >= lifted_count)
            or (
                quotient_count
                and not np.array_equal(
                    lifted_to[representatives], np.arange(quotient_count)
                )
            )
        ):
            raise ValueError(
                "Periodic entity orbit indices, orientations, or representatives are invalid."
            )
        if lifted_count:
            expected_corners = np.arange(arity, dtype=np.int32)
            if any(not np.array_equal(np.sort(row), expected_corners) for row in corners):
                raise ValueError("Periodic entity corner rows must be true permutations.")
        descriptor = descriptors[dimension]
        descriptor.dimension = dimension
        descriptor.arity = arity
        descriptor.rank = rank
        descriptor.lifted_offset = lifted_offset
        descriptor.lifted_count = lifted_count
        descriptor.quotient_offset = quotient_offset
        descriptor.quotient_count = quotient_count
        descriptor.winding_offset = winding_offset
        descriptor.corner_permutation_offset = corner_offset
        descriptor.representative_offset = representative_offset
        lifted_ids_rows.append(lifted_ids)
        quotient_ids_rows.append(quotient_ids)
        lifted_to_quotient_rows.append(lifted_to)
        orientation_rows.append(orientations)
        winding_rows.append(winding.reshape(-1))
        corner_rows.append(corners.reshape(-1))
        representative_rows.append(representatives)
        lifted_offset += lifted_count
        quotient_offset += quotient_count
        winding_offset += winding.size
        corner_offset += corners.size
        representative_offset += quotient_count

    def concatenate(rows: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        result = np.concatenate(rows) if rows else np.empty((0,), dtype=dtype)
        return np.ascontiguousarray(result, dtype=dtype)

    lifted_ids = concatenate(lifted_ids_rows, np.dtype(np.uint64))
    quotient_ids = concatenate(quotient_ids_rows, np.dtype(np.uint64))
    lifted_to = concatenate(lifted_to_quotient_rows, np.dtype(np.int32))
    orientations = concatenate(orientation_rows, np.dtype(np.int32))
    windings = concatenate(winding_rows, np.dtype(np.int64))
    corners = concatenate(corner_rows, np.dtype(np.int32))
    representatives = concatenate(representative_rows, np.dtype(np.int32))
    view = _TetPeriodicIdentityMapView()
    view.identity_bound = 1
    for name, data, buffer in zip(identity_names, encoded, buffers, strict=True):
        setattr(view, name, _tet_mesh_identity_pointer(1, buffer))
        setattr(view, f"{name}_length", len(data))
    view.entity_map_count = len(maps)
    view.entity_maps = ctypes.cast(descriptors, _P)
    view.entity_lifted_count = lifted_ids.size
    view.entity_lifted_ids = _tet_mesh_pointer(lifted_ids)
    view.entity_lifted_to_quotient = _tet_mesh_pointer(lifted_to)
    view.entity_orientation_signs = _tet_mesh_pointer(orientations)
    view.entity_quotient_count = quotient_ids.size
    view.entity_quotient_ids = _tet_mesh_pointer(quotient_ids)
    view.entity_winding_count = windings.size
    view.entity_winding_shifts = _tet_mesh_pointer(windings)
    view.entity_corner_count = corners.size
    view.entity_corner_permutations = _tet_mesh_pointer(corners)
    view.entity_representative_count = representatives.size
    view.entity_representative_rows = _tet_mesh_pointer(representatives)
    keepalive = (
        *buffers,
        descriptors,
        *lifted_ids_rows,
        *quotient_ids_rows,
        *lifted_to_quotient_rows,
        *orientation_rows,
        *winding_rows,
        *corner_rows,
        *representative_rows,
        lifted_ids,
        quotient_ids,
        lifted_to,
        orientations,
        windings,
        corners,
        representatives,
    )
    return view, keepalive


def _tet_mesh_periodic_state_payload(
    state: TetMeshPeriodicState, /
) -> _TetPeriodicPayload:
    if not isinstance(state, TetMeshPeriodicState):
        raise TypeError("periodic_state must be a TetMeshPeriodicState.")
    identity_view, identity_keepalive = _periodic_identity_view(state.identity_map)
    orbit_ids = []
    members: list[TetMeshPeriodicMember] = []
    orbit_descriptors = (_TetPeriodicOrbit * len(state.orbits))()
    for row, orbit in enumerate(state.orbits):
        if not isinstance(orbit, TetMeshPeriodicOrbit) or not orbit.members:
            raise ValueError("Periodic orbits require retained member rows.")
        orbit_id = _tet_mesh_uint64(orbit.orbit_id, "periodic orbit ID")
        representative = _tet_mesh_uint64(
            orbit.representative_vertex_id, "periodic representative vertex ID"
        )
        if orbit_id in orbit_ids:
            raise ValueError("Periodic orbit IDs must be unique.")
        orbit_ids.append(orbit_id)
        descriptor = orbit_descriptors[row]
        descriptor.id = orbit_id
        descriptor.representative_vertex_id = representative
        descriptor.member_offset = len(members)
        descriptor.member_count = len(orbit.members)
        local_ids = []
        for member in orbit.members:
            if not isinstance(member, TetMeshPeriodicMember):
                raise TypeError(
                    "Periodic orbit members must be TetMeshPeriodicMember records."
                )
            local_ids.append(
                _tet_mesh_uint64(member.vertex_id, "periodic member vertex ID")
            )
            _periodic_integer_bank(
                member.winding, "periodic member winding", np.dtype(np.int64), (3,)
            )
            _periodic_float_bank(
                member.representative_to_member, "periodic member isometry", (3, 4)
            )
            members.append(member)
        if len(set(local_ids)) != len(local_ids) or representative not in local_ids:
            raise ValueError(
                "Periodic orbit members must uniquely contain the representative vertex."
            )
    member_descriptors = (_TetPeriodicVertexMember * len(members))()
    for row, member in enumerate(members):
        member_descriptors[row].vertex_id = _tet_mesh_uint64(
            member.vertex_id, "periodic member vertex ID"
        )
        winding = _periodic_integer_bank(
            member.winding, "periodic member winding", np.dtype(np.int64), (3,)
        )
        transform = _periodic_float_bank(
            member.representative_to_member, "periodic member isometry", (3, 4)
        )
        for axis, value in enumerate(winding.tolist()):
            member_descriptors[row].winding[axis] = int(value)
        for index, value in enumerate(transform.reshape(-1).tolist()):
            member_descriptors[row].representative_to_member[index] = float(value)

    quotient_ids = []
    lifted_ids: list[int] = []
    quotient_descriptors = (_TetPeriodicQuotientCell * len(state.quotient_cells))()
    for row, cell in enumerate(state.quotient_cells):
        if (
            not isinstance(cell, TetMeshPeriodicQuotientCell)
            or not cell.lifted_cell_ids.size
        ):
            raise ValueError(
                "Periodic quotient cells require retained lifted-cell ancestry."
            )
        orbit_id = _tet_mesh_uint64(cell.orbit_id, "periodic quotient group ID")
        cell_id = _tet_mesh_uint64(cell.cell_id, "periodic quotient cell ID")
        if cell_id in quotient_ids:
            raise ValueError("Periodic quotient cell IDs must be unique.")
        quotient_ids.append(cell_id)
        vertices = _tet_mesh_global_ids(
            cell.representative_vertices, 4, "periodic representative cell vertices"
        )
        assert vertices is not None
        winding = _periodic_integer_bank(
            cell.winding, "periodic quotient cell winding", np.dtype(np.int64), (4, 3)
        )
        lifted = np.asarray(cell.lifted_cell_ids)
        if lifted.ndim != 1:
            raise TypeError("periodic quotient lifted_cell_ids must be one-dimensional.")
        lifted_bank = _tet_mesh_global_ids(
            lifted, lifted.size, "periodic quotient lifted cell IDs"
        )
        assert lifted_bank is not None
        if set(lifted_ids) & set(int(value) for value in lifted_bank.tolist()):
            raise ValueError(
                "A lifted periodic cell may belong to only one quotient cell."
            )
        lifted_ids.extend(int(value) for value in lifted_bank.tolist())
        descriptor = quotient_descriptors[row]
        descriptor.orbit_id = orbit_id
        descriptor.id = cell_id
        for index, value in enumerate(vertices.tolist()):
            descriptor.representative_vertices[index] = int(value)
        for index, value in enumerate(winding.reshape(-1).tolist()):
            descriptor.winding[index] = int(value)
        descriptor.region = _int32_argument(cell.region, "periodic quotient region")
        descriptor.lifted_cell_offset = sum(
            int(value.lifted_cell_ids.size) for value in state.quotient_cells[:row]
        )
        descriptor.lifted_cell_count = lifted_bank.size

    cell_map = state.identity_map.entity_maps[3]
    if set(quotient_ids) != set(
        int(value) for value in np.asarray(cell_map.quotient_entity_ids).tolist()
    ):
        raise ValueError(
            "Periodic quotient-cell IDs must match the dimension-3 entity orbit map."
        )
    map_row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(cell_map.quotient_entity_ids).tolist()
        )
    }
    if any(
        _tet_mesh_uint64(cell.orbit_id, "periodic quotient group ID") not in map_row_by_id
        for cell in state.quotient_cells
    ):
        raise ValueError(
            "Periodic quotient grouping IDs must belong to the dim-3 quotient entity bank."
        )
    mapped_lifted = np.asarray(cell_map.lifted_entity_ids, dtype=np.uint64)
    lifted_to_quotient = np.asarray(cell_map.lifted_to_quotient, dtype=np.int64)
    for cell in state.quotient_cells:
        qrow = map_row_by_id[_tet_mesh_uint64(cell.cell_id, "periodic quotient cell ID")]
        expected_lifted = set(
            int(value) for value in mapped_lifted[lifted_to_quotient == qrow].tolist()
        )
        actual_lifted = set(
            int(value) for value in np.asarray(cell.lifted_cell_ids).tolist()
        )
        if expected_lifted != actual_lifted:
            raise ValueError(
                "Periodic dimension-3 entity map must exactly cover each quotient cell's lifted members."
            )

    lifted_bank = np.ascontiguousarray(lifted_ids, dtype=np.uint64)
    lifted_bank.setflags(write=False)
    view = _TetPeriodicStateView()
    view.identity_map = identity_view
    view.periodic_orbit_count = len(state.orbits)
    view.periodic_orbits = ctypes.cast(orbit_descriptors, _P) if state.orbits else None
    view.periodic_member_count = len(members)
    view.periodic_members = ctypes.cast(member_descriptors, _P) if members else None
    view.periodic_quotient_cell_count = len(state.quotient_cells)
    view.periodic_quotient_cells = (
        ctypes.cast(quotient_descriptors, _P) if state.quotient_cells else None
    )
    view.periodic_quotient_lifted_count = lifted_bank.size
    view.periodic_quotient_lifted_cell_ids = _tet_mesh_pointer(lifted_bank)
    return _TetPeriodicPayload(
        view,
        (
            *identity_keepalive,
            orbit_descriptors,
            member_descriptors,
            quotient_descriptors,
            lifted_bank,
        ),
    )


def _tet_mesh_periodic_edit_payload(edit: TetMeshPeriodicEdit, /) -> _TetPeriodicPayload:
    if not isinstance(edit, TetMeshPeriodicEdit):
        raise TypeError("periodic must be a TetMeshPeriodicEdit.")
    if not isinstance(edit.target_identity_map, TetMeshPeriodicIdentityMap):
        raise TypeError("target_identity_map must be a TetMeshPeriodicIdentityMap.")
    target_identity, identity_keepalive = _periodic_identity_view(
        edit.target_identity_map
    )
    target_cell_map = edit.target_identity_map.entity_maps[3]
    target_cell_quotient_ids = np.asarray(
        target_cell_map.quotient_entity_ids, dtype=np.uint64
    )
    target_cell_id_rows = {
        int(identifier): row
        for row, identifier in enumerate(target_cell_quotient_ids.tolist())
    }
    removed_orbits_raw = np.asarray(edit.removed_periodic_orbit_ids)
    if removed_orbits_raw.ndim != 1:
        raise TypeError("removed_periodic_orbit_ids must be a one-dimensional bank.")
    removed_orbits = _tet_mesh_global_ids(
        removed_orbits_raw, removed_orbits_raw.size, "removed_periodic_orbit_ids"
    )
    assert removed_orbits is not None
    removed_cells_raw = np.asarray(edit.removed_quotient_cell_ids)
    if removed_cells_raw.ndim != 1:
        raise TypeError("removed_quotient_cell_ids must be a one-dimensional bank.")
    removed_cells = _tet_mesh_global_ids(
        removed_cells_raw, removed_cells_raw.size, "removed_quotient_cell_ids"
    )
    assert removed_cells is not None
    if not isinstance(edit.periodic_members, tuple):
        raise TypeError("periodic_members must be a tuple of exact member records.")
    members = edit.periodic_members
    member_descriptors = (_TetPeriodicVertexMember * len(members))()
    member_keepalive: list[object] = []
    for row, member in enumerate(members):
        if not isinstance(member, TetMeshPeriodicMember):
            raise TypeError(
                "periodic_members must contain TetMeshPeriodicMember records."
            )
        member_descriptors[row].vertex_id = _tet_mesh_uint64(
            member.vertex_id, "periodic member vertex ID"
        )
        member_winding = _periodic_integer_bank(
            member.winding, "periodic member winding", np.dtype(np.int64), (3,)
        )
        member_isometry = _periodic_float_bank(
            member.representative_to_member, "periodic member isometry", (3, 4)
        )
        for index, value in enumerate(member_winding.tolist()):
            member_descriptors[row].winding[index] = int(value)
        for index, value in enumerate(member_isometry.reshape(-1).tolist()):
            member_descriptors[row].representative_to_member[index] = float(value)
        member_keepalive.extend((member_winding, member_isometry))

    if not isinstance(edit.replacement_periodic_orbits, tuple):
        raise TypeError("replacement_periodic_orbits must be a tuple.")
    replacement_orbits = edit.replacement_periodic_orbits
    replacement_orbit_descriptors = (_TetPeriodicOrbit * len(replacement_orbits))()
    replacement_orbit_members: list[TetMeshPeriodicMember] = []
    replacement_orbit_ids: list[int] = []
    for row, orbit in enumerate(replacement_orbits):
        if not isinstance(orbit, TetMeshPeriodicOrbit) or not orbit.members:
            raise ValueError("Replacement periodic orbits require retained members.")
        orbit_id_value = _tet_mesh_uint64(orbit.orbit_id, "replacement periodic orbit ID")
        representative = _tet_mesh_uint64(
            orbit.representative_vertex_id,
            "replacement periodic representative vertex ID",
        )
        if (
            orbit_id_value == np.iinfo(np.uint64).max
            or orbit_id_value in replacement_orbit_ids
        ):
            raise ValueError(
                "Replacement periodic orbit IDs must be unique and nonreserved."
            )
        replacement_orbit_ids.append(orbit_id_value)
        local_member_ids = tuple(
            _tet_mesh_uint64(member.vertex_id, "replacement periodic member vertex ID")
            for member in orbit.members
        )
        if (
            len(set(local_member_ids)) != len(local_member_ids)
            or representative not in local_member_ids
        ):
            raise ValueError(
                "Replacement periodic members must uniquely contain their representative."
            )
        descriptor = replacement_orbit_descriptors[row]
        descriptor.id = orbit_id_value
        descriptor.representative_vertex_id = representative
        descriptor.member_offset = len(replacement_orbit_members)
        descriptor.member_count = len(orbit.members)
        replacement_orbit_members.extend(orbit.members)
    replacement_member_descriptors = (
        _TetPeriodicVertexMember * len(replacement_orbit_members)
    )()
    replacement_member_keepalive: list[object] = []
    for row, member in enumerate(replacement_orbit_members):
        replacement_member_descriptors[row].vertex_id = _tet_mesh_uint64(
            member.vertex_id, "replacement periodic member vertex ID"
        )
        member_winding = _periodic_integer_bank(
            member.winding,
            "replacement periodic member winding",
            np.dtype(np.int64),
            (3,),
        )
        member_isometry = _periodic_float_bank(
            member.representative_to_member,
            "replacement periodic member isometry",
            (3, 4),
        )
        for index, value in enumerate(member_winding.tolist()):
            replacement_member_descriptors[row].winding[index] = int(value)
        for index, value in enumerate(member_isometry.reshape(-1).tolist()):
            replacement_member_descriptors[row].representative_to_member[index] = float(
                value
            )
        replacement_member_keepalive.extend((member_winding, member_isometry))

    if not isinstance(edit.replacement_quotient_cells, tuple):
        raise TypeError("replacement_quotient_cells must be a tuple.")
    replacement_cells = edit.replacement_quotient_cells
    replacement_cell_descriptors = (_TetPeriodicQuotientCell * len(replacement_cells))()
    replacement_lifted_ids: list[int] = []
    replacement_cell_ids: list[int] = []
    for row, cell in enumerate(replacement_cells):
        if (
            not isinstance(cell, TetMeshPeriodicQuotientCell)
            or not cell.lifted_cell_ids.size
        ):
            raise ValueError("Replacement quotient cells require lifted-cell ancestry.")
        orbit = _tet_mesh_uint64(cell.orbit_id, "replacement quotient orbit ID")
        identifier = _tet_mesh_uint64(cell.cell_id, "replacement quotient cell ID")
        if (
            orbit == np.iinfo(np.uint64).max
            or orbit not in target_cell_id_rows
            or identifier == np.iinfo(np.uint64).max
            or identifier in replacement_cell_ids
            or identifier not in target_cell_id_rows
        ):
            raise ValueError(
                "Replacement quotient group and cell IDs must belong to the target dim-3 entity bank."
            )
        replacement_cell_ids.append(identifier)
        vertices = _tet_mesh_global_ids(
            cell.representative_vertices, 4, "replacement representative cell vertices"
        )
        assert vertices is not None
        cell_winding = _periodic_integer_bank(
            cell.winding, "replacement quotient cell winding", np.dtype(np.int64), (4, 3)
        )
        lifted_raw = np.asarray(cell.lifted_cell_ids)
        if lifted_raw.ndim != 1:
            raise TypeError(
                "replacement quotient lifted-cell IDs must be one-dimensional."
            )
        lifted = _tet_mesh_global_ids(
            lifted_raw, lifted_raw.size, "replacement quotient lifted-cell IDs"
        )
        assert lifted is not None
        if set(replacement_lifted_ids) & set(int(value) for value in lifted.tolist()):
            raise ValueError(
                "Replacement lifted cells must belong to exactly one quotient cell."
            )
        target_row = target_cell_id_rows[identifier]
        expected_lifted = set(
            int(value)
            for value in np.asarray(target_cell_map.lifted_entity_ids)[
                np.asarray(target_cell_map.lifted_to_quotient) == target_row
            ].tolist()
        )
        if expected_lifted != set(int(value) for value in lifted.tolist()):
            raise ValueError(
                "Replacement quotient cell ancestry must match the target dim-3 map."
            )
        replacement_lifted_ids.extend(int(value) for value in lifted.tolist())
        descriptor = replacement_cell_descriptors[row]
        descriptor.orbit_id = orbit
        descriptor.id = identifier
        for index, value in enumerate(vertices.tolist()):
            descriptor.representative_vertices[index] = int(value)
        for index, value in enumerate(cell_winding.reshape(-1).tolist()):
            descriptor.winding[index] = int(value)
        descriptor.region = _int32_argument(cell.region, "replacement quotient region")
        descriptor.lifted_cell_offset = len(replacement_lifted_ids) - lifted.size
        descriptor.lifted_cell_count = lifted.size
        member_keepalive.extend((vertices, cell_winding, lifted))
    replacement_lifted_bank = np.ascontiguousarray(
        replacement_lifted_ids, dtype=np.uint64
    )
    replacement_lifted_bank.setflags(write=False)
    view = _TetPeriodicEditView()
    view.periodic_member_count = len(members)
    view.periodic_members = ctypes.cast(member_descriptors, _P) if members else None
    view.removed_periodic_orbit_count = removed_orbits.size
    view.removed_periodic_orbit_ids = _tet_mesh_pointer(removed_orbits)
    view.replacement_periodic_orbit_count = len(replacement_orbits)
    view.replacement_periodic_orbits = (
        ctypes.cast(replacement_orbit_descriptors, _P) if replacement_orbits else None
    )
    view.replacement_periodic_member_count = len(replacement_orbit_members)
    view.replacement_periodic_members = (
        ctypes.cast(replacement_member_descriptors, _P)
        if replacement_orbit_members
        else None
    )
    view.removed_quotient_cell_count = removed_cells.size
    view.removed_quotient_cell_ids = _tet_mesh_pointer(removed_cells)
    view.replacement_quotient_cell_count = len(replacement_cells)
    view.replacement_quotient_cells = (
        ctypes.cast(replacement_cell_descriptors, _P) if replacement_cells else None
    )
    view.replacement_quotient_lifted_count = replacement_lifted_bank.size
    view.replacement_quotient_lifted_cell_ids = _tet_mesh_pointer(replacement_lifted_bank)
    view.target_identity_map = ctypes.cast(ctypes.pointer(target_identity), _P)
    return _TetPeriodicPayload(
        view,
        (
            *identity_keepalive,
            removed_orbits,
            removed_cells,
            member_descriptors,
            *member_keepalive,
            replacement_orbit_descriptors,
            replacement_member_descriptors,
            *replacement_member_keepalive,
            replacement_cell_descriptors,
            replacement_lifted_bank,
            target_identity,
        ),
    )


def _periodic_snapshot_from_buffers(
    identity_strings: tuple[str, str, str, str, str, str] | None,
    orbit_descriptors: ctypes.Array[_TetPeriodicOrbit],
    member_descriptors: ctypes.Array[_TetPeriodicVertexMember],
    quotient_descriptors: ctypes.Array[_TetPeriodicQuotientCell],
    quotient_lifted_ids: np.ndarray,
    entity_map_descriptors: ctypes.Array[_TetPeriodicEntityOrbitMap],
    entity_lifted_ids: np.ndarray,
    entity_quotient_ids: np.ndarray,
    entity_lifted_to_quotient: np.ndarray,
    entity_orientation_signs: np.ndarray,
    entity_winding_shifts: np.ndarray,
    entity_corner_permutations: np.ndarray,
    entity_representative_rows: np.ndarray,
    /,
) -> TetMeshPeriodicSnapshot | None:
    descriptor_banks = (
        orbit_descriptors,
        member_descriptors,
        quotient_descriptors,
        entity_map_descriptors,
    )
    array_banks = (
        quotient_lifted_ids,
        entity_lifted_ids,
        entity_quotient_ids,
        entity_lifted_to_quotient,
        entity_orientation_signs,
        entity_winding_shifts,
        entity_corner_permutations,
        entity_representative_rows,
    )
    if identity_strings is None:
        if any(len(bank) for bank in descriptor_banks) or any(
            bank.size for bank in array_banks
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "unbound periodic identity has nonempty snapshot banks",
            )
        return None
    if len(identity_strings) != 6 or len(entity_map_descriptors) != 4:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "invalid periodic identity map extent"
        )
    source_mesh_id = canonical_identifier(identity_strings[0], "source_mesh_id")
    source_topology_id = canonical_identifier(identity_strings[1], "source_topology_id")
    lifted_topology_id = canonical_identifier(identity_strings[2], "lifted_topology_id")
    periodic_topology_id = canonical_identifier(
        identity_strings[3], "periodic_topology_id"
    )
    cell_entity_set_id = canonical_identifier(identity_strings[4], "cell_entity_set_id")
    edge_entity_set_id = canonical_identifier(identity_strings[5], "edge_entity_set_id")
    entity_maps: list[TetMeshPeriodicEntityOrbitMap] = []
    lifted_cursor = quotient_cursor = winding_cursor = corner_cursor = (
        representative_cursor
    ) = 0
    for dimension, descriptor in enumerate(entity_map_descriptors):
        arity = int(descriptor.arity)
        rank = int(descriptor.rank)
        lifted_offset = int(descriptor.lifted_offset)
        lifted_count = int(descriptor.lifted_count)
        quotient_offset = int(descriptor.quotient_offset)
        quotient_count = int(descriptor.quotient_count)
        winding_offset = int(descriptor.winding_offset)
        corner_offset = int(descriptor.corner_permutation_offset)
        representative_offset = int(descriptor.representative_offset)
        if (
            int(descriptor.dimension) != dimension
            or arity <= 0
            or rank < 0
            or rank > 3
            or min(lifted_count, quotient_count) < 0
            or (
                lifted_offset,
                quotient_offset,
                winding_offset,
                corner_offset,
                representative_offset,
            )
            != (
                lifted_cursor,
                quotient_cursor,
                winding_cursor,
                corner_cursor,
                representative_cursor,
            )
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic entity-map descriptor"
            )
        lifted = _tet_mesh_global_ids(
            entity_lifted_ids[lifted_offset : lifted_offset + lifted_count],
            lifted_count,
            f"periodic entity map {dimension} lifted IDs",
        )
        quotient = _tet_mesh_global_ids(
            entity_quotient_ids[quotient_offset : quotient_offset + quotient_count],
            quotient_count,
            f"periodic entity map {dimension} quotient IDs",
        )
        assert lifted is not None and quotient is not None
        lifted_to = _periodic_integer_bank(
            entity_lifted_to_quotient[lifted_offset : lifted_offset + lifted_count],
            f"periodic entity map {dimension} lifted-to-quotient rows",
            np.dtype(np.int32),
            (lifted_count,),
        )
        orientations = _periodic_integer_bank(
            entity_orientation_signs[lifted_offset : lifted_offset + lifted_count],
            f"periodic entity map {dimension} orientation signs",
            np.dtype(np.int32),
            (lifted_count,),
        )
        winding = _periodic_integer_bank(
            entity_winding_shifts[
                winding_offset : winding_offset + lifted_count * rank
            ].reshape((lifted_count, rank)),
            f"periodic entity map {dimension} winding shifts",
            np.dtype(np.int64),
            (lifted_count, rank),
        )
        corners = _periodic_integer_bank(
            entity_corner_permutations[
                corner_offset : corner_offset + lifted_count * arity
            ].reshape((lifted_count, arity)),
            f"periodic entity map {dimension} corner permutations",
            np.dtype(np.int32),
            (lifted_count, arity),
        )
        representatives = _periodic_integer_bank(
            entity_representative_rows[
                representative_offset : representative_offset + quotient_count
            ],
            f"periodic entity map {dimension} representatives",
            np.dtype(np.int32),
            (quotient_count,),
        )
        if (
            (
                lifted_count
                and (np.any(lifted_to < 0) or np.any(lifted_to >= quotient_count))
            )
            or np.any(~np.isin(orientations, (-1, 1)))
            or (
                quotient_count
                and (
                    np.any(representatives < 0) or np.any(representatives >= lifted_count)
                )
            )
            or (
                quotient_count
                and not np.array_equal(
                    lifted_to[representatives], np.arange(quotient_count)
                )
            )
            or any(
                not np.array_equal(np.sort(row), np.arange(arity, dtype=np.int32))
                for row in corners
            )
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic entity-map rows"
            )
        entity_maps.append(
            TetMeshPeriodicEntityOrbitMap(
                dimension,
                arity,
                rank,
                lifted,
                quotient,
                lifted_to,
                orientations,
                winding,
                corners,
                representatives,
            )
        )
        lifted_cursor += lifted_count
        quotient_cursor += quotient_count
        winding_cursor += lifted_count * rank
        corner_cursor += lifted_count * arity
        representative_cursor += quotient_count
    if (
        lifted_cursor,
        quotient_cursor,
        winding_cursor,
        corner_cursor,
        representative_cursor,
    ) != (
        entity_lifted_ids.size,
        entity_quotient_ids.size,
        entity_winding_shifts.size,
        entity_corner_permutations.size,
        entity_representative_rows.size,
    ):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "periodic entity-map banks are not exhausted"
        )
    identity_map = TetMeshPeriodicIdentityMap(
        source_mesh_id,
        source_topology_id,
        lifted_topology_id,
        periodic_topology_id,
        cell_entity_set_id,
        edge_entity_set_id,
        tuple(entity_maps),
    )

    members_seen = np.zeros((len(member_descriptors),), dtype=np.bool_)
    seen_vertex_ids: set[int] = set()
    orbits: list[TetMeshPeriodicOrbit] = []
    seen_orbit_ids: set[int] = set()
    for descriptor in orbit_descriptors:
        orbit_id = _tet_mesh_uint64(descriptor.id, "periodic orbit ID")
        representative_id = _tet_mesh_uint64(
            descriptor.representative_vertex_id,
            "periodic representative vertex ID",
        )
        offset, count = int(descriptor.member_offset), int(descriptor.member_count)
        if (
            orbit_id == np.iinfo(np.uint64).max
            or orbit_id in seen_orbit_ids
            or count <= 0
            or offset < 0
            or offset + count > len(member_descriptors)
            or np.any(members_seen[offset : offset + count])
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic vertex orbit descriptor"
            )
        seen_orbit_ids.add(orbit_id)
        members_seen[offset : offset + count] = True
        members: list[TetMeshPeriodicMember] = []
        member_ids: set[int] = set()
        for member_descriptor in member_descriptors[offset : offset + count]:
            vertex_id = _tet_mesh_uint64(
                member_descriptor.vertex_id, "periodic member vertex ID"
            )
            if (
                vertex_id == np.iinfo(np.uint64).max
                or vertex_id in seen_vertex_ids
                or vertex_id in member_ids
            ):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "periodic vertex belongs to multiple orbits",
                )
            member_ids.add(vertex_id)
            winding = _periodic_integer_bank(
                np.asarray(member_descriptor.winding, dtype=np.int64),
                "periodic member winding",
                np.dtype(np.int64),
                (3,),
            )
            isometry = _periodic_float_bank(
                np.asarray(
                    member_descriptor.representative_to_member, dtype=np.float64
                ).reshape((3, 4)),
                "periodic member isometry",
                (3, 4),
            )
            members.append(TetMeshPeriodicMember(vertex_id, winding, isometry))
        if representative_id not in member_ids:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "periodic orbit omits its representative vertex",
            )
        seen_vertex_ids.update(member_ids)
        orbits.append(TetMeshPeriodicOrbit(orbit_id, representative_id, tuple(members)))
    if members_seen.size and not np.all(members_seen):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "periodic vertex orbit bank is not exhausted"
        )

    quotient_seen = np.zeros((quotient_lifted_ids.size,), dtype=np.bool_)
    quotient_cells: list[TetMeshPeriodicQuotientCell] = []
    seen_qcell_ids: set[int] = set()
    dim3_map = entity_maps[3]
    qrow_by_id = {
        int(identifier): row
        for row, identifier in enumerate(dim3_map.quotient_entity_ids.tolist())
    }
    for descriptor in quotient_descriptors:
        group_id = _tet_mesh_uint64(descriptor.orbit_id, "periodic quotient group ID")
        cell_id = _tet_mesh_uint64(descriptor.id, "periodic quotient cell ID")
        offset, count = (
            int(descriptor.lifted_cell_offset),
            int(descriptor.lifted_cell_count),
        )
        if (
            group_id == np.iinfo(np.uint64).max
            or cell_id == np.iinfo(np.uint64).max
            or cell_id in seen_qcell_ids
            or cell_id not in qrow_by_id
            or group_id not in qrow_by_id
            or count <= 0
            or offset < 0
            or offset + count > quotient_lifted_ids.size
            or np.any(quotient_seen[offset : offset + count])
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic quotient-cell descriptor"
            )
        seen_qcell_ids.add(cell_id)
        quotient_seen[offset : offset + count] = True
        vertices = _tet_mesh_global_ids(
            np.asarray(descriptor.representative_vertices, dtype=np.uint64),
            4,
            "periodic quotient representative vertices",
        )
        lifted = _tet_mesh_global_ids(
            quotient_lifted_ids[offset : offset + count],
            count,
            "periodic quotient lifted cell IDs",
        )
        assert vertices is not None and lifted is not None
        winding = _periodic_integer_bank(
            np.asarray(descriptor.winding, dtype=np.int64).reshape((4, 3)),
            "periodic quotient winding",
            np.dtype(np.int64),
            (4, 3),
        )
        quotient_cells.append(
            TetMeshPeriodicQuotientCell(
                group_id,
                cell_id,
                vertices,
                winding,
                int(descriptor.region),
                lifted,
            )
        )
        qrow = qrow_by_id[cell_id]
        expected_lifted = set(
            int(value)
            for value in dim3_map.lifted_entity_ids[
                dim3_map.lifted_to_quotient == qrow
            ].tolist()
        )
        if expected_lifted != set(int(value) for value in lifted.tolist()):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "periodic dim-3 map does not match quotient-cell lifted ancestry",
            )
    if quotient_seen.size and not np.all(quotient_seen):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "periodic quotient lifted-cell bank is not exhausted",
        )
    if seen_qcell_ids != set(qrow_by_id):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "periodic quotient cells do not cover the dim-3 entity map",
        )
    return TetMeshPeriodicSnapshot(identity_map, tuple(orbits), tuple(quotient_cells))


def _tet_mesh_pointer(array: np.ndarray | None, /) -> int | None:
    return None if array is None or array.size == 0 else int(array.ctypes.data)


@final
class TetMesh3D:
    """Owned native tetrahedral mesh with atomic edits and checked snapshots."""

    def __init__(
        self,
        points: object,
        tetrahedra: object,
        tetrahedron_regions: object,
        faces: object,
        face_sources: object,
        segments: object,
        segment_sources: object,
        /,
        *,
        boundary_policy: PlcBoundaryPolicy | int = PlcBoundaryPolicy.FIXED,
        protection_radii: object | None = None,
        protected_vertices: object | None = None,
        vertex_ids: object | None = None,
        cell_ids: object | None = None,
        facet_ids: object | None = None,
        face_orientations: object | None = None,
        segment_ids: object | None = None,
        source: TetMeshSourceComplex | None = None,
        source_id: str | None = None,
        source_revision: str | None = None,
        source_map_id: str | None = None,
        frame_id: str | None = None,
        periodic_state: TetMeshPeriodicState | None = None,
    ) -> None:
        point_array = _points(points, "points", 3)
        tet_array = _tet_mesh_rows(tetrahedra, "tetrahedra", 4, np.dtype(np.int32))
        face_array = _tet_mesh_rows(faces, "faces", 3, np.dtype(np.int32))
        segment_array = _tet_mesh_rows(segments, "segments", 2, np.dtype(np.int32))
        vertex_id_bank = _tet_mesh_global_ids(
            vertex_ids, point_array.shape[0], "vertex_ids"
        )
        cell_id_bank = _tet_mesh_global_ids(cell_ids, tet_array.shape[0], "cell_ids")
        facet_id_bank = _tet_mesh_global_ids(facet_ids, face_array.shape[0], "facet_ids")
        segment_id_bank = _tet_mesh_global_ids(
            segment_ids, segment_array.shape[0], "segment_ids"
        )
        protected_mask = _tet_mesh_protected_mask(
            protected_vertices, point_array.shape[0]
        )
        if tet_array.shape[0] == 0:
            raise ValueError("tetrahedra must contain at least one cell.")
        if tetrahedron_regions is None:
            region_array = None
        else:
            region_array = _tet_mesh_vector(
                tetrahedron_regions, "tetrahedron_regions", np.dtype(np.int32)
            )
            if region_array.shape != (tet_array.shape[0],):
                raise ValueError(
                    "tetrahedron_regions must have one label per tetrahedron."
                )
        face_values = _index_array(face_sources, "face_sources")
        face_orientation_bank = (
            None
            if face_orientations is None
            else _native_integer_array(
                face_orientations, "face_orientations", np.dtype(np.int8)
            )
        )
        if face_orientation_bank is not None and (
            face_orientation_bank.shape != (face_array.shape[0],)
            or np.any(~np.isin(face_orientation_bank, (-1, 1)))
        ):
            raise ValueError(
                "face_orientations must contain one signed orientation per face."
            )
        segment_values = _index_array(segment_sources, "segment_sources")
        if face_values.shape != (face_array.shape[0],):
            raise ValueError("face_sources must have one label per constrained face.")
        if segment_values.shape != (segment_array.shape[0],):
            raise ValueError(
                "segment_sources must have one label per constrained segment."
            )

        source_banks: dict[str, np.ndarray | None] = {
            name: None
            for name in (
                "points",
                "faces",
                "face_groups",
                "face_ids",
                "face_tolerances",
                "segments",
                "segment_groups",
                "segment_ids",
                "segment_tolerances",
                "witness_strata",
                "witness_entities",
                "witness_parameters",
                "face_planes",
                "face_plane_tolerances",
            )
        }
        if source is not None:
            if not isinstance(source, TetMeshSourceComplex):
                raise TypeError("source must be a TetMeshSourceComplex or None.")
            source_points = _points(source.points, "source.points", 3)
            source_faces = _tet_mesh_rows(
                source.faces, "source.faces", 3, np.dtype(np.int32)
            )
            source_segments = _tet_mesh_rows(
                source.segments, "source.segments", 2, np.dtype(np.int32)
            )
            face_groups = _tet_mesh_vector(
                source.face_group_rows, "source.face_group_rows", np.dtype(np.int32)
            )
            segment_groups = _tet_mesh_vector(
                source.segment_group_rows, "source.segment_group_rows", np.dtype(np.int32)
            )
            face_ids = _tet_mesh_vector(
                source.face_ids, "source.face_ids", np.dtype(np.int64)
            )
            segment_ids = _tet_mesh_vector(
                source.segment_ids, "source.segment_ids", np.dtype(np.int64)
            )
            face_tolerances = _real_array(
                source.face_tolerances, "source.face_tolerances"
            )
            segment_tolerances = _real_array(
                source.segment_tolerances, "source.segment_tolerances"
            )
            strata = _native_integer_array(
                source.witness_strata, "source.witness_strata", np.dtype(np.int8)
            )
            entities = _native_integer_array(
                source.witness_entities, "source.witness_entities", np.dtype(np.int32)
            )
            parameters = _real_array(
                source.witness_parameters, "source.witness_parameters"
            )
            face_planes = _real_array(source.face_planes, "source.face_planes")
            face_plane_tolerances = _real_array(
                source.face_plane_tolerances, "source.face_plane_tolerances"
            )
            if (
                face_groups.shape != (source_faces.shape[0],)
                or face_tolerances.shape != (source_faces.shape[0],)
                or face_planes.shape != (source_faces.shape[0], 3, 3)
                or face_plane_tolerances.shape != (source_faces.shape[0],)
                or segment_groups.shape != (source_segments.shape[0],)
                or segment_tolerances.shape != (source_segments.shape[0],)
                or strata.shape != (point_array.shape[0],)
                or entities.shape != (point_array.shape[0],)
                or parameters.shape != (point_array.shape[0], 2)
            ):
                raise ValueError(
                    "source banks and witness rows must align with the mesh input."
                )
            if (
                np.any(face_ids[1:] <= face_ids[:-1])
                or np.any(segment_ids[1:] <= segment_ids[:-1])
                or np.any(face_groups < 0)
                or np.any(face_groups >= face_ids.size)
                or np.any(segment_groups < 0)
                or np.any(segment_groups >= segment_ids.size)
                or not np.all(np.isfinite(face_tolerances))
                or np.any(face_tolerances < 0.0)
                or not np.all(np.isfinite(segment_tolerances))
                or np.any(segment_tolerances < 0.0)
            ):
                raise ValueError("source identity rows and deviation bounds are invalid.")
            face_labels = _native_integer_array(
                face_values, "face_sources", np.dtype(np.int64)
            )
            segment_labels = _native_integer_array(
                segment_values, "segment_sources", np.dtype(np.int64)
            )
            face_source_rows = np.searchsorted(face_ids, face_labels)
            segment_source_rows = np.searchsorted(segment_ids, segment_labels)
            if (
                np.any(face_source_rows >= face_ids.size)
                or np.any(segment_source_rows >= segment_ids.size)
                or np.any(face_ids[face_source_rows] != face_labels)
                or np.any(segment_ids[segment_source_rows] != segment_labels)
            ):
                raise ValueError(
                    "mesh constraints reference an undeclared source identity."
                )
            face_source_array = np.ascontiguousarray(face_source_rows, dtype=np.int32)
            segment_source_array = np.ascontiguousarray(
                segment_source_rows, dtype=np.int32
            )
            source_banks = {
                "points": source_points,
                "faces": source_faces,
                "face_groups": face_groups,
                "face_ids": np.ascontiguousarray(face_ids[face_groups], dtype=np.int64),
                "face_tolerances": face_tolerances,
                "segments": source_segments,
                "segment_groups": segment_groups,
                "segment_ids": np.ascontiguousarray(
                    segment_ids[segment_groups], dtype=np.int64
                ),
                "segment_tolerances": segment_tolerances,
                "witness_strata": strata,
                "witness_entities": entities,
                "witness_parameters": parameters,
                "face_planes": face_planes,
                "face_plane_tolerances": face_plane_tolerances,
            }
        else:
            face_source_array = _native_integer_array(
                face_values, "face_sources", np.dtype(np.int32)
            )
            segment_source_array = _native_integer_array(
                segment_values, "segment_sources", np.dtype(np.int32)
            )

        if protection_radii is None:
            protection = None
        else:
            protection = _real_array(protection_radii, "protection_radii")
            if (
                protection.shape != (point_array.shape[0],)
                or not np.all(np.isfinite(protection))
                or np.any(protection < 0.0)
            ):
                raise ValueError(
                    "protection_radii must be finite, nonnegative, and match points."
                )
        policy = _int32_argument(boundary_policy, "boundary_policy")
        if policy not in (
            int(PlcBoundaryPolicy.FIXED),
            int(PlcBoundaryPolicy.CONFORMING),
        ):
            raise ValueError("boundary_policy must be fixed or conforming.")
        identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
            source_id, source_revision, source_map_id, frame_id
        )
        periodic_payload = (
            None
            if periodic_state is None
            else _tet_mesh_periodic_state_payload(periodic_state)
        )
        library = load_meshcore()
        handle = ctypes.c_void_p()
        request = _TetMeshCreateRequest()
        request.point_count = point_array.shape[0]
        request.points = _tet_mesh_pointer(point_array)
        request.vertex_ids = _tet_mesh_pointer(vertex_id_bank)
        request.tetrahedron_count = tet_array.shape[0]
        request.tetrahedra = _tet_mesh_pointer(tet_array)
        request.tetrahedron_regions = _tet_mesh_pointer(region_array)
        request.cell_ids = _tet_mesh_pointer(cell_id_bank)
        request.face_count = face_array.shape[0]
        request.faces = _tet_mesh_pointer(face_array)
        request.face_sources = _tet_mesh_pointer(face_source_array)
        request.face_orientations = _tet_mesh_pointer(face_orientation_bank)
        request.facet_ids = _tet_mesh_pointer(facet_id_bank)
        request.segment_count = segment_array.shape[0]
        request.segments = _tet_mesh_pointer(segment_array)
        request.segment_sources = _tet_mesh_pointer(segment_source_array)
        request.segment_ids = _tet_mesh_pointer(segment_id_bank)
        request.protection_radii = _tet_mesh_pointer(protection)
        request.protected_vertices = _tet_mesh_pointer(protected_mask)
        request.source_point_count = (
            0 if source_banks["points"] is None else source_banks["points"].shape[0]
        )
        request.source_points = _tet_mesh_pointer(source_banks["points"])
        request.source_face_count = (
            0 if source_banks["faces"] is None else source_banks["faces"].shape[0]
        )
        request.source_faces = _tet_mesh_pointer(source_banks["faces"])
        request.source_face_groups = _tet_mesh_pointer(source_banks["face_groups"])
        request.source_face_ids = _tet_mesh_pointer(source_banks["face_ids"])
        request.source_face_tolerances = _tet_mesh_pointer(
            source_banks["face_tolerances"]
        )
        request.source_segment_count = (
            0 if source_banks["segments"] is None else source_banks["segments"].shape[0]
        )
        request.source_segments = _tet_mesh_pointer(source_banks["segments"])
        request.source_segment_groups = _tet_mesh_pointer(source_banks["segment_groups"])
        request.source_segment_ids = _tet_mesh_pointer(source_banks["segment_ids"])
        request.source_segment_tolerances = _tet_mesh_pointer(
            source_banks["segment_tolerances"]
        )
        request.source_face_planes = _tet_mesh_pointer(source_banks["face_planes"])
        request.source_face_plane_tolerances = _tet_mesh_pointer(
            source_banks["face_plane_tolerances"]
        )
        request.witness_strata = _tet_mesh_pointer(source_banks["witness_strata"])
        request.witness_entities = _tet_mesh_pointer(source_banks["witness_entities"])
        request.witness_parameters = _tet_mesh_pointer(source_banks["witness_parameters"])
        request.witness_deviations = None
        request.boundary_policy = policy
        request.identity_bound = identity_bound
        request.source_id = _tet_mesh_identity_pointer(
            identity_bound, identity_buffers[0]
        )
        request.source_id_length = len(identity_bytes[0])
        request.source_revision = _tet_mesh_identity_pointer(
            identity_bound, identity_buffers[1]
        )
        request.source_revision_length = len(identity_bytes[1])
        request.source_map_id = _tet_mesh_identity_pointer(
            identity_bound, identity_buffers[2]
        )
        request.source_map_id_length = len(identity_bytes[2])
        request.frame_id = _tet_mesh_identity_pointer(identity_bound, identity_buffers[3])
        request.frame_id_length = len(identity_bytes[3])
        request.periodic_state = (
            None
            if periodic_payload is None
            else ctypes.cast(ctypes.pointer(periodic_payload.view), _P)
        )
        refusal = _TetPlaneRefusal()
        request.plane_refusal = ctypes.addressof(refusal)
        request.result = ctypes.addressof(handle)
        status = library["phx_mc_tet_mesh_create"](ctypes.byref(request))
        if refusal.refused:
            raise TetMeshPlaneRefusal(
                MeshcoreStatus(status),
                refusal.vertex,
                refusal.face_row,
                (refusal.parameters[0], refusal.parameters[1]),
                refusal.deviation,
                refusal.tolerance,
            )
        _raise_for_call(status, "tet_mesh_create")
        if not handle.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "tet_mesh_create returned no handle"
            )
        self._library = library
        self._handle = handle
        self._lock = threading.RLock()
        self._closed = False

    def _live(self, /) -> ctypes.c_void_p:
        if self._closed or not self._handle.value:
            raise RuntimeError("TetMesh3D is closed.")
        return self._handle

    def _identity(self, /) -> TetMeshIdentity:
        generation = _U64()
        bound = _I32()
        lengths = tuple(_I64() for _ in range(4))
        _raise_for_call(
            self._library["phx_mc_tet_mesh_identity_sizes"](
                self._live(),
                ctypes.addressof(generation),
                ctypes.addressof(bound),
                *(ctypes.addressof(value) for value in lengths),
            ),
            "tet_mesh_identity_sizes",
        )
        if bound.value not in (0, 1):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid TetMesh identity-bound status"
            )
        byte_lengths = tuple(int(value.value) for value in lengths)
        if any(value < 0 or value >= np.iinfo(np.intp).max for value in byte_lengths):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid TetMesh identity byte length"
            )
        if not bound.value:
            if any(byte_lengths):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "unbound TetMesh identity has nonempty strings",
                )
            return TetMeshIdentity(int(generation.value), False, "", "", "", "")
        if byte_lengths[1] == 0 or byte_lengths[3] == 0:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "bound TetMesh identity omits revision or frame",
            )
        buffers = tuple(
            ctypes.create_string_buffer(max(length + 1, 1)) for length in byte_lengths
        )
        arguments = tuple(
            value
            for length, buffer in zip(byte_lengths, buffers, strict=True)
            for value in (length + 1, ctypes.cast(buffer, _P))
        )
        _raise_for_call(
            self._library["phx_mc_tet_mesh_identity"](self._live(), *arguments),
            "tet_mesh_identity",
        )
        try:
            decoded = tuple(
                ctypes.string_at(buffer, length).decode("utf-8", "strict")
                for buffer, length in zip(buffers, byte_lengths, strict=True)
            )
        except UnicodeDecodeError as error:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "native TetMesh identity is not valid UTF-8",
            ) from error
        return TetMeshIdentity(int(generation.value), True, *decoded)

    def _periodic_identity_strings(self, /) -> tuple[str, str, str, str, str, str] | None:
        bound = _I32()
        lengths = tuple(_I64() for _ in range(6))
        _raise_for_call(
            self._library["phx_mc_tet_mesh_periodic_identity_sizes"](
                self._live(),
                ctypes.addressof(bound),
                *(ctypes.addressof(value) for value in lengths),
            ),
            "tet_mesh_periodic_identity_sizes",
        )
        if bound.value not in (0, 1):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic identity-bound status"
            )
        byte_lengths = tuple(int(value.value) for value in lengths)
        if any(value < 0 or value >= np.iinfo(np.intp).max for value in byte_lengths):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic identity byte length"
            )
        if not bound.value:
            if any(byte_lengths):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "unbound periodic identity has nonempty strings",
                )
            return None
        if any(value == 0 for value in byte_lengths):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "bound periodic identity omits a required identifier",
            )
        buffers = tuple(
            ctypes.create_string_buffer(length + 1) for length in byte_lengths
        )
        arguments = tuple(
            value
            for length, buffer in zip(byte_lengths, buffers, strict=True)
            for value in (length + 1, ctypes.cast(buffer, _P))
        )
        _raise_for_call(
            self._library["phx_mc_tet_mesh_periodic_identity"](self._live(), *arguments),
            "tet_mesh_periodic_identity",
        )
        try:
            decoded = tuple(
                ctypes.string_at(buffer, length).decode("utf-8", "strict")
                for buffer, length in zip(buffers, byte_lengths, strict=True)
            )
            return (
                decoded[0],
                decoded[1],
                decoded[2],
                decoded[3],
                decoded[4],
                decoded[5],
            )
        except UnicodeDecodeError as error:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "native periodic TetMesh identity is not valid UTF-8",
            ) from error

    @property
    def identity(self) -> TetMeshIdentity:
        with self._lock:
            return self._identity()

    @property
    def generation(self) -> int:
        return self.identity.generation

    @property
    def identity_bound(self) -> bool:
        return self.identity.identity_bound

    @property
    def source_id(self) -> str:
        return self.identity.source_id

    @property
    def source_revision(self) -> str:
        return self.identity.source_revision

    @property
    def source_map_id(self) -> str:
        return self.identity.source_map_id

    @property
    def frame_id(self) -> str:
        return self.identity.frame_id

    def arrays(self) -> TetMeshArrays:
        """Copy the complete live mesh and its owned source banks."""

        with self._lock:
            sizes = np.empty((PHX_MC_TET_MESH_SNAPSHOT_SIZES,), dtype=np.int64)
            _raise_for_call(
                self._library["phx_mc_tet_mesh_snapshot_sizes"](
                    self._live(), sizes.ctypes.data
                ),
                "tet_mesh_snapshot_sizes",
            )
            if np.any(sizes < 0):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "negative tetrahedral snapshot size"
                )
            (
                vertex_count,
                cell_count,
                facet_count,
                segment_count,
                source_point_count,
                source_face_count,
                source_segment_count,
                periodic_orbit_count,
                periodic_member_count,
                periodic_quotient_cell_count,
                periodic_quotient_lifted_count,
                periodic_entity_map_count,
                periodic_entity_lifted_count,
                periodic_entity_quotient_count,
                periodic_entity_winding_count,
                periodic_entity_corner_count,
                periodic_entity_representative_count,
                vertex_ancestry_count,
                cell_ancestor_count,
            ) = map(int, sizes.tolist())
            vertex_ids = np.empty((vertex_count,), dtype=np.uint64)
            points = np.empty((vertex_count, 3), dtype=np.float64)
            witness_strata = np.empty((vertex_count,), dtype=np.int8)
            witness_entities = np.empty((vertex_count,), dtype=np.int32)
            witness_parameters = np.empty((vertex_count, 2), dtype=np.float64)
            witness_deviations = np.empty((vertex_count,), dtype=np.float64)
            dimensions = np.empty((vertex_count,), dtype=np.int8)
            radii = np.empty((vertex_count,), dtype=np.float64)
            protected_vertices = np.empty((vertex_count,), dtype=np.int8)
            edge_witness_present = np.empty((vertex_count,), dtype=np.int8)
            edge_witness_endpoints = np.empty((vertex_count, 2), dtype=np.uint64)
            edge_parameters = np.empty((vertex_count,), dtype=np.float64)
            vertex_ancestry_offsets = np.empty((vertex_count + 1,), dtype=np.int64)
            vertex_ancestor_ids = np.empty((vertex_ancestry_count,), dtype=np.uint64)
            vertex_ancestor_winding = np.empty((vertex_ancestry_count, 3), dtype=np.int64)
            vertex_ancestry_weights = np.empty((vertex_ancestry_count,), dtype=np.float64)
            cell_ancestry_offsets = np.empty((cell_count + 1,), dtype=np.int64)
            cell_ancestor_ids = np.empty((cell_ancestor_count,), dtype=np.uint64)
            cell_ancestry_edit_kinds = np.empty((cell_count,), dtype=np.int32)
            cell_ids = np.empty((cell_count,), dtype=np.uint64)
            cell_vertices = np.empty((cell_count, 4), dtype=np.uint64)
            regions = np.empty((cell_count,), dtype=np.int32)
            facet_ids = np.empty((facet_count,), dtype=np.uint64)
            facet_vertices = np.empty((facet_count, 3), dtype=np.uint64)
            facet_sources = np.empty((facet_count,), dtype=np.int32)
            facet_orientations = np.empty((facet_count,), dtype=np.int8)
            protected_facets = np.empty((facet_count,), dtype=np.int8)
            segment_ids = np.empty((segment_count,), dtype=np.uint64)
            segment_vertices = np.empty((segment_count, 2), dtype=np.uint64)
            segment_sources = np.empty((segment_count,), dtype=np.int32)
            protected_segments = np.empty((segment_count,), dtype=np.int8)
            source_points = np.empty((source_point_count, 3), dtype=np.float64)
            source_faces = np.empty((source_face_count, 3), dtype=np.int32)
            source_face_groups = np.empty((source_face_count,), dtype=np.int32)
            source_face_ids = np.empty((source_face_count,), dtype=np.int64)
            source_face_tolerances = np.empty((source_face_count,), dtype=np.float64)
            source_segments = np.empty((source_segment_count, 2), dtype=np.int32)
            source_segment_groups = np.empty((source_segment_count,), dtype=np.int32)
            source_segment_ids = np.empty((source_segment_count,), dtype=np.int64)
            source_segment_tolerances = np.empty(
                (source_segment_count,), dtype=np.float64
            )
            source_face_planes = np.empty((source_face_count, 3, 3), dtype=np.float64)
            source_face_plane_tolerances = np.empty(
                (source_face_count,), dtype=np.float64
            )
            periodic_orbits = (_TetPeriodicOrbit * periodic_orbit_count)()
            periodic_members = (_TetPeriodicVertexMember * periodic_member_count)()
            periodic_quotient_cells = (
                _TetPeriodicQuotientCell * periodic_quotient_cell_count
            )()
            periodic_quotient_lifted_cell_ids = np.empty(
                (periodic_quotient_lifted_count,), dtype=np.uint64
            )
            periodic_entity_maps = (
                _TetPeriodicEntityOrbitMap * periodic_entity_map_count
            )()
            periodic_entity_lifted_ids = np.empty(
                (periodic_entity_lifted_count,), dtype=np.uint64
            )
            periodic_entity_quotient_ids = np.empty(
                (periodic_entity_quotient_count,), dtype=np.uint64
            )
            periodic_entity_lifted_to_quotient = np.empty(
                (periodic_entity_lifted_count,), dtype=np.int32
            )
            periodic_entity_orientation_signs = np.empty(
                (periodic_entity_lifted_count,), dtype=np.int32
            )
            periodic_entity_winding_shifts = np.empty(
                (periodic_entity_winding_count,), dtype=np.int64
            )
            periodic_entity_corner_permutations = np.empty(
                (periodic_entity_corner_count,), dtype=np.int32
            )
            periodic_entity_representative_rows = np.empty(
                (periodic_entity_representative_count,), dtype=np.int32
            )
            generation = _U64()
            request = _TetMeshSnapshotRequest()
            for name, value in (
                ("vertex_capacity", vertex_count),
                ("cell_capacity", cell_count),
                ("facet_capacity", facet_count),
                ("segment_capacity", segment_count),
                ("source_point_capacity", source_point_count),
                ("source_face_capacity", source_face_count),
                ("source_segment_capacity", source_segment_count),
                ("periodic_orbit_capacity", periodic_orbit_count),
                ("periodic_member_capacity", periodic_member_count),
                ("periodic_quotient_cell_capacity", periodic_quotient_cell_count),
                ("periodic_quotient_lifted_capacity", periodic_quotient_lifted_count),
                ("periodic_entity_map_capacity", periodic_entity_map_count),
                ("periodic_entity_lifted_capacity", periodic_entity_lifted_count),
                ("periodic_entity_quotient_capacity", periodic_entity_quotient_count),
                ("periodic_entity_winding_capacity", periodic_entity_winding_count),
                ("periodic_entity_corner_capacity", periodic_entity_corner_count),
                (
                    "periodic_entity_representative_capacity",
                    periodic_entity_representative_count,
                ),
                ("vertex_ancestry_capacity", vertex_ancestry_count),
                ("cell_ancestor_capacity", cell_ancestor_count),
            ):
                setattr(request, name, value)
            for name, array in (
                ("vertex_ids", vertex_ids),
                ("points", points),
                ("witness_strata", witness_strata),
                ("witness_entities", witness_entities),
                ("witness_parameters", witness_parameters),
                ("witness_deviations", witness_deviations),
                ("vertex_dimensions", dimensions),
                ("protection_radii", radii),
                ("protected_vertices", protected_vertices),
                ("cell_ids", cell_ids),
                ("vertex_edge_witness_present", edge_witness_present),
                ("vertex_edge_witness_endpoints", edge_witness_endpoints),
                ("vertex_edge_parameters", edge_parameters),
                ("vertex_ancestry_offsets", vertex_ancestry_offsets),
                ("vertex_ancestor_ids", vertex_ancestor_ids),
                ("vertex_ancestor_winding", vertex_ancestor_winding),
                ("vertex_ancestry_weights", vertex_ancestry_weights),
                ("cell_ancestry_offsets", cell_ancestry_offsets),
                ("cell_ancestor_ids", cell_ancestor_ids),
                ("cell_ancestry_edit_kinds", cell_ancestry_edit_kinds),
                ("cell_vertices", cell_vertices),
                ("cell_regions", regions),
                ("facet_ids", facet_ids),
                ("facet_vertices", facet_vertices),
                ("facet_sources", facet_sources),
                ("facet_orientations", facet_orientations),
                ("protected_facets", protected_facets),
                ("segment_ids", segment_ids),
                ("segment_vertices", segment_vertices),
                ("segment_sources", segment_sources),
                ("protected_segments", protected_segments),
                ("source_points", source_points),
                ("source_faces", source_faces),
                ("source_face_groups", source_face_groups),
                ("source_face_ids", source_face_ids),
                ("source_face_tolerances", source_face_tolerances),
                ("source_segments", source_segments),
                ("source_segment_groups", source_segment_groups),
                ("source_segment_ids", source_segment_ids),
                ("source_segment_tolerances", source_segment_tolerances),
                ("source_face_planes", source_face_planes),
                ("source_face_plane_tolerances", source_face_plane_tolerances),
            ):
                setattr(request, name, _tet_mesh_pointer(array) if array.size else None)
            request.periodic_orbits = (
                ctypes.cast(periodic_orbits, _P) if periodic_orbit_count else None
            )
            request.periodic_members = (
                ctypes.cast(periodic_members, _P) if periodic_member_count else None
            )
            request.periodic_quotient_cells = (
                ctypes.cast(periodic_quotient_cells, _P)
                if periodic_quotient_cell_count
                else None
            )
            request.periodic_quotient_lifted_cell_ids = _tet_mesh_pointer(
                periodic_quotient_lifted_cell_ids
            )
            request.periodic_entity_maps = (
                ctypes.cast(periodic_entity_maps, _P)
                if periodic_entity_map_count
                else None
            )
            request.periodic_entity_lifted_ids = _tet_mesh_pointer(
                periodic_entity_lifted_ids
            )
            request.periodic_entity_quotient_ids = _tet_mesh_pointer(
                periodic_entity_quotient_ids
            )
            request.periodic_entity_lifted_to_quotient = _tet_mesh_pointer(
                periodic_entity_lifted_to_quotient
            )
            request.periodic_entity_orientation_signs = _tet_mesh_pointer(
                periodic_entity_orientation_signs
            )
            request.periodic_entity_winding_shifts = _tet_mesh_pointer(
                periodic_entity_winding_shifts
            )
            request.periodic_entity_corner_permutations = _tet_mesh_pointer(
                periodic_entity_corner_permutations
            )
            request.periodic_entity_representative_rows = _tet_mesh_pointer(
                periodic_entity_representative_rows
            )
            request.generation = ctypes.addressof(generation)
            _raise_for_call(
                self._library["phx_mc_tet_mesh_snapshot"](
                    self._live(), ctypes.byref(request)
                ),
                "tet_mesh_snapshot",
            )
            identity = self._identity()
            if identity.generation != int(generation.value):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "TetMesh identity generation changed during snapshot.",
                )
            periodic_snapshot = _periodic_snapshot_from_buffers(
                self._periodic_identity_strings(),
                periodic_orbits,
                periodic_members,
                periodic_quotient_cells,
                periodic_quotient_lifted_cell_ids,
                periodic_entity_maps,
                periodic_entity_lifted_ids,
                periodic_entity_quotient_ids,
                periodic_entity_lifted_to_quotient,
                periodic_entity_orientation_signs,
                periodic_entity_winding_shifts,
                periodic_entity_corner_permutations,
                periodic_entity_representative_rows,
            )
            if (
                vertex_ancestry_offsets.shape != (vertex_count + 1,)
                or vertex_ancestry_offsets[0] != 0
                or vertex_ancestry_offsets[-1] != vertex_ancestry_count
                or np.any(np.diff(vertex_ancestry_offsets) < 0)
                or vertex_ancestor_winding.shape != (vertex_ancestry_count, 3)
                or vertex_ancestry_weights.shape != (vertex_ancestry_count,)
                or np.any(~np.isfinite(vertex_ancestry_weights))
                or cell_ancestry_offsets.shape != (cell_count + 1,)
                or cell_ancestry_offsets[0] != 0
                or cell_ancestry_offsets[-1] != cell_ancestor_count
                or np.any(np.diff(cell_ancestry_offsets) < 0)
                or cell_ancestry_edit_kinds.shape != (cell_count,)
            ):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "invalid TetMesh ancestry snapshot extents",
                )
            valid_edit_kinds = (-1, *(int(kind) for kind in TetMeshEditKind))
            if np.any(~np.isin(cell_ancestry_edit_kinds, valid_edit_kinds)):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "unknown TetMesh ancestry edit kind",
                )

            vertex_index = {
                int(identifier): row for row, identifier in enumerate(vertex_ids)
            }

            def remap(values: np.ndarray, width: int) -> np.ndarray:
                mapped = np.fromiter(
                    (vertex_index[int(identifier)] for identifier in values.reshape(-1)),
                    dtype=np.int64,
                    count=values.size,
                )
                return mapped.reshape((-1, width))

            face_id_bank = np.unique(source_face_ids)
            segment_id_bank = np.unique(source_segment_ids)
            face_group_rows = np.searchsorted(face_id_bank, source_face_ids).astype(
                np.int32
            )
            segment_group_rows = np.searchsorted(
                segment_id_bank, source_segment_ids
            ).astype(np.int32)
            face_labels_by_group: dict[int, int] = {}
            for group, label in zip(
                source_face_groups.tolist(), source_face_ids.tolist(), strict=True
            ):
                if group in face_labels_by_group and face_labels_by_group[group] != label:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "one face group has multiple source IDs",
                    )
                face_labels_by_group[group] = label
            segment_labels_by_group: dict[int, int] = {}
            for group, label in zip(
                source_segment_groups.tolist(), source_segment_ids.tolist(), strict=True
            ):
                if (
                    group in segment_labels_by_group
                    and segment_labels_by_group[group] != label
                ):
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "one segment group has multiple source IDs",
                    )
                segment_labels_by_group[group] = label
            cell_order = np.argsort(cell_ids, kind="stable")
            facet_order = np.argsort(facet_ids, kind="stable")
            segment_order = np.argsort(segment_ids, kind="stable")
            output_face_sources = np.asarray(
                [
                    face_labels_by_group.get(int(value), int(value))
                    for value in facet_sources
                ],
                dtype=np.int64,
            )
            output_segment_sources = np.asarray(
                [
                    segment_labels_by_group.get(int(value), int(value))
                    for value in segment_sources
                ],
                dtype=np.int64,
            )
            cell_ancestry_offsets_sorted = np.empty((cell_count + 1,), dtype=np.int64)
            cell_ancestry_offsets_sorted[0] = 0
            cell_ancestor_ids_sorted = np.empty_like(cell_ancestor_ids)
            ancestor_cursor = 0
            for output_row, source_row in enumerate(cell_order.tolist()):
                start = int(cell_ancestry_offsets[source_row])
                end = int(cell_ancestry_offsets[source_row + 1])
                count = end - start
                cell_ancestor_ids_sorted[ancestor_cursor : ancestor_cursor + count] = (
                    cell_ancestor_ids[start:end]
                )
                ancestor_cursor += count
                cell_ancestry_offsets_sorted[output_row + 1] = ancestor_cursor
            cell_ancestry_edit_kinds_sorted = cell_ancestry_edit_kinds[cell_order]
            return TetMeshArrays(
                points,
                remap(cell_vertices[cell_order], 4),
                regions[cell_order],
                remap(facet_vertices[facet_order], 3),
                output_face_sources[facet_order],
                remap(segment_vertices[segment_order], 2),
                output_segment_sources[segment_order],
                dimensions,
                radii,
                protected_vertices,
                edge_witness_present,
                edge_witness_endpoints,
                edge_parameters,
                vertex_ids,
                cell_ids[cell_order],
                facet_ids[facet_order],
                facet_orientations[facet_order],
                protected_facets[facet_order],
                segment_ids[segment_order],
                protected_segments[segment_order],
                source_points,
                source_faces,
                face_group_rows,
                face_id_bank,
                source_face_tolerances,
                source_segments,
                segment_group_rows,
                segment_id_bank,
                source_segment_tolerances,
                source_face_planes,
                source_face_plane_tolerances,
                witness_strata,
                witness_entities,
                witness_parameters,
                witness_deviations,
                identity.generation,
                identity.identity_bound,
                identity.source_id,
                identity.source_revision,
                identity.source_map_id,
                identity.frame_id,
                vertex_ancestry_offsets,
                vertex_ancestor_ids,
                vertex_ancestor_winding,
                vertex_ancestry_weights,
                cell_ancestry_offsets_sorted,
                cell_ancestor_ids_sorted,
                cell_ancestry_edit_kinds_sorted,
                periodic_snapshot,
            )

    def source_evidence(self) -> TetMeshSourceEvidence:
        arrays = self.arrays()
        bounds = np.concatenate(
            (arrays.source_face_tolerances, arrays.source_segment_tolerances)
        )
        requested = float(np.max(bounds)) if bounds.size else 0.0
        achieved = (
            float(np.max(arrays.witness_deviations))
            if arrays.witness_deviations.size
            else 0.0
        )
        return TetMeshSourceEvidence(
            requested,
            achieved,
            arrays.witness_strata,
            arrays.witness_entities,
            arrays.witness_parameters,
            arrays.witness_deviations,
        )

    def unmet(self) -> TetMeshUnmet:
        """Copy the live mesh's numerical criteria that remain unmet."""

        with self._lock:
            count = _I64()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_unmet_count"](
                    self._live(), ctypes.addressof(count)
                ),
                "tet_mesh_unmet_count",
            )
            if count.value < 0:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "negative tetrahedral unmet count"
                )
            row_count = int(count.value)
            tetrahedra = np.empty((row_count,), dtype=np.uint64)
            criteria = np.empty((row_count,), dtype=np.int32)
            reasons = np.empty((row_count,), dtype=np.int32)
            values = np.empty((row_count,), dtype=np.float64)
            apex_vertex_ids = np.empty((row_count,), dtype=np.uint64)
            apex_angles = np.empty((row_count,), dtype=np.float64)
            _raise_for_call(
                self._library["phx_mc_tet_mesh_unmet"](
                    self._live(),
                    row_count,
                    _tet_mesh_pointer(tetrahedra),
                    _tet_mesh_pointer(criteria),
                    _tet_mesh_pointer(reasons),
                    _tet_mesh_pointer(values),
                    _tet_mesh_pointer(apex_vertex_ids),
                    _tet_mesh_pointer(apex_angles),
                ),
                "tet_mesh_unmet",
            )
            return TetMeshUnmet(
                tetrahedra, criteria, reasons, values, apex_vertex_ids, apex_angles
            )

    def plane_records(self) -> TetMeshPlaneRecords:
        """Copy the live mesh's vertex records on declared facet planes."""

        with self._lock:
            count = _I64()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_plane_record_count"](
                    self._live(), ctypes.addressof(count)
                ),
                "tet_mesh_plane_record_count",
            )
            if count.value < 0:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "negative tetrahedral plane record count",
                )
            row_count = int(count.value)
            vertex_ids = np.empty((row_count,), dtype=np.uint64)
            face_rows = np.empty((row_count,), dtype=np.int32)
            parameters = np.empty((row_count, 2), dtype=np.float64)
            deviations = np.empty((row_count,), dtype=np.float64)
            _raise_for_call(
                self._library["phx_mc_tet_mesh_plane_records"](
                    self._live(),
                    row_count,
                    _tet_mesh_pointer(vertex_ids),
                    _tet_mesh_pointer(face_rows),
                    _tet_mesh_pointer(parameters),
                    _tet_mesh_pointer(deviations),
                ),
                "tet_mesh_plane_records",
            )
            return TetMeshPlaneRecords(vertex_ids, face_rows, parameters, deviations)

    def apply_edit(self, edit: TetMeshEdit) -> TetMeshEditStatus:
        if not isinstance(edit, TetMeshEdit):
            raise TypeError("edit must be a TetMeshEdit.")
        generation = _tet_mesh_uint64(edit.expected_generation, "expected_generation")
        kind = TetMeshEditKind(edit.kind)
        collapse_pairs = _tet_mesh_rows(
            np.empty((0, 2), dtype=np.uint64)
            if edit.collapse_pairs is None
            else edit.collapse_pairs,
            "collapse_pairs",
            2,
            np.dtype(np.uint64),
        )
        split_edges = _tet_mesh_rows(
            np.empty((0, 2), dtype=np.uint64)
            if edit.split_edges is None
            else edit.split_edges,
            "split_edges",
            2,
            np.dtype(np.uint64),
        )
        collapse_removed = (
            collapse_pairs[:, 0]
            if edit.removed_vertex_ids is None and collapse_pairs.size
            else np.empty((0,), dtype=np.uint64)
            if edit.removed_vertex_ids is None
            else _tet_mesh_vector(
                edit.removed_vertex_ids, "removed_vertex_ids", np.dtype(np.uint64)
            )
        )

        def rows(
            value: object | None, name: str, width: int, dtype: np.dtype
        ) -> np.ndarray:
            return _tet_mesh_rows(
                np.empty((0, width), dtype=dtype) if value is None else value,
                name,
                width,
                dtype,
            )

        def vector(value: object | None, name: str, dtype: np.dtype) -> np.ndarray:
            return _tet_mesh_vector(
                np.empty((0,), dtype=dtype) if value is None else value, name, dtype
            )

        removed_cells = vector(
            edit.removed_cell_ids, "removed_cell_ids", np.dtype(np.uint64)
        )
        changed_ids = vector(
            edit.changed_vertex_ids, "changed_vertex_ids", np.dtype(np.uint64)
        )
        changed_points = (
            None
            if edit.changed_vertex_points is None
            else _points(edit.changed_vertex_points, "changed_vertex_points", 3)
        )
        if (changed_points is None) != (changed_ids.size == 0) or (
            changed_points is not None and changed_points.shape[0] != changed_ids.size
        ):
            raise ValueError(
                "changed_vertex_ids and changed_vertex_points must have matching rows."
            )
        changed_strata = (
            None
            if edit.changed_witness_strata is None
            else _native_integer_array(
                edit.changed_witness_strata, "changed_witness_strata", np.dtype(np.int8)
            )
        )
        changed_entities = (
            None
            if edit.changed_witness_entities is None
            else _native_integer_array(
                edit.changed_witness_entities,
                "changed_witness_entities",
                np.dtype(np.int32),
            )
        )
        changed_parameters = (
            None
            if edit.changed_witness_parameters is None
            else _real_array(
                edit.changed_witness_parameters, "changed_witness_parameters"
            )
        )
        changed_deviations = (
            None
            if edit.changed_witness_deviations is None
            else _real_array(
                edit.changed_witness_deviations, "changed_witness_deviations"
            )
        )
        changed_radii = (
            None
            if edit.changed_protection_radii is None
            else _real_array(edit.changed_protection_radii, "changed_protection_radii")
        )
        changed_protected = (
            None
            if edit.changed_protected_vertices is None
            else _native_integer_array(
                edit.changed_protected_vertices,
                "changed_protected_vertices",
                np.dtype(np.int8),
            )
        )
        edge_witness_values = (
            edit.changed_edge_witness_present,
            edit.changed_edge_witness_endpoints,
            edit.changed_edge_parameters,
        )
        if all(value is None for value in edge_witness_values):
            edge_witness_present = edge_witness_endpoints = edge_parameters = None
        elif any(value is None for value in edge_witness_values):
            raise ValueError(
                "Changed edge witness presence, endpoints, and parameters must be supplied together."
            )
        else:
            raw_present = np.asarray(edit.changed_edge_witness_present)
            if raw_present.dtype.kind == "b":
                raw_present = raw_present.astype(np.int8)
            edge_witness_present = _native_integer_array(
                raw_present, "changed_edge_witness_present", np.dtype(np.int8)
            )
            edge_witness_endpoints = _tet_mesh_rows(
                edit.changed_edge_witness_endpoints,
                "changed_edge_witness_endpoints",
                2,
                np.dtype(np.uint64),
            )
            edge_parameters = _real_array(
                edit.changed_edge_parameters, "changed_edge_parameters"
            )
            if (
                edge_witness_present.shape != (changed_ids.size,)
                or edge_witness_endpoints.shape != (changed_ids.size, 2)
                or edge_parameters.shape != (changed_ids.size,)
                or np.any(~np.isin(edge_witness_present, (0, 1)))
            ):
                raise ValueError(
                    "Changed edge witness arrays must align with changed vertices."
                )
            witnessed = np.flatnonzero(edge_witness_present)
            if witnessed.size:
                witnessed_edges = edge_witness_endpoints[witnessed]
                if (
                    np.any(witnessed_edges[:, 0] == witnessed_edges[:, 1])
                    or np.any(edge_parameters[witnessed] <= 0.0)
                    or np.any(edge_parameters[witnessed] >= 1.0)
                ):
                    raise ValueError(
                        "Changed edge witnesses require distinct endpoints and parameters in (0, 1)."
                    )
                declared_edges = {
                    tuple(int(value) for value in row) for row in split_edges.tolist()
                }
                if any(
                    tuple(int(value) for value in row) not in declared_edges
                    for row in witnessed_edges.tolist()
                ):
                    raise ValueError(
                        "Changed edge witnesses must match a declared split-edge orientation."
                    )
        for name, value in (
            ("changed_witness_strata", changed_strata),
            ("changed_witness_entities", changed_entities),
            ("changed_witness_deviations", changed_deviations),
            ("changed_protection_radii", changed_radii),
            ("changed_protected_vertices", changed_protected),
        ):
            if value is not None and value.shape != (changed_ids.size,):
                raise ValueError(f"{name} must have one value per changed vertex.")
        if changed_parameters is not None and changed_parameters.shape != (
            changed_ids.size,
            2,
        ):
            raise ValueError("changed_witness_parameters must have shape (n, 2).")

        def ancestry_inputs(
            offsets_value: object | None,
            ancestor_ids_value: object | None,
            winding_value: object | None,
            weights_value: object | None,
            parent_count: int,
            name: str,
        ) -> tuple[
            np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
        ]:
            supplied = (offsets_value, ancestor_ids_value, winding_value, weights_value)
            if all(value is None for value in supplied):
                return None, None, None, None
            if any(value is None for value in supplied):
                raise ValueError(
                    f"{name} offsets, IDs, winding, and weights must be supplied together."
                )
            offsets = _native_integer_array(
                offsets_value, f"{name}_offsets", np.dtype(np.int64)
            )
            raw_ids = np.asarray(ancestor_ids_value)
            if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
                raise TypeError(
                    f"{name}_ancestor_ids must be a one-dimensional integer array."
                )
            if raw_ids.dtype.kind == "i" and np.any(raw_ids < 0):
                raise ValueError(f"{name}_ancestor_ids must be nonnegative.")
            ancestor_ids = _native_integer_array(
                raw_ids, f"{name}_ancestor_ids", np.dtype(np.uint64)
            )
            if np.any(ancestor_ids == np.iinfo(np.uint64).max):
                raise ValueError(
                    f"{name}_ancestor_ids must not contain the reserved sentinel."
                )
            winding = _native_integer_array(
                winding_value, f"{name}_ancestor_winding", np.dtype(np.int64)
            )
            weights = _real_array(weights_value, f"{name}_ancestry_weights")
            if (
                offsets.shape != (parent_count + 1,)
                or offsets.size == 0
                or offsets[0] != 0
                or offsets[-1] != ancestor_ids.size
                or np.any(np.diff(offsets) < 0)
                or winding.shape != (ancestor_ids.size, 3)
                or weights.shape != (ancestor_ids.size,)
            ):
                raise ValueError(f"{name} ancestry arrays must form aligned CSR rows.")
            return offsets, ancestor_ids, winding, weights

        (
            changed_ancestry_offsets,
            changed_ancestor_ids,
            changed_ancestor_winding,
            changed_ancestry_weights,
        ) = ancestry_inputs(
            edit.changed_vertex_ancestry_offsets,
            edit.changed_vertex_ancestor_ids,
            edit.changed_vertex_ancestor_winding,
            edit.changed_vertex_ancestry_weights,
            changed_ids.size,
            "changed_vertex_ancestry",
        )
        replacement_cells = rows(
            edit.replacement_cell_vertices,
            "replacement_cell_vertices",
            4,
            np.dtype(np.uint64),
        )
        replacement_cell_ids = vector(
            edit.replacement_cell_ids, "replacement_cell_ids", np.dtype(np.uint64)
        )
        if replacement_cell_ids.shape != (replacement_cells.shape[0],):
            raise ValueError("replacement_cell_ids must match replacement_cell_vertices.")
        replacement_cell_sources = (
            None
            if edit.replacement_cell_facet_sources is None
            else _tet_mesh_rows(
                edit.replacement_cell_facet_sources,
                "replacement_cell_facet_sources",
                4,
                np.dtype(np.int32),
            )
        )
        if (
            replacement_cell_sources is not None
            and replacement_cell_sources.shape != replacement_cells.shape
        ):
            raise ValueError(
                "replacement_cell_facet_sources must match replacement cells."
            )
        replacement_regions = (
            None
            if edit.replacement_regions is None
            else vector(
                edit.replacement_regions, "replacement_regions", np.dtype(np.int32)
            )
        )
        if (
            replacement_regions is not None
            and replacement_regions.shape != replacement_cell_ids.shape
        ):
            raise ValueError("replacement_regions must match replacement cells.")
        replacement_ancestry_values = (
            edit.replacement_cell_ancestry_offsets,
            edit.replacement_cell_ancestor_ids,
        )
        if all(value is None for value in replacement_ancestry_values):
            replacement_ancestry_offsets = replacement_ancestor_ids = None
        elif any(value is None for value in replacement_ancestry_values):
            raise ValueError(
                "Replacement cell ancestry offsets and IDs must be supplied together."
            )
        else:
            replacement_ancestry_offsets = _native_integer_array(
                edit.replacement_cell_ancestry_offsets,
                "replacement_cell_ancestry_offsets",
                np.dtype(np.int64),
            )
            raw_ancestor_ids = np.asarray(edit.replacement_cell_ancestor_ids)
            if raw_ancestor_ids.ndim != 1 or raw_ancestor_ids.dtype.kind not in "iu":
                raise TypeError(
                    "replacement_cell_ancestor_ids must be a one-dimensional integer array."
                )
            if raw_ancestor_ids.dtype.kind == "i" and np.any(raw_ancestor_ids < 0):
                raise ValueError("replacement_cell_ancestor_ids must be nonnegative.")
            replacement_ancestor_ids = _native_integer_array(
                raw_ancestor_ids, "replacement_cell_ancestor_ids", np.dtype(np.uint64)
            )
            if (
                np.any(replacement_ancestor_ids == np.iinfo(np.uint64).max)
                or replacement_ancestry_offsets.shape != (replacement_cell_ids.size + 1,)
                or replacement_ancestry_offsets.size == 0
                or replacement_ancestry_offsets[0] != 0
                or replacement_ancestry_offsets[-1] != replacement_ancestor_ids.size
                or np.any(np.diff(replacement_ancestry_offsets) < 0)
            ):
                raise ValueError(
                    "Replacement cell ancestry arrays must form aligned CSR rows."
                )
        removed_facets = vector(
            edit.removed_facet_ids, "removed_facet_ids", np.dtype(np.uint64)
        )
        replacement_facet_ids = vector(
            edit.replacement_facet_ids, "replacement_facet_ids", np.dtype(np.uint64)
        )
        replacement_facets = rows(
            edit.replacement_facet_vertices,
            "replacement_facet_vertices",
            3,
            np.dtype(np.uint64),
        )
        replacement_facet_sources = vector(
            edit.replacement_facet_sources,
            "replacement_facet_sources",
            np.dtype(np.int32),
        )
        replacement_facet_orientations = (
            None
            if edit.replacement_facet_orientations is None
            else _native_integer_array(
                edit.replacement_facet_orientations,
                "replacement_facet_orientations",
                np.dtype(np.int8),
            )
        )
        replacement_protected_facets = (
            None
            if edit.replacement_protected_facets is None
            else _native_integer_array(
                edit.replacement_protected_facets,
                "replacement_protected_facets",
                np.dtype(np.int8),
            )
        )
        if not (
            replacement_facet_ids.size
            == replacement_facets.shape[0]
            == replacement_facet_sources.size
        ):
            raise ValueError("replacement facet arrays must have matching row counts.")
        for name, value in (
            ("replacement_facet_orientations", replacement_facet_orientations),
            ("replacement_protected_facets", replacement_protected_facets),
        ):
            if value is not None and value.shape != replacement_facet_ids.shape:
                raise ValueError(f"{name} must match replacement facets.")
        removed_segments = vector(
            edit.removed_segment_ids, "removed_segment_ids", np.dtype(np.uint64)
        )
        replacement_segment_ids = vector(
            edit.replacement_segment_ids, "replacement_segment_ids", np.dtype(np.uint64)
        )
        replacement_segments = rows(
            edit.replacement_segment_vertices,
            "replacement_segment_vertices",
            2,
            np.dtype(np.uint64),
        )
        replacement_segment_sources = vector(
            edit.replacement_segment_sources,
            "replacement_segment_sources",
            np.dtype(np.int32),
        )
        replacement_protected_segments = (
            None
            if edit.replacement_protected_segments is None
            else _native_integer_array(
                edit.replacement_protected_segments,
                "replacement_protected_segments",
                np.dtype(np.int8),
            )
        )
        if not (
            replacement_segment_ids.size
            == replacement_segments.shape[0]
            == replacement_segment_sources.size
        ):
            raise ValueError("replacement segment arrays must have matching row counts.")
        if (
            replacement_protected_segments is not None
            and replacement_protected_segments.shape != replacement_segment_ids.shape
        ):
            raise ValueError(
                "replacement_protected_segments must match replacement segments."
            )

        periodic_payload = (
            None
            if edit.periodic is None
            else _tet_mesh_periodic_edit_payload(edit.periodic)
        )
        request = _TetMeshEditRequest()
        request.expected_generation = generation
        request.kind = int(kind)
        pointer_banks = (
            ("collapse_pair_count", collapse_pairs),
            ("collapse_removed_ids", collapse_removed),
            ("collapse_survivor_ids", collapse_pairs[:, 1]),
            ("split_edge_count", split_edges),
            ("split_edges", split_edges),
            ("removed_vertex_count", collapse_removed),
            ("removed_vertex_ids", collapse_removed),
            ("removed_cell_count", removed_cells),
            ("removed_cell_ids", removed_cells),
            ("changed_vertex_count", changed_ids),
            ("changed_vertex_ids", changed_ids),
            ("changed_vertex_points", changed_points),
            ("changed_witness_strata", changed_strata),
            ("changed_witness_entities", changed_entities),
            ("changed_witness_parameters", changed_parameters),
            ("changed_witness_deviations", changed_deviations),
            ("changed_protection_radii", changed_radii),
            ("changed_protected_vertices", changed_protected),
            ("changed_edge_witness_present", edge_witness_present),
            ("changed_edge_witness_endpoints", edge_witness_endpoints),
            ("changed_edge_parameters", edge_parameters),
            ("changed_vertex_ancestry_offsets", changed_ancestry_offsets),
            ("changed_vertex_ancestor_ids", changed_ancestor_ids),
            ("changed_vertex_ancestor_winding", changed_ancestor_winding),
            ("changed_vertex_ancestry_weights", changed_ancestry_weights),
            ("replacement_cell_count", replacement_cell_ids),
            ("replacement_cell_ids", replacement_cell_ids),
            ("replacement_cell_vertices", replacement_cells),
            ("replacement_cell_facet_sources", replacement_cell_sources),
            ("replacement_regions", replacement_regions),
            ("removed_facet_count", removed_facets),
            ("replacement_cell_ancestry_offsets", replacement_ancestry_offsets),
            ("replacement_cell_ancestor_ids", replacement_ancestor_ids),
            ("removed_facet_ids", removed_facets),
            ("replacement_facet_count", replacement_facet_ids),
            ("replacement_facet_ids", replacement_facet_ids),
            ("replacement_facet_vertices", replacement_facets),
            ("replacement_facet_sources", replacement_facet_sources),
            ("replacement_facet_orientations", replacement_facet_orientations),
            ("replacement_protected_facets", replacement_protected_facets),
            ("removed_segment_count", removed_segments),
            ("removed_segment_ids", removed_segments),
            ("replacement_segment_count", replacement_segment_ids),
            ("replacement_segment_ids", replacement_segment_ids),
            ("replacement_segment_vertices", replacement_segments),
            ("replacement_segment_sources", replacement_segment_sources),
            ("replacement_protected_segments", replacement_protected_segments),
        )
        request.changed_vertex_ancestry_count = (
            0 if changed_ancestor_ids is None else changed_ancestor_ids.size
        )
        request.replacement_cell_ancestry_count = (
            0 if replacement_ancestor_ids is None else replacement_ancestor_ids.size
        )
        for name, value in pointer_banks:
            if name.endswith("_count"):
                setattr(request, name, 0 if value is None else value.shape[0])
            else:
                setattr(request, name, _tet_mesh_pointer(value))
        with self._lock:
            identity = self._identity()
            requested_identity = (
                edit.source_id,
                edit.source_revision,
                edit.source_map_id,
                edit.frame_id,
            )
            if all(value is None for value in requested_identity):
                requested_identity = (
                    identity.source_id if identity.identity_bound else None,
                    identity.source_revision if identity.identity_bound else None,
                    identity.source_map_id if identity.identity_bound else None,
                    identity.frame_id if identity.identity_bound else None,
                )
            identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
                *requested_identity
            )
            request.identity_bound = identity_bound
            request.source_id = _tet_mesh_identity_pointer(
                identity_bound, identity_buffers[0]
            )
            request.source_id_length = len(identity_bytes[0])
            request.source_revision = _tet_mesh_identity_pointer(
                identity_bound, identity_buffers[1]
            )
            request.source_revision_length = len(identity_bytes[1])
            request.source_map_id = _tet_mesh_identity_pointer(
                identity_bound, identity_buffers[2]
            )
            request.source_map_id_length = len(identity_bytes[2])
            request.frame_id = _tet_mesh_identity_pointer(
                identity_bound, identity_buffers[3]
            )
            request.frame_id_length = len(identity_bytes[3])
            request.periodic = (
                None
                if periodic_payload is None
                else ctypes.cast(ctypes.pointer(periodic_payload.view), _P)
            )
            edit_status = _I32()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_apply_edit"](
                    self._live(), ctypes.byref(request), ctypes.addressof(edit_status)
                ),
                "tet_mesh_apply_edit",
            )
            return TetMeshEditStatus(edit_status.value)

    def _vertex_size_array(self, sizes: object | None) -> np.ndarray:
        """Validated per-vertex size targets (snapshot order); empty for None."""
        if sizes is None:
            return np.empty((0,), dtype=np.float64)
        size_array = _real_array(sizes, "sizes")
        expected = np.empty((PHX_MC_TET_MESH_SNAPSHOT_SIZES,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_snapshot_sizes"](
                self._live(), expected.ctypes.data
            ),
            "tet_mesh_snapshot_sizes",
        )
        if size_array.shape != (int(expected[0]),):
            raise ValueError("sizes must have one value per current mesh vertex.")
        return size_array

    def refine(
        self, *, radius_edge_bound: float, sizes: object | None = None
    ) -> TetMeshRun:
        """Circumcenter refinement to the radius-edge bound and size targets.

        ``sizes`` (one circumradius target per current vertex, 0 for none) becomes
        the mesh's retained size field; ``None`` keeps the retained field.
        """
        radius_edge = finite_real_scalar(radius_edge_bound, "radius_edge_bound")
        with self._lock:
            identity = self._identity()
            identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
                identity.source_id if identity.identity_bound else None,
                identity.source_revision if identity.identity_bound else None,
                identity.source_map_id if identity.identity_bound else None,
                identity.frame_id if identity.identity_bound else None,
            )
            size_array = self._vertex_size_array(sizes)
            edit_status, run_status = _I32(), _I32()
            inserted, unmet = _I64(), _I64()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_refine"](
                    self._live(),
                    identity.generation,
                    identity_bound,
                    *_tet_mesh_identity_arguments(
                        identity_bound, identity_bytes, identity_buffers
                    ),
                    radius_edge,
                    size_array.size,
                    _tet_mesh_pointer(size_array),
                    ctypes.addressof(edit_status),
                    ctypes.addressof(run_status),
                    ctypes.addressof(inserted),
                    ctypes.addressof(unmet),
                ),
                "tet_mesh_refine",
            )
            return TetMeshRun(
                TetMeshEditStatus(edit_status.value),
                MeshcoreStatus(run_status.value),
                int(inserted.value),
                0,
                0,
                int(unmet.value),
            )

    def improve(
        self,
        *,
        radius_edge_bound: float,
        min_dihedral_degrees: float,
        minimum_relative_determinant: float = 0.0,
        sizes: object | None = None,
    ) -> TetMeshRun:
        """Quality improvement gated by the determinant floor and the size field.

        No edit creates a cell whose circumradius exceeds its size target by more
        than the largest excess among the cells it removes (none where they were
        within target), except one removing a degenerate cell. ``sizes`` follows
        :meth:`refine` (``None`` keeps the retained field, e.g. refinement's). The
        run reports its score quantum.
        """
        radius_edge = finite_real_scalar(radius_edge_bound, "radius_edge_bound")
        minimum_angle = finite_real_scalar(min_dihedral_degrees, "min_dihedral_degrees")
        determinant_floor = finite_real_scalar(
            minimum_relative_determinant, "minimum_relative_determinant"
        )
        with self._lock:
            identity = self._identity()
            identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
                identity.source_id if identity.identity_bound else None,
                identity.source_revision if identity.identity_bound else None,
                identity.source_map_id if identity.identity_bound else None,
                identity.frame_id if identity.identity_bound else None,
            )
            size_array = self._vertex_size_array(sizes)
            edit_status, run_status = _I32(), _I32()
            inserted, collapsed, relocated, unmet = _I64(), _I64(), _I64(), _I64()
            score_quantum = _F64()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_improve_quality"](
                    self._live(),
                    identity.generation,
                    identity_bound,
                    *_tet_mesh_identity_arguments(
                        identity_bound, identity_bytes, identity_buffers
                    ),
                    radius_edge,
                    minimum_angle,
                    determinant_floor,
                    size_array.size,
                    _tet_mesh_pointer(size_array),
                    ctypes.addressof(edit_status),
                    ctypes.addressof(run_status),
                    ctypes.addressof(inserted),
                    ctypes.addressof(collapsed),
                    ctypes.addressof(relocated),
                    ctypes.addressof(unmet),
                    ctypes.addressof(score_quantum),
                ),
                "tet_mesh_improve_quality",
            )
            return TetMeshRun(
                TetMeshEditStatus(edit_status.value),
                MeshcoreStatus(run_status.value),
                int(inserted.value),
                int(collapsed.value),
                int(relocated.value),
                int(unmet.value),
                float(score_quantum.value),
            )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._library["phx_mc_tet_mesh_free"](self._handle)
                self._handle = ctypes.c_void_p()
                self._closed = True

    def __enter__(self) -> TetMesh3D:
        self._live()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        library = getattr(self, "_library", None)
        handle = getattr(self, "_handle", None)
        if library is not None and handle is not None and handle.value:
            try:
                library["phx_mc_tet_mesh_free"](handle)
            except BaseException:
                pass


__all__ = [
    "ARRANGEMENT_UNCLASSIFIED",
    "ArrangementClassification",
    "ArrangementFailure",
    "ConstrainedDelaunay2D",
    "ContactIntersections",
    "IntersectionClass",
    "MeshcoreError",
    "MeshcoreLibrary",
    "MixedElementKind",
    "MixedTemplateCounts",
    "mixed_template",
    "mixed_template_counts",
    "MeshcoreStatus",
    "MeshcoreUnavailableError",
    "NativeArrangement",
    "NativeReferenceTetrahedronClip",
    "NativeReferenceTriangleClip",
    "NativeTetrahedraArrangement",
    "PowerCellFailureReason",
    "PlcBoundaryPolicy",
    "PlcEntityKind",
    "PlcFailureReason",
    "PlcRecovery3D",
    "PlcSourceData",
    "PlcSuppressionSearch",
    "PlcSourceFailure",
    "prepare_plc_source",
    "recover_plc",
    "recover_plc_3d",
    "TetMesh3D",
    "TetMeshArrays",
    "TetMeshEdit",
    "TetMeshEditKind",
    "TetMeshEditStatus",
    "TetMeshPeriodicEdit",
    "TetMeshPeriodicEntityOrbitMap",
    "TetMeshPeriodicIdentityMap",
    "TetMeshPeriodicMember",
    "TetMeshPeriodicOrbit",
    "TetMeshPeriodicQuotientCell",
    "TetMeshPeriodicSnapshot",
    "TetMeshPeriodicState",
    "TetMeshPlaneRecords",
    "TetMeshPlaneRefusal",
    "TetMeshRun",
    "TetMeshSourceComplex",
    "TetMeshSourceEvidence",
    "TetMeshUnmet",
    "TetMeshUnmetReason",
    "TET_MESH_UNMET_NO_APEX",
    "QuadraticValidityDecision",
    "RestrictedPowerCells",
    "RestrictedDual3D",
    "RestrictedPowerFailure",
    "arrange_triangles",
    "arrange_tetrahedra_exact",
    "clip_box_halfplanes",
    "clip_reference_tetrahedron_exact",
    "clip_reference_triangle_exact",
    "clip_box_halfspaces",
    "constrained_delaunay_2d",
    "delaunay_2d",
    "delaunay_3d",
    "exact_incircle",
    "exact_insphere",
    "exact_orient2d",
    "exact_orient3d",
    "graph_partition",
    "parallel_for",
    "interval_program_execute",
    "load_meshcore",
    "meshcore_available",
    "meshcore_exact_domain",
    "meshcore_identity",
    "point_triangle_locations",
    "polygon_intersection_moments",
    "polygon_intersection_simplices",
    "polyhedron_clip_moments",
    "quadratic_tetrahedron_validity",
    "regular_2d",
    "regular_3d",
    "restricted_power_cells",
    "restricted_dual_3d",
    "restricted_centers_3d",
    "restricted_rays_3d",
    "restricted_power_cells_exact",
    "segment_triangle_intersections",
    "simplex_quadratic_bounds",
    "symbolic_incircle",
    "symbolic_insphere",
    "symbolic_orient2d",
    "symbolic_orient3d",
    "tetrahedron_intersection_moments",
    "tetrahedron_intersection_simplices",
    "tetrahedron_dual_hexes",
    "triangle_dual_quads",
    "triangle_intersection_classes",
    "triangle_intersections",
    "SurfaceEnvelopeResult",
    "surface_envelope",
]
