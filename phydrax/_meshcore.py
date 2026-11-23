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
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any, final, NamedTuple

import numpy as np

from ._config_environment import LIBRARY_ENVIRONMENT
from ._config_state import library_input
from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
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
    STALE_GENERATION = 12
    IO_FAILURE = 13


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
_INTP_MAX = int(np.iinfo(np.intp).max)
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


class _OversetParts(ctypes.Structure):
    _fields_ = (
        ("part_count", _I32),
        ("vertex_offsets", _P),
        ("vertices", _P),
        ("cell_offsets", _P),
        ("cells", _P),
        ("cell_ids", _P),
        ("cell_excluded", _P),
        ("face_offsets", _P),
        ("faces", _P),
        ("face_roles", _P),
        ("priorities", _P),
        ("name_ranks", _P),
        ("fringe_layers", _I32),
        ("donor_priority", _I32),
        ("fringe_donors", _I32),
        ("prove", _I32),
    )


class _OversetResult(ctypes.Structure):
    _fields_ = (
        ("vertex_status", _P),
        ("fringe_layer", _P),
        ("wall_contact", _P),
        ("protected_conflict", _P),
        ("cell_status", _P),
        ("ambiguous", _P),
        ("donor_status", _P),
        ("donor_cell", _P),
        ("donor_reference", _P),
        ("donor_residual", _P),
        ("donor_measure", _P),
        ("candidate_parts", _P),
        ("stats", _P),
    )


class _PrismLayersRequest(ctypes.Structure):
    _fields_ = (
        ("vertex_count", _I64),
        ("vertices", _P),
        ("triangle_count", _I64),
        ("triangles", _P),
        ("layer_count", _I32),
        ("heights", _P),
        ("feature_angle_degrees", _F64),
        ("minimum_scale", _F64),
        ("prove", _I32),
        ("source_normals", _P),
        ("source_shape_operators", _P),
    )


class _PrismLayersResult(ctypes.Structure):
    _fields_ = (
        ("points", _P),
        ("prisms", _P),
        ("displacement", _P),
        ("scale", _P),
        ("vertex_kind", _P),
        ("stats", _P),
    )


class _ConformingVoronoiRequest(ctypes.Structure):
    _fields_ = (
        ("vertex_count", _I64),
        ("vertices", _P),
        ("triangle_count", _I64),
        ("triangles", _P),
        ("triangle_regions", _P),
        ("triangle_facets", _P),
        ("size", _F64),
        ("feature_angle_degrees", _F64),
        ("prove", _I32),
        ("result", ctypes.POINTER(ctypes.c_void_p)),
    )


class _ConformingVoronoiArrays(ctypes.Structure):
    _fields_ = (
        ("vertices", _P),
        ("vertex_triangles", _P),
        ("face_offsets", _P),
        ("face_vertices", _P),
        ("face_cells", _P),
        ("face_facets", _P),
        ("cell_seeds", _P),
        ("cell_regions", _P),
        ("cell_volumes", _P),
        ("cell_centroids", _P),
        ("seeds", _P),
        ("seed_regions", _P),
        ("stats", _P),
        ("exact_offsets", _P),
        ("exact_values", _P),
    )


class _HexDominantRequest(ctypes.Structure):
    _fields_ = (
        ("vertex_count", _I64),
        ("vertices", _P),
        ("triangle_count", _I64),
        ("triangles", _P),
        ("triangle_facets", _P),
        ("core_size", _F64),
        ("boundary_size", _F64),
        ("feature_angle_degrees", _F64),
        ("layer_count", _I32),
        ("layer_thicknesses", _P),
        ("prove", _I32),
        ("result", ctypes.POINTER(ctypes.c_void_p)),
    )


class _HexDominantArrays(ctypes.Structure):
    _fields_ = (
        ("points", _P),
        ("face_offsets", _P),
        ("face_vertices", _P),
        ("face_owner", _P),
        ("face_neighbor", _P),
        ("face_facets", _P),
        ("cell_kinds", _P),
        ("cell_roles", _P),
        ("cell_vertex_offsets", _P),
        ("cell_vertices", _P),
        ("vertex_triangles", _P),
        ("vertex_kinds", _P),
        ("stats", _P),
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
        ("selected_site_count", _I64),
        ("selected_sites", _P),
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


class _TetMeshImplicitFacetRefusal(ctypes.Structure):
    _fields_ = (
        ("refused", _I32),
        ("generation", _U64),
        ("facet_id", _U64),
        ("facet_source", _I32),
        ("orientation", _I32),
        ("vertex_ids", _U64 * 3),
        ("points", _F64 * 9),
        ("transversality_refusals", _I64),
        ("distance_refusals", _I64),
        ("status", _I32),
        ("has_distance_barycentric", _I32),
        ("distance_barycentric", _F64 * 3),
    )


class _TetrahedraVerdict(ctypes.Structure):
    _fields_ = (
        ("status", _I32),
        ("boundary_faces", _I64),
        ("boundary_shells", _I64),
        ("outside_domain_points", _I64),
        ("first_outside_domain_point", _I64),
        ("nonpositive_cells", _I64),
        ("first_nonpositive_cell", _I64),
        ("nonmanifold_faces", _I64),
        ("first_nonmanifold_cell", _I64),
        ("nonmanifold_boundary_edges", _I64),
        ("first_nonmanifold_boundary_cell", _I64),
        ("boundary_contacts", _I64),
        ("first_contact_cell", _I64),
        ("second_contact_cell", _I64),
        ("misnested_shells", _I64),
        ("first_misnested_shell_cell", _I64),
        ("unresolved_shells", _I64),
        ("first_unresolved_shell_cell", _I64),
    )


class _VolumeVerdict(ctypes.Structure):
    _fields_ = (
        ("certified", _I32),
        ("outside_domain_points", _I64),
        ("first_outside_domain_point", _I64),
        ("nonpositive_cells", _I64),
        ("first_nonpositive_cell", _I64),
        ("nonmanifold_faces", _I64),
        ("first_nonmanifold_cell", _I64),
        ("boundary_faces", _I64),
        ("unmatched_boundary_faces", _I64),
        ("first_unmatched_cell", _I64),
        ("unmatched_surface_triangles", _I64),
        ("first_unmatched_surface_triangle", _I64),
        ("region_mismatches", _I64),
        ("first_region_mismatch_cell", _I64),
    )


class _CadModel(ctypes.Structure):
    _fields_ = (
        ("node_count", _I64),
        ("node_kinds", _P),
        ("node_integer_offsets", _P),
        ("node_integers", _P),
        ("node_real_offsets", _P),
        ("node_reals", _P),
        ("corner_count", _I64),
        ("corner_points", _P),
        ("surface_count", _I64),
        ("surface_nodes", _P),
        ("surface_orientations", _P),
        ("surface_periods", _P),
        ("surface_parameter_domains", _P),
        ("curve_count", _I64),
        ("curve_nodes", _P),
        ("curve_corners", _P),
        ("curve_ranges", _P),
        ("loop_count", _I64),
        ("loop_use_offsets", _P),
        ("loop_surfaces", _P),
        ("use_count", _I64),
        ("use_kinds", _P),
        ("use_entities", _P),
        ("use_senses", _P),
        ("use_ranges", _P),
        ("use_pcurve_nodes", _P),
        ("region_count", _I64),
        ("region_surface_offsets", _P),
        ("region_surfaces", _P),
        ("region_orientations", _P),
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


class _IntervalBoxProgram(ctypes.Structure):
    _fields_ = (
        ("instructions", _P),
        ("instruction_count", _I64),
        ("indices", _P),
        ("index_count", _I64),
        ("parameters", _P),
        ("parameter_count", _I64),
        ("workspace_lower", _P),
        ("workspace_upper", _P),
        ("workspace_size", _I64),
        ("outputs", _P),
        ("output_count", _I64),
        ("guards", _P),
        ("guard_count", _I64),
    )


class _TetImplicitSource(ctypes.Structure):
    _fields_ = (
        ("value", _IntervalBoxProgram),
        ("gradient", _IntervalBoxProgram),
        ("value_events", _P),
        ("value_event_count", _I64),
        ("groups", _P),
        ("group_count", _I64),
        ("tolerance", _F64),
        ("orientation", _I32),
        ("source_id", _P),
        ("source_id_length", _I64),
        ("source_revision", _P),
        ("source_revision_length", _I64),
        ("binding_id", _P),
        ("binding_id_length", _I64),
        ("frame_id", _P),
        ("frame_id_length", _I64),
        ("transversality_refusals", _I64),
        ("distance_refusals", _I64),
    )


class _ZeroSetRemeshRequest(ctypes.Structure):
    _fields_ = (
        ("vertices", _P),
        ("vertex_count", _I64),
        ("triangles", _P),
        ("triangle_count", _I64),
        ("value", _IntervalBoxProgram),
        ("tolerance", _F64),
        ("carrier_distance", _F64),
        ("edge_length", _F64),
        ("minimum_angle_degrees", _F64),
        ("result", _P),
        ("counters", _P),
    )


class _ZeroSetCertifyRequest(ctypes.Structure):
    _fields_ = (
        ("vertices", _P),
        ("vertex_count", _I64),
        ("triangles", _P),
        ("triangle_count", _I64),
        ("value", _IntervalBoxProgram),
        ("gradient", _IntervalBoxProgram),
        ("value_events", _P),
        ("value_event_count", _I64),
        ("tolerance", _F64),
        ("orientation", _I32),
        ("carrier_vertices", _P),
        ("carrier_vertex_count", _I64),
        ("carrier_triangles", _P),
        ("carrier_triangle_count", _I64),
        ("carrier_boxes", _P),
        ("domain", _P),
        ("edge_length", _F64),
        ("minimum_angle_degrees", _F64),
        ("cover_budget", _I64),
        ("result", _P),
        ("distances", _P),
        ("counters", _P),
    )


class _MeshRemeshRequest(ctypes.Structure):
    _fields_ = (
        ("vertices", _P),
        ("vertex_count", _I64),
        ("triangles", _P),
        ("triangle_count", _I64),
        ("tolerance", _F64),
        ("edge_length", _F64),
        ("minimum_angle_degrees", _F64),
        ("result", _P),
        ("counters", _P),
    )


class _MeshCertifyRequest(ctypes.Structure):
    _fields_ = (
        ("source_vertices", _P),
        ("source_vertex_count", _I64),
        ("source_triangles", _P),
        ("source_triangle_count", _I64),
        ("vertices", _P),
        ("vertex_count", _I64),
        ("triangles", _P),
        ("triangle_count", _I64),
        ("tolerance", _F64),
        ("edge_length", _F64),
        ("minimum_angle_degrees", _F64),
        ("result", _P),
        ("distances", _P),
        ("counters", _P),
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
        ("search_node_budget", _I64),
        ("search_byte_budget", _I64),
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
        ("initial_vertex_sizes", _P),
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
        ("vertex_size_targets", _P),
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
        ("cell_vertex_rows", _P),
        ("cell_regions", _P),
        ("cell_size_targets", _P),
        ("facet_ids", _P),
        ("facet_vertex_rows", _P),
        ("facet_sources", _P),
        ("facet_orientations", _P),
        ("protected_facets", _P),
        ("segment_ids", _P),
        ("segment_vertex_rows", _P),
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


class _MetricAdaptRequest(ctypes.Structure):
    _fields_ = (
        ("source_count", _I64),
        ("source_points", _P),
        ("source_metrics", _P),
        ("source_logarithms", _P),
        ("source_vertex_ids", _P),
        ("source_cell_count", _I64),
        ("source_cells", _P),
        ("vertex_count", _I64),
        ("vertex_ids", _P),
        ("stencil_offsets", _P),
        ("stencil_rows", _P),
        ("stencil_weights", _P),
        ("vertex_target_ids", _P),
        ("cell_count", _I64),
        ("cell_ids", _P),
        ("cell_blocks", _P),
        ("cell_target_ids", _P),
        ("ancestry_offsets", _P),
        ("ancestry_ids", _P),
        ("ancestry_kinds", _P),
        ("protected_edge_count", _I64),
        ("protected_edges", _P),
        ("next_vertex_id", _U64),
        ("next_cell_id", _U64),
        ("next_target_vertex", _U64),
        ("next_target_cell", _U64),
        ("lower", ctypes.c_double),
        ("topology_operations", _I32),
        ("relocation", _I32),
        ("exact_source", _P),
    )


class _MetricAdaptationOutput(ctypes.Structure):
    _fields_ = (
        ("status", _P),
        ("passes", _P),
        ("counters", _P),
        ("vertex_ids", _P),
        ("vertex_target_ids", _P),
        ("stencil_offsets", _P),
        ("stencil_rows", _P),
        ("stencil_weights", _P),
        ("metrics", _P),
        ("cell_ids", _P),
        ("cell_target_ids", _P),
        ("cell_blocks", _P),
        ("ancestry_offsets", _P),
        ("ancestry_ids", _P),
        ("ancestry_kinds", _P),
        ("cell_relocated", _P),
        ("vertex_relation_sources", _P),
        ("vertex_relation_targets", _P),
        ("vertex_relation_kinds", _P),
        ("unrepresentable_edges", _P),
    )


_SIGNATURES: dict[str, tuple[object, tuple[object, ...]]] = {
    "phx_mc_version": (ctypes.c_char_p, ()),
    "phx_mc_build_hash": (ctypes.c_char_p, ()),
    "phx_mc_exact_domain": (None, (_P, _P)),
    "phx_mc_symmetric_polynomial_minors": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32),
    ),
    "phx_mc_polynomial_jacobian_integrals": (
        _I32,
        (_I64, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _I32, _I32),
    ),
    "phx_mc_basis_jacobian_integrals": (
        _I32,
        (
            _I64,
            _I32,
            _I32,
            _I64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I32,
            _I32,
        ),
    ),
    "phx_mc_symmetric_polynomial_verdicts": (
        _I32,
        (_I64, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _I32, _I32),
    ),
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
    "phx_mc_tetrahedron_intersection_quadratic_moments": (
        _I32,
        (_I64, _P, _P, _P, _P, _P, _P),
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
    "phx_mc_interval_program_batch": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _P, _I64, _I64, _I32),
    ),
    "phx_mc_interval_program_branches": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _P, _I64, _P, _I64, _I64, _P, _P),
    ),
    "phx_mc_program_evaluate": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _I64, _I64, _P, _I64, _P, _I64, _P),
    ),
    "phx_mc_program_jet": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _I64, _I64, _P, _I64, _P, _I64, _P),
    ),
    "phx_mc_verify_tetrahedra": (_I32, (_I64, _P, _I64, _P, _P)),
    "phx_mc_verify_volume": (_I32, (_I64, _P, _I64, _P, _P, _I64, _P, _P, _P)),
    "phx_mc_tet_mesh_entities": (
        _I32,
        (_I64, _I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_polyhedral_mesh_entities": (
        _I32,
        (_I64, _I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_incidence_order": (_I32, (_I64, _I64, _I64, _P, _P, _P, _P)),
    "phx_mc_verify_surface_embedding": (_I32, (_I64, _P, _I64, _P, _I64, _P, _P)),
    "phx_mc_verify_chart_partition": (
        _I32,
        (_I64, _I64, _P, _P, _I64, _P, _P, _I64, _I64, _P, _P, _P, _P, _I64, _P, _P),
    ),
    "phx_mc_tetrahedra_validity": (_I32, (_I64, _P, _I64, _P, _F64, _P, _P, _P)),
    "phx_mc_cell_map_validity": (
        _I32,
        (
            _I32,
            _I64,
            _P,
            _P,
            _P,
            _I32,
            _P,
            _I64,
            _I64,
            _P,
            _P,
            ctypes.c_int16,
            _I64,
            _P,
            _F64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I32,
        ),
    ),
    "phx_mc_welded_topology": (
        _I32,
        (
            _I32,
            _I32,
            _I32,
            _I32,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _I32,
        ),
    ),
    "phx_mc_polyhedral_star_validity": (
        _I32,
        (_I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _F64, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_polyhedral_star_decomposition": (
        _I32,
        (_I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_face_triangulation": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_polyhedral_declared_contacts": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _I32,
        ),
    ),
    "phx_mc_cad_geometry_evaluate": (_I32, (_P, _I32, _I64, _I64, _P, _I32, _P)),
    "phx_mc_cad_mesh_surface": (_I32, (_P, _P, _P, _I32)),
    "phx_mc_cad_surface_result_sizes": (_I32, (_P, _P)),
    "phx_mc_cad_surface_result_export": (_I32, (_P, _P)),
    "phx_mc_cad_surface_result_free": (None, (_P,)),
    "phx_mc_cad_surface_pole_sectors": (_I32, (_P, _I64, _P, _P)),
    "phx_mc_cad_surface_refinement": (_I32, (_P, _I64, _P, _P, _P)),
    "phx_mc_cad_chart_sizes": (_I32, (_P, _P)),
    "phx_mc_cad_chart_export": (_I32, (_P, _P)),
    "phx_mc_cad_mesh_volume": (_I32, (_P, _P, _P, _P, _I32)),
    "phx_mc_cad_volume_sizes": (_I32, (_P, _P)),
    "phx_mc_cad_volume_export": (_I32, (_P, _P)),
    "phx_mc_implicit_restricted_delaunay": (
        _I32,
        (
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
        ),
    ),
    "phx_mc_restricted_volume_sizes": (_I32, (_P, _P, _P)),
    "phx_mc_restricted_volume_copy": (_I32, (_P, _P, _P, _P)),
    "phx_mc_restricted_volume_free": (None, (_P,)),
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
    "phx_mc_interval_simplex_restrict": (
        _I32,
        (_I32, _I32, _I64, _P, _P, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_interval_basis_bounds": (
        _I32,
        (_I64, _I64, _I64, _I64, _P, _P, _P, _P, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_interval_bernstein_restrict": (
        _I32,
        (_I64, _I32, _I64, _P, _P, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_interval_multiply": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_interval_sum": (_I32, (_I64, _P, _P, _P, _P, _I32, _P, _P)),
    "phx_mc_interval_elementary": (_I32, (_I32, _I64, _P, _P, _P, _P, _I32)),
    "phx_mc_bernstein_controls": (
        _I32,
        (_I64, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32),
    ),
    "phx_mc_simplex_bernstein_blossoms": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _P, _P, _P, _I32, _I32),
    ),
    "phx_mc_bernstein_bisect": (
        _I32,
        (_I64, _I64, _I32, _I32, _P, _I32, _I32, _P, _P, _P, _P, _P, _I32, _I32),
    ),
    "phx_mc_separation_screen": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_bvh_build": (
        _I32,
        (_I32, _I64, _I32, _P, _P, _P, _I64, _I32, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_bvh_pair_stream_create": (
        _I32,
        (_P, _P, _I32, _I32, _F64, _F64, _I64, _I64, _I32, _P),
    ),
    "phx_mc_bvh_pair_stream_next": (_I32, (_P, _I64, _I64, _P, _P, _P, _P)),
    "phx_mc_bvh_pair_stream_free": (None, (_P,)),
    "phx_mc_bvh_query_nearest_triangles": (
        _I32,
        (_P, _I64, _P, _P, _I64, _P, _I32, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_bvh_query_signed_distance": (
        _I32,
        (_P, _I64, _P, _P, _P, _I64, _P, _P, _P, _P, _I32),
    ),
    "phx_mc_bvh_query_winding": (
        _I32,
        (_P, _I64, _P, _P, _I32, _P, _I64, _P, _P, _F64, _I64, _P, _P, _I32),
    ),
    "phx_mc_bvh_query_locate_tetrahedra": (
        _I32,
        (
            _P,
            _I64,
            _P,
            _P,
            _P,
            _F64,
            _I64,
            _P,
            _I32,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I32,
        ),
    ),
    "phx_mc_overset_assemble": (
        _I32,
        (ctypes.POINTER(_OversetParts), ctypes.POINTER(_OversetResult)),
    ),
    "phx_mc_prism_layers": (
        _I32,
        (ctypes.POINTER(_PrismLayersRequest), ctypes.POINTER(_PrismLayersResult)),
    ),
    "phx_mc_conforming_voronoi": (_I32, (ctypes.POINTER(_ConformingVoronoiRequest),)),
    "phx_mc_conforming_voronoi_sizes": (_I32, (_P, _P)),
    "phx_mc_conforming_voronoi_export": (
        _I32,
        (_P, ctypes.POINTER(_ConformingVoronoiArrays)),
    ),
    "phx_mc_conforming_voronoi_free": (None, (_P,)),
    "phx_mc_hex_dominant": (_I32, (ctypes.POINTER(_HexDominantRequest),)),
    "phx_mc_hex_dominant_sizes": (_I32, (_P, _P)),
    "phx_mc_hex_dominant_export": (_I32, (_P, ctypes.POINTER(_HexDominantArrays))),
    "phx_mc_hex_dominant_free": (None, (_P,)),
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
    "phx_mc_power_cells_export_face_remotes": (_I32, (_P, _I64, _P, _P)),
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
    "phx_mc_point_triangle_distances": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_point_triangle_nearest_distances": (_I32, (_I64, _P, _I64, _P, _P)),
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
    "phx_mc_restricted_centers_3d": (_I32, (_I64, _P, _I64, _P, _P, _P)),
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
    "phx_mc_surface_envelope_remesh": (_I32, (_P, _P, _P)),
    "phx_mc_zero_set_remesh": (_I32, (ctypes.POINTER(_ZeroSetRemeshRequest),)),
    "phx_mc_zero_set_remesh_i32": (_I32, (ctypes.POINTER(_ZeroSetRemeshRequest),)),
    "phx_mc_zero_set_certify": (_I32, (ctypes.POINTER(_ZeroSetCertifyRequest),)),
    "phx_mc_mesh_remesh": (_I32, (ctypes.POINTER(_MeshRemeshRequest),)),
    "phx_mc_mesh_certify": (_I32, (ctypes.POINTER(_MeshCertifyRequest),)),
    "phx_mc_surface_result_sizes": (_I32, (_P, _P)),
    "phx_mc_surface_result_arrays": (_I32, (_P, _I64, _I64, _P, _P)),
    "phx_mc_surface_result_carried": (_I32, (_P, _I64, _P)),
    "phx_mc_surface_result_free": (None, (_P,)),
    "phx_mc_level_set_split_3d": (_I32, (ctypes.POINTER(_LevelSetSplit3DRequest),)),
    "phx_mc_implicit_contours": (
        _I32,
        (_I64, _I64, *(_P,) * 7, _F64, *(_P,) * 4),
    ),
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
    "phx_mc_plc3d_recover_with_affine_metric": (
        _I32,
        (ctypes.POINTER(_Plc3DPrepareSourceRequest), _P),
    ),
    "phx_mc_plc3d_recover_protected": (
        _I32,
        (ctypes.POINTER(_Plc3DPrepareSourceRequest), _P, _P),
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
    "phx_mc_plc3d_suppression_routes": (_I32, (_P, _I64, _P, _P)),
    "phx_mc_plc3d_free": (None, (_P,)),
    "phx_mc_tet_mesh_create": (
        _I32,
        (ctypes.POINTER(_TetMeshCreateRequest),),
    ),
    "phx_mc_tet_mesh_create_with_affine_metric": (
        _I32,
        (ctypes.POINTER(_TetMeshCreateRequest), _P),
    ),
    "phx_mc_tet_mesh_create_with_implicit": (
        _I32,
        (ctypes.POINTER(_TetMeshCreateRequest), ctypes.POINTER(_TetImplicitSource)),
    ),
    "phx_mc_tet_mesh_create_from_plc": (
        _I32,
        (_P, ctypes.POINTER(_TetMeshCreateRequest), _P, _I32),
    ),
    "phx_mc_tet_mesh_bind_implicit": (_I32, (_P, ctypes.POINTER(_TetImplicitSource))),
    "phx_mc_tet_mesh_implicit_source": (_I32, (_P, ctypes.POINTER(_TetImplicitSource))),
    "phx_mc_tet_mesh_protect_facet_sources": (_I32, (_P, _I64, _P)),
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
    "phx_mc_tet_mesh_unmet": (_I32, (_P, _I64, _P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_evaluate_quality": (
        _I32,
        (_P, _U64, _F64, _F64, _F64, _I64, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_evaluate_candidate_quality": (
        _I32,
        (_P, _U64, _I64, _P, _F64, _F64, _F64, _I64, _P, _P, _P, _P, _P),
    ),
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
    "phx_mc_tet_mesh_probe_edit": (
        _I32,
        (_P, ctypes.POINTER(_TetMeshEditRequest), _P, _P),
    ),
    "phx_mc_tet_mesh_probe_relocations": (
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
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_tet_mesh_apply_relocation": (
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
            _I64,
            _P,
            _P,
            _I64,
            _P,
            ctypes.POINTER(_I32),
            ctypes.POINTER(_I32),
        ),
    ),
    "phx_mc_tet_mesh_metric_adapt": (
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
            ctypes.POINTER(_MetricAdaptRequest),
            ctypes.POINTER(_I32),
            ctypes.POINTER(_P),
        ),
    ),
    "phx_mc_metric_adaptation_sizes": (_I32, (_P, _P)),
    "phx_mc_metric_adaptation_read": (
        _I32,
        (_P, ctypes.POINTER(_MetricAdaptationOutput)),
    ),
    "phx_mc_metric_adaptation_free": (None, (_P,)),
    "phx_mc_tet_mesh_implicit_facet_refusal": (
        _I32,
        (_P, ctypes.POINTER(_TetMeshImplicitFacetRefusal)),
    ),
    "phx_mc_tet_mesh_clear_implicit_facet_refusal": (_I32, (_P,)),
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
            _I32,
            _I64,
            _P,
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
    "phx_mc_delaunay_2d_into": (_I32, (_I64, _P, _I64, _P, _P, _P)),
    "phx_mc_regular_2d": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_delaunay_3d": (_I32, (_I64, _P, _P, _I32)),
    "phx_mc_regular_3d": (_I32, (_I64, _P, _P, _P, _I32)),
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
    "phx_mc_mesh_shape": (_I32, (_P, _P)),
    "phx_mc_mesh_copy_all": (_I32, (_P, _I64, _I64, _I64, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_mesh_free": (None, (_P,)),
    "phx_mc_deterministic_float_self_test": (_I32, (ctypes.POINTER(_I32),)),
    "phx_mc_metric_logarithms": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_metric_interpolate": (_I32, (_I64, _P, _P, _P, _I64, _P, _P, _P, _P)),
    "phx_mc_metric_lengths": (_I32, (_I64, _P, _I64, _P, _P, _P)),
    "phx_mc_metric_band_quanta": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_metric_relative_determinants": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_metric_tetra_quality": (_I32, (_I64, _P, _I64, _P, _P, _P)),
    "phx_mc_metric_midpoints": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_metric_survivor_goals": (_I32, (_I64, _P, _P, _I32, _P, _P, _P)),
    "phx_mc_metric_tensor_properties": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_metric_congruence_defects": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_metric_hull_frame": (_I32, (_I32, _P, _P, _P, ctypes.POINTER(_I32))),
    "phx_mc_metric_project_goals": (_I32, (_I64, _P, _P, _P, _P)),
    "phx_mc_metric_relocation_pulls": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _P,
            _P,
            _I32,
            _I32,
            _I32,
            _P,
            _P,
            _P,
            _I64,
            _P,
            ctypes.POINTER(_I64),
        ),
    ),
    "phx_mc_metric_source_create": (
        _I32,
        (_I64, _P, _P, _P, _P, _I64, _P, _P, _P, ctypes.POINTER(ctypes.c_void_p)),
    ),
    "phx_mc_metric_source_free": (None, (_P,)),
    "phx_mc_metric_source_carriers": (
        _I32,
        (
            _P,
            _P,
            _P,
            _I64,
            _I64,
            _P,
            _P,
            _I32,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            ctypes.POINTER(_I64),
        ),
    ),
    "phx_mc_metric_collapse_positions": (
        _I32,
        (_I64, _P, _P, _I64, _P, _I64, _P, _I64, _P, _F64, _I64, _P, _P, _P, _P, _P),
    ),
    "phx_mc_metric_source_survivor_moves": (
        _I32,
        (
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _F64,
            _I64,
            _P,
            _P,
            _I32,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            ctypes.POINTER(_I64),
            ctypes.POINTER(_I32),
        ),
    ),
    "phx_mc_planar_partition": (
        _I32,
        (_I64, _I64, _I64, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_angle_sine": (_I32, (_I64, _P, _P)),
    "phx_mc_angle_cosine": (_I32, (_I64, _P, _P)),
    "phx_mc_angle_arcsine": (_I32, (_I64, _P, _P)),
    "phx_mc_angle_arccosine": (_I32, (_I64, _P, _P)),
    "phx_mc_angle_between": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_angle_arctangent2": (_I32, (_I64, _P, _P, _P)),
    "phx_mc_quality_dihedral_angles": (_I32, (_I64, _I64, _P, _I64, _P, _P)),
    "phx_mc_quality_solid_corners": (
        _I32,
        (_I64, _I64, _P, _I64, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_metric_relocation_create": (
        _I32,
        (
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _F64,
            _I64,
            _P,
            _F64,
            _F64,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _I32,
            ctypes.POINTER(_P),
        ),
    ),
    "phx_mc_metric_relocation_free": (None, (_P,)),
    "phx_mc_metric_relocation_next": (
        _I32,
        (
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            ctypes.POINTER(_I64),
            ctypes.POINTER(_I32),
        ),
    ),
}

# CAD fidelity verifier.
_SIGNATURES["phx_mc_verify_cad_fidelity"] = (
    _I32,
    (
        _P,
        _I64,
        _P,
        _P,
        _I64,
        _P,
        _I64,
        _P,
        _P,
        _P,
        _I64,
        _P,
        _P,
        _I64,
        _P,
        _P,
        _I64,
        _I64,
        _P,
        _P,
        _P,
        _P,
        _P,
        _F64,
        _I32,
        _I64,
        _P,
        _P,
        _P,
        _P,
        _P,
        _I64,
        _P,
        _P,
    ),
)
_SIGNATURES["phx_mc_verify_cad_quadratic_fidelity"] = (
    _I32,
    (*_SIGNATURES["phx_mc_verify_cad_fidelity"][1], _P),
)


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
        "_implicit_face_proposals",
        "_implicit_face_proposals_checked",
        "_implicit_face_proposals_lock",
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
        self._implicit_face_proposals: Any = None
        self._implicit_face_proposals_checked = False
        self._implicit_face_proposals_lock = threading.Lock()

    def __getitem__(self, name: str, /) -> Any:
        return self._functions[name]

    def implicit_face_proposals(self, /) -> Any:
        """Resolve the optional proposal entry once for this loaded library."""
        if self._implicit_face_proposals_checked:
            return self._implicit_face_proposals
        with self._implicit_face_proposals_lock:
            if not self._implicit_face_proposals_checked:
                # ctypes reports a missing exported symbol as AttributeError.
                try:
                    function = self._library["phx_mc_implicit_face_proposals"]
                except AttributeError:
                    function = None
                if function is not None:
                    function.restype = _I32
                    function.argtypes = (_P, _I64, _P, _P, _I64, _P, _P, _P)
                self._implicit_face_proposals = function
                self._implicit_face_proposals_checked = True
            return self._implicit_face_proposals

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
    # The deterministic binary64 kernels give one result per input only under
    # the arithmetic they assume; a library without it is not this release.
    failed_check = _I32(0)
    if functions["phx_mc_deterministic_float_self_test"](ctypes.byref(failed_check)):
        reasons = {
            1: "binary64 rounding is not to nearest even",
            2: "a product and a sum are fused into one operation",
            3: "subnormal numbers are flushed to zero",
            4: "a kernel result differs from its recorded bits",
        }
        reason = reasons.get(failed_check.value, "the floating-point mode cannot be set")
        return (
            f"meshcore library {str(path)!r} fails its deterministic arithmetic "
            f"self-test: {reason}."
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
    if array.dtype == np.dtype(np.float64):
        return np.ascontiguousarray(array)
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
        case MeshcoreStatus.STALE_GENERATION:
            raise MeshcoreError(code, f"{operation}: mesh generation is stale")
        case MeshcoreStatus.IO_FAILURE:
            raise MeshcoreError(code, f"{operation}: native scratch storage failed")
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
    # Values already of the native dtype are in its range by construction.
    if array.dtype != dtype and array.size:
        bounds = np.iinfo(dtype)
        if array.dtype.kind == "u":
            if np.any(array > bounds.max):
                raise OverflowError(f"{name} exceeds the native integer range.")
        elif np.any(array < bounds.min) or np.any(array > bounds.max):
            raise OverflowError(f"{name} exceeds the native integer range.")
    return np.ascontiguousarray(array, dtype=dtype)


def _interval_program_arrays(
    instructions: object,
    indices: object,
    parameters: object,
    lower: object,
    upper: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return instruction_array, index_array, parameter_array, lower_array, upper_array


def implicit_face_proposals(
    coefficients: np.ndarray, face_rows: np.ndarray, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray] | None:
    """Propose implicit-face parameters, or return None for native fallback.

    Coefficients have shape ``(face_count, 3, 10)``, face rows shape
    ``(count,)``, and points shape ``(count, 3)``. The returned writable
    float64 arrays have shape ``(count,)``. Inputs must be aligned,
    C-contiguous float64 arrays and int64 face rows. None means the inputs
    are unsuitable, the optional entry is unavailable, or native declined
    the entire batch; no proposal is then usable.
    """
    if (
        not isinstance(coefficients, np.ndarray)
        or coefficients.dtype != np.dtype(np.float64)
        or coefficients.ndim != 3
        or coefficients.shape[1:] != (3, 10)
        or not coefficients.flags.c_contiguous
        or not coefficients.flags.aligned
        or not isinstance(face_rows, np.ndarray)
        or face_rows.dtype != np.dtype(np.int64)
        or face_rows.ndim != 1
        or not face_rows.flags.c_contiguous
        or not face_rows.flags.aligned
        or not isinstance(points, np.ndarray)
        or points.dtype != np.dtype(np.float64)
        or points.ndim != 2
        or points.shape[1] != 3
        or not points.flags.c_contiguous
        or not points.flags.aligned
        or face_rows.shape[0] != points.shape[0]
        or np.any(face_rows < 0)
        or np.any(face_rows >= coefficients.shape[0])
    ):
        return None
    library = _current()
    if isinstance(library, str):
        return None
    function = library.implicit_face_proposals()
    if function is None:
        return None
    count = points.shape[0]
    first = np.empty((count,), dtype=np.float64)
    second = np.empty((count,), dtype=np.float64)
    handled = _I32(0)
    status = function(
        coefficients.ctypes.data,
        coefficients.shape[0],
        face_rows.ctypes.data,
        points.ctypes.data,
        count,
        first.ctypes.data,
        second.ctypes.data,
        ctypes.byref(handled),
    )
    _raise_for_call(status, "implicit_face_proposals")
    if not handled.value:
        return None
    return first, second


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
    instruction_array, index_array, parameter_array, lower_array, upper_array = (
        _interval_program_arrays(instructions, indices, parameters, lower, upper)
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


def interval_program_branches(
    instructions: object,
    indices: object,
    parameters: object,
    lower: object,
    upper: object,
    events: object,
    box_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute an interval program and record its per-box branch evidence.

    As `interval_program_execute`, plus ``events`` of shape ``(rows, 6)``
    (kind, test, size, a, a_index, boxes; see ``phx_mc_interval_program_branches``).
    Returns the workspaces and the ``(box_count,)`` smooth and continuous flags.
    """
    instruction_array, index_array, parameter_array, lower_array, upper_array = (
        _interval_program_arrays(instructions, indices, parameters, lower, upper)
    )
    event_array = _native_integer_array(events, "events", np.dtype(np.int64))
    if event_array.ndim != 2 or event_array.shape[1] != 6:
        raise ValueError("events must have shape (rows, 6).")
    if np.shares_memory(event_array, lower_array) or np.shares_memory(
        event_array, upper_array
    ):
        raise ValueError("interval program inputs must not overlap output workspaces.")
    boxes = int(box_count)
    if boxes < 0:
        raise ValueError("box_count must be nonnegative.")
    smooth = np.zeros((boxes,), dtype=np.uint8)
    continuous = np.zeros((boxes,), dtype=np.uint8)
    library = load_meshcore()
    status = library["phx_mc_interval_program_branches"](
        instruction_array.ctypes.data,
        instruction_array.shape[0],
        index_array.ctypes.data,
        index_array.shape[0],
        parameter_array.ctypes.data,
        parameter_array.shape[0],
        lower_array.ctypes.data,
        upper_array.ctypes.data,
        lower_array.size,
        event_array.ctypes.data,
        event_array.shape[0],
        boxes,
        smooth.ctypes.data,
        continuous.ctypes.data,
    )
    _raise_for_call(status, "interval_program_branches")
    return lower_array, upper_array, smooth.astype(np.bool_), continuous.astype(np.bool_)


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


class OversetAssembly(NamedTuple):
    """Per-vertex and per-cell records of ``overset_assemble`` (global rows)."""

    vertex_status: np.ndarray
    fringe_layer: np.ndarray
    wall_contact: np.ndarray
    protected_conflict: np.ndarray
    cell_status: np.ndarray
    ambiguous: np.ndarray
    donor_status: np.ndarray
    donor_cell: np.ndarray
    donor_reference: np.ndarray
    donor_residual: np.ndarray
    donor_measure: np.ndarray
    candidate_parts: np.ndarray
    # Seconds of hole map, cut cells, fringe, donor index, coverage, donor
    # construction, certification and total; then hole vertices, receptors,
    # orphans, rejected candidates, located candidates, ambiguous cells,
    # fringe donors and shared-face donors.
    stats: np.ndarray


def overset_assemble(
    vertex_offsets: object,
    vertices: object,
    cell_offsets: object,
    cells: object,
    cell_ids: object,
    cell_excluded: object,
    face_offsets: object,
    faces: object,
    face_roles: object,
    priorities: object,
    name_ranks: object,
    /,
    *,
    fringe_layers: int,
    donor_priority: int,
    fringe_donors: bool,
    prove: bool,
) -> OversetAssembly:
    """Assemble overset blanking and certified donors of 3D tet/hex parts.

    Rows follow ``phx_mc_overset_assemble``: world-frame vertices, cells as
    rows of eight global vertex indices (tetrahedra padded with -1), exterior
    faces as rows of four (triangles padded with -1) with roles 0 other,
    1 wall, 2 receptor boundary.
    """

    vertex_bounds = _native_integer_array(
        vertex_offsets, "vertex_offsets", np.dtype(np.int64)
    )
    points = _points(vertices, "vertices", 3)
    cell_bounds = _native_integer_array(cell_offsets, "cell_offsets", np.dtype(np.int64))
    cell_rows = _native_integer_array(cells, "cells", np.dtype(np.int32))
    identifiers = _native_integer_array(cell_ids, "cell_ids", np.dtype(np.int64))
    excluded = np.ascontiguousarray(cell_excluded, dtype=np.uint8)
    face_bounds = _native_integer_array(face_offsets, "face_offsets", np.dtype(np.int64))
    face_rows = _native_integer_array(faces, "faces", np.dtype(np.int32))
    roles = np.ascontiguousarray(face_roles, dtype=np.uint8)
    part_priorities = _native_integer_array(priorities, "priorities", np.dtype(np.int32))
    ranks = _native_integer_array(name_ranks, "name_ranks", np.dtype(np.int32))
    part_count = part_priorities.shape[0]
    vertex_count, cell_count, face_count = (
        points.shape[0],
        cell_rows.shape[0],
        face_rows.shape[0],
    )
    if (
        part_priorities.ndim != 1
        or ranks.shape != part_priorities.shape
        or any(
            bounds.shape != (part_count + 1,)
            for bounds in (vertex_bounds, cell_bounds, face_bounds)
        )
        or cell_rows.shape != (cell_count, 8)
        or face_rows.shape != (face_count, 4)
        or identifiers.shape != (cell_count,)
        or excluded.shape != (cell_count,)
        or roles.shape != (face_count,)
        or int(vertex_bounds[-1]) != vertex_count
        or int(cell_bounds[-1]) != cell_count
        or int(face_bounds[-1]) != face_count
    ):
        raise ValueError("Overset part arrays have inconsistent shapes or offsets.")
    for name, value in (
        ("fringe_layers", fringe_layers),
        ("donor_priority", donor_priority),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
    if not isinstance(prove, bool) or not isinstance(fringe_donors, bool):
        raise TypeError("prove and fringe_donors must be bools.")
    result = OversetAssembly(
        np.empty((vertex_count,), dtype=np.int8),
        np.empty((vertex_count,), dtype=np.int32),
        np.empty((vertex_count,), dtype=np.uint8),
        np.empty((vertex_count,), dtype=np.uint8),
        np.empty((cell_count,), dtype=np.int8),
        np.empty((cell_count,), dtype=np.uint8),
        np.empty((vertex_count,), dtype=np.int32),
        np.empty((vertex_count,), dtype=np.int64),
        np.empty((vertex_count, 3), dtype=np.float64),
        np.empty((vertex_count,), dtype=np.float64),
        np.empty((vertex_count,), dtype=np.float64),
        np.empty((vertex_count,), dtype=np.int32),
        np.empty((16,), dtype=np.float64),
    )
    parts = _OversetParts(
        part_count=part_count,
        vertex_offsets=vertex_bounds.ctypes.data,
        vertices=points.ctypes.data,
        cell_offsets=cell_bounds.ctypes.data,
        cells=cell_rows.ctypes.data,
        cell_ids=identifiers.ctypes.data,
        cell_excluded=excluded.ctypes.data,
        face_offsets=face_bounds.ctypes.data,
        faces=face_rows.ctypes.data,
        face_roles=roles.ctypes.data,
        priorities=part_priorities.ctypes.data,
        name_ranks=ranks.ctypes.data,
        fringe_layers=_int32_argument(fringe_layers, "fringe_layers"),
        donor_priority=_int32_argument(donor_priority, "donor_priority"),
        fringe_donors=1 if fringe_donors else 0,
        prove=1 if prove else 0,
    )
    outputs = _OversetResult(*(array.ctypes.data for array in result))
    status = load_meshcore()["phx_mc_overset_assemble"](
        ctypes.byref(parts), ctypes.byref(outputs)
    )
    _raise_for_call(status, "overset_assemble")
    return result


class PrismLayers(NamedTuple):
    """Levels, prisms and evidence of ``prism_layers``.

    ``points`` holds level ``k`` of wall vertex ``v`` at row ``k * n + v``
    (level 0 is the wall bitwise); ``prisms`` holds layer ``k`` of triangle
    ``t`` at row ``k * m + t`` as bottom ``(a, b, c)`` then the tops over them.
    ``status`` is OK, or UNRESOLVED when a failure remained at the minimum
    scale (the arrays then hold the last construction).
    """

    status: MeshcoreStatus
    points: np.ndarray
    prisms: np.ndarray
    displacement: np.ndarray
    scale: np.ndarray
    vertex_kind: np.ndarray
    # Seconds of topology, jets, directions, relaxation, construction,
    # certification and total; then ridge vertices, corner vertices, relaxed
    # vertices, thinned vertices, thinned ridge/corner vertices, minimum
    # scale, repair rounds, invalid prisms, top-level contacts (-1 without
    # prove), certified, minimum float corner ratio, feature edges and
    # relaxation updates and the failures detected over every repair round.
    stats: np.ndarray


def prism_layers(
    vertices: object,
    triangles: object,
    heights: object,
    /,
    *,
    feature_angle_degrees: float,
    minimum_scale: float,
    prove: bool,
    source_normals: object | None = None,
    source_shape_operators: object | None = None,
) -> PrismLayers:
    """Grow prism layers from a closed oriented triangle wall.

    Rows follow ``phx_mc_prism_layers``: triangles oriented so their normals
    point into the layer domain, ``heights`` the increasing cumulative level
    heights. ``source_normals`` (m, 3, 3) and ``source_shape_operators``
    (m, 3, 6), given together, are the per-corner source-surface frames
    (unit normal into the layer domain, or zero where the chart is singular;
    shape-operator entries xx, xy, xz, yy, yz, zz) from which the levels are
    measured; without them the kernel fits the surface through the vertices.
    """

    points = _points(vertices, "vertices", 3)
    rows = _native_integer_array(triangles, "triangles", np.dtype(np.int32))
    levels = np.ascontiguousarray(heights, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 3 or levels.ndim != 1 or levels.size < 1:
        raise ValueError("prism_layers takes (m, 3) triangles and a heights vector.")
    if not isinstance(prove, bool):
        raise TypeError("prove must be a bool.")
    for name, value in (
        ("feature_angle_degrees", feature_angle_degrees),
        ("minimum_scale", minimum_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a real number.")
    vertex_count, triangle_count = points.shape[0], rows.shape[0]
    if (source_normals is None) != (source_shape_operators is None):
        raise ValueError(
            "source_normals and source_shape_operators are given together or not at all."
        )
    frames = None
    if source_normals is not None:
        frames = (
            _real_array(source_normals, "source_normals"),
            _real_array(source_shape_operators, "source_shape_operators"),
        )
        if frames[0].shape != (triangle_count, 3, 3) or frames[1].shape != (
            triangle_count,
            3,
            6,
        ):
            raise ValueError(
                "source frames take (m, 3, 3) normals and (m, 3, 6) shape operators."
            )
    layer_count = levels.shape[0]
    arrays = (
        np.empty(((layer_count + 1) * vertex_count, 3), dtype=np.float64),
        np.empty((layer_count * triangle_count, 6), dtype=np.int32),
        np.empty((vertex_count, 3), dtype=np.float64),
        np.empty((vertex_count,), dtype=np.float64),
        np.empty((vertex_count,), dtype=np.uint8),
        np.empty((21,), dtype=np.float64),
    )
    request = _PrismLayersRequest(
        vertex_count=vertex_count,
        vertices=points.ctypes.data,
        triangle_count=triangle_count,
        triangles=rows.ctypes.data,
        layer_count=_int32_argument(layer_count, "layer_count"),
        heights=levels.ctypes.data,
        feature_angle_degrees=float(feature_angle_degrees),
        minimum_scale=float(minimum_scale),
        prove=1 if prove else 0,
        source_normals=None if frames is None else frames[0].ctypes.data,
        source_shape_operators=None if frames is None else frames[1].ctypes.data,
    )
    outputs = _PrismLayersResult(*(array.ctypes.data for array in arrays))
    status = load_meshcore()["phx_mc_prism_layers"](
        ctypes.byref(request), ctypes.byref(outputs)
    )
    if status != MeshcoreStatus.UNRESOLVED:
        _raise_for_call(
            status,
            "prism_layers",
            contracts={
                MeshcoreStatus.INVALID_INPUT: "the wall must be a closed, "
                "consistently oriented triangle manifold using every vertex.",
                MeshcoreStatus.DEGENERATE_INPUT: "a wall triangle has zero area.",
            },
        )
    return PrismLayers(MeshcoreStatus(status), *arrays)


class ConformingVoronoi(NamedTuple):
    """Cells of ``conforming_voronoi``: Voronoi cells of conforming seeds.

    Face ``f`` is the counterclockwise loop
    ``face_vertices[face_offsets[f]:face_offsets[f + 1]]`` whose normal points
    out of cell ``face_cells[f, 0]`` into ``face_cells[f, 1]`` (``-1`` on the
    exterior); ``face_facets[f]`` is the source facet of a boundary or
    interface face and ``-1`` otherwise. ``vertex_triangles[v]`` is the
    nearest input triangle of a vertex on a boundary or interface face, ``-1``
    otherwise. ``seeds`` holds every seed (interior seeds first) and
    ``seed_regions`` their regions (``-1``: exterior ghost); cell ``c`` is the
    Voronoi cell of ``seeds[cell_seeds[c]]`` in the diagram of all seeds.
    Vertex ``v`` is exact: ``x_k = N_k / D`` with ``N_0, N_1, N_2, D`` the
    expansions ``exact_values[exact_offsets[4 v + k]:exact_offsets[4 v + k + 1]]``
    (sums of their components); ``vertices`` holds the nearest binary64
    values. ``status`` is OK, or UNRESOLVED when a cell leaks after the
    repairs or certification failed (the arrays then hold the last
    construction). ``stats`` follows ``phx_mc_conforming_voronoi``.
    """

    status: MeshcoreStatus
    vertices: np.ndarray
    vertex_triangles: np.ndarray
    face_offsets: np.ndarray
    face_vertices: np.ndarray
    face_cells: np.ndarray
    face_facets: np.ndarray
    cell_seeds: np.ndarray
    cell_regions: np.ndarray
    cell_volumes: np.ndarray
    cell_centroids: np.ndarray
    seeds: np.ndarray
    seed_regions: np.ndarray
    # Seconds of seeding, Delaunay, cells, publication, certification and
    # total; seeds, interior seeds, ghosts, lattice seeds, cells, boundary
    # cells, boundary faces, faces, repair rounds, certified (1/0, -1 without
    # prove), open cell edges and local Delaunay violations (-1 without
    # prove), signed relative volume error, largest boundary vertex distance /
    # size, certified boundary distance bound / size (-1 without prove),
    # feature edges, size, surface spacing, exact boundary (1/0) and a bound
    # of the absolute relative volume error.
    stats: np.ndarray
    exact_offsets: np.ndarray
    exact_values: np.ndarray


def conforming_voronoi(
    vertices: object,
    triangles: object,
    triangle_regions: object,
    triangle_facets: object,
    /,
    *,
    size: float,
    feature_angle_degrees: float,
    prove: bool,
) -> ConformingVoronoi:
    """Conforming Voronoi cells of a triangulated solid for the cell size ``size``.

    Rows follow ``phx_mc_conforming_voronoi``: triangle ``t`` separates region
    ``triangle_regions[t, 0]`` in front (along its normal) from
    ``triangle_regions[t, 1]`` behind, ``-1`` the exterior.
    """

    points = _points(vertices, "vertices", 3)
    rows = _native_integer_array(triangles, "triangles", np.dtype(np.int32))
    sides = _native_integer_array(
        triangle_regions, "triangle_regions", np.dtype(np.int32)
    )
    facets = _native_integer_array(triangle_facets, "triangle_facets", np.dtype(np.int32))
    if (
        rows.ndim != 2
        or rows.shape[1] != 3
        or sides.shape != (rows.shape[0], 2)
        or facets.shape != (rows.shape[0],)
    ):
        raise ValueError(
            "conforming_voronoi takes (m, 3) triangles, (m, 2) regions and (m,) facets."
        )
    if not isinstance(prove, bool):
        raise TypeError("prove must be a bool.")
    for name, value in (("size", size), ("feature_angle_degrees", feature_angle_degrees)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a real number.")
    handle = ctypes.c_void_p()
    request = _ConformingVoronoiRequest(
        vertex_count=points.shape[0],
        vertices=points.ctypes.data,
        triangle_count=rows.shape[0],
        triangles=rows.ctypes.data,
        triangle_regions=sides.ctypes.data,
        triangle_facets=facets.ctypes.data,
        size=float(size),
        feature_angle_degrees=float(feature_angle_degrees),
        prove=1 if prove else 0,
        result=ctypes.pointer(handle),
    )
    library = load_meshcore()
    status = library["phx_mc_conforming_voronoi"](ctypes.byref(request))
    if not handle.value:
        _raise_for_call(
            status,
            "conforming_voronoi",
            contracts={
                MeshcoreStatus.DEGENERATE_INPUT: "a zero-area triangle.",
                MeshcoreStatus.UNRESOLVED: "no seed lies in the solid.",
            },
        )
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "missing Voronoi result")
    try:
        sizes = np.empty((6,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_conforming_voronoi_sizes"](handle, sizes.ctypes.data),
            "conforming_voronoi_sizes",
        )
        vertex_count, face_count, entry_count, cell_count, seed_count, exact_count = (
            int(value) for value in sizes
        )
        arrays = (
            np.empty((vertex_count, 3), dtype=np.float64),
            np.empty((vertex_count,), dtype=np.int32),
            np.empty((face_count + 1,), dtype=np.int64),
            np.empty((entry_count,), dtype=np.int32),
            np.empty((face_count, 2), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
            np.empty((cell_count,), dtype=np.int32),
            np.empty((cell_count,), dtype=np.int32),
            np.empty((cell_count,), dtype=np.float64),
            np.empty((cell_count, 3), dtype=np.float64),
            np.empty((seed_count, 3), dtype=np.float64),
            np.empty((seed_count,), dtype=np.int32),
            np.empty((26,), dtype=np.float64),
            np.empty((4 * vertex_count + 1,), dtype=np.int64),
            np.empty((exact_count,), dtype=np.float64),
        )
        outputs = _ConformingVoronoiArrays(*(array.ctypes.data for array in arrays))
        _raise_for_call(
            library["phx_mc_conforming_voronoi_export"](handle, ctypes.byref(outputs)),
            "conforming_voronoi_export",
        )
    finally:
        library["phx_mc_conforming_voronoi_free"](handle)
    return ConformingVoronoi(MeshcoreStatus(status), *arrays)


class HexDominant(NamedTuple):
    """Face-based mixed mesh of ``hex_dominant``.

    Face ``f`` is the loop ``face_vertices[face_offsets[f]:face_offsets[f + 1]]``
    oriented out of ``face_owner[f]``; ``face_neighbor[f]`` is ``-1`` on the
    boundary, where ``face_facets[f]`` is the facet of the input triangle
    nearest the face (``-1`` inside). ``cell_kinds``: 0 hexahedron (its eight
    vertices ``cell_vertices[cell_vertex_offsets[c]:cell_vertex_offsets[c + 1]]``
    in the reference order), 1 polyhedron (no vertex list). ``cell_roles``: 0
    core, 1 projection layer, 2 boundary layer. ``vertex_triangles`` is the
    nearest input triangle of a wall vertex (``-1`` elsewhere);
    ``vertex_kinds``: 0 interior, 1 patch, 2 feature curve, 3 corner, 4 layer.
    ``status`` is OK, or UNRESOLVED when a cell failed certification (the
    arrays then hold the construction).
    """

    status: MeshcoreStatus
    points: np.ndarray
    face_offsets: np.ndarray
    face_vertices: np.ndarray
    face_owner: np.ndarray
    face_neighbor: np.ndarray
    face_facets: np.ndarray
    cell_kinds: np.ndarray
    cell_roles: np.ndarray
    cell_vertex_offsets: np.ndarray
    cell_vertices: np.ndarray
    vertex_triangles: np.ndarray
    vertex_kinds: np.ndarray
    # Seconds of the whole call, surface, octree and core, faces, projection,
    # smoothing and layers, certification; cells, hexahedra, polyhedra,
    # certified hexahedra, layer cells, staircase faces, feature edges,
    # patches, largest wall vertex distance, largest sampled boundary
    # distance, manifold removals, partition flips, curve vertices, corner
    # vertices, undone smoothing moves, certified (1/0, -1 without prove),
    # uncertified cells, hexahedra published as polyhedra, octree levels.
    stats: np.ndarray


def hex_dominant(
    vertices: object,
    triangles: object,
    triangle_facets: object,
    /,
    *,
    core_size: float,
    boundary_size: float,
    feature_angle_degrees: float,
    layer_thicknesses: object,
    prove: bool,
) -> HexDominant:
    """Hex-dominant mesh of a closed outward-oriented triangle surface.

    Rows follow ``phx_mc_hex_dominant``: ``core_size`` is ``boundary_size``
    times a power of two; ``layer_thicknesses`` (possibly empty) are the
    hexahedral layer thicknesses from the wall outward.
    """

    points = _points(vertices, "vertices", 3)
    rows = _native_integer_array(triangles, "triangles", np.dtype(np.int32))
    facets = _native_integer_array(triangle_facets, "triangle_facets", np.dtype(np.int32))
    layers = np.ascontiguousarray(layer_thicknesses, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[1] != 3
        or facets.shape != (rows.shape[0],)
        or layers.ndim != 1
    ):
        raise ValueError(
            "hex_dominant takes (m, 3) triangles, (m,) facets and a thickness vector."
        )
    if not isinstance(prove, bool):
        raise TypeError("prove must be a bool.")
    for name, value in (
        ("core_size", core_size),
        ("boundary_size", boundary_size),
        ("feature_angle_degrees", feature_angle_degrees),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a real number.")
    handle = ctypes.c_void_p()
    request = _HexDominantRequest(
        vertex_count=points.shape[0],
        vertices=points.ctypes.data,
        triangle_count=rows.shape[0],
        triangles=rows.ctypes.data,
        triangle_facets=facets.ctypes.data,
        core_size=float(core_size),
        boundary_size=float(boundary_size),
        feature_angle_degrees=float(feature_angle_degrees),
        layer_count=_int32_argument(layers.shape[0], "layer_count"),
        layer_thicknesses=layers.ctypes.data,
        prove=1 if prove else 0,
        result=ctypes.pointer(handle),
    )
    library = load_meshcore()
    status = library["phx_mc_hex_dominant"](ctypes.byref(request))
    if not handle.value:
        _raise_for_call(
            status,
            "hex_dominant",
            contracts={
                MeshcoreStatus.INVALID_INPUT: "the surface must be a closed, "
                "consistently outward-oriented triangle manifold using every vertex.",
                MeshcoreStatus.DEGENERATE_INPUT: "a surface triangle has zero area.",
            },
        )
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "missing hex-dominant result")
    try:
        sizes = np.empty((5,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_hex_dominant_sizes"](handle, sizes.ctypes.data),
            "hex_dominant_sizes",
        )
        point_count, face_count, entry_count, cell_count, cell_entry_count = (
            int(value) for value in sizes
        )
        arrays = (
            np.empty((point_count, 3), dtype=np.float64),
            np.empty((face_count + 1,), dtype=np.int64),
            np.empty((entry_count,), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
            np.empty((face_count,), dtype=np.int32),
            np.empty((cell_count,), dtype=np.uint8),
            np.empty((cell_count,), dtype=np.uint8),
            np.empty((cell_count + 1,), dtype=np.int64),
            np.empty((cell_entry_count,), dtype=np.int32),
            np.empty((point_count,), dtype=np.int32),
            np.empty((point_count,), dtype=np.uint8),
            np.empty((28,), dtype=np.float64),
        )
        outputs = _HexDominantArrays(*(array.ctypes.data for array in arrays))
        _raise_for_call(
            library["phx_mc_hex_dominant_export"](handle, ctypes.byref(outputs)),
            "hex_dominant_export",
        )
    finally:
        library["phx_mc_hex_dominant_free"](handle)
    return HexDominant(MeshcoreStatus(status), *arrays)


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


def tetrahedron_intersection_quadratic_moments(
    first: object, second: object, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symbolic tetrahedron intersections through the full second moment."""
    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
    count = first_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    second_moment = np.zeros((count, 3, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_quadratic_moments"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        second_moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_quadratic_moments")
    return volume, moment, second_moment, status


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


def _mesh_shape(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> tuple[int, int, int, int]:
    shape = (ctypes.c_int64 * 4)()
    _raise_for_call(library["phx_mc_mesh_shape"](handle, shape), "mesh_shape")
    dimension, point_count, cell_count, input_count = shape
    index_limit = np.iinfo(np.int32).max
    if (
        dimension not in (2, 3)
        or not 0 <= point_count < index_limit
        or cell_count < 0
        or not 0 <= input_count < index_limit
    ):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "mesh handle reported invalid output sizes"
        )
    return dimension, point_count, cell_count, input_count


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
    # The native handle is released even when a copy fails. Every array is
    # written in full by the copy, so it is allocated uninitialized.
    try:
        dimension, point_count, cell_count, input_count = _mesh_shape(library, handle)
        points = np.empty((point_count, dimension), dtype=np.float64)
        cells = np.empty((cell_count, dimension + 1), dtype=np.int32)
        vertex_map = np.empty((input_count,), dtype=np.int32)
        cell_segments = np.empty((cell_count, 3), dtype=np.int32)
        constraint_sources = np.empty((point_count,), dtype=np.int32)
        constraint_parents = np.empty((point_count, 2), dtype=np.int32)
        constraint_ratios = np.empty((point_count,), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_mesh_copy_all"](
                handle,
                point_count,
                cell_count,
                input_count,
                points.ctypes.data,
                cells.ctypes.data,
                vertex_map.ctypes.data,
                cell_segments.ctypes.data,
                constraint_sources.ctypes.data,
                constraint_parents.ctypes.data,
                constraint_ratios.ctypes.data,
            ),
            "mesh_copy_all",
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
    trailing: tuple[int, ...] = (),
    /,
) -> tuple[np.ndarray, np.ndarray]:
    library = load_meshcore()
    point_array = _points(points, "points", width)
    count = point_array.shape[0]
    if name == "phx_mc_delaunay_2d" and count <= 4096:
        # A planar triangulation has at most 2n - 5 finite triangles.
        # Bound the speculative buffer to 96 KiB; larger calls allocate
        # only after native construction knows the exact count.
        capacity = max(0, 2 * count - 5)
        cells = np.empty((capacity, 3), dtype=np.int32)
        vertex_map = np.empty((count,), dtype=np.int32)
        written = ctypes.c_int64()
        status = library["phx_mc_delaunay_2d_into"](
            count,
            point_array.ctypes.data,
            capacity,
            cells.ctypes.data,
            vertex_map.ctypes.data,
            ctypes.addressof(written),
        )
        if status != MeshcoreStatus.OK:
            _raise_for_call(status, "delaunay_2d")
        return cells[: written.value], vertex_map
    handle = ctypes.c_void_p()
    if weights is None:
        status = library[name](
            count, point_array.ctypes.data, ctypes.addressof(handle), *trailing
        )
    else:
        weight_array = _real_array(weights, "weights")
        if weight_array.shape != (count,):
            raise ValueError("weights must have shape (n,).")
        status = library[name](
            count,
            point_array.ctypes.data,
            weight_array.ctypes.data,
            ctypes.addressof(handle),
            *trailing,
        )
    operation = name.removeprefix("phx_mc_")
    if status != MeshcoreStatus.OK:
        _raise_for_call(status, operation)
    if not handle.value:
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, f"{operation}: missing mesh")
    try:
        dimension, _, cell_count, input_count = _mesh_shape(library, handle)
        cells = np.empty((cell_count, dimension + 1), dtype=np.int32)
        vertex_map = np.empty((input_count,), dtype=np.int32)
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
    finally:
        library["phx_mc_mesh_free"](handle)
    return cells, vertex_map


def delaunay_2d(points: object, /) -> tuple[np.ndarray, np.ndarray]:
    """Exact Delaunay triangulation of ``(n, 2)`` points.

    Returns ``(triangles (T, 3) int32, vertex_map (n,) int32)``; triangles are
    counterclockwise, canonically ordered, and cocircular ties are resolved by
    index-ordered symbolic perturbation. ``vertex_map`` sends duplicates to
    their smallest identical index.
    """

    return _triangulate("phx_mc_delaunay_2d", points, None, 2)


def delaunay_3d(points: object, /, *, parallelism: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact Delaunay tetrahedralization of ``(n, 3)`` points.

    Returns positively oriented, canonically ordered tetrahedra and the input
    vertex map. ``parallelism`` workers insert in rounds of conflict-free
    cavities; the result does not depend on it.
    """

    return _triangulate("phx_mc_delaunay_3d", points, None, 3, (parallelism,))


def regular_2d(points: object, weights: object, /) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) triangulation; redundant points map to -1."""

    return _triangulate("phx_mc_regular_2d", points, weights, 2)


def regular_3d(
    points: object, weights: object, /, *, parallelism: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) tetrahedralization; redundant points map to -1.

    ``parallelism`` workers insert in rounds; the result does not depend on it.
    """

    return _triangulate("phx_mc_regular_3d", points, weights, 3, (parallelism,))


class VerdictStatus(IntEnum):
    """Outcome of an exact native certificate."""

    CERTIFIED = 0
    VIOLATED = 1
    UNRESOLVED = 2


class TetrahedraVerdict(NamedTuple):
    """Exact global-embedding certificate of affine tetrahedra.

    Violations: ``nonpositive_cells``, ``nonmanifold_faces`` (a face in more
    than two cells, or two with equal induced orientation), ``boundary_contacts``
    (pairs of boundary triangles meeting beyond a shared vertex or edge; the
    cells owning the smallest pair) and ``misnested_shells`` (a boundary shell
    with winding number other than 0 in front of it). Outside the theorem
    (unresolved): ``outside_domain_points``, ``nonmanifold_boundary_edges``
    (edge uses) and ``unresolved_shells``. Each ``first_*`` names the smallest
    offending row, or -1.
    """

    status: VerdictStatus
    boundary_faces: int
    boundary_shells: int
    outside_domain_points: int
    first_outside_domain_point: int
    nonpositive_cells: int
    first_nonpositive_cell: int
    nonmanifold_faces: int
    first_nonmanifold_cell: int
    nonmanifold_boundary_edges: int
    first_nonmanifold_boundary_cell: int
    boundary_contacts: int
    first_contact_cell: int
    second_contact_cell: int
    misnested_shells: int
    first_misnested_shell_cell: int
    unresolved_shells: int
    first_unresolved_shell_cell: int


def _tetrahedral_mesh(points: object, cells: object, /) -> tuple[np.ndarray, np.ndarray]:
    point_array = np.ascontiguousarray(_points(points, "points", 3), dtype=np.float64)
    cell_array = _native_integer_array(cells, "cells", np.dtype(np.int32))
    if cell_array.ndim != 2 or cell_array.shape[1] != 4:
        raise ValueError("cells must have shape (m, 4).")
    if point_array.shape[0] > np.iinfo(np.int32).max:
        raise OverflowError("points exceed native int32 vertex indices.")
    return point_array, cell_array


def verify_tetrahedra(points: object, cells: object, /) -> TetrahedraVerdict:
    """Certify that positive ``(m, 4)`` cells tile the solid their boundary bounds.

    Exact on the stored binary64 coordinates (theorem in the native
    ``verify_volume.cpp``): positive cells, faces paired with opposite
    orientations, an embedded edge-manifold boundary whose shells have winding
    number 0 in front and 1 behind.
    """
    point_array, cell_array = _tetrahedral_mesh(points, cells)
    verdict = _TetrahedraVerdict()
    status = load_meshcore()["phx_mc_verify_tetrahedra"](
        point_array.shape[0],
        point_array.ctypes.data,
        cell_array.shape[0],
        cell_array.ctypes.data,
        ctypes.addressof(verdict),
    )
    _raise_for_call(status, "verify_tetrahedra")
    return TetrahedraVerdict(
        status=VerdictStatus(verdict.status),
        boundary_faces=verdict.boundary_faces,
        boundary_shells=verdict.boundary_shells,
        outside_domain_points=verdict.outside_domain_points,
        first_outside_domain_point=verdict.first_outside_domain_point,
        nonpositive_cells=verdict.nonpositive_cells,
        first_nonpositive_cell=verdict.first_nonpositive_cell,
        nonmanifold_faces=verdict.nonmanifold_faces,
        first_nonmanifold_cell=verdict.first_nonmanifold_cell,
        nonmanifold_boundary_edges=verdict.nonmanifold_boundary_edges,
        first_nonmanifold_boundary_cell=verdict.first_nonmanifold_boundary_cell,
        boundary_contacts=verdict.boundary_contacts,
        first_contact_cell=verdict.first_contact_cell,
        second_contact_cell=verdict.second_contact_cell,
        misnested_shells=verdict.misnested_shells,
        first_misnested_shell_cell=verdict.first_misnested_shell_cell,
        unresolved_shells=verdict.unresolved_shells,
        first_unresolved_shell_cell=verdict.first_unresolved_shell_cell,
    )


class VolumeVerdict(NamedTuple):
    """Exact region coverage of a declared oriented surface by labeled tetrahedra.

    ``certified`` holds when every failure count is zero; each ``first_*``
    names the smallest offending cell, point or surface triangle, or is -1.
    """

    certified: bool
    outside_domain_points: int
    first_outside_domain_point: int
    nonpositive_cells: int
    first_nonpositive_cell: int
    nonmanifold_faces: int
    first_nonmanifold_cell: int
    boundary_faces: int
    unmatched_boundary_faces: int
    first_unmatched_cell: int
    unmatched_surface_triangles: int
    first_unmatched_surface_triangle: int
    region_mismatches: int
    first_region_mismatch_cell: int


def verify_volume(
    points: object,
    cells: object,
    surface: object,
    /,
    *,
    cell_regions: object | None = None,
    surface_regions: object | None = None,
) -> VolumeVerdict:
    """Certify labeled ``(m, 4)`` cells against declared ``(f, 3)`` surface triangles.

    Exact on the stored binary64 coordinates: every cell positively oriented,
    every face shared by at most two cells with opposite orientations, every
    boundary face and every face between regions a declared triangle with the
    cells' regions on its sides, and every declared triangle a mesh face. A
    triangle's normal points from ``surface_regions[:, 0]`` into
    ``surface_regions[:, 1]`` (-1: exterior). Without labels every cell is
    region 0 and every triangle its outward boundary ``(0, -1)``. With the
    mesh globally embedded (``verify_tetrahedra``), each region's cells tile
    exactly the solid its declared boundary bounds.
    """
    point_array, cell_array = _tetrahedral_mesh(points, cells)
    surface_array = _native_integer_array(surface, "surface", np.dtype(np.int32))
    if surface_array.ndim != 2 or surface_array.shape[1] != 3:
        raise ValueError("surface must have shape (f, 3).")
    regions = (
        None
        if cell_regions is None
        else _native_integer_array(cell_regions, "cell_regions", np.dtype(np.int32))
    )
    if regions is not None and regions.shape != (cell_array.shape[0],):
        raise ValueError("cell_regions must have shape (m,).")
    sides = (
        None
        if surface_regions is None
        else _native_integer_array(surface_regions, "surface_regions", np.dtype(np.int32))
    )
    if sides is not None and sides.shape != (surface_array.shape[0], 2):
        raise ValueError("surface_regions must have shape (f, 2).")
    verdict = _VolumeVerdict()
    status = load_meshcore()["phx_mc_verify_volume"](
        point_array.shape[0],
        point_array.ctypes.data,
        cell_array.shape[0],
        cell_array.ctypes.data,
        None if regions is None else regions.ctypes.data,
        surface_array.shape[0],
        surface_array.ctypes.data,
        None if sides is None else sides.ctypes.data,
        ctypes.addressof(verdict),
    )
    _raise_for_call(status, "verify_volume")
    return VolumeVerdict(
        certified=verdict.certified == 1,
        outside_domain_points=verdict.outside_domain_points,
        first_outside_domain_point=verdict.first_outside_domain_point,
        nonpositive_cells=verdict.nonpositive_cells,
        first_nonpositive_cell=verdict.first_nonpositive_cell,
        nonmanifold_faces=verdict.nonmanifold_faces,
        first_nonmanifold_cell=verdict.first_nonmanifold_cell,
        boundary_faces=verdict.boundary_faces,
        unmatched_boundary_faces=verdict.unmatched_boundary_faces,
        first_unmatched_cell=verdict.first_unmatched_cell,
        unmatched_surface_triangles=verdict.unmatched_surface_triangles,
        first_unmatched_surface_triangle=verdict.first_unmatched_surface_triangle,
        region_mismatches=verdict.region_mismatches,
        first_region_mismatch_cell=verdict.first_region_mismatch_cell,
    )


class TetMeshEntities(NamedTuple):
    """Sorted unique edges and faces of a tetrahedral mesh with their routes.

    ``edges`` ``(e, 2)`` and ``faces`` ``(f, 3)`` are the distinct increasing
    vertex rows in lexicographic order; ``cell_edges`` ``(m, 6)`` and
    ``cell_faces`` ``(m, 4)`` route each cell's local edges and faces in
    combination order; ``face_edges`` ``(f, 3)`` routes each face's edges
    opposite its vertices in turn. ``incidence_orders`` sort each degree's
    flat face routes (simplex-major; a cell's faces opposite its stably sorted
    vertices) stably by face row. All int32.
    """

    edges: np.ndarray
    faces: np.ndarray
    cell_edges: np.ndarray
    cell_faces: np.ndarray
    face_edges: np.ndarray
    incidence_orders: tuple[np.ndarray, np.ndarray, np.ndarray]


def tet_mesh_entities(cells: object, vertex_count: int, /) -> TetMeshEntities:
    """Sorted unique edges and faces of ``(m, 4)`` tetrahedra and their routes."""
    cell_array = _native_integer_array(cells, "cells", np.dtype(np.int32))
    if cell_array.ndim != 2 or cell_array.shape[1] != 4:
        raise ValueError("cells must have shape (m, 4).")
    count = cell_array.shape[0]
    edges = np.empty((6 * count, 2), dtype=np.int32)
    faces = np.empty((4 * count, 3), dtype=np.int32)
    face_edges = np.empty((4 * count, 3), dtype=np.int32)
    cell_edges = np.empty((count, 6), dtype=np.int32)
    cell_faces = np.empty((count, 4), dtype=np.int32)
    vertex_edge_order = np.empty((12 * count,), dtype=np.int32)
    edge_face_order = np.empty((12 * count,), dtype=np.int32)
    face_cell_order = np.empty((4 * count,), dtype=np.int32)
    edge_count = ctypes.c_int64()
    face_count = ctypes.c_int64()
    status = load_meshcore()["phx_mc_tet_mesh_entities"](
        int(vertex_count),
        count,
        cell_array.ctypes.data,
        edges.ctypes.data,
        ctypes.byref(edge_count),
        faces.ctypes.data,
        ctypes.byref(face_count),
        cell_edges.ctypes.data,
        cell_faces.ctypes.data,
        face_edges.ctypes.data,
        vertex_edge_order.ctypes.data,
        edge_face_order.ctypes.data,
        face_cell_order.ctypes.data,
    )
    _raise_for_call(status, "tet_mesh_entities")
    edge_total = edge_count.value
    face_total = face_count.value
    return TetMeshEntities(
        edges[:edge_total].copy(),
        faces[:face_total].copy(),
        cell_edges,
        cell_faces,
        face_edges[:face_total].copy(),
        (
            vertex_edge_order[: 2 * edge_total].copy(),
            edge_face_order[: 3 * face_total].copy(),
            face_cell_order,
        ),
    )


class PolyhedralMeshEntities(NamedTuple):
    """Oriented entities of standard polyhedral cells, first-appearance numbered.

    ``loop_faces`` and ``loop_signs`` route every cell-major face loop to its
    face and orientation; ``face_owner_loops`` and ``face_neighbor_loops``
    (``-1`` when unshared) are each face's first and second loop;
    ``face_vertex_offsets`` / ``face_vertex_values`` hold the canonical face
    loops; ``edges`` ``(e, 2)`` are increasing vertex pairs routed by
    ``face_edge_values`` per face side, ``edge_first_sides`` their first
    side; ``face_order`` and ``edge_order`` sort the faces and edges by
    increasing vertex rows. All int32.
    """

    loop_faces: np.ndarray
    loop_signs: np.ndarray
    face_owner_loops: np.ndarray
    face_neighbor_loops: np.ndarray
    face_vertex_offsets: np.ndarray
    face_vertex_values: np.ndarray
    edges: np.ndarray
    edge_first_sides: np.ndarray
    face_edge_values: np.ndarray
    face_order: np.ndarray
    edge_order: np.ndarray


# Face loops and summed loop sizes of tetrahedra, pyramids, prisms, hexahedra.
_POLYHEDRAL_KIND_LOOPS = np.asarray((4, 5, 5, 6), dtype=np.int64)
_POLYHEDRAL_KIND_SIDES = np.asarray((12, 16, 18, 24), dtype=np.int64)


def polyhedral_mesh_entities(
    cell_kinds: object, cells: object, vertex_count: int, /
) -> PolyhedralMeshEntities | None:
    """Entities of standard cells (kinds 0..3: tetrahedron, pyramid, prism,
    hexahedron; ``cells`` their packed vertex rows), or ``None`` for a cell
    set the generic construction rejects."""
    kinds = _native_integer_array(cell_kinds, "cell_kinds", np.dtype(np.int32))
    values = _native_integer_array(cells, "cells", np.dtype(np.int32))
    if kinds.ndim != 1 or values.ndim != 1:
        raise ValueError("cell_kinds and cells must be rank one.")
    if np.any((kinds < 0) | (kinds > 3)):
        raise ValueError("cell_kinds must lie in 0..3.")
    if values.size != int(np.sum(np.asarray((4, 5, 6, 8))[kinds])):
        raise ValueError("cells must pack one vertex row per cell kind.")
    count = kinds.size
    loops = int(np.sum(_POLYHEDRAL_KIND_LOOPS[kinds]))
    sides = int(np.sum(_POLYHEDRAL_KIND_SIDES[kinds]))
    loop_faces = np.empty((loops,), dtype=np.int32)
    loop_signs = np.empty((loops,), dtype=np.int32)
    owners = np.empty((loops,), dtype=np.int32)
    neighbors = np.empty((loops,), dtype=np.int32)
    face_offsets = np.empty((loops + 1,), dtype=np.int32)
    face_values = np.empty((sides,), dtype=np.int32)
    edges = np.empty((sides, 2), dtype=np.int32)
    edge_first_sides = np.empty((sides,), dtype=np.int32)
    face_edges = np.empty((sides,), dtype=np.int32)
    face_order = np.empty((loops,), dtype=np.int32)
    edge_order = np.empty((sides,), dtype=np.int32)
    face_count = ctypes.c_int64()
    edge_count = ctypes.c_int64()
    status = load_meshcore()["phx_mc_polyhedral_mesh_entities"](
        int(vertex_count),
        count,
        kinds.ctypes.data,
        values.ctypes.data,
        loop_faces.ctypes.data,
        loop_signs.ctypes.data,
        ctypes.byref(face_count),
        owners.ctypes.data,
        neighbors.ctypes.data,
        face_offsets.ctypes.data,
        face_values.ctypes.data,
        ctypes.byref(edge_count),
        edges.ctypes.data,
        edge_first_sides.ctypes.data,
        face_edges.ctypes.data,
        face_order.ctypes.data,
        edge_order.ctypes.data,
    )
    if status == MeshcoreStatus.INVALID_INPUT:
        return None
    _raise_for_call(status, "polyhedral_mesh_entities")
    faces = face_count.value
    edge_total = edge_count.value
    entries = int(face_offsets[faces])
    return PolyhedralMeshEntities(
        loop_faces,
        loop_signs,
        owners[:faces].copy(),
        neighbors[:faces].copy(),
        face_offsets[: faces + 1].copy(),
        face_values[:entries].copy(),
        edges[:edge_total].copy(),
        edge_first_sides[:edge_total].copy(),
        face_edges[:entries].copy(),
        face_order[:faces].copy(),
        edge_order[:edge_total].copy(),
    )


def incidence_order(
    lower: object, upper: object, signs: object, lower_count: int, upper_count: int, /
) -> np.ndarray:
    """Stable order of routes sorting their ``(lower, upper, sign)`` rows.

    ``lower`` and ``upper`` are dense entity ranks below ``lower_count`` and
    ``upper_count``; ``signs`` are ±1. Linear time.
    """
    lower_array = _native_integer_array(lower, "lower", np.dtype(np.int32))
    upper_array = _native_integer_array(upper, "upper", np.dtype(np.int32))
    sign_array = _native_integer_array(signs, "signs", np.dtype(np.int8))
    count = lower_array.size
    if (
        lower_array.ndim != 1
        or upper_array.shape != lower_array.shape
        or sign_array.shape != lower_array.shape
    ):
        raise ValueError("lower, upper and signs must be equal rank-one arrays.")
    order = np.empty((count,), dtype=np.int32)
    status = load_meshcore()["phx_mc_incidence_order"](
        int(lower_count),
        int(upper_count),
        count,
        lower_array.ctypes.data,
        upper_array.ctypes.data,
        sign_array.ctypes.data,
        order.ctypes.data,
    )
    _raise_for_call(status, "incidence_order")
    return order


def validity_coordinates(values: object, /) -> np.ndarray:
    """Which values ``tetrahedra_validity`` decides exactly (elementwise mask).

    A cell's coordinates and the floor must be zero or of magnitude in
    [2**-60, 2**60], so that the degree-eight validity margin is free of
    overflow and underflow.
    """
    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    return (magnitudes == 0.0) | ((magnitudes >= 2.0**-60) & (magnitudes <= 2.0**60))


def tetrahedra_validity(
    points: object, cells: object, floor: float, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact affine validity of ``(m, 4)`` tetrahedra under a determinant floor.

    Returns ``(valid, lower, upper)``: ``valid`` when the Jacobian determinant
    ``d = det[v1 - v0, v2 - v0, v3 - v0]`` and ``d^2 - floor^2 |v1 - v0|^2
    |v2 - v0|^2 |v3 - v0|^2`` are positive (exact signs), and a rigorous
    binary64 enclosure ``lower <= d <= upper``. Cell coordinates outside
    ``validity_coordinates`` raise ``ValueError``.
    """
    point_array, cell_array = _tetrahedral_mesh(points, cells)
    count = cell_array.shape[0]
    valid = np.zeros((count,), dtype=np.int8)
    lower = np.zeros((count,), dtype=np.float64)
    upper = np.zeros((count,), dtype=np.float64)
    status = load_meshcore()["phx_mc_tetrahedra_validity"](
        point_array.shape[0],
        point_array.ctypes.data,
        count,
        cell_array.ctypes.data,
        float(floor),
        valid.ctypes.data,
        lower.ctypes.data,
        upper.ctypes.data,
    )
    _raise_for_call(status, "tetrahedra_validity")
    return valid.astype(np.bool_), lower, upper


def cell_map_validity(
    domain: str,
    basis: Sequence[Sequence[tuple[tuple[int, int, int], int]]],
    denominator_log2: int,
    kind: str,
    significands: object,
    exponents: object,
    sparse_exponent: int,
    routes: object,
    floor: float,
    /,
    *,
    parallelism: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact root validity proof of dof-linear polynomial cell maps in 3D.

    ``basis`` lists each dof's integer terms ``(exponent, numerator)`` of the
    basis ``numerator / 2**denominator_log2 * xi**exponent``; the coordinate
    bank is the exact dyadic ``significands * 2**exponents`` of shape
    ``(rows, 3)`` (``sparse_exponent`` marks non-dyadic entries). Returns
    ``(decided, status, lower, upper, digests)``: a decided cell's
    ``CellValidityStatus`` value (0 certified, 1 invalid), directed
    determinant endpoints and exact-map digest equal those of the rational
    per-cell proof; undecided cells are left to it.
    """
    significand_array = np.ascontiguousarray(significands, dtype=np.int64)
    exponent_array = np.ascontiguousarray(exponents, dtype=np.int16)
    route_array = np.ascontiguousarray(routes, dtype=np.int64)
    if (
        significand_array.ndim != 2
        or significand_array.shape[1] != 3
        or exponent_array.shape != significand_array.shape
        or route_array.ndim != 2
        or route_array.shape[1] != len(basis)
    ):
        raise ValueError("Cell map validity requires (rows, 3) banks and dof routes.")
    starts = np.zeros((len(basis) + 1,), dtype=np.int64)
    starts[1:] = np.cumsum([len(terms) for terms in basis])
    term_exponents = np.asarray(
        [exponent for terms in basis for exponent, _ in terms], dtype=np.int32
    ).reshape(-1, 3)
    term_numerators = np.asarray(
        [numerator for terms in basis for _, numerator in terms], dtype=np.int64
    )
    encoded_kind = kind.encode("ascii")
    count = route_array.shape[0]
    decided = np.zeros((count,), dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    lower = np.zeros((count,), dtype=np.float64)
    upper = np.zeros((count,), dtype=np.float64)
    digests = np.zeros((count, 16), dtype=np.uint8)
    result = load_meshcore()["phx_mc_cell_map_validity"](
        {"simplex": 0, "box": 1, "prism": 2}[domain],
        len(basis),
        starts.ctypes.data,
        term_exponents.ctypes.data,
        term_numerators.ctypes.data,
        int(denominator_log2),
        encoded_kind,
        len(encoded_kind),
        significand_array.shape[0],
        significand_array.ctypes.data,
        exponent_array.ctypes.data,
        int(sparse_exponent),
        count,
        route_array.ctypes.data,
        float(floor),
        decided.ctypes.data,
        status.ctypes.data,
        lower.ctypes.data,
        upper.ctypes.data,
        digests.ctypes.data,
        parallelism,
    )
    _raise_for_call(result, "cell_map_validity")
    return decided.astype(np.bool_), status, lower, upper, digests


def welded_topology(
    dimension: int,
    key_count: int,
    cells: np.ndarray,
    audited_cells: int,
    facets: np.ndarray,
    facet_cells: np.ndarray,
    facet_signs: np.ndarray,
    audited_facets: int,
    audited_facet_width: int,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    scope: np.ndarray | None,
    /,
    *,
    embedded: bool,
    check_manifold: bool,
    check_watertight: bool,
    parallelism: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combinatorial welded-topology counts of one star complex.

    The first ``audited_cells`` cells and ``audited_facets`` facets are the
    audited mesh. Returns ``counts`` (collapsed cells, duplicate cells,
    nonmanifold facets, inconsistent facets, open boundary, nonmanifold
    vertices, nonmanifold edges) and the audited boundary-occurrence mask, as
    ``phx_mc_welded_topology`` documents.
    """
    cell_array = np.ascontiguousarray(cells, dtype=np.int64)
    facet_array = np.ascontiguousarray(facets, dtype=np.int64)
    facet_cell_array = np.ascontiguousarray(facet_cells, dtype=np.int64)
    sign_array = np.ascontiguousarray(facet_signs, dtype=np.int8)
    edge_array = np.ascontiguousarray(edges, dtype=np.int64)
    edge_cell_array = np.ascontiguousarray(edge_cells, dtype=np.int64)
    scope_array = (
        None if scope is None else np.ascontiguousarray(scope, dtype=np.int64).reshape(-1)
    )
    if (
        cell_array.ndim != 2
        or facet_array.ndim != 2
        or facet_cell_array.shape != (facet_array.shape[0],)
        or sign_array.shape != (facet_array.shape[0],)
        or edge_array.ndim != 2
        or edge_array.shape[1] != 2
        or edge_cell_array.shape != (edge_array.shape[0],)
    ):
        raise ValueError("Welded topology requires padded cell, facet and edge rows.")
    counts = np.zeros((7,), dtype=np.int64)
    boundary = np.zeros((audited_facets,), dtype=np.int8)
    status = load_meshcore()["phx_mc_welded_topology"](
        dimension,
        int(embedded),
        int(check_manifold),
        int(check_watertight),
        key_count,
        cell_array.shape[0],
        audited_cells,
        cell_array.shape[1],
        cell_array.ctypes.data,
        facet_array.shape[0],
        audited_facets,
        facet_array.shape[1],
        audited_facet_width,
        facet_array.ctypes.data,
        facet_cell_array.ctypes.data,
        sign_array.ctypes.data,
        edge_array.shape[0],
        edge_array.ctypes.data,
        edge_cell_array.ctypes.data,
        -1 if scope_array is None else scope_array.size,
        None if scope_array is None else scope_array.ctypes.data,
        counts.ctypes.data,
        boundary.ctypes.data,
        parallelism,
    )
    _raise_for_call(
        status,
        "welded_topology",
        contracts={
            MeshcoreStatus.INVALID_INPUT: (
                "Welded star links reference unknown incidences."
            )
        },
    )
    return counts, boundary.astype(np.bool_)


def common_denominator_rows(
    points: Sequence[Sequence[object]], /
) -> tuple[list[int], int] | None:
    """Integer rows ``X = L x`` of rational points over their least common ``L``.

    Flattened row-major; ``None`` when a coordinate is not rational.
    """
    values: list[Fraction] = []
    for point in points:
        for value in point:
            if not isinstance(value, (Fraction, int)) or isinstance(value, bool):
                return None
            values.append(Fraction(value))
    common = lcm(*(value.denominator for value in values)) if values else 1
    return [value.numerator * (common // value.denominator) for value in values], common


def _integer_word_bank(
    values: Sequence[int], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Little-endian 32-bit magnitude words, offsets and signs of integers."""
    counts = np.asarray(
        [(abs(value).bit_length() + 31) // 32 for value in values], dtype=np.int64
    )
    offsets = np.zeros((counts.size + 1,), dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    payload = b"".join(
        abs(value).to_bytes(4 * int(count), "little")
        for value, count in zip(values, counts.tolist(), strict=True)
    )
    words = np.frombuffer(payload, dtype="<u4").astype(np.uint32)
    signs = np.asarray([-1 if value < 0 else 1 for value in values], dtype=np.int8)
    return offsets, np.ascontiguousarray(words), signs


def polyhedral_star_validity(
    point_offsets: np.ndarray,
    integers: Sequence[int],
    cell_face_offsets: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_values: np.ndarray,
    star_offsets: np.ndarray,
    stars: np.ndarray,
    floor: float,
    /,
    *,
    parallelism: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact star validity of polyhedra (``phx_mc_polyhedral_star_validity``).

    ``integers`` holds every cell's integer point rows ``X = L x`` (three per
    point, cells in order) followed by each cell's common denominator ``L``;
    ``stars`` rows are ``(local face, first point, second point)``. Returns
    ``(decided, status, lower, upper)`` per cell, equal to the rational
    polyhedral star proof where decided.
    """
    point_offset_array = np.ascontiguousarray(point_offsets, dtype=np.int64)
    cell_faces = np.ascontiguousarray(cell_face_offsets, dtype=np.int64)
    face_offsets = np.ascontiguousarray(face_vertex_offsets, dtype=np.int64)
    face_values = np.ascontiguousarray(face_vertex_values, dtype=np.int64)
    star_offset_array = np.ascontiguousarray(star_offsets, dtype=np.int64)
    star_array = np.ascontiguousarray(stars, dtype=np.int64)
    if (
        point_offset_array.ndim != 1
        or point_offset_array.size < 1
        or cell_faces.shape != point_offset_array.shape
        or star_offset_array.shape != point_offset_array.shape
        or face_offsets.ndim != 1
        or face_offsets.size < 1
        or star_array.ndim != 2
        or star_array.shape[1] != 3
        or len(integers) != 3 * int(point_offset_array[-1]) + point_offset_array.size - 1
    ):
        raise ValueError("Polyhedral star validity requires CSR cells, faces and stars.")
    word_offsets, words, signs = _integer_word_bank(integers)
    count = point_offset_array.size - 1
    decided = np.zeros((count,), dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    lower = np.zeros((count,), dtype=np.float64)
    upper = np.zeros((count,), dtype=np.float64)
    result = load_meshcore()["phx_mc_polyhedral_star_validity"](
        count,
        point_offset_array.ctypes.data,
        word_offsets.ctypes.data,
        words.ctypes.data,
        signs.ctypes.data,
        cell_faces.ctypes.data,
        face_offsets.ctypes.data,
        face_values.ctypes.data,
        star_offset_array.ctypes.data,
        star_array.ctypes.data,
        float(floor),
        decided.ctypes.data,
        status.ctypes.data,
        lower.ctypes.data,
        upper.ctypes.data,
        parallelism,
    )
    _raise_for_call(result, "polyhedral_star_validity")
    return decided.astype(np.bool_), status, lower, upper


def polyhedral_star_decomposition(
    point_offsets: np.ndarray,
    integers: Sequence[int],
    face_integers: Sequence[int],
    cell_face_offsets: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_values: np.ndarray,
    /,
    *,
    parallelism: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact star decompositions of linear polytopes.

    ``integers`` holds every cell's integer point rows ``X = L x`` (three per
    point, cells in order) followed by each cell's ``L``; ``face_integers``
    every face-loop entry's row over its own face's common denominator.
    Returns ``(codes, face_reversed, triangle_offsets, triangle_counts,
    triangles)`` of ``phx_mc_polyhedral_star_decomposition``: per cell 0
    (decomposed), -1 (undecided) or the first refusal code; per face its
    outward reversal; per cell the ``triangle_counts`` rows ``(face, i, j, k)``
    from ``triangle_offsets`` of its oriented face triangles, local index -1
    being the face's exact vertex average (a fan apex).
    """
    point_offset_array = np.ascontiguousarray(point_offsets, dtype=np.int64)
    cell_faces = np.ascontiguousarray(cell_face_offsets, dtype=np.int64)
    face_offsets = np.ascontiguousarray(face_vertex_offsets, dtype=np.int64)
    face_values = np.ascontiguousarray(face_vertex_values, dtype=np.int64)
    if (
        point_offset_array.ndim != 1
        or point_offset_array.size < 1
        or cell_faces.shape != point_offset_array.shape
        or face_offsets.ndim != 1
        or face_offsets.size < 1
        or len(integers) != 3 * int(point_offset_array[-1]) + point_offset_array.size - 1
        or len(face_integers) != 3 * int(face_offsets[-1])
    ):
        raise ValueError("Polyhedral star decomposition requires CSR cells and faces.")
    word_offsets, words, signs = _integer_word_bank(integers)
    face_word_offsets, face_words, face_signs = _integer_word_bank(face_integers)
    count = point_offset_array.size - 1
    # A face has at most as many triangles as vertices (its fan).
    triangle_offsets = np.ascontiguousarray(face_offsets[cell_faces])
    codes = np.zeros((count,), dtype=np.int32)
    face_reversed = np.zeros((face_offsets.size - 1,), dtype=np.int8)
    triangles = np.zeros((int(triangle_offsets[-1]), 4), dtype=np.int32)
    triangle_counts = np.zeros((count,), dtype=np.int64)
    result = load_meshcore()["phx_mc_polyhedral_star_decomposition"](
        count,
        point_offset_array.ctypes.data,
        word_offsets.ctypes.data,
        words.ctypes.data,
        signs.ctypes.data,
        face_word_offsets.ctypes.data,
        face_words.ctypes.data,
        face_signs.ctypes.data,
        cell_faces.ctypes.data,
        face_offsets.ctypes.data,
        face_values.ctypes.data,
        triangle_offsets.ctypes.data,
        codes.ctypes.data,
        face_reversed.ctypes.data,
        triangles.ctypes.data,
        triangle_counts.ctypes.data,
        parallelism,
    )
    _raise_for_call(result, "polyhedral_star_decomposition")
    return (
        codes,
        face_reversed.astype(np.bool_),
        triangle_offsets,
        triangle_counts,
        triangles,
    )


def face_triangulation(
    face_vertex_offsets: np.ndarray, face_integers: Sequence[int], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ear-clipping triangulations of planar face loops.

    ``face_integers`` holds every loop entry's integer row over its own face's
    common denominator. Returns ``(codes, triangle_offsets, triangles)`` of
    ``phx_mc_face_triangulation``: per face 0 or its polygon refusal code, and
    rows ``[triangle_offsets[f], triangle_offsets[f + 1])`` of local loop
    index triples (``-1`` on refusal), those of ``triangulate_planar_polygon``.
    """
    offsets = np.ascontiguousarray(face_vertex_offsets, dtype=np.int64)
    if (
        offsets.ndim != 1
        or offsets.size < 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or len(face_integers) != 3 * int(offsets[-1])
    ):
        raise ValueError("Face triangulation requires CSR face loops and integer rows.")
    word_offsets, words, signs = _integer_word_bank(face_integers)
    count = offsets.size - 1
    triangle_offsets = np.zeros((offsets.size,), dtype=np.int64)
    np.cumsum(np.maximum(np.diff(offsets) - 2, 0), out=triangle_offsets[1:])
    codes = np.zeros((count,), dtype=np.int32)
    triangles = np.zeros((int(triangle_offsets[-1]), 3), dtype=np.int32)
    result = load_meshcore()["phx_mc_face_triangulation"](
        count,
        offsets.ctypes.data,
        word_offsets.ctypes.data,
        words.ctypes.data,
        signs.ctypes.data,
        codes.ctypes.data,
        triangles.ctypes.data,
    )
    _raise_for_call(result, "face_triangulation")
    return codes, triangle_offsets, triangles


class PolyhedralContactBank(NamedTuple):
    """Prepared operands of ``phx_mc_polyhedral_declared_contacts``."""

    point_count: int
    point_word_offsets: np.ndarray
    point_words: np.ndarray
    point_signs: np.ndarray
    point_lower: np.ndarray
    point_upper: np.ndarray
    tet_points: np.ndarray
    cell_tet_offsets: np.ndarray
    cell_corner_offsets: np.ndarray
    corner_points: np.ndarray
    corner_vertices: np.ndarray
    row_word_offsets: np.ndarray
    row_words: np.ndarray
    row_signs: np.ndarray
    cell_face_offsets: np.ndarray
    face_offsets: np.ndarray
    face_corners: np.ndarray


def polyhedral_contact_bank(
    point_integers: Sequence[int],
    point_lower: np.ndarray,
    point_upper: np.ndarray,
    tet_points: np.ndarray,
    cell_tet_offsets: np.ndarray,
    cell_corner_offsets: np.ndarray,
    corner_points: np.ndarray,
    corner_vertices: np.ndarray,
    corner_integers: Sequence[int],
    cell_face_offsets: np.ndarray,
    face_offsets: np.ndarray,
    face_corners: np.ndarray,
    /,
) -> PolyhedralContactBank:
    """Prepare star-decomposed cells for ``polyhedral_declared_contacts``.

    ``point_integers`` holds the homogeneous integer row ``(X, W)`` of every
    exact point ``X / W`` (``W > 0``), whose outward binary64 enclosures are
    ``point_lower`` and ``point_upper``. Cell ``c`` owns the tetrahedra
    ``[cell_tet_offsets[c], cell_tet_offsets[c + 1])`` of ``tet_points``
    (four point indices each) and the corners ``[cell_corner_offsets[c],
    cell_corner_offsets[c + 1])``: their point indices, mesh vertex
    identifiers and integer rows (``corner_integers``: three per corner, cells
    in order, then each cell's positive common denominator). Its face loops are
    the local corner indices of faces ``[cell_face_offsets[c],
    cell_face_offsets[c + 1])`` in ``face_offsets`` CSR. Every corner coordinate
    is a ``Fraction``. A cell without tetrahedra is never proved.
    """
    point_word_offsets, point_words, point_signs = _integer_word_bank(point_integers)
    row_word_offsets, row_words, row_signs = _integer_word_bank(corner_integers)
    corner_offsets = np.ascontiguousarray(cell_corner_offsets, dtype=np.int64)
    lower = np.ascontiguousarray(point_lower, dtype=np.float64)
    if (
        lower.ndim != 2
        or lower.shape[1] != 3
        or lower.shape != np.shape(point_upper)
        or len(point_integers) != 4 * lower.shape[0]
        or len(corner_integers) != 3 * int(corner_offsets[-1]) + corner_offsets.size - 1
    ):
        raise ValueError(
            "Polyhedral contacts require four integers and an enclosure per point."
        )
    return PolyhedralContactBank(
        int(lower.shape[0]),
        point_word_offsets,
        point_words,
        point_signs,
        lower,
        np.ascontiguousarray(point_upper, dtype=np.float64),
        np.ascontiguousarray(tet_points, dtype=np.int64),
        np.ascontiguousarray(cell_tet_offsets, dtype=np.int64),
        corner_offsets,
        np.ascontiguousarray(corner_points, dtype=np.int64),
        np.ascontiguousarray(corner_vertices, dtype=np.int64),
        row_word_offsets,
        row_words,
        row_signs,
        np.ascontiguousarray(cell_face_offsets, dtype=np.int64),
        np.ascontiguousarray(face_offsets, dtype=np.int64),
        np.ascontiguousarray(face_corners, dtype=np.int64),
    )


def polyhedral_declared_contacts(
    bank: PolyhedralContactBank,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    parallelism: int,
) -> np.ndarray:
    """Pairs of ``bank`` cells whose every tetrahedron contact is proved to
    lie in one convex piece of their declared shared stratum
    (``phx_mc_polyhedral_declared_contacts``); ``False`` decides nothing."""
    pairs = np.ascontiguousarray(
        np.stack(
            (np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)),
            axis=1,
        )
    )
    declared = np.zeros((pairs.shape[0],), dtype=np.int8)
    result = load_meshcore()["phx_mc_polyhedral_declared_contacts"](
        bank.point_count,
        bank.point_word_offsets.ctypes.data,
        bank.point_words.ctypes.data,
        bank.point_signs.ctypes.data,
        bank.point_lower.ctypes.data,
        bank.point_upper.ctypes.data,
        bank.tet_points.ctypes.data,
        bank.cell_tet_offsets.size - 1,
        bank.cell_tet_offsets.ctypes.data,
        bank.cell_corner_offsets.ctypes.data,
        bank.corner_points.ctypes.data,
        bank.corner_vertices.ctypes.data,
        bank.row_word_offsets.ctypes.data,
        bank.row_words.ctypes.data,
        bank.row_signs.ctypes.data,
        bank.cell_face_offsets.ctypes.data,
        bank.face_offsets.ctypes.data,
        bank.face_corners.ctypes.data,
        pairs.shape[0],
        pairs.ctypes.data,
        declared.ctypes.data,
        parallelism,
    )
    _raise_for_call(result, "polyhedral_declared_contacts")
    return declared.astype(np.bool_)


class CadEntityKind(IntEnum):
    """Entity table addressed by ``cad_geometry_evaluate``."""

    SURFACE = 0
    CURVE = 1
    PCURVE = 2


class CadGeometryQuantity(IntEnum):
    """Values and derivatives (by order) or differential quantities."""

    VALUE = 0
    FIRST_DERIVATIVES = 1
    SECOND_DERIVATIVES = 2
    DIFFERENTIAL = 3


class CadModel(NamedTuple):
    """The flat CAD model of ``phx_mc_cad_model``: a geometry node graph and
    corner, surface, curve, loop, use and region tables (see the C header)."""

    node_kinds: np.ndarray
    node_integer_offsets: np.ndarray
    node_integers: np.ndarray
    node_real_offsets: np.ndarray
    node_reals: np.ndarray
    corner_points: np.ndarray
    surface_nodes: np.ndarray
    surface_orientations: np.ndarray
    surface_periods: np.ndarray
    surface_parameter_domains: np.ndarray
    curve_nodes: np.ndarray
    curve_corners: np.ndarray
    curve_ranges: np.ndarray
    loop_use_offsets: np.ndarray
    loop_surfaces: np.ndarray
    use_kinds: np.ndarray
    use_entities: np.ndarray
    use_senses: np.ndarray
    use_ranges: np.ndarray
    use_pcurve_nodes: np.ndarray
    region_surface_offsets: np.ndarray
    region_surfaces: np.ndarray
    region_orientations: np.ndarray


_CAD_MODEL_DTYPES = (
    np.int32,
    np.int64,
    np.int64,
    np.int64,
    np.float64,
    np.float64,
    np.int32,
    np.int8,
    np.float64,
    np.float64,
    np.int32,
    np.int32,
    np.float64,
    np.int64,
    np.int32,
    np.int8,
    np.int32,
    np.int8,
    np.float64,
    np.int32,
    np.int64,
    np.int32,
    np.int8,
)


def _cad_model_structure(model: CadModel, /) -> tuple[_CadModel, tuple[np.ndarray, ...]]:
    """The native view of ``model`` and the contiguous arrays it points into."""
    arrays = tuple(
        np.ascontiguousarray(value, dtype=dtype)
        for value, dtype in zip(model, _CAD_MODEL_DTYPES, strict=True)
    )
    view = CadModel(*arrays)
    structure = _CadModel(
        node_count=view.node_kinds.size,
        node_kinds=view.node_kinds.ctypes.data,
        node_integer_offsets=view.node_integer_offsets.ctypes.data,
        node_integers=view.node_integers.ctypes.data,
        node_real_offsets=view.node_real_offsets.ctypes.data,
        node_reals=view.node_reals.ctypes.data,
        corner_count=view.corner_points.size // 3,
        corner_points=view.corner_points.ctypes.data,
        surface_count=view.surface_nodes.size,
        surface_nodes=view.surface_nodes.ctypes.data,
        surface_orientations=view.surface_orientations.ctypes.data,
        surface_periods=view.surface_periods.ctypes.data,
        surface_parameter_domains=view.surface_parameter_domains.ctypes.data,
        curve_count=view.curve_nodes.size,
        curve_nodes=view.curve_nodes.ctypes.data,
        curve_corners=view.curve_corners.ctypes.data,
        curve_ranges=view.curve_ranges.ctypes.data,
        loop_count=view.loop_surfaces.size,
        loop_use_offsets=view.loop_use_offsets.ctypes.data,
        loop_surfaces=view.loop_surfaces.ctypes.data,
        use_count=view.use_kinds.size,
        use_kinds=view.use_kinds.ctypes.data,
        use_entities=view.use_entities.ctypes.data,
        use_senses=view.use_senses.ctypes.data,
        use_ranges=view.use_ranges.ctypes.data,
        use_pcurve_nodes=view.use_pcurve_nodes.ctypes.data,
        region_count=view.region_surface_offsets.size - 1,
        region_surface_offsets=view.region_surface_offsets.ctypes.data,
        region_surfaces=view.region_surfaces.ctypes.data,
        region_orientations=view.region_orientations.ctypes.data,
    )
    return structure, arrays


def cad_geometry_evaluate(
    model: CadModel,
    entity_kind: CadEntityKind,
    entity: int,
    parameters: object,
    quantity: CadGeometryQuantity,
    /,
) -> np.ndarray:
    """Binary64 geometry of one surface, 3D curve or p-curve of ``model``.

    ``parameters`` is ``(n, 2)`` for a surface and ``(n,)`` otherwise. Returns
    ``(n, width)``: the value and its derivatives order by order (surface x,
    Su, Sv, Suu, Suv, Svv; curve C, C', C''), or for ``DIFFERENTIAL`` the first
    fundamental form, unit normal and principal curvatures of a surface (8) or
    the speed and curvature of a curve (2). Floating-point proposals that
    bound nothing; a point outside an entity's evaluation domain is refused.
    """
    kind = CadEntityKind(entity_kind)
    selected = CadGeometryQuantity(quantity)
    inputs = 2 if kind is CadEntityKind.SURFACE else 1
    points = _real_array(parameters, "parameters")
    if (inputs == 2 and (points.ndim != 2 or points.shape[1] != 2)) or (
        inputs == 1 and points.ndim != 1
    ):
        raise ValueError(
            "parameters must have shape (n, 2) for a surface and (n,) otherwise."
        )
    match kind:
        case CadEntityKind.PCURVE:
            outputs = 2
        case CadEntityKind.SURFACE | CadEntityKind.CURVE:
            outputs = 3
    match selected:
        case CadGeometryQuantity.DIFFERENTIAL:
            width = 8 if inputs == 2 else 2
        case _:
            terms = (selected + 1) * (selected + 2) // 2 if inputs == 2 else selected + 1
            width = outputs * terms
    # The native view borrows these contiguous arrays for the call.
    structure, arrays = _cad_model_structure(model)
    count = points.shape[0]
    result = np.empty((count, width), dtype=np.float64)
    status = load_meshcore()["phx_mc_cad_geometry_evaluate"](
        ctypes.addressof(structure),
        kind,
        entity,
        count,
        points.ctypes.data,
        selected,
        result.ctypes.data,
    )
    del arrays
    _raise_for_call(status, "cad_geometry_evaluate")
    return result


# CAD fidelity verifier.
class CadFidelityMesh(NamedTuple):
    """Finished surface arrays; edge_chord names the V4 chord table or -1."""

    points: np.ndarray
    triangles: np.ndarray
    face_rows: np.ndarray
    uv: np.ndarray
    mesh_triangles: np.ndarray
    chart_to_mesh: np.ndarray
    edge_chord: np.ndarray


class CadFidelityChords(NamedTuple):
    """The same use and chord tables consumed by the chart verifier."""

    use_patch: np.ndarray
    use_range: np.ndarray
    loop_count: int
    chord_use: np.ndarray
    chord_loop: np.ndarray
    chord_uv: np.ndarray
    chord_range: np.ndarray


class CadFidelityResult(NamedTuple):
    """Directed triangle bounds and explicit unresolved/association evidence."""

    status: MeshcoreStatus
    upper: np.ndarray
    item_status: np.ndarray
    reasons: np.ndarray
    subdivisions: np.ndarray
    maximum_upper: float
    findings: np.ndarray


_CAD_FIDELITY_FINDING_DTYPE = np.dtype(
    [
        ("kind", np.int32),
        ("triangle", np.int64),
        ("edge", np.int64),
        ("other_triangle", np.int64),
        ("other_edge", np.int64),
        ("chord", np.int64),
    ],
    align=True,
)


def _cad_fidelity_array(
    value: np.ndarray, dtype: type[np.generic], name: str, /
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}.")
    return np.ascontiguousarray(array)


def verify_cad_fidelity(
    model: CadModel,
    coefficient_lower: np.ndarray,
    coefficient_upper: np.ndarray,
    mesh: CadFidelityMesh,
    chords: CadFidelityChords,
    tolerance: float,
    max_depth: int,
    /,
    *,
    triangle_edge_nodes: np.ndarray | None = None,
) -> CadFidelityResult:
    """Verify bounds conditional on an independently checked chart partition.

    With ``triangle_edge_nodes``, use quadratic triangle traces with the given
    midpoint nodes on chart-local edges (0, 1), (1, 2), and (2, 0). A collapsed
    edge uses the repeated corner as its node.
    """
    lower = _cad_fidelity_array(coefficient_lower, np.float64, "coefficient_lower")
    upper = _cad_fidelity_array(coefficient_upper, np.float64, "coefficient_upper")
    points = _cad_fidelity_array(mesh.points, np.float64, "points")
    triangles = _cad_fidelity_array(mesh.triangles, np.int64, "triangles")
    faces = _cad_fidelity_array(mesh.face_rows, np.int32, "face_rows")
    uv = _cad_fidelity_array(mesh.uv, np.float64, "uv")
    mesh_triangles = _cad_fidelity_array(mesh.mesh_triangles, np.int64, "mesh_triangles")
    chart_to_mesh = _cad_fidelity_array(mesh.chart_to_mesh, np.int64, "chart_to_mesh")
    edges = _cad_fidelity_array(mesh.edge_chord, np.int64, "edge_chord")
    use_patch = _cad_fidelity_array(chords.use_patch, np.int64, "use_patch")
    use_range = _cad_fidelity_array(chords.use_range, np.float64, "use_range")
    chord_use = _cad_fidelity_array(chords.chord_use, np.int64, "chord_use")
    chord_loop = _cad_fidelity_array(chords.chord_loop, np.int64, "chord_loop")
    chord_uv = _cad_fidelity_array(chords.chord_uv, np.float64, "chord_uv")
    chord_range = _cad_fidelity_array(chords.chord_range, np.float64, "chord_range")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3).")
    count = triangles.shape[0]
    symbol = "phx_mc_verify_cad_fidelity"
    trailing: tuple[int, ...] = ()
    if triangle_edge_nodes is not None:
        triangle_edge_nodes = _cad_fidelity_array(
            triangle_edge_nodes, np.float64, "triangle_edge_nodes"
        )
        if triangle_edge_nodes.shape != (count, 3, 3):
            raise ValueError("triangle_edge_nodes must have shape (m, 3, 3).")
        if not np.all(np.isfinite(triangle_edge_nodes)):
            raise ValueError("triangle_edge_nodes must be finite.")
        symbol = "phx_mc_verify_cad_quadratic_fidelity"
        trailing = (triangle_edge_nodes.ctypes.data,)
    if faces.shape != (count,) or uv.shape != (count, 3, 2) or edges.shape != (count, 3):
        raise ValueError("face_rows, uv and edge_chord must align with triangles.")
    if mesh_triangles.ndim != 2 or mesh_triangles.shape[1] != 3:
        raise ValueError("mesh_triangles must have shape (k, 3).")
    if chart_to_mesh.shape != (count,):
        raise ValueError("chart_to_mesh must have one entry per chart triangle.")
    if lower.shape != model.node_reals.shape or upper.shape != lower.shape:
        raise ValueError("Coefficient bounds must match the model real bank.")
    if use_patch.ndim != 1 or use_range.shape != (use_patch.size, 2):
        raise ValueError("Use rows and ranges must align.")
    if chord_use.ndim != 1:
        raise ValueError("chord_use must be one-dimensional.")
    chord_count = chord_use.size
    if (
        chord_loop.shape != (chord_count,)
        or chord_uv.shape != (chord_count, 2, 2)
        or chord_range.shape != (chord_count, 2)
    ):
        raise ValueError("Chord rows, UV and ranges must align.")
    structure, arrays = _cad_model_structure(model)
    bounds = np.empty(count, dtype=np.float64)
    statuses = np.empty(count, dtype=np.int32)
    reasons = np.empty(count, dtype=np.int32)
    subdivisions = np.empty(count, dtype=np.int32)
    maximum = ctypes.c_double()
    evidence_count = ctypes.c_int64()
    findings = np.empty(
        7 * count + chord_count + len(mesh_triangles), dtype=_CAD_FIDELITY_FINDING_DTYPE
    )
    status = load_meshcore()[symbol](
        ctypes.addressof(structure),
        lower.size,
        lower.ctypes.data,
        upper.ctypes.data,
        points.shape[0],
        points.ctypes.data,
        count,
        triangles.ctypes.data,
        faces.ctypes.data,
        uv.ctypes.data,
        len(mesh_triangles),
        mesh_triangles.ctypes.data,
        chart_to_mesh.ctypes.data,
        use_patch.size,
        use_patch.ctypes.data,
        use_range.ctypes.data,
        chords.loop_count,
        chord_count,
        chord_use.ctypes.data,
        chord_loop.ctypes.data,
        chord_uv.ctypes.data,
        chord_range.ctypes.data,
        edges.ctypes.data,
        tolerance,
        max_depth,
        count,
        bounds.ctypes.data,
        statuses.ctypes.data,
        reasons.ctypes.data,
        subdivisions.ctypes.data,
        ctypes.addressof(maximum),
        findings.size,
        findings.ctypes.data,
        ctypes.addressof(evidence_count),
        *trailing,
    )
    del arrays
    if status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
        _raise_for_call(status, "verify_cad_fidelity")
    return CadFidelityResult(
        MeshcoreStatus(status),
        bounds,
        statuses,
        reasons,
        subdivisions,
        maximum.value,
        findings[: evidence_count.value],
    )


# Surface embedding (V2) and chart partition (V4) verifiers.
_SURFACE_FINDING_DTYPE = np.dtype(
    [
        ("kind", np.int32),
        ("first", np.int64),
        ("second", np.int64),
        ("classification", np.int32),
    ],
    align=True,
)
_CHART_FAILURE_DTYPE = np.dtype(
    [("kind", np.int32), ("patch", np.int64), ("row", np.int64), ("other", np.int64)],
    align=True,
)


class _SurfaceEmbeddingVerdict(ctypes.Structure):
    _fields_ = (("status", ctypes.c_int32), ("finding_count", ctypes.c_int64))


class SurfaceFindingKind(IntEnum):
    """Finding kinds of ``verify_surface_embedding``."""

    OUTSIDE_DOMAIN = 1
    DEGENERATE = 2
    EDGE = 3
    VERTEX_LINK = 4
    CONTACT = 5
    EMPTY = 6


class ChartFailureKind(IntEnum):
    """Failure kinds of ``verify_chart_partition``."""

    ORIENTATION = 1
    EDGE = 2
    USE_COVERAGE = 3
    LOOP = 4
    OVERLAP = 5


class SurfaceEmbeddingResult(NamedTuple):
    """Exact embedding verdict of a closed triangle surface and its findings."""

    status: VerdictStatus
    findings: np.ndarray


def verify_surface_embedding(
    points: object, triangles: object, /
) -> SurfaceEmbeddingResult:
    """Certify that ``(m, 3)`` triangles form an embedded edge-manifold surface."""
    point_array = np.ascontiguousarray(points, dtype=np.float64)
    triangle_array = _cad_fidelity_array(np.asarray(triangles), np.int64, "triangles")
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3).")
    capacity = 3 * triangle_array.shape[0] + point_array.shape[0] + 64
    while True:
        findings = np.empty(capacity, dtype=_SURFACE_FINDING_DTYPE)
        verdict = _SurfaceEmbeddingVerdict()
        status = load_meshcore()["phx_mc_verify_surface_embedding"](
            point_array.shape[0],
            point_array.ctypes.data,
            triangle_array.shape[0],
            triangle_array.ctypes.data,
            capacity,
            findings.ctypes.data,
            ctypes.addressof(verdict),
        )
        if status != MeshcoreStatus.CAPACITY_EXCEEDED:
            break
        capacity *= 2
    _raise_for_call(status, "verify_surface_embedding")
    return SurfaceEmbeddingResult(
        VerdictStatus(verdict.status), findings[: verdict.finding_count]
    )


def verify_chart_partition(
    patch_count: int,
    triangle_patches: np.ndarray,
    triangle_uv: np.ndarray,
    chords: CadFidelityChords,
    /,
) -> np.ndarray:
    """Chart partition failures (orientation, edges, uses, loops, overlap); empty certifies."""
    patches = _cad_fidelity_array(triangle_patches, np.int32, "triangle_patches")
    uv = _cad_fidelity_array(triangle_uv, np.float64, "triangle_uv")
    count = patches.shape[0]
    if patches.shape != (count,) or uv.shape != (count, 3, 2):
        raise ValueError("triangle_patches and triangle_uv must align.")
    use_patch = _cad_fidelity_array(chords.use_patch, np.int64, "use_patch")
    use_range = _cad_fidelity_array(chords.use_range, np.float64, "use_range")
    chord_use = _cad_fidelity_array(chords.chord_use, np.int64, "chord_use")
    chord_loop = _cad_fidelity_array(chords.chord_loop, np.int64, "chord_loop")
    chord_uv = _cad_fidelity_array(chords.chord_uv, np.float64, "chord_uv")
    chord_range = _cad_fidelity_array(chords.chord_range, np.float64, "chord_range")
    evidence_count = ctypes.c_int64()
    capacity = 3 * count + chord_use.size + 16
    while True:
        evidence = np.empty(capacity, dtype=_CHART_FAILURE_DTYPE)
        status = load_meshcore()["phx_mc_verify_chart_partition"](
            patch_count,
            count,
            patches.ctypes.data,
            uv.ctypes.data,
            use_patch.size,
            use_patch.ctypes.data,
            use_range.ctypes.data,
            chords.loop_count,
            chord_use.size,
            chord_use.ctypes.data,
            chord_loop.ctypes.data,
            chord_uv.ctypes.data,
            chord_range.ctypes.data,
            capacity,
            evidence.ctypes.data,
            ctypes.addressof(evidence_count),
        )
        if status != MeshcoreStatus.CAPACITY_EXCEEDED:
            break
        capacity = evidence_count.value
    _raise_for_call(status, "verify_chart_partition")
    return evidence[: evidence_count.value]


class RestrictedDelaunayCriteria(NamedTuple):
    """Refinement targets of `implicit_restricted_delaunay`.

    Restricted facets are refined while their longest edge exceeds
    ``facet_edge``, their circumcenter lies farther than ``facet_distance``
    from their surface Delaunay ball center, or their smallest angle is below
    ``facet_angle_degrees`` (at most 30). Inside cells are refined while their
    circumradius exceeds ``cell_size`` or their radius-edge ratio exceeds
    ``radius_edge_bound`` (None: no ratio target); cells then below the ratio,
    ``minimum_dihedral_degrees`` or ``minimum_relative_determinant`` go through
    the fixed-boundary quality improvement. Box roots closer than
    ``seed_spacing * facet_edge`` to an earlier seed are left to refinement.
    """

    facet_edge: float
    facet_distance: float
    facet_angle_degrees: float
    cell_size: float
    radius_edge_bound: float | None
    minimum_dihedral_degrees: float
    minimum_relative_determinant: float
    seed_spacing: float


class RestrictedImplicitVolume(NamedTuple):
    """Tetrahedra of ``{f <= 0}`` by restricted Delaunay refinement (construction only).

    ``points`` rows are the vertices, ``cells`` positive tetrahedra and
    ``boundary_faces`` the outward boundary triangles. The construction report:
    refinement ``insertions``, surface ``repair_rounds``, whether the
    restricted surface matched the requested topology, the box roots no
    surface vertex covers, the facets still violating the criteria because
    the separation bound refused their refinement point, and the cells the
    quality improvement left unmet. ``status`` is ``UNRESOLVED`` when the
    separation bound stopped or refused the refinement, or an inside cell
    reached the far box; the arrays are then the proposal reached.
    """

    points: np.ndarray
    cells: np.ndarray
    boundary_faces: np.ndarray
    insertions: int
    repair_rounds: int
    topology_matched: bool
    uncovered_roots: int
    refused_facets: int
    unmet_cells: int
    status: MeshcoreStatus


def implicit_restricted_delaunay(
    instructions: np.ndarray,
    indices: np.ndarray,
    parameters: np.ndarray,
    seed_lower: np.ndarray,
    seed_upper: np.ndarray,
    output_entry: int,
    guards: np.ndarray,
    lower: object,
    upper: object,
    criteria: RestrictedDelaunayCriteria,
    components: int,
    euler_characteristic: int,
    extra_seeds: object,
    /,
) -> RestrictedImplicitVolume:
    """Restricted Delaunay meshing of an interval program's solid ``{f <= 0}``.

    ``instructions``, ``indices`` and ``parameters`` are a validated native
    interval program with three inputs; ``seed_lower`` and ``seed_upper`` its
    interval workspace templates, ``output_entry`` the workspace entry of f
    and ``guards`` the workspace entries whose lower bound must be ``>= 0``
    (the program's domain guards: a box on which one may fail is kept by the
    seed cover). The box ``[lower, upper]`` contains the zero set, whose restricted
    surface should have ``components`` edge-connected components and Euler
    characteristic ``euler_characteristic``. ``extra_seeds`` ``(n, 3)`` are
    points near the zero set added to the box-tree seeds. The result is a
    proposal: it is certified afterwards.
    """
    lower_array = _real_array(lower, "lower")
    upper_array = _real_array(upper, "upper")
    if lower_array.shape != (3,) or upper_array.shape != (3,):
        raise ValueError("lower and upper must have shape (3,).")
    guard_array = np.ascontiguousarray(
        _native_integer_array(guards, "guards", np.dtype(np.int64))
    )
    if guard_array.ndim != 1:
        raise ValueError("guards must be one-dimensional.")
    seeds = np.ascontiguousarray(extra_seeds, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("extra_seeds must have shape (n, 3).")
    values = np.asarray(
        (
            criteria.facet_edge,
            criteria.facet_distance,
            criteria.facet_angle_degrees,
            criteria.cell_size,
            0.0 if criteria.radius_edge_bound is None else criteria.radius_edge_bound,
            criteria.minimum_dihedral_degrees,
            criteria.minimum_relative_determinant,
            criteria.seed_spacing,
        ),
        dtype=np.float64,
    )
    topology = np.asarray((components, euler_characteristic), dtype=np.int64)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_implicit_restricted_delaunay"](
            instructions.ctypes.data,
            instructions.shape[0],
            indices.ctypes.data,
            indices.size,
            parameters.ctypes.data,
            parameters.size,
            seed_lower.ctypes.data,
            seed_upper.ctypes.data,
            seed_lower.size,
            output_entry,
            guard_array.ctypes.data,
            guard_array.size,
            lower_array.ctypes.data,
            upper_array.ctypes.data,
            values.ctypes.data,
            topology.ctypes.data,
            seeds.shape[0],
            seeds.ctypes.data,
            ctypes.addressof(handle),
        )
    )
    if status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
        _raise_for_call(status, "implicit_restricted_delaunay")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "implicit_restricted_delaunay: missing volume"
        )
    # The native handle is released even when a copy fails.
    try:
        sizes = np.zeros((3,), dtype=np.int64)
        report = np.zeros((6,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_restricted_volume_sizes"](
                handle, sizes.ctypes.data, report.ctypes.data
            ),
            "restricted_volume_sizes",
        )
        points = np.empty((int(sizes[0]), 3), dtype=np.float64)
        cells = np.empty((int(sizes[1]), 4), dtype=np.int32)
        faces = np.empty((int(sizes[2]), 3), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_restricted_volume_copy"](
                handle, points.ctypes.data, cells.ctypes.data, faces.ctypes.data
            ),
            "restricted_volume_copy",
        )
    finally:
        library["phx_mc_restricted_volume_free"](handle)
    return RestrictedImplicitVolume(
        points,
        cells,
        faces,
        int(report[0]),
        int(report[1]),
        bool(report[2]),
        int(report[3]),
        int(report[4]),
        int(report[5]),
        status,
    )


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
    face_remote_sites: np.ndarray
    face_remote_tets: np.ndarray


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
    largest such region in cells. Then the gift-wrapped constrained
    Delaunay fills: how many ran and the cells they laid in total and at
    most in one fill (one step per cell), a deterministic work measure.
    Then the declared ``PlcSearchBudget``, the most counted bytes one search
    held and its deepest state (cells placed), and whether a search went over
    the budget with the states, bytes and depth of the first such search when
    it stopped. An over-budget search is inconclusive: the recovery goes on
    to relocation by peeling and refuses ``STEINER_SUPPRESSION`` only if that
    fails too. Last, the inconclusive searches, the points removed by
    relocation by peeling, and those of them after an inconclusive search
    (``PlcSuppressionRoute`` rows give each such point).
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
    node_budget: int
    byte_budget: int
    max_bytes: int
    max_depth: int
    exceeded: int
    exceeded_nodes: int
    exceeded_bytes: int
    exceeded_depth: int
    inconclusive: int
    peeled: int
    peeled_after_inconclusive: int


class PlcSuppressionRouteKind(IntEnum):
    SEARCH = 1
    PEEL = 2
    PEEL_AFTER_INCONCLUSIVE = 3


class PlcSuppressionRoute(NamedTuple):
    """Route that removed one suppressed split point, one row per wedge.

    Recorded for the exhaustive search and for relocation by peeling, in
    removal order: the point (work id, as refusals name it), its witness
    stratum and carrier entity, the route, and the search's states, bytes and
    depth (the filling search; the point's first inconclusive search before
    peeling; zeros for peeling without one). A deterministic function of the
    declared inputs, the search budget included.
    """

    point: int
    stratum: int
    entity: int
    route: PlcSuppressionRouteKind
    states: int
    bytes: int
    depth: int


class PlcSearchBudget(NamedTuple):
    """Declared resource budget of the exhaustive suppression search.

    Deciding whether a cavity has a Steiner-free fill is NP-complete, so the
    search has no structural bound. ``nodes`` bounds the states the searches
    of one recovery expand in total, ``bytes`` the counted bytes one search
    holds (40 per expanded state in chunks of 4096, 4 per memo table slot,
    64 per ranked open side plus 4 per apex; the same on every platform).
    An over-budget search is inconclusive (``PlcSuppressionSearch``): recovery
    goes on to relocation by peeling. It is a plan
    input, never derived from host memory; ``PLC_SEARCH_BUDGET`` is the
    documented default (the native ``PHX_MC_PLC3D_SEARCH_*_DEFAULT``).
    """

    nodes: int
    bytes: int


PLC_SEARCH_BUDGET = PlcSearchBudget(nodes=1 << 24, bytes=1 << 30)


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
    suppression_routes: tuple[PlcSuppressionRoute, ...] = ()


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


class NativePlcRecovery:
    """The native result of a PLC recovery, owned for ``TetMesh3D.from_recovery``.

    It holds the recovered mesh's banks and cell adjacency natively, so the
    mesh is adopted from them rather than built from the exported arrays; it
    is released when no recovery or mesh construction refers to it.
    """

    __slots__ = ("_handle", "_library")

    def __init__(self, library: MeshcoreLibrary, handle: ctypes.c_void_p, /) -> None:
        self._library = library
        self._handle = handle

    def __del__(self) -> None:
        self._library["phx_mc_plc3d_free"](self._handle)


class PlcRecovery3D(NamedTuple):
    """Recovered PLC; witness entities index raw carrier rows by stratum.

    Stratum 1 indexes ``plc_edges`` and stratum 2 indexes ``input_triangles``.
    ``native`` is the native result the arrays were exported from (None when
    the recovery was assembled from arrays), which ``TetMesh3D.from_recovery``
    adopts.
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
    suppression_routes: tuple[PlcSuppressionRoute, ...] = ()
    native: NativePlcRecovery | None = None


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


class TetMeshImplicitFacetRefusal(NamedTuple):
    """Actual implicit-facet predicate refusal for the latest edit attempt."""

    identity: TetMeshIdentity
    facet_id: int
    facet_source: int
    orientation: int
    vertex_ids: np.ndarray
    points: np.ndarray
    transversality_refusals: int
    distance_refusals: int
    status: TetMeshEditStatus
    distance_barycentric: np.ndarray | None


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
    vertex_size_targets: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    cell_size_targets: np.ndarray
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
    implicit_source: TetMeshImplicitSnapshot | None


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
    # Refinement: the circumcenter lies beyond a fixed face. Improvement: a
    # radius-edge residual on a constrained face whose own triangle already
    # exceeds the bound, so no tetrahedron on that face can meet it.
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
# Facet ID of an unmet row without a forcing constrained facet.
TET_MESH_UNMET_NO_FACET = np.iinfo(np.uint64).max


class TetMeshUnmet(NamedTuple):
    cell_ids: np.ndarray
    criteria: np.ndarray
    reasons: np.ndarray
    values: np.ndarray
    # INPUT_ANGLE rows: the acute feature's apex vertex ID (a ridge's first
    # endpoint) and angle in degrees; TET_MESH_UNMET_NO_APEX and NaN otherwise.
    apex_vertex_ids: np.ndarray
    apex_angles: np.ndarray
    # Improvement FIXED_BOUNDARY radius-edge rows: the constrained facet whose
    # own triangle radius-edge ratio (at most the cell's) exceeds the bound;
    # TET_MESH_UNMET_NO_FACET and NaN otherwise.
    facet_ids: np.ndarray
    facet_ratios: np.ndarray


class TetMeshQuality(NamedTuple):
    """Read-only final quality residuals bound to a mesh source and generation."""

    identity: TetMeshIdentity
    cell_ids: np.ndarray
    criteria: np.ndarray
    reasons: np.ndarray
    values: np.ndarray


class TetMeshCandidateQuality(NamedTuple):
    """Quality of a separate candidate, bound to its payload and native history."""

    historical_identity: TetMeshIdentity
    candidate_id: str
    cell_ids: np.ndarray
    criteria: np.ndarray
    reasons: np.ndarray
    values: np.ndarray


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
    remesh_counters: np.ndarray | None = None


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
    *,
    remesh: bool = False,
) -> SurfaceEnvelopeResult:
    """Tetrahedralize an explicit unsigned-distance envelope around a triangle soup.

    ``offset`` is the requested envelope radius. The native kernel returns its
    independent source/envelope deviation, envelope-distance interval, and
    interpolation error in ``bounds``; it refuses when the offset cannot
    dominate those certified errors. This does not infer an interior from an
    unsigned zero set.

    With ``remesh`` the extracted surface is remeshed toward edge length
    ``2 * spacing`` and a 30 degree minimum angle by floating-point
    construction, then certified: every triangle inside the certified band
    ``[bounds[2], bounds[1]]`` and the surface an embedded oriented closed
    2-manifold, refused triangles repaired locally by splits (a triangle left
    refused raises ``MeshcoreError``), and ``bounds[0]`` is recomputed. The
    tetrahedral carrier is then dropped (no tetrahedra; the volume vertices are
    the surface vertices), and ``remesh_counters`` reports the splits,
    collapses, flips, relocations and the triangles below the angle.
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
        remesh_counters: np.ndarray | None = None
        if remesh:
            remesh_counters = np.zeros((5,), dtype=np.int64)
            _raise_for_call(
                library["phx_mc_surface_envelope_remesh"](
                    handle, bounds.ctypes.data, remesh_counters.ctypes.data
                ),
                "surface_envelope_remesh",
            )
            remesh_counters.setflags(write=False)
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
            or (tetrahedra.shape[0] == 0) != remesh
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
            remesh_counters,
        )
    finally:
        library["phx_mc_surface_envelope_free"](handle)


class IntervalBoxProgram(NamedTuple):
    """A native interval program over one three-dimensional box.

    The workspace template carries the program's constant seeds; entries
    ``[0, 3)`` receive the box and ``outputs`` name the result entries.
    """

    instructions: np.ndarray
    indices: np.ndarray
    parameters: np.ndarray
    workspace_lower: np.ndarray
    workspace_upper: np.ndarray
    outputs: np.ndarray
    guards: np.ndarray
    """Workspace entries (real-power bases) whose lower bound must be >= 0."""


class IntervalBranchProgram(NamedTuple):
    """A one-box native interval program with its branch-evidence events.

    ``events`` (``(rows, 6)`` int64: kind, test, size, a, a_index, boxes; see
    ``phx_mc_interval_program_branches``) clear the box's ``continuous`` flag
    where a jump (a select that may switch, a pole, a domain edge) may lie in
    it; ``program`` is the field lowered together with those events.
    """

    program: IntervalBoxProgram
    events: np.ndarray


class TetMeshImplicitSource(NamedTuple):
    """Source-bound native interval programs and exterior facet group rows.

    ``value`` carries f's branch-evidence events: every enclosure a facet
    admission reads is over a box on which they report f continuous (else the
    facet is refused), and transversality also requires f smooth there.
    """

    value: IntervalBranchProgram
    gradient: IntervalBoxProgram
    group_rows: np.ndarray
    tolerance: float
    orientation: int
    source_id: str
    source_revision: str
    binding_id: str
    frame_id: str


class TetMeshImplicitSnapshot(NamedTuple):
    source: TetMeshImplicitSource
    transversality_refusals: int
    distance_refusals: int


class ZeroSetRemeshResult(NamedTuple):
    vertices: np.ndarray
    triangles: np.ndarray
    orientation: int
    counters: np.ndarray
    """Splits, collapses, flips, relocations, triangles below the minimum angle,
    the orientation, triangles sampled above the tolerance, and 0."""


def _interval_box_program(
    program: IntervalBoxProgram, outputs: int, name: str, /
) -> tuple[_IntervalBoxProgram, tuple[np.ndarray, ...]]:
    instructions = _native_integer_array(
        program.instructions, f"{name}.instructions", np.dtype(np.int64)
    )
    indices = _native_integer_array(
        program.indices, f"{name}.indices", np.dtype(np.int64)
    )
    locations = _native_integer_array(
        program.outputs, f"{name}.outputs", np.dtype(np.int64)
    )
    guards = _native_integer_array(
        program.guards, f"{name}.guards", np.dtype(np.int64)
    ).reshape((-1,))
    parameters = np.ascontiguousarray(program.parameters, dtype=np.float64)
    lower = np.ascontiguousarray(program.workspace_lower, dtype=np.float64)
    upper = np.ascontiguousarray(program.workspace_upper, dtype=np.float64)
    if (
        instructions.ndim != 2
        or instructions.shape[1] != 8
        or indices.ndim != 1
        or parameters.ndim != 1
        or lower.ndim != 1
        or upper.shape != lower.shape
        or locations.shape != (outputs,)
    ):
        raise ValueError(
            f"{name} must be an interval program with (rows, 8) instructions, "
            f"aligned one-dimensional workspaces, and {outputs} output locations."
        )
    view = _IntervalBoxProgram(
        instructions.ctypes.data,
        instructions.shape[0],
        indices.ctypes.data,
        indices.shape[0],
        parameters.ctypes.data,
        parameters.shape[0],
        lower.ctypes.data,
        upper.ctypes.data,
        lower.shape[0],
        locations.ctypes.data,
        locations.shape[0],
        guards.ctypes.data,
        guards.shape[0],
    )
    return view, (instructions, indices, locations, guards, parameters, lower, upper)


def _tet_implicit_source_payload(
    source: TetMeshImplicitSource, /
) -> tuple[_TetImplicitSource, tuple[object, ...]]:
    if not isinstance(source, TetMeshImplicitSource):
        raise TypeError("implicit_source must be TetMeshImplicitSource.")
    if not isinstance(source.value, IntervalBranchProgram):
        raise TypeError("implicit_source.value must be IntervalBranchProgram.")
    value, value_banks = _interval_box_program(
        source.value.program, 1, "implicit_source.value.program"
    )
    events = np.ascontiguousarray(
        _native_integer_array(
            source.value.events, "implicit_source.value.events", np.dtype(np.int64)
        )
    )
    if events.ndim != 2 or events.shape[1] != 6:
        raise ValueError("implicit_source.value.events must have shape (rows, 6).")
    gradient, gradient_banks = _interval_box_program(
        source.gradient, 3, "implicit_source.gradient"
    )
    groups = _native_integer_array(
        source.group_rows, "implicit_source.group_rows", np.dtype(np.int32)
    )
    if (
        groups.ndim != 1
        or groups.size == 0
        or np.any(groups < 0)
        or np.unique(groups).size != groups.size
    ):
        raise ValueError(
            "Implicit source group rows must be distinct nonnegative integers."
        )
    tolerance = float(source.tolerance)
    orientation = _int32_argument(source.orientation, "implicit_source.orientation")
    if not np.isfinite(tolerance) or tolerance <= 0 or orientation not in (-1, 0, 1):
        raise ValueError("Implicit source tolerance or orientation is invalid.")
    bound, encoded, buffers = _tet_mesh_identity_payload(
        source.source_id, source.source_revision, source.binding_id, source.frame_id
    )
    if not bound or any(not item for item in encoded):
        raise ValueError("Implicit source identity and frame must be nonempty.")
    view = _TetImplicitSource(
        value,
        gradient,
        events.ctypes.data,
        events.shape[0],
        _tet_mesh_pointer(groups),
        groups.size,
        tolerance,
        orientation,
        _tet_mesh_identity_pointer(bound, buffers[0]),
        len(encoded[0]),
        _tet_mesh_identity_pointer(bound, buffers[1]),
        len(encoded[1]),
        _tet_mesh_identity_pointer(bound, buffers[2]),
        len(encoded[2]),
        _tet_mesh_identity_pointer(bound, buffers[3]),
        len(encoded[3]),
        0,
        0,
    )
    return view, (*value_banks, *gradient_banks, events, groups, *buffers)


def _tet_implicit_source_snapshot(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> TetMeshImplicitSnapshot | None:
    view = _TetImplicitSource()
    _raise_for_call(
        library["phx_mc_tet_mesh_implicit_source"](handle, ctypes.byref(view)),
        "tet_mesh_implicit_source",
    )
    if view.group_count == 0:
        return None

    def copy_bank(pointer: int | None, length: int, dtype: np.dtype[Any]) -> np.ndarray:
        if length < 0:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid implicit source bank"
            )
        if length:
            if not pointer:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "invalid implicit source bank"
                )
            ctype = np.ctypeslib.as_ctypes_type(dtype)
            result = np.ctypeslib.as_array(
                ctypes.cast(pointer, ctypes.POINTER(ctype)), shape=(length,)
            ).copy()
        else:
            result = np.empty(0, dtype=dtype)
        result.setflags(write=False)
        return result

    def copy_program(program: _IntervalBoxProgram) -> IntervalBoxProgram:
        integer, real = np.dtype(np.int64), np.dtype(np.float64)
        return IntervalBoxProgram(
            copy_bank(
                program.instructions, 8 * program.instruction_count, integer
            ).reshape(-1, 8),
            copy_bank(program.indices, program.index_count, integer),
            copy_bank(program.parameters, program.parameter_count, real),
            copy_bank(program.workspace_lower, program.workspace_size, real),
            copy_bank(program.workspace_upper, program.workspace_size, real),
            copy_bank(program.outputs, program.output_count, integer),
            copy_bank(program.guards, program.guard_count, integer),
        )

    def text(pointer: int | None, length: int) -> str:
        if not pointer or length <= 0:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid implicit source identity"
            )
        return ctypes.string_at(pointer, length).decode("utf-8", "strict")

    source = TetMeshImplicitSource(
        IntervalBranchProgram(
            copy_program(view.value),
            copy_bank(
                view.value_events, 6 * view.value_event_count, np.dtype(np.int64)
            ).reshape(-1, 6),
        ),
        copy_program(view.gradient),
        copy_bank(view.groups, view.group_count, np.dtype(np.int32)),
        float(view.tolerance),
        int(view.orientation),
        text(view.source_id, view.source_id_length),
        text(view.source_revision, view.source_revision_length),
        text(view.binding_id, view.binding_id_length),
        text(view.frame_id, view.frame_id_length),
    )
    return TetMeshImplicitSnapshot(
        source, int(view.transversality_refusals), int(view.distance_refusals)
    )


def zero_set_remesh(
    vertices: object,
    triangles: object,
    value: IntervalBoxProgram,
    tolerance: float,
    carrier_distance: float,
    edge_length: float,
    minimum_angle_degrees: float,
    /,
) -> ZeroSetRemeshResult:
    """Remesh a closed oriented carrier of ``{f = 0}`` by floating point.

    Construction only, proving nothing: splits, collapses, flips and
    tangential relocations toward ``edge_length``, lowered where the zero set
    curves so that a well-shaped triangle stays within ``tolerance``; new and
    moved vertices are projected onto ``{f = 0}``. Input edges turning by more
    than 45 degrees are creases: every input triangle touching a crease vertex
    is kept bitwise. ``orientation`` is the carrier's orientation
    relative to ``grad f``; certify the result with ``zero_set_certify``. A
    vertex further than ``edge_length + carrier_distance`` from the zero set
    raises ``MeshcoreError`` (``UNRESOLVED``). A carrier that is not closed (an
    edge not held by exactly two triangles, in opposite directions) raises
    ``ValueError``.
    """
    points = np.ascontiguousarray(_points(vertices, "vertices", 3), dtype=np.float64)
    input_faces = _index_array(triangles, "triangles")
    compact_faces = input_faces.dtype == np.dtype(np.int32)
    faces = (
        np.ascontiguousarray(input_faces)
        if compact_faces
        else _native_integer_array(input_faces, "triangles", np.dtype(np.int64))
    )
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("triangles must be a nonempty (N, 3) integer array.")
    controls = (tolerance, carrier_distance, edge_length, minimum_angle_degrees)
    if any(not np.isfinite(value_) for value_ in controls) or not (
        tolerance > 0.0
        and carrier_distance >= 0.0
        and edge_length > 0.0
        and 0.0 < minimum_angle_degrees < 60.0
    ):
        raise ValueError(
            "tolerance and edge_length must be positive, carrier_distance nonnegative, "
            "and minimum_angle_degrees in (0, 60)."
        )
    value_view, value_arrays = _interval_box_program(value, 1, "value")
    handle = ctypes.c_void_p()
    counters = np.zeros((8,), dtype=np.int64)
    request = _ZeroSetRemeshRequest(
        points.ctypes.data,
        points.shape[0],
        faces.ctypes.data,
        faces.shape[0],
        value_view,
        float(tolerance),
        float(carrier_distance),
        float(edge_length),
        float(minimum_angle_degrees),
        ctypes.addressof(handle),
        counters.ctypes.data,
    )
    library = load_meshcore()
    entry = "phx_mc_zero_set_remesh_i32" if compact_faces else "phx_mc_zero_set_remesh"
    status = library[entry](ctypes.byref(request))
    del value_arrays  # kept alive through the native call
    _raise_for_call(status, "zero_set_remesh")
    out_vertices, out_triangles, _ = _take_surface_result(
        library, handle, "zero_set_remesh"
    )
    orientation = int(counters[5])
    if orientation not in (-1, 1):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "zero_set_remesh returned an invalid orientation.",
        )
    counters.setflags(write=False)
    return ZeroSetRemeshResult(out_vertices, out_triangles, orientation, counters)


def _take_surface_result(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, call: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy out and free an owned surface result (read-only arrays).

    Returns the vertices, the triangles and the per-triangle carried flags
    (bool; all false except for triangles a zero-set certificate accepted from
    its carrier by their witness boxes).
    """
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, f"{call} returned no owned result handle."
        )
    try:
        sizes = np.empty((2,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_surface_result_sizes"](handle, sizes.ctypes.data),
            "surface_result_sizes",
        )
        vertex_count, triangle_count = (int(size) for size in sizes.tolist())
        if vertex_count < 3 or triangle_count < 1:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, f"{call} returned an empty surface."
            )
        vertices = np.empty((vertex_count, 3), dtype=np.float64)
        triangles = np.empty((triangle_count, 3), dtype=np.int64)
        carried = np.empty((triangle_count,), dtype=np.uint8)
        _raise_for_call(
            library["phx_mc_surface_result_arrays"](
                handle,
                vertex_count,
                triangle_count,
                vertices.ctypes.data,
                triangles.ctypes.data,
            ),
            "surface_result_arrays",
        )
        _raise_for_call(
            library["phx_mc_surface_result_carried"](
                handle, triangle_count, carried.ctypes.data
            ),
            "surface_result_carried",
        )
    finally:
        library["phx_mc_surface_result_free"](handle)
    if (
        not np.all(np.isfinite(vertices))
        or np.any(triangles < 0)
        or np.any(triangles >= vertex_count)
        or np.any(carried > 1)
    ):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, f"{call} returned invalid geometry."
        )
    flags = carried.astype(np.bool_)
    vertices.setflags(write=False)
    triangles.setflags(write=False)
    flags.setflags(write=False)
    return vertices, triangles, flags


class ZeroSetCertificate(NamedTuple):
    """A surface certified against ``{f = 0}`` both ways, repaired locally.

    ``vertices`` and ``triangles`` are the certified surface: the input unless
    a refused triangle was repaired. ``distance`` bounds the distance from the
    surface to the zero set: the maximum over the proved triangles (each
    within the tolerance) and, for triangles accepted from the carrier, their
    witness box diagonals, which may exceed the tolerance.
    ``reverse_distance`` bounds the distance from every zero of ``f`` in the
    domain to the surface. ``counters`` (read-only int64) are the
    certifications refused for transversality and for distance and the
    triangles refused for embedding, all over every round and carrier
    triangles included, the triangles accepted from the carrier, the repair
    splits, the certification rounds run, the tube triangles refused in its
    last round, the cover leaves left outside by the last cover run (-1 past
    the box tree's level limit, -2 past ``cover_budget``, -3 for a surface
    vertex outside ``domain``), the proof of ``reverse_distance`` (1 tube, 2
    cover; either within ``tolerance``) and the boxes the covers evaluated.
    ``carried`` (read-only
    bool, one per triangle) flags the triangles accepted from the carrier.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    distance: float
    reverse_distance: float
    counters: np.ndarray
    carried: np.ndarray


def zero_set_certify(
    vertices: object,
    triangles: object,
    value: IntervalBranchProgram,
    gradient: IntervalBoxProgram,
    tolerance: float,
    orientation: int,
    carrier_vertices: object,
    carrier_triangles: object,
    carrier_boxes: object,
    domain: object,
    edge_length: float,
    minimum_angle_degrees: float,
    /,
    *,
    cover_budget: int = 1 << 23,
) -> ZeroSetCertificate:
    """Certify a ``zero_set_remesh`` result against ``{f = 0}``, repairing it locally.

    A triangle is certified when interval forms of ``f`` against a certified
    lower bound of ``|grad f|`` bound its distance by ``tolerance``
    (subdividing where needed) and its normal is certified transversal to
    ``grad f`` with ``orientation`` (+1 or -1), or, when that proof fails,
    when it has the corner coordinates, in the same cyclic order, of a carrier
    triangle (``zero_set_remesh`` keeps the carrier's triangles around
    creases, where transversality is not decidable) proved by its witness
    box: ``carrier_boxes`` (one ``(2, 3)`` lower/upper box per carrier
    triangle) must contain the triangle's corners, be proved free of jumps
    by the branch-evidence events of ``value`` (``f`` lowered with them: the
    same program gives the corner signs and every other proof), pass the
    guards of ``value`` over the whole box, and have two
    corners where the interval enclosures of ``f`` have strictly opposite
    signs, so the box holds a zero of ``f``; the triangle is then accepted at
    the box diagonal, and a box failing a check accepts nothing. Every
    certified triangle is also nondegenerate with only legal exact contacts
    with nearby triangles. Refused triangles are repaired: the longest edge
    of each is split at its midpoint projected onto ``{f = 0}`` and only what
    changed is certified again, for at most four rounds. Once a repair is
    needed, every triangle decided in a round must also be proved within
    ``0.8 * tolerance``; one proved only within ``tolerance`` is repaired as
    well (not refused), so the zero set near repaired triangles leaves the
    reverse direction room. A surface that needed no repair but whose tube
    proof bounds the zero set beyond ``tolerance`` is repaired into that band
    once and both directions are decided again. Every enclosure of
    ``f`` or ``grad f`` that a triangle, tube or witness proof reads is over a
    box on which ``value``'s branch evidence reports ``f`` continuous (else
    that proof refuses), and transversality also requires ``f`` smooth there;
    ``gradient`` is still trusted to enclose ``grad f`` (its Clarke gradient
    at kinks) wherever its enclosure is finite.

    The zero set is then bounded against the surface over ``domain`` (a
    ``(2, 3)`` lower/upper box) through the native possible-zero cover of the
    domain, which holds every zero of ``f`` in it: each kept box is accepted
    once an outward bound of its distance to the surface is below a radius
    ``D``. Proof 1 (tube): along segments ``y +- w(y)`` through the surface
    (interpolated vertex normals of length ``edge_length``, adapted) ``f`` is
    proved strictly monotone with one zero and caps clear of the surface by
    ``rho``; the cover at ``D = rho`` puts every zero on such a segment, and
    ``reverse_distance`` is the smaller of the tube's and the cover's bound.
    Proof 2 (cover): when the tube refuses or a leaf is left outside ``rho``,
    the cover alone at ``D = tolerance``. Each cover run evaluates at most
    ``cover_budget`` boxes or is refused. A surface that is not a closed
    oriented 2-manifold raises ``ValueError``; a triangle left refused, or a
    zero set neither proof holds, raises ``MeshcoreError`` (``UNRESOLVED``)
    naming the refusals.
    """
    if orientation not in (-1, 1):
        raise ValueError("orientation must be 1 or -1.")
    controls = (tolerance, edge_length, minimum_angle_degrees)
    if any(not np.isfinite(value_) for value_ in controls) or not (
        tolerance > 0.0 and edge_length > 0.0 and 0.0 < minimum_angle_degrees < 60.0
    ):
        raise ValueError(
            "tolerance and edge_length must be positive, minimum_angle_degrees in (0, 60)."
        )
    if isinstance(cover_budget, bool) or not isinstance(cover_budget, int):
        raise TypeError("cover_budget must be an int.")
    if cover_budget < 1:
        raise ValueError("cover_budget must be positive.")
    points = np.ascontiguousarray(_points(vertices, "vertices", 3), dtype=np.float64)
    faces = _native_integer_array(
        _index_array(triangles, "triangles"), "triangles", np.dtype(np.int64)
    )
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("triangles must be a nonempty (N, 3) integer array.")
    carrier_points = np.ascontiguousarray(
        _points(carrier_vertices, "carrier_vertices", 3), dtype=np.float64
    )
    carrier_faces = _native_integer_array(
        _index_array(carrier_triangles, "carrier_triangles"),
        "carrier_triangles",
        np.dtype(np.int64),
    )
    if carrier_faces.ndim != 2 or carrier_faces.shape[1] != 3:
        raise ValueError("carrier_triangles must be an (N, 3) integer array.")
    boxes = np.ascontiguousarray(np.asarray(carrier_boxes, dtype=np.float64))
    if boxes.shape != (carrier_faces.shape[0], 2, 3):
        raise ValueError("carrier_boxes must hold one (2, 3) box per carrier triangle.")
    bounds = np.ascontiguousarray(np.asarray(domain, dtype=np.float64))
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or not np.all(bounds[0] < bounds[1])
    ):
        raise ValueError("domain must be a finite (2, 3) box with lower < upper.")
    value_view, value_arrays = _interval_box_program(value.program, 1, "value.program")
    gradient_view, gradient_arrays = _interval_box_program(gradient, 3, "gradient")
    events = np.ascontiguousarray(
        _native_integer_array(value.events, "value.events", np.dtype(np.int64))
    )
    if events.ndim != 2 or events.shape[1] != 6:
        raise ValueError("value.events must have shape (rows, 6).")
    handle = ctypes.c_void_p()
    distances = np.zeros((2,), dtype=np.float64)
    counters = np.zeros((10,), dtype=np.int64)
    request = _ZeroSetCertifyRequest(
        points.ctypes.data,
        points.shape[0],
        faces.ctypes.data,
        faces.shape[0],
        value_view,
        gradient_view,
        events.ctypes.data,
        events.shape[0],
        float(tolerance),
        int(orientation),
        carrier_points.ctypes.data,
        carrier_points.shape[0],
        carrier_faces.ctypes.data,
        carrier_faces.shape[0],
        boxes.ctypes.data,
        bounds.ctypes.data,
        float(edge_length),
        float(minimum_angle_degrees),
        int(cover_budget),
        ctypes.addressof(handle),
        distances.ctypes.data,
        counters.ctypes.data,
    )
    library = load_meshcore()
    status = library["phx_mc_zero_set_certify"](ctypes.byref(request))
    del value_arrays, gradient_arrays  # kept alive through the native call
    if status == MeshcoreStatus.UNRESOLVED:
        outside = int(counters[7])
        cover = (
            f"the zero-set cover exceeded its budget of {int(cover_budget)} boxes"
            if outside == -2
            else "the zero-set cover exceeded the box tree's level limit"
            if outside == -1
            else "a surface vertex lies outside the domain"
            if outside == -3
            else f"{outside} zero-set cover leaves left outside"
        )
        raise MeshcoreError(
            MeshcoreStatus.UNRESOLVED,
            f"zero_set_certify left triangles refused after {int(counters[5])} rounds "
            f"and {int(counters[4])} repair splits ({int(counters[0])} refusals for "
            f"transversality, {int(counters[1])} for distance, {int(counters[2])} for "
            f"embedding; {int(counters[3])} triangles accepted from the carrier; "
            f"{int(counters[6])} tube triangles refused, {cover}).",
        )
    _raise_for_call(status, "zero_set_certify")
    out_vertices, out_triangles, carried = _take_surface_result(
        library, handle, "zero_set_certify"
    )
    certified, reverse = float(distances[0]), float(distances[1])
    if (
        not np.isfinite(certified)
        or certified < 0.0
        or (int(counters[3]) == 0 and certified > tolerance)
        or int(counters[3]) != int(np.count_nonzero(carried))
        or not np.isfinite(reverse)
        or reverse < 0.0
        or int(counters[8]) not in (1, 2)
        or reverse > tolerance
        or (int(counters[8]) == 2 and not reverse < tolerance)
    ):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "zero_set_certify returned invalid distance evidence.",
        )
    counters.setflags(write=False)
    return ZeroSetCertificate(
        out_vertices, out_triangles, certified, reverse, counters, carried
    )


_MESH_SOURCE_CONTRACTS = {
    MeshcoreStatus.INVALID_INPUT: (
        "every surface must be a closed oriented 2-manifold with nondegenerate triangles."
    )
}


class MeshRemeshResult(NamedTuple):
    """A surface constructed from a triangle-mesh source (nothing proved).

    ``counters`` (read-only int64): splits, collapses, flips, relocations,
    triangles below the minimum angle, and triangles whose sampled distance to
    the source exceeds the tolerance.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    counters: np.ndarray


def mesh_remesh(
    vertices: object,
    triangles: object,
    tolerance: float,
    edge_length: float,
    minimum_angle_degrees: float,
    /,
) -> MeshRemeshResult:
    """Remesh a closed oriented triangle mesh with itself as the source.

    Construction only, proving nothing: as ``zero_set_remesh`` with the input
    mesh as the surface (vertices projected to their nearest source point,
    normals and principal curvature estimates interpolated from the source's
    vertices). A source rough at this tolerance (vertex noise) is first
    Taubin-smoothed at a radius of about ``edge_length / 2``, each vertex kept
    within ``tolerance / 2`` of the input, and construction follows that copy;
    certify against the input with ``mesh_certify``. A source that is not a
    closed oriented 2-manifold with nondegenerate triangles raises
    ``ValueError``.
    """
    points, faces = _surface_arrays(vertices, triangles, "vertices", "triangles")
    _check_remesh_controls(tolerance, edge_length, minimum_angle_degrees)
    handle = ctypes.c_void_p()
    counters = np.zeros((6,), dtype=np.int64)
    request = _MeshRemeshRequest(
        points.ctypes.data,
        points.shape[0],
        faces.ctypes.data,
        faces.shape[0],
        float(tolerance),
        float(edge_length),
        float(minimum_angle_degrees),
        ctypes.addressof(handle),
        counters.ctypes.data,
    )
    library = load_meshcore()
    _raise_for_call(
        library["phx_mc_mesh_remesh"](ctypes.byref(request)),
        "mesh_remesh",
        contracts=_MESH_SOURCE_CONTRACTS,
    )
    out_vertices, out_triangles, _ = _take_surface_result(library, handle, "mesh_remesh")
    counters.setflags(write=False)
    return MeshRemeshResult(out_vertices, out_triangles, counters)


def _surface_arrays(
    vertices: object, triangles: object, vertices_name: str, triangles_name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous float64 (V, 3) vertices and nonempty int64 (T, 3) triangles."""
    points = np.ascontiguousarray(_points(vertices, vertices_name, 3), dtype=np.float64)
    faces = _native_integer_array(
        _index_array(triangles, triangles_name), triangles_name, np.dtype(np.int64)
    )
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"{triangles_name} must be a nonempty (N, 3) integer array.")
    return points, faces


def _check_remesh_controls(
    tolerance: float, edge_length: float, minimum_angle_degrees: float, /
) -> None:
    controls = (tolerance, edge_length, minimum_angle_degrees)
    if any(not np.isfinite(value_) for value_ in controls) or not (
        tolerance > 0.0 and edge_length > 0.0 and 0.0 < minimum_angle_degrees < 60.0
    ):
        raise ValueError(
            "tolerance and edge_length must be positive, minimum_angle_degrees in (0, 60)."
        )


class MeshCertificate(NamedTuple):
    """A surface certified against its source mesh in both directions.

    ``vertices`` and ``triangles`` are the certified surface: the input unless
    a refused triangle was repaired. ``distances`` (read-only float64) are the
    largest certified bound from the surface to the source and from the source
    to the surface, each within the tolerance. ``counters`` (read-only int64)
    are the certifications refused for distance and the triangles refused for
    embedding, the repair splits and the rounds run, all summed over every
    pass, the source triangles not proved within the tolerance (0 here), and
    the relocations restoring angles after repairs.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    distances: np.ndarray
    counters: np.ndarray


def mesh_certify(
    source_vertices: object,
    source_triangles: object,
    vertices: object,
    triangles: object,
    tolerance: float,
    edge_length: float,
    minimum_angle_degrees: float,
    /,
) -> MeshCertificate:
    """Certify a ``mesh_remesh`` result against its source mesh, repairing it locally.

    Every surface triangle gets an upper bound on the distance from each of
    its points to the source: the source triangles over it (reached across
    edges whose planar images along its normal meet its own) must cover it,
    proved by the winding of their boundary with exact orientation
    predicates, and the bound is the largest gap between the two surfaces
    over the same planar point, taken at the vertices of the overlaps (edge
    crossings within certified offsets, with rounding pads). Refused
    triangles are repaired (longest edge split at its nearest source point,
    flips, at most four rounds) with embedding checked exactly; then every
    source triangle is bounded the same way against the certified surface,
    along the surface's normals, and the surface triangles nearest a refused
    one are split. After any repair, angles are restored by flips and
    relocations and both directions are decided again, for at most three
    further passes. Sources that are not closed oriented 2-manifolds with
    nondegenerate triangles raise ``ValueError``; a refused triangle in
    either direction raises ``MeshcoreError`` (``UNRESOLVED``) naming the
    refusals.
    """
    source_points, source_faces = _surface_arrays(
        source_vertices, source_triangles, "source_vertices", "source_triangles"
    )
    points, faces = _surface_arrays(vertices, triangles, "vertices", "triangles")
    _check_remesh_controls(tolerance, edge_length, minimum_angle_degrees)
    handle = ctypes.c_void_p()
    distances = np.zeros((2,), dtype=np.float64)
    counters = np.zeros((6,), dtype=np.int64)
    request = _MeshCertifyRequest(
        source_points.ctypes.data,
        source_points.shape[0],
        source_faces.ctypes.data,
        source_faces.shape[0],
        points.ctypes.data,
        points.shape[0],
        faces.ctypes.data,
        faces.shape[0],
        float(tolerance),
        float(edge_length),
        float(minimum_angle_degrees),
        ctypes.addressof(handle),
        distances.ctypes.data,
        counters.ctypes.data,
    )
    library = load_meshcore()
    status = library["phx_mc_mesh_certify"](ctypes.byref(request))
    if status == MeshcoreStatus.UNRESOLVED:
        raise MeshcoreError(
            MeshcoreStatus.UNRESOLVED,
            f"mesh_certify left triangles refused after {int(counters[3])} rounds and "
            f"{int(counters[2])} repair splits ({int(counters[0])} refusals for distance, "
            f"{int(counters[1])} for embedding, {int(counters[4])} source triangles not "
            f"within {tolerance:g}; bounds {distances[0]:.3e} / {distances[1]:.3e}).",
        )
    _raise_for_call(status, "mesh_certify", contracts=_MESH_SOURCE_CONTRACTS)
    out_vertices, out_triangles, _ = _take_surface_result(library, handle, "mesh_certify")
    if not (
        np.all(np.isfinite(distances))
        and np.all(distances >= 0.0)
        and np.all(distances <= tolerance)
        and counters[4] == 0
    ):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "mesh_certify returned invalid distance evidence.",
        )
    distances.setflags(write=False)
    counters.setflags(write=False)
    return MeshCertificate(out_vertices, out_triangles, distances, counters)


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
        np.empty((faces_count,), dtype=np.int32),
        np.empty((faces_count,), dtype=np.int32),
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
    _raise_for_call(
        library["phx_mc_power_cells_export_face_remotes"](
            handle,
            faces_count,
            result.face_remote_sites.ctypes.data,
            result.face_remote_tets.ctypes.data,
        ),
        "power_cells_export_face_remotes",
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
    *,
    selected_sites: object | None = None,
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
    selection = None
    if selected_sites is not None:
        selection = _native_integer_array(
            selected_sites, "selected_sites", np.dtype(np.int32)
        )
        if (
            selection.ndim != 1
            or np.any(selection < 0)
            or np.any(selection >= owners.size)
            or np.unique(selection).size != selection.size
        ):
            raise ValueError("selected_sites must contain distinct valid image IDs.")
        selection = np.ascontiguousarray(np.sort(selection))
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
        selected_site_count=-1 if selection is None else selection.size,
        selected_sites=None if selection is None else selection.ctypes.data,
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


def point_triangle_distances(points: object, triangles: object, /) -> np.ndarray:
    """Return specified distances for paired points and closed triangles.

    Each result is the maximum of the fixed-order candidate distance and the
    point-to-triangle-box distance. The candidate considers the three edges
    and, when the normal resolves the face, its perpendicular projection.
    Results are approximate; collinear or nearly collinear triangles, very
    distant queries, and arithmetic underflow can reduce accuracy. For finite
    inputs outside the native exact-coordinate domain (nonzero magnitudes
    between 2**-120 and 2**120), only the computed box lower bound is assured.
    Nonfinite inputs or computed candidate/box distances are refused; the
    contiguous point and triangle input buffers must not overlap.
    """
    point_array = _points(points, "points", 3)
    triangle_array = _contact_simplices(triangles, "triangles", 3)
    if point_array.shape[0] != triangle_array.shape[0]:
        raise ValueError("points and triangles must have equal length.")
    distances = np.empty((point_array.shape[0],), dtype=np.float64)
    library = load_meshcore()
    call = library["phx_mc_point_triangle_distances"](
        point_array.shape[0],
        point_array.ctypes.data,
        triangle_array.ctypes.data,
        distances.ctypes.data,
    )
    _raise_for_call(
        call,
        "point_triangle_distances",
        contracts={MeshcoreStatus.RANGE_ERROR: "computed distance is nonfinite"},
    )
    return distances


def point_triangle_nearest_distances(points: object, triangles: object, /) -> np.ndarray:
    """Least ``point_triangle_distances`` value of every point over all triangles.

    Equal to the exhaustive minimum (a box tree skips only subtrees whose box
    distance exceeds the best value found).
    """
    point_array = _points(points, "points", 3)
    triangle_array = _contact_simplices(triangles, "triangles", 3)
    if triangle_array.shape[0] < 1:
        raise ValueError("triangles must be nonempty.")
    distances = np.empty((point_array.shape[0],), dtype=np.float64)
    call = load_meshcore()["phx_mc_point_triangle_nearest_distances"](
        point_array.shape[0],
        point_array.ctypes.data,
        triangle_array.shape[0],
        triangle_array.ctypes.data,
        distances.ctypes.data,
    )
    _raise_for_call(
        call,
        "point_triangle_nearest_distances",
        contracts={MeshcoreStatus.RANGE_ERROR: "computed distance is nonfinite"},
    )
    return distances


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
    affine_metric: object | None = None,
    protected_facets: object | None = None,
    search_budget: PlcSearchBudget | None = None,
) -> tuple[PlcSourceData, NativePlcRecovery]:
    """Run native PLC source preparation or full protected tetrahedral recovery.

    Returns the exported result and the native one it was exported from.
    """

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
    protected_array = None
    if protected_facets is not None:
        if operation != "phx_mc_plc3d_recover":
            raise ValueError("Facet protection applies only to PLC recovery.")
        flags = np.asarray(protected_facets)
        if (
            flags.shape != (facet_count,)
            or flags.dtype.kind not in "biu"
            or np.any(~np.isin(flags, (0, 1)))
        ):
            raise ValueError("protected_facets must contain one boolean flag per facet.")
        protected_array = np.ascontiguousarray(flags, dtype=np.uint8)
    if search_budget is None:
        if operation != "phx_mc_plc3d_prepare_source":
            raise ValueError("PLC recovery requires a declared search budget.")
        budget = (0, 0)
    else:
        if operation == "phx_mc_plc3d_prepare_source":
            raise ValueError("Source preparation runs no suppression search.")
        if not isinstance(search_budget, PlcSearchBudget) or any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 < value < 2**63
            for value in search_budget
        ):
            raise ValueError(
                "search_budget must be a PlcSearchBudget of positive int64 values."
            )
        budget = (search_budget.nodes, search_budget.bytes)

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
        *budget,
        ctypes.addressof(handle),
    )
    metric = (
        None
        if affine_metric is None
        else _periodic_float_bank(affine_metric, "affine_metric", (3, 3))
    )
    if protected_array is not None:
        code = MeshcoreStatus(
            library["phx_mc_plc3d_recover_protected"](
                ctypes.byref(request),
                protected_array.ctypes.data,
                None if metric is None else metric.ctypes.data,
            )
        )
    elif metric is None:
        code = MeshcoreStatus(library[operation](ctypes.byref(request)))
    else:
        code = MeshcoreStatus(
            library["phx_mc_plc3d_recover_with_affine_metric"](
                ctypes.byref(request), metric.ctypes.data
            )
        )
    if not handle.value:
        _raise_for_call(code, operation_label)
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"{operation_label}: missing result",
        )
    # The handle is owned from here on: released when the owner is dropped,
    # or kept with the recovery for TetMesh3D.from_recovery.
    native = NativePlcRecovery(library, handle)
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
    route_count = np.zeros((1,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_plc3d_suppression_routes"](
            handle, 0, route_count.ctypes.data, None
        ),
        "plc3d_suppression_routes",
    )
    route_rows = np.zeros(
        (int(route_count[0]), len(PlcSuppressionRoute._fields)), dtype=np.int64
    )
    _raise_for_call(
        library["phx_mc_plc3d_suppression_routes"](
            handle,
            route_rows.shape[0],
            route_count.ctypes.data,
            route_rows.ctypes.data,
        ),
        "plc3d_suppression_routes",
    )
    suppression_routes = tuple(
        PlcSuppressionRoute(
            int(row[0]),
            int(row[1]),
            int(row[2]),
            PlcSuppressionRouteKind(int(row[3])),
            int(row[4]),
            int(row[5]),
            int(row[6]),
        )
        for row in route_rows
    )
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
    point_count, tet_count, face_count, segment_count, edge_count, triangle_count = map(
        int, sizes
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
        suppression_routes,
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
    return result, native


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
    )[0]


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
    affine_metric: object | None = None,
    protected_facets: object | None = None,
    search_budget: PlcSearchBudget = PLC_SEARCH_BUDGET,
) -> PlcSourceData:
    """Recover a PLC with witnesses indexed by exported raw carrier banks.

    Stratum 1 entities index ``plc_edges`` and stratum 2 entities index
    ``input_triangles``. Python consumers must resolve these rows through exact
    source identity before constructing a persistent source bank. Declared
    facet planes are as for ``prepare_plc_source``. ``affine_metric`` is an
    optional nonsingular (3, 3) index-to-physical linear map. Coordinates,
    topology and witnesses remain in index space; distances are physical.
    ``protected_facets`` fixes the selected facets and all their incident edges
    while other facets may be conformingly subdivided. ``search_budget`` is the
    declared budget of the exhaustive suppression search (``PlcSearchBudget``).
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
        affine_metric=affine_metric,
        protected_facets=protected_facets,
        search_budget=search_budget,
    )[0]


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
    affine_metric: object | None = None,
    protected_facets: object | None = None,
    search_budget: PlcSearchBudget = PLC_SEARCH_BUDGET,
) -> PlcRecovery3D:
    """Recover a PLC with raw carrier-row witnesses and region labels.

    Stratum 1 entities index ``plc_edges``; stratum 2 entities index
    ``input_triangles``. These are not persistent canonical source-bank rows.
    Declared facet planes are as for ``prepare_plc_source``. ``affine_metric``
    is an optional nonsingular (3, 3) index-to-physical linear map.
    ``search_budget`` is as for ``recover_plc``.
    """

    segment_bank = np.empty((0, 2), dtype=np.int32) if segments is None else segments
    recovered, native = _plc3d_build(
        "phx_mc_plc3d_recover",
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
        affine_metric=affine_metric,
        protected_facets=protected_facets,
        search_budget=search_budget,
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
        recovered.suppression_routes,
        native,
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
    """Address of a C-contiguous bank; the caller keeps ``array`` alive.

    C reads a bank densely from its first element, so a strided view (a
    column of a row bank, say) would be read as interleaved values: it is
    refused rather than copied (a copy here would die before the call).
    """
    if array is None or array.size == 0:
        return None
    if not array.flags.c_contiguous:
        raise ValueError("Native banks must be C-contiguous arrays.")
    return int(array.ctypes.data)


def _source_group_labels(
    groups: np.ndarray, labels: np.ndarray, values: np.ndarray, kind: str, /
) -> np.ndarray:
    """``values`` (source group rows) as their group's source ID (int64).

    ``groups`` and ``labels`` give each source row's group and source ID; a
    group carries one source ID. A value naming no group is kept.
    """
    result = values.astype(np.int64)
    if groups.size == 0 or np.array_equal(groups, labels):
        # Every group is labelled by its own row: the map is the identity.
        return result
    # Groups are source rows (nonnegative and below the row count), so a
    # dense table maps them; any row whose label differs from its group's
    # table entry (whichever row wrote it) proves a group with two labels.
    if groups.min() < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, f"a {kind} source group row is negative"
        )
    span = int(groups.max()) + 1
    present = np.zeros((span,), dtype=bool)
    present[groups] = True
    table = np.empty((span,), dtype=labels.dtype)
    table[groups] = labels
    if np.any(labels != table[groups]):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, f"one {kind} group has multiple source IDs"
        )
    named = (values >= 0) & (values < span)
    named[named] = present[values[named]]
    result[named] = table[values[named]]
    return result


# The flat banks of a TetMesh snapshot: request field, dtype, row shape, the
# PHX_MC_TET_MESH_SNAPSHOT_SIZES entry counting its rows, the extra rows (an
# offsets bank closes its last row) and the bytes per row.
_TET_SNAPSHOT_BANKS: tuple[
    tuple[str, np.dtype[Any], tuple[int, ...], int, int, int], ...
] = tuple(
    (
        name,
        np.dtype(dtype),
        shape,
        count,
        extra,
        np.dtype(dtype).itemsize * int(np.prod(shape)),
    )
    for name, dtype, shape, count, extra in (
        ("vertex_ids", np.uint64, (), 0, 0),
        ("points", np.float64, (3,), 0, 0),
        ("vertex_size_targets", np.float64, (), 0, 0),
        ("witness_strata", np.int8, (), 0, 0),
        ("witness_entities", np.int32, (), 0, 0),
        ("witness_parameters", np.float64, (2,), 0, 0),
        ("witness_deviations", np.float64, (), 0, 0),
        ("vertex_dimensions", np.int8, (), 0, 0),
        ("protection_radii", np.float64, (), 0, 0),
        ("protected_vertices", np.int8, (), 0, 0),
        ("vertex_edge_witness_present", np.int8, (), 0, 0),
        ("vertex_edge_witness_endpoints", np.uint64, (2,), 0, 0),
        ("vertex_edge_parameters", np.float64, (), 0, 0),
        ("vertex_ancestry_offsets", np.int64, (), 0, 1),
        ("vertex_ancestor_ids", np.uint64, (), 17, 0),
        ("vertex_ancestor_winding", np.int64, (3,), 17, 0),
        ("vertex_ancestry_weights", np.float64, (), 17, 0),
        ("cell_ancestry_offsets", np.int64, (), 1, 1),
        ("cell_ancestor_ids", np.uint64, (), 18, 0),
        ("cell_ancestry_edit_kinds", np.int32, (), 1, 0),
        ("cell_ids", np.uint64, (), 1, 0),
        ("cell_vertex_rows", np.int64, (4,), 1, 0),
        ("cell_regions", np.int32, (), 1, 0),
        ("cell_size_targets", np.float64, (), 1, 0),
        ("facet_ids", np.uint64, (), 2, 0),
        ("facet_vertex_rows", np.int64, (3,), 2, 0),
        ("facet_sources", np.int32, (), 2, 0),
        ("facet_orientations", np.int8, (), 2, 0),
        ("protected_facets", np.int8, (), 2, 0),
        ("segment_ids", np.uint64, (), 3, 0),
        ("segment_vertex_rows", np.int64, (2,), 3, 0),
        ("segment_sources", np.int32, (), 3, 0),
        ("protected_segments", np.int8, (), 3, 0),
        ("source_points", np.float64, (3,), 4, 0),
        ("source_faces", np.int32, (3,), 5, 0),
        ("source_face_groups", np.int32, (), 5, 0),
        ("source_face_ids", np.int64, (), 5, 0),
        ("source_face_tolerances", np.float64, (), 5, 0),
        ("source_segments", np.int32, (2,), 6, 0),
        ("source_segment_groups", np.int32, (), 6, 0),
        ("source_segment_ids", np.int64, (), 6, 0),
        ("source_segment_tolerances", np.float64, (), 6, 0),
        ("source_face_planes", np.float64, (3, 3), 5, 0),
        ("source_face_plane_tolerances", np.float64, (), 5, 0),
        ("periodic_quotient_lifted_cell_ids", np.uint64, (), 10, 0),
        ("periodic_entity_lifted_ids", np.uint64, (), 12, 0),
        ("periodic_entity_quotient_ids", np.uint64, (), 13, 0),
        ("periodic_entity_lifted_to_quotient", np.int32, (), 12, 0),
        ("periodic_entity_orientation_signs", np.int32, (), 12, 0),
        ("periodic_entity_winding_shifts", np.int64, (), 14, 0),
        ("periodic_entity_corner_permutations", np.int32, (), 15, 0),
        ("periodic_entity_representative_rows", np.int32, (), 16, 0),
    )
)


def _source_id_bank(ids: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """The sorted distinct source IDs and each row's index into them (int32)."""
    if ids.size == 0 or not np.all(ids[1:] >= ids[:-1]):
        bank, rows = np.unique(ids, return_inverse=True)
        return bank, rows.astype(np.int32)
    # Already sorted: a row opens a new ID where it differs from its predecessor.
    opens = np.empty(ids.shape, dtype=bool)
    opens[0] = True
    np.not_equal(ids[1:], ids[:-1], out=opens[1:])
    rows = np.cumsum(opens, dtype=np.int32)
    rows -= 1
    return ids[opens], rows


def _tet_radius_target(value: float | None, /) -> float | None:
    """Validate the optional target before its native zero encoding."""
    if value is None:
        return None
    radius = finite_real_scalar(value, "radius_edge_bound")
    if radius <= 0.0:
        raise ValueError("radius_edge_bound must be positive or None.")
    return radius


def relocated_tet_mesh_arrays(
    arrays: TetMeshArrays,
    vertex_ids: object,
    vertex_rows: object,
    points: object,
    implicit_refusals: tuple[int, int] | None,
    /,
) -> TetMeshArrays:
    """Return the arrays of a mesh after one applied in-place relocation.

    ``arrays`` is the complete snapshot taken before
    :meth:`TetMesh3D.apply_relocation` applied the relocation of ``vertex_ids``
    (snapshot rows ``vertex_rows``) to ``points``, and ``implicit_refusals``
    the live refusal counters after it. The result equals
    :meth:`TetMesh3D.arrays` of the edited mesh: the relocation writes each
    moved vertex's point, clears its source witness and edge parameter,
    erases its ancestry row and advances the generation, and changes no other
    bank. Unchanged banks are shared with ``arrays``.
    """
    moved_ids = _tet_mesh_vector(vertex_ids, "vertex_ids", np.dtype(np.uint64))
    rows = _tet_mesh_vector(vertex_rows, "vertex_rows", np.dtype(np.int64))
    targets = _points(points, "points", 3)
    if rows.shape != moved_ids.shape or targets.shape[0] != rows.size:
        raise ValueError("vertex_rows and points must hold one row per moved vertex.")
    if (
        np.any(rows < 0)
        or np.any(rows >= arrays.vertex_ids.shape[0])
        or not np.array_equal(arrays.vertex_ids[rows], moved_ids)
    ):
        raise ValueError("vertex_rows must be the snapshot rows of vertex_ids.")
    if np.any(arrays.witness_strata[rows] != 0):
        raise ValueError("An in-place relocation moves no source-witness vertex.")
    implicit = arrays.implicit_source
    if (implicit is None) != (implicit_refusals is None):
        raise ValueError(
            "implicit_refusals must be given exactly for an implicit-source mesh."
        )
    if implicit is not None and implicit_refusals is not None:
        implicit = implicit._replace(
            transversality_refusals=implicit_refusals[0],
            distance_refusals=implicit_refusals[1],
        )

    def written(bank: np.ndarray, value: float | np.ndarray) -> np.ndarray:
        # Most moved vertices already hold the cleared record: share the bank
        # when the written rows are bit-identical (a signed zero is not).
        current = bank[rows]
        replacement = np.broadcast_to(np.asarray(value, dtype=bank.dtype), current.shape)
        if current.tobytes() == replacement.tobytes():
            return bank
        result = bank.copy()
        result[rows] = replacement
        return result

    offsets = arrays.vertex_ancestry_offsets
    ancestor_ids = arrays.vertex_ancestor_ids
    ancestor_winding = arrays.vertex_ancestor_winding
    ancestry_weights = arrays.vertex_ancestry_weights
    counts = np.diff(offsets)
    if np.any(counts[rows] != 0):
        erased = np.zeros(counts.shape, dtype=np.bool_)
        erased[rows] = True
        retained = ~np.repeat(erased, counts)
        offsets = np.zeros(offsets.shape, dtype=offsets.dtype)
        np.cumsum(np.where(erased, 0, counts), out=offsets[1:])
        ancestor_ids = ancestor_ids[retained]
        ancestor_winding = ancestor_winding[retained]
        ancestry_weights = ancestry_weights[retained]
    return arrays._replace(
        points=written(arrays.points, targets),
        witness_entities=written(arrays.witness_entities, -1),
        witness_parameters=written(arrays.witness_parameters, 0.0),
        witness_deviations=written(arrays.witness_deviations, 0.0),
        vertex_edge_witness_present=written(arrays.vertex_edge_witness_present, 0),
        vertex_edge_witness_endpoints=written(arrays.vertex_edge_witness_endpoints, 0),
        vertex_edge_parameters=written(arrays.vertex_edge_parameters, 0.0),
        generation=arrays.generation + 1,
        vertex_ancestry_offsets=offsets,
        vertex_ancestor_ids=ancestor_ids,
        vertex_ancestor_winding=ancestor_winding,
        vertex_ancestry_weights=ancestry_weights,
        implicit_source=implicit,
    )


@final
@dataclass(frozen=True, slots=True)
class TetMeshMetricAdaptation:
    """Result of :meth:`TetMesh3D.metric_adapt` (ascending native IDs).

    ``status`` is 0 complete, 1 stalled, 2 numerical stop; ``counters`` the
    splits, collapses, flips 2-3, flips 3-2, ring reconnections, relocations
    and the refusals for topology, protection, uncertainty and invalidity.
    Per live vertex: native and target ID, stencil (CSR over source rows,
    binary64 weights) and tensor. Per live cell: native and target ID, block,
    source-cell ancestry (CSR) with its ``EntityLineageKind`` value and whether
    a relocation changed it. ``vertex_relations`` are (source target ID,
    target ID) rows with their kinds for collapsed and relocated source
    vertices and ``unrepresentable_edges`` the long edges without an exact
    split point (target vertex ID pairs).
    """

    status: int
    passes: int
    counters: np.ndarray
    vertex_ids: np.ndarray
    vertex_target_ids: np.ndarray
    stencil_offsets: np.ndarray
    stencil_rows: np.ndarray
    stencil_weights: np.ndarray
    metrics: np.ndarray
    cell_ids: np.ndarray
    cell_target_ids: np.ndarray
    cell_blocks: np.ndarray
    ancestry_offsets: np.ndarray
    ancestry_ids: np.ndarray
    ancestry_kinds: np.ndarray
    cell_relocated: np.ndarray
    vertex_relations: np.ndarray
    vertex_relation_kinds: np.ndarray
    unrepresentable_edges: np.ndarray


@final
class TetMesh3D:
    """Owned native tetrahedral mesh with atomic edits and checked snapshots.

    ``affine_metric``, when supplied, is the finite nonsingular 3×3
    index-to-world linear map. It is bound before source validation; points
    and source witnesses stay in index coordinates, while size targets,
    tolerances and quality are measured in world units. Translation does
    not affect the metric. Reflections are accepted.
    """

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
        initial_vertex_sizes: object | None = None,
        vertex_ids: object | None = None,
        cell_ids: object | None = None,
        facet_ids: object | None = None,
        face_orientations: object | None = None,
        segment_ids: object | None = None,
        source: TetMeshSourceComplex | None = None,
        implicit_source: TetMeshImplicitSource | None = None,
        source_id: str | None = None,
        source_revision: str | None = None,
        source_map_id: str | None = None,
        frame_id: str | None = None,
        periodic_state: TetMeshPeriodicState | None = None,
        affine_metric: object | None = None,
    ) -> None:
        self._create(
            points,
            tetrahedra,
            tetrahedron_regions,
            faces,
            face_sources,
            segments,
            segment_sources,
            None,
            False,
            boundary_policy=boundary_policy,
            protection_radii=protection_radii,
            protected_vertices=protected_vertices,
            initial_vertex_sizes=initial_vertex_sizes,
            vertex_ids=vertex_ids,
            cell_ids=cell_ids,
            facet_ids=facet_ids,
            face_orientations=face_orientations,
            segment_ids=segment_ids,
            source=source,
            implicit_source=implicit_source,
            source_id=source_id,
            source_revision=source_revision,
            source_map_id=source_map_id,
            frame_id=frame_id,
            periodic_state=periodic_state,
            affine_metric=affine_metric,
        )

    @classmethod
    def from_recovery(
        cls,
        recovery: PlcRecovery3D,
        /,
        *,
        face_sources: object | None = None,
        segment_sources: object | None = None,
        boundary_policy: PlcBoundaryPolicy | int = PlcBoundaryPolicy.FIXED,
        protected_vertices: object | None = None,
        initial_vertex_sizes: object | None = None,
        vertex_ids: object | None = None,
        cell_ids: object | None = None,
        facet_ids: object | None = None,
        segment_ids: object | None = None,
        source: TetMeshSourceComplex | None = None,
        source_id: str | None = None,
        source_revision: str | None = None,
        source_map_id: str | None = None,
        frame_id: str | None = None,
        affine_metric: object | None = None,
        validate: bool = False,
    ) -> TetMesh3D:
        """The mesh of a successful PLC recovery, adopted from its native result.

        The state is the one the constructor builds from the recovery's
        points, tetrahedra, regions, faces, face orientations, segments and
        protection radii, with ``face_sources`` and ``segment_sources``
        (default: the recovery's) and the other arguments as there. Unless
        ``validate``, what recovery proved is adopted rather than proved
        again: its cell adjacency and, without an explicit ``source``, the
        coverage of the constrained faces and segments by their source rows
        (the mesh is then its own source bank). A recovery without a native
        result (assembled from arrays) is refused, as is a failed one.
        """
        if not isinstance(recovery, PlcRecovery3D):
            raise TypeError("recovery must be a PlcRecovery3D.")
        if recovery.native is None:
            raise ValueError(
                "The recovery holds no native result to adopt; construct the mesh from its arrays."
            )
        mesh = cls.__new__(cls)
        mesh._create(
            recovery.points,
            recovery.tetrahedra,
            recovery.tetrahedron_regions,
            recovery.faces,
            recovery.face_sources if face_sources is None else face_sources,
            recovery.segments,
            recovery.segment_sources if segment_sources is None else segment_sources,
            recovery.native,
            validate,
            boundary_policy=boundary_policy,
            protection_radii=recovery.protection_radii,
            protected_vertices=protected_vertices,
            initial_vertex_sizes=initial_vertex_sizes,
            vertex_ids=vertex_ids,
            cell_ids=cell_ids,
            facet_ids=facet_ids,
            face_orientations=recovery.face_orientations,
            segment_ids=segment_ids,
            source=source,
            implicit_source=None,
            source_id=source_id,
            source_revision=source_revision,
            source_map_id=source_map_id,
            frame_id=frame_id,
            periodic_state=None,
            affine_metric=affine_metric,
        )
        return mesh

    def _create(
        self,
        points: object,
        tetrahedra: object,
        tetrahedron_regions: object,
        faces: object,
        face_sources: object,
        segments: object,
        segment_sources: object,
        recovery: NativePlcRecovery | None,
        validate: bool,
        /,
        *,
        boundary_policy: PlcBoundaryPolicy | int,
        protection_radii: object | None,
        protected_vertices: object | None,
        initial_vertex_sizes: object | None,
        vertex_ids: object | None,
        cell_ids: object | None,
        facet_ids: object | None,
        face_orientations: object | None,
        segment_ids: object | None,
        source: TetMeshSourceComplex | None,
        implicit_source: TetMeshImplicitSource | None,
        source_id: str | None,
        source_revision: str | None,
        source_map_id: str | None,
        frame_id: str | None,
        periodic_state: TetMeshPeriodicState | None,
        affine_metric: object | None,
    ) -> None:
        # `recovery`: the banks are adopted from this native recovery, whose
        # exported arrays these are (from_recovery).
        metric_array = (
            None
            if affine_metric is None
            else _periodic_float_bank(affine_metric, "affine_metric", (3, 3))
        )
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
            or np.any(np.abs(face_orientation_bank) != 1)
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
        initial_sizes = (
            None
            if initial_vertex_sizes is None
            else _real_array(initial_vertex_sizes, "initial_vertex_sizes")
        )
        if initial_sizes is not None and (
            initial_sizes.shape != (point_array.shape[0],)
            or np.any(~np.isfinite(initial_sizes))
            or np.any(initial_sizes < 0.0)
        ):
            raise ValueError(
                "initial_vertex_sizes must be finite, nonnegative, and match points."
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
        implicit_payload = None
        if implicit_source is not None:
            if metric_array is not None or periodic_state is not None:
                raise ValueError(
                    "Implicit tetra sources do not support affine or periodic state."
                )
            implicit_payload = _tet_implicit_source_payload(implicit_source)
            if not identity_bound or (
                implicit_source.source_id,
                implicit_source.source_revision,
                implicit_source.binding_id,
                implicit_source.frame_id,
            ) != (source_id, source_revision, source_map_id, frame_id):
                raise ValueError(
                    "Implicit tetra source must match the bound source identity and frame."
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
        request.initial_vertex_sizes = _tet_mesh_pointer(initial_sizes)
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
        if recovery is not None:
            # The recovery's banks are adopted natively; the exported arrays
            # above only fixed the counts and validated the request's rows.
            request.points = None
            request.tetrahedra = None
            request.tetrahedron_regions = None
            request.faces = None
            request.face_orientations = None
            request.segments = None
            request.protection_radii = None
            status = library["phx_mc_tet_mesh_create_from_plc"](
                recovery._handle,
                ctypes.byref(request),
                None if metric_array is None else _tet_mesh_pointer(metric_array),
                1 if validate else 0,
            )
        elif implicit_payload is None:
            status = library["phx_mc_tet_mesh_create_with_affine_metric"](
                ctypes.byref(request),
                None if metric_array is None else _tet_mesh_pointer(metric_array),
            )
        else:
            implicit_view, _implicit_banks = implicit_payload
            status = library["phx_mc_tet_mesh_create_with_implicit"](
                ctypes.byref(request), ctypes.byref(implicit_view)
            )
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

    def bind_implicit_source(self, source: TetMeshImplicitSource, /) -> None:
        """Bind validated interval authority to existing exterior source groups."""
        view, _banks = _tet_implicit_source_payload(source)
        with self._lock:
            _raise_for_call(
                self._library["phx_mc_tet_mesh_bind_implicit"](
                    self._live(), ctypes.byref(view)
                ),
                "tet_mesh_bind_implicit",
            )

    def protect_facet_group_rows(self, sources: object, /) -> None:
        """Fix native source group rows and their incident edges.

        With a source complex, rows index its sorted ``face_ids`` bank, not
        the source entity IDs themselves.
        """
        values = _tet_mesh_vector(sources, "sources", np.dtype(np.int32))
        if np.any(values < 0):
            raise ValueError("Protected facet source groups must be nonnegative.")
        with self._lock:
            _raise_for_call(
                self._library["phx_mc_tet_mesh_protect_facet_sources"](
                    self._live(), values.size, _tet_mesh_pointer(values)
                ),
                "tet_mesh_protect_facet_sources",
            )

    def _live(self, /) -> ctypes.c_void_p:
        if self._closed or not self._handle.value:
            raise RuntimeError("TetMesh3D is closed.")
        return self._handle

    def _identity(self, /) -> TetMeshIdentity:
        generation = _U64()
        bound = _I32()
        lengths = (_I64 * 4)()
        base = ctypes.addressof(lengths)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_identity_sizes"](
                self._live(),
                ctypes.addressof(generation),
                ctypes.addressof(bound),
                base,
                base + 8,
                base + 16,
                base + 24,
            ),
            "tet_mesh_identity_sizes",
        )
        if bound.value not in (0, 1):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid TetMesh identity-bound status"
            )
        byte_lengths = tuple(lengths)
        if min(byte_lengths) < 0 or max(byte_lengths) >= _INTP_MAX:
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
        lengths = (_I64 * 6)()
        base = ctypes.addressof(lengths)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_periodic_identity_sizes"](
                self._live(),
                ctypes.addressof(bound),
                *range(base, base + 48, 8),
            ),
            "tet_mesh_periodic_identity_sizes",
        )
        if bound.value not in (0, 1):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid periodic identity-bound status"
            )
        byte_lengths = tuple(lengths)
        if min(byte_lengths) < 0 or max(byte_lengths) >= _INTP_MAX:
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
    def implicit_facet_refusal(self) -> TetMeshImplicitFacetRefusal | None:
        with self._lock:
            record = _TetMeshImplicitFacetRefusal()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_implicit_facet_refusal"](
                    self._live(), ctypes.byref(record)
                ),
                "tet_mesh_implicit_facet_refusal",
            )
            if not record.refused:
                return None
            identity = self._identity()
            if identity.generation != record.generation:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "implicit facet refusal identity changed",
                )
            return TetMeshImplicitFacetRefusal(
                identity,
                int(record.facet_id),
                int(record.facet_source),
                int(record.orientation),
                np.array(record.vertex_ids, dtype=np.uint64),
                np.array(record.points, dtype=np.float64).reshape(3, 3),
                int(record.transversality_refusals),
                int(record.distance_refusals),
                TetMeshEditStatus(record.status),
                (
                    np.array(record.distance_barycentric, dtype=np.float64)
                    if record.has_distance_barycentric
                    else None
                ),
            )

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

    @property
    def implicit_source(self) -> TetMeshImplicitSnapshot | None:
        """Return the owned implicit programs, source binding, and refusal counts."""
        with self._lock:
            return _tet_implicit_source_snapshot(self._library, self._live())

    @property
    def implicit_refusals(self) -> tuple[int, int] | None:
        """Return transversality and distance refusals without copying programs."""
        with self._lock:
            view = _TetImplicitSource()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_implicit_source"](
                    self._live(), ctypes.byref(view)
                ),
                "tet_mesh_implicit_source",
            )
            if view.group_count == 0:
                return None
            return int(view.transversality_refusals), int(view.distance_refusals)

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
            counts = sizes.tolist()
            if min(counts) < 0:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "negative tetrahedral snapshot size"
                )
            # One buffer holds every flat bank (in 8-byte aligned slices); the
            # returned banks are views of it.
            extents = []
            total = 0
            for _name, _dtype, _shape, count, extra, row_bytes in _TET_SNAPSHOT_BANKS:
                rows = counts[count] + extra
                extents.append((rows, total))
                total += (rows * row_bytes + 7) & ~7
            buffer = np.empty((total,), dtype=np.uint8)
            address = buffer.ctypes.data
            # The capacity fields lead the request, in snapshot-size order.
            request = _TetMeshSnapshotRequest(*counts)
            b: dict[str, np.ndarray] = {}
            for (name, dtype, shape, _count, _extra, _row_bytes), (rows, offset) in zip(
                _TET_SNAPSHOT_BANKS, extents, strict=True
            ):
                b[name] = np.ndarray((rows, *shape), dtype, buffer, offset)
                if rows:
                    setattr(request, name, address + offset)
            periodic_orbits = (_TetPeriodicOrbit * counts[7])()
            periodic_members = (_TetPeriodicVertexMember * counts[8])()
            periodic_quotient_cells = (_TetPeriodicQuotientCell * counts[9])()
            periodic_entity_maps = (_TetPeriodicEntityOrbitMap * counts[11])()
            if counts[7]:
                request.periodic_orbits = ctypes.cast(periodic_orbits, _P)
            if counts[8]:
                request.periodic_members = ctypes.cast(periodic_members, _P)
            if counts[9]:
                request.periodic_quotient_cells = ctypes.cast(periodic_quotient_cells, _P)
            if counts[11]:
                request.periodic_entity_maps = ctypes.cast(periodic_entity_maps, _P)
            generation = _U64()
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
                b["periodic_quotient_lifted_cell_ids"],
                periodic_entity_maps,
                b["periodic_entity_lifted_ids"],
                b["periodic_entity_quotient_ids"],
                b["periodic_entity_lifted_to_quotient"],
                b["periodic_entity_orientation_signs"],
                b["periodic_entity_winding_shifts"],
                b["periodic_entity_corner_permutations"],
                b["periodic_entity_representative_rows"],
            )
            vertex_size_targets = b["vertex_size_targets"]
            # NaN fails both comparisons.
            if not np.all((vertex_size_targets >= 0.0) & (vertex_size_targets < np.inf)):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "invalid TetMesh vertex size targets"
                )
            vertex_ancestry_offsets = b["vertex_ancestry_offsets"]
            cell_ancestry_offsets = b["cell_ancestry_offsets"]
            if (
                vertex_ancestry_offsets[0] != 0
                or vertex_ancestry_offsets[-1] != counts[17]
                or np.any(vertex_ancestry_offsets[1:] < vertex_ancestry_offsets[:-1])
                or not np.all(np.isfinite(b["vertex_ancestry_weights"]))
                or cell_ancestry_offsets[0] != 0
                or cell_ancestry_offsets[-1] != counts[18]
            ):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    "invalid TetMesh ancestry snapshot extents",
                )
            # Native writes cells, facets and segments in ID order with their
            # corners as vertex rows, and every cell's size target and lineage.
            source_face_ids = b["source_face_ids"]
            source_segment_ids = b["source_segment_ids"]
            face_id_bank, face_group_rows = _source_id_bank(source_face_ids)
            segment_id_bank, segment_group_rows = _source_id_bank(source_segment_ids)
            vertex_size_targets.setflags(write=False)
            b["cell_size_targets"].setflags(write=False)
            return TetMeshArrays(
                b["points"],
                vertex_size_targets,
                b["cell_vertex_rows"],
                b["cell_regions"],
                b["cell_size_targets"],
                b["facet_vertex_rows"],
                _source_group_labels(
                    b["source_face_groups"], source_face_ids, b["facet_sources"], "face"
                ),
                b["segment_vertex_rows"],
                _source_group_labels(
                    b["source_segment_groups"],
                    source_segment_ids,
                    b["segment_sources"],
                    "segment",
                ),
                b["vertex_dimensions"],
                b["protection_radii"],
                b["protected_vertices"],
                b["vertex_edge_witness_present"],
                b["vertex_edge_witness_endpoints"],
                b["vertex_edge_parameters"],
                b["vertex_ids"],
                b["cell_ids"],
                b["facet_ids"],
                b["facet_orientations"],
                b["protected_facets"],
                b["segment_ids"],
                b["protected_segments"],
                b["source_points"],
                b["source_faces"],
                face_group_rows,
                face_id_bank,
                b["source_face_tolerances"],
                b["source_segments"],
                segment_group_rows,
                segment_id_bank,
                b["source_segment_tolerances"],
                b["source_face_planes"],
                b["source_face_plane_tolerances"],
                b["witness_strata"],
                b["witness_entities"],
                b["witness_parameters"],
                b["witness_deviations"],
                identity.generation,
                identity.identity_bound,
                identity.source_id,
                identity.source_revision,
                identity.source_map_id,
                identity.frame_id,
                vertex_ancestry_offsets,
                b["vertex_ancestor_ids"],
                b["vertex_ancestor_winding"],
                b["vertex_ancestry_weights"],
                cell_ancestry_offsets,
                b["cell_ancestor_ids"],
                b["cell_ancestry_edit_kinds"],
                periodic_snapshot,
                _tet_implicit_source_snapshot(self._library, self._live()),
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

    def evaluate_quality(
        self,
        *,
        radius_edge_bound: float | None,
        min_dihedral_degrees: float,
        minimum_relative_determinant: float = 0.0,
    ) -> TetMeshQuality:
        """Evaluate current cells; ``None`` omits only the radius-edge target."""
        radius_edge = _tet_radius_target(radius_edge_bound)
        angle = finite_real_scalar(min_dihedral_degrees, "min_dihedral_degrees")
        determinant = finite_real_scalar(
            minimum_relative_determinant, "minimum_relative_determinant"
        )
        with self._lock:
            identity = self._identity()
            count = _I64()
            evaluate = self._library["phx_mc_tet_mesh_evaluate_quality"]
            arguments = (
                self._live(),
                identity.generation,
                0.0 if radius_edge is None else radius_edge,
                angle,
                determinant,
            )
            _raise_for_call(
                evaluate(*arguments, 0, ctypes.addressof(count), None, None, None, None),
                "tet_mesh_evaluate_quality",
            )
            if count.value < 0:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "negative tetrahedral quality count"
                )
            row_count = int(count.value)
            cells = np.empty((row_count,), dtype=np.uint64)
            criteria = np.empty((row_count,), dtype=np.int32)
            reasons = np.empty((row_count,), dtype=np.int32)
            values = np.empty((row_count,), dtype=np.float64)
            if row_count:
                _raise_for_call(
                    evaluate(
                        *arguments,
                        row_count,
                        ctypes.addressof(count),
                        _tet_mesh_pointer(cells),
                        _tet_mesh_pointer(criteria),
                        _tet_mesh_pointer(reasons),
                        _tet_mesh_pointer(values),
                    ),
                    "tet_mesh_evaluate_quality",
                )
                if count.value != row_count:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "tetrahedral quality count changed during evaluation",
                    )
            return TetMeshQuality(identity, cells, criteria, reasons, values)

    def evaluate_candidate_quality(
        self,
        points: object,
        /,
        *,
        vertex_ids: object,
        radius_edge_bound: float | None,
        min_dihedral_degrees: float,
        minimum_relative_determinant: float = 0.0,
    ) -> TetMeshCandidateQuality:
        """Assess a candidate; ``None`` omits only the radius-edge target."""
        candidate = _points(points, "points", 3).copy()
        if not np.all(np.isfinite(candidate)):
            raise ValueError("Candidate points must be finite.")
        candidate_ids = _tet_mesh_vector(
            vertex_ids, "vertex_ids", np.dtype(np.uint64)
        ).copy()
        radius_edge = _tet_radius_target(radius_edge_bound)
        angle = finite_real_scalar(min_dihedral_degrees, "min_dihedral_degrees")
        determinant = finite_real_scalar(
            minimum_relative_determinant, "minimum_relative_determinant"
        )
        with self._lock:
            identity = self._identity()
            snapshot = self.arrays()
            if candidate.shape != snapshot.points.shape:
                raise ValueError(
                    "Candidate points must align with the live snapshot vertices."
                )
            if not np.array_equal(candidate_ids, snapshot.vertex_ids):
                raise ValueError(
                    "Candidate vertex IDs must match the live snapshot order exactly."
                )
            candidate_id = canonical_fingerprint(
                {
                    "kind": "native-tet-quality-candidate",
                    "history": tuple(identity),
                    "vertices": array_tree_fingerprint(snapshot.vertex_ids),
                    "points": array_tree_fingerprint(candidate),
                    "radius_edge_bound": radius_edge,
                    "minimum_dihedral_degrees": angle,
                    "minimum_relative_determinant": determinant,
                }
            )
            count = _I64()
            evaluate = self._library["phx_mc_tet_mesh_evaluate_candidate_quality"]
            arguments = (
                self._live(),
                identity.generation,
                candidate.shape[0],
                _tet_mesh_pointer(candidate),
                0.0 if radius_edge is None else radius_edge,
                angle,
                determinant,
            )
            _raise_for_call(
                evaluate(*arguments, 0, ctypes.addressof(count), None, None, None, None),
                "tet_mesh_evaluate_candidate_quality",
            )
            if count.value < 0:
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "negative candidate quality count"
                )
            row_count = int(count.value)
            cells = np.empty((row_count,), dtype=np.uint64)
            criteria = np.empty((row_count,), dtype=np.int32)
            reasons = np.empty((row_count,), dtype=np.int32)
            values = np.empty((row_count,), dtype=np.float64)
            if row_count:
                _raise_for_call(
                    evaluate(
                        *arguments,
                        row_count,
                        ctypes.addressof(count),
                        _tet_mesh_pointer(cells),
                        _tet_mesh_pointer(criteria),
                        _tet_mesh_pointer(reasons),
                        _tet_mesh_pointer(values),
                    ),
                    "tet_mesh_evaluate_candidate_quality",
                )
                if count.value != row_count:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "candidate quality count changed during evaluation",
                    )
            return TetMeshCandidateQuality(
                identity, candidate_id, cells, criteria, reasons, values
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
            facet_ids = np.empty((row_count,), dtype=np.uint64)
            facet_ratios = np.empty((row_count,), dtype=np.float64)
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
                    _tet_mesh_pointer(facet_ids),
                    _tet_mesh_pointer(facet_ratios),
                ),
                "tet_mesh_unmet",
            )
            return TetMeshUnmet(
                tetrahedra,
                criteria,
                reasons,
                values,
                apex_vertex_ids,
                apex_angles,
                facet_ids,
                facet_ratios,
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

    def probe_relocations(
        self,
        expected_generation: int,
        vertex_offsets: object,
        vertex_ids: object,
        points: object,
        cell_offsets: object,
        cell_ids: object,
        *,
        source_id: str | None = None,
        source_revision: str | None = None,
        source_map_id: str | None = None,
        frame_id: str | None = None,
    ) -> tuple[TetMeshEditStatus | None, ...]:
        """Probe independent relocations against the live mesh without editing it.

        Group ``g`` moves ``vertex_ids[vertex_offsets[g]:vertex_offsets[g + 1]]``
        to the matching rows of ``points`` and declares the star cells
        ``cell_ids[cell_offsets[g]:cell_offsets[g + 1]]``. Entry ``g`` of the
        result is the status with which :meth:`apply_edit` would refuse that
        relocation alone (unchanged witnesses and protection, cleared edge
        parameter, no ancestry), or ``None`` when it may apply or the native
        probe does not decide it. A decided refusal advances the implicit
        refusal counters exactly as the refused edit would. The source identity
        is the one the edit would name: all four fields, or none for an
        unbound mesh.
        """
        generation = _tet_mesh_uint64(expected_generation, "expected_generation")
        vertex_starts = _tet_mesh_vector(
            vertex_offsets, "vertex_offsets", np.dtype(np.int64)
        )
        cell_starts = _tet_mesh_vector(cell_offsets, "cell_offsets", np.dtype(np.int64))
        moved_ids = _tet_mesh_vector(vertex_ids, "vertex_ids", np.dtype(np.uint64))
        star_ids = _tet_mesh_vector(cell_ids, "cell_ids", np.dtype(np.uint64))
        targets = np.ascontiguousarray(_points(points, "points", 3))
        if vertex_starts.size == 0 or vertex_starts.size != cell_starts.size:
            raise ValueError(
                "vertex_offsets and cell_offsets must hold one more entry than groups."
            )
        for starts, size, name in (
            (vertex_starts, moved_ids.size, "vertex_offsets"),
            (cell_starts, star_ids.size, "cell_offsets"),
        ):
            if starts[0] != 0 or starts[-1] != size or np.any(np.diff(starts) < 0):
                raise ValueError(
                    f"{name} must start at zero, not decrease and end at the bank size."
                )
        if targets.shape[0] != moved_ids.size:
            raise ValueError("points must hold one row per moved vertex.")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Relocation points must be finite.")
        group_count = vertex_starts.size - 1
        decided = np.zeros((group_count,), dtype=np.int32)
        statuses = np.zeros((group_count,), dtype=np.int32)
        identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
            source_id, source_revision, source_map_id, frame_id
        )
        with self._lock:
            _raise_for_call(
                self._library["phx_mc_tet_mesh_probe_relocations"](
                    self._live(),
                    generation,
                    identity_bound,
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[0]),
                    len(identity_bytes[0]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[1]),
                    len(identity_bytes[1]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[2]),
                    len(identity_bytes[2]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[3]),
                    len(identity_bytes[3]),
                    group_count,
                    _tet_mesh_pointer(vertex_starts),
                    _tet_mesh_pointer(moved_ids),
                    _tet_mesh_pointer(targets),
                    _tet_mesh_pointer(cell_starts),
                    _tet_mesh_pointer(star_ids),
                    _tet_mesh_pointer(decided),
                    _tet_mesh_pointer(statuses),
                ),
                "tet_mesh_probe_relocations",
            )
        return tuple(
            TetMeshEditStatus(status) if refused else None
            for refused, status in zip(decided.tolist(), statuses.tolist(), strict=True)
        )

    def apply_relocation(
        self,
        expected_generation: int,
        vertex_ids: object,
        points: object,
        cell_ids: object,
        *,
        source_id: str | None = None,
        source_revision: str | None = None,
        source_map_id: str | None = None,
        frame_id: str | None = None,
    ) -> TetMeshEditStatus | None:
        """Apply one relocation to the live mesh in place.

        The relocation moves ``vertex_ids`` together to the matching rows of
        ``points`` and declares the star cells ``cell_ids``; it is the edit of
        one :meth:`probe_relocations` group. The result is the status
        :meth:`apply_edit` would return for it: ``APPLIED`` after the moved
        vertices' points, witnesses, edge parameters and ancestry are written
        and the generation advanced (see :func:`relocated_tet_mesh_arrays`),
        or a refusal that changes no mesh state. ``None`` means the mesh is
        periodic or has exhausted its generations: nothing changed, and
        :meth:`apply_edit` decides the relocation.
        """
        generation = _tet_mesh_uint64(expected_generation, "expected_generation")
        moved_ids = _tet_mesh_vector(vertex_ids, "vertex_ids", np.dtype(np.uint64))
        star_ids = _tet_mesh_vector(cell_ids, "cell_ids", np.dtype(np.uint64))
        targets = np.ascontiguousarray(_points(points, "points", 3))
        if targets.shape[0] != moved_ids.size:
            raise ValueError("points must hold one row per moved vertex.")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Relocation points must be finite.")
        decided = _I32(0)
        edit_status = _I32(0)
        identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
            source_id, source_revision, source_map_id, frame_id
        )
        with self._lock:
            _raise_for_call(
                self._library["phx_mc_tet_mesh_apply_relocation"](
                    self._live(),
                    generation,
                    identity_bound,
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[0]),
                    len(identity_bytes[0]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[1]),
                    len(identity_bytes[1]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[2]),
                    len(identity_bytes[2]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[3]),
                    len(identity_bytes[3]),
                    moved_ids.size,
                    _tet_mesh_pointer(moved_ids),
                    _tet_mesh_pointer(targets),
                    star_ids.size,
                    _tet_mesh_pointer(star_ids),
                    ctypes.byref(decided),
                    ctypes.byref(edit_status),
                ),
                "tet_mesh_apply_relocation",
            )
        return TetMeshEditStatus(edit_status.value) if decided.value else None

    def metric_adapt(
        self,
        expected_generation: int,
        *,
        source_points: np.ndarray,
        source_metrics: np.ndarray,
        source_logarithms: np.ndarray,
        source_vertex_ids: np.ndarray,
        source_cells: np.ndarray,
        vertex_ids: np.ndarray,
        stencil_offsets: np.ndarray,
        stencil_rows: np.ndarray,
        stencil_weights: np.ndarray,
        vertex_target_ids: np.ndarray,
        cell_ids: np.ndarray,
        cell_blocks: np.ndarray,
        cell_target_ids: np.ndarray,
        ancestry_offsets: np.ndarray,
        ancestry_ids: np.ndarray,
        ancestry_kinds: np.ndarray,
        protected_edges: np.ndarray,
        next_ids: tuple[int, int, int, int],
        lower: float,
        topology_operations: bool,
        relocation: bool,
        source_id: str | None = None,
        source_revision: str | None = None,
        source_map_id: str | None = None,
        frame_id: str | None = None,
        exact_source_points: Sequence[Sequence[Fraction]] | None = None,
    ) -> TetMeshMetricAdaptation:
        """Adapt this mesh in place toward unit metric edge length.

        The source is the exact P1 complex (binary64 points, SPD tensors and
        their logarithms, target vertex IDs); every live vertex and cell
        (ascending native ID) carries its stencil, target ID, block and
        source-cell ancestry; ``next_ids`` are fresh native vertex, native
        cell, target vertex and target cell IDs. A fatal commit status
        (stale state, allocation, identifier range) raises.
        ``exact_source_points`` supplies the authored P1 vertex bank when it
        differs from its binary64 realization in ``source_points``.
        """
        generation = _tet_mesh_uint64(expected_generation, "expected_generation")
        banks = (
            np.ascontiguousarray(source_points, dtype=np.float64).reshape((-1, 3)),
            np.ascontiguousarray(source_metrics, dtype=np.float64).reshape((-1, 3, 3)),
            np.ascontiguousarray(source_logarithms, dtype=np.float64).reshape((-1, 3, 3)),
            _tet_mesh_vector(source_vertex_ids, "source_vertex_ids", np.dtype(np.uint64)),
            _tet_mesh_vector(vertex_ids, "vertex_ids", np.dtype(np.uint64)),
            _tet_mesh_vector(stencil_offsets, "stencil_offsets", np.dtype(np.int64)),
            _tet_mesh_vector(stencil_rows, "stencil_rows", np.dtype(np.int32)),
            np.ascontiguousarray(stencil_weights, dtype=np.float64).reshape((-1,)),
            _tet_mesh_vector(vertex_target_ids, "vertex_target_ids", np.dtype(np.uint64)),
            _tet_mesh_vector(cell_ids, "cell_ids", np.dtype(np.uint64)),
            _tet_mesh_vector(cell_blocks, "cell_blocks", np.dtype(np.int32)),
            _tet_mesh_vector(cell_target_ids, "cell_target_ids", np.dtype(np.uint64)),
            _tet_mesh_vector(ancestry_offsets, "ancestry_offsets", np.dtype(np.int64)),
            _tet_mesh_vector(ancestry_ids, "ancestry_ids", np.dtype(np.uint64)),
            _tet_mesh_vector(ancestry_kinds, "ancestry_kinds", np.dtype(np.int32)),
            np.ascontiguousarray(protected_edges, dtype=np.uint64).reshape((-1, 2)),
            np.ascontiguousarray(source_cells, dtype=np.int64).reshape((-1, 4)),
        )
        sources = banks[3].size
        if (
            banks[0].shape[0] != sources
            or banks[1].shape[0] != sources
            or banks[2].shape[0] != sources
        ):
            raise ValueError("The source banks must hold one row per source vertex.")
        vertices, cells = banks[4].size, banks[9].size
        if banks[8].size != vertices:
            raise ValueError("vertex_target_ids must hold one entry per vertex.")
        if any(banks[index].size != cells for index in (10, 11, 14)):
            raise ValueError(
                "cell_blocks, cell_target_ids and ancestry_kinds must hold one entry per cell."
            )
        # The native request has owner counts and raw pointers, not the
        # lengths of the Python buffers. Check every extent before the call.
        for offsets, owners, counts, name in (
            (banks[5], vertices, (banks[6].size, banks[7].size), "stencil_offsets"),
            (banks[12], cells, (banks[13].size,), "ancestry_offsets"),
        ):
            if offsets.size != owners + 1:
                raise ValueError(f"{name} must hold one more entry than its owners.")
            if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"{name} must start at zero and be nondecreasing.")
            if any(offsets[-1] != count for count in counts):
                raise ValueError(f"{name} must end at the length of each values bank.")
        exact_source = None
        if exact_source_points is not None:
            exact_source = MetricSource(
                banks[0], exact_source_points, banks[16], banks[1], banks[2]
            )
            if exact_source._library is not self._library:
                raise ValueError(
                    "The authored source and mesh must use the same meshcore library."
                )
        request = _MetricAdaptRequest(
            sources,
            banks[0].ctypes.data,
            banks[1].ctypes.data,
            banks[2].ctypes.data,
            banks[3].ctypes.data,
            banks[16].shape[0],
            banks[16].ctypes.data,
            vertices,
            banks[4].ctypes.data,
            banks[5].ctypes.data,
            banks[6].ctypes.data,
            banks[7].ctypes.data,
            banks[8].ctypes.data,
            cells,
            banks[9].ctypes.data,
            banks[10].ctypes.data,
            banks[11].ctypes.data,
            banks[12].ctypes.data,
            banks[13].ctypes.data,
            banks[14].ctypes.data,
            banks[15].shape[0],
            banks[15].ctypes.data,
            *(_tet_mesh_uint64(value, "next_ids") for value in next_ids),
            lower,
            topology_operations,
            relocation,
            None if exact_source is None else exact_source._handle,
        )
        edit_status = _I32(0)
        handle = _P()
        identity_bound, identity_bytes, identity_buffers = _tet_mesh_identity_payload(
            source_id, source_revision, source_map_id, frame_id
        )
        with self._lock:
            _raise_for_call(
                self._library["phx_mc_tet_mesh_metric_adapt"](
                    self._live(),
                    generation,
                    identity_bound,
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[0]),
                    len(identity_bytes[0]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[1]),
                    len(identity_bytes[1]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[2]),
                    len(identity_bytes[2]),
                    _tet_mesh_identity_pointer(identity_bound, identity_buffers[3]),
                    len(identity_bytes[3]),
                    ctypes.byref(request),
                    ctypes.byref(edit_status),
                    ctypes.byref(handle),
                ),
                "tet_mesh_metric_adapt",
            )
        status = TetMeshEditStatus(edit_status.value)
        if status is not TetMeshEditStatus.APPLIED or not handle.value:
            raise RuntimeError(f"Native metric adaptation failed with {status.name}.")
        result = self._metric_adaptation(handle)
        self._library["phx_mc_metric_adaptation_free"](handle)
        return result

    def _metric_adaptation(self, handle: ctypes.c_void_p, /) -> TetMeshMetricAdaptation:
        sizes = np.zeros(6, dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_metric_adaptation_sizes"](handle, sizes.ctypes.data),
            "metric_adaptation_sizes",
        )
        vertices, terms, cells, ancestors, relations, unrepresentable = sizes.tolist()
        status = np.zeros(1, dtype=np.int32)
        passes = np.zeros(1, dtype=np.int64)
        counters = np.zeros(10, dtype=np.int64)
        out = {
            "vertex_ids": np.zeros(vertices, dtype=np.uint64),
            "vertex_target_ids": np.zeros(vertices, dtype=np.uint64),
            "stencil_offsets": np.zeros(vertices + 1, dtype=np.int64),
            "stencil_rows": np.zeros(terms, dtype=np.int32),
            "stencil_weights": np.zeros(terms, dtype=np.float64),
            "metrics": np.zeros((vertices, 3, 3), dtype=np.float64),
            "cell_ids": np.zeros(cells, dtype=np.uint64),
            "cell_target_ids": np.zeros(cells, dtype=np.uint64),
            "cell_blocks": np.zeros(cells, dtype=np.int32),
            "ancestry_offsets": np.zeros(cells + 1, dtype=np.int64),
            "ancestry_ids": np.zeros(ancestors, dtype=np.uint64),
            "ancestry_kinds": np.zeros(cells, dtype=np.int32),
            "cell_relocated": np.zeros(cells, dtype=np.int8),
            "vertex_relation_sources": np.zeros(relations, dtype=np.uint64),
            "vertex_relation_targets": np.zeros(relations, dtype=np.uint64),
            "vertex_relation_kinds": np.zeros(relations, dtype=np.int32),
            "unrepresentable_edges": np.zeros((unrepresentable, 2), dtype=np.uint64),
        }
        banks = _MetricAdaptationOutput(
            status.ctypes.data,
            passes.ctypes.data,
            counters.ctypes.data,
            *(array.ctypes.data for array in out.values()),
        )
        _raise_for_call(
            self._library["phx_mc_metric_adaptation_read"](handle, ctypes.byref(banks)),
            "metric_adaptation_read",
        )
        return TetMeshMetricAdaptation(
            int(status[0]),
            int(passes[0]),
            counters,
            out["vertex_ids"],
            out["vertex_target_ids"],
            out["stencil_offsets"],
            out["stencil_rows"],
            out["stencil_weights"],
            out["metrics"],
            out["cell_ids"],
            out["cell_target_ids"],
            out["cell_blocks"],
            out["ancestry_offsets"],
            out["ancestry_ids"],
            out["ancestry_kinds"],
            out["cell_relocated"],
            np.stack(
                (out["vertex_relation_sources"], out["vertex_relation_targets"]), axis=1
            ),
            out["vertex_relation_kinds"],
            out["unrepresentable_edges"],
        )

    def apply_edit(self, edit: TetMeshEdit) -> TetMeshEditStatus:
        with self._lock:
            _raise_for_call(
                self._library["phx_mc_tet_mesh_clear_implicit_facet_refusal"](
                    self._live()
                ),
                "tet_mesh_clear_implicit_facet_refusal",
            )
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
        # Each pair names its own removed vertex and survivor: the C request
        # takes them as two dense columns of ``collapse_pairs``.
        collapse_pair_removed = np.ascontiguousarray(
            collapse_pairs[:, 0], dtype=np.uint64
        )
        collapse_survivors = np.ascontiguousarray(collapse_pairs[:, 1], dtype=np.uint64)
        collapse_removed = (
            collapse_pair_removed
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
            ("collapse_removed_ids", collapse_pair_removed),
            ("collapse_survivor_ids", collapse_survivors),
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
            # A refusal the exact local probe decides needs no staged commit.
            decided = _I32()
            edit_status = _I32()
            _raise_for_call(
                self._library["phx_mc_tet_mesh_probe_edit"](
                    self._live(),
                    ctypes.byref(request),
                    ctypes.addressof(decided),
                    ctypes.addressof(edit_status),
                ),
                "tet_mesh_probe_edit",
            )
            if decided.value:
                return TetMeshEditStatus(edit_status.value)
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
        self, *, radius_edge_bound: float | None, sizes: object | None = None
    ) -> TetMeshRun:
        """Circumcenter refinement to the radius-edge bound and size targets.

        ``radius_edge_bound=None`` requests no radius-edge target.

        ``sizes`` (one circumradius target per current vertex, 0 for none) becomes
        the mesh's retained size field; ``None`` keeps the retained field. On a
        fixed boundary with a target at every vertex, the interior is first
        seeded with the body-centered cubic lattice whose cells meet the
        smallest target (where the target is within 2 % of it, at least half a
        lattice spacing from the boundary), then refined as above.
        """
        radius_edge = _tet_radius_target(radius_edge_bound)
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
                    0.0 if radius_edge is None else radius_edge,
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
        radius_edge_bound: float | None,
        min_dihedral_degrees: float,
        minimum_relative_determinant: float = 0.0,
        sizes: object | None = None,
        boundary_collapse_quality: bool = False,
        original_vertex_ids: object | None = None,
    ) -> TetMeshRun:
        """Quality improvement gated by the determinant floor and the size field.

        The determinant floor is ``minimum_relative_determinant`` or 2^-22,
        whichever is larger: below 2^-22 a cell's binary64 metric determinant
        det(E^T E) cannot be certified positive, so such cells are repaired
        (or reported) whatever floor is requested. No edit creates a cell whose
        circumradius exceeds its size target by more than the largest excess
        among the cells it removes (none where they were within target), except
        one removing a degenerate cell. ``sizes`` follows :meth:`refine`
        (``None`` keeps the retained field, e.g. refinement's). The run reports
        its score quantum. ``radius_edge_bound=None`` requests no radius-edge target.
        """
        radius_edge = _tet_radius_target(radius_edge_bound)
        minimum_angle = finite_real_scalar(min_dihedral_degrees, "min_dihedral_degrees")
        determinant_floor = finite_real_scalar(
            minimum_relative_determinant, "minimum_relative_determinant"
        )
        if not isinstance(boundary_collapse_quality, bool):
            raise TypeError("boundary_collapse_quality must be bool.")
        anchors = _tet_mesh_vector(
            np.empty((0,), dtype=np.uint64)
            if original_vertex_ids is None
            else original_vertex_ids,
            "original_vertex_ids",
            np.dtype(np.uint64),
        )
        if boundary_collapse_quality != bool(anchors.size):
            raise ValueError(
                "Boundary collapse quality requires exactly one nonempty original vertex bank."
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
                    0.0 if radius_edge is None else radius_edge,
                    minimum_angle,
                    determinant_floor,
                    int(boundary_collapse_quality),
                    anchors.size,
                    _tet_mesh_pointer(anchors),
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


# ------------------------------------------- deterministic metric measurements
#
# Binary64 values of the native deterministic kernels: the same bits for the
# same input on every platform. They order and screen floating-point
# proposals; every acceptance that rests on them is an exact comparison of
# these values, and no validity decision does.


def _metric_tensors(value: np.ndarray, name: str, /) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"{name} must have shape (n, 3, 3).")
    return array


def _metric_rows(value: np.ndarray, name: str, width: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array.")
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    return np.ascontiguousarray(array, dtype=np.int64)


def _metric_point_bank(
    points: np.ndarray, metrics: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.ascontiguousarray(points, dtype=np.float64)
    tensors = _metric_tensors(metrics, "metrics")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")
    if tensors.shape[0] != coordinates.shape[0]:
        raise ValueError("metrics must have one tensor per point.")
    return coordinates, tensors


def metric_logarithms(metrics: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """``log M`` of ``(n, 3, 3)`` metrics and whether each tensor is a metric.

    A tensor whose eigenvalues are not all positive has no logarithm: its row
    is zero and its flag is false.
    """
    tensors = _metric_tensors(metrics, "metrics")
    logarithms = np.empty_like(tensors)
    valid = np.empty((tensors.shape[0],), dtype=np.int8)
    status = load_meshcore()["phx_mc_metric_logarithms"](
        tensors.shape[0], tensors.ctypes.data, logarithms.ctypes.data, valid.ctypes.data
    )
    _raise_for_call(status, "metric_logarithms")
    return logarithms, valid.astype(np.bool_)


def metric_interpolations(
    offsets: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
    source_metrics: np.ndarray,
    source_logarithms: np.ndarray | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-Euclidean means ``exp(sum_k w_k log M_k)`` over source supports.

    Row ``r`` averages the source rows ``support[offsets[r]:offsets[r + 1]]``
    with the weights at the same positions, summed in that order. A support
    whose tensors are all equal returns its first tensor unchanged, once that
    tensor is a metric. ``source_logarithms`` are the logarithms of
    ``source_metrics``, or ``None`` to evaluate them per row; supplied
    logarithms are taken as those of the bank, whose tensors are then not
    examined again. The flag of a row is false, and the row zero, when a
    support tensor has no logarithm (it is not a metric) or the mean is not
    finite.
    """
    sources = _metric_tensors(source_metrics, "source_metrics")
    row_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    rows = np.ascontiguousarray(support, dtype=np.int64)
    coefficients = np.ascontiguousarray(weights, dtype=np.float64)
    if row_offsets.ndim != 1 or row_offsets.shape[0] == 0:
        raise ValueError("offsets must have shape (rows + 1,).")
    if rows.shape != coefficients.shape or rows.shape != (int(row_offsets[-1]),):
        raise ValueError("support and weights must have one entry per offset position.")
    logarithms = None
    if source_logarithms is not None:
        logarithms = _metric_tensors(source_logarithms, "source_logarithms")
        if logarithms.shape != sources.shape:
            raise ValueError("source_logarithms must match source_metrics.")
    count = row_offsets.shape[0] - 1
    metrics = np.empty((count, 3, 3), dtype=np.float64)
    valid = np.empty((count,), dtype=np.int8)
    status = load_meshcore()["phx_mc_metric_interpolate"](
        count,
        row_offsets.ctypes.data,
        rows.ctypes.data,
        coefficients.ctypes.data,
        sources.shape[0],
        sources.ctypes.data,
        None if logarithms is None else logarithms.ctypes.data,
        metrics.ctypes.data,
        valid.ctypes.data,
    )
    _raise_for_call(status, "metric_interpolations")
    return metrics, valid.astype(np.bool_)


def metric_lengths(
    points: np.ndarray, metrics: np.ndarray, edges: np.ndarray, /
) -> np.ndarray:
    """Riemannian lengths of ``(n, 2)`` edges of point rows.

    The log-mean ``(l_p - l_q) / log(l_p / l_q)`` of the two endpoint lengths
    ``sqrt(d^T M d)`` (Alauzet 2010), and their arithmetic mean when they
    agree within ``1e-6``. NaN where an endpoint tensor gives a negative form.
    """
    coordinates, tensors = _metric_point_bank(points, metrics)
    rows = _metric_rows(edges, "edges", 2)
    lengths = np.empty((rows.shape[0],), dtype=np.float64)
    status = load_meshcore()["phx_mc_metric_lengths"](
        rows.shape[0],
        rows.ctypes.data,
        coordinates.shape[0],
        coordinates.ctypes.data,
        tensors.ctypes.data,
        lengths.ctypes.data,
    )
    _raise_for_call(status, "metric_lengths")
    return lengths


def metric_band_quanta(lengths: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Quanta by which each length leaves the band ``[1/sqrt(2), sqrt(2)]``.

    ``(above, below)`` as float arrays of the shape of ``lengths``: the
    violation ``|log(length / band edge)|`` in quanta of ``2**-10`` rounded
    up, at least one for a length outside the band and zero inside it. A zero
    or infinite length has an infinite count. A NaN length was not measured
    and has no count: both are NaN.
    """
    values = np.ascontiguousarray(lengths, dtype=np.float64)
    above = np.empty_like(values)
    below = np.empty_like(values)
    status = load_meshcore()["phx_mc_metric_band_quanta"](
        values.size, values.ctypes.data, above.ctypes.data, below.ctypes.data
    )
    _raise_for_call(status, "metric_band_quanta")
    return above, below


def metric_relative_determinants(corners: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Determinants and relative determinants of ``(n, 4, 3)`` cell corners.

    ``det(p1 - p0, p2 - p0, p3 - p0)``, and its magnitude over the largest
    product of the three edge lengths at a corner (zero when either is not
    finite or the product vanishes). The evaluated determinant is within
    ``50 * 2**-53`` times that product of the exact one.
    """
    values = np.ascontiguousarray(corners, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 3):
        raise ValueError("corners must have shape (n, 4, 3).")
    determinants = np.empty((values.shape[0],), dtype=np.float64)
    relative = np.empty((values.shape[0],), dtype=np.float64)
    status = load_meshcore()["phx_mc_metric_relative_determinants"](
        values.shape[0],
        values.ctypes.data,
        determinants.ctypes.data,
        relative.ctypes.data,
    )
    _raise_for_call(status, "metric_relative_determinants")
    return determinants, relative


def metric_tetra_quality(
    points: np.ndarray, metrics: np.ndarray, cells: np.ndarray, /
) -> np.ndarray:
    """Metric mean-ratio quality of ``(n, 4)`` cells of point rows.

    ``12 (3 V)^(2/3) / sum l^2`` with the six metric edge lengths ``l`` and
    the volume ``V`` in the log-Euclidean barycentric cell metric, whose
    determinant is ``exp(mean_k log det M_k)``. Zero unless the metric volume
    and the sum are both positive; NaN when both are infinite.
    """
    coordinates, tensors = _metric_point_bank(points, metrics)
    rows = _metric_rows(cells, "cells", 4)
    quality = np.empty((rows.shape[0],), dtype=np.float64)
    status = load_meshcore()["phx_mc_metric_tetra_quality"](
        rows.shape[0],
        rows.ctypes.data,
        coordinates.shape[0],
        coordinates.ctypes.data,
        tensors.ctypes.data,
        quality.ctypes.data,
    )
    _raise_for_call(status, "metric_tetra_quality")
    return quality


def metric_tensor_properties(tensors: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Whether ``(n, 3, 3)`` tensors are symmetric and positive definite.

    The gate of a metric field. With ``lambda`` the eigenvalues of the
    symmetric part ``(M + M^T) / 2`` and ``tol = 64 eps max |lambda|``, a
    tensor is symmetric when ``max |M - M^T| <= tol`` and positive definite
    when it is symmetric and ``min lambda > tol``. Neither holds for a
    non-finite entry. A deterministic kernel verdict: the same on every
    platform.
    """
    values = _metric_tensors(tensors, "tensors")
    symmetric = np.empty((values.shape[0],), dtype=np.int8)
    positive_definite = np.empty((values.shape[0],), dtype=np.int8)
    status = load_meshcore()["phx_mc_metric_tensor_properties"](
        values.shape[0],
        values.ctypes.data,
        symmetric.ctypes.data,
        positive_definite.ctypes.data,
    )
    _raise_for_call(status, "metric_tensor_properties")
    return symmetric.astype(np.bool_), positive_definite.astype(np.bool_)


def metric_congruence_defects(
    linear: np.ndarray, metrics: np.ndarray, references: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Defects of the congruences ``A^T M A = R`` of ``(n, 3, 3)`` arrays.

    The form in which a metric ``M`` at the image of a linear map ``A`` agrees
    with the metric ``R`` at its source. Returns the infinity norms of
    ``A^T M A - R`` and, as their scale, the larger of the infinity norms of
    ``A^T M A`` and ``R``. Deterministic kernel values: the same bits on
    every platform.
    """
    maps = _metric_tensors(linear, "linear")
    tensors = _metric_tensors(metrics, "metrics")
    sources = _metric_tensors(references, "references")
    if tensors.shape != maps.shape or sources.shape != maps.shape:
        raise ValueError("linear, metrics and references must have the same shape.")
    errors = np.empty((maps.shape[0],), dtype=np.float64)
    scales = np.empty((maps.shape[0],), dtype=np.float64)
    status = load_meshcore()["phx_mc_metric_congruence_defects"](
        maps.shape[0],
        maps.ctypes.data,
        tensors.ctypes.data,
        sources.ctypes.data,
        errors.ctypes.data,
        scales.ctypes.data,
    )
    _raise_for_call(status, "metric_congruence_defects")
    return errors, scales


def metric_midpoints(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_metrics: np.ndarray,
    second_metrics: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Metric midpoints of ``n`` segments: parameters ``(n,)`` and points ``(n, 3)``.

    The parameter ``t`` of the point at half the metric length of the segment
    when the size varies geometrically between the metrics at its ends,
    ``log1p((r - 1) / 2) / log1p(r - 1)`` with ``r`` the ratio of the end
    lengths, and the point ``first + t (second - first)``. ``t`` is ``1/2``
    for equal end lengths and where an end length is not positive and finite.
    A deterministic kernel value: the same bits on every platform.
    """
    first = np.ascontiguousarray(first_points, dtype=np.float64)
    second = np.ascontiguousarray(second_points, dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 3 or second.shape != first.shape:
        raise ValueError("first_points and second_points must both have shape (n, 3).")
    first_tensors = _metric_tensors(first_metrics, "first_metrics")
    second_tensors = _metric_tensors(second_metrics, "second_metrics")
    if (
        first_tensors.shape[0] != first.shape[0]
        or second_tensors.shape != first_tensors.shape
    ):
        raise ValueError("Metrics must have one tensor per segment end.")
    parameters = np.empty((first.shape[0],), dtype=np.float64)
    points = np.empty_like(first)
    status = load_meshcore()["phx_mc_metric_midpoints"](
        first.shape[0],
        first.ctypes.data,
        second.ctypes.data,
        first_tensors.ctypes.data,
        second_tensors.ctypes.data,
        parameters.ctypes.data,
        points.ctypes.data,
    )
    _raise_for_call(status, "metric_midpoints")
    return parameters, points


def metric_survivor_goals(
    points: np.ndarray, metrics: np.ndarray, corners: np.ndarray | None, /
) -> tuple[np.ndarray, np.ndarray]:
    """The three goals of a collapse survivor, in the order they are tried.

    ``points`` hold the survivor, the removed vertex and then the merged
    link; ``metrics`` the ``(2, 3, 3)`` tensors of the first two. The goals
    ``(3, 3)`` are the metric midpoint of the collapsed edge, the point
    halfway to it, and the center of the smallest ball that holds the link in
    the log-Euclidean mean of the two tensors (it minimizes the longest
    survivor spoke). With ``corners`` (two or three points) each goal is
    projected onto their affine hull, the survivor's carrier. The flags
    ``(3,)`` are false, with a zero goal, where a goal does not exist: no link
    vertex, a tensor that is not positive definite, or a value that is not
    finite. Deterministic kernel values: the same bits on every platform.
    """
    coordinates = np.ascontiguousarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] < 2:
        raise ValueError("points must have shape (n, 3) with n >= 2.")
    tensors = _metric_tensors(metrics, "metrics")
    if tensors.shape[0] != 2:
        raise ValueError(
            "metrics must hold the tensors of the survivor and the removed vertex."
        )
    hull = None
    if corners is not None:
        hull = np.ascontiguousarray(corners, dtype=np.float64)
        if hull.ndim != 2 or hull.shape[1] != 3 or hull.shape[0] not in (2, 3):
            raise ValueError("corners must have shape (2, 3) or (3, 3).")
    goals = np.empty((3, 3), dtype=np.float64)
    valid = np.empty((3,), dtype=np.int8)
    status = load_meshcore()["phx_mc_metric_survivor_goals"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        tensors.ctypes.data,
        0 if hull is None else hull.shape[0],
        None if hull is None else hull.ctypes.data,
        goals.ctypes.data,
        valid.ctypes.data,
    )
    _raise_for_call(status, "metric_survivor_goals")
    return goals, valid.astype(np.bool_)


def _angle_map(entry: str, name: str, values: np.ndarray, /) -> np.ndarray:
    arguments = _real_array(values, "values")
    results = np.empty_like(arguments)
    status = load_meshcore()[entry](
        arguments.size, arguments.ctypes.data, results.ctypes.data
    )
    _raise_for_call(status, name)
    return results


def angle_arcsine(values: np.ndarray, /) -> np.ndarray:
    """``asin`` of values in ``[-1, 1]``, the same binary64 result on every
    platform (within 7 units in the last place); NaN outside the interval.

    The value the native meshers use. It is not correctly rounded and no
    exact decision may rest on it.
    """
    return _angle_map("phx_mc_angle_arcsine", "angle_arcsine", values)


def angle_arccosine(values: np.ndarray, /) -> np.ndarray:
    """``acos`` of values in ``[-1, 1]``, the same binary64 result on every
    platform (within 5 units in the last place); NaN outside the interval."""
    return _angle_map("phx_mc_angle_arccosine", "angle_arccosine", values)


def angle_sine(values: np.ndarray, /) -> np.ndarray:
    """``sin`` of finite values, the same binary64 result on every platform:
    the lower endpoint of the native interval enclosure, at most the true
    value and at most four binary64 values below it; NaN otherwise."""
    return _angle_map("phx_mc_angle_sine", "angle_sine", values)


def angle_cosine(values: np.ndarray, /) -> np.ndarray:
    """``cos`` of finite values, as :func:`angle_sine`."""
    return _angle_map("phx_mc_angle_cosine", "angle_cosine", values)


def angle_between(across: np.ndarray, along: np.ndarray, /) -> np.ndarray:
    """The angle in ``[0, pi]`` of each direction ``(along, across)`` with
    ``across >= 0``: ``atan2(across, along)``, the same binary64 result on
    every platform (within 7 units in the last place).

    Between two vectors these are the length of their cross product and their
    dot product. The quadrant follows the sign of ``along`` exactly; a
    negative ``across`` and a value that is not finite give NaN.
    """
    cross = _real_array(across, "across")
    dot = _real_array(along, "along")
    if cross.shape != dot.shape:
        raise ValueError("across and along must have one shape.")
    results = np.empty_like(cross)
    status = load_meshcore()["phx_mc_angle_between"](
        cross.size, cross.ctypes.data, dot.ctypes.data, results.ctypes.data
    )
    _raise_for_call(status, "angle_between")
    return results


BVH_BUILD_MEDIAN = 0
BVH_BUILD_SAH = 1
BVH_BUILD_MORTON = 2


def bvh_build(
    kind: int,
    lower: np.ndarray,
    upper: np.ndarray,
    codes: np.ndarray | None,
    leaf_size: int,
    sah_bins: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Level-ordered BVH topology over item boxes (``phx_mc_bvh_build``).

    Returns the item order and, per node in breadth-first level order, its
    item range ``[start, stop)`` and children (``-1`` at leaves), with the
    node offsets of every level. ``codes`` (uint64 per item) are required
    for ``BVH_BUILD_MORTON`` and unused otherwise.
    """
    low = np.ascontiguousarray(lower, dtype=np.float64)
    high = np.ascontiguousarray(upper, dtype=np.float64)
    if low.ndim != 2 or high.shape != low.shape:
        raise ValueError("lower and upper must share shape (items, dimension).")
    count, dimension = low.shape
    keys = None if codes is None else np.ascontiguousarray(codes, dtype=np.uint64)
    if keys is not None and keys.shape != (count,):
        raise ValueError("codes must hold one value per item.")
    capacity = max(2 * count - 1, 1)
    order = np.empty((count,), dtype=np.int64)
    start, stop, left, right = (np.empty((capacity,), dtype=np.int64) for _ in range(4))
    level_offsets = np.empty((capacity + 1,), dtype=np.int64)
    node_count, level_count = _I64(), _I64()
    status = load_meshcore()["phx_mc_bvh_build"](
        kind,
        count,
        dimension,
        low.ctypes.data,
        high.ctypes.data,
        None if keys is None else keys.ctypes.data,
        leaf_size,
        sah_bins,
        order.ctypes.data,
        start.ctypes.data,
        stop.ctypes.data,
        left.ctypes.data,
        right.ctypes.data,
        level_offsets.ctypes.data,
        ctypes.addressof(node_count),
        ctypes.addressof(level_count),
    )
    _raise_for_call(status, "bvh_build")
    nodes = int(node_count.value)
    return (
        order,
        start[:nodes],
        stop[:nodes],
        left[:nodes],
        right[:nodes],
        level_offsets[: int(level_count.value) + 1],
    )


def angle_arctangent2(ordinates: np.ndarray, abscissas: np.ndarray, /) -> np.ndarray:
    """``atan2(ordinates, abscissas)`` in ``[-pi, pi]``, the same binary64
    result on every platform (within 7 units in the last place).

    :func:`angle_between` of the magnitude of the ordinate, with the
    ordinate's sign. Zeros follow IEEE 754: the result has the sign of the
    ordinate, also of a negative zero, and a zero ordinate gives a zero for an
    abscissa that is positive or ``+0`` and pi for one that is negative or
    ``-0``. A value that is not finite gives NaN.
    """
    first = _real_array(ordinates, "ordinates")
    second = _real_array(abscissas, "abscissas")
    if first.shape != second.shape:
        raise ValueError("ordinates and abscissas must have one shape.")
    results = np.empty_like(first)
    status = load_meshcore()["phx_mc_angle_arctangent2"](
        first.size, first.ctypes.data, second.ctypes.data, results.ctypes.data
    )
    _raise_for_call(status, "angle_arctangent2")
    return results


def _quality_cells(
    points: np.ndarray, rows: np.ndarray, width: int, /
) -> tuple[np.ndarray, np.ndarray]:
    corners = _real_array(points, "points")
    table = np.ascontiguousarray(rows, dtype=np.int64)
    if corners.ndim != 3 or corners.shape[2] != 3 or corners.shape[1] == 0:
        raise ValueError("points must have shape (cells, corners, 3).")
    if table.ndim != 2 or table.shape[1] != width:
        raise ValueError(f"Corner index rows must have shape (rows, {width}).")
    return corners, table


def quality_dihedral_angles(points: np.ndarray, samples: np.ndarray, /) -> np.ndarray:
    """Dihedral angles ``(cells, samples)`` of ``points`` ``(cells, corners,
    3)`` with the host quality formula's rounded binary64 operations: per
    sample row ``(origin, end, first, second)`` of corner indices, the angle
    about the edge ``(origin, end)`` from the first to the second corner by
    :func:`angle_arctangent2`, plus ``2 pi`` when negative.
    """
    corners, rows = _quality_cells(points, samples, 4)
    angles = np.empty((corners.shape[0], rows.shape[0]), dtype=np.float64)
    status = load_meshcore()["phx_mc_quality_dihedral_angles"](
        corners.shape[0],
        corners.shape[1],
        corners.ctypes.data,
        rows.shape[0],
        rows.ctypes.data,
        angles.ctypes.data,
    )
    _raise_for_call(status, "quality_dihedral_angles")
    return angles


def quality_solid_corners(
    points: np.ndarray,
    frames: np.ndarray,
    ideal_inverse: np.ndarray,
    ideal_scaled: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Corner measures ``(cells, rows)`` of ``points`` ``(cells, corners, 3)``
    with the host quality formula's rounded binary64 operations: per frame
    row ``(apex, n0, n1, n2)``, the frame determinant, the scaled Jacobian,
    the shape ratio and the condition number against the row's ideal inverse
    ``(3, 3)`` and ideal scaled Jacobian (``phx_mc_quality_solid_corners``).
    """
    corners, rows = _quality_cells(points, frames, 4)
    inverse = np.ascontiguousarray(ideal_inverse, dtype=np.float64)
    scale = np.ascontiguousarray(ideal_scaled, dtype=np.float64)
    if inverse.shape != (rows.shape[0], 3, 3) or scale.shape != (rows.shape[0],):
        raise ValueError("Ideal frames must align with the frame rows.")
    results = tuple(
        np.empty((corners.shape[0], rows.shape[0]), dtype=np.float64) for _ in range(4)
    )
    status = load_meshcore()["phx_mc_quality_solid_corners"](
        corners.shape[0],
        corners.shape[1],
        corners.ctypes.data,
        rows.shape[0],
        rows.ctypes.data,
        inverse.ctypes.data,
        scale.ctypes.data,
        *(result.ctypes.data for result in results),
    )
    _raise_for_call(status, "quality_solid_corners")
    return results[0], results[1], results[2], results[3]


def metric_hull_frame(corners: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis and projector of the affine hull of 2 or 3 corners.

    The basis ``(rank, 3)`` spans the edges from the first corner
    (Gram-Schmidt in corner order; dependent edges add no row) and the
    ``(3, 3)`` projector is the orthogonal projector onto it. Deterministic
    kernel values: the same bits on every platform.
    """
    hull = np.ascontiguousarray(corners, dtype=np.float64)
    if hull.ndim != 2 or hull.shape[1] != 3 or hull.shape[0] not in (2, 3):
        raise ValueError("corners must have shape (2, 3) or (3, 3).")
    basis = np.empty((2, 3), dtype=np.float64)
    projector = np.empty((3, 3), dtype=np.float64)
    rank = _I32(0)
    status = load_meshcore()["phx_mc_metric_hull_frame"](
        hull.shape[0],
        hull.ctypes.data,
        basis.ctypes.data,
        projector.ctypes.data,
        ctypes.byref(rank),
    )
    _raise_for_call(status, "metric_hull_frame")
    return basis[: rank.value].copy(), projector


def metric_project_goals(
    goals: np.ndarray, origin: np.ndarray, projector: np.ndarray, /
) -> np.ndarray:
    """``origin + projector (goal - origin)`` for ``(n, 3)`` goals.

    Each coordinate is a fixed-order sum of three products: the same bits on
    every platform.
    """
    rows = np.ascontiguousarray(goals, dtype=np.float64)
    start = np.ascontiguousarray(origin, dtype=np.float64)
    matrix = np.ascontiguousarray(projector, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[1] != 3
        or start.shape != (3,)
        or matrix.shape != (3, 3)
    ):
        raise ValueError(
            "goals, origin and projector must have shapes (n, 3), (3,), (3, 3)."
        )
    projected = np.empty_like(rows)
    status = load_meshcore()["phx_mc_metric_project_goals"](
        rows.shape[0],
        rows.ctypes.data,
        start.ctypes.data,
        matrix.ctypes.data,
        projected.ctypes.data,
    )
    _raise_for_call(status, "metric_project_goals")
    return projected


def metric_relocation_pulls(
    points: np.ndarray,
    metric: np.ndarray,
    lengths: np.ndarray,
    measured: np.ndarray,
    /,
    *,
    deep: bool,
    interior: bool,
    basis: np.ndarray,
    projector: np.ndarray | None,
    jacobian: np.ndarray | None,
) -> np.ndarray:
    """Pulls ``(p, 3)`` of a relocation: displacements of the first point.

    ``points`` hold a vertex and then its ``n`` neighbors, ``metric`` the
    tensor at the vertex, ``lengths`` the metric lengths of its spokes and
    ``measured`` which of them count. A spoke is outside when it is measured
    and its length is outside the band. The pulls, in order:

    - with ``jacobian`` (derivatives of the spoke lengths along the rows of
      ``basis``, shape ``(len(basis), n)``), for a deep search with an outside
      spoke: the least-squares step that moves every measured spoke to the
      band, then the gradient direction of each outside spoke;
    - the spring forces of the outside spokes and their mean over all spokes
      (forces first for a deep search, the mean first otherwise);
    - for a deep search with an outside spoke: each outside force without its
      part along every band normal it pushes across the band, then, for an
      ``interior`` vertex, its part along each cross product of two normals.
      Forces and normals are projected by ``projector`` when it is given.

    A pull that is not finite is left out. Deterministic kernel values: the
    same bits on every platform.
    """
    coordinates = np.ascontiguousarray(points, dtype=np.float64)
    tensor = np.ascontiguousarray(metric, dtype=np.float64)
    spokes = np.ascontiguousarray(lengths, dtype=np.float64)
    counted = np.ascontiguousarray(measured, dtype=np.int8)
    count = spokes.shape[0]
    if (
        coordinates.shape != (count + 1, 3)
        or count < 1
        or tensor.shape != (3, 3)
        or spokes.ndim != 1
        or counted.shape != (count,)
    ):
        raise ValueError(
            "points, metric, lengths and measured must have shapes (n + 1, 3), "
            "(3, 3), (n,), (n,) with n >= 1."
        )
    frame = np.ascontiguousarray(basis, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[1] != 3 or frame.shape[0] > 3:
        raise ValueError("basis must have shape (k, 3) with k <= 3.")
    matrix = None
    if projector is not None:
        matrix = np.ascontiguousarray(projector, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("projector must have shape (3, 3).")
    derivatives = None
    if jacobian is not None:
        derivatives = np.ascontiguousarray(jacobian, dtype=np.float64)
        if derivatives.shape != (frame.shape[0], count) or frame.shape[0] == 0:
            raise ValueError("jacobian must have shape (len(basis), n) with a basis.")
    # Rows the kernel can write if every measured spoke is outside the band:
    # gradient pulls, forces and mean, tangents.
    active = int(np.count_nonzero(counted))
    capacity = active + 1
    if derivatives is not None:
        capacity += active + 1
    if deep:
        capacity += active * (active + (active * (active - 1) // 2 if interior else 0))
    pulls = np.empty((capacity, 3), dtype=np.float64)
    written = _I64(0)
    status = load_meshcore()["phx_mc_metric_relocation_pulls"](
        count,
        coordinates.ctypes.data,
        tensor.ctypes.data,
        spokes.ctypes.data,
        counted.ctypes.data,
        int(deep),
        int(interior),
        frame.shape[0],
        frame.ctypes.data,
        None if matrix is None else matrix.ctypes.data,
        None if derivatives is None else derivatives.ctypes.data,
        capacity,
        pulls.ctypes.data,
        ctypes.byref(written),
    )
    _raise_for_call(status, "metric_relocation_pulls")
    return pulls[: written.value].copy()


class MetricCarrierOrder(IntEnum):
    """Order in which the dyadic resolutions of one goal are tried."""

    COARSE_FIRST = 0
    FINE_FIRST = 1
    COARSE_THEN_FINE = 2


class MetricCarrier(NamedTuple):
    """One exact carrier: ``sum_i weights[i] / 2**bits * X[rows[i]]``."""

    rows: tuple[int, ...]
    """Distinct source point rows, ascending."""
    weights: tuple[int, ...]
    """Positive integers that sum to ``2**bits``."""
    bits: int
    point: np.ndarray
    """The correctly rounded binary64 image of the exact point."""
    exact: bool
    """Whether the image is the exact point itself."""


class MetricPositionVerdict(IntEnum):
    """Verdict of one position of a collapse survivor, in the order decided."""

    ADMITTED = 0
    COINCIDENT = 1
    """The position is the other end of a spoke."""
    NOT_DECREASING = 2
    """The band violations do not strictly decrease."""
    NOT_ORIENTED = 3
    """A cell is not positively oriented or is under the determinant floor."""
    NO_QUALITY = 4
    """A cell quality is not positive and finite."""
    UNMEASURED = 5
    """A length is not a positive finite number."""
    OUTSIDE_DOMAIN = 6
    """A cell coordinate is outside the exact predicate domain."""


class MetricSearchEnd(IntEnum):
    """How a search of survivor moves ended."""

    EXHAUSTED = 0
    IN_BAND = 1
    """The last move leaves no spoke above the band."""
    GOAL_NOT_FINITE = 2
    """A goal is not a finite point; the moves before it stand."""
    UNMEASURED = 3
    """The last entry is not a move: its verdict is ``UNMEASURED``."""
    OUTSIDE_DOMAIN = 4
    """The last entry is not a move: its verdict is ``OUTSIDE_DOMAIN``."""


class MetricCollapseStar(NamedTuple):
    """The star of an edge collapse; row 0 is the survivor."""

    points: np.ndarray
    """``(n, 3)`` coordinates."""
    metrics: np.ndarray
    """``(n, 3, 3)`` tensors."""
    spokes: np.ndarray
    """Star row (not 0) of the other end of each measured survivor spoke."""
    before_lengths: np.ndarray
    """Metric lengths of the measured edges the collapse removes."""
    cells: np.ndarray
    """``(m, 4)`` star rows of the survivor's cells after the collapse."""
    determinant_floor: float
    """A cell is oriented only above this relative determinant."""


class MetricSurvivorMove(NamedTuple):
    """A carrier of the source a collapse survivor may move to."""

    goal: int
    rows: tuple[int, ...]
    """Distinct source point rows, ascending."""
    weights: tuple[int, ...]
    """Positive integers that sum to ``2**bits``."""
    bits: int
    point: np.ndarray
    metric: np.ndarray
    """``(3, 3)`` log-Euclidean mean of the source tensors over ``rows``."""
    quality: float
    """Smallest cell quality of the star at the move."""
    lengths: np.ndarray
    """Metric lengths of the survivor's spokes at the move."""


def _collapse_star_arrays(
    star: MetricCollapseStar, least: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.ascontiguousarray(star.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < least:
        raise ValueError(f"star points must have shape (n, 3) with n >= {least}.")
    metrics = _metric_tensors(star.metrics, "star metrics")
    if metrics.shape[0] != points.shape[0]:
        raise ValueError("star metrics must hold one tensor per point.")
    spokes = np.ascontiguousarray(star.spokes, dtype=np.int64)
    before = np.ascontiguousarray(star.before_lengths, dtype=np.float64)
    if spokes.ndim != 1 or before.ndim != 1:
        raise ValueError("star spokes and lengths must be one-dimensional.")
    cells = _metric_rows(star.cells, "star cells", 4)
    if cells.shape[0] == 0:
        raise ValueError("A collapse star has at least one cell.")
    return points, metrics, spokes, before, cells


def metric_collapse_positions(
    star: MetricCollapseStar, positions: np.ndarray, metrics: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verdicts of positions ``(p, 3)`` of a collapse survivor with tensors ``(p, 3, 3)``.

    A position replaces row 0 of the star. Returns the int32
    :class:`MetricPositionVerdict` of each, the smallest cell quality of an
    admitted one and its spoke lengths ``(p, s)`` (zero for a coincident
    one). A position is admitted when the violations of its spokes are
    strictly smaller than those of the lengths before (quanta above the band
    largest first, a proper prefix smaller, then those below it), every cell
    is positive by the exact orientation predicate and above the determinant
    floor, and every cell quality is positive. The same verdicts and values
    on every platform.
    """
    points, tensors, spokes, before, cells = _collapse_star_arrays(star, 1)
    candidates = np.ascontiguousarray(positions, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("positions must have shape (p, 3).")
    candidate_metrics = _metric_tensors(metrics, "metrics")
    if candidate_metrics.shape[0] != candidates.shape[0]:
        raise ValueError("metrics must hold one tensor per position.")
    verdicts = np.empty((candidates.shape[0],), dtype=np.int32)
    qualities = np.empty((candidates.shape[0],), dtype=np.float64)
    lengths = np.empty((candidates.shape[0], spokes.shape[0]), dtype=np.float64)
    status = load_meshcore()["phx_mc_metric_collapse_positions"](
        points.shape[0],
        points.ctypes.data,
        tensors.ctypes.data,
        spokes.shape[0],
        spokes.ctypes.data,
        before.shape[0],
        before.ctypes.data,
        cells.shape[0],
        cells.ctypes.data,
        star.determinant_floor,
        candidates.shape[0],
        candidates.ctypes.data,
        candidate_metrics.ctypes.data,
        verdicts.ctypes.data,
        qualities.ctypes.data,
        lengths.ctypes.data,
    )
    _raise_for_call(status, "metric_collapse_positions")
    return verdicts, qualities, lengths


def _simplex_arrays(
    simplices: Sequence[tuple[tuple[int, ...], bool]], /
) -> tuple[np.ndarray, np.ndarray]:
    """Three point rows per edge or triangle (unused -1) and its declared flag."""
    simplex_rows = np.full((len(simplices), 3), -1, dtype=np.int64)
    declared = np.zeros((len(simplices),), dtype=np.int8)
    for index, (vertices, is_declared) in enumerate(simplices):
        simplex_rows[index, : len(vertices)] = vertices
        declared[index] = is_declared
    return simplex_rows, declared


# Three goals at the 53 resolutions of a carrier.
_SURVIVOR_MOVE_CAPACITY = 159


@final
class MetricSource:
    """Exact tetrahedral source of a metric adaptation (owned native handle).

    A carrier of a floating-point goal is an exact combination of source
    points with dyadic weights whose binary64 image is closer to the goal than
    the current point. The search is integer arithmetic over the exact
    points, so its result is the same on every platform.
    """

    __slots__ = ("_handle", "_library", "_searches")

    def __init__(
        self,
        points: np.ndarray,
        exact_points: Sequence[Sequence[Fraction]],
        cells: np.ndarray,
        metrics: np.ndarray,
        logarithms: np.ndarray,
        /,
    ) -> None:
        """``points`` are the binary64 roundings of ``exact_points``, ``cells``
        four point rows per cell in the order cells are searched, and
        ``metrics`` and ``logarithms`` the ``(n, 3, 3)`` tensors of the points
        and their logarithms."""
        self._handle = ctypes.c_void_p()
        self._library = load_meshcore()
        self._searches: weakref.WeakSet[MetricRelocationSearch] = weakref.WeakSet()
        coordinates = np.ascontiguousarray(points, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("points must have shape (n, 3).")
        rows = _metric_rows(cells, "cells", 4)
        if len(exact_points) != coordinates.shape[0] or any(
            len(point) != 3 for point in exact_points
        ):
            raise ValueError("exact_points must hold three rationals per point.")
        tensors = _metric_tensors(metrics, "metrics")
        tensor_logarithms = _metric_tensors(logarithms, "logarithms")
        if (
            tensors.shape[0] != coordinates.shape[0]
            or tensor_logarithms.shape != tensors.shape
        ):
            raise ValueError("metrics and logarithms must hold one tensor per point.")
        # One common denominator: point p is numerators[p] / denominator.
        denominator = lcm(
            *(value.denominator for point in exact_points for value in point)
        )
        integers = [
            value.numerator * (denominator // value.denominator)
            for point in exact_points
            for value in point
        ]
        integers.append(denominator)
        lengths = [(abs(value).bit_length() + 31) // 32 for value in integers]
        offsets = np.zeros((len(integers) + 1,), dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        words = np.frombuffer(
            b"".join(
                abs(value).to_bytes(4 * length, "little")
                for value, length in zip(integers, lengths, strict=True)
            ),
            dtype="<u4",
        )
        signs = np.asarray(
            [-1 if value < 0 else 1 for value in integers[:-1]], dtype=np.int8
        )
        status = self._library["phx_mc_metric_source_create"](
            coordinates.shape[0],
            coordinates.ctypes.data,
            words.ctypes.data,
            offsets.ctypes.data,
            signs.ctypes.data,
            rows.shape[0],
            rows.ctypes.data,
            tensors.ctypes.data,
            tensor_logarithms.ctypes.data,
            ctypes.byref(self._handle),
        )
        _raise_for_call(status, "MetricSource")

    def carriers(
        self,
        target: np.ndarray,
        current: np.ndarray,
        simplices: Sequence[tuple[tuple[int, ...], bool]] | None,
        order: MetricCarrierOrder,
        limit: int,
        /,
        *,
        cell: int | None = None,
    ) -> tuple[MetricCarrier, ...]:
        """Carriers of ``target`` closer to it than ``current``, at most ``limit``.

        With ``simplices`` ``None`` the target is located in the first cell
        that holds it (in ``cell`` alone when given) and is its own goal.
        Otherwise each simplex is an edge or a triangle (point rows) with
        whether it is a declared exact face; the goal is the rounded
        orthogonal projection on the first simplex that holds it, and a
        carrier there must be exact unless the simplex is declared. At
        ``2**-bits`` the partial sums of the barycentric weights are rounded
        half to even; ``order`` walks the resolutions from the coarsest that
        can resolve the move to 53 bits, or back.
        """
        if not self._handle.value:
            raise RuntimeError("The metric source is closed.")
        goal = np.ascontiguousarray(target, dtype=np.float64)
        start = np.ascontiguousarray(current, dtype=np.float64)
        if goal.shape != (3,) or start.shape != (3,):
            raise ValueError("target and current must have shape (3,).")
        # Only edges and triangles are projected on; a located route with
        # none of them has no carrier (and is not a search of the cells).
        usable = (
            []
            if simplices is None
            else [entry for entry in simplices if len(entry[0]) in (2, 3)]
        )
        if simplices is not None and not usable:
            return ()
        simplex_rows, declared = _simplex_arrays(usable)
        room = max(limit, 0)
        rows = np.empty((room, 4), dtype=np.int64)
        weights = np.empty((room, 4), dtype=np.int64)
        bits = np.empty((room,), dtype=np.int32)
        points = np.empty((room, 3), dtype=np.float64)
        exact = np.empty((room,), dtype=np.int8)
        found = _I64(0)
        status = self._library["phx_mc_metric_source_carriers"](
            self._handle,
            goal.ctypes.data,
            start.ctypes.data,
            -1 if cell is None or simplices is not None else cell,
            simplex_rows.shape[0],
            simplex_rows.ctypes.data,
            declared.ctypes.data,
            int(order),
            room,
            rows.ctypes.data,
            weights.ctypes.data,
            bits.ctypes.data,
            points.ctypes.data,
            exact.ctypes.data,
            ctypes.byref(found),
        )
        _raise_for_call(status, "MetricSource.carriers")
        carriers = []
        for index in range(found.value):
            used = int(np.count_nonzero(rows[index] >= 0))
            carriers.append(
                MetricCarrier(
                    tuple(rows[index, :used].tolist()),
                    tuple(weights[index, :used].tolist()),
                    int(bits[index]),
                    points[index].copy(),
                    bool(exact[index]),
                )
            )
        return tuple(carriers)

    def survivor_moves(
        self,
        star: MetricCollapseStar,
        simplices: Sequence[tuple[tuple[int, ...], bool]] | None,
        every_resolution: bool,
        /,
    ) -> tuple[tuple[MetricSurvivorMove, ...], MetricSearchEnd]:
        """Moves of the survivor of ``star`` to carriers of the source, in order.

        Row 1 of the star is the removed vertex and the merged link follows.
        The three goals of :func:`metric_survivor_goals` are tried in order,
        projected onto the hull of the first simplex when ``simplices`` (edges
        or triangles, as for :meth:`carriers`) are given. The carriers of a
        goal are searched coarse first from the survivor: all of them with
        ``every_resolution``, the coarsest only without. A carrier at the
        survivor or the removed vertex, or without a tensor, is skipped; the
        others are evaluated as positions (:func:`metric_collapse_positions`)
        and the admitted ones returned. The search ends as the returned
        :class:`MetricSearchEnd` states; for its last two values the last
        entry is not a move but the carrier whose verdict the caller reports.
        """
        if not self._handle.value:
            raise RuntimeError("The metric source is closed.")
        points, tensors, spokes, before, cells = _collapse_star_arrays(star, 2)
        simplex_rows, declared = _simplex_arrays(() if simplices is None else simplices)
        room = _SURVIVOR_MOVE_CAPACITY
        goals = np.empty((room,), dtype=np.int32)
        rows = np.empty((room, 4), dtype=np.int64)
        weights = np.empty((room, 4), dtype=np.int64)
        bits = np.empty((room,), dtype=np.int32)
        carrier_points = np.empty((room, 3), dtype=np.float64)
        carrier_metrics = np.empty((room, 3, 3), dtype=np.float64)
        qualities = np.empty((room,), dtype=np.float64)
        lengths = np.empty((room, spokes.shape[0]), dtype=np.float64)
        found = _I64(0)
        end = _I32(0)
        status = self._library["phx_mc_metric_source_survivor_moves"](
            self._handle,
            points.shape[0],
            points.ctypes.data,
            tensors.ctypes.data,
            spokes.shape[0],
            spokes.ctypes.data,
            before.shape[0],
            before.ctypes.data,
            cells.shape[0],
            cells.ctypes.data,
            star.determinant_floor,
            simplex_rows.shape[0],
            simplex_rows.ctypes.data,
            declared.ctypes.data,
            int(every_resolution),
            room,
            goals.ctypes.data,
            rows.ctypes.data,
            weights.ctypes.data,
            bits.ctypes.data,
            carrier_points.ctypes.data,
            carrier_metrics.ctypes.data,
            qualities.ctypes.data,
            lengths.ctypes.data,
            ctypes.byref(found),
            ctypes.byref(end),
        )
        _raise_for_call(status, "MetricSource.survivor_moves")
        moves = []
        for index in range(found.value):
            used = int(np.count_nonzero(rows[index] >= 0))
            moves.append(
                MetricSurvivorMove(
                    int(goals[index]),
                    tuple(rows[index, :used].tolist()),
                    tuple(weights[index, :used].tolist()),
                    int(bits[index]),
                    carrier_points[index].copy(),
                    carrier_metrics[index].copy(),
                    float(qualities[index]),
                    lengths[index].copy(),
                )
            )
        return tuple(moves), MetricSearchEnd(end.value)

    def relocation_search(
        self,
        star: MetricRelocationStar,
        pulls: np.ndarray,
        frame: tuple[np.ndarray, np.ndarray] | None,
        simplices: Sequence[tuple[tuple[int, ...], bool]] | None,
        /,
        *,
        deep: bool,
        refine: bool,
        band_only: bool,
        constant_field: bool,
    ) -> MetricRelocationSearch:
        """The candidate positions of one relocated vertex, as a sequence.

        ``pulls`` are ``(p, 3)`` displacements; ``frame`` the origin and the
        ``(3, 3)`` projector of the vertex's carrier hull, or ``None``;
        ``simplices`` the carrier's edges or triangles as for
        :meth:`carriers`, or ``None`` for a vertex that moves in the source
        cells. See :class:`MetricRelocationSearch`.
        """
        if not self._handle.value:
            raise RuntimeError("The metric source is closed.")
        search = MetricRelocationSearch(
            self,
            star,
            pulls,
            frame,
            simplices,
            deep=deep,
            refine=refine,
            band_only=band_only,
            constant_field=constant_field,
        )
        self._searches.add(search)
        return search

    def close(self) -> None:
        # A search reads this source: it ends first.
        for search in tuple(self._searches):
            search.close()
        if self._handle.value:
            self._library["phx_mc_metric_source_free"](self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        # A constructor that raised before the handle existed leaves nothing
        # to free; interpreter shutdown may already have unloaded the library.
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_library", None)
        if library is not None and handle is not None and handle.value:
            try:
                library["phx_mc_metric_source_free"](handle)
            except BaseException:
                pass


def planar_partition(
    target_count: int,
    declared_count: int,
    words: np.ndarray,
    offsets: np.ndarray,
    signs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[int, int, int]:
    """Decide a partition; the native entry verifies every supplied box exactly."""
    words = np.ascontiguousarray(words, dtype=np.uint32)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    signs = np.ascontiguousarray(signs, dtype=np.int8)
    lower = np.ascontiguousarray(lower, dtype=np.float64)
    upper = np.ascontiguousarray(upper, dtype=np.float64)
    count = target_count + declared_count
    if (
        target_count < 0
        or declared_count < 0
        or words.ndim != 1
        or offsets.shape != (6 * count + 2,)
        or signs.shape != (6 * count,)
        or lower.shape != (count, 2)
        or upper.shape != (count, 2)
    ):
        raise ValueError("Planar partition arrays have incompatible shapes.")
    verdict = _I32()
    first = _I64()
    second = _I64()
    status = load_meshcore()["phx_mc_planar_partition"](
        target_count,
        declared_count,
        words.size,
        words.ctypes.data,
        offsets.ctypes.data,
        signs.ctypes.data,
        lower.ctypes.data,
        upper.ctypes.data,
        ctypes.byref(verdict),
        ctypes.byref(first),
        ctypes.byref(second),
    )
    _raise_for_call(status, "planar_partition")
    return verdict.value, first.value, second.value


class MetricRelocationEnd(IntEnum):
    """What a call of :meth:`MetricRelocationSearch.next` returned."""

    GROUP = 0
    """The candidates of one goal, at least one."""
    EXHAUSTED = 1
    """No goal is left."""
    UNMEASURED_GOAL = 2
    """A goal whose screen met a spoke without a finite length."""
    UNMEASURED_CARRIER = 3
    """A carrier whose evaluation met one."""
    NO_TENSOR = 4
    """A carrier whose tensor could not be interpolated."""


class MetricRelocationStar(NamedTuple):
    """The star of a relocated vertex.

    Row 0 of ``points`` and ``metrics`` is the vertex and rows 1.. are its
    neighbors; spoke ``j`` runs from row 0 to row ``j + 1``.
    """

    points: np.ndarray
    metrics: np.ndarray
    measured: np.ndarray
    """Per spoke, whether its length counts toward the band violations."""
    before_lengths: np.ndarray
    """Per spoke, its metric length now."""
    before_quality: float
    """Smallest cell quality now."""
    cells: np.ndarray
    """``(k, 4)`` star rows of the cells."""
    determinant_floor: float
    improvement: float


class MetricRelocationCandidate(NamedTuple):
    """A carrier of the source a relocated vertex may move to."""

    rows: tuple[int, ...]
    """Distinct source point rows, ascending; empty for a reported goal."""
    weights: tuple[int, ...]
    """Positive integers that sum to ``2**bits``."""
    bits: int
    point: np.ndarray
    metric: np.ndarray
    """``(3, 3)`` log-Euclidean mean of the source tensors over ``rows``."""
    quality: float
    """Smallest cell quality of the star with the vertex at the carrier."""
    lengths: np.ndarray
    """Metric lengths of the vertex's spokes at the carrier."""


# The carriers of one goal: at most the 53 resolutions and one more.
_RELOCATION_GROUP_CAPACITY = 54


@final
class MetricRelocationSearch:
    """Candidate positions of one relocated vertex (owned native handle).

    Each pull ``p`` gives the goals ``x + 2**-s p`` for ``s = 0, 1, ...``
    until the goal is the vertex ``x`` itself (``s <= 5`` unless the search
    is deep). A goal is projected onto the carrier hull when the vertex has a
    frame; one seen before is passed over. A goal is screened in binary64
    (in a constant field by the full evaluation with the vertex's own tensor,
    otherwise by the determinants of the star) and, when kept, replaced by
    exact carriers of the source: the coarsest; for a deep search the
    coarsest and the finest; for a refining search the finest and, after
    every goal, the remaining resolutions of the kept goals in order. A
    carrier is a candidate when no neighbor lies at it, the band violations
    of its measured spokes do not increase (and decrease when the search is
    band-only), every cell keeps a positive determinant above the floor and,
    with unchanged violations, the smallest cell quality improves by more
    than the improvement.

    :meth:`next` returns the candidates of the next goal that has any, in the
    order of its carriers. The sequence does not depend on how the calls are
    spaced and is the same on every platform. The search reads its source,
    which closes it when it closes.
    """

    __slots__ = ("__weakref__", "_handle", "_library", "_source", "_spokes")

    def __init__(
        self,
        source: MetricSource,
        star: MetricRelocationStar,
        pulls: np.ndarray,
        frame: tuple[np.ndarray, np.ndarray] | None,
        simplices: Sequence[tuple[tuple[int, ...], bool]] | None,
        /,
        *,
        deep: bool,
        refine: bool,
        band_only: bool,
        constant_field: bool,
    ) -> None:
        self._handle = ctypes.c_void_p()
        self._library = source._library
        self._source = source
        points = np.ascontiguousarray(star.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("star points must have shape (n, 3) with n >= 2.")
        metrics = _metric_tensors(star.metrics, "star metrics")
        if metrics.shape[0] != points.shape[0]:
            raise ValueError("star metrics must hold one tensor per point.")
        self._spokes = points.shape[0] - 1
        measured = np.ascontiguousarray(star.measured, dtype=np.int8)
        before = np.ascontiguousarray(star.before_lengths, dtype=np.float64)
        if measured.shape != (self._spokes,) or before.shape != (self._spokes,):
            raise ValueError("A relocation star holds one flag and one length per spoke.")
        cells = _metric_rows(star.cells, "star cells", 4)
        if cells.shape[0] == 0:
            raise ValueError("A relocation star has at least one cell.")
        directions = np.ascontiguousarray(pulls, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("pulls must have shape (p, 3).")
        origin = projector = None
        if frame is not None:
            origin = np.ascontiguousarray(frame[0], dtype=np.float64)
            projector = np.ascontiguousarray(frame[1], dtype=np.float64)
            if origin.shape != (3,) or projector.shape != (3, 3):
                raise ValueError("A frame is an origin (3,) and a projector (3, 3).")
        simplex_rows, declared = _simplex_arrays(() if simplices is None else simplices)
        status = self._library["phx_mc_metric_relocation_create"](
            source._handle,
            points.shape[0],
            points.ctypes.data,
            metrics.ctypes.data,
            measured.ctypes.data,
            before.ctypes.data,
            star.before_quality,
            cells.shape[0],
            cells.ctypes.data,
            star.determinant_floor,
            star.improvement,
            directions.shape[0],
            directions.ctypes.data,
            None if origin is None else origin.ctypes.data,
            None if projector is None else projector.ctypes.data,
            simplex_rows.shape[0],
            simplex_rows.ctypes.data,
            declared.ctypes.data,
            int(deep) | int(refine) << 1 | int(band_only) << 2 | int(constant_field) << 3,
            ctypes.byref(self._handle),
        )
        _raise_for_call(status, "MetricRelocationSearch")

    def next(
        self,
    ) -> tuple[tuple[MetricRelocationCandidate, ...], MetricRelocationEnd]:
        """The candidates of the next goal that has any, and how the call ended.

        For the three failures of :class:`MetricRelocationEnd` the one entry
        is the goal or the carrier that cannot be decided, for the caller to
        report.
        """
        if not self._handle.value:
            raise RuntimeError("The relocation search is closed.")
        room = _RELOCATION_GROUP_CAPACITY
        rows = np.empty((room, 4), dtype=np.int64)
        weights = np.empty((room, 4), dtype=np.int64)
        bits = np.empty((room,), dtype=np.int32)
        points = np.empty((room, 3), dtype=np.float64)
        metrics = np.empty((room, 3, 3), dtype=np.float64)
        qualities = np.empty((room,), dtype=np.float64)
        lengths = np.zeros((room, self._spokes), dtype=np.float64)
        found = _I64(0)
        end = _I32(0)
        status = self._library["phx_mc_metric_relocation_next"](
            self._handle,
            room,
            rows.ctypes.data,
            weights.ctypes.data,
            bits.ctypes.data,
            points.ctypes.data,
            metrics.ctypes.data,
            qualities.ctypes.data,
            lengths.ctypes.data,
            ctypes.byref(found),
            ctypes.byref(end),
        )
        _raise_for_call(status, "MetricRelocationSearch.next")
        group = []
        for index in range(found.value):
            used = int(np.count_nonzero(rows[index] >= 0))
            group.append(
                MetricRelocationCandidate(
                    tuple(rows[index, :used].tolist()),
                    tuple(weights[index, :used].tolist()),
                    int(bits[index]),
                    points[index].copy(),
                    metrics[index].copy(),
                    float(qualities[index]),
                    lengths[index].copy(),
                )
            )
        return tuple(group), MetricRelocationEnd(end.value)

    def close(self) -> None:
        if self._handle.value:
            self._library["phx_mc_metric_relocation_free"](self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        # A constructor that raised before the handle existed leaves nothing
        # to free; interpreter shutdown may already have unloaded the library.
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_library", None)
        if library is not None and handle is not None and handle.value:
            try:
                library["phx_mc_metric_relocation_free"](handle)
            except BaseException:
                pass


__all__ = [
    "BVH_BUILD_MEDIAN",
    "BVH_BUILD_MORTON",
    "BVH_BUILD_SAH",
    "bvh_build",
    "angle_arccosine",
    "angle_arctangent2",
    "quality_dihedral_angles",
    "quality_solid_corners",
    "angle_arcsine",
    "angle_between",
    "angle_cosine",
    "angle_sine",
    "planar_partition",
    "ARRANGEMENT_UNCLASSIFIED",
    "ArrangementClassification",
    "ArrangementFailure",
    "CadEntityKind",
    "CadGeometryQuantity",
    "CadModel",
    "cad_geometry_evaluate",
    "ConstrainedDelaunay2D",
    "ContactIntersections",
    "IntersectionClass",
    "MeshcoreError",
    "MeshcoreLibrary",
    "MetricCarrier",
    "MetricCarrierOrder",
    "MetricCollapseStar",
    "MetricPositionVerdict",
    "MetricRelocationCandidate",
    "MetricRelocationEnd",
    "MetricRelocationSearch",
    "MetricRelocationStar",
    "MetricSearchEnd",
    "MetricSource",
    "MetricSurvivorMove",
    "MixedElementKind",
    "MixedTemplateCounts",
    "mixed_template",
    "mixed_template_counts",
    "MeshcoreStatus",
    "MeshcoreUnavailableError",
    "NativeArrangement",
    "NativePlcRecovery",
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
    "PlcSearchBudget",
    "PLC_SEARCH_BUDGET",
    "PlcSourceFailure",
    "prepare_plc_source",
    "recover_plc",
    "recover_plc_3d",
    "TetMesh3D",
    "TetMeshMetricAdaptation",
    "TetMeshArrays",
    "TetMeshEdit",
    "TetMeshEditKind",
    "TetMeshEditStatus",
    "TetMeshImplicitFacetRefusal",
    "TetMeshImplicitSource",
    "TetMeshImplicitSnapshot",
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
    "TetMeshQuality",
    "TetMeshRun",
    "TetMeshSourceComplex",
    "TetMeshSourceEvidence",
    "TetMeshUnmet",
    "TetMeshUnmetReason",
    "TET_MESH_UNMET_NO_APEX",
    "TET_MESH_UNMET_NO_FACET",
    "QuadraticValidityDecision",
    "RestrictedPowerCells",
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
    "implicit_face_proposals",
    "parallel_for",
    "interval_program_branches",
    "interval_program_execute",
    "load_meshcore",
    "meshcore_available",
    "meshcore_exact_domain",
    "meshcore_identity",
    "metric_band_quanta",
    "metric_congruence_defects",
    "metric_interpolations",
    "metric_lengths",
    "metric_hull_frame",
    "metric_logarithms",
    "metric_midpoints",
    "metric_project_goals",
    "metric_relative_determinants",
    "metric_relocation_pulls",
    "metric_survivor_goals",
    "metric_tensor_properties",
    "metric_tetra_quality",
    "point_triangle_distances",
    "point_triangle_locations",
    "polygon_intersection_moments",
    "polygon_intersection_simplices",
    "polyhedron_clip_moments",
    "quadratic_tetrahedron_validity",
    "regular_2d",
    "regular_3d",
    "relocated_tet_mesh_arrays",
    "restricted_power_cells",
    "restricted_centers_3d",
    "restricted_power_cells_exact",
    "segment_triangle_intersections",
    "simplex_quadratic_bounds",
    "symbolic_incircle",
    "symbolic_insphere",
    "symbolic_orient2d",
    "symbolic_orient3d",
    "tetrahedron_intersection_moments",
    "tetrahedron_intersection_quadratic_moments",
    "tetrahedron_intersection_simplices",
    "tetrahedron_dual_hexes",
    "triangle_dual_quads",
    "triangle_intersection_classes",
    "triangle_intersections",
    "SurfaceEnvelopeResult",
    "surface_envelope",
    "IntervalBoxProgram",
    "MeshCertificate",
    "MeshRemeshResult",
    "mesh_certify",
    "mesh_remesh",
    "ZeroSetCertificate",
    "ZeroSetRemeshResult",
    "zero_set_certify",
    "zero_set_remesh",
    "CadSurfaceResult",
    "cad_mesh",
]

__all__ += ["ConformingVoronoi", "conforming_voronoi"]
__all__ += ["HexDominant", "hex_dominant"]

__all__ += [
    "CadFidelityChords",
    "CadFidelityMesh",
    "CadFidelityResult",
    "verify_cad_fidelity",
    "ChartFailureKind",
    "SurfaceEmbeddingResult",
    "SurfaceFindingKind",
    "verify_chart_partition",
    "verify_surface_embedding",
]


class _CadSurfaceControls(ctypes.Structure):
    _fields_ = (
        ("face_sizes", _P),
        ("face_tolerances", _P),
        ("curve_tolerances", _P),
        ("tolerance_fraction", _F64),
        ("grading", _F64),
        ("minimum_size", _F64),
    )


class _CadSurfaceArrays(ctypes.Structure):
    _fields_ = (
        ("vertex_capacity", _I64),
        ("triangle_capacity", _I64),
        ("sliver_capacity", _I64),
        ("curve_edge_capacity", _I64),
        ("face_capacity", _I64),
        ("curve_capacity", _I64),
        ("parameter_capacity", _I64),
        ("failure_capacity", _I64),
        ("face_failure_capacity", _I64),
        ("points", _P),
        ("vertex_dimensions", _P),
        ("vertex_entities", _P),
        ("vertex_parameters", _P),
        ("triangles", _P),
        ("triangle_patches", _P),
        ("triangle_uv", _P),
        ("triangle_edge_uses", _P),
        ("facet_regions", _P),
        ("sliver_uv", _P),
        ("sliver_patches", _P),
        ("curve_edges", _P),
        ("curve_edge_curves", _P),
        ("curve_edge_parameters", _P),
        ("face_stats", _P),
        ("curve_offsets", _P),
        ("curve_parameters", _P),
        ("curve_sag_splits", _P),
        ("curve_failures", _P),
        ("curve_statistics", _P),
        ("face_failures", _P),
    )


_CAD_FACE_STATS_DTYPE = np.dtype(
    [
        (name, np.int64)
        for name in (
            "metric_evaluations",
            "insertions",
            "unresolved_size",
            "unresolved_angle",
            "encroachment_refusals",
            "nonstar_refusals",
            "outside_refusals",
            "proximity_refusals",
            "unresolved_triangles",
            "excluded_refusals",
            "fixed_triangles_skipped",
            "omitted_pole_points",
            "periodic_corner_gaps",
        )
    ]
    + [("max_periodic_corner_gap_ulps", np.uint64)],
    align=True,
)
_CAD_CURVE_STATS_DTYPE = np.dtype(
    [
        (name, np.int64)
        for name in (
            "first_order_evaluations",
            "second_order_evaluations",
            "sag_evaluations",
            "small_feature_resamples",
            "sag_splits",
            "parameter_count",
            "unresolved_count",
        )
    ],
    align=True,
)
_CAD_CURVE_FAILURE_DTYPE = np.dtype(
    [("curve", np.int32), ("first", np.float64), ("last", np.float64)],
    align=True,
)
_CAD_FACE_FAILURE_DTYPE = np.dtype(
    [
        ("face", np.int32),
        ("loop", np.int32),
        ("use", np.int32),
        ("axis", np.int32),
        ("reason", np.int32),
        ("turns", np.int64),
        ("parameter", np.float64),
        ("triangle", np.int32),
        ("bad_size", np.int32),
        ("bad_angle", np.int32),
        ("refusal_kind", np.int32),
        ("curve", np.int32),
        ("curve_segment", np.int32),
        ("candidate_uv", np.float64, (2,)),
        ("triangle_uv", np.float64, (3, 2)),
        ("segment_parameters", np.float64, (2,)),
    ],
    align=True,
)


class _CadChartArrays(ctypes.Structure):
    _fields_ = (
        ("triangle_capacity", _I64),
        ("use_capacity", _I64),
        ("chord_capacity", _I64),
        ("triangles", _P),
        ("patches", _P),
        ("uv", _P),
        ("chart_to_mesh", _P),
        ("edge_chord", _P),
        ("use_patch", _P),
        ("use_range", _P),
        ("chord_use", _P),
        ("chord_loop", _P),
        ("chord_uv", _P),
        ("chord_range", _P),
    )


class CadChartResult(NamedTuple):
    """Chart partition and physical correspondence, including pole slivers."""

    triangles: np.ndarray
    patches: np.ndarray
    uv: np.ndarray
    chart_to_mesh: np.ndarray
    edge_chord: np.ndarray
    use_patch: np.ndarray
    use_range: np.ndarray
    chord_use: np.ndarray
    chord_loop: np.ndarray
    chord_uv: np.ndarray
    chord_range: np.ndarray


class CadSurfaceResult(NamedTuple):
    """Owned surface candidate and the curve/face construction findings.

    ``initial_face_failures`` contains the ordered findings before the first
    shared-curve retry; it is empty if no retry ran. ``face_failures`` and
    ``face_stats`` describe only the emitted surface. ``curve_statistics``
    counts the initial discretization and the refinement that produced the
    emitted curve bank, not discarded proposals. The solid's curve phase
    time includes work on discarded proposals.
    """

    status: MeshcoreStatus
    points: np.ndarray
    vertex_dimensions: np.ndarray
    vertex_entities: np.ndarray
    vertex_parameters: np.ndarray
    triangles: np.ndarray
    triangle_patches: np.ndarray
    triangle_uv: np.ndarray
    triangle_edge_uses: np.ndarray
    facet_regions: np.ndarray
    sliver_uv: np.ndarray
    sliver_patches: np.ndarray
    curve_edges: np.ndarray
    curve_edge_curves: np.ndarray
    curve_edge_parameters: np.ndarray
    face_stats: np.ndarray
    curve_offsets: np.ndarray
    curve_parameters: np.ndarray
    curve_sag_splits: np.ndarray
    curve_failures: np.ndarray
    curve_statistics: np.ndarray
    face_failures: np.ndarray
    pole_sectors: np.ndarray
    chart: CadChartResult
    initial_face_failures: np.ndarray
    surface_refinement_retries: int


class _CadProximityControl(ctypes.Structure):
    _fields_ = (
        ("source_count", _I64),
        ("source_faces", ctypes.c_void_p),
        ("target_count", _I64),
        ("target_faces", ctypes.c_void_p),
        ("elements_per_gap", _I32),
        ("opposite_normals_only", _I32),
        ("minimum_size", ctypes.c_double),
        ("maximum_size", ctypes.c_double),
    )


class _CadVolumeControls(ctypes.Structure):
    _fields_ = (
        ("radius_edge_bound", ctypes.c_double),
        ("min_dihedral_degrees", ctypes.c_double),
        ("min_relative_determinant", ctypes.c_double),
        ("geometry_order", _I32),
        ("proximity_count", _I64),
        ("proximity_controls", ctypes.c_void_p),
    )


class CadProximityControl(NamedTuple):
    """A proximity size control of ``cad_mesh_solid`` over model face rows.

    At a point of a scoped face the size is the local gap to the other
    scope's faces over ``elements_per_gap``, clamped to ``minimum_size`` and
    ``maximum_size`` (``None``: no bound); see ``phx_mc_cad_proximity_control``.
    """

    source_faces: tuple[int, ...]
    target_faces: tuple[int, ...]
    elements_per_gap: int
    minimum_size: float | None = None
    maximum_size: float | None = None
    opposite_normals_only: bool = True


_CAD_QUADRATIC_STATS_FIELDS = (
    "curve_edges",
    "surface_edges",
    "interior_edges",
    "initial_invalid",
    "initial_unresolved",
    "harmonic_nodes",
    "harmonic_repaired_cells",
    "straightened_edges",
    "final_invalid",
    "final_unresolved",
)
_CAD_VOLUME_STATS_DTYPE = np.dtype(
    [
        (name, np.int64)
        for name in (
            "curve_ns",
            "face_ns",
            "assembly_ns",
            "volume_ns",
            "quadratic_ns",
            "recovery_attempts",
            "boundary_splits",
            "refine_inserted",
            "refine_unmet",
            "improve_inserted",
            "improve_collapsed",
            "improve_relocated",
            "unmet_cells",
        )
    ]
    + [
        ("recovery_status", np.int32),
        ("recovery_reason", np.int32),
        ("failure_kind", np.int32),
        ("failure_entity", np.int64),
        (
            "quadratic",
            np.dtype(
                [(name, np.int64) for name in _CAD_QUADRATIC_STATS_FIELDS], align=True
            ),
        ),
        ("failure_measure", np.float64),
        ("unmet_forced_cells", np.int64),
        ("unmet_unforced_cells", np.int64),
        ("unmet_shape_rows", np.int64),
        ("unmet_hard_cells", np.int64),
    ],
    align=True,
)
_CAD_BOUNDARY_SPLIT_DTYPE = np.dtype(
    [
        ("reason", np.int32),
        ("entity_kind", np.int32),
        ("entity", np.int64),
        ("vertices", np.int32, (2,)),
        ("vertex", np.int32),
        ("dimension", np.int32),
        ("carrier", np.int32),
        ("parameters", np.float64, (2,)),
    ],
    align=True,
)


class _CadVolumeArrays(ctypes.Structure):
    _fields_ = (
        ("point_capacity", _I64),
        ("tetrahedron_capacity", _I64),
        ("triangle_capacity", _I64),
        ("edge_capacity", _I64),
        ("record_capacity", _I64),
        ("split_capacity", _I64),
        *(
            (name, _P)
            for name in (
                "points",
                "tetrahedra",
                "tetrahedron_regions",
                "boundary_faces",
                "edges",
                "nodes",
                "edge_kinds",
                "edge_entities",
                "cell_edges",
                "cell_decisions",
                "records",
                "splits",
                "stats",
            )
        ),
        ("curve_capacity", _I64),
        ("curve_rim_wedges", _P),
    )


class CadVolumeResult(NamedTuple):
    """Fixed-boundary tetrahedra of a CAD surface candidate and its P2 nodes.

    Points begin with the surface points; cells are right-handed;
    ``boundary_faces[t]`` is the (cell, local face opposite that vertex) of
    surface triangle ``t``. P2 arrays are empty for geometry order 1.
    """

    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    boundary_faces: np.ndarray
    edges: np.ndarray
    nodes: np.ndarray
    edge_kinds: np.ndarray
    edge_entities: np.ndarray
    cell_edges: np.ndarray
    cell_decisions: np.ndarray
    records: np.ndarray
    splits: np.ndarray
    stats: np.ndarray
    # Per model curve: the smallest true wedge (degrees) between its two
    # faces, written when some unmet cell is not forced, else NaN.
    curve_rim_wedges: np.ndarray


class CadSolidResult(NamedTuple):
    """The surface (after any boundary split) and the volume of a CAD solid."""

    status: MeshcoreStatus
    surface: CadSurfaceResult
    volume: CadVolumeResult


# Default share theta of each face's tolerance the construction may use (the
# rest is the fidelity certificate's margin): the largest theta at which every
# row of every measured mesh certified (V3, pole chart switch at latitude
# 0.6230). Measured at tol = kappa_max h^2 / 4 on spheres of radius 0.5 and
# 1.7 and on the quadric fixture, each at the 20 sizes
# h = 0.02 ... 0.25: 0 refused rows at theta 0.8, 0.75 and 0.7; at 0.85 four
# rows refused (radius 1.7, h 0.0229 and 0.25), at 0.9 nine, at 1 over 16000.
# At 0.8 the largest exact sphere sag is 0.8045 tol (the kernel's
# kappa R^2 / 2 omits the exact sag's relative theta h^2 / (8 R^2) term) and
# the quadric's largest certified bound is 0.826 tol.
CAD_TOLERANCE_FRACTION = 0.8


def cad_mesh(
    model: CadModel,
    face_sizes: object,
    face_tolerances: object,
    curve_tolerances: object,
    /,
    *,
    tolerance_fraction: float = CAD_TOLERANCE_FRACTION,
    grading: float = 1.3,
    minimum_size: float = 1e-12,
) -> CadSurfaceResult:
    """Discretize the curves and assemble a native CAD surface candidate."""
    return _cad_mesh(
        model,
        face_sizes,
        face_tolerances,
        curve_tolerances,
        tolerance_fraction,
        grading,
        minimum_size,
        None,
    )[0]


def cad_mesh_solid(
    model: CadModel,
    face_sizes: object,
    face_tolerances: object,
    curve_tolerances: object,
    /,
    *,
    tolerance_fraction: float = CAD_TOLERANCE_FRACTION,
    grading: float = 1.3,
    minimum_size: float = 1e-12,
    radius_edge_bound: float = 2.0,
    min_dihedral_degrees: float = 15.5,
    min_relative_determinant: float = 0.0,
    geometry_order: int = 1,
    proximity_controls: tuple[CadProximityControl, ...] = (),
) -> CadSolidResult:
    """Curves, faces and the fixed-boundary volume of a CAD solid in one call.

    ``face_sizes`` are the requested edge lengths h; the volume is sized from
    them. ``proximity_controls`` lower the size near thin regions (a first
    surface pass at h is their distance proxy). UNRESOLVED returns the result
    with unmet quality targets, unresolved P2 cells, or no cells when recovery
    still refused after the one boundary split (``volume.stats``).
    """
    faces = [
        (
            np.ascontiguousarray(control.source_faces, dtype=np.int32),
            np.ascontiguousarray(control.target_faces, dtype=np.int32),
        )
        for control in proximity_controls
    ]
    rows = (_CadProximityControl * max(len(faces), 1))(
        *(
            _CadProximityControl(
                source.size,
                source.ctypes.data,
                target.size,
                target.ctypes.data,
                int(control.elements_per_gap),
                int(bool(control.opposite_normals_only)),
                0.0 if control.minimum_size is None else float(control.minimum_size),
                np.inf if control.maximum_size is None else float(control.maximum_size),
            )
            for control, (source, target) in zip(proximity_controls, faces, strict=True)
        )
    )
    controls = _CadVolumeControls(
        radius_edge_bound,
        min_dihedral_degrees,
        min_relative_determinant,
        geometry_order,
        len(faces),
        ctypes.addressof(rows),
    )
    surface, volume = _cad_mesh(
        model,
        face_sizes,
        face_tolerances,
        curve_tolerances,
        tolerance_fraction,
        grading,
        minimum_size,
        controls,
    )
    if volume is None:
        raise MeshcoreError(surface.status, "cad_mesh_solid: no volume")
    return CadSolidResult(surface.status, surface, volume)


def _cad_volume_export(
    library: Any, handle: ctypes.c_void_p, curve_count: int, /
) -> CadVolumeResult:
    counts = np.empty((6,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_cad_volume_sizes"](handle, counts.ctypes.data),
        "cad_volume_sizes",
    )
    p, m, t, n, r, s = (int(c) for c in counts)
    cells = m if n else 0
    banks = (
        np.empty((p, 3), dtype=np.float64),
        np.empty((m, 4), dtype=np.int32),
        np.empty((m,), dtype=np.int32),
        np.empty((t, 2), dtype=np.int32),
        np.empty((n, 2), dtype=np.int32),
        np.empty((n, 3), dtype=np.float64),
        np.empty((n,), dtype=np.int8),
        np.empty((n,), dtype=np.int32),
        np.empty((cells, 6), dtype=np.int64),
        np.empty((cells,), dtype=np.int32),
        np.empty((r, 2), dtype=np.int64),
        np.empty((s,), dtype=_CAD_BOUNDARY_SPLIT_DTYPE),
        np.empty((1,), dtype=_CAD_VOLUME_STATS_DTYPE),
    )
    wedges = np.full((curve_count,), np.nan, dtype=np.float64)
    outputs = _CadVolumeArrays(
        p, m, t, n, r, s, *(a.ctypes.data for a in banks), curve_count, wedges.ctypes.data
    )
    _raise_for_call(
        library["phx_mc_cad_volume_export"](handle, ctypes.addressof(outputs)),
        "cad_volume_export",
    )
    return CadVolumeResult(*banks, wedges)


def _cad_surface_refinement(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> tuple[np.ndarray, int]:
    count = ctypes.c_int64()
    retries = ctypes.c_int32()
    read = library["phx_mc_cad_surface_refinement"]
    _raise_for_call(
        read(handle, 0, None, ctypes.addressof(count), ctypes.addressof(retries)),
        "cad_surface_refinement",
    )
    if count.value < 0 or retries.value < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "negative CAD surface refinement size"
        )
    findings = np.empty((count.value,), dtype=_CAD_FACE_FAILURE_DTYPE)
    if count.value:
        _raise_for_call(
            read(
                handle,
                count.value,
                findings.ctypes.data,
                ctypes.addressof(count),
                ctypes.addressof(retries),
            ),
            "cad_surface_refinement",
        )
    return findings, retries.value


def _cad_mesh(
    model: CadModel,
    face_sizes: object,
    face_tolerances: object,
    curve_tolerances: object,
    tolerance_fraction: float,
    grading: float,
    minimum_size: float,
    volume: _CadVolumeControls | None,
    /,
) -> tuple[CadSurfaceResult, CadVolumeResult | None]:
    from ._exact_kernels import exact_kernel_parallelism

    sizes = _real_array(face_sizes, "face_sizes")
    tolerances = _real_array(face_tolerances, "face_tolerances")
    curve_tols = _real_array(curve_tolerances, "curve_tolerances")
    if sizes.shape != (model.surface_nodes.size,) or tolerances.shape != sizes.shape:
        raise ValueError("face sizes and tolerances must have one value per face.")
    if curve_tols.shape != (model.curve_nodes.size,):
        raise ValueError("curve tolerances must have one value per curve.")
    structure, retained = _cad_model_structure(model)
    controls = _CadSurfaceControls(
        sizes.ctypes.data,
        tolerances.ctypes.data,
        curve_tols.ctypes.data,
        tolerance_fraction,
        grading,
        minimum_size,
    )
    library = load_meshcore()
    handle = ctypes.c_void_p()
    parallelism = exact_kernel_parallelism()
    status = MeshcoreStatus(
        library["phx_mc_cad_mesh_surface"](
            ctypes.addressof(structure),
            ctypes.addressof(controls),
            ctypes.addressof(handle),
            parallelism,
        )
        if volume is None
        else library["phx_mc_cad_mesh_volume"](
            ctypes.addressof(structure),
            ctypes.addressof(controls),
            ctypes.addressof(volume),
            ctypes.addressof(handle),
            parallelism,
        )
    )
    del retained
    if status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
        _raise_for_call(status, "cad_mesh")
    if not handle.value:
        raise MeshcoreError(status, "cad_mesh: no complete candidate")
    try:
        counts = np.empty((9,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_cad_surface_result_sizes"](handle, counts.ctypes.data),
            "cad_surface_result_sizes",
        )
        if np.any(counts < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "negative CAD surface size"
            )
        v, t, s, e, f, c, p, failures, face_failure_count = counts
        points = np.empty((v, 3), dtype=np.float64)
        dimensions = np.empty((v,), dtype=np.int8)
        entities = np.empty((v,), dtype=np.int32)
        parameters = np.empty((v, 2), dtype=np.float64)
        triangles = np.empty((t, 3), dtype=np.int32)
        patches = np.empty((t,), dtype=np.int32)
        uv = np.empty((t, 3, 2), dtype=np.float64)
        uses = np.empty((t, 3), dtype=np.int32)
        regions = np.empty((t, 2), dtype=np.int32)
        sliver_uv = np.empty((s, 3, 2), dtype=np.float64)
        sliver_patches = np.empty((s,), dtype=np.int32)
        edges = np.empty((e, 2), dtype=np.int32)
        edge_curves = np.empty((e,), dtype=np.int32)
        edge_parameters = np.empty((e, 2), dtype=np.float64)
        face_stats = np.empty((f,), dtype=_CAD_FACE_STATS_DTYPE)
        offsets = np.empty((c + 1,), dtype=np.int64)
        curve_parameters = np.empty((p,), dtype=np.float64)
        splits = np.empty((c,), dtype=np.int64)
        curve_failures = np.empty((failures,), dtype=_CAD_CURVE_FAILURE_DTYPE)
        curve_statistics = np.empty((1,), dtype=_CAD_CURVE_STATS_DTYPE)
        face_failures = np.empty((face_failure_count,), dtype=_CAD_FACE_FAILURE_DTYPE)
        banks = (
            points,
            dimensions,
            entities,
            parameters,
            triangles,
            patches,
            uv,
            uses,
            regions,
            sliver_uv,
            sliver_patches,
            edges,
            edge_curves,
            edge_parameters,
            face_stats,
            offsets,
            curve_parameters,
            splits,
            curve_failures,
            curve_statistics,
            face_failures,
        )
        outputs = _CadSurfaceArrays(
            *(int(n) for n in counts), *(a.ctypes.data for a in banks)
        )
        _raise_for_call(
            library["phx_mc_cad_surface_result_export"](
                handle, ctypes.addressof(outputs)
            ),
            "cad_surface_result_export",
        )
        sector_count = ctypes.c_int64()
        _raise_for_call(
            library["phx_mc_cad_surface_pole_sectors"](
                handle, 0, None, ctypes.addressof(sector_count)
            ),
            "cad_surface_pole_sectors",
        )
        pole_sectors = np.empty((sector_count.value, 4), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_cad_surface_pole_sectors"](
                handle,
                sector_count.value,
                pole_sectors.ctypes.data,
                ctypes.addressof(sector_count),
            ),
            "cad_surface_pole_sectors",
        )
        chart_counts = np.empty((3,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_cad_chart_sizes"](handle, chart_counts.ctypes.data),
            "cad_chart_sizes",
        )
        if np.any(chart_counts < 0):
            raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "negative CAD chart size")
        chart_triangles, chart_uses, chart_chords = chart_counts
        chart_banks = (
            np.empty((chart_triangles, 3), dtype=np.int64),
            np.empty((chart_triangles,), dtype=np.int32),
            np.empty((chart_triangles, 3, 2), dtype=np.float64),
            np.empty((chart_triangles,), dtype=np.int64),
            np.empty((chart_triangles, 3), dtype=np.int64),
            np.empty((chart_uses,), dtype=np.int64),
            np.empty((chart_uses, 2), dtype=np.float64),
            np.empty((chart_chords,), dtype=np.int64),
            np.empty((chart_chords,), dtype=np.int64),
            np.empty((chart_chords, 2, 2), dtype=np.float64),
            np.empty((chart_chords, 2), dtype=np.float64),
        )
        chart_outputs = _CadChartArrays(
            *(int(n) for n in chart_counts), *(a.ctypes.data for a in chart_banks)
        )
        _raise_for_call(
            library["phx_mc_cad_chart_export"](handle, ctypes.addressof(chart_outputs)),
            "cad_chart_export",
        )
        initial_faces, retries = _cad_surface_refinement(library, handle)
        surface = CadSurfaceResult(
            status,
            *banks,
            pole_sectors,
            CadChartResult(*chart_banks),
            initial_faces,
            retries,
        )
        if volume is None:
            return surface, None
        return surface, _cad_volume_export(library, handle, int(model.curve_nodes.size))
    finally:
        library["phx_mc_cad_surface_result_free"](handle)
