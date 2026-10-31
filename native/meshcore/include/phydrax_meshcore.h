/*
 * Copyright © 2026 PHYDRA, Inc. All rights reserved.
 *
 * Plain C ABI of the Phydrax meshing core. All array inputs are C-contiguous.
 * Geometric decisions use the documented exact coordinate domain.
 * Functions returning int32_t report call status; batched functions also
 * report one item status per entry. No C++ exception crosses the ABI:
 * insufficient caller output storage reports PHX_MC_CAPACITY_EXCEEDED,
 * allocation failure reports PHX_MC_ALLOCATION_FAILURE, and any other
 * escaped failure reports PHX_MC_INTERNAL_ERROR.
 */
#ifndef PHYDRAX_MESHCORE_H
#define PHYDRAX_MESHCORE_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(PHX_MC_BUILDING)
#define PHX_MC_API __declspec(dllexport)
#else
#define PHX_MC_API __declspec(dllimport)
#endif
#else
#define PHX_MC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum phx_mc_status {
  PHX_MC_OK = 0,
  PHX_MC_INVALID_ARGUMENT = 1,       /* null pointer, negative or unaddressable count, bad extent */
  PHX_MC_NONFINITE_INPUT = 2,        /* NaN or infinity in coordinates/weights */
  PHX_MC_RANGE_ERROR = 3,            /* value outside the exact coordinate/weight domain */
  PHX_MC_DEGENERATE_INPUT = 4,       /* zero-measure cell, collinear/coplanar point set */
  PHX_MC_INVALID_INPUT = 5,          /* nonconvex polygon, bad index, zero normal */
  PHX_MC_CAPACITY_EXCEEDED = 6,      /* caller-provided output buffer is too small */
  PHX_MC_CONSTRAINT_INTERSECTION = 7,/* constraint segments cross in their interiors */
  PHX_MC_UNRESOLVED = 8,             /* finite numerical resolution did not decide */
  PHX_MC_INTERNAL_ERROR = 9,
  PHX_MC_ALLOCATION_FAILURE = 10,    /* ordinary native allocation failed */
  PHX_MC_INDEX_RANGE_ERROR = 11      /* result cannot be represented by native indices */
};

/* ------------------------------------------------------------------ identity */

PHX_MC_API const char* phx_mc_version(void);
/* SHA-256 of the library sources recorded at configure time. */
PHX_MC_API const char* phx_mc_build_hash(void);
PHX_MC_API void phx_mc_exact_domain(int32_t* min_exponent, int32_t* max_exponent);

/* ---------------------------------------------------------------- predicates
 * Inputs a, b, ... are (count, 2) or (count, 3) arrays; signs is (count,) with
 * values in {-1, 0, 1}.
 *   orient2d(a, b, c)       = sign det[b - a, c - a]         (> 0 counterclockwise)
 *   orient3d(a, b, c, d)    = sign det[b - a, c - a, d - a]  (> 0 right-handed)
 *   incircle(a, b, c, d)    > 0 iff d is inside the circle of CCW (a, b, c)
 *   insphere(a, b, c, d, e) > 0 iff e is inside the sphere of positively oriented (a..d)
 * The *_sos variants take ids of shape (count, k) (distinct per row) and apply
 * index-ordered Simulation of Simplicity: orient*_sos never returns 0;
 * incircle_sos/insphere_sos return 0 only when all points are collinear/coplanar.
 */
PHX_MC_API int32_t phx_mc_orient2d(int64_t count, const double* a, const double* b,
                                   const double* c, int8_t* signs);
PHX_MC_API int32_t phx_mc_orient3d(int64_t count, const double* a, const double* b,
                                   const double* c, const double* d, int8_t* signs);
PHX_MC_API int32_t phx_mc_incircle(int64_t count, const double* a, const double* b,
                                   const double* c, const double* d, int8_t* signs);
PHX_MC_API int32_t phx_mc_insphere(int64_t count, const double* a, const double* b,
                                   const double* c, const double* d, const double* e,
                                   int8_t* signs);
PHX_MC_API int32_t phx_mc_orient2d_sos(int64_t count, const double* a, const double* b,
                                       const double* c, const int64_t* ids, int8_t* signs);
PHX_MC_API int32_t phx_mc_orient3d_sos(int64_t count, const double* a, const double* b,
                                       const double* c, const double* d, const int64_t* ids,
                                       int8_t* signs);
PHX_MC_API int32_t phx_mc_incircle_sos(int64_t count, const double* a, const double* b,
                                       const double* c, const double* d, const int64_t* ids,
                                       int8_t* signs);
PHX_MC_API int32_t phx_mc_insphere_sos(int64_t count, const double* a, const double* b,
                                       const double* c, const double* d, const double* e,
                                       const int64_t* ids, int8_t* signs);

/* ------------------------------------------------------------------ clipping
 * Item status: PHX_MC_OK (including empty and zero-measure contact, reported
 * with zero measure), PHX_MC_NONFINITE_INPUT, PHX_MC_RANGE_ERROR,
 * PHX_MC_DEGENERATE_INPUT, PHX_MC_INVALID_INPUT, PHX_MC_CAPACITY_EXCEEDED,
 * PHX_MC_INTERNAL_ERROR.  Measures and first moments of failed items are zero.
 * Vertex classifications are exact: every vertex of an intermediate polytope is
 * represented symbolically by its defining input lines/planes.  Halfspace
 * normal components follow the coordinate domain and offsets h the weight
 * domain (both are degree-one and degree-two quantities of a plane n . x = h).
 */

/* Convex polygons (either orientation; repeated consecutive vertices allowed),
 * first_vertices (count, first_capacity, 2), first_counts (count,).
 * Outputs: areas (count,), first_moments (count, 2) = integral of x over the
 * intersection. */
PHX_MC_API int32_t phx_mc_polygon_intersection_moments(
    int64_t count, int32_t first_capacity, const double* first_vertices,
    const int32_t* first_counts, int32_t second_capacity, const double* second_vertices,
    const int32_t* second_counts, double* areas, double* first_moments, int32_t* item_status);

/* Tetrahedra (count, 4, 3), either orientation. */
PHX_MC_API int32_t phx_mc_tetrahedron_intersection_moments(
    int64_t count, const double* first, const double* second, double* volumes,
    double* first_moments, int32_t* item_status);

/* Simplex partitions of the intersections above, for quadrature on the common
 * refinement.  Each nonempty intersection is fanned from the vertex average o of
 * its vertices (the same fan as the moments, so the signed simplex measures sum
 * to the reported measure term by term): polygon intersections as triangles
 * (o, v_i, v_{i+1}) over the counterclockwise boundary loop, simplices
 * (count, simplex_capacity, 3, 2); tetrahedron intersections as tetrahedra
 * (o, x_0, x_i, x_{i+1}) over the fan triangles of every outward face,
 * simplices (count, simplex_capacity, 4, 3).  Simplices are positively oriented
 * up to the rounding of o.  simplex_counts (count,) receives the number written;
 * empty and zero-measure intersections write none.  An intersection needing more
 * than simplex_capacity simplices reports PHX_MC_CAPACITY_EXCEEDED:
 * first_capacity + second_capacity always suffices for polygons and 20 for
 * tetrahedra. */
PHX_MC_API int32_t phx_mc_polygon_intersection_simplices(
    int64_t count, int32_t first_capacity, const double* first_vertices,
    const int32_t* first_counts, int32_t second_capacity, const double* second_vertices,
    const int32_t* second_counts, int32_t simplex_capacity, double* simplices,
    int32_t* simplex_counts, double* areas, double* first_moments, int32_t* item_status);
PHX_MC_API int32_t phx_mc_tetrahedron_intersection_simplices(
    int64_t count, const double* first, const double* second, int32_t simplex_capacity,
    double* simplices, int32_t* simplex_counts, double* volumes, double* first_moments,
    int32_t* item_status);

/* Convex polyhedra given as halfspaces {x : n . x <= h}: normals
 * (count, plane_capacity, 3), offsets (count, plane_capacity), plane_counts
 * (count,), intersected with tetrahedra (count, 4, 3). */
PHX_MC_API int32_t phx_mc_polyhedron_clip_moments(
    int64_t count, int32_t plane_capacity, const double* normals, const double* offsets,
    const int32_t* plane_counts, const double* tetrahedra, double* volumes,
    double* first_moments, int32_t* item_status);

/* Axis-aligned box [box_lower, box_upper] (2,) clipped by halfplanes
 * {x : n . x <= h} per item.  Outputs the CCW cell polygon: vertices
 * (count, vertex_capacity, 2), edge_labels (count, vertex_capacity) where
 * edge_labels[k] labels the edge from vertex k to vertex k + 1 (cyclic) with the
 * item-local plane index (>= 0) or -(1 + 2 axis + side) for box sides
 * (side 0 lower, 1 upper), vertex_counts (count,), areas, first_moments (count, 2). */
PHX_MC_API int32_t phx_mc_clip_box_halfplanes(
    int64_t count, const double* box_lower, const double* box_upper, int32_t plane_capacity,
    const double* normals, const double* offsets, const int32_t* plane_counts,
    int32_t vertex_capacity, double* vertices, int32_t* edge_labels, int32_t* vertex_counts,
    double* areas, double* first_moments, int32_t* item_status);

/* Axis-aligned box (3,) clipped by halfspaces per item.  Outputs a closed
 * polyhedron: vertices (count, vertex_capacity, 3), vertex_counts (count,),
 * faces as per-item CSR face_offsets (count, face_capacity + 1) into
 * face_vertices (count, face_vertex_capacity), face_labels (count, face_capacity)
 * using the labels of phx_mc_clip_box_halfplanes, face_counts (count,).
 * Faces are sorted by label; each face loop is counterclockwise seen from
 * outside and starts at its smallest vertex index. */
PHX_MC_API int32_t phx_mc_clip_box_halfspaces(
    int64_t count, const double* box_lower, const double* box_upper, int32_t plane_capacity,
    const double* normals, const double* offsets, const int32_t* plane_counts,
    int32_t vertex_capacity, int32_t face_capacity, int32_t face_vertex_capacity,
    double* vertices, int32_t* vertex_counts, int32_t* face_offsets, int32_t* face_labels,
    int32_t* face_vertices, int32_t* face_counts, double* volumes, double* first_moments,
    int32_t* item_status);

/* ------------------------------------------------------------ triangulations
 * Results are returned as an owned mesh handle (NULL on failure).  Cells are
 * positively oriented (CCW triangles, right-handed tetrahedra), each cell starts
 * at its smallest vertex (orientation-preserving rotation) and cells are sorted
 * lexicographically.  vertex_map (input_point_count,) maps each input point to
 * its triangulation vertex: itself, the smallest index of an identical point
 * (identical weight for regular triangulations), or -1 for a redundant weighted
 * point.  Delaunay ties are resolved by index-ordered symbolic perturbation.
 * Point counts must be below 2^31 - 1.  DEGENERATE_INPUT is returned when the
 * distinct points do not span the plane/space.
 */
typedef struct phx_mc_mesh phx_mc_mesh;

PHX_MC_API int32_t phx_mc_delaunay_2d(int64_t point_count, const double* points,
                                      phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_regular_2d(int64_t point_count, const double* points,
                                     const double* weights, phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_delaunay_3d(int64_t point_count, const double* points,
                                      phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_regular_3d(int64_t point_count, const double* points,
                                     const double* weights, phx_mc_mesh** mesh);

/* Constrained Delaunay triangulation of points (n, 2) and segments (m, 2)
 * (input point indices), refined to the requested angle and area criteria.
 * keep_convex_hull != 0 meshes the convex hull; otherwise outside and hole
 * regions are removed. Segment splits retain their input segment, parent-edge
 * endpoint rows, and split ratio as per-point ancestry. Returns UNRESOLVED
 * with a valid conforming mesh when finite numerical resolution prevents
 * satisfying a requested criterion. */
PHX_MC_API int32_t phx_mc_constrained_delaunay_2d(
    int64_t point_count, const double* points, int64_t segment_count, const int32_t* segments,
    int64_t hole_count, const double* holes, int32_t keep_convex_hull, double min_angle_degrees,
    double max_area, phx_mc_mesh** mesh);

/* Scalar shape queries return -1 for a null handle. Copy capacities are row
 * extents and are validated before writes; zero-sized banks accept null outputs. */
PHX_MC_API int32_t phx_mc_mesh_dimension(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_point_count(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_cell_count(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_input_point_count(const phx_mc_mesh* mesh);
PHX_MC_API int32_t phx_mc_mesh_copy_points(
    const phx_mc_mesh* mesh, int64_t point_capacity, double* points);
PHX_MC_API int32_t phx_mc_mesh_copy_cells(
    const phx_mc_mesh* mesh, int64_t cell_capacity, int32_t* cells);
PHX_MC_API int32_t phx_mc_mesh_copy_vertex_map(
    const phx_mc_mesh* mesh, int64_t point_capacity, int32_t* vertex_map);
/* (cell_count, 3); filled with -1 for meshes without constraints. */
PHX_MC_API int32_t phx_mc_mesh_copy_cell_segments(
    const phx_mc_mesh* mesh, int64_t cell_capacity, int32_t* cell_segments);
/* One source segment, parent-edge pair and split ratio per mesh point. */
PHX_MC_API int32_t phx_mc_mesh_copy_constraint_sources(
    const phx_mc_mesh* mesh, int64_t point_capacity, int32_t* sources);
PHX_MC_API int32_t phx_mc_mesh_copy_constraint_parents(
    const phx_mc_mesh* mesh, int64_t point_capacity, int32_t* parents);
PHX_MC_API int32_t phx_mc_mesh_copy_constraint_ratios(
    const phx_mc_mesh* mesh, int64_t point_capacity, double* ratios);
PHX_MC_API void phx_mc_mesh_free(phx_mc_mesh* mesh);

/* ------------------------------------- retained native numerical kernels */

#define PHX_MC_TET_MESH_BOUNDARY_FIXED 0
#define PHX_MC_TET_MESH_BOUNDARY_CONFORMING 1
#define PHX_MC_SOURCE_STRATUM_NONE 0
#define PHX_MC_SOURCE_STRATUM_SEGMENT 1
#define PHX_MC_SOURCE_STRATUM_FACET 2
/* A point origin + s * first_axis + t * second_axis of the declared exact
 * plane of a source face row; parameters (s, t) are any binary64 pair. */
#define PHX_MC_SOURCE_STRATUM_PLANE 3

/* Persistent source-aware tetrahedral state. Creation borrows every input
 * synchronously; the returned handle owns all mesh and source banks. */
typedef struct phx_mc_tet_mesh phx_mc_tet_mesh;
enum phx_mc_tet_edit_kind {
  PHX_MC_TET_EDIT_SPLIT = 0,
  PHX_MC_TET_EDIT_COLLAPSE = 1,
  PHX_MC_TET_EDIT_FLIP23 = 2,
  PHX_MC_TET_EDIT_FLIP32 = 3,
  PHX_MC_TET_EDIT_RECONNECT = 4,
  PHX_MC_TET_EDIT_RELOCATE = 5,
  PHX_MC_TET_EDIT_REFINE_CELL = 6,
  PHX_MC_TET_EDIT_RETRIANGULATE = 7
};
enum phx_mc_tet_edit_status {
  PHX_MC_TET_EDIT_APPLIED = 0,
  PHX_MC_TET_EDIT_REJECTED_TOPOLOGY = 1,
  PHX_MC_TET_EDIT_REJECTED_PROTECTED = 2,
  PHX_MC_TET_EDIT_INVALID_GEOMETRY = 3,
  PHX_MC_TET_EDIT_UNCERTAIN_PREDICATE = 4,
  PHX_MC_TET_EDIT_STALE_GENERATION = 5,
  PHX_MC_TET_EDIT_ALLOCATION_FAILURE = 6,
  PHX_MC_TET_EDIT_INDEX_OVERFLOW = 7
};
enum phx_mc_tet_unmet_criterion {
  PHX_MC_TET_UNMET_RADIUS_EDGE = 0,
  PHX_MC_TET_UNMET_SIZE = 1,
  PHX_MC_TET_UNMET_DIHEDRAL = 2,
  PHX_MC_TET_UNMET_CONSTRUCTION = 3,
  PHX_MC_TET_UNMET_VALIDITY = 4
};
/* INPUT_ANGLE: conforming refinement left the cell because the boundary split
 * it required falls in the protected region of an acute input feature
 * (segments, or a segment and a facet, meeting below 90 degrees, or facets
 * meeting at a dihedral angle below 90 degrees through the domain). */
enum phx_mc_tet_unmet_reason {
  PHX_MC_TET_UNMET_FIXED_BOUNDARY = 0,
  PHX_MC_TET_UNMET_PROTECTED = 1,
  PHX_MC_TET_UNMET_NONREPRESENTABLE = 2,
  PHX_MC_TET_UNMET_REFUSED = 3,
  PHX_MC_TET_UNMET_NO_IMPROVEMENT = 4,
  PHX_MC_TET_UNMET_INPUT_ANGLE = 5
};
/* Apex vertex ID of an unmet row without INPUT_ANGLE evidence. */
#define PHX_MC_TET_UNMET_NO_APEX UINT64_MAX
PHX_MC_API int32_t phx_mc_tet_mesh_unmet_count(const phx_mc_tet_mesh* mesh,
                                                int64_t* count);
/* One row per unmet (cell, criterion). INPUT_ANGLE rows carry the acute
 * feature's apex vertex ID (a ridge's first endpoint) and its angle in
 * degrees; other rows PHX_MC_TET_UNMET_NO_APEX and NaN. */
PHX_MC_API int32_t phx_mc_tet_mesh_unmet(
    const phx_mc_tet_mesh* mesh, int64_t capacity, uint64_t* cell_ids,
    int32_t* criteria, int32_t* reasons, double* values, uint64_t* apex_vertex_ids,
    double* apex_angles);
#define PHX_MC_TET_MESH_SNAPSHOT_SIZES 19
typedef struct phx_mc_tet_periodic_vertex_member {
  uint64_t vertex_id;
  int64_t winding[3];
  double representative_to_member[12];
} phx_mc_tet_periodic_vertex_member;
typedef struct phx_mc_tet_periodic_orbit {
  uint64_t id;
  uint64_t representative_vertex_id;
  int64_t member_offset;
  int64_t member_count;
} phx_mc_tet_periodic_orbit;
typedef struct phx_mc_tet_periodic_quotient_cell {
  /* Cell-orbit grouping key anchored to a dimension-3 quotient entity ID. */
  uint64_t orbit_id;
  /* Stable dimension-3 quotient entity ID from the identity map. */
  uint64_t id;
  uint64_t representative_vertices[4];
  int64_t winding[12];
  int32_t region;
  int64_t lifted_cell_offset;
  int64_t lifted_cell_count;
} phx_mc_tet_periodic_quotient_cell;
typedef struct phx_mc_tet_periodic_entity_orbit_map {
  int32_t dimension;
  int32_t arity;
  int32_t rank;
  int64_t lifted_offset;
  int64_t lifted_count;
  int64_t quotient_offset;
  int64_t quotient_count;
  int64_t winding_offset;
  int64_t corner_permutation_offset;
  int64_t representative_offset;
} phx_mc_tet_periodic_entity_orbit_map;
typedef struct phx_mc_tet_periodic_identity_map_view {
  int32_t identity_bound;
  const char* source_mesh_id;
  int64_t source_mesh_id_length;
  const char* source_topology_id;
  int64_t source_topology_id_length;
  const char* lifted_topology_id;
  int64_t lifted_topology_id_length;
  const char* periodic_topology_id;
  int64_t periodic_topology_id_length;
  const char* cell_entity_set_id;
  int64_t cell_entity_set_id_length;
  const char* edge_entity_set_id;
  int64_t edge_entity_set_id_length;
  int64_t entity_map_count;
  const phx_mc_tet_periodic_entity_orbit_map* entity_maps;
  int64_t entity_lifted_count;
  const uint64_t* entity_lifted_ids;
  const int32_t* entity_lifted_to_quotient;
  const int32_t* entity_orientation_signs;
  int64_t entity_quotient_count;
  const uint64_t* entity_quotient_ids;
  int64_t entity_winding_count;
  const int64_t* entity_winding_shifts;
  int64_t entity_corner_count;
  const int32_t* entity_corner_permutations;
  int64_t entity_representative_count;
  const int32_t* entity_representative_rows;
} phx_mc_tet_periodic_identity_map_view;
typedef struct phx_mc_tet_periodic_state_view {
  phx_mc_tet_periodic_identity_map_view identity_map;
  int64_t periodic_orbit_count;
  const phx_mc_tet_periodic_orbit* periodic_orbits;
  int64_t periodic_member_count;
  const phx_mc_tet_periodic_vertex_member* periodic_members;
  int64_t periodic_quotient_cell_count;
  const phx_mc_tet_periodic_quotient_cell* periodic_quotient_cells;
  int64_t periodic_quotient_lifted_count;
  const uint64_t* periodic_quotient_lifted_cell_ids;
} phx_mc_tet_periodic_state_view;
typedef struct phx_mc_tet_periodic_edit_view {
  int64_t periodic_member_count;
  const phx_mc_tet_periodic_vertex_member* periodic_members;
  int64_t removed_periodic_orbit_count;
  const uint64_t* removed_periodic_orbit_ids;
  int64_t replacement_periodic_orbit_count;
  const phx_mc_tet_periodic_orbit* replacement_periodic_orbits;
  int64_t replacement_periodic_member_count;
  const phx_mc_tet_periodic_vertex_member* replacement_periodic_members;
  int64_t removed_quotient_cell_count;
  const uint64_t* removed_quotient_cell_ids;
  int64_t replacement_quotient_cell_count;
  const phx_mc_tet_periodic_quotient_cell* replacement_quotient_cells;
  int64_t replacement_quotient_lifted_count;
  const uint64_t* replacement_quotient_lifted_cell_ids;
  const phx_mc_tet_periodic_identity_map_view* target_identity_map;
} phx_mc_tet_periodic_edit_view;
/* Refusal of a declared facet plane by input vertex `vertex` of one of its
 * constrained facets: the plane neither meets the closed rounding box of the
 * vertex carrier nor lies within the declared tolerance of it. The record:
 * the plane's (first) source face row, the correctly rounded parameters of
 * the carrier's orthogonal projection onto the plane, and an outward bound
 * of the carrier's distance to it. */
typedef struct phx_mc_tet_plane_refusal {
  int32_t refused;
  int32_t face_row;
  int64_t vertex;
  double parameters[2];
  double deviation;
  double tolerance;
} phx_mc_tet_plane_refusal;
typedef struct phx_mc_tet_mesh_create_request {
  int64_t point_count;
  const double* points;
  const uint64_t* vertex_ids;
  int64_t tetrahedron_count;
  const int32_t* tetrahedra;
  const int32_t* tetrahedron_regions;
  const uint64_t* cell_ids;
  int64_t face_count;
  const int32_t* faces;
  const int32_t* face_sources;
  const int8_t* face_orientations;
  const uint64_t* facet_ids;
  int64_t segment_count;
  const int32_t* segments;
  const int32_t* segment_sources;
  const uint64_t* segment_ids;
  const double* protection_radii;
  const int8_t* protected_vertices;
  int64_t source_point_count;
  const double* source_points;
  int64_t source_face_count;
  const int32_t* source_faces;
  const int32_t* source_face_groups;
  const int64_t* source_face_ids;
  const double* source_face_tolerances;
  int64_t source_segment_count;
  const int32_t* source_segments;
  const int32_t* source_segment_groups;
  const int64_t* source_segment_ids;
  const double* source_segment_tolerances;
  /* Declared exact facet planes, one row of nine values per source face row
   * (origin, first axis, second axis; an all-NaN row is undeclared), and
   * their declared tolerances (null: zero). Null: no facet declares a plane.
   * Every row of a face group declares the same plane or none. */
  const double* source_face_planes;
  const double* source_face_plane_tolerances;
  /* witness_entities indexes source_segments for stratum 1 and source_faces
   * for strata 2 and 3 (stratum 3: a face row that declares a plane). */
  const int8_t* witness_strata;
  const int32_t* witness_entities;
  const double* witness_parameters;
  const double* witness_deviations;
  int32_t boundary_policy;
  int32_t identity_bound;
  const char* source_id;
  int64_t source_id_length;
  const char* source_revision;
  int64_t source_revision_length;
  const char* source_map_id;
  int64_t source_map_id_length;
  const char* frame_id;
  int64_t frame_id_length;
  const phx_mc_tet_periodic_state_view* periodic_state;
  /* Optional: filled when creation refuses a declared plane. */
  phx_mc_tet_plane_refusal* plane_refusal;
  phx_mc_tet_mesh** result;
} phx_mc_tet_mesh_create_request;
PHX_MC_API int32_t phx_mc_tet_mesh_create(const phx_mc_tet_mesh_create_request* request);
PHX_MC_API int32_t phx_mc_tet_mesh_snapshot_sizes(const phx_mc_tet_mesh* mesh,
                                                  int64_t* sizes);
PHX_MC_API int32_t phx_mc_tet_mesh_identity_sizes(
    const phx_mc_tet_mesh* mesh, uint64_t* generation, int32_t* identity_bound,
    int64_t* source_id_length, int64_t* source_revision_length,
    int64_t* source_map_id_length, int64_t* frame_id_length);
PHX_MC_API int32_t phx_mc_tet_mesh_identity(
    const phx_mc_tet_mesh* mesh, int64_t source_id_capacity, char* source_id,
    int64_t source_revision_capacity, char* source_revision,
    int64_t source_map_id_capacity, char* source_map_id,
    int64_t frame_id_capacity, char* frame_id);
PHX_MC_API int32_t phx_mc_tet_mesh_periodic_identity_sizes(
    const phx_mc_tet_mesh* mesh, int32_t* identity_bound,
    int64_t* source_mesh_id_length, int64_t* source_topology_id_length,
    int64_t* lifted_topology_id_length, int64_t* periodic_topology_id_length,
    int64_t* cell_entity_set_id_length, int64_t* edge_entity_set_id_length);
PHX_MC_API int32_t phx_mc_tet_mesh_periodic_identity(
    const phx_mc_tet_mesh* mesh,
    int64_t source_mesh_id_capacity, char* source_mesh_id,
    int64_t source_topology_id_capacity, char* source_topology_id,
    int64_t lifted_topology_id_capacity, char* lifted_topology_id,
    int64_t periodic_topology_id_capacity, char* periodic_topology_id,
    int64_t cell_entity_set_id_capacity, char* cell_entity_set_id,
    int64_t edge_entity_set_id_capacity, char* edge_entity_set_id);
typedef struct phx_mc_tet_mesh_snapshot_request {
  int64_t vertex_capacity;
  int64_t cell_capacity;
  int64_t facet_capacity;
  int64_t segment_capacity;
  int64_t source_point_capacity;
  int64_t source_face_capacity;
  int64_t source_segment_capacity;
  int64_t periodic_orbit_capacity;
  int64_t periodic_member_capacity;
  int64_t periodic_quotient_cell_capacity;
  int64_t periodic_quotient_lifted_capacity;
  int64_t periodic_entity_map_capacity;
  int64_t periodic_entity_lifted_capacity;
  int64_t periodic_entity_quotient_capacity;
  int64_t periodic_entity_winding_capacity;
  int64_t periodic_entity_corner_capacity;
  int64_t periodic_entity_representative_capacity;
  int64_t vertex_ancestry_capacity;
  int64_t cell_ancestor_capacity;
  uint64_t* vertex_ids;
  double* points;
  int8_t* witness_strata;
  int32_t* witness_entities;
  double* witness_parameters;
  double* witness_deviations;
  int8_t* vertex_dimensions;
  double* protection_radii;
  int8_t* protected_vertices;
  int8_t* vertex_edge_witness_present;
  uint64_t* vertex_edge_witness_endpoints;
  double* vertex_edge_parameters;
  int64_t* vertex_ancestry_offsets;
  uint64_t* vertex_ancestor_ids;
  int64_t* vertex_ancestor_winding;
  double* vertex_ancestry_weights;
  phx_mc_tet_periodic_orbit* periodic_orbits;
  phx_mc_tet_periodic_vertex_member* periodic_members;
  phx_mc_tet_periodic_quotient_cell* periodic_quotient_cells;
  uint64_t* periodic_quotient_lifted_cell_ids;
  phx_mc_tet_periodic_entity_orbit_map* periodic_entity_maps;
  uint64_t* periodic_entity_lifted_ids;
  uint64_t* periodic_entity_quotient_ids;
  int32_t* periodic_entity_lifted_to_quotient;
  int32_t* periodic_entity_orientation_signs;
  int64_t* periodic_entity_winding_shifts;
  int32_t* periodic_entity_corner_permutations;
  int32_t* periodic_entity_representative_rows;
  int64_t* cell_ancestry_offsets;
  uint64_t* cell_ancestor_ids;
  int32_t* cell_ancestry_edit_kinds;
  uint64_t* cell_ids;
  uint64_t* cell_vertices;
  int32_t* cell_regions;
  uint64_t* facet_ids;
  uint64_t* facet_vertices;
  int32_t* facet_sources;
  int8_t* facet_orientations;
  int8_t* protected_facets;
  uint64_t* segment_ids;
  uint64_t* segment_vertices;
  int32_t* segment_sources;
  int8_t* protected_segments;
  double* source_points;
  int32_t* source_faces;
  int32_t* source_face_groups;
  int64_t* source_face_ids;
  double* source_face_tolerances;
  int32_t* source_segments;
  int32_t* source_segment_groups;
  double* source_segment_tolerances;
  int64_t* source_segment_ids;
  /* Nine values per source face row (NaN rows undeclared) and one tolerance. */
  double* source_face_planes;
  double* source_face_plane_tolerances;
  uint64_t* generation;
} phx_mc_tet_mesh_snapshot_request;
PHX_MC_API int32_t phx_mc_tet_mesh_snapshot(
    const phx_mc_tet_mesh* mesh, const phx_mc_tet_mesh_snapshot_request* request);
/* Plane records: one row per (live vertex, declared face group) incidence of
 * the constrained facets, ordered by vertex ID then face row: the vertex ID,
 * the group's first source face row, the correctly rounded parameters of the
 * carrier's orthogonal projection onto the declared plane, and an outward
 * bound of the carrier's distance to it. Every recorded carrier is admitted
 * (rounding box or declared tolerance). */
PHX_MC_API int32_t phx_mc_tet_mesh_plane_record_count(const phx_mc_tet_mesh* mesh,
                                                      int64_t* count);
PHX_MC_API int32_t phx_mc_tet_mesh_plane_records(
    const phx_mc_tet_mesh* mesh, int64_t capacity, uint64_t* vertex_ids, int32_t* face_rows,
    double* parameters, double* deviations);
typedef struct phx_mc_tet_mesh_edit_request {
  uint64_t expected_generation;
  int32_t identity_bound;
  const char* source_id;
  int64_t source_id_length;
  const char* source_revision;
  int64_t source_revision_length;
  const char* source_map_id;
  int64_t source_map_id_length;
  const char* frame_id;
  int64_t frame_id_length;
  const phx_mc_tet_periodic_edit_view* periodic;
  int32_t kind;
  int64_t collapse_pair_count;
  const uint64_t* collapse_removed_ids;
  const uint64_t* collapse_survivor_ids;
  int64_t split_edge_count;
  const uint64_t* split_edges;
  int64_t removed_vertex_count;
  const uint64_t* removed_vertex_ids;
  int64_t removed_cell_count;
  const uint64_t* removed_cell_ids;
  int64_t changed_vertex_count;
  const uint64_t* changed_vertex_ids;
  const double* changed_vertex_points;
  const int8_t* changed_witness_strata;
  const int32_t* changed_witness_entities;
  const double* changed_witness_parameters;
  const double* changed_witness_deviations;
  const double* changed_protection_radii;
  const int8_t* changed_protected_vertices;
  const int8_t* changed_edge_witness_present;
  const uint64_t* changed_edge_witness_endpoints;
  const double* changed_edge_parameters;
  int64_t changed_vertex_ancestry_count;
  const int64_t* changed_vertex_ancestry_offsets;
  const uint64_t* changed_vertex_ancestor_ids;
  const int64_t* changed_vertex_ancestor_winding;
  const double* changed_vertex_ancestry_weights;
  int64_t replacement_cell_count;
  const uint64_t* replacement_cell_ids;
  const uint64_t* replacement_cell_vertices;
  const int32_t* replacement_cell_facet_sources; /* Optional; defaults to no constraint. */
  const int32_t* replacement_regions;
  int64_t removed_facet_count;
  int64_t replacement_cell_ancestry_count;
  const int64_t* replacement_cell_ancestry_offsets;
  const uint64_t* replacement_cell_ancestor_ids;
  const uint64_t* removed_facet_ids;
  int64_t replacement_facet_count;
  const uint64_t* replacement_facet_ids;
  const uint64_t* replacement_facet_vertices;
  const int32_t* replacement_facet_sources;
  const int8_t* replacement_facet_orientations; /* Optional; defaults to +1. */
  const int8_t* replacement_protected_facets;
  int64_t removed_segment_count;
  const uint64_t* removed_segment_ids;
  int64_t replacement_segment_count;
  const uint64_t* replacement_segment_ids;
  const uint64_t* replacement_segment_vertices;
  const int32_t* replacement_segment_sources;
  const int8_t* replacement_protected_segments;
} phx_mc_tet_mesh_edit_request;
PHX_MC_API int32_t phx_mc_tet_mesh_apply_edit(
    phx_mc_tet_mesh* mesh, const phx_mc_tet_mesh_edit_request* request,
    int32_t* edit_status);
/* Refine inserts circumcenters until every cell meets the radius-edge bound
 * and its circumradius size target. `sizes` (one per current vertex in
 * snapshot order, 0 = unconstrained; empty keeps the mesh's retained field)
 * becomes the mesh's retained size field; inserted vertices receive
 * interpolated targets. */
PHX_MC_API int32_t phx_mc_tet_mesh_refine(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length,
    const char* source_revision, int64_t source_revision_length,
    const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length, double radius_edge_bound,
    int64_t size_count, const double* sizes, int32_t* edit_status,
    int32_t* run_status, int64_t* inserted_count, int64_t* unmet_count);
/* Improve preserves every constrained face/segment and region. Flips,
 * collapses, relocations, and interior vertex insertions are accepted by one
 * sorted-score order; each cell's score is floored to multiples of
 * `*score_quantum` (reported, 2^-20). The determinant floor gates every edit.
 * No edit creates a cell whose circumradius exceeds its size target by more
 * than the largest excess among the cells it removes (none where they were
 * within target), except one removing a degenerate cell (below the
 * determinant floor or scoring 0), whose repair takes precedence. `sizes`
 * follows refine (empty keeps the retained field, e.g. refine's). The run
 * status is UNRESOLVED while the radius-edge or dihedral target, or the
 * determinant floor, remains unmet;
 * `phx_mc_tet_mesh_unmet` reports each residual criterion. The size field is
 * a constraint on improvement, not one of its targets. */
PHX_MC_API int32_t phx_mc_tet_mesh_improve_quality(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length,
    const char* source_revision, int64_t source_revision_length,
    const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length,
    double radius_edge_bound, double minimum_dihedral_degrees,
    double minimum_relative_determinant, int64_t size_count, const double* sizes,
    int32_t* edit_status, int32_t* run_status, int64_t* inserted_vertices,
    int64_t* collapsed_vertices, int64_t* relocated_vertices, int64_t* unmet_cells,
    double* score_quantum);
PHX_MC_API void phx_mc_tet_mesh_free(phx_mc_tet_mesh* mesh);

#define PHX_MC_QUADRATIC_VALIDITY_VALID 0
#define PHX_MC_QUADRATIC_VALIDITY_INVALID 1
#define PHX_MC_QUADRATIC_VALIDITY_UNRESOLVED 2
PHX_MC_API int32_t phx_mc_interval_program_execute(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, double* lower,
    double* upper, int64_t workspace_size);
/* Bit n advertises interval-program opcode n; bit 0 is reserved. */
PHX_MC_API int32_t phx_mc_interval_program_capabilities(uint64_t* opcodes);
PHX_MC_API int32_t phx_mc_simplex_quadratic_bounds(
    int64_t count, int32_t dimension, int32_t components, const double* q_lower,
    const double* q_upper, const double* barycentric_lower, const double* barycentric_upper,
    double* lower, double* upper);
PHX_MC_API int32_t phx_mc_quadratic_tetrahedron_validity(
    int64_t count, const double* nodes, const int32_t* derivative_weights, double relative_floor,
    int32_t* decisions, double* lower, double* upper);

PHX_MC_API int32_t phx_mc_graph_partition(
    int64_t vertex_count, int64_t edge_count, const int64_t* offsets,
    const int32_t* neighbors, const int64_t* edge_weights, const int64_t* vertex_weights,
    int32_t part_count, const int64_t* part_targets, const int64_t* part_capacities,
    int32_t require_nonempty, int32_t refinement_passes, int32_t* parts);
/* Invoke disjoint indexed chunks through the native process-wide worker pool.
 * The callback receives [begin, end) and a stable zero-based chunk index.
 * Chunks all complete; the call returns the first non-OK callback status in
 * chunk order. A zero count invokes no callbacks but still validates the
 * request. */
typedef int32_t (*phx_mc_parallel_chunk_callback)(
    void* context, int64_t begin, int64_t end, int64_t chunk_index);
PHX_MC_API int32_t phx_mc_parallel_for(
    int64_t count, int64_t chunk_size, int32_t parallelism,
    phx_mc_parallel_chunk_callback callback, void* context);
/* Barycentric dual of each positively oriented tetrahedron. nodes is (T,15):
 * four cell vertices, six edge-centroid IDs in (01,02,03,12,13,23) order,
 * four face-centroid IDs in (012,013,023,123) order, and one cell-centroid ID.
 * The output contains four orientation-preserving hexahedra per tetrahedron;
 * hex_capacity is a row capacity for (H,8) IDs. */
PHX_MC_API int32_t phx_mc_tetrahedron_dual_hexes(
    int64_t tetrahedron_count, int64_t vertex_count, const int64_t* nodes,
    int64_t hex_capacity, int64_t* hexes);
/* Barycentric dual quadrilaterals of positively oriented triangles. nodes is
 * (T,7): three vertices, shared edge-centroid IDs (01,12,20), and a
 * triangle-centroid ID. The output contains three cyclic quads per triangle. */
PHX_MC_API int32_t phx_mc_triangle_dual_quads(
    int64_t triangle_count, int64_t vertex_count, const int64_t* nodes,
    int64_t quad_capacity, int64_t* quads);

/* ------------------------------------------------ restricted power cells */
#define PHX_MC_POWER_CELLS_SIZES 5
enum phx_mc_power_failure_reason {
  PHX_MC_POWER_NO_FAILURE = 0,
  PHX_MC_POWER_DEGENERATE_VERTEX = 1,
  PHX_MC_POWER_MISSING_PIECE = 2,
  PHX_MC_POWER_INCONSISTENT_FACE = 3,
  PHX_MC_POWER_INVALID_DOMAIN = 4,
  PHX_MC_POWER_CLIP_FAILURE = 5,
  PHX_MC_POWER_SEED_FAILURE = 6,
  PHX_MC_POWER_IMAGE_SOURCE_DOMAIN = 7,
  PHX_MC_POWER_OUTPUT_COUNT_OVERFLOW = 8
};
typedef struct phx_mc_power_cells phx_mc_power_cells;
typedef struct phx_mc_restricted_power_cells_request {
  int64_t site_count;
  const double* sites;
  const double* weights;
  const int64_t* neighbor_offsets;
  const int32_t* neighbors;
  const int8_t* split_sites;
  int64_t point_count;
  const double* points;
  int64_t tet_count;
  const int32_t* tets;
  const int32_t* tet_regions;
  const int32_t* tet_face_facets;
  phx_mc_power_cells** result;
} phx_mc_restricted_power_cells_request;
typedef struct phx_mc_restricted_power_cells_exact_request {
  int64_t original_site_count;
  const double* original_sites;
  const double* original_weights;
  int64_t image_count;
  const int32_t* image_site_owners;
  const int64_t* image_coordinate_offsets;
  int64_t component_count;
  const double* image_coordinate_components;
  const int64_t* neighbor_offsets;
  const int32_t* neighbors;
  const int8_t* split_sites;
  int64_t point_count;
  const double* points;
  int64_t tet_count;
  const int32_t* tets;
  const int32_t* tet_regions;
  const int32_t* tet_face_facets;
  phx_mc_power_cells** result;
  int64_t* failed_image_coordinate;
  int64_t* failed_component;
} phx_mc_restricted_power_cells_exact_request;
PHX_MC_API int32_t phx_mc_restricted_power_cells(
    const phx_mc_restricted_power_cells_request* request);
PHX_MC_API int32_t phx_mc_restricted_power_cells_exact(
    const phx_mc_restricted_power_cells_exact_request* request);
PHX_MC_API int32_t phx_mc_power_cells_sizes(const phx_mc_power_cells* result, int64_t* sizes);
PHX_MC_API int32_t phx_mc_power_cells_export(
    const phx_mc_power_cells* result, int64_t vertex_capacity, int64_t face_offset_capacity,
    int64_t face_capacity, int64_t face_entry_capacity, int64_t cell_capacity,
    int64_t piece_capacity, double* vertices, int64_t* face_offsets, int32_t* face_vertices,
    int32_t* face_cells, int32_t* face_facets, int32_t* cell_sites, int32_t* cell_regions,
    double* cell_volumes, double* cell_moments, double* cell_second_moments,
    int32_t* piece_sites, int32_t* piece_tets, int32_t* piece_cells, double* piece_volumes,
    double* piece_moments);
PHX_MC_API int64_t phx_mc_power_cells_site_witness_size(const phx_mc_power_cells* result);
PHX_MC_API int32_t phx_mc_power_cells_export_vertex_sites(
    const phx_mc_power_cells* result, int64_t vertex_offset_capacity, int64_t vertex_capacity,
    int64_t witness_capacity, int64_t* offsets, int32_t* sites);
PHX_MC_API int32_t phx_mc_power_cells_export_vertex_carriers(
    const phx_mc_power_cells* result, int64_t vertex_capacity, int32_t* carriers);
PHX_MC_API int32_t phx_mc_power_cells_failure(
    const phx_mc_power_cells* result, int64_t* reason, int64_t* site, int64_t* tet,
    int32_t* clip_status);
PHX_MC_API void phx_mc_power_cells_free(phx_mc_power_cells* result);

/* ---------------------------------------------------------- periodic topology */
typedef struct phx_mc_periodic_triangulation phx_mc_periodic_triangulation;
PHX_MC_API int32_t phx_mc_periodic_delaunay(
    int32_t dimension, int64_t point_count, const double* points, const double* fractional,
    const double* lattice, const double* inverse, double initial_margin,
    phx_mc_periodic_triangulation** triangulation);
PHX_MC_API int64_t phx_mc_periodic_cell_count(
    const phx_mc_periodic_triangulation* triangulation);
PHX_MC_API int32_t phx_mc_periodic_copy_cells(
    const phx_mc_periodic_triangulation* triangulation, int64_t cell_capacity,
    int32_t* vertices, int32_t* shifts);
PHX_MC_API void phx_mc_periodic_free(phx_mc_periodic_triangulation* triangulation);
/* -------------------------------------------------- owner-local periodic topology */
typedef struct phx_mc_periodic_local_delaunay phx_mc_periodic_local_delaunay;
typedef struct phx_mc_periodic_local_delaunay_request {
  int32_t dimension;
  int64_t point_count;
  const double* points;                 /* (n, d) owned and geometric-ghost source sites */
  const double* fractional;             /* (n, d) lattice coordinates */
  const int64_t* point_global_ids;      /* (n,) stable source IDs */
  const int32_t* point_owner_ranks;     /* (n,) source owner rank */
  int32_t local_owner_rank;
  const double* lattice;                /* (d, d) rows are lattice vectors */
  const double* inverse;                /* (d, d) with fractional = x @ inverse */
  double initial_margin;
  phx_mc_periodic_local_delaunay** result;
} phx_mc_periodic_local_delaunay_request;
PHX_MC_API int32_t phx_mc_periodic_delaunay_local(
    const phx_mc_periodic_local_delaunay_request* request);
PHX_MC_API int32_t phx_mc_periodic_local_delaunay_sizes(
    const phx_mc_periodic_local_delaunay* result, int64_t* cell_count,
    double* certified_margin);
/* Arrays are cell-major: vertices (C,d+1) indexes the supplied local point bank;
 * shifts (C,d+1,d) are absolute periodic image windings; cell_key_ids and
 * cell_key_relative_shifts form canonical sorted (source ID, winding-anchor)
 * rows; cell IDs hash that full key; support_cell_mask marks cells incident to
 * any locally-owned site. support_fractional_lower/upper (C,d) are outward
 * AABBs of each exact circumball projection, routing only relevant owner witnesses. */
PHX_MC_API int32_t phx_mc_periodic_local_delaunay_copy(
    const phx_mc_periodic_local_delaunay* result, int64_t cell_capacity,
    int32_t* vertices, int32_t* shifts, int64_t* cell_ids, int64_t* cell_key_ids,
    int32_t* cell_key_relative_shifts, int32_t* cell_owner_ranks,
    int8_t* support_cell_mask, double* support_fractional_lower,
    double* support_fractional_upper);
/* Exact outward fractional bounds of this rank's owned source rows (dimension
 * entries per bound); has_owned_sites is false for an empty owner. */
PHX_MC_API int32_t phx_mc_periodic_local_delaunay_owner_bounds(
    const phx_mc_periodic_local_delaunay* result, int8_t* has_owned_sites,
    double* fractional_lower, double* fractional_upper);
PHX_MC_API void phx_mc_periodic_local_delaunay_free(
    phx_mc_periodic_local_delaunay* result);

typedef struct phx_mc_periodic_support_query_result phx_mc_periodic_support_query_result;
typedef struct phx_mc_periodic_support_query_request {
  int32_t dimension;
  int64_t cell_count;
  const double* cell_vertex_points;       /* (C,d+1,d), exact source base coordinates */
  const int64_t* cell_vertex_global_ids;  /* (C,d+1,) source identities */
  const int32_t* cell_vertex_shifts;      /* (C,d+1,d), exact image windings */
  int64_t point_count;
  const double* points;                   /* remote owner's owned source rows */
  const int64_t* point_global_ids;
  const double* lattice;
  uint64_t candidate_revision;
  phx_mc_periodic_support_query_result** result;
} phx_mc_periodic_support_query_request;
PHX_MC_API int32_t phx_mc_periodic_owner_support_query(
    const phx_mc_periodic_support_query_request* request);
PHX_MC_API int32_t phx_mc_periodic_support_query_sizes(
    const phx_mc_periodic_support_query_result* result, int64_t* hit_count,
    uint64_t* candidate_revision);
/* Every hit is an image with exact in-sphere sign >= 0, including all ties. */
PHX_MC_API int32_t phx_mc_periodic_support_query_copy_hits(
    const phx_mc_periodic_support_query_result* result, int64_t hit_capacity,
    int64_t* cell_rows, int64_t* point_global_ids, int32_t* image_shifts,
    int8_t* predicate_signs);
PHX_MC_API void phx_mc_periodic_support_query_free(
    phx_mc_periodic_support_query_result* result);

typedef struct phx_mc_periodic_refinement_input {
  int64_t point_count;
  const double* points;
  const double* lattice;
  const double* inverse;
  int64_t cell_count;
  const int32_t* vertices;
  const int32_t* shifts;
  int64_t control_count;
  const double* controls;
  int64_t statistic_count;
  const double* quantiles;
  double absolute_tolerance;
  double relative_tolerance;
  int64_t protected_edge_count;
  const int32_t* protected_edges;
} phx_mc_periodic_refinement_input;
typedef struct phx_mc_periodic_refinement phx_mc_periodic_refinement;
#define PHX_MC_PERIODIC_REFINEMENT_SIZES 2
PHX_MC_API int32_t phx_mc_periodic_refine_2d(
    const phx_mc_periodic_refinement_input* input,
    phx_mc_periodic_refinement** refinement);
PHX_MC_API int32_t phx_mc_periodic_refinement_sizes(
    const phx_mc_periodic_refinement* refinement, int64_t* sizes);
PHX_MC_API int32_t phx_mc_periodic_refinement_copy_points(
    const phx_mc_periodic_refinement* refinement, int64_t point_capacity, double* points);
PHX_MC_API int32_t phx_mc_periodic_refinement_copy_cells(
    const phx_mc_periodic_refinement* refinement, int64_t cell_capacity,
    int32_t* vertices, int32_t* shifts);
PHX_MC_API void phx_mc_periodic_refinement_free(
    phx_mc_periodic_refinement* refinement);

/* ---------------------------------------------------------- surface kernels */
PHX_MC_API int32_t phx_mc_restricted_dual_3d(
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    const double* domain, int64_t facet_capacity, int32_t* facets, int32_t* cells,
    double* endpoints, int32_t* kinds, int32_t* item_status, int64_t* facet_count);
PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    double* center_bounds, int32_t* item_status);
PHX_MC_API int32_t phx_mc_restricted_rays_3d(
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    const double* center_bounds, int64_t ray_count, const int32_t* facets,
    const int32_t* cells, const double* domain, double* endpoint_bounds,
    int32_t* kinds, int32_t* item_status);

typedef struct phx_mc_surface_reconnect_request {
  int64_t vertex_count;
  const double* charts;
  const double* points;
  const double* normals;
  const double* vertex_metrics;
  const int64_t* vertex_pole_ids;
  int64_t triangle_count;
  const int32_t* triangles;
  const int8_t* constrained;
  int64_t insert_count;
  const double* insert_charts;
  const double* insert_points;
  const double* insert_normals;
  const double* insert_metrics;
  const int64_t* insert_pole_ids;
  const double* insert_spacing;
  const int32_t* insert_hints;
  const int32_t* insert_edges;
  int32_t sweep;
  int64_t triangle_capacity;
  int32_t* triangles_out;
  int8_t* constrained_out;
  int64_t* triangle_count_out;
  int32_t* vertex_ids;
  int32_t* item_status;
} phx_mc_surface_reconnect_request;
PHX_MC_API int32_t phx_mc_surface_reconnect(
    const phx_mc_surface_reconnect_request* request);

typedef struct phx_mc_surface_envelope_create_request {
  const double* vertices;
  int64_t vertex_count;
  const int64_t* faces;
  int64_t face_count;
  double offset;
  double spacing;
  double certificate_spacing;
  void** result;
  double* bounds;
} phx_mc_surface_envelope_create_request;
PHX_MC_API int32_t phx_mc_surface_envelope_create(
    const phx_mc_surface_envelope_create_request* request);
PHX_MC_API int32_t phx_mc_surface_envelope_sizes(const void* result, int64_t* sizes);
PHX_MC_API int32_t phx_mc_surface_envelope_arrays(
    void* result, int64_t surface_vertex_capacity, int64_t surface_triangle_capacity,
    int64_t volume_vertex_capacity, int64_t tetrahedron_capacity, double* surface_vertices,
    int64_t* surface_triangles, double* volume_vertices, int64_t* tetrahedra);
PHX_MC_API void phx_mc_surface_envelope_free(void* result);

/* --------------------------------------------------------------- level sets */
#define PHX_MC_LEVEL_SET_FULL_CARRIERS 0
#define PHX_MC_LEVEL_SET_TOPOLOGY_ONLY 1
typedef struct phx_mc_level_set_split_3d_request {
  int64_t point_count;
  const double* points;
  const double* values;
  int64_t cell_count;
  const int32_t* cells;
  int64_t face_count;
  const int32_t* faces;
  int64_t edge_count;
  const int32_t* edges;
  const int8_t* protected_edges;
  int32_t output_mode;
  int64_t out_point_capacity;
  int64_t out_cell_capacity;
  double* out_points;
  int32_t* out_cells;
  int32_t* out_parents;
  int32_t* out_sources;
  double* out_weights;
  double* out_deviations;
  int64_t* out_point_count;
  int64_t* out_cell_count;
  int64_t* failed_edge;
} phx_mc_level_set_split_3d_request;
PHX_MC_API int32_t phx_mc_level_set_split_3d(
    const phx_mc_level_set_split_3d_request* request);

/* ---------------------------------------- mixed reference topology templates */
#define PHX_MC_MIXED_TEMPLATE_COUNTS 3
enum phx_mc_mixed_template_count_index {
  PHX_MC_MIXED_TEMPLATE_COUNT = 0,
  PHX_MC_MIXED_CHILD_COUNT = 1,
  PHX_MC_MIXED_CORNER_COUNT = 2
};
enum phx_mc_mixed_template_kind {
  PHX_MC_MIXED_TETRAHEDRON = 0,
  PHX_MC_MIXED_HEXAHEDRON = 1,
  PHX_MC_MIXED_PRISM = 2,
  PHX_MC_MIXED_PYRAMID = 3,
  PHX_MC_MIXED_QUADRILATERAL = 4
};
PHX_MC_API int32_t phx_mc_mixed_template_counts(
    int32_t kind, int32_t axial, int32_t index, int32_t* counts);
PHX_MC_API int32_t phx_mc_mixed_template(
    int32_t kind, int32_t axial, int32_t index, int64_t element_capacity,
    double* references);
/* ------------------------------------------------ exact contacts and arrangements */
typedef struct phx_mc_arrangement phx_mc_arrangement;
PHX_MC_API int32_t phx_mc_triangle_intersections(
    int64_t count, int32_t point_capacity, const double* first, const double* second,
    const int64_t* first_ids, const int64_t* second_ids, int8_t* classes,
    int32_t* point_counts, double* points, double* point_bounds, int8_t* point_features,
    int32_t* item_status);
PHX_MC_API int32_t phx_mc_segment_triangle_intersections(
    int64_t count, int32_t point_capacity, const double* segments, const double* triangles,
    const int64_t* segment_ids, const int64_t* triangle_ids, int8_t* classes,
    int32_t* point_counts, double* points, double* point_bounds, int8_t* point_features,
    int32_t* item_status);
PHX_MC_API int32_t phx_mc_point_triangle_locations(
    int64_t count, const double* points, const double* triangles, int8_t* sides,
    int8_t* features, int32_t* item_status);
PHX_MC_API int32_t phx_mc_arrange_triangles(
    int64_t vertex_count, const double* vertices, int64_t triangle_count,
    const int64_t* triangles, const int32_t* surfaces, int64_t pair_count,
    const int64_t* pairs, int64_t* failed_face, phx_mc_arrangement** result);
PHX_MC_API int32_t phx_mc_arrangement_sizes(const phx_mc_arrangement* result, int64_t* sizes);
PHX_MC_API int32_t phx_mc_arrangement_export(
    const phx_mc_arrangement* result, int64_t vertex_capacity, int64_t fragment_capacity,
    int64_t contact_edge_capacity, int64_t coplanar_capacity, int64_t location_capacity,
    double* vertices, double* bounds, int8_t* constructions, int64_t* origins,
    int64_t* fragments, int64_t* fragment_faces, int64_t* contact_edges,
    int64_t* coplanar, int64_t* locations);
PHX_MC_API int32_t phx_mc_arrangement_classify(
    phx_mc_arrangement* result, int64_t operand_count, int64_t* failed_face);
PHX_MC_API int32_t phx_mc_arrangement_classification_sizes(
    const phx_mc_arrangement* result, int64_t* sizes);
PHX_MC_API int32_t phx_mc_arrangement_export_classification(
    const phx_mc_arrangement* result, int64_t fragment_capacity, int64_t component_capacity,
    int64_t operand_capacity, int64_t* fragment_components, int32_t* component_windings);
PHX_MC_API void phx_mc_arrangement_free(phx_mc_arrangement* result);

/* Exact reference-triangle clipping retains the caller's output-row extent. */
PHX_MC_API int32_t phx_mc_clip_reference_triangle_exact(
    const uint32_t* words, int64_t word_count, const int64_t* offsets, const int8_t* signs,
    int32_t vertex_capacity, int32_t* supporting_planes, int32_t* edge_labels,
    int32_t* vertex_count);

typedef struct phx_mc_reference_tetrahedron_clip phx_mc_reference_tetrahedron_clip;
typedef struct phx_mc_clip_reference_tetrahedron_exact_request {
  const uint32_t* words;
  int64_t word_count;
  const int64_t* offsets;
  const int8_t* signs;
  int32_t plane_count;
  phx_mc_reference_tetrahedron_clip** result;
} phx_mc_clip_reference_tetrahedron_exact_request;
PHX_MC_API int32_t phx_mc_clip_reference_tetrahedron_exact(
    const phx_mc_clip_reference_tetrahedron_exact_request* request);
#define PHX_MC_REFERENCE_TETRAHEDRON_CLIP_SIZES 4
PHX_MC_API int32_t phx_mc_reference_tetrahedron_clip_sizes(
    const phx_mc_reference_tetrahedron_clip* result, int64_t* sizes);
/* Export exact plane-support witnesses and closed face topology. sizes gives
 * vertex, face, face-entry and vertex-plane-incidence counts. supporting_planes
 * has shape (vertex_capacity, 3); vertex_incident_offsets is CSR with
 * vertex_capacity + 1 entries. face_offsets is CSR with face_capacity + 1
 * entries. All plane ids index the exact input plane bank; no rounded vertices
 * are published by this topology-only result. */
PHX_MC_API int32_t phx_mc_reference_tetrahedron_clip_export(
    const phx_mc_reference_tetrahedron_clip* result, int64_t vertex_capacity,
    int32_t* supporting_planes, int64_t incident_offset_capacity,
    int32_t* vertex_incident_offsets, int64_t incident_plane_capacity,
    int32_t* vertex_incident_planes, int64_t face_offset_capacity, int32_t* face_offsets,
    int64_t face_plane_capacity, int32_t* face_planes, int64_t face_vertex_capacity,
    int32_t* face_vertices);
PHX_MC_API void phx_mc_reference_tetrahedron_clip_free(
    phx_mc_reference_tetrahedron_clip* result);

typedef struct phx_mc_tetrahedra_arrangement phx_mc_tetrahedra_arrangement;
typedef struct phx_mc_arrange_tetrahedra_exact_request {
  int64_t point_count;
  int64_t plane_count;
  int64_t tetrahedron_count;
  int64_t job_count;
  int64_t cut_count;
  int64_t point_word_count;
  int64_t plane_word_count;
  const uint32_t* point_words;
  const int64_t* point_offsets;
  const int8_t* point_signs;
  const uint32_t* plane_words;
  const int64_t* plane_offsets;
  const int8_t* plane_signs;
  const int32_t* tetrahedra;
  const int32_t* tetrahedron_planes;
  const int32_t* parents;
  const int64_t* cut_offsets;
  const int32_t* cut_planes;
  phx_mc_tetrahedra_arrangement** result;
} phx_mc_arrange_tetrahedra_exact_request;
PHX_MC_API int32_t phx_mc_arrange_tetrahedra_exact(
    const phx_mc_arrange_tetrahedra_exact_request* request);
#define PHX_MC_TETRAHEDRA_ARRANGEMENT_SIZES 5
PHX_MC_API int32_t phx_mc_tetrahedra_arrangement_sizes(
    const phx_mc_tetrahedra_arrangement* result, int64_t* sizes);
PHX_MC_API int32_t phx_mc_tetrahedra_arrangement_export(
    const phx_mc_tetrahedra_arrangement* result, int64_t vertex_capacity,
    int64_t vertex_word_capacity, int64_t job_offset_capacity, int64_t face_offset_capacity,
    int64_t face_capacity, int64_t face_entry_capacity, uint32_t* vertex_words,
    int64_t* vertex_word_offsets, int8_t* vertex_signs, int32_t* vertex_input_index,
    int64_t* cell_face_offsets, int64_t* face_offsets, int32_t* face_vertices,
    int32_t* face_planes);
PHX_MC_API void phx_mc_tetrahedra_arrangement_free(phx_mc_tetrahedra_arrangement* result);

/* Source-only PLC preparation. The polygon_vertex_count is the true length of
 * polygon_vertices; polygon_offsets has polygon_count + 1 entries. Optional
 * facet_tolerances and segment_tolerances have one value per corresponding row. */
enum phx_mc_plc_failure_reason {
  PHX_MC_PLC_NO_FAILURE = 0,
  PHX_MC_PLC_DUPLICATE_POINT = 1,
  PHX_MC_PLC_INVALID_POLYGON = 2,
  PHX_MC_PLC_INVALID_FACET = 3,
  PHX_MC_PLC_INTERSECTING = 4,
  PHX_MC_PLC_OPEN_BOUNDARY = 5,
  PHX_MC_PLC_INCONSISTENT_REGIONS = 6,
  PHX_MC_PLC_REGION_LEAK = 7,
  PHX_MC_PLC_FIXED_SEGMENT = 8,
  PHX_MC_PLC_FIXED_FACET = 9,
  PHX_MC_PLC_NONREPRESENTABLE = 10,
  PHX_MC_PLC_INVALID_SEED = 14,
  PHX_MC_PLC_OUTSIDE_DOMAIN = 15,
  PHX_MC_PLC_SOURCE_DEVIATION = 17,
  /* A non-representable segment split point could not be moved off the
   * boundary: first names the PLC edge row, second the split point. */
  PHX_MC_PLC_STEINER_SUPPRESSION = 18
};
enum phx_mc_plc_entity_kind {
  PHX_MC_PLC_ENTITY_NONE = 0,
  PHX_MC_PLC_ENTITY_POINT = 1,
  PHX_MC_PLC_ENTITY_POLYGON = 2,
  PHX_MC_PLC_ENTITY_FACET = 3,
  PHX_MC_PLC_ENTITY_EDGE = 4,
  PHX_MC_PLC_ENTITY_REGION = 5,
  PHX_MC_PLC_ENTITY_SEED = 6
};
#define PHX_MC_PLC3D_SIZES 6
typedef struct phx_mc_plc3d_result phx_mc_plc3d_result;
typedef struct phx_mc_plc3d_prepare_source_request {
  int64_t point_count;
  const double* points;
  int64_t polygon_count;
  int64_t polygon_vertex_count;
  const int64_t* polygon_offsets;
  const int32_t* polygon_vertices;
  const int32_t* polygon_facets;
  int64_t facet_count;
  const int32_t* facet_regions;
  int64_t segment_count;
  const int32_t* segments;
  int64_t seed_count;
  const double* seeds;
  const int32_t* seed_regions;
  int32_t boundary_policy;
  const double* facet_tolerances;
  const double* segment_tolerances;
  /* Optional declared exact facet planes: 9 values per facet (origin, first
   * axis, second axis; an all-NaN row declares none) and their tolerances
   * (facet_count values; null = 0). A declared facet's triangles form one
   * group whatever their rounding once every vertex is admitted (the plane
   * meets its rounding box or lies within the tolerance), else the facet
   * refuses with SOURCE_DEVIATION naming the vertex. */
  const double* facet_planes;
  const double* facet_plane_tolerances;
  phx_mc_plc3d_result** result;
} phx_mc_plc3d_prepare_source_request;
PHX_MC_API int32_t phx_mc_plc3d_prepare_source(
    const phx_mc_plc3d_prepare_source_request* request);
/* Protected PLC recovery from the same authored source banks. */
PHX_MC_API int32_t phx_mc_plc3d_recover(
    const phx_mc_plc3d_prepare_source_request* request);
PHX_MC_API int32_t phx_mc_plc3d_sizes(const phx_mc_plc3d_result* result, int64_t* sizes);
/* sizes rows are points, tetrahedra, faces, segments, PLC edges, and source
 * triangles. Export capacities are row extents: points (P,3), tetrahedra
 * (T,4) plus regions (T,), faces (F,3) plus sources/orientations (F,),
 * segments/PLC edges (S,2)/(E,2), and source triangles (I,3) plus authored
 * polygon ids (I,). PLC recovery witness entities index the exported raw
 * carrier banks: kSegment indexes plc_edges and kFacet indexes input_triangles;
 * kNone uses -1. Persistent TetMesh witnesses instead index canonical source
 * segments/faces. Failure returns authored entity locations and any certified
 * source-deviation refusal. */
PHX_MC_API int32_t phx_mc_plc3d_export(
    const phx_mc_plc3d_result* result, int64_t point_capacity, int64_t tetrahedron_capacity,
    int64_t face_capacity, int64_t segment_capacity, int64_t edge_capacity,
    int64_t input_triangle_capacity, double* points, int32_t* tetrahedra,
    int32_t* tetrahedron_regions, int32_t* faces, int32_t* face_sources,
    int8_t* face_orientations, int32_t* segments, int32_t* segment_sources,
    double* protection, int8_t* vertex_dimensions, int32_t* plc_edges,
    int32_t* input_triangles, int32_t* input_polygons);
PHX_MC_API int32_t phx_mc_plc3d_source_witnesses(
    const phx_mc_plc3d_result* result, int64_t point_capacity, int8_t* strata,
    int32_t* entities, double* parameters, double* deviations);
PHX_MC_API int32_t phx_mc_plc3d_failure(
    const phx_mc_plc3d_result* result, int32_t* reason, int32_t* first_kind,
    int64_t* first, int32_t* second_kind, int64_t* second, double* certified_deviation,
    double* declared_bound);
// Steiner-suppression search evidence: values[0..11] = wedges searched,
// wedges filled, states expanded, most states in one wedge, largest searched
// wedge (cells), wedges filled after merging with neighbouring cells, most
// merged layers, wedges whose region was exhausted unfilled, largest
// exhausted region (cells), constrained Delaunay wraps run, cells they laid,
// most cells laid by one wrap. Available after a refusal too.
PHX_MC_API int32_t phx_mc_plc3d_suppression_search(
    const phx_mc_plc3d_result* result, int64_t* values);
PHX_MC_API void phx_mc_plc3d_free(phx_mc_plc3d_result* result);

enum phx_mc_intersection_class {
  PHX_MC_DISJOINT = 0,
  PHX_MC_SHARED_VERTEX = 1,
  PHX_MC_SHARED_EDGE = 2,
  PHX_MC_TOUCHING = 3,
  PHX_MC_CROSSING = 4,
  PHX_MC_COPLANAR_OVERLAP = 5,
  PHX_MC_COINCIDENT = 6
};
#ifdef __cplusplus
}
#endif

#endif /* PHYDRAX_MESHCORE_H */
