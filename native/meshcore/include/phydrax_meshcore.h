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
  PHX_MC_INDEX_RANGE_ERROR = 11,     /* result cannot be represented by native indices */
  PHX_MC_STALE_GENERATION = 12,     /* requested mesh generation no longer current */
  PHX_MC_IO_FAILURE = 13            /* owned scratch storage operation failed */
};

/* ------------------------------------------------------------------ identity */

PHX_MC_API const char* phx_mc_version(void);
/* SHA-256 of the library sources recorded at configure time. */
PHX_MC_API const char* phx_mc_build_hash(void);
PHX_MC_API void phx_mc_exact_domain(int32_t* min_exponent, int32_t* max_exponent);

/* Bounded implicit-cell contour batches. All arrays are caller-owned and
 * contiguous: signs/values (count,8), boxes (count,2,3), roots (count,12),
 * vertices (root_count), points (root_count,3), root_issues (root_count).
 * Roots contain -1 for noncrossing edges, otherwise a valid root index.
 * Outputs are handled/counts/issues (count) and triangles (count,10,3).
 * Unhandled cells require the canonical contour algorithm; triangles for
 * unhandled rows and unused triangle slots are unspecified. Invalid inputs
 * leave all outputs unchanged. Callers must supply disjoint output buffers.
 */
PHX_MC_API int32_t phx_mc_implicit_contours(
    int64_t count, int64_t root_count, const int8_t* signs, const double* values,
    const double* boxes, const int64_t* roots, const int64_t* vertices,
    const double* points, const int64_t* root_issues, double minimum_area,
    uint8_t* handled, int8_t* counts, int64_t* triangles, int64_t* issues);

/* Gauss-Newton proposals for implicit faces. coefficients is (face_count,3,10),
 * face_rows is (count), points is (count,3), and first and second are (count).
 * Outputs must not overlap any input or each other, including handled. Invalid
 * arguments or face indices leave every output unchanged. On PHX_MC_OK,
 * handled is 1 if the whole batch was computed; when it is 0, proposal outputs
 * are unspecified and the caller must recompute the entire batch. This optional
 * entry requires round-to-nearest and gradual underflow without changing them.
 */
PHX_MC_API int32_t phx_mc_implicit_face_proposals(
    const double* coefficients, int64_t face_count, const int64_t* face_rows,
    const double* points, int64_t count, double* first, double* second,
    int32_t* handled);

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
 * PHX_MC_UNRESOLVED, PHX_MC_INTERNAL_ERROR. Measures and first moments of
 * failed items are zero.
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

/* Tetrahedra (count, 4, 3), either orientation. The moments are integrated
 * from the symbolic intersection vertices, not their rounded carriers.
 * A positive overlap whose measured volume cannot be represented as a positive
 * finite binary64 value reports PHX_MC_UNRESOLVED, not an empty intersection. */
PHX_MC_API int32_t phx_mc_tetrahedron_intersection_moments(
    int64_t count, const double* first, const double* second, double* volumes,
    double* first_moments, int32_t* item_status);

/* The same symbolic intersection, including second_moments (count, 3, 3),
 * the row-major integral of x x^T. No simplex carriers are constructed. */
PHX_MC_API int32_t phx_mc_tetrahedron_intersection_quadratic_moments(
    int64_t count, const double* first, const double* second, double* volumes,
    double* first_moments, double* second_moments, int32_t* item_status);

/* Simplex partitions of the intersections above, for quadrature on the common
 * refinement.  Each nonempty intersection is fanned from the vertex average o of
 * its rounded vertices: polygon intersections as triangles (o, v_i, v_{i+1})
 * over the counterclockwise boundary loop, simplices (count, simplex_capacity,
 * 3, 2); tetrahedron intersections as tetrahedra (o, x_0, x_i, x_{i+1}) over
 * the fan triangles of every outward face, simplices (count, simplex_capacity,
 * 4, 3).  The returned measures and first moments are calculated from this
 * rounded fan and reproduce its signed simplex integrals up to arithmetic
 * rounding; tetrahedron moments-only results can differ because they integrate
 * symbolic vertices instead.  Simplices are positively oriented up to the
 * rounding of o. simplex_counts (count,) receives the number written;
 * empty and zero-measure intersections write none. A positive tetrahedron
 * overlap whose rounded fan has nonpositive signed volume, or whose positive
 * signed volume underflows to zero, reports PHX_MC_UNRESOLVED with zero count.
 * An intersection needing more
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

/* The triangulations above, of dimension 2 or 3, with options. weights NULL
 * gives the Delaunay triangulation, otherwise the regular one. Without
 * PHX_MC_TRIANGULATION_CANONICAL the cells are positively oriented and the
 * cell set is the same, but cell rotation and order are unspecified
 * (deterministic for the input); the canonical ordering above costs a sort. */
#define PHX_MC_TRIANGULATION_CANONICAL 1u
PHX_MC_API int32_t phx_mc_triangulate(int32_t dimension, int64_t point_count,
                                      const double* points, const double* weights,
                                      uint32_t options, phx_mc_mesh** mesh);

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
/* FIXED_BOUNDARY: refinement's circumcenter lies beyond a fixed face, or an
 * improvement radius-edge residual lies on a constrained face whose own
 * triangle already exceeds the bound (no tetrahedron on it can meet it).
 * INPUT_ANGLE: conforming refinement left the cell because the boundary split
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
/* Facet ID of an unmet row without a forcing constrained facet. */
#define PHX_MC_TET_UNMET_NO_FACET UINT64_MAX
/* Read-only current-cell quality assessment, independent of stored unmet
 * records. The determinant floor is max(minimum_relative_determinant, 2^-22).
 * For quality/refinement calls, radius_edge_bound == 0 encodes no radius-edge
 * target; a positive finite value is an active target. This does not disable
 * determinant, finite-measurement, orientation, size, or angle checks.
 * minimum_dihedral_degrees == 0 requests no positive angle threshold.
 * Size uses the retained vertex-size field and is checked only when all four
 * vertices have targets. Values are radius-edge ratio, circumradius/mean size,
 * angle in degrees, or determinant minus effective floor for VALIDITY.
 * NONREPRESENTABLE denotes an unrepresentable measurement; otherwise rows use
 * NO_IMPROVEMENT, without asserting that improvement was attempted.
 * With capacity zero and all arrays null, returns the required row count and
 * PHX_MC_OK. Otherwise all rows are written or PHX_MC_CAPACITY_EXCEEDED is
 * returned with the required count and no array writes. A generation mismatch
 * returns PHX_MC_STALE_GENERATION. The caller must synchronize against edits.
 */
PHX_MC_API int32_t phx_mc_tet_mesh_evaluate_quality(
    const phx_mc_tet_mesh* mesh, uint64_t expected_generation,
    double radius_edge_bound, double minimum_dihedral_degrees,
    double minimum_relative_determinant, int64_t capacity, int64_t* count,
    uint64_t* cell_ids, int32_t* criteria, int32_t* reasons, double* values);
/* Evaluate a separate finite candidate in snapshot vertex order without changing
 * the mesh, its source witnesses, history, or retained per-vertex size field.
 * This does not certify the candidate's source or constraint fidelity. */
PHX_MC_API int32_t phx_mc_tet_mesh_evaluate_candidate_quality(
    const phx_mc_tet_mesh* mesh, uint64_t expected_generation,
    int64_t vertex_count, const double* points,
    double radius_edge_bound, double minimum_dihedral_degrees,
    double minimum_relative_determinant, int64_t capacity, int64_t* count,
    uint64_t* cell_ids, int32_t* criteria, int32_t* reasons, double* values);
PHX_MC_API int32_t phx_mc_tet_mesh_unmet_count(const phx_mc_tet_mesh* mesh,
                                                int64_t* count);
/* One row per unmet (cell, criterion). INPUT_ANGLE rows carry the acute
 * feature's apex vertex ID (a ridge's first endpoint) and its angle in
 * degrees; other rows PHX_MC_TET_UNMET_NO_APEX and NaN. Improvement
 * FIXED_BOUNDARY radius-edge rows carry the forcing constrained facet's ID and
 * its triangle radius-edge ratio; other rows PHX_MC_TET_UNMET_NO_FACET and
 * NaN. */
PHX_MC_API int32_t phx_mc_tet_mesh_unmet(
    const phx_mc_tet_mesh* mesh, int64_t capacity, uint64_t* cell_ids,
    int32_t* criteria, int32_t* reasons, double* values, uint64_t* apex_vertex_ids,
    double* apex_angles, uint64_t* facet_ids, double* facet_ratios);
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
  /* Optional source-row circumradius targets, one per point; zero means absent. */
  const double* initial_vertex_sizes;
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
/* Protect source facet groups and their incident edges/vertices in an existing
 * mesh. Unknown groups are rejected without mutation. */
PHX_MC_API int32_t phx_mc_tet_mesh_protect_facet_sources(
    phx_mc_tet_mesh* mesh, int64_t count, const int32_t* sources);
/* As create, but install the row-major index-to-world linear map before
 * validating and constructing source witnesses and protection data.
 * A null linear pointer selects the identity metric. */
PHX_MC_API int32_t phx_mc_tet_mesh_create_with_affine_metric(
    const phx_mc_tet_mesh_create_request* request, const double* linear);
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
/* Vertices are written in slot order; cells, facets and segments in ascending
 * ID order, their corners as rows of the vertex banks. */
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
  /* Current retained circumradius target per vertex ID; zero = unconstrained. */
  double* vertex_size_targets;
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
  int64_t* cell_vertex_rows;
  int32_t* cell_regions;
  /* Canonical mean circumradius target; zero if any corner is unconstrained. */
  double* cell_size_targets;
  uint64_t* facet_ids;
  int64_t* facet_vertex_rows;
  int32_t* facet_sources;
  int8_t* facet_orientations;
  int8_t* protected_facets;
  uint64_t* segment_ids;
  int64_t* segment_vertex_rows;
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
/* Exact refusal probe of an edit request, mutating no mesh state. `decided`
 * is 1 when phx_mc_tet_mesh_apply_edit would refuse the request with
 * `edit_status`, and 0 when it may apply or the probe does not decide it
 * (flips, ring reconnections, collapses and relocations of an unbound,
 * nonperiodic state are
 * decided). A decided refusal advances the implicit-source refusal counters
 * exactly as that refused apply_edit would. */
PHX_MC_API int32_t phx_mc_tet_mesh_probe_edit(
    phx_mc_tet_mesh* mesh, const phx_mc_tet_mesh_edit_request* request,
    int32_t* decided, int32_t* edit_status);
/* Exact refusal probes of independent relocations, mutating no mesh state.
 * Group g moves vertices vertex_ids[vertex_offsets[g]..vertex_offsets[g+1])
 * to the matching rows of `points` and declares the star cells
 * cell_ids[cell_offsets[g]..cell_offsets[g+1]). Each group is probed alone
 * against the current state, as the relocation request that states no source
 * witness and no protection for the moved vertices, clears their edge
 * parameters and supplies no ancestry. decided[g] is 1 when
 * phx_mc_tet_mesh_apply_edit would refuse that
 * request with edit_statuses[g], and 0 when it may apply or the probe does not
 * decide it. Every decided refusal advances the implicit-source refusal
 * counters exactly as that refused apply_edit would; the implicit facet
 * refusal record afterwards is that of the last group. Offsets start at zero
 * and do not decrease. A generation or source identity mismatch returns
 * PHX_MC_STALE_GENERATION. The caller must synchronize against edits. */
PHX_MC_API int32_t phx_mc_tet_mesh_probe_relocations(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length, const char* source_revision,
    int64_t source_revision_length, const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length, int64_t group_count,
    const int64_t* vertex_offsets, const uint64_t* vertex_ids, const double* points,
    const int64_t* cell_offsets, const uint64_t* cell_ids, int32_t* decided,
    int32_t* edit_statuses);
/* Applies one relocation in place: the request of
 * phx_mc_tet_mesh_probe_relocations for a single group, moving `vertex_count`
 * vertices together. `decided` is 1 with the `edit_status`
 * phx_mc_tet_mesh_apply_edit would return for that request: APPLIED after the
 * moved vertices' points, witnesses, edge parameters and ancestry are written
 * and the generation advanced, or a refusal that changes no mesh state and
 * advances the implicit-source refusal counters exactly as the refused
 * apply_edit would. `decided` is 0, with nothing changed, when the mesh
 * carries periodic records or a bound periodic identity or has exhausted its
 * generations; phx_mc_tet_mesh_apply_edit decides those. A generation or
 * source identity mismatch returns PHX_MC_STALE_GENERATION. The caller must
 * synchronize against edits. */
PHX_MC_API int32_t phx_mc_tet_mesh_apply_relocation(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length, const char* source_revision,
    int64_t source_revision_length, const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length, int64_t vertex_count,
    const uint64_t* vertex_ids, const double* points, int64_t cell_count,
    const uint64_t* cell_ids, int32_t* decided, int32_t* edit_status);
/* Ephemeral evidence from the latest actual implicit-facet admission refusal.
 * The record belongs to the current mesh identity and generation only. */
typedef struct phx_mc_tet_implicit_facet_refusal {
  int32_t refused;
  uint64_t generation;
  uint64_t facet_id;
  int32_t facet_source;
  int32_t orientation;
  uint64_t vertex_ids[3];
  double points[9];
  int64_t transversality_refusals;
  int64_t distance_refusals;
  int32_t status;
  /* A proposal sample from the first rejecting distance subdivision piece,
   * in vertex_ids order; not a distance extremum or acceptance certificate. */
  int32_t has_distance_barycentric;
  double distance_barycentric[3];
} phx_mc_tet_implicit_facet_refusal;
PHX_MC_API int32_t phx_mc_tet_mesh_implicit_facet_refusal(
    const phx_mc_tet_mesh* mesh, phx_mc_tet_implicit_facet_refusal* output);
PHX_MC_API int32_t phx_mc_tet_mesh_clear_implicit_facet_refusal(phx_mc_tet_mesh* mesh);
/* Refine inserts circumcenters until every cell meets the radius-edge bound
 * and its circumradius size target. `sizes` (one per current vertex in
 * snapshot order, 0 = unconstrained; empty keeps the mesh's retained field)
 * becomes the mesh's retained size field; inserted vertices receive
 * interpolated targets. radius_edge_bound == 0 omits the radius-edge target;
 * retained size and validity obligations remain active. */
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
 * radius_edge_bound == 0 omits that target, without disabling validity or size.
 * No edit creates a cell whose circumradius exceeds its size target by more
 * than the largest excess among the cells it removes (none where they were
 * within target), except one removing a degenerate cell (below the
 * determinant floor or scoring 0), whose repair takes precedence. `sizes`
 * follows refine (empty keeps the retained field, e.g. refine's). The run
 * status is UNRESOLVED while the radius-edge or dihedral target, or the
 * determinant floor, remains unmet;
 * `phx_mc_tet_mesh_unmet` reports each residual criterion. The size field is
 * a constraint on improvement, not one of its targets.
 *
 * `boundary_collapse_quality` opts into transactional generated-facet-vertex
 * coarsening followed by ordinary improvement. The supplied original vertex
 * IDs must cover the source-point bank in source order. A trial commits only
 * when the entire final mesh meets size and quality; failed trials leave the
 * state, IDs and history unchanged. Use only for an explicitly authorized
 * conforming boundary policy. */
PHX_MC_API int32_t phx_mc_tet_mesh_improve_quality(
    phx_mc_tet_mesh* mesh, uint64_t expected_generation, int32_t identity_bound,
    const char* source_id, int64_t source_id_length,
    const char* source_revision, int64_t source_revision_length,
    const char* source_map_id, int64_t source_map_id_length,
    const char* frame_id, int64_t frame_id_length,
    double radius_edge_bound, double minimum_dihedral_degrees,
    double minimum_relative_determinant, int32_t boundary_collapse_quality,
    int64_t original_vertex_count, const uint64_t* original_vertex_ids,
    int64_t size_count, const double* sizes,
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
/* Execute one validated program over count independent, row-major workspaces.
 * The arithmetic matches the elementwise interval kernels, and the result does
 * not depend on parallelism. Workspaces must be disjoint and contain count *
 * workspace_size entries each. Unsupported opcodes are refused before writes.
 * Opcode 25 divides intervals using the elementwise host-kernel convention;
 * reduction opcode 7 uses a_index -2 to accumulate from positive zero and
 * -1 to begin from the first term. Opcode 27 evaluates power per element;
 * its second size output entries are -1 for invalid real domains, else 0.
 * Opcode 28 is the tangent: one operand, like the sine and cosine rows, with
 * the enclosure of phx_mc_interval_elementary's tangent, which is the whole
 * line for an interval that reaches a pole. */
PHX_MC_API int32_t phx_mc_interval_program_batch(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, double* lower,
    double* upper, int64_t workspace_size, int64_t count, int32_t parallelism);
/* Bit n advertises interval-program opcode n; bit 0 is reserved. */
PHX_MC_API int32_t phx_mc_interval_program_capabilities(uint64_t* opcodes);
/* Point evaluation of a validated interval program at `count` points:
 * binary64 with ordinary rounding, for proposals, never a certificate. seeds
 * is a workspace_size template holding each seed's value at its entries; the
 * point's input_dimension coordinates occupy entries 0 .. input_dimension - 1;
 * points is (count, input_dimension) row-major. result is (count,
 * output_count): the values of workspace entries `outputs`. Min/max
 * winner-flag rows (21, 22) write zeros and decision rows (23) pass their
 * comparison through. Points are processed in cache-sized blocks. */
PHX_MC_API int32_t phx_mc_program_evaluate(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, const double* seeds,
    int64_t workspace_size, int64_t input_dimension, const double* points, int64_t count,
    const int64_t* outputs, int64_t output_count, double* result);
/* Second-order forward jets of the same program in its input_dimension (1 to
 * 3) inputs, as phx_mc_program_evaluate. result is (count, output_count,
 * components) with components = 1 + n + n (n + 1) / 2: the value, the
 * gradient, and the upper triangle of the Hessian row by row. */
PHX_MC_API int32_t phx_mc_program_jet(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, const double* seeds,
    int64_t workspace_size, int64_t input_dimension, const double* points, int64_t count,
    const int64_t* outputs, int64_t output_count, double* result);
/* Deterministic binary64 kernels: the same bits for the same input on every
 * platform, from correctly rounded + - * / and sqrt in a fixed order, without
 * libm or LAPACK. They steer floating-point proposals; their error bounds
 * (exp 4 ulp, log 6 ulp, x^(2/3) relative 2^-43 on [2^-128, 2^128]) are no
 * basis for an exact decision. Rows are independent and evaluated in order.
 * phx_mc_deterministic_float_self_test returns PHX_MC_OK when this build and
 * thread have the arithmetic the kernels assume and fixed kernel results have
 * their expected bits, else PHX_MC_INTERNAL_ERROR with `failed_check` naming
 * the first failure: 1 not round-to-nearest-even in binary64, 2 a product
 * fused with a sum, 3 subnormals flushed, 4 a kernel result differs. */
PHX_MC_API int32_t phx_mc_deterministic_float_self_test(int32_t* failed_check);
PHX_MC_API int32_t phx_mc_deterministic_exp(int64_t count, const double* values,
                                            double* results);
PHX_MC_API int32_t phx_mc_deterministic_log(int64_t count, const double* values,
                                            double* results);
PHX_MC_API int32_t phx_mc_deterministic_pow_two_thirds(int64_t count, const double* values,
                                                       double* results);
/* Cyclic Jacobi eigen decomposition of `count` symmetric 3x3 matrices, each
 * given by the upper triangle of nine row-major values. values[3r..] are the
 * eigenvalues of row r in ascending order and vectors[9r..] the row-major
 * matrix whose columns are the matching orthonormal eigenvectors.
 * converged[r] is 0, with zero outputs, for a non-finite matrix or one the
 * fixed sweep cap does not diagonalize. */
PHX_MC_API int32_t phx_mc_deterministic_symmetric_eigen3(int64_t count, const double* matrices,
                                                         double* values, double* vectors,
                                                         int8_t* converged);
/* Riemannian metric measurements on the deterministic kernels: the same bits
 * for the same input on every platform. A metric is a symmetric
 * positive-definite 3x3 tensor of nine row-major values. The results order
 * and screen floating-point proposals; no exact decision rests on them. Rows
 * are independent and evaluated in order; a NaN result is 0x7ff8000000000000.
 * An index outside its bank is PHX_MC_INVALID_INPUT, with no output written.
 *
 * phx_mc_metric_logarithms: log M = V diag(log lambda) V^T of each tensor;
 * valid[r] is 0, with a zero row, when the eigen iteration does not end or an
 * eigenvalue is not positive. */
PHX_MC_API int32_t phx_mc_metric_logarithms(int64_t count, const double* metrics,
                                            double* logarithms, int8_t* valid);
/* Log-Euclidean mean exp(sum_k w_k log M_k) of each row's support
 * support[offsets[r] .. offsets[r + 1]) (rows of the source bank, at least
 * one per row, with weights at the same positions), summed in support order.
 * A support whose tensors all compare equal returns its first tensor
 * unchanged, once that tensor has a logarithm. source_logarithms holds the
 * bank's logarithms, or is null to evaluate them per row; supplied
 * logarithms are taken as those of the bank and its tensors are then not
 * examined again. valid[r] is 0, with a zero row, when a logarithm fails or
 * the mean is not finite. */
PHX_MC_API int32_t phx_mc_metric_interpolate(
    int64_t row_count, const int64_t* offsets, const int64_t* support, const double* weights,
    int64_t source_count, const double* source_metrics, const double* source_logarithms,
    double* metrics, int8_t* valid);
/* Length of each edge (first, second) of point rows: the logarithmic mean
 * (a - b) / log(a / b) of a = sqrt(d^T M_first d) and b = sqrt(d^T M_second d),
 * d = p_second - p_first, and (a + b) / 2 when |b / a - 1| <= 1e-6. NaN when
 * a quadratic form is negative or not a number. */
PHX_MC_API int32_t phx_mc_metric_lengths(int64_t edge_count, const int64_t* edges,
                                         int64_t point_count, const double* points,
                                         const double* metrics, double* lengths);
/* Violation of the length band [1 / fl(sqrt 2), fl(sqrt 2)] in quanta of
 * 2^-10: above[r] is max(ceil(log(length / fl(sqrt 2)) * 2^10), 1) for a
 * length above the band and 0 otherwise; below[r] likewise for
 * log((1 / fl(sqrt 2)) / length). An infinite or zero length gives an
 * infinite count. A NaN length was not measured and has no count: both
 * outputs are NaN. */
PHX_MC_API int32_t phx_mc_metric_band_quanta(int64_t count, const double* lengths,
                                             double* above, double* below);
/* For each cell of four corners (twelve values): determinants[c] is
 * det(p1 - p0, p2 - p0, p3 - p0) and relative[c] its magnitude over the
 * largest product of the three edge lengths at a corner, 0 when either is not
 * finite or that product is 0. The evaluated determinant differs from the
 * exact one by at most 50 * 2^-53 times that product. */
PHX_MC_API int32_t phx_mc_metric_relative_determinants(int64_t cell_count,
                                                       const double* corners,
                                                       double* determinants,
                                                       double* relative);
/* Mean-ratio quality 12 (3 V)^(2/3) / sum l^2 of each cell of four point
 * rows: V is the Euclidean volume times exp(mean_k log det M_k / 2) and l the
 * six metric edge lengths in the order (0,1), (0,2), (0,3), (1,2), (1,3),
 * (2,3). 0 unless V and the sum are both positive; NaN when both are
 * infinite. */
PHX_MC_API int32_t phx_mc_metric_tetra_quality(int64_t cell_count, const int64_t* cells,
                                               int64_t point_count, const double* points,
                                               const double* metrics, double* quality);
/* Goals of metric edits on the deterministic kernels: points an edit would
 * like to reach, the same bits for the same input on every platform. They
 * propose only; the caller rounds a goal to an exact source point and decides
 * the edit with exact predicates and the measured objective.
 *
 * phx_mc_metric_midpoints: for each segment from first to second with the
 * metrics at its ends, parameters[r] is t = log1p((q - 1) / 2) / log1p(q - 1)
 * with q = sqrt(d^T M_second d / d^T M_first d), d = second - first (the
 * point at half the metric length when the size varies geometrically), and
 * 1/2 when q is 1, zero or not finite or a form is not positive and finite;
 * points[3r..] is first + t d. */
PHX_MC_API int32_t phx_mc_metric_midpoints(
    int64_t count, const double* first_points, const double* second_points,
    const double* first_metrics, const double* second_metrics, double* parameters,
    double* points);
/* The three goals of the survivor of a collapse, in the order they are tried.
 * points holds the survivor, the removed vertex and then the merged link
 * (point_count >= 2 rows of three values), metrics the tensors of the first
 * two. With t the midpoint parameter of the collapsed edge, goals[0..3) and
 * goals[3..6) are its points at t and t / 2; goals[6..9) is the center of the
 * smallest ball holding the link in the log-Euclidean mean of the two
 * tensors (Welzl's recursion over a fixed pseudo-random order of the link in
 * the Cholesky frame of that tensor). With corner_count 2 or 3 (else 0) each
 * goal is projected orthogonally onto the affine hull of those corners.
 * valid[g] is 0, with a zero goal, when a step of goal g fails: no link
 * vertex, a tensor that is not positive definite, or a value that is not
 * finite. */
PHX_MC_API int32_t phx_mc_metric_survivor_goals(
    int64_t point_count, const double* points, const double* metrics, int32_t corner_count,
    const double* corners, double* goals, int8_t* valid);
/* Frame of the affine hull of two or three corners (three values each):
 * basis holds *rank orthonormal vectors (two rows of three values; unused
 * rows are zero) spanning the edges from corner 0, by Gram-Schmidt in corner
 * order, and projector (row-major 3x3, exactly symmetric) the orthogonal
 * projector onto them. */
PHX_MC_API int32_t phx_mc_metric_hull_frame(int32_t corner_count, const double* corners,
                                            double* basis, double* projector, int32_t* rank);
/* projected[r] = origin + projector (goals[r] - origin) for count goals. */
PHX_MC_API int32_t phx_mc_metric_project_goals(int64_t count, const double* goals,
                                               const double* origin, const double* projector,
                                               double* projected);
/* Pulls of a relocation of vertex 0 of a star: displacements that would bring
 * its spokes toward the length band. points holds the vertex and then its
 * neighbor_count neighbors, metric the tensor at the vertex, lengths the
 * metric lengths of the spokes and measured which of them count; a spoke is
 * outside when it is measured and its length is outside the band. Written in
 * this order, three values each:
 *   - with jacobian (derivatives of the spoke lengths along the basis_count
 *     vectors of basis, row-major basis_count x neighbor_count), for a deep
 *     search with an outside spoke: the minimum-norm least-squares step that
 *     moves every measured spoke to the band, if it is not zero; then for each
 *     outside spoke the gradient direction of its length, scaled to the size
 *     of its force and pointing to shorten a long spoke, lengthen a short one;
 *   - the spring forces (1 - 1/l) (p_j - p_0) of the outside spokes and their
 *     mean over all spokes: the forces first for a deep search, else the mean;
 *   - for a deep search with an outside spoke, for each outside force
 *     (projected by projector when it is not null): its part orthogonal to
 *     each band normal (p_j - p_0)^T M (projected likewise) it pushes across
 *     the band, measured spokes ordered by their distance to the band edges;
 *     then, for an interior vertex, its part along each nonzero cross product
 *     of two such normals.
 * A pull with a component that is not finite is left out. *count is the
 * number written; PHX_MC_CAPACITY_EXCEEDED when capacity pulls do not hold
 * them (pulls is then unspecified). */
PHX_MC_API int32_t phx_mc_metric_relocation_pulls(
    int64_t neighbor_count, const double* points, const double* metric, const double* lengths,
    const int8_t* measured, int32_t deep, int32_t interior, int32_t basis_count,
    const double* basis, const double* projector, const double* jacobian, int64_t capacity,
    double* pulls, int64_t* count);
/* The exact tetrahedral source of a metric adaptation: exact points, cells
 * and the metric tensors of the points.
 *
 * A carrier is an exact combination of source points with dyadic weights
 * (positive integers over 2^bits that sum to 2^bits) together with its
 * correctly rounded binary64 image. The search is integer arithmetic except
 * for the final comparison of binary64 distances, so its result is the same
 * on every platform.
 *
 * phx_mc_metric_source_create: exact point p is
 * numerator[3 p + axis] / denominator. The 3 * point_count numerators and
 * then the positive denominator are unsigned integers of little-endian
 * 32-bit words: integer i is words[word_offsets[i] .. word_offsets[i + 1]),
 * with signs[i] negative for a negative numerator. points holds the binary64
 * roundings of the exact points, cells four point rows per cell in the order
 * the cells are searched, and metrics and logarithms the tensor of every
 * point and its logarithm (nine values each). The source owns copies of
 * everything. */
typedef struct phx_mc_metric_source phx_mc_metric_source;
PHX_MC_API int32_t phx_mc_metric_source_create(
    int64_t point_count, const double* points, const uint32_t* words,
    const int64_t* word_offsets, const int8_t* signs, int64_t cell_count, const int64_t* cells,
    const double* metrics, const double* logarithms, phx_mc_metric_source** source);
PHX_MC_API void phx_mc_metric_source_free(phx_mc_metric_source* source);
/* Carriers of target that are closer to it than current, at most limit.
 *
 * Location: with simplex_count 0, the first cell whose closed binary64 box
 * holds the target and in which its exact barycentric weights are all
 * nonnegative, among all cells (cell negative) or cell alone; the goal is the
 * target. Otherwise (cell negative) the first simplex (simplex_rows holds
 * three point rows per simplex, the third negative for an edge) with
 * independent edges on which the orthogonal projection of the target has
 * nonnegative weights; the goal is the correctly rounded projection. Without
 * a location, with a goal at current or a support of zero extent there is no
 * carrier.
 *
 * Resolutions: at 2^-bits the partial sums of the weights in located order
 * are rounded half to even, made monotone and capped at one; each weight is
 * the difference of consecutive rounded sums. With
 * first = floor(log2(extent / |goal - current|)) clamped to [1, 53], order 0
 * tries first..53, order 1 tries 53..first, and order 2 tries first.. up to
 * its first carrier and then 53..first.
 *
 * A carrier is written when its rounded image is closer to the goal than
 * current (squared binary64 distances summed x, y, z) and, on a simplex, when
 * the image is exact or declared[s] is nonzero. Per carrier: four rows in
 * ascending order (unused -1) with their weights (unused 0), bits, three
 * point values and whether the image is exact. *count is the number
 * written. */
PHX_MC_API int32_t phx_mc_metric_source_carriers(
    const phx_mc_metric_source* source, const double* target, const double* current,
    int64_t cell, int64_t simplex_count, const int64_t* simplex_rows, const int8_t* declared,
    int32_t order, int64_t limit, int64_t* rows, int64_t* weights, int32_t* bits,
    double* points, int8_t* exact, int64_t* count);
/* Positions of the survivor of an edge collapse.
 *
 * The star: points (three values each) and metrics (nine each) with the
 * survivor in row 0; spokes, the star row (not 0) of the other end of each
 * measured spoke of the survivor after the collapse; before_lengths, the
 * metric lengths of the measured edges the collapse removes; cells, four
 * star rows per cell of the survivor after the collapse (at least one).
 *
 * A position replaces row 0. Its verdict is the first of: 1 it is the other
 * end of a spoke; 5 a length before, or a spoke length at the position, has
 * no finite band quantum; 2 the violations do not strictly decrease (the
 * nonzero quanta above the band, largest first, compared lexicographically
 * with a proper prefix smaller, and on a tie those below it); 6 a cell
 * coordinate is outside the exact predicate domain; 3 a cell is not
 * positively oriented (exact) or its relative determinant does not exceed
 * determinant_floor; 4 a cell quality is not positive and finite; 0 it is
 * admitted. qualities[i] is the smallest cell quality of an admitted
 * position and lengths (spoke_count values per position) its spoke lengths,
 * zero for verdict 1. The same verdicts and values on every platform. */
PHX_MC_API int32_t phx_mc_metric_collapse_positions(
    int64_t point_count, const double* points, const double* metrics, int64_t spoke_count,
    const int64_t* spokes, int64_t before_count, const double* before_lengths,
    int64_t cell_count, const int64_t* cells, double determinant_floor, int64_t position_count,
    const double* positions, const double* position_metrics, int32_t* verdicts,
    double* qualities, double* lengths);
/* Moves of the survivor of an edge collapse to carriers of the source, in the
 * order they are found. The star is that of
 * phx_mc_metric_collapse_positions with the removed vertex in row 1 and the
 * merged link after it.
 *
 * The three goals of phx_mc_metric_survivor_goals are tried in order,
 * projected onto the hull of the first simplex when simplex_count is not 0
 * (every simplex is then an edge or a triangle). The carriers of a goal are
 * those of phx_mc_metric_source_carriers from the survivor, order 0: all of
 * them with every_resolution, the first only without. A carrier at the
 * survivor or at the removed vertex is skipped, and so is one whose tensor,
 * the log-Euclidean mean of the source tensors over its rows, does not
 * exist. Every other carrier is evaluated as a position: an admitted one is
 * a move.
 *
 * *end: 0 every carrier was evaluated; 1 the last move leaves no spoke above
 * the band and ends the search; 2 a goal is not a finite point, the moves
 * before it stand; 3 and 4 the last entry is not a move but a carrier with
 * verdict 5 or 6, which the caller reports. Per entry: the goal (0, 1, 2),
 * the carrier as in phx_mc_metric_source_carriers, its tensor, its smallest
 * cell quality and its spoke lengths (spoke_count values). capacity is the
 * number of entries the arrays hold, at least 159. */
PHX_MC_API int32_t phx_mc_metric_source_survivor_moves(
    const phx_mc_metric_source* source, int64_t point_count, const double* points,
    const double* metrics, int64_t spoke_count, const int64_t* spokes, int64_t before_count,
    const double* before_lengths, int64_t cell_count, const int64_t* cells,
    double determinant_floor, int64_t simplex_count, const int64_t* simplex_rows,
    const int8_t* declared, int32_t every_resolution, int64_t capacity, int32_t* goals,
    int64_t* rows, int64_t* weights, int32_t* bits, double* carrier_points,
    double* carrier_metrics, double* qualities, double* lengths, int64_t* count, int32_t* end);
/* Angles with one binary64 result per input on every platform: the values
 * the native meshers use, for host code that must agree with them. Rows are
 * independent and evaluated in order.
 *
 * phx_mc_angle_arcsine and phx_mc_angle_arccosine: asin and acos of a value
 * in [-1, 1] by binary64 operations in a fixed order, within 7 and 5 units in
 * the last place; NaN outside the interval and for NaN. asin keeps the sign
 * of a zero and acos(1) is +0.
 *
 * phx_mc_angle_between: the angle in [0, pi] of the direction
 * (along[r], across[r]) for across[r] >= 0, atan2(across, along), within 7
 * units in the last place: between two vectors, from the length of their
 * cross product and their dot product. The quadrant follows the sign of
 * along exactly; (0, 0) gives 0; a negative across and a value that is not
 * finite give NaN.
 *
 * phx_mc_angle_arctangent2: atan2(ordinates[r], abscissas[r]) in [-pi, pi],
 * within 7 units in the last place: phx_mc_angle_between of the magnitude of
 * the ordinate, with the ordinate's sign. Zeros follow IEEE 754: the result
 * has the sign of the ordinate, also of a negative zero, and a zero ordinate
 * gives a zero for an abscissa that is positive or +0 and pi for one that is
 * negative or -0. A value that is not finite gives NaN.
 *
 * phx_mc_angle_sine and phx_mc_angle_cosine: sin and cos of a finite value
 * as the lower endpoint of phx_mc_interval_elementary's enclosure, at most
 * the true value and at most four binary64 values below it; NaN for a value
 * that is not finite. Slower than the three kernels above.
 *
 * None is correctly rounded and no exact decision may rest on them; a bound
 * comes from phx_mc_interval_elementary. */
PHX_MC_API int32_t phx_mc_angle_sine(int64_t count, const double* values, double* results);
PHX_MC_API int32_t phx_mc_angle_cosine(int64_t count, const double* values, double* results);
PHX_MC_API int32_t phx_mc_angle_arcsine(int64_t count, const double* values, double* results);
PHX_MC_API int32_t phx_mc_angle_arccosine(int64_t count, const double* values,
                                          double* results);
PHX_MC_API int32_t phx_mc_angle_between(int64_t count, const double* across,
                                        const double* along, double* results);
PHX_MC_API int32_t phx_mc_angle_arctangent2(int64_t count, const double* ordinates,
                                            const double* abscissas, double* results);
/* Candidate positions of one relocated vertex, as a sequence.
 *
 * The star: row 0 of points and metrics is the vertex, rows 1.. its
 * neighbors; spoke j runs from row 0 to row j + 1 (point_count - 1 spokes).
 * measured[j] says whether spoke j counts toward the band violations and
 * before_lengths[j] is its metric length now; before_quality is the smallest
 * cell quality now; cells holds four star rows per cell.
 *
 * Each pull p (three values) gives the goals x + 2^-s p, s = 0, 1, ..., until
 * the goal is x (s <= 5 unless flag 1 is set). A goal is projected with
 * phx_mc_metric_project_goals(origin, projector) when both are given, and a
 * projected goal seen before in this search is passed over. With flag 8 (a
 * constant field) a goal is kept when its position with the vertex's own
 * tensor is admitted as below; without it when every cell of the star has a
 * positive determinant and a relative determinant above determinant_floor.
 * A kept goal is replaced by the carriers of
 * phx_mc_metric_source_carriers from the vertex, on the simplices or, with
 * simplex_count 0, in the source cells: order 0 and the first carrier; with
 * flag 1 order 2 and the first two; with flag 2 order 1 and the first, and
 * once every goal has been treated the kept goals again in order with the
 * carriers after the first. A carrier at the vertex or at a neighbor is
 * skipped, and so is one whose support and weights were examined before.
 * Its tensor is the log-Euclidean mean of the source tensors over its rows.
 *
 * A position is admitted when no neighbor lies at it; the band violations
 * of its measured spokes are not larger than before, and smaller with flag
 * 4; every cell has a positive determinant and a relative determinant above
 * determinant_floor; and, when the violations are unchanged, its smallest
 * cell quality exceeds before_quality + improvement.
 *
 * flags: 1 deep, 2 refine, 4 band-only, 8 constant field. The search copies
 * its arrays; the source must outlive it. PHX_MC_INVALID_INPUT when a
 * measured spoke has no finite length now.
 *
 * phx_mc_metric_relocation_next writes the candidates of the next goal that
 * has any, in the order of its carriers: the carrier as in
 * phx_mc_metric_source_carriers, its tensor, the smallest cell quality and
 * the spoke lengths (point_count - 1 values per candidate). *end: 0 a group
 * of *count candidates; 1 no goal is left. Otherwise the search cannot go
 * on and one entry is written for the caller to report: 2 a goal (rows -1,
 * the vertex's tensor, lengths written) whose screen met a spoke without a
 * finite quantum; 3 a carrier whose evaluation met one (lengths written);
 * 4 a carrier whose tensor does not exist. capacity is the number of
 * entries the arrays hold, at least 54. The sequence does not depend on how
 * the calls are spaced, and it is the same on every platform. */
typedef struct phx_mc_metric_relocation phx_mc_metric_relocation;
PHX_MC_API int32_t phx_mc_metric_relocation_create(
    const phx_mc_metric_source* source, int64_t point_count, const double* points,
    const double* metrics, const int8_t* measured, const double* before_lengths,
    double before_quality, int64_t cell_count, const int64_t* cells, double determinant_floor,
    double improvement, int64_t pull_count, const double* pulls, const double* origin,
    const double* projector, int64_t simplex_count, const int64_t* simplex_rows,
    const int8_t* declared, int32_t flags, phx_mc_metric_relocation** search);
PHX_MC_API void phx_mc_metric_relocation_free(phx_mc_metric_relocation* search);
PHX_MC_API int32_t phx_mc_metric_relocation_next(
    phx_mc_metric_relocation* search, int64_t capacity, int64_t* rows, int64_t* weights,
    int32_t* bits, double* carrier_points, double* carrier_metrics, double* qualities,
    double* lengths, int64_t* count, int32_t* end);
/* Gates of a metric field on the deterministic kernels: the same verdict for
 * the same input on every platform.
 *
 * phx_mc_metric_tensor_properties: each tensor is a full row-major 3x3 array.
 * With lambda the eigenvalues of its symmetric part (M + M^T) / 2 and
 * tol = 64 * 2^-52 * max |lambda|, symmetric[r] is 1 when max |M - M^T| <= tol
 * and positive_definite[r] is 1 when it is symmetric and min lambda > tol.
 * Both are 0 for a non-finite entry or an eigenvalue beyond the binary64
 * range. */
PHX_MC_API int32_t phx_mc_metric_tensor_properties(int64_t count, const double* tensors,
                                                   int8_t* symmetric,
                                                   int8_t* positive_definite);
/* Defect of the congruence A^T M A = R, the form in which a metric M at the
 * image of a linear map A agrees with the metric R at its source (row-major
 * 3x3 arrays): errors[r] is the infinity norm of A^T M A - R and scales[r]
 * the larger of the infinity norms of A^T M A and R. Both are NaN when a
 * value is not a number. */
PHX_MC_API int32_t phx_mc_metric_congruence_defects(
    int64_t count, const double* linear, const double* metrics, const double* references,
    double* errors, double* scales);
/* Execute an interval program as phx_mc_interval_program_execute, then record
 * per-box branch evidence: smooth[b] and continuous[b] start at 1 for every
 * box b < box_count, and each event row (kind, test, size, a, a_index, boxes;
 * six int64 fields) clears them where its test holds on a workspace entry
 * a + (a_index < 0 ? k : indices[a_index + k]), k < size: kind 0 (kink) clears
 * smooth, kind 1 (jump) clears smooth and continuous, for box
 * indices[boxes + k], or every box when that entry or `boxes` is -1. Tests:
 * 0 the enclosure meets zero, 1 lower <= 0 (or NaN), 2 lower < 0 (or NaN),
 * 3 lower != upper, 4 a nonfinite endpoint, 5 always. The program and the
 * events are validated before any write. */
PHX_MC_API int32_t phx_mc_interval_program_branches(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, double* lower,
    double* upper, int64_t workspace_size, const int64_t* events, int64_t event_count,
    int64_t box_count, uint8_t* smooth, uint8_t* continuous);
PHX_MC_API int32_t phx_mc_simplex_quadratic_bounds(
    int64_t count, int32_t dimension, int32_t components, const double* q_lower,
    const double* q_upper, const double* barycentric_lower, const double* barycentric_upper,
    double* lower, double* upper);
/* lower/upper may be null (both or neither); then only verdicts are produced,
 * and cells may be decided by a rigorous binary64 screen; verdicts are
 * identical either way. With bounds requested every cell takes the exact path. */
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

/* ------------------------------------------------------------- exact kernels
 * Batched exact and filtered kernels of Bernstein and box-pair proofs. Every
 * batched kernel takes `parallelism` >= 1 and splits items into fixed chunks by
 * item index; each item writes only its own output slots, so results are
 * bit-identical for every parallelism.
 *
 * Exact integer kernels read and write two's-complement words of
 * 8 * word_limbs little-endian bytes, word_limbs in {4, 8, 16, 32, 64} (256 to
 * 4096 bits) chosen by the caller per call. Every addition, product and shift
 * is checked: an item whose exact value leaves the word range reports
 * PHX_MC_RANGE_ERROR in its item status, its outputs are unspecified, and the
 * caller retries it with wider words or recomputes it on its own exact path.
 * Other items are unaffected.
 * Sizes and intermediate strides must be addressable even for zero items.
 * Malformed dimensions are rejected before output mutation.
 */
enum phx_mc_bernstein_domain {
  PHX_MC_BERNSTEIN_SIMPLEX = 0, /* degree (n); controls ordered by compositions of n
                                   into dimension + 1 parts, first part descending */
  PHX_MC_BERNSTEIN_BOX = 1,     /* degree (n_0..n_{d-1}); C order over control indices */
  PHX_MC_BERNSTEIN_PRISM = 2    /* dimension 3, degree (triangle, line); triangle
                                   compositions of 3 parts, then line index */
};
/* Interval de Casteljau restriction of Bernstein nets to [first, last] of [0, 1]
 * along axis 0. lower/upper/out_* are (count, degree + 1, width). parameters is
 * (count, 4): enclosures (last_lo, last_hi) of `last` and (ratio_lo, ratio_hi)
 * of first / last. sides bit 0 applies the left split (last < 1), bit 1 the
 * right split (first > 0). Each step encloses (1 - t) x_i + t x_{i+1} for t at
 * the parameter endpoints, rounded to nearest, widened by
 * 4 eps ((1 - t)|x_i| + t|x_{i+1}|) + 2^-1074 and stepped outward once. */
PHX_MC_API int32_t phx_mc_interval_bernstein_restrict(
    int64_t count, int32_t degree, int64_t width, const double* lower, const double* upper,
    const double* parameters, const int32_t* sides, double* out_lower, double* out_upper,
    int32_t parallelism);
/* Elementwise set-based interval product of count aligned interval pairs: a
 * zero endpoint times an unbounded one contributes 0, a NaN endpoint product
 * gives (-inf, inf). Each endpoint product is enclosed independently before
 * extrema reduction: it rounds outward one step unless proved error free
 * (Dekker TwoProduct within its exact window, or an exact zero factor).
 * Sign-fixed results keep their sign unless the NaN fallback applies. */
PHX_MC_API int32_t phx_mc_interval_multiply(
    int64_t count, const double* first_lower, const double* first_upper,
    const double* second_lower, const double* second_upper, double* lower, double* upper);
/* Elementwise interval sum (subtract = 0) or difference (subtract = 1) of count
 * aligned interval pairs: an endpoint rounds outward one step only when its
 * binary64 sum is not error free (TwoSum). Either NaN sum endpoint forces the
 * whole line, overriding exactness and sign retention of the other endpoint. */
PHX_MC_API int32_t phx_mc_interval_sum(
    int64_t count, const double* first_lower, const double* first_upper,
    const double* second_lower, const double* second_upper, int32_t subtract, double* lower,
    double* upper);
/* Batched simplex Bernstein restriction. Controls are shared across all items
 * and ordered by descending compositions of degree into dimension + 1 parts.
 * Corners have shape (count, dimension + 1, dimension); each lower/upper pair
 * encloses a Cartesian coordinate, including coordinates outside the reference
 * simplex. Results enclose all restricted controls, hence the polynomial on
 * each simplex. Nonfinite or reversed inputs and nonfinite intermediate
 * results produce the whole real line for the affected item. Dimension is
 * 1..3, degree and count are nonnegative, and workers is positive. Inputs and
 * outputs must not overlap. */
PHX_MC_API int32_t phx_mc_interval_simplex_restrict(
    int32_t dimension, int32_t degree, int64_t count, const double* control_lower,
    const double* control_upper, const double* corners_lower, const double* corners_upper,
    double* result_lower, double* result_upper, int32_t workers);
/* Bounds of linear combinations of common-degree Bernstein control banks.
 * Basis endpoints are (basis_count, controls), coefficient endpoints are
 * (count, basis_count, components), outputs are (count, components).
 * The inner endpoints bracket each exact extremal control from the opposite
 * side; they bound the possible excess of the outer endpoints. Input banks
 * may overlap each other, but output banks must not overlap any input or output
 * bank; overlapping banks are rejected with PHX_MC_INVALID_ARGUMENT. */
PHX_MC_API int32_t phx_mc_interval_basis_bounds(
    int64_t count, int64_t basis_count, int64_t controls, int64_t components,
    const double* basis_lower, const double* basis_upper,
    const double* coefficient_lower, const double* coefficient_upper,
    double* result_lower, double* result_upper, double* inner_lower, double* inner_upper,
    int32_t workers);
#define PHX_MC_ELEMENTARY_EXP 0
#define PHX_MC_ELEMENTARY_LOG 1
#define PHX_MC_ELEMENTARY_SIN 2
#define PHX_MC_ELEMENTARY_COS 3
#define PHX_MC_ELEMENTARY_TAN 4
#define PHX_MC_ELEMENTARY_ATAN 5
#define PHX_MC_ELEMENTARY_TANH 6
#define PHX_MC_ELEMENTARY_ASIN 7
#define PHX_MC_ELEMENTARY_ACOS 8
/* Elementwise rigorous enclosure [out_lower, out_upper] of the selected f over
 * each box [lower, upper] of count items (lower <= upper unless an endpoint is
 * NaN; infinite endpoints bound the real set and are never members), computed
 * without a platform libm from binary64 basic operations and fma in
 * round-to-nearest with gradual underflow. That mode is established before any
 * argument is inspected; if it cannot be, the call returns
 * PHX_MC_INTERNAL_ERROR. count < 0, parallelism < 1, an unknown function code
 * or a null pointer with count > 0 returns PHX_MC_INVALID_ARGUMENT. Outputs may
 * alias inputs (element i is read before it is written). Results are
 * bit-identical for every parallelism and platform.
 * Rules:
 *   - A NaN endpoint gives (-inf, inf) for every f.
 *   - Monotone f (exp, log, atan, tanh) bound f at the lower endpoint from below
 *     and at the upper endpoint from above (an inverted box stays inverted).
 *   - exp: exp(-inf) contributes 0 below and exp([-inf, -inf]) = [0, 2^-1074];
 *     exp(+inf) contributes +inf above and exp([+inf, +inf]) = [DBL_MAX, +inf].
 *   - log: lower <= 0 gives out_lower = -inf and upper <= 0 gives
 *     out_upper = +inf (a box <= 0 maps to the whole line); log(+inf) is
 *     bounded by [DBL_MAX, +inf].
 *   - atan(+-inf) contributes the outward-rounded +-pi/2 bound; tanh(+-inf)
 *     contributes +-1 exactly.
 *   - asin (increasing) and acos (decreasing): the box is clipped to [-1, 1];
 *     a box with no point in [-1, 1] (upper < -1 or lower > 1) gives the whole
 *     line, as log does for an empty domain. asin(+-1), acos(0) and acos(-1)
 *     enclose +-pi/2, pi/2 and pi outward.
 *   - sin, cos: an infinite endpoint, an inverted box or a width >= 2 pi
 *     (decided exactly) gives [-1, 1]; otherwise the endpoint enclosures widened
 *     to -1 or 1 wherever an extremum may lie in the box, clipped to [-1, 1].
 *   - tan: an infinite endpoint, an inverted box or a possible pole (decided or
 *     undecided) gives (-inf, +inf); otherwise [bound at lower, bound at upper].
 *   - The argument reduction of sin, cos and tan is exact Payne-Hanek with a
 *     proven error bound for every finite binary64 x (R = DBL_MAX), so no finite
 *     argument falls back to the full range; extremum and pole membership is
 *     decided from the reduced quadrants, undecided meaning contained.
 *   - Exact results are returned exactly: exp(0) = 1, log(1) = 0, cos(0) = 1,
 *     acos(1) = 0, and sin, tan, atan, tanh, asin of a signed zero return that
 *     zero; other zero results are +0, and zeros order as IEEE 754-2019
 *     minimum/maximum.
 * Point enclosures of finite non-exact results are one or two binary64 steps
 * wide; the error proofs are documented in src/elementary.cpp. */
PHX_MC_API int32_t phx_mc_interval_elementary(
    int32_t function, int64_t count, const double* lower, const double* upper,
    double* out_lower, double* out_upper, int32_t parallelism);
/* Exact Bernstein controls of polynomials sum_t N_t x^{e_t} / Q on a domain.
 * Item i owns terms [term_offsets[i], term_offsets[i + 1]) with exponents
 * (terms, dimension) and numerators (terms) words, degrees (count, degree_width)
 * with degree_width 1 (simplex), dimension (box) or 2 (prism), and controls
 * [control_offsets[i], control_offsets[i + 1]) of the domain's control count.
 * Control c equals controls[c] / (Q * denominators[i]); extremes[i] holds the
 * first local index of a least and of a greatest control. */
PHX_MC_API int32_t phx_mc_bernstein_controls(
    int64_t count, int32_t domain, int32_t dimension, const int32_t* degrees,
    const int64_t* term_offsets, const int32_t* exponents, const uint8_t* numerators,
    const int64_t* control_offsets, uint8_t* controls, int64_t* denominators,
    int64_t* extremes, int32_t* item_status, int32_t word_limbs, int32_t parallelism);
/* Ordinary Jacobian determinant integrated via its Bernstein control mean.
 * Matrix entries use the offsets/exponents/coefficients layout above. Each
 * successful item returns the signed control sum, its positive common control
 * denominator and control count. For input coefficient denominator Q, divide
 * the sum by Q^dimension, denominator and count, then multiply by the exact
 * reference volume (1/dimension! for simplex, 1 for box, 1/2 for prism).
 * Overflow or excessive control counts return per-item RANGE_ERROR. */
PHX_MC_API int32_t phx_mc_polynomial_jacobian_integrals(
    int64_t count, int32_t dimension, int32_t domain, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients, uint8_t* sums,
    int64_t* denominators, int64_t* counts, int32_t* item_status,
    int32_t word_limbs, int32_t parallelism);
/* Shared nodal basis and routed exact coordinates. Sparse coordinates use
 * numerator/denominator words; a zero denominator refuses the owning item.
 * Successful sums are divided by (basis scale * returned coordinate scale)^d,
 * then by the returned control denominator and count and by the reference
 * volume denominator, as in phx_mc_polynomial_jacobian_integrals. */
PHX_MC_API int32_t phx_mc_basis_jacobian_integrals(
    int64_t count, int32_t dimension, int32_t domain, int64_t nodes, int64_t bank_rows,
    const int64_t* routes, const int64_t* significands, const int16_t* exponents,
    const int64_t* sparse_index, const int64_t* sparse_rows,
    const uint8_t* sparse_numerators, const uint8_t* sparse_denominators,
    const int64_t* basis_offsets, const int32_t* basis_exponents, const uint8_t* basis,
    uint8_t* sums, uint8_t* scales, int64_t* denominators, int64_t* counts,
    int32_t* item_status, int32_t word_limbs, int32_t parallelism);
/* Leading principal minors of J + J^T. Each matrix entry owns a term range
 * in offsets, with integer coefficients and dimension-wide nonnegative
 * exponents. Output slots are item-major then minor order; each has capacity
 * output_offsets[s+1]-output_offsets[s]. counts gives the written term count.
 * Checked overflow or insufficient per-slot capacity reports RANGE_ERROR for
 * that item. Caller supplies the common positive input coefficient scale Q;
 * minor m represents its integer output divided by (2 Q)^(m+1). */
PHX_MC_API int32_t phx_mc_symmetric_polynomial_minors(
    int64_t count, int32_t dimension, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients,
    const int64_t* output_offsets, int32_t* out_exponents,
    uint8_t* out_coefficients, int64_t* counts, int32_t* item_status,
    int32_t word_limbs, int32_t parallelism);
/* Fused leading-minor Bernstein sign proofs. scales contains one positive
 * integer word Q per Jacobian. verdicts[i * dimension + m] is 1 when every
 * control of leading minor m is strictly positive, 0 otherwise. On either
 * verdict, digests stores 32 SHA-256 bytes per minor in item-major order.
 * RANGE_ERROR is a per-item typed refusal.
 * For each control set, write its controls as integer numerators n[i] over a
 * common positive denominator Q, then divide Q and every n[i] by their joint
 * gcd. The SHA-256 input begins with 24 bytes: ASCII
 * "phydrax-exact-controls", a zero byte, and ASCII 'R'. Next comes the
 * control count as an unsigned 64-bit big-endian integer, then Q as a framed
 * magnitude, then each n[i] in control order as a sign byte (0 nonnegative,
 * 1 negative) and framed magnitude. A framed magnitude is its unsigned 64-bit
 * big-endian byte length followed by its minimal unsigned big-endian bytes;
 * zero has length zero and a nonnegative sign. */
PHX_MC_API int32_t phx_mc_symmetric_polynomial_verdicts(
    int64_t count, int32_t dimension, int32_t domain, const int64_t* offsets,
    const int32_t* exponents, const uint8_t* coefficients, const uint8_t* scales,
    uint8_t* digests, int32_t* verdicts, int32_t* item_status,
    int32_t word_limbs, int32_t parallelism);
/* Restricted controls of simplex Bernstein nets: item i has degree degrees[i],
 * `parts` = simplex dimension + 1, controls [control_offsets[i], ...) in
 * composition order and corners (count, parts, parts) words, row j holding the
 * integer barycentric weights of corner j over one common positive scale s.
 * Output entry m (a composition of the degree: corner multiplicities) is the
 * blossom at the corners times s^degree; extremes as for bernstein_controls. */
PHX_MC_API int32_t phx_mc_simplex_bernstein_blossoms(
    int64_t count, int32_t parts, const int32_t* degrees, const int64_t* control_offsets,
    const uint8_t* controls, const uint8_t* corners, uint8_t* values, int64_t* extremes,
    int32_t* item_status, int32_t word_limbs, int32_t parallelism);
/* Halving de Casteljau split of multi-index Bernstein nets along slots
 * (first, second): indices is (control_count, slot_count); nets, left and right
 * are (count, control_count, components) words. The left half keeps slot
 * `first`; outputs are numerators over 2^(*exponent) times the input scale. */
PHX_MC_API int32_t phx_mc_bernstein_bisect(
    int64_t count, int64_t control_count, int32_t slot_count, int32_t components,
    const int32_t* indices, int32_t first, int32_t second, const uint8_t* nets,
    uint8_t* left, uint8_t* right, int32_t* exponent, int32_t* item_status,
    int32_t word_limbs, int32_t parallelism);
#define PHX_MC_SEPARATION_UNDECIDED 0
#define PHX_MC_SEPARATION_SEPARATED 1
#define PHX_MC_SEPARATION_NOT_SEPARATED 2
/* Filtered plane screen of point-set pairs: pair i has first points
 * [first_offsets[i], first_offsets[i + 1]) and second points likewise, each a
 * (points, dimension) interval box, and a binary64 normal (count, dimension).
 * SEPARATED proves max n.b < min n.a over the boxes, NOT_SEPARATED proves
 * max n.b >= min n.a for every point in the boxes, UNDECIDED leaves the exact
 * decision to the caller. */
PHX_MC_API int32_t phx_mc_separation_screen(
    int64_t count, int32_t dimension, const int64_t* first_offsets,
    const double* first_lower, const double* first_upper, const int64_t* second_offsets,
    const double* second_lower, const double* second_upper, const double* normals,
    int8_t* decisions, int32_t parallelism);
/* Simultaneous traversal of two packed bounding-volume hierarchies yielding
 * overlapping item pairs in bounded blocks. Arrays follow PackedBVH: node
 * bounds (nodes, dimension), node_size (nodes) the descent key (the caller's
 * sum of node extents), left/right/leaf (nodes), leaf items (leaves,
 * leaf_width) with -1 padding, item bounds (items, dimension). The caller keeps
 * every array alive and unchanged until phx_mc_bvh_pair_stream_free.
 * Reachable nodes must form a rooted tree (no cycles or shared children);
 * leaf nodes have left = right = -1. Validation is iterative without a depth cap.
 * Blocks follow the LIFO node-block traversal of the host reference: node blocks are
 * split at node_block pairs, leaf pairs are tested in chunks of
 * max(1, item_block / (leaf_width_a * leaf_width_b)), and every chunk with a hit
 * is one block. A non-leaf node of the first tree descends when the second
 * node is a leaf or node_size is not smaller. Boxes overlap when
 * min(max) - max(min) > 0 on every axis or, with include_touching,
 * >= -(atol + rtol * max(1, |bounds|)). */
typedef struct phx_mc_bvh_tree_view {
  int64_t node_count;
  int64_t leaf_count;
  int64_t leaf_width;
  int64_t item_count;
  const double* node_min;
  const double* node_max;
  const double* node_size;
  const int64_t* left;
  const int64_t* right;
  const int64_t* leaf;
  const int64_t* leaf_items;
  const double* item_min;
  const double* item_max;
} phx_mc_bvh_tree_view;
typedef struct phx_mc_bvh_pair_stream phx_mc_bvh_pair_stream;
PHX_MC_API int32_t phx_mc_bvh_pair_stream_create(
    const phx_mc_bvh_tree_view* first, const phx_mc_bvh_tree_view* second, int32_t dimension,
    int32_t include_touching, double absolute_tolerance, double relative_tolerance,
    int64_t item_block, int64_t node_block, int32_t parallelism,
    phx_mc_bvh_pair_stream** stream);
/* Copies whole blocks while they fit: pair_capacity must hold at least
 * max(1, item_block / (leaf_width_a * leaf_width_b)) * leaf_width_a * leaf_width_b
 * pairs. block_ends receives the exclusive end of each copied block.
 * On PHX_MC_OK, *blocks == 0 means traversal is complete. Invalid capacities
 * do not consume work and may be retried. An operational failure (including
 * allocation failure) is terminal: subsequent valid calls return that failure,
 * never successful incomplete traversal. Outputs of a failed call are
 * unspecified; the handle remains safe to free. */
PHX_MC_API int32_t phx_mc_bvh_pair_stream_next(
    phx_mc_bvh_pair_stream* stream, int64_t pair_capacity, int64_t block_capacity,
    int64_t* first_items, int64_t* second_items, int64_t* block_ends, int64_t* blocks);
PHX_MC_API void phx_mc_bvh_pair_stream_free(phx_mc_bvh_pair_stream* stream);

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
  /* -1 selects all images; nonnegative counts publish exactly these image IDs.
   * IDs may be unordered but must be distinct and in [0, image_count).
   * Actual face neighbors are clipped as support to determine remote
   * components; support site neighbors are not recursively expanded and
   * support cells are not published. All clipped pieces require a
   * representable positive binary64 measure. */
  int64_t selected_site_count;
  const int32_t* selected_sites;
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
/* Per face, an unselected counterpart image and an incident tet of its
 * component, or -1 in both arrays when there is no remote counterpart. A
 * remote face has second cell -1; its facet remains the input facet if it
 * lies on a constrained tet face. The witness identifies a real piece, not
 * a fabricated remote component. */
PHX_MC_API int32_t phx_mc_power_cells_export_face_remotes(
    const phx_mc_power_cells* result, int64_t face_capacity,
    int32_t* remote_sites, int32_t* remote_tets);
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
PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    double* center_bounds, int32_t* item_status);

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
/* Remeshes a created envelope surface in place toward edge length 2 * spacing and a
 * 30 degree minimum angle: floating-point construction (vertices projected onto the
 * offset of the covering cloud), then certification of every triangle inside the
 * certified distance band [bounds[2], bounds[1]] and of the surface as an embedded
 * oriented closed 2-manifold, refused triangles repaired locally by splits; a
 * triangle left refused is PHX_MC_UNRESOLVED and leaves the envelope unchanged.
 * Writes the updated four bounds and counters (splits, collapses, flips,
 * relocations, triangles below the angle). The tetrahedral carrier is dropped: the
 * volume vertices become the surface vertices, no cells. */
PHX_MC_API int32_t phx_mc_surface_envelope_remesh(void* result, double* bounds,
                                                  int64_t* counters);

/* ------------------------------------------------------ zero-set surface remesh */
/* One straight-line interval program (phx_mc_interval_program_execute) over a
 * single three-dimensional box: the workspace template carries the constant seeds,
 * entries [0, 3) receive the box, and `outputs` name the result entries. */
typedef struct phx_mc_interval_box_program {
  const int64_t* instructions;
  int64_t instruction_count;
  const int64_t* indices;
  int64_t index_count;
  const double* parameters;
  int64_t parameter_count;
  const double* workspace_lower;
  const double* workspace_upper;
  int64_t workspace_size;
  const int64_t* outputs;
  int64_t output_count;
  const int64_t* guards; /* entries whose lower bound must be >= 0 (real-power bases) */
  int64_t guard_count;
} phx_mc_interval_box_program;
/* A TetMesh implicit source owns both validated programs after binding. Groups
 * name actual exterior facet-source rows, all proved against the programs on
 * bind. Zero orientation asks the owner to derive the common certified sign;
 * a nonzero sign must agree. A view returned by `implicit_source` borrows
 * immutable banks until the mesh is freed. Refusal counts are diagnostic and
 * restart at zero on a new binding. */
typedef struct phx_mc_tet_implicit_source {
  phx_mc_interval_box_program value;
  phx_mc_interval_box_program gradient;
  const int32_t* groups;
  int64_t group_count;
  double tolerance;
  int32_t orientation;
  const char* source_id;
  int64_t source_id_length;
  const char* source_revision;
  int64_t source_revision_length;
  const char* binding_id;
  int64_t binding_id_length;
  const char* frame_id;
  int64_t frame_id_length;
  int64_t transversality_refusals;
  int64_t distance_refusals;
} phx_mc_tet_implicit_source;
PHX_MC_API int32_t phx_mc_tet_mesh_bind_implicit(
    phx_mc_tet_mesh* mesh, const phx_mc_tet_implicit_source* source);
/* Constructs/restores with implicit admission installed before source-face
 * validation, so already edited faces are never reinterpreted as PWL faces. */
PHX_MC_API int32_t phx_mc_tet_mesh_create_with_implicit(
    const phx_mc_tet_mesh_create_request* request,
    const phx_mc_tet_implicit_source* source);
PHX_MC_API int32_t phx_mc_tet_mesh_implicit_source(
    const phx_mc_tet_mesh* mesh, phx_mc_tet_implicit_source* source);
/* Remeshes a closed, consistently oriented carrier of the zero set of f toward
 * `edge_length` (lowered where the zero set curves, so that a well-shaped
 * triangle stays within `tolerance`) and `minimum_angle_degrees`. Closed is
 * validated: every edge is held by exactly two triangles, in opposite
 * directions, else PHX_MC_INVALID_INPUT. Construction only, floating point,
 * proving nothing: edge splits, collapses, flips and tangential relocations
 * on flat arrays, new and moved vertices projected onto {f = 0} by Newton
 * steps on second-order point jets of `value` (one output). Input edges
 * whose triangles turn by more than 45 degrees are creases: every input
 * triangle touching a crease vertex is kept bitwise (its corners and edges
 * are never moved, split, flipped or collapsed). A vertex further than
 * edge_length + carrier_distance from its projection is PHX_MC_UNRESOLVED.
 * Writes counters
 * (splits, collapses, flips, relocations, triangles below the minimum angle,
 * the carrier's orientation relative to grad f by the floating-point majority
 * of the input triangles, triangles whose sampled distance exceeds
 * `tolerance`, 0). `result` owns the surface (phx_mc_surface_result_*).
 * Certify the result with phx_mc_zero_set_certify. */
typedef struct phx_mc_zero_set_remesh_request {
  const double* vertices;
  int64_t vertex_count;
  const int64_t* triangles;
  int64_t triangle_count;
  phx_mc_interval_box_program value;
  double tolerance;
  double carrier_distance;
  double edge_length;
  double minimum_angle_degrees;
  void** result;
  int64_t* counters;
} phx_mc_zero_set_remesh_request;
PHX_MC_API int32_t phx_mc_zero_set_remesh(const phx_mc_zero_set_remesh_request* request);
/* As above, request->triangles contains int32 indices. */
PHX_MC_API int32_t phx_mc_zero_set_remesh_i32(const phx_mc_zero_set_remesh_request* request);
/* Certifies a closed triangle surface (a phx_mc_zero_set_remesh result)
 * against the zero set of f, repairing it locally. A triangle is certified
 * when its distance to {f = 0} is at most `tolerance` (interval |f| over the
 * triangle against a certified lower bound of |grad f| over its widened box,
 * subdividing where needed) and its normal is transversal to grad f with
 * `orientation` (+1 or -1), or, when that proof fails, when it has the
 * corner coordinates, in the same cyclic order, of a carrier triangle (the
 * surface remeshed; a NULL carrier with zero counts accepts none) whose
 * witness box proves it: carrier_boxes holds, per carrier triangle, a box
 * (lower xyz, upper xyz) that contains the triangle's three corners (exact
 * comparisons) and has two corners at which the interval enclosures of f
 * have strictly opposite signs, so the box holds a zero of f on the segment
 * between them; such a triangle is accepted at the box diagonal (rounded
 * up). A carrier triangle whose box fails either check is never accepted
 * from the carrier. And a triangle is certified only when it is nondegenerate
 * with
 * only legal contacts (disjoint, a shared vertex or a shared edge, exact)
 * with every triangle whose box meets its own. The surface must be a closed,
 * consistently oriented 2-manifold (else PHX_MC_INVALID_INPUT). Refused
 * triangles are repaired: the longest edge of each, unless kept at a crease
 * (see phx_mc_zero_set_remesh), is split at its midpoint projected onto
 * {f = 0} (edge_length, minimum_angle_degrees and the largest witness
 * diagonal steer the projection as carrier_distance does in
 * phx_mc_zero_set_remesh), Delaunay flips follow, and only
 * changed or refused triangles are decided again, for at most four rounds.
 * `value` has one output, `gradient` three. On PHX_MC_OK `result` owns the
 * certified surface (phx_mc_surface_result_*, with per-triangle carried
 * flags) and `distance` is the maximum over the proved triangles (each at
 * most `tolerance`) and, for triangles accepted from the carrier, their
 * witness diagonals, which may exceed `tolerance`; a
 * triangle left refused is PHX_MC_UNRESOLVED. counters receives (triangle
 * certifications refused for transversality, refused for distance,
 * triangles refused for embedding, all over every round, carrier triangles
 * included; triangles accepted from the carrier; repair splits;
 * certification rounds run) on PHX_MC_OK and PHX_MC_UNRESOLVED. */
typedef struct phx_mc_zero_set_certify_request {
  const double* vertices;
  int64_t vertex_count;
  const int64_t* triangles;
  int64_t triangle_count;
  phx_mc_interval_box_program value;
  phx_mc_interval_box_program gradient;
  double tolerance;
  int32_t orientation;
  const double* carrier_vertices;
  int64_t carrier_vertex_count;
  const int64_t* carrier_triangles;
  int64_t carrier_triangle_count;
  const double* carrier_boxes;
  double edge_length;
  double minimum_angle_degrees;
  void** result;
  double* distance;
  int64_t* counters;
} phx_mc_zero_set_certify_request;
PHX_MC_API int32_t phx_mc_zero_set_certify(const phx_mc_zero_set_certify_request* request);
/* An owned surface result: sizes = (vertices, triangles); arrays copies them
 * (PHX_MC_CAPACITY_EXCEEDED when a capacity is short); carried writes one
 * flag per triangle, 1 for a triangle accepted from the carrier by its
 * witness box (phx_mc_zero_set_certify; 0 for every other result); free
 * releases it. */
PHX_MC_API int32_t phx_mc_surface_result_sizes(const void* result, int64_t* sizes);
PHX_MC_API int32_t phx_mc_surface_result_arrays(void* result, int64_t vertex_capacity,
                                                int64_t triangle_capacity, double* vertices,
                                                int64_t* triangles);
PHX_MC_API int32_t phx_mc_surface_result_carried(const void* result, int64_t triangle_capacity,
                                                 uint8_t* carried);
PHX_MC_API void phx_mc_surface_result_free(void* result);
/* Remeshes a closed, consistently oriented triangle mesh (the source, with
 * nondegenerate triangles; else PHX_MC_INVALID_INPUT) toward `edge_length`
 * (lowered where the source curves, so that a well-shaped triangle stays
 * within `tolerance`) and `minimum_angle_degrees`. Construction only, as
 * phx_mc_zero_set_remesh, with the source as the surface: new and moved
 * vertices go to their nearest point of the source (bucket grid), normals
 * interpolate angle-weighted vertex normals, the two principal curvatures
 * per-vertex estimates (the largest and smallest normal curvature along the
 * ring edges). A source rough at this tolerance (mean |L(l)| above 2% of it,
 * l = L(x) . n the normal part of the uniform Laplacian, as vertex noise
 * gives) is first
 * Taubin-smoothed at a radius of about edge_length / 2, every vertex kept
 * within tolerance / 2 of the input, and construction follows the smoothed
 * copy. Writes counters (splits, collapses, flips, relocations, triangles
 * below the minimum angle, triangles whose sampled distance to the
 * construction source exceeds `tolerance`). `result` owns the surface
 * (phx_mc_surface_result_*). */
typedef struct phx_mc_mesh_remesh_request {
  const double* vertices;
  int64_t vertex_count;
  const int64_t* triangles;
  int64_t triangle_count;
  double tolerance;
  double edge_length;
  double minimum_angle_degrees;
  void** result;
  int64_t* counters;
} phx_mc_mesh_remesh_request;
PHX_MC_API int32_t phx_mc_mesh_remesh(const phx_mc_mesh_remesh_request* request);
/* Certifies a remeshed surface (vertices, triangles) against its source mesh
 * in both directions, repairing it locally as phx_mc_zero_set_certify does.
 * Surface -> source: every triangle T gets an upper bound on the distance
 * from each of its points to the source. Points take planar coordinates
 * along T's normal, and each triangle maps affinely onto the planar
 * triangle of its corners' (binary64) coordinates; the patch (source
 * triangles reached across edges whose planar triangles meet T's) must wind
 * once around T's planar triangle (exact orient2d), so every x = X(y) in T
 * has a patch point Y(y) over the same planar y, and d(x, source) <=
 * |X(y) - Y(y)|, convex over each overlap of planar triangles: the bound is
 * its value at the overlap vertices (corners and edge crossings decided
 * exactly; crossings evaluated within a certified offset, with the maps'
 * slopes and a first-order rounding pad, rounded up). Refused triangles
 * (bound above `tolerance`, or no winding) are split and flipped for at
 * most four rounds, with embedding checked exactly. Source -> surface: the
 * same bound for every source triangle against the certified surface, along
 * the surface's interpolated normal (a source triangle may be rougher than
 * the surface); the surface triangles nearest the corners of a refused
 * source triangle are split. After any repair, angles are restored by flips
 * and relocations and both directions are decided again, for at most three
 * further passes. Both sources must be closed oriented 2-manifolds with
 * nondegenerate triangles. On PHX_MC_OK `result` owns the certified surface
 * (phx_mc_surface_result_*) and distances = (largest surface -> source
 * bound, largest source -> surface bound), each at most `tolerance`; when a
 * pass after one that certified both ways fails (angle relaxation is a
 * target, not part of the certificate), that certified surface is returned.
 * PHX_MC_UNRESOLVED when no pass certifies both ways: a surface triangle
 * stays refused or a source triangle is not proved within `tolerance` (no
 * result). counters receives
 * (triangle certifications refused for distance, triangles refused for
 * embedding, repair splits, rounds run, all summed over every pass; source
 * triangles not proved within `tolerance` in the last pass; relocations
 * restoring angles after repairs) on PHX_MC_OK and PHX_MC_UNRESOLVED,
 * distances on both too (the second +inf when the surface direction
 * failed). */
typedef struct phx_mc_mesh_certify_request {
  const double* source_vertices;
  int64_t source_vertex_count;
  const int64_t* source_triangles;
  int64_t source_triangle_count;
  const double* vertices;
  int64_t vertex_count;
  const int64_t* triangles;
  int64_t triangle_count;
  double tolerance;
  double edge_length;
  double minimum_angle_degrees;
  void** result;
  double* distances;
  int64_t* counters;
} phx_mc_mesh_certify_request;
PHX_MC_API int32_t phx_mc_mesh_certify(const phx_mc_mesh_certify_request* request);

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
/* Paired rows: points[count][3], triangles[count][3][3], distances[count].
 * Their nonempty byte ranges must be pairwise disjoint; overlap returns
 * PHX_MC_INVALID_ARGUMENT before writing. For zero rows, distances must still
 * be nonnull, while points and triangles may be null.
 * Returns max(candidate distance, ordered point-to-triangle-box distance).
 * Inputs and computed distances must be finite; an error can leave a prefix
 * of the output written. Degenerate and poorly conditioned inputs are accepted.
 * This operation-sequence approximation has no proved accuracy bound: it can
 * be above or below the true distance, and a true zero can return a small
 * nonzero value. On 75,066 in-domain rows, the observed error in ulps of the
 * largest input-coordinate magnitude had maximum 41.4, median 0, and p99 2;
 * these observations are not a bound. */
PHX_MC_API int32_t phx_mc_point_triangle_distances(
    int64_t count, const double* points, const double* triangles, double* distances);
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
#define PHX_MC_PLC3D_SEARCH_NODES_DEFAULT (INT64_C(1) << 24)
#define PHX_MC_PLC3D_SEARCH_BYTES_DEFAULT (INT64_C(1) << 30)
#define PHX_MC_PLC3D_SEARCH_VALUES 23
/* Routes of phx_mc_plc3d_suppression_routes (column 3). */
#define PHX_MC_PLC3D_ROUTE_SEARCH 1
#define PHX_MC_PLC3D_ROUTE_PEEL 2
#define PHX_MC_PLC3D_ROUTE_PEEL_AFTER_INCONCLUSIVE 3
#define PHX_MC_PLC3D_ROUTE_VALUES 7
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
  /* Declared budget of the exhaustive Steiner-free suppression search of
   * recovery (deciding whether a cavity has a Steiner-free fill is
   * NP-complete, so the search has no structural bound): the states the
   * searches of one recovery may expand in total, and the bytes one search
   * may hold (40 per expanded state, 4 per memo table slot, 64 per ranked
   * open side plus 4 per apex). Part of the request; 0 selects the
   * documented defaults PHX_MC_PLC3D_SEARCH_NODES_DEFAULT and
   * PHX_MC_PLC3D_SEARCH_BYTES_DEFAULT; negative is INVALID_ARGUMENT. An
   * over-budget search is inconclusive: recovery goes on to relocation by
   * peeling, and a point that also fails it refuses UNRESOLVED /
   * STEINER_SUPPRESSION naming the point, with the search evidence
   * (phx_mc_plc3d_suppression_search, phx_mc_plc3d_suppression_routes).
   * Source preparation runs no search and ignores them. */
  int64_t search_node_budget;
  int64_t search_byte_budget;
  phx_mc_plc3d_result** result;
} phx_mc_plc3d_prepare_source_request;
PHX_MC_API int32_t phx_mc_plc3d_prepare_source(
    const phx_mc_plc3d_prepare_source_request* request);
/* Protected PLC recovery from the same authored source banks. */
PHX_MC_API int32_t phx_mc_plc3d_recover(
    const phx_mc_plc3d_prepare_source_request* request);
/* Row-major index-to-physical linear map; topology and witnesses remain in index space. */
PHX_MC_API int32_t phx_mc_plc3d_recover_with_affine_metric(
    const phx_mc_plc3d_prepare_source_request* request, const double* affine_metric);
/* Facet-count 0/1 protection flags; incident edges are protected as well.
 * The optional affine metric has the same semantics as recover_with_affine_metric. */
PHX_MC_API int32_t phx_mc_plc3d_recover_protected(
    const phx_mc_plc3d_prepare_source_request* request,
    const uint8_t* protected_facets, const double* affine_metric);
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
// Steiner-suppression search evidence: values[0..22] = wedges searched,
// wedges filled, states expanded, most states in one wedge, largest searched
// wedge (cells), wedges filled after merging with neighbouring cells, most
// merged layers, wedges whose region was exhausted unfilled, largest
// exhausted region (cells), constrained Delaunay wraps run (relocation peels
// included), cells they laid, most cells laid by one wrap; the declared
// state and byte budgets, the most bytes one search held, its deepest state
// (cells placed); 1 when a search went over budget (it is inconclusive and
// recovery goes on), and the first such search's states, bytes and depth
// when it stopped; the inconclusive searches, the points removed by
// relocation by peeling, and those of them that had an inconclusive search.
// Available after a refusal too.
PHX_MC_API int32_t phx_mc_plc3d_suppression_search(
    const phx_mc_plc3d_result* result, int64_t* values);
// The route of every suppressed point removed by the exhaustive search or
// by relocation by peeling, one row of PHX_MC_PLC3D_ROUTE_VALUES per wedge
// so filled, in removal order: the point (work id, as refusals name it),
// its witness stratum and carrier entity (as phx_mc_plc3d_source_witnesses),
// the route (PHX_MC_PLC3D_ROUTE_*), and the search's states, bytes and depth
// (the filling search; the point's first inconclusive search before
// peeling; zeros for peeling without one). `count` receives the rows;
// `rows` (null to query the count) needs capacity for all of them.
PHX_MC_API int32_t phx_mc_plc3d_suppression_routes(
    const phx_mc_plc3d_result* result, int64_t capacity, int64_t* count, int64_t* rows);
PHX_MC_API void phx_mc_plc3d_free(phx_mc_plc3d_result* result);
/* The mesh of a successful PLC recovery (phx_mc_plc3d_recover*), adopted
 * from the recovery's native banks: its points, tetrahedra, regions, faces,
 * face orientations, segments and protection radii are the recovery's, so the
 * request's pointers for them must be null and its point, tetrahedron, face
 * and segment counts must equal the recovery's (phx_mc_plc3d_sizes).
 * face_sources, segment_sources and every other field are the request's, as
 * for phx_mc_tet_mesh_create_with_affine_metric; periodic state is refused
 * (PHX_MC_INVALID_ARGUMENT), as is a failed recovery. Unless `validate` is
 * nonzero, what recovery proved is adopted rather than proved again: its
 * cell adjacency, and, without an explicit source complex or witnesses (the
 * mesh is then its own source bank, whose rows cover its constrained faces
 * and segments by construction), the coverage proofs. A nonzero `validate`
 * repeats every check of the array route. The resulting state is the one
 * phx_mc_tet_mesh_create_with_affine_metric builds from the recovery's
 * exported banks. */
PHX_MC_API int32_t phx_mc_tet_mesh_create_from_plc(
    const phx_mc_plc3d_result* recovery, const phx_mc_tet_mesh_create_request* request,
    const double* linear, int32_t validate);

enum phx_mc_intersection_class {
  PHX_MC_DISJOINT = 0,
  PHX_MC_SHARED_VERTEX = 1,
  PHX_MC_SHARED_EDGE = 2,
  PHX_MC_TOUCHING = 3,
  PHX_MC_CROSSING = 4,
  PHX_MC_COPLANAR_OVERLAP = 5,
  PHX_MC_COINCIDENT = 6
};
// Exact planar partition. Coordinates are six signed integers per triangle
// followed by one positive common denominator, all in little-endian 32-bit
// limbs indexed by offsets. Target triangles precede declared triangles.
// Closed boxes must enclose every exact triangle corner; invalid bounds are
// refused. Verdict 0 succeeds, 1 reports area mismatch, 2 incomplete coverage,
// 3 overlapping targets, and 4 singular clipping.
PHX_MC_API int32_t phx_mc_planar_partition(
    int64_t target_count, int64_t declared_count, int64_t word_count,
    const uint32_t* words, const int64_t* offsets, const int8_t* signs,
    const double* lower, const double* upper, int32_t* verdict,
    int64_t* first, int64_t* second);

/* Exact certificates of affine tetrahedral meshes (cells (m, 4) over points
 * (n, 3)), checked on the stored binary64 coordinates; theorems in
 * src/verify_volume.cpp. Every point a cell uses must lie in the exact
 * predicate domain, otherwise the verdict is unresolved (counted in
 * outside_domain_points). Cell and point indices out of range return
 * PHX_MC_INVALID_INPUT. Each first_* names the smallest offending row, or -1. */
enum phx_mc_verdict_status {
  PHX_MC_VERDICT_CERTIFIED = 0,
  PHX_MC_VERDICT_VIOLATED = 1,
  PHX_MC_VERDICT_UNRESOLVED = 2
};

/* Global embedding: every cell positive (orient3d); every face in at most two
 * cells with opposite induced orientations; every edge of the boundary B (the
 * faces of exactly one cell, outward) in exactly two triangles of B with
 * opposite directions; triangles of B meet only in shared vertices or a shared
 * edge; the winding number of B is 0 in front of and 1 behind every shell
 * (edge-connected component) of B. Then the cells tile the solid B bounds
 * without overlap. Violated counts: nonpositive_cells, nonmanifold_faces,
 * boundary_contacts (pairs; cells owning the smallest pair), misnested_shells.
 * Unresolved counts (outside the theorem): outside_domain_points,
 * nonmanifold_boundary_edges (edge uses), unresolved_shells (a shell of
 * several without a vertex on no other shell). Shells are tested only when
 * all cells are positive and B is an embedded edge-manifold. */
typedef struct phx_mc_tetrahedra_verdict {
  int32_t status; /* phx_mc_verdict_status */
  int64_t boundary_faces;
  int64_t boundary_shells;
  int64_t outside_domain_points;
  int64_t first_outside_domain_point;
  int64_t nonpositive_cells;
  int64_t first_nonpositive_cell;
  int64_t nonmanifold_faces;
  int64_t first_nonmanifold_cell;
  int64_t nonmanifold_boundary_edges;
  int64_t first_nonmanifold_boundary_cell;
  int64_t boundary_contacts;
  int64_t first_contact_cell;
  int64_t second_contact_cell;
  int64_t misnested_shells;
  int64_t first_misnested_shell_cell;
  int64_t unresolved_shells;
  int64_t first_unresolved_shell_cell;
} phx_mc_tetrahedra_verdict;
PHX_MC_API int32_t phx_mc_verify_tetrahedra(int64_t point_count, const double* points,
                                            int64_t cell_count, const int32_t* cells,
                                            phx_mc_tetrahedra_verdict* verdict);

/* Region coverage of a declared oriented triangle surface (surface (f, 3),
 * surface_regions (f, 2): the triangle's right-hand normal points from region
 * [0] into region [1], -1 the exterior, the two distinct) by labeled cells
 * (cell_regions (m,), nonnegative): every cell positive; every face in at most
 * two cells with opposite orientations; every boundary face a declared
 * triangle with the cell on the side its region names and the exterior on the
 * other; every face between cells of different regions a declared triangle
 * with the two regions on its sides; no declared triangle between cells of one
 * region; every declared triangle, once, a face of the mesh. certified = 1
 * when every count is 0. With the mesh globally embedded
 * (phx_mc_verify_tetrahedra), the cells of each region then tile exactly the
 * points where that region's declared boundary has winding number one. Null
 * cell_regions means region 0 for every cell; null surface_regions means
 * (0, -1) for every triangle (the outward boundary of one solid). Labels out of
 * range return PHX_MC_INVALID_INPUT. */
typedef struct phx_mc_volume_verdict {
  int32_t certified;
  int64_t outside_domain_points;
  int64_t first_outside_domain_point;
  int64_t nonpositive_cells;
  int64_t first_nonpositive_cell;
  int64_t nonmanifold_faces;
  int64_t first_nonmanifold_cell;
  int64_t boundary_faces;
  int64_t unmatched_boundary_faces;
  int64_t first_unmatched_cell;
  int64_t unmatched_surface_triangles;
  int64_t first_unmatched_surface_triangle;
  int64_t region_mismatches;
  int64_t first_region_mismatch_cell;
} phx_mc_volume_verdict;
PHX_MC_API int32_t phx_mc_verify_volume(int64_t point_count, const double* points,
                                        int64_t cell_count, const int32_t* cells,
                                        const int32_t* cell_regions, int64_t surface_count,
                                        const int32_t* surface, const int32_t* surface_regions,
                                        phx_mc_volume_verdict* verdict);

/* Exact validity of affine tetrahedra: valid[k] = 1 when the Jacobian
 * determinant d = det[v1 - v0, v2 - v0, v3 - v0] and the scale-invariant
 * margin d^2 - floor^2 |v1 - v0|^2 |v2 - v0|^2 |v3 - v0|^2 are both positive
 * (exact signs), else 0; lower[k] <= d <= upper[k] (rigorous binary64
 * enclosure). floor is zero or in [2^-60, 1) (otherwise
 * PHX_MC_INVALID_ARGUMENT); a cell coordinate that is nonzero with magnitude
 * outside [2^-60, 2^60], where the degree-eight margin could leave exact
 * binary64 expansion arithmetic, returns PHX_MC_INVALID_INPUT. */
PHX_MC_API int32_t phx_mc_tetrahedra_validity(int64_t point_count, const double* points,
                                              int64_t cell_count, const int32_t* cells,
                                              double floor, int8_t* valid, double* lower,
                                              double* upper);

/* Tetrahedral mesh of {f <= 0} by restricted Delaunay refinement, surface
 * first (Boissonnat and Oudot; Rineau and Yvinec). f is entry output_entry of
 * an interval program with three inputs; seed_lower and seed_upper are its
 * interval workspace templates (the point executor uses their midpoints).
 * The box [lower, upper] contains the zero set. Seeds: the possible-zero cover
 * of f on the anchored dyadic lattice of the box grown by 1/16 of its largest
 * extent (box_tree.hpp, the cover of the adaptive discovery), down to cells of
 * diagonal at most sqrt(3) criteria[0]; every edge of a kept cell whose vertex values have strictly
 * opposite signs carries a root, and the roots farther than
 * criteria[7] * criteria[0] from earlier ones (with every extra_seeds point,
 * moved onto the zero set by Newton steps) start the Delaunay triangulation.
 * Restricted facets (dual Voronoi edge crossing the zero set) are refined at
 * their surface Delaunay ball centers while their longest edge exceeds
 * criteria[0], the distance from their circumcenter to the ball center
 * exceeds criteria[1] or their smallest angle is below criteria[2] degrees;
 * cells whose circumcenter lies in {f < 0} are refined at their circumcenter
 * while their circumradius exceeds criteria[3] or their radius-edge ratio
 * exceeds criteria[4], unless the circumcenter would destroy a restricted
 * facet, which is refined instead. The restricted surface is then checked
 * against topology[0] edge-connected components and Euler characteristic
 * topology[1] and against the box roots (every root within criteria[0] of a
 * surface vertex); a failure is repaired by inserting the roots of uncovered
 * boxes and the ball centers of facets at singular vertices, or else by
 * halving the edge bound's square, up to four rounds. The boundary vertices
 * are moved onto the zero set by Newton steps. Cells below criteria[5]
 * degrees minimum dihedral angle are removed with the boundary fixed by edge
 * removal, face removal and relocation of interior vertices (every new cell
 * within criteria[4] radius-edge ratio and criteria[6] relative
 * determinant); cells these leave are refined at their circumcenters and
 * the construction repeated, up to three rounds, then any cell still below
 * the targets goes through the fixed-boundary quality improvement. On
 * success *volume owns the points, positive tetrahedra, outward boundary
 * triangles and the report (phx_mc_restricted_volume_sizes). Construction
 * only: the result is certified afterwards. */
#define PHX_MC_RESTRICTED_CRITERIA_SIZE 8
typedef struct phx_mc_restricted_volume phx_mc_restricted_volume;
PHX_MC_API int32_t phx_mc_implicit_restricted_delaunay(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count,
    const double* seed_lower, const double* seed_upper, int64_t workspace_size,
    int64_t output_entry, const double* lower, const double* upper, const double* criteria,
    const int64_t* topology, int64_t extra_seed_count, const double* extra_seeds,
    phx_mc_restricted_volume** volume);
/* sizes receives the point, cell and boundary face counts; report receives
 * PHX_MC_RESTRICTED_REPORT_SIZE entries: refinement insertions, surface
 * repair rounds, whether the restricted surface matched the topology, the box
 * roots no surface vertex covers, the cells the quality improvement left
 * below the targets, and whether the refinement budget ran out. */
#define PHX_MC_RESTRICTED_REPORT_SIZE 6
PHX_MC_API int32_t phx_mc_restricted_volume_sizes(const phx_mc_restricted_volume* volume,
                                                  int64_t* sizes, int64_t* report);
/* points (point count x 3), cells (cell count x 4), faces (face count x 3). */
PHX_MC_API int32_t phx_mc_restricted_volume_copy(const phx_mc_restricted_volume* volume,
                                                 double* points, int32_t* cells, int32_t* faces);
PHX_MC_API void phx_mc_restricted_volume_free(phx_mc_restricted_volume* volume);

/* CAD geometry bundle: every surface, 3D curve and p-curve of a CAD model is a
 * node of one geometry graph; a node's children precede it. Per node kind,
 * the integers and reals (ragged, by offsets) are:
 *   TEMPLATE         [template] / coefficients: a straight-line kind of the
 *                    generated program table, coefficients in seed order
 *   BSPLINE_CURVE    [degree, count, dimension] / knots (count + degree + 1),
 *                    control points (count, dimension), weights (count);
 *                    parameters outside the active knot interval are refused
 *   BSPLINE_SURFACE  [u degree, v degree, u count, v count] / u knots, v knots,
 *                    control points (u count, v count, 3), weights (u count,
 *                    v count); parameters are clamped to the active domain
 *                    with zero derivative across a clamped axis
 *   TSPLINE_SURFACE  [pieces, coefficients, width, rows (pieces, width), -1
 *                    padded] / piece bounds (pieces, 4: u0 u1 v0 v1),
 *                    extraction operators (pieces, width, 16), control points
 *                    (coefficients, 3), weights (coefficients); a point
 *                    takes the first piece containing it
 *   AFFINE           [child, dimension] / L (dimension, dimension), b:
 *                    x -> L child(x) + b
 *   REPARAMETERIZED  [child, reversed] / first, last: t -> child(first +
 *                    s (last - first)), s = 1 - t when reversed, else t
 *   OFFSET_SURFACE   [child] / signed distance d: S + d (Su x Sv) / |Su x Sv|
 *   OFFSET_CURVE     [child, has direction] / distance, direction (3, when
 *                    present): C + d n / |n|, n = (C'1, -C'0) for a planar
 *                    child, else C' x direction
 *   RULED            [first, second]: (1 - v) first(u) + v second(u)
 *   EXTRUSION        [curve] / direction: curve(u) + v direction
 *   REVOLUTION       [curve] / axis origin, axis direction: the profile
 *                    point curve(v) turned by angle u about the axis
 *   SURFACE_MAPPED   [surface, pcurve]: surface(pcurve(t))
 *   ISOPARAMETRIC    [surface, fixed axis] / fixed value
 *   INTERSECTION     [first surface, second surface, charts, output (0 point
 *                    on the first surface, 1 first UV, 2 second UV), fixed
 *                    chart or -1, terminal chart or -1, reversed, chart axes
 *                    (charts)] / carrier sum, terminal threshold, then per
 *                    chart start (4), end (4), lower (4), upper (4),
 *                    preconditioner (3, 3): the root of S1(q01) = S2(q23)
 *                    with one chart coordinate fixed, from the chart's
 *                    interpolation by two preconditioned clipped corrections
 *                    (the source iteration) and Newton polishing to rounding
 *                    level, else the source's 64 corrections */
enum {
  PHX_MC_CAD_NODE_TEMPLATE = 0,
  PHX_MC_CAD_NODE_BSPLINE_CURVE = 1,
  PHX_MC_CAD_NODE_BSPLINE_SURFACE = 2,
  PHX_MC_CAD_NODE_TSPLINE_SURFACE = 3,
  PHX_MC_CAD_NODE_AFFINE = 4,
  PHX_MC_CAD_NODE_REPARAMETERIZED = 5,
  PHX_MC_CAD_NODE_OFFSET_SURFACE = 6,
  PHX_MC_CAD_NODE_OFFSET_CURVE = 7,
  PHX_MC_CAD_NODE_RULED = 8,
  PHX_MC_CAD_NODE_EXTRUSION = 9,
  PHX_MC_CAD_NODE_REVOLUTION = 10,
  PHX_MC_CAD_NODE_SURFACE_MAPPED = 11,
  PHX_MC_CAD_NODE_ISOPARAMETRIC = 12,
  PHX_MC_CAD_NODE_INTERSECTION = 13
};

/* The flat CAD model (all arrays row-major, caller-owned). Surfaces carry
 * their node, orientation, native periods (0 when not periodic) and parameter
 * domain (+-inf when unbounded); curves their 3D node, corners and parameter
 * range; loops their surface and uses; uses (curve traversals or collapsed
 * pole sides) their entity, sense, parameter range and p-curve node;
 * regions their oriented bounding surfaces. */
typedef struct phx_mc_cad_model {
  int64_t node_count;
  const int32_t* node_kinds;           /* (N,) PHX_MC_CAD_NODE_* */
  const int64_t* node_integer_offsets; /* (N + 1,) */
  const int64_t* node_integers;
  const int64_t* node_real_offsets; /* (N + 1,) */
  const double* node_reals;
  int64_t corner_count;
  const double* corner_points; /* (C, 3) */
  int64_t surface_count;
  const int32_t* surface_nodes;           /* (F,) */
  const int8_t* surface_orientations;     /* (F,) +1 or -1 */
  const double* surface_periods;          /* (F, 2) */
  const double* surface_parameter_domains; /* (F, 2, 2): (u lower, u upper), (v ...) */
  int64_t curve_count;
  const int32_t* curve_nodes;   /* (E,) */
  const int32_t* curve_corners; /* (E, 2) start, end */
  const double* curve_ranges;   /* (E, 2) */
  int64_t loop_count;
  const int64_t* loop_use_offsets; /* (L + 1,) */
  const int32_t* loop_surfaces;    /* (L,) */
  int64_t use_count;
  const int8_t* use_kinds;        /* (U,) 0 curve, 1 pole */
  const int32_t* use_entities;    /* (U,) curve or corner */
  const int8_t* use_senses;       /* (U,) +1 or -1 */
  const double* use_ranges;       /* (U, 2) first, last */
  const int32_t* use_pcurve_nodes; /* (U,) */
  int64_t region_count;
  const int64_t* region_surface_offsets; /* (R + 1,) */
  const int32_t* region_surfaces;
  const int8_t* region_orientations;
} phx_mc_cad_model;

enum { PHX_MC_CAD_SURFACE = 0, PHX_MC_CAD_CURVE = 1, PHX_MC_CAD_PCURVE = 2 };
enum {
  PHX_MC_CAD_VALUE = 0,
  PHX_MC_CAD_FIRST_DERIVATIVES = 1,
  PHX_MC_CAD_SECOND_DERIVATIVES = 2,
  PHX_MC_CAD_DIFFERENTIAL = 3
};

/* Binary64 geometry of one surface (by surface row), 3D curve (by curve row)
 * or p-curve (by use row) at `count` parameter points ((count, 2) for a
 * surface, (count,) otherwise). quantity VALUE, FIRST_DERIVATIVES or
 * SECOND_DERIVATIVES writes the value and its derivatives order by order
 * (surface: x, Su, Sv, Suu, Suv, Svv; curve: C, C', C''), as many as the
 * order: 3 / 9 / 18 doubles per surface point, 3 / 6 / 9 per curve point and
 * 2 / 4 / 6 per p-curve point. DIFFERENTIAL writes, per surface point, the
 * first fundamental form E F G, the unit normal Su x Sv / |Su x Sv| and the
 * principal curvatures k1 <= k2 for that normal (8 doubles), and per curve
 * point the speed |C'| and the curvature (2 doubles). A malformed model or
 * row is PHX_MC_INVALID_ARGUMENT / PHX_MC_INVALID_INPUT before any write; a
 * point outside an entity's evaluation domain (B-spline curve knots,
 * T-spline pieces, intersection charts) or a degenerate differential is
 * PHX_MC_INVALID_INPUT with `result` unspecified. Floating-point proposals:
 * nothing here bounds an error. */
PHX_MC_API int32_t phx_mc_cad_geometry_evaluate(const phx_mc_cad_model* model,
                                                int32_t entity_kind, int64_t entity,
                                                int64_t count, const double* parameters,
                                                int32_t quantity, double* result);

/* Sampled curve construction controls; tolerances and sizes are physical.
 * A sampled construction is not a chord-sag certificate. */
typedef struct phx_mc_cad_curve_controls {
  const double* face_sizes;       /* (surface_count,) */
  const double* face_tolerances;  /* (surface_count,) */
  const double* curve_tolerances; /* (curve_count,), infinity means none */
  double tolerance_fraction;
  double grading;
  double minimum_size;
} phx_mc_cad_curve_controls;

typedef struct phx_mc_cad_curve_failure {
  int32_t curve;
  double first, last; /* normalized curve parameters */
} phx_mc_cad_curve_failure;

typedef struct phx_mc_cad_curve_statistics {
  int64_t first_order_evaluations, second_order_evaluations, sag_evaluations;
  int64_t small_feature_resamples, sag_splits, parameter_count, unresolved_count;
} phx_mc_cad_curve_statistics;

/* Test entry for shared curve discretization. All capacities and pointers are
 * validated before any output write. offsets contains curve_count + 1 rows;
 * parameters contains normalized parameters shared by every use; sag_splits
 * has one count per curve. PHX_MC_UNRESOLVED exports the proposal and its
 * unresolved intervals if construction reached a complete proposal; if it
 * could not, outputs are untouched. This stage samples sag; it does not certify it. */
PHX_MC_API int32_t phx_mc_cad_curves_test(
    const phx_mc_cad_model* model, const phx_mc_cad_curve_controls* controls,
    int64_t offset_capacity, int64_t* offsets, int64_t parameter_capacity, double* parameters,
    int64_t failure_capacity, phx_mc_cad_curve_failure* failures,
    int64_t split_capacity, int64_t* sag_splits,
    phx_mc_cad_curve_statistics* statistics);

/* Surface embedding verifier. Findings are deterministic; edge findings carry
 * vertex indices, contact findings triangle indices and their exact intersection
 * classification. Outside-domain and degenerate findings identify the point
 * or triangle, respectively. Insufficient capacity leaves outputs unchanged.
 * More than 2^32 points are refused with PHX_MC_INDEX_RANGE_ERROR. */
enum phx_mc_surface_finding_kind {
  PHX_MC_SURFACE_OUTSIDE_DOMAIN = 1,
  PHX_MC_SURFACE_DEGENERATE = 2,
  PHX_MC_SURFACE_EDGE = 3,
  PHX_MC_SURFACE_VERTEX_LINK = 4,
  PHX_MC_SURFACE_CONTACT = 5,
  PHX_MC_SURFACE_EMPTY = 6
};
typedef struct phx_mc_surface_finding {
  int32_t kind;
  int64_t first;
  int64_t second;
  int32_t classification;
} phx_mc_surface_finding;
typedef struct phx_mc_surface_embedding_verdict {
  int32_t status;
  int64_t finding_count;
} phx_mc_surface_embedding_verdict;
PHX_MC_API int32_t phx_mc_verify_surface_embedding(
    int64_t point_count, const double* points, int64_t triangle_count,
    const int64_t* triangles, int64_t finding_capacity,
    phx_mc_surface_finding* findings, phx_mc_surface_embedding_verdict* verdict);

/* Chart partition verifier. UV triangles and trim chords are already in one
 * lifted chart per patch; the producer refuses non-closing lifts. No periodic
 * identification is performed. Each use belongs to a patch and has a directed
 * nonempty range; each directed chord belongs to one use and numbered loop.
 * Triangle UV is (triangle_count,3,2), chord UV (chord_count,2,2), and ranges
 * are (count,2). The source must establish that its oriented trim cycles bound
 * the patch. Endpoint and range joins compare bitwise, including signed zero.
 * On OK evidence_count gives the failures (zero certifies). Insufficient
 * capacity reports the required count without writing evidence. Consecutive
 * chords of a use must join bitwise in UV and share one loop (USE_COVERAGE).
 * OVERLAP: two correctly oriented triangles of one patch whose UV closures
 * touch other than along a bitwise-shared vertex or edge, including a contact
 * at points equal in value but not bitwise (-0.0 vs 0.0) (row/other: rows). */
enum phx_mc_chart_failure_kind {
  PHX_MC_CHART_ORIENTATION = 1,
  PHX_MC_CHART_EDGE = 2,
  PHX_MC_CHART_USE_COVERAGE = 3,
  PHX_MC_CHART_LOOP = 4,
  PHX_MC_CHART_OVERLAP = 5
};
typedef struct phx_mc_chart_failure {
  int32_t kind;
  int64_t patch;
  int64_t row;
  int64_t other;
} phx_mc_chart_failure;
PHX_MC_API int32_t phx_mc_verify_chart_partition(
    int64_t patch_count, int64_t triangle_count, const int32_t* triangle_patch,
    const double* triangle_uv, int64_t use_count, const int64_t* use_patch,
    const double* use_range, int64_t loop_count, int64_t chord_count,
    const int64_t* chord_use, const int64_t* chord_loop, const double* chord_uv,
    const double* chord_range, int64_t evidence_capacity,
    phx_mc_chart_failure* evidence, int64_t* evidence_count);
/* One-call curve and surface construction; controls have physical units.
   face_sizes are requested edge lengths h. Curves and faces use the internal
   size h_t = h/sqrt(2): the chart size test R <= h_t/sqrt(2) gives every
   edge <= 2R <= h. */
typedef struct phx_mc_cad_surface_controls {
  const double* face_sizes;
  const double* face_tolerances;
  const double* curve_tolerances;
  double tolerance_fraction;
  double grading;
  double minimum_size;
} phx_mc_cad_surface_controls;

typedef struct phx_mc_cad_face_stats {
  int64_t metric_evaluations, insertions, unresolved_size, unresolved_angle;
  int64_t encroachment_refusals, nonstar_refusals, outside_refusals, proximity_refusals;
  int64_t unresolved_triangles, excluded_refusals, fixed_triangles_skipped, omitted_pole_points;
  int64_t periodic_corner_gaps;
  uint64_t max_periodic_corner_gap_ulps;
} phx_mc_cad_face_stats;

/* A gap whose quotient by the period exceeds 2^52 in magnitude is refused
 * as unrepresentable. */
enum {
  PHX_MC_CAD_FACE_SHEET_MISMATCH = 1,
  PHX_MC_CAD_FACE_SHIFT_UNREPRESENTABLE = 2,
  PHX_MC_CAD_FACE_PERIODIC_WRAP = 3
};
enum {
  PHX_MC_CAD_FACE_REFINEMENT = 4
};
/* Refinement refusal kinds match the chart kernel's deterministic records. */
enum {
  PHX_MC_CAD_REFUSAL_NONE = 0,
  PHX_MC_CAD_REFUSAL_ENCROACHMENT = 1,
  PHX_MC_CAD_REFUSAL_OUTSIDE = 2,
  PHX_MC_CAD_REFUSAL_PROXIMITY = 3,
  PHX_MC_CAD_REFUSAL_NONSTAR = 4,
  PHX_MC_CAD_REFUSAL_EXCLUDED = 5
};
typedef struct phx_mc_cad_face_failure {
  int32_t face, loop, use, axis, reason;
  int64_t turns;
  double parameter;
  /* Refinement-only fields: triangle is a pre-flip chart slot, not a global
   * triangle index. Coordinates are in the face's native parameter chart.
   * curve_segment is the zero-based segment in that curve's shared samples;
   * curve and curve_segment are -1 when no curve segment is implicated.
   * segment_parameters are normalized curve parameters, ordered along the use.
   * candidate_uv is NaN when refusal_kind is NONE. */
  int32_t triangle, bad_size, bad_angle, refusal_kind, curve, curve_segment;
  double candidate_uv[2], triangle_uv[6], segment_parameters[2];
} phx_mc_cad_face_failure;
/* A REFINEMENT record with triangle -1, bad_size 1 and refusal NONE is an
 * emitted triangle (after the face stage's flips and moves) whose stored
 * circumradius fails the stored size test (cad_stored_size.hpp); its
 * triangle_uv are the emitted triangle's charts. */

typedef struct phx_mc_cad_surface_result phx_mc_cad_surface_result;
typedef struct phx_mc_cad_surface_arrays {
  int64_t vertex_capacity, triangle_capacity, sliver_capacity, curve_edge_capacity, face_capacity;
  int64_t curve_capacity, parameter_capacity, failure_capacity, face_failure_capacity;
  double* points; int8_t* vertex_dimensions; int32_t* vertex_entities; double* vertex_parameters;
  int32_t* triangles; int32_t* triangle_patches; double* triangle_uv;
  int32_t* triangle_edge_uses; int32_t* facet_regions;
  double* sliver_uv; int32_t* sliver_patches;
  int32_t* curve_edges; int32_t* curve_edge_curves; double* curve_edge_parameters;
  phx_mc_cad_face_stats* face_stats;
  int64_t* curve_offsets; double* curve_parameters; int64_t* curve_sag_splits;
  phx_mc_cad_curve_failure* curve_failures;
  phx_mc_cad_curve_statistics* curve_statistics;
  phx_mc_cad_face_failure* face_failures;
} phx_mc_cad_surface_arrays;

/* OK returns a complete candidate. UNRESOLVED returns a handle with either a
 * complete candidate or only findings (zero vertices and triangles); a curve
 * failure without a complete proposal may return a null handle. Sizes are
 * vertices, triangles, slivers, curve edges, faces, curves, curve parameters,
 * unresolved curve intervals and face failures, respectively. Export validates
 * every capacity and required output before writing anything. */
PHX_MC_API int32_t phx_mc_cad_mesh(const phx_mc_cad_model* model,
                                    const phx_mc_cad_surface_controls* controls,
                                    phx_mc_cad_surface_result** result);
PHX_MC_API int32_t phx_mc_cad_surface_result_sizes(const phx_mc_cad_surface_result* result,
                                                    int64_t sizes[9]);
PHX_MC_API int32_t phx_mc_cad_surface_result_export(const phx_mc_cad_surface_result* result,
                                                     const phx_mc_cad_surface_arrays* arrays);
PHX_MC_API void phx_mc_cad_surface_result_free(phx_mc_cad_surface_result* result);

/* Per-pole rows: face, pole use, protected sectors, free sectors. With
 * capacity zero and rows null, only the required row count is returned. */
PHX_MC_API int32_t phx_mc_cad_surface_pole_sectors(
    const phx_mc_cad_surface_result* result, int64_t capacity,
    int64_t* rows, int64_t* count);

/* Chart rows include collapsed pole slivers; chart_to_mesh is -1 for those
 * rows, otherwise the index in the exported physical triangle array.
 * edge_chord is -1 for interior edges. Chord rows include pole uses. */
typedef struct phx_mc_cad_chart_arrays {
  int64_t triangle_capacity, use_capacity, chord_capacity;
  int64_t* triangles; int32_t* patches; double* uv;
  int64_t* chart_to_mesh; int64_t* edge_chord;
  int64_t* use_patch; double* use_range;
  int64_t* chord_use; int64_t* chord_loop; double* chord_uv; double* chord_range;
} phx_mc_cad_chart_arrays;
PHX_MC_API int32_t phx_mc_cad_chart_sizes(
    const phx_mc_cad_surface_result* result, int64_t sizes[3]);
PHX_MC_API int32_t phx_mc_cad_chart_export(
    const phx_mc_cad_surface_result* result, const phx_mc_cad_chart_arrays* arrays);

enum {
  PHX_MC_FIDELITY_CERTIFIED = 0,
  PHX_MC_FIDELITY_RESOLUTION = 1,
  PHX_MC_FIDELITY_INTERVAL_REFUSAL = 2,
  PHX_MC_FIDELITY_TRIM_REFUSAL = 3,
  PHX_MC_FIDELITY_EDGE_ASSOCIATION = 4
};
enum {
  PHX_MC_FIDELITY_BAD_CHORD_EDGE = 1,
  PHX_MC_FIDELITY_DUPLICATE_CHORD = 2,
  PHX_MC_FIDELITY_MISSING_CHORD = 3,
  PHX_MC_FIDELITY_UNPAIRED_INTERIOR = 4,
  PHX_MC_FIDELITY_CHART_MESH_LINK = 5,
  PHX_MC_FIDELITY_MISSING_MESH_TRIANGLE = 6
};
typedef struct phx_mc_fidelity_finding {
  int32_t kind;
  int64_t triangle, edge, other_triangle, other_edge, chord;
} phx_mc_fidelity_finding;
/* CAD fidelity verifier. UV triangles and trim chords are already in one
 * lifted chart per patch; the producer refuses non-closing lifts. No periodic
 * identification is performed. Each use belongs to a patch and has a directed
 * nonempty range; each directed chord belongs to one use and numbered loop.
 * Triangle UV is (triangle_count,3,2), chord UV (chord_count,2,2), and ranges
 * are (count,2). The source must establish that its oriented trim cycles bound
 * the patch. Endpoint and range joins compare bitwise, including signed zero.
 * edge_chord (triangle_count,3) names the published chord row for each edge,
 * or -1 for an interior edge. Use rows refer to model use rows. Chart
 * triangles include collapsed pole slivers; mesh_triangles omit them and are
 * the input of V2. chart_to_mesh maps each chart triangle to its oriented mesh
 * triangle, or -1 exactly for a sliver with a pole-use chord whose physical
 * vertex IDs coincide. Every mesh triangle must have exactly one chart image.
 * Pole chart corners are binary64 and need not equal the exact pole parameter;
 * the p-curve lens and interval endpoint mismatch enclose that difference.
 * The exact-source coefficient bounds have model->node_reals layout.
 * Certification is conditional on a separately verified chart partition.
 * Sphere poles (a sphere template, its placement, or a circle revolved about
 * an axis through its center, certified from the exact coefficients): rows
 * touching the pole vertex or with a corner in the cap |v| >= 0.6230491932779061
 * (pi/2 - rho*, rho* = sin 2 rho*: where the pole chart's edge sag stops
 * being worse than the uv chart's; see verify_cad_fidelity.cpp) are associated in
 * the regular pole chart (x, y) = rho (cos u, sin u), rho = pi/2 - |v|, with
 * the triangle's linear map taken there; each edge where that region meets a
 * uv row (or a seam or pole chord) carries a seam lens bound charged to both
 * adjacent rows. Faces whose pole cap is not free of other trim chords keep
 * the uv chart.
 *
 * Outputs per triangle are upper distance bounds, status, reason and subdivision
 * counts. Unresolved bounds are +infinity. Curved trim lenses are bounded from
 * the face's p-curve; unsupported interval nodes report
 * PHX_MC_UNRESOLVED. Overall OK means all rows certified, UNRESOLVED means at
 * least one row did not certify.
 * Invalid input leaves outputs unchanged. max_depth is between 0 and 24. */
/* Edge-association findings identify triangle and local edge; -1 means absent.
 * evidence_count is required and reports the number of findings. Insufficient
 * evidence_capacity reports PHX_MC_CAPACITY_EXCEEDED before evidence or row
 * outputs are written. */
PHX_MC_API int32_t phx_mc_verify_cad_fidelity(
    const phx_mc_cad_model* model, int64_t coefficient_count,
    const double* coefficient_lower, const double* coefficient_upper,
    int64_t point_count, const double* points, int64_t triangle_count,
    const int64_t* triangles, const int32_t* face_rows, const double* uv,
    int64_t mesh_triangle_count, const int64_t* mesh_triangles, const int64_t* chart_to_mesh,
    int64_t use_count, const int64_t* use_patch, const double* use_range,
    int64_t loop_count, int64_t chord_count, const int64_t* chord_use,
    const int64_t* chord_loop, const double* chord_uv, const double* chord_range,
    const int64_t* edge_chord, double tolerance, int32_t max_depth,
    int64_t output_capacity, double* upper, int32_t* item_status, int32_t* reasons,
    int32_t* subdivisions, double* maximum_upper, int64_t evidence_capacity,
    phx_mc_fidelity_finding* evidence, int64_t* evidence_count);
/* phx_mc_verify_cad_fidelity preconditions, checked at entry (refused with
 * PHX_MC_INVALID_INPUT, outputs unchanged): loop_count equals the model's;
 * every surface is on a loop; use_patch is the model's loop surface of each
 * use; use_range is the model's use range bitwise, in its sense-directed
 * order (first, last for sense +1) or reversed, one direction for all uses
 * of a face; chord_loop is the model loop of the chord's use; V4's
 * use-coverage, loop-closure and orientation checks hold on these rows
 * (every use covered by chords chaining bitwise from its first to its last
 * range value; every loop one closed bitwise UV cycle; every chart triangle
 * oriented as its face's chord cycles). A sliver (chart_to_mesh -1) needs a
 * mesh triangle with the edge from its pole vertex to its third vertex (else
 * a CHART_MESH_LINK finding). Interior edge pairing is the association
 * check; with it, one direction per face and the model's loop nesting, the
 * chart triangles cover each trimmed face once (V4's overlap enumeration is
 * not repeated).
 * Trusted, not re-established natively: T-spline C2 continuity and partition
 * of unity (Python's ASTS certificate of the exported pieces); continuity of
 * an intersection root across chart seams and the terminal threshold (the
 * chart constructor; V3 treats each break as a C0 line); the source model's
 * 3D edge / face consistency. Rows are written only when the call completes. */

/* P2 curving from stored parameters (S2-G). One quadratic node per tet edge:
 * a curve edge (a curve_edges row) takes C((t_a + t_b) / 2) with the
 * closed-curve unwrap; a surface edge (an edge of a surface triangle) takes
 * S_f(uv_mid) from the lowest such triangle's UV record, a pole corner taking
 * the free p-curve coordinate of the other corner; every other edge takes
 * the straight midpoint. Surface triangles and curve edges index the first
 * surface_point_count points (the face stage's exported vertices, with their
 * dimensions and entities); later points are volume points. Tetrahedra are
 * positively oriented (det[p1-p0, p2-p0, p3-p0] > 0); every surface triangle
 * edge and curve edge must be a tet edge. Edges are the sorted (min, max)
 * vertex pairs; cell_edges (T, 6) lists each cell's edges in the order
 * 01 02 03 12 13 23, the quadratic node order of
 * phx_mc_quadratic_tetrahedron_validity. Every cell is decided by that
 * function at relative_floor; a cell that is not VALID is repaired in cell
 * order: one Jacobi pass of harmonic extension moves the interior nodes of the
 * failing cells and their edge neighbours, then every curved node of a cell
 * still not VALID returns to its chord midpoint, each curved boundary edge
 * so straightened recorded as one (edge, cell) row, until no cell with a
 * curved node fails. cell_decisions are the final decisions. Counts are
 * always written; a capacity below them is PHX_MC_CAPACITY_EXCEEDED before
 * any other output is written. Output is independent of thread count. */
enum {
  PHX_MC_CAD_EDGE_INTERIOR = 0,
  PHX_MC_CAD_EDGE_SURFACE = 1,
  PHX_MC_CAD_EDGE_CURVE = 2
};
typedef struct phx_mc_cad_quadratic_stats {
  int64_t curve_edges, surface_edges, interior_edges;
  int64_t initial_invalid, initial_unresolved;
  int64_t harmonic_nodes, harmonic_repaired_cells;
  int64_t straightened_edges, final_invalid, final_unresolved;
} phx_mc_cad_quadratic_stats;
typedef struct phx_mc_cad_quadratic_request {
  const phx_mc_cad_model* model;
  int64_t point_count;
  const double* points; /* (P, 3) */
  int64_t surface_point_count;
  const int8_t* vertex_dimensions; /* (surface_point_count,) */
  const int32_t* vertex_entities;  /* (surface_point_count,) */
  int64_t tetrahedron_count;
  const int32_t* tetrahedra; /* (T, 4) */
  int64_t triangle_count;
  const int32_t* triangles;        /* (S, 3) */
  const int32_t* triangle_patches; /* (S,) */
  const double* triangle_uv;       /* (S, 3, 2) */
  int64_t curve_edge_count;
  const int32_t* curve_edges;            /* (E, 2) */
  const int32_t* curve_edge_curves;      /* (E,) */
  const double* curve_edge_parameters;   /* (E, 2) native curve parameters */
  double relative_floor;
  int64_t edge_capacity, record_capacity;
  int32_t* edges;         /* (N, 2) */
  double* nodes;          /* (N, 3) */
  int8_t* edge_kinds;     /* (N,) PHX_MC_CAD_EDGE_* */
  int32_t* edge_entities; /* (N,) face, curve, or -1 */
  int64_t* cell_edges;    /* (T, 6) */
  int32_t* cell_decisions; /* (T,) PHX_MC_QUADRATIC_VALIDITY_* */
  int64_t* records;       /* (R, 2) straightened edge, cell */
  int64_t* edge_count;
  int64_t* record_count;
  phx_mc_cad_quadratic_stats* stats;
} phx_mc_cad_quadratic_request;
PHX_MC_API int32_t phx_mc_cad_quadratic_nodes(const phx_mc_cad_quadratic_request* request);

/* CAD solid: the curve and face stages of phx_mc_cad_mesh, then the volume
 * stage (S3 fixed-boundary recovery of the assembled surface, one facet per
 * triangle, curve segments as protected edges; refinement to the request h
 * (h passed unchanged as the circumradius target at every surface point, S3's graded field
 * inside); improvement to the quality targets) and, with geometry_order 2,
 * phx_mc_cad_quadratic_nodes on the final cells. A recovery refusing for a
 * boundary Steiner point (FIXED_SEGMENT/FIXED_FACET/STEINER_SUPPRESSION) has
 * its surface edge (the refused PLC edge, else the facet triangle's longest
 * edge) split on the source in both incident faces (curve parameter or UV
 * midpoint) and is retried once; each split is reported, and the surface
 * export of the same handle shows the split surface. Returns the handle on
 * OK and UNRESOLVED (unmet quality targets, unresolved P2 cells, or a
 * recovery still refused: zero cells, stats name the refusal). A surface
 * triangle whose relative area 2 A / l_max^2 has a square below the
 * determinant floor 2^-22 (zero-length edges included) is refused before
 * recovery: UNRESOLVED, zero cells, recovery_status UNRESOLVED,
 * recovery_reason INVALID_FACET, failure_kind FACET, failure_entity the
 * first such triangle row and failure_measure its relative area. */
typedef struct phx_mc_cad_volume_controls {
  double radius_edge_bound;        /* 2 */
  double min_dihedral_degrees;     /* 15.5 */
  double min_relative_determinant; /* 0; also the P2 relative floor */
  int32_t geometry_order;          /* 1 or 2 */
} phx_mc_cad_volume_controls;
typedef struct phx_mc_cad_volume_stats {
  /* Stage times, nanoseconds: curves, faces, assembly, volume, P2. */
  int64_t curve_ns, face_ns, assembly_ns, volume_ns, quadratic_ns;
  int64_t recovery_attempts, boundary_splits;
  int64_t refine_inserted, refine_unmet;
  int64_t improve_inserted, improve_collapsed, improve_relocated, unmet_cells;
  /* Last recovery: status and phx_mc_plc3d_failure fields. */
  int32_t recovery_status, recovery_reason, failure_kind;
  int64_t failure_entity;
  phx_mc_cad_quadratic_stats quadratic;
  /* The refused triangle's relative area with recovery_reason INVALID_FACET
   * before recovery (above), else 0. */
  double failure_measure;
  /* Unmet cells (of unmet_cells) forced by the input: every record of the
   * cell is a dihedral record and every edge below min_dihedral_degrees is a
   * surface curve edge whose true wedge between its two faces is below it. */
  int64_t unmet_forced_cells;
  /* Unmet cells not forced; the status is UNRESOLVED iff this is nonzero. */
  int64_t unmet_unforced_cells;
} phx_mc_cad_volume_stats;
typedef struct phx_mc_cad_boundary_split {
  int32_t reason, entity_kind; /* the refusal that asked for it (forced: reason 0) */
  int64_t entity;
  int32_t vertices[2];         /* the split surface edge */
  int32_t vertex;              /* the new surface vertex */
  int32_t dimension, carrier;  /* 1: curve row, 2: face row */
  double parameters[2];        /* curve parameter (t, NaN) or face (u, v) */
} phx_mc_cad_boundary_split;
typedef struct phx_mc_cad_volume_arrays {
  int64_t point_capacity, tetrahedron_capacity, triangle_capacity;
  int64_t edge_capacity, record_capacity, split_capacity;
  double* points;               /* (P, 3), surface points first */
  int32_t* tetrahedra;          /* (M, 4), det[b-a, c-a, d-a] > 0 */
  int32_t* tetrahedron_regions; /* (M,) */
  int32_t* boundary_faces;      /* (T, 2): cell and local face (opposite vertex) per surface triangle */
  /* geometry_order 2 (else N = R = 0): as phx_mc_cad_quadratic_nodes. */
  int32_t* edges; double* nodes; int8_t* edge_kinds; int32_t* edge_entities;
  int64_t* cell_edges;          /* (M, 6), null when N = 0 */
  int32_t* cell_decisions;      /* (M,), null when N = 0 */
  int64_t* records;             /* (R, 2) */
  phx_mc_cad_boundary_split* splits;
  phx_mc_cad_volume_stats* stats;
  /* Per model curve (curve_capacity >= the model's curve count; left
   * untouched when the volume stage stopped before quality): the smallest
   * true wedge in degrees between the curve's two faces, written when some
   * unmet cell is not forced, else NaN. A wedge below min_dihedral_degrees
   * makes that target unattainable near the curve for any mesh. */
  int64_t curve_capacity;
  double* curve_rim_wedges;
} phx_mc_cad_volume_arrays;
PHX_MC_API int32_t phx_mc_cad_mesh_solid(const phx_mc_cad_model* model,
                                         const phx_mc_cad_surface_controls* controls,
                                         const phx_mc_cad_volume_controls* volume,
                                         phx_mc_cad_surface_result** result);
/* Sizes: points, cells, surface triangles, P2 edges, P2 records, splits. */
PHX_MC_API int32_t phx_mc_cad_volume_sizes(const phx_mc_cad_surface_result* result,
                                           int64_t sizes[6]);
PHX_MC_API int32_t phx_mc_cad_volume_export(const phx_mc_cad_surface_result* result,
                                            const phx_mc_cad_volume_arrays* arrays);

#ifdef __cplusplus
}
#endif

#endif /* PHYDRAX_MESHCORE_H */
