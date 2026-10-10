/*
 * Copyright © 2026 PHYDRA, Inc. All rights reserved.
 *
 * Plain C ABI of the Phydrax meshing core.  All arrays are C-contiguous.
 * Coordinates are binary64 and must be finite with magnitude either zero or in
 * [2^min_exponent, 2^max_exponent] (see phx_mc_exact_domain); weights must be
 * zero or in [2^(2 min_exponent), 2^(2 max_exponent)].  Within that domain every
 * geometric decision is exact.  Every function is deterministic: ties are
 * ordered by input indices.  Functions returning int32_t report a call status;
 * batched functions additionally report one item status per entry.  No C++
 * exception crosses the ABI: a failed allocation reports
 * PHX_MC_CAPACITY_EXCEEDED and any other escaped failure PHX_MC_INTERNAL_ERROR.
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
  PHX_MC_INVALID_ARGUMENT = 1,       /* null pointer, negative or unaddressable count, bad capacity */
  PHX_MC_NONFINITE_INPUT = 2,        /* NaN or infinity in coordinates/weights */
  PHX_MC_RANGE_ERROR = 3,            /* value outside the exact domain */
  PHX_MC_DEGENERATE_INPUT = 4,       /* zero-measure cell, collinear/coplanar point set */
  PHX_MC_INVALID_INPUT = 5,          /* nonconvex polygon, bad index, zero normal */
  PHX_MC_CAPACITY_EXCEEDED = 6,      /* explicit output/work limit reached, or allocation failed */
  PHX_MC_CONSTRAINT_INTERSECTION = 7,/* constraints cross in their interiors, or an edit would cross or remove a declared constraint */
  PHX_MC_REFINEMENT_LIMIT = 8,       /* valid mesh; quality targets unmet within max_steiner */
  PHX_MC_INTERNAL_ERROR = 9,
  PHX_MC_TIMEOUT = 10              /* native steady-clock deadline reached */
};

/* ------------------------------------------------------------------ identity */

PHX_MC_API const char* phx_mc_version(void);
/* SHA-256 of this public header with CRLF line ends normalized to LF, recorded
 * at configure time: the canonical C ABI contract (every declaration, signature
 * and array extent below). Bindings verify it before binding any other entry
 * point. */
PHX_MC_API const char* phx_mc_abi_contract(void);
/* SHA-256 of the library sources recorded at configure time. */
PHX_MC_API const char* phx_mc_build_hash(void);
/* Actual compiler/platform/FP options and selected build configuration. */
PHX_MC_API const char* phx_mc_build_configuration(void);
PHX_MC_API void phx_mc_exact_domain(int32_t* min_exponent, int32_t* max_exponent);

typedef struct phx_mc_tet_mesh phx_mc_tet_mesh;
#define PHX_MC_EXECUTION_COUNTERS 9
#define PHX_MC_EXECUTION_MEMORY_VALUES 8
/* Thread-local, nestable execution phase. Every zero cap is a hard bound.
 * Handles must be ended in LIFO order on the creating thread. Counters are
 * [construction work, source geometry queries, peak cavity cells,
 * work/query/cavity refusals, externally charged work, externally charged
 * source queries, internal geometric primitive telemetry]; memory has
 * PHX_MC_EXECUTION_MEMORY_VALUES entries: actual native allocator
 * [limit, live, peak, request, allocations, refusals], then separately the
 * conservative external host [live upper, peak upper during this scope].
 * wall_seconds may be +infinity. The scope does not publish partial candidates. */
PHX_MC_API int32_t phx_mc_execution_begin(
    uint64_t work_limit, uint64_t query_limit, uint64_t cavity_limit,
    uint64_t scratch_limit, double wall_seconds, phx_mc_tet_mesh* memory_mesh,
    void** scope);
PHX_MC_API int32_t phx_mc_execution_end(
    void* scope, uint64_t* counters, uint64_t* memory, double* seconds);
/* Admit a separately owned host/device numerical batch on the creating thread.
 * Work/query counts must be charged before the batch, never copied from native
 * owner deltas already charged inside this scope. */
PHX_MC_API int32_t phx_mc_execution_charge(void* scope, uint64_t work, uint64_t queries);
/* Import an actual ended predecessor receipt atomically. Work/source queries
 * debit the active scope and all parents once. Its measured prior duration
 * shortens every native steady-clock deadline; raw scope elapsed is unchanged.
 * Invalid/nonfinite/negative duration is refused without a debit. */
PHX_MC_API int32_t phx_mc_execution_import_preparation(
    void* scope, uint64_t work, uint64_t queries, double prior_seconds);
/* Observe accepted prior-duration debits separately from actual scope elapsed. */
PHX_MC_API int32_t phx_mc_execution_prior_seconds(void* scope, double* seconds);
/* Dry preflight for a static numerical work bound. Does not claim the bound
 * as consumed work. A successful pure batch must subsequently charge its
 * actual measured work before any candidate from the batch can be committed. */
PHX_MC_API int32_t phx_mc_execution_admit_work_bound(void* scope, uint64_t maximum_work);
/* Bind the actual touched-cell count of one externally owned local edit to
 * the same nested cavity allowance, deadline, peak and refusal ledger. */
PHX_MC_API int32_t phx_mc_execution_admit_cavity(void* scope, uint64_t cells);
/* Observe [remaining construction work, remaining source queries, local
 * cavity limit, actual available managed bytes] and native wall allowance.
 * Minimum across all parent scopes; no allowance is renewed or consumed. */
PHX_MC_API int32_t phx_mc_execution_remaining(
    void* scope, uint64_t* remaining, double* wall_seconds);
/* Allocate an uninitialized host array through the active creating-thread
 * scope's actual managed resource. Both the owned handle's sizeof/alignment
 * request and the exact payload byte/alignment request are counted. bytes must
 * fit ptrdiff_t; alignment must be a nonzero size_t power of two. Zero bytes
 * still issues a zero-byte upstream request and returns valid aligned data.
 * Successful requests alone increase live/peak/allocations; a later payload
 * refusal does not erase the successful handle request. Invalid input changes
 * no allocator counters. Failure clears both outputs and preserves the scope's
 * original timeout/capacity status. The returned owner retains its resource
 * independently of lexical scope lifetime; data remains valid until release.
 * This does not account for arbitrary NumPy/JAX/device/compiler allocations. */
PHX_MC_API int32_t phx_mc_execution_allocate_host_array(
    void* execution, uint64_t bytes, uint64_t alignment, void** out_owner, void** out_data);
/* Release one successful owner exactly once (null is harmless), on any thread.
 * Release is synchronized with native pool operations, requires no live scope,
 * and frees both payload and handle. Any views must retain the owner until
 * their final reference disappears; a released/stale owner must not be reused. */
PHX_MC_API void phx_mc_execution_free_host_array(void* owner);
/* Reserve a conservative bound for actual live host objects that cannot use
 * managed raw array storage (e.g. Python fractions and topology containers).
 * This reduces the SAME active pool allowance without allocating a dummy
 * payload. Native memory evidence remains measured native allocations only;
 * callers must report their conservative host upper bound separately.
 * reservation is IN/OUT: initialize *reservation to null to create a token;
 * a live token belonging to the active pool is atomically resized to the
 * requested TOTAL bytes_upper. Refusal preserves its previous bound/token.
 * Shrinking an existing token is allowed during resource/deadline refusal.
 * Token metadata itself is a real managed allocation. The token retains its
 * pool until release, so its bound may outlive the originating scope. */
PHX_MC_API int32_t phx_mc_execution_reserve_host_storage(
    void* execution, uint64_t bytes_upper, void** reservation);
/* Release exactly once, on any thread, without requiring a live scope.
 * Null is harmless. A released/stale reservation must not be reused. */
PHX_MC_API void phx_mc_execution_release_host_storage(void* reservation);

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
    int64_t count, const double* first, const double* second, int32_t vertex_capacity,
    double* volumes, double* first_moments, int32_t* item_status);

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
    int64_t count, const double* first, const double* second, int32_t vertex_capacity,
    int32_t simplex_capacity, double* simplices, int32_t* simplex_counts, double* volumes,
    double* first_moments, int32_t* item_status);

/* Convex polyhedra given as halfspaces {x : n . x <= h}: normals
 * (count, plane_capacity, 3), offsets (count, plane_capacity), plane_counts
 * (count,), intersected with tetrahedra (count, 4, 3). */
PHX_MC_API int32_t phx_mc_polyhedron_clip_moments(
    int64_t count, int32_t plane_capacity, const double* normals, const double* offsets,
    const int32_t* plane_counts, const double* tetrahedra, int32_t vertex_capacity,
    double* volumes, double* first_moments, int32_t* item_status);

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

/* Canonical reference triangle clipped by three exact rational halfplanes.
 * Six rows (nx, ny, h) of signed integers encode nx*x + ny*y <= h;
 * the first three must be positive multiples of (-1,0,0), (0,-1,0),
 * (1,1,1). Eighteen signed little-endian uint32 magnitudes use offsets[19].
 * Fractions are cleared by positive row denominators before this API.
 * Outputs authoritative original supporting-plane pairs (V,2), outgoing
 * edge plane labels (V), and vertex count; numerical vertices are NOT output.
 * scratch_limit bounds a conservative arbitrary-integer resident footprint
 * before allocation; work_limit bounds exact arithmetic action count. */
PHX_MC_API int32_t phx_mc_clip_reference_triangle_exact(
    const uint32_t* words, int64_t word_count, const int64_t* offsets,
    const int8_t* signs, int64_t scratch_limit, int64_t work_limit,
    int32_t vertex_capacity, int32_t* supporting_planes, int32_t* edge_labels,
    int32_t* vertex_count, int64_t* work_units);

/* Exact original reference tetrahedron clipped by six rational box pullbacks.
 * Forty denominator-cleared signed integers describe ten (a,b,c,d) planes
 * with a*x+b*y+c*z+d<=0. The first four are canonical simplex planes.
 * Outputs source support triplets (V,3), complete incident plane masks (V),
 * oriented face CSR/plane labels, actual work; no numerical vertices.
 * Capacities V>=16, F>=10, face entries>=48. Counts commit only on success. */
PHX_MC_API int32_t phx_mc_clip_reference_tetrahedron_exact(
    const uint32_t* words, int64_t word_count, const int64_t* offsets,
    const int8_t* signs, int64_t scratch_limit, int64_t work_limit,
    int32_t vertex_capacity, int32_t face_capacity, int32_t entry_capacity,
    int32_t* supporting_planes, uint16_t* incident_planes,
    int32_t* face_offsets, int32_t* face_planes, int32_t* face_vertices,
    int32_t* vertex_count, int32_t* face_count, int64_t* work_units);

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

/* ------------------------------------------------------------- contacts in 3D
 * Exact contact classification of triangle pairs (count, 3, 3) and
 * segment/triangle pairs ((count, 2, 3), (count, 3, 3)).  Classes:
 *   DISJOINT         empty intersection;
 *   SHARED_VERTEX    exactly one common vertex;
 *   SHARED_EDGE      exactly one common edge (for a segment: the segment is an
 *                    edge of the triangle);
 *   TOUCHING         any other contact whose relative interiors are disjoint
 *                    (T-junctions, partial edge overlap, vertex or edge on a face);
 *   CROSSING         transversal contact through both relative interiors;
 *   COPLANAR_OVERLAP coplanar contact of positive length (segment) or area;
 *   COINCIDENT       identical triangles.
 * DISJOINT, SHARED_VERTEX and SHARED_EDGE are legal adjacency of a conforming
 * complex.  Optional vertex ids (count, 3) / (count, 2), both or neither,
 * make adjacency identity-based: a vertex-vertex contact is shared only when
 * the ids agree (coincident positions with distinct ids are TOUCHING); ids
 * repeated within one input or equal across inputs at distinct positions
 * report PHX_MC_INVALID_INPUT.  Failed items report class -1.
 *
 * Optional constructions (all four pointers or none): point_counts (count,);
 * points (count, P, 3) with P = 6 for triangle pairs and 2 for segments: the
 * point, the segment endpoints ordered along the intersection line, or the
 * convex polygon counterclockwise about the first triangle's normal;
 * point_bounds (count, P): max-norm distance bound of each constructed point
 * from the exact one (zero for input vertices); point_features (count, P, 2):
 * the feature of each input containing the point, 0..2 vertex k, 3..5 the edge
 * opposite vertex k - 3, 6 the relative interior (segments: 0, 1 endpoints,
 * 6 interior).  Unused slots hold zero coordinates and features -1.  Every
 * classification and feature is exact; coordinates are bounded constructions
 * x = p + t (q - p) with t a ratio of exact orientation expansions.
 * Item status: OK, NONFINITE_INPUT, RANGE_ERROR, DEGENERATE_INPUT (collinear
 * triangle, zero-length segment), INVALID_INPUT (ids). */
enum phx_mc_intersection_class {
  PHX_MC_DISJOINT = 0,
  PHX_MC_SHARED_VERTEX = 1,
  PHX_MC_SHARED_EDGE = 2,
  PHX_MC_TOUCHING = 3,
  PHX_MC_CROSSING = 4,
  PHX_MC_COPLANAR_OVERLAP = 5,
  PHX_MC_COINCIDENT = 6
};

PHX_MC_API int32_t phx_mc_triangle_intersections(
    int64_t count, const double* first, const double* second, const int64_t* first_ids,
    const int64_t* second_ids, int8_t* classes, int32_t* point_counts, double* points,
    double* point_bounds, int8_t* point_features, int32_t* item_status);
PHX_MC_API int32_t phx_mc_segment_triangle_intersections(
    int64_t count, const double* segments, const double* triangles, const int64_t* segment_ids,
    const int64_t* triangle_ids, int8_t* classes, int32_t* point_counts, double* points,
    double* point_bounds, int8_t* point_features, int32_t* item_status);

/* Points (count, 3) against triangles (count, 3, 3): sides (count,) receives
 * orient3d(t0, t1, t2, x) and features (count,) the triangle feature containing
 * the point (codes above) or -1 when it is off the closed triangle. */
PHX_MC_API int32_t phx_mc_point_triangle_locations(int64_t count, const double* points,
                                                   const double* triangles, int8_t* sides,
                                                   int8_t* features, int32_t* item_status);

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

/* Nonnegative limits are hard bounds, including zero. Optional work[9] holds
 * actual work, orient/incircle/power requests, insertions, flips, peak cavity,
 * work refusals and cavity refusals. Optional memory[6] holds actual limit,
 * live/peak/requested bytes, allocation count and refusal count. */
PHX_MC_API int32_t phx_mc_delaunay_2d(
    int64_t point_count, const double* points, int64_t max_triangles,
    int64_t max_cavity_cells, int64_t max_work, int64_t max_scratch_bytes,
    uint64_t* work_evidence, uint64_t* memory_evidence, phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_regular_2d(
    int64_t point_count, const double* points, const double* weights,
    int64_t max_triangles, int64_t max_cavity_cells, int64_t max_work,
    int64_t max_scratch_bytes, uint64_t* work_evidence,
    uint64_t* memory_evidence, phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_delaunay_3d(int64_t point_count, const double* points,
                                      int64_t max_tetrahedra, phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_regular_3d(int64_t point_count, const double* points,
                                     const double* weights, int64_t max_tetrahedra,
                                     phx_mc_mesh** mesh);

/* Constrained Delaunay triangulation of points (n, 2) and segments (m, 2)
 * (input point indices), refined by Ruppert/Chew insertion.
 * keep_convex_hull != 0 meshes the convex hull (hull edges act as boundary
 * constraints); otherwise triangles reachable from the hull or from hole seeds
 * holes (h, 2) without crossing a segment are removed.  min_angle_degrees in
 * [0, 60) (0 disables angle refinement), max_area > 0 (+inf disables), and
 * max_steiner >= 0 bound the refinement.  Steiner points are appended after the
 * input points.  cell_constraints (cell_count, 3) holds, for the edge opposite
 * each cell vertex, the input segment index it lies on or -1.  Returns
 * PHX_MC_REFINEMENT_LIMIT with a valid conforming mesh when the quality targets
 * are unmet after max_steiner insertions, and PHX_MC_CONSTRAINT_INTERSECTION when
 * two segments cross in their interiors. */
PHX_MC_API int32_t phx_mc_constrained_delaunay_2d(
    int64_t point_count, const double* points, int64_t segment_count, const int32_t* segments,
    int64_t hole_count, const double* holes, int32_t keep_convex_hull, double min_angle_degrees,
    double max_area, int64_t max_steiner, int64_t max_triangles,
    int64_t max_cavity_cells, int64_t max_work, int64_t max_scratch_bytes,
    uint64_t* work_evidence, uint64_t* memory_evidence, phx_mc_mesh** mesh);

PHX_MC_API int32_t phx_mc_mesh_dimension(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_point_count(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_cell_count(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_input_point_count(const phx_mc_mesh* mesh);
PHX_MC_API void phx_mc_mesh_copy_points(const phx_mc_mesh* mesh, double* points);
PHX_MC_API void phx_mc_mesh_copy_cells(const phx_mc_mesh* mesh, int32_t* cells);
PHX_MC_API void phx_mc_mesh_copy_vertex_map(const phx_mc_mesh* mesh, int32_t* vertex_map);
/* (cell_count, dimension + 1): the constraint reference of the edge (2D) or
 * facet (3D) opposite each cell vertex; filled with -1 without constraints. */
PHX_MC_API void phx_mc_mesh_copy_cell_constraints(const phx_mc_mesh* mesh,
                                                  int32_t* cell_constraints);
/* (cell_count,): region labels; filled with -1 without labels. */
PHX_MC_API void phx_mc_mesh_copy_cell_regions(const phx_mc_mesh* mesh, int32_t* cell_regions);
PHX_MC_API void phx_mc_mesh_free(phx_mc_mesh* mesh);

/* ------------------------------------------- prepared incremental 3D Delaunay
 * An owned, mutable Delaunay tetrahedralization for incremental construction.
 * create() triangulates the initial points exactly as phx_mc_delaunay_3d
 * (DEGENERATE_INPUT when they do not span space).  Every submitted point then
 * receives the next vertex id (initial points keep their indices); the vertex
 * map sends an id to itself (a vertex), to the id of an earlier identical
 * point, or to -1 (insertion refused).  The triangulation depends only on the
 * vertex set and ids, not on insertion order.  Limits: max_vertices ids
 * (< 2^31 - 1), max_tetrahedra finite cells, max_cavity tetrahedra removed by
 * one insertion.  Refusals are decided before any change, so a refused
 * insertion leaves the triangulation unchanged.
 *
 * insert(): points (count, 3) are validated as a batch (NONFINITE_INPUT,
 * RANGE_ERROR, or CAPACITY_EXCEEDED past max_vertices refuse the whole call);
 * work_limit >= 0 bounds the walk, conflict and creation work of the call.
 * vertices (count,) receives the vertex map of each point and item_status
 * (count,) OK, CAPACITY_EXCEEDED (work, cavity, cell or slot limit; once the
 * work limit is exhausted the remaining points are refused) or
 * CONSTRAINT_INTERSECTION (the conflict region bounded by constrained facets
 * is not star-shaped from the point, e.g. a point on a constrained facet).
 *
 * constrain_facets(): facets (count, 3) of live vertex ids and references
 * constraint_ids (count,) >= 0; each existing facet is marked on both sides
 * (idempotent for an equal reference).  Item status INVALID_INPUT for a facet
 * absent from the triangulation, dead or repeated ids, or a different
 * existing reference.  Later insertions never cross constrained facets.
 *
 * label_regions(): clears every region, then labels (>= 0) the finite cells
 * reachable from the cell strictly containing each seed (count, 3) without
 * crossing a constrained facet.  Item status INVALID_INPUT for a seed that is
 * outside, on a facet/edge/vertex, has a negative label, or reaches a region
 * labeled differently.  New cells inherit the region of the cell they replace.
 *
 * locate(): cells (count, 4) receives the canonical vertex tuple (even
 * permutation starting at its smallest entry; -1 is the ghost vertex) of a
 * tetrahedron containing each point and locations (count,) the number of its
 * facet planes through the point (0 interior, 1 facet, 2 edge, 3 vertex) or -1
 * outside the convex hull.
 *
 * finalize(): an owned canonical mesh snapshot (points and vertex map of every
 * id, cells, cell constraints and regions); the handle stays usable.
 * statistics(): values (PHX_MC_TRIANGULATION_3D_STATISTICS,) indexed below.
 * A handle whose internal consistency premise failed reports
 * PHX_MC_INTERNAL_ERROR from every later call. */
typedef struct phx_mc_triangulation_3d phx_mc_triangulation_3d;

#define PHX_MC_TRIANGULATION_3D_VERTEX_IDS 0
#define PHX_MC_TRIANGULATION_3D_LIVE_VERTICES 1
#define PHX_MC_TRIANGULATION_3D_FINITE_CELLS 2
#define PHX_MC_TRIANGULATION_3D_GHOST_CELLS 3
#define PHX_MC_TRIANGULATION_3D_CELL_SLOTS 4
#define PHX_MC_TRIANGULATION_3D_FREE_SLOTS 5
#define PHX_MC_TRIANGULATION_3D_CONSTRAINED_FACETS 6
#define PHX_MC_TRIANGULATION_3D_WORK 7
#define PHX_MC_TRIANGULATION_3D_COMMITTED 8
#define PHX_MC_TRIANGULATION_3D_REFUSED 9
#define PHX_MC_TRIANGULATION_3D_LARGEST_CAVITY 10
#define PHX_MC_TRIANGULATION_3D_RETAINED_BYTES 11
#define PHX_MC_TRIANGULATION_3D_PEAK_BYTES 12
#define PHX_MC_TRIANGULATION_3D_STATISTICS 13

PHX_MC_API int32_t phx_mc_triangulation_3d_create(int64_t point_count, const double* points,
                                                  int64_t max_vertices, int64_t max_tetrahedra,
                                                  int64_t max_cavity,
                                                  phx_mc_triangulation_3d** triangulation);
PHX_MC_API int32_t phx_mc_triangulation_3d_insert(phx_mc_triangulation_3d* triangulation,
                                                  int64_t count, const double* points,
                                                  int64_t work_limit, int32_t* vertices,
                                                  int32_t* item_status);
PHX_MC_API int32_t phx_mc_triangulation_3d_constrain_facets(
    phx_mc_triangulation_3d* triangulation, int64_t count, const int32_t* facets,
    const int32_t* constraint_ids, int32_t* item_status);
PHX_MC_API int32_t phx_mc_triangulation_3d_label_regions(phx_mc_triangulation_3d* triangulation,
                                                         int64_t seed_count, const double* seeds,
                                                         const int32_t* labels,
                                                         int32_t* item_status);
PHX_MC_API int32_t phx_mc_triangulation_3d_locate(phx_mc_triangulation_3d* triangulation,
                                                  int64_t count, const double* points,
                                                  int32_t* cells, int8_t* locations,
                                                  int32_t* item_status);
PHX_MC_API int32_t phx_mc_triangulation_3d_statistics(
    const phx_mc_triangulation_3d* triangulation, int64_t* values);
PHX_MC_API int32_t phx_mc_triangulation_3d_finalize(const phx_mc_triangulation_3d* triangulation,
                                                    phx_mc_mesh** mesh);
PHX_MC_API void phx_mc_triangulation_3d_free(phx_mc_triangulation_3d* triangulation);

/* ------------------------------------------------------------ graph partition
 * Deterministic multilevel k-way partition of a symmetric weighted graph in
 * canonical CSR form: offsets (vertex_count + 1,) int64 starting at 0,
 * neighbors (offsets[vertex_count],) int32 strictly increasing per row (no self
 * loops), edge_weights aligned with neighbors and equal for both directions of
 * an edge, vertex_weights (vertex_count,).  All weights are nonnegative, each
 * total is at most 2^53 and the vertex total is positive.  part_targets (k,)
 * sum to the vertex total; part_capacities (k,) are at least the targets and
 * bound every refinement move on the input graph (coarse levels relax them by
 * their heaviest merged vertex).  require_nonempty != 0 requires k <= vertex_count
 * and gives every part at least one vertex. refinement_passes bounds the FM
 * passes per level; work_limit bounds adjacency visits plus candidate evaluations,
 * reporting PHX_MC_CAPACITY_EXCEEDED with the counters reached when exhausted.
 * Outputs parts (vertex_count,) in [0, k) and counters (PHX_MC_GRAPH_PARTITION_COUNTERS,):
 * coarsening levels, coarsest vertex count, matched pairs, bisection trials,
 * refinement passes, committed moves, rolled-back moves, overload-shedding moves,
 * empty-part repairs, adjacency visits, candidate evaluations. A malformed graph or part table reports
 * PHX_MC_INVALID_INPUT.  Parts above capacity after refinement are not an error;
 * the caller measures them.  Decisions use integer gains and IEEE binary64
 * target scaling only, so the result depends on the inputs alone. */
#define PHX_MC_GRAPH_PARTITION_COUNTERS 11
#define PHX_MC_GRAPH_PARTITION_VISITS 9
PHX_MC_API int32_t phx_mc_graph_partition(
    int64_t vertex_count, const int64_t* offsets, const int32_t* neighbors,
    const int64_t* edge_weights, const int64_t* vertex_weights, int32_t part_count,
    const int64_t* part_targets, const int64_t* part_capacities, int32_t require_nonempty,
    int32_t refinement_passes, int64_t work_limit, int32_t* parts, int64_t* counters);

/* ------------------------------------------------ surface reconnection
 * One chart-embedded surface patch: chart coordinates charts (n, 2) in the
 * exact domain, physical coordinates points (n, 3), oriented source normals
 * normals (n, 3) (zero where the source supplies none, e.g. at a pole),
 * triangles (m, 3) strictly counterclockwise in the chart, and constrained
 * (m, 3) flags (nonzero) of the edge opposite each triangle vertex.  A
 * triangle is surface-consistent when its physical normal is nonzero and has
 * a positive component along every nonzero normal of its vertices.  Vertices
 * may share physical coordinates (a chart side collapsing to a pole).  Every
 * chart edge with one incident triangle must be constrained and a shared edge
 * carries
 * the same flag on both sides; otherwise, or for a non-counterclockwise,
 * repeated-vertex, out-of-range or non-manifold triangle, the call reports
 * PHX_MC_INVALID_INPUT.
 *
 * Points insert_charts (k, 2) with physical positions insert_points (k, 3)
 * and normals insert_normals (k, 3) are inserted in order.  Each is located
 * by an exact chart walk from
 * insert_hints[i] (a triangle row, or -1 for the last created triangle) and
 * splits the containing triangle, or the unconstrained edge through it.
 * Optional insert_edges (k, 2) contains local input vertex-row endpoints of
 * the producer-selected existing edge; null or (-1, -1) uses ordinary point
 * location. Other negative, equal, or out-of-input-range endpoints are invalid.
 * An explicit edge must currently be unconstrained and reciprocal; missing,
 * stale or invalid requests refuse unchanged, never fall back to location.
 * With sweep == 0, a noncollinear rounded chart point is the execution view
 * of a producer-owned exact rational restriction: it must remain within the
 * closed componentwise endpoint bounds and differ from both endpoints.  The
 * complete paired cavity is split without Lawson flips.  The native physical
 * guards still apply; only a declared radial pole complex may defer them to
 * the producer, which must validate the exact chart and continuous-source
 * physical children before publication or another reconnect.  Otherwise all
 * four children require exact positive chart orientation.
 * Item status: OK; INVALID_INPUT outside the triangulated chart domain;
 * CONSTRAINT_INTERSECTION on a constrained edge; DEGENERATE_INPUT on an
 * existing chart vertex, within physical distance insert_spacing[i] of a
 * vertex of the containing triangle or its neighbors, when a split triangle
 * would be degenerate, or when a split surface-consistent triangle would not
 * yield surface-consistent triangles on its side (an inconsistent, e.g.
 * coarse or folded, triangle may be split into any nondegenerate ones);
 * NONFINITE_INPUT/RANGE_ERROR for invalid values; CAPACITY_EXCEEDED past
 * max_triangles or the work limit (every later point is then refused).
 * Accepted points receive vertex ids n, n + 1, ... in order (vertex_ids (k,),
 * -1 when refused); refusals are decided before any change.
 *
 * After each insertion, and over every edge when sweep != 0, Lawson flips
 * restore the physical Delaunay criterion: the unconstrained edge (a, b)
 * opposite c and d is flipped when the physical angles at c and d sum beyond
 * pi, both flipped triangles are strictly counterclockwise in the chart
 * (exact orient2d), no two of a, b, c, d share physical coordinates, the new
 * triangles are nondegenerate, and, when either old triangle is
 * surface-consistent, both new ones are surface-consistent and on the side of
 * every consistent old one.  work_limit bounds walk
 * steps and flip tests; an exhausted budget stops further work and leaves a
 * valid chart triangulation.  Outputs: triangles_out and constrained_out
 * (max_triangles, 3), the first triangle_count_out rows written, and counters
 * (PHX_MC_SURFACE_COUNTERS,): inserted, refused, flips, walk steps, work,
 * exhausted (0 or 1). */
#define PHX_MC_SURFACE_COUNTERS 6
PHX_MC_API int32_t phx_mc_surface_reconnect(
    int64_t vertex_count, const double* charts, const double* points, const double* normals,
    const double* vertex_metrics, const int64_t* vertex_pole_ids,
    int64_t triangle_count, const int32_t* triangles, const int8_t* constrained,
    int64_t insert_count, const double* insert_charts, const double* insert_points,
    const double* insert_normals, const double* insert_metrics, const int64_t* insert_pole_ids,
    const double* insert_spacing, const int32_t* insert_hints, const int32_t* insert_edges,
    int32_t sweep, int64_t max_triangles, int64_t work_limit,
    int32_t* triangles_out, int8_t* constrained_out, int64_t* triangle_count_out,
    int32_t* vertex_ids, int32_t* item_status, int64_t* counters);

/* ----------------------------------------------------------- periodic Delaunay
 * Delaunay triangulation (dimension 2) or tetrahedralization (dimension 3) of
 * a periodic point set on the flat torus R^d / L.  points (n, d) are
 * representatives, fractional (n, d) their lattice coordinates (fractional =
 * x @ inverse, |fractional| <= 2^30, else PHX_MC_RANGE_ERROR), lattice (d, d)
 * holds the lattice vectors as rows and inverse (d, d) the Cartesian-to-
 * fractional map; images are x_r + s L.  Every predicate is
 * exact on the translated positions x_r + s L and cospherical ties use a
 * lattice-translation-invariant symbolic perturbation (exact lexicographic
 * order of positions), so the triangulation is unique.
 *
 * Rounds triangulate every image whose fractional coordinates lie in
 * [-m, 1 + m]^d, starting at m = initial_margin and doubling.  A round succeeds
 * when every extracted cell's circumball lies inside that box (conservative
 * slack on the constructed circumcenter) and the extracted cells are closed
 * under adjacency; it is then the complete periodic triangulation.  A round
 * needing more than max_images images, or more than max_cells finite cell
 * slots, reports PHX_MC_CAPACITY_EXCEEDED; two points in one lattice orbit
 * report PHX_MC_INVALID_INPUT.
 *
 * evidence (PHX_MC_PERIODIC_EVIDENCE,) is written on every return after the
 * pointers are accepted: rounds, final margin, images of that round,
 * uncertified or unclosed extracted cells, largest required margin of the
 * extracted cells, finite cell slots, exact predicate evaluations, perturbed
 * decisions, exhausted limit (0 none, 1 images, 2 cells) and the duplicate
 * representative (-1 none).  On PHX_MC_OK the handle holds one cell per orbit:
 * vertices (T, d + 1) representatives, positively oriented in their lift, and
 * image shifts (T, d + 1, d) of the translate whose anchor (smallest
 * representative, then smallest shift) is the anchor's base image, the one
 * with shift -floor(fractional); cells are sorted by their canonical key of
 * anchor-relative shifts. */
#define PHX_MC_PERIODIC_EVIDENCE 10
typedef struct phx_mc_periodic_triangulation phx_mc_periodic_triangulation;
PHX_MC_API int32_t phx_mc_periodic_delaunay(int32_t dimension, int64_t point_count,
                                            const double* points, const double* fractional,
                                            const double* lattice, const double* inverse,
                                            double initial_margin, int64_t max_images,
                                            int64_t max_cells, double* evidence,
                                            phx_mc_periodic_triangulation** triangulation);
PHX_MC_API int64_t phx_mc_periodic_cell_count(const phx_mc_periodic_triangulation* triangulation);
PHX_MC_API void phx_mc_periodic_copy_cells(const phx_mc_periodic_triangulation* triangulation,
                                           int32_t* vertices, int32_t* shifts);
PHX_MC_API void phx_mc_periodic_free(phx_mc_periodic_triangulation* triangulation);

/* ------------------------------------------------ constrained tetrahedral mesh
 * An owned tetrahedral mesh of a constrained piecewise-linear domain for
 * Delaunay refinement and quality improvement.  create() takes points
 * (point_count, 3), positively oriented domain tetrahedra (tet_count, 4)
 * with regions (>= 0), constrained faces (face_count, 3) with native source
 * group rows (>= 0) that include every domain-boundary face and region interface,
 * protected subsegments (segment_count, 2) with native source group rows (>= 0), optional
 * per-vertex protecting-ball radii (NULL: none, 0: unprotected) and the
 * boundary policy. An optional declared source complex retains an independent
 * immutable bank (source_point_count, 3); its rows never index carrier slots.
 * NULL source_points with count zero makes the initial faces/segments their
 * own exact source. Source triangles (source_face_count, 3) and segments
 * (source_segment_count, 2) carry the same native source group rows and
 * nonnegative deviation bounds (NULL: 0). These int32 values are row selectors,
 * not scientific IDs: the binding retains explicit immutable int64 scientific
 * group-ID sidecars and maps every public source label through them. No
 * scientific ID is narrowed to int32. The declared complex also names
 * per-point witnesses (NULL: none): stratum (PHX_MC_SOURCE_STRATUM_*), source
 * row and parameters (2,) -- a segment parameter t from its first endpoint,
 * or the barycentric weights of a triangle's second and third corners.
 * Witness deviations are recertified natively and must not exceed the row
 * bound; every initial face/segment must lie on a row of its own source.
 * A face edge that is not a segment must be shared by exactly two faces of
 * one source, coplanar unless a vertex carries a bounded witness.  Any
 * violated premise (bad index, inverted cell, nonmanifold facet,
 * unconstrained boundary/interface, absent face/segment, unsupported
 * witness) is INVALID_INPUT.  Vertex dimension is derived: 0 corner, 1
 * segment, 2 facet, 3 interior, -1 unused/removed.
 *
 * Every change is one validated cavity transaction; refusals leave the mesh
 * unchanged.  A constrained face or segment is split through a point lying
 * exactly on it (exact orient3d/collinearity), or, when none is
 * representable, by bisecting the constrained edge at the correctly rounded
 * carrier of an exact witness on its source row whose certified deviation
 * the row's declared bound admits.  Children inherit source, region and
 * orientation; every vertex's certified deviation (zero exactly on the
 * source) bounds its faces and segments from their source rows.  FIXED
 * policy never splits constrained faces or segments.
 *
 * refine(): constrained Delaunay refinement (Shewchuk order: encroached
 * subsegments at midpoints, encroached subfacets at circumcenters, then
 * tetrahedra with radius-edge ratio > radius_edge_bound or circumradius >
 * the mean target size of their vertices, worst first, at circumcenters).
 * sizes (vertex_count,) >= 0 replaces the per-vertex target sizes (NULL:
 * keep; 0: no size target); new vertices interpolate them.  Points inside a
 * protecting ball are not inserted.  OK when every criterion holds;
 * REFINEMENT_LIMIT when max_insertions is reached or elements remain unmet
 * for a recorded reason (NONREPRESENTABLE: no exact split and no carrier
 * within the declared source deviation); CAPACITY_EXCEEDED when work_limit,
 * max_vertices or max_tetrahedra stops it.  The mesh stays valid in every
 * case; unmet elements are listed by unmet() and refused bounded carriers by
 * source_refusals().  counters (PHX_MC_TET_MESH_REFINE_COUNTERS,).
 *
 * improve(): repeated passes over tetrahedra whose smallest dihedral angle
 * is below min_dihedral_degrees, worst first: 2-3 face removal, edge
 * removal (3-2, 4-4 and n-to-2n-4 by optimal ring triangulation) and exact
 * relocation of interior and planar-facet vertices, each committed only when
 * it raises the local minimum dihedral angle.  Constrained faces and
 * segments never change.  OK when no sliver remains, REFINEMENT_LIMIT after
 * max_passes or when no operation improves the remaining ones (listed by
 * unmet()), CAPACITY_EXCEEDED when work_limit stops it.
 *
 * Operations for adaptation: flip_face (2-3 on an unconstrained interior
 * face (3,)), remove_edge (unconstrained, non-segment interior edge),
 * relocate (interior vertex to position (3,), exact orientation check),
 * remove_vertex (interior vertex, star retriangulated by the best valid
 * collapse).  With require_improvement != 0 the change must raise the local
 * minimum dihedral angle.  applied receives 1 when committed, 0 when refused.
 *
 * quality(): values (PHX_MC_TET_MESH_QUALITY_VALUES,) minimum and maximum
 * dihedral angle (degrees), maximum and mean radius-edge ratio, minimum and
 * total volume, mean per-cell minimum dihedral angle; histogram
 * (PHX_MC_TET_MESH_QUALITY_BINS,) of per-cell minimum dihedral angles in 5
 * degree bins over [0, 90); slivers: cells below sliver_degrees.
 * counts() (PHX_MC_TET_MESH_COUNTS,): vertices, tetrahedra, faces, segments,
 * unmet records, cumulative work.  export(): points (vertices, 3) including
 * removed ids, canonical sorted tetrahedra (even permutation from the
 * smallest id) and regions, faces (outward from the domain or from the lower
 * region) and sources, sorted segments and sources, vertex dimension and
 * target size.  unmet(): cells (unmet, 4), codes (unmet, 2) = (criterion,
 * reason) and the measured value of the violated criterion. */

#define PHX_MC_TET_MESH_BOUNDARY_FIXED 0
#define PHX_MC_TET_MESH_BOUNDARY_CONFORMING 1

#define PHX_MC_SOURCE_STRATUM_NONE 0
#define PHX_MC_SOURCE_STRATUM_SEGMENT 1
#define PHX_MC_SOURCE_STRATUM_FACET 2

#define PHX_MC_TET_MESH_CRITERION_RADIUS_EDGE 0
#define PHX_MC_TET_MESH_CRITERION_SIZE 1
#define PHX_MC_TET_MESH_CRITERION_DIHEDRAL 2
#define PHX_MC_TET_MESH_CRITERION_CONSTRUCTION 3
#define PHX_MC_TET_MESH_CRITERION_VALIDITY 4

#define PHX_MC_TET_MESH_REASON_BUDGET 0
#define PHX_MC_TET_MESH_REASON_FIXED_BOUNDARY 1
#define PHX_MC_TET_MESH_REASON_PROTECTED 2
#define PHX_MC_TET_MESH_REASON_NONREPRESENTABLE 3
#define PHX_MC_TET_MESH_REASON_REFUSED 4
#define PHX_MC_TET_MESH_REASON_NO_IMPROVEMENT 5

/* circumcenter insertions, subfacet splits, subsegment splits, encroachment
 * deferrals, refused insertions, queue pops, unmet cells, work, largest
 * cavity, vertices */
#define PHX_MC_TET_MESH_REFINE_COUNTERS 10
/* face removals (2-3), edge removals, relocations, vertex removals, passes,
 * attempts, remaining slivers, work, determinant-floor vertex insertions,
 * multiface removals */
#define PHX_MC_TET_MESH_IMPROVE_COUNTERS 10
#define PHX_MC_TET_MESH_QUALITY_VALUES 7
#define PHX_MC_TET_MESH_QUALITY_BINS 18
#define PHX_MC_TET_MESH_COUNTS 6
#define PHX_MC_TET_MESH_MEMORY_VALUES 6
#define PHX_MC_TET_MESH_MEMORY_LIMIT_BYTES 0
#define PHX_MC_TET_MESH_MEMORY_LIVE_BYTES 1
#define PHX_MC_TET_MESH_MEMORY_PEAK_BYTES 2
#define PHX_MC_TET_MESH_MEMORY_REQUESTED_BYTES 3
#define PHX_MC_TET_MESH_MEMORY_ALLOCATIONS 4
#define PHX_MC_TET_MESH_MEMORY_REFUSALS 5
PHX_MC_API int32_t phx_mc_tet_mesh_set_memory_limit(
    phx_mc_tet_mesh* mesh, int64_t max_scratch_bytes);
PHX_MC_API int32_t phx_mc_tet_mesh_memory_evidence(
    const phx_mc_tet_mesh* mesh, uint64_t* values);

PHX_MC_API int32_t phx_mc_tet_mesh_create(
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    const int32_t* tet_regions, int64_t face_count, const int32_t* faces,
    const int32_t* face_sources, int64_t segment_count, const int32_t* segments,
    const int32_t* segment_sources, const double* protection_radii,
    int64_t source_point_count, const double* source_points,
    int64_t source_face_count, const int32_t* source_faces, const int32_t* source_face_group_rows,
    const double* source_face_tolerances, int64_t source_segment_count,
    const int32_t* source_segments, const int32_t* source_segment_group_rows,
    const double* source_segment_tolerances, const int8_t* witness_strata,
    const int32_t* witness_entities, const double* witness_parameters,
    int32_t boundary_policy, int64_t max_vertices, int64_t max_tetrahedra,
    int64_t max_scratch_bytes, phx_mc_tet_mesh** mesh);
PHX_MC_API int32_t phx_mc_tet_mesh_refine(phx_mc_tet_mesh* mesh, const double* sizes,
                                          double radius_edge_bound, int64_t max_insertions,
                                          int64_t work_limit, int64_t* counters);
PHX_MC_API int32_t phx_mc_tet_mesh_improve(phx_mc_tet_mesh* mesh, double min_dihedral_degrees,
                                           double minimum_relative_determinant,
                                           int32_t max_passes, int64_t work_limit,
                                           int64_t* counters);
PHX_MC_API int32_t phx_mc_tet_mesh_flip_face(phx_mc_tet_mesh* mesh, const int32_t* face,
                                             int32_t require_improvement, int32_t* applied);
PHX_MC_API int32_t phx_mc_tet_mesh_remove_edge(phx_mc_tet_mesh* mesh, int32_t a, int32_t b,
                                               int32_t require_improvement, int32_t* applied);
PHX_MC_API int32_t phx_mc_tet_mesh_relocate(phx_mc_tet_mesh* mesh, int32_t vertex,
                                            const double* position, int32_t require_improvement,
                                            int32_t* applied);
// One atomic coordinated source-stratum motion. Vertices are sorted and unique;
// a zero shape bound disables that bound. Refusal publishes no coordinate.
PHX_MC_API int32_t phx_mc_tet_mesh_relocate_vertices(
    phx_mc_tet_mesh* mesh, int64_t vertex_count, const int32_t* vertices,
    const double* positions, double radius_edge_bound, double minimum_dihedral_degrees,
    int64_t work_limit, int32_t* applied);
PHX_MC_API int32_t phx_mc_tet_mesh_remove_vertex(phx_mc_tet_mesh* mesh, int32_t vertex,
                                                 int32_t require_improvement, int32_t* applied);
PHX_MC_API int32_t phx_mc_tet_mesh_quality(const phx_mc_tet_mesh* mesh, double sliver_degrees,
                                           double* values, int64_t* histogram, int64_t* slivers);
/* Read-only shape query on borrowed points (point_count, 3) and tetrahedra
 * (tet_count, 4), using the same constructions as quality(). No source labels,
 * mesh reconstruction or geometry mutation. values (3,): maximum circumradius
 * / shortest edge, minimum dihedral angle in degrees, and the number of cells
 * at or below minimum_relative_determinant; empty cells yield 0, 180, 0.
 * Degenerate/nonrepresentable constructions yield nonfinite radius-edge shape.
 * vertex_ids (point_count,) are the nonnegative int64 scientific vertex
 * identities of the borrowed points (a proposed vertex carries the identity it
 * will receive); negative or repeated identities within a cell are
 * INVALID_INPUT, never inferred from point or row order. Each
 * cell's floor is decided exactly on its canonical chart (smallest identity
 * first, orientation preserved) as det^2 > floor^2 * prod |v_k - v_0|^2, the
 * publishing cell validity semantics; minimum_relative_determinant lies in
 * [0, 1) and 0 evaluates no floor.
 * work_limit >= 0 additionally caps this query's coordinate-validation visits,
 * index visits, cell shape evaluations and floor evaluations, charged to
 * counts()[5] through the existing owner allowance. Neither allowance is reset;
 * capacity/allocation refusal leaves geometry intact and writes no result. */
PHX_MC_API int32_t phx_mc_tet_mesh_proposal_shape(
    phx_mc_tet_mesh* mesh, int64_t point_count, const double* points,
    const int64_t* vertex_ids, int64_t tet_count, const int32_t* tets,
    double minimum_relative_determinant, int64_t work_limit, double* values);
PHX_MC_API int32_t phx_mc_tet_mesh_counts(const phx_mc_tet_mesh* mesh, int64_t* counts);
/* Allocation-free cumulative work observation, also after a resource refusal. */
PHX_MC_API int32_t phx_mc_tet_mesh_work_units(const phx_mc_tet_mesh* mesh, int64_t* work);
/* Set a nonnegative remaining allowance relative to cumulative native work.
 * Editing operations with an explicit work budget may only narrow it and
 * restore the original absolute ceiling; proposal_shape() additionally caps it. */
PHX_MC_API int32_t phx_mc_tet_mesh_set_work_limit(
    phx_mc_tet_mesh* mesh, int64_t remaining_work_units);
PHX_MC_API int32_t phx_mc_tet_mesh_export(const phx_mc_tet_mesh* mesh, double* points,
                                          int32_t* tets, int32_t* tet_regions, int32_t* faces,
                                          int32_t* face_sources, int32_t* segments,
                                          int32_t* segment_sources, int8_t* vertex_dimension,
                                          double* vertex_sizes);
PHX_MC_API int32_t phx_mc_tet_mesh_unmet(const phx_mc_tet_mesh* mesh, int32_t* tets,
                                         int32_t* codes, double* values);
PHX_MC_API void phx_mc_tet_mesh_free(phx_mc_tet_mesh* mesh);

/* ------------------------------------------------ PLC boundary recovery (3D)
 * Validates an oriented piecewise-linear complex and recovers it in a
 * constrained tetrahedralization.  Polygons are loops polygon_vertices
 * [polygon_offsets[p], polygon_offsets[p + 1]) of distinct exactly coplanar
 * points; polygon_facets[p] names their facet and facet_regions (f, 2) the
 * region on the positive side of the loop's right-hand normal and on the
 * negative side (-1: void; equal: internal sheet).  segments (s, 2) are
 * explicit internal curves; seeds (k, 3) with seed_regions (k,) (-1: void)
 * cross-check the labeling.  boundary_policy PHX_MC_PLC3D_FIXED forbids every
 * boundary Steiner point; PHX_MC_PLC3D_CONFORMING splits PLC edges and facets
 * at exactly representable points exactly on them or, when none exists and
 * the declared source deviation is positive (facet_tolerances (f,),
 * segment_tolerances (s,), NULL: zero; an edge takes the smallest bound of
 * its segment and incident facets), at the correctly rounded carrier of an
 * exact witness on the PLC edge or input triangle whose certified deviation
 * that bound admits.  max_vertices bounds the output vertices,
 * max_tetrahedra the finite tetrahedra of the construction and work_limit
 * its work units.
 *
 * Returns PHX_MC_OK, or INVALID_INPUT / CONSTRAINT_INTERSECTION /
 * CAPACITY_EXCEEDED with *result holding failure evidence
 * (PHX_MC_PLC3D_FAILURE int64: reason, first entity kind, first id, second
 * kind, second id; see plc3d.hpp for the codes); argument and point-domain
 * errors leave *result null.  Sizes (PHX_MC_PLC3D_SIZES): points, domain
 * tetrahedra, constrained subfacets, subsegments, PLC edges, input triangles.
 * Export writes points (N, 3) (input first), positively oriented tetrahedra
 * (T, 4) with regions (T,), subfacets (F, 3) oriented like their source facet
 * with face_sources (F,), subsegments (S, 2) with their PLC edge (S,),
 * protecting radii (N,) (0 unprotected), the PLC edge table (E, 2) (explicit
 * segments, then facet-group boundary edges), vertex dimensions (N,) int8 and
 * the exact triangulation of the input polygons (P, 3) with the polygon of
 * each triangle (P,).  source_witnesses() writes per output point the
 * stratum (PHX_MC_SOURCE_STRATUM_*), PLC edge or input triangle, parameters
 * (N, 2) and certified deviation (N,) (zero exactly on the source), and
 * refusal (2,): the certified deviation and declared bound of a
 * source-deviation failure (plc3d.hpp reason 17), zero otherwise. */
#define PHX_MC_PLC3D_FIXED 0
#define PHX_MC_PLC3D_CONFORMING 1
#define PHX_MC_PLC3D_SIZES 6
#define PHX_MC_PLC3D_FAILURE 5
#define PHX_MC_PLC3D_COUNTERS 14
typedef struct phx_mc_plc3d phx_mc_plc3d;
PHX_MC_API int32_t phx_mc_plc3d_recover(
    int64_t point_count, const double* points, int64_t polygon_count,
    const int64_t* polygon_offsets, const int32_t* polygon_vertices,
    const int32_t* polygon_facets, int64_t facet_count, const int32_t* facet_regions,
    int64_t segment_count, const int32_t* segments, int64_t seed_count, const double* seeds,
    const int32_t* seed_regions, int32_t boundary_policy, const double* facet_tolerances,
    const double* segment_tolerances, int64_t max_vertices,
    int64_t max_tetrahedra, int64_t work_limit, uint64_t maximum_scratch_bytes,
    int32_t measure_phases, phx_mc_plc3d** result);
/* Prepare the authoritative source constraints through the same exact PLC
 * validation and polygon triangulation used by recovery, without tetrahedral
 * fill, feature protection, or region classification. On success sizes are
 * (original points, 0, 0, 0, PLC edges, input triangles); export supplies those
 * points, edges, triangles, and original polygon indices. Failure, measured
 * work, optional validation timing, memory evidence, and free use the same
 * result getters as recovery. The source-only operation has no recovery
 * vertex/cell capacity and preserves the caller's original source indices. */
PHX_MC_API int32_t phx_mc_plc3d_source_constraints(
    int64_t point_count, const double* points, int64_t polygon_count,
    const int64_t* polygon_offsets, const int32_t* polygon_vertices,
    const int32_t* polygon_facets, int64_t facet_count, const int32_t* facet_regions,
    int64_t segment_count, const int32_t* segments, int64_t work_limit,
    uint64_t maximum_scratch_bytes, int32_t measure_phases, phx_mc_plc3d** result);
PHX_MC_API void phx_mc_plc3d_sizes(const phx_mc_plc3d* result, int64_t* sizes);
PHX_MC_API void phx_mc_plc3d_counters(const phx_mc_plc3d* result, int64_t* counters);
PHX_MC_API void phx_mc_plc3d_failure(const phx_mc_plc3d* result, int64_t* failure);
PHX_MC_API void phx_mc_plc3d_export(const phx_mc_plc3d* result, double* points, int32_t* tets,
                                    int32_t* tet_regions, int32_t* faces,
                                    int32_t* face_sources, int32_t* segments,
                                    int32_t* segment_sources, double* protection,
                                    int32_t* plc_edges, int8_t* vertex_dimension,
                                    int32_t* input_triangles, int32_t* input_polygons);
PHX_MC_API void phx_mc_plc3d_source_witnesses(const phx_mc_plc3d* result, int8_t* strata,
                                              int32_t* entities, double* parameters,
                                              double* deviations, double* refusal);
PHX_MC_API void phx_mc_plc3d_free(phx_mc_plc3d* result);
PHX_MC_API int32_t phx_mc_plc3d_phase_times(
    const phx_mc_plc3d* result, int64_t* nanoseconds,
    int64_t* invocations, int32_t* enabled);
/* UINT64_MAX selects ordinary untracked allocation; zero is a hard byte cap.
 * enabled reports whether an actual allocator owner supplied memory[6]:
 * limit/live/peak/last request bytes, allocations, refusals. */
PHX_MC_API int32_t phx_mc_plc3d_memory_evidence(
    const phx_mc_plc3d* result, uint64_t* memory_evidence, int32_t* enabled);
PHX_MC_API int32_t phx_mc_tet_mesh_measure_execution(
    phx_mc_tet_mesh*, int32_t enabled);
PHX_MC_API int32_t phx_mc_tet_mesh_execution_times(
    const phx_mc_tet_mesh*, double* seconds, int32_t* measured);

/* ------------------------------------------------ restricted power cells
 * Power (Laguerre) cells of weighted sites (n, 3), weights (n,) restricted to a
 * tetrahedral domain decomposition and assembled into conforming polyhedra.
 * neighbor_offsets (n + 1,) int64 / neighbors int32 is the regular-triangulation
 * adjacency (sites without neighbors own no cell).  The domain is points
 * (m, 3), positively oriented tets (t, 4), tet_regions (t,) and
 * tet_face_facets (t, 4): the facet id (>= 0) of the constrained face opposite
 * each tet vertex or -1 for an unconstrained interior face; every domain
 * boundary face must be constrained.  Each cell is clipped exactly against the
 * tets its region meets (breadth-first over the adjacency from the owner of
 * the tet centroid); a vertex of a clipped piece is identified by the sites at
 * equal power distance and its carrier simplex of the decomposition, which
 * welds the pieces into one vertex set.  Pieces of one site joined through an
 * unconstrained face form one cell per connected component (split_sites (n,)
 * nonzero keeps every piece of that site a separate cell); constrained faces
 * always separate cells.  Coplanar fragments between the same two cells (or a
 * cell and one boundary facet) are merged into outward loops when their union
 * is a set of simple disks; vertices left on exactly two edges are removed.
 * Configurations whose vertex identities are not generic (more sites at equal
 * power distance or more constraint planes through a vertex than its
 * dimension admits), and fragments whose two sides disagree, report
 * PHX_MC_DEGENERATE_INPUT; exceeding max_pieces, max_vertices or work_limit
 * (clips plus walk steps) reports PHX_MC_CAPACITY_EXCEEDED.  The result handle
 * carries counters and failure evidence even when the status is not OK.
 * sizes (PHX_MC_POWER_CELLS_SIZES,): vertices, faces, face entries, cells,
 * pieces.  failure (PHX_MC_POWER_CELLS_FAILURE,): reason (0 none, 1 degenerate
 * vertex, 2 missing reciprocal piece, 3 inconsistent face, 4 piece budget,
 * 5 vertex budget, 6 work budget, 7 invalid domain, 8 clip failure, 9 seed
 * failure), site, tet, clip status.  counters (PHX_MC_POWER_CELLS_COUNTERS,):
 * clips, pieces, empty clips, walk steps, merged links, unmerged fragment
 * groups, removed collinear vertices, cells, faces, vertices.
 * Export: vertices (V, 3); faces as face_offsets (F + 1,) int64 into
 * face_vertices, oriented outward from face_cells[:, 0], face_cells (F, 2)
 * holding the owner cell and the other cell or -(1 + facet) on the domain
 * boundary, face_facets (F,) the facet id of constrained faces or -1; cells
 * with their site, region, volume, first moment (C, 3) and second moment
 * about the site; pieces with their site, tet, cell and volume. */
#define PHX_MC_POWER_CELLS_SIZES 5
#define PHX_MC_POWER_CELLS_FAILURE 4
#define PHX_MC_POWER_CELLS_COUNTERS 10
typedef struct phx_mc_power_cells phx_mc_power_cells;
PHX_MC_API int32_t phx_mc_restricted_power_cells(
    int64_t site_count, const double* sites, const double* weights,
    const int64_t* neighbor_offsets, const int32_t* neighbors, const int8_t* split_sites,
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    const int32_t* tet_regions, const int32_t* tet_face_facets, int64_t max_pieces,
    int64_t max_vertices, int64_t work_limit, int8_t record_phases,
    phx_mc_power_cells** result);
/* Exact prepared image coordinates are nonoverlapping, ascending-magnitude
 * expansion components. Offsets have 3*image_count+1 entries; site witnesses
 * and exported cell/piece site IDs use the image axis, whose original owner
 * is image_site_owners. No rounded image-site carrier defines a bisector. */
PHX_MC_API int32_t phx_mc_restricted_power_cells_exact(
    int64_t original_site_count, const double* original_sites,
    const double* original_weights, int64_t image_count,
    const int32_t* image_site_owners, const int64_t* image_coordinate_offsets,
    int64_t component_count, const double* image_coordinate_components,
    const int64_t* neighbor_offsets, const int32_t* neighbors, const int8_t* split_sites,
    int64_t point_count, const double* points, int64_t tet_count, const int32_t* tets,
    const int32_t* tet_regions, const int32_t* tet_face_facets, int64_t max_pieces,
    int64_t max_vertices, int64_t work_limit, int8_t record_phases,
    phx_mc_power_cells** result);
PHX_MC_API void phx_mc_power_cells_sizes(const phx_mc_power_cells* result, int64_t* sizes);
PHX_MC_API void phx_mc_power_cells_counters(const phx_mc_power_cells* result, int64_t* counters);
/* Actual charged source-validation/expansion-decode visits, including a
 * refused prefix. The ordinary binary64 source path reports zero. */
PHX_MC_API int64_t phx_mc_power_cells_source_work_units(const phx_mc_power_cells* result);
PHX_MC_API void phx_mc_power_cells_failure(const phx_mc_power_cells* result, int64_t* failure);
PHX_MC_API void phx_mc_power_cells_export(
    const phx_mc_power_cells* result, double* vertices, int64_t* face_offsets,
    int32_t* face_vertices, int32_t* face_cells, int32_t* face_facets, int32_t* cell_sites,
    int32_t* cell_regions, double* cell_volumes, double* cell_moments,
    double* cell_second_moments, int32_t* piece_sites, int32_t* piece_tets,
    int32_t* piece_cells, double* piece_volumes);
PHX_MC_API void phx_mc_power_cells_free(phx_mc_power_cells* result);
PHX_MC_API void phx_mc_power_cells_export_vertex_carriers(
    const phx_mc_power_cells* result, int32_t* vertex_carriers);
/* One sorted equal-power site CSR row per compacted output vertex. Carrier
 * simplex rows share that exact same vertex order. */
PHX_MC_API int64_t phx_mc_power_cells_site_witness_size(const phx_mc_power_cells* result);
PHX_MC_API void phx_mc_power_cells_export_vertex_sites(
    const phx_mc_power_cells* result, int64_t* offsets, int32_t* sites);
PHX_MC_API void phx_mc_power_cells_phase_seconds(
    const phx_mc_power_cells* result, double* seconds);

/* Simultaneous exact indirect-predicate triangle arrangements.
 * sizes: vertices, fragments, contact edges, coplanar rows, location rows.
 * Failed calls never publish a successful partial arrangement. */
typedef struct phx_mc_arrangement phx_mc_arrangement;
PHX_MC_API int32_t phx_mc_arrange_triangles(
    int64_t vertex_count, const double* vertices, int64_t triangle_count,
    const int64_t* triangles, const int32_t* surfaces, int64_t pair_count,
    const int64_t* pairs, int64_t max_points, int64_t max_fragments,
    int64_t* failed_face, phx_mc_arrangement** result);
PHX_MC_API void phx_mc_arrangement_sizes(const phx_mc_arrangement*, int64_t*);
PHX_MC_API void phx_mc_arrangement_counters(const phx_mc_arrangement*, int64_t*);
PHX_MC_API void phx_mc_arrangement_export(
    const phx_mc_arrangement*, double* vertices, double* bounds,
    int8_t* constructions, int64_t* origins, int64_t* fragments,
    int64_t* fragment_faces, int64_t* contact_edges, int64_t* coplanar,
    int64_t* locations);
/* Exact membership of a successful arrangement's fragments.  surfaces labels
 * must form the declared dense operand range [0, operand_count) (every label
 * used, operand_count <= triangles; INVALID_INPUT otherwise).  Fragments of one
 * operand are joined across shared edges that are not contact edges; each
 * component is numbered by, and represented by, its lowest fragment.
 * component_windings (components, operand_count) receives the exact winding
 * number of every other operand's input triangles at the centroid of the
 * representative's implicit corners (axis ray with symbolic perturbation,
 * filtered then exact dyadic predicates), or INT32_MIN for the component's own
 * operand and operands coplanar with it.  The matrix entries plus the
 * representative/triangle pair scans are admitted against work_limit and the
 * ambient execution scope before any allocation for them; CAPACITY_EXCEEDED
 * leaves no classification.  sizes (4,) receives components, operand_count,
 * pair scans and matrix entries (as far as evaluated on refusal);
 * fragment_components is (fragments,). */
PHX_MC_API int32_t phx_mc_arrangement_classify(phx_mc_arrangement* result,
                                               int64_t operand_count, int64_t work_limit,
                                               int64_t* failed_face, int64_t* sizes);
PHX_MC_API void phx_mc_arrangement_export_classification(const phx_mc_arrangement* result,
                                                         int64_t* fragment_components,
                                                         int32_t* component_windings);
PHX_MC_API void phx_mc_arrangement_free(phx_mc_arrangement*);

/* Allocation-free barycentric dual connectivity over shared entity IDs. */
PHX_MC_API int32_t phx_mc_triangle_dual_quads(
    int64_t triangle_count, int64_t vertex_count, const int64_t* nodes,
    int64_t quad_capacity, int64_t* quads);
PHX_MC_API int32_t phx_mc_tetrahedron_dual_hexes(
    int64_t tetrahedron_count, int64_t vertex_count, const int64_t* nodes,
    int64_t hex_capacity, int64_t* hexes);

/* SCI-reconciled convex cut-cell corner templates; source faces<=10 and
 * vertices<=16 per cell. Barycenter bank order is vertices, edges, faces,
 * cells. Exact original source outgoing-edge orientations select handedness.
 * On a nonsimple link REFINEMENT_LIMIT publishes zero hexes and witness
 * (original cell row, vertex row, edge degree, face degree), not nonexistence.
 * Original scratch/work caps and actual performed work are mandatory. */
PHX_MC_API int32_t phx_mc_polyhedron_corner_hexes(
    int64_t nv, int64_t ne, int64_t nf, int64_t nc,
    int64_t face_entries, int64_t cell_face_entries, int64_t corner_entries,
    const int64_t* edges, const int64_t* face_offsets, const int64_t* face_vertices,
    const int64_t* cell_offsets, const int64_t* cell_faces,
    const int64_t* vertex_offsets, const int64_t* cell_vertices,
    const int8_t* corner_orientations, int64_t vertex_count,
    int64_t hex_capacity, int64_t scratch_limit, int64_t work_limit, int64_t* hexes,
    int64_t* parent_cells, int64_t* hex_count, int64_t* witness, int64_t* work_units);

/* P1 level-set insertion; counts carry committed sizes and failed-edge evidence. */
PHX_MC_API int32_t phx_mc_level_set_split_3d(
    int64_t point_count, const double* points, const double* values,
    int64_t cell_count, const int32_t* cells, int64_t face_count,
    const int32_t* faces, int64_t edge_count, const int32_t* edges,
    const int8_t* protected_edges, int64_t max_vertices, int64_t max_cells,
    int64_t work_limit, int64_t cavity_limit, double* out_points,
    int32_t* out_cells, int32_t* out_parents, int32_t* out_sources,
    double* out_weights, int64_t* counters);

/* Explicit distance-envelope surface construction with two-sided bounds. */
PHX_MC_API int32_t phx_mc_surface_envelope_create(
    const double* vertices, int64_t vertex_count, const int64_t* faces,
    int64_t face_count, double offset, double spacing,
    double certificate_spacing, int64_t sample_limit, int64_t work_limit,
    int64_t surface_vertex_capacity, int64_t surface_triangle_capacity,
    int64_t volume_vertex_capacity, int64_t tetrahedron_capacity,
    void** result, int64_t* counts, double* bounds);
PHX_MC_API int32_t phx_mc_surface_envelope_arrays(
    void* result, double* surface_vertices, int64_t* surface_triangles,
    double* volume_vertices, int64_t* tetrahedra);
PHX_MC_API void phx_mc_surface_envelope_free(void* result);

/* Numerical restricted-Delaunay dual workset; endpoints are NOT enclosures. */
PHX_MC_API int32_t phx_mc_restricted_dual_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* domain, int64_t max_facets,
    int64_t work_limit, int32_t* facets, int32_t* cells, double* endpoints,
    int32_t* kinds, int32_t* item_status, int64_t* facet_count, int64_t* counters);

/* Same-family affine reference refinement templates; canonical status codes. */
PHX_MC_API int32_t phx_mc_mixed_template_counts(
    int32_t kind, int32_t axial, int32_t index, int32_t* counts);
PHX_MC_API int32_t phx_mc_mixed_template(
    int32_t kind, int32_t axial, int32_t index, int64_t capacity, double* references);

/* Rigorous circumcenter coordinate intervals; unresolved rows retain status. */
PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status);

/* Transactional metric operations; refused local edits have OK with no commit.
 * remove_multiface uses cell vertex tuples, never recycled native slot IDs. */
/* source_fraction is zero for an exactly incident represented construction.
 * A positive fraction in (0,1) additionally permits the recertified RNE carrier
 * of the exact source-row split; row bounds, protection and regions still govern
 * the same atomic cavity transaction. */
PHX_MC_API int32_t phx_mc_tet_mesh_split_edge(
    phx_mc_tet_mesh*, int32_t first, int32_t second, const double* position,
    double target_size, double source_fraction, int64_t work_limit, int32_t* inserted_vertex);
/* Atomic final two-child edge star; split_position is an exact strict old-edge
 * witness, final_position is the only new vertex coordinate ever published.
 * Curves retain their exact interval; moved facet wiring retains exact oriented
 * source-plane/material patch unions and reciprocal ghost closure. Quality
 * policy is caller-owned; this primitive requires native legality/positivity. */
PHX_MC_API int32_t phx_mc_tet_mesh_split_edge_relocate(
    phx_mc_tet_mesh*, int32_t first, int32_t second, const double* split_position,
    const double* final_position, double target_size, int64_t work_limit,
    int32_t* inserted_vertex);
/* Inspect the existing exact constrained conflict-cavity insertion at final_position.
 * split_position is an exact strict old-edge source witness; no vertex is published.
 * removed/proposed buffers each hold maximum_cavity_cells rows of four vertex IDs;
 * their combined finite row count is bounded by that allowance. counts is
 * [finite removed, finite proposed, insertion kind, accepted-state generation].
 * Ordinary geometric refusal
 * returns OK with zero counts; work/allocation/cavity refusal returns CAPACITY.
 * A nonzero source_fraction admits the original bounded source witness when
 * its carrier is not collinear with the represented seed edge; final_position
 * must then equal that carrier. Ghost closure/source/material reconstruction
 * remain owned by native prepare. */
PHX_MC_API int32_t phx_mc_tet_mesh_inspect_edge_insertion(
    phx_mc_tet_mesh*, int32_t first, int32_t second, const double* split_position,
    const double* final_position, double source_fraction, int64_t maximum_cavity_cells, int64_t work_limit,
    int32_t* removed_tetrahedra, int32_t* proposed_tetrahedra, int64_t* counts);
/* Reprepare and compare the actual inspected final cavity/source witness, then
 * commit once. Different/stale tuples refuse without changing accepted geometry. */
PHX_MC_API int32_t phx_mc_tet_mesh_commit_edge_insertion(
    phx_mc_tet_mesh*, int32_t first, int32_t second, const double* split_position,
    const double* final_position, double source_fraction, int64_t maximum_cavity_cells, int32_t insertion_kind,
    int64_t accepted_generation,
    int64_t removed_count, const int32_t* removed_tetrahedra, int64_t proposed_count,
    const int32_t* proposed_tetrahedra, double target_size, int64_t work_limit,
    int32_t* inserted_vertex);
PHX_MC_API int32_t phx_mc_tet_mesh_collapse_edge(
    phx_mc_tet_mesh*, int32_t remove, int32_t keep, int32_t require_improvement,
    int64_t work_limit, int32_t* applied);
PHX_MC_API int32_t phx_mc_tet_mesh_remove_multiface(
    phx_mc_tet_mesh*, int64_t cell_count, const int32_t* cell_vertices,
    int32_t apex, int32_t require_improvement, int64_t work_limit, int32_t* applied);
/* Bounded local weighted sliver exudation; no globally regular connectivity
 * is claimed. Only unprotected interior vertices receive power weights below
 * maximum_weight_fraction (0, 0.25] times their shortest incident edge squared.
 * No reconnection creates a cell at or below minimum_relative_determinant [0, 1)
 * on its canonical chart; remaining cells are unmet records (VALIDITY for
 * below-floor cells). weights (vertex_count,) receive accepted weights;
 * counters (6,): trials, flips, weighted vertices, passes, remaining slivers,
 * work. Returns OK, REFINEMENT_LIMIT or CAPACITY_EXCEEDED. */
PHX_MC_API int32_t phx_mc_tet_mesh_exude(
    phx_mc_tet_mesh*, double minimum_dihedral, double maximum_weight_fraction,
    double radius_edge_bound, double minimum_relative_determinant, int32_t maximum_passes,
    int64_t work_limit, double* weights, int64_t* counters);

/* Outward hull-ray endpoint tubes covering the complete declared domain. */
PHX_MC_API int32_t phx_mc_restricted_rays_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, const double* center_bounds, int64_t ray_count,
    const int32_t* facets, const int32_t* cells, const double* domain,
    double* endpoint_bounds, int32_t* kinds, int32_t* item_status);

PHX_MC_API int32_t phx_mc_tet_mesh_reconnect(
    phx_mc_tet_mesh*, int64_t removed_count, const int32_t* removed_tetrahedra,
    int64_t proposed_count, const int32_t* proposed_tetrahedra,
    int64_t work_limit, int32_t* applied);

PHX_MC_API int32_t phx_mc_tet_mesh_protect_vertices(
    phx_mc_tet_mesh*, int64_t count, const int32_t* vertices);
/* Nonmutating exact edge construction with cumulative work accounting.
 * preferred_fraction == 0 retains the original dyadic order; otherwise it must
 * be finite and strictly between 0 and 1. Try that fraction exactly, then rank
 * exact dyadic candidates by distance if refused. If none is representable,
 * try the bounded source-row RNE carrier at that preference (default one half).
 * parameter is the actual constructed fraction. Witness outputs name the
 * original source stratum/row/parameters and certified deviation, even for
 * exact zero-deviation constrained points. No candidate broadens a stratum or
 * its declared bound. If none is admitted, return OK with constructed == 0. */
PHX_MC_API int32_t phx_mc_tet_mesh_construct_edge_split(
    phx_mc_tet_mesh*, int32_t first, int32_t second, double preferred_fraction,
    int64_t work_limit, double* position, double* parameter, int32_t* constructed,
    int8_t* witness_stratum, int32_t* witness_entity, double* witness_parameters,
    double* witness_deviation);
/* Nonmutating exact constructions on actual smooth source curves. neighbors
 * (n,2) are the two same-source incident segment endpoints of vertices (n).
 * desired_positions (n,3) select dominant physical source coordinates, not
 * binary affine fractions. Refused rows have NaN point/coordinate outputs.
 * No mesh coordinate, topology, source identity or accepted state is changed. */
PHX_MC_API int32_t phx_mc_tet_mesh_construct_curve_points(
    phx_mc_tet_mesh*, int64_t count, const int32_t* vertices,
    const int32_t* neighbors, const double* desired_positions, int64_t work_limit,
    double* positions, double* coordinates, int32_t* constructed);
/* Source evidence: values (2,) the largest declared row bound and the achieved
 * bound (the largest certified deviation of a live vertex, 0 when exact);
 * face_ancestors (faces,) / segment_ancestors (segments,) the source row of
 * each exported face/segment (-1: none); per vertex the witness stratum,
 * row, parameters (vertices, 2) and certified deviation.  source_refusals()
 * writes count (1,) and, when the arrays are non-NULL with capacity >= count,
 * the refused bounded carriers of the latest refinement: represented edges
 * (count, 2), stratum, source row and values (count, 2) = (certified
 * deviation, declared bound). */
PHX_MC_API int32_t phx_mc_tet_mesh_source_evidence(
    const phx_mc_tet_mesh*, double* values, int32_t* face_ancestors,
    int32_t* segment_ancestors, int8_t* witness_strata, int32_t* witness_entities,
    double* witness_parameters, double* witness_deviations);
PHX_MC_API int32_t phx_mc_tet_mesh_source_refusals(
    const phx_mc_tet_mesh*, int64_t capacity, int64_t* count, int32_t* edges,
    int8_t* strata, int32_t* entities, double* values);
PHX_MC_API int32_t phx_mc_subdivide_hex_grid(
    int64_t parent_count, int64_t vertex_count, const int64_t* nodes,
    int64_t hex_capacity, int64_t* hexes);

#ifdef __cplusplus
}
#endif

#endif /* PHYDRAX_MESHCORE_H */
