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
  PHX_MC_CONSTRAINT_INTERSECTION = 7,/* constraint segments cross in their interiors */
  PHX_MC_REFINEMENT_LIMIT = 8,       /* valid mesh; quality targets unmet within max_steiner */
  PHX_MC_INTERNAL_ERROR = 9
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
                                      int64_t max_triangles, phx_mc_mesh** mesh);
PHX_MC_API int32_t phx_mc_regular_2d(int64_t point_count, const double* points,
                                     const double* weights, int64_t max_triangles,
                                     phx_mc_mesh** mesh);
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
 * input points.  cell_segments (cell_count, 3) holds, for the edge opposite each
 * cell vertex, the input segment index it lies on or -1.  Returns
 * PHX_MC_REFINEMENT_LIMIT with a valid conforming mesh when the quality targets
 * are unmet after max_steiner insertions, and PHX_MC_CONSTRAINT_INTERSECTION when
 * two segments cross in their interiors. */
PHX_MC_API int32_t phx_mc_constrained_delaunay_2d(
    int64_t point_count, const double* points, int64_t segment_count, const int32_t* segments,
    int64_t hole_count, const double* holes, int32_t keep_convex_hull, double min_angle_degrees,
    double max_area, int64_t max_steiner, int64_t max_triangles, phx_mc_mesh** mesh);

PHX_MC_API int32_t phx_mc_mesh_dimension(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_point_count(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_cell_count(const phx_mc_mesh* mesh);
PHX_MC_API int64_t phx_mc_mesh_input_point_count(const phx_mc_mesh* mesh);
PHX_MC_API void phx_mc_mesh_copy_points(const phx_mc_mesh* mesh, double* points);
PHX_MC_API void phx_mc_mesh_copy_cells(const phx_mc_mesh* mesh, int32_t* cells);
PHX_MC_API void phx_mc_mesh_copy_vertex_map(const phx_mc_mesh* mesh, int32_t* vertex_map);
/* (cell_count, 3); filled with -1 for meshes without constraints. */
PHX_MC_API void phx_mc_mesh_copy_cell_segments(const phx_mc_mesh* mesh, int32_t* cell_segments);
PHX_MC_API void phx_mc_mesh_free(phx_mc_mesh* mesh);

#ifdef __cplusplus
}
#endif

#endif /* PHYDRAX_MESHCORE_H */
