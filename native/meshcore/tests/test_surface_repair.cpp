//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// C++ unit tests for single-soup exact self-arrangement, solid extraction,
// displacement-bounded welding/cleaning, Liepa hole filling, and orientation.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "phydrax_meshcore.h"

namespace {

#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

struct Mesh {
  std::vector<double> vertices;
  std::vector<int64_t> triangles;
};

Mesh make_box(double x0, double y0, double z0, double x1, double y1, double z1,
              bool invert = false) {
  Mesh m;
  m.vertices = {
      x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
      x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
  };
  const int64_t tris[12][3] = {
      {0, 2, 1}, {0, 3, 2},  // -z
      {4, 5, 6}, {4, 6, 7},  // +z
      {0, 1, 5}, {0, 5, 4},  // -y
      {3, 7, 6}, {3, 6, 2},  // +y
      {0, 4, 7}, {0, 7, 3},  // -x
      {1, 2, 6}, {1, 6, 5},  // +x
  };
  for (const auto& t : tris) {
    if (invert) {
      m.triangles.push_back(t[0]);
      m.triangles.push_back(t[2]);
      m.triangles.push_back(t[1]);
    } else {
      m.triangles.push_back(t[0]);
      m.triangles.push_back(t[1]);
      m.triangles.push_back(t[2]);
    }
  }
  return m;
}

void append_mesh(Mesh& dst, const Mesh& src) {
  const int64_t offset = static_cast<int64_t>(dst.vertices.size() / 3);
  dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
  for (const int64_t idx : src.triangles) dst.triangles.push_back(idx + offset);
}

Mesh take_surface(void* handle) {
  CHECK(handle != nullptr);
  int64_t sizes[2] = {0, 0};
  CHECK(phx_mc_surface_result_sizes(handle, sizes) == PHX_MC_OK);
  Mesh m;
  m.vertices.resize(static_cast<std::size_t>(3 * sizes[0]));
  m.triangles.resize(static_cast<std::size_t>(3 * sizes[1]));
  CHECK(phx_mc_surface_result_arrays(handle, sizes[0], sizes[1],
                                     m.vertices.empty() ? nullptr : m.vertices.data(),
                                     m.triangles.empty() ? nullptr : m.triangles.data()) ==
        PHX_MC_OK);
  phx_mc_surface_result_free(handle);
  return m;
}

double signed_volume(const Mesh& m) {
  const std::size_t n = m.triangles.size() / 3;
  double v6 = 0.0;
  for (std::size_t f = 0; f < n; ++f) {
    const double* a = m.vertices.data() + 3 * m.triangles[3 * f];
    const double* b = m.vertices.data() + 3 * m.triangles[3 * f + 1];
    const double* c = m.vertices.data() + 3 * m.triangles[3 * f + 2];
    const double cx = b[1] * c[2] - b[2] * c[1];
    const double cy = b[2] * c[0] - b[0] * c[2];
    const double cz = b[0] * c[1] - b[1] * c[0];
    v6 += a[0] * cx + a[1] * cy + a[2] * cz;
  }
  return v6 / 6.0;
}

int64_t verify_embedding_findings(const Mesh& m) {
  const auto nv = static_cast<int64_t>(m.vertices.size() / 3);
  const auto nt = static_cast<int64_t>(m.triangles.size() / 3);
  phx_mc_surface_embedding_verdict verdict{};
  CHECK(phx_mc_verify_surface_embedding(nv, m.vertices.data(), nt, m.triangles.data(), 0, nullptr,
                                        &verdict) == PHX_MC_OK);
  return verdict.finding_count;
}

void test_self_intersecting_overlapping_boxes() {
  Mesh soup = make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  append_mesh(soup, make_box(0.5, 0.5, 0.5, 1.5, 1.5, 1.5));
  // Add an internal baffle wall cutting straight across the first cube.
  const int64_t off = static_cast<int64_t>(soup.vertices.size() / 3);
  const double baffle_v[12] = {0.25, 0.0, 0.0, 0.25, 1.0, 0.0, 0.25, 1.0, 1.0, 0.25, 0.0, 1.0};
  soup.vertices.insert(soup.vertices.end(), baffle_v, baffle_v + 12);
  const int64_t baffle_f[6] = {off, off + 1, off + 2, off, off + 2, off + 3};
  soup.triangles.insert(soup.triangles.end(), baffle_f, baffle_f + 6);

  int64_t failed_face = -1;
  phx_mc_arrangement* arr = nullptr;
  CHECK(phx_mc_arrange_soup(static_cast<int64_t>(soup.vertices.size() / 3), soup.vertices.data(),
                            static_cast<int64_t>(soup.triangles.size() / 3), soup.triangles.data(),
                            0, nullptr, &failed_face, &arr) == PHX_MC_OK);
  CHECK(arr != nullptr);

  phx_mc_solid_extraction_report rep{};
  void* out_handle = nullptr;
  CHECK(phx_mc_arrangement_extract_solid(arr, PHX_MC_WINDING_NONZERO, 0, &rep, &out_handle) ==
        PHX_MC_OK);
  phx_mc_arrangement_free(arr);

  const Mesh solid = take_surface(out_handle);
  CHECK(rep.closed == 1);
  CHECK(rep.boundary_edge_count == 0);
  CHECK(rep.nonmanifold_edge_count == 0);
  CHECK(rep.nonmanifold_vertex_count == 0);
  CHECK(verify_embedding_findings(solid) == 0);
  // Volume of [0,1]^3 U [0.5,1.5]^3 = 1 + 1 - 0.5^3 = 1.875.
  CHECK(std::fabs(signed_volume(solid) - 1.875) < 1.0e-12);
}

void test_coplanar_cancellation_and_nested_cavity() {
  // Outer box [0, 2]^3 (volume 8) with an inner box [0.5, 1.5]^3 (volume 1),
  // plus an exact canceling opposite pair on the outer face.
  Mesh soup = make_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
  append_mesh(soup, make_box(0.5, 0.5, 0.5, 1.5, 1.5, 1.5, /*invert=*/true));
  // Add an extra forward + reversed duplicate of face 0 (net multiplicity 0).
  const int64_t a = soup.triangles[0], b = soup.triangles[1], c = soup.triangles[2];
  soup.triangles.insert(soup.triangles.end(), {a, b, c, a, c, b});

  int64_t failed_face = -1;
  phx_mc_arrangement* arr = nullptr;
  CHECK(phx_mc_arrange_soup(static_cast<int64_t>(soup.vertices.size() / 3), soup.vertices.data(),
                            static_cast<int64_t>(soup.triangles.size() / 3), soup.triangles.data(),
                            0, nullptr, &failed_face, &arr) == PHX_MC_OK);
  phx_mc_solid_extraction_report rep{};
  void* out_handle = nullptr;
  CHECK(phx_mc_arrangement_extract_solid(arr, PHX_MC_WINDING_POSITIVE, 0, &rep, &out_handle) ==
        PHX_MC_OK);
  phx_mc_arrangement_free(arr);

  const Mesh solid = take_surface(out_handle);
  CHECK(rep.closed == 1);
  CHECK(rep.canceled_coplanar_fragments >= 2);
  CHECK(rep.used_generalized_winding == 0);
  CHECK(verify_embedding_findings(solid) == 0);
  // Hollow box net volume = 8 - 1 = 7.
  CHECK(std::fabs(signed_volume(solid) - 7.0) < 1.0e-12);
}

void test_touching_cubes_pinch_split() {
  // Two unit cubes touching along shared edge x=1, y=1, z in [0,1].
  Mesh soup = make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  append_mesh(soup, make_box(1.0, 1.0, 0.0, 2.0, 2.0, 1.0));
  int64_t failed_face = -1;
  phx_mc_arrangement* arr = nullptr;
  CHECK(phx_mc_arrange_soup(static_cast<int64_t>(soup.vertices.size() / 3), soup.vertices.data(),
                            static_cast<int64_t>(soup.triangles.size() / 3), soup.triangles.data(),
                            0, nullptr, &failed_face, &arr) == PHX_MC_OK);

  phx_mc_solid_extraction_report rep_unsplit{};
  void* h_unsplit = nullptr;
  CHECK(phx_mc_arrangement_extract_solid(arr, PHX_MC_WINDING_NONZERO, 0, &rep_unsplit, &h_unsplit) ==
        PHX_MC_OK);
  const Mesh unsplit = take_surface(h_unsplit);
  CHECK(rep_unsplit.nonmanifold_edge_count >= 1);
  CHECK(std::fabs(signed_volume(unsplit) - 2.0) < 1.0e-12);

  phx_mc_solid_extraction_report rep_split{};
  void* h_split = nullptr;
  CHECK(phx_mc_arrangement_extract_solid(arr, PHX_MC_WINDING_NONZERO, 1, &rep_split, &h_split) ==
        PHX_MC_OK);
  phx_mc_arrangement_free(arr);

  const Mesh split = take_surface(h_split);
  CHECK(rep_split.closed == 1);
  CHECK(rep_split.nonmanifold_edge_count == 0);
  CHECK(rep_split.nonmanifold_vertex_count == 0);
  CHECK(rep_split.split_pinch_vertices >= 2);
  CHECK(std::fabs(signed_volume(split) - 2.0) < 1.0e-12);
}

void test_weld_t_junction_and_sliver_clean() {
  // Construct a cracked cube where vertex 0 of face 0 is perturbed by 1e-4, plus a degenerate face.
  Mesh cube = make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  // Duplicate vertex 0 with a 1e-4 crack offset and attach it to face 0.
  const int64_t v_crack = static_cast<int64_t>(cube.vertices.size() / 3);
  cube.vertices.insert(cube.vertices.end(), {1.0e-4, 0.0, 0.0});
  cube.triangles[0] = v_crack;
  // Add an exact collinear degenerate triangle.
  const int64_t v_mid = static_cast<int64_t>(cube.vertices.size() / 3);
  cube.vertices.insert(cube.vertices.end(), {0.5, 0.0, 0.0});
  cube.triangles.insert(cube.triangles.end(), {0, v_mid, 1});

  phx_mc_surface_clean_request req{};
  req.vertices = cube.vertices.data();
  req.vertex_count = static_cast<int64_t>(cube.vertices.size() / 3);
  req.triangles = cube.triangles.data();
  req.triangle_count = static_cast<int64_t>(cube.triangles.size() / 3);
  req.weld_tolerance = 5.0e-4;
  req.drop_degenerate = 1;
  req.preserve_embedding = 1;

  phx_mc_surface_clean_report rep{};
  void* out_handle = nullptr;
  CHECK(phx_mc_surface_weld_and_clean(&req, &rep, &out_handle) == PHX_MC_OK);
  const Mesh cleaned = take_surface(out_handle);
  CHECK(rep.welded_vertices >= 1);
  CHECK(rep.dropped_degenerate_faces >= 1);
  CHECK(rep.max_vertex_displacement <= 5.0e-4 + 1.0e-12);
  CHECK(cleaned.triangles.size() / 3 == 12);
  CHECK(verify_embedding_findings(cleaned) == 0);
}

void test_liepa_hole_filling_and_embedding_gate() {
  // Open cube missing top face (+z, 2 triangles -> 4-edge hole).
  Mesh open_cube = make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
  open_cube.triangles.erase(open_cube.triangles.begin() + 6, open_cube.triangles.begin() + 12);
  CHECK(open_cube.triangles.size() / 3 == 10);

  phx_mc_surface_hole_fill_request req{};
  req.vertices = open_cube.vertices.data();
  req.vertex_count = static_cast<int64_t>(open_cube.vertices.size() / 3);
  req.triangles = open_cube.triangles.data();
  req.triangle_count = static_cast<int64_t>(open_cube.triangles.size() / 3);
  req.max_hole_edges = 16;
  req.max_hole_diameter = 2.0;
  req.fair_caps = 1;
  req.refine_caps = 1;
  req.check_embedding = 1;

  phx_mc_surface_hole_fill_report rep{};
  void* out_handle = nullptr;
  CHECK(phx_mc_surface_fill_holes(&req, &rep, &out_handle) == PHX_MC_OK);
  const Mesh filled = take_surface(out_handle);
  CHECK(rep.filled_hole_count == 1);
  CHECK(rep.skipped_large_holes == 0);
  CHECK(rep.colliding_cap_triangles == 0);
  CHECK(rep.remaining_boundary_edges == 0);
  CHECK(std::fabs(rep.filled_hole_area - 1.0) < 1.0e-9);
  CHECK(verify_embedding_findings(filled) == 0);
  CHECK(std::fabs(signed_volume(filled) - 1.0) < 1.0e-9);
}

void test_orient_nested_cavities_and_prune_debris() {
  // Outer box inverted, inner cavity outward (both wrong!), plus a tiny 1-triangle floater.
  Mesh mesh = make_box(0.0, 0.0, 0.0, 4.0, 4.0, 4.0, /*invert=*/true);
  append_mesh(mesh, make_box(1.0, 1.0, 1.0, 3.0, 3.0, 3.0, /*invert=*/false));
  const int64_t off = static_cast<int64_t>(mesh.vertices.size() / 3);
  mesh.vertices.insert(mesh.vertices.end(), {10.0, 10.0, 10.0, 10.01, 10.0, 10.0, 10.0, 10.01, 10.0});
  mesh.triangles.insert(mesh.triangles.end(), {off, off + 1, off + 2});

  phx_mc_surface_orient_request req{};
  req.vertices = mesh.vertices.data();
  req.vertex_count = static_cast<int64_t>(mesh.vertices.size() / 3);
  req.triangles = mesh.triangles.data();
  req.triangle_count = static_cast<int64_t>(mesh.triangles.size() / 3);
  req.orient_shells = 1;
  req.prune_small = 1;
  req.min_component_faces = 4;

  phx_mc_surface_orient_report rep{};
  void* out_handle = nullptr;
  CHECK(phx_mc_surface_orient_and_prune(&req, &rep, &out_handle) == PHX_MC_OK);
  const Mesh fixed = take_surface(out_handle);
  CHECK(rep.component_count == 3);
  CHECK(rep.cavity_component_count == 1);
  CHECK(rep.pruned_component_count == 1);
  CHECK(rep.pruned_face_count == 1);
  CHECK(rep.flipped_face_count == 24);
  CHECK(verify_embedding_findings(fixed) == 0);
  // Outer volume 64 minus inner cavity volume 8 = 56.
  CHECK(std::fabs(signed_volume(fixed) - 56.0) < 1.0e-9);
}

}  // namespace

int main() {
  test_self_intersecting_overlapping_boxes();
  test_coplanar_cancellation_and_nested_cavity();
  test_touching_cubes_pinch_split();
  test_weld_t_junction_and_sliver_clean();
  test_liepa_hole_filling_and_embedding_gate();
  test_orient_nested_cavities_and_prune_debris();
  std::printf("test_surface_repair: all 6 native test groups passed\n");
  return 0;
}
