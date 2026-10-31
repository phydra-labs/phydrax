//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Explicit distance-envelope construction, certified two-sided bounds and owned-array extents.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {
void triangle_envelope() {
  constexpr double vertices[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  constexpr int64_t faces[] = {0, 1, 2};
  void* handle = nullptr;
  std::array<double, 4> bounds{};
  const phx_mc_surface_envelope_create_request request{
      .vertices = vertices,
      .vertex_count = 3,
      .faces = faces,
      .face_count = 1,
      .offset = 3.0,
      .spacing = 0.5,
      .certificate_spacing = 0.5,
      .result = &handle,
      .bounds = bounds.data(),
  };
  const int32_t create_status = phx_mc_surface_envelope_create(&request);
  if (create_status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "triangle envelope create failed: status=%d offset=%.17g spacing=%.17g "
                 "certificate_spacing=%.17g handle=%p\n",
                 create_status, request.offset, request.spacing, request.certificate_spacing,
                 handle);
    PHX_CHECK(create_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;
  PHX_CHECK(bounds[0] >= 0.0);
  PHX_CHECK(bounds[1] > bounds[2]);
  PHX_CHECK(bounds[2] > 0.0);
  PHX_CHECK(bounds[3] > 0.0);
  std::array<int64_t, 4> sizes{};
  const int32_t sizes_status = phx_mc_surface_envelope_sizes(handle, sizes.data());
  PHX_CHECK(sizes_status == PHX_MC_OK);
  if (sizes_status != PHX_MC_OK) {
    phx_mc_surface_envelope_free(handle);
    return;
  }
  PHX_CHECK(sizes[0] >= 4);
  PHX_CHECK(sizes[1] >= 4);
  PHX_CHECK(sizes[2] >= sizes[0]);
  PHX_CHECK(sizes[3] >= 1);
  if (sizes[0] < 4 || sizes[1] < 4 || sizes[2] < sizes[0] || sizes[3] < 1) {
    phx_mc_surface_envelope_free(handle);
    return;
  }
  std::vector<double> surface_vertices(static_cast<std::size_t>(3 * sizes[0]), -7.0);
  std::vector<int64_t> surface_triangles(static_cast<std::size_t>(3 * sizes[1]), -7);
  std::vector<double> volume_vertices(static_cast<std::size_t>(3 * sizes[2]), -7.0);
  std::vector<int64_t> tetrahedra(static_cast<std::size_t>(4 * sizes[3]), -7);
  const int32_t arrays_status = phx_mc_surface_envelope_arrays(
      handle, sizes[0], sizes[1], sizes[2], sizes[3], surface_vertices.data(),
      surface_triangles.data(), volume_vertices.data(), tetrahedra.data());
  PHX_CHECK(arrays_status == PHX_MC_OK);
  if (arrays_status != PHX_MC_OK) {
    phx_mc_surface_envelope_free(handle);
    return;
  }
  for (int64_t cell = 0; cell < sizes[3]; ++cell) {
    const int64_t* t = tetrahedra.data() + 4 * cell;
    PHX_CHECK(phx::mc::orient3d(volume_vertices.data() + 3 * t[0],
                                volume_vertices.data() + 3 * t[1],
                                volume_vertices.data() + 3 * t[2],
                                volume_vertices.data() + 3 * t[3]) > 0);
  }
  std::vector<double> short_vertices(static_cast<std::size_t>(3 * (sizes[0] - 1)), -9.0);
  std::vector<int64_t> short_triangles(static_cast<std::size_t>(3 * sizes[1]), -9);
  std::vector<double> short_volume(static_cast<std::size_t>(3 * sizes[2]), -9.0);
  std::vector<int64_t> short_tets(static_cast<std::size_t>(4 * sizes[3]), -9);
  PHX_CHECK(phx_mc_surface_envelope_arrays(
                handle, sizes[0] - 1, sizes[1], sizes[2], sizes[3], short_vertices.data(),
                short_triangles.data(), short_volume.data(), short_tets.data()) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(short_vertices.front() == -9.0);
  PHX_CHECK(short_triangles.front() == -9);
  phx_mc_surface_envelope_free(handle);
}
void source_tetrahedron_envelope() {
  constexpr double vertices[] = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr int64_t faces[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  void* handle = nullptr;
  std::array<double, 4> bounds{};
  const phx_mc_surface_envelope_create_request request{
      .vertices = vertices,
      .vertex_count = 4,
      .faces = faces,
      .face_count = 4,
      .offset = 1.5,
      .spacing = 0.4,
      .certificate_spacing = 0.4,
      .result = &handle,
      .bounds = bounds.data(),
  };
  const int32_t create_status = phx_mc_surface_envelope_create(&request);
  if (create_status != PHX_MC_OK) {
    std::fprintf(stderr,
                 "source tetrahedron envelope create failed: status=%d offset=%.17g "
                 "spacing=%.17g certificate_spacing=%.17g handle=%p\n",
                 create_status, request.offset, request.spacing, request.certificate_spacing,
                 handle);
    PHX_CHECK(create_status == PHX_MC_OK);
    return;
  }
  PHX_CHECK(bounds[0] >= 0.0);
  PHX_CHECK(bounds[1] > bounds[2] && bounds[2] > 0.0 && bounds[3] > 0.0);
  PHX_CHECK(handle != nullptr);
  if (handle == nullptr) return;

  std::array<int64_t, 4> sizes{};
  PHX_CHECK(phx_mc_surface_envelope_sizes(handle, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] > 0 && sizes[1] > 0 && sizes[2] >= sizes[0] && sizes[3] > 0);
  if (sizes[0] <= 0 || sizes[1] <= 0 || sizes[2] < sizes[0] || sizes[3] <= 0) {
    phx_mc_surface_envelope_free(handle);
    return;
  }
  std::vector<double> surface_vertices(static_cast<std::size_t>(3 * sizes[0]));
  std::vector<int64_t> surface_triangles(static_cast<std::size_t>(3 * sizes[1]));
  std::vector<double> volume_vertices(static_cast<std::size_t>(3 * sizes[2]));
  std::vector<int64_t> tetrahedra(static_cast<std::size_t>(4 * sizes[3]));
  const int32_t arrays_status = phx_mc_surface_envelope_arrays(
      handle, sizes[0], sizes[1], sizes[2], sizes[3], surface_vertices.data(),
      surface_triangles.data(), volume_vertices.data(), tetrahedra.data());
  PHX_CHECK(arrays_status == PHX_MC_OK);
  if (arrays_status != PHX_MC_OK) {
    phx_mc_surface_envelope_free(handle);
    return;
  }

  using Face = std::array<int64_t, 3>;
  const auto face_key = [](int64_t a, int64_t b, int64_t c) {
    Face key{a, b, c};
    std::sort(key.begin(), key.end());
    return key;
  };
  std::map<Face, int32_t> surface;
  std::map<Face, int32_t> incidence;
  std::vector<int8_t> referenced(static_cast<std::size_t>(sizes[0]), 0);
  bool surface_indices_valid = true;
  for (int64_t triangle = 0; triangle < sizes[1]; ++triangle) {
    const int64_t* face = surface_triangles.data() + 3 * triangle;
    bool valid = true;
    for (int corner = 0; corner < 3; ++corner) {
      PHX_CHECK(face[corner] >= 0 && face[corner] < sizes[0]);
      valid = valid && face[corner] >= 0 && face[corner] < sizes[0];
      if (face[corner] >= 0 && face[corner] < sizes[0]) {
        referenced[static_cast<std::size_t>(face[corner])] = 1;
      }
    }
    surface_indices_valid = surface_indices_valid && valid;
    if (!valid) continue;
    const auto [unused, inserted] = surface.emplace(face_key(face[0], face[1], face[2]), 1);
    (void)unused;
    PHX_CHECK(inserted);
  }
  if (!surface_indices_valid) {
    phx_mc_surface_envelope_free(handle);
    return;
  }
  for (std::size_t i = 0; i < surface_vertices.size(); ++i) {
    PHX_CHECK(surface_vertices[i] == volume_vertices[i]);
  }
  for (int8_t used : referenced) PHX_CHECK(used != 0);
  // The source tetrahedron puts grid vertices at distance exactly 1.5 (the
  // offset) from its faces, e.g. (-1.5, 0.5, 0.5). Their sampled values round
  // next to the offset and used to collapse ~10 roots within 4e-14 of each
  // such vertex. Separated grid values keep every root at least
  // band / (L + 2 band) of its edge (band = spacing / 8 here) from the grid
  // vertices, so distinct surface vertices stay far apart.
  double closest = std::numeric_limits<double>::infinity();
  for (int64_t first = 0; first < sizes[0]; ++first) {
    for (int64_t second = first + 1; second < sizes[0]; ++second) {
      const double* p = surface_vertices.data() + 3 * first;
      const double* q = surface_vertices.data() + 3 * second;
      closest = std::min(closest, std::hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
  }
  PHX_CHECK(closest >= request.spacing / 64);

  const auto determinant = [](const double* a, const double* b,
                              const double* c, const double* d) {
    const long double u0 = static_cast<long double>(b[0]) - a[0];
    const long double u1 = static_cast<long double>(b[1]) - a[1];
    const long double u2 = static_cast<long double>(b[2]) - a[2];
    const long double v0 = static_cast<long double>(c[0]) - a[0];
    const long double v1 = static_cast<long double>(c[1]) - a[1];
    const long double v2 = static_cast<long double>(c[2]) - a[2];
    const long double w0 = static_cast<long double>(d[0]) - a[0];
    const long double w1 = static_cast<long double>(d[1]) - a[1];
    const long double w2 = static_cast<long double>(d[2]) - a[2];
    return u0 * (v1 * w2 - v2 * w1) -
           u1 * (v0 * w2 - v2 * w0) +
           u2 * (v0 * w1 - v1 * w0);
  };
  long double tetrahedra_volume6 = 0.0L;
  for (int64_t cell = 0; cell < sizes[3]; ++cell) {
    const int64_t* tet = tetrahedra.data() + 4 * cell;
    bool valid = true;
    for (int corner = 0; corner < 4; ++corner) {
      PHX_CHECK(tet[corner] >= 0 && tet[corner] < sizes[2]);
      valid = valid && tet[corner] >= 0 && tet[corner] < sizes[2];
    }
    if (!valid) continue;
    const double* a = volume_vertices.data() + 3 * tet[0];
    const double* b = volume_vertices.data() + 3 * tet[1];
    const double* c = volume_vertices.data() + 3 * tet[2];
    const double* d = volume_vertices.data() + 3 * tet[3];
    PHX_CHECK(phx::mc::orient3d(a, b, c, d) > 0);
    tetrahedra_volume6 += determinant(a, b, c, d);
    for (int opposite = 0; opposite < 4; ++opposite) {
      std::array<int64_t, 3> face{};
      int slot = 0;
      for (int corner = 0; corner < 4; ++corner)
        if (corner != opposite) face[static_cast<std::size_t>(slot++)] = tet[corner];
      ++incidence[face_key(face[0], face[1], face[2])];
    }
  }

  long double surface_volume6 = 0.0L;
  int64_t boundary_triangles = 0;
  for (int64_t triangle = 0; triangle < sizes[1]; ++triangle) {
    const int64_t* face = surface_triangles.data() + 3 * triangle;
    const double origin[3] = {0.0, 0.0, 0.0};
    const double* a = surface_vertices.data() + 3 * face[0];
    const double* b = surface_vertices.data() + 3 * face[1];
    const double* c = surface_vertices.data() + 3 * face[2];
    surface_volume6 += determinant(origin, a, b, c);
    PHX_CHECK(incidence[face_key(face[0], face[1], face[2])] == 1);
  }
  for (const auto& [face, count] : incidence) {
    PHX_CHECK(count == 1 || count == 2);
    if (count == 1) {
      ++boundary_triangles;
      PHX_CHECK(surface.contains(face));
    } else {
      PHX_CHECK(!surface.contains(face));
    }
  }
  PHX_CHECK(boundary_triangles == sizes[1]);
  PHX_CHECK(std::abs(tetrahedra_volume6 - surface_volume6) <=
            1.0e-10L * std::max(1.0L, std::abs(tetrahedra_volume6)));
  PHX_CHECK(surface_volume6 > 0.0L);
  phx_mc_surface_envelope_free(handle);
}

}  // namespace

int main() {
  triangle_envelope();
  source_tetrahedron_envelope();
  return phx::mc::test::finish("test_surface_envelope");
}
