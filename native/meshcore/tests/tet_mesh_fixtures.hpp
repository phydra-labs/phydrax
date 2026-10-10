//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Box-domain fixtures for the constrained tetrahedral mesh tests: the exact
// Delaunay tetrahedralization of dyadic points in an axis-aligned box, its
// boundary faces labeled by box side, an optional interface plane x = split
// dividing two regions, and segments where constrained faces of different
// sources meet.  Independent oracles (exact planes, volumes, areas and
// dihedral angles) are computed from the exported arrays, not from the
// mesh's own flags.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "tet_mesh.hpp"

// Structural audit of the owned complex (reciprocal adjacency, opposite
// facet orientations, symmetric constraint marks, finite count).
inline bool handle_audit(const phx_mc_tet_mesh* mesh) { return mesh->mesh->complex().audit(); }

namespace phx::mc::test {

struct Domain {
  std::vector<double> points;
  std::vector<int32_t> tets;
  std::vector<int32_t> regions;
  std::vector<int32_t> faces;
  std::vector<int32_t> face_sources;
  std::vector<int32_t> segments;
  std::vector<int32_t> segment_sources;
};

// Original reconstruction cell whose exact positive orientation was lost in
// scalar arithmetic. Its immutable boundary cannot admit a local repair.
inline Domain reconstruction_sliver_domain() {
  Domain domain;
  domain.points = {
      -0x1.6997c8725ce81p-4, -0x1.607bcd6e0e171p-4, 0x1.0a53d3cbfd5e4p-3,
      -0x1.33f8fffb7559cp-4, -0x1.2add04f72688cp-4, 0x1.53eaa4520219cp-3,
      -0x1.8baf686bac825p-5, -0x1.1a993566d43abp-3, 0x1.1cd9fe6f328d9p-3,
      -0x1.08bd25a4ba5f7p-4, -0x1.f24cf95ec4574p-4, 0x1.0a53d3cbfd5e4p-3};
  domain.tets = {0, 1, 2, 3};
  domain.regions = {0};
  domain.faces = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  domain.face_sources = {0, 1, 2, 3};
  domain.segments = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
  domain.segment_sources = {0, 1, 2, 3, 4, 5};
  return domain;
}

// Closed original-coordinate reconstruction star. Both candidate diagonals
// contain an exact inverted child, so its 1--3 edge must not be removed.
inline Domain reconstruction_ring_domain() {
  Domain domain;
  domain.points = {
      -0x1.5b5058b95c69ap-4, -0x1.2d1f600a096a0p-3, -0x1.a1b2b96bae346p-6,
      -0x1.25ca4670d3f1ep-4, -0x1.6f9ed3cabd71bp-3, -0x1.9734e09318ab0p-7,
      -0x1.828ce27d414e4p-5, -0x1.3d5ce931a3cc5p-3, -0x1.4ffbfe6abc03fp-4,
      -0x1.e3cbb62294db2p-5, -0x1.2d1f600a096a0p-3, -0x1.1f5c9498123dap-4,
      -0x1.4196bdded7c50p-5, -0x1.89b7dfa29bbfep-3, -0x1.9c62cf087ec70p-6,
      -0x1.4196bdded7c50p-5, -0x1.549ee5ec10e46p-3, -0x1.217804f5e1d3ep-4};
  domain.tets = {0, 1, 3, 4, 0, 1, 2, 3, 1, 3, 4, 5, 1, 3, 5, 2};
  domain.regions = {0, 0, 0, 0};
  domain.faces = {0, 1, 2, 0, 1, 4, 0, 2, 3, 0, 3, 4,
                  1, 2, 5, 1, 4, 5, 2, 3, 5, 3, 4, 5};
  domain.face_sources = {0, 1, 2, 3, 4, 5, 6, 7};
  domain.segments = {0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 4,
                     1, 5, 2, 3, 2, 5, 3, 4, 3, 5, 4, 5};
  domain.segment_sources = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  return domain;
}

// Actual five-point cavity after its adjacent accepted reconnection. Removing
// 1--3 now gives two strictly positive cells and retains every boundary face.
inline Domain reconstruction_repair_domain() {
  Domain domain = reconstruction_ring_domain();
  domain.points.resize(15);
  domain.tets = {0, 1, 3, 4, 0, 1, 2, 3, 1, 3, 4, 2};
  domain.regions = {0, 0, 0};
  domain.faces = {0, 1, 2, 0, 1, 4, 0, 2, 3, 0, 3, 4, 1, 2, 4, 2, 3, 4};
  domain.face_sources = {132, 1000, 131, 1001, 157, 1002};
  domain.segments = {0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 4, 2, 3, 2, 4, 3, 4};
  domain.segment_sources = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  return domain;
}

// Original reconstruction star of the flat interior edge 0--1. Its cell
// (0, 2, 5, 1) spans the nearly coplanar source faces (0, 2, 5) and (1, 2, 5)
// at hinge 2--5: positive but far below the 1e-12 relative determinant floor.
// Every face, edge and ring reconnection keeps a chart over that plane.
inline Domain reconstruction_hinge_domain() {
  Domain domain;
  domain.points = {
      -0x1.f4a73d0757580p-7, 0x1.8d166f8c59a88p-4, -0x1.d2ad880871c04p-4,
      0x1.18d663e2f5544p-6, 0x1.2adff83dd8b10p-4, -0x1.2602ba7d5ea9cp-3,
      0x1.81cd8b7cc8a7cp-7, 0x1.2adff83dd8b10p-4, -0x1.20be108c7ae04p-3,
      0x1.b6e8970e23fb8p-7, 0x1.0144350784898p-3, -0x1.707710b9f0c88p-4,
      0x1.0f37a5cdc18eep-6, 0x1.046b0601f9166p-3, -0x1.707710b9f0c88p-4,
      0x1.093d6cc3a3decp-5, 0x1.c8053f1590a14p-4, -0x1.06ce2bc8d45c9p-3};
  domain.tets = {0, 2, 1, 3, 0, 2, 5, 1, 0, 3, 1, 4, 0, 4, 1, 5};
  domain.regions = {0, 0, 0, 0};
  domain.faces = {0, 2, 3, 0, 2, 5, 0, 3, 4, 0, 4, 5,
                  1, 2, 3, 1, 2, 5, 1, 3, 4, 1, 4, 5};
  domain.face_sources = {0, 1, 2, 3, 4, 5, 6, 7};
  domain.segments = {0, 2, 0, 3, 0, 4, 0, 5, 1, 2, 1, 3,
                     1, 4, 1, 5, 2, 3, 3, 4, 4, 5, 2, 5};
  domain.segment_sources = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  return domain;
}

// Deterministic dyadic sample: a lattice of `per_side` points per axis on
// [0, 1]^3 plus `interior` pseudo-random points with 1/1024 resolution.
inline std::vector<double> box_points(int per_side, int interior, std::uint64_t seed) {
  std::vector<double> points;
  for (int i = 0; i < per_side; ++i) {
    for (int j = 0; j < per_side; ++j) {
      for (int k = 0; k < per_side; ++k) {
        const double h = 1.0 / (per_side - 1);
        points.insert(points.end(), {i * h, j * h, k * h});
      }
    }
  }
  std::uint64_t state = seed;
  for (int n = 0; n < interior; ++n) {
    for (int axis = 0; axis < 3; ++axis) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      const auto cell = static_cast<int>((state >> 33) % 1022) + 1;
      points.push_back(cell / 1024.0);
    }
  }
  return points;
}

// Side source of a face on the unit box boundary (0..5), or the interface
// source 6 for a face on x = split, or -1.
inline int32_t face_side(const double* a, const double* b, const double* c, double split) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a[axis] == b[axis] && b[axis] == c[axis]) {
      if (a[axis] == 0.0) {
        return 2 * axis;
      }
      if (a[axis] == 1.0) {
        return 2 * axis + 1;
      }
      if (axis == 0 && a[axis] == split) {
        return 6;
      }
    }
  }
  return -1;
}

// Box domain from the Delaunay tetrahedralization; with split in (0, 1) the
// cells with centroid x above split form region 1 and the plane faces are
// constrained (the split must be a lattice plane).
inline Domain box_domain(const std::vector<double>& points, double split) {
  Domain domain;
  domain.points = points;
  phx_mc_mesh* mesh = nullptr;
  const auto count = static_cast<int64_t>(points.size() / 3);
  if (phx_mc_delaunay_3d(count, points.data(), 1 << 22, &mesh) != PHX_MC_OK) {
    return domain;
  }
  domain.tets.assign(mesh->cells.begin(), mesh->cells.end());
  phx_mc_mesh_free(mesh);
  const std::size_t cells = domain.tets.size() / 4;
  std::map<std::array<int32_t, 3>, std::array<int32_t, 4>> faces;
  for (std::size_t t = 0; t < cells; ++t) {
    const int32_t* v = domain.tets.data() + 4 * t;
    double centroid = 0.0;
    for (int k = 0; k < 4; ++k) {
      centroid += points[3 * static_cast<std::size_t>(v[k])] / 4.0;
    }
    const int32_t region = split > 0.0 && centroid > split ? 1 : 0;
    domain.regions.push_back(region);
    for (int k = 0; k < 4; ++k) {
      std::array<int32_t, 3> key{};
      int r = 0;
      for (int j = 0; j < 4; ++j) {
        if (j != k) {
          key[static_cast<std::size_t>(r++)] = v[j];
        }
      }
      std::sort(key.begin(), key.end());
      auto& entry = faces[key];
      ++entry[0];
      entry[1 + std::min(entry[0] - 1, 1)] = region;
    }
  }
  std::map<std::array<int32_t, 2>, std::vector<int32_t>> edges;
  for (const auto& [key, entry] : faces) {
    const double* a = points.data() + 3 * key[0];
    const double* b = points.data() + 3 * key[1];
    const double* c = points.data() + 3 * key[2];
    const bool boundary = entry[0] == 1;
    const bool interface = entry[0] == 2 && entry[1] != entry[2];
    if (!boundary && !interface) {
      continue;
    }
    const int32_t source = face_side(a, b, c, split);
    domain.faces.insert(domain.faces.end(), key.begin(), key.end());
    domain.face_sources.push_back(source);
    for (int r = 0; r < 3; ++r) {
      std::array<int32_t, 2> edge{key[static_cast<std::size_t>(r)],
                                  key[static_cast<std::size_t>((r + 1) % 3)]};
      std::sort(edge.begin(), edge.end());
      edges[edge].push_back(source);
    }
  }
  int32_t next = 0;
  for (const auto& [edge, sources] : edges) {
    if (sources.size() != 2 || sources[0] != sources[1]) {
      domain.segments.insert(domain.segments.end(), edge.begin(), edge.end());
      domain.segment_sources.push_back(next++);
    }
  }
  return domain;
}

inline phx_mc_tet_mesh* create_mesh(const Domain& domain, int32_t policy,
                                    const double* protection = nullptr,
                                    int64_t max_vertices = 1 << 20,
                                    int64_t max_tetrahedra = 1 << 23, int32_t* status = nullptr) {
  phx_mc_tet_mesh* mesh = nullptr;
  const int32_t result = phx_mc_tet_mesh_create(
      static_cast<int64_t>(domain.points.size() / 3), domain.points.data(),
      static_cast<int64_t>(domain.regions.size()), domain.tets.data(), domain.regions.data(),
      static_cast<int64_t>(domain.face_sources.size()), domain.faces.data(),
      domain.face_sources.data(), static_cast<int64_t>(domain.segment_sources.size()),
      domain.segments.data(), domain.segment_sources.data(), protection, 0, nullptr,
      0, nullptr, nullptr,
      nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, policy, max_vertices,
      max_tetrahedra, INT64_MAX, &mesh);
  if (status != nullptr) {
    *status = result;
  }
  return mesh;
}

struct Exported {
  std::vector<double> points;
  std::vector<int32_t> tets;
  std::vector<int32_t> regions;
  std::vector<int32_t> faces;
  std::vector<int32_t> face_sources;
  std::vector<int32_t> segments;
  std::vector<int32_t> segment_sources;
  std::vector<int8_t> dimension;
  std::vector<double> sizes;

  bool operator==(const Exported& other) const {
    return points == other.points && tets == other.tets && regions == other.regions &&
           faces == other.faces && face_sources == other.face_sources &&
           segments == other.segments && segment_sources == other.segment_sources &&
           dimension == other.dimension;
  }
};

inline Exported export_mesh(const phx_mc_tet_mesh* mesh) {
  int64_t counts[PHX_MC_TET_MESH_COUNTS];
  phx_mc_tet_mesh_counts(mesh, counts);
  Exported out;
  const auto n = static_cast<std::size_t>(counts[0]);
  out.points.resize(3 * n);
  out.dimension.resize(n);
  out.sizes.resize(n);
  out.tets.resize(4 * static_cast<std::size_t>(counts[1]));
  out.regions.resize(static_cast<std::size_t>(counts[1]));
  out.faces.resize(3 * static_cast<std::size_t>(counts[2]));
  out.face_sources.resize(static_cast<std::size_t>(counts[2]));
  out.segments.resize(2 * static_cast<std::size_t>(counts[3]));
  out.segment_sources.resize(static_cast<std::size_t>(counts[3]));
  phx_mc_tet_mesh_export(mesh, out.points.data(), out.tets.data(), out.regions.data(),
                         out.faces.data(), out.face_sources.data(), out.segments.data(),
                         out.segment_sources.data(), out.dimension.data(), out.sizes.data());
  return out;
}

inline const double* at(const Exported& mesh, int32_t v) {
  return mesh.points.data() + 3 * static_cast<std::size_t>(v);
}

inline double tet_volume(const Exported& mesh, std::size_t t) {
  const int32_t* v = mesh.tets.data() + 4 * t;
  const double* a = at(mesh, v[0]);
  double u[3], w[3], z[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = at(mesh, v[1])[i] - a[i];
    w[i] = at(mesh, v[2])[i] - a[i];
    z[i] = at(mesh, v[3])[i] - a[i];
  }
  return (u[0] * (w[1] * z[2] - w[2] * z[1]) - u[1] * (w[0] * z[2] - w[2] * z[0]) +
          u[2] * (w[0] * z[1] - w[1] * z[0])) /
         6.0;
}

inline double region_volume(const Exported& mesh, int32_t region) {
  double total = 0.0;
  for (std::size_t t = 0; t < mesh.regions.size(); ++t) {
    total += mesh.regions[t] == region ? tet_volume(mesh, t) : 0.0;
  }
  return total;
}

inline double face_area(const Exported& mesh, std::size_t f) {
  const int32_t* v = mesh.faces.data() + 3 * f;
  double u[3], w[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = at(mesh, v[1])[i] - at(mesh, v[0])[i];
    w[i] = at(mesh, v[2])[i] - at(mesh, v[0])[i];
  }
  const double n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                       u[0] * w[1] - u[1] * w[0]};
  return 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

// Whether every exported face lies exactly on the plane of its source side
// (all three vertices share the side coordinate) and inside the box.
inline bool faces_on_sides(const Exported& mesh, double split) {
  for (std::size_t f = 0; f < mesh.face_sources.size(); ++f) {
    const int32_t* v = mesh.faces.data() + 3 * f;
    const int32_t side = face_side(at(mesh, v[0]), at(mesh, v[1]), at(mesh, v[2]), split);
    if (side != mesh.face_sources[f]) {
      return false;
    }
  }
  return true;
}

inline double source_area(const Exported& mesh, int32_t source) {
  double total = 0.0;
  for (std::size_t f = 0; f < mesh.face_sources.size(); ++f) {
    total += mesh.face_sources[f] == source ? face_area(mesh, f) : 0.0;
  }
  return total;
}

// Exact positive orientation of every exported cell.
inline bool all_positive(const Exported& mesh) {
  for (std::size_t t = 0; t < mesh.regions.size(); ++t) {
    const int32_t* v = mesh.tets.data() + 4 * t;
    if (orient3d(at(mesh, v[0]), at(mesh, v[1]), at(mesh, v[2]), at(mesh, v[3])) <= 0) {
      return false;
    }
  }
  return true;
}

// Independent radius-edge ratio and smallest dihedral angle (degrees).
inline void cell_quality(const Exported& mesh, std::size_t t, double& ratio, double& dihedral) {
  const int32_t* v = mesh.tets.data() + 4 * t;
  const double* p[4] = {at(mesh, v[0]), at(mesh, v[1]), at(mesh, v[2]), at(mesh, v[3])};
  // Circumcenter from the 3x3 system 2 (p_i - p_0) . c' = |p_i - p_0|^2.
  double m[3][3];
  double r[3];
  for (int i = 0; i < 3; ++i) {
    double norm = 0.0;
    for (int j = 0; j < 3; ++j) {
      m[i][j] = 2.0 * (p[i + 1][j] - p[0][j]);
      norm += (p[i + 1][j] - p[0][j]) * (p[i + 1][j] - p[0][j]);
    }
    r[i] = norm;
  }
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                     m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  double c[3];
  for (int col = 0; col < 3; ++col) {
    double a[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        a[i][j] = j == col ? r[i] : m[i][j];
      }
    }
    c[col] = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
              a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
              a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])) /
             det;
  }
  const double radius = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  double shortest = 1e300;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      double d = 0.0;
      for (int k = 0; k < 3; ++k) {
        d += (p[i][k] - p[j][k]) * (p[i][k] - p[j][k]);
      }
      shortest = std::min(shortest, std::sqrt(d));
    }
  }
  ratio = radius / shortest;
  // Dihedral angles from outward face normals: pi minus the normal angle.
  static constexpr int kFaces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  double normals[4][3];
  for (int f = 0; f < 4; ++f) {
    const double* a = p[kFaces[f][0]];
    const double* b = p[kFaces[f][1]];
    const double* e = p[kFaces[f][2]];
    double u[3], w[3];
    for (int i = 0; i < 3; ++i) {
      u[i] = b[i] - a[i];
      w[i] = e[i] - a[i];
    }
    const double n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                         u[0] * w[1] - u[1] * w[0]};
    const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int i = 0; i < 3; ++i) {
      normals[f][i] = n[i] / length;
    }
  }
  dihedral = 180.0;
  for (int f = 0; f < 4; ++f) {
    for (int g = f + 1; g < 4; ++g) {
      const double cosine = normals[f][0] * normals[g][0] + normals[f][1] * normals[g][1] +
                            normals[f][2] * normals[g][2];
      const double angle = 180.0 - std::acos(std::clamp(cosine, -1.0, 1.0)) * 57.29577951308232;
      dihedral = std::min(dihedral, angle);
    }
  }
}

}  // namespace phx::mc::test
