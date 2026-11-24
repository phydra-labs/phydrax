//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Hex-dominant meshing against independent references: every cell closed
// (directed edges paired) and of positive decomposition volume, hexahedron
// corner Jacobians recomputed, the mesh volume against the enclosed volume,
// wall vertices on the input surface by brute-force distance; on a box every
// wall vertex lies on a face plane (closed form), the eight corners are mesh
// vertices and the volume is exact; layer vertices at their exact cumulative
// thickness along the column; invalid surfaces and sizes are refused.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

struct Surface {
  std::vector<double> x;
  std::vector<int32_t> f;
};

Surface icosphere(int levels) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  Surface s;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0},  {-1, -t, 0}, {1, -t, 0},
                              {0, -1, t}, {0, 1, t},  {0, -1, -t}, {0, 1, -t},
                              {t, 0, -1}, {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
  for (const auto& p : base) {
    const double l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    s.x.insert(s.x.end(), {p[0] / l, p[1] / l, p[2] / l});
  }
  s.f = {0, 11, 5, 0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
         4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9,  4,  3,  4,  2, 3, 2, 6, 3,
         6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6, 2,  10, 8,  6, 7, 9, 8, 1};
  for (int level = 0; level < levels; ++level) {
    std::map<std::pair<int32_t, int32_t>, int32_t> mid;
    auto middle = [&](int32_t a, int32_t b) {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      const auto it = mid.find(key);
      if (it != mid.end()) return it->second;
      double p[3];
      for (int k = 0; k < 3; ++k) p[k] = s.x[3 * a + k] + s.x[3 * b + k];
      const double l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const int32_t id = static_cast<int32_t>(s.x.size() / 3);
      s.x.insert(s.x.end(), {p[0] / l, p[1] / l, p[2] / l});
      mid.emplace(key, id);
      return id;
    };
    std::vector<int32_t> next;
    for (size_t i = 0; i < s.f.size(); i += 3) {
      const int32_t a = s.f[i], b = s.f[i + 1], c = s.f[i + 2];
      const int32_t ab = middle(a, b), bc = middle(b, c), ca = middle(c, a);
      next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    s.f = next;
  }
  return s;
}

// Box [0, l]^3 with n x n quads per face, outward.
Surface box(double l, int n) {
  Surface s;
  std::map<std::array<int, 3>, int32_t> ids;
  auto vertex = [&](std::array<int, 3> g) {
    const auto it = ids.find(g);
    if (it != ids.end()) return it->second;
    const int32_t id = static_cast<int32_t>(s.x.size() / 3);
    for (int a = 0; a < 3; ++a) s.x.push_back(l * g[a] / n);
    ids.emplace(g, id);
    return id;
  };
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    for (int side = 0; side <= n; side += n) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          std::array<int32_t, 4> q;
          const int di[4] = {0, 1, 1, 0}, dj[4] = {0, 0, 1, 1};
          for (int c = 0; c < 4; ++c) {
            std::array<int, 3> g{};
            g[axis] = side;
            g[u] = i + di[c];
            g[v] = j + dj[c];
            q[c] = vertex(g);
          }
          if (side == n) {
            s.f.insert(s.f.end(), {q[0], q[1], q[2], q[0], q[2], q[3]});
          } else {
            s.f.insert(s.f.end(), {q[0], q[2], q[1], q[0], q[3], q[2]});
          }
        }
      }
    }
  }
  return s;
}

// Closed cylinder of radius r over [0, h + slope x] (n segments, m rings,
// caps fanned to their centers), outward: two closed convex feature curves,
// the top one elliptic when the cap is oblique.
Surface cylinder(double radius, double height, int n, int m, double slope = 0.0) {
  Surface s;
  const double pi = 3.14159265358979323846;
  for (int k = 0; k <= m; ++k) {
    for (int i = 0; i < n; ++i) {
      const double a = 2.0 * pi * i / n;
      const double x = radius * std::cos(a);
      s.x.insert(s.x.end(), {x, radius * std::sin(a), (height + slope * x) * k / m});
    }
  }
  auto ring = [&](int k, int i) { return static_cast<int32_t>(k * n + (i % n)); };
  for (int k = 0; k < m; ++k) {
    for (int i = 0; i < n; ++i) {
      s.f.insert(s.f.end(), {ring(k, i), ring(k, i + 1), ring(k + 1, i + 1), ring(k, i), ring(k + 1, i + 1), ring(k + 1, i)});
    }
  }
  const int32_t bottom = static_cast<int32_t>(s.x.size() / 3);
  s.x.insert(s.x.end(), {0.0, 0.0, 0.0});
  const int32_t top = bottom + 1;
  s.x.insert(s.x.end(), {0.0, 0.0, height});
  for (int i = 0; i < n; ++i) {
    s.f.insert(s.f.end(), {bottom, ring(0, i + 1), ring(0, i)});
    s.f.insert(s.f.end(), {top, ring(m, i), ring(m, i + 1)});
  }
  return s;
}

struct Result {
  int32_t status = -1;
  std::vector<double> points;
  std::vector<int64_t> face_offsets;
  std::vector<int32_t> face_vertices, face_owner, face_neighbor, face_facets;
  std::vector<uint8_t> cell_kinds, cell_roles, vertex_kinds;
  std::vector<int64_t> cell_vertex_offsets;
  std::vector<int32_t> cell_vertices, vertex_triangles;
  std::vector<double> stats = std::vector<double>(28, 0.0);
};

Result mesh(const Surface& s, double core, double boundary, const std::vector<double>& layers,
            int32_t prove = 1, int64_t* failed_cells = nullptr, int32_t parallelism = 1,
            double layer_floor = 0.5) {
  Result r;
  const std::vector<int32_t> facets(s.f.size() / 3, 0);
  phx_mc_hex_dominant_mesh* handle = nullptr;
  const phx_mc_hex_dominant_request request{static_cast<int64_t>(s.x.size() / 3),
                                            s.x.data(),
                                            static_cast<int64_t>(s.f.size() / 3),
                                            s.f.data(),
                                            facets.data(),
                                            core,
                                            boundary,
                                            45.0,
                                            static_cast<int32_t>(layers.size()),
                                            layers.data(),
                                            layer_floor,
                                            prove,
                                            &handle};
  r.status = phx_mc_hex_dominant_build(&request, parallelism);
  if (handle == nullptr) return r;
  if (failed_cells != nullptr) {
    PHX_CHECK(phx_mc_hex_dominant_certify(handle, failed_cells) == PHX_MC_OK);
  }
  int64_t sizes[5];
  PHX_CHECK(phx_mc_hex_dominant_sizes(handle, sizes) == PHX_MC_OK);
  r.points.resize(3 * sizes[0]);
  r.face_offsets.resize(sizes[1] + 1);
  r.face_vertices.resize(sizes[2]);
  r.face_owner.resize(sizes[1]);
  r.face_neighbor.resize(sizes[1]);
  r.face_facets.resize(sizes[1]);
  r.cell_kinds.resize(sizes[3]);
  r.cell_roles.resize(sizes[3]);
  r.cell_vertex_offsets.resize(sizes[3] + 1);
  r.cell_vertices.resize(sizes[4]);
  r.vertex_triangles.resize(sizes[0]);
  r.vertex_kinds.resize(sizes[0]);
  const phx_mc_hex_dominant_arrays arrays{
      r.points.data(),       r.face_offsets.data(),        r.face_vertices.data(),
      r.face_owner.data(),   r.face_neighbor.data(),       r.face_facets.data(),
      r.cell_kinds.data(),   r.cell_roles.data(),          r.cell_vertex_offsets.data(),
      r.cell_vertices.data(), r.vertex_triangles.data(),   r.vertex_kinds.data(),
      r.stats.data()};
  PHX_CHECK(phx_mc_hex_dominant_export(handle, &arrays) == PHX_MC_OK);
  phx_mc_hex_dominant_free(handle);
  return r;
}

using V = std::array<double, 3>;
V at(const std::vector<double>& x, int64_t v) { return {x[3 * v], x[3 * v + 1], x[3 * v + 2]}; }
V minus(const V& a, const V& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot3(const V& a, const V& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V cross3(const V& a, const V& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Independent checks: closed cells (every directed edge paired), positive
// decomposition tetrahedra (center, face vertex average, edge) about the
// distinct-vertex average (the published local map) and about the OpenFOAM
// centroid, positive hexahedron corner Jacobians; returns the mesh volume.
double check_cells(const Result& r) {
  const int64_t cells = static_cast<int64_t>(r.cell_kinds.size());
  const int64_t faces = static_cast<int64_t>(r.face_owner.size());
  std::vector<std::vector<std::pair<int64_t, int>>> of(cells);
  for (int64_t f = 0; f < faces; ++f) {
    of[r.face_owner[f]].push_back({f, 1});
    if (r.face_neighbor[f] >= 0) of[r.face_neighbor[f]].push_back({f, -1});
  }
  double total = 0.0;
  int64_t open = 0, negative = 0, inverted = 0;
  for (int64_t c = 0; c < cells; ++c) {
    std::vector<std::pair<int32_t, int32_t>> directed;
    // Distinct-vertex average; face vertex averages, centers and areas.
    std::vector<int32_t> vertices;
    std::vector<V> averages, centers, areas;
    for (const auto& [f, sign] : of[c]) {
      const int64_t first = r.face_offsets[f], last = r.face_offsets[f + 1];
      V average{0, 0, 0};
      for (int64_t q = first; q < last; ++q) {
        vertices.push_back(r.face_vertices[q]);
        const V p = at(r.points, r.face_vertices[q]);
        for (int a = 0; a < 3; ++a) average[a] += p[a] / static_cast<double>(last - first);
      }
      V center{0, 0, 0}, area{0, 0, 0};
      double weight = 0.0;
      for (int64_t q = first; q < last; ++q) {
        const V p = at(r.points, r.face_vertices[q]);
        const V n = at(r.points, r.face_vertices[q + 1 < last ? q + 1 : first]);
        const V a = cross3(minus(p, average), minus(n, average));
        const double m = std::sqrt(dot3(a, a));
        for (int k = 0; k < 3; ++k) {
          area[k] += 0.5 * sign * a[k];
          center[k] += m * (p[k] + n[k] + average[k]) / 3.0;
        }
        weight += m;
      }
      for (int k = 0; k < 3; ++k) center[k] /= weight;
      averages.push_back(average);
      centers.push_back(center);
      areas.push_back(area);
    }
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
    V vertex_average{0, 0, 0};
    for (const int32_t v : vertices) {
      const V p = at(r.points, v);
      for (int a = 0; a < 3; ++a) vertex_average[a] += p[a] / static_cast<double>(vertices.size());
    }
    // OpenFOAM centroid: pyramids about the mean of the face centers.
    V estimate{0, 0, 0};
    for (const V& center : centers) {
      for (int a = 0; a < 3; ++a) estimate[a] += center[a] / static_cast<double>(centers.size());
    }
    V centroid{0, 0, 0};
    double pyramids = 0.0;
    for (size_t i = 0; i < centers.size(); ++i) {
      const double v = dot3(areas[i], minus(centers[i], estimate)) / 3.0;
      pyramids += v;
      for (int a = 0; a < 3; ++a) centroid[a] += v * (0.75 * centers[i][a] + 0.25 * estimate[a]);
    }
    for (int a = 0; a < 3; ++a) centroid[a] /= pyramids;
    double volume = 0.0;
    for (size_t i = 0; i < of[c].size(); ++i) {
      const auto& [f, sign] = of[c][i];
      const V& average = averages[i];
      const int64_t first = r.face_offsets[f], last = r.face_offsets[f + 1];
      for (int64_t q = first; q < last; ++q) {
        const int64_t n = q + 1 < last ? q + 1 : first;
        int32_t u = r.face_vertices[q], w = r.face_vertices[n];
        if (sign < 0) std::swap(u, w);
        directed.push_back({u, w});
        const V fan = cross3(minus(at(r.points, u), average), minus(at(r.points, w), average));
        const double tet = dot3(fan, minus(average, vertex_average)) / 6.0;
        if (!(tet > 0.0) || !(dot3(fan, minus(average, centroid)) > 0.0)) ++negative;
        volume += tet;
      }
    }
    std::sort(directed.begin(), directed.end());
    for (const auto& e : directed) {
      if (!std::binary_search(directed.begin(), directed.end(), std::make_pair(e.second, e.first))) ++open;
    }
    total += volume;
    if (r.cell_kinds[c] == 0) {
      const int32_t* h = r.cell_vertices.data() + r.cell_vertex_offsets[c];
      const int corner[8][3] = {{1, 3, 4}, {2, 0, 5}, {3, 1, 6}, {0, 2, 7},
                                {7, 5, 0}, {4, 6, 1}, {5, 7, 2}, {6, 4, 3}};
      for (int i = 0; i < 8; ++i) {
        const V p = at(r.points, h[i]);
        const double j = dot3(minus(at(r.points, h[corner[i][0]]), p),
                              cross3(minus(at(r.points, h[corner[i][1]]), p), minus(at(r.points, h[corner[i][2]]), p)));
        if (!(j > 0.0)) ++inverted;
      }
    }
  }
  PHX_CHECK(open == 0);
  PHX_CHECK(negative == 0);
  PHX_CHECK(inverted == 0);
  return total;
}

double point_triangle_distance(const V& p, const V& a, const V& b, const V& c) {
  // Minimum over the plane projection (when inside) and the three edges.
  const V n = cross3(minus(b, a), minus(c, a));
  const double nn = dot3(n, n);
  const double t = dot3(minus(p, a), n) / nn;
  const V q = {p[0] - t * n[0], p[1] - t * n[1], p[2] - t * n[2]};
  const bool inside = dot3(cross3(minus(b, a), minus(q, a)), n) >= 0 &&
                      dot3(cross3(minus(c, b), minus(q, b)), n) >= 0 &&
                      dot3(cross3(minus(a, c), minus(q, c)), n) >= 0;
  if (inside) return std::fabs(t) * std::sqrt(nn);
  double best = 1e300;
  const V ends[3][2] = {{a, b}, {b, c}, {c, a}};
  for (const auto& e : ends) {
    const V d = minus(e[1], e[0]);
    const double s = std::clamp(dot3(minus(p, e[0]), d) / dot3(d, d), 0.0, 1.0);
    const V r = {e[0][0] + s * d[0] - p[0], e[0][1] + s * d[1] - p[1], e[0][2] + s * d[2] - p[2]};
    best = std::min(best, std::sqrt(dot3(r, r)));
  }
  return best;
}

void sphere_is_certified_and_on_the_surface() {
  const Surface s = icosphere(4);
  const Result r = mesh(s, 0.25, 0.125, {});
  PHX_CHECK(r.status == PHX_MC_OK);
  PHX_CHECK(r.stats[22] == 1.0);
  PHX_CHECK(r.stats[25] == 1.0);
  const double volume = check_cells(r);
  double enclosed = 0.0;
  for (size_t i = 0; i < s.f.size(); i += 3) {
    enclosed += dot3(at(s.x, s.f[i]), cross3(at(s.x, s.f[i + 1]), at(s.x, s.f[i + 2]))) / 6.0;
  }
  PHX_CHECK(std::fabs(volume / enclosed - 1.0) < 0.01);
  // Wall vertices on the input surface (brute force over its triangles).
  double worst = 0.0;
  int64_t wall = 0;
  for (size_t v = 0; v < r.vertex_kinds.size(); ++v) {
    if (r.vertex_kinds[v] == 0 || r.vertex_kinds[v] == 4) continue;
    ++wall;
    double best = 1e300;
    for (size_t i = 0; i < s.f.size(); i += 3) {
      best = std::min(best, point_triangle_distance(at(r.points, static_cast<int64_t>(v)), at(s.x, s.f[i]),
                                                    at(s.x, s.f[i + 1]), at(s.x, s.f[i + 2])));
    }
    worst = std::max(worst, best);
    PHX_CHECK(r.vertex_triangles[v] >= 0);
  }
  PHX_CHECK(wall > 0);
  PHX_CHECK(worst < 1e-12);
  // Hexahedra dominate; the polyhedra are the 2:1 transitions.
  PHX_CHECK(r.stats[8] > 0.8 * r.stats[7]);
}

void box_corners_and_volume_are_exact() {
  const Surface s = box(1.0, 8);
  const Result r = mesh(s, 0.25, 0.125, {});
  PHX_CHECK(r.status == PHX_MC_OK);
  const double volume = check_cells(r);
  PHX_CHECK(std::fabs(volume - 1.0) < 1e-9);
  double worst = 0.0;
  for (size_t v = 0; v < r.vertex_kinds.size(); ++v) {
    if (r.vertex_kinds[v] == 0 || r.vertex_kinds[v] == 4) continue;
    double d = 1e300;
    for (int a = 0; a < 3; ++a) {
      d = std::min({d, std::fabs(r.points[3 * v + a]), std::fabs(1.0 - r.points[3 * v + a])});
    }
    worst = std::max(worst, d);
  }
  PHX_CHECK(worst < 1e-14);
  int corners = 0;
  for (int c = 0; c < 8; ++c) {
    const V target = {static_cast<double>(c & 1), static_cast<double>((c >> 1) & 1),
                      static_cast<double>((c >> 2) & 1)};
    for (size_t v = 0; v < r.vertex_kinds.size(); ++v) {
      if (r.vertex_kinds[v] == 3 && r.points[3 * v] == target[0] && r.points[3 * v + 1] == target[1] &&
          r.points[3 * v + 2] == target[2]) {
        ++corners;
        break;
      }
    }
  }
  PHX_CHECK(corners == 8);
}

void layers_sit_at_their_thicknesses() {
  const Surface s = icosphere(4);
  const std::vector<double> layers = {0.01, 0.012};
  const Result r = mesh(s, 0.25, 0.125, layers);
  PHX_CHECK(r.status == PHX_MC_OK);
  check_cells(r);
  // Layer cells: two per wall face; their wall-side top vertices at 0 and
  // 0.01 along the column, the second at 0.022.
  int64_t checked = 0;
  for (size_t c = 0; c < r.cell_kinds.size(); ++c) {
    if (r.cell_roles[c] != 2) continue;
    const int32_t* h = r.cell_vertices.data() + r.cell_vertex_offsets[c];
    for (int k = 0; k < 4; ++k) {
      const V inner = at(r.points, h[k]), outer = at(r.points, h[k + 4]);
      const double gap = std::sqrt(dot3(minus(inner, outer), minus(inner, outer)));
      const bool first = r.vertex_kinds[h[k + 4]] != 4;
      PHX_CHECK(std::fabs(gap - (first ? 0.01 : 0.012)) < 1e-12);
      ++checked;
    }
  }
  PHX_CHECK(checked > 0);
}

// The least layer-cell thickness over the requested one, measured as the
// compliance report does: the mean height of a cell's upper corners over the
// plane of its wall-side corners. Layer cells follow the core, levels + 1
// cells (the last a projection cell) per wall piece.
double least_layer_ratio(const Result& r, const std::vector<double>& layers) {
  int64_t core = 0;
  for (const uint8_t role : r.cell_roles) core += role == 0;
  double least = std::numeric_limits<double>::infinity();
  for (size_t c = 0; c < r.cell_kinds.size(); ++c) {
    const int64_t first = r.cell_vertex_offsets[c], count = r.cell_vertex_offsets[c + 1] - first;
    if (r.cell_roles[c] != 2 || (count != 6 && count != 8)) continue;
    const int64_t half = count / 2;
    const int32_t* h = r.cell_vertices.data() + first;
    V normal{0, 0, 0};
    for (int64_t i = 0; i < half; ++i) {
      const V n = cross3(at(r.points, h[half + i]), at(r.points, h[half + (i + 1) % half]));
      for (int a = 0; a < 3; ++a) normal[a] += n[a];
    }
    const double size = std::sqrt(dot3(normal, normal));
    double height = 0.0;
    for (int64_t i = 0; i < half; ++i) height += dot3(minus(at(r.points, h[i]), at(r.points, h[half + i])), normal);
    const auto level = static_cast<size_t>((static_cast<int64_t>(c) - core) % static_cast<int64_t>(layers.size() + 1));
    least = std::min(least, std::fabs(height) / size / static_cast<double>(half) / layers[level]);
  }
  return least;
}

// Smoothing tilts layer columns to repair skewed faces between layer cells.
// On this sphere the default floor admits tilts that thin a layer cell below
// 0.8 of its thickness; a 0.9 floor keeps every layer cell at 0.9 or more.
void layer_tilts_keep_the_requested_floor() {
  const Surface s = icosphere(4);
  const std::vector<double> layers = {0.01, 0.012};
  const Result loose = mesh(s, 0.25, 0.125, layers, 1, nullptr, 1, 0.5);
  const Result tight = mesh(s, 0.25, 0.125, layers, 1, nullptr, 1, 0.9);
  PHX_CHECK(loose.status == PHX_MC_OK && tight.status == PHX_MC_OK);
  check_cells(tight);
  const double thinnest = least_layer_ratio(loose, layers);
  PHX_CHECK(thinnest >= 0.5 && thinnest < 0.9);
  PHX_CHECK(least_layer_ratio(tight, layers) >= 0.9);
}

// Every cell is star-shaped about its distinct-vertex average (the published
// local map) as well as about its OpenFOAM centroid: on this cylinder at
// h = 0.09 a mesher screening only the centroid star published 2:1
// transition polyhedra with a negative vertex-average tetrahedron.
void cells_are_stars_about_their_vertex_average() {
  const Result r = mesh(cylinder(0.5, 0.8, 96, 16), 0.09, 0.045, {});
  PHX_CHECK(r.status == PHX_MC_OK);
  check_cells(r);
}

// At a convex feature curve the layers do not wrap the curve: corner
// hexahedra own one wall strip per patch, every cell is certified and every
// wall vertex lies on the surface.
void convex_curve_layers_split() {
  const Surface s = cylinder(0.5, 0.8, 96, 16);
  const Result r = mesh(s, 0.16, 0.08, {0.01, 0.012});
  PHX_CHECK(r.status == PHX_MC_OK);
  check_cells(r);
  std::vector<int> walls(r.cell_kinds.size(), 0);
  for (size_t f = 0; f < r.face_owner.size(); ++f) walls[r.face_owner[f]] += r.face_neighbor[f] < 0;
  int64_t corners = 0;
  for (size_t c = 0; c < walls.size(); ++c) {
    if (walls[c] < 2) continue;
    PHX_CHECK(r.cell_kinds[c] == 0 && r.cell_roles[c] == 2);
    ++corners;
  }
  PHX_CHECK(corners > 0);
  double worst = 0.0;
  for (size_t v = 0; v < r.vertex_kinds.size(); ++v) {
    if (r.vertex_kinds[v] == 0 || r.vertex_kinds[v] == 4) continue;
    double best = 1e300;
    for (size_t i = 0; i < s.f.size(); i += 3) {
      best = std::min(best, point_triangle_distance(at(r.points, static_cast<int64_t>(v)), at(s.x, s.f[i]),
                                                    at(s.x, s.f[i + 1]), at(s.x, s.f[i + 2])));
    }
    worst = std::max(worst, best);
  }
  PHX_CHECK(worst < 1e-12);
}

// Layer cells that hold a curve-segment midpoint: polyhedra outside the core.
int64_t midpoint_cells(const Result& r) {
  int64_t count = 0;
  for (size_t c = 0; c < r.cell_kinds.size(); ++c) count += r.cell_roles[c] != 0 && r.cell_kinds[c] == 1;
  return count;
}

// An oblique top cap (elliptic rim, like the F8 quadric's curves): curve
// segments whose chord deviates from the rim more than any patch wall edge
// from its patch gain a midpoint; every cell is valid and every curve
// vertex, midpoints included, lies on the rim polyline.
void oblique_curve_segments_gain_midpoints() {
  const int n = 96, m = 16;
  const Surface s = cylinder(0.5, 0.8, n, m, 0.6);
  const Result r = mesh(s, 0.16, 0.08, {0.01, 0.012});
  PHX_CHECK(r.status == PHX_MC_OK);
  check_cells(r);
  PHX_CHECK(midpoint_cells(r) > 0);
  double worst = 0.0;
  int64_t curve = 0;
  for (size_t v = 0; v < r.vertex_kinds.size(); ++v) {
    if (r.vertex_kinds[v] != 2) continue;
    ++curve;
    const V p = at(r.points, static_cast<int64_t>(v));
    double best = 1e300;
    for (const int k : {0, m}) {
      for (int i = 0; i < n; ++i) {
        const V a = at(s.x, k * n + i), b = at(s.x, k * n + (i + 1) % n);
        const V d = minus(b, a);
        const double t = std::clamp(dot3(minus(p, a), d) / dot3(d, d), 0.0, 1.0);
        const V e = {a[0] + t * d[0] - p[0], a[1] + t * d[1] - p[1], a[2] + t * d[2] - p[2]};
        best = std::min(best, std::sqrt(dot3(e, e)));
      }
    }
    worst = std::max(worst, best);
  }
  PHX_CHECK(curve > 0);
  PHX_CHECK(worst < 1e-12);
}

// The split and segment-midpoint decisions are scale-free: the oblique
// fixture scaled by a power of two with sizes and layers scaled alike
// (every floating operation then scales exactly) gives the same mesh,
// scaled, midpoints included.
void convex_curve_split_is_scale_free() {
  const Surface s = cylinder(0.5, 0.8, 96, 16, 0.6);
  Surface big = s;
  for (double& x : big.x) x *= 8.0;
  const Result r = mesh(s, 0.16, 0.08, {0.01, 0.012});
  const Result q = mesh(big, 1.28, 0.64, {0.08, 0.096});
  PHX_CHECK(r.status == PHX_MC_OK && q.status == PHX_MC_OK);
  PHX_CHECK(midpoint_cells(r) > 0);
  PHX_CHECK(r.face_offsets == q.face_offsets && r.face_vertices == q.face_vertices);
  PHX_CHECK(r.face_owner == q.face_owner && r.face_neighbor == q.face_neighbor && r.cell_kinds == q.cell_kinds);
  PHX_CHECK(r.points.size() == q.points.size());
  bool scaled = r.points.size() == q.points.size();
  for (size_t i = 0; i < r.points.size() && scaled; ++i) scaled = q.points[i] == 8.0 * r.points[i];
  PHX_CHECK(scaled);
}

void deferred_proof_preserves_the_layered_mesh() {
  const Surface source = icosphere(4);
  const Result eager = mesh(source, 0.25, 0.125, {0.01, 0.012});
  int64_t failed_cells = -1;
  const Result deferred = mesh(source, 0.25, 0.125, {0.01, 0.012}, 0, &failed_cells, 4);
  PHX_CHECK(eager.status == PHX_MC_OK && deferred.status == PHX_MC_OK);
  PHX_CHECK(failed_cells == 0);
  PHX_CHECK(eager.points == deferred.points);
  PHX_CHECK(eager.face_offsets == deferred.face_offsets);
  PHX_CHECK(eager.face_vertices == deferred.face_vertices);
  PHX_CHECK(eager.face_owner == deferred.face_owner);
  PHX_CHECK(eager.face_neighbor == deferred.face_neighbor);
  PHX_CHECK(eager.cell_roles == deferred.cell_roles);
}

void invalid_input_is_refused() {
  Surface open = icosphere(2);
  open.f.resize(open.f.size() - 3);
  PHX_CHECK(mesh(open, 0.5, 0.25, {}).status == PHX_MC_INVALID_INPUT);
  Surface inverted = icosphere(2);
  for (size_t i = 0; i < inverted.f.size(); i += 3) std::swap(inverted.f[i + 1], inverted.f[i + 2]);
  PHX_CHECK(mesh(inverted, 0.5, 0.25, {}).status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(mesh(icosphere(2), 0.75, 0.25, {}).status == PHX_MC_INVALID_ARGUMENT);
  for (const double fraction : {0.0, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
    PHX_CHECK(mesh(icosphere(2), 0.5, 0.25, {0.01}, 1, nullptr, 1, fraction).status == PHX_MC_INVALID_ARGUMENT);
  }
}

}  // namespace

int main() {
  sphere_is_certified_and_on_the_surface();
  box_corners_and_volume_are_exact();
  layers_sit_at_their_thicknesses();
  layer_tilts_keep_the_requested_floor();
  cells_are_stars_about_their_vertex_average();
  convex_curve_layers_split();
  oblique_curve_segments_gain_midpoints();
  convex_curve_split_is_scale_free();
  deferred_proof_preserves_the_layered_mesh();
  invalid_input_is_refused();
  return phx::mc::test::finish("test_hex_dominant");
}
