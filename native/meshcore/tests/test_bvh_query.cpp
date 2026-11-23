//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// BVH point queries against brute-force references over every item.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kParallelQueryCount = 4099;

struct Tree {
  std::vector<double> node_min, node_max, node_size;
  std::vector<int64_t> left, right, leaf, leaf_items;
  std::vector<double> item_min, item_max;
  std::vector<std::vector<int64_t>> node_items;
  int64_t leaf_width = 3;

  phx_mc_bvh_tree_view view() const {
    return {static_cast<int64_t>(leaf.size()),
            static_cast<int64_t>(leaf_items.size()) / leaf_width,
            leaf_width,
            static_cast<int64_t>(item_min.size()) / 3,
            node_min.data(),
            node_max.data(),
            node_size.data(),
            left.data(),
            right.data(),
            leaf.data(),
            leaf_items.data(),
            item_min.data(),
            item_max.data()};
  }
};

// Median-split tree over the bounding boxes of simplices with `corners` vertices.
Tree build_tree(const std::vector<double>& vertices, const std::vector<int32_t>& simplices,
                int corners, int64_t leaf_width) {
  Tree tree;
  tree.leaf_width = leaf_width;
  const std::size_t count = simplices.size() / static_cast<std::size_t>(corners);
  for (std::size_t item = 0; item < count; ++item) {
    double lo[3] = {INFINITY, INFINITY, INFINITY};
    double hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int corner = 0; corner < corners; ++corner) {
      const double* v = &vertices[3 * static_cast<std::size_t>(
                                          simplices[item * static_cast<std::size_t>(corners) +
                                                    static_cast<std::size_t>(corner)])];
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::min(lo[axis], v[axis]);
        hi[axis] = std::max(hi[axis], v[axis]);
      }
    }
    tree.item_min.insert(tree.item_min.end(), lo, lo + 3);
    tree.item_max.insert(tree.item_max.end(), hi, hi + 3);
  }
  const auto build = [&](auto&& self, std::vector<int64_t> items) -> int64_t {
    const int64_t node = static_cast<int64_t>(tree.leaf.size());
    tree.leaf.push_back(-1);
    tree.left.push_back(-1);
    tree.right.push_back(-1);
    tree.node_items.push_back(items);
    double lo[3] = {INFINITY, INFINITY, INFINITY};
    double hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (const int64_t item : items) {
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::min(lo[axis], tree.item_min[static_cast<std::size_t>(3 * item + axis)]);
        hi[axis] = std::max(hi[axis], tree.item_max[static_cast<std::size_t>(3 * item + axis)]);
      }
    }
    tree.node_min.insert(tree.node_min.end(), lo, lo + 3);
    tree.node_max.insert(tree.node_max.end(), hi, hi + 3);
    tree.node_size.push_back((hi[0] - lo[0]) + (hi[1] - lo[1]) + (hi[2] - lo[2]));
    if (static_cast<int64_t>(items.size()) <= tree.leaf_width) {
      tree.leaf[static_cast<std::size_t>(node)] =
          static_cast<int64_t>(tree.leaf_items.size()) / tree.leaf_width;
      for (int64_t slot = 0; slot < tree.leaf_width; ++slot) {
        tree.leaf_items.push_back(slot < static_cast<int64_t>(items.size())
                                      ? items[static_cast<std::size_t>(slot)]
                                      : -1);
      }
      return node;
    }
    int axis = 0;
    for (int candidate = 1; candidate < 3; ++candidate) {
      if (hi[candidate] - lo[candidate] > hi[axis] - lo[axis]) axis = candidate;
    }
    std::sort(items.begin(), items.end(), [&](int64_t a, int64_t b) {
      return tree.item_min[static_cast<std::size_t>(3 * a + axis)] +
                 tree.item_max[static_cast<std::size_t>(3 * a + axis)] <
             tree.item_min[static_cast<std::size_t>(3 * b + axis)] +
                 tree.item_max[static_cast<std::size_t>(3 * b + axis)];
    });
    const auto half = static_cast<std::ptrdiff_t>(items.size() / 2);
    const int64_t l = self(self, std::vector<int64_t>(items.begin(), items.begin() + half));
    const int64_t r = self(self, std::vector<int64_t>(items.begin() + half, items.end()));
    tree.left[static_cast<std::size_t>(node)] = l;
    tree.right[static_cast<std::size_t>(node)] = r;
    return node;
  };
  std::vector<int64_t> items(count);
  for (std::size_t item = 0; item < count; ++item) items[item] = static_cast<int64_t>(item);
  build(build, items);
  return tree;
}

// A closed outward-oriented UV torus (R = 1, r = 0.4).
void torus(int around, std::vector<double>& vertices, std::vector<int32_t>& faces) {
  const int across = around / 2;
  for (int i = 0; i < around; ++i) {
    for (int j = 0; j < across; ++j) {
      const double u = 2.0 * kPi * i / around;
      const double w = 2.0 * kPi * j / across;
      const double ring = 1.0 + 0.4 * std::cos(w);
      vertices.insert(vertices.end(), {ring * std::cos(u), ring * std::sin(u), 0.4 * std::sin(w)});
    }
  }
  for (int i = 0; i < around; ++i) {
    for (int j = 0; j < across; ++j) {
      const int32_t a = i * across + j;
      const int32_t b = ((i + 1) % around) * across + j;
      const int32_t c = ((i + 1) % around) * across + (j + 1) % across;
      const int32_t d = i * across + (j + 1) % across;
      faces.insert(faces.end(), {a, b, c, a, c, d});
    }
  }
}

double torus_distance(const double* p) {
  const double ring = std::sqrt(p[0] * p[0] + p[1] * p[1]) - 1.0;
  return std::sqrt(ring * ring + p[2] * p[2]) - 0.4;
}

// Independent closest distance: minimum over the interior projection (Gram
// solve) and the clamped edge points.
double triangle_distance(const double* p, const double* a, const double* b, const double* c) {
  double u[3], v[3], w[3];
  for (int axis = 0; axis < 3; ++axis) {
    u[axis] = b[axis] - a[axis];
    v[axis] = c[axis] - a[axis];
    w[axis] = p[axis] - a[axis];
  }
  const auto dot = [](const double* x, const double* y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
  const double uu = dot(u, u), uv = dot(u, v), vv = dot(v, v), wu = dot(w, u), wv = dot(w, v);
  const double det = uu * vv - uv * uv;
  const double s = (vv * wu - uv * wv) / det;
  const double t = (uu * wv - uv * wu) / det;
  double best = INFINITY;
  if (s >= 0.0 && t >= 0.0 && s + t <= 1.0) {
    double q[3];
    for (int axis = 0; axis < 3; ++axis) q[axis] = a[axis] + s * u[axis] + t * v[axis] - p[axis];
    best = dot(q, q);
  }
  const double* corners[3] = {a, b, c};
  for (int edge = 0; edge < 3; ++edge) {
    const double* start = corners[edge];
    const double* end = corners[(edge + 1) % 3];
    double d[3], o[3];
    for (int axis = 0; axis < 3; ++axis) {
      d[axis] = end[axis] - start[axis];
      o[axis] = p[axis] - start[axis];
    }
    const double along = std::clamp(dot(o, d) / dot(d, d), 0.0, 1.0);
    double q[3];
    for (int axis = 0; axis < 3; ++axis) q[axis] = start[axis] + along * d[axis] - p[axis];
    best = std::min(best, dot(q, q));
  }
  return best;
}

double solid_angle(const double* p, const double* x, const double* y, const double* z) {
  double a[3], b[3], c[3];
  for (int axis = 0; axis < 3; ++axis) {
    a[axis] = x[axis] - p[axis];
    b[axis] = y[axis] - p[axis];
    c[axis] = z[axis] - p[axis];
  }
  const auto dot = [](const double* s, const double* t) { return s[0] * t[0] + s[1] * t[1] + s[2] * t[2]; };
  const double la = std::sqrt(dot(a, a)), lb = std::sqrt(dot(b, b)), lc = std::sqrt(dot(c, c));
  const double triple = a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) +
                        a[2] * (b[0] * c[1] - b[1] * c[0]);
  return 2.0 * std::atan2(triple, la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb);
}

const double* corner(const std::vector<double>& vertices, int32_t index) {
  return &vertices[3 * static_cast<std::size_t>(index)];
}

std::vector<double> random_points(std::size_t count, double extent, unsigned seed) {
  std::mt19937_64 random(seed);
  std::uniform_real_distribution<double> position(-extent, extent);
  std::vector<double> points(3 * count);
  for (double& value : points) value = position(random);
  return points;
}

template <class Query, class... T>
void check_query_workers(Query query, std::vector<T>&... serial) {
  // Use distinct initial values so an unwritten output cannot compare equal.
  auto trial = std::tuple{std::vector<T>(serial.size(), T{42})...};
  std::apply(
      [&](auto&... output) {
        for (const int32_t workers : {0, -1}) {
          PHX_CHECK(query(workers, output...) == PHX_MC_INVALID_ARGUMENT);
          PHX_CHECK((std::all_of(output.begin(), output.end(),
                                 [](auto value) { return value == 42; }) && ...));
        }
        PHX_CHECK(query(1, serial...) == PHX_MC_OK);
        for (const int32_t workers : {2, 3, 4, 8}) {
          (std::fill(output.begin(), output.end(), T{42}), ...);
          PHX_CHECK(query(workers, output...) == PHX_MC_OK);
          PHX_CHECK(((std::memcmp(serial.data(), output.data(), serial.size() * sizeof(T)) == 0) &&
                     ...));
        }
      },
      trial);
}

void test_nearest_matches_brute_force() {
  std::vector<double> vertices;
  std::vector<int32_t> faces;
  torus(24, vertices, faces);
  const Tree tree = build_tree(vertices, faces, 3, 3);
  const phx_mc_bvh_tree_view view = tree.view();
  const std::size_t count = 200;
  const std::vector<double> points = random_points(count, 1.6, 7);
  const int32_t k = 3;
  std::vector<int32_t> found(count * k);
  std::vector<double> squared(count * k), closest(count * k * 3);
  std::vector<int8_t> features(count * k);
  PHX_CHECK(phx_mc_bvh_query_nearest_triangles(
                &view, static_cast<int64_t>(vertices.size() / 3), vertices.data(), faces.data(),
                static_cast<int64_t>(count), points.data(), k, found.data(), squared.data(),
                closest.data(), features.data(), 1) == PHX_MC_OK);
  const std::size_t face_count = faces.size() / 3;
  for (std::size_t query = 0; query < count; ++query) {
    std::vector<std::pair<double, int32_t>> ranked;
    for (std::size_t face = 0; face < face_count; ++face) {
      ranked.emplace_back(triangle_distance(&points[3 * query], corner(vertices, faces[3 * face]),
                                            corner(vertices, faces[3 * face + 1]),
                                            corner(vertices, faces[3 * face + 2])),
                          static_cast<int32_t>(face));
    }
    std::sort(ranked.begin(), ranked.end());
    for (int32_t slot = 0; slot < k; ++slot) {
      const std::size_t entry = query * k + static_cast<std::size_t>(slot);
      PHX_CHECK_NEAR(squared[entry], ranked[static_cast<std::size_t>(slot)].first, 1.0e-12);
      PHX_CHECK(features[entry] >= 0 && features[entry] < PHX_MC_BVH_FEATURE_COUNT);
      double offset = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double d = closest[3 * entry + static_cast<std::size_t>(axis)] - points[3 * query + static_cast<std::size_t>(axis)];
        offset += d * d;
      }
      PHX_CHECK_NEAR(offset, squared[entry], 1.0e-12);
    }
    PHX_CHECK(squared[query * k] <= squared[query * k + 1]);
  }
}

void test_nearest_ties_and_padding() {
  // Two copies of one triangle: exact ties order by index; k beyond the item
  // count pads with -1 and +inf.
  const std::vector<double> vertices = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const std::vector<int32_t> faces = {0, 1, 2, 0, 1, 2};
  const Tree tree = build_tree(vertices, faces, 3, 1);
  phx_mc_bvh_tree_view view = tree.view();
  view.node_size = nullptr;  // Point-query input does not need pair ordering.
  const double samples[][3] = {{0.25, 0.25, 2.0}, {NAN, 0.0, 0.0}, {0.75, 0.75, -1.0}};
  std::vector<double> points(3 * kParallelQueryCount);
  for (std::size_t row = 0; row < kParallelQueryCount; ++row) {
    std::copy_n(samples[row % 3], 3, &points[3 * row]);
  }
  constexpr int32_t k = 3;
  std::vector<int32_t> found(kParallelQueryCount * k);
  std::vector<double> squared(kParallelQueryCount * k), closest(kParallelQueryCount * k * 3);
  std::vector<int8_t> features(kParallelQueryCount * k);
  check_query_workers(
      [&](int32_t workers, auto&... output) {
        return phx_mc_bvh_query_nearest_triangles(
            &view, 3, vertices.data(), faces.data(), static_cast<int64_t>(kParallelQueryCount),
            points.data(), k, output.data()..., workers);
      },
      found, squared, closest, features);
  for (std::size_t row = 0; row < kParallelQueryCount; ++row) {
    const std::size_t entry = row * k;
    if (row % 3 != 1) {
      const double distance = row % 3 == 0 ? 4.0 : 1.125;
      const int8_t feature = row % 3 == 0 ? 0 : 2;
      PHX_CHECK(found[entry] == 0 && found[entry + 1] == 1);
      PHX_CHECK(squared[entry] == distance && squared[entry + 1] == distance);
      PHX_CHECK(features[entry] == feature && features[entry + 1] == feature);
    }
    for (std::size_t slot = row % 3 == 1 ? 0 : 2; slot < k; ++slot) {
      const std::size_t padded = entry + slot;
      PHX_CHECK(found[padded] == -1 && features[padded] == -1);
      PHX_CHECK(squared[padded] == std::numeric_limits<double>::infinity());
      PHX_CHECK(std::isnan(closest[3 * padded]) && std::isnan(closest[3 * padded + 1]) &&
                std::isnan(closest[3 * padded + 2]));
    }
  }
  // Corner and edge features: beyond corner b, beside edge (a, b).
  const double beyond[6] = {2.0, -1.0, 0.0, 0.5, -1.0, 0.0};
  PHX_CHECK(phx_mc_bvh_query_nearest_triangles(
                &view, 3, vertices.data(), faces.data(), 2, beyond, 1, found.data(),
                squared.data(), closest.data(), features.data(), 1) == PHX_MC_OK);
  PHX_CHECK(features[0] == 5 && features[1] == 1);
  PHX_CHECK(phx_mc_bvh_query_nearest_triangles(
                &view, 3, vertices.data(), faces.data(), 1, samples[0], 0, found.data(),
                squared.data(), closest.data(), features.data(), 1) == PHX_MC_INVALID_ARGUMENT);
  const std::vector<int32_t> bad_faces = {0, 1, 3, 0, 1, 2};
  PHX_CHECK(phx_mc_bvh_query_nearest_triangles(
                &view, 3, vertices.data(), bad_faces.data(), 1, samples[0], 1, found.data(),
                squared.data(), closest.data(), features.data(), 1) == PHX_MC_INVALID_ARGUMENT);
}

// Angle-weighted pseudonormals per (face, feature), the table of the Python owner.
std::vector<double> feature_normals(const std::vector<double>& vertices,
                                    const std::vector<int32_t>& faces) {
  const std::size_t face_count = faces.size() / 3;
  std::vector<double> unit(3 * face_count), vertex(vertices.size(), 0.0);
  std::map<std::pair<int32_t, int32_t>, std::vector<double>> edge;
  for (std::size_t face = 0; face < face_count; ++face) {
    const double* p[3] = {corner(vertices, faces[3 * face]), corner(vertices, faces[3 * face + 1]),
                          corner(vertices, faces[3 * face + 2])};
    double u[3], v[3], n[3];
    for (int axis = 0; axis < 3; ++axis) {
      u[axis] = p[1][axis] - p[0][axis];
      v[axis] = p[2][axis] - p[0][axis];
    }
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
    const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int axis = 0; axis < 3; ++axis) unit[3 * face + static_cast<std::size_t>(axis)] = n[axis] / length;
    for (int k = 0; k < 3; ++k) {
      const double* o = p[k];
      const double* x = p[(k + 1) % 3];
      const double* y = p[(k + 2) % 3];
      double e[3], f[3];
      for (int axis = 0; axis < 3; ++axis) {
        e[axis] = x[axis] - o[axis];
        f[axis] = y[axis] - o[axis];
      }
      const double angle = std::acos(std::clamp(
          (e[0] * f[0] + e[1] * f[1] + e[2] * f[2]) /
              std::sqrt((e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) * (f[0] * f[0] + f[1] * f[1] + f[2] * f[2])),
          -1.0, 1.0));
      for (int axis = 0; axis < 3; ++axis) {
        vertex[3 * static_cast<std::size_t>(faces[3 * face + static_cast<std::size_t>(k)]) + static_cast<std::size_t>(axis)] +=
            angle * unit[3 * face + static_cast<std::size_t>(axis)];
      }
      const int32_t s = faces[3 * face + static_cast<std::size_t>(k)];
      const int32_t t = faces[3 * face + static_cast<std::size_t>((k + 1) % 3)];
      auto& sum = edge[{std::min(s, t), std::max(s, t)}];
      sum.resize(3, 0.0);
      for (int axis = 0; axis < 3; ++axis) sum[static_cast<std::size_t>(axis)] += unit[3 * face + static_cast<std::size_t>(axis)];
    }
  }
  std::vector<double> table;
  for (std::size_t face = 0; face < face_count; ++face) {
    table.insert(table.end(), &unit[3 * face], &unit[3 * face] + 3);
    for (int k = 0; k < 3; ++k) {
      const int32_t s = faces[3 * face + static_cast<std::size_t>(k)];
      const int32_t t = faces[3 * face + static_cast<std::size_t>((k + 1) % 3)];
      const auto& sum = edge[{std::min(s, t), std::max(s, t)}];
      table.insert(table.end(), sum.begin(), sum.end());
    }
    for (int k = 0; k < 3; ++k) {
      const double* n = &vertex[3 * static_cast<std::size_t>(faces[3 * face + static_cast<std::size_t>(k)])];
      table.insert(table.end(), n, n + 3);
    }
  }
  return table;
}

void test_signed_distance_sign() {
  std::vector<double> vertices;
  std::vector<int32_t> faces;
  torus(48, vertices, faces);
  const Tree tree = build_tree(vertices, faces, 3, 4);
  const phx_mc_bvh_tree_view view = tree.view();
  const std::vector<double> normals = feature_normals(vertices, faces);
  const std::size_t count = kParallelQueryCount;
  std::vector<double> points = random_points(count, 1.5, 11);
  points.back() = NAN;
  std::vector<double> distance(count), closest(3 * count);
  std::vector<int32_t> found(count);
  check_query_workers(
      [&](int32_t workers, auto&... output) {
        return phx_mc_bvh_query_signed_distance(
            &view, static_cast<int64_t>(vertices.size() / 3), vertices.data(), faces.data(),
            normals.data(), static_cast<int64_t>(count), points.data(), output.data()..., workers);
      },
      distance, found, closest);
  int mismatched = 0;
  for (std::size_t query = 0; query + 1 < count; ++query) {
    const double truth = torus_distance(&points[3 * query]);
    // The chord error of this torus stays below 0.02.
    if (std::fabs(truth) > 0.02 && (truth < 0.0) != (distance[query] < 0.0)) ++mismatched;
    PHX_CHECK(found[query] >= 0);
  }
  PHX_CHECK(mismatched == 0);
  PHX_CHECK(std::isnan(distance.back()) && found.back() == -1 && std::isnan(closest.back()));
}

// Oriented boundary chain of every node's patch as (offsets, edges).
void closing_fans(const Tree& tree, const std::vector<int32_t>& faces, std::vector<int64_t>& offsets,
                  std::vector<int32_t>& edges) {
  offsets.assign(1, 0);
  for (const auto& items : tree.node_items) {
    std::map<std::pair<int32_t, int32_t>, int> net;
    for (const int64_t item : items) {
      for (int k = 0; k < 3; ++k) {
        const int32_t s = faces[static_cast<std::size_t>(3 * item + k)];
        const int32_t t = faces[static_cast<std::size_t>(3 * item + (k + 1) % 3)];
        net[{std::min(s, t), std::max(s, t)}] += s < t ? 1 : -1;
      }
    }
    for (const auto& [edge, multiplicity] : net) {
      for (int copy = 0; copy < std::abs(multiplicity); ++copy) {
        if (multiplicity > 0) {
          edges.insert(edges.end(), {edge.first, edge.second});
        } else {
          edges.insert(edges.end(), {edge.second, edge.first});
        }
      }
    }
    offsets.push_back(static_cast<int64_t>(edges.size() / 2));
  }
}

void test_winding_routes() {
  std::vector<double> vertices;
  std::vector<int32_t> faces;
  torus(24, vertices, faces);
  // An open patch too: the exact route is exact for open surfaces as well.
  for (const std::size_t kept : {faces.size(), faces.size() / 2 + 9}) {
    std::vector<int32_t> surface(faces.begin(), faces.begin() + static_cast<std::ptrdiff_t>(kept));
    const Tree tree = build_tree(vertices, surface, 3, 3);
    const phx_mc_bvh_tree_view view = tree.view();
    std::vector<int64_t> offsets;
    std::vector<int32_t> edges;
    closing_fans(tree, surface, offsets, edges);
    const std::size_t count = kParallelQueryCount;
    const std::vector<double> points = random_points(count, 1.5, 5);
    std::vector<double> exact(count);
    check_query_workers(
        [&](int32_t workers, auto&... output) {
          return phx_mc_bvh_query_winding(
              &view, static_cast<int64_t>(vertices.size() / 3), vertices.data(), surface.data(),
              PHX_MC_WINDING_EXACT, offsets.data(), static_cast<int64_t>(edges.size() / 2),
              edges.data(), nullptr, 0.0, static_cast<int64_t>(count), points.data(),
              output.data()..., workers);
        },
        exact);
    for (std::size_t query = 0; query < 150; ++query) {
      double reference = 0.0;
      for (std::size_t face = 0; face < surface.size() / 3; ++face) {
        reference += solid_angle(&points[3 * query], corner(vertices, surface[3 * face]),
                                 corner(vertices, surface[3 * face + 1]),
                                 corner(vertices, surface[3 * face + 2]));
      }
      PHX_CHECK_NEAR(exact[query], reference / (4.0 * kPi), 1.0e-10);
    }
  }
  // FAST_DIPOLE: dipoles never used (huge opening angle) reproduce the exact
  // sum; at beta = 2 the sign agrees away from the surface.
  const Tree tree = build_tree(vertices, faces, 3, 3);
  const phx_mc_bvh_tree_view view = tree.view();
  std::vector<double> dipoles;
  for (std::size_t node = 0; node < tree.node_items.size(); ++node) {
    double normal[3] = {0, 0, 0}, center[3] = {0, 0, 0}, area = 0.0;
    for (const int64_t item : tree.node_items[node]) {
      const double* p[3] = {corner(vertices, faces[static_cast<std::size_t>(3 * item)]),
                            corner(vertices, faces[static_cast<std::size_t>(3 * item + 1)]),
                            corner(vertices, faces[static_cast<std::size_t>(3 * item + 2)])};
      double u[3], v[3], n[3];
      for (int axis = 0; axis < 3; ++axis) {
        u[axis] = p[1][axis] - p[0][axis];
        v[axis] = p[2][axis] - p[0][axis];
      }
      n[0] = 0.5 * (u[1] * v[2] - u[2] * v[1]);
      n[1] = 0.5 * (u[2] * v[0] - u[0] * v[2]);
      n[2] = 0.5 * (u[0] * v[1] - u[1] * v[0]);
      const double a = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      area += a;
      for (int axis = 0; axis < 3; ++axis) {
        normal[axis] += n[axis];
        center[axis] += a * (p[0][axis] + p[1][axis] + p[2][axis]) / 3.0;
      }
    }
    double reach = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      center[axis] /= area;
      const double r = std::max(std::fabs(tree.node_max[3 * node + static_cast<std::size_t>(axis)] - center[axis]),
                                std::fabs(center[axis] - tree.node_min[3 * node + static_cast<std::size_t>(axis)]));
      reach += r * r;
    }
    dipoles.insert(dipoles.end(), {normal[0], normal[1], normal[2], center[0], center[1], center[2],
                                   std::sqrt(reach)});
  }
  const std::size_t count = kParallelQueryCount;
  const std::vector<double> points = random_points(count, 1.5, 9);
  std::vector<double> opened(400), fast(count);
  const int64_t vertex_count = static_cast<int64_t>(vertices.size() / 3);
  PHX_CHECK(phx_mc_bvh_query_winding(
                &view, vertex_count, vertices.data(), faces.data(), PHX_MC_WINDING_FAST_DIPOLE,
                nullptr, 0, nullptr, dipoles.data(), 1.0e9, static_cast<int64_t>(opened.size()),
                points.data(), opened.data(), 1) == PHX_MC_OK);
  check_query_workers(
      [&](int32_t workers, auto&... output) {
        return phx_mc_bvh_query_winding(
            &view, vertex_count, vertices.data(), faces.data(), PHX_MC_WINDING_FAST_DIPOLE,
            nullptr, 0, nullptr, dipoles.data(), 2.0, static_cast<int64_t>(count), points.data(),
            output.data()..., workers);
      },
      fast);
  int mismatched = 0;
  for (std::size_t query = 0; query < opened.size(); ++query) {
    double reference = 0.0;
    for (std::size_t face = 0; face < faces.size() / 3; ++face) {
      reference += solid_angle(&points[3 * query], corner(vertices, faces[3 * face]),
                               corner(vertices, faces[3 * face + 1]),
                               corner(vertices, faces[3 * face + 2]));
    }
    PHX_CHECK_NEAR(opened[query], reference / (4.0 * kPi), 1.0e-10);
    const double truth = torus_distance(&points[3 * query]);
    if (std::fabs(truth) > 0.05 && (fast[query] > 0.5) != (truth < 0.0)) ++mismatched;
  }
  PHX_CHECK(mismatched == 0);
  PHX_CHECK(phx_mc_bvh_query_winding(
                &view, vertex_count, vertices.data(), faces.data(), 2, nullptr, 0, nullptr,
                dipoles.data(), 2.0, 1, points.data(), fast.data(), 1) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_bvh_query_winding(
                &view, vertex_count, vertices.data(), faces.data(), PHX_MC_WINDING_FAST_DIPOLE,
                nullptr, 0, nullptr, dipoles.data(), 0.0, 1, points.data(), fast.data(), 1) ==
            PHX_MC_INVALID_ARGUMENT);
}

// Barycentric reference coordinates of a point in one tetrahedron by Cramer's rule.
std::vector<double> reference_of(const std::vector<double>& vertices, const int32_t* cell,
                                 const double* point) {
  const double* o = corner(vertices, cell[0]);
  double e[3][3], q[3];
  for (int column = 0; column < 3; ++column) {
    for (int axis = 0; axis < 3; ++axis) e[column][axis] = corner(vertices, cell[column + 1])[axis] - o[axis];
  }
  for (int axis = 0; axis < 3; ++axis) q[axis] = point[axis] - o[axis];
  const auto det = [](const double* a, const double* b, const double* c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
  };
  const double d = det(e[0], e[1], e[2]);
  return {det(q, e[1], e[2]) / d, det(e[0], q, e[2]) / d, det(e[0], e[1], q) / d};
}

void test_tetrahedron_locate() {
  // A 3 x 3 x 3 grid of unit cubes, six positive tetrahedra per cube.
  const int n = 3;
  std::vector<double> vertices;
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      for (int k = 0; k <= n; ++k) vertices.insert(vertices.end(), {double(i), double(j), double(k)});
    }
  }
  const auto id = [&](int i, int j, int k) { return static_cast<int32_t>((i * (n + 1) + j) * (n + 1) + k); };
  std::vector<int32_t> cells;
  const int paths[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        for (const auto& path : paths) {
          int at[3] = {i, j, k};
          std::vector<int32_t> cell = {id(at[0], at[1], at[2])};
          for (const int axis : path) {
            ++at[axis];
            cell.push_back(id(at[0], at[1], at[2]));
          }
          // Orient positively.
          const double* o = corner(vertices, cell[0]);
          double e[3][3];
          for (int c = 0; c < 3; ++c) {
            for (int axis = 0; axis < 3; ++axis) e[c][axis] = corner(vertices, cell[static_cast<std::size_t>(c + 1)])[axis] - o[axis];
          }
          const double volume = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
          if (volume < 0.0) std::swap(cell[2], cell[3]);
          cells.insert(cells.end(), cell.begin(), cell.end());
        }
      }
    }
  }
  const Tree tree = build_tree(vertices, cells, 4, 4);
  const phx_mc_bvh_tree_view view = tree.view();
  std::vector<double> points = random_points(kParallelQueryCount - 3, 1.6, 13);
  for (double& value : points) value += 1.5;
  // A shared vertex, a shared face point, and a nonfinite point.
  points.insert(points.end(), {1.0, 1.0, 1.0, 0.5, 0.5, 0.5, NAN, 0.0, 0.0});
  const std::size_t count = points.size() / 3;
  const int32_t capacity = 2;
  const double tolerance = 1.0e-12;
  std::vector<int32_t> found(count), status(count), containing(count), candidates(count * capacity);
  std::vector<double> reference(3 * count), residual(count), condition(count),
      candidate_reference(3 * count * capacity);
  const int64_t vertex_count = static_cast<int64_t>(vertices.size() / 3);
  const auto locate = [&](const uint8_t* mask, int32_t workers, auto&... output) {
    return phx_mc_bvh_query_locate_tetrahedra(
        &view, vertex_count, vertices.data(), cells.data(), mask, tolerance,
        static_cast<int64_t>(count), points.data(), capacity, output.data()..., workers);
  };
  check_query_workers(
      [&](int32_t workers, auto&... output) { return locate(nullptr, workers, output...); },
      found, reference, residual, condition, status, containing, candidates, candidate_reference);
  const std::size_t cell_count = cells.size() / 4;
  // Keep the brute-force reference bounded, including the two shared-boundary points.
  constexpr std::size_t random_reference_count = 300;
  for (std::size_t sample = 0; sample < random_reference_count + 2; ++sample) {
    const std::size_t query =
        sample < random_reference_count ? sample : count - 3 + (sample - random_reference_count);
    std::vector<int32_t> containing_cells;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
      const std::vector<double> r = reference_of(vertices, &cells[4 * cell], &points[3 * query]);
      if (r[0] >= -tolerance && r[1] >= -tolerance && r[2] >= -tolerance &&
          r[0] + r[1] + r[2] <= 1.0 + tolerance) {
        containing_cells.push_back(static_cast<int32_t>(cell));
      }
    }
    PHX_CHECK(containing[query] == static_cast<int32_t>(containing_cells.size()));
    if (containing_cells.empty()) {
      PHX_CHECK(found[query] == -1 && status[query] == PHX_MC_LOCATE_OUTSIDE);
      continue;
    }
    PHX_CHECK(found[query] == containing_cells.front());
    PHX_CHECK(status[query] == PHX_MC_LOCATE_LOCATED);
    PHX_CHECK(residual[query] < 1.0e-12);
    PHX_CHECK(condition[query] >= 1.0 && std::isfinite(condition[query]));
    for (int32_t slot = 0; slot < capacity; ++slot) {
      const std::size_t entry = query * capacity + static_cast<std::size_t>(slot);
      PHX_CHECK(candidates[entry] == (static_cast<std::size_t>(slot) < containing_cells.size()
                                          ? containing_cells[static_cast<std::size_t>(slot)]
                                          : -1));
    }
  }
  PHX_CHECK(containing[count - 3] == 24);  // the interior grid vertex lies in 24 cells
  PHX_CHECK(status[count - 1] == PHX_MC_LOCATE_NONFINITE && found[count - 1] == -1);
  // Masking the lowest containing cell selects the next one.
  std::vector<uint8_t> mask(cell_count, 1);
  const int32_t first = found[count - 2];
  PHX_CHECK(first >= 0 && static_cast<std::size_t>(first) < mask.size());
  if (first >= 0 && static_cast<std::size_t>(first) < mask.size()) {
    mask[static_cast<std::size_t>(first)] = 0;
    const int32_t second = candidates[(count - 2) * capacity + 1];
    PHX_CHECK(locate(mask.data(), 4, found, reference, residual, condition, status, containing,
                     candidates, candidate_reference) == PHX_MC_OK);
    PHX_CHECK(found[count - 2] == second);
  }
  // A flattened cell is degenerate, not outside.
  std::vector<double> flat = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
  const std::vector<int32_t> flat_cell = {0, 1, 2, 3};
  const Tree flat_tree = build_tree(flat, flat_cell, 4, 1);
  const phx_mc_bvh_tree_view flat_view = flat_tree.view();
  const double inside[3] = {0.2, 0.2, 0.0};
  PHX_CHECK(phx_mc_bvh_query_locate_tetrahedra(
                &flat_view, 4, flat.data(), flat_cell.data(), nullptr, 0.0, 1, inside, 1,
                found.data(), reference.data(), residual.data(), condition.data(), status.data(),
                containing.data(), candidates.data(), candidate_reference.data(), 1) == PHX_MC_OK);
  PHX_CHECK(status[0] == PHX_MC_LOCATE_DEGENERATE_CELL && found[0] == -1);
  PHX_CHECK(phx_mc_bvh_query_locate_tetrahedra(
                &flat_view, 4, flat.data(), flat_cell.data(), nullptr, -1.0, 1, inside, 1,
                found.data(), reference.data(), residual.data(), condition.data(), status.data(),
                containing.data(), candidates.data(), candidate_reference.data(), 1) ==
            PHX_MC_INVALID_ARGUMENT);
}

void test_tetrahedron_locate_refuses_malformed_input_without_writes() {
  const std::vector<double> vertices{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
  const std::vector<int32_t> cells{0, 1, 2, 3, 1, 2, 3, 4};
  const double query[]{0.2, 0.2, 0.2};
  Tree tree = build_tree(vertices, cells, 4, 1);
  const auto valid = tree.view();
  int32_t found = 42, status = 42, containing = 42, candidate = 42;
  double reference[3]{}, residual = 42.0, condition = 42.0, candidate_reference[3]{};
  const auto check_invalid = [&](const phx_mc_bvh_tree_view& view,
                                 const std::vector<int32_t>& rows) {
    PHX_CHECK(phx_mc_bvh_query_locate_tetrahedra(
                  &view, 5, vertices.data(), rows.data(), nullptr, 0.0, 1, query, 1, &found,
                  reference, &residual, &condition, &status, &containing, &candidate,
                  candidate_reference, 1) == PHX_MC_INVALID_ARGUMENT);
    PHX_CHECK(found == 42 && status == 42 && containing == 42 && candidate == 42);
  };
  auto bad_cells = cells;
  for (int32_t invalid : {-1, 5}) {
    bad_cells.back() = invalid;
    check_invalid(valid, bad_cells);
  }
  const int64_t left = tree.left[0];
  tree.left[0] = 0;  // the root refers to itself
  check_invalid(valid, cells);
  tree.left[0] = left;
  tree.right[0] = left;  // two edges reach the same leaf
  check_invalid(valid, cells);
}

void test_tetrahedron_locate_handles_empty_tree_item_banks() {
  const double lower[]{0, 0, 0}, upper[]{1, 1, 1};
  const int64_t absent = -1, leaf = 0;
  const phx_mc_bvh_tree_view view{1, 1, 1, 0, lower, upper, nullptr, &absent, &absent,
                                  &leaf, &absent, nullptr, nullptr};
  const double points[]{0.2, 0.2, 0.2, NAN, 0.0, 0.0};
  int32_t found[2], status[2], containing[2], candidates[2];
  double reference[6], residual[2], condition[2], candidate_reference[6];
  const int32_t result = phx_mc_bvh_query_locate_tetrahedra(
      &view, 0, nullptr, nullptr, nullptr, 0.0, 2, points, 1, found, reference, residual,
      condition, status, containing, candidates, candidate_reference, 1);
  PHX_CHECK(result == PHX_MC_OK);
  if (result != PHX_MC_OK) return;
  PHX_CHECK(status[0] == PHX_MC_LOCATE_OUTSIDE);
  PHX_CHECK(status[1] == PHX_MC_LOCATE_NONFINITE);
  PHX_CHECK(found[0] == -1 && found[1] == -1 && containing[0] == 0 && containing[1] == 0);
}

}  // namespace

int main() {
  test_nearest_matches_brute_force();
  test_nearest_ties_and_padding();
  test_signed_distance_sign();
  test_winding_routes();
  test_tetrahedron_locate();
  test_tetrahedron_locate_refuses_malformed_input_without_writes();
  test_tetrahedron_locate_handles_empty_tree_item_banks();
  return phx::mc::test::finish("test_bvh_query");
}
