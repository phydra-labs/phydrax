// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_surface_embedding.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "intersections.hpp"
#include "predicates.hpp"

namespace {
using Finding = phx_mc_surface_finding;

struct Edge {
  int64_t a, b;
  int64_t triangle;
  int corner_a, corner_b;
  bool forward;
};

int64_t root(std::vector<int64_t>& parent, int64_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void topology(int64_t n, const int64_t* triangles, std::vector<Finding>& out) {
  std::vector<Edge> edges;
  std::vector<std::pair<int64_t, int64_t>> corners;
  edges.reserve(static_cast<size_t>(3 * n));
  corners.reserve(static_cast<size_t>(3 * n));
  std::vector<int64_t> parent(static_cast<size_t>(3 * n));
  std::iota(parent.begin(), parent.end(), int64_t{0});
  for (int64_t t = 0; t < n; ++t) {
    for (int k = 0; k < 3; ++k) {
      const int l = (k + 1) % 3;
      const int64_t a = triangles[3*t+k], b = triangles[3*t+l];
      corners.emplace_back(a, 3*t+k);
      edges.push_back({std::min(a,b), std::max(a,b), t,
                       a < b ? k : l, a < b ? l : k, a < b});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
    if (x.a != y.a) return x.a < y.a;
    if (x.b != y.b) return x.b < y.b;
    return x.triangle < y.triangle;
  });
  for (size_t i = 0; i < edges.size();) {
    size_t j = i + 1;
    while (j < edges.size() && edges[i].a == edges[j].a && edges[i].b == edges[j].b) ++j;
    if (j-i != 2 || edges[i].forward == edges[i+1].forward) {
      out.push_back({PHX_MC_SURFACE_EDGE, edges[i].a, edges[i].b, 0});
    }
    if (j-i == 2) {
      const Edge& x = edges[i];
      const Edge& y = edges[i+1];
      for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int64_t a = 3*x.triangle + (endpoint == 0 ? x.corner_a : x.corner_b);
        const int64_t b = 3*y.triangle + (endpoint == 0 ? y.corner_a : y.corner_b);
        parent[root(parent, a)] = root(parent, b);
      }
    }
    i = j;
  }
  std::sort(corners.begin(), corners.end());
  for (size_t i = 0; i < corners.size();) {
    size_t j = i + 1;
    const int64_t r = root(parent, corners[i].second);
    bool disconnected = false;
    while (j < corners.size() && corners[j].first == corners[i].first) {
      disconnected |= root(parent, corners[j].second) != r;
      ++j;
    }
    if (disconnected) out.push_back({PHX_MC_SURFACE_VERTEX_LINK, corners[i].first, -1, 0});
    i = j;
  }
}

bool nondegenerate(const double* a, const double* b, const double* c) {
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis+1)%3, v = (axis+2)%3;
    const double aa[2] = {a[u], a[v]}, bb[2] = {b[u], b[v]}, cc[2] = {c[u], c[v]};
    if (phx::mc::orient2d(aa, bb, cc) != 0) return true;
  }
  return false;
}

struct Tree {
  std::vector<double> lo, hi, size, item_lo, item_hi;
  std::vector<int64_t> left, right, leaf, items, order;
  int64_t build(int64_t begin, int64_t end) {
    const int64_t node = static_cast<int64_t>(left.size());
    left.push_back(-1); right.push_back(-1); leaf.push_back(-1); size.push_back(0);
    lo.resize(lo.size()+3); hi.resize(hi.size()+3);
    for (int d = 0; d < 3; ++d) {
      double a = item_lo[3*order[begin]+d], b = item_hi[3*order[begin]+d];
      for (int64_t i = begin+1; i < end; ++i) {
        a = std::min(a, item_lo[3*order[i]+d]); b = std::max(b, item_hi[3*order[i]+d]);
      }
      lo[3*node+d] = a; hi[3*node+d] = b; size[node] += b-a;
    }
    if (end-begin == 1) {
      leaf[node] = static_cast<int64_t>(items.size()); items.push_back(order[begin]);
    } else {
      int axis = 0;
      for (int d = 1; d < 3; ++d)
        if (hi[3*node+d]-lo[3*node+d] > hi[3*node+axis]-lo[3*node+axis]) axis = d;
      const int64_t mid = begin+(end-begin)/2;
      std::nth_element(order.begin()+begin, order.begin()+mid, order.begin()+end,
                       [&](int64_t a, int64_t b) {
        const double ca = item_lo[3*a+axis]+item_hi[3*a+axis];
        const double cb = item_lo[3*b+axis]+item_hi[3*b+axis];
        return ca == cb ? a < b : ca < cb;
      });
      const int64_t l = build(begin, mid), r = build(mid, end);
      left[node] = l; right[node] = r;
    }
    return node;
  }
  phx_mc_bvh_tree_view view() const {
    return {static_cast<int64_t>(left.size()), static_cast<int64_t>(items.size()), 1,
            static_cast<int64_t>(order.size()), lo.data(), hi.data(), size.data(),
            left.data(), right.data(), leaf.data(), items.data(), item_lo.data(), item_hi.data()};
  }
};

int32_t contacts(const double* points, int64_t n, const int64_t* triangles,
                 std::vector<Finding>& out) {
  if (n == 0) return PHX_MC_OK;
  Tree tree;
  tree.order.resize(static_cast<size_t>(n));
  std::iota(tree.order.begin(), tree.order.end(), int64_t{0});
  tree.item_lo.resize(static_cast<size_t>(3*n)); tree.item_hi.resize(static_cast<size_t>(3*n));
  for (int64_t t = 0; t < n; ++t) for (int d = 0; d < 3; ++d) {
    const double a = points[3*triangles[3*t]+d], b = points[3*triangles[3*t+1]+d],
                 c = points[3*triangles[3*t+2]+d];
    tree.item_lo[3*t+d] = std::min({a,b,c}); tree.item_hi[3*t+d] = std::max({a,b,c});
  }
  tree.build(0, n);
  const auto view = tree.view();
  phx_mc_bvh_pair_stream* raw = nullptr;
  int32_t status = phx_mc_bvh_pair_stream_create(&view, &view, 3, 1, 0, 0, 256, 256, 1, &raw);
  if (status != PHX_MC_OK) return status;
  std::unique_ptr<phx_mc_bvh_pair_stream, decltype(&phx_mc_bvh_pair_stream_free)> stream(raw, phx_mc_bvh_pair_stream_free);
  int64_t first[256], second[256], ends[256], blocks;
  do {
    status = phx_mc_bvh_pair_stream_next(stream.get(), 256, 256, first, second, ends, &blocks);
    if (status != PHX_MC_OK) return status;
    const int64_t count = blocks ? ends[blocks-1] : 0;
    for (int64_t k = 0; k < count; ++k) {
      const int64_t a = first[k], b = second[k];
      if (a >= b) continue;
      const double* p[3]; const double* q[3]; int64_t pi[3], qi[3];
      for (int j = 0; j < 3; ++j) {
        pi[j] = triangles[3*a+j]; qi[j] = triangles[3*b+j];
        p[j] = points+3*pi[j]; q[j] = points+3*qi[j];
      }
      phx::mc::Contact contact;
      status = phx::mc::intersect_triangles(p, q, pi, qi, contact);
      if (status != PHX_MC_OK) return status;
      if (contact.kind != PHX_MC_DISJOINT && contact.kind != PHX_MC_SHARED_VERTEX &&
          contact.kind != PHX_MC_SHARED_EDGE)
        out.push_back({PHX_MC_SURFACE_CONTACT, a, b, contact.kind});
    }
  } while (blocks != 0);
  return PHX_MC_OK;
}
}  // namespace

extern "C" int32_t phx_mc_verify_surface_embedding(
    int64_t point_count, const double* points, int64_t triangle_count, const int64_t* triangles,
    int64_t finding_capacity, phx_mc_surface_finding* findings,
    phx_mc_surface_embedding_verdict* verdict) {
  return phx::mc::guarded([&]() -> int32_t {
    if (point_count < 0 || triangle_count < 0 || finding_capacity < 0 || !verdict ||
        (point_count && !points) || (triangle_count && !triangles) ||
        (finding_capacity && !findings) ||
        !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(finding_capacity, 1, sizeof(Finding))) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t k = 0; k < 3*triangle_count; ++k)
      if (triangles[k] < 0 || triangles[k] >= point_count) return PHX_MC_INVALID_INPUT;
    const phx::mc::GradualNearestScope mode;
    std::vector<Finding> result;
    if (triangle_count == 0) result.push_back({PHX_MC_SURFACE_EMPTY, -1, -1, 0});
    std::vector<uint8_t> used(static_cast<size_t>(point_count), 0);
    for (int64_t k = 0; k < 3*triangle_count; ++k) used[triangles[k]] = 1;
    for (int64_t i = 0; i < point_count; ++i) if (used[i]) {
      for (int d = 0; d < 3; ++d) if (!phx::mc::coordinate_in_domain(points[3*i+d])) {
        result.push_back({PHX_MC_SURFACE_OUTSIDE_DOMAIN, i, -1, 0}); break;
      }
    }
    const bool outside = !result.empty() && result[0].kind == PHX_MC_SURFACE_OUTSIDE_DOMAIN;
    if (!outside && triangle_count != 0) {
      for (int64_t t = 0; t < triangle_count; ++t) {
        const int64_t* v = triangles+3*t;
        if (v[0] == v[1] || v[1] == v[2] || v[2] == v[0] ||
            !nondegenerate(points+3*v[0], points+3*v[1], points+3*v[2]))
          result.push_back({PHX_MC_SURFACE_DEGENERATE, t, -1, 0});
      }
      if (result.empty()) {
        topology(triangle_count, triangles, result);
        const int32_t status = contacts(points, triangle_count, triangles, result);
        if (status != PHX_MC_OK) return status;
      }
    }
    std::sort(result.begin(), result.end(), [](const Finding& a, const Finding& b) {
      if (a.kind != b.kind) return a.kind < b.kind;
      if (a.first != b.first) return a.first < b.first;
      return a.second < b.second;
    });
    if (static_cast<uint64_t>(result.size()) > static_cast<uint64_t>(finding_capacity)) return PHX_MC_CAPACITY_EXCEEDED;
    std::copy(result.begin(), result.end(), findings);
    *verdict = {outside ? PHX_MC_VERDICT_UNRESOLVED :
                (result.empty() ? PHX_MC_VERDICT_CERTIFIED : PHX_MC_VERDICT_VIOLATED),
                static_cast<int64_t>(result.size())};
    return PHX_MC_OK;
  });
}
