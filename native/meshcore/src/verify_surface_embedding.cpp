// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_surface_embedding.hpp"

#include <algorithm>
#include <array>
#include <limits>
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

struct Box {
  double lo[3], hi[3];
};

struct Node {
  Box box;
  int64_t begin, end;
  int64_t left, right;
};

struct Item {
  uint64_t morton;
  int64_t triangle;
};

bool overlaps(const Box& a, const Box& b) {
  for (int d = 0; d < 3; ++d)
    if (a.hi[d] < b.lo[d] || b.hi[d] < a.lo[d]) return false;
  return true;
}

uint32_t spread(uint32_t value) {
  value = (value | (value << 16)) & 0x030000ffU;
  value = (value | (value << 8)) & 0x0300f00fU;
  value = (value | (value << 4)) & 0x030c30c3U;
  return (value | (value << 2)) & 0x09249249U;
}

struct Tree {
  std::vector<Box> boxes;
  std::vector<Item> items;
  std::vector<Node> nodes;

  Tree(const double* points, int64_t n, const int64_t* triangles) {
    boxes.resize(static_cast<size_t>(n));
    items.resize(static_cast<size_t>(n));
    Box bounds{};
    for (int64_t t = 0; t < n; ++t) {
      Box& box = boxes[t];
      for (int d = 0; d < 3; ++d) {
        const double a = points[3*triangles[3*t]+d];
        const double b = points[3*triangles[3*t+1]+d];
        const double c = points[3*triangles[3*t+2]+d];
        box.lo[d] = std::min({a, b, c});
        box.hi[d] = std::max({a, b, c});
        if (t == 0) {
          bounds.lo[d] = box.lo[d];
          bounds.hi[d] = box.hi[d];
        } else {
          bounds.lo[d] = std::min(bounds.lo[d], box.lo[d]);
          bounds.hi[d] = std::max(bounds.hi[d], box.hi[d]);
        }
      }
    }
    for (int64_t t = 0; t < n; ++t) {
      uint32_t code = 0;
      for (int d = 0; d < 3; ++d) {
        const double width = bounds.hi[d] - bounds.lo[d];
        const double center = (boxes[t].lo[d] + boxes[t].hi[d]) * 0.5;
        const double unit = width > 0 ? (center - bounds.lo[d]) / width : 0;
        const uint32_t coordinate = static_cast<uint32_t>(std::clamp(unit * 1023.0, 0.0, 1023.0));
        code |= spread(coordinate) << d;
      }
      items[t] = {code, t};
    }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
      return a.morton == b.morton ? a.triangle < b.triangle : a.morton < b.morton;
    });
    nodes.reserve(static_cast<size_t>(n / 2 + 1));
    build(0, n);
  }

  int64_t build(int64_t begin, int64_t end) {
    const int64_t index = static_cast<int64_t>(nodes.size());
    nodes.push_back({});
    Node result{};
    result.begin = begin;
    result.end = end;
    result.left = result.right = -1;
    if (end - begin <= 8) {
      result.box = boxes[items[begin].triangle];
      for (int64_t i = begin + 1; i < end; ++i) {
        const Box& box = boxes[items[i].triangle];
        for (int d = 0; d < 3; ++d) {
          result.box.lo[d] = std::min(result.box.lo[d], box.lo[d]);
          result.box.hi[d] = std::max(result.box.hi[d], box.hi[d]);
        }
      }
    } else {
      const int64_t mid = begin + (end - begin) / 2;
      result.left = build(begin, mid);
      result.right = build(mid, end);
      for (int d = 0; d < 3; ++d) {
        result.box.lo[d] = std::min(nodes[result.left].box.lo[d], nodes[result.right].box.lo[d]);
        result.box.hi[d] = std::max(nodes[result.left].box.hi[d], nodes[result.right].box.hi[d]);
      }
    }
    nodes[index] = result;
    return index;
  }
};

// A strict separating plane proves that the closed triangles do not meet
// beyond a shared vertex. The filtered predicate falls back to exact signs.
bool same_strict_side(const double* const plane[3], const double* a, const double* b) {
  const int first = phx::mc::orient3d_filtered(plane[0], plane[1], plane[2], a);
  return first != 0 &&
         first == phx::mc::orient3d_filtered(plane[0], plane[1], plane[2], b);
}

bool separate_triangle(const double* const plane[3], const double* const other[3]) {
  const int first = phx::mc::orient3d_filtered(plane[0], plane[1], plane[2], other[0]);
  return first != 0 &&
         first == phx::mc::orient3d_filtered(plane[0], plane[1], plane[2], other[1]) &&
         first == phx::mc::orient3d_filtered(plane[0], plane[1], plane[2], other[2]);
}

bool clear_shared_edge(const double* a, const double* b, const double* c, const double* d) {
  if (phx::mc::orient3d_filtered(a, b, c, d) != 0) return true;
  // Coplanar nondegenerate triangles lie on opposite sides of their common
  // edge exactly when their intersection is just that edge.
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    const double aa[2] = {a[u], a[v]}, bb[2] = {b[u], b[v]};
    const double cc[2] = {c[u], c[v]}, dd[2] = {d[u], d[v]};
    const int side = phx::mc::orient2d_filtered(aa, bb, cc);
    if (side != 0) return side == -phx::mc::orient2d_filtered(aa, bb, dd);
  }
  return false;
}

bool clear_pair(const double* const p[3], const double* const q[3],
                const int64_t* pi, const int64_t* qi) {
  bool shared_p[3] = {}, shared_q[3] = {};
  int shared = 0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    if (pi[i] == qi[j]) {
      shared_p[i] = shared_q[j] = true;
      ++shared;
    }
  }
  if (shared == 2) {
    const double *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    for (int i = 0; i < 3; ++i) {
      if (shared_p[i]) {
        if (!a) a = p[i]; else b = p[i];
      } else c = p[i];
      if (!shared_q[i]) d = q[i];
    }
    return clear_shared_edge(a, b, c, d);
  }
  if (shared == 1) {
    const double *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    for (int i = 0; i < 3; ++i) {
      if (!shared_p[i]) { if (!a) a = p[i]; else b = p[i]; }
      if (!shared_q[i]) { if (!c) c = q[i]; else d = q[i]; }
    }
    return same_strict_side(p, c, d) || same_strict_side(q, a, b);
  }
  if (shared == 0) return separate_triangle(p, q) || separate_triangle(q, p);
  return false;
}

int32_t classify_pair(const double* points, const int64_t* triangles, int64_t a, int64_t b,
                      std::vector<Finding>& out) {
  if (a > b) std::swap(a, b);
  const double* p[3];
  const double* q[3];
  const int64_t* pi = triangles + 3*a;
  const int64_t* qi = triangles + 3*b;
  for (int j = 0; j < 3; ++j) {
    p[j] = points + 3*pi[j];
    q[j] = points + 3*qi[j];
  }
  if (clear_pair(p, q, pi, qi)) return PHX_MC_OK;
  phx::mc::Contact contact;
  const int32_t status = phx::mc::intersect_triangles(p, q, pi, qi, contact);
  if (status != PHX_MC_OK) return status;
  if (contact.kind != PHX_MC_DISJOINT && contact.kind != PHX_MC_SHARED_VERTEX &&
      contact.kind != PHX_MC_SHARED_EDGE)
    out.push_back({PHX_MC_SURFACE_CONTACT, a, b, contact.kind});
  return PHX_MC_OK;
}

int32_t contacts(const double* points, int64_t n, const int64_t* triangles,
                 std::vector<Finding>& out) {
  if (n == 0) return PHX_MC_OK;
  Tree tree(points, n, triangles);
  std::vector<std::array<int64_t, 2>> stack{{0, 0}};
  while (!stack.empty()) {
    const auto [a, b] = stack.back();
    stack.pop_back();
    const Node& first = tree.nodes[a];
    const Node& second = tree.nodes[b];
    if (!overlaps(first.box, second.box)) continue;
    if (first.left < 0 && second.left < 0) {
      for (int64_t i = first.begin; i < first.end; ++i) {
        for (int64_t j = a == b ? i + 1 : second.begin; j < second.end; ++j) {
          const int64_t x = tree.items[i].triangle, y = tree.items[j].triangle;
          if (!overlaps(tree.boxes[x], tree.boxes[y])) continue;
          const int32_t status = classify_pair(points, triangles, x, y, out);
          if (status != PHX_MC_OK) return status;
        }
      }
    } else if (a == b) {
      stack.push_back({first.left, first.left});
      stack.push_back({first.left, first.right});
      stack.push_back({first.right, first.right});
    } else if (second.left < 0 ||
               (first.left >= 0 && first.end - first.begin >= second.end - second.begin)) {
      stack.push_back({first.left, b});
      stack.push_back({first.right, b});
    } else {
      stack.push_back({a, second.left});
      stack.push_back({a, second.right});
    }
  }
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
