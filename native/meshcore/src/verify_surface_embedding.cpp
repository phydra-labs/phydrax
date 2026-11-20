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

int64_t root(std::vector<int64_t>& parent, int64_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

int64_t next_corner(int64_t corner) {
  return corner % 3 == 2 ? corner - 2 : corner + 1;
}

void pair_packed_edges(int64_t point_count, int64_t n, const int64_t* triangles,
                       std::vector<int64_t>& parent, std::vector<Finding>& out) {
  using PackedEdge = std::pair<uint64_t, int64_t>;
  std::vector<PackedEdge> edges;
  edges.reserve(static_cast<size_t>(3*n));
  for (int64_t corner = 0; corner < 3*n; ++corner) {
    const int64_t a = triangles[corner], b = triangles[next_corner(corner)];
    const uint64_t key = (static_cast<uint64_t>(std::min(a, b)) << 32) |
                         static_cast<uint64_t>(std::max(a, b));
    edges.emplace_back(key, corner);
  }
  if (point_count <= 3*n) {
    std::vector<size_t> offsets(static_cast<size_t>(point_count) + 1, 0);
    for (const PackedEdge& edge : edges) ++offsets[(edge.first >> 32) + 1];
    for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    std::vector<size_t> cursor(offsets.begin(), offsets.end() - 1);
    std::vector<PackedEdge> grouped(edges.size());
    for (const PackedEdge& edge : edges) grouped[cursor[edge.first >> 32]++] = edge;
    for (size_t v = 0; v < cursor.size(); ++v)
      std::sort(grouped.begin() + offsets[v], grouped.begin() + offsets[v + 1]);
    edges.swap(grouped);
  } else {
    std::sort(edges.begin(), edges.end());
  }
  for (size_t i = 0; i < edges.size();) {
    size_t j = i + 1;
    while (j < edges.size() && edges[j].first == edges[i].first) ++j;
    const int64_t x = edges[i].second, x_next = next_corner(x);
    const bool x_forward = triangles[x] < triangles[x_next];
    const int64_t a = static_cast<int64_t>(edges[i].first >> 32);
    const int64_t b = static_cast<int64_t>(edges[i].first & 0xffffffffULL);
    if (j - i != 2) {
      out.push_back({PHX_MC_SURFACE_EDGE, a, b, 0});
    } else {
      const int64_t y = edges[i+1].second, y_next = next_corner(y);
      const bool y_forward = triangles[y] < triangles[y_next];
      if (x_forward == y_forward) out.push_back({PHX_MC_SURFACE_EDGE, a, b, 0});
      const int64_t xa = x_forward ? x : x_next, xb = x_forward ? x_next : x;
      const int64_t ya = y_forward ? y : y_next, yb = y_forward ? y_next : y;
      parent[root(parent, xa)] = root(parent, ya);
      parent[root(parent, xb)] = root(parent, yb);
    }
    i = j;
  }
}

void topology(int64_t point_count, int64_t n, const int64_t* triangles,
              std::vector<Finding>& out) {
  std::vector<int64_t> parent(static_cast<size_t>(3 * n));
  std::iota(parent.begin(), parent.end(), int64_t{0});
  pair_packed_edges(point_count, n, triangles, parent, out);
  // Every corner at a vertex must be in one edge-connected component.
  // The point-indexed table avoids sorting a second bank of 3*n corners.
  std::vector<int64_t> first_root(static_cast<size_t>(point_count), -1);
  for (int64_t corner = 0; corner < 3*n; ++corner) {
    const int64_t vertex = triangles[corner];
    int64_t& first = first_root[vertex];
    if (first == -2) continue;
    const int64_t component = root(parent, corner);
    if (first == -1) first = component;
    else if (first != component) {
      out.push_back({PHX_MC_SURFACE_VERTEX_LINK, vertex, -1, 0});
      first = -2;
    }
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
    {
      // Stable radix passes preserve input order for equal Morton keys.
      std::vector<Item> scratch(items.size());
      for (int shift = 0; shift < 30; shift += 10) {
        std::array<size_t, 1024> offsets{};
        for (const Item& item : items) ++offsets[(item.morton >> shift) & 1023U];
        size_t total = 0;
        for (size_t& count : offsets) {
          const size_t size = count;
          count = total;
          total += size;
        }
        for (const Item& item : items) scratch[offsets[(item.morton >> shift) & 1023U]++] = item;
        items.swap(scratch);
      }
    }
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

// Coplanar pairs (chart partitions lifted to z = 0, flat surface regions) are decided by
// exact 2D sides in an axis projection where the triangle is nondegenerate; the
// projection preserves every side relation up to one common sign.
int projection_axis(const double* const t[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    const double a[2] = {t[0][u], t[0][v]}, b[2] = {t[1][u], t[1][v]}, c[2] = {t[2][u], t[2][v]};
    if (phx::mc::orient2d_filtered(a, b, c) != 0) return axis;
  }
  return -1;
}
int side2(int axis, const double* a, const double* b, const double* c) {
  const int u = (axis + 1) % 3, v = (axis + 2) % 3;
  const double aa[2] = {a[u], a[v]}, bb[2] = {b[u], b[v]}, cc[2] = {c[u], c[v]};
  return phx::mc::orient2d_filtered(aa, bb, cc);
}
// The line through o and x leaves `keep` on one side and every `other` strictly on the
// other: the two triangles then meet only on that line, i.e. nowhere or at o.
bool line_separates(int axis, const double* o, const double* x, const double* keep,
                    const double* const* other, int count) {
  const int side = side2(axis, o, x, keep);
  if (side == 0) return false;
  for (int k = 0; k < count; ++k)
    if (side2(axis, o, x, other[k]) != -side) return false;
  return true;
}
bool coplanar(const double* const p[3], const double* const q[3]) {
  for (int k = 0; k < 3; ++k)
    if (phx::mc::orient3d_filtered(p[0], p[1], p[2], q[k]) != 0) return false;
  return true;
}
// All six vertices have equal values (==, so -0.0 and +0.0 agree) in one coordinate
// (charts lifted to z = 0, axis-aligned flat regions): the pair is exactly coplanar and
// dropping that coordinate is an isometry, so exact 2D sides decide without the (always
// zero, exact-stage) orient3d calls.
int common_axis(const double* const p[3], const double* const q[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    const double value = p[0][axis];
    if (p[1][axis] == value && p[2][axis] == value && q[0][axis] == value &&
        q[1][axis] == value && q[2][axis] == value)
      return axis;
  }
  return -1;
}
bool clear_planar(int axis, const double* const p[3], const double* const q[3],
                  const bool shared_p[3], const bool shared_q[3], int shared) {
  if (shared == 2) {
    const double *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    for (int i = 0; i < 3; ++i) {
      if (shared_p[i]) { if (!a) a = p[i]; else b = p[i]; } else c = p[i];
      if (!shared_q[i]) d = q[i];
    }
    const int side = side2(axis, a, b, c);
    return side != 0 && side2(axis, a, b, d) == -side;
  }
  if (shared == 1) {
    const double *common = nullptr, *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    for (int i = 0; i < 3; ++i) {
      if (shared_p[i]) common = p[i]; else if (!a) a = p[i]; else b = p[i];
      if (!shared_q[i]) { if (!c) c = q[i]; else d = q[i]; }
    }
    const double* cd[2] = {c, d};
    const double* ab[2] = {a, b};
    return line_separates(axis, common, a, b, cd, 2) ||
           line_separates(axis, common, b, a, cd, 2) ||
           line_separates(axis, common, c, d, ab, 2) ||
           line_separates(axis, common, d, c, ab, 2);
  }
  if (shared == 0) {
    for (int i = 0; i < 3; ++i) {
      if (line_separates(axis, p[i], p[(i+1)%3], p[(i+2)%3], q, 3)) return true;
      if (line_separates(axis, q[i], q[(i+1)%3], q[(i+2)%3], p, 3)) return true;
    }
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
  const int flat = common_axis(p, q);
  if (flat >= 0) return clear_planar(flat, p, q, shared_p, shared_q, shared);
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
    const double *common = nullptr, *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
    for (int i = 0; i < 3; ++i) {
      if (shared_p[i]) common = p[i];
      else if (!a) a = p[i]; else b = p[i];
      if (!shared_q[i]) { if (!c) c = q[i]; else d = q[i]; }
    }
    const int s0 = phx::mc::orient3d_filtered(common, a, b, c);
    const int s1 = phx::mc::orient3d_filtered(common, a, b, d);
    if (s0 == 0 && s1 == 0) {
      const int axis = projection_axis(p);
      return axis >= 0 && clear_planar(axis, p, q, shared_p, shared_q, shared);
    }
    if (s0 == 0 || s1 == 0) return false;
    if (s0 == s1) return true;
    const int t0 = phx::mc::orient3d_filtered(common, c, d, a);
    const int t1 = phx::mc::orient3d_filtered(common, c, d, b);
    if (t0 == 0 || t1 == 0) return false;
    if (t0 == t1) return true;
    // Both opposite edges cross the other plane (s1 = -s0, t1 = -t0). Expanding
    // orient(a,b,c,d) about the common vertex gives a sum of a term of sign -s0 and a
    // term of sign -t0, so the former test "orient(a,b,c,d) != 0, != s0, != t0" holds
    // exactly when s0 == t0; that sign agreement is the certificate (cross-checked
    // against the full classifier by the randomized shared-vertex pair test).
    return s0 == t0;
  }
  if (shared == 0) {
    if (separate_triangle(p, q) || separate_triangle(q, p)) return true;
    if (!coplanar(p, q)) return false;
    const int axis = projection_axis(p);
    return axis >= 0 && clear_planar(axis, p, q, shared_p, shared_q, shared);
  }
  return false;
}

int32_t classify_pair(const double* points, const int64_t* triangles, int64_t a, int64_t b,
                      std::vector<phx::mc::SurfaceContact>& out) {
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
    out.push_back({a, b, contact.kind});
  return PHX_MC_OK;
}
}  // namespace

int32_t phx::mc::surface_embedding_contacts(const double* points, int64_t n,
                                            const int64_t* triangles,
                                            std::vector<SurfaceContact>& out) {
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

bool phx::mc::surface_embedding_pair_clear(const double* const first[3],
                                            const double* const second[3],
                                            const int64_t* first_ids,
                                            const int64_t* second_ids) {
  return clear_pair(first, second, first_ids, second_ids);
}

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
    // Edge keys pack two vertex ids into 64 bits.
    if (static_cast<uint64_t>(point_count) > uint64_t{1} << 32) return PHX_MC_INDEX_RANGE_ERROR;
    for (int64_t k = 0; k < 3*triangle_count; ++k)
      if (triangles[k] < 0 || triangles[k] >= point_count) return PHX_MC_INVALID_INPUT;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
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
        topology(point_count, triangle_count, triangles, result);
        std::vector<phx::mc::SurfaceContact> contacts;
        const int32_t status =
            phx::mc::surface_embedding_contacts(points, triangle_count, triangles, contacts);
        if (status != PHX_MC_OK) return status;
        for (const phx::mc::SurfaceContact& c : contacts)
          result.push_back({PHX_MC_SURFACE_CONTACT, c.first, c.second, c.kind});
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
