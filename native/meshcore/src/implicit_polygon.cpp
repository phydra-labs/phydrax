// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Bounded, allocation-free simple implicit contour batches. Ambiguous and
// numerically marginal cells are deliberately left to the canonical kernel.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "phydrax_meshcore.h"
#include "capi_guard.hpp"

namespace {
constexpr int edges[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},
                              {6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
constexpr int faces[6][4] = {{0,1,2,3},{4,5,6,7},{0,1,5,4},
                              {1,2,6,5},{2,3,7,6},{3,0,4,7}};
constexpr int offsets[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
struct Template { int count = 0; int segment[6][2] = {}; int adjacent[12][2] = {}; };
std::array<Template, 256> templates() {
  std::array<Template, 256> result{};
  for (int pattern = 0; pattern < 256; ++pattern) {
    Template t; int degree[12] = {}; bool valid = true;
    for (const auto& face : faces) {
      int crossing[4], nc = 0;
      for (int i = 0; i < 4; ++i) {
        int a = face[i], b = face[(i+1)%4];
        if (((pattern >> a) & 1) == ((pattern >> b) & 1)) continue;
        for (int e = 0; e < 12; ++e) {
          if ((edges[e][0] == a && edges[e][1] == b) ||
              (edges[e][0] == b && edges[e][1] == a)) { crossing[nc++] = e; break; }
        }
      }
      if (nc == 0) continue;
      if (nc != 2 || t.count == 6) { valid = false; break; }
      int a = crossing[0], b = crossing[1];
      if (degree[a] >= 2 || degree[b] >= 2) { valid = false; break; }
      t.segment[t.count][0] = a; t.segment[t.count++][1] = b;
      t.adjacent[a][degree[a]++] = b; t.adjacent[b][degree[b]++] = a;
    }
    if (!valid || t.count < 3) continue;
    int start = -1, active = 0;
    for (int e = 0; e < 12; ++e) if (degree[e]) {
      if (degree[e] != 2) valid = false;
      if (start < 0) start = e;
      ++active;
    }
    if (!valid || active != t.count) continue;
    int previous = -1, current = start, visited = 0;
    do {
      ++visited;
      int next = t.adjacent[current][0] == previous ? t.adjacent[current][1] : t.adjacent[current][0];
      previous = current; current = next;
    } while (current != start && visited <= active);
    if (current == start && visited == active) result[pattern] = t;
  }
  return result;
}
struct Vec { double x[3]; };
Vec sub(const Vec& a, const Vec& b) { return {{a.x[0]-b.x[0],a.x[1]-b.x[1],a.x[2]-b.x[2]}}; }
Vec cross(const Vec& a, const Vec& b) { return {{a.x[1]*b.x[2]-a.x[2]*b.x[1], a.x[2]*b.x[0]-a.x[0]*b.x[2], a.x[0]*b.x[1]-a.x[1]*b.x[0]}}; }
double dot(const Vec& a, const Vec& b) { return (a.x[0]*b.x[0]+a.x[1]*b.x[1])+a.x[2]*b.x[2]; }
double area(const Vec& a, const Vec& b, const Vec& c) {
  const Vec n = cross(sub(b,a), sub(c,a)); return 0.5*std::sqrt(dot(n,n));
}
// Return only decisions separated from projection/reduction rounding.  Any
// indeterminate ear is replayed by the canonical scalar implementation.
bool long_polygon(const Vec* p, const int64_t* ids, int k, double world_scale,
                  int triangles[4][3]) {
  const double eps = std::numeric_limits<double>::epsilon();
  Vec center{{0,0,0}}, normal{{0,0,0}}, centered[6];
  for (int i = 0; i < k; ++i)
    for (int a = 0; a < 3; ++a) center.x[a] += p[i].x[a];
  for (int a = 0; a < 3; ++a) center.x[a] /= k;
  double extent = 0;
  for (int i = 0; i < k; ++i) {
    centered[i] = sub(p[i],center);
    for (double x : centered[i].x) extent = std::max(extent,std::abs(x));
  }
  for (int i = 0; i < k; ++i) {
    Vec n = cross(centered[i],centered[(i+1)%k]);
    for (int a = 0; a < 3; ++a) normal.x[a] += n.x[a];
  }
  double length = std::sqrt(dot(normal,normal));
  if (!std::isfinite(length) || !std::isfinite(extent) || extent <= 0 ||
      length <= std::sqrt(std::numeric_limits<double>::min()) ||
      length <= 65536*eps*world_scale*extent) return false;
  for (double& x : normal.x) x /= length;
  double projection = dot(centered[0],normal);
  Vec reference;
  for (int a = 0; a < 3; ++a) reference.x[a] = centered[0].x[a]-normal.x[a]*projection;
  double ref_length = std::sqrt(dot(reference,reference));
  if (!std::isfinite(ref_length) ||
      ref_length <= std::sqrt(std::numeric_limits<double>::min()) ||
      ref_length <= 65536*eps*world_scale) return false;
  Vec first, second;
  for (int a = 0; a < 3; ++a) first.x[a] = reference.x[a]/ref_length;
  second = cross(normal,first);
  double xy[6][2], signed_area = 0;
  for (int i = 0; i < k; ++i) {
    xy[i][0] = dot(centered[i],first); xy[i][1] = dot(centered[i],second);
    if (!std::isfinite(xy[i][0]) || !std::isfinite(xy[i][1])) return false;
  }
  for (int i = 0; i < k; ++i)
    signed_area += xy[i][0]*xy[(i+1)%k][1]-xy[i][1]*xy[(i+1)%k][0];
  const double condition = 1.0 + extent*extent/length + extent/ref_length;
  const double margin = 1048576*eps*world_scale*extent*condition;
  if (!std::isfinite(signed_area) || !std::isfinite(margin) ||
      std::abs(signed_area) <= margin) return false;
  const double orientation = signed_area > 0 ? 1.0 : -1.0;
  auto side = [&](int a, int b, int c) {
    return orientation*((xy[b][0]-xy[a][0])*(xy[c][1]-xy[a][1]) -
                        (xy[b][1]-xy[a][1])*(xy[c][0]-xy[a][0]));
  };
  int remaining[6]; for (int i = 0; i < k; ++i) remaining[i] = i;
  int n = k, output = 0;
  while (n > 3) {
    int best = -1; double best_area = 0;
    for (int position = 0; position < n; ++position) {
      int previous = remaining[(position+n-1)%n], current = remaining[position],
          following = remaining[(position+1)%n];
      const double convex = side(previous,current,following);
      if (!std::isfinite(convex) || std::abs(convex) <= margin) return false;
      if (convex < 0) continue;
      bool contains = false;
      for (int j = 0; j < n; ++j) {
        int candidate = remaining[j];
        if (candidate == previous || candidate == current || candidate == following) continue;
        const double a = side(previous,current,candidate), b = side(current,following,candidate),
                     c = side(following,previous,candidate);
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) return false;
        if (a < -margin || b < -margin || c < -margin) continue;
        if (a <= margin || b <= margin || c <= margin) return false;
        contains = true; break;
      }
      if (contains) continue;
      double candidate_area = area(p[previous],p[current],p[following]);
      if (!std::isfinite(candidate_area) ||
          candidate_area <= std::sqrt(std::numeric_limits<double>::min())) return false;
      if (best >= 0 && std::abs(candidate_area-best_area) <=
          4096*eps*std::max(candidate_area,best_area)) return false;
      if (best < 0 || candidate_area > best_area ||
          (candidate_area == best_area && ids[current] > ids[remaining[best]])) {
        best = position; best_area = candidate_area;
      }
    }
    if (best < 0) return false;
    triangles[output][0] = remaining[(best+n-1)%n];
    triangles[output][1] = remaining[best];
    triangles[output++][2] = remaining[(best+1)%n];
    for (int i = best; i+1 < n; ++i) remaining[i] = remaining[i+1];
    --n;
  }
  for (int i = 0; i < 3; ++i) triangles[output][i] = remaining[i];
  return true;
}
}

// All arrays are caller-owned and contiguous. Roots index the shared root
// arrays; triangles contain encounter-ordered vertex IDs, never root IDs.
// Outputs are committed only after complete input validation.
extern "C" PHX_MC_API int32_t phx_mc_implicit_contours(
    int64_t count, int64_t root_count, const int8_t* signs, const double* values,
    const double* boxes, const int64_t* roots, const int64_t* vertices,
    const double* points, const int64_t* root_issues, double minimum_area,
    uint8_t* handled, int8_t* counts, int64_t* triangles, int64_t* issues) {
  using phx::mc::addressable;
  if (count < 0 || root_count < 0 || !addressable(count, 30, sizeof(int64_t)) ||
      !addressable(root_count, 3, sizeof(double)) || !std::isfinite(minimum_area) || minimum_area < 0 ||
      (count && (!signs || !values || !boxes || !roots || !handled || !counts || !triangles || !issues)) ||
      (root_count && (!vertices || !points || !root_issues))) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t i = 0; i < count*8; ++i)
    if (signs[i] != -1 && signs[i] != 1) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t i = 0; i < count*12; ++i) {
    const int64_t root = roots[i];
    if (root < -1 || root >= root_count || (root >= 0 && vertices[root] < 0))
      return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t row = 0; row < count; ++row) {
    for (int e = 0; e < 12; ++e) {
      const int64_t root = roots[row*12+e];
      const bool crossing = signs[row*8+edges[e][0]] != signs[row*8+edges[e][1]];
      if ((root >= 0) != crossing) return PHX_MC_INVALID_ARGUMENT;
      if (root >= 0) for (int prior = 0; prior < e; ++prior) {
        const int64_t other = roots[row*12+prior];
        if (other >= 0 && (root == other || vertices[root] == vertices[other]))
          return PHX_MC_INVALID_ARGUMENT;
      }
    }
  }
  static const auto table = templates();
  const double eps = std::numeric_limits<double>::epsilon();
  const double tiny = std::numeric_limits<double>::min();
  for (int64_t row = 0; row < count; ++row) {
    handled[row] = 0; counts[row] = 0; issues[row] = 0;
    int pattern = 0;
    for (int i = 0; i < 8; ++i) if (signs[row*8+i] > 0) pattern |= 1 << i;
    const Template& t = table[pattern]; const int k = t.count;
    if (!k) continue;
    const int64_t* r = roots + row*12;
    bool valid = true;
    for (int j = 0; j < k; ++j)
      for (int l = 0; l < 2; ++l) if (r[t.segment[j][l]] < 0) valid = false;
    if (!valid) continue;
    int first[6], second[6], best = 0;
    for (int j = 0; j < k; ++j) {
      int a = t.segment[j][0], b = t.segment[j][1];
      if (vertices[r[a]] > vertices[r[b]]) std::swap(a,b);
      first[j] = a; second[j] = b;
      if (j && (r[a] < r[first[best]] || (r[a] == r[first[best]] && r[b] < r[second[best]]))) best = j;
    }
    int cycle[6]; int a = first[best], b = second[best];
    cycle[0] = r[a] < r[b] ? a : b; cycle[1] = r[a] < r[b] ? b : a;
    for (int j = 2; j < k; ++j) {
      const int* adj = t.adjacent[cycle[j-1]];
      cycle[j] = adj[0] == cycle[j-2] ? adj[1] : adj[0];
    }
    Vec p[6], center{{0,0,0}}, gradient{{0,0,0}}, normal{{0,0,0}};
    double scale = tiny, grad_scale = 0;
    for (int j = 0; j < k; ++j) for (int axis = 0; axis < 3; ++axis) {
      p[j].x[axis] = points[r[cycle[j]]*3+axis];
      center.x[axis] += p[j].x[axis];
      scale = std::max(scale, std::abs(p[j].x[axis]));
      if (!std::isfinite(p[j].x[axis])) valid = false;
    }
    for (int axis = 0; axis < 3; ++axis) {
      center.x[axis] /= k;
      double low = 0, high = 0;
      for (int j = 0; j < 8; ++j) {
        if (offsets[j][axis]) high += values[row*8+j]; else low += values[row*8+j];
      }
      gradient.x[axis] = (high/4.0-low/4.0)/(boxes[row*6+3+axis]-boxes[row*6+axis]);
      grad_scale = std::max(grad_scale, std::abs(gradient.x[axis]));
      if (!std::isfinite(gradient.x[axis])) valid = false;
    }
    for (int j = 0; j < k; ++j) {
      Vec n = cross(sub(p[j],center),sub(p[(j+1)%k],center));
      for (int axis = 0; axis < 3; ++axis) normal.x[axis] += n.x[axis];
    }
    double orient = dot(normal,gradient), margin = ((4096*eps)*scale)*scale*grad_scale;
    if (!valid || !std::isfinite(orient) || !std::isfinite(margin) ||
        grad_scale <= std::sqrt(tiny) || std::abs(orient) <= margin) continue;
    if (orient < 0) { std::reverse(cycle,cycle+k); std::reverse(p,p+k); }
    constexpr int split1[2][3] = {{0,1,2},{0,2,3}};
    constexpr int split2[2][3] = {{0,1,3},{1,2,3}};
    const int (*split)[3] = split1;
    const double amargin = ((4096*eps)*scale)*scale;
    if (!std::isfinite(amargin)) continue;
    if (k == 4) {
      const double a0 = area(p[0],p[1],p[2]), a1 = area(p[0],p[2],p[3]);
      const double b0 = area(p[0],p[1],p[3]), b1 = area(p[1],p[2],p[3]);
      if (!std::isfinite(a0) || !std::isfinite(a1) ||
          !std::isfinite(b0) || !std::isfinite(b1)) continue;
      const double m1 = std::min(a0,a1), m2 = std::min(b0,b1);
      // Both implementations use the same coordinate differences and cross
      // products. Only the positive three-term norm reduction can differ.
      // Bound that rounding by the actual area rather than world-coordinate
      // scale; leave ties, subnormal norms and unresolved decisions to scalar.
      const double local_margin = (4096*eps)*std::max(m1,m2);
      if (std::min(m1,m2) <= std::sqrt(tiny) ||
          !std::isfinite(local_margin) || std::abs(m1-m2) <= local_margin) continue;
      if (m2 > m1) split = split2;
    } else if (k > 4) {
      int64_t ids[6];
      for (int j = 0; j < k; ++j) ids[j] = vertices[r[cycle[j]]];
      int long_split[4][3];
      if (!long_polygon(p,ids,k,scale,long_split)) continue;
      // Copy out before leaving this scope.
      int64_t output[4][3];
      for (int j = 0; j < k-2; ++j) {
        int i0 = long_split[j][0], i1 = long_split[j][1], i2 = long_split[j][2];
        Vec n = cross(sub(p[i1],p[i0]),sub(p[i2],p[i0]));
        double ar = 0.5*std::sqrt(dot(n,n)), d = dot(n,gradient), dm = amargin*grad_scale;
        if (!std::isfinite(ar) || ar <= std::sqrt(tiny) ||
            ar <= minimum_area+amargin || !std::isfinite(d) ||
            std::abs(d) <= dm) { valid = false; break; }
        if (d < 0) std::swap(i1,i2);
        output[j][0] = ids[i0]; output[j][1] = ids[i1]; output[j][2] = ids[i2];
      }
      if (!valid) continue;
      for (int j = 0; j < k-2; ++j) for (int l = 0; l < 3; ++l)
        triangles[row*30+j*3+l] = output[j][l];
      for (int j = 0; j < 12; ++j) if (r[j] >= 0) issues[row] |= root_issues[r[j]];
      handled[row] = 1; counts[row] = static_cast<int8_t>(k-2);
      continue;
    }
    int64_t output[2][3];
    for (int j = 0; j < k-2; ++j) {
      int i0 = split[j][0], i1 = split[j][1], i2 = split[j][2];
      Vec n = cross(sub(p[i1],p[i0]),sub(p[i2],p[i0]));
      double ar = 0.5*std::sqrt(dot(n,n)), d = dot(n,gradient), dm = amargin*grad_scale;
      if (!std::isfinite(ar) || ar <= std::sqrt(tiny) || ar <= minimum_area+amargin ||
          !std::isfinite(d) || std::abs(d) <= dm) { valid = false; break; }
      if (d < 0) std::swap(i1,i2);
      output[j][0] = vertices[r[cycle[i0]]]; output[j][1] = vertices[r[cycle[i1]]]; output[j][2] = vertices[r[cycle[i2]]];
    }
    if (!valid) continue;
    for (int j = 0; j < k-2; ++j) for (int l = 0; l < 3; ++l) triangles[row*30+j*3+l] = output[j][l];
    for (int j = 0; j < 12; ++j) if (r[j] >= 0) issues[row] |= root_issues[r[j]];
    handled[row] = 1; counts[row] = static_cast<int8_t>(k-2);
  }
  return PHX_MC_OK;
}
