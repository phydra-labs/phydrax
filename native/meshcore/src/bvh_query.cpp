//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Point traversals over one caller-built packed BVH. Formulas follow the JAX
// owners operation for operation (phydrax/_bvh.py bvh_nearest_items and
// bvh_hierarchical_sum, geometry/simplicial _closest_points_on_triangles and
// _solid_angles) so both routes take the same discrete decisions up to
// floating-point reduction order.
#include "bvh_query.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "capi_guard.hpp"

namespace phx::mc::bvh {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kQuietNaN = std::numeric_limits<double>::quiet_NaN();

inline double dot3(const double* u, const double* v) noexcept {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

inline void cross3(const double* u, const double* v, double* out) noexcept {
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

inline void subtract3(const double* u, const double* v, double* out) noexcept {
  out[0] = u[0] - v[0];
  out[1] = u[1] - v[1];
  out[2] = u[2] - v[2];
}

inline double squared_distance3(const double* u, const double* v) noexcept {
  double difference[3];
  subtract3(u, v, difference);
  return dot3(difference, difference);
}

// Squared distance from a point to a box (aabb_dist2).
inline double box_distance(const double* point, const double* lower, const double* upper) noexcept {
  double total = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double gap = std::max(0.0, std::max(lower[axis] - point[axis], point[axis] - upper[axis]));
    total += gap * gap;
  }
  return total;
}

inline bool box_contains(const double* point, const double* lower, const double* upper,
                         double padding) noexcept {
  for (int axis = 0; axis < 3; ++axis) {
    if (!(point[axis] >= lower[axis] - padding && point[axis] <= upper[axis] + padding)) return false;
  }
  return true;
}

struct Closest {
  double point[3];
  double squared;
  int8_t feature;
};

// Clamped projection onto the segment from start to end (_closest_segment);
// returns the clamped parameter.
inline double closest_on_segment(const double* point, const double* start, const double* end,
                                 double* out) noexcept {
  double direction[3];
  double offset[3];
  subtract3(end, start, direction);
  subtract3(point, start, offset);
  double parameter = dot3(offset, direction) / dot3(direction, direction);
  parameter = std::min(std::max(parameter, 0.0), 1.0);
  for (int axis = 0; axis < 3; ++axis) out[axis] = start[axis] + parameter * direction[axis];
  return parameter;
}

// Closest point of one triangle and its feature, as _closest_points_on_triangles.
Closest closest_on_triangle(const double* point, const double* a, const double* b,
                            const double* c) noexcept {
  double ab[3];
  double ac[3];
  double normal[3];
  subtract3(b, a, ab);
  subtract3(c, a, ac);
  cross3(ab, ac, normal);
  const double normal_squared = dot3(normal, normal);
  double offset[3];
  subtract3(point, a, offset);
  const double height = dot3(offset, normal) / normal_squared;
  double projection[3];
  for (int axis = 0; axis < 3; ++axis) projection[axis] = point[axis] - height * normal[axis];
  double v2[3];
  subtract3(projection, a, v2);
  const double d00 = dot3(ab, ab);
  const double d01 = dot3(ab, ac);
  const double d11 = dot3(ac, ac);
  const double d20 = dot3(v2, ab);
  const double d21 = dot3(v2, ac);
  const double denominator = d00 * d11 - d01 * d01;
  const double second = (d11 * d20 - d01 * d21) / denominator;
  const double third = (d00 * d21 - d01 * d20) / denominator;
  const double first = 1.0 - second - third;
  const bool inside = first >= 0.0 && second >= 0.0 && third >= 0.0;

  double candidates[4][3];
  std::copy(projection, projection + 3, candidates[0]);
  double parameters[4] = {0.0, 0.0, 0.0, 0.0};
  parameters[1] = closest_on_segment(point, a, b, candidates[1]);
  parameters[2] = closest_on_segment(point, b, c, candidates[2]);
  parameters[3] = closest_on_segment(point, c, a, candidates[3]);
  double squared[4];
  for (int slot = 0; slot < 4; ++slot) squared[slot] = squared_distance3(candidates[slot], point);
  if (!inside) squared[0] = kInfinity;
  // First minimum, NaN first as in argmin.
  int selected = 0;
  for (int slot = 1; slot < 4; ++slot) {
    if (std::isnan(squared[selected])) break;
    if (std::isnan(squared[slot]) || squared[slot] < squared[selected]) selected = slot;
  }
  Closest result{};
  std::copy(candidates[selected], candidates[selected] + 3, result.point);
  result.squared = squared[selected];
  if (selected == 0) {
    result.feature = 0;
  } else {
    // Edge (start, end) of slot s is corners (s - 1, s % 3); clamped ends are corners.
    const double parameter = parameters[selected];
    if (parameter == 0.0) {
      result.feature = static_cast<int8_t>(4 + (selected - 1));
    } else if (parameter == 1.0) {
      result.feature = static_cast<int8_t>(4 + selected % 3);
    } else {
      result.feature = static_cast<int8_t>(selected);
    }
  }
  return result;
}

struct Ranked {
  double squared;
  int32_t face;
  double point[3];
  int8_t feature;
};

// The k best (squared distance, face) of one query, ascending.
class NearestList {
 public:
  explicit NearestList(int32_t k) : k_(static_cast<std::size_t>(k)) { ranked_.reserve(k_ + 1); }

  void reset() noexcept { ranked_.clear(); }

  double worst() const noexcept { return ranked_.size() < k_ ? kInfinity : ranked_.back().squared; }

  void offer(const Ranked& candidate) {
    if (ranked_.size() == k_) {
      const Ranked& last = ranked_.back();
      if (!(candidate.squared < last.squared ||
            (candidate.squared == last.squared && candidate.face < last.face))) {
        return;
      }
      ranked_.pop_back();
    }
    auto position = ranked_.end();
    while (position != ranked_.begin()) {
      const Ranked& before = *(position - 1);
      if (before.squared < candidate.squared ||
          (before.squared == candidate.squared && before.face < candidate.face)) {
        break;
      }
      --position;
    }
    ranked_.insert(position, candidate);
  }

  const std::vector<Ranked>& ranked() const noexcept { return ranked_; }

 private:
  std::size_t k_;
  std::vector<Ranked> ranked_;
};

struct Pending {
  int64_t node;
  double bound;
};

// A lower bound of a triangle's squared distance, screening faces before the
// reference formula: the Voronoi-region closest point (Ericson, Real-Time
// Collision Detection 5.1.5) less a rounding allowance relative to the
// triangle's extent from the point. It only skips faces that cannot reach the
// current k-th distance, so it never changes a result beyond rounding ties.
inline double distance_screen(const double* p, const double* a, const double* b,
                              const double* c) noexcept {
  double ab[3];
  double ac[3];
  double ap[3];
  subtract3(b, a, ab);
  subtract3(c, a, ac);
  subtract3(p, a, ap);
  const double allowance = 0x1p-36 * (dot3(ap, ap) + dot3(ab, ab) + dot3(ac, ac));
  const double d1 = dot3(ab, ap);
  const double d2 = dot3(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return dot3(ap, ap) - allowance;
  double bp[3];
  subtract3(p, b, bp);
  const double d3 = dot3(ab, bp);
  const double d4 = dot3(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return dot3(bp, bp) - allowance;
  const double vc = d1 * d4 - d3 * d2;
  double q[3];
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    for (int axis = 0; axis < 3; ++axis) q[axis] = ap[axis] - v * ab[axis];
    return dot3(q, q) - allowance;
  }
  double cp[3];
  subtract3(p, c, cp);
  const double d5 = dot3(ab, cp);
  const double d6 = dot3(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return dot3(cp, cp) - allowance;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    for (int axis = 0; axis < 3; ++axis) q[axis] = ap[axis] - w * ac[axis];
    return dot3(q, q) - allowance;
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    for (int axis = 0; axis < 3; ++axis) q[axis] = bp[axis] - w * (c[axis] - b[axis]);
    return dot3(q, q) - allowance;
  }
  const double denominator = 1.0 / (va + vb + vc);
  const double v = vb * denominator;
  const double w = vc * denominator;
  for (int axis = 0; axis < 3; ++axis) q[axis] = ap[axis] - ab[axis] * v - ac[axis] * w;
  return dot3(q, q) - allowance;
}

// Depth-first branch and bound of _nearest_items_one: the nearer child is
// visited first and a node is opened while its box distance does not exceed
// the current k-th distance.
void nearest_one(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup, const double* point,
                 NearestList& list, std::vector<Pending>& stack) {
  list.reset();
  stack.clear();
  stack.push_back({0, box_distance(point, tree.node_min, tree.node_max)});
  while (!stack.empty()) {
    const Pending pending = stack.back();
    stack.pop_back();
    if (!(pending.bound <= list.worst())) continue;
    const int64_t node = pending.node;
    const int64_t leaf = tree.leaf[node];
    if (leaf < 0) {
      const int64_t left = tree.left[node];
      const int64_t right = tree.right[node];
      const double left_bound = box_distance(point, tree.node_min + 3 * left, tree.node_max + 3 * left);
      const double right_bound =
          box_distance(point, tree.node_min + 3 * right, tree.node_max + 3 * right);
      if (left_bound <= right_bound) {
        stack.push_back({right, right_bound});
        stack.push_back({left, left_bound});
      } else {
        stack.push_back({left, left_bound});
        stack.push_back({right, right_bound});
      }
      continue;
    }
    const int64_t* items = tree.leaf_items + leaf * tree.leaf_width;
    for (int64_t slot = 0; slot < tree.leaf_width; ++slot) {
      const int64_t item = items[slot];
      if (item < 0) continue;
      // The item box bounds its triangle's distance from below.
      const double worst = list.worst();
      if (box_distance(point, tree.item_min + 3 * item, tree.item_max + 3 * item) > worst) {
        continue;
      }
      const int32_t* face = soup.faces + 3 * item;
      const double* a = soup.vertices + 3 * static_cast<int64_t>(face[0]);
      const double* b = soup.vertices + 3 * static_cast<int64_t>(face[1]);
      const double* c = soup.vertices + 3 * static_cast<int64_t>(face[2]);
      if (worst < kInfinity && distance_screen(point, a, b, c) > worst) continue;
      const Closest closest = closest_on_triangle(point, a, b, c);
      if (!std::isfinite(closest.squared)) continue;
      Ranked candidate{closest.squared, static_cast<int32_t>(item), {}, closest.feature};
      std::copy(closest.point, closest.point + 3, candidate.point);
      list.offer(candidate);
    }
  }
}

// Sum of atan2(numerator, denominator) terms by complex multiplication with
// one atan2 at the end. Half-turn crossings are counted exactly from the
// signs of the factors and the product, so the sum is not reduced modulo 2 pi:
// sum = arg(product) + 2 pi turns. Factors are normalized to unit max-norm
// and the product is rescaled by powers of two, so neither can overflow.
class AngleSum {
 public:
  void add(double numerator, double denominator) noexcept {
    const double scale = std::max(std::fabs(numerator), std::fabs(denominator));
    if (!(scale > 0.0) || !std::isfinite(scale)) {
      // Coincident or nonfinite geometry: the term itself, as atan2 defines it.
      direct_ += std::atan2(numerator, denominator);
      return;
    }
    const double inverse = 1.0 / scale;
    const double w_re = denominator * inverse;
    const double w_im = numerator * inverse;
    const double re = re_ * w_re - im_ * w_im;
    const double im = re_ * w_im + im_ * w_re;
    const bool z_upper = !std::signbit(im_);
    const bool w_upper = !std::signbit(w_im);
    const bool upper = !std::signbit(im);
    if (z_upper && w_upper && !upper) ++turns_;
    if (!z_upper && !w_upper && upper) --turns_;
    re_ = re;
    im_ = im;
    if (std::fabs(re_) + std::fabs(im_) > 0x1p256) {
      re_ *= 0x1p-256;
      im_ *= 0x1p-256;
    }
  }

  double value() const noexcept {
    return std::atan2(im_, re_) + 2.0 * kPi * static_cast<double>(turns_) + direct_;
  }

 private:
  static constexpr double kPi = 3.14159265358979323846;
  double re_ = 1.0;
  double im_ = 0.0;
  int64_t turns_ = 0;
  double direct_ = 0.0;
};

// Adds the Van Oosterom-Strackee half solid angle atan2(numerator, denominator)
// of one triangle seen from a point (_solid_angles is twice this angle).
inline void add_half_solid_angle(AngleSum& sum, const double* point, const double* first,
                                 const double* second, const double* third) noexcept {
  double a[3];
  double b[3];
  double c[3];
  subtract3(first, point, a);
  subtract3(second, point, b);
  subtract3(third, point, c);
  const double length_a = std::sqrt(dot3(a, a));
  const double length_b = std::sqrt(dot3(b, b));
  const double length_c = std::sqrt(dot3(c, c));
  double bc[3];
  cross3(b, c, bc);
  const double numerator = dot3(a, bc);
  const double denominator = length_a * length_b * length_c + dot3(a, b) * length_c +
                             dot3(b, c) * length_a + dot3(c, a) * length_b;
  sum.add(numerator, denominator);
}

double leaf_solid_angle(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup, int64_t leaf,
                        const double* point) noexcept {
  const int64_t* items = tree.leaf_items + leaf * tree.leaf_width;
  AngleSum sum;
  for (int64_t slot = 0; slot < tree.leaf_width; ++slot) {
    const int64_t item = items[slot];
    if (item < 0) continue;
    const int32_t* face = soup.faces + 3 * item;
    add_half_solid_angle(sum, point, soup.vertices + 3 * static_cast<int64_t>(face[0]),
                         soup.vertices + 3 * static_cast<int64_t>(face[1]),
                         soup.vertices + 3 * static_cast<int64_t>(face[2]));
  }
  return 2.0 * sum.value();
}

// bvh_hierarchical_sum: `open` decides whether a node is replaced by its
// children (or summed exactly at a leaf); otherwise `far` is its contribution.
template <class Open, class Far>
double hierarchical_sum(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                        const double* point, std::vector<int64_t>& stack, Open&& open, Far&& far) {
  double total = 0.0;
  stack.clear();
  stack.push_back(0);
  while (!stack.empty()) {
    const int64_t node = stack.back();
    stack.pop_back();
    if (!open(node)) {
      total += far(node);
      continue;
    }
    const int64_t leaf = tree.leaf[node];
    if (leaf >= 0) {
      total += leaf_solid_angle(tree, soup, leaf, point);
      continue;
    }
    stack.push_back(tree.right[node]);
    stack.push_back(tree.left[node]);
  }
  return total;
}

constexpr double kFourPi = 4.0 * 3.14159265358979323846;

// Morton order of the queries over their finite bounding box. Queries are
// independent, so the order changes no result; consecutive nearby queries
// reuse the cached tree path and geometry.
std::vector<int64_t> spatial_order(const double* points, int64_t count) {
  double lower[3] = {kInfinity, kInfinity, kInfinity};
  double upper[3] = {-kInfinity, -kInfinity, -kInfinity};
  for (int64_t query = 0; query < count; ++query) {
    for (int axis = 0; axis < 3; ++axis) {
      const double value = points[3 * query + axis];
      if (!std::isfinite(value)) continue;
      lower[axis] = std::min(lower[axis], value);
      upper[axis] = std::max(upper[axis], value);
    }
  }
  constexpr double kCells = 2097151.0;  // 2^21 - 1 per axis
  const auto spread = [](uint64_t value) {
    value &= 0x1fffffULL;
    value = (value | (value << 32)) & 0x1f00000000ffffULL;
    value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
    value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
    value = (value | (value << 2)) & 0x1249249249249249ULL;
    return value;
  };
  std::vector<std::pair<uint64_t, int64_t>> keyed(static_cast<std::size_t>(count));
  for (int64_t query = 0; query < count; ++query) {
    uint64_t key = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double extent = upper[axis] - lower[axis];
      const double value = points[3 * query + axis];
      double scaled = extent > 0.0 ? (value - lower[axis]) / extent * kCells : 0.0;
      if (!(scaled >= 0.0)) scaled = 0.0;  // nonfinite coordinates sort first
      key |= spread(static_cast<uint64_t>(std::min(scaled, kCells))) << axis;
    }
    keyed[static_cast<std::size_t>(query)] = {key, query};
  }
  std::sort(keyed.begin(), keyed.end());
  std::vector<int64_t> order(keyed.size());
  for (std::size_t slot = 0; slot < keyed.size(); ++slot) order[slot] = keyed[slot].second;
  return order;
}

}  // namespace

bool tree_valid(const phx_mc_bvh_tree_view* tree, int32_t dimension) {
  if (tree == nullptr || tree->node_count < 1 || tree->leaf_count < 0 || tree->leaf_width < 1 ||
      tree->item_count < 0 || tree->node_min == nullptr || tree->node_max == nullptr ||
      tree->node_size == nullptr || tree->left == nullptr || tree->right == nullptr ||
      tree->leaf == nullptr || (tree->leaf_count > 0 && tree->leaf_items == nullptr) ||
      (tree->item_count > 0 && (tree->item_min == nullptr || tree->item_max == nullptr)) ||
      !addressable(tree->node_count, dimension, sizeof(double)) ||
      !addressable(tree->leaf_count, tree->leaf_width, sizeof(int64_t)) ||
      !addressable(tree->item_count, dimension, sizeof(double))) {
    return false;
  }
  for (int64_t node = 0; node < tree->node_count; ++node) {
    const int64_t leaf = tree->leaf[node];
    if (leaf >= tree->leaf_count) return false;
    if (leaf < 0 && (tree->left[node] < 0 || tree->left[node] >= tree->node_count ||
                     tree->right[node] < 0 || tree->right[node] >= tree->node_count)) {
      return false;
    }
  }
  // A rooted tree has exactly one incoming edge per reachable non-root node.
  // Mark on discovery: this rejects cycles and shared children without recursion.
  std::vector<uint8_t> seen(static_cast<std::size_t>(tree->node_count), 0);
  std::vector<int64_t> pending{0};
  seen[0] = 1;
  while (!pending.empty()) {
    const int64_t node = pending.back();
    pending.pop_back();
    if (tree->leaf[node] >= 0) {
      if (tree->left[node] != -1 || tree->right[node] != -1) return false;
      continue;
    }
    for (const int64_t child : {tree->left[node], tree->right[node]}) {
      if (seen[static_cast<std::size_t>(child)]) return false;
      seen[static_cast<std::size_t>(child)] = 1;
      pending.push_back(child);
    }
  }
  const int64_t entries = tree->leaf_count * tree->leaf_width;
  for (int64_t entry = 0; entry < entries; ++entry) {
    if (tree->leaf_items[entry] >= tree->item_count) return false;
  }
  return true;
}

void triangle_nearest(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup, int64_t count,
                      const double* points, int32_t k, const NearestOutput& output) {
  NearestList list(k);
  std::vector<Pending> stack;
  stack.reserve(128);
  const std::size_t width = static_cast<std::size_t>(k);
  for (const int64_t query : spatial_order(points, count)) {
    nearest_one(tree, soup, points + 3 * query, list, stack);
    const std::vector<Ranked>& ranked = list.ranked();
    const std::size_t row = static_cast<std::size_t>(query) * width;
    for (std::size_t slot = 0; slot < width; ++slot) {
      double* point = output.closest + 3 * (row + slot);
      if (slot < ranked.size()) {
        output.faces[row + slot] = ranked[slot].face;
        output.distance_squared[row + slot] = ranked[slot].squared;
        output.features[row + slot] = ranked[slot].feature;
        std::copy(ranked[slot].point, ranked[slot].point + 3, point);
      } else {
        output.faces[row + slot] = -1;
        output.distance_squared[row + slot] = kInfinity;
        output.features[row + slot] = -1;
        std::fill(point, point + 3, kQuietNaN);
      }
    }
  }
}

void triangle_signed_distance(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                              const double* feature_normals, int64_t count, const double* points,
                              double* distance, int32_t* faces, double* closest) {
  NearestList list(1);
  std::vector<Pending> stack;
  stack.reserve(128);
  for (const int64_t query : spatial_order(points, count)) {
    const double* point = points + 3 * query;
    nearest_one(tree, soup, point, list, stack);
    double* out = closest + 3 * query;
    if (list.ranked().empty()) {
      distance[query] = kQuietNaN;
      faces[query] = -1;
      std::fill(out, out + 3, kQuietNaN);
      continue;
    }
    const Ranked& best = list.ranked().front();
    double away[3];
    subtract3(point, best.point, away);
    const double* normal =
        feature_normals + 3 * (static_cast<int64_t>(best.face) * PHX_MC_BVH_FEATURE_COUNT + best.feature);
    const double magnitude = std::sqrt(best.squared);
    distance[query] = dot3(away, normal) < 0.0 ? -magnitude : magnitude;
    faces[query] = best.face;
    std::copy(best.point, best.point + 3, out);
  }
}

void triangle_winding_exact(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                            const ClosingFans& fans, int64_t count, const double* points,
                            double* values) {
  std::vector<int64_t> stack;
  stack.reserve(128);
  for (const int64_t query : spatial_order(points, count)) {
    const double* point = points + 3 * query;
    const auto open = [&](int64_t node) {
      return box_contains(point, tree.node_min + 3 * node, tree.node_max + 3 * node, 0.0);
    };
    // The closing fan of a node's boundary from its box center reproduces the
    // patch's solid angle at every point outside the box.
    const auto far = [&](int64_t node) {
      const double* lower = tree.node_min + 3 * node;
      const double* upper = tree.node_max + 3 * node;
      const double apex[3] = {0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]),
                              0.5 * (lower[2] + upper[2])};
      AngleSum sum;
      for (int64_t edge = fans.offsets[node]; edge < fans.offsets[node + 1]; ++edge) {
        add_half_solid_angle(sum, point, apex,
                             soup.vertices + 3 * static_cast<int64_t>(fans.edges[2 * edge]),
                             soup.vertices + 3 * static_cast<int64_t>(fans.edges[2 * edge + 1]));
      }
      return 2.0 * sum.value();
    };
    values[query] = hierarchical_sum(tree, soup, point, stack, open, far) / kFourPi;
  }
}

void triangle_winding_dipole(const phx_mc_bvh_tree_view& tree, const TriangleSoup& soup,
                             const double* dipoles, double opening_angle, int64_t count,
                             const double* points, double* values) {
  // One cache-friendly record per node: everything the cut test, the far
  // value, and the descent read.
  struct Node {
    double normal[3];
    double center[3];
    double reach;
    int64_t left;
    int64_t right;
    int64_t leaf;
  };
  std::vector<Node> nodes(static_cast<std::size_t>(tree.node_count));
  for (int64_t node = 0; node < tree.node_count; ++node) {
    const double* dipole = dipoles + 7 * node;
    Node& record = nodes[static_cast<std::size_t>(node)];
    std::copy(dipole, dipole + 3, record.normal);
    std::copy(dipole + 3, dipole + 6, record.center);
    record.reach = opening_angle * dipole[6];
    record.left = tree.left[node];
    record.right = tree.right[node];
    record.leaf = tree.leaf[node];
  }
  std::vector<int64_t> stack;
  stack.reserve(128);
  for (const int64_t query : spatial_order(points, count)) {
    const double* point = points + 3 * query;
    double total = 0.0;
    stack.clear();
    stack.push_back(0);
    while (!stack.empty()) {
      const Node& node = nodes[static_cast<std::size_t>(stack.back())];
      stack.pop_back();
      double offset[3];
      subtract3(node.center, point, offset);
      const double distance = std::sqrt(dot3(offset, offset));
      if (!(distance <= node.reach)) {
        total += dot3(node.normal, offset) / (distance * distance * distance);
      } else if (node.leaf >= 0) {
        total += leaf_solid_angle(tree, soup, node.leaf, point);
      } else {
        stack.push_back(node.right);
        stack.push_back(node.left);
      }
    }
    values[query] = total / kFourPi;
  }
}

namespace {

struct Accepted {
  int32_t cell;
  double reference[3];
};

// One record per node with everything the containment descent reads.
struct BoxNode {
  double lower[3];
  double upper[3];
  int64_t left;
  int64_t right;
  int64_t leaf;
};

std::vector<BoxNode> box_nodes(const phx_mc_bvh_tree_view& tree) {
  std::vector<BoxNode> nodes(static_cast<std::size_t>(tree.node_count));
  for (int64_t node = 0; node < tree.node_count; ++node) {
    BoxNode& record = nodes[static_cast<std::size_t>(node)];
    std::copy(tree.node_min + 3 * node, tree.node_min + 3 * node + 3, record.lower);
    std::copy(tree.node_max + 3 * node, tree.node_max + 3 * node + 3, record.upper);
    record.left = tree.left[node];
    record.right = tree.right[node];
    record.leaf = tree.leaf[node];
  }
  return nodes;
}

// Locates one finite point; returns its status.
int32_t locate_one(const phx_mc_bvh_tree_view& tree, const std::vector<BoxNode>& nodes,
                   const double* vertices, const int32_t* cells, const uint8_t* cell_mask,
                   double tolerance, const double* point, std::vector<int64_t>& stack,
                   std::vector<Accepted>& accepted) {
  accepted.clear();
  bool candidate_seen = false;
  bool valid_seen = false;
  stack.clear();
  stack.push_back(0);
  while (!stack.empty()) {
    const BoxNode& node = nodes[static_cast<std::size_t>(stack.back())];
    stack.pop_back();
    if (!box_contains(point, node.lower, node.upper, tolerance)) continue;
    const int64_t leaf = node.leaf;
    if (leaf < 0) {
      stack.push_back(node.right);
      stack.push_back(node.left);
      continue;
    }
    // Screen every slot's box first without branching on earlier results, so
    // the item box loads of a leaf overlap; then resolve the hits.
    const int64_t* items = tree.leaf_items + leaf * tree.leaf_width;
    for (int64_t first = 0; first < tree.leaf_width; first += 32) {
      const int64_t last = std::min<int64_t>(tree.leaf_width, first + 32);
      uint32_t hits = 0;
      for (int64_t slot = first; slot < last; ++slot) {
        const int64_t item = std::max<int64_t>(items[slot], 0);
        const double* lower = tree.item_min + 3 * item;
        const double* upper = tree.item_max + 3 * item;
        const bool hit = (items[slot] >= 0) & (point[0] >= lower[0] - tolerance) &
                         (point[1] >= lower[1] - tolerance) & (point[2] >= lower[2] - tolerance) &
                         (point[0] <= upper[0] + tolerance) & (point[1] <= upper[1] + tolerance) &
                         (point[2] <= upper[2] + tolerance);
        hits |= static_cast<uint32_t>(hit) << (slot - first);
      }
      for (; hits != 0; hits &= hits - 1) {
        const int64_t item = items[first + std::countr_zero(hits)];
        if (cell_mask != nullptr && cell_mask[item] == 0) continue;
        candidate_seen = true;
        const int32_t* cell = cells + 4 * item;
        const double* origin = vertices + 3 * static_cast<int64_t>(cell[0]);
        double e1[3];
        double e2[3];
        double e3[3];
        double q[3];
        subtract3(vertices + 3 * static_cast<int64_t>(cell[1]), origin, e1);
        subtract3(vertices + 3 * static_cast<int64_t>(cell[2]), origin, e2);
        subtract3(vertices + 3 * static_cast<int64_t>(cell[3]), origin, e3);
        double c23[3];
        cross3(e2, e3, c23);
        const double determinant = dot3(e1, c23);
        if (!(std::isfinite(determinant) && determinant > 0.0)) continue;
        valid_seen = true;
        subtract3(point, origin, q);
        // Signed-volume ratios: r_i is the volume with q in place of e_i over det J.
        double c31[3];
        double c12[3];
        cross3(e3, e1, c31);
        cross3(e1, e2, c12);
        const double r1 = dot3(q, c23) / determinant;
        const double r2 = dot3(q, c31) / determinant;
        const double r3 = dot3(q, c12) / determinant;
        if (r1 >= -tolerance && r2 >= -tolerance && r3 >= -tolerance &&
            r1 + r2 + r3 <= 1.0 + tolerance) {
          accepted.push_back({static_cast<int32_t>(item), {r1, r2, r3}});
        }
      }
    }
  }
  if (!accepted.empty()) return PHX_MC_LOCATE_LOCATED;
  return candidate_seen && !valid_seen ? PHX_MC_LOCATE_DEGENERATE_CELL : PHX_MC_LOCATE_OUTSIDE;
}

// Residual |x(r) - point| and Frobenius condition of one cell's affine map.
void cell_evidence(const double* vertices, const int32_t* cell, const double* reference,
                   const double* point, double& residual, double& condition) noexcept {
  const double* origin = vertices + 3 * static_cast<int64_t>(cell[0]);
  double e[3][3];
  for (int column = 0; column < 3; ++column) {
    subtract3(vertices + 3 * static_cast<int64_t>(cell[column + 1]), origin, e[column]);
  }
  double mapped[3];
  for (int axis = 0; axis < 3; ++axis) {
    mapped[axis] = origin[axis] + e[0][axis] * reference[0] + e[1][axis] * reference[1] +
                   e[2][axis] * reference[2];
  }
  residual = std::sqrt(squared_distance3(mapped, point));
  double rows[3][3];
  cross3(e[1], e[2], rows[0]);
  cross3(e[2], e[0], rows[1]);
  cross3(e[0], e[1], rows[2]);
  const double determinant = dot3(e[0], rows[0]);
  const double jacobian = std::sqrt(dot3(e[0], e[0]) + dot3(e[1], e[1]) + dot3(e[2], e[2]));
  const double adjugate = std::sqrt(dot3(rows[0], rows[0]) + dot3(rows[1], rows[1]) + dot3(rows[2], rows[2]));
  condition = jacobian * (adjugate / determinant);
}

}  // namespace

void tetrahedron_locate(const phx_mc_bvh_tree_view& tree, const double* vertices,
                        const int32_t* cells, const uint8_t* cell_mask, double tolerance,
                        int64_t count, const double* points, int32_t capacity,
                        const LocateOutput& output) {
  const std::vector<BoxNode> nodes = box_nodes(tree);
  std::vector<int64_t> stack;
  stack.reserve(128);
  std::vector<Accepted> accepted;
  const std::size_t width = static_cast<std::size_t>(capacity);
  for (const int64_t query : spatial_order(points, count)) {
    const double* point = points + 3 * query;
    const std::size_t row = static_cast<std::size_t>(query) * width;
    int32_t* candidates = output.candidates + row;
    double* candidate_reference = output.candidate_reference + 3 * row;
    std::fill(candidates, candidates + width, -1);
    std::fill(candidate_reference, candidate_reference + 3 * width, 0.0);
    double* reference = output.reference + 3 * query;
    const bool finite =
        std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
    const int32_t status = finite ? locate_one(tree, nodes, vertices, cells, cell_mask, tolerance,
                                               point, stack, accepted)
                                  : PHX_MC_LOCATE_NONFINITE;
    output.status[query] = status;
    if (status != PHX_MC_LOCATE_LOCATED) {
      output.cells[query] = -1;
      output.containing[query] = 0;
      std::fill(reference, reference + 3, 0.0);
      output.residual[query] = kInfinity;
      output.condition[query] = kInfinity;
      continue;
    }
    std::sort(accepted.begin(), accepted.end(),
              [](const Accepted& a, const Accepted& b) { return a.cell < b.cell; });
    output.containing[query] = static_cast<int32_t>(accepted.size());
    const std::size_t kept = std::min(width, accepted.size());
    for (std::size_t slot = 0; slot < kept; ++slot) {
      candidates[slot] = accepted[slot].cell;
      std::copy(accepted[slot].reference, accepted[slot].reference + 3, candidate_reference + 3 * slot);
    }
    const Accepted& lowest = accepted.front();
    output.cells[query] = lowest.cell;
    std::copy(lowest.reference, lowest.reference + 3, reference);
    cell_evidence(vertices, cells + 4 * static_cast<int64_t>(lowest.cell), lowest.reference, point,
                  output.residual[query], output.condition[query]);
  }
}

}  // namespace phx::mc::bvh
