// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Exact planar partition decisions over common-denominator integer coordinates.
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "dyadic.hpp"
#include "phydrax_meshcore.h"
#include "wide_integer.hpp"

// All state is operation-local. The BVHs and candidate lists use O(n) rows;
// cached exact vertices and edges use O(n B) bits for B-bit input integers.
// Each clipped triangle has at most six vertices and intermediates have O(B)
// bits; a k-term unreduced area sum has O(k B) bits. Checked inline predicates
// fall back per pair to unbounded integers without changing their result.
namespace {
using phx::mc::Dyadic;
using Fast = phx::mc::Int256;

struct FastPoint { Fast x, y; };
struct FastLine { Fast a, b, c; };

struct Point { Dyadic x, y, w; };
struct Triangle {
  std::array<Point, 3> p;
  std::array<Point, 3> lines;
  Dyadic oriented_area;
  std::array<FastPoint, 3> fast_points;
  std::array<FastLine, 3> fast_lines;
  bool fast_valid = false;
};
struct Rational { Dyadic n, d; };
struct Box { double lo[2], hi[2]; };

Dyadic integer_one() {
  const std::uint32_t word = 1;
  return Dyadic::integer(&word, 1, 1);
}
Dyadic absolute(Dyadic value) {
  if (value.sign() < 0) value.negate();
  return value;
}
Rational add(const Rational& a, const Rational& b) {
  return {a.n * b.d + b.n * a.d, a.d * b.d};
}
bool equal(const Rational& a, const Rational& b) {
  return (a.n * b.d - b.n * a.d).is_zero();
}
Point cross(const Point& a, const Point& b) {
  return {a.y * b.w - a.w * b.y, a.w * b.x - a.x * b.w,
          a.x * b.y - a.y * b.x};
}
Dyadic dot(const Point& a, const Point& b) {
  return a.x * b.x + a.y * b.y + a.w * b.w;
}
// The common coordinate denominator is positive, so the sign of each
// affine edge predicate is the sign of its integer numerator. Checked inline
// arithmetic is only a representation choice: any overflow retries this
// pair with the unbounded homogeneous predicate below.
bool fast_sub(const Fast& a, const Fast& b, Fast& out) {
  const Fast negated = phx::mc::wrapping_negate(b);
  if (b.negative() && negated.negative()) return false;
  return phx::mc::checked_add(a, negated, out);
}

bool fast_load(const uint32_t* words, int64_t start, int64_t stop, int sign, Fast& out) {
  if (stop - start > 8) return false;
  Fast value;
  for (int64_t j = start; j < stop; ++j) {
    const auto index = static_cast<std::size_t>(j - start);
    value.limb[index / 2] |= static_cast<uint64_t>(words[j]) << (32 * (index % 2));
  }
  if (value.negative()) return false;
  out = sign < 0 ? phx::mc::wrapping_negate(value) : value;
  return true;
}

bool fast_line(const FastPoint& p, const FastPoint& q, FastLine& line) {
  Fast first, second;
  return fast_sub(p.y, q.y, line.a) && fast_sub(q.x, p.x, line.b) &&
         phx::mc::checked_multiply(p.x, q.y, first) &&
         phx::mc::checked_multiply(q.x, p.y, second) && fast_sub(first, second, line.c);
}

bool fast_value(const FastLine& line, const FastPoint& point, int& sign) {
  Fast first, second, sum, value;
  if (!phx::mc::checked_multiply(line.a, point.x, first) ||
      !phx::mc::checked_multiply(line.b, point.y, second) ||
      !phx::mc::checked_add(first, second, sum) ||
      !phx::mc::checked_add(sum, line.c, value)) return false;
  sign = value.zero() ? 0 : value.negative() ? -1 : 1;
  return true;
}

bool fast_directed(const Triangle& a, const Triangle& b, bool& result) {
  const int orientation_sign = a.oriented_area.sign();
  if (orientation_sign == 0) { result = true; return true; }
  for (int edge = 0; edge < 3; ++edge) {
    bool all_outside = true;
    for (const FastPoint& point : b.fast_points) {
      int sign = 0;
      if (!fast_value(a.fast_lines[edge], point, sign)) return false;
      if (sign * orientation_sign > 0) { all_outside = false; break; }
    }
    if (all_outside) { result = true; return true; }
  }
  result = false;
  return true;
}
const Dyadic& orientation(const Triangle& t) { return t.oriented_area; }
bool same_point(const Point& a, const Point& b) {
  return (a.x * b.w - b.x * a.w).is_zero() &&
         (a.y * b.w - b.y * a.w).is_zero();
}
Point crossing(const Point& a, const Point& b, const Dyadic& av, const Dyadic& bv) {
  Point result{av * b.x - bv * a.x, av * b.y - bv * a.y, av * b.w - bv * a.w};
  if (result.w.sign() < 0) { result.x.negate(); result.y.negate(); result.w.negate(); }
  return result;
}
bool separated_directed(const Triangle& a, const Triangle& b) {
  const int orientation_sign = orientation(a).sign();
  if (orientation_sign == 0) return true;
  for (int edge = 0; edge < 3; ++edge) {
    const Point& line = a.lines[edge];
    bool all_outside = true;
    for (const Point& point : b.p) {
      if (dot(line, point).sign() * orientation_sign > 0) { all_outside = false; break; }
    }
    if (all_outside) return true;
  }
  return false;
}
bool separated(const Triangle& a, const Triangle& b) {
  if (a.fast_valid && b.fast_valid) {
    bool first = false, second = false;
    if (fast_directed(a, b, first) &&
        (first || fast_directed(b, a, second))) return first || second;
  }
  return separated_directed(a, b) || separated_directed(b, a);
}
Rational polygon_area(const std::vector<Point>& points) {
  Rational area{Dyadic(), integer_one()};
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Point& a = points[i];
    const Point& b = points[(i + 1) % points.size()];
    area = add(area, {a.x * b.y - a.y * b.x, a.w * b.w});
  }
  area.n = absolute(std::move(area.n));
  return area;
}
Rational intersection_area(const Triangle& subject, const Triangle& clip, bool& singular) {
  std::vector<Point> polygon(subject.p.begin(), subject.p.end());
  const int sign = orientation(clip).sign();
  if (sign == 0) return {Dyadic(), integer_one()};
  for (int edge = 0; edge < 3 && !polygon.empty(); ++edge) {
    Point line = clip.lines[edge];
    if (sign < 0) { line.x.negate(); line.y.negate(); line.w.negate(); }
    std::vector<Point> output;
    Point previous = polygon.back();
    Dyadic previous_value = dot(line, previous);
    for (const Point& current : polygon) {
      Dyadic current_value = dot(line, current);
      if ((previous_value.sign() >= 0) != (current_value.sign() >= 0)) {
        Point point = crossing(previous, current, previous_value, current_value);
        if (point.w.is_zero()) { singular = true; return {Dyadic(), integer_one()}; }
        if (output.empty() || !same_point(output.back(), point)) output.push_back(std::move(point));
      }
      if (current_value.sign() >= 0 && (output.empty() || !same_point(output.back(), current))) {
        output.push_back(current);
      }
      previous = current;
      previous_value = std::move(current_value);
    }
    if (output.size() > 1 && same_point(output.front(), output.back())) output.pop_back();
    polygon = std::move(output);
  }
  return polygon_area(polygon);
}
bool boxes_meet(const Box& a, const Box& b) {
  return a.lo[0] <= b.hi[0] && b.lo[0] <= a.hi[0] &&
         a.lo[1] <= b.hi[1] && b.lo[1] <= a.hi[1];
}
struct Node { Box box; std::size_t begin, end, left, right; };
class Tree {
 public:
  explicit Tree(const std::vector<Box>& boxes) : boxes_(boxes), order_(boxes.size()) {
    std::iota(order_.begin(), order_.end(), std::size_t{0});
    if (!boxes.empty()) build(0, boxes.size(), 0);
  }
  std::vector<std::size_t> candidates(const Box& box) const {
    std::vector<std::size_t> out;
    if (nodes_.empty()) return out;
    std::vector<std::size_t> stack{0};
    while (!stack.empty()) {
      const Node& node = nodes_[stack.back()]; stack.pop_back();
      if (!boxes_meet(box, node.box)) continue;
      if (node.end - node.begin <= 8) {
        for (std::size_t i = node.begin; i < node.end; ++i)
          if (boxes_meet(box, boxes_[order_[i]])) out.push_back(order_[i]);
      } else { stack.push_back(node.left); stack.push_back(node.right); }
    }
    std::sort(out.begin(), out.end());
    return out;
  }
 private:
  std::size_t build(std::size_t begin, std::size_t end, int axis) {
    Box bounds = boxes_[order_[begin]];
    for (std::size_t i = begin + 1; i < end; ++i) {
      for (int k = 0; k < 2; ++k) {
        bounds.lo[k] = std::min(bounds.lo[k], boxes_[order_[i]].lo[k]);
        bounds.hi[k] = std::max(bounds.hi[k], boxes_[order_[i]].hi[k]);
      }
    }
    const std::size_t index = nodes_.size();
    nodes_.push_back({bounds, begin, end, 0, 0});
    if (end - begin > 8) {
      const auto first = order_.begin() + static_cast<std::ptrdiff_t>(begin);
      const auto last = order_.begin() + static_cast<std::ptrdiff_t>(end);
      std::sort(first, last, [&](std::size_t a, std::size_t b) {
        return boxes_[a].lo[axis] < boxes_[b].lo[axis] ||
               (boxes_[a].lo[axis] == boxes_[b].lo[axis] && a < b);
      });
      const std::size_t middle = begin + (end - begin) / 2;
      const std::size_t left = build(begin, middle, 1 - axis);
      const std::size_t right = build(middle, end, 1 - axis);
      nodes_[index].left = left; nodes_[index].right = right;
    }
    return index;
  }
  const std::vector<Box>& boxes_;
  std::vector<std::size_t> order_;
  std::vector<Node> nodes_;
};
}  // namespace

extern "C" int32_t phx_mc_planar_partition(
    int64_t target_count, int64_t declared_count, int64_t word_count,
    const uint32_t* words, const int64_t* offsets, const int8_t* signs,
    const double* lower, const double* upper, int32_t* verdict,
    int64_t* first, int64_t* second) {
  return phx::mc::guarded([&]() -> int32_t {
    if (target_count < 0 || declared_count < 0 ||
        target_count > INT64_MAX - declared_count || word_count < 0 ||
        offsets == nullptr || verdict == nullptr || first == nullptr || second == nullptr) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    const int64_t count = target_count + declared_count;
    if (!phx::mc::addressable(count, 6, sizeof(int64_t)) ||
        (count != 0 && (signs == nullptr || lower == nullptr || upper == nullptr)) ||
        (word_count != 0 && words == nullptr)) return PHX_MC_INVALID_ARGUMENT;
    const int64_t coordinates = count * 6;
    if (offsets[0] != 0 || offsets[coordinates + 1] != word_count) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i <= coordinates; ++i) {
      if (offsets[i] < 0 || offsets[i] > offsets[i + 1] || offsets[i + 1] > word_count)
        return PHX_MC_INVALID_ARGUMENT;
    }
    const auto value = [&](int64_t index, int sign) {
      const int64_t start = offsets[index];
      const int64_t length = offsets[index + 1] - start;
      return Dyadic::integer(length == 0 ? nullptr : words + start,
                             static_cast<std::size_t>(length), sign);
    };
    Dyadic denominator = value(coordinates, 1);
    if (denominator.sign() <= 0) return PHX_MC_INVALID_ARGUMENT;
    std::vector<Triangle> triangles;
    std::vector<Box> target_boxes, declared_boxes;
    triangles.reserve(static_cast<std::size_t>(count));
    for (int64_t row = 0; row < count; ++row) {
      Triangle triangle;
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t index = row * 6 + corner * 2;
        if ((signs[index] != -1 && signs[index] != 1) ||
            (signs[index + 1] != -1 && signs[index + 1] != 1)) return PHX_MC_INVALID_ARGUMENT;
        triangle.p[corner] = {value(index, signs[index]), value(index + 1, signs[index + 1]),
                              denominator};
      }
      for (int edge = 0; edge < 3; ++edge) {
        triangle.lines[edge] = cross(triangle.p[edge], triangle.p[(edge + 1) % 3]);
      }
      triangle.oriented_area = dot(triangle.lines[0], triangle.p[2]);
      triangle.fast_valid = true;
      for (int corner = 0; corner < 3; ++corner) {
        const int64_t index = row * 6 + corner * 2;
        triangle.fast_valid = triangle.fast_valid &&
            fast_load(words, offsets[index], offsets[index + 1], signs[index],
                      triangle.fast_points[corner].x) &&
            fast_load(words, offsets[index + 1], offsets[index + 2], signs[index + 1],
                      triangle.fast_points[corner].y);
      }
      for (int edge = 0; edge < 3 && triangle.fast_valid; ++edge) {
        triangle.fast_valid = fast_line(triangle.fast_points[edge],
                                       triangle.fast_points[(edge + 1) % 3],
                                       triangle.fast_lines[edge]);
      }
      triangles.push_back(std::move(triangle));
      Box box{{lower[row * 2], lower[row * 2 + 1]},
              {upper[row * 2], upper[row * 2 + 1]}};
      if (!(box.lo[0] <= box.hi[0] && box.lo[1] <= box.hi[1])) return PHX_MC_INVALID_ARGUMENT;
      (row < target_count ? target_boxes : declared_boxes).push_back(box);
    }
    *verdict = 0; *first = -1; *second = -1;
    if (declared_count == 0) { *verdict = 1; return PHX_MC_OK; }
    Dyadic target_area, declared_area;
    for (int64_t row = 0; row < target_count; ++row)
      target_area = target_area + absolute(orientation(triangles[static_cast<std::size_t>(row)]));
    for (int64_t row = target_count; row < count; ++row)
      declared_area = declared_area + absolute(orientation(triangles[static_cast<std::size_t>(row)]));
    if (!(target_area - declared_area).is_zero()) { *verdict = 1; return PHX_MC_OK; }
    const Tree declared_tree(declared_boxes);
    const Dyadic denominator_cube = denominator * denominator * denominator;
    for (int64_t row = 0; row < target_count; ++row) {
      const Triangle& triangle = triangles[static_cast<std::size_t>(row)];
      Rational covered{Dyadic(), integer_one()};
      for (const std::size_t candidate : declared_tree.candidates(target_boxes[static_cast<std::size_t>(row)])) {
        const Triangle& source = triangles[static_cast<std::size_t>(target_count) + candidate];
        if (separated(triangle, source)) continue;
        bool singular = false;
        covered = add(covered, intersection_area(triangle, source, singular));
        if (singular) { *verdict = 4; *first = row; *second = static_cast<int64_t>(candidate); return PHX_MC_OK; }
      }
      if (!equal(covered, {absolute(orientation(triangle)), denominator_cube})) {
        *verdict = 2; *first = row; return PHX_MC_OK;
      }
    }
    const Tree target_tree(target_boxes);
    for (int64_t row = 0; row < target_count; ++row) {
      const Triangle& triangle = triangles[static_cast<std::size_t>(row)];
      for (const std::size_t candidate : target_tree.candidates(target_boxes[static_cast<std::size_t>(row)])) {
        if (candidate <= static_cast<std::size_t>(row)) continue;
        if (!separated(triangle, triangles[candidate])) {
          *verdict = 3; *first = row; *second = static_cast<int64_t>(candidate); return PHX_MC_OK;
        }
      }
    }
    return PHX_MC_OK;
  });
}
