//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Possible-zero covers of interval programs on the anchored dyadic lattice.
#include "box_tree.hpp"

#include <algorithm>
#include <cmath>
#include <span>

#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "geometry3.hpp"
#include "key_map.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc {
namespace {

// The lattice coordinates of one axis down to a finest level, computed once
// per exact coordinate: outward lower, outward upper and nearest binary64 of
// (1 - n / 2^level) lower + (n / 2^level) upper. Index n at level L is index
// n 2^(finest - L) at the finest level, the same exact coordinate, so every
// level shares one row per coordinate.
class AxisCoordinates {
 public:
  AxisCoordinates(double lower, double upper, int32_t finest)
      : lower_(lower), upper_(upper), finest_(finest), rows_(64) {}
  const std::array<double, 3>& at(int32_t level, int64_t index) {
    const int64_t finest_index = index << (finest_ - level);
    int64_t& row = rows_.slot(static_cast<std::uint64_t>(finest_index));
    if (row < 0) {
      row = static_cast<int64_t>(values_.size());
      values_.push_back(compute(finest_index));
    }
    return values_[static_cast<std::size_t>(row)];
  }

 private:
  std::array<double, 3> compute(int64_t index) const {
    const int64_t denominator = int64_t{1} << finest_;
    if (index <= 0) return {lower_, lower_, lower_};
    if (index >= denominator) return {upper_, upper_, upper_};
    const Dyadic numerator =
        Dyadic(lower_) * Dyadic(static_cast<double>(denominator - index)) +
        Dyadic(upper_) * Dyadic(static_cast<double>(index));
    const Dyadic scale(static_cast<double>(denominator));
    const exact::QuotientBounds bounds = exact::directed_quotient_bounds(numerator, scale);
    return {bounds.lower, bounds.upper, exact::rounded_quotient(numerator, scale)};
  }
  double lower_, upper_;
  int32_t finest_;
  KeyMap rows_;
  std::vector<std::array<double, 3>> values_;
};

using Axes = std::array<AxisCoordinates, 3>;

// keep[b] = 1 when f's interval enclosure over box b (lower then upper
// corner) has value_lower <= 0 <= value_upper or is not a number. The boxes
// run in batches of kStraddlingBatch through the workspaces `lower` and
// `upper`, which stay cache-sized and are reused across calls.
constexpr std::size_t kStraddlingBatch = 512;

int32_t straddling(const IntervalProgramView& program, std::span<const double> boxes,
                   std::vector<char>& keep, std::vector<double>& lower, std::vector<double>& upper) {
  const std::size_t count = boxes.size() / 6;
  keep.assign(count, 0);
  const auto width = static_cast<std::size_t>(program.workspace_size);
  const auto output = static_cast<std::size_t>(program.output);
  for (std::size_t first = 0; first < count; first += kStraddlingBatch) {
    const std::size_t batch = std::min(kStraddlingBatch, count - first);
    lower.resize(batch * width);
    upper.resize(batch * width);
    for (std::size_t b = 0; b < batch; ++b) {
      std::copy_n(program.seed_lower, width, lower.begin() + static_cast<std::ptrdiff_t>(b * width));
      std::copy_n(program.seed_upper, width, upper.begin() + static_cast<std::ptrdiff_t>(b * width));
      for (std::size_t d = 0; d < 3; ++d) {
        lower[b * width + d] = boxes[6 * (first + b) + d];
        upper[b * width + d] = boxes[6 * (first + b) + 3 + d];
      }
    }
    const int32_t status = phx_mc_interval_program_batch(
        program.instructions, program.instruction_count, program.indices, program.index_count,
        program.parameters, program.parameter_count, lower.data(), upper.data(),
        program.workspace_size, static_cast<int64_t>(batch), 1);
    if (status != PHX_MC_OK) return status;
    for (std::size_t b = 0; b < batch; ++b)
      keep[first + b] = !(upper[b * width + output] < 0.0) && !(lower[b * width + output] > 0.0);
  }
  return PHX_MC_OK;
}

using Index = std::array<int64_t, 3>;

// The kept leaves of the anchored lattice, level by level as discover_leaves,
// on `axes` with finest level target_level.
int32_t possible_leaves(const IntervalProgramView& interval, Axes& axes, int32_t target_level,
                        std::vector<DyadicLeaf>& leaves) {
  std::vector<Index> frontier{{0, 0, 0}}, children;
  std::vector<double> boxes;
  std::vector<char> valid, keep;
  std::vector<double> workspace_lower, workspace_upper;
  std::vector<std::array<double, 6>> frontier_boxes;
  for (int32_t level = 0; !frontier.empty(); ++level) {
    boxes.clear();
    valid.assign(frontier.size(), 1);
    frontier_boxes.resize(frontier.size());
    for (std::size_t b = 0; b < frontier.size(); ++b) {
      for (std::size_t d = 0; d < 3; ++d) {
        frontier_boxes[b][d] = axes[d].at(level, frontier[b][d])[0];
        frontier_boxes[b][d + 3] = axes[d].at(level, frontier[b][d] + 1)[1];
        if (!(frontier_boxes[b][d] < frontier_boxes[b][d + 3])) valid[b] = 0;
      }
      if (valid[b]) boxes.insert(boxes.end(), frontier_boxes[b].begin(), frontier_boxes[b].end());
    }
    const int32_t status = straddling(interval, boxes, keep, workspace_lower, workspace_upper);
    if (status != PHX_MC_OK) return status;
    children.clear();
    std::size_t evaluated = 0;
    for (std::size_t b = 0; b < frontier.size(); ++b) {
      const Index& index = frontier[b];
      if (!valid[b]) {
        leaves.push_back({level, index, frontier_boxes[b], true});
        continue;
      }
      if (!keep[evaluated++]) continue;
      if (level < target_level) {
        bool stalled = false;
        for (std::size_t d = 0; d < 3; ++d) {
          const double mid = axes[d].at(level + 1, 2 * index[d] + 1)[2];
          const double low = axes[d].at(level, index[d])[2];
          const double high = axes[d].at(level, index[d] + 1)[2];
          stalled = stalled || !(mid > low && mid < high);
        }
        if (stalled) {
          leaves.push_back({level, index, frontier_boxes[b], true});
          continue;
        }
        for (int64_t octant = 0; octant < 8; ++octant)
          children.push_back({2 * index[0] + (octant & 1), 2 * index[1] + ((octant >> 1) & 1),
                              2 * index[2] + ((octant >> 2) & 1)});
      } else {
        leaves.push_back({level, index, frontier_boxes[b], false});
      }
    }
    frontier.swap(children);
  }
  return PHX_MC_OK;
}

}  // namespace

int32_t dyadic_target_level(const double* lower, const double* upper, double diameter) {
  double squares = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double extent = geometry::next_up(upper[d] - lower[d]);
    squares = geometry::next_up(squares + geometry::next_up(extent * extent));
  }
  const double diagonal = geometry::next_up(std::sqrt(squares));
  int32_t level = 0;
  while (level < kMaximumBoxTreeLevel + 1 && geometry::next_up(std::ldexp(diagonal, -level)) > diameter) ++level;
  return level;
}

int32_t possible_zero_boxes(const IntervalProgramView& interval, const ProgramView& point,
                            const double* lower, const double* upper, int32_t target_level,
                            PossibleZeroBoxes& out) {
  if (target_level < 0 || target_level > kMaximumBoxTreeLevel) return PHX_MC_INVALID_ARGUMENT;
  Axes axes{AxisCoordinates(lower[0], upper[0], target_level), AxisCoordinates(lower[1], upper[1], target_level),
            AxisCoordinates(lower[2], upper[2], target_level)};
  int32_t status = possible_leaves(interval, axes, target_level, out.leaves);
  if (status != PHX_MC_OK) return status;
  // The lattice vertices of the leaves, each once, keyed by their index at
  // the target level, at their nearest binary64 coordinates.
  const auto key_of = [](const Index& index) {
    return (static_cast<std::uint64_t>(index[0]) << 42) | (static_cast<std::uint64_t>(index[1]) << 21) |
           static_cast<std::uint64_t>(index[2]);
  };
  KeyMap vertex_rows(4 * out.leaves.size());
  std::vector<double> vertex_points;
  std::vector<std::array<int64_t, 8>> leaf_vertices(out.leaves.size());
  for (std::size_t b = 0; b < out.leaves.size(); ++b) {
    const DyadicLeaf& leaf = out.leaves[b];
    const int32_t shift = target_level - leaf.level;
    for (int c = 0; c < 8; ++c) {
      const Index index{(leaf.index[0] + (c & 1)) << shift, (leaf.index[1] + ((c >> 1) & 1)) << shift,
                        (leaf.index[2] + ((c >> 2) & 1)) << shift};
      int64_t& row = vertex_rows.slot(key_of(index));
      if (row < 0) {
        row = static_cast<int64_t>(vertex_points.size() / 3);
        for (std::size_t d = 0; d < 3; ++d) vertex_points.push_back(axes[d].at(target_level, index[d])[2]);
      }
      leaf_vertices[b][static_cast<std::size_t>(c)] = row;
    }
  }
  std::vector<double> vertex_values(vertex_points.size() / 3);
  if (!vertex_values.empty()) {
    PointWorkspace workspace;
    evaluate_point(point, vertex_points.data(), static_cast<int64_t>(vertex_values.size()),
                   vertex_values.data(), workspace);
  }
  // Sign-change edges of the leaves (vertex c to c + 2^axis), each root once:
  // an edge is its first vertex, axis and level.
  KeyMap edge_roots(out.leaves.size());
  for (std::size_t b = 0; b < out.leaves.size(); ++b) {
    for (int axis = 0; axis < 3; ++axis)
      for (int c = 0; c < 8; ++c) {
        if ((c >> axis) & 1) continue;
        const int64_t ra = leaf_vertices[b][static_cast<std::size_t>(c)];
        const int64_t rb = leaf_vertices[b][static_cast<std::size_t>(c | (1 << axis))];
        const double fa = vertex_values[static_cast<std::size_t>(ra)];
        const double fb = vertex_values[static_cast<std::size_t>(rb)];
        if (!((fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0))) continue;
        int64_t& root = edge_roots.slot((static_cast<std::uint64_t>(ra) << 7) |
                                        (static_cast<std::uint64_t>(axis) << 5) |
                                        static_cast<std::uint64_t>(out.leaves[b].level));
        if (root < 0) {
          root = static_cast<int64_t>(out.roots.size());
          const bool forward = fa < 0.0;
          const double* pa = vertex_points.data() + 3 * static_cast<std::size_t>(forward ? ra : rb);
          const double* pb = vertex_points.data() + 3 * static_cast<std::size_t>(forward ? rb : ra);
          Bracket bracket;
          bracket.negative = {pa[0], pa[1], pa[2]};
          bracket.positive = {pb[0], pb[1], pb[2]};
          bracket.f_negative = forward ? fa : fb;
          bracket.f_positive = forward ? fb : fa;
          const double t = bracket.f_negative / (bracket.f_negative - bracket.f_positive);
          for (std::size_t d = 0; d < 3; ++d)
            bracket.root[d] = bracket.negative[d] + t * (bracket.positive[d] - bracket.negative[d]);
          out.roots.push_back(bracket);
          out.root_leaf.push_back(static_cast<int32_t>(b));
        }
        out.leaf_roots.push_back(static_cast<int32_t>(root));
      }
    out.leaf_offsets.push_back(static_cast<int32_t>(out.leaf_roots.size()));
  }
  return PHX_MC_OK;
}

}  // namespace phx::mc
