//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Possible-zero box trees of interval programs.
#include "box_tree.hpp"

#include <algorithm>
#include <cmath>
#include <span>

#include "key_map.hpp"
#include "phydrax_meshcore.h"

namespace phx::mc {
namespace {

constexpr int kMaximumBoxLevel = 12;

// keep[b] = 1 when f's interval enclosure over box b (lower then upper
// corner) contains zero or is not a number.
int32_t straddling(const IntervalProgramView& program, std::span<const double> boxes,
                   std::vector<char>& keep) {
  const std::size_t count = boxes.size() / 6;
  const auto width = static_cast<std::size_t>(program.workspace_size);
  std::vector<double> lower(count * width), upper(count * width);
  for (std::size_t b = 0; b < count; ++b) {
    std::copy_n(program.seed_lower, width, lower.begin() + static_cast<std::ptrdiff_t>(b * width));
    std::copy_n(program.seed_upper, width, upper.begin() + static_cast<std::ptrdiff_t>(b * width));
    for (std::size_t d = 0; d < 3; ++d) {
      lower[b * width + d] = boxes[6 * b + d];
      upper[b * width + d] = boxes[6 * b + 3 + d];
    }
  }
  const int32_t status = phx_mc_interval_program_batch(
      program.instructions, program.instruction_count, program.indices, program.index_count,
      program.parameters, program.parameter_count, lower.data(), upper.data(),
      program.workspace_size, static_cast<int64_t>(count), 1);
  if (status != PHX_MC_OK) return status;
  keep.resize(count);
  const auto output = static_cast<std::size_t>(program.output);
  for (std::size_t b = 0; b < count; ++b)
    keep[b] = !(lower[b * width + output] > 0.0) && !(upper[b * width + output] < 0.0);
  return PHX_MC_OK;
}

}  // namespace

int32_t possible_zero_boxes(const IntervalProgramView& interval, const ProgramView& point,
                            const double* lower, const double* upper, double size,
                            PossibleZeroBoxes& out) {
  double largest = 0.0;
  for (int d = 0; d < 3; ++d) largest = std::max(largest, upper[d] - lower[d]);
  const double side = 1.125 * largest;
  std::array<double, 3> origin{};
  for (std::size_t d = 0; d < 3; ++d) origin[d] = 0.5 * (lower[d] + upper[d]) - 0.5 * side;
  int levels = 0;
  while (levels < kMaximumBoxLevel && side > std::ldexp(size, levels)) ++levels;
  using Index = std::array<int32_t, 3>;
  std::vector<Index> frontier{{0, 0, 0}}, children;
  std::vector<double> boxes;
  std::vector<char> keep;
  for (int level = 0;; ++level) {
    const double width = std::ldexp(side, -level);
    boxes.resize(6 * frontier.size());
    for (std::size_t b = 0; b < frontier.size(); ++b)
      for (std::size_t d = 0; d < 3; ++d) {
        boxes[6 * b + d] = origin[d] + width * frontier[b][d];
        boxes[6 * b + 3 + d] = origin[d] + width * (frontier[b][d] + 1);
      }
    const int32_t status = straddling(interval, boxes, keep);
    if (status != PHX_MC_OK) return status;
    children.clear();
    for (std::size_t b = 0; b < frontier.size(); ++b) {
      if (!keep[b]) continue;
      if (level == levels) {
        children.push_back(frontier[b]);
        continue;
      }
      for (int32_t octant = 0; octant < 8; ++octant)
        children.push_back({2 * frontier[b][0] + (octant & 1), 2 * frontier[b][1] + ((octant >> 1) & 1),
                            2 * frontier[b][2] + ((octant >> 2) & 1)});
    }
    frontier.swap(children);
    if (level == levels) break;
  }
  // f at the leaf corners, each grid point once.
  const int64_t grid = (int64_t{1} << levels) + 1;
  const double width = std::ldexp(side, -levels);
  const auto key_of = [grid](int64_t i, int64_t j, int64_t k) {
    return static_cast<std::uint64_t>((i * grid + j) * grid + k);
  };
  KeyMap corner_rows(4 * frontier.size());
  std::vector<double> corner_points;
  std::vector<std::array<int64_t, 8>> leaf_corners(frontier.size());
  out.boxes.resize(frontier.size());
  for (std::size_t b = 0; b < frontier.size(); ++b) {
    for (std::size_t d = 0; d < 3; ++d) {
      out.boxes[b][d] = origin[d] + width * frontier[b][d];
      out.boxes[b][d + 3] = origin[d] + width * (frontier[b][d] + 1);
    }
    for (int c = 0; c < 8; ++c) {
      const int64_t index[3] = {frontier[b][0] + (c & 1), frontier[b][1] + ((c >> 1) & 1),
                                frontier[b][2] + ((c >> 2) & 1)};
      int64_t& row = corner_rows.slot(key_of(index[0], index[1], index[2]));
      if (row < 0) {
        row = static_cast<int64_t>(corner_points.size() / 3);
        for (std::size_t d = 0; d < 3; ++d) corner_points.push_back(origin[d] + width * static_cast<double>(index[d]));
      }
      leaf_corners[b][static_cast<std::size_t>(c)] = row;
    }
  }
  std::vector<double> corner_values(corner_points.size() / 3);
  if (!corner_values.empty()) {
    PointWorkspace workspace;
    evaluate_point(point, corner_points.data(), static_cast<int64_t>(corner_values.size()),
                   corner_values.data(), workspace);
  }
  // Sign-change edges of the leaves (corner c to c + 2^axis), each root once.
  KeyMap edge_roots(frontier.size());
  for (std::size_t b = 0; b < frontier.size(); ++b) {
    for (int axis = 0; axis < 3; ++axis)
      for (int c = 0; c < 8; ++c) {
        if ((c >> axis) & 1) continue;
        const int64_t ra = leaf_corners[b][static_cast<std::size_t>(c)];
        const int64_t rb = leaf_corners[b][static_cast<std::size_t>(c | (1 << axis))];
        const double fa = corner_values[static_cast<std::size_t>(ra)];
        const double fb = corner_values[static_cast<std::size_t>(rb)];
        if (!((fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0))) continue;
        int64_t& root = edge_roots.slot((static_cast<std::uint64_t>(ra) << 2) | static_cast<std::uint64_t>(axis));
        if (root < 0) {
          root = static_cast<int64_t>(out.roots.size());
          const bool forward = fa < 0.0;
          const double* pa = corner_points.data() + 3 * static_cast<std::size_t>(forward ? ra : rb);
          const double* pb = corner_points.data() + 3 * static_cast<std::size_t>(forward ? rb : ra);
          Bracket bracket;
          bracket.negative = {pa[0], pa[1], pa[2]};
          bracket.positive = {pb[0], pb[1], pb[2]};
          bracket.f_negative = forward ? fa : fb;
          bracket.f_positive = forward ? fb : fa;
          const double t = bracket.f_negative / (bracket.f_negative - bracket.f_positive);
          for (std::size_t d = 0; d < 3; ++d)
            bracket.root[d] = bracket.negative[d] + t * (bracket.positive[d] - bracket.negative[d]);
          out.roots.push_back(bracket);
          out.root_box.push_back(static_cast<int32_t>(b));
        }
        out.box_roots.push_back(static_cast<int32_t>(root));
      }
    out.box_offsets.push_back(static_cast<int32_t>(out.box_roots.size()));
  }
  return PHX_MC_OK;
}

}  // namespace phx::mc
