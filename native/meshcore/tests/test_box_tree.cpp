// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// The box tree's acceptance hook (box_tree.hpp).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "box_tree.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

constexpr int64_t kSubtract = 3;
constexpr int64_t kReduceSum = 7;
constexpr int64_t kSquare = 12;
constexpr double kGamma = 1.0e-15;  // >= 3u / (1 - 3u) for the short reduction
constexpr int64_t kUnbounded = std::numeric_limits<int64_t>::max();

// f = x^2 + y^2 + z^2 - 1, hand-lowered: box at [0, 3), the seed 1 at 3.
struct Sphere {
  std::vector<int64_t> rows{kSquare,     4, 3, 0, -1, -1, -1, 0,  //
                            kReduceSum,  7, 1, 4, -1, -1, 0,  3,  //
                            kSubtract,   8, 1, 7, -1, 3,  -1, 0};
  std::vector<double> parameters{kGamma};
  std::vector<double> seeds{0, 0, 0, 1, 0, 0, 0, 0, 0};
  std::vector<int64_t> outputs{8};

  phx::mc::IntervalProgramView interval() const {
    return {rows.data(),       static_cast<int64_t>(rows.size() / 8),
            nullptr,           0,
            parameters.data(), static_cast<int64_t>(parameters.size()),
            seeds.data(),      seeds.data(),
            static_cast<int64_t>(seeds.size()), outputs[0]};
  }
  phx::mc::ProgramView point() const {
    return {rows.data(),  static_cast<int64_t>(rows.size() / 8),
            nullptr,      seeds.data(),
            static_cast<int64_t>(seeds.size()), 3,
            outputs.data(), 1};
  }
};

// The unit sphere's cover of [-2, 2]^3 at level 6. Without a predicate the
// leaves are possible_zero_boxes' exactly. With a predicate accepting boxes
// below z = 0, every accepted leaf lies below it, is flagged and stops at the
// first level it is offered, every unaccepted leaf reaches z >= 0 at the
// target level, and every leaf of the full cover lies in exactly one leaf of
// the accepted cover (an ancestor or itself).
void acceptance_stops_subdivision() {
  const Sphere sphere;
  const phx::mc::IntervalProgramView interval = sphere.interval();
  const double lower[3]{-2, -2, -2}, upper[3]{2, 2, 2};
  constexpr int32_t level = 6;
  phx::mc::PossibleZeroBoxes full;
  PHX_CHECK(phx::mc::possible_zero_boxes(interval, sphere.point(), lower, upper, level, full) ==
            PHX_MC_OK);
  std::vector<phx::mc::DyadicLeaf> plain;
  int64_t plain_boxes = 0;
  PHX_CHECK(phx::mc::possible_zero_leaves(interval, lower, upper, level, nullptr, kUnbounded,
                                          plain_boxes, plain) == PHX_MC_OK);
  PHX_CHECK(plain.size() == full.leaves.size() && !plain.empty());
  for (size_t b = 0; b < std::min(plain.size(), full.leaves.size()); ++b)
    PHX_CHECK(plain[b].level == full.leaves[b].level && plain[b].index == full.leaves[b].index &&
              plain[b].box == full.leaves[b].box && !plain[b].accepted &&
              !full.leaves[b].accepted);
  const phx::mc::BoxAcceptance below = [](const std::array<double, 6>& box) {
    return box[5] < 0;
  };
  std::vector<phx::mc::DyadicLeaf> stopped;
  int64_t stopped_boxes = 0;
  PHX_CHECK(phx::mc::possible_zero_leaves(interval, lower, upper, level, &below, kUnbounded,
                                          stopped_boxes, stopped) == PHX_MC_OK);
  PHX_CHECK(0 < stopped_boxes && stopped_boxes < plain_boxes);
  int64_t accepted = 0;
  std::map<std::pair<int32_t, std::array<int64_t, 3>>, size_t> by_index;
  for (size_t b = 0; b < stopped.size(); ++b) {
    const phx::mc::DyadicLeaf& leaf = stopped[b];
    by_index[{leaf.level, leaf.index}] = b;
    if (leaf.accepted) {
      ++accepted;
      PHX_CHECK(leaf.box[5] < 0 && !leaf.representability);
      // Its parent (offered one level up, unaccepted) reaches z >= 0; the
      // lattice coordinates of [-2, 2] are exact dyadics.
      PHX_CHECK(leaf.level >= 1 &&
                -2 + 4.0 * static_cast<double>((leaf.index[2] >> 1) + 1) /
                             std::ldexp(1.0, leaf.level - 1) >= 0);
    } else {
      PHX_CHECK(leaf.box[5] >= 0 && leaf.level == level);
    }
  }
  PHX_CHECK(accepted > 0 && stopped.size() < plain.size());
  for (const phx::mc::DyadicLeaf& leaf : plain) {
    int64_t holders = 0;
    for (int32_t up = 0; up <= leaf.level; ++up) {
      const std::array<int64_t, 3> ancestor{leaf.index[0] >> (leaf.level - up),
                                            leaf.index[1] >> (leaf.level - up),
                                            leaf.index[2] >> (leaf.level - up)};
      holders += by_index.count({up, ancestor}) > 0 ? 1 : 0;
    }
    PHX_CHECK(holders == 1);
  }
}

// A target level beyond the tree's limit is refused, with or without a
// predicate.
void level_limit_refused() {
  const Sphere sphere;
  const double lower[3]{-2, -2, -2}, upper[3]{2, 2, 2};
  std::vector<phx::mc::DyadicLeaf> leaves;
  int64_t boxes = 0;
  PHX_CHECK(phx::mc::possible_zero_leaves(sphere.interval(), lower, upper, 21, nullptr, kUnbounded,
                                          boxes, leaves) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(leaves.empty());
}

// The box budget: the level-6 sphere cover evaluates `needed` boxes; with
// exactly that budget it completes, with one box less it is refused before
// the last level is evaluated (CAPACITY_EXCEEDED), and a budget below 1 is
// an invalid argument.
void box_budget_refuses() {
  const Sphere sphere;
  const double lower[3]{-2, -2, -2}, upper[3]{2, 2, 2};
  std::vector<phx::mc::DyadicLeaf> leaves;
  int64_t needed = 0;
  PHX_CHECK(phx::mc::possible_zero_leaves(sphere.interval(), lower, upper, 6, nullptr, kUnbounded,
                                          needed, leaves) == PHX_MC_OK);
  int64_t boxes = 0;
  std::vector<phx::mc::DyadicLeaf> exact;
  PHX_CHECK(phx::mc::possible_zero_leaves(sphere.interval(), lower, upper, 6, nullptr, needed,
                                          boxes, exact) == PHX_MC_OK);
  PHX_CHECK(boxes == needed && exact.size() == leaves.size());
  std::vector<phx::mc::DyadicLeaf> short_of;
  PHX_CHECK(phx::mc::possible_zero_leaves(sphere.interval(), lower, upper, 6, nullptr, needed - 1,
                                          boxes, short_of) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(boxes < needed);
  PHX_CHECK(phx::mc::possible_zero_leaves(sphere.interval(), lower, upper, 6, nullptr, 0, boxes,
                                          short_of) == PHX_MC_INVALID_ARGUMENT);
}

// A reciprocal row the batch kernel refuses sends the box tree to the
// general executor: the unit sphere with a dead reciprocal (1 / 1 into an
// unused entry) gets an enclosure from it. The two executors' arithmetic
// differs, but on this program and lattice they decide every sign alike, so
// the leaves and roots equal the plain sphere's from the batch kernel.
void general_executor_matches_batch() {
  constexpr int64_t kReciprocal = 11;
  const Sphere plain;
  Sphere dead;
  dead.rows.insert(dead.rows.end(), {kReciprocal, 9, 1, 3, -1, -1, -1, 0});
  dead.seeds.push_back(0);
  std::vector<double> low = dead.seeds, high = dead.seeds;
  PHX_CHECK(phx_mc_interval_program_batch(dead.rows.data(),
                                           static_cast<int64_t>(dead.rows.size() / 8), nullptr, 0,
                                           dead.parameters.data(), 1, low.data(), high.data(),
                                           static_cast<int64_t>(dead.seeds.size()), 1,
                                           1) == PHX_MC_INVALID_INPUT);
  const double lower[3]{-1.3, -1.1, -0.9}, upper[3]{1.7, 1.2, 1.4};
  for (const int32_t level : {3, 6}) {
    phx::mc::PossibleZeroBoxes batch, general;
    PHX_CHECK(phx::mc::possible_zero_boxes(plain.interval(), plain.point(), lower, upper, level,
                                           batch) == PHX_MC_OK);
    PHX_CHECK(phx::mc::possible_zero_boxes(dead.interval(), dead.point(), lower, upper, level,
                                           general) == PHX_MC_OK);
    PHX_CHECK(!batch.leaves.empty() && batch.leaves.size() == general.leaves.size() &&
              batch.roots.size() == general.roots.size());
    for (size_t b = 0; b < std::min(batch.leaves.size(), general.leaves.size()); ++b)
      PHX_CHECK(batch.leaves[b].level == general.leaves[b].level &&
                batch.leaves[b].index == general.leaves[b].index &&
                batch.leaves[b].box == general.leaves[b].box &&
                batch.leaves[b].representability == general.leaves[b].representability);
    for (size_t r = 0; r < std::min(batch.roots.size(), general.roots.size()); ++r)
      PHX_CHECK(batch.roots[r].negative == general.roots[r].negative &&
                batch.roots[r].positive == general.roots[r].positive &&
                batch.roots[r].root == general.roots[r].root);
    PHX_CHECK(batch.leaf_roots == general.leaf_roots && batch.leaf_offsets == general.leaf_offsets);
  }
}

}  // namespace

int main() {
  acceptance_stops_subdivision();
  level_limit_refused();
  box_budget_refuses();
  general_executor_matches_batch();
  return phx::mc::test::finish("test_box_tree");
}
