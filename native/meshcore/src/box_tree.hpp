//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Possible-zero covers of interval programs on the anchored dyadic lattice of
// a domain box, with the sign-change roots of their leaves' edges.
//
// The lattice and leaf identity are those of discover_leaves
// (phydrax/geometry/implicit/_adaptive_packed_frontier.py): at level L the
// leaf with integer index (i, j, k) spans, per axis, the exact coordinates
// (1 - n / 2^L) lower + (n / 2^L) upper at n = index and n = index + 1, its
// box rounded outward; children are 2 index + {0, 1}^3. A box is kept while
// f's interval enclosure over it has value_lower <= 0 <= value_upper (or is
// not a number); a kept box whose midpoint coordinate does not round strictly
// between its ends, or whose rounded box is empty, stops subdividing with
// the representability issue.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "interval_program.hpp"

namespace phx::mc {

// An interval program of f over three inputs (workspace entries 0 to 2):
// its validated bytecode, its interval workspace templates and f's entry.
struct IntervalProgramView {
  const int64_t* instructions;
  int64_t instruction_count;
  const int64_t* indices;
  int64_t index_count;
  const double* parameters;
  int64_t parameter_count;
  const double* seed_lower;
  const double* seed_upper;
  int64_t workspace_size;
  int64_t output;
};

// A segment from `negative` (f < 0) to `positive` (f > 0 at its making) and
// a point of it, the root estimate.
struct Bracket {
  std::array<double, 3> negative, positive;
  double f_negative = 0.0, f_positive = 0.0;
  std::array<double, 3> root{};
};

// A possible-zero leaf: its lattice level and integer index, its outward box
// (lower then upper corner), whether it stopped above the target level at
// the representability limit, and whether an acceptance predicate stopped it.
struct DyadicLeaf {
  int32_t level = 0;
  std::array<int64_t, 3> index{};
  std::array<double, 6> box{};
  bool representability = false;
  bool accepted = false;
};

// A per-box acceptance predicate over outward boxes (lower then upper
// corner). A kept box it accepts stops subdividing and becomes an accepted
// leaf; the predicate must itself prove that the box needs no finer cover.
using BoxAcceptance = std::function<bool(const std::array<double, 6>& box)>;

// The possible-zero leaves and their sign-change roots: one bracket per leaf
// edge whose binary64 values at its lattice vertices (the nearest binary64
// coordinates) have strictly opposite signs, with its linear estimate, each
// edge once. Leaf b's roots are roots[leaf_roots[k]] for k in
// [leaf_offsets[b], leaf_offsets[b + 1]); root_leaf names the first leaf of
// each root.
struct PossibleZeroBoxes {
  std::vector<DyadicLeaf> leaves;
  std::vector<Bracket> roots;
  std::vector<int32_t> leaf_offsets{0};
  std::vector<int32_t> leaf_roots;
  std::vector<int32_t> root_leaf;
};

// The smallest level whose cell diagonal, an upper bound of the domain's
// diagonal times 2^-level, is at most `diameter` (_target_level of the
// adaptive discovery with relative diameter diameter / diagonal).
int32_t dyadic_target_level(const double* lower, const double* upper, double diameter);

// The possible-zero cover of the anchored lattice of [lower, upper] down to
// `target_level` (at most 30), f at the lattice vertices of its target-level
// leaves by the point program `point` (binary64 proposals). By inclusion of
// the enclosure every zero of f in the domain lies in a leaf. Returns a
// PHX_MC status.
int32_t possible_zero_boxes(const IntervalProgramView& interval, const ProgramView& point,
                            const double* lower, const double* upper, int32_t target_level,
                            PossibleZeroBoxes& out);

// The possible-zero leaves alone (no lattice vertices, no roots), each kept
// box first offered to `accept`, if not null, which may stop it at any level
// up to target_level (at most kMaximumLevel of the tree, 20). With a null
// predicate the leaves are possible_zero_boxes'. Returns a PHX_MC status.
int32_t possible_zero_leaves(const IntervalProgramView& interval, const double* lower,
                             const double* upper, int32_t target_level,
                             const BoxAcceptance* accept, std::vector<DyadicLeaf>& leaves);

}  // namespace phx::mc
