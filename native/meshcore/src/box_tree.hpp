//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Possible-zero box trees of interval programs: the subdivision of a domain
// box into the boxes where an interval program's enclosure of f contains
// zero, with the sign-change roots of their edges.
#pragma once

#include <array>
#include <cstdint>
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

// The kept leaf boxes (lower then upper corner) and their sign-change roots:
// one bracket per leaf edge whose binary64 corner values have strictly
// opposite signs, with its linear estimate, each edge once. Box b's roots
// are roots[box_roots[k]] for k in [box_offsets[b], box_offsets[b + 1]);
// root_box names the first box of each root.
struct PossibleZeroBoxes {
  std::vector<std::array<double, 6>> boxes;
  std::vector<Bracket> roots;
  std::vector<int32_t> box_offsets{0};
  std::vector<int32_t> box_roots;
  std::vector<int32_t> root_box;
};

// The octree of the cube centered on [lower, upper] with 9/8 of its largest
// extent (which keeps the zero set off the cube's faces), whose boxes are
// kept while f's interval enclosure over them contains zero or is not a
// number, down to boxes no wider than `size` (at most 2^12 per axis). By
// inclusion of the enclosure, every zero of f in the cube lies in a kept
// leaf. f at the leaf corners is evaluated by the point program `point`
// (binary64 proposals). Returns a PHX_MC status.
int32_t possible_zero_boxes(const IntervalProgramView& interval, const ProgramView& point,
                            const double* lower, const double* upper, double size,
                            PossibleZeroBoxes& out);

}  // namespace phx::mc
