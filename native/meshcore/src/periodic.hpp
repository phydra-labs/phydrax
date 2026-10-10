//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Delaunay triangulation of a translationally periodic point set on the flat
// torus R^d / L (d = 2, 3), built on bounded image neighborhoods with certified
// sufficiency.
//
// Periodic points are pairs (representative r, integer lattice shift s) with
// exact position x_r + s L.  Every predicate is evaluated on exact coordinate
// differences (x_a - x_b) + (s_a - s_b) L with a forward-error filter and an
// adaptive-expansion fallback, so each decision depends only on relative
// positions and is invariant under lattice translation.  Cospherical ties use
// the Devillers-Teillaud symbolic perturbation ordered by the exact
// lexicographic order of positions, which is also translation invariant; the
// perturbed periodic Delaunay triangulation is therefore unique and every
// member of an orbit is decided identically.
//
// Round with fractional margin m: the images of every representative whose
// fractional coordinates lie in the box B_m = [-m, 1 + m]^d (all of them,
// within a rounding slack that only adds images) are triangulated inside a
// far bounding simplex.  A finite cell whose circumball lies in B_m (checked
// with a conservative slack on the constructed circumcenter) is empty of every
// periodic point, hence a cell of the periodic triangulation.  One cell per
// orbit is extracted: the translate whose anchor vertex (smallest
// representative, then lexicographically smallest shift) is the anchor's base
// image, wrapped into [0, 1)^d by the shift -floor(fractional).  The
// round succeeds when every extracted cell is certified and the extracted set
// is closed under facet adjacency (every finite neighbor's orbit is
// extracted); a nonempty closed set of cells of the connected periodic
// triangulation is all of it.  Otherwise the margin doubles, and a round whose
// image count exceeds the budget is refused with evidence.
#pragma once

#include <cstdint>
#include <vector>

namespace phx::mc {

struct PeriodicDelaunayInput {
  int dimension = 0;
  int64_t point_count = 0;
  const double* points = nullptr;      // (n, d) Cartesian representatives
  const double* fractional = nullptr;  // (n, d) lattice coordinates
  const double* lattice = nullptr;     // (d, d) rows are lattice vectors
  const double* inverse = nullptr;     // (d, d) with fractional = x @ inverse
  double initial_margin = 0.0;
  int64_t max_images = 0;
  int64_t max_cells = 0;
};

// Evidence slots; see PHX_MC_PERIODIC_EVIDENCE in phydrax_meshcore.h.
enum PeriodicEvidenceSlot : int {
  kPeriodicRounds = 0,
  kPeriodicMargin = 1,
  kPeriodicImages = 2,
  kPeriodicUncertified = 3,
  kPeriodicRequiredMargin = 4,
  kPeriodicFiniteCells = 5,
  kPeriodicExactEvaluations = 6,
  kPeriodicPerturbedDecisions = 7,
  kPeriodicExhaustedLimit = 8,
  kPeriodicDuplicatePoint = 9,
  kPeriodicEvidenceCount = 10,
};

struct PeriodicDelaunayResult {
  int dimension = 0;
  std::vector<int32_t> vertices;  // (T, d + 1) representatives
  std::vector<int32_t> shifts;    // (T, d + 1, d) image shifts of the anchored lift
  double evidence[kPeriodicEvidenceCount] = {};
};

// Returns a phx_mc_status.  The evidence is filled on every return after the
// arguments are accepted; cells are published only on PHX_MC_OK.
int32_t periodic_delaunay(const PeriodicDelaunayInput& input, PeriodicDelaunayResult& result);

}  // namespace phx::mc
