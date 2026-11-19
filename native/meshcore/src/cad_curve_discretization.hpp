//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#pragma once

#include <cstdint>

namespace phx::mc {

// Shared discretization of every CAD source curve: the curve stage produces it,
// the face stage consumes it as a view. offsets has curve_count + 1 entries
// (curve order of the geometry graph), starts at 0 and is nondecreasing; each
// curve owns parameters[offsets[c] .. offsets[c + 1]): values of the normalized
// parameter s, at least two, the first exactly 0 and the last exactly 1,
// strictly increasing in between. Every use u of the curve, the owner use
// included, maps an interior value to its own parameter as
// u.first + s * (u.last - u.first) (the mapping of _use_source_parameter) and
// takes u.first and u.last themselves at s = 0 and s = 1, so no use's end
// value is ever recomputed. A closed curve carries both s = 0 and s = 1.
struct CadCurveDiscretization {
  const int64_t* offsets;
  const double* parameters;
};

}  // namespace phx::mc
