//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

#include <bit>

namespace phx::mc {

namespace geometry {
double signed_volume(const double* const* p) {
  return orient3d_exact(p[0], p[1], p[2], p[3]).estimate() / 6.0;
}
}  // namespace geometry

namespace geometry {
double shortest_edge(const double* const* p) {
  double shortest = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      shortest = std::min(shortest, squared_distance(p[i], p[j]));
    }
  }
  return std::sqrt(shortest);
}
}  // namespace geometry

namespace geometry {
bool triangle_circumcenter(const double* a, const double* b, const double* c, double* center) {
  native_execution_primitive_query();
  double u[3], v[3], n[3], vn[3], nu[3];
  difference(b, a, u);
  difference(c, a, v);
  cross(u, v, n);
  const double nn = dot(n, n);
  if (!(nn > 0.0) || !std::isfinite(nn)) {
    return false;
  }
  cross(v, n, vn);
  cross(n, u, nu);
  const double uu = dot(u, u);
  const double vv = dot(v, v);
  for (int i = 0; i < 3; ++i) {
    center[i] = a[i] + (uu * vn[i] + vv * nu[i]) / (2.0 * nn);
    if (a[i] == b[i] && b[i] == c[i]) {
      center[i] = a[i];
    }
  }
  return finite3(center);
}
}  // namespace geometry

namespace geometry {
bool tetrahedron_circumcenter(const double* a, const double* b, const double* c, const double* d,
                              double* center) {
  native_execution_primitive_query();
  // Evaluate the existing closed-form construction in the predicate owner's
  // expansion arithmetic. A rounded scalar triple product can have the wrong
  // sign (or vanish) for an exactly positive, representable tetrahedron.
  Expansion u[3], v[3], w[3], vw[3], wu[3], uv[3];
  for (int i = 0; i < 3; ++i) {
    u[i] = Expansion::difference(b[i], a[i]);
    v[i] = Expansion::difference(c[i], a[i]);
    w[i] = Expansion::difference(d[i], a[i]);
  }
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;
    vw[i] = v[j] * w[k] - v[k] * w[j];
    wu[i] = w[j] * u[k] - w[k] * u[j];
    uv[i] = u[j] * v[k] - u[k] * v[j];
  }
  const Expansion determinant = u[0] * vw[0] + u[1] * vw[1] + u[2] * vw[2];
  const double det = determinant.scaled(2.0).estimate();
  if (determinant.is_zero() || det == 0.0 || !std::isfinite(det)) {
    return false;
  }
  const Expansion uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  const Expansion vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const Expansion ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  for (int i = 0; i < 3; ++i) {
    center[i] = a[i] + (uu * vw[i] + vv * wu[i] + ww * uv[i]).estimate() / det;
  }
  return finite3(center);
}
}  // namespace geometry

bool TetMesh::construct_edge_point(int32_t a, int32_t b, double* position, double* parameter,
                                   double preferred_fraction) {
  const MemoryScope memory_scope(memory_owner_);
  if (!live_vertex(a) || !live_vertex(b) || a == b || !std::isfinite(preferred_fraction) ||
      preferred_fraction < 0.0 || preferred_fraction >= 1.0) {
    return false;
  }
  const auto construct = [&](double fraction, bool general) {
    if (!spend(TetWorkCost::kCoordinates)) {
      return false;
    }
    // Domain coordinates lie on a lattice with spacing at least 2^(min-52),
    // and an edge component is at most 2^(max+1). Smaller fractions cannot
    // produce a distinct domain point. Refusing that candidate also keeps
    // every scaled expansion component above binary64 product underflow.
    constexpr double minimum_fraction =
        (kCoordinateMinMagnitude / kCoordinateMaxMagnitude) * 0x1p-53;
    if (general && fraction < minimum_fraction) {
      return false;
    }
    bool exact = true;
    for (int k = 0; k < 3; ++k) {
      // A general preference must not round 1-f. Keep the original algebra
      // for the default dyadics, whose complementary fractions are exact.
      const Expansion value =
          general ? Expansion(point(a)[k]) +
                        Expansion::difference(point(b)[k], point(a)[k]).scaled(fraction)
                  : Expansion::product(point(a)[k], 1.0 - fraction) +
                        Expansion::product(point(b)[k], fraction);
      position[k] = value.estimate();
      exact = exact && (value - Expansion(position[k])).is_zero();
    }
    if (exact && coordinate_in_domain(position[0]) && coordinate_in_domain(position[1]) &&
        coordinate_in_domain(position[2]) && geometry::collinear(point(a), point(b), position) &&
        !std::equal(position, position + 3, point(a)) &&
        !std::equal(position, position + 3, point(b))) {
      if (parameter != nullptr) {
        *parameter = fraction;
      }
      return true;
    }
    return false;
  };
  const auto dyadics_excluded = [&]() {
    // For same-sign nonzero endpoints, every interior image has magnitude at
    // least that of the smaller endpoint. Let delta be that endpoint bin's
    // binary64 spacing. Both endpoints are integers A,B times delta, and
    // every representable interior image must also lie on delta*Z. If B-A is
    // odd, no proper dyadic n/2^level can do so. This excludes every remaining
    // level-1..8 candidate, without constructing its expansions or predicates.
    // Unsupported coordinate/exponent cases keep the original search.
    bool excluded = false;
    constexpr uint64_t sign = uint64_t{1} << 63;
    constexpr uint64_t exponent = uint64_t{0x7ff} << 52;
    for (int k = 0; k < 3; ++k) {
      const double first = point(a)[k], second = point(b)[k];
      if (!coordinate_in_domain(first) || !coordinate_in_domain(second)) {
        return false;
      }
      const uint64_t x = std::bit_cast<uint64_t>(first);
      const uint64_t y = std::bit_cast<uint64_t>(second);
      const uint64_t ex = x & exponent, ey = y & exponent;
      if (ex == 0 || ey == 0 || ((x ^ y) & sign) != 0) {
        continue;
      }
      excluded = excluded ||
                 (ex == ey ? ((x ^ y) & 1) != 0 : ((ex < ey ? x : y) & 1) != 0);
    }
    return excluded;
  };
  if (preferred_fraction != 0.0) {
    if (construct(preferred_fraction, true)) {
      return true;
    }
    if (!spend(TetWorkCost::kCoordinates) || dyadics_excluded()) {
      return false;
    }
    // The existing level-1..8 candidates are exactly [64,192]/256.
    // Walk the two nearest frontiers without allocating or sorting. Ties
    // retain the original order: coarser dyadics first, then lower numerator.
    int lower = std::clamp(static_cast<int>(preferred_fraction * 256.0), 63, 192);
    int upper = lower + 1;
    while (!work_exhausted() && (lower >= 64 || upper <= 192)) {
      bool take_lower = upper > 192;
      if (lower >= 64 && upper <= 192) {
        const double lower_distance = preferred_fraction - std::ldexp(lower, -8);
        const double upper_distance = std::ldexp(upper, -8) - preferred_fraction;
        take_lower = lower_distance < upper_distance ||
                     (lower_distance == upper_distance && (lower & -lower) >= (upper & -upper));
      }
      const double fraction = std::ldexp(take_lower ? lower-- : upper++, -8);
      if (fraction != preferred_fraction && construct(fraction, false)) {
        return true;
      }
    }
    return false;
  }
  for (int level = 1; level <= 8; ++level) {
    if (level == 2 && (!spend(TetWorkCost::kCoordinates) || dyadics_excluded())) {
      return false;
    }
    const int denominator = 1 << level;
    for (int numerator = 1; numerator < denominator; numerator += 2) {
      const double fraction = std::ldexp(static_cast<double>(numerator), -level);
      if (fraction < 0.25 || fraction > 0.75) {
        continue;
      }
      if (construct(fraction, false)) {
        return true;
      }
      if (work_exhausted()) {
        return false;
      }
    }
  }
  return false;
}

bool TetMesh::construct_interior_edge_point(int32_t a, int32_t b, double fraction,
                                            double* position) {
  const MemoryScope memory_scope(memory_owner_);
  if (!live_vertex(a) || !live_vertex(b) || a == b || !(fraction > 0.0 && fraction < 1.0) ||
      segment_source(a, b) >= 0 || !spend(TetWorkCost::kCoordinates)) {
    return false;
  }
  int axis = 0;
  for (int k = 0; k < 3; ++k) {
    position[k] = (Expansion(point(a)[k]) +
                   Expansion::difference(point(b)[k], point(a)[k]).scaled(fraction))
                      .estimate();
    if (std::abs(point(a)[k] - point(b)[k]) > std::abs(point(a)[axis] - point(b)[axis])) {
      axis = k;
    }
  }
  const double low = std::min(point(a)[axis], point(b)[axis]);
  const double high = std::max(point(a)[axis], point(b)[axis]);
  return coordinate_in_domain(position[0]) && coordinate_in_domain(position[1]) &&
         coordinate_in_domain(position[2]) && low < position[axis] && position[axis] < high;
}

bool TetMesh::construct_facet_point(int32_t a, int32_t b, int32_t c, double* position) {
  const MemoryScope memory_scope(memory_owner_);
  if (!live_vertex(a) || !live_vertex(b) || !live_vertex(c)) {
    return false;
  }
  const double* triangle[3] = {point(a), point(b), point(c)};
  for (int level = 2; level <= 6; ++level) {
    const int denominator = 1 << level;
    for (int u = 1; u < denominator; ++u) {
      for (int v = 1; u + v < denominator; ++v) {
        const double x = std::ldexp(static_cast<double>(u), -level);
        const double y = std::ldexp(static_cast<double>(v), -level);
        if (x < 0.125 || y < 0.125 || x + y > 0.875) {
          continue;
        }
        if (!spend(TetWorkCost::kFacet)) {
          return false;
        }
        for (int k = 0; k < 3; ++k) {
          position[k] =
              (Expansion::product(triangle[0][k], 1.0 - x - y) +
               Expansion::product(triangle[1][k], x) + Expansion::product(triangle[2][k], y))
                  .estimate();
        }
        int side = 0;
        if (coordinate_in_domain(position[0]) && coordinate_in_domain(position[1]) &&
            coordinate_in_domain(position[2]) &&
            locate_on_triangle(position, triangle, side) == kFeatureInterior) {
          return true;
        }
      }
    }
  }
  return false;
}

double TetMesh::interpolated_size(int32_t t, const double* p) const {
  const Tetrahedron& current = tet(t);
  double weights[4];
  double total = 0.0;
  for (int k = 0; k < 4; ++k) {
    const double* q[4];
    for (int j = 0; j < 4; ++j) {
      q[j] = j == k ? p : point(current.v[j]);
    }
    weights[k] = std::max(0.0, geometry::signed_volume(q));
    total += weights[k];
  }
  double value = 0.0;
  for (int k = 0; k < 4; ++k) {
    const double w = total > 0.0 ? weights[k] / total : 0.25;
    value += w * size(current.v[k]);
  }
  return std::isfinite(value) ? value : 0.0;
}

Insertion TetMesh::prepare_source_facet(int32_t t, int slot, const double* center, double* position,
                                        SourceWitness& witness) {
  int32_t face[3];
  oriented_facet(tet(t).v, slot, face);
  const int32_t source = complex_.constraint(t, slot);
  NativeVector<int32_t>& contributors = facet_contributors_;
  if (!face_ancestry(face, source, contributors) ||
      !spend(TetWorkCost::kSourceFacetConstruction))
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  const int32_t row = contributors.size() == 1
                          ? contributors.front()
                          : position_row(SourceStratum::kFacet, source, center);
  if (row < 0)
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  const double* corners[3];
  row_corners(SourceStratum::kFacet, static_cast<std::size_t>(row), corners);
  witness = SourceWitness{SourceStratum::kFacet, row, {0.0, 0.0}, 0.0};
  source_locator(corners, SourceStratum::kFacet, center, witness.parameters, false);
  if (!source_carrier(corners, SourceStratum::kFacet, witness.parameters, position,
                      witness.deviation))
    return Insertion::kRefused;
  if (!facet_witness_admitted(face, source, witness)) return Insertion::kRefused;
  const double tolerance = row_tolerance(SourceStratum::kFacet, row);
  if (witness.deviation > tolerance) {
    source_refusals_.push_back(
        {{face[0], face[1]}, SourceStratum::kFacet, row, witness.deviation, tolerance});
    return Insertion::kDeviation;
  }
  pending_ = static_cast<int32_t>(vertex_count());
  pending_source_ = witness;
  pending_source_authored_ = true;
  std::copy_n(position, 3, pending_point_);
  int32_t blocked = -1;
  int blocked_slot = -1;
  const int32_t located = locate(pending_point_, t, blocked, blocked_slot);
  if (located < 0) {
    abandon();
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  return prepare(position, located, -1, -1, std::numeric_limits<std::size_t>::max(), &witness);
}

}  // namespace phx::mc
