//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted Delaunay meshing of an implicit solid, and outward bounds of
// exact tetrahedron circumcenters.
//
// phx_mc_implicit_restricted_delaunay meshes {f <= 0} surface first, in one
// 3D Delaunay triangulation (Boissonnat and Oudot, "Provably good sampling
// and meshing of surfaces", 2005; Rineau and Yvinec, "Meshing 3D domains
// bounded by piecewise smooth surfaces", 2007):
//
//   0. seeds: the possible-zero box tree of f (box_tree.hpp) down to the
//      facet edge bound; every edge of a kept box whose corner values have
//      strictly opposite signs carries a root (binary64 corner values,
//      regula falsi on the edge). A component of the
//      zero set crossing a kept box's edges therefore has a root; roots
//      closer than a fraction of the edge bound to an earlier one are left
//      to the refinement;
//   1. the seeds and eight far corners are triangulated. A cell is inside
//      when f < 0 at its circumcenter; a facet between an inside and an
//      outside cell is restricted, and its dual Voronoi edge (circumcenter
//      to circumcenter, or a ray toward a ghost) crosses the zero set at the
//      center of its surface Delaunay ball (regula falsi on the edge);
//   2. restricted facets with a long edge, a far ball center or a small
//      angle are refined at their ball center, largest ball first; then
//      inside cells with a large radius-edge ratio or circumradius at their
//      circumcenter, unless its conflict region holds a restricted facet,
//      which is refined instead; no point is inserted closer to a vertex
//      than a separation bound, which bounds the vertex count by packing
//      (RestrictedMesher);
//   3. the restricted surface (the boundary of the inside cells) is checked
//      for edge-manifoldness, its topology and coverage of every box root;
//      failures insert box roots or ball centers there and refine again;
//   4. with the boundary vertices moved onto the zero set, slivers are
//      removed by flips and relocation; the circumcenters of those left are
//      inserted and steps 2-4 repeated, and cells still missing the
//      requested shapes go through the fixed-boundary quality improvement.
//
// Construction only: floating-point evaluation proposes; the volume, source
// fidelity and topology certificates run on the result afterwards.
#include "restricted_delaunay.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <vector>

#include "box_tree.hpp"
#include "capi_guard.hpp"
#include "dyadic.hpp"
#include "exact_scalar.hpp"
#include "expansion.hpp"
#include "floating_point_mode.hpp"
#include "geometry3.hpp"
#include "improve3d.hpp"
#include "interval_arithmetic.hpp"
#include "interval_program.hpp"
#include "key_map.hpp"
#include "mesh.hpp"
#include "predicates.hpp"
#include "refine3d.hpp"
#include "spatial_sort.hpp"
#include "tet_mesh.hpp"
#include "triangulation3d.hpp"

// The constructed restricted volume: points, positive tetrahedra, outward
// boundary triangles and the construction report.
struct phx_mc_restricted_volume {
  std::vector<double> points;
  std::vector<int32_t> cells, faces;
  std::array<int64_t, PHX_MC_RESTRICTED_REPORT_SIZE> report{};
};

namespace phx::mc {
namespace {
using namespace restricted_layout;
using ExactVector = std::array<Expansion, kDimensions>;

ExactVector exact_cross(const ExactVector& a, const ExactVector& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

Expansion exact_dot(const ExactVector& a, const ExactVector& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 2> expansion_bounds(const Expansion& value) {
  double lower = 0.0;
  double upper = 0.0;
  for (int index = 0; index < value.size(); ++index) {
    lower = geometry::next_down(lower + value.data()[index]);
    upper = geometry::next_up(upper + value.data()[index]);
  }
  return {lower, upper};
}

bool exact_center_bounds(const double* const* p, std::array<double, kCenterBounds>& bounds) {
  ExactVector u, v, w;
  for (int axis = 0; axis < kDimensions; ++axis) {
    u[axis] = Expansion::difference(p[1][axis], p[0][axis]);
    v[axis] = Expansion::difference(p[2][axis], p[0][axis]);
    w[axis] = Expansion::difference(p[3][axis], p[0][axis]);
  }
  const ExactVector vw = exact_cross(v, w);
  const ExactVector wu = exact_cross(w, u);
  const ExactVector uv = exact_cross(u, v);
  const Expansion denominator = exact_dot(u, vw).scaled(2.0);
  const auto d = expansion_bounds(denominator);
  if (!(d[0] > 0.0) || !std::isfinite(d[1])) return false;
  const Expansion uu = exact_dot(u, u);
  const Expansion vv = exact_dot(v, v);
  const Expansion ww = exact_dot(w, w);
  for (int axis = 0; axis < kDimensions; ++axis) {
    const Expansion numerator = uu * vw[axis] + vv * wu[axis] + ww * uv[axis] +
                                denominator.scaled(p[0][axis]);
    const auto n = expansion_bounds(numerator);
    const double quotients[4] = {n[0] / d[0], n[0] / d[1], n[1] / d[0], n[1] / d[1]};
    bounds[axis] = geometry::next_down(*std::min_element(quotients, quotients + 4));
    bounds[axis + kUpper] = geometry::next_up(*std::max_element(quotients, quotients + 4));
    if (!std::isfinite(bounds[axis]) || !std::isfinite(bounds[axis + kUpper])) return false;
  }
  return true;
}

int32_t validate_tets(int64_t point_count, const double* points, int64_t tet_count,
                      const int32_t* tets) {
  if (point_count > kMaxMeshPoints ||
      tet_count > std::numeric_limits<int32_t>::max()) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  if (point_count < kTetVertices || tet_count < 1 || points == nullptr || tets == nullptr ||
      !addressable(point_count, kDimensions, sizeof(double)) ||
      !addressable(tet_count, kTetVertices, sizeof(int32_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  const int32_t valid = validate_points(points, point_count, kDimensions, nullptr);
  if (valid != PHX_MC_OK) return valid;
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    for (int vertex = 0; vertex < kTetVertices; ++vertex) {
      if (ids[vertex] < 0 || ids[vertex] >= point_count ||
          std::count(ids, ids + kTetVertices, ids[vertex]) != 1) {
        return PHX_MC_INVALID_INPUT;
      }
    }
    if (orient3d_filtered(points + kDimensions * ids[0], points + kDimensions * ids[1],
                          points + kDimensions * ids[2], points + kDimensions * ids[3]) <= 0) {
      return PHX_MC_DEGENERATE_INPUT;
    }
  }
  return PHX_MC_OK;
}

int32_t construct_centers(int64_t point_count, const double* points, int64_t tet_count,
                          const int32_t* tets, double* bounds, int32_t* item_status) {
  if (item_status == nullptr || bounds == nullptr ||
      !addressable(tet_count, kCenterBounds, sizeof(double))) return PHX_MC_INVALID_ARGUMENT;
  const int32_t valid = validate_tets(point_count, points, tet_count, tets);
  if (valid != PHX_MC_OK) return valid;
  std::vector<double> staged(static_cast<std::size_t>(tet_count) * kCenterBounds);
  std::vector<int32_t> statuses(static_cast<std::size_t>(tet_count), PHX_MC_OK);
  for (int64_t cell = 0; cell < tet_count; ++cell) {
    const int32_t* ids = tets + kTetVertices * cell;
    const double* p[kTetVertices] = {
        points + kDimensions * ids[0], points + kDimensions * ids[1],
        points + kDimensions * ids[2], points + kDimensions * ids[3]};
    std::array<double, kCenterBounds> value{};
    if (!exact_center_bounds(p, value)) {
      statuses[static_cast<std::size_t>(cell)] = PHX_MC_DEGENERATE_INPUT;
      std::fill(value.begin(), value.begin() + kUpper, -std::numeric_limits<double>::infinity());
      std::fill(value.begin() + kUpper, value.end(), std::numeric_limits<double>::infinity());
    }
    std::copy(value.begin(), value.end(), staged.begin() + kCenterBounds * cell);
  }
  std::copy(staged.begin(), staged.end(), bounds);
  std::copy(statuses.begin(), statuses.end(), item_status);
  return PHX_MC_OK;
}

// ---------------------------------------------------------------- meshing

using Point = std::array<double, 3>;

constexpr int kCorners = 8;
constexpr int kRootIterations = 60;
// Regula falsi stops when its step or bracket is below these fractions of
// the segment.
constexpr double kRootStep = 0x1p-32;
constexpr double kRootWidth = 0x1p-44;
// Construction roots then move along their segment by an offset of at most
// this fraction of it (CGAL's implicit oracle stops at 1e-6). Without it,
// points on a zero set that is a sphere or a plane are cospherical or
// coplanar to rounding, so the Delaunay predicates on them fall through to
// exact arithmetic (three times the construction time on the ledger
// sphere). The offset is a hash of the bracket's endpoint bits, a function
// of the input alone (no addresses, clocks or random state), so equal
// inputs give equal meshes. The restricted surface's vertices are moved
// onto the zero set to float accuracy after the refinement.
constexpr double kRootOffset = 0x1p-24;
constexpr int kNewtonSteps = 8;
constexpr int kPolishSteps = 3;
constexpr int kRepairRounds = 4;
// Outward faces of a positive tetrahedron: face k omits vertex k.
constexpr int kOutwardFace[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
constexpr int kEdges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

Point minus(const double* a, const double* b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Point cross(const Point& a, const Point& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Point along(const Point& a, const Point& b, double t) {
  return {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
}

// IEEE 754-2019 minimum/maximum matching interval_program.cpp (preserve first
// NaN, order -0 below +0).
inline double ieee_min(double a, double b) noexcept {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (a == b) return std::signbit(a) ? a : b;
  return a < b ? a : b;
}
inline double ieee_max(double a, double b) noexcept {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (a == b) return std::signbit(a) ? b : a;
  return a > b ? a : b;
}

// f at points in binary64: proposals that bound nothing. When the validated
// bytecode uses the standard elementwise/reduction subset and fits in a small
// workspace, flatten it once into scalar ops and evaluate in cache-resident
// SoA blocks (switch outside the point loop for NEON vectorization and zero
// per-point branch mispredictions, with constants pre-seeded once) for both
// values and first-order value+gradient (no Hessian). Other programs fall back
// to evaluate_point / evaluate_jet.
class Field {
 public:
  explicit Field(const ProgramView& view) : view_(view) { compile(); }
  const ProgramView& view() const { return view_; }
  bool is_fast() const noexcept { return fast_; }
  static constexpr std::size_t block_capacity() noexcept { return kBlock; }
  double* block_x() noexcept { return val_buf_.data(); }
  double* block_y() noexcept { return val_buf_.data() + kBlock; }
  double* block_z() noexcept { return val_buf_.data() + 2 * kBlock; }
  const double* eval_loaded_block(std::size_t n) noexcept {
    double* __restrict base = val_buf_.data();
    execute_val_block(base, n);
    return base + static_cast<std::size_t>(output_) * kBlock;
  }

  void values(std::span<const double> points, double* result) {
    const auto count = static_cast<std::size_t>(points.size() / 3);
    if (count == 0) return;
    if (!fast_) {
      evaluate_point(view_, points.data(), static_cast<int64_t>(count), result, workspace_);
      return;
    }
    double* __restrict base = val_buf_.data();
    const std::size_t out_off = static_cast<std::size_t>(output_) * kBlock;
    for (std::size_t start = 0; start < count; start += kBlock) {
      const std::size_t n = std::min(kBlock, count - start);
      double* __restrict x = base;
      double* __restrict y = base + kBlock;
      double* __restrict z = base + 2 * kBlock;
      const double* src = points.data() + 3 * start;
      for (std::size_t j = 0; j < n; ++j) {
        x[j] = src[3 * j];
        y[j] = src[3 * j + 1];
        z[j] = src[3 * j + 2];
      }
      execute_val_block(base, n);
      const double* __restrict out = base + out_off;
      for (std::size_t j = 0; j < n; ++j) result[start + j] = out[j];
    }
  }

  // Newton-projects `points` onto the zero set by `steps` gradient steps using
  // batched first-order forward AD (value + gradient, no Hessian).
  void project_points(std::vector<Point>& points, int steps) {
    if (points.empty() || steps <= 0) return;
    if (!fast_) {
      std::vector<double> coordinates, jets_out;
      const auto width = static_cast<std::size_t>(jet_components(3));
      for (int step = 0; step < steps; ++step) {
        coordinates.clear();
        coordinates.reserve(3 * points.size());
        for (const Point& p : points) coordinates.insert(coordinates.end(), p.begin(), p.end());
        jets(coordinates, jets_out);
        for (std::size_t k = 0; k < points.size(); ++k) {
          const double* jet = jets_out.data() + width * k;
          const double g2 = jet[1] * jet[1] + jet[2] * jet[2] + jet[3] * jet[3];
          if (!(g2 > 0.0) || !std::isfinite(jet[0])) continue;
          for (std::size_t d = 0; d < 3; ++d) points[k][d] -= jet[0] * jet[1 + d] / g2;
        }
      }
      return;
    }
    ensure_vg_buf();
    double* __restrict base = vg_buf_.data();
    const std::size_t count = points.size();
    const std::size_t out_base = static_cast<std::size_t>(output_) * 4 * kBlock;
    for (std::size_t start = 0; start < count; start += kBlock) {
      const std::size_t n = std::min(kBlock, count - start);
      double* __restrict x = base;
      double* __restrict y = base + 4 * kBlock;
      double* __restrict z = base + 8 * kBlock;
      for (std::size_t j = 0; j < n; ++j) {
        x[j] = points[start + j][0];
        y[j] = points[start + j][1];
        z[j] = points[start + j][2];
      }
      for (int step = 0; step < steps; ++step) {
        execute_vg_block(base, n);
        const double* __restrict f = base + out_base;
        const double* __restrict gx = base + out_base + kBlock;
        const double* __restrict gy = base + out_base + 2 * kBlock;
        const double* __restrict gz = base + out_base + 3 * kBlock;
        bool changed = false;
        for (std::size_t j = 0; j < n; ++j) {
          const double g2 = gx[j] * gx[j] + gy[j] * gy[j] + gz[j] * gz[j];
          const double val = f[j];
          if (!(g2 > 0.0) || !std::isfinite(val)) continue;
          const double nx = x[j] - val * gx[j] / g2;
          const double ny = y[j] - val * gy[j] / g2;
          const double nz = z[j] - val * gz[j] / g2;
          if (nx != x[j] || ny != y[j] || nz != z[j]) {
            x[j] = nx;
            y[j] = ny;
            z[j] = nz;
            changed = true;
          }
        }
        if (!changed) break;
      }
      for (std::size_t j = 0; j < n; ++j) points[start + j] = {x[j], y[j], z[j]};
    }
  }

  // Value, gradient and Hessian (jet_components(3) entries per point).
  void jets(std::span<const double> points, std::vector<double>& result) {
    const auto count = static_cast<int64_t>(points.size() / 3);
    result.resize(static_cast<std::size_t>(count * jet_components(3)));
    if (count > 0) evaluate_jet(view_, points.data(), count, result.data(), workspace_);
  }

 private:
  static constexpr std::size_t kBlock = 64;
  static constexpr std::size_t kMaxFastWorkspace = 256;
  static constexpr std::int64_t kRowFields = 8;
  static constexpr std::uint8_t kOpGather = 1;
  static constexpr std::uint8_t kOpAdd = 2;
  static constexpr std::uint8_t kOpSubtract = 3;
  static constexpr std::uint8_t kOpMultiply = 5;
  static constexpr std::uint8_t kOpNegate = 6;
  static constexpr std::uint8_t kOpReduceSum = 7;
  static constexpr std::uint8_t kOpSine = 8;
  static constexpr std::uint8_t kOpCosine = 9;
  static constexpr std::uint8_t kOpSqrt = 10;
  static constexpr std::uint8_t kOpReciprocal = 11;
  static constexpr std::uint8_t kOpSquare = 12;
  static constexpr std::uint8_t kOpCompare = 13;
  static constexpr std::uint8_t kOpSelect = 14;
  static constexpr std::uint8_t kOpMaximum = 15;
  static constexpr std::uint8_t kOpMinimum = 16;
  static constexpr std::uint8_t kOpExp = 17;
  static constexpr std::uint8_t kOpLog = 18;
  static constexpr std::uint8_t kOpEqual = 19;
  static constexpr std::uint8_t kOpAbsolute = 20;
  static constexpr std::uint8_t kOpDecide = 23;
  static constexpr std::uint8_t kOpDivide = 25;
  static constexpr std::uint8_t kOpAtan = 26;
  static constexpr std::uint8_t kOpTan = 28;
  // Internal specializations for binary ops with a compile-time constant RHS.
  static constexpr std::uint8_t kOpAddConst = 100;
  static constexpr std::uint8_t kOpSubConst = 101;
  static constexpr std::uint8_t kOpMulConst = 102;

  struct ScalarOp {
    std::uint8_t opcode;
    std::uint16_t aux;
    std::uint16_t out;
    std::uint16_t a;
    std::uint16_t b;
    std::uint16_t c;
    double cval;
  };

  void execute_val_block(double* __restrict base, std::size_t n) const noexcept {
    for (const ScalarOp& op : ops_) {
      double* __restrict out = base + static_cast<std::size_t>(op.out) * kBlock;
      const double* __restrict a = base + static_cast<std::size_t>(op.a) * kBlock;
      switch (op.opcode) {
        case kOpGather:
        case kOpDecide:
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j];
          break;
        case kOpAddConst: {
          const double cv = op.cval;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] + cv;
          break;
        }
        case kOpSubConst: {
          const double cv = op.cval;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] - cv;
          break;
        }
        case kOpMulConst: {
          const double cv = op.cval;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] * cv;
          break;
        }
        case kOpAdd: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] + b[j];
          break;
        }
        case kOpSubtract: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] - b[j];
          break;
        }
        case kOpMultiply: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] * b[j];
          break;
        }
        case kOpDivide: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] / b[j];
          break;
        }
        case kOpNegate:
          for (std::size_t j = 0; j < n; ++j) out[j] = -a[j];
          break;
        case kOpReduceSum:
          if (op.aux == 2) {
            const double* __restrict a1 = a + kBlock;
            for (std::size_t j = 0; j < n; ++j) out[j] = a[j] + a1[j];
          } else if (op.aux == 3) {
            const double* __restrict a1 = a + kBlock;
            const double* __restrict a2 = a + 2 * kBlock;
            for (std::size_t j = 0; j < n; ++j) out[j] = (a[j] + a1[j]) + a2[j];
          } else if (op.aux == 0) {
            for (std::size_t j = 0; j < n; ++j) out[j] = 0.0;
          } else {
            for (std::size_t j = 0; j < n; ++j) out[j] = a[j];
            for (std::uint16_t t = 1; t < op.aux; ++t) {
              const double* __restrict at = a + static_cast<std::size_t>(t) * kBlock;
              for (std::size_t j = 0; j < n; ++j) out[j] += at[j];
            }
          }
          break;
        case kOpSquare:
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] * a[j];
          break;
        case kOpSqrt:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::sqrt(a[j]);
          break;
        case kOpReciprocal:
          for (std::size_t j = 0; j < n; ++j) out[j] = 1.0 / a[j];
          break;
        case kOpSine:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::sin(a[j]);
          break;
        case kOpCosine:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::cos(a[j]);
          break;
        case kOpExp:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::exp(a[j]);
          break;
        case kOpLog:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::log(a[j]);
          break;
        case kOpAtan:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::atan(a[j]);
          break;
        case kOpTan:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::tan(a[j]);
          break;
        case kOpAbsolute:
          for (std::size_t j = 0; j < n; ++j) out[j] = std::fabs(a[j]);
          break;
        case kOpCompare: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          if (op.aux == 0) {
            for (std::size_t j = 0; j < n; ++j) out[j] = a[j] > b[j] ? 1.0 : 0.0;
          } else {
            for (std::size_t j = 0; j < n; ++j) out[j] = a[j] >= b[j] ? 1.0 : 0.0;
          }
          break;
        }
        case kOpEqual: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          const bool want = op.aux == 0;
          for (std::size_t j = 0; j < n; ++j) out[j] = ((a[j] == b[j]) == want) ? 1.0 : 0.0;
          break;
        }
        case kOpSelect: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          const double* __restrict c = base + static_cast<std::size_t>(op.c) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] > 0.0 ? c[j] : b[j];
          break;
        }
        case kOpMaximum: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = ieee_max(a[j], b[j]);
          break;
        }
        case kOpMinimum: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kBlock;
          for (std::size_t j = 0; j < n; ++j) out[j] = ieee_min(a[j], b[j]);
          break;
        }
        default:
          break;
      }
    }
  }

  void execute_vg_block(double* __restrict base, std::size_t n) const noexcept {
    constexpr std::size_t kStride = 4 * kBlock;
    for (const ScalarOp& op : ops_) {
      double* __restrict out = base + static_cast<std::size_t>(op.out) * kStride;
      const double* __restrict a = base + static_cast<std::size_t>(op.a) * kStride;
      switch (op.opcode) {
        case kOpGather:
        case kOpDecide:
          for (std::size_t c = 0; c < 4; ++c)
            for (std::size_t j = 0; j < n; ++j) out[c * kBlock + j] = a[c * kBlock + j];
          break;
        case kOpAddConst: {
          const double cv = op.cval;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] + cv;
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = a[i * kBlock + j];
          break;
        }
        case kOpSubConst: {
          const double cv = op.cval;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] + (-1.0) * cv;
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = a[i * kBlock + j];
          break;
        }
        case kOpMulConst: {
          const double cv = op.cval;
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = cv * a[i * kBlock + j];
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] * cv;
          break;
        }
        case kOpAdd: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t c = 0; c < 4; ++c)
            for (std::size_t j = 0; j < n; ++j)
              out[c * kBlock + j] = a[c * kBlock + j] + b[c * kBlock + j];
          break;
        }
        case kOpSubtract: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t c = 0; c < 4; ++c)
            for (std::size_t j = 0; j < n; ++j)
              out[c * kBlock + j] = a[c * kBlock + j] + (-1.0) * b[c * kBlock + j];
          break;
        }
        case kOpMultiply: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j)
              out[i * kBlock + j] = a[j] * b[i * kBlock + j] + b[j] * a[i * kBlock + j];
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] * b[j];
          break;
        }
        case kOpDivide: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t j = 0; j < n; ++j) out[j] = a[j] / b[j];
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j)
              out[i * kBlock + j] = (a[i * kBlock + j] - out[j] * b[i * kBlock + j]) / b[j];
          break;
        }
        case kOpNegate:
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = -1.0 * a[i * kBlock + j];
          for (std::size_t j = 0; j < n; ++j) out[j] = -a[j];
          break;
        case kOpReduceSum:
          for (std::size_t c = 0; c < 4; ++c) {
            double* __restrict oc = out + c * kBlock;
            if (op.aux == 2) {
              const double* __restrict a0 = a + c * kBlock;
              const double* __restrict a1 = a + kStride + c * kBlock;
              for (std::size_t j = 0; j < n; ++j) oc[j] = a0[j] + a1[j];
            } else if (op.aux == 3) {
              const double* __restrict a0 = a + c * kBlock;
              const double* __restrict a1 = a + kStride + c * kBlock;
              const double* __restrict a2 = a + 2 * kStride + c * kBlock;
              for (std::size_t j = 0; j < n; ++j) oc[j] = (a0[j] + a1[j]) + a2[j];
            } else if (op.aux == 0) {
              for (std::size_t j = 0; j < n; ++j) oc[j] = 0.0;
            } else {
              const double* __restrict a0 = a + c * kBlock;
              for (std::size_t j = 0; j < n; ++j) oc[j] = a0[j];
              for (std::uint16_t t = 1; t < op.aux; ++t) {
                const double* __restrict at = a + static_cast<std::size_t>(t) * kStride + c * kBlock;
                for (std::size_t j = 0; j < n; ++j) oc[j] += at[j];
              }
            }
          }
          break;
        case kOpSquare: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            const double x = a[j];
            d1[j] = 2.0 * x;
            out[j] = x * x;
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpSqrt: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            const double v = std::sqrt(a[j]);
            out[j] = v;
            d1[j] = 0.5 / v;
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpReciprocal: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            const double v = 1.0 / a[j];
            out[j] = v;
            d1[j] = -v * v;
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpSine: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            out[j] = std::sin(a[j]);
            d1[j] = std::cos(a[j]);
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpCosine: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            out[j] = std::cos(a[j]);
            d1[j] = -std::sin(a[j]);
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpExp: {
          for (std::size_t j = 0; j < n; ++j) out[j] = std::exp(a[j]);
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = out[j] * a[i * kBlock + j];
          break;
        }
        case kOpLog: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            out[j] = std::log(a[j]);
            d1[j] = 1.0 / a[j];
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpAtan: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            const double x = a[j];
            out[j] = std::atan(x);
            d1[j] = 1.0 / (1.0 + x * x);
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpTan: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            const double v = std::tan(a[j]);
            out[j] = v;
            d1[j] = 1.0 + v * v;
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpAbsolute: {
          double d1[kBlock];
          for (std::size_t j = 0; j < n; ++j) {
            const double x = a[j];
            out[j] = std::fabs(x);
            d1[j] = std::signbit(x) ? -1.0 : 1.0;
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = d1[j] * a[i * kBlock + j];
          break;
        }
        case kOpSelect: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          const double* __restrict c = base + static_cast<std::size_t>(op.c) * kStride;
          for (std::size_t j = 0; j < n; ++j) {
            const double* src = a[j] > 0.0 ? c : b;
            for (std::size_t comp = 0; comp < 4; ++comp) out[comp * kBlock + j] = src[comp * kBlock + j];
          }
          break;
        }
        case kOpMaximum: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t j = 0; j < n; ++j) {
            const double* src = b[j] > a[j] ? b : a;
            for (std::size_t comp = 0; comp < 4; ++comp) out[comp * kBlock + j] = src[comp * kBlock + j];
          }
          break;
        }
        case kOpMinimum: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t j = 0; j < n; ++j) {
            const double* src = b[j] < a[j] ? b : a;
            for (std::size_t comp = 0; comp < 4; ++comp) out[comp * kBlock + j] = src[comp * kBlock + j];
          }
          break;
        }
        case kOpCompare:
        case kOpEqual: {
          const double* __restrict b = base + static_cast<std::size_t>(op.b) * kStride;
          for (std::size_t j = 0; j < n; ++j) {
            const bool truth = op.opcode == kOpCompare
                                   ? (op.aux == 0 ? a[j] > b[j] : a[j] >= b[j])
                                   : ((a[j] == b[j]) == (op.aux == 0));
            out[j] = truth ? 1.0 : 0.0;
          }
          for (std::size_t i = 1; i < 4; ++i)
            for (std::size_t j = 0; j < n; ++j) out[i * kBlock + j] = 0.0;
          break;
        }
        default:
          break;
      }
    }
  }

  void ensure_vg_buf() {
    if (!vg_buf_.empty()) return;
    const auto ws = static_cast<std::size_t>(view_.workspace_size);
    vg_buf_.assign(ws * 4 * kBlock, 0.0);
    for (std::size_t e = 0; e < ws; ++e) {
      std::fill_n(vg_buf_.data() + e * 4 * kBlock, kBlock, view_.seeds[e]);
    }
    for (std::size_t d = 0; d < 3; ++d) {
      std::fill_n(vg_buf_.data() + (d * 4 + 1 + d) * kBlock, kBlock, 1.0);
    }
  }

  void compile() {
    fast_ = false;
    if (view_.input_dimension != 3 || view_.output_count != 1 || view_.workspace_size < 3 ||
        static_cast<std::size_t>(view_.workspace_size) > kMaxFastWorkspace) {
      return;
    }
    output_ = static_cast<std::uint16_t>(view_.outputs[0]);
    const auto addr = [&](int64_t base, int64_t index, int64_t k) -> std::uint16_t {
      return static_cast<std::uint16_t>(base + (index < 0 ? k : view_.indices[index + k]));
    };
    const auto ws = static_cast<std::size_t>(view_.workspace_size);
    std::vector<std::uint8_t> written(ws, 0);
    written[0] = written[1] = written[2] = 1;
    ops_.clear();
    for (int64_t r = 0; r < view_.instruction_count; ++r) {
      const int64_t* f = view_.instructions + r * kRowFields;
      const int64_t opcode = f[0], out = f[1], size = f[2], a = f[3], a_idx = f[4], b = f[5],
                    b_idx = f[6], aux = f[7];
      switch (opcode) {
        case kOpReduceSum:
          if (aux < 0 || aux > 255) return;
          for (int64_t k = 0; k < size; ++k) {
            const auto dst = static_cast<std::uint16_t>(out + k);
            ops_.push_back({static_cast<std::uint8_t>(opcode), static_cast<std::uint16_t>(aux),
                            dst, static_cast<std::uint16_t>(a + k * aux), 0, 0, 0.0});
            written[dst] = 1;
          }
          break;
        case kOpSelect:
          for (int64_t k = 0; k < size; ++k) {
            const auto dst = static_cast<std::uint16_t>(out + k);
            ops_.push_back({static_cast<std::uint8_t>(opcode), 0, dst, addr(a, a_idx, k),
                            static_cast<std::uint16_t>(b + k),
                            static_cast<std::uint16_t>(b + aux + k), 0.0});
            written[dst] = 1;
          }
          break;
        case kOpGather:
        case kOpDecide:
        case kOpNegate:
        case kOpAbsolute:
        case kOpSquare:
        case kOpSqrt:
        case kOpReciprocal:
        case kOpSine:
        case kOpCosine:
        case kOpExp:
        case kOpLog:
        case kOpAtan:
        case kOpTan:
          for (int64_t k = 0; k < size; ++k) {
            const auto dst = static_cast<std::uint16_t>(out + k);
            ops_.push_back({static_cast<std::uint8_t>(opcode), 0, dst, addr(a, a_idx, k), 0, 0,
                            0.0});
            written[dst] = 1;
          }
          break;
        case kOpAdd:
        case kOpSubtract:
        case kOpMultiply:
        case kOpDivide:
        case kOpMaximum:
        case kOpMinimum:
        case kOpCompare:
        case kOpEqual:
          for (int64_t k = 0; k < size; ++k) {
            const auto dst = static_cast<std::uint16_t>(out + k);
            const auto lhs = addr(a, a_idx, k);
            const auto rhs = addr(b, b_idx, k);
            std::uint8_t op_code = static_cast<std::uint8_t>(opcode);
            double cval = 0.0;
            if (!written[rhs]) {
              if (opcode == kOpAdd) {
                op_code = kOpAddConst;
                cval = view_.seeds[rhs];
              } else if (opcode == kOpSubtract) {
                op_code = kOpSubConst;
                cval = view_.seeds[rhs];
              } else if (opcode == kOpMultiply) {
                op_code = kOpMulConst;
                cval = view_.seeds[rhs];
              }
            }
            ops_.push_back({op_code, static_cast<std::uint16_t>(aux), dst, lhs, rhs, 0, cval});
            written[dst] = 1;
          }
          break;
        default:
          return;
      }
    }
    val_buf_.resize(ws * kBlock);
    for (std::size_t e = 0; e < ws; ++e) {
      std::fill_n(val_buf_.data() + e * kBlock, kBlock, view_.seeds[e]);
    }
    fast_ = true;
  }

  ProgramView view_;
  PointWorkspace workspace_;
  bool fast_ = false;
  std::uint16_t output_ = 0;
  std::vector<ScalarOp> ops_;
  std::vector<double> val_buf_;
  std::vector<double> vg_buf_;
};

// Illinois regula falsi on every bracket at once, one batched evaluation per
// step, leaving each bracket's root where it stopped; a bracket leaves the
// batch when f vanishes at its trial point, its step is negligible or its
// parameter interval closes.
class RootSolver {
 public:
  void solve(Field& field, std::span<Bracket> brackets) {
    const std::size_t n = brackets.size();
    if (n == 0) return;
    if (field.is_fast() && n <= Field::block_capacity()) {
      solve_small(field, brackets);
      return;
    }
    low_.assign(n, 0.0);
    high_.assign(n, 1.0);
    t_.assign(n, 0.5);
    previous_.assign(n, -1.0);
    side_.assign(n, 0);
    f_low_.resize(n);
    f_high_.resize(n);
    delta_.resize(3 * n);
    active_.resize(n);
    for (std::size_t c = 0; c < n; ++c) {
      const Bracket& b = brackets[c];
      f_low_[c] = b.f_negative;
      f_high_[c] = b.f_positive;
      delta_[3 * c] = b.positive[0] - b.negative[0];
      delta_[3 * c + 1] = b.positive[1] - b.negative[1];
      delta_[3 * c + 2] = b.positive[2] - b.negative[2];
      active_[c] = c;
    }
    for (int iteration = 0; iteration < kRootIterations && !active_.empty(); ++iteration) {
      const std::size_t count = active_.size();
      trial_.resize(3 * count);
      double* __restrict tr = trial_.data();
      for (std::size_t k = 0; k < count; ++k) {
        const std::size_t c = active_[k];
        double t = low_[c] - f_low_[c] * (high_[c] - low_[c]) / (f_high_[c] - f_low_[c]);
        if (!(t > low_[c] && t < high_[c])) t = 0.5 * (low_[c] + high_[c]);
        t_[c] = t;
        const Bracket& b = brackets[c];
        tr[3 * k] = b.negative[0] + t * delta_[3 * c];
        tr[3 * k + 1] = b.negative[1] + t * delta_[3 * c + 1];
        tr[3 * k + 2] = b.negative[2] + t * delta_[3 * c + 2];
      }
      values_.resize(count);
      field.values(trial_, values_.data());
      std::size_t kept = 0;
      for (std::size_t k = 0; k < count; ++k) {
        const std::size_t c = active_[k];
        const double value = values_[k];
        const double tc = t_[c];
        const double step = std::abs(tc - previous_[c]);
        previous_[c] = tc;
        if (value == 0.0 || step <= kRootStep || high_[c] - low_[c] <= kRootWidth) continue;
        if (value < 0.0) {
          low_[c] = tc;
          f_low_[c] = value;
          if (side_[c] == -1) f_high_[c] *= 0.5;
          side_[c] = -1;
        } else {
          high_[c] = tc;
          f_high_[c] = value;
          if (side_[c] == 1) f_low_[c] *= 0.5;
          side_[c] = 1;
        }
        active_[kept++] = c;
      }
      active_.resize(kept);
    }
    for (std::size_t c = 0; c < n; ++c) finalize_bracket(brackets[c], t_[c]);
  }

 private:
  static void finalize_bracket(Bracket& b, double tc) noexcept {
    std::uint64_t hash = 0;
    for (std::size_t d = 0; d < 3; ++d)
      hash = splitmix64(hash ^ std::bit_cast<std::uint64_t>(b.negative[d]) ^
                        std::rotl(std::bit_cast<std::uint64_t>(b.positive[d]), 17));
    // A uniform offset in [-1, 1) times kRootOffset, hashed from the
    // bracket's endpoints only (kRootOffset).
    const double unit = static_cast<double>(hash >> 11) * 0x1p-52 - 1.0;
    const double t = std::clamp(tc + kRootOffset * unit, 0.0, 1.0);
    b.root = along(b.negative, b.positive, t);
  }

  // Stack-allocated zero-transpose path for batches fitting in one SoA block.
  static void solve_small(Field& field, std::span<Bracket> brackets) noexcept {
    constexpr std::size_t kCap = Field::block_capacity();
    struct State {
      double low, high, t, prev, f_low, f_high;
      double neg[3], delta[3];
      int side;
    };
    State st[kCap];
    std::uint8_t active[kCap];
    const std::size_t n = brackets.size();
    for (std::size_t c = 0; c < n; ++c) {
      const Bracket& b = brackets[c];
      st[c] = {0.0,
               1.0,
               0.5,
               -1.0,
               b.f_negative,
               b.f_positive,
               {b.negative[0], b.negative[1], b.negative[2]},
               {b.positive[0] - b.negative[0], b.positive[1] - b.negative[1],
                b.positive[2] - b.negative[2]},
               0};
      active[c] = static_cast<std::uint8_t>(c);
    }
    double* __restrict bx = field.block_x();
    double* __restrict by = field.block_y();
    double* __restrict bz = field.block_z();
    std::size_t count = n;
    for (int iteration = 0; iteration < kRootIterations && count > 0; ++iteration) {
      for (std::size_t k = 0; k < count; ++k) {
        State& s = st[active[k]];
        double t = s.low - s.f_low * (s.high - s.low) / (s.f_high - s.f_low);
        if (!(t > s.low && t < s.high)) t = 0.5 * (s.low + s.high);
        s.t = t;
        bx[k] = s.neg[0] + t * s.delta[0];
        by[k] = s.neg[1] + t * s.delta[1];
        bz[k] = s.neg[2] + t * s.delta[2];
      }
      const double* __restrict vals = field.eval_loaded_block(count);
      std::size_t kept = 0;
      for (std::size_t k = 0; k < count; ++k) {
        const std::uint8_t c = active[k];
        State& s = st[c];
        const double value = vals[k];
        const double tc = s.t;
        const double step = std::abs(tc - s.prev);
        s.prev = tc;
        if (value == 0.0 || step <= kRootStep || s.high - s.low <= kRootWidth) continue;
        if (value < 0.0) {
          s.low = tc;
          s.f_low = value;
          if (s.side == -1) s.f_high *= 0.5;
          s.side = -1;
        } else {
          s.high = tc;
          s.f_high = value;
          if (s.side == 1) s.f_low *= 0.5;
          s.side = 1;
        }
        active[kept++] = c;
      }
      count = kept;
    }
    for (std::size_t c = 0; c < n; ++c) finalize_bracket(brackets[c], st[c].t);
  }

  std::vector<double> low_, high_, t_, previous_, f_low_, f_high_, delta_, trial_, values_;
  std::vector<int> side_;
  std::vector<std::size_t> active_;
};

// Points by uniform spatial cells, for nearest-neighbor-within-radius queries.
// Uses a flat 3D array when a bounding box fits within kMaxDenseCells (zero
// hashing), falling back to an open-addressed KeyMap otherwise.
class PointGrid {
 public:
  PointGrid(double cell, std::size_t expected, const double* box_lower = nullptr,
            const double* box_upper = nullptr)
      : cell_(cell) {
    next_.reserve(expected);
    points_.reserve(expected);
    if (box_lower != nullptr && box_upper != nullptr && cell > 0.0) {
      int64_t dims[3];
      bool ok = true;
      std::size_t total = 1;
      for (int d = 0; d < 3; ++d) {
        base_[d] = static_cast<int64_t>(std::floor(box_lower[d] / cell_));
        const int64_t hi = static_cast<int64_t>(std::floor(box_upper[d] / cell_));
        dims[d] = hi - base_[d] + 1;
        if (dims[d] <= 0 || dims[d] > static_cast<int64_t>(kMaxDenseCells) ||
            total > kMaxDenseCells / static_cast<std::size_t>(dims[d])) {
          ok = false;
          break;
        }
        total *= static_cast<std::size_t>(dims[d]);
      }
      if (ok) {
        dense_ = true;
        nx_ = dims[0];
        ny_ = dims[1];
        nz_ = dims[2];
        dense_heads_.assign(total, -1);
        return;
      }
    }
    heads_.emplace(expected);
  }
  void add(const Point& p) {
    const std::array<int64_t, 3> c = cell_of(p);
    const auto id = static_cast<int32_t>(points_.size());
    points_.push_back(p);
    if (dense_) {
      const int64_t ix = c[0] - base_[0], iy = c[1] - base_[1], iz = c[2] - base_[2];
      if (ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_) {
        const auto idx = static_cast<std::size_t>((ix * ny_ + iy) * nz_ + iz);
        next_.push_back(dense_heads_[idx]);
        dense_heads_[idx] = id;
      } else {
        next_.push_back(-1);
        overflow_.push_back(id);
      }
      return;
    }
    int64_t& head = heads_->slot(key_of(c));
    next_.push_back(static_cast<int32_t>(head));
    head = id;
  }
  // Whether some point lies within `radius` (at most the cell size) of p.
  bool near(const Point& p, double radius) const {
    const double gx = p[0] / cell_, gy = p[1] / cell_, gz = p[2] / cell_;
    const int64_t cx = static_cast<int64_t>(std::floor(gx));
    const int64_t cy = static_cast<int64_t>(std::floor(gy));
    const int64_t cz = static_cast<int64_t>(std::floor(gz));
    const double fx = gx - static_cast<double>(cx);
    const double fy = gy - static_cast<double>(cy);
    const double fz = gz - static_cast<double>(cz);
    const double r2 = radius * radius;
    const double r_norm = radius / cell_ + 1e-9;
    const double r_norm2 = r_norm * r_norm;
    const auto check_list = [&](int32_t e) -> bool {
      for (; e >= 0; e = next_[static_cast<std::size_t>(e)]) {
        const Point& q = points_[static_cast<std::size_t>(e)];
        const double dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
        if (dx * dx + dy * dy + dz * dz <= r2) return true;
      }
      return false;
    };
    const auto axis_gap = [](double f, int64_t d) noexcept {
      if (d < 0) return std::max(0.0, f - 1e-9);
      if (d > 0) return std::max(0.0, 1.0 - f - 1e-9);
      return 0.0;
    };
    if (dense_) {
      const int64_t bx = cx - base_[0], by = cy - base_[1], bz = cz - base_[2];
      if (bx >= 0 && bx < nx_ && by >= 0 && by < ny_ && bz >= 0 && bz < nz_) {
        const int32_t e = dense_heads_[static_cast<std::size_t>((bx * ny_ + by) * nz_ + bz)];
        if (e >= 0 && check_list(e)) return true;
      }
      const int64_t ix0 = std::max<int64_t>(0, bx - 1), ix1 = std::min<int64_t>(nx_ - 1, bx + 1);
      const int64_t iy0 = std::max<int64_t>(0, by - 1), iy1 = std::min<int64_t>(ny_ - 1, by + 1);
      const int64_t iz0 = std::max<int64_t>(0, bz - 1), iz1 = std::min<int64_t>(nz_ - 1, bz + 1);
      for (int64_t ix = ix0; ix <= ix1; ++ix) {
        const double dx = axis_gap(fx, ix - bx);
        const double dx2 = dx * dx;
        if (dx2 > r_norm2) continue;
        for (int64_t iy = iy0; iy <= iy1; ++iy) {
          const double dy = axis_gap(fy, iy - by);
          const double dxy2 = dx2 + dy * dy;
          if (dxy2 > r_norm2) continue;
          const std::size_t row = static_cast<std::size_t>((ix * ny_ + iy) * nz_);
          for (int64_t iz = iz0; iz <= iz1; ++iz) {
            if (ix == bx && iy == by && iz == bz) continue;
            const double dz = axis_gap(fz, iz - bz);
            if (dxy2 + dz * dz > r_norm2) continue;
            const int32_t e = dense_heads_[row + static_cast<std::size_t>(iz)];
            if (e >= 0 && check_list(e)) return true;
          }
        }
      }
      for (int32_t id : overflow_) {
        const Point& q = points_[static_cast<std::size_t>(id)];
        const double dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
        if (dx * dx + dy * dy + dz * dz <= r2) return true;
      }
      return false;
    }
    // Sparse path: check center cell first, then sphere-pruned neighbors.
    if (const int64_t e = heads_->find(key_of({cx, cy, cz})); e >= 0 && check_list(static_cast<int32_t>(e)))
      return true;
    for (int64_t i = -1; i <= 1; ++i) {
      const double dx = axis_gap(fx, i);
      const double dx2 = dx * dx;
      if (dx2 > r_norm2) continue;
      for (int64_t j = -1; j <= 1; ++j) {
        const double dy = axis_gap(fy, j);
        const double dxy2 = dx2 + dy * dy;
        if (dxy2 > r_norm2) continue;
        for (int64_t k = -1; k <= 1; ++k) {
          if (i == 0 && j == 0 && k == 0) continue;
          const double dz = axis_gap(fz, k);
          if (dxy2 + dz * dz > r_norm2) continue;
          if (const int64_t e = heads_->find(key_of({cx + i, cy + j, cz + k}));
              e >= 0 && check_list(static_cast<int32_t>(e)))
            return true;
        }
      }
    }
    return false;
  }

 private:
  static constexpr std::size_t kMaxDenseCells = 32768;
  std::array<int64_t, 3> cell_of(const Point& p) const {
    return {static_cast<int64_t>(std::floor(p[0] / cell_)), static_cast<int64_t>(std::floor(p[1] / cell_)),
            static_cast<int64_t>(std::floor(p[2] / cell_))};
  }
  // 21 bits per axis: the grid spans far fewer cells than 2^20 per axis.
  static std::uint64_t key_of(const std::array<int64_t, 3>& c) {
    constexpr int64_t kMask = (int64_t{1} << 21) - 1;
    return (static_cast<std::uint64_t>(c[0] & kMask) << 42) |
           (static_cast<std::uint64_t>(c[1] & kMask) << 21) | static_cast<std::uint64_t>(c[2] & kMask);
  }
  double cell_;
  bool dense_ = false;
  int64_t base_[3]{}, nx_ = 0, ny_ = 0, nz_ = 0;
  std::vector<int32_t> dense_heads_;
  std::vector<int32_t> overflow_;
  std::optional<KeyMap> heads_;
  std::vector<int32_t> next_;
  std::vector<Point> points_;
};

struct Criteria {
  double edge2;        // longest restricted facet edge, squared
  double distance2;    // facet circumcenter to ball center, squared
  double angle_sine2;  // smallest facet angle's sine, squared
  double size2;        // inside cell circumradius, squared
  double ratio2;       // inside cell radius-edge ratio, squared
};

struct BallData {
  Point center{};
  double radius2 = 0.0;
};

struct SphereData {
  Point center{};
  double radius2 = 0.0;
};

struct CellData {
  double value = 0.0;
  double shortest2 = 0.0;
  std::uint32_t generation = 0;
  // Index into RestrictedMesher::balls_ for each restricted facet slot, or -1.
  std::array<int32_t, 4> ball{-1, -1, -1, -1};
};

struct FacetEntry {
  double key;
  int32_t cell, slot, other;
  std::uint32_t cell_generation, other_generation;
  Point center;
  bool operator<(const FacetEntry& other_entry) const { return key < other_entry.key; }
};

struct CellEntry {
  double key;
  int32_t cell;
  std::uint32_t generation;
  bool operator<(const CellEntry& other) const { return key < other.key; }
};

// The surface topology of outward triangles: whether every directed edge
// occurs once with its reverse, the edge-connected components, the Euler
// characteristic, and the vertices of non-manifold edges or pinched links.
struct SurfaceTopology {
  bool manifold = true;
  int64_t components = 0;
  int64_t euler = 0;
  std::vector<int32_t> singular;
};

int32_t find_root(std::vector<int32_t>& parent, int32_t x) {
  while (parent[static_cast<std::size_t>(x)] != x) {
    parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    x = parent[static_cast<std::size_t>(x)];
  }
  return x;
}

void unite(std::vector<int32_t>& parent, int32_t a, int32_t b) {
  a = find_root(parent, a);
  b = find_root(parent, b);
  if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
}

struct TopologyScratch {
  struct HalfEdge {
    int32_t head;
    int32_t face;
    int32_t next_corner;
  };
  struct EdgeSlot {
    std::uint64_t key;
    // Packed (face << 2) | slot for forward (tail < head) and reverse (tail > head) edges, or -1.
    int32_t fwd;
    int32_t rev;
  };
  std::vector<EdgeSlot> slots;
  std::vector<std::uint32_t> slot_stamp;
  std::uint32_t epoch = 0;
  std::vector<int32_t> first_corner, degree, next_at_vertex;
  std::vector<int32_t> offsets, cursor, face_parent, corner_parent, fans;
  std::vector<HalfEdge> tails;
};

SurfaceTopology surface_topology_general(std::span<const int32_t> faces, TopologyScratch& scratch) {
  SurfaceTopology result;
  const std::size_t face_count = faces.size() / 3;
  std::size_t vertex_count = 0;
  for (int32_t v : faces) vertex_count = std::max(vertex_count, static_cast<std::size_t>(v) + 1);
  scratch.offsets.assign(vertex_count + 1, 0);
  scratch.cursor.resize(vertex_count);
  scratch.tails.resize(faces.size());
  for (int32_t v : faces) ++scratch.offsets[static_cast<std::size_t>(v) + 1];
  for (std::size_t v = 0; v < vertex_count; ++v) {
    scratch.offsets[v + 1] += scratch.offsets[v];
    scratch.cursor[v] = scratch.offsets[v];
  }
  for (std::size_t f = 0; f < face_count; ++f) {
    const auto fi = static_cast<int32_t>(f);
    const auto base = static_cast<int32_t>(3 * f);
    const int32_t v0 = faces[3 * f], v1 = faces[3 * f + 1], v2 = faces[3 * f + 2];
    scratch.tails[static_cast<std::size_t>(scratch.cursor[static_cast<std::size_t>(v0)]++)] = {v1, fi, base + 1};
    scratch.tails[static_cast<std::size_t>(scratch.cursor[static_cast<std::size_t>(v1)]++)] = {v2, fi, base + 2};
    scratch.tails[static_cast<std::size_t>(scratch.cursor[static_cast<std::size_t>(v2)]++)] = {v0, fi, base};
  }
  scratch.face_parent.resize(face_count);
  scratch.corner_parent.resize(faces.size());
  std::iota(scratch.face_parent.begin(), scratch.face_parent.end(), 0);
  std::iota(scratch.corner_parent.begin(), scratch.corner_parent.end(), 0);
  const auto mark_singular = [&](int32_t tail, int32_t head) {
    result.manifold = false;
    result.singular.push_back(tail);
    result.singular.push_back(head);
  };
  const int32_t* offsets = scratch.offsets.data();
  const TopologyScratch::HalfEdge* tails = scratch.tails.data();
  for (std::size_t f = 0; f < face_count; ++f) {
    const auto fi = static_cast<int32_t>(f);
    const int32_t v[3] = {faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]};
    for (int r = 0; r < 3; ++r) {
      const int32_t corner = static_cast<int32_t>(3 * f) + r;
      const int32_t tail = v[r], head = v[r == 2 ? 0 : r + 1];
      const int32_t first = offsets[tail], last = offsets[tail + 1];
      int matches = 0;
      for (int32_t k = first; k < last; ++k) {
        if (tails[k].head == head && ++matches > 1) break;
      }
      if (matches > 1) {
        mark_singular(tail, head);
        continue;
      }
      const int32_t reverse_first = offsets[head], reverse_last = offsets[head + 1];
      int32_t rev_idx = -1;
      for (int32_t k = reverse_first; k < reverse_last; ++k) {
        if (tails[k].head == tail) {
          rev_idx = k;
          break;
        }
      }
      if (rev_idx < 0) {
        mark_singular(tail, head);
        continue;
      }
      unite(scratch.face_parent, fi, tails[rev_idx].face);
      unite(scratch.corner_parent, corner, tails[rev_idx].next_corner);
    }
  }
  for (std::size_t f = 0; f < face_count; ++f)
    result.components +=
        find_root(scratch.face_parent, static_cast<int32_t>(f)) == static_cast<int32_t>(f);
  scratch.fans.assign(vertex_count, 0);
  int64_t vertices = 0;
  for (std::size_t c = 0; c < faces.size(); ++c) {
    if (find_root(scratch.corner_parent, static_cast<int32_t>(c)) != static_cast<int32_t>(c)) continue;
    if (scratch.fans[static_cast<std::size_t>(faces[c])]++ == 0) {
      ++vertices;
    } else {
      result.manifold = false;
      result.singular.push_back(faces[c]);
    }
  }
  result.euler = vertices - static_cast<int64_t>(faces.size() / 2) + static_cast<int64_t>(face_count);
  std::sort(result.singular.begin(), result.singular.end());
  result.singular.erase(std::unique(result.singular.begin(), result.singular.end()),
                        result.singular.end());
  return result;
}

// Fast manifold path: open-addressed undirected edge pairing + per-vertex
// cyclic fan verification (no corner union-find). Any non-manifold, open, or
// pinched surface delegates to surface_topology_general for exact singular
// vertices and Euler characteristic.
SurfaceTopology surface_topology(std::span<const int32_t> faces, TopologyScratch& scratch) {
  const std::size_t face_count = faces.size() / 3;
  if (face_count == 0) return {};
  if ((faces.size() & 1U) != 0) return surface_topology_general(faces, scratch);
  std::size_t vertex_count = 0;
  for (int32_t v : faces) vertex_count = std::max(vertex_count, static_cast<std::size_t>(v) + 1);
  const std::size_t table_size = std::bit_ceil(std::max<std::size_t>(16, faces.size() * 2));
  if (scratch.slots.size() < table_size) {
    scratch.slots.resize(table_size);
    scratch.slot_stamp.assign(table_size, 0);
    scratch.epoch = 0;
  }
  if (++scratch.epoch == 0) {
    std::fill(scratch.slot_stamp.begin(), scratch.slot_stamp.end(), 0);
    scratch.epoch = 1;
  }
  const std::size_t mask = scratch.slots.size() - 1;
  const std::uint32_t epoch = scratch.epoch;
  scratch.first_corner.assign(vertex_count, -1);
  scratch.degree.assign(vertex_count, 0);
  scratch.next_at_vertex.resize(faces.size());
  scratch.face_parent.resize(face_count);
  std::iota(scratch.face_parent.begin(), scratch.face_parent.end(), 0);
  TopologyScratch::EdgeSlot* __restrict slots = scratch.slots.data();
  std::uint32_t* __restrict stamps = scratch.slot_stamp.data();
  int32_t* __restrict first_corner = scratch.first_corner.data();
  int32_t* __restrict degree = scratch.degree.data();
  int32_t* __restrict next_at_vertex = scratch.next_at_vertex.data();
  std::size_t unmatched = 0;
  for (std::size_t f = 0; f < face_count; ++f) {
    const auto fi = static_cast<int32_t>(f);
    const int32_t base = static_cast<int32_t>(3 * f);
    const int32_t v[3] = {faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]};
    for (int r = 0; r < 3; ++r) {
      const int32_t tail = v[r];
      const int32_t head = v[r == 2 ? 0 : r + 1];
      if (tail == head) return surface_topology_general(faces, scratch);
      const int32_t corner = base + r;
      if (first_corner[tail] < 0) first_corner[tail] = corner;
      ++degree[tail];
      const bool forward = tail < head;
      const std::uint64_t key =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(forward ? tail : head)) << 32) |
          static_cast<std::uint32_t>(forward ? head : tail);
      std::size_t pos = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
      while (stamps[pos] == epoch && slots[pos].key != key) pos = (pos + 1) & mask;
      const int32_t packed = (fi << 2) | r;
      if (stamps[pos] != epoch) {
        stamps[pos] = epoch;
        slots[pos] = {key, forward ? packed : -1, forward ? -1 : packed};
        ++unmatched;
      } else {
        int32_t& mine = forward ? slots[pos].fwd : slots[pos].rev;
        const int32_t other = forward ? slots[pos].rev : slots[pos].fwd;
        if (mine >= 0) return surface_topology_general(faces, scratch);
        mine = packed;
        --unmatched;
        const int32_t of = other >> 2;
        const int or_slot = other & 3;
        const int32_t other_corner = 3 * of + or_slot;
        const int32_t other_next = 3 * of + (or_slot == 2 ? 0 : or_slot + 1);
        const int32_t my_next = base + (r == 2 ? 0 : r + 1);
        next_at_vertex[corner] = other_next;
        next_at_vertex[other_corner] = my_next;
        unite(scratch.face_parent, fi, of);
      }
    }
  }
  if (unmatched != 0) return surface_topology_general(faces, scratch);
  int64_t active_vertices = 0;
  for (std::size_t v = 0; v < vertex_count; ++v) {
    const int32_t deg = degree[v];
    if (deg == 0) continue;
    ++active_vertices;
    const int32_t start = first_corner[v];
    int32_t cur = next_at_vertex[static_cast<std::size_t>(start)];
    int32_t steps = 1;
    while (cur != start && steps < deg) {
      cur = next_at_vertex[static_cast<std::size_t>(cur)];
      ++steps;
    }
    if (cur != start || steps != deg) return surface_topology_general(faces, scratch);
  }
  SurfaceTopology result;
  result.manifold = true;
  for (std::size_t f = 0; f < face_count; ++f)
    result.components +=
        find_root(scratch.face_parent, static_cast<int32_t>(f)) == static_cast<int32_t>(f);
  result.euler = active_vertices - static_cast<int64_t>(faces.size() / 2) + static_cast<int64_t>(face_count);
  return result;
}

// One Delaunay triangulation of the seeds, the refinement points and eight
// far corners (vertices 0 to 7), with the circumcenter, its f value and the
// inside label of every cell slot.
//
// Termination is a packing argument (Ruppert; Shewchuk). The seeds are
// pairwise at least `separation` apart, and a point is inserted only inside
// the far corners' box and at least `separation` from every vertex present:
// in a Delaunay triangulation its nearest vertex is a vertex of its conflict
// zone, which the cell refinement walks anyway, and for a facet's surface
// Delaunay ball center it is a vertex of the facet, the ball being empty.
// Balls of radius separation / 2 around the inserted points are disjoint
// and lie in the far box grown by separation / 2, so their count is at
// most that volume over a ball's, whatever the field. A facet whose
// refinement point the guard refuses is recorded, and still violating the
// criteria at the end, makes the construction PHX_MC_UNRESOLVED; a cell is
// left to the quality stage. When the cavity of an admitted point, which
// holds its nearest vertex, shows a vertex closer than the bound (rounding
// in a binary64 circumcenter or ball center), the refinement stops at once
// with PHX_MC_UNRESOLVED. The guard never binds in a refinement that meets
// its criteria: a facet with an edge above the bound e or a ball center
// farther than d from its circumcenter has a ball radius above e / 2 or d,
// a small-angle facet one above its shortest edge, and a cell above its
// size bound or B times its shortest edge (radius-edge bound B), so
// insertion radii stay above min(separation, e / 2, d, size) (Chew;
// Boissonnat and Oudot). It binds where a facet keeps a vertex off the zero
// set, or binary64 circumcenters lose their meaning.
class RestrictedMesher {
 public:
  RestrictedMesher(Field& field, const double* lower, const double* upper, const Criteria& criteria,
                   double separation)
      : field_(field), criteria_(criteria), separation2_(separation * separation),
        triangulation_(nullptr, nullptr, 0) {
    double largest = 0.0;
    for (int d = 0; d < 3; ++d) largest = std::max(largest, upper[d] - lower[d]);
    for (int c = 0; c < kCorners; ++c)
      for (int d = 0; d < 3; ++d)
        points_.push_back((c >> d) & 1 ? upper[d] + largest : lower[d] - largest);
    for (std::size_t d = 0; d < 3; ++d) {
      far_lower_[d] = lower[d] - largest;
      far_upper_[d] = upper[d] + largest;
    }
    far_length_ = 8.0 * largest;
    triangulation_.complex().journal = &journal_;
  }
  RestrictedMesher(const RestrictedMesher&) = delete;
  RestrictedMesher& operator=(const RestrictedMesher&) = delete;

  Criteria& criteria() { return criteria_; }
  int64_t vertex_count() const { return static_cast<int64_t>(points_.size() / 3); }
  const double* point(int32_t v) const { return points_.data() + 3 * static_cast<std::size_t>(v); }

  int32_t start(std::span<const Point> seeds) {
    const std::size_t expected_verts = std::max<std::size_t>(256, 4 * (seeds.size() + kCorners));
    points_.reserve(3 * expected_verts);
    alive_.reserve(expected_verts);
    on_surface_.reserve(expected_verts);
    for (const Point& p : seeds) {
      points_.push_back(p[0]);
      points_.push_back(p[1]);
      points_.push_back(p[2]);
    }
    const auto count = static_cast<int32_t>(points_.size() / 3);
    alive_.assign(static_cast<std::size_t>(count), 1);
    on_surface_.assign(static_cast<std::size_t>(count), 1);
    std::fill_n(on_surface_.begin(), kCorners, 0);
    rebind();
    std::vector<int32_t> candidates(static_cast<std::size_t>(count - kCorners));
    std::iota(candidates.begin(), candidates.end(), kCorners);
    std::vector<int32_t> order(kCorners);
    std::iota(order.begin(), order.end(), 0);
    if (!candidates.empty()) {
      const std::vector<int32_t> sorted = brio_hilbert_order(points_.data(), 3, candidates);
      order.insert(order.end(), sorted.begin(), sorted.end());
    }
    const int32_t status = triangulation_.build(order, alive_);
    if (status != PHX_MC_OK) return status;
    journal_.clear();
    std::vector<int32_t> cells;
    cells.reserve(tets().size());
    for (std::size_t t = 0; t < tets().size(); ++t)
      if (complex().live(static_cast<int32_t>(t))) cells.push_back(static_cast<int32_t>(t));
    balls_.reserve(tets().size());
    update(cells);
    return PHX_MC_OK;
  }

  // Inserts points of the zero set, except those within the separation
  // bound of a vertex (already covered at that distance).
  int32_t insert_all(std::span<const Point> points) {
    for (const Point& p : points) {
      if (!in_far_box(p)) continue;
      const int32_t start = triangulation_.locate(p.data());
      if (start < 0 || is_ghost(tets()[static_cast<std::size_t>(start)])) continue;
      if (conflict_zone(start, p) < separation2_) continue;
      triangulation_.set_hint(start);
      const int32_t status = insert(p, true);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  // Inserts each point inside the solid, or the ball center of a restricted
  // facet its insertion would remove, unless the separation bound refuses
  // it.
  int32_t refine_at(std::span<const Point> points) {
    std::vector<double> coordinates;
    coordinates.reserve(3 * points.size());
    for (const Point& p : points) {
      coordinates.push_back(p[0]);
      coordinates.push_back(p[1]);
      coordinates.push_back(p[2]);
    }
    std::vector<double> values(points.size());
    field_.values(coordinates, values.data());
    for (std::size_t k = 0; k < points.size(); ++k) {
      const Point& p = points[k];
      if (!(values[k] < 0.0) || !in_far_box(p)) continue;
      const int32_t start = triangulation_.locate(p.data());
      if (start < 0 || is_ghost(tets()[static_cast<std::size_t>(start)])) continue;
      bool surface = false;
      double radius2 = 0.0;
      int32_t hint = start;
      const Point chosen = target(start, p, surface, radius2, hint);
      if (radius2 < separation2_ || !in_far_box(chosen)) continue;
      triangulation_.set_hint(hint);
      const int32_t status = insert(chosen, surface);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  // Every live cell's faces and shape against the (changed) criteria.
  void reexamine() {
    std::vector<int32_t> cells;
    cells.reserve(tets().size());
    for (std::size_t t = 0; t < tets().size(); ++t)
      if (complex().live(static_cast<int32_t>(t))) cells.push_back(static_cast<int32_t>(t));
    examine(cells);
  }

  // Refines until no facet or cell violates the criteria except those the
  // separation bound refused; PHX_MC_UNRESOLVED when an admitted point's
  // cavity breaks the bound.
  int32_t refine(int64_t& insertions) {
    for (;;) {
      Point chosen;
      bool surface = false;
      double radius2 = 0.0;
      int32_t hint = -1;
      // Largest circumball first, facet or cell: on a zero set that is a
      // sphere the seeds alone are cospherical, so every inside cell's
      // circumball is the sphere and each surface insertion would rebuild
      // all of them; the center insertions the large cells ask for first
      // keep the cavities local.
      const bool facet_first = !facets_.empty() &&
                               (cells_.empty() || facets_.top().key >= cells_.top().key);
      if (facet_first) {
        const FacetEntry entry = facets_.top();
        facets_.pop();
        if (!current(entry)) continue;
        if (entry.key < separation2_ || !in_far_box(entry.center)) {
          refused_.push_back(entry);
          continue;
        }
        chosen = entry.center;
        surface = true;
        hint = entry.cell;
      } else if (!cells_.empty()) {
        const CellEntry entry = cells_.top();
        cells_.pop();
        if (!complex().live(entry.cell) || data(entry.cell).generation != entry.generation) continue;
        hint = entry.cell;
        chosen = cell_target(entry.cell, surface, radius2, hint);
        if (radius2 < separation2_ || !in_far_box(chosen)) continue;
      } else {
        return PHX_MC_OK;
      }
      if (hint >= 0) triangulation_.set_hint(hint);
      const int32_t status = insert(chosen, surface);
      if (status != PHX_MC_OK) return status;
      ++insertions;
    }
  }

  // The refused facets still in the triangulation, each once.
  int64_t refused_facets() const {
    std::vector<std::array<int32_t, 2>> facets;
    for (const FacetEntry& entry : refused_)
      if (current(entry)) facets.push_back({entry.cell, entry.slot});
    std::sort(facets.begin(), facets.end());
    return std::unique(facets.begin(), facets.end()) - facets.begin();
  }

  // The inside cells and their outward boundary faces, with the inside cell
  // and slot of each face.
  void surface(std::vector<int32_t>& cells, std::vector<int32_t>& faces,
               std::vector<std::array<int32_t, 2>>& facets) const {
    cells.clear();
    faces.clear();
    facets.clear();
    const std::size_t n_tets = tets().size();
    cells.reserve(2 * n_tets);
    faces.reserve(n_tets);
    facets.reserve(n_tets / 3 + 16);
    const std::uint8_t* flags = flags_.data();
    for (std::size_t t = 0; t < n_tets; ++t) {
      const auto id = static_cast<int32_t>(t);
      if ((flags[t] & kFlagInside) == 0 || !complex().live(id)) continue;
      const Tetrahedron& cell = tets()[t];
      cells.push_back(cell.v[0]);
      cells.push_back(cell.v[1]);
      cells.push_back(cell.v[2]);
      cells.push_back(cell.v[3]);
      for (int k = 0; k < 4; ++k) {
        if ((flags[static_cast<std::size_t>(cell.n[k])] & kFlagInside) != 0) continue;
        faces.push_back(cell.v[kOutwardFace[k][0]]);
        faces.push_back(cell.v[kOutwardFace[k][1]]);
        faces.push_back(cell.v[kOutwardFace[k][2]]);
        facets.push_back({id, k});
      }
    }
  }

  // Ball centers of the given restricted facets (inside cell, slot).
  void ball_centers(std::span<const std::array<int32_t, 2>> facets, std::vector<Point>& centers) {
    centers.clear();
    if (facets.empty()) return;
    centers.resize(facets.size());
    std::vector<std::size_t> missing;
    brackets_.clear();
    for (std::size_t k = 0; k < facets.size(); ++k) {
      const int32_t bid = data(facets[k][0]).ball[static_cast<std::size_t>(facets[k][1])];
      if (bid >= 0) {
        centers[k] = balls_[static_cast<std::size_t>(bid)].center;
      } else {
        missing.push_back(k);
        brackets_.push_back(dual_bracket(facets[k][0], facets[k][1]));
      }
    }
    if (missing.empty()) return;
    solve_rays();
    solver_.solve(field_, brackets_);
    for (std::size_t m = 0; m < missing.size(); ++m) centers[missing[m]] = brackets_[m].root;
  }

 private:
  static constexpr std::uint8_t kFlagInside = 1U;
  static constexpr std::uint8_t kFlagCenterSafe = 2U;
  std::vector<Tetrahedron>& tets() { return complex().tets; }
  const std::vector<Tetrahedron>& tets() const { return triangulation_.complex().tets; }
  TetrahedralComplex& complex() { return triangulation_.complex(); }
  const TetrahedralComplex& complex() const { return triangulation_.complex(); }
  const CellData& data(int32_t t) const { return data_[static_cast<std::size_t>(t)]; }
  bool is_inside(int32_t t) const noexcept {
    return (flags_[static_cast<std::size_t>(t)] & kFlagInside) != 0;
  }

  void rebind() {
    const int64_t count = vertex_count();
    triangulation_.rebind(points_.data(), count);
    triangulation_.reserve_vertices(count);
  }

  // Inserts p, a point of the zero set when `surface`; PHX_MC_UNRESOLVED
  // when its cavity, which holds its nearest vertex, shows a vertex within
  // the separation bound.
  int32_t insert(const Point& p, bool surface) {
    points_.push_back(p[0]);
    points_.push_back(p[1]);
    points_.push_back(p[2]);
    alive_.push_back(1);
    on_surface_.push_back(surface ? 1 : 0);
    rebind();
    const auto id = static_cast<int32_t>(vertex_count() - 1);
    journal_.clear();
    const int32_t status = triangulation_.extend(std::span<const int32_t>(&id, 1), alive_);
    if (status != PHX_MC_OK) return status;
    created_.clear();
    double nearest = std::numeric_limits<double>::infinity();
    for (const CellEvent& event : journal_) {
      if (!event.created) continue;
      created_.push_back(event.slot);
      for (int32_t v : event.v) {
        if (v == id || v == kGhostVertex) continue;
        const double* q = point(v);
        const double dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
        nearest = std::min(nearest, dx * dx + dy * dy + dz * dz);
      }
    }
    if (nearest < separation2_) return PHX_MC_UNRESOLVED;
    update(created_);
    return PHX_MC_OK;
  }

  // Circumcenters, values and labels of new cells, then their faces and shapes.
  void update(std::span<const int32_t> cells) {
    const std::size_t n_tets = tets().size();
    data_.resize(n_tets);
    spheres_.resize(n_tets);
    flags_.resize(n_tets, 0);
    mark_.resize(n_tets, 0);
    centers_.clear();
    finite_.clear();
    const auto d2 = [](const double* x, const double* y) noexcept {
      const double dx = x[0] - y[0], dy = x[1] - y[1], dz = x[2] - y[2];
      return dx * dx + dy * dy + dz * dz;
    };
    for (int32_t t : cells) {
      const auto ti = static_cast<std::size_t>(t);
      CellData& cell = data_[ti];
      SphereData& sph = spheres_[ti];
      cell.generation = ++generation_;
      flags_[ti] = 0;
      cell.ball = {-1, -1, -1, -1};
      const Tetrahedron& tet = tets()[ti];
      if (is_ghost(tet)) continue;
      const double* p0 = point(tet.v[0]);
      const double* p1 = point(tet.v[1]);
      const double* p2 = point(tet.v[2]);
      const double* p3 = point(tet.v[3]);
      const Point a = minus(p1, p0), b = minus(p2, p0), c = minus(p3, p0);
      const Point bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
      const double det = dot(a, bc);
      const double scale = 0.5 / det;
      const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
      cell.shortest2 = std::min({aa, bb, cc, d2(p1, p2), d2(p1, p3), d2(p2, p3)});
      const Point offset{scale * (aa * bc[0] + bb * ca[0] + cc * ab[0]),
                         scale * (aa * bc[1] + bb * ca[1] + cc * ab[1]),
                         scale * (aa * bc[2] + bb * ca[2] + cc * ab[2])};
      sph.radius2 = dot(offset, offset);
      sph.center = {p0[0] + offset[0], p0[1] + offset[1], p0[2] + offset[2]};
      if (std::isfinite(sph.radius2) && sph.radius2 > 0.0 && det > 0.0 &&
          (det * det) > 1e-10 * (aa * bb * cc)) {
        flags_[ti] = kFlagCenterSafe;
      }
      centers_.push_back(sph.center[0]);
      centers_.push_back(sph.center[1]);
      centers_.push_back(sph.center[2]);
      finite_.push_back(t);
    }
    values_.resize(finite_.size());
    field_.values(centers_, values_.data());
    for (std::size_t k = 0; k < finite_.size(); ++k) {
      const auto ti = static_cast<std::size_t>(finite_[k]);
      data_[ti].value = values_[k];
      if (values_[k] < 0.0) flags_[ti] |= kFlagInside;
    }
    examine(cells);
  }

  void store_ball(int32_t t, int slot, const Point& center, double radius2) {
    int32_t& bid = data_[static_cast<std::size_t>(t)].ball[static_cast<std::size_t>(slot)];
    if (bid >= 0) {
      balls_[static_cast<std::size_t>(bid)] = {center, radius2};
      return;
    }
    bid = static_cast<int32_t>(balls_.size());
    balls_.push_back({center, radius2});
  }

  // Queues the restricted facets and inside cells of `cells` that violate
  // the criteria, and caches each restricted facet's surface Delaunay ball on
  // its inside cell for target().
  void examine(std::span<const int32_t> cells) {
    ++batch_;
    std::uint32_t* __restrict mark = mark_.data();
    const std::uint8_t* __restrict flags = flags_.data();
    const Tetrahedron* __restrict tet_data = tets().data();
    for (int32_t t : cells) mark[static_cast<std::size_t>(t)] = batch_;
    pending_.clear();
    for (int32_t t : cells) {
      const Tetrahedron& tet = tet_data[static_cast<std::size_t>(t)];
      const bool inside = (flags[static_cast<std::size_t>(t)] & kFlagInside) != 0;
      if (inside) check_cell(t);
      for (int k = 0; k < 4; ++k) {
        const int32_t n = tet.n[k];
        if (((flags[static_cast<std::size_t>(n)] & kFlagInside) != 0) == inside) continue;
        if (inside) {
          pending_.push_back({t, k});
        } else if (mark[static_cast<std::size_t>(n)] != batch_) {
          pending_.push_back({n, neighbor_slot(tet_data[static_cast<std::size_t>(n)], t)});
        }
      }
    }
    if (pending_.empty()) return;
    brackets_.clear();
    for (const auto& facet : pending_) brackets_.push_back(dual_bracket(facet[0], facet[1]));
    solve_rays();
    solver_.solve(field_, brackets_);
    for (std::size_t k = 0; k < pending_.size(); ++k) {
      if (!(brackets_[k].f_positive >= 0.0)) {
        data_[static_cast<std::size_t>(pending_[k][0])].ball[static_cast<std::size_t>(pending_[k][1])] = -1;
        continue;
      }
      check_facet(pending_[k][0], pending_[k][1], brackets_[k].root);
    }
  }

  // The dual Voronoi edge of face `slot` of inside cell t: to the outside
  // neighbor's circumcenter, or along the outward normal past the far box
  // toward a ghost (its far value is filled by solve_rays).
  Bracket dual_bracket(int32_t t, int slot) {
    const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
    const CellData& inner = data(t);
    const SphereData& inner_sph = spheres_[static_cast<std::size_t>(t)];
    const int32_t n = tet.n[slot];
    Bracket bracket;
    bracket.negative = inner_sph.center;
    bracket.f_negative = inner.value;
    if (!is_ghost(tets()[static_cast<std::size_t>(n)])) {
      bracket.positive = spheres_[static_cast<std::size_t>(n)].center;
      bracket.f_positive = data(n).value;
      return bracket;
    }
    const int* face = kOutwardFace[slot];
    const double* p0 = point(tet.v[face[0]]);
    const Point normal = cross(minus(point(tet.v[face[1]]), p0), minus(point(tet.v[face[2]]), p0));
    const double length = std::sqrt(dot(normal, normal));
    for (std::size_t d = 0; d < 3; ++d)
      bracket.positive[d] = inner_sph.center[d] + far_length_ * normal[d] / length;
    bracket.f_positive = std::numeric_limits<double>::quiet_NaN();
    return bracket;
  }

  void solve_rays() {
    rays_.clear();
    ray_points_.clear();
    for (std::size_t k = 0; k < brackets_.size(); ++k) {
      if (!std::isnan(brackets_[k].f_positive)) continue;
      rays_.push_back(k);
      ray_points_.insert(ray_points_.end(), brackets_[k].positive.begin(), brackets_[k].positive.end());
    }
    if (rays_.empty()) return;
    values_.resize(rays_.size());
    field_.values(ray_points_, values_.data());
    for (std::size_t r = 0; r < rays_.size(); ++r) brackets_[rays_[r]].f_positive = values_[r];
    // A ray that does not leave the solid has no crossing: its bracket is
    // degenerate at the inside circumcenter and its facet is left alone.
    for (std::size_t r = 0; r < rays_.size(); ++r) {
      Bracket& bracket = brackets_[rays_[r]];
      if (bracket.f_positive < 0.0) {
        bracket.positive = bracket.negative;
        bracket.f_positive = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }

  void check_cell(int32_t t) {
    const CellData& cell = data(t);
    const double r2 = spheres_[static_cast<std::size_t>(t)].radius2;
    if (r2 > criteria_.size2 || r2 > criteria_.ratio2 * cell.shortest2)
      cells_.push({r2, t, cell.generation});
  }

  void check_facet(int32_t t, int slot, const Point& center) {
    const Tetrahedron& tet = tets()[static_cast<std::size_t>(t)];
    const int32_t v0 = tet.v[kOutwardFace[slot][0]], v1 = tet.v[kOutwardFace[slot][1]],
                  v2 = tet.v[kOutwardFace[slot][2]];
    const double* p = point(v0);
    const Point ball = minus(center.data(), p);
    const double key = dot(ball, ball);
    store_ball(t, slot, center, key);
    // A restricted facet must join points of the zero set.
    bool bad = !on_surface_[static_cast<std::size_t>(v0)] ||
               !on_surface_[static_cast<std::size_t>(v1)] ||
               !on_surface_[static_cast<std::size_t>(v2)];
    if (!bad) {
      const double* p1 = point(v1);
      const double* p2 = point(v2);
      const Point a = minus(p1, p), b = minus(p2, p), c = minus(p2, p1);
      const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c);
      if (std::max({aa, bb, cc}) > criteria_.edge2) {
        bad = true;
      } else {
        const Point normal = cross(a, b);
        const double nn = dot(normal, normal);
        if (!(nn > 0.0)) {
          bad = true;
        } else {
          const Point ra = cross(b, normal), rb = cross(normal, a);
          const double inv_2nn = 0.5 / nn;
          const Point offset{(aa * ra[0] + bb * rb[0]) * inv_2nn,
                             (aa * ra[1] + bb * rb[1]) * inv_2nn,
                             (aa * ra[2] + bb * rb[2]) * inv_2nn};
          const double rho2 = dot(offset, offset);
          const Point gap{ball[0] - offset[0], ball[1] - offset[1], ball[2] - offset[2]};
          // The smallest angle is opposite the shortest edge: sin = shortest / (2 rho).
          bad = dot(gap, gap) > criteria_.distance2 ||
                std::min({aa, bb, cc}) < 4.0 * criteria_.angle_sine2 * rho2;
        }
      }
    }
    if (!bad) return;
    const int32_t n = tet.n[slot];
    facets_.push({key, t, slot, n, data(t).generation, data(n).generation, center});
  }

  bool current(const FacetEntry& entry) const {
    return complex().live(entry.cell) && data(entry.cell).generation == entry.cell_generation &&
           tets()[static_cast<std::size_t>(entry.cell)].n[entry.slot] == entry.other &&
           data(entry.other).generation == entry.other_generation;
  }

  // The circumcenter of inside cell t, or the ball center of a restricted
  // facet that the circumcenter's conflict region would remove.
  Point cell_target(int32_t t, bool& surface, double& radius2, int32_t& hint) {
    return target(t, spheres_[static_cast<std::size_t>(t)].center, surface, radius2, hint);
  }

  bool in_far_box(const Point& p) const {
    for (std::size_t d = 0; d < 3; ++d)
      if (!(p[d] > far_lower_[d] && p[d] < far_upper_[d])) return false;
    return true;
  }

  // Grows zone_, the cells whose circumspheres hold p, from `start` (one of
  // them); returns the squared distance from p to the zone's nearest vertex,
  // its nearest vertex in the Delaunay triangulation. Uses two consecutive
  // batch marks (tested < in_zone) so non-conflicting neighbors are tested once,
  // and pre-filters well-conditioned cells with their cached circumsphere
  // before falling back to exact insphere_filtered.
  double conflict_zone(int32_t start, const Point& p) {
    const std::uint32_t tested = ++batch_;
    const std::uint32_t in_zone = ++batch_;
    zone_.assign(1, start);
    mark_[static_cast<std::size_t>(start)] = in_zone;
    double nearest = std::numeric_limits<double>::infinity();
    const Tetrahedron* __restrict tet_data = tets().data();
    const SphereData* __restrict sphere_data = spheres_.data();
    const std::uint8_t* __restrict flags = flags_.data();
    std::uint32_t* __restrict mark = mark_.data();
    for (std::size_t k = 0; k < zone_.size(); ++k) {
      const Tetrahedron& cell = tet_data[static_cast<std::size_t>(zone_[k])];
      for (int32_t v : cell.v) {
        const double* q = point(v);
        const double dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
        nearest = std::min(nearest, dx * dx + dy * dy + dz * dz);
      }
      for (int32_t n : cell.n) {
        const auto ni = static_cast<std::size_t>(n);
        std::uint32_t& m = mark[ni];
        if (m >= tested) continue;
        m = tested;
        const Tetrahedron& next = tet_data[ni];
        if (is_ghost(next)) continue;
        bool in_sphere = false;
        if ((flags[ni] & kFlagCenterSafe) != 0) {
          const SphereData& nd = sphere_data[ni];
          const double dx = p[0] - nd.center[0], dy = p[1] - nd.center[1], dz = p[2] - nd.center[2];
          const double d2 = dx * dx + dy * dy + dz * dz;
          const double margin = 1e-5 * nd.radius2;
          if (d2 > nd.radius2 + margin) continue;
          if (d2 < nd.radius2 - margin) {
            in_sphere = true;
          } else {
            in_sphere = insphere_filtered(point(next.v[0]), point(next.v[1]), point(next.v[2]),
                                          point(next.v[3]), p.data()) > 0;
          }
        } else {
          in_sphere = insphere_filtered(point(next.v[0]), point(next.v[1]), point(next.v[2]),
                                        point(next.v[3]), p.data()) > 0;
        }
        if (!in_sphere) continue;
        m = in_zone;
        zone_.push_back(n);
      }
    }
    return nearest;
  }

  // `center`, or the ball center of a restricted facet whose surface
  // Delaunay ball holds it (the encroachment rule of Rineau and Yvinec),
  // with the squared distance to the chosen point's nearest vertex. A
  // facet's surface ball lies in the union of its two cells' circumballs
  // (the ends of the pencil of spheres through its vertices), so only the
  // restricted facets of the conflict region of `center` (grown from
  // `start`, a cell in conflict), removed or on its boundary, can hold it.
  // An interior point inside a surface ball would lie next to the zero
  // set, a vertex of restricted facets off it.
  Point target(int32_t start, Point center, bool& surface, double& radius2, int32_t& hint) {
    surface = false;
    hint = start;
    // `start` is always zone_[0]; when it is an inside cell, its own restricted
    // facets are checked first, so an encroached face returns without growing
    // the conflict zone at all.
    if (is_inside(start)) {
      const Tetrahedron& cell = tets()[static_cast<std::size_t>(start)];
      for (int k = 0; k < 4; ++k) {
        if (is_inside(cell.n[k])) continue;
        const int32_t bid = data(start).ball[static_cast<std::size_t>(k)];
        if (bid < 0) continue;
        const BallData& b = balls_[static_cast<std::size_t>(bid)];
        const Point offset = minus(center.data(), b.center.data());
        if (!(dot(offset, offset) < b.radius2)) continue;
        radius2 = b.radius2;
        surface = true;
        hint = start;
        return b.center;
      }
    }
    radius2 = conflict_zone(start, center);
    const std::uint32_t in_zone = batch_;
    const std::size_t first_idx = is_inside(start) ? 1 : 0;
    const std::uint8_t* __restrict flags = flags_.data();
    const Tetrahedron* __restrict tet_data = tets().data();
    for (std::size_t idx = first_idx; idx < zone_.size(); ++idx) {
      const int32_t x = zone_[idx];
      const bool inside = (flags[static_cast<std::size_t>(x)] & kFlagInside) != 0;
      const Tetrahedron& cell = tet_data[static_cast<std::size_t>(x)];
      for (int k = 0; k < 4; ++k) {
        const int32_t n = cell.n[k];
        if (((flags[static_cast<std::size_t>(n)] & kFlagInside) != 0) == inside) continue;
        int32_t in_cell = x, in_slot = k;
        if (!inside) {
          if (mark_[static_cast<std::size_t>(n)] == in_zone) continue;
          in_cell = n;
          in_slot = neighbor_slot(tet_data[static_cast<std::size_t>(n)], x);
        }
        const int32_t bid = data(in_cell).ball[static_cast<std::size_t>(in_slot)];
        if (bid < 0) continue;
        const BallData& b = balls_[static_cast<std::size_t>(bid)];
        const Point offset = minus(center.data(), b.center.data());
        if (!(dot(offset, offset) < b.radius2)) continue;
        radius2 = b.radius2;
        surface = true;
        hint = in_cell;
        return b.center;
      }
    }
    return center;
  }

  Field& field_;
  Criteria criteria_;
  double separation2_;
  std::vector<double> points_;
  std::vector<char> alive_, on_surface_;
  Triangulation3D triangulation_;
  std::vector<CellEvent> journal_;
  std::vector<CellData> data_;
  std::vector<SphereData> spheres_;
  std::vector<std::uint8_t> flags_;
  std::vector<BallData> balls_;
  std::vector<std::uint32_t> mark_;
  std::uint32_t batch_ = 0;
  std::uint32_t generation_ = 0;
  Point far_lower_{}, far_upper_{};
  double far_length_ = 0.0;
  std::priority_queue<FacetEntry> facets_;
  std::priority_queue<CellEntry> cells_;
  std::vector<FacetEntry> refused_;
  std::vector<int32_t> created_, finite_, zone_;
  std::vector<std::size_t> rays_;
  std::vector<double> centers_, values_, ray_points_;
  std::vector<std::array<int32_t, 2>> pending_;
  std::vector<Bracket> brackets_;
  RootSolver solver_;
};

// Moves points onto the zero set by `steps` Newton steps along the gradient.
void project(Field& field, std::vector<Point>& points, int steps) {
  field.project_points(points, steps);
}

// Per-mesher-vertex cache of Newton-projected zero-set positions and scratch
// buffers reused across quality rounds.
struct PolishCache {
  std::vector<char> valid;
  std::vector<Point> position;
  std::vector<char> on_boundary;
  std::vector<int32_t> boundary, uncached_rows;
  std::vector<Point> uncached_points, saved;
};

// Moves the boundary vertices onto the zero set to float accuracy, keeping
// the construction position of any vertex of a cell the move would leave
// without positive orientation.
void polish_boundary(Field& field, std::vector<double>& points, std::span<const int32_t> cells,
                     std::span<const int32_t> faces, std::span<const int32_t> mesher_ids,
                     PolishCache& cache) {
  const std::size_t vertex_count = points.size() / 3;
  cache.on_boundary.assign(vertex_count, 0);
  for (int32_t v : faces) cache.on_boundary[static_cast<std::size_t>(v)] = 1;
  cache.boundary.clear();
  cache.uncached_rows.clear();
  cache.uncached_points.clear();
  for (std::size_t v = 0; v < vertex_count; ++v) {
    if (!cache.on_boundary[v]) continue;
    const auto vi = static_cast<int32_t>(v);
    cache.boundary.push_back(vi);
    const auto mv = static_cast<std::size_t>(mesher_ids[v]);
    if (mv < cache.valid.size() && cache.valid[mv]) continue;
    const double* p = points.data() + 3 * v;
    cache.uncached_rows.push_back(vi);
    cache.uncached_points.push_back({p[0], p[1], p[2]});
  }
  if (!cache.uncached_points.empty()) {
    project(field, cache.uncached_points, kPolishSteps);
    for (std::size_t k = 0; k < cache.uncached_rows.size(); ++k) {
      const auto mv = static_cast<std::size_t>(
          mesher_ids[static_cast<std::size_t>(cache.uncached_rows[k])]);
      if (mv >= cache.valid.size()) {
        cache.valid.resize(mv + 1, 0);
        cache.position.resize(mv + 1);
      }
      cache.valid[mv] = 1;
      cache.position[mv] = cache.uncached_points[k];
    }
  }
  cache.saved.resize(vertex_count);
  for (int32_t v : cache.boundary) {
    const auto row = static_cast<std::size_t>(v);
    double* dst = points.data() + 3 * row;
    cache.saved[row] = {dst[0], dst[1], dst[2]};
    const Point& moved = cache.position[static_cast<std::size_t>(mesher_ids[row])];
    dst[0] = moved[0];
    dst[1] = moved[1];
    dst[2] = moved[2];
  }
  const std::size_t cell_count = cells.size() / 4;
  const char* on_boundary = cache.on_boundary.data();
  for (bool reverted = true; reverted;) {
    reverted = false;
    for (std::size_t c = 0; c < cell_count; ++c) {
      const int32_t* v = cells.data() + 4 * c;
      if (!(on_boundary[v[0]] | on_boundary[v[1]] | on_boundary[v[2]] | on_boundary[v[3]]))
        continue;
      const auto at = [&](int k) { return points.data() + 3 * static_cast<std::size_t>(v[k]); };
      if (orient3d_filtered(at(0), at(1), at(2), at(3)) > 0) continue;
      for (int k = 0; k < 4; ++k) {
        const auto row = static_cast<std::size_t>(v[k]);
        if (!on_boundary[row]) continue;
        double* dst = points.data() + 3 * row;
        const Point& orig = cache.saved[row];
        if (dst[0] != orig[0] || dst[1] != orig[1] || dst[2] != orig[2]) {
          dst[0] = orig[0];
          dst[1] = orig[1];
          dst[2] = orig[2];
          reverted = true;
        }
      }
    }
  }
}

// For a positive tetrahedron (v0, v1, v2, v3), kEdgeOpposite[ia][ib] = {ic, id}
// makes (ia, ib, ic, id) an even permutation of (0, 1, 2, 3), so
// orient3d(v[ia], v[ib], v[ic], v[id]) > 0 without evaluating orient3d.
constexpr int kEdgeOpposite[4][4][2] = {
    {{-1, -1}, {2, 3}, {3, 1}, {1, 2}},
    {{3, 2}, {-1, -1}, {0, 3}, {2, 0}},
    {{1, 3}, {3, 0}, {-1, -1}, {0, 1}},
    {{2, 1}, {0, 2}, {1, 0}, {-1, -1}},
};

// Removes cells whose smallest dihedral angle is below a target from a mesh
// with a fixed boundary: edge removal (the ring around an interior edge
// retriangulated by the dynamic program of Klingner and Shewchuk, "Aggressive
// tetrahedral mesh improvement", 2007), face removal (2-3 flips) and
// relocation of interior vertices. Every change strictly raises the
// smallest angle among the cells it replaces and keeps them positive (exact
// orientation) within the radius-edge ratio and relative determinant.
class SliverRemover {
 public:
  SliverRemover(std::vector<double>& points, std::vector<int32_t>& cells,
                std::span<const int32_t> faces, double dihedral, double ratio, double determinant)
      : points_(points), target_(std::cos(dihedral * (std::numbers::pi / 180.0))), ratio_(ratio),
        determinant_(determinant), fixed_(points.size() / 3, 0), stars_(points.size() / 3) {
    for (int32_t v : faces) fixed_[static_cast<std::size_t>(v)] = 1;
    // Flat open-addressed set of boundary edges so remove_edge skips boundary
    // edges in O(1) without scanning vertex stars or walking open rings.
    const std::size_t face_count = faces.size() / 3;
    const std::size_t edge_cap = std::bit_ceil(std::max<std::size_t>(16, face_count * 4));
    boundary_edges_.assign(edge_cap, kEmptyEdge);
    boundary_mask_ = edge_cap - 1;
    for (std::size_t f = 0; f < face_count; ++f) {
      const int32_t v0 = faces[3 * f], v1 = faces[3 * f + 1], v2 = faces[3 * f + 2];
      insert_boundary_edge(v0, v1);
      insert_boundary_edge(v1, v2);
      insert_boundary_edge(v2, v0);
    }
    // One contiguous arena for all vertex stars, with 8 spare slots per vertex.
    const std::size_t vertex_count = stars_.size();
    std::vector<int32_t> degree(vertex_count, 0);
    for (int32_t v : cells) ++degree[static_cast<std::size_t>(v)];
    int32_t total = 0;
    for (std::size_t v = 0; v < vertex_count; ++v) {
      const int32_t cap = degree[v] + 8;
      stars_[v] = {total, 0, cap};
      total += cap;
    }
    star_pool_.resize(static_cast<std::size_t>(total));
    const std::size_t count = cells.size() / 4;
    cells_.reserve(2 * count);
    alive_.reserve(2 * count);
    cosine_.reserve(2 * count);
    safe_.reserve(2 * count);
    failed_epoch_.reserve(2 * count);
    vertex_epoch_.assign(vertex_count, 1);
    for (std::size_t c = 0; c < count; ++c)
      add({cells[4 * c], cells[4 * c + 1], cells[4 * c + 2], cells[4 * c + 3]});
  }

  // Runs up to `passes` sweeps over the cells below the target; returns the
  // remaining cells (live, in order) in `cells`, the circumcenters of those
  // still below it in `leftover`, their count, and sets `all_safe` true when
  // every surviving cell clears all three quality targets with a safe margin.
  int64_t run(int passes, std::vector<int32_t>& cells, std::vector<Point>& leftover,
              bool& all_safe) {
    for (int pass = 0; pass < passes; ++pass) {
      bool changed = false, bad = false;
      const std::size_t count = cells_.size();
      for (std::size_t c = 0; c < count; ++c) {
        if (!alive_[c] || cosine_[c] <= target_) continue;
        bad = true;
        const Cell& cell = cells_[c];
        const std::uint32_t cur_epoch = std::max({
            vertex_epoch_[static_cast<std::size_t>(cell[0])],
            vertex_epoch_[static_cast<std::size_t>(cell[1])],
            vertex_epoch_[static_cast<std::size_t>(cell[2])],
            vertex_epoch_[static_cast<std::size_t>(cell[3])]});
        if (failed_epoch_[c] == cur_epoch) continue;
        if (remove_edge(c) || remove_face(c) || relocate(c)) {
          changed = true;
        } else {
          failed_epoch_[c] = cur_epoch;
        }
      }
      if (!bad || !changed) break;
    }
    cells.clear();
    leftover.clear();
    all_safe = true;
    for (std::size_t c = 0; c < cells_.size(); ++c) {
      if (!alive_[c]) continue;
      const Cell& cell = cells_[c];
      cells.push_back(cell[0]);
      cells.push_back(cell[1]);
      cells.push_back(cell[2]);
      cells.push_back(cell[3]);
      if (!safe_[c]) all_safe = false;
      if (cosine_[c] <= target_) continue;
      const double* p0 = at(cell[0]);
      const Point a = minus(at(cell[1]), p0), b = minus(at(cell[2]), p0), d = minus(at(cell[3]), p0);
      const Point bd = cross(b, d), da = cross(d, a), ab = cross(a, b);
      const double scale = 0.5 / dot(a, bd);
      const double aa = dot(a, a), bb = dot(b, b), dd = dot(d, d);
      Point center{};
      for (std::size_t k = 0; k < 3; ++k) center[k] = p0[k] + scale * (aa * bd[k] + bb * da[k] + dd * ab[k]);
      if (std::isfinite(center[0]) && std::isfinite(center[1]) && std::isfinite(center[2]))
        leftover.push_back(center);
    }
    return static_cast<int64_t>(leftover.size());
  }

 private:
  using Cell = std::array<int32_t, 4>;
  struct StarView {
    int32_t offset = 0;
    int32_t size = 0;
    int32_t cap = 0;
  };
  const double* at(int32_t v) const { return points_.data() + 3 * static_cast<std::size_t>(v); }

  std::span<const int32_t> star_of(int32_t v) const {
    const StarView& s = stars_[static_cast<std::size_t>(v)];
    if (s.offset >= 0) return {star_pool_.data() + s.offset, static_cast<std::size_t>(s.size)};
    const auto& ov = overflow_[static_cast<std::size_t>(v)];
    return {ov.data(), ov.size()};
  }
  void star_push(int32_t v, int32_t id) {
    StarView& s = stars_[static_cast<std::size_t>(v)];
    if (s.offset >= 0) {
      if (s.size < s.cap) {
        star_pool_[static_cast<std::size_t>(s.offset + s.size++)] = id;
        return;
      }
      if (overflow_.empty()) overflow_.resize(stars_.size());
      auto& ov = overflow_[static_cast<std::size_t>(v)];
      ov.reserve(static_cast<std::size_t>(s.cap) * 2);
      ov.assign(star_pool_.data() + s.offset, star_pool_.data() + s.offset + s.size);
      s.offset = -1;
    }
    overflow_[static_cast<std::size_t>(v)].push_back(id);
  }
  void star_erase(int32_t v, int32_t id) {
    StarView& s = stars_[static_cast<std::size_t>(v)];
    if (s.offset >= 0) {
      int32_t* data = star_pool_.data() + s.offset;
      const int32_t n = s.size;
      for (int32_t i = 0; i < n; ++i) {
        if (data[i] != id) continue;
        for (int32_t j = i + 1; j < n; ++j) data[j - 1] = data[j];
        --s.size;
        return;
      }
      return;
    }
    auto& ov = overflow_[static_cast<std::size_t>(v)];
    ov.erase(std::find(ov.begin(), ov.end(), id));
  }

  // The largest cosine of the cell's interior dihedral angles; +inf unless
  // it is positive, and, when it meets the target, +inf unless its
  // radius-edge ratio is within ratio_ (0: any) and its relative determinant
  // at least determinant_. Sets `*safe` true when every bound clears with a
  // 1e-6 margin (guaranteeing meets_quality() agrees).
  double shape(const Cell& cell, bool* safe = nullptr) const {
    if (safe != nullptr) *safe = false;
    const double* p[4] = {at(cell[0]), at(cell[1]), at(cell[2]), at(cell[3])};
    if (orient3d_filtered(p[0], p[1], p[2], p[3]) <= 0)
      return std::numeric_limits<double>::infinity();
    const Point a = minus(p[1], p[0]);
    const Point b = minus(p[2], p[0]);
    const Point c = minus(p[3], p[0]);
    const Point e12 = minus(p[2], p[1]);
    const Point e13 = minus(p[3], p[1]);
    // Matches kOutwardFace = {{1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}} bit-for-bit.
    const Point normals[4] = {cross(e12, e13), cross(c, b), cross(a, c), cross(b, a)};
    const double n2[4] = {dot(normals[0], normals[0]), dot(normals[1], normals[1]),
                          dot(normals[2], normals[2]), dot(normals[3], normals[3])};
    const double scale[4] = {1.0 / std::sqrt(n2[0]), 1.0 / std::sqrt(n2[1]),
                             1.0 / std::sqrt(n2[2]), 1.0 / std::sqrt(n2[3])};
    double cosine = -1.0;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        cosine = std::max(cosine, -dot(normals[i], normals[j]) * scale[i] * scale[j]);
    if (cosine > target_) return cosine;
    const Point e23 = minus(p[3], p[2]);
    // normals[1..3] are cross(c,b), cross(a,c), cross(b,a), exact negations of bc, ca, ab.
    const Point bc{-normals[1][0], -normals[1][1], -normals[1][2]};
    const double volume = dot(a, bc);
    const double lengths[6] = {dot(a, a),     dot(b, b),     dot(c, c),
                               dot(e12, e12), dot(e13, e13), dot(e23, e23)};
    const double shortest =
        std::min({lengths[0], lengths[1], lengths[2], lengths[3], lengths[4], lengths[5]});
    double r2 = 0.0;
    if (ratio_ > 0.0) {
      const Point ca{-normals[2][0], -normals[2][1], -normals[2][2]};
      const Point ab{-normals[3][0], -normals[3][1], -normals[3][2]};
      const double inv_2vol = 0.5 / volume;
      const Point offset{(lengths[0] * bc[0] + lengths[1] * ca[0] + lengths[2] * ab[0]) * inv_2vol,
                         (lengths[0] * bc[1] + lengths[1] * ca[1] + lengths[2] * ab[1]) * inv_2vol,
                         (lengths[0] * bc[2] + lengths[1] * ca[2] + lengths[2] * ab[2]) * inv_2vol};
      r2 = dot(offset, offset);
      if (r2 > ratio_ * ratio_ * shortest) return std::numeric_limits<double>::infinity();
    }
    double largest = 0.0;
    if (determinant_ > 0.0 || safe != nullptr) {
      // Products of the three squared edge lengths at each corner (kEdges
      // order: 01 02 03 12 13 23).
      const double products[4] = {lengths[0] * lengths[1] * lengths[2],
                                  lengths[0] * lengths[3] * lengths[4],
                                  lengths[1] * lengths[3] * lengths[5],
                                  lengths[2] * lengths[4] * lengths[5]};
      largest = std::max({products[0], products[1], products[2], products[3]});
      if (determinant_ > 0.0 && volume * volume < determinant_ * determinant_ * largest)
        return std::numeric_limits<double>::infinity();
    }
    if (safe != nullptr) {
      const bool finite_normals = (n2[0] > 0.0) && std::isfinite(n2[0]) && (n2[1] > 0.0) &&
                                  std::isfinite(n2[1]) && (n2[2] > 0.0) && std::isfinite(n2[2]) &&
                                  (n2[3] > 0.0) && std::isfinite(n2[3]);
      if (finite_normals && ratio_ > 1e-6 && std::isfinite(cosine) && cosine <= target_ - 1e-6 &&
          std::isfinite(volume) && volume > 0.0 && std::isfinite(shortest) && shortest > 0.0 &&
          std::isfinite(r2) && r2 > 0.0 && std::isfinite(largest) && largest > 0.0) {
        const double safe_ratio = ratio_ - 1e-6;
        const double safe_det = determinant_ + 1e-6;
        if (r2 <= safe_ratio * safe_ratio * shortest &&
            volume * volume >= safe_det * safe_det * largest) {
          *safe = true;
        }
      }
    }
    return cosine;
  }

  void add_known(const Cell& cell, double cos_val, bool is_safe, bool bump = true) {
    const auto id = static_cast<int32_t>(cells_.size());
    cells_.push_back(cell);
    alive_.push_back(1);
    cosine_.push_back(cos_val);
    safe_.push_back(is_safe ? 1 : 0);
    failed_epoch_.push_back(0);
    if (bump) {
      ++epoch_;
      for (int32_t v : cell) vertex_epoch_[static_cast<std::size_t>(v)] = epoch_;
    }
    for (int32_t v : cell) star_push(v, id);
  }
  void add(const Cell& cell) {
    bool is_safe = false;
    const double cos_val = shape(cell, &is_safe);
    add_known(cell, cos_val, is_safe, false);
  }
  void kill(std::size_t c) {
    alive_[c] = 0;
    const auto cid = static_cast<int32_t>(c);
    ++epoch_;
    for (int32_t v : cells_[c]) {
      vertex_epoch_[static_cast<std::size_t>(v)] = epoch_;
      star_erase(v, cid);
    }
  }
  static bool has(const Cell& cell, int32_t v) {
    return cell[0] == v || cell[1] == v || cell[2] == v || cell[3] == v;
  }

  // The ring of cells around edge (a, b) and its vertices, ordered so that
  // (a, b, r[i], r[i + 1]) is positive; false unless the ring is closed.
  struct Ring {
    std::array<int32_t, 7> cells, vertices;
    std::size_t size = 0;
  };
  bool ring(int32_t a, int32_t b, Ring& result) const {
    std::array<int32_t, 7> around;
    std::size_t n = 0;
    for (int32_t c : star_of(a)) {
      if (!has(cells_[static_cast<std::size_t>(c)], b)) continue;
      if (n == around.size()) return false;
      around[n++] = c;
    }
    if (n < 3) return false;
    std::array<std::array<int32_t, 2>, 7> links{};
    for (std::size_t k = 0; k < n; ++k) {
      const Cell& cell = cells_[static_cast<std::size_t>(around[k])];
      int ia = 0, ib = 0;
      for (int s = 0; s < 4; ++s) {
        if (cell[static_cast<std::size_t>(s)] == a) ia = s;
        if (cell[static_cast<std::size_t>(s)] == b) ib = s;
      }
      links[k] = {cell[static_cast<std::size_t>(kEdgeOpposite[ia][ib][0])],
                  cell[static_cast<std::size_t>(kEdgeOpposite[ia][ib][1])]};
    }
    result.vertices[0] = links[0][0];
    result.cells[0] = around[0];
    // Follow the directed link edges from the first one.
    std::array<char, 7> used{};
    used[0] = 1;
    int32_t head = links[0][1];
    for (std::size_t step = 1; step < n; ++step) {
      std::size_t next = n;
      for (std::size_t k = 0; k < n; ++k)
        if (!used[k] && links[k][0] == head) next = k;
      if (next == n) return false;
      used[next] = 1;
      result.vertices[step] = head;
      result.cells[step] = around[next];
      head = links[next][1];
    }
    if (head != result.vertices[0]) return false;
    result.size = n;
    return true;
  }

  double worst(std::span<const int32_t> cells) const {
    double value = -1.0;
    for (int32_t c : cells) value = std::max(value, cosine_[static_cast<std::size_t>(c)]);
    return value;
  }

  // Retriangulates the ring of the best of c's interior edges.
  bool remove_edge(std::size_t c) {
    const Cell cell = cells_[c];
    Ring around;
    for (const auto& e : kEdges) {
      const int32_t a = cell[static_cast<std::size_t>(e[0])], b = cell[static_cast<std::size_t>(e[1])];
      if (fixed_[static_cast<std::size_t>(a)] && fixed_[static_cast<std::size_t>(b)] &&
          is_boundary_edge(a, b))
        continue;
      if (!ring(a, b, around)) continue;
      const std::span<const int32_t> vertices(around.vertices.data(), around.size);
      const double before = worst(std::span<const int32_t>(around.cells.data(), around.size));
      const int n = static_cast<int>(around.size);
      // best[i][j]: the smallest largest cosine over triangulations of the
      // polygon vertices i..j, capped at `before` since only strict improvement
      // commits; split[i][j] its apex.
      double best[7][7];
      double cos_a[7][7], cos_b[7][7];
      bool safe_a[7][7], safe_b[7][7];
      int split[7][7];
      for (int i = 0; i + 1 < n; ++i) best[i][i + 1] = -1.0;
      for (int length = 2; length < n; ++length)
        for (int i = 0; i + length < n; ++i) {
          const int j = i + length;
          double limit = before;
          best[i][j] = before;
          for (int k = i + 1; k < j; ++k) {
            const double sides = std::max(best[i][k], best[k][j]);
            if (sides >= limit) continue;
            const int32_t ri = vertices[static_cast<std::size_t>(i)], rk = vertices[static_cast<std::size_t>(k)],
                          rj = vertices[static_cast<std::size_t>(j)];
            bool sa_safe = false;
            const double sa = shape({a, ri, rk, rj}, &sa_safe);
            if (sa >= limit) continue;
            bool sb_safe = false;
            const double sb = shape({b, rj, rk, ri}, &sb_safe);
            const double value = std::max({sides, sa, sb});
            if (value < limit) {
              limit = value;
              best[i][j] = value;
              split[i][j] = k;
              cos_a[i][j] = sa;
              cos_b[i][j] = sb;
              safe_a[i][j] = sa_safe;
              safe_b[i][j] = sb_safe;
            }
          }
        }
      if (!(best[0][n - 1] < before)) continue;
      for (std::size_t k = 0; k < around.size; ++k) kill(static_cast<std::size_t>(around.cells[k]));
      // Depth-first over the split tree, at most n intervals pending.
      std::array<std::array<int, 2>, 8> stack{};
      std::size_t pending = 0;
      stack[pending++] = {0, n - 1};
      while (pending > 0) {
        const auto [i, j] = stack[--pending];
        if (j - i < 2) continue;
        const int k = split[i][j];
        const int32_t ri = vertices[static_cast<std::size_t>(i)], rk = vertices[static_cast<std::size_t>(k)],
                      rj = vertices[static_cast<std::size_t>(j)];
        add_known({a, ri, rk, rj}, cos_a[i][j], safe_a[i][j]);
        add_known({b, rj, rk, ri}, cos_b[i][j], safe_b[i][j]);
        stack[pending++] = {i, k};
        stack[pending++] = {k, j};
      }
      return true;
    }
    return false;
  }

  // Replaces c and a face neighbor by three cells around their apexes' edge.
  bool remove_face(std::size_t c) {
    const Cell cell = cells_[c];
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t x = cell[static_cast<std::size_t>(kOutwardFace[slot][0])],
                    y = cell[static_cast<std::size_t>(kOutwardFace[slot][1])],
                    z = cell[static_cast<std::size_t>(kOutwardFace[slot][2])];
      const int32_t a = cell[static_cast<std::size_t>(slot)];
      int32_t other = -1;
      for (int32_t d : star_of(x))
        if (static_cast<std::size_t>(d) != c && has(cells_[static_cast<std::size_t>(d)], y) &&
            has(cells_[static_cast<std::size_t>(d)], z))
          other = d;
      if (other < 0) continue;
      int32_t b = -1;
      for (int32_t v : cells_[static_cast<std::size_t>(other)])
        if (v != x && v != y && v != z) b = v;
      const Cell created[3] = {{a, b, x, y}, {a, b, y, z}, {a, b, z, x}};
      const double before = std::max(cosine_[c], cosine_[static_cast<std::size_t>(other)]);
      double after = -1.0;
      double cos_new[3];
      bool safe_new[3];
      for (int r = 0; r < 3; ++r) {
        cos_new[r] = shape(created[r], &safe_new[r]);
        after = std::max(after, cos_new[r]);
        if (!(after < before)) break;
      }
      if (!(after < before)) continue;
      kill(c);
      kill(static_cast<std::size_t>(other));
      for (int r = 0; r < 3; ++r) add_known(created[r], cos_new[r], safe_new[r]);
      return true;
    }
    return false;
  }

  // Moves an interior vertex of c to raise the smallest angle of its star:
  // repeatedly away from the face opposite it in the star's worst cell
  // (which raises that cell's height), else toward the star's centroid.
  bool relocate(std::size_t c) {
    constexpr int kAscents = 6;
    constexpr double kSteps[] = {0.3, 0.15, 0.075};
    const Cell cell = cells_[c];
    bool moved = false;
    for (int k = 0; k < 4; ++k) {
      const int32_t v = cell[static_cast<std::size_t>(k)];
      if (fixed_[static_cast<std::size_t>(v)]) continue;
      const std::span<const int32_t> star = star_of(v);
      double* p = points_.data() + 3 * static_cast<std::size_t>(v);
      double best = worst(star);
      trial_cos_.resize(star.size());
      trial_safe_.resize(star.size());
      // The worst cosine over the star with v at q, abandoned at `best`.
      // Tests `blocker_idx` (the cell that rejected the previous candidate
      // along the same ascent ray) and `worst_idx` first to fail fast on
      // non-improving moves while caching every cell's shape on success.
      const auto star_cosine = [&](const Point& q, std::size_t worst_idx,
                                   std::size_t& blocker_idx) {
        p[0] = q[0];
        p[1] = q[1];
        p[2] = q[2];
        double value = -1.0;
        const std::size_t first = blocker_idx;
        {
          bool ok_f = false;
          const double fv = shape(cells_[static_cast<std::size_t>(star[first])], &ok_f);
          trial_cos_[first] = fv;
          trial_safe_[first] = ok_f ? 1 : 0;
          if (fv >= best) return fv;
          value = fv;
        }
        if (worst_idx != first) {
          bool ok_w = false;
          const double wv = shape(cells_[static_cast<std::size_t>(star[worst_idx])], &ok_w);
          trial_cos_[worst_idx] = wv;
          trial_safe_[worst_idx] = ok_w ? 1 : 0;
          if (wv >= best) {
            blocker_idx = worst_idx;
            return wv;
          }
          value = std::max(value, wv);
        }
        for (std::size_t i = 0; i < star.size(); ++i) {
          if (i == first || i == worst_idx) continue;
          bool ok = false;
          const double sv = shape(cells_[static_cast<std::size_t>(star[i])], &ok);
          trial_cos_[i] = sv;
          trial_safe_[i] = ok ? 1 : 0;
          value = std::max(value, sv);
          if (value >= best) {
            blocker_idx = i;
            break;
          }
        }
        return value;
      };
      for (int ascent = 0; ascent < kAscents; ++ascent) {
        const Point origin{p[0], p[1], p[2]};
        std::size_t worst_idx = 0;
        auto worst_cell = static_cast<std::size_t>(star[0]);
        for (std::size_t i = 1; i < star.size(); ++i) {
          const auto s = static_cast<std::size_t>(star[i]);
          if (cosine_[s] > cosine_[worst_cell]) {
            worst_cell = s;
            worst_idx = i;
          }
        }
        const Cell& w = cells_[worst_cell];
        const auto slot = static_cast<int>(std::find(w.begin(), w.end(), v) - w.begin());
        const double* q0 = at(w[static_cast<std::size_t>(kOutwardFace[slot][0])]);
        const Point normal = cross(minus(at(w[static_cast<std::size_t>(kOutwardFace[slot][2])]), q0),
                                  minus(at(w[static_cast<std::size_t>(kOutwardFace[slot][1])]), q0));
        double edge = 0.0;
        for (const auto& e : kEdges) {
          const Point d = minus(at(w[static_cast<std::size_t>(e[0])]), at(w[static_cast<std::size_t>(e[1])]));
          edge += std::sqrt(dot(d, d)) / 6.0;
        }
        const double scale = edge / std::sqrt(dot(normal, normal));
        bool improved = false;
        std::size_t blocker_idx = worst_idx;
        for (std::size_t j = 0; j < 3; ++j) {
          const Point q{origin[0] + kSteps[j] * scale * normal[0],
                        origin[1] + kSteps[j] * scale * normal[1],
                        origin[2] + kSteps[j] * scale * normal[2]};
          const double value = star_cosine(q, worst_idx, blocker_idx);
          if (value < best) {
            best = value;
            improved = true;
            break;
          }
        }
        if (!improved) {
          // Compute centroid only if normal-offset candidates failed.
          const double denom = 4.0 * static_cast<double>(star.size());
          Point centroid{};
          for (int32_t s : star)
            for (int32_t u : cells_[static_cast<std::size_t>(s)])
              for (std::size_t d = 0; d < 3; ++d)
                centroid[d] += (u == v ? origin[d] : at(u)[d]) / denom;
          std::size_t centroid_blocker = worst_idx;
          const double value = star_cosine(centroid, worst_idx, centroid_blocker);
          if (value < best) {
            best = value;
            improved = true;
          }
        }
        if (!improved) {
          p[0] = origin[0];
          p[1] = origin[1];
          p[2] = origin[2];
          break;
        }
        moved = true;
        ++epoch_;
        for (std::size_t i = 0; i < star.size(); ++i) {
          const auto s = static_cast<std::size_t>(star[i]);
          cosine_[s] = trial_cos_[i];
          safe_[s] = trial_safe_[i];
          for (int32_t u : cells_[s]) vertex_epoch_[static_cast<std::size_t>(u)] = epoch_;
        }
        if (cosine_[c] <= target_) return true;
      }
    }
    return moved;
  }

  static constexpr std::uint64_t kEmptyEdge = ~std::uint64_t{0};
  static std::uint64_t edge_key(int32_t a, int32_t b) noexcept {
    const auto lo = static_cast<std::uint32_t>(a < b ? a : b);
    const auto hi = static_cast<std::uint32_t>(a < b ? b : a);
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
  }
  void insert_boundary_edge(int32_t a, int32_t b) noexcept {
    const std::uint64_t key = edge_key(a, b);
    std::size_t pos = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & boundary_mask_;
    while (boundary_edges_[pos] != kEmptyEdge && boundary_edges_[pos] != key)
      pos = (pos + 1) & boundary_mask_;
    boundary_edges_[pos] = key;
  }
  bool is_boundary_edge(int32_t a, int32_t b) const noexcept {
    const std::uint64_t key = edge_key(a, b);
    std::size_t pos = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & boundary_mask_;
    while (boundary_edges_[pos] != kEmptyEdge) {
      if (boundary_edges_[pos] == key) return true;
      pos = (pos + 1) & boundary_mask_;
    }
    return false;
  }

  std::vector<double>& points_;
  double target_, ratio_, determinant_;
  std::vector<char> fixed_;
  std::vector<std::uint64_t> boundary_edges_;
  std::size_t boundary_mask_ = 0;
  std::vector<StarView> stars_;
  std::vector<int32_t> star_pool_;
  std::vector<std::vector<int32_t>> overflow_;
  std::vector<Cell> cells_;
  std::vector<char> alive_;
  std::vector<double> cosine_;
  std::vector<std::uint8_t> safe_;
  std::vector<std::uint32_t> vertex_epoch_;
  std::vector<std::uint32_t> failed_epoch_;
  std::uint32_t epoch_ = 1;
  std::vector<double> trial_cos_;
  std::vector<std::uint8_t> trial_safe_;
};

// Whether binary64 shape estimates meet the quality targets (0 disables the
// radius-edge target).
bool meets_quality(std::span<const double> points, std::span<const int32_t> cells,
                   double radius_edge, double dihedral, double determinant) {
  for (std::size_t c = 0; c < cells.size() / 4; ++c) {
    const double* p[4];
    for (std::size_t k = 0; k < 4; ++k) p[k] = points.data() + 3 * static_cast<std::size_t>(cells[4 * c + k]);
    geometry::TetrahedronShape shape;
    if (!geometry::tetrahedron_shape(p, shape)) return false;
    if ((radius_edge > 0.0 && shape.radius_edge_ratio > radius_edge) ||
        geometry::minimum_dihedral_degrees(p) < dihedral ||
        geometry::relative_determinant(p) < determinant)
      return false;
  }
  return true;
}

// Outward faces of positive tetrahedra that no other cell shares.
void outward_boundary(std::span<const int32_t> cells, std::vector<int32_t>& faces) {
  struct Face {
    std::array<int32_t, 3> sorted, oriented;
  };
  std::vector<Face> all;
  all.reserve(cells.size());
  for (std::size_t c = 0; c < cells.size() / 4; ++c)
    for (const auto& face : kOutwardFace) {
      Face entry;
      for (std::size_t r = 0; r < 3; ++r) entry.oriented[r] = cells[4 * c + static_cast<std::size_t>(face[r])];
      entry.sorted = entry.oriented;
      std::sort(entry.sorted.begin(), entry.sorted.end());
      all.push_back(entry);
    }
  std::sort(all.begin(), all.end(), [](const Face& a, const Face& b) { return a.sorted < b.sorted; });
  faces.clear();
  for (std::size_t k = 0; k < all.size();) {
    std::size_t end = k + 1;
    while (end < all.size() && all[end].sorted == all[k].sorted) ++end;
    if (end == k + 1) faces.insert(faces.end(), all[k].oriented.begin(), all[k].oriented.end());
    k = end;
  }
}

// The fixed-boundary quality improvement of (points, cells, faces):
// improvement, then radius-edge refinement and improvement again when cells
// stay unresolved.
int32_t improve_quality(std::vector<double>& points, std::vector<int32_t>& cells,
                        std::vector<int32_t>& faces, const TetQualityRequest& request,
                        int64_t& unmet) {
  TetMesh mesh(BoundaryPolicy::kFixed);
  const std::size_t face_count = faces.size() / 3;
  std::vector<int32_t> regions(cells.size() / 4, 0), sources(face_count);
  std::iota(sources.begin(), sources.end(), 0);
  const std::vector<int8_t> orientations(face_count, 1);
  int32_t status = mesh.build(points, cells, regions, faces, sources, orientations, {}, {}, {}, nullptr);
  if (status != PHX_MC_OK) return status;
  status = mesh.assign_initial_ids({}, {}, {}, {});
  if (status != PHX_MC_OK) return status;
  int64_t inserted = 0, collapsed = 0, relocated = 0;
  status = improve_mesh(mesh, request, {}, inserted, collapsed, relocated, unmet);
  if (status == PHX_MC_UNRESOLVED) {
    status = refine_mesh(mesh, request.radius_edge_bound, {}, inserted, unmet);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    status = improve_mesh(mesh, request, {}, inserted, collapsed, relocated, unmet);
  }
  if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
  TetMeshSnapshot snapshot;
  if (mesh.snapshot(snapshot) != TetEditStatus::kApplied) return PHX_MC_INTERNAL_ERROR;
  std::vector<std::pair<std::uint64_t, int32_t>> rows;
  rows.reserve(snapshot.vertices.size());
  points.clear();
  for (const TetSnapshotVertex& vertex : snapshot.vertices) {
    rows.push_back({vertex.id, static_cast<int32_t>(rows.size())});
    points.insert(points.end(), vertex.carrier.begin(), vertex.carrier.end());
  }
  std::sort(rows.begin(), rows.end());
  const auto row_of = [&](std::uint64_t id) {
    return std::lower_bound(rows.begin(), rows.end(), std::pair<std::uint64_t, int32_t>{id, -1})->second;
  };
  cells.clear();
  for (const TetSnapshotCell& cell : snapshot.cells)
    for (std::uint64_t id : cell.vertices) cells.push_back(row_of(id));
  outward_boundary(cells, faces);
  return PHX_MC_OK;
}

struct MeshRequest {
  const double* lower;
  const double* upper;
  const double* criteria;
  const int64_t* topology;
  std::span<const double> extra_seeds;
};

// The seeds: the extra points moved onto the zero set, then the box roots
// farther than `spacing` from every earlier seed, polished on their edges;
// of these, the points farther than `separation` from every earlier one.
std::vector<Point> initial_seeds(Field& field, const PossibleZeroBoxes& roots, std::span<const double> extra,
                                 const double* box_lower, const double* box_upper, double spacing,
                                 double cell, double separation) {
  std::vector<Point> candidates;
  for (std::size_t k = 0; k + 2 < extra.size(); k += 3) candidates.push_back({extra[k], extra[k + 1], extra[k + 2]});
  project(field, candidates, kNewtonSteps);
  PointGrid seeded(std::max(spacing, cell), roots.roots.size() + candidates.size(), box_lower, box_upper);
  for (const Point& p : candidates) seeded.add(p);
  std::vector<Bracket> chosen;
  chosen.reserve(roots.roots.size());
  for (const Bracket& root : roots.roots) {
    if (seeded.near(root.root, spacing)) continue;
    seeded.add(root.root);
    chosen.push_back(root);
  }
  RootSolver solver;
  solver.solve(field, chosen);
  candidates.reserve(candidates.size() + chosen.size());
  for (const Bracket& root : chosen) candidates.push_back(root.root);
  PointGrid separated(separation, candidates.size(), box_lower, box_upper);
  std::vector<Point> seeds;
  seeds.reserve(candidates.size());
  for (const Point& p : candidates) {
    if (separated.near(p, separation)) continue;
    separated.add(p);
    seeds.push_back(p);
  }
  return seeds;
}

// The refinement state and the inside cells' restricted surface.
struct SurfaceState {
  int64_t insertions = 0;
  int64_t rounds = 0;
  bool matches = false;
  int64_t uncovered = 0;
  std::vector<int32_t> cells, faces;
  std::vector<std::array<int32_t, 2>> facets;
  TopologyScratch topo_scratch;
};

// Refines, then checks the restricted surface for edge-manifoldness, the
// requested topology and coverage of every box root; repairs by failure
// kind (the roots of uncovered boxes, the ball centers of facets at
// singular vertices, or else a finer surface everywhere) for up to
// kRepairRounds rounds. PHX_MC_UNRESOLVED when an insertion broke the
// separation bound.
int32_t refine_surface(RestrictedMesher& mesher, Field& field, const PossibleZeroBoxes& roots,
                       const double* box_lower, const double* box_upper, const int64_t* topology,
                       double width, SurfaceState& state) {
  RootSolver solver;
  for (int round = 0;; ++round) {
    int32_t status = mesher.refine(state.insertions);
    if (status != PHX_MC_OK) return status;
    mesher.surface(state.cells, state.faces, state.facets);
    const SurfaceTopology surface = surface_topology(state.faces, state.topo_scratch);
    // The surface's vertices, each once (a closed surface has about one per
    // two faces).
    PointGrid vertices(width, state.faces.size() / 6, box_lower, box_upper);
    std::vector<char> added(static_cast<std::size_t>(mesher.vertex_count()), 0);
    for (int32_t v : state.faces) {
      if (added[static_cast<std::size_t>(v)]) continue;
      added[static_cast<std::size_t>(v)] = 1;
      const double* p = mesher.point(v);
      vertices.add({p[0], p[1], p[2]});
    }
    std::vector<char> box_missed(roots.leaf_offsets.size() - 1, 0);
    state.uncovered = 0;
    for (std::size_t r = 0; r < roots.roots.size(); ++r) {
      if (vertices.near(roots.roots[r].root, width)) continue;
      ++state.uncovered;
      box_missed[static_cast<std::size_t>(roots.root_leaf[r])] = 1;
    }
    state.matches = surface.manifold && surface.components == topology[0] && surface.euler == topology[1];
    if ((state.matches && state.uncovered == 0) || round == kRepairRounds) return PHX_MC_OK;
    ++state.rounds;
    std::vector<Bracket> repair;
    for (std::size_t b = 0; b + 1 < roots.leaf_offsets.size(); ++b) {
      if (!box_missed[b]) continue;
      for (int32_t k = roots.leaf_offsets[b]; k < roots.leaf_offsets[b + 1]; ++k)
        repair.push_back(roots.roots[static_cast<std::size_t>(roots.leaf_roots[static_cast<std::size_t>(k)])]);
    }
    solver.solve(field, repair);
    std::vector<Point> points;
    for (const Bracket& root : repair) points.push_back(root.root);
    std::vector<std::array<int32_t, 2>> singular_facets;
    for (std::size_t f = 0; f < state.facets.size(); ++f)
      for (std::size_t r = 0; r < 3; ++r)
        if (std::binary_search(surface.singular.begin(), surface.singular.end(), state.faces[3 * f + r])) {
          singular_facets.push_back(state.facets[f]);
          break;
        }
    std::vector<Point> centers;
    mesher.ball_centers(singular_facets, centers);
    points.insert(points.end(), centers.begin(), centers.end());
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.empty()) {
      mesher.criteria().edge2 *= 0.5;
      mesher.reexamine();
    }
    status = mesher.insert_all(points);
    if (status != PHX_MC_OK) return status;
  }
}

// The inside cells and their boundary faces with the vertices they use, in
// vertex order; records each compacted row's mesher vertex ID in `mesher_ids`,
// and returns false when a far corner is one of them.
bool compact(const RestrictedMesher& mesher, const SurfaceState& state,
             phx_mc_restricted_volume& result, std::vector<int32_t>& mesher_ids) {
  std::vector<int32_t> rows(static_cast<std::size_t>(mesher.vertex_count()), -1);
  for (int32_t v : state.cells) rows[static_cast<std::size_t>(v)] = 0;
  result.points.clear();
  mesher_ids.clear();
  int32_t next = 0;
  for (std::size_t v = 0; v < rows.size(); ++v) {
    if (rows[v] < 0) continue;
    rows[v] = next++;
    mesher_ids.push_back(static_cast<int32_t>(v));
    const double* p = mesher.point(static_cast<int32_t>(v));
    result.points.push_back(p[0]);
    result.points.push_back(p[1]);
    result.points.push_back(p[2]);
  }
  result.cells = state.cells;
  result.faces = state.faces;
  for (int32_t& v : result.cells) v = rows[static_cast<std::size_t>(v)];
  for (int32_t& v : result.faces) v = rows[static_cast<std::size_t>(v)];
  return std::none_of(rows.begin(), rows.begin() + kCorners, [](int32_t row) { return row >= 0; });
}

constexpr std::size_t kBoxBlock = 64;

struct FastIntervalOp {
  std::uint8_t opcode;
  std::uint16_t aux;
  std::uint16_t out;
  std::uint16_t a;
  std::uint16_t b;
};

// Compiles a batch-admissible, unguarded interval program into flat scalar ops
// matching phx_mc_interval_program_batch (elementwise_arithmetic = true).
bool compile_fast_interval_ops(const IntervalProgramView& interval,
                               std::vector<FastIntervalOp>& ops) {
  constexpr std::int64_t kRowFields = 8;
  constexpr std::uint8_t kOpGather = 1, kOpAdd = 2, kOpSubtract = 3, kOpMultiply = 5,
                         kOpNegate = 6, kOpReduceSum = 7, kOpSqrt = 10, kOpSquare = 12,
                         kOpMaximum = 15, kOpMinimum = 16, kOpAbsolute = 20;
  const auto addr = [&](int64_t base, int64_t index, int64_t k) -> std::uint16_t {
    return static_cast<std::uint16_t>(base + (index < 0 ? k : interval.indices[index + k]));
  };
  ops.clear();
  ops.reserve(static_cast<std::size_t>(interval.instruction_count) * 2);
  for (int64_t r = 0; r < interval.instruction_count; ++r) {
    const int64_t* f = interval.instructions + r * kRowFields;
    const int64_t opcode = f[0], dst = f[1], size = f[2], a = f[3], a_idx = f[4], b = f[5],
                  b_idx = f[6], aux = f[7];
    switch (opcode) {
      case kOpReduceSum:
        if (aux < 0 || aux > 255) return false;
        for (int64_t k = 0; k < size; ++k) {
          ops.push_back({kOpReduceSum, static_cast<std::uint16_t>(aux),
                         static_cast<std::uint16_t>(dst + k),
                         static_cast<std::uint16_t>(a + k * aux),
                         static_cast<std::uint16_t>(a_idx == -1 ? 1 : 0)});
        }
        break;
      case kOpGather:
      case kOpNegate:
      case kOpSquare:
      case kOpSqrt:
      case kOpAbsolute:
        for (int64_t k = 0; k < size; ++k) {
          ops.push_back({static_cast<std::uint8_t>(opcode), 0, static_cast<std::uint16_t>(dst + k),
                         addr(a, a_idx, k), 0});
        }
        break;
      case kOpAdd:
      case kOpSubtract:
      case kOpMultiply:
      case kOpMaximum:
      case kOpMinimum:
        for (int64_t k = 0; k < size; ++k) {
          ops.push_back({static_cast<std::uint8_t>(opcode), 0, static_cast<std::uint16_t>(dst + k),
                         addr(a, a_idx, k), addr(b, b_idx, k)});
        }
        break;
      default:
        return false;
    }
  }
  return true;
}

// Evaluates `n <= kBoxBlock` boxes in SoA layout with exact elementwise
// interval arithmetic identical to phx_mc_interval_program_batch.
void execute_fast_interval_block(std::span<const FastIntervalOp> ops, double* __restrict lo,
                                 double* __restrict hi, std::size_t n) noexcept {
  constexpr std::uint8_t kOpGather = 1, kOpAdd = 2, kOpSubtract = 3, kOpMultiply = 5,
                         kOpNegate = 6, kOpReduceSum = 7, kOpSqrt = 10, kOpSquare = 12,
                         kOpMaximum = 15, kOpMinimum = 16, kOpAbsolute = 20;
  for (const FastIntervalOp& op : ops) {
    double* __restrict olo = lo + static_cast<std::size_t>(op.out) * kBoxBlock;
    double* __restrict ohi = hi + static_cast<std::size_t>(op.out) * kBoxBlock;
    const double* __restrict alo = lo + static_cast<std::size_t>(op.a) * kBoxBlock;
    const double* __restrict ahi = hi + static_cast<std::size_t>(op.a) * kBoxBlock;
    switch (op.opcode) {
      case kOpGather:
        for (std::size_t j = 0; j < n; ++j) {
          olo[j] = alo[j];
          ohi[j] = ahi[j];
        }
        break;
      case kOpNegate:
        for (std::size_t j = 0; j < n; ++j) {
          const double l = alo[j], h = ahi[j];
          olo[j] = geometry::next_down(-h);
          ohi[j] = geometry::next_up(-l);
        }
        break;
      case kOpAdd: {
        const double* __restrict blo = lo + static_cast<std::size_t>(op.b) * kBoxBlock;
        const double* __restrict bhi = hi + static_cast<std::size_t>(op.b) * kBoxBlock;
        for (std::size_t j = 0; j < n; ++j)
          interval::interval_sum(alo[j], ahi[j], blo[j], bhi[j], false, olo[j], ohi[j]);
        break;
      }
      case kOpSubtract: {
        const double* __restrict blo = lo + static_cast<std::size_t>(op.b) * kBoxBlock;
        const double* __restrict bhi = hi + static_cast<std::size_t>(op.b) * kBoxBlock;
        for (std::size_t j = 0; j < n; ++j)
          interval::interval_sum(alo[j], ahi[j], blo[j], bhi[j], true, olo[j], ohi[j]);
        break;
      }
      case kOpMultiply: {
        const double* __restrict blo = lo + static_cast<std::size_t>(op.b) * kBoxBlock;
        const double* __restrict bhi = hi + static_cast<std::size_t>(op.b) * kBoxBlock;
        for (std::size_t j = 0; j < n; ++j)
          interval::interval_product(alo[j], ahi[j], blo[j], bhi[j], olo[j], ohi[j]);
        break;
      }
      case kOpSquare:
        for (std::size_t j = 0; j < n; ++j) {
          const double l = alo[j], h = ahi[j];
          if (std::isnan(l) || std::isnan(h)) {
            olo[j] = -std::numeric_limits<double>::infinity();
            ohi[j] = std::numeric_limits<double>::infinity();
          } else {
            const bool crosses = l <= 0.0 && h >= 0.0;
            const double ls = l * l, hs = h * h;
            const double rlo = geometry::next_down(crosses ? 0.0 : ieee_min(ls, hs));
            olo[j] = ieee_max(rlo, 0.0);
            ohi[j] = geometry::next_up(ieee_max(ls, hs));
          }
        }
        break;
      case kOpSqrt:
        for (std::size_t j = 0; j < n; ++j) {
          const double l = alo[j], h = ahi[j];
          if (std::isnan(l) || std::isnan(h) || h < 0.0) {
            olo[j] = -std::numeric_limits<double>::infinity();
            ohi[j] = std::numeric_limits<double>::infinity();
          } else {
            olo[j] = geometry::next_down(std::sqrt(ieee_max(l, 0.0)));
            ohi[j] = geometry::next_up(std::sqrt(h));
          }
        }
        break;
      case kOpAbsolute:
        for (std::size_t j = 0; j < n; ++j) {
          const double l = alo[j], h = ahi[j];
          const bool crosses = l <= 0.0 && h >= 0.0;
          const double fl = std::fabs(l), fh = std::fabs(h);
          const double least = crosses ? 0.0 : ieee_min(fl, fh);
          const double most = ieee_max(fl, fh);
          if (std::isnan(least) || std::isnan(most)) {
            olo[j] = -std::numeric_limits<double>::infinity();
            ohi[j] = std::numeric_limits<double>::infinity();
          } else {
            olo[j] = least == 0.0 ? 0.0 : geometry::next_down(least);
            ohi[j] = geometry::next_up(most);
          }
        }
        break;
      case kOpMaximum: {
        const double* __restrict blo = lo + static_cast<std::size_t>(op.b) * kBoxBlock;
        const double* __restrict bhi = hi + static_cast<std::size_t>(op.b) * kBoxBlock;
        for (std::size_t j = 0; j < n; ++j) {
          olo[j] = ieee_max(alo[j], blo[j]);
          ohi[j] = ieee_max(ahi[j], bhi[j]);
        }
        break;
      }
      case kOpMinimum: {
        const double* __restrict blo = lo + static_cast<std::size_t>(op.b) * kBoxBlock;
        const double* __restrict bhi = hi + static_cast<std::size_t>(op.b) * kBoxBlock;
        for (std::size_t j = 0; j < n; ++j) {
          olo[j] = ieee_min(alo[j], blo[j]);
          ohi[j] = ieee_min(ahi[j], bhi[j]);
        }
        break;
      }
      case kOpReduceSum: {
        const bool start_at_first = op.b != 0 && op.aux > 0;
        for (std::size_t j = 0; j < n; ++j) {
          double l = start_at_first ? alo[j] : 0.0;
          double h = start_at_first ? ahi[j] : 0.0;
          for (std::uint16_t t = start_at_first ? 1 : 0; t < op.aux; ++t) {
            interval::interval_sum(l, h, alo[static_cast<std::size_t>(t) * kBoxBlock + j],
                                   ahi[static_cast<std::size_t>(t) * kBoxBlock + j], false, l, h);
          }
          olo[j] = l;
          ohi[j] = h;
        }
        break;
      }
      default:
        break;
    }
  }
}

// Fast possible-zero cover for small dyadic levels (<= 5) and batch-admissible,
// unguarded programs: deduplicated flat 1D coordinate tables, zero-copy SoA
// interval evaluation matching phx_mc_interval_program_batch bit-for-bit, and
// dense 3D lattice vertex/edge indexing. Falls back to possible_zero_boxes for
// deeper levels, guarded programs, or uncommon opcodes.
int32_t fast_possible_zero_boxes(const IntervalProgramView& interval, Field& field,
                                 const double* lower, const double* upper, int32_t target_level,
                                 PossibleZeroBoxes& out) {
  constexpr int32_t kMaxFastLevel = 5;
  constexpr std::size_t kMaxWs = 256;
  constexpr std::size_t kStackWs = 32;
  std::vector<FastIntervalOp> ops;
  if (target_level < 0 || target_level > kMaxFastLevel || interval.guard_count != 0 ||
      !field.is_fast() || interval.workspace_size < 3 ||
      static_cast<std::size_t>(interval.workspace_size) > kMaxWs ||
      !batch_admissible(interval.instructions, interval.instruction_count) ||
      !compile_fast_interval_ops(interval, ops)) {
    return possible_zero_boxes(interval, field.view(), lower, upper, target_level, out);
  }

  // Deduplicate identical axis intervals (common on cube/torus bounding boxes).
  std::size_t axis_src[3] = {0, 1, 2};
  for (std::size_t d = 1; d < 3; ++d) {
    for (std::size_t p = 0; p < d; ++p) {
      if (lower[d] == lower[p] && upper[d] == upper[p]) {
        axis_src[d] = p;
        break;
      }
    }
  }
  const int64_t denom = int64_t{1} << target_level;
  const int64_t grid_n = denom + 1;
  const Dyadic scale(static_cast<double>(denom));
  std::array<std::array<std::array<double, 3>, 33>, 3> coords{};
  std::array<std::array<std::uint8_t, 33>, 3> ready{};
  const auto coord_at = [&](std::size_t d, int32_t level, int64_t idx) -> const std::array<double, 3>& {
    const std::size_t s = axis_src[d];
    const int64_t fi = idx << (target_level - level);
    const auto u = static_cast<std::size_t>(fi);
    if (!ready[s][u]) {
      if (fi <= 0) {
        coords[s][u] = {lower[s], lower[s], lower[s]};
      } else if (fi >= denom) {
        coords[s][u] = {upper[s], upper[s], upper[s]};
      } else {
        const Dyadic num = Dyadic(lower[s]) * Dyadic(static_cast<double>(denom - fi)) +
                           Dyadic(upper[s]) * Dyadic(static_cast<double>(fi));
        const exact::QuotientBounds b = exact::directed_quotient_bounds(num, scale);
        coords[s][u] = {b.lower, b.upper, exact::rounded_quotient(num, scale)};
      }
      ready[s][u] = 1;
    }
    return coords[s][u];
  };

  const auto ws = static_cast<std::size_t>(interval.workspace_size);
  const auto output_slot = static_cast<std::size_t>(interval.output);
  double lo_stack[kStackWs * kBoxBlock];
  double hi_stack[kStackWs * kBoxBlock];
  std::vector<double> lo_heap, hi_heap;
  double* lo_ptr = lo_stack;
  double* hi_ptr = hi_stack;
  if (ws > kStackWs) {
    lo_heap.resize(ws * kBoxBlock);
    hi_heap.resize(ws * kBoxBlock);
    lo_ptr = lo_heap.data();
    hi_ptr = hi_heap.data();
  }
  for (std::size_t e = 0; e < ws; ++e) {
    std::fill_n(lo_ptr + e * kBoxBlock, kBoxBlock, interval.seed_lower[e]);
    std::fill_n(hi_ptr + e * kBoxBlock, kBoxBlock, interval.seed_upper[e]);
  }

  using GridIndex = std::array<int64_t, 3>;
  std::vector<GridIndex> frontier{{0, 0, 0}}, children;
  bool all_at_target = true;
  for (int32_t level = 0; !frontier.empty(); ++level) {
    children.clear();
    const std::size_t fcount = frontier.size();
    for (std::size_t first = 0; first < fcount; first += kBoxBlock) {
      const std::size_t chunk = std::min(kBoxBlock, fcount - first);
      std::uint8_t valid[kBoxBlock];
      std::size_t eval_idx[kBoxBlock];
      std::size_t n_eval = 0;
      for (std::size_t j = 0; j < chunk; ++j) {
        const GridIndex& idx = frontier[first + j];
        const double lx = coord_at(0, level, idx[0])[0], ux = coord_at(0, level, idx[0] + 1)[1];
        const double ly = coord_at(1, level, idx[1])[0], uy = coord_at(1, level, idx[1] + 1)[1];
        const double lz = coord_at(2, level, idx[2])[0], uz = coord_at(2, level, idx[2] + 1)[1];
        if (lx < ux && ly < uy && lz < uz) {
          valid[j] = 1;
          lo_ptr[n_eval] = lx;
          lo_ptr[kBoxBlock + n_eval] = ly;
          lo_ptr[2 * kBoxBlock + n_eval] = lz;
          hi_ptr[n_eval] = ux;
          hi_ptr[kBoxBlock + n_eval] = uy;
          hi_ptr[2 * kBoxBlock + n_eval] = uz;
          eval_idx[j] = n_eval++;
        } else {
          valid[j] = 0;
          eval_idx[j] = 0;
        }
      }
      if (n_eval > 0) execute_fast_interval_block(ops, lo_ptr, hi_ptr, n_eval);
      const double* __restrict res_lo = lo_ptr + output_slot * kBoxBlock;
      const double* __restrict res_hi = hi_ptr + output_slot * kBoxBlock;
      for (std::size_t j = 0; j < chunk; ++j) {
        const GridIndex& idx = frontier[first + j];
        const auto box_of = [&]() -> std::array<double, 6> {
          return {coord_at(0, level, idx[0])[0],     coord_at(1, level, idx[1])[0],
                  coord_at(2, level, idx[2])[0],     coord_at(0, level, idx[0] + 1)[1],
                  coord_at(1, level, idx[1] + 1)[1], coord_at(2, level, idx[2] + 1)[1]};
        };
        if (!valid[j]) {
          out.leaves.push_back({level, idx, box_of(), true, false});
          if (level != target_level) all_at_target = false;
          continue;
        }
        const std::size_t e = eval_idx[j];
        if (res_hi[e] < 0.0 || res_lo[e] > 0.0) continue;
        if (level < target_level) {
          bool stalled = false;
          for (std::size_t d = 0; d < 3; ++d) {
            const double mid = coord_at(d, level + 1, 2 * idx[d] + 1)[2];
            const double low = coord_at(d, level, idx[d])[2];
            const double high = coord_at(d, level, idx[d] + 1)[2];
            stalled = stalled || !(mid > low && mid < high);
          }
          if (stalled) {
            out.leaves.push_back({level, idx, box_of(), true, false});
            all_at_target = false;
            continue;
          }
          for (int64_t octant = 0; octant < 8; ++octant)
            children.push_back({2 * idx[0] + (octant & 1), 2 * idx[1] + ((octant >> 1) & 1),
                                2 * idx[2] + ((octant >> 2) & 1)});
        } else {
          out.leaves.push_back({level, idx, box_of(), false, false});
        }
      }
    }
    frontier.swap(children);
  }

  // Dense 3D lattice vertex table: N <= 33, N^3 <= 35,937.
  const auto N = static_cast<std::size_t>(grid_n);
  std::vector<int32_t> vertex_rows(N * N * N, -1);
  std::vector<double> vertex_points;
  vertex_points.reserve(3 * std::min<std::size_t>(N * N * N, 4 * out.leaves.size() + 8));
  std::vector<std::array<int32_t, 8>> leaf_vertices(out.leaves.size());
  for (std::size_t b = 0; b < out.leaves.size(); ++b) {
    const DyadicLeaf& leaf = out.leaves[b];
    const int32_t shift = target_level - leaf.level;
    for (int c = 0; c < 8; ++c) {
      const auto ix = static_cast<std::size_t>((leaf.index[0] + (c & 1)) << shift);
      const auto iy = static_cast<std::size_t>((leaf.index[1] + ((c >> 1) & 1)) << shift);
      const auto iz = static_cast<std::size_t>((leaf.index[2] + ((c >> 2) & 1)) << shift);
      int32_t& row = vertex_rows[(ix * N + iy) * N + iz];
      if (row < 0) {
        row = static_cast<int32_t>(vertex_points.size() / 3);
        vertex_points.push_back(coord_at(0, target_level, static_cast<int64_t>(ix))[2]);
        vertex_points.push_back(coord_at(1, target_level, static_cast<int64_t>(iy))[2]);
        vertex_points.push_back(coord_at(2, target_level, static_cast<int64_t>(iz))[2]);
      }
      leaf_vertices[b][static_cast<std::size_t>(c)] = row;
    }
  }
  const std::size_t vcount = vertex_points.size() / 3;
  std::vector<double> vertex_values(vcount);
  if (vcount > 0) field.values(vertex_points, vertex_values.data());

  const bool dense_edges = all_at_target && vcount <= 16384;
  std::vector<int32_t> dense_edge_roots;
  std::optional<KeyMap> sparse_edge_roots;
  if (dense_edges) {
    dense_edge_roots.assign(3 * vcount, -1);
  } else {
    sparse_edge_roots.emplace(out.leaves.size());
  }
  out.roots.reserve(out.leaves.size());
  out.root_leaf.reserve(out.leaves.size());
  out.leaf_roots.reserve(4 * out.leaves.size());
  out.leaf_offsets.reserve(out.leaves.size() + 1);
  constexpr int kAxisEdges[3][4][2] = {
      {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
      {{0, 2}, {1, 3}, {4, 6}, {5, 7}},
      {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
  };
  for (std::size_t b = 0; b < out.leaves.size(); ++b) {
    const auto& lv = leaf_vertices[b];
    double fv[8];
    bool has_neg = false, has_pos = false;
    for (int c = 0; c < 8; ++c) {
      const double v = vertex_values[static_cast<std::size_t>(lv[static_cast<std::size_t>(c)])];
      fv[c] = v;
      has_neg = has_neg || v < 0.0;
      has_pos = has_pos || v > 0.0;
    }
    if (has_neg && has_pos) {
      for (int axis = 0; axis < 3; ++axis) {
        for (int e = 0; e < 4; ++e) {
          const int c0 = kAxisEdges[axis][e][0], c1 = kAxisEdges[axis][e][1];
          const double fa = fv[c0], fb = fv[c1];
          if (!((fa < 0.0 && fb > 0.0) || (fa > 0.0 && fb < 0.0))) continue;
          const int32_t ra = lv[static_cast<std::size_t>(c0)];
          const int32_t rb = lv[static_cast<std::size_t>(c1)];
          int32_t root_id = -1;
          if (dense_edges) {
            root_id =
                dense_edge_roots[static_cast<std::size_t>(ra) * 3 + static_cast<std::size_t>(axis)];
          } else {
            const int64_t slot_val = sparse_edge_roots->slot(
                (static_cast<std::uint64_t>(ra) << 7) | (static_cast<std::uint64_t>(axis) << 5) |
                static_cast<std::uint64_t>(out.leaves[b].level));
            root_id = static_cast<int32_t>(slot_val);
          }
          if (root_id < 0) {
            root_id = static_cast<int32_t>(out.roots.size());
            if (dense_edges) {
              dense_edge_roots[static_cast<std::size_t>(ra) * 3 + static_cast<std::size_t>(axis)] =
                  root_id;
            } else {
              sparse_edge_roots->slot((static_cast<std::uint64_t>(ra) << 7) |
                                      (static_cast<std::uint64_t>(axis) << 5) |
                                      static_cast<std::uint64_t>(out.leaves[b].level)) = root_id;
            }
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
          out.leaf_roots.push_back(root_id);
        }
      }
    }
    out.leaf_offsets.push_back(static_cast<int32_t>(out.leaf_roots.size()));
  }
  return PHX_MC_OK;
}

// PHX_MC_OK, or PHX_MC_UNRESOLVED with the proposal reached: an insertion
// broke the separation bound, an inside cell reached a far corner, or a
// facet the bound refused still violates the criteria.
int32_t mesh_restricted(const IntervalProgramView& interval, Field& field, const MeshRequest& request,
                        phx_mc_restricted_volume& result) {
  constexpr int kQualityRounds = 3;
  constexpr int kSliverPasses = 6;
  const double* criteria = request.criteria;
  const double width = criteria[0];
  // A quarter of the smallest insertion radius the criteria ask for, so
  // that the bound binds only where they cannot be met (RestrictedMesher).
  const double separation = 0.25 * std::min({0.5 * width, criteria[1], criteria[3]});
  // The anchored domain of the possible-zero cover: the box grown by 1/16 of
  // its largest extent on every side, which keeps the zero set (tangent to
  // the box's faces on a sphere) off the domain boundary, where no lattice
  // edge crosses it; its target-level cells are no wider than the edge bound
  // (diagonal at most sqrt(3) times it).
  double largest = 0.0;
  for (int d = 0; d < 3; ++d) largest = std::max(largest, request.upper[d] - request.lower[d]);
  double domain_lower[3], domain_upper[3];
  for (int d = 0; d < 3; ++d) {
    domain_lower[d] = request.lower[d] - 0.0625 * largest;
    domain_upper[d] = request.upper[d] + 0.0625 * largest;
  }
  PossibleZeroBoxes roots;
  int32_t status = fast_possible_zero_boxes(
      interval, field, domain_lower, domain_upper,
      dyadic_target_level(domain_lower, domain_upper, std::sqrt(3.0) * width), roots);
  if (status != PHX_MC_OK) return status;
  const std::vector<Point> seeds =
      initial_seeds(field, roots, request.extra_seeds, domain_lower, domain_upper,
                    criteria[7] * width, width, separation);
  const double angle = criteria[2] * (std::numbers::pi / 180.0);
  const Criteria refinement{width * width, criteria[1] * criteria[1],
                            std::sin(angle) * std::sin(angle), criteria[3] * criteria[3],
                            criteria[4] > 0.0 ? criteria[4] * criteria[4]
                                              : std::numeric_limits<double>::infinity()};
  RestrictedMesher mesher(field, request.lower, request.upper, refinement, separation);
  status = mesher.start(seeds);
  if (status != PHX_MC_OK) return status;
  SurfaceState state;
  std::vector<Point> leftover;
  std::vector<int32_t> mesher_ids;
  PolishCache polish_cache;
  bool verified_safe = false;
  for (int round = 0;; ++round) {
    verified_safe = false;
    status = refine_surface(mesher, field, roots, domain_lower, domain_upper, request.topology, width, state);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    if (!compact(mesher, state, result, mesher_ids)) status = PHX_MC_UNRESOLVED;
    if (status == PHX_MC_UNRESOLVED) {
      result.report = {state.insertions, state.rounds, state.matches, state.uncovered, mesher.refused_facets(), 0};
      return status;
    }
    polish_boundary(field, result.points, result.cells, result.faces, mesher_ids, polish_cache);
    if (!(criteria[5] > 0.0)) break;
    SliverRemover remover(result.points, result.cells, result.faces, criteria[5], criteria[4], criteria[6]);
    // Cells the flips and relocations leave below the targets are refined
    // at their circumcenters in the Delaunay triangulation, and the surface
    // and its interior constructed again.
    bool all_safe = false;
    const int64_t remaining = remover.run(kSliverPasses, result.cells, leftover, all_safe);
    verified_safe = remaining == 0 && all_safe;
    if (remaining == 0 || round == kQualityRounds) break;
    status = mesher.refine_at(leftover);
    if (status == PHX_MC_UNRESOLVED) {
      // The arrays are the previous round's construction.
      result.report = {state.insertions, state.rounds, state.matches, state.uncovered, mesher.refused_facets(), 0};
      return status;
    }
    if (status != PHX_MC_OK) return status;
  }
  const int64_t refused = mesher.refused_facets();
  result.report = {state.insertions, state.rounds, state.matches, state.uncovered, refused, 0};
  const TetQualityRequest quality{criteria[4], criteria[5], criteria[6]};
  if (!result.cells.empty() && !verified_safe &&
      !meets_quality(result.points, result.cells, quality.radius_edge_bound,
                     quality.minimum_dihedral_degrees, quality.minimum_relative_determinant)) {
    status = improve_quality(result.points, result.cells, result.faces, quality, result.report[5]);
    if (status != PHX_MC_OK) return status;
  }
  return refused > 0 ? PHX_MC_UNRESOLVED : PHX_MC_OK;
}
}  // namespace
}  // namespace phx::mc
extern "C" PHX_MC_API int32_t phx_mc_restricted_centers_3d(
    int64_t point_count, const double* points, int64_t tet_count,
    const int32_t* tets, double* center_bounds, int32_t* item_status) {
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::construct_centers(point_count, points, tet_count, tets,
                                      center_bounds, item_status);
  });
}

extern "C" PHX_MC_API int32_t phx_mc_implicit_restricted_delaunay(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count,
    const double* seed_lower, const double* seed_upper, int64_t workspace_size,
    int64_t output_entry, const int64_t* guards, int64_t guard_count, const double* lower,
    const double* upper, const double* criteria,
    const int64_t* topology, int64_t extra_seed_count, const double* extra_seeds,
    phx_mc_restricted_volume** volume) {
  return phx::mc::guarded([&]() -> int32_t {
    if (volume == nullptr) return PHX_MC_INVALID_ARGUMENT;
    *volume = nullptr;
    if (seed_lower == nullptr || seed_upper == nullptr || lower == nullptr ||
        upper == nullptr || criteria == nullptr || topology == nullptr || extra_seed_count < 0 ||
        (extra_seed_count > 0 && extra_seeds == nullptr) ||
        !phx::mc::addressable(extra_seed_count, 3, sizeof(double)))
      return PHX_MC_INVALID_ARGUMENT;
    for (int d = 0; d < 3; ++d)
      if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] < upper[d]))
        return PHX_MC_INVALID_ARGUMENT;
    // Edge bound, facet distance, facet angle (at most 30 degrees, the
    // Boissonnat-Oudot bound that keeps the refinement finite), cell size,
    // radius-edge ratio (0: none), dihedral angle, relative determinant and
    // the seed spacing per edge bound.
    if (!(criteria[0] > 0.0) || !std::isfinite(criteria[0]) || !(criteria[1] > 0.0) ||
        !(criteria[2] >= 0.0 && criteria[2] <= 30.0) || !(criteria[3] > 0.0) ||
        !(criteria[4] >= 0.0) || !(criteria[5] >= 0.0 && criteria[5] < 70.0) ||
        !(criteria[6] >= 0.0 && criteria[6] < 1.0) || !(criteria[7] >= 0.0 && criteria[7] <= 1.0))
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t k = 0; k < 3 * extra_seed_count; ++k)
      if (!std::isfinite(extra_seeds[k])) return PHX_MC_INVALID_ARGUMENT;
    if (output_entry < 0 || output_entry >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
    if (guard_count < 0 || (guard_count > 0 && guards == nullptr) ||
        !phx::mc::addressable(guard_count, 1, sizeof(int64_t)))
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t g = 0; g < guard_count; ++g)
      if (guards[g] < 0 || guards[g] >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    int32_t status = phx::mc::validate_point_program(
        instructions, instruction_count, indices, index_count, parameters, parameter_count,
        workspace_size, 3, &output_entry, 1);
    if (status != PHX_MC_OK) return status;
    // The point executor reads each seed's midpoint, as phx_mc_program_evaluate's callers pass.
    std::vector<double> point_seeds(static_cast<std::size_t>(workspace_size));
    for (std::size_t k = 0; k < point_seeds.size(); ++k)
      point_seeds[k] = 0.5 * (seed_lower[k] + seed_upper[k]);
    const phx::mc::ProgramView view{instructions,      instruction_count, indices, point_seeds.data(),
                                    workspace_size,    3,                 &output_entry, 1};
    phx::mc::Field field(view);
    const phx::mc::IntervalProgramView interval{instructions, instruction_count, indices,
                                                index_count,  parameters,        parameter_count,
                                                seed_lower,   seed_upper,        workspace_size,
                                                output_entry, guards,            guard_count};
    auto result = std::make_unique<phx_mc_restricted_volume>();
    status = phx::mc::mesh_restricted(
        interval, field,
        {lower, upper, criteria, topology,
         std::span<const double>(extra_seeds, static_cast<std::size_t>(3 * extra_seed_count))},
        *result);
    if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
    *volume = result.release();
    return status;
  });
}

extern "C" PHX_MC_API int32_t phx_mc_restricted_volume_sizes(
    const phx_mc_restricted_volume* volume, int64_t* sizes, int64_t* report) {
  if (volume == nullptr || sizes == nullptr || report == nullptr) return PHX_MC_INVALID_ARGUMENT;
  sizes[0] = static_cast<int64_t>(volume->points.size() / 3);
  sizes[1] = static_cast<int64_t>(volume->cells.size() / 4);
  sizes[2] = static_cast<int64_t>(volume->faces.size() / 3);
  std::copy(volume->report.begin(), volume->report.end(), report);
  return PHX_MC_OK;
}

extern "C" PHX_MC_API int32_t phx_mc_restricted_volume_copy(
    const phx_mc_restricted_volume* volume, double* points, int32_t* cells, int32_t* faces) {
  if (volume == nullptr || (!volume->points.empty() && points == nullptr) ||
      (!volume->cells.empty() && cells == nullptr) || (!volume->faces.empty() && faces == nullptr))
    return PHX_MC_INVALID_ARGUMENT;
  std::copy(volume->points.begin(), volume->points.end(), points);
  std::copy(volume->cells.begin(), volume->cells.end(), cells);
  std::copy(volume->faces.begin(), volume->faces.end(), faces);
  return PHX_MC_OK;
}

extern "C" PHX_MC_API void phx_mc_restricted_volume_free(phx_mc_restricted_volume* volume) {
  delete volume;
}
