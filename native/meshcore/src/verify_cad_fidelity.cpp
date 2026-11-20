// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_cad_fidelity.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "capi_guard.hpp"
#include "directed_interval.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace phx::mc::cad {
namespace {
namespace di = phx::mc::remesh;

struct Vertex { double weight[3]; };
struct Leaf { Vertex vertices[3]; int32_t depth; };

Vertex midpoint(const Vertex& a, const Vertex& b) {
  Vertex value{};
  for (int i = 0; i < 3; ++i) value.weight[i] = (a.weight[i] + b.weight[i]) * 0.5;
  return value;
}

void chart_point(const Vertex& vertex, const double* uv, di::Interval* point) {
  for (int axis = 0; axis < 2; ++axis) {
    point[axis] = di::point(0.0);
    for (int i = 0; i < 3; ++i)
      point[axis] = di::add(point[axis], di::multiply(di::point(vertex.weight[i]),
                                                     di::point(uv[2 * i + axis])));
  }
}

// chart_point, but a point interval when every product and sum is exact in
// binary64 (checked with error-free transformations), as for the original
// vertices and most dyadic midpoints.
void exact_chart_point(const Vertex& vertex, const double* uv, di::Interval* point) {
  chart_point(vertex, uv, point);
  for (int axis = 0; axis < 2; ++axis) {
    double sum = 0.0;
    bool exact = true;
    for (int i = 0; i < 3 && exact; ++i) {
      const double product = vertex.weight[i] * uv[2 * i + axis];
      const double next = sum + product;
      // Knuth's TwoSum: the rounding error of sum + product.
      const double bv = next - sum, av = next - bv;
      exact = std::fma(vertex.weight[i], uv[2 * i + axis], -product) == 0.0 &&
              (sum - av) + (product - bv) == 0.0;
      sum = next;
    }
    if (exact && std::isfinite(sum)) point[axis] = di::point(sum);
  }
}

// 0: Hessian bound applicable, 1: use a first-order bound, 2: refuse.
int spline_knot_regularity(const double* lower, const double* upper, int degree, int count,
                           double box_lower, double box_upper) {
  int result = 0;
  for (int k = degree + 1; k < count; ++k) {
    if (box_upper < lower[k] || box_lower > upper[k]) continue;
    int multiplicity = 1;
    for (int j = k - 1; j >= 0 && upper[j] >= lower[k]; --j) ++multiplicity;
    for (int j = k + 1; j < count + degree + 1 && lower[j] <= upper[k]; ++j)
      ++multiplicity;
    if (multiplicity > degree) return 2;
    if (multiplicity >= degree) result = 1;
  }
  return result;
}

int spline_regularity(const CadGeometry& geometry, int32_t index, const double* box_lower,
                      const double* box_upper, const IntervalEvaluationWorkspace& workspace) {
  const Node& node = geometry.node(index);
  const int64_t offset = geometry.model().node_real_offsets[index];
  const auto child = [&](int slot, const double* lo, const double* hi) {
    return spline_regularity(geometry, static_cast<int32_t>(node.integers[slot]),
                             lo, hi, workspace);
  };
  const double all_lower[2] = {-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()};
  const double all_upper[2] = {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
  switch (node.kind) {
    case PHX_MC_CAD_NODE_BSPLINE_CURVE:
    case PHX_MC_CAD_NODE_BSPLINE_SURFACE: {
      const bool surface = node.kind == PHX_MC_CAD_NODE_BSPLINE_SURFACE;
      const int p = static_cast<int>(node.integers[0]);
      const int nu = static_cast<int>(node.integers[surface ? 2 : 1]);
      const double* lo = workspace.coefficient_lower + offset;
      const double* hi = workspace.coefficient_upper + offset;
      int result = spline_knot_regularity(lo, hi, p, nu, box_lower[0], box_upper[0]);
      if (surface) {
        const int q = static_cast<int>(node.integers[1]), nv = static_cast<int>(node.integers[3]);
        const int uk = nu + p + 1;
        result = std::max(result, spline_knot_regularity(lo + uk, hi + uk, q, nv,
                                                         box_lower[1], box_upper[1]));
      }
      return result;
    }
    case PHX_MC_CAD_NODE_AFFINE:
      return child(0, box_lower, box_upper);
    // An offset needs one more base derivative than its own jet: only template
    // children provide third-order interval jets, so others get first order.
    case PHX_MC_CAD_NODE_OFFSET_SURFACE:
    case PHX_MC_CAD_NODE_OFFSET_CURVE: {
      const auto base = static_cast<int32_t>(node.integers[0]);
      const int regularity = node.kind == PHX_MC_CAD_NODE_OFFSET_SURFACE
                                 ? child(0, box_lower, box_upper)
                                 : child(0, all_lower, all_upper);
      if (regularity != 0) return 2;
      return geometry.node(base).kind == PHX_MC_CAD_NODE_TEMPLATE ? 0 : 1;
    }
    case PHX_MC_CAD_NODE_EXTRUSION:
      return child(0, box_lower, box_upper);
    case PHX_MC_CAD_NODE_REVOLUTION:
      return child(0, box_lower + 1, box_upper + 1);
    case PHX_MC_CAD_NODE_RULED:
      return std::max(child(0, box_lower, box_upper), child(1, box_lower, box_upper));
    case PHX_MC_CAD_NODE_REPARAMETERIZED:
    case PHX_MC_CAD_NODE_ISOPARAMETRIC:
      return child(0, all_lower, all_upper);
    case PHX_MC_CAD_NODE_SURFACE_MAPPED:
    case PHX_MC_CAD_NODE_INTERSECTION:
      return std::max(child(0, all_lower, all_upper), child(1, all_lower, all_upper));
    default:
      return 0;
  }
}

// A C0 (not C1) knot line {uv[axis] = knot} of a spline surface chart.
struct KnotLine { int axis; double knot; };
constexpr int kMaxKnotLines = 8;
constexpr int kMaxPieces = 32;
constexpr int kMaxPieceVertices = 8;
// A clip point whose enclosure is wider than this fraction of its edge's
// length (an edge nearly parallel to the knot line) makes the leaf take the
// first-order bound instead.
constexpr double kClipConditioning = 0x1p-20;

// Appends the C0 knot lines strictly inside the leaf box, in chart axes.
// False when the chart's knot lines cannot be located exactly (a knot that is
// not a point, a child in another parameter, too many lines): the caller then
// takes the first-order bound.
bool c0_knot_lines(const CadGeometry& geometry, int32_t index, int curve_axis,
                   const double* box_lower, const double* box_upper,
                   const IntervalEvaluationWorkspace& workspace, KnotLine* lines, int& count) {
  const Node& node = geometry.node(index);
  const auto child = [&](int slot, int axis) {
    return c0_knot_lines(geometry, static_cast<int32_t>(node.integers[slot]), axis, box_lower,
                         box_upper, workspace, lines, count);
  };
  const auto knots = [&](const double* lo, const double* hi, int degree, int n, int axis) {
    for (int k = degree + 1; k < n; ++k) {
      if (lo[k] == lo[k - 1] && hi[k] == hi[k - 1]) continue;
      if (box_upper[axis] < lo[k] || box_lower[axis] > hi[k]) continue;
      int multiplicity = 1;
      for (int j = k + 1; j < n + degree + 1 && lo[j] <= hi[k]; ++j) ++multiplicity;
      if (multiplicity < degree) continue;
      if (multiplicity > degree || lo[k] != hi[k]) return false;
      // A line on the box boundary leaves the leaf on one side.
      if (!(box_lower[axis] < lo[k] && lo[k] < box_upper[axis])) continue;
      if (count == kMaxKnotLines) return false;
      lines[count++] = {axis, lo[k]};
    }
    return true;
  };
  const int64_t offset = geometry.model().node_real_offsets[index];
  const double* lo = workspace.coefficient_lower + offset;
  const double* hi = workspace.coefficient_upper + offset;
  switch (node.kind) {
    case PHX_MC_CAD_NODE_BSPLINE_SURFACE: {
      const int p = static_cast<int>(node.integers[0]), q = static_cast<int>(node.integers[1]);
      const int nu = static_cast<int>(node.integers[2]), nv = static_cast<int>(node.integers[3]);
      return knots(lo, hi, p, nu, 0) && knots(lo + nu + p + 1, hi + nu + p + 1, q, nv, 1);
    }
    case PHX_MC_CAD_NODE_BSPLINE_CURVE:
      return curve_axis >= 0 &&
             knots(lo, hi, static_cast<int>(node.integers[0]),
                   static_cast<int>(node.integers[1]), curve_axis);
    case PHX_MC_CAD_NODE_AFFINE:
      return child(0, curve_axis);
    case PHX_MC_CAD_NODE_EXTRUSION:
      return child(0, 0);
    case PHX_MC_CAD_NODE_REVOLUTION:
      return child(0, 1);
    case PHX_MC_CAD_NODE_RULED:
      return child(0, 0) && child(1, 0);
    case PHX_MC_CAD_NODE_OFFSET_SURFACE:
    case PHX_MC_CAD_NODE_OFFSET_CURVE:
    case PHX_MC_CAD_NODE_REPARAMETERIZED:
    case PHX_MC_CAD_NODE_ISOPARAMETRIC:
    case PHX_MC_CAD_NODE_SURFACE_MAPPED:
    case PHX_MC_CAD_NODE_INTERSECTION:
      return false;
    default:
      return true;
  }
}

// A polygon vertex: outward enclosures of its exact barycentric coordinates
// in the original triangle T and of its exact chart point.
struct ClipVertex { di::Interval weight[3]; di::Interval uv[2]; };
struct Piece { ClipVertex vertex[kMaxPieceVertices]; int size; };

// Directed quotient for a denominator of strict sign.
di::Interval divide(const di::Interval& a, const di::Interval& b) {
  const double q[4] = {a.lower / b.lower, a.lower / b.upper, a.upper / b.lower,
                       a.upper / b.upper};
  return {geometry::next_down(di::least(q[0], q[1], q[2], q[3])),
          geometry::next_up(di::greatest(q[0], q[1], q[2], q[3]))};
}

// -1 / +1: the exact point is strictly below / above the line; 0: it is on
// the line exactly; 2: undecided.
int line_side(const ClipVertex& vertex, const KnotLine& line) {
  const di::Interval& value = vertex.uv[line.axis];
  if (value.upper < line.knot) return -1;
  if (value.lower > line.knot) return 1;
  return value.lower == line.knot && value.upper == line.knot ? 0 : 2;
}

// Splits a convex polygon by one line into its two closed halves (exact
// Sutherland-Hodgman with enclosed crossing points). False: undecided side,
// an edge on the line, an ill-conditioned crossing, or capacity.
bool split_piece(const Piece& piece, const KnotLine& line, Piece& below, Piece& above) {
  below.size = above.size = 0;
  const auto push = [](Piece& target, const ClipVertex& vertex) {
    if (target.size == kMaxPieceVertices) return false;
    target.vertex[target.size++] = vertex;
    return true;
  };
  const int other = 1 - line.axis;
  for (int i = 0; i < piece.size; ++i) {
    const ClipVertex& p = piece.vertex[i];
    const ClipVertex& q = piece.vertex[(i + 1) % piece.size];
    const int sp = line_side(p, line), sq = line_side(q, line);
    if (sp == 2 || sq == 2 || (sp == 0 && sq == 0)) return false;
    if ((sp <= 0 && !push(below, p)) || (sp >= 0 && !push(above, p))) return false;
    if (sp * sq >= 0) continue;
    // c = p + s (q - p), s = (k - p_a) / (q_a - p_a); the denominator has
    // the strict sign of sq because the sides are strictly opposite.
    const auto denominator = di::subtract(q.uv[line.axis], p.uv[line.axis]);
    if (!(denominator.lower > 0.0 || denominator.upper < 0.0)) return false;
    const auto s = divide(di::subtract(di::point(line.knot), p.uv[line.axis]), denominator);
    ClipVertex c{};
    for (int w = 0; w < 3; ++w)
      c.weight[w] = di::add(p.weight[w], di::multiply(s, di::subtract(q.weight[w], p.weight[w])));
    c.uv[line.axis] = di::point(line.knot);
    c.uv[other] = di::add(p.uv[other], di::multiply(s, di::subtract(q.uv[other], p.uv[other])));
    // With point endpoints, a rounded crossing that the exact orientation
    // test puts on the segment's line is the exact crossing (its axis
    // coordinate is the knot), e.g. where an edge meets two knot lines.
    if (p.uv[0].lower == p.uv[0].upper && p.uv[1].lower == p.uv[1].upper &&
        q.uv[0].lower == q.uv[0].upper && q.uv[1].lower == q.uv[1].upper) {
      const double a[2] = {p.uv[0].lower, p.uv[1].lower}, b[2] = {q.uv[0].lower, q.uv[1].lower};
      double candidate[2];
      candidate[line.axis] = line.knot;
      candidate[other] = a[other] + (line.knot - a[line.axis]) * (b[other] - a[other]) /
                                        (b[line.axis] - a[line.axis]);
      if (std::isfinite(candidate[other]) && phx::mc::orient2d(a, b, candidate) == 0)
        c.uv[other] = di::point(candidate[other]);
    }
    const double length =
        std::hypot(0.5 * (q.uv[0].lower + q.uv[0].upper) - 0.5 * (p.uv[0].lower + p.uv[0].upper),
                   0.5 * (q.uv[1].lower + q.uv[1].upper) - 0.5 * (p.uv[1].lower + p.uv[1].upper));
    if (!di::finite(c.uv[other]) ||
        !(c.uv[other].upper - c.uv[other].lower <= kClipConditioning * length))
      return false;
    if (!push(below, c) || !push(above, c)) return false;
  }
  return below.size >= 3 && above.size >= 3;
}

// Piecewise rule at C0 knot lines (Python's _spline_piecewise_interpolation_bound).
// The exact leaf polygon is split by every C0 line strictly inside its box;
// clipping a convex polygon by the two closed half-planes of a line gives two
// convex polygons whose union is the polygon, so the exact pieces cover the
// leaf, and each lies in one closed knot cell where S is one smooth span
// piece. A fan triangle t of a piece has exact vertices inside the stored
// enclosures; on t, S - I_T S = (S - I_t S) + (I_t S - I_T S). The first term
// is bounded by the Hessian over the hull of t's vertex enclosures (it
// contains t; any span polynomial the hull touches is enclosed too); the
// second is affine and maximal at an exact vertex x of t, where it is
// |S(x) - sum_i w_i(x) X_i| evaluated over the enclosures of x and w(x).
// Returns false to request the first-order fallback.
bool clipped_leaf_bound(const CadGeometry& geometry, int32_t node, const Leaf& leaf,
                        const double* vertices, const double* uv, const double* box_lower,
                        const double* box_upper, IntervalEvaluationWorkspace& workspace,
                        int32_t& status, double& upper) {
  KnotLine lines[kMaxKnotLines];
  int line_count = 0;
  if (!c0_knot_lines(geometry, node, -1, box_lower, box_upper, workspace, lines, line_count))
    return false;
  Piece pieces[kMaxPieces];
  int piece_count = 1;
  pieces[0].size = 3;
  for (int i = 0; i < 3; ++i) {
    ClipVertex& vertex = pieces[0].vertex[i];
    for (int w = 0; w < 3; ++w) vertex.weight[w] = di::point(leaf.vertices[i].weight[w]);
    exact_chart_point(leaf.vertices[i], uv, vertex.uv);
  }
  for (int l = 0; l < line_count; ++l) {
    const int existing = piece_count;
    for (int i = 0; i < existing; ++i) {
      int sides = 0;
      for (int j = 0; j < pieces[i].size; ++j) {
        const int side = line_side(pieces[i].vertex[j], lines[l]);
        if (side == 2) return false;
        sides |= side < 0 ? 1 : side > 0 ? 2 : 0;
      }
      if (sides != 3) continue;
      if (piece_count == kMaxPieces) return false;
      Piece below, above;
      if (!split_piece(pieces[i], lines[l], below, above)) return false;
      pieces[i] = below;
      pieces[piece_count++] = above;
    }
  }
  const auto vertex_error = [&](const ClipVertex& vertex, double& error) {
    const double lo[2] = {vertex.uv[0].lower, vertex.uv[1].lower};
    const double hi[2] = {vertex.uv[0].upper, vertex.uv[1].upper};
    double surface_lower[3], surface_upper[3];
    status = geometry.interval_jets(node, lo, hi, 0, surface_lower, surface_upper, workspace);
    if (status != PHX_MC_OK) return false;
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      auto linear = di::point(0.0);
      for (int i = 0; i < 3; ++i)
        linear = di::add(linear, di::multiply(vertex.weight[i], di::point(vertices[3 * i + axis])));
      const auto difference = di::subtract({surface_lower[axis], surface_upper[axis]}, linear);
      if (!di::finite(difference)) { status = PHX_MC_UNRESOLVED; return false; }
      magnitude[axis] = di::magnitude(difference);
    }
    error = phx::mc::geometry::norm_upper(magnitude);
    return true;
  };
  double maximum = 0.0;
  status = PHX_MC_OK;
  for (int p = 0; p < piece_count; ++p) {
    const Piece& piece = pieces[p];
    double errors[kMaxPieceVertices];
    for (int j = 0; j < piece.size; ++j)
      if (!vertex_error(piece.vertex[j], errors[j])) return true;
    for (int j = 1; j + 1 < piece.size; ++j) {
      const ClipVertex* fan[3] = {&piece.vertex[0], &piece.vertex[j], &piece.vertex[j + 1]};
      double lo[2], hi[2];
      for (int axis = 0; axis < 2; ++axis) {
        lo[axis] = std::min({fan[0]->uv[axis].lower, fan[1]->uv[axis].lower,
                             fan[2]->uv[axis].lower});
        hi[axis] = std::max({fan[0]->uv[axis].upper, fan[1]->uv[axis].upper,
                             fan[2]->uv[axis].upper});
      }
      double jet_lower[18], jet_upper[18];
      status = geometry.interval_jets(node, lo, hi, 2, jet_lower, jet_upper, workspace);
      if (status != PHX_MC_OK) return true;
      const auto du = di::subtract(di::point(hi[0]), di::point(lo[0]));
      const auto dv = di::subtract(di::point(hi[1]), di::point(lo[1]));
      double magnitude[3];
      for (int axis = 0; axis < 3; ++axis) {
        const auto huu = di::point(di::magnitude({jet_lower[9 + axis], jet_upper[9 + axis]}));
        const auto huv = di::point(di::magnitude({jet_lower[12 + axis], jet_upper[12 + axis]}));
        const auto hvv = di::point(di::magnitude({jet_lower[15 + axis], jet_upper[15 + axis]}));
        auto sum = di::multiply(huu, di::multiply(du, du));
        const auto mixed = di::multiply(huv, di::multiply(du, dv));
        sum = di::add(sum, di::add(mixed, mixed));
        sum = di::add(sum, di::multiply(hvv, di::multiply(dv, dv)));
        magnitude[axis] = di::multiply(di::point(0.5), sum).upper;
      }
      const double vertex_term = std::max({errors[0], errors[j], errors[j + 1]});
      maximum = std::max(maximum, di::add(di::point(vertex_term),
                                          di::point(phx::mc::geometry::norm_upper(magnitude)))
                                      .upper);
    }
  }
  upper = maximum;
  if (!std::isfinite(upper)) status = PHX_MC_UNRESOLVED;
  return true;
}

// For each subdivision leaf L of the original triangle T,
// S - I_T = (S - I_L) + (I_L - I_T). On a C1 piece, the first term
// uses the leaf Hessian. A leaf meeting a continuous spline knot with a
// possible tangent jump takes the piecewise rule of clipped_leaf_bound; when
// that cannot locate or clip its lines soundly, use
// |S(x)-I_L S(x)| <= sum_i lambda_i |S(x)-S(v_i)| and bound each difference
// by the interval Jacobian over the leaf box.
// The second term is affine and maximized at a leaf vertex, including the
// mismatch against the stored original triangle. A possible discontinuity
// is refused: neither the Hessian nor the Jacobian implies a Lipschitz bound.
int32_t leaf_bound(const CadGeometry& geometry, int32_t node, const Leaf& leaf,
                   const double* vertices, const double* uv,
                   IntervalEvaluationWorkspace& workspace, double& upper) {
  double box_lower[2] = {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()};
  double box_upper[2] = {-std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()};
  double mismatch = 0.0;
  for (const Vertex& vertex : leaf.vertices) {
    di::Interval point[2];
    chart_point(vertex, uv, point);
    double lo[2], hi[2];
    for (int axis = 0; axis < 2; ++axis) {
      lo[axis] = point[axis].lower;
      hi[axis] = point[axis].upper;
      box_lower[axis] = std::min(box_lower[axis], lo[axis]);
      box_upper[axis] = std::max(box_upper[axis], hi[axis]);
    }
    double surface_lower[3], surface_upper[3];
    const int32_t status = geometry.interval_jets(node, lo, hi, 0, surface_lower,
                                                   surface_upper, workspace);
    if (status != PHX_MC_OK) return status;
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      auto linear = di::point(0.0);
      for (int i = 0; i < 3; ++i)
        linear = di::add(linear, di::multiply(di::point(vertex.weight[i]),
                                              di::point(vertices[3 * i + axis])));
      const auto difference = di::subtract({surface_lower[axis], surface_upper[axis]}, linear);
      if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
      magnitude[axis] = di::magnitude(difference);
    }
    mismatch = std::max(mismatch, phx::mc::geometry::norm_upper(magnitude));
  }
  const int regularity = spline_regularity(geometry, node, box_lower, box_upper, workspace);
  if (regularity == 2) return PHX_MC_UNRESOLVED;
  if (regularity == 1) {
    int32_t status = PHX_MC_OK;
    if (clipped_leaf_bound(geometry, node, leaf, vertices, uv, box_lower, box_upper, workspace,
                           status, upper))
      return status;
  }
  double lower[18], higher[18];
  const int32_t status = geometry.interval_jets(node, box_lower, box_upper,
                                                 regularity == 1 ? 1 : 2,
                                                 lower, higher, workspace);
  if (status != PHX_MC_OK) return status;
  const auto du = di::subtract(di::point(box_upper[0]), di::point(box_lower[0]));
  const auto dv = di::subtract(di::point(box_upper[1]), di::point(box_lower[1]));
  double magnitude[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (regularity == 1) {
      const auto su = di::point(di::magnitude({lower[3 + axis], higher[3 + axis]}));
      const auto sv = di::point(di::magnitude({lower[6 + axis], higher[6 + axis]}));
      magnitude[axis] = di::add(di::multiply(su, du), di::multiply(sv, dv)).upper;
      continue;
    }
    const auto huu = di::point(di::magnitude({lower[9 + axis], higher[9 + axis]}));
    const auto huv = di::point(di::magnitude({lower[12 + axis], higher[12 + axis]}));
    const auto hvv = di::point(di::magnitude({lower[15 + axis], higher[15 + axis]}));
    auto sum = di::multiply(huu, di::multiply(du, du));
    const auto mixed = di::multiply(huv, di::multiply(du, dv));
    sum = di::add(sum, di::add(mixed, mixed));
    sum = di::add(sum, di::multiply(hvv, di::multiply(dv, dv)));
    magnitude[axis] = di::multiply(di::point(0.5), sum).upper;
  }
  upper = di::add(di::point(mismatch),
                  di::point(phx::mc::geometry::norm_upper(magnitude))).upper;
  return std::isfinite(upper) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// The source boundary on a face is S(p(t)). The linear p-curve interpolant
// connects the chord endpoints; its second-derivative remainder plus endpoint
// mismatch bounds the distance to the chord. The straight homotopy from this
// chord to the arc fills the enclosed lens, and remains in the padded box.
// The mean-value bound |DS| times that distance therefore maps the lens to
// the surface image of the chord. The interior triangle bound already covers
// that image in both directions, so a separate 3D curve chord sag is not
// needed. Source 3D edge/face consistency is a source-model obligation.
int32_t trim_lens_bound(const CadGeometry& geometry, int32_t face, int64_t use,
                        const double* chord_uv, const double* parameters,
                        IntervalEvaluationWorkspace& workspace, double& cap) {
  const double low = std::min(parameters[0], parameters[1]);
  const double high = std::max(parameters[0], parameters[1]);
  const int regularity = spline_regularity(geometry, geometry.pcurve_node(use), &low, &high,
                                            workspace);
  if (regularity == 2) return PHX_MC_UNRESOLVED;
  double pc_lower[6], pc_upper[6];
  int32_t status = geometry.interval_jets(geometry.pcurve_node(use), &low, &high,
                                           regularity == 1 ? 1 : 2,
                                           pc_lower, pc_upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  const auto dt = di::subtract(di::point(high), di::point(low));
  const int derivative = regularity == 1 ? 2 : 4;
  double derivative_magnitude[3] = {
      di::magnitude({pc_lower[derivative], pc_upper[derivative]}),
      di::magnitude({pc_lower[derivative + 1], pc_upper[derivative + 1]}), 0.0};
  const double derivative_bound = phx::mc::geometry::norm_upper(derivative_magnitude);
  double endpoint_error = 0.0;
  for (int endpoint = 0; endpoint < 2; ++endpoint) {
    double lo[2], hi[2];
    status = geometry.interval_jets(geometry.pcurve_node(use), parameters + endpoint,
                                    parameters + endpoint, 0, lo, hi, workspace);
    if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
    double magnitude[3] = {0.0, 0.0, 0.0};
    for (int axis = 0; axis < 2; ++axis) {
      const auto difference = di::subtract({lo[axis], hi[axis]},
                                           di::point(chord_uv[2 * endpoint + axis]));
      if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
      magnitude[axis] = di::magnitude(difference);
    }
    endpoint_error = std::max(endpoint_error, phx::mc::geometry::norm_upper(magnitude));
  }
  const auto remainder = regularity == 1
      ? di::multiply(di::point(derivative_bound), dt)
      : di::multiply(di::point(0.125),
                     di::multiply(di::point(derivative_bound), di::multiply(dt, dt)));
  const auto error = di::add(di::point(endpoint_error), remainder);
  if (!di::finite(error)) return PHX_MC_UNRESOLVED;
  double box_lower[2], box_upper[2];
  for (int axis = 0; axis < 2; ++axis) {
    box_lower[axis] = di::subtract(di::point(std::min(chord_uv[axis], chord_uv[2 + axis])),
                                  di::point(error.upper)).lower;
    box_upper[axis] = di::add(di::point(std::max(chord_uv[axis], chord_uv[2 + axis])),
                              di::point(error.upper)).upper;
  }
  double surface_lower[9], surface_upper[9];
  status = geometry.interval_jets(geometry.surface_node(face), box_lower, box_upper, 1,
                                  surface_lower, surface_upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  double norms[3] = {};
  for (int derivative = 0; derivative < 2; ++derivative) {
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      const int index = 3 * (derivative + 1) + axis;
      magnitude[axis] = di::magnitude({surface_lower[index], surface_upper[index]});
    }
    norms[derivative] = phx::mc::geometry::norm_upper(magnitude);
  }
  cap = di::multiply(di::point(error.upper),
                     di::point(phx::mc::geometry::norm_upper(norms))).upper;
  return std::isfinite(cap) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// A collapsed source pole side maps to a single source point. Along a chart
// axis-aligned pole chord, moving radially from that chord to the sliver costs
// at most sup|S_r| times its radial extent. Two lens caps bridge the binary64
// chord to the exact source pole and back at the owner corner; the remaining
// interval endpoint mismatch reaches the stored mesh pole vertex. If the pole
// chord is oblique, the full two-coordinate derivative bound is used instead.
int32_t pole_sliver_bound(const CadGeometry& geometry, int32_t face, const double* uv,
                          int pole_edge, const double* pole_vertex, double lens,
                          IntervalEvaluationWorkspace& workspace, double& cap) {
  const int first = pole_edge, second = (pole_edge + 1) % 3;
  double box_lower[2], box_upper[2];
  for (int axis = 0; axis < 2; ++axis) {
    box_lower[axis] = std::min({uv[axis], uv[2 + axis], uv[4 + axis]});
    box_upper[axis] = std::max({uv[axis], uv[2 + axis], uv[4 + axis]});
  }
  double lower[9], upper[9];
  int32_t status = geometry.interval_jets(geometry.surface_node(face), box_lower, box_upper,
                                           1, lower, upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  double endpoint_lower[3], endpoint_upper[3];
  status = geometry.interval_jets(geometry.surface_node(face), uv + 2 * first, uv + 2 * first,
                                  0, endpoint_lower, endpoint_upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  double mismatch[3];
  for (int axis = 0; axis < 3; ++axis) {
    const auto difference = di::subtract({endpoint_lower[axis], endpoint_upper[axis]},
                                         di::point(pole_vertex[axis]));
    if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
    mismatch[axis] = di::magnitude(difference);
  }
  auto total = di::add(di::point(phx::mc::geometry::norm_upper(mismatch)),
                       di::add(di::point(lens), di::point(lens)));
  const bool same_u = uv[2 * first] == uv[2 * second];
  const bool same_v = uv[2 * first + 1] == uv[2 * second + 1];
  for (int axis = 0; axis < 2; ++axis) {
    // On a coordinate-aligned pole chord the tangential direction collapses.
    if ((axis == 0 && same_v && !same_u) || (axis == 1 && same_u && !same_v)) continue;
    const double origin = uv[2 * first + axis];
    const auto low = di::subtract(di::point(box_lower[axis]), di::point(origin));
    const auto high = di::subtract(di::point(box_upper[axis]), di::point(origin));
    const double extent = std::max(di::magnitude(low), di::magnitude(high));
    double derivative[3];
    for (int component = 0; component < 3; ++component) {
      const int index = 3 * (axis + 1) + component;
      derivative[component] = di::magnitude({lower[index], upper[index]});
    }
    total = di::add(total, di::multiply(di::point(extent),
                                       di::point(phx::mc::geometry::norm_upper(derivative))));
  }
  cap = total.upper;
  return std::isfinite(cap) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

struct EdgeSlot {
  std::array<uint64_t, 4> endpoints{};
  int32_t patch = -1;
  int64_t first = -1;
  int32_t count = 0;
  bool direction = false;
};

std::vector<phx_mc_fidelity_finding> association_findings(
    int64_t triangle_count, const int32_t* face_rows, const double* uv,
    int64_t chord_count, const int64_t* chord_use, const int64_t* use_patch,
    const double* chord_uv, const int64_t* edge_chord) {
  const auto edge_count = static_cast<std::size_t>(triangle_count) * 3;
  std::size_t table_size = 1;
  while (table_size < edge_count * 2) table_size *= 2;
  std::vector<EdgeSlot> table(table_size);
  std::vector<int64_t> owners(static_cast<std::size_t>(chord_count), -1);
  std::vector<phx_mc_fidelity_finding> findings;
  for (std::size_t index = 0; index < edge_count; ++index) {
    const int64_t triangle = static_cast<int64_t>(index / 3), edge = index % 3;
    const double* a = uv + 6 * triangle + 2 * edge;
    const double* b = uv + 6 * triangle + 2 * ((edge + 1) % 3);
    const int64_t chord = edge_chord[index];
    if (chord >= 0) {
      const double* c = chord_uv + 4 * chord;
      if (use_patch[chord_use[chord]] != face_rows[triangle] ||
          std::memcmp(a, c, 2 * sizeof(double)) != 0 ||
          std::memcmp(b, c + 2, 2 * sizeof(double)) != 0)
        findings.push_back({PHX_MC_FIDELITY_BAD_CHORD_EDGE, triangle, edge, -1, -1, chord});
      int64_t& owner = owners[static_cast<std::size_t>(chord)];
      if (owner >= 0)
        findings.push_back({PHX_MC_FIDELITY_DUPLICATE_CHORD, triangle, edge,
                            owner / 3, owner % 3, chord});
      else
        owner = static_cast<int64_t>(index);
      continue;
    }
    const std::array<uint64_t, 2> first{std::bit_cast<uint64_t>(a[0]), std::bit_cast<uint64_t>(a[1])};
    const std::array<uint64_t, 2> second{std::bit_cast<uint64_t>(b[0]), std::bit_cast<uint64_t>(b[1])};
    const bool direction = first < second;
    const auto& low = direction ? first : second;
    const auto& high = direction ? second : first;
    const std::array<uint64_t, 4> key{low[0], low[1], high[0], high[1]};
    uint64_t hash = static_cast<uint32_t>(face_rows[triangle]);
    for (uint64_t value : key) {
      value += UINT64_C(0x9e3779b97f4a7c15);
      value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
      value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
      value ^= value >> 31;
      hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    }
    std::size_t position = static_cast<std::size_t>(hash) & (table_size - 1);
    while (table[position].first >= 0 &&
           (table[position].patch != face_rows[triangle] || table[position].endpoints != key))
      position = (position + 1) & (table_size - 1);
    EdgeSlot& slot = table[position];
    if (slot.first < 0) {
      slot = {key, face_rows[triangle], static_cast<int64_t>(index), 1, direction};
    } else {
      ++slot.count;
      if (slot.count > 2 || slot.direction == direction || first == second)
        findings.push_back({PHX_MC_FIDELITY_UNPAIRED_INTERIOR, triangle, edge,
                            slot.first / 3, slot.first % 3, -1});
    }
  }
  for (const EdgeSlot& slot : table)
    if (slot.count == 1)
      findings.push_back({PHX_MC_FIDELITY_UNPAIRED_INTERIOR,
                          slot.first / 3, slot.first % 3, -1, -1, -1});
  for (int64_t chord = 0; chord < chord_count; ++chord)
    if (owners[static_cast<std::size_t>(chord)] < 0)
      findings.push_back({PHX_MC_FIDELITY_MISSING_CHORD, -1, -1, -1, -1, chord});
  return findings;
}

void chart_mesh_findings(
    const phx_mc_cad_model& model, int64_t point_count, int64_t chart_count,
    const int64_t* chart_triangles, int64_t mesh_count, const int64_t* mesh_triangles,
    const int64_t* chart_to_mesh, const int64_t* edge_chord, const int64_t* chord_use,
    std::vector<int32_t>& pole_edges, std::vector<phx_mc_fidelity_finding>& findings) {
  std::vector<int64_t> owners(static_cast<std::size_t>(mesh_count), -1);
  std::vector<uint8_t> mesh_vertices(static_cast<std::size_t>(point_count), 0);
  for (int64_t i = 0; i < mesh_count * 3; ++i)
    mesh_vertices[static_cast<std::size_t>(mesh_triangles[i])] = 1;
  for (int64_t row = 0; row < chart_count; ++row) {
    int32_t pole_edge = -1;
    bool bad = false;
    for (int edge = 0; edge < 3; ++edge) {
      const int64_t chord = edge_chord[3 * row + edge];
      if (chord < 0 || model.use_kinds[chord_use[chord]] != 1) continue;
      if (pole_edge >= 0 ||
          chart_triangles[3 * row + edge] != chart_triangles[3 * row + (edge + 1) % 3])
        bad = true;
      pole_edge = edge;
    }
    pole_edges[static_cast<std::size_t>(row)] = pole_edge;
    const int64_t mapped = chart_to_mesh[row];
    if (mapped == -1) {
      if (pole_edge < 0 ||
          !mesh_vertices[static_cast<std::size_t>(chart_triangles[3 * row + pole_edge])])
        bad = true;
    } else if (mapped < 0 || mapped >= mesh_count || pole_edge >= 0) {
      bad = true;
    } else {
      const int64_t* chart = chart_triangles + 3 * row;
      const int64_t* mesh = mesh_triangles + 3 * mapped;
      bool same = false;
      for (int rotation = 0; rotation < 3; ++rotation)
        same |= chart[0] == mesh[rotation] && chart[1] == mesh[(rotation + 1) % 3] &&
                chart[2] == mesh[(rotation + 2) % 3];
      int64_t& owner = owners[static_cast<std::size_t>(mapped)];
      if (!same || owner >= 0) bad = true;
      if (owner < 0) owner = row;
    }
    if (bad)
      findings.push_back({PHX_MC_FIDELITY_CHART_MESH_LINK, row, pole_edge, mapped, -1, -1});
  }
  for (int64_t row = 0; row < mesh_count; ++row)
    if (owners[static_cast<std::size_t>(row)] < 0)
      findings.push_back({PHX_MC_FIDELITY_MISSING_MESH_TRIANGLE, -1, -1, row, -1, -1});
}
}  // namespace

FidelityTriangleResult fidelity_triangle(const CadGeometry& geometry, int32_t face,
                                         const double* vertices, const double* uv,
                                         double tolerance, int32_t max_depth,
                                         IntervalEvaluationWorkspace& workspace) {
  FidelityTriangleResult result{std::numeric_limits<double>::infinity(), PHX_MC_UNRESOLVED,
                                PHX_MC_FIDELITY_INTERVAL_REFUSAL, 0};
  if (face < 0 || face >= geometry.surface_count() || vertices == nullptr || uv == nullptr ||
      !std::isfinite(tolerance) || tolerance < 0.0 || max_depth < 0 || max_depth > 24) {
    result.status = PHX_MC_INVALID_ARGUMENT;
    return result;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) { result.status = PHX_MC_INTERNAL_ERROR; return result; }
  Leaf stack[73];
  stack[0] = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}, 0};
  int size = 1;
  double maximum = 0.0;
  while (size > 0) {
    const Leaf leaf = stack[--size];
    double bound = 0.0;
    result.status = leaf_bound(geometry, geometry.surface_node(face), leaf, vertices, uv,
                               workspace, bound);
    if (result.status != PHX_MC_OK) {
      if (result.status == PHX_MC_INVALID_ARGUMENT || result.status == PHX_MC_INVALID_INPUT)
        result.status = PHX_MC_UNRESOLVED;
      return result;
    }
    if (bound <= tolerance) { maximum = std::max(maximum, bound); continue; }
    if (leaf.depth == max_depth || result.subdivisions >= 1000000) {
      result.status = PHX_MC_UNRESOLVED;
      result.reason = PHX_MC_FIDELITY_RESOLUTION;
      return result;
    }
    ++result.subdivisions;
    const Vertex ab = midpoint(leaf.vertices[0], leaf.vertices[1]);
    const Vertex bc = midpoint(leaf.vertices[1], leaf.vertices[2]);
    const Vertex ca = midpoint(leaf.vertices[2], leaf.vertices[0]);
    const int32_t depth = leaf.depth + 1;
    stack[size++] = {{leaf.vertices[0], ab, ca}, depth};
    stack[size++] = {{ab, leaf.vertices[1], bc}, depth};
    stack[size++] = {{ca, bc, leaf.vertices[2]}, depth};
    stack[size++] = {{ab, bc, ca}, depth};
  }
  result.upper = maximum;
  result.reason = PHX_MC_FIDELITY_CERTIFIED;
  return result;
}
}  // namespace phx::mc::cad

extern "C" int32_t phx_mc_verify_cad_fidelity(
    const phx_mc_cad_model* model, int64_t coefficient_count,
    const double* coefficient_lower, const double* coefficient_upper,
    int64_t point_count, const double* points, int64_t triangle_count,
    const int64_t* triangles, const int32_t* face_rows, const double* uv,
    int64_t mesh_triangle_count, const int64_t* mesh_triangles, const int64_t* chart_to_mesh,
    int64_t use_count, const int64_t* use_patch, const double* use_range,
    int64_t loop_count, int64_t chord_count, const int64_t* chord_use,
    const int64_t* chord_loop, const double* chord_uv, const double* chord_range,
    const int64_t* edge_chord, double tolerance, int32_t max_depth,
    int64_t output_capacity, double* upper, int32_t* item_status, int32_t* reasons,
    int32_t* subdivisions, double* maximum_upper, int64_t evidence_capacity,
    phx_mc_fidelity_finding* evidence, int64_t* evidence_count) {
  return phx::mc::guarded([&]() -> int32_t {
    if (model == nullptr || !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 6, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(mesh_triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(use_count, 2, sizeof(double)) ||
        !phx::mc::addressable(chord_count, 4, sizeof(double)) || loop_count < 0 ||
        coefficient_count < 0 || output_capacity < 0 || evidence_capacity < 0 ||
        maximum_upper == nullptr || evidence_count == nullptr ||
        !std::isfinite(tolerance) || tolerance < 0 || max_depth < 0 || max_depth > 24)
      return PHX_MC_INVALID_ARGUMENT;
    if (output_capacity < triangle_count) return PHX_MC_CAPACITY_EXCEEDED;
    if ((point_count && !points) ||
        (coefficient_count && (!coefficient_lower || !coefficient_upper)) ||
        (use_count && (!use_patch || !use_range)) ||
        (chord_count && (!chord_use || !chord_loop || !chord_uv || !chord_range)) ||
        (mesh_triangle_count && !mesh_triangles) ||
        (evidence_capacity && !evidence) ||
        (triangle_count && (!triangles || !face_rows || !uv || !edge_chord || !chart_to_mesh ||
                            !upper || !item_status || !reasons || !subdivisions)))
      return PHX_MC_INVALID_ARGUMENT;
    phx::mc::cad::CadGeometry geometry;
    const int32_t bind_status = geometry.bind(*model);
    if (bind_status != PHX_MC_OK) return bind_status;
    if (use_count != model->use_count) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < use_count; ++i)
      if (use_patch[i] < 0 || use_patch[i] >= model->surface_count ||
          !std::isfinite(use_range[2 * i]) || !std::isfinite(use_range[2 * i + 1]) ||
          use_range[2 * i] == use_range[2 * i + 1])
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < chord_count; ++i) {
      if (chord_use[i] < 0 || chord_use[i] >= use_count ||
          chord_loop[i] < 0 || chord_loop[i] >= loop_count ||
          !std::isfinite(chord_range[2 * i]) || !std::isfinite(chord_range[2 * i + 1]) ||
          chord_range[2 * i] == chord_range[2 * i + 1])
        return PHX_MC_INVALID_ARGUMENT;
      for (int j = 0; j < 4; ++j)
        if (!std::isfinite(chord_uv[4 * i + j])) return PHX_MC_NONFINITE_INPUT;
    }
    if (coefficient_count != model->node_real_offsets[model->node_count])
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < coefficient_count; ++i)
      if (std::isnan(coefficient_lower[i]) || std::isnan(coefficient_upper[i]) ||
          coefficient_lower[i] > coefficient_upper[i])
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < point_count * 3; ++i)
      if (!std::isfinite(points[i])) return PHX_MC_NONFINITE_INPUT;
    for (int64_t i = 0; i < mesh_triangle_count * 3; ++i)
      if (mesh_triangles[i] < 0 || mesh_triangles[i] >= point_count)
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < triangle_count; ++i) {
      if (face_rows[i] < 0 || face_rows[i] >= model->surface_count)
        return PHX_MC_INVALID_ARGUMENT;
      for (int j = 0; j < 3; ++j) {
        const int64_t chord = edge_chord[3 * i + j];
        if (triangles[3 * i + j] < 0 || triangles[3 * i + j] >= point_count ||
            chord < -1 || chord >= chord_count)
          return PHX_MC_INVALID_ARGUMENT;
      }
      for (int j = 0; j < 6; ++j)
        if (!std::isfinite(uv[6 * i + j])) return PHX_MC_NONFINITE_INPUT;
    }
    auto findings = phx::mc::cad::association_findings(
        triangle_count, face_rows, uv, chord_count, chord_use, use_patch, chord_uv, edge_chord);
    std::vector<int32_t> pole_edges(static_cast<std::size_t>(triangle_count), -1);
    phx::mc::cad::chart_mesh_findings(*model, point_count, triangle_count, triangles,
                                     mesh_triangle_count, mesh_triangles, chart_to_mesh,
                                     edge_chord, chord_use, pole_edges, findings);
    *evidence_count = static_cast<int64_t>(findings.size());
    if (*evidence_count > evidence_capacity) return PHX_MC_CAPACITY_EXCEEDED;
    if (!findings.empty()) {
      std::copy(findings.begin(), findings.end(), evidence);
      for (int64_t i = 0; i < triangle_count; ++i) {
        upper[i] = std::numeric_limits<double>::infinity();
        item_status[i] = PHX_MC_UNRESOLVED;
        reasons[i] = PHX_MC_FIDELITY_EDGE_ASSOCIATION;
        subdivisions[i] = 0;
      }
      *maximum_upper = std::numeric_limits<double>::infinity();
      return PHX_MC_UNRESOLVED;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    phx::mc::cad::IntervalEvaluationWorkspace workspace;
    workspace.coefficient_lower = coefficient_lower;
    workspace.coefficient_upper = coefficient_upper;
    workspace.coefficient_count = coefficient_count;
    int32_t aggregate = PHX_MC_OK;
    double maximum = 0.0;
    for (int64_t i = 0; i < triangle_count; ++i) {
      bool trim_refused = false;
      double lens = 0.0;
      double vertices[9];
      for (int j = 0; j < 3; ++j) {
        const int64_t chord = edge_chord[3 * i + j];
        if (chord >= 0) {
          const int64_t use = chord_use[chord];
          double cap = 0.0;
          if (phx::mc::cad::trim_lens_bound(geometry, face_rows[i], use, chord_uv + 4 * chord,
                                            chord_range + 2 * chord, workspace, cap) != PHX_MC_OK) {
            trim_refused = true;
          } else {
            lens = phx::mc::remesh::add(phx::mc::remesh::point(lens),
                                         phx::mc::remesh::point(cap)).upper;
          }
        }
        const int64_t vertex = triangles[3 * i + j];
        for (int axis = 0; axis < 3; ++axis)
          vertices[3 * j + axis] = points[3 * vertex + axis];
      }
      if (trim_refused || !std::isfinite(lens) || lens >= tolerance) {
        upper[i] = std::numeric_limits<double>::infinity();
        item_status[i] = PHX_MC_UNRESOLVED;
        reasons[i] = trim_refused ? PHX_MC_FIDELITY_TRIM_REFUSAL : PHX_MC_FIDELITY_RESOLUTION;
        subdivisions[i] = 0;
        aggregate = PHX_MC_UNRESOLVED;
        continue;
      }
      const int pole_edge = pole_edges[static_cast<std::size_t>(i)];
      if (pole_edge >= 0) {
        double cap = 0.0;
        const int32_t status = phx::mc::cad::pole_sliver_bound(
            geometry, face_rows[i], uv + 6 * i, pole_edge, vertices + 3 * pole_edge,
            lens, workspace, cap);
        if (status != PHX_MC_OK || cap > tolerance) {
          upper[i] = std::numeric_limits<double>::infinity();
          item_status[i] = PHX_MC_UNRESOLVED;
          reasons[i] = status == PHX_MC_OK ? PHX_MC_FIDELITY_RESOLUTION
                                           : PHX_MC_FIDELITY_INTERVAL_REFUSAL;
          subdivisions[i] = 0;
          aggregate = PHX_MC_UNRESOLVED;
        } else {
          upper[i] = cap;
          item_status[i] = PHX_MC_OK;
          reasons[i] = PHX_MC_FIDELITY_CERTIFIED;
          subdivisions[i] = 0;
          maximum = std::max(maximum, cap);
        }
        continue;
      }
      const double budget = phx::mc::remesh::subtract(phx::mc::remesh::point(tolerance),
                                                       phx::mc::remesh::point(lens)).lower;
      const auto result = phx::mc::cad::fidelity_triangle(
          geometry, face_rows[i], vertices, uv + 6 * i, budget, max_depth, workspace);
      if (result.status != PHX_MC_OK && result.status != PHX_MC_UNRESOLVED)
        return result.status;
      upper[i] = result.status == PHX_MC_OK
                     ? phx::mc::remesh::add(phx::mc::remesh::point(result.upper),
                                            phx::mc::remesh::point(lens)).upper
                     : result.upper;
      item_status[i] = result.status;
      reasons[i] = result.reason;
      subdivisions[i] = result.subdivisions;
      maximum = std::max(maximum, upper[i]);
      if (result.status != PHX_MC_OK) aggregate = PHX_MC_UNRESOLVED;
    }
    *maximum_upper = aggregate == PHX_MC_OK ? maximum : std::numeric_limits<double>::infinity();
    return aggregate;
  });
}
