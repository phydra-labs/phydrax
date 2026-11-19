// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// kZeroSetBand admission and the zero-set surface remesh C entry; contract in
// zero_set_admission.hpp and phydrax_meshcore.h.
#include "zero_set_admission.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <system_error>
#include <utility>
#include <vector>

#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"
#include "surface_adapt.hpp"

namespace phx::mc::remesh {
namespace {
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr int kNewtonSteps = 16;
// Midpoint levels of the uncertifiable-point search: the residual maximum of
// a triangle is located to 2^-6 of its extent before the full subdivision.
constexpr int kWitnessLevels = 6;

// The three quotients of the axis are each rounded once, which turns a unit
// vector by at most sqrt(3) 2^-53 radians; a common error of the length does
// not turn it.
constexpr double kAxisRounding = 0x1p-51;
void bounding_box(const Point& a, const Point& b, const Point& c, Point& lower, Point& upper) {
  for (int d = 0; d < 3; ++d) {
    lower[d] = least(a[d], b[d], c[d]);
    upper[d] = greatest(a[d], b[d], c[d]);
  }
}
bool gradient_over(const IntervalBoxProgram& program, const Point& lower, const Point& upper,
                   std::array<Interval, 3>& gradient) {
  if (!program.evaluate(lower, upper, gradient.data())) return false;
  return finite(gradient[0]) && finite(gradient[1]) && finite(gradient[2]);
}

// Translation preserves these edge vectors exactly. Anchor enclosures may
// therefore describe either the original triangle or an outward-enclosed
// translate without rounding its geometry into a different triangle.
bool tighten_triangle_field(const Point& a, const Point& b, const Point& c,
                            const std::array<VertexEvidence, 3>& evidence,
                            const std::array<Interval, 3>& gradient, Interval& field) {
  const std::array<const Point*, 3> corners{&a, &b, &c};
  for (size_t k = 0; k < 3; ++k) {
    Interval form{evidence[k].lower, evidence[k].upper};
    if (!finite(form)) continue;
    const Point& p = *corners[k];
    const Point& q = *corners[(k + 1) % 3];
    const Point& r = *corners[(k + 2) % 3];
    Interval along_q{0, 0}, along_r{0, 0};
    for (int d = 0; d < 3; ++d) {
      const Interval& g = gradient[static_cast<size_t>(d)];
      along_q = add(along_q, multiply(g, subtract(point(q[d]), point(p[d]))));
      along_r = add(along_r, multiply(g, subtract(point(r[d]), point(p[d]))));
    }
    form = add(form, {least(0.0, along_q.lower, along_r.lower),
                      greatest(0.0, along_q.upper, along_r.upper)});
    if (!finite(form)) continue;
    field = {std::max(field.lower, form.lower), std::min(field.upper, form.upper)};
    if (field.lower > field.upper) return false;
  }
  return true;
}
}  // namespace

int32_t IntervalBoxProgram::bind(const int64_t* instructions, int64_t instruction_count,
                                 const int64_t* indices, int64_t index_count,
                                 const double* parameters, int64_t parameter_count,
                                 const double* workspace_lower, const double* workspace_upper,
                                 int64_t workspace_size, const int64_t* outputs,
                                 int64_t output_count, const int64_t* guards,
                                 int64_t guard_count) {
  program_.reset();
  if (instruction_count < 0 || index_count < 0 || parameter_count < 0 || workspace_size < 3 ||
      output_count < 1 || guard_count < 0 || (instruction_count > 0 && instructions == nullptr) ||
      (index_count > 0 && indices == nullptr) || (parameter_count > 0 && parameters == nullptr) ||
      (guard_count > 0 && guards == nullptr) || workspace_lower == nullptr ||
      workspace_upper == nullptr || outputs == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  if (!phx::mc::addressable(instruction_count, 8, sizeof(int64_t)) ||
      !phx::mc::addressable(workspace_size, 1, sizeof(double)))
    return PHX_MC_INVALID_ARGUMENT;
  for (int64_t k = 0; k < output_count; ++k)
    if (outputs[k] < 3 || outputs[k] >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t k = 0; k < guard_count; ++k)
    if (guards[k] < 0 || guards[k] >= workspace_size) return PHX_MC_INVALID_ARGUMENT;
  const int32_t status = program_.bind(instructions, instruction_count, indices, index_count,
                                       parameters, parameter_count, workspace_size);
  if (status != PHX_MC_OK) return status;
  template_lower_.assign(workspace_lower, workspace_lower + workspace_size);
  template_upper_.assign(workspace_upper, workspace_upper + workspace_size);
  outputs_.assign(outputs, outputs + output_count);
  guards_.assign(guards, guards + guard_count);
  lower_.resize(static_cast<size_t>(workspace_size));
  upper_.resize(static_cast<size_t>(workspace_size));
  // One dry execution over the origin validates the whole program once.
  lower_ = template_lower_;
  upper_ = template_upper_;
  std::fill(lower_.begin(), lower_.begin() + 3, 0.0);
  std::fill(upper_.begin(), upper_.begin() + 3, 0.0);
  return program_.execute(lower_.data(), upper_.data(), workspace_size);
}

bool IntervalBoxProgram::evaluate(const Point& lower, const Point& upper,
                                  Interval* result) const {
  for (int d = 0; d < 3; ++d)
    if (!(lower[d] <= upper[d]) || !std::isfinite(lower[d]) || !std::isfinite(upper[d]))
      return false;
  std::copy(template_lower_.begin(), template_lower_.end(), lower_.begin());
  std::copy(template_upper_.begin(), template_upper_.end(), upper_.begin());
  for (int d = 0; d < 3; ++d) {
    lower_[static_cast<size_t>(d)] = lower[d];
    upper_[static_cast<size_t>(d)] = upper[d];
  }
  const int32_t status = program_.execute(lower_.data(), upper_.data(),
                                          static_cast<int64_t>(lower_.size()));
  if (status != PHX_MC_OK) return false;
  // A real-power base outside its nonnegative domain leaves f undefined there.
  for (int64_t guard : guards_)
    if (lower_[static_cast<size_t>(guard)] < 0.0) return false;
  for (size_t k = 0; k < outputs_.size(); ++k) {
    const size_t at = static_cast<size_t>(outputs_[k]);
    result[k] = {lower_[at], upper_[at]};
    if (std::isnan(result[k].lower) || std::isnan(result[k].upper)) return false;
  }
  return true;
}

VertexEvidence ZeroSetBandAdmission::evaluate(const Point& position) const {
  Interval value{};
  if (!value_.evaluate(position, position, &value) || !finite(value))
    return {-kInfinity, kInfinity};
  return {value.lower, value.upper};
}

double ZeroSetBandAdmission::piece_distance(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient)) return kInfinity;
  Interval field{};
  if (!value_.evaluate(lower, upper, &field) || !finite(field)) return kInfinity;
  // Mean-value forms anchored at each corner p with edges e1, e2 to the other
  // corners: x = p + s e1 + t e2 (s, t >= 0, s + t <= 1) and f(x) - f(p) =
  // s (g . e1) + t (g . e2) for some g = grad f(xi), xi on [p, x] in bbox(T).
  // Over the simplex this lies in the hull of 0, G . e1 and G . e2; tangent
  // edges make it second order in the triangle size.
  if (!tighten_triangle_field(a, b, c, evidence, gradient, field)) return kInfinity;
  const double residual = magnitude(field);
  if (residual == 0) return 0;
  const double bbox_norm = norm_lower(gradient);
  if (!(bbox_norm > 0)) return kInfinity;
  const double estimate = next_up(residual / bbox_norm);
  double radii[2] = {std::min(tolerance_, next_up(2 * estimate)), tolerance_};
  for (const double radius : radii) {
    if (!(radius > 0) || !std::isfinite(radius)) continue;
    Point wide_lower{}, wide_upper{};
    for (int d = 0; d < 3; ++d) {
      wide_lower[d] = next_down(lower[d] - radius);
      wide_upper[d] = next_up(upper[d] + radius);
    }
    std::array<Interval, 3> wide{};
    if (!gradient_over(gradient_, wide_lower, wide_upper, wide)) continue;
    const double norm = norm_lower(wide);
    if (!(norm > 0)) continue;
    const double bound = next_up(residual / norm);
    if (bound <= radius) return bound;
  }
  return kInfinity;
}

bool ZeroSetBandAdmission::translated_field(const Point& a, const Point& b, const Point& c,
                                           const Point& displacement, double sign,
                                           Interval& field) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  for (int d = 0; d < 3; ++d) {
    const double shift = sign * displacement[d];
    lower[d] = next_down(lower[d] + shift);
    upper[d] = next_up(upper[d] + shift);
  }
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient) ||
      !value_.evaluate(lower, upper, &field) || !finite(field))
    return false;
  std::array<VertexEvidence, 3> evidence{};
  const std::array<const Point*, 3> corners{&a, &b, &c};
  for (size_t k = 0; k < corners.size(); ++k) {
    for (int d = 0; d < 3; ++d) {
      const Interval shifted = add(point((*corners[k])[d]), point(sign * displacement[d]));
      lower[d] = shifted.lower;
      upper[d] = shifted.upper;
    }
    Interval value{};
    if (!value_.evaluate(lower, upper, &value) || !finite(value)) return false;
    evidence[k] = {value.lower, value.upper};
  }
  return tighten_triangle_field(a, b, c, evidence, gradient, field);
}

double ZeroSetBandAdmission::extrusion_distance(const Point& a, const Point& b,
                                                const Point& c, double radius) const {
  if (!(radius > 0) || !std::isfinite(radius)) return kInfinity;
  Point lower{}, upper{}, center{}, direction{}, displacement{};
  bounding_box(a, b, c, lower, upper);
  for (int d = 0; d < 3; ++d) center[d] = 0.5 * lower[d] + 0.5 * upper[d];
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, center, center, gradient)) return kInfinity;
  double scale = 0;
  for (int d = 0; d < 3; ++d) {
    direction[d] = 0.5 * gradient[d].lower + 0.5 * gradient[d].upper;
    scale = std::max(scale, std::fabs(direction[d]));
  }
  if (!(scale > 0) || !std::isfinite(scale)) return kInfinity;
  double squared = 0;
  for (double& component : direction) {
    component /= scale;
    squared = next_up(squared + next_up(component * component));
  }
  const double norm = next_up(std::sqrt(squared));
  const double length = next_down(radius / norm);
  if (!(length > 0)) return kInfinity;
  for (int d = 0; d < 3; ++d) {
    const double component = length * direction[d];
    // Each magnitude is no larger than the exact product; length is no
    // larger than radius / norm, so the real displacement norm is <= radius.
    displacement[d] =
        std::copysign(std::max(0.0, next_down(std::fabs(component))), component);
    lower[d] = next_down(lower[d] - std::fabs(displacement[d]));
    upper[d] = next_up(upper[d] + std::fabs(displacement[d]));
  }
  Interval slab_value{};
  if (!value_.evaluate(lower, upper, &slab_value) || !finite(slab_value) ||
      !gradient_over(gradient_, lower, upper, gradient))
    return kInfinity;
  Interval along{0, 0};
  for (int d = 0; d < 3; ++d)
    along = add(along, multiply(gradient[d], point(displacement[d])));
  if (!finite(along) || !(along.lower > 0 || along.upper < 0)) return kInfinity;
  Interval minus{}, plus{};
  if (!translated_field(a, b, c, displacement, -1.0, minus) ||
      !translated_field(a, b, c, displacement, 1.0, plus))
    return kInfinity;
  // Uniform signs on both complete endpoint triangles and a strict derivative
  // sign on the entire extrusion isolate one zero along every point's segment.
  if ((along.lower > 0 && minus.upper <= 0 && plus.lower >= 0) ||
      (along.upper < 0 && minus.lower >= 0 && plus.upper <= 0))
    return radius;
  return kInfinity;
}

// Pieces whose bound misses the tolerance split into four at edge midpoints
// until each piece is decided: certified within the tolerance, refuted (no
// point of it lies within the tolerance), or at binary64 resolution, which
// refuses. Binary64 resolution is relative to the triangle: a piece whose
// corners all lie within one slack (two coordinate ulps at the triangle's
// scale) of each other cannot be split into representable distinct pieces at
// that scale. An absolute test alone (a rounded midpoint equal to an endpoint)
// would run to subnormal spacing near a zero coordinate, about 1074 levels.
// Each level halves the piece's extent, so the depth is at most about 53, and
// the pieces are kept on an explicit stack in any case. Rounded midpoints leave
// every point of T within one coordinate ulp per level of the union of the
// pieces, so a piece at depth k adds k such slacks.
AdmissionMemo::Entry& AdmissionMemo::slot(const Key& key) {
  // SplitMix64 over the coordinate bit patterns.
  std::uint64_t hash = 0x9e3779b97f4a7c15ull;
  for (const double value : key) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hash += bits + 0x9e3779b97f4a7c15ull;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ull;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebull;
    hash ^= hash >> 31;
  }
  Entry& entry = entries_[static_cast<size_t>(hash) & (kEntries - 1)];
  if (!entry.used || std::memcmp(entry.key.data(), key.data(), sizeof(Key)) != 0) {
    entry = Entry{};
    entry.key = key;
    entry.used = true;
  }
  return entry;
}

AdmissionMemo::Entry* ZeroSetBandAdmission::memo_entry(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  if (memo_ == nullptr) return nullptr;
  AdmissionMemo::Key key{};
  size_t at = 0;
  for (const Point* p : {&a, &b, &c})
    for (const double coordinate : *p) key[at++] = coordinate;
  for (const VertexEvidence& value : evidence) {
    key[at++] = value.lower;
    key[at++] = value.upper;
  }
  return &memo_->slot(key);
}

double ZeroSetBandAdmission::distance_bound(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  AdmissionMemo::Entry* entry = memo_entry(a, b, c, evidence);
  if (entry != nullptr && entry->has_bound) {
    if (entry->has_sample) distance_barycentric_ = entry->sample;
    else distance_barycentric_.reset();
    return entry->bound;
  }
  distance_barycentric_.reset();
  double scale = 0;
  for (const Point* p : {&a, &b, &c})
    for (double coordinate : *p) scale = std::max(scale, std::fabs(coordinate));
  const double slack = next_up(2 * std::numeric_limits<double>::epsilon() * scale);
  const double bound = split_distance(a, b, c, evidence, slack);
  if (entry != nullptr) {
    entry->bound = bound;
    entry->has_bound = true;
    entry->has_sample = distance_barycentric_.has_value();
    if (entry->has_sample) entry->sample = *distance_barycentric_;
  }
  return bound;
}

// No point of the piece lies within the tolerance of Z: otherwise a zero of f
// lies in the tolerance-widened box B within the tolerance of that point, so
// |f| <= max |grad f| over B times the tolerance there.
bool ZeroSetBandAdmission::piece_refuted(const Point& a, const Point& b,
                                         const Point& c) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  Interval field{};
  if (!value_.evaluate(lower, upper, &field) || !finite(field)) return false;
  Point wide_lower{}, wide_upper{};
  for (int d = 0; d < 3; ++d) {
    wide_lower[d] = next_down(lower[d] - tolerance_);
    wide_upper[d] = next_up(upper[d] + tolerance_);
  }
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, wide_lower, wide_upper, gradient)) return false;
  double sum = 0;
  for (const Interval& component : gradient) {
    const double m = magnitude(component);
    sum = next_up(sum + next_up(m * m));
  }
  const double lipschitz = next_up(std::sqrt(sum));
  return mignitude(field) > next_up(lipschitz * tolerance_);
}

// The piece proofs applied to the single point p, at depth zero. Every
// enclosure over a piece containing p contains p's own, and a deeper piece
// only adds its rounding charge, so a point these proofs cannot certify lies
// in no certifiable piece: the subdivision of a triangle through it ends in
// a refusal. Refusing early is sound regardless (the admission is incomplete
// by contract); the proofs of an admitted triangle are unchanged.
bool ZeroSetBandAdmission::point_uncertifiable(const Point& p,
                                               const VertexEvidence& evidence) const {
  const std::array<VertexEvidence, 3> corners{evidence, evidence, evidence};
  if (piece_distance(p, p, p, corners) <= tolerance_) return false;
  const double radius = next_down(next_down(tolerance_) - next_up(0.0));
  return !(extrusion_distance(p, p, p, radius) <= tolerance_);
}

// Follows the largest corner residual through `kWitnessLevels` midpoint
// subdivisions of the triangle (a proposal: ties take the subdivision's own
// piece order) and tests the best point found. Returns its barycentric
// coordinates in the triangle when no piece proof can certify it.
std::optional<Point> ZeroSetBandAdmission::uncertifiable_point(
    const Point& a, const Point& b, const Point& c,
    const std::array<VertexEvidence, 3>& evidence) const {
  struct Corner {
    Point position;
    VertexEvidence evidence;
    Point barycentric;
    double residual;
  };
  const auto residual = [](const VertexEvidence& value) {
    if (value.lower > 0) return value.lower;
    if (value.upper < 0) return -value.upper;
    return 0.0;
  };
  const auto middle = [](const Point& p, const Point& q) {
    return Point{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
  };
  std::array<Corner, 3> piece{{{a, evidence[0], {1, 0, 0}, residual(evidence[0])},
                               {b, evidence[1], {0, 1, 0}, residual(evidence[1])},
                               {c, evidence[2], {0, 0, 1}, residual(evidence[2])}}};
  Corner best = piece[0];
  for (const Corner& corner : piece)
    if (corner.residual > best.residual) best = corner;
  for (int level = 0; level < kWitnessLevels; ++level) {
    std::array<Corner, 3> mid{};
    for (int k = 0; k < 3; ++k) {
      const Corner& p = piece[static_cast<size_t>(k)];
      const Corner& q = piece[static_cast<size_t>((k + 1) % 3)];
      Corner& m = mid[static_cast<size_t>(k)];  // midpoint of the edge after corner k
      m.position = middle(p.position, q.position);
      if (m.position == p.position || m.position == q.position) return std::nullopt;
      m.evidence = evaluate(m.position);
      m.barycentric = middle(p.barycentric, q.barycentric);
      m.residual = residual(m.evidence);
      if (m.residual > best.residual) best = m;
    }
    // The four sub-pieces in the subdivision's order; the first of the
    // largest corner residual is followed.
    const std::array<std::array<Corner, 3>, 4> children{{{piece[0], mid[0], mid[2]},
                                                         {mid[0], piece[1], mid[1]},
                                                         {mid[2], mid[1], piece[2]},
                                                         {mid[1], mid[2], mid[0]}}};
    size_t selected = 0;
    double selected_residual = -1;
    for (size_t child = 0; child < children.size(); ++child) {
      double largest = 0;
      for (const Corner& corner : children[child]) largest = std::max(largest, corner.residual);
      if (largest > selected_residual) {
        selected_residual = largest;
        selected = child;
      }
    }
    piece = children[selected];
  }
  if (!point_uncertifiable(best.position, best.evidence)) return std::nullopt;
  return best.barycentric;
}

// Depth-first over the pieces, in the order of the four sub-pieces; the first
// piece that is neither certified nor splittable refuses the whole triangle.
// An undecided root reports its own bound, a refused sub-piece +inf.
double ZeroSetBandAdmission::split_distance(const Point& a, const Point& b, const Point& c,
                                            const std::array<VertexEvidence, 3>& evidence,
                                            double slack) const {
  struct Piece {
    Point a, b, c;
    std::array<VertexEvidence, 3> evidence;
    int depth;
    std::array<Point, 3> barycentric;
  };
  const auto middle = [](const Point& p, const Point& q) {
    return Point{0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])};
  };
  constexpr std::array<Point, 3> reference{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  const auto record_sample = [&](const Piece& piece) {
    Point sample{};
    for (int d = 0; d < 3; ++d)
      sample[d] = (piece.barycentric[0][d] + piece.barycentric[1][d] +
                   piece.barycentric[2][d]) / 3;
    distance_barycentric_ = sample;
  };
  std::vector<Piece> stack{{a, b, c, evidence, 0, reference}};
  double worst = 0;
  while (!stack.empty()) {
    const Piece piece = stack.back();
    stack.pop_back();
    const double own = piece_distance(piece.a, piece.b, piece.c, piece.evidence);
    const double bound =
        std::isfinite(own) ? next_up(own + next_up(piece.depth * slack)) : kInfinity;
    if (bound <= tolerance_) {
      worst = std::max(worst, bound);
      continue;
    }
    if (piece_refuted(piece.a, piece.b, piece.c)) {
      record_sample(piece);
      return piece.depth == 0 ? bound : kInfinity;
    }
    const double charge = next_up(piece.depth * slack);
    // Leave room for the final outward addition, without a heuristic tolerance
    // reduction. Existing subdivision-rounding evidence remains authoritative.
    const double radius = next_down(next_down(tolerance_) - charge);
    const double translated = extrusion_distance(piece.a, piece.b, piece.c, radius);
    const double translated_bound =
        std::isfinite(translated) ? next_up(translated + charge) : kInfinity;
    if (translated_bound <= tolerance_) {
      worst = std::max(worst, translated_bound);
      continue;
    }
    // The first undecided piece is the triangle itself: look for a point no
    // piece can certify before subdividing. A deep undecided piece has
    // shrunk onto such a point's neighborhood; its largest corner is tested.
    if (piece.depth == 0) {
      if (const auto witness = uncertifiable_point(piece.a, piece.b, piece.c, piece.evidence)) {
        distance_barycentric_ = *witness;
        return kInfinity;
      }
    } else if (piece.depth >= kWitnessLevels) {
      size_t corner = 0;
      double largest = -1;
      const std::array<const Point*, 3> corners{&piece.a, &piece.b, &piece.c};
      for (size_t k = 0; k < 3; ++k) {
        const VertexEvidence& value = piece.evidence[k];
        const double residual =
            value.lower > 0 ? value.lower : (value.upper < 0 ? -value.upper : 0.0);
        if (residual > largest) {
          largest = residual;
          corner = k;
        }
      }
      if (point_uncertifiable(*corners[corner], piece.evidence[corner])) {
        distance_barycentric_ = piece.barycentric[corner];
        return kInfinity;
      }
    }
    const Point ab = middle(piece.a, piece.b), bc = middle(piece.b, piece.c),
                ca = middle(piece.c, piece.a);
    if (ab == piece.a || ab == piece.b || bc == piece.b || bc == piece.c || ca == piece.c ||
        ca == piece.a) {
      record_sample(piece);
      return kInfinity;
    }
    double extent = 0;
    for (int d = 0; d < 3; ++d) {
      extent = greatest(extent, std::fabs(piece.a[d] - piece.b[d]),
                        std::fabs(piece.b[d] - piece.c[d]), std::fabs(piece.c[d] - piece.a[d]));
    }
    if (extent <= slack) {
      record_sample(piece);
      return kInfinity;
    }
    const VertexEvidence eab = evaluate(ab), ebc = evaluate(bc), eca = evaluate(ca);
    const auto& e = piece.evidence;
    const int depth = piece.depth + 1;
    const auto& weights = piece.barycentric;
    const Point wab = middle(weights[0], weights[1]), wbc = middle(weights[1], weights[2]),
                wca = middle(weights[2], weights[0]);
    // Pushed in reverse so the first sub-piece is decided first.
    stack.push_back({bc, ca, ab, {ebc, eca, eab}, depth, {wbc, wca, wab}});
    stack.push_back({ca, bc, piece.c, {eca, ebc, e[2]}, depth, {wca, wbc, weights[2]}});
    stack.push_back({ab, piece.b, bc, {eab, e[1], ebc}, depth, {wab, weights[1], wbc}});
    stack.push_back({piece.a, ab, ca, {e[0], eab, eca}, depth, {weights[0], wab, wca}});
  }
  return worst;
}

int ZeroSetBandAdmission::transversal_sign(const Point& a, const Point& b,
                                           const Point& c) const {
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient)) return 0;
  std::array<Interval, 3> first{}, second{};
  for (int d = 0; d < 3; ++d) {
    first[static_cast<size_t>(d)] = subtract(point(b[d]), point(a[d]));
    second[static_cast<size_t>(d)] = subtract(point(c[d]), point(a[d]));
  }
  Interval product{0, 0};
  for (int d = 0; d < 3; ++d) {
    const size_t i = static_cast<size_t>((d + 1) % 3), j = static_cast<size_t>((d + 2) % 3);
    const Interval normal =
        subtract(multiply(first[i], second[j]), multiply(first[j], second[i]));
    product = add(product, multiply(normal, gradient[static_cast<size_t>(d)]));
  }
  if (!finite(product)) return 0;
  if (product.lower > 0) return 1;
  if (product.upper < 0) return -1;
  return 0;
}

bool ZeroSetBandAdmission::admits(const Point& a, const Point& b, const Point& c,
                                  const std::array<VertexEvidence, 3>& evidence) const {
  AdmissionMemo::Entry* entry = memo_entry(a, b, c, evidence);
  int sign = 0;
  if (entry != nullptr && entry->has_sign) {
    sign = entry->sign;
  } else {
    sign = transversal_sign(a, b, c);
    if (entry != nullptr) {
      entry->sign = static_cast<std::int8_t>(sign);
      entry->has_sign = true;
    }
  }
  if (sign != orientation_) {
    distance_barycentric_.reset();
    ++refusals_.transversality;
    return false;
  }
  if (!(distance_bound(a, b, c, evidence) <= tolerance_)) {
    ++refusals_.distance;
    return false;
  }
  return true;
}

// The source normals near the triangle are orientation * grad f / |grad f| at
// zero-set points within the tolerance of it, so over bbox(T) widened by the
// tolerance. With the gradient enclosure's midpoint g and radius vector r
// (|r| < |g|), every enclosed direction lies within asin(|r| / |g|) of g. An
// admitted triangle's points lie within the tolerance of Z, so across its
// smallest altitude it tilts from the local tangent plane by up to
// asin(2 tolerance / altitude), which widens the cone. Both terms are upper
// bounds: every norm, quotient and arcsine is rounded toward the wider cone
// with binary64 operations and the elementary interval kernel, so the
// half-angle is the same double on every platform and never too small. An
// undecided or zero-straddling enclosure constrains nothing (a crease
// enclosure holds both sides' gradients, so the crease is kept).
NormalCone ZeroSetBandAdmission::normal_cone(const Point& a, const Point& b, const Point& c,
                                             const std::array<VertexEvidence, 3>& evidence) const {
  NormalCone cone;
  const double tilt = normal_cone_floor(a, b, c, evidence);
  if (!(tilt < kOpenCone)) return cone;
  Point lower{}, upper{};
  bounding_box(a, b, c, lower, upper);
  for (int d = 0; d < 3; ++d) {
    lower[d] = next_down(lower[d] - tolerance_);
    upper[d] = next_up(upper[d] + tolerance_);
  }
  std::array<Interval, 3> gradient{};
  if (!gradient_over(gradient_, lower, upper, gradient)) return cone;
  Point middle{}, spread{};
  for (int d = 0; d < 3; ++d) {
    const Interval& g = gradient[static_cast<size_t>(d)];
    middle[d] = g.lower + 0.5 * (g.upper - g.lower);
    spread[d] = next_up(std::max(g.upper - middle[d], middle[d] - g.lower));
  }
  // |r| from above and |g| from below, their quotient and its arcsine rounded
  // up: the half-angle holds every enclosed direction about the midpoint.
  const double least_parts[3] = {std::fabs(middle[0]), std::fabs(middle[1]),
                                 std::fabs(middle[2])};
  const double middle_norm = geometry::norm_lower(least_parts);
  const double spread_norm = geometry::norm_upper(spread.data());
  if (!(spread_norm < middle_norm) || !std::isfinite(middle_norm)) return cone;
  const double half =
      next_up(next_up(asin_upper(next_up(spread_norm / middle_norm)) + tilt) + kAxisRounding);
  if (!(half < cone.half_angle)) return cone;
  // The axis is the midpoint's direction with its length rounded to nearest in
  // a fixed order; kAxisRounding above covers how far that rounding turns it.
  const double length =
      std::sqrt((middle[0] * middle[0] + middle[1] * middle[1]) + middle[2] * middle[2]);
  if (!(length > 0) || !std::isfinite(length)) return cone;
  for (int d = 0; d < 3; ++d) cone.axis[d] = orientation_ * middle[d] / length;
  cone.half_angle = half;
  return cone;
}

// asin(2 tolerance / altitude) from above. The smallest altitude is twice the
// area over the longest edge: the edges are enclosed componentwise, the area
// is bounded from below and the longest edge from above, and every product
// and quotient on the way is rounded up.
double ZeroSetBandAdmission::normal_cone_floor(const Point& a, const Point& b, const Point& c,
                                               const std::array<VertexEvidence, 3>&) const {
  const IntervalVector u = enclosed_difference(b, a);
  const IntervalVector w = enclosed_difference(c, a);
  const IntervalVector e = enclosed_difference(c, b);
  const double twice_area = norm_lower(cross(u, w));
  const double longest = greatest(norm_upper(u), norm_upper(w), norm_upper(e));
  if (!(twice_area > 0) || !(longest > 0)) return kOpenCone;
  // 2 tolerance is exact.
  const double tilt = next_up(next_up(2 * tolerance_ * longest) / twice_area);
  return tilt < 1 ? asin_upper(tilt) : kOpenCone;
}

Point ZeroSetBandAdmission::project(const Point& position) const {
  std::array<Interval, 3> gradient{};
  Interval value{};
  if (!gradient_over(gradient_, position, position, gradient) ||
      !value_.evaluate(position, position, &value) || !finite(value))
    return position;
  Point direction{};
  double norm = 0;
  for (int d = 0; d < 3; ++d) {
    direction[d] = 0.5 * (gradient[static_cast<size_t>(d)].lower +
                          gradient[static_cast<size_t>(d)].upper);
    norm += direction[d] * direction[d];
  }
  norm = std::sqrt(norm);
  if (!(norm > 0) || !std::isfinite(norm)) return position;
  for (double& component : direction) component /= norm;
  const double start = 0.5 * (value.lower + value.upper);
  if (start == 0) return position;
  // phi(t) = f(position - t u); its root lies near t = f / |grad f|.
  const double reach = 2 * std::fabs(start) / norm;
  Interval step{-reach, reach};
  for (int iteration = 0; iteration < kNewtonSteps; ++iteration) {
    const double middle = 0.5 * (step.lower + step.upper);
    Point at{}, lower{}, upper{};
    for (int d = 0; d < 3; ++d) {
      at[d] = position[d] - middle * direction[d];
      const double first = position[d] - step.lower * direction[d];
      const double second = position[d] - step.upper * direction[d];
      lower[d] = next_down(std::min(first, second));
      upper[d] = next_up(std::max(first, second));
    }
    Interval at_value{};
    std::array<Interval, 3> slope_gradient{};
    if (!value_.evaluate(at, at, &at_value) || !finite(at_value) ||
        !gradient_over(gradient_, lower, upper, slope_gradient))
      return position;
    Interval slope{0, 0};
    for (int d = 0; d < 3; ++d)
      slope = subtract(slope, multiply(slope_gradient[static_cast<size_t>(d)],
                                       point(direction[d])));
    if (!(slope.lower > 0) && !(slope.upper < 0)) break;
    const double q[4] = {at_value.lower / slope.lower, at_value.lower / slope.upper,
                         at_value.upper / slope.lower, at_value.upper / slope.upper};
    const Interval newton{next_down(middle - next_up(greatest(q[0], q[1], q[2], q[3]))),
                          next_up(middle - next_down(least(q[0], q[1], q[2], q[3])))};
    const Interval narrowed{std::max(step.lower, newton.lower),
                            std::min(step.upper, newton.upper)};
    if (narrowed.lower > narrowed.upper) return position;  // no root on this segment
    const bool settled = narrowed.lower == step.lower && narrowed.upper == step.upper;
    step = narrowed;
    if (settled) break;
  }
  const double t = 0.5 * (step.lower + step.upper);
  Point projected{};
  for (int d = 0; d < 3; ++d) projected[d] = position[d] - t * direction[d];
  return projected;
}

namespace {
struct ZeroSetRemeshResult {
  std::vector<Point> points;
  std::vector<Triangle> triangles;
};


// The implicit source as a floating-point oracle: second-order jets of f from
// the point executor (value, gradient, Hessian), Newton projection along the
// gradient, and principal curvatures of the level set. Proposals only.
class ImplicitOracle final : public SurfaceOracle {
 public:
  int32_t bind(const phx_mc_interval_box_program& value) {
    if (value.output_count != 1 || value.workspace_size < 3 || value.workspace_lower == nullptr ||
        value.workspace_upper == nullptr)
      return PHX_MC_INVALID_ARGUMENT;
    const int32_t status = phx::mc::validate_point_program(
        value.instructions, value.instruction_count, value.indices, value.index_count,
        value.parameters, value.parameter_count, value.workspace_size, 3, value.outputs, 1);
    if (status != PHX_MC_OK) return status;
    // Point seeds: the midpoint of each constant's enclosure.
    seeds_.resize(static_cast<size_t>(value.workspace_size));
    for (size_t e = 0; e < seeds_.size(); ++e) {
      const double lower = value.workspace_lower[e], upper = value.workspace_upper[e];
      seeds_[e] = std::isfinite(lower) && std::isfinite(upper) ? lower + (upper - lower) / 2
                  : std::isfinite(lower)                       ? lower
                  : std::isfinite(upper)                       ? upper
                                                               : 0.0;
    }
    view_ = phx::mc::ProgramView{value.instructions, value.instruction_count, value.indices,
                                 seeds_.data(),      value.workspace_size,    3,
                                 value.outputs,      1};
    return PHX_MC_OK;
  }
  void set_orientation(int orientation) { orientation_ = orientation; }
  void set_reach(double reach) { reach_ = reach; }
  // f and grad f at x; false when not finite.
  bool gradient(const Point& x, double& f, Point& g) const {
    if (!jet(x)) return false;
    f = jet_[0];
    g = {jet_[1], jet_[2], jet_[3]};
    return true;
  }
  bool project(const Point& position, SurfacePoint& out) const override {
    Point x = position;
    for (int iteration = 0; iteration < kProjectionSteps; ++iteration) {
      if (!jet(x)) return false;
      switch (newton(jet_.data(), x)) {
        case Step::kFailed:
          return false;
        case Step::kConverged:
          return surface_point(jet_.data(), position, x, out);
        case Step::kMoved:
          break;
      }
    }
    return false;
  }
  // The same iterations as project(), every point of a round evaluated in one
  // block of jets.
  void project_all(const std::vector<Point>& positions, std::vector<SurfacePoint>& out,
                   std::vector<uint8_t>& found) const override {
    const size_t n = positions.size();
    out.assign(n, SurfacePoint{});
    found.assign(n, 0);
    std::vector<Point> current = positions;
    std::vector<size_t> active(n);
    for (size_t i = 0; i < n; ++i) active[i] = i;
    std::vector<Point> batch;
    std::vector<double> jets;
    for (int iteration = 0; iteration < kProjectionSteps && !active.empty(); ++iteration) {
      batch.resize(active.size());
      for (size_t k = 0; k < active.size(); ++k) batch[k] = current[active[k]];
      jets.resize(10 * batch.size());
      phx::mc::evaluate_jet(view_, batch.data()->data(), static_cast<int64_t>(batch.size()),
                            jets.data(), workspace_);
      size_t kept = 0;
      for (size_t k = 0; k < active.size(); ++k) {
        const size_t i = active[k];
        const double* row = jets.data() + 10 * k;
        if (!finite_jet(row)) continue;
        switch (newton(row, current[i])) {
          case Step::kFailed:
            break;
          case Step::kConverged:
            found[i] = surface_point(row, positions[i], current[i], out[i]) ? 1 : 0;
            break;
          case Step::kMoved:
            active[kept++] = i;
            break;
        }
      }
      active.resize(kept);
    }
  }
  void distances(const std::vector<Point>& positions, std::vector<double>& out) const override {
    std::vector<double> jets(10 * positions.size());
    if (!positions.empty())
      phx::mc::evaluate_jet(view_, positions.data()->data(),
                            static_cast<int64_t>(positions.size()), jets.data(), workspace_);
    out.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
      const double* row = jets.data() + 10 * i;
      const double g2 = row[1] * row[1] + row[2] * row[2] + row[3] * row[3];
      out[i] = finite_jet(row) && g2 > 0 ? std::abs(row[0]) / std::sqrt(g2) : kInfinity;
    }
  }

 private:
  static constexpr int kProjectionSteps = 8;
  enum class Step { kFailed, kConverged, kMoved };
  static bool finite_jet(const double* row) {
    for (int c = 0; c < 10; ++c)
      if (!std::isfinite(row[c])) return false;
    return true;
  }
  // One Newton step along the gradient from the jet at x. A step below 2^-36
  // of the coordinate scale (far below any tolerance this places points for)
  // is not taken, so the jet at x is the jet at the returned point.
  static Step newton(const double* row, Point& x) {
    const double g2 = row[1] * row[1] + row[2] * row[2] + row[3] * row[3];
    if (!(g2 > 0)) return Step::kFailed;
    const double t = row[0] / g2;
    if (std::abs(t) * std::sqrt(g2) <=
        0x1p-36 * (1 + std::abs(x[0]) + std::abs(x[1]) + std::abs(x[2])))
      return Step::kConverged;
    for (int d = 0; d < 3; ++d) x[d] -= t * row[1 + d];
    return Step::kMoved;
  }
  // The surface point at x from the jet there: oriented unit normal and the
  // principal curvatures of the level set (the shape operator P H P / |g|, P
  // the tangent projector, from its trace and squared Frobenius norm).
  bool surface_point(const double* row, const Point& position, const Point& x,
                     SurfacePoint& out) const {
    double moved = 0;
    for (int d = 0; d < 3; ++d) moved += (x[d] - position[d]) * (x[d] - position[d]);
    if (!(std::sqrt(moved) <= reach_)) return false;
    const double g2 = row[1] * row[1] + row[2] * row[2] + row[3] * row[3];
    const double length = std::sqrt(g2);
    const Point n{row[1] / length, row[2] / length, row[3] / length};
    // Hessian upper triangle: xx, xy, xz, yy, yz, zz.
    const double hxx = row[4], hxy = row[5], hxz = row[6], hyy = row[7], hyz = row[8],
                 hzz = row[9];
    const Point hn{hxx * n[0] + hxy * n[1] + hxz * n[2], hxy * n[0] + hyy * n[1] + hyz * n[2],
                   hxz * n[0] + hyz * n[1] + hzz * n[2]};
    const double nhn = n[0] * hn[0] + n[1] * hn[1] + n[2] * hn[2];
    const double frobenius = hxx * hxx + hyy * hyy + hzz * hzz +
                             2 * (hxy * hxy + hxz * hxz + hyz * hyz);
    const double trace = (hxx + hyy + hzz - nhn) / length;
    const double squares =
        (frobenius - 2 * (hn[0] * hn[0] + hn[1] * hn[1] + hn[2] * hn[2]) + nhn * nhn) / g2;
    out.on = x;
    out.normal = {orientation_ * n[0], orientation_ * n[1], orientation_ * n[2]};
    const double spread = std::sqrt(std::max(0.0, 2 * squares - trace * trace));
    out.curvature_first = (trace + spread) / 2;
    out.curvature_second = (trace - spread) / 2;
    return std::isfinite(out.curvature_first) && std::isfinite(out.curvature_second);
  }
  bool jet(const Point& x) const {
    phx::mc::evaluate_jet(view_, x.data(), 1, jet_.data(), workspace_);
    return finite_jet(jet_.data());
  }
  std::vector<double> seeds_;
  phx::mc::ProgramView view_{};
  int orientation_ = 1;
  double reach_ = kInfinity;
  mutable std::array<double, 10> jet_{};
  mutable phx::mc::PointWorkspace workspace_;
};

// Floating-point construction (see phx_mc_zero_set_remesh): no interval
// program is bound and nothing is proved.
int32_t remesh_zero_set(const phx_mc_zero_set_remesh_request& request,
                        ZeroSetRemeshResult& result, int64_t* counters, bool indices_i32) {
  if (request.vertex_count < 3 || request.triangle_count < 1 ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.triangle_count, 3, indices_i32 ? sizeof(int32_t) : sizeof(int64_t)) ||
      request.vertices == nullptr || request.triangles == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  if (!(request.tolerance > 0) || !std::isfinite(request.tolerance) ||
      !(request.carrier_distance >= 0) || !std::isfinite(request.carrier_distance) ||
      !(request.edge_length > 0) || !std::isfinite(request.edge_length) ||
      !(request.minimum_angle_degrees > 0) || !(request.minimum_angle_degrees < 60))
    return PHX_MC_INVALID_ARGUMENT;
  ImplicitOracle oracle;
  int32_t status = oracle.bind(request.value);
  if (status != PHX_MC_OK) return status;
  const auto input_index = [&](size_t offset) -> int64_t {
    if (indices_i32) return reinterpret_cast<const int32_t*>(request.triangles)[offset];
    return request.triangles[offset];
  };
  std::vector<Point> points(static_cast<size_t>(request.vertex_count));
  for (size_t v = 0; v < points.size(); ++v)
    for (int d = 0; d < 3; ++d) {
      const double coordinate = request.vertices[3 * v + static_cast<size_t>(d)];
      if (!std::isfinite(coordinate)) return PHX_MC_NONFINITE_INPUT;
      points[v][static_cast<size_t>(d)] = coordinate;
    }
  std::vector<Triangle> triangles(static_cast<size_t>(request.triangle_count));
  for (size_t t = 0; t < triangles.size(); ++t)
    for (int k = 0; k < 3; ++k) {
      const int64_t v = input_index(3 * t + static_cast<size_t>(k));
      if (v < 0 || v >= request.vertex_count) return PHX_MC_INVALID_INPUT;
      triangles[t][static_cast<size_t>(k)] = v;
    }
  // The carrier's orientation relative to grad f: the majority of its
  // triangles by floating point. Certification checks every triangle of the
  // result against it.
  double agreement = 0;
  for (const Triangle& t : triangles) {
    const Point& a = points[static_cast<size_t>(t[0])];
    const Point& b = points[static_cast<size_t>(t[1])];
    const Point& c = points[static_cast<size_t>(t[2])];
    Point centroid{};
    for (int d = 0; d < 3; ++d) centroid[d] = (a[d] + b[d] + c[d]) / 3;
    double f = 0;
    Point g{};
    if (!oracle.gradient(centroid, f, g)) continue;
    Point n{};
    for (int d = 0; d < 3; ++d) {
      const int i = (d + 1) % 3, j = (d + 2) % 3;
      n[d] = (b[i] - a[i]) * (c[j] - a[j]) - (b[j] - a[j]) * (c[i] - a[i]);
    }
    const double along = n[0] * g[0] + n[1] * g[1] + n[2] * g[2];
    agreement += along > 0 ? 1 : along < 0 ? -1 : 0;
  }
  if (agreement == 0) return PHX_MC_UNRESOLVED;
  const int orientation = agreement > 0 ? 1 : -1;
  oracle.set_orientation(orientation);
  oracle.set_reach(request.edge_length + request.carrier_distance);
  AdaptCounters adapted{};
  status = adapt_surface(points, triangles, oracle,
                         AdaptTargets{request.edge_length, request.tolerance,
                                      request.minimum_angle_degrees},
                         adapted);
  if (status != PHX_MC_OK) return status;
  counters[0] = adapted.splits;
  counters[1] = adapted.collapses;
  counters[2] = adapted.flips;
  counters[3] = adapted.relocations;
  counters[4] = adapted.below_angle;
  counters[5] = orientation;
  counters[6] = adapted.above_tolerance;
  counters[7] = 0;
  result.points = std::move(points);
  result.triangles = std::move(triangles);
  return PHX_MC_OK;
}

// A triangle's corner coordinates in its own cyclic order, rotated to start
// at the lexicographically least corner: equal for the same oriented
// triangle whatever corner it lists first.
using TriangleKey = std::array<Point, 3>;
TriangleKey triangle_key(const Point& a, const Point& b, const Point& c) {
  if (b < a && b < c) return {b, c, a};
  if (c < a && c < b) return {c, a, b};
  return {a, b, c};
}

// Certification of a closed surface against the zero set (see
// phx_mc_zero_set_certify).
int32_t certify_zero_set(const phx_mc_zero_set_certify_request& request) {
  if (request.vertex_count < 3 || request.triangle_count < 1 ||
      !phx::mc::addressable(request.vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.triangle_count, 3, sizeof(int64_t)) ||
      request.vertices == nullptr || request.triangles == nullptr ||
      !(request.tolerance > 0) || !std::isfinite(request.tolerance) ||
      (request.orientation != 1 && request.orientation != -1) ||
      request.carrier_vertex_count < 0 || request.carrier_triangle_count < 0 ||
      !phx::mc::addressable(request.carrier_vertex_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.carrier_triangle_count, 3, sizeof(int64_t)) ||
      (request.carrier_vertex_count > 0 && request.carrier_vertices == nullptr) ||
      (request.carrier_triangle_count > 0 && request.carrier_triangles == nullptr) ||
      !(request.carrier_distance >= 0) || !std::isfinite(request.carrier_distance))
    return PHX_MC_INVALID_ARGUMENT;
  IntervalBoxProgram value, gradient;
  const auto bind = [](IntervalBoxProgram& program, const phx_mc_interval_box_program& source,
                       int64_t outputs) -> int32_t {
    if (source.output_count != outputs) return PHX_MC_INVALID_ARGUMENT;
    return program.bind(source.instructions, source.instruction_count, source.indices,
                        source.index_count, source.parameters, source.parameter_count,
                        source.workspace_lower, source.workspace_upper, source.workspace_size,
                        source.outputs, source.output_count, source.guards,
                        source.guard_count);
  };
  int32_t status = bind(value, request.value, 1);
  if (status != PHX_MC_OK) return status;
  status = bind(gradient, request.gradient, 3);
  if (status != PHX_MC_OK) return status;
  const auto corner = [](const double* vertices, int64_t vertex_count, const int64_t* triangles,
                         size_t row, size_t k, Point& p) {
    const int64_t v = triangles[3 * row + k];
    if (v < 0 || v >= vertex_count) return false;
    for (size_t d = 0; d < 3; ++d) p[d] = vertices[3 * static_cast<size_t>(v) + d];
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
  };
  std::vector<TriangleKey> carrier(static_cast<size_t>(request.carrier_triangle_count));
  for (size_t row = 0; row < carrier.size(); ++row) {
    Point a{}, b{}, c{};
    if (!corner(request.carrier_vertices, request.carrier_vertex_count, request.carrier_triangles,
                row, 0, a) ||
        !corner(request.carrier_vertices, request.carrier_vertex_count, request.carrier_triangles,
                row, 1, b) ||
        !corner(request.carrier_vertices, request.carrier_vertex_count, request.carrier_triangles,
                row, 2, c))
      return PHX_MC_INVALID_INPUT;
    carrier[row] = triangle_key(a, b, c);
  }
  std::sort(carrier.begin(), carrier.end());
  AdmissionMemo memo;
  const ZeroSetBandAdmission admission(value, gradient, request.tolerance, request.orientation,
                                       &memo);
  double maximum = 0;
  int64_t kept = 0;
  for (size_t row = 0; row < static_cast<size_t>(request.triangle_count); ++row) {
    Point a{}, b{}, c{};
    if (!corner(request.vertices, request.vertex_count, request.triangles, row, 0, a) ||
        !corner(request.vertices, request.vertex_count, request.triangles, row, 1, b) ||
        !corner(request.vertices, request.vertex_count, request.triangles, row, 2, c))
      return PHX_MC_INVALID_INPUT;
    if (std::binary_search(carrier.begin(), carrier.end(), triangle_key(a, b, c))) {
      ++kept;
      maximum = std::max(maximum, request.carrier_distance);
      continue;
    }
    const std::array<VertexEvidence, 3> evidence{admission.evaluate(a), admission.evaluate(b),
                                                 admission.evaluate(c)};
    double bound = kInfinity;
    if (admission.admits(a, b, c, evidence)) bound = admission.distance_bound(a, b, c, evidence);
    maximum = std::max(maximum, bound);
  }
  const ZeroSetBandAdmission::Refusals refusals = admission.refusals();
  request.failures[0] = refusals.transversality;
  request.failures[1] = refusals.distance;
  request.failures[2] = kept;
  if (!(maximum <= request.tolerance)) return PHX_MC_UNRESOLVED;
  *request.distance = maximum;
  return PHX_MC_OK;
}
}  // namespace
}  // namespace phx::mc::remesh

extern "C" {

PHX_MC_API int32_t phx_mc_zero_set_remesh(const phx_mc_zero_set_remesh_request* request) {
  if (request == nullptr || request->result == nullptr || request->counters == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    auto owned = std::make_unique<phx::mc::remesh::ZeroSetRemeshResult>();
    const int32_t status =
        phx::mc::remesh::remesh_zero_set(*request, *owned, request->counters, false);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

PHX_MC_API int32_t phx_mc_zero_set_certify(const phx_mc_zero_set_certify_request* request) {
  if (request == nullptr || request->distance == nullptr || request->failures == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    return phx::mc::remesh::certify_zero_set(*request);
  });
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_i32(const phx_mc_zero_set_remesh_request* request) {
  if (request == nullptr || request->result == nullptr || request->counters == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    auto owned = std::make_unique<phx::mc::remesh::ZeroSetRemeshResult>();
    const int32_t status =
        phx::mc::remesh::remesh_zero_set(*request, *owned, request->counters, true);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_sizes(const void* result, int64_t* sizes) {
  if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::ZeroSetRemeshResult*>(result);
  sizes[0] = static_cast<int64_t>(data.points.size());
  sizes[1] = static_cast<int64_t>(data.triangles.size());
  return PHX_MC_OK;
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_arrays(void* result, int64_t vertex_capacity,
                                                 int64_t triangle_capacity, double* vertices,
                                                 int64_t* triangles) {
  if (result == nullptr || vertices == nullptr || triangles == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  const auto& data = *static_cast<const phx::mc::remesh::ZeroSetRemeshResult*>(result);
  if (vertex_capacity < static_cast<int64_t>(data.points.size()) ||
      triangle_capacity < static_cast<int64_t>(data.triangles.size()))
    return PHX_MC_CAPACITY_EXCEEDED;
  for (size_t v = 0; v < data.points.size(); ++v)
    for (size_t d = 0; d < 3; ++d) vertices[3 * v + d] = data.points[v][d];
  for (size_t t = 0; t < data.triangles.size(); ++t)
    for (size_t k = 0; k < 3; ++k) triangles[3 * t + k] = data.triangles[t][k];
  return PHX_MC_OK;
}

PHX_MC_API void phx_mc_zero_set_remesh_free(void* result) {
  delete static_cast<phx::mc::remesh::ZeroSetRemeshResult*>(result);
}

}  // extern "C"
