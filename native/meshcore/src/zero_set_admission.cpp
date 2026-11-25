// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// kZeroSetBand admission and the zero-set surface remesh C entry; contract in
// zero_set_admission.hpp and phydrax_meshcore.h.
#include "zero_set_admission.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

#include "box_tree.hpp"
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "rounding.hpp"
#include "surface_adapt.hpp"
#include "zero_set_tube.hpp"

namespace phx::mc::remesh {
namespace {
using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

constexpr double kInfinity = std::numeric_limits<double>::infinity();
// Midpoint levels of the uncertifiable-point search: the residual maximum of
// a triangle is located to 2^-6 of its extent before the full subdivision.
constexpr int kWitnessLevels = 6;
// Once a repair is needed, every triangle decided in a round must also be
// proved within this fraction of the tolerance. A repair splits where the
// zero set strays about the tolerance from the surface, and the reverse bound
// there is about the forward one plus the tube's slack (0.0089 against 0.008
// when repaired only to the tolerance; 0.0072 at 0.8 of it).
constexpr double kRepairFraction = 0.8;

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
  evidence_ = nullptr;
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

int32_t IntervalBoxProgram::bind_events(const int64_t* events, int64_t event_count) {
  return program_.bind_events(events, event_count);
}

bool IntervalBoxProgram::load(const Point& lower, const Point& upper) const {
  for (int d = 0; d < 3; ++d)
    if (!(lower[d] <= upper[d]) || !std::isfinite(lower[d]) || !std::isfinite(upper[d]))
      return false;
  std::copy(template_lower_.begin(), template_lower_.end(), lower_.begin());
  std::copy(template_upper_.begin(), template_upper_.end(), upper_.begin());
  for (int d = 0; d < 3; ++d) {
    lower_[static_cast<size_t>(d)] = lower[d];
    upper_[static_cast<size_t>(d)] = upper[d];
  }
  return true;
}

bool IntervalBoxProgram::flags(const Point& lower, const Point& upper, bool& smooth,
                               bool& continuous) const {
  return load(lower, upper) &&
         program_.execute_branches(lower_.data(), upper_.data(),
                                   static_cast<int64_t>(lower_.size()), smooth,
                                   continuous) == PHX_MC_OK;
}

bool IntervalBoxProgram::evaluate(const Point& lower, const Point& upper,
                                  Interval* result) const {
  // A required continuity comes from the same execution when this program
  // carries the evidence, else from one execution of the evidence program.
  bool smooth = true, continuous = true;
  if (evidence_ != nullptr && evidence_ != this &&
      (!evidence_->flags(lower, upper, smooth, continuous) || !continuous))
    return false;
  if (!load(lower, upper)) return false;
  const auto size = static_cast<int64_t>(lower_.size());
  const int32_t status = evidence_ == this ? program_.execute_branches(lower_.data(),
                                                                       upper_.data(), size,
                                                                       smooth, continuous)
                                           : program_.execute(lower_.data(), upper_.data(), size);
  if (status != PHX_MC_OK || !continuous) return false;
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
  // The certificate reads the gradient enclosure as grad f only where the
  // value program's branch evidence reports f smooth over the box.
  bool smooth = true, continuous = true;
  if (value_.checks_branches() &&
      (!value_.flags(lower, upper, smooth, continuous) || !smooth || !continuous))
    return 0;
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

namespace {
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
                        SurfaceResult& result, int64_t* counters, bool indices_i32) {
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
                         nullptr, adapted);
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

// The band admission as a triangle certifier: the certified distance bound
// of an admitted triangle, +inf otherwise.
class ZeroSetCertifier final : public TriangleCertifier {
 public:
  explicit ZeroSetCertifier(const ZeroSetBandAdmission& admission) : admission_(admission) {}
  double bound(const Point& a, const Point& b, const Point& c) const override {
    const std::array<VertexEvidence, 3> evidence{admission_.evaluate(a), admission_.evaluate(b),
                                                 admission_.evaluate(c)};
    return admission_.admits(a, b, c, evidence) ? admission_.distance_bound(a, b, c, evidence)
                                                : kInfinity;
  }

 private:
  const ZeroSetBandAdmission& admission_;
};

// The distance bound a witness box proves for a carrier triangle (see
// phx_mc_zero_set_certify): its corners lie in the box (exact comparisons),
// `value`'s gated evaluation over the box succeeds (its branch evidence
// reports f continuous there and no guard fails), and two box corners have
// value enclosures of strictly opposite sign, so a zero of f lies on the
// segment between them, in the box; every point of the triangle is then
// within the box diagonal (rounded up) of it. +inf when a check fails.
double witness_bound(const Point& a, const Point& b, const Point& c, const Point& lower,
                     const Point& upper, const IntervalBoxProgram& value) {
  for (int d = 0; d < 3; ++d) {
    if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] <= upper[d]))
      return kInfinity;
    for (const Point* p : {&a, &b, &c})
      if (!((*p)[d] >= lower[d] && (*p)[d] <= upper[d])) return kInfinity;
  }
  Interval whole{};
  if (!value.checks_branches() || !value.evaluate(lower, upper, &whole)) return kInfinity;
  bool negative = false, positive = false;
  for (int corner = 0; corner < 8 && !(negative && positive); ++corner) {
    Point at{};
    for (int d = 0; d < 3; ++d) at[d] = (corner >> d) & 1 ? upper[d] : lower[d];
    Interval enclosure{};
    if (!value.evaluate(at, at, &enclosure) || !finite(enclosure)) continue;
    negative = negative || enclosure.upper < 0;
    positive = positive || enclosure.lower > 0;
  }
  if (!(negative && positive)) return kInfinity;
  const IntervalVector extent = enclosed_difference(upper, lower);
  return norm_upper(extent);
}

// The possible-zero cover of the request's domain (box_tree.hpp), each kept
// box accepted at the first level where box_reach puts it within `radius` of
// the surface, down to a target diameter of radius / 2, evaluating at most
// request.cover_budget boxes. Every zero of f in the domain lies in a leaf
// (the tree's inclusion), so when no leaf is left outside, every zero lies
// within `bound` (the largest leaf reach, < radius) of the surface. A leaf
// holding a zero at distance delta from the surface is accepted only once its
// diameter is below about radius - delta, so when every refused leaf's reach
// is within one leaf diameter of `radius`, the cover is run again down to
// half the slack between `radius` and `forward` (the proved surface to zero
// set distance, which the reverse distance typically matches), never past
// the tree's last level. A refused leaf farther out keeps the refusal
// without the finer run. `outside` counts the leaves left outside by the
// last run, -1 when radius / 2 exceeds the tree's level, -2 when a run would
// exceed the budget; `boxes` accumulates the boxes evaluated by both runs.
int32_t cover_within(const phx_mc_zero_set_certify_request& request, const MeshSource& surface,
                     double radius, double forward, int64_t& outside, double& bound,
                     int64_t& boxes) {
  const double* lower = request.domain;
  const double* upper = request.domain + 3;
  const phx_mc_interval_box_program& field = request.value;
  const phx::mc::IntervalProgramView interval{
      field.instructions,    field.instruction_count, field.indices,
      field.index_count,     field.parameters,        field.parameter_count,
      field.workspace_lower, field.workspace_upper,   field.workspace_size,
      field.outputs[0],      field.guards,            field.guard_count};
  const auto reach = [&](const std::array<double, 6>& box) {
    return box_reach(surface, {box[0], box[1], box[2]}, {box[3], box[4], box[5]});
  };
  const phx::mc::BoxAcceptance accept = [&](const std::array<double, 6>& box) {
    const double r = reach(box);
    if (!(r < radius)) return false;
    bound = std::max(bound, r);
    return true;
  };
  // A radius whose target level exceeds the tree's limit cannot be covered.
  const int32_t coarsest = phx::mc::dyadic_target_level(lower, upper, 0.5 * radius);
  if (coarsest > phx::mc::kMaximumBoxTreeLevel) {
    outside = -1;
    return PHX_MC_OK;
  }
  std::vector<phx::mc::DyadicLeaf> leaves;
  bool near = true;
  const auto run = [&](int32_t target) -> int32_t {
    bound = 0;
    leaves.clear();
    int64_t evaluated = 0;
    const int32_t status = phx::mc::possible_zero_leaves(interval, lower, upper, target, &accept,
                                                         request.cover_budget, evaluated, leaves);
    boxes += evaluated;
    if (status == PHX_MC_CAPACITY_EXCEEDED) {
      outside = -2;
      return PHX_MC_OK;
    }
    if (status != PHX_MC_OK) return status;
    // Leaves the predicate did not accept: refused at the target level, or
    // stopped at the representability limit before it was offered.
    outside = 0;
    near = true;
    for (const phx::mc::DyadicLeaf& leaf : leaves) {
      if (leaf.accepted) continue;
      const double r = reach(leaf.box);
      if (r < radius) {
        bound = std::max(bound, r);
        continue;
      }
      ++outside;
      const Point low{leaf.box[0], leaf.box[1], leaf.box[2]};
      const Point high{leaf.box[3], leaf.box[4], leaf.box[5]};
      const double diameter = norm_upper(enclosed_difference(high, low));
      near = near && r - diameter < radius;
    }
    return PHX_MC_OK;
  };
  int32_t status = run(coarsest);
  if (status != PHX_MC_OK || outside <= 0 || !near) return status;
  const double slack = radius - forward;
  if (!(slack > 0)) return PHX_MC_OK;
  const int32_t finer = std::min(phx::mc::kMaximumBoxTreeLevel,
                                 phx::mc::dyadic_target_level(lower, upper, 0.5 * slack));
  if (finer <= coarsest) return PHX_MC_OK;
  return run(finer);
}

// Zero set -> surface (see phx_mc_zero_set_certify), by one of two proofs.
// Tube: the tube of the surface (zero_set_tube.hpp) and the cover within its
// rho, so every zero lies in the tube, within the tube's reverse bound, and
// within the cover's bound: reverse = the smaller. Cover: when the tube
// refuses or leaves a leaf outside rho, the cover within the tolerance
// alone, reverse = its bound. Proof 1 stands only within the tolerance (else
// the cover alone is tried). With `wants_room`, a tube proof that holds but
// bounds the zero set beyond the tolerance stops there instead (UNRESOLVED,
// *wants_room set): the surface needs room, not the cover alone. The cover
// alone refines into the slack left by `forward`, the proved surface to zero
// set distance (see cover_within); the tube's cover keeps radius / 2.
// counters receives (tube triangles refused in its last round; cover leaves
// left outside by the last cover run, -1 or -2 as cover_within, -3 when a
// surface vertex lies outside the domain; the proof, 1 tube or 2 cover;
// boxes evaluated).
int32_t zero_set_reverse(const phx_mc_zero_set_certify_request& request,
                         const std::vector<Point>& points, const std::vector<Triangle>& triangles,
                         const IntervalBoxProgram& value, const IntervalBoxProgram& gradient,
                         double forward, double& reverse, int64_t* counters, bool* wants_room) {
  // The cover proves every zero in the domain near the surface; a surface
  // that leaves the domain is not what that proves.
  const double* lower = request.domain;
  const double* upper = request.domain + 3;
  for (const Point& p : points)
    for (int d = 0; d < 3; ++d)
      if (!(p[d] >= lower[d] && p[d] <= upper[d])) {
        counters[1] = -3;
        return PHX_MC_UNRESOLVED;
      }
  TubeCertificate tube;
  int32_t status = certify_tube(points, triangles, value, gradient, request.orientation,
                                request.edge_length, request.tolerance, tube);
  counters[0] = tube.refused;
  if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;
  MeshSource surface;
  if (surface.bind(points, triangles) != PHX_MC_OK) return PHX_MC_INTERNAL_ERROR;
  double bound = 0;
  if (status == PHX_MC_OK) {
    status = cover_within(request, surface, tube.rho, 0.0, counters[1], bound, counters[3]);
    if (status != PHX_MC_OK) return status;
    if (counters[1] == 0 && std::min(tube.reverse, bound) <= request.tolerance) {
      reverse = std::min(tube.reverse, bound);
      counters[2] = 1;
      return PHX_MC_OK;
    }
    if (wants_room != nullptr && counters[1] == 0) {
      *wants_room = true;
      return PHX_MC_UNRESOLVED;
    }
  }
  status = cover_within(request, surface, request.tolerance, forward, counters[1], bound,
                        counters[3]);
  if (status != PHX_MC_OK) return status;
  if (counters[1] != 0) return PHX_MC_UNRESOLVED;
  reverse = bound;
  counters[2] = 2;
  return PHX_MC_OK;
}

// Certification with local repair of a closed surface against the zero set
// (see phx_mc_zero_set_certify).
int32_t certify_zero_set(const phx_mc_zero_set_certify_request& request,
                         SurfaceResult& result) {
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
      (request.carrier_triangle_count > 0 && request.carrier_boxes == nullptr) ||
      !phx::mc::addressable(request.carrier_triangle_count, 6, sizeof(double)) ||
      request.value_event_count < 0 ||
      (request.value_event_count > 0 && request.value_events == nullptr) ||
      request.domain == nullptr || !(request.edge_length > 0) ||
      !std::isfinite(request.edge_length) || !(request.minimum_angle_degrees > 0) ||
      !(request.minimum_angle_degrees < 60) || request.cover_budget < 1)
    return PHX_MC_INVALID_ARGUMENT;
  for (int d = 0; d < 3; ++d)
    if (!std::isfinite(request.domain[d]) || !std::isfinite(request.domain[d + 3]) ||
        !(request.domain[d] < request.domain[d + 3]))
      return PHX_MC_INVALID_ARGUMENT;
  std::fill(request.counters, request.counters + 10, 0);
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
  // Every enclosure the proofs read holds only where f's branch evidence
  // (the value program's own events) reports f continuous; transversality
  // also requires smooth (ZeroSetBandAdmission::transversal_sign).
  status = value.bind_events(request.value_events, request.value_event_count);
  if (status != PHX_MC_OK) return status;
  value.require_continuity(&value);
  gradient.require_continuity(&value);
  ImplicitOracle oracle;
  status = oracle.bind(request.value);
  if (status != PHX_MC_OK) return status;
  oracle.set_orientation(request.orientation);
  // The widest proved witness steers the projection reach (below), as the
  // carrier distance does in the remesh (construction only).
  const auto load = [](const double* vertices, int64_t vertex_count, const int64_t* triangles,
                       int64_t triangle_count, std::vector<Point>& points,
                       std::vector<Triangle>& rows) {
    points.resize(static_cast<size_t>(vertex_count));
    for (size_t v = 0; v < points.size(); ++v)
      for (size_t d = 0; d < 3; ++d) {
        points[v][d] = vertices[3 * v + d];
        if (!std::isfinite(points[v][d])) return false;
      }
    rows.resize(static_cast<size_t>(triangle_count));
    for (size_t t = 0; t < rows.size(); ++t)
      for (size_t k = 0; k < 3; ++k) {
        rows[t][k] = triangles[3 * t + k];
        if (rows[t][k] < 0 || rows[t][k] >= vertex_count) return false;
      }
    return true;
  };
  std::vector<Point> points, carrier_points;
  std::vector<Triangle> triangles, carrier_triangles;
  if (!load(request.vertices, request.vertex_count, request.triangles, request.triangle_count,
            points, triangles) ||
      !load(request.carrier_vertices, request.carrier_vertex_count, request.carrier_triangles,
            request.carrier_triangle_count, carrier_points, carrier_triangles))
    return PHX_MC_INVALID_INPUT;
  CarrierEvidence carrier;
  double reach = 0;
  for (size_t t = 0; t < carrier_triangles.size(); ++t) {
    const Point& a = carrier_points[static_cast<size_t>(carrier_triangles[t][0])];
    const Point& b = carrier_points[static_cast<size_t>(carrier_triangles[t][1])];
    const Point& c = carrier_points[static_cast<size_t>(carrier_triangles[t][2])];
    const double* box = request.carrier_boxes + 6 * t;
    const double bound = witness_bound(a, b, c, {box[0], box[1], box[2]},
                                       {box[3], box[4], box[5]}, value);
    if (!(bound < kInfinity)) continue;
    carrier.entries.push_back({triangle_key(a, b, c), bound});
    reach = std::max(reach, bound);
  }
  std::sort(carrier.entries.begin(), carrier.entries.end());
  oracle.set_reach(request.edge_length + reach);
  AdmissionMemo memo, repair_memo;
  const ZeroSetBandAdmission admission(value, gradient, request.tolerance, request.orientation,
                                       &memo);
  const ZeroSetCertifier certifier(admission);
  // Once a repair is needed, triangles are held to the tighter band too.
  const ZeroSetBandAdmission repair_admission(value, gradient,
                                              kRepairFraction * request.tolerance,
                                              request.orientation, &repair_memo);
  const ZeroSetCertifier repair_certifier(repair_admission);
  CertifyCounters certified;
  const auto forward = [&](bool margin_from_start) {
    const int32_t outcome = certify_surface(
        points, triangles, oracle,
        AdaptTargets{request.edge_length, request.tolerance, request.minimum_angle_degrees},
        certifier, carrier, &repair_certifier, margin_from_start, certified);
    const ZeroSetBandAdmission::Refusals refusals = admission.refusals();
    request.counters[0] = refusals.transversality;
    request.counters[1] = refusals.distance;
    request.counters[2] = certified.refused_embedding;
    request.counters[3] = certified.carrier;
    request.counters[4] = certified.repair_splits;
    request.counters[5] = certified.rounds;
    return outcome;
  };
  status = forward(false);
  if (status != PHX_MC_OK) return status;
  // Zero set -> surface over the domain (phx_mc_zero_set_certify).
  double reverse = 0;
  // A surface that needed no repair was held to the tolerance alone; when its
  // tube proof holds but bounds the zero set beyond the tolerance, it is
  // repaired into the tighter band once (before any cover-alone run) and
  // both directions are decided again.
  bool wants_room = false;
  status = zero_set_reverse(request, points, triangles, value, gradient, certified.distance,
                            reverse, request.counters + 6,
                            certified.rounds == 1 ? &wants_room : nullptr);
  if (wants_room) {
    status = forward(true);
    if (status != PHX_MC_OK) return status;
    status = zero_set_reverse(request, points, triangles, value, gradient, certified.distance,
                              reverse, request.counters + 6, nullptr);
  }
  if (status != PHX_MC_OK) return status;
  // Proved triangles are within the tolerance; carried triangles keep their
  // witness bound, which the caller weighs against its request.
  request.distances[0] = certified.distance;
  request.distances[1] = reverse;
  result.points = std::move(points);
  result.triangles = std::move(triangles);
  result.carried = std::move(certified.carried);
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
    auto owned = std::make_unique<phx::mc::remesh::SurfaceResult>();
    const int32_t status =
        phx::mc::remesh::remesh_zero_set(*request, *owned, request->counters, false);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

PHX_MC_API int32_t phx_mc_zero_set_certify(const phx_mc_zero_set_certify_request* request) {
  if (request == nullptr || request->distances == nullptr || request->counters == nullptr ||
      request->result == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    auto owned = std::make_unique<phx::mc::remesh::SurfaceResult>();
    const int32_t status = phx::mc::remesh::certify_zero_set(*request, *owned);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

PHX_MC_API int32_t phx_mc_zero_set_remesh_i32(const phx_mc_zero_set_remesh_request* request) {
  if (request == nullptr || request->result == nullptr || request->counters == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  *request->result = nullptr;
  return phx::mc::guarded([&]() -> int32_t {
    auto owned = std::make_unique<phx::mc::remesh::SurfaceResult>();
    const int32_t status =
        phx::mc::remesh::remesh_zero_set(*request, *owned, request->counters, true);
    if (status != PHX_MC_OK) return status;
    *request->result = owned.release();
    return PHX_MC_OK;
  });
}

}  // extern "C"
