// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_faces.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "chart_surface.hpp"
#include "expansion.hpp"
#include "predicates.hpp"

#include "phydrax_meshcore.h"

namespace phx::mc {

static_assert(static_cast<int32_t>(CadRefine2DRefusal::none) == PHX_MC_CAD_REFUSAL_NONE);
static_assert(static_cast<int32_t>(CadRefine2DRefusal::encroachment) == PHX_MC_CAD_REFUSAL_ENCROACHMENT);
static_assert(static_cast<int32_t>(CadRefine2DRefusal::outside) == PHX_MC_CAD_REFUSAL_OUTSIDE);
static_assert(static_cast<int32_t>(CadRefine2DRefusal::proximity) == PHX_MC_CAD_REFUSAL_PROXIMITY);
static_assert(static_cast<int32_t>(CadRefine2DRefusal::nonstar) == PHX_MC_CAD_REFUSAL_NONSTAR);
static_assert(static_cast<int32_t>(CadRefine2DRefusal::excluded) == PHX_MC_CAD_REFUSAL_EXCLUDED);

int32_t validate_cad_curve_discretization(const phx_mc_cad_model& model,
                                          const CadCurveDiscretization& curves) {
  if (model.curve_count < 0 || !curves.offsets ||
      (model.curve_count != 0 && !curves.parameters)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (curves.offsets[0] != 0) return PHX_MC_INVALID_INPUT;
  for (int64_t c = 0; c < model.curve_count; ++c) {
    const int64_t begin = curves.offsets[c];
    const int64_t end = curves.offsets[c + 1];
    if (begin < 0 || end < begin || end - begin < 2 ||
        curves.parameters[begin] != 0.0 || curves.parameters[end - 1] != 1.0) {
      return PHX_MC_INVALID_INPUT;
    }
    for (int64_t i = begin + 1; i < end; ++i) {
      if (!(curves.parameters[i - 1] < curves.parameters[i])) return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}
namespace {


struct PoleBoundary {
  int64_t use;
  int32_t first_vertex;
  int32_t count;
};

struct FaceBoundary {
  std::vector<CadFaceVertex> vertices;
  std::vector<int32_t> segments;
  std::vector<CadFaceEdge> edges;
  std::vector<std::vector<int32_t>> loops;
  std::vector<PoleBoundary> poles;
};

double use_parameter(const phx_mc_cad_model& model, int64_t use, double s) {
  const double first = model.use_ranges[2 * use];
  const double last = model.use_ranges[2 * use + 1];
  if (s == 0.0) return first;
  if (s == 1.0) return last;
  return first + s * (last - first);
}

bool chart_point(const cad::CadGeometry& geometry, int64_t use, double parameter,
                 double interior, cad::EvaluationWorkspace& workspace,
                 std::array<double, 2>& uv) {
  return geometry.jets_in_span(geometry.pcurve_node(use), parameter, interior, 0,
                               uv.data(), workspace) &&
         std::isfinite(uv[0]) && std::isfinite(uv[1]);
}
bool chart_point(const cad::CadGeometry& geometry, int64_t use, double parameter,
                 cad::EvaluationWorkspace& workspace, std::array<double, 2>& uv) {
  return geometry.jets(geometry.pcurve_node(use), &parameter, 0, uv.data(), workspace) &&
         std::isfinite(uv[0]) && std::isfinite(uv[1]);
}


uint64_t ulp_gap(double a, double b) {
  const auto ordered = [](double value) {
    const uint64_t bits = std::bit_cast<uint64_t>(value);
    constexpr uint64_t sign = uint64_t{1} << 63;
    return bits & sign ? ~bits : bits | sign;
  };
  if (a == b) return 0;
  const uint64_t x = ordered(a), y = ordered(b);
  return x > y ? x - y : y - x;
}

// Floating point proposes the integer; expansion arithmetic alone decides
// whether the exact difference is a nonzero integer number of periods.
int exact_turn(double first, double last, double period, int64_t& turns) {
  const double ratio = (last - first) / period;
  if (!std::isfinite(ratio) || std::abs(ratio) > 0x1p52) return -1;
  const double candidate = std::nearbyint(ratio);
  if (candidate == 0.0) return 0;
  if (!(Expansion::difference(last, first) - Expansion::product(candidate, period)).is_zero())
    return 0;
  turns = static_cast<int64_t>(candidate);
  return 1;
}

int32_t check_loop_closure(const phx_mc_cad_model& model, int32_t face, int32_t loop,
                           const std::array<double, 2>& first, const std::array<double, 2>& last,
                           CadFaceMesh& result) {
  for (int axis = 0; axis < 2; ++axis) {
    const double period = model.surface_periods[2 * face + axis];
    if (!(period > 0.0) || first[axis] == last[axis]) continue;
    int64_t turns = 0;
    const int decision = exact_turn(first[axis], last[axis], period, turns);
    if (decision != 0) {
      result.failure = {face, loop, -1, axis,
                        decision > 0 ? PHX_MC_CAD_FACE_PERIODIC_WRAP
                                     : PHX_MC_CAD_FACE_SHIFT_UNREPRESENTABLE,
                        turns, 0.0, -1, 0, 0, 0, -1, -1, {}, {}, {}};
      return PHX_MC_UNRESOLVED;
    }
    ++result.periodic_corner_gaps;
    result.max_periodic_corner_gap_ulps =
        std::max(result.max_periodic_corner_gap_ulps, ulp_gap(first[axis], last[axis]));
  }
  return PHX_MC_OK;
}

// One endpoint value is chosen for each consecutive use occurrence: the end
// of the preceding use, except at the loop closure where the first start wins.
int32_t make_boundary(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                      int32_t face, cad::EvaluationWorkspace& workspace, FaceBoundary& boundary,
                      CadFaceMesh& result) {
  const auto& model = geometry.model();
  for (int64_t loop = 0; loop < model.loop_count; ++loop) {
    if (model.loop_surfaces[loop] != face) continue;
    const int64_t first_use = model.loop_use_offsets[loop];
    const int64_t last_use = model.loop_use_offsets[loop + 1];
    if (first_use == last_use) return PHX_MC_INVALID_INPUT;
    std::vector<int32_t> rows;
    std::array<double, 2> first_uv{}, previous_uv{};
    int32_t previous_corner = -1;
    for (int64_t use = first_use; use < last_use; ++use) {
      const bool pole = model.use_kinds[use] == 1;
      if (!pole && model.use_kinds[use] != 0) return PHX_MC_INVALID_INPUT;
      const int32_t entity = model.use_entities[use];
      if (entity < 0 || (pole ? entity >= model.corner_count : entity >= model.curve_count))
        return PHX_MC_INVALID_INPUT;
      const int64_t begin = pole ? 0 : curves.offsets[entity];
      const int64_t count = pole ? 9 : curves.offsets[entity + 1] - begin;
      const bool forward = model.use_senses[use] > 0;
      const int32_t start_corner = pole ? entity : model.curve_corners[2 * entity + (forward ? 0 : 1)];
      const int32_t end_corner = pole ? entity : model.curve_corners[2 * entity + (forward ? 1 : 0)];
      if (pole) boundary.poles.push_back({use, static_cast<int32_t>(boundary.vertices.size()),
                                          static_cast<int32_t>(count - 1)});
      if (use != first_use && start_corner != previous_corner) return PHX_MC_INVALID_INPUT;
      std::array<Expansion, 2> shift;
      std::array<double, 2> previous_raw{};
      for (int64_t j = 0; j <= count - 1; ++j) {
        const int64_t index = forward ? j : count - 1 - j;
        const double s = pole ? static_cast<double>(index) / (count - 1)
                              : curves.parameters[begin + index];
        const double parameter = use_parameter(model, use, s);
        const int64_t adjacent = j == count - 1 ? (forward ? index - 1 : index + 1)
                                                : (forward ? index + 1 : index - 1);
        const double adjacent_s = pole ? static_cast<double>(adjacent) / (count - 1)
                                       : curves.parameters[begin + adjacent];
        const double adjacent_parameter = use_parameter(model, use, adjacent_s);
        const double interior = std::midpoint(parameter, adjacent_parameter);
        std::array<double, 2> uv;
        if (!chart_point(geometry, use, parameter, interior, workspace, uv))
          return PHX_MC_INVALID_INPUT;
        if (j > 0 && j < count - 1) {
          for (int axis = 0; axis < 2; ++axis) {
            const double period = model.surface_periods[2 * face + axis];
            if (!(period > 0.0) || uv[axis] == previous_raw[axis]) continue;
            int64_t turns = 0;
            const int decision = exact_turn(previous_raw[axis], uv[axis], period, turns);
            if (decision != 1) {
              result.failure = {face, static_cast<int32_t>(loop), static_cast<int32_t>(use), axis,
                                decision < 0 ? PHX_MC_CAD_FACE_SHIFT_UNREPRESENTABLE
                                             : PHX_MC_CAD_FACE_SHEET_MISMATCH,
                                turns, parameter, -1, 0, 0, 0, -1, -1, {}, {}, {}};
              return PHX_MC_UNRESOLVED;
            }
            shift[axis] = shift[axis] - Expansion::product(static_cast<double>(turns), period);
          }
        }
        for (int axis = 0; axis < 2; ++axis) {
          const Expansion exact = Expansion(uv[axis]) + shift[axis];
          const double shifted = exact.estimate();
          if (!std::isfinite(shifted) || !(exact - Expansion(shifted)).is_zero()) {
            result.failure = {face, static_cast<int32_t>(loop), static_cast<int32_t>(use), axis,
                              PHX_MC_CAD_FACE_SHIFT_UNREPRESENTABLE, 0, parameter,
                              -1, 0, 0, 0, -1, -1, {}, {}, {}};
            return PHX_MC_UNRESOLVED;
          }
          uv[axis] = shifted;
        }
        if (j < count - 1 &&
            !chart_point(geometry, use, adjacent_parameter, interior, workspace, previous_raw))
          return PHX_MC_INVALID_INPUT;
        if (j == 0 && use == first_use) first_uv = uv;
        if (j == 0 && use != first_use) uv = previous_uv;
        if (j == count - 1) {
          previous_uv = uv;
          previous_corner = end_corner;
          continue;
        }
        const int32_t dimension = (pole || j == 0) ? 0 : 1;
        const int32_t classification = dimension == 0 ? start_corner : entity;
        rows.push_back(static_cast<int32_t>(boundary.vertices.size()));
        boundary.vertices.push_back({uv, dimension, classification, static_cast<int32_t>(index)});
        const int64_t next_index = forward ? index + 1 : index - 1;
        const double next_s = pole ? static_cast<double>(next_index) / (count - 1)
                                   : curves.parameters[begin + next_index];
        boundary.edges.push_back({{rows.back(), -1}, static_cast<int32_t>(use),
                                  pole ? -1 : entity, {s, next_s}});
      }
    }
    if (previous_corner != boundary.vertices[rows.front()].entity) return PHX_MC_INVALID_INPUT;
    const int32_t closure = check_loop_closure(model, face, static_cast<int32_t>(loop),
                                               first_uv, previous_uv, result);
    if (closure != PHX_MC_OK) return closure;
    for (std::size_t j = 0; j < rows.size(); ++j) {
      boundary.segments.push_back(rows[j]);
      boundary.segments.push_back(rows[(j + 1) % rows.size()]);
      boundary.edges[rows[j]].vertices[1] = rows[(j + 1) % rows.size()];
    }
    boundary.loops.push_back(std::move(rows));
  }
  return boundary.loops.empty() ? PHX_MC_INVALID_INPUT : PHX_MC_OK;
}

// Size grows at most this much per unit distance away from the boundary.
constexpr double kBoundaryGrading = 0.3;

struct BoundarySample {
  std::array<double, 3> xyz;
  double spacing;
};

// A fixed boundary chord between consecutive loop samples.
struct BoundaryChord {
  std::array<double, 3> a;
  std::array<double, 3> b;
  double length;
  // The chart chord's lens: the face's chart arc at the middle of the curve
  // segment vs the chart chord's middle, mapped (0 off a curve).
  double lens;
};

struct BoundaryNode {
  std::array<double, 3> low;
  std::array<double, 3> high;
  double spacing;  // Sample tree: minimum spacing; chord tree: maximum length.
  int32_t begin;
  int32_t end;
  int32_t left;
  int32_t right;
};


struct PoleZone {
  int axis;
  double origin;
  double direction;
  double width;
  double spacing;
  // A pole's first band takes no kernel point (the fan owns it); a trim
  // line's band is the kernel's where the ring chain leaves it open (at a
  // seam, whose vertices need not lie on a ring).
  bool pole;
};

struct FaceMetric {
  const cad::CadGeometry* geometry;
  int32_t face;
  double size;
  double tolerance;
  double theta;
  double scale_u;
  double scale_v;
  cad::EvaluationWorkspace workspace;
  std::vector<BoundarySample> boundary;
  std::vector<int32_t> boundary_order;
  std::vector<BoundaryNode> boundary_tree;
  std::vector<BoundaryChord> chords;
  std::vector<int32_t> chord_order;
  std::vector<BoundaryNode> chord_tree;
  std::vector<PoleZone> poles;
  std::vector<std::array<double, 2>>* metric_points;
  // Input boundary charts with their classified corner/curve positions, sorted
  // by chart, so the position hook agrees with assembly bit for bit.
  std::vector<std::pair<std::array<double, 2>, std::array<double, 3>>> boundary_positions;
  std::vector<std::array<double, 5>>* position_calls;
  // Solid route: pole rings follow the graded base size (seed_poles).
  bool graded_pole_rings = false;
};

bool surface_jet(FaceMetric& context, const double uv[2], double* jet) {
  return context.geometry->jets(context.geometry->surface_node(context.face), uv, 2, jet,
                                context.workspace);
}

int32_t build_boundary_tree(FaceMetric& context, int32_t begin, int32_t end) {
  const int32_t index = static_cast<int32_t>(context.boundary_tree.size());
  const double inf = std::numeric_limits<double>::infinity();
  BoundaryNode node{{inf, inf, inf}, {-inf, -inf, -inf}, inf, begin, end, -1, -1};
  for (int32_t i = begin; i < end; ++i) {
    const auto& sample = context.boundary[context.boundary_order[i]];
    node.spacing = std::min(node.spacing, sample.spacing);
    for (int k = 0; k < 3; ++k) {
      node.low[k] = std::min(node.low[k], sample.xyz[k]);
      node.high[k] = std::max(node.high[k], sample.xyz[k]);
    }
  }
  context.boundary_tree.push_back(node);
  if (end - begin > 8) {
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (node.high[k] - node.low[k] > node.high[axis] - node.low[axis]) axis = k;
    const int32_t middle = begin + (end - begin) / 2;
    std::nth_element(context.boundary_order.begin() + begin,
                     context.boundary_order.begin() + middle,
                     context.boundary_order.begin() + end, [&](int32_t a, int32_t b) {
      const double x = context.boundary[a].xyz[axis], y = context.boundary[b].xyz[axis];
      return x != y ? x < y : a < b;
    });
    const int32_t left = build_boundary_tree(context, begin, middle);
    const int32_t right = build_boundary_tree(context, middle, end);
    context.boundary_tree[index].left = left;
    context.boundary_tree[index].right = right;
  }
  return index;
}

void grade_boundary(const FaceMetric& context, int32_t index, const double* point, double& h) {
  const BoundaryNode& node = context.boundary_tree[index];
  double box_distance2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double delta = std::max({node.low[k] - point[k], 0.0, point[k] - node.high[k]});
    box_distance2 += delta * delta;
  }
  if (node.spacing + kBoundaryGrading * std::sqrt(box_distance2) >= h) return;
  if (node.left >= 0) {
    grade_boundary(context, node.left, point, h);
    grade_boundary(context, node.right, point, h);
    return;
  }
  for (int32_t i = node.begin; i < node.end; ++i) {
    const BoundarySample& sample = context.boundary[context.boundary_order[i]];
    const double dx = point[0] - sample.xyz[0], dy = point[1] - sample.xyz[1];
    const double dz = point[2] - sample.xyz[2];
    h = std::min(h, sample.spacing +
                        kBoundaryGrading * std::sqrt(dx * dx + dy * dy + dz * dz));
  }
}

int32_t build_chord_tree(FaceMetric& context, int32_t begin, int32_t end) {
  const int32_t index = static_cast<int32_t>(context.chord_tree.size());
  const double inf = std::numeric_limits<double>::infinity();
  BoundaryNode node{{inf, inf, inf}, {-inf, -inf, -inf}, 0.0, begin, end, -1, -1};
  for (int32_t i = begin; i < end; ++i) {
    const auto& chord = context.chords[context.chord_order[i]];
    node.spacing = std::max(node.spacing, chord.length);
    for (int k = 0; k < 3; ++k) {
      node.low[k] = std::min({node.low[k], chord.a[k], chord.b[k]});
      node.high[k] = std::max({node.high[k], chord.a[k], chord.b[k]});
    }
  }
  context.chord_tree.push_back(node);
  if (end - begin > 8) {
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (node.high[k] - node.low[k] > node.high[axis] - node.low[axis]) axis = k;
    const int32_t middle = begin + (end - begin) / 2;
    std::nth_element(context.chord_order.begin() + begin, context.chord_order.begin() + middle,
                     context.chord_order.begin() + end, [&](int32_t a, int32_t b) {
      const auto& p = context.chords[a];
      const auto& q = context.chords[b];
      const double x = p.a[axis] + p.b[axis], y = q.a[axis] + q.b[axis];
      return x != y ? x < y : a < b;
    });
    const int32_t left = build_chord_tree(context, begin, middle);
    const int32_t right = build_chord_tree(context, middle, end);
    context.chord_tree[index].left = left;
    context.chord_tree[index].right = right;
  }
  return index;
}

void chord_distance(const FaceMetric& context, int32_t index, const double* point,
                    double& best, const BoundaryChord*& nearest);

// Isotropic base size at a face point: the request h, graded from the
// boundary samples and floored on the nearest fixed chord. Grading from the
// nearby sample spacing asks for small triangles around a short boundary
// segment, but a neighbouring long chord is fixed: every triangle on it has
// circumradius >= L/2 and no vertex may enter its diametral disk, so a size
// below L there cannot be met. The floor L - g d comes from the chord nearest
// the point only (the one a triangle there stands on, at its centroid): a
// floor taken from every chord lifted the size over a short lens-limited
// chord beside a longer one, and the kernel accepted the stretched triangle
// on it (measured: a 35.4 deg apex on a 0.35 h chord at h = 0.02286). It
// overrides the grading, never the request or the curvature sizes.
double base_size(const FaceMetric& context, const double* jet) {
  double h = context.size;
  const double ungraded = h;
  if (!context.boundary_tree.empty()) grade_boundary(context, 0, jet, h);
  // A chord farther than (L_max - h) / g cannot raise h, so the search for
  // the nearest chord stops there (none found: no floor).
  if (!context.chord_tree.empty() && context.chord_tree[0].spacing > h) {
    double distance = (context.chord_tree[0].spacing - h) / kBoundaryGrading;
    const BoundaryChord* nearest = nullptr;
    chord_distance(context, 0, jet, distance, nearest);
    if (nearest)
      h = std::max(h, std::min(ungraded, nearest->length - kBoundaryGrading * distance));
  }
  return h;
}

// Lowers `best` to the distance from point to the nearest boundary chord
// when nearer and sets `nearest` to that chord (the longest among equally
// near ones).
void chord_distance(const FaceMetric& context, int32_t index, const double* point,
                    double& best, const BoundaryChord*& nearest) {
  const BoundaryNode& node = context.chord_tree[index];
  double box_distance2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double delta = std::max({node.low[k] - point[k], 0.0, point[k] - node.high[k]});
    box_distance2 += delta * delta;
  }
  if (box_distance2 > best * best) return;
  if (node.left >= 0) {
    chord_distance(context, node.left, point, best, nearest);
    chord_distance(context, node.right, point, best, nearest);
    return;
  }
  for (int32_t i = node.begin; i < node.end; ++i) {
    const BoundaryChord& chord = context.chords[context.chord_order[i]];
    double t = 0.0, norm = 0.0, ab[3], ap[3];
    for (int k = 0; k < 3; ++k) {
      ab[k] = chord.b[k] - chord.a[k];
      ap[k] = point[k] - chord.a[k];
      t += ab[k] * ap[k];
      norm += ab[k] * ab[k];
    }
    t = std::clamp(t / norm, 0.0, 1.0);
    double distance2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double e = ap[k] - t * ab[k];
      distance2 += e * e;
    }
    const double distance = std::sqrt(distance2);
    if (distance < best || (distance == best && nearest && chord.length > nearest->length))
      nearest = &chord;
    best = std::min(best, distance);
  }
}

// The fraction of tol a face point's triangles get: the boundary layer's
// theta - lens / tol next to a curve chord (the nearest chord's own lens,
// at most kLensShare theta tol, takes the rest; cad_faces.hpp), theta away
// from it. The curvature size sqrt(4 theta' tol / |k|max) is the layer's
// s_l within s_l of a chord (which holds the centroid and circumcentre of a
// triangle on it) and grows from there at the boundary grading's rate up to
// theta's s_f: a stepped size left triangles across the step whose split
// points were all too close (measured: a proximity refusal on the quadric
// at h = 0.04925). The search stops at the smallest layer's reach.
double local_theta(const FaceMetric& context, const double* jet,
                   const cad::SurfaceDifferential& differential) {
  const double kappa = std::max(std::abs(differential.curvatures[0]),
                                std::abs(differential.curvatures[1]));
  if (context.chord_tree.empty() || !(kappa > 0.0)) return context.theta;
  const double outer = std::sqrt(4.0 * context.theta * context.tolerance / kappa);
  const double smallest = std::sqrt(1.0 - kLensShare) * outer;
  double distance = smallest + (outer - smallest) / kBoundaryGrading;
  const BoundaryChord* nearest = nullptr;
  chord_distance(context, 0, jet, distance, nearest);
  if (!nearest || !(nearest->lens > 0.0)) return context.theta;
  const double layer = context.theta -
      std::min(nearest->lens / context.tolerance, kLensShare * context.theta);
  const double inner = std::sqrt(4.0 * layer * context.tolerance / kappa);
  const double size = inner + kBoundaryGrading * std::max(0.0, distance - inner);
  return size >= outer ? context.theta : context.theta * (size * size) / (outer * outer);
}

CadMetric2D face_metric(const double chart[2], void* opaque) {
  auto& context = *static_cast<FaceMetric*>(opaque);
  const double uv[2] = {chart[0] / context.scale_u, chart[1] / context.scale_v};
  if (context.metric_points) context.metric_points->push_back({uv[0], uv[1]});
  double jet[18];
  cad::SurfaceDifferential differential;
  if (!surface_jet(context, uv, jet) || !cad::surface_differential(jet, differential)) {
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    return {invalid, invalid, invalid};
  }
  // Fidelity: the isotropic size intersected with the request's curvature
  // tensor (cad::fidelity_metric), at the boundary layer's share near a
  // curve.
  double m[3];
  const double h = base_size(context, jet);
  cad::fidelity_metric(differential, h, local_theta(context, jet, differential),
                       context.tolerance, m, nullptr);
  return {m[0] / (context.scale_u * context.scale_u), m[1] / (context.scale_u * context.scale_v),
          m[2] / (context.scale_v * context.scale_v)};
}

// The face metric at a vertex as a row-major 3D tensor for the face stage's
// flips: lambda_2 along the k_2 principal direction, lambda_1 across it
// (normal included). False where the fidelity term does not bind (the flips
// then stay the plain 3D Delaunay test) or the jet is degenerate.
bool vertex_metric(FaceMetric& context, const double chart[2], double tensor[9]) {
  const double uv[2] = {chart[0] / context.scale_u, chart[1] / context.scale_v};
  double jet[18];
  cad::SurfaceDifferential differential;
  if (!surface_jet(context, uv, jet) || !cad::surface_differential(jet, differential)) return false;
  double m[3], lambda[2];
  const double h = base_size(context, jet);
  cad::fidelity_metric(differential, h, local_theta(context, jet, differential),
                       context.tolerance, m, lambda);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) tensor[3 * i + j] = i == j ? lambda[0] : 0.0;
  const double base = 1.0 / (h * h);
  if (lambda[0] == lambda[1]) return lambda[0] != base;
  // Eigenvector of W = I^-1 II for k_2: a column of W - k_1 Id (the range of
  // the spectral projector), the longer one for conditioning.
  const double* I = differential.first_form;
  const double* II = differential.second_form;
  const double determinant = I[0] * I[2] - I[1] * I[1];
  const double a = (I[2] * II[0] - I[1] * II[1]) / determinant;
  const double b = (I[2] * II[1] - I[1] * II[2]) / determinant;
  const double c = (I[0] * II[1] - I[1] * II[0]) / determinant;
  const double d = (I[0] * II[2] - I[1] * II[1]) / determinant;
  const double k1 = differential.curvatures[0];
  double direction[2] = {a - k1, c};
  if (std::hypot(b, d - k1) > std::hypot(direction[0], direction[1])) {
    direction[0] = b;
    direction[1] = d - k1;
  }
  double e[3];
  for (int i = 0; i < 3; ++i) e[i] = jet[3 + i] * direction[0] + jet[6 + i] * direction[1];
  const double length = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  if (!(length > 0.0) || !std::isfinite(length)) return false;
  for (double& x : e) x /= length;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) tensor[3 * i + j] += (lambda[1] - lambda[0]) * e[i] * e[j];
  return true;
}

// Chart-to-physical position of a face vertex: the classified position at an
// input boundary chart, otherwise the face's order-0 evaluation. Every final
// face position is produced here, so the kernel's 3D guards see the points
// that assembly emits.
int32_t face_position(void* opaque, const double chart[2], double xyz[3]) {
  auto& context = *static_cast<FaceMetric*>(opaque);
  const std::array<double, 2> key{chart[0], chart[1]};
  const auto found = std::lower_bound(
      context.boundary_positions.begin(), context.boundary_positions.end(), key,
      [](const auto& entry, const std::array<double, 2>& value) { return entry.first < value; });
  const double uv[2] = {chart[0] / context.scale_u, chart[1] / context.scale_v};
  if (found != context.boundary_positions.end() && found->first == key) {
    std::copy(found->second.begin(), found->second.end(), xyz);
  } else if (!context.geometry->jets(context.geometry->surface_node(context.face), uv, 0, xyz,
                                     context.workspace)) {
    return 0;
  }
  if (context.position_calls)
    context.position_calls->push_back({uv[0], uv[1], xyz[0], xyz[1], xyz[2]});
  return 1;
}

int loop_orientation(const FaceBoundary& boundary, const std::vector<int32_t>& loop) {
  std::size_t left = 0;
  for (std::size_t i = 1; i < loop.size(); ++i) {
    if (boundary.vertices[loop[i]].uv < boundary.vertices[loop[left]].uv) left = i;
  }
  const auto& p = boundary.vertices[loop[left]].uv;
  for (std::size_t before = 1; before < loop.size(); ++before) {
    const auto& a = boundary.vertices[loop[(left + loop.size() - before) % loop.size()]].uv;
    for (std::size_t after = 1; after < loop.size(); ++after) {
      const auto& b = boundary.vertices[loop[(left + after) % loop.size()]].uv;
      const int sign = orient2d(a.data(), p.data(), b.data());
      if (sign != 0) return sign;
    }
  }
  return 0;
}

bool contains(const FaceBoundary& boundary, const std::vector<int32_t>& loop,
              const std::array<double, 2>& p) {
  bool inside = false;
  for (std::size_t i = 0; i < loop.size(); ++i) {
    const auto& a = boundary.vertices[loop[i]].uv;
    const auto& b = boundary.vertices[loop[(i + 1) % loop.size()]].uv;
    const int side = orient2d(a.data(), b.data(), p.data());
    if (side == 0 && p[0] >= std::min(a[0], b[0]) && p[0] <= std::max(a[0], b[0]) &&
        p[1] >= std::min(a[1], b[1]) && p[1] <= std::max(a[1], b[1])) return false;
    if ((a[1] <= p[1] && p[1] < b[1] && side > 0) ||
        (b[1] <= p[1] && p[1] < a[1] && side < 0)) inside = !inside;
  }
  return inside;
}

bool hole_seed(const FaceBoundary& boundary, const std::vector<int32_t>& loop,
               std::array<double, 2>& seed) {
  const int orientation = loop_orientation(boundary, loop);
  if (orientation == 0) return false;
  for (std::size_t i = 0; i < loop.size(); ++i) {
    const auto& a = boundary.vertices[loop[(i + loop.size() - 1) % loop.size()]].uv;
    const auto& b = boundary.vertices[loop[i]].uv;
    const auto& c = boundary.vertices[loop[(i + 1) % loop.size()]].uv;
    if (orientation * orient2d(a.data(), b.data(), c.data()) <= 0) continue;
    for (double fraction = 0.25; fraction > 1.0e-15; fraction *= 0.5) {
      seed = {b[0] + fraction * ((a[0] - b[0]) + (c[0] - b[0])),
              b[1] + fraction * ((a[1] - b[1]) + (c[1] - b[1]))};
      if (contains(boundary, loop, seed)) return true;
    }
  }
  return false;
}

bool evaluate_position(const cad::CadGeometry& geometry, int32_t face,
                       const std::array<double, 2>& uv, cad::EvaluationWorkspace& workspace,
                       std::array<double, 3>& point) {
  return geometry.jets(geometry.surface_node(face), uv.data(), 0, point.data(), workspace);
}

bool classified_position(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                         int32_t face, const CadFaceVertex& vertex,
                         cad::EvaluationWorkspace& workspace, std::array<double, 3>& point) {
  const auto& model = geometry.model();
  if (vertex.dimension == 0) {
    if (vertex.entity < 0 || vertex.entity >= model.corner_count) return false;
    std::copy_n(model.corner_points + 3 * vertex.entity, 3, point.begin());
    return true;
  }
  if (vertex.dimension == 1) {
    const int32_t curve = vertex.entity;
    const double s = curves.parameters[curves.offsets[curve] + vertex.curve_index];
    const double first = model.curve_ranges[2 * curve];
    const double last = model.curve_ranges[2 * curve + 1];
    const double parameter = s == 0.0 ? first : s == 1.0 ? last : first + s * (last - first);
    return geometry.jets(geometry.curve_node(curve), &parameter, 0, point.data(), workspace);
  }
  return evaluate_position(geometry, face, vertex.uv, workspace, point);
}


double distance(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
}

// The lens of boundary edge `edge` (its chart chord vs the face's chart arc
// at the middle of its curve segment, mapped; the quantity the curve stage
// bounds by kLensShare theta tol); 0 for a pole edge.
bool chord_lens(const cad::CadGeometry& geometry, const FaceBoundary& boundary, int32_t face,
                int32_t edge, cad::EvaluationWorkspace& workspace, double& lens) {
  const CadFaceEdge& e = boundary.edges[static_cast<std::size_t>(edge)];
  lens = 0.0;
  if (e.curve < 0) return true;
  const auto& a = boundary.vertices[static_cast<std::size_t>(e.vertices[0])].uv;
  const auto& b = boundary.vertices[static_cast<std::size_t>(e.vertices[1])].uv;
  std::array<double, 2> arc;
  const std::array<double, 2> chord{0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])};
  std::array<double, 3> x, y;
  if (!chart_point(geometry, e.use,
                   use_parameter(geometry.model(), e.use, 0.5 * (e.parameters[0] + e.parameters[1])),
                   workspace, arc) ||
      !evaluate_position(geometry, face, arc, workspace, x) ||
      !evaluate_position(geometry, face, chord, workspace, y)) return false;
  lens = distance(x, y);
  return std::isfinite(lens);
}

int32_t prepare_metric(const cad::CadGeometry& geometry, const CadCurveDiscretization& curves,
                       int32_t face, const FaceBoundary& boundary, FaceMetric& metric) {
  std::array<double, 2> low = boundary.vertices.front().uv;
  std::array<double, 2> high = low;
  metric.boundary.resize(boundary.vertices.size());
  for (std::size_t i = 0; i < boundary.vertices.size(); ++i) {
    const auto& uv = boundary.vertices[i].uv;
    for (int axis = 0; axis < 2; ++axis) {
      low[axis] = std::min(low[axis], uv[axis]);
      high[axis] = std::max(high[axis], uv[axis]);
    }
    if (!classified_position(geometry, curves, face, boundary.vertices[i], metric.workspace,
                             metric.boundary[i].xyz)) return PHX_MC_INVALID_INPUT;
    metric.boundary[i].spacing = std::numeric_limits<double>::infinity();
  }
  for (const auto& loop : boundary.loops) {
    for (std::size_t i = 0; i < loop.size(); ++i) {
      const int32_t a = loop[i], b = loop[(i + 1) % loop.size()];
      const double length = distance(metric.boundary[a].xyz, metric.boundary[b].xyz);
      if (!(length > 0.0)) {
        if (boundary.vertices[a].dimension == 0 && boundary.vertices[b].dimension == 0 &&
            boundary.vertices[a].entity == boundary.vertices[b].entity) continue;
        return PHX_MC_INVALID_INPUT;
      }
      metric.boundary[a].spacing = std::min(metric.boundary[a].spacing, length);
      metric.boundary[b].spacing = std::min(metric.boundary[b].spacing, length);
      double lens = 0.0;
      if (!chord_lens(geometry, boundary, face, a, metric.workspace, lens))
        return PHX_MC_INVALID_INPUT;
      metric.chords.push_back({metric.boundary[a].xyz, metric.boundary[b].xyz, length, lens});
    }
  }
  metric.boundary_order.reserve(metric.boundary.size());
  for (std::size_t i = 0; i < metric.boundary.size(); ++i)
    if (std::isfinite(metric.boundary[i].spacing))
      metric.boundary_order.push_back(static_cast<int32_t>(i));
  if (!metric.boundary_order.empty())
    build_boundary_tree(metric, 0, static_cast<int32_t>(metric.boundary_order.size()));
  metric.chord_order.resize(metric.chords.size());
  for (std::size_t i = 0; i < metric.chords.size(); ++i)
    metric.chord_order[i] = static_cast<int32_t>(i);
  if (!metric.chords.empty())
    build_chord_tree(metric, 0, static_cast<int32_t>(metric.chord_order.size()));
  const double center[2] = {low[0] + 0.5 * (high[0] - low[0]),
                            low[1] + 0.5 * (high[1] - low[1])};
  double jet[18];
  cad::SurfaceDifferential differential;
  if (!surface_jet(metric, center, jet) || !cad::surface_differential(jet, differential))
    return PHX_MC_INVALID_INPUT;
  const double ratio = std::sqrt(differential.first_form[0] / differential.first_form[2]);
  const int exponent = static_cast<int>(std::nearbyint(std::log2(ratio)));
  if (exponent < -512 || exponent > 512) return PHX_MC_INVALID_INPUT;
  metric.scale_u = std::ldexp(1.0, exponent);
  metric.scale_v = 1.0;
  return PHX_MC_OK;
}




bool in_pole_zone(const FaceMetric& metric, const double uv[2]) {
  for (const PoleZone& zone : metric.poles) {
    const double distance = zone.direction * (uv[zone.axis] - zone.origin);
    if (distance >= 0.0 && distance <= zone.width) return true;
  }
  return false;
}

// The kernel leaves a triangle of fixed points alone inside a zone only: a
// sphere cap's pole and trim zones face each other on one axis, and the
// triangles from one's last ring to the other's (edges to 1.44 h at cap
// below 85 deg, h/r 0.6, theta 0.9) are no zone's construction.
int32_t in_fixed_zone(void* opaque, const double point[2]) {
  const auto& metric = *static_cast<FaceMetric*>(opaque);
  const double uv[2] = {point[0] / metric.scale_u, point[1] / metric.scale_v};
  return in_pole_zone(metric, uv) ? 1 : 0;
}

int32_t admit_face_point(void* opaque, const double point[2]) {
  const auto& metric = *static_cast<FaceMetric*>(opaque);
  const double uv[2] = {point[0] / metric.scale_u, point[1] / metric.scale_v};
  for (const PoleZone& zone : metric.poles) {
    const double distance = zone.direction * (uv[zone.axis] - zone.origin);
    if (zone.pole && distance >= 0.0 && distance <= zone.spacing) return 0;
  }
  return 1;
}

bool in_trim(const FaceBoundary& boundary, std::size_t outer,
             const std::array<double, 2>& uv) {
  if (!contains(boundary, boundary.loops[outer], uv)) return false;
  for (const auto& loop : boundary.loops) {
    for (std::size_t j = 0; j < loop.size(); ++j) {
      const auto& a = boundary.vertices[loop[j]].uv;
      const auto& b = boundary.vertices[loop[(j + 1) % loop.size()]].uv;
      if (orient2d(a.data(), b.data(), uv.data()) == 0 &&
          uv[0] >= std::min(a[0], b[0]) && uv[0] <= std::max(a[0], b[0]) &&
          uv[1] >= std::min(a[1], b[1]) && uv[1] <= std::max(a[1], b[1])) return false;
    }
  }
  for (std::size_t i = 0; i < boundary.loops.size(); ++i)
    if (i != outer && contains(boundary, boundary.loops[i], uv)) return false;
  return true;
}

int near_boundary(const FaceBoundary& boundary, FaceMetric& metric,
                  const std::array<double, 2>& uv) {
  const double p[2] = {uv[0] * metric.scale_u, uv[1] * metric.scale_v};
  const CadMetric2D m = face_metric(p, &metric);
  if (!std::isfinite(m.uu) || !std::isfinite(m.uv) || !std::isfinite(m.vv)) return -1;
  const auto dot = [&](double ax, double ay, double bx, double by) {
    return ax * (m.uu * bx + m.uv * by) + ay * (m.uv * bx + m.vv * by);
  };
  // Check vertices explicitly; the following loop handles segment interiors.
  // Half a local metric unit matches the refinement kernel's proximity floor.
  for (const CadFaceVertex& vertex : boundary.vertices) {
    const double dx = p[0] - vertex.uv[0] * metric.scale_u;
    const double dy = p[1] - vertex.uv[1] * metric.scale_v;
    if (dot(dx, dy, dx, dy) < 0.25) return 1;
  }
  for (std::size_t i = 0; i < boundary.segments.size(); i += 2) {
    const auto& a = boundary.vertices[boundary.segments[i]].uv;
    const auto& b = boundary.vertices[boundary.segments[i + 1]].uv;
    const double ax = a[0] * metric.scale_u, ay = a[1] * metric.scale_v;
    const double dx = (b[0] - a[0]) * metric.scale_u;
    const double dy = (b[1] - a[1]) * metric.scale_v;
    const double denominator = dot(dx, dy, dx, dy);
    if (!(denominator > 0.0)) continue;
    const double t = dot(p[0] - ax, p[1] - ay, dx, dy) / denominator;
    if (!(t > 0.0 && t < 1.0)) continue;
    const double ex = p[0] - ax - t * dx, ey = p[1] - ay - t * dy;
    if (dot(ex, ey, ex, ey) < 0.25) return true;
  }
  return false;
}

bool point_on_segment(const std::array<double, 2>& p, const std::array<double, 2>& a,
                      const std::array<double, 2>& b) {
  return orient2d(a.data(), b.data(), p.data()) == 0 &&
         p[0] >= std::min(a[0], b[0]) && p[0] <= std::max(a[0], b[0]) &&
         p[1] >= std::min(a[1], b[1]) && p[1] <= std::max(a[1], b[1]);
}

bool segments_meet(const std::array<double, 2>& a, const std::array<double, 2>& b,
                   const std::array<double, 2>& c, const std::array<double, 2>& d) {
  const double ab_c = orient2d(a.data(), b.data(), c.data());
  const double ab_d = orient2d(a.data(), b.data(), d.data());
  const double cd_a = orient2d(c.data(), d.data(), a.data());
  const double cd_b = orient2d(c.data(), d.data(), b.data());
  if (ab_c == 0 && point_on_segment(c, a, b)) return true;
  if (ab_d == 0 && point_on_segment(d, a, b)) return true;
  if (cd_a == 0 && point_on_segment(a, c, d)) return true;
  if (cd_b == 0 && point_on_segment(b, c, d)) return true;
  return ((ab_c > 0 && ab_d < 0) || (ab_c < 0 && ab_d > 0)) &&
         ((cd_a > 0 && cd_b < 0) || (cd_a < 0 && cd_b > 0));
}

bool point_in_triangle(const std::array<double, 2>& p, const std::array<double, 2>& a,
                       const std::array<double, 2>& b, const std::array<double, 2>& c) {
  const double x = orient2d(a.data(), b.data(), p.data());
  const double y = orient2d(b.data(), c.data(), p.data());
  const double z = orient2d(c.data(), a.data(), p.data());
  return ((x >= 0 && y >= 0 && z >= 0) || (x <= 0 && y <= 0 && z <= 0)) &&
         orient2d(a.data(), b.data(), c.data()) != 0;
}

}  // namespace

bool cad_face_segments_conflict(int32_t ai, int32_t bi, const std::array<double, 2>& a,
                                const std::array<double, 2>& b, int32_t ci, int32_t di,
                                const std::array<double, 2>& c, const std::array<double, 2>& d) {
  if (!segments_meet(a, b, c, d)) return false;
  if ((ai == ci && bi == di) || (ai == di && bi == ci)) return false;
  if (ai != ci && ai != di && bi != ci && bi != di) return true;
  // A shared endpoint is permitted only when it is the entire intersection.
  // Checking the other endpoints rejects collinear overlap and T-junctions.
  if (ai != ci && ai != di && point_on_segment(a, c, d)) return true;
  if (bi != ci && bi != di && point_on_segment(b, c, d)) return true;
  if (ci != ai && ci != bi && point_on_segment(c, a, b)) return true;
  if (di != ai && di != bi && point_on_segment(d, a, b)) return true;
  return false;
}

namespace {

bool chord_meets_triangle(const FaceBoundary& boundary, const std::array<int32_t, 3>& ids,
                          const std::array<std::array<double, 2>, 3>& uv) {
  for (const CadFaceEdge& edge : boundary.edges) {
    if (edge.curve < 0) continue;
    const int32_t p = edge.vertices[0], q = edge.vertices[1];
    const auto& puv = boundary.vertices[p].uv;
    const auto& quv = boundary.vertices[q].uv;
    for (int i = 0; i < 3; ++i) {
      const int j = (i + 1) % 3;
      if (cad_face_segments_conflict(ids[i], ids[j], uv[i], uv[j], p, q, puv, quv)) return true;
    }
    if (p != ids[0] && p != ids[1] && p != ids[2] &&
        point_in_triangle(puv, uv[0], uv[1], uv[2])) return true;
    if (q != ids[0] && q != ids[1] && q != ids[2] &&
        point_in_triangle(quv, uv[0], uv[1], uv[2])) return true;
  }
  return false;
}

bool chord_meets_segment(const FaceBoundary& boundary, int32_t a, int32_t b,
                         const std::array<double, 2>& auv, const std::array<double, 2>& buv) {
  for (const CadFaceEdge& edge : boundary.edges) {
    if (edge.curve < 0) continue;
    const int32_t p = edge.vertices[0], q = edge.vertices[1];
    if (cad_face_segments_conflict(a, b, auv, buv, p, q, boundary.vertices[p].uv,
                                   boundary.vertices[q].uv)) return true;
  }
  return false;
}

bool pole_snap_latitude(const FaceBoundary& boundary, const PoleBoundary& pole, int axis,
                        double origin, double direction, double spacing, double width,
                        double& latitude) {
  const int32_t corner = boundary.vertices[pole.first_vertex].entity;
  bool found = false;
  Expansion common;
  for (const CadFaceEdge& edge : boundary.edges) {
    if (edge.curve < 0) continue;
    for (int k = 0; k < 2; ++k) {
      const auto& endpoint = boundary.vertices[edge.vertices[k]];
      if (endpoint.dimension != 0 || endpoint.entity != corner) continue;
      const double adjacent = boundary.vertices[edge.vertices[1-k]].uv[axis];
      const double offset = direction * (adjacent - origin);
      if (!(offset >= 0.5 * spacing && offset <= spacing && offset <= width)) return false;
      const Expansion exact = Expansion::difference(adjacent, origin);
      if (found && !(exact - common).is_zero()) return false;
      common = exact;
      latitude = adjacent;
      found = true;
    }
  }
  return found;
}

// A trim line's first ring takes the latitude of the boundary vertices next
// to the line's corners off it (a seam's first vertex) when all lie at one
// exact offset within a spacing, so the ring chain ends on them and closes
// the band between ring and trim.
bool trim_snap_latitude(const FaceBoundary& boundary, int axis, double origin, double direction,
                        double spacing, double width, double& latitude) {
  bool found = false;
  Expansion common;
  for (const CadFaceEdge& edge : boundary.edges) {
    if (edge.curve < 0) continue;
    for (int k = 0; k < 2; ++k) {
      const auto& corner = boundary.vertices[static_cast<std::size_t>(edge.vertices[k])];
      const auto& adjacent = boundary.vertices[static_cast<std::size_t>(edge.vertices[1 - k])];
      if (corner.dimension != 0 || corner.uv[axis] != origin || adjacent.uv[axis] == origin)
        continue;
      const double offset = direction * (adjacent.uv[axis] - origin);
      if (!(offset > 0.0 && offset <= spacing && offset <= width)) return false;
      const Expansion exact = Expansion::difference(adjacent.uv[axis], origin);
      if (found && !(exact - common).is_zero()) return false;
      common = exact;
      latitude = adjacent.uv[axis];
      found = true;
    }
  }
  return found;
}

int32_t seed_poles(const cad::CadGeometry& geometry, const FaceBoundary& boundary,
                   std::size_t outer, FaceMetric& metric, std::vector<double>& points,
                   std::vector<uint8_t>& fixed, std::vector<int32_t>& segments,
                   std::vector<CadPoleSectors>& sectors,
                   std::vector<std::array<int32_t, 2>>& protected_edges, int64_t& omitted) {
  std::array<double, 2> low = boundary.vertices.front().uv, high = low;
  for (const CadFaceVertex& vertex : boundary.vertices)
    for (int axis = 0; axis < 2; ++axis) {
      low[axis] = std::min(low[axis], vertex.uv[axis]);
      high[axis] = std::max(high[axis], vertex.uv[axis]);
    }
  // Zone anchors: each chart pole, and each polar trim line. A polar trim
  // line is a face box side along a periodic free axis that boundary curve
  // edges cover over one whole period and where the free axis's chart speed
  // is under half the zone axis's (the ratio test that ends a zone): a trim
  // circle near a sphere's cut-off pole. The chart there is as anisotropic
  // as beside a pole, and the kernel met neither size nor encroachment rules
  // in the band between the trim and the rest (measured: caps below 75-85
  // deg, h/r 0.4-0.8, surface UNRESOLVED); its rings take over as at a pole.
  struct Anchor {
    const PoleBoundary* pole;  // nullptr: a polar trim line.
    int axis;
    std::array<double, 2> start;
    std::array<double, 2> end;  // Trim line: the free axis's far end.
  };
  std::vector<Anchor> anchors;
  for (const PoleBoundary& pole : boundary.poles) {
    const auto& start = boundary.vertices[pole.first_vertex].uv;
    const auto& almost_end = boundary.vertices[pole.first_vertex + pole.count - 1].uv;
    const int axis = std::abs(start[0] - almost_end[0]) <
                             std::abs(start[1] - almost_end[1]) ? 0 : 1;
    anchors.push_back({&pole, axis, start, start});
  }
  const auto& model = geometry.model();
  for (int axis = 0; axis < 2; ++axis) {
    const int free_axis = 1 - axis;
    int64_t turns = 0;
    if (exact_turn(low[free_axis], high[free_axis], model.surface_periods[2 * metric.face + free_axis],
                   turns) != 1 || turns != 1) continue;
    for (const double line : {low[axis], high[axis]}) {
      std::vector<std::array<double, 2>> spans;
      for (const CadFaceEdge& edge : boundary.edges) {
        if (edge.curve < 0) continue;
        const auto& a = boundary.vertices[static_cast<std::size_t>(edge.vertices[0])].uv;
        const auto& b = boundary.vertices[static_cast<std::size_t>(edge.vertices[1])].uv;
        if (a[axis] != line || b[axis] != line) continue;
        spans.push_back({std::min(a[free_axis], b[free_axis]), std::max(a[free_axis], b[free_axis])});
      }
      std::sort(spans.begin(), spans.end());
      double reach = low[free_axis];
      for (const auto& span : spans)
        if (span[0] == reach) reach = span[1];
      if (spans.empty() || reach != high[free_axis]) continue;
      double uv[2];
      uv[axis] = line;
      uv[free_axis] = 0.5 * (low[free_axis] + high[free_axis]);
      double jet[18];
      if (!surface_jet(metric, uv, jet)) return PHX_MC_INVALID_INPUT;
      const double scale[2] = {metric.scale_u, metric.scale_v};
      const auto speed2 = [&](int k) {
        const double* d = jet + 3 + 3 * k;
        return (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (scale[k] * scale[k]);
      };
      if (!(speed2(free_axis) < 0.5 * speed2(axis))) continue;
      std::array<double, 2> start{}, end{};
      start[axis] = end[axis] = line;
      start[free_axis] = low[free_axis];
      end[free_axis] = high[free_axis];
      anchors.push_back({nullptr, axis, start, end});
    }
  }
  for (const Anchor& anchor : anchors) {
    const auto& start = anchor.start;
    const int axis = anchor.axis;
    const int free_axis = 1 - axis;
    const double direction = (low[axis] + high[axis]) * 0.5 > start[axis] ? 1.0 : -1.0;
    double limit = direction > 0.0 ? high[axis] - start[axis] : start[axis] - low[axis];
    if (!(limit > 0.0)) return PHX_MC_INVALID_INPUT;
    // Half the distance to the nearest pole ahead on this axis (a sphere's
    // other pole). Both zones extend toward each other, and without this bound
    // each could end with a ring at the common midline: on the sphere at
    // h/r = 0.6 and theta 0.8, two rings 2 ulps apart at the equator, whose
    // zero-area strip left a boundary that tetrahedral improvement churned
    // on for minutes.
    double half_gap = std::numeric_limits<double>::infinity();
    for (const Anchor& other : anchors) {
      const double gap = direction * (other.start[axis] - start[axis]);
      if (gap > 0.0) half_gap = std::min(half_gap, 0.5 * gap);
    }
    limit = std::min(limit, half_gap);
    const double middle[2] = {0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])};
    double jet[18];
    if (!surface_jet(metric, middle, jet)) return PHX_MC_INVALID_INPUT;
    const int derivative = axis == 0 ? 3 : 6;
    const double speed = std::hypot(std::hypot(jet[derivative], jet[derivative + 1]),
                                    jet[derivative + 2]);
    if (!(speed > 0.0)) return PHX_MC_INVALID_INPUT;
    // The rings are isotropic: their spacing is the face metric's smallest
    // size, the request h or the boundary layer's fidelity size
    // sqrt(4 (theta - lens / tol) tol / |k|max) (local_theta) with the
    // largest lens of the curve chords at the pole: the curves that end at
    // the pole converge there, so its zone lies in their layer (measured:
    // rings at theta's size beside lens-carrying chords left the sphere's
    // kernel triangles between them unresolved; rings at the layer size
    // beside a chart-straight seam left a 16 deg fan triangle between the
    // first ring and the seam's request-sized pole segment).
    // A trim line's zone lies in the layer of its own chords.
    double lens = 0.0;
    for (std::size_t e = 0; e < boundary.edges.size(); ++e) {
      const CadFaceEdge& edge = boundary.edges[e];
      if (edge.curve < 0) continue;
      bool at_anchor = !anchor.pole;
      for (const int32_t v : edge.vertices) {
        const CadFaceVertex& vertex = boundary.vertices[static_cast<std::size_t>(v)];
        if (anchor.pole)
          at_anchor = at_anchor || (vertex.dimension == 0 &&
                                    vertex.entity == boundary.vertices[anchor.pole->first_vertex].entity);
        else
          at_anchor = at_anchor && vertex.uv[axis] == start[axis];
      }
      double edge_lens = 0.0;
      if (at_anchor && !chord_lens(geometry, boundary, metric.face, static_cast<int32_t>(e),
                                   metric.workspace, edge_lens)) return PHX_MC_INVALID_INPUT;
      lens = std::max(lens, edge_lens);
    }
    const double layer = metric.theta - std::min(lens / metric.tolerance,
                                                 kLensShare * metric.theta);
    double curvature = 0.0;
    if (!pole_zone_curvature(geometry, metric.face, metric.workspace, curvature))
      return PHX_MC_INVALID_INPUT;
    // Solid route (graded_pole_rings): a pole's rings start from the face's
    // graded base size at the pole (base_size), not the bare request: a
    // boundary there finer than h (a seam graded from a wedge-graded rim,
    // cad_curves.cpp) otherwise left a zone as wide as the face with no
    // ring, a fan of the boundary (measured: cap above 70 deg, h/r 0.4: a
    // pole triangle across two seam points, flat). A trim line keeps the
    // request: its first ring snaps to the seam's first vertex off the line
    // (trim_snap_latitude), and rings at the trim's own spacing left the dome
    // of caps below 80-85 deg UNRESOLVED. The surface route has no
    // wedge-graded rim and keeps the request.
    double size = metric.size;
    if (anchor.pole && metric.graded_pole_rings) {
      double pole_jet[18];
      if (!surface_jet(metric, start.data(), pole_jet)) return PHX_MC_INVALID_INPUT;
      size = std::min(size, base_size(metric, pole_jet));
    }
    if (curvature > 0.0)
      size = std::min(size, std::sqrt(4.0 * layer * metric.tolerance / curvature));
    double spacing = std::min(limit, size / speed);
    if (!(spacing > 0.0)) return PHX_MC_INVALID_INPUT;
    // The last ring stays half a spacing short of the midline, so the two
    // zones' last rings are at least a spacing apart; a spacing of at most
    // 2/3 of the half gap keeps one ring.
    if (std::isfinite(half_gap)) {
      spacing = std::min(spacing, half_gap * (2.0 / 3.0));
      limit = half_gap - 0.5 * spacing;
    }
    double width = std::min(limit, pole_ring_width(spacing));
    while (width < limit) {
      double uv[2] = {middle[0], middle[1]};
      uv[axis] = start[axis] + direction * width;
      if (!surface_jet(metric, uv, jet)) return PHX_MC_INVALID_INPUT;
      const double e = jet[3] * jet[3] + jet[4] * jet[4] + jet[5] * jet[5];
      const double g = jet[6] * jet[6] + jet[7] * jet[7] + jet[8] * jet[8];
      const double ratio = (e / (metric.scale_u * metric.scale_u)) /
                           (g / (metric.scale_v * metric.scale_v));
      if (ratio >= 0.5 && ratio <= 2.0) break;
      const double next = std::min(limit, width + spacing);
      if (!(next > width)) return PHX_MC_RANGE_ERROR;
      width = next;
    }
    double latitude = 0.0;
    const bool snapped = anchor.pole
        ? pole_snap_latitude(boundary, *anchor.pole, axis, start[axis], direction, spacing, width,
                             latitude)
        : trim_snap_latitude(boundary, axis, start[axis], direction, spacing, width, latitude);
    if (snapped) spacing = direction * (latitude - start[axis]);
    metric.poles.push_back({axis, start[axis], direction, width, spacing, anchor.pole != nullptr});
    std::array<double, 2> end = anchor.end;
    if (anchor.pole) {
      const auto& pole = *anchor.pole;
      const double first = model.use_ranges[2 * pole.use];
      const double last = model.use_ranges[2 * pole.use + 1];
      if (!chart_point(geometry, pole.use, model.use_senses[pole.use] > 0 ? last : first,
                       metric.workspace, end)) return PHX_MC_INVALID_INPUT;
    }
    // Solid route: each ring's point count and the step to the next ring
    // follow the graded base size at the ring, at most the zone's spacing:
    // rings beside a boundary finer than the request (a wedge-graded rim)
    // start as fine and grow at the grading rate instead of joining a fine
    // side to a coarse ring (measured: caps above 45-65 deg, solid scan 90
    // of 195 runs UNRESOLVED with rings at the zone spacing, 15 with graded
    // rings).
    double step = spacing;
    for (double offset = spacing; offset <= width; offset += step) {
      std::array<double, 2> uv = start;
      uv[axis] = snapped && offset == spacing ? latitude : start[axis] + direction * offset;
      if (!snapped && offset == spacing && direction * (uv[axis] - start[axis]) < spacing)
        uv[axis] = std::nextafter(uv[axis], direction > 0.0
                                  ? std::numeric_limits<double>::infinity()
                                  : -std::numeric_limits<double>::infinity());
      if (metric.graded_pole_rings) {
        double ring_middle[2];
        ring_middle[axis] = uv[axis];
        ring_middle[free_axis] = middle[free_axis];
        if (!surface_jet(metric, ring_middle, jet)) return PHX_MC_INVALID_INPUT;
        step = std::min(spacing, base_size(metric, jet) / speed);
      }
      std::array<double, 3> previous;
      if (!evaluate_position(geometry, metric.face, uv, metric.workspace, previous))
        return PHX_MC_INVALID_INPUT;
      double length = 0.0;
      for (int i = 1; i <= 16; ++i) {
        uv[free_axis] = start[free_axis] + (end[free_axis] - start[free_axis]) * i / 16.0;
        std::array<double, 3> next;
        if (!evaluate_position(geometry, metric.face, uv, metric.workspace, next))
          return PHX_MC_INVALID_INPUT;
        length += distance(previous, next);
        previous = next;
      }
      const int count = std::max(8, static_cast<int>(std::ceil(length / (step * speed))));
      const bool first_ring = offset == spacing;
      std::vector<std::array<double, 2>> ring_uv;
      std::vector<int32_t> ring_rows;
      if (first_ring) {
        ring_uv.resize(static_cast<std::size_t>(count + 1));
        ring_rows.assign(static_cast<std::size_t>(count + 1), -1);
      }
      for (int i = 0; i <= count; ++i) {
        uv[free_axis] = i == count ? end[free_axis] :
            start[free_axis] + (end[free_axis] - start[free_axis]) * i / count;
        if (first_ring) ring_uv[static_cast<std::size_t>(i)] = uv;
        if (i == 0 || i == count) {
          if (first_ring && snapped) {
            for (std::size_t j = 0; j < boundary.vertices.size(); ++j)
              if (boundary.vertices[j].uv == uv) {
                ring_rows[static_cast<std::size_t>(i)] = static_cast<int32_t>(j);
                break;
              }
          }
          continue;
        }
        if (!in_trim(boundary, outer, uv)) continue;
        const int proximity = near_boundary(boundary, metric, uv);
        if (proximity < 0) return PHX_MC_INVALID_INPUT;
        if (proximity > 0) {
          ++omitted;
          continue;
        }
        if (first_ring) ring_rows[static_cast<std::size_t>(i)] = static_cast<int32_t>(points.size()/2);
        points.push_back(uv[0] * metric.scale_u);
        points.push_back(uv[1] * metric.scale_v);
        fixed.push_back(1);
      }
      if (first_ring && !anchor.pole) {
        // A trim line's first ring chain closes the band between it and the
        // trim, whose triangles then join fixed points only, as a pole's fan;
        // open, triangles from the trim to kernel points beyond the ring
        // asked for points in the band, which admission refuses.
        for (int i = 0; i < count; ++i) {
          const int32_t a = ring_rows[static_cast<std::size_t>(i)];
          const int32_t b = ring_rows[static_cast<std::size_t>(i+1)];
          if (a < 0 || b < 0 || chord_meets_segment(boundary, a, b, ring_uv[i], ring_uv[i+1]))
            continue;
          segments.push_back(a);
          segments.push_back(b);
          protected_edges.push_back({a, b});
        }
      } else if (first_ring) {
        const PoleBoundary& pole = *anchor.pole;
        CadPoleSectors counts{static_cast<int32_t>(pole.use), 0, 0};
        const auto add_segment = [&](int32_t a, int32_t b) {
          for (std::size_t j = 0; j < segments.size(); j += 2)
            if ((segments[j] == a && segments[j+1] == b) ||
                (segments[j] == b && segments[j+1] == a)) return;
          segments.push_back(a);
          segments.push_back(b);
        };
        const auto pole_row = [&](int i) {
          if (i == count)
            return boundary.edges[static_cast<std::size_t>(pole.first_vertex + pole.count - 1)].vertices[1];
          return pole.first_vertex + i * pole.count / count;
        };
        for (int i = 0; i < count; ++i) {
          const int32_t a = ring_rows[static_cast<std::size_t>(i)];
          const int32_t b = ring_rows[static_cast<std::size_t>(i+1)];
          if (a < 0 || b < 0 || chord_meets_segment(boundary, a, b, ring_uv[i], ring_uv[i+1])) {
            ++counts.free;
            continue;
          }
          add_segment(a, b);
          const int32_t p = pole_row(i), q = pole_row(i+1);
          const auto& puv = boundary.vertices[p].uv;
          const auto& quv = boundary.vertices[q].uv;
          if (chord_meets_triangle(boundary, {p, a, b}, {puv, ring_uv[i], ring_uv[i+1]}) ||
              (p != q && chord_meets_triangle(boundary, {p, b, q}, {puv, ring_uv[i+1], quv}))) {
            ++counts.free;
            continue;
          }
          add_segment(p, a);
          add_segment(q, b);
          if (p != q) add_segment(p, b);
          protected_edges.push_back({a, b});
          ++counts.fixed;
        }
        sectors.push_back(counts);
      }
    }
  }
  for (std::size_t i = 0; i < boundary.vertices.size(); ++i)
    if (in_pole_zone(metric, boundary.vertices[i].uv.data())) fixed[i] = 1;
  return PHX_MC_OK;
}

double maximum_cosine(const SurfaceTriangulation& surface,
                      const std::vector<int32_t>& cells) {
  double maximum = -1.0;
  for (const int32_t cell : cells) {
    const auto& t = surface.cells[cell];
    for (int i = 0; i < 3; ++i) {
      const auto p = surface.point(t[i]);
      const auto a = surface.point(t[(i + 1) % 3]) - p;
      const auto b = surface.point(t[(i + 2) % 3]) - p;
      const double denominator = std::sqrt(geometry::dot(a, a) * geometry::dot(b, b));
      if (!(denominator > 0.0)) return std::numeric_limits<double>::infinity();
      maximum = std::max(maximum, geometry::dot(a, b) / denominator);
    }
  }
  return maximum;
}

// Largest metric circumradius over cells (centroid metric, as the chart
// kernel's size test); infinity on a degenerate or invalid cell. Stops at the
// first cell past `bound`, returning that cell's radius.
double maximum_metric_radius(FaceMetric& metric, const SurfaceTriangulation& surface,
                             const std::vector<int32_t>& cells, double bound) {
  double maximum = 0.0;
  for (const int32_t cell : cells) {
    const auto& t = surface.cells[cell];
    const double* a = surface.chart(t[0]);
    const double* b = surface.chart(t[1]);
    const double* c = surface.chart(t[2]);
    const double centroid[2] = {(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0};
    const CadMetric2D m = face_metric(centroid, &metric);
    const double x = b[0] - a[0], y = b[1] - a[1], u = c[0] - a[0], v = c[1] - a[1];
    const double bx = m.uu * x + m.uv * y, by = m.uv * x + m.vv * y;
    const double cx = m.uu * u + m.uv * v, cy = m.uv * u + m.vv * v;
    const double bb = x * bx + y * by, cc = u * cx + v * cy;
    const double determinant = bx * cy - by * cx;
    const double dx = (bb * cy - cc * by) / (2 * determinant);
    const double dy = (bx * cc - cx * bb) / (2 * determinant);
    const double radius2 = dx * (m.uu * dx + m.uv * dy) + dy * (m.uv * dx + m.vv * dy);
    if (!std::isfinite(radius2)) return std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, radius2);
    if (std::sqrt(radius2) > bound) break;
  }
  return std::sqrt(maximum);
}

// Moves of free interior vertices, each guarded: the chart star stays
// positively oriented and keeps the kernel's size bound (metric circumradius
// <= 1/sqrt(2), on which the fidelity bound rests, or a star already past it).
// 1. The inverse-distance neighbour centroid (first two passes), kept when
//    the star's worst 3D angle does not drop.
// 2. Below the kernel's angle-pass target (halfway between
//    min_angle_degrees and 60), a compass search on the star's worst 3D
//    angle: eight chart directions from a quarter of the shortest incident
//    chart edge, halved four times, best strict improvement per step. This
//    lifts the fans at fixed curve vertices, which hold the smallest angles
//    after refinement (measured on the quadric: every triangle below Gmsh's
//    minimum had its smallest angle at a curve vertex).
// 3. While the longest incident edge exceeds one in the face metric (the
//    size h; its sag is the mesh-to-surface distance), a step of 1/4, 1/8 or
//    1/16 toward its far end, the first that shortens it without the worst
//    angle dropping below the target or its current value.
// Passes revisit the vertices that moved or neighbour a move, until none
// moves (at most eight).
void smooth(FaceMetric& metric, const std::vector<uint8_t>& fixed, double min_angle_degrees,
            std::vector<CadFaceVertex>& vertices, SurfaceTriangulation& surface) {
  std::vector<std::vector<int32_t>> incident(vertices.size());
  for (std::size_t cell = 0; cell < surface.cells.size(); ++cell) {
    for (const int32_t v : surface.cells[cell]) incident[v].push_back(static_cast<int32_t>(cell));
  }
  const double target_cos =
      std::cos((min_angle_degrees + 60.0) * 3.14159265358979323846 / 360.0);
  // The visited vertex's chart at the visit's start.
  std::array<double, 2> visit_start{};
  // The size guard for v at its current chart: false when the star's radius
  // exceeds both the kernel's bound and the star's radius at the visit start.
  // `previous_radius` (that start radius) < 0 is computed on first need;
  // infinity (a degenerate star) skips the guard.
  const auto fits = [&](std::size_t v, double& previous_radius) {
    if (!std::isfinite(previous_radius)) return true;
    constexpr double bound = 0.70710678118654752440;
    // Within the kernel's bound the move passes whatever the start was, so
    // the start radius is evaluated only past it.
    double radius = maximum_metric_radius(
        metric, surface, incident[v],
        previous_radius < 0.0 ? bound : std::max(bound, previous_radius));
    if (!(radius > bound)) return true;
    if (previous_radius < 0.0) {
      // The star's radius at the visit's start (no move is kept before the
      // first size check).
      const std::array<double, 2> chart{surface.charts[2 * v], surface.charts[2 * v + 1]};
      std::copy(visit_start.begin(), visit_start.end(), surface.charts.begin() + 2 * v);
      previous_radius = maximum_metric_radius(metric, surface, incident[v],
                                              std::numeric_limits<double>::infinity());
      std::copy(chart.begin(), chart.end(), surface.charts.begin() + 2 * v);
      if (!std::isfinite(previous_radius)) return true;
      if (previous_radius > bound)
        radius = maximum_metric_radius(metric, surface, incident[v], previous_radius);
    }
    return !(radius > std::max(bound, previous_radius));
  };
  // Moves v to `chart` when the orientation holds and the star's worst cosine
  // is at most `limit` (and, with `sized`, the size guard holds); returns the
  // new worst cosine, or infinity (v unchanged).
  const auto attempt = [&](std::size_t v, const double chart[2], double limit,
                           double& previous_radius, bool sized) {
    const std::array<double, 2> old_chart{surface.charts[2 * v], surface.charts[2 * v + 1]};
    const std::array<double, 3> old_point{surface.points[3 * v], surface.points[3 * v + 1],
                                          surface.points[3 * v + 2]};
    const auto reject = [&] {
      std::copy(old_chart.begin(), old_chart.end(), surface.charts.begin() + 2 * v);
      std::copy(old_point.begin(), old_point.end(), surface.points.begin() + 3 * v);
      return std::numeric_limits<double>::infinity();
    };
    // The chart orientation needs no surface evaluation: checked first.
    std::copy_n(chart, 2, surface.charts.begin() + 2 * v);
    for (const int32_t cell : incident[v]) {
      const auto& t = surface.cells[cell];
      if (orient2d(surface.chart(t[0]), surface.chart(t[1]), surface.chart(t[2])) <= 0)
        return reject();
    }
    std::array<double, 3> xyz;
    if (!face_position(&metric, chart, xyz.data())) return reject();
    std::copy(xyz.begin(), xyz.end(), surface.points.begin() + 3 * v);
    const double quality = maximum_cosine(surface, incident[v]);
    if (!(quality <= limit)) return reject();
    if (sized && !fits(v, previous_radius)) return reject();
    vertices[v].uv = {chart[0] / metric.scale_u, chart[1] / metric.scale_v};
    return quality;
  };
  // Face metric at each vertex as a 3D tensor (vertex_metric) for the edge
  // lengths of the long-edge step; refreshed when the vertex moves.
  std::vector<double> tensors(9 * vertices.size(), 0.0);
  const auto refresh = [&](std::size_t v) {
    double* tensor = tensors.data() + 9 * v;
    const double isotropic = 1.0 / (metric.size * metric.size);
    for (int k = 0; k < 9; ++k) tensor[k] = k % 4 == 0 ? isotropic : 0.0;
    vertex_metric(metric, surface.chart(static_cast<int32_t>(v)), tensor);
  };
  for (std::size_t v = 0; v < vertices.size(); ++v) refresh(v);
  std::vector<int32_t> neighbors;
  // A vertex is visited while it or a neighbour moved in the previous pass.
  std::vector<uint8_t> active(vertices.size(), 1), next(vertices.size(), 0);
  for (int pass = 0; pass < 8; ++pass) {
    bool moved = false;
    std::fill(next.begin(), next.end(), 0);
    for (std::size_t v = 0; v < vertices.size(); ++v) {
      if (vertices[v].dimension != 2 || fixed[v] || incident[v].empty() || !active[v]) continue;
      const std::array<double, 2> start{surface.charts[2 * v], surface.charts[2 * v + 1]};
      visit_start = start;
      [&] {
      neighbors.clear();
      for (const int32_t cell : incident[v])
        for (const int32_t u : surface.cells[cell])
          if (u != static_cast<int32_t>(v)) neighbors.push_back(u);
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
      const auto p = surface.point(static_cast<int32_t>(v));
      double sum = 0.0, chart[2] = {}, step = std::numeric_limits<double>::infinity();
      const double* own = surface.chart(static_cast<int32_t>(v));
      for (const int32_t u : neighbors) {
        const auto d = surface.point(u) - p;
        const double length = std::sqrt(geometry::dot(d, d));
        const double* q = surface.chart(u);
        step = std::min(step, std::hypot(q[0] - own[0], q[1] - own[1]));
        if (!(length > 0.0)) continue;
        const double weight = 1.0 / length;
        chart[0] += weight * q[0];
        chart[1] += weight * q[1];
        sum += weight;
      }
      if (!(sum > 0.0)) return;
      chart[0] /= sum;
      chart[1] /= sum;
      double quality = maximum_cosine(surface, incident[v]);
      double previous_radius = -1.0;
      // The first two passes smooth every vertex; later passes only search.
      if (pass < 2) {
        const double after = attempt(v, chart, quality, previous_radius, true);
        if (after <= quality) quality = after;
      }
      // Compass search on the star's worst cosine from a quarter of the
      // shortest incident chart edge, halved four times, over moves whose
      // worst cosine stays <= `limit`; v returns to its saved state between
      // probes.
      const double initial_step = 0.25 * step;
      double limit = std::numeric_limits<double>::infinity();
      std::array<double, 2> saved_chart{}, saved_uv{};
      std::array<double, 3> saved_point{};
      const auto save = [&] {
        saved_chart = {surface.charts[2 * v], surface.charts[2 * v + 1]};
        saved_point = {surface.points[3 * v], surface.points[3 * v + 1], surface.points[3 * v + 2]};
        saved_uv = vertices[v].uv;
      };
      const auto restore = [&] {
        std::copy(saved_chart.begin(), saved_chart.end(), surface.charts.begin() + 2 * v);
        std::copy(saved_point.begin(), saved_point.end(), surface.points.begin() + 3 * v);
        vertices[v].uv = saved_uv;
      };
      const auto search = [&] {
        double current = quality;
        double width = initial_step;
        for (int level = 0; level < 5; ++level, width *= 0.5) {
          // The move just made at this width: probing back along it returns
          // to the worse previous position.
          int back_x = 0, back_y = 0;
          for (;;) {
            save();
            double best = current;
            std::array<double, 2> best_chart{}, best_uv{};
            std::array<double, 3> best_point{};
            int best_x = 0, best_y = 0;
            for (int dx = -1; dx <= 1; ++dx)
              for (int dy = -1; dy <= 1; ++dy) {
                if ((dx == 0 && dy == 0) || (dx == back_x && dy == back_y && (dx || dy)))
                  continue;
                const double probe[2] = {saved_chart[0] + dx * width, saved_chart[1] + dy * width};
                // Only a strictly better probe can be kept, so the costlier
                // size guard runs only for those.
                const double after = attempt(
                    v, probe,
                    std::min(limit, std::nextafter(best, -std::numeric_limits<double>::infinity())),
                    previous_radius, true);
                if (!std::isfinite(after)) continue;
                best = after;
                best_chart = {probe[0], probe[1]};
                best_point = {surface.points[3 * v], surface.points[3 * v + 1],
                              surface.points[3 * v + 2]};
                best_uv = vertices[v].uv;
                best_x = dx;
                best_y = dy;
                restore();
              }
            if (!(best < current)) break;
            std::copy(best_chart.begin(), best_chart.end(), surface.charts.begin() + 2 * v);
            std::copy(best_point.begin(), best_point.end(), surface.points.begin() + 3 * v);
            vertices[v].uv = best_uv;
            current = best;
            back_x = -best_x;
            back_y = -best_y;
          }
        }
        return current;
      };
      if (!(step > 0.0) || !std::isfinite(step)) return;
      // First the worst 3D angle, while it is below the target.
      if (quality > target_cos) quality = search();
      // Then the longest incident edge in the face metric while it exceeds
      // one (the size h): its sag is what the distance row measures. The
      // worst angle may not drop below the target or its current value.
      int32_t far = -1;
      const auto longest = [&] {
        double maximum = 0.0;
        const auto here = surface.point(static_cast<int32_t>(v));
        const double* s = tensors.data() + 9 * v;
        for (const int32_t u : neighbors) {
          const auto e = surface.point(u) - here;
          const double d[3] = {e.x, e.y, e.z};
          const double* t = tensors.data() + 9 * static_cast<std::size_t>(u);
          double length2 = 0.0;
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) length2 += 0.5 * (s[3 * i + j] + t[3 * i + j]) * d[i] * d[j];
          if (!std::isfinite(length2)) length2 = std::numeric_limits<double>::infinity();
          if (length2 > maximum) maximum = length2, far = u;
        }
        return maximum;
      };
      // A step toward the far end of the longest edge (1/4, 1/8, 1/16 of
      // it), the first that shortens the longest edge (checked before the
      // costlier size guard).
      const double current = longest();
      if (!(current > 1.0) || far < 0) return;
      limit = std::max(quality, target_cos);
      save();
      const double* target = surface.chart(far);
      for (double fraction = 0.25; fraction >= 0.0625; fraction *= 0.5) {
        const double probe[2] = {saved_chart[0] + fraction * (target[0] - saved_chart[0]),
                                 saved_chart[1] + fraction * (target[1] - saved_chart[1])};
        if (!std::isfinite(attempt(v, probe, limit, previous_radius, false))) continue;
        if (longest() < current && fits(v, previous_radius)) break;
        restore();
      }
      }();
      if (surface.charts[2 * v] == start[0] && surface.charts[2 * v + 1] == start[1]) continue;
      moved = true;
      refresh(v);
      next[v] = 1;
      for (const int32_t u : neighbors) next[static_cast<std::size_t>(u)] = 1;
    }
    if (!moved && pass >= 1) break;
    // The second pass smooths every vertex again.
    if (pass == 0) std::fill(active.begin(), active.end(), 1);
    else active.swap(next);
  }
}

}  // namespace

bool pole_zone_curvature(const cad::CadGeometry& geometry, int32_t face,
                         cad::EvaluationWorkspace& workspace, double& kappa) {
  const auto& model = geometry.model();
  const double inf = std::numeric_limits<double>::infinity();
  double low[2] = {inf, inf}, high[2] = {-inf, -inf};
  for (int64_t loop = 0; loop < model.loop_count; ++loop) {
    if (model.loop_surfaces[loop] != face) continue;
    for (int64_t use = model.loop_use_offsets[loop]; use < model.loop_use_offsets[loop + 1]; ++use)
      for (int k = 0; k < 2; ++k) {
        std::array<double, 2> uv;
        if (!chart_point(geometry, use, model.use_ranges[2 * use + k], workspace, uv)) return false;
        for (int axis = 0; axis < 2; ++axis) {
          low[axis] = std::min(low[axis], uv[axis]);
          high[axis] = std::max(high[axis], uv[axis]);
        }
      }
  }
  const double middle[2] = {0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])};
  double jet[18];
  cad::SurfaceDifferential differential;
  if (!geometry.jets(geometry.surface_node(face), middle, 2, jet, workspace) ||
      !cad::surface_differential(jet, differential)) return false;
  kappa = std::max(std::abs(differential.curvatures[0]), std::abs(differential.curvatures[1]));
  return std::isfinite(kappa);
}

int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result, std::vector<std::array<double, 2>>* metric_points,
                      std::vector<std::array<double, 5>>* position_calls,
                      bool graded_pole_rings) {
  if (face < 0 || face >= geometry.surface_count() || !(size > 0.0) ||
      !(tolerance > 0.0) || !(theta > 0.0) || !std::isfinite(size) ||
      !std::isfinite(tolerance) || !std::isfinite(theta)) return PHX_MC_INVALID_ARGUMENT;
  FaceBoundary boundary;
  cad::EvaluationWorkspace workspace;
  result = CadFaceMesh{};
  int32_t status = make_boundary(geometry, curves, face, workspace, boundary, result);
  if (status != PHX_MC_OK) return status;
  FaceMetric metric{&geometry, face, size, tolerance, theta, 1.0, 1.0, {}, {}, {}, {}, {}, {},
                    {}, {}, metric_points, {}, position_calls, graded_pole_rings};
  if (position_calls) position_calls->clear();
  if (metric_points) metric_points->clear();
  status = prepare_metric(geometry, curves, face, boundary, metric);
  if (status != PHX_MC_OK) return status;

  std::vector<double> points;
  points.reserve(2 * boundary.vertices.size());
  for (const auto& vertex : boundary.vertices) {
    points.push_back(vertex.uv[0] * metric.scale_u);
    points.push_back(vertex.uv[1] * metric.scale_v);
  }
  std::vector<std::array<double, 2>> seeds(boundary.loops.size());
  for (std::size_t i = 0; i < boundary.loops.size(); ++i)
    if (!hole_seed(boundary, boundary.loops[i], seeds[i])) return PHX_MC_INVALID_INPUT;
  std::size_t outer = boundary.loops.size();
  for (std::size_t i = 0; i < boundary.loops.size(); ++i) {
    bool contains_all = true;
    for (std::size_t j = 0; j < boundary.loops.size(); ++j) {
      if (i == j) continue;
      bool contains_vertex = false;
      for (const int32_t row : boundary.loops[j]) {
        if (contains(boundary, boundary.loops[i], boundary.vertices[row].uv)) {
          contains_vertex = true;
          break;
        }
      }
      if (!contains_vertex) {
        contains_all = false;
        break;
      }
    }
    if (contains_all) {
      if (outer != boundary.loops.size()) return PHX_MC_INVALID_INPUT;
      outer = i;
    }
  }
  if (outer == boundary.loops.size()) return PHX_MC_INVALID_INPUT;
  std::vector<double> holes;
  for (std::size_t i = 0; i < boundary.loops.size(); ++i) {
    if (i == outer) continue;
    const auto& seed = seeds[i];
    holes.push_back(seed[0] * metric.scale_u);
    holes.push_back(seed[1] * metric.scale_v);
  }
  std::vector<uint8_t> fixed(boundary.vertices.size(), 0);
  std::vector<int32_t> segments = boundary.segments;
  int64_t omitted_pole_points = 0;
  status = seed_poles(geometry, boundary, outer, metric, points, fixed, segments,
                      result.pole_sectors, result.protected_pole_edges, omitted_pole_points);
  if (status != PHX_MC_OK) return status;
  if (points.size()/2 > static_cast<std::size_t>(kMaxMeshPoints) ||
      segments.size()/2 > static_cast<std::size_t>(kMaxMeshPoints) ||
      holes.size()/2 > static_cast<std::size_t>(kMaxMeshPoints)) return PHX_MC_INDEX_RANGE_ERROR;
  metric.boundary_positions.resize(boundary.vertices.size());
  for (std::size_t i = 0; i < boundary.vertices.size(); ++i) {
    auto& entry = metric.boundary_positions[i];
    entry.first = {points[2 * i], points[2 * i + 1]};
    if (!classified_position(geometry, curves, face, boundary.vertices[i], workspace, entry.second))
      return PHX_MC_INVALID_INPUT;
  }
  std::sort(metric.boundary_positions.begin(), metric.boundary_positions.end());
  CadRefine2DInput input;
  input.points = points.data();
  input.point_count = static_cast<int32_t>(points.size() / 2);
  input.segments = segments.data();
  input.segment_count = static_cast<int32_t>(segments.size() / 2);
  input.holes = holes.data();
  input.hole_count = static_cast<int32_t>(holes.size() / 2);
  input.point_cap = static_cast<int32_t>(std::min<int64_t>(
      kMaxMeshPoints, std::max<int64_t>(int64_t{input.point_count} * 100, 10000)));
  input.metric = face_metric;
  input.metric_context = &metric;
  input.fixed_vertices = fixed.data();
  input.admit_point = metric.poles.empty() ? nullptr : admit_face_point;
  input.admit_context = &metric;
  input.fixed_region = in_fixed_zone;
  input.fixed_region_context = &metric;
  input.surface_position = face_position;
  input.surface_context = &metric;
  CadRefine2DResult refined;
  status = cad_refine2d(input, refined);
  if (status != PHX_MC_OK && status != PHX_MC_UNRESOLVED) return status;

  auto& tri = refined.triangulation;
  SurfaceTriangulation surface;
  surface.charts = tri.coordinates;
  surface.points.resize(3 * static_cast<std::size_t>(tri.vertex_count()));
  surface.normals.resize(3 * static_cast<std::size_t>(tri.vertex_count()), 0.0);
  std::vector<CadFaceVertex> vertices(static_cast<std::size_t>(tri.vertex_count()));
  std::vector<uint8_t> mapped_fixed(static_cast<std::size_t>(tri.vertex_count()), 0);
  for (std::size_t i = 0; i < fixed.size(); ++i) {
    const int32_t mapped = refined.vertex_map[i];
    if (mapped < 0 || mapped >= tri.vertex_count()) return PHX_MC_INVALID_INPUT;
    mapped_fixed[mapped] |= fixed[i];
  }
  for (int32_t i = 0; i < tri.vertex_count(); ++i) {
    const double* p = tri.point(i);
    vertices[i] = {{p[0] / metric.scale_u, p[1] / metric.scale_v}, 2, face, -1};
    if (!face_position(&metric, p, surface.points.data() + 3 * i)) return PHX_MC_INVALID_INPUT;
  }
  for (std::size_t i = 0; i < boundary.vertices.size(); ++i) {
    const int32_t mapped = refined.vertex_map[i];
    if (mapped < 0 || mapped >= tri.vertex_count() || vertices[mapped].dimension != 2)
      return PHX_MC_INVALID_INPUT;
    // The hook's classified lookup is keyed by the input chart point.
    const double* p = tri.point(mapped);
    if (p[0] != points[2 * i] || p[1] != points[2 * i + 1]) return PHX_MC_INTERNAL_ERROR;
    vertices[mapped] = boundary.vertices[i];
  }
  for (std::size_t t = 0; t < tri.triangles.size(); ++t) {
    if (!tri.live[t] || !tri.region[t] || tri.is_ghost(static_cast<int32_t>(t))) continue;
    const auto& cell = tri.triangles[t];
    surface.cells.push_back({cell.v[0], cell.v[1], cell.v[2]});
    surface.constrained.push_back({static_cast<int8_t>(tri.constraint(static_cast<int32_t>(t), 0) != -1),
                                   static_cast<int8_t>(tri.constraint(static_cast<int32_t>(t), 1) != -1),
                                   static_cast<int8_t>(tri.constraint(static_cast<int32_t>(t), 2) != -1)});
  }
  const int32_t prepare_status = surface.prepare();
  if (prepare_status != PHX_MC_OK) return prepare_status;
  // Where the fidelity tensor binds, the flips judge Delaunay in the face
  // metric (vertex tensors averaged by the sweep), so they keep the metric
  // shape the kernel built instead of undoing it in 3D (measured: plain 3D
  // flips left interior triangles past the theta * tol sag bound).
  {
    std::vector<double> tensors(9 * static_cast<std::size_t>(tri.vertex_count()));
    bool binding = false;
    const double isotropic = 1.0 / (metric.size * metric.size);
    for (int32_t i = 0; i < tri.vertex_count(); ++i) {
      double* tensor = tensors.data() + 9 * static_cast<std::size_t>(i);
      for (int k = 0; k < 9; ++k) tensor[k] = k % 4 == 0 ? isotropic : 0.0;
      binding |= vertex_metric(metric, tri.point(i), tensor);
    }
    if (binding) surface.metrics = std::move(tensors);
  }
  surface.sweep();
  smooth(metric, mapped_fixed, input.min_angle_degrees, vertices, surface);
  result.slivers.clear();
  result.triangles.clear();
  std::vector<std::pair<int32_t, int32_t>> pole_edges;
  for (const auto& edge : boundary.edges) {
    if (edge.curve >= 0) continue;
    const int32_t a = refined.vertex_map[edge.vertices[0]];
    const int32_t b = refined.vertex_map[edge.vertices[1]];
    pole_edges.emplace_back(std::min(a, b), std::max(a, b));
  }
  std::sort(pole_edges.begin(), pole_edges.end());
  for (const auto& cell : surface.cells) {
    bool sliver = false;
    for (int i = 0; i < 3; ++i) {
      const int32_t a = cell[i], b = cell[(i + 1) % 3];
      if (std::binary_search(pole_edges.begin(), pole_edges.end(),
                             std::make_pair(std::min(a, b), std::max(a, b)))) sliver = true;
    }
    if (sliver) result.slivers.push_back(cell);
    else result.triangles.push_back(cell);
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto& model = geometry.model();
  result.refinement_failures.reserve(refined.unresolved.size());
  for (const auto& failure : refined.unresolved) {
    phx_mc_cad_face_failure record{};
    record.face = face;
    record.loop = record.use = record.axis = -1;
    record.reason = PHX_MC_CAD_FACE_REFINEMENT;
    record.triangle = failure.triangle;
    record.bad_size = failure.bad_size;
    record.bad_angle = failure.bad_angle;
    record.refusal_kind = static_cast<int32_t>(failure.refusal);
    record.curve = record.curve_segment = -1;
    record.candidate_uv[0] = record.candidate_uv[1] = nan;
    record.segment_parameters[0] = record.segment_parameters[1] = nan;
    if (failure.refusal != CadRefine2DRefusal::none) {
      record.candidate_uv[0] = failure.candidate[0] / metric.scale_u;
      record.candidate_uv[1] = failure.candidate[1] / metric.scale_v;
    }
    const auto& cell = tri.triangles[static_cast<std::size_t>(failure.triangle)];
    for (int k = 0; k < 3; ++k) {
      const double* point = tri.point(cell.v[k]);
      record.triangle_uv[2*k] = point[0] / metric.scale_u;
      record.triangle_uv[2*k+1] = point[1] / metric.scale_v;
    }
    if (failure.segment >= 0 &&
        static_cast<std::size_t>(failure.segment) < boundary.edges.size()) {
      const auto& edge = boundary.edges[static_cast<std::size_t>(failure.segment)];
      record.use = edge.use;
      record.curve = edge.curve;
      record.segment_parameters[0] = edge.parameters[0];
      record.segment_parameters[1] = edge.parameters[1];
      for (int64_t loop = 0; loop < model.loop_count; ++loop)
        if (edge.use >= model.loop_use_offsets[loop] && edge.use < model.loop_use_offsets[loop+1]) {
          record.loop = static_cast<int32_t>(loop);
          break;
        }
      if (edge.curve >= 0) {
        const int64_t first = curves.offsets[edge.curve];
        const int64_t last = curves.offsets[edge.curve+1];
        const double s = std::min(edge.parameters[0], edge.parameters[1]);
        const double* found = std::lower_bound(curves.parameters+first, curves.parameters+last, s);
        if (found == curves.parameters+last || *found != s ||
            found-curves.parameters-first >= last-first-1) return PHX_MC_INTERNAL_ERROR;
        record.curve_segment = static_cast<int32_t>(found-curves.parameters-first);
      }
    }
    result.refinement_failures.push_back(record);
  }
  result.vertices = std::move(vertices);
  result.boundary_edges = std::move(boundary.edges);
  for (auto& edge : result.boundary_edges) {
    edge.vertices[0] = refined.vertex_map[edge.vertices[0]];
    edge.vertices[1] = refined.vertex_map[edge.vertices[1]];
  }
  for (auto& edge : result.protected_pole_edges) {
    edge[0] = refined.vertex_map[edge[0]];
    edge[1] = refined.vertex_map[edge[1]];
  }
  result.stats = refined.stats;
  result.omitted_pole_points = omitted_pole_points;
  return status;
}


}  // namespace phx::mc
