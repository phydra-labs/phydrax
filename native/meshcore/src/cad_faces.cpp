// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_faces.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#include "chart_surface.hpp"
#include "expansion.hpp"
#include "predicates.hpp"

#include "phydrax_meshcore.h"

namespace phx::mc {

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
                        turns, 0.0};
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
      for (int64_t j = 0; j <= count - 1; ++j) {
        const int64_t index = forward ? j : count - 1 - j;
        const double s = pole ? static_cast<double>(index) / (count - 1)
                              : curves.parameters[begin + index];
        std::array<double, 2> uv;
        if (!chart_point(geometry, use, use_parameter(model, use, s), workspace, uv))
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

struct BoundarySample {
  std::array<double, 3> xyz;
  double spacing;
};

struct BoundaryNode {
  std::array<double, 3> low;
  std::array<double, 3> high;
  double spacing;
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
  std::vector<PoleZone> poles;
  std::vector<std::array<double, 2>>* metric_points;
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
  if (node.spacing + 0.3 * std::sqrt(box_distance2) >= h) return;
  if (node.left >= 0) {
    grade_boundary(context, node.left, point, h);
    grade_boundary(context, node.right, point, h);
    return;
  }
  for (int32_t i = node.begin; i < node.end; ++i) {
    const BoundarySample& sample = context.boundary[context.boundary_order[i]];
    const double dx = point[0] - sample.xyz[0], dy = point[1] - sample.xyz[1];
    const double dz = point[2] - sample.xyz[2];
    h = std::min(h, sample.spacing + 0.3 * std::sqrt(dx * dx + dy * dy + dz * dz));
  }
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
  double h = context.size;
  const double curvature = std::max(std::abs(differential.curvatures[0]),
                                    std::abs(differential.curvatures[1]));
  if (curvature > 0.0)
    h = std::min(h, std::sqrt(6.0 * context.theta * context.tolerance / curvature));
  if (!context.boundary_tree.empty()) grade_boundary(context, 0, jet, h);
  const double inverse = 1.0 / (h * h);
  return {differential.first_form[0] * inverse / (context.scale_u * context.scale_u),
          differential.first_form[1] * inverse / (context.scale_u * context.scale_v),
          differential.first_form[2] * inverse / (context.scale_v * context.scale_v)};
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
    }
  }
  metric.boundary_order.reserve(metric.boundary.size());
  for (std::size_t i = 0; i < metric.boundary.size(); ++i)
    if (std::isfinite(metric.boundary[i].spacing))
      metric.boundary_order.push_back(static_cast<int32_t>(i));
  if (!metric.boundary_order.empty())
    build_boundary_tree(metric, 0, static_cast<int32_t>(metric.boundary_order.size()));
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

int32_t admit_face_point(void* opaque, const double point[2]) {
  const auto& metric = *static_cast<FaceMetric*>(opaque);
  const double uv[2] = {point[0] / metric.scale_u, point[1] / metric.scale_v};
  for (const PoleZone& zone : metric.poles) {
    const double distance = zone.direction * (uv[zone.axis] - zone.origin);
    if (distance >= 0.0 && distance <= zone.spacing) return 0;
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

int32_t seed_poles(const cad::CadGeometry& geometry, const FaceBoundary& boundary,
                   std::size_t outer, FaceMetric& metric, std::vector<double>& points,
                   std::vector<uint8_t>& fixed, int64_t& omitted) {
  std::array<double, 2> low = boundary.vertices.front().uv, high = low;
  for (const CadFaceVertex& vertex : boundary.vertices)
    for (int axis = 0; axis < 2; ++axis) {
      low[axis] = std::min(low[axis], vertex.uv[axis]);
      high[axis] = std::max(high[axis], vertex.uv[axis]);
    }
  for (const PoleBoundary& pole : boundary.poles) {
    const auto& start = boundary.vertices[pole.first_vertex].uv;
    const auto& almost_end = boundary.vertices[pole.first_vertex + pole.count - 1].uv;
    const int axis = std::abs(start[0] - almost_end[0]) <
                             std::abs(start[1] - almost_end[1]) ? 0 : 1;
    const int free_axis = 1 - axis;
    const double direction = (low[axis] + high[axis]) * 0.5 > start[axis] ? 1.0 : -1.0;
    const double limit = direction > 0.0 ? high[axis] - start[axis] : start[axis] - low[axis];
    if (!(limit > 0.0)) return PHX_MC_INVALID_INPUT;
    const double middle[2] = {0.5 * (low[0] + high[0]), 0.5 * (low[1] + high[1])};
    double jet[18];
    if (!surface_jet(metric, middle, jet)) return PHX_MC_INVALID_INPUT;
    const int derivative = axis == 0 ? 3 : 6;
    const double speed = std::hypot(std::hypot(jet[derivative], jet[derivative + 1]),
                                    jet[derivative + 2]);
    if (!(speed > 0.0)) return PHX_MC_INVALID_INPUT;
    const double spacing = std::min(limit, metric.size / speed);
    if (!(spacing > 0.0)) return PHX_MC_INVALID_INPUT;
    double width = std::min(limit, 3.0 * spacing);
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
    metric.poles.push_back({axis, start[axis], direction, width, spacing});
    const double first = geometry.model().use_ranges[2 * pole.use];
    const double last = geometry.model().use_ranges[2 * pole.use + 1];
    std::array<double, 2> end;
    if (!chart_point(geometry, pole.use,
                     geometry.model().use_senses[pole.use] > 0 ? last : first,
                     metric.workspace, end)) return PHX_MC_INVALID_INPUT;
    for (double offset = spacing; offset <= width; offset += spacing) {
      std::array<double, 2> uv = start;
      uv[axis] = start[axis] + direction * offset;
      if (offset == spacing && direction * (uv[axis] - start[axis]) < spacing)
        uv[axis] = std::nextafter(uv[axis], direction > 0.0
                                  ? std::numeric_limits<double>::infinity()
                                  : -std::numeric_limits<double>::infinity());
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
      const int count = std::max(8, static_cast<int>(std::ceil(length / metric.size)));
      for (int i = 1; i < count; ++i) {
        uv[free_axis] = start[free_axis] + (end[free_axis] - start[free_axis]) * i / count;
        if (!in_trim(boundary, outer, uv)) continue;
        const int proximity = near_boundary(boundary, metric, uv);
        if (proximity < 0) return PHX_MC_INVALID_INPUT;
        if (proximity > 0) {
          ++omitted;
          continue;
        }
        points.push_back(uv[0] * metric.scale_u);
        points.push_back(uv[1] * metric.scale_v);
        fixed.push_back(1);
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

void smooth(const cad::CadGeometry& geometry, int32_t face, const FaceMetric& metric,
            const std::vector<uint8_t>& fixed, std::vector<CadFaceVertex>& vertices,
            SurfaceTriangulation& surface) {
  std::vector<std::vector<int32_t>> incident(vertices.size());
  for (std::size_t cell = 0; cell < surface.cells.size(); ++cell) {
    for (const int32_t v : surface.cells[cell]) incident[v].push_back(static_cast<int32_t>(cell));
  }
  cad::EvaluationWorkspace workspace;
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t v = 0; v < vertices.size(); ++v) {
      if (vertices[v].dimension != 2 || fixed[v] || incident[v].empty()) continue;
      std::vector<int32_t> neighbors;
      for (const int32_t cell : incident[v])
        for (const int32_t u : surface.cells[cell])
          if (u != static_cast<int32_t>(v)) neighbors.push_back(u);
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
      const auto p = surface.point(static_cast<int32_t>(v));
      double sum = 0.0, chart[2] = {};
      for (const int32_t u : neighbors) {
        const auto d = surface.point(u) - p;
        const double length = std::sqrt(geometry::dot(d, d));
        if (!(length > 0.0)) continue;
        const double weight = 1.0 / length;
        const double* q = surface.chart(u);
        chart[0] += weight * q[0];
        chart[1] += weight * q[1];
        sum += weight;
      }
      if (!(sum > 0.0)) continue;
      chart[0] /= sum;
      chart[1] /= sum;
      const std::array<double, 2> uv{chart[0] / metric.scale_u, chart[1] / metric.scale_v};
      std::array<double, 3> xyz;
      if (!evaluate_position(geometry, face, uv, workspace, xyz)) continue;
      const double previous_quality = maximum_cosine(surface, incident[v]);
      const std::array<double, 2> old_chart{surface.charts[2 * v], surface.charts[2 * v + 1]};
      const std::array<double, 3> old_point{surface.points[3 * v], surface.points[3 * v + 1],
                                            surface.points[3 * v + 2]};
      std::copy_n(chart, 2, surface.charts.begin() + 2 * v);
      std::copy(xyz.begin(), xyz.end(), surface.points.begin() + 3 * v);
      bool legal = true;
      for (const int32_t cell : incident[v]) {
        const auto& t = surface.cells[cell];
        if (orient2d(surface.chart(t[0]), surface.chart(t[1]), surface.chart(t[2])) <= 0) {
          legal = false;
          break;
        }
      }
      if (!legal || maximum_cosine(surface, incident[v]) > previous_quality) {
        std::copy(old_chart.begin(), old_chart.end(), surface.charts.begin() + 2 * v);
        std::copy(old_point.begin(), old_point.end(), surface.points.begin() + 3 * v);
      } else {
        vertices[v].uv = uv;
      }
    }
  }
}

}  // namespace

int32_t cad_mesh_face(const cad::CadGeometry& geometry,
                      const CadCurveDiscretization& curves, int32_t face,
                      double size, double tolerance, double theta,
                      CadFaceMesh& result, std::vector<std::array<double, 2>>* metric_points) {
  if (face < 0 || face >= geometry.surface_count() || !(size > 0.0) ||
      !(tolerance > 0.0) || !(theta > 0.0) || !std::isfinite(size) ||
      !std::isfinite(tolerance) || !std::isfinite(theta)) return PHX_MC_INVALID_ARGUMENT;
  FaceBoundary boundary;
  cad::EvaluationWorkspace workspace;
  result = CadFaceMesh{};
  int32_t status = make_boundary(geometry, curves, face, workspace, boundary, result);
  if (status != PHX_MC_OK) return status;
  FaceMetric metric{&geometry, face, size, tolerance, theta, 1.0, 1.0, {}, {}, {}, {}, {}, metric_points};
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
  int64_t omitted_pole_points = 0;
  status = seed_poles(geometry, boundary, outer, metric, points, fixed, omitted_pole_points);
  if (status != PHX_MC_OK) return status;
  if (points.size()/2 > static_cast<std::size_t>(kMaxMeshPoints) ||
      boundary.segments.size()/2 > static_cast<std::size_t>(kMaxMeshPoints) ||
      holes.size()/2 > static_cast<std::size_t>(kMaxMeshPoints)) return PHX_MC_INDEX_RANGE_ERROR;
  CadRefine2DInput input;
  input.points = points.data();
  input.point_count = static_cast<int32_t>(points.size() / 2);
  input.segments = boundary.segments.data();
  input.segment_count = static_cast<int32_t>(boundary.segments.size() / 2);
  input.holes = holes.data();
  input.hole_count = static_cast<int32_t>(holes.size() / 2);
  input.point_cap = static_cast<int32_t>(std::min<int64_t>(
      kMaxMeshPoints, std::max<int64_t>(int64_t{input.point_count} * 100, 10000)));
  input.metric = face_metric;
  input.metric_context = &metric;
  input.fixed_vertices = fixed.data();
  input.admit_point = metric.poles.empty() ? nullptr : admit_face_point;
  input.admit_context = &metric;
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
    std::array<double, 2> uv{p[0] / metric.scale_u, p[1] / metric.scale_v};
    vertices[i] = {uv, 2, face, -1};
    std::array<double, 3> xyz;
    if (!evaluate_position(geometry, face, uv, workspace, xyz)) return PHX_MC_INVALID_INPUT;
    std::copy(xyz.begin(), xyz.end(), surface.points.begin() + 3 * i);
  }
  for (std::size_t i = 0; i < boundary.vertices.size(); ++i) {
    const int32_t mapped = refined.vertex_map[i];
    if (mapped < 0 || mapped >= tri.vertex_count() || vertices[mapped].dimension != 2)
      return PHX_MC_INVALID_INPUT;
    vertices[mapped] = boundary.vertices[i];
    std::array<double, 3> xyz;
    if (!classified_position(geometry, curves, face, vertices[mapped], workspace, xyz))
      return PHX_MC_INVALID_INPUT;
    std::copy(xyz.begin(), xyz.end(), surface.points.begin() + 3 * mapped);
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
  surface.sweep();
  smooth(geometry, face, metric, mapped_fixed, vertices, surface);
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
  result.vertices = std::move(vertices);
  result.boundary_edges = std::move(boundary.edges);
  for (auto& edge : result.boundary_edges) {
    edge.vertices[0] = refined.vertex_map[edge.vertices[0]];
    edge.vertices[1] = refined.vertex_map[edge.vertices[1]];
  }
  result.stats = refined.stats;
  result.omitted_pole_points = omitted_pole_points;
  return status;
}


}  // namespace phx::mc
