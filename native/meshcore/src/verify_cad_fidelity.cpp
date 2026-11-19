// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_cad_fidelity.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "capi_guard.hpp"
#include "directed_interval.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"

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

// For each subdivision leaf L of the original triangle T,
// S - I_T = (S - I_L) + (I_L - I_T). The first term has the leaf
// Hessian bound; the second is affine on L and its norm is maximized at
// a leaf vertex. Its interval value compares S there directly with the
// original stored triangle, so endpoint mismatch is included exactly once.
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
  double lower[18], higher[18];
  const int32_t status = geometry.interval_jets(node, box_lower, box_upper, 2,
                                                 lower, higher, workspace);
  if (status != PHX_MC_OK) return status;
  const auto du = di::subtract(di::point(box_upper[0]), di::point(box_lower[0]));
  const auto dv = di::subtract(di::point(box_upper[1]), di::point(box_lower[1]));
  double magnitude[3];
  for (int axis = 0; axis < 3; ++axis) {
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
    int64_t use_count, const int64_t* use_patch, const double* use_range,
    int64_t loop_count, int64_t chord_count, const int64_t* chord_use,
    const int64_t* chord_loop, const double* chord_uv, const double* chord_range,
    const int64_t* edge_chord, double tolerance, int32_t max_depth,
    int64_t output_capacity, double* upper, int32_t* item_status, int32_t* reasons,
    int32_t* subdivisions, double* maximum_upper) {
  return phx::mc::guarded([&]() -> int32_t {
    if (model == nullptr || !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 6, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(use_count, 2, sizeof(double)) ||
        !phx::mc::addressable(chord_count, 4, sizeof(double)) || loop_count < 0 ||
        coefficient_count < 0 || output_capacity < 0 || maximum_upper == nullptr ||
        !std::isfinite(tolerance) || tolerance < 0 || max_depth < 0 || max_depth > 24)
      return PHX_MC_INVALID_ARGUMENT;
    if (output_capacity < triangle_count) return PHX_MC_CAPACITY_EXCEEDED;
    if ((point_count && !points) ||
        (coefficient_count && (!coefficient_lower || !coefficient_upper)) ||
        (use_count && (!use_patch || !use_range)) ||
        (chord_count && (!chord_use || !chord_loop || !chord_uv || !chord_range)) ||
        (triangle_count && (!triangles || !face_rows || !uv || !edge_chord ||
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
      if (!std::isfinite(coefficient_lower[i]) || !std::isfinite(coefficient_upper[i]) ||
          coefficient_lower[i] > coefficient_upper[i])
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < point_count * 3; ++i)
      if (!std::isfinite(points[i])) return PHX_MC_NONFINITE_INPUT;
    for (int64_t i = 0; i < triangle_count; ++i) {
      if (face_rows[i] < 0 || face_rows[i] >= model->surface_count)
        return PHX_MC_INVALID_ARGUMENT;
      for (int j = 0; j < 3; ++j) {
        const int64_t chord = edge_chord[3 * i + j];
        if (triangles[3 * i + j] < 0 || triangles[3 * i + j] >= point_count ||
            chord < -1 || chord >= chord_count)
          return PHX_MC_INVALID_ARGUMENT;
        if (chord >= 0) {
          if (use_patch[chord_use[chord]] != face_rows[i]) return PHX_MC_INVALID_ARGUMENT;
          const double* a = uv + 6 * i + 2 * j;
          const double* b = uv + 6 * i + 2 * ((j + 1) % 3);
          const double* c = chord_uv + 4 * chord;
          const bool forward = std::memcmp(a, c, 2 * sizeof(double)) == 0 &&
                               std::memcmp(b, c + 2, 2 * sizeof(double)) == 0;
          const bool reverse = std::memcmp(a, c + 2, 2 * sizeof(double)) == 0 &&
                               std::memcmp(b, c, 2 * sizeof(double)) == 0;
          if (!forward && !reverse) return PHX_MC_INVALID_ARGUMENT;
        }
      }
      for (int j = 0; j < 6; ++j)
        if (!std::isfinite(uv[6 * i + j])) return PHX_MC_NONFINITE_INPUT;
    }
    phx::mc::cad::IntervalEvaluationWorkspace workspace;
    workspace.coefficient_lower = coefficient_lower;
    workspace.coefficient_upper = coefficient_upper;
    workspace.coefficient_count = coefficient_count;
    int32_t aggregate = PHX_MC_OK;
    double maximum = 0.0;
    for (int64_t i = 0; i < triangle_count; ++i) {
      bool trim = false;
      double vertices[9];
      for (int j = 0; j < 3; ++j) {
        trim |= edge_chord[3 * i + j] >= 0;
        const int64_t vertex = triangles[3 * i + j];
        for (int axis = 0; axis < 3; ++axis)
          vertices[3 * j + axis] = points[3 * vertex + axis];
      }
      if (trim) {
        upper[i] = std::numeric_limits<double>::infinity();
        item_status[i] = PHX_MC_UNRESOLVED;
        reasons[i] = PHX_MC_FIDELITY_TRIM_REFUSAL;
        subdivisions[i] = 0;
        aggregate = PHX_MC_UNRESOLVED;
        continue;
      }
      const auto result = phx::mc::cad::fidelity_triangle(
          geometry, face_rows[i], vertices, uv + 6 * i, tolerance, max_depth, workspace);
      if (result.status != PHX_MC_OK && result.status != PHX_MC_UNRESOLVED)
        return result.status;
      upper[i] = result.upper;
      item_status[i] = result.status;
      reasons[i] = result.reason;
      subdivisions[i] = result.subdivisions;
      maximum = std::max(maximum, result.upper);
      if (result.status != PHX_MC_OK) aggregate = PHX_MC_UNRESOLVED;
    }
    *maximum_upper = aggregate == PHX_MC_OK ? maximum : std::numeric_limits<double>::infinity();
    return aggregate;
  });
}
