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
  double pc_lower[6], pc_upper[6];
  int32_t status = geometry.interval_jets(geometry.pcurve_node(use), &low, &high, 2,
                                           pc_lower, pc_upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  const auto dt = di::subtract(di::point(high), di::point(low));
  double second_magnitude[3] = {
      di::magnitude({pc_lower[4], pc_upper[4]}),
      di::magnitude({pc_lower[5], pc_upper[5]}), 0.0};
  const double curvature = phx::mc::geometry::norm_upper(second_magnitude);
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
  const auto error = di::add(
      di::point(endpoint_error),
      di::multiply(di::point(0.125), di::multiply(di::point(curvature), di::multiply(dt, dt))));
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
    int32_t* subdivisions, double* maximum_upper, int64_t evidence_capacity,
    phx_mc_fidelity_finding* evidence, int64_t* evidence_count) {
  return phx::mc::guarded([&]() -> int32_t {
    if (model == nullptr || !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 6, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)) ||
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
        (evidence_capacity && !evidence) ||
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
      if (std::isnan(coefficient_lower[i]) || std::isnan(coefficient_upper[i]) ||
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
      }
      for (int j = 0; j < 6; ++j)
        if (!std::isfinite(uv[6 * i + j])) return PHX_MC_NONFINITE_INPUT;
    }
    const auto findings = phx::mc::cad::association_findings(
        triangle_count, face_rows, uv, chord_count, chord_use, use_patch, chord_uv, edge_chord);
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
          if (model->use_kinds[use] != 0 ||
              phx::mc::cad::trim_lens_bound(geometry, face_rows[i], use, chord_uv + 4 * chord,
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
