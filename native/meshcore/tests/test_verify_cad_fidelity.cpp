// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cad_geometry.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "verify_cad_fidelity.hpp"

int main() {
  using namespace phx::mc::cad;
  const int32_t kinds[] = {PHX_MC_CAD_NODE_TEMPLATE, PHX_MC_CAD_NODE_AFFINE,
                           PHX_MC_CAD_NODE_TEMPLATE, PHX_MC_CAD_NODE_REPARAMETERIZED,
                           PHX_MC_CAD_NODE_REPARAMETERIZED, PHX_MC_CAD_NODE_ISOPARAMETRIC,
                           PHX_MC_CAD_NODE_ISOPARAMETRIC, PHX_MC_CAD_NODE_TEMPLATE,
                           PHX_MC_CAD_NODE_SURFACE_MAPPED, PHX_MC_CAD_NODE_EXTRUSION,
                           PHX_MC_CAD_NODE_RULED, PHX_MC_CAD_NODE_REVOLUTION};
  const int64_t offsets[] = {0, 1, 3, 4, 6, 8, 10, 12, 13, 15, 16, 18, 19};
  const int64_t integers[] = {template_index("sphere"), 0, 3, template_index("circle3"),
                              2, 0, 2, 1, 0, 0, 0, 1, template_index("circle2"), 0, 7, 2, 2, 3, 2};
  const std::vector<double> coefficients = {
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0,
      1.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, -0.25, 0.75,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0,
      0.2, 0.8, 0.2, 0.8, 0.25, -0.125,
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.25, -0.5, 1.0,
      0.1, 0.2, 0.3, 0.0, 0.0, 1.0};
  const int64_t real_offsets[] = {0, 13, 25, 35, 37, 39, 40, 41, 48, 48, 51, 51, 57};
  const int64_t empty[] = {0};
  const int32_t surface_nodes[] = {0};
  const int8_t orientations[] = {1};
  const double periods[] = {0.0, 0.0};
  const double domains[] = {-2.0, 2.0, -2.0, 2.0};
  phx_mc_cad_model model{};
  model.node_count = 12;
  model.node_kinds = kinds;
  model.node_integer_offsets = offsets;
  model.node_integers = integers;
  model.node_real_offsets = real_offsets;
  model.node_reals = coefficients.data();
  model.loop_use_offsets = empty;
  model.region_surface_offsets = empty;
  model.surface_count = 1;
  model.surface_nodes = surface_nodes;
  model.surface_orientations = orientations;
  model.surface_periods = periods;
  model.surface_parameter_domains = domains;
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model, Evaluators::kPrograms) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  workspace.coefficient_lower = coefficients.data();
  workspace.coefficient_upper = coefficients.data();
  workspace.coefficient_count = static_cast<int64_t>(coefficients.size());
  EvaluationWorkspace point_workspace;
  const double lo[] = {0.1, -0.4}, hi[] = {0.6, 0.3};
  for (int node : {0, 1, 9, 10, 11}) {
    for (int order = 0; order <= 2; ++order) {
      std::array<double, 18> lower{}, upper{}, jet{};
      PHX_CHECK(geometry.interval_jets(node, lo, hi, order, lower.data(), upper.data(), workspace) ==
                PHX_MC_OK);
      for (int i = 0; i <= 20; ++i) {
        for (int j = 0; j <= 20; ++j) {
          const double uv[] = {lo[0] + (hi[0] - lo[0]) * i / 20,
                               lo[1] + (hi[1] - lo[1]) * j / 20};
          PHX_CHECK(geometry.jets(node, uv, order, jet.data(), point_workspace));
          const int count = 3 * (order + 1) * (order + 2) / 2;
          for (int k = 0; k < count; ++k) {
            PHX_CHECK(lower[k] <= jet[k]);
            PHX_CHECK(jet[k] <= upper[k]);
          }
        }
      }
      const double point[] = {0.25, -0.125};
      PHX_CHECK(geometry.interval_jets(node, point, point, order, lower.data(), upper.data(),
                                       workspace) == PHX_MC_OK);
      PHX_CHECK(geometry.jets(node, point, order, jet.data(), point_workspace));
      const int count = 3 * (order + 1) * (order + 2) / 2;
      for (int k = 0; k < count; ++k) {
        PHX_CHECK(lower[k] <= jet[k]);
        PHX_CHECK(jet[k] <= upper[k]);
      }
    }
  }
  for (int node = 2; node < 7; ++node) {
    for (int order = 0; order <= 2; ++order) {
      std::array<double, 9> lower{}, upper{}, jet{};
      const double first = 0.25, last = 0.75;
      PHX_CHECK(geometry.interval_jets(node, &first, &last, order, lower.data(), upper.data(),
                                       workspace) == PHX_MC_OK);
      for (int i = 0; i <= 20; ++i) {
        const double t = first + (last - first) * i / 20;
        PHX_CHECK(geometry.jets(node, &t, order, jet.data(), point_workspace));
        for (int k = 0; k < 3 * (order + 1); ++k) {
          PHX_CHECK(lower[k] <= jet[k]);
          PHX_CHECK(jet[k] <= upper[k]);
        }
      }
    }
  }
  for (int order = 0; order <= 2; ++order) {
    std::array<double, 9> lower{}, upper{}, jet{};
    const double first = 0.25, last = 0.75;
    PHX_CHECK(geometry.interval_jets(8, &first, &last, order, lower.data(), upper.data(),
                                     workspace) == PHX_MC_OK);
    for (int i = 0; i <= 20; ++i) {
      const double t = first + (last - first) * i / 20;
      PHX_CHECK(geometry.jets(8, &t, order, jet.data(), point_workspace));
      for (int k = 0; k < 3 * (order + 1); ++k) {
        PHX_CHECK(lower[k] <= jet[k]);
        PHX_CHECK(jet[k] <= upper[k]);
      }
    }
  }
  const double uv[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
  const double points[] = {2.0, 0.0, 0.0, 2 * std::cos(1.0), 2 * std::sin(1.0), 0.0,
                           2 * std::cos(1.0), 0.0, 2 * std::sin(1.0)};
  const int64_t triangles[] = {0, 1, 2};
  const int32_t face_rows[] = {0};
  double bound, maximum;
  int32_t status, reason, splits;
  const auto certified = fidelity_triangle(geometry, 0, points, uv, 1.0, 8, workspace);
  bound = certified.upper;
  PHX_CHECK(certified.status == PHX_MC_OK &&
            certified.reason == PHX_MC_FIDELITY_CERTIFIED && certified.subdivisions > 0);
  for (int i = 0; i <= 50; ++i) {
    for (int j = 0; j <= 50 - i; ++j) {
      const double u = i / 50.0, v = j / 50.0;
      const double sample[] = {2 * std::cos(v) * std::cos(u),
                               2 * std::cos(v) * std::sin(u), 2 * std::sin(v)};
      double squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double linear = (1 - u - v) * points[axis] + u * points[3 + axis] +
                              v * points[6 + axis];
        squared += (sample[axis] - linear) * (sample[axis] - linear);
      }
      PHX_CHECK(std::sqrt(squared) <= bound);
    }
  }
  const auto unresolved = fidelity_triangle(geometry, 0, points, uv, 1e-4, 2, workspace);
  PHX_CHECK(unresolved.status == PHX_MC_UNRESOLVED &&
            unresolved.reason == PHX_MC_FIDELITY_RESOLUTION);
  const double corner[] = {0.0, 0.0, 0.0};
  const int64_t loop_offsets[] = {0, 1};
  const int32_t loop_surface[] = {0}, use_entity[] = {0}, pcurve[] = {7};
  const int8_t use_kind[] = {0}, use_sense[] = {1};
  const double use_parameters[] = {0.0, 1.0};
  const int32_t curve_node[] = {2}, curve_corners[] = {0, 0};
  int64_t chart_to_mesh[] = {0}, mesh_triangles[] = {0, 1, 2};
  model.corner_count = 1;
  model.corner_points = corner;
  model.curve_count = 1;
  model.curve_nodes = curve_node;
  model.curve_corners = curve_corners;
  model.curve_ranges = use_parameters;
  model.loop_count = 1;
  model.loop_use_offsets = loop_offsets;
  model.loop_surfaces = loop_surface;
  model.use_count = 1;
  model.use_kinds = use_kind;
  model.use_entities = use_entity;
  model.use_senses = use_sense;
  model.use_ranges = use_parameters;
  model.use_pcurve_nodes = pcurve;
  const int64_t use_patch[] = {0}, chord_use[] = {0, 0, 0}, chord_loop[] = {0, 0, 0};
  const double chords[] = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                           0.0, 1.0, 0.0, 0.0};
  const double chord_parameters[] = {0.0, 0.25, 0.25, 0.75, 0.75, 1.0};
  int64_t chord_edges[] = {0, 1, 2};
  phx_mc_fidelity_finding evidence[12];
  int64_t evidence_count;
  const auto verify = [&]() {
    return phx_mc_verify_cad_fidelity(
        &model, coefficients.size(), coefficients.data(), coefficients.data(),
        3, points, 1, triangles, face_rows, uv, 1, mesh_triangles, chart_to_mesh,
        1, use_patch, use_parameters,
        1, 3, chord_use, chord_loop, chords, chord_parameters, chord_edges, 1.0, 8,
        1, &bound, &status, &reason, &splits, &maximum, 12, evidence, &evidence_count);
  };
  const int32_t initial = verify();
  PHX_CHECK((initial == PHX_MC_OK || initial == PHX_MC_UNRESOLVED) && evidence_count == 0);
  chord_edges[0] = -1;
  PHX_CHECK(verify() == PHX_MC_UNRESOLVED);
  PHX_CHECK(evidence_count >= 2 && reason == PHX_MC_FIDELITY_EDGE_ASSOCIATION);
  bool missing = false, unpaired = false;
  for (int64_t i = 0; i < evidence_count; ++i) {
    missing |= evidence[i].kind == PHX_MC_FIDELITY_MISSING_CHORD && evidence[i].chord == 0;
    unpaired |= evidence[i].kind == PHX_MC_FIDELITY_UNPAIRED_INTERIOR &&
                evidence[i].triangle == 0 && evidence[i].edge == 0;
  }
  PHX_CHECK(missing && unpaired);
  chord_edges[0] = 1;
  PHX_CHECK(verify() == PHX_MC_UNRESOLVED);
  bool duplicate = false;
  for (int64_t i = 0; i < evidence_count; ++i)
    duplicate |= evidence[i].kind == PHX_MC_FIDELITY_DUPLICATE_CHORD && evidence[i].chord == 1;
  PHX_CHECK(duplicate);
  chord_edges[0] = 0;
  chart_to_mesh[0] = -1;
  PHX_CHECK(verify() == PHX_MC_UNRESOLVED);
  bool bad_link = false, missing_mesh = false;
  for (int64_t i = 0; i < evidence_count; ++i) {
    bad_link |= evidence[i].kind == PHX_MC_FIDELITY_CHART_MESH_LINK &&
                evidence[i].triangle == 0;
    missing_mesh |= evidence[i].kind == PHX_MC_FIDELITY_MISSING_MESH_TRIANGLE &&
                    evidence[i].other_triangle == 0;
  }
  PHX_CHECK(bad_link && missing_mesh);
  chart_to_mesh[0] = 0;
  std::swap(mesh_triangles[1], mesh_triangles[2]);
  PHX_CHECK(verify() == PHX_MC_UNRESOLVED);
  PHX_CHECK(evidence_count >= 1 && evidence[0].kind == PHX_MC_FIDELITY_CHART_MESH_LINK);
  return 0;
}
