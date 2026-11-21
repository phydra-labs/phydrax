// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "cad_geometry.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"
#include "verify_cad_fidelity.hpp"

namespace {
using namespace phx::mc::cad;

// A flat CAD model assembled row by row; it owns every table it exposes.
struct Model {
  std::vector<int32_t> kinds;
  std::vector<int64_t> integer_offsets{0}, integers, real_offsets{0};
  std::vector<double> reals;
  std::vector<int32_t> surfaces;
  std::vector<int8_t> orientations;
  std::vector<double> periods, domains, corners;
  std::vector<int32_t> curves, curve_corners;
  std::vector<double> curve_ranges;
  std::vector<int64_t> loop_offsets{0}, region_offsets{0};
  std::vector<int32_t> loop_surfaces, use_entities, use_pcurves;
  std::vector<int8_t> use_kinds, use_senses;
  std::vector<double> use_ranges;

  int32_t add(int32_t kind, std::vector<int64_t> ints, std::vector<double> values) {
    kinds.push_back(kind);
    integers.insert(integers.end(), ints.begin(), ints.end());
    reals.insert(reals.end(), values.begin(), values.end());
    integer_offsets.push_back(static_cast<int64_t>(integers.size()));
    real_offsets.push_back(static_cast<int64_t>(reals.size()));
    return static_cast<int32_t>(kinds.size() - 1);
  }
  int32_t add_template(const char* name, std::vector<double> values) {
    const int32_t id = template_index(name);
    PHX_CHECK(id >= 0);
    return add(PHX_MC_CAD_NODE_TEMPLATE, {id}, std::move(values));
  }
  void surface(int32_t node) {
    surfaces.push_back(node);
    orientations.push_back(1);
    periods.insert(periods.end(), {0.0, 0.0});
    domains.insert(domains.end(), {-8.0, 8.0, -8.0, 8.0});
  }
  void corner(double x, double y, double z) { corners.insert(corners.end(), {x, y, z}); }
  void curve(int32_t node, int32_t first, int32_t last) {
    curves.push_back(node);
    curve_corners.insert(curve_corners.end(), {first, last});
    curve_ranges.insert(curve_ranges.end(), {0.0, 1.0});
  }
  void use(int8_t kind, int32_t entity, double first, double last, int32_t pcurve) {
    use_kinds.push_back(kind);
    use_entities.push_back(entity);
    use_senses.push_back(1);
    use_ranges.insert(use_ranges.end(), {first, last});
    use_pcurves.push_back(pcurve);
  }
  void close_loop(int32_t surface_row) {
    loop_surfaces.push_back(surface_row);
    loop_offsets.push_back(static_cast<int64_t>(use_kinds.size()));
  }
  phx_mc_cad_model view() const {
    phx_mc_cad_model m{};
    m.node_count = static_cast<int64_t>(kinds.size());
    m.node_kinds = kinds.data();
    m.node_integer_offsets = integer_offsets.data();
    m.node_integers = integers.data();
    m.node_real_offsets = real_offsets.data();
    m.node_reals = reals.data();
    m.corner_count = static_cast<int64_t>(corners.size() / 3);
    m.corner_points = corners.data();
    m.surface_count = static_cast<int64_t>(surfaces.size());
    m.surface_nodes = surfaces.data();
    m.surface_orientations = orientations.data();
    m.surface_periods = periods.data();
    m.surface_parameter_domains = domains.data();
    m.curve_count = static_cast<int64_t>(curves.size());
    m.curve_nodes = curves.data();
    m.curve_corners = curve_corners.data();
    m.curve_ranges = curve_ranges.data();
    m.loop_count = static_cast<int64_t>(loop_surfaces.size());
    m.loop_use_offsets = loop_offsets.data();
    m.loop_surfaces = loop_surfaces.data();
    m.use_count = static_cast<int64_t>(use_kinds.size());
    m.use_kinds = use_kinds.data();
    m.use_entities = use_entities.data();
    m.use_senses = use_senses.data();
    m.use_ranges = use_ranges.data();
    m.use_pcurve_nodes = use_pcurves.data();
    m.region_surface_offsets = region_offsets.data();
    return m;
  }
  // Point coefficient bounds: the binary64 reals are the exact source.
  void bind_bounds(IntervalEvaluationWorkspace& workspace) const {
    workspace.coefficient_lower = reals.data();
    workspace.coefficient_upper = reals.data();
    workspace.coefficient_count = static_cast<int64_t>(reals.size());
  }
};

// One intersection chart: start, end, box lower, box upper (4 each), P (3, 3).
void add_chart(std::vector<double>& reals, std::array<double, 4> start, std::array<double, 4> end,
               std::array<double, 8> box, std::array<double, 9> preconditioner) {
  reals.insert(reals.end(), start.begin(), start.end());
  reals.insert(reals.end(), end.begin(), end.end());
  reals.insert(reals.end(), box.begin(), box.end());
  reals.insert(reals.end(), preconditioner.begin(), preconditioner.end());
}

// Chart and chord rows of phx_mc_verify_cad_fidelity.
struct Mesh {
  std::vector<double> points;
  std::vector<int64_t> triangles;
  std::vector<int32_t> face_rows;
  std::vector<double> uv;
  std::vector<int64_t> mesh_triangles, chart_to_mesh, edge_chord;
  std::vector<int64_t> use_patch;
  std::vector<double> use_range;
  int64_t loop_count = 0;
  std::vector<int64_t> chord_use, chord_loop;
  std::vector<double> chord_uv, chord_range;
};

struct Verdict {
  int32_t status;
  std::vector<double> upper;
  std::vector<int32_t> item_status, reasons, subdivisions;
  double maximum;
  std::vector<phx_mc_fidelity_finding> findings;
};

// Outputs start at sentinels, so a refusal that must leave them unchanged
// can be checked.
constexpr double kSentinel = -7.0;

Verdict verify(const Model& model, const Mesh& mesh, double tolerance, int32_t depth) {
  const phx_mc_cad_model view = model.view();
  const auto rows = mesh.face_rows.size();
  Verdict verdict{0, std::vector<double>(rows, kSentinel), std::vector<int32_t>(rows, -7),
                  std::vector<int32_t>(rows, -7), std::vector<int32_t>(rows, -7), kSentinel,
                  std::vector<phx_mc_fidelity_finding>(64)};
  int64_t evidence_count = 0;
  verdict.status = phx_mc_verify_cad_fidelity(
      &view, static_cast<int64_t>(model.reals.size()), model.reals.data(), model.reals.data(),
      static_cast<int64_t>(mesh.points.size() / 3), mesh.points.data(),
      static_cast<int64_t>(rows), mesh.triangles.data(), mesh.face_rows.data(), mesh.uv.data(),
      static_cast<int64_t>(mesh.mesh_triangles.size() / 3), mesh.mesh_triangles.data(),
      mesh.chart_to_mesh.data(), static_cast<int64_t>(mesh.use_patch.size()),
      mesh.use_patch.data(), mesh.use_range.data(), mesh.loop_count,
      static_cast<int64_t>(mesh.chord_use.size()), mesh.chord_use.data(), mesh.chord_loop.data(),
      mesh.chord_uv.data(), mesh.chord_range.data(), mesh.edge_chord.data(), tolerance, depth,
      static_cast<int64_t>(rows), verdict.upper.data(), verdict.item_status.data(),
      verdict.reasons.data(), verdict.subdivisions.data(), &verdict.maximum,
      static_cast<int64_t>(verdict.findings.size()), verdict.findings.data(), &evidence_count);
  verdict.findings.resize(static_cast<std::size_t>(std::max<int64_t>(evidence_count, 0)));
  return verdict;
}

// |S(x) - I_T(x)| sampled on the chart triangle for a closed form S.
template <class Surface>
double sampled_interpolation_error(const double* uv, const double* points, Surface surface) {
  double worst = 0.0;
  constexpr int kSteps = 400;
  for (int i = 0; i <= kSteps; ++i)
    for (int j = 0; j <= kSteps - i; ++j) {
      const double w[3] = {1.0 - (i + j) / double(kSteps), i / double(kSteps), j / double(kSteps)};
      const double u = w[0] * uv[0] + w[1] * uv[2] + w[2] * uv[4];
      const double v = w[0] * uv[1] + w[1] * uv[3] + w[2] * uv[5];
      const std::array<double, 3> s = surface(u, v);
      double squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double linear = w[0] * points[axis] + w[1] * points[3 + axis] + w[2] * points[6 + axis];
        squared += (s[axis] - linear) * (s[axis] - linear);
      }
      worst = std::max(worst, std::sqrt(squared));
    }
  return worst;
}

// F1 (leaf), MAC's demo_ab B: the extrusion along z of the line where the
// planes z = 0 and y = 0 meet, charted by x in [0, 1] on carrier [0, 1] and
// by x in [1, 1.01] on carrier [1, 2]: S(u, v) = (x(u), 0, v) has a speed
// jump at the seam u = 1. The chart triangle (0.5, 0), (1.5, 0), (0.5, 1)
// straddles it; the sampled |S - I_T| is 0.2475 at (1, 0) (certified
// 2.0e-15 before). Fault caught: a seam treated as a C2 point (the
// per-chart Hessians are zero).
void intersection_seam_leaf() {
  Model m;
  const int32_t first = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0});
  const int32_t second = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 0, 1});
  // R = S1(u1, v1) - S2(u2, v2) = (u1 - u2, v1, -v2); free (v1, u2, v2).
  const std::array<double, 9> inverse{0, 1, 0, -1, 0, 0, 0, 0, -1};
  std::vector<double> reals{0.0, 0.0};
  add_chart(reals, {0, 0, 0, 0}, {1, 0, 1, 0}, {0, -0.1, -0.1, -0.1, 1, 0.1, 1.1, 0.1}, inverse);
  add_chart(reals, {1, 0, 1, 0}, {1.01, 0, 1.01, 0}, {1, -0.1, 0.9, -0.1, 1.01, 0.1, 1.11, 0.1},
            inverse);
  const int32_t line = m.add(PHX_MC_CAD_NODE_INTERSECTION, {first, second, 2, 0, -1, -1, 0, 0, 0},
                             reals);
  const int32_t swept = m.add(PHX_MC_CAD_NODE_EXTRUSION, {line}, {0, 0, 1});
  m.surface(swept);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  m.bind_bounds(workspace);
  EvaluationWorkspace point_workspace;
  const double uv[6] = {0.5, 0.0, 1.5, 0.0, 0.5, 1.0};
  double points[9];
  for (int i = 0; i < 3; ++i)
    PHX_CHECK(geometry.jets(swept, uv + 2 * i, 0, points + 3 * i, point_workspace));
  const double truth = sampled_interpolation_error(uv, points, [](double u, double v) {
    return std::array<double, 3>{u <= 1.0 ? u : 1.0 + 0.01 * (u - 1.0), 0.0, v};
  });
  PHX_CHECK(truth >= 0.2474);
  const auto strict = fidelity_triangle(geometry, 0, points, uv, 1e-9, 8, workspace);
  std::printf("F1 leaf (tolerance 1e-9): status %d upper %.6e sampled %.6e\n", strict.status,
              strict.upper, truth);
  PHX_CHECK(strict.status != PHX_MC_OK || strict.upper >= truth);
  // The seam is an exact C0 line, so the clipped pieces certify a tight bound.
  const auto result = fidelity_triangle(geometry, 0, points, uv, 1.0, 8, workspace);
  std::printf("F1 leaf (tolerance 1): status %d upper %.6e\n", result.status, result.upper);
  PHX_CHECK(result.status == PHX_MC_OK && result.upper >= truth && result.upper <= 0.25);
}

// F1 (lens): the plane z = 0 capped by the extrusion along z of the C1
// quadratic B-spline (w, w^2 / 2, 0), w in [0, 2]; the p-curve (w, w^2 / 2)
// is charted by w in [0.49994, 0.99994] (carrier [0, 1]) and [0.99994,
// 1.015625] (carrier [1, 2]). The chord over [0.875, 2] crosses the seam,
// where the speed drops 32-fold; its sampled distance sup |p(t) - L(t)| (the
// plane is an isometry) must be within the cap. Fault caught: a seam
// treated as a C2 point in the second-derivative remainder.
void intersection_seam_lens() {
  Model m;
  const int32_t plane = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0});
  const int32_t spline = m.add(PHX_MC_CAD_NODE_BSPLINE_CURVE, {2, 3, 3},
                               {0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0, 0, 2, 2, 0, 1, 1, 1});
  const int32_t carrier = m.add(PHX_MC_CAD_NODE_EXTRUSION, {spline}, {0, 0, 1});
  // R = (u1 - w, v1 - w^2 / 2, -h); free (u1, v1, h): P = diag(1, 1, -1).
  const std::array<double, 9> inverse{1, 0, 0, 0, 1, 0, 0, 0, -1};
  const std::array<double, 8> box{-1, -1, -1, -1, 3, 3, 3, 1};
  const double w[3] = {0.49994, 0.99994, 1.015625};
  std::vector<double> reals{0.0, 0.0};
  for (int c = 0; c < 2; ++c)
    add_chart(reals, {w[c], 0.5 * w[c] * w[c], w[c], 0}, {w[c + 1], 0.5 * w[c + 1] * w[c + 1],
                                                          w[c + 1], 0},
              box, inverse);
  const int32_t point = m.add(PHX_MC_CAD_NODE_INTERSECTION,
                              {plane, carrier, 2, 0, -1, -1, 0, 2, 2}, reals);
  const int32_t pcurve = m.add(PHX_MC_CAD_NODE_INTERSECTION,
                               {plane, carrier, 2, 1, -1, -1, 0, 2, 2}, reals);
  m.surface(plane);
  m.corner(0, 0, 0);
  m.curve(point, 0, 0);
  m.use(0, 0, 0.875, 2.0, pcurve);
  m.close_loop(0);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  m.bind_bounds(workspace);
  EvaluationWorkspace point_workspace;
  const double parameters[2] = {0.875, 2.0};
  double chord[4];
  for (int end = 0; end < 2; ++end) {
    double jet[2];
    PHX_CHECK(geometry.jets(pcurve, parameters + end, 0, jet, point_workspace));
    chord[2 * end] = jet[0], chord[2 * end + 1] = jet[1];
  }
  double truth = 0.0;
  for (int i = 0; i <= 90000; ++i) {
    const double t = 0.875 + 1.125 * i / 90000.0;
    const double wt = t <= 1.0 ? w[0] + t * (w[1] - w[0]) : w[1] + (t - 1.0) * (w[2] - w[1]);
    const double s = (t - 0.875) / 1.125;
    const double du = wt - ((1 - s) * chord[0] + s * chord[2]);
    const double dv = 0.5 * wt * wt - ((1 - s) * chord[1] + s * chord[3]);
    truth = std::max(truth, std::hypot(du, dv));
  }
  double cap = 0.0, padded[4];
  const int32_t status = fidelity_trim_lens(geometry, 0, 0, chord, parameters, workspace, &cap,
                                            padded);
  std::printf("F1 lens: status %d cap %.6e sampled %.6e\n", status, cap, truth);
  PHX_CHECK(status != PHX_MC_OK || cap >= truth);
}

double segment_distance(const double* p, const double* a, const double* b) {
  const double ab[2] = {b[0] - a[0], b[1] - a[1]}, ap[2] = {p[0] - a[0], p[1] - a[1]};
  const double s = std::clamp((ab[0] * ap[0] + ab[1] * ap[1]) / (ab[0] * ab[0] + ab[1] * ab[1]),
                              0.0, 1.0);
  return std::hypot(ap[0] - s * ab[0], ap[1] - s * ab[1]);
}

// F1 (lens), MAC's demo_c: the plane (u, v, 0) trimmed by its intersection
// with the extrusion along z of the C1 quadratic B-spline whose curve is
// (w - 1.5, 0) for w <= 1 and (w - 1.5, -32 (w - 1)^2) after; charts w in
// [0.49994, 0.99994] and [0.99994, 1.015625], use range [0.875, 2], one
// mesh triangle A B C. The source boundary point farthest from the triangle
// is 6.298e-3 away; V3 certified 3.523e-3 before. Fault caught: a seam
// treated as a C2 point in the p-curve remainder.
void intersection_seam_lens_demo() {
  const double c = 64.0, delta = 0x1p-6, eps = 0x1p-14, alpha = 0.125, dw0 = 0.5;
  const double ws = 1.0 - eps, wa = ws - dw0, we = 1.0 + delta;
  const auto gx = [&](double w) { return w - 1.5; };
  const auto gy = [&](double w) { return w <= 1.0 ? 0.0 : -c * (w - 1.0) * (w - 1.0) / 2; };
  Model m;
  const int32_t plane = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0});
  const int32_t spline = m.add(PHX_MC_CAD_NODE_BSPLINE_CURVE, {2, 4, 3},
                               {0, 0, 0, 1, 2, 2, 2, -1.5, 0, 0, -1, 0, 0, 0, 0, 0, 0.5, -c / 2,
                                0, 1, 1, 1, 1});
  const int32_t carrier = m.add(PHX_MC_CAD_NODE_EXTRUSION, {spline}, {0, 0, 1});
  std::vector<double> reals{0.0, 0.0};
  const std::array<double, 9> inverse{1, 0, 0, 0, 1, 0, 0, 0, -1};
  add_chart(reals, {gx(wa), gy(wa), wa, 0}, {gx(ws), gy(ws), ws, 0},
            {gx(wa) - 1, -1, wa, -1, gx(ws) + 1, 1, ws, 1}, inverse);
  add_chart(reals, {gx(ws), gy(ws), ws, 0}, {gx(we), gy(we), we, 0},
            {gx(ws) - 1, -1, ws, -1, gx(we) + 1, 1, we, 1}, inverse);
  const int32_t pcurve = m.add(PHX_MC_CAD_NODE_INTERSECTION,
                               {plane, carrier, 2, 1, -1, -1, 0, 2, 2}, reals);
  const double a[2] = {gx(wa + (1 - alpha) * dw0), 0.0}, b[2] = {gx(we), gy(we)}, cc[2] = {0, -1};
  const int32_t bc = m.add_template("line2", {b[0], b[1], cc[0] - b[0], cc[1] - b[1]});
  const int32_t ca = m.add_template("line2", {cc[0], cc[1], a[0] - cc[0], a[1] - cc[1]});
  m.surface(plane);
  m.corner(0, 0, 0);
  m.curve(spline, 0, 0);
  m.curve_ranges = {0.0, 2.0};
  m.use(0, 0, 1 - alpha, 2.0, pcurve);
  m.use(0, 0, 0.0, 1.0, bc);
  m.use(0, 0, 0.0, 1.0, ca);
  m.close_loop(0);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  // The source boundary's distance to the triangle, by the closed form.
  double worst = 0.0;
  for (int i = 0; i <= 45000; ++i) {
    const double t = (1 - alpha) + (1 + alpha) * i / 45000.0;
    const double w = t <= 1.0 ? wa + t * (ws - wa) : ws + (t - 1.0) * (we - ws);
    const double p[2] = {gx(w), gy(w)};
    worst = std::max(worst, std::min({segment_distance(p, a, b), segment_distance(p, b, cc),
                                      segment_distance(p, cc, a)}));
  }
  Mesh mesh;
  mesh.points = {a[0], a[1], 0, b[0], b[1], 0, cc[0], cc[1], 0};
  mesh.triangles = {0, 1, 2};
  mesh.face_rows = {0};
  mesh.uv = {a[0], a[1], b[0], b[1], cc[0], cc[1]};
  mesh.mesh_triangles = {0, 1, 2};
  mesh.chart_to_mesh = {0};
  mesh.edge_chord = {0, 1, 2};
  mesh.use_patch = {0, 0, 0};
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 1;
  mesh.chord_use = {0, 1, 2};
  mesh.chord_loop = {0, 0, 0};
  mesh.chord_uv = {a[0], a[1], b[0], b[1], b[0], b[1], cc[0], cc[1], cc[0], cc[1], a[0], a[1]};
  mesh.chord_range = m.use_ranges;
  const Verdict verdict = verify(m, mesh, 1e-2, 8);
  std::printf("F1 lens (demo_c): status %d row %d upper %.6e; source boundary %.6e away\n",
              verdict.status, verdict.item_status[0], verdict.upper[0], worst);
  PHX_CHECK(worst > 6.29e-3 && verdict.findings.empty());
  PHX_CHECK(verdict.item_status[0] != PHX_MC_OK || verdict.upper[0] >= worst);
}

// F2: the bilinear B-spline (u, v, 0) on [0, 1]^2 is clamped outside, so
// u = 1 is a C0 line. The chart triangle (0.5, 0), (1.5, 0), (0.5, 1) has
// sampled |S - I_T| = 0.25 at (1, 0). Fault caught: the clamped domain end
// left out of the knot scan.
void clamped_domain_end() {
  Model m;
  const int32_t surface = m.add(PHX_MC_CAD_NODE_BSPLINE_SURFACE, {1, 1, 2, 2},
                                {0, 0, 1, 1, 0, 0, 1, 1,
                                 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1});
  m.surface(surface);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  m.bind_bounds(workspace);
  const double uv[6] = {0.5, 0.0, 1.5, 0.0, 0.5, 1.0};
  const double points[9] = {0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0};
  const double truth = sampled_interpolation_error(uv, points, [](double u, double v) {
    return std::array<double, 3>{std::min(u, 1.0), v, 0.0};
  });
  PHX_CHECK(truth >= 0.2499);
  const auto result = fidelity_triangle(geometry, 0, points, uv, 1.0, 8, workspace);
  std::printf("F2 clamp: status %d upper %.6e sampled %.6e\n", result.status, result.upper, truth);
  PHX_CHECK(result.status == PHX_MC_OK && result.upper >= truth);
}

// MAC's demo_d: the plane z = 0 trimmed by the arc of radius 2.125 about
// (0, -1.875) from A = (-1, 0) to B = (1, 0), which bulges 0.25 above the
// chord AB into the polygon A B D (D = (0, 2)), then the lines B D and D A.
// Rows: the owner sliver A B C of height 0.05 (C = (0, 0.05)), then A C D
// and C B D.
struct ArcFixture {
  Model model;
  Mesh mesh;
};

ArcFixture arc_fixture() {
  ArcFixture f;
  Model& m = f.model;
  const double cy = -1.875, r = 2.125;
  const std::array<double, 2> a{-1, 0}, b{1, 0}, c{0, 0.05}, d{0, 2};
  const double ta = std::atan2(a[1] - cy, a[0]), tb = std::atan2(b[1] - cy, b[0]);
  const int32_t plane = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0});
  const int32_t arc = m.add_template("circle2", {0, cy, 1, 0, 0, 1, r});
  const int32_t bd = m.add_template("line2", {b[0], b[1], d[0] - b[0], d[1] - b[1]});
  const int32_t da = m.add_template("line2", {d[0], d[1], a[0] - d[0], a[1] - d[1]});
  const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
  m.surface(plane);
  m.corner(0, 0, 0);
  m.curve(space, 0, 0);
  m.use(0, 0, ta, tb, arc);
  m.use(0, 0, 0.0, 1.0, bd);
  m.use(0, 0, 0.0, 1.0, da);
  m.close_loop(0);
  Mesh& mesh = f.mesh;
  const std::array<std::array<double, 2>, 4> vertex{a, b, c, d};
  for (const auto& p : vertex) mesh.points.insert(mesh.points.end(), {p[0], p[1], 0.0});
  mesh.triangles = {0, 1, 2, 0, 2, 3, 2, 1, 3};
  mesh.face_rows.assign(3, 0);
  for (int64_t id : mesh.triangles)
    mesh.uv.insert(mesh.uv.end(), {vertex[static_cast<std::size_t>(id)][0],
                                   vertex[static_cast<std::size_t>(id)][1]});
  mesh.mesh_triangles = mesh.triangles;
  mesh.chart_to_mesh = {0, 1, 2};
  mesh.edge_chord = {0, -1, -1, -1, -1, 2, -1, 1, -1};
  mesh.use_patch.assign(3, 0);
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 1;
  mesh.chord_use = {0, 1, 2};
  mesh.chord_loop.assign(3, 0);
  mesh.chord_uv = {a[0], a[1], b[0], b[1], b[0], b[1], d[0], d[1], d[0], d[1], a[0], a[1]};
  mesh.chord_range = m.use_ranges;
  return f;
}

// F4 (demo_d): the neighbour row A C D contains (-0.3, 0.15), inside the
// arc's circle, so 2.125 - |(-0.3, 0.15) - (0, -1.875)| = 0.0779 from the
// trimmed face; only the owner row has the chord (neighbour 2.4e-15
// before). Fault caught: a lens charged to its owner row only.
void lens_charged_to_neighbours() {
  const ArcFixture f = arc_fixture();
  const Verdict verdict = verify(f.model, f.mesh, 0.5, 8);
  const double truth = 2.125 - std::hypot(-0.3, 0.15 + 1.875);
  std::printf("F4 lens: status %d neighbour %.6e (status %d) owner %.6e true %.6e\n",
              verdict.status, verdict.upper[1], verdict.item_status[1], verdict.upper[0], truth);
  PHX_CHECK(verdict.findings.empty());
  PHX_CHECK(verdict.item_status[1] != PHX_MC_OK || verdict.upper[1] >= truth);
}

// F5: use coverage, loop closure and the use/loop binding are checked, and
// a refusal leaves every output unchanged. Faults caught: an empty mesh
// certified with maximum 0; caller loop or use rows not bound to the model.
void use_and_loop_preconditions() {
  const ArcFixture f = arc_fixture();
  const auto refused = [&](const Mesh& mesh) {
    const Verdict verdict = verify(f.model, mesh, 0.5, 8);
    bool unchanged = verdict.maximum == kSentinel;
    for (std::size_t i = 0; i < verdict.upper.size(); ++i)
      unchanged = unchanged && verdict.upper[i] == kSentinel && verdict.item_status[i] == -7 &&
                  verdict.reasons[i] == -7 && verdict.subdivisions[i] == -7;
    return verdict.status == PHX_MC_INVALID_INPUT && unchanged;
  };
  Mesh empty = f.mesh;
  empty.triangles.clear(), empty.face_rows.clear(), empty.uv.clear();
  empty.mesh_triangles.clear(), empty.chart_to_mesh.clear(), empty.edge_chord.clear();
  empty.chord_use.clear(), empty.chord_loop.clear(), empty.chord_uv.clear();
  empty.chord_range.clear();
  const Verdict before = verify(f.model, empty, 0.5, 8);
  std::printf("F5 empty: status %d maximum %.6e\n", before.status, before.maximum);
  PHX_CHECK(before.status == PHX_MC_INVALID_INPUT);
  Mesh loops = f.mesh;
  loops.loop_count = 2;
  PHX_CHECK(refused(loops));
  Mesh range = f.mesh;
  range.use_range[0] = std::nextafter(range.use_range[0], 1.0);
  range.chord_range[0] = range.use_range[0];
  PHX_CHECK(refused(range));
  Mesh reversed = f.mesh;
  std::swap(reversed.use_range[0], reversed.use_range[1]);
  PHX_CHECK(refused(reversed));  // a reversed use range then needs reversed chords
  Mesh gap = f.mesh;
  gap.chord_range[1] = 0.5 * (gap.use_range[0] + gap.use_range[1]);  // the arc chord stops short
  PHX_CHECK(refused(gap));
  Mesh open = f.mesh;
  open.chord_uv[0] = -0.75;  // the loop no longer closes bitwise
  PHX_CHECK(refused(open));
  PHX_CHECK(verify(f.model, f.mesh, 0.5, 8).status == PHX_MC_OK);
  // A use's range must be directed and nonempty.
  Mesh empty_range = f.mesh;
  empty_range.use_range[3] = empty_range.use_range[2];
  PHX_CHECK(verify(f.model, empty_range, 0.5, 8).status == PHX_MC_INVALID_ARGUMENT);
  // Every mesh triangle has exactly one chart image.
  Mesh twice = f.mesh;
  twice.chart_to_mesh = {0, 0, 2};
  const Verdict images = verify(f.model, twice, 0.5, 8);
  bool link = false, missing = false;
  for (const auto& finding : images.findings) {
    link |= finding.kind == PHX_MC_FIDELITY_CHART_MESH_LINK && finding.triangle == 1;
    missing |= finding.kind == PHX_MC_FIDELITY_MISSING_MESH_TRIANGLE && finding.other_triangle == 1;
  }
  PHX_CHECK(images.status == PHX_MC_UNRESOLVED && link && missing);
}

// F3 and the sphere pole sliver: unit sphere, chart corners c0 = (0, pi/2 -
// 1/8), c1 = (1, pi/2 - 1/8), pole c2 = (1, pi/2), c3 = (0, pi/2). The sliver
// (Q, P, P) stands for the mesh edge P-Q. Fault caught: only the pole vertex
// checked, so a sliver whose Q no mesh triangle uses is certified.
void sliver_edge_in_mesh() {
  Model m;
  const int32_t sphere = m.add_template("sphere", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1});
  const double top = M_PI / 2;
  const std::array<std::array<double, 2>, 4> c{{{0, top - 0.125}, {1, top - 0.125}, {1, top},
                                                {0, top}}};
  m.surface(sphere);
  const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
  m.corner(0, 0, 0), m.corner(0, 0, 0), m.corner(0, 0, 1);
  m.curve(space, 0, 1), m.curve(space, 1, 2), m.curve(space, 2, 0);
  const auto line = [&](int from, int to) {
    return m.add_template("line2", {c[from][0], c[from][1], c[to][0] - c[from][0],
                                    c[to][1] - c[from][1]});
  };
  m.use(0, 0, 0, 1, line(0, 1));
  m.use(0, 1, 0, 1, line(1, 2));
  m.use(1, 2, 0, 1, line(2, 3));
  m.use(0, 2, 0, 1, line(3, 0));
  m.close_loop(0);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  EvaluationWorkspace point_workspace;
  Mesh mesh;
  for (int k : {0, 1, 2, 0}) {
    double xyz[3];
    PHX_CHECK(geometry.jets(sphere, c[static_cast<std::size_t>(k)].data(), 0, xyz,
                            point_workspace));
    mesh.points.insert(mesh.points.end(), xyz, xyz + 3);
  }
  mesh.triangles = {0, 1, 2, 0, 2, 2};
  mesh.face_rows = {0, 0};
  for (int k : {0, 1, 2, 0, 2, 3})
    mesh.uv.insert(mesh.uv.end(), c[static_cast<std::size_t>(k)].begin(),
                   c[static_cast<std::size_t>(k)].end());
  mesh.mesh_triangles = {0, 1, 2};
  mesh.chart_to_mesh = {0, -1};
  mesh.edge_chord = {0, 1, -1, -1, 2, 3};
  mesh.use_patch.assign(4, 0);
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 1;
  mesh.chord_use = {0, 1, 2, 3};
  mesh.chord_loop.assign(4, 0);
  for (int k = 0; k < 4; ++k) {
    const auto& from = c[static_cast<std::size_t>(k)];
    const auto& to = c[static_cast<std::size_t>((k + 1) % 4)];
    mesh.chord_uv.insert(mesh.chord_uv.end(), {from[0], from[1], to[0], to[1]});
    mesh.chord_range.insert(mesh.chord_range.end(), {0.0, 1.0});
  }
  const Verdict good = verify(m, mesh, 0.1, 8);
  std::printf("F3 sliver (edge in mesh): status %d sliver %.6e\n", good.status, good.upper[1]);
  PHX_CHECK(good.status == PHX_MC_OK && good.findings.empty());
  // Q = point 3 (the coordinates of point 0, used by no mesh triangle).
  mesh.triangles[3] = 3;
  const Verdict bad = verify(m, mesh, 0.1, 8);
  std::printf("F3 sliver (edge not in mesh): status %d sliver %.6e findings %zu\n", bad.status,
              bad.upper[1], bad.findings.size());
  bool link = false;
  for (const auto& finding : bad.findings)
    link |= finding.kind == PHX_MC_FIDELITY_CHART_MESH_LINK && finding.triangle == 1;
  PHX_CHECK(bad.status == PHX_MC_UNRESOLVED && link &&
            bad.reasons[1] == PHX_MC_FIDELITY_EDGE_ASSOCIATION);
}

// F3, MAC's demo_e: unit sphere with the pole P = (0, 0, 1); the sliver
// (P, P, Q) has the chart corners F = (0, pi/2), G = (0.5, pi/2), apex
// (0.25, pi/2 - 0.3), and its third vertex Q = S(apex) is used by no mesh
// triangle (the only one, on a second plane face, is P, (0,0,2), (-1,0,1)).
// S(apex) is >= 0.286 from the mesh; V3 certified 2.70e-2 at tolerance 0.1
// before. Fault caught: only the pole vertex checked, not the edge P-Q.
void sliver_segment_in_mesh_demo() {
  Model m;
  const double vp = 0x1.921fb54442d18p+0;
  const std::array<std::array<double, 2>, 3> sliver{{{0, vp}, {0.5, vp}, {0.25, vp - 0.3}}};
  const std::array<std::array<double, 2>, 3> flat{{{0, 0}, {1, 0}, {0, 1}}};
  const int32_t sphere = m.add_template("sphere", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1});
  const int32_t plane = m.add_template("plane", {0, 0, 1, 0, 0, 1, -1, 0, 0});
  std::vector<int32_t> lines;
  for (const auto* corners : {&sliver, &flat})
    for (int k = 0; k < 3; ++k) {
      const auto& from = (*corners)[static_cast<std::size_t>(k)];
      const auto& to = (*corners)[static_cast<std::size_t>((k + 1) % 3)];
      lines.push_back(m.add_template("line2", {from[0], from[1], to[0] - from[0],
                                               to[1] - from[1]}));
    }
  const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
  m.surface(sphere);
  m.surface(plane);
  m.corner(0, 0, 1);
  m.curve(space, 0, 0);
  for (int k = 0; k < 6; ++k) {
    m.use(k == 0 ? 1 : 0, 0, 0.0, 1.0, lines[static_cast<std::size_t>(k)]);
    if (k % 3 == 2) m.close_loop(k / 3);
  }
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  EvaluationWorkspace point_workspace;
  double apex[3];
  PHX_CHECK(geometry.jets(sphere, sliver[2].data(), 0, apex, point_workspace));
  Mesh mesh;
  mesh.points = {0, 0, 1, apex[0], apex[1], apex[2], 0, 0, 2, -1, 0, 1};
  mesh.triangles = {0, 0, 1, 0, 2, 3};
  mesh.face_rows = {0, 1};
  for (const auto* corners : {&sliver, &flat})
    for (const auto& p : *corners) mesh.uv.insert(mesh.uv.end(), p.begin(), p.end());
  mesh.mesh_triangles = {0, 2, 3};
  mesh.chart_to_mesh = {-1, 0};
  mesh.edge_chord = {0, 1, 2, 3, 4, 5};
  mesh.use_patch = {0, 0, 0, 1, 1, 1};
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 2;
  mesh.chord_use = {0, 1, 2, 3, 4, 5};
  mesh.chord_loop = {0, 0, 0, 1, 1, 1};
  for (const auto* corners : {&sliver, &flat})
    for (int k = 0; k < 3; ++k) {
      const auto& from = (*corners)[static_cast<std::size_t>(k)];
      const auto& to = (*corners)[static_cast<std::size_t>((k + 1) % 3)];
      mesh.chord_uv.insert(mesh.chord_uv.end(), {from[0], from[1], to[0], to[1]});
      mesh.chord_range.insert(mesh.chord_range.end(), {0.0, 1.0});
    }
  const Verdict verdict = verify(m, mesh, 0.1, 8);
  std::printf("F3 sliver (demo_e): status %d row 0 status %d upper %.6e; S(apex) x = %.6f\n",
              verdict.status, verdict.item_status[0], verdict.upper[0], apex[0]);
  bool link = false;
  for (const auto& finding : verdict.findings)
    link |= finding.kind == PHX_MC_FIDELITY_CHART_MESH_LINK && finding.triangle == 0;
  PHX_CHECK(apex[0] >= 0.286 && verdict.status == PHX_MC_UNRESOLVED && link);
}

// F5, MAC's demo_f: a unit-square plane face with one loop of four line
// uses; no chart triangles and no chords. V3 alone returned OK with maximum
// 0 before. Fault caught: use coverage assumed, not checked.
void empty_mesh_demo() {
  Model m;
  const double p[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  const int32_t plane = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0});
  std::vector<int32_t> lines;
  for (int k = 0; k < 4; ++k)
    lines.push_back(m.add_template("line2", {p[k][0], p[k][1], p[(k + 1) % 4][0] - p[k][0],
                                             p[(k + 1) % 4][1] - p[k][1]}));
  const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
  m.surface(plane);
  m.corner(0, 0, 0);
  m.curve(space, 0, 0);
  for (int k = 0; k < 4; ++k) m.use(0, 0, 0.0, 1.0, lines[static_cast<std::size_t>(k)]);
  m.close_loop(0);
  Mesh mesh;
  mesh.use_patch = {0, 0, 0, 0};
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 1;
  const Verdict verdict = verify(m, mesh, 1e-6, 8);
  std::printf("F5 empty (demo_f): status %d maximum %g\n", verdict.status, verdict.maximum);
  PHX_CHECK(verdict.status == PHX_MC_INVALID_INPUT && verdict.maximum == kSentinel);
}

// F6: the lens mean-value bound needs S Lipschitz on the padded box. The
// degree-1 surface with the double knot u = 0.5 jumps in z there; a chord
// across it is refused. Fault caught: |DS| over a box with a discontinuity.
void lens_requires_lipschitz_surface() {
  Model m;
  const int32_t surface = m.add(
      PHX_MC_CAD_NODE_BSPLINE_SURFACE, {1, 1, 4, 2},
      {0, 0, 0.5, 0.5, 1, 1, 0, 0, 1, 1,
       0, 0, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0, 1, 0.5, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1});
  const int32_t pcurve = m.add_template("line2", {0.25, 0.5, 0.5, 0.0});
  const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
  m.surface(surface);
  m.corner(0, 0, 0);
  m.curve(space, 0, 0);
  m.use(0, 0, 0.0, 1.0, pcurve);
  m.close_loop(0);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  m.bind_bounds(workspace);
  const double chord[4] = {0.25, 0.5, 0.75, 0.5}, parameters[2] = {0.0, 1.0};
  double cap = 0.0, padded[4];
  const int32_t status = fidelity_trim_lens(geometry, 0, 0, chord, parameters, workspace, &cap,
                                            padded);
  std::printf("F6 lens over a jump: status %d cap %.6e\n", status, cap);
  PHX_CHECK(status == PHX_MC_UNRESOLVED);
}

// The plane z = 0.5 and the clamped B-spline S2(u, v) = (u, u^2, v) on [0,
// 1]^2, one chart u in [ua, ub] (MAC's re-read demo_h / demo_h2): the
// intersection stays strictly inside S2's domain.
int32_t parabola_carrier(Model& m, int32_t& plane, double ua, double ub, int64_t output) {
  plane = m.add_template("plane", {0, 0, 0.5, 1, 0, 0, 0, 1, 0});
  std::vector<double> r = {0, 0, 0, 1, 1, 1, 0, 0, 1, 1};
  const double cx[3] = {0, 0.5, 1}, cy[3] = {0, 0, 1};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j) r.insert(r.end(), {cx[i], cy[i], double(j)});
  r.insert(r.end(), 6, 1.0);
  const int32_t spline = m.add(PHX_MC_CAD_NODE_BSPLINE_SURFACE, {2, 1, 3, 2}, r);
  return m.add(PHX_MC_CAD_NODE_INTERSECTION, {plane, spline, 1, output, -1, -1, 0, 2},
               {1.0, 0.0, 0, 0, ua, 0.5, 0, 0, ub, 0.5, -1, -1, ua, 0.25, 2, 2, ub, 0.75,
                1, 0, 0, 0, 1, 0, 0, 0, -1});
}

// A (demo_h): the trim chord over u in [1/4, 1/4 + 1/32]; the arc is 2.16e-4
// from it. Refused before (the clamped ends of S2 counted over (-inf, inf));
// certified at 4.06e-4 originally. Fault caught: a carrier asked over the
// whole parameter plane.
void carrier_inside_domain_lens() {
  Model m;
  int32_t plane = -1;
  const double ua = 0.25, ub = ua + 1.0 / 32;
  const int32_t pc = parabola_carrier(m, plane, ua, ub, 1);
  const double c[2] = {ub, ua * ua - 0.5};
  m.surface(plane);
  m.corner(0, 0, 0);
  m.curve(m.add_template("line3", {0, 0, 0, 1, 0, 0}), 0, 0);
  m.use(0, 0, 0, 1, pc);
  CadGeometry geometry;
  EvaluationWorkspace point_workspace;
  // The p-curve ends, by point evaluation (the line uses join them).
  double ends[4];
  {
    Model ends_model = m;
    ends_model.close_loop(0);
    const phx_mc_cad_model view = ends_model.view();
    PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
    const double t0 = 0, t1 = 1;
    PHX_CHECK(geometry.jets(pc, &t0, 0, ends, point_workspace));
    PHX_CHECK(geometry.jets(pc, &t1, 0, ends + 2, point_workspace));
  }
  const double a[2] = {ends[0], ends[1]}, b[2] = {ends[2], ends[3]};
  m.use(0, 0, 0, 1, m.add_template("line2", {ub, ub * ub, c[0] - ub, c[1] - ub * ub}));
  m.use(0, 0, 0, 1, m.add_template("line2", {c[0], c[1], ua - c[0], ua * ua - c[1]}));
  m.close_loop(0);
  double worst = 0.0;  // the arc y = x^2 against its chord, closed form
  for (int i = 0; i <= 2000; ++i) {
    const double x = ua + (ub - ua) * i / 2000.0;
    const double p[2] = {x, x * x};
    worst = std::max(worst, segment_distance(p, a, b));
  }
  Mesh mesh;
  mesh.points = {a[0], a[1], 0.5, b[0], b[1], 0.5, c[0], c[1], 0.5};
  mesh.triangles = {0, 1, 2};
  mesh.face_rows = {0};
  mesh.uv = {a[0], a[1], b[0], b[1], c[0], c[1]};
  mesh.mesh_triangles = {0, 1, 2};
  mesh.chart_to_mesh = {0};
  mesh.edge_chord = {0, 1, 2};
  mesh.use_patch = {0, 0, 0};
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 1;
  mesh.chord_use = {0, 1, 2};
  mesh.chord_loop = {0, 0, 0};
  mesh.chord_uv = {a[0], a[1], b[0], b[1], b[0], b[1], c[0], c[1], c[0], c[1], a[0], a[1]};
  mesh.chord_range = m.use_ranges;
  const Verdict verdict = verify(m, mesh, 1e-3, 8);
  std::printf("A lens (demo_h): status %d row %d reason %d upper %.6e; true %.6e\n",
              verdict.status, verdict.item_status[0], verdict.reasons[0], verdict.upper[0], worst);
  PHX_CHECK(verdict.status == PHX_MC_OK && verdict.upper[0] >= worst && verdict.upper[0] < 1e-3);
}

// A (demo_h2 and its family): the extrusion along z of a curve on S2 that
// stays strictly inside S2's domain, reached through an intersection node
// (MAC's demo_h2: x = 1/4 + t/2), an isoparametric node (v = 1/2) and a
// surface-mapped node (p-curve (t, 1/2)); chart triangle (0.25, 0),
// (0.5, 0), (0.25, 0.25). Each certifies at tolerance 1e-2 in at most a few
// splits (demo_h2: 1,302 splits before). Fault caught: as above, on a leaf.
void carrier_inside_domain_leaf() {
  for (int kind = 0; kind < 3; ++kind) {
    Model m;
    int32_t plane = -1;
    const int32_t intersection = parabola_carrier(m, plane, 0.25, 0.75, 0);
    const int32_t spline = intersection - 1;
    int32_t curve = intersection;
    if (kind == 1) curve = m.add(PHX_MC_CAD_NODE_ISOPARAMETRIC, {spline, 1}, {0.5});
    if (kind == 2)
      curve = m.add(PHX_MC_CAD_NODE_SURFACE_MAPPED,
                    {spline, m.add_template("line2", {0, 0.5, 1, 0})}, {});
    const int32_t swept = m.add(PHX_MC_CAD_NODE_EXTRUSION, {curve}, {0, 0, 1});
    m.surface(swept);
    const phx_mc_cad_model view = m.view();
    CadGeometry geometry;
    PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
    IntervalEvaluationWorkspace workspace;
    m.bind_bounds(workspace);
    EvaluationWorkspace point_workspace;
    const double uv[6] = {0.25, 0, 0.5, 0, 0.25, 0.25};
    double points[9];
    for (int i = 0; i < 3; ++i)
      PHX_CHECK(geometry.jets(swept, uv + 2 * i, 0, points + 3 * i, point_workspace));
    const double base = kind == 0 ? 0.5 : 0.0;  // the plane z = 0.5 carries the intersection
    const double truth = sampled_interpolation_error(uv, points, [&](double t, double v) {
      const double x = kind == 0 ? 0.25 + 0.5 * t : t;
      return std::array<double, 3>{x, x * x, base + (kind == 0 ? 0.0 : 0.5) + v};
    });
    const char* name[3] = {"intersection", "isoparametric", "surface-mapped"};
    for (const double factor : {4.0, 2.0, 1.25}) {
      const double tolerance = factor * truth;
      const auto result = fidelity_triangle(geometry, 0, points, uv, tolerance, 8, workspace);
      std::printf("A leaf (%s) tolerance %.4g (x%g sampled): status %d upper %.6e splits %d "
                  "(sampled %.6e)\n", name[kind], tolerance, factor, result.status, result.upper,
                  result.subdivisions, truth);
      PHX_CHECK(result.status != PHX_MC_OK || result.upper >= truth);
      if (factor >= 2.0) PHX_CHECK(result.status == PHX_MC_OK && result.subdivisions <= 16);
      // S is quadratic in t with a constant Hessian here: the remainder about
      // the box centre (half_extent) is sharp (2x truth with box extents).
      if (kind > 0) PHX_CHECK(result.status == PHX_MC_OK && result.upper <= 1.0625 * truth);
    }
  }
}

// A (demo_h family, trims): the faces of carrier_inside_domain_leaf trimmed
// by the arc of the circle about (0.375, 0.025) of radius 1/8 from
// A = (0.45, 0.125) to B = (0.3, 0.125) (chart sagitta 0.025, strictly
// inside S2's domain), one chart triangle A B C with C = (0.375, 0). Each
// certifies through the whole entry, the lens included, at tolerance 1/8 in
// the splits of the first V3 (0186b551c and dd7715b2c: 0 per kind; c7a0e82fc:
// 5, 14, 14, bounds 0.092 / 0.125 against 0.058 / 0.084). Fault caught: a
// trim on an intersection, isoparametric or surface-mapped carrier asked
// over the whole parameter plane.
void carrier_inside_domain_trim() {
  const double a[2] = {0.45, 0.125}, b[2] = {0.3, 0.125}, c[2] = {0.375, 0.0};
  const double cx = 0.375, cy = 0.025, r = 0.125;
  const double ta = std::atan2(a[1] - cy, a[0] - cx), tb = std::atan2(b[1] - cy, b[0] - cx);
  for (int kind = 0; kind < 3; ++kind) {
    Model m;
    int32_t plane = -1;
    const int32_t intersection = parabola_carrier(m, plane, 0.25, 0.75, 0);
    const int32_t spline = intersection - 1;
    int32_t curve = intersection;
    if (kind == 1) curve = m.add(PHX_MC_CAD_NODE_ISOPARAMETRIC, {spline, 1}, {0.5});
    if (kind == 2)
      curve = m.add(PHX_MC_CAD_NODE_SURFACE_MAPPED,
                    {spline, m.add_template("line2", {0, 0.5, 1, 0})}, {});
    const int32_t swept = m.add(PHX_MC_CAD_NODE_EXTRUSION, {curve}, {0, 0, 1});
    const int32_t arc = m.add_template("circle2", {cx, cy, 1, 0, 0, 1, r});
    const int32_t bc = m.add_template("line2", {b[0], b[1], c[0] - b[0], c[1] - b[1]});
    const int32_t ca = m.add_template("line2", {c[0], c[1], a[0] - c[0], a[1] - c[1]});
    const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
    m.surface(swept);
    m.corner(0, 0, 0);
    m.curve(space, 0, 0);
    m.use(0, 0, ta, tb, arc);
    m.use(0, 0, 0.0, 1.0, bc);
    m.use(0, 0, 0.0, 1.0, ca);
    m.close_loop(0);
    const phx_mc_cad_model view = m.view();
    CadGeometry geometry;
    PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
    EvaluationWorkspace point_workspace;
    Mesh mesh;
    mesh.uv = {a[0], a[1], b[0], b[1], c[0], c[1]};
    mesh.points.resize(9);
    for (int i = 0; i < 3; ++i)
      PHX_CHECK(geometry.jets(swept, mesh.uv.data() + 2 * i, 0, mesh.points.data() + 3 * i,
                              point_workspace));
    // S on the arc against the mesh edge S(A) S(B): a lower bound of the row's distance.
    double worst = 0.0;
    const double* pa = mesh.points.data();
    const double* pb = mesh.points.data() + 3;
    for (int i = 0; i <= 2000; ++i) {
      const double t = ta + (tb - ta) * i / 2000.0;
      const double uv[2] = {cx + r * std::cos(t), cy + r * std::sin(t)};
      double s[3];
      PHX_CHECK(geometry.jets(swept, uv, 0, s, point_workspace));
      double ab = 0.0, ap = 0.0;
      for (int k = 0; k < 3; ++k) {
        ab += (pb[k] - pa[k]) * (pb[k] - pa[k]);
        ap += (s[k] - pa[k]) * (pb[k] - pa[k]);
      }
      const double w = std::clamp(ap / ab, 0.0, 1.0);
      double squared = 0.0;
      for (int k = 0; k < 3; ++k) {
        const double d = s[k] - (pa[k] + w * (pb[k] - pa[k]));
        squared += d * d;
      }
      worst = std::max(worst, std::sqrt(squared));
    }
    mesh.triangles = {0, 1, 2};
    mesh.face_rows = {0};
    mesh.mesh_triangles = {0, 1, 2};
    mesh.chart_to_mesh = {0};
    mesh.edge_chord = {0, 1, 2};
    mesh.use_patch = {0, 0, 0};
    mesh.use_range = m.use_ranges;
    mesh.loop_count = 1;
    mesh.chord_use = {0, 1, 2};
    mesh.chord_loop = {0, 0, 0};
    mesh.chord_uv = {a[0], a[1], b[0], b[1], b[0], b[1], c[0], c[1], c[0], c[1], a[0], a[1]};
    mesh.chord_range = m.use_ranges;
    const Verdict verdict = verify(m, mesh, 0.125, 8);
    const char* name[3] = {"intersection", "isoparametric", "surface-mapped"};
    std::printf("A trim (%s): status %d row %d reason %d upper %.6e splits %d; arc %.6e\n",
                name[kind], verdict.status, verdict.item_status[0], verdict.reasons[0],
                verdict.upper[0], verdict.subdivisions[0], worst);
    // Lens cap from per-axis p-curve remainders and |S_u|, |S_v| (1.35x / 1.82x
    // the sampled arc distance; 2.32x / 3.30x with one Frobenius |DS|).
    PHX_CHECK(verdict.status == PHX_MC_OK && verdict.upper[0] >= worst &&
              verdict.upper[0] <= 2.0 * worst && verdict.subdivisions[0] <= 4);
  }
}

// C (demo_g): the square [0, 4]^2 with the clockwise hole H; (a) the hole
// filled by a clockwise triangle, (b) the hole loop's uses given reversed and
// the hole covered by a counter-clockwise triangle. V3 certified both
// (maximum 9.7e-15) while the hole triangle's centroid is 0.586 from the
// face. Fault caught: orientation and per-face use direction unchecked.
void hole_orientation_demo() {
  const double o[4][2] = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
  const double h[3][2] = {{1, 1}, {1, 3}, {3, 1}};
  Model m;
  const int32_t plane = m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0});
  m.surface(plane);
  m.corner(0, 0, 0);
  m.curve(m.add_template("line3", {0, 0, 0, 1, 0, 0}), 0, 0);
  for (int k = 0; k < 4; ++k)
    m.use(0, 0, 0, 1, m.add_template("line2", {o[k][0], o[k][1], o[(k + 1) % 4][0] - o[k][0],
                                               o[(k + 1) % 4][1] - o[k][1]}));
  m.close_loop(0);
  for (int k = 0; k < 3; ++k)
    m.use(0, 0, 0, 1, m.add_template("line2", {h[k][0], h[k][1], h[(k + 1) % 3][0] - h[k][0],
                                               h[(k + 1) % 3][1] - h[k][1]}));
  m.close_loop(0);
  Mesh base;
  base.points = {0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0, 1, 1, 0, 1, 3, 0, 3, 1, 0};
  base.face_rows = {0, 0, 0};
  base.chart_to_mesh = {0, 1, 2};
  base.use_patch.assign(7, 0);
  base.loop_count = 2;
  base.chord_use = {0, 1, 2, 3, 4, 5, 6};
  base.chord_loop = {0, 0, 0, 0, 1, 1, 1};
  for (int k = 0; k < 4; ++k) {
    base.chord_uv.insert(base.chord_uv.end(), {o[k][0], o[k][1], o[(k + 1) % 4][0],
                                               o[(k + 1) % 4][1]});
    base.chord_range.insert(base.chord_range.end(), {0, 1});
  }
  Mesh filled = base;
  filled.use_range = m.use_ranges;
  for (int k = 0; k < 3; ++k) {
    filled.chord_uv.insert(filled.chord_uv.end(), {h[k][0], h[k][1], h[(k + 1) % 3][0],
                                                   h[(k + 1) % 3][1]});
    filled.chord_range.insert(filled.chord_range.end(), {0, 1});
  }
  filled.triangles = {0, 1, 2, 0, 2, 3, 4, 5, 6};
  filled.uv = {0, 0, 4, 0, 4, 4, 0, 0, 4, 4, 0, 4, 1, 1, 1, 3, 3, 1};
  filled.mesh_triangles = filled.triangles;
  filled.edge_chord = {0, 1, -1, -1, 2, 3, 4, 5, 6};
  const Verdict a = verify(m, filled, 1e-6, 8);
  Mesh reversed = base;
  reversed.use_range = m.use_ranges;
  for (int k = 0; k < 3; ++k) {
    reversed.chord_uv.insert(reversed.chord_uv.end(), {h[(k + 1) % 3][0], h[(k + 1) % 3][1],
                                                       h[k][0], h[k][1]});
    reversed.chord_range.insert(reversed.chord_range.end(), {1, 0});
    std::swap(reversed.use_range[2 * (4 + k)], reversed.use_range[2 * (4 + k) + 1]);
  }
  reversed.triangles = {0, 1, 2, 0, 2, 3, 4, 6, 5};
  reversed.uv = {0, 0, 4, 0, 4, 4, 0, 0, 4, 4, 0, 4, 1, 1, 3, 1, 1, 3};
  reversed.mesh_triangles = reversed.triangles;
  reversed.edge_chord = {0, 1, -1, -1, 2, 3, 6, 5, 4};
  const Verdict b = verify(m, reversed, 1e-6, 8);
  std::printf("C hole (demo_g): filled status %d maximum %g; reversed status %d maximum %g\n",
              a.status, a.maximum, b.status, b.maximum);
  PHX_CHECK(a.status == PHX_MC_INVALID_INPUT && a.maximum == kSentinel);
  PHX_CHECK(b.status == PHX_MC_INVALID_INPUT && b.maximum == kSentinel);
}

// D (demo_i): a model surface on no loop and no rows; certified empty before.
void surface_without_loop_demo() {
  Model m;
  m.surface(m.add_template("sphere", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1}));
  const Verdict verdict = verify(m, Mesh{}, 1e-6, 8);
  std::printf("D surface on no loop (demo_i): status %d maximum %g\n", verdict.status,
              verdict.maximum);
  PHX_CHECK(verdict.status == PHX_MC_INVALID_INPUT && verdict.maximum == kSentinel);
}

// B (index_overflow): two point-like chords 1 apart with padded boxes of
// 1e-12, and a row box over both: about 1e12 cells per axis, whose product
// overflowed int64 (undefined; a 20 s hang). The charge is the larger cap.
void lens_index_overflow() {
  const int32_t faces[2] = {0, 0};
  const double extent = 1e-12, spread = 1.0;
  const double boxes[8] = {0, 0, extent, extent, spread, spread, spread + extent, spread + extent};
  const double caps[2] = {0.25, 0.5};
  const double row[4] = {0, 0, spread, spread};
  const double charge = fidelity_lens_charge(1, 2, faces, boxes, caps, 0, row);
  std::printf("B lens index (index_overflow): charge %.2f\n", charge);
  PHX_CHECK(charge == 0.5);
}

// Pole chart (verify_cad_pole_chart.hpp) on a sphere of radius 2 about
// (0.5, -0.25, 1): fan rows (one pole corner) and band rows (none) at both
// poles, against an independent reference: S(x, y) = o + r (sin rho / rho
// (x, y) + sigma cos rho a) sampled densely over the pole-chart triangle
// (the same-parameter deviation), and the exact distance | |p - o| - r | of
// densely sampled mesh points. Fault caught: a jet sign, a series
// coefficient or a missing Taylor term (the bound would fall below the
// sampled deviation or the jets miss S); a displaced pole or band vertex
// must refuse at the tolerance; a non-sphere face is refused.
void pole_chart_rows() {
  const double r = 2.0, centre[3] = {0.5, -0.25, 1.0}, top = M_PI / 2;
  Model m;
  const int32_t sphere =
      m.add_template("sphere", {centre[0], centre[1], centre[2], 1, 0, 0, 0, 1, 0, 0, 0, 1, r});
  m.surface(sphere);
  m.surface(m.add_template("plane", {0, 0, 0, 1, 0, 0, 0, 1, 0}));
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  m.bind_bounds(workspace);
  EvaluationWorkspace point_workspace;
  const auto reference = [&](double x, double y, int sigma) {
    const double rho = std::hypot(x, y), g = rho == 0.0 ? 1.0 : std::sin(rho) / rho;
    return std::array<double, 3>{centre[0] + r * g * x, centre[1] + r * g * y,
                                 centre[2] + sigma * r * std::cos(rho)};
  };
  // Jets enclose the source map at the matching chart point.
  for (int sigma : {1, -1})
    for (int k = 0; k < 16; ++k) {
      const double uv[2] = {0.4 * k, sigma * (top - 0.05 * (k + 1))};
      double image[4], xyz[3], lo[12], hi[12];
      PHX_CHECK(fidelity_pole_image(geometry, 0, workspace, sigma, uv, image) == PHX_MC_OK);
      const double box_lower[2] = {image[0], image[2]}, box_upper[2] = {image[1], image[3]};
      PHX_CHECK(fidelity_pole_jets(geometry, 0, workspace, sigma, box_lower, box_upper, lo, hi) ==
                PHX_MC_OK);
      PHX_CHECK(geometry.jets(sphere, uv, 0, xyz, point_workspace));
      for (int axis = 0; axis < 3; ++axis)
        PHX_CHECK(lo[axis] - 1e-14 <= xyz[axis] && xyz[axis] <= hi[axis] + 1e-14);
      // Second differences of the reference lie in the Hessian enclosure of
      // a box holding the stencil (up to the O(step^2) stencil error).
      const double x = 0.5 * (image[0] + image[1]), y = 0.5 * (image[2] + image[3]), e = 1e-3;
      const double wide_lower[2] = {x - e, y - e}, wide_upper[2] = {x + e, y + e};
      PHX_CHECK(fidelity_pole_jets(geometry, 0, workspace, sigma, wide_lower, wide_upper, lo,
                                   hi) == PHX_MC_OK);
      for (int axis = 0; axis < 3; ++axis) {
        const auto at = [&](double dx, double dy) {
          return reference(x + dx, y + dy, sigma)[static_cast<std::size_t>(axis)];
        };
        const double sxx = (at(e, 0) - 2 * at(0, 0) + at(-e, 0)) / (e * e);
        const double syy = (at(0, e) - 2 * at(0, 0) + at(0, -e)) / (e * e);
        const double sxy = (at(e, e) - at(e, -e) - at(-e, e) + at(-e, -e)) / (4 * e * e);
        PHX_CHECK(lo[3 + axis] - 1e-5 <= sxx && sxx <= hi[3 + axis] + 1e-5);
        PHX_CHECK(lo[6 + axis] - 1e-5 <= sxy && sxy <= hi[6 + axis] + 1e-5);
        PHX_CHECK(lo[9 + axis] - 1e-5 <= syy && syy <= hi[9 + axis] + 1e-5);
      }
    }
  struct Row { const char* name; std::array<double, 6> uv; int mask; };
  // Unit-sphere colatitudes 0.2 (ring 1) and 0.4 (ring 2), ring 1 every pi/4.
  const double r1 = top - 0.2, r2 = top - 0.4;
  const Row rows[] = {{"fan", {0.3, top, 0.0, r1, M_PI / 4, r1}, 1},
                      {"band", {0.0, r1, M_PI / 8, r2, M_PI / 4, r1}, 0},
                      {"band outer", {M_PI / 4, r1, M_PI / 8, r2, 3 * M_PI / 8, r2}, 0}};
  const double tolerance = 0.01 * r * r;
  for (int sigma : {1, -1})
    for (const Row& row : rows) {
      double uv[6], points[9], xy[6];
      for (int j = 0; j < 3; ++j) {
        uv[2 * j] = row.uv[static_cast<std::size_t>(2 * j)];
        uv[2 * j + 1] = sigma * row.uv[static_cast<std::size_t>(2 * j + 1)];
        PHX_CHECK(geometry.jets(sphere, uv + 2 * j, 0, points + 3 * j, point_workspace));
        double image[4];
        PHX_CHECK(fidelity_pole_image(geometry, 0, workspace, sigma, uv + 2 * j, image) ==
                  PHX_MC_OK);
        const bool at_pole = (row.mask >> j) & 1;
        xy[2 * j] = at_pole ? 0.0 : image[0] + 0.5 * (image[1] - image[0]);
        xy[2 * j + 1] = at_pole ? 0.0 : image[2] + 0.5 * (image[3] - image[2]);
      }
      double sampled = 0.0, distance = 0.0;
      const int n = 96;
      for (int a = 0; a <= n; ++a)
        for (int b = 0; a + b <= n; ++b) {
          const double w[3] = {static_cast<double>(a) / n, static_cast<double>(b) / n,
                               static_cast<double>(n - a - b) / n};
          const double x = w[0] * xy[0] + w[1] * xy[2] + w[2] * xy[4];
          const double y = w[0] * xy[1] + w[1] * xy[3] + w[2] * xy[5];
          const auto s = reference(x, y, sigma);
          double deviation = 0.0, radius = 0.0;
          for (int axis = 0; axis < 3; ++axis) {
            const double p = w[0] * points[axis] + w[1] * points[3 + axis] + w[2] * points[6 + axis];
            deviation += (s[static_cast<std::size_t>(axis)] - p) * (s[static_cast<std::size_t>(axis)] - p);
            radius += (p - centre[axis]) * (p - centre[axis]);
          }
          sampled = std::max(sampled, std::sqrt(deviation));
          distance = std::max(distance, std::fabs(std::sqrt(radius) - r));
        }
      const auto pole = fidelity_pole_triangle(geometry, 0, points, uv, sigma, row.mask,
                                               tolerance, 12, workspace);
      const auto chart = fidelity_triangle(geometry, 0, points, uv, tolerance, 12, workspace);
      std::printf("P pole chart (%s, sigma %d): status %d reason %d splits %d upper %.6e sampled "
                  "%.6e distance %.6e uv-chart status %d upper %.6e\n",
                  row.name, sigma, pole.status, pole.reason, pole.subdivisions, pole.upper,
                  sampled, distance, chart.status, chart.upper);
      PHX_CHECK(pole.status == PHX_MC_OK && pole.upper <= tolerance && pole.upper >= sampled &&
                pole.upper >= distance && sampled >= distance);
      // A displaced vertex (pole for the fan row, ring otherwise) by 0.05 r
      // along the normal refuses at the tolerance.
      const int moved = row.mask != 0 ? 0 : 1;
      double shifted[9];
      std::copy(points, points + 9, shifted);
      for (int axis = 0; axis < 3; ++axis)
        shifted[3 * moved + axis] += 0.05 * (points[3 * moved + axis] - centre[axis]);
      const auto refused = fidelity_pole_triangle(geometry, 0, shifted, uv, sigma, row.mask,
                                                  tolerance, 12, workspace);
      PHX_CHECK(refused.status == PHX_MC_UNRESOLVED);
    }
  // Families without a certified pole chart are refused.
  const double flat_uv[6] = {0, 0, 1, 0, 0, 1}, flat[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const auto plane = fidelity_pole_triangle(geometry, 1, flat, flat_uv, 1, 0, 1.0, 8, workspace);
  PHX_CHECK(plane.status == PHX_MC_UNRESOLVED && plane.reason == PHX_MC_FIDELITY_INTERVAL_REFUSAL);
  // A corner past the pole latitude has no colatitude.
  const double past[6] = {0, std::nextafter(top, 4.0), 0, 1, 1, 1};
  double past_points[9] = {0, 0, 3, 2.5, -0.25, 1, 0.5, 1.75, 1};
  const auto beyond = fidelity_pole_triangle(geometry, 0, past_points, past, 1, 0, 1.0, 8,
                                             workspace);
  PHX_CHECK(beyond.status == PHX_MC_UNRESOLVED);
}

// Seam lens (MAC bc1b665 (i)) of a pole-region boundary edge on the unit
// sphere: the lens between the uv-straight edge's pole-chart image and the
// pole-chart-straight edge, sampled densely in (t, s), against the cap: cap
// >= sup |S(H(t, s)) - L(t)| and >= the exact distance of every sampled lens
// point to the mesh segment. Fault caught: a seam charged from the pole side
// only (the pole row's straight-edge deviation B, the "same curve" a broken
// association would assume) misses lens points (B < their distance).
void seam_lens_charge() {
  Model m;
  const int32_t sphere = m.add_template("sphere", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1});
  m.surface(sphere);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  m.bind_bounds(workspace);
  EvaluationWorkspace point_workspace;
  struct Edge { double ua, ra, ub, rb; };
  for (const Edge edge : {Edge{0.0, 0.5, 1.2, 0.5}, Edge{0.0, 0.3, 1.5, 0.6}, Edge{0.25, 0.6, 1.15, 0.3}})
    for (int sigma : {1, -1}) {
      const double uv_a[2] = {edge.ua, sigma * (M_PI / 2 - edge.ra)};
      const double uv_b[2] = {edge.ub, sigma * (M_PI / 2 - edge.rb)};
      double image[4], c_a[2], c_b[2], x_a[3], x_b[3];
      for (int end = 0; end < 2; ++end) {
        const double* point = end == 0 ? uv_a : uv_b;
        double* c = end == 0 ? c_a : c_b;
        PHX_CHECK(fidelity_pole_image(geometry, 0, workspace, sigma, point, image) == PHX_MC_OK);
        c[0] = image[0] + 0.5 * (image[1] - image[0]);
        c[1] = image[2] + 0.5 * (image[3] - image[2]);
        PHX_CHECK(geometry.jets(sphere, point, 0, end == 0 ? x_a : x_b, point_workspace));
      }
      double out[3];
      PHX_CHECK(fidelity_seam_lens(geometry, 0, workspace, sigma, uv_a, uv_b, c_a, c_b, x_a, x_b,
                                   out) == PHX_MC_OK);
      double lens = 0.0, distance = 0.0;
      const int n = 200;
      double d[3], dd = 0.0;
      for (int axis = 0; axis < 3; ++axis) d[axis] = x_b[axis] - x_a[axis], dd += d[axis] * d[axis];
      for (int i = 0; i <= n; ++i) {
        const double t = static_cast<double>(i) / n;
        const double u = edge.ua + t * (edge.ub - edge.ua), rho = edge.ra + t * (edge.rb - edge.ra);
        const double px = rho * std::cos(u), py = rho * std::sin(u);
        const double fx = c_a[0] + t * (c_b[0] - c_a[0]), fy = c_a[1] + t * (c_b[1] - c_a[1]);
        for (int k = 0; k <= n; ++k) {
          const double s = static_cast<double>(k) / n;
          const double x = (1 - s) * px + s * fx, y = (1 - s) * py + s * fy;
          const double q = std::hypot(x, y), g = q == 0.0 ? 1.0 : std::sin(q) / q;
          const double p[3] = {g * x, g * y, sigma * std::cos(q)};
          double along = 0.0, to_line = 0.0, to_segment = 0.0;
          for (int axis = 0; axis < 3; ++axis) along += (p[axis] - x_a[axis]) * d[axis];
          along = std::clamp(along / dd, 0.0, 1.0);
          for (int axis = 0; axis < 3; ++axis) {
            const double l = x_a[axis] + t * d[axis], e = x_a[axis] + along * d[axis];
            to_line += (p[axis] - l) * (p[axis] - l);
            to_segment += (p[axis] - e) * (p[axis] - e);
          }
          lens = std::max(lens, std::sqrt(to_line));
          distance = std::max(distance, std::sqrt(to_segment));
        }
      }
      std::printf("P seam lens (%.2f %.2f -> %.2f %.2f, sigma %d): cap %.6e uv edge %.6e pole edge "
                  "%.6e sampled %.6e distance %.6e\n",
                  edge.ua, edge.ra, edge.ub, edge.rb, sigma, out[0], out[1], out[2], lens, distance);
      PHX_CHECK(out[0] >= lens && out[0] >= distance && out[1] <= out[0] && out[2] < distance);
    }
}

// A chart grid of the unit sphere's upper hemisphere over u in [0, span],
// v in [0, pi/2] (pole line collapsed, slivers there), with chords on the
// equator, the two meridians and the pole line.
struct SphereGrid {
  Model model;
  Mesh mesh;
  int32_t sphere = -1;
};

SphereGrid sphere_grid(double span, int columns, int rows, double low = 0.0) {
  SphereGrid g;
  Model& m = g.model;
  const double top = 0x1.921fb54442d18p+0;
  g.sphere = m.add_template("sphere", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1});
  m.surface(g.sphere);
  const int32_t space = m.add_template("line3", {0, 0, 0, 1, 0, 0});
  m.corner(0, 0, 1);
  for (int k = 0; k < 3; ++k) m.curve(space, 0, 0);
  const double ends[4][2] = {{0, low}, {span, low}, {span, top}, {0, top}};
  for (int k = 0; k < 4; ++k) {
    const double* p = ends[k];
    const double* q = ends[(k + 1) % 4];
    const int32_t pcurve = m.add_template("line2", {p[0], p[1], q[0] - p[0], q[1] - p[1]});
    m.use(k == 2 ? 1 : 0, k == 2 ? 0 : (k < 2 ? k : 2), 0.0, 1.0, pcurve);
  }
  m.close_loop(0);
  const phx_mc_cad_model view = m.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  EvaluationWorkspace point_workspace;
  const bool full = span >= 2.0 * 0x1.921fb54442d18p+1;
  const auto u = [&](int i) { return i == columns ? span : span * i / columns; };
  const auto v = [&](int j) { return j == rows ? top : j == 0 ? low : low + (top - low) * j / rows; };
  const auto id = [&](int i, int j) -> int64_t {
    if (j == rows) return 0;
    if (full && i == columns) i = 0;
    return 1 + static_cast<int64_t>(j) * (columns + 1) + i;
  };
  Mesh& mesh = g.mesh;
  mesh.points.assign(3 * static_cast<std::size_t>(1 + rows * (columns + 1)), 0.0);
  for (int j = 0; j <= rows; ++j)
    for (int i = 0; i <= columns; ++i) {
      const double point[2] = {u(i), v(j)};
      double xyz[3];
      PHX_CHECK(geometry.jets(g.sphere, point, 0, xyz, point_workspace));
      std::copy(xyz, xyz + 3, mesh.points.begin() + 3 * id(i, j));
    }
  // Boundary side of a grid edge: 0 equator, 1 meridian at span, 2 pole, 3 meridian at 0.
  const auto side = [&](int i0, int j0, int i1, int j1) {
    if (j0 == 0 && j1 == 0) return 0;
    if (i0 == columns && i1 == columns) return 1;
    if (j0 == rows && j1 == rows) return 2;
    if (i0 == 0 && i1 == 0) return 3;
    return -1;
  };
  const auto parameter = [&](int s, int i, int j) {
    return s == 0 ? static_cast<double>(i) / columns
         : s == 1 ? static_cast<double>(j) / rows
         : s == 2 ? static_cast<double>(columns - i) / columns
                  : static_cast<double>(rows - j) / rows;
  };
  for (int j = 0; j < rows; ++j)
    for (int i = 0; i < columns; ++i) {
      const int cell[4][2] = {{i, j}, {i + 1, j}, {i + 1, j + 1}, {i, j + 1}};
      for (const auto& local : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
        int64_t ids[3];
        for (int k = 0; k < 3; ++k) {
          const int* c = cell[local[static_cast<std::size_t>(k)]];
          ids[k] = id(c[0], c[1]);
          mesh.triangles.push_back(ids[k]);
          mesh.uv.insert(mesh.uv.end(), {u(c[0]), v(c[1])});
        }
        for (int k = 0; k < 3; ++k) {
          const int* a = cell[local[static_cast<std::size_t>(k)]];
          const int* b = cell[local[static_cast<std::size_t>((k + 1) % 3)]];
          const int s = side(a[0], a[1], b[0], b[1]);
          if (s < 0) {
            mesh.edge_chord.push_back(-1);
            continue;
          }
          mesh.edge_chord.push_back(static_cast<int64_t>(mesh.chord_use.size()));
          mesh.chord_use.push_back(s);
          mesh.chord_loop.push_back(0);
          mesh.chord_uv.insert(mesh.chord_uv.end(), {u(a[0]), v(a[1]), u(b[0]), v(b[1])});
          mesh.chord_range.insert(mesh.chord_range.end(),
                                  {parameter(s, a[0], a[1]), parameter(s, b[0], b[1])});
        }
        mesh.face_rows.push_back(0);
        const bool sliver = ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0];
        mesh.chart_to_mesh.push_back(
            sliver ? -1 : static_cast<int64_t>(mesh.mesh_triangles.size() / 3));
        if (!sliver) mesh.mesh_triangles.insert(mesh.mesh_triangles.end(), ids, ids + 3);
      }
    }
  mesh.use_patch.assign(4, 0);
  mesh.use_range = m.use_ranges;
  mesh.loop_count = 1;
  return g;
}

// MAC 69172b0 P2: a sphere face spanning less than a full turn of u (octant,
// half turn, 6.2 rad) keeps the uv chart and certifies as before. Fault
// caught: an angular charge r rho (2 pi - span) refusing every region row.
// P3: on a full hemisphere, every boundary edge of the pole region carries
// the seam lens cap, and both adjacent rows' bounds are at least it. Fault
// caught: either charge call dropped (or overwritten by a later region).
void full_turn_charges(int columns, int rows_count, double tolerance, double low = 0.0);

void pole_region_turns_and_charges() {
  for (const double span : {0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1, 6.2}) {
    SphereGrid g = sphere_grid(span, 8, 8);
    const Verdict verdict = verify(g.model, g.mesh, 0.1, 8);
    double worst = 0.0;
    for (double x : verdict.upper) worst = std::max(worst, x);
    std::printf("P partial turn %.4f: status %d max %.6e\n", span, verdict.status, worst);
    PHX_CHECK(verdict.status == PHX_MC_OK && verdict.findings.empty());
  }
  full_turn_charges(8, 8, 0.1);
  // Coarse in u: the uv edge sag A exceeds the pole row's own bound, so the
  // row charge decides (mutation check: dropping it fails here).
  full_turn_charges(3, 8, 1.0);
  full_turn_charges(4, 3, 1.0);
  // Interface ring at latitude 0.25 (rho 1.32, pole-chart sag factor 1.52 vs
  // the uv chart's 1.03): the neighbour charge decides.
  full_turn_charges(4, 3, 1.0, -0.409);
}

// Every boundary edge of the pole region of a full-turn hemisphere grid:
// both adjacent rows' bounds are at least the edge's seam lens cap.
void full_turn_charges(int columns, int rows_count, double tolerance, double low) {
  SphereGrid g = sphere_grid(2.0 * 0x1.921fb54442d18p+1, columns, rows_count, low);
  const Verdict verdict = verify(g.model, g.mesh, tolerance, 8);
  PHX_CHECK(verdict.status == PHX_MC_OK && verdict.findings.empty());
  const phx_mc_cad_model view = g.model.view();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(view) == PHX_MC_OK);
  IntervalEvaluationWorkspace workspace;
  g.model.bind_bounds(workspace);
  const Mesh& mesh = g.mesh;
  const auto rows = mesh.face_rows.size();
  const double top = 0x1.921fb54442d18p+0, cap_latitude = 0x1.3f004dc9db074p-1;
  std::vector<int> region(rows, 0);
  for (std::size_t r = 0; r < rows; ++r)
    for (int k = 0; k < 3; ++k) {
      const double lat = mesh.uv[6 * r + 2 * static_cast<std::size_t>(k) + 1];
      region[r] |= lat == top || lat >= cap_latitude;
    }
  const auto same = [&](std::size_t r, int a, std::size_t s, int b) {
    return mesh.uv[6 * r + 2 * static_cast<std::size_t>(a)] == mesh.uv[6 * s + 2 * static_cast<std::size_t>(b)] &&
           mesh.uv[6 * r + 2 * static_cast<std::size_t>(a) + 1] ==
               mesh.uv[6 * s + 2 * static_cast<std::size_t>(b) + 1];
  };
  int interface_edges = 0;
  double row_ratio = std::numeric_limits<double>::infinity(), neighbour_ratio = row_ratio;
  double largest_cap = 0.0;
  std::size_t cap_row = 0, cap_neighbour = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    if (!region[r]) continue;
    for (int k = 0; k < 3; ++k) {
      const int a = k, b = (k + 1) % 3;
      std::size_t neighbour = rows;
      bool inside = false;
      for (std::size_t s = 0; s < rows; ++s) {
        if (s == r) continue;
        for (int l = 0; l < 3; ++l)
          if (same(r, a, s, (l + 1) % 3) && same(r, b, s, l)) {
            if (region[s]) inside = true;
            else neighbour = s;
          }
      }
      if (inside || neighbour == rows) continue;
      double uv_a[2], uv_b[2], c[2][2], out[3];
      for (int axis = 0; axis < 2; ++axis) {
        uv_a[axis] = mesh.uv[6 * r + 2 * static_cast<std::size_t>(a) + static_cast<std::size_t>(axis)];
        uv_b[axis] = mesh.uv[6 * r + 2 * static_cast<std::size_t>(b) + static_cast<std::size_t>(axis)];
      }
      for (int end = 0; end < 2; ++end) {
        double image[4];
        PHX_CHECK(fidelity_pole_image(geometry, 0, workspace, 1, end == 0 ? uv_a : uv_b, image) ==
                  PHX_MC_OK);
        c[end][0] = image[0] + 0.5 * (image[1] - image[0]);
        c[end][1] = image[2] + 0.5 * (image[3] - image[2]);
      }
      const double* x_a = mesh.points.data() + 3 * mesh.triangles[3 * r + static_cast<std::size_t>(a)];
      const double* x_b = mesh.points.data() + 3 * mesh.triangles[3 * r + static_cast<std::size_t>(b)];
      PHX_CHECK(fidelity_seam_lens(geometry, 0, workspace, 1, uv_a, uv_b, c[0], c[1], x_a, x_b, out) ==
                PHX_MC_OK);
      PHX_CHECK(verdict.upper[r] >= out[0] && verdict.upper[neighbour] >= out[0]);
      row_ratio = std::min(row_ratio, verdict.upper[r] / out[0]);
      neighbour_ratio = std::min(neighbour_ratio, verdict.upper[neighbour] / out[0]);
      ++interface_edges;
      if (out[0] > largest_cap) largest_cap = out[0], cap_row = r, cap_neighbour = neighbour;
    }
  }
  std::printf("P full turn %dx%d: status %d interface edges %d min upper/cap row %.6f "
              "neighbour %.6f\n",
              columns, rows_count, verdict.status, interface_edges, row_ratio, neighbour_ratio);
  PHX_CHECK(interface_edges > 0);
  // Just below the largest cap both adjacent rows must refuse: the cap is
  // charged to each, whatever their own bounds (the uv neighbour's own bound
  // can be well below a cap set by the pole-chart edge).
  const Verdict tight = verify(g.model, g.mesh, 0.999 * largest_cap, 12);
  std::printf("P full turn %dx%d at 0.999 cap: row status %d neighbour status %d\n", columns,
              rows_count, tight.item_status[cap_row], tight.item_status[cap_neighbour]);
  PHX_CHECK(tight.item_status[cap_row] != PHX_MC_OK &&
            tight.item_status[cap_neighbour] != PHX_MC_OK);
}
}  // namespace

int main() {
  using namespace phx::mc::cad;
  intersection_seam_leaf();
  intersection_seam_lens();
  clamped_domain_end();
  lens_charged_to_neighbours();
  use_and_loop_preconditions();
  sliver_edge_in_mesh();
  lens_requires_lipschitz_surface();
  intersection_seam_lens_demo();
  sliver_segment_in_mesh_demo();
  empty_mesh_demo();
  carrier_inside_domain_lens();
  carrier_inside_domain_leaf();
  carrier_inside_domain_trim();
  hole_orientation_demo();
  surface_without_loop_demo();
  lens_index_overflow();
  pole_chart_rows();
  seam_lens_charge();
  pole_region_turns_and_charges();
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
