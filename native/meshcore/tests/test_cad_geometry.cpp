// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// CAD geometry nodes against closed forms written here, independently of the
// generated programs and native evaluators: values within 4 ulp of the
// coordinate scale, derivatives against analytic derivatives and central
// finite differences, splines against Bernstein / rational identities, an
// implicit intersection chart against the closed-form cylinder pair, and
// the differential quantities of the C entry against known curvatures.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

#include "cad_geometry.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

using phx::mc::cad::CadGeometry;
using phx::mc::cad::EvaluationWorkspace;
using phx::mc::cad::template_index;

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kPi = 3.141592653589793;

using Vector = std::array<double, 3>;

// Geometry graph and entity tables of one test model.
struct Builder {
  std::vector<int32_t> kinds;
  std::vector<int64_t> integer_offsets{0}, integers;
  std::vector<int64_t> real_offsets{0};
  std::vector<double> reals;
  std::vector<int32_t> surfaces, curves, pcurves;

  int32_t add(int32_t kind, std::vector<int64_t> ints, std::vector<double> values) {
    kinds.push_back(kind);
    integers.insert(integers.end(), ints.begin(), ints.end());
    reals.insert(reals.end(), values.begin(), values.end());
    integer_offsets.push_back(static_cast<int64_t>(integers.size()));
    real_offsets.push_back(static_cast<int64_t>(reals.size()));
    return static_cast<int32_t>(kinds.size() - 1);
  }
  int32_t add_template(const char* name, std::vector<double> coefficients) {
    const int32_t id = template_index(name);
    PHX_CHECK(id >= 0);
    return add(PHX_MC_CAD_NODE_TEMPLATE, {id}, std::move(coefficients));
  }

  // Tables sized for the registered entities; one corner, one use per p-curve.
  std::vector<int8_t> orientations, use_kinds, senses;
  std::vector<double> periods, domains, ranges, use_ranges, corner{0.0, 0.0, 0.0};
  std::vector<int32_t> corners, use_entities, loop_surfaces;
  std::vector<int64_t> loop_offsets{0}, region_offsets{0};

  phx_mc_cad_model model() {
    const auto f = surfaces.size(), e = curves.size(), u = pcurves.size();
    orientations.assign(f, 1);
    periods.assign(2 * f, 0.0);
    domains.assign(4 * f, 0.0);
    corners.assign(2 * e, 0);
    ranges.assign(2 * e, 0.0);
    use_kinds.assign(u, 1);
    use_entities.assign(u, 0);
    senses.assign(u, 1);
    use_ranges.assign(2 * u, 0.0);
    loop_surfaces.assign(u ? 1 : 0, 0);
    loop_offsets = {0};
    if (u) loop_offsets.push_back(static_cast<int64_t>(u));
    if (u && f == 0) {
      PHX_CHECK(false);  // a loop needs a surface
    }
    phx_mc_cad_model m{};
    m.node_count = static_cast<int64_t>(kinds.size());
    m.node_kinds = kinds.data();
    m.node_integer_offsets = integer_offsets.data();
    m.node_integers = integers.data();
    m.node_real_offsets = real_offsets.data();
    m.node_reals = reals.data();
    m.corner_count = 1;
    m.corner_points = corner.data();
    m.surface_count = static_cast<int64_t>(f);
    m.surface_nodes = surfaces.data();
    m.surface_orientations = orientations.data();
    m.surface_periods = periods.data();
    m.surface_parameter_domains = domains.data();
    m.curve_count = static_cast<int64_t>(e);
    m.curve_nodes = curves.data();
    m.curve_corners = corners.data();
    m.curve_ranges = ranges.data();
    m.loop_count = u ? 1 : 0;
    m.loop_use_offsets = loop_offsets.data();
    m.loop_surfaces = loop_surfaces.data();
    m.use_count = static_cast<int64_t>(u);
    m.use_kinds = use_kinds.data();
    m.use_entities = use_entities.data();
    m.use_senses = senses.data();
    m.use_ranges = use_ranges.data();
    m.use_pcurve_nodes = pcurves.data();
    m.region_count = 0;
    m.region_surface_offsets = region_offsets.data();
    return m;
  }
};

Vector add(const Vector& a, const Vector& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vector scaled(double s, const Vector& a) { return {s * a[0], s * a[1], s * a[2]}; }
Vector cross(const Vector& a, const Vector& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
std::vector<double> concat(std::initializer_list<std::vector<double>> parts) {
  std::vector<double> out;
  for (const auto& part : parts) out.insert(out.end(), part.begin(), part.end());
  return out;
}
std::vector<double> v3(const Vector& a) { return {a[0], a[1], a[2]}; }

// A rotated orthonormal frame (e1, e2, a = e1 x e2) and a nonzero origin.
const Vector kE1{0.36, 0.48, 0.8};
const Vector kE2{0.8, -0.6, 0.0};
const Vector kAxis = cross(kE1, kE2);
const Vector kOrigin{0.5, -1.25, 2.0};

using SurfaceMap = std::function<Vector(double, double)>;
using CurveMap = std::function<std::vector<double>(double)>;

double scale_of(const std::vector<double>& values) {
  double scale = 1.0;
  for (double v : values) scale = std::max(scale, std::fabs(v));
  return scale;
}

// Central differences with one Richardson step: truncation error O(h^4).
template <class F>
double first_difference(const F& f, double h) {
  const auto central = [&](double step) { return (f(step) - f(-step)) / (2 * step); };
  return (4 * central(0.5 * h) - central(h)) / 3;
}
template <class F>
double second_difference(const F& f, double h) {
  const auto central = [&](double step) { return (f(step) - 2 * f(0.0) + f(-step)) / (step * step); };
  return (4 * central(0.5 * h) - central(h)) / 3;
}

constexpr double kStep = 1e-3;

// Values within 4 ulp of the scale; first and second derivatives against
// extrapolated central differences of the closed form (relative 1e-7).
void check_surface(const CadGeometry& geometry, int32_t node, const SurfaceMap& closed,
                   double u0, double u1, double v0, double v1, int samples = 200) {
  std::mt19937_64 random(1234);
  std::uniform_real_distribution<double> uu(u0, u1), vv(v0, v1);
  EvaluationWorkspace workspace;
  for (int k = 0; k < samples; ++k) {
    const double point[2] = {uu(random), vv(random)};
    double jet[18];
    PHX_CHECK(geometry.jets(node, point, 2, jet, workspace));
    const Vector x = closed(point[0], point[1]);
    const double scale = scale_of({x[0], x[1], x[2]});
    for (int i = 0; i < 3; ++i) PHX_CHECK_NEAR(jet[i], x[i], 4.0 * kEpsilon * scale);
    for (int i = 0; i < 3; ++i) {
      const auto f = [&](double du, double dv) { return closed(point[0] + du, point[1] + dv)[i]; };
      const auto along_u = [&](double s) { return f(s, 0.0); };
      const auto along_v = [&](double s) { return f(0.0, s); };
      const auto mixed = [&](double s) {
        return (f(s, s) - f(s, -s) - f(-s, s) + f(-s, -s)) / (4 * s * s);
      };
      PHX_CHECK_NEAR(jet[3 + i], first_difference(along_u, kStep), 1e-7 * scale);
      PHX_CHECK_NEAR(jet[6 + i], first_difference(along_v, kStep), 1e-7 * scale);
      PHX_CHECK_NEAR(jet[9 + i], second_difference(along_u, kStep), 1e-7 * scale);
      PHX_CHECK_NEAR(jet[12 + i], (4 * mixed(0.5 * kStep) - mixed(kStep)) / 3, 1e-7 * scale);
      PHX_CHECK_NEAR(jet[15 + i], second_difference(along_v, kStep), 1e-7 * scale);
    }
  }
}

void check_curve(const CadGeometry& geometry, int32_t node, const CurveMap& closed, double t0,
                 double t1, double value_slack = 4.0, int samples = 200) {
  std::mt19937_64 random(99);
  std::uniform_real_distribution<double> tt(t0, t1);
  EvaluationWorkspace workspace;
  const int d = geometry.node(node).outputs;
  for (int k = 0; k < samples; ++k) {
    const double t = tt(random);
    double jet[9];
    PHX_CHECK(geometry.jets(node, &t, 2, jet, workspace));
    const std::vector<double> x = closed(t);
    const double scale = scale_of(x);
    for (int i = 0; i < d; ++i) PHX_CHECK_NEAR(jet[i], x[i], value_slack * kEpsilon * scale);
    for (int i = 0; i < d; ++i) {
      const auto f = [&](double s) { return closed(t + s)[static_cast<std::size_t>(i)]; };
      PHX_CHECK_NEAR(jet[d + i], first_difference(f, kStep), 1e-7 * scale);
      PHX_CHECK_NEAR(jet[2 * d + i], second_difference(f, kStep), 1e-7 * scale);
    }
  }
}

Vector radial(double angle) { return add(scaled(std::cos(angle), kE1), scaled(std::sin(angle), kE2)); }

void quadric_templates() {
  Builder b;
  const double radius = 1.5, minor = 0.4, r0 = 0.7, angle = 0.3;
  const int32_t plane = b.add_template("plane", concat({v3(kOrigin), v3(kE1), v3(kE2)}));
  const int32_t cylinder =
      b.add_template("cylinder", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t cone = b.add_template(
      "cone", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {r0}, {angle}}));
  const int32_t sphere =
      b.add_template("sphere", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t torus = b.add_template(
      "torus", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}, {minor}}));
  b.surfaces = {plane, cylinder, cone, sphere, torus};
  phx_mc_cad_model model = b.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  check_surface(geometry, plane, [](double u, double v) {
    return add(kOrigin, add(scaled(u, kE1), scaled(v, kE2)));
  }, -2, 2, -2, 2);
  check_surface(geometry, cylinder, [&](double u, double v) {
    return add(kOrigin, add(scaled(radius, radial(u)), scaled(v, kAxis)));
  }, -7, 7, -2, 2);
  check_surface(geometry, cone, [&](double u, double v) {
    return add(kOrigin, add(scaled(v, kAxis), scaled(r0 + v * std::tan(angle), radial(u))));
  }, -7, 7, -2, 2);
  check_surface(geometry, sphere, [&](double u, double v) {
    return add(kOrigin, scaled(radius, add(scaled(std::cos(v), radial(u)),
                                           scaled(std::sin(v), kAxis))));
  }, -7, 7, -1.5, 1.5);
  check_surface(geometry, torus, [&](double u, double v) {
    return add(kOrigin, add(scaled(radius + minor * std::cos(v), radial(u)),
                            scaled(minor * std::sin(v), kAxis)));
  }, -7, 7, -7, 7);
}

// The closed-form evaluators against the generic route (template programs and
// Taylor composition) on the same graph: every normalized Taylor coefficient
// through order 4 within 4 ulp of the expansion's scale. Covers the quadric
// templates, revolutions of both line kinds and placements of them.
void closed_forms_match_programs() {
  using phx::mc::cad::ClosedForm;
  using phx::mc::cad::Evaluators;
  using phx::mc::cad::kMaxOrder;
  using phx::mc::cad::Taylor;
  using phx::mc::cad::term;
  Builder b;
  const double radius = 1.5, minor = 0.4, r0 = 0.7, angle = 0.3;
  const int32_t plane = b.add_template("plane", concat({v3(kOrigin), v3(kE1), v3(kE2)}));
  const int32_t cylinder =
      b.add_template("cylinder", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t cone = b.add_template(
      "cone", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {r0}, {angle}}));
  const int32_t sphere =
      b.add_template("sphere", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t torus = b.add_template(
      "torus", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}, {minor}}));
  // A slanted line (a cone of revolution) and an exact line segment.
  const int32_t line = b.add_template(
      "line3", concat({v3(add(kOrigin, scaled(radius, kE1))), v3(add(kAxis, scaled(0.3, kE2)))}));
  const int32_t exact = b.add_template(
      "exact_line3", concat({v3(add(kOrigin, scaled(radius, kE1))),
                             v3(add(kOrigin, add(scaled(0.5, kE1), scaled(2.0, kAxis))))}));
  const int32_t revolution =
      b.add(PHX_MC_CAD_NODE_REVOLUTION, {line}, concat({v3(kOrigin), v3(kAxis)}));
  const int32_t exact_revolution =
      b.add(PHX_MC_CAD_NODE_REVOLUTION, {exact}, concat({v3(kOrigin), v3(kAxis)}));
  const std::vector<double> placement{0, -1, 0, 1, 0, 0, 0, 0, 1, 1.0, 2.0, 3.0};
  const int32_t placed_revolution = b.add(PHX_MC_CAD_NODE_AFFINE, {revolution, 3}, placement);
  const int32_t placed_torus = b.add(PHX_MC_CAD_NODE_AFFINE, {torus, 3}, placement);
  const int32_t twice = b.add(PHX_MC_CAD_NODE_AFFINE, {placed_torus, 3}, placement);
  const std::vector<int32_t> nodes{plane,           cylinder,          cone,
                                   sphere,          torus,             revolution,
                                   exact_revolution, placed_revolution, placed_torus,
                                   twice};
  b.surfaces = nodes;
  phx_mc_cad_model model = b.model();
  CadGeometry closed, programs;
  PHX_CHECK(closed.bind(model) == PHX_MC_OK);
  PHX_CHECK(programs.bind(model, Evaluators::kPrograms) == PHX_MC_OK);
  EvaluationWorkspace workspace;
  std::mt19937_64 random(77);
  std::uniform_real_distribution<double> uu(-7.0, 7.0), vv(-1.4, 1.4);
  for (const int32_t node : nodes) {
    PHX_CHECK(closed.node(node).closed != ClosedForm::kNone);
    PHX_CHECK(programs.node(node).closed == ClosedForm::kNone);
    for (int sample = 0; sample < 500; ++sample) {
      const double point[2] = {uu(random), vv(random)};
      Taylor fast[3], reference[3];
      PHX_CHECK(closed.expand(node, point, kMaxOrder, fast, workspace));
      PHX_CHECK(programs.expand(node, point, kMaxOrder, reference, workspace));
      double scale = 1.0;
      for (int i = 0; i < 3; ++i)
        for (int n = 0; n <= kMaxOrder; ++n)
          for (int j = 0; j <= n; ++j)
            scale = std::max(scale, std::fabs(reference[i].c[term(n - j, j)]));
      for (int i = 0; i < 3; ++i)
        for (int n = 0; n <= kMaxOrder; ++n)
          for (int j = 0; j <= n; ++j)
            PHX_CHECK_NEAR(fast[i].c[term(n - j, j)], reference[i].c[term(n - j, j)],
                           4.0 * kEpsilon * scale);
    }
  }
}

void curve_templates() {
  Builder b;
  const int32_t line = b.add_template("line3", concat({v3(kOrigin), v3(kE1)}));
  const int32_t exact_line = b.add_template("exact_line2", {0.5, -1.0, 2.0, 3.0});
  const int32_t circle = b.add_template("circle3", concat({v3(kOrigin), v3(kE1), v3(kE2), {1.25}}));
  const int32_t ellipse =
      b.add_template("ellipse2", {0.5, -0.25, 0.6, 0.8, -0.8, 0.6, 2.0, 0.75});
  const int32_t parabola = b.add_template("parabola3", concat({v3(kOrigin), v3(kE1), v3(kE2), {0.5}}));
  const int32_t hyperbola =
      b.add_template("hyperbola2", {0.5, -0.25, 0.6, 0.8, -0.8, 0.6, 2.0, 0.75});
  const int32_t root_line = b.add_template("conic_root_line", {0.9, 0.25, 1.5});
  const int32_t rooted = b.add_template("torus_rooted_circle_reverse", {0.75, 0.9});
  const int32_t section = b.add_template(
      "plane_cylinder_section", {0.1, 0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, 0.25});
  phx_mc_cad_model model = b.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const auto as_vector = [](const Vector& x) { return std::vector<double>{x[0], x[1], x[2]}; };
  check_curve(geometry, line, [&](double t) { return as_vector(add(kOrigin, scaled(t, kE1))); }, -3, 3);
  check_curve(geometry, exact_line, [](double t) {
    return std::vector<double>{(1 - t) * 0.5 + t * 2.0, (1 - t) * -1.0 + t * 3.0};
  }, -1, 2);
  check_curve(geometry, circle, [&](double t) {
    return as_vector(add(kOrigin, scaled(1.25, radial(t))));
  }, -7, 7);
  check_curve(geometry, ellipse, [](double t) {
    return std::vector<double>{0.5 + 2.0 * std::cos(t) * 0.6 + 0.75 * std::sin(t) * -0.8,
                               -0.25 + 2.0 * std::cos(t) * 0.8 + 0.75 * std::sin(t) * 0.6};
  }, -7, 7);
  check_curve(geometry, parabola, [&](double t) {
    return as_vector(add(kOrigin, add(scaled(t * t / 2.0, kE1), scaled(t, kE2))));
  }, -3, 3);
  check_curve(geometry, hyperbola, [](double t) {
    return std::vector<double>{0.5 + 2.0 * std::cosh(t) * 0.6 + 0.75 * std::sinh(t) * -0.8,
                               -0.25 + 2.0 * std::cosh(t) * 0.8 + 0.75 * std::sinh(t) * 0.6};
  }, -2, 2, 8.0);
  check_curve(geometry, root_line, [](double t) { return std::vector<double>{0.9, 0.25 + 1.5 * t}; }, -2, 2);
  check_curve(geometry, rooted, [](double t) { return std::vector<double>{-t + 0.75, 0.9}; }, -2, 2);
  check_curve(geometry, section, [](double t) {
    return std::vector<double>{0.1 + t * 0.3 + std::cos(t) * 0.5 + std::sin(t) * -0.7 + 0.25,
                               0.2 + t * -0.4 + std::cos(t) * 0.6 + std::sin(t) * 0.8};
  }, -3, 3);
}

// Swept, mapped, offset and placed nodes reproduce the quadric they describe.
void composite_nodes() {
  Builder b;
  const double radius = 1.5;
  const int32_t line = b.add_template("line3", concat({v3(add(kOrigin, scaled(radius, kE1))), v3(kAxis)}));
  const int32_t revolution =
      b.add(PHX_MC_CAD_NODE_REVOLUTION, {line}, concat({v3(kOrigin), v3(kAxis)}));
  const int32_t circle = b.add_template("circle3", concat({v3(kOrigin), v3(kE1), v3(kE2), {radius}}));
  const int32_t extrusion = b.add(PHX_MC_CAD_NODE_EXTRUSION, {circle}, v3(kAxis));
  const int32_t top =
      b.add_template("circle3", concat({v3(add(kOrigin, kAxis)), v3(kE1), v3(kE2), {radius}}));
  const int32_t ruled = b.add(PHX_MC_CAD_NODE_RULED, {circle, top}, {});
  const int32_t sphere =
      b.add_template("sphere", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t offset = b.add(PHX_MC_CAD_NODE_OFFSET_SURFACE, {sphere}, {0.25});
  // Fourth-order template jets: an offset of an offset.
  const int32_t nested = b.add(PHX_MC_CAD_NODE_OFFSET_SURFACE, {offset}, {0.5});
  const int32_t plane = b.add_template("plane", concat({v3(kOrigin), v3(kE1), v3(kE2)}));
  const int32_t circle2 = b.add_template("circle2", {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, radius});
  const int32_t mapped = b.add(PHX_MC_CAD_NODE_SURFACE_MAPPED, {plane, circle2}, {});
  const int32_t iso = b.add(PHX_MC_CAD_NODE_ISOPARAMETRIC, {sphere, 1}, {0.4});
  const int32_t offset_curve = b.add(PHX_MC_CAD_NODE_OFFSET_CURVE, {circle2, 0}, {0.5});
  // Rotation by a quarter turn about z and a translation.
  const int32_t placed = b.add(PHX_MC_CAD_NODE_AFFINE, {revolution, 3},
                               {0, -1, 0, 1, 0, 0, 0, 0, 1, 1.0, 2.0, 3.0});
  const int32_t reversed = b.add(PHX_MC_CAD_NODE_REPARAMETERIZED, {circle, 1}, {0.5, 2.5});
  const int32_t chain = b.add(PHX_MC_CAD_NODE_AFFINE, {offset_curve, 2}, {1, 0, 0, 1, 0.0, 6.2831853071795862});
  b.surfaces = {revolution, extrusion, ruled, offset, placed, nested};
  phx_mc_cad_model model = b.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const auto cylinder = [&](double u, double v) {
    return add(kOrigin, add(scaled(radius, radial(u)), scaled(v, kAxis)));
  };
  // The revolution turns e1 about the axis: angle u measured from e1 toward
  // axis x e1 = e2.
  check_surface(geometry, revolution, cylinder, -7, 7, -2, 2);
  check_surface(geometry, extrusion, cylinder, -7, 7, -2, 2);
  check_surface(geometry, ruled, cylinder, -7, 7, -2, 2);
  check_surface(geometry, offset, [&](double u, double v) {
    return add(kOrigin, scaled(radius + 0.25, add(scaled(std::cos(v), radial(u)),
                                                  scaled(std::sin(v), kAxis))));
  }, -7, 7, -1.4, 1.4);
  check_surface(geometry, nested, [&](double u, double v) {
    return add(kOrigin, scaled(radius + 0.75, add(scaled(std::cos(v), radial(u)),
                                                  scaled(std::sin(v), kAxis))));
  }, -7, 7, -1.4, 1.4);
  check_surface(geometry, placed, [&](double u, double v) {
    const Vector x = cylinder(u, v);
    return Vector{-x[1] + 1.0, x[0] + 2.0, x[2] + 3.0};
  }, -7, 7, -2, 2);
  const auto as_vector = [](const Vector& x) { return std::vector<double>{x[0], x[1], x[2]}; };
  check_curve(geometry, mapped, [&](double t) {
    return as_vector(add(kOrigin, add(scaled(radius * std::cos(t), kE1),
                                      scaled(radius * std::sin(t), kE2))));
  }, -7, 7);
  check_curve(geometry, iso, [&](double t) {
    return as_vector(add(kOrigin, scaled(radius, add(scaled(std::cos(0.4), radial(t)),
                                                     scaled(std::sin(0.4), kAxis)))));
  }, -7, 7);
  check_curve(geometry, offset_curve, [&](double t) {
    return std::vector<double>{(radius + 0.5) * std::cos(t), (radius + 0.5) * std::sin(t)};
  }, -7, 7, 8.0);
  check_curve(geometry, chain, [&](double t) {
    return std::vector<double>{(radius + 0.5) * std::cos(t),
                               (radius + 0.5) * std::sin(t) + 6.2831853071795862};
  }, -7, 7, 8.0);
  check_curve(geometry, reversed, [&](double t) {
    return as_vector(add(kOrigin, scaled(radius, radial(0.5 + (1.0 - t) * 2.0))));
  }, -1, 2);
}

double bernstein(int degree, int index, double t) {
  double binomial = 1.0;
  for (int k = 1; k <= index; ++k) binomial = binomial * (degree - index + k) / k;
  return binomial * std::pow(t, index) * std::pow(1.0 - t, degree - index);
}

void spline_nodes() {
  Builder b;
  // Rational quadratic quarter circle and a cubic Bezier curve.
  const double w = std::sqrt(0.5);
  const int32_t quarter = b.add(PHX_MC_CAD_NODE_BSPLINE_CURVE, {2, 3, 2},
                                {0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, w, 1});
  const std::vector<double> bezier_points{0, 0, 0, 1, 2, 0, 3, -1, 1, 4, 0.5, -1};
  const int32_t bezier = b.add(PHX_MC_CAD_NODE_BSPLINE_CURVE, {3, 4, 3},
                               concat({{0, 0, 0, 0, 2, 2, 2, 2}, bezier_points, {1, 1, 1, 1}}));
  // Bilinear B-spline surface and a rational quadratic-by-linear cylinder.
  const int32_t bilinear = b.add(
      PHX_MC_CAD_NODE_BSPLINE_SURFACE, {1, 1, 2, 2},
      concat({{0, 0, 1, 1}, {0, 0, 2, 2}, {0, 0, 0, 2, 1, 3, 0, 0.5, 3, 2, 2, 0}, {1, 1, 1, 1}}));
  const int32_t rational = b.add(
      PHX_MC_CAD_NODE_BSPLINE_SURFACE, {2, 1, 3, 2},
      concat({{0, 0, 0, 1, 1, 1},
              {0, 0, 1, 1},
              {1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1},
              {1, 1, w, w, 1, 1}}));
  // One bicubic Bezier piece (identity extraction) and its mirror piece.
  std::vector<double> operators(2 * 16 * 16, 0.0);
  for (int piece = 0; piece < 2; ++piece)
    for (int l = 0; l < 16; ++l) operators[256 * piece + 16 * l + l] = 1.0;
  std::vector<double> controls, weights;
  std::vector<int64_t> rows{2, 32, 16};
  for (int l = 0; l < 32; ++l) {
    const int a = (l % 16) / 4, c = l % 4;
    const double shift = l < 16 ? 0.0 : 1.0;
    controls.insert(controls.end(), {shift + a / 3.0, c / 3.0, std::sin(0.3 * l)});
    weights.push_back(1.0 + 0.1 * (l % 5));
  }
  for (int64_t l = 0; l < 32; ++l) rows.push_back(l);
  const int32_t tspline = b.add(PHX_MC_CAD_NODE_TSPLINE_SURFACE, rows,
                                concat({{0, 1, 0, 1, 1, 2, 0, 1}, operators, controls, weights}));
  b.surfaces = {bilinear, rational, tspline};
  phx_mc_cad_model model = b.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  EvaluationWorkspace workspace;
  check_curve(geometry, bezier, [&](double t) {
    std::vector<double> x(3, 0.0);
    for (int i = 0; i < 4; ++i)
      for (int k = 0; k < 3; ++k) x[k] += bernstein(3, i, t / 2.0) * bezier_points[3 * i + k];
    return x;
  }, 0.0001, 1.9999);
  for (double t = 0.0; t <= 1.0; t += 0.03125) {
    double jet[6];
    PHX_CHECK(geometry.jets(quarter, &t, 2, jet, workspace));
    PHX_CHECK_NEAR(std::hypot(jet[0], jet[1]), 1.0, 4 * kEpsilon);
    // The speed vector is tangent to the unit circle.
    PHX_CHECK_NEAR(jet[0] * jet[2] + jet[1] * jet[3], 0.0, 1e-14);
  }
  const double outside = 1.5;
  double jet[9];
  PHX_CHECK(!geometry.jets(quarter, &outside, 0, jet, workspace));
  check_surface(geometry, bilinear, [](double u, double v) {
    const double s = v / 2.0;
    const Vector p00{0, 0, 0}, p01{2, 1, 3}, p10{0, 0.5, 3}, p11{2, 2, 0};
    Vector x{};
    for (int k = 0; k < 3; ++k)
      x[k] = (1 - u) * (1 - s) * p00[k] + (1 - u) * s * p01[k] + u * (1 - s) * p10[k] +
             u * s * p11[k];
    return x;
  }, 0.001, 0.999, 0.001, 1.999);
  for (double u = 0.0; u <= 1.0; u += 0.0625) {
    const double point[2] = {u, 0.5};
    double surface[18];
    PHX_CHECK(geometry.jets(rational, point, 2, surface, workspace));
    PHX_CHECK_NEAR(std::hypot(surface[0], surface[1]), 1.0, 4 * kEpsilon);
    PHX_CHECK_NEAR(surface[2], 0.5, 4 * kEpsilon);
  }
  // Clamped beyond the domain: the value of the edge, no u derivative.
  const double clamped[2] = {1.25, 0.5};
  double edge[18];
  PHX_CHECK(geometry.jets(rational, clamped, 1, edge, workspace));
  PHX_CHECK_NEAR(edge[0], 0.0, 4 * kEpsilon);
  PHX_CHECK(edge[3] == 0.0 && edge[4] == 0.0 && edge[5] == 0.0);
  const auto bezier_patch = [&](double u, double v) {
    const int base = u <= 1.0 ? 0 : 16;
    const double s = u <= 1.0 ? u : u - 1.0;
    Vector numerator{};
    double denominator = 0.0;
    for (int a = 0; a < 4; ++a)
      for (int c = 0; c < 4; ++c) {
        const int l = base + 4 * a + c;
        const double basis = bernstein(3, a, s) * bernstein(3, c, v) * weights[l];
        denominator += basis;
        for (int k = 0; k < 3; ++k) numerator[k] += basis * controls[3 * l + k];
      }
    return scaled(1.0 / denominator, numerator);
  };
  check_surface(geometry, tspline, bezier_patch, 0.01, 0.99, 0.01, 0.99);
  check_surface(geometry, tspline, bezier_patch, 1.01, 1.99, 0.01, 0.99);
  // On the shared break the first containing piece is used.
  const double seam[2] = {1.0, 0.25};
  double value[3];
  PHX_CHECK(geometry.jets(tspline, seam, 0, value, workspace));
  const Vector expected = bezier_patch(1.0, 0.25);
  for (int k = 0; k < 3; ++k) PHX_CHECK_NEAR(value[k], expected[k], 8 * kEpsilon);
  const double beyond[2] = {2.5, 0.5};
  PHX_CHECK(!geometry.jets(tspline, beyond, 0, value, workspace));
}

// The branch x > 0 of the z cylinder (R = 1) and the x cylinder (r = 0.6),
// charted by the small cylinder's angle: the closed form is
// (sqrt(1 - 0.36 cos^2), 0.6 cos, 0.6 sin).
void intersection_chart() {
  Builder b;
  const int32_t large = b.add_template(
      "cylinder", {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1.0});
  const int32_t small = b.add_template(
      "cylinder", {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.6});
  const auto closed = [](double theta) {
    const double y = 0.6 * std::cos(theta), z = 0.6 * std::sin(theta);
    return Vector{std::sqrt(1.0 - y * y), y, z};
  };
  const auto coordinates = [&](double theta) {
    const Vector x = closed(theta);
    return std::array<double, 4>{std::atan2(x[1], x[0]), x[2], theta, x[0]};
  };
  const double theta0 = -0.5, theta1 = 0.7;
  const auto start = coordinates(theta0), end = coordinates(theta1);
  // Preconditioner: the inverse free Jacobian (u1, v1, v2 columns) at the middle.
  const auto middle = coordinates(0.5 * (theta0 + theta1));
  const double c = std::cos(middle[0]), s = std::sin(middle[0]);
  // J = [[-s, 0, -1], [c, 0, 0], [0, 1, 0]]: R = S1 - S2 with S2 = (v2, ., .).
  const double inverse[9] = {0, 1 / c, 0, 0, 0, 1, -1, -s / c, 0};
  std::vector<double> reals{0.0, 0.0};
  reals.insert(reals.end(), start.begin(), start.end());
  reals.insert(reals.end(), end.begin(), end.end());
  reals.insert(reals.end(), {-1.0, -1.0, theta0, 0.5, 1.0, 1.0, theta1, 1.0});
  reals.insert(reals.end(), inverse, inverse + 9);
  const int32_t point = b.add(PHX_MC_CAD_NODE_INTERSECTION, {large, small, 1, 0, -1, -1, 0, 2}, reals);
  const int32_t first_uv =
      b.add(PHX_MC_CAD_NODE_INTERSECTION, {large, small, 1, 1, 0, -1, 0, 2}, reals);
  std::vector<double> reversed_reals = reals;
  reversed_reals[0] = 1.0;
  const int32_t second_uv =
      b.add(PHX_MC_CAD_NODE_INTERSECTION, {large, small, 1, 2, 0, -1, 1, 2}, reversed_reals);
  phx_mc_cad_model model = b.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const auto theta = [&](double t) { return theta0 + t * (theta1 - theta0); };
  check_curve(geometry, point, [&](double t) {
    const Vector x = closed(theta(t));
    return std::vector<double>{x[0], x[1], x[2]};
  }, 0.0, 1.0, 8.0);
  check_curve(geometry, first_uv, [&](double t) {
    const auto q = coordinates(theta(t));
    return std::vector<double>{q[0], q[1]};
  }, 0.0, 1.0, 8.0);
  check_curve(geometry, second_uv, [&](double t) {
    const auto q = coordinates(theta(1.0 - t));
    return std::vector<double>{q[2], q[3]};
  }, 0.0, 1.0, 8.0);
  EvaluationWorkspace workspace;
  const double outside = 1.25;
  double jet[9];
  PHX_CHECK(!geometry.jets(point, &outside, 0, jet, workspace));
}

void entry_point() {
  Builder b;
  const double radius = 2.0, minor = 0.5;
  const int32_t sphere =
      b.add_template("sphere", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t cylinder =
      b.add_template("cylinder", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  const int32_t torus = b.add_template(
      "torus", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}, {minor}}));
  const int32_t circle = b.add_template("circle3", concat({v3(kOrigin), v3(kE1), v3(kE2), {radius}}));
  const int32_t pcurve = b.add_template("line2", {0.0, 0.0, 1.0, 0.5});
  b.surfaces = {sphere, cylinder, torus};
  b.curves = {circle};
  b.pcurves = {pcurve};
  phx_mc_cad_model model = b.model();
  const double uv[4] = {0.3, 0.2, -1.0, 0.9};
  double out[16];
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_SURFACE, 0, 2, uv,
                                         PHX_MC_CAD_DIFFERENTIAL, out) == PHX_MC_OK);
  // Sphere: E = R^2 cos^2 v, G = R^2, outward normal, both curvatures -1/R.
  PHX_CHECK_NEAR(out[0], radius * radius * std::cos(0.2) * std::cos(0.2), 1e-13);
  PHX_CHECK_NEAR(out[1], 0.0, 1e-13);
  PHX_CHECK_NEAR(out[2], radius * radius, 1e-13);
  PHX_CHECK_NEAR(out[6], -1.0 / radius, 1e-13);
  PHX_CHECK_NEAR(out[7], -1.0 / radius, 1e-13);
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_SURFACE, 1, 1, uv,
                                         PHX_MC_CAD_DIFFERENTIAL, out) == PHX_MC_OK);
  PHX_CHECK_NEAR(std::min(std::fabs(out[6]), std::fabs(out[7])), 0.0, 1e-13);
  PHX_CHECK_NEAR(std::max(std::fabs(out[6]), std::fabs(out[7])), 1.0 / radius, 1e-13);
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_SURFACE, 2, 1, uv,
                                         PHX_MC_CAD_DIFFERENTIAL, out) == PHX_MC_OK);
  // Torus: |k| = 1/r around the tube, cos v / (R + r cos v) along it.
  const double along = std::cos(0.2) / (radius + minor * std::cos(0.2));
  PHX_CHECK_NEAR(std::max(std::fabs(out[6]), std::fabs(out[7])), 1.0 / minor, 1e-12);
  PHX_CHECK_NEAR(std::min(std::fabs(out[6]), std::fabs(out[7])), along, 1e-12);
  const double t = 0.7;
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_CURVE, 0, 1, &t,
                                         PHX_MC_CAD_DIFFERENTIAL, out) == PHX_MC_OK);
  PHX_CHECK_NEAR(out[0], radius, 1e-14);
  PHX_CHECK_NEAR(out[1], 1.0 / radius, 1e-14);
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_PCURVE, 0, 1, &t,
                                         PHX_MC_CAD_SECOND_DERIVATIVES, out) == PHX_MC_OK);
  PHX_CHECK(out[0] == 0.7 && out[1] == 0.35 && out[2] == 1.0 && out[3] == 0.5 && out[4] == 0.0);
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_PCURVE, 0, 1, &t,
                                         PHX_MC_CAD_DIFFERENTIAL, out) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&model, PHX_MC_CAD_SURFACE, 3, 1, uv,
                                         PHX_MC_CAD_VALUE, out) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_cad_geometry_evaluate(nullptr, PHX_MC_CAD_SURFACE, 0, 1, uv,
                                         PHX_MC_CAD_VALUE, out) == PHX_MC_INVALID_ARGUMENT);
  // A child that does not precede its parent, and a template of the wrong arity.
  Builder bad;
  bad.add(PHX_MC_CAD_NODE_OFFSET_SURFACE, {1}, {0.5});
  bad.add_template("sphere", concat({v3(kOrigin), v3(kE1), v3(kE2), v3(kAxis), {radius}}));
  bad.surfaces = {0};
  phx_mc_cad_model broken = bad.model();
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&broken, PHX_MC_CAD_SURFACE, 0, 1, uv,
                                         PHX_MC_CAD_VALUE, out) == PHX_MC_INVALID_INPUT);
  Builder short_template;
  short_template.add_template("sphere", {0.0, 1.0});
  short_template.surfaces = {0};
  phx_mc_cad_model truncated = short_template.model();
  PHX_CHECK(phx_mc_cad_geometry_evaluate(&truncated, PHX_MC_CAD_SURFACE, 0, 1, uv,
                                         PHX_MC_CAD_VALUE, out) == PHX_MC_INVALID_INPUT);
}

}  // namespace

int main() {
  quadric_templates();
  closed_forms_match_programs();
  curve_templates();
  composite_nodes();
  spline_nodes();
  intersection_chart();
  entry_point();
  return phx::mc::test::finish("test_cad_geometry");
}
