// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cad_faces.hpp"
#include "check.hpp"
#include "predicates.hpp"

namespace {

using phx::mc::CadCurveDiscretization;
using phx::mc::CadFaceMesh;
using phx::mc::cad::CadGeometry;

struct PlaneWithHole {
  std::vector<int32_t> kinds, surface_nodes, curve_nodes, pcurve_nodes;
  std::vector<int64_t> integer_offsets{0}, integers, real_offsets{0};
  std::vector<double> reals;
  std::vector<double> corners, ranges, use_ranges;
  std::vector<int32_t> curve_corners, use_entities;
  std::vector<int8_t> use_kinds, senses;
  std::vector<int64_t> loop_offsets{0, 4, 8}, region_offsets{0};
  std::vector<int32_t> loop_surfaces{0, 0};
  std::vector<int8_t> orientation{1};
  std::vector<double> periods{0, 0}, domains{-2, 2, -2, 2};
  std::vector<int64_t> sample_offsets{0};
  std::vector<double> samples;

  int32_t add(const char* name, const std::vector<double>& values) {
    const int32_t kind = phx::mc::cad::template_index(name);
    PHX_CHECK(kind >= 0);
    kinds.push_back(PHX_MC_CAD_NODE_TEMPLATE);
    integers.push_back(kind);
    integer_offsets.push_back(static_cast<int64_t>(integers.size()));
    reals.insert(reals.end(), values.begin(), values.end());
    real_offsets.push_back(static_cast<int64_t>(reals.size()));
    return static_cast<int32_t>(kinds.size() - 1);
  }

  PlaneWithHole() {
    surface_nodes.push_back(add("plane", {0,0,0, 1,0,0, 0,1,0}));
    const std::array<std::array<double, 2>, 8> p{{
        {-1,-1}, {1,-1}, {1,1}, {-1,1},
        {-0.3,-0.3}, {-0.3,0.3}, {0.3,0.3}, {0.3,-0.3}}};
    for (const auto& q : p) corners.insert(corners.end(), {q[0], q[1], 0});
    for (int32_t i = 0; i < 8; ++i) {
      const int32_t j = i < 4 ? (i + 1) % 4 : 4 + (i - 3) % 4;
      const double dx = p[j][0] - p[i][0], dy = p[j][1] - p[i][1];
      curve_nodes.push_back(add("line3", {p[i][0], p[i][1], 0, dx, dy, 0}));
      pcurve_nodes.push_back(add("line2", {p[i][0], p[i][1], dx, dy}));
      curve_corners.insert(curve_corners.end(), {i, j});
      ranges.insert(ranges.end(), {0, 1});
      use_ranges.insert(use_ranges.end(), {0, 1});
      use_entities.push_back(i);
      use_kinds.push_back(0);
      senses.push_back(1);
      for (int k = 0; k <= 8; ++k) samples.push_back(k / 8.0);
      sample_offsets.push_back(static_cast<int64_t>(samples.size()));
    }
  }

  phx_mc_cad_model model() const {
    phx_mc_cad_model m{};
    m.node_count = static_cast<int64_t>(kinds.size());
    m.node_kinds = kinds.data();
    m.node_integer_offsets = integer_offsets.data();
    m.node_integers = integers.data();
    m.node_real_offsets = real_offsets.data();
    m.node_reals = reals.data();
    m.corner_count = 8; m.corner_points = corners.data();
    m.surface_count = 1; m.surface_nodes = surface_nodes.data();
    m.surface_orientations = orientation.data(); m.surface_periods = periods.data();
    m.surface_parameter_domains = domains.data();
    m.curve_count = 8; m.curve_nodes = curve_nodes.data();
    m.curve_corners = curve_corners.data(); m.curve_ranges = ranges.data();
    m.loop_count = 2; m.loop_use_offsets = loop_offsets.data();
    m.loop_surfaces = loop_surfaces.data();
    m.use_count = 8; m.use_kinds = use_kinds.data(); m.use_entities = use_entities.data();
    m.use_senses = senses.data(); m.use_ranges = use_ranges.data();
    m.use_pcurve_nodes = pcurve_nodes.data(); m.region_surface_offsets = region_offsets.data();
    return m;
  }
};

void check_angle(const std::array<double, 3>& p, const std::array<double, 3>& q,
                 const std::array<double, 3>& r) {
  const std::array<double, 3> a{q[0]-p[0], q[1]-p[1], q[2]-p[2]};
  const std::array<double, 3> b{r[0]-p[0], r[1]-p[1], r[2]-p[2]};
  const double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const double aa = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
  const double bb = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
  PHX_CHECK(aa > 0 && bb > 0);
  PHX_CHECK(dot / std::sqrt(aa * bb) <= std::cos(25.0 * 3.14159265358979323846 / 180) + 1e-12);
}

void check_loop(const CadFaceMesh& mesh, std::size_t begin, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const auto& edge = mesh.boundary_edges[begin + i];
    const auto& next = mesh.boundary_edges[begin + (i + 1) % count];
    PHX_CHECK(edge.vertices[1] == next.vertices[0]);
  }
}

void plane_with_hole() {
  PlaneWithHole fixture;
  // The following use starts one ulp away from the preceding use's end.
  // Closure must choose the preceding endpoint for this corner occurrence.
  const auto offset = fixture.real_offsets[fixture.pcurve_nodes[1]];
  fixture.reals[offset] = std::nextafter(fixture.reals[offset], 2.0);
  const phx_mc_cad_model model = fixture.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const CadCurveDiscretization curves{fixture.sample_offsets.data(), fixture.samples.data()};
  PHX_CHECK(phx::mc::validate_cad_curve_discretization(model, curves) == PHX_MC_OK);
  CadFaceMesh result;
  PHX_CHECK(phx::mc::cad_mesh_face(geometry, curves, 0, 0.4, 0.1, 0.5, result) == PHX_MC_OK);
  PHX_CHECK(result.boundary_edges.size() == 64);
  PHX_CHECK(result.slivers.empty());
  PHX_CHECK(!result.triangles.empty());
  check_loop(result, 0, 32);
  check_loop(result, 32, 32);
  PHX_CHECK(result.vertices[result.boundary_edges[8].vertices[0]].uv[0] == 1.0);
  for (const auto& triangle : result.triangles) {
    const auto& a = result.vertices[triangle[0]].uv;
    const auto& b = result.vertices[triangle[1]].uv;
    const auto& c = result.vertices[triangle[2]].uv;
    PHX_CHECK(phx::mc::orient2d(a.data(), b.data(), c.data()) > 0);
    const std::array<double, 3> pa{a[0], a[1], 0}, pb{b[0], b[1], 0}, pc{c[0], c[1], 0};
    check_angle(pa, pb, pc); check_angle(pb, pc, pa); check_angle(pc, pa, pb);
    const double x = (a[0] + b[0] + c[0]) / 3;
    const double y = (a[1] + b[1] + c[1]) / 3;
    PHX_CHECK(std::abs(x) >= 0.3 || std::abs(y) >= 0.3);
  }
  // A missing carve would leave triangle centroids inside the square hole.
}

// Rectangular charts of a cylinder and a torus. Opposite chart sides reuse
// the same curve row but have distinct UVs; this catches accidental seam welds.
void periodic_patch(bool torus, bool ungraded = false) {
  constexpr double pi = 3.14159265358979323846;
  constexpr double turn = 2 * pi;
  PlaneWithHole bank;
  bank.kinds.clear(); bank.integers.clear(); bank.reals.clear();
  bank.integer_offsets = {0}; bank.real_offsets = {0};
  bank.surface_nodes.clear(); bank.curve_nodes.clear(); bank.pcurve_nodes.clear();
  const std::vector<double> frame{0,0,0, 1,0,0, 0,1,0, 0,0,1};
  std::vector<double> coefficients = frame;
  coefficients.push_back(torus ? 2.0 : 1.0);
  if (torus) coefficients.push_back(0.5);
  bank.surface_nodes.push_back(bank.add(torus ? "torus" : "cylinder", coefficients));
  if (torus) {
    bank.curve_nodes.push_back(bank.add("circle3", {0,0,0, 1,0,0, 0,1,0, 2.5}));
    bank.curve_nodes.push_back(bank.add("circle3", {2,0,0, 1,0,0, 0,0,1, 0.5}));
    bank.corners = {2.5,0,0};
    bank.curve_corners = {0,0, 0,0};
    bank.ranges = {0,turn, 0,turn};
  } else {
    bank.curve_nodes.push_back(bank.add("circle3", {0,0,0, 1,0,0, 0,1,0, 1}));
    bank.curve_nodes.push_back(bank.add("line3", {1,0,0, 0,0,1}));
    bank.curve_nodes.push_back(bank.add("circle3", {0,0,1, 1,0,0, 0,1,0, 1}));
    bank.corners = {1,0,0, 1,0,1};
    bank.curve_corners = {0,0, 0,1, 1,1};
    bank.ranges = {0,turn, 0,1, 0,turn};
  }
  const double height = torus ? turn : 1.0;
  bank.pcurve_nodes.push_back(bank.add("line2", {0,0, 1,0}));
  bank.pcurve_nodes.push_back(bank.add("line2", {turn,0, 0,1}));
  bank.pcurve_nodes.push_back(bank.add("line2", {0,height, 1,0}));
  bank.pcurve_nodes.push_back(bank.add("line2", {0,0, 0,1}));
  bank.use_entities = torus ? std::vector<int32_t>{0,1,0,1}
                            : std::vector<int32_t>{0,1,2,1};
  bank.use_kinds = {0,0,0,0}; bank.senses = {1,1,-1,-1};
  bank.use_ranges = {0,turn, 0,height, 0,turn, 0,height};
  bank.loop_offsets = {0,4}; bank.loop_surfaces = {0};
  bank.periods = {turn, torus ? turn : 0};
  bank.domains = {0,turn, 0,height};
  const double requested_size = ungraded ? 0.3 : 0.1;
  const double curvature_size = std::sqrt(6.0 * 0.5 * 0.4 / (torus ? 2.0 : 1.0));
  const double boundary_size = std::min(requested_size, curvature_size);
  bank.samples.clear(); bank.sample_offsets = {0};
  for (std::size_t i = 0; i < bank.curve_nodes.size(); ++i) {
    const double length = torus ? (i == 0 ? 2.5 * turn : 0.5 * turn)
                                : (i == 1 ? 1.0 : turn);
    // Independent analytic arclength integration: ceil(length / h) gives
    // every segment metric length <= 1 in the requested/curvature size field.
    const int n = ungraded ? 80 : static_cast<int>(std::ceil(length / boundary_size));
    if (!ungraded) PHX_CHECK(length / n <= boundary_size);
    for (int j = 0; j <= n; ++j) bank.samples.push_back(static_cast<double>(j) / n);
    bank.sample_offsets.push_back(static_cast<int64_t>(bank.samples.size()));
  }
  phx_mc_cad_model model = bank.model();
  model.corner_count = torus ? 1 : 2;
  model.curve_count = static_cast<int64_t>(bank.curve_nodes.size());
  model.loop_count = 1; model.use_count = 4;
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const CadCurveDiscretization curves{bank.sample_offsets.data(), bank.samples.data()};
  PHX_CHECK(phx::mc::validate_cad_curve_discretization(model, curves) == PHX_MC_OK);
  CadFaceMesh result;
  const int32_t status = phx::mc::cad_mesh_face(geometry, curves, 0, requested_size, 0.4, 0.5, result);
  std::fprintf(stderr, "%s status=%d triangles=%zu unresolved=%lld lens=%lld outside=%lld proximity=%lld nonstar=%lld\n",
               ungraded ? "torus-80x80" : torus ? "torus" : "cylinder", status, result.triangles.size(),
               static_cast<long long>(result.stats.unresolved_triangles),
               static_cast<long long>(result.stats.encroachment_refusals),
               static_cast<long long>(result.stats.outside_refusals),
               static_cast<long long>(result.stats.proximity_refusals),
               static_cast<long long>(result.stats.nonstar_refusals));
  if (ungraded) {
    PHX_CHECK(status == PHX_MC_UNRESOLVED);
    PHX_CHECK(result.stats.encroachment_refusals == 44);
    PHX_CHECK(result.stats.unresolved_triangles == 32);
  } else {
    PHX_CHECK(status == PHX_MC_OK);
  }
  std::size_t expected_edges = 0;
  for (int32_t use : bank.use_entities)
    expected_edges += bank.sample_offsets[use + 1] - bank.sample_offsets[use] - 1;
  PHX_CHECK(result.boundary_edges.size() == expected_edges);
  PHX_CHECK(result.slivers.empty());
  check_loop(result, 0, result.boundary_edges.size());
  for (const auto& edge : result.boundary_edges) {
    const int32_t curve = bank.use_entities[edge.use];
    PHX_CHECK(edge.curve == curve);
    const auto begin = bank.samples.begin() + bank.sample_offsets[curve];
    const auto end = bank.samples.begin() + bank.sample_offsets[curve + 1];
    const auto a = std::find(begin, end, edge.parameters[0]);
    const auto b = std::find(begin, end, edge.parameters[1]);
    PHX_CHECK(a != end && b != end);
    if (a != end && b != end) PHX_CHECK(std::abs(a - b) == 1);
  }
  // Every seam interior is represented by two distinct UV points with the
  // same curve row and sample index, rather than welded in the chart.
  for (int32_t seam = 0; seam < static_cast<int32_t>(bank.curve_nodes.size()); ++seam) {
    if (!torus && seam != 1) continue;
    const int64_t count = bank.sample_offsets[seam + 1] - bank.sample_offsets[seam];
    for (int32_t index = 1; index < count - 1; ++index) {
      int copies = 0;
      std::array<double, 2> first{};
      for (const auto& vertex : result.vertices) {
        if (vertex.dimension != 1 || vertex.entity != seam || vertex.curve_index != index)
          continue;
        if (copies == 0) first = vertex.uv;
        else PHX_CHECK(vertex.uv != first);
        ++copies;
      }
      PHX_CHECK(copies == 2);
    }
  }
  for (const auto& triangle : result.triangles) {
    const auto& a = result.vertices[triangle[0]].uv;
    const auto& b = result.vertices[triangle[1]].uv;
    const auto& c = result.vertices[triangle[2]].uv;
    PHX_CHECK(phx::mc::orient2d(a.data(), b.data(), c.data()) > 0);
    const auto position = [torus](const std::array<double, 2>& uv) {
      const double radial = torus ? 2.0 + 0.5 * std::cos(uv[1]) : 1.0;
      return std::array<double, 3>{radial * std::cos(uv[0]), radial * std::sin(uv[0]),
                                   torus ? 0.5 * std::sin(uv[1]) : uv[1]};
    };
    const auto x = position(a), y = position(b), z = position(c);
    if (!ungraded) {
      check_angle(x, y, z); check_angle(y, z, x); check_angle(z, x, y);
    }
    const double px = (x[0] + y[0] + z[0]) / 3;
    const double py = (x[1] + y[1] + z[1]) / 3;
    const double pz = (x[2] + y[2] + z[2]) / 3;
    const double radial = std::hypot(px, py);
    const double error = torus ? std::abs(std::hypot(radial - 2.0, pz) - 0.5)
                               : std::abs(radial - 1.0);
    PHX_CHECK(error < 0.2);
  }
}

}  // namespace

int main() {
  plane_with_hole();
  periodic_patch(false);
  periodic_patch(true);
  periodic_patch(true, true);
  return phx::mc::test::finish("test_cad_mesh");
}
