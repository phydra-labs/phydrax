// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#include "cad_faces.hpp"
#include "cad_mesh.hpp"
#include "check.hpp"
#include "predicates.hpp"
#include "cad_quadric_model.inc"

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
  int32_t mapped(int32_t surface, int32_t pcurve) {
    kinds.push_back(PHX_MC_CAD_NODE_SURFACE_MAPPED);
    integers.insert(integers.end(), {surface, pcurve});
    integer_offsets.push_back(static_cast<int64_t>(integers.size()));
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

bool angle_ok(const std::array<double, 3>& p, const std::array<double, 3>& q,
              const std::array<double, 3>& r) {
  const std::array<double, 3> a{q[0]-p[0], q[1]-p[1], q[2]-p[2]};
  const std::array<double, 3> b{r[0]-p[0], r[1]-p[1], r[2]-p[2]};
  const double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const double aa = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
  const double bb = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
  return aa > 0 && bb > 0 &&
         dot / std::sqrt(aa * bb) <= std::cos(25.0 * 3.14159265358979323846 / 180) + 1e-12;
}

void check_angle(const std::array<double, 3>& p, const std::array<double, 3>& q,
                 const std::array<double, 3>& r) {
  PHX_CHECK(angle_ok(p, q, r));
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

// A noninteger one-ulp closure gap on a periodic axis is rounding, not a wrap.
void periodic_rounding_gap() {
  PlaneWithHole fixture;
  fixture.periods = {10.0,0.0};
  fixture.reals[fixture.real_offsets[fixture.pcurve_nodes[3]] + 2] = 0x1p-53;
  const phx_mc_cad_model model = fixture.model();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const CadCurveDiscretization curves{fixture.sample_offsets.data(),fixture.samples.data()};
  CadFaceMesh result;
  PHX_CHECK(phx::mc::cad_mesh_face(geometry,curves,0,0.4,0.1,0.5,result) == PHX_MC_OK);
  PHX_CHECK(result.failure.reason == 0);
  PHX_CHECK(result.periodic_corner_gaps == 1);
  PHX_CHECK(result.max_periodic_corner_gap_ulps == 1);
}
// A single closed square use has one in-use sheet jump. Its raw endpoint
// differs by one period, but lifting the later spans cancels that turn.
void cancelling_sheet_jump(bool huge = false, bool wrapped = false) {
  PlaneWithHole bank;
  bank.kinds.clear(); bank.integers.clear(); bank.reals.clear();
  bank.integer_offsets = {0}; bank.real_offsets = {0};
  bank.surface_nodes = {bank.add("plane", {0,0,0, 1,0,0, 0,1,0})};
  const auto spline = [&](int dimension, const std::vector<double>& data, int count) {
    bank.kinds.push_back(PHX_MC_CAD_NODE_BSPLINE_CURVE);
    bank.integers.insert(bank.integers.end(), {1,count,dimension});
    bank.integer_offsets.push_back(static_cast<int64_t>(bank.integers.size()));
    bank.reals.insert(bank.reals.end(), data.begin(), data.end());
    bank.real_offsets.push_back(static_cast<int64_t>(bank.reals.size()));
    return static_cast<int32_t>(bank.kinds.size()-1);
  };
  bank.curve_nodes = {spline(3, {0,0,.25,.5,.75,1,1,
                                  -1,-1,0, 1,-1,0, 1,1,0, -1,1,0, -1,-1,0,
                                  1,1,1,1,1}, 5)};
  const double shift = huge ? 0x1p55 : 4.0;
  bank.pcurve_nodes = {spline(2, {0,0,.25,.5,.5,.75,1,1,
                                   -1,-1, 1,-1, (wrapped ? shift+1 : 1.0),1, shift+1,1,
                                   shift-1,1, shift-1,-1, 1,1,1,1,1,1}, 6)};
  bank.corners = {-1,-1,0}; bank.curve_corners = {0,0};
  bank.ranges = {0,1}; bank.use_ranges = {0,1};
  bank.use_kinds = {0}; bank.use_entities = {0}; bank.senses = {1};
  bank.loop_offsets = {0,1}; bank.loop_surfaces = {0};
  bank.periods = {4,0}; bank.domains = {-1,3,-1,1};
  bank.samples = {0,.125,.25,.375,.5,.625,.75,.875,1};
  bank.sample_offsets = {0,9};
  auto model = bank.model();
  model.corner_count = 1; model.curve_count = 1; model.loop_count = 1; model.use_count = 1;
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  const CadCurveDiscretization curves{bank.sample_offsets.data(),bank.samples.data()};
  CadFaceMesh mesh;
  const int32_t status = phx::mc::cad_mesh_face(geometry,curves,0,1.0,1.0,0.5,mesh);
  if (wrapped) {
    PHX_CHECK(status == PHX_MC_UNRESOLVED);
    PHX_CHECK(mesh.failure.reason == PHX_MC_CAD_FACE_PERIODIC_WRAP);
    PHX_CHECK(mesh.failure.face == 0 && mesh.failure.loop == 0 &&
              mesh.failure.axis == 0 && mesh.failure.turns == 1);
  } else if (huge) {
    PHX_CHECK(status == PHX_MC_UNRESOLVED);
    PHX_CHECK(mesh.failure.reason == PHX_MC_CAD_FACE_SHIFT_UNREPRESENTABLE);
    PHX_CHECK(mesh.failure.face == 0 && mesh.failure.use == 0 && mesh.failure.axis == 0);
  } else {
    PHX_CHECK(status == PHX_MC_OK);
    PHX_CHECK(mesh.failure.reason == 0);
    PHX_CHECK(!mesh.triangles.empty());
    for (const auto& t : mesh.triangles)
      PHX_CHECK(phx::mc::orient2d(mesh.vertices[t[0]].uv.data(),
                                  mesh.vertices[t[1]].uv.data(),
                                  mesh.vertices[t[2]].uv.data()) > 0);
  }
}


struct SurfaceArrays {
  std::vector<double> points, parameters, uv, sliver_uv, edge_parameters;
  std::vector<int8_t> dimensions;
  std::vector<int32_t> entities, triangles, patches, uses, regions, sliver_patches, edges, edge_curves;
  std::vector<phx_mc_cad_face_stats> stats;
  std::vector<int64_t> curve_offsets, sag_splits;
  std::vector<double> curve_parameters;
  std::vector<phx_mc_cad_curve_failure> curve_failures;
  std::vector<phx_mc_cad_face_failure> face_failures;
  phx_mc_cad_curve_statistics curve_stats{};
};
// Closed-form first fundamental forms, independent of the CAD jet evaluator.
void check_boundary_lengths(const phx_mc_cad_model& model, const SurfaceArrays& mesh,
                            double size) {
  std::vector<double> maximum(static_cast<std::size_t>(model.surface_count), 0.0);
  for (std::size_t t = 0; t < mesh.patches.size(); ++t) {
    const int face = mesh.patches[t];
    const int32_t node = model.surface_nodes[face];
    const bool quadric = model.node_kinds[node] != PHX_MC_CAD_NODE_TEMPLATE;
    const int kind = quadric ? -1 : static_cast<int>(model.node_integers[model.node_integer_offsets[node]]);
    const double* c = model.node_reals + model.node_real_offsets[node];
    for (int k = 0; k < 3; ++k) {
      if (mesh.uses[3*t+k] < 0) continue;
      const int next = (k+1)%3;
      const double u = (mesh.uv[6*t+2*k] + mesh.uv[6*t+2*next])/2;
      const double v = (mesh.uv[6*t+2*k+1] + mesh.uv[6*t+2*next+1])/2;
      const double du = mesh.uv[6*t+2*k] - mesh.uv[6*t+2*next];
      const double dv = mesh.uv[6*t+2*k+1] - mesh.uv[6*t+2*next+1];
      double metric_length = 0.0;
      if (quadric) {
        const double e = face == 2 ? 0.36 : 1.0, g = face == 2 ? 9.0 : 5.76;
        metric_length = std::hypot(std::sqrt(e)*du, std::sqrt(g)*dv);
      } else {
        const bool plane = kind == phx::mc::cad::template_index("plane");
        const bool cylinder = kind == phx::mc::cad::template_index("cylinder");
        const bool sphere = kind == phx::mc::cad::template_index("sphere");
        const bool cone = kind == phx::mc::cad::template_index("cone");
        const bool torus = kind == phx::mc::cad::template_index("torus");
        PHX_CHECK(plane || cylinder || sphere || cone || torus);
        double sum = 0.0;
        for (int j = 0; j < 3; ++j) {
          const double radial = std::cos(u)*c[3+j] + std::sin(u)*c[6+j];
          const double tangent = -std::sin(u)*c[3+j] + std::cos(u)*c[6+j];
          double su = c[3+j], sv = c[6+j];
          if (cylinder) { su = c[12]*tangent; sv = c[9+j]; }
          if (sphere) { su = c[12]*std::cos(v)*tangent;
                        sv = c[12]*(-std::sin(v)*radial+std::cos(v)*c[9+j]); }
          if (cone) { const double slope = std::tan(c[13]); su = (c[12]+v*slope)*tangent;
                      sv = c[9+j]+slope*radial; }
          if (torus) { su = (c[12]+c[13]*std::cos(v))*tangent;
                       sv = c[13]*(-std::sin(v)*radial+std::cos(v)*c[9+j]); }
          const double delta = su*du+sv*dv;
          sum += delta*delta;
        }
        metric_length = std::sqrt(sum);
      }
      maximum[face] = std::max(maximum[face], metric_length/size);
    }
  }
  for (std::size_t face = 0; face < maximum.size(); ++face) {
    std::fprintf(stderr, "assembly h=%.17g face=%zu max_boundary_metric_length=%.17g\n",
                 size, face, maximum[face]);
    PHX_CHECK(maximum[face] <= 1.0 + 1e-12);
  }
}


SurfaceArrays assemble_surface(const phx_mc_cad_model& model, double size) {
  const std::vector<double> sizes(static_cast<std::size_t>(model.surface_count), size);
  const std::vector<double> tolerances(static_cast<std::size_t>(model.surface_count), 0.4);
  const std::vector<double> curve_tolerances(static_cast<std::size_t>(model.curve_count),
                                             std::numeric_limits<double>::infinity());
  const phx_mc_cad_surface_controls controls{sizes.data(), tolerances.data(),
                                             curve_tolerances.data(), 0.5, 1.3, 1e-8};
  phx_mc_cad_surface_result* handle = nullptr;
  const int32_t status = phx_mc_cad_mesh(&model, &controls, &handle);
  if (status != PHX_MC_OK)
    std::fprintf(stderr, "assembly faces=%lld curves=%lld status=%d handle=%d\n",
                 static_cast<long long>(model.surface_count), static_cast<long long>(model.curve_count),
                 status, handle != nullptr);
  PHX_CHECK(status == PHX_MC_OK);
  if (!handle) return {};
  int64_t counts[9]{};
  PHX_CHECK(phx_mc_cad_surface_result_sizes(handle, counts) == PHX_MC_OK);
  const auto v = static_cast<std::size_t>(counts[0]), t = static_cast<std::size_t>(counts[1]);
  const auto s = static_cast<std::size_t>(counts[2]), e = static_cast<std::size_t>(counts[3]);
  const auto f = static_cast<std::size_t>(counts[4]);
  SurfaceArrays result;
  result.points.resize(3*v); result.dimensions.resize(v); result.entities.resize(v);
  result.parameters.resize(2*v); result.triangles.resize(3*t); result.patches.resize(t);
  result.uv.resize(6*t); result.uses.resize(3*t); result.regions.resize(2*t);
  result.sliver_uv.resize(6*s); result.sliver_patches.resize(s); result.edges.resize(2*e);
  result.edge_curves.resize(e); result.edge_parameters.resize(2*e); result.stats.resize(f);
  result.curve_offsets.resize(static_cast<std::size_t>(counts[5])+1);
  result.curve_parameters.resize(static_cast<std::size_t>(counts[6]));
  result.sag_splits.resize(static_cast<std::size_t>(counts[5]));
  result.curve_failures.resize(static_cast<std::size_t>(counts[7]));
  result.face_failures.resize(static_cast<std::size_t>(counts[8]));
  phx_mc_cad_surface_arrays arrays{counts[0],counts[1],counts[2],counts[3],counts[4],
      counts[5],counts[6],counts[7],counts[8],
      result.points.data(),result.dimensions.data(),result.entities.data(),result.parameters.data(),
      result.triangles.data(),result.patches.data(),result.uv.data(),result.uses.data(),result.regions.data(),
      result.sliver_uv.data(),result.sliver_patches.data(),result.edges.data(),result.edge_curves.data(),
      result.edge_parameters.data(),result.stats.data(), result.curve_offsets.data(),
      result.curve_parameters.data(), result.sag_splits.data(), result.curve_failures.data(),
      &result.curve_stats, result.face_failures.data()};
  phx_mc_cad_surface_arrays short_arrays = arrays;
  short_arrays.triangle_capacity = counts[1] - 1;
  if (!result.points.empty()) result.points[0] = 42.0;
  PHX_CHECK(phx_mc_cad_surface_result_export(handle, &short_arrays) == PHX_MC_CAPACITY_EXCEEDED);
  if (!result.points.empty()) PHX_CHECK(result.points[0] == 42.0);
  PHX_CHECK(phx_mc_cad_surface_result_export(handle, &arrays) == PHX_MC_OK);
  check_boundary_lengths(model, result, size);
  for (const auto& failure : result.face_failures)
    std::fprintf(stderr, "failure face=%d loop=%d use=%d axis=%d reason=%d turns=%lld parameter=%.17g\n",
                 failure.face, failure.loop, failure.use, failure.axis, failure.reason,
                 static_cast<long long>(failure.turns), failure.parameter);
  if (status != PHX_MC_OK) {
    for (std::size_t i = 0; i < result.stats.size(); ++i)
      std::fprintf(stderr, "  face=%zu unresolved=%lld lens=%lld proximity=%lld excluded=%lld\n",
                   i, static_cast<long long>(result.stats[i].unresolved_triangles),
                   static_cast<long long>(result.stats[i].encroachment_refusals),
                   static_cast<long long>(result.stats[i].proximity_refusals),
                   static_cast<long long>(result.stats[i].excluded_refusals));
    std::fprintf(stderr, "  curve unresolved=%lld sag_splits=%lld\n",
                 static_cast<long long>(result.curve_stats.unresolved_count),
                 static_cast<long long>(result.curve_stats.sag_splits));
  }
  phx_mc_cad_surface_result_free(handle);
  return result;
}

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(T)) == 0);
}

void check_deterministic(const SurfaceArrays& a, const SurfaceArrays& b) {
  PHX_CHECK(same_bytes(a.points,b.points) && same_bytes(a.parameters,b.parameters) &&
            same_bytes(a.uv,b.uv) && same_bytes(a.sliver_uv,b.sliver_uv) &&
            same_bytes(a.edge_parameters,b.edge_parameters) && same_bytes(a.dimensions,b.dimensions) &&
            same_bytes(a.entities,b.entities) && same_bytes(a.triangles,b.triangles) &&
            same_bytes(a.patches,b.patches) && same_bytes(a.uses,b.uses) &&
            same_bytes(a.regions,b.regions) && same_bytes(a.sliver_patches,b.sliver_patches) &&
            same_bytes(a.edges,b.edges) && same_bytes(a.edge_curves,b.edge_curves) &&
            same_bytes(a.stats,b.stats) && same_bytes(a.curve_offsets,b.curve_offsets) &&
            same_bytes(a.curve_parameters,b.curve_parameters) && same_bytes(a.sag_splits,b.sag_splits) &&
            std::memcmp(&a.curve_stats,&b.curve_stats,sizeof(a.curve_stats)) == 0);
}

void check_closed(const SurfaceArrays& mesh, int expected_euler) {
  std::map<std::pair<int32_t,int32_t>, std::pair<int,int>> edges;
  for (std::size_t i = 0; i < mesh.triangles.size(); i += 3)
    for (int k = 0; k < 3; ++k) {
      const int32_t a = mesh.triangles[i+k], b = mesh.triangles[i+(k+1)%3];
      auto& entry = edges[std::minmax(a,b)];
      ++entry.first; entry.second += a < b ? 1 : -1;
    }
  for (const auto& [edge, incidence] : edges) {
    (void)edge;
    PHX_CHECK(incidence.first == 2 && incidence.second == 0);
  }
  PHX_CHECK(static_cast<int64_t>(mesh.points.size()/3) - static_cast<int64_t>(edges.size()) +
            static_cast<int64_t>(mesh.triangles.size()/3) == expected_euler);
}

void check_outward(const SurfaceArrays& mesh, bool torus) {
  for (std::size_t i = 0; i < mesh.patches.size(); ++i) {
    const double* a = mesh.points.data() + 3 * mesh.triangles[3*i];
    const double* b = mesh.points.data() + 3 * mesh.triangles[3*i+1];
    const double* c = mesh.points.data() + 3 * mesh.triangles[3*i+2];
    const double x = (a[0]+b[0]+c[0])/3, y = (a[1]+b[1]+c[1])/3, z = (a[2]+b[2]+c[2])/3;
    const double ux=b[0]-a[0], uy=b[1]-a[1], uz=b[2]-a[2];
    const double vx=c[0]-a[0], vy=c[1]-a[1], vz=c[2]-a[2];
    const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    if (torus) {
      const double radial = std::hypot(x,y);
      PHX_CHECK(nx*x*(radial-2)/radial + ny*y*(radial-2)/radial + nz*z > 0);
    } else if (mesh.patches[i] == 0) {
      PHX_CHECK(nx*x + ny*y > 0);
    } else {
      PHX_CHECK(nz * (mesh.patches[i] == 1 ? -1 : 1) > 0);
    }
  }
}

void check_torus_assembly(const phx_mc_cad_model& model, double size) {
  const SurfaceArrays mesh = assemble_surface(model, size);
  const SurfaceArrays again = assemble_surface(model, size);
  check_deterministic(mesh, again);
  check_closed(mesh, 0);
  check_outward(mesh, true);
  for (std::size_t i = 0; i < mesh.points.size(); i += 3) {
    const double radial = std::hypot(mesh.points[i], mesh.points[i+1]);
    PHX_CHECK(std::abs((radial-2)*(radial-2)+mesh.points[i+2]*mesh.points[i+2]-0.25) < 4e-15);
  }
}

void check_cylinder_assembly(PlaneWithHole bank, double size) {
  constexpr double turn = 6.28318530717958647692;
  bank.surface_nodes.push_back(bank.add("plane", {0,0,0, 1,0,0, 0,1,0}));
  bank.surface_nodes.push_back(bank.add("plane", {0,0,1, 1,0,0, 0,1,0}));
  bank.orientation = {1,-1,1};
  bank.periods = {turn,0, 0,0, 0,0};
  bank.domains = {0,turn,0,1, -1,1,-1,1, -1,1,-1,1};
  for (int face = 1; face <= 2; ++face) {
    bank.pcurve_nodes.push_back(bank.add("circle2", {0,0, 1,0, 0,1, 1}));
    bank.use_entities.push_back(face == 1 ? 0 : 2);
    bank.use_kinds.push_back(0); bank.senses.push_back(1);
    bank.use_ranges.insert(bank.use_ranges.end(), {0,turn});
  }
  bank.loop_offsets = {0,4,5,6}; bank.loop_surfaces = {0,1,2};
  phx_mc_cad_model model = bank.model();
  model.corner_count = 2; model.surface_count = 3; model.curve_count = 3;
  model.loop_count = 3; model.use_count = 6;
  const SurfaceArrays mesh = assemble_surface(model, size);
  const SurfaceArrays again = assemble_surface(model, size);
  check_deterministic(mesh, again);
  check_closed(mesh, 2);
  check_outward(mesh, false);
  for (std::size_t i = 0; i < mesh.points.size(); i += 3) {
    const double radius = std::hypot(mesh.points[i], mesh.points[i+1]);
    PHX_CHECK(std::abs(radius - 1.0) < 4e-15 || std::abs(mesh.points[i+2]) < 4e-15 ||
              std::abs(mesh.points[i+2] - 1.0) < 4e-15);
  }
}

void check_sphere_assembly(PlaneWithHole bank, double size) {
  constexpr double pi = 3.14159265358979323846;
  bank.curve_nodes = {bank.add("circle3", {0,0,0, 1,0,0, 0,0,1, 1})};
  bank.corners = {0,0,-1, 0,0,1}; bank.curve_corners = {0,1};
  bank.ranges = {-pi/2,pi/2};
  bank.pcurve_nodes = {bank.add("line2", {0,-pi/2, 1,0}),
                       bank.add("line2", {2*pi,0, 0,1}),
                       bank.add("line2", {0,pi/2, 1,0}),
                       bank.add("line2", {0,0, 0,1})};
  bank.use_kinds = {1,0,1,0}; bank.use_entities = {0,0,1,0};
  bank.senses = {1,1,-1,-1};
  bank.use_ranges = {0,2*pi,-pi/2,pi/2,0,2*pi,-pi/2,pi/2};
  bank.loop_offsets = {0,4}; bank.loop_surfaces = {0};
  bank.periods = {2*pi,0}; bank.domains = {0,2*pi,-pi/2,pi/2};
  bank.samples.clear(); bank.sample_offsets = {0};
  const int n = static_cast<int>(std::ceil(pi/size));
  for (int i = 0; i <= n; ++i) bank.samples.push_back(static_cast<double>(i)/n);
  bank.sample_offsets.push_back(static_cast<int64_t>(bank.samples.size()));
  phx_mc_cad_model model = bank.model();
  model.corner_count = 2; model.surface_count = 1; model.curve_count = 1;
  model.loop_count = 1; model.use_count = 4;
  const SurfaceArrays mesh = assemble_surface(model, size);
  const SurfaceArrays again = assemble_surface(model, size);
  check_deterministic(mesh, again);
  check_closed(mesh, 2);
  for (std::size_t i = 0; i < mesh.points.size(); i += 3)
    PHX_CHECK(std::abs(mesh.points[i]*mesh.points[i] + mesh.points[i+1]*mesh.points[i+1] +
                       mesh.points[i+2]*mesh.points[i+2] - 1.0) < 4e-15);
  for (std::size_t i = 0; i < mesh.triangles.size(); i += 3) {
    const double* a = mesh.points.data() + 3*mesh.triangles[i];
    const double* b = mesh.points.data() + 3*mesh.triangles[i+1];
    const double* c = mesh.points.data() + 3*mesh.triangles[i+2];
    const double ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2];
    const double vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
    PHX_CHECK((uy*vz-uz*vy)*a[0] + (uz*vx-ux*vz)*a[1] + (ux*vy-uy*vx)*a[2] > 0);
  }
}

void check_cone_assembly(PlaneWithHole bank, double size) {
  constexpr double pi = 3.14159265358979323846;
  bank.surface_nodes.push_back(bank.add("plane", {0,0,1, 1,0,0, 0,1,0}));
  bank.orientation = {1,1}; bank.periods = {2*pi,0, 0,0};
  bank.domains = {0,2*pi,0,1, -1,1,-1,1};
  bank.pcurve_nodes.push_back(bank.add("circle2", {0,0, 1,0, 0,1, 1}));
  bank.use_entities.push_back(0); bank.use_kinds.push_back(0); bank.senses.push_back(1);
  bank.use_ranges.insert(bank.use_ranges.end(), {0,2*pi});
  bank.loop_offsets = {0,4,5}; bank.loop_surfaces = {0,1};
  phx_mc_cad_model model = bank.model();
  model.corner_count = 2; model.surface_count = 2; model.curve_count = 2;
  model.loop_count = 2; model.use_count = 5;
  const SurfaceArrays mesh = assemble_surface(model, size);
  const SurfaceArrays again = assemble_surface(model, size);
  check_deterministic(mesh, again);
  check_closed(mesh, 2);
  for (std::size_t i = 0; i < mesh.points.size(); i += 3) {
    const double r = std::hypot(mesh.points[i], mesh.points[i+1]);
    PHX_CHECK(std::abs(r-mesh.points[i+2]) < 4e-15 || std::abs(mesh.points[i+2]-1.0) < 4e-15);
  }
  for (std::size_t i = 0; i < mesh.patches.size(); ++i) {
    const double* a=mesh.points.data()+3*mesh.triangles[3*i];
    const double* b=mesh.points.data()+3*mesh.triangles[3*i+1];
    const double* c=mesh.points.data()+3*mesh.triangles[3*i+2];
    const double ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2];
    const double vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
    const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    if (mesh.patches[i] == 0)
      PHX_CHECK(nx*(a[0]+b[0]+c[0]) + ny*(a[1]+b[1]+c[1]) -
                nz*(a[2]+b[2]+c[2]) > 0);
    else PHX_CHECK(nz > 0);
  }
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
  if (torus && !ungraded) check_torus_assembly(model, requested_size);
  if (!torus && !ungraded) check_cylinder_assembly(bank, requested_size);
  CadFaceMesh result;
  const int32_t status = phx::mc::cad_mesh_face(geometry, curves, 0, requested_size, 0.4, 0.5, result);
  if (status != (ungraded ? PHX_MC_UNRESOLVED : PHX_MC_OK))
  std::fprintf(stderr, "%s status=%d triangles=%zu unresolved=%lld lens=%lld outside=%lld proximity=%lld nonstar=%lld\n",
               ungraded ? "torus-80x80" : torus ? "torus" : "cylinder", status, result.triangles.size(),
               static_cast<long long>(result.stats.unresolved_triangles),
               static_cast<long long>(result.stats.encroachment_refusals),
               static_cast<long long>(result.stats.outside_refusals),
               static_cast<long long>(result.stats.proximity_refusals),
               static_cast<long long>(result.stats.nonstar_refusals));
  if (ungraded) {
    PHX_CHECK(status == PHX_MC_UNRESOLVED);
    PHX_CHECK(result.stats.encroachment_refusals > 0);
    PHX_CHECK(result.stats.unresolved_triangles > 0);
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

void polar_patch(bool cone, bool trim = false, double size = 0.2, double trim_top = 1.45,
                 bool probe = false) {
  constexpr double pi = 3.14159265358979323846;
  constexpr double turn = 2.0 * pi;
  constexpr double angle = 0.78539816339744830962;
  const double low = cone ? 1.0 : 0.5;
  const double pole = cone ? 0.0 : pi / 2.0;
  const double radius = cone ? std::tan(angle) : std::cos(low);
  const double height = cone ? low : std::sin(low);
  PlaneWithHole bank;
  bank.kinds.clear(); bank.integers.clear(); bank.reals.clear();
  bank.integer_offsets = {0}; bank.real_offsets = {0};
  bank.surface_nodes.clear(); bank.curve_nodes.clear(); bank.pcurve_nodes.clear();
  std::vector<double> frame{0,0,0, 1,0,0, 0,1,0, 0,0,1};
  if (cone) { frame.push_back(0); frame.push_back(angle); }
  else frame.push_back(1);
  bank.surface_nodes.push_back(bank.add(cone ? "cone" : "sphere", frame));
  bank.curve_nodes.push_back(bank.add("circle3", {0,0,height, 1,0,0, 0,1,0, radius}));
  if (cone) bank.curve_nodes.push_back(bank.add("line3", {0,0,0, std::tan(angle),0,1}));
  else bank.curve_nodes.push_back(bank.add("circle3", {0,0,0, 1,0,0, 0,0,1, 1}));
  bank.corners = {radius,0,height, 0,0,cone ? 0.0 : 1.0};
  bank.curve_corners = {0,0, 0,1};
  bank.ranges = {0,turn, low,pole};
  bank.pcurve_nodes.push_back(bank.add("line2", {0,low, 1,0}));
  bank.pcurve_nodes.push_back(bank.add("line2", {turn,0, 0,1}));
  bank.pcurve_nodes.push_back(bank.add("line2", {0,pole, 1,0}));
  bank.pcurve_nodes.push_back(bank.add("line2", {0,0, 0,1}));
  bank.use_entities = {0,1,1,1}; bank.use_kinds = {0,0,1,0};
  bank.senses = {1,1,-1,-1};
  bank.use_ranges = {0,turn, low,pole, 0,turn, low,pole};
  bank.loop_offsets = {0,4}; bank.loop_surfaces = {0};
  bank.periods = {turn,0}; bank.domains = {0,turn, std::min(low,pole),std::max(low,pole)};
  const double meridian = std::abs(pole-low) * (cone ? 1/std::cos(angle) : 1);
  const int counts[2] = {static_cast<int>(std::ceil(turn * radius / size)),
                         static_cast<int>(std::ceil(meridian / size))};
  bank.samples.clear(); bank.sample_offsets = {0};
  for (int n : counts) {
    for (int i = 0; i <= n; ++i) bank.samples.push_back(static_cast<double>(i) / n);
    bank.sample_offsets.push_back(static_cast<int64_t>(bank.samples.size()));
  }
  if (trim) {
    const std::array<std::array<double, 2>, 4> uv{{{2,1.0},{2,trim_top},{3,trim_top},{3,1.0}}};
    for (const auto& p : uv)
      bank.corners.insert(bank.corners.end(), {std::cos(p[1])*std::cos(p[0]),
                                               std::cos(p[1])*std::sin(p[0]), std::sin(p[1])});
    for (int32_t i = 0; i < 4; ++i) {
      const int32_t j = (i + 1) % 4;
      const double du = uv[j][0] - uv[i][0], dv = uv[j][1] - uv[i][1];
      const int32_t pcurve = bank.add("line2", {uv[i][0], uv[i][1], du, dv});
      bank.pcurve_nodes.push_back(pcurve);
      bank.curve_nodes.push_back(bank.mapped(bank.surface_nodes[0], pcurve));
      bank.curve_corners.insert(bank.curve_corners.end(), {2+i, 2+j});
      bank.ranges.insert(bank.ranges.end(), {0,1});
      bank.use_entities.push_back(2+i);
      bank.use_kinds.push_back(0); bank.senses.push_back(1);
      bank.use_ranges.insert(bank.use_ranges.end(), {0,1});
      const double length = du == 0 ? std::abs(dv) : std::abs(du * std::cos(uv[i][1]));
      const int n = static_cast<int>(std::ceil(length / size));
      for (int k = 0; k <= n; ++k) bank.samples.push_back(static_cast<double>(k) / n);
      bank.sample_offsets.push_back(static_cast<int64_t>(bank.samples.size()));
    }
    bank.loop_offsets = {0,4,8}; bank.loop_surfaces = {0,0};
  }
  phx_mc_cad_model model = bank.model();
  model.corner_count = trim ? 6 : 2; model.curve_count = trim ? 6 : 2;
  model.loop_count = trim ? 2 : 1; model.use_count = trim ? 8 : 4;
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  if (!cone && !trim) check_sphere_assembly(bank, size);
  if (cone) check_cone_assembly(bank, size);
  const CadCurveDiscretization curves{bank.sample_offsets.data(), bank.samples.data()};
  std::vector<std::array<double, 2>> metric_points;
  CadFaceMesh result;
  const int32_t status = phx::mc::cad_mesh_face(geometry, curves, 0, size, 0.4, 0.5,
                                                result, &metric_points);
  if (status != PHX_MC_OK) {
    std::fprintf(stderr, "trim probe h=%.17g top=%.17g\n", size, trim_top);
    std::fprintf(stderr, "%s status=%d faces=%zu slivers=%zu omitted=%lld fixed=%lld excluded=%lld unresolved=%lld lens=%lld outside=%lld proximity=%lld nonstar=%lld\n",
               trim ? "sphere-trim" : cone ? "cone" : "sphere-cap", status, result.triangles.size(),
               result.slivers.size(), static_cast<long long>(result.omitted_pole_points),
               static_cast<long long>(result.stats.fixed_triangles_skipped),
               static_cast<long long>(result.stats.excluded_refusals),
               static_cast<long long>(result.stats.unresolved_triangles),
               static_cast<long long>(result.stats.encroachment_refusals),
               static_cast<long long>(result.stats.outside_refusals),
               static_cast<long long>(result.stats.proximity_refusals),
               static_cast<long long>(result.stats.nonstar_refusals));
  }
  if (probe) {
    PHX_CHECK(status == PHX_MC_OK || status == PHX_MC_UNRESOLVED);
  } else if (trim) {
    PHX_CHECK(status == PHX_MC_OK);
    PHX_CHECK(result.stats.unresolved_triangles == 0);
    PHX_CHECK(result.stats.encroachment_refusals == 0);
    PHX_CHECK(result.stats.excluded_refusals == 0);
    PHX_CHECK(result.stats.proximity_refusals == 0);
    PHX_CHECK(result.omitted_pole_points > 0);
  } else {
    PHX_CHECK(status == PHX_MC_OK);
  }
  PHX_CHECK(!result.slivers.empty());
  PHX_CHECK(result.stats.fixed_triangles_skipped > 0);
  PHX_CHECK(!metric_points.empty());
  const double ring_spacing = size * (cone ? std::cos(angle) : 1.0);
  for (const auto& uv : metric_points) PHX_CHECK(std::abs(uv[1] - pole) >= ring_spacing);
  int bad_angles = 0;
  for (const auto& cell : result.triangles) {
    const auto& a = result.vertices[cell[0]].uv;
    const auto& b = result.vertices[cell[1]].uv;
    const auto& c = result.vertices[cell[2]].uv;
    PHX_CHECK(phx::mc::orient2d(a.data(), b.data(), c.data()) > 0);
    const auto position = [cone](const std::array<double, 2>& uv) {
      const double r = cone ? uv[1] : std::cos(uv[1]);
      return std::array<double, 3>{r*std::cos(uv[0]), r*std::sin(uv[0]),
                                   cone ? uv[1] : std::sin(uv[1])};
    };
    const auto x = position(a), y = position(b), z = position(c);
    const std::array<std::array<double, 3>, 3> xyz{x,y,z};
    for (int i = 0; i < 3; ++i)
      if (!angle_ok(xyz[i], xyz[(i+1)%3], xyz[(i+2)%3])) ++bad_angles;
    const double px = (x[0]+y[0]+z[0])/3, py = (x[1]+y[1]+z[1])/3;
    const double pz = (x[2]+y[2]+z[2])/3;
    const double error = cone ? std::abs(std::hypot(px,py)-pz)/std::sqrt(2.0)
                              : std::abs(std::hypot(std::hypot(px,py),pz)-1.0);
    PHX_CHECK(error < 0.2);
  }
  if (!trim) PHX_CHECK(bad_angles == 0);
}

// A full cylinder turn with no seam is not a closed lifted chart loop. The
// failure must survive the C entry without fabricating a surface candidate.
void periodic_wrap_failure(bool huge = false) {
  constexpr double turn = 6.28318530717958647692;
  PlaneWithHole bank;
  bank.kinds.clear(); bank.integers.clear(); bank.reals.clear();
  bank.integer_offsets = {0}; bank.real_offsets = {0};
  bank.surface_nodes = {bank.add("cylinder", {0,0,0, 1,0,0, 0,1,0, 0,0,1, 1})};
  bank.curve_nodes = {bank.add("circle3", {0,0,0, 1,0,0, 0,1,0, 1})};
  bank.pcurve_nodes = {bank.add("line2", {0,0, 1,0})};
  bank.corners = {1,0,0}; bank.curve_corners = {0,0}; bank.ranges = {0,turn};
  bank.use_kinds = {0}; bank.use_entities = {0}; bank.senses = {1};
  bank.use_ranges = {0,turn}; bank.loop_offsets = {0,1}; bank.loop_surfaces = {0};
  bank.orientation = {1}; bank.periods = {huge ? std::ldexp(turn,-53) : turn,0};
  bank.domains = {0,turn,-1,1};
  phx_mc_cad_model model = bank.model();
  model.corner_count = 1; model.surface_count = 1; model.curve_count = 1;
  model.loop_count = 1; model.use_count = 1;
  const double size = 0.2, tolerance = 0.4, curve_tolerance = 1.0;
  const phx_mc_cad_surface_controls controls{&size,&tolerance,&curve_tolerance,0.5,1.3,1e-8};
  phx_mc_cad_surface_result* handle = nullptr;
  PHX_CHECK(phx_mc_cad_mesh(&model, &controls, &handle) == PHX_MC_UNRESOLVED);
  PHX_CHECK(handle != nullptr);
  if (!handle) return;
  int64_t counts[9]{};
  PHX_CHECK(phx_mc_cad_surface_result_sizes(handle, counts) == PHX_MC_OK);
  PHX_CHECK(counts[0] == 0 && counts[1] == 0 && counts[8] == 1);
  std::vector<int64_t> offsets(static_cast<std::size_t>(counts[5])+1), splits(counts[5]);
  std::vector<double> parameters(counts[6]);
  std::vector<phx_mc_cad_curve_failure> curve_failures(counts[7]);
  phx_mc_cad_curve_statistics curve_stats{};
  phx_mc_cad_face_failure failure{};
  phx_mc_cad_surface_arrays arrays{};
  arrays.curve_capacity=counts[5]; arrays.parameter_capacity=counts[6];
  arrays.failure_capacity=counts[7]; arrays.face_failure_capacity=1;
  arrays.curve_offsets=offsets.data(); arrays.curve_parameters=parameters.data();
  arrays.curve_sag_splits=splits.data(); arrays.curve_failures=curve_failures.data();
  arrays.curve_statistics=&curve_stats; arrays.face_failures=&failure;
  PHX_CHECK(phx_mc_cad_surface_result_export(handle,&arrays) == PHX_MC_OK);
  PHX_CHECK(failure.face == 0 && failure.loop == 0 && failure.axis == 0);
  if (huge) PHX_CHECK(failure.reason == PHX_MC_CAD_FACE_SHIFT_UNREPRESENTABLE);
  else PHX_CHECK(failure.reason == PHX_MC_CAD_FACE_PERIODIC_WRAP && failure.turns == 1);
  phx_mc_cad_surface_result_free(handle);
}

void quadric_assembly() {
  phx_mc_cad_model model{};
  model.node_count = quadric_node_kinds.size();
  model.node_kinds = quadric_node_kinds.data();
  model.node_integer_offsets = quadric_node_integer_offsets.data();
  model.node_integers = quadric_node_integers.data();
  model.node_real_offsets = quadric_node_real_offsets.data();
  model.node_reals = quadric_node_reals.data();
  model.corner_count = quadric_corner_points.size()/3; model.corner_points = quadric_corner_points.data();
  model.surface_count = quadric_surface_nodes.size(); model.surface_nodes = quadric_surface_nodes.data();
  model.surface_orientations = quadric_surface_orientations.data();
  model.surface_periods = quadric_surface_periods.data();
  model.surface_parameter_domains = quadric_surface_parameter_domains.data();
  model.curve_count = quadric_curve_nodes.size(); model.curve_nodes = quadric_curve_nodes.data();
  model.curve_corners = quadric_curve_corners.data(); model.curve_ranges = quadric_curve_ranges.data();
  model.loop_count = quadric_loop_surfaces.size(); model.loop_use_offsets = quadric_loop_use_offsets.data();
  model.loop_surfaces = quadric_loop_surfaces.data();
  model.use_count = quadric_use_kinds.size(); model.use_kinds = quadric_use_kinds.data();
  model.use_entities = quadric_use_entities.data(); model.use_senses = quadric_use_senses.data();
  model.use_ranges = quadric_use_ranges.data(); model.use_pcurve_nodes = quadric_use_pcurve_nodes.data();
  model.region_count = quadric_region_surface_offsets.size()-1;
  model.region_surface_offsets = quadric_region_surface_offsets.data();
  model.region_surfaces = quadric_region_surfaces.data();
  model.region_orientations = quadric_region_orientations.data();
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(model) == PHX_MC_OK);
  constexpr double spacing = 0.12;
  constexpr int intervals = 4096;
  std::vector<int64_t> offsets{0};
  std::vector<double> samples;
  phx::mc::cad::EvaluationWorkspace workspace;
  for (int32_t curve = 0; curve < model.curve_count; ++curve) {
    std::array<double, intervals+1> cumulative{};
    const double first = model.curve_ranges[2*curve], last = model.curve_ranges[2*curve+1];
    for (int i = 0; i < intervals; ++i) {
      const double parameter = first + (last-first)*(i+0.5)/intervals;
      double jet[6];
      PHX_CHECK(geometry.jets(geometry.curve_node(curve), &parameter, 1, jet, workspace));
      cumulative[i+1] = cumulative[i] + std::hypot(std::hypot(jet[3],jet[4]),jet[5]) *
                                      std::abs(last-first)/intervals;
    }
    std::vector<double> breaks{0.0, 1.0};
    for (int64_t use = 0; use < model.use_count; ++use) {
      if (model.use_kinds[use] != 0 || model.use_entities[use] != curve) continue;
      int degree = 0;
      geometry.curve_breakpoints(geometry.pcurve_node(use), model.use_ranges[2*use],
                                 model.use_ranges[2*use+1], breaks, degree);
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    const auto arc = [&](double s) {
      const double position = s * intervals;
      const int j = std::min(static_cast<int>(position), intervals - 1);
      return cumulative[j] + (position-j)*(cumulative[j+1]-cumulative[j]);
    };
    samples.push_back(0.0);
    for (std::size_t span = 1; span < breaks.size(); ++span) {
      const double low = arc(breaks[span-1]), high = arc(breaks[span]);
      const int count = std::max(1, static_cast<int>(std::ceil((high-low)/spacing)));
      for (int i = 1; i < count; ++i) {
        const double target = low + (high-low)*i/count;
        const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), target);
        const int j = static_cast<int>(it-cumulative.begin());
        const double fraction = (target-cumulative[j-1])/(cumulative[j]-cumulative[j-1]);
        samples.push_back((j-1+fraction)/intervals);
      }
      samples.push_back(breaks[span]);
    }
    offsets.push_back(static_cast<int64_t>(samples.size()));
  }
  const CadCurveDiscretization input{offsets.data(),samples.data()};
  const double sizes[3]{spacing,spacing,spacing}, tolerances[3]{0.4,0.4,0.4};
  const auto run = [&]() {
    phx::mc::CadSurfaceMesh native;
    PHX_CHECK(phx::mc::cad_mesh_surface(geometry,input,sizes,tolerances,0.5,native) == PHX_MC_OK);
    SurfaceArrays out;
    for (const auto& p : native.points) out.points.insert(out.points.end(),p.begin(),p.end());
    for (const auto& t : native.triangles) out.triangles.insert(out.triangles.end(),t.begin(),t.end());
    for (const auto& u : native.triangle_uv) out.uv.insert(out.uv.end(),u.begin(),u.end());
    for (const auto& r : native.facet_regions) out.regions.insert(out.regions.end(),r.begin(),r.end());
    out.patches = native.triangle_patches;
    return out;
  };
  const SurfaceArrays mesh = run(), again = run();
  check_deterministic(mesh, again);
  check_closed(mesh, 2);
  const SurfaceArrays generated = assemble_surface(model, spacing);
  check_closed(generated, 2);
  for (std::size_t i = 0; i < mesh.regions.size(); i += 2)
    PHX_CHECK(mesh.regions[i] == -1 && mesh.regions[i+1] == 0);
  for (std::size_t i = 0; i < mesh.points.size(); i += 3) {
    const double x=mesh.points[i], y=mesh.points[i+1], z=mesh.points[i+2];
    const double first = std::abs(x*x+y*y-1.0);
    const double second = std::abs(y*y+z*z-0.36);
    PHX_CHECK(std::min(first,second) < 8e-15);
  }
  for (std::size_t i = 0; i < mesh.patches.size(); ++i) {
    const double* a=mesh.points.data()+3*mesh.triangles[3*i];
    const double* b=mesh.points.data()+3*mesh.triangles[3*i+1];
    const double* c=mesh.points.data()+3*mesh.triangles[3*i+2];
    const double ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2];
    const double vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
    const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    const double x=a[0]+b[0]+c[0], y=a[1]+b[1]+c[1], z=a[2]+b[2]+c[2];
    PHX_CHECK(mesh.patches[i] < 2 ? nx*x+ny*y > 0 : ny*y+nz*z > 0);
  }
}

}  // namespace

int main() {
  plane_with_hole();
  periodic_rounding_gap();
  cancelling_sheet_jump();
  cancelling_sheet_jump(true);
  cancelling_sheet_jump(false, true);
  periodic_patch(false);
  periodic_patch(true);
  periodic_patch(true, true);
  polar_patch(false);
  polar_patch(false, true);
  polar_patch(false, true, 0.19, 1.44, true);
  polar_patch(false, true, 0.21, 1.46, true);
  polar_patch(false, true, 0.20, 1.40, true);
  polar_patch(true);
  periodic_wrap_failure();
  periodic_wrap_failure(true);
  quadric_assembly();
  return phx::mc::test::finish("test_cad_mesh");
}
