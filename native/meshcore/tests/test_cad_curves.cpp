// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Independent construction checks; these sample sag, they do not certify it.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "cad_curves.hpp"
#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {
using namespace phx::mc::cad;
using Vec = std::array<double, 3>;
constexpr double pi = 3.14159265358979323846;

struct Fixture {
  std::vector<int32_t> kinds, surfaces, curves, pcurves, entities, loop_faces, corners;
  std::vector<int64_t> io{0}, ro{0}, ints, loops{0}, regions{0};
  std::vector<double> reals, ranges, use_ranges, periods, domains;
  std::vector<double> corner_points{0, 0, 0, 1, 0, 0};
  std::vector<int8_t> orientations, use_kinds, senses;

  int32_t node(int32_t kind, const std::vector<int64_t>& ii, const std::vector<double>& rr) {
    kinds.push_back(kind);
    ints.insert(ints.end(), ii.begin(), ii.end());
    reals.insert(reals.end(), rr.begin(), rr.end());
    io.push_back(static_cast<int64_t>(ints.size()));
    ro.push_back(static_cast<int64_t>(reals.size()));
    return static_cast<int32_t>(kinds.size() - 1);
  }
  int32_t templ(const char* name, const std::vector<double>& rr) {
    const int32_t id = template_index(name);
    PHX_CHECK(id >= 0);
    return node(PHX_MC_CAD_NODE_TEMPLATE, {id}, rr);
  }
  void face(int32_t n) {
    surfaces.push_back(n); orientations.push_back(1);
    periods.insert(periods.end(), {0, 0}); domains.insert(domains.end(), {-10, 10, -10, 10});
  }
  void curve(int32_t n, double a, double b, bool closed = false) {
    curves.push_back(n); ranges.insert(ranges.end(), {a, b});
    corners.insert(corners.end(), {0, closed ? 0 : 1});
  }
  void use(int32_t face_index, int32_t curve_index, int32_t pc, double a, double b, int8_t sense = 1) {
    loop_faces.push_back(face_index); entities.push_back(curve_index); pcurves.push_back(pc);
    use_kinds.push_back(0); senses.push_back(sense); use_ranges.insert(use_ranges.end(), {a, b});
    loops.push_back(static_cast<int64_t>(entities.size()));
  }
  phx_mc_cad_model model() const {
    phx_mc_cad_model m{};
    m.node_count = kinds.size(); m.node_kinds = kinds.data(); m.node_integer_offsets = io.data();
    m.node_integers = ints.data(); m.node_real_offsets = ro.data(); m.node_reals = reals.data();
    m.corner_count = 2; m.corner_points = corner_points.data();
    m.surface_count = surfaces.size(); m.surface_nodes = surfaces.data();
    m.surface_orientations = orientations.data(); m.surface_periods = periods.data();
    m.surface_parameter_domains = domains.data(); m.curve_count = curves.size();
    m.curve_nodes = curves.data(); m.curve_corners = corners.data(); m.curve_ranges = ranges.data();
    m.loop_count = loop_faces.size(); m.loop_use_offsets = loops.data(); m.loop_surfaces = loop_faces.data();
    m.use_count = entities.size(); m.use_kinds = use_kinds.data(); m.use_entities = entities.data();
    m.use_senses = senses.data(); m.use_ranges = use_ranges.data(); m.use_pcurve_nodes = pcurves.data();
    m.region_surface_offsets = regions.data();
    return m;
  }
};

struct Run {
  CadGeometry geometry;
  CurveDiscretization result;
  explicit Run(const Fixture& f, const std::vector<double>& sizes, const std::vector<double>& ftol,
               const std::vector<double>& ctol, double theta = 0.5, double grading = 1.3,
               bool sag_check = true) {
    PHX_CHECK(geometry.bind(f.model()) == PHX_MC_OK);
    CurveControls controls{sizes.data(), ftol.data(), ctol.data(), theta, grading, 1e-8,
                           sag_check};
    PHX_CHECK(discretize_curves(geometry, controls, result) == PHX_MC_OK);
    PHX_CHECK(result.statistics.unresolved.empty());
    PHX_CHECK(result.offsets.size() == f.curves.size() + 1);
    if (result.offsets.size() != f.curves.size() + 1) return;
    PHX_CHECK(result.offsets.front() == 0);
    PHX_CHECK(result.offsets.back() == static_cast<int64_t>(result.parameters.size()));
    for (std::size_t c = 0; c < f.curves.size(); ++c) {
      const auto a = result.offsets[c], b = result.offsets[c + 1];
      PHX_CHECK(a >= 0 && b >= a + 2 && b <= static_cast<int64_t>(result.parameters.size()));
      if (a < 0 || b < a + 2 || b > static_cast<int64_t>(result.parameters.size())) continue;
      PHX_CHECK(result.parameters[a] == 0.0); PHX_CHECK(result.parameters[b - 1] == 1.0);
      for (auto i = a + 1; i < b; ++i) PHX_CHECK(result.parameters[i] > result.parameters[i - 1]);
    }
  }
  int segments(int c = 0) const {
    if (c < 0 || result.offsets.size() <= static_cast<std::size_t>(c + 1)) return 0;
    const auto a=result.offsets[c], b=result.offsets[c+1];
    if (a < 0 || b < a + 2 || b > static_cast<int64_t>(result.parameters.size())) return 0;
    return static_cast<int>(b-a-1);
  }
};

int32_t plane(Fixture& f) { return f.templ("plane", {0,0,0, 1,0,0, 0,1,0}); }
int32_t line2(Fixture& f) { return f.templ("line2", {0,0, 1,0}); }

double dist(Vec a, Vec b) { return std::hypot(std::hypot(a[0]-b[0], a[1]-b[1]), a[2]-b[2]); }
// Point-to-segment distance, not distance to its midpoint.
double chord_distance(Vec p, Vec a, Vec b) {
  Vec d{b[0]-a[0], b[1]-a[1], b[2]-a[2]};
  const double dd = d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
  const double t = dd == 0 ? 0 : std::clamp(((p[0]-a[0])*d[0]+(p[1]-a[1])*d[1]+(p[2]-a[2])*d[2])/dd, 0.0, 1.0);
  return dist(p, {a[0]+t*d[0], a[1]+t*d[1], a[2]+t*d[2]});
}
void sag(const Run& run, const std::function<Vec(double)>& point, double limit) {
  const auto& p = run.result.parameters;
  for (std::size_t i = 1; i < p.size(); ++i) {
    const Vec a = point(p[i-1]), b = point(p[i]);
    for (int j = 1; j < 128; ++j)
      PHX_CHECK(chord_distance(point(p[i-1] + (p[i]-p[i-1])*j/128.0), a, b) <= limit + 1e-12);
  }
}

double integrate(const std::function<double(double)>& f, double a, double b) {
  constexpr int n = 20000;
  double sum = f(a) + f(b);
  for (int i = 1; i < n; ++i) sum += (i % 2 ? 4 : 2) * f(a + (b-a)*i/n);
  return sum * (b-a) / (3*n);
}

// Detects incorrect integration or normalization of a non-unit native range.
void line_integral() {
  Fixture f; auto s = plane(f), pc = line2(f);
  auto line = f.templ("line3", {0,0,0, 2,0,0});
  f.face(s); f.curve(line, 2, 5); f.use(0, 0, pc, 2, 5);
  Run r(f, {0.7}, {100}, {100});
  const int expected=static_cast<int>(std::ceil(2.0*(5.0-2.0)/0.7));
  const int count=r.segments();
  PHX_CHECK(std::abs(count-expected)<=1);
  if (count>0)
    for (int i=0;i<=count;++i)
      PHX_CHECK_NEAR(r.result.parameters[i],static_cast<double>(i)/count,1e-10);
}

// Detects omission of curvature sizing, incorrect circle range, or sag repair.
void circle_integral() {
  Fixture f; auto s = plane(f), pc = line2(f);
  auto circle = f.templ("circle3", {0,0,0, 1,0,0, 0,1,0, 2});
  f.face(s); f.curve(circle, 0, 2*pi, true); f.use(0, 0, pc, 0, 2*pi);
  const double theta = 0.5, tol = 0.02;
  Run r(f, {10}, {tol}, {tol}, theta);
  const int expected = static_cast<int>(std::ceil(4*pi / std::sqrt(8*theta*tol*2)));
  PHX_CHECK(std::abs(r.segments()-expected) <= 1);
  sag(r, [](double t) -> Vec { return {2*std::cos(2*pi*t),2*std::sin(2*pi*t),0}; }, theta*tol);
}

// Detects treating varying ellipse speed as constant; reference uses independent quadrature.
void ellipse_integral() {
  Fixture f; auto s = plane(f), pc = line2(f);
  auto e = f.templ("ellipse3", {0,0,0, 1,0,0, 0,1,0, 3,1});
  f.face(s); f.curve(e, 0, pi/2); f.use(0, 0, pc, 0, pi/2);
  Run r(f, {0.3}, {100}, {100});
  const double length = integrate([](double t) { return std::hypot(3*std::sin(t), std::cos(t)); }, 0, pi/2);
  PHX_CHECK(std::abs(r.segments()-static_cast<int>(std::ceil(length/0.3))) <= 1);
}

// Each use selects the curve's shared source slice and maps it into its own
// chart. Evaluating the actual p-curve and face detects a wrong range or seam.
void check_uses(const Fixture& f, const Run& r) {
  if (r.result.offsets.size()!=f.curves.size()+1) {
    PHX_CHECK(false);
    return;
  }
  EvaluationWorkspace workspace;
  for (std::size_t u=0; u<f.entities.size(); ++u) {
    const int curve=f.entities[u];
    const auto begin=r.result.offsets[curve], end=r.result.offsets[curve+1];
    if (begin < 0 || end < begin+2 ||
        end > static_cast<int64_t>(r.result.parameters.size())) {
      PHX_CHECK(false);
      continue;
    }
    const double* shared=r.result.parameters.data()+begin;
    for (std::size_t v=0; v<u; ++v) {
      if (f.entities[v]!=curve) continue;
      const auto other=r.result.offsets[f.entities[v]];
      PHX_CHECK(std::memcmp(shared,r.result.parameters.data()+other,
                            static_cast<std::size_t>(end-begin)*sizeof(double))==0);
    }
    for (auto i=begin; i<end; ++i) {
      const double t=r.result.parameters[i];
      const double source=f.ranges[2*curve]+t*(f.ranges[2*curve+1]-f.ranges[2*curve]);
      const double use=f.use_ranges[2*u]+t*(f.use_ranges[2*u+1]-f.use_ranges[2*u]);
      double xyz[3], uv[2], on_face[3];
      const bool curve_ok=r.geometry.jets(f.curves[curve],&source,0,xyz,workspace);
      const bool pc_ok=r.geometry.jets(f.pcurves[u],&use,0,uv,workspace);
      PHX_CHECK(curve_ok && pc_ok);
      if (!curve_ok || !pc_ok) continue;
      const bool face_ok=r.geometry.jets(f.surfaces[f.loop_faces[u]],uv,0,on_face,workspace);
      PHX_CHECK(face_ok);
      if (!face_ok) continue;
      for (int k=0;k<3;++k) PHX_CHECK_NEAR(xyz[k],on_face[k],2e-12);
    }
  }
}

// Independently form physical tangent vectors for the UV chord. The test
// fixtures supply their analytic constant principal curvatures.
void check_face_metrics(const char* name, const Fixture& f, const Run& r,
                        const std::vector<double>& sizes, const std::vector<double>& tolerances,
                        const std::vector<double>& curvature, double theta=0.5) {
  EvaluationWorkspace workspace;
  double maximum=0;
  for (std::size_t u=0;u<f.entities.size();++u) {
    const int face=f.loop_faces[u], curve=f.entities[u];
    double h=sizes[face];
    if (curvature[face]>0)
      h=std::min(h,std::sqrt(6*theta*tolerances[face]/curvature[face]));
    const auto map=[&](double s) {
      const double a=f.use_ranges[2*u], b=f.use_ranges[2*u+1];
      return s==0 ? a : s==1 ? b : a+s*(b-a);
    };
    const auto begin=r.result.offsets[curve], end=r.result.offsets[curve+1];
    for (auto i=begin+1;i<end;++i) {
      const double a=r.result.parameters[i-1], b=r.result.parameters[i];
      const double interior=map(a+(b-a)*0.5);
      double uv[3][2], jet[9];
      const bool left=r.geometry.jets_in_span(f.pcurves[u],map(a),interior,0,uv[0],workspace);
      const bool right=r.geometry.jets_in_span(f.pcurves[u],map(b),interior,0,uv[2],workspace);
      PHX_CHECK(left && right);
      if (!left || !right) continue;
      for (int k=0;k<2;++k) uv[1][k]=(uv[0][k]+uv[2][k])*0.5;
      const double du=uv[2][0]-uv[0][0], dv=uv[2][1]-uv[0][1];
      for (const auto& q:uv) {
        const bool ok=r.geometry.jets(f.surfaces[face],q,1,jet,workspace);
        PHX_CHECK(ok);
        if (!ok) continue;
        const double x=du*jet[3]+dv*jet[6], y=du*jet[4]+dv*jet[7], z=du*jet[5]+dv*jet[8];
        const double metric=std::hypot(x,y,z)/h;
        maximum=std::max(maximum,metric);
        PHX_CHECK(metric<=1+1e-12);
      }
    }
  }
  std::printf("%s maximum boundary metric %.17g\n",name,maximum);
}

// The same rational quarter arc belongs to two faces, including a NURBS surface;
// catches separate per-use sampling, reversed use ranges and seam duplication.
void shared_nurbs_and_seam() {
  Fixture f; const double w = std::sqrt(0.5);
  auto p = plane(f);
  auto surf = f.node(PHX_MC_CAD_NODE_BSPLINE_SURFACE, {2,1,3,2},
      {0,0,0,1,1,1, 0,0,1,1, 1,0,0, 1,0,1, 1,1,0, 1,1,1, 0,1,0, 0,1,1, 1,1,w,w,1,1});
  auto arc = f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE, {2,3,3},
      {0,0,0,1,1,1, 1,0,0, 1,1,0, 0,1,0, 1,w,1});
  auto plane_arc=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE,{2,3,2},
      {0,0,0,1,1,1, 1,0, 1,1, 0,1, 1,w,1});
  auto reverse=f.templ("line2",{1,0,-1,0});
  f.face(p); f.face(surf); f.curve(arc,0,1);
  f.use(0,0,plane_arc,0,1); f.use(1,0,reverse,1,0,-1);
  Run r(f, {0.18,0.13}, {0.01,0.01}, {0.01});
  PHX_CHECK(r.segments() >= 8);
  sag(r, [w](double t) -> Vec {
    const double a=(1-t)*(1-t), b=2*w*t*(1-t), c=t*t, d=a+b+c;
    return {(a+b)/d,(b+c)/d,0};
  }, 0.005);
  Run repeat(f, {0.18,0.13}, {0.01,0.01}, {0.01});
  PHX_CHECK(r.result.offsets == repeat.result.offsets);
  PHX_CHECK(r.result.parameters.size() == repeat.result.parameters.size());
  if (r.result.parameters.size() == repeat.result.parameters.size())
    PHX_CHECK(std::memcmp(r.result.parameters.data(), repeat.result.parameters.data(),
                          r.result.parameters.size()*sizeof(double)) == 0);
  check_uses(f,r);
  check_face_metrics("shared-nurbs",f,r,{0.18,0.13},{0.01,0.01},{0,1});
}

// A real cylindrical seam has two distinct UV copies but one 3D edge.
// This catches applying the same UV copy to both uses or changing the source slice.
void cylindrical_seam() {
  Fixture f;
  auto cylinder=f.templ("cylinder",{0,0,0,1,0,0,0,1,0,0,0,1,1});
  auto edge=f.templ("line3",{1,0,0,0,0,1});
  auto left=f.templ("line2",{0,0,0,1});
  auto right=f.templ("line2",{2*pi,0,0,1});
  f.face(cylinder); f.periods[0]=2*pi;
  f.curve(edge,0,1); f.use(0,0,left,0,1); f.use(0,0,right,0,1,-1);
  Run r(f,{0.2},{0.1},{0.1});
  PHX_CHECK(r.result.statistics.small_feature_resamples==0);
  PHX_CHECK(r.segments()>=5);
  check_uses(f,r);
  check_face_metrics("cylindrical-seam",f,r,{0.2},{0.1},{1});
}

// Large UV separation can be a small physical gap, and conversely. This
// catches measuring proximity in raw UV instead of the first fundamental form.
void anisotropic_small_features() {
  for (int close=0;close<2;++close) {
    Fixture f;
    const double scale=close ? 0.01 : 10.0;
    const double uv_gap=close ? 5.0 : 0.1;
    auto surface=f.templ("plane",{0,0,0,1,0,0,0,scale,0});
    auto first=f.templ("line3",{0,0,0,2,0,0});
    auto second=f.templ("line3",{0,scale*uv_gap,0,2,0,0});
    auto pc0=f.templ("line2",{0,0,2,0});
    auto pc1=f.templ("line2",{0,uv_gap,2,0});
    f.face(surface); f.curve(first,0,1); f.curve(second,0,1);
    f.use(0,0,pc0,0,1); f.use(0,1,pc1,0,1);
    Run r(f,{0.5},{100},{100,100});
    check_uses(f,r);
    if (close) {
      PHX_CHECK(r.result.statistics.small_feature_resamples==2);
      PHX_CHECK(r.segments(0)>=39 && r.segments(1)>=39);
    } else {
      PHX_CHECK(r.result.statistics.small_feature_resamples==0);
      PHX_CHECK(r.segments(0)<=5 && r.segments(1)<=5);
    }
  }
}

// One U-shaped edge has close but nonadjacent spans; excluding an entire
// source curve rather than only adjacent segments misses this feature.
void same_curve_small_feature() {
  Fixture f; auto surface=plane(f);
  auto curve=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE,{1,4,3},
      {0,0,0.49,0.51,1,1, 0,0,0, 3,0,0, 3,0.1,0, 0,0.1,0, 1,1,1,1});
  auto pc=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE,{1,4,2},
      {0,0,0.49,0.51,1,1, 0,0, 3,0, 3,0.1, 0,0.1, 1,1,1,1});
  f.face(surface); f.curve(curve,0,1); f.use(0,0,pc,0,1);
  Run r(f,{0.5},{100},{100});
  PHX_CHECK(r.result.statistics.small_feature_resamples==1);
  PHX_CHECK(r.segments()>30);
  check_uses(f,r);
}

// A cubic inflection shared by a plane and a NURBS face catches both
// midpoint-only sizing and independently sampled uses.
void cubic_inflection() {
  Fixture f; auto s=plane(f);
  auto curve=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE, {3,4,3},
      {0,0,0,0,1,1,1,1, 0,0,0, 1.0/3,1,0, 2.0/3,-1,0, 1,0,0, 1,1,1,1});
  auto pc=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE, {3,4,2},
      {0,0,0,0,1,1,1,1, 0,0, 1.0/3,1, 2.0/3,-1, 1,0, 1,1,1,1});
  auto nurbs=f.node(PHX_MC_CAD_NODE_BSPLINE_SURFACE, {1,1,2,2},
      {0,0,1,1, 0,0,1,1, 0,-1,0, 0,1,0, 1,-1,0, 1,1,0, 1,1,1,1});
  auto npc=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE, {3,4,2},
      {0,0,0,0,1,1,1,1, 0,0.5, 1.0/3,1, 2.0/3,0, 1,0.5, 1,1,1,1});
  f.face(s); f.face(nurbs); f.curve(curve,0,1);
  f.use(0,0,pc,0,1); f.use(1,0,npc,0,1);
  Run r(f,{10,10},{0.004,0.004},{0.004});
  PHX_CHECK(r.segments()>2);
  sag(r, [](double t) -> Vec { return {t,3*t*(1-t)*(1-2*t),0}; }, 0.002);
  check_uses(f,r);
  check_face_metrics("cubic-inflection",f,r,{10,10},{0.004,0.004},{0,0});
}

// Interior knot is a mandatory break even when curvature and size are uniform.
void knot_breakpoint() {
  Fixture f; auto s=plane(f), pc=line2(f);
  auto curve=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE, {2,5,3},
      {0,0,0,0.25,0.25,1,1,1,
       0,0,0, 0.125,0,0, 0.25,0,0, 0.625,0,0, 1,0,0, 1,1,1,1,1});
  f.face(s); f.curve(curve,0,1); f.use(0,0,pc,0,1);
  Run r(f,{10},{100},{100});
  bool found=false; for (double t:r.result.parameters) found |= t==0.25;
  PHX_CHECK(found);
}

// Two span integrals of 1.1 each require four intervals, not ceil(2.2)=3.
void span_allocation() {
  Fixture f; auto surface=plane(f);
  auto curve=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE,{1,3,3},
      {0,0,0.5,1,1, 0,0,0, 1.1,0,0, 2.2,0,0, 1,1,1});
  auto pc=f.templ("line2",{0,0,2.2,0});
  f.face(surface); f.curve(curve,0,1); f.use(0,0,pc,0,1);
  Run r(f,{1},{100},{100});
  PHX_CHECK(r.segments()>=4);
  for (std::size_t i=1;i<r.result.parameters.size();++i)
    PHX_CHECK(2.2*(r.result.parameters[i]-r.result.parameters[i-1])<=1+1e-12);
}

// Detects losing an interior repeated knot when a composite reverses its parameter.
void graph_breakpoints() {
  Fixture f;
  auto curve = f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE, {2,5,3},
      {0,0,0,0.25,0.25,1,1,1,
       0,0,0, 0.2,0,0, 0.25,0,0, 0.5,0,0, 1,0,0, 1,1,1,1,1});
  auto reverse = f.node(PHX_MC_CAD_NODE_REPARAMETERIZED, {curve,1}, {0,1});
  CadGeometry geometry;
  PHX_CHECK(geometry.bind(f.model()) == PHX_MC_OK);
  std::vector<double> values;
  int degree = 0;
  geometry.curve_breakpoints(reverse, 0, 1, values, degree);
  std::sort(values.begin(), values.end());
  PHX_CHECK(degree == 2);
  PHX_CHECK(values.size() == 2);
  if (values.size() == 2) {
    PHX_CHECK(values[0] == 0.75);
    PHX_CHECK(values[1] == 0.75);
  }
}

// A face-curvature kink at a surface knot is not a curve breakpoint. Dense
// independent sag sampling catches relying on a smooth size integrand there.
void surface_knot_crossing() {
  int64_t evaluations[2] = {}, subdivisions[2] = {};
  for (int curved = 0; curved != 2; ++curved) {
    Fixture f;
    std::vector<double> data{0,0,0,0,0.5,0.5,1,1,1,1, 0,0,0,0,1,1,1,1};
    const double x[] = {0, 1.0/6, 1.0/3, 2.0/3, 5.0/6, 1};
    const double z[] = {0, 0, 0, 0, 1.0/12, 0.25};
    for (int i=0; i<6; ++i)
      for (int j=0; j<4; ++j)
        data.insert(data.end(), {x[i], j/3.0, curved ? z[i] : 0.0});
    data.insert(data.end(), 24, 1.0);
    const auto surface = f.node(PHX_MC_CAD_NODE_BSPLINE_SURFACE, {3,3,6,4}, data);
    const auto pc = f.templ("line2", {0,0.5,1,0});
    const auto curve = f.node(PHX_MC_CAD_NODE_SURFACE_MAPPED, {surface,pc}, {});
    f.face(surface); f.curve(curve,0,1); f.use(0,0,pc,0,1);
    Run r(f,{0.2},{0.02},{0.02});
    evaluations[curved] = r.result.statistics.second_order_evaluations;
    subdivisions[curved] = r.result.statistics.simpson_subdivisions;
    sag(r, [curved](double t) -> Vec {
      const double d=std::max(t-0.5,0.0);
      return {t,0.5,curved ? d*d : 0.0};
    }, 0.01);
  }
  std::printf("surface-knot evaluations: flat=%lld kink=%lld; Simpson subdivisions: flat=%lld kink=%lld\n",
              static_cast<long long>(evaluations[0]), static_cast<long long>(evaluations[1]),
              static_cast<long long>(subdivisions[0]), static_cast<long long>(subdivisions[1]));
}

// Simpson's common five samples all see zero second derivative in this degree-15
// polynomial. The degree floor and off-midpoint sag probes must expose its lobes.
void degree_fifteen_alias() {
  Fixture f; auto s=plane(f), pc=line2(f);
  std::vector<double> power{1};
  for (double root : {0.0,0.25,0.5,0.75,1.0}) for (int k=0;k<3;++k) {
    std::vector<double> next(power.size()+1,0);
    for (std::size_t j=0;j<power.size();++j) { next[j]-=root*power[j]; next[j+1]+=power[j]; }
    power=next;
  }
  const double scale=35184372088832.0/1157625.0;
  auto choose=[](int n,int k) { double v=1; for(int j=1;j<=k;++j) v=v*(n-j+1)/j; return v; };
  std::vector<double> data(16,0); data.insert(data.end(),16,1);
  for (int i=0;i<16;++i) {
    double y=0; for(int k=0;k<=i;++k) y+=scale*power[k]*choose(i,k)/choose(15,k);
    data.insert(data.end(), {i/15.0,y,0});
  }
  data.insert(data.end(),16,1);
  auto curve=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE,{15,16,3},data);
  f.face(s); f.curve(curve,0,1); f.use(0,0,pc,0,1);
  for (bool sag_check : {false, true}) {
    Run r(f,{2},{0.2},{0.2},0.5,1.3,sag_check);
    PHX_CHECK(r.segments()>2);
    sag(r, [scale](double t) -> Vec {
      double y=scale; for(double root:{0.0,0.25,0.5,0.75,1.0}) y*=std::pow(t-root,3);
      return {t,y,0};
    },0.1);
  }
}

// Detects truncating degree support or losing the parameter map of a composite.
void degree_thirty_one_composite() {
  Fixture f;
  auto surface=plane(f), pc=line2(f);
  std::vector<double> data(32,0);
  data.insert(data.end(),32,1);
  for (int i=0;i<32;++i) data.insert(data.end(), {i/31.0,0,0});
  data.insert(data.end(),32,1);
  auto spline=f.node(PHX_MC_CAD_NODE_BSPLINE_CURVE,{31,32,3},data);
  auto reversed=f.node(PHX_MC_CAD_NODE_REPARAMETERIZED,{spline,1},{0,1});
  auto restored=f.node(PHX_MC_CAD_NODE_REPARAMETERIZED,{reversed,1},{0,1});
  f.face(surface); f.curve(restored,0,1); f.use(0,0,pc,0,1);
  Run r(f,{0.2},{100},{100});
  PHX_CHECK(std::abs(r.segments()-5)<=1);
  sag(r, [](double t) -> Vec { return {t,0,0}; }, 0.01);
}
// Detects an unsmoothed curvature size field through adjacent physical steps.
void gradation() {
  Fixture f; auto s=plane(f), pc=line2(f);
  auto e=f.templ("ellipse3", {0,0,0, 1,0,0, 0,1,0, 4,0.5});
  f.face(s); f.curve(e,0,pi/2); f.use(0,0,pc,0,pi/2);
  Run r(f,{2},{0.005},{0.005},0.5,1.3);
  const auto point=[](double t) -> Vec { return {4*std::cos(t*pi/2),0.5*std::sin(t*pi/2),0}; };
  const auto& p=r.result.parameters;
  for (std::size_t i=2;i<p.size();++i) {
    const double a=dist(point(p[i-2]),point(p[i-1]));
    const double b=dist(point(p[i-1]),point(p[i]));
    PHX_CHECK(std::abs(a-b) <= 0.15*(a+b)+1e-8);
  }
}

// Detects capacity checks after writes and C-entry divergence from the C++ API.
void c_entry() {
  Fixture f; auto s=plane(f), pc=line2(f);
  auto line=f.templ("line3",{0,0,0,2,0,0});
  f.face(s); f.curve(line,2,5); f.use(0,0,pc,2,5);
  auto m=f.model();
  const double sizes[]={0.7}, tolerances[]={100};
  phx_mc_cad_curve_controls c{sizes,tolerances,tolerances,0.5,1.3,1e-8};
  int64_t offsets[2]={-7,-7}; double parameters[32];
  std::fill(parameters,parameters+32,-7.0);
  phx_mc_cad_curve_failure failures[2]{{-7,-7,-7},{-7,-7,-7}};
  int64_t splits[1]={-7};
  phx_mc_cad_curve_statistics stats{-7,-7,-7,-7,-7,-7,-7};
  auto unchanged=[&]() {
    PHX_CHECK(offsets[0]==-7 && offsets[1]==-7);
    for (double x:parameters) PHX_CHECK(x==-7);
    PHX_CHECK(failures[0].curve==-7 && failures[1].curve==-7);
    PHX_CHECK(splits[0]==-7);
    PHX_CHECK(stats.parameter_count==-7 && stats.unresolved_count==-7);
  };
  PHX_CHECK(phx_mc_cad_curves_test(nullptr,&c,2,offsets,32,parameters,2,failures,1,splits,&stats)==PHX_MC_INVALID_ARGUMENT);
  unchanged();
  PHX_CHECK(phx_mc_cad_curves_test(&m,&c,1,offsets,32,parameters,2,failures,1,splits,&stats)==PHX_MC_CAPACITY_EXCEEDED);
  unchanged();
  PHX_CHECK(phx_mc_cad_curves_test(&m,&c,2,offsets,1,parameters,2,failures,1,splits,&stats)==PHX_MC_CAPACITY_EXCEEDED);
  unchanged();
  PHX_CHECK(phx_mc_cad_curves_test(&m,&c,2,offsets,32,parameters,2,failures,0,splits,&stats)==PHX_MC_CAPACITY_EXCEEDED);
  unchanged();
  PHX_CHECK(phx_mc_cad_curves_test(&m,&c,2,offsets,32,parameters,2,failures,1,splits,&stats)==PHX_MC_OK);
  const int64_t count=offsets[1]-offsets[0]-1;
  const int expected=static_cast<int>(std::ceil(2.0*(5.0-2.0)/0.7));
  PHX_CHECK(offsets[0]==0 && std::abs(count-expected)<=1);
  PHX_CHECK(stats.parameter_count==offsets[1] && stats.unresolved_count==0);
  PHX_CHECK(splits[0]==stats.sag_splits && splits[0]==0);
  PHX_CHECK(stats.first_order_evaluations+stats.second_order_evaluations>0);
  if (count>0 && count<32)
    for(int64_t i=0;i<=count;++i)
      PHX_CHECK_NEAR(parameters[i],static_cast<double>(i)/count,1e-10);
}


// Independent closed-form branch of orthogonal cylinders; catches using a
// surrogate curve or skipping the implicit intersection's derivative jets.
void intersection_integral() {
  Fixture f;
  auto a=f.templ("cylinder",{0,0,0,1,0,0,0,1,0,0,0,1,1});
  auto b=f.templ("cylinder",{0,0,0,0,1,0,0,0,1,1,0,0,0.6});
  auto coord=[](double t) { double y=0.6*std::cos(t); return std::array<double,4>{std::asin(y),0.6*std::sin(t),t,std::sqrt(1-y*y)}; };
  const double lo=-0.5, hi=0.7; auto start=coord(lo), end=coord(hi), mid=coord((lo+hi)/2);
  double c=std::cos(mid[0]), s=std::sin(mid[0]);
  std::vector<double> data{0,0}; data.insert(data.end(),start.begin(),start.end()); data.insert(data.end(),end.begin(),end.end());
  data.insert(data.end(),{-1,-1,lo,0.5,1,1,hi,1, 0,1/c,0,0,0,1,-1,-s/c,0});
  auto curve=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,1,0,-1,-1,0,2},data);
  auto pc=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,1,1,0,-1,0,2},data);
  f.face(a); f.curve(curve,0,1); f.use(0,0,pc,0,1);
  Run r(f,{0.12},{100},{100});
  const double length=integrate([](double t) {
    double y=0.6*std::cos(t), yp=-0.6*std::sin(t), zp=0.6*std::cos(t);
    return std::sqrt(yp*yp/(1-y*y)+zp*zp);
  },lo,hi);
  PHX_CHECK(std::abs(r.segments()-static_cast<int>(std::ceil(length/0.12)))<=1);
}

// The full closed branch of the quadric fixture, checked against independent
// quadrature, catches under-allocating the integral of an individual chart.
void quadric_closed_integral() {
  Fixture f;
  auto a=f.templ("cylinder",{0,0,0,1,0,0,0,1,0,0,0,1,1});
  auto b=f.templ("cylinder",{0,0,0,0,1,0,0,0,1,1,0,0,0.6});
  auto coord=[](double t) {
    const double y=0.6*std::cos(t);
    return std::array<double,4>{std::asin(y),0.6*std::sin(t),t,std::sqrt(1-y*y)};
  };
  std::vector<double> data{0,0};
  for (int i=0;i<4;++i) {
    const double lo=i*pi/2, hi=(i+1)*pi/2;
    auto start=coord(lo), end=coord(hi), mid=coord((lo+hi)/2);
    const double c=std::cos(mid[0]), s=std::sin(mid[0]);
    data.insert(data.end(),start.begin(),start.end());
    data.insert(data.end(),end.begin(),end.end());
    data.insert(data.end(),{-1,-1,lo,0.5,1,1,hi,1, 0,1/c,0,0,0,1,-1,-s/c,0});
  }
  auto curve=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,4,0,-1,-1,0,2,2,2,2},data);
  auto pc=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,4,1,-1,-1,0,2,2,2,2},data);
  auto pc2=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,4,2,-1,-1,0,2,2,2,2},data);
  f.face(a); f.face(b); f.curve(curve,0,4,true);
  f.use(0,0,pc,0,4); f.use(1,0,pc2,0,4);
  Run r(f,{0.229,0.229},{100,100},{100});
  const double length=integrate([](double t) {
    const double s=std::sin(t), c=std::cos(t);
    const double dx=-0.36*s*c/std::sqrt(1-0.36*s*s);
    return std::sqrt(dx*dx+0.36);
  },0,pi/2);
  PHX_CHECK(r.segments()==4*static_cast<int>(std::ceil(length/0.229)));
  Run fine(f,{0.12,0.12},{100,100},{100});
  check_face_metrics("quadric-h012",f,fine,{0.12,0.12},{100,100},{1,1/0.6});
  const auto& params=fine.result.parameters;
  for (std::size_t i=1;i<params.size();++i) {
    const auto uv=[](double s) {
      const double t=2*pi*s;
      return std::array<double,2>{std::asin(0.6*std::cos(t)),0.6*std::sin(t)};
    };
    const auto x=uv(params[i-1]), y=uv(params[i]);
    PHX_CHECK(std::hypot(y[0]-x[0],y[1]-x[1])<=0.12+1e-12);
    const double ta=2*pi*params[i-1], tb=2*pi*params[i];
    const double za=std::sqrt(1-0.36*std::cos(ta)*std::cos(ta));
    const double zb=std::sqrt(1-0.36*std::cos(tb)*std::cos(tb));
    PHX_CHECK(std::hypot(0.6*(tb-ta),zb-za)<=0.12+1e-12);
  }
}

// A chart sheet change is an exact graph breakpoint; omitting it mixes gauges.
void intersection_chart_shift() {
  Fixture f;
  auto a=f.templ("cylinder",{0,0,0,1,0,0,0,1,0,0,0,1,1});
  auto b=f.templ("cylinder",{0,0,0,0,1,0,0,0,1,1,0,0,0.6});
  auto coord=[](double t, double shift) {
    double y=0.6*std::cos(t);
    return std::array<double,4>{std::asin(y)+shift,0.6*std::sin(t),t,std::sqrt(1-y*y)};
  };
  std::vector<double> data{0,0};
  for (int i=0;i<2;++i) {
    const double lo=-0.5+0.6*i, hi=lo+0.6, shift=i*2*pi;
    auto start=coord(lo,shift), end=coord(hi,shift), mid=coord((lo+hi)/2,shift);
    const double c=std::cos(mid[0]), s=std::sin(mid[0]);
    data.insert(data.end(),start.begin(),start.end());
    data.insert(data.end(),end.begin(),end.end());
    data.insert(data.end(),{-1+shift,-1,lo,0.5,1+shift,1,hi,1,
                            0,1/c,0,0,0,1,-1,-s/c,0});
  }
  auto curve=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,2,0,-1,-1,0,2,2},data);
  auto pc=f.node(PHX_MC_CAD_NODE_INTERSECTION,{a,b,2,1,-1,-1,0,2,2},data);
  auto wrapped=f.node(PHX_MC_CAD_NODE_AFFINE,{pc,2},{1,0,0,1,0,0});
  f.face(a); f.curve(curve,0,2); f.use(0,0,wrapped,0,2);
  Run r(f,{0.2},{100},{100});
  PHX_CHECK(std::find(r.result.parameters.begin(),r.result.parameters.end(),0.5)
            != r.result.parameters.end());
  EvaluationWorkspace workspace;
  double before[2], after[2];
  const double left=std::nextafter(1.0,0.0), right=1.0;
  PHX_CHECK(r.geometry.jets(pc,&left,0,before,workspace));
  PHX_CHECK(r.geometry.jets(pc,&right,0,after,workspace));
  PHX_CHECK_NEAR(after[0]-before[0],2*pi,1e-12);
  double left_limit[2], right_limit[2];
  PHX_CHECK(r.geometry.jets_in_span(wrapped,1.0,0.5,0,left_limit,workspace));
  PHX_CHECK(r.geometry.jets_in_span(wrapped,1.0,1.5,0,right_limit,workspace));
  PHX_CHECK(!r.geometry.jets_in_span(wrapped,1.0,1.0,0,before,workspace));
  const auto expected=coord(0.1,0);
  PHX_CHECK_NEAR(left_limit[0],expected[0],1e-12);
  PHX_CHECK_NEAR(left_limit[1],expected[1],1e-12);
  PHX_CHECK_NEAR(right_limit[0]-left_limit[0],2*pi,1e-12);
  PHX_CHECK_NEAR(right_limit[1],left_limit[1],1e-12);
}
}  // namespace

int main() {
  line_integral(); circle_integral(); ellipse_integral(); shared_nurbs_and_seam(); cylindrical_seam();
  cubic_inflection(); knot_breakpoint(); span_allocation(); graph_breakpoints(); degree_fifteen_alias(); intersection_integral();
  surface_knot_crossing(); degree_thirty_one_composite(); intersection_chart_shift(); quadric_closed_integral();
  anisotropic_small_features(); same_curve_small_feature(); gradation(); c_entry();
  return phx::mc::test::finish("test_cad_curves");
}
