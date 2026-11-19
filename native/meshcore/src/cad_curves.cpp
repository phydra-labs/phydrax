// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_curves.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace phx::mc::cad {
namespace {

constexpr int kDepth = 32;
constexpr int kSagDepth = 12;
constexpr std::size_t kMaxPoints = 1000000;

struct Use { int32_t row, face; };
struct Sample { double s, speed, h; };

struct Context {
  const CadGeometry& geometry;
  const CurveControls& controls;
  CurveStatistics& stats;
  int32_t curve;
  const std::vector<Use>& uses;
  EvaluationWorkspace workspace{};
  double feature_size = std::numeric_limits<double>::infinity();
  bool failed = false;
  int32_t failure_status = PHX_MC_UNRESOLVED;

  double parameter(double s) const {
    const double* r = geometry.model().curve_ranges + 2 * curve;
    if (s == 0) return r[0];
    if (s == 1) return r[1];
    return r[0] + s * (r[1] - r[0]);
  }
  double use_parameter(int32_t use, double s) const {
    const double* r = geometry.model().use_ranges + 2 * use;
    if (s == 0) return r[0];
    if (s == 1) return r[1];
    return r[0] + s * (r[1] - r[0]);
  }
  bool value(double s, double* xyz) {
    const double t = parameter(s);
    ++stats.sag_evaluations;
    ++stats.value_evaluations;
    return geometry.jets(geometry.curve_node(curve), &t, 0, xyz, workspace);
  }
  Sample sample(double s) {
    const auto& m = geometry.model();
    const double t = parameter(s);
    double jet[9], speed, curvature;
    ++stats.second_order_evaluations;
    if (!geometry.jets(geometry.curve_node(curve), &t, 2, jet, workspace)) {
      failure_status=PHX_MC_INVALID_INPUT;
      failed=true; return {s,0,0};
    }
    if (!curve_differential(jet, &speed, &curvature) || !std::isfinite(curvature)) {
      failed = true; return {s, 0, 0};
    }
    speed *= std::abs(m.curve_ranges[2 * curve + 1] - m.curve_ranges[2 * curve]);
    double h = feature_size;
    const double theta = controls.tolerance_fraction;
    const double ct = controls.curve_tolerances[curve];
    if (curvature > 0 && std::isfinite(ct)) h = std::min(h, std::sqrt(8 * theta * ct / curvature));
    for (const Use& use : uses) {
      double kn=0;
      bool resolved=false, evaluated=false;
      for (int attempt=0; attempt<9 && !resolved; ++attempt) {
        if (attempt && s!=0 && s!=1) break;
        const double delta=std::ldexp(std::sqrt(std::numeric_limits<double>::epsilon()),attempt-1);
        const double sample_s=attempt ? (s==0 ? delta : 1-delta) : s;
        const double u=use_parameter(use.row,sample_s);
        double pc[4], sj[18];
        ++stats.first_order_evaluations;
        if (!geometry.jets(geometry.pcurve_node(use.row), &u, 1, pc, workspace)) continue;
        ++stats.second_order_evaluations;
        if (!geometry.jets(geometry.surface_node(use.face), pc, 2, sj, workspace)) continue;
        evaluated=true;
        const double* a=sj+3; const double* b=sj+6;
        double n[3]={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
        const double nl=std::hypot(n[0],n[1],n[2]);
        if (!(nl>0) || !std::isfinite(nl)) continue;
        for (double& x:n) x/=nl;
        double tangent[3],second[3];
        for (int k=0;k<3;++k) {
          tangent[k]=a[k]*pc[2]+b[k]*pc[3];
          second[k]=sj[9+k]*pc[2]*pc[2]+2*sj[12+k]*pc[2]*pc[3]+sj[15+k]*pc[3]*pc[3];
        }
        const double v2=tangent[0]*tangent[0]+tangent[1]*tangent[1]+tangent[2]*tangent[2];
        if (!(v2>0) || !std::isfinite(v2)) continue;
        kn=std::abs(n[0]*second[0]+n[1]*second[1]+n[2]*second[2])/v2;
        resolved=std::isfinite(kn);
      }
      if (!resolved) {
        if (!evaluated) failure_status=PHX_MC_INVALID_INPUT;
        failed=true; return {s,0,0};
      }
      h = std::min(h, controls.face_sizes[use.face]);
      if (kn > 0 && std::isfinite(controls.face_tolerances[use.face]))
        h = std::min(h, std::sqrt(6*theta*controls.face_tolerances[use.face]/kn));
    }
    if (!(h > 0) || !std::isfinite(h) || !(speed > 0) || !std::isfinite(speed)) failed = true;
    return {s, speed, h};
  }
};

double density(const Sample& x) { return x.speed / x.h; }
double simpson(const Sample& a, const Sample& b, const Sample& c) {
  return (c.s-a.s)*(density(a)+4*density(b)+density(c))/6;
}

void subdivide(Context& c, const Sample& a, const Sample& b, const Sample& d,
               double whole, int depth, std::vector<Sample>& table) {
  const double left = (a.s+b.s)*0.5, right = (b.s+d.s)*0.5;
  if (!(left > a.s && right < d.s)) { c.failed = true; return; }
  const Sample l = c.sample(left), r = c.sample(right);
  if (c.failed) return;
  const double x = simpson(a,l,b), y = simpson(b,r,d);
  // Quadrature convergence alone does not control interpolation of its inverse:
  // even a linear density integrates exactly but changes within a table cell.
  const double f[5]={density(a),density(l),density(b),density(r),density(d)};
  bool inverse_resolved=true;
  for (int i=1;i<5;++i)
    if (std::abs(f[i]-f[i-1])>0.1*std::max(f[i],f[i-1]) &&
        (d.s-a.s)*std::abs(f[i]-f[i-1])>1e-8) inverse_resolved=false;
  if (inverse_resolved && std::abs(x+y-whole) <= 1e-8 + 0.001*std::abs(x+y)) {
    table.push_back(l); table.push_back(b); table.push_back(r); table.push_back(d);
  } else if (depth == kDepth || table.size() >= kMaxPoints) {
    c.failed = true;
  } else {
    ++c.stats.simpson_subdivisions;
    subdivide(c,a,l,b,x,depth+1,table);
    if (!c.failed) subdivide(c,b,r,d,y,depth+1,table);
  }
}

bool discretize(Context& c, std::vector<double>& points) {
  const auto& m = c.geometry.model();
  std::vector<double> fixed{0,1}; int degree = 2;
  const double* cr = m.curve_ranges + 2*c.curve;
  c.geometry.curve_breakpoints(c.geometry.curve_node(c.curve), cr[0], cr[1], fixed, degree);
  for (const Use& u : c.uses) {
    const double* r = m.use_ranges + 2*u.row;
    c.geometry.curve_breakpoints(c.geometry.pcurve_node(u.row), r[0], r[1], fixed, degree);
  }
  std::sort(fixed.begin(), fixed.end()); fixed.erase(std::unique(fixed.begin(),fixed.end()),fixed.end());
  std::vector<Sample> table;
  std::vector<std::size_t> begins, borders;
  for (std::size_t j=1; j<fixed.size(); ++j) {
    // A degree-d polynomial's second derivative has degree d-2 and cannot
    // vanish at d-1 distinct samples unless it is identically zero. Rational
    // derivatives are not polynomials, so that argument does not carry over.
    const int panels = std::max(1, (degree-1+3)/4);
    for (int p=0; p<panels; ++p) {
      const double a = fixed[j-1]+(fixed[j]-fixed[j-1])*p/panels;
      const double b = fixed[j-1]+(fixed[j]-fixed[j-1])*(p+1)/panels;
      const double shift=std::min((b-a)*0.25,8*std::numeric_limits<double>::epsilon());
      const double left=(p==0 && j>1) ? a+shift : a;
      const double right=(p+1==panels && j+1<fixed.size()) ? b-shift : b;
      if (!(left<b && right>a)) return false;
      Sample x = c.sample(left), y = c.sample((a+b)*0.5), z = c.sample(right);
      x.s=a; z.s=b;
      if (c.failed) return false;
      if (p==0) { begins.push_back(table.size()); table.push_back(x); }
      subdivide(c,x,y,z,simpson(x,y,z),0,table);
      if (c.failed) return false;
    }
    borders.push_back(table.size()-1);
  }
  // Equal-integral steps in a linearly graded field can differ by exp(slope);
  // use log(g), which is also no larger than the requested Lipschitz g-1.
  const double slope = std::log(c.controls.grading);
  for (std::size_t i=1; i<table.size(); ++i) {
    const double ds = (table[i].s-table[i-1].s)*(table[i].speed+table[i-1].speed)*0.5;
    table[i].h = std::min(table[i].h,table[i-1].h+slope*ds);
  }
  for (std::size_t i=table.size()-1; i>0; --i) {
    const double ds = (table[i].s-table[i-1].s)*(table[i].speed+table[i-1].speed)*0.5;
    table[i-1].h = std::min(table[i-1].h,table[i].h+slope*ds);
  }
  struct Span {
    std::vector<double> cumulative;
    double quota = 0;
    int count = 1;
  };
  std::vector<Span> spans(borders.size());
  double total=0;
  for (std::size_t span=0; span<borders.size(); ++span) {
    const std::size_t start=begins[span], end=borders[span];
    auto& cumulative=spans[span].cumulative;
    cumulative.resize(end-start+1,0);
    for (std::size_t i=start+2; i<=end; i+=2) {
      const double left=(density(table[i-2])+density(table[i-1]))*0.5;
      const double right=(density(table[i-1])+density(table[i]))*0.5;
      const double area=simpson(table[i-2],table[i-1],table[i]);
      cumulative[i-start-1]=cumulative[i-start-2]+area*left/(left+right);
      cumulative[i-start]=cumulative[i-start-2]+area;
    }
    if (!(cumulative.back()>0) || !std::isfinite(cumulative.back())) return false;
    total+=cumulative.back();
  }
  if (!std::isfinite(total) || total>kMaxPoints) return false;
  int count=std::max(static_cast<int>(spans.size()),static_cast<int>(std::ceil(total)));
  if (m.curve_corners[2*c.curve]==m.curve_corners[2*c.curve+1]) count=std::max(count,3);
  int assigned=0;
  for (auto& span:spans) {
    span.quota=count*span.cumulative.back()/total;
    span.count=std::max(1,static_cast<int>(std::floor(span.quota)));
    assigned+=span.count;
  }
  while (assigned<count) {
    auto best=std::max_element(spans.begin(),spans.end(),[](const Span& a,const Span& b) {
      return a.quota-a.count < b.quota-b.count;
    });
    ++best->count; ++assigned;
  }
  while (assigned>count) {
    auto best=spans.end();
    for (auto it=spans.begin();it!=spans.end();++it)
      if (it->count>1 && (best==spans.end() ||
          it->count-it->quota > best->count-best->quota)) best=it;
    if (best==spans.end()) return false;
    --best->count; --assigned;
  }
  points.clear(); points.reserve(static_cast<std::size_t>(count)+1); points.push_back(0);
  for (std::size_t span=0; span<spans.size(); ++span) {
    const std::size_t start=begins[span], end=borders[span];
    const int n=spans[span].count;
    std::vector<double> parameters(end-start+1), placed(n+1);
    for (std::size_t i=start; i<=end; ++i) parameters[i-start]=table[i].s;
    if (inverse_curve_table(parameters.data(),spans[span].cumulative.data(),
                            static_cast<int64_t>(parameters.size()),n,placed.data()) != PHX_MC_OK)
      return false;
    points.insert(points.end(),placed.begin()+1,placed.end());
  }
  return true;
}
struct ChartSegment {
  int32_t curve, face, use, first_corner, last_corner;
  std::size_t index;
  double a[2], b[2];
};

bool chart_point(const CadGeometry& g, EvaluationWorkspace& workspace, int32_t use,
                 double s, double* uv) {
  const double* range = g.model().use_ranges + 2*use;
  const double t = s == 0 ? range[0] : s == 1 ? range[1] : range[0]+s*(range[1]-range[0]);
  return g.jets(g.pcurve_node(use), &t, 0, uv, workspace);
}

bool feature_sizes(const CadGeometry& g, const CurveControls& controls,
                   const std::vector<std::vector<double>>& points,
                   const std::vector<std::vector<Use>>& uses, CurveStatistics& stats,
                   std::vector<double>& limits) {
  const auto& m = g.model();
  EvaluationWorkspace workspace;
  std::vector<ChartSegment> segments;
  std::vector<std::vector<double>> arc(static_cast<std::size_t>(m.use_count));
  std::vector<std::vector<double>> charts(static_cast<std::size_t>(m.use_count));
  std::vector<std::vector<double>> metrics(static_cast<std::size_t>(m.use_count));
  for (int32_t e=0; e<m.curve_count; ++e) for (const Use& use : uses[e]) {
    double a[2], b[2], previous[3], next[3], jet[9];
    auto& metric=metrics[use.row];
    metric.reserve(3*points[e].size());
    auto evaluate_surface = [&](const double* uv, double* position) {
      ++stats.first_order_evaluations;
      if (!g.jets(g.surface_node(use.face),uv,1,jet,workspace)) return false;
      for (int k=0;k<3;++k) position[k]=jet[k];
      const double* u=jet+3; const double* v=jet+6;
      metric.insert(metric.end(), {u[0]*u[0]+u[1]*u[1]+u[2]*u[2],
                                  u[0]*v[0]+u[1]*v[1]+u[2]*v[2],
                                  v[0]*v[0]+v[1]*v[1]+v[2]*v[2]});
      return true;
    };
    if (!chart_point(g,workspace,use.row,points[e][0],a) ||
        !evaluate_surface(a,previous)) return false;
    auto& distance=arc[use.row];
    distance.reserve(points[e].size()); distance.push_back(0);
    auto& chart=charts[use.row];
    chart.reserve(2*points[e].size());
    chart.insert(chart.end(),{a[0],a[1]});
    stats.value_evaluations+=static_cast<int64_t>(points[e].size());
    for (std::size_t i=1; i<points[e].size(); ++i) {
      if (!chart_point(g,workspace,use.row,points[e][i],b) ||
          !evaluate_surface(b,next)) return false;
      chart.insert(chart.end(),{b[0],b[1]});
      distance.push_back(distance.back()+std::hypot(next[0]-previous[0],next[1]-previous[1],next[2]-previous[2]));
      for (int k=0;k<3;++k) previous[k]=next[k];
      segments.push_back({e,use.face,use.row,m.curve_corners[2*e],m.curve_corners[2*e+1],i-1,
                          {a[0],a[1]},{b[0],b[1]}});
      a[0]=b[0]; a[1]=b[1];
    }
  }
  struct Grid {
    double lower[2], width[2];
    int side;
    std::vector<std::vector<std::size_t>> cells;
    int cell(double x, int axis) const {
      const double t=(x-lower[axis])/width[axis];
      if (t<=0) return 0;
      if (t>=side) return side-1;
      return static_cast<int>(t);
    }
  };
  std::vector<Grid> grids(static_cast<std::size_t>(m.surface_count));
  for (int32_t face=0; face<m.surface_count; ++face) {
    Grid& grid=grids[face];
    grid.lower[0]=grid.lower[1]=std::numeric_limits<double>::infinity();
    double upper[2]={-grid.lower[0],-grid.lower[1]};
    std::size_t count=0;
    for (const ChartSegment& s:segments) if (s.face==face) {
      ++count;
      for (int k=0;k<2;++k) {
        grid.lower[k]=std::min(grid.lower[k],std::min(s.a[k],s.b[k]));
        upper[k]=std::max(upper[k],std::max(s.a[k],s.b[k]));
      }
    }
    if (!count) continue;
    grid.side=std::max(1,std::min(512,static_cast<int>(std::sqrt(static_cast<double>(count)))));
    for (int k=0;k<2;++k) {
      grid.width[k]=(upper[k]-grid.lower[k])/grid.side;
      if (!(grid.width[k]>0) || !std::isfinite(grid.width[k])) grid.width[k]=1;
    }
    grid.cells.resize(static_cast<std::size_t>(grid.side)*grid.side);
    for (std::size_t n=0;n<segments.size();++n) {
      const ChartSegment& s=segments[n];
      if (s.face!=face) continue;
      const int x0=grid.cell(std::min(s.a[0],s.b[0]),0), x1=grid.cell(std::max(s.a[0],s.b[0]),0);
      const int y0=grid.cell(std::min(s.a[1],s.b[1]),1), y1=grid.cell(std::max(s.a[1],s.b[1]),1);
      for (int x=x0;x<=x1;++x) for (int y=y0;y<=y1;++y)
        grid.cells[static_cast<std::size_t>(x)*grid.side+y].push_back(n);
    }
  }
  for (int32_t e=0; e<m.curve_count; ++e) for (const Use& use : uses[e]) {
    const Grid& grid=grids[use.face];
    for (std::size_t i=0; i<points[e].size(); ++i) {
      const double* uv=charts[use.row].data()+2*i;
      const double* metric=metrics[use.row].data()+3*i;
      const double E=metric[0], F=metric[1], G=metric[2];
      const double det=E*G-F*F;
      if (!std::isfinite(det) || !std::isfinite(E) || !std::isfinite(G)) return false;
      const double rx=det>0 ? controls.face_sizes[use.face]*std::sqrt(G/det) :
                              std::numeric_limits<double>::infinity();
      const double ry=det>0 ? controls.face_sizes[use.face]*std::sqrt(E/det) :
                              std::numeric_limits<double>::infinity();
      const int x0=std::isfinite(rx) ? grid.cell(uv[0]-rx,0) : 0;
      const int x1=std::isfinite(rx) ? grid.cell(uv[0]+rx,0) : grid.side-1;
      const int y0=std::isfinite(ry) ? grid.cell(uv[1]-ry,1) : 0;
      const int y1=std::isfinite(ry) ? grid.cell(uv[1]+ry,1) : grid.side-1;
      for (int gx=x0;gx<=x1;++gx) for (int gy=y0;gy<=y1;++gy)
      for (std::size_t row:grid.cells[static_cast<std::size_t>(gx)*grid.side+gy]) {
        const ChartSegment& segment=segments[row];
        if (segment.curve==e) {
          const auto& length=arc[use.row];
          double separation= i<=segment.index ? length[segment.index]-length[i] :
                             (i>segment.index+1 ? length[i]-length[segment.index+1] : 0);
          if (m.curve_corners[2*e]==m.curve_corners[2*e+1])
            separation=std::min(separation,length.back()-separation);
          if (separation<=2*controls.face_sizes[use.face]) continue;
          if (segment.index+2>=i && segment.index<=i+1) continue;
          if (m.curve_corners[2*e]==m.curve_corners[2*e+1] &&
              ((i<=1 && segment.index+2>=points[e].size()-1) ||
               (i+2>=points[e].size() && segment.index<=1))) continue;
        }
        // Incident edges are locally adjacent through their shared corner,
        // not a small feature merely because the first samples meet there.
        if (segment.curve != e) {
          const auto& current_arc = arc[use.row];
          const auto& other_arc = arc[segment.use];
          bool adjacent = false;
          for (int endpoint=0; endpoint<2; ++endpoint) {
            const int32_t corner_id=m.curve_corners[2*e+endpoint];
            for (int other=0; other<2; ++other) {
              if (corner_id != m.curve_corners[2*segment.curve+other]) continue;
              const double a=endpoint ? current_arc.back()-current_arc[i] : current_arc[i];
              const double b=other ? other_arc.back()-other_arc[segment.index+1]
                                   : other_arc[segment.index];
              if (a+b <= 2*controls.face_sizes[use.face]) adjacent=true;
            }
          }
          if (adjacent) continue;
        }
        const int corner = i==0 ? m.curve_corners[2*e] :
                           (i+1==points[e].size() ? m.curve_corners[2*e+1] : -1);
        if (corner>=0 && (corner==segment.first_corner || corner==segment.last_corner)) continue;
        const double dx=segment.b[0]-segment.a[0], dy=segment.b[1]-segment.a[1];
        const double px=uv[0]-segment.a[0], py=uv[1]-segment.a[1];
        const double den=E*dx*dx+2*F*dx*dy+G*dy*dy;
        const double t=den>0 ? std::clamp((E*px*dx+F*(px*dy+py*dx)+G*py*dy)/den,0.0,1.0) : 0.0;
        const double x=px-t*dx, y=py-t*dy;
        const double distance=std::sqrt(std::max(0.0,E*x*x+2*F*x*y+G*y*y));
        if (distance<controls.face_sizes[use.face]) {
          const double h=std::max(distance,controls.minimum_size);
          limits[e]=std::min(limits[e],h);
          limits[segment.curve]=std::min(limits[segment.curve],h);
        }
      }
    }
  }
  return true;
}
bool sag(Context& c, double a, double b, const double* x, const double* y,
         int depth, std::vector<double>& out) {
  constexpr double q = 0.28867513459481288225;
  double p[3], r[3];
  const double s1=a+(b-a)*(0.5-q), s2=a+(b-a)*(0.5+q);
  if (!c.value(s1,p) || !c.value(s2,r)) return false;
  double tol=c.controls.curve_tolerances[c.curve];
  for (const Use& u:c.uses) tol=std::min(tol,c.controls.face_tolerances[u.face]);
  tol*=c.controls.tolerance_fraction;
  auto deviation=[&](const double* z) {
    const double dx=y[0]-x[0], dy=y[1]-x[1], dz=y[2]-x[2];
    const double px=z[0]-x[0], py=z[1]-x[1], pz=z[2]-x[2];
    const double length2=dx*dx+dy*dy+dz*dz;
    const double t=length2>0 ? std::clamp((px*dx+py*dy+pz*dz)/length2,0.0,1.0) : 0.0;
    return std::hypot(px-t*dx,py-t*dy,pz-t*dz);
  };
  if (deviation(p)<=tol && deviation(r)<=tol) { out.push_back(b); return true; }
  const double mid=(a+b)*0.5;
  if (depth==kSagDepth || !(mid>a && mid<b) || out.size()>=kMaxPoints) {
    c.stats.unresolved.push_back({c.curve,a,b}); out.push_back(b); return true;
  }
  ++c.stats.sag_splits;
  ++c.stats.sag_splits_per_curve[c.curve];
  double z[3]; if (!c.value(mid,z)) return false;
  return sag(c,a,mid,x,z,depth+1,out) && sag(c,mid,b,z,y,depth+1,out);
}

}  // namespace

int32_t inverse_curve_table(const double* parameters, const double* integrals,
                            int64_t table_count, int64_t segments, double* output) {
  if (!parameters || !integrals || !output || table_count<2 || segments<1)
    return PHX_MC_INVALID_ARGUMENT;
  for (int64_t i=0; i<table_count; ++i)
    if (!std::isfinite(parameters[i]) || !std::isfinite(integrals[i]) ||
        (i && (!(parameters[i]>parameters[i-1]) || !(integrals[i]>integrals[i-1]))))
      return PHX_MC_INVALID_INPUT;
  const double total=integrals[table_count-1]-integrals[0];
  if (!std::isfinite(total)) return PHX_MC_UNRESOLVED;
  output[0]=parameters[0];
  int64_t i=1;
  for (int64_t k=1; k<segments; ++k) {
    const double target=integrals[0]+total*(static_cast<double>(k)/segments);
    while (i+1<table_count && integrals[i]<target) ++i;
    const double s=parameters[i-1]+(parameters[i]-parameters[i-1])*
                   ((target-integrals[i-1])/(integrals[i]-integrals[i-1]));
    if (!(s>output[k-1] && s<parameters[table_count-1])) return PHX_MC_UNRESOLVED;
    output[k]=s;
  }
  output[segments]=parameters[table_count-1];
  return PHX_MC_OK;
}

int32_t repair_curve_sag(const CadGeometry& geometry, const CurveControls& controls,
                         int32_t curve, const std::vector<double>& input,
                         std::vector<double>& output, CurveStatistics& stats) {
  const auto& m=geometry.model();
  if (curve<0 || curve>=m.curve_count || input.size()<2 || !controls.curve_tolerances ||
      (m.surface_count && !controls.face_tolerances) ||
      !(controls.tolerance_fraction>0) || !std::isfinite(controls.tolerance_fraction))
    return PHX_MC_INVALID_ARGUMENT;
  for (std::size_t i=0;i<input.size();++i)
    if (!std::isfinite(input[i]) || input[i]<0 || input[i]>1 ||
        (i && !(input[i]>input[i-1]))) return PHX_MC_INVALID_INPUT;
  std::vector<Use> uses;
  for (int64_t l=0;l<m.loop_count;++l)
    for (int64_t u=m.loop_use_offsets[l];u<m.loop_use_offsets[l+1];++u)
      if (m.use_kinds[u]==0 && m.use_entities[u]==curve)
        uses.push_back({static_cast<int32_t>(u),m.loop_surfaces[l]});
  if (stats.sag_splits_per_curve.size()<static_cast<std::size_t>(m.curve_count))
    stats.sag_splits_per_curve.resize(static_cast<std::size_t>(m.curve_count),0);
  Context c{geometry,controls,stats,curve,uses};
  std::vector<double> repaired{input[0]};
  double x[3],y[3];
  if (!c.value(input[0],x)) return PHX_MC_INVALID_INPUT;
  const std::size_t before=stats.unresolved.size();
  for (std::size_t i=1;i<input.size();++i) {
    if (!c.value(input[i],y) || !sag(c,input[i-1],input[i],x,y,0,repaired)) return PHX_MC_INVALID_INPUT;
    for (int k=0;k<3;++k) x[k]=y[k];
  }
  output=std::move(repaired);
  return stats.unresolved.size()==before ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

int32_t discretize_curves(const CadGeometry& geometry, const CurveControls& controls,
                          CurveDiscretization& result) {
  const auto& m=geometry.model();
  if ((m.surface_count && (!controls.face_sizes || !controls.face_tolerances)) ||
      (m.curve_count && !controls.curve_tolerances) ||
      !std::isfinite(controls.tolerance_fraction) || !(controls.tolerance_fraction>0) ||
      controls.tolerance_fraction>1 || !std::isfinite(controls.grading) || controls.grading<1 ||
      !std::isfinite(controls.minimum_size) || !(controls.minimum_size>0)) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t f=0; f<m.surface_count; ++f)
    if (!(controls.face_sizes[f]>0) || !std::isfinite(controls.face_sizes[f]) ||
        !(controls.face_tolerances[f]>0) || std::isnan(controls.face_tolerances[f])) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t e=0; e<m.curve_count; ++e)
    if (!(controls.curve_tolerances[e]>0) || std::isnan(controls.curve_tolerances[e]) ||
        m.curve_ranges[2*e]==m.curve_ranges[2*e+1]) return PHX_MC_INVALID_ARGUMENT;
  std::vector<std::vector<Use>> uses(static_cast<std::size_t>(m.curve_count));
  for (int64_t l=0;l<m.loop_count;++l)
    for (int64_t u=m.loop_use_offsets[l];u<m.loop_use_offsets[l+1];++u)
      if (m.use_kinds[u]==0) uses[m.use_entities[u]].push_back({static_cast<int32_t>(u),m.loop_surfaces[l]});
  CurveDiscretization output; output.offsets.push_back(0);
  output.statistics.sag_splits_per_curve.resize(static_cast<std::size_t>(m.curve_count),0);
  std::vector<std::vector<double>> all_points(static_cast<std::size_t>(m.curve_count));
  for (int32_t e=0;e<m.curve_count;++e) {
    Context c{geometry,controls,output.statistics,e,uses[e]};
    if (!discretize(c,all_points[e])) return c.failure_status;
  }
  std::vector<double> limits(static_cast<std::size_t>(m.curve_count),std::numeric_limits<double>::infinity());
  if (!feature_sizes(geometry,controls,all_points,uses,output.statistics,limits))
    return PHX_MC_INVALID_INPUT;
  for (int32_t e=0;e<m.curve_count;++e) {
    Context c{geometry,controls,output.statistics,e,uses[e]};
    auto& points=all_points[e];
    if (std::isfinite(limits[e])) {
      c.feature_size=limits[e];
      ++output.statistics.small_feature_resamples;
      if (!discretize(c,points)) return c.failure_status;
    }
    std::vector<double> repaired;
    if (controls.sag_check) {
      const int32_t status=repair_curve_sag(geometry,controls,e,points,repaired,output.statistics);
      if (status!=PHX_MC_OK && status!=PHX_MC_UNRESOLVED) return status;
      output.parameters.insert(output.parameters.end(),repaired.begin(),repaired.end());
    } else {
      output.parameters.insert(output.parameters.end(),points.begin(),points.end());
    }
    output.offsets.push_back(static_cast<int64_t>(output.parameters.size()));
  }
  const int32_t status=output.statistics.unresolved.empty()?PHX_MC_OK:PHX_MC_UNRESOLVED;
  result=std::move(output); return status;
}
}  // namespace phx::mc::cad
