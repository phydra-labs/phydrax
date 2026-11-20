// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cad_refine2d.hpp"
#include "check.hpp"
#include "predicates.hpp"

namespace {
using namespace phx::mc;

CadMetric2D constant(const double*, void* context) {
  return *static_cast<CadMetric2D*>(context);
}
CadMetric2D height_field(const double* p, void*) {
  // S(u,v)=(u,v,0.02*u*v), with the first fundamental form calculated independently.
  const double du=0.02*p[1], dv=0.02*p[0];
  return {1+du*du,du*dv,1+dv*dv};
}
CadMetric2D gradient(const double* p, void*) {
  const double scale=std::exp(0.1*p[0]);
  return {scale*scale,0.2*scale,1.04};
}
CadMetric2D strong_patch(const double* p, void*) {
  // S(u,v)=(u+u*u+0.2*v,v,0); du changes fivefold over the patch.
  const double du=1+2*p[0];
  return {du*du,0.2*du,1.04};
}
CadMetric2D adversarial(const double* p, void* context) {
  const auto seed=*static_cast<const unsigned*>(context);
  const std::uint64_t first=splitmix64(seed);
  const std::uint64_t second=splitmix64(first);
  const double frequency_x=10.0+static_cast<double>(first%1900)*0.01;
  const double frequency_y=10.0+static_cast<double>(second%1900)*0.01;
  const double angle=seed+frequency_x*p[0]+frequency_y*p[1];
  const double a=2.0+std::sin(angle);
  const double b=2.0+std::cos(3*angle);
  return {a*a,0.2*a*b,b*b};
}
CadMetric2D invalid(const double*, void*) { return {1,2,1}; }
CadMetric2D nonfinite(const double*, void*) { return {NAN,0,1}; }

struct Fixture { std::vector<double> points; std::vector<int32_t> segments; std::vector<double> holes; };
Fixture rectangle(double width, double height, int nx, int ny) {
  Fixture f;
  auto add=[&](double x,double y) { f.points.push_back(x); f.points.push_back(y); };
  for(int i=0;i<nx;++i) add(width*i/nx,0);
  for(int i=0;i<ny;++i) add(width,height*i/ny);
  for(int i=0;i<nx;++i) add(width-width*i/nx,height);
  for(int i=0;i<ny;++i) add(0,height-height*i/ny);
  const int n=static_cast<int>(f.points.size()/2);
  for(int i=0;i<n;++i) { f.segments.push_back(i); f.segments.push_back((i+1)%n); }
  return f;
}
CadMetric2D graded_identity(const double* p, void* context) {
  const auto& f=*static_cast<const Fixture*>(context);
  const std::size_t n=f.points.size()/2;
  double h=1;
  for(std::size_t i=0;i<n;++i) {
    const double* b=&f.points[2*i];
    const double* prev=&f.points[2*((i+n-1)%n)];
    const double* next=&f.points[2*((i+1)%n)];
    const double spacing=std::min(std::hypot(b[0]-prev[0],b[1]-prev[1]),
                                  std::hypot(b[0]-next[0],b[1]-next[1]));
    h=std::min(h,spacing+0.3*std::hypot(p[0]-b[0],p[1]-b[1]));
  }
  const double scale=1/(h*h);
  return {scale,0,scale};
}
CadRefine2DResult run(const Fixture& f,CadMetric2DFunction metric,void* context,int32_t& status,int cap=2000) {
  CadRefine2DInput in;
  in.points=f.points.data(); in.point_count=static_cast<int32_t>(f.points.size()/2);
  in.segments=f.segments.data(); in.segment_count=static_cast<int32_t>(f.segments.size()/2);
  in.holes=f.holes.data(); in.hole_count=static_cast<int32_t>(f.holes.size()/2);
  in.metric=metric; in.metric_context=context; in.point_cap=cap;
  CadRefine2DResult result;
  status=cad_refine2d(in,result);
  PHX_CHECK(result.stats.relocations_tried==result.stats.relocations_accepted+result.stats.relocations_failed);
  // Fault caught: retaining transient refusal evidence after successful refinement.
  if(status==PHX_MC_OK) PHX_CHECK(result.unresolved.empty());
  if(status==PHX_MC_OK || status==PHX_MC_UNRESOLVED) {
    PHX_CHECK(result.unresolved.size()==static_cast<std::size_t>(result.stats.unresolved_triangles));
    int32_t previous=-1;
    for(const auto& evidence:result.unresolved) {
      PHX_CHECK(evidence.triangle>previous && (evidence.bad_size || evidence.bad_angle));
      previous=evidence.triangle;
    }
  }
  return result;
}
void report_refusals(const char* name,int32_t status,const CadRefine2DResult& r) {
  std::fprintf(stderr,"%s status=%d refused(lens/outside/proximity/nonstar)=%lld/%lld/%lld/%lld\n",
      name,status,static_cast<long long>(r.stats.encroachment_refusals),
      static_cast<long long>(r.stats.outside_refusals),static_cast<long long>(r.stats.proximity_refusals),
      static_cast<long long>(r.stats.nonstar_refusals));
}
void check_topology(const Fixture& f,const CadRefine2DResult& result) {
  const auto& tr=result.triangulation;
  // Fault caught: altered boundary coordinates or splitting a supplied edge.
  PHX_CHECK(std::memcmp(f.points.data(),tr.coordinates.data(),f.points.size()*sizeof(double))==0);
  for(std::size_t i=0;i<f.segments.size();i+=2) {
    const int32_t a=result.vertex_map[f.segments[i]], b=result.vertex_map[f.segments[i+1]];
    bool found=false;
    for(std::size_t t=0;t<tr.triangles.size();++t) if(tr.live[t] && tr.region[t] && !tr.is_ghost(t)) {
      for(int k=0;k<3;++k) {
        const auto& cell=tr.triangles[t];
        if (cell.v[next3(k)]==a && cell.v[prev3(k)]==b && tr.constraint(static_cast<int32_t>(t),k)!=-1) found=true;
      }
    }
    PHX_CHECK(found);
  }
  // Fault caught: a flipped cell or nonreciprocal interior edge.
  for(std::size_t t=0;t<tr.triangles.size();++t) if(tr.live[t] && tr.region[t] && !tr.is_ghost(t)) {
    const auto& c=tr.triangles[t];
    PHX_CHECK(orient2d(tr.point(c.v[0]),tr.point(c.v[1]),tr.point(c.v[2]))>0);
    for(int k=0;k<3;++k) {
      int32_t n=c.n[k];
      if(tr.constraint(static_cast<int32_t>(t),k)!=-1 || tr.is_ghost(n) || !tr.region[n]) continue;
      const int j=tr.edge_index(n,c.v[prev3(k)],c.v[next3(k)]);
      PHX_CHECK(j>=0);
      if (j>=0) PHX_CHECK(tr.triangles[n].n[j]==static_cast<int32_t>(t));
    }
  }
}
void check_metric_quality(const CadRefine2DResult& result, const CadMetric2D& m) {
  const auto& tr=result.triangulation;
  for (std::size_t t=0;t<tr.triangles.size();++t) if(tr.live[t] && tr.region[t] && !tr.is_ghost(t)) {
    const auto& c=tr.triangles[t];
    double lengths[3];
    for (int k=0;k<3;++k) {
      const double* a=tr.point(c.v[next3(k)]);
      const double* b=tr.point(c.v[prev3(k)]);
      const double x=b[0]-a[0], y=b[1]-a[1];
      lengths[k]=std::sqrt(m.uu*x*x+2*m.uv*x*y+m.vv*y*y);
      // Fault caught: oversizing or insertion closer than the metric spacing floor.
      PHX_CHECK(lengths[k]>=0.5-1e-12 && lengths[k]<=1.5+1e-12);
    }
    for(int k=0;k<3;++k) {
      const double cosine=(lengths[next3(k)]*lengths[next3(k)]+lengths[prev3(k)]*lengths[prev3(k)]-
                           lengths[k]*lengths[k])/(2*lengths[next3(k)]*lengths[prev3(k)]);
      // Fault caught: a metric angle below 25 degrees.
      PHX_CHECK(cosine<=std::cos(25.0*3.14159265358979323846/180.0)+1e-12);
    }
  }
}
void metric_case(CadMetric2D m,double width,double height,int nx,int ny) {
  const Fixture f=rectangle(width,height,nx,ny);
  int32_t status;
  const auto first=run(f,constant,&m,status);
  std::fprintf(stderr,"metric=(%g,%g,%g) status=%d insertions=%lld refused=%lld/%lld/%lld unresolved=%lld\n",
      m.uu,m.uv,m.vv,status,static_cast<long long>(first.stats.insertions),
      static_cast<long long>(first.stats.encroachment_refusals),
      static_cast<long long>(first.stats.outside_refusals),
      static_cast<long long>(first.stats.proximity_refusals),
      static_cast<long long>(first.stats.unresolved_triangles));
  {
    int metric_illegal=0, interior_edges=0;
    const auto& tr=first.triangulation;
    for (std::size_t t=0;t<tr.triangles.size();++t) {
      if (!tr.live[t] || !tr.region[t] || tr.is_ghost(static_cast<int32_t>(t))) continue;
      const auto& cell=tr.triangles[t];
      for(int k=0;k<3;++k) {
        const int32_t nb=cell.n[k];
        if (nb<=static_cast<int32_t>(t) || tr.constraint(static_cast<int32_t>(t),k)!=-1 ||
            tr.is_ghost(nb) || !tr.region[nb]) continue;
        const int j=tr.edge_index(nb,cell.v[prev3(k)],cell.v[next3(k)]);
        if (j<0) continue;
        const double* p=tr.point(tr.triangles[nb].v[j]);
        double x[3],y[3],lift[3];
        for(int i=0;i<3;++i) {
          const double* q=tr.point(cell.v[i]);
          x[i]=q[0]-p[0]; y[i]=q[1]-p[1];
          lift[i]=m.uu*x[i]*x[i]+2*m.uv*x[i]*y[i]+m.vv*y[i]*y[i];
        }
        const double d=lift[0]*(x[1]*y[2]-y[1]*x[2])+
                       lift[1]*(x[2]*y[0]-y[2]*x[0])+
                       lift[2]*(x[0]*y[1]-y[0]*x[1]);
        ++interior_edges;
        if (d>1e-12) ++metric_illegal;
      }
    }
    std::fprintf(stderr,"interior_edges=%d euclidean_valid_metric_illegal=%d\n",interior_edges,metric_illegal);
    PHX_CHECK(metric_illegal==0);
  }
  PHX_CHECK(status==PHX_MC_OK);
  check_topology(f,first);
  check_metric_quality(first,m);
  int32_t second_status;
  const auto second=run(f,constant,&m,second_status);
  // Fault caught: unstable queue ordering or point construction.
  PHX_CHECK(status==second_status);
  PHX_CHECK(first.triangulation.coordinates==second.triangulation.coordinates);
  PHX_CHECK(first.triangulation.triangles.size()==second.triangulation.triangles.size());
  PHX_CHECK(std::memcmp(first.triangulation.triangles.data(),second.triangulation.triangles.data(),
                        first.triangulation.triangles.size()*sizeof(Triangle2D))==0);
}
void metric_errors() {
  const Fixture f=rectangle(4,4,4,4);
  int32_t status;
  auto a=run(f,invalid,nullptr,status);
  PHX_CHECK(status==PHX_MC_INVALID_INPUT);
  PHX_CHECK(a.stats.metric_evaluations>0);
  auto b=run(f,nonfinite,nullptr,status);
  PHX_CHECK(status==PHX_MC_INVALID_INPUT);
  PHX_CHECK(b.stats.metric_evaluations>0);
  auto c=run(f,height_field,nullptr,status,static_cast<int>(f.points.size()/2));
  report_refusals("cap",status,c);
  // Fault caught: exceeding the caller's point cap without an unresolved status.
  PHX_CHECK(status==PHX_MC_UNRESOLVED);
  CadRefine2DInput recount;
  recount.point_count=static_cast<int32_t>(f.points.size()/2);
  recount.metric=height_field;
  int64_t total=0, sizes=0, angles=0;
  for(std::size_t t=0;t<c.triangulation.triangles.size();++t) {
    if(!c.triangulation.live[t] || !c.triangulation.region[t] ||
       c.triangulation.is_ghost(static_cast<int32_t>(t))) continue;
    const auto& tr=c.triangulation;
    const auto& cell=tr.triangles[t];
    const double* p[3]={tr.point(cell.v[0]),tr.point(cell.v[1]),tr.point(cell.v[2])};
    const double centroid[2]={(p[0][0]+p[1][0]+p[2][0])/3,
                              (p[0][1]+p[1][1]+p[2][1])/3};
    const auto m=height_field(centroid,nullptr);
    double length[3];
    int shortest=0;
    for(int k=0;k<3;++k) {
      const double dx=p[next3(k)][0]-p[prev3(k)][0];
      const double dy=p[next3(k)][1]-p[prev3(k)][1];
      length[k]=std::sqrt(m.uu*dx*dx+2*m.uv*dx*dy+m.vv*dy*dy);
      if(length[k]<length[shortest]) shortest=k;
    }
    const double semiperimeter=(length[0]+length[1]+length[2])/2;
    const double area=std::sqrt(semiperimeter*(semiperimeter-length[0])*
                                (semiperimeter-length[1])*(semiperimeter-length[2]));
    const double radius=length[0]*length[1]*length[2]/(4*area);
    const int j=next3(shortest), k=prev3(shortest);
    const double cosine=(length[j]*length[j]+length[k]*length[k]-
                         length[shortest]*length[shortest])/(2*length[j]*length[k]);
    const double min_angle=std::acos(std::clamp(cosine,-1.0,1.0));
    const bool protected_angle=cell.v[shortest]<recount.point_count &&
        tr.constraint(static_cast<int32_t>(t),j)!=-1 &&
        tr.constraint(static_cast<int32_t>(t),k)!=-1;
    const bool expected_size=radius>std::sqrt(0.5);
    const bool expected_angle=!protected_angle &&
        min_angle<25.0*3.14159265358979323846/180.0;
    bool size=false, angle=false;
    PHX_CHECK(cad_refine2d_bad_triangle(recount,c,static_cast<int32_t>(t),size,angle)==PHX_MC_OK);
    // Fault caught: using a vertex metric or a wrong circumradius in the predicate.
    PHX_CHECK(size==expected_size && angle==expected_angle);
    total+=expected_size || expected_angle;
    sizes+=expected_size; angles+=expected_angle;
  }
  // Fault caught: reporting only one queued triangle on a point-cap exit.
  PHX_CHECK(total>1 && c.stats.unresolved_triangles==total);
  PHX_CHECK(c.stats.unresolved_size==sizes && c.stats.unresolved_angle==angles);
  check_topology(f,c);
}
void hole_and_small_angle() {
  Fixture f=rectangle(8,8,8,8);
  const int base=static_cast<int>(f.points.size()/2);
  const double hole[]={3,3,3,4,3,5,4,5,5,5,5,4,5,3,4,3};
  f.points.insert(f.points.end(),std::begin(hole),std::end(hole));
  for(int i=0;i<8;++i) { f.segments.push_back(base+i); f.segments.push_back(base+(i+1)%8); }
  f.holes={4,4};
  CadMetric2D metric{1,0,1};
  int32_t status;
  const auto r=run(f,constant,&metric,status);
  report_refusals("hole",status,r);
  PHX_CHECK(status==PHX_MC_OK);
  check_topology(f,r);
  for(std::size_t t=0;t<r.triangulation.triangles.size();++t) {
    const auto& tr=r.triangulation;
    if (!tr.live[t] || !tr.region[t] || tr.is_ghost(static_cast<int32_t>(t))) continue;
    const auto& c=tr.triangles[t];
    const double x=(tr.point(c.v[0])[0]+tr.point(c.v[1])[0]+tr.point(c.v[2])[0])/3;
    const double y=(tr.point(c.v[0])[1]+tr.point(c.v[1])[1]+tr.point(c.v[2])[1])/3;
    // Fault caught: flood fill retaining the hole interior.
    PHX_CHECK(!(x>3 && x<5 && y>3 && y<5));
  }
  Fixture wedge{{0,0,0.8,0,0.8,0.1},{0,1,1,2,2,0},{}};
  const auto small=run(wedge,constant,&metric,status);
  report_refusals("small-angle",status,small);
  // Fault caught: treating a protected acute input angle as refinable.
  PHX_CHECK(status==PHX_MC_OK);
  PHX_CHECK(small.stats.insertions==0);
  check_topology(wedge,small);
}
void adversarial_metrics() {
  const Fixture f=rectangle(8,8,8,8);
  for(unsigned seed : {17u,239u,8191u}) {
    int32_t status;
    const auto first=run(f,adversarial,&seed,status,static_cast<int>(f.points.size()/2)+5);
    report_refusals("seeded-metric",status,first);
    // Fault caught: cycling past a finite point cap or silently accepting it.
    PHX_CHECK(status==PHX_MC_UNRESOLVED);
    PHX_CHECK(first.triangulation.vertex_count()<=static_cast<int>(f.points.size()/2)+5);
    int32_t second_status;
    const auto second=run(f,adversarial,&seed,second_status,static_cast<int>(f.points.size()/2)+5);
    PHX_CHECK(status==second_status);
    PHX_CHECK(first.triangulation.coordinates==second.triangulation.coordinates);
    check_topology(f,first);
  }
}
void varying_metric() {
  Fixture f;
  auto add=[&](double x,double y) { f.points.push_back(x); f.points.push_back(y); };
  auto u=[](int i) { return (std::sqrt(1.0+4*i)-1)/2; };
  for(int i=0;i<6;++i) add(u(i),0);
  for(int i=0;i<6;++i) add(u(6),i);
  for(int i=6;i>0;--i) add(u(i),6);
  for(int i=6;i>0;--i) add(0,i);
  const int n=static_cast<int>(f.points.size()/2);
  for(int i=0;i<n;++i) { f.segments.push_back(i); f.segments.push_back((i+1)%n); }
  int32_t status;
  auto r=run(f,strong_patch,nullptr,status);
  std::fprintf(stderr,"strong patch status=%d refused=%lld/%lld/%lld/%lld\n",status,
      static_cast<long long>(r.stats.encroachment_refusals),static_cast<long long>(r.stats.outside_refusals),
      static_cast<long long>(r.stats.proximity_refusals),static_cast<long long>(r.stats.nonstar_refusals));
  PHX_CHECK(status==PHX_MC_OK);
  check_topology(f,r);
  const auto& tr=r.triangulation;
  for(std::size_t t=0;t<tr.triangles.size();++t) if(tr.live[t] && tr.region[t] && !tr.is_ghost(static_cast<int32_t>(t))) {
    const auto& cell=tr.triangles[t];
    const double* p[3]={tr.point(cell.v[0]),tr.point(cell.v[1]),tr.point(cell.v[2])};
    double center[2]={(p[0][0]+p[1][0]+p[2][0])/3,(p[0][1]+p[1][1]+p[2][1])/3};
    const auto m=strong_patch(center,nullptr);
    double lengths[3];
    for(int k=0;k<3;++k) {
      const double* a=p[next3(k)]; const double* b=p[prev3(k)];
      const double x=b[0]-a[0],y=b[1]-a[1], mid[2]={(a[0]+b[0])/2,(a[1]+b[1])/2};
      const auto edge=strong_patch(mid,nullptr);
      const double edge_length=std::sqrt(edge.uu*x*x+2*edge.uv*x*y+edge.vv*y*y);
      PHX_CHECK(edge_length>=0.5-1e-12 && edge_length<=1.5+1e-12);
      lengths[k]=std::sqrt(m.uu*x*x+2*m.uv*x*y+m.vv*y*y);
    }
    for(int k=0;k<3;++k) {
      const double cosine=(lengths[next3(k)]*lengths[next3(k)]+lengths[prev3(k)]*lengths[prev3(k)]-
          lengths[k]*lengths[k])/(2*lengths[next3(k)]*lengths[prev3(k)]);
      PHX_CHECK(cosine<=std::cos(25*3.14159265358979323846/180)+1e-12);
    }
  }
}
struct PoleProbe {
  std::vector<std::array<double,2>> candidates;
  std::vector<std::array<double,2>> evaluations;
};

CadMetric2D pole_metric(const double* p,void* context) {
  auto& probe=*static_cast<PoleProbe*>(context);
  probe.evaluations.push_back({p[0],p[1]});
  return {1,0,1};
}

int32_t pole_admit(void* context,const double* p) {
  auto& probe=*static_cast<PoleProbe*>(context);
  probe.candidates.push_back({p[0],p[1]});
  return std::hypot(p[0],p[1])>=0.9 && p[0]<=2;
}

void fixed_pole_zone() {
  Fixture f;
  constexpr double pi=3.14159265358979323846;
  f.points={0,0};
  for(int i=0;i<8;++i) {
    const double a=2*pi*i/8;
    f.points.push_back(std::cos(a)); f.points.push_back(std::sin(a));
  }
  for(int i=0;i<16;++i) {
    const double a=2*pi*i/16;
    f.points.push_back(4*std::cos(a)); f.points.push_back(4*std::sin(a));
  }
  for(int i=0;i<8;++i) { f.segments.push_back(1+i); f.segments.push_back(1+(i+1)%8); }
  for(int i=0;i<16;++i) { f.segments.push_back(9+i); f.segments.push_back(9+(i+1)%16); }
  std::vector<uint8_t> fixed(25,0);
  std::fill(fixed.begin(),fixed.begin()+9,1);
  PoleProbe probe;
  CadRefine2DInput in;
  in.points=f.points.data(); in.point_count=25;
  in.segments=f.segments.data(); in.segment_count=24;
  in.metric=pole_metric; in.metric_context=&probe; in.point_cap=2000;
  in.fixed_vertices=fixed.data(); in.admit_point=pole_admit; in.admit_context=&probe;
  CadRefine2DResult result;
  const int status=cad_refine2d(in,result);
  PHX_CHECK(status==PHX_MC_OK || status==PHX_MC_UNRESOLVED);
  int64_t excluded=0;
  for(const auto& p:probe.candidates) excluded+=std::hypot(p[0],p[1])<0.9 || p[0]>2;
  // Fault caught: skipping candidate admission or counting a different refusal.
  PHX_CHECK(excluded>0 && result.stats.excluded_refusals==excluded);
  // Fault caught: evaluating the singular zone or inserting through its guard.
  PHX_CHECK(!probe.evaluations.empty());
  for(const auto& p:probe.evaluations) PHX_CHECK(std::hypot(p[0],p[1])>=0.9);
  for(int32_t i=in.point_count;i<result.triangulation.vertex_count();++i) {
    const double* p=result.triangulation.point(i);
    PHX_CHECK(std::hypot(p[0],p[1])>=0.9);
  }
  int64_t skipped=0;
  const auto& tr=result.triangulation;
  for(std::size_t t=0;t<tr.triangles.size();++t) {
    if(!tr.live[t] || !tr.region[t] || tr.is_ghost(static_cast<int32_t>(t))) continue;
    const auto& cell=tr.triangles[t];
    if(cell.v[0]<9 && cell.v[1]<9 && cell.v[2]<9) ++skipped;
  }
  // Fault caught: scanning fixed cap triangles or misreporting their count.
  PHX_CHECK(skipped>0 && result.stats.fixed_triangles_skipped==skipped);
  check_topology(f,result);
}

void permutation_invariance() {
  Fixture first, second;
  constexpr int n=12;
  constexpr double pi=3.14159265358979323846;
  for(int i=0;i<n;++i) {
    const double a=2*pi*i/n, r=2+0.07*std::sin(3*a+0.11);
    first.points.push_back(r*std::cos(a));
    first.points.push_back(r*std::sin(a));
    first.segments.push_back(i); first.segments.push_back((i+1)%n);
  }
  for(int i=n-1;i>=0;--i) {
    second.points.push_back(first.points[2*i]);
    second.points.push_back(first.points[2*i+1]);
  }
  for(int i=n-1;i>=0;--i) {
    second.segments.push_back(n-1-first.segments[2*i]);
    second.segments.push_back(n-1-first.segments[2*i+1]);
  }
  CadMetric2D m{1,0,1};
  int32_t a_status,b_status;
  const auto a=run(first,constant,&m,a_status);
  const auto b=run(second,constant,&m,b_status);
  PHX_CHECK(a_status==PHX_MC_OK && b_status==PHX_MC_OK);
  PHX_CHECK(a.stats.insertions>0 && b.stats.insertions>0);
  auto cells=[](const CadRefine2DResult& result) {
    using Point=std::array<double,2>;
    std::vector<std::array<Point,3>> triangles;
    const auto& tr=result.triangulation;
    for(std::size_t t=0;t<tr.triangles.size();++t) {
      if(!tr.live[t] || !tr.region[t] || tr.is_ghost(static_cast<int32_t>(t))) continue;
      std::array<Point,3> cell;
      for(int k=0;k<3;++k) {
        const double* p=tr.point(tr.triangles[t].v[k]);
        cell[k]={p[0],p[1]};
      }
      std::sort(cell.begin(),cell.end());
      triangles.push_back(cell);
    }
    std::sort(triangles.begin(),triangles.end());
    return triangles;
  };
  // Fault caught: input-index tie breaks or cyclic rotation changing constructed coordinates.
  const auto a_cells=cells(a), b_cells=cells(b);
  PHX_CHECK(a_cells.size()==b_cells.size());
  if(a_cells.size()==b_cells.size())
    PHX_CHECK(std::memcmp(a_cells.data(),b_cells.data(),a_cells.size()*sizeof(a_cells[0]))==0);
}

void blocked_boundary_evidence() {
  const double points[]={0,0,4,0,4,1,3.9,1,0,1};
  const int32_t segments[]={2,3,0,1};
  CadMetric2D m{1,0,1};
  CadRefine2DInput in;
  in.points=points; in.point_count=5; in.segments=segments; in.segment_count=2;
  in.keep_convex_hull=true; in.metric=constant; in.metric_context=&m; in.point_cap=1000;
  CadRefine2DResult result;
  const int32_t status=cad_refine2d(in,result);
  PHX_CHECK(status==PHX_MC_UNRESOLVED);
  int observed=0;
  for(const auto& e:result.unresolved) if(e.refusal==CadRefine2DRefusal::encroachment) {
    ++observed;
    // Fault caught: returning a BVH position instead of the input segment index.
    PHX_CHECK(e.segment==1);
    PHX_CHECK(e.relocation_failed);
    // Independent Euclidean diametral-lens test for the blocked bottom segment.
    PHX_CHECK((e.candidate[0]-points[0])*(e.candidate[0]-points[2])+
              (e.candidate[1]-points[1])*(e.candidate[1]-points[3])<0);
    PHX_CHECK(e.triangle>=0 && static_cast<std::size_t>(e.triangle)<result.triangulation.triangles.size());
    PHX_CHECK(e.bad_size || e.bad_angle);
  }
  PHX_CHECK(observed>0);
  PHX_CHECK(result.unresolved.size()==static_cast<std::size_t>(result.stats.unresolved_triangles));
}

}  // namespace

int main() {
  metric_case({1,0,1},8,8,8,8);
  metric_case({100,0,1},0.8,8,8,8);
  metric_case({2.5,1.5,2.5},4,4,8,8);
  metric_errors();
  hole_and_small_angle();
  varying_metric();
  adversarial_metrics();
  fixed_pole_zone();
  permutation_invariance();
  blocked_boundary_evidence();
  const Fixture f=rectangle(6,6,6,6);
  int32_t status;
  auto r=run(f,height_field,nullptr,status);
  report_refusals("height-field",status,r);
  PHX_CHECK(status==PHX_MC_OK || (status==PHX_MC_UNRESOLVED && r.stats.unresolved_triangles>0));
  check_topology(f,r);
  r=run(f,gradient,nullptr,status);
  report_refusals("gradient",status,r);
  // Fault caught: failing to relocate any candidate blocked by a fixed segment,
  // or dropping the remaining failed-relocation evidence.
  PHX_CHECK(status==PHX_MC_UNRESOLVED);
  PHX_CHECK(r.stats.encroachment_refusals==9 && r.stats.outside_refusals==1);
  PHX_CHECK(r.stats.proximity_refusals==0 && r.stats.nonstar_refusals==0 &&
            r.stats.unresolved_triangles>0 && r.stats.relocations_accepted>0 &&
            r.stats.relocations_failed>0);
  check_topology(f,r);
  Fixture fine=rectangle(6,6,24,24);
  r=run(fine,graded_identity,&fine,status);
  report_refusals("graded-quarter",status,r);
  // Fault caught: failure to grade refinement from a finely sampled boundary.
  PHX_CHECK(status==PHX_MC_OK && r.stats.unresolved_triangles==0);
  PHX_CHECK(r.stats.encroachment_refusals==0 && r.stats.outside_refusals==0 &&
            r.stats.proximity_refusals==0 && r.stats.nonstar_refusals==0);
  check_topology(fine,r);
  return phx::mc::test::finish("test_cad_refine2d");
}
