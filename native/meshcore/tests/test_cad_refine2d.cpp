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
CadRefine2DResult run(const Fixture& f,CadMetric2DFunction metric,void* context,int32_t& status,int cap=2000) {
  CadRefine2DInput in;
  in.points=f.points.data(); in.point_count=static_cast<int32_t>(f.points.size()/2);
  in.segments=f.segments.data(); in.segment_count=static_cast<int32_t>(f.segments.size()/2);
  in.holes=f.holes.data(); in.hole_count=static_cast<int32_t>(f.holes.size()/2);
  in.metric=metric; in.metric_context=context; in.point_cap=cap;
  CadRefine2DResult result;
  status=cad_refine2d(in,result);
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
}  // namespace

int main() {
  metric_case({1,0,1},8,8,8,8);
  metric_case({100,0,1},0.8,8,8,8);
  metric_case({2.5,1.5,2.5},4,4,8,8);
  metric_errors();
  hole_and_small_angle();
  varying_metric();
  adversarial_metrics();
  const Fixture f=rectangle(6,6,6,6);
  int32_t status;
  auto r=run(f,height_field,nullptr,status);
  report_refusals("height-field",status,r);
  PHX_CHECK(status==PHX_MC_OK || (status==PHX_MC_UNRESOLVED && r.stats.unresolved_triangles>0));
  check_topology(f,r);
  r=run(f,gradient,nullptr,status);
  report_refusals("gradient",status,r);
  PHX_CHECK(status==PHX_MC_OK || (status==PHX_MC_UNRESOLVED && r.stats.unresolved_triangles>0));
  check_topology(f,r);
  return phx::mc::test::finish("test_cad_refine2d");
}
