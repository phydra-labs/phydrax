// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Explicit sampled unsigned-distance wrapping of a triangle soup. Source
// triangles have a bounded barycentric covering cloud. Distance to that cloud
// is sampled on a conforming Freudenthal grid; no inside inference or repair.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {
using V = std::array<double, 3>;
using F = std::array<int64_t, 3>;
using phx::mc::NativeMap;
using phx::mc::NativeVector;
using C = std::array<int64_t, 4>;
struct EnvelopeData : phx::mc::NativeAllocatedObject {
  NativeVector<V> surface_points, volume_points;
  NativeVector<F> triangles;
  NativeVector<C> cells;
};
V sub(const V& a, const V& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
V add(const V& a, const V& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
V mul(const V& a, double t) { return {a[0]*t, a[1]*t, a[2]*t}; }
double dot(const V& a, const V& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
V cross(const V& a, const V& b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
double length(const V& a) { return std::hypot(a[0],a[1],a[2]); }
V point(const double* p, int64_t i) { return {p[3*i],p[3*i+1],p[3*i+2]}; }
double upward(double value) { return std::nextafter(value,std::numeric_limits<double>::infinity()); }
double downward(double value) { return std::nextafter(value,-std::numeric_limits<double>::infinity()); }
double upper_sum(double a,double b) { return upward(a+b); }
struct Work {
  int64_t used=0, limit;
  bool spend() {
    if(used>=limit) return false;
    phx::mc::native_execution_charge(1);
    ++used;
    return true;
  }
};
bool distance(const V& p,const NativeVector<V>& cloud,Work& work,double& result) {
  // One represented nearest-cloud distance query; pair comparisons below are
  // primitive work, not additional source-geometry evaluations.
  phx::mc::native_execution_charge(0,1);
  result=std::numeric_limits<double>::infinity();
  for(const V& q:cloud) {
    if(!work.spend()) return false;
    result=std::min(result,length(sub(p,q)));
  }
  return true;
}

bool represented(const V& p) {
  return phx::mc::coordinate_in_domain(p[0]) &&
         phx::mc::coordinate_in_domain(p[1]) &&
         phx::mc::coordinate_in_domain(p[2]);
}
bool collinear(const V& a,const V& b,const V& c) {
  for(int d=0;d<3;++d) {
    const int e=(d+1)%3;
    const double p[2]={a[d],a[e]},q[2]={b[d],b[e]},r[2]={c[d],c[e]};
    if(phx::mc::orient2d(p,q,r)!=0) return false;
  }
  return true;
}
struct Carrier {
  NativeVector<V> centers;
  NativeVector<C> cells;
  NativeVector<int8_t> grid_used;
  int64_t grid_vertices=0;
  Carrier(int64_t total) : grid_used(static_cast<size_t>(total),0) {}
};

// Negative ids name authoritative surface roots; nonnegative ids below total
// name original grid nodes; ids above total name this carrier's cell centers.
// This symbolic numbering keeps root-prefix identity independent of when an
// interior vertex is first encountered during extraction.
template<class Location,class Root>
int32_t cut_cell(const C& tet,const NativeVector<double>& field,double offset,
                 int64_t total,const NativeVector<V>& roots,const NativeVector<F>& skin,
                 size_t skin_begin,Location& location,Root& root,Carrier& carrier,
                 int64_t vertex_capacity,int64_t cell_capacity,Work& work) {
  auto point_at=[&](int64_t id) -> V {
    if(id<0) return roots[static_cast<size_t>(-id-1)];
    if(id<total) return location(id);
    return carrier.centers[static_cast<size_t>(id-total)];
  };
  auto put=[&](C cell) -> int32_t {
    if(!work.spend()||static_cast<int64_t>(carrier.cells.size())>=cell_capacity)
      return PHX_MC_CAPACITY_EXCEEDED;
    const V a=point_at(cell[0]),b=point_at(cell[1]),c=point_at(cell[2]),d=point_at(cell[3]);
    if(!represented(a)||!represented(b)||!represented(c)||!represented(d)) return PHX_MC_RANGE_ERROR;
    if(phx::mc::orient3d(a.data(),b.data(),c.data(),d.data())<=0) return PHX_MC_CONSTRAINT_INTERSECTION;
    for(int64_t id:cell) if(id>=0&&id<total&&!carrier.grid_used[id]) {
      carrier.grid_used[id]=1;++carrier.grid_vertices;
    }
    if(static_cast<int64_t>(roots.size()+carrier.centers.size())+carrier.grid_vertices>vertex_capacity)
      return PHX_MC_CAPACITY_EXCEEDED;
    carrier.cells.push_back(cell);return PHX_MC_OK;
  };
  int inside=0;
  for(int64_t id:tet) inside+=field[id]<offset;
  if(inside==0) return PHX_MC_OK;
  if(inside==4) {
    C cell=tet;const V a=location(cell[0]),b=location(cell[1]),c=location(cell[2]),d=location(cell[3]);
    if(!represented(a)||!represented(b)||!represented(c)||!represented(d)) return PHX_MC_RANGE_ERROR;
    if(phx::mc::orient3d(a.data(),b.data(),c.data(),d.data())<0) std::swap(cell[1],cell[2]);
    return put(cell);
  }
  std::array<F,10> boundary;size_t boundary_count=0;
  for(size_t i=skin_begin;i<skin.size();++i)
    boundary[boundary_count++]={-skin[i][0]-1,-skin[i][1]-1,-skin[i][2]-1};
  for(int opposite=0;opposite<4;++opposite) {
    std::array<int64_t,3> face;int slot=0;
    for(int i=0;i<4;++i) if(i!=opposite) face[slot++]=tet[i];
    if(field[face[0]]==offset&&field[face[1]]==offset&&field[face[2]]==offset) continue;
    const V a=location(face[0]),b=location(face[1]),c=location(face[2]),d=location(tet[opposite]);
    if(!represented(a)||!represented(b)||!represented(c)||!represented(d)) return PHX_MC_RANGE_ERROR;
    if(phx::mc::orient3d(a.data(),b.data(),c.data(),d.data())>0) std::swap(face[1],face[2]);
    std::array<int64_t,4> polygon;size_t n=0;
    auto append=[&](int64_t id) {if(n==0||polygon[n-1]!=id) polygon[n++]=id;};
    for(int i=0;i<3;++i) {
      const int64_t v=face[i],w=face[(i+1)%3];
      if(field[v]<offset) append(v);
      else if(field[v]==offset) {const int64_t r=root(v,v);if(r<0) return PHX_MC_CAPACITY_EXCEEDED;append(-r-1);}
      if((field[v]<offset&&field[w]>offset)||(field[v]>offset&&field[w]<offset)) {
        const int64_t r=root(v,w);if(r<0) return PHX_MC_CAPACITY_EXCEEDED;append(-r-1);
      }
    }
    if(n>1&&polygon[0]==polygon[n-1]) --n;
    if(n<3) continue;
    auto less=[&](int64_t u,int64_t v) {
      if((u<0)!=(v<0)) return u<0;
      return u<0 ? -u-1 < -v-1 : u<v;
    };
    size_t first=0;for(size_t i=1;i<n;++i) if(less(polygon[i],polygon[first])) first=i;
    for(size_t i=1;i+1<n;++i) {
      const F triangle={polygon[first],polygon[(first+i)%n],polygon[(first+i+1)%n]};
      const V p=point_at(triangle[0]),q=point_at(triangle[1]),r=point_at(triangle[2]);
      if(!represented(p)||!represented(q)||!represented(r)) return PHX_MC_RANGE_ERROR;
      if(!collinear(p,q,r)) {
        if(boundary_count==boundary.size()) return PHX_MC_INTERNAL_ERROR;
        boundary[boundary_count++]=triangle;
      }
    }
  }
  std::array<int64_t,6> vertices;size_t count=0;
  for(size_t i=0;i<boundary_count;++i) for(int64_t id:boundary[i]) {
    bool present=false;for(size_t j=0;j<count;++j) present|=vertices[j]==id;
    if(!present) {
      if(count==vertices.size()) return PHX_MC_INTERNAL_ERROR;
      vertices[count++]=id;
    }
  }
  if(count<4) return PHX_MC_DEGENERATE_INPUT;
  const V anchor=point_at(vertices[0]);V displacement={0,0,0};
  for(size_t i=1;i<count;++i) displacement=add(displacement,mul(sub(point_at(vertices[i]),anchor),1.0/count));
  const int64_t center=total+static_cast<int64_t>(carrier.centers.size());
  if(static_cast<int64_t>(roots.size()+carrier.centers.size())+carrier.grid_vertices>=vertex_capacity)
    return PHX_MC_CAPACITY_EXCEEDED;
  carrier.centers.push_back(add(anchor,displacement));
  for(size_t i=0;i<boundary_count;++i) {
    const int32_t status=put({center,boundary[i][0],boundary[i][1],boundary[i][2]});
    if(status!=PHX_MC_OK) return status;
  }
  return PHX_MC_OK;
}
}

extern "C" PHX_MC_API int32_t phx_mc_surface_envelope_create(
    const double* vertices,int64_t nv,const int64_t* faces,int64_t nf,
    double offset,double spacing,double certificate_spacing,
    int64_t sample_limit,int64_t work_limit,int64_t vertex_capacity,int64_t triangle_capacity,
    int64_t volume_vertex_capacity,int64_t tetrahedron_capacity,
    void** output_handle,int64_t* counts,double* bounds) {
  return phx::mc::guarded([&]() -> int32_t {
    if(!output_handle) return PHX_MC_INVALID_ARGUMENT;
    *output_handle=nullptr;
    if(!vertices||!faces||!counts||!bounds||nv<1||nf<1||
       !std::isfinite(offset)||!std::isfinite(spacing)||!std::isfinite(certificate_spacing)||
       offset<=0||spacing<=0||certificate_spacing<=0||sample_limit<1||work_limit<1||
       vertex_capacity<1||triangle_capacity<1||!phx::mc::addressable(nv,3,sizeof(double))||
       !phx::mc::addressable(nf,3,sizeof(int64_t))||!phx::mc::addressable(vertex_capacity,3,sizeof(double))||
       !phx::mc::addressable(triangle_capacity,3,sizeof(int64_t))||
       volume_vertex_capacity<1||tetrahedron_capacity<1||
       !phx::mc::addressable(volume_vertex_capacity,3,sizeof(double))||
       !phx::mc::addressable(tetrahedron_capacity,4,sizeof(int64_t))) return PHX_MC_INVALID_ARGUMENT;
    std::fill(counts,counts+7,0); std::fill(bounds,bounds+4,0.0);
    auto data=phx::mc::make_native_unique<EnvelopeData>();
    V lo=point(vertices,0),hi=lo;
    for(int64_t i=0;i<nv;++i) for(int d=0;d<3;++d) {
      const double x=vertices[3*i+d]; if(!std::isfinite(x)) return PHX_MC_NONFINITE_INPUT;
      lo[d]=std::min(lo[d],x); hi[d]=std::max(hi[d],x);
    }
    for(int64_t i=0;i<3*nf;++i) if(faces[i]<0||faces[i]>=nv) return PHX_MC_INVALID_INPUT;
    double scale=offset+spacing;
    for(int d=0;d<3;++d) scale=std::max({scale,std::abs(lo[d]),std::abs(hi[d])});
    // A nonzero absolute floor also covers underflow-rounding when the source
    // coordinates and the declared sampling scales are finite subnormals.
    const double roundoff=upward(std::max(256*std::numeric_limits<double>::epsilon()*scale,
                                       256*std::numeric_limits<double>::denorm_min()));
    if(!std::isfinite(roundoff)) return PHX_MC_RANGE_ERROR;
    const double grid_diameter=upward(upward(std::sqrt(3.0))*spacing);
    const double arithmetic_error=upward(4*roundoff);
    double error=upper_sum(upper_sum(grid_diameter,certificate_spacing),arithmetic_error);
    if(!(offset>error)) return PHX_MC_RANGE_ERROR;
    std::array<int64_t,3> shape;
    int64_t total=1;
    for(int d=0;d<3;++d) {
      lo[d]-=offset+2*spacing; hi[d]+=offset+2*spacing;
      const double n=std::ceil((hi[d]-lo[d])/spacing)+1;
      if(!std::isfinite(n)||n<2||n>static_cast<double>(sample_limit)||
         n>=static_cast<double>(std::numeric_limits<int64_t>::max())) return PHX_MC_CAPACITY_EXCEEDED;
      shape[d]=static_cast<int64_t>(n);
      if(total>sample_limit/shape[d]) return PHX_MC_CAPACITY_EXCEEDED;
      total*=shape[d];
    }
    if(!phx::mc::addressable(total,1,sizeof(double))) return PHX_MC_CAPACITY_EXCEEDED;
    auto index=[&](int64_t x,int64_t y,int64_t z) {return (x*shape[1]+y)*shape[2]+z;};
    auto location=[&](int64_t i) -> V {
      const int64_t z=i%shape[2]; i/=shape[2]; const int64_t y=i%shape[1],x=i/shape[1];
      return {lo[0]+x*spacing,lo[1]+y*spacing,lo[2]+z*spacing};
    };
    Work work{0,work_limit};
    NativeVector<V> cloud;
    double covering_radius=0;
    for(int64_t f=0;f<nf;++f) {
      const V a=point(vertices,faces[3*f]),b=point(vertices,faces[3*f+1]),c=point(vertices,faces[3*f+2]);
      const double diameter=upper_sum(upward(std::max({length(sub(a,b)),length(sub(a,c)),length(sub(b,c))})),roundoff);
      const double needed=std::max(1.0,std::ceil(diameter/certificate_spacing));
      if(!std::isfinite(needed)||needed>=static_cast<double>(sample_limit)||
         needed>=static_cast<double>(std::numeric_limits<int64_t>::max())) return PHX_MC_CAPACITY_EXCEEDED;
      const int64_t n=static_cast<int64_t>(needed);
      const long double samples=(static_cast<long double>(n)+1)*(static_cast<long double>(n)+2)/2;
      if(samples>sample_limit-total-counts[4]) return PHX_MC_CAPACITY_EXCEEDED;
      covering_radius=std::max(covering_radius,upper_sum(upward(diameter/n),roundoff));
      for(int64_t i=0;i<=n;++i) for(int64_t j=0;j<=n-i;++j) {
        if(!work.spend()) {counts[3]=work.used;return PHX_MC_CAPACITY_EXCEEDED;}
        cloud.push_back(add(a,add(mul(sub(b,a),static_cast<double>(i)/n),mul(sub(c,a),static_cast<double>(j)/n))));
        ++counts[4];
      }
    }
    error=upward(upper_sum(upper_sum(grid_diameter,covering_radius),arithmetic_error));
    if(!std::isfinite(error)||!(offset>error)) return PHX_MC_RANGE_ERROR;
    NativeVector<double> field(static_cast<size_t>(total));
    for(int64_t i=0;i<total;++i) {
      if(!distance(location(i),cloud,work,field[i])) {counts[3]=work.used;return PHX_MC_CAPACITY_EXCEEDED;}
      if(!std::isfinite(field[i])) return PHX_MC_RANGE_ERROR;
    }
    counts[2]=total;
    auto& result=data->surface_points;
    auto& triangles=data->triangles;
    Carrier carrier(total);
    NativeMap<std::pair<int64_t,int64_t>,int64_t> roots;
    auto root=[&](int64_t a,int64_t b) -> int64_t {
      if(field[a]==offset) b=a;
      else if(field[b]==offset) a=b;
      const std::pair<int64_t,int64_t> key={std::min(a,b),std::max(a,b)};
      const auto found=roots.find(key);
      if(found!=roots.end()) return found->second;
      if(static_cast<int64_t>(result.size())>=vertex_capacity||
         static_cast<int64_t>(result.size()+carrier.centers.size())+carrier.grid_vertices>=volume_vertex_capacity) return -1;
      const double t=a==b ? 0.0 : (offset-field[a])/(field[b]-field[a]);
      const int64_t id=static_cast<int64_t>(result.size());
      result.push_back(add(location(a),mul(sub(location(b),location(a)),t))); roots.emplace(key,id); return id;
    };
    auto emit=[&](int64_t a,int64_t b,int64_t c,const V& outward) -> bool {
      if(a<0||b<0||c<0) return false;
      if(a==b||b==c||a==c) return true;
      const V normal=cross(sub(result[b],result[a]),sub(result[c],result[a]));
      if(dot(normal,normal)==0) return true;
      if(static_cast<int64_t>(triangles.size())>=triangle_capacity) return false;
      if(dot(normal,outward)<0) std::swap(b,c);
      triangles.push_back({a,b,c}); return true;
    };
    constexpr int permutations[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for(int64_t x=0;x<shape[0]-1;++x) for(int64_t y=0;y<shape[1]-1;++y) for(int64_t z=0;z<shape[2]-1;++z) {
      for(const auto& permutation:permutations) {
        if(!work.spend()) {counts[3]=work.used;return PHX_MC_CAPACITY_EXCEEDED;}
        std::array<int64_t,3> p={x,y,z}; std::array<int64_t,4> tet;
        tet[0]=index(p[0],p[1],p[2]);
        for(int k=0;k<3;++k) {++p[permutation[k]];tet[k+1]=index(p[0],p[1],p[2]);}
        std::array<int64_t,4> inside{},outside{};int ni=0,no=0;
        for(int64_t id:tet) {if(field[id]<offset) inside[ni++]=id;else outside[no++]=id;}
        if(ni==0) continue;
        const size_t skin_begin=triangles.size();
        if(ni==4) {
          const int32_t status=cut_cell(tet,field,offset,total,result,triangles,skin_begin,location,root,carrier,volume_vertex_capacity,tetrahedron_capacity,work);
          if(status!=PHX_MC_OK) {counts[3]=work.used;return status;}
          continue;
        }
        const V outward=sub(location(outside[0]),location(inside[0]));
        bool ok;
        if(ni==1) ok=emit(root(inside[0],outside[0]),root(inside[0],outside[1]),root(inside[0],outside[2]),outward);
        else if(ni==3) ok=emit(root(outside[0],inside[0]),root(outside[0],inside[1]),root(outside[0],inside[2]),outward);
        else {
          const int64_t a=root(inside[0],outside[0]),b=root(inside[0],outside[1]),c=root(inside[1],outside[0]),d=root(inside[1],outside[1]);
          ok=emit(a,b,d,outward)&&emit(a,d,c,outward);
        }
        if(!ok) {counts[3]=work.used;return PHX_MC_CAPACITY_EXCEEDED;}
        const int32_t status=cut_cell(tet,field,offset,total,result,triangles,skin_begin,location,root,carrier,volume_vertex_capacity,tetrahedron_capacity,work);
        if(status!=PHX_MC_OK) {counts[3]=work.used;return status;}
      }
    }
    if(triangles.empty()) return PHX_MC_DEGENERATE_INPUT;
    // Preserve source boundary roots as the exact prefix of the volume
    // carrier. Root-only zero-dimensional internal contacts remain interior
    // carrier vertices; they never enter the surface deviation certificate.
    NativeVector<int64_t> remap(result.size(),-1),root_volume(result.size(),-1);
    NativeVector<int8_t> root_used(result.size(),0);
    for(const F& t:triangles) for(int64_t id:t) remap[id]=0;
    for(const C& cell:carrier.cells) for(int64_t id:cell) if(id<0) root_used[-id-1]=1;
    int64_t kept=0;
    for(size_t i=0;i<result.size();++i) if(remap[i]>=0) {
      root_volume[i]=remap[i]=kept++;data->volume_points.push_back(result[i]);
    }
    for(size_t i=0;i<result.size();++i) if(remap[i]<0&&root_used[i]) {
      root_volume[i]=static_cast<int64_t>(data->volume_points.size());
      data->volume_points.push_back(result[i]);
    }
    NativeVector<int64_t> grid_volume(static_cast<size_t>(total),-1);
    for(int64_t i=0;i<total;++i) if(carrier.grid_used[i]) {
      grid_volume[i]=static_cast<int64_t>(data->volume_points.size());
      data->volume_points.push_back(location(i));
    }
    const int64_t center_begin=static_cast<int64_t>(data->volume_points.size());
    data->volume_points.insert(data->volume_points.end(),carrier.centers.begin(),carrier.centers.end());
    data->cells=std::move(carrier.cells);
    for(C& cell:data->cells) for(int64_t& id:cell)
      id=id<0 ? root_volume[-id-1] : id<total ? grid_volume[id] : center_begin+(id-total);
    for(size_t i=0;i<result.size();++i) if(remap[i]>=0) result[remap[i]]=result[i];
    result.resize(static_cast<size_t>(kept));
    for(F& t:triangles) for(int64_t& id:t) id=remap[id];
    if(static_cast<int64_t>(data->volume_points.size())>volume_vertex_capacity) return PHX_MC_CAPACITY_EXCEEDED;
    double source_upper=0;
    for(const V& p:cloud) {
      double nearest;
      if(!distance(p,result,work,nearest)) {counts[3]=work.used;return PHX_MC_CAPACITY_EXCEEDED;}
      source_upper=std::max(source_upper,upper_sum(upper_sum(upward(nearest),covering_radius),arithmetic_error));
    }
    counts[0]=static_cast<int64_t>(result.size());counts[1]=static_cast<int64_t>(triangles.size());counts[3]=work.used;
    // Triangle covering + 1-Lipschitz cloud distance + P1 interpolation:
    // |d_source-I_h d_cloud| <= covering radius + grid-cell diameter.
    // I_h d_cloud=offset on every output triangle and <=error on the source.
    bounds[0]=upward(source_upper);
    bounds[1]=upper_sum(offset,error);
    bounds[2]=downward(offset-error);
    bounds[3]=error;
    if(!std::isfinite(bounds[0])||!std::isfinite(bounds[1])||!std::isfinite(bounds[3])||
       !(bounds[2]>0)) return PHX_MC_RANGE_ERROR;
    counts[5]=static_cast<int64_t>(data->volume_points.size());
    counts[6]=static_cast<int64_t>(data->cells.size());
    *output_handle=data.release();
    return PHX_MC_OK;
  });
}

extern "C" PHX_MC_API int32_t phx_mc_surface_envelope_arrays(
    void* handle,double* surface_vertices,int64_t* surface_triangles,
    double* volume_vertices,int64_t* tetrahedra) {
  if(!handle||!surface_vertices||!surface_triangles||!volume_vertices||!tetrahedra)
    return PHX_MC_INVALID_ARGUMENT;
  const auto& data=*static_cast<const EnvelopeData*>(handle);
  for(size_t i=0;i<data.surface_points.size();++i)
    std::copy(data.surface_points[i].begin(),data.surface_points[i].end(),surface_vertices+3*i);
  for(size_t i=0;i<data.triangles.size();++i)
    std::copy(data.triangles[i].begin(),data.triangles[i].end(),surface_triangles+3*i);
  for(size_t i=0;i<data.volume_points.size();++i)
    std::copy(data.volume_points[i].begin(),data.volume_points[i].end(),volume_vertices+3*i);
  for(size_t i=0;i<data.cells.size();++i)
    std::copy(data.cells[i].begin(),data.cells[i].end(),tetrahedra+4*i);
  return PHX_MC_OK;
}
extern "C" PHX_MC_API void phx_mc_surface_envelope_free(void* handle) {
  phx::mc::destroy_native_object(static_cast<EnvelopeData*>(handle));
}
