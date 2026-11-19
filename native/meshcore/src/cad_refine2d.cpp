// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_refine2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

#include "cdt2d.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace {

struct Entry {
  double radius;
  std::array<int32_t, 3> vertices;
  int32_t slot;
};
struct Worse {
  bool operator()(const Entry& a, const Entry& b) const {
    if (a.radius != b.radius) return a.radius < b.radius;
    return a.vertices > b.vertices;
  }
};
struct Boundary {
  int32_t u;
  int32_t v;
  int32_t outside;
  int32_t constraint;
};
struct SegmentNode {
  double low[2], high[2], max_length;
  int begin, end, left, right;
};

class Refinement {
 public:
  Refinement(const CadRefine2DInput& input, CadRefine2DResult& result)
      : input_(input), result_(result), tri_(result.triangulation),
        sine_(std::sin(input.min_angle_degrees * (3.14159265358979323846 / 180.0))) {}


  void initialize_grid() {
    const double* first=tri_.point(0);
    low_[0]=high_[0]=first[0]; low_[1]=high_[1]=first[1];
    for (int32_t i=1;i<tri_.vertex_count();++i) {
      const double* p=tri_.point(i);
      for (int k=0;k<2;++k) { low_[k]=std::min(low_[k],p[k]); high_[k]=std::max(high_[k],p[k]); }
    }
    grid_size_=std::max(1,std::min(2048,static_cast<int>(std::sqrt(input_.point_cap))));
    heads_.assign(static_cast<std::size_t>(grid_size_)*grid_size_,-1);
    next_vertex_.reserve(static_cast<std::size_t>(input_.point_cap));
    for (int32_t i=0;i<tri_.vertex_count();++i) add_to_grid(i);
    segment_order_.resize(static_cast<std::size_t>(input_.segment_count));
    for (int i=0;i<input_.segment_count;++i) segment_order_[i]=i;
    if (!segment_order_.empty()) build_segment_tree(0,input_.segment_count);
  }

  int grid_coordinate(double x,int axis) const {
    const double scaled=(x-low_[axis])/(high_[axis]-low_[axis])*grid_size_;
    if (!(scaled>0)) return 0;
    if (scaled>=grid_size_) return grid_size_-1;
    return static_cast<int>(scaled);
  }

  void add_to_grid(int32_t vertex) {
    const double* p=tri_.point(vertex);
    const int x=grid_coordinate(p[0],0), y=grid_coordinate(p[1],1);
    const std::size_t bucket=static_cast<std::size_t>(y)*grid_size_+x;
    next_vertex_.push_back(heads_[bucket]);
    heads_[bucket]=vertex;
  }

  bool too_close(const double* p,const CadMetric2D& m) const {
    const double determinant=m.uu*m.vv-m.uv*m.uv;
    const double rx=0.5*std::sqrt(m.vv/determinant);
    const double ry=0.5*std::sqrt(m.uu/determinant);
    const int x0=grid_coordinate(p[0]-rx,0), x1=grid_coordinate(p[0]+rx,0);
    const int y0=grid_coordinate(p[1]-ry,1), y1=grid_coordinate(p[1]+ry,1);
    for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x) {
      for (int32_t i=heads_[static_cast<std::size_t>(y)*grid_size_+x];i>=0;i=next_vertex_[i]) {
        if (tri_.vertex_triangle[i]<0) continue;
        const double* q=tri_.point(i);
        if (dot(m,p[0]-q[0],p[1]-q[1],p[0]-q[0],p[1]-q[1]) < 0.25) return true;
      }
    }
    return false;
  }
  int32_t run() {
    initialize_grid();
    for (std::size_t t = 0; t < tri_.triangles.size(); ++t) {
      if (inside(static_cast<int32_t>(t))) enqueue(static_cast<int32_t>(t));
      if (status_ != PHX_MC_OK) return status_;
    }
    while (!heap_.empty()) {
      const Entry entry = heap_[0];
      heap_remove(entry.slot);
      if (!inside(entry.slot) || vertices(entry.slot) != entry.vertices) continue;
      if (tri_.vertex_count() >= input_.point_cap) break;
      process(entry.slot);
      if (status_ != PHX_MC_OK) return status_;
    }
    return summarize();
  }

  int32_t classify(int32_t t, bool& bad_size, bool& bad_angle) {
    if (!inside(t)) return PHX_MC_INVALID_ARGUMENT;
    double radius=0;
    geometry(t,radius,nullptr,&bad_size,&bad_angle);
    return status_;
  }

  int32_t summarize() {
    result_.stats.unresolved_triangles=0;
    result_.stats.unresolved_size=0;
    result_.stats.unresolved_angle=0;
    for (std::size_t t=0;t<tri_.triangles.size();++t) {
      if (!inside(static_cast<int32_t>(t))) continue;
      double radius=0.0;
      bool bad_size=false, bad_angle=false;
      if (geometry(static_cast<int32_t>(t),radius,nullptr,&bad_size,&bad_angle))
        ++result_.stats.unresolved_triangles;
      result_.stats.unresolved_size+=bad_size;
      result_.stats.unresolved_angle+=bad_angle;
      if (status_!=PHX_MC_OK) return status_;
    }
    return result_.stats.unresolved_triangles ? PHX_MC_UNRESOLVED : PHX_MC_OK;
  }

 private:
  const CadRefine2DInput& input_;
  CadRefine2DResult& result_;
  Triangulation2D& tri_;
  double sine_;
  int32_t status_ = PHX_MC_OK;
  std::vector<Entry> heap_;
  std::vector<int32_t> heap_positions_;
  std::vector<uint64_t> marks_;
  uint64_t epoch_ = 0;
  std::vector<int32_t> stack_;
  std::vector<int32_t> cavity_;
  std::vector<Boundary> boundary_;
  std::vector<int32_t> created_;
  std::vector<int32_t> vertex_slots_;
  std::vector<uint64_t> boundary_vertex_marks_;
  double low_[2]{}, high_[2]{};
  int grid_size_=1;
  std::vector<int32_t> heads_;
  std::vector<int32_t> next_vertex_;
  std::vector<int32_t> segment_order_;
  std::vector<SegmentNode> segment_nodes_;
  std::vector<int32_t> segment_stack_;

  void heap_swap(int a,int b) {
    std::swap(heap_[a],heap_[b]);
    heap_positions_[heap_[a].slot]=a;
    heap_positions_[heap_[b].slot]=b;
  }
  void heap_repair(int pos) {
    while (pos>0) {
      const int parent=(pos-1)/2;
      if (!Worse{}(heap_[parent],heap_[pos])) break;
      heap_swap(parent,pos); pos=parent;
    }
    for (;;) {
      const int left=2*pos+1;
      if (left>=static_cast<int>(heap_.size())) break;
      const int right=left+1;
      const int best=right<static_cast<int>(heap_.size()) && Worse{}(heap_[left],heap_[right]) ? right : left;
      if (!Worse{}(heap_[pos],heap_[best])) break;
      heap_swap(pos,best); pos=best;
    }
  }
  void heap_remove(int32_t slot) {
    if (static_cast<std::size_t>(slot)>=heap_positions_.size()) return;
    const int pos=heap_positions_[slot];
    if (pos<0) return;
    const int last=static_cast<int>(heap_.size())-1;
    if (pos!=last) heap_swap(pos,last);
    heap_.pop_back();
    heap_positions_[slot]=-1;
    if (pos<static_cast<int>(heap_.size())) heap_repair(pos);
  }
  void heap_add(const Entry& entry) {
    if (heap_positions_.size()<=static_cast<std::size_t>(entry.slot))
      heap_positions_.resize(static_cast<std::size_t>(entry.slot)+1,-1);
    int pos=heap_positions_[entry.slot];
    if (pos<0) {
      pos=static_cast<int>(heap_.size());
      heap_.push_back(entry);
      heap_positions_[entry.slot]=pos;
    } else {
      heap_[pos]=entry;
    }
    heap_repair(pos);
  }

  const double* segment_endpoint(int segment,int endpoint) const {
    return tri_.point(result_.vertex_map[input_.segments[2*segment+endpoint]]);
  }
  int build_segment_tree(int begin,int end) {
    SegmentNode node{{INFINITY,INFINITY},{-INFINITY,-INFINITY},0.0,begin,end,-1,-1};
    for (int i=begin;i<end;++i) {
      const double* a=segment_endpoint(segment_order_[i],0);
      const double* b=segment_endpoint(segment_order_[i],1);
      for(int k=0;k<2;++k) {
        node.low[k]=std::min(node.low[k],std::min(a[k],b[k]));
        node.high[k]=std::max(node.high[k],std::max(a[k],b[k]));
      }
      node.max_length=std::max(node.max_length,std::hypot(a[0]-b[0],a[1]-b[1]));
    }
    const int index=static_cast<int>(segment_nodes_.size());
    segment_nodes_.push_back(node);
    if (end-begin>8) {
      const int axis=node.high[0]-node.low[0] >= node.high[1]-node.low[1] ? 0 : 1;
      const int middle=(begin+end)/2;
      std::nth_element(segment_order_.begin()+begin,segment_order_.begin()+middle,
          segment_order_.begin()+end,[&](int a,int b) {
            const double aa=segment_endpoint(a,0)[axis]+segment_endpoint(a,1)[axis];
            const double bb=segment_endpoint(b,0)[axis]+segment_endpoint(b,1)[axis];
            return aa==bb ? a<b : aa<bb;
          });
      const int left=build_segment_tree(begin,middle);
      const int right=build_segment_tree(middle,end);
      segment_nodes_[index].left=left;
      segment_nodes_[index].right=right;
    }
    return index;
  }

  bool inside(int32_t t) const {
    return t >= 0 && tri_.live[static_cast<std::size_t>(t)] &&
           tri_.region[static_cast<std::size_t>(t)] && !tri_.is_ghost(t);
  }
  std::array<int32_t, 3> vertices(int32_t t) const {
    const Triangle2D& x = tri_.triangles[static_cast<std::size_t>(t)];
    std::array<int32_t, 3> v{x.v[0], x.v[1], x.v[2]};
    std::sort(v.begin(), v.end());
    return v;
  }
  CadMetric2D metric(const double* p) {
    ++result_.stats.metric_evaluations;
    const CadMetric2D m = input_.metric(p, input_.metric_context);
    if (!std::isfinite(m.uu) || !std::isfinite(m.uv) || !std::isfinite(m.vv) ||
        !(m.uu > 0.0) || !(m.vv > 0.0) || !(m.uu * m.vv - m.uv * m.uv > 0.0)) {
      status_ = PHX_MC_INVALID_INPUT;
      result_.stats.invalid_metric_point[0] = p[0];
      result_.stats.invalid_metric_point[1] = p[1];
    }
    return m;
  }
  static double dot(const CadMetric2D& m, double x, double y, double u, double v) {
    return x * (m.uu * u + m.uv * v) + y * (m.uv * u + m.vv * v);
  }
  bool geometry(int32_t t, double& radius, double* center,
                bool* size_reason=nullptr, bool* angle_reason=nullptr) {
    const Triangle2D& cell = tri_.triangles[static_cast<std::size_t>(t)];
    const double* a = tri_.point(cell.v[0]);
    const double* b = tri_.point(cell.v[1]);
    const double* c = tri_.point(cell.v[2]);
    const double centroid[2] = {(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0};
    const CadMetric2D m = metric(centroid);
    if (status_ != PHX_MC_OK) return false;
    const double x = b[0] - a[0], y = b[1] - a[1];
    const double u = c[0] - a[0], v = c[1] - a[1];
    const double bx = m.uu*x + m.uv*y, by = m.uv*x + m.vv*y;
    const double cx = m.uu*u + m.uv*v, cy = m.uv*u + m.vv*v;
    const double determinant = bx*cy - by*cx;
    const double bb = dot(m,x,y,x,y), cc = dot(m,u,v,u,v);
    const double dx = (bb*cy - cc*by) / (2*determinant);
    const double dy = (bx*cc - cx*bb) / (2*determinant);
    radius = std::sqrt(dot(m,dx,dy,dx,dy));
    if (!std::isfinite(radius)) { status_ = PHX_MC_UNRESOLVED; return false; }
    double lengths[3];
    const double* p[3] = {a,b,c};
    int shortest = 0;
    for (int k=0;k<3;++k) {
      const double ex=p[prev3(k)][0]-p[next3(k)][0];
      const double ey=p[prev3(k)][1]-p[next3(k)][1];
      lengths[k]=dot(m,ex,ey,ex,ey);
      if (lengths[k] < lengths[shortest]) shortest=k;
    }
    const double area2 = std::abs(x*v-y*u)*std::sqrt(m.uu*m.vv-m.uv*m.uv);
    const bool protected_angle = cell.v[shortest] < input_.point_count &&
        tri_.constraint(t,next3(shortest)) != -1 && tri_.constraint(t,prev3(shortest)) != -1;
    const bool bad_angle = !protected_angle && sine_ * std::sqrt(lengths[next3(shortest)] *
        lengths[prev3(shortest)]) > area2;
    const bool bad_size=radius > 0.70710678118654752440;
    if (size_reason) *size_reason=bad_size;
    if (angle_reason) *angle_reason=bad_angle;
    if (center) {
      center[0] = a[0]+dx; center[1] = a[1]+dy;
      if (bad_angle && input_.min_angle_degrees > 0) {
        const double* e = p[next3(shortest)];
        const double* f = p[prev3(shortest)];
        const double ex=f[0]-e[0], ey=f[1]-e[1];
        const double mx=0.5*(e[0]+f[0]), my=0.5*(e[1]+f[1]);
        // M-orthogonal direction, pointing to the opposite vertex.
        double nx=-(m.uv*ex+m.vv*ey), ny=m.uu*ex+m.uv*ey;
        if ((ex*(p[shortest][1]-my)-ey*(p[shortest][0]-mx)) * (ex*ny-ey*nx) < 0) { nx=-nx; ny=-ny; }
        const double nlen=std::sqrt(dot(m,nx,ny,nx,ny));
        const double height=0.5*std::sqrt(lengths[shortest]) /
            std::tan(input_.min_angle_degrees*(3.14159265358979323846/180.0));
        const double ox=mx+height*nx/nlen, oy=my+height*ny/nlen;
        if (dot(m,ox-mx,oy-my,ox-mx,oy-my) < dot(m,center[0]-mx,center[1]-my,center[0]-mx,center[1]-my)) {
          center[0]=ox; center[1]=oy;
        }
      }
    }
    return bad_size || bad_angle;
  }
  void enqueue(int32_t t) {
    double radius=0;
    if (geometry(t,radius,nullptr)) heap_add({radius,vertices(t),t});
  }
  bool encroaches(const double* p, const CadMetric2D& m) {
    if (segment_nodes_.empty()) return false;
    const double factor=0.5*(m.uu+m.vv)/std::sqrt(m.uu*m.vv-m.uv*m.uv);
    segment_stack_.clear();
    segment_stack_.push_back(0);
    while (!segment_stack_.empty()) {
      const int index=segment_stack_.back(); segment_stack_.pop_back();
      const SegmentNode& node=segment_nodes_[index];
      const double radius=node.max_length*factor;
      if (p[0]<node.low[0]-radius || p[0]>node.high[0]+radius ||
          p[1]<node.low[1]-radius || p[1]>node.high[1]+radius) continue;
      if (node.left>=0) {
        segment_stack_.push_back(node.left);
        segment_stack_.push_back(node.right);
      } else {
        for(int i=node.begin;i<node.end;++i) {
          const double* a=segment_endpoint(segment_order_[i],0);
          const double* b=segment_endpoint(segment_order_[i],1);
          if (dot(m,p[0]-a[0],p[1]-a[1],p[0]-b[0],p[1]-b[1])<0) return true;
        }
      }
    }
    return false;
  }
  void refuse(int64_t& counter) { ++counter; ++result_.stats.unresolved_triangles; }
  void process(int32_t t) {
    double radius, p[2];
    if (!geometry(t,radius,p) || status_ != PHX_MC_OK) return;
    if (!std::isfinite(p[0]) || !std::isfinite(p[1])) { ++result_.stats.unresolved_triangles; return; }
    std::pair<int32_t,int> blocked{-1,-1};
    const int32_t host=tri_.walk(t,p,&blocked);
    if (!inside(host)) { refuse(result_.stats.outside_refusals); return; }
    const CadMetric2D m=metric(p);
    if (status_ != PHX_MC_OK) return;
    if (encroaches(p,m)) { refuse(result_.stats.encroachment_refusals); return; }
    if (too_close(p,m)) { refuse(result_.stats.proximity_refusals); return; }
    insert(host,p,m);
  }
  bool metric_conflict(int32_t t,const double* p,const CadMetric2D& m) const {
    const Triangle2D& cell=tri_.triangles[t];
    double x[3],y[3],lift[3];
    for(int i=0;i<3;++i) {
      const double* q=tri_.point(cell.v[i]);
      x[i]=q[0]-p[0]; y[i]=q[1]-p[1];
      lift[i]=dot(m,x[i],y[i],x[i],y[i]);
    }
    return lift[0]*(x[1]*y[2]-y[1]*x[2])+
           lift[1]*(x[2]*y[0]-y[2]*x[0])+
           lift[2]*(x[0]*y[1]-y[0]*x[1]) > 0.0;
  }
  bool star_boundary(int32_t host,const double* p) {
    for (;;) {
      int32_t remove=-1;
      boundary_.clear();
      for (int32_t t : cavity_) {
        const Triangle2D cell=tri_.triangles[t];
        for (int k=0;k<3;++k) {
          const int32_t nb=cell.n[k];
          if (marks_[nb]==epoch_ && tri_.constraint(t,k)==-1) continue;
          if (marks_[nb]==epoch_) { remove=t; break; }
          const int32_t u=cell.v[next3(k)], v=cell.v[prev3(k)];
          if (orient2d(tri_.point(u),tri_.point(v),p)<=0) { remove=t; break; }
          boundary_.push_back({u,v,nb,tri_.constraint(t,k)});
        }
        if (remove>=0) break;
      }
      if (remove<0) return true;
      if (remove==host) return false;
      marks_[remove]=0;
      stack_.clear();
      stack_.push_back(host);
      marks_[host]=epoch_+1;
      while (!stack_.empty()) {
        const int32_t t=stack_.back(); stack_.pop_back();
        for (int k=0;k<3;++k) {
          const int32_t nb=tri_.triangles[t].n[k];
          if (tri_.constraint(t,k)==-1 && marks_[nb]==epoch_) {
            marks_[nb]=epoch_+1; stack_.push_back(nb);
          }
        }
      }
      cavity_.erase(std::remove_if(cavity_.begin(),cavity_.end(),[&](int32_t t) {
        if (marks_[t]==epoch_+1) { marks_[t]=epoch_; return false; }
        marks_[t]=0; return true;
      }),cavity_.end());
    }
  }
  void insert(int32_t host, const double* p,const CadMetric2D& m) {
    if (marks_.size() < tri_.triangles.size()) marks_.resize(tri_.triangles.size());
    epoch_+=2; stack_.clear(); cavity_.clear(); boundary_.clear();
    stack_.push_back(host); marks_[host]=epoch_;
    const int32_t id=tri_.vertex_count();
    while (!stack_.empty()) {
      const int32_t t=stack_.back(); stack_.pop_back(); cavity_.push_back(t);
      const Triangle2D cell=tri_.triangles[t];
      for (int k=0;k<3;++k) {
        const int32_t nb=cell.n[k];
        if (marks_[nb]==epoch_) continue;
        if (tri_.constraint(t,k)==-1 && inside(nb) && metric_conflict(nb,p,m)) {
          marks_[nb]=epoch_; stack_.push_back(nb);
        }
      }
    }
    if (!star_boundary(host,p)) { refuse(result_.stats.nonstar_refusals); return; }
    // A varying metric can produce a cavity enclosing an existing vertex.
    // Insertion may not silently remove a vertex or its classification.
    if (boundary_vertex_marks_.size()<static_cast<std::size_t>(id))
      boundary_vertex_marks_.resize(static_cast<std::size_t>(id),0);
    for (const Boundary& e : boundary_) {
      boundary_vertex_marks_[e.u]=epoch_;
      boundary_vertex_marks_[e.v]=epoch_;
    }
    for (int32_t t : cavity_) for (int32_t vertex : tri_.triangles[t].v) {
      if (boundary_vertex_marks_[vertex]!=epoch_) {
        refuse(result_.stats.nonstar_refusals);
        return;
      }
    }
    tri_.add_vertex(p);
    add_to_grid(id);
    for (int32_t t : cavity_) { heap_remove(t); tri_.release(t); }
    created_.clear();
    vertex_slots_.resize(static_cast<std::size_t>(id)+1,-1);
    for (const Boundary& e : boundary_) {
      const int32_t t=tri_.allocate();
      tri_.set_triangle(t,e.u,e.v,id,-1,-1,e.constraint);
      tri_.region[t]=1;
      tri_.link_edge(t,e.u,e.v,e.outside);
      vertex_slots_[e.u]=t;
      created_.push_back(t);
    }
    for (int32_t t : created_) {
      const Triangle2D cell=tri_.triangles[t];
      const int32_t next=vertex_slots_[cell.v[1]];
      tri_.link_edge(t,cell.v[1],id,next);
    }
    ++result_.stats.insertions;
    for (int32_t t : created_) { enqueue(t); if (status_ != PHX_MC_OK) return; }
  }
};

}  // namespace
int32_t cad_refine2d_bad_triangle(const CadRefine2DInput& input,
                                  CadRefine2DResult& result, int32_t triangle,
                                  bool& bad_size, bool& bad_angle) {
  if (!input.metric || triangle<0 ||
      static_cast<std::size_t>(triangle)>=result.triangulation.triangles.size() ||
      !(input.min_angle_degrees>=0 && input.min_angle_degrees<60))
    return PHX_MC_INVALID_ARGUMENT;
  return Refinement(input,result).classify(triangle,bad_size,bad_angle);
}


int32_t cad_refine2d(const CadRefine2DInput& input, CadRefine2DResult& result) {
  if (!input.points || input.point_count < 3 || input.segment_count < 0 || input.hole_count < 0 ||
      (input.segment_count && !input.segments) || (input.hole_count && !input.holes) ||
      !input.metric || input.point_cap < input.point_count ||
      !(input.min_angle_degrees >= 0 && input.min_angle_degrees < 60)) return PHX_MC_INVALID_ARGUMENT;
  for (int32_t i=0;i<2*input.segment_count;++i)
    if (input.segments[i]<0 || input.segments[i]>=input.point_count) return PHX_MC_INVALID_INPUT;
  result = CadRefine2DResult{};
  Triangulation2D& tri=result.triangulation;
  int32_t status=validate_points(input.points,input.point_count,2,nullptr);
  if (status!=PHX_MC_OK) return status;
  status=validate_points(input.holes,input.hole_count,2,nullptr);
  if (status!=PHX_MC_OK) return status;
  tri.reset(input.points,input.point_count,nullptr);
  status=build_delaunay_2d(tri,input.points,input.point_count,nullptr,result.vertex_map);
  if (status!=PHX_MC_OK) return status;
  ConstrainedMesher mesher(tri,input.point_count,0,std::numeric_limits<double>::infinity());
  try {
    for (int32_t i=0;i<input.segment_count;++i) {
      const int32_t a=result.vertex_map[input.segments[2*i]], b=result.vertex_map[input.segments[2*i+1]];
      if (a==b) return PHX_MC_INVALID_INPUT;
      mesher.insert_segment(a,b,i);
    }
    if (input.keep_convex_hull) mesher.constrain_hull();
    mesher.carve(input.keep_convex_hull,input.hole_count,input.holes);
  } catch (const StatusError& e) { return e.status; }
  return Refinement(input,result).run();
}

}  // namespace phx::mc
