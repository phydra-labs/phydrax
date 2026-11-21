// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_refine2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "cad_stored_size.hpp"
#include "cdt2d.hpp"
#include "predicates.hpp"
#include "refine_queue.hpp"

namespace phx::mc {
namespace {

// The geometric tie-break key is the triangle's vertices in lexicographic
// coordinate order; vertices do not move while the queue is in use, so
// entries carry vertex ids and compare their coordinates.
struct Entry {
  double radius;
  int32_t slot;
  uint32_t generation;
  std::array<int32_t,3> key;
  const std::vector<double>* coordinates;
};
struct Worse {
  bool operator()(const Entry& a, const Entry& b) const {
    if (a.radius != b.radius) return a.radius < b.radius;
    const double* c=a.coordinates->data();
    for (int k=0;k<3;++k) {
      const double* p=c+2*static_cast<std::size_t>(a.key[k]);
      const double* q=c+2*static_cast<std::size_t>(b.key[k]);
      if (p[0]!=q[0]) return p[0]>q[0];
      if (p[1]!=q[1]) return p[1]>q[1];
    }
    return false;
  }
};
using Queue = SpatialRefinementQueue<Entry, Worse>;
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
        sine_(std::sin(input.min_angle_degrees * (3.14159265358979323846 / 180.0))) {
    if (input_.fixed_vertices) {
      fixed_.assign(static_cast<std::size_t>(tri_.vertex_count()),0);
      for (int32_t i=0;i<input_.point_count;++i)
        if (input_.fixed_vertices[i]) fixed_[result_.vertex_map[i]]=1;
    }
  }


  void initialize_grid() {
    const double* first=tri_.point(0);
    low_[0]=high_[0]=first[0]; low_[1]=high_[1]=first[1];
    for (int32_t i=1;i<tri_.vertex_count();++i) {
      const double* p=tri_.point(i);
      for (int k=0;k<2;++k) { low_[k]=std::min(low_[k],p[k]); high_[k]=std::max(high_[k],p[k]); }
    }
    queue_.emplace(low_[0],low_[1],high_[0],high_[1],input_.point_cap,input_.point_count);
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
    states_.resize(tri_.triangles.size(),0);
    queued_.resize(tri_.triangles.size(),0);
    radii_.resize(tri_.triangles.size(),0);
    triangle_metrics_.resize(tri_.triangles.size());
    angle_scores_.resize(tri_.triangles.size());
    for (std::size_t t=0;t<tri_.triangles.size();++t) if (inside(static_cast<int32_t>(t))) {
      classify_state(static_cast<int32_t>(t));
      if (status_!=PHX_MC_OK) return status_;
    }
    for (std::size_t t=0;t<tri_.triangles.size();++t)
      if (inside(static_cast<int32_t>(t)) && states_[t]==2) enqueue(static_cast<int32_t>(t));
    const auto drain=[&]() {
      Entry entry;
      while (queue_->pop_current(entry,[&](const Entry& e) {
        if (e.generation!=generations_[e.slot]) return false;
        queued_[e.slot]=0;
        return inside(e.slot) && states_[e.slot]==2 && has_front(e.slot);
      })) {
        if (tri_.vertex_count() >= input_.point_cap) return false;
        states_[entry.slot]=3;
        process(entry.slot);
        if (status_ != PHX_MC_OK) return false;
      }
      return true;
    };
    bool drained=drain();
    if (status_ != PHX_MC_OK) return status_;
    // Triangles of fixed points outside the caller's region that no
    // insertion's cavity removed are refined next, under the same rules,
    // and judged by the census; until then every triangle of fixed points
    // is left alone, so meshes where none survives are unchanged.
    if (input_.fixed_region) {
      region_only_=true;
      for (std::size_t t=0;t<tri_.triangles.size();++t) {
        const auto slot=static_cast<int32_t>(t);
        if (!inside(slot) || t>=states_.size() || states_[t]!=1 || !all_fixed(slot) ||
            fixed_triangle(slot)) continue;
        classify_state(slot);
        if (status_!=PHX_MC_OK) return status_;
        if (states_[t]==2) enqueue(slot);
      }
      if (drained) drained=drain();
      if (status_ != PHX_MC_OK) return status_;
    }
    // The caller's edge splits (edge_split) come once the passes above are
    // drained: the mesh up to then is the one without the hook, and each
    // split and the refinement it brings stay beside its edge (requested
    // within the first pass, they reordered its queue, and kernel refusals
    // far from any split edge appeared: random coarse caps, UNRESOLVED).
    if (input_.edge_split && drained) {
      split_phase_=true;
      // Only a triangle asking a split changes state; the others keep the
      // classification they hold.
      for (std::size_t t=0;t<tri_.triangles.size();++t) {
        const auto slot=static_cast<int32_t>(t);
        if (!inside(slot) || t>=states_.size() || states_[t]!=1 || fixed_triangle(slot)) continue;
        const auto& cell=tri_.triangles[t];
        if (input_.edge_split(input_.edge_split_context,tri_.point(cell.v[0]),
                              tri_.point(cell.v[1]),tri_.point(cell.v[2]))==0) continue;
        classify_state(slot);
        if (status_!=PHX_MC_OK) return status_;
        if (states_[t]==2) enqueue(slot);
      }
      drain();
      if (status_ != PHX_MC_OK) return status_;
    }
    const int32_t initial=summarize();
    if (initial!=PHX_MC_OK) return initial;
    improve_worst_angles();
    if (status_!=PHX_MC_OK) return status_;
    // Moves recheck every triangle of the moved vertex's star and leave
    // fixed triangles alone, so the successful census remains valid.
    return PHX_MC_OK;
  }

  int32_t classify(int32_t t, bool& bad_size, bool& bad_angle) {
    if (!inside(t)) return PHX_MC_INVALID_ARGUMENT;
    if (fixed_triangle(t)) { bad_size=false; bad_angle=false; return PHX_MC_OK; }
    double radius=0;
    geometry(t,radius,nullptr,&bad_size,&bad_angle);
    return status_;
  }

  int32_t summarize() {
    result_.stats.unresolved_triangles=0;
    result_.stats.unresolved_size=0;
    result_.stats.unresolved_angle=0;
    result_.stats.fixed_triangles_skipped=0;
    result_.unresolved.clear();
    angle_scores_.resize(tri_.triangles.size(),INFINITY);
    triangle_metrics_.resize(tri_.triangles.size());
    for (std::size_t t=0;t<tri_.triangles.size();++t) {
      if (!inside(static_cast<int32_t>(t))) { angle_scores_[t]=INFINITY; continue; }
      if (fixed_triangle(static_cast<int32_t>(t))) {
        angle_scores_[t]=INFINITY;
        ++result_.stats.fixed_triangles_skipped;
        continue;
      }
      // A triangle classified good holds its census values already.
      if (states_[t]==1) continue;
      double radius=0.0, score=INFINITY;
      bool bad_size=false, bad_angle=false;
      const bool bad=geometry(static_cast<int32_t>(t),radius,nullptr,&bad_size,&bad_angle,nullptr,
                              &score,&triangle_metrics_[t]);
      if (status_!=PHX_MC_OK) return status_;
      angle_scores_[t]=score;
      if (bad) {
        ++result_.stats.unresolved_triangles;
        CadRefine2DUnresolved evidence;
        if (t<last_refusals_.size()) evidence=last_refusals_[t];
        evidence.triangle=static_cast<int32_t>(t);
        evidence.bad_size=bad_size;
        evidence.bad_angle=bad_angle;
        result_.unresolved.push_back(evidence);
      }
      result_.stats.unresolved_size+=bad_size;
      result_.stats.unresolved_angle+=bad_angle;
    }
    return result_.stats.unresolved_triangles ? PHX_MC_UNRESOLVED : PHX_MC_OK;
  }

 private:
  const CadRefine2DInput& input_;
  CadRefine2DResult& result_;
  Triangulation2D& tri_;
  double sine_;
  int32_t status_ = PHX_MC_OK;
  std::vector<uint8_t> fixed_;
  bool region_only_ = false;
  // Set once the passes before the caller's edge splits are drained.
  bool split_phase_ = false;
  std::vector<std::array<double,3>> physical_;
  std::vector<uint8_t> physical_valid_;
  // The angle pass's moving vertex and its candidate position, which the
  // stored size test reads in place of the cached one (-1: none).
  int32_t moving_vertex_ = -1;
  std::array<double,3> moving_xyz_{};
  // Centroid metric and metric worst-angle cosine per classified non-fixed
  // triangle (the cosine only once classified good); the census completes
  // them, and the angle pass keeps them current.
  std::vector<double> angle_scores_;
  std::vector<CadMetric2D> triangle_metrics_;
  struct StarCell { double score; CadMetric2D metric; };
  std::optional<Queue> queue_;
  std::vector<uint32_t> generations_;
  std::vector<uint8_t> states_;
  std::vector<uint8_t> queued_;
  std::vector<double> radii_;
  std::vector<CadRefine2DUnresolved> last_refusals_;
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

  bool all_fixed(int32_t t) const {
    if (fixed_.empty()) return false;
    const auto& cell=tri_.triangles[t];
    for (int k=0;k<3;++k) {
      const auto v=static_cast<std::size_t>(cell.v[k]);
      if (v>=fixed_.size() || !fixed_[v]) return false;
    }
    return true;
  }
  // A triangle of fixed points is the caller's, except, once the first
  // pass is drained (region_only_), where fixed_region declines its centroid.
  bool fixed_triangle(int32_t t) const {
    if (!all_fixed(t)) return false;
    if (!region_only_) return true;
    const auto& cell=tri_.triangles[t];
    const double* a=tri_.point(cell.v[0]); const double* b=tri_.point(cell.v[1]);
    const double* c=tri_.point(cell.v[2]);
    const double centroid[2]={(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3};
    return input_.fixed_region(input_.fixed_region_context,centroid)!=0;
  }
  bool inside(int32_t t) const {
    return t >= 0 && tri_.live[static_cast<std::size_t>(t)] &&
           tri_.region[static_cast<std::size_t>(t)] && !tri_.is_ghost(t);
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
  struct PhysicalQuality { double worst=-1; double max_edge2=0; };
  const std::array<double,3>& physical_position(int32_t vertex) {
    if(physical_valid_.size()<=static_cast<std::size_t>(vertex)) {
      physical_.resize(static_cast<std::size_t>(vertex)+1);
      physical_valid_.resize(static_cast<std::size_t>(vertex)+1,0);
    }
    if(!physical_valid_[vertex]) {
      auto& xyz=physical_[vertex];
      if(!input_.surface_position(input_.surface_context,tri_.point(vertex),xyz.data()) ||
         !std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2]))
        status_=PHX_MC_INVALID_INPUT;
      physical_valid_[vertex]=1;
    }
    return physical_[vertex];
  }
  void physical_cell(const int32_t* vertices,PhysicalQuality& quality,
                     int32_t candidate=-1,const std::array<double,3>* proposed=nullptr) {
    const std::array<double,3>* p[3];
    for(int i=0;i<3;++i)
      p[i]=(vertices[i]==candidate) ? proposed : &physical_position(vertices[i]);
    if(status_!=PHX_MC_OK) return;
    // Squared edge lengths opposite each corner; the smallest angle faces
    // the shortest edge.
    double lengths[3];
    int shortest=0;
    for(int k=0;k<3;++k) {
      double sum=0;
      for(int j=0;j<3;++j) {const double d=(*p[prev3(k)])[j]-(*p[next3(k)])[j]; sum+=d*d;}
      lengths[k]=sum;
      if(sum<lengths[shortest]) shortest=k;
    }
    if(!(lengths[shortest]>0)) {quality.worst=INFINITY; return;}
    const double a=lengths[next3(shortest)],b=lengths[prev3(shortest)];
    quality.worst=std::max(quality.worst,(a+b-lengths[shortest])/(2*std::sqrt(a*b)));
    quality.max_edge2=std::max({quality.max_edge2,lengths[0],lengths[1],lengths[2]});
  }
  PhysicalQuality physical_star(const std::vector<int32_t>& star,int32_t candidate=-1,
                                const std::array<double,3>* proposed=nullptr) {
    PhysicalQuality quality;
    for(int32_t t:star) physical_cell(tri_.triangles[t].v,quality,candidate,proposed);
    return quality;
  }
  bool physical_edges_fit(int32_t vertex,const std::vector<int32_t>& star,
                          const std::array<double,3>& proposed,double limit) {
    for(int32_t t:star) for(int32_t u:tri_.triangles[t].v) {
      if(u==vertex) continue;
      const auto& q=physical_position(u);
      double length2=0;
      for(int k=0;k<3;++k) {const double d=proposed[k]-q[k]; length2+=d*d;}
      if(length2>limit) return false;
    }
    return true;
  }
  // True when no unconstrained neighbor's opposite vertex lies inside the
  // metric circumcircle of t (m: t's centroid metric).
  bool locally_delaunay(int32_t t,const CadMetric2D& m) const {
    const auto& cell=tri_.triangles[t];
    for(int k=0;k<3;++k) {
      const int32_t nb=cell.n[k];
      if(tri_.constraint(t,k)!=-1 || !inside(nb)) continue;
      const int j=tri_.edge_index(nb,cell.v[prev3(k)],cell.v[next3(k)]);
      if(j<0) continue;
      const double* q=tri_.point(tri_.triangles[nb].v[j]);
      double x[3],y[3],lift[3];
      for(int i=0;i<3;++i) {
        const double* a=tri_.point(cell.v[i]);
        x[i]=a[0]-q[0]; y[i]=a[1]-q[1]; lift[i]=dot(m,x[i],y[i],x[i],y[i]);
      }
      if(lift[0]*(x[1]*y[2]-y[1]*x[2])+lift[1]*(x[2]*y[0]-y[2]*x[0])+
         lift[2]*(x[0]*y[1]-y[0]*x[1])>0) return false;
    }
    return true;
  }
  // Writes each star triangle's metric smallest-angle cosine; INFINITY when
  // the star is inverted or contains a triangle that fails the refinement
  // test, and, without surface positions (whose 3D guards then decide),
  // when it loses the local metric Delaunay property.
  double star_score(const std::vector<int32_t>& star,std::vector<StarCell>& cells) {
    cells.clear();
    double worst=-1;
    for(int32_t t:star) {
      const auto& cell=tri_.triangles[t];
      if (orient2d(tri_.point(cell.v[0]),tri_.point(cell.v[1]),tri_.point(cell.v[2]))<=0)
        return INFINITY;
      double radius=0, score=0;
      CadMetric2D m;
      if (geometry(t,radius,nullptr,nullptr,nullptr,&m,&score)) return INFINITY;
      if (status_!=PHX_MC_OK) return INFINITY;
      if (!input_.surface_position && !locally_delaunay(t,m)) return INFINITY;
      cells.push_back({score,m});
      worst=std::max(worst,cells.back().score);
    }
    return worst;
  }
  void improve_worst_angles() {
    const auto vertex_count=static_cast<std::size_t>(tri_.vertex_count());
    // Vertices move only here (which refreshes the cache on a move), so
    // positions the stored size test cached during refinement stay valid.
    if(input_.surface_position) {
      physical_.resize(vertex_count);
      physical_valid_.resize(vertex_count,0);
    }
    std::vector<int32_t> offsets(vertex_count+1,0);
    std::vector<uint8_t> immovable(vertex_count,0);
    std::vector<uint8_t> selected(vertex_count,0);
    // Halfway between the requested minimum and the equilateral ideal.
    const double target_cos=std::cos((input_.min_angle_degrees+60.0)*
                                     3.14159265358979323846/360.0);
    for(std::size_t t=0;t<tri_.triangles.size();++t) {
      if(!inside(static_cast<int32_t>(t))) continue;
      const auto& cell=tri_.triangles[t];
      for(int k=0;k<3;++k) {
        ++offsets[cell.v[k]+1];
        if (tri_.constraint(static_cast<int32_t>(t),k)!=-1) {
          immovable[cell.v[next3(k)]]=1; immovable[cell.v[prev3(k)]]=1;
        }
      }
      // Scores come from the final census; fixed triangles have none.
      if(!fixed_triangle(static_cast<int32_t>(t)) && angle_scores_[t]>target_cos)
        for(int32_t v:cell.v) selected[v]=1;
    }
    for(std::size_t v=0;v<vertex_count;++v) offsets[v+1]+=offsets[v];
    std::vector<int32_t> incident(static_cast<std::size_t>(offsets[vertex_count]));
    {
      std::vector<int32_t> fill(offsets.begin(),offsets.end()-1);
      for(std::size_t t=0;t<tri_.triangles.size();++t) {
        if(!inside(static_cast<int32_t>(t))) continue;
        for(int32_t v:tri_.triangles[t].v) incident[fill[v]++]=static_cast<int32_t>(t);
      }
    }
    for(std::size_t v=0;v<fixed_.size();++v) if(fixed_[v]) immovable[v]=1;
    // Worst star first, so the vertices around the worst triangles move
    // before their neighbors' moves can block them; geometric tie-break.
    struct Queued { double worst; double x, y; int32_t vertex; };
    std::vector<Queued> vertices;
    for(std::size_t v=0;v<selected.size();++v) {
      if(!selected[v] || immovable[v]) continue;
      double worst=-1;
      for(int32_t k=offsets[v];k<offsets[v+1];++k) worst=std::max(worst,angle_scores_[incident[k]]);
      const double* p=tri_.point(static_cast<int32_t>(v));
      vertices.push_back({worst,p[0],p[1],static_cast<int32_t>(v)});
    }
    std::sort(vertices.begin(),vertices.end(),[](const Queued& a,const Queued& b) {
      if(a.worst!=b.worst) return a.worst>b.worst;
      return a.x!=b.x ? a.x<b.x : a.y<b.y;
    });
    std::vector<int32_t> star,neighbors;
    std::vector<StarCell> scores,best_scores;
    struct Candidate {
      double physical;
      std::array<double,2> point;
      std::array<double,3> xyz;
      int target, fraction;
    };
    std::vector<Candidate> candidates;
    for(const auto& queued:vertices) {
      const int32_t v=queued.vertex;
      star.assign(incident.begin()+offsets[v],incident.begin()+offsets[v+1]);
      // The current star passed the census or an accepted move, so its stored
      // scores are its exact star_score. A star an earlier move already
      // repaired is left alone.
      double best=-1;
      for(int32_t t:star) best=std::max(best,angle_scores_[t]);
      if(star.empty() || best<=target_cos) continue;
      double* p=tri_.coordinates.data()+2*static_cast<std::size_t>(v);
      const double old[2]={p[0],p[1]};
      // The worst star triangle (geometric tie-break) supplies the metric,
      // already evaluated at its centroid, and the edge opposite v.
      int32_t worst_triangle=-1;
      double worst=-1;
      std::array<double,4> worst_key{};
      for(int32_t t:star) {
        const auto& cell=tri_.triangles[t]; const int k=tri_.index_of(t,v);
        const double* a=tri_.point(cell.v[next3(k)]); const double* b=tri_.point(cell.v[prev3(k)]);
        if(a[0]==b[0] && a[1]==b[1]) continue;
        std::array<double,2> x{a[0],a[1]},y{b[0],b[1]};
        if(y<x) std::swap(x,y);
        const std::array<double,4> key{x[0],x[1],y[0],y[1]};
        const double score=angle_scores_[t];
        if(score<worst || (score==worst && !(key<worst_key))) continue;
        worst=score; worst_key=key; worst_triangle=t;
      }
      if(worst_triangle<0) continue;
      const CadMetric2D m=triangle_metrics_[worst_triangle];
      neighbors.clear();
      for(int32_t t:star) for(int32_t u:tri_.triangles[t].v)
        if(u!=v) neighbors.push_back(u);
      std::sort(neighbors.begin(),neighbors.end(),[&](int32_t a,int32_t b) {
        const double* x=tri_.point(a); const double* y=tri_.point(b);
        return x[0]!=y[0] ? x[0]<y[0] : x[1]<y[1];
      });
      neighbors.erase(std::unique(neighbors.begin(),neighbors.end()),neighbors.end());
      double weighted[2]={0,0},sum=0,step=INFINITY;
      for(int32_t u:neighbors) {
        const double* q=tri_.point(u);
        const double dx=q[0]-old[0],dy=q[1]-old[1];
        const double length=std::sqrt(dot(m,dx,dy,dx,dy));
        step=std::min(step,std::hypot(dx,dy));
        if(!(length>0)) continue;
        const double weight=1/length;
        weighted[0]+=weight*q[0]; weighted[1]+=weight*q[1]; sum+=weight;
      }
      if(!(sum>0)) continue;
      // Targets: the metric-weighted neighbor centroid and the apex of the
      // metric-equilateral triangle on the worst triangle's opposite edge;
      // then, as a fallback, eight chart directions at a quarter of the
      // shortest incident chart edge.
      double targets[10][2]={{weighted[0]/sum,weighted[1]/sum},{}};
      {
        const auto& cell=tri_.triangles[worst_triangle]; const int k=tri_.index_of(worst_triangle,v);
        const double* a=tri_.point(cell.v[next3(k)]); const double* b=tri_.point(cell.v[prev3(k)]);
        const double ex=b[0]-a[0],ey=b[1]-a[1];
        double nx=-(m.uv*ex+m.vv*ey),ny=m.uu*ex+m.uv*ey;
        const double norm=std::sqrt(dot(m,nx,ny,nx,ny));
        const double mx=(a[0]+b[0])*0.5,my=(a[1]+b[1])*0.5;
        if((old[0]-mx)*nx+(old[1]-my)*ny<0) {nx=-nx;ny=-ny;}
        const double height=std::sqrt(0.75*dot(m,ex,ey,ex,ey))/norm;
        targets[1][0]=mx+height*nx; targets[1][1]=my+height*ny;
      }
      step*=0.25;
      int target_count=2;
      for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) {
        if(dx==0 && dy==0) continue;
        targets[target_count][0]=old[0]+dx*step;
        targets[target_count][1]=old[1]+dy*step;
        ++target_count;
      }
      PhysicalQuality current_physical;
      if(input_.surface_position) current_physical=physical_star(star);
      if(status_!=PHX_MC_OK) return;
      double best_point[2]={old[0],old[1]};
      std::array<double,3> best_xyz{};
      // Steps 1, 1/2, 1/4 toward a target. With positions, the first step
      // from `fraction` on that passes the 3D guards (which need no metric
      // evaluation) becomes the candidate; returns false on none or error.
      constexpr double fractions[3]={1.0,0.5,0.25};
      auto physical_candidate=[&](Candidate& candidate) {
        const auto& target=targets[candidate.target];
        for(;candidate.fraction<3;++candidate.fraction) {
          const double fraction=fractions[candidate.fraction];
          candidate.point={old[0]+fraction*(target[0]-old[0]),old[1]+fraction*(target[1]-old[1])};
          if(input_.admit_point && !input_.admit_point(input_.admit_context,candidate.point.data()))
            continue;
          auto& xyz=candidate.xyz;
          if(!input_.surface_position(input_.surface_context,candidate.point.data(),xyz.data()) ||
             !std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) {
            status_=PHX_MC_INVALID_INPUT; return false;
          }
          // Edges not incident to v keep their length.
          if(!physical_edges_fit(v,star,xyz,current_physical.max_edge2)) continue;
          const auto quality=physical_star(star,v,&xyz);
          if(status_!=PHX_MC_OK) return false;
          if(quality.worst>current_physical.worst) continue;
          candidate.physical=quality.worst;
          return true;
        }
        return false;
      };
      for(int phase=0;phase<2;++phase) {
        if(phase==1 &&
           (best<=target_cos || best_point[0]!=old[0] || best_point[1]!=old[1])) break;
        const int first=phase ? 2 : 0, last=phase ? target_count : 2;
        if(!input_.surface_position) {
          // Without positions, each target's longest improving step competes.
          for(int target_index=first;target_index<last;++target_index) {
            const auto& target=targets[target_index];
            for(double fraction:fractions) {
              p[0]=old[0]+fraction*(target[0]-old[0]); p[1]=old[1]+fraction*(target[1]-old[1]);
              if(input_.admit_point && !input_.admit_point(input_.admit_context,p)) continue;
              const double score=star_score(star,scores);
              if(status_!=PHX_MC_OK) return;
              if(!(score<best)) continue;
              best=score; best_point[0]=p[0]; best_point[1]=p[1];
              best_scores.swap(scores);
              break;
            }
          }
          continue;
        }
        candidates.clear();
        for(int target_index=first;target_index<last;++target_index) {
          // The fallback directions are probed at their shortest step only.
          Candidate candidate{-1,{},{},target_index,phase ? 2 : 0};
          const bool found=physical_candidate(candidate);
          if(status_!=PHX_MC_OK) return;
          if(!found) continue;
          candidates.push_back(candidate);
          // A candidate whose 3D star passes the selection test ends probing.
          if(candidate.physical<=target_cos) break;
        }
        // The better 3D star goes first; the move needs metric improvement
        // too, and a rejected candidate falls back to its target's next step.
        std::stable_sort(candidates.begin(),candidates.end(),
                         [](const Candidate& a,const Candidate& b){return a.physical<b.physical;});
        bool moved=false;
        for(auto candidate:candidates) {
          do {
            p[0]=candidate.point[0]; p[1]=candidate.point[1];
            moving_vertex_=v; moving_xyz_=candidate.xyz;
            const double score=star_score(star,scores);
            moving_vertex_=-1;
            if(status_!=PHX_MC_OK) return;
            if(score<best) {
              best=score; best_point[0]=p[0]; best_point[1]=p[1]; best_xyz=candidate.xyz;
              best_scores.swap(scores);
              moved=true;
              break;
            }
            ++candidate.fraction;
          } while(physical_candidate(candidate));
          if(status_!=PHX_MC_OK) return;
          if(moved) break;
        }
      }
      p[0]=best_point[0]; p[1]=best_point[1];
      if(p[0]==old[0] && p[1]==old[1]) continue;
      for(std::size_t k=0;k<star.size();++k) {
        angle_scores_[star[k]]=best_scores[k].score;
        triangle_metrics_[star[k]]=best_scores[k].metric;
      }
      if(input_.surface_position) {physical_[v]=best_xyz; physical_valid_[v]=1;}
    }
  }
  // The metric length of chart segment pq, integrated along it (two-point
  // Gauss-Legendre at canonical endpoints); NaN with status_ set when a
  // metric evaluation fails. The nodes are interior, so an edge ending at a
  // chart pole is measured where the metric exists.
  double integrated_length(const double* p,const double* q) {
    if (q[0]<p[0] || (q[0]==p[0] && q[1]<p[1])) std::swap(p,q);
    const double ex=q[0]-p[0], ey=q[1]-p[1];
    constexpr double node=0.21132486540518711775;  // (1 - 1/sqrt(3)) / 2
    double total=0;
    for (const double s : {node, 1-node}) {
      const double x[2]={p[0]+s*ex, p[1]+s*ey};
      const CadMetric2D mx=metric(x);
      if (status_!=PHX_MC_OK) return std::numeric_limits<double>::quiet_NaN();
      total+=0.5*std::sqrt(dot(mx,ex,ey,ex,ey));
    }
    return total;
  }
  // The stored size test (cad_stored_size.hpp) on the surface positions of
  // t's vertices against input_.stored_size, at half the rounding allowance
  // so that compliance, computing the allowance in its own rounding,
  // accepts every triangle the kernel does. A triangle on a constrained
  // side of stored length >= sqrt(2) h is exempt: its circumradius is at
  // least half that side and no insertion shortens it.
  bool stored_oversized(int32_t t) {
    if (!(input_.stored_size > 0.0) || !input_.surface_position) return false;
    const auto& cell=tri_.triangles[t];
    std::array<double,3> xyz[3];
    for (int k=0;k<3;++k) {
      xyz[k]=cell.v[k]==moving_vertex_ ? moving_xyz_ : physical_position(cell.v[k]);
      if (status_!=PHX_MC_OK) return false;
    }
    const double h=input_.stored_size;
    for (int k=0;k<3;++k) {
      if (tri_.constraint(t,k)==-1) continue;
      const auto& a=xyz[next3(k)];
      const auto& b=xyz[prev3(k)];
      double length2=0;
      for (int i=0;i<3;++i) length2+=(b[i]-a[i])*(b[i]-a[i]);
      if (length2>=2*h*h) return false;
    }
    return stored_size_fails(xyz[0].data(),xyz[1].data(),xyz[2].data(),h);
  }
  // Confirms a centroid-metric oversize with edge-integrated metric lengths:
  // the circumradius of the triangle with those side lengths against
  // sqrt(1/2) of the larger of one unit and each fixed side's length. Where
  // the chart metric varies across a triangle (a latitude chart: cos(v)
  // along u), the centroid metric misstates sides that run far from the
  // centroid; the band between a sphere's pole rings and a trim circle at
  // h/r 0.8 held a triangle of physical circumradius 0.64 h judged 0.71 h
  // whose every split point lay in a trim chord's lens or by a ring point.
  bool integrated_oversized(int32_t t) {
    const auto& cell=tri_.triangles[t];
    double side[3], limit=0.70710678118654752440;
    for (int k=0;k<3;++k) {
      side[k]=integrated_length(tri_.point(cell.v[next3(k)]),tri_.point(cell.v[prev3(k)]));
      if (status_!=PHX_MC_OK) return false;
      if (tri_.constraint(t,k)!=-1) limit=std::max(limit,0.70710678118654752440*side[k]);
    }
    const double heron=(side[0]+side[1]+side[2])*(side[1]+side[2]-side[0])*
                       (side[0]+side[2]-side[1])*(side[0]+side[1]-side[2]);
    return !(heron>0) || side[0]*side[1]*side[2] > limit*std::sqrt(heron);
  }
  // angle_cosine, when given, receives the metric cosine of the smallest
  // angle (not written on the early oversized exit). A known centroid metric
  // skips the callback.
  bool geometry(int32_t t, double& radius, double* center,
                bool* size_reason=nullptr, bool* angle_reason=nullptr, CadMetric2D* center_metric=nullptr,
                double* angle_cosine=nullptr, const CadMetric2D* known_metric=nullptr) {
    const Triangle2D& cell = tri_.triangles[static_cast<std::size_t>(t)];
    // Canonical geometric rotation keeps floating-point construction independent
    // of the triangle's cyclic vertex numbering.
    int first=0;
    for (int i=1;i<3;++i) {
      const double* candidate=tri_.point(cell.v[i]);
      const double* best=tri_.point(cell.v[first]);
      if (candidate[0]<best[0] || (candidate[0]==best[0] && candidate[1]<best[1])) first=i;
    }
    const double* a = tri_.point(cell.v[first]);
    const double* b = tri_.point(cell.v[next3(first)]);
    const double* c = tri_.point(cell.v[prev3(first)]);
    const double centroid[2] = {(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0};
    const CadMetric2D m = known_metric ? *known_metric : metric(centroid);
    if (status_ != PHX_MC_OK) return false;
    if (center_metric) *center_metric=m;
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
    double size_limit=0.70710678118654752440;
    if (radius > size_limit) for (int k=0;k<3;++k) {
      if (tri_.constraint(t,k)==-1) continue;
      const double* e0=tri_.point(cell.v[next3(k)]); const double* e1=tri_.point(cell.v[prev3(k)]);
      const double ex=e1[0]-e0[0], ey=e1[1]-e0[1];
      size_limit=std::max(size_limit,0.70710678118654752440*std::sqrt(dot(m,ex,ey,ex,ey)));
    }
    // Refinement splits by the centroid test alone; the final verdict (a
    // caller asking for reasons: the census and classify) also asks the
    // side-integrated lengths, so a triangle the kernel could not split is
    // accepted when they clear it. Splitting every centroid-oversized
    // triangle it can keeps the parent's meshes where the kernel resolves
    // them, and spends no metric evaluations on the confirmation there.
    bool oversized = radius > size_limit && (!size_reason || integrated_oversized(t));
    if (status_ != PHX_MC_OK) return false;
    // The stored and caller edge-split tests split (refinement, angle-pass
    // guard) but take no part in the final verdict: the face stage's flips
    // can resolve a stored failure the kernel could not split, and the face
    // stage checks the emitted triangles itself.
    if (!oversized && !size_reason) oversized = stored_oversized(t);
    if (!oversized && !size_reason && split_phase_)
      oversized = input_.edge_split(input_.edge_split_context, a, b, c) != 0;
    if (status_ != PHX_MC_OK) return false;
    if (!size_reason && !angle_reason && oversized) {
      if (center) { center[0]=a[0]+dx; center[1]=a[1]+dy; }
      return true;
    }
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
    const int original_shortest=(first+shortest)%3;
    const bool protected_angle = cell.v[original_shortest] < input_.point_count &&
        tri_.constraint(t,next3(original_shortest)) != -1 && tri_.constraint(t,prev3(original_shortest)) != -1;
    const double side_product=std::sqrt(lengths[next3(shortest)]*lengths[prev3(shortest)]);
    if (angle_cosine)
      *angle_cosine=(lengths[next3(shortest)]+lengths[prev3(shortest)]-lengths[shortest])/
                    (2*side_product);
    const bool bad_angle = !protected_angle && sine_ * side_product > area2;
    const bool bad_size=oversized;
    if (size_reason) *size_reason=bad_size;
    if (angle_reason) *angle_reason=bad_angle;
    if (center) { center[0]=a[0]+dx; center[1]=a[1]+dy; }
    return bad_size || bad_angle;
  }
  void classify_state(int32_t t) {
    if (states_.size()<=static_cast<std::size_t>(t)) {
      states_.resize(static_cast<std::size_t>(t)+1,0);
      queued_.resize(static_cast<std::size_t>(t)+1,0);
      radii_.resize(static_cast<std::size_t>(t)+1,0);
      triangle_metrics_.resize(static_cast<std::size_t>(t)+1);
      angle_scores_.resize(static_cast<std::size_t>(t)+1);
    }
    queued_[t]=0;
    if (fixed_triangle(t)) { states_[t]=1; return; }
    states_[t]=geometry(t,radii_[t],nullptr,nullptr,nullptr,&triangle_metrics_[t],&angle_scores_[t]) ? 2 : 1;
  }
  bool has_front(int32_t t) const {
    const auto& cell=tri_.triangles[t];
    for(int k=0;k<3;++k) {
      const int32_t nb=cell.n[k];
      if(tri_.constraint(t,k)!=-1 || !inside(nb) ||
         (static_cast<std::size_t>(nb)<states_.size() && states_[nb]==1)) return true;
    }
    return false;
  }
  int front_edge(int32_t t) const {
    const auto& cell=tri_.triangles[t];
    int best=-1;
    std::array<double,4> best_key{};
    for(int k=0;k<3;++k) {
      const int32_t nb=cell.n[k];
      if (tri_.constraint(t,k)==-1 && inside(nb) &&
          (static_cast<std::size_t>(nb)>=states_.size() || states_[nb]!=1)) continue;
      const double* a=tri_.point(cell.v[next3(k)]);
      const double* b=tri_.point(cell.v[prev3(k)]);
      std::array<double,2> x{a[0],a[1]},y{b[0],b[1]};
      if(y<x) std::swap(x,y);
      const std::array<double,4> key{x[0],x[1],y[0],y[1]};
      if(best<0 || key<best_key) { best=k; best_key=key; }
    }
    return best;
  }
  void frontal_point(int32_t t,double* center,const CadMetric2D& m) {
    const int edge=front_edge(t);
    const auto& cell=tri_.triangles[t];
    const double* a=tri_.point(cell.v[next3(edge)]);
    const double* b=tri_.point(cell.v[prev3(edge)]);
    const double ex=b[0]-a[0],ey=b[1]-a[1];
    const double mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2;
    double nx=-(m.uv*ex+m.vv*ey),ny=m.uu*ex+m.uv*ey;
    const double nlen=std::sqrt(dot(m,nx,ny,nx,ny));
    nx/=nlen; ny/=nlen;
    const double edge2=dot(m,ex,ey,ex,ey);
    const double ideal=std::sqrt(std::max(0.0,1.0-edge2/4));
    const double bound=std::abs(dot(m,center[0]-mx,center[1]-my,nx,ny));
    const double height=std::min(ideal,bound);
    center[0]=mx+height*nx; center[1]=my+height*ny;
  }
  // Callers skip triangles that are not bad or already queued.
  void enqueue(int32_t t) {
    if (!has_front(t)) return;
    const double radius=radii_[t];
    if (generations_.size()<=static_cast<std::size_t>(t)) generations_.resize(static_cast<std::size_t>(t)+1,0);
    const auto& cell=tri_.triangles[t];
    std::array<int32_t,3> key={cell.v[0],cell.v[1],cell.v[2]};
    auto less=[&](int32_t u,int32_t v) {
      const double* p=tri_.point(u); const double* q=tri_.point(v);
      return p[0]<q[0] || (p[0]==q[0] && p[1]<q[1]);
    };
    // Geometric ties remain stable when input vertex numbering changes.
    if (less(key[1],key[0])) std::swap(key[0],key[1]);
    if (less(key[2],key[1])) std::swap(key[1],key[2]);
    if (less(key[1],key[0])) std::swap(key[0],key[1]);
    const double* p0=tri_.point(key[0]); const double* p1=tri_.point(key[1]);
    const double* p2=tri_.point(key[2]);
    queued_[t]=1;
    queue_->push({radius,t,++generations_[t],key,&tri_.coordinates},
                 (p0[0]+p1[0]+p2[0])/3,(p0[1]+p1[1]+p2[1])/3);
  }
  int32_t encroaches(const double* p, const CadMetric2D& m) {
    if (segment_nodes_.empty()) return -1;
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
          if (dot(m,p[0]-a[0],p[1]-a[1],p[0]-b[0],p[1]-b[1])<0) return segment_order_[i];
        }
      }
    }
    return -1;
  }
  void refuse(int64_t& counter,CadRefine2DRefusal kind,int32_t t,const double* p,int32_t segment=-1) {
    ++counter;
    ++result_.stats.unresolved_triangles;
    if (last_refusals_.size()<=static_cast<std::size_t>(t))
      last_refusals_.resize(static_cast<std::size_t>(t)+1);
    auto& evidence=last_refusals_[t];
    evidence.refusal=kind;
    evidence.candidate[0]=p[0]; evidence.candidate[1]=p[1];
    evidence.segment=segment;
    evidence.relocation_failed=kind==CadRefine2DRefusal::encroachment ||
                               kind==CadRefine2DRefusal::proximity;
  }
  bool relocate_ray(int32_t t,const double* p,const CadMetric2D& m,int32_t segment) {
    const double* a=segment_endpoint(segment,0);
    const double* b=segment_endpoint(segment,1);
    const double mx=(a[0]+b[0])/2,my=(a[1]+b[1])/2;
    const double dx=p[0]-mx,dy=p[1]-my;
    const double ex=b[0]-a[0],ey=b[1]-a[1];
    const double d2=dot(m,dx,dy,dx,dy);
    if (!(d2>0)) return false;
    auto sample=[&](double scale,double* q,CadMetric2D& mq,int32_t& host,double& lens) {
      q[0]=mx+scale*dx; q[1]=my+scale*dy;
      if (input_.admit_point && !input_.admit_point(input_.admit_context,q)) return false;
      std::pair<int32_t,int> blocked{-1,-1};
      host=tri_.walk(t,q,&blocked);
      if (!inside(host)) return false;
      mq=metric(q);
      if (status_!=PHX_MC_OK) return false;
      lens=dot(mq,q[0]-a[0],q[1]-a[1],q[0]-b[0],q[1]-b[1]);
      return true;
    };
    double low=1, high=std::nextafter(std::sqrt(dot(m,ex,ey,ex,ey)/(4*d2)),
                                      std::numeric_limits<double>::infinity());
    double q[2], lens=0; CadMetric2D mq; int32_t host=-1;
    if (!(high>low)) high=std::nextafter(low,std::numeric_limits<double>::infinity());
    if (!sample(high,q,mq,host,lens)) return false;
    for(int i=0;lens<0 && i<20;++i) {
      high=1+2*(high-1);
      if (!sample(high,q,mq,host,lens)) return false;
    }
    if (lens<0) return false;
    for(int i=0;i<52;++i) {
      const double mid=(low+high)/2;
      if (!(mid>low && mid<high)) break;
      double trial[2], value; CadMetric2D trial_metric; int32_t trial_host;
      if (!sample(mid,trial,trial_metric,trial_host,value)) return false;
      if (value<0) low=mid;
      else { high=mid; q[0]=trial[0]; q[1]=trial[1]; mq=trial_metric; host=trial_host; }
    }
    if (encroaches(q,mq)>=0 || too_close(q,mq)) return false;
    return insert(host,q,mq,t,true);
  }
  bool relocate_offcenter(int32_t t,const CadMetric2D& m,int32_t segment) {
    const auto& cell=tri_.triangles[t];
    const double* sa=segment_endpoint(segment,0); const double* sb=segment_endpoint(segment,1);
    const double sx=(sa[0]+sb[0])/2,sy=(sa[1]+sb[1])/2;
    int edge=-1; double best=-1;
    std::array<double,4> bestkey{};
    for(int k=0;k<3;++k) {
      const double* a=tri_.point(cell.v[next3(k)]);
      const double* b=tri_.point(cell.v[prev3(k)]);
      if ((a==sa && b==sb) || (a==sb && b==sa)) continue;
      const double x=(a[0]+b[0])/2-sx,y=(a[1]+b[1])/2-sy;
      const double distance=dot(m,x,y,x,y);
      std::array<double,2> u{a[0],a[1]},v{b[0],b[1]};
      if(v<u) std::swap(u,v);
      const std::array<double,4> key{u[0],u[1],v[0],v[1]};
      if(edge<0 || distance>best || (distance==best && key<bestkey)) {
        edge=k; best=distance; bestkey=key;
      }
    }
    if(edge<0) return false;
    const double* a=tri_.point(cell.v[next3(edge)]);
    const double* b=tri_.point(cell.v[prev3(edge)]);
    const double ex=b[0]-a[0],ey=b[1]-a[1];
    const double nx=-(m.uv*ex+m.vv*ey),ny=m.uu*ex+m.uv*ey;
    const double nlen=std::sqrt(dot(m,nx,ny,nx,ny));
    if (!(nlen>0)) return false;
    // With no angle target, use the equilateral offcenter.
    const double angle=input_.min_angle_degrees>0 ? input_.min_angle_degrees : 30;
    const double height=0.5*std::sqrt(dot(m,ex,ey,ex,ey))/
        std::tan(angle*(3.14159265358979323846/180));
    const double q[2]={(a[0]+b[0])/2+height*nx/nlen,(a[1]+b[1])/2+height*ny/nlen};
    if (input_.admit_point && !input_.admit_point(input_.admit_context,q)) return false;
    std::pair<int32_t,int> blocked{-1,-1};
    const int32_t host=tri_.walk(t,q,&blocked);
    if (!inside(host)) return false;
    const auto mq=metric(q);
    if (status_!=PHX_MC_OK || encroaches(q,mq)>=0 || too_close(q,mq)) return false;
    return insert(host,q,mq,t,true);
  }
  // Inserts t's circumcenter under the same admission, location, lens and
  // proximity checks as any candidate.
  bool insert_circumcenter(int32_t t,const double* circumcenter) {
    if (input_.admit_point && !input_.admit_point(input_.admit_context,circumcenter)) return false;
    std::pair<int32_t,int> blocked{-1,-1};
    const int32_t host=tri_.walk(t,circumcenter,&blocked);
    if (!inside(host)) return false;
    const CadMetric2D m=metric(circumcenter);
    return status_==PHX_MC_OK && encroaches(circumcenter,m)<0 && !too_close(circumcenter,m) &&
           insert(host,circumcenter,m,t,true);
  }
  // Inserts the midpoint of t's longest unconstrained edge (metric m) under
  // the same checks. A varying metric can leave a vertex inside t's metric
  // circumdisk near its circumcenter; the edge midpoint then still splits t.
  bool insert_edge_midpoint(int32_t t,const CadMetric2D& m) {
    const auto& cell=tri_.triangles[t];
    int edge=-1; double best=-1;
    std::array<double,4> bestkey{};
    for(int k=0;k<3;++k) {
      if (tri_.constraint(t,k)!=-1) continue;
      const double* a=tri_.point(cell.v[next3(k)]);
      const double* b=tri_.point(cell.v[prev3(k)]);
      const double length=dot(m,b[0]-a[0],b[1]-a[1],b[0]-a[0],b[1]-a[1]);
      std::array<double,2> u{a[0],a[1]},v{b[0],b[1]};
      if(v<u) std::swap(u,v);
      const std::array<double,4> key{u[0],u[1],v[0],v[1]};
      if(edge<0 || length>best || (length==best && key<bestkey)) {
        edge=k; best=length; bestkey=key;
      }
    }
    if(edge<0) return false;
    const double* a=tri_.point(cell.v[next3(edge)]);
    const double* b=tri_.point(cell.v[prev3(edge)]);
    const double q[2]={(a[0]+b[0])/2,(a[1]+b[1])/2};
    if (input_.admit_point && !input_.admit_point(input_.admit_context,q)) return false;
    std::pair<int32_t,int> blocked{-1,-1};
    const int32_t host=tri_.walk(t,q,&blocked);
    if (!inside(host)) return false;
    const CadMetric2D mq=metric(q);
    return status_==PHX_MC_OK && encroaches(q,mq)<0 && !too_close(q,mq) &&
           insert(host,q,mq,t,true);
  }
  // The metric circumradius of the counterclockwise chart triangle abc at
  // its centroid metric, rotated canonically as in geometry(); NaN with
  // status_ set when the metric evaluation fails.
  double metric_radius(int32_t va,int32_t vb,int32_t vc) {
    const int32_t ids[3]={va,vb,vc};
    int first=0;
    for (int i=1;i<3;++i) {
      const double* candidate=tri_.point(ids[i]);
      const double* best=tri_.point(ids[first]);
      if (candidate[0]<best[0] || (candidate[0]==best[0] && candidate[1]<best[1])) first=i;
    }
    const double* a=tri_.point(ids[first]);
    const double* b=tri_.point(ids[next3(first)]);
    const double* c=tri_.point(ids[prev3(first)]);
    const double centroid[2]={(a[0]+b[0]+c[0])/3.0,(a[1]+b[1]+c[1])/3.0};
    const CadMetric2D m=metric(centroid);
    if (status_!=PHX_MC_OK) return std::numeric_limits<double>::quiet_NaN();
    const double x=b[0]-a[0], y=b[1]-a[1];
    const double u=c[0]-a[0], v=c[1]-a[1];
    const double bx=m.uu*x+m.uv*y, by=m.uv*x+m.vv*y;
    const double cx=m.uu*u+m.uv*v, cy=m.uv*u+m.vv*v;
    const double determinant=bx*cy-by*cx;
    const double bb=dot(m,x,y,x,y), cc=dot(m,u,v,u,v);
    const double dx=(bb*cy-cc*by)/(2*determinant);
    const double dy=(bx*cc-cx*bb)/(2*determinant);
    return std::sqrt(dot(m,dx,dy,dx,dy));
  }
  // A triangle whose every candidate point was refused (encroachment or
  // proximity) loses its unconstrained edge to a flip when the pair that
  // replaces it and its neighbour has a smaller larger metric circumradius
  // than the pair it replaces (the smallest such, ties by the edge's
  // canonical key). Insertion cavities judge conflict in the new point's
  // metric and the size test in each triangle's centroid metric, so a
  // varying metric leaves triangles that are not Delaunay in their own
  // (measured, quadric h 0.0998 theta 0.8: the neighbour's vertex 0.43 from
  // the circumcenter of a triangle of metric circumradius 0.78, both other
  // candidates within 0.5 of a vertex and the edge midpoint encroaching the
  // curve segment; the flip leaves both triangles under the bound). Each
  // flip lowers the descending sequence of radii lexicographically on a
  // fixed vertex set, so flips between insertions are finite.
  bool flip_refused(int32_t t) {
    const Triangle2D cell=tri_.triangles[t];
    int best_edge=-1, best_twin=0;
    double best=0;
    std::array<double,4> best_key{};
    for (int k=0;k<3;++k) {
      const int32_t nb=cell.n[k];
      if (tri_.constraint(t,k)!=-1 || !inside(nb) || fixed_triangle(nb)) continue;
      const int32_t p=cell.v[k], u=cell.v[next3(k)], w=cell.v[prev3(k)];
      const int j=tri_.edge_index(nb,w,u);
      if (j<0) continue;
      const int32_t q=tri_.triangles[nb].v[j];
      if (orient2d(tri_.point(p),tri_.point(u),tri_.point(q))<=0 ||
          orient2d(tri_.point(p),tri_.point(q),tri_.point(w))<=0) continue;
      const double before=std::max(metric_radius(p,u,w),metric_radius(q,w,u));
      const double after=std::max(metric_radius(p,u,q),metric_radius(p,q,w));
      if (status_!=PHX_MC_OK) return false;
      if (!(after<before)) continue;
      std::array<double,2> x{tri_.point(u)[0],tri_.point(u)[1]},y{tri_.point(w)[0],tri_.point(w)[1]};
      if (y<x) std::swap(x,y);
      const std::array<double,4> key{x[0],x[1],y[0],y[1]};
      if (best_edge<0 || after<best || (after==best && key<best_key)) {
        best_edge=k; best_twin=j; best=after; best_key=key;
      }
    }
    if (best_edge<0) return false;
    const int k=best_edge;
    const int32_t nb=cell.n[k];
    const Triangle2D other=tri_.triangles[nb];
    const int32_t p=cell.v[k], u=cell.v[next3(k)], w=cell.v[prev3(k)];
    const int j=best_twin;
    const int32_t q=other.v[j];
    // Outer edges: p->u and w->p of t, u->q and q->w of nb.
    const int32_t n_pu=cell.n[prev3(k)], n_wp=cell.n[next3(k)];
    const int32_t n_uq=other.n[next3(j)], n_qw=other.n[prev3(j)];
    const int32_t c_pu=tri_.constraint(t,prev3(k)), c_wp=tri_.constraint(t,next3(k));
    const int32_t c_uq=tri_.constraint(nb,next3(j)), c_qw=tri_.constraint(nb,prev3(j));
    for (const int32_t s : {t,nb}) {
      if (static_cast<std::size_t>(s)<generations_.size()) ++generations_[s];
      if (static_cast<std::size_t>(s)<last_refusals_.size()) last_refusals_[s]=CadRefine2DUnresolved{};
      tri_.release(s);
    }
    const int32_t first=tri_.allocate(), second=tri_.allocate();
    tri_.set_triangle(first,p,u,q,c_uq,-1,c_pu);
    tri_.set_triangle(second,p,q,w,c_qw,c_wp,-1);
    tri_.region[first]=tri_.region[second]=1;
    tri_.link_edge(first,p,u,n_pu);
    tri_.link_edge(first,u,q,n_uq);
    tri_.link_edge(second,q,w,n_qw);
    tri_.link_edge(second,w,p,n_wp);
    tri_.link_edge(first,q,p,second);
    ++result_.stats.flips;
    const std::pair<int32_t,int32_t> outer[4]={{first,n_pu},{first,n_uq},{second,n_qw},{second,n_wp}};
    for (const int32_t s : {first,second}) { classify_state(s); if (status_!=PHX_MC_OK) return true; }
    // As after an insertion: a new triangle and a processed neighbour that
    // gains a front from it are enqueued.
    for (const int32_t s : {first,second}) if (states_[s]==2 && !queued_[s]) enqueue(s);
    for (const auto& [s,nb_outer] : outer) {
      if (!inside(nb_outer) || static_cast<std::size_t>(nb_outer)>=states_.size()) continue;
      if (states_[nb_outer]==3 && states_[s]==1) states_[nb_outer]=2;
      if (states_[nb_outer]==2 && !queued_[nb_outer]) enqueue(nb_outer);
    }
    return true;
  }
  // The caller's requested edge (edge_split): its midpoint, when the edge is
  // unconstrained, under the same checks as any candidate.
  bool split_requested_edge(int32_t t) {
    const auto& cell=tri_.triangles[t];
    const double* p[3]={tri_.point(cell.v[0]),tri_.point(cell.v[1]),tri_.point(cell.v[2])};
    const int32_t edge=input_.edge_split(input_.edge_split_context,p[0],p[1],p[2])-1;
    // The edge from point k to k + 1 lies opposite point k + 2.
    if (edge<0 || edge>2 || tri_.constraint(t,prev3(edge))!=-1) return false;
    const double* a=p[edge]; const double* b=p[next3(edge)];
    const double q[2]={(a[0]+b[0])/2,(a[1]+b[1])/2};
    if (input_.admit_point && !input_.admit_point(input_.admit_context,q)) return false;
    std::pair<int32_t,int> blocked{-1,-1};
    const int32_t host=tri_.walk(t,q,&blocked);
    if (!inside(host)) return false;
    const CadMetric2D mq=metric(q);
    return status_==PHX_MC_OK && encroaches(q,mq)<0 && !too_close(q,mq) &&
           insert(host,q,mq,t,true);
  }
  void process(int32_t t) {
    double radius, p[2];
    CadMetric2D center_metric;
    if (!geometry(t,radius,p,nullptr,nullptr,&center_metric,nullptr,&triangle_metrics_[t]) ||
        status_ != PHX_MC_OK) return;
    const double circumcenter[2]={p[0],p[1]};
    if (split_phase_ && split_requested_edge(t)) return;
    if (status_ != PHX_MC_OK) return;
    frontal_point(t,p,center_metric);
    if (!std::isfinite(p[0]) || !std::isfinite(p[1])) { ++result_.stats.unresolved_triangles; return; }
    if (input_.admit_point && !input_.admit_point(input_.admit_context,p)) {
      refuse(result_.stats.excluded_refusals,CadRefine2DRefusal::excluded,t,p);
      return;
    }
    std::pair<int32_t,int> blocked{-1,-1};
    const int32_t host=tri_.walk(t,p,&blocked);
    if (!inside(host)) { refuse(result_.stats.outside_refusals,CadRefine2DRefusal::outside,t,p); return; }
    const CadMetric2D m=metric(p);
    if (status_ != PHX_MC_OK) return;
    const int32_t segment=encroaches(p,m);
    if (segment>=0) {
      ++result_.stats.relocations_tried;
      if (relocate_ray(t,p,m,segment) ||
          (status_==PHX_MC_OK && relocate_offcenter(t,m,segment))) {
        ++result_.stats.relocations_accepted;
        return;
      }
      if (status_==PHX_MC_OK && insert_circumcenter(t,circumcenter)) {
        ++result_.stats.relocations_accepted;
        return;
      }
      ++result_.stats.relocations_failed;
      if (status_!=PHX_MC_OK || flip_refused(t) || status_!=PHX_MC_OK) return;
      refuse(result_.stats.encroachment_refusals,CadRefine2DRefusal::encroachment,t,p,segment);
      return;
    }
    if (too_close(p,m)) {
      // A frontal point too close to a vertex falls back to the circumcenter,
      // then to the midpoint of the longest unconstrained edge.
      ++result_.stats.relocations_tried;
      if (insert_circumcenter(t,circumcenter) ||
          (status_==PHX_MC_OK && insert_edge_midpoint(t,center_metric))) {
        ++result_.stats.relocations_accepted;
        return;
      }
      ++result_.stats.relocations_failed;
      if (status_!=PHX_MC_OK || flip_refused(t) || status_!=PHX_MC_OK) return;
      refuse(result_.stats.proximity_refusals,CadRefine2DRefusal::proximity,t,p);
      return;
    }
    insert(host,p,m,t);
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
  bool insert(int32_t host, const double* p,const CadMetric2D& m,int32_t source,bool speculative=false) {
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
    if (!star_boundary(host,p)) {
      if (!speculative) refuse(result_.stats.nonstar_refusals,CadRefine2DRefusal::nonstar,source,p);
      return false;
    }
    if (speculative && std::find(cavity_.begin(),cavity_.end(),source)==cavity_.end()) return false;
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
        if (!speculative) refuse(result_.stats.nonstar_refusals,CadRefine2DRefusal::nonstar,source,p);
        return false;
      }
    }
    tri_.add_vertex(p);
    add_to_grid(id);
    for (int32_t t : cavity_) {
      if (static_cast<std::size_t>(t)<generations_.size()) ++generations_[t];
      if (static_cast<std::size_t>(t)<last_refusals_.size()) last_refusals_[t]=CadRefine2DUnresolved{};
      tri_.release(t);
    }
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
    for (int32_t t : created_) { classify_state(t); if (status_ != PHX_MC_OK) return true; }
    // Each new triangle is enqueued here at most once; among its neighbors
    // only the one across the cavity boundary can gain a front (the others
    // are new triangles, handled in turn; the pop order is independent of
    // push order).
    for (std::size_t i=0;i<created_.size();++i) {
      const int32_t t=created_[i];
      if (states_[t]==2 && !queued_[t]) enqueue(t);
      const int32_t nb=boundary_[i].outside;
      if(inside(nb) && static_cast<std::size_t>(nb)<states_.size()) {
        if(states_[nb]==3 && states_[t]==1) states_[nb]=2;
        if(states_[nb]==2 && !queued_[nb]) enqueue(nb);
      }
    }
    return true;
  }
};

}  // namespace
int32_t cad_refine2d_bad_triangle(const CadRefine2DInput& input,
                                  CadRefine2DResult& result, int32_t triangle,
                                  bool& bad_size, bool& bad_angle) {
  if (!input.metric || triangle<0 ||
      static_cast<std::size_t>(triangle)>=result.triangulation.triangles.size() ||
      (input.fixed_vertices && (input.point_count<0 ||
       static_cast<std::size_t>(input.point_count)!=result.vertex_map.size())) ||
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
