//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Periodic Delaunay on bounded certified image neighborhoods: cell counts from
// the torus Euler characteristic, exact covering measure, positive lifted
// orientation, empty circumballs against every nearby image (independent
// brute force), cocircular lattices, skew lattices, single-point tori, and
// refusal of exhausted image/cell budgets and repeated orbits.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <array>
#include <cstring>
#include <limits>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "parallel.hpp"
#include "cavity.hpp"

namespace {

struct Periodic {
  int32_t status = PHX_MC_OK;
  std::vector<double> evidence = std::vector<double>(PHX_MC_PERIODIC_EVIDENCE, 0.0);
  std::vector<int32_t> vertices;
  std::vector<int32_t> shifts;
  int64_t cells = 0;
};

// Lattice rows `lattice`; inverse computed for d = 2 or 3.
std::vector<double> inverse_of(const std::vector<double>& lattice, int d) {
  std::vector<double> inverse(static_cast<std::size_t>(d * d), 0.0);
  if (d == 2) {
    const double det = lattice[0] * lattice[3] - lattice[1] * lattice[2];
    inverse = {lattice[3] / det, -lattice[1] / det, -lattice[2] / det, lattice[0] / det};
    return inverse;
  }
  const double* m = lattice.data();
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  inverse = {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det,
             (m[1] * m[5] - m[2] * m[4]) / det, (m[5] * m[6] - m[3] * m[8]) / det,
             (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
             (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det,
             (m[0] * m[4] - m[1] * m[3]) / det};
  return inverse;
}

Periodic triangulate(int d, const std::vector<double>& fractional, const std::vector<double>& lattice,
                     double margin, int64_t max_images, int64_t max_cells) {
  const int64_t n = static_cast<int64_t>(fractional.size()) / d;
  std::vector<double> points(fractional.size(), 0.0);
  for (int64_t p = 0; p < n; ++p) {
    for (int c = 0; c < d; ++c) {
      for (int j = 0; j < d; ++j) {
        points[static_cast<std::size_t>(p * d + c)] +=
            fractional[static_cast<std::size_t>(p * d + j)] * lattice[static_cast<std::size_t>(j * d + c)];
      }
    }
  }
  // Fractional coordinates of the rounded Cartesian points, as a caller would.
  const std::vector<double> inverse = inverse_of(lattice, d);
  std::vector<double> wrapped(fractional.size(), 0.0);
  for (int64_t p = 0; p < n; ++p) {
    for (int j = 0; j < d; ++j) {
      double value = 0.0;
      for (int c = 0; c < d; ++c) {
        value += points[static_cast<std::size_t>(p * d + c)] * inverse[static_cast<std::size_t>(c * d + j)];
      }
      wrapped[static_cast<std::size_t>(p * d + j)] = std::clamp(value, 0.0, std::nextafter(1.0, 0.0));
    }
  }
  Periodic result;
  phx_mc_periodic_triangulation* handle = nullptr;
  result.status = phx_mc_periodic_delaunay(d, n, points.data(), wrapped.data(), lattice.data(),
                                           inverse.data(), margin, max_images, max_cells,
                                           result.evidence.data(), &handle);
  if (result.status == PHX_MC_OK) {
    result.cells = phx_mc_periodic_cell_count(handle);
    result.vertices.resize(static_cast<std::size_t>(result.cells * (d + 1)));
    result.shifts.resize(static_cast<std::size_t>(result.cells * (d + 1) * d));
    PHX_CHECK(phx_mc_periodic_copy_cells(handle, result.vertices.data(), result.shifts.data()) == PHX_MC_OK);
  }
  phx_mc_periodic_free(handle);
  return result;
}

std::vector<double> lifted(const std::vector<double>& fractional, const std::vector<double>& lattice,
                           int d, int32_t representative, const int32_t* shift) {
  std::vector<double> x(static_cast<std::size_t>(d), 0.0);
  for (int c = 0; c < d; ++c) {
    for (int j = 0; j < d; ++j) {
      x[static_cast<std::size_t>(c)] +=
          (fractional[static_cast<std::size_t>(representative * d + j)] + shift[j]) *
          lattice[static_cast<std::size_t>(j * d + c)];
    }
  }
  return x;
}

// Total measure; checks positive orientation and that no image within two
// lattice shells lies inside any circumball (independent brute force).
double check_cells(const Periodic& result, int d, const std::vector<double>& fractional,
                   const std::vector<double>& lattice) {
  const int32_t n = static_cast<int32_t>(fractional.size()) / d;
  double total = 0.0;
  int negative = 0;
  int violations = 0;
  for (int64_t cell = 0; cell < result.cells; ++cell) {
    std::vector<std::vector<double>> corners;
    for (int i = 0; i <= d; ++i) {
      const std::size_t row = static_cast<std::size_t>(cell * (d + 1) + i);
      corners.push_back(lifted(fractional, lattice, d, result.vertices[row],
                               &result.shifts[row * static_cast<std::size_t>(d)]));
    }
    double e[3][3] = {};
    for (int i = 0; i < d; ++i) {
      for (int c = 0; c < d; ++c) {
        e[i][c] = corners[static_cast<std::size_t>(i + 1)][static_cast<std::size_t>(c)] -
                  corners[0][static_cast<std::size_t>(c)];
      }
    }
    double center[3] = {};
    double measure = 0.0;
    if (d == 2) {
      const double det = e[0][0] * e[1][1] - e[0][1] * e[1][0];
      const double b0 = 0.5 * (e[0][0] * e[0][0] + e[0][1] * e[0][1]);
      const double b1 = 0.5 * (e[1][0] * e[1][0] + e[1][1] * e[1][1]);
      center[0] = (b0 * e[1][1] - b1 * e[0][1]) / det;
      center[1] = (e[0][0] * b1 - e[1][0] * b0) / det;
      measure = det / 2.0;
    } else {
      const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                         e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                         e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
      measure = det / 6.0;
      double rhs[3];
      for (int i = 0; i < 3; ++i) {
        rhs[i] = 0.5 * (e[i][0] * e[i][0] + e[i][1] * e[i][1] + e[i][2] * e[i][2]);
      }
      for (int c = 0; c < 3; ++c) {
        double m[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int k = 0; k < 3; ++k) {
            m[i][k] = k == c ? rhs[i] : e[i][k];
          }
        }
        center[c] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                     m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) /
                    det;
      }
    }
    negative += measure <= 0.0;
    total += measure;
    double radius2 = 0.0;
    for (int c = 0; c < d; ++c) {
      center[c] += corners[0][static_cast<std::size_t>(c)];
      radius2 += (corners[0][static_cast<std::size_t>(c)] - center[c]) *
                 (corners[0][static_cast<std::size_t>(c)] - center[c]);
    }
    const int32_t shells = 2;
    for (int32_t p = 0; p < n; ++p) {
      int32_t s[3] = {-shells, -shells, d == 3 ? -shells : 0};
      while (true) {
        const std::vector<double> x = lifted(fractional, lattice, d, p, s);
        double distance2 = 0.0;
        for (int c = 0; c < d; ++c) {
          distance2 += (x[static_cast<std::size_t>(c)] - center[c]) * (x[static_cast<std::size_t>(c)] - center[c]);
        }
        violations += distance2 < radius2 * (1.0 - 1e-9);
        int axis = 0;
        while (axis < d) {
          if (s[axis] < shells) {
            ++s[axis];
            break;
          }
          s[axis] = -shells;
          ++axis;
        }
        if (axis == d) {
          break;
        }
      }
    }
  }
  PHX_CHECK(negative == 0);
  PHX_CHECK(violations == 0);
  return total;
}

std::vector<double> random_fractional(int n, int d, unsigned seed) {
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> values(static_cast<std::size_t>(n * d));
  for (double& value : values) {
    value = uniform(engine);
  }
  return values;
}

void random_torus_2d() {
  const std::vector<double> lattice = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> fractional = random_fractional(60, 2, 7);
  const Periodic result = triangulate(2, fractional, lattice, 0.25, 1 << 20, 1 << 22);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.cells == 120);
  PHX_CHECK(result.evidence[PHX_MC_PERIODIC_UNCERTIFIED] == 0.0);
  PHX_CHECK(result.evidence[PHX_MC_PERIODIC_REQUIRED_MARGIN] <= result.evidence[PHX_MC_PERIODIC_MARGIN]);
  PHX_CHECK_NEAR(check_cells(result, 2, fractional, lattice), 1.0, 1e-12);
}

void cocircular_lattice_2d() {
  std::vector<double> fractional;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      fractional.push_back(0.25 * i);
      fractional.push_back(0.25 * j);
    }
  }
  const std::vector<double> lattice = {1.0, 0.0, 0.0, 1.0};
  const Periodic result = triangulate(2, fractional, lattice, 0.25, 1 << 20, 1 << 22);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.cells == 32);
  PHX_CHECK(result.evidence[PHX_MC_PERIODIC_PERTURBED_DECISIONS] > 0.0);
  PHX_CHECK_NEAR(check_cells(result, 2, fractional, lattice), 1.0, 1e-12);
}

void skew_and_single_point_2d() {
  const std::vector<double> skew = {1.0, 0.0, 0.4, 0.9};
  const std::vector<double> fractional = random_fractional(40, 2, 11);
  const Periodic result = triangulate(2, fractional, skew, 0.25, 1 << 20, 1 << 22);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.cells == 80);
  PHX_CHECK_NEAR(check_cells(result, 2, fractional, skew), 0.9, 1e-12);

  const Periodic single = triangulate(2, {0.5, 0.5}, {1.0, 0.0, 0.0, 1.0}, 0.25, 1 << 20, 1 << 22);
  PHX_CHECK(single.status == PHX_MC_OK);
  PHX_CHECK(single.cells == 2);
}

void random_torus_3d() {
  const std::vector<double> lattice = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::vector<double> fractional = random_fractional(40, 3, 5);
  const Periodic result = triangulate(3, fractional, lattice, 0.25, 1 << 22, 1 << 24);
  PHX_CHECK(result.status == PHX_MC_OK);
  PHX_CHECK(result.cells > 0);
  PHX_CHECK_NEAR(check_cells(result, 3, fractional, lattice), 1.0, 1e-12);

  std::vector<double> grid;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        grid.insert(grid.end(), {i / 3.0, j / 3.0, k / 3.0});
      }
    }
  }
  const Periodic cubic = triangulate(3, grid, lattice, 0.25, 1 << 22, 1 << 24);
  PHX_CHECK(cubic.status == PHX_MC_OK);
  PHX_CHECK_NEAR(check_cells(cubic, 3, grid, lattice), 1.0, 1e-12);
}

void refusals() {
  const std::vector<double> lattice = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> fractional = random_fractional(30, 2, 3);
  const Periodic images = triangulate(2, fractional, lattice, 0.01, 40, 1 << 22);
  PHX_CHECK(images.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(images.evidence[PHX_MC_PERIODIC_EXHAUSTED_LIMIT] == 1.0);
  PHX_CHECK(images.evidence[PHX_MC_PERIODIC_IMAGES] > 40.0);

  const Periodic cells = triangulate(2, fractional, lattice, 0.25, 1 << 20, 16);
  PHX_CHECK(cells.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(cells.evidence[PHX_MC_PERIODIC_EXHAUSTED_LIMIT] == 2.0);

  const Periodic duplicate =
      triangulate(2, {0.1, 0.2, 0.7, 0.3, 0.1, 0.2}, lattice, 0.25, 1 << 20, 1 << 22);
  PHX_CHECK(duplicate.status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(duplicate.evidence[PHX_MC_PERIODIC_DUPLICATE_POINT] >= 0.0);

  double evidence[PHX_MC_PERIODIC_EVIDENCE];
  phx_mc_periodic_triangulation* handle = nullptr;
  const double point[2] = {0.5, 0.5};
  const double identity[4] = {1.0, 0.0, 0.0, 1.0};
  // A representative outside the fundamental cell keeps its coordinates; its
  // base image carries the wrap shift.
  const double outside[2] = {1.5, 0.5};
  PHX_CHECK(phx_mc_periodic_delaunay(2, 1, outside, outside, identity, identity, 0.25, 100, 100,
                                     evidence, &handle) == PHX_MC_OK);
  PHX_CHECK(phx_mc_periodic_cell_count(handle) == 2);
  int32_t vertices[6];
  int32_t shifts[12];
  PHX_CHECK(phx_mc_periodic_copy_cells(handle, vertices, shifts) == PHX_MC_OK);
  PHX_CHECK(shifts[0] == -1 && shifts[1] == 0);
  phx_mc_periodic_free(handle);
  handle = nullptr;
  const double translated[4] = {0.25, 0.5, 1.25, 0.5};
  PHX_CHECK(phx_mc_periodic_delaunay(2, 2, translated, translated, identity, identity, 0.25, 100,
                                     100, evidence, &handle) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(handle == nullptr);
  PHX_CHECK(evidence[PHX_MC_PERIODIC_DUPLICATE_POINT] >= 0.0);
  PHX_CHECK(phx_mc_periodic_delaunay(4, 1, point, point, identity, identity, 0.25, 100, 100,
                                     evidence, &handle) == PHX_MC_INVALID_ARGUMENT);
}


struct RefinedPeriodic {
  int32_t status=PHX_MC_OK;
  std::array<uint64_t,PHX_MC_PERIODIC_REFINEMENT_COUNTERS> counters{};
  std::vector<double> points;
  Periodic topology;
};

RefinedPeriodic refine_periodic(const std::vector<double>& fractional,
    const std::vector<double>& lattice,const std::vector<double>& controls,
    const std::vector<double>& quantiles,uint32_t parallelism,
    int64_t max_work=1<<26,int64_t max_images=1<<20,
    const std::vector<int32_t>& protected_edges={},int64_t max_scratch=1<<28) {
  const Periodic initial=triangulate(2,fractional,lattice,0.25,1<<20,1<<22);
  PHX_CHECK(initial.status==PHX_MC_OK);
  const auto inverse=inverse_of(lattice,2);
  std::vector<double> points(fractional.size());
  for (std::size_t i=0;i<points.size()/2;++i) for (int c=0;c<2;++c)
    points[2*i+c]=fractional[2*i]*lattice[c]+fractional[2*i+1]*lattice[2+c];
  phx_mc_periodic_refinement_input input{};
  input.point_count=static_cast<int64_t>(points.size()/2);
  input.points=points.data();input.lattice=lattice.data();input.inverse=inverse.data();
  input.cell_count=initial.cells;input.vertices=initial.vertices.data();input.shifts=initial.shifts.data();
  input.control_count=static_cast<int64_t>(controls.size()/5);input.controls=controls.data();
  input.statistic_count=static_cast<int64_t>(quantiles.size());input.quantiles=quantiles.data();
  input.max_vertices=4096;input.max_cells=8192;input.max_images=max_images;
  input.max_work_units=max_work;input.max_cavity_cells=8192;input.max_scratch_bytes=max_scratch;
  input.protected_edge_count=static_cast<int64_t>(protected_edges.size()/4);
  input.protected_edges=protected_edges.data();
  RefinedPeriodic result;
  phx_mc_periodic_refinement* handle=nullptr;
  phx::mc::NativeParallelPolicy policy(parallelism);
  result.status=phx_mc_periodic_refine_2d(&input,result.counters.data(),&handle);
  if (result.status==PHX_MC_OK) {
    const auto count=phx_mc_periodic_refinement_point_count(handle);
    result.points.resize(static_cast<std::size_t>(2*count));
    result.topology.cells=phx_mc_periodic_refinement_cell_count(handle);
    result.topology.vertices.resize(static_cast<std::size_t>(3*result.topology.cells));
    result.topology.shifts.resize(static_cast<std::size_t>(6*result.topology.cells));
    PHX_CHECK(phx_mc_periodic_refinement_copy_points(handle,result.points.data())==PHX_MC_OK);
    PHX_CHECK(phx_mc_periodic_refinement_copy_cells(handle,result.topology.vertices.data(),
        result.topology.shifts.data())==PHX_MC_OK);
  } else PHX_CHECK(handle==nullptr);
  phx_mc_periodic_refinement_free(handle);
  return result;
}

std::vector<double> refined_edge_lengths(const RefinedPeriodic& result,
                                        const std::vector<double>& lattice) {
  using Image=std::array<int32_t,3>;
  using Edge=std::array<int32_t,4>;
  std::vector<Image> corners;
  std::vector<Edge> edges;
  for (std::size_t i=0;i<result.topology.vertices.size();++i)
    corners.push_back({result.topology.vertices[i],result.topology.shifts[2*i],
                      result.topology.shifts[2*i+1]});
  std::vector<Image> lifts(result.points.size()/2,{INT32_MAX,INT32_MAX,INT32_MAX});
  for (Image image:corners) {
    Image& best=lifts[image[0]];
    if (best[0]==INT32_MAX || image<best) best=image;
  }
  for (Image image:corners) if (image[1]==0 && image[2]==0) lifts[image[0]]=image;
  for (std::size_t cell=0;cell<corners.size()/3;++cell)
    for (const auto pair:{std::array{0,1},std::array{0,2},std::array{1,2}}) {
      Image a=corners[3*cell+pair[0]],b=corners[3*cell+pair[1]];
      if (b<a) std::swap(a,b);
      edges.push_back({a[0],b[0],b[1]-a[1],b[2]-a[2]});
    }
  std::sort(edges.begin(),edges.end());
  edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  std::vector<double> lengths;
  for (const Edge& edge:edges) {
    double norm=0.0;
    for (int c=0;c<2;++c) {
      const auto a=lifts[edge[0]],b=lifts[edge[1]];
      const double first=result.points[2*edge[0]+c]+(a[1]*lattice[c]+a[2]*lattice[2+c]);
      const double second=result.points[2*edge[1]+c]+(b[1]*lattice[c]+b[2]*lattice[2+c]);
      const double delta=(second-first)+
          ((edge[2]-b[1]+a[1])*lattice[c]+(edge[3]-b[2]+a[2])*lattice[2+c]);
      norm+=delta*delta;
    }
    lengths.push_back(std::sqrt(norm));
  }
  std::sort(lengths.begin(),lengths.end());
  return lengths;
}

double linear_quantile(const std::vector<double>& values,double quantile) {
  const double rank=quantile*static_cast<double>(values.size()-1);
  const auto lo=static_cast<std::size_t>(std::floor(rank));
  const auto hi=static_cast<std::size_t>(std::ceil(rank));
  const double t=rank-static_cast<double>(lo);
  return t>=0.5 ? values[hi]-(values[hi]-values[lo])*(1.0-t) :
                  values[lo]+(values[hi]-values[lo])*t;
}

void incremental_periodic_refinement() {
  const double absent=std::numeric_limits<double>::quiet_NaN();
  std::vector<double> fractional;
  for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
    fractional.push_back(0.25*i);fractional.push_back(0.25*j);
  }
  const std::vector<double> lattice={1.0,0.0,0.25,1.0};
  const std::vector<double> controls={0.25,absent,0.375,absent,1.0};
  const auto baseline=refine_periodic(fractional,lattice,controls,{0.5,0.95},1);
  PHX_CHECK(baseline.status==PHX_MC_OK);
  if (baseline.status!=PHX_MC_OK) return;
  PHX_CHECK(baseline.counters[4]>0);
  PHX_CHECK(baseline.counters[7]==0 && baseline.counters[9]==0);
  PHX_CHECK(baseline.counters[1]==baseline.counters[2]);
  PHX_CHECK(baseline.counters[3]>=baseline.counters[4]);
  PHX_CHECK(baseline.topology.cells==static_cast<int64_t>(baseline.points.size()));
  const auto lengths=refined_edge_lengths(baseline,lattice);
  PHX_CHECK(linear_quantile(lengths,0.5)==0.25);
  PHX_CHECK(linear_quantile(lengths,0.95)==0.25);
  PHX_CHECK(lengths.back()<=0.375);
  const auto inverse=inverse_of(lattice,2);
  std::vector<double> final_fractional(baseline.points.size());
  for (std::size_t i=0;i<baseline.points.size()/2;++i) for (int j=0;j<2;++j)
    final_fractional[2*i+j]=baseline.points[2*i]*inverse[j]+baseline.points[2*i+1]*inverse[2+j];
  PHX_CHECK_NEAR(check_cells(baseline.topology,2,final_fractional,lattice),1.0,1e-12);
  // The unchanged covering-image owner is the independent exact/SOS oracle.
  phx_mc_periodic_triangulation* reference=nullptr;
  double evidence[PHX_MC_PERIODIC_EVIDENCE];
  PHX_CHECK(phx_mc_periodic_delaunay(2,static_cast<int64_t>(baseline.points.size()/2),
      baseline.points.data(),final_fractional.data(),lattice.data(),inverse.data(),
      0.25,1<<20,1<<22,evidence,&reference)==PHX_MC_OK);
  if (reference!=nullptr) {
    const auto cells=phx_mc_periodic_cell_count(reference);
    std::vector<int32_t> vertices(static_cast<std::size_t>(3*cells)),shifts(static_cast<std::size_t>(6*cells));
    PHX_CHECK(phx_mc_periodic_copy_cells(reference,vertices.data(),shifts.data())==PHX_MC_OK);
    PHX_CHECK(vertices==baseline.topology.vertices && shifts==baseline.topology.shifts);
  }
  phx_mc_periodic_free(reference);
  for (uint32_t threads:{2U,7U,64U}) {
    const auto parallel=refine_periodic(fractional,lattice,controls,{0.5,0.95},threads);
    PHX_CHECK(parallel.status==baseline.status);
    PHX_CHECK(parallel.counters==baseline.counters);
    PHX_CHECK(parallel.points.size()==baseline.points.size());
    if (parallel.points.size()==baseline.points.size())
      PHX_CHECK(std::memcmp(parallel.points.data(),baseline.points.data(),
                           baseline.points.size()*sizeof(double))==0);
    PHX_CHECK(parallel.topology.vertices==baseline.topology.vertices);
    PHX_CHECK(parallel.topology.shifts==baseline.topology.shifts);
  }
}

void periodic_target_rank_continuation_is_truthful() {
  const double absent=std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> lattice={1.0,0.0,0.25,1.0};
  std::vector<double> fractional;
  for (int i=0;i<8;++i) for (int j=0;j<8;++j) {
    fractional.push_back(0.125*i);fractional.push_back(0.125*j);
  }
  const auto result=refine_periodic(fractional,lattice,
      {0.125,absent,0.1875,absent,1.0},{0.5,0.95},1,40960000,20000,{},81920000);
  PHX_CHECK(result.status==PHX_MC_OK);
  if (result.status!=PHX_MC_OK) return;
  const auto lengths=refined_edge_lengths(result,lattice);
  const double p50=linear_quantile(lengths,0.5),p95=linear_quantile(lengths,0.95);
  PHX_CHECK_NEAR(p95,0.125,0.0);
  PHX_CHECK(lengths.back()<=0.1875);
  const uint64_t issues=static_cast<uint64_t>(p50!=0.125)+static_cast<uint64_t>(p95!=0.125);
  PHX_CHECK(result.counters[7]==issues);
  PHX_CHECK(result.counters[9]==(issues ? 2U : 0U));
  if (issues) {
    PHX_CHECK((std::abs(p50-0.125)+std::abs(p95-0.125))/0.125<=
              64.0*std::numeric_limits<double>::epsilon());
  }
}

void periodic_source_edge_orbits_are_retained() {
  const double absent=std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> lattice={1.0,0.0,0.25,1.0};
  std::vector<double> fractional;
  for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
    fractional.push_back(0.25*i);fractional.push_back(0.25*j);
  }
  std::vector<int32_t> protected_edges;
  for (int i:{0,2}) for (int j=0;j<4;++j) {
    const int32_t midpoint=static_cast<int32_t>(fractional.size()/2);
    fractional.push_back(0.25*i);fractional.push_back(0.25*j+0.125);
    protected_edges.insert(protected_edges.end(),{4*i+j,midpoint,0,0});
    protected_edges.insert(protected_edges.end(),{4*i+(j+1)%4,midpoint,0,j==3 ? -1 : 0});
  }
  const auto result=refine_periodic(fractional,lattice,
      {0.25,absent,0.375,absent,1.0},{0.5,0.95},1,40960000,20000,protected_edges,81920000);
  PHX_CHECK(result.status==PHX_MC_OK);
  if (result.status!=PHX_MC_OK) return;
  PHX_CHECK(result.counters[4]>0);
  const auto lengths=refined_edge_lengths(result,lattice);
  const uint64_t issues=static_cast<uint64_t>(linear_quantile(lengths,0.5)!=0.25)+
                        static_cast<uint64_t>(linear_quantile(lengths,0.95)!=0.25)+
                        static_cast<uint64_t>(lengths.back()>0.375);
  PHX_CHECK(result.counters[7]==issues);
  for (std::size_t row=0;row<protected_edges.size()/4;++row) {
    const std::array<int32_t,4> expected={protected_edges[4*row],protected_edges[4*row+1],
        protected_edges[4*row+2],protected_edges[4*row+3]};
    int incidence=0;
    for (std::size_t cell=0;cell<result.topology.vertices.size()/3;++cell)
      for (int side=0;side<3;++side) {
        const std::size_t a=3*cell+side,b=3*cell+(side+1)%3;
        int32_t first=result.topology.vertices[a],second=result.topology.vertices[b];
        int32_t dx=result.topology.shifts[2*b]-result.topology.shifts[2*a];
        int32_t dy=result.topology.shifts[2*b+1]-result.topology.shifts[2*a+1];
        if (first>second) {std::swap(first,second);dx=-dx;dy=-dy;}
        incidence+=std::array<int32_t,4>{first,second,dx,dy}==expected;
      }
    PHX_CHECK(incidence==2);
  }
  const auto parallel=refine_periodic(fractional,lattice,
      {0.25,absent,0.375,absent,1.0},{0.5,0.95},7,1<<26,1<<20,protected_edges);
  PHX_CHECK(parallel.status==result.status && parallel.counters==result.counters);
  PHX_CHECK(parallel.points==result.points);
  PHX_CHECK(parallel.topology.vertices==result.topology.vertices);
  PHX_CHECK(parallel.topology.shifts==result.topology.shifts);
  auto invalid=protected_edges;
  invalid[2]=1;
  const auto rejected=refine_periodic(fractional,lattice,
      {0.25,absent,0.375,absent,1.0},{0.5,0.95},1,1<<26,1<<20,invalid);
  PHX_CHECK(rejected.status==PHX_MC_INVALID_INPUT);
}

void periodic_refinement_stop_and_limits() {
  const double absent=std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> lattice={1.0,0.0,0.0,1.0};
  const auto stopped=refine_periodic({0.5,0.5},lattice,{0.1,absent,absent,absent,1.0},{0.5},1);
  PHX_CHECK(stopped.status==PHX_MC_OK);
  PHX_CHECK(stopped.counters[4]==0 && stopped.counters[7]>0 && stopped.counters[9]==1);
  PHX_CHECK(stopped.points==std::vector<double>({0.5,0.5}) && stopped.topology.cells==2);
  const auto refused=refine_periodic({0.5,0.5},lattice,{0.6,absent,absent,absent,1.0},{0.5},1,1);
  PHX_CHECK(refused.status==PHX_MC_CAPACITY_EXCEEDED);
  const auto images=refine_periodic({0.5,0.5},lattice,{0.6,absent,absent,absent,1.0},{0.5},7,1<<26,1);
  PHX_CHECK(images.status==PHX_MC_OK);
  PHX_CHECK(images.counters[5]>0 && images.counters[4]==0 && images.counters[9]==1);
  // A single-orbit quotient forces insertion to handle intersecting image
  // cavities, rather than assuming the lifted cavity is an embedded disk.
  const auto sparse=refine_periodic({0.5,0.5},lattice,{0.8,absent,0.9,absent,1.0},{0.5},2);
  PHX_CHECK(sparse.status==PHX_MC_OK);
  if (sparse.status==PHX_MC_OK) {
    PHX_CHECK(sparse.counters[4]>0);
    PHX_CHECK(sparse.topology.cells==static_cast<int64_t>(sparse.points.size()));
    PHX_CHECK_NEAR(check_cells(sparse.topology,2,sparse.points,lattice),1.0,1e-12);
  }
  const auto growth=refine_periodic({0.5,0.5},lattice,
      {0.1,0.01,1.2,1.1,1.0},{0.5,0.95},7);
  PHX_CHECK(growth.status==PHX_MC_OK && growth.counters[7]==4 && growth.counters[9]==1);
  // Soft controls retain merit evidence without inventing hard issues.
  const auto soft=refine_periodic({0.5,0.5},lattice,{0.1,0.01,2.0,2.0,0.0},{0.5,0.95},1);
  PHX_CHECK(soft.status==PHX_MC_OK && soft.counters[7]==0 && soft.counters[4]==0);
  PHX_CHECK(phx_mc_periodic_refinement_point_count(nullptr)==0);
  PHX_CHECK(phx_mc_periodic_refinement_cell_count(nullptr)==0);
  PHX_CHECK(phx_mc_periodic_refinement_copy_points(nullptr,nullptr)==PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_periodic_refinement_copy_cells(nullptr,nullptr,nullptr)==PHX_MC_INVALID_ARGUMENT);
  phx_mc_periodic_refinement_free(nullptr);
}

void quotient_transaction_rollback() {
  struct Cell { int id; bool live=true; };
  phx::mc::NativeVector<Cell> cells={{3,true}};
  phx::mc::QuotientCavityEdit<Cell> transaction(cells);
  const std::array<int32_t,1> removed={0};
  const std::array<Cell,2> proposed={Cell{4,true},Cell{5,true}};
  const auto before=cells;
  PHX_CHECK(transaction.prepare(removed,proposed,1,1,
      [](const auto&,const auto&){return true;})==phx::mc::EditStatus::kCellLimit);
  PHX_CHECK(cells.size()==before.size() && cells[0].id==before[0].id && cells[0].live);
  PHX_CHECK(transaction.prepare(removed,proposed,1,2,
      [](const auto&,const auto&){return false;})==phx::mc::EditStatus::kBoundaryMismatch);
  PHX_CHECK(cells.size()==before.size() && cells[0].id==before[0].id && cells[0].live);
  PHX_CHECK(transaction.prepare(removed,proposed,1,2,
      [](const auto&,const auto&){return true;})==phx::mc::EditStatus::kOk);
  int publications=0;
  PHX_CHECK(transaction.commit([&]() noexcept {++publications;})==phx::mc::EditStatus::kOk);
  PHX_CHECK(publications==1 && cells.size()==2 && cells[0].id==4 && cells[1].id==5);
}

}  // namespace

int main() {
  random_torus_2d();
  cocircular_lattice_2d();
  skew_and_single_point_2d();
  random_torus_3d();
  refusals();
  incremental_periodic_refinement();
  periodic_target_rank_continuation_is_truthful();
  periodic_source_edge_orbits_are_retained();
  periodic_refinement_stop_and_limits();
  quotient_transaction_rollback();
  return phx::mc::test::finish("test_periodic");
}
