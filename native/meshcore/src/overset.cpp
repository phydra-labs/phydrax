//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Overset assembly (overset.hpp). Construction runs in binary64 with exact
// predicates where a decision is topological (hole parity, wall contact, cut
// separation); donor certification re-checks every chosen pair afterwards:
// exact orientation signs for tetrahedra and a Krawczyk interval enclosure of
// the trilinear inverse for hexahedra, moving to the next ordered candidate
// when a float choice fails the exact check.
#include "overset.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "directed_interval.hpp"
#include "predicates.hpp"

namespace phx::mc::overset {
namespace {

using Clock = std::chrono::steady_clock;
using remesh::Interval;

constexpr int8_t kVertexActive = 0;
constexpr int8_t kVertexReceptor = 1;
constexpr int8_t kVertexHole = 2;
constexpr int8_t kCellActive = 0;
constexpr int8_t kCellFringe = 1;
constexpr int8_t kCellHole = 2;
constexpr int32_t kFound = 0;
constexpr int32_t kOutside = 1;
constexpr int32_t kExcludedDonor = 2;
constexpr int32_t kUnresolved = 5;
constexpr uint8_t kRoleWall = 1;
constexpr uint8_t kRoleBoundary = 2;

// Float location tolerances: construction keeps a candidate within the loose
// band (certification decides it); prove = 0 accepts within the tight band.
constexpr double kCandidateBand = 1e-6;
constexpr double kUnprovedBand = 1e-10;

// Hexahedron corner k sits at reference (kBit[0][k], kBit[1][k], kBit[2][k]).
constexpr std::array<std::array<int, 8>, 3> kBit = {{{0, 1, 1, 0, 0, 1, 1, 0},
                                                     {0, 0, 1, 1, 0, 0, 1, 1},
                                                     {0, 0, 0, 0, 1, 1, 1, 1}}};
// Hexahedron faces as corner cycles.
constexpr std::array<std::array<int, 4>, 6> kHexFaces = {
    {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}}};
// Tetrahedron faces with their opposite corner.
constexpr std::array<std::array<int, 4>, 4> kTetFaces = {
    {{1, 2, 3, 0}, {0, 2, 3, 1}, {0, 1, 3, 2}, {0, 1, 2, 3}}};

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Box {
  std::array<double, 3> lo{}, hi{};
};

Box empty_box() {
  Box box;
  box.lo.fill(std::numeric_limits<double>::infinity());
  box.hi.fill(-std::numeric_limits<double>::infinity());
  return box;
}

void extend(Box& box, const double* p) {
  for (int a = 0; a < 3; ++a) {
    box.lo[a] = std::min(box.lo[a], p[a]);
    box.hi[a] = std::max(box.hi[a], p[a]);
  }
}

void extend(Box& box, const Box& other) {
  for (int a = 0; a < 3; ++a) {
    box.lo[a] = std::min(box.lo[a], other.lo[a]);
    box.hi[a] = std::max(box.hi[a], other.hi[a]);
  }
}

bool box_empty(const Box& box) { return !(box.lo[0] <= box.hi[0]); }

bool contains(const Box& box, const double* p) {
  return box.lo[0] <= p[0] && p[0] <= box.hi[0] && box.lo[1] <= p[1] && p[1] <= box.hi[1] &&
         box.lo[2] <= p[2] && p[2] <= box.hi[2];
}

bool overlaps(const Box& a, const Box& b) {
  return a.lo[0] <= b.hi[0] && b.lo[0] <= a.hi[0] && a.lo[1] <= b.hi[1] && b.lo[1] <= a.hi[1] &&
         a.lo[2] <= b.hi[2] && b.lo[2] <= a.hi[2];
}

// A uniform grid over item boxes. A point's bin and a box's bin range come
// from the same monotone rounded map, so every item whose closed box holds a
// point is listed in that point's bin.
class Bins {
 public:
  void build(const std::vector<Box>& boxes, bool use_x) {
    const int64_t count = static_cast<int64_t>(boxes.size());
    Box all = empty_box();
    for (const Box& box : boxes) extend(all, box);
    lo_ = all.lo;
    double product = 1.0;
    int dimensions = 0;
    std::array<double, 3> extent{};
    for (int a = 0; a < 3; ++a) {
      extent[a] = count == 0 ? 0.0 : all.hi[a] - all.lo[a];
      if ((a > 0 || use_x) && extent[a] > 0.0) {
        product *= extent[a];
        ++dimensions;
      }
    }
    const double side =
        dimensions == 0 ? 1.0 : std::pow(product / static_cast<double>(std::max<int64_t>(count, 1)),
                                         1.0 / dimensions);
    for (int a = 0; a < 3; ++a) {
      n_[a] = 1;
      if ((a > 0 || use_x) && extent[a] > 0.0 && side > 0.0) {
        n_[a] = static_cast<int64_t>(std::clamp(std::ceil(extent[a] / side), 1.0, 4096.0));
      }
    }
    while (n_[0] * n_[1] * n_[2] > 2 * count + 64) {
      int largest = 0;
      for (int a = 1; a < 3; ++a) largest = n_[a] > n_[largest] ? a : largest;
      n_[largest] = std::max<int64_t>(1, n_[largest] / 2);
    }
    for (int a = 0; a < 3; ++a) {
      scale_[a] = n_[a] > 1 ? static_cast<double>(n_[a]) / extent[a] : 0.0;
    }
    const int64_t total = n_[0] * n_[1] * n_[2];
    start_.assign(static_cast<size_t>(total + 1), 0);
    for (const Box& box : boxes) {
      range(box, [&](int64_t bin) { ++start_[static_cast<size_t>(bin + 1)]; });
    }
    for (int64_t bin = 0; bin < total; ++bin) start_[bin + 1] += start_[bin];
    items_.resize(static_cast<size_t>(start_[total]));
    std::vector<int64_t> cursor(start_.begin(), start_.end() - 1);
    for (int64_t item = 0; item < count; ++item) {
      range(boxes[item], [&](int64_t bin) { items_[cursor[bin]++] = static_cast<int32_t>(item); });
    }
  }

  template <class Visit>
  void visit_point(const double* p, Visit&& visit) const {
    if (items_.empty()) return;
    const int64_t bin = (coordinate(2, p[2]) * n_[1] + coordinate(1, p[1])) * n_[0] +
                        coordinate(0, p[0]);
    for (int64_t k = start_[bin]; k < start_[bin + 1]; ++k) visit(items_[k]);
  }

  // Items of every bin the box meets; an item may repeat.
  template <class Visit>
  void visit_box(const Box& box, Visit&& visit) const {
    if (items_.empty()) return;
    range(box, [&](int64_t bin) {
      for (int64_t k = start_[bin]; k < start_[bin + 1]; ++k) visit(items_[k]);
    });
  }

 private:
  int64_t coordinate(int axis, double x) const {
    const double t = (x - lo_[axis]) * scale_[axis];
    if (!(t > 0.0)) return 0;
    if (t >= static_cast<double>(n_[axis])) return n_[axis] - 1;
    return static_cast<int64_t>(t);
  }

  template <class Visit>
  void range(const Box& box, Visit&& visit) const {
    const int64_t i0 = coordinate(0, box.lo[0]), i1 = coordinate(0, box.hi[0]);
    const int64_t j0 = coordinate(1, box.lo[1]), j1 = coordinate(1, box.hi[1]);
    const int64_t k0 = coordinate(2, box.lo[2]), k1 = coordinate(2, box.hi[2]);
    for (int64_t k = k0; k <= k1; ++k) {
      for (int64_t j = j0; j <= j1; ++j) {
        for (int64_t i = i0; i <= i1; ++i) visit((k * n_[1] + j) * n_[0] + i);
      }
    }
  }

  std::array<double, 3> lo_{}, scale_{};
  std::array<int64_t, 3> n_{1, 1, 1};
  std::vector<int64_t> start_{0, 0};
  std::vector<int32_t> items_;
};

// ------------------------------------------------------------- predicates

int orient(const double* a, const double* b, const double* c, const double* d) {
  return orient3d_filtered(a, b, c, d);
}

int orient_plane(const double* a, const double* b, const double* c, int u, int v) {
  const double a2[2] = {a[u], a[v]}, b2[2] = {b[u], b[v]}, c2[2] = {c[u], c[v]};
  return orient2d_filtered(a2, b2, c2);
}

// The side of p from edge (a, b) in the (y, z) projection, ties broken as the
// Python owner does, so a ray through a shared edge or vertex counts once.
int perturbed_side(const double* a, const double* b, const double* p) {
  const int exact = orient_plane(a, b, p, 1, 2);
  if (exact != 0) return exact;
  if (b[2] != a[2]) return a[2] > b[2] ? 1 : -1;
  return b[1] > a[1] ? 1 : (b[1] < a[1] ? -1 : 0);
}

// Whether p, coplanar with triangle (a, b, c), lies in its closed interior.
bool coplanar_touch(const double* a, const double* b, const double* c, const double* p) {
  static constexpr int kPlanes[3][2] = {{1, 2}, {0, 2}, {0, 1}};
  for (const auto& plane : kPlanes) {
    const int area = orient_plane(a, b, c, plane[0], plane[1]);
    if (area == 0) continue;
    return orient_plane(a, b, p, plane[0], plane[1]) * area >= 0 &&
           orient_plane(b, c, p, plane[0], plane[1]) * area >= 0 &&
           orient_plane(c, a, p, plane[0], plane[1]) * area >= 0;
  }
  // A degenerate triangle: contact on its segment hull.
  for (int axis = 0; axis < 3; ++axis) {
    const double lo = std::min({a[axis], b[axis], c[axis]});
    const double hi = std::max({a[axis], b[axis], c[axis]});
    if (p[axis] < lo || p[axis] > hi) return false;
  }
  return true;
}

// The +x ray from p against triangle (a, b, c): crossing parity and contact.
void ray_triangle(const double* a, const double* b, const double* c, const double* p,
                  bool& crossing, bool& touching) {
  const int volume = orient(a, b, c, p);
  if (volume == 0) {
    touching = touching || coplanar_touch(a, b, c, p);
    return;
  }
  const int s0 = perturbed_side(a, b, p);
  const int s1 = perturbed_side(b, c, p);
  const int s2 = perturbed_side(c, a, p);
  if (s0 == 0 || s0 != s1 || s1 != s2) return;
  const int normal = orient_plane(a, b, c, 1, 2);
  if (volume == -normal) crossing = !crossing;
}

// Closed containment of p in the tetrahedron (a, b, c, d) of orientation sign s != 0.
bool in_tetrahedron(const double* a, const double* b, const double* c, const double* d, int s,
                    const double* p) {
  return orient(p, b, c, d) * s >= 0 && orient(a, p, c, d) * s >= 0 &&
         orient(a, b, p, d) * s >= 0 && orient(a, b, c, p) * s >= 0;
}

// ----------------------------------------------------------- cell geometry

struct Mesh {
  const Input* input;
  const double* x(int64_t vertex) const { return input->vertices + 3 * vertex; }
  const int32_t* row(int64_t cell) const { return input->cells + 8 * cell; }
  bool hex(int64_t cell) const { return input->cells[8 * cell + 4] >= 0; }
  int corners(int64_t cell) const { return hex(cell) ? 8 : 4; }
  Box cell_box(int64_t cell) const {
    Box box = empty_box();
    const int32_t* r = row(cell);
    for (int k = 0, n = corners(cell); k < n; ++k) extend(box, x(r[k]));
    return box;
  }
  int face_corners(int64_t face) const { return input->faces[4 * face + 3] >= 0 ? 4 : 3; }
  Box face_box(int64_t face) const {
    Box box = empty_box();
    for (int k = 0, n = face_corners(face); k < n; ++k) extend(box, x(input->faces[4 * face + k]));
    return box;
  }
};

void hex_corners(const Mesh& mesh, int64_t cell, double (&v)[8][3]) {
  const int32_t* r = mesh.row(cell);
  for (int k = 0; k < 8; ++k) {
    for (int a = 0; a < 3; ++a) v[k][a] = mesh.x(r[k])[a];
  }
}

void trilinear(const double (&v)[8][3], const double* xi, double* point, double (*jacobian)[3]) {
  for (int d = 0; d < 3; ++d) {
    point[d] = 0.0;
    for (int a = 0; a < 3; ++a) jacobian[d][a] = 0.0;
  }
  for (int k = 0; k < 8; ++k) {
    double f[3], g[3];
    for (int a = 0; a < 3; ++a) {
      f[a] = kBit[a][k] ? xi[a] : 1.0 - xi[a];
      g[a] = kBit[a][k] ? 1.0 : -1.0;
    }
    const double shape = f[0] * f[1] * f[2];
    const double derivative[3] = {g[0] * f[1] * f[2], f[0] * g[1] * f[2], f[0] * f[1] * g[2]};
    for (int d = 0; d < 3; ++d) {
      point[d] += shape * v[k][d];
      for (int a = 0; a < 3; ++a) jacobian[d][a] += derivative[a] * v[k][d];
    }
  }
}

double determinant(const double (*m)[3]) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// The inverse of m by cofactors; false when m is singular.
bool inverse(const double (*m)[3], double (*out)[3]) {
  const double det = determinant(m);
  if (!(std::fabs(det) > 0.0) || !std::isfinite(det)) return false;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int r0 = (j + 1) % 3, r1 = (j + 2) % 3, c0 = (i + 1) % 3, c1 = (i + 2) % 3;
      out[i][j] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / det;
    }
  }
  return true;
}

// Newton from the cell center for X(xi) = p; false without convergence. A
// step below 1e-8 is in the quadratic regime: one more step reaches rounding
// level, which certification then encloses.
bool hex_newton(const double (&v)[8][3], const double* p, double* xi) {
  xi[0] = xi[1] = xi[2] = 0.5;
  bool polish = false;
  for (int iteration = 0; iteration < 40; ++iteration) {
    double point[3], jacobian[3][3], inv[3][3];
    trilinear(v, xi, point, jacobian);
    if (!inverse(jacobian, inv)) return false;
    const double r[3] = {point[0] - p[0], point[1] - p[1], point[2] - p[2]};
    double step = 0.0;
    for (int a = 0; a < 3; ++a) {
      const double delta = inv[a][0] * r[0] + inv[a][1] * r[1] + inv[a][2] * r[2];
      xi[a] -= delta;
      step = std::max(step, std::fabs(delta));
    }
    if (!std::isfinite(step) || std::fabs(xi[0]) + std::fabs(xi[1]) + std::fabs(xi[2]) > 1e3) {
      return false;
    }
    if (polish) return true;
    polish = step <= 1e-8;
  }
  return false;
}

double hex_volume(const double (&v)[8][3]) {
  // The Jacobian determinant has degree two per axis: two Gauss points are exact.
  const double g[2] = {0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
  double volume = 0.0;
  for (double a : g) {
    for (double b : g) {
      for (double c : g) {
        const double xi[3] = {a, b, c};
        double point[3], jacobian[3][3];
        trilinear(v, xi, point, jacobian);
        volume += determinant(jacobian) / 8.0;
      }
    }
  }
  return std::fabs(volume);
}

Interval ival(double value) { return remesh::point(value); }

// Faces as corner cycles by (axis, side): the corners with that reference bit.
constexpr int kBoundFaces[3][2][4] = {{{0, 3, 7, 4}, {1, 2, 6, 5}},
                                      {{0, 1, 5, 4}, {3, 2, 6, 7}},
                                      {{0, 1, 2, 3}, {4, 5, 6, 7}}};

constexpr int kUndecided = 2;

using ExactVector = std::array<Expansion, 3>;

Expansion exact_det(const ExactVector& x, const ExactVector& y, const ExactVector& z) {
  return x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) +
         x[2] * (y[0] * z[1] - y[1] * z[0]);
}

remesh::IntervalVector interval_difference(const double* b, const double* a) {
  return remesh::enclosed_difference({b[0], b[1], b[2]}, {a[0], a[1], a[2]});
}

// For a root xi* of X(xi) = p in a box over which jac encloses the Jacobian
// and image encloses X, the exact sign of xi*_axis - side (0: on the face).
// The face (corners a, b, c, d in cycle order) lies on the zero set of
//   Q(x) = t det(e1, e2, x - a) + det(e2, g, x - a) det(e1, g, x - a),
// e1 = b - a, e2 = d - a, g = a - b + c - d, t = det(e1, e2, g) (the plane
// det(e1, e2, x - a) when t = 0), and the slice xi_axis = side maps onto it,
// so Q(X(xi*)) = (xi*_axis - side) dQ(X)/dxi_axis somewhere in the box: the
// exact sign of Q(p) times the box's sign of grad Q . dX/dxi_axis.
// kUndecided when the box does not fix that sign.
int face_side(const double (&v)[8][3], const double* p, int axis, int side,
              const Interval (&jac)[3][3], const Interval (&image)[3]) {
  const int* face = kBoundFaces[axis][side];
  const double *a = v[face[0]], *b = v[face[1]], *c = v[face[2]], *d = v[face[3]];
  const remesh::IntervalVector e1 = interval_difference(b, a), e2 = interval_difference(d, a);
  const remesh::IntervalVector ab = interval_difference(a, b), cd = interval_difference(c, d);
  const remesh::IntervalVector g = {remesh::add(ab[0], cd[0]), remesh::add(ab[1], cd[1]),
                                    remesh::add(ab[2], cd[2])};
  const remesh::IntervalVector q = {remesh::subtract(image[0], ival(a[0])),
                                    remesh::subtract(image[1], ival(a[1])),
                                    remesh::subtract(image[2], ival(a[2]))};
  const remesh::IntervalVector derivative = {jac[0][axis], jac[1][axis], jac[2][axis]};
  const int planar = orient(a, b, d, c);
  Interval h;
  int value;
  if (planar == 0) {
    h = remesh::dot(remesh::cross(e1, e2), derivative);
    value = orient(a, b, d, p);
  } else {
    const remesh::IntervalVector n0 = remesh::cross(e1, e2), n1 = remesh::cross(e2, g),
                                 n2 = remesh::cross(e1, g);
    const Interval t = remesh::dot(e1, remesh::cross(e2, g));
    const Interval s1 = remesh::dot(q, remesh::cross(e1, g));
    const Interval s2 = remesh::dot(q, remesh::cross(e2, g));
    // grad Q = t (e1 x e2) + det(e1, g, q) (e2 x g) + det(e2, g, q) (e1 x g);
    // det(x, y, q) = q . (x x y).
    remesh::IntervalVector gradient;
    for (int k = 0; k < 3; ++k) {
      gradient[k] = remesh::add(remesh::add(remesh::multiply(t, n0[k]), remesh::multiply(s1, n1[k])),
                                remesh::multiply(s2, n2[k]));
    }
    h = remesh::dot(gradient, derivative);
    ExactVector x1, x2, xg, xq;
    for (int k = 0; k < 3; ++k) {
      x1[k] = Expansion::difference(b[k], a[k]);
      x2[k] = Expansion::difference(d[k], a[k]);
      xg[k] = Expansion::difference(a[k], b[k]) + Expansion::difference(c[k], d[k]);
      xq[k] = Expansion::difference(p[k], a[k]);
    }
    const Expansion exact = exact_det(x1, x2, xg) * exact_det(x1, x2, xq) +
                            exact_det(x2, xg, xq) * exact_det(x1, xg, xq);
    value = exact.sign();
  }
  const int direction = h.lower > 0.0 ? 1 : (h.upper < 0.0 ? -1 : 0);
  if (direction == 0) return kUndecided;
  return direction * value;
}

// A containment verdict: 1 inside the closed cell, -1 outside, 0 undecided.
// After a successful enclosure, `straddled` marks the faces (bit 2 axis +
// side) whose bound the enclosure straddles and `open` those of them whose
// side stayed undecided.
struct Verdict {
  int value = 0;
  uint8_t open = 0;
  uint8_t straddled = 0;
};

// Krawczyk test of the trilinear inverse at p around xt with box radius r:
// 1 when a unique root is enclosed inside the closed unit cube, -1 when it is
// enclosed outside it, 0 when the test decides nothing.
Verdict krawczyk(const double (&v)[8][3], const double* p, const double* xt, double radius) {
  double point[3], jacobian[3][3], y[3][3];
  trilinear(v, xt, point, jacobian);
  if (!inverse(jacobian, y)) return {};
  Interval box[3], pt[3], one_minus_pt[3], one_minus_box[3];
  for (int a = 0; a < 3; ++a) {
    box[a] = {geometry::next_down(xt[a] - radius), geometry::next_up(xt[a] + radius)};
    pt[a] = ival(xt[a]);
    one_minus_pt[a] = remesh::subtract(ival(1.0), pt[a]);
    one_minus_box[a] = remesh::subtract(ival(1.0), box[a]);
  }
  Interval residual[3] = {ival(0.0), ival(0.0), ival(0.0)};
  Interval image[3] = {ival(0.0), ival(0.0), ival(0.0)};
  Interval jac[3][3];
  for (int d = 0; d < 3; ++d) {
    for (int a = 0; a < 3; ++a) jac[d][a] = ival(0.0);
  }
  for (int k = 0; k < 8; ++k) {
    Interval f[3], fb[3];
    for (int a = 0; a < 3; ++a) {
      f[a] = kBit[a][k] ? pt[a] : one_minus_pt[a];
      fb[a] = kBit[a][k] ? box[a] : one_minus_box[a];
    }
    const Interval shape = remesh::multiply(remesh::multiply(f[0], f[1]), f[2]);
    const Interval shape_box = remesh::multiply(remesh::multiply(fb[0], fb[1]), fb[2]);
    const Interval derivative[3] = {
        remesh::multiply(fb[1], fb[2]), remesh::multiply(fb[0], fb[2]),
        remesh::multiply(fb[0], fb[1])};
    for (int d = 0; d < 3; ++d) {
      const Interval corner = ival(v[k][d]);
      residual[d] = remesh::add(residual[d], remesh::multiply(shape, corner));
      image[d] = remesh::add(image[d], remesh::multiply(shape_box, corner));
      for (int a = 0; a < 3; ++a) {
        const Interval term = remesh::multiply(derivative[a], corner);
        jac[d][a] = kBit[a][k] ? remesh::add(jac[d][a], term) : remesh::subtract(jac[d][a], term);
      }
    }
  }
  for (int d = 0; d < 3; ++d) residual[d] = remesh::subtract(residual[d], ival(p[d]));
  Interval enclosure[3];
  for (int a = 0; a < 3; ++a) {
    Interval k = pt[a];
    for (int d = 0; d < 3; ++d) {
      k = remesh::subtract(k, remesh::multiply(ival(y[a][d]), residual[d]));
    }
    for (int b = 0; b < 3; ++b) {
      Interval m = ival(a == b ? 1.0 : 0.0);
      for (int d = 0; d < 3; ++d) {
        m = remesh::subtract(m, remesh::multiply(ival(y[a][d]), jac[d][b]));
      }
      k = remesh::add(k, remesh::multiply(m, remesh::subtract(box[b], pt[b])));
    }
    if (!remesh::finite(k) || !(k.lower > box[a].lower) || !(k.upper < box[a].upper)) return {};
    enclosure[a] = k;
  }
  // The unique root xi* of the box lies in the enclosure. A bound the
  // enclosure straddles is decided by the exact side of the face surface.
  Verdict verdict{1, 0, 0};
  for (int a = 0; a < 3; ++a) {
    const Interval& k = enclosure[a];
    if (k.upper < 0.0 || k.lower > 1.0) return {-1, 0, 0};
    for (int side = 0; side < 2; ++side) {
      if (k.lower <= side && side <= k.upper) {
        verdict.straddled |= static_cast<uint8_t>(1u << (2 * a + side));
      }
    }
  }
  bool outside = false;
  for (int bit = 0; bit < 6; ++bit) {
    if (!(verdict.straddled & (1u << bit))) continue;
    const int side = bit % 2;
    const int sign = face_side(v, p, bit / 2, side, jac, image);
    if (sign == kUndecided) {
      verdict.value = 0;
      verdict.open |= static_cast<uint8_t>(1u << bit);
    } else if (sign != 0 && (side == 0) != (sign > 0)) {
      outside = true;
    }
  }
  if (outside) verdict.value = -1;
  return verdict;
}

// Certified containment of p in a hexahedron near the float root xt.
Verdict hex_certify(const double (&v)[8][3], const double* p, const double* xt) {
  Verdict last;
  for (double radius : {1e-9, 1e-12, 1e-14}) {
    const Verdict verdict = krawczyk(v, p, xt, radius);
    if (verdict.value != 0) return verdict;
    if (verdict.open != 0) last = verdict;
  }
  return last;
}

// --------------------------------------------------------------- location

struct Location {
  bool candidate = false;  // within the construction band (or undecided)
  bool converged = true;
  std::array<double, 3> reference{};
  double residual = 0.0;
};

void tet_reference(const Mesh& mesh, int64_t cell, const double* p, Location& location) {
  const int32_t* r = mesh.row(cell);
  const double *a = mesh.x(r[0]), *b = mesh.x(r[1]), *c = mesh.x(r[2]), *d = mesh.x(r[3]);
  double e[3][3], q[3];
  for (int k = 0; k < 3; ++k) {
    e[0][k] = b[k] - a[k];
    e[1][k] = c[k] - a[k];
    e[2][k] = d[k] - a[k];
    q[k] = p[k] - a[k];
  }
  const double m[3][3] = {{e[0][0], e[1][0], e[2][0]},
                          {e[0][1], e[1][1], e[2][1]},
                          {e[0][2], e[1][2], e[2][2]}};
  double inv[3][3];
  location.converged = inverse(m, inv);
  if (!location.converged) return;
  for (int a2 = 0; a2 < 3; ++a2) {
    location.reference[a2] = inv[a2][0] * q[0] + inv[a2][1] * q[1] + inv[a2][2] * q[2];
  }
  // Clamped and renormalized barycentric weights, as the Python stencil uses.
  double w[4] = {1.0 - location.reference[0] - location.reference[1] - location.reference[2],
                 location.reference[0], location.reference[1], location.reference[2]};
  double total = 0.0;
  for (double& value : w) {
    value = std::max(value, 0.0);
    total += value;
  }
  double residual = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double value =
        (w[0] * a[k] + w[1] * b[k] + w[2] * c[k] + w[3] * d[k]) / total - p[k];
    residual += value * value;
  }
  location.residual = std::sqrt(residual);
}

// Float location within `band`; with keep_undecided an unconverged hexahedron
// stays a candidate for certification to decide.
void locate(const Mesh& mesh, int64_t cell, const double* p, double band, bool keep_undecided,
            Location& location) {
  if (mesh.hex(cell)) {
    double v[8][3];
    hex_corners(mesh, cell, v);
    location.converged = hex_newton(v, p, location.reference.data());
    if (!location.converged) {
      // Undecided: certification gets its chance only with a converged root.
      location.candidate = keep_undecided;
      return;
    }
    double point[3], jacobian[3][3];
    trilinear(v, location.reference.data(), point, jacobian);
    location.residual = std::hypot(point[0] - p[0], point[1] - p[1], point[2] - p[2]);
    location.candidate = true;
    for (double value : location.reference) {
      location.candidate = location.candidate && value >= -band && value <= 1.0 + band;
    }
    return;
  }
  tet_reference(mesh, cell, p, location);
  if (!location.converged) return;
  const double sum = location.reference[0] + location.reference[1] + location.reference[2];
  location.candidate = sum <= 1.0 + band;
  for (double value : location.reference) location.candidate = location.candidate && value >= -band;
}

// Exact or certified containment (Verdict).
Verdict certify(const Mesh& mesh, int64_t cell, const double* p, const Location& location) {
  const int32_t* r = mesh.row(cell);
  if (!mesh.hex(cell)) {
    const double *a = mesh.x(r[0]), *b = mesh.x(r[1]), *c = mesh.x(r[2]), *d = mesh.x(r[3]);
    const int s = orient(a, b, c, d);
    if (s == 0) return {};
    return {in_tetrahedron(a, b, c, d, s, p) ? 1 : -1, 0};
  }
  if (!location.converged) return {};
  double v[8][3];
  hex_corners(mesh, cell, v);
  return hex_certify(v, p, location.reference.data());
}

// The sorted global vertices of a hexahedron's face (axis, side).
std::array<int32_t, 4> face_key(const Mesh& mesh, int64_t cell, int axis, int side) {
  std::array<int32_t, 4> key{};
  for (int k = 0; k < 4; ++k) key[k] = mesh.row(cell)[kBoundFaces[axis][side][k]];
  std::sort(key.begin(), key.end());
  return key;
}

double cell_measure(const Mesh& mesh, int64_t cell) {
  if (mesh.hex(cell)) {
    double v[8][3];
    hex_corners(mesh, cell, v);
    return hex_volume(v);
  }
  const int32_t* r = mesh.row(cell);
  const double *a = mesh.x(r[0]), *b = mesh.x(r[1]), *c = mesh.x(r[2]), *d = mesh.x(r[3]);
  const double m[3][3] = {{b[0] - a[0], c[0] - a[0], d[0] - a[0]},
                          {b[1] - a[1], c[1] - a[1], d[1] - a[1]},
                          {b[2] - a[2], c[2] - a[2], d[2] - a[2]}};
  return std::fabs(determinant(m)) / 6.0;
}

// ------------------------------------------------------------- cut cells

// Whether every point lies strictly on side -s of plane (a, b, c).
bool all_beyond(const double* a, const double* b, const double* c, int s,
                const double* const* points, int count) {
  for (int k = 0; k < count; ++k) {
    if (orient(a, b, c, points[k]) != -s) return false;
  }
  return true;
}

// Whether plane (a, b, c) separates `first` (on one closed side, not all on
// the plane) from `second` (strictly on the other side).
bool supporting_plane_separates(const double* a, const double* b, const double* c,
                                const double* const* first, int first_count,
                                const double* const* second, int second_count) {
  int side = 0;
  for (int k = 0; k < first_count; ++k) {
    const int s = orient(a, b, c, first[k]);
    if (s == 0) continue;
    if (side == 0) {
      side = s;
    } else if (s != side) {
      return false;
    }
  }
  if (side == 0) {
    // A flat first set: either strict side of its plane separates.
    int other = 0;
    for (int k = 0; k < second_count; ++k) {
      const int s = orient(a, b, c, second[k]);
      if (s == 0 || (other != 0 && s != other)) return false;
      other = s;
    }
    return second_count > 0;
  }
  return all_beyond(a, b, c, side, second, second_count);
}

// Whether a cell and a wall element are proved disjoint (by a supporting plane
// of either through three of its corners); false is a conservative cut.
bool separated(const Mesh& mesh, int64_t cell, int64_t face) {
  const int32_t* r = mesh.row(cell);
  const int cell_count = mesh.corners(cell);
  const double* cell_points[8];
  for (int k = 0; k < cell_count; ++k) cell_points[k] = mesh.x(r[k]);
  const int face_count = mesh.face_corners(face);
  const double* face_points[4];
  for (int k = 0; k < face_count; ++k) face_points[k] = mesh.x(mesh.input->faces[4 * face + k]);
  // Wall planes: the triangle plane, or the corner-hull faces of a quad.
  if (face_count == 3) {
    if (supporting_plane_separates(face_points[0], face_points[1], face_points[2], face_points, 3,
                                   cell_points, cell_count)) {
      return true;
    }
  } else {
    static constexpr int kQuadHull[4][3] = {{0, 1, 2}, {0, 2, 3}, {0, 1, 3}, {1, 2, 3}};
    for (const auto& t : kQuadHull) {
      if (supporting_plane_separates(face_points[t[0]], face_points[t[1]], face_points[t[2]],
                                     face_points, 4, cell_points, cell_count)) {
        return true;
      }
    }
  }
  // Cell planes: tetrahedron faces, or hexahedron face triangles that support the cell.
  if (cell_count == 4) {
    for (const auto& t : kTetFaces) {
      if (supporting_plane_separates(cell_points[t[0]], cell_points[t[1]], cell_points[t[2]],
                                     cell_points, 4, face_points, face_count)) {
        return true;
      }
    }
    return false;
  }
  for (const auto& f : kHexFaces) {
    static constexpr int kSplit[2][3] = {{0, 1, 2}, {0, 2, 3}};
    for (const auto& t : kSplit) {
      if (supporting_plane_separates(cell_points[f[t[0]]], cell_points[f[t[1]]],
                                     cell_points[f[t[2]]], cell_points, 8, face_points,
                                     face_count)) {
        return true;
      }
    }
  }
  return false;
}

// ----------------------------------------------------------------- driver

struct Candidate {
  int32_t tier;  // 0 an active donor, 1 a fringe donor (only with fringe_donors)
  double first, second, measure;
  int32_t name_rank;
  int64_t cell_id;
  int64_t cell;
  int32_t part;
  Location location;
};

bool candidate_before(const Candidate& a, const Candidate& b) {
  if (a.tier != b.tier) return a.tier < b.tier;
  if (a.first != b.first) return a.first < b.first;
  if (a.second != b.second) return a.second < b.second;
  if (a.name_rank != b.name_rank) return a.name_rank < b.name_rank;
  return a.cell_id < b.cell_id;
}

class Assembly {
 public:
  Assembly(const Input& input, const Output& output) : in_(input), out_(output), mesh_{&input} {}

  void run() {
    const auto total = Clock::now();
    std::fill(out_.stats, out_.stats + kStatCount, 0.0);
    const int64_t vertex_count = in_.vertex_offsets[in_.part_count];
    const int64_t cell_count = in_.cell_offsets[in_.part_count];
    hole_.assign(static_cast<size_t>(vertex_count), 0);
    std::fill(out_.wall_contact, out_.wall_contact + vertex_count, uint8_t{0});
    cut_.assign(static_cast<size_t>(cell_count), 0);
    cut_by_.clear();

    auto start = Clock::now();
    for (int32_t part = 0; part < in_.part_count; ++part) prepare_walls(part);
    for (int32_t part = 0; part < in_.part_count; ++part) hole_map(part);
    out_.stats[0] = seconds_since(start);

    start = Clock::now();
    for (int32_t part = 0; part < in_.part_count; ++part) cut_cells(part);
    out_.stats[1] = seconds_since(start);

    start = Clock::now();
    for (int32_t part = 0; part < in_.part_count; ++part) fringe(part);
    out_.stats[2] = seconds_since(start);

    start = Clock::now();
    build_donor_indices();
    out_.stats[3] = seconds_since(start);

    start = Clock::now();
    coverage();
    out_.stats[4] = seconds_since(start);

    start = Clock::now();
    construct_donors();
    out_.stats[5] = seconds_since(start);

    start = Clock::now();
    certify_donors();
    out_.stats[6] = seconds_since(start);
    out_.stats[7] = seconds_since(total);
  }

 private:
  struct Wall {
    std::vector<int64_t> faces;  // role-1 faces
    std::vector<Box> boxes;
    Box box = empty_box();
    Bins rays;  // (y, z) bins of the wall faces
  };

  struct DonorIndex {
    std::vector<int64_t> cells;
    std::vector<Box> boxes;
    Bins bins;
  };

  int32_t part_of_vertex(int64_t vertex) const {
    return static_cast<int32_t>(std::upper_bound(in_.vertex_offsets,
                                                 in_.vertex_offsets + in_.part_count + 1, vertex) -
                                in_.vertex_offsets - 1);
  }

  void prepare_walls(int32_t part) {
    if (walls_.empty()) walls_.resize(static_cast<size_t>(in_.part_count));
    Wall& wall = walls_[part];
    for (int64_t face = in_.face_offsets[part]; face < in_.face_offsets[part + 1]; ++face) {
      if (in_.face_roles[face] != kRoleWall) continue;
      wall.faces.push_back(face);
      wall.boxes.push_back(mesh_.face_box(face));
      extend(wall.box, wall.boxes.back());
    }
    if (!wall.faces.empty()) wall.rays.build(wall.boxes, false);
  }

  void hole_map(int32_t part) {
    for (int32_t other = 0; other < in_.part_count; ++other) {
      const Wall& wall = walls_[other];
      if (other == part || wall.faces.empty()) continue;
      for (int64_t v = in_.vertex_offsets[part]; v < in_.vertex_offsets[part + 1]; ++v) {
        const double* p = mesh_.x(v);
        if (!contains(wall.box, p)) continue;
        bool crossing = false, touching = false;
        wall.rays.visit_point(p, [&](int32_t item) {
          const Box& box = wall.boxes[item];
          if (box.hi[0] < p[0] || p[1] < box.lo[1] || p[1] > box.hi[1] || p[2] < box.lo[2] ||
              p[2] > box.hi[2]) {
            return;
          }
          const int64_t face = wall.faces[item];
          const int32_t* q = in_.faces + 4 * face;
          const double *q0 = mesh_.x(q[0]), *q1 = mesh_.x(q[1]), *q2 = mesh_.x(q[2]);
          ray_triangle(q0, q1, q2, p, crossing, touching);
          if (q[3] < 0) return;
          const double* q3 = mesh_.x(q[3]);
          ray_triangle(q0, q2, q3, p, crossing, touching);
          // The bilinear patch lies in its corner tetrahedron.
          const int s = orient(q0, q1, q2, q3);
          if (s != 0 && contains(box, p) && in_tetrahedron(q0, q1, q2, q3, s, p)) touching = true;
        });
        if (crossing || touching) hole_[v] = 1;
        if (touching) out_.wall_contact[v] = 1;
      }
    }
  }

  bool enclosing(int64_t cell) const {
    const int32_t* r = mesh_.row(cell);
    for (int k = 0, n = mesh_.corners(cell); k < n; ++k) {
      if (hole_[r[k]]) return true;
    }
    return false;
  }

  void cut_cells(int32_t part) {
    const int64_t first = in_.cell_offsets[part], last = in_.cell_offsets[part + 1];
    for (int32_t other = 0; other < in_.part_count; ++other) {
      const Wall& wall = walls_[other];
      if (other == part || wall.faces.empty()) continue;
      std::vector<int64_t> cells;
      std::vector<Box> boxes;
      for (int64_t cell = first; cell < last; ++cell) {
        const Box box = mesh_.cell_box(cell);
        if (overlaps(box, wall.box) && !enclosing(cell)) {
          cells.push_back(cell);
          boxes.push_back(box);
        }
      }
      if (cells.empty()) continue;
      Bins bins;
      bins.build(boxes, true);
      std::vector<int64_t> seen(cells.size(), -1);
      std::vector<uint8_t> hit(cells.size(), 0);
      for (size_t item = 0; item < wall.faces.size(); ++item) {
        const int64_t face = wall.faces[item];
        bins.visit_box(wall.boxes[item], [&](int32_t k) {
          if (hit[k] || seen[k] == static_cast<int64_t>(item)) return;
          seen[k] = static_cast<int64_t>(item);
          const int64_t cell = cells[k];
          if (!overlaps(boxes[k], wall.boxes[item]) || separated(mesh_, cell, face)) return;
          hit[k] = 1;
          cut_[cell] = 1;
          cut_by_.emplace_back(cell, other);
        });
      }
    }
  }

  void fringe(int32_t part) {
    const int64_t v0 = in_.vertex_offsets[part], v1 = in_.vertex_offsets[part + 1];
    const int64_t c0 = in_.cell_offsets[part], c1 = in_.cell_offsets[part + 1];
    std::vector<uint8_t> boundary(static_cast<size_t>(v1 - v0), 0);
    std::vector<uint8_t> protect(static_cast<size_t>(v1 - v0), 0);
    for (int64_t face = in_.face_offsets[part]; face < in_.face_offsets[part + 1]; ++face) {
      const uint8_t role = in_.face_roles[face];
      if (role != kRoleWall && role != kRoleBoundary) continue;
      auto& mark = role == kRoleWall ? protect : boundary;
      for (int k = 0, n = mesh_.face_corners(face); k < n; ++k) mark[in_.faces[4 * face + k] - v0] = 1;
    }
    std::vector<uint8_t> cell_hole(static_cast<size_t>(c1 - c0), 0);
    std::vector<uint8_t> frontier(static_cast<size_t>(v1 - v0), 0);
    for (int64_t cell = c0; cell < c1; ++cell) {
      const bool excluded = in_.cell_excluded[cell] != 0;
      const bool hole = excluded || (cut_[cell] & 1) || enclosing(cell);
      cell_hole[cell - c0] = hole;
      if (!hole) continue;
      const int32_t* r = mesh_.row(cell);
      for (int k = 0, n = mesh_.corners(cell); k < n; ++k) frontier[r[k] - v0] = 1;
    }
    int32_t* layers = out_.fringe_layer + v0;
    for (int64_t v = 0; v < v1 - v0; ++v) {
      frontier[v] = (frontier[v] || boundary[v]) && !hole_[v0 + v];
      layers[v] = frontier[v] ? 1 : 0;
    }
    std::vector<uint8_t> grown(static_cast<size_t>(v1 - v0));
    for (int32_t depth = 2; depth <= in_.fringe_layers; ++depth) {
      std::fill(grown.begin(), grown.end(), uint8_t{0});
      for (int64_t cell = c0; cell < c1; ++cell) {
        if (cell_hole[cell - c0]) continue;
        const int32_t* r = mesh_.row(cell);
        const int n = mesh_.corners(cell);
        bool touches = false;
        for (int k = 0; k < n && !touches; ++k) touches = frontier[r[k] - v0] != 0;
        if (!touches) continue;
        for (int k = 0; k < n; ++k) grown[r[k] - v0] = 1;
      }
      for (int64_t v = 0; v < v1 - v0; ++v) {
        frontier[v] = grown[v] && !hole_[v0 + v] && layers[v] == 0;
        if (frontier[v]) layers[v] = depth;
      }
    }
    for (int64_t v = 0; v < v1 - v0; ++v) {
      if (protect[v] && !hole_[v0 + v]) layers[v] = 0;
      const int8_t status =
          hole_[v0 + v] ? kVertexHole : (layers[v] > 0 ? kVertexReceptor : kVertexActive);
      out_.vertex_status[v0 + v] = status;
      out_.protected_conflict[v0 + v] = protect[v] && status != kVertexActive;
    }
    for (int64_t cell = c0; cell < c1; ++cell) {
      int8_t status = kCellActive;
      if (cell_hole[cell - c0]) {
        status = kCellHole;
      } else {
        const int32_t* r = mesh_.row(cell);
        for (int k = 0, n = mesh_.corners(cell); k < n; ++k) {
          if (layers[r[k] - v0] > 0) status = kCellFringe;
        }
      }
      out_.cell_status[cell] = status;
      // A cut cell with no hole vertex stays ambiguous until coverage() proves it.
      out_.ambiguous[cell] = (cut_[cell] & 1) && !enclosing(cell) ? 1 : 0;
    }
  }

  void build_donor_indices() {
    std::vector<Box> receptor_boxes(static_cast<size_t>(in_.part_count), empty_box());
    for (int32_t part = 0; part < in_.part_count; ++part) {
      for (int64_t v = in_.vertex_offsets[part]; v < in_.vertex_offsets[part + 1]; ++v) {
        if (out_.vertex_status[v] == kVertexReceptor) extend(receptor_boxes[part], mesh_.x(v));
      }
    }
    indices_.resize(static_cast<size_t>(in_.part_count));
    for (int32_t donor = 0; donor < in_.part_count; ++donor) {
      Box query = empty_box();
      for (int32_t part = 0; part < in_.part_count; ++part) {
        if (part != donor && !box_empty(receptor_boxes[part])) extend(query, receptor_boxes[part]);
      }
      DonorIndex& index = indices_[donor];
      if (box_empty(query)) continue;
      for (int64_t cell = in_.cell_offsets[donor]; cell < in_.cell_offsets[donor + 1]; ++cell) {
        const Box box = mesh_.cell_box(cell);
        if (!overlaps(box, query)) continue;
        index.cells.push_back(cell);
        index.boxes.push_back(box);
      }
      index.bins.build(index.boxes, true);
    }
  }

  // Certified containment of p in some cell of `part`.
  bool inside_part(int32_t part, const double* p) const {
    const DonorIndex& index = indices_[part];
    bool found = false;
    index.bins.visit_point(p, [&](int32_t k) {
      if (found || !contains(index.boxes[k], p)) return;
      Location location;
      locate(mesh_, index.cells[k], p, kCandidateBand, true, location);
      found = location.candidate && certify(mesh_, index.cells[k], p, location).value == 1;
    });
    return found;
  }

  // A cut cell C with no hole vertex, cut by part j's wall, is covered when a
  // vertex of C lies (certified) in part j and C's bounds miss every non-wall
  // exterior face of part j: then C stays inside the region bounded by those
  // faces, so its part outside the wall lies in part j.
  void coverage() {
    std::vector<Bins> outer(static_cast<size_t>(in_.part_count));
    std::vector<std::vector<Box>> outer_boxes(static_cast<size_t>(in_.part_count));
    std::vector<uint8_t> outer_built(static_cast<size_t>(in_.part_count), 0);
    std::vector<uint8_t> unproved(static_cast<size_t>(cut_by_.size()), 0);
    for (size_t k = 0; k < cut_by_.size(); ++k) {
      const auto [cell, wall_part] = cut_by_[k];
      if (!out_.ambiguous[cell]) continue;
      if (!outer_built[wall_part]) {
        for (int64_t face = in_.face_offsets[wall_part]; face < in_.face_offsets[wall_part + 1];
             ++face) {
          if (in_.face_roles[face] != kRoleWall) outer_boxes[wall_part].push_back(mesh_.face_box(face));
        }
        outer[wall_part].build(outer_boxes[wall_part], true);
        outer_built[wall_part] = 1;
      }
      const Box box = mesh_.cell_box(cell);
      bool meets = false;
      outer[wall_part].visit_box(box, [&](int32_t item) {
        meets = meets || overlaps(outer_boxes[wall_part][item], box);
      });
      bool anchored = false;
      const int32_t* r = mesh_.row(cell);
      for (int c = 0, n = mesh_.corners(cell); c < n && !anchored && !meets; ++c) {
        anchored = out_.vertex_status[r[c]] == kVertexReceptor && inside_part(wall_part, mesh_.x(r[c]));
      }
      unproved[k] = meets || !anchored;
    }
    // A cell stays ambiguous when any wall that cuts it lacks the proof.
    std::vector<int64_t> still;
    for (size_t k = 0; k < cut_by_.size(); ++k) {
      if (unproved[k]) still.push_back(cut_by_[k].first);
    }
    for (const auto& [cell, wall_part] : cut_by_) {
      (void)wall_part;
      out_.ambiguous[cell] = 0;
    }
    for (int64_t cell : still) out_.ambiguous[cell] = 1;
  }

  void construct_donors() {
    const int64_t vertex_count = in_.vertex_offsets[in_.part_count];
    std::fill(out_.donor_status, out_.donor_status + vertex_count, int32_t{-1});
    std::fill(out_.donor_cell, out_.donor_cell + vertex_count, int64_t{-1});
    std::fill(out_.donor_reference, out_.donor_reference + 3 * vertex_count, 0.0);
    std::fill(out_.donor_residual, out_.donor_residual + vertex_count,
              std::numeric_limits<double>::infinity());
    std::fill(out_.donor_measure, out_.donor_measure + vertex_count,
              std::numeric_limits<double>::infinity());
    std::fill(out_.candidate_parts, out_.candidate_parts + vertex_count, int32_t{0});
    offsets_.assign(1, 0);
    receptors_.clear();
    candidates_.clear();
    blocked_.clear();
    std::vector<uint8_t> part_seen(static_cast<size_t>(in_.part_count));
    double candidates = 0.0;
    for (int32_t part = 0; part < in_.part_count; ++part) {
      for (int64_t v = in_.vertex_offsets[part]; v < in_.vertex_offsets[part + 1]; ++v) {
        if (out_.vertex_status[v] != kVertexReceptor) continue;
        const double* p = mesh_.x(v);
        const size_t begin = candidates_.size();
        bool blocked = false;
        std::fill(part_seen.begin(), part_seen.end(), uint8_t{0});
        for (int32_t donor = 0; donor < in_.part_count; ++donor) {
          if (donor == part) continue;
          const DonorIndex& index = indices_[donor];
          index.bins.visit_point(p, [&](int32_t k) {
            if (!contains(index.boxes[k], p)) return;
            const int64_t cell = index.cells[k];
            Location location;
            locate(mesh_, cell, p, kCandidateBand, true, location);
            if (!location.candidate) return;
            candidates += 1.0;
            const bool hole = out_.cell_status[cell] == kCellHole || out_.ambiguous[cell];
            const bool fringe = out_.cell_status[cell] == kCellFringe;
            if (hole || (fringe && !in_.fringe_donors)) {
              blocked = blocked || location.converged;
              return;
            }
            part_seen[donor] = 1;
            const double measure = cell_measure(mesh_, cell);
            const double priority = -static_cast<double>(in_.priorities[donor]);
            Candidate candidate{};
            candidate.tier = fringe ? 1 : 0;
            candidate.first = in_.donor_priority == 0 ? measure : priority;
            candidate.second = in_.donor_priority == 0 ? priority : measure;
            candidate.name_rank = in_.name_ranks[donor];
            candidate.cell_id = in_.cell_ids[cell];
            candidate.cell = cell;
            candidate.measure = measure;
            candidate.part = donor;
            candidate.location = location;
            candidates_.push_back(candidate);
          });
        }
        std::sort(candidates_.begin() + static_cast<std::ptrdiff_t>(begin), candidates_.end(),
                  candidate_before);
        int32_t seen = 0;
        for (uint8_t flag : part_seen) seen += flag;
        out_.candidate_parts[v] = seen;
        receptors_.push_back(v);
        blocked_.push_back(blocked ? 1 : 0);
        offsets_.push_back(static_cast<int64_t>(candidates_.size()));
      }
    }
    out_.stats[9] = static_cast<double>(receptors_.size());
    out_.stats[12] = candidates;
  }

  // A candidate whose enclosure straddles faces: either some side stayed
  // open (undecided) or a straddled face proved the receptor outside.
  struct StarCell {
    int64_t candidate;
    uint8_t straddled;
    bool open;
  };

  // Whether every straddled face of every star cell is a face of another star
  // cell. The star then surrounds the receptor: each non-straddled bound of
  // each cell holds, and crossing any straddled face leads into another star
  // cell, so the receptor lies in their closed union and, the outside ones
  // excluded, in an open one.
  bool star_closed(const std::vector<StarCell>& star) const {
    bool any_open = false;
    for (const StarCell& member : star) {
      any_open = any_open || member.open;
      const int64_t cell = candidates_[member.candidate].cell;
      for (int bit = 0; bit < 6; ++bit) {
        if (!(member.straddled & (1u << bit))) continue;
        const auto key = face_key(mesh_, cell, bit / 2, bit % 2);
        bool shared = false;
        for (const StarCell& other : star) {
          if (other.candidate == member.candidate) continue;
          for (int b2 = 0; b2 < 6 && !shared; ++b2) {
            shared = face_key(mesh_, candidates_[other.candidate].cell, b2 / 2, b2 % 2) == key;
          }
          if (shared) break;
        }
        if (!shared) return false;
      }
    }
    return any_open;
  }

  void accept(int64_t v, const Candidate& candidate, bool clamp) {
    out_.donor_status[v] = kFound;
    out_.donor_cell[v] = candidate.cell;
    for (int a = 0; a < 3; ++a) {
      const double value = candidate.location.reference[a];
      out_.donor_reference[3 * v + a] = clamp ? std::clamp(value, 0.0, 1.0) : value;
    }
    out_.donor_residual[v] = candidate.location.residual;
    out_.donor_measure[v] = candidate.measure;
  }

  void certify_donors() {
    double orphans = 0.0, repairs = 0.0, fringe_donors = 0.0, shared_faces = 0.0;
    std::vector<StarCell> star;
    for (size_t slot = 0; slot < receptors_.size(); ++slot) {
      const int64_t v = receptors_[slot];
      const double* p = mesh_.x(v);
      int32_t status = blocked_[slot] ? kExcludedDonor : kOutside;
      bool undecided = false;
      star.clear();
      int64_t chosen = -1;
      for (int64_t k = offsets_[slot]; k < offsets_[slot + 1] && chosen < 0; ++k) {
        const Candidate& candidate = candidates_[k];
        bool accepted = false;
        if (in_.prove) {
          const Verdict verdict = certify(mesh_, candidate.cell, p, candidate.location);
          undecided = undecided || verdict.value == 0;
          accepted = verdict.value == 1;
          if ((verdict.value == 0 && verdict.open != 0) ||
              (verdict.value == -1 && verdict.straddled != 0)) {
            star.push_back({k, verdict.straddled, verdict.value == 0});
          }
        } else {
          Location tight;
          locate(mesh_, candidate.cell, p, kUnprovedBand, false, tight);
          accepted = tight.candidate && tight.converged;
        }
        if (accepted) {
          chosen = k;
          accept(v, candidate, false);
        } else {
          repairs += 1.0;
        }
      }
      if (chosen < 0 && star_closed(star)) {
        // The first open star cell in order, its root clamped onto the face.
        for (const StarCell& member : star) {
          if (!member.open) continue;
          chosen = member.candidate;
          break;
        }
        accept(v, candidates_[chosen], true);
        shared_faces += 1.0;
      }
      if (chosen >= 0) {
        fringe_donors += candidates_[chosen].tier;
        continue;
      }
      if (undecided) status = kUnresolved;
      orphans += 1.0;
      out_.donor_status[v] = status;
    }
    out_.stats[10] = orphans;
    out_.stats[11] = repairs;
    out_.stats[14] = fringe_donors;
    out_.stats[15] = shared_faces;
    double holes = 0.0, ambiguous = 0.0;
    for (uint8_t value : hole_) holes += value;
    for (int64_t cell = 0; cell < in_.cell_offsets[in_.part_count]; ++cell) ambiguous += out_.ambiguous[cell];
    out_.stats[8] = holes;
    out_.stats[13] = ambiguous;
  }

  const Input& in_;
  const Output& out_;
  Mesh mesh_;
  std::vector<uint8_t> hole_, cut_;
  std::vector<std::pair<int64_t, int32_t>> cut_by_;
  std::vector<Wall> walls_;
  std::vector<DonorIndex> indices_;
  std::vector<Candidate> candidates_;
  std::vector<int64_t> offsets_, receptors_;
  std::vector<uint8_t> blocked_;
};

}  // namespace

void assemble(const Input& input, const Output& output) { Assembly(input, output).run(); }

}  // namespace phx::mc::overset
