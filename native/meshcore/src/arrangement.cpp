//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact arrangements of triangle surfaces over implicit points.
//
// Every point of the arrangement is either an input vertex or the intersection
// of three planes, each spanned by input vertices: the plane of an input
// triangle, or the plane through two input vertices parallel to a coordinate
// axis ("extruded" plane; two of them represent the line through an input
// edge).  An edge crossing a triangle's plane is a line-plane point, two
// coplanar edges cross in an edge-edge point, and two contact curves inside a
// triangle cross in a triple point of three triangle planes.  These are the
// explicit/implicit point families of M. Attene, "Indirect predicates for
// geometric constructions", Computer-Aided Design 126 (2020), and the
// arrangement follows the per-triangle construction of M. Cherchi, M. Livesu,
// R. Scateni and M. Attene, "Fast and robust mesh arrangements using
// floating-point arithmetic", ACM TOG 39(6) (2020); both are independently
// implemented here.
//
// An implicit point is carried by homogeneous coordinates (X, Y, Z, W) from
// Cramer's rule, polynomials of degree at most 7 in the input coordinates.
// Every predicate (orientation in a triangle's plane, coordinate order, point
// equality, plane side) is a polynomial sign.  It is decided by the dynamic
// forward-error filter of filtered.hpp and, when the filter cannot certify
// the sign, by exact dyadic big-integer arithmetic.  Floating-point expansions
// are not used for the fallback: triple-point predicates reach degree 20, whose
// expansion products underflow for ordinary binary64 inputs, while dyadic
// integers are exact for every finite input. Constraint topology and geometric
// classifications never depend on rounded coordinates. Publication uses exact
// nearest rounding with a rigorous max-norm bound and may choose a different
// exact-legal unconstrained cell diagonal to avoid a collapsed carrier ear.
// Remaining collapsed/inverted carriers are refused; distinct exact points
// are never merged to repair publication.
//
// Each triangle t is split independently in the coordinate plane that drops
// the dominant axis of its normal.  Its constraints are its three edges, the
// segment u ∩ plane(t) of every transversal candidate triangle u of another
// surface, and the edges of every coplanar candidate.  Constraints are split
// at every point lying in their relative interior and at every proper crossing
// (creating the triple/edge-edge points) until the planar graph is embedded;
// the part inside the closed triangle is triangulated by point insertion and
// Sloan's flip-based constraint recovery with exact orientations.  Points are
// finally welded across triangles by exact equality (candidate pairs from the
// rigorous rounding boxes), so all triangles sharing a point share one vertex.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bounded_memory.hpp"
#include "capi_guard.hpp"
#include "counter_records.hpp"
#include "dyadic.hpp"
#include "filtered.hpp"
#include "phydrax_meshcore.h"
#include "exact_line.hpp"
#include "exact_scalar.hpp"
#include "geometry3.hpp"
#include "facet_coverage.hpp"
#include "topology_keys.hpp"

namespace phx::mc {
namespace {

using exact::det3;
using exact::lift;

struct LineWork {
  bool (*spend)(void*, std::int64_t);
  void* context;
  void charge(std::size_t visits = 1) const {
    if (visits > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        !spend(context, static_cast<std::int64_t>(visits))) {
      throw ExecutionRefusal{native_execution_status(PHX_MC_CAPACITY_EXCEEDED)};
    }
  }
};

void line_shift_right(Limbs& value, std::int64_t bits, const LineWork& work) {
  work.charge(value.size());
  const std::size_t words = static_cast<std::size_t>(bits / 32);
  const int rest = static_cast<int>(bits % 32);
  if (words >= value.size()) {
    value.clear();
    return;
  }
  const std::size_t count = value.size() - words;
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t pair = static_cast<std::uint64_t>(value[i + words]) |
        (i + words + 1 < value.size() ? static_cast<std::uint64_t>(value[i + words + 1]) << 32 : 0);
    value[i] = static_cast<std::uint32_t>(pair >> rest);
  }
  value.resize(count);
  trim(value);
}

void line_subtract(Limbs& value, const Limbs& other, const LineWork& work) {
  work.charge(value.size());
  std::int64_t borrow = 0;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const std::int64_t next = static_cast<std::int64_t>(value[i]) -
        (i < other.size() ? other[i] : 0) - borrow;
    value[i] = static_cast<std::uint32_t>(next);
    borrow = next < 0 ? 1 : 0;
  }
  trim(value);
}

Limbs line_divide(const Limbs& numerator, const Limbs& denominator,
                  Limbs& remainder, const LineWork& work) {
  work.charge(numerator.size() + denominator.size());
  remainder = numerator;
  const std::int64_t shift = bit_length(numerator) - bit_length(denominator);
  if (shift < 0) return {};
  Limbs divisor = shift_left(denominator, shift);
  Limbs quotient(static_cast<std::size_t>(shift / 32 + 1), 0);
  for (std::int64_t bit = shift; bit >= 0; --bit) {
    work.charge(remainder.size() + divisor.size());
    if (compare_magnitude(remainder, divisor) >= 0) {
      line_subtract(remainder, divisor, work);
      quotient[static_cast<std::size_t>(bit / 32)] |= std::uint32_t{1} << (bit % 32);
    }
    line_shift_right(divisor, 1, work);
  }
  trim(quotient);
  return quotient;
}

Limbs line_gcd(Limbs first, Limbs second, const LineWork& work) {
  while (!second.empty()) {
    Limbs remainder;
    line_divide(first, second, remainder, work);
    first = std::move(second);
    second = std::move(remainder);
  }
  return first;
}

Limbs line_low_bits(Limbs value, std::int64_t bits, const LineWork& work) {
  work.charge(value.size());
  if (bits == 0) return {};
  const std::size_t words = static_cast<std::size_t>((bits + 31) / 32);
  if (value.size() > words) value.resize(words);
  if (bits % 32 && value.size() == words) value.back() &= (std::uint32_t{1} << (bits % 32)) - 1;
  trim(value);
  return value;
}

Limbs line_modular_negative(Limbs value, std::int64_t bits, const LineWork& work) {
  value = line_low_bits(std::move(value), bits, work);
  if (value.empty()) return value;
  Limbs modulus = shift_left(Limbs{1}, bits);
  line_subtract(modulus, value, work);
  return modulus;
}

Limbs line_modular_integer(const Dyadic& value, std::int64_t bits, const LineWork& work) {
  if (value.sign() == 0) return {};
  if (value.exponent() < 0) throw std::invalid_argument("A line lattice coefficient must be integral.");
  work.charge(value.magnitude().size());
  Limbs out = line_low_bits(shift_left(value.magnitude(), value.exponent()), bits, work);
  return value.sign() < 0 ? line_modular_negative(std::move(out), bits, work) : out;
}

Limbs line_modular_product(const Limbs& a, const Limbs& b,
                          std::int64_t bits, const LineWork& work) {
  work.charge(a.size() * b.size());
  return line_low_bits(multiply_magnitude(a, b), bits, work);
}

Limbs line_odd_inverse(const Dyadic& odd, std::int64_t bits, const LineWork& work) {
  Limbs inverse{1};
  for (std::int64_t precision = 1; precision < bits;) {
    precision = std::min(bits, 2 * precision);
    const Limbs coefficient = line_modular_integer(odd, precision, work);
    Limbs adjustment = line_modular_negative(
        line_modular_product(coefficient, inverse, precision, work), precision, work);
    adjustment = line_low_bits(add_magnitude(adjustment, Limbs{2}), precision, work);
    inverse = line_modular_product(inverse, adjustment, precision, work);
  }
  return line_low_bits(std::move(inverse), bits, work);
}

Dyadic line_nearest_integer(const Dyadic& numerator, const Dyadic& denominator,
                            const LineWork& work) {
  Limbs n = numerator.magnitude(), d = denominator.magnitude();
  const std::int64_t shift = numerator.exponent() - denominator.exponent();
  if (shift >= 0) n = shift_left(n, shift);
  else d = shift_left(d, -shift);
  Limbs remainder;
  Limbs quotient = line_divide(n, d, remainder, work);
  const int midpoint = compare_magnitude(shift_left(remainder, 1), d);
  if (midpoint > 0 || (midpoint == 0 && !quotient.empty() && (quotient[0] & 1))) {
    quotient = add_magnitude(quotient, Limbs{1});
  }
  return Dyadic::integer(quotient.data(), quotient.size(), numerator.sign() * denominator.sign());
}

  // The leading-integer quotient is a close candidate, not the correctly
  // rounded rational. Exact residual signs locate its two neighboring values;
  // their residual sum decides the midpoint, including ties to even.
  static double nearest(const Dyadic& numerator, const Dyadic& denominator, double x) {
    Dyadic residual = numerator - Dyadic(x) * denominator;
    while (residual.sign() != 0) {
      const int direction = residual.sign() * denominator.sign();
      const double next = std::nextafter(
          x, direction > 0 ? std::numeric_limits<double>::infinity()
                           : -std::numeric_limits<double>::infinity());
      if (!std::isfinite(next)) {
        break;
      }
      Dyadic next_residual = numerator - Dyadic(next) * denominator;
      if (next_residual.sign() == 0) {
        return next;
      }
      if (next_residual.sign() == residual.sign()) {
        x = next;
        residual = std::move(next_residual);
        continue;
      }
      const int closer = (residual + next_residual).sign() * residual.sign();
      if (closer > 0 || (closer == 0 && (std::bit_cast<std::uint64_t>(next) & 1u) == 0)) {
        x = next;
      }
      break;
    }
    return x;
  }
// ------------------------------------------------------------ implicit points


// kind 0: the plane through input vertices a, b, c; kind 1: the plane through
// a and b (a < b) parallel to coordinate axis `axis`.
struct PlaneDef {
  std::int64_t kind = 0;
  std::int64_t axis = 0;
  std::int64_t a = 0;
  std::int64_t b = 0;
  std::int64_t c = -1;

  std::array<std::int64_t, 5> key() const { return {kind, axis, a, b, c}; }
};

// Construction families of published points.
enum Construction : std::int8_t {
  kInputVertex = 0,
  kEdgePlane = 1,
  kEdgeEdge = 2,
  kTriplePlane = 3,
};

struct PointDef {
  bool implicit = false;
  std::int64_t vertex = -1;
  std::array<PlaneDef, 3> planes{};
};

template <class T>
void plane_value(const double* coords, const PlaneDef& plane, T normal[3], T& offset) {
  const double* a = coords + 3 * plane.a;
  const double* b = coords + 3 * plane.b;
  const T base[3] = {lift<T>(a[0]), lift<T>(a[1]), lift<T>(a[2])};
  const T first[3] = {lift<T>(b[0]) - base[0], lift<T>(b[1]) - base[1], lift<T>(b[2]) - base[2]};
  if (plane.kind == 0) {
    const double* c = coords + 3 * plane.c;
    const T second[3] = {lift<T>(c[0]) - base[0], lift<T>(c[1]) - base[1],
                         lift<T>(c[2]) - base[2]};
    normal[0] = first[1] * second[2] - first[2] * second[1];
    normal[1] = first[2] * second[0] - first[0] * second[2];
    normal[2] = first[0] * second[1] - first[1] * second[0];
  } else {
    // first x e_axis
    const T zero = lift<T>(0.0);
    switch (plane.axis) {
      case 0:
        normal[0] = zero;
        normal[1] = first[2];
        normal[2] = -first[1];
        break;
      case 1:
        normal[0] = -first[2];
        normal[1] = zero;
        normal[2] = first[0];
        break;
      default:
        normal[0] = first[1];
        normal[1] = -first[0];
        normal[2] = zero;
        break;
    }
  }
  offset = normal[0] * base[0] + normal[1] * base[1] + normal[2] * base[2];
}

template <class T>
void homogeneous(const double* coords, const PointDef& point, T h[4]) {
  if (!point.implicit) {
    const double* x = coords + 3 * point.vertex;
    h[0] = lift<T>(x[0]);
    h[1] = lift<T>(x[1]);
    h[2] = lift<T>(x[2]);
    h[3] = lift<T>(1.0);
    return;
  }
  T n[3][3];
  T o[3];
  for (int row = 0; row < 3; ++row) {
    plane_value(coords, point.planes[row], n[row], o[row]);
  }
  h[3] = det3(n[0][0], n[0][1], n[0][2], n[1][0], n[1][1], n[1][2], n[2][0], n[2][1], n[2][2]);
  h[0] = det3(o[0], n[0][1], n[0][2], o[1], n[1][1], n[1][2], o[2], n[2][1], n[2][2]);
  h[1] = det3(n[0][0], o[0], n[0][2], n[1][0], o[1], n[1][2], n[2][0], o[2], n[2][2]);
  h[2] = det3(n[0][0], n[0][1], o[0], n[1][0], n[1][1], o[1], n[2][0], n[2][1], o[2]);
}

struct PoolPoint {
  PointDef def;
  std::int8_t construction = kInputVertex;
  Approx h[4];
  int w_sign = 1;
  NativeVector<Dyadic> exact;
  double x[3] = {0.0, 0.0, 0.0};
  double bound = 0.0;  // max-norm distance of x from the exact point

  double lower(int axis) const {
    return bound == 0.0 ? x[axis]
                        : geometry::next_down(x[axis] - bound);
  }

  double upper(int axis) const {
    return bound == 0.0 ? x[axis]
                        : geometry::next_up(x[axis] + bound);
  }
};

// A refusal carrying its call status and the offending triangle.
struct Refusal {
  std::int32_t status;
  std::int64_t face;
};

// Point pool and exact predicates.  Pool ids [0, vertex_count) are the input
// vertices; implicit points are interned by their defining planes.
class Kernel {
 public:
  Kernel(const double* coords, std::int64_t vertex_count, std::int64_t max_points,
         bool bounded_work = false)
      : coords_(coords), max_points_(max_points), bounded_work_(bounded_work) {
    for (std::int64_t vertex = 0; vertex < vertex_count; ++vertex) {
      PoolPoint& point = pool_.emplace_back();
      point.def.vertex = vertex;
      homogeneous(coords_, point.def, point.h);
      for (int axis = 0; axis < 3; ++axis) {
        point.x[axis] = coords_[3 * vertex + axis];
      }
    }
  }

  std::int64_t size() const { return static_cast<std::int64_t>(pool_.size()); }
  void work() const {
    if (bounded_work_) native_execution_charge(1);
  }
  const PoolPoint& point(std::int64_t id) const { return pool_[static_cast<std::size_t>(id)]; }
  bool is_input(std::int64_t id) const { return !pool_[static_cast<std::size_t>(id)].def.implicit; }
  std::int64_t filtered() const { return filtered_; }
  std::int64_t exact() const { return exact_; }

  std::int64_t intern(PointDef def, std::int8_t construction, std::int64_t face) {
    def.implicit = true;
    std::sort(def.planes.begin(), def.planes.end(),
              [](const PlaneDef& left, const PlaneDef& right) { return left.key() < right.key(); });
    std::array<std::int64_t, 15> key{};
    for (int row = 0; row < 3; ++row) {
      const auto part = def.planes[row].key();
      std::copy(part.begin(), part.end(), key.begin() + 5 * row);
    }
    const auto found = interned_.find(key);
    if (found != interned_.end()) {
      return found->second;
    }
    if (size() >= max_points_) {
      throw Refusal{PHX_MC_CAPACITY_EXCEEDED, face};
    }
    const std::int64_t id = size();
    PoolPoint& point = pool_.emplace_back();
    point.def = def;
    point.construction = construction;
    homogeneous(coords_, def, point.h);
    const int w = decide([&]<class T>() -> T { return hom<T>(id)[3]; });
    if (w == 0) {
      // Defining planes are independent by construction (proper crossings,
      // strictly separated edge ends); a singular system is a kernel defect.
      throw Refusal{PHX_MC_INTERNAL_ERROR, face};
    }
    point.w_sign = w;
    round(point, id);
    interned_.emplace(key, id);
    return id;
  }

  // Sign of det[(p_i, p_j, 1), (q_i, q_j, 1), (r_i, r_j, 1)].
  int orient2d(std::int64_t p, std::int64_t q, std::int64_t r, int i, int j) {
    const int sign = decide([&]<class T>() -> T {
      const T* a = hom<T>(p);
      const T* b = hom<T>(q);
      const T* c = hom<T>(r);
      return det3(a[i], a[j], a[3], b[i], b[j], b[3], c[i], c[j], c[3]);
    });
    return sign * w(p) * w(q) * w(r);
  }

  // Publication certification only: topology remains over the implicit
  // points. A rounded carrier must preserve its certified planar orientation.
  int rounded_orient2d(std::int64_t p, std::int64_t q, std::int64_t r, int i, int j) {
    const double* a = point(p).x;
    const double* b = point(q).x;
    const double* c = point(r).x;
    return decide([&]<class T>() -> T {
      return (lift<T>(a[i]) - lift<T>(c[i])) * (lift<T>(b[j]) - lift<T>(c[j])) -
             (lift<T>(a[j]) - lift<T>(c[j])) * (lift<T>(b[i]) - lift<T>(c[i]));
    });
  }

  // Sign of p_axis - q_axis.
  int compare(std::int64_t p, std::int64_t q, int axis) {
    const int sign = decide([&]<class T>() -> T {
      const T* a = hom<T>(p);
      const T* b = hom<T>(q);
      return a[axis] * b[3] - b[axis] * a[3];
    });
    return sign * w(p) * w(q);
  }

  bool equal(std::int64_t p, std::int64_t q) {
    work();
    if (p == q) {
      return true;
    }
    const PoolPoint& a = point(p);
    const PoolPoint& b = point(q);
    if (!a.def.implicit && !b.def.implicit) {
      return a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2];
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (a.upper(axis) < b.lower(axis) || b.upper(axis) < a.lower(axis)) {
        return false;
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (compare(p, q, axis) != 0) {
        return false;
      }
    }
    return true;
  }

  // Sign of the side of input vertex `vertex` relative to a triangle plane
  // (orient3d of the plane's three vertices and the point).
  int side(const PlaneDef& plane, std::int64_t vertex) {
    return decide([&]<class T>() -> T {
      T normal[3];
      T offset;
      plane_value(coords_, plane, normal, offset);
      const double* x = coords_ + 3 * vertex;
      return normal[0] * lift<T>(x[0]) + normal[1] * lift<T>(x[1]) + normal[2] * lift<T>(x[2]) -
             offset;
    });
  }

  // Sign and filtered magnitude of one normal component of a triangle plane.
  int normal_sign(const PlaneDef& plane, int axis, double& magnitude) {
    Approx normal[3];
    Approx offset;
    plane_value(coords_, plane, normal, offset);
    magnitude = std::fabs(normal[axis].value);
    return decide([&]<class T>() -> T {
      T values[3];
      T unused;
      plane_value(coords_, plane, values, unused);
      return values[axis];
    });
  }

  // Exact winding number, at the centroid of pool points p, q and r, of the
  // input-vertex triangles faces[0..face_count), counted along the ray from
  // the centroid in direction +axis.  The centroid is perturbed symbolically
  // by (eps, eps^2) along the transverse axes (axis + 1, axis + 2): the
  // perturbed ray meets no triangle edge or vertex and no triangle parallel to
  // it, so a hit through a shared edge or vertex is counted exactly once and
  // the signed count is the exact winding number of a closed surface.
  // `normal_signs` caches each triangle's normal component signs (2 =
  // unknown).  Returns false when the centroid lies on one of the triangles,
  // where no winding number exists.
  bool winding(std::int64_t p, std::int64_t q, std::int64_t r, int axis,
               const std::int64_t* triangles, const std::int64_t* faces,
               std::size_t face_count, NativeVector<std::int8_t>& normal_signs, int& value) {
    const int i = (axis + 1) % 3;
    const int j = (axis + 2) % 3;
    Approx approx[4];
    centroid(p, q, r, approx);
    std::optional<std::array<Dyadic, 4>> exact;
    const auto center = [&]<class T>() -> const T* {
      if constexpr (std::is_same_v<T, Approx>) {
        return approx;
      } else {
        if (!exact) {
          exact.emplace();
          centroid(p, q, r, exact->data());
        }
        return exact->data();
      }
    };
    const int w_sign = w(p) * w(q) * w(r);
    // The exact centroid lies in the hull of its exact corners.
    double lower[3];
    double upper[3];
    for (int k = 0; k < 3; ++k) {
      lower[k] = std::min({point(p).lower(k), point(q).lower(k), point(r).lower(k)});
      upper[k] = std::max({point(p).upper(k), point(q).upper(k), point(r).upper(k)});
    }
    // Sign of the perturbed projected orientation of (u, v, centroid).
    const auto edge_side = [&](const double* u, const double* v) {
      const int sign = decide([&]<class T>() -> T {
        const T* h = center.template operator()<T>();
        const T ui = lift<T>(u[i]);
        const T uj = lift<T>(u[j]);
        return (lift<T>(v[i]) - ui) * (h[j] - uj * h[3]) -
               (lift<T>(v[j]) - uj) * (h[i] - ui * h[3]);
      });
      if (sign != 0) {
        return sign * w_sign;
      }
      if (v[j] != u[j]) {
        return v[j] > u[j] ? -1 : 1;
      }
      return v[i] > u[i] ? 1 : -1;
    };
    value = 0;
    for (std::size_t index = 0; index < face_count; ++index) {
      const std::int64_t face = faces[index];
      const std::int64_t* v = triangles + 3 * face;
      const double* corner[3] = {coords_ + 3 * v[0], coords_ + 3 * v[1], coords_ + 3 * v[2]};
      bool missed = std::max({corner[0][axis], corner[1][axis], corner[2][axis]}) < lower[axis];
      for (const int t : {i, j}) {
        missed = missed || std::max({corner[0][t], corner[1][t], corner[2][t]}) < lower[t] ||
                 std::min({corner[0][t], corner[1][t], corner[2][t]}) > upper[t];
      }
      if (missed) {
        continue;
      }
      const PlaneDef plane{0, 0, v[0], v[1], v[2]};
      std::int8_t& cached = normal_signs[static_cast<std::size_t>(3 * face + axis)];
      if (cached == 2) {
        double magnitude = 0.0;
        cached = static_cast<std::int8_t>(normal_sign(plane, axis, magnitude));
      }
      const int normal = cached;
      if (normal == 0 || edge_side(corner[0], corner[1]) != normal ||
          edge_side(corner[1], corner[2]) != normal ||
          edge_side(corner[2], corner[0]) != normal) {
        continue;
      }
      const int side = w_sign * decide([&]<class T>() -> T {
        const T* h = center.template operator()<T>();
        T n[3];
        T offset;
        plane_value(coords_, plane, n, offset);
        return n[0] * h[0] + n[1] * h[1] + n[2] * h[2] - offset * h[3];
      });
      if (side == 0) {
        return false;
      }
      // The hit lies ahead of the centroid exactly when the centroid is
      // behind the plane relative to the normal's axis component.
      if (side * normal < 0) {
        value += normal;
      }
    }
    return true;
  }

 private:
  template <class Eval>
  int decide(Eval&& eval) {
    if (bounded_work_) {
      native_execution_charge(1);
      native_execution_primitive_query();
    }
    const Approx filtered = eval.template operator()<Approx>();
    const int sign = filtered.certified_sign();
    if (sign != 2) {
      ++filtered_;
      return sign;
    }
    ++exact_;
    return eval.template operator()<Dyadic>().sign();
  }

  template <class T>
  const T* hom(std::int64_t id) {
    PoolPoint& point = pool_[static_cast<std::size_t>(id)];
    if constexpr (std::is_same_v<T, Approx>) {
      return point.h;
    } else {
      if (point.exact.empty()) {
        point.exact.resize(4);
        homogeneous(coords_, point.def, point.exact.data());
      }
      return point.exact.data();
    }
  }

  int w(std::int64_t id) const { return pool_[static_cast<std::size_t>(id)].w_sign; }

  // Homogeneous centroid (sum of corners over three) of pool points p, q, r.
  template <class T>
  void centroid(std::int64_t p, std::int64_t q, std::int64_t r, T out[4]) {
    const T* a = hom<T>(p);
    const T* b = hom<T>(q);
    const T* c = hom<T>(r);
    const T bc = b[3] * c[3];
    const T ac = a[3] * c[3];
    const T ab = a[3] * b[3];
    for (int k = 0; k < 3; ++k) {
      out[k] = a[k] * bc + b[k] * ac + c[k] * ab;
    }
    out[3] = lift<T>(3.0) * a[3] * bc;
  }

  // Bounds use directed outward operations, including gradual underflow.
  static double upward(double value) {
    return std::nextafter(value, std::numeric_limits<double>::infinity());
  }

  // A truncated 64-bit leading integer, rounded to binary64, differs from the
  // exact scaled magnitude by less than one adjacent binary64 spacing. Thus
  // nextafter brackets it even when the discarded integer tail is nonzero.
  static double quotient_bound(const Dyadic& numerator, const Dyadic& denominator) {
    if (numerator.sign() == 0) {
      return 0.0;
    }
    std::int64_t numerator_exponent = 0;
    std::int64_t denominator_exponent = 0;
    const double upper = upward(std::fabs(numerator.fraction(numerator_exponent)));
    const double lower = std::nextafter(
        std::fabs(denominator.fraction(denominator_exponent)), 0.0);
    const double ratio = upward(upper / lower);
    const auto shift = std::clamp<std::int64_t>(
        numerator_exponent - denominator_exponent,
        std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    return upward(std::ldexp(ratio, static_cast<int>(shift)));
  }


  // A canonical correctly rounded point avoids representation-dependent ULP
  // bias between equal LPI/TPI constructions. Its max-norm bound is computed
  // from the exact residual, not a relative-error heuristic.
  void round(PoolPoint& point, std::int64_t id) {
    const Dyadic* exact = hom<Dyadic>(id);
    std::int64_t w_exponent = 0;
    const double w_fraction = exact[3].fraction(w_exponent);
    double bound = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      std::int64_t exponent = 0;
      const double fraction = exact[axis].fraction(exponent);
      if (fraction == 0.0) {
        point.x[axis] = 0.0;
        continue;
      }
      const auto shift = std::clamp<std::int64_t>(
          exponent - w_exponent, std::numeric_limits<int>::min(),
          std::numeric_limits<int>::max());
      const double candidate = std::ldexp(fraction / w_fraction, static_cast<int>(shift));
      // Outside-face intersections may exceed binary64. Their finite saturated
      // candidate and outward bound keep all rejection conservative until
      // exact clipping removes them.
      const double initial = std::isfinite(candidate)
                                 ? candidate
                                 : std::copysign(std::numeric_limits<double>::max(), candidate);
      const double x = nearest(exact[axis], exact[3], initial);
      point.x[axis] = x;
      const Dyadic residual = exact[axis] - Dyadic(x) * exact[3];
      bound = std::max(bound, quotient_bound(residual, exact[3]));
    }
    point.bound = bound;
  }

  const double* coords_;
  std::int64_t max_points_;
  bool bounded_work_;
  NativeDeque<PoolPoint> pool_;
  NativeMap<std::array<std::int64_t, 15>, std::int64_t> interned_;
  std::int64_t filtered_ = 0;
  std::int64_t exact_ = 0;
};

// ------------------------------------------------------------ one triangle

// Supporting line of a constraint inside triangle t: kind 0 the line through
// input vertices a < b (lying in plane(t)), kind 1 plane(t) ∩ plane(face).
struct Line {
  int kind = 0;
  std::int64_t a = -1;
  std::int64_t b = -1;
  std::int64_t face = -1;
};

struct Segment {
  int a = 0;
  int b = 0;
  Line line;
  bool contact = false;
};

class LocalMesh {
 public:
  int add(int a, int b, int c) {
    const int index = static_cast<int>(triangles_.size());
    triangles_.push_back({a, b, c});
    alive_.push_back(1);
    owner_[key(a, b)] = index;
    owner_[key(b, c)] = index;
    owner_[key(c, a)] = index;
    return index;
  }

  void remove(int index) {
    alive_[static_cast<std::size_t>(index)] = 0;
    const auto& t = triangles_[static_cast<std::size_t>(index)];
    for (int k = 0; k < 3; ++k) {
      const auto found = owner_.find(key(t[k], t[(k + 1) % 3]));
      if (found != owner_.end() && found->second == index) {
        owner_.erase(found);
      }
    }
  }

  int find(int u, int v) const {
    const auto found = owner_.find(key(u, v));
    return found == owner_.end() ? -1 : found->second;
  }

  const std::array<int, 3>& triangle(int index) const {
    return triangles_[static_cast<std::size_t>(index)];
  }
  bool alive(int index) const { return alive_[static_cast<std::size_t>(index)] != 0; }
  int count() const { return static_cast<int>(triangles_.size()); }

  static int opposite(const std::array<int, 3>& t, int u, int v) {
    for (int vertex : t) {
      if (vertex != u && vertex != v) {
        return vertex;
      }
    }
    return -1;
  }

 private:
  static std::uint64_t key(int u, int v) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
           static_cast<std::uint32_t>(v);
  }

  NativeVector<std::array<int, 3>> triangles_;
  NativeVector<char> alive_;
  NativeUnorderedMap<std::uint64_t, int> owner_;
};

struct Output {
  NativeVector<std::array<std::int64_t, 3>> fragments;  // pool ids
  NativeVector<std::int64_t> fragment_faces;
  NativeVector<std::int8_t> fragment_frames;  // sign * (dropped axis + 1)
  NativeVector<std::array<std::int64_t, 2>> contact_edges;        // pool ids
  NativeVector<std::array<std::int64_t, 3>> coplanar;             // fragment, face, orientation
  NativeVector<std::array<std::int64_t, 3>> locations;            // pool id, face, feature code
  std::int64_t contact_pairs = 0;
  std::int64_t coplanar_pairs = 0;
  std::int64_t split_faces = 0;
};

class Arranger {
 public:
  Arranger(const double* coords, std::int64_t vertex_count, const std::int64_t* triangles,
           std::int64_t max_points, std::int64_t max_fragments, bool coverage_only = false)
      : coords_(coords),
        triangles_(triangles),
        kernel_(coords, vertex_count, max_points, coverage_only),
        max_fragments_(max_fragments), coverage_only_(coverage_only) {}

  Kernel& kernel() { return kernel_; }
  Output& output() { return output_; }

  void face(std::int64_t t, std::span<const std::int64_t> partners) {
    face_ = t;
    const std::int64_t* corners = triangles_ + 3 * t;
    const PlaneDef plane = triangle_plane(t);
    frame(plane);
    points_.assign(corners, corners + 3);
    segments_.clear();
    for (int k = 0; k < 3; ++k) {
      segments_.push_back({k, (k + 1) % 3, edge_line(corners[k], corners[(k + 1) % 3]), false});
    }
    NativeVector<std::int64_t> coplanar;
    for (const std::int64_t u : partners) {
      add_partner(t, plane, u, coplanar);
    }
    embed();
    clip();
    triangulate(t, coplanar);
  }

 private:
  PlaneDef triangle_plane(std::int64_t t) const {
    const std::int64_t* v = triangles_ + 3 * t;
    return {0, 0, v[0], v[1], v[2]};
  }

  PlaneDef extruded(std::int64_t a, std::int64_t b, int axis) const {
    return {1, axis, std::min(a, b), std::max(a, b), -1};
  }

  // Two independent extruded planes whose intersection is the line (a, b):
  // the axes other than the dominant axis of b - a (whose component is
  // nonzero exactly because its rounded difference is).
  void line_planes(std::int64_t a, std::int64_t b, PlaneDef* out) const {
    const std::int64_t low = std::min(a, b);
    const std::int64_t high = std::max(a, b);
    int dominant = 0;
    double largest = -1.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double difference = std::fabs(coords_[3 * high + axis] - coords_[3 * low + axis]);
      if (difference > largest) {
        largest = difference;
        dominant = axis;
      }
    }
    out[0] = extruded(low, high, (dominant + 1) % 3);
    out[1] = extruded(low, high, (dominant + 2) % 3);
  }

  static Line edge_line(std::int64_t a, std::int64_t b) {
    return {0, std::min(a, b), std::max(a, b), -1};
  }

  void frame(const PlaneDef& plane) {
    int chosen = -1;
    int chosen_sign = 0;
    double best = -1.0;
    for (int axis = 0; axis < 3; ++axis) {
      double magnitude = 0.0;
      const int sign = kernel_.normal_sign(plane, axis, magnitude);
      if (sign != 0 && (chosen < 0 || magnitude > best)) {
        best = magnitude;
        chosen = axis;
        chosen_sign = sign;
      }
    }
    if (chosen < 0) {
      throw Refusal{PHX_MC_DEGENERATE_INPUT, face_};
    }
    axis_ = chosen;
    face_sign_ = chosen_sign;
    first_axis_ = (chosen + 1) % 3;
    second_axis_ = (chosen + 2) % 3;
  }

  int orient(int p, int q, int r) {
    return face_sign_ * kernel_.orient2d(points_[static_cast<std::size_t>(p)],
                                         points_[static_cast<std::size_t>(q)],
                                         points_[static_cast<std::size_t>(r)], first_axis_,
                                         second_axis_);
  }

  int orient_pool(std::int64_t p, std::int64_t q, std::int64_t r) {
    return face_sign_ * kernel_.orient2d(p, q, r, first_axis_, second_axis_);
  }

  int add_point(std::int64_t id) {
    for (std::size_t index = 0; index < points_.size(); ++index) {
      if (kernel_.equal(points_[index], id)) {
        return static_cast<int>(index);
      }
    }
    points_.push_back(id);
    return static_cast<int>(points_.size() - 1);
  }

  std::int64_t edge_plane_point(std::int64_t a, std::int64_t b, const PlaneDef& plane) {
    PointDef def;
    line_planes(a, b, def.planes.data());
    def.planes[2] = plane;
    return kernel_.intern(def, kEdgePlane, face_);
  }

  void add_partner(std::int64_t t, const PlaneDef& plane, std::int64_t u,
                   NativeVector<std::int64_t>& coplanar) {
    const std::int64_t* tv = triangles_ + 3 * t;
    const std::int64_t* uv = triangles_ + 3 * u;
    const PlaneDef other = triangle_plane(u);
    int r[3];
    int s[3];
    for (int k = 0; k < 3; ++k) {
      r[k] = kernel_.side(other, tv[k]);
    }
    if ((r[0] > 0 && r[1] > 0 && r[2] > 0) || (r[0] < 0 && r[1] < 0 && r[2] < 0)) {
      return;
    }
    for (int k = 0; k < 3; ++k) {
      s[k] = kernel_.side(plane, uv[k]);
    }
    if ((s[0] > 0 && s[1] > 0 && s[2] > 0) || (s[0] < 0 && s[1] < 0 && s[2] < 0)) {
      return;
    }
    if (t < u) {
      ++output_.contact_pairs;
    }
    if (s[0] == 0 && s[1] == 0 && s[2] == 0) {
      if (t < u) {
        ++output_.coplanar_pairs;
      }
      coplanar.push_back(u);
      const int local[3] = {add_point(uv[0]), add_point(uv[1]), add_point(uv[2])};
      for (int k = 0; k < 3; ++k) {
        segments_.push_back(
            {local[k], local[(k + 1) % 3], edge_line(uv[k], uv[(k + 1) % 3]), true});
      }
      return;
    }
    std::int64_t ends[3];
    int count = 0;
    int vertices = 0;
    for (int k = 0; k < 3; ++k) {
      if (s[k] == 0) {
        ends[count++] = uv[k];
        ++vertices;
      }
    }
    for (int k = 0; k < 3; ++k) {
      const int next = (k + 1) % 3;
      if (s[k] * s[next] < 0) {
        ends[count++] = edge_plane_point(uv[k], uv[next], plane);
      }
    }
    if (count == 1) {
      add_point(ends[0]);
      return;
    }
    const int a = add_point(ends[0]);
    const int b = add_point(ends[1]);
    if (a == b) {
      return;
    }
    const Line line = vertices == 2 ? edge_line(ends[0], ends[1]) : Line{1, -1, -1, u};
    segments_.push_back({a, b, line, true});
  }

  bool box_disjoint(std::int64_t p, std::int64_t a, std::int64_t b) const {
    const PoolPoint& x = kernel_.point(p);
    const PoolPoint& first = kernel_.point(a);
    const PoolPoint& second = kernel_.point(b);
    for (int axis = 0; axis < 3; ++axis) {
      const double low = std::min(first.lower(axis), second.lower(axis));
      const double high = std::max(first.upper(axis), second.upper(axis));
      if (x.upper(axis) < low || x.lower(axis) > high) {
        return true;
      }
    }
    return false;
  }

  bool segment_boxes_disjoint(const Segment& first, const Segment& second) const {
    const std::int64_t ids[4] = {points_[static_cast<std::size_t>(first.a)],
                                 points_[static_cast<std::size_t>(first.b)],
                                 points_[static_cast<std::size_t>(second.a)],
                                 points_[static_cast<std::size_t>(second.b)]};
    for (int axis = 0; axis < 3; ++axis) {
      double low[2];
      double high[2];
      for (int s = 0; s < 2; ++s) {
        const PoolPoint& p = kernel_.point(ids[2 * s]);
        const PoolPoint& q = kernel_.point(ids[2 * s + 1]);
        low[s] = std::min(p.lower(axis), q.lower(axis));
        high[s] = std::max(p.upper(axis), q.upper(axis));
      }
      if (high[0] < low[1] || high[1] < low[0]) {
        return true;
      }
    }
    return false;
  }

  // Whether local point p lies in the relative interior of the segment.
  bool on_open(int p, const Segment& segment) {
    if (p == segment.a || p == segment.b) {
      return false;
    }
    const std::int64_t a = points_[static_cast<std::size_t>(segment.a)];
    const std::int64_t b = points_[static_cast<std::size_t>(segment.b)];
    const std::int64_t x = points_[static_cast<std::size_t>(p)];
    if (box_disjoint(x, a, b) || orient(segment.a, segment.b, p) != 0) {
      return false;
    }
    const int axis = kernel_.compare(a, b, first_axis_) != 0 ? first_axis_ : second_axis_;
    const int before = kernel_.compare(x, a, axis);
    return before != 0 && before == kernel_.compare(b, x, axis);
  }

  std::int64_t crossing(const Line& first, const Line& second) {
    PointDef def;
    std::int8_t construction = kTriplePlane;
    if (first.kind == 0 || second.kind == 0) {
      const Line& line = first.kind == 0 ? first : second;
      const Line& other = first.kind == 0 ? second : first;
      line_planes(line.a, line.b, def.planes.data());
      if (other.kind == 0) {
        def.planes[2] = extruded(other.a, other.b, axis_);
        construction = kEdgeEdge;
      } else {
        def.planes[2] = triangle_plane(other.face);
        construction = kEdgePlane;
      }
    } else {
      def.planes = {triangle_plane(face_), triangle_plane(first.face),
                    triangle_plane(second.face)};
    }
    return kernel_.intern(def, construction, face_);
  }

  void split_at_points() {
    for (std::size_t s = 0; s < segments_.size(); ++s) {
      for (std::size_t p = 0; p < points_.size(); ++p) {
        kernel_.work();
        if (on_open(static_cast<int>(p), segments_[s])) {
          Segment tail = segments_[s];
          tail.a = static_cast<int>(p);
          segments_[s].b = static_cast<int>(p);
          segments_.push_back(tail);
          p = static_cast<std::size_t>(-1);
        }
      }
    }
    for (Segment& segment : segments_) {
      if (segment.a > segment.b) {
        std::swap(segment.a, segment.b);
      }
    }
    std::sort(segments_.begin(), segments_.end(),
              [this](const Segment& left, const Segment& right) {
                kernel_.work();
                return std::pair(left.a, left.b) < std::pair(right.a, right.b);
              });
    NativeVector<Segment> unique;
    for (const Segment& segment : segments_) {
      if (!unique.empty() && unique.back().a == segment.a && unique.back().b == segment.b) {
        unique.back().contact = unique.back().contact || segment.contact;
      } else {
        unique.push_back(segment);
      }
    }
    segments_ = std::move(unique);
  }

  // Splits constraints at contained points and proper crossings until the
  // constraint graph is embedded.  Each round adds at least one new point, and
  // there are finitely many crossings of the finitely many supporting lines.
  void embed() {
    while (true) {
      split_at_points();
      const std::size_t before = points_.size();
      for (std::size_t i = 0; i < segments_.size(); ++i) {
        for (std::size_t j = i + 1; j < segments_.size(); ++j) {
          kernel_.work();
          const Segment first = segments_[i];
          const Segment second = segments_[j];
          if (first.a == second.a || first.a == second.b || first.b == second.a ||
              first.b == second.b || segment_boxes_disjoint(first, second)) {
            continue;
          }
          if (orient(first.a, first.b, second.a) * orient(first.a, first.b, second.b) >= 0 ||
              orient(second.a, second.b, first.a) * orient(second.a, second.b, first.b) >= 0) {
            continue;
          }
          add_point(crossing(first.line, second.line));
        }
      }
      if (points_.size() == before) {
        return;
      }
    }
  }

  // Keeps the points in the closed triangle and the constraints between them.
  void clip() {
    NativeVector<int> renumber(points_.size(), -1);
    NativeVector<std::int64_t> kept;
    for (std::size_t p = 0; p < points_.size(); ++p) {
      const int local = static_cast<int>(p);
      if (p < 3 || (orient(0, 1, local) >= 0 && orient(1, 2, local) >= 0 &&
                    orient(2, 0, local) >= 0)) {
        renumber[p] = static_cast<int>(kept.size());
        kept.push_back(points_[p]);
      }
    }
    NativeVector<Segment> segments;
    for (Segment segment : segments_) {
      const int a = renumber[static_cast<std::size_t>(segment.a)];
      const int b = renumber[static_cast<std::size_t>(segment.b)];
      if (a >= 0 && b >= 0) {
        segment.a = a;
        segment.b = b;
        segments.push_back(segment);
      }
    }
    points_ = std::move(kept);
    segments_ = std::move(segments);
  }

  void insert_point(LocalMesh& mesh, int p) {
    for (int index = 0; index < mesh.count(); ++index) {
      kernel_.work();
      if (!mesh.alive(index)) {
        continue;
      }
      const std::array<int, 3> t = mesh.triangle(index);
      int sides[3];
      bool outside = false;
      for (int k = 0; k < 3 && !outside; ++k) {
        sides[k] = orient(t[k], t[(k + 1) % 3], p);
        outside = sides[k] < 0;
      }
      if (outside) {
        continue;
      }
      const int zeros = (sides[0] == 0) + (sides[1] == 0) + (sides[2] == 0);
      if (zeros == 0) {
        mesh.remove(index);
        mesh.add(t[0], t[1], p);
        mesh.add(t[1], t[2], p);
        mesh.add(t[2], t[0], p);
        return;
      }
      if (zeros == 1) {
        const int k = sides[0] == 0 ? 0 : (sides[1] == 0 ? 1 : 2);
        const int u = t[k];
        const int v = t[(k + 1) % 3];
        const int w = t[(k + 2) % 3];
        const int neighbor = mesh.find(v, u);
        mesh.remove(index);
        mesh.add(u, p, w);
        mesh.add(p, v, w);
        if (neighbor >= 0) {
          const int x = LocalMesh::opposite(mesh.triangle(neighbor), u, v);
          mesh.remove(neighbor);
          mesh.add(v, p, x);
          mesh.add(p, u, x);
        }
        return;
      }
      break;
    }
    throw Refusal{PHX_MC_INTERNAL_ERROR, face_};
  }

  bool crosses(int a, int b, int u, int v) {
    if (u == a || u == b || v == a || v == b) {
      return false;
    }
    return orient(a, b, u) * orient(a, b, v) < 0 && orient(u, v, a) * orient(u, v, b) < 0;
  }

  // Sloan's constraint recovery: flip crossing edges of strictly convex
  // quadrilaterals until the segment is an edge; terminates for any
  // triangulation (S. W. Sloan, Adv. Eng. Software 1993).
  void insert_segment(LocalMesh& mesh, int a, int b) {
    if (mesh.find(a, b) >= 0 || mesh.find(b, a) >= 0) {
      return;
    }
    NativeDeque<std::pair<int, int>> crossing;
    for (int index = 0; index < mesh.count(); ++index) {
      kernel_.work();
      if (!mesh.alive(index)) {
        continue;
      }
      const std::array<int, 3> t = mesh.triangle(index);
      for (int k = 0; k < 3; ++k) {
        const int u = t[k];
        const int v = t[(k + 1) % 3];
        if (u < v && crosses(a, b, u, v)) {
          crossing.emplace_back(u, v);
        }
      }
    }
    const std::size_t initial = crossing.size();
    std::size_t budget = 16 * (initial + 1) * (initial + 1) + 1024;
    while (!crossing.empty()) {
      kernel_.work();
      if (budget-- == 0) {
        throw Refusal{PHX_MC_INTERNAL_ERROR, face_};
      }
      const auto [u, v] = crossing.front();
      crossing.pop_front();
      const int first = mesh.find(u, v);
      const int second = mesh.find(v, u);
      if (first < 0 || second < 0) {
        throw Refusal{PHX_MC_INTERNAL_ERROR, face_};
      }
      const int w1 = LocalMesh::opposite(mesh.triangle(first), u, v);
      const int w2 = LocalMesh::opposite(mesh.triangle(second), u, v);
      if (orient(w1, w2, u) * orient(w1, w2, v) >= 0) {
        crossing.emplace_back(u, v);
        continue;
      }
      mesh.remove(first);
      mesh.remove(second);
      mesh.add(w1, u, w2);
      mesh.add(w2, v, w1);
      if (crosses(a, b, w1, w2)) {
        crossing.emplace_back(std::min(w1, w2), std::max(w1, w2));
      }
    }
    if (mesh.find(a, b) < 0 && mesh.find(b, a) < 0) {
      throw Refusal{PHX_MC_INTERNAL_ERROR, face_};
    }
  }

  bool published_orientation(const std::array<int, 3>& triangle) {
    return face_sign_ * kernel_.rounded_orient2d(
                            points_[static_cast<std::size_t>(triangle[0])],
                            points_[static_cast<std::size_t>(triangle[1])],
                            points_[static_cast<std::size_t>(triangle[2])],
                            first_axis_, second_axis_) > 0;
  }

  bool publication_flip(LocalMesh& mesh, int u, int v) {
    const auto key = std::pair(std::min(u, v), std::max(u, v));
    // embed() orders constraints; clip() preserves that order under its
    // monotone renumbering. Only triangulation diagonals may be changed.
    const auto constraint = std::lower_bound(
        segments_.begin(), segments_.end(), key,
        [](const Segment& segment, const std::pair<int, int>& edge) {
          return std::pair(segment.a, segment.b) < edge;
        });
    if (constraint != segments_.end() &&
        std::pair(constraint->a, constraint->b) == key) {
      return false;
    }
    const int first = mesh.find(u, v);
    const int second = mesh.find(v, u);
    if (first < 0 || second < 0) {
      return false;
    }
    const int a = LocalMesh::opposite(mesh.triangle(first), u, v);
    const int b = LocalMesh::opposite(mesh.triangle(second), u, v);
    const std::array<int, 3> left{a, u, b};
    const std::array<int, 3> right{b, v, a};
    if (orient(a, u, b) <= 0 || orient(b, v, a) <= 0 ||
        !published_orientation(left) || !published_orientation(right)) {
      return false;
    }
    mesh.remove(first);
    mesh.remove(second);
    mesh.add(a, u, b);
    mesh.add(b, v, a);
    return true;
  }

  // Almost coplanar original faces can make a contact polyline kink by less
  // than one ULP. An unnecessary chord then creates a rounded zero-area ear.
  // Choose another EXACTLY legal cell triangulation instead of perturbing
  // geometry or erasing the original contact kink. Every accepted flip turns
  // a bad carrier triangle into two good ones, so bad triangles strictly
  // decrease and this process terminates without an arbitrary retry budget.
  void triangulate_publication(LocalMesh& mesh) {
    bool changed;
    do {
      changed = false;
      for (int index = 0; index < mesh.count(); ++index) {
        if (!mesh.alive(index) || published_orientation(mesh.triangle(index))) {
          continue;
        }
        const auto triangle = mesh.triangle(index);
        for (int corner = 0; corner < 3; ++corner) {
          if (publication_flip(mesh, triangle[corner], triangle[(corner + 1) % 3])) {
            changed = true;
            break;
          }
        }
      }
    } while (changed);
  }

  int feature(int p) {
    if (p < 3) {
      return p;
    }
    for (int k = 0; k < 3; ++k) {
      if (orient(k, (k + 1) % 3, p) == 0) {
        return 3 + (k + 2) % 3;  // edge opposite vertex (k + 2) % 3
      }
    }
    return 6;
  }

  void triangulate(std::int64_t t, const NativeVector<std::int64_t>& coplanar) {
    NativeVector<std::array<int, 3>> triangles;
    bool split = points_.size() > 3;
    for (const Segment& segment : segments_) {
      // Constraints other than the three boundary edges split the face.
      split = split || !((segment.a == 0 && segment.b == 1) || (segment.a == 1 && segment.b == 2) ||
                         (segment.a == 0 && segment.b == 2));
    }
    if (split) {
      ++output_.split_faces;
      LocalMesh mesh;
      mesh.add(0, 1, 2);
      for (int p = 3; p < static_cast<int>(points_.size()); ++p) {
        insert_point(mesh, p);
      }
      for (const Segment& segment : segments_) {
        insert_segment(mesh, segment.a, segment.b);
      }
      // Coverage consumes only exact topology, never rounded carrier quality.
      if (!coverage_only_) triangulate_publication(mesh);
      for (int index = 0; index < mesh.count(); ++index) {
        kernel_.work();
        if (mesh.alive(index)) {
          triangles.push_back(mesh.triangle(index));
        }
      }
    } else {
      triangles.push_back({0, 1, 2});
    }
    if (static_cast<std::int64_t>(output_.fragments.size() + triangles.size()) > max_fragments_) {
      throw Refusal{PHX_MC_CAPACITY_EXCEEDED, t};
    }
    // Coplanar coverage: a fragment crosses no edge of a coplanar partner, so
    // it lies in the partner exactly when its three corners do.
    NativeVector<NativeVector<char>> inside(coplanar.size());
    NativeVector<int> orientation(coplanar.size());
    for (std::size_t c = 0; c < coplanar.size(); ++c) {
      const std::int64_t* uv = triangles_ + 3 * coplanar[c];
      orientation[c] = orient_pool(uv[0], uv[1], uv[2]);
      inside[c].resize(points_.size());
      for (std::size_t p = 0; p < points_.size(); ++p) {
        bool in = true;
        for (int k = 0; k < 3 && in; ++k) {
          in = orientation[c] * orient_pool(uv[k], uv[(k + 1) % 3], points_[p]) >= 0;
        }
        inside[c][p] = in ? 1 : 0;
      }
    }
    for (const auto& triangle : triangles) {
      const std::int64_t fragment = static_cast<std::int64_t>(output_.fragments.size());
      output_.fragments.push_back({points_[static_cast<std::size_t>(triangle[0])],
                                   points_[static_cast<std::size_t>(triangle[1])],
                                   points_[static_cast<std::size_t>(triangle[2])]});
      output_.fragment_faces.push_back(t);
      output_.fragment_frames.push_back(static_cast<std::int8_t>(face_sign_ * (axis_ + 1)));
      for (std::size_t c = 0; c < coplanar.size(); ++c) {
        if (inside[c][static_cast<std::size_t>(triangle[0])] &&
            inside[c][static_cast<std::size_t>(triangle[1])] &&
            inside[c][static_cast<std::size_t>(triangle[2])]) {
          output_.coplanar.push_back({fragment, coplanar[c], orientation[c]});
        }
      }
    }
    for (const Segment& segment : segments_) {
      if (segment.contact) {
        output_.contact_edges.push_back({points_[static_cast<std::size_t>(segment.a)],
                                         points_[static_cast<std::size_t>(segment.b)]});
      }
    }
    for (int p = 0; p < static_cast<int>(points_.size()); ++p) {
      output_.locations.push_back({points_[static_cast<std::size_t>(p)], t, feature(p)});
    }
  }

  const double* coords_;
  const std::int64_t* triangles_;
  Kernel kernel_;
  std::int64_t max_fragments_;
  bool coverage_only_;
  Output output_;
  std::int64_t face_ = -1;
  int axis_ = 0;
  int face_sign_ = 1;
  int first_axis_ = 1;
  int second_axis_ = 2;
  NativeVector<std::int64_t> points_;
  NativeVector<Segment> segments_;
};

// ------------------------------------------------------------ welding


// Exact clipping of a canonical physical reference triangle by three original
// rational halfplanes. Positive denominator clearing supplies signed integers;
// this reuses the arrangement's arbitrary-integer predicates, not binary64
// representatives. The output retains original supporting-plane provenance.
struct ExactReferenceVertex {
  Dyadic x;
  Dyadic y;
  Dyadic w;
  std::array<std::int32_t, 2> support{};
};

struct ExactReferenceEntry {
  std::int32_t vertex = 0;
  std::int32_t incoming = 0;
};


using ExactReferencePlanes = std::array<std::array<Dyadic, 3>, 6>;

struct ExactReferenceClip {
  std::array<ExactReferenceVertex, 16> vertices;
  // A clipped convex polygon can repeat a point only when a crossing is an
  // already-retained zero-side endpoint. Preserve that exact construction
  // identity, rather than repeatedly comparing homogeneous cross products.
  std::array<std::int32_t, 16> identities{0, 1, 2};
  std::array<ExactReferenceEntry, 12> loop{};
  std::array<ExactReferenceEntry, 12> next{};
  std::int32_t used = 3;
  std::int32_t count = 3;
  std::int32_t next_count = 0;
  std::int64_t work = 0;

  ExactReferenceClip() {
    vertices[0] = {Dyadic(0.0), Dyadic(0.0), Dyadic(1.0), {0, 1}};
    vertices[1] = {Dyadic(1.0), Dyadic(0.0), Dyadic(1.0), {1, 2}};
    vertices[2] = {Dyadic(0.0), Dyadic(1.0), Dyadic(1.0), {2, 0}};
    loop[0] = {0, 0}; loop[1] = {1, 1}; loop[2] = {2, 2};
  }
};

int32_t validate_reference_integers(const std::int64_t* offsets,
                                   const std::int8_t* signs, std::int64_t word_count,
                                   std::int64_t& maximum_words) {
  for (int scalar = 0; scalar < 18; ++scalar) {
    if (offsets[scalar] < 0 || offsets[scalar + 1] < offsets[scalar] ||
        offsets[scalar + 1] > word_count || signs[scalar] < -1 || signs[scalar] > 1 ||
        ((offsets[scalar + 1] == offsets[scalar]) != (signs[scalar] == 0))) {
      return PHX_MC_INVALID_INPUT;
    }
    maximum_words = std::max(maximum_words, offsets[scalar + 1] - offsets[scalar]);
  }
  return PHX_MC_OK;
}

int32_t decode_reference_planes(const std::uint32_t* words, const std::int64_t* offsets,
                                const std::int8_t* signs, ExactReferencePlanes& planes) {
  for (int scalar = 0; scalar < 18; ++scalar) {
    planes[scalar / 3][scalar % 3] =
        Dyadic::integer(words + offsets[scalar],
                        static_cast<std::size_t>(offsets[scalar + 1] - offsets[scalar]),
                        signs[scalar]);
  }
  if (planes[0][0].sign() >= 0 || planes[0][1].sign() != 0 || planes[0][2].sign() != 0 ||
      planes[1][0].sign() != 0 || planes[1][1].sign() >= 0 || planes[1][2].sign() != 0 ||
      planes[2][0].sign() <= 0 || (planes[2][1] - planes[2][0]).sign() != 0 ||
      (planes[2][2] - planes[2][0]).sign() != 0) {
    return PHX_MC_INVALID_INPUT;
  }
  return PHX_MC_OK;
}

bool reserve_reference_work(ExactReferenceClip& clip, std::int64_t amount,
                            std::int64_t limit) {
  if (amount > limit - clip.work) return false;
  clip.work += amount;
  return true;
}

int32_t append_reference_entry(ExactReferenceClip& clip, std::int32_t& next_count,
                               ExactReferenceEntry entry, std::int64_t work_limit) {
  if (next_count) {
    if (!reserve_reference_work(clip, 1, work_limit)) return PHX_MC_CAPACITY_EXCEEDED;
    if (clip.identities[clip.next[next_count - 1].vertex] ==
        clip.identities[entry.vertex]) {
      return PHX_MC_OK;
    }
  }
  if (next_count >= static_cast<int>(clip.next.size())) {
    return PHX_MC_INTERNAL_ERROR;
  }
  clip.next[next_count++] = entry;
  return PHX_MC_OK;
}

int32_t construct_reference_intersection(ExactReferenceClip& clip,
                                         const ExactReferencePlanes& planes,
                                         int incoming, int clipping) {
  if (clip.used >= static_cast<int>(clip.vertices.size())) {
    return PHX_MC_INTERNAL_ERROR;
  }
  const auto& first = planes[incoming];
  const auto& second = planes[clipping];
  Dyadic determinant = first[0] * second[1] - first[1] * second[0];
  if (determinant.sign() == 0) {
    return PHX_MC_INTERNAL_ERROR;
  }
  Dyadic x = first[2] * second[1] - first[1] * second[2];
  Dyadic y = first[0] * second[2] - first[2] * second[0];
  if (determinant.sign() < 0) {
    determinant.negate(); x.negate(); y.negate();
  }
  clip.vertices[clip.used] = {std::move(x), std::move(y), std::move(determinant),
                             {incoming, clipping}};
  return PHX_MC_OK;
}

int32_t clip_reference_plane(ExactReferenceClip& clip,
                             const ExactReferencePlanes& planes, int clipping,
                             std::int64_t work_limit) {
  std::array<int, 12> sides{};
  bool all_inside = true;
  bool all_outside = true;
  for (int slot = 0; slot < clip.count; ++slot) {
    const int vertex = clip.loop[slot].vertex;
    const auto& point = clip.vertices[vertex];
    if (vertex < 3) {
      // These are the authored unit corners: evaluate the affine line there
      // directly, exactly as the containment proof did, without multiplications
      // by zero and one. The result is also the clipping predicate.
      if (vertex == 0) {
        if (!reserve_reference_work(clip, 1, work_limit)) return PHX_MC_CAPACITY_EXCEEDED;
        sides[slot] = planes[clipping][2].sign();
      } else {
        if (!reserve_reference_work(clip, 2, work_limit)) return PHX_MC_CAPACITY_EXCEEDED;
        sides[slot] =
            (planes[clipping][2] - planes[clipping][vertex - 1]).sign();
      }
    } else if (point.support[0] == clipping || point.support[1] == clipping) {
      sides[slot] = 0;
    } else {
      if (!reserve_reference_work(clip, 7, work_limit)) return PHX_MC_CAPACITY_EXCEEDED;
      sides[slot] = (planes[clipping][2] * point.w -
                    planes[clipping][0] * point.x -
                    planes[clipping][1] * point.y).sign();
    }
    all_inside = all_inside && sides[slot] >= 0;
    all_outside = all_outside && sides[slot] < 0;
  }
  if (all_inside) {
    // The ordinary retained-vertex loop rotates once; preserve its exact
    // publication order and supporting-plane provenance.
    for (int slot = 0; slot < clip.count; ++slot) {
      clip.next[slot] = clip.loop[(slot + 1) % clip.count];
    }
    clip.next_count = clip.count;
    return PHX_MC_OK;
  }
  if (all_outside) {
    clip.next_count = 0;
    return PHX_MC_OK;
  }
  std::int32_t& next_count = clip.next_count;
  next_count = 0;
  for (int slot = 0; slot < clip.count; ++slot) {
    const int following = (slot + 1) % clip.count;
    const auto& b = clip.loop[following];
    const bool a_inside = sides[slot] >= 0;
    const bool b_inside = sides[following] >= 0;
    if (a_inside != b_inside) {
      if (!reserve_reference_work(clip, 12, work_limit)) return PHX_MC_CAPACITY_EXCEEDED;
      int32_t status = construct_reference_intersection(clip, planes, b.incoming, clipping);
      if (status != PHX_MC_OK) return status;
      const int endpoint = sides[slot] == 0 ? clip.loop[slot].vertex
                           : sides[following] == 0 ? b.vertex : clip.used;
      clip.identities[clip.used] =
          endpoint == clip.used ? clip.used : clip.identities[endpoint];
      status = append_reference_entry(
          clip, next_count, {clip.used++, a_inside ? b.incoming : clipping}, work_limit);
      if (status != PHX_MC_OK) return status;
    }
    if (b_inside) {
      const int32_t status = append_reference_entry(clip, next_count, b, work_limit);
      if (status != PHX_MC_OK) return status;
    }
  }
  if (next_count > 1) {
    if (!reserve_reference_work(clip, 1, work_limit)) return PHX_MC_CAPACITY_EXCEEDED;
    if (clip.identities[clip.next[0].vertex] ==
        clip.identities[clip.next[next_count - 1].vertex]) {
      clip.next[0].incoming = clip.next[next_count - 1].incoming;
      --next_count;
    }
  }
  return PHX_MC_OK;
}


int32_t clip_reference_triangle_exact(
    const std::uint32_t* words, std::int64_t word_count,
    const std::int64_t* offsets, const std::int8_t* signs,
    std::int64_t scratch_limit, std::int64_t work_limit, std::int32_t capacity,
    std::int32_t* supporting_planes, std::int32_t* edge_labels,
    std::int32_t* vertex_count, std::int64_t* work_units) {
  if (words == nullptr || offsets == nullptr || signs == nullptr ||
      supporting_planes == nullptr || edge_labels == nullptr ||
      vertex_count == nullptr || work_units == nullptr || word_count < 0 ||
      offsets[0] != 0 || offsets[18] != word_count || capacity < 6 ||
      scratch_limit <= 0 || work_limit <= 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  *vertex_count = 0;
  *work_units = 0;
  std::int64_t maximum_words = 1;
  int32_t status = validate_reference_integers(offsets, signs, word_count, maximum_words);
  if (status != PHX_MC_OK) return status;
  // Original integers have at most M limbs. Vertex cofactors have 2M and
  // every side/equality predicate at most 4M. Eighteen coefficients, sixteen
  // cached homogeneous vertices and all simultaneous arithmetic temporaries
  // fit below this conservative resident bound before any limb allocation.
  if (scratch_limit < 16384 || maximum_words > (scratch_limit - 16384) / 4096) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  MemoryBudgetWindow memory(static_cast<std::size_t>(scratch_limit));
  MemoryScope memory_scope(memory.owner());
  ExactReferencePlanes planes;
  status = decode_reference_planes(words, offsets, signs, planes);
  if (status != PHX_MC_OK) return status;
  ExactReferenceClip clip;
  for (int clipping = 3; clipping < 6 && clip.count >= 3; ++clipping) {
    status = clip_reference_plane(clip, planes, clipping, work_limit);
    if (status != PHX_MC_OK) {
      *work_units = clip.work;
      return status;
    }
    clip.loop.swap(clip.next);
    clip.count = clip.next_count;
  }
  const auto& vertices = clip.vertices;
  const auto& loop = clip.loop;
  const std::int32_t count = clip.count;
  const std::int64_t work = clip.work;
  *work_units = work;
  if (count < 3) {
    return PHX_MC_OK;
  }
  if (count > capacity) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  for (int slot = 0; slot < count; ++slot) {
    const auto& vertex = vertices[loop[slot].vertex];
    supporting_planes[2 * slot] = vertex.support[0];
    supporting_planes[2 * slot + 1] = vertex.support[1];
    edge_labels[slot] = loop[(slot + 1) % count].incoming;
  }
  *vertex_count = count;
  return PHX_MC_OK;
}

#include "reference_tetrahedron_clip.inc"

// Outcome of the exact predicate facet coverage proofs.
enum class PredicateCover : std::uint8_t { kCovered, kRefused, kUnknown };

// Proves the arrangement's coverage result without building it, with exact
// orientation predicates only (one work unit each; one unit per compared
// edge pair):
//  * single row: the subject lies in one closed coplanar row and every other
//    coplanar row is weakly separated from it by one of the six edge lines;
//  * partition: the coplanar rows not weakly separated from the subject (the
//    positive-area contributors) are pairwise weakly separated, consistently
//    oriented, and no edge of their union that is not shared exactly (with
//    reversed endpoints) by two contributors meets the open subject. The open
//    subject is connected, misses the union's boundary and meets a
//    contributor's interior, so it lies inside the union; each fragment then
//    has exactly one owner.
// The same predicates decide plane membership, the projection and degenerate
// partners as the arrangement. A configuration neither proof decides (an
// overlap, a T-junction or boundary edge across the subject, a gap) is left to
// the arrangement.
PredicateCover predicate_facet_cover(const double* coordinates,
                                     std::span<const std::int64_t> triangles,
                                     std::size_t rows, NativeVector<std::size_t>& covering) {
  covering.clear();
  const auto corner = [&](std::size_t face, int k) {
    return coordinates + 3 * static_cast<std::size_t>(triangles[3 * face + k]);
  };
  const auto predicate = [] { native_execution_charge(1); };
  // Projection: the first axis whose subject normal component is nonzero, as
  // the arrangement's exact normal sign of face 0 selects it.
  int axis = -1;
  using Shape = std::array<std::array<double, 2>, 3>;
  const auto project = [&](std::size_t face) {
    Shape shape;
    for (int k = 0; k < 3; ++k) {
      shape[k][0] = corner(face, k)[(axis + 1) % 3];
      shape[k][1] = corner(face, k)[(axis + 2) % 3];
    }
    return shape;
  };
  Shape subject{};
  int subject_orientation = 0;
  for (int a = 0; a < 3 && subject_orientation == 0; ++a) {
    axis = a;
    subject = project(0);
    predicate();
    subject_orientation = orient2d(subject[0].data(), subject[1].data(), subject[2].data());
  }
  if (subject_orientation == 0) return PredicateCover::kRefused;
  // Coplanar partners in row order; a degenerate partner refuses the face.
  NativeVector<std::size_t> partners;
  NativeVector<int> orientations;
  NativeVector<Shape> shapes;
  partners.reserve(rows);
  orientations.reserve(rows);
  shapes.reserve(rows);
  for (std::size_t row = 1; row <= rows; ++row) {
    bool coplanar = true;
    for (int k = 0; k < 3 && coplanar; ++k) {
      predicate();
      coplanar = orient3d(corner(0, 0), corner(0, 1), corner(0, 2), corner(row, k)) == 0;
    }
    if (!coplanar) continue;
    const Shape shape = project(row);
    predicate();
    const int orientation = orient2d(shape[0].data(), shape[1].data(), shape[2].data());
    if (orientation == 0) return PredicateCover::kRefused;
    partners.push_back(row);
    orientations.push_back(orientation);
    shapes.push_back(shape);
  }
  if (partners.empty()) return PredicateCover::kRefused;
  // Whether every vertex of `second` lies weakly outside one edge line of
  // `first` (positively oriented when first_orientation > 0).
  const auto outside_edge = [&](const Shape& first, int first_orientation, const Shape& second) {
    for (int k = 0; k < 3; ++k) {
      bool outside = true;
      for (int v = 0; v < 3 && outside; ++v) {
        predicate();
        outside = first_orientation * orient2d(first[(k + 1) % 3].data(),
                                               first[(k + 2) % 3].data(), second[v].data()) <= 0;
      }
      if (outside) return true;
    }
    return false;
  };
  // Separation from the subject, decided once per partner: -1 unknown.
  NativeVector<std::int8_t> separated(partners.size(), -1);
  const auto separated_from_subject = [&](std::size_t p) {
    if (separated[p] < 0) {
      separated[p] = outside_edge(subject, subject_orientation, shapes[p]) ||
                             outside_edge(shapes[p], orientations[p], subject)
                         ? 1
                         : 0;
    }
    return separated[p] == 1;
  };
  // Single row: closed containment of every subject vertex in one partner.
  std::size_t found = partners.size();
  for (std::size_t p = 0; p < partners.size() && found == partners.size(); ++p) {
    bool inside = true;
    for (int v = 0; v < 3 && inside; ++v) {
      for (int k = 0; k < 3 && inside; ++k) {
        predicate();
        inside = orientations[p] * orient2d(shapes[p][(k + 1) % 3].data(),
                                            shapes[p][(k + 2) % 3].data(),
                                            subject[v].data()) >= 0;
      }
    }
    if (inside) found = p;
  }
  if (found != partners.size()) {
    bool alone = true;
    for (std::size_t p = 0; p < partners.size() && alone; ++p) {
      alone = p == found || separated_from_subject(p);
    }
    if (alone) {
      covering.assign(1, partners[found] - 1);
      return PredicateCover::kCovered;
    }
  }
  // Partition: positive-area contributors in row order.
  NativeVector<std::size_t> contributors;
  for (std::size_t p = 0; p < partners.size(); ++p) {
    if (!separated_from_subject(p)) contributors.push_back(p);
  }
  if (contributors.empty()) return PredicateCover::kUnknown;
  for (std::size_t i = 0; i < contributors.size(); ++i) {
    const std::size_t a = contributors[i];
    if (orientations[a] != orientations[contributors.front()]) return PredicateCover::kUnknown;
    for (std::size_t j = i + 1; j < contributors.size(); ++j) {
      const std::size_t b = contributors[j];
      if (!outside_edge(shapes[a], orientations[a], shapes[b]) &&
          !outside_edge(shapes[b], orientations[b], shapes[a])) {
        return PredicateCover::kUnknown;
      }
    }
  }
  const auto same_point = [](const double* x, const double* y) {
    return x[0] == y[0] && x[1] == y[1] && x[2] == y[2];
  };
  for (std::size_t a : contributors) {
    const std::size_t row = partners[a];
    for (int k = 0; k < 3; ++k) {
      const double* from = corner(row, (k + 1) % 3);
      const double* to = corner(row, (k + 2) % 3);
      bool shared = false;
      for (std::size_t b : contributors) {
        if (b == a) continue;
        const std::size_t other = partners[b];
        for (int j = 0; j < 3 && !shared; ++j) {
          native_execution_charge(1);
          shared = same_point(corner(other, (j + 1) % 3), to) &&
                   same_point(corner(other, (j + 2) % 3), from);
        }
        if (shared) break;
      }
      if (shared) continue;
      // The closed edge misses the open subject iff its line weakly
      // separates the subject or a subject edge line has it weakly outside.
      const double* u = shapes[a][(k + 1) % 3].data();
      const double* w = shapes[a][(k + 2) % 3].data();
      int positive = 0;
      int negative = 0;
      for (int v = 0; v < 3; ++v) {
        predicate();
        const int side = orient2d(u, w, subject[v].data());
        positive += side > 0;
        negative += side < 0;
      }
      if (positive == 0 || negative == 0) continue;
      bool clear = false;
      for (int e = 0; e < 3 && !clear; ++e) {
        predicate();
        clear = subject_orientation *
                    orient2d(subject[(e + 1) % 3].data(), subject[(e + 2) % 3].data(), u) <= 0;
        if (clear) {
          predicate();
          clear = subject_orientation *
                      orient2d(subject[(e + 1) % 3].data(), subject[(e + 2) % 3].data(), w) <= 0;
        }
      }
      if (!clear) return PredicateCover::kUnknown;
    }
  }
  for (std::size_t a : contributors) covering.push_back(partners[a] - 1);
  return PredicateCover::kCovered;
}

// Shared admission, row validation and arrangement proof. predicate_proof
// admits the exact predicate coverage proofs before the arrangement is built.
bool facet_coverage(const double* subject, const double* original_points,
                    std::span<const std::array<int32_t, 4>> original_faces,
                    std::span<const int32_t> rows, NativeVector<int32_t>& contributors,
                    bool predicate_proof) {
  contributors.clear();
  native_execution_charge(1, 1);
  if (rows.empty()) return false;
  NativeVector<int32_t> selected(rows.begin(), rows.end());
  std::sort(selected.begin(), selected.end(), [](int32_t a, int32_t b) {
    native_execution_charge(1);
    return a < b;
  });
  selected.erase(std::unique(selected.begin(), selected.end(), [](int32_t a, int32_t b) {
    native_execution_charge(1);
    return a == b;
  }), selected.end());

  // m supporting edges have at most m(m-1)/2 proper intersections.
  // A planar triangulation has fewer than twice as many triangles as vertices.
  // The local mesh uses int indices, so admit both bounds before allocating.
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 2);
  const auto n = static_cast<std::uint64_t>(selected.size());
  if (n > limit / 3 - 1) throw ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
  const auto m = 3 * (n + 1);
  if (m - 1 > (2 * (limit - m)) / m) {
    throw ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
  }
  const auto points_bound = static_cast<std::int64_t>(m + m * (m - 1) / 2);
  NativeVector<double> coordinates;
  NativeVector<std::int64_t> triangles;
  coordinates.reserve(static_cast<std::size_t>(3 * m));
  triangles.reserve(static_cast<std::size_t>(m));
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(subject[k])) return false;
    coordinates.push_back(subject[k]);
  }
  for (std::int64_t k = 0; k < 3; ++k) triangles.push_back(k);
  for (int32_t row : selected) {
    native_execution_charge(1);
    if (row < 0 || static_cast<std::size_t>(row) >= original_faces.size()) return false;
    const auto& face = original_faces[static_cast<std::size_t>(row)];
    for (int corner = 0; corner < 3; ++corner) {
      if (face[corner] < 0) return false;
      triangles.push_back(static_cast<std::int64_t>(coordinates.size() / 3));
      const double* point = original_points + 3 * static_cast<std::size_t>(face[corner]);
      coordinates.insert(coordinates.end(), point, point + 3);
    }
  }
  if (predicate_proof) {
    NativeVector<std::size_t> covering;
    switch (predicate_facet_cover(coordinates.data(), triangles, selected.size(), covering)) {
      case PredicateCover::kCovered:
        contributors.clear();
        for (std::size_t index : covering) contributors.push_back(selected[index]);
        return true;
      case PredicateCover::kRefused:
        return false;
      case PredicateCover::kUnknown:
        break;
    }
  }
  try {
    Arranger arranger(coordinates.data(), static_cast<std::int64_t>(m), triangles.data(),
                      points_bound, 2 * points_bound, true);
    Kernel& kernel = arranger.kernel();
    const PlaneDef plane{0, 0, 0, 1, 2};
    int axis = -1;
    for (int a = 0; a < 3; ++a) {
      double magnitude = 0.0;
      if (kernel.normal_sign(plane, a, magnitude) != 0) {
        axis = a;
        break;
      }
    }
    if (axis < 0) return false;
    NativeVector<std::int64_t> partners;
    partners.reserve(selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index) {
      native_execution_charge(1);
      const auto face = static_cast<std::int64_t>(index + 1);
      const auto* corners = triangles.data() + 3 * face;
      if (kernel.side(plane, corners[0]) != 0 || kernel.side(plane, corners[1]) != 0 ||
          kernel.side(plane, corners[2]) != 0) continue;
      const int orientation = kernel.orient2d(
          corners[0], corners[1], corners[2], (axis + 1) % 3, (axis + 2) % 3);
      if (orientation == 0) return false;
      partners.push_back(face);
    }
    if (partners.empty()) return false;
    arranger.face(0, partners);
    const Output& out = arranger.output();
    if (out.fragments.empty()) return false;
    NativeVector<int32_t> owners(out.fragments.size(), -1);
    std::int64_t source_orientation = 0;
    for (const auto& record : out.coplanar) {
      native_execution_charge(1);
      if (record[2] == 0 ||
          (source_orientation != 0 && record[2] != source_orientation)) return false;
      source_orientation = record[2];
      auto& owner = owners[static_cast<std::size_t>(record[0])];
      if (owner >= 0) return false;  // positive-area overlap
      owner = selected[static_cast<std::size_t>(record[1] - 1)];
    }
    for (int32_t owner : owners) {
      native_execution_charge(1);
      if (owner < 0) return false;  // positive-area gap
    }
    std::sort(owners.begin(), owners.end(), [](int32_t a, int32_t b) {
      native_execution_charge(1);
      return a < b;
    });
    owners.erase(std::unique(owners.begin(), owners.end(), [](int32_t a, int32_t b) {
      native_execution_charge(1);
      return a == b;
    }), owners.end());
    contributors = std::move(owners);
    return true;
  } catch (const Refusal& refusal) {
    if (refusal.status == PHX_MC_DEGENERATE_INPUT) return false;
    throw ExecutionRefusal{refusal.status};
  }
}

}  // namespace

bool exact_facet_coverage(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, NativeVector<int32_t>& contributors) {
  return facet_coverage(subject, original_points, original_faces, rows, contributors, true);
}

bool exact_facet_coverage_by_arrangement(
    const double* subject, const double* original_points,
    std::span<const std::array<int32_t, 4>> original_faces,
    std::span<const int32_t> rows, NativeVector<int32_t>& contributors) {
  return facet_coverage(subject, original_points, original_faces, rows, contributors, false);
}

int relative_orient3d_exact(const double* a, const double* b, const double* c, const double* d,
                            double relative_floor, double* score) {
  native_execution_primitive_query();
  const double* points[3] = {b, c, d};
  Dyadic columns[3][3], norms[3];
  for (int column = 0; column < 3; ++column) {
    for (int axis = 0; axis < 3; ++axis) {
      native_execution_charge(0);
      columns[column][axis] = Dyadic(points[column][axis]) - Dyadic(a[axis]);
      norms[column] = norms[column] + columns[column][axis] * columns[column][axis];
    }
  }
  const Dyadic determinant = det3(
      columns[0][0], columns[0][1], columns[0][2],
      columns[1][0], columns[1][1], columns[1][2],
      columns[2][0], columns[2][1], columns[2][2]);
  const Dyadic floor(relative_floor);
  const Dyadic numerator = determinant * determinant;
  const Dyadic threshold = floor * floor * norms[0] * norms[1] * norms[2];
  if (determinant.sign() <= 0 || threshold.sign() <= 0) {
    *score = -1.0;
    return -1;
  }
  std::int64_t ne = 0, de = 0;
  const double nf = numerator.fraction(ne), df = threshold.fraction(de);
  *score = std::ldexp(nf / df, static_cast<int>(ne - de));
  return (numerator - threshold).sign();
}

static bool construct_exact_line_point_in_binade(
    const double* origin, const double* endpoint, const double* first,
    const double* second, const double* desired, double* position,
    double* coordinate, bool (*spend)(void*, std::int64_t), void* context) {
  const LineWork work{spend, context};
  std::array<Dyadic, 3> anchor, difference, primitive;
  std::int64_t common = std::numeric_limits<std::int64_t>::max();
  for (int axis = 0; axis < 3; ++axis) {
    work.charge();
    anchor[axis] = Dyadic(origin[axis]);
    difference[axis] = Dyadic(endpoint[axis]) - anchor[axis];
    if (difference[axis].sign()) common = std::min(common, difference[axis].exponent());
  }
  if (common == std::numeric_limits<std::int64_t>::max()) return false;
  std::array<Limbs, 3> integers;
  Limbs divisor;
  for (int axis = 0; axis < 3; ++axis) {
    if (!difference[axis].sign()) continue;
    work.charge(difference[axis].magnitude().size());
    integers[axis] = shift_left(
        difference[axis].magnitude(), difference[axis].exponent() - common);
    divisor = divisor.empty() ? integers[axis] : line_gcd(divisor, integers[axis], work);
  }
  int dominant = 0;
  Limbs largest;
  for (int axis = 0; axis < 3; ++axis) {
    if (integers[axis].empty()) continue;
    Limbs remainder;
    Limbs value = line_divide(integers[axis], divisor, remainder, work);
    if (!remainder.empty()) throw std::invalid_argument("A primitive source direction is not integral.");
    if (compare_magnitude(value, largest) > 0) {
      largest = value;
      dominant = axis;
    }
    primitive[axis] = Dyadic::integer(value.data(), value.size(), difference[axis].sign());
  }
  const Dyadic target = Dyadic(desired[dominant]) - anchor[dominant];
  std::array<std::int64_t, 3> quantum{};
  std::int64_t step = std::numeric_limits<std::int64_t>::max();
  const auto approximate = [&](const Dyadic& n, const Dyadic& d) {
    std::int64_t ne = 0, de = 0;
    const double nf = n.fraction(ne), df = d.fraction(de);
    const double initial = std::ldexp(nf / df, static_cast<int>(ne - de));
    return std::isfinite(initial) ? nearest(n, d, initial) : initial;
  };
  for (int axis = 0; axis < 3; ++axis) {
    if (!primitive[axis].sign()) continue;
    work.charge();
    const Dyadic n = anchor[axis] * primitive[dominant] + target * primitive[axis];
    const double ideal = approximate(n, primitive[dominant]);
    if (!std::isfinite(ideal)) return false;
    int exponent = 0;
    std::frexp(ideal, &exponent);
    quantum[axis] = ideal == 0.0 ? -1074 : std::max<std::int64_t>(-1074, exponent - 53);
    const std::int64_t valuation = primitive[axis].exponent();
    step = std::min(step, quantum[axis] - valuation);
    if (anchor[axis].sign()) step = std::min(step, anchor[axis].exponent() - valuation);
  }
  Limbs residue;
  std::int64_t precision = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (!primitive[axis].sign()) continue;
    const std::int64_t valuation = primitive[axis].exponent();
    const std::int64_t bits = std::max<std::int64_t>(0, quantum[axis] - step - valuation);
    const Dyadic rhs = -anchor[axis].scaled_power(-step - valuation);
    const Dyadic odd = primitive[axis].scaled_power(-valuation);
    const Limbs inverse = line_odd_inverse(odd, bits, work);
    Limbs next = line_modular_product(line_modular_integer(rhs, bits, work), inverse, bits, work);
    if (bits > precision) {
      if (compare_magnitude(line_low_bits(next, precision, work), residue) != 0) return false;
      residue = std::move(next);
      precision = bits;
    } else if (compare_magnitude(line_low_bits(residue, bits, work), next) != 0) {
      return false;
    }
  }
  const Dyadic offset = Dyadic::integer(residue.data(), residue.size(), 1).scaled_power(step);
  const Dyadic denominator = primitive[dominant].scaled_power(step + precision);
  const Dyadic numerator = target - primitive[dominant] * offset;
  const Dyadic middle = line_nearest_integer(numerator, denominator, work);
  bool accepted = false;
  Dyadic best_distance;
  double best[3];
  for (int neighbor : {0, -1, 1}) {
    work.charge();
    const Dyadic coefficient = middle + Dyadic(static_cast<double>(neighbor));
    const Dyadic parameter = offset + coefficient.scaled_power(step + precision);
    double candidate[3];
    bool represented = true;
    for (int axis = 0; axis < 3; ++axis) {
      work.charge();
      const Dyadic value = anchor[axis] + primitive[axis] * parameter;
      candidate[axis] = approximate(value, Dyadic(1.0));
      represented = represented && std::isfinite(candidate[axis]) &&
                    (value - Dyadic(candidate[axis])).sign() == 0;
    }
    if (!represented || !(std::min(first[dominant], second[dominant]) < candidate[dominant] &&
                         candidate[dominant] < std::max(first[dominant], second[dominant]))) continue;
    bool interval = true;
    for (int axis = 0; axis < 3; ++axis) {
      interval = interval && std::min(first[axis], second[axis]) <= candidate[axis] &&
                 candidate[axis] <= std::max(first[axis], second[axis]);
    }
    if (!interval) continue;
    Dyadic distance = Dyadic(candidate[dominant]) - Dyadic(desired[dominant]);
    if (distance.sign() < 0) distance.negate();
    const int comparison = accepted ? (distance - best_distance).sign() : -1;
    if (!accepted || comparison < 0 || (comparison == 0 && candidate[dominant] < best[dominant])) {
      accepted = true;
      best_distance = std::move(distance);
      std::copy_n(candidate, 3, best);
    }
  }
  if (accepted) {
    std::copy_n(best, 3, position);
    *coordinate = best[dominant];
  }
  return accepted;
}

bool construct_exact_line_point(
    const double* origin, const double* endpoint, const double* first,
    const double* second, const double* desired, double* position,
    double* coordinate, bool (*spend)(void*, std::int64_t), void* context) {
  if (construct_exact_line_point_in_binade(
          origin, endpoint, first, second, desired, position, coordinate, spend, context)) {
    return true;
  }
  // Congruences depend on the three IEEE binades, not just the dominant
  // coordinate. A desired binade may contain no representable source point
  // while another open part of the same authored carrier does.
  const LineWork work{spend, context};
  int dominant = 0;
  for (int axis = 1; axis < 3; ++axis) {
    if (std::abs(endpoint[axis] - origin[axis]) >
        std::abs(endpoint[dominant] - origin[dominant])) dominant = axis;
  }
  NativeVector<double> probes;
  const auto attempt = [&](double probe) {
    for (double value : {probe, std::nextafter(probe, -std::numeric_limits<double>::infinity()),
                        std::nextafter(probe, std::numeric_limits<double>::infinity())}) {
      double trial[3] = {desired[0], desired[1], desired[2]};
      trial[dominant] = value;
      if (construct_exact_line_point_in_binade(
              origin, endpoint, first, second, trial, position, coordinate, spend, context)) return true;
    }
    return false;
  };
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = endpoint[axis] - origin[axis];
    if (delta != 0.0 && std::min(first[axis], second[axis]) < 0.0 &&
        0.0 < std::max(first[axis], second[axis])) {
      work.charge();
      const double fraction = -origin[axis] / delta;
      if (attempt(origin[dominant] + fraction * (endpoint[dominant] - origin[dominant]))) return true;
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const double delta = endpoint[axis] - origin[axis];
    if (delta == 0.0) continue;
    const double low = std::min(first[axis], second[axis]);
    const double high = std::max(first[axis], second[axis]);
    const auto append = [&](double boundary) {
      if (!(low < boundary && boundary < high)) return;
      const double fraction = (boundary - origin[axis]) / delta;
      const double probe = origin[dominant] + fraction * (endpoint[dominant] - origin[dominant]);
      if (std::isfinite(probe)) probes.push_back(probe);
    };
    int upper_exponent = 0, lower_exponent = -1073;
    std::frexp(std::max(std::abs(low), std::abs(high)), &upper_exponent);
    if (low > 0.0 || high < 0.0) {
      std::frexp(std::min(std::abs(low), std::abs(high)), &lower_exponent);
    }
    for (int exponent = std::max(-1074, lower_exponent - 1);
         exponent <= std::min(1023, upper_exponent); ++exponent) {
      work.charge();
      const double boundary = std::ldexp(1.0, exponent);
      append(boundary);
      append(-boundary);
    }
  }
  work.charge(static_cast<std::int64_t>(probes.size()));
  std::sort(probes.begin(), probes.end(), [&](double a, double b) {
    work.charge();
    return a < b;
  });
  probes.erase(std::unique(probes.begin(), probes.end(), [&](double a, double b) {
    work.charge();
    return a == b;
  }), probes.end());
  std::sort(probes.begin(), probes.end(), [&](double a, double b) {
    work.charge();
    const double da = std::abs(a - desired[dominant]), db = std::abs(b - desired[dominant]);
    return da < db || (da == db && a < b);
  });
  for (double probe : probes) {
    if (attempt(probe)) return true;
  }
  return false;
}

static_assert(PHX_MC_ARRANGEMENT_COUNTERS == 11);

}  // namespace phx::mc

struct phx_mc_arrangement {
  std::vector<double> vertices;
  std::vector<double> bounds;
  std::vector<std::int8_t> constructions;
  std::vector<std::int64_t> origins;
  std::vector<std::int64_t> fragments;
  std::vector<std::int64_t> fragment_faces;
  std::vector<std::int64_t> contact_edges;
  std::vector<std::int64_t> coplanar;
  std::vector<std::int64_t> locations;
  phx::mc::ArrangementCounters counters;
  // Immutable input copies and the exact construction pool, retained so that
  // membership is decided on the implicit points (never on the rounded
  // publication) and never on caller memory the caller may free or mutate.
  std::vector<double> input_vertices;
  std::vector<std::int64_t> input_triangles;
  std::vector<std::int32_t> input_surfaces;
  std::unique_ptr<phx::mc::Arranger> arranger;
  phx::mc::NativeVector<std::int64_t> fragment_components;
  phx::mc::NativeVector<std::int32_t> component_windings;
};

namespace {

using phx::mc::Kernel;

[[noreturn]] void refuse_publication(const phx::mc::Output& out, std::int64_t id) {
  for (std::size_t index = 0; index < out.fragments.size(); ++index) {
    const auto& triangle = out.fragments[index];
    if (std::find(triangle.begin(), triangle.end(), id) != triangle.end()) {
      throw phx::mc::Refusal{PHX_MC_RANGE_ERROR, out.fragment_faces[index]};
    }
  }
  throw phx::mc::Refusal{PHX_MC_INTERNAL_ERROR, -1};
}

struct ArrangementPublication {
  std::vector<std::int64_t> used;
  std::vector<std::int64_t> parent;
  std::vector<std::size_t> order;
  std::vector<std::int64_t> published;
  std::vector<std::int8_t> construction;
  std::int64_t coincident = 0;
  std::int64_t constructed[4] = {0, 0, 0, 0};

  std::int64_t root_index(std::int64_t id) const {
    const auto found = std::lower_bound(used.begin(), used.end(), id);
    if (found == used.end() || *found != id) {
      return std::int64_t{-1};
    }
    return parent[static_cast<std::size_t>(found - used.begin())];
  }

  std::int64_t map_id(std::int64_t id) const {
    const auto root = root_index(id);
    return root < 0 ? std::int64_t{-1} : published[static_cast<std::size_t>(root)];
  }
};

void prepare_publication_points(Kernel& kernel, const phx::mc::Output& out,
                                ArrangementPublication& state) {
  auto& used = state.used;
  auto& parent = state.parent;
  auto& order = state.order;
  for (const auto& fragment : out.fragments) {
    used.insert(used.end(), fragment.begin(), fragment.end());
  }
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  const std::size_t n = used.size();
  parent.resize(n);
  std::iota(parent.begin(), parent.end(), 0);
  order.resize(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    const auto& a = kernel.point(used[left]);
    const auto& b = kernel.point(used[right]);
    for (int axis = 0; axis < 3; ++axis) {
      if (a.x[axis] != b.x[axis]) {
        return a.x[axis] < b.x[axis];
      }
    }
    return left < right;
  });
}

void weld_publication_points(Kernel& kernel, ArrangementPublication& state) {
  const auto& used = state.used;
  auto& parent = state.parent;
  const auto& order = state.order;
  const std::size_t n = used.size();
  double widest = 0.0;
  for (const std::int64_t id : used) {
    widest = std::max(widest, kernel.point(id).bound);
  }
  for (std::size_t i = 0; i < n; ++i) {
    const phx::mc::PoolPoint& p = kernel.point(used[order[i]]);
    const double reach = phx::mc::geometry::next_up(p.upper(0) + widest);
    for (std::size_t j = i + 1; j < n; ++j) {
      const phx::mc::PoolPoint& q = kernel.point(used[order[j]]);
      if (q.x[0] > reach) {
        break;
      }
      if (kernel.is_input(used[order[i]]) && kernel.is_input(used[order[j]]) &&
          (p.x[1] != q.x[1] || p.x[2] != q.x[2] || p.x[0] != q.x[0])) {
        continue;
      }
      const std::int64_t a = phx::mc::topology::find_root(parent, static_cast<std::int64_t>(order[i]));
      const std::int64_t b = phx::mc::topology::find_root(parent, static_cast<std::int64_t>(order[j]));
      if (a != b && kernel.equal(used[order[i]], used[order[j]])) {
        parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
      }
    }
  }
  // Roots are the smallest member (pool ids, input vertices first).
  for (std::size_t index = 0; index < n; ++index) {
    parent[index] = phx::mc::topology::find_root(parent, static_cast<std::int64_t>(index));
  }
}

void certify_publication_points(Kernel& kernel, const phx::mc::Output& out,
                                const ArrangementPublication& state) {
  const auto& used = state.used;
  const auto& parent = state.parent;
  const auto& order = state.order;
  const std::size_t n = used.size();
  // Different exact points must never be merged to repair a rounded carrier.
  // The root representatives retain the lexicographic coordinate ordering.
  std::size_t previous = n;
  for (const std::size_t index : order) {
    if (parent[index] != static_cast<std::int64_t>(index)) {
      continue;
    }
    const auto& point = kernel.point(used[index]);
    if (!std::isfinite(point.bound) || !std::isfinite(point.x[0]) ||
        !std::isfinite(point.x[1]) || !std::isfinite(point.x[2])) {
      refuse_publication(out, used[index]);
    }
    if (previous != n) {
      const auto& before = kernel.point(used[previous]);
      if (point.x[0] == before.x[0] && point.x[1] == before.x[1] &&
          point.x[2] == before.x[2]) {
        refuse_publication(out, used[index]);
      }
    }
    previous = index;
  }
}

void publish_point_representatives(Kernel& kernel, ArrangementPublication& state,
                                   phx_mc_arrangement& result) {
  const auto& used = state.used;
  const auto& parent = state.parent;
  auto& published = state.published;
  auto& construction = state.construction;
  const std::size_t n = used.size();
  published.assign(n, -1);
  construction.assign(n, phx::mc::kTriplePlane);
  for (std::size_t index = 0; index < n; ++index) {
    auto& best = construction[static_cast<std::size_t>(parent[index])];
    best = std::min(best, kernel.point(used[index]).construction);
  }
  auto& coincident = state.coincident;
  auto& constructed = state.constructed;
  for (std::size_t index = 0; index < n; ++index) {
    const std::size_t root = static_cast<std::size_t>(parent[index]);
    if (root != index) {
      coincident += kernel.is_input(used[index]) ? 1 : 0;
      continue;
    }
    published[index] = static_cast<std::int64_t>(result.bounds.size());
    const phx::mc::PoolPoint& point = kernel.point(used[index]);
    result.vertices.insert(result.vertices.end(), point.x, point.x + 3);
    result.bounds.push_back(point.bound);
    result.constructions.push_back(construction[index]);
    result.origins.push_back(kernel.is_input(used[index]) ? point.def.vertex : -1);
    ++constructed[construction[index]];
  }
}

void publish_fragment_carrier(Kernel& kernel, const phx::mc::Output& out,
                              const ArrangementPublication& state,
                              phx_mc_arrangement& result) {
  const auto& used = state.used;
  const auto& published = state.published;
  for (std::size_t index = 0; index < out.fragments.size(); ++index) {
    const auto& fragment = out.fragments[index];
    std::array<std::int64_t, 3> representatives;
    for (int corner = 0; corner < 3; ++corner) {
      const auto root = static_cast<std::size_t>(state.root_index(fragment[corner]));
      representatives[corner] = used[root];
      result.fragments.push_back(published[root]);
    }
    const int frame = out.fragment_frames[index];
    const int axis = std::abs(frame) - 1;
    const int orientation = kernel.rounded_orient2d(
        representatives[0], representatives[1], representatives[2],
        (axis + 1) % 3, (axis + 2) % 3);
    if (orientation != (frame > 0 ? 1 : -1)) {
      throw phx::mc::Refusal{PHX_MC_RANGE_ERROR, out.fragment_faces[index]};
    }
  }
  result.fragment_faces.assign(out.fragment_faces.begin(), out.fragment_faces.end());
}

void publish_contact_relations(const phx::mc::Output& out,
                               const ArrangementPublication& state,
                               phx_mc_arrangement& result) {
  std::vector<std::array<std::int64_t, 2>> edges;
  for (const auto& edge : out.contact_edges) {
    const std::int64_t a = state.map_id(edge[0]);
    const std::int64_t b = state.map_id(edge[1]);
    edges.push_back({std::min(a, b), std::max(a, b)});
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  for (const auto& edge : edges) {
    result.contact_edges.insert(result.contact_edges.end(), edge.begin(), edge.end());
  }
  for (const auto& record : out.coplanar) {
    result.coplanar.insert(result.coplanar.end(), record.begin(), record.end());
  }
  std::vector<std::array<std::int64_t, 3>> locations;
  for (const auto& record : out.locations) {
    const std::int64_t vertex = state.map_id(record[0]);
    if (vertex >= 0) {
      locations.push_back({vertex, record[1], record[2]});
    }
  }
  std::sort(locations.begin(), locations.end());
  locations.erase(std::unique(locations.begin(), locations.end()), locations.end());
  for (const auto& record : locations) {
    result.locations.insert(result.locations.end(), record.begin(), record.end());
  }
}

// Pool order fixes the representative identity: input points first, then
// constructed points in creation order. Rounded carriers are certified only
// after exact welding and before any result coordinates are published.
void publish(Kernel& kernel, const phx::mc::Output& out, std::int64_t pair_count,
             phx_mc_arrangement& result) {
  ArrangementPublication state;
  prepare_publication_points(kernel, out, state);
  weld_publication_points(kernel, state);
  certify_publication_points(kernel, out, state);
  publish_point_representatives(kernel, state, result);
  publish_fragment_carrier(kernel, out, state, result);
  publish_contact_relations(out, state, result);
  result.counters = {pair_count,
                     out.contact_pairs,
                     out.coplanar_pairs,
                     state.coincident,
                     state.constructed[phx::mc::kEdgePlane],
                     state.constructed[phx::mc::kEdgeEdge],
                     state.constructed[phx::mc::kTriplePlane],
                     out.split_faces,
                     static_cast<std::int64_t>(out.fragments.size()),
                     kernel.filtered(),
                     kernel.exact()};
}

// Counts of a classification: components, operands, representative/triangle
// pair scans and winding-matrix entries.
static_assert(PHX_MC_ARRANGEMENT_CLASSIFICATION_VALUES == 4);

// Joins the fragments of each operand across shared edges that are not
// contact edges.  A component crosses no other operand's surface, so its
// membership in every other operand is that of its lowest fragment, whose
// exact winding number is counted on the implicit corners.  Operands coplanar
// with the component (and its own operand) have no membership and receive
// INT32_MIN.  The operand namespace is the declared dense range
// [0, operand_count); the matrix entries and every pair scan are admitted
// against work_limit and the ambient native scope before anything is
// allocated for them.
void classify(phx_mc_arrangement& result, std::int64_t operand_count, std::int64_t work_limit,
              phx::mc::ArrangementClassificationVisits& sizes) {
  using phx::mc::NativeVector;
  constexpr std::int32_t kUnclassified = std::numeric_limits<std::int32_t>::min();
  Kernel& kernel = result.arranger->kernel();
  const phx::mc::Output& out = result.arranger->output();
  const std::size_t count = out.fragments.size();
  const std::size_t face_total = result.input_surfaces.size();
  if (operand_count < 1 || static_cast<std::uint64_t>(operand_count) > face_total) {
    throw phx::mc::Refusal{PHX_MC_INVALID_INPUT, -1};
  }
  const auto operands = static_cast<std::size_t>(operand_count);
  NativeVector<std::int64_t> face_offsets(operands + 1, 0);
  for (std::size_t face = 0; face < face_total; ++face) {
    const std::int32_t label = result.input_surfaces[face];
    if (label < 0 || label >= operand_count) {
      throw phx::mc::Refusal{PHX_MC_INVALID_INPUT, static_cast<std::int64_t>(face)};
    }
    ++face_offsets[static_cast<std::size_t>(label) + 1];
  }
  for (std::size_t operand = 0; operand < operands; ++operand) {
    if (face_offsets[operand + 1] == 0) {
      throw phx::mc::Refusal{PHX_MC_INVALID_INPUT, -1};
    }
    face_offsets[operand + 1] += face_offsets[operand];
  }
  NativeVector<std::int64_t> operand_faces(face_total);
  {
    NativeVector<std::int64_t> cursor(face_offsets.begin(), face_offsets.end() - 1);
    for (std::size_t face = 0; face < face_total; ++face) {
      const auto label = static_cast<std::size_t>(result.input_surfaces[face]);
      operand_faces[static_cast<std::size_t>(cursor[label]++)] = static_cast<std::int64_t>(face);
    }
  }
  const auto operand_of = [&](std::size_t fragment) {
    return static_cast<std::size_t>(
        result.input_surfaces[static_cast<std::size_t>(out.fragment_faces[fragment])]);
  };
  NativeVector<std::array<std::int64_t, 4>> sides;  // low, high, operand, fragment
  sides.reserve(3 * count);
  for (std::size_t fragment = 0; fragment < count; ++fragment) {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const std::int64_t a = result.fragments[3 * fragment + corner];
      const std::int64_t b = result.fragments[3 * fragment + (corner + 1) % 3];
      sides.push_back({std::min(a, b), std::max(a, b), static_cast<std::int64_t>(operand_of(fragment)),
                       static_cast<std::int64_t>(fragment)});
    }
  }
  std::sort(sides.begin(), sides.end());
  const std::size_t edge_count = result.contact_edges.size() / 2;
  const auto contact = [&](std::int64_t low, std::int64_t high) {
    std::size_t first = 0;
    std::size_t last = edge_count;
    while (first < last) {
      const std::size_t middle = first + (last - first) / 2;
      const std::int64_t a = result.contact_edges[2 * middle];
      const std::int64_t b = result.contact_edges[2 * middle + 1];
      if (a < low || (a == low && b < high)) {
        first = middle + 1;
      } else {
        last = middle;
      }
    }
    return first < edge_count && result.contact_edges[2 * first] == low &&
           result.contact_edges[2 * first + 1] == high;
  };
  NativeVector<std::int64_t> parent(count);
  std::iota(parent.begin(), parent.end(), 0);
  for (std::size_t index = 1; index < sides.size(); ++index) {
    const auto& before = sides[index - 1];
    const auto& side = sides[index];
    if (before[0] != side[0] || before[1] != side[1] || before[2] != side[2] ||
        contact(side[0], side[1])) {
      continue;
    }
    const std::int64_t a = phx::mc::topology::find_root(parent, before[3]);
    const std::int64_t b = phx::mc::topology::find_root(parent, side[3]);
    parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
  }
  // Components are numbered by, and represented by, their lowest fragment.
  NativeVector<std::int64_t> representatives;
  NativeVector<std::int64_t> numbering(count, -1);
  NativeVector<std::int64_t> components(count, -1);
  for (std::size_t fragment = 0; fragment < count; ++fragment) {
    const auto top = static_cast<std::size_t>(
        phx::mc::topology::find_root(parent, static_cast<std::int64_t>(fragment)));
    if (numbering[top] < 0) {
      numbering[top] = static_cast<std::int64_t>(representatives.size());
      representatives.push_back(static_cast<std::int64_t>(fragment));
    }
    components[fragment] = numbering[top];
  }
  NativeVector<std::array<std::int64_t, 2>> coplanar;  // fragment, operand
  coplanar.reserve(out.coplanar.size());
  for (const auto& record : out.coplanar) {
    coplanar.push_back({record[0], result.input_surfaces[static_cast<std::size_t>(record[1])]});
  }
  std::sort(coplanar.begin(), coplanar.end());
  const auto classified = [&](std::size_t fragment, std::size_t operand) {
    return operand != operand_of(fragment) &&
           !std::binary_search(coplanar.begin(), coplanar.end(),
                               std::array<std::int64_t, 2>{static_cast<std::int64_t>(fragment),
                                                           static_cast<std::int64_t>(operand)});
  };
  const std::size_t component_count = representatives.size();
  sizes.components = static_cast<std::int64_t>(component_count);
  sizes.operands = operand_count;
  // Refusal evidence records the evaluated requirement before admission.
  // An unrepresentable requirement reports its largest representable lower bound.
  if (component_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                            operands) {
    sizes.matrix_entries = std::numeric_limits<std::int64_t>::max();
    throw phx::mc::Refusal{PHX_MC_CAPACITY_EXCEEDED, -1};
  }
  const auto entries = static_cast<std::int64_t>(component_count * operands);
  sizes.matrix_entries = entries;
  if (entries > work_limit) {
    throw phx::mc::Refusal{PHX_MC_CAPACITY_EXCEEDED, -1};
  }
  std::int64_t scans = 0;
  for (std::size_t component = 0; component < component_count; ++component) {
    const auto fragment = static_cast<std::size_t>(representatives[component]);
    for (std::size_t operand = 0; operand < operands; ++operand) {
      if (!classified(fragment, operand)) {
        continue;
      }
      scans += face_offsets[operand + 1] - face_offsets[operand];
      if (scans > work_limit - entries) {
        sizes.pair_scans = scans;
        throw phx::mc::Refusal{PHX_MC_CAPACITY_EXCEEDED, out.fragment_faces[fragment]};
      }
    }
  }
  sizes.pair_scans = scans;
  phx::mc::native_execution_charge(entries + scans);
  NativeVector<std::int32_t> windings(static_cast<std::size_t>(entries), kUnclassified);
  NativeVector<std::int8_t> normal_signs(3 * face_total, 2);
  for (std::size_t component = 0; component < component_count; ++component) {
    const auto fragment = static_cast<std::size_t>(representatives[component]);
    const auto& corners = out.fragments[fragment];
    const int axis = std::abs(out.fragment_frames[fragment]) - 1;
    for (std::size_t operand = 0; operand < operands; ++operand) {
      if (!classified(fragment, operand)) {
        continue;
      }
      const auto first = static_cast<std::size_t>(face_offsets[operand]);
      const auto last = static_cast<std::size_t>(face_offsets[operand + 1]);
      int value = 0;
      if (!kernel.winding(corners[0], corners[1], corners[2], axis,
                          result.input_triangles.data(), operand_faces.data() + first,
                          last - first, normal_signs, value)) {
        // A fragment interior meets another operand only where coplanar.
        throw phx::mc::Refusal{PHX_MC_INTERNAL_ERROR, out.fragment_faces[fragment]};
      }
      windings[component * operands + operand] = value;
    }
  }
  result.fragment_components = std::move(components);
  result.component_windings = std::move(windings);
}

}  // namespace

extern "C" {

int32_t phx_mc_arrange_triangles(int64_t vertex_count, const double* vertices,
                                 int64_t triangle_count, const int64_t* triangles,
                                 const int32_t* surfaces, int64_t pair_count,
                                 const int64_t* pairs, int64_t max_points,
                                 int64_t max_fragments, int64_t* failed_face,
                                 phx_mc_arrangement** result) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || failed_face == nullptr || vertices == nullptr ||
        triangles == nullptr || surfaces == nullptr || (pair_count > 0 && pairs == nullptr) ||
        !phx::mc::addressable(vertex_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(pair_count, 2, sizeof(int64_t)) || max_points < vertex_count ||
        max_fragments < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *result = nullptr;
    *failed_face = -1;
    for (int64_t index = 0; index < 3 * vertex_count; ++index) {
      if (!std::isfinite(vertices[index])) {
        return PHX_MC_NONFINITE_INPUT;
      }
    }
    for (int64_t index = 0; index < 3 * triangle_count; ++index) {
      if (triangles[index] < 0 || triangles[index] >= vertex_count) {
        *failed_face = index / 3;
        return PHX_MC_INVALID_INPUT;
      }
    }
    std::vector<std::vector<int64_t>> partners(static_cast<std::size_t>(triangle_count));
    for (int64_t index = 0; index < pair_count; ++index) {
      const int64_t a = pairs[2 * index];
      const int64_t b = pairs[2 * index + 1];
      if (a < 0 || b < 0 || a >= triangle_count || b >= triangle_count ||
          surfaces[a] == surfaces[b]) {
        *failed_face = a;
        return PHX_MC_INVALID_INPUT;
      }
      partners[static_cast<std::size_t>(a)].push_back(b);
      partners[static_cast<std::size_t>(b)].push_back(a);
    }
    for (auto& list : partners) {
      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());
    }
    auto handle = std::make_unique<phx_mc_arrangement>();
    handle->input_vertices.assign(vertices, vertices + 3 * vertex_count);
    handle->input_triangles.assign(triangles, triangles + 3 * triangle_count);
    handle->input_surfaces.assign(surfaces, surfaces + triangle_count);
    handle->arranger = std::make_unique<phx::mc::Arranger>(
        handle->input_vertices.data(), vertex_count, handle->input_triangles.data(), max_points,
        max_fragments);
    phx::mc::Arranger& arranger = *handle->arranger;
    try {
      for (int64_t t = 0; t < triangle_count; ++t) {
        arranger.face(t, partners[static_cast<std::size_t>(t)]);
      }
      publish(arranger.kernel(), arranger.output(), pair_count, *handle);
    } catch (const phx::mc::Refusal& refusal) {
      *failed_face = refusal.face;
      return refusal.status;
    }
    *result = handle.release();
    return PHX_MC_OK;
  });
}

int32_t phx_mc_arrangement_classify(phx_mc_arrangement* result, int64_t operand_count,
                                    int64_t work_limit, int64_t* failed_face, int64_t* sizes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || result->arranger == nullptr || failed_face == nullptr ||
        sizes == nullptr || work_limit < 0) {
      return PHX_MC_INVALID_ARGUMENT;
    }
    *failed_face = -1;
    phx::mc::ArrangementClassificationVisits state;
    phx::mc::CounterPublisher publisher{state, sizes};
    result->fragment_components.clear();
    result->component_windings.clear();
    try {
      classify(*result, operand_count, work_limit, state);
    } catch (const phx::mc::Refusal& refusal) {
      *failed_face = refusal.face;
      return refusal.status;
    } catch (const phx::mc::ExecutionRefusal& refusal) {
      return refusal.status;
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_arrangement_export_classification(const phx_mc_arrangement* result,
                                                 int64_t* fragment_components,
                                                 int32_t* component_windings) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr ||
        !phx::mc::output_bank_valid(result->fragment_components, fragment_components) ||
        !phx::mc::output_bank_valid(result->component_windings, component_windings))
      return PHX_MC_INVALID_ARGUMENT;
    phx::mc::copy_output_bank(result->fragment_components, fragment_components);
    phx::mc::copy_output_bank(result->component_windings, component_windings);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_arrangement_sizes(const phx_mc_arrangement* result, int64_t* sizes) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || sizes == nullptr) return PHX_MC_INVALID_ARGUMENT;
    sizes[0] = static_cast<int64_t>(result->bounds.size());
    sizes[1] = static_cast<int64_t>(result->fragment_faces.size());
    sizes[2] = static_cast<int64_t>(result->contact_edges.size() / 2);
    sizes[3] = static_cast<int64_t>(result->coplanar.size() / 3);
    sizes[4] = static_cast<int64_t>(result->locations.size() / 3);
    return PHX_MC_OK;
  });
}

int32_t phx_mc_arrangement_counters(const phx_mc_arrangement* result, int64_t* counters) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr || counters == nullptr) return PHX_MC_INVALID_ARGUMENT;
    phx::mc::CounterPublisher publisher{result->counters, counters};
    return PHX_MC_OK;
  });
}

int32_t phx_mc_arrangement_export(const phx_mc_arrangement* result, double* vertices,
                                  double* bounds, int8_t* constructions, int64_t* origins,
                                  int64_t* fragments, int64_t* fragment_faces,
                                  int64_t* contact_edges, int64_t* coplanar,
                                  int64_t* locations) {
  return phx::mc::guarded([&]() -> int32_t {
    if (result == nullptr ||
        !phx::mc::output_bank_valid(result->vertices, vertices) ||
        !phx::mc::output_bank_valid(result->bounds, bounds) ||
        !phx::mc::output_bank_valid(result->constructions, constructions) ||
        !phx::mc::output_bank_valid(result->origins, origins) ||
        !phx::mc::output_bank_valid(result->fragments, fragments) ||
        !phx::mc::output_bank_valid(result->fragment_faces, fragment_faces) ||
        !phx::mc::output_bank_valid(result->contact_edges, contact_edges) ||
        !phx::mc::output_bank_valid(result->coplanar, coplanar) ||
        !phx::mc::output_bank_valid(result->locations, locations)) return PHX_MC_INVALID_ARGUMENT;
    phx::mc::copy_output_bank(result->vertices, vertices);
    phx::mc::copy_output_bank(result->bounds, bounds);
    phx::mc::copy_output_bank(result->constructions, constructions);
    phx::mc::copy_output_bank(result->origins, origins);
    phx::mc::copy_output_bank(result->fragments, fragments);
    phx::mc::copy_output_bank(result->fragment_faces, fragment_faces);
    phx::mc::copy_output_bank(result->contact_edges, contact_edges);
    phx::mc::copy_output_bank(result->coplanar, coplanar);
    phx::mc::copy_output_bank(result->locations, locations);
    return PHX_MC_OK;
  });
}

void phx_mc_arrangement_free(phx_mc_arrangement* result) { delete result; }

int32_t phx_mc_clip_reference_triangle_exact(
    const uint32_t* words, int64_t word_count, const int64_t* offsets,
    const int8_t* signs, int64_t scratch_limit, int64_t work_limit,
    int32_t vertex_capacity, int32_t* supporting_planes, int32_t* edge_labels,
    int32_t* vertex_count, int64_t* work_units) {
  return phx::mc::guarded([&]() -> int32_t {
    return phx::mc::clip_reference_triangle_exact(
        words, word_count, offsets, signs, scratch_limit, work_limit,
        vertex_capacity, supporting_planes, edge_labels, vertex_count, work_units);
  });
}

int32_t phx_mc_clip_reference_tetrahedron_exact(
    const phx_mc_clip_reference_tetrahedron_exact_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  auto words = request->words;
  auto word_count = request->word_count;
  auto offsets = request->offsets;
  auto signs = request->signs;
  auto scratch_limit = request->scratch_limit;
  auto work_limit = request->work_limit;
  auto vertex_capacity = request->vertex_capacity;
  auto face_capacity = request->face_capacity;
  auto entry_capacity = request->entry_capacity;
  auto supporting_planes = request->supporting_planes;
  auto incident_planes = request->incident_planes;
  auto face_offsets = request->face_offsets;
  auto face_planes = request->face_planes;
  auto face_vertices = request->face_vertices;
  auto vertex_count = request->vertex_count;
  auto face_count = request->face_count;
  auto work_units = request->work_units;
  return phx::mc::guarded([&]() -> int32_t {
    return phx::mc::clip_reference_tetrahedron_exact(
        words, word_count, offsets, signs, scratch_limit, work_limit,
        vertex_capacity, face_capacity, entry_capacity, supporting_planes,
        incident_planes, face_offsets, face_planes, face_vertices,
        vertex_count, face_count, work_units);
  });
}

}  // extern "C"
