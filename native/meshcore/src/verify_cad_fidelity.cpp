// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_cad_fidelity.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "capi_guard.hpp"
#include "directed_interval.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"
#include "verify_chart_partition.hpp"
#include "verify_cad_pole_chart.hpp"

namespace phx::mc::cad {
namespace {
namespace di = phx::mc::remesh;

struct Vertex { double weight[3]; };
struct Leaf { Vertex vertices[3]; int32_t depth; };

Vertex midpoint(const Vertex& a, const Vertex& b) {
  Vertex value{};
  for (int i = 0; i < 3; ++i) value.weight[i] = (a.weight[i] + b.weight[i]) * 0.5;
  return value;
}

void chart_point(const Vertex& vertex, const double* uv, di::Interval* point) {
  for (int axis = 0; axis < 2; ++axis) {
    point[axis] = di::point(0.0);
    for (int i = 0; i < 3; ++i)
      point[axis] = di::add(point[axis], di::multiply(di::point(vertex.weight[i]),
                                                     di::point(uv[2 * i + axis])));
  }
}

// chart_point, but a point interval when every product and sum is exact in
// binary64 (checked with error-free transformations), as for the original
// vertices and most dyadic midpoints.
void exact_chart_point(const Vertex& vertex, const double* uv, di::Interval* point) {
  for (int axis = 0; axis < 2; ++axis) {
    double sum = 0.0;
    bool exact = true;
    for (int i = 0; i < 3 && exact; ++i) {
      const double product = vertex.weight[i] * uv[2 * i + axis];
      const double next = sum + product;
      // Knuth's TwoSum: the rounding error of sum + product.
      const double bv = next - sum, av = next - bv;
      exact = std::fma(vertex.weight[i], uv[2 * i + axis], -product) == 0.0 &&
              (sum - av) + (product - bv) == 0.0;
      sum = next;
    }
    if (exact && std::isfinite(sum)) {
      point[axis] = di::point(sum);
    } else {
      di::Interval enclosure[2];
      chart_point(vertex, uv, enclosure);
      point[axis] = enclosure[axis];
    }
  }
}

using phx::mc::geometry::next_down;
using phx::mc::geometry::next_up;

di::Interval negate(const di::Interval& x) { return {-x.upper, -x.lower}; }

// x / c for an exact positive constant c.
di::Interval divide_by(const di::Interval& x, double c) {
  return {next_down(x.lower / c), next_up(x.upper / c)};
}

di::Interval square(const di::Interval& x) {
  const double a = std::fabs(x.lower), b = std::fabs(x.upper);
  const double low = x.lower > 0.0 || x.upper < 0.0 ? std::min(a, b) : 0.0;
  const double high = std::max(a, b);
  return {low == 0.0 ? 0.0 : std::max(0.0, next_down(low * low)), next_up(high * high)};
}

// 1 / x for x > 0; false otherwise.
bool reciprocal(const di::Interval& x, di::Interval& result) {
  if (!(x.lower > 0.0) || !di::finite(x)) return false;
  result = {next_down(1.0 / x.upper), next_up(1.0 / x.lower)};
  return true;
}

// Rigorous cos and sin enclosures at a few interval operations per call.
// Anchors a_k = k 2^-6 hold enclosures of cos a_k and sin a_k from the
// elementary kernel, computed on first use. For x in X near a_k, with
// D = X - a_k (outward), cos x = cos a_k cos d - sin a_k sin d and
// sin x = sin a_k cos d + cos a_k sin d, where by Lagrange's remainder
// (every derivative of sin and cos is bounded by 1)
//   sin d = d - d^3/6 + d^5/120 + r7,           |r7| <= |d|^7 / 7!,
//   cos d = 1 - d^2/2 + d^4/24 - d^6/720 + r8,  |r8| <= d^8 / 8!.
// The polynomials are evaluated in outward interval arithmetic over D, so the
// result encloses cos and sin on all of X. Wide intervals (|D| would exceed
// 2^-4) and far arguments use the elementary kernel directly.
class AnchoredTrigonometry {
 public:
  void evaluate(const di::Interval& x, di::Interval& cosine, di::Interval& sine) {
    const double mid = 0.5 * x.lower + 0.5 * x.upper;
    const double scaled = mid * kScale;
    if (!(x.upper - x.lower <= 0x1p-4) || !(std::fabs(scaled) < kLimit)) {
      direct(x, cosine, sine);
      return;
    }
    const auto k = static_cast<int64_t>(std::nearbyint(scaled));
    di::Interval anchor_cosine, anchor_sine;
    anchor(k, anchor_cosine, anchor_sine);
    const di::Interval d = di::subtract(x, di::point(static_cast<double>(k) / kScale));
    const di::Interval d2 = square(d);
    const double m = di::magnitude(d);
    const double m2 = next_up(m * m), m4 = next_up(m2 * m2);
    const double r7 = next_up(next_up(m4 * next_up(m2 * m)) / 5040.0);
    const double r8 = next_up(next_up(m4 * m4) / 40320.0);
    const di::Interval one = di::point(1.0);
    di::Interval s = di::subtract(one, divide_by(d2, 20.0));
    s = di::subtract(one, di::multiply(divide_by(d2, 6.0), s));
    s = di::add(di::multiply(d, s), di::Interval{-r7, r7});
    di::Interval c = di::subtract(one, divide_by(d2, 30.0));
    c = di::subtract(one, di::multiply(divide_by(d2, 12.0), c));
    c = di::subtract(one, di::multiply(divide_by(d2, 2.0), c));
    c = di::add(c, di::Interval{-r8, r8});
    cosine = clip(di::subtract(di::multiply(anchor_cosine, c), di::multiply(anchor_sine, s)));
    sine = clip(di::add(di::multiply(anchor_sine, c), di::multiply(anchor_cosine, s)));
  }

 private:
  static constexpr double kScale = 64.0;
  static constexpr double kLimit = 0x1p22;
  static di::Interval clip(const di::Interval& x) {
    return {std::max(x.lower, -1.0), std::min(x.upper, 1.0)};
  }
  static void direct(const di::Interval& x, di::Interval& cosine, di::Interval& sine) {
    const auto c = phx::mc::elementary::cos(x.lower, x.upper);
    const auto s = phx::mc::elementary::sin(x.lower, x.upper);
    cosine = {c.lower, c.upper};
    sine = {s.lower, s.upper};
  }
  void anchor(int64_t k, di::Interval& cosine, di::Interval& sine) {
    const auto size = static_cast<int64_t>(ready_.size());
    if (size == 0 || k < first_ || k >= first_ + size) {
      const int64_t low = size == 0 ? k - 64 : std::min(first_, k - 64);
      const int64_t high = size == 0 ? k + 65 : std::max(first_ + size, k + 65);
      std::vector<di::Interval> c(static_cast<std::size_t>(high - low)), s(c.size());
      std::vector<uint8_t> ready(c.size(), 0);
      for (int64_t i = 0; i < size; ++i) {
        const auto from = static_cast<std::size_t>(i), to = static_cast<std::size_t>(first_ + i - low);
        c[to] = cosine_[from];
        s[to] = sine_[from];
        ready[to] = ready_[from];
      }
      cosine_.swap(c);
      sine_.swap(s);
      ready_.swap(ready);
      first_ = low;
    }
    const auto slot = static_cast<std::size_t>(k - first_);
    if (!ready_[slot]) {
      const double a = static_cast<double>(k) / kScale;
      direct(di::point(a), cosine_[slot], sine_[slot]);
      ready_[slot] = 1;
    }
    cosine = cosine_[slot];
    sine = sine_[slot];
  }
  std::vector<di::Interval> cosine_, sine_;
  std::vector<uint8_t> ready_;
  int64_t first_ = 0;
};

// Interval frame of a closed-form node (the closed_data layout of
// CadGeometry::bind_closed_form), built from the exact-source coefficient
// enclosures instead of the binary64 bank. The closed forms are the source's
// own formulas, so their interval evaluation encloses the source map.
struct ClosedFrame {
  ClosedForm kind = ClosedForm::kNone;
  di::Interval data[18];
};

bool closed_frame(const CadGeometry& geometry, int32_t index,
                  const IntervalEvaluationWorkspace& workspace, ClosedFrame& frame) {
  const Node& node = geometry.node(index);
  if (node.closed == ClosedForm::kNone) return false;
  const auto coefficient = [&](int32_t at, int64_t i) {
    const int64_t k = geometry.model().node_real_offsets[at] + i;
    return di::Interval{workspace.coefficient_lower[k], workspace.coefficient_upper[k]};
  };
  frame = ClosedFrame{};
  if (node.kind == PHX_MC_CAD_NODE_TEMPLATE) {
    if (node.real_count > 18) return false;
    frame.kind = node.closed;
    for (int64_t i = 0; i < node.real_count; ++i) frame.data[i] = coefficient(index, i);
    if (frame.kind == ClosedForm::kCone) {
      const auto slope = phx::mc::elementary::tan(frame.data[13].lower, frame.data[13].upper);
      frame.data[13] = {slope.lower, slope.upper};
    }
  } else if (node.kind == PHX_MC_CAD_NODE_AFFINE) {
    ClosedFrame child;
    if (!closed_frame(geometry, static_cast<int32_t>(node.integers[0]), workspace, child))
      return false;
    frame = child;
    const int vectors = child.kind == ClosedForm::kLineRevolution ? 5 : 3;
    for (int slot = 0; slot <= vectors; ++slot)
      for (int r = 0; r < 3; ++r) {
        di::Interval value = slot == 0 ? coefficient(index, 9 + r) : di::point(0.0);
        for (int c = 0; c < 3; ++c)
          value = di::add(value, di::multiply(coefficient(index, 3 * r + c),
                                              child.data[3 * slot + c]));
        frame.data[3 * slot + r] = value;
      }
  } else if (node.kind == PHX_MC_CAD_NODE_REVOLUTION) {
    // bind_closed_form's linearization of the revolution of a line profile.
    static const int32_t kExactLine = template_index("exact_line3");
    const auto profile = static_cast<int32_t>(node.integers[0]);
    const bool exact = geometry.node(profile).integers[0] == kExactLine;
    di::Interval origin[3], a[3], relative[3], d[3];
    for (int i = 0; i < 3; ++i) {
      origin[i] = coefficient(index, i);
      a[i] = coefficient(index, 3 + i);
      const di::Interval first = coefficient(profile, i), second = coefficient(profile, 3 + i);
      d[i] = exact ? di::subtract(second, first) : second;
      relative[i] = di::subtract(first, origin[i]);
    }
    di::Interval dot = di::point(0.0), slope = di::point(0.0);
    for (int i = 0; i < 3; ++i) {
      dot = di::add(dot, di::multiply(relative[i], a[i]));
      slope = di::add(slope, di::multiply(d[i], a[i]));
    }
    di::Interval* data = frame.data;
    for (int i = 0; i < 3; ++i) {
      data[i] = di::add(origin[i], di::multiply(a[i], dot));
      data[3 + i] = di::multiply(a[i], slope);
      data[6 + i] = di::subtract(relative[i], di::multiply(a[i], dot));
      data[9 + i] = di::subtract(d[i], di::multiply(a[i], slope));
    }
    for (int base : {6, 9}) {
      const di::Interval* r = data + base;
      di::Interval* c = data + base + 6;
      c[0] = di::subtract(di::multiply(a[1], r[2]), di::multiply(a[2], r[1]));
      c[1] = di::subtract(di::multiply(a[2], r[0]), di::multiply(a[0], r[2]));
      c[2] = di::subtract(di::multiply(a[0], r[1]), di::multiply(a[1], r[0]));
    }
    frame.kind = ClosedForm::kLineRevolution;
  } else {
    return false;
  }
  for (const di::Interval& value : frame.data)
    if (!di::finite(value)) return false;
  return true;
}

// Terms first..last of the point-jet layout (0 S, 1 S_u, 2 S_v, 3 S_uu,
// 4 S_uv, 5 S_vv) of a closed form over the box, by the derivatives of the
// formulas documented at CadGeometry::expand_closed. With
// R = cos u e1 + sin u e2 and R' = -sin u e1 + cos u e2:
//   plane     o + u e1 + v e2
//   cylinder  o + r R + v a;               S_uu = -r R
//   cone      o + v a + (r0 + v t) R;      S_uu = -(r0 + v t) R, S_uv = t R'
//   sphere    o + r (cos v R + sin v a);   S_uu = -r cos v R, S_uv = -r sin v R',
//                                          S_vv = -r (cos v R + sin v a)
//   torus     o + (A + b cos v) R + b sin v a; S_uu = -(A + b cos v) R,
//             S_uv = -b sin v R', S_vv = -b (cos v R + sin v a)
//   line revolution P0 + v P1 + (R0 + v R1) cos u + (C0 + v C1) sin u.
void closed_jets(const ClosedFrame& frame, AnchoredTrigonometry& trig, const double* lower,
                 const double* upper, int first, int last, di::Interval out[6][3]) {
  using di::add, di::multiply, di::subtract;
  const di::Interval u{lower[0], upper[0]}, v{lower[1], upper[1]};
  const di::Interval* d = frame.data;
  const di::Interval *o = d, *e1 = d + 3, *e2 = d + 6, *axis = d + 9;
  const auto want = [&](int term) { return first <= term && term <= last; };
  const di::Interval zero = di::point(0.0);
  di::Interval c{}, s{}, cv{}, sv{};
  if (frame.kind != ClosedForm::kPlane) trig.evaluate(u, c, s);
  if (frame.kind == ClosedForm::kSphere || frame.kind == ClosedForm::kTorus)
    trig.evaluate(v, cv, sv);
  di::Interval radial[3], turned[3];
  if (frame.kind != ClosedForm::kPlane && frame.kind != ClosedForm::kLineRevolution)
    for (int i = 0; i < 3; ++i) {
      radial[i] = add(multiply(c, e1[i]), multiply(s, e2[i]));
      turned[i] = subtract(multiply(c, e2[i]), multiply(s, e1[i]));
    }
  switch (frame.kind) {
    case ClosedForm::kPlane:
      for (int i = 0; i < 3; ++i) {
        if (want(0)) out[0][i] = add(o[i], add(multiply(u, e1[i]), multiply(v, e2[i])));
        out[1][i] = e1[i];
        out[2][i] = e2[i];
        out[3][i] = out[4][i] = out[5][i] = zero;
      }
      return;
    case ClosedForm::kCylinder: {
      const di::Interval r = d[12];
      for (int i = 0; i < 3; ++i) {
        const di::Interval scaled = multiply(r, radial[i]);
        if (want(0)) out[0][i] = add(o[i], add(scaled, multiply(v, axis[i])));
        if (want(1)) out[1][i] = multiply(r, turned[i]);
        out[2][i] = axis[i];
        out[3][i] = negate(scaled);
        out[4][i] = out[5][i] = zero;
      }
      return;
    }
    case ClosedForm::kCone: {
      const di::Interval slope = d[13], rho = add(d[12], multiply(v, slope));
      for (int i = 0; i < 3; ++i) {
        const di::Interval scaled = multiply(rho, radial[i]);
        if (want(0)) out[0][i] = add(o[i], add(multiply(v, axis[i]), scaled));
        if (want(1)) out[1][i] = multiply(rho, turned[i]);
        if (want(2)) out[2][i] = add(axis[i], multiply(slope, radial[i]));
        out[3][i] = negate(scaled);
        out[4][i] = multiply(slope, turned[i]);
        out[5][i] = zero;
      }
      return;
    }
    case ClosedForm::kSphere: {
      const di::Interval r = d[12], rc = multiply(r, cv), rs = multiply(r, sv);
      for (int i = 0; i < 3; ++i) {
        const di::Interval direction = add(multiply(cv, radial[i]), multiply(sv, axis[i]));
        if (want(0)) out[0][i] = add(o[i], multiply(r, direction));
        if (want(1)) out[1][i] = multiply(rc, turned[i]);
        if (want(2)) out[2][i] = subtract(multiply(rc, axis[i]), multiply(rs, radial[i]));
        out[3][i] = negate(multiply(rc, radial[i]));
        out[4][i] = negate(multiply(rs, turned[i]));
        out[5][i] = negate(multiply(r, direction));
      }
      return;
    }
    case ClosedForm::kTorus: {
      const di::Interval b = d[13], ring = add(d[12], multiply(b, cv));
      const di::Interval bc = multiply(b, cv), bs = multiply(b, sv);
      for (int i = 0; i < 3; ++i) {
        const di::Interval scaled = multiply(ring, radial[i]);
        if (want(0)) out[0][i] = add(o[i], add(scaled, multiply(bs, axis[i])));
        if (want(1)) out[1][i] = multiply(ring, turned[i]);
        if (want(2)) out[2][i] = subtract(multiply(bc, axis[i]), multiply(bs, radial[i]));
        out[3][i] = negate(scaled);
        out[4][i] = negate(multiply(bs, turned[i]));
        out[5][i] = negate(add(multiply(bc, radial[i]), multiply(bs, axis[i])));
      }
      return;
    }
    case ClosedForm::kLineRevolution: {
      const di::Interval *p0 = d, *p1 = d + 3, *r0 = d + 6, *r1 = d + 9, *c0 = d + 12,
                         *c1 = d + 15;
      for (int i = 0; i < 3; ++i) {
        const di::Interval radius = add(r0[i], multiply(v, r1[i]));
        const di::Interval cross = add(c0[i], multiply(v, c1[i]));
        const di::Interval rotated = add(multiply(radius, c), multiply(cross, s));
        if (want(0)) out[0][i] = add(p0[i], add(multiply(v, p1[i]), rotated));
        if (want(1)) out[1][i] = subtract(multiply(cross, c), multiply(radius, s));
        if (want(2)) out[2][i] = add(p1[i], add(multiply(r1[i], c), multiply(c1[i], s)));
        out[3][i] = negate(rotated);
        out[4][i] = subtract(multiply(c1[i], c), multiply(r1[i], s));
        out[5][i] = zero;
      }
      return;
    }
    case ClosedForm::kNone:
      return;
  }
}

// Tensor Bernstein polynomial of degree (m, n) on [0, 1]^2; m < 0 is zero.
struct BernsteinNet {
  int m = -1, n = -1;
  std::vector<di::Interval> c;  // c[i * (n + 1) + j]
  bool zero() const { return m < 0; }
  di::Interval& at(int i, int j) { return c[static_cast<std::size_t>(i * (n + 1) + j)]; }
  const di::Interval& at(int i, int j) const {
    return c[static_cast<std::size_t>(i * (n + 1) + j)];
  }
};

BernsteinNet make_net(int m, int n) {
  BernsteinNet net;
  net.m = m;
  net.n = n;
  net.c.assign(static_cast<std::size_t>((m + 1) * (n + 1)), di::point(0.0));
  return net;
}

double binomial(int n, int k) {
  double value = 1.0;
  for (int i = 1; i <= k; ++i) value = value * (n - k + i) / i;  // exact: integers < 2^53
  return value;
}

di::Interval ratio(double numerator, double denominator) {
  const double q = numerator / denominator;
  return {next_down(q), next_up(q)};
}

// d/ds (axis 0) or d/dt (axis 1) on [0, 1]^2: degree times differences.
BernsteinNet net_derivative(const BernsteinNet& a, int axis) {
  const int degree = axis == 0 ? a.m : a.n;
  if (a.zero() || degree == 0) return {};
  BernsteinNet out = make_net(axis == 0 ? a.m - 1 : a.m, axis == 0 ? a.n : a.n - 1);
  for (int i = 0; i <= out.m; ++i)
    for (int j = 0; j <= out.n; ++j)
      out.at(i, j) = di::multiply(di::point(degree),
                                  di::subtract(a.at(axis == 0 ? i + 1 : i, axis == 0 ? j : j + 1),
                                               a.at(i, j)));
  return out;
}

// Bernstein product: B^a_i B^b_k = C(a,i) C(b,k) / C(a+b,i+k) B^(a+b)_(i+k).
BernsteinNet net_product(const BernsteinNet& a, const BernsteinNet& b) {
  if (a.zero() || b.zero()) return {};
  BernsteinNet out = make_net(a.m + b.m, a.n + b.n);
  for (int i1 = 0; i1 <= a.m; ++i1)
    for (int j1 = 0; j1 <= a.n; ++j1)
      for (int i2 = 0; i2 <= b.m; ++i2)
        for (int j2 = 0; j2 <= b.n; ++j2) {
          const di::Interval factor =
              ratio(binomial(a.m, i1) * binomial(b.m, i2) * binomial(a.n, j1) * binomial(b.n, j2),
                    binomial(out.m, i1 + i2) * binomial(out.n, j1 + j2));
          di::Interval& target = out.at(i1 + i2, j1 + j2);
          target = di::add(target, di::multiply(factor, di::multiply(a.at(i1, j1), b.at(i2, j2))));
        }
  return out;
}

// Degree elevation along one axis by one: b'_k = k/(d+1) b_(k-1) + (1 - k/(d+1)) b_k.
BernsteinNet net_elevate(const BernsteinNet& a, int axis) {
  BernsteinNet out = make_net(axis == 0 ? a.m + 1 : a.m, axis == 0 ? a.n : a.n + 1);
  const int degree = (axis == 0 ? a.m : a.n) + 1;
  for (int i = 0; i <= out.m; ++i)
    for (int j = 0; j <= out.n; ++j) {
      const int k = axis == 0 ? i : j;
      di::Interval value = di::point(0.0);
      if (k > 0)
        value = di::multiply(ratio(k, degree), a.at(axis == 0 ? i - 1 : i, axis == 0 ? j : j - 1));
      if (k < degree)
        value = di::add(value, di::multiply(ratio(degree - k, degree), a.at(i, j)));
      out.at(i, j) = value;
    }
  return out;
}

struct NetTerm { double factor; BernsteinNet net; };

BernsteinNet net_combination(const std::vector<NetTerm>& terms) {
  int m = -1, n = -1;
  for (const NetTerm& term : terms)
    if (!term.net.zero()) m = std::max(m, term.net.m), n = std::max(n, term.net.n);
  if (m < 0) return {};
  BernsteinNet out = make_net(m, n);
  for (const NetTerm& term : terms) {
    if (term.net.zero()) continue;
    BernsteinNet net = term.net;
    while (net.m < m) net = net_elevate(net, 0);
    while (net.n < n) net = net_elevate(net, 1);
    for (std::size_t k = 0; k < out.c.size(); ++k)
      out.c[k] = di::add(out.c[k], di::multiply(di::point(term.factor), net.c[k]));
  }
  return out;
}

// De Casteljau restriction of degree-d coefficients (stride apart) from
// [0, 1] to a superset [a, a + (1 - a) z'] of [a, z], 0 <= a <= z <= 1:
// the right part at a, then the left part at z' >= (z - a) / (1 - a).
void restrict_axis(di::Interval* b, int degree, int stride, double a, double z) {
  const auto at = [&](int k) -> di::Interval& { return b[k * stride]; };
  if (a > 0.0)
    for (int r = 1; r <= degree; ++r)
      for (int k = 0; k <= degree - r; ++k)
        at(k) = di::add(at(k), di::multiply(di::point(a), di::subtract(at(k + 1), at(k))));
  if (a >= 1.0) return;
  const double relative =
      std::min(1.0, std::max(0.0, next_up(next_up(z - a) / next_down(1.0 - a))));
  if (relative >= 1.0) return;
  for (int r = 1; r <= degree; ++r)
    for (int k = degree; k >= r; --k)
      at(k) = di::add(at(k - 1), di::multiply(di::point(relative), di::subtract(at(k), at(k - 1))));
}

// Hull of the net restricted to [s0, s1] x [t0, t1] (convex hull property).
di::Interval restricted_hull(const BernsteinNet& net, double s0, double s1, double t0,
                             double t1, std::vector<di::Interval>& scratch) {
  scratch.assign(net.c.begin(), net.c.end());
  for (int i = 0; i <= net.m; ++i)
    restrict_axis(scratch.data() + i * (net.n + 1), net.n, 1, t0, t1);
  for (int j = 0; j <= net.n; ++j) restrict_axis(scratch.data() + j, net.m, net.n + 1, s0, s1);
  di::Interval hull = scratch[0];
  for (const di::Interval& value : scratch)
    hull = {std::min(hull.lower, value.lower), std::max(hull.upper, value.upper)};
  return hull;
}

// One nonempty knot span of a rational B-spline surface. With homogeneous
// P = (w x, w y, w z) and weight w on the span's local square,
//   S_ss = (P_ss w^2 - P w_ss w - 2 P_s w w_s + 2 P w_s^2) / w^3,
//   S_st = (P_st w^2 - P_s w_t w - P_t w_s w - P w_st w + 2 P w_s w_t) / w^3,
//   S_tt likewise; the numerators are exact Bernstein nets (products and
// degree elevation), enclosed outward from the coefficient enclosures. As in
// Python's restricted homogeneous nets (_bspline_extraction.py 1723-1814),
// a leaf restricts each net to its local box: the hull bounds the numerator
// and the restricted weight net bounds w from below, keeping the quotient's
// correlations that interval Cox--de Boor arithmetic loses.
struct SplineSpan {
  di::Interval u0, u1, v0, v1, du, dv;
  double weight_lower = 0.0;
  BernsteinNet weight;
  BernsteinNet numerator[3][3];  // [uu, uv, vv][component]
};

struct SplineNets {
  bool ready = false;
  int p = 0, q = 0, nu = 0, nv = 0;
  const double *knot_lower = nullptr, *knot_upper = nullptr;
  double domain[4] = {};
  std::vector<SplineSpan> spans;  // (us - p) * (nv - q) + (vs - q)
  std::vector<int8_t> valid;      // 0 no nets, 1 nets, 2 empty span
  std::vector<di::Interval> placements;  // 12 per affine level, outermost first
  std::vector<di::Interval> scratch;
};

constexpr int kMaxNetDegree = 5;

// Blossom of the degree-d B-spline with coefficients at span `span` (local
// controls[0..d] = global span - d .. span), at the argument list.
bool spline_blossom(const di::Interval* knot, int span, int degree, const di::Interval* control,
                    const di::Interval* arguments, di::Interval& result) {
  di::Interval d[kMaxNetDegree + 1];
  for (int k = 0; k <= degree; ++k) d[k] = control[k];
  for (int r = 1; r <= degree; ++r)
    for (int k = degree; k >= r; --k) {
      const int index = span - degree + k;
      di::Interval inverse;
      if (!reciprocal(di::subtract(knot[index + degree + 1 - r], knot[index]), inverse))
        return false;
      const di::Interval alpha = di::multiply(di::subtract(arguments[r - 1], knot[index]), inverse);
      d[k] = di::add(d[k - 1], di::multiply(alpha, di::subtract(d[k], d[k - 1])));
    }
  result = d[degree];
  return di::finite(result);
}

bool prepare_spline_nets(const CadGeometry& geometry, int32_t index,
                         const IntervalEvaluationWorkspace& workspace, SplineNets& nets) {
  nets = SplineNets{};
  int32_t current = index;
  std::vector<di::Interval> placements;
  while (geometry.node(current).kind == PHX_MC_CAD_NODE_AFFINE &&
         geometry.node(current).integers[1] == 3 && placements.size() < 48) {
    const int64_t offset = geometry.model().node_real_offsets[current];
    for (int i = 0; i < 12; ++i)
      placements.push_back({workspace.coefficient_lower[offset + i],
                            workspace.coefficient_upper[offset + i]});
    current = static_cast<int32_t>(geometry.node(current).integers[0]);
  }
  const Node& node = geometry.node(current);
  if (node.kind != PHX_MC_CAD_NODE_BSPLINE_SURFACE) return false;
  const int p = static_cast<int>(node.integers[0]), q = static_cast<int>(node.integers[1]);
  const int nu = static_cast<int>(node.integers[2]), nv = static_cast<int>(node.integers[3]);
  if (p > kMaxNetDegree || q > kMaxNetDegree || p < 1 || q < 1) return false;
  const int64_t offset = geometry.model().node_real_offsets[current];
  const double* lower = workspace.coefficient_lower + offset;
  const double* upper = workspace.coefficient_upper + offset;
  const int uk = nu + p + 1, vk = nv + q + 1;
  const int64_t controls = static_cast<int64_t>(nu) * nv;
  const int64_t points = uk + vk, weights = points + 3 * controls;
  for (int64_t i = 0; i < weights + controls; ++i)
    if (!di::finite({lower[i], upper[i]})) return false;
  const auto at = [&](int64_t i) { return di::Interval{lower[i], upper[i]}; };
  std::vector<di::Interval> knots(static_cast<std::size_t>(uk + vk));
  for (int i = 0; i < uk + vk; ++i) knots[static_cast<std::size_t>(i)] = at(i);
  const di::Interval* u_knots = knots.data();
  const di::Interval* v_knots = knots.data() + uk;
  nets.p = p, nets.q = q, nets.nu = nu, nets.nv = nv;
  nets.knot_lower = lower;
  nets.knot_upper = upper;
  nets.domain[0] = upper[p], nets.domain[1] = lower[nu];
  nets.domain[2] = upper[uk + q], nets.domain[3] = lower[uk + nv];
  nets.placements = std::move(placements);
  nets.spans.resize(static_cast<std::size_t>((nu - p) * (nv - q)));
  nets.valid.assign(nets.spans.size(), 0);
  for (int us = p; us < nu; ++us)
    for (int vs = q; vs < nv; ++vs) {
      SplineSpan& span = nets.spans[static_cast<std::size_t>((us - p) * (nv - q) + vs - q)];
      span.u0 = u_knots[us], span.u1 = u_knots[us + 1];
      span.v0 = v_knots[vs], span.v1 = v_knots[vs + 1];
      span.du = di::subtract(span.u1, span.u0);
      span.dv = di::subtract(span.v1, span.v0);
      // An exactly empty span (equal point knots) holds no parameter; others
      // need a provably positive length.
      const auto point_equal = [&](int64_t first, int64_t second) {
        return lower[first] == upper[first] && lower[second] == upper[second] &&
               lower[first] == lower[second];
      };
      if (point_equal(us, us + 1) || point_equal(uk + vs, uk + vs + 1)) {
        nets.valid[static_cast<std::size_t>(&span - nets.spans.data())] = 2;
        continue;
      }
      if (!(span.du.lower > 0.0) || !(span.dv.lower > 0.0)) return false;
      // Homogeneous Bernstein net by blossoming: coefficient (k, l) is the
      // blossom at (u0^(p-k), u1^k; v0^(q-l), v1^l).
      BernsteinNet homogeneous[4];
      for (auto& net : homogeneous) net = make_net(p, q);
      di::Interval v_arguments[kMaxNetDegree], u_arguments[kMaxNetDegree];
      for (int l = 0; l <= q; ++l) {
        for (int r = 0; r < q; ++r) v_arguments[r] = r < q - l ? span.v0 : span.v1;
        di::Interval along_v[kMaxNetDegree + 1][4];
        for (int a = 0; a <= p; ++a) {
          const int row = us - p + a;
          di::Interval column[4][kMaxNetDegree + 1];
          for (int b = 0; b <= q; ++b) {
            const int64_t control = static_cast<int64_t>(row) * nv + vs - q + b;
            const di::Interval weight = at(weights + control);
            for (int c = 0; c < 3; ++c)
              column[c][b] = di::multiply(weight, at(points + 3 * control + c));
            column[3][b] = weight;
          }
          for (int c = 0; c < 4; ++c)
            if (!spline_blossom(v_knots, vs, q, column[c], v_arguments, along_v[a][c]))
              return false;
        }
        for (int k = 0; k <= p; ++k) {
          for (int r = 0; r < p; ++r) u_arguments[r] = r < p - k ? span.u0 : span.u1;
          for (int c = 0; c < 4; ++c) {
            di::Interval row[kMaxNetDegree + 1];
            for (int a = 0; a <= p; ++a) row[a] = along_v[a][c];
            if (!spline_blossom(u_knots, us, p, row, u_arguments, homogeneous[c].at(k, l)))
              return false;
          }
        }
      }
      const BernsteinNet& w = homogeneous[3];
      span.weight = w;
      span.weight_lower = std::numeric_limits<double>::infinity();
      for (const di::Interval& value : w.c) span.weight_lower = std::min(span.weight_lower, value.lower);
      if (!(span.weight_lower > 0.0)) return false;
      const BernsteinNet ws = net_derivative(w, 0), wt = net_derivative(w, 1);
      const BernsteinNet wss = net_derivative(ws, 0), wst = net_derivative(ws, 1),
                         wtt = net_derivative(wt, 1);
      const BernsteinNet ww = net_product(w, w);
      for (int c = 0; c < 3; ++c) {
        const BernsteinNet& P = homogeneous[c];
        const BernsteinNet Ps = net_derivative(P, 0), Pt = net_derivative(P, 1);
        const BernsteinNet Pss = net_derivative(Ps, 0), Pst = net_derivative(Ps, 1),
                           Ptt = net_derivative(Pt, 1);
        span.numerator[0][c] = net_combination(
            {{1.0, net_product(Pss, ww)}, {-1.0, net_product(net_product(P, wss), w)},
             {-2.0, net_product(net_product(Ps, w), ws)}, {2.0, net_product(net_product(P, ws), ws)}});
        span.numerator[1][c] = net_combination(
            {{1.0, net_product(Pst, ww)}, {-1.0, net_product(net_product(Ps, wt), w)},
             {-1.0, net_product(net_product(Pt, ws), w)}, {-1.0, net_product(net_product(P, wst), w)},
             {2.0, net_product(net_product(P, ws), wt)}});
        span.numerator[2][c] = net_combination(
            {{1.0, net_product(Ptt, ww)}, {-1.0, net_product(net_product(P, wtt), w)},
             {-2.0, net_product(net_product(Pt, w), wt)}, {2.0, net_product(net_product(P, wt), wt)}});
      }
      nets.valid[static_cast<std::size_t>(&span - nets.spans.data())] = 1;
    }
  nets.ready = true;
  return true;
}

// Encloses S_uu, S_uv, S_vv ([term][component]) over the box from the
// restricted numerator nets of every span the box meets; false when the box
// leaves the spline's domain or meets a span without nets.
bool spline_net_hessian(SplineNets& nets, const double* lower, const double* upper,
                        di::Interval out[3][3]) {
  if (!nets.ready || lower[0] < nets.domain[0] || upper[0] > nets.domain[1] ||
      lower[1] < nets.domain[2] || upper[1] > nets.domain[3])
    return false;
  const int uk = nets.nu + nets.p + 1;
  bool found = false;
  for (int us = nets.p; us < nets.nu; ++us) {
    if (upper[0] < nets.knot_lower[us] || lower[0] > nets.knot_upper[us + 1]) continue;
    for (int vs = nets.q; vs < nets.nv; ++vs) {
      if (upper[1] < nets.knot_lower[uk + vs] || lower[1] > nets.knot_upper[uk + vs + 1])
        continue;
      const auto slot = static_cast<std::size_t>((us - nets.p) * (nets.nv - nets.q) + vs - nets.q);
      const SplineSpan& span = nets.spans[slot];
      if (nets.valid[slot] == 2) continue;
      if (nets.valid[slot] != 1) return false;
      di::Interval inverse_u, inverse_v;
      if (!reciprocal(span.du, inverse_u) || !reciprocal(span.dv, inverse_v)) return false;
      // Local box of the leaf on this span, outward over the knot enclosures.
      const double s0 = std::max(0.0, di::multiply(di::subtract(di::point(lower[0]), span.u0), inverse_u).lower);
      const double s1 = std::min(1.0, di::multiply(di::subtract(di::point(upper[0]), span.u0), inverse_u).upper);
      const double t0 = std::max(0.0, di::multiply(di::subtract(di::point(lower[1]), span.v0), inverse_v).lower);
      const double t1 = std::min(1.0, di::multiply(di::subtract(di::point(upper[1]), span.v0), inverse_v).upper);
      if (s0 > s1 || t0 > t1) continue;
      const di::Interval weight = restricted_hull(span.weight, s0, s1, t0, t1, nets.scratch);
      const di::Interval w{std::max(weight.lower, span.weight_lower), weight.upper};
      di::Interval inverse_cube;
      if (!reciprocal(di::multiply(w, di::multiply(w, w)), inverse_cube)) return false;
      const di::Interval scale[3] = {di::multiply(inverse_u, inverse_u),
                                     di::multiply(inverse_u, inverse_v),
                                     di::multiply(inverse_v, inverse_v)};
      for (int term = 0; term < 3; ++term)
        for (int c = 0; c < 3; ++c) {
          const BernsteinNet& numerator = span.numerator[term][c];
          di::Interval value = di::point(0.0);
          if (!numerator.zero())
            value = di::multiply(di::multiply(restricted_hull(numerator, s0, s1, t0, t1, nets.scratch),
                                              inverse_cube),
                                 scale[term]);
          out[term][c] = found ? di::Interval{std::min(out[term][c].lower, value.lower),
                                              std::max(out[term][c].upper, value.upper)}
                               : value;
        }
      found = true;
    }
  }
  if (!found) return false;
  // Placements were collected outermost first; the Hessian of M S + t is M
  // times S's, applied innermost first.
  for (std::size_t level = nets.placements.size() / 12; level-- > 0;) {
    const di::Interval* matrix = nets.placements.data() + 12 * level;
    for (int term = 0; term < 3; ++term) {
      di::Interval mapped[3];
      for (int r = 0; r < 3; ++r) {
        mapped[r] = di::point(0.0);
        for (int c = 0; c < 3; ++c)
          mapped[r] = di::add(mapped[r], di::multiply(matrix[3 * r + c], out[term][c]));
      }
      for (int r = 0; r < 3; ++r) out[term][r] = mapped[r];
    }
  }
  for (int term = 0; term < 3; ++term)
    for (int c = 0; c < 3; ++c)
      if (!di::finite(out[term][c])) return false;
  return true;
}

// Per-face evaluation state prepared once: the interval closed form or the
// rational spline nets of the face's node, and the trigonometric anchors.
struct FaceKernel {
  bool prepared = false;
  int32_t node = -1;
  bool closed = false;
  ClosedFrame frame;
  SplineNets nets;
  AnchoredTrigonometry trig;
};

void prepare_face_kernel(const CadGeometry& geometry, int32_t face,
                         const IntervalEvaluationWorkspace& workspace, FaceKernel& kernel) {
  kernel.prepared = true;
  kernel.node = geometry.surface_node(face);
  kernel.closed = closed_frame(geometry, kernel.node, workspace, kernel.frame);
  if (!kernel.closed) prepare_spline_nets(geometry, kernel.node, workspace, kernel.nets);
}

// 0: Hessian bound applicable, 1: use a first-order bound, 2: refuse.
// `clamped`: a spline surface, clamped outside its active knot interval with
// zero derivative across it, so the domain ends k = degree and k = count are
// C0 lines; a curve refuses parameters outside instead.
int spline_knot_regularity(const double* lower, const double* upper, int degree, int count,
                           double box_lower, double box_upper, bool clamped) {
  int result = 0;
  if (clamped)
    for (int k : {degree, count})
      if (box_lower < upper[k] && lower[k] < box_upper) result = 1;
  for (int k = degree + 1; k < count; ++k) {
    if (box_upper < lower[k] || box_lower > upper[k]) continue;
    int multiplicity = 1;
    for (int j = k - 1; j >= 0 && upper[j] >= lower[k]; --j) ++multiplicity;
    for (int j = k + 1; j < count + degree + 1 && lower[j] <= upper[k]; ++j)
      ++multiplicity;
    if (multiplicity > degree) return 2;
    if (multiplicity >= degree) result = 1;
  }
  return result;
}

// The parameter values where an intersection node changes chart: the carrier
// seams k = 1 .. charts - 1 and the terminal threshold (a fixed chart has
// none). The root is continuous there (the constructor's chart joins; not
// re-established here) but its parameterization is only C0. Appends to
// `breaks` the exact parameter values strictly inside (box_lower,
// box_upper) and returns their number, or -1 when a break may lie inside
// but is not an exact binary64 point, or more than `capacity` lie inside.
int intersection_breaks(const CadGeometry& geometry, int32_t index, double box_lower,
                        double box_upper, const IntervalEvaluationWorkspace& workspace,
                        double* breaks, int capacity) {
  const Node& node = geometry.node(index);
  const int64_t charts = node.integers[2], fixed = node.integers[4], terminal = node.integers[5];
  if (fixed >= 0) return 0;
  const bool reversed = node.integers[6] != 0;
  const int64_t offset = geometry.model().node_real_offsets[index];
  const di::Interval sum{workspace.coefficient_lower[offset], workspace.coefficient_upper[offset]};
  int count = 0;
  const auto place = [&](const di::Interval& carrier) {
    const di::Interval t = reversed ? di::subtract(sum, carrier) : carrier;
    if (!(t.upper > box_lower && t.lower < box_upper)) return true;
    if (!(t.lower == t.upper && std::isfinite(t.lower)) || count == capacity) return false;
    breaks[count++] = t.lower;
    return true;
  };
  // Carrier values the box can reach (outward).
  const di::Interval reach = reversed
      ? di::subtract(sum, di::Interval{box_lower, box_upper})
      : di::Interval{box_lower, box_upper};
  const double first = std::max(1.0, std::floor(reach.lower));
  const double last = std::min(static_cast<double>(charts - 1), std::ceil(reach.upper));
  for (double k = first; k <= last; k += 1.0)
    if (!place(di::point(k))) return -1;
  if (terminal >= 0 &&
      !place({workspace.coefficient_lower[offset + 1], workspace.coefficient_upper[offset + 1]}))
    return -1;
  return count;
}

// The hull (lo, hi: q0 q1 of the first surface, q2 q3 of the second) of the
// stored chart boxes and chart chords of every chart the parameter interval
// can select (a fixed chart alone; else the reachable carrier charts and the
// terminal chart). Every root of those charts lies in its chart box.
void intersection_carrier_hull(const CadGeometry& geometry, int32_t index, double box_lower,
                               double box_upper, const IntervalEvaluationWorkspace& workspace,
                               double* lo, double* hi) {
  const Node& node = geometry.node(index);
  const int64_t charts = node.integers[2], fixed = node.integers[4], terminal = node.integers[5];
  const int64_t offset = geometry.model().node_real_offsets[index];
  const double* lower = workspace.coefficient_lower + offset;
  const double* upper = workspace.coefficient_upper + offset;
  for (int k = 0; k < 4; ++k) {
    lo[k] = std::numeric_limits<double>::infinity();
    hi[k] = -std::numeric_limits<double>::infinity();
  }
  const auto include = [&](int64_t chart) {
    const int64_t base = 2 + 25 * chart;  // start, end, lower, upper (4 each), P (9)
    for (int k = 0; k < 4; ++k)
      for (int part = 0; part < 4; ++part) {
        lo[k] = std::min(lo[k], lower[base + 4 * part + k]);
        hi[k] = std::max(hi[k], upper[base + 4 * part + k]);
      }
  };
  if (fixed >= 0) {
    include(fixed);
    return;
  }
  const bool reversed = node.integers[6] != 0;
  const di::Interval sum{lower[0], upper[0]};
  const di::Interval reach = reversed
      ? di::subtract(sum, di::Interval{box_lower, box_upper})
      : di::Interval{box_lower, box_upper};
  const double first = std::max(0.0, std::floor(reach.lower));
  const double last = std::min(static_cast<double>(charts - 1), std::floor(reach.upper));
  for (double c = first; c <= last; c += 1.0) include(static_cast<int64_t>(c));
  if (terminal >= 0) include(terminal);
  if (!(lo[0] <= hi[0]))  // nothing reachable: the evaluation itself refuses
    for (int k = 0; k < 4; ++k) {
      lo[k] = -std::numeric_limits<double>::infinity();
      hi[k] = std::numeric_limits<double>::infinity();
    }
}

int spline_regularity(const CadGeometry& geometry, int32_t index, const double* box_lower,
                      const double* box_upper, IntervalEvaluationWorkspace& workspace) {
  const Node& node = geometry.node(index);
  const int64_t offset = geometry.model().node_real_offsets[index];
  const auto child = [&](int slot, const double* lo, const double* hi) {
    return spline_regularity(geometry, static_cast<int32_t>(node.integers[slot]),
                             lo, hi, workspace);
  };
  const double all_lower[2] = {-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()};
  const double all_upper[2] = {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
  switch (node.kind) {
    case PHX_MC_CAD_NODE_BSPLINE_CURVE:
    case PHX_MC_CAD_NODE_BSPLINE_SURFACE: {
      const bool surface = node.kind == PHX_MC_CAD_NODE_BSPLINE_SURFACE;
      const int p = static_cast<int>(node.integers[0]);
      const int nu = static_cast<int>(node.integers[surface ? 2 : 1]);
      const double* lo = workspace.coefficient_lower + offset;
      const double* hi = workspace.coefficient_upper + offset;
      int result = spline_knot_regularity(lo, hi, p, nu, box_lower[0], box_upper[0], surface);
      if (surface) {
        const int q = static_cast<int>(node.integers[1]), nv = static_cast<int>(node.integers[3]);
        const int uk = nu + p + 1;
        result = std::max(result, spline_knot_regularity(lo + uk, hi + uk, q, nv,
                                                         box_lower[1], box_upper[1], true));
      }
      return result;
    }
    case PHX_MC_CAD_NODE_AFFINE:
      return child(0, box_lower, box_upper);
    // An offset needs one more base derivative than its own jet: only template
    // children provide third-order interval jets, so others get first order.
    case PHX_MC_CAD_NODE_OFFSET_SURFACE:
    case PHX_MC_CAD_NODE_OFFSET_CURVE: {
      const auto base = static_cast<int32_t>(node.integers[0]);
      const int regularity = node.kind == PHX_MC_CAD_NODE_OFFSET_SURFACE
                                 ? child(0, box_lower, box_upper)
                                 : child(0, all_lower, all_upper);
      if (regularity != 0) return 2;
      return geometry.node(base).kind == PHX_MC_CAD_NODE_TEMPLATE ? 0 : 1;
    }
    case PHX_MC_CAD_NODE_EXTRUSION:
      return child(0, box_lower, box_upper);
    case PHX_MC_CAD_NODE_REVOLUTION:
      return child(0, box_lower + 1, box_upper + 1);
    case PHX_MC_CAD_NODE_RULED:
      return std::max(child(0, box_lower, box_upper), child(1, box_lower, box_upper));
    // The child parameter over the box: first + s (last - first), s = t or
    // 1 - t, outward.
    case PHX_MC_CAD_NODE_REPARAMETERIZED: {
      const di::Interval first{workspace.coefficient_lower[offset],
                               workspace.coefficient_upper[offset]};
      const di::Interval last{workspace.coefficient_lower[offset + 1],
                              workspace.coefficient_upper[offset + 1]};
      const di::Interval t{box_lower[0], box_upper[0]};
      const di::Interval s = node.integers[1] != 0 ? di::subtract(di::point(1.0), t) : t;
      const di::Interval mapped = di::add(first, di::multiply(s, di::subtract(last, first)));
      if (!di::finite(mapped)) return child(0, all_lower, all_upper);
      return child(0, &mapped.lower, &mapped.upper);
    }
    // The surface over the fixed coordinate and the curve interval.
    case PHX_MC_CAD_NODE_ISOPARAMETRIC: {
      const int fixed = static_cast<int>(node.integers[1]);
      double lo[2], hi[2];
      lo[fixed] = workspace.coefficient_lower[offset];
      hi[fixed] = workspace.coefficient_upper[offset];
      lo[1 - fixed] = box_lower[0];
      hi[1 - fixed] = box_upper[0];
      return child(0, lo, hi);
    }
    // The surface over the p-curve's interval image of the box (the whole
    // plane when the image is not finite).
    case PHX_MC_CAD_NODE_SURFACE_MAPPED: {
      const auto pcurve = static_cast<int32_t>(node.integers[1]);
      const int regularity = spline_regularity(geometry, pcurve, box_lower, box_upper, workspace);
      double lo[2], hi[2];
      if (geometry.interval_jets(pcurve, box_lower, box_upper, 0, lo, hi, workspace) != PHX_MC_OK ||
          !std::isfinite(lo[0]) || !std::isfinite(lo[1]) || !std::isfinite(hi[0]) ||
          !std::isfinite(hi[1]))
        return std::max(regularity, child(0, all_lower, all_upper));
      return std::max(regularity, child(0, lo, hi));
    }
    // A chart seam or the terminal threshold inside the interval is a C0
    // break of the parameterization (Python's IntersectionCurve.is_c1_on).
    // Each carrier is asked over the hull of the chart boxes the interval can
    // select: the root lies in its chart's box (the chart certificate).
    case PHX_MC_CAD_NODE_INTERSECTION: {
      double lo[4], hi[4];
      intersection_carrier_hull(geometry, index, box_lower[0], box_upper[0], workspace, lo, hi);
      const int result = std::max(child(0, lo, hi), child(1, lo + 2, hi + 2));
      if (result != 0) return result;
      return intersection_breaks(geometry, index, box_lower[0], box_upper[0], workspace, nullptr,
                                 0) != 0 ? 1 : 0;
    }
    default:
      return 0;
  }
}

// A C0 (not C1) knot line {uv[axis] = knot} of a spline surface chart.
struct KnotLine { int axis; double knot; };
constexpr int kMaxKnotLines = 8;
constexpr int kMaxPieces = 32;
constexpr int kMaxPieceVertices = 8;
// A clip point whose enclosure is wider than this fraction of its edge's
// length (an edge nearly parallel to the knot line) makes the leaf take the
// first-order bound instead.
constexpr double kClipConditioning = 0x1p-20;

// Appends the C0 knot lines strictly inside the leaf box, in chart axes.
// False when the chart's knot lines cannot be located exactly (a knot that is
// not a point, a child in another parameter, too many lines): the caller then
// takes the first-order bound.
bool c0_knot_lines(const CadGeometry& geometry, int32_t index, int curve_axis,
                   const double* box_lower, const double* box_upper,
                   IntervalEvaluationWorkspace& workspace, KnotLine* lines, int& count) {
  const Node& node = geometry.node(index);
  const auto child = [&](int slot, int axis) {
    return c0_knot_lines(geometry, static_cast<int32_t>(node.integers[slot]), axis, box_lower,
                         box_upper, workspace, lines, count);
  };
  const auto add = [&](int axis, double knot) {
    if (count == kMaxKnotLines) return false;
    lines[count++] = {axis, knot};
    return true;
  };
  const auto knots = [&](const double* lo, const double* hi, int degree, int n, int axis,
                         bool clamped) {
    // The clamped domain ends of a spline surface (see spline_knot_regularity).
    if (clamped)
      for (int k : {degree, n}) {
        if (!(box_lower[axis] < hi[k] && lo[k] < box_upper[axis])) continue;
        if (lo[k] != hi[k] || !add(axis, lo[k])) return false;
      }
    for (int k = degree + 1; k < n; ++k) {
      if (lo[k] == lo[k - 1] && hi[k] == hi[k - 1]) continue;
      if (box_upper[axis] < lo[k] || box_lower[axis] > hi[k]) continue;
      int multiplicity = 1;
      for (int j = k + 1; j < n + degree + 1 && lo[j] <= hi[k]; ++j) ++multiplicity;
      if (multiplicity < degree) continue;
      if (multiplicity > degree || lo[k] != hi[k]) return false;
      // A line on the box boundary leaves the leaf on one side.
      if (!(box_lower[axis] < lo[k] && lo[k] < box_upper[axis])) continue;
      if (!add(axis, lo[k])) return false;
    }
    return true;
  };
  const int64_t offset = geometry.model().node_real_offsets[index];
  const double* lo = workspace.coefficient_lower + offset;
  const double* hi = workspace.coefficient_upper + offset;
  switch (node.kind) {
    case PHX_MC_CAD_NODE_BSPLINE_SURFACE: {
      const int p = static_cast<int>(node.integers[0]), q = static_cast<int>(node.integers[1]);
      const int nu = static_cast<int>(node.integers[2]), nv = static_cast<int>(node.integers[3]);
      return knots(lo, hi, p, nu, 0, true) && knots(lo + nu + p + 1, hi + nu + p + 1, q, nv, 1, true);
    }
    case PHX_MC_CAD_NODE_BSPLINE_CURVE:
      return curve_axis >= 0 &&
             knots(lo, hi, static_cast<int>(node.integers[0]),
                   static_cast<int>(node.integers[1]), curve_axis, false);
    case PHX_MC_CAD_NODE_AFFINE:
      return child(0, curve_axis);
    case PHX_MC_CAD_NODE_EXTRUSION:
      return child(0, 0);
    case PHX_MC_CAD_NODE_REVOLUTION:
      return child(0, 1);
    case PHX_MC_CAD_NODE_RULED:
      return child(0, 0) && child(1, 0);
    case PHX_MC_CAD_NODE_OFFSET_SURFACE:
    case PHX_MC_CAD_NODE_OFFSET_CURVE:
    case PHX_MC_CAD_NODE_REPARAMETERIZED:
    case PHX_MC_CAD_NODE_ISOPARAMETRIC:
    case PHX_MC_CAD_NODE_SURFACE_MAPPED:
      return false;
    // Chart breaks of a curve parameter whose surfaces are smooth everywhere.
    case PHX_MC_CAD_NODE_INTERSECTION: {
      if (curve_axis < 0) return false;
      double lo[4], hi[4];
      intersection_carrier_hull(geometry, index, box_lower[curve_axis], box_upper[curve_axis],
                                workspace, lo, hi);
      for (int slot = 0; slot < 2; ++slot)
        if (spline_regularity(geometry, static_cast<int32_t>(node.integers[slot]), lo + 2 * slot,
                              hi + 2 * slot, workspace) != 0)
          return false;
      double breaks[kMaxKnotLines];
      const int found = intersection_breaks(geometry, index, box_lower[curve_axis],
                                            box_upper[curve_axis], workspace, breaks,
                                            kMaxKnotLines - count);
      if (found < 0) return false;
      for (int i = 0; i < found; ++i) lines[count++] = {curve_axis, breaks[i]};
      return true;
    }
    default:
      return true;
  }
}

// A polygon vertex: outward enclosures of its exact barycentric coordinates
// in the original triangle T and of its exact chart point.
struct ClipVertex { di::Interval weight[3]; di::Interval uv[2]; };
struct Piece { ClipVertex vertex[kMaxPieceVertices]; int size; };

// Directed quotient for a denominator of strict sign.
di::Interval divide(const di::Interval& a, const di::Interval& b) {
  const double q[4] = {a.lower / b.lower, a.lower / b.upper, a.upper / b.lower,
                       a.upper / b.upper};
  return {geometry::next_down(di::least(q[0], q[1], q[2], q[3])),
          geometry::next_up(di::greatest(q[0], q[1], q[2], q[3]))};
}

// -1 / +1: the exact point is strictly below / above the line; 0: it is on
// the line exactly; 2: undecided.
int line_side(const ClipVertex& vertex, const KnotLine& line) {
  const di::Interval& value = vertex.uv[line.axis];
  if (value.upper < line.knot) return -1;
  if (value.lower > line.knot) return 1;
  return value.lower == line.knot && value.upper == line.knot ? 0 : 2;
}

// Splits a convex polygon by one line into its two closed halves (exact
// Sutherland-Hodgman with enclosed crossing points). False: undecided side,
// an edge on the line, an ill-conditioned crossing, or capacity.
bool split_piece(const Piece& piece, const KnotLine& line, Piece& below, Piece& above) {
  below.size = above.size = 0;
  const auto push = [](Piece& target, const ClipVertex& vertex) {
    if (target.size == kMaxPieceVertices) return false;
    target.vertex[target.size++] = vertex;
    return true;
  };
  const int other = 1 - line.axis;
  for (int i = 0; i < piece.size; ++i) {
    const ClipVertex& p = piece.vertex[i];
    const ClipVertex& q = piece.vertex[(i + 1) % piece.size];
    const int sp = line_side(p, line), sq = line_side(q, line);
    if (sp == 2 || sq == 2 || (sp == 0 && sq == 0)) return false;
    if ((sp <= 0 && !push(below, p)) || (sp >= 0 && !push(above, p))) return false;
    if (sp * sq >= 0) continue;
    // c = p + s (q - p), s = (k - p_a) / (q_a - p_a); the denominator has
    // the strict sign of sq because the sides are strictly opposite.
    const auto denominator = di::subtract(q.uv[line.axis], p.uv[line.axis]);
    if (!(denominator.lower > 0.0 || denominator.upper < 0.0)) return false;
    const auto s = divide(di::subtract(di::point(line.knot), p.uv[line.axis]), denominator);
    ClipVertex c{};
    for (int w = 0; w < 3; ++w)
      c.weight[w] = di::add(p.weight[w], di::multiply(s, di::subtract(q.weight[w], p.weight[w])));
    c.uv[line.axis] = di::point(line.knot);
    c.uv[other] = di::add(p.uv[other], di::multiply(s, di::subtract(q.uv[other], p.uv[other])));
    // With point endpoints, a rounded crossing that the exact orientation
    // test puts on the segment's line is the exact crossing (its axis
    // coordinate is the knot), e.g. where an edge meets two knot lines.
    if (p.uv[0].lower == p.uv[0].upper && p.uv[1].lower == p.uv[1].upper &&
        q.uv[0].lower == q.uv[0].upper && q.uv[1].lower == q.uv[1].upper) {
      const double a[2] = {p.uv[0].lower, p.uv[1].lower}, b[2] = {q.uv[0].lower, q.uv[1].lower};
      double candidate[2];
      candidate[line.axis] = line.knot;
      candidate[other] = a[other] + (line.knot - a[line.axis]) * (b[other] - a[other]) /
                                        (b[line.axis] - a[line.axis]);
      if (std::isfinite(candidate[other]) && phx::mc::orient2d(a, b, candidate) == 0)
        c.uv[other] = di::point(candidate[other]);
    }
    const double length =
        std::hypot(0.5 * (q.uv[0].lower + q.uv[0].upper) - 0.5 * (p.uv[0].lower + p.uv[0].upper),
                   0.5 * (q.uv[1].lower + q.uv[1].upper) - 0.5 * (p.uv[1].lower + p.uv[1].upper));
    if (!di::finite(c.uv[other]) ||
        !(c.uv[other].upper - c.uv[other].lower <= kClipConditioning * length))
      return false;
    if (!push(below, c) || !push(above, c)) return false;
  }
  return below.size >= 3 && above.size >= 3;
}

// The interpolation remainder of a triangle t with vertices x_i in a box,
// for one component with |H_uu|, |H_uv|, |H_vv| <= a, b, c over the box:
// S(x) - I_t S(x) = -1/2 sum_i w_i (x_i - x)^T H(xi_i) (x_i - x), and
// 2 |d_u d_v| <= s d_u^2 + d_v^2 / s (s > 0) dominates each quadratic form by
// the diagonal D = diag(a + b s, c + b / s). For any centre m,
// sum_i w_i |x_i - x|_D^2 = sum_i w_i |x_i - m|_D^2 - |x - m|_D^2
// <= max_i |x_i - m|_D^2 <= D_u e_u^2 + D_v e_v^2, with e the box's largest
// offsets from m; s = e_v / e_u turns this into a e_u^2 + 2 b e_u e_v + c e_v^2.
// So the box extents of the Taylor remainder may be replaced by the offsets
// from the box centre: one quarter of the remainder. Returns e for one axis,
// rounded up (any binary64 centre is sound).
di::Interval half_extent(double lower, double upper) {
  const double centre = lower + 0.5 * (upper - lower);
  return di::point(std::max(di::subtract(di::point(centre), di::point(lower)).upper,
                            di::subtract(di::point(upper), di::point(centre)).upper));
}

// Piecewise rule at C0 knot lines (Python's _spline_piecewise_interpolation_bound).
// The exact leaf polygon is split by every C0 line strictly inside its box;
// clipping a convex polygon by the two closed half-planes of a line gives two
// convex polygons whose union is the polygon, so the exact pieces cover the
// leaf, and each lies in one closed knot cell where S is one smooth span
// piece. A fan triangle t of a piece has exact vertices inside the stored
// enclosures; on t, S - I_T S = (S - I_t S) + (I_t S - I_T S). The first term
// is bounded by the Hessian over the hull of t's vertex enclosures (it
// contains t; any span polynomial the hull touches is enclosed too); the
// second is affine and maximal at an exact vertex x of t, where it is
// |S(x) - sum_i w_i(x) X_i| evaluated over the enclosures of x and w(x).
// Returns false to request the first-order fallback.
bool clipped_leaf_bound(const CadGeometry& geometry, int32_t node, const Leaf& leaf,
                        const double* vertices, const double* uv, const double* box_lower,
                        const double* box_upper, IntervalEvaluationWorkspace& workspace,
                        int32_t& status, double& upper) {
  KnotLine lines[kMaxKnotLines];
  int line_count = 0;
  if (!c0_knot_lines(geometry, node, -1, box_lower, box_upper, workspace, lines, line_count))
    return false;
  Piece pieces[kMaxPieces];
  int piece_count = 1;
  pieces[0].size = 3;
  for (int i = 0; i < 3; ++i) {
    ClipVertex& vertex = pieces[0].vertex[i];
    for (int w = 0; w < 3; ++w) vertex.weight[w] = di::point(leaf.vertices[i].weight[w]);
    exact_chart_point(leaf.vertices[i], uv, vertex.uv);
  }
  for (int l = 0; l < line_count; ++l) {
    const int existing = piece_count;
    for (int i = 0; i < existing; ++i) {
      int sides = 0;
      for (int j = 0; j < pieces[i].size; ++j) {
        const int side = line_side(pieces[i].vertex[j], lines[l]);
        if (side == 2) return false;
        sides |= side < 0 ? 1 : side > 0 ? 2 : 0;
      }
      if (sides != 3) continue;
      if (piece_count == kMaxPieces) return false;
      Piece below, above;
      if (!split_piece(pieces[i], lines[l], below, above)) return false;
      pieces[i] = below;
      pieces[piece_count++] = above;
    }
  }
  const auto vertex_error = [&](const ClipVertex& vertex, double& error) {
    const double lo[2] = {vertex.uv[0].lower, vertex.uv[1].lower};
    const double hi[2] = {vertex.uv[0].upper, vertex.uv[1].upper};
    double surface_lower[3], surface_upper[3];
    status = geometry.interval_jets(node, lo, hi, 0, surface_lower, surface_upper, workspace);
    if (status != PHX_MC_OK) return false;
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      auto linear = di::point(0.0);
      for (int i = 0; i < 3; ++i)
        linear = di::add(linear, di::multiply(vertex.weight[i], di::point(vertices[3 * i + axis])));
      const auto difference = di::subtract({surface_lower[axis], surface_upper[axis]}, linear);
      if (!di::finite(difference)) { status = PHX_MC_UNRESOLVED; return false; }
      magnitude[axis] = di::magnitude(difference);
    }
    error = phx::mc::geometry::norm_upper(magnitude);
    return true;
  };
  double maximum = 0.0;
  status = PHX_MC_OK;
  for (int p = 0; p < piece_count; ++p) {
    const Piece& piece = pieces[p];
    double errors[kMaxPieceVertices];
    for (int j = 0; j < piece.size; ++j)
      if (!vertex_error(piece.vertex[j], errors[j])) return true;
    for (int j = 1; j + 1 < piece.size; ++j) {
      const ClipVertex* fan[3] = {&piece.vertex[0], &piece.vertex[j], &piece.vertex[j + 1]};
      double lo[2], hi[2];
      for (int axis = 0; axis < 2; ++axis) {
        lo[axis] = std::min({fan[0]->uv[axis].lower, fan[1]->uv[axis].lower,
                             fan[2]->uv[axis].lower});
        hi[axis] = std::max({fan[0]->uv[axis].upper, fan[1]->uv[axis].upper,
                             fan[2]->uv[axis].upper});
      }
      double jet_lower[18], jet_upper[18];
      status = geometry.interval_jets(node, lo, hi, 2, jet_lower, jet_upper, workspace);
      if (status != PHX_MC_OK) return true;
      const auto du = half_extent(lo[0], hi[0]);
      const auto dv = half_extent(lo[1], hi[1]);
      double magnitude[3];
      for (int axis = 0; axis < 3; ++axis) {
        const auto huu = di::point(di::magnitude({jet_lower[9 + axis], jet_upper[9 + axis]}));
        const auto huv = di::point(di::magnitude({jet_lower[12 + axis], jet_upper[12 + axis]}));
        const auto hvv = di::point(di::magnitude({jet_lower[15 + axis], jet_upper[15 + axis]}));
        auto sum = di::multiply(huu, di::multiply(du, du));
        const auto mixed = di::multiply(huv, di::multiply(du, dv));
        sum = di::add(sum, di::add(mixed, mixed));
        sum = di::add(sum, di::multiply(hvv, di::multiply(dv, dv)));
        magnitude[axis] = di::multiply(di::point(0.5), sum).upper;
      }
      const double vertex_term = std::max({errors[0], errors[j], errors[j + 1]});
      maximum = std::max(maximum, di::add(di::point(vertex_term),
                                          di::point(phx::mc::geometry::norm_upper(magnitude)))
                                      .upper);
    }
  }
  upper = maximum;
  if (!std::isfinite(upper)) status = PHX_MC_UNRESOLVED;
  return true;
}

// For each subdivision leaf L of the original triangle T,
// S - I_T = (S - I_L) + (I_L - I_T). On a C1 piece, the first term
// uses the leaf Hessian. A leaf meeting a continuous spline knot with a
// possible tangent jump takes the piecewise rule of clipped_leaf_bound; when
// that cannot locate or clip its lines soundly, use
// |S(x)-I_L S(x)| <= sum_i lambda_i |S(x)-S(v_i)| and bound each difference
// by the interval Jacobian over the leaf box.
// The second term is affine and maximized at a leaf vertex, including the
// mismatch against the stored original triangle. A possible discontinuity
// is refused: neither the Hessian nor the Jacobian implies a Lipschitz bound.
// Closed-form faces evaluate their interval formulas; rational spline faces
// intersect the restricted numerator-net Hessian with the generic one.

// |S(x) - sum_i w_i X_i| at a leaf vertex x, over its chart enclosure.
int32_t vertex_error(const CadGeometry& geometry, FaceKernel& kernel, const Vertex& vertex,
                     const di::Interval* point, const double* vertices,
                     IntervalEvaluationWorkspace& workspace, double& error) {
  const double lo[2] = {point[0].lower, point[1].lower};
  const double hi[2] = {point[0].upper, point[1].upper};
  di::Interval surface[3];
  if (kernel.closed) {
    di::Interval jets[6][3];
    closed_jets(kernel.frame, kernel.trig, lo, hi, 0, 0, jets);
    for (int axis = 0; axis < 3; ++axis) surface[axis] = jets[0][axis];
  } else {
    double surface_lower[3], surface_upper[3];
    const int32_t status = geometry.interval_jets(kernel.node, lo, hi, 0, surface_lower,
                                                   surface_upper, workspace);
    if (status != PHX_MC_OK) return status;
    for (int axis = 0; axis < 3; ++axis) surface[axis] = {surface_lower[axis], surface_upper[axis]};
  }
  double magnitude[3];
  for (int axis = 0; axis < 3; ++axis) {
    auto linear = di::point(0.0);
    for (int i = 0; i < 3; ++i) {
      const double weight = vertex.weight[i];
      if (weight == 0.0) continue;
      const auto term = weight == 1.0 ? di::point(vertices[3 * i + axis])
                                      : di::multiply(di::point(weight),
                                                     di::point(vertices[3 * i + axis]));
      linear = di::add(linear, term);
    }
    const auto difference = di::subtract(surface[axis], linear);
    if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
    magnitude[axis] = di::magnitude(difference);
  }
  error = phx::mc::geometry::norm_upper(magnitude);
  return std::isfinite(error) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// corner_error: the vertex error at the original corners when known, else
// null. A rational spline leaf whose net bound meets `tolerance` skips the
// generic Hessian; otherwise the two enclosures are intersected.
int32_t leaf_bound(const CadGeometry& geometry, FaceKernel& kernel, const Leaf& leaf,
                   const double* vertices, const double* uv, const double* corner_error,
                   double tolerance, IntervalEvaluationWorkspace& workspace,
                   double& vertex_term, double& upper) {
  const int32_t node = kernel.node;
  double box_lower[2] = {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()};
  double box_upper[2] = {-std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()};
  double mismatch = 0.0;
  for (const Vertex& vertex : leaf.vertices) {
    di::Interval point[2];
    exact_chart_point(vertex, uv, point);
    for (int axis = 0; axis < 2; ++axis) {
      box_lower[axis] = std::min(box_lower[axis], point[axis].lower);
      box_upper[axis] = std::max(box_upper[axis], point[axis].upper);
    }
    int corner = -1;
    for (int i = 0; i < 3; ++i)
      if (vertex.weight[i] == 1.0) corner = i;
    double error = 0.0;
    if (corner >= 0 && corner_error != nullptr) {
      error = corner_error[corner];
    } else {
      const int32_t status = vertex_error(geometry, kernel, vertex, point, vertices, workspace,
                                          error);
      if (status != PHX_MC_OK) return status;
    }
    mismatch = std::max(mismatch, error);
  }
  vertex_term = mismatch;
  const auto du = di::subtract(di::point(box_upper[0]), di::point(box_lower[0]));
  const auto dv = di::subtract(di::point(box_upper[1]), di::point(box_lower[1]));
  // Taylor remainder from |H| magnitudes per term (uu, uv, vv) and component,
  // with the offsets from the box centre (half_extent).
  const auto eu = half_extent(box_lower[0], box_upper[0]);
  const auto ev = half_extent(box_lower[1], box_upper[1]);
  const auto second_order = [&](const double (&hessian)[3][3]) {
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      const auto huu = di::point(hessian[0][axis]);
      const auto huv = di::point(hessian[1][axis]);
      const auto hvv = di::point(hessian[2][axis]);
      auto sum = di::multiply(huu, di::multiply(eu, eu));
      const auto mixed = di::multiply(huv, di::multiply(eu, ev));
      sum = di::add(sum, di::add(mixed, mixed));
      sum = di::add(sum, di::multiply(hvv, di::multiply(ev, ev)));
      magnitude[axis] = di::multiply(di::point(0.5), sum).upper;
    }
    return di::add(di::point(mismatch),
                   di::point(phx::mc::geometry::norm_upper(magnitude))).upper;
  };
  double hessian[3][3];
  int regularity = 0;
  if (kernel.closed) {
    di::Interval jets[6][3];
    closed_jets(kernel.frame, kernel.trig, box_lower, box_upper, 3, 5, jets);
    for (int term = 0; term < 3; ++term)
      for (int axis = 0; axis < 3; ++axis) {
        if (!di::finite(jets[3 + term][axis])) return PHX_MC_UNRESOLVED;
        hessian[term][axis] = di::magnitude(jets[3 + term][axis]);
      }
  } else {
    regularity = spline_regularity(geometry, node, box_lower, box_upper, workspace);
    if (regularity == 2) return PHX_MC_UNRESOLVED;
    if (regularity == 1) {
      int32_t status = PHX_MC_OK;
      if (clipped_leaf_bound(geometry, node, leaf, vertices, uv, box_lower, box_upper,
                             workspace, status, upper))
        return status;
    }
    di::Interval nets[3][3];
    const bool netted = regularity == 0 &&
                        spline_net_hessian(kernel.nets, box_lower, box_upper, nets);
    if (netted) {
      for (int term = 0; term < 3; ++term)
        for (int axis = 0; axis < 3; ++axis) hessian[term][axis] = di::magnitude(nets[term][axis]);
      upper = second_order(hessian);
      if (upper <= tolerance) return PHX_MC_OK;
    }
    double lower[18], higher[18];
    const int32_t status = geometry.interval_jets(node, box_lower, box_upper,
                                                   regularity == 1 ? 1 : 2,
                                                   lower, higher, workspace);
    if (status != PHX_MC_OK) return status;
    if (regularity == 1) {
      double magnitude[3];
      for (int axis = 0; axis < 3; ++axis) {
        const auto su = di::point(di::magnitude({lower[3 + axis], higher[3 + axis]}));
        const auto sv = di::point(di::magnitude({lower[6 + axis], higher[6 + axis]}));
        magnitude[axis] = di::add(di::multiply(su, du), di::multiply(sv, dv)).upper;
      }
      upper = di::add(di::point(mismatch),
                      di::point(phx::mc::geometry::norm_upper(magnitude))).upper;
      return std::isfinite(upper) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
    }
    // Both enclose the same Hessian: their intersection does too.
    for (int term = 0; term < 3; ++term)
      for (int axis = 0; axis < 3; ++axis) {
        const double generic =
            di::magnitude({lower[9 + 3 * term + axis], higher[9 + 3 * term + axis]});
        hessian[term][axis] = netted ? std::min(hessian[term][axis], generic) : generic;
      }
  }
  upper = second_order(hessian);
  return std::isfinite(upper) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// The source boundary on a face is S(p(t)). The linear p-curve interpolant
// connects the chord endpoints; its second-derivative remainder plus endpoint
// mismatch bounds the distance to the chord. The straight homotopy from this
// chord to the arc fills the enclosed lens, and remains in the padded box.
// The mean-value bound |DS| times that distance therefore maps the lens to
// the surface image of the chord. The interior triangle bound already covers
// that image in both directions, so a separate 3D curve chord sag is not
// needed. Source 3D edge/face consistency is a source-model obligation.
// The chord is cut into `pieces` equal parts in lambda = (t - t_0) /
// (t_1 - t_0) (the use's parameters[0] at lambda 0). With p and the chord
// interpolant l at the same lambda, w = p - l is, per chart axis, the
// interpolation remainder of p plus the interpolated endpoint mismatch, so
// on a piece [lambda_0, lambda_1]
//   |w_a| <= mismatch_a + M_a dt^2 / 2 max lambda (1 - lambda)   (C2 p),
//   |w_a| <= mismatch_a + M_a dt min(1, 2 max min(lambda, 1 - lambda)) (C1 p),
// with M_a the p-curve's |p_a''| (|p_a'|) over the whole interval: the lens
// is thin near the chord ends. A finer bound uses the kernel form: with
// s in [0, 1] and the chord parameter lambda in the piece,
//   C2: w = -dt^2 int G(lambda, s) p''(s) ds, |G| = min(s (1 - lambda), lambda (1 - s))
//       <= min(s (1 - lambda_0), lambda_1 (1 - s)),
//   C1: w = dt int K(lambda, s) p'(s) ds, |K| = 1 - lambda (s < lambda), lambda (s > lambda),
// so |w_a| <= mismatch_a + dt^2 (dt) sum_j M_a,j int_{s_j}^{s_j+1} |G| (|K|) bound ds,
// M_a,j the enclosure over the parameters of sub-interval j (kSubintervals
// equal parts of lambda, outward to t, clamped to the chord's interval). Per
// axis the smaller of the two bounds is used. Piece k's lens points (on the segments from
// l(lambda) to p(lambda), lambda in the piece) lie in its padded box: the box
// of l over the piece padded per axis by those bounds; caps[k] is the
// mean-value bound over that box. `boxes` holds (u lower, v lower, u upper,
// v upper) per piece. One piece is the whole chord's lens.
int32_t trim_lens_bound(const CadGeometry& geometry, int32_t face, int64_t use,
                        const double* chord_uv, const double* parameters,
                        IntervalEvaluationWorkspace& workspace, int pieces, double* caps,
                        double* boxes) {
  const double low = std::min(parameters[0], parameters[1]);
  const double high = std::max(parameters[0], parameters[1]);
  const int regularity = spline_regularity(geometry, geometry.pcurve_node(use), &low, &high,
                                            workspace);
  if (regularity == 2) return PHX_MC_UNRESOLVED;
  double pc_lower[6], pc_upper[6];
  int32_t status = geometry.interval_jets(geometry.pcurve_node(use), &low, &high,
                                           regularity == 1 ? 1 : 2,
                                           pc_lower, pc_upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  const auto dt = di::subtract(di::point(high), di::point(low));
  const int derivative = regularity == 1 ? 2 : 4;
  double endpoint_error[2] = {0.0, 0.0};
  for (int endpoint = 0; endpoint < 2; ++endpoint) {
    double lo[2], hi[2];
    status = geometry.interval_jets(geometry.pcurve_node(use), parameters + endpoint,
                                    parameters + endpoint, 0, lo, hi, workspace);
    if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
    for (int axis = 0; axis < 2; ++axis) {
      const auto difference = di::subtract({lo[axis], hi[axis]},
                                           di::point(chord_uv[2 * endpoint + axis]));
      if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
      endpoint_error[axis] = std::max(endpoint_error[axis], di::magnitude(difference));
    }
  }
  // The whole-chord remainder scale per axis: M dt (C1) or M dt^2 / 2 (C2).
  di::Interval scale[2];
  for (int axis = 0; axis < 2; ++axis) {
    const auto bound = di::point(
        di::magnitude({pc_lower[derivative + axis], pc_upper[derivative + axis]}));
    scale[axis] = regularity == 1
        ? di::multiply(bound, dt)
        : di::multiply(di::point(0.5), di::multiply(bound, di::multiply(dt, dt)));
  }
  // Finer enclosures: |p_a''| (|p_a'|) per sub-interval [s_j, s_j+1] of
  // lambda, weighted below by the remainder kernel's integral over it.
  constexpr int kSubintervals = 8;
  double sub_bound[kSubintervals][2];
  const auto lambda_t = [&](double lambda) {
    return di::add(di::point(parameters[0]),
                   di::multiply(di::point(lambda),
                                di::subtract(di::point(parameters[1]), di::point(parameters[0]))));
  };
  const auto clip = [](double a, double b, double lo, double hi) {
    const double from = std::max(a, lo), to = std::min(b, hi);
    return to > from ? di::subtract(di::point(to), di::point(from)) : di::point(0.0);
  };
  const auto chord_point = [&](double lambda, int axis) {
    return di::add(di::point(chord_uv[axis]),
                   di::multiply(di::point(lambda), di::subtract(di::point(chord_uv[2 + axis]),
                                                                di::point(chord_uv[axis]))));
  };
  for (int sub = 0; sub < kSubintervals; ++sub) {
    const auto ta = lambda_t(double(sub) / kSubintervals);
    const auto tb = lambda_t(double(sub + 1) / kSubintervals);
    const double sub_low = std::max(low, std::min(ta.lower, tb.lower));
    const double sub_high = std::min(high, std::max(ta.upper, tb.upper));
    double lo[6], hi[6];
    status = geometry.interval_jets(geometry.pcurve_node(use), &sub_low, &sub_high,
                                    regularity == 1 ? 1 : 2, lo, hi, workspace);
    if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
    for (int axis = 0; axis < 2; ++axis)
      sub_bound[sub][axis] = di::magnitude({lo[derivative + axis], hi[derivative + axis]});
  }
  for (int piece = 0; piece < pieces; ++piece) {
    const double first = double(piece) / pieces, last = double(piece + 1) / pieces;
    // The lambda of the piece nearest 1/2: where the remainder weight peaks.
    const double peak = last <= 0.5 ? last : first >= 0.5 ? first : 0.5;
    const auto weight = regularity == 1
        ? di::point(std::min(1.0, 2.0 * std::min(peak, 1.0 - peak)))
        : di::multiply(di::point(peak), di::subtract(di::point(1.0), di::point(peak)));
    // Kernel weights of the sub-intervals for lambda in [first, last].
    const auto alpha = di::subtract(di::point(1.0), di::point(first));
    const auto beta = di::point(last);
    const double crossing = last / (1.0 - first + last);
    di::Interval kernel[kSubintervals];
    for (int sub = 0; sub < kSubintervals; ++sub) {
      const double a = double(sub) / kSubintervals, b = double(sub + 1) / kSubintervals;
      if (regularity == 1) {
        const double middle = std::max(alpha.upper, beta.upper);
        kernel[sub] = di::add(di::multiply(alpha, clip(a, b, 0.0, first)),
                              di::add(di::multiply(di::point(middle), clip(a, b, first, last)),
                                      di::multiply(beta, clip(a, b, last, 1.0))));
      } else {
        // min(alpha s, beta (1 - s)) <= alpha s on [a, c], beta (1 - s) on
        // [c, b], for any c in [a, b].
        const double c = std::min(b, std::max(a, crossing));
        const auto pa = di::point(a), pc = di::point(c);
        const auto qc = di::subtract(di::point(1.0), pc);
        const auto qb = di::subtract(di::point(1.0), di::point(b));
        kernel[sub] = di::multiply(
            di::point(0.5),
            di::add(di::multiply(alpha, di::subtract(di::multiply(pc, pc), di::multiply(pa, pa))),
                    di::multiply(beta, di::subtract(di::multiply(qc, qc), di::multiply(qb, qb)))));
      }
    }
    double error[2], box_lower[2], box_upper[2];
    for (int axis = 0; axis < 2; ++axis) {
      const auto coarse = di::multiply(scale[axis], weight);
      auto integral = di::point(0.0);
      for (int sub = 0; sub < kSubintervals; ++sub)
        integral = di::add(integral, di::multiply(di::point(sub_bound[sub][axis]), kernel[sub]));
      const auto fine = di::multiply(integral, regularity == 1 ? dt : di::multiply(dt, dt));
      if (!di::finite(coarse) || !di::finite(fine)) return PHX_MC_UNRESOLVED;
      const auto sum = di::add(di::point(endpoint_error[axis]),
                               di::point(std::min(coarse.upper, fine.upper)));
      if (!di::finite(sum)) return PHX_MC_UNRESOLVED;
      error[axis] = sum.upper;
      const auto a = chord_point(first, axis), b = chord_point(last, axis);
      box_lower[axis] = di::subtract(di::point(std::min(a.lower, b.lower)),
                                     di::point(error[axis])).lower;
      box_upper[axis] = di::add(di::point(std::max(a.upper, b.upper)),
                                di::point(error[axis])).upper;
    }
    // The mean-value bound below needs S Lipschitz on the padded box: a
    // possible discontinuity there is refused, as in leaf_bound.
    if (spline_regularity(geometry, geometry.surface_node(face), box_lower, box_upper,
                          workspace) == 2)
      return PHX_MC_UNRESOLVED;
    double surface_lower[9], surface_upper[9];
    status = geometry.interval_jets(geometry.surface_node(face), box_lower, box_upper, 1,
                                    surface_lower, surface_upper, workspace);
    if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
    // S(x) - S(y) = integral of DS w along the segment (in the padded box);
    // component k is at most |S_u,k| |w_u| + |S_v,k| |w_v| over the box.
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      const auto su =
          di::point(di::magnitude({surface_lower[3 + axis], surface_upper[3 + axis]}));
      const auto sv =
          di::point(di::magnitude({surface_lower[6 + axis], surface_upper[6 + axis]}));
      magnitude[axis] = di::add(di::multiply(su, di::point(error[0])),
                                di::multiply(sv, di::point(error[1]))).upper;
    }
    caps[piece] = phx::mc::geometry::norm_upper(magnitude);
    if (!std::isfinite(caps[piece])) return PHX_MC_UNRESOLVED;
    double* box = boxes + 4 * piece;
    box[0] = box_lower[0], box[1] = box_lower[1], box[2] = box_upper[0], box[3] = box_upper[1];
  }
  return PHX_MC_OK;
}

// One distinct interior chart edge: its first occurrence (triangle * 3 +
// edge), the occurrence count and the hash's high bits. The key (patch and
// sorted endpoint bits) is re-read from the first occurrence on a tag match.
struct EdgeSlot {
  int64_t first = -1;
  int32_t count = 0;
  uint32_t tag = 0;
};

struct EdgeKey {
  std::array<uint64_t, 4> endpoints;
  bool direction;
  bool degenerate;
};

EdgeKey edge_key(const double* uv, std::size_t index) {
  const std::size_t triangle = index / 3, edge = index % 3;
  const double* a = uv + 6 * triangle + 2 * edge;
  const double* b = uv + 6 * triangle + 2 * ((edge + 1) % 3);
  const std::array<uint64_t, 2> first{std::bit_cast<uint64_t>(a[0]), std::bit_cast<uint64_t>(a[1])};
  const std::array<uint64_t, 2> second{std::bit_cast<uint64_t>(b[0]), std::bit_cast<uint64_t>(b[1])};
  const bool direction = first < second;
  const auto& low = direction ? first : second;
  const auto& high = direction ? second : first;
  return {{low[0], low[1], high[0], high[1]}, direction, first == second};
}

std::vector<phx_mc_fidelity_finding> association_findings(
    int64_t triangle_count, const int32_t* face_rows, const double* uv,
    int64_t chord_count, const int64_t* chord_use, const int64_t* use_patch,
    const double* chord_uv, const int64_t* edge_chord) {
  const auto edge_count = static_cast<std::size_t>(triangle_count) * 3;
  std::size_t table_size = 1;
  while (table_size < edge_count * 2) table_size *= 2;
  std::vector<EdgeSlot> table(table_size);
  std::vector<int64_t> owners(static_cast<std::size_t>(chord_count), -1);
  std::vector<phx_mc_fidelity_finding> findings;
  for (std::size_t index = 0; index < edge_count; ++index) {
    const int64_t triangle = static_cast<int64_t>(index / 3), edge = index % 3;
    const double* a = uv + 6 * triangle + 2 * edge;
    const double* b = uv + 6 * triangle + 2 * ((edge + 1) % 3);
    const int64_t chord = edge_chord[index];
    if (chord >= 0) {
      const double* c = chord_uv + 4 * chord;
      if (use_patch[chord_use[chord]] != face_rows[triangle] ||
          std::memcmp(a, c, 2 * sizeof(double)) != 0 ||
          std::memcmp(b, c + 2, 2 * sizeof(double)) != 0)
        findings.push_back({PHX_MC_FIDELITY_BAD_CHORD_EDGE, triangle, edge, -1, -1, chord});
      int64_t& owner = owners[static_cast<std::size_t>(chord)];
      if (owner >= 0)
        findings.push_back({PHX_MC_FIDELITY_DUPLICATE_CHORD, triangle, edge,
                            owner / 3, owner % 3, chord});
      else
        owner = static_cast<int64_t>(index);
      continue;
    }
    const EdgeKey key = edge_key(uv, index);
    uint64_t hash = static_cast<uint32_t>(face_rows[triangle]);
    for (uint64_t value : key.endpoints) {
      value += UINT64_C(0x9e3779b97f4a7c15);
      value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
      value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
      value ^= value >> 31;
      hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    }
    const auto tag = static_cast<uint32_t>(hash >> 32);
    const auto matches = [&](const EdgeSlot& slot) {
      return slot.tag == tag && face_rows[slot.first / 3] == face_rows[triangle] &&
             edge_key(uv, static_cast<std::size_t>(slot.first)).endpoints == key.endpoints;
    };
    std::size_t position = static_cast<std::size_t>(hash) & (table_size - 1);
    while (table[position].first >= 0 && !matches(table[position]))
      position = (position + 1) & (table_size - 1);
    EdgeSlot& slot = table[position];
    if (slot.first < 0) {
      slot = {static_cast<int64_t>(index), 1, tag};
    } else {
      ++slot.count;
      const bool direction = edge_key(uv, static_cast<std::size_t>(slot.first)).direction;
      if (slot.count > 2 || direction == key.direction || key.degenerate)
        findings.push_back({PHX_MC_FIDELITY_UNPAIRED_INTERIOR, triangle, edge,
                            slot.first / 3, slot.first % 3, -1});
    }
  }
  for (const EdgeSlot& slot : table)
    if (slot.count == 1)
      findings.push_back({PHX_MC_FIDELITY_UNPAIRED_INTERIOR,
                          slot.first / 3, slot.first % 3, -1, -1, -1});
  for (int64_t chord = 0; chord < chord_count; ++chord)
    if (owners[static_cast<std::size_t>(chord)] < 0)
      findings.push_back({PHX_MC_FIDELITY_MISSING_CHORD, -1, -1, -1, -1, chord});
  return findings;
}

// A sliver (chart_to_mesh -1) stands for the mesh edge from its pole vertex P
// to its third vertex Q: that edge {P, Q} must be an edge of a mesh triangle.
void chart_mesh_findings(
    const phx_mc_cad_model& model, int64_t chart_count,
    const int64_t* chart_triangles, int64_t mesh_count, const int64_t* mesh_triangles,
    const int64_t* chart_to_mesh, const int64_t* edge_chord, const int64_t* chord_use,
    std::vector<int32_t>& pole_edges, std::vector<phx_mc_fidelity_finding>& findings) {
  std::vector<int64_t> owners(static_cast<std::size_t>(mesh_count), -1);
  std::vector<std::array<int64_t, 2>> mesh_edges;
  const auto has_mesh_edge = [&](int64_t a, int64_t b) {
    if (mesh_edges.empty() && mesh_count > 0) {
      mesh_edges.reserve(static_cast<std::size_t>(mesh_count) * 3);
      for (int64_t t = 0; t < mesh_count; ++t)
        for (int k = 0; k < 3; ++k) {
          const int64_t p = mesh_triangles[3 * t + k], q = mesh_triangles[3 * t + (k + 1) % 3];
          mesh_edges.push_back({std::min(p, q), std::max(p, q)});
        }
      std::sort(mesh_edges.begin(), mesh_edges.end());
    }
    return a != b && std::binary_search(mesh_edges.begin(), mesh_edges.end(),
                                        std::array<int64_t, 2>{std::min(a, b), std::max(a, b)});
  };
  for (int64_t row = 0; row < chart_count; ++row) {
    int32_t pole_edge = -1;
    bool bad = false;
    for (int edge = 0; edge < 3; ++edge) {
      const int64_t chord = edge_chord[3 * row + edge];
      if (chord < 0 || model.use_kinds[chord_use[chord]] != 1) continue;
      if (pole_edge >= 0 ||
          chart_triangles[3 * row + edge] != chart_triangles[3 * row + (edge + 1) % 3])
        bad = true;
      pole_edge = edge;
    }
    pole_edges[static_cast<std::size_t>(row)] = pole_edge;
    const int64_t mapped = chart_to_mesh[row];
    if (mapped == -1) {
      if (pole_edge < 0 ||
          !has_mesh_edge(chart_triangles[3 * row + pole_edge],
                         chart_triangles[3 * row + (pole_edge + 2) % 3]))
        bad = true;
    } else if (mapped < 0 || mapped >= mesh_count || pole_edge >= 0) {
      bad = true;
    } else {
      const int64_t* chart = chart_triangles + 3 * row;
      const int64_t* mesh = mesh_triangles + 3 * mapped;
      bool same = false;
      for (int rotation = 0; rotation < 3; ++rotation)
        same |= chart[0] == mesh[rotation] && chart[1] == mesh[(rotation + 1) % 3] &&
                chart[2] == mesh[(rotation + 2) % 3];
      int64_t& owner = owners[static_cast<std::size_t>(mapped)];
      if (!same || owner >= 0) bad = true;
      if (owner < 0) owner = row;
    }
    if (bad)
      findings.push_back({PHX_MC_FIDELITY_CHART_MESH_LINK, row, pole_edge, mapped, -1, -1});
  }
  for (int64_t row = 0; row < mesh_count; ++row)
    if (owners[static_cast<std::size_t>(row)] < 0)
      findings.push_back({PHX_MC_FIDELITY_MISSING_MESH_TRIANGLE, -1, -1, row, -1, -1});
}

// Corner errors |S(uv_j) - X_j| of the original vertices, shared by every
// chart row that has the same mesh vertex at bitwise the same chart point.
struct CornerCache {
  double u = 0.0, v = 0.0, error = 0.0;
  int32_t face = -1;
};

// Trim lens charges, per subdivision leaf and per chord piece. A point x
// where the trimmed face and the chord polygon differ lies in the lens of one
// chord, on the segment from y = l(lambda) to z = p(lambda) for a lambda in
// some piece of that chord (trim_lens_bound), inside that piece's padded box
// (both y and the segment are), within the piece's per-axis chart bounds of
// y and z: |S(x) - S(y)| and |S(x) - S(z)| are at most the piece's cap. If x
// is outside the polygon (a face point), the mesh point I_T(y) of the leaf
// holding y is within that leaf's bound plus the cap; if x is inside it (a
// mesh point I_T(x)), the face point S(z) is within the bound of the leaf
// holding x plus the cap. Both leaves' chart boxes meet the piece's padded
// box (y and x lie in it), so each leaf is charged the largest cap of a piece
// of a chord of its face whose padded box meets the leaf's box; a leaf away
// from every lens piece is charged nothing. A refused chord has no pieces:
// every row of its face refuses. Pieces are
// binned per face on a grid whose cell is the face's largest padded-box
// extent (a box meets at most 2 x 2 cells); one monotone cell map serves
// boxes and rows, so meeting boxes share the cell of a common point.
// API: construct from one Chord entry per lens piece {face (-1: skipped),
// box = closed chart box (u lower, v lower, u upper, v upper) holding the
// piece's lens and matching chord points, cap = the distance charge of any
// point there}; any producer whose entries satisfy that box/cap contract (a
// trim chord piece, or another lens such as a seam) may add entries. charge
// (face, box) returns the largest cap of an entry of `face` whose box meets
// the closed `box` (0 when none); subdivided_bound passes each leaf's
// outward chart box and adds the result to the leaf bound.
class LensIndex {
 public:
  struct Chord { int32_t face = -1; double box[4]{}; double cap = 0.0; };
  LensIndex(int64_t surface_count, std::vector<Chord> chords)
      : chords_(std::move(chords)), faces_(static_cast<std::size_t>(surface_count)) {
    for (const Chord& chord : chords_) {
      if (chord.face < 0) continue;
      Face& face = faces_[static_cast<std::size_t>(chord.face)];
      if (!face.used) {
        face.used = true;
        face.origin[0] = chord.box[0], face.origin[1] = chord.box[1];
      }
      face.origin[0] = std::min(face.origin[0], chord.box[0]);
      face.origin[1] = std::min(face.origin[1], chord.box[1]);
      face.cell = std::max({face.cell, chord.box[2] - chord.box[0], chord.box[3] - chord.box[1]});
    }
    for (Face& face : faces_)
      if (!(face.cell > 0.0) || !std::isfinite(face.cell)) face.cell = 1.0;
    for (std::size_t c = 0; c < chords_.size(); ++c) {
      const Chord& chord = chords_[c];
      if (chord.face < 0) continue;
      const Face& face = faces_[static_cast<std::size_t>(chord.face)];
      const int64_t x0 = cell(face, 0, chord.box[0]), x1 = cell(face, 0, chord.box[2]);
      const int64_t y0 = cell(face, 1, chord.box[1]), y1 = cell(face, 1, chord.box[3]);
      for (int64_t x = x0; x <= x1; ++x)
        for (int64_t y = y0; y <= y1; ++y)
          entries_.push_back({chord.face, x, y, static_cast<int64_t>(c)});
    }
    std::sort(entries_.begin(), entries_.end());
    for (std::size_t e = 0; e < entries_.size(); ++e) {
      Face& face = faces_[static_cast<std::size_t>(entries_[e].face)];
      if (face.first < 0) face.first = static_cast<int64_t>(e);
      face.last = static_cast<int64_t>(e) + 1;
      for (int axis = 0; axis < 2; ++axis) {
        const int64_t value = axis == 0 ? entries_[e].x : entries_[e].y;
        face.low[axis] = std::min(face.low[axis], value);
        face.high[axis] = std::max(face.high[axis], value);
      }
    }
  }

  // The largest cap of a chord of `face` whose padded box meets `box`.
  double charge(int32_t face_index, const double* box) const {
    const Face& face = faces_[static_cast<std::size_t>(face_index)];
    if (face.first < 0) return 0.0;
    int64_t low[2], high[2];
    for (int axis = 0; axis < 2; ++axis) {
      low[axis] = std::max(face.low[axis], cell(face, axis, box[axis]));
      high[axis] = std::min(face.high[axis], cell(face, axis, box[2 + axis]));
      if (low[axis] > high[axis]) return 0.0;
    }
    double result = 0.0;
    const auto visit = [&](std::size_t begin, std::size_t end) {
      for (std::size_t e = begin; e < end; ++e) {
        const Chord& chord = chords_[static_cast<std::size_t>(entries_[e].chord)];
        if (box[2] < chord.box[0] || chord.box[2] < box[0] || box[3] < chord.box[1] ||
            chord.box[3] < box[1])
          continue;
        result = std::max(result, chord.cap);
      }
    };
    // Either span alone decides before the product could overflow.
    const int64_t span_x = high[0] - low[0], span_y = high[1] - low[1];
    if (span_x >= 64 || span_y >= 64 || (span_x + 1) * (span_y + 1) > 64) {
      visit(static_cast<std::size_t>(face.first), static_cast<std::size_t>(face.last));
      return result;
    }
    for (int64_t x = low[0]; x <= high[0]; ++x) {
      const Entry first{face_index, x, low[1], 0};
      const Entry last{face_index, x, high[1], std::numeric_limits<int64_t>::max()};
      const auto begin = std::lower_bound(entries_.begin(), entries_.end(), first);
      const auto end = std::upper_bound(begin, entries_.end(), last);
      visit(static_cast<std::size_t>(begin - entries_.begin()),
            static_cast<std::size_t>(end - entries_.begin()));
    }
    return result;
  }

 private:
  struct Face {
    bool used = false;
    double origin[2] = {0.0, 0.0};
    double cell = 0.0;
    int64_t first = -1, last = -1;
    int64_t low[2] = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()};
    int64_t high[2] = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min()};
  };
  struct Entry {
    int32_t face;
    int64_t x, y, chord;
    bool operator<(const Entry& other) const {
      return std::tie(face, x, y, chord) < std::tie(other.face, other.x, other.y, other.chord);
    }
  };
  // Monotone in value; clamped so far rows stay representable.
  static int64_t cell(const Face& face, int axis, double value) {
    const double scaled = std::floor((value - face.origin[axis]) / face.cell);
    return static_cast<int64_t>(std::clamp(scaled, -0x1p40, 0x1p40));
  }
  std::vector<Chord> chords_;
  std::vector<Face> faces_;
  std::vector<Entry> entries_;
};

// Tight mode (pole slivers): a leaf within tolerance is still split while
// its bound exceeds kTightRatio times the largest vertex error seen so far
// (a pointwise value of |S - I_T|, so the returned bound approaches the true
// maximum), up to kTightDepth or kTightSplits; it never refuses for that.
constexpr double kTightRatio = 1.25;
constexpr int32_t kTightDepth = 12;
constexpr int32_t kTightSplits = 4096;

// `lens` (or null): each leaf is charged lens->charge over its chart box
// (LensIndex) and certifies when its bound plus that charge is within
// `tolerance`; a leaf whose charge alone reaches it is split.
FidelityTriangleResult subdivided_bound(const CadGeometry& geometry, FaceKernel& kernel,
                                        const double* vertices, const double* uv,
                                        const double* corner_error, double tolerance,
                                        int32_t max_depth, bool tight, const LensIndex* lens,
                                        int32_t face, IntervalEvaluationWorkspace& workspace) {
  FidelityTriangleResult result{std::numeric_limits<double>::infinity(), PHX_MC_UNRESOLVED,
                                PHX_MC_FIDELITY_INTERVAL_REFUSAL, 0};
  Leaf stack[73];
  stack[0] = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}, 0};
  int size = 1;
  double maximum = 0.0, observed = 0.0;
  while (size > 0) {
    const Leaf leaf = stack[--size];
    double charge = 0.0, budget = tolerance;
    if (lens != nullptr) {
      double box[4] = {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()};
      for (const Vertex& vertex : leaf.vertices) {
        di::Interval point[2];
        exact_chart_point(vertex, uv, point);
        for (int axis = 0; axis < 2; ++axis) {
          box[axis] = std::min(box[axis], point[axis].lower);
          box[2 + axis] = std::max(box[2 + axis], point[axis].upper);
        }
      }
      charge = lens->charge(face, box);
      budget = di::subtract(di::point(tolerance), di::point(charge)).lower;
    }
    double bound = 0.0, vertex_term = 0.0;
    bool fits = false;
    if (charge < tolerance) {
      result.status = leaf_bound(geometry, kernel, leaf, vertices, uv, corner_error,
                                 tight ? 0.0 : budget, workspace, vertex_term, bound);
      if (result.status != PHX_MC_OK) {
        if (result.status == PHX_MC_INVALID_ARGUMENT || result.status == PHX_MC_INVALID_INPUT)
          result.status = PHX_MC_UNRESOLVED;
        return result;
      }
      observed = std::max(observed, vertex_term);
      fits = bound <= budget;
    }
    if (fits &&
        (!tight || bound <= kTightRatio * observed || leaf.depth >= kTightDepth ||
         result.subdivisions >= kTightSplits)) {
      maximum = std::max(maximum, charge > 0.0
                                      ? di::add(di::point(bound), di::point(charge)).upper
                                      : bound);
      continue;
    }
    if (!fits && (leaf.depth >= max_depth || result.subdivisions >= 1000000)) {
      result.status = PHX_MC_UNRESOLVED;
      result.reason = PHX_MC_FIDELITY_RESOLUTION;
      return result;
    }
    ++result.subdivisions;
    const Vertex ab = midpoint(leaf.vertices[0], leaf.vertices[1]);
    const Vertex bc = midpoint(leaf.vertices[1], leaf.vertices[2]);
    const Vertex ca = midpoint(leaf.vertices[2], leaf.vertices[0]);
    const int32_t depth = leaf.depth + 1;
    stack[size++] = {{leaf.vertices[0], ab, ca}, depth};
    stack[size++] = {{ab, leaf.vertices[1], bc}, depth};
    stack[size++] = {{ca, bc, leaf.vertices[2]}, depth};
    stack[size++] = {{ab, bc, ca}, depth};
  }
  result.upper = maximum;
  result.reason = PHX_MC_FIDELITY_CERTIFIED;
  return result;
}

// Pole chart (verify_cad_pole_chart.hpp): the sphere frame
//   S = o + r (cos v (cos u e1 + sin u e2) + sin v a)
// of a node, from the exact-source coefficient enclosures (any vectors
// e1, e2, a: the pole-chart form needs no orthonormality). Families:
// - the sphere template;
// - a placement (affine, integers[1] == 3) of a node with a sphere frame;
// - the revolution about (O, a) of a circle3 c(v) = center + R (cos v e1 +
//   sin v e2) whose center and e2 lie on the axis exactly. The evaluator's
//   O + a (a.w) + cos u w_r + sin u (a x w_r), w = c(v) - O, w_r = w - a (a.w),
//   is then the form with o = O + a (a.(center - O)), e1 := e1_r,
//   e2 := a x e1_r, a := (a.e2) a, r = R. "Exactly": point enclosures, a a
//   signed coordinate unit vector, e1 off the axis (its axis coordinate 0),
//   the off-axis coordinates of e2 zero and of center and O bitwise equal.
// - (disk centre, `pole_v` given: the chart parameter of the degenerate pole
//   chord) the revolution about (O, a) of a line p(v) = p0 + v d (line3:
//   point, direction; exact_line3: first, last with d = last - first)
//   perpendicular to the axis (d's axis coordinate 0) through the axis at
//   v = pole_v (p(pole_v) - O has zero off-axis coordinates, checked with
//   error-free transformations): the evaluator's O + a (a.w) + cos u w_r +
//   sin u (a x w_r), w_r = (v - pole_v) d, is the flat frame o = O + a
//   (a.(p0 - O)), e1 = d, e2 = a x d, with the same exactness conditions.
// False for every other family: a circle off the axis is a spindle or horn
// torus, a line meeting the axis off the perpendicular a cone apex, neither
// smooth at the pole; splines with a collapsed edge have no enclosure here.
bool pole_sphere_frame(const CadGeometry& geometry, const IntervalEvaluationWorkspace& workspace,
                       int32_t index, pole::SphereFrame& frame,
                       double pole_v = std::numeric_limits<double>::quiet_NaN()) {
  static const int32_t kLine = template_index("line3"), kExactLine = template_index("exact_line3");
  static const int32_t kSphere = template_index("sphere"), kCircle = template_index("circle3");
  const Node& node = geometry.node(index);
  const auto coefficient = [&](int32_t at, int64_t i) {
    const int64_t k = geometry.model().node_real_offsets[at] + i;
    return di::Interval{workspace.coefficient_lower[k], workspace.coefficient_upper[k]};
  };
  if (node.kind == PHX_MC_CAD_NODE_TEMPLATE) {
    if (node.integers[0] != kSphere || node.real_count != 13) return false;
    for (int i = 0; i < 3; ++i) {
      frame.o[i] = coefficient(index, i);
      frame.e1[i] = coefficient(index, 3 + i);
      frame.e2[i] = coefficient(index, 6 + i);
      frame.a[i] = coefficient(index, 9 + i);
    }
    frame.r = coefficient(index, 12);
  } else if (node.kind == PHX_MC_CAD_NODE_AFFINE && node.integers[1] == 3) {
    pole::SphereFrame child;
    if (!pole_sphere_frame(geometry, workspace, static_cast<int32_t>(node.integers[0]), child,
                           pole_v))
      return false;
    const auto apply = [&](const di::Interval* x, int r, bool offset) {
      di::Interval value = offset ? coefficient(index, 9 + r) : di::point(0.0);
      for (int c = 0; c < 3; ++c)
        value = di::add(value, di::multiply(coefficient(index, 3 * r + c), x[c]));
      return value;
    };
    for (int r = 0; r < 3; ++r) {
      frame.o[r] = apply(child.o, r, true);
      frame.e1[r] = apply(child.e1, r, false);
      frame.e2[r] = apply(child.e2, r, false);
      frame.a[r] = apply(child.a, r, false);
    }
    frame.r = child.r;
    frame.flat = child.flat;
    frame.pole_v = child.pole_v;
  } else if (node.kind == PHX_MC_CAD_NODE_REVOLUTION) {
    const auto profile = static_cast<int32_t>(node.integers[0]);
    const Node& curve = geometry.node(profile);
    const bool line = curve.kind == PHX_MC_CAD_NODE_TEMPLATE && curve.real_count == 6 &&
                      (curve.integers[0] == kLine || curve.integers[0] == kExactLine);
    if (line) {
      if (std::isnan(pole_v)) return false;
      const bool by_ends = curve.integers[0] == kExactLine;
      double origin[3], axis[3], p[3], q[3];
      for (int i = 0; i < 3; ++i) {
        const di::Interval values[4] = {coefficient(index, i), coefficient(index, 3 + i),
                                        coefficient(profile, i), coefficient(profile, 3 + i)};
        for (const di::Interval& x : values)
          if (!(x.lower == x.upper) || !std::isfinite(x.lower)) return false;
        origin[i] = values[0].lower, axis[i] = values[1].lower;
        p[i] = values[2].lower, q[i] = values[3].lower;
      }
      int k = -1;
      for (int i = 0; i < 3; ++i)
        if (std::fabs(axis[i]) == 1.0) k = i;
      if (k < 0) return false;
      // Perpendicular: d_k = 0 exactly.
      if (by_ends ? std::bit_cast<uint64_t>(p[k]) != std::bit_cast<uint64_t>(q[k]) : q[k] != 0.0)
        return false;
      for (int i = 0; i < 3; ++i) {
        if (i == k) continue;
        if (axis[i] != 0.0) return false;
        // p(pole_v)_i == O_i exactly.
        if (by_ends) {
          const double end = pole_v == 0.0 ? p[i] : pole_v == 1.0 ? q[i] : std::nan("");
          if (!(end == origin[i])) return false;
        } else {
          const double product = pole_v * q[i];
          const double sum = p[i] + product;
          const double bv = sum - p[i], av = sum - bv;
          if (std::fma(pole_v, q[i], -product) != 0.0 || (p[i] - av) + (product - bv) != 0.0 ||
              !(sum == origin[i]))
            return false;
        }
      }
      di::Interval d[3];
      for (int i = 0; i < 3; ++i) {
        d[i] = by_ends ? di::subtract(di::point(q[i]), di::point(p[i])) : di::point(q[i]);
        frame.o[i] = di::point(i == k ? p[k] : origin[i]);
        frame.e1[i] = i == k ? di::point(0.0) : d[i];
        frame.e2[i] = frame.a[i] = di::point(0.0);
      }
      const int a1 = (k + 1) % 3, a2 = (k + 2) % 3;
      const di::Interval s = di::point(axis[k]);
      frame.e2[a2] = di::multiply(s, d[a1]);
      frame.e2[a1] = di::multiply(di::point(-axis[k]), d[a2]);
      frame.r = di::point(1.0);
      frame.flat = true;
      frame.pole_v = pole_v;
    } else {
      if (curve.kind != PHX_MC_CAD_NODE_TEMPLATE || curve.integers[0] != kCircle ||
          curve.real_count != 10)
        return false;
      double origin[3], axis[3], centre[3], first[3], second[3], radius = 0.0;
      const auto exact = [&](int32_t at, int64_t i, double& value) {
        const di::Interval x = coefficient(at, i);
        value = x.lower;
        return x.lower == x.upper && std::isfinite(x.lower);
      };
      for (int i = 0; i < 3; ++i)
        if (!exact(index, i, origin[i]) || !exact(index, 3 + i, axis[i]) ||
            !exact(profile, i, centre[i]) || !exact(profile, 3 + i, first[i]) ||
            !exact(profile, 6 + i, second[i]))
          return false;
      if (!exact(profile, 9, radius)) return false;
      int k = -1;
      for (int i = 0; i < 3; ++i)
        if (std::fabs(axis[i]) == 1.0) k = i;
      if (k < 0 || first[k] != 0.0) return false;
      for (int i = 0; i < 3; ++i)
        if (i != k && (axis[i] != 0.0 || second[i] != 0.0 ||
                       std::bit_cast<uint64_t>(centre[i]) != std::bit_cast<uint64_t>(origin[i])))
          return false;
      for (int i = 0; i < 3; ++i) {
        frame.o[i] = di::point(i == k ? centre[i] : origin[i]);
        frame.e1[i] = di::point(i == k ? 0.0 : first[i]);
        frame.a[i] = di::point(i == k ? second[i] : 0.0);
      }
      // e2 = a x e1_r: a has the single entry axis[k].
      for (int i = 0; i < 3; ++i) frame.e2[i] = di::point(0.0);
      const int p = (k + 1) % 3, q = (k + 2) % 3;
      frame.e2[q] = di::point(axis[k] * first[p]);
      frame.e2[p] = di::point(-axis[k] * first[q]);
      frame.r = di::point(radius);
    }
  } else {
    return false;
  }
  for (int i = 0; i < 3; ++i)
    if (!di::finite(frame.o[i]) || !di::finite(frame.e1[i]) || !di::finite(frame.e2[i]) ||
        !di::finite(frame.a[i]))
      return false;
  return di::finite(frame.r);
}

// Enclosure of the pole-chart image rho (cos u, sin u) of a chart point.
bool pole_chart_image(const pole::SphereFrame& frame, AnchoredTrigonometry& trig,
                      const double* uv, int sigma, di::Interval* image) {
  di::Interval rho, c, s;
  if (!pole::colatitude(frame, uv[1], sigma, rho)) return false;
  trig.evaluate(di::point(uv[0]), c, s);
  image[0] = di::multiply(rho, c);
  image[1] = di::multiply(rho, s);
  return di::finite(image[0]) && di::finite(image[1]);
}

// The row's pole-chart corners (x0, y0, x1, y1, x2, y2): a pole corner (bit
// j of `pole_mask`, decided by the caller from exact topology) is the origin,
// the pole's own chart point; every other corner the binary64 midpoint of its
// image enclosure. The triangle's linear map is taken on these corners.
bool pole_corners(const pole::SphereFrame& frame, AnchoredTrigonometry& trig, const double* uv,
                  int sigma, int32_t pole_mask, double* xy) {
  for (int j = 0; j < 3; ++j) {
    if ((pole_mask >> j) & 1) {
      xy[2 * j] = xy[2 * j + 1] = 0.0;
      continue;
    }
    di::Interval image[2];
    if (!pole_chart_image(frame, trig, uv + 2 * j, sigma, image)) return false;
    for (int axis = 0; axis < 2; ++axis)
      xy[2 * j + axis] = image[axis].lower + 0.5 * (image[axis].upper - image[axis].lower);
  }
  return true;
}

// |S(x) - sum_i w_i X_i| at a pole-chart leaf vertex, over its enclosure.
int32_t pole_vertex_error(const pole::SphereFrame& frame, int sigma, const Vertex& vertex,
                          const di::Interval* point, const double* vertices, double& error) {
  di::Interval jets[4][3];
  if (!pole::jets(frame, sigma, point[0], point[1], 0, 0, jets)) return PHX_MC_UNRESOLVED;
  double magnitude[3];
  for (int axis = 0; axis < 3; ++axis) {
    auto linear = di::point(0.0);
    for (int i = 0; i < 3; ++i)
      if (vertex.weight[i] != 0.0)
        linear = di::add(linear, di::multiply(di::point(vertex.weight[i]),
                                              di::point(vertices[3 * i + axis])));
    const auto difference = di::subtract(jets[0][axis], linear);
    if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
    magnitude[axis] = di::magnitude(difference);
  }
  error = phx::mc::geometry::norm_upper(magnitude);
  return std::isfinite(error) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// leaf_bound in the pole chart: S - I_L by the Taylor remainder of the
// pole-chart Hessian over the leaf box (offsets from the box centre), plus
// the affine I_L - I_T, maximal at a leaf vertex.
int32_t pole_leaf_bound(const pole::SphereFrame& frame, int sigma, const Leaf& leaf,
                        const double* vertices, const double* xy, double& vertex_term,
                        double& upper) {
  double box_lower[2] = {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()};
  double box_upper[2] = {-std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()};
  double mismatch = 0.0, radius_squared = 0.0;
  for (const Vertex& vertex : leaf.vertices) {
    di::Interval point[2];
    exact_chart_point(vertex, xy, point);
    for (int axis = 0; axis < 2; ++axis) {
      box_lower[axis] = std::min(box_lower[axis], point[axis].lower);
      box_upper[axis] = std::max(box_upper[axis], point[axis].upper);
    }
    double reach[3] = {di::magnitude(point[0]), di::magnitude(point[1]), 0.0};
    const double norm = phx::mc::geometry::norm_upper(reach);
    radius_squared = std::max(radius_squared, di::multiply(di::point(norm), di::point(norm)).upper);
    double error = 0.0;
    const int32_t status = pole_vertex_error(frame, sigma, vertex, point, vertices, error);
    if (status != PHX_MC_OK) return status;
    mismatch = std::max(mismatch, error);
  }
  vertex_term = mismatch;
  // The remainder needs the Hessian on the leaf triangle only (half_extent),
  // which lies in the disk of its largest vertex norm.
  di::Interval jets[4][3];
  if (!pole::jets(frame, sigma, {box_lower[0], box_upper[0]}, {box_lower[1], box_upper[1]}, 1, 3,
                  jets, radius_squared))
    return PHX_MC_UNRESOLVED;
  const auto ex = half_extent(box_lower[0], box_upper[0]);
  const auto ey = half_extent(box_lower[1], box_upper[1]);
  double magnitude[3];
  for (int axis = 0; axis < 3; ++axis) {
    const auto hxx = di::point(di::magnitude(jets[1][axis]));
    const auto hxy = di::point(di::magnitude(jets[2][axis]));
    const auto hyy = di::point(di::magnitude(jets[3][axis]));
    auto sum = di::multiply(hxx, di::multiply(ex, ex));
    const auto mixed = di::multiply(hxy, di::multiply(ex, ey));
    sum = di::add(sum, di::add(mixed, mixed));
    sum = di::add(sum, di::multiply(hyy, di::multiply(ey, ey)));
    magnitude[axis] = di::multiply(di::point(0.5), sum).upper;
  }
  upper = di::add(di::point(mismatch), di::point(phx::mc::geometry::norm_upper(magnitude))).upper;
  return std::isfinite(upper) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
}

// sup over the pole-chart triangle of |S - I_T| (the same subdivision as
// subdivided_bound, without lens charges: the seam charge is the caller's).
// `tight` (a pole sliver, two corners at the origin): subdivided_bound's
// tight mode, refining toward the sampled maximum.
FidelityTriangleResult pole_subdivided_bound(const pole::SphereFrame& frame, int sigma,
                                             const double* vertices, const double* xy,
                                             double tolerance, int32_t max_depth, bool tight) {
  FidelityTriangleResult result{std::numeric_limits<double>::infinity(), PHX_MC_UNRESOLVED,
                                PHX_MC_FIDELITY_INTERVAL_REFUSAL, 0};
  Leaf stack[73];
  stack[0] = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}, 0};
  int size = 1;
  double maximum = 0.0, observed = 0.0;
  while (size > 0) {
    const Leaf leaf = stack[--size];
    double bound = 0.0, vertex_term = 0.0;
    result.status = pole_leaf_bound(frame, sigma, leaf, vertices, xy, vertex_term, bound);
    if (result.status != PHX_MC_OK) return result;
    observed = std::max(observed, vertex_term);
    const bool fits = bound <= tolerance;
    if (fits && (!tight || bound <= kTightRatio * observed || leaf.depth >= kTightDepth ||
                 result.subdivisions >= kTightSplits)) {
      maximum = std::max(maximum, bound);
      continue;
    }
    if (!fits && (leaf.depth >= max_depth || result.subdivisions >= 1000000)) {
      result.status = PHX_MC_UNRESOLVED;
      result.reason = PHX_MC_FIDELITY_RESOLUTION;
      return result;
    }
    ++result.subdivisions;
    const Vertex ab = midpoint(leaf.vertices[0], leaf.vertices[1]);
    const Vertex bc = midpoint(leaf.vertices[1], leaf.vertices[2]);
    const Vertex ca = midpoint(leaf.vertices[2], leaf.vertices[0]);
    const int32_t depth = leaf.depth + 1;
    stack[size++] = {{leaf.vertices[0], ab, ca}, depth};
    stack[size++] = {{ab, leaf.vertices[1], bc}, depth};
    stack[size++] = {{ca, bc, leaf.vertices[2]}, depth};
    stack[size++] = {{ab, bc, ca}, depth};
  }
  result.upper = maximum;
  result.reason = PHX_MC_FIDELITY_CERTIFIED;
  return result;
}

// Seam lens of one boundary edge e of a pole region (see PoleRegion), from
// chart corner a to b: gamma(t) = (1 - t) uv_a + t uv_b is the uv-straight
// edge, P(gamma(t)) its pole-chart image, Phi(t) = (1 - t) c_a + t c_b the
// pole-chart-straight edge and L(t) = (1 - t) X_a + t X_b the mesh edge. The
// lens is the straight homotopy H(t, s) = (1 - s) P(gamma(t)) + s Phi(t) in
// the pole chart, where S is smooth. With f(s) = S(H(t, s)) - L(t):
// f(0) = S(gamma(t)) - L(t) (S_xy o P = S_uv), f(1) = S_xy(Phi(t)) - L(t)
// and f'' = D^2 S_xy [Delta, Delta], Delta = Phi(t) - P(gamma(t)), so per
// component |f(s)| <= max(|f(0)|, |f(1)|) + sup |f''| / 8 and
//   |S(H(t, s)) - L(t)| <= max(A, B) + |D^2 S_xy| W^2 / 8 =: cap,
// A = sup_t |S_uv(gamma) - L|, B = sup_t |S_xy(Phi) - L| (each over
// kSeamPieces equal pieces: the larger endpoint value plus dt^2 / 8 sup
// |D^2 S [delta, delta]| over the piece's box), W >= sup |Delta|: the
// endpoint mismatch |c - P(uv)| (image enclosures) plus the chord sag
// (1/8) sup |(P o gamma)''| with |(P o gamma)''|^2 = rho^2 du^4 +
// 4 drho^2 du^2 (rho and u affine in t), and |D^2 S_xy [Delta, Delta]| <=
// (|S_xx| + 2 |S_xy| + |S_yy|) W^2 per component over the box |x|, |y| <=
// max rho + mismatch, which holds P(gamma), Phi and H. Every lens point is
// within `cap` of a point of the mesh edge.
constexpr int kSeamPieces = 16;

bool seam_lens_bound(const CadGeometry& geometry, IntervalEvaluationWorkspace& workspace,
                     int32_t node, const pole::SphereFrame& frame, int sigma,
                     const double* uv_a, const double* uv_b, const double* c_a,
                     const double* c_b, const double* x_a, const double* x_b, double& cap,
                     double* edges = nullptr) {
  AnchoredTrigonometry trig;
  const di::Interval dt = di::point(1.0 / kSeamPieces);
  const di::Interval eighth = di::point(0.125);
  const auto mesh_edge = [&](double t, int axis) {
    return di::add(di::point(x_a[axis]), di::multiply(di::point(t), di::subtract(
                                                          di::point(x_b[axis]), di::point(x_a[axis]))));
  };
  const auto along = [&](const double* a, const double* b, double t, int axis) {
    return di::add(di::point(a[axis]),
                   di::multiply(di::point(t), di::subtract(di::point(b[axis]), di::point(a[axis]))));
  };
  // sup over [0, 1] of |S(curve(t)) - L(t)| for the uv (uv = true) or the
  // pole-chart straight edge.
  const auto edge_sup = [&](bool uv, double& sup) {
    const double* a = uv ? uv_a : c_a;
    const double* b = uv ? uv_b : c_b;
    di::Interval delta[2], points[kSeamPieces + 1][2];
    double error[kSeamPieces + 1][3];
    for (int axis = 0; axis < 2; ++axis)
      delta[axis] = di::subtract(di::point(b[axis]), di::point(a[axis]));
    for (int k = 0; k <= kSeamPieces; ++k) {
      const double t = static_cast<double>(k) / kSeamPieces;
      for (int axis = 0; axis < 2; ++axis) points[k][axis] = along(a, b, t, axis);
      di::Interval surface[3];
      if (uv) {
        const double lo[2] = {points[k][0].lower, points[k][1].lower};
        const double hi[2] = {points[k][0].upper, points[k][1].upper};
        double jet_lower[18], jet_upper[18];
        if (geometry.interval_jets(node, lo, hi, 0, jet_lower, jet_upper, workspace) != PHX_MC_OK)
          return false;
        for (int axis = 0; axis < 3; ++axis) surface[axis] = {jet_lower[axis], jet_upper[axis]};
      } else {
        di::Interval jets[4][3];
        if (!pole::jets(frame, sigma, points[k][0], points[k][1], 0, 0, jets)) return false;
        for (int axis = 0; axis < 3; ++axis) surface[axis] = jets[0][axis];
      }
      for (int axis = 0; axis < 3; ++axis) {
        const auto difference = di::subtract(surface[axis], mesh_edge(t, axis));
        if (!di::finite(difference)) return false;
        error[k][axis] = di::magnitude(difference);
      }
    }
    sup = 0.0;
    for (int k = 0; k < kSeamPieces; ++k) {
      di::Interval x{std::min(points[k][0].lower, points[k + 1][0].lower),
                     std::max(points[k][0].upper, points[k + 1][0].upper)};
      di::Interval y{std::min(points[k][1].lower, points[k + 1][1].lower),
                     std::max(points[k][1].upper, points[k + 1][1].upper)};
      di::Interval second[3][3];  // (uu, uv, vv) or (xx, xy, yy) per component
      if (uv) {
        const double lo[2] = {x.lower, y.lower}, hi[2] = {x.upper, y.upper};
        double jet_lower[18], jet_upper[18];
        if (geometry.interval_jets(node, lo, hi, 2, jet_lower, jet_upper, workspace) != PHX_MC_OK)
          return false;
        for (int term = 0; term < 3; ++term)
          for (int axis = 0; axis < 3; ++axis)
            second[term][axis] = {jet_lower[9 + 3 * term + axis], jet_upper[9 + 3 * term + axis]};
      } else {
        di::Interval jets[4][3];
        // The piece (a segment) lies in the disk of its larger endpoint norm.
        double radius_squared = 0.0;
        for (const auto* end : {points[k], points[k + 1]}) {
          double reach[3] = {di::magnitude(end[0]), di::magnitude(end[1]), 0.0};
          const double norm = phx::mc::geometry::norm_upper(reach);
          radius_squared =
              std::max(radius_squared, di::multiply(di::point(norm), di::point(norm)).upper);
        }
        if (!pole::jets(frame, sigma, x, y, 1, 3, jets, radius_squared)) return false;
        for (int term = 0; term < 3; ++term)
          for (int axis = 0; axis < 3; ++axis) second[term][axis] = jets[1 + term][axis];
      }
      double magnitude[3];
      for (int axis = 0; axis < 3; ++axis) {
        const auto h0 = di::point(di::magnitude(second[0][axis]));
        const auto h1 = di::point(di::magnitude(second[1][axis]));
        const auto h2 = di::point(di::magnitude(second[2][axis]));
        const auto d0 = di::point(di::magnitude(delta[0])), d1 = di::point(di::magnitude(delta[1]));
        auto curvature = di::multiply(h0, di::multiply(d0, d0));
        const auto mixed = di::multiply(h1, di::multiply(d0, d1));
        curvature = di::add(curvature, di::add(mixed, mixed));
        curvature = di::add(curvature, di::multiply(h2, di::multiply(d1, d1)));
        const auto remainder = di::multiply(eighth, di::multiply(curvature, di::multiply(dt, dt)));
        magnitude[axis] =
            di::add(di::point(std::max(error[k][axis], error[k + 1][axis])), remainder).upper;
      }
      sup = std::max(sup, phx::mc::geometry::norm_upper(magnitude));
    }
    return std::isfinite(sup);
  };
  double a = 0.0, b = 0.0;
  if (!edge_sup(true, a) || !edge_sup(false, b)) return false;
  // W: endpoint mismatch plus the sag of P o gamma.
  di::Interval rho[2];
  double mismatch = 0.0;
  for (int end = 0; end < 2; ++end) {
    const double* point_uv = end == 0 ? uv_a : uv_b;
    const double* c = end == 0 ? c_a : c_b;
    di::Interval image[2];
    if (!pole::colatitude(frame, point_uv[1], sigma, rho[end]) ||
        !pole_chart_image(frame, trig, point_uv, sigma, image))
      return false;
    double offset[3] = {0.0, 0.0, 0.0};
    for (int axis = 0; axis < 2; ++axis)
      offset[axis] = std::max(di::subtract(di::point(c[axis]), di::point(image[axis].lower)).upper,
                              di::subtract(di::point(image[axis].upper), di::point(c[axis])).upper);
    mismatch = std::max(mismatch, phx::mc::geometry::norm_upper(offset));
  }
  const di::Interval du = di::subtract(di::point(uv_b[0]), di::point(uv_a[0]));
  const di::Interval drho = di::subtract(rho[1], rho[0]);
  const di::Interval rho_hull{std::min(rho[0].lower, rho[1].lower),
                              std::max(rho[0].upper, rho[1].upper)};
  const di::Interval du2 = pole::square(du);
  const di::Interval acceleration =
      di::add(pole::square(di::multiply(rho_hull, du2)),
              di::multiply(di::point(4.0), di::multiply(pole::square(drho), du2)));
  if (!di::finite(acceleration)) return false;
  const double sag = di::multiply(eighth, di::point(phx::mc::geometry::next_up(
                                              std::sqrt(acceleration.upper)))).upper;
  const double width = di::add(di::point(mismatch), di::point(sag)).upper;
  const double reach = di::add(di::point(rho_hull.upper), di::point(mismatch)).upper;
  di::Interval jets[4][3];
  // P(gamma), Phi and H lie in the disk |z| <= reach (|P(gamma)| = rho, the
  // corners within the mismatch of it, the disk convex).
  if (!pole::jets(frame, sigma, {-reach, reach}, {-reach, reach}, 1, 3, jets,
                  di::multiply(di::point(reach), di::point(reach)).upper))
    return false;
  const di::Interval w2 = di::multiply(di::point(width), di::point(width));
  double magnitude[3];
  for (int axis = 0; axis < 3; ++axis) {
    auto sum = di::add(di::point(di::magnitude(jets[1][axis])),
                       di::point(di::magnitude(jets[3][axis])));
    const auto mixed = di::point(di::magnitude(jets[2][axis]));
    sum = di::add(sum, di::add(mixed, mixed));
    magnitude[axis] = di::multiply(eighth, di::multiply(sum, w2)).upper;
  }
  cap = di::add(di::point(std::max(a, b)),
                di::point(phx::mc::geometry::norm_upper(magnitude))).upper;
  if (edges != nullptr) edges[0] = a, edges[1] = b;
  return std::isfinite(cap);
}

// V3's mixed atlas at a sphere pole. The pole vertex is exact topology: a
// mesh vertex at a corner of a pole chord (use_kinds 1) whose chart latitude
// is bitwise sigma kHalfPiLower. Its region R is the rows of the face that
// touch it (fan rows and slivers) or that have a corner in the closed cap
// sigma v >= kCapLatitude (an exact predicate on the input chart points, not
// on any bound). R holds every row meeting the cap, so an edge R shares with
// a uv row joins two corners below the cap (the uv row has none in it), and a
// mesh ring at the threshold cannot split rows between the charts. R is
// associated in the pole chart, the rest of the face in uv.
//
// The threshold. A chart-straight edge of surface length l whose image has
// total curvature k on the sphere of radius r sags l^2 k / 8; k^2 = 1/r^2 +
// k_g^2 (normal curvature 1/r for every curve). The worst direction in both
// charts is the azimuthal one at colatitude rho (radial lines are meridians,
// k_g = 0, in both). uv chart: a latitude circle, r k_g = cot rho, so the
// sag factor over the sphere's own l^2 / (8 r) is 1/sin rho. Pole chart, the
// metric d rho^2 + sin^2 rho d theta^2 in polar coordinates of (x, y): the
// Cartesian line tangent at rho has rho'' = 1/rho, theta' = 1/rho, speed
// sin rho / rho, and the geodesic rho'' = sin rho cos rho theta'^2, so
// r k_g = (rho - sin rho cos rho) / sin^2 rho. The pole chart is no worse
// than the uv chart where (rho - sin rho cos rho) / sin^2 rho <= cot rho,
// i.e. rho <= sin 2 rho: rho <= rho* = 0.947747133516990... (the positive
// root of rho = sin 2 rho), latitude pi/2 - rho* = 0.623049193277906...,
// rounded down to kCapLatitude (the cap only grows). A row's extent below
// the cap is not bounded and no argument needs it: a corner past rho = 2
// makes pole::series / pole::jets fail and the row refuses; nothing else
// depends on the row's extent. The seam charge is computed for every region
// boundary edge at its endpoints' colatitudes (extent affects only its
// size). The coverage argument uses rho_strip = the largest colatitude over
// the region's corners, and P is injective on the strip for every 0 < rho.
//
// Contract (MAC bc1b665 option (i), seam lens charge). For R's rows the
// linear map is taken on the pole-chart corners (pole_corners), Phi on each
// uv triangle the affine map onto its pole-chart triangle; Phi is continuous
// on R (shared chart corners share their pole-chart points). Each boundary
// edge of R (an edge no other row of R shares bitwise: edges shared with a
// uv row, seam and pole chord edges) carries a seam lens with cap c_e
// (seam_lens_bound), charged to both adjacent rows (row bound := max(bound,
// c_e)). Surface -> mesh: for x in R, z = P(x); P and Phi restricted to the
// boundary of R are homotopic through the lenses, so off the lenses
// deg(Phi, R, z) = deg(P, R, z) != 0 (P is injective with one Jacobian sign
// on the open strip), so z lies in some row's pole-chart triangle and S(x)
// is within that row's bound of the mesh; on a lens S(x) is within c_e of
// the mesh edge. Mesh -> surface: a pole row's mesh point I(z) is within its
// bound of S_xy(z) with |z| <= rho_strip; the face holds the whole strip
// {u in [u_min, u_max], rho <= rho_strip} (every chord of the face meeting
// it is a pole chord or a seam chord at u_min or u_max, checked exactly;
// the polygon/trim difference is the trim lens, charged over the strip's uv
// box), and an angle outside [u_min, u_max] is within r rho_strip (2 pi -
// (u_max - u_min)) of the face (charged). Faces failing a condition keep
// the uv chart for every row.
//
// Trimmed pole regions (a trim chord meets the strip, e.g. a sphere cap
// above a latitude cut, a disk whose fan reaches the rim): every chord
// meeting the strip that is not a pole or seam chord must be a latitude
// circle (constant v, exactly) and an edge of a region row, and every region
// corner must lie on the pole's side of it (exact comparisons of v), so the
// ring is outer: rho_corner <= rho_ring. Such chords are boundary edges of R
// and carry seam lenses like any other; their trim lens (the p-curve arc
// beyond the chord: constant v at the ends is checked, not along the arc) is
// added to the edge's seam cap, since a face point in it is within the
// lens cap of a chord point, itself within the seam cap of the mesh. Mesh
// -> surface: a pole-chart triangle lies in the disk of its largest corner
// (convex), so z in it has rho(z) <= rho_ring and P^{-1}(z), taken in the
// full turn, lies in the chord polygon's cap (the polygon vs the face is the
// trim-lens charge over the strip box). Faces whose pole is trimmed away
// (an inner ring) keep the uv chart: there the
// pole-chart chords of the ring leave the face.
struct PoleRegion {
  int32_t face = -1;
  int sigma = 0;
  int64_t vertex = -1;
  double strip = 0.0;  // rho_strip
  double gap = 0.0;    // the angular charge r rho_strip (2 pi - (u_max - u_min))+
  bool trimmed = false;  // a trim chord (a latitude circle) meets the strip
  pole::SphereFrame frame;
  std::vector<int64_t> rows;
  std::vector<int32_t> masks;
  std::vector<std::array<double, 6>> corners;  // pole-chart corners per row
};

constexpr double kCapLatitude = 0x1.3f004dc9db074p-1;

// The regions of every eligible pole; row_region[i] is the index of the
// region holding row i, or -1.
std::vector<PoleRegion> pole_regions(const CadGeometry& geometry,
                                     const phx_mc_cad_model& model,
                                     std::vector<FaceKernel>& kernels,
                                     IntervalEvaluationWorkspace& workspace,
                                     int64_t triangle_count, const int64_t* triangles,
                                     const int32_t* face_rows, const double* uv,
                                     int64_t chord_count, const int64_t* chord_use,
                                     const int64_t* use_patch, const double* chord_uv,
                                     const int64_t* edge_chord,
                                     const std::vector<uint8_t>& face_refused,
                                     std::vector<int32_t>& row_region) {
  const auto bits = [](double value) { return std::bit_cast<uint64_t>(value); };
  const auto at_pole = [&](const pole::SphereFrame& frame, double v, int sigma) {
    return bits(v) == bits(pole::pole_value(frame, sigma));
  };
  // Candidate poles (face, vertex, chart v bits), from the corners of pole chords.
  std::vector<std::array<int64_t, 3>> candidates;
  for (int64_t i = 0; i < triangle_count; ++i)
    for (int j = 0; j < 3; ++j) {
      const int64_t chord = edge_chord[3 * i + j];
      if (chord < 0 || model.use_kinds[chord_use[chord]] != 1) continue;
      for (const int corner : {j, (j + 1) % 3})
        candidates.push_back({face_rows[i], triangles[3 * i + corner],
                              std::bit_cast<int64_t>(uv[6 * i + 2 * corner + 1])});
    }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  struct Seed {
    int32_t face;
    int sigma;
    int64_t vertex;
    pole::SphereFrame frame;
  };
  std::vector<Seed> seeds;
  std::vector<PoleRegion> regions;
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    const auto [face64, vertex, v_bits] = candidates[c];
    const auto face = static_cast<int32_t>(face64);
    const double pole_v = std::bit_cast<double>(v_bits);
    // One chart parameter per pole vertex.
    const bool unique = (c == 0 || candidates[c - 1][0] != face64 || candidates[c - 1][1] != vertex) &&
                        (c + 1 == candidates.size() || candidates[c + 1][0] != face64 ||
                         candidates[c + 1][1] != vertex);
    if (!unique || face_refused[static_cast<std::size_t>(face)]) continue;
    auto& kernel = kernels[static_cast<std::size_t>(face)];
    if (!kernel.prepared) prepare_face_kernel(geometry, face, workspace, kernel);
    pole::SphereFrame frame;
    int sigma = 0;
    if (pole_sphere_frame(geometry, workspace, kernel.node, frame)) {
      sigma = pole_v > 0.0 ? 1 : -1;
      if (!at_pole(frame, pole_v, sigma)) continue;
    } else if (pole_sphere_frame(geometry, workspace, kernel.node, frame, pole_v) && frame.flat) {
      // The face's side of the disk centre: any corner of a row touching it
      // off the centre parameter.
      for (int64_t i = 0; i < triangle_count && sigma == 0; ++i)
        if (face_rows[i] == face)
          for (int j = 0; j < 3; ++j)
            if (triangles[3 * i + j] == vertex)
              for (int l = 0; l < 3; ++l) {
                const double v = uv[6 * i + 2 * l + 1];
                if (v != pole_v && sigma == 0) sigma = v > pole_v ? 1 : -1;
              }
      if (sigma == 0) continue;
    } else {
      continue;
    }
    seeds.push_back({face, sigma, vertex, frame});
  }
  for (const Seed& seed : seeds) {
    const int32_t face = seed.face;
    const int sigma = seed.sigma;
    const int64_t vertex = seed.vertex;
    const pole::SphereFrame& frame = seed.frame;
    auto& kernel = kernels[static_cast<std::size_t>(face)];
    PoleRegion region;
    region.face = face, region.sigma = sigma, region.vertex = vertex, region.frame = frame;
    bool valid = true;
    for (int64_t i = 0; i < triangle_count && valid; ++i) {
      if (face_rows[i] != face) continue;
      int32_t mask = 0;
      bool cap = false;
      for (int j = 0; j < 3; ++j) {
        const double v = uv[6 * i + 2 * j + 1];
        if (vertex >= 0 && triangles[3 * i + j] == vertex) {
          if (!at_pole(frame, v, sigma)) valid = false;
          mask |= 1 << j;
        }
        cap = cap || (!frame.flat && sigma * v >= kCapLatitude);
      }
      if (mask == 0 && !cap) continue;
      std::array<double, 6> xy{};
      if (!pole_corners(frame, kernel.trig, uv + 6 * i, sigma, mask, xy.data())) valid = false;
      for (int j = 0; j < 3 && valid; ++j) {
        di::Interval rho;
        if (!pole::colatitude(frame, uv[6 * i + 2 * j + 1], sigma, rho)) {
          valid = false;
          break;
        }
        const double radius[3] = {std::fabs(xy[2 * j]), std::fabs(xy[2 * j + 1]), 0.0};
        region.strip = std::max({region.strip, rho.upper, phx::mc::geometry::norm_upper(radius)});
      }
      region.rows.push_back(i);
      region.masks.push_back(mask);
      region.corners.push_back(xy);
    }
    if (!valid || region.rows.empty()) continue;
    // Chords meeting the strip: pole chords on this pole and seam chords at
    // the face's u ends need nothing; any other chord there must be an edge
    // of a region row (a trim chord on the region boundary), which makes the
    // region trimmed (checked below).
    double u_min = std::numeric_limits<double>::infinity(), u_max = -u_min;
    for (int64_t k = 0; k < chord_count; ++k)
      if (use_patch[chord_use[k]] == face)
        for (int end = 0; end < 2; ++end) {
          u_min = std::min(u_min, chord_uv[4 * k + 2 * end]);
          u_max = std::max(u_max, chord_uv[4 * k + 2 * end]);
        }
    std::vector<std::array<uint64_t, 4>> region_edges;
    for (const int64_t row : region.rows)
      for (int j = 0; j < 3; ++j) {
        const double* a = uv + 6 * row + 2 * j;
        const double* b = uv + 6 * row + 2 * ((j + 1) % 3);
        std::array<uint64_t, 2> p{bits(a[0]), bits(a[1])}, q{bits(b[0]), bits(b[1])};
        if (q < p) std::swap(p, q);
        region_edges.push_back({p[0], p[1], q[0], q[1]});
      }
    std::sort(region_edges.begin(), region_edges.end());
    for (int64_t k = 0; k < chord_count && valid; ++k) {
      if (use_patch[chord_use[k]] != face) continue;
      const double* p = chord_uv + 4 * k;
      if (model.use_kinds[chord_use[k]] == 1 && at_pole(frame, p[1], sigma) &&
          at_pole(frame, p[3], sigma))
        continue;
      if (bits(p[0]) == bits(p[2]) && (p[0] == u_min || p[0] == u_max)) continue;
      bool inside = false;
      for (int end = 0; end < 2; ++end) {
        di::Interval rho;
        if (!pole::colatitude(frame, p[2 * end + 1], sigma, rho) || !(rho.lower > region.strip))
          inside = true;
      }
      if (!inside) continue;
      // A trim chord in the strip: a latitude circle (constant v, exactly)
      // and an edge of a region row.
      std::array<uint64_t, 2> a{bits(p[0]), bits(p[1])}, b{bits(p[2]), bits(p[3])};
      if (b < a) std::swap(a, b);
      if (bits(p[1]) != bits(p[3]) ||
          !std::binary_search(region_edges.begin(), region_edges.end(),
                              std::array<uint64_t, 4>{a[0], a[1], b[0], b[1]}))
        valid = false;
      region.trimmed = true;
    }
    if (!valid) continue;
    // Every region corner lies inside each trim ring (exact comparisons of
    // v; the rings are outer: rho_corner <= rho_ring), so the pole-chart
    // triangles lie in the disk of their largest corner, inside the face.
    if (region.trimmed) {
      for (int64_t k = 0; k < chord_count && valid; ++k) {
        const double* p = chord_uv + 4 * k;
        if (use_patch[chord_use[k]] != face || model.use_kinds[chord_use[k]] == 1 ||
            bits(p[1]) != bits(p[3]) || bits(p[0]) == bits(p[2]))
          continue;
        di::Interval rho;
        if (pole::colatitude(frame, p[1], sigma, rho) && rho.lower > region.strip) continue;
        for (const int64_t row : region.rows)
          for (int j = 0; j < 3; ++j) {
            const double w = sigma * uv[6 * row + 2 * j + 1], t = sigma * p[1];
            // rho grows with sigma v at a disk centre, falls on a sphere.
            if (frame.flat ? w > t : w < t) valid = false;
          }
      }
      if (!valid) continue;
    }
    // The strip must make a full turn: u_max - u_min >= 2 pi rounded down
    // (exact comparison); a face spanning less keeps the uv chart. The real
    // excess 2 pi - (u_max - u_min) (at most one rounding) is charged:
    // |S(u, v) - S(u', v)| <= |r| (|e1| + |e2|) rho |u - u'| (|dS/du| = |r|
    // rho_s |-sin u e1 + cos u e2|, rho_s = sin rho or rho; e1, e2 need not
    // be orthogonal).
    if (!(u_max - u_min >= 2.0 * 0x1.921fb54442d18p+1)) continue;
    const di::Interval span = di::subtract(di::point(u_max), di::point(u_min));
    const di::Interval gap = di::subtract(di::point(2.0 * di::kPiUpper), span);
    const auto norm = [](const di::Interval* e) {
      double m[3];
      for (int i = 0; i < 3; ++i) m[i] = di::magnitude(e[i]);
      return phx::mc::geometry::norm_upper(m);
    };
    const double scale =
        di::multiply(di::point(di::magnitude(frame.r)),
                     di::add(di::point(norm(frame.e1)), di::point(norm(frame.e2))))
            .upper;
    region.gap = gap.upper > 0.0
                     ? di::multiply(di::point(scale),
                                    di::multiply(di::point(region.strip), di::point(gap.upper)))
                           .upper
                     : 0.0;
    if (!std::isfinite(region.gap)) continue;
    regions.push_back(std::move(region));
  }
  // A row claimed by two regions (two poles' regions meet): both keep uv.
  std::vector<int32_t> claims(row_region.size(), 0);
  for (const PoleRegion& region : regions)
    for (const int64_t row : region.rows) ++claims[static_cast<std::size_t>(row)];
  for (std::size_t r = 0; r < regions.size(); ++r) {
    bool shared = false;
    for (const int64_t row : regions[r].rows) shared |= claims[static_cast<std::size_t>(row)] > 1;
    if (shared) continue;
    for (const int64_t row : regions[r].rows)
      row_region[static_cast<std::size_t>(row)] = static_cast<int32_t>(r);
  }
  for (std::size_t r = 0; r < regions.size(); ++r)
    if (!regions[r].rows.empty() &&
        row_region[static_cast<std::size_t>(regions[r].rows[0])] != static_cast<int32_t>(r))
      regions[r].rows.clear();
  return regions;
}

// The rows of every pole region in the pole chart, each charged the trim
// lens over the strip's uv box and the angular gap, then each boundary
// edge's seam lens charged to both adjacent rows (PoleRegion's contract).
void pole_region_bounds(const CadGeometry& geometry, IntervalEvaluationWorkspace& workspace,
                        const std::vector<PoleRegion>& regions, const std::vector<FaceKernel>& kernels,
                        int64_t triangle_count, const int64_t* triangles,
                        const int32_t* face_rows, const double* uv, const double* points,
                        const std::vector<int32_t>& row_region, const LensIndex& lens,
                        double tolerance, int32_t max_depth, std::vector<double>& row_upper,
                        std::vector<int32_t>& row_status, std::vector<int32_t>& row_reason,
                        std::vector<int32_t>& row_splits) {
  using Key = std::array<uint64_t, 4>;
  const auto bits = [](double value) { return std::bit_cast<uint64_t>(value); };
  const auto edge = [&](int64_t row, int j) {
    const double* a = uv + 6 * row + 2 * j;
    const double* b = uv + 6 * row + 2 * ((j + 1) % 3);
    std::array<uint64_t, 2> p{bits(a[0]), bits(a[1])}, q{bits(b[0]), bits(b[1])};
    if (q < p) std::swap(p, q);
    return Key{p[0], p[1], q[0], q[1]};
  };
  const auto refuse = [&](int64_t row, int32_t reason) {
    const auto r = static_cast<std::size_t>(row);
    row_upper[r] = std::numeric_limits<double>::infinity();
    row_status[r] = PHX_MC_UNRESOLVED;
    row_reason[r] = reason;
  };
  const auto charge = [&](int64_t row, double cap) {
    const auto r = static_cast<std::size_t>(row);
    if (row_status[r] != PHX_MC_OK) return;
    if (!(cap <= tolerance)) {
      refuse(row, PHX_MC_FIDELITY_RESOLUTION);
      return;
    }
    row_upper[r] = std::max(row_upper[r], cap);
  };
  for (std::size_t r = 0; r < regions.size(); ++r) {
    const PoleRegion& region = regions[r];
    if (region.rows.empty()) continue;
    const pole::SphereFrame& frame = region.frame;
    // The strip's v range: from the pole parameter to rho = strip (v grows
    // with rho at a disk centre on side sigma, falls on a sphere's north).
    const double pole_v = pole::pole_value(frame, region.sigma);
    const int toward = frame.flat ? region.sigma : -region.sigma;
    const di::Interval far = di::add(di::point(pole_v),
                                     di::multiply(di::point(toward), di::point(region.strip)));
    const double box[4] = {-std::numeric_limits<double>::max(), std::min(pole_v, far.lower),
                           std::numeric_limits<double>::max(), std::max(pole_v, far.upper)};
    const double fixed =
        di::add(di::point(lens.charge(region.face, box)), di::point(region.gap)).upper;
    for (std::size_t k = 0; k < region.rows.size(); ++k) {
      const int64_t row = region.rows[k];
      double vertices[9];
      for (int j = 0; j < 3; ++j)
        for (int axis = 0; axis < 3; ++axis)
          vertices[3 * j + axis] = points[3 * triangles[3 * row + j] + axis];
      row_splits[static_cast<std::size_t>(row)] = 0;
      if (!(fixed < tolerance)) {
        refuse(row, PHX_MC_FIDELITY_RESOLUTION);
        continue;
      }
      const int32_t mask = region.masks[k];
      const bool sliver = mask == 3 || mask == 5 || mask == 6;
      const auto result = pole_subdivided_bound(
          frame, region.sigma, vertices, region.corners[k].data(),
          di::subtract(di::point(tolerance), di::point(fixed)).lower, max_depth, sliver);
      row_splits[static_cast<std::size_t>(row)] = result.subdivisions;
      if (result.status != PHX_MC_OK) {
        refuse(row, result.reason);
        continue;
      }
      row_upper[static_cast<std::size_t>(row)] =
          di::add(di::point(result.upper), di::point(fixed)).upper;
      row_status[static_cast<std::size_t>(row)] = PHX_MC_OK;
      row_reason[static_cast<std::size_t>(row)] = PHX_MC_FIDELITY_CERTIFIED;
    }
  }
  // Charges only after every region's rows are bounded, so no region's
  // bounding overwrites another's seam charge or refusal.
  for (std::size_t r = 0; r < regions.size(); ++r) {
    const PoleRegion& region = regions[r];
    if (region.rows.empty()) continue;
    const int32_t node = kernels[static_cast<std::size_t>(region.face)].node;
    const pole::SphereFrame& frame = region.frame;
    // Boundary edges: no other row of the region shares them bitwise.
    std::vector<std::pair<Key, int64_t>> inside, outside;
    for (std::size_t k = 0; k < region.rows.size(); ++k)
      for (int j = 0; j < 3; ++j)
        inside.push_back({edge(region.rows[k], j), static_cast<int64_t>(3 * k + j)});
    for (int64_t row = 0; row < triangle_count; ++row)
      if (face_rows[row] == region.face &&
          row_region[static_cast<std::size_t>(row)] != static_cast<int32_t>(r))
        for (int j = 0; j < 3; ++j) outside.push_back({edge(row, j), row});
    std::sort(inside.begin(), inside.end());
    std::sort(outside.begin(), outside.end());
    for (std::size_t e = 0; e < inside.size(); ++e) {
      if ((e > 0 && inside[e - 1].first == inside[e].first) ||
          (e + 1 < inside.size() && inside[e + 1].first == inside[e].first))
        continue;
      const auto k = static_cast<std::size_t>(inside[e].second / 3);
      const int a = static_cast<int>(inside[e].second % 3), b = (a + 1) % 3;
      const int64_t row = region.rows[k];
      const double* corners = region.corners[k].data();
      double cap = std::numeric_limits<double>::infinity();
      if (!seam_lens_bound(geometry, workspace, node, frame, region.sigma, uv + 6 * row + 2 * a,
                           uv + 6 * row + 2 * b, corners + 2 * a, corners + 2 * b,
                           points + 3 * triangles[3 * row + a],
                           points + 3 * triangles[3 * row + b], cap))
        cap = std::numeric_limits<double>::infinity();
      // The edge's own trim lens (a ring chord whose p-curve leaves the
      // chord): a face point beyond it is within that lens cap of a chord
      // point, which is within the seam cap of the mesh. LensIndex charges
      // every lens piece whose box meets the edge's uv box.
      {
        const double* p = uv + 6 * row + 2 * a;
        const double* q = uv + 6 * row + 2 * b;
        const double edge_box[4] = {std::min(p[0], q[0]), std::min(p[1], q[1]),
                                    std::max(p[0], q[0]), std::max(p[1], q[1])};
        cap = di::add(di::point(cap), di::point(lens.charge(region.face, edge_box))).upper;
      }
      charge(row, cap);
      const auto first = std::lower_bound(outside.begin(), outside.end(),
                                          std::pair<Key, int64_t>{inside[e].first, -1});
      for (auto it = first; it != outside.end() && it->first == inside[e].first; ++it)
        charge(it->second, cap);
    }
  }
}

}  // namespace

FidelityTriangleResult fidelity_triangle(const CadGeometry& geometry, int32_t face,
                                         const double* vertices, const double* uv,
                                         double tolerance, int32_t max_depth,
                                         IntervalEvaluationWorkspace& workspace) {
  FidelityTriangleResult result{std::numeric_limits<double>::infinity(), PHX_MC_UNRESOLVED,
                                PHX_MC_FIDELITY_INTERVAL_REFUSAL, 0};
  if (face < 0 || face >= geometry.surface_count() || vertices == nullptr || uv == nullptr ||
      !std::isfinite(tolerance) || tolerance < 0.0 || max_depth < 0 || max_depth > 24) {
    result.status = PHX_MC_INVALID_ARGUMENT;
    return result;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) { result.status = PHX_MC_INTERNAL_ERROR; return result; }
  FaceKernel kernel;
  prepare_face_kernel(geometry, face, workspace, kernel);
  return subdivided_bound(geometry, kernel, vertices, uv, nullptr, tolerance, max_depth, false,
                          nullptr, face, workspace);
}

FidelityTriangleResult fidelity_pole_triangle(const CadGeometry& geometry, int32_t face,
                                              const double* vertices, const double* uv,
                                              int32_t pole_side, int32_t pole_mask,
                                              double tolerance, int32_t max_depth,
                                              IntervalEvaluationWorkspace& workspace) {
  FidelityTriangleResult result{std::numeric_limits<double>::infinity(), PHX_MC_UNRESOLVED,
                                PHX_MC_FIDELITY_INTERVAL_REFUSAL, 0};
  if (face < 0 || face >= geometry.surface_count() || vertices == nullptr || uv == nullptr ||
      (pole_side != 1 && pole_side != -1) || pole_mask < 0 || pole_mask > 7 ||
      !std::isfinite(tolerance) || tolerance < 0.0 || max_depth < 0 || max_depth > 24) {
    result.status = PHX_MC_INVALID_ARGUMENT;
    return result;
  }
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) { result.status = PHX_MC_INTERNAL_ERROR; return result; }
  AnchoredTrigonometry trig;
  pole::SphereFrame frame;
  double xy[6];
  if (!pole_sphere_frame(geometry, workspace, geometry.surface_node(face), frame) ||
      !pole_corners(frame, trig, uv, pole_side, pole_mask, xy))
    return result;
  const bool sliver = pole_mask == 3 || pole_mask == 5 || pole_mask == 6;
  return pole_subdivided_bound(frame, pole_side, vertices, xy, tolerance, max_depth, sliver);
}

int32_t fidelity_pole_image(const CadGeometry& geometry, int32_t face,
                            IntervalEvaluationWorkspace& workspace, int32_t pole_side,
                            const double* uv, double* image) {
  if (face < 0 || face >= geometry.surface_count() || (pole_side != 1 && pole_side != -1))
    return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  AnchoredTrigonometry trig;
  pole::SphereFrame frame;
  di::Interval point[2];
  if (!pole_sphere_frame(geometry, workspace, geometry.surface_node(face), frame) ||
      !pole_chart_image(frame, trig, uv, pole_side, point))
    return PHX_MC_UNRESOLVED;
  image[0] = point[0].lower, image[1] = point[0].upper;
  image[2] = point[1].lower, image[3] = point[1].upper;
  return PHX_MC_OK;
}

int32_t fidelity_seam_lens(const CadGeometry& geometry, int32_t face,
                           IntervalEvaluationWorkspace& workspace, int32_t pole_side,
                           const double* uv_a, const double* uv_b, const double* c_a,
                           const double* c_b, const double* x_a, const double* x_b,
                           double* out) {
  if (face < 0 || face >= geometry.surface_count() || (pole_side != 1 && pole_side != -1))
    return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  const int32_t node = geometry.surface_node(face);
  pole::SphereFrame frame;
  if (!pole_sphere_frame(geometry, workspace, node, frame) ||
      !seam_lens_bound(geometry, workspace, node, frame, pole_side, uv_a, uv_b, c_a, c_b, x_a, x_b,
                       out[0], out + 1))
    return PHX_MC_UNRESOLVED;
  return PHX_MC_OK;
}

int32_t fidelity_pole_jets(const CadGeometry& geometry, int32_t face,
                           IntervalEvaluationWorkspace& workspace, int32_t pole_side,
                           const double* lower, const double* upper, double* jet_lower,
                           double* jet_upper) {
  if (face < 0 || face >= geometry.surface_count() || (pole_side != 1 && pole_side != -1))
    return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  pole::SphereFrame frame;
  di::Interval jets[4][3];
  if (!pole_sphere_frame(geometry, workspace, geometry.surface_node(face), frame) ||
      !pole::jets(frame, pole_side, {lower[0], upper[0]}, {lower[1], upper[1]}, 0, 3, jets))
    return PHX_MC_UNRESOLVED;
  for (int term = 0; term < 4; ++term)
    for (int axis = 0; axis < 3; ++axis) {
      jet_lower[3 * term + axis] = jets[term][axis].lower;
      jet_upper[3 * term + axis] = jets[term][axis].upper;
    }
  return PHX_MC_OK;
}

int32_t fidelity_anchored_sincos(double lower, double upper, double* out) {
  if (!(lower <= upper) || !std::isfinite(lower) || !std::isfinite(upper) || out == nullptr)
    return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  AnchoredTrigonometry trig;
  di::Interval c, s;
  trig.evaluate({lower, upper}, c, s);
  out[0] = c.lower, out[1] = c.upper, out[2] = s.lower, out[3] = s.upper;
  return PHX_MC_OK;
}

int32_t fidelity_closed_jets(const CadGeometry& geometry, int32_t face,
                             IntervalEvaluationWorkspace& workspace, const double* lower,
                             const double* upper, double* jet_lower, double* jet_upper) {
  if (face < 0 || face >= geometry.surface_count()) return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  FaceKernel kernel;
  prepare_face_kernel(geometry, face, workspace, kernel);
  if (!kernel.closed) return PHX_MC_UNRESOLVED;
  di::Interval jets[6][3];
  closed_jets(kernel.frame, kernel.trig, lower, upper, 0, 5, jets);
  for (int term = 0; term < 6; ++term)
    for (int axis = 0; axis < 3; ++axis) {
      jet_lower[3 * term + axis] = jets[term][axis].lower;
      jet_upper[3 * term + axis] = jets[term][axis].upper;
    }
  return PHX_MC_OK;
}

int32_t fidelity_trim_lens(const CadGeometry& geometry, int32_t face, int64_t use,
                           const double* chord_uv, const double* parameters,
                           IntervalEvaluationWorkspace& workspace, double* cap, double* padded) {
  if (face < 0 || face >= geometry.surface_count() || use < 0 ||
      use >= geometry.model().use_count)
    return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  return trim_lens_bound(geometry, face, use, chord_uv, parameters, workspace, 1, cap, padded);
}

double fidelity_lens_charge(int64_t surface_count, int64_t chord_count, const int32_t* faces,
                            const double* boxes, const double* caps, int32_t face,
                            const double* box) {
  std::vector<LensIndex::Chord> chords(static_cast<std::size_t>(chord_count));
  for (int64_t c = 0; c < chord_count; ++c) {
    auto& chord = chords[static_cast<std::size_t>(c)];
    chord.face = faces[c];
    std::copy(boxes + 4 * c, boxes + 4 * c + 4, chord.box);
    chord.cap = caps[c];
  }
  return LensIndex(surface_count, std::move(chords)).charge(face, box);
}

int32_t fidelity_net_hessian(const CadGeometry& geometry, int32_t face,
                             IntervalEvaluationWorkspace& workspace, const double* lower,
                             const double* upper, double* hessian_lower, double* hessian_upper) {
  if (face < 0 || face >= geometry.surface_count()) return PHX_MC_INVALID_ARGUMENT;
  const phx::mc::GradualNearestScope mode;
  if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
  FaceKernel kernel;
  prepare_face_kernel(geometry, face, workspace, kernel);
  di::Interval hessian[3][3];
  if (!spline_net_hessian(kernel.nets, lower, upper, hessian)) return PHX_MC_UNRESOLVED;
  for (int term = 0; term < 3; ++term)
    for (int axis = 0; axis < 3; ++axis) {
      hessian_lower[3 * term + axis] = hessian[term][axis].lower;
      hessian_upper[3 * term + axis] = hessian[term][axis].upper;
    }
  return PHX_MC_OK;
}
}  // namespace phx::mc::cad

extern "C" int32_t phx_mc_verify_cad_fidelity(
    const phx_mc_cad_model* model, int64_t coefficient_count,
    const double* coefficient_lower, const double* coefficient_upper,
    int64_t point_count, const double* points, int64_t triangle_count,
    const int64_t* triangles, const int32_t* face_rows, const double* uv,
    int64_t mesh_triangle_count, const int64_t* mesh_triangles, const int64_t* chart_to_mesh,
    int64_t use_count, const int64_t* use_patch, const double* use_range,
    int64_t loop_count, int64_t chord_count, const int64_t* chord_use,
    const int64_t* chord_loop, const double* chord_uv, const double* chord_range,
    const int64_t* edge_chord, double tolerance, int32_t max_depth,
    int64_t output_capacity, double* upper, int32_t* item_status, int32_t* reasons,
    int32_t* subdivisions, double* maximum_upper, int64_t evidence_capacity,
    phx_mc_fidelity_finding* evidence, int64_t* evidence_count) {
  return phx::mc::guarded([&]() -> int32_t {
    if (model == nullptr || !phx::mc::addressable(point_count, 3, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 6, sizeof(double)) ||
        !phx::mc::addressable(triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(mesh_triangle_count, 3, sizeof(int64_t)) ||
        !phx::mc::addressable(use_count, 2, sizeof(double)) ||
        !phx::mc::addressable(chord_count, 4, sizeof(double)) || loop_count < 0 ||
        coefficient_count < 0 || output_capacity < 0 || evidence_capacity < 0 ||
        maximum_upper == nullptr || evidence_count == nullptr ||
        !std::isfinite(tolerance) || tolerance < 0 || max_depth < 0 || max_depth > 24)
      return PHX_MC_INVALID_ARGUMENT;
    if (output_capacity < triangle_count) return PHX_MC_CAPACITY_EXCEEDED;
    if ((point_count && !points) ||
        (coefficient_count && (!coefficient_lower || !coefficient_upper)) ||
        (use_count && (!use_patch || !use_range)) ||
        (chord_count && (!chord_use || !chord_loop || !chord_uv || !chord_range)) ||
        (mesh_triangle_count && !mesh_triangles) ||
        (evidence_capacity && !evidence) ||
        (triangle_count && (!triangles || !face_rows || !uv || !edge_chord || !chart_to_mesh ||
                            !upper || !item_status || !reasons || !subdivisions)))
      return PHX_MC_INVALID_ARGUMENT;
    phx::mc::cad::CadGeometry geometry;
    const int32_t bind_status = geometry.bind(*model);
    if (bind_status != PHX_MC_OK) return bind_status;
    if (use_count != model->use_count) return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < use_count; ++i)
      if (use_patch[i] < 0 || use_patch[i] >= model->surface_count ||
          !std::isfinite(use_range[2 * i]) || !std::isfinite(use_range[2 * i + 1]) ||
          use_range[2 * i] == use_range[2 * i + 1])
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < chord_count; ++i) {
      if (chord_use[i] < 0 || chord_use[i] >= use_count ||
          chord_loop[i] < 0 || chord_loop[i] >= loop_count ||
          !std::isfinite(chord_range[2 * i]) || !std::isfinite(chord_range[2 * i + 1]) ||
          chord_range[2 * i] == chord_range[2 * i + 1])
        return PHX_MC_INVALID_ARGUMENT;
      for (int j = 0; j < 4; ++j)
        if (!std::isfinite(chord_uv[4 * i + j])) return PHX_MC_NONFINITE_INPUT;
    }
    if (coefficient_count != model->node_real_offsets[model->node_count])
      return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < coefficient_count; ++i)
      if (std::isnan(coefficient_lower[i]) || std::isnan(coefficient_upper[i]) ||
          coefficient_lower[i] > coefficient_upper[i])
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < point_count * 3; ++i)
      if (!std::isfinite(points[i])) return PHX_MC_NONFINITE_INPUT;
    for (int64_t i = 0; i < mesh_triangle_count * 3; ++i)
      if (mesh_triangles[i] < 0 || mesh_triangles[i] >= point_count)
        return PHX_MC_INVALID_ARGUMENT;
    for (int64_t i = 0; i < triangle_count; ++i) {
      if (face_rows[i] < 0 || face_rows[i] >= model->surface_count)
        return PHX_MC_INVALID_ARGUMENT;
      for (int j = 0; j < 3; ++j) {
        const int64_t chord = edge_chord[3 * i + j];
        if (triangles[3 * i + j] < 0 || triangles[3 * i + j] >= point_count ||
            chord < -1 || chord >= chord_count)
          return PHX_MC_INVALID_ARGUMENT;
      }
      for (int j = 0; j < 6; ++j)
        if (!std::isfinite(uv[6 * i + j])) return PHX_MC_NONFINITE_INPUT;
    }
    // V4's use coverage, loop closure and orientation are this verifier's
    // precondition, checked here: the caller's use and loop rows are bound
    // to the model (patch from loop_surfaces, chord loop from
    // loop_use_offsets, every surface on some loop), every use of one face
    // runs one way (all in the model's sense-directed range order, or all
    // reversed, as cad_mesh exports them), then V4's own checks run on them.
    // With one direction per face, consistent orientation and the boundary
    // chain, the chart triangles cover the trimmed region once.
    if (loop_count != model->loop_count) return PHX_MC_INVALID_INPUT;
    std::vector<int64_t> use_loop(static_cast<std::size_t>(use_count), -1);
    std::vector<int8_t> direction(static_cast<std::size_t>(model->surface_count), 0);
    for (int64_t loop = 0; loop < loop_count; ++loop)
      for (int64_t use = model->loop_use_offsets[loop]; use < model->loop_use_offsets[loop + 1];
           ++use) {
        use_loop[static_cast<std::size_t>(use)] = loop;
        const auto bits = [](double value) { return std::bit_cast<uint64_t>(value); };
        const double* given = use_range + 2 * use;
        const double* source = model->use_ranges + 2 * use;
        const bool sense = model->use_senses[use] > 0;
        const double first = source[sense ? 0 : 1], last = source[sense ? 1 : 0];
        const bool forward = bits(given[0]) == bits(first) && bits(given[1]) == bits(last);
        const bool backward = bits(given[0]) == bits(last) && bits(given[1]) == bits(first);
        const int32_t face = model->loop_surfaces[loop];
        int8_t& way = direction[static_cast<std::size_t>(face)];
        const int8_t found = forward ? 1 : backward ? -1 : 0;
        if (use_patch[use] != face || found == 0 || (way != 0 && way != found))
          return PHX_MC_INVALID_INPUT;
        way = found;
      }
    for (const int8_t way : direction)
      if (way == 0) return PHX_MC_INVALID_INPUT;  // a surface on no loop
    for (int64_t i = 0; i < chord_count; ++i)
      if (chord_loop[i] != use_loop[static_cast<std::size_t>(chord_use[i])])
        return PHX_MC_INVALID_INPUT;
    {
      const phx::mc::GradualNearestScope exact;
      if (!exact.established()) return PHX_MC_INTERNAL_ERROR;
      std::vector<phx_mc_chart_failure> failures;
      std::vector<uint8_t> oriented;
      phx::mc::chart::check_uses(use_count, use_patch, use_range, chord_count, chord_use,
                                 chord_loop, chord_uv, chord_range, failures);
      phx::mc::chart::check_loops(loop_count, chord_count, chord_loop, chord_use, use_patch,
                                  chord_uv, failures);
      phx::mc::chart::check_orientation(model->surface_count, triangle_count, face_rows, uv,
                                        chord_count, chord_use, use_patch, chord_uv, oriented,
                                        failures);
      if (!failures.empty()) return PHX_MC_INVALID_INPUT;
    }
    auto findings = phx::mc::cad::association_findings(
        triangle_count, face_rows, uv, chord_count, chord_use, use_patch, chord_uv, edge_chord);
    std::vector<int32_t> pole_edges(static_cast<std::size_t>(triangle_count), -1);
    phx::mc::cad::chart_mesh_findings(*model, triangle_count, triangles,
                                     mesh_triangle_count, mesh_triangles, chart_to_mesh,
                                     edge_chord, chord_use, pole_edges, findings);
    *evidence_count = static_cast<int64_t>(findings.size());
    if (*evidence_count > evidence_capacity) return PHX_MC_CAPACITY_EXCEEDED;
    if (!findings.empty()) {
      std::copy(findings.begin(), findings.end(), evidence);
      for (int64_t i = 0; i < triangle_count; ++i) {
        upper[i] = std::numeric_limits<double>::infinity();
        item_status[i] = PHX_MC_UNRESOLVED;
        reasons[i] = PHX_MC_FIDELITY_EDGE_ASSOCIATION;
        subdivisions[i] = 0;
      }
      *maximum_upper = std::numeric_limits<double>::infinity();
      return PHX_MC_UNRESOLVED;
    }
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    phx::mc::cad::IntervalEvaluationWorkspace workspace;
    workspace.coefficient_lower = coefficient_lower;
    workspace.coefficient_upper = coefficient_upper;
    workspace.coefficient_count = coefficient_count;
    // Rows are staged and published only when the call completes, so a hard
    // error leaves every output unchanged.
    const auto rows = static_cast<std::size_t>(triangle_count);
    std::vector<double> row_upper(rows);
    std::vector<int32_t> row_status(rows), row_reason(rows), row_splits(rows);
    std::vector<phx::mc::cad::FaceKernel> kernels(static_cast<std::size_t>(model->surface_count));
    std::vector<phx::mc::cad::CornerCache> corners(static_cast<std::size_t>(point_count));
    // Every chord's lens caps once, kLensPieces pieces along the chord
    // (trim_lens_bound), then the per-leaf charges of phx::mc::cad::LensIndex.
    constexpr int kLensPieces = 8;
    std::vector<phx::mc::cad::LensIndex::Chord> lens_pieces(
        static_cast<std::size_t>(chord_count) * kLensPieces);
    std::vector<double> chord_cap(static_cast<std::size_t>(chord_count), 0.0);
    std::vector<uint8_t> face_refused(static_cast<std::size_t>(model->surface_count), 0);
    for (int64_t chord = 0; chord < chord_count; ++chord) {
      const auto face = static_cast<int32_t>(use_patch[chord_use[chord]]);
      double caps[kLensPieces], boxes[4 * kLensPieces];
      if (phx::mc::cad::trim_lens_bound(geometry, face, chord_use[chord], chord_uv + 4 * chord,
                                        chord_range + 2 * chord, workspace, kLensPieces, caps,
                                        boxes) != PHX_MC_OK) {
        face_refused[static_cast<std::size_t>(face)] = 1;
        continue;
      }
      for (int piece = 0; piece < kLensPieces; ++piece) {
        auto& entry = lens_pieces[static_cast<std::size_t>(chord * kLensPieces + piece)];
        entry.face = face;
        entry.cap = caps[piece];
        std::copy(boxes + 4 * piece, boxes + 4 * piece + 4, entry.box);
        chord_cap[static_cast<std::size_t>(chord)] =
            std::max(chord_cap[static_cast<std::size_t>(chord)], caps[piece]);
      }
    }
    const phx::mc::cad::LensIndex lens_index(model->surface_count, std::move(lens_pieces));
    // Pole regions take the pole chart (phx::mc::cad::PoleRegion); their
    // rows are bounded after the uv rows, whose seam neighbours they charge.
    std::vector<int32_t> row_region(static_cast<std::size_t>(triangle_count), -1);
    const auto regions = phx::mc::cad::pole_regions(
        geometry, *model, kernels, workspace, triangle_count, triangles, face_rows, uv,
        chord_count, chord_use, use_patch, chord_uv, edge_chord, face_refused, row_region);
    for (int64_t i = 0; i < triangle_count; ++i) {
      const auto row = static_cast<std::size_t>(i);
      if (row_region[row] >= 0) continue;
      bool trim_refused = face_refused[static_cast<std::size_t>(face_rows[i])] != 0;
      // The leaves along the middle of an own chord pay its largest piece cap
      // at every depth: a cap at or above tolerance refuses the row at once.
      // Every other charge is the per-leaf one of subdivided_bound.
      double own = 0.0;
      double vertices[9];
      for (int j = 0; j < 3; ++j) {
        const int64_t chord = edge_chord[3 * i + j];
        if (chord >= 0) own = std::max(own, chord_cap[static_cast<std::size_t>(chord)]);
        const int64_t vertex = triangles[3 * i + j];
        for (int axis = 0; axis < 3; ++axis)
          vertices[3 * j + axis] = points[3 * vertex + axis];
      }
      if (trim_refused || !std::isfinite(own) || own >= tolerance) {
        row_upper[row] = std::numeric_limits<double>::infinity();
        row_status[row] = PHX_MC_UNRESOLVED;
        row_reason[row] = trim_refused ? PHX_MC_FIDELITY_TRIM_REFUSAL : PHX_MC_FIDELITY_RESOLUTION;
        row_splits[row] = 0;
        continue;
      }
      // A pole sliver is the chart triangle of a collapsed side: its two pole
      // corners carry the same mesh vertex, and S - I_T on the chart triangle
      // takes the same interpolation theorem as every other row (S is smooth
      // on the chart, the collapse only makes I_T degenerate). I_T lies on
      // the mesh edge from the pole to the third vertex. Tight mode refines
      // it toward its sampled maximum.
      const int pole_edge = pole_edges[static_cast<std::size_t>(i)];
      auto& kernel = kernels[static_cast<std::size_t>(face_rows[i])];
      if (!kernel.prepared) phx::mc::cad::prepare_face_kernel(geometry, face_rows[i], workspace, kernel);
      double corner_error[3];
      bool corners_known = true;
      for (int j = 0; j < 3 && corners_known; ++j) {
        const double* point_uv = uv + 6 * i + 2 * j;
        auto& cached = corners[static_cast<std::size_t>(triangles[3 * i + j])];
        if (cached.face == face_rows[i] && std::memcmp(&cached.u, point_uv, sizeof(double)) == 0 &&
            std::memcmp(&cached.v, point_uv + 1, sizeof(double)) == 0) {
          corner_error[j] = cached.error;
          continue;
        }
        phx::mc::cad::Vertex corner{};
        corner.weight[j] = 1.0;
        const phx::mc::remesh::Interval chart[2] = {phx::mc::remesh::point(point_uv[0]),
                                                    phx::mc::remesh::point(point_uv[1])};
        corners_known = phx::mc::cad::vertex_error(geometry, kernel, corner, chart, vertices,
                                                   workspace, corner_error[j]) == PHX_MC_OK;
        if (corners_known) cached = {point_uv[0], point_uv[1], corner_error[j], face_rows[i]};
      }
      const auto result = phx::mc::cad::subdivided_bound(
          geometry, kernel, vertices, uv + 6 * i, corners_known ? corner_error : nullptr,
          tolerance, max_depth, pole_edge >= 0, &lens_index, face_rows[i], workspace);
      if (result.status != PHX_MC_OK && result.status != PHX_MC_UNRESOLVED)
        return result.status;
      row_upper[row] = result.upper;
      row_status[row] = result.status;
      row_reason[row] = result.reason;
      row_splits[row] = result.subdivisions;
    }
    phx::mc::cad::pole_region_bounds(geometry, workspace, regions, kernels, triangle_count,
                                     triangles, face_rows,
                                     uv, points, row_region, lens_index, tolerance, max_depth,
                                     row_upper, row_status, row_reason, row_splits);
    int32_t aggregate = PHX_MC_OK;
    double maximum = 0.0;
    for (int64_t i = 0; i < triangle_count; ++i) {
      const auto row = static_cast<std::size_t>(i);
      if (row_status[row] != PHX_MC_OK) aggregate = PHX_MC_UNRESOLVED;
      maximum = std::max(maximum, row_upper[row]);
    }
    std::copy(row_upper.begin(), row_upper.end(), upper);
    std::copy(row_status.begin(), row_status.end(), item_status);
    std::copy(row_reason.begin(), row_reason.end(), reasons);
    std::copy(row_splits.begin(), row_splits.end(), subdivisions);
    *maximum_upper = aggregate == PHX_MC_OK ? maximum : std::numeric_limits<double>::infinity();
    return aggregate;
  });
}
