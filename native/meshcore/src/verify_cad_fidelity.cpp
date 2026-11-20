// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "verify_cad_fidelity.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "capi_guard.hpp"
#include "directed_interval.hpp"
#include "floating_point_mode.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

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
int spline_knot_regularity(const double* lower, const double* upper, int degree, int count,
                           double box_lower, double box_upper) {
  int result = 0;
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

int spline_regularity(const CadGeometry& geometry, int32_t index, const double* box_lower,
                      const double* box_upper, const IntervalEvaluationWorkspace& workspace) {
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
      int result = spline_knot_regularity(lo, hi, p, nu, box_lower[0], box_upper[0]);
      if (surface) {
        const int q = static_cast<int>(node.integers[1]), nv = static_cast<int>(node.integers[3]);
        const int uk = nu + p + 1;
        result = std::max(result, spline_knot_regularity(lo + uk, hi + uk, q, nv,
                                                         box_lower[1], box_upper[1]));
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
    case PHX_MC_CAD_NODE_REPARAMETERIZED:
    case PHX_MC_CAD_NODE_ISOPARAMETRIC:
      return child(0, all_lower, all_upper);
    case PHX_MC_CAD_NODE_SURFACE_MAPPED:
    case PHX_MC_CAD_NODE_INTERSECTION:
      return std::max(child(0, all_lower, all_upper), child(1, all_lower, all_upper));
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
                   const IntervalEvaluationWorkspace& workspace, KnotLine* lines, int& count) {
  const Node& node = geometry.node(index);
  const auto child = [&](int slot, int axis) {
    return c0_knot_lines(geometry, static_cast<int32_t>(node.integers[slot]), axis, box_lower,
                         box_upper, workspace, lines, count);
  };
  const auto knots = [&](const double* lo, const double* hi, int degree, int n, int axis) {
    for (int k = degree + 1; k < n; ++k) {
      if (lo[k] == lo[k - 1] && hi[k] == hi[k - 1]) continue;
      if (box_upper[axis] < lo[k] || box_lower[axis] > hi[k]) continue;
      int multiplicity = 1;
      for (int j = k + 1; j < n + degree + 1 && lo[j] <= hi[k]; ++j) ++multiplicity;
      if (multiplicity < degree) continue;
      if (multiplicity > degree || lo[k] != hi[k]) return false;
      // A line on the box boundary leaves the leaf on one side.
      if (!(box_lower[axis] < lo[k] && lo[k] < box_upper[axis])) continue;
      if (count == kMaxKnotLines) return false;
      lines[count++] = {axis, lo[k]};
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
      return knots(lo, hi, p, nu, 0) && knots(lo + nu + p + 1, hi + nu + p + 1, q, nv, 1);
    }
    case PHX_MC_CAD_NODE_BSPLINE_CURVE:
      return curve_axis >= 0 &&
             knots(lo, hi, static_cast<int>(node.integers[0]),
                   static_cast<int>(node.integers[1]), curve_axis);
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
    case PHX_MC_CAD_NODE_INTERSECTION:
      return false;
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
      const auto du = di::subtract(di::point(hi[0]), di::point(lo[0]));
      const auto dv = di::subtract(di::point(hi[1]), di::point(lo[1]));
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
  // Taylor remainder from |H| magnitudes per term (uu, uv, vv) and component.
  const auto second_order = [&](const double (&hessian)[3][3]) {
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      const auto huu = di::point(hessian[0][axis]);
      const auto huv = di::point(hessian[1][axis]);
      const auto hvv = di::point(hessian[2][axis]);
      auto sum = di::multiply(huu, di::multiply(du, du));
      const auto mixed = di::multiply(huv, di::multiply(du, dv));
      sum = di::add(sum, di::add(mixed, mixed));
      sum = di::add(sum, di::multiply(hvv, di::multiply(dv, dv)));
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
int32_t trim_lens_bound(const CadGeometry& geometry, int32_t face, int64_t use,
                        const double* chord_uv, const double* parameters,
                        IntervalEvaluationWorkspace& workspace, double& cap) {
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
  double derivative_magnitude[3] = {
      di::magnitude({pc_lower[derivative], pc_upper[derivative]}),
      di::magnitude({pc_lower[derivative + 1], pc_upper[derivative + 1]}), 0.0};
  const double derivative_bound = phx::mc::geometry::norm_upper(derivative_magnitude);
  double endpoint_error = 0.0;
  for (int endpoint = 0; endpoint < 2; ++endpoint) {
    double lo[2], hi[2];
    status = geometry.interval_jets(geometry.pcurve_node(use), parameters + endpoint,
                                    parameters + endpoint, 0, lo, hi, workspace);
    if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
    double magnitude[3] = {0.0, 0.0, 0.0};
    for (int axis = 0; axis < 2; ++axis) {
      const auto difference = di::subtract({lo[axis], hi[axis]},
                                           di::point(chord_uv[2 * endpoint + axis]));
      if (!di::finite(difference)) return PHX_MC_UNRESOLVED;
      magnitude[axis] = di::magnitude(difference);
    }
    endpoint_error = std::max(endpoint_error, phx::mc::geometry::norm_upper(magnitude));
  }
  const auto remainder = regularity == 1
      ? di::multiply(di::point(derivative_bound), dt)
      : di::multiply(di::point(0.125),
                     di::multiply(di::point(derivative_bound), di::multiply(dt, dt)));
  const auto error = di::add(di::point(endpoint_error), remainder);
  if (!di::finite(error)) return PHX_MC_UNRESOLVED;
  double box_lower[2], box_upper[2];
  for (int axis = 0; axis < 2; ++axis) {
    box_lower[axis] = di::subtract(di::point(std::min(chord_uv[axis], chord_uv[2 + axis])),
                                  di::point(error.upper)).lower;
    box_upper[axis] = di::add(di::point(std::max(chord_uv[axis], chord_uv[2 + axis])),
                              di::point(error.upper)).upper;
  }
  double surface_lower[9], surface_upper[9];
  status = geometry.interval_jets(geometry.surface_node(face), box_lower, box_upper, 1,
                                  surface_lower, surface_upper, workspace);
  if (status != PHX_MC_OK) return PHX_MC_UNRESOLVED;
  double norms[3] = {};
  for (int derivative = 0; derivative < 2; ++derivative) {
    double magnitude[3];
    for (int axis = 0; axis < 3; ++axis) {
      const int index = 3 * (derivative + 1) + axis;
      magnitude[axis] = di::magnitude({surface_lower[index], surface_upper[index]});
    }
    norms[derivative] = phx::mc::geometry::norm_upper(magnitude);
  }
  cap = di::multiply(di::point(error.upper),
                     di::point(phx::mc::geometry::norm_upper(norms))).upper;
  return std::isfinite(cap) ? PHX_MC_OK : PHX_MC_UNRESOLVED;
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

void chart_mesh_findings(
    const phx_mc_cad_model& model, int64_t point_count, int64_t chart_count,
    const int64_t* chart_triangles, int64_t mesh_count, const int64_t* mesh_triangles,
    const int64_t* chart_to_mesh, const int64_t* edge_chord, const int64_t* chord_use,
    std::vector<int32_t>& pole_edges, std::vector<phx_mc_fidelity_finding>& findings) {
  std::vector<int64_t> owners(static_cast<std::size_t>(mesh_count), -1);
  std::vector<uint8_t> mesh_vertices(static_cast<std::size_t>(point_count), 0);
  for (int64_t i = 0; i < mesh_count * 3; ++i)
    mesh_vertices[static_cast<std::size_t>(mesh_triangles[i])] = 1;
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
          !mesh_vertices[static_cast<std::size_t>(chart_triangles[3 * row + pole_edge])])
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

// Tight mode (pole slivers): a leaf within tolerance is still split while
// its bound exceeds kTightRatio times the largest vertex error seen so far
// (a pointwise value of |S - I_T|, so the returned bound approaches the true
// maximum), up to kTightDepth or kTightSplits; it never refuses for that.
constexpr double kTightRatio = 1.25;
constexpr int32_t kTightDepth = 12;
constexpr int32_t kTightSplits = 4096;

FidelityTriangleResult subdivided_bound(const CadGeometry& geometry, FaceKernel& kernel,
                                        const double* vertices, const double* uv,
                                        const double* corner_error, double tolerance,
                                        int32_t max_depth, bool tight,
                                        IntervalEvaluationWorkspace& workspace) {
  FidelityTriangleResult result{std::numeric_limits<double>::infinity(), PHX_MC_UNRESOLVED,
                                PHX_MC_FIDELITY_INTERVAL_REFUSAL, 0};
  Leaf stack[73];
  stack[0] = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}, 0};
  int size = 1;
  double maximum = 0.0, observed = 0.0;
  while (size > 0) {
    const Leaf leaf = stack[--size];
    double bound = 0.0, vertex_term = 0.0;
    result.status = leaf_bound(geometry, kernel, leaf, vertices, uv, corner_error,
                               tight ? 0.0 : tolerance, workspace,
                               vertex_term, bound);
    if (result.status != PHX_MC_OK) {
      if (result.status == PHX_MC_INVALID_ARGUMENT || result.status == PHX_MC_INVALID_INPUT)
        result.status = PHX_MC_UNRESOLVED;
      return result;
    }
    observed = std::max(observed, vertex_term);
    if (bound <= tolerance &&
        (!tight || bound <= kTightRatio * observed || leaf.depth >= kTightDepth ||
         result.subdivisions >= kTightSplits)) {
      maximum = std::max(maximum, bound);
      continue;
    }
    if (bound > tolerance && (leaf.depth >= max_depth || result.subdivisions >= 1000000)) {
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

// Corner errors |S(uv_j) - X_j| of the original vertices, shared by every
// chart row that has the same mesh vertex at bitwise the same chart point.
struct CornerCache {
  double u = 0.0, v = 0.0, error = 0.0;
  int32_t face = -1;
};
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
                          workspace);
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
    auto findings = phx::mc::cad::association_findings(
        triangle_count, face_rows, uv, chord_count, chord_use, use_patch, chord_uv, edge_chord);
    std::vector<int32_t> pole_edges(static_cast<std::size_t>(triangle_count), -1);
    phx::mc::cad::chart_mesh_findings(*model, point_count, triangle_count, triangles,
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
    int32_t aggregate = PHX_MC_OK;
    double maximum = 0.0;
    std::vector<phx::mc::cad::FaceKernel> kernels(static_cast<std::size_t>(model->surface_count));
    std::vector<phx::mc::cad::CornerCache> corners(static_cast<std::size_t>(point_count));
    for (int64_t i = 0; i < triangle_count; ++i) {
      bool trim_refused = false;
      double lens = 0.0;
      double vertices[9];
      for (int j = 0; j < 3; ++j) {
        const int64_t chord = edge_chord[3 * i + j];
        if (chord >= 0) {
          const int64_t use = chord_use[chord];
          double cap = 0.0;
          if (phx::mc::cad::trim_lens_bound(geometry, face_rows[i], use, chord_uv + 4 * chord,
                                            chord_range + 2 * chord, workspace, cap) != PHX_MC_OK) {
            trim_refused = true;
          } else {
            lens = phx::mc::remesh::add(phx::mc::remesh::point(lens),
                                         phx::mc::remesh::point(cap)).upper;
          }
        }
        const int64_t vertex = triangles[3 * i + j];
        for (int axis = 0; axis < 3; ++axis)
          vertices[3 * j + axis] = points[3 * vertex + axis];
      }
      if (trim_refused || !std::isfinite(lens) || lens >= tolerance) {
        upper[i] = std::numeric_limits<double>::infinity();
        item_status[i] = PHX_MC_UNRESOLVED;
        reasons[i] = trim_refused ? PHX_MC_FIDELITY_TRIM_REFUSAL : PHX_MC_FIDELITY_RESOLUTION;
        subdivisions[i] = 0;
        aggregate = PHX_MC_UNRESOLVED;
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
      const double budget = phx::mc::remesh::subtract(phx::mc::remesh::point(tolerance),
                                                       phx::mc::remesh::point(lens)).lower;
      const auto result = phx::mc::cad::subdivided_bound(
          geometry, kernel, vertices, uv + 6 * i, corners_known ? corner_error : nullptr, budget,
          max_depth, pole_edge >= 0, workspace);
      if (result.status != PHX_MC_OK && result.status != PHX_MC_UNRESOLVED)
        return result.status;
      upper[i] = result.status == PHX_MC_OK
                     ? phx::mc::remesh::add(phx::mc::remesh::point(result.upper),
                                            phx::mc::remesh::point(lens)).upper
                     : result.upper;
      item_status[i] = result.status;
      reasons[i] = result.reason;
      subdivisions[i] = result.subdivisions;
      maximum = std::max(maximum, upper[i]);
      if (result.status != PHX_MC_OK) aggregate = PHX_MC_UNRESOLVED;
    }
    *maximum_upper = aggregate == PHX_MC_OK ? maximum : std::numeric_limits<double>::infinity();
    return aggregate;
  });
}
