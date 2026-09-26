//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Floating-point expansion arithmetic after J. R. Shewchuk, "Adaptive Precision
// Floating-Point Arithmetic and Fast Robust Geometric Predicates", Discrete &
// Computational Geometry 18:305-363, 1997.  This is an independent
// implementation of the published algorithms.
//
// Correctness requires IEEE-754 binary64 arithmetic with round-to-nearest-even,
// no excess intermediate precision, and no floating-point contraction.  The
// build compiles every translation unit with -ffp-contract=off; two_product
// uses an explicit fused multiply-add, which is exact by definition.
//
// An expansion is a sum of nonoverlapping doubles stored in increasing order of
// magnitude with zero components eliminated.  Its sign is the sign of its most
// significant (last) component.  Results are exact as long as no intermediate
// product overflows or underflows; callers enforce the documented input domain.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

namespace phx::mc {

inline void fast_two_sum(double a, double b, double& x, double& y) {
  // Requires |a| >= |b| (or a == 0).
  x = a + b;
  const double b_virtual = x - a;
  y = b - b_virtual;
}

inline void two_sum(double a, double b, double& x, double& y) {
  x = a + b;
  const double b_virtual = x - a;
  const double a_virtual = x - b_virtual;
  const double b_round = b - b_virtual;
  const double a_round = a - a_virtual;
  y = a_round + b_round;
}

inline void two_diff(double a, double b, double& x, double& y) {
  x = a - b;
  const double b_virtual = a - x;
  const double a_virtual = x + b_virtual;
  const double b_round = b_virtual - b;
  const double a_round = a - a_virtual;
  y = a_round + b_round;
}

// Roundoff of x = fl(a - b); zero exactly when the difference is representable.
inline double two_diff_tail(double a, double b, double x) {
  const double b_virtual = a - x;
  const double a_virtual = x + b_virtual;
  const double b_round = b_virtual - b;
  const double a_round = a - a_virtual;
  return a_round + b_round;
}

inline void two_product(double a, double b, double& x, double& y) {
  x = a * b;
  y = std::fma(a, b, -x);
}

// Exact (a1 + a0) - (b1 + b0) for two-component expansions (a1, a0) and
// (b1, b0) (Shewchuk's Two_Two_Diff).  h[0..3] is increasing in magnitude and
// may contain zeros.
inline void two_two_diff(double a1, double a0, double b1, double b0, double* h) {
  double i, j, zero;
  two_diff(a0, b0, i, h[0]);
  two_sum(a1, i, j, zero);
  two_diff(zero, b1, i, h[1]);
  two_sum(j, i, h[3], h[2]);
}

// Small-buffer storage for expansion components.  Most predicate expansions
// have fewer than 24 components; larger exact products spill to the heap.
class Expansion {
 public:
  Expansion() = default;
  explicit Expansion(double value) {
    if (value != 0.0) {
      data()[0] = value;
      size_ = 1;
    }
  }
  Expansion(const Expansion& other) { assign(other.data(), other.size_); }
  Expansion(Expansion&& other) noexcept { move_from(std::move(other)); }
  Expansion& operator=(const Expansion& other) {
    if (this != &other) {
      assign(other.data(), other.size_);
    }
    return *this;
  }
  Expansion& operator=(Expansion&& other) noexcept {
    if (this != &other) {
      heap_.reset();
      capacity_ = kInline;
      size_ = 0;
      move_from(std::move(other));
    }
    return *this;
  }

  static Expansion product(double a, double b) {
    Expansion result;
    result.reserve(2);
    double x, y;
    two_product(a, b, x, y);
    result.push_nonzero(y);
    result.push_nonzero(x);
    return result;
  }

  static Expansion difference(double a, double b) {
    Expansion result;
    result.reserve(2);
    double x, y;
    two_diff(a, b, x, y);
    result.push_nonzero(y);
    result.push_nonzero(x);
    return result;
  }

  static Expansion sum(double a, double b) {
    Expansion result;
    result.reserve(2);
    double x, y;
    two_sum(a, b, x, y);
    result.push_nonzero(y);
    result.push_nonzero(x);
    return result;
  }

  // Builds an expansion from possibly-zero, nonoverlapping, increasing components.
  static Expansion from_components(const double* values, int count) {
    Expansion result;
    result.reserve(count);
    for (int index = 0; index < count; ++index) {
      result.push_nonzero(values[index]);
    }
    return result;
  }

  int size() const { return size_; }
  bool is_zero() const { return size_ == 0; }
  const double* data() const { return heap_ ? heap_.get() : inline_; }

  int sign() const {
    if (size_ == 0) {
      return 0;
    }
    return data()[size_ - 1] > 0.0 ? 1 : -1;
  }

  // Approximation of the exact value (Shewchuk's estimate()).
  double estimate() const {
    const double* e = data();
    double total = 0.0;
    for (int index = 0; index < size_; ++index) {
      total += e[index];
    }
    return total;
  }

  Expansion operator-() const {
    Expansion result(*this);
    double* e = result.data();
    for (int index = 0; index < result.size_; ++index) {
      e[index] = -e[index];
    }
    return result;
  }

  // Shewchuk's scale_expansion_zeroelim.
  Expansion scaled(double b) const {
    Expansion result;
    if (size_ == 0 || b == 0.0) {
      return result;
    }
    result.reserve(2 * size_);
    const double* e = data();
    double q, hh, product1, product0, sum;
    two_product(e[0], b, q, hh);
    result.push_nonzero(hh);
    for (int index = 1; index < size_; ++index) {
      two_product(e[index], b, product1, product0);
      two_sum(q, product0, sum, hh);
      result.push_nonzero(hh);
      fast_two_sum(product1, sum, q, hh);
      result.push_nonzero(hh);
    }
    result.push_nonzero(q);
    return result;
  }

  friend Expansion operator+(const Expansion& left, const Expansion& right) {
    return fast_sum(left.data(), left.size_, right.data(), right.size_, false);
  }

  friend Expansion operator-(const Expansion& left, const Expansion& right) {
    return fast_sum(left.data(), left.size_, right.data(), right.size_, true);
  }

  friend Expansion operator*(const Expansion& left, const Expansion& right) {
    if (left.size_ == 0 || right.size_ == 0) {
      return Expansion();
    }
    const Expansion& shorter = left.size_ <= right.size_ ? left : right;
    const Expansion& longer = left.size_ <= right.size_ ? right : left;
    const double* s = shorter.data();
    Expansion total = longer.scaled(s[0]);
    for (int index = 1; index < shorter.size_; ++index) {
      total = total + longer.scaled(s[index]);
    }
    return total;
  }

  Expansion operator*(double b) const { return scaled(b); }

 private:
  static constexpr int kInline = 24;

  double* data() { return heap_ ? heap_.get() : inline_; }

  void reserve(int count) {
    if (count <= capacity_) {
      return;
    }
    std::unique_ptr<double[]> grown(new double[static_cast<std::size_t>(count)]);
    if (size_ > 0) {
      std::memcpy(grown.get(), data(), static_cast<std::size_t>(size_) * sizeof(double));
    }
    heap_ = std::move(grown);
    capacity_ = count;
  }

  void push_nonzero(double value) {
    if (value != 0.0) {
      data()[size_++] = value;
    }
  }

  void assign(const double* values, int count) {
    size_ = 0;
    reserve(count);
    if (count > 0) {
      std::memcpy(data(), values, static_cast<std::size_t>(count) * sizeof(double));
    }
    size_ = count;
  }

  void move_from(Expansion&& other) {
    if (other.heap_) {
      heap_ = std::move(other.heap_);
      capacity_ = other.capacity_;
      size_ = other.size_;
    } else {
      std::memcpy(inline_, other.inline_, static_cast<std::size_t>(other.size_) * sizeof(double));
      size_ = other.size_;
    }
    other.size_ = 0;
    other.capacity_ = kInline;
  }

  // Shewchuk's fast_expansion_sum_zeroelim; negate_right computes e - f.
  static Expansion fast_sum(const double* e, int elen, const double* f, int flen, bool negate_right) {
    Expansion result;
    if (elen == 0 && flen == 0) {
      return result;
    }
    const double sign = negate_right ? -1.0 : 1.0;
    if (elen == 0) {
      result.reserve(flen);
      for (int index = 0; index < flen; ++index) {
        result.push_nonzero(sign * f[index]);
      }
      return result;
    }
    if (flen == 0) {
      result.assign(e, elen);
      return result;
    }
    result.reserve(elen + flen);
    double q, qnew, hh;
    double enow = e[0];
    double fnow = sign * f[0];
    int eindex = 0;
    int findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
      q = enow;
      ++eindex;
      enow = eindex < elen ? e[eindex] : 0.0;
    } else {
      q = fnow;
      ++findex;
      fnow = findex < flen ? sign * f[findex] : 0.0;
    }
    if (eindex < elen && findex < flen) {
      if ((fnow > enow) == (fnow > -enow)) {
        fast_two_sum(enow, q, qnew, hh);
        ++eindex;
        enow = eindex < elen ? e[eindex] : 0.0;
      } else {
        fast_two_sum(fnow, q, qnew, hh);
        ++findex;
        fnow = findex < flen ? sign * f[findex] : 0.0;
      }
      q = qnew;
      result.push_nonzero(hh);
      while (eindex < elen && findex < flen) {
        if ((fnow > enow) == (fnow > -enow)) {
          two_sum(q, enow, qnew, hh);
          ++eindex;
          enow = eindex < elen ? e[eindex] : 0.0;
        } else {
          two_sum(q, fnow, qnew, hh);
          ++findex;
          fnow = findex < flen ? sign * f[findex] : 0.0;
        }
        q = qnew;
        result.push_nonzero(hh);
      }
    }
    while (eindex < elen) {
      two_sum(q, enow, qnew, hh);
      ++eindex;
      enow = eindex < elen ? e[eindex] : 0.0;
      q = qnew;
      result.push_nonzero(hh);
    }
    while (findex < flen) {
      two_sum(q, fnow, qnew, hh);
      ++findex;
      fnow = findex < flen ? sign * f[findex] : 0.0;
      q = qnew;
      result.push_nonzero(hh);
    }
    if (q != 0.0 || result.size_ == 0) {
      result.push_nonzero(q);
    }
    return result;
  }

  double inline_[kInline];
  std::unique_ptr<double[]> heap_;
  int size_ = 0;
  int capacity_ = kInline;
};

}  // namespace phx::mc
