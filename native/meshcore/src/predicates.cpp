//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "predicates.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace phx::mc {
namespace {

// Shewchuk's error-bound constants; epsilon is the unit roundoff 2^-53.
constexpr double kEpsilon = 0x1p-53;
constexpr double kCcwErrBoundA = (3.0 + 16.0 * kEpsilon) * kEpsilon;
constexpr double kCcwErrBoundB = (2.0 + 12.0 * kEpsilon) * kEpsilon;
constexpr double kO3dErrBoundA = (7.0 + 56.0 * kEpsilon) * kEpsilon;
constexpr double kO3dErrBoundB = (3.0 + 28.0 * kEpsilon) * kEpsilon;
constexpr double kIccErrBoundA = (10.0 + 96.0 * kEpsilon) * kEpsilon;
constexpr double kIccErrBoundB = (4.0 + 48.0 * kEpsilon) * kEpsilon;
constexpr double kIspErrBoundA = (16.0 + 224.0 * kEpsilon) * kEpsilon;
constexpr double kIspErrBoundB = (5.0 + 72.0 * kEpsilon) * kEpsilon;

inline int sign_of(double value) { return (value > 0.0) - (value < 0.0); }

// Shewchuk's orient3d value det[a - d; b - d; c - d] on difference rows.
template <class T>
T orient3d_rows(const T* x, const T* y, const T* z) {
  return z[0] * (x[1] * y[2] - x[2] * y[1]) + z[1] * (x[2] * y[0] - x[0] * y[2]) +
         z[2] * (x[0] * y[1] - x[1] * y[0]);
}

// det[[x_k, y_k, lift_k]] for k = 0, 1, 2 (incircle/power2d value).
template <class T>
T lifted2_rows(const T* x, const T* y, const T* lift) {
  return lift[0] * (x[1] * y[2] - x[2] * y[1]) + lift[1] * (x[2] * y[0] - x[0] * y[2]) +
         lift[2] * (x[0] * y[1] - x[1] * y[0]);
}

// Shewchuk's insphere value det[[x_k, y_k, z_k, lift_k]] for k = 0..3.
template <class T>
T lifted3_rows(const T* x, const T* y, const T* z, const T* lift) {
  const T ab = x[0] * y[1] - x[1] * y[0];
  const T bc = x[1] * y[2] - x[2] * y[1];
  const T cd = x[2] * y[3] - x[3] * y[2];
  const T da = x[3] * y[0] - x[0] * y[3];
  const T ac = x[0] * y[2] - x[2] * y[0];
  const T bd = x[1] * y[3] - x[3] * y[1];
  const T abc = z[0] * bc - z[1] * ac + z[2] * ab;
  const T bcd = z[1] * cd - z[2] * bd + z[3] * bc;
  const T cda = z[2] * da + z[3] * ac + z[0] * cd;
  const T dab = z[3] * ab + z[0] * bd + z[1] * da;
  return (lift[3] * abc - lift[2] * dab) + (lift[1] * cda - lift[0] * bcd);
}

// ---------------------------------------------------------------- orient2d

int orient2d_adapt(const double* a, const double* b, const double* c, double detsum) {
  const double acx = a[0] - c[0];
  const double bcx = b[0] - c[0];
  const double acy = a[1] - c[1];
  const double bcy = b[1] - c[1];
  double detleft, detlefttail, detright, detrighttail;
  two_product(acx, bcy, detleft, detlefttail);
  two_product(acy, bcx, detright, detrighttail);
  double b_components[4];
  two_two_diff(detleft, detlefttail, detright, detrighttail, b_components);
  const Expansion stage_b = Expansion::from_components(b_components, 4);
  const double det = stage_b.estimate();
  const double errbound = kCcwErrBoundB * detsum;
  if (det >= errbound || -det >= errbound) {
    return sign_of(det);
  }
  if (two_diff_tail(a[0], c[0], acx) == 0.0 && two_diff_tail(b[0], c[0], bcx) == 0.0 &&
      two_diff_tail(a[1], c[1], acy) == 0.0 && two_diff_tail(b[1], c[1], bcy) == 0.0) {
    return stage_b.sign();
  }
  const Expansion ex = Expansion::difference(a[0], c[0]);
  const Expansion ey = Expansion::difference(a[1], c[1]);
  const Expansion fx = Expansion::difference(b[0], c[0]);
  const Expansion fy = Expansion::difference(b[1], c[1]);
  return (ex * fy - ey * fx).sign();
}

// ---------------------------------------------------------------- orient3d (Shewchuk sign)

int orient3d_shewchuk(const double* a, const double* b, const double* c, const double* d) {
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double adz = a[2] - d[2], bdz = b[2] - d[2], cdz = c[2] - d[2];
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
  const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * std::fabs(adz) +
                           (std::fabs(cdxady) + std::fabs(adxcdy)) * std::fabs(bdz) +
                           (std::fabs(adxbdy) + std::fabs(bdxady)) * std::fabs(cdz);
  const double errbound_a = kO3dErrBoundA * permanent;
  if (det > errbound_a || -det > errbound_a) {
    return sign_of(det);
  }
  const Expansion rx[3] = {Expansion(adx), Expansion(bdx), Expansion(cdx)};
  const Expansion ry[3] = {Expansion(ady), Expansion(bdy), Expansion(cdy)};
  const Expansion rz[3] = {Expansion(adz), Expansion(bdz), Expansion(cdz)};
  const Expansion stage_b = orient3d_rows(rx, ry, rz);
  const double estimate = stage_b.estimate();
  const double errbound_b = kO3dErrBoundB * permanent;
  if (estimate >= errbound_b || -estimate >= errbound_b) {
    return sign_of(estimate);
  }
  const bool exact_differences =
      two_diff_tail(a[0], d[0], adx) == 0.0 && two_diff_tail(b[0], d[0], bdx) == 0.0 &&
      two_diff_tail(c[0], d[0], cdx) == 0.0 && two_diff_tail(a[1], d[1], ady) == 0.0 &&
      two_diff_tail(b[1], d[1], bdy) == 0.0 && two_diff_tail(c[1], d[1], cdy) == 0.0 &&
      two_diff_tail(a[2], d[2], adz) == 0.0 && two_diff_tail(b[2], d[2], bdz) == 0.0 &&
      two_diff_tail(c[2], d[2], cdz) == 0.0;
  if (exact_differences) {
    return stage_b.sign();
  }
  const Expansion fx[3] = {Expansion::difference(a[0], d[0]), Expansion::difference(b[0], d[0]),
                           Expansion::difference(c[0], d[0])};
  const Expansion fy[3] = {Expansion::difference(a[1], d[1]), Expansion::difference(b[1], d[1]),
                           Expansion::difference(c[1], d[1])};
  const Expansion fz[3] = {Expansion::difference(a[2], d[2]), Expansion::difference(b[2], d[2]),
                           Expansion::difference(c[2], d[2])};
  return orient3d_rows(fx, fy, fz).sign();
}

// ---------------------------------------------------------------- incircle

int incircle_adapt(const double* const* p, double permanent) {
  const double* d = p[3];
  double x[3], y[3];
  bool exact_differences = true;
  for (int k = 0; k < 3; ++k) {
    x[k] = p[k][0] - d[0];
    y[k] = p[k][1] - d[1];
    exact_differences = exact_differences && two_diff_tail(p[k][0], d[0], x[k]) == 0.0 &&
                        two_diff_tail(p[k][1], d[1], y[k]) == 0.0;
  }
  Expansion rx[3], ry[3], lift[3];
  for (int k = 0; k < 3; ++k) {
    rx[k] = Expansion(x[k]);
    ry[k] = Expansion(y[k]);
    lift[k] = rx[k] * rx[k] + ry[k] * ry[k];
  }
  const Expansion stage_b = lifted2_rows(rx, ry, lift);
  const double estimate = stage_b.estimate();
  const double errbound = kIccErrBoundB * permanent;
  if (estimate >= errbound || -estimate >= errbound) {
    return sign_of(estimate);
  }
  if (exact_differences) {
    return stage_b.sign();
  }
  for (int k = 0; k < 3; ++k) {
    rx[k] = Expansion::difference(p[k][0], d[0]);
    ry[k] = Expansion::difference(p[k][1], d[1]);
    lift[k] = rx[k] * rx[k] + ry[k] * ry[k];
  }
  return lifted2_rows(rx, ry, lift).sign();
}

// ---------------------------------------------------------------- insphere (Shewchuk sign)

int insphere_adapt(const double* const* p, double permanent) {
  const double* e = p[4];
  double x[4], y[4], z[4];
  bool exact_differences = true;
  for (int k = 0; k < 4; ++k) {
    x[k] = p[k][0] - e[0];
    y[k] = p[k][1] - e[1];
    z[k] = p[k][2] - e[2];
    exact_differences = exact_differences && two_diff_tail(p[k][0], e[0], x[k]) == 0.0 &&
                        two_diff_tail(p[k][1], e[1], y[k]) == 0.0 &&
                        two_diff_tail(p[k][2], e[2], z[k]) == 0.0;
  }
  Expansion rx[4], ry[4], rz[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    rx[k] = Expansion(x[k]);
    ry[k] = Expansion(y[k]);
    rz[k] = Expansion(z[k]);
    lift[k] = rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k];
  }
  const Expansion stage_b = lifted3_rows(rx, ry, rz, lift);
  const double estimate = stage_b.estimate();
  const double errbound = kIspErrBoundB * permanent;
  if (estimate >= errbound || -estimate >= errbound) {
    return sign_of(estimate);
  }
  if (exact_differences) {
    return stage_b.sign();
  }
  for (int k = 0; k < 4; ++k) {
    rx[k] = Expansion::difference(p[k][0], e[0]);
    ry[k] = Expansion::difference(p[k][1], e[1]);
    rz[k] = Expansion::difference(p[k][2], e[2]);
    lift[k] = rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k];
  }
  return lifted3_rows(rx, ry, rz, lift).sign();
}

int insphere_shewchuk(const double* const* p) {
  const double* e = p[4];
  const double aex = p[0][0] - e[0], bex = p[1][0] - e[0], cex = p[2][0] - e[0], dex = p[3][0] - e[0];
  const double aey = p[0][1] - e[1], bey = p[1][1] - e[1], cey = p[2][1] - e[1], dey = p[3][1] - e[1];
  const double aez = p[0][2] - e[2], bez = p[1][2] - e[2], cez = p[2][2] - e[2], dez = p[3][2] - e[2];

  const double aexbey = aex * bey, bexaey = bex * aey;
  const double ab = aexbey - bexaey;
  const double bexcey = bex * cey, cexbey = cex * bey;
  const double bc = bexcey - cexbey;
  const double cexdey = cex * dey, dexcey = dex * cey;
  const double cd = cexdey - dexcey;
  const double dexaey = dex * aey, aexdey = aex * dey;
  const double da = dexaey - aexdey;
  const double aexcey = aex * cey, cexaey = cex * aey;
  const double ac = aexcey - cexaey;
  const double bexdey = bex * dey, dexbey = dex * bey;
  const double bd = bexdey - dexbey;

  const double abc = aez * bc - bez * ac + cez * ab;
  const double bcd = bez * cd - cez * bd + dez * bc;
  const double cda = cez * da + dez * ac + aez * cd;
  const double dab = dez * ab + aez * bd + bez * da;

  const double alift = aex * aex + aey * aey + aez * aez;
  const double blift = bex * bex + bey * bey + bez * bez;
  const double clift = cex * cex + cey * cey + cez * cez;
  const double dlift = dex * dex + dey * dey + dez * dez;

  const double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

  const double aezplus = std::fabs(aez), bezplus = std::fabs(bez);
  const double cezplus = std::fabs(cez), dezplus = std::fabs(dez);
  const double aexbeyplus = std::fabs(aexbey), bexaeyplus = std::fabs(bexaey);
  const double bexceyplus = std::fabs(bexcey), cexbeyplus = std::fabs(cexbey);
  const double cexdeyplus = std::fabs(cexdey), dexceyplus = std::fabs(dexcey);
  const double dexaeyplus = std::fabs(dexaey), aexdeyplus = std::fabs(aexdey);
  const double aexceyplus = std::fabs(aexcey), cexaeyplus = std::fabs(cexaey);
  const double bexdeyplus = std::fabs(bexdey), dexbeyplus = std::fabs(dexbey);
  const double permanent =
      ((cexdeyplus + dexceyplus) * bezplus + (dexbeyplus + bexdeyplus) * cezplus +
       (bexceyplus + cexbeyplus) * dezplus) *
          alift +
      ((dexaeyplus + aexdeyplus) * cezplus + (aexceyplus + cexaeyplus) * dezplus +
       (cexdeyplus + dexceyplus) * aezplus) *
          blift +
      ((aexbeyplus + bexaeyplus) * dezplus + (bexdeyplus + dexbeyplus) * aezplus +
       (dexaeyplus + aexdeyplus) * bezplus) *
          clift +
      ((bexceyplus + cexbeyplus) * aezplus + (cexaeyplus + aexceyplus) * bezplus +
       (aexbeyplus + bexaeyplus) * cezplus) *
          dlift;
  const double errbound = kIspErrBoundA * permanent;
  if (det > errbound || -det > errbound) {
    return sign_of(det);
  }
  return insphere_adapt(p, permanent);
}

// ---------------------------------------------------------------- weighted power tests

template <class T, class Diff>
T power2d_value(const double* const* p, const double* w, Diff diff) {
  T x[3], y[3], lift[3];
  for (int k = 0; k < 3; ++k) {
    x[k] = diff(p[k][0], p[3][0]);
    y[k] = diff(p[k][1], p[3][1]);
    lift[k] = x[k] * x[k] + y[k] * y[k] - diff(w[k], w[3]);
  }
  return lifted2_rows(x, y, lift);
}

template <class T, class Diff>
T power3d_shewchuk_value(const double* const* p, const double* w, Diff diff) {
  T x[4], y[4], z[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    x[k] = diff(p[k][0], p[4][0]);
    y[k] = diff(p[k][1], p[4][1]);
    z[k] = diff(p[k][2], p[4][2]);
    lift[k] = x[k] * x[k] + y[k] * y[k] + z[k] * z[k] - diff(w[k], w[4]);
  }
  return lifted3_rows(x, y, z, lift);
}

Approx approx_difference(double a, double b) { return Approx::exact(a) - Approx::exact(b); }
Expansion exact_difference(double a, double b) { return Expansion::difference(a, b); }

int power2d_impl(const double* const* p, const double* w) {
  const Approx filtered = power2d_value<Approx>(p, w, approx_difference);
  const int sign = filtered.certified_sign();
  if (sign != 2) {
    return sign;
  }
  return power2d_value<Expansion>(p, w, exact_difference).sign();
}

int power3d_impl(const double* const* p, const double* w) {
  const Approx filtered = power3d_shewchuk_value<Approx>(p, w, approx_difference);
  const int sign = filtered.certified_sign();
  if (sign != 2) {
    return -sign;
  }
  return -power3d_shewchuk_value<Expansion>(p, w, exact_difference).sign();
}

// ---------------------------------------------------------------- Simulation of Simplicity

// Exact determinant of the n x n submatrix of m selected by rows[0..n) and
// cols[0..n), by Laplace expansion along the first selected row.
Expansion determinant_exact(const double m[4][4], const int* rows, const int* cols, int n) {
  if (n == 1) {
    return Expansion(m[rows[0]][cols[0]]);
  }
  Expansion total;
  int minor_cols[4];
  for (int j = 0; j < n; ++j) {
    const double entry = m[rows[0]][cols[j]];
    if (entry == 0.0) {
      continue;
    }
    int count = 0;
    for (int k = 0; k < n; ++k) {
      if (k != j) {
        minor_cols[count++] = cols[k];
      }
    }
    const Expansion minor = determinant_exact(m, rows + 1, minor_cols, n - 1).scaled(entry);
    total = (j % 2 == 0) ? total + minor : total - minor;
  }
  return total;
}

struct SosMonomial {
  int count;
  int rank[3];
  int column[3];
};

// Perturbation monomials of the (d + 1) x (d + 1) orientation matrix with rows
// [p_r, 1] in rank order, sorted by decreasing significance, ending with the
// first full matching (whose coefficient is +-1 and therefore nonzero).
template <int Dimension>
struct SosTable {
  static constexpr std::size_t kMaximum = Dimension == 2 ? 12 : 72;
  std::array<SosMonomial, kMaximum> values{};
  std::size_t count = 0;
};

template <int Dimension>
constexpr SosTable<Dimension> build_sos_monomials() {
  constexpr int dimension = Dimension;
  struct Keyed {
    std::uint64_t key;
    SosMonomial monomial;
  };
  std::array<Keyed, SosTable<Dimension>::kMaximum> all{};
  std::size_t all_count = 0;
  const int points = dimension + 1;
  // Enumerate injective partial maps rank -> column with 1..dimension pairs.
  const int total_masks = 1 << points;
  for (int mask = 1; mask < total_masks; ++mask) {
    int ranks[4] = {};
    int count = 0;
    for (int r = 0; r < points; ++r) {
      if (mask & (1 << r)) {
        ranks[count++] = r;
      }
    }
    if (count > dimension) {
      continue;
    }
    std::array<int, 3> columns = {0, 1, 2};
    // Iterate over all ordered selections of `count` distinct columns.
    std::array<int, 3> selection{};
    const int combinations = [&] {
      int value = 1;
      for (int k = 0; k < count; ++k) {
        value *= dimension;
      }
      return value;
    }();
    for (int code = 0; code < combinations; ++code) {
      int remainder = code;
      bool distinct = true;
      int used = 0;
      for (int k = 0; k < count; ++k) {
        selection[static_cast<std::size_t>(k)] = columns[static_cast<std::size_t>(remainder % dimension)];
        remainder /= dimension;
        const int bit = 1 << selection[static_cast<std::size_t>(k)];
        if (used & bit) {
          distinct = false;
        }
        used |= bit;
      }
      if (!distinct) {
        continue;
      }
      Keyed keyed{};
      keyed.monomial.count = count;
      for (int k = 0; k < count; ++k) {
        const int r = ranks[k];
        const int j = selection[static_cast<std::size_t>(k)];
        keyed.monomial.rank[k] = r;
        keyed.monomial.column[k] = j;
        keyed.key |= std::uint64_t{1} << (dimension * (r + 1) - (j + 1));
      }
      all[all_count++] = keyed;
    }
  }
  std::sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(all_count),
            [](const Keyed& left, const Keyed& right) { return left.key < right.key; });
  SosTable<Dimension> result;
  for (std::size_t i = 0; i < all_count; ++i) {
    const Keyed& keyed = all[i];
    result.values[result.count++] = keyed.monomial;
    if (keyed.monomial.count == dimension) {
      break;
    }
  }
  return result;
}

std::span<const SosMonomial> sos_monomials(int dimension) {
  // Fixed integer tables cannot capture a caller's bounded allocation pool.
  static constexpr auto two = build_sos_monomials<2>();
  static constexpr auto three = build_sos_monomials<3>();
  return dimension == 2
             ? std::span<const SosMonomial>(two.values.data(), two.count)
             : std::span<const SosMonomial>(three.values.data(), three.count);
}

// Sorts point slots by index; returns the permutation parity (+1/-1).
int rank_order(const std::int64_t* ids, int count, int* order) {
  for (int k = 0; k < count; ++k) {
    order[k] = k;
  }
  int parity = 1;
  for (int i = 1; i < count; ++i) {
    for (int j = i; j > 0 && ids[order[j - 1]] > ids[order[j]]; --j) {
      std::swap(order[j - 1], order[j]);
      parity = -parity;
    }
  }
  return parity;
}

// Sign of det of the perturbed matrix with rows [p_rank, 1] in rank order.
int orientation_sos_sign(const double* const* points, const std::int64_t* ids, int dimension,
                         int* parity_out) {
  int order[4];
  const int points_count = dimension + 1;
  *parity_out = rank_order(ids, points_count, order);
  double base[4][4];
  for (int r = 0; r < points_count; ++r) {
    for (int j = 0; j < dimension; ++j) {
      base[r][j] = points[order[r]][j];
    }
    base[r][dimension] = 1.0;
  }
  const int all_indices[4] = {0, 1, 2, 3};
  for (const SosMonomial& monomial : sos_monomials(dimension)) {
    double matrix[4][4];
    for (int r = 0; r < points_count; ++r) {
      for (int j = 0; j <= dimension; ++j) {
        matrix[r][j] = base[r][j];
      }
    }
    for (int k = 0; k < monomial.count; ++k) {
      const int r = monomial.rank[k];
      for (int j = 0; j <= dimension; ++j) {
        matrix[r][j] = j == monomial.column[k] ? 1.0 : 0.0;
      }
    }
    const int sign = determinant_exact(matrix, all_indices, all_indices, points_count).sign();
    if (sign != 0) {
      return sign;
    }
  }
  return 0;
}

// Lifted-coordinate SoS in 2D: cofactor of row t's lift is (-1)^t orient2d(others).
int lifted_sos_2d(const double* const* p, const std::int64_t* ids) {
  int order[4];
  rank_order(ids, 4, order);
  for (int k = 0; k < 4; ++k) {
    const int t = order[k];
    const double* others[3];
    int count = 0;
    for (int s = 0; s < 4; ++s) {
      if (s != t) {
        others[count++] = p[s];
      }
    }
    const int cofactor = orient2d(others[0], others[1], others[2]);
    if (cofactor != 0) {
      return (t % 2 == 0) ? cofactor : -cofactor;
    }
  }
  return 0;
}

// Lifted-coordinate SoS in 3D: the result is -(-1)^t orient3d(others).
int lifted_sos_3d(const double* const* p, const std::int64_t* ids) {
  int order[5];
  rank_order(ids, 5, order);
  for (int k = 0; k < 5; ++k) {
    const int t = order[k];
    const double* others[4];
    int count = 0;
    for (int s = 0; s < 5; ++s) {
      if (s != t) {
        others[count++] = p[s];
      }
    }
    const int cofactor = orient3d(others[0], others[1], others[2], others[3]);
    if (cofactor != 0) {
      return (t % 2 == 0) ? -cofactor : cofactor;
    }
  }
  return 0;
}

}  // namespace

// ---------------------------------------------------------------- public predicates

int orient2d(const double* a, const double* b, const double* c) {
  native_execution_primitive_query();
  const double detleft = (a[0] - c[0]) * (b[1] - c[1]);
  const double detright = (a[1] - c[1]) * (b[0] - c[0]);
  const double det = detleft - detright;
  double detsum;
  if (detleft > 0.0) {
    if (detright <= 0.0) {
      return sign_of(det);
    }
    detsum = detleft + detright;
  } else if (detleft < 0.0) {
    if (detright >= 0.0) {
      return sign_of(det);
    }
    detsum = -detleft - detright;
  } else {
    return sign_of(det);
  }
  const double errbound = kCcwErrBoundA * detsum;
  if (det >= errbound || -det >= errbound) {
    return sign_of(det);
  }
  return orient2d_adapt(a, b, c, detsum);
}

int orient3d(const double* a, const double* b, const double* c, const double* d) {
  native_execution_primitive_query();
  return -orient3d_shewchuk(a, b, c, d);
}

int incircle(const double* a, const double* b, const double* c, const double* d) {
  native_execution_primitive_query();
  const double adx = a[0] - d[0], bdx = b[0] - d[0], cdx = c[0] - d[0];
  const double ady = a[1] - d[1], bdy = b[1] - d[1], cdy = c[1] - d[1];
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double alift = adx * adx + ady * ady;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double blift = bdx * bdx + bdy * bdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double clift = cdx * cdx + cdy * cdy;
  const double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
  const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift +
                           (std::fabs(cdxady) + std::fabs(adxcdy)) * blift +
                           (std::fabs(adxbdy) + std::fabs(bdxady)) * clift;
  const double errbound = kIccErrBoundA * permanent;
  if (det > errbound || -det > errbound) {
    return sign_of(det);
  }
  const double* p[4] = {a, b, c, d};
  return incircle_adapt(p, permanent);
}

int insphere(const double* a, const double* b, const double* c, const double* d, const double* e) {
  native_execution_primitive_query();
  const double* p[5] = {a, b, c, d, e};
  return -insphere_shewchuk(p);
}

int insphere_expansion(const Expansion* a, const Expansion* b, const Expansion* c,
                       const Expansion* d, const Expansion* e) {
  native_execution_primitive_query();
  const Expansion* points[4] = {a, b, c, d};
  Expansion x[4], y[4], z[4], lift[4];
  for (int k = 0; k < 4; ++k) {
    x[k] = points[k][0] - e[0];
    y[k] = points[k][1] - e[1];
    z[k] = points[k][2] - e[2];
    lift[k] = x[k] * x[k] + y[k] * y[k] + z[k] * z[k];
  }
  return -lifted3_rows(x, y, z, lift).sign();
}

int power2d(const double* a, const double* b, const double* c, const double* d, double wa,
            double wb, double wc, double wd) {
  native_execution_primitive_query();
  const double* p[4] = {a, b, c, d};
  const double w[4] = {wa, wb, wc, wd};
  return power2d_impl(p, w);
}

int power3d(const double* a, const double* b, const double* c, const double* d, const double* e,
            double wa, double wb, double wc, double wd, double we) {
  native_execution_primitive_query();
  const double* p[5] = {a, b, c, d, e};
  const double w[5] = {wa, wb, wc, wd, we};
  return power3d_impl(p, w);
}

int orient2d_sos(const double* a, const double* b, const double* c, std::int64_t ia,
                 std::int64_t ib, std::int64_t ic) {
  const int base = orient2d(a, b, c);
  if (base != 0) {
    return base;
  }
  const double* p[3] = {a, b, c};
  const std::int64_t ids[3] = {ia, ib, ic};
  int parity = 1;
  const int sign = orientation_sos_sign(p, ids, 2, &parity);
  return parity * sign;
}

int orient3d_sos(const double* a, const double* b, const double* c, const double* d,
                 std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id) {
  const int base = orient3d(a, b, c, d);
  if (base != 0) {
    return base;
  }
  const double* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  int parity = 1;
  const int sign = orientation_sos_sign(p, ids, 3, &parity);
  // orient3d(a, b, c, d) = -det[[a, 1]; [b, 1]; [c, 1]; [d, 1]].
  return -parity * sign;
}

int incircle_sos(const double* a, const double* b, const double* c, const double* d,
                 std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id) {
  const int base = incircle(a, b, c, d);
  if (base != 0) {
    return base;
  }
  const double* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  return lifted_sos_2d(p, ids);
}

int insphere_sos(const double* a, const double* b, const double* c, const double* d,
                 const double* e, std::int64_t ia, std::int64_t ib, std::int64_t ic,
                 std::int64_t id, std::int64_t ie) {
  const int base = insphere(a, b, c, d, e);
  if (base != 0) {
    return base;
  }
  const double* p[5] = {a, b, c, d, e};
  const std::int64_t ids[5] = {ia, ib, ic, id, ie};
  return lifted_sos_3d(p, ids);
}

int power2d_sos(const double* a, const double* b, const double* c, const double* d, double wa,
                double wb, double wc, double wd, std::int64_t ia, std::int64_t ib,
                std::int64_t ic, std::int64_t id) {
  const int base = power2d(a, b, c, d, wa, wb, wc, wd);
  if (base != 0) {
    return base;
  }
  const double* p[4] = {a, b, c, d};
  const std::int64_t ids[4] = {ia, ib, ic, id};
  return lifted_sos_2d(p, ids);
}

int power3d_sos(const double* a, const double* b, const double* c, const double* d,
                const double* e, double wa, double wb, double wc, double wd, double we,
                std::int64_t ia, std::int64_t ib, std::int64_t ic, std::int64_t id,
                std::int64_t ie) {
  const int base = power3d(a, b, c, d, e, wa, wb, wc, wd, we);
  if (base != 0) {
    return base;
  }
  const double* p[5] = {a, b, c, d, e};
  const std::int64_t ids[5] = {ia, ib, ic, id, ie};
  return lifted_sos_3d(p, ids);
}

Expansion orient2d_exact(const double* a, const double* b, const double* c) {
  native_execution_primitive_query();
  const Expansion bx = Expansion::difference(b[0], a[0]);
  const Expansion by = Expansion::difference(b[1], a[1]);
  const Expansion cx = Expansion::difference(c[0], a[0]);
  const Expansion cy = Expansion::difference(c[1], a[1]);
  return bx * cy - by * cx;
}

Expansion orient3d_exact(const double* a, const double* b, const double* c, const double* d) {
  native_execution_primitive_query();
  const Expansion x[3] = {Expansion::difference(a[0], d[0]), Expansion::difference(b[0], d[0]),
                          Expansion::difference(c[0], d[0])};
  const Expansion y[3] = {Expansion::difference(a[1], d[1]), Expansion::difference(b[1], d[1]),
                          Expansion::difference(c[1], d[1])};
  const Expansion z[3] = {Expansion::difference(a[2], d[2]), Expansion::difference(b[2], d[2]),
                          Expansion::difference(c[2], d[2])};
  return -orient3d_rows(x, y, z);
}

Approx orient2d_approx(const double* a, const double* b, const double* c) {
  const Approx bx = approx_difference(b[0], a[0]);
  const Approx by = approx_difference(b[1], a[1]);
  const Approx cx = approx_difference(c[0], a[0]);
  const Approx cy = approx_difference(c[1], a[1]);
  return bx * cy - by * cx;
}

Approx orient3d_approx(const double* a, const double* b, const double* c, const double* d) {
  const Approx x[3] = {approx_difference(a[0], d[0]), approx_difference(b[0], d[0]),
                       approx_difference(c[0], d[0])};
  const Approx y[3] = {approx_difference(a[1], d[1]), approx_difference(b[1], d[1]),
                       approx_difference(c[1], d[1])};
  const Approx z[3] = {approx_difference(a[2], d[2]), approx_difference(b[2], d[2]),
                       approx_difference(c[2], d[2])};
  return -orient3d_rows(x, y, z);
}

}  // namespace phx::mc
