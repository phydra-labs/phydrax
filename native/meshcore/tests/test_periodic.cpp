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

#include "check.hpp"
#include "phydrax_meshcore.h"

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
    phx_mc_periodic_copy_cells(handle, result.vertices.data(), result.shifts.data());
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
  PHX_CHECK(result.evidence[3] == 0.0);
  PHX_CHECK(result.evidence[4] <= result.evidence[1]);
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
  PHX_CHECK(result.evidence[7] > 0.0);
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
  PHX_CHECK(images.evidence[8] == 1.0);
  PHX_CHECK(images.evidence[2] > 40.0);

  const Periodic cells = triangulate(2, fractional, lattice, 0.25, 1 << 20, 16);
  PHX_CHECK(cells.status == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(cells.evidence[8] == 2.0);

  const Periodic duplicate =
      triangulate(2, {0.1, 0.2, 0.7, 0.3, 0.1, 0.2}, lattice, 0.25, 1 << 20, 1 << 22);
  PHX_CHECK(duplicate.status == PHX_MC_INVALID_INPUT);
  PHX_CHECK(duplicate.evidence[9] >= 0.0);

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
  phx_mc_periodic_copy_cells(handle, vertices, shifts);
  PHX_CHECK(shifts[0] == -1 && shifts[1] == 0);
  phx_mc_periodic_free(handle);
  handle = nullptr;
  const double translated[4] = {0.25, 0.5, 1.25, 0.5};
  PHX_CHECK(phx_mc_periodic_delaunay(2, 2, translated, translated, identity, identity, 0.25, 100,
                                     100, evidence, &handle) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(handle == nullptr);
  PHX_CHECK(evidence[9] >= 0.0);
  PHX_CHECK(phx_mc_periodic_delaunay(4, 1, point, point, identity, identity, 0.25, 100, 100,
                                     evidence, &handle) == PHX_MC_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  random_torus_2d();
  cocircular_lattice_2d();
  skew_and_single_point_2d();
  random_torus_3d();
  refusals();
  return phx::mc::test::finish("test_periodic");
}
