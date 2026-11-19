//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral mesh of {f <= 0} by isosurface stuffing on a body-centered cubic
// (BCC) lattice (Labelle and Shewchuk, "Isosurface Stuffing", SIGGRAPH 2007):
//
//   1. f at every lattice vertex (corner grid A and center grid B), evaluated
//      in binary64 through the point executor of an interval program;
//   2. a cut point on every lattice edge whose endpoints have strictly
//      opposite signs, by safeguarded regula falsi (Illinois) in batches;
//   3. warping: a lattice vertex within alpha of a cut point on one of its
//      edges moves to the nearest such cut point and becomes a zero vertex
//      (alpha = 0.24999 on long edges, 0.41189 on short edges);
//   4. per lattice tetrahedron, the part with f <= 0 of the piecewise-linear
//      interpolant (a convex polytope with at most six vertices) is split into
//      tetrahedra. Shared quadrilateral faces are split through their vertex of
//      smallest global identity on both sides, which makes neighbouring
//      tetrahedra conform; prisms and pyramids follow the same rule
//      (Dompierre et al., "How to subdivide pyramids, prisms and hexahedra").
//
// Construction only: floating-point evaluation proposes, nothing here is a
// certificate. The volume certificate and the source-fidelity bounds run on
// the result afterwards.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "capi_guard.hpp"
#include "mesh.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {

constexpr double kAlphaLong = 0.24999;
constexpr double kAlphaShort = 0.41189;
constexpr int kRootIterations = 48;

struct Program {
  const int64_t* instructions;
  int64_t instruction_count;
  const int64_t* indices;
  int64_t index_count;
  const double* parameters;
  int64_t parameter_count;
  const double* seeds;
  int64_t workspace_size;
  int64_t output_entry;
};

// f at points (count, 3); status of the point executor.
int32_t evaluate(const Program& program, const std::vector<double>& points,
                 std::vector<double>& values) {
  const int64_t count = static_cast<int64_t>(points.size() / 3);
  values.resize(static_cast<std::size_t>(count));
  if (count == 0) return PHX_MC_OK;
  return phx_mc_program_evaluate(program.instructions, program.instruction_count,
                                 program.indices, program.index_count, program.parameters,
                                 program.parameter_count, program.seeds, program.workspace_size,
                                 3, points.data(), count, &program.output_entry, 1,
                                 values.data());
}

int sign_of(double value) { return value < 0.0 ? -1 : (value > 0.0 ? 1 : 0); }

class Lattice {
 public:
  Lattice(const double* lower, const double* upper, double spacing) : spacing_(spacing) {
    for (int d = 0; d < 3; ++d) {
      // One padding cell on each side keeps the zero set inside the lattice.
      const double extent = upper[d] - lower[d];
      cells_[d] = static_cast<int64_t>(std::ceil(extent / spacing)) + 2;
      origin_[d] = 0.5 * (lower[d] + upper[d]) - 0.5 * static_cast<double>(cells_[d]) * spacing;
    }
  }
  int64_t cells(int d) const { return cells_[d]; }
  int64_t corner_count() const { return (cells_[0] + 1) * (cells_[1] + 1) * (cells_[2] + 1); }
  int64_t center_count() const { return cells_[0] * cells_[1] * cells_[2]; }
  int64_t corner(int64_t i, int64_t j, int64_t k) const {
    return (i * (cells_[1] + 1) + j) * (cells_[2] + 1) + k;
  }
  int64_t center(int64_t i, int64_t j, int64_t k) const {
    return corner_count() + (i * cells_[1] + j) * cells_[2] + k;
  }
  void corner_point(int64_t i, int64_t j, int64_t k, double* p) const {
    p[0] = origin_[0] + spacing_ * static_cast<double>(i);
    p[1] = origin_[1] + spacing_ * static_cast<double>(j);
    p[2] = origin_[2] + spacing_ * static_cast<double>(k);
  }
  void center_point(int64_t i, int64_t j, int64_t k, double* p) const {
    p[0] = origin_[0] + spacing_ * (static_cast<double>(i) + 0.5);
    p[1] = origin_[1] + spacing_ * (static_cast<double>(j) + 0.5);
    p[2] = origin_[2] + spacing_ * (static_cast<double>(k) + 0.5);
  }

 private:
  double spacing_;
  std::array<int64_t, 3> cells_{};
  std::array<double, 3> origin_{};
};

// One lattice edge with a sign change and its cut point.
struct Cut {
  int64_t negative;  // lattice vertex with f < 0
  int64_t positive;  // lattice vertex with f > 0
  double t;          // position: negative + t (positive - negative)
  bool long_edge;
};

std::uint64_t edge_key(int64_t a, int64_t b) {
  const auto low = static_cast<std::uint64_t>(std::min(a, b));
  const auto high = static_cast<std::uint64_t>(std::max(a, b));
  return (low << 32) | high;
}

// Open-addressing map from edge keys to cut indices (keys fit 32 bits per end).
class EdgeMap {
 public:
  explicit EdgeMap(std::size_t expected) {
    std::size_t capacity = 16;
    while (capacity < 2 * expected + 16) capacity *= 2;
    keys_.assign(capacity, kEmpty);
    values_.assign(capacity, -1);
  }
  int64_t& slot(std::uint64_t key) {
    std::size_t index = hash(key) & (keys_.size() - 1);
    while (keys_[index] != kEmpty && keys_[index] != key) index = (index + 1) & (keys_.size() - 1);
    if (keys_[index] == kEmpty) keys_[index] = key;
    return values_[index];
  }
  int64_t find(std::uint64_t key) const {
    std::size_t index = hash(key) & (keys_.size() - 1);
    while (keys_[index] != kEmpty) {
      if (keys_[index] == key) return values_[index];
      index = (index + 1) & (keys_.size() - 1);
    }
    return -1;
  }

 private:
  static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
  static std::size_t hash(std::uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return static_cast<std::size_t>(key);
  }
  std::vector<std::uint64_t> keys_;
  std::vector<int64_t> values_;
};

using Tet = std::array<int64_t, 4>;

// Lattice tetrahedra: four around every B-B edge (two adjacent cell centers),
// each with one A-A edge of the square face the two cells share.
template <class Visit>
void for_each_lattice_tet(const Lattice& lattice, Visit visit) {
  for (int axis = 0; axis < 3; ++axis) {
    const int p = (axis + 1) % 3;
    const int q = (axis + 2) % 3;
    for (int64_t i = 0; i < lattice.cells(0); ++i)
      for (int64_t j = 0; j < lattice.cells(1); ++j)
        for (int64_t k = 0; k < lattice.cells(2); ++k) {
          std::array<int64_t, 3> c{i, j, k};
          if (c[axis] + 1 >= lattice.cells(axis)) continue;
          std::array<int64_t, 3> d = c;
          d[axis] += 1;
          const int64_t b0 = lattice.center(c[0], c[1], c[2]);
          const int64_t b1 = lattice.center(d[0], d[1], d[2]);
          // Corners of the shared face: x_axis = c[axis] + 1.
          std::array<int64_t, 4> square{};
          const int64_t offsets[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
          for (int s = 0; s < 4; ++s) {
            std::array<int64_t, 3> a{};
            a[axis] = c[axis] + 1;
            a[p] = c[p] + offsets[s][0];
            a[q] = c[q] + offsets[s][1];
            square[static_cast<std::size_t>(s)] = lattice.corner(a[0], a[1], a[2]);
          }
          for (int s = 0; s < 4; ++s) visit(Tet{b0, b1, square[s], square[(s + 1) % 4]});
        }
  }
}

class Output {
 public:
  explicit Output(const std::vector<double>& points) : points_(points) {}
  void tet(int64_t a, int64_t b, int64_t c, int64_t d) {
    Tet t{a, b, c, d};
    const int sign = phx::mc::orient3d(points_.data() + 3 * a, points_.data() + 3 * b,
                                        points_.data() + 3 * c, points_.data() + 3 * d);
    if (sign == 0) {
      ++degenerate_;
      return;
    }
    if (sign < 0) std::swap(t[2], t[3]);
    cells_.insert(cells_.end(), t.begin(), t.end());
  }
  // Pyramid with base (q0 q1 q2 q3), split through the base's smallest identity.
  void pyramid(int64_t q0, int64_t q1, int64_t q2, int64_t q3, int64_t apex) {
    if (std::min(q0, q2) < std::min(q1, q3)) {
      tet(q0, q1, q2, apex);
      tet(q0, q2, q3, apex);
    } else {
      tet(q1, q2, q3, apex);
      tet(q1, q3, q0, apex);
    }
  }
  // Prism (v0 v1 v2)-(v3 v4 v5), v3 over v0, v4 over v1, v5 over v2.
  void prism(std::array<int64_t, 6> v) {
    // The prism's symmetries: rotate the triangles, or swap them.
    static constexpr int kMaps[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 2, 0, 4, 5, 3},
                                        {2, 0, 1, 5, 3, 4}, {3, 5, 4, 0, 2, 1},
                                        {4, 3, 5, 1, 0, 2}, {5, 4, 3, 2, 1, 0}};
    int lowest = 0;
    for (int k = 1; k < 6; ++k)
      if (v[static_cast<std::size_t>(k)] < v[static_cast<std::size_t>(lowest)]) lowest = k;
    std::array<int64_t, 6> w{};
    for (int k = 0; k < 6; ++k) w[static_cast<std::size_t>(k)] = v[static_cast<std::size_t>(kMaps[lowest][k])];
    if (std::min(w[1], w[5]) < std::min(w[2], w[4])) {
      tet(w[0], w[1], w[2], w[5]);
      tet(w[0], w[1], w[5], w[4]);
      tet(w[0], w[4], w[5], w[3]);
    } else {
      tet(w[0], w[1], w[2], w[4]);
      tet(w[0], w[4], w[2], w[5]);
      tet(w[0], w[4], w[5], w[3]);
    }
  }
  std::vector<int64_t>& cells() { return cells_; }
  int64_t degenerate() const { return degenerate_; }

 private:
  const std::vector<double>& points_;
  std::vector<int64_t> cells_;
  int64_t degenerate_ = 0;
};

}  // namespace

extern "C" int32_t phx_mc_implicit_stuffing(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, const double* seeds,
    int64_t workspace_size, int64_t output_entry, const double* lower, const double* upper,
    double spacing, int64_t* degenerate_count, phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    if (mesh == nullptr || degenerate_count == nullptr || lower == nullptr ||
        upper == nullptr || !(spacing > 0.0) || !std::isfinite(spacing))
      return PHX_MC_INVALID_ARGUMENT;
    *mesh = nullptr;
    for (int d = 0; d < 3; ++d)
      if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] < upper[d]))
        return PHX_MC_INVALID_ARGUMENT;
    const Lattice lattice(lower, upper, spacing);
    const double budget = static_cast<double>(lattice.corner_count()) +
                          static_cast<double>(lattice.center_count());
    if (!(budget < 2.0e8)) return PHX_MC_CAPACITY_EXCEEDED;
    const Program program{instructions,   instruction_count, indices, index_count, parameters,
                          parameter_count, seeds,             workspace_size, output_entry};

    // 1. Lattice vertices and their values.
    const int64_t lattice_count = lattice.corner_count() + lattice.center_count();
    std::vector<double> points(static_cast<std::size_t>(3 * lattice_count));
    for (int64_t i = 0; i <= lattice.cells(0); ++i)
      for (int64_t j = 0; j <= lattice.cells(1); ++j)
        for (int64_t k = 0; k <= lattice.cells(2); ++k)
          lattice.corner_point(i, j, k, points.data() + 3 * lattice.corner(i, j, k));
    for (int64_t i = 0; i < lattice.cells(0); ++i)
      for (int64_t j = 0; j < lattice.cells(1); ++j)
        for (int64_t k = 0; k < lattice.cells(2); ++k)
          lattice.center_point(i, j, k, points.data() + 3 * lattice.center(i, j, k));
    std::vector<double> values;
    int32_t status = evaluate(program, points, values);
    if (status != PHX_MC_OK) return status;
    for (double value : values)
      if (std::isnan(value)) return PHX_MC_INVALID_INPUT;

    // 2. Cut points on lattice edges with strictly opposite signs.
    std::vector<Cut> cuts;
    EdgeMap cut_index(static_cast<std::size_t>(lattice_count));
    for_each_lattice_tet(lattice, [&](const Tet& t) {
      for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b) {
          const int64_t u = t[a];
          const int64_t v = t[b];
          const int su = sign_of(values[static_cast<std::size_t>(u)]);
          const int sv = sign_of(values[static_cast<std::size_t>(v)]);
          if (su * sv >= 0) continue;
          int64_t& slot = cut_index.slot(edge_key(u, v));
          if (slot >= 0) continue;
          slot = static_cast<int64_t>(cuts.size());
          // (b0, b1) and (square corner, square corner) are the long edges.
          const bool long_edge = (a == 0 && b == 1) || (a == 2 && b == 3);
          cuts.push_back(su < 0 ? Cut{u, v, 0.5, long_edge} : Cut{v, u, 0.5, long_edge});
        }
    });
    {
      // Illinois regula falsi on every cut edge at once.
      const std::size_t n = cuts.size();
      std::vector<double> low(n, 0.0), high(n, 1.0), f_low(n), f_high(n), trial(3 * n), f_trial;
      std::vector<int> side(n, 0);
      for (std::size_t c = 0; c < n; ++c) {
        f_low[c] = values[static_cast<std::size_t>(cuts[c].negative)];
        f_high[c] = values[static_cast<std::size_t>(cuts[c].positive)];
      }
      for (int iteration = 0; iteration < kRootIterations; ++iteration) {
        for (std::size_t c = 0; c < n; ++c) {
          double t = low[c] - f_low[c] * (high[c] - low[c]) / (f_high[c] - f_low[c]);
          if (!(t > low[c] && t < high[c])) t = 0.5 * (low[c] + high[c]);
          cuts[c].t = t;
          const double* a = points.data() + 3 * cuts[c].negative;
          const double* b = points.data() + 3 * cuts[c].positive;
          for (int d = 0; d < 3; ++d) trial[3 * c + static_cast<std::size_t>(d)] = a[d] + t * (b[d] - a[d]);
        }
        status = evaluate(program, trial, f_trial);
        if (status != PHX_MC_OK) return status;
        bool converged = true;
        for (std::size_t c = 0; c < n; ++c) {
          const double value = f_trial[c];
          if (value == 0.0 || high[c] - low[c] <= 1e-15) continue;
          converged = false;
          if (value < 0.0) {
            low[c] = cuts[c].t;
            f_low[c] = value;
            if (side[c] == -1) f_high[c] *= 0.5;
            side[c] = -1;
          } else {
            high[c] = cuts[c].t;
            f_high[c] = value;
            if (side[c] == 1) f_low[c] *= 0.5;
            side[c] = 1;
          }
        }
        if (converged) break;
      }
    }

    // 3. Warping: a vertex near a cut point on one of its edges moves there.
    std::vector<int> signs(static_cast<std::size_t>(lattice_count));
    for (int64_t v = 0; v < lattice_count; ++v)
      signs[static_cast<std::size_t>(v)] = sign_of(values[static_cast<std::size_t>(v)]);
    std::vector<double> warp_distance(static_cast<std::size_t>(lattice_count),
                                      std::numeric_limits<double>::infinity());
    std::vector<int64_t> warp_target(static_cast<std::size_t>(lattice_count), -1);
    for (std::size_t c = 0; c < cuts.size(); ++c) {
      const Cut& cut = cuts[c];
      const double alpha = cut.long_edge ? kAlphaLong : kAlphaShort;
      const double length = cut.long_edge ? spacing : 0.5 * std::sqrt(3.0) * spacing;
      if (cut.t < alpha) {
        const double distance = cut.t * length;
        if (distance < warp_distance[static_cast<std::size_t>(cut.negative)]) {
          warp_distance[static_cast<std::size_t>(cut.negative)] = distance;
          warp_target[static_cast<std::size_t>(cut.negative)] = static_cast<int64_t>(c);
        }
      } else if (1.0 - cut.t < alpha) {
        const double distance = (1.0 - cut.t) * length;
        if (distance < warp_distance[static_cast<std::size_t>(cut.positive)]) {
          warp_distance[static_cast<std::size_t>(cut.positive)] = distance;
          warp_target[static_cast<std::size_t>(cut.positive)] = static_cast<int64_t>(c);
        }
      }
    }
    std::vector<double> cut_points(3 * cuts.size());
    for (std::size_t c = 0; c < cuts.size(); ++c) {
      const double* a = points.data() + 3 * cuts[c].negative;
      const double* b = points.data() + 3 * cuts[c].positive;
      for (int d = 0; d < 3; ++d)
        cut_points[3 * c + static_cast<std::size_t>(d)] = a[d] + cuts[c].t * (b[d] - a[d]);
    }
    for (int64_t v = 0; v < lattice_count; ++v) {
      const int64_t target = warp_target[static_cast<std::size_t>(v)];
      if (target < 0) continue;
      for (int d = 0; d < 3; ++d)
        points[static_cast<std::size_t>(3 * v + d)] =
            cut_points[static_cast<std::size_t>(3 * target + d)];
      signs[static_cast<std::size_t>(v)] = 0;
    }

    // 4. Stencils. Global identities: lattice vertices, then cut points.
    std::vector<double> all_points = points;
    all_points.insert(all_points.end(), cut_points.begin(), cut_points.end());
    Output output(all_points);
    std::vector<Tet> all_zero;
    for_each_lattice_tet(lattice, [&](const Tet& t) {
      int negative = 0, positive = 0;
      for (int64_t v : t) {
        const int s = signs[static_cast<std::size_t>(v)];
        negative += s < 0;
        positive += s > 0;
      }
      if (negative == 0) {
        if (positive == 0) all_zero.push_back(t);
        return;
      }
      if (positive == 0) {
        output.tet(t[0], t[1], t[2], t[3]);
        return;
      }
      const auto cut_id = [&](int64_t u, int64_t v) {
        const int64_t c = cut_index.find(edge_key(u, v));
        return lattice_count + c;
      };
      std::vector<int64_t> minus, plus, zero;
      for (int64_t v : t) {
        const int s = signs[static_cast<std::size_t>(v)];
        (s < 0 ? minus : (s > 0 ? plus : zero)).push_back(v);
      }
      if (minus.size() == 1) {
        // A tetrahedron: the negative vertex, the zeros and the cuts toward the positives.
        std::vector<int64_t> corners{minus[0]};
        corners.insert(corners.end(), zero.begin(), zero.end());
        for (int64_t p : plus) corners.push_back(cut_id(minus[0], p));
        output.tet(corners[0], corners[1], corners[2], corners[3]);
      } else if (minus.size() == 2 && plus.size() == 1) {
        // A pyramid: base (a, b, cut(b, p), cut(a, p)), apex the zero vertex.
        const int64_t a = minus[0], b = minus[1], p = plus[0];
        output.pyramid(a, b, cut_id(b, p), cut_id(a, p), zero[0]);
      } else if (minus.size() == 2) {
        // A wedge: triangles (a, cut(a, p), cut(a, q)) and (b, cut(b, p), cut(b, q)).
        const int64_t a = minus[0], b = minus[1], p = plus[0], q = plus[1];
        output.prism({a, cut_id(a, p), cut_id(a, q), b, cut_id(b, p), cut_id(b, q)});
      } else {
        // The tetrahedron without its positive corner: (a, b, c) over their cuts.
        const int64_t p = plus[0];
        output.prism({minus[0], minus[1], minus[2], cut_id(minus[0], p), cut_id(minus[1], p),
                      cut_id(minus[2], p)});
      }
    });
    if (!all_zero.empty()) {
      // A tetrahedron whose four vertices lie on the zero set is kept when its
      // centroid is inside.
      std::vector<double> centroids(3 * all_zero.size());
      for (std::size_t z = 0; z < all_zero.size(); ++z)
        for (int d = 0; d < 3; ++d) {
          double sum = 0.0;
          for (int64_t v : all_zero[z]) sum += all_points[static_cast<std::size_t>(3 * v + d)];
          centroids[3 * z + static_cast<std::size_t>(d)] = 0.25 * sum;
        }
      std::vector<double> inside;
      status = evaluate(program, centroids, inside);
      if (status != PHX_MC_OK) return status;
      for (std::size_t z = 0; z < all_zero.size(); ++z)
        if (inside[z] < 0.0) output.tet(all_zero[z][0], all_zero[z][1], all_zero[z][2], all_zero[z][3]);
    }

    // Compact the used vertices in identity order.
    std::vector<int64_t> cells = std::move(output.cells());
    std::vector<int32_t> renumber(all_points.size() / 3, -1);
    for (int64_t v : cells) renumber[static_cast<std::size_t>(v)] = 0;
    auto result = std::make_unique<phx_mc_mesh>();
    result->dimension = 3;
    int32_t next = 0;
    for (std::size_t v = 0; v < renumber.size(); ++v) {
      if (renumber[v] < 0) continue;
      renumber[v] = next++;
      result->points.insert(result->points.end(), all_points.begin() + 3 * static_cast<std::ptrdiff_t>(v),
                            all_points.begin() + 3 * static_cast<std::ptrdiff_t>(v) + 3);
    }
    result->cells.reserve(cells.size());
    for (int64_t v : cells) result->cells.push_back(renumber[static_cast<std::size_t>(v)]);
    result->input_point_count = 0;
    *degenerate_count = output.degenerate();
    *mesh = result.release();
    return PHX_MC_OK;
  });
}
