//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Tetrahedral mesh of {f <= 0} by isosurface stuffing on a graded
// body-centered cubic (BCC) lattice (Labelle and Shewchuk, "Isosurface
// Stuffing", SIGGRAPH 2007):
//
//   0. grading: a BCC lattice `levels` times coarser than the requested
//      spacing h is red-refined (each BCC tetrahedron into eight BCC
//      tetrahedra of half its size; Molino, Bridson, Teran and Fedkiw, "A
//      crystalline, red green strategy for meshing highly deformable objects
//      with tetrahedra", 2003) wherever it may come within one spacing of the
//      zero set, so the band around the zero set is exactly the uniform
//      lattice of spacing h. Red closure keeps neighbors within one level;
//      a coarse tetrahedron left with hanging edge midpoints on one edge, two
//      edges of one face, or the three edges of one face is split green, as
//      the cone from its fourth vertex over that face's subdivision (shared
//      quadrilaterals through their smallest identity). Coarse and green
//      tetrahedra never meet the band, so stuffing never cuts or warps them;
//   1. f at every lattice vertex, evaluated in binary64 through the point
//      executor of an interval program;
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
// the result afterwards. With zero levels the lattice is the uniform one.
#include <algorithm>
#include <array>
#include <bit>
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
constexpr int kMaximumLevels = 16;
// Lattice coordinates are packed in 21 bits per axis.
constexpr int64_t kCoordinateLimit = int64_t{1} << 21;

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

// Open-addressing map from 64-bit keys to int64 values, grown at half load.
class KeyMap {
 public:
  explicit KeyMap(std::size_t expected) { allocate(expected); }
  // The value of key, inserted as -1 when absent.
  int64_t& slot(std::uint64_t key) {
    if (2 * (size_ + 1) > keys_.size()) grow();
    std::size_t index = hash(key) & (keys_.size() - 1);
    while (keys_[index] != kEmpty && keys_[index] != key) index = (index + 1) & (keys_.size() - 1);
    if (keys_[index] == kEmpty) {
      keys_[index] = key;
      ++size_;
    }
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
  void allocate(std::size_t expected) {
    std::size_t capacity = 16;
    while (capacity < 2 * expected + 16) capacity *= 2;
    keys_.assign(capacity, kEmpty);
    values_.assign(capacity, -1);
    size_ = 0;
  }
  void grow() {
    std::vector<std::uint64_t> keys = std::move(keys_);
    std::vector<int64_t> values = std::move(values_);
    allocate(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
      if (keys[i] != kEmpty) slot(keys[i]) = values[i];
  }
  std::vector<std::uint64_t> keys_;
  std::vector<int64_t> values_;
  std::size_t size_ = 0;
};

std::uint64_t edge_key(int64_t a, int64_t b) {
  const auto low = static_cast<std::uint64_t>(std::min(a, b));
  const auto high = static_cast<std::uint64_t>(std::max(a, b));
  return (low << 32) | high;
}

using Coordinate = std::array<int64_t, 3>;
using Tet = std::array<int64_t, 4>;

// Lattice vertices at integer coordinates in half-spacing units of the finest
// lattice: its corners have even coordinates, its cell centers odd ones.
class Vertices {
 public:
  Vertices(const std::array<double, 3>& origin, double half, std::size_t expected)
      : origin_(origin), half_(half), index_(expected) {}
  int64_t find(const Coordinate& c) const { return index_.find(key(c)); }
  int64_t get(const Coordinate& c) {
    int64_t& slot = index_.slot(key(c));
    if (slot < 0) {
      slot = static_cast<int64_t>(coordinates_.size());
      coordinates_.push_back(c);
      values_.push_back(std::numeric_limits<double>::quiet_NaN());
      known_.push_back(0);
    }
    return slot;
  }
  // The midpoint of a lattice edge, when it is a lattice point.
  static bool midpoint(const Coordinate& a, const Coordinate& b, Coordinate& m) {
    for (int d = 0; d < 3; ++d) {
      const int64_t sum = a[static_cast<std::size_t>(d)] + b[static_cast<std::size_t>(d)];
      if (sum % 2 != 0) return false;
      m[static_cast<std::size_t>(d)] = sum / 2;
    }
    return true;
  }
  int64_t find_midpoint(int64_t a, int64_t b) const {
    Coordinate m{};
    if (!midpoint(coordinates_[static_cast<std::size_t>(a)],
                  coordinates_[static_cast<std::size_t>(b)], m))
      return -1;
    return find(m);
  }
  int64_t get_midpoint(int64_t a, int64_t b) {
    Coordinate m{};
    midpoint(coordinates_[static_cast<std::size_t>(a)], coordinates_[static_cast<std::size_t>(b)],
             m);
    return get(m);
  }
  void point(int64_t v, double* p) const {
    const Coordinate& c = coordinates_[static_cast<std::size_t>(v)];
    for (int d = 0; d < 3; ++d)
      p[d] = origin_[static_cast<std::size_t>(d)] +
             half_ * static_cast<double>(c[static_cast<std::size_t>(d)]);
  }
  // Evaluates f at every vertex without a value; false on a NaN value.
  int32_t evaluate_unknown(const Program& program) {
    std::vector<int64_t> pending;
    for (std::size_t v = 0; v < known_.size(); ++v)
      if (!known_[v]) pending.push_back(static_cast<int64_t>(v));
    std::vector<double> points(3 * pending.size()), result;
    for (std::size_t i = 0; i < pending.size(); ++i) point(pending[i], points.data() + 3 * i);
    const int32_t status = evaluate(program, points, result);
    if (status != PHX_MC_OK) return status;
    for (std::size_t i = 0; i < pending.size(); ++i) {
      if (std::isnan(result[i])) return PHX_MC_INVALID_INPUT;
      values_[static_cast<std::size_t>(pending[i])] = result[i];
      known_[static_cast<std::size_t>(pending[i])] = 1;
    }
    return PHX_MC_OK;
  }
  int64_t count() const { return static_cast<int64_t>(coordinates_.size()); }
  double value(int64_t v) const { return values_[static_cast<std::size_t>(v)]; }

 private:
  static std::uint64_t key(const Coordinate& c) {
    return (static_cast<std::uint64_t>(c[0]) << 42) | (static_cast<std::uint64_t>(c[1]) << 21) |
           static_cast<std::uint64_t>(c[2]);
  }
  std::array<double, 3> origin_;
  double half_;
  KeyMap index_;
  std::vector<Coordinate> coordinates_;
  std::vector<double> values_;
  std::vector<std::uint8_t> known_;
};

// A BCC tetrahedron (p0, p1, q0, q1): p0 p1 and q0 q1 are its two opposite
// long edges.
struct Cell {
  Tet vertices;
  int level;
  bool alive;
};

constexpr int kEdges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
// Edges of the face opposite each vertex, as a mask over kEdges.
constexpr unsigned kFaceEdges[4] = {0b111000u, 0b100110u, 0b010101u, 0b001011u};

class GradedLattice {
 public:
  GradedLattice(Vertices& vertices, int levels) : vertices_(vertices), levels_(levels) {}

  void add(const Tet& t) { cells_.push_back(Cell{t, 0, true}); }

  // Red refinement into eight BCC tetrahedra of half the size: four corner
  // tetrahedra and the inner octahedron split along its shortest diagonal,
  // which joins the midpoints of the two long edges.
  void refine(std::size_t index) {
    const Cell parent = cells_[index];
    cells_[index].alive = false;
    const auto& [b0, b1, a0, a1] = parent.vertices;
    const int64_t m = vertices_.get_midpoint(b0, b1);
    const int64_t n = vertices_.get_midpoint(a0, a1);
    const int64_t c00 = vertices_.get_midpoint(b0, a0);
    const int64_t c01 = vertices_.get_midpoint(b0, a1);
    const int64_t c10 = vertices_.get_midpoint(b1, a0);
    const int64_t c11 = vertices_.get_midpoint(b1, a1);
    const int level = parent.level + 1;
    for (const Tet& child : {Tet{b0, m, c00, c01}, Tet{b1, m, c10, c11}, Tet{a0, n, c00, c10},
                             Tet{a1, n, c01, c11}, Tet{m, n, c00, c01}, Tet{m, n, c01, c11},
                             Tet{m, n, c11, c10}, Tet{m, n, c10, c00}})
      cells_.push_back(Cell{child, level, true});
  }

  // Refines the cells of one level that may come within `band` of the zero
  // set. f is lipschitz-Lipschitz, so |f(x)| / lipschitz bounds the distance
  // from x to the zero set from below.
  int32_t refine_near(const Program& program, int level, double diameter, double band,
                      double lipschitz) {
    std::vector<std::size_t> candidates;
    for (std::size_t i = 0; i < cells_.size(); ++i)
      if (cells_[i].alive && cells_[i].level == level) candidates.push_back(i);
    std::vector<double> centroids(3 * candidates.size()), values;
    for (std::size_t k = 0; k < candidates.size(); ++k) {
      double* centroid = centroids.data() + 3 * k;
      centroid[0] = centroid[1] = centroid[2] = 0.0;
      for (int64_t v : cells_[candidates[k]].vertices) {
        double p[3];
        vertices_.point(v, p);
        for (int d = 0; d < 3; ++d) centroid[d] += 0.25 * p[d];
      }
    }
    const int32_t status = evaluate(program, centroids, values);
    if (status != PHX_MC_OK) return status;
    // Every point of a BCC tetrahedron lies within its diameter of each vertex
    // and within sqrt(5) / 4 of its diameter of its centroid.
    const double radius = 0.25 * std::sqrt(5.0) * diameter;
    for (std::size_t k = 0; k < candidates.size(); ++k) {
      const Cell& cell = cells_[candidates[k]];
      double clearance = std::abs(values[k]) / lipschitz - radius;
      for (int64_t v : cell.vertices)
        clearance = std::max(clearance, std::abs(vertices_.value(v)) / lipschitz - diameter);
      if (!(clearance > band)) refine(candidates[k]);
    }
    return PHX_MC_OK;
  }

  // Red closure: a coarse cell is refined while a neighbor is two levels finer
  // along one of its edges or faces, or while its hanging midpoints admit no
  // green split.
  void close() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t i = 0; i < cells_.size(); ++i) {
        if (!cells_[i].alive || cells_[i].level == levels_) continue;
        if (needs_red(cells_[i].vertices)) {
          refine(i);
          changed = true;
        }
      }
    }
  }

  // Every leaf, coarse cells with hanging midpoints split green.
  std::vector<Tet> leaves() const {
    std::vector<Tet> out;
    out.reserve(cells_.size());
    for (const Cell& cell : cells_) {
      if (!cell.alive) continue;
      std::array<int64_t, 6> mid{};
      const unsigned mask = cell.level == levels_ ? 0u : hanging(cell.vertices, mid);
      if (mask == 0) {
        out.push_back(cell.vertices);
        continue;
      }
      green(cell.vertices, mask, mid, out);
    }
    return out;
  }

 private:
  unsigned hanging(const Tet& t, std::array<int64_t, 6>& mid) const {
    unsigned mask = 0;
    for (int e = 0; e < 6; ++e) {
      mid[static_cast<std::size_t>(e)] =
          vertices_.find_midpoint(t[static_cast<std::size_t>(kEdges[e][0])],
                                  t[static_cast<std::size_t>(kEdges[e][1])]);
      if (mid[static_cast<std::size_t>(e)] >= 0) mask |= 1u << e;
    }
    return mask;
  }

  bool needs_red(const Tet& t) const {
    std::array<int64_t, 6> mid{};
    const unsigned mask = hanging(t, mid);
    if (mask == 0) return false;
    for (int e = 0; e < 6; ++e) {
      const int64_t m = mid[static_cast<std::size_t>(e)];
      if (m < 0) continue;
      if (vertices_.find_midpoint(t[static_cast<std::size_t>(kEdges[e][0])], m) >= 0 ||
          vertices_.find_midpoint(m, t[static_cast<std::size_t>(kEdges[e][1])]) >= 0)
        return true;
    }
    for (unsigned face : kFaceEdges) {
      if ((mask & face) != face) continue;
      std::array<int64_t, 3> inner{};
      int k = 0;
      for (int e = 0; e < 6; ++e)
        if (face & (1u << e)) inner[static_cast<std::size_t>(k++)] = mid[static_cast<std::size_t>(e)];
      if (vertices_.find_midpoint(inner[0], inner[1]) >= 0 ||
          vertices_.find_midpoint(inner[1], inner[2]) >= 0 ||
          vertices_.find_midpoint(inner[2], inner[0]) >= 0)
        return true;
    }
    const int count = std::popcount(mask);
    if (count == 1) return false;
    for (unsigned face : kFaceEdges)
      if ((mask & ~face) == 0) return false;  // two or three edges of one face
    return true;
  }

  // The cone from the vertex opposite the face that holds every hanging
  // midpoint over that face's subdivision.
  void green(const Tet& t, unsigned mask, const std::array<int64_t, 6>& mid,
             std::vector<Tet>& out) const {
    int apex = 0;
    while ((mask & ~kFaceEdges[apex]) != 0) ++apex;
    std::array<int64_t, 3> face{};
    int k = 0;
    for (int v = 0; v < 4; ++v)
      if (v != apex) face[static_cast<std::size_t>(k++)] = t[static_cast<std::size_t>(v)];
    const int64_t top = t[static_cast<std::size_t>(apex)];
    const auto midpoint = [&](int64_t a, int64_t b) -> int64_t {
      for (int e = 0; e < 6; ++e) {
        const int64_t u = t[static_cast<std::size_t>(kEdges[e][0])];
        const int64_t v = t[static_cast<std::size_t>(kEdges[e][1])];
        if ((u == a && v == b) || (u == b && v == a)) return mid[static_cast<std::size_t>(e)];
      }
      return -1;
    };
    std::array<int64_t, 3> m{};  // m[i] on the face edge opposite face[i]
    int split = 0;
    for (int i = 0; i < 3; ++i) {
      m[static_cast<std::size_t>(i)] = midpoint(face[static_cast<std::size_t>((i + 1) % 3)],
                                                face[static_cast<std::size_t>((i + 2) % 3)]);
      split += m[static_cast<std::size_t>(i)] >= 0;
    }
    const auto emit = [&](int64_t a, int64_t b, int64_t c) { out.push_back(Tet{top, a, b, c}); };
    if (split == 3) {
      emit(face[0], m[2], m[1]);
      emit(face[1], m[0], m[2]);
      emit(face[2], m[1], m[0]);
      emit(m[0], m[1], m[2]);
      return;
    }
    if (split == 1) {
      int i = 0;
      while (m[static_cast<std::size_t>(i)] < 0) ++i;
      const int64_t s = face[static_cast<std::size_t>(i)];
      const int64_t u = face[static_cast<std::size_t>((i + 1) % 3)];
      const int64_t v = face[static_cast<std::size_t>((i + 2) % 3)];
      emit(s, u, m[static_cast<std::size_t>(i)]);
      emit(s, m[static_cast<std::size_t>(i)], v);
      return;
    }
    // Two split edges meet at s, opposite the unsplit one: the corner triangle
    // and the quadrilateral (mu, u, v, mv), split through its smallest identity.
    int i = 0;
    while (m[static_cast<std::size_t>(i)] >= 0) ++i;
    const int64_t s = face[static_cast<std::size_t>(i)];
    const int64_t u = face[static_cast<std::size_t>((i + 1) % 3)];
    const int64_t v = face[static_cast<std::size_t>((i + 2) % 3)];
    const int64_t mu = midpoint(s, u);
    const int64_t mv = midpoint(s, v);
    emit(s, mu, mv);
    if (std::min(mu, v) < std::min(u, mv)) {
      emit(mu, u, v);
      emit(mu, v, mv);
    } else {
      emit(mu, u, mv);
      emit(u, v, mv);
    }
  }

  Vertices& vertices_;
  int levels_;
  std::vector<Cell> cells_;
};

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

// One lattice edge with a sign change and its cut point.
struct Cut {
  int64_t negative;  // lattice vertex with f < 0
  int64_t positive;  // lattice vertex with f > 0
  double t;          // position: negative + t (positive - negative)
  bool long_edge;
};

// The graded lattice's leaves over the box: the coarse lattice is padded by
// one coarse cell on each side, which keeps the zero set inside it.
int32_t graded_leaves(const Program& program, const double* lower, const double* upper,
                      double spacing, double lipschitz, int levels,
                      std::unique_ptr<Vertices>& vertices, std::vector<Tet>& leaves) {
  const double coarse = std::ldexp(spacing, levels);
  std::array<int64_t, 3> cells{};
  std::array<double, 3> origin{};
  double fine_points = 1.0;
  for (int d = 0; d < 3; ++d) {
    const double extent = upper[d] - lower[d];
    const double count = std::ceil(extent / coarse) + 2.0;
    const double fine = std::ldexp(count, levels);
    if (!(2.0 * fine < static_cast<double>(kCoordinateLimit))) return PHX_MC_CAPACITY_EXCEEDED;
    fine_points *= fine + 1.0;
    cells[static_cast<std::size_t>(d)] = static_cast<int64_t>(count);
    origin[static_cast<std::size_t>(d)] = 0.5 * (lower[d] + upper[d]) - 0.5 * count * coarse;
  }
  if (!(2.0 * fine_points < 2.0e8)) return PHX_MC_CAPACITY_EXCEEDED;
  const int64_t corner_count = (cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1);
  const int64_t center_count = cells[0] * cells[1] * cells[2];
  vertices = std::make_unique<Vertices>(origin, 0.5 * spacing,
                                        static_cast<std::size_t>(corner_count + center_count));
  // Coarse corners, then coarse centers, in lattice order: their identities.
  const int64_t step = int64_t{2} << levels;
  const int64_t offset = int64_t{1} << levels;
  for (int64_t i = 0; i <= cells[0]; ++i)
    for (int64_t j = 0; j <= cells[1]; ++j)
      for (int64_t k = 0; k <= cells[2]; ++k) vertices->get({step * i, step * j, step * k});
  for (int64_t i = 0; i < cells[0]; ++i)
    for (int64_t j = 0; j < cells[1]; ++j)
      for (int64_t k = 0; k < cells[2]; ++k)
        vertices->get({step * i + offset, step * j + offset, step * k + offset});
  const auto corner = [&](int64_t i, int64_t j, int64_t k) {
    return (i * (cells[1] + 1) + j) * (cells[2] + 1) + k;
  };
  const auto center = [&](int64_t i, int64_t j, int64_t k) {
    return corner_count + (i * cells[1] + j) * cells[2] + k;
  };
  GradedLattice lattice(*vertices, levels);
  // Four tetrahedra around every B-B edge (two adjacent cell centers), each
  // with one A-A edge of the square face the two cells share.
  for (int axis = 0; axis < 3; ++axis) {
    const int p = (axis + 1) % 3;
    const int q = (axis + 2) % 3;
    for (int64_t i = 0; i < cells[0]; ++i)
      for (int64_t j = 0; j < cells[1]; ++j)
        for (int64_t k = 0; k < cells[2]; ++k) {
          std::array<int64_t, 3> c{i, j, k};
          if (c[static_cast<std::size_t>(axis)] + 1 >= cells[static_cast<std::size_t>(axis)])
            continue;
          std::array<int64_t, 3> e = c;
          e[static_cast<std::size_t>(axis)] += 1;
          const int64_t b0 = center(c[0], c[1], c[2]);
          const int64_t b1 = center(e[0], e[1], e[2]);
          std::array<int64_t, 4> square{};
          const int64_t offsets[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
          for (int s = 0; s < 4; ++s) {
            std::array<int64_t, 3> a{};
            a[static_cast<std::size_t>(axis)] = c[static_cast<std::size_t>(axis)] + 1;
            a[static_cast<std::size_t>(p)] = c[static_cast<std::size_t>(p)] + offsets[s][0];
            a[static_cast<std::size_t>(q)] = c[static_cast<std::size_t>(q)] + offsets[s][1];
            square[static_cast<std::size_t>(s)] = corner(a[0], a[1], a[2]);
          }
          for (int s = 0; s < 4; ++s) lattice.add(Tet{b0, b1, square[s], square[(s + 1) % 4]});
        }
  }
  for (int level = 0; level < levels; ++level) {
    int32_t status = vertices->evaluate_unknown(program);
    if (status != PHX_MC_OK) return status;
    // A cell of level l has diameter (its long edge) spacing 2^(levels - l).
    const double diameter = std::ldexp(spacing, levels - level);
    status = lattice.refine_near(program, level, diameter, spacing, lipschitz);
    if (status != PHX_MC_OK) return status;
  }
  lattice.close();
  const int32_t status = vertices->evaluate_unknown(program);
  if (status != PHX_MC_OK) return status;
  leaves = lattice.leaves();
  return PHX_MC_OK;
}

}  // namespace

extern "C" int32_t phx_mc_implicit_stuffing(
    const int64_t* instructions, int64_t instruction_count, const int64_t* indices,
    int64_t index_count, const double* parameters, int64_t parameter_count, const double* seeds,
    int64_t workspace_size, int64_t output_entry, const double* lower, const double* upper,
    double spacing, int32_t levels, double lipschitz, int64_t* degenerate_count,
    phx_mc_mesh** mesh) {
  return phx::mc::guarded([&]() -> int32_t {
    if (mesh == nullptr || degenerate_count == nullptr || lower == nullptr ||
        upper == nullptr || !(spacing > 0.0) || !std::isfinite(spacing) || levels < 0 ||
        levels > kMaximumLevels || !(lipschitz > 0.0) || !std::isfinite(lipschitz))
      return PHX_MC_INVALID_ARGUMENT;
    *mesh = nullptr;
    for (int d = 0; d < 3; ++d)
      if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || !(lower[d] < upper[d]))
        return PHX_MC_INVALID_ARGUMENT;
    const Program program{instructions,   instruction_count, indices, index_count, parameters,
                          parameter_count, seeds,             workspace_size, output_entry};

    // 0-1. The graded lattice and f at its vertices.
    std::unique_ptr<Vertices> vertices;
    std::vector<Tet> leaves;
    int32_t status =
        graded_leaves(program, lower, upper, spacing, lipschitz, levels, vertices, leaves);
    if (status != PHX_MC_OK) return status;
    const int64_t lattice_count = vertices->count();
    std::vector<double> points(static_cast<std::size_t>(3 * lattice_count));
    std::vector<double> values(static_cast<std::size_t>(lattice_count));
    for (int64_t v = 0; v < lattice_count; ++v) {
      double* p = points.data() + 3 * v;
      vertices->point(v, p);
      const double value = vertices->value(v);
      values[static_cast<std::size_t>(v)] = value;
      // The padding outside the box must lie outside the solid: otherwise
      // the solid reaches the lattice's outer layer, and lattice tetrahedra
      // would stand in for its missing boundary.
      if (!(value > 0.0))
        for (int d = 0; d < 3; ++d)
          if (p[d] < lower[d] || p[d] > upper[d]) return PHX_MC_INVALID_INPUT;
    }
    vertices.reset();

    // 2. Cut points on lattice edges with strictly opposite signs.
    std::vector<Cut> cuts;
    KeyMap cut_index(static_cast<std::size_t>(lattice_count));
    for (const Tet& t : leaves) {
      for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b) {
          const int64_t u = t[static_cast<std::size_t>(a)];
          const int64_t v = t[static_cast<std::size_t>(b)];
          const int su = sign_of(values[static_cast<std::size_t>(u)]);
          const int sv = sign_of(values[static_cast<std::size_t>(v)]);
          if (su * sv >= 0) continue;
          int64_t& slot = cut_index.slot(edge_key(u, v));
          if (slot >= 0) continue;
          slot = static_cast<int64_t>(cuts.size());
          // (p0, p1) and (q0, q1) are the long edges of a BCC tetrahedron.
          const bool long_edge = (a == 0 && b == 1) || (a == 2 && b == 3);
          cuts.push_back(su < 0 ? Cut{u, v, 0.5, long_edge} : Cut{v, u, 0.5, long_edge});
        }
    }
    {
      // Illinois regula falsi on every cut edge at once; a cut leaves the
      // batch when f vanishes at its point or its bracket closes.
      const std::size_t n = cuts.size();
      std::vector<double> low(n, 0.0), high(n, 1.0), f_low(n), f_high(n), trial, f_trial;
      std::vector<int> side(n, 0);
      std::vector<std::size_t> active(n);
      for (std::size_t c = 0; c < n; ++c) {
        f_low[c] = values[static_cast<std::size_t>(cuts[c].negative)];
        f_high[c] = values[static_cast<std::size_t>(cuts[c].positive)];
        active[c] = c;
      }
      for (int iteration = 0; iteration < kRootIterations && !active.empty(); ++iteration) {
        trial.resize(3 * active.size());
        for (std::size_t k = 0; k < active.size(); ++k) {
          const std::size_t c = active[k];
          double t = low[c] - f_low[c] * (high[c] - low[c]) / (f_high[c] - f_low[c]);
          if (!(t > low[c] && t < high[c])) t = 0.5 * (low[c] + high[c]);
          cuts[c].t = t;
          const double* a = points.data() + 3 * cuts[c].negative;
          const double* b = points.data() + 3 * cuts[c].positive;
          for (int d = 0; d < 3; ++d) trial[3 * k + static_cast<std::size_t>(d)] = a[d] + t * (b[d] - a[d]);
        }
        status = evaluate(program, trial, f_trial);
        if (status != PHX_MC_OK) return status;
        std::size_t kept = 0;
        for (std::size_t k = 0; k < active.size(); ++k) {
          const std::size_t c = active[k];
          const double value = f_trial[k];
          if (value == 0.0 || high[c] - low[c] <= 1e-15) continue;
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
          active[kept++] = c;
        }
        active.resize(kept);
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
    std::vector<double> all_points = std::move(points);
    all_points.insert(all_points.end(), cut_points.begin(), cut_points.end());
    Output output(all_points);
    std::vector<Tet> all_zero;
    for (const Tet& t : leaves) {
      int negative = 0, positive = 0;
      for (int64_t v : t) {
        const int s = signs[static_cast<std::size_t>(v)];
        negative += s < 0;
        positive += s > 0;
      }
      if (negative == 0) {
        if (positive == 0) all_zero.push_back(t);
        continue;
      }
      if (positive == 0) {
        output.tet(t[0], t[1], t[2], t[3]);
        continue;
      }
      const auto cut_id = [&](int64_t u, int64_t v) {
        const int64_t c = cut_index.find(edge_key(u, v));
        return lattice_count + c;
      };
      std::array<int64_t, 4> minus{}, plus{}, zero{};
      std::size_t nm = 0, np = 0, nz = 0;
      for (int64_t v : t) {
        const int s = signs[static_cast<std::size_t>(v)];
        if (s < 0) minus[nm++] = v;
        else if (s > 0) plus[np++] = v;
        else zero[nz++] = v;
      }
      if (nm == 1) {
        // A tetrahedron: the negative vertex, the zeros and the cuts toward the positives.
        std::array<int64_t, 4> corners{minus[0]};
        std::size_t k = 1;
        for (std::size_t z = 0; z < nz; ++z) corners[k++] = zero[z];
        for (std::size_t p = 0; p < np; ++p) corners[k++] = cut_id(minus[0], plus[p]);
        output.tet(corners[0], corners[1], corners[2], corners[3]);
      } else if (nm == 2 && np == 1) {
        // A pyramid: base (a, b, cut(b, p), cut(a, p)), apex the zero vertex.
        const int64_t a = minus[0], b = minus[1], p = plus[0];
        output.pyramid(a, b, cut_id(b, p), cut_id(a, p), zero[0]);
      } else if (nm == 2) {
        // A wedge: triangles (a, cut(a, p), cut(a, q)) and (b, cut(b, p), cut(b, q)).
        const int64_t a = minus[0], b = minus[1], p = plus[0], q = plus[1];
        output.prism({a, cut_id(a, p), cut_id(a, q), b, cut_id(b, p), cut_id(b, q)});
      } else {
        // The tetrahedron without its positive corner: (a, b, c) over their cuts.
        const int64_t p = plus[0];
        output.prism({minus[0], minus[1], minus[2], cut_id(minus[0], p), cut_id(minus[1], p),
                      cut_id(minus[2], p)});
      }
    }
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
