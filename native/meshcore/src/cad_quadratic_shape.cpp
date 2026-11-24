// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "cad_quadratic_shape.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <utility>

#include "bounded_lbfgs.hpp"
#include "geometry3.hpp"

namespace phx::mc {
namespace {

using Point = std::array<double, 3>;
using Jacobian = std::array<Point, 3>;  // Columns in reference-coordinate order.
using CornerMaps = std::array<Jacobian, 4>;
using EdgeSlots = std::array<std::size_t, 6>;

constexpr int kEdgeCorners[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
constexpr auto kFixed = std::numeric_limits<std::size_t>::max();
constexpr double kComponentBound = .025;
constexpr double kMeritMargin = .00015;
constexpr double kSqrtTwo = std::numbers::sqrt2;
constexpr std::uint8_t kScaled = 1, kRatio = 2;

struct Targets {
  double scaled, ratio;

  std::uint8_t enabled() const {
    return (scaled > 0.0 ? kScaled : 0) | (ratio > 0.0 ? kRatio : 0);
  }
};

struct Proposal {
  Point point;
  bool changed;
};

struct ActiveCell {
  std::int64_t id;
  EdgeSlots slots;  // Index in the deterministic interior-edge workset, or kFixed.
  CornerMaps maps;
  double step_scale;  // 4 h in the cell's local coordinates.
};

// Write P2 as the affine map plus 4*l_i*l_j times each edge's displacement
// from its affine midpoint. The Jacobian is affine; at corner i that edge
// contributes only 4*displacement times dl_j, and vice versa at corner j.
void add_column(Jacobian& j, int corner, const Point& d, double scale) {
  if (corner != 0) {
    for (int v = 0; v < 3; ++v) j[corner - 1][v] += scale * d[v];
  } else {
    for (auto& column : j)
      for (int v = 0; v < 3; ++v) column[v] -= scale * d[v];
  }
}

void add_edge(CornerMaps& maps, int e, const Point& d, double scale) {
  const auto [i, j] = kEdgeCorners[e];
  add_column(maps[i], j, d, scale);
  add_column(maps[j], i, d, scale);
}

bool corner_maps(const CadQuadraticInput& input, const CadQuadraticMesh& mesh, std::int64_t id,
                 CornerMaps& maps, double& scale, const ActiveCell* cell = nullptr,
                 std::span<const Proposal> proposal = {}) {
  const std::int32_t* tet = input.tetrahedra + 4 * id;
  const double* origin = input.points + 3 * static_cast<std::int64_t>(tet[0]);
  std::array<Point, 10> local;
  local[0] = {};
  for (int i = 1; i < 4; ++i)
    geometry::difference(input.points + 3 * static_cast<std::int64_t>(tet[i]), origin,
                         local[i].data());
  const auto& edges = mesh.cell_edges[id];
  for (int e = 0; e < 6; ++e) {
    const Point& p = cell != nullptr && cell->slots[e] != kFixed
                         ? proposal[cell->slots[e]].point : mesh.nodes[edges[e]];
    geometry::difference(p.data(), origin, local[4 + e].data());
  }
  scale = 0.0;
  for (int i = 1; i < 10; ++i)
    for (double x : local[i]) scale = std::max(scale, std::abs(x));
  if (!(scale > 0.0) || !std::isfinite(scale)) return false;
  const double inverse = 1.0 / scale;
  const auto normalize = [=](double x) { return std::isfinite(inverse) ? x * inverse : x / scale; };
  Jacobian affine;
  for (int c = 0; c < 3; ++c)
    for (int v = 0; v < 3; ++v) affine[c][v] = normalize(local[c + 1][v]);
  maps.fill(affine);
  for (int e = 0; e < 6; ++e) {
    const auto [i, j] = kEdgeCorners[e];
    Point displacement;
    for (int v = 0; v < 3; ++v)
      displacement[v] = normalize(local[4 + e][v] - (.5 * local[i][v] + .5 * local[j][v]));
    add_edge(maps, e, displacement, 4.0);
  }
  return true;
}

Jacobian cofactor(const Jacobian& j) {
  return {geometry::cross(j[1], j[2]), geometry::cross(j[2], j[0]),
          geometry::cross(j[0], j[1])};
}

struct Sample {
  double det;
  double scaled;
  double norm_product;
  Point squares;
};

Sample measure(const Jacobian& j, bool scaled) {
  Sample sample{};
  sample.det = geometry::dot(j[0].data(), geometry::cross(j[1], j[2]).data());
  if (scaled) {
    for (int d = 0; d < 3; ++d) sample.squares[d] = geometry::dot(j[d].data(), j[d].data());
    const auto [a, b, c] = sample.squares;
    const double product = a * b * c;
    sample.norm_product = product >= std::numeric_limits<double>::min()
                              ? std::sqrt(product) : std::sqrt(a) * std::sqrt(b) * std::sqrt(c);
    sample.scaled = sample.norm_product > 0.0 ? kSqrtTwo * (sample.det / sample.norm_product)
                                              : std::numeric_limits<double>::quiet_NaN();
  }
  return sample;
}

struct Corners {
  std::array<Sample, 4> samples;
  int low = 0, high = 0;
  double scaled_min = std::numeric_limits<double>::infinity();
  bool finite_determinants = true;

  explicit Corners(const CornerMaps& maps, bool scaled) {
    for (int i = 0; i < 4; ++i) {
      const Sample s = measure(maps[i], scaled);
      samples[i] = s;
      finite_determinants &= std::isfinite(s.det);
      if (scaled)
        scaled_min = std::isfinite(s.scaled) ? std::min(scaled_min, s.scaled)
                                              : -std::numeric_limits<double>::infinity();
      if (s.det < samples[low].det) low = i;
      if (s.det > samples[high].det) high = i;
    }
  }

  std::uint8_t misses(Targets targets) const {
    std::uint8_t missed = 0;
    if (targets.scaled > 0.0 && scaled_min < targets.scaled) missed |= kScaled;
    if (targets.ratio > 0.0) {
      const double lo = samples[low].det, hi = samples[high].det;
      if (!finite_determinants || !(lo > 0.0 && lo / hi >= targets.ratio)) missed |= kRatio;
    }
    return missed;
  }
};

// The local nodes have magnitude at most one per component, so each
// Jacobian column has norm below 32. If corner columns agree to 2^-44
// per component, their convex combinations change the determinant and
// column norm product by less than 2^-30 (Hadamard and the product rule).
// A cell near a bound still runs the ordinary full shape screen.
bool clearly_constant_shape(const CornerMaps& maps, Targets targets) {
  for (std::size_t corner = 1; corner < maps.size(); ++corner) {
    for (int column = 0; column < 3; ++column) {
      for (int axis = 0; axis < 3; ++axis) {
        if (!(std::abs(maps[corner][column][axis] - maps[0][column][axis]) <= 0x1p-44))
          return false;
      }
    }
  }
  const Sample base = measure(maps[0], targets.scaled > 0.0);
  constexpr double margin = 0x1p-30;
  const double lower = base.det - margin;
  return lower > 0.0 && lower > targets.ratio * (base.det + margin) &&
         lower * kSqrtTwo > targets.scaled * (base.norm_product + margin);
}

// Screen obviously good cells before sampling. Expand det(J0 + E) by columns:
// the linear term reaches its extrema at corners; the other four terms are
// bounded by products of column norms. Each norm of J is convex as well.
// The cushion is in normalized coordinates (all local node components <= 1)
// and only disables the shortcut near a threshold, never changes the target.
std::uint8_t screen(const CornerMaps& maps, const Corners& corners, Targets targets,
                    std::uint8_t pending) {
  const Jacobian cof = cofactor(maps[0]);
  Point radius2{}, norm2 = corners.samples[0].squares;
  double linear_min = 0.0, linear_max = 0.0;
  for (int a = 1; a < 4; ++a) {
    double linear = 0.0;
    for (int c = 0; c < 3; ++c) {
      const Point d = geometry::difference(maps[a][c], maps[0][c]);
      radius2[c] = std::max(radius2[c], geometry::dot(d.data(), d.data()));
      linear += geometry::dot(d.data(), cof[c].data());
      if (pending & kScaled) norm2[c] = std::max(norm2[c], corners.samples[a].squares[c]);
    }
    linear_min = std::min(linear_min, linear);
    linear_max = std::max(linear_max, linear);
  }
  const double a = geometry::length(maps[0][0]), b = geometry::length(maps[0][1]);
  const double c = geometry::length(maps[0][2]);
  const double x = std::sqrt(radius2[0]), y = std::sqrt(radius2[1]);
  const double z = std::sqrt(radius2[2]);
  const double magnitude = std::max({1.0, a + x, b + y, c + z});
  const double epsilon = 0x1p-36 * magnitude;
  const double error = epsilon * magnitude * magnitude;
  const double rest = x * y * c + x * z * b + y * z * a + x * y * z + error;
  const double lo = corners.samples[0].det + linear_min - rest;
  const double hi = corners.samples[0].det + linear_max + rest;
  if (lo > 0.0) {
    if ((pending & kRatio) && lo > targets.ratio * hi) pending &= ~kRatio;
    if (pending & kScaled) {
      const double norms = (std::sqrt(norm2[0]) + epsilon) * (std::sqrt(norm2[1]) + epsilon) *
                           (std::sqrt(norm2[2]) + epsilon);
      if (kSqrtTwo * lo > targets.scaled * norms) pending &= ~kScaled;
    }
  }
  return pending;
}

// The 80 non-corner points of the degree-six barycentric tetrahedral lattice.
constexpr auto kLattice = [] {
  constexpr int degree = 6;
  std::array<Point, (degree + 1) * (degree + 2) * (degree + 3) / 6 - 4> sites{};
  std::size_t n = 0;
  for (int i = 0; i <= degree; ++i)
    for (int j = 0; i + j <= degree; ++j)
      for (int k = 0; i + j + k <= degree; ++k) {
        if (i == degree || j == degree || k == degree || i + j + k == 0) continue;
        sites[n++] = {double(i) / degree, double(j) / degree, double(k) / degree};
      }
  return sites;
}();

std::uint8_t sampled_misses(const CornerMaps& maps, const Corners& corners, Targets targets) {
  std::uint8_t missed = corners.misses(targets);
  std::uint8_t pending = targets.enabled() & ~missed;
  if (pending == 0 || (pending = screen(maps, corners, targets, pending)) == 0) return missed;

  std::array<Jacobian, 3> delta;
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c) delta[a][c] = geometry::difference(maps[a + 1][c], maps[0][c]);
  double lo = corners.samples[corners.low].det, hi = corners.samples[corners.high].det;
  for (const Point& weights : kLattice) {
    Jacobian j;
    for (int c = 0; c < 3; ++c)
      for (int v = 0; v < 3; ++v)
        j[c][v] = maps[0][c][v] + weights[0] * delta[0][c][v] + weights[1] * delta[1][c][v] +
                  weights[2] * delta[2][c][v];
    const Sample s = measure(j, (pending & kScaled) != 0);
    if ((pending & kScaled) && (!(s.scaled >= targets.scaled) || !std::isfinite(s.scaled))) {
      missed |= kScaled;
      pending &= ~kScaled;
    }
    if (pending & kRatio) {
      lo = std::min(lo, s.det);
      hi = std::max(hi, s.det);
      if (!(lo > 0.0 && lo / hi >= targets.ratio) || !std::isfinite(s.det)) {
        missed |= kRatio;
        pending &= ~kRatio;
      }
    }
    if (pending == 0) break;
  }
  return missed;
}

// Chain the Jacobian adjoint through the same corner-edge derivatives as add_edge.
void accumulate(const ActiveCell& cell, int corner, const Jacobian& adjoint, double weight,
                std::span<double> gradient) {
  weight *= cell.step_scale;
  for (int e = 0; e < 6; ++e) {
    if (cell.slots[e] == kFixed) continue;
    const auto [i, j] = kEdgeCorners[e];
    const int other = i == corner ? j : j == corner ? i : -1;
    if (other < 0) continue;
    double* g = gradient.data() + 3 * cell.slots[e];
    if (other == 0) {
      for (int v = 0; v < 3; ++v) g[v] -= weight * (adjoint[0][v] + adjoint[1][v] + adjoint[2][v]);
    } else {
      for (int v = 0; v < 3; ++v) g[v] += weight * adjoint[other - 1][v];
    }
  }
}

double merit(std::span<const ActiveCell> cells, Targets targets, std::span<const double> q,
             std::span<double> gradient) {
  std::fill(gradient.begin(), gradient.end(), 0.0);
  double value = 0.0;
  for (const ActiveCell& cell : cells) {
    CornerMaps maps = cell.maps;
    for (int e = 0; e < 6; ++e) {
      if (cell.slots[e] == kFixed) continue;
      const double* d = q.data() + 3 * cell.slots[e];
      add_edge(maps, e, Point{d[0], d[1], d[2]}, cell.step_scale);
    }
    const Corners corners(maps, targets.scaled > 0.0);
    if (!corners.finite_determinants ||
        (targets.scaled > 0.0 && !std::isfinite(corners.scaled_min)))
      return std::numeric_limits<double>::quiet_NaN();
    if (targets.ratio > 0.0) {
      const double hi = corners.samples[corners.high].det;
      if (!(hi > 0.0)) return std::numeric_limits<double>::quiet_NaN();
      const double ratio = corners.samples[corners.low].det / hi;
      const double deficit = targets.ratio - ratio;
      if (deficit > 0.0) {
        value += .5 * deficit * deficit;
        const double w = -deficit / hi;
        accumulate(cell, corners.low, cofactor(maps[corners.low]), w, gradient);
        accumulate(cell, corners.high, cofactor(maps[corners.high]), -w * ratio, gradient);
      }
    }
    if (targets.scaled > 0.0) {
      for (int a = 0; a < 4; ++a) {
        const Sample& s = corners.samples[a];
        const double deficit = targets.scaled - s.scaled;
        if (deficit <= 0.0) continue;
        value += .5 * deficit * deficit;
        Jacobian derivative = cofactor(maps[a]);
        const double factor = kSqrtTwo / s.norm_product;
        for (int c = 0; c < 3; ++c)
          for (int v = 0; v < 3; ++v)
            derivative[c][v] = factor * derivative[c][v] - s.scaled * maps[a][c][v] / s.squares[c];
        accumulate(cell, a, derivative, -deficit, gradient);
      }
    }
  }
  return value;
}

}  // namespace

std::int32_t polish_cad_quadratic_shape(const CadQuadraticInput& input, double length_scale,
                                       double minimum_scaled, double minimum_ratio,
                                       CadQuadraticMesh& mesh,
                                       std::vector<std::int64_t>& touched_cells,
                                       std::array<std::int64_t, 2>& unmet,
                                       std::vector<std::int64_t>* moved_edges) {
  touched_cells.clear();
  if (moved_edges != nullptr) moved_edges->clear();
  unmet = {};
  if (!(minimum_scaled >= 0.0 && std::isfinite(minimum_scaled) &&
        minimum_ratio >= 0.0 && std::isfinite(minimum_ratio))) return PHX_MC_INVALID_ARGUMENT;
  const Targets targets{minimum_scaled, minimum_ratio};
  if (targets.enabled() == 0) return PHX_MC_OK;
  if (!(length_scale > 0.0 && std::isfinite(length_scale))) return PHX_MC_INVALID_ARGUMENT;

  // Keep a small gap between the corner merit and the sampled acceptance so
  // convergence tolerance is not mistaken for meeting a requested threshold.
  const Targets goals{
      minimum_scaled > 0.0 && minimum_scaled < kSqrtTwo
          ? std::min(kSqrtTwo, minimum_scaled + kMeritMargin) : minimum_scaled,
      minimum_ratio > 0.0 && minimum_ratio < 1.0
          ? std::min(1.0, minimum_ratio + kMeritMargin) : minimum_ratio};
  std::vector<std::int64_t> failing;
  std::vector<std::int64_t> edges;
  for (std::int64_t id = 0; id < input.tetrahedron_count; ++id) {
    CornerMaps maps;
    double scale;
    std::uint8_t missed;
    if (corner_maps(input, mesh, id, maps, scale)) {
      if (clearly_constant_shape(maps, goals)) continue;
      const Corners corners(maps, targets.scaled > 0.0);
      missed = sampled_misses(maps, corners, targets);
      if (corners.misses(goals) != 0)
        for (std::int64_t e : mesh.cell_edges[id])
          if (mesh.edge_kinds[e] == PHX_MC_CAD_EDGE_INTERIOR) edges.push_back(e);
    } else {
      missed = targets.enabled();
    }
    if (missed != 0) {
      failing.push_back(id);
      unmet[0] += (missed & kScaled) != 0;
      unmet[1] += (missed & kRatio) != 0;
    }
  }
  if (failing.empty()) return PHX_MC_OK;
  if (edges.empty()) return PHX_MC_UNRESOLVED;

  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::vector<std::size_t> slots(mesh.nodes.size(), kFixed);
  for (std::size_t i = 0; i < edges.size(); ++i) slots[edges[i]] = i;

  // Scan in cell order to take the full incidence closure, including cells
  // which already satisfy both targets but share an eligible edge.
  std::vector<ActiveCell> active;
  for (std::int64_t id = 0; id < input.tetrahedron_count; ++id) {
    EdgeSlots cell_slots;
    bool affected = false;
    for (int e = 0; e < 6; ++e) {
      cell_slots[e] = slots[mesh.cell_edges[id][e]];
      affected |= cell_slots[e] != kFixed;
    }
    if (!affected) continue;
    ActiveCell& cell = active.emplace_back();
    cell.id = id;
    cell.slots = cell_slots;
    double scale;
    if (!corner_maps(input, mesh, id, cell.maps, scale)) return PHX_MC_UNRESOLVED;
    cell.step_scale = 4.0 * (length_scale / scale);
    if (!(cell.step_scale > 0.0 && std::isfinite(cell.step_scale))) return PHX_MC_UNRESOLVED;
  }

  double gap = kMeritMargin;
  if (targets.scaled > 0.0) gap = std::min(gap, goals.scaled - targets.scaled);
  if (targets.ratio > 0.0) gap = std::min(gap, goals.ratio - targets.ratio);
  // A corner that misses the request contributes at least gap²/2 to the
  // merit. Stop short of that gap; the rounded sample check still owns
  // acceptance, including non-corner sites and near-limit targets.
  const double stop = gap > 0.0 ? 0.25 * gap * gap : 1e-14;
  std::vector<double> position(3 * edges.size(), 0.0);
  const auto result = detail::bounded_lbfgs(position, kComponentBound, .02, 20, stop,
      [&](std::span<const double> q, std::span<double> gradient) {
        return merit(active, goals, q, gradient);
      });
  if (result.status == detail::BoundedLbfgsStatus::kInvalidObjective || result.steps == 0)
    return PHX_MC_UNRESOLVED;

  const double cap = kComponentBound * length_scale;
  std::vector<Proposal> proposal;
  proposal.reserve(edges.size());
  for (std::size_t k = 0; k < edges.size(); ++k) {
    const Point& initial = mesh.nodes[edges[k]];
    Point point = initial;
    for (int v = 0; v < 3; ++v) {
      const double step = position[3 * k + v];
      if (step == 0.0) continue;
      point[v] = std::fma(length_scale, step, initial[v]);
      if (!std::isfinite(point[v])) return PHX_MC_UNRESOLVED;
      if (std::abs(point[v] - initial[v]) > cap) point[v] = std::nextafter(point[v], initial[v]);
    }
    proposal.push_back({point, point != initial});
  }

  // Verify the rounded bank, not optimizer scratch or just the corner merit.
  // Untouched cells retain their original evidence; on failure so does the bank.
  auto failed = failing.begin();
  std::vector<std::int64_t> touched;
  for (const ActiveCell& cell : active) {
    if (std::none_of(cell.slots.begin(), cell.slots.end(), [&](std::size_t slot) {
          return slot != kFixed && proposal[slot].changed;
        })) continue;
    if (failed != failing.end() && *failed < cell.id) return PHX_MC_UNRESOLVED;
    if (failed != failing.end() && *failed == cell.id) ++failed;
    CornerMaps maps;
    double scale;
    if (!corner_maps(input, mesh, cell.id, maps, scale, &cell, proposal) ||
        sampled_misses(maps, Corners(maps, targets.scaled > 0.0), targets) != 0)
      return PHX_MC_UNRESOLVED;
    touched.push_back(cell.id);
  }
  if (failed != failing.end()) return PHX_MC_UNRESOLVED;

  if (moved_edges != nullptr) moved_edges->reserve(edges.size());
  for (std::size_t k = 0; k < edges.size(); ++k) {
    if (!proposal[k].changed) continue;
    mesh.nodes[edges[k]] = proposal[k].point;
    if (moved_edges != nullptr) moved_edges->push_back(edges[k]);
  }
  touched_cells = std::move(touched);
  unmet = {};
  return PHX_MC_OK;
}

}  // namespace phx::mc
