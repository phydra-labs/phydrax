// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include "metric_layer_grid.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "bounded_lbfgs.hpp"
#include "deterministic_float.hpp"
#include "exact_scalar.hpp"
#include "geometry3.hpp"
#include "metric_box_rate.hpp"
#include "metric_scalar_box.hpp"
#include "metric_float.hpp"
#include "predicates.hpp"

namespace phx::mc::metric {
namespace {

using Point = std::array<double, 3>;
using Row = std::int32_t;
using Edge = std::array<Row, 2>;
using Cell = std::array<Row, 4>;

constexpr double kLocationTolerance = 1e-10;
constexpr double kTransverseWidth = 0.85;
constexpr std::size_t kMaxGridCells = 125'000;
constexpr std::size_t kMaxBinMemberships = 4'000'000;
constexpr std::size_t kMaxColumnCandidates = 1'000'000;
constexpr std::size_t kMaxAxisIntervals = 1024;
constexpr double kInvalidMerit = std::numeric_limits<double>::quiet_NaN();

// Work in a unit box with a cyclic permutation of the physical axes, so
// coordinates 0 and 1 are tangential and the orientation is unchanged.
struct Field {
  Point low;
  Point high;
  Point extent;
  std::array<int, 3> axes;
  std::array<double, 2> tangent_squared;

  Point to_unit(const double* point) const {
    Point q;
    for (int k = 0; k < 3; ++k) {
      const int axis = axes[k];
      q[k] = (point[axis] - low[axis]) / extent[k];
    }
    return q;
  }

  Point to_world(const Point& q) const {
    Point point;
    for (int k = 0; k < 3; ++k) {
      const int axis = axes[k];
      point[axis] = q[k] == 0.0 ? low[axis] : q[k] == 1.0 ? high[axis] :
                                                           std::lerp(low[axis], high[axis], q[k]);
    }
    return point;
  }
};

std::optional<Field> read_field(const AdaptInput& input, const CarrierSource& source,
                                const Point& low, const Point& high) {
  if (source.cell_count() <= 0 || source.cell_count() > kMaxBoxSourceCells ||
      source.point_count() <= 0 || source.point_count() > 4 * source.cell_count() ||
      !geometry::point_in_domain(low) || !geometry::point_in_domain(high)) {
    return std::nullopt;
  }
  const auto points = static_cast<std::size_t>(source.point_count());
  // The transaction already matched the exact source against these banks.
  if (input.source_metrics.size() != 9 * points || input.source_logarithms.size() != 9 * points ||
      input.source_points.size() != 3 * points ||
      input.source_cells.size() != 4 * static_cast<std::size_t>(source.cell_count())) {
    return std::nullopt;
  }
  // Permit only last-place roundoff in the constant diagonal entries. No
  // off-diagonal entry is dropped from the model.
  constexpr double roundoff = 64 * std::numeric_limits<double>::epsilon();
  unsigned varying = 0;
  for (std::size_t row = 0; row < points; ++row) {
    for (int i = 0; i < 9; ++i) {
      const double metric = input.source_metrics[9 * row + i];
      const double log = input.source_logarithms[9 * row + i];
      if (i % 4 != 0) {
        if (metric != 0.0 || log != 0.0) return std::nullopt;
      } else {
        if (!(metric > 0.0) || !std::isfinite(metric) || !std::isfinite(log)) return std::nullopt;
        const double ref = input.source_metrics[i];
        const double ref_log = input.source_logarithms[i];
        if (std::abs(metric - ref) > roundoff * std::max(metric, ref) ||
            std::abs(log - ref_log) > roundoff * std::max({1.0, std::abs(log), std::abs(ref_log)})) {
          varying |= 1U << (i / 4);
        }
      }
    }
  }
  if (!std::has_single_bit(varying)) return std::nullopt;
  const int normal = std::countr_zero(varying);
  Field field{low, high, {}, {(normal + 1) % 3, (normal + 2) % 3, normal}, {}};
  for (int k = 0; k < 3; ++k) {
    const int axis = field.axes[k];
    field.extent[k] = high[axis] - low[axis];
    if (!(field.extent[k] > 0.0)) return std::nullopt;
    if (k == 2) continue;
    const double width = field.extent[k] * std::sqrt(input.source_metrics[4 * axis]);
    field.tangent_squared[k] = width * width;
    if (!(field.tangent_squared[k] > 0.0) || !std::isfinite(field.tangent_squared[k])) {
      return std::nullopt;
    }
  }
  return field;
}

struct SourceTet {
  std::array<Point, 4> corners;
  std::array<Point, 4> barycentric_grad;
  Point log_grad;
  double anchor_log_rate;

  double log_rate(const Point& q) const {
    return anchor_log_rate + geometry::dot(log_grad.data(), geometry::difference(q, corners[0]).data());
  }

  bool contains(const Point& q) const {
    const auto d = geometry::difference(q, corners[0]);
    for (int i = 0; i < 4; ++i) {
      if ((i == 0 ? 1.0 : 0.0) + geometry::dot(barycentric_grad[i].data(), d.data()) <
          -kLocationTolerance) return false;
    }
    return true;
  }

  // Intersect the column with the *shared triangles*, not two independently
  // inverted neighboring tetrahedra. A nearly vertical face amplifies even a
  // tiny barycentric discrepancy into a gap in the column partition.
  bool column_interval(const Point& base, double& start, double& end) const {
    const auto area = [&](const Point& a, const Point& b) {
      return (Dyadic(a[0]) - Dyadic(base[0])) * (Dyadic(b[1]) - Dyadic(base[1])) -
             (Dyadic(a[1]) - Dyadic(base[1])) * (Dyadic(b[0]) - Dyadic(base[0]));
    };
    start = 1.0;
    end = 0.0;
    for (const auto& side : kOrientedFacet) {
      const auto& a = corners[side[0]];
      const auto& b = corners[side[1]];
      const auto& c = corners[side[2]];
      const int sign = orient2d(a.data(), b.data(), c.data());
      if (sign == 0 || sign * orient2d(a.data(), b.data(), base.data()) < 0 ||
          sign * orient2d(b.data(), c.data(), base.data()) < 0 ||
          sign * orient2d(c.data(), a.data(), base.data()) < 0) continue;
      const Dyadic u = area(b, c), v = area(c, a), w = area(a, b);
      const double z = exact::rounded_quotient(
          u * Dyadic(a[2]) + v * Dyadic(b[2]) + w * Dyadic(c[2]), u + v + w);
      start = std::min(start, z);
      end = std::max(end, z);
    }
    return end > start;
  }
};

bool prepare_tet(const std::array<Point, 4>& p, const std::array<double, 4>& log_rate,
                 SourceTet& tet) {
  tet.corners = p;
  tet.anchor_log_rate = log_rate[0];
  return prepare_box_rate(p, log_rate, tet.barycentric_grad, tet.log_grad);
}

struct Interpolator {
  std::vector<SourceTet> tetrahedra;
  CarrierBoxGrid spatial;
  CarrierBoxGrid projected;

  bool prepare(const AdaptInput& input, const Field& field) {
    const std::size_t count = input.source_cells.size() / 4;
    std::vector<Point> points(input.source_points.size() / 3);
    for (std::size_t row = 0; row < points.size(); ++row) {
      points[row] = field.to_unit(input.source_points.data() + 3 * row);
      for (double q : points[row]) {
        if (!(q >= 0.0 && q <= 1.0)) return false;
      }
    }
    std::vector<double> lower(3 * count), upper(3 * count);
    tetrahedra.resize(count);
    const int component = 4 * field.axes[2];
    const double log_extent = deterministic::log(field.extent[2]);
    for (std::size_t cell = 0; cell < count; ++cell) {
      std::array<Point, 4> corners;
      std::array<double, 4> rates;
      for (int i = 0; i < 4; ++i) {
        const auto row = static_cast<std::size_t>(input.source_cells[4 * cell + i]);
        if (row >= points.size()) return false;
        corners[i] = points[row];
        rates[i] = log_extent + 0.5 * input.source_logarithms[9 * row + component];
      }
      if (!prepare_tet(corners, rates, tetrahedra[cell])) return false;
      for (int k = 0; k < 3; ++k) {
        const auto [lo, hi] = std::minmax({corners[0][k], corners[1][k], corners[2][k], corners[3][k]});
        lower[3 * cell + k] = lo;
        upper[3 * cell + k] = hi;
      }
    }
    spatial.prepare(lower, upper, 0b111, kMaxBinMemberships);
    projected.prepare(lower, upper, 0b011, kMaxBinMemberships);
    return spatial.indexed() && projected.indexed();
  }

  Row locate(const Point& q, Row hint) const {
    if (hint >= 0 && tetrahedra[hint].contains(q)) return hint;
    for (const auto row : spatial.candidates(q.data())) {
      if (tetrahedra[row].contains(q)) return static_cast<Row>(row);
    }
    return -1;
  }
};

struct Node {
  Point q;
  double normal_squared;
  Row tetrahedron;
};

struct ColumnPiece {
  double end;
  double mass;
  double log_rate;
  Row tetrahedron;
};

struct ColumnHit {
  double start;
  double end;
  Row tetrahedron;
};


class Columns {
 public:
  bool prepare(const Interpolator& source, std::size_t nx, std::size_t ny);
  double mean_mass() const { return mean_mass_; }
  void seed(std::size_t levels, std::vector<Node>& nodes, const Interpolator& source) const;
  bool sample(std::size_t column, Point q, const Interpolator& source, Node& node) const;

 private:
  bool append(const Point& base, const Interpolator& source, std::vector<ColumnHit>& hits);

  std::vector<Point> bases_;
  std::vector<std::size_t> offsets_;
  std::vector<ColumnPiece> pieces_;
  double mean_mass_ = 0.0;
};

bool Columns::append(const Point& base, const Interpolator& source, std::vector<ColumnHit>& hits) {
  hits.clear();
  for (const auto row : source.projected.candidates(base.data())) {
    ColumnHit hit{0.0, 0.0, static_cast<Row>(row)};
    if (source.tetrahedra[row].column_interval(base, hit.start, hit.end)) hits.push_back(hit);
  }
  std::sort(hits.begin(), hits.end(), [](const ColumnHit& a, const ColumnHit& b) {
    if (a.start != b.start) return a.start < b.start;
    if (a.end != b.end) return a.end > b.end;
    return a.tetrahedron < b.tetrahedron;
  });
  double end = 0.0;
  double mass = 0.0;
  for (const auto& hit : hits) {
    if (hit.end <= end) continue;
    if (hit.start > end) return false;
    const auto& tet = source.tetrahedra[hit.tetrahedron];
    const double log_rate = tet.log_rate({base[0], base[1], end});
    const double right = hit.end;
    const double span = right - end;
    const double area = deterministic::exp(log_rate) * span *
                        integrated_exponential(tet.log_grad[2] * span);
    if (!(area > 0.0) || !std::isfinite(area) || !(mass + area > mass)) return false;
    mass += area;
    if (!std::isfinite(mass)) return false;
    pieces_.push_back({right, mass, log_rate, hit.tetrahedron});
    end = right;
    if (end == 1.0) break;
  }
  if (end != 1.0) return false;
  offsets_.push_back(pieces_.size());
  mean_mass_ += mass;
  return true;
}

bool Columns::prepare(const Interpolator& source, std::size_t nx, std::size_t ny) {
  const std::size_t count = (nx + 1) * (ny + 1);
  bases_.reserve(count);
  offsets_.reserve(count + 1);
  offsets_.push_back(0);
  std::size_t candidates = 0;
  std::size_t largest = 0;
  for (std::size_t i = 0; i <= nx; ++i) {
    for (std::size_t j = 0; j <= ny; ++j) {
      auto& base = bases_.emplace_back(Point{static_cast<double>(i) / static_cast<double>(nx),
                                             static_cast<double>(j) / static_cast<double>(ny), 0.0});
      const auto size = source.projected.candidates(base.data()).size();
      if (size > kMaxColumnCandidates - candidates) return false;
      candidates += size;
      largest = std::max(largest, size);
    }
  }
  pieces_.reserve(count);
  std::vector<ColumnHit> hits;
  hits.reserve(largest);
  for (const auto& base : bases_) {
    if (!append(base, source, hits)) return false;
  }
  mean_mass_ /= static_cast<double>(count);
  return std::isfinite(mean_mass_);
}

void Columns::seed(std::size_t levels, std::vector<Node>& nodes, const Interpolator& source) const {
  std::vector<double> shared(levels, 0.0);
  const auto columns = std::span<const ColumnPiece>(pieces_);
  nodes.resize(bases_.size() * levels);
  for (std::size_t column = 0; column < bases_.size(); ++column) {
    const auto pieces = columns.subspan(offsets_[column], offsets_[column + 1] - offsets_[column]);
    auto& low = nodes[column * levels];
    low.q = bases_[column];
    std::size_t piece = 0;
    for (std::size_t level = 1; level + 1 < levels; ++level) {
      const double target = pieces.back().mass * static_cast<double>(level) /
                            static_cast<double>(levels - 1);
      while (pieces[piece].mass < target) ++piece;
      const double left = piece == 0 ? 0.0 : pieces[piece - 1].end;
      const double left_mass = piece == 0 ? 0.0 : pieces[piece - 1].mass;
      const double rate = deterministic::exp(pieces[piece].log_rate);
      const double slope = source.tetrahedra[pieces[piece].tetrahedron].log_grad[2];
      const double mass = target - left_mass;
      const double depth = slope == 0.0 ? mass / rate :
                           deterministic::log1p(slope * (mass / rate)) / slope;
      const double height = std::clamp(left + depth, left, pieces[piece].end);
      nodes[column * levels + level].q = {low.q[0], low.q[1], height};
      shared[level] += height;
    }
    nodes[(column + 1) * levels - 1].q = {low.q[0], low.q[1], 1.0};
  }
  for (double& height : shared) height /= static_cast<double>(bases_.size());
  for (std::size_t column = 0; column < bases_.size(); ++column) {
    for (std::size_t level = 1; level + 1 < levels; ++level) {
      double& height = nodes[column * levels + level].q[2];
      height = std::lerp(shared[level], height, 0.25);
    }
  }
}

bool Columns::sample(std::size_t column, Point q, const Interpolator& source, Node& node) const {
  const auto first = pieces_.begin() + static_cast<std::ptrdiff_t>(offsets_[column]);
  const auto end = pieces_.begin() + static_cast<std::ptrdiff_t>(offsets_[column + 1]);
  const auto it = std::lower_bound(first, end, q[2], [](const ColumnPiece& piece, double z) {
    return piece.end < z;
  });
  if (it == end) return false;
  const auto& tet = source.tetrahedra[it->tetrahedron];
  const double left = it == first ? 0.0 : (it - 1)->end;
  const double metric = deterministic::exp(2.0 * (it->log_rate + tet.log_grad[2] * (q[2] - left)));
  if (!(metric > 0.0) || !std::isfinite(metric)) return false;
  node = {q, metric, it->tetrahedron};
  return true;
}

std::size_t interval_count(double integral, double width) {
  const double count = std::floor(integral / width + 0.5);
  if (!(count >= 0.0 && count <= kMaxAxisIntervals)) return 0;
  return std::max<std::size_t>(1, static_cast<std::size_t>(count));
}

std::vector<Cell> checkerboard(std::size_t nx, std::size_t ny, std::size_t nz) {
  constexpr std::array<std::array<int, 4>, 2> centers{{{0, 3, 5, 6}, {1, 4, 2, 7}}};
  const std::size_t sy = nz + 1;
  const std::size_t sx = (ny + 1) * sy;
  const std::array<std::size_t, 8> corner{0, 1, sy, sy + 1, sx, sx + 1, sx + sy, sx + sy + 1};
  std::vector<Cell> cells;
  cells.reserve(5 * nx * ny * nz);
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < ny; ++j) {
      for (std::size_t k = 0; k < nz; ++k) {
        const auto row = [&](unsigned a) { return static_cast<Row>(i * sx + j * sy + k + corner[a]); };
        const int parity = static_cast<int>((i + j + k) & 1);
        const auto& center = centers[parity];
        cells.push_back({row(center[0]), row(center[1]), row(center[2]), row(center[3])});
        for (unsigned a = 0; a < 8; ++a) {
          if ((std::popcount(a) & 1) == parity) continue;
          Cell cell{row(a), row(a ^ 4), row(a ^ 2), row(a ^ 1)};
          if (parity == 0) std::swap(cell[1], cell[2]);
          cells.push_back(cell);
        }
      }
    }
  }
  return cells;
}

std::vector<Edge> unique_edges(std::span<const Cell> cells) {
  std::vector<Edge> edges;
  edges.reserve(6 * cells.size());
  for (const auto& cell : cells) {
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        edges.push_back({std::min(cell[i], cell[j]), std::max(cell[i], cell[j])});
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

struct Band {
  double lower;
  double upper;
  double lower_squared;
  double upper_squared;

  Band(double lo, double hi) : lower(lo), upper(hi), lower_squared(lo * lo), upper_squared(hi * hi) {}
};

struct EdgeModel {
  Point delta;
  double axial_squared;
  double residual;
  double factor_a;
  double factor_b;
};

EdgeModel model_edge(const Node& a, const Node& b, const Field& field, const Band& band) {
  const auto d = geometry::difference(b.q, a.q);
  const double tangent = (d[0] * d[0]) * field.tangent_squared[0] +
                         (d[1] * d[1]) * field.tangent_squared[1];
  const double axial = d[2] * d[2];
  const double a2 = tangent + axial * a.normal_squared;
  const double b2 = tangent + axial * b.normal_squared;
  if (a2 >= band.lower_squared && b2 >= band.lower_squared &&
      a2 <= band.upper_squared && b2 <= band.upper_squared) return {};
  if (!(a2 > 0.0 && b2 > 0.0)) return {d, axial, kInvalidMerit, 0.0, 0.0};
  const double p = std::sqrt(a2);
  const double q = std::sqrt(b2);
  const double ratio = q / p;
  const bool close = std::abs(ratio - 1.0) <= 1e-6;
  const double log_ratio = close ? 0.0 : deterministic::log(ratio);
  const double mean = close ? 0.5 * (p + q) : (q - p) / log_ratio;
  if (!std::isfinite(mean)) return {d, axial, kInvalidMerit, 0.0, 0.0};
  const double residual = mean - std::clamp(mean, band.lower, band.upper);
  if (residual == 0.0) return {};
  const double f = close ? 0.5 : (mean / p - 1.0) / log_ratio;
  const double g = close ? 0.5 : (1.0 - mean / q) / log_ratio;
  return {d, axial, residual, f / p, g / q};
}

struct Mover {
  Row row;
  Point origin;
  Point scale;
};

struct Placement {
  const Field& field;
  const Interpolator& source;
  std::span<const Edge> edges;
  std::span<Node> nodes;
  std::span<const Mover> movers;
  std::span<const Row> mover_at;
  Band band;
  const Columns* columns;
  std::size_t levels;

  bool update(std::span<const double> x) const;
  double merit(std::span<const double> x, std::span<double> gradient) const;
};

bool Placement::update(std::span<const double> x) const {
  const std::size_t width = columns ? 1 : 3;
  for (std::size_t i = 0; i < movers.size(); ++i) {
    const auto& move = movers[i];
    auto& node = nodes[move.row];
    auto q = move.origin;
    if (columns) q[2] += move.scale[2] * x[i];
    else {
      for (int k = 0; k < 3; ++k) q[k] += move.scale[k] * x[width * i + k];
    }
    if (q == node.q) continue;
    if (columns) {
      if (!columns->sample(static_cast<std::size_t>(move.row) / levels, q, source, node)) return false;
    } else {
      const Row cell = source.locate(q, node.tetrahedron);
      if (cell < 0) return false;
      const double metric = deterministic::exp(2.0 * source.tetrahedra[cell].log_rate(q));
      if (!(metric > 0.0) || !std::isfinite(metric)) return false;
      node = {q, metric, cell};
    }
  }
  return true;
}

// The endpoint metric changes with its position. The gradient of the P1
// half-log density contributes in every free Cartesian direction, not just
// in the direction of the diagonal that changes.
double Placement::merit(std::span<const double> x, std::span<double> gradient) const {
  if (!update(x)) return kInvalidMerit;
  std::fill(gradient.begin(), gradient.end(), 0.0);
  double value = 0.0;
  const int first = columns ? 2 : 0;
  const std::size_t width = columns ? 1 : 3;
  for (const auto& [u, v] : edges) {
    const auto& a = nodes[u];
    const auto& b = nodes[v];
    const auto m = model_edge(a, b, field, band);
    if (m.residual == 0.0) continue;
    if (!std::isfinite(m.residual)) return kInvalidMerit;
    value += m.residual * m.residual;
    const auto& ga = source.tetrahedra[a.tetrahedron].log_grad;
    const auto& gb = source.tetrahedra[b.tetrahedron].log_grad;
    for (int k = first; k < 3; ++k) {
      const double da = k == 2 ? a.normal_squared : field.tangent_squared[k];
      const double db = k == 2 ? b.normal_squared : field.tangent_squared[k];
      const double base = m.delta[k] * (m.factor_a * da + m.factor_b * db);
      const auto add = [&](Row row, double derivative) {
        const Row i = mover_at[row];
        if (i >= 0) gradient[width * i + (k - first)] +=
            m.residual * derivative * movers[i].scale[k];
      };
      add(u, -base + m.factor_a * a.normal_squared * m.axial_squared * ga[k]);
      add(v, base + m.factor_b * b.normal_squared * m.axial_squared * gb[k]);
    }
  }
  return 0.5 * value;
}

bool correct(const Field& field, const Interpolator& source, std::span<const Edge> edges,
             std::span<Node> nodes, std::span<const std::uint8_t> active, const Band& band,
             const Columns* columns, std::size_t levels, double bound, int steps, double tolerance) {
  std::vector<Mover> movers;
  movers.reserve(active.empty() ? nodes.size() :
                 nodes.size() - std::count(active.begin(), active.end(), std::uint8_t{0}));
  std::vector<Row> at(nodes.size(), -1);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!active.empty() && active[i] == 0) continue;
    const auto& node = nodes[i];
    Point scale{};
    for (int k = columns ? 2 : 0; k < 3; ++k) {
      const double room = std::min(node.q[k], 1.0 - node.q[k]);
      if (room <= 0.0) continue;  // Cartesian wall coordinate
      const double size = 1.0 / std::sqrt(k == 2 ? node.normal_squared : field.tangent_squared[k]);
      scale[k] = std::min(size, 0.99 * room / bound);
    }
    if (scale == Point{}) continue;
    at[i] = static_cast<Row>(movers.size());
    movers.push_back({static_cast<Row>(i), node.q, scale});
  }
  if (movers.empty()) return true;
  const Placement placement{field, source, edges, nodes, movers, at, band, columns, levels};
  std::vector<double> x((columns ? 1 : 3) * movers.size(), 0.0);
  const auto result = detail::bounded_lbfgs(x, bound, 0.15, steps, tolerance,
      [&](std::span<const double> trial, std::span<double> gradient) {
        return placement.merit(trial, gradient);
      });
  // A stalled optimizer may still supply a valid placement. Only the
  // accepted iterate is retained, and the snapped grid is measured anew.
  return std::isfinite(result.value) && placement.update(x);
}

bool correct_star(const Field& field, const Interpolator& source, std::span<const Edge> edges,
                  std::span<Node> nodes, const Band& band, double margin) {
  std::vector<std::uint8_t> active(nodes.size(), 0);
  bool found = false;
  for (const auto& [a, b] : edges) {
    if (model_edge(nodes[a], nodes[b], field, band).residual == 0.0) continue;
    active[a] = active[b] = 1;
    found = true;
  }
  if (!found) return true;
  // Two synchronous rings around the offending edges. Include every edge
  // incident to a mover in the merit, including those crossing the ring.
  for (std::uint8_t ring = 1; ring <= 2; ++ring) {
    for (const auto& [a, b] : edges) {
      if (active[a] == ring && active[b] == 0) active[b] = ring + 1;
      if (active[b] == ring && active[a] == 0) active[a] = ring + 1;
    }
  }
  std::vector<Edge> star;
  star.reserve(edges.size());
  for (const auto& edge : edges) {
    if (active[edge[0]] != 0 || active[edge[1]] != 0) star.push_back(edge);
  }
  const double residual = margin / 4.0;
  return correct(field, source, star, nodes, active, band, nullptr, 0, 0.15, 120,
                 0.5 * residual * residual);
}

bool bind_box_point(const AdaptInput& input, const CarrierSource& source, const Point& low,
                    const Point& high, const Point& goal, const Point& current,
                    unsigned fixed, Row hint, Carrier& carrier, std::array<double, 9>& tensor) {
  auto hits = source.carriers(goal.data(), current.data(), {}, hint,
                              CarrierOrder::kFineFirst, 1);
  if (hits.empty()) {
    hits = source.carriers(goal.data(), current.data(), {}, -1, CarrierOrder::kFineFirst, 1);
  }
  if (hits.empty()) return false;
  carrier = hits.front();
  if (!geometry::point_in_domain(carrier.point)) return false;
  std::array<StencilTerm, 4> stencil;
  std::array<double, 4> weights;
  std::array<double, 36> metrics, logs;
  for (int i = 0; i < carrier.size; ++i) {
    const auto row = carrier.rows[i];
    weights[i] = std::ldexp(static_cast<double>(carrier.weights[i]), -carrier.bits);
    stencil[i] = {static_cast<Row>(row), weights[i]};
    std::copy_n(input.source_metrics.data() + 9 * row, 9, metrics.data() + 9 * i);
    std::copy_n(input.source_logarithms.data() + 9 * row, 9, logs.data() + 9 * i);
  }
  for (int a = 0; a < 3; ++a) {
    const double point = carrier.point[a];
    if ((fixed & (1U << a)) == 0) {
      if (!(point > low[a] && point < high[a])) return false;
      continue;
    }
    if (point != goal[a]) return false;
    for (int i = 0; i < carrier.size; ++i) {
      if (source.point(carrier.rows[i])[a] != goal[a]) return false;
    }
    bool exact = false;
    if (source.stencil_coordinate(std::span(stencil).first(carrier.size), a, true, exact) != goal[a] ||
        !exact) return false;
  }
  return interpolate(carrier.size, weights.data(), metrics.data(), logs.data(), tensor.data());
}

bool bind_node(const AdaptInput& input, const CarrierSource& source, const Field& field,
               const Node& node, Carrier& carrier, std::array<double, 9>& tensor) {
  const Point goal = field.to_world(node.q);
  auto current = goal;
  const auto axis = static_cast<int>(std::max_element(field.extent.begin(), field.extent.end()) -
                                     field.extent.begin());
  current[field.axes[axis]] = node.q[axis] <= 0.5 ? field.high[field.axes[axis]] :
                                                    field.low[field.axes[axis]];
  unsigned fixed = 0;
  for (int k = 0; k < 3; ++k) {
    if (node.q[k] == 0.0 || node.q[k] == 1.0) fixed |= 1U << field.axes[k];
  }
  return bind_box_point(input, source, field.low, field.high, goal, current, fixed,
                        node.tetrahedron, carrier, tensor);
}

bool verify_grid(const MetricBoxGrid& grid, std::span<const Edge> edges, double lower) {
  for (const auto& [a, b] : edges) {
    const double measure = length(grid.vertices[a].point.data(), grid.vertices[b].point.data(),
                                  grid.tensors[a].data(), grid.tensors[b].data());
    if (!(measure >= lower && measure <= kUpperLength)) return false;
  }
  for (const auto& [a, b, c, d] : grid.cells) {
    if (orient3d_filtered(grid.vertices[a].point.data(), grid.vertices[b].point.data(),
                          grid.vertices[c].point.data(), grid.vertices[d].point.data()) <= 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::optional<MetricBoxGrid> propose_metric_layer_grid(
    const AdaptInput& input, const CarrierSource& source, const Point& low, const Point& high) {
  const double lower = std::max(input.lower, kLowerLength);
  if (!(lower < kUpperLength)) return std::nullopt;
  const auto field = read_field(input, source, low, high);
  if (!field) return std::nullopt;
  const double tx = std::sqrt(field->tangent_squared[0]);
  const double ty = std::sqrt(field->tangent_squared[1]);
  const std::size_t nx = interval_count(tx, kTransverseWidth);
  const std::size_t ny = interval_count(ty, kTransverseWidth);
  if (nx == 0 || ny == 0 || 5 * nx * ny > kMaxGridCells) return std::nullopt;
  Interpolator interpolator;
  if (!interpolator.prepare(input, *field)) return std::nullopt;
  Columns columns;
  if (!columns.prepare(interpolator, nx, ny)) return std::nullopt;
  const double width = 0.5 * (tx / static_cast<double>(nx) + ty / static_cast<double>(ny));
  const std::size_t nz = interval_count(columns.mean_mass(), width);
  if (nz == 0 || 5 * nx * ny * nz > kMaxGridCells) return std::nullopt;

  MetricBoxGrid grid;
  grid.cells = checkerboard(nx, ny, nz);
  const auto edges = unique_edges(grid.cells);
  std::vector<Node> nodes;
  columns.seed(nz + 1, nodes, interpolator);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!columns.sample(i / (nz + 1), nodes[i].q, interpolator, nodes[i])) return std::nullopt;
  }
  const double gap = kUpperLength - lower;
  const double axial_margin = std::min(0.003, gap / 8.0);
  const double star_margin = std::min(0.001, gap / 16.0);
  const Band axial(lower + axial_margin, kUpperLength - axial_margin);
  const Band star(lower + star_margin, kUpperLength - star_margin);
  if (!correct(*field, interpolator, edges, nodes, {}, axial, &columns, nz + 1, 0.8, 350, 1e-14) ||
      !correct_star(*field, interpolator, edges, nodes, star, star_margin)) return std::nullopt;

  grid.vertices.resize(nodes.size());
  grid.tensors.resize(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!bind_node(input, source, *field, nodes[i], grid.vertices[i], grid.tensors[i])) {
      return std::nullopt;
    }
  }
  if (!verify_grid(grid, edges, lower)) return std::nullopt;
  return grid;
}

std::optional<MetricBoxGrid> propose_scalar_box_grid(
    const AdaptInput& input, const CarrierSource& source, const Point& low, const Point& high) {
  const double lower = std::max(input.lower, kLowerLength);
  ScalarBoxField field;
  if (!(lower < kUpperLength) || !field.prepare(input, low, high)) return std::nullopt;
  ScalarBoxSites sites;
  if (!seed_scalar_box(input, field, sites) || !relax_scalar_box(field, sites, 6)) {
    return std::nullopt;
  }
  if (!correct_scalar_box(field, sites, lower)) {
    // Moving sites cannot create missing wall sites. Only rebuild when a
    // separated, chart-bound proposal can actually fill a boundary gap.
    if (!fill_scalar_wall_gaps(field, sites) || !relax_scalar_box(field, sites, 6) ||
        !correct_scalar_box(field, sites, lower)) return std::nullopt;
  }
  MetricBoxGrid grid;
  grid.cells = std::move(sites.cells);
  grid.vertices.resize(sites.points.size());
  grid.tensors.resize(sites.points.size());
  const auto extent = geometry::difference(high, low);
  const auto axis = std::max_element(extent.begin(), extent.end()) - extent.begin();
  for (std::size_t row = 0; row < sites.points.size(); ++row) {
    const auto& goal = sites.points[row];
    auto current = goal;
    current[axis] = goal[axis] - low[axis] <= high[axis] - goal[axis] ? high[axis] : low[axis];
    unsigned fixed = 0;
    for (int k = 0; k < 3; ++k) {
      if (goal[k] == low[k] || goal[k] == high[k]) fixed |= 1U << k;
    }
    if (!bind_box_point(input, source, low, high, goal, current, fixed, sites.samples[row].cell,
                        grid.vertices[row], grid.tensors[row])) return std::nullopt;
  }
  // The transaction checks every snapped edge in stable native-ID order.
  // The scalar proposal additionally owns its post-snap shape floor.
  for (const auto& cell : grid.cells) {
    std::array<double, 12> corners;
    for (int i = 0; i < 4; ++i) {
      std::copy_n(grid.vertices[cell[i]].point.data(), 3, corners.data() + 3 * i);
    }
    double det, relative;
    relative_determinant(corners.data(), det, relative);
    if (!(det > 0.0 && relative > 0.0 && geometric_shape(corners.data(), det) >= 0.60)) {
      return std::nullopt;
    }
  }
  return grid;
}

}  // namespace phx::mc::metric
