//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// P1 zero insertion on an existing tetrahedral complex. Every source face is
// constrained and every source edge is a subsegment, so the target is the exact
// subdivision obtained by bisecting each crossing source edge (a, b) at its
// exact source-owned P1 root
//   S = (|v_b| A + |v_a| B) / (|v_a| + |v_b|),
// in request edge order. S is never a rounded line parameter: it is named by
// the edge and the exact homogeneous weights (|v_b|, |v_a|) of the source
// values, so the P1 field vanishes on it exactly. Bisecting a positive
// tetrahedron at an interior point of one of its edges yields two positive
// children, hence the exact subdivision is a conforming refinement of the
// source by construction. The binary64 vertex is the correctly rounded
// carrier of S, with an outward bound of its distance from S. TetMesh owns
// exact validation of the source complex; output is published only after
// every carrier cell is positively oriented and all carriers are distinct.
// In TOPOLOGY_ONLY mode the caller owns the exact source positions (the
// binary64 points are only their rounded views), so no carrier or deviation
// is computed or published; source, topology and protection validation are
// unchanged. A refusal discards the private native state.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "capi_guard.hpp"
#include "dyadic.hpp"
#include "geometry3.hpp"
#include "tet_mesh.hpp"

namespace {
using phx::mc::Dyadic;
using Cell = std::array<int32_t, 4>;


struct LevelSetRoots {
  std::vector<std::array<int32_t, 2>> sources;
  std::vector<std::array<double, 2>> weights;
  std::vector<double> deviations;
  std::vector<double> points;
  std::vector<int64_t> edges;  // crossing edge row of each vertex; -1 for source vertices
};


// Whether the request publishes binary64 carriers and their deviations.
bool publishes_carriers(const phx_mc_level_set_split_3d_request& request) {
  return request.output_mode == PHX_MC_LEVEL_SET_FULL_CARRIERS;
}

int32_t validate_level_set_request(const phx_mc_level_set_split_3d_request& request) {
  if (request.point_count > std::numeric_limits<int32_t>::max() - 1 ||
      request.cell_count > std::numeric_limits<int32_t>::max() ||
      request.face_count > std::numeric_limits<int32_t>::max() ||
      request.edge_count > std::numeric_limits<int32_t>::max()) {
    return PHX_MC_INDEX_RANGE_ERROR;
  }
  const bool carriers = publishes_carriers(request);
  const bool topology = request.output_mode == PHX_MC_LEVEL_SET_TOPOLOGY_ONLY;
  if ((!carriers && !topology) || (request.out_points == nullptr) == carriers ||
      (request.out_deviations == nullptr) == carriers || request.point_count < 4 ||
      request.cell_count < 1 || request.face_count < 4 || request.edge_count < 6 ||
      request.points == nullptr || request.values == nullptr || request.cells == nullptr ||
      request.faces == nullptr || request.edges == nullptr || request.protected_edges == nullptr ||
      request.out_cells == nullptr || request.out_parents == nullptr ||
      request.out_sources == nullptr || request.out_weights == nullptr ||
      request.out_point_count == nullptr || request.out_cell_count == nullptr ||
      request.failed_edge == nullptr ||
      !phx::mc::addressable(request.point_count, 3, sizeof(double)) ||
      !phx::mc::addressable(request.point_count, 1, sizeof(double)) ||
      !phx::mc::addressable(request.cell_count, 4, sizeof(int32_t)) ||
      !phx::mc::addressable(request.face_count, 3, sizeof(int32_t)) ||
      !phx::mc::addressable(request.edge_count, 2, sizeof(int32_t)) ||
      !phx::mc::addressable(request.edge_count, 1, sizeof(int8_t))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.out_point_capacity < 0 || request.out_cell_capacity < 0) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (request.out_point_capacity < request.point_count ||
      request.out_cell_capacity < request.cell_count) {
    return PHX_MC_CAPACITY_EXCEEDED;
  }
  if (!phx::mc::addressable(request.out_point_capacity, 3, sizeof(double)) ||
      !phx::mc::addressable(request.out_point_capacity, 2, sizeof(int32_t)) ||
      !phx::mc::addressable(request.out_point_capacity, 2, sizeof(double)) ||
      !phx::mc::addressable(request.out_cell_capacity, 4, sizeof(int32_t)) ||
      !phx::mc::addressable(request.out_cell_capacity, 1, sizeof(int32_t)) ||
      (carriers &&
       !phx::mc::addressable(request.out_point_capacity, 1, sizeof(double)))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return PHX_MC_OK;
}

int32_t prepare_level_set_sources(const phx_mc_level_set_split_3d_request& request,
                                  std::vector<int32_t>& regions,
                                  std::vector<int32_t>& face_sources,
                                  std::vector<int32_t>& edge_sources) {
  for (int64_t i = 0; i < request.point_count; ++i) {
    if (!std::isfinite(request.values[i])) return PHX_MC_INVALID_INPUT;
    for (int k = 0; k < 3; ++k) {
      if (!phx::mc::coordinate_in_domain(request.points[3 * i + k])) return PHX_MC_INVALID_INPUT;
    }
  }
  for (int32_t i = 0; i < request.cell_count; ++i) regions[static_cast<std::size_t>(i)] = i;
  for (int32_t i = 0; i < request.face_count; ++i) face_sources[static_cast<std::size_t>(i)] = i;
  for (int32_t i = 0; i < request.edge_count; ++i) edge_sources[static_cast<std::size_t>(i)] = i;
  return PHX_MC_OK;
}


bool even_significand(double value) { return (std::bit_cast<std::uint64_t>(value) & 1U) == 0; }

Dyadic magnitude(const Dyadic& value) { return value.sign() < 0 ? -value : value; }

// A binary64 value within a few units of the exact quotient (denominator > 0);
// its finite saturation keeps the exact neighbour search well defined.
double quotient_estimate(const Dyadic& numerator, const Dyadic& denominator) {
  std::int64_t numerator_exponent = 0;
  std::int64_t denominator_exponent = 0;
  const double top = numerator.fraction(numerator_exponent);
  if (top == 0.0) return 0.0;
  const double bottom = denominator.fraction(denominator_exponent);
  const auto shift = std::clamp<std::int64_t>(numerator_exponent - denominator_exponent,
                                              std::numeric_limits<int>::min(),
                                              std::numeric_limits<int>::max());
  const double candidate = std::ldexp(top / bottom, static_cast<int>(shift));
  return std::isfinite(candidate) ? candidate
                                  : std::copysign(std::numeric_limits<double>::max(), candidate);
}

// Sign of |n - d y| - |n - d x| for d > 0: negative when y is nearer n / d.
int quotient_closer(const Dyadic& numerator, const Dyadic& denominator, double y, double x) {
  return (magnitude(numerator - Dyadic(y) * denominator) -
          magnitude(numerator - Dyadic(x) * denominator))
      .sign();
}

// The binary64 value nearest the exact quotient numerator / denominator
// (denominator > 0), ties to even, decided by exact residual comparison.
double nearest_quotient(const Dyadic& numerator, const Dyadic& denominator) {
  double x = quotient_estimate(numerator, denominator);
  for (;;) {
    const double down = phx::mc::geometry::next_down(x);
    const double up = phx::mc::geometry::next_up(x);
    const int below = quotient_closer(numerator, denominator, down, x);
    const int above = quotient_closer(numerator, denominator, up, x);
    if (below < 0 || (below == 0 && even_significand(down))) {
      x = down;
    } else if (above < 0 || (above == 0 && even_significand(up))) {
      x = up;
    } else {
      return x;
    }
  }
}

// Outward binary64 bound of |numerator| / denominator (denominator > 0). The
// truncated leading fractions differ from the exact magnitudes by less than
// one adjacent spacing, which the directed steps cover, including underflow.
double outward_quotient(const Dyadic& numerator, const Dyadic& denominator) {
  if (numerator.sign() == 0) return 0.0;
  std::int64_t numerator_exponent = 0;
  std::int64_t denominator_exponent = 0;
  const double upper =
      phx::mc::geometry::next_up(std::fabs(numerator.fraction(numerator_exponent)));
  const double lower = std::nextafter(std::fabs(denominator.fraction(denominator_exponent)), 0.0);
  const double ratio = phx::mc::geometry::next_up(upper / lower);
  const auto shift = std::clamp<std::int64_t>(numerator_exponent - denominator_exponent,
                                              std::numeric_limits<int>::min(),
                                              std::numeric_limits<int>::max());
  return phx::mc::geometry::next_up(std::ldexp(ratio, static_cast<int>(shift)));
}

// Outward Euclidean norm of nonnegative outward components, as the shared
// source deviation bound evaluates it.
double outward_norm(const double* components) {
  const double scale = std::max({components[0], components[1], components[2]});
  if (scale == 0.0) return 0.0;
  double sum = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double ratio = components[k] / scale;
    sum += ratio * ratio;
  }
  return phx::mc::geometry::next_up(scale * std::sqrt(sum) * (1.0 + 0x1p-49));
}

// The exact homogeneous weights of the root of edge (a, b): the swapped
// source magnitudes, on which the P1 field vanishes exactly.
std::array<double, 2> level_set_root_weights(const phx_mc_level_set_split_3d_request& request,
                                             int32_t a, int32_t b) {
  return {std::abs(request.values[b]), std::abs(request.values[a])};
}

// The exact root of edge (a, b) for any finite opposite-sign values, its
// correctly rounded carrier and the outward bound of |carrier - S|. Exact
// dyadic arithmetic has no exponent window; a carrier outside the coordinate
// domain is INVALID_INPUT, as for every input coordinate.
int32_t construct_level_set_root(const phx_mc_level_set_split_3d_request& request, int32_t a,
                                 int32_t b, const std::array<double, 2>& weights, double* carrier,
                                 double& deviation) {
  const Dyadic first(weights[0]);
  const Dyadic second(weights[1]);
  const Dyadic denominator = first + second;
  double components[3];
  for (int k = 0; k < 3; ++k) {
    const Dyadic numerator =
        first * Dyadic(request.points[3 * a + k]) + second * Dyadic(request.points[3 * b + k]);
    carrier[k] = nearest_quotient(numerator, denominator);
    if (!phx::mc::coordinate_in_domain(carrier[k])) return PHX_MC_INVALID_INPUT;
    components[k] = outward_quotient(numerator - Dyadic(carrier[k]) * denominator, denominator);
  }
  deviation = outward_norm(components);
  return PHX_MC_OK;
}

class LevelSetSplit {
 public:
  explicit LevelSetSplit(const phx_mc_level_set_split_3d_request& request)
      : request_(request), carriers_(publishes_carriers(request)) {}

  int32_t run() {
    int32_t status = initialize();
    if (status == PHX_MC_OK) status = insert_roots();
    if (status == PHX_MC_OK) status = certify_carriers();
    if (status == PHX_MC_OK) status = certify_sides();
    if (status == PHX_MC_OK) publish();
    return status;
  }

 private:
  int32_t initialize() {
    const auto count = static_cast<std::size_t>(request_.point_count);
    roots_.sources.reserve(count);
    roots_.weights.reserve(count);
    roots_.edges.reserve(count);
    if (carriers_) {
      roots_.deviations.reserve(count);
      roots_.points.assign(request_.points, request_.points + 3 * request_.point_count);
    }
    for (int32_t i = 0; i < request_.point_count; ++i) {
      roots_.sources.push_back({i, i});
      roots_.weights.push_back({1.0, 0.0});
      if (carriers_) roots_.deviations.push_back(0.0);
      roots_.edges.push_back(-1);
    }
    cells_.reserve(static_cast<std::size_t>(request_.cell_count));
    parents_.reserve(static_cast<std::size_t>(request_.cell_count));
    incident_.resize(count);
    for (int32_t c = 0; c < request_.cell_count; ++c) {
      const Cell cell{request_.cells[4 * c], request_.cells[4 * c + 1],
                      request_.cells[4 * c + 2], request_.cells[4 * c + 3]};
      cells_.push_back(cell);
      parents_.push_back(c);
      for (int32_t v : cell) incident_[static_cast<std::size_t>(v)].push_back(c);
    }
    return PHX_MC_OK;
  }

  int32_t insert_roots() {
    for (int64_t e = 0; e < request_.edge_count; ++e) {
      const int32_t a = request_.edges[2 * e], b = request_.edges[2 * e + 1];
      if (a < 0 || b < 0 || a >= request_.point_count || b >= request_.point_count || a == b) {
        return PHX_MC_INVALID_INPUT;
      }
      if (request_.values[a] == 0.0 || request_.values[b] == 0.0 ||
          std::signbit(request_.values[a]) == std::signbit(request_.values[b])) {
        continue;
      }
      *request_.failed_edge = e;
      if (request_.protected_edges[e] != 0) return PHX_MC_CONSTRAINT_INTERSECTION;
      const int32_t status = split_edge(e, a, b);
      if (status != PHX_MC_OK) return status;
    }
    return PHX_MC_OK;
  }

  int32_t split_edge(int64_t e, int32_t a, int32_t b) {
    const std::array<double, 2> weights = level_set_root_weights(request_, a, b);
    double carrier[3] = {};
    double deviation = 0.0;
    if (carriers_) {
      const int32_t constructed =
          construct_level_set_root(request_, a, b, weights, carrier, deviation);
      if (constructed != PHX_MC_OK) return constructed;
    }
    star_.clear();
    for (int32_t c : incident_[static_cast<std::size_t>(a)]) {
      const Cell& cell = cells_[static_cast<std::size_t>(c)];
      if (std::find(cell.begin(), cell.end(), b) != cell.end()) star_.push_back(c);
    }
    if (star_.empty()) return PHX_MC_INTERNAL_ERROR;
    const int64_t vertex = static_cast<int64_t>(roots_.sources.size());
    if (vertex >= std::numeric_limits<int32_t>::max() - 1 ||
        static_cast<int64_t>(cells_.size() + star_.size()) >
            std::numeric_limits<int32_t>::max()) {
      return PHX_MC_INDEX_RANGE_ERROR;
    }
    if (vertex >= request_.out_point_capacity ||
        static_cast<int64_t>(cells_.size() + star_.size()) > request_.out_cell_capacity) {
      return PHX_MC_CAPACITY_EXCEEDED;
    }
    const auto r = static_cast<int32_t>(vertex);
    roots_.sources.push_back({a, b});
    roots_.weights.push_back(weights);
    roots_.edges.push_back(e);
    if (carriers_) {
      roots_.deviations.push_back(deviation);
      roots_.points.insert(roots_.points.end(), carrier, carrier + 3);
    }
    incident_.emplace_back();
    auto& at_root = incident_.back();
    for (int32_t c : star_) {
      // Exact children: b -> S keeps c's slot order and a; a -> S keeps b.
      Cell second = cells_[static_cast<std::size_t>(c)];
      Cell& first = cells_[static_cast<std::size_t>(c)];
      *std::find(first.begin(), first.end(), b) = r;
      *std::find(second.begin(), second.end(), a) = r;
      const auto child = static_cast<int32_t>(cells_.size());
      cells_.push_back(second);
      parents_.push_back(parents_[static_cast<std::size_t>(c)]);
      *std::find(incident_[static_cast<std::size_t>(b)].begin(),
                 incident_[static_cast<std::size_t>(b)].end(), c) = child;
      for (int32_t v : second) {
        if (v != b && v != r) incident_[static_cast<std::size_t>(v)].push_back(child);
      }
      at_root.push_back(c);
      at_root.push_back(child);
    }
    return PHX_MC_OK;
  }

  const double* carrier(int32_t v) const {
    return roots_.points.data() + 3 * static_cast<std::size_t>(v);
  }

  int64_t root_edge(const Cell& cell) const {
    for (int32_t v : cell) {
      if (v >= request_.point_count) return roots_.edges[static_cast<std::size_t>(v)];
    }
    return -1;
  }

  int32_t certify_carriers() {
    if (!carriers_ || roots_.sources.size() == static_cast<std::size_t>(request_.point_count)) {
      return PHX_MC_OK;
    }
    for (const Cell& cell : cells_) {
      if (root_edge(cell) >= 0 &&
          phx::mc::orient3d(carrier(cell[0]), carrier(cell[1]), carrier(cell[2]),
                            carrier(cell[3])) <= 0) {
        *request_.failed_edge = root_edge(cell);
        return PHX_MC_DEGENERATE_INPUT;
      }
    }
    const auto count = static_cast<int64_t>(roots_.sources.size());
    std::vector<int32_t> order(static_cast<std::size_t>(count));
    for (int32_t v = 0; v < count; ++v) order[static_cast<std::size_t>(v)] = v;
    std::sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
      return std::lexicographical_compare(carrier(x), carrier(x) + 3, carrier(y), carrier(y) + 3);
    });
    for (std::size_t k = 1; k < order.size(); ++k) {
      const int32_t x = order[k - 1], y = order[k];
      if (phx::mc::geometry::same_point(carrier(x), carrier(y))) {
        *request_.failed_edge = roots_.edges[static_cast<std::size_t>(std::max(x, y))];
        return PHX_MC_DEGENERATE_INPUT;
      }
    }
    return PHX_MC_OK;
  }

  int32_t certify_sides() const {
    for (const Cell& cell : cells_) {
      bool negative = false, positive = false;
      for (int32_t v : cell) {
        if (v < request_.point_count) {
          negative = negative || request_.values[v] < 0.0;
          positive = positive || request_.values[v] > 0.0;
        }
      }
      if (negative == positive) return PHX_MC_INVALID_INPUT;
    }
    return PHX_MC_OK;
  }

  void publish() {
    const auto count = static_cast<int64_t>(roots_.sources.size());
    if (carriers_) {
      std::copy(roots_.points.begin(), roots_.points.end(), request_.out_points);
      std::copy(roots_.deviations.begin(), roots_.deviations.end(), request_.out_deviations);
    }
    for (int64_t i = 0; i < count; ++i) {
      const auto row = static_cast<std::size_t>(i);
      std::copy_n(roots_.sources[row].data(), 2, request_.out_sources + 2 * i);
      std::copy_n(roots_.weights[row].data(), 2, request_.out_weights + 2 * i);
    }
    std::vector<int64_t> offsets(static_cast<std::size_t>(request_.cell_count) + 1, 0);
    for (int32_t parent : parents_) ++offsets[static_cast<std::size_t>(parent) + 1];
    for (std::size_t p = 1; p < offsets.size(); ++p) offsets[p] += offsets[p - 1];
    for (std::size_t c = 0; c < cells_.size(); ++c) {
      const int64_t row = offsets[static_cast<std::size_t>(parents_[c])]++;
      std::copy_n(cells_[c].data(), 4, request_.out_cells + 4 * row);
      request_.out_parents[row] = parents_[c];
    }
    *request_.out_point_count = count;
    *request_.out_cell_count = static_cast<int64_t>(cells_.size());
    *request_.failed_edge = -1;
  }

  const phx_mc_level_set_split_3d_request& request_;
  const bool carriers_;
  LevelSetRoots roots_;
  std::vector<Cell> cells_;
  std::vector<int32_t> parents_;
  std::vector<std::vector<int32_t>> incident_;
  std::vector<int32_t> star_;
};
}  // namespace

extern "C" PHX_MC_API int32_t
phx_mc_level_set_split_3d(const phx_mc_level_set_split_3d_request* request) {
  if (request == nullptr) return PHX_MC_INVALID_ARGUMENT;
  return phx::mc::guarded([&]() -> int32_t {
    int32_t status = validate_level_set_request(*request);
    if (status != PHX_MC_OK) return status;
    *request->failed_edge = -1;
    std::vector<int32_t> regions(static_cast<std::size_t>(request->cell_count));
    std::vector<int32_t> face_sources(static_cast<std::size_t>(request->face_count));
    std::vector<int32_t> edge_sources(static_cast<std::size_t>(request->edge_count));
    status = prepare_level_set_sources(*request, regions, face_sources, edge_sources);
    if (status != PHX_MC_OK) return status;
    {
      // Validate the complete positive source complex and its conforming
      // faces/subsegments before publishing any subdivision.
      phx::mc::TetMesh source(phx::mc::BoundaryPolicy::kConforming);
      const auto point_span = std::span<const double>(
          request->points, static_cast<std::size_t>(3 * request->point_count));
      const auto cell_span = std::span<const int32_t>(
          request->cells, static_cast<std::size_t>(4 * request->cell_count));
      const auto region_span = std::span<const int32_t>(regions);
      const auto face_span = std::span<const int32_t>(
          request->faces, static_cast<std::size_t>(3 * request->face_count));
      const auto face_source_span = std::span<const int32_t>(face_sources);
      const auto edge_span = std::span<const int32_t>(
          request->edges, static_cast<std::size_t>(2 * request->edge_count));
      const auto edge_source_span = std::span<const int32_t>(edge_sources);
      status = source.build(point_span, cell_span, region_span, face_span,
                           face_source_span, edge_span, edge_source_span);
      if (status != PHX_MC_OK) return status;
    }
    return LevelSetSplit(*request).run();
  });
}