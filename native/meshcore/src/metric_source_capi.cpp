//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Owned C boundary of the exact metric source: carrier searches, collapse
// positions and survivor moves. Every entry establishes round-to-nearest
// with gradual underflow for its duration and validates every index before
// it writes.
#include "capi_guard.hpp"
#include "floating_point_mode.hpp"
#include "metric_carrier.hpp"
#include "metric_collapse.hpp"
#include "phydrax_meshcore.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

struct phx_mc_metric_source {
  phx::mc::metric::CarrierSource source;
  std::vector<double> metrics;
  std::vector<double> logarithms;
};

namespace {

using phx::mc::addressable;
using phx::mc::metric::Carrier;
using phx::mc::metric::CarrierSimplex;
using phx::mc::metric::CollapseStar;

// Three goals at the 53 resolutions of a carrier.
constexpr int64_t kSurvivorMoves = 159;

bool finite3(const double* values) noexcept {
  for (int axis = 0; axis < 3; ++axis) {
    const double size = values[axis] < 0.0 ? -values[axis] : values[axis];
    if (!(size <= std::numeric_limits<double>::max())) return false;
  }
  return true;
}

// Three slots per simplex; a negative slot ends its vertices.
int32_t read_simplices(int64_t simplex_count, const int64_t* simplex_rows,
                       const int8_t* declared, int64_t point_count, int least,
                       std::vector<CarrierSimplex>& simplices) {
  simplices.reserve(static_cast<std::size_t>(simplex_count));
  for (int64_t index = 0; index < simplex_count; ++index) {
    CarrierSimplex simplex;
    simplex.declared = declared[index] != 0;
    for (int slot = 0; slot < 3; ++slot) {
      const int64_t row = simplex_rows[3 * index + slot];
      if (row < 0) break;
      if (row >= point_count) return PHX_MC_INVALID_INPUT;
      simplex.rows[static_cast<std::size_t>(slot)] = row;
      simplex.size = slot + 1;
    }
    if (simplex.size < least) return PHX_MC_INVALID_INPUT;
    simplices.push_back(simplex);
  }
  return PHX_MC_OK;
}

void write_carrier(const Carrier& carrier, std::size_t index, int64_t* rows, int64_t* weights,
                   int32_t* bits, double* points) {
  for (std::size_t slot = 0; slot < 4; ++slot) {
    const bool used = static_cast<int>(slot) < carrier.size;
    rows[4 * index + slot] = used ? carrier.rows[slot] : -1;
    weights[4 * index + slot] = used ? carrier.weights[slot] : 0;
  }
  bits[index] = carrier.bits;
  for (std::size_t axis = 0; axis < 3; ++axis) points[3 * index + axis] = carrier.point[axis];
}

// Addressability and index ranges of a star with at least `least` points.
int32_t validate_star(const CollapseStar& star, int64_t least) {
  if (star.point_count < least || !addressable(star.point_count, 9, sizeof(double)) ||
      !addressable(star.spoke_count, 1, sizeof(int64_t)) ||
      !addressable(star.before_count, 1, sizeof(double)) || star.cell_count < 1 ||
      !addressable(star.cell_count, 4, sizeof(int64_t)) || star.points == nullptr ||
      star.metrics == nullptr || star.cells == nullptr ||
      (star.spoke_count > 0 && star.spokes == nullptr) ||
      (star.before_count > 0 && star.before_lengths == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t spoke = 0; spoke < star.spoke_count; ++spoke) {
    if (star.spokes[spoke] < 1 || star.spokes[spoke] >= star.point_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (int64_t entry = 0; entry < 4 * star.cell_count; ++entry) {
    if (star.cells[entry] < 0 || star.cells[entry] >= star.point_count) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  return PHX_MC_OK;
}

}  // namespace

extern "C" {

int32_t phx_mc_metric_source_create(int64_t point_count, const double* points,
                                    const uint32_t* words, const int64_t* word_offsets,
                                    const int8_t* signs, int64_t cell_count, const int64_t* cells,
                                    const double* metrics, const double* logarithms,
                                    phx_mc_metric_source** source) {
  if (source == nullptr) return PHX_MC_INVALID_ARGUMENT;
  *source = nullptr;
  if (!addressable(point_count, 9, sizeof(double)) ||
      !addressable(cell_count, 4, sizeof(int64_t)) || word_offsets == nullptr ||
      (point_count > 0 && (points == nullptr || signs == nullptr || metrics == nullptr ||
                           logarithms == nullptr)) ||
      (cell_count > 0 && cells == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  // 3 * point_count numerators and then the common denominator.
  const int64_t integers = 3 * point_count + 1;
  if (word_offsets[0] != 0) return PHX_MC_INVALID_INPUT;
  for (int64_t index = 0; index < integers; ++index) {
    if (word_offsets[index + 1] < word_offsets[index]) return PHX_MC_INVALID_INPUT;
  }
  if (word_offsets[integers] > 0 && words == nullptr) return PHX_MC_INVALID_ARGUMENT;
  for (int64_t point = 0; point < point_count; ++point) {
    if (!finite3(points + 3 * point)) return PHX_MC_NONFINITE_INPUT;
  }
  return phx::mc::guarded([&]() -> int32_t {
    const auto integer = [&](int64_t index, int sign) {
      const int64_t begin = word_offsets[index];
      return phx::mc::Dyadic::integer(words + begin,
                                      static_cast<std::size_t>(word_offsets[index + 1] - begin),
                                      sign);
    };
    std::vector<phx::mc::Dyadic> numerators;
    numerators.reserve(static_cast<std::size_t>(3 * point_count));
    for (int64_t index = 0; index < 3 * point_count; ++index) {
      numerators.push_back(integer(index, signs[index] < 0 ? -1 : 1));
    }
    phx::mc::Dyadic denominator = integer(3 * point_count, 1);
    if (denominator.sign() <= 0) return PHX_MC_INVALID_INPUT;
    for (int64_t index = 0; index < 4 * cell_count; ++index) {
      if (cells[index] < 0 || cells[index] >= point_count) return PHX_MC_INVALID_INPUT;
    }
    *source = new phx_mc_metric_source{
        phx::mc::metric::CarrierSource(
            std::vector<double>(points, points + 3 * point_count), std::move(numerators),
            std::move(denominator), std::vector<std::int64_t>(cells, cells + 4 * cell_count)),
        std::vector<double>(metrics, metrics + 9 * point_count),
        std::vector<double>(logarithms, logarithms + 9 * point_count)};
    return PHX_MC_OK;
  });
}

void phx_mc_metric_source_free(phx_mc_metric_source* source) { delete source; }

int32_t phx_mc_metric_source_carriers(const phx_mc_metric_source* source, const double* target,
                                      const double* current, int64_t simplex_count,
                                      const int64_t* simplex_rows, const int8_t* declared,
                                      int32_t order, int64_t limit, int64_t* rows,
                                      int64_t* weights, int32_t* bits, double* points,
                                      int8_t* exact, int64_t* count) {
  if (source == nullptr || target == nullptr || current == nullptr || count == nullptr ||
      !addressable(simplex_count, 3, sizeof(int64_t)) || !addressable(limit, 4, sizeof(int64_t)) ||
      (simplex_count > 0 && (simplex_rows == nullptr || declared == nullptr)) ||
      (limit > 0 && (rows == nullptr || weights == nullptr || bits == nullptr ||
                     points == nullptr || exact == nullptr)) ||
      order < 0 || order > 2) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  if (!finite3(target) || !finite3(current)) return PHX_MC_NONFINITE_INPUT;
  return phx::mc::guarded([&]() -> int32_t {
    std::vector<CarrierSimplex> simplices;
    const int32_t status = read_simplices(simplex_count, simplex_rows, declared,
                                          source->source.point_count(), 0, simplices);
    if (status != PHX_MC_OK) return status;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    const std::vector<Carrier> found = source->source.carriers(
        target, current, simplices, static_cast<phx::mc::metric::CarrierOrder>(order), limit);
    for (std::size_t index = 0; index < found.size(); ++index) {
      write_carrier(found[index], index, rows, weights, bits, points);
      exact[index] = found[index].exact ? 1 : 0;
    }
    *count = static_cast<int64_t>(found.size());
    return PHX_MC_OK;
  });
}

int32_t phx_mc_metric_collapse_positions(int64_t point_count, const double* points,
                                         const double* metrics, int64_t spoke_count,
                                         const int64_t* spokes, int64_t before_count,
                                         const double* before_lengths, int64_t cell_count,
                                         const int64_t* cells, double determinant_floor,
                                         int64_t position_count, const double* positions,
                                         const double* position_metrics, int32_t* verdicts,
                                         double* qualities, double* lengths) {
  CollapseStar star;
  star.point_count = point_count;
  star.points = points;
  star.metrics = metrics;
  star.spoke_count = spoke_count;
  star.spokes = spokes;
  star.before_count = before_count;
  star.before_lengths = before_lengths;
  star.cell_count = cell_count;
  star.cells = cells;
  star.determinant_floor = determinant_floor;
  const int32_t status = validate_star(star, 1);
  if (status != PHX_MC_OK) return status;
  if (!addressable(position_count, 9, sizeof(double)) ||
      (spoke_count > 0 && !addressable(position_count, spoke_count, sizeof(double))) ||
      (position_count > 0 && (positions == nullptr || position_metrics == nullptr ||
                              verdicts == nullptr || qualities == nullptr ||
                              (spoke_count > 0 && lengths == nullptr)))) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return phx::mc::guarded([&]() -> int32_t {
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    phx::mc::metric::CollapseObjective objective(star);
    std::vector<double> measured(static_cast<std::size_t>(spoke_count));
    for (int64_t position = 0; position < position_count; ++position) {
      for (double& value : measured) value = 0.0;
      double quality = 0.0;
      verdicts[position] = static_cast<int32_t>(objective.evaluate(
          positions + 3 * position, position_metrics + 9 * position, measured.data(), quality));
      qualities[position] = quality;
      for (int64_t spoke = 0; spoke < spoke_count; ++spoke) {
        lengths[position * spoke_count + spoke] = measured[static_cast<std::size_t>(spoke)];
      }
    }
    return PHX_MC_OK;
  });
}

int32_t phx_mc_metric_source_survivor_moves(
    const phx_mc_metric_source* source, int64_t point_count, const double* points,
    const double* metrics, int64_t spoke_count, const int64_t* spokes, int64_t before_count,
    const double* before_lengths, int64_t cell_count, const int64_t* cells,
    double determinant_floor, int64_t simplex_count, const int64_t* simplex_rows,
    const int8_t* declared, int32_t every_resolution, int64_t capacity, int32_t* goals,
    int64_t* rows, int64_t* weights, int32_t* bits, double* carrier_points,
    double* carrier_metrics, double* qualities, double* lengths, int64_t* count, int32_t* end) {
  CollapseStar star;
  star.point_count = point_count;
  star.points = points;
  star.metrics = metrics;
  star.spoke_count = spoke_count;
  star.spokes = spokes;
  star.before_count = before_count;
  star.before_lengths = before_lengths;
  star.cell_count = cell_count;
  star.cells = cells;
  star.determinant_floor = determinant_floor;
  if (source == nullptr || count == nullptr || end == nullptr) return PHX_MC_INVALID_ARGUMENT;
  const int32_t status = validate_star(star, 2);
  if (status != PHX_MC_OK) return status;
  if (!addressable(simplex_count, 3, sizeof(int64_t)) ||
      (simplex_count > 0 && (simplex_rows == nullptr || declared == nullptr)) ||
      capacity < kSurvivorMoves || !addressable(capacity, 9, sizeof(double)) ||
      (spoke_count > 0 && !addressable(capacity, spoke_count, sizeof(double))) ||
      goals == nullptr || rows == nullptr || weights == nullptr || bits == nullptr ||
      carrier_points == nullptr || carrier_metrics == nullptr || qualities == nullptr ||
      (spoke_count > 0 && lengths == nullptr)) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  return phx::mc::guarded([&]() -> int32_t {
    std::vector<CarrierSimplex> simplices;
    const int32_t parsed = read_simplices(simplex_count, simplex_rows, declared,
                                          source->source.point_count(), 2, simplices);
    if (parsed != PHX_MC_OK) return parsed;
    const phx::mc::GradualNearestScope mode;
    if (!mode.established()) return PHX_MC_INTERNAL_ERROR;
    std::vector<phx::mc::metric::SurvivorMove> moves;
    std::vector<double> measured;
    const phx::mc::metric::SearchEnd ended = phx::mc::metric::survivor_search(
        source->source, source->metrics.data(), source->logarithms.data(), star, simplices,
        every_resolution != 0, moves, measured);
    if (static_cast<int64_t>(moves.size()) > capacity) return PHX_MC_CAPACITY_EXCEEDED;
    for (std::size_t index = 0; index < moves.size(); ++index) {
      const phx::mc::metric::SurvivorMove& move = moves[index];
      goals[index] = move.goal;
      write_carrier(move.carrier, index, rows, weights, bits, carrier_points);
      for (std::size_t entry = 0; entry < 9; ++entry) {
        carrier_metrics[9 * index + entry] = move.metric[entry];
      }
      qualities[index] = move.quality;
    }
    for (std::size_t index = 0; index < measured.size(); ++index) lengths[index] = measured[index];
    *count = static_cast<int64_t>(moves.size());
    *end = static_cast<int32_t>(ended);
    return PHX_MC_OK;
  });
}

}  // extern "C"
