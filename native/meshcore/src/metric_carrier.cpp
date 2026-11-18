//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact carriers of a floating-point goal in a tetrahedral source; see
// metric_carrier.hpp. Barycentric weights are kept as integer numerators over
// one denominator (Cramer's rule), so no exact division is needed: the only
// quotients are the correctly rounded ones of exact_scalar.hpp.
#include "metric_carrier.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "exact_scalar.hpp"

namespace phx::mc::metric {
namespace {

using Vector = std::array<Dyadic, 3>;

Dyadic dot(const Vector& a, const Vector& b) { return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; }

// Determinant of the matrix with rows a, b, c.
Dyadic determinant(const Vector& a, const Vector& b, const Vector& c) {
  return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

int compare(const Dyadic& a, const Dyadic& b) { return (a - b).sign(); }

// The integer nearest numerator * 2^bits / denominator, ties to even, for
// 0 <= numerator <= denominator and bits <= 53: the quotient is at most 2^53.
// The correctly rounded binary64 quotient is within one of its floor; the
// exact remainder settles the floor and then the rounding.
std::int64_t rounded_partial(const Dyadic& numerator, const Dyadic& denominator, int bits) {
  const Dyadic scaled = numerator.scaled_power(bits);
  std::int64_t quotient =
      static_cast<std::int64_t>(std::floor(exact::rounded_quotient(scaled, denominator)));
  Dyadic remainder = scaled - Dyadic(static_cast<double>(quotient)) * denominator;
  while (remainder.sign() < 0) {
    quotient -= 1;
    remainder = remainder + denominator;
  }
  while (compare(remainder, denominator) >= 0) {
    quotient += 1;
    remainder = remainder - denominator;
  }
  const int half = compare(remainder + remainder, denominator);
  if (half > 0 || (half == 0 && (quotient & 1) != 0)) quotient += 1;
  return quotient;
}

}  // namespace

CarrierSource::CarrierSource(std::vector<double> points, std::vector<Dyadic> numerators,
                             Dyadic denominator, std::vector<std::int64_t> cells)
    : points_(std::move(points)),
      numerators_(std::move(numerators)),
      denominator_(std::move(denominator)),
      cells_(std::move(cells)) {
  if (points_.size() % 3 != 0 || numerators_.size() != points_.size() ||
      cells_.size() % 4 != 0 || denominator_.sign() <= 0) {
    throw std::invalid_argument("An exact carrier source needs aligned points and cells.");
  }
  const std::int64_t count = point_count();
  const std::size_t cell_count = cells_.size() / 4;
  lower_.resize(3 * cell_count);
  upper_.resize(3 * cell_count);
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for (std::size_t corner = 0; corner < 4; ++corner) {
      const std::int64_t row = cells_[4 * cell + corner];
      if (row < 0 || row >= count) {
        throw std::invalid_argument("An exact carrier source cell names an absent point.");
      }
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const double value = points_[3 * static_cast<std::size_t>(row) + axis];
        if (corner == 0 || value < lower_[3 * cell + axis]) lower_[3 * cell + axis] = value;
        if (corner == 0 || value > upper_[3 * cell + axis]) upper_[3 * cell + axis] = value;
      }
    }
  }
}

bool CarrierSource::locate_cell(const double* target, std::int64_t only, Located& located) const {
  const std::size_t begin = only < 0 ? 0 : static_cast<std::size_t>(only);
  const std::size_t end = only < 0 ? cells_.size() / 4 : begin + 1;
  bool scaled = false;
  Vector exact_target;
  for (std::size_t cell = begin; cell < end; ++cell) {
    // Rounding is monotone and the box corners are binary64 source points, so
    // a cell that holds the exact target passes this comparison.
    bool inside = true;
    for (std::size_t axis = 0; axis < 3 && inside; ++axis) {
      inside = lower_[3 * cell + axis] <= target[axis] && target[axis] <= upper_[3 * cell + axis];
    }
    if (!inside) continue;
    if (!scaled) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        exact_target[axis] = Dyadic(target[axis]) * denominator_;
      }
      scaled = true;
    }
    const std::int64_t* rows = cells_.data() + 4 * cell;
    Vector edges[3];
    Vector offset;
    for (int axis = 0; axis < 3; ++axis) {
      const Dyadic& origin = numerator(rows[0], axis);
      for (int corner = 0; corner < 3; ++corner) {
        edges[corner][static_cast<std::size_t>(axis)] = numerator(rows[corner + 1], axis) - origin;
      }
      offset[static_cast<std::size_t>(axis)] = exact_target[static_cast<std::size_t>(axis)] - origin;
    }
    Dyadic volume = determinant(edges[0], edges[1], edges[2]);
    if (volume.sign() == 0) continue;
    Dyadic weights[4];
    weights[1] = determinant(offset, edges[1], edges[2]);
    weights[2] = determinant(edges[0], offset, edges[2]);
    weights[3] = determinant(edges[0], edges[1], offset);
    if (volume.sign() < 0) {
      volume.negate();
      for (int corner = 1; corner < 4; ++corner) weights[corner].negate();
    }
    weights[0] = ((volume - weights[1]) - weights[2]) - weights[3];
    bool holds = true;
    for (const Dyadic& weight : weights) holds = holds && weight.sign() >= 0;
    if (!holds) continue;
    located.size = 4;
    located.declared = false;
    located.denominator = std::move(volume);
    for (std::size_t corner = 0; corner < 4; ++corner) {
      located.rows[corner] = rows[corner];
      located.weights[corner] = std::move(weights[corner]);
    }
    return true;
  }
  return false;
}

bool CarrierSource::locate_simplex(const double* target, const CarrierSimplex& simplex,
                                   Located& located) const {
  Vector edges[2];
  Vector offset;
  for (int axis = 0; axis < 3; ++axis) {
    const std::size_t slot = static_cast<std::size_t>(axis);
    const Dyadic& origin = numerator(simplex.rows[0], axis);
    for (int corner = 1; corner < simplex.size; ++corner) {
      edges[corner - 1][slot] = numerator(simplex.rows[static_cast<std::size_t>(corner)], axis) - origin;
    }
    offset[slot] = Dyadic(target[axis]) * denominator_ - origin;
  }
  Dyadic weights[3];
  Dyadic total;
  if (simplex.size == 2) {
    total = dot(edges[0], edges[0]);
    if (total.sign() == 0) return false;
    weights[1] = dot(edges[0], offset);
    if (weights[1].sign() < 0 || compare(weights[1], total) > 0) return false;
    weights[0] = total - weights[1];
  } else {
    const Dyadic g00 = dot(edges[0], edges[0]);
    const Dyadic g01 = dot(edges[0], edges[1]);
    const Dyadic g11 = dot(edges[1], edges[1]);
    total = g00 * g11 - g01 * g01;
    if (total.sign() == 0) return false;
    const Dyadic r0 = dot(edges[0], offset);
    const Dyadic r1 = dot(edges[1], offset);
    weights[1] = r0 * g11 - r1 * g01;
    if (weights[1].sign() < 0 || compare(weights[1], total) > 0) return false;
    const Dyadic remaining = total - weights[1];
    weights[2] = r1 * g00 - r0 * g01;
    if (weights[2].sign() < 0 || compare(weights[2], remaining) > 0) return false;
    weights[0] = remaining - weights[2];
  }
  located.size = simplex.size;
  located.declared = simplex.declared;
  located.denominator = std::move(total);
  for (int corner = 0; corner < simplex.size; ++corner) {
    const std::size_t slot = static_cast<std::size_t>(corner);
    located.rows[slot] = simplex.rows[slot];
    located.weights[slot] = std::move(weights[corner]);
  }
  return true;
}

std::vector<Carrier> CarrierSource::carriers(const double* target, const double* current,
                                             const std::vector<CarrierSimplex>& simplices,
                                             std::int64_t cell, CarrierOrder order,
                                             std::int64_t limit) const {
  std::vector<Carrier> found;
  if (limit <= 0) return found;
  Located located;
  const bool on_simplex = !simplices.empty();
  bool has_location = false;
  if (on_simplex) {
    for (const CarrierSimplex& simplex : simplices) {
      if (simplex.size != 2 && simplex.size != 3) continue;
      if (locate_simplex(target, simplex, located)) {
        has_location = true;
        break;
      }
    }
  } else {
    has_location = locate_cell(target, cell, located);
  }
  if (!has_location) return found;

  // The distinct support points in located order, and the support slot of
  // every located vertex (a vertex named twice shares its slot).
  std::array<std::int64_t, 4> support{};
  std::array<int, 4> column{};
  int support_size = 0;
  for (int vertex = 0; vertex < located.size; ++vertex) {
    int slot = 0;
    while (slot < support_size && support[static_cast<std::size_t>(slot)] !=
                                      located.rows[static_cast<std::size_t>(vertex)]) {
      ++slot;
    }
    if (slot == support_size) {
      support[static_cast<std::size_t>(support_size)] = located.rows[static_cast<std::size_t>(vertex)];
      ++support_size;
    }
    column[static_cast<std::size_t>(vertex)] = slot;
  }

  // The goal: the target in a cell, the correctly rounded projection on a simplex.
  double goal[3] = {target[0], target[1], target[2]};
  if (on_simplex) {
    const Dyadic image_denominator = located.denominator * denominator_;
    for (int axis = 0; axis < 3; ++axis) {
      Dyadic sum;
      for (int vertex = 0; vertex < located.size; ++vertex) {
        const std::size_t slot = static_cast<std::size_t>(vertex);
        sum = sum + located.weights[slot] * numerator(located.rows[slot], axis);
      }
      goal[axis] = exact::rounded_quotient(sum, image_denominator);
    }
  }
  const double moved[3] = {goal[0] - current[0], goal[1] - current[1], goal[2] - current[2]};
  const double distance = (moved[0] * moved[0] + moved[1] * moved[1]) + moved[2] * moved[2];
  double span = 0.0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    double low = 0.0;
    double high = 0.0;
    for (int slot = 0; slot < support_size; ++slot) {
      const double value =
          points_[3 * static_cast<std::size_t>(support[static_cast<std::size_t>(slot)]) + axis];
      if (slot == 0 || value < low) low = value;
      if (slot == 0 || value > high) high = value;
    }
    if (high - low > span) span = high - low;
  }
  if (!(distance > 0.0) || !(span > 0.0)) return found;

  // A grid coarser than the move cannot resolve it: start at the binary
  // exponent of span / |move|, which is floor(log2) exactly.
  const double ratio = span / std::sqrt(distance);
  int exponent = 0;
  if (ratio <= std::numeric_limits<double>::max()) {
    std::frexp(ratio, &exponent);
  }
  const int first = exponent - 1 < 1 ? 1 : (exponent - 1 > 53 ? 53 : exponent - 1);

  // Partial sums of the weight numerators in located order, all but the last.
  std::array<Dyadic, 3> partial;
  for (int vertex = 0; vertex + 1 < located.size; ++vertex) {
    const std::size_t slot = static_cast<std::size_t>(vertex);
    partial[slot] = vertex == 0 ? located.weights[0] : partial[slot - 1] + located.weights[slot];
  }

  const auto try_resolution = [&](int bits) {
    const std::int64_t scale = std::int64_t{1} << bits;
    std::array<std::int64_t, 4> merged{};
    std::int64_t previous = 0;
    for (int vertex = 0; vertex + 1 < located.size; ++vertex) {
      const std::size_t slot = static_cast<std::size_t>(vertex);
      std::int64_t quotient = rounded_partial(partial[slot], located.denominator, bits);
      if (quotient < previous) quotient = previous;
      if (quotient > scale) quotient = scale;
      merged[static_cast<std::size_t>(column[slot])] += quotient - previous;
      previous = quotient;
    }
    merged[static_cast<std::size_t>(column[static_cast<std::size_t>(located.size - 1)])] +=
        scale - previous;
    Carrier carrier;
    carrier.bits = bits;
    carrier.exact = true;
    const Dyadic image_denominator = denominator_.scaled_power(bits);
    for (int axis = 0; axis < 3; ++axis) {
      Dyadic value;
      for (int slot = 0; slot < support_size; ++slot) {
        const std::size_t index = static_cast<std::size_t>(slot);
        if (merged[index] == 0) continue;
        value = value + Dyadic(static_cast<double>(merged[index])) * numerator(support[index], axis);
      }
      const double rounded = exact::rounded_quotient(value, image_denominator);
      carrier.point[static_cast<std::size_t>(axis)] = rounded;
      if ((value - Dyadic(rounded) * image_denominator).sign() != 0) carrier.exact = false;
    }
    const double dx = carrier.point[0] - goal[0];
    const double dy = carrier.point[1] - goal[1];
    const double dz = carrier.point[2] - goal[2];
    if (!((dx * dx + dy * dy) + dz * dz < distance)) return false;
    if (on_simplex && !carrier.exact && !located.declared) return false;
    // The support in ascending row order: one stencil has one spelling.
    for (int slot = 0; slot < support_size; ++slot) {
      const std::size_t index = static_cast<std::size_t>(slot);
      if (merged[index] == 0) continue;
      std::size_t place = static_cast<std::size_t>(carrier.size);
      while (place > 0 && carrier.rows[place - 1] > support[index]) {
        carrier.rows[place] = carrier.rows[place - 1];
        carrier.weights[place] = carrier.weights[place - 1];
        --place;
      }
      carrier.rows[place] = support[index];
      carrier.weights[place] = merged[index];
      ++carrier.size;
    }
    found.push_back(carrier);
    return true;
  };

  const auto full = [&]() { return static_cast<std::int64_t>(found.size()) >= limit; };
  if (order == CarrierOrder::kFineFirst) {
    for (int bits = 53; bits >= first && !full(); --bits) try_resolution(bits);
    return found;
  }
  for (int bits = first; bits <= 53 && !full(); ++bits) {
    if (try_resolution(bits) && order == CarrierOrder::kCoarseThenFine) {
      for (int fine = 53; fine >= first && !full(); --fine) try_resolution(fine);
      return found;
    }
  }
  return found;
}

}  // namespace phx::mc::metric
