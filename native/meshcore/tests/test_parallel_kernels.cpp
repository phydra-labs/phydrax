// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Batch-level equivalence: scalar predicate tests own the numerical oracle.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {
constexpr std::size_t kCount = 193;  // Three full chunks and one partial chunk.
constexpr std::array<uint32_t, 4> kThreads{1, 2, 7, 64};
// Validation polls the scope once per fixed 64-row chunk; the large batch
// spans more than 1024 chunks plus a partial one.
constexpr std::size_t kChunkRows = 64;
constexpr std::size_t kLargeCount = 1100 * kChunkRows + 17;
// A deadline that a scope meets on entry and misses after the sleep.
constexpr double kDeadlineSeconds = 0.05;
constexpr auto kPastDeadline = std::chrono::milliseconds(80);

struct Receipt {
  std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS> counters{};
  std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES> memory{};
  bool operator==(const Receipt&) const = default;
};

class Execution final {
 public:
  explicit Execution(uint64_t bytes = UINT64_MAX, double seconds = INFINITY) {
    PHX_CHECK(phx_mc_execution_begin(0, 0, 0, bytes, seconds, nullptr, &scope_) == PHX_MC_OK);
  }
  Receipt finish(int32_t status = PHX_MC_OK) {
    Receipt receipt;
    double seconds = 0;
    PHX_CHECK(phx_mc_execution_end(scope_, receipt.counters.data(), receipt.memory.data(),
                                   &seconds) == status);
    scope_ = nullptr;
    return receipt;
  }
  std::array<uint64_t, 4> remaining(int32_t status = PHX_MC_OK) const {
    std::array<uint64_t, 4> values{};
    double seconds = 0;
    PHX_CHECK(phx_mc_execution_remaining(scope_, values.data(), &seconds) == status);
    return values;
  }
  void* scope() const { return scope_; }
  ~Execution() {
    if (scope_ != nullptr) finish();
  }

 private:
  void* scope_ = nullptr;
};

class Policy final {
 public:
  explicit Policy(uint32_t threads) {
    PHX_CHECK(phx_mc_parallel_begin(static_cast<int32_t>(threads), INFINITY, &policy_) ==
              PHX_MC_OK);
  }
  ~Policy() { PHX_CHECK(phx_mc_parallel_end(policy_) == PHX_MC_OK); }
  Policy(const Policy&) = delete;
  Policy& operator=(const Policy&) = delete;

 private:
  void* policy_ = nullptr;
};

enum class Predicate {
  Orient2d,
  Orient3d,
  Incircle,
  Insphere,
  Orient2dSos,
  Orient3dSos,
  IncircleSos,
  InsphereSos
};

int32_t batch(Predicate kind, const std::array<std::vector<double>, 5>& p,
              const std::vector<int64_t>& ids, int8_t* output, std::size_t count = kCount) {
  const auto n = static_cast<int64_t>(count);
  switch (kind) {
    case Predicate::Orient2d:
      return phx_mc_orient2d(n, p[0].data(), p[1].data(), p[2].data(), output);
    case Predicate::Orient3d:
      return phx_mc_orient3d(n, p[0].data(), p[1].data(), p[2].data(), p[3].data(), output);
    case Predicate::Incircle:
      return phx_mc_incircle(n, p[0].data(), p[1].data(), p[2].data(), p[3].data(), output);
    case Predicate::Insphere:
      return phx_mc_insphere(n, p[0].data(), p[1].data(), p[2].data(), p[3].data(), p[4].data(),
                             output);
    case Predicate::Orient2dSos:
      return phx_mc_orient2d_sos(n, p[0].data(), p[1].data(), p[2].data(), ids.data(), output);
    case Predicate::Orient3dSos:
      return phx_mc_orient3d_sos(n, p[0].data(), p[1].data(), p[2].data(), p[3].data(), ids.data(),
                                 output);
    case Predicate::IncircleSos:
      return phx_mc_incircle_sos(n, p[0].data(), p[1].data(), p[2].data(), p[3].data(), ids.data(),
                                 output);
    case Predicate::InsphereSos:
      return phx_mc_insphere_sos(n, p[0].data(), p[1].data(), p[2].data(), p[3].data(), p[4].data(),
                                 ids.data(), output);
  }
  return PHX_MC_INTERNAL_ERROR;
}

int scalar(Predicate kind, const std::array<std::vector<double>, 5>& p, const int64_t* ids,
           std::size_t row, int width) {
  using namespace phx::mc;
  const double* a = p[0].data() + width * row;
  const double* b = p[1].data() + width * row;
  const double* c = p[2].data() + width * row;
  const double* d = p[3].data() + width * row;
  const double* e = p[4].data() + width * row;
  switch (kind) {
    case Predicate::Orient2d:
      return orient2d(a, b, c);
    case Predicate::Orient3d:
      return orient3d(a, b, c, d);
    case Predicate::Incircle:
      return incircle(a, b, c, d);
    case Predicate::Insphere:
      return insphere(a, b, c, d, e);
    case Predicate::Orient2dSos:
      return orient2d_sos(a, b, c, ids[0], ids[1], ids[2]);
    case Predicate::Orient3dSos:
      return orient3d_sos(a, b, c, d, ids[0], ids[1], ids[2], ids[3]);
    case Predicate::IncircleSos:
      return incircle_sos(a, b, c, d, ids[0], ids[1], ids[2], ids[3]);
    case Predicate::InsphereSos:
      return insphere_sos(a, b, c, d, e, ids[0], ids[1], ids[2], ids[3], ids[4]);
  }
  return 0;
}

void predicate_equivalence(Predicate kind, int width, int arity) {
  std::array<std::vector<double>, 5> p;
  std::vector<int64_t> ids(kCount * arity);
  for (std::size_t point = 0; point < p.size(); ++point) {
    p[point].resize(kCount * width);
    for (std::size_t row = 0; row < kCount; ++row) {
      const double scale = std::ldexp(1.0, static_cast<int>(row % 3) * 100 - 100);
      for (int axis = 0; axis < width; ++axis) {
        const int value = static_cast<int>((row * 13 + point * 7 + axis * 11) % 31) - 15;
        p[point][width * row + axis] = row % 4 == 0 ? 0.0 : value * scale;
      }
    }
  }
  for (std::size_t row = 0; row < kCount; ++row) {
    for (int point = 0; point < arity; ++point) ids[arity * row + point] = arity - point;
  }
  std::vector<int8_t> expected(kCount);
  Receipt scalar_receipt;
  {
    Execution execution;
    for (std::size_t row = 0; row < kCount; ++row) {
      expected[row] = static_cast<int8_t>(scalar(kind, p, ids.data() + arity * row, row, width));
    }
    scalar_receipt = execution.finish();
  }
  Receipt reference;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    Execution execution;
    const auto before = execution.remaining();
    std::vector<int8_t> output(kCount, 9);
    PHX_CHECK(batch(kind, p, ids, output.data()) == PHX_MC_OK);
    PHX_CHECK(output == expected);
    PHX_CHECK(execution.remaining() == before);  // Unused reservation is released.
    const Receipt receipt = execution.finish();
    PHX_CHECK(receipt.counters == scalar_receipt.counters);
    PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_LIVE_BYTES] == 0);
    if (threads == 1)
      reference = receipt;
    else
      PHX_CHECK(receipt == reference);
  }
}

void location_equivalence() {
  std::vector<double> points(3 * kCount), triangles(9 * kCount);
  std::vector<int8_t> expected_sides(kCount), expected_features(kCount);
  std::vector<int32_t> expected_status(kCount);
  for (std::size_t row = 0; row < kCount; ++row) {
    const double t[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    std::copy(t, t + 9, triangles.data() + 9 * row);
    points[3 * row] = 0.25;
    points[3 * row + 1] = 0.25;
    expected_features[row] = 6;  // Interior.
    switch (row % 6) {
      case 0:
        break;
      case 1:
        points[3 * row + 2] = 1;
        expected_sides[row] = 1;
        expected_features[row] = -1;
        break;
      case 2:
        points[3 * row] = points[3 * row + 1] = 0;
        expected_features[row] = 0;
        break;
      case 3:
        triangles[9 * row + 3] = triangles[9 * row + 7] = 0;
        expected_status[row] = PHX_MC_DEGENERATE_INPUT;
        expected_features[row] = -1;
        break;
      case 4:
        points[3 * row] = NAN;
        expected_status[row] = PHX_MC_NONFINITE_INPUT;
        expected_features[row] = -1;
        break;
      case 5:
        points[3 * row] = 0x1p121;
        expected_status[row] = PHX_MC_RANGE_ERROR;
        expected_features[row] = -1;
        break;
    }
  }
  Receipt reference;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    Execution execution;
    std::vector<int8_t> sides(kCount, 9), features(kCount, 9);
    std::vector<int32_t> status(kCount, -1);
    PHX_CHECK(phx_mc_point_triangle_locations(kCount, points.data(), triangles.data(), sides.data(),
                                              features.data(), status.data()) == PHX_MC_OK);
    PHX_CHECK(sides == expected_sides && features == expected_features &&
              status == expected_status);
    const Receipt receipt = execution.finish();
    if (threads == 1)
      reference = receipt;
    else
      PHX_CHECK(receipt == reference);
  }
}

void restricted_bound_equivalence() {
  const double points[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const double domain[6] = {-1, -1, -1, 2, 2, 2};
  std::vector<int32_t> tets(4 * kCount), facets(3 * kCount), cells(kCount);
  for (std::size_t row = 0; row < kCount; ++row) {
    for (int j = 0; j < 4; ++j) tets[4 * row + j] = j;
    for (int j = 0; j < 3; ++j) facets[3 * row + j] = j;
    cells[row] = static_cast<int32_t>(row);
  }
  std::vector<double> reference_centers, reference_rays;
  std::vector<int32_t> reference_kinds, reference_status;
  Receipt center_receipt, ray_receipt;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    std::vector<double> centers(6 * kCount), rays(12 * kCount);
    std::vector<int32_t> status(kCount, -1), kinds(kCount, -1);
    {
      Execution execution;
      PHX_CHECK(phx_mc_restricted_centers_3d(4, points, kCount, tets.data(), centers.data(),
                                             status.data()) == PHX_MC_OK);
      PHX_CHECK(
          std::all_of(status.begin(), status.end(), [](int32_t s) { return s == PHX_MC_OK; }));
      for (double value : centers) PHX_CHECK(value > 0.49 && value < 0.51);
      const Receipt receipt = execution.finish();
      if (threads == 1) {
        reference_centers = centers;
        center_receipt = receipt;
      } else {
        PHX_CHECK(std::memcmp(centers.data(), reference_centers.data(),
                              centers.size() * sizeof(double)) == 0);
        PHX_CHECK(receipt == center_receipt);
      }
    }
    // Scientific failures in all four chunks must not cancel sibling rows.
    for (std::size_t row = 1; row < kCount; row += 5) {
      std::fill_n(centers.data() + 6 * row, 3, -INFINITY);
      std::fill_n(centers.data() + 6 * row + 3, 3, INFINITY);
    }
    {
      Execution execution;
      PHX_CHECK(phx_mc_restricted_rays_3d(4, points, kCount, tets.data(), centers.data(), kCount,
                                          facets.data(), cells.data(), domain, rays.data(),
                                          kinds.data(), status.data()) == PHX_MC_OK);
      for (std::size_t row = 0; row < kCount; ++row) {
        PHX_CHECK(status[row] == (row % 5 == 1 ? PHX_MC_DEGENERATE_INPUT : PHX_MC_OK));
        if (row % 5 == 1) {
          PHX_CHECK(rays[12 * row] == -INFINITY && rays[12 * row + 3] == INFINITY);
        }
      }
      const Receipt receipt = execution.finish();
      if (threads == 1) {
        reference_rays = rays;
        reference_kinds = kinds;
        reference_status = status;
        ray_receipt = receipt;
      } else {
        PHX_CHECK(std::memcmp(rays.data(), reference_rays.data(), rays.size() * sizeof(double)) ==
                  0);
        PHX_CHECK(kinds == reference_kinds && status == reference_status && receipt == ray_receipt);
      }
    }
  }
}

void resource_refusal_is_atomic() {
  std::vector<double> a(2 * kCount), b(2 * kCount), c(2 * kCount);
  for (std::size_t row = 0; row < kCount; ++row) {
    b[2 * row] = 1;
    c[2 * row + 1] = 1;
  }
  Receipt reference;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    Execution execution(0);
    const auto before = execution.remaining();
    std::vector<int8_t> output(kCount, 9);
    PHX_CHECK(phx_mc_orient2d(kCount, a.data(), b.data(), c.data(), output.data()) ==
              PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(std::all_of(output.begin(), output.end(), [](int8_t s) { return s == 9; }));
    PHX_CHECK(execution.remaining(PHX_MC_CAPACITY_EXCEEDED) == before);
    const Receipt receipt = execution.finish(PHX_MC_CAPACITY_EXCEEDED);
    PHX_CHECK(receipt.counters[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == 0);
    if (threads == 1)
      reference = receipt;
    else
      PHX_CHECK(receipt == reference);
  }
}

void validation_precedes_reservation() {
  std::vector<double> a(2 * kCount), b(2 * kCount), c(2 * kCount);
  std::vector<int64_t> ids(3 * kCount, 0);  // Invalid IDs cannot mask coordinate errors.
  a.back() = NAN;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    Execution execution(0);
    std::vector<int8_t> output(kCount, 9);
    PHX_CHECK(phx_mc_orient2d_sos(kCount, a.data(), b.data(), c.data(), ids.data(),
                                  output.data()) == PHX_MC_NONFINITE_INPUT);
    PHX_CHECK(std::all_of(output.begin(), output.end(), [](int8_t s) { return s == 9; }));
    const Receipt receipt = execution.finish();
    PHX_CHECK((receipt.counters == std::array<uint64_t, PHX_MC_EXECUTION_COUNTERS>{}));
    PHX_CHECK((receipt.memory == std::array<uint64_t, PHX_MC_EXECUTION_MEMORY_VALUES>{}));
  }
}

void batch_borrows_host_ticket_once() {
  std::vector<double> a(2 * kCount), b(2 * kCount), c(2 * kCount);
  for (std::size_t row = 0; row < kCount; ++row) {
    b[2 * row] = 1;
    c[2 * row + 1] = 1;
  }
  Receipt reference;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    Execution execution;
    const auto before = execution.remaining();
    void* ticket = nullptr;
    PHX_CHECK(phx_mc_execution_batch_admit(execution.scope(), nullptr, 0, 0, 0, 1 << 20, &ticket) ==
              PHX_MC_OK);
    void* target = nullptr;
    phx_mc_execution_usage* usage = nullptr;
    PHX_CHECK(phx_mc_execution_target_begin(ticket, &target, &usage) == PHX_MC_OK);
    std::vector<int8_t> output(kCount, 9);
    PHX_CHECK(phx_mc_orient2d(kCount, a.data(), b.data(), c.data(), output.data()) == PHX_MC_OK);
    PHX_CHECK(std::all_of(output.begin(), output.end(), [](int8_t s) { return s == 1; }));
    PHX_CHECK(phx_mc_execution_target_end(target) == PHX_MC_OK);
    PHX_CHECK(phx_mc_execution_batch_settle(ticket, 0, 0, 0, 0) == PHX_MC_OK);
    PHX_CHECK(execution.remaining() == before);
    const Receipt receipt = execution.finish();
    PHX_CHECK(receipt.counters[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == kCount);
    PHX_CHECK(receipt.counters[PHX_MC_EXECUTION_CONSTRUCTION_WORK] == 0 &&
              receipt.counters[PHX_MC_EXECUTION_EXTERNAL_WORK] == 0 &&
              receipt.counters[PHX_MC_EXECUTION_EXTERNAL_QUERIES] == 0);
    if (threads == 1)
      reference = receipt;
    else
      PHX_CHECK(receipt == reference);
  }
}

void finite_scratch_batch_is_admitted() {
  // 32 MiB encloses four fixed chunks' single-row cumulative insphere arenas
  // (under 5 MiB each, including allocation/owner metadata), but not 193
  // separate row arenas. No expansion or scratch pointer escapes a row.
  constexpr uint64_t scratch = 32 * 1024 * 1024;
  const double vertices[5][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
  std::array<std::vector<double>, 5> p;
  for (std::size_t point = 0; point < p.size(); ++point) {
    p[point].resize(3 * kCount);
    for (std::size_t row = 0; row < kCount; ++row) {
      std::copy_n(vertices[point], 3, p[point].data() + 3 * row);
    }
  }
  Receipt reference;
  for (uint32_t threads : kThreads) {
    Policy policy(threads);
    Execution execution(scratch);
    const auto before = execution.remaining();
    std::vector<int8_t> output(kCount, 9);
    PHX_CHECK(phx_mc_insphere(kCount, p[0].data(), p[1].data(), p[2].data(), p[3].data(),
                              p[4].data(), output.data()) == PHX_MC_OK);
    PHX_CHECK(std::all_of(output.begin(), output.end(), [](int8_t s) { return s == 0; }));
    PHX_CHECK(execution.remaining() == before);
    const Receipt receipt = execution.finish();
    PHX_CHECK(receipt.counters[PHX_MC_EXECUTION_PRIMITIVE_QUERIES] == kCount);
    PHX_CHECK(receipt.memory[PHX_MC_EXECUTION_MEMORY_PEAK_BYTES] <= scratch);
    if (threads == 1)
      reference = receipt;
    else
      PHX_CHECK(receipt == reference);
  }
}

struct Kind {
  Predicate predicate;
  int width;
  int arity;
  bool symbolic;
};
constexpr std::array<Kind, 8> kKinds{{{Predicate::Orient2d, 2, 3, false},
                                      {Predicate::Orient3d, 3, 4, false},
                                      {Predicate::Incircle, 2, 4, false},
                                      {Predicate::Insphere, 3, 5, false},
                                      {Predicate::Orient2dSos, 2, 3, true},
                                      {Predicate::Orient3dSos, 3, 4, true},
                                      {Predicate::IncircleSos, 2, 4, true},
                                      {Predicate::InsphereSos, 3, 5, true}}};

// Finite in-domain coordinates (small integers, zero included), distinct row IDs.
struct BatchInput {
  BatchInput(const Kind& kind, std::size_t count) : ids(count * kind.arity) {
    for (int point = 0; point < kind.arity; ++point) {
      p[point].resize(count * kind.width);
      for (std::size_t index = 0; index < p[point].size(); ++index) {
        p[point][index] = static_cast<double>(static_cast<int>((index * 7 + point * 3) % 17) - 8);
      }
    }
    for (std::size_t row = 0; row < count; ++row) {
      for (int point = 0; point < kind.arity; ++point) ids[kind.arity * row + point] = point;
    }
  }
  std::array<std::vector<double>, 5> p;
  std::vector<int64_t> ids;
};

bool untouched(const std::vector<int8_t>& output) {
  return std::all_of(output.begin(), output.end(), [](int8_t s) { return s == 9; });
}

// No work, query or memory trace beyond the scope's own limit.
bool unconsumed(const Receipt& receipt) {
  bool memory = true;
  for (int slot = 0; slot < PHX_MC_EXECUTION_MEMORY_VALUES; ++slot) {
    memory = memory && (slot == PHX_MC_EXECUTION_MEMORY_LIMIT_BYTES || receipt.memory[slot] == 0);
  }
  return receipt.counters == decltype(receipt.counters){} && memory;
}

// Runs `call` in a scope refused by an expired deadline or by an earlier
// capacity refusal.
template <class Call>
Receipt in_refused_scope(int32_t refusal, Call&& call) {
  Execution execution(UINT64_MAX, refusal == PHX_MC_TIMEOUT ? 0.0 : INFINITY);
  if (refusal == PHX_MC_CAPACITY_EXCEEDED) {
    PHX_CHECK(phx_mc_execution_charge(execution.scope(), 1, 0) == PHX_MC_CAPACITY_EXCEEDED);
  }
  call();
  return execution.finish(refusal);
}

// Every coordinate and row ID is validated at near and far chunk edges and in
// the final partial chunk. Refused input publishes no output and leaves no
// work, query or memory trace on the scope.
void validation_covers_chunk_boundaries() {
  constexpr std::array<std::size_t, 7> rows{0, kChunkRows - 1, kChunkRows, 2 * kChunkRows - 1,
                                            1024 * kChunkRows - 1, 1024 * kChunkRows,
                                            kLargeCount - 1};
  constexpr std::array<double, 3> nonfinite{NAN, INFINITY, -INFINITY};
  constexpr std::array<double, 2> outside{0x1p121, -0x1p-121};
  for (const Kind& kind : kKinds) {
    BatchInput input(kind, kLargeCount);
    std::vector<int8_t> output(kLargeCount, 9);
    const auto expect = [&](int32_t status) {
      PHX_CHECK(batch(kind.predicate, input.p, input.ids, output.data(), kLargeCount) == status);
      PHX_CHECK(untouched(output));
    };
    {
      Execution execution;
      for (std::size_t row : rows) {
        for (int point = 0; point < kind.arity; ++point) {
          for (int axis : {0, kind.width - 1}) {
            double& slot = input.p[point][kind.width * row + axis];
            const double saved = slot;
            for (double value : nonfinite) {
              slot = value;
              expect(PHX_MC_NONFINITE_INPUT);
            }
            for (double value : outside) {
              slot = value;
              expect(PHX_MC_RANGE_ERROR);
            }
            slot = saved;
          }
        }
      }
      // An earlier domain violation never masks a later nonfinite coordinate.
      double& first = input.p[0].front();
      double& last = input.p[kind.arity - 1].back();
      const double saved_first = first, saved_last = last;
      first = 0x1p121;
      last = NAN;
      expect(PHX_MC_NONFINITE_INPUT);
      last = saved_last;
      expect(PHX_MC_RANGE_ERROR);
      first = saved_first;
      if (kind.symbolic) {
        for (std::size_t row : rows) {
          int64_t& id = input.ids[kind.arity * row + kind.arity - 1];
          const int64_t saved = id;
          id = input.ids[kind.arity * row];
          expect(PHX_MC_INVALID_INPUT);
          // Any coordinate error outranks a repeated row ID.
          last = 0x1p121;
          expect(PHX_MC_RANGE_ERROR);
          last = NAN;
          expect(PHX_MC_NONFINITE_INPUT);
          last = saved_last;
          id = saved;
        }
      }
      PHX_CHECK(unconsumed(execution.finish()));
    }
    // The restored batch is admitted and matches the scalar oracle in every chunk.
    Policy policy(7);
    PHX_CHECK(batch(kind.predicate, input.p, input.ids, output.data(), kLargeCount) == PHX_MC_OK);
    bool matches = true;
    for (std::size_t row = 0; row < kLargeCount; ++row) {
      matches = matches && output[row] == scalar(kind.predicate, input.p,
                                                 input.ids.data() + kind.arity * row, row,
                                                 kind.width);
    }
    PHX_CHECK(matches);
  }
}

// The checkpoint before the first chunk observes a refused scope before any
// input is read, so the refusal outranks even a nonfinite first coordinate.
// The refused call publishes nothing and leaves the scope's receipt unchanged.
void refused_scope_precedes_validation() {
  for (const Kind& kind : kKinds) {
    BatchInput input(kind, kLargeCount);
    const double saved = input.p[0].front();
    for (bool invalid : {false, true}) {
      input.p[0].front() = invalid ? NAN : saved;
      for (int32_t refusal : {PHX_MC_TIMEOUT, PHX_MC_CAPACITY_EXCEEDED}) {
        std::vector<int8_t> output(kLargeCount, 9);
        const Receipt receipt = in_refused_scope(refusal, [&] {
          PHX_CHECK(batch(kind.predicate, input.p, input.ids, output.data(), kLargeCount) ==
                    refusal);
        });
        PHX_CHECK(untouched(output));
        PHX_CHECK(receipt == in_refused_scope(refusal, [] {}));
      }
    }
    input.p[0].front() = saved;
  }
}

// A scope that has already been charged and whose deadline has then passed:
// the next predicate call refuses with TIMEOUT before reporting any input
// error, whatever the logical charge cadence, and publishes nothing. The
// timeout consumes no work, queries or memory.
template <class Call>
Receipt after_deadline(Call&& call) {
  Execution execution(UINT64_MAX, kDeadlineSeconds);
  PHX_CHECK(phx_mc_execution_charge(execution.scope(), 0, 0) == PHX_MC_OK);
  std::this_thread::sleep_for(kPastDeadline);
  call();
  return execution.finish(PHX_MC_TIMEOUT);
}

void expired_deadline_outranks_input_errors() {
  for (const Kind& kind : kKinds) {
    BatchInput input(kind, kCount);
    const auto timed_out = [&] {
      std::vector<int8_t> output(kCount, 9);
      const Receipt receipt = after_deadline([&] {
        PHX_CHECK(batch(kind.predicate, input.p, input.ids, output.data()) == PHX_MC_TIMEOUT);
      });
      PHX_CHECK(untouched(output));
      PHX_CHECK(unconsumed(receipt));
    };
    timed_out();
    double& first = input.p[0].front();
    const double saved = first;
    first = NAN;
    timed_out();
    first = 0x1p121;
    timed_out();
    first = saved;
    if (kind.symbolic) {
      input.ids[kind.arity - 1] = input.ids[0];
      timed_out();
    }
  }
}

struct TargetTimeout {
  int32_t call = PHX_MC_OK, target_end = PHX_MC_OK, settle = PHX_MC_OK, refusal_status = PHX_MC_OK;
  std::array<uint64_t, PHX_MC_EXECUTION_BATCH_REFUSAL_VALUES> refusal{};
  Receipt receipt;
  bool operator==(const TargetTimeout&) const = default;
};

// Runs `call` under an active native target after the scope's deadline passed.
template <class Call>
TargetTimeout target_timeout(Call&& call) {
  TargetTimeout out;
  Execution execution(UINT64_MAX, kDeadlineSeconds);
  void* ticket = nullptr;
  PHX_CHECK(phx_mc_execution_batch_admit(execution.scope(), nullptr, 0, 0, 0, 1 << 20, &ticket) ==
            PHX_MC_OK);
  void* target = nullptr;
  phx_mc_execution_usage* usage = nullptr;
  PHX_CHECK(phx_mc_execution_target_begin(ticket, &target, &usage) == PHX_MC_OK);
  std::this_thread::sleep_for(kPastDeadline);
  out.call = call(execution.scope());
  out.target_end = phx_mc_execution_target_end(target);
  out.settle = phx_mc_execution_batch_settle(ticket, 0, 0, 0, 0);
  out.refusal_status = phx_mc_execution_batch_refusal(execution.scope(), out.refusal.data());
  out.receipt = execution.finish(PHX_MC_TIMEOUT);
  return out;
}

// Under an active target, a predicate refused by the deadline leaves the same
// target refusal evidence, statuses and receipt as a zero charge refused by
// that deadline, and publishes nothing.
void target_timeout_keeps_refusal_evidence() {
  const Kind& kind = kKinds[7];  // InsphereSos: coordinates and row IDs.
  BatchInput input(kind, kCount);
  input.p[0].front() = NAN;
  std::vector<int8_t> output(kCount, 9);
  const TargetTimeout predicate = target_timeout([&](void*) {
    return batch(kind.predicate, input.p, input.ids, output.data());
  });
  PHX_CHECK(untouched(output));
  PHX_CHECK(predicate.call == PHX_MC_TIMEOUT && predicate.refusal_status == PHX_MC_OK);
  const TargetTimeout reference =
      target_timeout([](void* scope) { return phx_mc_execution_charge(scope, 0, 0); });
  PHX_CHECK(predicate == reference);
}
}  // namespace

int main() {
  predicate_equivalence(Predicate::Orient2d, 2, 3);
  predicate_equivalence(Predicate::Orient3d, 3, 4);
  predicate_equivalence(Predicate::Incircle, 2, 4);
  predicate_equivalence(Predicate::Insphere, 3, 5);
  predicate_equivalence(Predicate::Orient2dSos, 2, 3);
  predicate_equivalence(Predicate::Orient3dSos, 3, 4);
  predicate_equivalence(Predicate::IncircleSos, 2, 4);
  predicate_equivalence(Predicate::InsphereSos, 3, 5);
  location_equivalence();
  restricted_bound_equivalence();
  resource_refusal_is_atomic();
  validation_precedes_reservation();
  batch_borrows_host_ticket_once();
  finite_scratch_batch_is_admitted();
  validation_covers_chunk_boundaries();
  refused_scope_precedes_validation();
  expired_deadline_outranks_input_errors();
  target_timeout_keeps_refusal_evidence();
  return phx::mc::test::finish("test_parallel_kernels");
}
