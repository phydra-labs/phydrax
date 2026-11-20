// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Planar Ruppert probe: per-phase time, minor page faults and operator new
// census of phx_mc_constrained_delaunay_2d on the ledger rows (square 1e3,
// ring-slit 1e4, square 1e4), N warm calls each. Portable POSIX (macOS and
// Linux): steady_clock, getrusage(RUSAGE_SELF).ru_minflt, a replaced global
// operator new. It compiles cdt2d.cpp itself with the phase hook defined, and
// links every other meshcore object of an existing Release build:
//
//   B=<release build dir>; S=native/meshcore
//   c++ -std=c++20 -O3 -DNDEBUG -ffp-contract=off -fno-fast-math -DPHX_MC_BUILDING \
//     -DPHX_MC_BUILD_HASH='"probe"' -DPHX_MC_VERSION='"probe"' -I$S/include -I$S/src \
//     $S/tools/planar_probe.cpp \
//     $(ls $B/CMakeFiles/phydrax_meshcore_objects.dir/src/*.o | grep -v '/cdt2d.cpp.o') \
//     -o planar_probe        # add -fopenmp (Linux GCC) if the objects need it
//   ./planar_probe benchmarks/native_meshing_inputs 50
//
// A second binary built with -DPROBE_SCOPES -o planar_probe_scoped adds the
// refine breakdown (pop, process, segment, walk, cavity, new_vertex, split,
// legalize, after_insertion: inclusive, overlapping, accumulated per call; the timers perturb
// the totals, compare across platforms only). Keep planar_probe.cpp before
// the objects on the link line: cad_refine2d.o carries an unscoped copy of
// the inline mesher functions and the first definition wins.
//
// Columns per phase: median microseconds, mean minor faults, mean operator
// new calls and bytes. "call" is the whole C entry plus phx_mc_mesh_free.
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace probe {
struct Mark {
  const char* name;
  std::chrono::steady_clock::time_point time;
  long faults;
  std::uint64_t allocations;
  std::uint64_t bytes;
};
std::uint64_t allocations = 0;
std::uint64_t bytes = 0;
bool counting = false;
std::vector<Mark>* marks = nullptr;

long minor_faults() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_minflt;
}

void mark(const char* name) {
  if (marks == nullptr) return;
  const bool was = counting;
  counting = false;
  marks->push_back(Mark{name, std::chrono::steady_clock::now(), minor_faults(), allocations, bytes});
  counting = was;
}

// Inclusive accumulated time of the refine scopes (nested scopes overlap).
struct Accumulator {
  const char* name;
  double microseconds;
  std::uint64_t count;
};
Accumulator scopes[16];
int scope_count = 0;

struct Scope {
  int index;
  std::chrono::steady_clock::time_point start;
  explicit Scope(const char* name) {
    index = 0;
    while (index < scope_count && std::strcmp(scopes[index].name, name) != 0) ++index;
    if (index == scope_count) scopes[scope_count++] = Accumulator{name, 0.0, 0};
    start = std::chrono::steady_clock::now();
  }
  ~Scope() {
    scopes[index].microseconds +=
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    ++scopes[index].count;
  }
};
}  // namespace probe

void* operator new(std::size_t size) {
  if (probe::counting) {
    ++probe::allocations;
    probe::bytes += size;
  }
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

#define PHX_CDT2D_PHASE(name) ::probe::mark(name)
#ifdef PROBE_SCOPES
#define PROBE_CAT2(a, b) a##b
#define PROBE_CAT(a, b) PROBE_CAT2(a, b)
#define PHX_CDT2D_SCOPE(name) ::probe::Scope PROBE_CAT(probe_scope_, __LINE__)(name)
#endif
#include "../src/cdt2d.cpp"

namespace {

// Minimal .npy reader: version 1/2 header, little-endian, C order.
template <class T>
std::vector<T> load_npy(const std::string& path, const char* descr) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  unsigned char magic[8];
  if (std::fread(magic, 1, 8, file) != 8 || std::memcmp(magic, "\x93NUMPY", 6) != 0) std::exit(2);
  std::uint32_t header_length = 0;
  if (magic[6] == 1) {
    unsigned char length[2];
    if (std::fread(length, 1, 2, file) != 2) std::exit(2);
    header_length = length[0] | (length[1] << 8);
  } else {
    unsigned char length[4];
    if (std::fread(length, 1, 4, file) != 4) std::exit(2);
    header_length = length[0] | (length[1] << 8) | (length[2] << 16) |
                    (static_cast<std::uint32_t>(length[3]) << 24);
  }
  std::string header(header_length, '\0');
  if (std::fread(header.data(), 1, header_length, file) != header_length) std::exit(2);
  if (header.find(descr) == std::string::npos || header.find("'fortran_order': False") == std::string::npos) {
    std::fprintf(stderr, "%s: expected %s C-order\n", path.c_str(), descr);
    std::exit(2);
  }
  std::vector<T> values;
  T value;
  while (std::fread(&value, sizeof(T), 1, file) == 1) values.push_back(value);
  std::fclose(file);
  return values;
}

struct Row {
  const char* name;
  std::vector<double> points;
  std::vector<std::int32_t> segments;
  double max_area;
};

void run(const Row& row, int calls) {
  std::vector<std::vector<probe::Mark>> samples;
  std::vector<double> totals;
  std::vector<long> total_faults;
  std::vector<std::uint64_t> total_allocations, total_bytes;
  for (int call = -5; call < calls; ++call) {
    if (call == 0) probe::scope_count = 0;
    std::vector<probe::Mark> marks;
    marks.reserve(16);
    probe::marks = &marks;
    probe::allocations = probe::bytes = 0;
    const long faults0 = probe::minor_faults();
    probe::counting = true;
    probe::mark("start");
    phx_mc_mesh* mesh = nullptr;
    const int32_t status = phx_mc_constrained_delaunay_2d(
        static_cast<int64_t>(row.points.size() / 2), row.points.data(),
        static_cast<int64_t>(row.segments.size() / 2), row.segments.data(), 0, nullptr, 1, 20.0,
        row.max_area, &mesh);
    probe::mark("result");
    phx_mc_mesh_free(mesh);
    probe::mark("free");
    probe::counting = false;
    probe::marks = nullptr;
    if (status != PHX_MC_OK) {
      std::fprintf(stderr, "%s: status %d\n", row.name, status);
      std::exit(1);
    }
    if (call < 0) continue;
    totals.push_back(std::chrono::duration<double, std::micro>(marks.back().time - marks.front().time).count());
    total_faults.push_back(probe::minor_faults() - faults0);
    total_allocations.push_back(probe::allocations);
    total_bytes.push_back(probe::bytes);
    samples.push_back(std::move(marks));
  }
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  std::printf("%s  calls %d  call median %.1f us min %.1f us  faults/call %.1f  new/call %.1f  bytes/call %.0f\n",
              row.name, calls, median(totals), *std::min_element(totals.begin(), totals.end()),
              [&] { double s = 0; for (long f : total_faults) s += f; return s / calls; }(),
              [&] { double s = 0; for (auto a : total_allocations) s += a; return s / calls; }(),
              [&] { double s = 0; for (auto b : total_bytes) s += b; return s / calls; }());
  const std::size_t phases = samples.front().size();
  for (std::size_t p = 1; p < phases; ++p) {
    std::vector<double> times;
    double faults = 0, allocations = 0, bytes = 0;
    for (const auto& marks : samples) {
      times.push_back(std::chrono::duration<double, std::micro>(marks[p].time - marks[p - 1].time).count());
      faults += marks[p].faults - marks[p - 1].faults;
      allocations += marks[p].allocations - marks[p - 1].allocations;
      bytes += marks[p].bytes - marks[p - 1].bytes;
    }
    std::printf("  %-15s %9.1f us  faults %7.1f  new %7.1f  bytes %10.0f\n", samples.front()[p].name,
                median(times), faults / calls, allocations / calls, bytes / calls);
  }
  for (int s = 0; s < probe::scope_count; ++s) {
    std::printf("    %-20s %9.1f us/call  %9.1f entries/call  (inclusive, scoped build)\n",
                probe::scopes[s].name, probe::scopes[s].microseconds / calls,
                static_cast<double>(probe::scopes[s].count) / calls);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <benchmarks/native_meshing_inputs> [calls]\n", argv[0]);
    return 2;
  }
  const std::string inputs = argv[1];
  const int calls = argc > 2 ? std::atoi(argv[2]) : 50;
  const std::vector<double> square = {0, 0, 1, 0, 1, 1, 0, 1};
  const std::vector<std::int32_t> square_segments = {0, 1, 1, 2, 2, 3, 3, 0};
  const std::vector<double> ring = load_npy<double>(inputs + "/q2ring_pts.npy", "'<f8'");
  const std::vector<std::int32_t> ring_segments = load_npy<std::int32_t>(inputs + "/q2ring_seg.npy", "'<i4'");
  run(Row{"ruppert square 1e3", square, square_segments, 1.5 / 1000}, calls);
  run(Row{"ruppert ring-slit 1e4", ring, ring_segments, 1.5 / 10000}, calls);
  run(Row{"ruppert square 1e4", square, square_segments, 1.5 / 10000}, calls);
  return 0;
}
