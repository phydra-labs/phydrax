//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Independent exact oracle for phx_mc_arrange_tetrahedra_exact. The oracle
// owns its own small signed big integers and recomputes every job's chain
// polytope from the request alone by brute-force vertex enumeration over all
// plane triples. It shares no code with the native pool, welding, caches or
// section engine; native topology is accepted only when every vertex, facet
// incidence, outward strict convexity, edge pairing and exact six-volume
// agrees with that enumeration.
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

// ------------------------------------------------------------ oracle integers

using Limbs = std::vector<std::uint32_t>;

void trim_limbs(Limbs& value) {
  while (!value.empty() && value.back() == 0) value.pop_back();
}

int compare_limbs(const Limbs& a, const Limbs& b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (std::size_t index = a.size(); index-- > 0;)
    if (a[index] != b[index]) return a[index] < b[index] ? -1 : 1;
  return 0;
}

Limbs add_limbs(const Limbs& a, const Limbs& b) {
  Limbs out(std::max(a.size(), b.size()) + 1, 0);
  std::uint64_t carry = 0;
  for (std::size_t index = 0; index + 1 < out.size(); ++index) {
    const std::uint64_t sum = carry + (index < a.size() ? a[index] : 0u) +
                              (index < b.size() ? b[index] : 0u);
    out[index] = static_cast<std::uint32_t>(sum);
    carry = sum >> 32;
  }
  out.back() = static_cast<std::uint32_t>(carry);
  trim_limbs(out);
  return out;
}

// a - b for a >= b.
Limbs subtract_limbs(const Limbs& a, const Limbs& b) {
  Limbs out(a.size(), 0);
  std::int64_t borrow = 0;
  for (std::size_t index = 0; index < a.size(); ++index) {
    std::int64_t value = static_cast<std::int64_t>(a[index]) -
                         static_cast<std::int64_t>(index < b.size() ? b[index] : 0u) - borrow;
    borrow = value < 0 ? 1 : 0;
    if (value < 0) value += std::int64_t{1} << 32;
    out[index] = static_cast<std::uint32_t>(value);
  }
  trim_limbs(out);
  return out;
}

Limbs multiply_limbs(const Limbs& a, const Limbs& b) {
  if (a.empty() || b.empty()) return {};
  Limbs out(a.size() + b.size(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::uint64_t carry = 0;
    for (std::size_t j = 0; j < b.size(); ++j) {
      const std::uint64_t value =
          static_cast<std::uint64_t>(a[i]) * b[j] + out[i + j] + carry;
      out[i + j] = static_cast<std::uint32_t>(value);
      carry = value >> 32;
    }
    for (std::size_t k = i + b.size(); carry != 0; ++k) {
      const std::uint64_t value = static_cast<std::uint64_t>(out[k]) + carry;
      out[k] = static_cast<std::uint32_t>(value);
      carry = value >> 32;
    }
  }
  trim_limbs(out);
  return out;
}

struct Big {
  bool negative = false;
  Limbs limbs;

  Big() = default;
  Big(long long value) : negative(value < 0) {  // NOLINT(google-explicit-constructor)
    unsigned long long magnitude = value < 0 ? 0ULL - static_cast<unsigned long long>(value)
                                             : static_cast<unsigned long long>(value);
    for (; magnitude != 0; magnitude >>= 32) limbs.push_back(static_cast<std::uint32_t>(magnitude));
  }
  static Big make(bool negative, Limbs limbs) {
    Big out;
    out.limbs = std::move(limbs);
    trim_limbs(out.limbs);
    out.negative = negative && !out.limbs.empty();
    return out;
  }
  int sign() const { return limbs.empty() ? 0 : negative ? -1 : 1; }
};

Big operator-(const Big& a) { return Big::make(!a.negative, a.limbs); }

Big operator+(const Big& a, const Big& b) {
  if (a.negative == b.negative) return Big::make(a.negative, add_limbs(a.limbs, b.limbs));
  const int order = compare_limbs(a.limbs, b.limbs);
  if (order == 0) return Big();
  return order > 0 ? Big::make(a.negative, subtract_limbs(a.limbs, b.limbs))
                   : Big::make(b.negative, subtract_limbs(b.limbs, a.limbs));
}

Big operator-(const Big& a, const Big& b) { return a + (-b); }

Big operator*(const Big& a, const Big& b) {
  return Big::make(a.negative != b.negative, multiply_limbs(a.limbs, b.limbs));
}

bool operator==(const Big& a, const Big& b) { return a.negative == b.negative && a.limbs == b.limbs; }

Big power(const Big& base, int exponent) {
  Big out = 1;
  for (int step = 0; step < exponent; ++step) out = out * base;
  return out;
}

// --------------------------------------------------------------- oracle geometry

using Point = std::array<Big, 4>;  // (X, Y, Z, W), W > 0
using Row = std::array<Big, 4>;    // a*x + b*y + c*z + d

Big minor3(const std::array<Big, 4>& a, const std::array<Big, 4>& b,
           const std::array<Big, 4>& c, int x, int y, int z) {
  return a[x] * (b[y] * c[z] - b[z] * c[y]) - a[y] * (b[x] * c[z] - b[z] * c[x]) +
         a[z] * (b[x] * c[y] - b[y] * c[x]);
}

Big det4(const Point& a, const Point& b, const Point& c, const Point& d) {
  return a[0] * minor3(b, c, d, 1, 2, 3) - a[1] * minor3(b, c, d, 0, 2, 3) +
         a[2] * minor3(b, c, d, 0, 1, 3) - a[3] * minor3(b, c, d, 0, 1, 2);
}

Big dot(const Row& row, const Point& point) {
  return row[0] * point[0] + row[1] * point[1] + row[2] * point[2] + row[3] * point[3];
}

bool same_point(const Point& a, const Point& b) {
  for (int axis = 0; axis < 3; ++axis)
    if (!(a[axis] * b[3] == b[axis] * a[3])) return false;
  return true;
}

std::optional<Point> meet(const Row& a, const Row& b, const Row& c) {
  const Big w = minor3(a, b, c, 0, 1, 2);
  if (w.sign() == 0) return std::nullopt;
  Point point{-minor3(a, b, c, 1, 2, 3), minor3(a, b, c, 0, 2, 3), -minor3(a, b, c, 0, 1, 3), w};
  if (w.sign() < 0)
    for (auto& value : point) value = -value;
  return point;
}

Row plane_through(const Point& p, const Point& q, const Point& r) {
  return {minor3(p, q, r, 1, 2, 3), -minor3(p, q, r, 0, 2, 3), minor3(p, q, r, 0, 1, 3),
          -minor3(p, q, r, 0, 1, 2)};
}

bool proportional(const Row& a, const Row& b) {
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      if (!(a[i] * b[j] == a[j] * b[i])) return false;
  return true;
}

// Exact rational n / d with d > 0.
struct Ratio {
  Big numerator = 0;
  Big denominator = 1;
};
Ratio operator+(const Ratio& a, const Ratio& b) {
  return {a.numerator * b.denominator + b.numerator * a.denominator, a.denominator * b.denominator};
}
bool same_ratio(const Ratio& a, const Ratio& b) {
  return a.numerator * b.denominator == b.numerator * a.denominator;
}

// ------------------------------------------------------------- request builder

struct PackedRows {
  std::vector<std::uint32_t> words;
  std::vector<std::int64_t> offsets{0};
  std::vector<std::int8_t> signs;
};

PackedRows pack_rows(const std::vector<std::array<Big, 4>>& rows) {
  PackedRows out;
  for (const auto& row : rows)
    for (const auto& value : row) {
      out.words.insert(out.words.end(), value.limbs.begin(), value.limbs.end());
      out.signs.push_back(static_cast<std::int8_t>(value.sign()));
      out.offsets.push_back(static_cast<std::int64_t>(out.words.size()));
    }
  return out;
}

std::int32_t from_tetrahedron(std::int32_t tetrahedron) { return -(tetrahedron + 1); }

struct ArrangementInput {
  std::vector<Point> points;
  std::vector<Row> planes;
  std::vector<std::int32_t> tetrahedra, tetrahedron_planes, parents, cut_planes;
  std::vector<std::int64_t> cut_offsets{0};

  std::int32_t point(const Big& x, const Big& y, const Big& z, const Big& w = Big(1)) {
    points.push_back({x, y, z, w});
    return static_cast<std::int32_t>(points.size()) - 1;
  }

  // Signed 1-based reference of `row <= 0` over canonical, deduplicated rows.
  std::int32_t halfspace(Row row) {
    int leading = 0;
    for (int axis = 0; axis < 3 && leading == 0; ++axis) leading = row[axis].sign();
    PHX_CHECK(leading != 0);
    if (leading < 0)
      for (auto& value : row) value = -value;
    for (std::size_t index = 0; index < planes.size(); ++index)
      if (proportional(planes[index], row)) return leading * static_cast<std::int32_t>(index + 1);
    planes.push_back(row);
    return leading * static_cast<std::int32_t>(planes.size());
  }

  Row signed_row(std::int32_t reference) const {
    Row row = planes[static_cast<std::size_t>(reference > 0 ? reference - 1 : -reference - 1)];
    if (reference < 0)
      for (auto& value : row) value = -value;
    return row;
  }

  int orientation(const std::array<std::int32_t, 4>& corners) const {
    return -det4(points[corners[0]], points[corners[1]], points[corners[2]], points[corners[3]])
                .sign();
  }

  // A positive tetrahedron (corners 1 and 2 swap if needed) with outward
  // faces opposite each corner.
  std::int32_t tetrahedron(std::array<std::int32_t, 4> corners) {
    if (orientation(corners) < 0) std::swap(corners[1], corners[2]);
    PHX_CHECK(orientation(corners) > 0);
    for (int face = 0; face < 4; ++face) {
      std::array<std::int32_t, 3> others{};
      int count = 0;
      for (int corner = 0; corner < 4; ++corner)
        if (corner != face) others[count++] = corners[corner];
      std::int32_t reference =
          halfspace(plane_through(points[others[0]], points[others[1]], points[others[2]]));
      if (dot(signed_row(reference), points[corners[face]]).sign() > 0) reference = -reference;
      tetrahedron_planes.push_back(reference);
    }
    tetrahedra.insert(tetrahedra.end(), corners.begin(), corners.end());
    return static_cast<std::int32_t>(tetrahedra.size() / 4) - 1;
  }

  std::int32_t job(std::int32_t parent, std::initializer_list<std::int32_t> cuts) {
    parents.push_back(parent);
    cut_planes.insert(cut_planes.end(), cuts.begin(), cuts.end());
    cut_offsets.push_back(static_cast<std::int64_t>(cut_planes.size()));
    return static_cast<std::int32_t>(parents.size()) - 1;
  }
};

struct ArrangementOutput {
  int32_t status = -1;
  std::array<std::int64_t, PHX_MC_TETRAHEDRA_ARRANGEMENT_EVIDENCE> evidence{};
  bool published = false;
  std::vector<std::uint32_t> words;
  std::vector<std::int64_t> word_offsets;
  std::vector<std::int8_t> signs;
  std::vector<Point> vertices;
  std::vector<std::int32_t> input_index;
  std::vector<std::int64_t> cell_face_offsets, face_offsets;
  std::vector<std::int32_t> face_vertices, face_planes;
  std::array<std::int64_t, PHX_MC_TETRAHEDRA_ARRANGEMENT_COUNTERS> counters{};

  bool same(const ArrangementOutput& other) const {
    return status == other.status && published == other.published && words == other.words &&
           word_offsets == other.word_offsets && signs == other.signs &&
           input_index == other.input_index && cell_face_offsets == other.cell_face_offsets &&
           face_offsets == other.face_offsets && face_vertices == other.face_vertices &&
           face_planes == other.face_planes && counters == other.counters;
  }
};

using PackHook = std::function<void(PackedRows&, PackedRows&)>;
using RequestHook = std::function<void(phx_mc_arrange_tetrahedra_exact_request&)>;

ArrangementOutput run_arrangement(const ArrangementInput& input,
                                  std::int64_t scratch = std::int64_t{1} << 28,
                                  std::int64_t work = std::int64_t{1} << 40,
                                  const PackHook& pack_hook = {},
                                  const RequestHook& request_hook = {}) {
  PackedRows points = pack_rows(input.points), planes = pack_rows(input.planes);
  if (pack_hook) pack_hook(points, planes);
  ArrangementOutput out;
  out.evidence.fill(-7);
  phx_mc_tetrahedra_arrangement* handle = nullptr;
  phx_mc_arrange_tetrahedra_exact_request request{
      .point_count = static_cast<std::int64_t>(input.points.size()),
      .plane_count = static_cast<std::int64_t>(input.planes.size()),
      .tetrahedron_count = static_cast<std::int64_t>(input.tetrahedra.size() / 4),
      .job_count = static_cast<std::int64_t>(input.parents.size()),
      .cut_count = static_cast<std::int64_t>(input.cut_planes.size()),
      .point_word_count = static_cast<std::int64_t>(points.words.size()),
      .plane_word_count = static_cast<std::int64_t>(planes.words.size()),
      .point_words = points.words.data(),
      .point_offsets = points.offsets.data(),
      .point_signs = points.signs.data(),
      .plane_words = planes.words.data(),
      .plane_offsets = planes.offsets.data(),
      .plane_signs = planes.signs.data(),
      .tetrahedra = input.tetrahedra.data(),
      .tetrahedron_planes = input.tetrahedron_planes.data(),
      .parents = input.parents.data(),
      .cut_offsets = input.cut_offsets.data(),
      .cut_planes = input.cut_planes.data(),
      .scratch_limit = scratch,
      .work_limit = work,
      .evidence = out.evidence.data(),
      .result = &handle,
  };
  if (request_hook) request_hook(request);
  out.status = phx_mc_arrange_tetrahedra_exact(&request);
  out.published = handle != nullptr;
  PHX_CHECK(out.published == (out.status == PHX_MC_OK));
  if (handle == nullptr) return out;
  std::int64_t sizes[PHX_MC_TETRAHEDRA_ARRANGEMENT_SIZES] = {-1, -1, -1, -1, -1};
  PHX_CHECK(phx_mc_tetrahedra_arrangement_sizes(handle, sizes) == PHX_MC_OK);
  const auto vertices = static_cast<std::size_t>(sizes[0]);
  out.words.resize(static_cast<std::size_t>(sizes[1]));
  out.word_offsets.resize(4 * vertices + 1);
  out.signs.resize(4 * vertices);
  out.input_index.resize(vertices);
  out.cell_face_offsets.resize(static_cast<std::size_t>(sizes[2]) + 1);
  out.face_offsets.resize(static_cast<std::size_t>(sizes[3]) + 1);
  out.face_vertices.resize(static_cast<std::size_t>(sizes[4]));
  out.face_planes.resize(static_cast<std::size_t>(sizes[3]));
  PHX_CHECK(sizes[2] == static_cast<std::int64_t>(input.parents.size()));
  PHX_CHECK(phx_mc_tetrahedra_arrangement_export(
                handle, out.words.data(), out.word_offsets.data(), out.signs.data(),
                out.input_index.data(), out.cell_face_offsets.data(), out.face_offsets.data(),
                out.face_vertices.data(), out.face_planes.data(), out.counters.data()) == PHX_MC_OK);
  phx_mc_tetrahedra_arrangement_free(handle);
  for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
    Point point;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      const std::size_t scalar = 4 * vertex + axis;
      const auto begin = static_cast<std::size_t>(out.word_offsets[scalar]);
      const auto end = static_cast<std::size_t>(out.word_offsets[scalar + 1]);
      PHX_CHECK((begin == end) == (out.signs[scalar] == 0));
      point[axis] = Big::make(out.signs[scalar] < 0,
                              Limbs(out.words.begin() + static_cast<std::ptrdiff_t>(begin),
                                    out.words.begin() + static_cast<std::ptrdiff_t>(end)));
    }
    out.vertices.push_back(point);
  }
  return out;
}

// --------------------------------------------------------------------- oracle

std::vector<std::int32_t> job_chain(const ArrangementInput& input, std::int32_t job) {
  std::vector<std::int32_t> chain;
  for (std::int32_t current = job;;) {
    for (std::int64_t cut = input.cut_offsets[current]; cut < input.cut_offsets[current + 1]; ++cut)
      chain.push_back(input.cut_planes[static_cast<std::size_t>(cut)]);
    const std::int32_t parent = input.parents[current];
    if (parent < 0) {
      const std::int32_t tetrahedron = -parent - 1;
      for (int face = 0; face < 4; ++face)
        chain.push_back(input.tetrahedron_planes[4 * tetrahedron + face]);
      return chain;
    }
    current = parent;
  }
}

// Every feasible three-plane intersection of the chain: exactly the vertices
// of the closed chain polytope.
std::vector<Point> enumerate_vertices(const ArrangementInput& input,
                                      const std::vector<std::int32_t>& chain) {
  std::vector<Row> rows;
  for (const std::int32_t reference : chain) rows.push_back(input.signed_row(reference));
  std::vector<Point> vertices;
  for (std::size_t a = 0; a < rows.size(); ++a)
    for (std::size_t b = a + 1; b < rows.size(); ++b)
      for (std::size_t c = b + 1; c < rows.size(); ++c) {
        const auto point = meet(rows[a], rows[b], rows[c]);
        if (!point) continue;
        bool feasible = true;
        for (const auto& row : rows) feasible = feasible && dot(row, *point).sign() <= 0;
        if (!feasible) continue;
        bool known = false;
        for (const auto& vertex : vertices) known = known || same_point(vertex, *point);
        if (!known) vertices.push_back(*point);
      }
  return vertices;
}

std::array<Big, 4> difference(const Point& to, const Point& from) {
  return {to[0] * from[3] - from[0] * to[3], to[1] * from[3] - from[1] * to[3],
          to[2] * from[3] - from[2] * to[3], Big()};
}

bool full_dimensional(const std::vector<Point>& vertices) {
  if (vertices.size() < 4) return false;
  for (std::size_t i = 1; i < vertices.size(); ++i)
    for (std::size_t j = i + 1; j < vertices.size(); ++j)
      for (std::size_t k = j + 1; k < vertices.size(); ++k)
        if (minor3(difference(vertices[i], vertices[0]), difference(vertices[j], vertices[0]),
                   difference(vertices[k], vertices[0]), 0, 1, 2)
                .sign() != 0)
          return true;
  return false;
}

std::vector<std::int32_t> job_loop(const ArrangementOutput& out, std::int64_t face) {
  const auto begin = out.face_vertices.begin() + out.face_offsets[static_cast<std::size_t>(face)];
  const auto end = out.face_vertices.begin() + out.face_offsets[static_cast<std::size_t>(face) + 1];
  return std::vector<std::int32_t>(begin, end);
}

std::set<std::int32_t> job_vertices(const ArrangementOutput& out, std::int32_t job) {
  std::set<std::int32_t> vertices;
  for (std::int64_t face = out.cell_face_offsets[job]; face < out.cell_face_offsets[job + 1]; ++face)
    for (const std::int32_t vertex : job_loop(out, face)) vertices.insert(vertex);
  return vertices;
}

// Validates the whole published record against the independent oracle and
// returns each job's exact six-volume.
std::vector<Ratio> check_arrangement(const ArrangementInput& input, const ArrangementOutput& out) {
  std::vector<Ratio> volumes;
  PHX_CHECK(out.status == PHX_MC_OK && out.published);
  if (out.status != PHX_MC_OK || !out.published) return volumes;
  const auto jobs = static_cast<std::int64_t>(input.parents.size());
  PHX_CHECK(out.evidence[0] > 0 && out.evidence[1] == jobs && out.evidence[2] > 0 &&
            out.evidence[3] == jobs && out.evidence[4] == 0 && out.evidence[5] == -1);
  PHX_CHECK(out.cell_face_offsets.front() == 0 && out.face_offsets.front() == 0);
  PHX_CHECK(out.cell_face_offsets.back() == static_cast<std::int64_t>(out.face_planes.size()));
  PHX_CHECK(out.face_offsets.back() == static_cast<std::int64_t>(out.face_vertices.size()));
  // Global welding: one exact point per id; input index is the smallest equal input.
  for (std::size_t vertex = 0; vertex < out.vertices.size(); ++vertex) {
    PHX_CHECK(out.vertices[vertex][3].sign() > 0);
    std::int32_t smallest = -1;
    for (std::size_t point = 0; point < input.points.size() && smallest < 0; ++point)
      if (same_point(out.vertices[vertex], input.points[point])) smallest = static_cast<std::int32_t>(point);
    PHX_CHECK(out.input_index[vertex] == smallest);
    for (std::size_t other = 0; other < vertex; ++other)
      PHX_CHECK(!same_point(out.vertices[vertex], out.vertices[other]));
  }
  // Compacted first-use numbering: each loop introduces the next consecutive
  // ids (in loop order before its rotation to the smallest vertex).
  std::int32_t next = 0;
  for (std::size_t face = 0; face < out.face_planes.size(); ++face) {
    const auto loop = job_loop(out, static_cast<std::int64_t>(face));
    std::int32_t fresh = 0;
    for (const std::int32_t vertex : loop) fresh += vertex >= next ? 1 : 0;
    for (const std::int32_t vertex : loop) PHX_CHECK(vertex >= 0 && vertex < next + fresh);
    next += fresh;
  }
  PHX_CHECK(next == static_cast<std::int32_t>(out.vertices.size()));
  for (std::int32_t job = 0; job < jobs; ++job) {
    const auto chain = job_chain(input, job);
    const auto oracle = enumerate_vertices(input, chain);
    const std::int64_t first_face = out.cell_face_offsets[job], last_face = out.cell_face_offsets[job + 1];
    Ratio six;
    if (!full_dimensional(oracle)) {
      PHX_CHECK(first_face == last_face);
      volumes.push_back(six);
      continue;
    }
    PHX_CHECK(last_face - first_face >= 4);
    const auto used = job_vertices(out, job);
    PHX_CHECK(used.size() == oracle.size());
    for (const std::int32_t vertex : used) {
      int matches = 0;
      for (const auto& point : oracle) matches += same_point(out.vertices[vertex], point) ? 1 : 0;
      PHX_CHECK(matches == 1);
    }
    std::map<std::pair<std::int32_t, std::int32_t>, int> directed;
    std::set<std::int32_t> face_rows;
    for (std::int64_t face = first_face; face < last_face; ++face) {
      const std::int32_t reference = out.face_planes[static_cast<std::size_t>(face)];
      PHX_CHECK(std::count(chain.begin(), chain.end(), reference) > 0);
      PHX_CHECK(face_rows.insert(reference > 0 ? reference : -reference).second);
      const Row row = input.signed_row(reference);
      const auto loop = job_loop(out, face);
      PHX_CHECK(loop.size() >= 3);
      if (loop.size() < 3) continue;
      PHX_CHECK(loop.front() == *std::min_element(loop.begin(), loop.end()));
      int on_plane = 0;
      for (const auto& point : oracle) on_plane += dot(row, point).sign() == 0 ? 1 : 0;
      PHX_CHECK(on_plane == static_cast<int>(loop.size()));
      const std::array<Big, 4> normal{row[0], row[1], row[2], Big()};
      for (std::size_t slot = 0; slot < loop.size(); ++slot) {
        const Point& a = out.vertices[loop[slot]];
        const Point& b = out.vertices[loop[(slot + 1) % loop.size()]];
        const Point& c = out.vertices[loop[(slot + 2) % loop.size()]];
        PHX_CHECK(dot(row, a).sign() == 0);
        // Strictly convex and counterclockwise about the outward normal.
        PHX_CHECK(minor3(difference(b, a), difference(c, b), normal, 0, 1, 2).sign() > 0);
        ++directed[{loop[slot], loop[(slot + 1) % loop.size()]}];
      }
      for (std::size_t slot = 1; slot + 1 < loop.size(); ++slot) {
        const Point& a = out.vertices[loop[0]];
        const Point& b = out.vertices[loop[slot]];
        const Point& c = out.vertices[loop[slot + 1]];
        six = six + Ratio{minor3(a, b, c, 0, 1, 2), a[3] * b[3] * c[3]};
      }
    }
    for (const auto& [edge, count] : directed) {
      PHX_CHECK(count == 1);
      const auto reverse = directed.find({edge.second, edge.first});
      PHX_CHECK(reverse != directed.end() && reverse->second == 1);
    }
    const auto edges = static_cast<std::int64_t>(directed.size() / 2);
    PHX_CHECK(static_cast<std::int64_t>(used.size()) - edges + (last_face - first_face) == 2);
    PHX_CHECK(six.numerator.sign() > 0);
    volumes.push_back(six);
  }
  return volumes;
}

// Loop `a` is loop `b` traversed backwards (same welded ids).
bool reversed_loops(std::vector<std::int32_t> a, const std::vector<std::int32_t>& b) {
  if (a.size() != b.size()) return false;
  std::reverse(a.begin(), a.end());
  for (std::size_t shift = 0; shift < a.size(); ++shift) {
    std::rotate(a.begin(), a.begin() + 1, a.end());
    if (a == b) return true;
  }
  return false;
}

std::int64_t face_with_plane(const ArrangementOutput& out, std::int32_t job, std::int32_t reference) {
  for (std::int64_t face = out.cell_face_offsets[job]; face < out.cell_face_offsets[job + 1]; ++face)
    if (out.face_planes[static_cast<std::size_t>(face)] == reference) return face;
  return -1;
}

std::int32_t vertex_at(const ArrangementOutput& out, const Point& point) {
  for (std::size_t vertex = 0; vertex < out.vertices.size(); ++vertex)
    if (same_point(out.vertices[vertex], point)) return static_cast<std::int32_t>(vertex);
  return -1;
}

Ratio ratio(long long numerator, long long denominator = 1) { return {numerator, denominator}; }

// Job clipping `parent` by tetrahedron `other`'s outward faces in slot order;
// any trailing `extra` cut makes it an arbitrary job.
std::int32_t meet_job(ArrangementInput& input, std::int32_t parent, std::int32_t other,
                      std::initializer_list<std::int32_t> extra = {}) {
  input.parents.push_back(parent);
  for (int face = 0; face < 4; ++face)
    input.cut_planes.push_back(input.tetrahedron_planes[static_cast<std::size_t>(4 * other + face)]);
  input.cut_planes.insert(input.cut_planes.end(), extra.begin(), extra.end());
  input.cut_offsets.push_back(static_cast<std::int64_t>(input.cut_planes.size()));
  return static_cast<std::int32_t>(input.parents.size()) - 1;
}

// Jobs `a` and `b` publish identical faces: planes, order and welded loops.
bool same_job(const ArrangementOutput& out, std::int32_t a, std::int32_t b) {
  const std::int64_t first_a = out.cell_face_offsets[a], first_b = out.cell_face_offsets[b];
  const std::int64_t count = out.cell_face_offsets[a + 1] - first_a;
  if (count != out.cell_face_offsets[b + 1] - first_b) return false;
  for (std::int64_t face = 0; face < count; ++face)
    if (out.face_planes[static_cast<std::size_t>(first_a + face)] !=
            out.face_planes[static_cast<std::size_t>(first_b + face)] ||
        job_loop(out, first_a + face) != job_loop(out, first_b + face))
      return false;
  return true;
}

// ---------------------------------------------------------------------- cases

ArrangementInput unit_tetrahedron(std::int32_t scale = 1) {
  ArrangementInput input;
  const auto p0 = input.point(0, 0, 0), p1 = input.point(scale, 0, 0);
  const auto p2 = input.point(0, scale, 0), p3 = input.point(0, 0, scale);
  input.tetrahedron({p0, p1, p2, p3});
  return input;
}

void test_identity_and_corner_permutation() {
  ArrangementInput input = unit_tetrahedron();
  input.job(from_tetrahedron(0), {});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 1 && same_ratio(volumes[0], ratio(1)));
  if (volumes.size() != 1) return;
  PHX_CHECK(out.vertices.size() == 4 && out.face_planes.size() == 4);
  // No cut: faces are the supplied outward slots, vertices the input points.
  for (int face = 0; face < 4; ++face) PHX_CHECK(out.face_planes[face] == input.tetrahedron_planes[face]);
  std::vector<std::int32_t> inputs = out.input_index;
  std::sort(inputs.begin(), inputs.end());
  PHX_CHECK((inputs == std::vector<std::int32_t>{0, 1, 2, 3}));
  PHX_CHECK(out.counters[1] == 0 && out.counters[4] == 0);
  // An even corner permutation describes the same region exactly.
  ArrangementInput rotated;
  rotated.points = input.points;
  rotated.tetrahedron({1, 2, 0, 3});
  rotated.job(from_tetrahedron(0), {});
  const auto rotated_volumes = check_arrangement(rotated, run_arrangement(rotated));
  PHX_CHECK(rotated_volumes.size() == 1 && same_ratio(rotated_volumes[0], ratio(1)));
  // Determinism: identical bytes on rerun.
  PHX_CHECK(run_arrangement(input).same(out));
}

ArrangementInput partition_dag() {
  ArrangementInput input = unit_tetrahedron(4);
  const auto a = input.halfspace({1, 0, 0, -1});  // x <= 1
  const auto b = input.halfspace({0, 1, 0, -1});  // y <= 1
  const auto j0 = input.job(from_tetrahedron(0), {a});
  const auto j1 = input.job(from_tetrahedron(0), {-a});
  input.job(j0, {b});
  input.job(j0, {-b});
  input.job(j1, {});
  return input;
}

void test_dag_partition_and_pool_reuse() {
  const ArrangementInput input = partition_dag();
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 5);
  if (volumes.size() != 5) return;
  // Independent volumes: 64 - 27, 27 = (4-1)^3, 18 + 19 = 37.
  PHX_CHECK(same_ratio(volumes[0], ratio(37)) && same_ratio(volumes[1], ratio(27)));
  PHX_CHECK(same_ratio(volumes[2], ratio(18)) && same_ratio(volumes[3], ratio(19)));
  PHX_CHECK(same_ratio(volumes[4], ratio(27)));
  PHX_CHECK(same_ratio(volumes[0] + volumes[1], ratio(64)));
  PHX_CHECK(same_ratio(volumes[2] + volumes[3], volumes[0]));
  // The two section caps are one welded polygon traversed oppositely.
  const std::int32_t a = input.cut_planes[0];
  const auto cap = face_with_plane(out, 0, a), opposite = face_with_plane(out, 1, -a);
  PHX_CHECK(cap >= 0 && opposite >= 0);
  if (cap >= 0 && opposite >= 0) PHX_CHECK(reversed_loops(job_loop(out, cap), job_loop(out, opposite)));
  // A job without cuts repeats its immutable parent.
  PHX_CHECK(out.cell_face_offsets[5] - out.cell_face_offsets[4] == out.cell_face_offsets[2] - out.cell_face_offsets[1]);
  PHX_CHECK(job_vertices(out, 4) == job_vertices(out, 1));
  // Four sections; the -a and -b siblings reuse sides and crossings exactly.
  PHX_CHECK(out.counters[0] == 0 && out.counters[1] == 4);
  PHX_CHECK(out.counters[3] >= 4 + 6 && out.counters[5] >= 3 + 3);
  PHX_CHECK(out.counters[7] >= static_cast<std::int64_t>(out.vertices.size()));
}

void test_contact_empty_sliver_and_vertex_sections() {
  ArrangementInput input = unit_tetrahedron();
  const auto whole = input.halfspace({-1, -1, -1, 0});
  const auto vertex = input.halfspace({1, 1, 1, 0});
  const auto face = input.halfspace({1, 0, 0, 0});
  const auto edge = input.halfspace({1, 1, 0, 0});
  const auto far = input.halfspace({1, 0, 0, 1});
  const auto low = input.halfspace({-1000, 0, 0, 1});
  const auto high = input.halfspace({1000, 0, 0, -2});
  const auto diagonal = input.halfspace({1, -1, 0, 0});
  const auto apex = input.halfspace({1, 1, -2, 0});
  const auto j0 = input.job(from_tetrahedron(0), {whole});
  const auto j1 = input.job(from_tetrahedron(0), {vertex});
  input.job(from_tetrahedron(0), {face});
  input.job(from_tetrahedron(0), {edge});
  const auto j4 = input.job(from_tetrahedron(0), {far});
  input.job(j4, {diagonal});
  input.job(j1, {});
  input.job(from_tetrahedron(0), {low, high});
  input.job(from_tetrahedron(0), {diagonal});
  input.job(from_tetrahedron(0), {apex});
  input.job(j0, {-diagonal});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 11);
  if (volumes.size() != 11) return;
  PHX_CHECK(same_ratio(volumes[0], ratio(1)));
  for (int job = 1; job <= 6; ++job) PHX_CHECK(volumes[job].numerator.sign() == 0);
  // (999^3 - 998^3) / 10^9: a sliver between x = 1/1000 and x = 2/1000.
  PHX_CHECK(same_ratio(volumes[7], ratio(2991007, 1000000000)));
  // x = y holds an edge of the tetrahedron and halves it.
  PHX_CHECK(same_ratio(volumes[8], ratio(1, 2)) && same_ratio(volumes[10], ratio(1, 2)));
  PHX_CHECK(volumes[9].numerator.sign() > 0);
  // Both empty parents propagate without clipping or plane evaluation.
  PHX_CHECK(out.counters[0] == 2);
}

void test_welding_and_rounding_buckets() {
  ArrangementInput input = unit_tetrahedron(2);
  // An unused input point inside edge (0,0,0)-(2,0,0).
  const auto marker = input.point(1, 0, 0);
  const auto first = input.halfspace({1, 1, 0, -1});   // through the marker
  const auto second = input.halfspace({1, 0, 1, -1});  // through the marker
  input.job(from_tetrahedron(0), {first});
  input.job(from_tetrahedron(0), {second});
  // Two tetrahedra whose corners 10 and 10 + 2^-80 share one binary64 value.
  const Big scale = power(2, 80);
  const auto q0 = input.point(10, 0, 0), q1 = input.point(11, 0, 0);
  const auto q2 = input.point(10, 1, 0), q3 = input.point(10, 0, 1);
  const auto r0 = input.point(Big(10) * scale + 1, 0, 0, scale);
  const auto r1 = input.point(Big(11) * scale + 1, 0, 0, scale);
  const auto r2 = input.point(Big(10) * scale + 1, scale, 0, scale);
  const auto r3 = input.point(Big(10) * scale + 1, 0, scale, scale);
  const auto near = input.tetrahedron({q0, q1, q2, q3});
  const auto shifted = input.tetrahedron({r0, r1, r2, r3});
  input.job(from_tetrahedron(near), {});
  input.job(from_tetrahedron(shifted), {});
  // A repeated exact input point is reported by its smallest index.
  const auto duplicate = input.point(0, 0, 0);
  const auto m1 = input.point(-1, 0, 0), m2 = input.point(0, -1, 0), m3 = input.point(0, 0, -1);
  const auto mirrored = input.tetrahedron({duplicate, m1, m2, m3});
  input.job(from_tetrahedron(mirrored), {});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 5);
  if (volumes.size() != 5) return;
  const auto x = vertex_at(out, input.points[static_cast<std::size_t>(marker)]);
  PHX_CHECK(x >= 0);
  if (x >= 0) {
    PHX_CHECK(out.input_index[x] == marker);
    PHX_CHECK(job_vertices(out, 0).count(x) == 1 && job_vertices(out, 1).count(x) == 1);
  }
  // Distinct supports ({face, face, first} and {face, face, second}) weld.
  PHX_CHECK(out.counters[6] >= 1);
  const auto n0 = vertex_at(out, input.points[static_cast<std::size_t>(q0)]);
  const auto s0 = vertex_at(out, input.points[static_cast<std::size_t>(r0)]);
  PHX_CHECK(n0 >= 0 && s0 >= 0 && n0 != s0);
  if (n0 >= 0 && s0 >= 0) PHX_CHECK(out.input_index[n0] == q0 && out.input_index[s0] == r0);
  const auto origin = vertex_at(out, input.points[0]);
  PHX_CHECK(origin >= 0 && job_vertices(out, 4).count(origin) == 1);
  if (origin >= 0) PHX_CHECK(out.input_index[origin] == 0);
}

void test_shared_faces_and_edges() {
  ArrangementInput input = unit_tetrahedron();
  const auto top = input.point(1, 1, 1);
  const auto second = input.tetrahedron({1, 2, 3, top});
  const auto shared = input.tetrahedron_planes[0];  // x + y + z = 1, outward of tetrahedron 0
  const auto cut = input.halfspace({0, 0, 3, -1});  // z <= 1/3
  input.job(from_tetrahedron(0), {cut});
  input.job(from_tetrahedron(second), {cut});
  input.job(from_tetrahedron(0), {-cut});
  input.job(from_tetrahedron(second), {-cut});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 4);
  if (volumes.size() != 4) return;
  PHX_CHECK(same_ratio(volumes[0] + volumes[2], ratio(1)));
  PHX_CHECK(same_ratio(volumes[1] + volumes[3], ratio(2)));
  for (const auto& [low, high] : {std::pair{0, 1}, std::pair{2, 3}}) {
    const auto own = face_with_plane(out, low, shared), other = face_with_plane(out, high, -shared);
    PHX_CHECK(own >= 0 && other >= 0);
    if (own >= 0 && other >= 0) PHX_CHECK(reversed_loops(job_loop(out, own), job_loop(out, other)));
  }
  // The caps of both lower pieces share exactly their welded section edge.
  const auto cap0 = face_with_plane(out, 0, cut), cap1 = face_with_plane(out, 1, cut);
  PHX_CHECK(cap0 >= 0 && cap1 >= 0);
  if (cap0 >= 0 && cap1 >= 0) {
    const auto a = job_loop(out, cap0), b = job_loop(out, cap1);
    int common = 0;
    for (const auto vertex : a) common += std::count(b.begin(), b.end(), vertex) > 0 ? 1 : 0;
    PHX_CHECK(common == 2);
  }
}

ArrangementInput high_bit_input(const std::vector<Big>& multipliers) {
  ArrangementInput input;
  const Big d = power(3, 45);  // ~2^71: every coordinate and plane is multi-limb
  const Big a = power(3, 44) + 5, b = power(3, 43) + 7, c = power(3, 42) + 11;
  const auto p0 = input.point(a, b, c, d);
  const auto p1 = input.point(a + d, b, c, d);
  const auto p2 = input.point(a, b + d, c, d);
  const auto p3 = input.point(a, b, c + d, d);
  input.tetrahedron({p0, p1, p2, p3});
  const auto x = input.halfspace({Big(2) * d, 0, 0, -(Big(2) * a + d)});  // x <= a/d + 1/2
  const auto y = input.halfspace({0, Big(3) * d, 0, -(Big(3) * b + d)});  // y <= b/d + 1/3
  const auto j0 = input.job(from_tetrahedron(0), {x});
  input.job(from_tetrahedron(0), {-x});
  input.job(j0, {y});
  input.job(j0, {-y});
  for (std::size_t row = 0; row < input.planes.size() && row < multipliers.size(); ++row)
    for (auto& value : input.planes[row]) value = value * multipliers[row];
  return input;
}

void test_high_bit_rationals() {
  const ArrangementInput reduced = high_bit_input({});
  const auto out = run_arrangement(reduced);
  const auto volumes = check_arrangement(reduced, out);
  PHX_CHECK(volumes.size() == 4);
  if (volumes.size() != 4) return;
  PHX_CHECK(same_ratio(volumes[1], ratio(1, 8)) && same_ratio(volumes[0], ratio(7, 8)));
  PHX_CHECK(same_ratio(volumes[2] + volumes[3], volumes[0]));
  std::int64_t widest = 0;
  for (std::size_t scalar = 0; scalar + 1 < out.word_offsets.size(); ++scalar)
    widest = std::max(widest, out.word_offsets[scalar + 1] - out.word_offsets[scalar]);
  PHX_CHECK(widest >= 3);
  // Positive odd multi-limb multipliers keep every plane and point exact.
  std::vector<Big> multipliers;
  for (int row = 0; row < 8; ++row) multipliers.push_back(power(3, 40) + 2 * row);
  const ArrangementInput scaled = high_bit_input(multipliers);
  const auto scaled_out = run_arrangement(scaled);
  check_arrangement(scaled, scaled_out);
  PHX_CHECK(scaled_out.cell_face_offsets == out.cell_face_offsets &&
            scaled_out.face_offsets == out.face_offsets && scaled_out.face_vertices == out.face_vertices &&
            scaled_out.face_planes == out.face_planes && scaled_out.input_index == out.input_index);
  PHX_CHECK(scaled_out.vertices.size() == out.vertices.size());
  for (std::size_t vertex = 0; vertex < std::min(out.vertices.size(), scaled_out.vertices.size()); ++vertex)
    PHX_CHECK(same_point(out.vertices[vertex], scaled_out.vertices[vertex]));
}

void test_nonsimple_support_and_cut_order() {
  ArrangementInput input = unit_tetrahedron(4);
  const auto x = input.halfspace({1, 0, 0, -1});
  const auto y = input.halfspace({0, 1, 0, -1});
  const auto sum = input.halfspace({1, 1, 0, -2});  // contains the line x = y = 1
  const auto j0 = input.job(from_tetrahedron(0), {x});
  const auto j1 = input.job(j0, {y});
  input.job(j1, {sum});
  input.job(from_tetrahedron(0), {sum, x, y});
  input.job(from_tetrahedron(0), {y, sum, x});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 5);
  if (volumes.size() != 5) return;
  // Every cut order reaches the same welded vertices and exact volume; the
  // redundant x + y <= 2 meets the cell only in its four-plane edge.
  for (std::int32_t job = 2; job < 5; ++job) {
    PHX_CHECK(same_ratio(volumes[job], volumes[1]));
    PHX_CHECK(job_vertices(out, job) == job_vertices(out, 1));
    std::multiset<std::int32_t> planes(out.face_planes.begin() + out.cell_face_offsets[job],
                                       out.face_planes.begin() + out.cell_face_offsets[job + 1]);
    std::multiset<std::int32_t> expected(out.face_planes.begin() + out.cell_face_offsets[1],
                                         out.face_planes.begin() + out.cell_face_offsets[2]);
    PHX_CHECK(planes == expected);
  }
  const Point corner{1, 1, 0, 1};
  const auto nonsimple = vertex_at(out, corner);
  PHX_CHECK(nonsimple >= 0 && job_vertices(out, 1).count(nonsimple) == 1);
}

void expect_invalid(const ArrangementInput& input, std::int64_t category, std::int64_t row,
                    const PackHook& pack_hook = {}) {
  const auto out = run_arrangement(input, std::int64_t{1} << 28, std::int64_t{1} << 40, pack_hook);
  PHX_CHECK(out.status == PHX_MC_INVALID_INPUT && !out.published);
  PHX_CHECK(out.evidence[1] == 0 && out.evidence[3] == 0);
  PHX_CHECK(out.evidence[4] == category && out.evidence[5] == row);
}

void test_chain_limit_and_refusals() {
  // Twelve cuts along the chain reach the sixteen-halfspace cell capacity.
  ArrangementInput chain = unit_tetrahedron();
  std::vector<std::int32_t> slabs;
  for (int step = 1; step <= 13; ++step) slabs.push_back(chain.halfspace({2 * step, 0, 0, -step - 1}));
  const auto j0 = chain.job(from_tetrahedron(0), {slabs[0], slabs[1], slabs[2], slabs[3], slabs[4], slabs[5]});
  chain.job(j0, {slabs[6], slabs[7], slabs[8], slabs[9], slabs[10], slabs[11]});
  check_arrangement(chain, run_arrangement(chain));
  ArrangementInput over = chain;
  over.job(1, {slabs[12]});
  expect_invalid(over, 4, 2);

  const ArrangementInput base = partition_dag();
  ArrangementInput forward = base;
  forward.parents[1] = 1;
  expect_invalid(forward, 4, 1);
  ArrangementInput missing = base;
  missing.parents[0] = from_tetrahedron(1);
  expect_invalid(missing, 4, 0);
  ArrangementInput zero_cut = base;
  zero_cut.cut_planes[2] = 0;
  expect_invalid(zero_cut, 5, 2);
  ArrangementInput far_cut = base;
  far_cut.cut_planes[0] = static_cast<std::int32_t>(base.planes.size()) + 1;
  expect_invalid(far_cut, 5, 0);
  // Consistent faces around a negatively ordered tetrahedron.
  ArrangementInput negative = base;
  std::swap(negative.tetrahedra[1], negative.tetrahedra[2]);
  std::swap(negative.tetrahedron_planes[1], negative.tetrahedron_planes[2]);
  expect_invalid(negative, 3, 0);
  ArrangementInput inward = base;
  inward.tetrahedron_planes[3] = -inward.tetrahedron_planes[3];
  expect_invalid(inward, 3, 0);
  ArrangementInput foreign = base;
  foreign.tetrahedron_planes[0] = base.cut_planes[0];
  expect_invalid(foreign, 3, 0);
  ArrangementInput repeated = base;
  repeated.tetrahedra[3] = repeated.tetrahedra[0];
  expect_invalid(repeated, 3, 0);
  ArrangementInput leading = base;
  for (auto& value : leading.planes[4]) value = -value;
  expect_invalid(leading, 2, 4);
  ArrangementInput flat = base;
  flat.planes[4] = {0, 0, 0, 1};
  expect_invalid(flat, 2, 4);
  // (0, 4, 0) written with a negative homogeneous weight.
  ArrangementInput behind = base;
  behind.points[2] = {0, -4, 0, -1};
  expect_invalid(behind, 1, 2);
  // Malformed limb encodings: a zero top limb and a signed empty scalar.
  expect_invalid(base, 1, base.points.size() - 1, [](PackedRows& points, PackedRows&) {
    points.words.push_back(0);
    ++points.offsets.back();
  });
  const PackedRows packed_planes = pack_rows(base.planes);
  const auto empty_scalar = static_cast<std::int64_t>(
      std::find(packed_planes.signs.begin(), packed_planes.signs.end(), 0) -
      packed_planes.signs.begin());
  PHX_CHECK(empty_scalar < static_cast<std::int64_t>(packed_planes.signs.size()));
  expect_invalid(base, 2, empty_scalar / 4, [&](PackedRows&, PackedRows& planes) {
    planes.signs[static_cast<std::size_t>(empty_scalar)] = 1;
  });
  // Count/CSR disagreement and missing outputs are argument errors.
  const auto mismatch = run_arrangement(base, std::int64_t{1} << 28, std::int64_t{1} << 40, {},
                                        [](phx_mc_arrange_tetrahedra_exact_request& request) {
                                          --request.cut_count;
                                        });
  PHX_CHECK(mismatch.status == PHX_MC_INVALID_ARGUMENT && !mismatch.published);
  const auto no_work = run_arrangement(base, std::int64_t{1} << 28, 0);
  PHX_CHECK(no_work.status == PHX_MC_INVALID_ARGUMENT && !no_work.published);
  PHX_CHECK(phx_mc_arrange_tetrahedra_exact(nullptr) == PHX_MC_INVALID_ARGUMENT);
  std::int64_t sizes[PHX_MC_TETRAHEDRA_ARRANGEMENT_SIZES];
  std::int64_t counters[PHX_MC_TETRAHEDRA_ARRANGEMENT_COUNTERS];
  PHX_CHECK(phx_mc_tetrahedra_arrangement_sizes(nullptr, sizes) == PHX_MC_INVALID_ARGUMENT);
  PHX_CHECK(phx_mc_tetrahedra_arrangement_export(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 nullptr, nullptr, nullptr, nullptr,
                                                 counters) == PHX_MC_INVALID_ARGUMENT);
  phx_mc_tetrahedra_arrangement_free(nullptr);
}

void check_resource_boundaries(const ArrangementInput& input) {
  const auto out = run_arrangement(input);
  check_arrangement(input, out);
  const std::int64_t work = out.evidence[0], scratch = out.evidence[2];
  PHX_CHECK(work > 0 && scratch > 0);
  // Work: success at W with identical output, refusal at W - 1 publishes nothing.
  const auto at_work = run_arrangement(input, std::int64_t{1} << 28, work);
  PHX_CHECK(at_work.same(out) && at_work.evidence[0] == work);
  const auto below_work = run_arrangement(input, std::int64_t{1} << 28, work - 1);
  PHX_CHECK(below_work.status == PHX_MC_CAPACITY_EXCEEDED && !below_work.published);
  PHX_CHECK(below_work.evidence[0] <= work - 1 && below_work.evidence[4] != 0);
  PHX_CHECK(below_work.evidence[3] <= static_cast<std::int64_t>(input.parents.size()));
  // Scratch: the observed peak admits exactly; one byte less is refused.
  const auto at_scratch = run_arrangement(input, scratch);
  PHX_CHECK(at_scratch.same(out) && at_scratch.evidence[2] == scratch);
  const auto below_scratch = run_arrangement(input, scratch - 1);
  PHX_CHECK(below_scratch.status == PHX_MC_CAPACITY_EXCEEDED && !below_scratch.published);
  PHX_CHECK(below_scratch.evidence[2] <= scratch - 1 && below_scratch.evidence[4] != 0);
}

// Tetrahedron 0 is P, the corner tetrahedron of size 4. Tetrahedron 1
// crosses only P's slanted face: one facet of P splits it while three of its
// faces split P. Tetrahedron 2 is P shifted by x = 1, sharing P's y = 0 and
// z = 0 faces: exactly one face splits the other in each direction. Jobs 0-2
// meet them through declared face sets.
ArrangementInput declared_choice_input() {
  ArrangementInput input = unit_tetrahedron(4);
  const auto s0 = input.point(1, 1, 1), s1 = input.point(3, 1, 1);
  const auto s2 = input.point(1, 3, 1), s3 = input.point(1, 1, 3);
  const auto poking = input.tetrahedron({s0, s1, s2, s3});
  const auto t0 = input.point(1, 0, 0), t1 = input.point(5, 0, 0);
  const auto t2 = input.point(1, 4, 0), t3 = input.point(1, 0, 4);
  const auto shifted = input.tetrahedron({t0, t1, t2, t3});
  meet_job(input, from_tetrahedron(0), poking);
  meet_job(input, from_tetrahedron(poking), 0);
  meet_job(input, from_tetrahedron(0), shifted);
  return input;
}

void test_resource_boundaries() {
  check_resource_boundaries(partition_dag());
  check_resource_boundaries(declared_choice_input());
}

void test_sparse_exact_arithmetic() {
  ArrangementInput unit = unit_tetrahedron(4);
  const auto unit_outside = unit.halfspace({1, 0, 0, 1});
  unit.job(from_tetrahedron(0), {unit_outside});
  const auto unit_result = run_arrangement(unit);
  const auto unit_volumes = check_arrangement(unit, unit_result);
  PHX_CHECK(unit_volumes.size() == 1 && unit_volumes[0].numerator.sign() == 0);

  ArrangementInput scaled = unit_tetrahedron(4);
  const auto scaled_outside = scaled.halfspace({2, 0, 0, 2});
  scaled.job(from_tetrahedron(0), {scaled_outside});
  const auto scaled_result = run_arrangement(scaled);
  const auto scaled_volumes = check_arrangement(scaled, scaled_result);
  PHX_CHECK(scaled_volumes.size() == 1 && scaled_volumes[0].numerator.sign() == 0);
  // Identical separated geometry and input sizes; the scaled row executes
  // only the nonunit x-product, while its positive-unit W factor is skipped.
  PHX_CHECK(scaled_result.evidence[0] == unit_result.evidence[0] + 1);
  check_resource_boundaries(unit);

  ArrangementInput mixed = unit_tetrahedron(4);
  const auto axis = mixed.halfspace({1, 0, 0, -1});
  const auto signed_plane = mixed.halfspace({1, -1, 2, 0});
  const auto dense = mixed.halfspace({2, 3, 5, -4});
  const auto axis_job = mixed.job(from_tetrahedron(0), {axis});
  const auto repeated_axis_job = mixed.job(from_tetrahedron(0), {axis});
  mixed.job(from_tetrahedron(0), {signed_plane});
  mixed.job(axis_job, {signed_plane});
  mixed.job(from_tetrahedron(0), {dense});
  mixed.job(from_tetrahedron(0), {axis, signed_plane, dense});
  const auto mixed_result = run_arrangement(mixed);
  const auto mixed_volumes = check_arrangement(mixed, mixed_result);
  PHX_CHECK(mixed_volumes.size() == 6);
  PHX_CHECK(same_job(mixed_result, axis_job, repeated_axis_job));
  PHX_CHECK(mixed_result.counters[3] > 0 && mixed_result.counters[5] > 0);
  check_resource_boundaries(mixed);
}

void test_declared_separation_and_contacts() {
  // A needle whose apex touches the centroid of P's slanted face and whose
  // base lies beyond it: every needle face has a corner of P strictly
  // inside, so only P's slanted face at the needle's corners separates.
  ArrangementInput input = unit_tetrahedron();
  const auto a = input.point(1, 1, 0), b = input.point(1, 0, 1), c = input.point(0, 1, 1);
  const auto apex = input.point(1, 1, 1, 3);
  const auto needle = input.tetrahedron({a, b, c, apex});
  meet_job(input, from_tetrahedron(0), needle);
  const auto alone = run_arrangement(input);
  const auto alone_volumes = check_arrangement(input, alone);
  PHX_CHECK(alone_volumes.size() == 1);
  if (alone_volumes.size() == 1) PHX_CHECK(alone_volumes[0].numerator.sign() == 0);
  PHX_CHECK(alone.counters[1] == 0 && alone.counters[4] == 0);
  // The same cuts as an arbitrary job section P before reaching no measure.
  const auto far = input.halfspace({1, 0, 0, -100});
  meet_job(input, from_tetrahedron(0), needle, {far});
  // Swapped roles: P's faces at the needle's corners separate first.
  meet_job(input, from_tetrahedron(needle), 0);
  // Opposite coincident faces and an edge contact, in both roles.
  const auto top = input.point(1, 1, 1), below = input.point(1, 1, -1);
  const auto glued = input.tetrahedron({1, 2, 3, top});
  const auto wedge = input.tetrahedron({1, 2, top, below});
  meet_job(input, from_tetrahedron(0), glued);
  meet_job(input, from_tetrahedron(glued), 0);
  meet_job(input, from_tetrahedron(0), wedge);
  meet_job(input, from_tetrahedron(wedge), 0);
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 7);
  for (const auto& volume : volumes) PHX_CHECK(volume.numerator.sign() == 0);
  PHX_CHECK(out.counters[1] > 0);
}

void test_declared_containment_and_equal_operands() {
  ArrangementInput input = unit_tetrahedron(4);
  const auto q0 = input.point(1, 1, 1), q1 = input.point(2, 1, 1);
  const auto q2 = input.point(1, 2, 1), q3 = input.point(1, 1, 2);
  // Inside P, with its slanted face on P's slanted face (one signed plane).
  const auto inner = input.tetrahedron({q0, q1, q2, q3});
  const auto rotated = input.tetrahedron({q1, q2, q0, q3});
  const auto b_whole = meet_job(input, from_tetrahedron(0), inner);
  const auto p_whole = meet_job(input, from_tetrahedron(inner), 0);
  // Equal operands publish the first declared tetrahedron with that face set.
  const auto equal = meet_job(input, from_tetrahedron(rotated), inner);
  const auto self = meet_job(input, from_tetrahedron(inner), inner);
  const auto* faces = &input.tetrahedron_planes[static_cast<std::size_t>(4 * inner)];
  const std::int32_t f0 = faces[0], f1 = faces[1], f2 = faces[2], f3 = faces[3];
  const auto permuted = input.job(from_tetrahedron(0), {f2, f3, f0, f1});
  const auto alone = run_arrangement(input);
  check_arrangement(input, alone);
  PHX_CHECK(alone.counters[1] == 0 && alone.counters[4] == 0);
  const auto inner_copy = input.job(from_tetrahedron(inner), {});
  const auto rotated_copy = input.job(from_tetrahedron(rotated), {});
  // A flipped sign or a missing face is an arbitrary job.
  const auto flipped = input.job(from_tetrahedron(0), {f0, f1, f2, -f3});
  const auto missing = input.job(from_tetrahedron(0), {f0, f1, f2});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 9);
  if (volumes.size() != 9) return;
  PHX_CHECK(same_ratio(volumes[b_whole], ratio(1)));
  for (const auto job : {b_whole, p_whole, equal, self, permuted})
    PHX_CHECK(same_job(out, job, inner_copy));
  PHX_CHECK(!same_job(out, rotated_copy, inner_copy));
  // Without z >= 1 the region is x, y >= 1 inside P: six-volume 2^3.
  PHX_CHECK(volumes[flipped].numerator.sign() > 0 && same_ratio(volumes[missing], ratio(8)));
  PHX_CHECK(run_arrangement(input).same(out));
}

void test_declared_operand_choice() {
  ArrangementInput input = declared_choice_input();
  const auto alone = run_arrangement(input);
  check_arrangement(input, alone);
  // One section each: tetrahedron 1 cut by P's slanted face in both roles,
  // and P cut by x >= 1 on the tie.
  PHX_CHECK(alone.counters[1] == 3);
  const auto far = input.halfspace({1, 0, 0, -100});
  const auto small_cut = meet_job(input, from_tetrahedron(1), 0, {far});
  const auto big_cut = meet_job(input, from_tetrahedron(0), 1, {far});
  const auto tie_cut = meet_job(input, from_tetrahedron(0), 2, {far});
  const auto shifted_cut = meet_job(input, from_tetrahedron(2), 0, {far});
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 7);
  if (volumes.size() != 7) return;
  PHX_CHECK(same_job(out, 0, small_cut) && same_job(out, 1, small_cut));
  PHX_CHECK(same_ratio(volumes[0], ratio(1)) && same_ratio(volumes[big_cut], ratio(1)));
  // The tie clips P exactly as the arbitrary path does, not the shifted copy.
  PHX_CHECK(same_job(out, 2, tie_cut) && !same_job(out, 2, shifted_cut));
  PHX_CHECK(same_ratio(volumes[2], ratio(27)) && same_ratio(volumes[shifted_cut], ratio(27)));
  // The arbitrary companions need 1 + 3 + 1 + 1 sections.
  PHX_CHECK(out.counters[1] == 3 + 6);
}

// A job parent with twelve inherited halfspaces: the corner tetrahedron of
// size 8 cut to 1 <= x, y, z <= 4, x + y >= 3 (x + y + z <= 9 is redundant).
// Tetrahedron 1 has corners (2,2,2), (5,2,2), (2,5,2), (2,2,7/2) and faces
// x, y, z >= 2 and x + y + 2z <= 11.
ArrangementInput declared_job_parent_input(bool extra_cut) {
  ArrangementInput input = unit_tetrahedron(8);
  const auto x_low = input.halfspace({-1, 0, 0, 1}), y_low = input.halfspace({0, -1, 0, 1});
  const auto z_low = input.halfspace({0, 0, -1, 1}), x_high = input.halfspace({1, 0, 0, -4});
  const auto y_high = input.halfspace({0, 1, 0, -4}), z_high = input.halfspace({0, 0, 1, -4});
  const auto pair = input.halfspace({-1, -1, 0, 3}), sum = input.halfspace({1, 1, 1, -9});
  const auto parent =
      extra_cut ? input.job(from_tetrahedron(0), {x_low, y_low, z_low, x_high, y_high, z_high,
                                                  pair, sum, input.halfspace({1, 1, 1, -10})})
                : input.job(from_tetrahedron(0),
                            {x_low, y_low, z_low, x_high, y_high, z_high, pair, sum});
  const auto b0 = input.point(2, 2, 2), b1 = input.point(5, 2, 2);
  const auto b2 = input.point(2, 5, 2), b3 = input.point(4, 4, 7, 2);
  meet_job(input, parent, input.tetrahedron({b0, b1, b2, b3}));
  return input;
}

void test_declared_job_parent_and_chain_limit() {
  const ArrangementInput input = declared_job_parent_input(false);
  const auto out = run_arrangement(input);
  const auto volumes = check_arrangement(input, out);
  PHX_CHECK(volumes.size() == 2);
  if (volumes.size() == 2) PHX_CHECK(volumes[1].numerator.sign() > 0);
  // Seven sections build the parent. Its facets s <= 8, x <= 4 and y <= 4
  // split tetrahedron 1 while all four of its faces split the parent, so
  // tetrahedron 1 is clipped; after s <= 8 the other two are redundant.
  PHX_CHECK(out.counters[1] == 7 + 1);
  // Twelve inherited halfspaces plus four declared faces fill the cell; a
  // thirteenth inherited halfspace is refused before execution.
  expect_invalid(declared_job_parent_input(true), 4, 1);
}

void test_declared_invalid_unused_inputs() {
  const ArrangementInput base = declared_choice_input();
  // An unused duplicate of P with a negative corner order.
  ArrangementInput inverted = base;
  const auto duplicate = inverted.tetrahedron({0, 1, 2, 3});
  const auto row = static_cast<std::size_t>(4 * duplicate);
  std::swap(inverted.tetrahedra[row + 1], inverted.tetrahedra[row + 2]);
  std::swap(inverted.tetrahedron_planes[row + 1], inverted.tetrahedron_planes[row + 2]);
  expect_invalid(inverted, 3, duplicate);
  ArrangementInput leading = base;
  leading.planes.push_back({-1, 0, 0, 7});
  expect_invalid(leading, 2, static_cast<std::int64_t>(leading.planes.size()) - 1);
  ArrangementInput behind = base;
  behind.points.push_back({1, 1, 1, -1});
  expect_invalid(behind, 1, static_cast<std::int64_t>(behind.points.size()) - 1);
}

}  // namespace

int main() {
  test_identity_and_corner_permutation();
  test_dag_partition_and_pool_reuse();
  test_contact_empty_sliver_and_vertex_sections();
  test_welding_and_rounding_buckets();
  test_shared_faces_and_edges();
  test_high_bit_rationals();
  test_nonsimple_support_and_cut_order();
  test_chain_limit_and_refusals();
  test_resource_boundaries();
  test_sparse_exact_arithmetic();
  test_declared_separation_and_contacts();
  test_declared_containment_and_equal_operands();
  test_declared_operand_choice();
  test_declared_job_parent_and_chain_limit();
  test_declared_invalid_unused_inputs();
  return phx::mc::test::finish("test_tetrahedra_arrangement");
}
