// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "floating_point_mode.hpp"
#include "intersections.hpp"
#include "predicates.hpp"
#include "verify_surface_embedding.hpp"

namespace {
struct Mesh {
  std::vector<double> points;
  std::vector<int64_t> triangles;
};

Mesh tetrahedron() {
  return {{0,0,0, 1,0,0, 0,1,0, 0,0,1},
          {1,2,3, 0,3,2, 0,1,3, 0,2,1}};
}

std::vector<phx_mc_surface_finding> verify(const Mesh& mesh, int32_t expected) {
  std::vector<phx_mc_surface_finding> findings(256);
  phx_mc_surface_embedding_verdict verdict{};
  PHX_CHECK(phx_mc_verify_surface_embedding(mesh.points.size()/3, mesh.points.data(),
      mesh.triangles.size()/3, mesh.triangles.data(), findings.size(), findings.data(),
      &verdict) == PHX_MC_OK);
  PHX_CHECK(verdict.status == expected);
  findings.resize(static_cast<size_t>(verdict.finding_count));
  return findings;
}

bool has(const std::vector<phx_mc_surface_finding>& findings, int kind) {
  return std::any_of(findings.begin(), findings.end(),
                     [kind](const auto& f) { return f.kind == kind; });
}

void check_closed_and_edges() {
  Mesh mesh = tetrahedron();
  PHX_CHECK(verify(mesh, PHX_MC_VERDICT_CERTIFIED).empty());
  mesh.triangles.resize(9);
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_EDGE));
  mesh = tetrahedron();
  std::swap(mesh.triangles[0], mesh.triangles[1]);
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_EDGE));
}

void check_vertex_link() {
  Mesh mesh = tetrahedron();
  // Two closed tetrahedral shells touching at exactly one shared vertex.
  mesh.points.insert(mesh.points.end(), {-1,0,0, 0,-1,0, 0,0,-1});
  mesh.triangles.insert(mesh.triangles.end(), {4,6,5, 0,5,6, 0,6,4, 0,4,5});
  const auto findings = verify(mesh, PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(has(findings, PHX_MC_SURFACE_VERTEX_LINK));
}

void check_contacts() {
  Mesh mesh = tetrahedron();
  // An independently indexed tetrahedron intersects the first shell.
  mesh.points.insert(mesh.points.end(), {0.25,0.25,-0.5, 0.75,0.25,-0.5,
                                        0.25,0.75,-0.5, 0.25,0.25,0.5});
  mesh.triangles.insert(mesh.triangles.end(), {5,6,7, 4,7,6, 4,5,7, 4,6,5});
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_CONTACT));
  mesh = tetrahedron();
  // Equal coordinates with distinct identities are not combinatorial adjacency.
  mesh.points.insert(mesh.points.end(), {0,0,0, -1,0,0, 0,-1,0, 0,0,-1});
  mesh.triangles.insert(mesh.triangles.end(), {5,7,6, 4,6,7, 4,7,5, 4,5,6});
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_CONTACT));
}

void check_adjacent_contacts() {
  // Each pair is deliberately open; inspect contact evidence independently of
  // the expected boundary-edge findings.
  Mesh mesh{{0,0,0, 2,0,0, 0,2,0, 0.5,0.5,-1, 0.5,0.5,1},
            {0,1,2, 0,3,4}};
  auto findings = verify(mesh, PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(has(findings, PHX_MC_SURFACE_CONTACT));  // shared-vertex piercing
  // Reverse the pair: the first triangle's opposite edge now pierces.
  mesh.triangles = {0,3,4, 0,1,2};
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_CONTACT));

  // One opposite vertex lies in the first triangle's plane; this must use
  // the full exact classifier, and the two triangles overlap along a segment.
  mesh = {{0,0,0, 2,0,0, 0,2,0, 1,0,0, 0,0,1}, {0,1,2, 0,3,4}};
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_CONTACT));

  // Both planes cut the opposite triangle, but their cut segments miss.
  mesh = {{0,0,0, 1,0,0, 0,1,0, -1,0,1, 0,-1,-1}, {0,1,2, 0,3,4}};
  PHX_CHECK(!has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_CONTACT));

  mesh = {{0,0,0, 2,0,0, 0,2,0, 1,1,0}, {0,1,2, 0,1,3}};
  findings = verify(mesh, PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(has(findings, PHX_MC_SURFACE_CONTACT));  // coplanar shared-edge fold

  mesh = {{0,0,0, 2,0,0, 0,2,0, 1,0,0, 0,1,0}, {0,1,2, 0,3,4}};
  findings = verify(mesh, PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(has(findings, PHX_MC_SURFACE_CONTACT));  // coplanar shared-vertex overlap

  mesh = {{0,0,0, 2,0,0, 0,2,0, 0,0,1}, {0,1,2, 0,1,3}};
  findings = verify(mesh, PHX_MC_VERDICT_VIOLATED);
  PHX_CHECK(!has(findings, PHX_MC_SURFACE_CONTACT));  // noncoplanar shared edge
}

bool triangle_valid(const double* const triangle[3]) {
  for (int d = 0; d < 3; ++d) {
    const int u = (d+1)%3, v = (d+2)%3;
    const double a[2] = {triangle[0][u], triangle[0][v]};
    const double b[2] = {triangle[1][u], triangle[1][v]};
    const double c[2] = {triangle[2][u], triangle[2][v]};
    if (phx::mc::orient2d_filtered(a, b, c) != 0) return true;
  }
  return false;
}

void check_random_adjacent_pairs() {
  uint64_t state = 0x159d8ef1a42bc473ULL;
  const auto coordinate = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(static_cast<int>(state % 7) - 3);
  };
  for (int shared_edge = 0; shared_edge < 2; ++shared_edge) {
    int tested = 0, zeros = 0, cleared = 0, failures = 0;
    while (tested < 100000) {
      std::array<std::array<double, 3>, 5> points{};
      for (int i = 1; i < 5; ++i)
        for (int d = 0; d < 3; ++d) points[i][d] = coordinate();
      const double* p[3] = {points[0].data(), points[1].data(), points[2].data()};
      const double* q[3] = {points[0].data(),
                            points[shared_edge ? 1 : 3].data(), points[4].data()};
      int64_t first_ids[3] = {0, 1, 2};
      int64_t second_ids[3] = {0, shared_edge ? 1 : 3, 4};
      const int p_shift = tested % 3, q_shift = (tested / 3) % 3;
      std::rotate(p, p + p_shift, p + 3);
      std::rotate(first_ids, first_ids + p_shift, first_ids + 3);
      std::rotate(q, q + q_shift, q + 3);
      std::rotate(second_ids, second_ids + q_shift, second_ids + 3);
      if (!triangle_valid(p) || !triangle_valid(q)) continue;
      const bool clear = phx::mc::surface_embedding_pair_clear(p, q, first_ids, second_ids);
      phx::mc::Contact contact;
      PHX_CHECK(phx::mc::intersect_triangles(p, q, first_ids, second_ids, contact) == PHX_MC_OK);
      const bool violation = contact.kind != PHX_MC_DISJOINT &&
                             contact.kind != PHX_MC_SHARED_VERTEX &&
                             contact.kind != PHX_MC_SHARED_EDGE;
      const bool reduced_violation = clear ? false : violation;
      PHX_CHECK(reduced_violation == violation);
      cleared += clear;
      failures += violation;
      if (!shared_edge) {
        const int s0 = phx::mc::orient3d_filtered(p[0], p[1], p[2], points[3].data());
        const int s1 = phx::mc::orient3d_filtered(p[0], p[1], p[2], points[4].data());
        if (s0 == 0 && s1 == 0) ++zeros;  // Coplanar: decided by exact 2D separating lines.
        else if (s0 == 0 || s1 == 0) { ++zeros; PHX_CHECK(!clear); }
        else if (s0 != s1) {
          const int t0 = phx::mc::orient3d_filtered(q[0], q[1], q[2], points[1].data());
          const int t1 = phx::mc::orient3d_filtered(q[0], q[1], q[2], points[2].data());
          if (t0 == 0 || t1 == 0) { ++zeros; PHX_CHECK(!clear); }
          else if (t0 != t1) {
            for (int edge = 0; edge < 3; ++edge) {
              const int x = phx::mc::orient3d_filtered(points[1].data(), points[2].data(),
                                                       q[edge], q[(edge+1)%3]);
              const int y = phx::mc::orient3d_filtered(points[3].data(), points[4].data(),
                                                       p[edge], p[(edge+1)%3]);
              if (x == 0 || y == 0) { ++zeros; PHX_CHECK(!clear); }
            }
          }
        }
      } else if (phx::mc::orient3d_filtered(p[0], p[1], p[2], points[4].data()) == 0) {
        ++zeros;  // Coplanar shared edges are decided by exact 2D sides.
      }
      ++tested;
    }
    PHX_CHECK(zeros > 0);
    PHX_CHECK(cleared > 0 && failures > 0);
  }
}

// Coplanar pairs (the lifted chart partition case) sharing no vertex, one vertex or one
// edge, in the plane z = 0 (axis-aligned fast path) and in the tilted plane z = x + y
// (general coplanar path): the reduced 2D separation must never clear a pair the full
// classifier flags, and must clear many disjoint ones.
void check_random_coplanar_pairs() {
  uint64_t state = 0x9e3779b97f4a7c15ULL;
  const auto coordinate = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(static_cast<int>(state % 9) - 4);
  };
  for (int tilted = 0; tilted < 2; ++tilted)
  for (int shared = 0; shared < 3; ++shared) {
    int tested = 0, cleared = 0, failures = 0;
    while (tested < 50000) {
      std::array<std::array<double, 3>, 6> points{};
      for (int i = 0; i < 6; ++i) {
        points[i][0] = coordinate();
        points[i][1] = coordinate();
        points[i][2] = tilted ? points[i][0] + points[i][1] : 0.0;
      }
      const int64_t first_ids[3] = {0, 1, 2};
      const int64_t second_ids[3] = {shared >= 1 ? 0 : 3, shared == 2 ? 1 : 4, 5};
      const double* p[3] = {points[0].data(), points[1].data(), points[2].data()};
      const double* q[3] = {points[second_ids[0]].data(), points[second_ids[1]].data(),
                            points[5].data()};
      if (!triangle_valid(p) || !triangle_valid(q)) continue;
      bool coincident = false;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          if (first_ids[i] != second_ids[j] && p[i][0] == q[j][0] && p[i][1] == q[j][1])
            coincident = true;
      if (coincident) continue;
      const bool clear = phx::mc::surface_embedding_pair_clear(p, q, first_ids, second_ids);
      phx::mc::Contact contact;
      PHX_CHECK(phx::mc::intersect_triangles(p, q, first_ids, second_ids, contact) == PHX_MC_OK);
      const bool violation = contact.kind != PHX_MC_DISJOINT &&
                             contact.kind != PHX_MC_SHARED_VERTEX &&
                             contact.kind != PHX_MC_SHARED_EDGE;
      PHX_CHECK(!(clear && violation));
      cleared += clear;
      failures += violation;
      ++tested;
    }
    PHX_CHECK(cleared > 0 && failures > 0);
  }
}

// The tree descent must visit every candidate pair and prune the rest: a soup of >= 200
// local triangles spread over a 16x16x4 integer lattice (shared vertices, coplanar and
// crossing pairs nearby, many far-apart pairs) must give exactly the all-pairs exact
// classification.
void check_random_soup() {
  uint64_t state = 0x2545f4914f6cdd1dULL;
  const auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  constexpr int64_t nx = 16, ny = 16, nz = 4;
  std::vector<double> points;
  for (int64_t z = 0; z < nz; ++z)
    for (int64_t y = 0; y < ny; ++y)
      for (int64_t x = 0; x < nx; ++x) {
        points.push_back(static_cast<double>(x));
        points.push_back(static_cast<double>(y));
        points.push_back(static_cast<double>(z));
      }
  const auto near = [&](int64_t base, int64_t extent) {
    const int64_t offset = static_cast<int64_t>(next() % 5) - 2;
    return std::clamp(base + offset, int64_t{0}, extent - 1);
  };
  std::vector<int64_t> triangles;
  while (triangles.size() < 3*240) {
    const int64_t x = static_cast<int64_t>(next() % nx), y = static_cast<int64_t>(next() % ny);
    const int64_t z = static_cast<int64_t>(next() % nz);
    int64_t ids[3];
    ids[0] = x + nx*(y + ny*z);
    for (int k = 1; k < 3; ++k) ids[k] = near(x, nx) + nx*(near(y, ny) + ny*near(z, nz));
    if (ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0]) continue;
    const double* p[3] = {&points[3*ids[0]], &points[3*ids[1]], &points[3*ids[2]]};
    if (!triangle_valid(p)) continue;
    triangles.insert(triangles.end(), ids, ids + 3);
  }
  const int64_t n = static_cast<int64_t>(triangles.size() / 3);
  const phx::mc::GradualNearestScope mode;
  PHX_CHECK(mode.established());
  std::vector<phx::mc::SurfaceContact> found;
  PHX_CHECK(phx::mc::surface_embedding_contacts(points.data(), n, triangles.data(), found) ==
            PHX_MC_OK);
  std::vector<phx::mc::SurfaceContact> expected;
  for (int64_t a = 0; a < n; ++a)
    for (int64_t b = a + 1; b < n; ++b) {
      const double* p[3];
      const double* q[3];
      for (int j = 0; j < 3; ++j) {
        p[j] = &points[3*triangles[3*a+j]];
        q[j] = &points[3*triangles[3*b+j]];
      }
      phx::mc::Contact contact;
      PHX_CHECK(phx::mc::intersect_triangles(p, q, &triangles[3*a], &triangles[3*b], contact) ==
                PHX_MC_OK);
      if (contact.kind != PHX_MC_DISJOINT && contact.kind != PHX_MC_SHARED_VERTEX &&
          contact.kind != PHX_MC_SHARED_EDGE)
        expected.push_back({a, b, contact.kind});
    }
  const auto order = [](const phx::mc::SurfaceContact& x, const phx::mc::SurfaceContact& y) {
    return x.first != y.first ? x.first < y.first : x.second < y.second;
  };
  std::sort(found.begin(), found.end(), order);
  std::sort(expected.begin(), expected.end(), order);
  PHX_CHECK(!expected.empty());
  PHX_CHECK(found.size() == expected.size());
  for (size_t i = 0; i < std::min(found.size(), expected.size()); ++i)
    PHX_CHECK(found[i].first == expected[i].first && found[i].second == expected[i].second &&
              found[i].kind == expected[i].kind);
}

void check_refusal() {
  Mesh mesh = tetrahedron();
  mesh.triangles.resize(9);
  phx_mc_surface_embedding_verdict verdict{123, 456};
  phx_mc_surface_finding sentinel{123, 456, 789, 42};
  PHX_CHECK(phx_mc_verify_surface_embedding(4, mesh.points.data(), 3, mesh.triangles.data(),
      0, &sentinel, &verdict) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(verdict.status == 123 && verdict.finding_count == 456);
  PHX_CHECK(sentinel.kind == 123 && sentinel.first == 456 && sentinel.second == 789);
  // Edge keys pack two 32-bit vertex ids; larger point sets are refused before any read.
  PHX_CHECK(phx_mc_verify_surface_embedding((int64_t{1} << 32) + 1, mesh.points.data(), 0,
      nullptr, 1, &sentinel, &verdict) == PHX_MC_INDEX_RANGE_ERROR);
  PHX_CHECK(verdict.status == 123 && sentinel.kind == 123);
  mesh.triangles[0] = 99;
  PHX_CHECK(phx_mc_verify_surface_embedding(4, mesh.points.data(), 3, mesh.triangles.data(),
      1, &sentinel, &verdict) == PHX_MC_INVALID_INPUT);
  PHX_CHECK(verdict.status == 123 && sentinel.kind == 123);
  mesh = tetrahedron();
  mesh.points[0] = 0x1p-121;
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_UNRESOLVED), PHX_MC_SURFACE_OUTSIDE_DOMAIN));
  mesh = tetrahedron();
  mesh.triangles[0] = mesh.triangles[1];
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_DEGENERATE));
  mesh.triangles.clear();
  PHX_CHECK(has(verify(mesh, PHX_MC_VERDICT_VIOLATED), PHX_MC_SURFACE_EMPTY));
}
}  // namespace

int main() {
  check_closed_and_edges();
  check_vertex_link();
  check_contacts();
  check_adjacent_contacts();
  check_random_adjacent_pairs();
  check_random_coplanar_pairs();
  check_random_soup();
  check_refusal();
  return phx::mc::test::finish("test_verify_surface_embedding");
}
