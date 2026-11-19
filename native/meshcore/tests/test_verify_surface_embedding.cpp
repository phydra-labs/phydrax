// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "check.hpp"
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
        if (s0 == 0 || s1 == 0) { ++zeros; PHX_CHECK(!clear); }
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

void check_refusal() {
  Mesh mesh = tetrahedron();
  mesh.triangles.resize(9);
  phx_mc_surface_embedding_verdict verdict{123, 456};
  phx_mc_surface_finding sentinel{123, 456, 789, 42};
  PHX_CHECK(phx_mc_verify_surface_embedding(4, mesh.points.data(), 3, mesh.triangles.data(),
      0, &sentinel, &verdict) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(verdict.status == 123 && verdict.finding_count == 456);
  PHX_CHECK(sentinel.kind == 123 && sentinel.first == 456 && sentinel.second == 789);
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
  check_refusal();
  return phx::mc::test::finish("test_verify_surface_embedding");
}
