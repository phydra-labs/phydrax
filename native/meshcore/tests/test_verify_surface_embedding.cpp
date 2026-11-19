// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "check.hpp"
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
  check_refusal();
  return phx::mc::test::finish("test_verify_surface_embedding");
}
