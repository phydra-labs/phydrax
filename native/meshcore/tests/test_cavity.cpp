//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Cavity staging refusal must leave the live tetrahedral state unchanged.
#include <array>
#include <cstdint>

#include "cavity.hpp"
#include "check.hpp"

namespace {

using namespace phx::mc;

void test_duplicate_removal_refuses_without_mutation() {
  TetrahedralComplex complex;
  complex.tets.push_back(Tetrahedron{{0, 1, 2, 3}, {-1, -1, -1, -1}, 7, 3});
  complex.finite_count = 1;
  const std::array<int32_t, 4> original_vertices{
      complex.tets[0].v[0], complex.tets[0].v[1], complex.tets[0].v[2], complex.tets[0].v[3]};
  const std::array<int32_t, 4> original_neighbors{
      complex.tets[0].n[0], complex.tets[0].n[1], complex.tets[0].n[2], complex.tets[0].n[3]};
  CavityEdit edit(complex);
  edit.begin();
  PHX_CHECK(edit.remove(0) == EditStatus::kOk);
  PHX_CHECK(edit.remove(0) == EditStatus::kOk);
  PHX_CHECK(edit.validate([](const int32_t*) { return true; }) == EditStatus::kDuplicate);
  edit.rollback();
  PHX_CHECK(complex.tets.size() == 1);
  PHX_CHECK(complex.live(0));
  PHX_CHECK(complex.finite_count == 1);
  PHX_CHECK(complex.tets[0].stamp == 7 && complex.tets[0].state == 3);
  for (int k = 0; k < 4; ++k) {
    PHX_CHECK(complex.tets[0].v[k] == original_vertices[static_cast<std::size_t>(k)]);
    PHX_CHECK(complex.tets[0].n[k] == original_neighbors[static_cast<std::size_t>(k)]);
  }
}

void test_nonmanifold_boundary_refuses_without_mutation() {
  TetrahedralComplex complex;
  complex.tets.push_back(Tetrahedron{{0, 1, 2, 3}, {-1, -1, -1, -1}, 11, 2});
  complex.finite_count = 1;
  CavityEdit edit(complex);
  edit.begin();
  PHX_CHECK(edit.remove(0) == EditStatus::kOk);
  PHX_CHECK(edit.add(complex.tets[0].v, nullptr, kNoRegion) == EditStatus::kOk);
  PHX_CHECK(edit.validate([](const int32_t*) { return true; }) == EditStatus::kNonManifold);
  edit.rollback();
  PHX_CHECK(complex.tets.size() == 1);
  PHX_CHECK(complex.tets[0].v[0] == 0 && complex.tets[0].v[3] == 3);
  PHX_CHECK(complex.finite_count == 1);
  PHX_CHECK(complex.free_slots.empty());
}

}  // namespace

int main() {
  test_duplicate_removal_refuses_without_mutation();
  test_nonmanifold_boundary_refuses_without_mutation();
  return phx::mc::test::finish("test_cavity");
}
