//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include <array>
#include <cstdint>
#include <cstdio>

#include "check.hpp"
#include "phydrax_meshcore.h"

namespace {

void template_and_validation() {
  constexpr std::array<int64_t, 15> nodes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  constexpr std::array<int64_t, 32> expected{
      0, 4, 10, 5, 6, 11, 14, 12,
      1, 4, 11, 8, 7, 10, 14, 13,
      2, 5, 10, 7, 9, 12, 14, 13,
      3, 6, 12, 9, 8, 11, 14, 13};
  std::array<int64_t, 32> hexes{};
  PHX_CHECK(phx_mc_tetrahedron_dual_hexes(1, 15, nodes.data(), 4, hexes.data()) == PHX_MC_OK);
  PHX_CHECK(hexes == expected);

  hexes.fill(-77);
  PHX_CHECK(phx_mc_tetrahedron_dual_hexes(1, 15, nodes.data(), 3, hexes.data()) ==
            PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(hexes[0] == -77 && hexes[31] == -77);

  auto duplicate = nodes;
  duplicate[14] = duplicate[13];
  PHX_CHECK(phx_mc_tetrahedron_dual_hexes(1, 15, duplicate.data(), 4, hexes.data()) ==
            PHX_MC_INVALID_INPUT);
  PHX_CHECK(hexes[0] == -77 && hexes[31] == -77);
  PHX_CHECK(phx_mc_tetrahedron_dual_hexes(0, 0, nullptr, 0, nullptr) == PHX_MC_OK);
}

}  // namespace

int main() {
  template_and_validation();
  return phx::mc::test::finish("test_hex_topology");
}
