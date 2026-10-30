//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Restricted power-cell geometry, oriented boundary closure and exact source witnesses.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>
#include <map>
#include <utility>
#include <iterator>

#include "check.hpp"
#include "clip3d.hpp"
#include "expansion.hpp"
#include "filtered.hpp"
#include "phydrax_meshcore.h"
#include "predicates.hpp"

namespace {
void single_site_tetrahedron() {
  constexpr double sites[] = {0.25, 0.25, 0.25};
  constexpr double weights[] = {0.0};
  constexpr int64_t neighbor_offsets[] = {0, 0};
  constexpr int8_t split_sites[] = {0};
  constexpr double points[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  constexpr int32_t tets[] = {0, 1, 2, 3};
  constexpr int32_t regions[] = {0};
  constexpr int32_t facets[] = {0, 1, 2, 3};

  phx_mc_power_cells* handle = nullptr;
  const phx_mc_restricted_power_cells_request request{
      .site_count = 1,
      .sites = sites,
      .weights = weights,
      .neighbor_offsets = neighbor_offsets,
      .neighbors = nullptr,
      .split_sites = split_sites,
      .point_count = 4,
      .points = points,
      .tet_count = 1,
      .tets = tets,
      .tet_regions = regions,
      .tet_face_facets = facets,
      .result = &handle,
  };
  PHX_CHECK(phx_mc_restricted_power_cells(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  std::array<int64_t, 5> sizes{};
  PHX_CHECK(phx_mc_power_cells_sizes(handle, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[0] == 4);
  PHX_CHECK(sizes[1] == 4);
  PHX_CHECK(sizes[2] == 12);
  PHX_CHECK(sizes[3] == 1);
  PHX_CHECK(sizes[4] == 1);

  const int64_t witness_count = phx_mc_power_cells_site_witness_size(handle);
  PHX_CHECK(witness_count == 4);
  std::vector<double> vertices(static_cast<std::size_t>(3 * sizes[0]));
  std::vector<int64_t> face_offsets(static_cast<std::size_t>(sizes[1] + 1));
  std::vector<int32_t> face_vertices(static_cast<std::size_t>(sizes[2]));
  std::vector<int32_t> face_cells(static_cast<std::size_t>(2 * sizes[1]));
  std::vector<int32_t> face_facets(static_cast<std::size_t>(sizes[1]));
  std::vector<int32_t> cell_sites(static_cast<std::size_t>(sizes[3]));
  std::vector<int32_t> cell_regions(static_cast<std::size_t>(sizes[3]));
  std::vector<double> cell_volumes(static_cast<std::size_t>(sizes[3]));
  std::vector<double> cell_moments(static_cast<std::size_t>(3 * sizes[3]));
  std::vector<double> cell_second(static_cast<std::size_t>(sizes[3]));
  std::vector<int32_t> piece_sites(static_cast<std::size_t>(sizes[4]));
  std::vector<int32_t> piece_tets(static_cast<std::size_t>(sizes[4]));
  std::vector<int32_t> piece_cells(static_cast<std::size_t>(sizes[4]));
  std::vector<double> piece_volumes(static_cast<std::size_t>(sizes[4]));
  std::vector<double> piece_moments(static_cast<std::size_t>(3 * sizes[4]));
  PHX_CHECK(phx_mc_power_cells_export(
                handle, sizes[0], sizes[1] + 1, sizes[1], sizes[2], sizes[3], sizes[4],
                vertices.data(), face_offsets.data(), face_vertices.data(), face_cells.data(),
                face_facets.data(), cell_sites.data(), cell_regions.data(), cell_volumes.data(),
                cell_moments.data(), cell_second.data(), piece_sites.data(), piece_tets.data(),
                piece_cells.data(), piece_volumes.data(), piece_moments.data()) == PHX_MC_OK);
  PHX_CHECK(face_offsets.front() == 0);
  PHX_CHECK(face_offsets.back() == static_cast<int64_t>(face_vertices.size()));
  PHX_CHECK(cell_sites[0] == 0 && cell_regions[0] == 0);
  PHX_CHECK(piece_sites[0] == 0 && piece_tets[0] == 0 && piece_cells[0] == 0);
  PHX_CHECK(std::fabs(cell_volumes[0] - 1.0 / 6.0) < 1.0e-14);
  PHX_CHECK(std::fabs(cell_moments[0] - 1.0 / 24.0) < 1.0e-14);
  PHX_CHECK(std::fabs(cell_moments[1] - 1.0 / 24.0) < 1.0e-14);
  PHX_CHECK(std::fabs(cell_moments[2] - 1.0 / 24.0) < 1.0e-14);
  PHX_CHECK(std::fabs(piece_volumes[0] - cell_volumes[0]) < 1.0e-14);
  PHX_CHECK(std::fabs(piece_moments[0] - cell_moments[0]) < 1.0e-14);
  PHX_CHECK(std::fabs(piece_moments[1] - cell_moments[1]) < 1.0e-14);
  PHX_CHECK(std::fabs(piece_moments[2] - cell_moments[2]) < 1.0e-14);

  std::vector<int64_t> witness_offsets(static_cast<std::size_t>(sizes[0] + 1));
  std::vector<int32_t> witness_sites(static_cast<std::size_t>(witness_count));
  PHX_CHECK(phx_mc_power_cells_export_vertex_sites(
                handle, sizes[0] + 1, sizes[0], witness_count,
                witness_offsets.data(), witness_sites.data()) == PHX_MC_OK);
  PHX_CHECK(witness_offsets.front() == 0);
  PHX_CHECK(witness_offsets.back() == witness_count);
  PHX_CHECK(std::all_of(witness_sites.begin(), witness_sites.end(),
                        [](int32_t site) { return site == 0; }));
  phx_mc_power_cells_free(handle);
}
void exact_image_coordinates_keep_source_identity() {
  constexpr double original_sites[] = {0.25, 0.25, 0.25};
  constexpr double original_weights[] = {0.0};
  constexpr int32_t image_owners[] = {0};
  constexpr int64_t coordinate_offsets[] = {0, 1, 2, 3};
  constexpr double coordinate_components[] = {0.25, 0.25, 0.25};
  constexpr int64_t neighbor_offsets[] = {0, 0};
  constexpr int8_t split_sites[] = {0};
  constexpr double domain_points[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  constexpr int32_t domain_tets[] = {0, 1, 2, 3};
  constexpr int32_t domain_regions[] = {0};
  constexpr int32_t facets[] = {0, 1, 2, 3};
  phx_mc_power_cells* handle = nullptr;
  int64_t failed_coordinate = -2, failed_component = -2;
  const phx_mc_restricted_power_cells_exact_request request{
      .original_site_count = 1,
      .original_sites = original_sites,
      .original_weights = original_weights,
      .image_count = 1,
      .image_site_owners = image_owners,
      .image_coordinate_offsets = coordinate_offsets,
      .component_count = 3,
      .image_coordinate_components = coordinate_components,
      .neighbor_offsets = neighbor_offsets,
      .neighbors = nullptr,
      .split_sites = split_sites,
      .point_count = 4,
      .points = domain_points,
      .tet_count = 1,
      .tets = domain_tets,
      .tet_regions = domain_regions,
      .tet_face_facets = facets,
      .result = &handle,
      .failed_image_coordinate = &failed_coordinate,
      .failed_component = &failed_component,
  };
  PHX_CHECK(phx_mc_restricted_power_cells_exact(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  PHX_CHECK(failed_coordinate == -1 && failed_component == -1);
  std::array<int64_t, 5> sizes{};
  PHX_CHECK(phx_mc_power_cells_sizes(handle, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[3] == 1 && sizes[4] == 1);
  int64_t reason = -1, site = -1, tet = -1;
  int32_t clip_status = -1;
  PHX_CHECK(phx_mc_power_cells_failure(handle, &reason, &site, &tet, &clip_status) == PHX_MC_OK);
  PHX_CHECK(reason == PHX_MC_POWER_NO_FAILURE);
  PHX_CHECK(site == -1 && tet == -1 && clip_status == PHX_MC_OK);
  phx_mc_power_cells_free(handle);
}

struct Domain {
  std::vector<double> points;
  std::vector<int32_t> tets;
  std::vector<int32_t> regions;
  std::vector<int32_t> facets;
};

int32_t facet_on_face(const std::vector<double>& points,
                      const std::array<int32_t, 3>& face) {
  for (int axis = 0; axis < 3; ++axis) {
    const double lower = 0.0;
    const double upper = axis == 0 ? 2.0 : 1.0;
    bool on_lower = true, on_upper = true;
    for (int32_t vertex : face) {
      on_lower = on_lower && points[3 * vertex + axis] == lower;
      on_upper = on_upper && points[3 * vertex + axis] == upper;
    }
    if (on_lower) return 2 * axis;
    if (on_upper) return 2 * axis + 1;
  }
  bool interface = true;
  for (int32_t vertex : face) interface = interface && points[3 * vertex] == 1.0;
  return interface ? 6 : -1;
}

Domain unit_cube_box() {
  Domain domain;
  constexpr double points[] = {
      0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
  };
  constexpr int32_t tets[] = {
      0, 4, 6, 7, 0, 5, 4, 7, 0, 6, 2, 7,
      0, 2, 3, 7, 0, 1, 5, 7, 0, 3, 1, 7,
  };
  domain.points.assign(std::begin(points), std::end(points));
  domain.tets.assign(std::begin(tets), std::end(tets));
  domain.regions.assign(6, 0);
  for (int32_t tet = 0; tet < 6; ++tet) {
    for (int opposite = 0; opposite < 4; ++opposite) {
      bool boundary = false;
      int32_t facet = -1;
      for (int axis = 0; axis < 3; ++axis) {
        bool lower = true;
        bool upper = true;
        for (int corner = 0; corner < 4; ++corner) {
          if (corner == opposite) continue;
          const int32_t vertex = tets[4 * tet + corner];
          lower = lower && points[3 * vertex + axis] == 0.0;
          upper = upper && points[3 * vertex + axis] == 1.0;
        }
        if (lower || upper) {
          boundary = true;
          facet = 2 * axis + (upper ? 1 : 0);
          break;
        }
      }
      domain.facets.push_back(boundary ? facet : -1);
    }
  }
  return domain;
}

void exact_clipper_classifies_constructed_vertices_against_base_faces() {
  constexpr double tetrahedron[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  // These bisectors meet inside the base z=0 face, requiring a three-plane vertex.
  constexpr double sites[] = {
      0.125, 0.125, 0.125, 0.625, 0.125, 0.125, 0.125, 0.625, 0.125, 0.125, 0.125, 0.625,
  };
  constexpr double weights[] = {0, 0, 0, 0};
  constexpr int32_t neighbors[] = {1, 2, 3};
  std::array<phx::mc::Approx, 12> approximate{};
  std::array<phx::mc::Expansion, 12> exact{};
  std::array<phx::mc::ExactPowerCoordinates, 4> coordinates{};
  for (int32_t site = 0; site < 4; ++site) {
    coordinates[site] = {approximate.data() + 3 * site, exact.data() + 3 * site};
    for (int axis = 0; axis < 3; ++axis) {
      approximate[3 * site + axis] = phx::mc::Approx::exact(sites[3 * site + axis]);
      exact[3 * site + axis] = phx::mc::Expansion(sites[3 * site + axis]);
    }
  }
  phx::mc::TetrahedronClipper clipper;
  phx::mc::ClippedPolytope polytope;
  PHX_CHECK(clipper.clip_power(tetrahedron, sites, weights, 0, neighbors, 3, polytope,
                               coordinates.data()) == PHX_MC_OK);
  PHX_CHECK(!polytope.empty());
  bool saw_supported_three_plane_vertex = false;
  for (int32_t vertex = 0; vertex < static_cast<int32_t>(polytope.vertices.size() / 3);
       ++vertex) {
    int base_support_count = 0;
    bool off_face = false;
    for (int32_t face = 0; face < 4; ++face) {
      const int side = clipper.vertex_base_side(vertex, face);
      PHX_CHECK(side >= -1 && side <= 1);
      PHX_CHECK(side >= 0);
      base_support_count += side == 0;
      off_face = off_face || side > 0;
    }
    int tied_power_planes = 0;
    for (const int32_t neighbor : neighbors) {
      const int side = clipper.vertex_power_side(
          vertex, sites, weights[0], sites + 3 * neighbor, weights[neighbor],
          coordinates.data(), coordinates.data() + neighbor);
      PHX_CHECK(side >= -1 && side <= 1);
      PHX_CHECK(side >= 0);
      tied_power_planes += side == 0;
    }
    PHX_CHECK(base_support_count > 0 && off_face);
    saw_supported_three_plane_vertex =
        saw_supported_three_plane_vertex ||
        (base_support_count == 1 && tied_power_planes == 2);
  }
  PHX_CHECK(saw_supported_three_plane_vertex);
}

void exact_power_clip_rounds_constructed_vertex_to_nearest_binary64() {
  constexpr double tetrahedron[] = {
      -0x1.b5f71a9c61e91p-1, -0x1.9df71a9c61e91p-1, -0x1.b5f71a9c61e91p-1,
      0x1.007dc6a7187a5p+1,  -0x1.9df71a9c61e91p-1, -0x1.b5f71a9c61e91p-1,
      0x1.007dc6a7187a5p+1,  0x1.eafb8d4e30f49p+0,  -0x1.b5f71a9c61e91p-1,
      0x1.007dc6a7187a5p+1,  0x1.eafb8d4e30f49p+0,  0x1.077dc6a7187a5p+1,
  };
  constexpr double sites[] = {
      0.640625, 0.203125, 0.15625, 1.28125, 0.4765625, -0.171875,
      0.8671875, 0.546875, 0.046875, 0.6953125, -0.09375, -0.484375,
  };
  constexpr double weights[] = {0.015, 0.02, -0.015, 0.005};
  constexpr int32_t neighbors[] = {1, 2, 3};
  std::array<phx::mc::Approx, 12> approximate{};
  std::array<phx::mc::Expansion, 12> exact{};
  std::array<phx::mc::ExactPowerCoordinates, 4> coordinates{};
  for (int32_t site = 0; site < 4; ++site) {
    coordinates[site] = {approximate.data() + 3 * site, exact.data() + 3 * site};
    for (int axis = 0; axis < 3; ++axis) {
      approximate[3 * site + axis] = phx::mc::Approx::exact(sites[3 * site + axis]);
      exact[3 * site + axis] = phx::mc::Expansion(sites[3 * site + axis]);
    }
  }
  phx::mc::TetrahedronClipper clipper;
  phx::mc::ClippedPolytope polytope;
  PHX_CHECK(clipper.clip_power(tetrahedron, sites, weights, 0, neighbors, 3, polytope,
                               coordinates.data()) == PHX_MC_OK);
  // Correctly-rounded coordinates of the unique exact four-site/carrier root.
  constexpr std::array<std::uint64_t, 3> expected_bits{
      4606230517804646053ULL, 4597121506750838735ULL, 13820808751096205813ULL,
  };
  int32_t exact_vertex = -1;
  for (int32_t vertex = 0; vertex < static_cast<int32_t>(polytope.vertices.size() / 3);
       ++vertex) {
    bool active = true;
    for (const int32_t neighbor : neighbors) {
      active = active &&
               clipper.vertex_power_side(vertex, sites, weights[0], sites + 3 * neighbor,
                                         weights[neighbor], coordinates.data(),
                                         coordinates.data() + neighbor) == 0;
    }
    for (int32_t face = 0; face < 4; ++face) {
      active = active && clipper.vertex_base_side(vertex, face) > 0;
    }
    if (active) {
      PHX_CHECK(exact_vertex < 0);
      exact_vertex = vertex;
    }
  }
  PHX_CHECK(exact_vertex >= 0);
  for (int axis = 0; axis < 3; ++axis) {
    PHX_CHECK(std::bit_cast<std::uint64_t>(polytope.vertices[3 * exact_vertex + axis]) ==
              expected_bits[axis]);
  }
}

void positive_exact_power_cell_collapsing_to_a_plane_is_refused() {
  constexpr double sites[] = {0, 0, 0, 1, 0, 0, 2, 0, 0};
  const double neighbor_weight = std::nextafter(1.0, 0.0);
  const double weights[] = {neighbor_weight, 0.0, neighbor_weight};
  constexpr int64_t neighbor_offsets[] = {0, 2, 4, 6};
  constexpr int32_t neighbors[] = {1, 2, 0, 2, 0, 1};
  constexpr int8_t split_sites[] = {0, 0, 0};
  constexpr double points[] = {0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2};
  constexpr int32_t tets[] = {0, 1, 2, 3};
  constexpr int32_t regions[] = {0};
  constexpr int32_t facets[] = {0, 1, 2, 3};
  phx_mc_power_cells* handle = nullptr;
  const phx_mc_restricted_power_cells_request request{
      .site_count = 3,
      .sites = sites,
      .weights = weights,
      .neighbor_offsets = neighbor_offsets,
      .neighbors = neighbors,
      .split_sites = split_sites,
      .point_count = 4,
      .points = points,
      .tet_count = 1,
      .tets = tets,
      .tet_regions = regions,
      .tet_face_facets = facets,
      .result = &handle,
  };
  PHX_CHECK(phx_mc_restricted_power_cells(&request) == PHX_MC_DEGENERATE_INPUT);
  PHX_CHECK(handle != nullptr);
  int64_t reason = -1;
  int64_t site = -1;
  int64_t tet = -1;
  int32_t clip_status = -1;
  PHX_CHECK(phx_mc_power_cells_failure(handle, &reason, &site, &tet, &clip_status) == PHX_MC_OK);
  PHX_CHECK(reason == PHX_MC_POWER_DEGENERATE_VERTEX);
  PHX_CHECK(site == 1 && tet == 0 && clip_status == PHX_MC_OK);
  phx_mc_power_cells_free(handle);
}

void exact_weighted_periodic_cube_images_match_owner_volumes() {
  constexpr double original_sites[] = {0.25, 0.5, 0.5, 0.75, 0.5, 0.5};
  constexpr double original_weights[] = {1.0 / 16.0, 0.0};
  const Domain domain = unit_cube_box();
  std::vector<int32_t> image_owners;
  std::vector<int64_t> coordinate_offsets{0};
  std::vector<double> coordinate_components;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        const int translation[] = {x, y, z};
        for (int32_t owner = 0; owner < 2; ++owner) {
          image_owners.push_back(owner);
          for (int axis = 0; axis < 3; ++axis) {
            coordinate_components.push_back(original_sites[3 * owner + axis] +
                                            translation[axis]);
            coordinate_offsets.push_back(static_cast<int64_t>(coordinate_components.size()));
          }
        }
      }
    }
  }
  const int64_t image_count = static_cast<int64_t>(image_owners.size());
  std::vector<int64_t> neighbor_offsets(static_cast<std::size_t>(image_count + 1));
  std::vector<int32_t> neighbors;
  for (int32_t image = 0; image < image_count; ++image) {
    neighbor_offsets[image] = static_cast<int64_t>(neighbors.size());
    for (int32_t other = 0; other < image_count; ++other) {
      if (other != image) neighbors.push_back(other);
    }
  }
  neighbor_offsets[image_count] = static_cast<int64_t>(neighbors.size());
  std::vector<int8_t> split_sites(static_cast<std::size_t>(image_count), 0);
  phx_mc_power_cells* handle = nullptr;
  int64_t failed_coordinate = -2;
  int64_t failed_component = -2;
  const phx_mc_restricted_power_cells_exact_request request{
      .original_site_count = 2,
      .original_sites = original_sites,
      .original_weights = original_weights,
      .image_count = image_count,
      .image_site_owners = image_owners.data(),
      .image_coordinate_offsets = coordinate_offsets.data(),
      .component_count = static_cast<int64_t>(coordinate_components.size()),
      .image_coordinate_components = coordinate_components.data(),
      .neighbor_offsets = neighbor_offsets.data(),
      .neighbors = neighbors.data(),
      .split_sites = split_sites.data(),
      .point_count = 8,
      .points = domain.points.data(),
      .tet_count = 6,
      .tets = domain.tets.data(),
      .tet_regions = domain.regions.data(),
      .tet_face_facets = domain.facets.data(),
      .result = &handle,
      .failed_image_coordinate = &failed_coordinate,
      .failed_component = &failed_component,
  };
  PHX_CHECK(phx_mc_restricted_power_cells_exact(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  PHX_CHECK(failed_coordinate == -1 && failed_component == -1);
  std::array<int64_t, 5> sizes{};
  PHX_CHECK(phx_mc_power_cells_sizes(handle, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[3] == 3);
  std::vector<double> vertices(static_cast<std::size_t>(3 * sizes[0]));
  std::vector<int64_t> face_offsets(static_cast<std::size_t>(sizes[1] + 1));
  std::vector<int32_t> face_vertices(static_cast<std::size_t>(sizes[2]));
  std::vector<int32_t> face_cells(static_cast<std::size_t>(2 * sizes[1]));
  std::vector<int32_t> face_facets(static_cast<std::size_t>(sizes[1]));
  std::vector<int32_t> cell_sites(static_cast<std::size_t>(sizes[3]));
  std::vector<int32_t> cell_regions(static_cast<std::size_t>(sizes[3]));
  std::vector<double> cell_volumes(static_cast<std::size_t>(sizes[3]));
  std::vector<double> cell_moments(static_cast<std::size_t>(3 * sizes[3]));
  std::vector<double> cell_second(static_cast<std::size_t>(sizes[3]));
  std::vector<int32_t> piece_sites(static_cast<std::size_t>(sizes[4]));
  std::vector<int32_t> piece_tets(static_cast<std::size_t>(sizes[4]));
  std::vector<int32_t> piece_cells(static_cast<std::size_t>(sizes[4]));
  std::vector<double> piece_volumes(static_cast<std::size_t>(sizes[4]));
  std::vector<double> piece_moments(static_cast<std::size_t>(3 * sizes[4]));
  PHX_CHECK(phx_mc_power_cells_export(
                handle, sizes[0], sizes[1] + 1, sizes[1], sizes[2], sizes[3], sizes[4],
                vertices.data(), face_offsets.data(), face_vertices.data(), face_cells.data(),
                face_facets.data(), cell_sites.data(), cell_regions.data(), cell_volumes.data(),
                cell_moments.data(), cell_second.data(), piece_sites.data(), piece_tets.data(),
                piece_cells.data(), piece_volumes.data(), piece_moments.data()) == PHX_MC_OK);
  std::array<double, 2> owner_volumes{};
  for (int32_t cell = 0; cell < static_cast<int32_t>(sizes[3]); ++cell) {
    PHX_CHECK(cell_sites[cell] >= 0 && cell_sites[cell] < image_count);
    owner_volumes[image_owners[cell_sites[cell]]] += cell_volumes[cell];
  }
  PHX_CHECK_NEAR(owner_volumes[0], 5.0 / 8.0, 1e-12);
  PHX_CHECK_NEAR(owner_volumes[1], 3.0 / 8.0, 1e-12);
  phx_mc_power_cells_free(handle);
}

Domain two_region_box() {
  Domain domain;
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        domain.points.insert(domain.points.end(), {double(x), double(y), double(z)});
  const auto id = [](int x, int y, int z) { return 4 * x + 2 * y + z; };
  constexpr int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int cube = 0; cube < 2; ++cube) {
    for (const auto& permutation : permutations) {
      int corner[3] = {0, 0, 0};
      std::array<int32_t, 4> tet{};
      tet[0] = id(cube, 0, 0);
      for (int step = 0; step < 3; ++step) {
        corner[permutation[step]] = 1;
        tet[step + 1] = id(cube + corner[0], corner[1], corner[2]);
      }
      if (phx::mc::orient3d(domain.points.data() + 3 * tet[0],
                            domain.points.data() + 3 * tet[1],
                            domain.points.data() + 3 * tet[2],
                            domain.points.data() + 3 * tet[3]) < 0) {
        std::swap(tet[0], tet[1]);
      }
      domain.tets.insert(domain.tets.end(), tet.begin(), tet.end());
      domain.regions.push_back(cube);
      for (int opposite = 0; opposite < 4; ++opposite) {
        std::array<int32_t, 3> face{};
        int slot = 0;
        for (int vertex = 0; vertex < 4; ++vertex)
          if (vertex != opposite) face[slot++] = tet[vertex];
        domain.facets.push_back(facet_on_face(domain.points, face));
      }
    }
  }
  return domain;
}

void two_region_cells_close_orientably() {
  const Domain domain = two_region_box();
  constexpr double sites[] = {0.5, 0.5, 0.5, 1.5, 0.5, 0.5};
  constexpr double weights[] = {0.0, 0.0};
  constexpr int64_t neighbor_offsets[] = {0, 1, 2};
  constexpr int32_t neighbors[] = {1, 0};
  constexpr int8_t split_sites[] = {0, 0};
  phx_mc_power_cells* handle = nullptr;
  const phx_mc_restricted_power_cells_request request{
      .site_count = 2,
      .sites = sites,
      .weights = weights,
      .neighbor_offsets = neighbor_offsets,
      .neighbors = neighbors,
      .split_sites = split_sites,
      .point_count = 12,
      .points = domain.points.data(),
      .tet_count = 12,
      .tets = domain.tets.data(),
      .tet_regions = domain.regions.data(),
      .tet_face_facets = domain.facets.data(),
      .result = &handle,
  };
  PHX_CHECK(phx_mc_restricted_power_cells(&request) == PHX_MC_OK);
  PHX_CHECK(handle != nullptr);
  std::array<int64_t, 5> sizes{};
  PHX_CHECK(phx_mc_power_cells_sizes(handle, sizes.data()) == PHX_MC_OK);
  PHX_CHECK(sizes[3] == 2);
  std::vector<double> vertices(static_cast<std::size_t>(3 * sizes[0]));
  std::vector<int64_t> face_offsets(static_cast<std::size_t>(sizes[1] + 1));
  std::vector<int32_t> face_vertices(static_cast<std::size_t>(sizes[2]));
  std::vector<int32_t> face_cells(static_cast<std::size_t>(2 * sizes[1]));
  std::vector<int32_t> face_facets(static_cast<std::size_t>(sizes[1]));
  std::vector<int32_t> cell_sites(static_cast<std::size_t>(sizes[3]));
  std::vector<int32_t> cell_regions(static_cast<std::size_t>(sizes[3]));
  std::vector<double> cell_volumes(static_cast<std::size_t>(sizes[3]));
  std::vector<double> cell_moments(static_cast<std::size_t>(3 * sizes[3]));
  std::vector<double> cell_second(static_cast<std::size_t>(sizes[3]));
  std::vector<int32_t> piece_sites(static_cast<std::size_t>(sizes[4]));
  std::vector<int32_t> piece_tets(static_cast<std::size_t>(sizes[4]));
  std::vector<int32_t> piece_cells(static_cast<std::size_t>(sizes[4]));
  std::vector<double> piece_volumes(static_cast<std::size_t>(sizes[4]));
  std::vector<double> piece_moments(static_cast<std::size_t>(3 * sizes[4]));
  PHX_CHECK(phx_mc_power_cells_export(
                handle, sizes[0], sizes[1] + 1, sizes[1], sizes[2], sizes[3], sizes[4],
                vertices.data(), face_offsets.data(), face_vertices.data(), face_cells.data(),
                face_facets.data(), cell_sites.data(), cell_regions.data(), cell_volumes.data(),
                cell_moments.data(), cell_second.data(), piece_sites.data(), piece_tets.data(),
                piece_cells.data(), piece_volumes.data(), piece_moments.data()) == PHX_MC_OK);
  PHX_CHECK(std::fabs(cell_volumes[0] - 1.0) < 1.0e-12);
  PHX_CHECK(std::fabs(cell_volumes[1] - 1.0) < 1.0e-12);
  PHX_CHECK(cell_sites[0] == 0 && cell_regions[0] == 0);
  PHX_CHECK(cell_sites[1] == 1 && cell_regions[1] == 1);
  for (int32_t cell = 0; cell < 2; ++cell) {
    std::map<std::pair<int32_t, int32_t>, int> directed_edges;
    for (std::size_t face = 0; face < face_facets.size(); ++face) {
      int side = face_cells[2 * face] == cell ? 0 : face_cells[2 * face + 1] == cell ? 1 : -1;
      if (side < 0) continue;
      const int64_t begin = face_offsets[face], end = face_offsets[face + 1];
      for (int64_t corner = begin; corner < end; ++corner) {
        int32_t a = face_vertices[corner];
        int32_t b = face_vertices[corner + 1 == end ? begin : corner + 1];
        if (side == 1) std::swap(a, b);
        ++directed_edges[{a, b}];
      }
    }
    for (const auto& [edge, count] : directed_edges)
      PHX_CHECK((count == directed_edges[{edge.second, edge.first}]));
  }
  phx_mc_power_cells_free(handle);
}
}  // namespace
int main() {
  single_site_tetrahedron();
  exact_image_coordinates_keep_source_identity();
  two_region_cells_close_orientably();
  exact_clipper_classifies_constructed_vertices_against_base_faces();
  exact_power_clip_rounds_constructed_vertex_to_nearest_binary64();
  positive_exact_power_cell_collapsing_to_a_plane_is_refused();
  exact_weighted_periodic_cube_images_match_owner_volumes();
  return phx::mc::test::finish("test_power_diagram");
}
