// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Bounded executable boundary around the public Omega_h API.
#include <Omega_h_adapt.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_metric.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr std::uint64_t hard_max_vertices = 10000000;
constexpr std::uint64_t hard_max_cells = 20000000;
constexpr std::uint64_t hard_max_connectivity = 500000000;
constexpr std::uint64_t hard_max_bytes = 4000000000;

std::uint64_t checked_product(
    std::uint64_t left, std::uint64_t right, std::uint64_t maximum,
    char const* name) {
  if (left != 0 && right > maximum / left)
    throw std::runtime_error(std::string(name) + " exceeds its bound");
  return left * right;
}

Omega_h::LO local_count(std::uint64_t value, char const* name) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<Omega_h::LO>::max()))
    throw std::runtime_error(std::string(name) + " exceeds Omega_h local indexing");
  return static_cast<Omega_h::LO>(value);
}

template <class T>
Omega_h::Read<T> read_array(std::istream& input, std::uint64_t count) {
  Omega_h::HostWrite<T> values(local_count(count, "array count"));
  for (Omega_h::LO index = 0; index < values.size(); ++index) {
    if (!(input >> values[index]))
      throw std::runtime_error("Truncated Omega_h input");
  }
  return Omega_h::Read<T>(values.write());
}

template <class T>
void json_array(std::ostream& output, Omega_h::Read<T> values) {
  Omega_h::HostRead<T> host(values);
  output << '[';
  for (Omega_h::LO index = 0; index < host.size(); ++index) {
    if (index) output << ',';
    output << host[index];
  }
  output << ']';
}

void write_partition(
    Omega_h::Mesh& mesh, std::string const& directory, int iterations,
    std::uint64_t max_vertices, std::uint64_t max_cells,
    std::uint64_t max_connectivity, std::uint64_t max_bytes) {
  auto const rank = mesh.comm()->rank();
  auto const dimension = mesh.dim();
  auto const rank_byte_limit =
      max_bytes / static_cast<std::uint64_t>(mesh.comm()->size());
  auto const local_vertices = static_cast<std::uint64_t>(mesh.nents(0));
  auto const local_cells = static_cast<std::uint64_t>(mesh.nents(dimension));
  auto const global_vertices =
      static_cast<std::uint64_t>(mesh.nglobal_ents(0));
  auto const global_cells =
      static_cast<std::uint64_t>(mesh.nglobal_ents(dimension));
  if (local_vertices > max_vertices || global_vertices > max_vertices ||
      local_cells > max_cells || global_cells > max_cells)
    throw std::runtime_error("Omega_h output exceeds entity bounds");
  auto const connectivity = checked_product(
      local_cells, static_cast<std::uint64_t>(dimension + 1),
      max_connectivity, "output connectivity");
  auto const metric = checked_product(
      local_vertices,
      static_cast<std::uint64_t>(Omega_h::symm_ncomps(dimension)),
      max_connectivity, "output metric");
  std::uint64_t scalar_count = 4096;
  if (rank_byte_limit < 4096)
    throw std::runtime_error("Omega_h output byte bound is too small");
  for (auto count : {
           local_vertices,
           checked_product(local_vertices, dimension, max_connectivity,
                           "output coordinates"),
           local_vertices,
           local_vertices,
           local_cells,
           connectivity,
           local_cells,
           local_cells,
           metric,
       }) {
    if (count > (rank_byte_limit - scalar_count) / 32)
      throw std::runtime_error("Omega_h output exceeds byte bound");
    scalar_count += count * 32;
  }
  auto const vertex_owners = mesh.ask_owners(0);
  auto const cell_owners = mesh.ask_owners(dimension);
  auto const vertex_globals = mesh.globals(0);
  auto const cell_globals = mesh.globals(dimension);
  auto const coordinates = mesh.coords();
  auto const elements = mesh.ask_elem_verts();
  auto const metric_values = Omega_h::symms_osh2inria(
      dimension, mesh.get_array<Omega_h::Real>(0, "metric"));
  Omega_h::HostRead<Omega_h::Real> host_coordinates(coordinates);
  Omega_h::HostRead<Omega_h::Real> host_metric(metric_values);
  Omega_h::HostRead<Omega_h::LO> host_elements(elements);
  Omega_h::HostRead<Omega_h::I32> host_vertex_owner_ranks(vertex_owners.ranks);
  Omega_h::HostRead<Omega_h::GO> host_vertex_globals(vertex_globals);
  Omega_h::HostRead<Omega_h::GO> host_cell_globals(cell_globals);
  for (Omega_h::LO index = 0; index < host_vertex_globals.size(); ++index)
    if (host_vertex_globals[index] < 0)
      throw std::runtime_error("Omega_h output vertex ID is negative");
  for (Omega_h::LO index = 0; index < host_cell_globals.size(); ++index)
    if (host_cell_globals[index] < 0)
      throw std::runtime_error("Omega_h output cell ID is negative");
  Omega_h::HostRead<Omega_h::LO> host_vertex_owner_indices(vertex_owners.idxs);
  Omega_h::HostRead<Omega_h::I32> host_cell_owner_ranks(cell_owners.ranks);
  Omega_h::HostRead<Omega_h::LO> host_cell_owner_indices(cell_owners.idxs);
  for (Omega_h::LO index = 0; index < host_coordinates.size(); ++index)
    if (!std::isfinite(host_coordinates[index]))
      throw std::runtime_error("Omega_h output coordinates are nonfinite");
  for (Omega_h::LO index = 0; index < host_metric.size(); ++index)
    if (!std::isfinite(host_metric[index]))
      throw std::runtime_error("Omega_h output metric is nonfinite");
  for (Omega_h::LO index = 0; index < host_elements.size(); ++index)
    if (host_elements[index] < 0 ||
        static_cast<std::uint64_t>(host_elements[index]) >= local_vertices)
      throw std::runtime_error("Omega_h output connectivity is out of range");
  for (Omega_h::LO index = 0; index < host_vertex_owner_ranks.size(); ++index)
    if (host_vertex_owner_ranks[index] < 0 ||
        host_vertex_owner_ranks[index] >= mesh.comm()->size() ||
        host_vertex_owner_indices[index] < 0)
      throw std::runtime_error("Omega_h output vertex ownership is invalid");
  for (Omega_h::LO index = 0; index < host_cell_owner_ranks.size(); ++index)
    if (host_cell_owner_ranks[index] < 0 ||
        host_cell_owner_ranks[index] >= mesh.comm()->size() ||
        host_cell_owner_indices[index] < 0)
      throw std::runtime_error("Omega_h output cell ownership is invalid");
  std::ofstream output(
      directory + "/rank-" + std::to_string(rank) + ".json",
      std::ios::binary);
  if (!output)
    throw std::runtime_error("Cannot open Omega_h partition output");
  output << std::setprecision(17);
  output << "{\"protocol\":1,\"rank\":" << rank
         << ",\"size\":" << mesh.comm()->size()
         << ",\"dimension\":" << dimension
         << ",\"global_vertices\":" << global_vertices
         << ",\"global_cells\":" << global_cells
         << ",\"iterations\":" << iterations << ",\"vertex_ids\":";
  json_array(output, vertex_globals);
  output << ",\"coordinates\":";
  json_array(output, coordinates);
  output << ",\"vertex_owner_ranks\":";
  json_array(output, vertex_owners.ranks);
  output << ",\"vertex_owner_indices\":";
  json_array(output, vertex_owners.idxs);
  output << ",\"cell_ids\":";
  json_array(output, cell_globals);
  output << ",\"cells\":";
  json_array(output, elements);
  output << ",\"cell_owner_ranks\":";
  json_array(output, cell_owners.ranks);
  output << ",\"cell_owner_indices\":";
  json_array(output, cell_owners.idxs);
  output << ",\"metric\":";
  json_array(output, metric_values);
  output << "}\n";
  auto const written = output.tellp();
  if (written < 0 ||
      static_cast<std::uint64_t>(written) > rank_byte_limit)
    throw std::runtime_error("Omega_h output exceeded byte bound");
  output.close();
  if (!output)
    throw std::runtime_error("Cannot finish Omega_h partition output");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "{\"protocol\":1,\"version\":\"" << OMEGA_H_SEMVER
              << "\",\"commit\":\"" << OMEGA_H_COMMIT << "\",\"mpi\":"
#ifdef OMEGA_H_USE_MPI
              << "true";
#else
              << "false";
#endif
    std::cout << "}\n";
    return 0;
  }
  if (argc != 3) {
    std::cerr << "Usage: phydrax_omega_h INPUT OUTPUT_DIRECTORY\n";
    return 2;
  }
  auto library = Omega_h::Library(&argc, &argv);
  auto world = library.world();
  try {
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input)
      throw std::runtime_error("Cannot open Omega_h input");
    auto const input_bytes = input.tellg();
    if (input_bytes < 0)
      throw std::runtime_error("Cannot determine Omega_h input size");
    input.seekg(0);
    std::string magic;
    int dimension = 0, max_iterations = 0;
    std::uint64_t vertex_count = 0, cell_count = 0;
    std::uint64_t max_vertices = 0, max_cells = 0;
    std::uint64_t max_connectivity = 0, max_bytes = 0;
    double gradation = 0.0, feature_angle = 0.0;
    if (!(input >> magic >> dimension >> vertex_count >> cell_count >>
          gradation >> feature_angle >> max_iterations >> max_vertices >>
          max_cells >> max_connectivity >> max_bytes) ||
        magic != "PHYDRAX_OMEGA_H_1" ||
        (dimension != 2 && dimension != 3) || vertex_count == 0 ||
        cell_count == 0 || !std::isfinite(gradation) || gradation < 1.0 ||
        !std::isfinite(feature_angle) || feature_angle <= 0.0 ||
        max_iterations <= 0 || max_vertices == 0 ||
        max_vertices > hard_max_vertices || max_cells == 0 ||
        max_cells > hard_max_cells || max_connectivity == 0 ||
        max_connectivity > hard_max_connectivity || max_bytes == 0 ||
        max_bytes > hard_max_bytes ||
        static_cast<std::uint64_t>(input_bytes) > max_bytes ||
        vertex_count > max_vertices || cell_count > max_cells) {
      throw std::runtime_error("Invalid Omega_h bridge input header");
    }
    auto const coordinate_count = checked_product(
        vertex_count, static_cast<std::uint64_t>(dimension),
        max_connectivity, "coordinate count");
    auto const connectivity_count = checked_product(
        cell_count, static_cast<std::uint64_t>(dimension + 1),
        max_connectivity, "connectivity count");
    auto const metric_count = checked_product(
        vertex_count,
        static_cast<std::uint64_t>(Omega_h::symm_ncomps(dimension)),
        max_connectivity, "metric count");
    Omega_h::Mesh mesh(&library);
    if (world->rank() == 0) {
      auto vertex_ids = read_array<Omega_h::GO>(input, vertex_count);
      auto coordinates =
          read_array<Omega_h::Real>(input, coordinate_count);
      auto cells = read_array<Omega_h::LO>(input, connectivity_count);
      auto cell_ids = read_array<Omega_h::GO>(input, cell_count);
      auto metric = read_array<Omega_h::Real>(input, metric_count);
      Omega_h::HostRead<Omega_h::LO> host_cells(cells);
      for (Omega_h::LO index = 0; index < host_cells.size(); ++index)
        if (host_cells[index] < 0 ||
            static_cast<std::uint64_t>(host_cells[index]) >= vertex_count)
          throw std::runtime_error("Omega_h connectivity index is out of range");
      Omega_h::HostRead<Omega_h::Real> host_coordinates(coordinates);
      for (Omega_h::LO index = 0; index < host_coordinates.size(); ++index)
        if (!std::isfinite(host_coordinates[index]))
          throw std::runtime_error("Omega_h coordinates must be finite");
      Omega_h::HostRead<Omega_h::Real> host_metric(metric);
      for (Omega_h::LO index = 0; index < host_metric.size(); ++index)
        if (!std::isfinite(host_metric[index]))
          throw std::runtime_error("Omega_h metric must be finite");
      input >> std::ws;
      if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected trailing Omega_h input");
      Omega_h::build_from_elems2verts(
          &mesh, library.self(), OMEGA_H_SIMPLEX, dimension, cells,
          vertex_ids);
      mesh.add_coords(coordinates);
      mesh.set_tag(dimension, "global", cell_ids);
      Omega_h::classify_by_angles(&mesh, feature_angle);
      Omega_h::add_implied_metric_tag(&mesh);
      Omega_h::add_metric_tag(
          &mesh, Omega_h::symms_inria2osh(dimension, metric),
          "target_metric");
    }
    mesh.set_comm(world);
    if (world->size() > 1) mesh.balance();
    mesh.set_parting(OMEGA_H_GHOSTED);
    auto target = Omega_h::limit_metric_gradation(
        &mesh, mesh.get_array<Omega_h::Real>(0, "target_metric"),
        gradation, 1.0e-2, false);
    mesh.set_tag(0, "target_metric", target);
    auto options = Omega_h::AdaptOpts(&mesh);
    options.verbosity = Omega_h::SILENT;
    int iterations = 0;
    while (Omega_h::approach_metric(&mesh, options)) {
      if (++iterations > max_iterations)
        throw std::runtime_error("Omega_h metric approach iteration limit");
      Omega_h::adapt(&mesh, options);
    }
    Omega_h::adapt(&mesh, options);
    mesh.set_parting(OMEGA_H_ELEM_BASED);
    if (world->size() > 1) mesh.balance();
    mesh.set_parting(OMEGA_H_GHOSTED, 1, false);
    write_partition(
        mesh, argv[2], iterations, max_vertices, max_cells,
        max_connectivity, max_bytes);
    world->barrier();
  } catch (std::exception const& error) {
    std::cerr << "phydrax_omega_h rank " << world->rank() << ": "
              << error.what() << '\n';
#ifdef OMEGA_H_USE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }
  return 0;
}
