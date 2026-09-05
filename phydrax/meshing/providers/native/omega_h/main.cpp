// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Small executable boundary around the public Omega_h API. MPI owns its lifetime;
// no MPI initialization or optional shared library loading occurs in Python.
#include <Omega_h_adapt.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_matrix.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
template <class T>
Omega_h::Read<T> read_array(std::istream& in, Omega_h::LO count) {
  Omega_h::HostWrite<T> values(count);
  for (Omega_h::LO i = 0; i < count; ++i) {
    if (!(in >> values[i])) throw std::runtime_error("Truncated Omega_h input");
  }
  return Omega_h::Read<T>(values.write());
}

template <class T>
void json_array(std::ostream& out, Omega_h::Read<T> values) {
  Omega_h::HostRead<T> host(values);
  out << '[';
  for (Omega_h::LO i = 0; i < host.size(); ++i) {
    if (i) out << ',';
    out << host[i];
  }
  out << ']';
}

void write_partition(Omega_h::Mesh& mesh, std::string const& directory,
    int iterations) {
  auto const rank = mesh.comm()->rank();
  auto const dim = mesh.dim();
  auto const vertex_owners = mesh.ask_owners(0);
  auto const cell_owners = mesh.ask_owners(dim);
  // Resolve collective data before opening a rank-local output stream.
  auto const global_vertices = mesh.nglobal_ents(0);
  auto const global_cells = mesh.nglobal_ents(dim);
  std::ofstream out(directory + "/rank-" + std::to_string(rank) + ".json");
  if (!out) throw std::runtime_error("Cannot open Omega_h partition output");
  out << std::setprecision(17);
  out << "{\"protocol\":1,\"rank\":" << rank
      << ",\"size\":" << mesh.comm()->size() << ",\"dimension\":" << dim
      << ",\"global_vertices\":" << global_vertices
      << ",\"global_cells\":" << global_cells
      << ",\"iterations\":" << iterations << ",\"vertex_ids\":";
  json_array(out, mesh.globals(0));
  out << ",\"coordinates\":";
  json_array(out, mesh.coords());
  out << ",\"vertex_owner_ranks\":";
  json_array(out, vertex_owners.ranks);
  out << ",\"vertex_owner_indices\":";
  json_array(out, vertex_owners.idxs);
  out << ",\"cell_ids\":";
  json_array(out, mesh.globals(dim));
  out << ",\"cells\":";
  json_array(out, mesh.ask_elem_verts());
  out << ",\"cell_owner_ranks\":";
  json_array(out, cell_owners.ranks);
  out << ",\"cell_owner_indices\":";
  json_array(out, cell_owners.idxs);
  out << ",\"metric\":";
  json_array(out, Omega_h::symms_osh2inria(dim,
      mesh.get_array<Omega_h::Real>(0, "metric")));
  out << "}\n";
  out.close();
  if (!out) throw std::runtime_error("Cannot finish Omega_h partition output");
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
    // All ranks read the tiny header; only rank zero imports the source carrier.
    std::ifstream input(argv[1]);
    std::string magic;
    int dim = 0, nv = 0, nc = 0, max_iterations = 0;
    double gradation = 0.0, feature_angle = 0.0;
    if (!(input >> magic >> dim >> nv >> nc >> gradation >> feature_angle
          >> max_iterations) || magic != "PHYDRAX_OMEGA_H_1" ||
        (dim != 2 && dim != 3) || nv <= 0 || nc <= 0 ||
        gradation < 1.0 || feature_angle <= 0.0 || max_iterations <= 0) {
      throw std::runtime_error("Invalid Omega_h bridge input header");
    }
    Omega_h::Mesh mesh(&library);
    if (world->rank() == 0) {
      auto vertex_ids = read_array<Omega_h::GO>(input, nv);
      auto coordinates = read_array<Omega_h::Real>(input, nv * dim);
      auto cells = read_array<Omega_h::LO>(input, nc * (dim + 1));
      auto cell_ids = read_array<Omega_h::GO>(input, nc);
      auto metric = read_array<Omega_h::Real>(input,
          nv * Omega_h::symm_ncomps(dim));
      Omega_h::build_from_elems2verts(&mesh, library.self(), OMEGA_H_SIMPLEX,
          dim, cells, vertex_ids);
      mesh.add_coords(coordinates);
      mesh.set_tag(dim, "global", cell_ids);
      Omega_h::classify_by_angles(&mesh, feature_angle);
      Omega_h::add_implied_metric_tag(&mesh);
      Omega_h::add_metric_tag(&mesh, Omega_h::symms_inria2osh(dim, metric),
          "target_metric");
    }
    // Official parallel-adaptation pattern: expand a rank-zero self mesh,
    // migrate its tags through balance(), then adapt collectively on world.
    mesh.set_comm(world);
    if (world->size() > 1) mesh.balance();
    mesh.set_parting(OMEGA_H_GHOSTED);
    auto target = Omega_h::limit_metric_gradation(&mesh,
        mesh.get_array<Omega_h::Real>(0, "target_metric"), gradation,
        1.0e-2, false);
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
    write_partition(mesh, argv[2], iterations);
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
