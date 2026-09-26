// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Persistent VoroCrust extraction worker (phydrax_worker.hpp protocol).
//
// "extract" receives Voronoi seeds as binary arrays and returns the packed
// public-API polyhedral output: vertices, face vertex CSR, face seed pairs, and
// the seed data VoroCrust consumed. The exact source, configuration, and
// library identities of the linked VoroCrust build are reported once in hello.
#include "MeshingVoronoiMesher.h"
#include "phydrax_worker.hpp"

#ifndef PHYDRAX_VOROCRUST_REVISION
#error "Build with the packaged CMake project and an exact VoroCrust revision"
#endif
#ifndef PHYDRAX_VOROCRUST_SOURCE_SHA256
#error "Missing VoroCrust source identity"
#endif
#ifndef PHYDRAX_VOROCRUST_CONFIG_SHA256
#error "Missing VoroCrust configuration identity"
#endif
#ifndef PHYDRAX_VOROCRUST_LIBRARY_SHA256
#error "Missing VoroCrust library identity"
#endif
#ifndef PHYDRAX_VOROCRUST_OPENMP
#error "Missing VoroCrust OpenMP configuration"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using phydrax::exchange::DType;
using phydrax::json::Array;
using phydrax::json::Object;
using phydrax::json::Value;
namespace worker = phydrax::worker;

constexpr std::int64_t hard_max_seeds = 40000000;
constexpr std::int64_t hard_max_vertices = 10000000;
constexpr std::int64_t hard_max_faces = 50000000;
constexpr std::int64_t hard_max_connectivity = 500000000;

std::int64_t bound(Value const& parameters, char const* name, std::int64_t hard) {
  std::int64_t const value = parameters.at(name).as_int();
  if (value <= 0 || value > hard)
    throw std::invalid_argument(std::string("Invalid VoroCrust ") + name);
  return value;
}

// Owns the arrays VoroCrust allocates for its caller.
struct MesherOutput {
  std::size_t vertex_count = 0, face_count = 0;
  double* vertices = nullptr;
  std::size_t** faces = nullptr;

  MesherOutput() = default;
  MesherOutput(MesherOutput const&) = delete;
  MesherOutput& operator=(MesherOutput const&) = delete;
  ~MesherOutput() {
    if (faces != nullptr)
      for (std::size_t face = 0; face < face_count; ++face) delete[] faces[face];
    delete[] faces;
    delete[] vertices;
  }
};

Value extract(worker::Request const& request) {
  auto const& parameters = request.parameters;
  auto const max_vertices = bound(parameters, "maximum_vertices", hard_max_vertices);
  auto const max_faces = bound(parameters, "maximum_faces", hard_max_faces);
  auto const max_connectivity =
      bound(parameters, "maximum_connectivity_entries", hard_max_connectivity);
  auto const input = request.input();
  auto const& points = input.require("seeds", DType::float64, {-1, 3});
  std::int64_t const count = static_cast<std::int64_t>(points.shape[0]);
  if (count == 0 || count > hard_max_seeds)
    throw std::invalid_argument("VoroCrust seed count is out of range");
  std::vector<double> seeds = points.to_vector<double>();
  std::vector<double> sizing =
      input.require("seed_radii", DType::float64, {count}).to_vector<double>();
  auto const* raw_regions =
      input.require("seed_regions", DType::int64, {count}).data<std::int64_t>();
  std::vector<std::size_t> regions(static_cast<std::size_t>(count));
  for (std::int64_t seed = 0; seed < count; ++seed) {
    if (!std::isfinite(seeds[3 * seed]) || !std::isfinite(seeds[3 * seed + 1]) ||
        !std::isfinite(seeds[3 * seed + 2]))
      throw std::invalid_argument("VoroCrust seeds must be finite");
    if (!std::isfinite(sizing[seed]) || sizing[seed] <= 0)
      throw std::invalid_argument("VoroCrust seed radius must be positive");
    if (raw_regions[seed] < 0 || raw_regions[seed] > count)
      throw std::invalid_argument("VoroCrust seed region is out of range");
    regions[seed] = static_cast<std::size_t>(raw_regions[seed]);
  }
  // Each emitted cell is the convex Voronoi polytope of one interior seed with
  // at most n - 1 facets: at most 2(n - 1) - 4 corners and 6(n - 1) - 12
  // face-vertex entries; each face is emitted by one cell.
  auto const interior = static_cast<std::uint64_t>(std::count_if(
      regions.begin(), regions.end(), [](std::size_t region) { return region != 0; }));
  auto const seed_count = static_cast<std::uint64_t>(count);
  auto const facets = seed_count - 1;
  if (interior * (facets >= 2 ? 2 * facets - 4 : 0) > static_cast<std::uint64_t>(max_vertices) ||
      interior * facets > static_cast<std::uint64_t>(max_faces) ||
      2 * interior * (facets >= 2 ? 6 * facets - 12 : 0) >
          static_cast<std::uint64_t>(max_connectivity))
    throw std::length_error("Conservative VoroCrust output bounds exceed configured limits");

  MesherOutput mesh;
  MeshingVoronoiMesher mesher;
  int const status = mesher.generate_3d_voronoi_mesh(
      1, regions.size(), seeds.data(), regions.data(), sizing.data(), mesh.vertex_count,
      mesh.vertices, mesh.face_count, mesh.faces);
  if (status != 0 || mesh.vertex_count == 0 || mesh.face_count == 0)
    throw worker::Failure("library_failure", "VoroCrust extraction failed");
  if (mesh.vertex_count > static_cast<std::uint64_t>(max_vertices) ||
      mesh.face_count > static_cast<std::uint64_t>(max_faces))
    throw std::length_error("VoroCrust output exceeds entity bounds");
  for (std::size_t index = 0; index < 3 * mesh.vertex_count; ++index)
    if (!std::isfinite(mesh.vertices[index]))
      throw worker::Failure("library_failure", "VoroCrust output vertex is nonfinite");

  std::vector<std::int64_t> offsets{0}, face_vertices, face_seeds;
  offsets.reserve(mesh.face_count + 1);
  face_seeds.reserve(2 * mesh.face_count);
  for (std::size_t face = 0; face < mesh.face_count; ++face) {
    std::size_t const* record = mesh.faces[face];
    std::size_t const arity = record[0];
    if (arity < 3) throw worker::Failure("library_failure", "VoroCrust face has invalid arity");
    if (arity > static_cast<std::uint64_t>(max_connectivity) - face_vertices.size())
      throw std::length_error("VoroCrust connectivity exceeds its bound");
    for (std::size_t entry = 1; entry <= arity; ++entry) {
      if (record[entry] >= mesh.vertex_count)
        throw worker::Failure("library_failure", "VoroCrust face vertex is out of range");
      face_vertices.push_back(static_cast<std::int64_t>(record[entry]));
    }
    for (std::size_t side = 1; side <= 2; ++side) {
      if (record[arity + side] >= regions.size())
        throw worker::Failure("library_failure", "VoroCrust face seed is out of range");
      face_seeds.push_back(static_cast<std::int64_t>(record[arity + side]));
    }
    offsets.push_back(static_cast<std::int64_t>(face_vertices.size()));
  }
  std::vector<std::int64_t> seed_regions(regions.begin(), regions.end());

  auto output = request.output();
  output.add("vertices", {mesh.vertex_count, 3}, mesh.vertices);
  output.add("face_offsets", {offsets.size()}, offsets);
  output.add("face_vertices", {face_vertices.size()}, face_vertices);
  output.add("face_seeds", {mesh.face_count, 2}, face_seeds);
  output.add("seed_points", {seed_count, 3}, seeds);
  output.add("seed_regions", {seed_count}, seed_regions);
  output.finish();
  return Object{{"faces", static_cast<std::int64_t>(mesh.face_count)},
                {"vertices", static_cast<std::int64_t>(mesh.vertex_count)}};
}

}  // namespace

int main() {
  Value const identity = Object{
      {"provider", "vorocrust"},
      {"revision", PHYDRAX_VOROCRUST_REVISION},
      {"source_sha256", PHYDRAX_VOROCRUST_SOURCE_SHA256},
      {"config_sha256", PHYDRAX_VOROCRUST_CONFIG_SHA256},
      {"library_sha256", PHYDRAX_VOROCRUST_LIBRARY_SHA256},
      {"openmp", PHYDRAX_VOROCRUST_OPENMP != 0},
      {"operations", Array{"extract"}},
  };
  return worker::serve(identity, [](worker::Request const& request) -> Value {
    if (request.operation == "extract") return extract(request);
    throw worker::Failure("unsupported",
                          "Unknown VoroCrust worker operation '" + request.operation + "'");
  });
}
