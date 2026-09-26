// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Persistent collective TIOGA worker (phydrax_worker.hpp protocol).
//
// Complete mesh parts are assigned round-robin to MPI ranks; only the owning
// rank keeps and registers a part. One registration stays resident between
// requests. TIOGA keeps pointers into the coordinate, IBLANK, boundary, and
// connectivity buffers registered with it, so "move" rewrites the coordinates
// of the named parts in place, re-registers the same buffers (TIOGA's
// moving-mesh path), and reruns connectivity without restarting the process.
//
// Every rank reads the same request and validates it identically before any
// state changes. Rank-local work that may fail is agreed collectively before
// the next collective call, so one failing rank can never strand its peers.
#include <mpi.h>
#include <tioga.h>

#include "phydrax_worker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef PHYDRAX_TIOGA_REVISION
#error "Build with the packaged CMake project and an exact TIOGA revision"
#endif
#ifndef PHYDRAX_TIOGA_ENABLE_UNIQUEID
#error "Declare the TIOGA_ENABLE_UNIQUEID option of the linked TIOGA build"
#endif

static_assert(sizeof(int) == 4 && sizeof(double) == 8, "Unsupported TIOGA ABI");
static_assert(BASE == 1, "This worker expects upstream one-based connectivity");

namespace {

using phydrax::exchange::DType;
using phydrax::json::Array;
using phydrax::json::Object;
using phydrax::json::Value;
namespace exchange = phydrax::exchange;
namespace worker = phydrax::worker;

constexpr std::int64_t hard_max_vertices = 10000000;
constexpr std::int64_t hard_max_cells = 20000000;
constexpr std::int64_t hard_max_connectivity = 500000000;
constexpr std::int64_t int32_max = std::numeric_limits<int>::max();

struct Block {
  int part = 0;
  std::vector<double> xyz;
  std::vector<int> node_blank, cell_blank, walls, overset, arities, counts;
  std::vector<std::vector<int>> connectivity;
  std::vector<int*> routes;
  std::vector<std::uint64_t> cell_gids, node_gids;
};

struct Registration {
  std::int64_t registration = 0;
  std::int64_t state = 0;
  std::vector<std::int64_t> part_nodes, part_cells;
  // Declared before the assembler so TIOGA is destroyed before the buffers
  // it points into.
  std::vector<std::unique_ptr<Block>> blocks;
  std::unique_ptr<TIOGA::tioga> assembler;
};

// The worker agrees on outcomes over `comm`; TIOGA exchanges its own messages
// over the duplicated `library` communicator.
struct Communicator {
  MPI_Comm comm;
  MPI_Comm library;
  int rank;
  int size;
};

// Runs rank-local work and agrees on its outcome. A local failure is rethrown;
// when only a peer failed this returns false so the peer's error is reported.
template <class Local>
bool collectively(Communicator const& world, Local&& local) {
  std::exception_ptr failure;
  try {
    local();
  } catch (...) {
    failure = std::current_exception();
  }
  int const mine = failure ? 1 : 0;
  int any = 0;
  MPI_Allreduce(&mine, &any, 1, MPI_INT, MPI_MAX, world.comm);
  if (failure) std::rethrow_exception(failure);
  return any == 0;
}

std::int64_t integer_parameter(Value const& parameters, char const* name,
                               std::int64_t minimum, std::int64_t maximum) {
  std::int64_t const value = parameters.at(name).as_int();
  if (value < minimum || value > maximum)
    throw std::invalid_argument(std::string("TIOGA parameter ") + name +
                                " is out of range");
  return value;
}

void require_offsets(exchange::Array const& offsets, std::uint64_t total,
                     char const* name) {
  auto const* rows = offsets.data<std::int64_t>();
  if (rows[0] != 0 || static_cast<std::uint64_t>(rows[offsets.count() - 1]) != total)
    throw std::invalid_argument(std::string("Invalid TIOGA ") + name + " offsets");
  for (std::uint64_t index = 1; index < offsets.count(); ++index)
    if (rows[index] < rows[index - 1])
      throw std::invalid_argument(std::string("Invalid TIOGA ") + name + " offsets");
}

void require_finite(exchange::Array const& coordinates) {
  auto const* xyz = coordinates.data<double>();
  for (std::uint64_t index = 0; index < coordinates.count(); ++index)
    if (!std::isfinite(xyz[index]))
      throw std::invalid_argument("TIOGA coordinates must be finite");
}

// Validated registration request, identical on every rank.
struct Layout {
  std::int64_t parts = 0;
  int fringe = 1;
  int exclusion = 0;
  std::vector<std::int64_t> part_nodes, part_cells, node_offsets, cell_offsets;
  std::vector<std::int64_t> first_block, block_entries;
};

Layout validate_registration(exchange::Input const& input, Value const& parameters,
                             int ranks) {
  Layout layout;
  layout.fringe = static_cast<int>(integer_parameter(parameters, "fringe_layers", 1, int32_max));
  layout.exclusion =
      static_cast<int>(integer_parameter(parameters, "exclusion_layers", 0, int32_max));
  std::int64_t const max_vertices =
      integer_parameter(parameters, "maximum_vertices", 1, hard_max_vertices);
  std::int64_t const max_cells =
      integer_parameter(parameters, "maximum_cells", 1, hard_max_cells);
  std::int64_t const max_connectivity =
      integer_parameter(parameters, "maximum_connectivity_entries", 1, hard_max_connectivity);

  auto const& part_nodes = input.require("part_nodes", DType::int64, {-1});
  layout.parts = static_cast<std::int64_t>(part_nodes.count());
  if (layout.parts < 2 || layout.parts < ranks)
    throw std::invalid_argument("TIOGA requires at least two parts and ranks <= parts");
  auto const& part_cells = input.require("part_cells", DType::int64, {layout.parts});
  layout.part_nodes = part_nodes.to_vector<std::int64_t>();
  layout.part_cells = part_cells.to_vector<std::int64_t>();
  std::int64_t nodes = 0, cells = 0;
  for (std::int64_t part = 0; part < layout.parts; ++part) {
    std::int64_t const count = layout.part_nodes[part], cell_count = layout.part_cells[part];
    if (count <= 0 || 3 * count > int32_max || cell_count <= 0 || cell_count > int32_max)
      throw std::invalid_argument("Invalid TIOGA part node or cell count");
    layout.node_offsets.push_back(nodes);
    layout.cell_offsets.push_back(cells);
    nodes += count;
    cells += cell_count;
    if (nodes > max_vertices || cells > max_cells)
      throw std::length_error("TIOGA input exceeds its entity bounds");
  }
  layout.node_offsets.push_back(nodes);
  layout.cell_offsets.push_back(cells);
  require_finite(input.require("coordinates", DType::float64, {nodes, 3}));

  auto const& block_parts = input.require("block_parts", DType::int64, {-1});
  std::int64_t const blocks = static_cast<std::int64_t>(block_parts.count());
  auto const* owner = block_parts.data<std::int64_t>();
  auto const* arity =
      input.require("block_arities", DType::int64, {blocks}).data<std::int64_t>();
  auto const* count = input.require("block_cells", DType::int64, {blocks}).data<std::int64_t>();
  auto const& connectivity = input.require("connectivity", DType::int32, {-1});
  auto const* rows = connectivity.data<std::int32_t>();
  std::vector<std::int64_t> cells_seen(static_cast<std::size_t>(layout.parts), 0);
  std::int64_t entries = 0;
  for (std::int64_t block = 0; block < blocks; ++block) {
    std::int64_t const part = owner[block];
    if (part < 0 || part >= layout.parts || (block > 0 && part < owner[block - 1]))
      throw std::invalid_argument("TIOGA blocks must be grouped by ascending part");
    if (arity[block] != 4 && arity[block] != 5 && arity[block] != 6 && arity[block] != 8)
      throw std::invalid_argument("Unsupported TIOGA cell arity");
    if (count[block] <= 0 || count[block] > layout.part_cells[part] - cells_seen[part])
      throw std::invalid_argument("TIOGA block cell counts contradict their part");
    cells_seen[part] += count[block];
    if (block == 0 || part != owner[block - 1]) {
      if (static_cast<std::int64_t>(layout.first_block.size()) != part)
        throw std::invalid_argument("Every TIOGA part requires cell blocks");
      layout.first_block.push_back(block);
    }
    layout.block_entries.push_back(entries);
    entries += arity[block] * count[block];
    if (entries > max_connectivity || entries > static_cast<std::int64_t>(connectivity.count()))
      throw std::invalid_argument("TIOGA connectivity contradicts its blocks");
    std::int64_t const upper = layout.part_nodes[part];
    for (std::int64_t entry = layout.block_entries.back(); entry < entries; ++entry)
      if (rows[entry] < 0 || rows[entry] >= upper)
        throw std::invalid_argument("Invalid TIOGA connectivity index");
  }
  layout.first_block.push_back(blocks);
  layout.block_entries.push_back(entries);
  if (static_cast<std::int64_t>(layout.first_block.size()) != layout.parts + 1 ||
      entries != static_cast<std::int64_t>(connectivity.count()) ||
      cells_seen != layout.part_cells)
    throw std::invalid_argument("TIOGA blocks do not cover every part");

  for (char const* role : {"wall", "overset"}) {
    std::string const name = role;
    auto const& offsets = input.require(name + "_offsets", DType::int64, {layout.parts + 1});
    auto const& members = input.require(name + "_nodes", DType::int32, {-1});
    require_offsets(offsets, members.count(), role);
    auto const* offset = offsets.data<std::int64_t>();
    auto const* node = members.data<std::int32_t>();
    for (std::int64_t part = 0; part < layout.parts; ++part)
      for (std::int64_t entry = offset[part]; entry < offset[part + 1]; ++entry)
        if (node[entry] < 0 || node[entry] >= layout.part_nodes[part])
          throw std::invalid_argument("Invalid TIOGA " + name + " node index");
  }
  return layout;
}

std::vector<int> one_based(exchange::Array const& array, std::int64_t begin,
                           std::int64_t end) {
  auto const* rows = array.data<std::int32_t>();
  std::vector<int> result(static_cast<std::size_t>(end - begin));
  for (std::int64_t entry = begin; entry < end; ++entry)
    result[static_cast<std::size_t>(entry - begin)] = rows[entry] + 1;
  return result;
}

std::unique_ptr<Block> build_block(exchange::Input const& input, Layout const& layout,
                                   int part) {
  auto block = std::make_unique<Block>();
  block->part = part;
  std::int64_t const node_begin = layout.node_offsets[part];
  std::int64_t const nodes = layout.part_nodes[part];
  auto const* xyz = input.get("coordinates").data<double>();
  block->xyz.assign(xyz + 3 * node_begin, xyz + 3 * (node_begin + nodes));
  block->node_gids.resize(static_cast<std::size_t>(nodes));
  // Collision-free part-namespaced node IDs: TIOGA de-duplicates query points
  // by node global ID, so equal source IDs in different parts must not alias.
  for (std::int64_t node = 0; node < nodes; ++node)
    block->node_gids[static_cast<std::size_t>(node)] =
        static_cast<std::uint64_t>(node_begin + node);
  block->cell_gids.resize(static_cast<std::size_t>(layout.part_cells[part]));
  for (std::int64_t cell = 0; cell < layout.part_cells[part]; ++cell)
    block->cell_gids[static_cast<std::size_t>(cell)] =
        static_cast<std::uint64_t>(layout.cell_offsets[part] + cell);
  auto const* arity = input.get("block_arities").data<std::int64_t>();
  auto const* count = input.get("block_cells").data<std::int64_t>();
  auto const& connectivity = input.get("connectivity");
  for (std::int64_t index = layout.first_block[part]; index < layout.first_block[part + 1];
       ++index) {
    block->arities.push_back(static_cast<int>(arity[index]));
    block->counts.push_back(static_cast<int>(count[index]));
    block->connectivity.push_back(one_based(connectivity, layout.block_entries[index],
                                            layout.block_entries[index + 1]));
  }
  for (auto& rows : block->connectivity) block->routes.push_back(rows.data());
  auto const* walls = input.get("wall_offsets").data<std::int64_t>();
  auto const* overset = input.get("overset_offsets").data<std::int64_t>();
  block->walls = one_based(input.get("wall_nodes"), walls[part], walls[part + 1]);
  block->overset = one_based(input.get("overset_nodes"), overset[part], overset[part + 1]);
  block->node_blank.assign(static_cast<std::size_t>(nodes), 1);
  block->cell_blank.assign(block->cell_gids.size(), 1);
  return block;
}

void register_block(TIOGA::tioga& assembler, Block& block) {
  int const tag = block.part + 1;
  assembler.registerGridData(
      tag, static_cast<int>(block.node_gids.size()), block.xyz.data(),
      block.node_blank.data(), static_cast<int>(block.walls.size()),
      static_cast<int>(block.overset.size()), block.walls.data(), block.overset.data(),
      static_cast<int>(block.arities.size()), block.arities.data(), block.counts.data(),
      block.routes.data(), block.cell_gids.data(), block.node_gids.data());
  assembler.set_cell_iblank(tag, block.cell_blank.data());
}

// Rank-local IBLANK and donor records of the owned parts, in part order.
struct RankOutput {
  std::vector<std::int64_t> parts, donor_parts, receptor_parts, receptor_nodes,
      donor_cells, stencil_offsets{0}, stencil_nodes;
  std::vector<std::int32_t> node_iblank, cell_iblank;
  std::vector<double> stencil_weights;
};

void collect_donors(TIOGA::tioga& assembler, Block& block, Registration const& state,
                    Communicator const& world, RankOutput& output) {
  int donors = 0, fractions = 0;
  assembler.getDonorCount(block.part + 1, &donors, &fractions);
  if (donors < 0 || fractions < 0 || donors > hard_max_vertices ||
      fractions > hard_max_connectivity)
    throw std::runtime_error("TIOGA reported invalid donor counts");
  std::vector<int> receptors(4 * static_cast<std::size_t>(donors));
  std::vector<int> indices(static_cast<std::size_t>(fractions));
  std::vector<double> weights(static_cast<std::size_t>(fractions));
  int returned = donors;
  assembler.getDonorInfo(block.part + 1, receptors.data(), indices.data(), weights.data(),
                         &returned);
  if (returned != donors) throw std::runtime_error("TIOGA donor count changed");
  std::int64_t const parts = static_cast<std::int64_t>(state.part_nodes.size());
  std::int64_t const nodes = static_cast<std::int64_t>(block.node_gids.size());
  std::int64_t const cells = static_cast<std::int64_t>(block.cell_gids.size());
  std::size_t offset = 0;
  for (int donor = 0; donor < donors; ++donor) {
    int const* record = receptors.data() + 4 * static_cast<std::size_t>(donor);
    // A receptor's local block index on its rank maps back to its part
    // through the round-robin distribution.
    std::int64_t const receptor_part =
        static_cast<std::int64_t>(record[2]) * world.size + record[0];
    int const width = record[3];
    if (record[0] < 0 || record[0] >= world.size || record[2] < 0 ||
        receptor_part >= parts || receptor_part == block.part)
      throw std::runtime_error("TIOGA reported an invalid receptor part");
    if (record[1] < 0 || record[1] >= state.part_nodes[static_cast<std::size_t>(receptor_part)])
      throw std::runtime_error("TIOGA reported an invalid receptor node");
    if (width <= 0 || width > 8 || offset + static_cast<std::size_t>(width) >= indices.size())
      throw std::runtime_error("TIOGA reported an invalid donor stencil extent");
    int const cell = indices[offset + static_cast<std::size_t>(width)];
    if (cell < 0 || cell >= cells) throw std::runtime_error("TIOGA reported an invalid donor cell");
    for (int entry = 0; entry < width; ++entry) {
      std::size_t const position = offset + static_cast<std::size_t>(entry);
      if (indices[position] < 0 || indices[position] >= nodes)
        throw std::runtime_error("TIOGA reported an invalid donor node");
      if (!std::isfinite(weights[position]))
        throw std::runtime_error("TIOGA reported a nonfinite donor weight");
      output.stencil_nodes.push_back(indices[position]);
      output.stencil_weights.push_back(weights[position]);
    }
    output.donor_parts.push_back(block.part);
    output.receptor_parts.push_back(receptor_part);
    output.receptor_nodes.push_back(record[1]);
    output.donor_cells.push_back(cell);
    output.stencil_offsets.push_back(static_cast<std::int64_t>(output.stencil_nodes.size()));
    offset += static_cast<std::size_t>(width) + 1;
  }
  if (offset != indices.size()) throw std::runtime_error("TIOGA donor records are incomplete");
}

void write_rank_output(worker::Request const& request, Registration& state,
                       Communicator const& world) {
  RankOutput output;
  for (auto& block : state.blocks) {
    output.parts.push_back(block->part);
    output.node_iblank.insert(output.node_iblank.end(), block->node_blank.begin(),
                              block->node_blank.end());
    output.cell_iblank.insert(output.cell_iblank.end(), block->cell_blank.begin(),
                              block->cell_blank.end());
    collect_donors(*state.assembler, *block, state, world, output);
  }
  auto part = exchange::Output::create_part(request.output_directory,
                                            "rank-" + std::to_string(world.rank),
                                            request.maximum_output_bytes /
                                                static_cast<std::uint64_t>(world.size));
  std::uint64_t const donors = output.donor_parts.size();
  part.add("parts", {output.parts.size()}, output.parts);
  part.add("node_iblank", {output.node_iblank.size()}, output.node_iblank);
  part.add("cell_iblank", {output.cell_iblank.size()}, output.cell_iblank);
  part.add("donor_parts", {donors}, output.donor_parts);
  part.add("receptor_parts", {donors}, output.receptor_parts);
  part.add("receptor_nodes", {donors}, output.receptor_nodes);
  part.add("donor_cells", {donors}, output.donor_cells);
  part.add("stencil_offsets", {donors + 1}, output.stencil_offsets);
  part.add("stencil_nodes", {output.stencil_nodes.size()}, output.stencil_nodes);
  part.add("stencil_weights", {output.stencil_weights.size()}, output.stencil_weights);
  part.finish();
}

// Connectivity, rank outputs, and the root manifest; the registration becomes
// resident only after every rank completed every step.
Value connect_and_report(worker::Request const& request, std::unique_ptr<Registration> working,
                         std::unique_ptr<Registration>& resident, Communicator const& world) {
  working->assembler->performConnectivity();
  working->state = request.sequence;
  if (!collectively(world, [&] { write_rank_output(request, *working, world); }))
    return Object{};
  if (!collectively(world, [&] {
        if (world.rank != 0) return;
        auto root = request.output();
        for (int rank = 0; rank < world.size; ++rank)
          root.declare_part("rank-" + std::to_string(rank));
        root.finish();
      }))
    return Object{};
  Value result = Object{{"registration", working->registration}, {"state", working->state}};
  resident = std::move(working);
  return result;
}

Value register_parts(worker::Request const& request, std::unique_ptr<Registration>& resident,
                     Communicator const& world) {
  auto const input = request.input();
  Layout const layout = validate_registration(input, request.parameters, world.size);
  // A new registration replaces the resident one on every rank.
  resident.reset();
  auto working = std::make_unique<Registration>();
  working->registration = request.sequence;
  working->part_nodes = layout.part_nodes;
  working->part_cells = layout.part_cells;
  if (!collectively(world, [&] {
        for (std::int64_t part = world.rank; part < layout.parts; part += world.size)
          working->blocks.push_back(build_block(input, layout, static_cast<int>(part)));
        working->assembler = std::make_unique<TIOGA::tioga>();
        working->assembler->setCommunicator(world.library, world.rank, world.size);
        int fringe = layout.fringe, exclusion = layout.exclusion;
        working->assembler->setNfringe(&fringe);
        working->assembler->setMexclude(&exclusion);
        for (auto& block : working->blocks) register_block(*working->assembler, *block);
        working->assembler->profile();
      }))
    return Object{};
  return connect_and_report(request, std::move(working), resident, world);
}

Value move_parts(worker::Request const& request, std::unique_ptr<Registration>& resident,
                 Communicator const& world) {
  std::int64_t const registration = request.parameters.at("registration").as_int();
  std::int64_t const state = request.parameters.at("state").as_int();
  if (!resident || resident->registration != registration)
    throw worker::Failure("invalid_request",
                          "The TIOGA registration is not resident in this worker session");
  if (resident->state != state)
    throw worker::Failure("invalid_request",
                          "The TIOGA registration moved since the supplied assembly state");
  auto const input = request.input();
  auto const& moved = input.require("moved_parts", DType::int64, {-1});
  auto const* part = moved.data<std::int64_t>();
  std::int64_t const parts = static_cast<std::int64_t>(resident->part_nodes.size());
  std::vector<std::int64_t> starts(static_cast<std::size_t>(parts), -1);
  std::int64_t nodes = 0;
  for (std::uint64_t index = 0; index < moved.count(); ++index) {
    if (part[index] < 0 || part[index] >= parts || (index > 0 && part[index] <= part[index - 1]))
      throw std::invalid_argument("Moved TIOGA parts must be unique ascending part indices");
    starts[static_cast<std::size_t>(part[index])] = nodes;
    nodes += resident->part_nodes[static_cast<std::size_t>(part[index])];
  }
  if (moved.count() == 0) throw std::invalid_argument("A TIOGA motion requires moved parts");
  auto const& coordinates = input.require("coordinates", DType::float64, {nodes, 3});
  require_finite(coordinates);
  // Any failure from here on leaves no resident registration on every rank.
  std::unique_ptr<Registration> working = std::move(resident);
  if (!collectively(world, [&] {
        auto const* xyz = coordinates.data<double>();
        for (auto& block : working->blocks) {
          std::int64_t const start = starts[static_cast<std::size_t>(block->part)];
          if (start < 0) continue;
          std::copy(xyz + 3 * start, xyz + 3 * start + block->xyz.size(), block->xyz.begin());
          register_block(*working->assembler, *block);
        }
        working->assembler->profile();
      }))
    return Object{};
  return connect_and_report(request, std::move(working), resident, world);
}

std::string mpi_library() {
  char text[MPI_MAX_LIBRARY_VERSION_STRING];
  int length = 0;
  MPI_Get_library_version(text, &length);
  std::string library(text, static_cast<std::size_t>(length));
  auto const end = library.find_first_of(std::string("\r\n\0", 3));
  if (end != std::string::npos) library.resize(end);
  while (!library.empty() && (library.back() == ' ' || library.back() == ','))
    library.pop_back();
  return library;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Communicator world{MPI_COMM_WORLD, MPI_COMM_NULL, 0, 1};
  MPI_Comm_rank(world.comm, &world.rank);
  MPI_Comm_size(world.comm, &world.size);
  MPI_Comm_dup(world.comm, &world.library);
  Value const identity = Object{
      {"provider", "tioga"},
      {"revision", PHYDRAX_TIOGA_REVISION},
      {"node_global_ids", true},
      {"unique_query_ids", PHYDRAX_TIOGA_ENABLE_UNIQUEID != 0},
      {"mpi_library", mpi_library()},
      {"mpi_standard", std::to_string(MPI_VERSION) + "." + std::to_string(MPI_SUBVERSION)},
      {"distribution", "whole-parts-round-robin"},
      {"operations", Array{"move", "register"}},
  };
  std::unique_ptr<Registration> resident;
  int const status = worker::serve_collective(
      world.comm, identity, [&](worker::Request const& request) -> Value {
        if (request.operation == "register") return register_parts(request, resident, world);
        if (request.operation == "move") return move_parts(request, resident, world);
        throw worker::Failure("unsupported",
                              "Unknown TIOGA worker operation '" + request.operation + "'");
      });
  resident.reset();
  MPI_Comm_free(&world.library);
  MPI_Finalize();
  return status;
}
