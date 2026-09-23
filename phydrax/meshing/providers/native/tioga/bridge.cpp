// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Process-isolated, little-endian protocol. Complete mesh parts are assigned
// round-robin to MPI ranks; only the owning rank stores and registers a part.
#include <mpi.h>
#include <tioga.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(int) == 4 && sizeof(double) == 8, "Unsupported TIOGA ABI");
static_assert(BASE == 1, "This bridge expects upstream one-based connectivity");

namespace {
constexpr std::uint64_t hard_max_vertices = 10000000;
constexpr std::uint64_t hard_max_cells = 20000000;
constexpr std::uint64_t hard_max_connectivity = 500000000;
constexpr std::uint64_t hard_max_bytes = 4000000000;

bool host_is_little_endian() {
    std::uint16_t value = 1;
    return *reinterpret_cast<unsigned char*>(&value) == 1;
}

std::uint64_t checked_product(
    std::uint64_t left, std::uint64_t right, std::uint64_t maximum,
    char const* name) {
    if (left != 0 && right > maximum / left)
        throw std::runtime_error(std::string(name) + " exceeds its bound");
    return left * right;
}

std::uint64_t checked_add(
    std::uint64_t left, std::uint64_t right, std::uint64_t maximum,
    char const* name) {
    if (right > maximum - left)
        throw std::runtime_error(std::string(name) + " exceeds its bound");
    return left + right;
}

template<class T> T scalar(std::istream& input) {
    unsigned char bytes[sizeof(T)];
    input.read(reinterpret_cast<char*>(bytes), sizeof(T));
    if (!input) throw std::runtime_error("Truncated TIOGA input");
    if (!host_is_little_endian()) std::reverse(bytes, bytes + sizeof(T));
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

template<class T>
std::vector<T> array(
    std::istream& input, std::uint64_t count, std::uint64_t maximum,
    char const* name) {
    if (count > maximum ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error(std::string(name) + " count exceeds its bound");
    std::vector<T> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = scalar<T>(input);
    return values;
}

template<class T> void emit(std::ostream& output, T value) {
    unsigned char bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    if (!host_is_little_endian()) std::reverse(bytes, bytes + sizeof(T));
    output.write(reinterpret_cast<char*>(bytes), sizeof(T));
}

template<class T>
void emit(std::ostream& output, const std::vector<T>& values) {
    for (auto value : values) emit(output, value);
}

int count(
    std::istream& input, int maximum, bool empty = false,
    char const* name = "input count") {
    int value = scalar<std::int32_t>(input);
    if (value < (empty ? 0 : 1) || value > maximum)
        throw std::runtime_error(std::string("Invalid ") + name);
    return value;
}

void enforce_output_bound(std::ostream& output, std::uint64_t maximum) {
    auto const position = output.tellp();
    if (position < 0 || static_cast<std::uint64_t>(position) > maximum)
        throw std::runtime_error("TIOGA output exceeds its byte bound");
}

struct Block {
    int part;
    std::vector<double> xyz;
    std::vector<std::uint64_t> nodes, cells, search_nodes;
    std::vector<int> node_blank, cell_blank, walls, overset, arities, counts;
    std::vector<std::vector<int>> connectivity;
    std::vector<int*> routes;
};

void assemble(const char* input_path, const char* output_prefix, int rank, int size) {
    std::ifstream input(input_path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open TIOGA input");
    auto const file_size = input.tellg();
    if (file_size < 0 || static_cast<std::uint64_t>(file_size) > hard_max_bytes)
        throw std::runtime_error("TIOGA input exceeds the hard byte bound");
    input.seekg(0);
    char magic[8];
    input.read(magic, 8);
    if (!input || std::memcmp(magic, "PXTIOGA2", 8))
        throw std::runtime_error("Invalid TIOGA input protocol");
    const int parts = count(
        input, static_cast<int>(hard_max_vertices), false, "mesh part count");
    if (size > parts) throw std::runtime_error("MPI ranks exceed mesh parts");
    int fringe = count(input, std::numeric_limits<int>::max(), false, "fringe count");
    int exclude = count(
        input, std::numeric_limits<int>::max(), true, "exclusion count");
    auto const max_vertices = scalar<std::uint64_t>(input);
    auto const max_cells = scalar<std::uint64_t>(input);
    auto const max_connectivity = scalar<std::uint64_t>(input);
    auto const max_bytes = scalar<std::uint64_t>(input);
    if (max_vertices == 0 || max_vertices > hard_max_vertices ||
        max_cells == 0 || max_cells > hard_max_cells ||
        max_connectivity == 0 || max_connectivity > hard_max_connectivity ||
        max_bytes == 0 || max_bytes > hard_max_bytes ||
        static_cast<std::uint64_t>(file_size) > max_bytes)
        throw std::runtime_error("Invalid TIOGA resource limits");
    auto part_nodes = array<std::uint64_t>(
        input, static_cast<std::uint64_t>(parts), max_vertices,
        "part node inventory");
    std::uint64_t declared_nodes = 0;
    for (auto nodes : part_nodes) {
        if (nodes == 0 || nodes > max_vertices)
            throw std::runtime_error("Invalid TIOGA part node count");
        declared_nodes = checked_add(
            declared_nodes, nodes, max_vertices, "total node count");
    }
    std::vector<Block> blocks;
    blocks.reserve(
        static_cast<std::size_t>((parts - 1) / size + 1));
    std::uint64_t total_cells = 0;
    std::uint64_t total_connectivity = 0;
    for (int part = 0; part < parts; ++part) {
        const auto bytes = scalar<std::uint64_t>(input);
        if (bytes > max_bytes)
            throw std::runtime_error("TIOGA part payload exceeds byte bound");
        const auto start = input.tellg();
        if (start < 0)
            throw std::runtime_error("Invalid TIOGA part payload position");
        if (part % size != rank) {
            input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
            if (!input) throw std::runtime_error("Truncated skipped part");
            continue;
        }
        blocks.emplace_back();
        Block& block = blocks.back();
        block.part = part;
        const int nodes = count(
            input, static_cast<int>(max_vertices), false, "node count");
        if (static_cast<std::uint64_t>(nodes) != part_nodes[part])
            throw std::runtime_error("Part node count differs from header");
        const int types = count(
            input, static_cast<int>(max_cells), false, "cell type count");
        const int walls = count(input, nodes, true, "wall count");
        const int overset = count(input, nodes, true, "overset count");
        block.xyz = array<double>(
            input,
            checked_product(3, static_cast<std::uint64_t>(nodes),
                            max_connectivity, "coordinate count"),
            max_connectivity, "coordinates");
        for (double coordinate : block.xyz)
            if (!std::isfinite(coordinate))
                throw std::runtime_error("TIOGA coordinates must be finite");
        block.nodes = array<std::uint64_t>(
            input, static_cast<std::uint64_t>(nodes), max_vertices, "node IDs");
        block.walls = array<int>(
            input, static_cast<std::uint64_t>(walls), max_vertices, "walls");
        block.overset = array<int>(
            input, static_cast<std::uint64_t>(overset), max_vertices, "overset");
        const auto node_offset = scalar<std::uint64_t>(input);
        if (nodes > 0 &&
            node_offset > std::numeric_limits<std::uint64_t>::max() -
                              static_cast<std::uint64_t>(nodes - 1))
            throw std::runtime_error("TIOGA node namespace overflows uint64");
        block.arities.resize(static_cast<std::size_t>(types));
        block.counts.resize(static_cast<std::size_t>(types));
        block.search_nodes.resize(static_cast<std::size_t>(nodes));
        for (int node = 0; node < nodes; ++node)
            block.search_nodes[static_cast<std::size_t>(node)] =
                node_offset + static_cast<std::uint64_t>(node);
        block.connectivity.resize(static_cast<std::size_t>(types));
        block.routes.resize(static_cast<std::size_t>(types));
        for (int type = 0; type < types; ++type) {
            const int arity = count(input, 8, false, "cell arity");
            const int cells = count(
                input, static_cast<int>(max_cells), false, "cell count");
            if (arity != 4 && arity != 5 && arity != 6 && arity != 8)
                throw std::runtime_error("Unsupported TIOGA cell arity");
            total_cells = checked_add(
                total_cells, static_cast<std::uint64_t>(cells),
                max_cells, "total cell count");
            auto const entries = checked_product(
                static_cast<std::uint64_t>(arity),
                static_cast<std::uint64_t>(cells),
                max_connectivity, "connectivity count");
            total_connectivity = checked_add(
                total_connectivity, entries, max_connectivity,
                "total connectivity");
            block.arities[static_cast<std::size_t>(type)] = arity;
            block.counts[static_cast<std::size_t>(type)] = cells;
            block.connectivity[static_cast<std::size_t>(type)] =
                array<int>(
                    input, entries, max_connectivity, "connectivity");
            block.routes[static_cast<std::size_t>(type)] =
                block.connectivity[static_cast<std::size_t>(type)].data();
            for (int node : block.connectivity[static_cast<std::size_t>(type)])
                if (node < 1 || node > nodes)
                    throw std::runtime_error("Invalid connectivity index");
            const auto offset = block.cells.size();
            if (static_cast<std::uint64_t>(offset) >
                max_cells - static_cast<std::uint64_t>(cells))
                throw std::runtime_error("Cell count exceeds its bound");
            auto identifiers = array<std::uint64_t>(
                input, static_cast<std::uint64_t>(cells), max_cells,
                "cell IDs");
            block.cells.insert(
                block.cells.end(), identifiers.begin(), identifiers.end());
        }
        for (int node : block.walls)
            if (node < 1 || node > nodes)
                throw std::runtime_error("Invalid wall index");
        for (int node : block.overset)
            if (node < 1 || node > nodes)
                throw std::runtime_error("Invalid overset index");
        block.node_blank.assign(static_cast<std::size_t>(nodes), 1);
        block.cell_blank.assign(block.cells.size(), 1);
        auto const end = input.tellg();
        if (end < start ||
            static_cast<std::uint64_t>(end - start) != bytes)
            throw std::runtime_error("Invalid part payload size");
    }
    input.peek();
    if (!input.eof())
        throw std::runtime_error("Unexpected trailing TIOGA input");

    TIOGA::tioga assembler;
    assembler.setCommunicator(MPI_COMM_WORLD, rank, size);
    assembler.setNfringe(&fringe);
    assembler.setMexclude(&exclude);
    for (Block& block : blocks) {
        const int tag = block.part + 1;
        assembler.registerGridData(
            tag, static_cast<int>(block.nodes.size()), block.xyz.data(),
            block.node_blank.data(), static_cast<int>(block.walls.size()),
            static_cast<int>(block.overset.size()), block.walls.data(),
            block.overset.data(), static_cast<int>(block.arities.size()),
            block.arities.data(), block.counts.data(), block.routes.data(),
            block.cells.data(), block.search_nodes.data());
        assembler.set_cell_iblank(tag, block.cell_blank.data());
    }
    assembler.profile();
    assembler.performConnectivity();
    auto const output_byte_cap =
        max_bytes / static_cast<std::uint64_t>(size);
    if (output_byte_cap < 8)
        throw std::runtime_error("TIOGA per-rank output byte bound is too small");

    std::ofstream output(
        std::string(output_prefix) + "." + std::to_string(rank),
        std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open TIOGA output");
    output.write("PXTIOGR2", 8);
    emit(output, static_cast<std::int32_t>(blocks.size()));
    for (Block& block : blocks) {
        emit(output, static_cast<std::int32_t>(block.part));
        emit(output, static_cast<std::int32_t>(block.nodes.size()));
        emit(output, static_cast<std::int32_t>(block.cells.size()));
        emit(output, block.node_blank);
        emit(output, block.cell_blank);
        int donors = 0, fractions = 0;
        assembler.getDonorCount(block.part + 1, &donors, &fractions);
        if (donors < 0 || fractions < 0 ||
            static_cast<std::uint64_t>(donors) > max_vertices ||
            static_cast<std::uint64_t>(fractions) > max_connectivity)
            throw std::runtime_error("Invalid native donor counts");
        auto const receptor_values = checked_product(
            4, static_cast<std::uint64_t>(donors),
            max_connectivity, "native receptor count");
        auto receptors = std::vector<int>(
            static_cast<std::size_t>(receptor_values));
        auto indices = std::vector<int>(static_cast<std::size_t>(fractions));
        auto weights = std::vector<double>(static_cast<std::size_t>(fractions));
        int returned_donors = donors;
        assembler.getDonorInfo(
            block.part + 1, receptors.data(), indices.data(),
            weights.data(), &returned_donors);
        if (returned_donors < 0 || returned_donors > donors)
            throw std::runtime_error("Native donor count changed beyond allocation");
        donors = returned_donors;
        emit(output, static_cast<std::int32_t>(donors));
        std::size_t offset = 0;
        for (int donor = 0; donor < donors; ++donor) {
            const int receptor_rank = receptors[4 * donor];
            const int receptor_node = receptors[4 * donor + 1];
            const int receptor_block = receptors[4 * donor + 2];
            const int width = receptors[4 * donor + 3];
            if (receptor_rank < 0 || receptor_rank >= size ||
                receptor_block < 0)
                throw std::runtime_error("Invalid native receptor rank/block");
            auto const combined =
                static_cast<std::int64_t>(receptor_block) * size +
                receptor_rank;
            if (combined < 0 || combined >= parts)
                throw std::runtime_error("Invalid native receptor part");
            const int receptor_part = static_cast<int>(combined);
            if (receptor_node < 0 ||
                static_cast<std::uint64_t>(receptor_node) >=
                    part_nodes[static_cast<std::size_t>(receptor_part)])
                throw std::runtime_error("Invalid native receptor node");
            if (width <= 0 || offset >= indices.size() ||
                static_cast<std::size_t>(width) > indices.size() - offset - 1)
                throw std::runtime_error("Invalid native donor extent");
            const int cell = indices[offset + static_cast<std::size_t>(width)];
            if (cell < 0 ||
                static_cast<std::size_t>(cell) >= block.cells.size())
                throw std::runtime_error("Invalid native donor cell");
            emit(output, static_cast<std::int32_t>(receptor_part));
            emit(output, static_cast<std::int32_t>(receptor_node));
            emit(output, static_cast<std::int32_t>(width));
            emit(output, block.cells[static_cast<std::size_t>(cell)]);
            for (int index = 0; index < width; ++index) {
                const int node = indices[offset + static_cast<std::size_t>(index)];
                if (node < 0 ||
                    static_cast<std::size_t>(node) >= block.nodes.size())
                    throw std::runtime_error("Invalid native donor node");
                auto const weight =
                    weights[offset + static_cast<std::size_t>(index)];
                if (!std::isfinite(weight))
                    throw std::runtime_error("Nonfinite native donor weight");
                emit(output, block.nodes[static_cast<std::size_t>(node)]);
                emit(output, weight);
            }
            offset += static_cast<std::size_t>(width) + 1;
        }
        if (offset != indices.size())
            throw std::runtime_error("Invalid native donor extent");
        enforce_output_bound(output, output_byte_cap);
    }
    output.close();
    if (!output) throw std::runtime_error("Cannot write TIOGA output");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "phydrax-tioga/2 tioga/" << PHYDRAX_TIOGA_REVISION << '\n';
        return 0;
    }
    if (argc != 3) {
        std::cerr << "Usage: phydrax-tioga INPUT OUTPUT_PREFIX\n";
        return 2;
    }
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        assemble(argv[1], argv[2], rank, size);
    } catch (const std::exception& error) {
        std::cerr << "phydrax-tioga rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return 0;
}
