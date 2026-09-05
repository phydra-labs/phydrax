// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Process-isolated, native-endian protocol. Mesh parts are assigned round-robin
// to MPI ranks; only the owning rank registers and stores each whole part.
#include <mpi.h>
#include <tioga.h>

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
template<class T> T scalar(std::istream& input) {
    T value;
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("Truncated TIOGA input");
    return value;
}
template<class T> std::vector<T> array(std::istream& input, std::size_t count) {
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!input) throw std::runtime_error("Truncated TIOGA input array");
    return values;
}
template<class T> void emit(std::ostream& output, T value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template<class T> void emit(std::ostream& output, const std::vector<T>& values) {
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}
int count(std::istream& input, bool empty = false) {
    int value = scalar<int>(input);
    if (value < (empty ? 0 : 1)) throw std::runtime_error("Invalid input count");
    return value;
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
    std::ifstream input(input_path, std::ios::binary);
    char magic[8];
    input.read(magic, 8);
    if (!input || std::memcmp(magic, "PXTIOGA1", 8))
        throw std::runtime_error("Invalid TIOGA input protocol");
    const int parts = count(input);
    if (size > parts) throw std::runtime_error("MPI ranks exceed mesh parts");
    int fringe = count(input), exclude = count(input, true);
    std::vector<Block> blocks;
    blocks.reserve((parts + size - 1) / size);
    for (int part = 0; part < parts; ++part) {
        const auto bytes = scalar<std::uint64_t>(input);
        const auto start = input.tellg();
        if (part % size != rank) {
            input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
            if (!input) throw std::runtime_error("Truncated skipped part");
            continue;
        }
        blocks.emplace_back();
        Block& block = blocks.back();
        block.part = part;
        const int nodes = count(input), types = count(input);
        const int walls = count(input, true), overset = count(input, true);
        block.xyz = array<double>(input, 3ull * nodes);
        block.nodes = array<std::uint64_t>(input, nodes);
        block.walls = array<int>(input, walls);
        block.overset = array<int>(input, overset);
        const auto node_offset = scalar<std::uint64_t>(input);
        block.arities.resize(types);
        block.counts.resize(types);
        // TIOGA de-duplicates search queries by global node ID across meshes.
        // Namespace native IDs while retaining original part-scoped IDs above.
        block.search_nodes.resize(nodes);
        for (int node = 0; node < nodes; ++node) block.search_nodes[node] = node_offset + node;
        block.connectivity.resize(types);
        block.routes.resize(types);
        for (int type = 0; type < types; ++type) {
            const int arity = count(input), cells = count(input);
            if (arity != 4 && arity != 5 && arity != 6 && arity != 8)
                throw std::runtime_error("Unsupported TIOGA cell arity");
            block.arities[type] = arity;
            block.counts[type] = cells;
            block.connectivity[type] = array<int>(input, static_cast<std::size_t>(arity) * cells);
            block.routes[type] = block.connectivity[type].data();
            for (int node : block.connectivity[type])
                if (node < 1 || node > nodes) throw std::runtime_error("Invalid connectivity index");
            const auto offset = block.cells.size();
            block.cells.resize(offset + cells);
            input.read(reinterpret_cast<char*>(block.cells.data() + offset), sizeof(std::uint64_t) * cells);
            if (!input) throw std::runtime_error("Truncated TIOGA cell IDs");
        }
        for (int node : block.walls)
            if (node < 1 || node > nodes) throw std::runtime_error("Invalid wall index");
        for (int node : block.overset)
            if (node < 1 || node > nodes) throw std::runtime_error("Invalid overset index");
        if (block.cells.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Too many cells for TIOGA");
        block.node_blank.assign(nodes, 1);
        block.cell_blank.assign(block.cells.size(), 1);
        if (input.tellg() - start != static_cast<std::streamoff>(bytes))
            throw std::runtime_error("Invalid part payload size");
    }
    // Destroy TIOGA before any registered caller-owned arrays and before MPI_Finalize.
    TIOGA::tioga assembler;
    assembler.setCommunicator(MPI_COMM_WORLD, rank, size);
    assembler.setNfringe(&fringe);
    assembler.setMexclude(&exclude);
    for (Block& block : blocks) {
        const int tag = block.part + 1;
        assembler.registerGridData(tag, block.nodes.size(), block.xyz.data(), block.node_blank.data(),
            block.walls.size(), block.overset.size(), block.walls.data(), block.overset.data(),
            block.arities.size(), block.arities.data(), block.counts.data(), block.routes.data(),
            block.cells.data(), block.search_nodes.data());
        assembler.set_cell_iblank(tag, block.cell_blank.data());
    }
    assembler.profile();
    assembler.performConnectivity();
    std::ofstream output(std::string(output_prefix) + "." + std::to_string(rank), std::ios::binary);
    output.write("PXTIOGR1", 8);
    emit(output, static_cast<int>(blocks.size()));
    for (Block& block : blocks) {
        emit(output, block.part);
        emit(output, static_cast<int>(block.nodes.size()));
        emit(output, static_cast<int>(block.cells.size()));
        emit(output, block.node_blank);
        emit(output, block.cell_blank);
        int donors = 0, fractions = 0;
        assembler.getDonorCount(block.part + 1, &donors, &fractions);
        std::vector<int> receptors(4ull * donors), indices(fractions);
        std::vector<double> weights(fractions);
        assembler.getDonorInfo(block.part + 1, receptors.data(), indices.data(), weights.data(), &donors);
        emit(output, donors);
        std::size_t offset = 0;
        for (int donor = 0; donor < donors; ++donor) {
            const int receptor_rank = receptors[4 * donor];
            const int receptor_node = receptors[4 * donor + 1];
            const int receptor_block = receptors[4 * donor + 2];
            const int width = receptors[4 * donor + 3];
            const int receptor_part = receptor_block * size + receptor_rank;
            if (receptor_part < 0 || receptor_part >= parts || width <= 0 ||
                offset + width >= indices.size()) throw std::runtime_error("Invalid native donor record");
            const int cell = indices[offset + width];
            if (cell < 0 || static_cast<std::size_t>(cell) >= block.cells.size())
                throw std::runtime_error("Invalid native donor cell");
            emit(output, receptor_part);
            emit(output, receptor_node);
            emit(output, width);
            emit(output, block.cells[cell]);
            for (int j = 0; j < width; ++j) {
                const int node = indices[offset + j];
                if (node < 0 || static_cast<std::size_t>(node) >= block.nodes.size())
                    throw std::runtime_error("Invalid native donor node");
                emit(output, block.nodes[node]);
                emit(output, weights[offset + j]);
            }
            offset += width + 1;
        }
        if (offset != indices.size()) throw std::runtime_error("Invalid native donor extent");
    }
    output.close();
    if (!output) throw std::runtime_error("Cannot write TIOGA output");
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "phydrax-tioga/1 tioga/" << PHYDRAX_TIOGA_REVISION << '\n';
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
