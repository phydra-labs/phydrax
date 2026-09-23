// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Bounded public-API VoroCrust extraction bridge.
#include "MeshingVoronoiMesher.h"
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t hard_max_seeds = 40000000;
constexpr std::uint64_t hard_max_vertices = 10000000;
constexpr std::uint64_t hard_max_faces = 50000000;
constexpr std::uint64_t hard_max_connectivity = 500000000;
constexpr std::uint64_t hard_max_bytes = 4000000000;

std::uint64_t limit(char const* text, std::uint64_t hard, char const* name) {
    std::string value(text);
    std::size_t parsed = 0;
    auto result = std::stoull(value, &parsed);
    if (parsed != value.size() || result == 0 || result > hard)
        throw std::runtime_error(std::string("Invalid ") + name);
    return result;
}

std::uint64_t checked_add(
    std::uint64_t left, std::uint64_t right, std::uint64_t maximum,
    char const* name) {
    if (left > maximum || right > maximum - left)
        throw std::runtime_error(std::string(name) + " exceeds its bound");
    return left + right;
}

std::uint64_t bounded_combination(
    std::uint64_t count, std::uint64_t choose, std::uint64_t maximum) {
    if (count < choose) return 0;
    std::uint64_t result = 1;
    for (std::uint64_t factor = 1; factor <= choose; ++factor) {
        auto const numerator = count - choose + factor;
        if (result > maximum / numerator) return maximum + 1;
        result = result * numerator / factor;
        if (result > maximum) return maximum + 1;
    }
    return result;
}

std::vector<std::string> csv_fields(std::string const& line) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(field);
    if (!line.empty() && line.back() == ',') fields.emplace_back();
    return fields;
}

double real_field(std::string const& text, char const* name) {
    std::istringstream input(text);
    double value = 0.0;
    if (!(input >> value) || !std::isfinite(value))
        throw std::runtime_error(std::string("Invalid ") + name);
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error(std::string("Trailing data in ") + name);
    return value;
}

std::uint64_t integer_field(std::string const& text, char const* name) {
    std::istringstream input(text);
    long long value = -1;
    if (!(input >> value) || value < 0)
        throw std::runtime_error(std::string("Invalid ") + name);
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error(std::string("Trailing data in ") + name);
    return static_cast<std::uint64_t>(value);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout
            << "phydrax-vorocrust/1 vorocrust/" << PHYDRAX_VOROCRUST_REVISION
            << " source-sha256/" << PHYDRAX_VOROCRUST_SOURCE_SHA256
            << " config-sha256/" << PHYDRAX_VOROCRUST_CONFIG_SHA256
            << " library-sha256/" << PHYDRAX_VOROCRUST_LIBRARY_SHA256 << '\n';
        return 0;
    }
    if (argc != 8) {
        std::cerr
            << "usage: phydrax-vorocrust seeds.csv output.mesh "
               "MAX_SEEDS MAX_VERTICES MAX_FACES MAX_CONNECTIVITY MAX_BYTES\n";
        return 2;
    }
    try {
        auto const max_seeds = limit(argv[3], hard_max_seeds, "seed limit");
        auto const max_vertices =
            limit(argv[4], hard_max_vertices, "vertex limit");
        auto const max_faces = limit(argv[5], hard_max_faces, "face limit");
        auto const max_connectivity =
            limit(argv[6], hard_max_connectivity, "connectivity limit");
        auto const max_bytes = limit(argv[7], hard_max_bytes, "byte limit");
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open VoroCrust seeds");
        auto const file_size = input.tellg();
        if (file_size < 0 || static_cast<std::uint64_t>(file_size) > max_bytes)
            throw std::runtime_error("VoroCrust seeds exceed the byte limit");
        input.seekg(0);
        std::string line;
        if (!std::getline(input, line))
            throw std::runtime_error("VoroCrust seeds are empty");
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line != "x1coord, x2coord, x3coord, radius")
            throw std::runtime_error("Unexpected VoroCrust seed CSV header");
        std::vector<double> seeds, sizing;
        std::vector<std::size_t> regions;
        std::vector<std::uint64_t> raw_regions;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (regions.size() >= max_seeds)
                throw std::runtime_error("VoroCrust seed count exceeds its bound");
            auto const fields = csv_fields(line);
            if (fields.size() != 5)
                throw std::runtime_error("VoroCrust seed rows require five columns");
            auto const x = real_field(fields[0], "seed x");
            auto const y = real_field(fields[1], "seed y");
            auto const z = real_field(fields[2], "seed z");
            auto const radius = real_field(fields[3], "seed radius");
            auto const region = integer_field(fields[4], "seed region");
            if (radius <= 0)
                throw std::runtime_error("VoroCrust seed radius must be positive");
            seeds.insert(seeds.end(), {x, y, z});
            sizing.push_back(radius);
            raw_regions.push_back(region);
            regions.push_back(static_cast<std::size_t>(region));
        }
        if (regions.empty())
            throw std::runtime_error("VoroCrust requires at least one seed");
        if (regions.size() >
            std::numeric_limits<std::size_t>::max() / 3)
            throw std::runtime_error("VoroCrust seed coordinate count overflows");
        for (auto region : raw_regions)
            if (region > regions.size())
                throw std::runtime_error("VoroCrust seed region is out of range");

        std::size_t vertex_count = 0, face_count = 0;
        double* vertices = nullptr;
        auto const vertex_bound =
            bounded_combination(regions.size(), 4, max_vertices);
        auto const face_bound =
            bounded_combination(regions.size(), 2, max_faces);
        auto const face_width =
            std::max<std::uint64_t>(regions.size() - std::min<std::size_t>(regions.size(), 2), 3);
        if (vertex_bound > max_vertices || face_bound > max_faces ||
            face_bound > max_connectivity / face_width)
            throw std::runtime_error(
                "Conservative VoroCrust output bounds exceed configured limits");
        std::size_t** faces = nullptr;
        MeshingVoronoiMesher mesher;
        int status = mesher.generate_3d_voronoi_mesh(
            1, regions.size(), seeds.data(), regions.data(), sizing.data(),
            vertex_count, vertices, face_count, faces);
        auto cleanup = [&]() {
            if (faces != nullptr) {
                for (std::size_t index = 0; index < face_count; ++index)
                    delete[] faces[index];
                delete[] faces;
                faces = nullptr;
            }
            delete[] vertices;
            vertices = nullptr;
        };
        if (status != 0 || vertex_count == 0 || face_count == 0) {
            cleanup();
            throw std::runtime_error("VoroCrust extraction failed");
        }
        try {
            if (vertex_count > max_vertices || face_count > max_faces)
                throw std::runtime_error(
                    "VoroCrust output exceeds entity bounds");
            std::uint64_t connectivity = 0;
            for (std::size_t index = 0; index < 3 * vertex_count; ++index)
                if (!std::isfinite(vertices[index]))
                    throw std::runtime_error(
                        "VoroCrust output vertex is nonfinite");
            for (std::size_t face = 0; face < face_count; ++face) {
                auto const count = faces[face][0];
                if (count < 3)
                    throw std::runtime_error("VoroCrust face has invalid arity");
                connectivity = checked_add(
                    connectivity, static_cast<std::uint64_t>(count),
                    max_connectivity, "VoroCrust connectivity");
                for (std::size_t offset = 1; offset <= count; ++offset)
                    if (faces[face][offset] >= vertex_count)
                        throw std::runtime_error(
                            "VoroCrust face vertex is out of range");
                if (faces[face][count + 1] >= regions.size() ||
                    faces[face][count + 2] >= regions.size())
                    throw std::runtime_error(
                        "VoroCrust face seed is out of range");
            }
            std::uint64_t scalar_count = 4096;
            scalar_count = checked_add(
                scalar_count, 3 * static_cast<std::uint64_t>(vertex_count),
                max_bytes / 32, "VoroCrust output scalars");
            scalar_count = checked_add(
                scalar_count, 4 * static_cast<std::uint64_t>(regions.size()),
                max_bytes / 32, "VoroCrust output scalars");
            scalar_count = checked_add(
                scalar_count, connectivity + 3 * face_count,
                max_bytes / 32, "VoroCrust output scalars");
            std::ofstream output(argv[2], std::ios::binary);
            if (!output)
                throw std::runtime_error("Cannot open VoroCrust output");
            output << std::setprecision(17);
            output << vertex_count << ' ' << face_count << ' '
                   << regions.size() << '\n';
            for (std::size_t index = 0; index < vertex_count; ++index)
                output << vertices[3 * index] << ' '
                       << vertices[3 * index + 1] << ' '
                       << vertices[3 * index + 2] << '\n';
            for (std::size_t index = 0; index < regions.size(); ++index)
                output << regions[index] << ' ' << seeds[3 * index] << ' '
                       << seeds[3 * index + 1] << ' '
                       << seeds[3 * index + 2] << '\n';
            for (std::size_t face = 0; face < face_count; ++face) {
                auto const count = faces[face][0];
                output << count;
                for (std::size_t offset = 1; offset <= count + 2; ++offset)
                    output << ' ' << faces[face][offset];
                output << '\n';
            }
            auto const written = output.tellp();
            if (written < 0 ||
                static_cast<std::uint64_t>(written) > max_bytes)
                throw std::runtime_error(
                    "VoroCrust output exceeds its byte bound");
            output.close();
            if (!output)
                throw std::runtime_error("Cannot finish VoroCrust output");
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "phydrax-vorocrust: " << error.what() << '\n';
        return 1;
    }
}
