// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Uses the public VoroCrust API; no meshing algorithms are duplicated here.
#include "MeshingVoronoiMesher.h"
#include "Version.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << GIT_COMMIT_SHA1 << '\n';
        return 0;
    }
    if (argc != 3) {
        std::cerr << "usage: phydrax-vorocrust seeds.csv output.mesh\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    std::string line;
    if (!std::getline(input, line)) return 3;
    std::vector<double> seeds, sizing;
    std::vector<size_t> regions;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double x, y, z, radius;
        long long region;
        if (!(row >> x >> y >> z >> radius >> region) || region < 0 ||
            !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
            !std::isfinite(radius) || radius <= 0) return 4;
        seeds.insert(seeds.end(), {x, y, z});
        sizing.push_back(radius);
        regions.push_back(static_cast<size_t>(region));
    }
    if (regions.empty()) return 5;
    size_t vertex_count = 0, face_count = 0;
    double* vertices = nullptr;
    size_t** faces = nullptr;
    MeshingVoronoiMesher mesher;
    int status = mesher.generate_3d_voronoi_mesh(
        1, regions.size(), seeds.data(), regions.data(), sizing.data(),
        vertex_count, vertices, face_count, faces);
    if (status != 0 || vertex_count == 0 || face_count == 0) return 6;
    std::ofstream output(argv[2]);
    output << std::setprecision(17);
    output << vertex_count << ' ' << face_count << ' ' << regions.size() << '\n';
    for (size_t i = 0; i < vertex_count; ++i)
        output << vertices[3*i] << ' ' << vertices[3*i+1] << ' ' << vertices[3*i+2] << '\n';
    for (size_t i = 0; i < regions.size(); ++i)
        output << regions[i] << ' ' << seeds[3*i] << ' ' << seeds[3*i+1] << ' ' << seeds[3*i+2] << '\n';
    for (size_t i = 0; i < face_count; ++i) {
        const size_t count = faces[i][0];
        output << count;
        // Public API appends the two adjacent seed IDs after each polygon loop.
        for (size_t j = 1; j <= count + 2; ++j) output << ' ' << faces[i][j];
        output << '\n';
        delete[] faces[i];
    }
    delete[] faces;
    delete[] vertices;
    return output.good() ? 0 : 7;
}
