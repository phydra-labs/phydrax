// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Independent exact source: canonical tetra cut by [1/8,2/5]^3.
struct ReferenceTetraCutFixture {
  const std::array<int64_t, 40> coefficients{
      -1,0,0,0, 0,-1,0,0, 0,0,-1,0, 1,1,1,-1,
      -8,0,0,1, 5,0,0,-2, 0,-8,0,1, 0,5,0,-2, 0,0,-8,1, 0,0,5,-2};
  std::array<int64_t, 41> offsets{};
  std::array<int8_t, 40> signs{};
  std::vector<uint32_t> words;
  std::array<int32_t, 48> supports{}, face_vertices{};
  std::array<uint16_t, 16> incidence{};
  std::array<int32_t, 11> face_offsets{};
  std::array<int32_t, 10> face_planes{};
  int32_t vertices = 0, faces = 0;
  int64_t work = 0;

  ReferenceTetraCutFixture() {
    for (int scalar = 0; scalar < 40; ++scalar) {
      signs[scalar] = (coefficients[scalar] > 0) - (coefficients[scalar] < 0);
      if (coefficients[scalar]) words.push_back(static_cast<uint32_t>(std::abs(coefficients[scalar])));
      offsets[scalar + 1] = static_cast<int64_t>(words.size());
    }
  }
  int32_t run(int64_t scratch = 1 << 20, int64_t limit = 100000) {
    vertices = faces = 0;
    work = 0;
    return phx_mc_clip_reference_tetrahedron_exact(
        words.data(), words.size(), offsets.data(), signs.data(), scratch, limit,
        16, 10, 48, supports.data(), incidence.data(), face_offsets.data(),
        face_planes.data(), face_vertices.data(), &vertices, &faces, &work);
  }
};

int64_t reference_test_det(const std::array<int64_t, 3>& a,
                           const std::array<int64_t, 3>& b,
                           const std::array<int64_t, 3>& c) {
  return a[0] * (b[1] * c[2] - b[2] * c[1]) -
         a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

void test_exact_reference_tetrahedron_and_corner_closure() {
  ReferenceTetraCutFixture fixture;
  PHX_CHECK(fixture.run() == PHX_MC_OK);
  PHX_CHECK(fixture.vertices == 10 && fixture.faces == 7 && fixture.work > 0);
  PHX_CHECK(fixture.face_offsets[fixture.faces] == 30);
  std::vector<std::array<int64_t, 3>> points(fixture.vertices);
  for (int vertex = 0; vertex < fixture.vertices; ++vertex) {
    std::array<int64_t, 3> columns[4];
    for (int axis = 0; axis < 4; ++axis)
      for (int row = 0; row < 3; ++row)
        columns[axis][row] = fixture.coefficients[4 * fixture.supports[3 * vertex + row] + axis];
    const int64_t denominator = reference_test_det(columns[0], columns[1], columns[2]);
    const std::array<int64_t, 3> numerator{
        -reference_test_det(columns[3], columns[1], columns[2]),
        -reference_test_det(columns[0], columns[3], columns[2]),
        -reference_test_det(columns[0], columns[1], columns[3])};
    PHX_CHECK(denominator != 0);
    for (int axis = 0; axis < 3; ++axis) {
      PHX_CHECK((40 * numerator[axis]) % denominator == 0);
      points[vertex][axis] = 40 * numerator[axis] / denominator;
    }
    int incident = 0;
    for (int plane = 0; plane < 10; ++plane) {
      int64_t side = 40 * fixture.coefficients[4 * plane + 3];
      for (int axis = 0; axis < 3; ++axis) side += fixture.coefficients[4 * plane + axis] * points[vertex][axis];
      PHX_CHECK(side <= 0);
      PHX_CHECK((side == 0) == ((fixture.incidence[vertex] & (1u << plane)) != 0));
      incident += side == 0;
    }
    PHX_CHECK(incident == 3);
  }
  // Original tetra face x+y+z=1 is a genuine triangular cut face; no box proxy.
  const auto source_face = std::find(fixture.face_planes.begin(), fixture.face_planes.begin() + fixture.faces, 3);
  PHX_CHECK(source_face != fixture.face_planes.begin() + fixture.faces);
  const int face_row = static_cast<int>(source_face - fixture.face_planes.begin());
  PHX_CHECK(fixture.face_offsets[face_row + 1] - fixture.face_offsets[face_row] == 3);
  int64_t determinant_sum = 0;
  std::map<std::array<int64_t, 2>, int64_t> edge_rows;
  for (int face = 0; face < fixture.faces; ++face) {
    const int begin = fixture.face_offsets[face], end = fixture.face_offsets[face + 1];
    const auto& first = points[fixture.face_vertices[begin]];
    for (int slot = begin + 1; slot + 1 < end; ++slot)
      determinant_sum += reference_test_det(first, points[fixture.face_vertices[slot]], points[fixture.face_vertices[slot + 1]]);
    for (int slot = begin; slot < end; ++slot) {
      const int64_t a = fixture.face_vertices[slot], b = fixture.face_vertices[slot + 1 == end ? begin : slot + 1];
      edge_rows.emplace(std::array<int64_t, 2>{std::min(a,b), std::max(a,b)}, 0);
    }
  }
  // Independent volume: (11/40)^3 - (1/5)^3/6 = 3737/192000.
  PHX_CHECK(determinant_sum == 7474);
  std::vector<int64_t> edges;
  for (auto& [pair, row] : edge_rows) {
    row = static_cast<int64_t>(edges.size() / 2);
    edges.insert(edges.end(), pair.begin(), pair.end());
  }
  std::vector<int64_t> face_offsets(fixture.face_offsets.begin(), fixture.face_offsets.begin() + fixture.faces + 1);
  std::vector<int64_t> face_vertices(fixture.face_vertices.begin(), fixture.face_vertices.begin() + face_offsets.back());
  std::vector<int64_t> face_ids(fixture.faces), vertex_ids(fixture.vertices);
  std::iota(face_ids.begin(), face_ids.end(), 0);
  std::iota(vertex_ids.begin(), vertex_ids.end(), 0);
  std::vector<int8_t> orientations(fixture.vertices);
  for (int vertex = 0; vertex < fixture.vertices; ++vertex) {
    std::array<std::array<int64_t, 3>, 3> directions;
    int count = 0;
    for (int edge = 0; edge < static_cast<int>(edges.size() / 2); ++edge) {
      const auto a = edges[2 * edge], b = edges[2 * edge + 1];
      if (a != vertex && b != vertex) continue;
      PHX_CHECK(count < 3);
      const int64_t other = a == vertex ? b : a;
      for (int axis = 0; axis < 3; ++axis) directions[count][axis] = points[other][axis] - points[vertex][axis];
      ++count;
    }
    PHX_CHECK(count == 3);
    const int64_t determinant = reference_test_det(directions[0], directions[1], directions[2]);
    PHX_CHECK(determinant != 0);
    orientations[vertex] = determinant > 0 ? 1 : -1;
  }
  const int64_t cell_faces[2] = {0, fixture.faces}, cell_vertices[2] = {0, fixture.vertices};
  std::vector<int64_t> hexes(8 * fixture.vertices), parents(fixture.vertices);
  int64_t count = 0, work = 0, witness[4];
  auto run = [&](int64_t capacity, int64_t scratch, int64_t work_limit) {
    return phx_mc_polyhedron_corner_hexes(
        fixture.vertices, edges.size()/2, fixture.faces, 1,
        face_vertices.size(), fixture.faces, fixture.vertices,
        edges.data(), face_offsets.data(), face_vertices.data(), cell_faces, face_ids.data(),
        cell_vertices, vertex_ids.data(), orientations.data(),
        fixture.vertices + edges.size()/2 + fixture.faces + 1, capacity, scratch, work_limit,
        hexes.data(), parents.data(), &count, witness, &work);
  };
  PHX_CHECK(run(fixture.vertices, 1 << 20, 100000) == PHX_MC_OK && count == 10);
  for (auto parent : parents) PHX_CHECK(parent == 0);
  for (int vertex = 0; vertex < fixture.vertices; ++vertex) {
    std::array<int64_t, 8> nodes;
    std::copy(hexes.begin() + 8 * vertex, hexes.begin() + 8 * (vertex + 1), nodes.begin());
    std::sort(nodes.begin(), nodes.end());
    PHX_CHECK(std::adjacent_find(nodes.begin(), nodes.end()) == nodes.end());
  }
  PHX_CHECK(run(9, 1 << 20, 100000) == PHX_MC_CAPACITY_EXCEEDED && count == 0);
  PHX_CHECK(run(10, 16, 100000) == PHX_MC_CAPACITY_EXCEEDED && count == 0);
  PHX_CHECK(run(10, 1 << 20, 1) == PHX_MC_CAPACITY_EXCEEDED && count == 0);
  PHX_CHECK(fixture.run(32768) == PHX_MC_CAPACITY_EXCEEDED && fixture.vertices == 0 && fixture.faces == 0);
  PHX_CHECK(fixture.run(1 << 20, 1) == PHX_MC_CAPACITY_EXCEEDED && fixture.vertices == 0 && fixture.faces == 0);
}
