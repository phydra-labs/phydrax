// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Independent exact sources: the canonical tetra cut by 0..12 rational
// halfspaces. The oracle below uses only small-integer __int128 homogeneous
// arithmetic on the reduced rows; expected counts/volumes come from an
// independent Fraction vertex enumeration.
using ReferenceRows = std::vector<std::array<int64_t, 4>>;
using ReferenceWide = __int128;

ReferenceRows reference_rows(std::initializer_list<ReferenceRows> parts) {
  ReferenceRows rows{{-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {1, 1, 1, -1}};
  for (const auto& part : parts) rows.insert(rows.end(), part.begin(), part.end());
  return rows;
}

// [1/8,2/5]^3, the original ten-plane source.
const ReferenceRows kReferenceBox{{-8, 0, 0, 1}, {5, 0, 0, -2}, {0, -8, 0, 1},
                                  {0, 5, 0, -2}, {0, 0, -8, 1}, {0, 0, 5, -2}};
// A source cell C and a pulled-back target A^-1(t), both rational tetrahedra.
const ReferenceRows kReferenceCell{{-10, 0, 0, 1}, {0, -9, 0, 1}, {0, 0, -8, 1}, {2, 3, 4, -3}};
const ReferenceRows kReferenceTarget{{5, 1, -1, -3}, {-1, 6, 1, -3}, {1, -1, 7, -3}, {-3, -2, -2, 1}};
// [1/10,2/5]^3 with six chamfers; three output vertices carry four planes.
const ReferenceRows kReferenceChamfered{
    {-10, 0, 0, 1}, {5, 0, 0, -2}, {0, -10, 0, 1}, {0, 5, 0, -2}, {0, 0, -10, 1}, {0, 0, 5, -2},
    {10, 10, 0, -7}, {0, 10, 10, -7}, {10, 0, 10, -7}, {-5, -5, -5, 2}, {4, -4, 0, -1}, {0, 4, -4, -1}};

struct ReferenceClipFixture {
  ReferenceRows rows;
  std::vector<unsigned __int128> scales;
  std::vector<int64_t> coefficients;
  int32_t plane_count = 0;
  int32_t vertex_capacity = 0, face_capacity = 0, entry_capacity = 0;
  std::vector<int64_t> offsets;
  std::vector<int8_t> signs;
  std::vector<uint32_t> words;
  std::vector<int32_t> supports, face_vertices, face_offsets, face_planes;
  std::vector<uint16_t> incidence;
  int32_t vertices = 0, faces = 0;
  int64_t work = 0, limbs = 1;

  // Each plane row is multiplied by a positive scale before limb packing.
  explicit ReferenceClipFixture(ReferenceRows source, std::vector<unsigned __int128> multipliers = {})
      : rows(std::move(source)), scales(std::move(multipliers)),
        plane_count(static_cast<int32_t>(rows.size())), vertex_capacity(2 * plane_count - 4),
        face_capacity(plane_count), entry_capacity(6 * plane_count - 12),
        offsets(4 * rows.size() + 1, 0), signs(4 * rows.size(), 0),
        supports(3 * static_cast<std::size_t>(vertex_capacity), -1),
        face_vertices(static_cast<std::size_t>(entry_capacity), -1),
        face_offsets(static_cast<std::size_t>(face_capacity) + 1, -1),
        face_planes(static_cast<std::size_t>(face_capacity), -1),
        incidence(static_cast<std::size_t>(vertex_capacity), 0) {
    if (scales.empty()) scales.assign(rows.size(), 1);
    for (std::size_t plane = 0; plane < rows.size(); ++plane) {
      for (int axis = 0; axis < 4; ++axis) {
        const int64_t value = rows[plane][axis];
        const std::size_t scalar = 4 * plane + static_cast<std::size_t>(axis);
        coefficients.push_back(value);
        signs[scalar] = static_cast<int8_t>((value > 0) - (value < 0));
        auto magnitude = static_cast<unsigned __int128>(value < 0 ? -value : value) * scales[plane];
        const std::size_t begin = words.size();
        for (; magnitude != 0; magnitude >>= 32) words.push_back(static_cast<uint32_t>(magnitude));
        limbs = std::max<int64_t>(limbs, static_cast<int64_t>(words.size() - begin));
        offsets[scalar + 1] = static_cast<int64_t>(words.size());
      }
    }
  }
  // Native pre-allocation bound: (2F-4) * (2048 + 4096 M) bytes.
  int64_t resident() const { return (2 * static_cast<int64_t>(rows.size()) - 4) * (2048 + 4096 * limbs); }
  int32_t run(int64_t scratch = 1 << 20, int64_t limit = 10000000) {
    vertices = faces = 0;
    work = 0;
    const phx_mc_clip_reference_tetrahedron_exact_request request{
        .words = words.data(),
        .word_count = static_cast<int64_t>(words.size()),
        .offsets = offsets.data(),
        .signs = signs.data(),
        .scratch_limit = scratch,
        .work_limit = limit,
        .plane_count = plane_count,
        .vertex_capacity = vertex_capacity,
        .face_capacity = face_capacity,
        .entry_capacity = entry_capacity,
        .supporting_planes = supports.data(),
        .incident_planes = incidence.data(),
        .face_offsets = face_offsets.data(),
        .face_planes = face_planes.data(),
        .face_vertices = face_vertices.data(),
        .vertex_count = &vertices,
        .face_count = &faces,
        .work_units = &work,
    };
    return phx_mc_clip_reference_tetrahedron_exact(&request);
  }
  bool same_output(const ReferenceClipFixture& other) const {
    if (vertices != other.vertices || faces != other.faces) return false;
    const int entries = face_offsets[faces];
    return std::equal(supports.begin(), supports.begin() + 3 * vertices, other.supports.begin()) &&
           std::equal(incidence.begin(), incidence.begin() + vertices, other.incidence.begin()) &&
           std::equal(face_offsets.begin(), face_offsets.begin() + faces + 1, other.face_offsets.begin()) &&
           std::equal(face_planes.begin(), face_planes.begin() + faces, other.face_planes.begin()) &&
           std::equal(face_vertices.begin(), face_vertices.begin() + entries, other.face_vertices.begin());
  }
};

int64_t reference_test_det(const std::array<int64_t, 3>& a,
                           const std::array<int64_t, 3>& b,
                           const std::array<int64_t, 3>& c) {
  return a[0] * (b[1] * c[2] - b[2] * c[1]) -
         a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

ReferenceWide reference_wide_det(const std::array<ReferenceWide, 3>& a,
                                 const std::array<ReferenceWide, 3>& b,
                                 const std::array<ReferenceWide, 3>& c) {
  return a[0] * (b[1] * c[2] - b[2] * c[1]) -
         a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

// Homogeneous source point (N, D), D > 0, solved from three reduced rows.
struct ReferencePoint {
  std::array<ReferenceWide, 3> numerator{};
  ReferenceWide denominator = 0;
};

ReferencePoint reference_point(const ReferenceRows& rows, const int32_t* support) {
  std::array<ReferenceWide, 3> columns[4];
  for (int axis = 0; axis < 4; ++axis)
    for (int row = 0; row < 3; ++row) columns[axis][row] = rows[support[row]][axis];
  ReferencePoint point;
  point.denominator = reference_wide_det(columns[0], columns[1], columns[2]);
  point.numerator = {-reference_wide_det(columns[3], columns[1], columns[2]),
                     -reference_wide_det(columns[0], columns[3], columns[2]),
                     -reference_wide_det(columns[0], columns[1], columns[3])};
  if (point.denominator < 0) {
    point.denominator = -point.denominator;
    for (auto& value : point.numerator) value = -value;
  }
  return point;
}

ReferenceWide reference_side(const std::array<int64_t, 4>& plane, const ReferencePoint& point) {
  return plane[0] * point.numerator[0] + plane[1] * point.numerator[1] +
         plane[2] * point.numerator[2] + plane[3] * point.denominator;
}

// Exact closed outward convex polytope whose vertex count, facet labels/sizes
// and L-scaled six-volume equal the independent enumeration.
void check_reference_clip(const ReferenceClipFixture& fixture, int32_t expected_vertices,
                          const std::vector<std::array<int32_t, 2>>& expected_faces,
                          int64_t scale, ReferenceWide expected_six_volume) {
  const auto& rows = fixture.rows;
  const int planes = static_cast<int>(rows.size());
  PHX_CHECK(fixture.vertices == expected_vertices);
  PHX_CHECK(fixture.faces == static_cast<int32_t>(expected_faces.size()));
  if (fixture.vertices != expected_vertices || fixture.faces != static_cast<int32_t>(expected_faces.size())) return;
  PHX_CHECK(fixture.vertices <= 2 * planes - 4 && fixture.faces <= planes && fixture.work > 0);
  std::vector<ReferencePoint> points(fixture.vertices);
  for (int vertex = 0; vertex < fixture.vertices; ++vertex) {
    const int32_t* support = fixture.supports.data() + 3 * vertex;
    PHX_CHECK(support[0] != support[1] && support[0] != support[2] && support[1] != support[2]);
    for (int slot = 0; slot < 3; ++slot) PHX_CHECK(support[slot] >= 0 && support[slot] < planes);
    points[vertex] = reference_point(rows, support);
    PHX_CHECK(points[vertex].denominator > 0);
    PHX_CHECK((fixture.incidence[vertex] >> planes) == 0);
    int incident = 0;
    for (int plane = 0; plane < planes; ++plane) {
      const ReferenceWide side = reference_side(rows[plane], points[vertex]);
      PHX_CHECK(side <= 0);
      PHX_CHECK((side == 0) == ((fixture.incidence[vertex] & (1u << plane)) != 0));
      incident += side == 0;
    }
    PHX_CHECK(incident >= 3);
    for (int slot = 0; slot < 3; ++slot) PHX_CHECK(fixture.incidence[vertex] & (1u << support[slot]));
    for (int other = 0; other < vertex; ++other) {
      bool same = true;
      for (int axis = 0; axis < 3; ++axis)
        same &= points[vertex].numerator[axis] * points[other].denominator ==
                points[other].numerator[axis] * points[vertex].denominator;
      PHX_CHECK(!same);
    }
  }
  PHX_CHECK(fixture.face_offsets[0] == 0);
  std::vector<std::array<int32_t, 2>> labels;
  std::map<std::array<int32_t, 2>, int> directed;
  std::vector<int> used(fixture.vertices, 0);
  for (int face = 0; face < fixture.faces; ++face) {
    const int begin = fixture.face_offsets[face], end = fixture.face_offsets[face + 1];
    const int32_t plane = fixture.face_planes[face];
    PHX_CHECK(end - begin >= 3 && plane >= 0 && plane < planes);
    labels.push_back({plane, end - begin});
    for (int slot = begin; slot < end; ++slot) {
      const int32_t a = fixture.face_vertices[slot];
      const int32_t b = fixture.face_vertices[slot + 1 == end ? begin : slot + 1];
      const int32_t c = fixture.face_vertices[slot + 2 >= end ? begin + slot + 2 - end : slot + 2];
      PHX_CHECK(a >= 0 && a < fixture.vertices);
      if (a < 0 || a >= fixture.vertices || b < 0 || b >= fixture.vertices || c < 0 || c >= fixture.vertices) return;
      used[a] = 1;
      PHX_CHECK(fixture.incidence[a] & (1u << plane));
      ++directed[{a, b}];
      // Strictly convex and counterclockwise about the outward row normal.
      std::array<ReferenceWide, 3> first, second;
      for (int axis = 0; axis < 3; ++axis) {
        first[axis] = points[b].numerator[axis] * points[a].denominator -
                      points[a].numerator[axis] * points[b].denominator;
        second[axis] = points[c].numerator[axis] * points[b].denominator -
                       points[b].numerator[axis] * points[c].denominator;
      }
      const std::array<ReferenceWide, 3> normal{rows[plane][0], rows[plane][1], rows[plane][2]};
      PHX_CHECK(reference_wide_det(normal, first, second) > 0);
    }
  }
  for (int vertex = 0; vertex < fixture.vertices; ++vertex) PHX_CHECK(used[vertex]);
  for (const auto& [edge, count] : directed) {
    PHX_CHECK(count == 1);
    const auto reverse = directed.find({edge[1], edge[0]});
    PHX_CHECK(reverse != directed.end() && reverse->second == 1);
  }
  const int edges = static_cast<int>(directed.size()) / 2;
  PHX_CHECK(fixture.vertices - edges + fixture.faces == 2);
  PHX_CHECK(fixture.face_offsets[fixture.faces] == 2 * edges && 2 * edges <= 6 * planes - 12);
  std::sort(labels.begin(), labels.end());
  PHX_CHECK(labels == expected_faces);
  ReferenceWide six_volume = 0;
  std::vector<std::array<ReferenceWide, 3>> scaled(fixture.vertices);
  for (int vertex = 0; vertex < fixture.vertices; ++vertex)
    for (int axis = 0; axis < 3; ++axis) {
      PHX_CHECK((scale * points[vertex].numerator[axis]) % points[vertex].denominator == 0);
      scaled[vertex][axis] = scale * points[vertex].numerator[axis] / points[vertex].denominator;
    }
  for (int face = 0; face < fixture.faces; ++face) {
    const int begin = fixture.face_offsets[face], end = fixture.face_offsets[face + 1];
    for (int slot = begin + 1; slot + 1 < end; ++slot)
      six_volume += reference_wide_det(scaled[fixture.face_vertices[begin]], scaled[fixture.face_vertices[slot]],
                                       scaled[fixture.face_vertices[slot + 1]]);
  }
  PHX_CHECK(six_volume == expected_six_volume);
}

// Capacity, scratch and work refusals at exact R-1, success at R; refusals
// publish no counts and scratch refusal happens before any charged work.
void check_reference_limits(ReferenceClipFixture& fixture) {
  PHX_CHECK(fixture.run() == PHX_MC_OK);
  const ReferenceClipFixture expected = fixture;
  const int64_t required = fixture.work;
  for (int32_t ReferenceClipFixture::* capacity :
       {&ReferenceClipFixture::vertex_capacity, &ReferenceClipFixture::face_capacity,
        &ReferenceClipFixture::entry_capacity}) {
    --(fixture.*capacity);
    PHX_CHECK(fixture.run() == PHX_MC_INVALID_ARGUMENT);
    ++(fixture.*capacity);
    PHX_CHECK(fixture.run() == PHX_MC_OK && fixture.same_output(expected));
  }
  PHX_CHECK(fixture.run(fixture.resident()) == PHX_MC_OK && fixture.same_output(expected));
  PHX_CHECK(fixture.run(fixture.resident() - 1) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(fixture.vertices == 0 && fixture.faces == 0 && fixture.work == 0);
  PHX_CHECK(fixture.run(1 << 20, required) == PHX_MC_OK && fixture.work == required);
  PHX_CHECK(fixture.same_output(expected));
  PHX_CHECK(fixture.run(1 << 20, required - 1) == PHX_MC_CAPACITY_EXCEEDED);
  PHX_CHECK(fixture.vertices == 0 && fixture.faces == 0 && fixture.work <= required - 1);
}

void test_exact_reference_tetrahedron_and_corner_closure() {
  ReferenceClipFixture fixture(reference_rows({kReferenceBox}));
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
    const phx_mc_polyhedron_corner_hexes_request request{
        .nv = fixture.vertices,
        .ne = static_cast<int64_t>(edges.size()/2),
        .nf = fixture.faces,
        .nc = 1,
        .face_entries = static_cast<int64_t>(face_vertices.size()),
        .cell_face_entries = fixture.faces,
        .corner_entries = fixture.vertices,
        .edges = edges.data(),
        .face_offsets = face_offsets.data(),
        .face_vertices = face_vertices.data(),
        .cell_offsets = cell_faces,
        .cell_faces = face_ids.data(),
        .vertex_offsets = cell_vertices,
        .cell_vertices = vertex_ids.data(),
        .corner_orientations = orientations.data(),
        .vertex_count = static_cast<int64_t>(fixture.vertices + edges.size()/2 + fixture.faces + 1),
        .hex_capacity = capacity,
        .scratch_limit = scratch,
        .work_limit = work_limit,
        .hexes = hexes.data(),
        .parent_cells = parents.data(),
        .hex_count = &count,
        .witness = witness,
        .work_units = &work,
    };
    return phx_mc_polyhedron_corner_hexes(&request);
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
  // The ten-plane resident tariff is unchanged: 32768 + 65536 M bytes.
  PHX_CHECK(fixture.resident() == 98304);
  PHX_CHECK(fixture.run() == PHX_MC_OK);
  check_reference_clip(fixture, 10, {{3, 3}, {4, 4}, {5, 5}, {6, 4}, {7, 5}, {8, 4}, {9, 5}}, 40, 7474);
  check_reference_limits(fixture);
}

void test_exact_reference_clip_plane_counts() {
  ReferenceClipFixture canonical(reference_rows({}));
  PHX_CHECK(canonical.run() == PHX_MC_OK);
  check_reference_clip(canonical, 4, {{0, 3}, {1, 3}, {2, 3}, {3, 3}}, 1, 1);
  check_reference_limits(canonical);

  // Identity source tetra intersected with a rational cell tetra: 8 planes.
  ReferenceClipFixture cell(reference_rows({kReferenceCell}));
  PHX_CHECK(cell.run() == PHX_MC_OK);
  check_reference_clip(cell, 6, {{3, 3}, {4, 3}, {5, 4}, {6, 4}, {7, 4}}, 360, 11745538);
  check_reference_limits(cell);

  // s ∩ C ∩ A^-1(t), all in source reference coordinates: 12 planes.
  ReferenceClipFixture chain(reference_rows({kReferenceCell, kReferenceTarget}));
  PHX_CHECK(chain.run() == PHX_MC_OK);
  check_reference_clip(chain, 14,
                       {{3, 6}, {4, 6}, {5, 6}, {6, 6}, {7, 4}, {8, 3}, {9, 4}, {10, 4}, {11, 3}},
                       75600, static_cast<ReferenceWide>(90258142810500LL));
  check_reference_limits(chain);

  // Sixteen planes with non-simple four-plane vertices.
  ReferenceClipFixture chamfered(reference_rows({kReferenceChamfered}));
  PHX_CHECK(chamfered.run() == PHX_MC_OK);
  check_reference_clip(chamfered, 19,
                       {{3, 3}, {4, 7}, {5, 4}, {6, 6}, {7, 4}, {8, 7}, {9, 4}, {10, 5},
                        {11, 4}, {12, 5}, {13, 3}, {14, 4}, {15, 4}},
                       20, 1078);
  check_reference_limits(chamfered);
}

void test_exact_reference_cap_incidence_under_plane_orders() {
  // The chamfered polytope has four-plane vertices. Changing the clipping
  // order exercises sections through existing vertices and edges, while its
  // exact volume, strict facet convexity and outward edge pairing stay fixed.
  constexpr std::array<int32_t, 12> facet_sizes{7, 4, 6, 4, 7, 4, 5, 4, 5, 3, 4, 4};
  for (int reversed = 0; reversed < 2; ++reversed) {
    for (int shift = 0; shift < 12; ++shift) {
      ReferenceRows cuts;
      std::vector<std::array<int32_t, 2>> expected{{3, 3}};
      for (int slot = 0; slot < 12; ++slot) {
        const int original = (shift + (reversed ? 11 - slot : slot)) % 12;
        cuts.push_back(kReferenceChamfered[original]);
        expected.push_back({4 + slot, facet_sizes[original]});
      }
      ReferenceClipFixture fixture(reference_rows({cuts}));
      PHX_CHECK(fixture.run() == PHX_MC_OK);
      check_reference_clip(fixture, 19, expected, 20, 1078);
    }
  }
}

void test_exact_reference_clip_high_bit_rationals() {
  // Every row, canonical rows included, is multiplied by a distinct odd
  // non-power-of-two 72-bit integer, so coefficients span three limbs and all
  // cofactors/predicates are multi-limb. The exact geometry is unchanged, so
  // support, incidence and face bits must equal the reduced-row result.
  unsigned __int128 base = 1;
  for (int power = 0; power < 45; ++power) base *= 3;
  const auto rows = reference_rows({kReferenceCell, kReferenceTarget});
  std::vector<unsigned __int128> multipliers;
  for (std::size_t plane = 0; plane < rows.size(); ++plane) multipliers.push_back(base + 2 * plane);
  ReferenceClipFixture reduced(rows), scaled(rows, multipliers);
  PHX_CHECK(scaled.limbs == 3);
  PHX_CHECK(reduced.run() == PHX_MC_OK && scaled.run() == PHX_MC_OK);
  PHX_CHECK(scaled.same_output(reduced));
  check_reference_clip(scaled, 14,
                       {{3, 6}, {4, 6}, {5, 6}, {6, 6}, {7, 4}, {8, 3}, {9, 4}, {10, 4}, {11, 3}},
                       75600, static_cast<ReferenceWide>(90258142810500LL));
  PHX_CHECK(scaled.resident() == 20 * (2048 + 3 * 4096));
  check_reference_limits(scaled);
}

void test_exact_reference_clip_redundant_contact_and_empty() {
  // A scaled duplicate of x>=1/8, a plane touching only (1/8,1/8,1/8), one
  // containing the edge x=y=1/8, and a disjoint far plane: same cell, extra
  // incidence only, no extra facets.
  ReferenceClipFixture redundant(reference_rows(
      {kReferenceBox, {{-24, 0, 0, 3}, {-8, -8, -8, 3}, {-4, -4, 0, 1}, {1, 0, 0, -5}}}));
  PHX_CHECK(redundant.run() == PHX_MC_OK);
  check_reference_clip(redundant, 10, {{3, 3}, {4, 4}, {5, 5}, {6, 4}, {7, 5}, {8, 4}, {9, 5}}, 40, 7474);
  int maximum_incidence = 0;
  for (int vertex = 0; vertex < redundant.vertices; ++vertex) {
    int incident = 0;
    for (int plane = 0; plane < 16; ++plane) incident += (redundant.incidence[vertex] >> plane) & 1;
    maximum_incidence = std::max(maximum_incidence, incident);
    PHX_CHECK(!(redundant.incidence[vertex] & (1u << 13)));
    PHX_CHECK(((redundant.incidence[vertex] >> 4) & 1u) == ((redundant.incidence[vertex] >> 10) & 1u));
  }
  PHX_CHECK(maximum_incidence == 6);
  check_reference_limits(redundant);

  // Disjoint, vertex contact, face contact and a zero-width slab publish no
  // cell; a far plane after a complete twelve-plane chain empties it.
  for (const auto& rows : {reference_rows({{{-1, 0, 0, 2}}}), reference_rows({{{-1, 0, 0, 1}}}),
                           reference_rows({{{-1, -1, -1, 1}}}),
                           reference_rows({{{-5, 0, 0, 2}, {5, 0, 0, -2}}}),
                           reference_rows({kReferenceCell, kReferenceTarget, {{-1, 0, 0, 1}}})}) {
    ReferenceClipFixture empty(rows);
    PHX_CHECK(empty.run() == PHX_MC_OK);
    PHX_CHECK(empty.vertices == 0 && empty.faces == 0 && empty.work > 0);
  }
}

void test_exact_reference_clip_refusals() {
  // Canonical rows must be positive multiples of the simplex planes.
  const ReferenceRows malformed[] = {
      {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {1, 1, 1, -1}},
      {{-1, 0, 0, 0}, {0, -1, 1, 0}, {0, 0, -1, 0}, {1, 1, 1, -1}},
      {{-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, -1}},
      {{-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {1, 1, 2, -1}},
      {{-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {2, 2, 2, -1}},
      {{-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {-1, -1, -1, 1}},
  };
  for (const auto& rows : malformed) {
    ReferenceClipFixture fixture(rows);
    PHX_CHECK(fixture.run() == PHX_MC_INVALID_INPUT);
    PHX_CHECK(fixture.vertices == 0 && fixture.faces == 0);
  }
  ReferenceClipFixture positive({{-7, 0, 0, 0}, {0, -2, 0, 0}, {0, 0, -9, 0}, {3, 3, 3, -3}});
  PHX_CHECK(positive.run() == PHX_MC_OK);
  check_reference_clip(positive, 4, {{0, 3}, {1, 3}, {2, 3}, {3, 3}}, 1, 1);

  // Sign/limb encoding mismatch is malformed input.
  ReferenceClipFixture chain(reference_rows({kReferenceCell, kReferenceTarget}));
  chain.signs[1] = 1;
  PHX_CHECK(chain.run() == PHX_MC_INVALID_INPUT && chain.vertices == 0 && chain.faces == 0);
  chain.signs[1] = 0;
  // Plane counts outside [4,16], or disagreeing with the packed offsets.
  for (const int32_t count : {-1, 0, 3, 17, 8}) {
    chain.plane_count = count;
    PHX_CHECK(chain.run() == PHX_MC_INVALID_ARGUMENT);
  }
  chain.plane_count = 12;
  PHX_CHECK(chain.run() == PHX_MC_OK && chain.vertices == 14);
}
