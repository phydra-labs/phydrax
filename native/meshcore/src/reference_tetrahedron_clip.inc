// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Included inside arrangement.cpp's canonical arbitrary-integer owner.
// Plane convention: a*x+b*y+c*z+d <= 0. No rounded point is an input/output.
struct ExactReferenceVertex3 {
  std::array<Dyadic, 4> coordinates;
  std::array<int32_t, 3> support{};
  uint16_t incidence = 0;
};
struct ExactReferenceFace3 {
  std::array<int32_t, 16> vertices{};
  int count = 0;
  int plane = 0;
};

Dyadic reference_minor3(const std::array<Dyadic, 4>& a,
                       const std::array<Dyadic, 4>& b,
                       const std::array<Dyadic, 4>& c, int x, int y, int z) {
  return a[x] * (b[y] * c[z] - b[z] * c[y]) -
         a[y] * (b[x] * c[z] - b[z] * c[x]) +
         a[z] * (b[x] * c[y] - b[y] * c[x]);
}

int32_t clip_reference_tetrahedron_exact(
    const uint32_t* words, int64_t word_count, const int64_t* offsets,
    const int8_t* signs, int64_t scratch_limit, int64_t work_limit,
    int32_t vertex_capacity, int32_t face_capacity, int32_t entry_capacity,
    int32_t* supporting_planes, uint16_t* incident_planes,
    int32_t* face_offsets, int32_t* face_planes, int32_t* face_vertices,
    int32_t* vertex_count, int32_t* face_count, int64_t* work_units) {
  if (offsets == nullptr || signs == nullptr || supporting_planes == nullptr ||
      incident_planes == nullptr || face_offsets == nullptr || face_planes == nullptr ||
      face_vertices == nullptr || vertex_count == nullptr || face_count == nullptr ||
      work_units == nullptr || word_count < 0 || (word_count && words == nullptr) ||
      offsets[0] != 0 || offsets[40] != word_count || scratch_limit <= 0 ||
      work_limit <= 0 || vertex_capacity < 16 || face_capacity < 10 || entry_capacity < 48)
    return PHX_MC_INVALID_ARGUMENT;
  *vertex_count = 0;
  *face_count = 0;
  *work_units = 0;
  int64_t maximum_words = 1;
  for (int scalar = 0; scalar < 40; ++scalar) {
    if (offsets[scalar] < 0 || offsets[scalar + 1] < offsets[scalar] ||
        offsets[scalar + 1] > word_count || signs[scalar] < -1 || signs[scalar] > 1 ||
        ((offsets[scalar + 1] == offsets[scalar]) != (signs[scalar] == 0)))
      return PHX_MC_INVALID_INPUT;
    maximum_words = std::max(maximum_words, offsets[scalar + 1] - offsets[scalar]);
  }
  // Cofactors have at most 3M limbs; hull predicates at most 10M. Source
  // planes, both bounded vertex banks and all live temporaries fit this bound.
  if (scratch_limit < 32768 || maximum_words > (scratch_limit - 32768) / 65536)
    return PHX_MC_CAPACITY_EXCEEDED;
  MemoryBudgetWindow memory(static_cast<std::size_t>(scratch_limit));
  MemoryScope memory_scope(memory.owner());
  auto charge = [&](int64_t operations) {
    if (operations < 0 || operations > work_limit - *work_units)
      throw ExecutionRefusal{PHX_MC_CAPACITY_EXCEEDED};
    native_execution_charge(operations);
    *work_units += operations;
  };
  native_execution_primitive_query();
  std::array<std::array<Dyadic, 4>, 10> planes;
  for (int scalar = 0; scalar < 40; ++scalar) {
    charge(offsets[scalar + 1] - offsets[scalar]);
    planes[scalar / 4][scalar % 4] = Dyadic::integer(
        words == nullptr ? nullptr : words + offsets[scalar],
        static_cast<std::size_t>(offsets[scalar + 1] - offsets[scalar]), signs[scalar]);
  }
  charge(20);
  for (int plane = 0; plane < 3; ++plane)
    for (int axis = 0; axis < 4; ++axis)
      if (axis == plane ? planes[plane][axis].sign() >= 0 : planes[plane][axis].sign() != 0)
        return PHX_MC_INVALID_INPUT;
  if (planes[3][0].sign() <= 0 || (planes[3][1] - planes[3][0]).sign() != 0 ||
      (planes[3][2] - planes[3][0]).sign() != 0 || (planes[3][3] + planes[3][0]).sign() != 0)
    return PHX_MC_INVALID_INPUT;
  std::array<ExactReferenceVertex3, 16> vertices, next_vertices;
  std::array<ExactReferenceFace3, 10> faces{}, next_faces{};
  const int32_t original_support[4][3] = {{0,1,2}, {1,2,3}, {0,2,3}, {0,1,3}};
  const int32_t original_faces[4][3] = {{0,3,2}, {0,1,3}, {0,2,1}, {1,2,3}};
  for (int vertex = 0; vertex < 4; ++vertex) {
    vertices[vertex].coordinates[3] = Dyadic(1.0);
    if (vertex) vertices[vertex].coordinates[vertex - 1] = Dyadic(1.0);
    for (int slot = 0; slot < 3; ++slot) {
      vertices[vertex].support[slot] = original_support[vertex][slot];
      vertices[vertex].incidence |= static_cast<uint16_t>(1u << original_support[vertex][slot]);
    }
    faces[vertex].plane = vertex;
    faces[vertex].count = 3;
    std::copy(original_faces[vertex], original_faces[vertex] + 3, faces[vertex].vertices.begin());
  }
  int used = 4, face_used = 4;
  auto plane_side = [&](int plane, const ExactReferenceVertex3& vertex) {
    charge(7);
    const auto& p = planes[plane];
    const auto& v = vertex.coordinates;
    return (p[0]*v[0] + p[1]*v[1] + p[2]*v[2] + p[3]*v[3]).sign();
  };
  for (int clipping = 4; clipping < 10; ++clipping) {
    std::array<int, 16> sides{};
    bool outside = false, strictly_inside = false;
    for (int vertex = 0; vertex < used; ++vertex) {
      sides[vertex] = plane_side(clipping, vertices[vertex]);
      outside |= sides[vertex] > 0;
      strictly_inside |= sides[vertex] < 0;
      if (sides[vertex] == 0) vertices[vertex].incidence |= static_cast<uint16_t>(1u << clipping);
    }
    if (!outside) continue;
    // A contact-only intersection has no volume and no published cut cell.
    if (!strictly_inside) return PHX_MC_OK;
    std::array<int32_t, 16> retained;
    retained.fill(-1);
    struct EdgeCrossing { int32_t first, second, vertex; };
    std::array<EdgeCrossing, 24> crossings;
    int crossing_count = 0, next_used = 0, next_face_used = 0;
    auto keep = [&](int32_t old_vertex) {
      if (retained[old_vertex] >= 0) return retained[old_vertex];
      if (next_used == 16) throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
      retained[old_vertex] = next_used;
      next_vertices[next_used] = std::move(vertices[old_vertex]);
      return static_cast<int32_t>(next_used++);
    };
    auto crossing = [&](int32_t first, int32_t second) {
      if (sides[first] == 0) return keep(first);
      if (sides[second] == 0) return keep(second);
      const int32_t low = std::min(first, second), high = std::max(first, second);
      for (int row = 0; row < crossing_count; ++row) {
        charge(1);
        if (crossings[row].first == low && crossings[row].second == high) return crossings[row].vertex;
      }
      const uint16_t common = vertices[first].incidence & vertices[second].incidence;
      int first_plane = -1, second_plane = -1;
      Dyadic denominator;
      for (int a = 0; a < 9 && first_plane < 0; ++a) {
        if (!(common & (1u << a))) continue;
        for (int b = a + 1; b < 10; ++b) {
          if (!(common & (1u << b))) continue;
          charge(18);
          denominator = reference_minor3(planes[a], planes[b], planes[clipping], 0, 1, 2);
          if (denominator.sign()) { first_plane = a; second_plane = b; break; }
        }
      }
      if (first_plane < 0 || next_used == 16 || crossing_count == 24)
        throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
      const auto& a = planes[first_plane];
      const auto& b = planes[second_plane];
      const auto& c = planes[clipping];
      charge(54);
      ExactReferenceVertex3 vertex;
      vertex.coordinates[0] = -reference_minor3(a,b,c,1,2,3);
      vertex.coordinates[1] = reference_minor3(a,b,c,0,2,3);
      vertex.coordinates[2] = -reference_minor3(a,b,c,0,1,3);
      vertex.coordinates[3] = std::move(denominator);
      if (vertex.coordinates[3].sign() < 0)
        for (auto& value : vertex.coordinates) value.negate();
      vertex.support = {first_plane, second_plane, clipping};
      // All original incidence, including tangent or later grid planes, is
      // scientific provenance. Only the currently admitted faces decide clipping.
      for (int plane = 0; plane < 10; ++plane)
        if (plane_side(plane, vertex) == 0) vertex.incidence |= static_cast<uint16_t>(1u << plane);
      const int32_t row = next_used++;
      next_vertices[row] = std::move(vertex);
      crossings[crossing_count++] = {low, high, row};
      return row;
    };
    for (int face = 0; face < face_used; ++face) {
      ExactReferenceFace3 clipped;
      clipped.plane = faces[face].plane;
      auto append = [&](int32_t vertex) {
        if (clipped.count && clipped.vertices[clipped.count - 1] == vertex) return;
        if (clipped.count == 16) throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
        clipped.vertices[clipped.count++] = vertex;
      };
      const auto& old = faces[face];
      for (int slot = 0; slot < old.count; ++slot) {
        charge(1);
        const int32_t a = old.vertices[slot], b = old.vertices[(slot + 1) % old.count];
        const bool first_inside = sides[a] <= 0, second_inside = sides[b] <= 0;
        if (first_inside != second_inside) append(crossing(a,b));
        if (second_inside) append(keep(b));
      }
      if (clipped.count > 1 && clipped.vertices[0] == clipped.vertices[clipped.count - 1]) --clipped.count;
      if (clipped.count >= 3) {
        if (next_face_used == 10) throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
        next_faces[next_face_used++] = clipped;
      }
    }
    // The new cap is an exact oriented convex hull of the on-plane source
    // intersections, not an angle sort of numerical/nearest vertices.
    ExactReferenceFace3 cap;
    cap.plane = clipping;
    std::array<int32_t, 16> members;
    int member_count = 0;
    for (int vertex = 0; vertex < next_used; ++vertex)
      if (next_vertices[vertex].incidence & (1u << clipping)) members[member_count++] = vertex;
    int axis = 0;
    while (axis < 3 && planes[clipping][axis].sign() == 0) ++axis;
    if (axis == 3 || member_count < 3) return PHX_MC_INTERNAL_ERROR;
    const int x = (axis + 1) % 3, y = (axis + 2) % 3;
    auto compare_axis = [&](int a, int b, int coordinate) {
      charge(3);
      return (next_vertices[a].coordinates[coordinate] * next_vertices[b].coordinates[3] -
              next_vertices[b].coordinates[coordinate] * next_vertices[a].coordinates[3]).sign();
    };
    std::sort(members.begin(), members.begin() + member_count, [&](int a, int b) {
      const int first = compare_axis(a,b,x);
      return first ? first < 0 : compare_axis(a,b,y) < 0;
    });
    auto turn = [&](int a, int b, int c) {
      charge(18);
      const auto& A = next_vertices[a].coordinates;
      const auto& B = next_vertices[b].coordinates;
      const auto& C = next_vertices[c].coordinates;
      return (A[x]*(B[y]*C[3]-B[3]*C[y]) - A[y]*(B[x]*C[3]-B[3]*C[x]) + A[3]*(B[x]*C[y]-B[y]*C[x])).sign();
    };
    std::array<int32_t, 32> hull;
    int length = 0;
    for (int slot = 0; slot < member_count; ++slot) {
      while (length >= 2 && turn(hull[length-2], hull[length-1], members[slot]) <= 0) --length;
      hull[length++] = members[slot];
    }
    const int lower = length;
    for (int slot = member_count - 2; slot >= 0; --slot) {
      while (length > lower && turn(hull[length-2], hull[length-1], members[slot]) <= 0) --length;
      hull[length++] = members[slot];
    }
    --length;
    if (length < 3 || length > 16 || next_face_used == 10) return PHX_MC_INTERNAL_ERROR;
    if (planes[clipping][axis].sign() < 0) std::reverse(hull.begin(), hull.begin() + length);
    cap.count = length;
    std::copy(hull.begin(), hull.begin() + length, cap.vertices.begin());
    next_faces[next_face_used++] = cap;
    vertices.swap(next_vertices);
    faces.swap(next_faces);
    used = next_used;
    face_used = next_face_used;
  }
  if (face_used < 4) return PHX_MC_INTERNAL_ERROR;
  // Compact only genuine facet-used vertices; all supporting-plane source
  // witnesses survive. Nothing is committed before all capacities hold.
  std::array<int32_t, 16> rows;
  rows.fill(-1);
  int count = 0, entries = 0;
  for (int face = 0; face < face_used; ++face) {
    entries += faces[face].count;
    for (int slot = 0; slot < faces[face].count; ++slot)
      rows[faces[face].vertices[slot]] = 0;
  }
  if (entries > 48) return PHX_MC_INTERNAL_ERROR;
  for (int vertex = 0; vertex < used; ++vertex)
    if (rows[vertex] == 0) rows[vertex] = count++;
  for (int vertex = 0; vertex < used; ++vertex) {
    if (rows[vertex] < 0) continue;
    for (int slot = 0; slot < 3; ++slot) supporting_planes[3 * rows[vertex] + slot] = vertices[vertex].support[slot];
    incident_planes[rows[vertex]] = vertices[vertex].incidence;
  }
  face_offsets[0] = 0;
  entries = 0;
  for (int face = 0; face < face_used; ++face) {
    auto& current = faces[face];
    const auto minimum = std::min_element(current.vertices.begin(), current.vertices.begin() + current.count);
    std::rotate(current.vertices.begin(), minimum, current.vertices.begin() + current.count);
    face_planes[face] = current.plane;
    for (int slot = 0; slot < current.count; ++slot) face_vertices[entries++] = rows[current.vertices[slot]];
    face_offsets[face + 1] = entries;
  }
  *vertex_count = count;
  *face_count = face_used;
  return PHX_MC_OK;
}
