// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Included inside arrangement.cpp's canonical arbitrary-integer owner.
// Plane convention: a*x+b*y+c*z+d <= 0. No rounded point is an input/output.
//
// Every intermediate and final cell is a bounded 3D convex polytope whose
// facets carry distinct input plane labels, so with F input planes it has at
// most F facets, V <= 2F-4 vertices, E <= 3F-6 edges, 2E <= 6F-12 face
// entries, and every facet polygon has at most F-1 vertices (one more while a
// cyclic walk repeats its first vertex). The uint16 incidence mask holds F<=16.
constexpr int kReferenceTetraMaximumPlanes = 16;
constexpr int kReferenceTetraMaximumVertices = 2 * kReferenceTetraMaximumPlanes - 4;
struct ExactReferenceVertex3 {
  std::array<Dyadic, 4> coordinates;
  std::array<int32_t, 3> support{};
  uint16_t incidence = 0;
};
struct ExactReferenceFace3 {
  std::array<int32_t, kReferenceTetraMaximumPlanes> vertices{};
  int count = 0;
  int plane = 0;
};

struct ReferenceNoopAdmission {
  void operator()(std::int64_t) const noexcept {}
};

template <std::size_t N>
std::int64_t reference_signed_product_sum_work(
    const std::array<const Dyadic*, N>& left, const std::array<const Dyadic*, N>& right,
    const std::array<int, N>& signs) {
  std::int64_t operations = 0;
  int terms = 0, first_sign = 1;
  for (std::size_t index = 0; index < N; ++index) {
    if (left[index]->is_zero() || right[index]->is_zero()) continue;
    if (!left[index]->is_positive_one() && !right[index]->is_positive_one()) ++operations;
    if (terms == 0) {
      first_sign = signs[index];
    } else {
      ++operations;
    }
    ++terms;
  }
  if (terms == 1 && first_sign < 0) ++operations;
  return operations;
}

// Executes an already admitted sum of signed products in its original term
// order. Product/unit/zero decisions use exact Dyadic values only.
template <std::size_t N>
Dyadic reference_signed_product_sum_admitted(
    const std::array<const Dyadic*, N>& left, const std::array<const Dyadic*, N>& right,
    const std::array<int, N>& signs) {
  std::array<Dyadic, N> products;
  std::array<const Dyadic*, N> terms{};
  std::array<bool, N> present{};
  for (std::size_t index = 0; index < N; ++index) {
    if (left[index]->is_zero() || right[index]->is_zero()) continue;
    present[index] = true;
    if (left[index]->is_positive_one()) {
      terms[index] = right[index];
    } else if (right[index]->is_positive_one()) {
      terms[index] = left[index];
    } else {
      products[index] = *left[index] * *right[index];
      terms[index] = &products[index];
    }
  }

  Dyadic result;
  const Dyadic* first = nullptr;
  int first_index = -1, first_sign = 1;
  bool combined = false;
  for (std::size_t index = 0; index < N; ++index) {
    if (!present[index]) continue;
    if (first == nullptr) {
      first = terms[index];
      first_index = static_cast<int>(index);
      first_sign = signs[index];
    } else if (!combined) {
      result = Dyadic::signed_sum(*first, first_sign < 0, *terms[index], signs[index] < 0);
      combined = true;
    } else {
      result = Dyadic::signed_sum(result, false, *terms[index], signs[index] < 0);
    }
  }
  if (first == nullptr) return {};
  if (!combined) {
    if (first == &products[static_cast<std::size_t>(first_index)])
      result = std::move(products[static_cast<std::size_t>(first_index)]);
    else
      result = *first;
    if (first_sign < 0) result.negate();
  }
  return result;
}

template <std::size_t N, class Admit>
Dyadic reference_signed_product_sum(const std::array<const Dyadic*, N>& left,
                                    const std::array<const Dyadic*, N>& right,
                                    const std::array<int, N>& signs, Admit&& admit) {
  admit(reference_signed_product_sum_work(left, right, signs));
  return reference_signed_product_sum_admitted(left, right, signs);
}

// Exact sign of the plane row at a homogeneous point (X, Y, Z, W), W > 0.
template <class Admit>
int reference_plane_side(const std::array<Dyadic, 4>& plane,
                         const std::array<Dyadic, 4>& point, Admit&& admit) {
  std::array<bool, 4> terms{};
  int operations = 0, nonzero_terms = 0;
  for (std::size_t axis = 0; axis < 4; ++axis) {
    terms[axis] = !plane[axis].is_zero() && !point[axis].is_zero();
    if (!terms[axis]) continue;
    if (!plane[axis].is_positive_one() && !point[axis].is_positive_one()) ++operations;
    ++nonzero_terms;
  }
  if (nonzero_terms > 1) operations += nonzero_terms - 1;
  admit(operations);

  std::array<Dyadic, 4> products;
  std::array<const Dyadic*, 4> values{};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (!terms[axis]) continue;
    if (plane[axis].is_positive_one()) {
      values[axis] = &point[axis];
    } else if (point[axis].is_positive_one()) {
      values[axis] = &plane[axis];
    } else {
      products[axis] = plane[axis] * point[axis];
      values[axis] = &products[axis];
    }
  }
  Dyadic sum;
  const Dyadic* accumulated = nullptr;
  for (const Dyadic* value : values) {
    if (value == nullptr) continue;
    if (accumulated == nullptr) {
      accumulated = value;
    } else {
      sum = *accumulated + *value;
      accumulated = &sum;
    }
  }
  return accumulated == nullptr ? 0 : accumulated->sign();
}

int reference_plane_side(const std::array<Dyadic, 4>& plane,
                         const std::array<Dyadic, 4>& point) {
  return reference_plane_side(plane, point, ReferenceNoopAdmission{});
}

Dyadic reference_minor3(const std::array<Dyadic, 4>& a,
                        const std::array<Dyadic, 4>& b,
                        const std::array<Dyadic, 4>& c, int i, int j, int k) {
  return phx::mc::exact::det3(a[i], a[j], a[k], b[i], b[j], b[k], c[i], c[j], c[k]);
}

// Cramer solution of three plane rows whose normal determinant minor(0,1,2)
// is the nonzero `denominator`, published with W > 0.
std::array<Dyadic, 4> reference_plane_triple_point(const std::array<Dyadic, 4>& a,
                                                   const std::array<Dyadic, 4>& b,
                                                   const std::array<Dyadic, 4>& c,
                                                   Dyadic denominator) {
  std::array<Dyadic, 4> point;
  point[0] = -reference_minor3(a, b, c, 1, 2, 3);
  point[1] = reference_minor3(a, b, c, 0, 2, 3);
  point[2] = -reference_minor3(a, b, c, 0, 1, 3);
  point[3] = std::move(denominator);
  if (point[3].sign() < 0)
    for (auto& value : point) value.negate();
  return point;
}

// The six 2x2 minors p_ij = a_i b_j - a_j b_i (ij = 01,02,03,12,13,23) of a
// plane pair. With them, every 3x3 minor of (a, b, c) is a three-term
// combination of c, so the Cramer point of the pair and any third plane is
// the same exact integer vector as reference_plane_triple_point.
template <class Admit>
std::array<Dyadic, 6> reference_pair_minors(const std::array<Dyadic, 4>& a,
                                            const std::array<Dyadic, 4>& b, Admit&& admit) {
  constexpr std::array<std::array<int, 2>, 6> indices{{{0, 1}, {0, 2}, {0, 3},
                                                       {1, 2}, {1, 3}, {2, 3}}};
  constexpr std::array<int, 2> signs{1, -1};
  std::int64_t operations = 0;
  for (const auto& pair : indices) {
    const std::array<const Dyadic*, 2> left{&a[pair[0]], &a[pair[1]]};
    const std::array<const Dyadic*, 2> right{&b[pair[1]], &b[pair[0]]};
    operations += reference_signed_product_sum_work(left, right, signs);
  }
  admit(operations);

  std::array<Dyadic, 6> result;
  for (std::size_t minor = 0; minor < indices.size(); ++minor) {
    const auto& pair = indices[minor];
    const std::array<const Dyadic*, 2> left{&a[pair[0]], &a[pair[1]]};
    const std::array<const Dyadic*, 2> right{&b[pair[1]], &b[pair[0]]};
    result[minor] = reference_signed_product_sum_admitted(left, right, signs);
  }
  return result;
}

// minor(0,1,2) of (a, b, c) from the pair minors: the normal determinant.
template <class Admit>
Dyadic reference_pair_denominator(const std::array<Dyadic, 6>& p,
                                  const std::array<Dyadic, 4>& c, Admit&& admit) {
  const std::array<const Dyadic*, 3> left{&c[0], &c[1], &c[2]};
  const std::array<const Dyadic*, 3> right{&p[3], &p[1], &p[0]};
  constexpr std::array<int, 3> signs{1, -1, 1};
  return reference_signed_product_sum(left, right, signs, std::forward<Admit>(admit));
}

void reference_pair_point_terms(const std::array<Dyadic, 6>& p, const std::array<Dyadic, 4>& c,
                               int coordinate, std::array<const Dyadic*, 3>& left,
                               std::array<const Dyadic*, 3>& right,
                               std::array<int, 3>& signs) {
  if (coordinate == 0) {
    left = {&c[1], &c[2], &c[3]};
    right = {&p[5], &p[4], &p[3]};
    signs = {-1, 1, -1};
  } else if (coordinate == 1) {
    left = {&c[0], &c[2], &c[3]};
    right = {&p[5], &p[2], &p[1]};
    signs = {1, -1, 1};
  } else {
    left = {&c[0], &c[1], &c[3]};
    right = {&p[4], &p[2], &p[0]};
    signs = {-1, 1, -1};
  }
}

// The Cramer point of (a, b, c) from the pair minors and the nonzero
// denominator, published with W > 0.
template <class Admit>
std::array<Dyadic, 4> reference_pair_point(const std::array<Dyadic, 6>& p,
                                           const std::array<Dyadic, 4>& c, Dyadic denominator,
                                           Admit&& admit) {
  std::array<const Dyadic*, 3> left{}, right{};
  std::array<int, 3> signs{};
  std::int64_t operations = 0;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    reference_pair_point_terms(p, c, coordinate, left, right, signs);
    operations += reference_signed_product_sum_work(left, right, signs);
  }
  admit(operations);

  std::array<Dyadic, 4> point;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    reference_pair_point_terms(p, c, coordinate, left, right, signs);
    point[coordinate] = reference_signed_product_sum_admitted(left, right, signs);
  }
  point[3] = std::move(denominator);
  if (point[3].sign() < 0)
    for (auto& value : point) value.negate();
  return point;
}


// Walks the section boundary. Every retained facet is convex and outward
// oriented; its on-plane edge a->b bounds the new cap in the opposite
// direction (successor[b] = a). These edges form the section's one closed
// boundary, so no geometric hull or angle sort is needed. Starting from a
// tetrahedron and clipping by halfspaces cannot introduce a nonextreme vertex
// in the interior of a section edge.
template <class Charge>
bool reference_cap_walk(const std::array<int32_t, kReferenceTetraMaximumVertices>& successor,
                        int32_t first, int edges, int used, int clipping,
                        ExactReferenceFace3& cap, Charge& charge) {
  cap.plane = clipping;
  if (first < 0 || edges < 3 || edges > kReferenceTetraMaximumPlanes) return false;
  int32_t vertex = first;
  do {
    charge(1);
    if (vertex < 0 || vertex >= used || cap.count == edges) return false;
    cap.vertices[cap.count++] = vertex;
    vertex = successor[vertex];
  } while (vertex != first);
  return cap.count == edges;
}

// One exact halfspace section of a bounded outward convex polytope; shared by
// the fixed reference clipper and the pooled tetrahedra arrangement. Every
// current vertex has an exact side and both strict signs occur. keep(old) and
// crossing(face, first, second) return next-vertex rows counted by next_used;
// on_plane(row) tells whether a next row lies on the clipping plane. Retained
// facets keep their order and label, the cap labelled `clipping` comes last.
// The cap's boundary edges are recorded while the clipped loops are built, in
// the same facet and slot order as a scan of the retained loops would visit
// them. False reports a violated convex-polytope invariant.
template <class Sides, class Keep, class Crossing, class OnPlane, class Charge>
bool reference_section(
    const std::array<ExactReferenceFace3, kReferenceTetraMaximumPlanes>& faces, int face_used,
    const Sides& sides, Keep&& keep, Crossing&& crossing, OnPlane&& on_plane,
    const int& next_used, int face_limit, int clipping,
    std::array<ExactReferenceFace3, kReferenceTetraMaximumPlanes>& next_faces,
    int& next_face_used, Charge& charge) {
  std::array<int32_t, kReferenceTetraMaximumVertices> successor;
  successor.fill(-1);
  int32_t first = -1;
  int edges = 0;
  bool boundary = true;
  next_face_used = 0;
  for (int face = 0; face < face_used; ++face) {
    ExactReferenceFace3 clipped;
    clipped.plane = faces[face].plane;
    // On-plane edges of this loop, committed only if the loop is retained.
    std::array<std::array<int32_t, 2>, kReferenceTetraMaximumPlanes> pending;
    int pending_count = 0;
    auto append = [&](int32_t vertex) {
      if (clipped.count && clipped.vertices[clipped.count - 1] == vertex) return;
      if (clipped.count == kReferenceTetraMaximumPlanes) throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
      if (clipped.count) {
        const int32_t previous = clipped.vertices[clipped.count - 1];
        if (on_plane(previous) && on_plane(vertex)) pending[pending_count++] = {previous, vertex};
      }
      clipped.vertices[clipped.count++] = vertex;
    };
    const auto& old = faces[face];
    for (int slot = 0; slot < old.count; ++slot) {
      charge(1);
      const int32_t a = old.vertices[slot], b = old.vertices[(slot + 1) % old.count];
      const bool first_inside = sides[a] <= 0, second_inside = sides[b] <= 0;
      if (first_inside != second_inside) append(crossing(face, a, b));
      if (second_inside) append(keep(b));
    }
    if (clipped.count > 1 && clipped.vertices[0] == clipped.vertices[clipped.count - 1]) {
      // The last append already recorded the closing edge into vertex 0.
      --clipped.count;
    } else if (clipped.count > 1) {
      const int32_t last = clipped.vertices[clipped.count - 1], start = clipped.vertices[0];
      if (on_plane(last) && on_plane(start)) pending[pending_count++] = {last, start};
    }
    if (clipped.count >= 3) {
      if (next_face_used == face_limit) throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
      next_faces[next_face_used++] = clipped;
      for (int edge = 0; edge < pending_count; ++edge) {
        const int32_t a = pending[edge][0], b = pending[edge][1];
        if (a == b || successor[b] >= 0) {
          boundary = false;
          continue;
        }
        successor[b] = a;
        first = b;
        ++edges;
      }
    }
  }
  ExactReferenceFace3 cap;
  if (!boundary || next_face_used == face_limit ||
      !reference_cap_walk(successor, first, edges, next_used, clipping, cap, charge))
    return false;
  next_faces[next_face_used++] = cap;
  return true;
}

// Keep the fixed-capacity clipping state machine together: crossing reuse,
// incidence updates, pre-mutation charges and the final output commit share
// the same vertex banks. Cap topology has its own independently checked helper.
int32_t clip_reference_tetrahedron_exact(
    const uint32_t* words, int64_t word_count, const int64_t* offsets,
    const int8_t* signs, int32_t plane_count, int64_t scratch_limit,
    int64_t work_limit, int32_t vertex_capacity, int32_t face_capacity,
    int32_t entry_capacity, int32_t* supporting_planes, uint16_t* incident_planes,
    int32_t* face_offsets, int32_t* face_planes, int32_t* face_vertices,
    int32_t* vertex_count, int32_t* face_count, int64_t* work_units) {
  if (plane_count < 4 || plane_count > kReferenceTetraMaximumPlanes)
    return PHX_MC_INVALID_ARGUMENT;
  const int scalars = 4 * plane_count;
  const int vertex_limit = 2 * plane_count - 4;
  const int crossing_limit = 3 * plane_count - 6;
  const int entry_limit = 6 * plane_count - 12;
  if (offsets == nullptr || signs == nullptr || supporting_planes == nullptr ||
      incident_planes == nullptr || face_offsets == nullptr || face_planes == nullptr ||
      face_vertices == nullptr || vertex_count == nullptr || face_count == nullptr ||
      work_units == nullptr || word_count < 0 || (word_count && words == nullptr) ||
      offsets[0] != 0 || offsets[scalars] != word_count || scratch_limit <= 0 ||
      work_limit <= 0 || vertex_capacity < vertex_limit || face_capacity < plane_count ||
      entry_capacity < entry_limit)
    return PHX_MC_INVALID_ARGUMENT;
  *vertex_count = 0;
  *face_count = 0;
  *work_units = 0;
  int64_t maximum_words = 1;
  for (int scalar = 0; scalar < scalars; ++scalar) {
    if (offsets[scalar] < 0 || offsets[scalar + 1] < offsets[scalar] ||
        offsets[scalar + 1] > word_count || signs[scalar] < -1 || signs[scalar] > 1 ||
        ((offsets[scalar + 1] == offsets[scalar]) != (signs[scalar] == 0)))
      return PHX_MC_INVALID_INPUT;
    maximum_words = std::max(maximum_words, offsets[scalar + 1] - offsets[scalar]);
  }
  // With M-limb coefficients, cofactor coordinates hold at most 3M+2 limbs
  // and hull predicate temporaries at most 9M+2. Peak resident limbs are at
  // most 4FM (rows) + 8(2F-4)(3M+2) (both vertex banks) + 18(9M+2) (one
  // hull predicate). Each of the 2F-4 bounded vertex slots is admitted
  // 2048 + 4096M bytes, which exceeds that peak for every F in [4,16] and
  // keeps the ten-plane admission at 32768 + 65536M; it is checked before
  // any limb allocation.
  const int64_t slot_bytes = 2048;
  const int64_t slot_limb_bytes = 4096;
  if (scratch_limit < vertex_limit * slot_bytes ||
      maximum_words > (scratch_limit - vertex_limit * slot_bytes) / (vertex_limit * slot_limb_bytes))
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
  std::array<std::array<Dyadic, 4>, kReferenceTetraMaximumPlanes> planes;
  for (int scalar = 0; scalar < scalars; ++scalar) {
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
  std::array<ExactReferenceVertex3, kReferenceTetraMaximumVertices> vertices, next_vertices;
  std::array<ExactReferenceFace3, kReferenceTetraMaximumPlanes> faces{}, next_faces{};
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
    return reference_plane_side(planes[plane], vertex.coordinates);
  };
  for (int clipping = 4; clipping < plane_count; ++clipping) {
    std::array<int, kReferenceTetraMaximumVertices> sides{};
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
    std::array<int32_t, kReferenceTetraMaximumVertices> retained;
    retained.fill(-1);
    struct EdgeCrossing { int32_t first, second, vertex; };
    std::array<EdgeCrossing, 3 * kReferenceTetraMaximumPlanes - 6> crossings;
    int crossing_count = 0, next_used = 0, next_face_used = 0;
    auto keep = [&](int32_t old_vertex) {
      if (retained[old_vertex] >= 0) return retained[old_vertex];
      if (next_used == vertex_limit) throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
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
      for (int a = 0; a + 1 < plane_count && first_plane < 0; ++a) {
        if (!(common & (1u << a))) continue;
        for (int b = a + 1; b < plane_count; ++b) {
          if (!(common & (1u << b))) continue;
          charge(18);
          denominator = reference_minor3(planes[a], planes[b], planes[clipping], 0, 1, 2);
          if (denominator.sign()) { first_plane = a; second_plane = b; break; }
        }
      }
      if (first_plane < 0 || next_used == vertex_limit || crossing_count == crossing_limit)
        throw Refusal{PHX_MC_INTERNAL_ERROR, -1};
      const auto& a = planes[first_plane];
      const auto& b = planes[second_plane];
      const auto& c = planes[clipping];
      charge(54);
      ExactReferenceVertex3 vertex;
      vertex.coordinates = reference_plane_triple_point(a, b, c, std::move(denominator));
      vertex.support = {first_plane, second_plane, clipping};
      // A strict edge crossing lies on every common plane and the clipping
      // plane. Every other processed plane is strictly negative by convexity.
      // Future zero incidences are discovered by their own clipping pass
      // before use; an independent processed support pair always comes first.
      vertex.incidence = common | static_cast<uint16_t>(1u << clipping);
      // No additional plane evaluation is needed to preserve the final witness.
      const int32_t row = next_used++;
      next_vertices[row] = std::move(vertex);
      crossings[crossing_count++] = {low, high, row};
      return row;
    };
    const auto on_plane = static_cast<uint16_t>(1u << clipping);
    if (!reference_section(
            faces, face_used, sides, keep,
            [&](int, int32_t first, int32_t second) { return crossing(first, second); },
            [&](int32_t vertex) { return (next_vertices[vertex].incidence & on_plane) != 0; },
            next_used, plane_count, clipping, next_faces, next_face_used, charge))
      return PHX_MC_INTERNAL_ERROR;
    vertices.swap(next_vertices);
    faces.swap(next_faces);
    used = next_used;
    face_used = next_face_used;
  }
  if (face_used < 4) return PHX_MC_INTERNAL_ERROR;
  // Compact only genuine facet-used vertices; all supporting-plane source
  // witnesses survive. Nothing is committed before all capacities hold.
  std::array<int32_t, kReferenceTetraMaximumVertices> rows;
  rows.fill(-1);
  int count = 0, entries = 0;
  for (int face = 0; face < face_used; ++face) {
    entries += faces[face].count;
    for (int slot = 0; slot < faces[face].count; ++slot)
      rows[faces[face].vertices[slot]] = 0;
  }
  if (entries > entry_limit) return PHX_MC_INTERNAL_ERROR;
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
