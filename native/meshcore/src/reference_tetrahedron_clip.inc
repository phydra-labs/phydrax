// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Included inside arrangement.cpp after its exact scalar helpers.
// Plane convention: a*x+b*y+c*z+d <= 0. All topology is constructed and
// classified from exact dyadics; rounded coordinates are never an input.

struct ExactReferenceVertex3 {
  std::array<Dyadic, 4> coordinates;
  std::array<int32_t, 3> support{};
  std::vector<uint8_t> incidence;
};

struct ExactReferenceFace3 {
  std::vector<int32_t> vertices;
  int32_t plane = 0;
};

using ExactReferencePlane3 = std::array<Dyadic, 4>;
using ExactReferencePoint3 = std::array<Dyadic, 4>;

Dyadic reference_minor3(const ExactReferencePlane3& a, const ExactReferencePlane3& b,
                        const ExactReferencePlane3& c, int i, int j, int k) {
  return exact::det3(a[i], a[j], a[k], b[i], b[j], b[k], c[i], c[j], c[k]);
}

int reference_plane_side(const ExactReferencePlane3& plane, const ExactReferencePoint3& point) {
  Dyadic value;
  for (int axis = 0; axis < 4; ++axis) value = value + plane[axis] * point[axis];
  return value.sign();
}

ExactReferencePoint3 reference_plane_triple_point(const ExactReferencePlane3& a,
                                                   const ExactReferencePlane3& b,
                                                   const ExactReferencePlane3& c,
                                                   Dyadic denominator) {
  ExactReferencePoint3 point;
  point[0] = -reference_minor3(a, b, c, 1, 2, 3);
  point[1] = reference_minor3(a, b, c, 0, 2, 3);
  point[2] = -reference_minor3(a, b, c, 0, 1, 3);
  point[3] = std::move(denominator);
  if (point[3].sign() < 0) {
    for (auto& value : point) value.negate();
  }
  return point;
}

// Pair minors p_ij = a_i b_j - a_j b_i in order 01,02,03,12,13,23.
std::array<Dyadic, 6> reference_pair_minors(const ExactReferencePlane3& a,
                                            const ExactReferencePlane3& b) {
  return {a[0] * b[1] - a[1] * b[0], a[0] * b[2] - a[2] * b[0],
          a[0] * b[3] - a[3] * b[0], a[1] * b[2] - a[2] * b[1],
          a[1] * b[3] - a[3] * b[1], a[2] * b[3] - a[3] * b[2]};
}


ExactReferencePoint3 reference_pair_point(const std::array<Dyadic, 6>& p,
                                           const ExactReferencePlane3& c,
                                           Dyadic denominator) {
  ExactReferencePoint3 point;
  point[0] = -c[1] * p[5] + c[2] * p[4] - c[3] * p[3];
  point[1] = c[0] * p[5] - c[2] * p[2] + c[3] * p[1];
  point[2] = -c[0] * p[4] + c[1] * p[2] - c[3] * p[0];
  point[3] = std::move(denominator);
  if (point[3].sign() < 0) {
    for (auto& value : point) value.negate();
  }
  return point;
}

bool reference_cap_walk(const std::map<int32_t, int32_t>& successor, int32_t first,
                        std::size_t edges, std::size_t used, int32_t clipping,
                        ExactReferenceFace3& cap) {
  cap.plane = clipping;
  if (first < 0 || edges < 3) return false;
  int32_t vertex = first;
  do {
    if (vertex < 0 || static_cast<std::size_t>(vertex) >= used || cap.vertices.size() == edges)
      return false;
    cap.vertices.push_back(vertex);
    const auto found = successor.find(vertex);
    if (found == successor.end()) return false;
    vertex = found->second;
  } while (vertex != first);
  return cap.vertices.size() == edges;
}

// One exact halfspace section of a bounded convex polytope. Retained facets
// preserve their orientation and label; the opposite cap is walked from the
// reversed on-plane boundary edges.
template <class Sides, class Keep, class Crossing, class OnPlane, class Used>
bool reference_section(const std::vector<ExactReferenceFace3>& faces, const Sides& sides,
                       Keep&& keep, Crossing&& crossing, OnPlane&& on_plane, Used&& used,
                       int32_t clipping, std::vector<ExactReferenceFace3>& next_faces) {
  std::map<int32_t, int32_t> successor;
  int32_t first = -1;
  std::size_t edges = 0;
  bool boundary = true;
  next_faces.clear();
  for (const ExactReferenceFace3& old : faces) {
    ExactReferenceFace3 clipped;
    clipped.plane = old.plane;
    std::vector<std::array<int32_t, 2>> pending;
    auto append = [&](int32_t vertex) {
      if (!clipped.vertices.empty() && clipped.vertices.back() == vertex) return;
      if (!clipped.vertices.empty()) {
        const int32_t previous = clipped.vertices.back();
        if (on_plane(previous) && on_plane(vertex)) pending.push_back({previous, vertex});
      }
      clipped.vertices.push_back(vertex);
    };
    for (std::size_t slot = 0; slot < old.vertices.size(); ++slot) {
      const int32_t a = old.vertices[slot];
      const int32_t b = old.vertices[(slot + 1) % old.vertices.size()];
      const bool first_inside = sides[static_cast<std::size_t>(a)] <= 0;
      const bool second_inside = sides[static_cast<std::size_t>(b)] <= 0;
      if (first_inside != second_inside) append(crossing(a, b));
      if (second_inside) append(keep(b));
    }
    if (clipped.vertices.size() > 1 && clipped.vertices.front() == clipped.vertices.back()) {
      clipped.vertices.pop_back();
    } else if (clipped.vertices.size() > 1) {
      const int32_t last = clipped.vertices.back(), start = clipped.vertices.front();
      if (on_plane(last) && on_plane(start)) pending.push_back({last, start});
    }
    if (clipped.vertices.size() >= 3) {
      next_faces.push_back(std::move(clipped));
      for (const auto& edge : pending) {
        const int32_t a = edge[0], b = edge[1];
        if (a == b || !successor.emplace(b, a).second) {
          boundary = false;
          continue;
        }
        first = b;
        ++edges;
      }
    }
  }
  ExactReferenceFace3 cap;
  if (!boundary || !reference_cap_walk(successor, first, edges, used(), clipping, cap))
    return false;
  next_faces.push_back(std::move(cap));
  return true;
}

int32_t clip_reference_tetrahedron_exact(const uint32_t* words, int64_t word_count,
                                         const int64_t* offsets, const int8_t* signs,
                                         int32_t plane_count, ReferenceTetrahedronClip& out) {
  out = {};
  ReferenceTetrahedronClip result;
  result.incident_offsets.push_back(0);
  result.face_offsets.push_back(0);
  const int64_t scalar_count = 4 * static_cast<int64_t>(plane_count);
  if (plane_count < 4 || word_count < 0 || offsets == nullptr || signs == nullptr ||
      (word_count > 0 && words == nullptr) || offsets[0] != 0 ||
      offsets[scalar_count] != word_count) {
    return PHX_MC_INVALID_ARGUMENT;
  }
  for (int64_t scalar = 0; scalar < scalar_count; ++scalar) {
    const int64_t begin = offsets[scalar], end = offsets[scalar + 1];
    if (begin < 0 || end < begin || end > word_count || signs[scalar] < -1 || signs[scalar] > 1 ||
        ((begin == end) != (signs[scalar] == 0)) || (end > begin && words[end - 1] == 0)) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  std::vector<ExactReferencePlane3> planes(static_cast<std::size_t>(plane_count));
  for (int32_t plane = 0; plane < plane_count; ++plane) {
    for (int axis = 0; axis < 4; ++axis) {
      const int64_t scalar = 4 * static_cast<int64_t>(plane) + axis;
      const int64_t begin = offsets[scalar], end = offsets[scalar + 1];
      planes[static_cast<std::size_t>(plane)][static_cast<std::size_t>(axis)] =
          Dyadic::integer(words == nullptr ? nullptr : words + begin,
                          static_cast<std::size_t>(end - begin), signs[scalar]);
    }
    bool nonzero_normal = false;
    for (int axis = 0; axis < 3; ++axis)
      nonzero_normal = nonzero_normal || planes[static_cast<std::size_t>(plane)][axis].sign() != 0;
    if (!nonzero_normal) return PHX_MC_INVALID_INPUT;
  }
  for (int plane = 0; plane < 3; ++plane) {
    for (int axis = 0; axis < 4; ++axis) {
      if (axis == plane ? planes[plane][axis].sign() >= 0 : planes[plane][axis].sign() != 0)
        return PHX_MC_INVALID_INPUT;
    }
  }
  if (planes[3][0].sign() <= 0 || (planes[3][1] - planes[3][0]).sign() != 0 ||
      (planes[3][2] - planes[3][0]).sign() != 0 || (planes[3][3] + planes[3][0]).sign() != 0) {
    return PHX_MC_INVALID_INPUT;
  }

  std::vector<ExactReferenceVertex3> vertices(4), next_vertices;
  std::vector<ExactReferenceFace3> faces(4), next_faces;
  const int32_t original_support[4][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
  const int32_t original_faces[4][3] = {{0, 3, 2}, {0, 1, 3}, {0, 2, 1}, {1, 2, 3}};
  for (int vertex = 0; vertex < 4; ++vertex) {
    auto& point = vertices[static_cast<std::size_t>(vertex)];
    point.coordinates[3] = Dyadic(1.0);
    if (vertex) point.coordinates[vertex - 1] = Dyadic(1.0);
    point.incidence.assign(static_cast<std::size_t>(plane_count), 0);
    for (int slot = 0; slot < 3; ++slot) {
      point.support[slot] = original_support[vertex][slot];
      point.incidence[static_cast<std::size_t>(original_support[vertex][slot])] = 1;
    }
    auto& face = faces[static_cast<std::size_t>(vertex)];
    face.plane = vertex;
    face.vertices.assign(original_faces[vertex], original_faces[vertex] + 3);
  }

  for (int32_t clipping = 4; clipping < plane_count; ++clipping) {
    std::vector<int> sides(vertices.size());
    bool outside = false, strictly_inside = false;
    for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
      sides[vertex] = reference_plane_side(planes[static_cast<std::size_t>(clipping)],
                                           vertices[vertex].coordinates);
      outside = outside || sides[vertex] > 0;
      strictly_inside = strictly_inside || sides[vertex] < 0;
      if (sides[vertex] == 0)
        vertices[vertex].incidence[static_cast<std::size_t>(clipping)] = 1;
    }
    if (!outside) continue;
    if (!strictly_inside) {
      out = std::move(result);
      return PHX_MC_OK;
    }

    std::vector<int32_t> retained(vertices.size(), -1);
    std::map<std::pair<int32_t, int32_t>, int32_t> crossings;
    next_vertices.clear();
    auto keep = [&](int32_t old_vertex) -> int32_t {
      int32_t& row = retained[static_cast<std::size_t>(old_vertex)];
      if (row >= 0) return row;
      if (next_vertices.size() >= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::length_error("Reference clip vertex ids exceed int32 range.");
      row = static_cast<int32_t>(next_vertices.size());
      next_vertices.push_back(vertices[static_cast<std::size_t>(old_vertex)]);
      return row;
    };
    auto crossing = [&](int32_t first, int32_t second) -> int32_t {
      if (sides[static_cast<std::size_t>(first)] == 0) return keep(first);
      if (sides[static_cast<std::size_t>(second)] == 0) return keep(second);
      const std::pair<int32_t, int32_t> edge{std::min(first, second), std::max(first, second)};
      const auto known = crossings.find(edge);
      if (known != crossings.end()) return known->second;
      const auto& a = vertices[static_cast<std::size_t>(first)];
      const auto& b = vertices[static_cast<std::size_t>(second)];
      int32_t support_a = -1, support_b = -1;
      Dyadic denominator;
      for (int32_t i = 0; i + 1 < plane_count && support_a < 0; ++i) {
        if (!a.incidence[static_cast<std::size_t>(i)] || !b.incidence[static_cast<std::size_t>(i)])
          continue;
        for (int32_t j = i + 1; j < plane_count; ++j) {
          if (!a.incidence[static_cast<std::size_t>(j)] || !b.incidence[static_cast<std::size_t>(j)])
            continue;
          denominator = reference_minor3(planes[static_cast<std::size_t>(i)],
                                         planes[static_cast<std::size_t>(j)],
                                         planes[static_cast<std::size_t>(clipping)], 0, 1, 2);
          if (denominator.sign() != 0) {
            support_a = i;
            support_b = j;
            break;
          }
        }
      }
      if (support_a < 0) throw std::logic_error("Crossing edge has no independent support pair.");
      if (next_vertices.size() >= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::length_error("Reference clip vertex ids exceed int32 range.");
      ExactReferenceVertex3 vertex;
      vertex.coordinates = reference_plane_triple_point(
          planes[static_cast<std::size_t>(support_a)], planes[static_cast<std::size_t>(support_b)],
          planes[static_cast<std::size_t>(clipping)], std::move(denominator));
      vertex.support = {support_a, support_b, clipping};
      vertex.incidence.resize(static_cast<std::size_t>(plane_count));
      for (int32_t plane = 0; plane < plane_count; ++plane) {
        vertex.incidence[static_cast<std::size_t>(plane)] =
            static_cast<uint8_t>(a.incidence[static_cast<std::size_t>(plane)] &&
                                 b.incidence[static_cast<std::size_t>(plane)]);
      }
      vertex.incidence[static_cast<std::size_t>(clipping)] = 1;
      const int32_t row = static_cast<int32_t>(next_vertices.size());
      next_vertices.push_back(std::move(vertex));
      crossings.emplace(edge, row);
      return row;
    };
    auto on_plane = [&](int32_t vertex) {
      return next_vertices[static_cast<std::size_t>(vertex)]
                 .incidence[static_cast<std::size_t>(clipping)] != 0;
    };
    if (!reference_section(faces, sides, keep, crossing, on_plane,
                           [&] { return next_vertices.size(); }, clipping, next_faces)) {
      return PHX_MC_INTERNAL_ERROR;
    }
    vertices.swap(next_vertices);
    faces.swap(next_faces);
  }

  if (faces.size() < 4) return PHX_MC_INTERNAL_ERROR;
  std::vector<int32_t> rows(vertices.size(), -1);
  int32_t count = 0;
  std::size_t entries = 0;
  for (const auto& face : faces) {
    entries += face.vertices.size();
    for (int32_t vertex : face.vertices) rows[static_cast<std::size_t>(vertex)] = 0;
  }
  for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
    if (rows[vertex] == 0) rows[vertex] = count++;
  }
  result.supporting_planes.reserve(3 * static_cast<std::size_t>(count));
  result.incident_offsets.clear();
  result.incident_offsets.push_back(0);
  for (std::size_t vertex = 0; vertex < vertices.size(); ++vertex) {
    if (rows[vertex] < 0) continue;
    const auto& current = vertices[vertex];
    result.supporting_planes.insert(result.supporting_planes.end(), current.support.begin(),
                                    current.support.end());
    for (int32_t plane = 0; plane < plane_count; ++plane) {
      if (current.incidence[static_cast<std::size_t>(plane)])
        result.incident_planes.push_back(plane);
    }
    if (result.incident_planes.size() >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
      return PHX_MC_INDEX_RANGE_ERROR;
    result.incident_offsets.push_back(static_cast<int32_t>(result.incident_planes.size()));
  }
  result.face_offsets.clear();
  result.face_offsets.push_back(0);
  for (auto& face : faces) {
    const auto minimum = std::min_element(face.vertices.begin(), face.vertices.end(),
                                          [&](int32_t a, int32_t b) {
                                            return rows[static_cast<std::size_t>(a)] <
                                                   rows[static_cast<std::size_t>(b)];
                                          });
    std::rotate(face.vertices.begin(), minimum, face.vertices.end());
    result.face_planes.push_back(face.plane);
    for (int32_t vertex : face.vertices)
      result.face_vertices.push_back(rows[static_cast<std::size_t>(vertex)]);
    if (result.face_vertices.size() >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
      return PHX_MC_INDEX_RANGE_ERROR;
    result.face_offsets.push_back(static_cast<int32_t>(result.face_vertices.size()));
  }
  if (entries != result.face_vertices.size()) return PHX_MC_INTERNAL_ERROR;
  out = std::move(result);
  return PHX_MC_OK;
}
