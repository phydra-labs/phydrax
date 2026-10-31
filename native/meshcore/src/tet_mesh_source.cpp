//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
// Exact source-witness validation, ancestry coverage, and orientation authority.
#include "tet_mesh.hpp"
#include "dyadic.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include "facet_coverage.hpp"
#include "geometry3.hpp"
#include "exact_scalar.hpp"
#include "predicates.hpp"

namespace phx::mc {
namespace {

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

int exact_orientation(const Expansion* const points[4][3]) {
  const Expansion x0 = *points[1][0] - *points[0][0];
  const Expansion y0 = *points[1][1] - *points[0][1];
  const Expansion z0 = *points[1][2] - *points[0][2];
  const Expansion x1 = *points[2][0] - *points[0][0];
  const Expansion y1 = *points[2][1] - *points[0][1];
  const Expansion z1 = *points[2][2] - *points[0][2];
  const Expansion x2 = *points[3][0] - *points[0][0];
  const Expansion y2 = *points[3][1] - *points[0][1];
  const Expansion z2 = *points[3][2] - *points[0][2];
  return (x0 * (y1 * z2 - z1 * y2) - y0 * (x1 * z2 - z1 * x2) +
          z0 * (x1 * y2 - y1 * x2)).sign();
}

// These scratch records validate source witnesses only; edit ancestry is
// validated on the original change payload before it enters the mesh state.
TetEditVertex source_witness_record(std::uint64_t id, const double* carrier,
                                    const SourceWitness& witness) {
  TetEditVertex vertex{};
  vertex.id = id;
  std::copy_n(carrier, 3, vertex.carrier.begin());
  vertex.witness = witness;
  vertex.ancestry = {};
  return vertex;
}
}  // namespace

bool TetMesh::source_carrier_valid(const TetEditVertex& vertex) const {
  for (double coordinate : vertex.carrier) {
    if (!std::isfinite(coordinate) || !coordinate_in_domain(coordinate)) return false;
  }
  if (vertex.has_edge_parameter) {
    if (!std::isfinite(vertex.edge_parameter) ||
        !(vertex.edge_parameter > 0.0 && vertex.edge_parameter < 1.0) ||
        vertex.edge_endpoints[0] == vertex.edge_endpoints[1]) return false;
    const std::int32_t first = vertex_slot(vertex.edge_endpoints[0]);
    const std::int32_t second = vertex_slot(vertex.edge_endpoints[1]);
    if (!live_vertex(first) || !live_vertex(second)) return false;
    const Dyadic parameter(vertex.edge_parameter);
    const Dyadic one(1.0);
    for (int axis = 0; axis < 3; ++axis) {
      const Dyadic origin(point(first)[axis]);
      const Dyadic destination(point(second)[axis]);
      const Dyadic exact_coordinate =
          origin + (destination - origin) * parameter;
      const double rounded = exact::rounded_quotient(exact_coordinate, one);
      if (!same_bits(rounded, vertex.carrier[static_cast<std::size_t>(axis)])) return false;
    }
  }
  if (!vertex.ancestry.empty()) {
    Dyadic weight_sum(0.0);
    const Dyadic one(1.0);
    std::array<Dyadic, 3> coordinates{
        Dyadic(0.0), Dyadic(0.0), Dyadic(0.0)};
    for (std::size_t row = 0; row < vertex.ancestry.size(); ++row) {
      const TetVertexAncestry& ancestor = vertex.ancestry[row];
      if (!std::isfinite(ancestor.weight) || ancestor.weight < 0.0 ||
          ancestor.weight > 1.0) return false;
      for (std::size_t prior = 0; prior < row; ++prior) {
        if (vertex.ancestry[prior].source_vertex_id == ancestor.source_vertex_id) return false;
      }
      const std::int32_t source_slot = vertex_slot(ancestor.source_vertex_id);
      if (!live_vertex(source_slot)) return false;
      const Dyadic weight(ancestor.weight);
      weight_sum = weight_sum + weight;
      for (int axis = 0; axis < 3; ++axis) {
        coordinates[static_cast<std::size_t>(axis)] =
            coordinates[static_cast<std::size_t>(axis)] +
            Dyadic(point(source_slot)[axis]) * weight;
      }
    }
    if (!(weight_sum - one).is_zero()) return false;
    for (int axis = 0; axis < 3; ++axis) {
      const double rounded = exact::rounded_quotient(
          coordinates[static_cast<std::size_t>(axis)], one);
      if (!same_bits(rounded, vertex.carrier[static_cast<std::size_t>(axis)])) return false;
    }
  }
  const SourceWitness& witness = vertex.witness;
  if (witness.stratum == SourceStratum::kNone) {
    return witness.entity == -1 && witness.deviation == 0.0;
  }
  if (!source_parameters_valid(witness.stratum, witness.parameters) ||
      !std::isfinite(witness.deviation) || witness.deviation < 0.0) {
    return false;
  }
  // The exact source point S of the witness is authoritative and the carrier
  // must be its correctly rounded realization. The recorded deviation must
  // bound |carrier - S|; it may exceed the entity tolerance (which governs an
  // authored approximation) only when it is exactly the realization rounding
  // bound, i.e. the carrier is round(S) and nothing else was approximated.
  const double* corners[3] = {nullptr, nullptr, nullptr};
  double carrier[3];
  double deviation = 0.0;
  if (!source_corners(witness.stratum, witness.entity, corners)) return false;
  const std::size_t row = static_cast<std::size_t>(witness.entity);
  const double tolerance = witness.stratum == SourceStratum::kSegment
                               ? source_segment_tolerances_[row]
                               : source_face_tolerances_[row];
  if (!source_carrier(corners, witness.stratum, witness.parameters, carrier, deviation) ||
      witness.deviation < deviation ||
      (witness.deviation > tolerance && witness.deviation != deviation)) {
    return false;
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (!same_bits(carrier[axis], vertex.carrier[static_cast<std::size_t>(axis)])) return false;
  }
  return true;
}

bool TetMesh::source_corners(SourceStratum stratum, std::int32_t row,
                             const double** corners) const {
  if (row < 0) return false;
  const std::size_t entity = static_cast<std::size_t>(row);
  if (stratum == SourceStratum::kSegment) {
    if (entity >= source_segments_.size() / 2) return false;
    for (std::size_t k = 0; k < 2; ++k) {
      corners[k] = source_points_.data() + 3 * static_cast<std::size_t>(source_segments_[2 * entity + k]);
    }
    corners[2] = nullptr;
    return true;
  }
  if (stratum == SourceStratum::kFacet) {
    if (entity >= source_faces_.size() / 3) return false;
    for (std::size_t k = 0; k < 3; ++k) {
      corners[k] = source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * entity + k]);
    }
    return true;
  }
  return false;
}

bool TetMesh::witness_position(const SourceWitness& witness, const double* carrier,
                               Expansion* position) const {
  if (witness.stratum == SourceStratum::kNone) {
    for (int axis = 0; axis < 3; ++axis) position[axis] = Expansion(carrier[axis]);
    return true;
  }
  const double* corners[3] = {nullptr, nullptr, nullptr};
  return source_corners(witness.stratum, witness.entity, corners) &&
         source_point(corners, witness.stratum, witness.parameters, position);
}

bool TetMesh::exact_position(std::int32_t slot, Expansion* position) const {
  return live_vertex(slot) &&
         witness_position(witnesses_[static_cast<std::size_t>(slot)], point(slot), position);
}

bool TetMesh::source_face_valid(const TetEditFacet& facet,
                                const RelocationOverlay* overlay) const {
  if (facet.source < 0 || (facet.orientation != 1 && facet.orientation != -1)) return false;
  std::array<std::int32_t, 3> slots{};
  for (int k = 0; k < 3; ++k) {
    slots[k] = vertex_slot(facet.vertices[k]);
    if (!live_vertex(slots[k])) return false;
    for (int j = 0; j < k; ++j) if (slots[k] == slots[j]) return false;
  }
  const auto overlay_vertex = [&](std::int32_t slot) {
    return overlay == nullptr ? nullptr : overlay->find(vertex_id(slot));
  };
  const auto carrier_at = [&](std::int32_t slot) -> const double* {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr ? point(slot) : changed->carrier.data();
  };
  const auto witness_at = [&](std::int32_t slot) -> const SourceWitness& {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr
               ? witnesses_[static_cast<std::size_t>(slot)]
               : changed->witness;
  };

  std::vector<std::array<std::int32_t, 4>> primitives;
  std::vector<std::int32_t> rows;
  primitives.reserve(source_faces_.size() / 3);
  rows.reserve(source_faces_.size() / 3);
  for (std::size_t row = 0; row < source_faces_.size() / 3; ++row) {
    if (source_face_groups_[row] != facet.source) continue;
    // Coverage uses only the first three fields; retain the exact bank row in slot four.
    primitives.push_back({source_faces_[3 * row], source_faces_[3 * row + 1],
                          source_faces_[3 * row + 2], static_cast<std::int32_t>(row)});
    rows.push_back(static_cast<std::int32_t>(primitives.size() - 1));
  }
  if (rows.empty()) return false;

  double subject[9];
  bool exact_carriers = true;
  for (int vertex = 0; vertex < 3; ++vertex) {
    const std::int32_t slot = slots[vertex];
    const double* carrier = carrier_at(slot);
    std::copy_n(carrier, 3, subject + 3 * vertex);
    const SourceWitness& witness = witness_at(slot);
    if (!source_carrier_valid(source_witness_record(vertex_id(slot), carrier, witness))) return false;
    exact_carriers = exact_carriers && witness.deviation == 0.0;
  }
  std::vector<std::int32_t> contributors;
  if (exact_carriers && exact_facet_coverage(subject, source_points_.data(), primitives,
                                              rows, contributors) && !contributors.empty()) {
    // Coverage returns local primitive rows; slot four maps each to the source bank.
    // Unrelated triangles in one authored group can have different normals.
    for (std::int32_t contributor : contributors) {
      if (contributor < 0 || static_cast<std::size_t>(contributor) >= primitives.size()) {
        return false;
      }
      const std::int32_t source_row = primitives[static_cast<std::size_t>(contributor)][3];
      if (source_row < 0 || static_cast<std::size_t>(source_row) >= source_faces_.size() / 3) {
        return false;
      }
      const std::size_t row = static_cast<std::size_t>(source_row);
      const double* primitive[3] = {
          source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row]),
          source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 1]),
          source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 2])};
      bool oriented = false;
      for (int axis = 0; axis < 3; ++axis) {
        const int first = (axis + 1) % 3;
        const int second = (axis + 2) % 3;
        const double source_a[2] = {primitive[0][first], primitive[0][second]};
        const double source_b[2] = {primitive[1][first], primitive[1][second]};
        const double source_c[2] = {primitive[2][first], primitive[2][second]};
        const int source_orientation = orient2d(source_a, source_b, source_c);
        if (source_orientation == 0) continue;
        const double carrier_a[2] = {subject[first], subject[second]};
        const double carrier_b[2] = {subject[3 + first], subject[3 + second]};
        const double carrier_c[2] = {subject[6 + first], subject[6 + second]};
        const int carrier_orientation = orient2d(carrier_a, carrier_b, carrier_c);
        if (carrier_orientation == 0 ||
            carrier_orientation != source_orientation * facet.orientation) {
          return false;
        }
        oriented = true;
        break;
      }
      if (!oriented) return false;
    }
    return true;
  }

  // Exact source positions, not rounded carriers, authorize an off-plane
  // binary64 realization. A common convex primitive contains the complete
  // source triangle; multi-primitive coverage was proved above. A carrier that
  // is only the realization rounding of its exact source point approximates
  // nothing, so only an authored deviation is held to the row tolerance.
  Expansion exact_points[3][3];
  double point_deviations[3]{};
  for (int vertex = 0; vertex < 3; ++vertex) {
    const std::int32_t slot = slots[vertex];
    const SourceWitness& witness = witness_at(slot);
    const double* carrier = carrier_at(slot);
    if (witness.stratum != SourceStratum::kNone &&
        !source_carrier_valid(source_witness_record(vertex_id(slot), carrier, witness))) {
      return false;
    }
    if (!witness_position(witness, carrier, exact_points[vertex])) return false;
    if (witness.deviation != 0.0 &&
        witness.deviation != outward_distance(carrier, exact_points[vertex])) {
      point_deviations[vertex] = witness.deviation;
    }
  }

  for (std::size_t row = 0; row < source_faces_.size() / 3; ++row) {
    if (source_face_groups_[row] != facet.source) continue;
    if (point_deviations[0] > source_face_tolerances_[row] ||
        point_deviations[1] > source_face_tolerances_[row] ||
        point_deviations[2] > source_face_tolerances_[row]) continue;
    const double* primitive[3] = {
        source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row]),
        source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 1]),
        source_points_.data() + 3 * static_cast<std::size_t>(source_faces_[3 * row + 2])};
    // Projected containment is valid only after exact 3D plane incidence.
    std::array<Expansion, 3> first_edge{};
    std::array<Expansion, 3> second_edge{};
    std::array<Expansion, 3> normal{};
    for (int axis = 0; axis < 3; ++axis) {
      first_edge[static_cast<std::size_t>(axis)] =
          Expansion::difference(primitive[1][axis], primitive[0][axis]);
      second_edge[static_cast<std::size_t>(axis)] =
          Expansion::difference(primitive[2][axis], primitive[0][axis]);
    }
    normal[0] = first_edge[1] * second_edge[2] -
                first_edge[2] * second_edge[1];
    normal[1] = first_edge[2] * second_edge[0] -
                first_edge[0] * second_edge[2];
    normal[2] = first_edge[0] * second_edge[1] -
                first_edge[1] * second_edge[0];
    bool exact_coplanar = true;
    for (int vertex = 0; vertex < 3 && exact_coplanar; ++vertex) {
      std::array<Expansion, 3> offset{};
      for (int axis = 0; axis < 3; ++axis) {
        offset[static_cast<std::size_t>(axis)] =
            exact_points[vertex][axis] - Expansion(primitive[0][axis]);
      }
      const Expansion distance =
          normal[0] * offset[0] + normal[1] * offset[1] +
          normal[2] * offset[2];
      exact_coplanar = distance.is_zero();
    }
    if (!exact_coplanar) continue;
    for (int axis = 0; axis < 3; ++axis) {
      const int first = (axis + 1) % 3;
      const int second = (axis + 2) % 3;
      const double source_a[2] = {primitive[0][first], primitive[0][second]};
      const double source_b[2] = {primitive[1][first], primitive[1][second]};
      const double source_c[2] = {primitive[2][first], primitive[2][second]};
      const int source_orientation = orient2d(source_a, source_b, source_c);
      if (source_orientation == 0) continue;
      Expansion source_projection[3][2];
      for (int corner = 0; corner < 3; ++corner) {
        source_projection[corner][0] = Expansion(primitive[corner][first]);
        source_projection[corner][1] = Expansion(primitive[corner][second]);
      }
      Expansion projected[3][2];
      for (int vertex = 0; vertex < 3; ++vertex) {
        projected[vertex][0] = exact_points[vertex][first];
        projected[vertex][1] = exact_points[vertex][second];
      }
      const auto orientation = [](const Expansion* a, const Expansion* b,
                                  const Expansion* c) {
        return ((b[0] - a[0]) * (c[1] - a[1]) -
                (b[1] - a[1]) * (c[0] - a[0])).sign();
      };
      bool inside = true;
      for (int vertex = 0; vertex < 3; ++vertex) {
        const int s0 = orientation(source_projection[0], source_projection[1],
                                   projected[vertex]);
        const int s1 = orientation(source_projection[1], source_projection[2],
                                   projected[vertex]);
        const int s2 = orientation(source_projection[2], source_projection[0],
                                   projected[vertex]);
        inside = inside && (source_orientation > 0 ? s0 >= 0 && s1 >= 0 && s2 >= 0
                                                   : s0 <= 0 && s1 <= 0 && s2 <= 0);
      }
      if (!inside) break;
      const int subject_orientation = orientation(projected[0], projected[1], projected[2]);
      if (subject_orientation != 0 &&
          subject_orientation == source_orientation * facet.orientation) {
        return true;
      }
      break;
    }
  }
  return false;
}

bool TetMesh::source_segment_valid(const TetEditSegment& segment,
                                   const RelocationOverlay* overlay) const {
  if (segment.source < 0 || segment.vertices[0] == segment.vertices[1]) return false;
  const std::int32_t first = vertex_slot(segment.vertices[0]);
  const std::int32_t second = vertex_slot(segment.vertices[1]);
  if (!live_vertex(first) || !live_vertex(second)) return false;
  const auto overlay_vertex = [&](std::int32_t slot) {
    return overlay == nullptr ? nullptr : overlay->find(vertex_id(slot));
  };
  const auto carrier_at = [&](std::int32_t slot) -> const double* {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr ? point(slot) : changed->carrier.data();
  };
  const auto witness_at = [&](std::int32_t slot) -> const SourceWitness& {
    const TetEditVertex* changed = overlay_vertex(slot);
    return changed == nullptr
               ? witnesses_[static_cast<std::size_t>(slot)]
               : changed->witness;
  };
  const SourceWitness& first_witness = witness_at(first);
  const SourceWitness& second_witness = witness_at(second);
  for (std::size_t row = 0; row < source_segment_groups_.size(); ++row) {
    if (source_segment_groups_[row] != segment.source) continue;
    const double* corners[3] = {
        source_points_.data() + 3 * static_cast<std::size_t>(source_segments_[2 * row]),
        source_points_.data() + 3 * static_cast<std::size_t>(source_segments_[2 * row + 1]),
        nullptr};
    const auto parameter_for = [&](std::int32_t slot, const SourceWitness& witness,
                                   double& parameter) {
      if (witness.stratum == SourceStratum::kSegment &&
          witness.entity == static_cast<std::int32_t>(row)) {
        if (!source_carrier_valid(
                source_witness_record(vertex_id(slot), carrier_at(slot), witness))) return false;
        parameter = witness.parameters[0];
        return true;
      }
      if (witness.stratum != SourceStratum::kNone || witness.entity != -1 ||
          witness.deviation != 0.0) return false;
      double parameters[2]{};
      const double* current_carrier = carrier_at(slot);
      source_locator(corners, SourceStratum::kSegment, current_carrier,
                     parameters, false);
      double carrier[3];
      double deviation = 0.0;
      if (!source_carrier(corners, SourceStratum::kSegment, parameters, carrier, deviation) ||
          deviation != 0.0 || !same_bits(carrier[0], current_carrier[0]) ||
          !same_bits(carrier[1], current_carrier[1]) ||
          !same_bits(carrier[2], current_carrier[2])) {
        return false;
      }
      parameter = parameters[0];
      return true;
    };
    double first_parameter = 0.0;
    double second_parameter = 0.0;
    if (parameter_for(first, first_witness, first_parameter) &&
        parameter_for(second, second_witness, second_parameter) &&
        first_parameter != second_parameter) {
      return true;
    }
  }
  return false;
}

template <bool WithRelocation>
bool TetMesh::positive_impl(
    const std::int32_t* vertices, const RelocationOverlay* overlay) const {
  if (vertices == nullptr) return false;
  const double* carriers[4];
  const SourceWitness* witness_records[4];
  for (int vertex = 0; vertex < 4; ++vertex) {
    const std::int32_t slot = vertices[vertex];
    if (!live_vertex(slot)) return false;
    const TetEditVertex* changed = nullptr;
    if constexpr (WithRelocation) {
      if (overlay == nullptr) return false;
      changed = overlay->find(vertex_id(slot));
      carriers[vertex] = changed == nullptr ? point(slot) : changed->carrier.data();
    } else {
      carriers[vertex] = point(slot);
    }
    witness_records[vertex] =
        &witnesses_[static_cast<std::size_t>(slot)];
    if constexpr (WithRelocation) {
      if (changed != nullptr) witness_records[vertex] = &changed->witness;
    }
  }
  return carriers_positive(carriers, witness_records);
}

bool TetMesh::carriers_positive(const double* const* carriers,
                                const SourceWitness* const* witness_records) const {
  // orient3d uses static error filters and exact fallback. When every witness
  // is kNone, its binary64 carriers are also the exact source points.
  if (orient3d(carriers[0], carriers[1], carriers[2], carriers[3]) <= 0) return false;
  bool all_carrier_points = true;
  for (int vertex = 0; vertex < 4; ++vertex) {
    all_carrier_points =
        all_carrier_points && witness_records[vertex]->stratum == SourceStratum::kNone;
  }
  if (all_carrier_points) return true;

  Expansion exact_storage[4][3];
  const Expansion* exact_points[4][3];
  for (int vertex = 0; vertex < 4; ++vertex) {
    if (!witness_position(*witness_records[vertex], carriers[vertex], exact_storage[vertex])) {
      return false;
    }
    for (int axis = 0; axis < 3; ++axis) exact_points[vertex][axis] = &exact_storage[vertex][axis];
  }
  return exact_orientation(exact_points) > 0;
}

bool TetMesh::cone_positive(const std::int32_t* vertices, int apex,
                            const TetEditVertex& point) const {
  const double* carriers[4];
  const SourceWitness* witness_records[4];
  for (int vertex = 0; vertex < 4; ++vertex) {
    if (vertex == apex) {
      carriers[vertex] = point.carrier.data();
      witness_records[vertex] = &point.witness;
      continue;
    }
    const std::int32_t slot = vertices[vertex];
    if (!live_vertex(slot)) return false;
    carriers[vertex] = this->point(slot);
    witness_records[vertex] = &witnesses_[static_cast<std::size_t>(slot)];
  }
  return carriers_positive(carriers, witness_records);
}

bool TetMesh::positive(const std::int32_t* vertices) const {
  return positive_impl<false>(vertices, nullptr);
}

bool TetMesh::positive_with_relocation(
    const std::int32_t* vertices, const RelocationOverlay& overlay) const {
  return positive_impl<true>(vertices, &overlay);
}

}  // namespace phx::mc
