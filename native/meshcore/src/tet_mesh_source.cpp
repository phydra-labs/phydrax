//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

#include <bit>
#include <mutex>
#include <numeric>

#include "facet_coverage.hpp"
#include "spatial_sort.hpp"

namespace phx::mc {

int32_t TetMesh::prepare_source_points(int64_t source_count, const SourceComplex& source) {
  if (source.points == nullptr) {
    original_points_ = points_;
  } else {
    original_points_.assign(source.points, source.points + 3 * source_count);
    for (double value : original_points_) {
      native_execution_charge(TetWorkCost::kVisit);
      if (!std::isfinite(value)) {
        return PHX_MC_INVALID_INPUT;
      }
    }
  }
  return PHX_MC_OK;
}

bool TetMesh::source_vertex_admitted(int32_t vertex, int64_t count) {
  return vertex >= 0 && vertex < count;
}

bool TetMesh::source_bound_admitted(double value) { return std::isfinite(value) && value >= 0.0; }

int32_t TetMesh::build_source_faces(int64_t source_count, const SourceComplex& source) {
  for (int64_t i = 0; i < source.face_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t* f = source.faces + 3 * i;
    const double bound = source.face_tolerances == nullptr ? 0.0 : source.face_tolerances[i];
    if (!source_vertex_admitted(f[0], source_count) ||
        !source_vertex_admitted(f[1], source_count) ||
        !source_vertex_admitted(f[2], source_count) || source.face_sources[i] < 0 ||
        !source_bound_admitted(bound) ||
        geometry::collinear(original_point(f[0]), original_point(f[1]), original_point(f[2]))) {
      return PHX_MC_INVALID_INPUT;
    }
    original_faces_.push_back({f[0], f[1], f[2], source.face_sources[i]});
    face_tolerance_.push_back(bound);
    face_rows_.push_back({source.face_sources[i], static_cast<int32_t>(i)});
  }
  return PHX_MC_OK;
}

int32_t TetMesh::build_source_segments(int64_t source_count, const SourceComplex& source) {
  for (int64_t i = 0; i < source.segment_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const int32_t* s = source.segments + 2 * i;
    const double bound = source.segment_tolerances == nullptr ? 0.0 : source.segment_tolerances[i];
    if (!source_vertex_admitted(s[0], source_count) ||
        !source_vertex_admitted(s[1], source_count) || s[0] == s[1] ||
        source.segment_sources[i] < 0 || !source_bound_admitted(bound) ||
        std::equal(original_point(s[0]), original_point(s[0]) + 3, original_point(s[1]))) {
      return PHX_MC_INVALID_INPUT;
    }
    original_segments_.push_back({s[0], s[1], source.segment_sources[i]});
    segment_tolerance_.push_back(bound);
    segment_rows_.push_back({source.segment_sources[i], static_cast<int32_t>(i)});
  }
  return PHX_MC_OK;
}

void TetMesh::tighten_source_segment_bounds() {
  // A source segment geometrically contained in a source triangle must obey
  // both declarations, including authored subsegments whose endpoints are
  // not the triangle's corner pair.
  for (std::size_t face_row = 0; face_row < original_faces_.size(); ++face_row) {
    const double* face_corners[3];
    row_corners(SourceStratum::kFacet, face_row, face_corners);
    for (std::size_t segment_row = 0; segment_row < original_segments_.size(); ++segment_row) {
      // A zero segment bound cannot be tightened by any nonnegative facet
      // bound; no geometric containment query is needed for that row.
      if (segment_tolerance_[segment_row] == 0.0) continue;
      native_execution_charge(TetWorkCost::kEndpointPair);
      const auto& segment = original_segments_[segment_row];
      if (on_source_entity(face_corners, SourceStratum::kFacet, original_point(segment[0])) &&
          on_source_entity(face_corners, SourceStratum::kFacet, original_point(segment[1]))) {
        segment_tolerance_[segment_row] =
            std::min(segment_tolerance_[segment_row], face_tolerance_[face_row]);
      }
    }
  }
}

int32_t TetMesh::build_source_witnesses(int64_t point_count, const SourceComplex& source) {
  witnesses_.assign(static_cast<std::size_t>(point_count), SourceWitness{});
  if (source.witness_strata == nullptr) {
    return PHX_MC_OK;
  }
  for (int64_t v = 0; v < point_count; ++v) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    SourceWitness witness;
    witness.entity = source.witness_entities[v];
    witness.parameters[0] = source.witness_parameters[2 * v];
    witness.parameters[1] = source.witness_parameters[2 * v + 1];
    switch (source.witness_strata[v]) {
      case static_cast<int8_t>(SourceStratum::kNone):
        if (witness.entity != -1 || witness.parameters[0] != 0.0 || witness.parameters[1] != 0.0) {
          return PHX_MC_INVALID_INPUT;
        }
        continue;
      case static_cast<int8_t>(SourceStratum::kSegment):
        witness.stratum = SourceStratum::kSegment;
        break;
      case static_cast<int8_t>(SourceStratum::kFacet):
        witness.stratum = SourceStratum::kFacet;
        break;
      default:
        return PHX_MC_INVALID_INPUT;
    }
    const std::size_t rows = witness.stratum == SourceStratum::kSegment ? original_segments_.size()
                                                                        : original_faces_.size();
    if (witness.entity < 0 || static_cast<std::size_t>(witness.entity) >= rows) {
      return PHX_MC_INVALID_INPUT;
    }
    const double* corners[3];
    row_corners(witness.stratum, static_cast<std::size_t>(witness.entity), corners);
    witness.deviation =
        source_deviation(corners, witness.stratum, witness.parameters, points_.data() + 3 * v);
    if (witness.deviation < 0.0 ||
        witness.deviation > row_tolerance(witness.stratum, witness.entity) ||
        (witness.deviation > 0.0 &&
         !rounded_carrier(witness.stratum, witness.entity, witness, points_.data() + 3 * v))) {
      return PHX_MC_INVALID_INPUT;
    }
    witnesses_[static_cast<std::size_t>(v)] = witness;
  }
  return PHX_MC_OK;
}

void TetMesh::build_source_face_index() {
  source_face_nodes_.clear();
  source_face_roots_.clear();
  const std::size_t count = original_faces_.size();
  if (count == 0) return;
  std::size_t groups = 0;
  for (std::size_t i = 0; i < face_rows_.size(); ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    groups += i == 0 || face_rows_[i - 1].first != face_rows_[i].first;
  }
  if (count > source_face_nodes_.max_size() / 2) throw std::bad_alloc();
  source_face_nodes_.reserve(2 * count - groups);
  source_face_roots_.reserve(groups);
  // Leaf indices equal original row indices. No rounded construction is used
  // in the bounds, including zero-width boxes on coordinate planes.
  for (std::size_t row = 0; row < count; ++row) {
    native_execution_charge(TetWorkCost::kVisit);
    SourceFaceNode leaf;
    leaf.row = static_cast<int32_t>(row);
    const auto& face = original_faces_[row];
    for (int axis = 0; axis < 3; ++axis) {
      const double a = original_point(face[0])[axis];
      const double b = original_point(face[1])[axis];
      const double c = original_point(face[2])[axis];
      leaf.lower[axis] = std::min({a, b, c});
      leaf.upper[axis] = std::max({a, b, c});
    }
    source_face_nodes_.push_back(leaf);
  }
  NativeVector<std::pair<std::uint64_t, int32_t>> order;
  const auto checkpoint = [] { native_execution_charge(TetWorkCost::kCheckpoint); };
  for (std::size_t first = 0; first < face_rows_.size();) {
    std::size_t end = first + 1;
    while (end < face_rows_.size() && face_rows_[end].first == face_rows_[first].first) {
      checkpoint();
      ++end;
    }
    if (end - first == 1) {
      source_face_roots_.push_back({face_rows_[first].first,
                                    static_cast<std::size_t>(face_rows_[first].second)});
      first = end;
      continue;
    }
    const auto& initial = source_face_nodes_[static_cast<std::size_t>(face_rows_[first].second)];
    double lower[3], upper[3];
    std::copy_n(initial.lower, 3, lower);
    std::copy_n(initial.upper, 3, upper);
    for (std::size_t i = first + 1; i < end; ++i) {
      checkpoint();
      const auto& leaf = source_face_nodes_[static_cast<std::size_t>(face_rows_[i].second)];
      for (int axis = 0; axis < 3; ++axis) {
        lower[axis] = std::min(lower[axis], leaf.lower[axis]);
        upper[axis] = std::max(upper[axis], leaf.upper[axis]);
      }
    }
    order.clear();
    order.reserve(end - first);
    constexpr int bits = 21;  // Shared native 3D Hilbert ordering.
    constexpr double cells = static_cast<double>((std::uint64_t{1} << bits) - 1);
    for (std::size_t i = first; i < end; ++i) {
      const int32_t row = face_rows_[i].second;
      const auto& leaf = source_face_nodes_[static_cast<std::size_t>(row)];
      std::uint32_t quantized[3]{};
      for (int axis = 0; axis < 3; ++axis) {
        const double center = std::midpoint(leaf.lower[axis], leaf.upper[axis]);
        const double extent = 0.5 * upper[axis] - 0.5 * lower[axis];
        const double scaled = extent > 0.0
            ? (0.5 * center - 0.5 * lower[axis]) / extent * cells : 0.0;
        quantized[axis] = static_cast<std::uint32_t>(std::clamp(scaled, 0.0, cells));
      }
      order.emplace_back(hilbert_key(quantized, 3, bits, checkpoint), row);
    }
    std::sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
      checkpoint();
      return a < b;
    });
    const auto build = [&](auto&& self, std::size_t begin, std::size_t finish) -> std::size_t {
      if (finish - begin == 1) return static_cast<std::size_t>(order[begin].second);
      const std::size_t middle = begin + (finish - begin) / 2;
      const std::size_t left = self(self, begin, middle);
      const std::size_t right = self(self, middle, finish);
      native_execution_charge(TetWorkCost::kVisit);
      SourceFaceNode node;
      node.left = left;
      node.right = right;
      for (int axis = 0; axis < 3; ++axis) {
        node.lower[axis] = std::min(source_face_nodes_[left].lower[axis],
                                    source_face_nodes_[right].lower[axis]);
        node.upper[axis] = std::max(source_face_nodes_[left].upper[axis],
                                    source_face_nodes_[right].upper[axis]);
      }
      const std::size_t index = source_face_nodes_.size();
      source_face_nodes_.push_back(node);
      return index;
    };
    source_face_roots_.push_back({face_rows_[first].first, build(build, 0, order.size())});
    first = end;
  }
}

void TetMesh::source_face_candidates(int32_t source, const double* lower, const double* upper,
                                      NativeVector<int32_t>& rows) const {
  rows.clear();
  const auto overlaps = [&](const double* low, const double* high) {
    for (int axis = 0; axis < 3; ++axis)
      if (high[axis] < lower[axis] || low[axis] > upper[axis]) return false;
    return true;
  };
  const auto root = std::lower_bound(source_face_roots_.begin(), source_face_roots_.end(),
      source, [](const auto& entry, int32_t label) { return entry.first < label; });
  if (root != source_face_roots_.end() && root->first == source) {
    // Balanced source-local trees bound recursion by the original row-count
    // logarithm. Closed overlap retains every boundary/edge/point contact.
    const auto visit = [&](auto&& self, std::size_t index) -> void {
      native_execution_charge(TetWorkCost::kVisit);
      const auto& node = source_face_nodes_[index];
      if (!overlaps(node.lower, node.upper)) return;
      if (node.row >= 0) rows.push_back(node.row);
      else {
        self(self, node.left);
        self(self, node.right);
      }
    };
    visit(visit, root->second);
  }
  // Committed curved rows of this source are few and local to their patch.
  const std::pair<int32_t, int32_t> probe{source, std::numeric_limits<int32_t>::min()};
  for (auto claimed = std::lower_bound(claimed_rows_.begin(), claimed_rows_.end(), probe);
       claimed != claimed_rows_.end() && claimed->first == source; ++claimed) {
    native_execution_charge(TetWorkCost::kVisit);
    const auto& face = original_faces_[static_cast<std::size_t>(claimed->second)];
    double low[3], high[3];
    for (int axis = 0; axis < 3; ++axis) {
      const double a = original_point(face[0])[axis];
      const double b = original_point(face[1])[axis];
      const double c = original_point(face[2])[axis];
      low[axis] = std::min({a, b, c});
      high[axis] = std::max({a, b, c});
    }
    if (overlaps(low, high)) rows.push_back(claimed->second);
  }
  std::sort(rows.begin(), rows.end(), [](int32_t a, int32_t b) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    return a < b;
  });
}

double TetMesh::curved_tolerance(int32_t source) const {
  const auto found = std::lower_bound(
      curved_sources_.begin(), curved_sources_.end(), source,
      [](const CurvedSource& entry, int32_t label) { return entry.source < label; });
  return found != curved_sources_.end() && found->source == source
             ? found->tolerance
             : std::numeric_limits<double>::quiet_NaN();
}

const std::uint8_t* TetMesh::curved_digest(int32_t source) const {
  const auto found = std::lower_bound(
      curved_sources_.begin(), curved_sources_.end(), source,
      [](const CurvedSource& entry, int32_t label) { return entry.source < label; });
  return found != curved_sources_.end() && found->source == source ? found->digest.data()
                                                                   : nullptr;
}

int32_t TetMesh::declare_curved_sources(int64_t face_count, const int32_t* face_sources,
                                        const CurvedSourceDeclaration& curved) {
  if (curved.sources == nullptr || curved.tolerances == nullptr || curved.digests == nullptr ||
      (face_count > 0 && curved.face_claims == nullptr) ||
      static_cast<std::size_t>(face_count) != original_faces_.size()) {
    return PHX_MC_INVALID_INPUT;
  }
  curved_sources_.clear();
  curved_claims_.clear();
  claimed_rows_.clear();
  curved_sources_.reserve(static_cast<std::size_t>(curved.source_count));
  for (int64_t i = 0; i < curved.source_count; ++i) {
    native_execution_charge(TetWorkCost::kVisit);
    const double tolerance = curved.tolerances[i];
    if (curved.sources[i] < 0 || !std::isfinite(tolerance) || tolerance < 0.0) {
      return PHX_MC_INVALID_INPUT;
    }
    CurvedSource entry{curved.sources[i], tolerance, {}};
    std::copy_n(curved.digests + 32 * i, 32, entry.digest.begin());
    curved_sources_.push_back(entry);
  }
  std::sort(curved_sources_.begin(), curved_sources_.end(),
            [](const CurvedSource& a, const CurvedSource& b) { return a.source < b.source; });
  for (std::size_t i = 1; i < curved_sources_.size(); ++i) {
    if (curved_sources_[i - 1].source == curved_sources_[i].source) return PHX_MC_INVALID_INPUT;
  }
  for (int64_t i = 0; i < face_count; ++i) {
    native_execution_charge(TetWorkCost::kVisit);
    const double tolerance = curved_tolerance(face_sources[i]);
    const double claim = curved.face_claims[i];
    if (std::isnan(tolerance)) {
      if (!std::isnan(claim)) return PHX_MC_INVALID_INPUT;
      continue;
    }
    if (!std::isfinite(claim) || claim < 0.0 || claim > tolerance) return PHX_MC_INVALID_INPUT;
    curved_claims_.push_back(
        {static_cast<int32_t>(i), face_sources[i], claim, accepted_generation_});
  }
  return PHX_MC_OK;
}

int32_t TetMesh::build_source(int64_t point_count, const SourceComplex& source) {
  const MemoryScope memory_scope(memory_owner_);
  if (source.face_count < 0 || source.segment_count < 0 ||
      (source.face_count > 0 && (source.faces == nullptr || source.face_sources == nullptr)) ||
      (source.segment_count > 0 &&
       (source.segments == nullptr || source.segment_sources == nullptr)) ||
      ((source.witness_strata == nullptr) != (source.witness_entities == nullptr)) ||
      ((source.witness_strata == nullptr) != (source.witness_parameters == nullptr))) {
    return PHX_MC_INVALID_INPUT;
  }
  const int64_t source_count = source.points == nullptr ? point_count : source.point_count;
  if (source_count < 1 || source_count > kMaxMeshPoints) {
    return PHX_MC_INVALID_INPUT;
  }
  int32_t status = prepare_source_points(source_count, source);
  if (status != PHX_MC_OK) return status;
  original_faces_.clear();
  source_face_nodes_.clear();
  source_face_roots_.clear();
  original_segments_.clear();
  face_tolerance_.clear();
  segment_tolerance_.clear();
  face_rows_.clear();
  segment_rows_.clear();
  status = build_source_faces(source_count, source);
  if (status != PHX_MC_OK) return status;
  status = build_source_segments(source_count, source);
  if (status != PHX_MC_OK) return status;
  std::sort(face_rows_.begin(), face_rows_.end());
  std::sort(segment_rows_.begin(), segment_rows_.end());
  build_source_face_index();
  tighten_source_segment_bounds();
  return build_source_witnesses(point_count, source);
}

int32_t TetMesh::complete_boundary_witnesses() {
  for (int32_t v = 0; v < vertex_count(); ++v) {
    native_execution_charge(TetWorkCost::kVisit);
    if (dimension(v) < 0 || dimension(v) >= 3 || witness(v).stratum != SourceStratum::kNone) {
      continue;
    }
    SourceWitness completed;
    const Insertion admitted = relocation_witness(v, point(v), completed);
    if (admitted != Insertion::kOk) {
      return admitted == Insertion::kCapacity ? PHX_MC_CAPACITY_EXCEEDED : PHX_MC_INVALID_INPUT;
    }
    set_witness(v, completed);
  }
  return PHX_MC_OK;
}

int32_t TetMesh::verify_ancestry(int64_t point_count, int64_t face_count, const int32_t* faces,
                                 const int32_t* face_sources, int64_t segment_count,
                                 const int32_t* segments, const int32_t* segment_sources) const {
  const MemoryScope memory_scope(memory_owner_);
  for (int64_t v = 0; v < point_count; ++v) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    const auto index = static_cast<std::size_t>(v);
    if ((witnesses_[index].stratum != SourceStratum::kNone) !=
        (dimension_[index] >= 0 && dimension_[index] < 3)) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  NativeVector<std::array<int32_t, 4>> exported_faces;
  NativeVector<std::size_t> proof_order;
  NativeVector<int64_t> proof_offsets;
  NativeVector<int32_t> proof_rows;
  // Optional preparation for the already-performed constructor proof. Keep
  // room for staging, the retained bank and vector growth; a constrained
  // allocation continues through the ordinary uncached verification path.
  const std::size_t cache_limit = memory_owner_ ? memory_owner_->available_bytes() / 4 : 0;
  const std::size_t per_face =
      2 * sizeof(std::array<int32_t, 4>) + sizeof(std::size_t) +
      2 * sizeof(int64_t) + 9 * sizeof(double) + 3 * sizeof(SourceWitness);
  const auto count = static_cast<std::size_t>(face_count);
  bool ordered = face_count > 0 && count <= cache_limit / (per_face + 12);
  if (ordered) {
    exported_faces.reserve(count);
    collect_faces(exported_faces);
    ordered = exported_faces.size() == count;
    if (ordered) {
      proof_order.assign(count, std::numeric_limits<std::size_t>::max());
      for (std::size_t i = 0; i < count; ++i) {
        native_execution_charge(TetWorkCost::kVisit);
        const FaceKey key = FaceKey::of(faces[3 * i], faces[3 * i + 1], faces[3 * i + 2]);
        std::array<int32_t, 4> probe{key.v[0], key.v[1], key.v[2], face_sources[i]};
        auto found = std::lower_bound(exported_faces.begin(), exported_faces.end(), probe);
        if (found == exported_faces.end() || *found != probe) {
          std::swap(probe[1], probe[2]);
          found = std::lower_bound(exported_faces.begin(), exported_faces.end(), probe);
        }
        if (found == exported_faces.end() || *found != probe) {
          ordered = false;
          break;
        }
        const auto slot = static_cast<std::size_t>(found - exported_faces.begin());
        if (proof_order[slot] != std::numeric_limits<std::size_t>::max()) {
          ordered = false;
          break;
        }
        proof_order[slot] = i;
      }
    }
    if (!ordered) {
      NativeVector<std::array<int32_t, 4>>{}.swap(exported_faces);
      NativeVector<std::size_t>{}.swap(proof_order);
    }
  }
  bool retain = ordered;
  if (retain) {
    proof_offsets.reserve(count + 1);
    proof_rows.reserve(count);
    proof_offsets.push_back(0);
  }
  NativeVector<int32_t> contributors;
  for (int64_t position = 0; position < face_count; ++position) {
    // Contributor sets are independent of triangle cyclic order and reversal.
    // Verify the same input faces once, in their canonical export order.
    const auto i = ordered ? proof_order[static_cast<std::size_t>(position)]
                           : static_cast<std::size_t>(position);
    native_execution_charge(TetWorkCost::kCheckpoint);
    if (!face_ancestry(faces + 3 * i, face_sources[i], contributors)) {
      return PHX_MC_INVALID_INPUT;
    }
    if (retain) {
      const std::size_t row_limit = (cache_limit - per_face * count) / 12;
      if (contributors.size() > row_limit - proof_rows.size()) {
        retain = false;
        NativeVector<int64_t>{}.swap(proof_offsets);
        NativeVector<int32_t>{}.swap(proof_rows);
      } else {
        proof_rows.insert(proof_rows.end(), contributors.begin(), contributors.end());
        proof_offsets.push_back(static_cast<int64_t>(proof_rows.size()));
      }
    }
  }
  for (int64_t i = 0; i < segment_count; ++i) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    if (original_segment_ancestor(segments[2 * i], segments[2 * i + 1], segment_sources[i]) < 0) {
      return PHX_MC_INVALID_INPUT;
    }
  }
  for (int32_t t = 0; t < static_cast<int32_t>(complex_.tets.size()); ++t) {
    if (finite(t) && !positive(tet(t).v)) return PHX_MC_INVALID_INPUT;
  }
  if (retain) store_face_ancestry(exported_faces, proof_offsets, proof_rows);
  return PHX_MC_OK;
}

void TetMesh::row_corners(SourceStratum stratum, std::size_t row, const double** corners) const {
  if (stratum == SourceStratum::kSegment) {
    const auto& segment = original_segments_[row];
    corners[0] = original_point(segment[0]);
    corners[1] = original_point(segment[1]);
    return;
  }
  const auto& face = original_faces_[row];
  for (int k = 0; k < 3; ++k) {
    corners[k] = original_point(face[static_cast<std::size_t>(k)]);
  }
}

bool TetMesh::witness_on_facet(const SourceWitness& witness, std::size_t row) const {
  if (witness.stratum != SourceStratum::kSegment && witness.stratum != SourceStratum::kFacet) {
    return false;
  }
  const double* witness_corners[3];
  row_corners(witness.stratum, static_cast<std::size_t>(witness.entity), witness_corners);
  Expansion source[3];
  if (!source_point(witness_corners, witness.stratum, witness.parameters, source)) {
    return false;
  }
  const double* triangle[3];
  row_corners(SourceStratum::kFacet, row, triangle);
  Expansion first[3], second[3], normal[3];
  for (int axis = 0; axis < 3; ++axis) {
    first[axis] = Expansion::difference(triangle[1][axis], triangle[0][axis]);
    second[axis] = Expansion::difference(triangle[2][axis], triangle[0][axis]);
  }
  normal[0] = first[1] * second[2] + (first[2] * second[1]).scaled(-1.0);
  normal[1] = first[2] * second[0] + (first[0] * second[2]).scaled(-1.0);
  normal[2] = first[0] * second[1] + (first[1] * second[0]).scaled(-1.0);
  Expansion plane;
  for (int axis = 0; axis < 3; ++axis) {
    plane = plane + normal[axis] * (source[axis] + Expansion(-triangle[0][axis]));
  }
  if (plane.sign() != 0) {
    return false;
  }
  int drop = 0;
  while (drop < 3 && normal[drop].sign() == 0) {
    ++drop;
  }
  if (drop == 3) {
    return false;
  }
  const int x = (drop + 1) % 3;
  const int y = (drop + 2) % 3;
  const int orientation = normal[drop].sign();
  for (int edge = 0; edge < 3; ++edge) {
    const double* a = triangle[edge];
    const double* b = triangle[(edge + 1) % 3];
    const Expansion turn =
        Expansion::difference(b[x], a[x]) * (source[y] + Expansion(-a[y])) +
        (Expansion::difference(b[y], a[y]) * (source[x] + Expansion(-a[x]))).scaled(-1.0);
    if (turn.sign() != 0 && turn.sign() != orientation) {
      return false;
    }
  }
  return true;
}

bool TetMesh::on_source_row(int32_t v, SourceStratum stratum, std::size_t row) const {
  const SourceWitness& witness = v == pending_ ? pending_source_ : witnesses_[static_cast<std::size_t>(v)];
  const double* corners[3];
  row_corners(stratum, row, corners);
  if (witness.deviation == 0.0) {
    return on_source_entity(corners, stratum, point(v));
  }
  if (witness.stratum == stratum && static_cast<std::size_t>(witness.entity) == row) {
    return true;
  }
  if (stratum != SourceStratum::kFacet) {
    return false;
  }
  return witness.deviation <= face_tolerance_[row] && witness_on_facet(witness, row);
}

int32_t TetMesh::source_ancestor(SourceStratum stratum, int32_t source,
                                 std::span<const int32_t> support) const {
  const MemoryScope memory_scope(memory_owner_);
  if (stratum == SourceStratum::kFacet) {
    for (int32_t vertex : support) {
      const SourceWitness& own = vertex == pending_ ? pending_source_ : witness(vertex);
      if (own.deviation != 0.0) continue;
      // Any containing source row must contain this exact support point.
      // Bounded witnesses alone do not justify a carrier-box filter.
      NativeVector<int32_t> candidates;
      source_face_candidates(source, point(vertex), point(vertex), candidates);
      for (int32_t row : candidates) {
        native_execution_charge(TetWorkCost::kCheckpoint);
        bool contained = true;
        for (int32_t v : support) {
          contained = contained && on_source_row(v, stratum, static_cast<std::size_t>(row));
        }
        if (contained) return row;
      }
      return -1;
    }
  }
  const auto& rows = stratum == SourceStratum::kSegment ? segment_rows_ : face_rows_;
  const std::pair<int32_t, int32_t> probe{source, std::numeric_limits<int32_t>::min()};
  for (auto row = std::lower_bound(rows.begin(), rows.end(), probe);
       row != rows.end() && row->first == source; ++row) {
    native_execution_charge(TetWorkCost::kCheckpoint);
    bool contained = true;
    for (int32_t v : support) {
      contained = contained && on_source_row(v, stratum, static_cast<std::size_t>(row->second));
    }
    if (contained) {
      return row->second;
    }
  }
  return -1;
}

bool TetMesh::face_ancestry(const int32_t* vertices, int32_t source,
                            NativeVector<int32_t>& contributors) const {
  const MemoryScope memory_scope(memory_owner_);
  contributors.clear();
  // Convex single-row witnesses justify bounded carriers. Across primitive
  // edges, a carrier-only planar proof says nothing about a rounded source
  // witness, so that different construction is refused rather than relabeled.
  double subject[9];
  for (int k = 0; k < 3; ++k) {
    const SourceWitness& own = vertices[k] == pending_ ? pending_source_ : witness(vertices[k]);
    if (own.deviation != 0.0) {
      const int32_t primitive =
          source_ancestor(SourceStratum::kFacet, source, std::span<const int32_t>(vertices, 3));
      if (primitive < 0) return false;
      contributors.push_back(primitive);
      return true;
    }
    std::copy_n(point(vertices[k]), 3, subject + 3 * k);
  }
  double lower[3], upper[3];
  for (int axis = 0; axis < 3; ++axis) {
    lower[axis] = std::min({subject[axis], subject[3 + axis], subject[6 + axis]});
    upper[axis] = std::max({subject[axis], subject[3 + axis], subject[6 + axis]});
  }
  NativeVector<int32_t> rows;
  source_face_candidates(source, lower, upper, rows);
  if (rows.size() == 1) {
    for (int k = 0; k < 3; ++k)
      if (!on_source_row(vertices[k], SourceStratum::kFacet, rows.front())) return false;
    contributors.push_back(rows.front());
    return true;
  }
  if (!claimed_rows_.empty()) {
    NativeVector<int64_t> epochs;
    epochs.reserve(rows.size());
    for (int32_t row : rows) {
      native_execution_charge(TetWorkCost::kVisit);
      const auto claim = std::lower_bound(curved_claims_.begin(), curved_claims_.end(), row,
          [](const CurvedRowClaim& value, int32_t key) { return value.row < key; });
      epochs.push_back(claim != curved_claims_.end() && claim->row == row ? claim->generation : 0);
    }
    return exact_facet_history_coverage(subject, original_points_.data(), original_faces_,
                                        rows, epochs, contributors);
  }
  return exact_facet_coverage(subject, original_points_.data(), original_faces_, rows,
                              contributors);
}

bool TetMesh::facet_witness_admitted(const int32_t* vertices, int32_t source,
                                     const SourceWitness& witness) const {
  NativeVector<int32_t> contributors;
  if (!face_ancestry(vertices, source, contributors) ||
      (contributors.size() > 1 && witness.deviation != 0.0)) return false;
  for (int32_t row : contributors) {
    native_execution_charge(TetWorkCost::kVisit);
    if (witness.deviation <= row_tolerance(SourceStratum::kFacet, row) &&
        witness_on_facet(witness, static_cast<std::size_t>(row))) return true;
  }
  return false;
}

namespace {

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool same_witness(const SourceWitness& a, const SourceWitness& b) {
  return a.stratum == b.stratum && a.entity == b.entity &&
         same_bits(a.parameters[0], b.parameters[0]) &&
         same_bits(a.parameters[1], b.parameters[1]) && same_bits(a.deviation, b.deviation);
}

// Serializes memo lookups/stores of const evidence calls that may run on
// different threads for one handle; never held across ancestry computation.
std::mutex ancestry_memo_mutex;

template <class T>
void release_storage(NativeVector<T>& values) noexcept {
  NativeVector<T> empty(values.get_allocator());
  values.swap(empty);
}

}  // namespace

bool TetMesh::cached_face_ancestry(std::span<const std::array<int32_t, 4>> faces,
                                   NativeVector<int64_t>& offsets,
                                   NativeVector<int32_t>& rows) const {
  const std::lock_guard lock(ancestry_memo_mutex);
  if (!ancestry_cached_ || pending_ != kNoPending || faces.size() != ancestry_faces_.size()) {
    return false;
  }
  for (std::size_t i = 0; i < faces.size(); ++i) {
    native_execution_charge(TetWorkCost::kVisit);
    if (faces[i] != ancestry_faces_[i]) return false;
    for (int k = 0; k < 3; ++k) {
      const int32_t v = faces[i][static_cast<std::size_t>(k)];
      const std::size_t slot = 3 * i + static_cast<std::size_t>(k);
      const double* p = point(v);
      for (int axis = 0; axis < 3; ++axis) {
        if (!same_bits(p[axis], ancestry_points_[3 * slot + static_cast<std::size_t>(axis)])) {
          return false;
        }
      }
      if (!same_witness(witness(v), ancestry_witnesses_[slot])) return false;
    }
  }
  offsets.assign(ancestry_offsets_.begin(), ancestry_offsets_.end());
  rows.assign(ancestry_rows_.begin(), ancestry_rows_.end());
  return true;
}

void TetMesh::release_face_ancestry() const noexcept {
  const std::lock_guard lock(ancestry_memo_mutex);
  ancestry_cached_ = false;
  release_storage(ancestry_faces_);
  release_storage(ancestry_points_);
  release_storage(ancestry_witnesses_);
  release_storage(ancestry_offsets_);
  release_storage(ancestry_rows_);
}

void TetMesh::store_face_ancestry(std::span<const std::array<int32_t, 4>> faces,
                                  std::span<const int64_t> offsets,
                                  std::span<const int32_t> rows) const {
  // A replaced memo returns its storage before the new one is sized, so a
  // stale memo never holds managed bytes against later work.
  release_face_ancestry();
  const std::lock_guard lock(ancestry_memo_mutex);
  if (pending_ != kNoPending) return;
  const std::size_t bytes = faces.size() * (sizeof(std::array<int32_t, 4>) +
                                            9 * sizeof(double) + 3 * sizeof(SourceWitness)) +
                            offsets.size() * sizeof(int64_t) + rows.size() * sizeof(int32_t);
  // Room for the payload and one growth copy; never a refused request.
  if (!memory_owner_ || bytes > memory_owner_->available_bytes() / 2) return;
  try {
    ancestry_faces_.reserve(faces.size());
    ancestry_points_.reserve(9 * faces.size());
    ancestry_witnesses_.reserve(3 * faces.size());
    ancestry_offsets_.reserve(offsets.size());
    ancestry_rows_.reserve(rows.size());
  } catch (const std::bad_alloc&) {
    release_storage(ancestry_faces_);
    release_storage(ancestry_points_);
    release_storage(ancestry_witnesses_);
    release_storage(ancestry_offsets_);
    release_storage(ancestry_rows_);
    return;
  }
  for (const auto& face : faces) {
    ancestry_faces_.push_back(face);
    for (int k = 0; k < 3; ++k) {
      const int32_t v = face[static_cast<std::size_t>(k)];
      const double* p = point(v);
      ancestry_points_.insert(ancestry_points_.end(), p, p + 3);
      ancestry_witnesses_.push_back(witness(v));
    }
  }
  ancestry_offsets_.assign(offsets.begin(), offsets.end());
  ancestry_rows_.assign(rows.begin(), rows.end());
  ancestry_cached_ = true;
}

bool TetMesh::source_position(int32_t vertex, Expansion* coordinates) const {
  const SourceWitness& source = vertex == pending_ ? pending_source_ : witness(vertex);
  if (source.deviation > 0.0) {
    const double* corners[3];
    row_corners(source.stratum, static_cast<std::size_t>(source.entity), corners);
    if (!on_source_entity(corners, source.stratum, point(vertex))) {
      return source_point(corners, source.stratum, source.parameters, coordinates);
    }
  }
  for (int axis = 0; axis < 3; ++axis) coordinates[axis] = Expansion(point(vertex)[axis]);
  return true;
}

int TetMesh::source_orientation(const int32_t* v, int carrier_sign) const {
  const MemoryScope memory_scope(memory_owner_);
  bool bounded = false;
  for (int k = 0; k < 4; ++k) {
    bounded =
        bounded || (v[k] == pending_ ? pending_source_.deviation : witness(v[k]).deviation) > 0.0;
  }
  if (!bounded)
    return carrier_sign == 2 ? orient3d(point(v[0]), point(v[1]), point(v[2]), point(v[3]))
                             : carrier_sign;
  native_execution_primitive_query();
  Expansion coordinates[4][3];
  for (int k = 0; k < 4; ++k) {
    if (!source_position(v[k], coordinates[k])) return 0;
  }
  Expansion u[3], w[3], z[3];
  for (int axis = 0; axis < 3; ++axis) {
    u[axis] = coordinates[1][axis] - coordinates[0][axis];
    w[axis] = coordinates[2][axis] - coordinates[0][axis];
    z[axis] = coordinates[3][axis] - coordinates[0][axis];
  }
  return (u[0] * (w[1] * z[2] - w[2] * z[1]) - u[1] * (w[0] * z[2] - w[2] * z[0]) +
          u[2] * (w[0] * z[1] - w[1] * z[0]))
      .sign();
}

Insertion TetMesh::edge_construction_witness(int32_t a, int32_t b, const double* position,
                                             SourceWitness& witness) {
  NativeVector<int32_t> star, ring;
  if (!edge_ring(a, b, star, ring)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  bool constrained = false;
  for (int32_t t : star) {
    if (!finite(t)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      constrained = constrained || (tet(t).v[k] != a && tet(t).v[k] != b &&
                                    complex_.constraint(t, k) != kNoConstraint);
    }
  }
  return star_witness(a, b, star, position, segment_source(a, b), constrained, witness);
}

Insertion TetMesh::admit_split_facets(int32_t a, int32_t b, const SourceWitness& witness,
                                      std::span<const int32_t> star) {
  for (int32_t t : star) {
    if (!finite(t)) continue;
    for (int k = 0; k < 4; ++k) {
      const int32_t source = complex_.constraint(t, k);
      if (tet(t).v[k] == a || tet(t).v[k] == b || source == kNoConstraint) continue;
      const int32_t support[3] = {a, b, star_apex(t, k, a, b)};
      const int32_t row = source_ancestor(SourceStratum::kFacet, source, support);
      if (row >= 0 && witness_on_facet(witness, static_cast<std::size_t>(row)) &&
          witness.deviation > face_tolerance_[static_cast<std::size_t>(row)]) {
        const double tolerance = face_tolerance_[static_cast<std::size_t>(row)];
        source_refusals_.push_back(
            {{a, b}, SourceStratum::kFacet, row, witness.deviation, tolerance});
        return Insertion::kDeviation;
      }
      if (!facet_witness_admitted(support, source, witness)) {
        return Insertion::kRefused;
      }
    }
  }
  return Insertion::kOk;
}

Insertion TetMesh::construct_source_split(int32_t a, int32_t b, double fraction, double* position,
                                          SourceWitness& witness) try {
  const MemoryScope memory_scope(memory_owner_);
  if (!live_vertex(a) || !live_vertex(b) || a == b || policy_ == BoundaryPolicy::kFixed ||
      !(fraction > 0.0 && fraction < 1.0)) {
    return Insertion::kRefused;
  }
  NativeVector<int32_t> star;
  NativeVector<int32_t> ring;
  if (!edge_ring(a, b, star, ring)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  int axis = 0;
  for (int k = 1; k < 3; ++k) {
    if (std::abs(point(a)[k] - point(b)[k]) > std::abs(point(a)[axis] - point(b)[axis])) {
      axis = k;
    }
  }
  const double low = std::min(point(a)[axis], point(b)[axis]);
  const double high = std::max(point(a)[axis], point(b)[axis]);
  const auto construct = [&](SourceStratum stratum, int32_t row) {
    if (!spend(TetWorkCost::kSourceConstruction)) {
      return Insertion::kCapacity;
    }
    const double* corners[3];
    row_corners(stratum, static_cast<std::size_t>(row), corners);
    double first[2];
    double second[2];
    if (!row_parameters(a, stratum, row, corners, first) ||
        !row_parameters(b, stratum, row, corners, second)) {
      return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
    }
    witness = SourceWitness{stratum, row, {0.0, 0.0}, 0.0};
    for (int k = 0; k < 2; ++k) {
      witness.parameters[k] = first[k] + fraction * (second[k] - first[k]);
    }
    clamp_source_parameters(stratum, witness.parameters);
    if (!source_carrier(corners, stratum, witness.parameters, position, witness.deviation) ||
        !(low < position[axis] && position[axis] < high)) {
      return Insertion::kRefused;
    }
    const double tolerance = row_tolerance(stratum, row);
    if (witness.deviation > tolerance) {
      source_refusals_.push_back({{a, b}, stratum, row, witness.deviation, tolerance});
      return Insertion::kDeviation;
    }
    return admit_split_facets(a, b, witness, star);
  };
  const int32_t segment = segment_source(a, b);
  if (segment >= 0) {
    const int32_t support[2] = {a, b};
    const int32_t row = source_ancestor(SourceStratum::kSegment, segment, support);
    return row < 0 ? Insertion::kRefused : construct(SourceStratum::kSegment, row);
  }
  for (int32_t t : star) {
    if (!finite(t)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      const int32_t source = complex_.constraint(t, k);
      if (tet(t).v[k] == a || tet(t).v[k] == b || source == kNoConstraint) {
        continue;
      }
      const int32_t support[3] = {a, b, star_apex(t, k, a, b)};
      const int32_t row = source_ancestor(SourceStratum::kFacet, source, support);
      if (row >= 0) {
        const Insertion constructed = construct(SourceStratum::kFacet, row);
        if (constructed != Insertion::kRefused) return constructed;
      } else {
        NativeVector<int32_t> contributors;
        if (!face_ancestry(support, source, contributors)) continue;
        double parameter = 0.0;
        if (!construct_edge_point(a, b, position, &parameter, fraction))
          return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
        const Insertion constructed =
            derive_witness(SourceStratum::kFacet, source, support, position, witness);
        if (constructed == Insertion::kOk)
          return admit_split_facets(a, b, witness, star);
        if (constructed != Insertion::kRefused) return constructed;
      }
    }
  }
  return Insertion::kRefused;
} catch (const std::bad_alloc&) {
  return Insertion::kCapacity;
}

Insertion TetMesh::relocation_witness(int32_t v, const double* position, SourceWitness& witness) {
  witness = SourceWitness{};
  if (dimension(v) >= 3) {
    return Insertion::kOk;
  }
  if (dimension(v) <= 1) {
    for (const auto& [edge, source] : segments_) {
      if (edge_low(edge) == v || edge_high(edge) == v) {
        const int32_t support[2] = {edge_low(edge), edge_high(edge)};
        const Insertion result =
            derive_witness(SourceStratum::kSegment, source, support, position, witness);
        return result == Insertion::kOk ? admit_relocation_witness(v, position, witness) : result;
      }
    }
    if (dimension(v) == 1) {
      return Insertion::kRefused;
    }
  }
  if (!vertex_star(v, scratch_star_)) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  for (int32_t t : scratch_star_) {
    if (!finite(t)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      const int32_t source = complex_.constraint(t, k);
      if (tet(t).v[k] == v || source == kNoConstraint) {
        continue;
      }
      int32_t f[3];
      oriented_facet(tet(t).v, k, f);
      const Insertion result =
          derive_witness(SourceStratum::kFacet, source, f, position, witness);
      return result == Insertion::kOk ? admit_relocation_witness(v, position, witness) : result;
    }
  }
  return Insertion::kRefused;
}

Insertion TetMesh::admit_relocation_witness(int32_t v, const double* position,
                                            const SourceWitness& proposed) {
  if (!vertex_star(v, scratch_star_))
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  bool bounded = proposed.deviation != 0.0;
  for (int32_t t : scratch_star_) {
    if (!finite(t)) continue;
    for (int32_t vertex : tet(t).v)
      bounded = bounded || witness(vertex).deviation != 0.0;
  }
  // Exact planar fans already preserve their whole oriented source patch.
  if (!bounded) return Insertion::kOk;
  double* coordinates = mutable_point(v);
  SourceWitness& own = witnesses_[static_cast<std::size_t>(v)];
  struct Restore final {
    double* coordinates;
    SourceWitness& witness;
    std::array<double, 3> point;
    SourceWitness source;
    ~Restore() {
      std::copy(point.begin(), point.end(), coordinates);
      witness = source;
    }
  } restore{coordinates, own, {coordinates[0], coordinates[1], coordinates[2]}, own};
  std::copy_n(position, 3, coordinates);
  own = proposed;
  NativeVector<int32_t> contributors;
  for (int32_t t : scratch_star_) {
    if (!finite(t)) continue;
    for (int slot = 0; slot < 4; ++slot) {
      const int32_t source = complex_.constraint(t, slot);
      if (tet(t).v[slot] == v || source == kNoConstraint) continue;
      int32_t face[3];
      oriented_facet(tet(t).v, slot, face);
      if (!face_ancestry(face, source, contributors)) return Insertion::kRefused;
    }
  }
  return Insertion::kOk;
}

bool TetMesh::exact_row_parameter(const Expansion& numerator, const Expansion& denominator,
                                  double& parameter) {
  std::uint64_t low = 0;
  std::uint64_t high = std::bit_cast<std::uint64_t>(1.0);
  const int denominator_sign = denominator.sign();
  if (denominator_sign == 0) return false;
  while (low <= high) {
    if (!spend(TetWorkCost::kVisit)) return false;
    const std::uint64_t middle = low + (high - low) / 2;
    const double candidate = std::bit_cast<double>(middle);
    const int comparison = (numerator - denominator.scaled(candidate)).sign() * denominator_sign;
    if (comparison == 0) {
      parameter = candidate;
      return true;
    }
    if (comparison > 0)
      low = middle + 1;
    else {
      if (middle == 0) break;
      high = middle - 1;
    }
  }
  return false;
}

bool TetMesh::row_parameters(int32_t v, SourceStratum stratum, int32_t row,
                             const double* const* corners, double* parameters) {
  const SourceWitness& own = witnesses_[static_cast<std::size_t>(v)];
  if (own.deviation == 0.0) {
    source_locator(corners, stratum, point(v), parameters);
    return true;
  }
  if (own.stratum == stratum && own.entity == row) {
    parameters[0] = own.parameters[0];
    parameters[1] = own.parameters[1];
    return true;
  }
  const double* own_corners[3];
  row_corners(own.stratum, static_cast<std::size_t>(own.entity), own_corners);
  Expansion source[3], u[3], r[3];
  if (!source_point(own_corners, own.stratum, own.parameters, source)) return false;
  for (int axis = 0; axis < 3; ++axis) {
    u[axis] = Expansion::difference(corners[1][axis], corners[0][axis]);
    r[axis] = source[axis] - Expansion(corners[0][axis]);
  }
  parameters[1] = 0.0;
  if (stratum == SourceStratum::kSegment) {
    for (int axis = 0; axis < 3; ++axis) {
      if (u[axis].sign() != 0) return exact_row_parameter(r[axis], u[axis], parameters[0]);
    }
    return false;
  }
  Expansion w[3];
  for (int axis = 0; axis < 3; ++axis) {
    w[axis] = Expansion::difference(corners[2][axis], corners[0][axis]);
  }
  for (int drop = 0; drop < 3; ++drop) {
    const int x = (drop + 1) % 3, y = (drop + 2) % 3;
    const Expansion area = u[x] * w[y] - u[y] * w[x];
    if (area.sign() != 0) {
      return exact_row_parameter(r[x] * w[y] - r[y] * w[x], area, parameters[0]) &&
             exact_row_parameter(u[x] * r[y] - u[y] * r[x], area, parameters[1]);
    }
  }
  return false;
}

bool TetMesh::rounded_carrier(SourceStratum stratum, int32_t row, const SourceWitness& witness,
                              const double* position) const {
  const double* corners[3];
  row_corners(stratum, static_cast<std::size_t>(row), corners);
  double carrier[3];
  double deviation = 0.0;
  return source_carrier(corners, stratum, witness.parameters, carrier, deviation) &&
         std::equal(carrier, carrier + 3, position);
}

Insertion TetMesh::derive_witness(SourceStratum stratum, int32_t source,
                                  std::span<const int32_t> support, const double* position,
                                  SourceWitness& witness) {
  witness = SourceWitness{stratum, -1, {0.0, 0.0}, 0.0};
  if (source < 0) {
    return Insertion::kRefused;
  }
  bool exact = true;
  for (int32_t v : support) {
    exact = exact && witnesses_[static_cast<std::size_t>(v)].deviation == 0.0;
  }
  if (!spend(static_cast<int64_t>(TetWorkCost::kFacet * support.size()))) {
    return Insertion::kCapacity;
  }
  const int32_t row =
      exact ? position_row(stratum, source, position) : source_ancestor(stratum, source, support);
  if (row < 0) {
    return work_exhausted_ ? Insertion::kCapacity : Insertion::kRefused;
  }
  const double* corners[3];
  row_corners(stratum, static_cast<std::size_t>(row), corners);
  witness.entity = row;
  source_locator(corners, stratum, position, witness.parameters);
  witness.deviation = source_deviation(corners, stratum, witness.parameters, position);
  if (witness.deviation < 0.0 || (exact && witness.deviation != 0.0) ||
      (witness.deviation > 0.0 && !rounded_carrier(stratum, row, witness, position))) {
    return Insertion::kRefused;
  }
  const double tolerance = row_tolerance(stratum, row);
  if (witness.deviation > tolerance) {
    source_refusals_.push_back(
        {{support[0], support[1]}, stratum, row, witness.deviation, tolerance});
    return Insertion::kDeviation;
  }
  return Insertion::kOk;
}

int32_t TetMesh::position_row(SourceStratum stratum, int32_t source, const double* position) {
  if (stratum == SourceStratum::kFacet) {
    // A point on a closed original triangle is in that triangle's exact
    // coordinate box. The source-local index retains every such row,
    // including contacts on box boundaries and committed curved rows.
    // Sorted candidates preserve the exhaustive lookup's first-row rule.
    NativeVector<int32_t> candidates;
    source_face_candidates(source, position, position, candidates);
    for (int32_t row : candidates) {
      if (!spend(TetWorkCost::kFacet)) return -1;
      const double* corners[3];
      row_corners(stratum, static_cast<std::size_t>(row), corners);
      if (on_source_entity(corners, stratum, position)) return row;
    }
    return -1;
  }
  const auto& rows = stratum == SourceStratum::kSegment ? segment_rows_ : face_rows_;
  const std::pair<int32_t, int32_t> probe{source, std::numeric_limits<int32_t>::min()};
  for (auto row = std::lower_bound(rows.begin(), rows.end(), probe);
       row != rows.end() && row->first == source; ++row) {
    if (!spend(TetWorkCost::kFacet)) {
      return -1;
    }
    const double* corners[3];
    row_corners(stratum, static_cast<std::size_t>(row->second), corners);
    if (on_source_entity(corners, stratum, position)) {
      return row->second;
    }
  }
  return -1;
}

Insertion TetMesh::star_witness(int32_t a, int32_t b, std::span<const int32_t> star,
                                const double* position, int32_t source, bool constrained,
                                SourceWitness& witness) {
  witness = SourceWitness{};
  if (source >= 0) {
    const int32_t support[2] = {a, b};
    return derive_witness(SourceStratum::kSegment, source, support, position, witness);
  }
  if (!constrained) {
    return Insertion::kOk;
  }
  Insertion result = Insertion::kRefused;
  for (int32_t t : star) {
    if (!finite(t)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      const int32_t face_source = complex_.constraint(t, k);
      if (tet(t).v[k] == a || tet(t).v[k] == b || face_source == kNoConstraint) {
        continue;
      }
      const int32_t support[3] = {a, b, star_apex(t, k, a, b)};
      result = derive_witness(SourceStratum::kFacet, face_source, support, position, witness);
      if (result != Insertion::kRefused) {
        return result;
      }
    }
  }
  return result;
}

Insertion TetMesh::pending_witness(InsertKind kind, SourceWitness& witness) {
  witness = SourceWitness{};
  if (pending_source_authored_) {
    witness = pending_source_;
    return Insertion::kOk;
  }
  const double* p = pending_point_;
  if (kind == InsertKind::kSubsegment) {
    const int32_t support[2] = {split_a_, split_b_};
    return derive_witness(SourceStratum::kSegment, segment_source(split_a_, split_b_), support, p,
                          witness);
  }
  if (kind != InsertKind::kSubfacet && crossed_.empty()) {
    return Insertion::kOk;
  }
  const auto attempt = [&](int32_t t, int32_t slot, int32_t source) {
    int32_t f[3];
    oriented_facet(tet(t).v, slot, f);
    if (source == kNoConstraint || f[0] < 0 || f[1] < 0 || f[2] < 0) {
      return Insertion::kRefused;
    }
    const double* triangle[3] = {point(f[0]), point(f[1]), point(f[2])};
    int side = 0;
    return locate_on_triangle(p, triangle, side) == kFeatureNone
               ? Insertion::kRefused
               : derive_witness(SourceStratum::kFacet, source, f, p, witness);
  };
  for (const Crossed& face : crossed_) {
    const Insertion result = attempt(face.tet, face.slot, face.source);
    if (result != Insertion::kRefused) {
      return result;
    }
  }
  for (const BoundaryFacet& face : boundary_) {
    const Insertion result = attempt(face.tet, face.slot, complex_.constraint(face.tet, face.slot));
    if (result != Insertion::kRefused) {
      return result;
    }
  }
  return Insertion::kRefused;
}

}  // namespace phx::mc
