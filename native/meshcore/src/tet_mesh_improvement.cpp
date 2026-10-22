//
// Copyright © 2026 PHYDRA, Inc. All rights reserved.
//
#include "tet_mesh.hpp"

namespace phx::mc {

namespace geometry {
double minimum_dihedral(const double* const* p) {
  double minimum = 0.0;
  double maximum = 0.0;
  dihedral_extremes(p, minimum, maximum);
  return minimum;
}
}  // namespace geometry

namespace geometry {
void dihedral_extremes(const double* const* p, double& minimum, double& maximum) {
  native_execution_primitive_query();
  static constexpr int kEdges[6][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2},
                                       {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}};
  minimum = 180.0;
  maximum = 0.0;
  for (const auto& edge : kEdges) {
    double e[3], f[3], g[3], n1[3], n2[3];
    difference(p[edge[1]], p[edge[0]], e);
    difference(p[edge[2]], p[edge[0]], f);
    difference(p[edge[3]], p[edge[0]], g);
    cross(e, f, n1);
    cross(e, g, n2);
    const double scale = std::sqrt(dot(n1, n1) * dot(n2, n2));
    double angle = 0.0;
    if (scale > 0.0 && std::isfinite(scale)) {
      double normal_cross[3];
      cross(n1, n2, normal_cross);
      angle =
          std::atan2(std::hypot(normal_cross[0], normal_cross[1], normal_cross[2]), dot(n1, n2)) *
          kDegreesPerRadian;
    }
    minimum = std::min(minimum, angle);
    maximum = std::max(maximum, angle);
  }
}
}  // namespace geometry

TetMesh::PreparedReplacement TetMesh::prepare_replacement(
    std::span<const int32_t> removed, std::span<const std::array<int32_t, 4>> proposed) {
  const MemoryScope memory_scope(memory_owner_);
  cancel_prepared_edit();
  EditTransaction transaction(*this, {});
  if (removed.empty() || proposed.empty()) return {};
  const int32_t region = complex_.region(removed.front());
  for (int32_t t : removed) {
    if (complex_.region(t) != region) return PreparedReplacement(EditStatus::kConstraintViolation);
  }
  reserve_for(slot_reason_, complex_.tets.size() + proposed.size());
  edit_.begin(EditLimits{std::numeric_limits<std::size_t>::max(),
                         std::numeric_limits<std::size_t>::max(), max_tetrahedra_});
  for (int32_t t : removed) {
    const EditStatus status = edit_.remove(t);
    if (status != EditStatus::kOk) return PreparedReplacement(status);
  }
  for (const auto& v : proposed) {
    const EditStatus status = edit_.add(v.data(), nullptr, region);
    if (status != EditStatus::kOk) return PreparedReplacement(status);
  }
  const EditStatus status = edit_.validate([&](const int32_t* v) { return positive(v); });
  if (status != EditStatus::kOk) {
    broken_ = broken_ || status == EditStatus::kInternal;
    return PreparedReplacement(status);
  }
  transaction.release();
  return PreparedReplacement(*this, status);
}

EditStatus TetMesh::replace(std::span<const int32_t> removed,
                            std::span<const std::array<int32_t, 4>> proposed) {
  auto prepared = prepare_replacement(removed, proposed);
  return prepared.status() == EditStatus::kOk ? prepared.commit() : prepared.status();
}

void TetMesh::retire_vertex(int32_t v) {
  dimension_[static_cast<std::size_t>(v)] = -1;
  vertex_tet_[static_cast<std::size_t>(v)] = -1;
  witnesses_[static_cast<std::size_t>(v)] = SourceWitness{};
}

EditStatus TetMesh::commit_prepared_replacement() {
  const MemoryScope memory_scope(memory_owner_);
  EditTransaction transaction(*this, {});
  const EditStatus status = edit_.commit();
  if (status != EditStatus::kOk) {
    broken_ = broken_ || status == EditStatus::kInternal;
    return status;
  }
  transaction.release();
  after_commit();
  return status;
}

double TetMesh::min_dihedral(int32_t t) const {
  const MemoryScope memory_scope(memory_owner_);
  const double* p[4];
  corners(t, p);
  return geometry::minimum_dihedral(p);
}

}  // namespace phx::mc
