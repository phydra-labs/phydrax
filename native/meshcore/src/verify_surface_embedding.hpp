// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once
#include <vector>

#include "phydrax_meshcore.h"

namespace phx::mc {
// True only when exact predicates certify that no undeclared contact exists.
// False requires the full triangle classifier.
bool surface_embedding_pair_clear(const double* const first[3], const double* const second[3],
                                  const int64_t* first_ids, const int64_t* second_ids);

struct SurfaceContact {
  int64_t first, second;
  int32_t kind;
};
// Every pair of nondegenerate triangles (3D points, vertex ids into `points`) whose exact
// contact is not disjoint, a shared vertex or a shared edge by id; each unordered pair once,
// first < second. Requires the caller's GradualNearestScope and in-domain coordinates.
int32_t surface_embedding_contacts(const double* points, int64_t triangle_count,
                                   const int64_t* triangles, std::vector<SurfaceContact>& out);
}

