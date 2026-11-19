// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once
#include "phydrax_meshcore.h"

namespace phx::mc {
// True only when exact predicates certify that no undeclared contact exists.
// False requires the full triangle classifier.
bool surface_embedding_pair_clear(const double* const first[3], const double* const second[3],
                                  const int64_t* first_ids, const int64_t* second_ids);
}

