#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from .._sampling import (
    host_design,
    host_design_factory,
    normalize_design_name,
    seed_from_key,
    unit_design,
)
from ._quadrature_rules import (
    clenshaw_curtis_data,
    gauss_kronrod_data,
    gauss_legendre_data,
    QuadratureRuleData,
    tanh_sinh_data,
)
from ._smolyak import (
    axis_level,
    dense_index,
    normalize_anisotropy,
    normalize_axis_rules,
    smolyak_axis_data,
    smolyak_terms,
    SmolyakAxisData,
    SmolyakAxisRule,
    SmolyakTerm,
    weighted_total_degree_indices,
)
from ._stable_reductions import log_normalize, signed_logsumexp, weight_ess
from ._weighted_moments import (
    LogWeightedAccumulator,
    weighted_diagnostics,
    WeightedMomentsDiagnostics,
)


__all__ = [
    "SmolyakAxisData",
    "SmolyakAxisRule",
    "SmolyakTerm",
    "LogWeightedAccumulator",
    "QuadratureRuleData",
    "WeightedMomentsDiagnostics",
    "axis_level",
    "clenshaw_curtis_data",
    "gauss_kronrod_data",
    "gauss_legendre_data",
    "host_design",
    "host_design_factory",
    "log_normalize",
    "normalize_design_name",
    "dense_index",
    "seed_from_key",
    "normalize_anisotropy",
    "normalize_axis_rules",
    "signed_logsumexp",
    "tanh_sinh_data",
    "smolyak_axis_data",
    "smolyak_terms",
    "unit_design",
    "weight_ess",
    "weighted_diagnostics",
    "weighted_total_degree_indices",
]
