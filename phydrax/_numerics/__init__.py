#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._designs import (
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
from ._stable_reductions import log_normalize, signed_logsumexp, weight_ess
from ._weighted_moments import (
    LogWeightedAccumulator,
    weighted_diagnostics,
    WeightedMomentsDiagnostics,
)


__all__ = [
    "LogWeightedAccumulator",
    "QuadratureRuleData",
    "WeightedMomentsDiagnostics",
    "clenshaw_curtis_data",
    "gauss_kronrod_data",
    "gauss_legendre_data",
    "host_design",
    "host_design_factory",
    "log_normalize",
    "normalize_design_name",
    "seed_from_key",
    "signed_logsumexp",
    "tanh_sinh_data",
    "unit_design",
    "weight_ess",
    "weighted_diagnostics",
]
