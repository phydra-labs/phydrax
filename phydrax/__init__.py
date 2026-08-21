#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

# Ensure JAX uses 64-bit floats by default for numerical robustness
import jax


jax.config.update("jax_enable_x64", True)

from . import (
    backends,
    conditions,
    continuation,
    control,
    coresets,
    data_utils,
    domain,
    dynamics,
    enforcement,
    equations,
    export,
    geometry,
    graph,
    integration,
    kernels,
    linalg,
    metrix,
    ml,
    nn,
    nonlinear,
    operators,
    optim,
    sampling,
    solver,
    sparse,
    special,
    stochastic,
    terms,
    transport,
    uq,
    weighting,
)


# Explicit re-exports for star import
__all__ = [
    "backends",
    "conditions",
    "control",
    "continuation",
    "coresets",
    "terms",
    "data_utils",
    "domain",
    "dynamics",
    "equations",
    "enforcement",
    "export",
    "integration",
    "transport",
    "geometry",
    "graph",
    "kernels",
    "linalg",
    "metrix",
    "ml",
    "nn",
    "nonlinear",
    "operators",
    "optim",
    "sampling",
    "sparse",
    "special",
    "solver",
    "stochastic",
    "uq",
    "weighting",
]
