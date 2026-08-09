#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

# Ensure JAX uses 64-bit floats by default for numerical robustness
import jax


jax.config.update("jax_enable_x64", True)

from . import (
    conditions,
    control,
    coresets,
    data_utils,
    domain,
    enforcement,
    equations,
    export,
    geometry,
    graph,
    integration,
    kernels,
    metrix,
    nn,
    operators,
    optim,
    sampling,
    solver,
    sparse,
    special,
    stochastic,
    terms,
    uq,
)


# Explicit re-exports for star import
__all__ = [
    "conditions",
    "control",
    "coresets",
    "terms",
    "data_utils",
    "domain",
    "equations",
    "enforcement",
    "export",
    "integration",
    "geometry",
    "graph",
    "kernels",
    "metrix",
    "nn",
    "operators",
    "optim",
    "sampling",
    "sparse",
    "special",
    "solver",
    "stochastic",
    "uq",
]
