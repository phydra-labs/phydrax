#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

# Ensure JAX uses 64-bit floats by default for numerical robustness
import jax


jax.config.update("jax_enable_x64", True)

from . import (
    constraints,
    data_utils,
    domain,
    equations,
    export,
    graph,
    integration,
    metrix,
    nn,
    objectives,
    operators,
    sampling,
    solver,
    stochastic,
    uq,
)


# Explicit re-exports for star import
__all__ = [
    "constraints",
    "data_utils",
    "domain",
    "equations",
    "export",
    "objectives",
    "integration",
    "graph",
    "metrix",
    "nn",
    "operators",
    "sampling",
    "solver",
    "stochastic",
    "uq",
]
