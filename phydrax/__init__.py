#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# ruff: noqa: I001

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
    discretization,
    topology,
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
    tensor_network,
    terms,
    transport,
    uq,
    weighting,
)

# Applications depend on public equation/solver substrates and load last.
from . import applications


# Explicit re-exports for star import
__all__ = [
    "applications",
    "backends",
    "conditions",
    "control",
    "continuation",
    "coresets",
    "terms",
    "data_utils",
    "discretization",
    "topology",
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
    "tensor_network",
    "weighting",
]
