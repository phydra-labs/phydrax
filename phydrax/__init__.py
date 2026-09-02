#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# ruff: noqa: I001

# Ensure JAX uses 64-bit floats by default for numerical robustness
import jax


jax.config.update("jax_enable_x64", True)

from . import ein as ein

from . import (
    backends,
    combinatorial,
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
    variational,
    equations,
    export,
    geometry,
    graph,
    interchange,
    integration,
    lifecycle,
    kernels,
    linalg,
    metrix,
    ml,
    signal,
    nn,
    nonlinear,
    operators,
    optim,
    pgm,
    precision,
    qualification,
    rom,
    service,
    sampling,
    solver,
    sparse,
    special,
    stochastic,
    tensor_network,
    tensor_train,
    terms,
    transport,
    uq,
    circuit,
    velocimetry,
    weighting,
)
from . import artifacts, events, observation
from ._physical import DimensionalScaleContract, LengthCoordinateKind

from . import atomistic

# Applications depend on public equation/solver substrates and load last.
from . import applications


# Explicit re-exports for star import
__all__ = [
    "atomistic",
    "artifacts",
    "applications",
    "backends",
    "combinatorial",
    "circuit",
    "conditions",
    "control",
    "continuation",
    "coresets",
    "terms",
    "data_utils",
    "discretization",
    "events",
    "topology",
    "domain",
    "dynamics",
    "ein",
    "equations",
    "enforcement",
    "export",
    "integration",
    "interchange",
    "transport",
    "geometry",
    "graph",
    "lifecycle",
    "kernels",
    "linalg",
    "metrix",
    "ml",
    "signal",
    "nn",
    "nonlinear",
    "observation",
    "operators",
    "optim",
    "precision",
    "qualification",
    "rom",
    "service",
    "pgm",
    "sampling",
    "sparse",
    "velocimetry",
    "special",
    "solver",
    "stochastic",
    "uq",
    "variational",
    "tensor_network",
    "tensor_train",
    "weighting",
    "DimensionalScaleContract",
    "LengthCoordinateKind",
]
