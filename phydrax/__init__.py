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
    series,
    discretization,
    topology,
    domain,
    dynamics,
    enforcement,
    variational,
    equations,
    export,
    execution,
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
    optics,
    velocimetry,
    weighting,
)
from . import artifacts, events, observation
from ._array_archive import ArrayArchiveLimits
from ._array_tree import ArrayLeafSchema, ArrayPyTreeSchema
from ._identity import (
    callable_payload,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from ._physical import DimensionalScaleContract, LengthCoordinateKind

from . import atomistic

# Closure-data and statistical-dynamics packages depend on loaded numerical substrates.
from . import closure_data, statistical_dynamics

# Applications depend on public equation/solver substrates and load last.
from . import applications


# Explicit re-exports for star import
__all__ = [
    "atomistic",
    "artifacts",
    "applications",
    "backends",
    "combinatorial",
    "closure_data",
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
    "execution",
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
    "optics",
    "optim",
    "precision",
    "qualification",
    "rom",
    "service",
    "pgm",
    "sampling",
    "series",
    "sparse",
    "velocimetry",
    "special",
    "solver",
    "stochastic",
    "statistical_dynamics",
    "uq",
    "variational",
    "tensor_network",
    "tensor_train",
    "weighting",
    "ArrayLeafSchema",
    "ArrayPyTreeSchema",
    "ExecutableSignature",
    "NumericRevision",
    "SemanticProvenance",
    "callable_payload",
    "strict_module_payload",
    "ArrayArchiveLimits",
    "DimensionalScaleContract",
    "LengthCoordinateKind",
]
