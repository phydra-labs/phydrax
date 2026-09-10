#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# ruff: noqa: I001

# Ensure JAX uses 64-bit floats by default for numerical robustness.
import jax


jax.config.update("jax_enable_x64", True)

# Launcher-provided rank settings must be consumed before any device-owning module
# is imported. Local imports remain unchanged and preserve the package's deliberate
# dependency order.
from ._execution_bootstrap import RuntimeBootstrap, initialize_from_environment


if RuntimeBootstrap.from_environment() is not None:
    initialize_from_environment()

from . import logging as logging

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
    fidelity,
    execution,
    geometry,
    graph,
    interchange,
    imaging,
    meshing,
    integration,
    lifecycle,
    kernels,
    linalg,
    metrix,
    ml,
    measurement,
    signal,
    nn,
    nonlinear,
    operators,
    optim,
    pgm,
    precision,
    qualification,
    rendering,
    sensing,
    rom,
    service,
    spatial_sampling,
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
    units,
    circuit,
    optics,
    velocimetry,
    weighting,
)
from . import artifacts, events, observation
from . import causal
from ._array_archive import ArrayArchiveLimits
from ._array_tree import ArrayLeafSchema, ArrayPyTreeSchema
from ._identity import (
    callable_payload,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from ._physical import (
    DimensionalScaleContract,
    LengthCoordinateKind,
    SpatialCoordinateContract,
)

from . import atomistic, nuclear

# Closure-data and statistical-dynamics packages depend on loaded numerical substrates.
from . import closure_data, statistical_dynamics

# Finance composes the loaded numerical, stochastic, and qualification substrates.
from . import finance

# Applications depend on public equation/solver substrates and load last.
from . import applications


# Explicit re-exports for star import
__all__ = [
    "atomistic",
    "artifacts",
    "applications",
    "backends",
    "causal",
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
    "finance",
    "fidelity",
    "execution",
    "integration",
    "interchange",
    "transport",
    "geometry",
    "graph",
    "lifecycle",
    "kernels",
    "imaging",
    "linalg",
    "logging",
    "metrix",
    "meshing",
    "measurement",
    "ml",
    "signal",
    "nn",
    "nonlinear",
    "nuclear",
    "observation",
    "operators",
    "optics",
    "optim",
    "precision",
    "qualification",
    "rendering",
    "sensing",
    "rom",
    "service",
    "pgm",
    "sampling",
    "series",
    "spatial_sampling",
    "sparse",
    "velocimetry",
    "special",
    "solver",
    "stochastic",
    "statistical_dynamics",
    "units",
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
    "SpatialCoordinateContract",
]
