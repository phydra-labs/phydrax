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
    acoustics,
    algebraic,
    axes,
    backends,
    combinatorial,
    conditions,
    continuation,
    control,
    coresets,
    data_utils,
    correlation,
    series,
    discretization,
    diagnostics,
    topology,
    domain,
    dynamics,
    durability,
    enforcement,
    electrochemistry,
    electrohydrodynamics,
    electromagnetics,
    variational,
    equations,
    export,
    fidelity,
    execution,
    geometry,
    frequency,
    graph,
    interchange,
    imaging,
    meshing,
    integration,
    lifecycle,
    interfacial_transport,
    kernels,
    linalg,
    metrix,
    ml,
    measurement,
    signal,
    nn,
    manufacturing,
    materials,
    membranes,
    nonlinear,
    operators,
    optim,
    pgm,
    precision,
    privacy,
    particle_physics,
    qualification,
    rendering,
    optomechanics,
    phoresis,
    population_balance,
    process_systems,
    sensing,
    rom,
    service,
    rheology,
    spatial_sampling,
    sampling,
    solver,
    sparse,
    special,
    stochastic,
    tensor_network,
    tensor_decomposition,
    tensor_train,
    structural_dynamics,
    surface_chemistry,
    system_modeling,
    thermal_systems,
    tribology,
    smart_materials,
    chemo_mechanics,
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
from ._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    AdmissibilityTransitionRequest,
    DerivativeAvailability,
    combine_admissibility,
    reason_bits_where,
)
from ._array_tree import ArrayLeafSchema, ArrayPyTreeSchema
from ._fingerprint import array_tree_fingerprint
from ._execution_resources import ExecutionResourceEvidence
from ._identity import (
    callable_payload,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from ._model import AbstractArrayModel
from ._strict import StrictModule
from ._physical import (
    DimensionalScaleContract,
    LengthCoordinateKind,
    RelativityScaleContract,
    SpatialCoordinateContract,
)

from . import atomistic, chemistry, nuclear

# Closure-data and statistical-dynamics packages depend on loaded numerical substrates.
from . import closure_data, statistical_dynamics

# Finance composes the loaded numerical, stochastic, and qualification substrates.
from . import finance

# Applications depend on public equation/solver substrates and load last.
from . import applications

# Cosmology interchange depends on application products and therefore loads after apps.
from .interchange import cosmology as _cosmology_interchange  # noqa: F401


# Explicit re-exports for star import
__all__ = [

    "AbstractArrayModel",
    "AdmissibilityHeader",
    "AdmissibilityReason",
    "AdmissibilityTransitionRequest",
    "ArrayArchiveLimits",
    "ArrayLeafSchema",
    "ArrayPyTreeSchema",
    "DerivativeAvailability",
    "DimensionalScaleContract",
    "ExecutableSignature",
    "ExecutionResourceEvidence",
    "LengthCoordinateKind",
    "NumericRevision",
    "RelativityScaleContract",
    "SemanticProvenance",
    "SpatialCoordinateContract",
    "StrictModule",
    "acoustics",
    "algebraic",
    "applications",
    "array_tree_fingerprint",
    "artifacts",
    "atomistic",
    "axes",
    "backends",
    "callable_payload",
    "causal",
    "chemistry",
    "chemo_mechanics",
    "circuit",
    "closure_data",
    "combinatorial",
    "combine_admissibility",
    "conditions",
    "continuation",
    "control",
    "coresets",
    "correlation",
    "data_utils",
    "diagnostics",
    "discretization",
    "domain",
    "durability",
    "dynamics",
    "ein",
    "electrochemistry",
    "electrohydrodynamics",
    "electromagnetics",
    "enforcement",
    "equations",
    "events",
    "execution",
    "export",
    "fidelity",
    "finance",
    "frequency",
    "geometry",
    "graph",
    "imaging",
    "integration",
    "interchange",
    "interfacial_transport",
    "kernels",
    "lifecycle",
    "linalg",
    "logging",
    "manufacturing",
    "materials",
    "measurement",
    "membranes",
    "meshing",
    "metrix",
    "ml",
    "nn",
    "nonlinear",
    "nuclear",
    "observation",
    "operators",
    "optics",
    "optim",
    "optomechanics",
    "particle_physics",
    "pgm",
    "phoresis",
    "population_balance",
    "precision",
    "privacy",
    "process_systems",
    "qualification",
    "reason_bits_where",
    "rendering",
    "rheology",
    "rom",
    "sampling",
    "sensing",
    "series",
    "service",
    "signal",
    "smart_materials",
    "solver",
    "sparse",
    "spatial_sampling",
    "special",
    "statistical_dynamics",
    "stochastic",
    "strict_module_payload",
    "structural_dynamics",
    "surface_chemistry",
    "system_modeling",
    "tensor_decomposition",
    "tensor_network",
    "tensor_train",
    "terms",
    "thermal_systems",
    "topology",
    "transport",
    "tribology",
    "units",
    "uq",
    "variational",
    "velocimetry",
    "weighting",
]
