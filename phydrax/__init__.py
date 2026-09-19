#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# ruff: noqa: I001, RUF022

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
    topology,
    domain,
    dynamics,
    enforcement,
    electrochemistry,
    electrohydrodynamics,
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

# HEP interchange also depends on application products and loads after apps.
from .interchange import hep as _hep_interchange  # noqa: F401


# Explicit re-exports for star import
__all__ = [
    "acoustics",
    "algebraic",
    "atomistic",
    "artifacts",
    "applications",
    "backends",
    "axes",
    "causal",
    "chemistry",
    "combinatorial",
    "closure_data",
    "circuit",
    "conditions",
    "correlation",
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
    "electrochemistry",
    "electrohydrodynamics",
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
    "frequency",
    "lifecycle",
    "kernels",
    "imaging",
    "linalg",
    "logging",
    "metrix",
    "meshing",
    "interfacial_transport",
    "measurement",
    "ml",
    "manufacturing",
    "materials",
    "membranes",
    "signal",
    "nn",
    "nonlinear",
    "nuclear",
    "particle_physics",
    "observation",
    "operators",
    "optics",
    "optomechanics",
    "phoresis",
    "population_balance",
    "process_systems",
    "optim",
    "precision",
    "qualification",
    "rendering",
    "sensing",
    "rom",
    "rheology",
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
    "structural_dynamics",
    "surface_chemistry",
    "system_modeling",
    "thermal_systems",
    "tribology",
    "smart_materials",
    "chemo_mechanics",
    "units",
    "uq",
    "variational",
    "tensor_network",
    "tensor_decomposition",
    "tensor_train",
    "weighting",
    "ArrayLeafSchema",
    "AbstractArrayModel",
    "ArrayPyTreeSchema",
    "array_tree_fingerprint",
    "ExecutableSignature",
    "NumericRevision",
    "SemanticProvenance",
    "callable_payload",
    "strict_module_payload",
    "ArrayArchiveLimits",
    "AdmissibilityHeader",
    "AdmissibilityReason",
    "AdmissibilityTransitionRequest",
    "DerivativeAvailability",
    "combine_admissibility",
    "reason_bits_where",
    "DimensionalScaleContract",
    "ExecutionResourceEvidence",
    "LengthCoordinateKind",
    "RelativityScaleContract",
    "SpatialCoordinateContract",
]
