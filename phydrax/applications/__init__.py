#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Executable scientific application workflows built on Phydrax substrates."""

from . import (
    astrodynamics,
    astrophysics,
    cardiovascular,
    cellular_mechanics,
    compact_objects,
    compressible_flow,
    contact,
    cosmology,
    crystal_plasticity,
    electrophysiology,
    fracture,
    free_boundary,
    hydrodynamics,
    incompressible_flow,
    ocean,
    phase_field,
    reacting_flow,
    robotics,
    solid_mechanics,
    systems_biology,
    thermofluids,
    two_phase_flow,
    vortex_flow,
)
from .contact import (
    PreparedReynoldsFilm,
    ReynoldsFilmEvidence,
    ReynoldsFilmPlan,
    ReynoldsFilmResult,
    ReynoldsFilmState,
    ReynoldsPressureBoundaryConditions,
)


__all__ = [
    "astrodynamics",
    "astrophysics",
    "cardiovascular",
    "cellular_mechanics",
    "compact_objects",
    "compressible_flow",
    "contact",
    "cosmology",
    "crystal_plasticity",
    "electrophysiology",
    "fracture",
    "free_boundary",
    "hydrodynamics",
    "incompressible_flow",
    "ocean",
    "phase_field",
    "reacting_flow",
    "robotics",
    "solid_mechanics",
    "systems_biology",
    "thermofluids",
    "two_phase_flow",
    "vortex_flow",
    "PreparedReynoldsFilm",
    "ReynoldsFilmEvidence",
    "ReynoldsFilmPlan",
    "ReynoldsFilmResult",
    "ReynoldsFilmState",
    "ReynoldsPressureBoundaryConditions",
]
