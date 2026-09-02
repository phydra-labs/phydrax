#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Executable scientific application workflows built on Phydrax substrates."""

from . import (
    astrodynamics,
    astrophysics,
    compact_objects,
    contact,
    cosmology,
    crystal_plasticity,
    fracture,
    free_boundary,
    hydrodynamics,
    incompressible_flow,
    ocean,
    phase_field,
    solid_mechanics,
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
    "compact_objects",
    "contact",
    "cosmology",
    "crystal_plasticity",
    "fracture",
    "free_boundary",
    "hydrodynamics",
    "incompressible_flow",
    "ocean",
    "phase_field",
    "solid_mechanics",
    "two_phase_flow",
    "vortex_flow",
    "PreparedReynoldsFilm",
    "ReynoldsFilmEvidence",
    "ReynoldsFilmPlan",
    "ReynoldsFilmResult",
    "ReynoldsFilmState",
    "ReynoldsPressureBoundaryConditions",
]
