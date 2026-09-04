#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-shape geometric, sequential, paraxial, and non-sequential optics."""

from ._interface import (
    evaluate_refractive_interface,
    OpticalRayState,
    RefractiveInterfaceResult,
    RefractiveInterfaceStatus,
)
from ._nonsequential import (
    NonSequentialBranchMode,
    NonSequentialOpticsPlan,
    NonSequentialOpticsResult,
    NonSequentialOpticsStatus,
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
    prepare_nonsequential_optics,
    PreparedNonSequentialOptics,
    trace_nonsequential_optics,
)
from ._paraxial import (
    DifferentialRayMap,
    linearize_sequential_optics,
    ParaxialOpticsPlan,
    ParaxialOpticsResult,
    ParaxialOpticsStatus,
    PreparedParaxialOptics,
)
from ._planar import PlanarRefractiveStack, trace_planar_refractive_stack
from ._sequential import (
    PreparedSequentialOptics,
    SequentialOpticsPlan,
    SequentialOpticsResult,
    SequentialOpticsStatus,
    SurfaceInteraction,
    SurfaceKind,
)


__all__ = [
    "DifferentialRayMap",
    "NonSequentialBranchMode",
    "NonSequentialOpticsPlan",
    "NonSequentialOpticsResult",
    "NonSequentialOpticsStatus",
    "NonSequentialSurfaceKind",
    "NonSequentialSurfaceTable",
    "OpticalRayState",
    "ParaxialOpticsPlan",
    "ParaxialOpticsResult",
    "ParaxialOpticsStatus",
    "PlanarRefractiveStack",
    "PreparedNonSequentialOptics",
    "PreparedParaxialOptics",
    "PreparedSequentialOptics",
    "RefractiveInterfaceResult",
    "RefractiveInterfaceStatus",
    "SequentialOpticsPlan",
    "SequentialOpticsResult",
    "SequentialOpticsStatus",
    "SurfaceInteraction",
    "SurfaceKind",
    "evaluate_refractive_interface",
    "linearize_sequential_optics",
    "prepare_nonsequential_optics",
    "trace_nonsequential_optics",
    "trace_planar_refractive_stack",
]
