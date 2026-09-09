#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-shape geometric, sequential, paraxial, and non-sequential optics."""

from ._graded_index import (
    AbstractRefractiveIndexField,
    AnalyticRefractiveIndexField,
    GradedIndexRayEvidence,
    GradedIndexRayPlan,
    GradedIndexRayResult,
    GradedIndexRayState,
    PreparedGradedIndexRay,
    RayFanPlan,
    RayFanResult,
    StructuredRefractiveIndexField,
    TetrahedralRefractiveIndexField,
)
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
    "AbstractRefractiveIndexField",
    "AnalyticRefractiveIndexField",
    "DifferentialRayMap",
    "NonSequentialBranchMode",
    "NonSequentialOpticsPlan",
    "NonSequentialOpticsResult",
    "NonSequentialOpticsStatus",
    "NonSequentialSurfaceKind",
    "NonSequentialSurfaceTable",
    "GradedIndexRayEvidence",
    "GradedIndexRayPlan",
    "GradedIndexRayResult",
    "GradedIndexRayState",
    "OpticalRayState",
    "ParaxialOpticsPlan",
    "ParaxialOpticsResult",
    "ParaxialOpticsStatus",
    "PlanarRefractiveStack",
    "PreparedNonSequentialOptics",
    "PreparedGradedIndexRay",
    "PreparedParaxialOptics",
    "PreparedSequentialOptics",
    "RefractiveInterfaceResult",
    "RayFanPlan",
    "RayFanResult",
    "RefractiveInterfaceStatus",
    "SequentialOpticsPlan",
    "SequentialOpticsResult",
    "SequentialOpticsStatus",
    "SurfaceInteraction",
    "SurfaceKind",
    "StructuredRefractiveIndexField",
    "TetrahedralRefractiveIndexField",
    "evaluate_refractive_interface",
    "linearize_sequential_optics",
    "prepare_nonsequential_optics",
    "trace_nonsequential_optics",
    "trace_planar_refractive_stack",
]
