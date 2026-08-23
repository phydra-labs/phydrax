#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._homogeneous import (
    fermat_polynomial,
    HomogeneousPolynomial,
    HomogeneousPolynomialReport,
)
from ._hypersurface import fermat_hypersurface, ProjectiveHypersurface
from ._hypersurface_patch import (
    HypersurfacePatchEvaluation,
    HypersurfacePatchGeometry,
    ResidueCanonicalSection,
)
from ._kahler_metric import HypersurfaceKahlerEvaluation, HypersurfaceKahlerGeometry
from ._line_sampling import (
    intersect_projective_line,
    ProjectiveLineSamples,
    sample_projective_hypersurface,
)
from ._projective import ComplexProjectiveAtlas
from ._references import FlatComplexTorus


__all__ = [
    "ComplexProjectiveAtlas",
    "FlatComplexTorus",
    "ProjectiveHypersurface",
    "fermat_hypersurface",
    "HomogeneousPolynomial",
    "HomogeneousPolynomialReport",
    "HypersurfaceKahlerEvaluation",
    "HypersurfaceKahlerGeometry",
    "HypersurfacePatchEvaluation",
    "HypersurfacePatchGeometry",
    "ProjectiveLineSamples",
    "ResidueCanonicalSection",
    "fermat_polynomial",
    "intersect_projective_line",
    "sample_projective_hypersurface",
]
