#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native deterministic reconstruction primitives."""

from ._barycentric import (
    barycentric_basis,
    barycentric_differentiation_matrix,
    barycentric_interpolate,
)
from ._bspline import (
    bspline_batched_evaluate,
    bspline_evaluate,
    bspline_jet_stencil,
    bspline_stencil,
    BSplineJetStencil,
)
from ._bspline_grid import BSplineGrid, TrainableBSplineGrid
from ._bspline_grid_bank import BSplineGridBank
from ._bspline_projection import (
    bspline_cross_gram,
    bspline_mass_matrix,
    bspline_projection_matrix,
    BSplineGridTransfer,
    project_bspline_coefficients,
    ProjectionMethod,
)
from ._fourier import (
    FOURIER_CAPABILITIES,
    fourier_interpolate,
    fourier_resample,
    FourierEvaluationMethod,
)
from ._inverse_distance import (
    INVERSE_DISTANCE_CAPABILITIES,
    inverse_distance_stencil,
    SnapPolicy,
)
from ._piecewise import (
    CUBIC_HERMITE_CAPABILITIES,
    cubic_hermite_interpolate,
    cubic_hermite_segment,
    LINEAR_CAPABILITIES,
    linear_interpolate,
    linear_segment,
    linear_stencil,
    linear_stencil_from_indices,
    local_cubic_slope,
    local_cubic_slopes,
    NEAREST_CAPABILITIES,
    nearest_interpolate,
    nearest_stencil,
    nearest_stencil_from_indices,
)
from ._rational_spline import RationalSplineJet
from ._rectilinear import (
    AxisBound,
    RECTILINEAR_CAPABILITIES,
    rectilinear_stencil,
    RectilinearBoundaryMode,
)
from ._stencil import apply_gather_stencil, gather_patches, GatherStencil
from ._tensor_bspline import TensorBSplineJetPlan
from ._types import (
    BoundsMode,
    InterpolationCapabilities,
    InterpolationResult,
    MaskMode,
    NearestTiePolicy,
)


__all__ = [
    "CUBIC_HERMITE_CAPABILITIES",
    "LINEAR_CAPABILITIES",
    "NEAREST_CAPABILITIES",
    "INVERSE_DISTANCE_CAPABILITIES",
    "FOURIER_CAPABILITIES",
    "FourierEvaluationMethod",
    "RECTILINEAR_CAPABILITIES",
    "BSplineGrid",
    "BSplineJetStencil",
    "BSplineGridBank",
    "BSplineGridTransfer",
    "TrainableBSplineGrid",
    "RationalSplineJet",
    "TensorBSplineJetPlan",
    "ProjectionMethod",
    "BoundsMode",
    "AxisBound",
    "GatherStencil",
    "InterpolationCapabilities",
    "InterpolationResult",
    "MaskMode",
    "NearestTiePolicy",
    "SnapPolicy",
    "RectilinearBoundaryMode",
    "apply_gather_stencil",
    "barycentric_basis",
    "barycentric_differentiation_matrix",
    "bspline_evaluate",
    "bspline_batched_evaluate",
    "bspline_cross_gram",
    "bspline_mass_matrix",
    "bspline_projection_matrix",
    "bspline_jet_stencil",
    "bspline_stencil",
    "cubic_hermite_interpolate",
    "cubic_hermite_segment",
    "fourier_interpolate",
    "fourier_resample",
    "linear_interpolate",
    "linear_segment",
    "linear_stencil",
    "inverse_distance_stencil",
    "linear_stencil_from_indices",
    "local_cubic_slope",
    "local_cubic_slopes",
    "nearest_interpolate",
    "project_bspline_coefficients",
    "nearest_stencil",
    "nearest_stencil_from_indices",
    "rectilinear_stencil",
    "barycentric_interpolate",
    "gather_patches",
]
