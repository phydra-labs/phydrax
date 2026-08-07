#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable coordinate and Riemannian geometry for Phydrax."""

from ._chart import ChartTransition, CoordinateChart
from ._connection import (
    christoffel_from_metric_jet,
    geodesic_acceleration,
    geodesic_rhs,
    LeviCivitaConnection,
    parallel_transport_rhs,
)
from ._curvature import (
    einstein_tensor,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
    sectional_curvature,
)
from ._embedded import EmbeddedChart, tangent_projector_from_normal
from ._jet import metric_jet, MetricJet
from ._metric import (
    cholesky_metric,
    diagonal_metric,
    euclidean_metric,
    pullback_metric,
    RiemannianMetric,
)
from ._operators import (
    covariant_derivative,
    covariant_hessian,
    divergence,
    gradient,
    laplace_beltrami,
)
from ._stochastic import (
    brownian_generator,
    coordinate_stratonovich_to_ito_drift,
    coordinate_to_covariant_drift,
    covariant_fokker_planck_operator,
    covariant_kolmogorov_generator,
)
from ._state_geometry import (
    AbstractStateGeometry,
    EmbeddedStateGeometry,
    EuclideanStateGeometry,
    LocalRetraction,
    PointwiseStateGeometry,
    SpecialOrthogonalStateGeometry,
    SymmetricPositiveDefiniteStateGeometry,
)
from ._tensor import (
    contract_indices,
    COVECTOR_TENSOR,
    DENSITY_TENSOR,
    inner_product,
    lower_index,
    pullback_covector,
    pushforward_vector,
    raise_index,
    reexpress_tensor,
    SCALAR_TENSOR,
    tensor_norm_squared,
    TensorType,
    TensorVariance,
    VECTOR_TENSOR,
)
from ._validation import MetricValidationReport, validate_metric


__all__ = [
    "AbstractStateGeometry",
    "COVECTOR_TENSOR",
    "ChartTransition",
    "CoordinateChart",
    "EmbeddedChart",
    "EmbeddedStateGeometry",
    "EuclideanStateGeometry",
    "LeviCivitaConnection",
    "MetricJet",
    "MetricValidationReport",
    "RiemannianMetric",
    "LocalRetraction",
    "PointwiseStateGeometry",
    "SpecialOrthogonalStateGeometry",
    "SymmetricPositiveDefiniteStateGeometry",
    "DENSITY_TENSOR",
    "SCALAR_TENSOR",
    "TensorType",
    "TensorVariance",
    "VECTOR_TENSOR",
    "brownian_generator",
    "christoffel_from_metric_jet",
    "cholesky_metric",
    "contract_indices",
    "diagonal_metric",
    "covariant_derivative",
    "covariant_hessian",
    "coordinate_stratonovich_to_ito_drift",
    "coordinate_to_covariant_drift",
    "covariant_fokker_planck_operator",
    "covariant_kolmogorov_generator",
    "einstein_tensor",
    "euclidean_metric",
    "metric_jet",
    "inner_product",
    "lower_index",
    "divergence",
    "pullback_metric",
    "pullback_covector",
    "geodesic_acceleration",
    "pushforward_vector",
    "raise_index",
    "reexpress_tensor",
    "ricci_tensor",
    "riemann_tensor",
    "geodesic_rhs",
    "gradient",
    "laplace_beltrami",
    "parallel_transport_rhs",
    "tensor_norm_squared",
    "tangent_projector_from_normal",
    "scalar_curvature",
    "sectional_curvature",
    "validate_metric",
]
