#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._costs import (
    AbstractGroundCost,
    PeriodicSquaredEuclideanCost,
    PrecomputedCost,
    SquaredEuclideanCost,
    WeightedSquaredEuclideanCost,
)
from ._divergence import sinkhorn_divergence, SinkhornDivergenceResult
from ._fast_order import fast_soft_rank, fast_soft_sort
from ._problem import (
    discrete_problem,
    DiscreteTransportProblem,
    TransportProblemProvenance,
)
from ._references import (
    prepare_sinkhorn_reference,
    PreparedSinkhornReference,
    sinkhorn_divergence_against,
)
from ._results import (
    require_converged,
    SinkhornDiagnostics,
    SinkhornResult,
    TransportProvenance,
)
from ._sinkhorn import Sinkhorn
from ._sliced import sliced_wasserstein_distance, SlicedWassersteinResult
from ._soft import (
    soft_order_transport,
    soft_quantile,
    soft_quantile_normalize,
    soft_quantize,
    soft_rank,
    soft_sort,
    soft_sort_by,
    soft_topk_mask,
    soft_topk_values,
)
from ._status import status_message, TransportStatus
from ._univariate import wasserstein_distance_1d


__all__ = [
    "AbstractGroundCost",
    "DiscreteTransportProblem",
    "PeriodicSquaredEuclideanCost",
    "PrecomputedCost",
    "PreparedSinkhornReference",
    "Sinkhorn",
    "SinkhornDiagnostics",
    "SinkhornDivergenceResult",
    "SlicedWassersteinResult",
    "SinkhornResult",
    "SquaredEuclideanCost",
    "TransportProblemProvenance",
    "TransportProvenance",
    "TransportStatus",
    "WeightedSquaredEuclideanCost",
    "discrete_problem",
    "fast_soft_rank",
    "fast_soft_sort",
    "sliced_wasserstein_distance",
    "soft_order_transport",
    "soft_quantile",
    "soft_quantile_normalize",
    "soft_quantize",
    "soft_rank",
    "soft_sort",
    "soft_sort_by",
    "soft_topk_mask",
    "soft_topk_values",
    "prepare_sinkhorn_reference",
    "require_converged",
    "sinkhorn_divergence",
    "sinkhorn_divergence_against",
    "status_message",
    "wasserstein_distance_1d",
]
