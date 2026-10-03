# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Error indicators, deterministic marking and bounded meshfree adaptation.

This module owns the decision layer of h/p/support adaptivity on point clouds:

* indicators: the strong-form PDE residual at oversampled off-node probes, the
  one-sided flux jump across stencil edges, the ``p`` versus ``p + 1``
  operator difference, and stencil amplification/spacing quality;
* stable-ID marking (Dörfler bulk or maximum fraction) with an explicit cap;
* a bounded adaptation proposal: deterministic edge-midpoint children of
  marked points, removal of a maximal independent set of low-indicator
  unmarked points, and cloud-wide degree or support changes;
* the conservative/constant/moment-preserving field transfer from the source
  to the proposed cloud and the explicit acceptance boundary.

Every quantity here is an *indicator*: no reliability or efficiency constant
is claimed, and a local fit residual is not a PDE error bound. A proposal is
only a candidate; it becomes a meshfree epoch through
:class:`~phydrax.discretization.meshfree.MeshfreeEpochChange` (cause
``adaptive-refinement``), whose staged composition rebind rebuilds every
dependent artifact and transports every state entry atomically. Rejection
publishes nothing.

Per-row (regional) polynomial degree or support width needs a separate fitted
basis per degree class in the point-cloud owner; degree and support changes
here are therefore cloud-wide transactions.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import comb
from numbers import Integral, Real
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import LinearSolvePolicy
from ...optim import ConvexSolvePolicy
from ...sparse import EdgeRelation
from ...typing import Bool, Dim, Float64, Int64, parse
from .._point_cloud import PreparedPointCloudDiscretization
from .._point_cloud_pde import PointCollocationStability, PointStabilityOutcome
from ..fem import dorfler_mark, maximum_mark
from ._capacity import bucketed_storage_capacity
from ._multilevel import _priority_independent_set, _SELECTED, _stable_priorities
from ._neighbors import MeshfreeNeighborhoodPlan
from ._operators import MeshfreeOperator
from ._stencils import (
    LocalStencilPolicy,
    LocalStencilReport,
    MeshfreeFunctional,
    prepare_local_stencils,
)
from ._transfer import (
    PointTransferPlan,
    PointTransferRequest,
    PointTransferStatus,
    PreparedPointTransfer,
)


MeshfreeIndicatorKind: TypeAlias = Literal[
    "probe-residual", "flux-jump", "degree-difference", "support-amplification"
]
MeshfreeMarkingStrategy: TypeAlias = Literal["dorfler", "maximum"]
MeshfreeSupportKind: TypeAlias = Literal["bulk", "surface"]
MeshfreeAdaptationChange: TypeAlias = Literal["insertion", "removal", "degree", "support"]
MeshfreeAdaptationRefusal: TypeAlias = Literal[
    "proposal-refused",
    "transfer-refused",
    "stencil-rows-refused",
    "solve-failed",
    "operator-unstable",
    "indicator-not-reduced",
]

# Strong-form residual R(x, u, grad u, hess u) evaluated at probe points.
type MeshfreeStrongResidual = Callable[[Array, MeshfreeProbeJet], Array]

# Host projection of candidate points onto a declared geometry: points, normals.
type MeshfreeProjection = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

# Host predicate admitting candidate points inside a declared domain.
type MeshfreeContainment = Callable[[np.ndarray], np.ndarray]


# Adapted clouds are strongly graded, so host-side k-neighbor queries declare a
# candidate buffer well above the default; the owner still certifies the gap.
_MINIMUM_CANDIDATES = 1024


def _candidate_capacity(sources: int, neighbors: int, /) -> int:
    return min(sources, max(_MINIMUM_CANDIDATES, 16 * (neighbors + 1)))


class IndicatorPointDim(Dim):
    """Cloud points carrying one indicator value."""


class ProbeDim(Dim):
    """Off-node probe points of a residual indicator."""


class ProbeAxisDim(Dim):
    """Ambient coordinates of probe points and their derivatives."""


class MarkedDim(Dim):
    """Stable IDs selected by a marking."""


class AdaptiveSourceDim(Dim):
    """Points of the cloud an adaptation starts from."""


class AdaptiveTargetDim(Dim):
    """Points of the proposed cloud."""


class AdaptiveAxisDim(Dim):
    """Ambient coordinates of an adaptive support."""


class InsertedDim(Dim):
    """Points inserted by an adaptation."""


class RemovedDim(Dim):
    """Points removed by an adaptation."""


class ParentPairDim(Dim):
    """The two parent stable IDs of an inserted edge-midpoint child."""


class MeshfreeAdaptationStatus(IntEnum):
    """Admission of one adaptation proposal.

    ``NO_CHANGE`` means nothing admissible was marked, inserted, removed or
    changed. ``CAPACITY_REFUSED`` means the proposal would exceed a declared
    insertion or point capacity; it is refused whole rather than truncated.
    """

    ADMITTED = 0
    NO_CHANGE = 1
    CAPACITY_REFUSED = 2


def _host_integer(value: int, name: str, /, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a host integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _host_fraction(value: float, name: str, /, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real host number.")
    result = float(value)
    if not (np.isfinite(result) and lower <= result <= upper):
        raise ValueError(f"{name} must lie in [{lower}, {upper}].")
    return result


def _stable_ids(value: ArrayLike, count: int, /) -> np.ndarray:
    ids = np.asarray(value)
    if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("stable_ids must be one integer identifier per point.")
    if np.unique(ids).size != count:
        raise ValueError("stable_ids must be unique.")
    return ids.astype(np.int64)


def _neighbor_lists(
    points: np.ndarray, neighbors: int, ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest non-self neighbors per point, ordered by distance then stable ID."""
    width = min(neighbors + 1, points.shape[0])
    prepared = MeshfreeNeighborhoodPlan(
        points,
        width,
        source_ids=ids,
        maximum_candidates=_candidate_capacity(points.shape[0], width),
    ).prepare()
    indices, valid = jax.device_get(
        (prepared.relation.source_indices, prepared.relation.valid)
    )
    rows = np.arange(points.shape[0])[:, None]
    usable = valid & (indices != rows)
    distance = np.linalg.norm(points[indices] - points[:, None, :], axis=-1)
    distance = np.where(usable, distance, np.inf)
    order = np.lexsort((np.where(usable, ids[indices], np.iinfo(np.int64).max), distance))
    ordered = np.take_along_axis(indices, order, axis=1)
    ordered_usable = np.take_along_axis(usable, order, axis=1)
    return ordered[:, :neighbors], ordered_usable[:, :neighbors]


def meshfree_fill_measures(
    points: ArrayLike,
    total_measure: float,
    /,
    *,
    intrinsic_dimension: int,
    neighbors: int = 6,
) -> np.ndarray:
    """Positive point measures proportional to the local fill volume.

    ``w_i = total * h_i^m / sum_j h_j^m`` with ``h_i`` the mean distance to
    the ``neighbors`` nearest other points and ``m`` the intrinsic dimension.
    This is a declared normalized quadrature estimate: its total equals the
    declared domain measure exactly; its local accuracy is not certified.
    """
    host = np.asarray(points, dtype=np.float64)
    if host.ndim != 2 or not np.all(np.isfinite(host)):
        raise ValueError("points must be a finite (N, D) array.")
    total = float(total_measure)
    if not (np.isfinite(total) and total > 0.0):
        raise ValueError("total_measure must be finite and positive.")
    dimension = _host_integer(intrinsic_dimension, "intrinsic_dimension", minimum=1)
    count = _host_integer(neighbors, "neighbors", minimum=1)
    if host.shape[0] <= count:
        raise ValueError("fill measures need more points than neighbors.")
    ids = np.arange(host.shape[0], dtype=np.int64)
    indices, usable = _neighbor_lists(host, count, ids)
    distance = np.linalg.norm(host[indices] - host[:, None, :], axis=-1)
    spacing = np.sum(np.where(usable, distance, 0.0), axis=1) / np.sum(usable, axis=1)
    volume = spacing**dimension
    return total * volume / np.sum(volume)


# --- Indicators -----------------------------------------------------------------------


@final
class MeshfreeProbeJet(StrictModule):
    """Value, gradient and Hessian of a nodal field at off-node probes."""

    __strict_contract__ = True
    value: Float64[ProbeDim]
    gradient: Float64[ProbeDim, ProbeAxisDim]
    hessian: Float64[ProbeDim, ProbeAxisDim, ProbeAxisDim]


@final
class MeshfreeErrorIndicator(StrictModule):
    """Per-point error indicator keyed by stable IDs.

    An indicator ranks where to adapt. It is not an error estimate: no
    reliability or efficiency constant relates it to the discretization error.
    ``probe_count`` is the number of off-node evaluation points the indicator
    consumed (zero for nodal indicators).
    """

    __strict_contract__ = True
    values: Float64[IndicatorPointDim]
    stable_ids: Int64[IndicatorPointDim]
    kind: MeshfreeIndicatorKind = eqx.field(static=True)
    probe_count: int = eqx.field(static=True)
    indicator_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        stable_ids: ArrayLike,
        /,
        *,
        kind: MeshfreeIndicatorKind,
        probe_count: int,
    ) -> None:
        host = np.asarray(values, dtype=np.float64)
        if host.ndim != 1 or not np.all(np.isfinite(host)) or np.any(host < 0.0):
            raise ValueError("Indicator values must be finite, nonnegative and rank-1.")
        ids = _stable_ids(stable_ids, host.shape[0])
        kind_ = parse(kind, MeshfreeIndicatorKind, "kind")
        probes = _host_integer(probe_count, "probe_count", minimum=0)
        self.values = jnp.asarray(host)
        self.stable_ids = jnp.asarray(ids)
        self.kind = kind_
        self.probe_count = probes
        self.indicator_id = canonical_fingerprint(
            {
                "kind": "meshfree-error-indicator",
                "indicator": kind_,
                "values": array_tree_fingerprint(host),
                "ids": array_tree_fingerprint(ids),
                "probes": probes,
            }
        )

    @property
    def aggregate(self) -> float:
        """``sqrt(sum_i eta_i^2)`` of the per-point values."""
        values = np.asarray(self.values)
        return float(np.sqrt(np.sum(values * values)))


def _jet_functionals(dimension: int, /) -> tuple[MeshfreeFunctional, ...]:
    def unit(*axes: int) -> tuple[int, ...]:
        index = [0] * dimension
        for axis in axes:
            index[axis] += 1
        return tuple(index)

    functionals = [MeshfreeFunctional((unit(),), (1.0,), name="value")]
    functionals += [
        MeshfreeFunctional((unit(axis),), (1.0,), name=f"d{axis}")
        for axis in range(dimension)
    ]
    functionals += [
        MeshfreeFunctional((unit(first, second),), (1.0,), name=f"d{first}{second}")
        for first in range(dimension)
        for second in range(first, dimension)
    ]
    return tuple(functionals)


def _probe_jet(
    points: np.ndarray,
    ids: np.ndarray,
    probes: np.ndarray,
    values: Array,
    policy: LocalStencilPolicy,
    neighbors: int,
    /,
) -> MeshfreeProbeJet:
    """Off-node jet from one fitted source-to-probe stencil family."""
    dimension = points.shape[1]
    neighborhood = MeshfreeNeighborhoodPlan(
        points,
        neighbors,
        targets=probes,
        source_ids=ids,
        maximum_candidates=_candidate_capacity(points.shape[0], neighbors),
    ).prepare()
    functionals = _jet_functionals(dimension)
    stencils = prepare_local_stencils(neighborhood, points, probes, functionals, policy)
    outputs = [
        MeshfreeOperator(stencils, index).apply(values)
        for index in range(len(functionals))
    ]
    gradient = jnp.stack(outputs[1 : 1 + dimension], axis=1)
    hessian = jnp.zeros((probes.shape[0], dimension, dimension), dtype=values.dtype)
    cursor = 1 + dimension
    for first in range(dimension):
        for second in range(first, dimension):
            hessian = hessian.at[:, first, second].set(outputs[cursor])
            hessian = hessian.at[:, second, first].set(outputs[cursor])
            cursor += 1
    return MeshfreeProbeJet(value=outputs[0], gradient=gradient, hessian=hessian)


def _edge_probes(
    points: np.ndarray,
    ids: np.ndarray,
    neighbors: int,
    per_point: int,
    admit: MeshfreeContainment | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints of each point's nearest edges and their owning point index."""
    indices, usable = _neighbor_lists(points, min(per_point, neighbors), ids)
    owners = np.broadcast_to(np.arange(points.shape[0])[:, None], indices.shape)
    owner = owners[usable]
    probes = 0.5 * (points[owner] + points[indices[usable]])
    if admit is not None:
        inside = np.asarray(admit(probes), dtype=np.bool_)
        if inside.shape != (probes.shape[0],):
            raise ValueError("The probe predicate must return one Boolean per probe.")
        owner, probes = owner[inside], probes[inside]
    return probes, owner


def _aggregate(
    owner: np.ndarray, squares: np.ndarray, measures: np.ndarray, /
) -> np.ndarray:
    """``eta_i^2 = (m_i / q_i) sum_{p of i} r_p^2`` with ``q_i`` probes of ``i``."""
    count = measures.shape[0]
    totals = np.bincount(owner, weights=squares, minlength=count)
    probes = np.bincount(owner, minlength=count)
    return np.sqrt(np.where(probes > 0, measures * totals / np.maximum(probes, 1), 0.0))


def probe_residual_indicator(
    cloud: PreparedPointCloudDiscretization,
    values: ArrayLike,
    residual: MeshfreeStrongResidual,
    /,
    *,
    probes_per_point: int = 4,
    admit: MeshfreeContainment | None = None,
) -> MeshfreeErrorIndicator:
    """Strong-form residual at oversampled off-node probes.

    Each point owns the midpoints of its ``probes_per_point`` nearest edges.
    The nodal solution is reconstructed there by one source-to-probe stencil
    family of the cloud's own policy and support width (value, gradient and
    Hessian), ``residual(probes, jet)`` evaluates the declared strong form, and
    ``eta_i^2 = (m_i / q_i) sum_p R_p^2``. Collocation drives the nodal
    residual to zero, so only off-node probes carry information. ``admit``
    removes probes outside a nonconvex domain; refused probe stencils raise.

    This is a local indicator, not an error bound. Elliptic pollution is not
    local: the max-norm error can sit in a graded band beside a refined region
    whose own residual is small (measured: the worst-error point ranked 124th
    of 665). Graded closure (``MeshfreeAdaptationPolicy.grading``), not this
    indicator, is what resolves such bands.
    """
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
    per_point = _host_integer(probes_per_point, "probes_per_point", minimum=1)
    points = np.asarray(cloud.points, dtype=np.float64)
    ids = np.asarray(cloud.stable_ids, dtype=np.int64)
    nodal = jnp.asarray(values, dtype=jnp.float64)
    if nodal.shape != (points.shape[0],):
        raise ValueError("values must be one nodal value per cloud point.")
    probes, owner = _edge_probes(points, ids, cloud.plan.neighbors, per_point, admit)
    jet = _probe_jet(points, ids, probes, nodal, cloud.plan.stencil, cloud.plan.neighbors)
    rows = jnp.asarray(residual(jnp.asarray(probes), jet), dtype=jnp.float64)
    if rows.shape != (probes.shape[0],):
        raise ValueError("The strong residual must return one value per probe.")
    squares = np.asarray(rows) ** 2
    if not np.all(np.isfinite(squares)):
        raise ValueError("The strong residual is not finite at the probes.")
    measures = np.asarray(cloud.quadrature_weights, dtype=np.float64)
    return MeshfreeErrorIndicator(
        _aggregate(owner, squares, measures),
        ids,
        kind="probe-residual",
        probe_count=probes.shape[0],
    )


def _nodal_hessian(cloud: PreparedPointCloudDiscretization, values: Array, /) -> Array:
    dimension = cloud.spatial_dimension
    hessian = jnp.zeros((values.shape[0], dimension, dimension), dtype=values.dtype)
    for first in range(dimension):
        for second in range(first, dimension):
            index = [0] * dimension
            index[first] += 1
            index[second] += 1
            entry = cloud.mixed_partial_derivative(values, multi_index=tuple(index))
            hessian = hessian.at[:, first, second].set(entry)
            hessian = hessian.at[:, second, first].set(entry)
    return hessian


def flux_jump_indicator(
    cloud: PreparedPointCloudDiscretization,
    values: ArrayLike,
    /,
    *,
    diffusivity: ArrayLike = 1.0,
    edges_per_point: int = 4,
) -> MeshfreeErrorIndicator:
    """Jump of the two one-sided diffusive fluxes across each nearest edge.

    Point ``i`` owns its local quadratic reconstruction
    ``u_i(x) = u_i + g_i . (x - x_i) + (x - x_i)^T H_i (x - x_i) / 2`` from the
    cloud's nodal derivative stencils. At the midpoint ``m_ij`` of each of its
    ``edges_per_point`` nearest edges the normal flux jump
    ``J_ij = n_ij . (K_i grad u_i(m_ij) - K_j grad u_j(m_ij))`` is formed with
    each side's own diffusivity (one-sided at material interfaces) and
    ``eta_i^2 = (m_i / q_i) sum_j (J_ij / h_ij)^2``.
    """
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
    per_point = _host_integer(edges_per_point, "edges_per_point", minimum=1)
    points = np.asarray(cloud.points, dtype=np.float64)
    count = points.shape[0]
    ids = np.asarray(cloud.stable_ids, dtype=np.int64)
    nodal = jnp.asarray(values, dtype=jnp.float64)
    if nodal.shape != (count,):
        raise ValueError("values must be one nodal value per cloud point.")
    conductivity = np.broadcast_to(np.asarray(diffusivity, dtype=np.float64), (count,))
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("diffusivity must be finite and positive at every point.")
    gradient = np.asarray(cloud.gradient(nodal), dtype=np.float64).reshape((count, -1))
    hessian = np.asarray(_nodal_hessian(cloud, nodal), dtype=np.float64)
    indices, usable = _neighbor_lists(points, min(per_point, cloud.plan.neighbors), ids)
    owner = np.broadcast_to(np.arange(count)[:, None], indices.shape)[usable]
    other = indices[usable]
    midpoint = 0.5 * (points[owner] + points[other])
    edge = points[other] - points[owner]
    length = np.linalg.norm(edge, axis=1)
    normal = edge / length[:, None]

    def flux(side: np.ndarray, /) -> np.ndarray:
        offset = midpoint - points[side]
        local = gradient[side] + np.einsum("eab,eb->ea", hessian[side], offset)
        return conductivity[side][:, None] * local

    jump = np.sum(normal * (flux(owner) - flux(other)), axis=1)
    measures = np.asarray(cloud.quadrature_weights, dtype=np.float64)
    return MeshfreeErrorIndicator(
        _aggregate(owner, (jump / length) ** 2, measures),
        ids,
        kind="flux-jump",
        probe_count=owner.size,
    )


def degree_difference_indicator(
    lower: Callable[[Array], Array],
    higher: Callable[[Array], Array],
    values: ArrayLike,
    measures: ArrayLike,
    stable_ids: ArrayLike,
    /,
    *,
    rows: ArrayLike | None = None,
) -> MeshfreeErrorIndicator:
    """Difference of the degree-``p`` and degree-``p + 1`` operators on one field.

    ``lower`` and ``higher`` apply the same declared nodal PDE operator fitted
    at two polynomial degrees on the same points (bulk Laplacians, surface
    Laplace-Beltrami operators, ...). ``eta_i = sqrt(m_i) |(L_{p+1} - L_p) u|_i``
    on the selected ``rows`` (Dirichlet rows are exact and excluded by passing
    the interior mask); unselected rows carry zero.
    """
    nodal = jnp.asarray(values, dtype=jnp.float64)
    if nodal.ndim != 1:
        raise ValueError("values must be one nodal value per point.")
    count = nodal.shape[0]
    weights = np.asarray(measures, dtype=np.float64)
    if (
        weights.shape != (count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise ValueError("measures must be finite positive values, one per point.")
    selected = np.ones((count,), dtype=np.bool_) if rows is None else np.asarray(rows)
    if selected.dtype != np.bool_ or selected.shape != (count,):
        raise ValueError("rows must be a Boolean mask with one entry per point.")
    difference = np.asarray(higher(nodal) - lower(nodal), dtype=np.float64)
    if difference.shape != (count,) or not np.all(np.isfinite(difference)):
        raise ValueError("Both operators must return finite nodal values.")
    return MeshfreeErrorIndicator(
        np.where(selected, np.sqrt(weights) * np.abs(difference), 0.0),
        stable_ids,
        kind="degree-difference",
        probe_count=0,
    )


@final
class MeshfreeSupportQuality(StrictModule):
    """Stencil conditioning, amplification and spacing quality per point.

    ``condition`` and ``amplification`` are the cloud's own stencil evidence;
    ``spacing_ratio`` is the farthest over the nearest support distance of
    each point. :attr:`indicator` ranks points by amplification, the stability
    constant of the derivative stencils.
    """

    __strict_contract__ = True
    condition: Float64[IndicatorPointDim]
    amplification: Float64[IndicatorPointDim]
    spacing_ratio: Float64[IndicatorPointDim]
    stable_ids: Int64[IndicatorPointDim]

    @property
    def indicator(self) -> MeshfreeErrorIndicator:
        return MeshfreeErrorIndicator(
            self.amplification,
            self.stable_ids,
            kind="support-amplification",
            probe_count=0,
        )


def support_quality(cloud: PreparedPointCloudDiscretization, /) -> MeshfreeSupportQuality:
    """Read the cloud's stencil evidence and measure its support spacing."""
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
    count = cloud.state_shape[0]
    evidence = cloud.stencils.evidence
    condition = np.asarray(evidence.condition, dtype=np.float64).reshape((count, -1))
    amplification = np.asarray(evidence.amplification, dtype=np.float64).reshape(
        (count, -1)
    )
    points = np.asarray(cloud.points, dtype=np.float64)
    indices, valid = jax.device_get((cloud.relation.source_indices, cloud.relation.valid))
    distance = np.linalg.norm(points[indices] - points[:, None, :], axis=-1)
    usable = valid & (distance > 0.0)
    nearest = np.min(np.where(usable, distance, np.inf), axis=1)
    farthest = np.max(np.where(usable, distance, 0.0), axis=1)
    return MeshfreeSupportQuality(
        condition=jnp.asarray(np.max(condition, axis=1)),
        amplification=jnp.asarray(np.max(amplification, axis=1)),
        spacing_ratio=jnp.asarray(farthest / nearest),
        stable_ids=jnp.asarray(np.asarray(cloud.stable_ids, dtype=np.int64)),
    )


# --- Marking --------------------------------------------------------------------------


@final
class MeshfreeMarkingPolicy(StrictModule, NonTrainableState):
    """Deterministic stable-ID marking with an explicit cap.

    ``dorfler`` selects the fewest points whose squared indicators reach
    ``fraction`` of the total; ``maximum`` selects ``ceil(fraction N)`` largest.
    Ties break by stable ID. More than ``maximum_marked`` selections keep the
    ``maximum_marked`` largest and report the achieved bulk share.
    """

    strategy: MeshfreeMarkingStrategy = eqx.field(static=True)
    fraction: float = eqx.field(static=True)
    maximum_marked: int = eqx.field(static=True)

    def __init__(
        self,
        strategy: MeshfreeMarkingStrategy = "dorfler",
        /,
        *,
        fraction: float = 0.5,
        maximum_marked: int = 4_096,
    ) -> None:
        strategy_ = parse(strategy, MeshfreeMarkingStrategy, "strategy")
        fraction_ = _host_fraction(fraction, "fraction", lower=0.0, upper=1.0)
        if fraction_ == 0.0:
            raise ValueError("fraction must be positive.")
        self.strategy = strategy_
        self.fraction = fraction_
        self.maximum_marked = _host_integer(maximum_marked, "maximum_marked", minimum=1)


@final
class MeshfreeMarking(StrictModule):
    """Stable IDs selected by one marking and the bulk share they capture."""

    __strict_contract__ = True
    marked_ids: Int64[MarkedDim]
    captured_fraction: float = eqx.field(static=True)
    capped: bool = eqx.field(static=True)
    policy: MeshfreeMarkingPolicy
    indicator_id: str = eqx.field(static=True)


def mark_points(
    indicator: MeshfreeErrorIndicator, policy: MeshfreeMarkingPolicy, /
) -> MeshfreeMarking:
    """Select points by the policy's strategy; the result is order independent."""
    if not isinstance(indicator, MeshfreeErrorIndicator):
        raise TypeError("indicator must be a MeshfreeErrorIndicator.")
    if not isinstance(policy, MeshfreeMarkingPolicy):
        raise TypeError("policy must be a MeshfreeMarkingPolicy.")
    values = np.asarray(indicator.values)
    ids = np.asarray(indicator.stable_ids)
    match policy.strategy:
        case "dorfler":
            marked = np.asarray(
                dorfler_mark(values, policy.fraction, cell_global_ids=ids)
            )
        case "maximum":
            marked = np.asarray(
                maximum_mark(values, policy.fraction, cell_global_ids=ids)
            )
        case unknown:
            assert_never(unknown)
    capped = marked.size > policy.maximum_marked
    if capped:
        position = np.searchsorted(ids, marked, sorter=np.argsort(ids))
        selected = np.argsort(ids)[position]
        order = np.lexsort((marked, -values[selected]))
        marked = np.sort(marked[order[: policy.maximum_marked]])
    squares = values * values
    total = float(np.sum(squares))
    captured = float(np.sum(squares[np.isin(ids, marked)]) / total) if total > 0 else 0.0
    return MeshfreeMarking(
        marked_ids=jnp.asarray(marked.astype(np.int64)),
        captured_fraction=captured,
        capped=bool(capped),
        policy=policy,
        indicator_id=indicator.indicator_id,
    )


# --- Adaptation proposal --------------------------------------------------------------


@final
class MeshfreeAdaptiveSupport(StrictModule):
    """Points, identities, boundary rows and fitting parameters of one cloud.

    ``kind="bulk"`` clouds refine boundary edges through a boundary projection;
    ``kind="surface"`` clouds project every child onto the declared manifold.
    ``intrinsic_dimension`` sets the polynomial basis size and fill measure.
    """

    __strict_contract__ = True
    points: Float64[AdaptiveSourceDim, AdaptiveAxisDim]
    stable_ids: Int64[AdaptiveSourceDim]
    boundary_mask: Bool[AdaptiveSourceDim]
    boundary_normals: Float64[AdaptiveSourceDim, AdaptiveAxisDim]
    kind: MeshfreeSupportKind = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    neighbors: int = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        stable_ids: ArrayLike,
        /,
        *,
        kind: MeshfreeSupportKind,
        intrinsic_dimension: int,
        degree: int,
        neighbors: int,
        boundary_mask: ArrayLike | None = None,
        boundary_normals: ArrayLike | None = None,
    ) -> None:
        host = np.asarray(points, dtype=np.float64)
        if host.ndim != 2 or host.shape[0] < 2 or not np.all(np.isfinite(host)):
            raise ValueError("points must be a finite (N, D) array with N >= 2.")
        count, ambient = host.shape
        kind_ = parse(kind, MeshfreeSupportKind, "kind")
        dimension = _host_integer(intrinsic_dimension, "intrinsic_dimension", minimum=1)
        if dimension > ambient or (kind_ == "bulk") != (dimension == ambient):
            raise ValueError(
                "A bulk support has intrinsic dimension equal to its ambient "
                "dimension; a surface support has a lower one."
            )
        boundary = (
            np.zeros((count,), dtype=np.bool_)
            if boundary_mask is None
            else np.asarray(boundary_mask)
        )
        if boundary.dtype != np.bool_ or boundary.shape != (count,):
            raise ValueError("boundary_mask must be Boolean with one entry per point.")
        normals = (
            np.zeros_like(host)
            if boundary_normals is None
            else np.asarray(boundary_normals, dtype=np.float64)
        )
        if normals.shape != host.shape or not np.all(np.isfinite(normals)):
            raise ValueError("boundary_normals must be finite with the points' shape.")
        self.points = jnp.asarray(host)
        self.stable_ids = jnp.asarray(_stable_ids(stable_ids, count))
        self.boundary_mask = jnp.asarray(boundary)
        self.boundary_normals = jnp.asarray(normals)
        self.kind = kind_
        self.intrinsic_dimension = dimension
        self.degree, self.neighbors = _fit_parameters(degree, neighbors, dimension, count)

    @classmethod
    def from_point_cloud(
        cls, cloud: PreparedPointCloudDiscretization, /
    ) -> MeshfreeAdaptiveSupport:
        """The bulk support of a prepared point cloud and its stencil policy."""
        if not isinstance(cloud, PreparedPointCloudDiscretization):
            raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
        return cls(
            cloud.points,
            cloud.stable_ids,
            kind="bulk",
            intrinsic_dimension=cloud.spatial_dimension,
            degree=cloud.plan.stencil.polynomial_degree,
            neighbors=cloud.plan.neighbors,
            boundary_mask=cloud.plan.boundary_mask,
            boundary_normals=cloud.plan.boundary_normals,
        )


def _fit_parameters(
    degree: int, neighbors: int, dimension: int, count: int, /
) -> tuple[int, int]:
    """Strong second derivatives need degree >= 2 and a unisolvent support."""
    degree_ = _host_integer(degree, "degree", minimum=2)
    neighbors_ = _host_integer(neighbors, "neighbors", minimum=1)
    basis = comb(dimension + degree_, degree_)
    if neighbors_ < basis or neighbors_ > count:
        raise ValueError(
            f"Degree {degree_} in {dimension} intrinsic dimensions needs between "
            f"{basis} and {count} neighbors; got {neighbors_}."
        )
    return degree_, neighbors_


@final
class MeshfreeAdaptationPolicy(StrictModule, NonTrainableState):
    """Bounded insertion, removal, grading and measure rules of one adaptation.

    The refined set is the marking, its ``closure_neighbors`` nearest
    neighbors, and then every point whose expected spacing exceeds
    ``grading`` times that of a refined neighbor among the same nearest
    neighbors (repeated until no violation remains). Bisection halves the
    spacing, so ``grading > 2`` admits one level of difference between
    neighbors and refuses two. The default 2.05, just above that minimum,
    widens the graded band around a refined region. On the
    ``eps = 0.04`` boundary layer the max-norm error then sat in the unrefined
    band next to the layer, a point the residual indicator ranked 124th of 665;
    with grading 2.5 three levels stalled (0.0290 -> 0.0300 at 383 -> 665
    points), with 2.05 two levels reached 0.0183 at 520 points. Scale-normalized
    stencil amplification (amplification times squared row scale) stayed at the
    uniform-cloud level, so the band is under-resolved, not destabilized.
    Each refined point bisects its ``children_per_point`` nearest edges. A child
    closer than ``minimum_separation`` times its edge length to an existing
    point or to a higher-priority child is dropped (a deterministic maximal
    independent set). Unrefined interior points whose indicator is at most
    ``coarsen_fraction`` times the largest one are removal candidates; a
    maximal independent set of them (never a parent of a new child) is removed,
    lowest indicators first up to ``maximum_removed``. A proposal with more than
    ``maximum_inserted`` children or ``maximum_points`` points is refused.
    Target measures follow :func:`meshfree_fill_measures` with
    ``total_measure`` and ``measure_neighbors``.
    """

    children_per_point: int = eqx.field(static=True)
    closure_neighbors: int = eqx.field(static=True)
    grading: float = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    coarsen_fraction: float = eqx.field(static=True)
    maximum_inserted: int = eqx.field(static=True)
    maximum_removed: int = eqx.field(static=True)
    maximum_points: int = eqx.field(static=True)
    total_measure: float = eqx.field(static=True)
    measure_neighbors: int = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        total_measure: float,
        children_per_point: int = 4,
        closure_neighbors: int = 8,
        grading: float = 2.05,
        minimum_separation: float = 0.3,
        coarsen_fraction: float = 0.0,
        maximum_inserted: int = 16_384,
        maximum_removed: int = 16_384,
        maximum_points: int = 65_536,
        measure_neighbors: int = 6,
        maximum_rounds: int = 256,
    ) -> None:
        total = float(total_measure)
        if not (np.isfinite(total) and total > 0.0):
            raise ValueError("total_measure must be finite and positive.")
        separation = _host_fraction(
            minimum_separation, "minimum_separation", lower=0.0, upper=0.5
        )
        if separation == 0.0:
            raise ValueError("minimum_separation must be positive.")
        self.children_per_point = _host_integer(
            children_per_point, "children_per_point", minimum=1
        )
        self.closure_neighbors = _host_integer(
            closure_neighbors, "closure_neighbors", minimum=0
        )
        if isinstance(grading, bool) or not isinstance(grading, Real):
            raise TypeError("grading must be a real host number.")
        if not (np.isfinite(grading) and grading > 2.0):
            raise ValueError("grading must exceed 2: bisection halves the spacing.")
        self.grading = float(grading)
        self.minimum_separation = separation
        self.coarsen_fraction = _host_fraction(
            coarsen_fraction, "coarsen_fraction", lower=0.0, upper=1.0
        )
        self.maximum_inserted = _host_integer(
            maximum_inserted, "maximum_inserted", minimum=0
        )
        self.maximum_removed = _host_integer(
            maximum_removed, "maximum_removed", minimum=0
        )
        self.maximum_points = _host_integer(maximum_points, "maximum_points", minimum=2)
        self.total_measure = total
        self.measure_neighbors = _host_integer(
            measure_neighbors, "measure_neighbors", minimum=1
        )
        self.maximum_rounds = _host_integer(maximum_rounds, "maximum_rounds", minimum=1)


@final
class MeshfreeAdaptationProposal(StrictModule):
    """One candidate adapted cloud with its identity, lineage and evidence.

    Retained points keep their stable IDs in source order and are followed by
    the inserted children with fresh IDs above every source ID in canonical
    parent-pair order. ``inserted_parents`` are the parent stable-ID pairs of
    the children; ``removed_ids`` are source IDs absent from the target.
    Dropped children are counted by cause: an unresolved boundary edge without
    a boundary projection, a point outside the admitted domain, or a
    separation conflict. ``refined_points`` counts the marked points with their
    closure and grading extension. Only an ``ADMITTED`` proposal may become an
    epoch.
    """

    __strict_contract__ = True
    source_ids: Int64[AdaptiveSourceDim]
    target_points: Float64[AdaptiveTargetDim, AdaptiveAxisDim]
    target_ids: Int64[AdaptiveTargetDim]
    target_boundary: Bool[AdaptiveTargetDim]
    target_normals: Float64[AdaptiveTargetDim, AdaptiveAxisDim]
    target_measures: Float64[AdaptiveTargetDim]
    inserted_ids: Int64[InsertedDim]
    inserted_parents: Int64[InsertedDim, ParentPairDim]
    removed_ids: Int64[RemovedDim]
    status: MeshfreeAdaptationStatus = eqx.field(static=True)
    changes: tuple[MeshfreeAdaptationChange, ...] = eqx.field(static=True)
    kind: MeshfreeSupportKind = eqx.field(static=True)
    source_degree: int = eqx.field(static=True)
    target_degree: int = eqx.field(static=True)
    source_neighbors: int = eqx.field(static=True)
    target_neighbors: int = eqx.field(static=True)
    unresolved_boundary_children: int = eqx.field(static=True)
    outside_children: int = eqx.field(static=True)
    separation_rejected_children: int = eqx.field(static=True)
    refined_points: int = eqx.field(static=True)
    removal_capped: bool = eqx.field(static=True)
    independent_set_rounds: int = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    @property
    def admitted(self) -> bool:
        return self.status is MeshfreeAdaptationStatus.ADMITTED


def _independent_set(
    conflicts: tuple[np.ndarray, np.ndarray],
    keys: np.ndarray,
    active: np.ndarray,
    maximum_rounds: int,
    /,
) -> tuple[np.ndarray, int]:
    """Stable-key priority maximal independent set of the active nodes."""
    count = keys.size
    storage = bucketed_storage_capacity(max(count, 1))
    first, second = conflicts
    relation = EdgeRelation(
        jnp.asarray(np.concatenate((first, second)).astype(np.int32)),
        jnp.asarray(np.concatenate((second, first)).astype(np.int32)),
        source_size=storage,
        target_size=storage,
    )
    status, rounds = jax.device_get(
        _priority_independent_set(
            relation,
            jnp.asarray(np.pad(_stable_priorities(keys), (0, storage - count))),
            jnp.zeros((storage,), dtype=jnp.bool_),
            jnp.asarray(np.pad(active, (0, storage - count))),
            jnp.asarray(maximum_rounds, dtype=jnp.int32),
        )
    )
    return np.asarray(status[:count] == _SELECTED), int(rounds)


def _graded_refinement(
    points: np.ndarray,
    marked: np.ndarray,
    neighbors: tuple[np.ndarray, np.ndarray],
    policy: MeshfreeAdaptationPolicy,
    /,
) -> np.ndarray:
    """Marking, its nearest-neighbor closure, and the spacing-grading extension.

    A refined point's spacing is expected to halve. A point whose expected
    spacing exceeds ``grading`` times that of a refined neighbor is refined as
    well; every round adds at least one point, so the loop ends within ``N``
    rounds with every graded constraint satisfied.
    """
    indices, usable = neighbors
    width = min(
        max(policy.closure_neighbors, policy.children_per_point), indices.shape[1]
    )
    local, valid = indices[:, :width], usable[:, :width]
    refined = marked.copy()
    closure = min(policy.closure_neighbors, indices.shape[1])
    refined[indices[marked, :closure][usable[marked, :closure]]] = True
    spread = min(4, indices.shape[1])
    distance = np.linalg.norm(points[indices[:, :spread]] - points[:, None, :], axis=-1)
    spacing = np.sum(np.where(usable[:, :spread], distance, 0.0), axis=1) / np.maximum(
        np.sum(usable[:, :spread], axis=1), 1
    )
    for _ in range(points.shape[0]):
        expected = np.where(refined, 0.5 * spacing, spacing)
        finest = np.min(np.where(valid, expected[local], np.inf), axis=1)
        violation = ~refined & (policy.grading * finest < expected)
        if not np.any(violation):
            break
        refined |= violation
    return refined


def _candidate_children(
    support: MeshfreeAdaptiveSupport,
    marked: np.ndarray,
    neighbors: tuple[np.ndarray, np.ndarray],
    policy: MeshfreeAdaptationPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique parent index pairs of marked points' nearest edges, canonical order."""
    ids = np.asarray(support.stable_ids)
    indices, usable = neighbors
    width = min(policy.children_per_point, indices.shape[1])
    rows = np.flatnonzero(marked)
    owner = np.broadcast_to(rows[:, None], (rows.size, width))
    other = indices[rows, :width]
    keep = usable[rows, :width]
    first, second = owner[keep], other[keep]
    low = np.where(ids[first] < ids[second], first, second)
    high = np.where(ids[first] < ids[second], second, first)
    pairs = np.unique(np.stack((ids[low], ids[high]), axis=1), axis=0)
    order = np.argsort(ids)
    parents = order[np.searchsorted(ids, pairs, sorter=order)].reshape((-1, 2))
    return parents, pairs


def _place_children(
    support: MeshfreeAdaptiveSupport,
    parents: np.ndarray,
    boundary_projection: MeshfreeProjection | None,
    manifold_projection: MeshfreeProjection | None,
    admit: MeshfreeContainment | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Midpoints projected onto the declared geometry; drop unresolved/outside."""
    points = np.asarray(support.points)
    boundary = np.asarray(support.boundary_mask)
    midpoint = 0.5 * (points[parents[:, 0]] + points[parents[:, 1]])
    on_boundary = boundary[parents[:, 0]] & boundary[parents[:, 1]]
    normals = np.zeros_like(midpoint)
    keep = np.ones((midpoint.shape[0],), dtype=np.bool_)
    unresolved = 0
    if np.any(on_boundary):
        if boundary_projection is None:
            unresolved = int(np.count_nonzero(on_boundary))
            keep &= ~on_boundary
        else:
            projected, projected_normals = _projected(
                boundary_projection, midpoint[on_boundary]
            )
            midpoint[on_boundary] = projected
            normals[on_boundary] = projected_normals
    interior = keep & ~on_boundary
    if support.kind == "surface" and np.any(interior):
        if manifold_projection is None:
            raise ValueError(
                "A surface support needs a manifold projection for children."
            )
        midpoint[interior] = _projected(manifold_projection, midpoint[interior])[0]
    outside = 0
    if admit is not None and np.any(interior):
        inside = np.asarray(admit(midpoint[interior]), dtype=np.bool_)
        if inside.shape != (int(np.count_nonzero(interior)),):
            raise ValueError("The domain predicate must return one Boolean per point.")
        rejected = np.flatnonzero(interior)[~inside]
        outside = rejected.size
        keep[rejected] = False
    return midpoint, normals, on_boundary, keep, unresolved, outside


def _projected(
    projection: MeshfreeProjection, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    projected, normals = projection(points)
    projected = np.asarray(projected, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if projected.shape != points.shape or normals.shape != points.shape:
        raise ValueError("A projection returns points and normals of the input shape.")
    if not (np.all(np.isfinite(projected)) and np.all(np.isfinite(normals))):
        raise ValueError("A projection returned nonfinite points or normals.")
    return projected, normals


def _separated_children(
    support: MeshfreeAdaptiveSupport,
    parents: np.ndarray,
    pairs: np.ndarray,
    placed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    policy: MeshfreeAdaptationPolicy,
    /,
) -> tuple[np.ndarray, int, int]:
    """Children far enough from every source point and from each other."""
    points = np.asarray(support.points)
    midpoint, _, _, keep = placed
    length = np.linalg.norm(points[parents[:, 0]] - points[parents[:, 1]], axis=1)
    threshold = policy.minimum_separation * length
    candidates = np.flatnonzero(keep)
    if candidates.size == 0:
        return keep, 0, 0
    nearest = MeshfreeNeighborhoodPlan(
        points,
        1,
        targets=midpoint[candidates],
        source_ids=np.asarray(support.stable_ids),
        maximum_candidates=_candidate_capacity(points.shape[0], 1),
    ).prepare()
    distance = np.asarray(nearest.distances)[:, 0]
    clear = distance >= threshold[candidates]
    rejected = int(np.count_nonzero(~clear))
    candidates = candidates[clear]
    # Distinct edges can share a midpoint (the diagonals of a parallelogram);
    # the canonical first parent pair keeps it.
    _, first = np.unique(midpoint[candidates], axis=0, return_index=True)
    rejected += candidates.size - first.size
    candidates = candidates[np.sort(first)]
    accepted = np.zeros_like(keep)
    rounds = 0
    if candidates.size > 1:
        local = midpoint[candidates]
        width = min(8, candidates.size - 1)
        peers, usable = _neighbor_lists(local, width, np.arange(candidates.size))
        gap = np.linalg.norm(local[peers] - local[:, None, :], axis=-1)
        limit = np.maximum(threshold[candidates][:, None], threshold[candidates][peers])
        conflict = usable & (gap < limit)
        rows = np.broadcast_to(np.arange(candidates.size)[:, None], peers.shape)
        span = int(pairs[:, 1].max()) + 1
        keys = pairs[candidates, 0] * span + pairs[candidates, 1]
        selected, rounds = _independent_set(
            (rows[conflict], peers[conflict]),
            keys,
            np.ones((candidates.size,), dtype=np.bool_),
            policy.maximum_rounds,
        )
        rejected += int(np.count_nonzero(~selected))
        candidates = candidates[selected]
    accepted[candidates] = True
    return accepted, rejected, rounds


def _removed_points(
    support: MeshfreeAdaptiveSupport,
    marked: np.ndarray,
    indicator: np.ndarray,
    parents: np.ndarray,
    neighbors: tuple[np.ndarray, np.ndarray],
    policy: MeshfreeAdaptationPolicy,
    /,
) -> tuple[np.ndarray, bool, int]:
    """Maximal independent set of low-indicator unmarked interior points."""
    count = marked.size
    removed = np.zeros((count,), dtype=np.bool_)
    if policy.coarsen_fraction == 0.0 or policy.maximum_removed == 0:
        return removed, False, 0
    candidates = (
        ~marked
        & ~np.asarray(support.boundary_mask)
        & (indicator <= policy.coarsen_fraction * np.max(indicator))
    )
    candidates[parents.reshape(-1)] = False
    if not np.any(candidates):
        return removed, False, 0
    indices, usable = neighbors
    rows = np.broadcast_to(np.arange(count)[:, None], indices.shape)
    removed, rounds = _independent_set(
        (rows[usable], indices[usable]),
        np.asarray(support.stable_ids),
        candidates,
        policy.maximum_rounds,
    )
    chosen = np.flatnonzero(removed)
    capped = chosen.size > policy.maximum_removed
    if capped:
        ids = np.asarray(support.stable_ids)
        order = np.lexsort((ids[chosen], indicator[chosen]))
        removed[:] = False
        removed[chosen[order[: policy.maximum_removed]]] = True
    return removed, capped, rounds


def propose_adaptation(
    support: MeshfreeAdaptiveSupport,
    indicator: MeshfreeErrorIndicator,
    marking: MeshfreeMarking,
    policy: MeshfreeAdaptationPolicy,
    /,
    *,
    target_degree: int | None = None,
    target_neighbors: int | None = None,
    boundary_projection: MeshfreeProjection | None = None,
    manifold_projection: MeshfreeProjection | None = None,
    admit: MeshfreeContainment | None = None,
) -> MeshfreeAdaptationProposal:
    """Build one bounded, deterministic adaptation candidate.

    The proposal depends only on stable IDs, coordinates, the marking and the
    policy, never on input ordering. ``target_degree`` / ``target_neighbors``
    request a cloud-wide degree or support change of the same points (and of
    any children). Boundary-edge children require ``boundary_projection``;
    surface children require ``manifold_projection``; ``admit`` refuses
    children outside a nonconvex domain.
    """
    if not isinstance(support, MeshfreeAdaptiveSupport):
        raise TypeError("support must be a MeshfreeAdaptiveSupport.")
    if not isinstance(indicator, MeshfreeErrorIndicator):
        raise TypeError("indicator must be a MeshfreeErrorIndicator.")
    if not isinstance(marking, MeshfreeMarking):
        raise TypeError("marking must be a MeshfreeMarking.")
    if not isinstance(policy, MeshfreeAdaptationPolicy):
        raise TypeError("policy must be a MeshfreeAdaptationPolicy.")
    ids = np.asarray(support.stable_ids)
    count = ids.size
    order = np.argsort(np.asarray(indicator.stable_ids), kind="stable")
    indicator_ids = np.asarray(indicator.stable_ids)[order]
    if not np.array_equal(indicator_ids, np.sort(ids)):
        raise ValueError("The indicator does not cover exactly the support's stable IDs.")
    values = np.asarray(indicator.values)[order][np.searchsorted(indicator_ids, ids)]
    marked = np.isin(ids, np.asarray(marking.marked_ids))
    if marking.indicator_id != indicator.indicator_id:
        raise ValueError("The marking was made from a different indicator.")
    degree, neighbors = _fit_parameters(
        support.degree if target_degree is None else target_degree,
        support.neighbors if target_neighbors is None else target_neighbors,
        support.intrinsic_dimension,
        count,
    )
    adjacency = _neighbor_lists(np.asarray(support.points), support.neighbors, ids)
    refined = _graded_refinement(np.asarray(support.points), marked, adjacency, policy)
    parents, pairs = _candidate_children(support, refined, adjacency, policy)
    placed = _place_children(
        support, parents, boundary_projection, manifold_projection, admit
    )
    accepted, conflicts, child_rounds = _separated_children(
        support, parents, pairs, placed[:4], policy
    )
    removed, capped, removal_rounds = _removed_points(
        support, refined, values, parents[accepted], adjacency, policy
    )
    midpoint, normals, on_boundary = placed[0], placed[1], placed[2]
    retained = ~removed
    inserted = int(np.count_nonzero(accepted))
    target_count = int(np.count_nonzero(retained)) + inserted
    changes = tuple(
        change
        for change, present in (
            ("insertion", inserted > 0),
            ("removal", bool(np.any(removed))),
            ("degree", degree != support.degree),
            ("support", neighbors != support.neighbors),
        )
        if present
    )
    if inserted > policy.maximum_inserted or target_count > policy.maximum_points:
        status = MeshfreeAdaptationStatus.CAPACITY_REFUSED
    elif not changes:
        status = MeshfreeAdaptationStatus.NO_CHANGE
    else:
        status = MeshfreeAdaptationStatus.ADMITTED
    new_ids = int(ids.max()) + 1 + np.arange(inserted, dtype=np.int64)
    points = np.asarray(support.points)
    target_points = np.concatenate((points[retained], midpoint[accepted]))
    target_ids = np.concatenate((ids[retained], new_ids))
    target_boundary = np.concatenate(
        (np.asarray(support.boundary_mask)[retained], on_boundary[accepted])
    )
    target_normals = np.concatenate(
        (np.asarray(support.boundary_normals)[retained], normals[accepted])
    )
    measures = meshfree_fill_measures(
        target_points,
        policy.total_measure,
        intrinsic_dimension=support.intrinsic_dimension,
        neighbors=min(policy.measure_neighbors, target_count - 1),
    )
    removed_ids = ids[removed]
    parent_ids = pairs[accepted]
    identity = canonical_fingerprint(
        {
            "kind": "meshfree-adaptation-proposal",
            "source": array_tree_fingerprint(ids),
            "points": array_tree_fingerprint(target_points),
            "ids": array_tree_fingerprint(target_ids),
            "parents": array_tree_fingerprint(parent_ids),
            "removed": array_tree_fingerprint(removed_ids),
            "degree": [support.degree, degree],
            "neighbors": [support.neighbors, neighbors],
            "status": int(status),
        }
    )
    return MeshfreeAdaptationProposal(
        source_ids=jnp.asarray(ids),
        target_points=jnp.asarray(target_points),
        target_ids=jnp.asarray(target_ids),
        target_boundary=jnp.asarray(target_boundary),
        target_normals=jnp.asarray(target_normals),
        target_measures=jnp.asarray(measures),
        inserted_ids=jnp.asarray(new_ids),
        inserted_parents=jnp.asarray(parent_ids.reshape((-1, 2))),
        removed_ids=jnp.asarray(removed_ids),
        status=status,
        changes=changes,
        kind=support.kind,
        source_degree=support.degree,
        target_degree=degree,
        source_neighbors=support.neighbors,
        target_neighbors=neighbors,
        unresolved_boundary_children=placed[4],
        outside_children=placed[5],
        separation_rejected_children=conflicts,
        refined_points=int(np.count_nonzero(refined)),
        removal_capped=capped,
        independent_set_rounds=max(child_rounds, removal_rounds),
        proposal_id=identity,
    )


# --- Transfer and acceptance ----------------------------------------------------------


def prepare_adaptation_transfer(
    cloud: PreparedPointCloudDiscretization,
    proposal: MeshfreeAdaptationProposal,
    /,
    *,
    moment_degree: int = 0,
    nonnegative: bool = False,
    linear_policy: LinearSolvePolicy | None = None,
    conic_policy: ConvexSolvePolicy | None = None,
) -> PreparedPointTransfer:
    """Joint conservative, constant and moment-preserving source-to-target transfer.

    The base routes are the cloud's own value stencils from the source points
    to the proposed points; the native transfer owner corrects them to conserve
    ``sum_i m_i u_i`` exactly, reproduce constants and the requested moments,
    and audits every equation. Source and target measures must have equal
    totals (both follow the declared total measure) or the owner reports
    ``MEASURE_OBSTRUCTION``. Moments of degree ``q`` together with
    conservation additionally require ``m_new^T x^a = m_old^T x^a`` for every
    monomial of degree ``<= q`` (apply ``T x^a = x^a`` and conserve): two fill
    quadratures generally differ already in their first moments, and the owner
    then certifies ``INFEASIBLE`` with a left-null witness. Request moments only
    for measures that integrate those monomials identically. Refusals are
    returned, never repaired. ``linear_policy`` / ``conic_policy`` select the
    owner's native correction solves (status-mode failure only); a step limit
    too small for the correction reports ``PROVIDER_UNRESOLVED``.
    """
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
    if not isinstance(proposal, MeshfreeAdaptationProposal):
        raise TypeError("proposal must be a MeshfreeAdaptationProposal.")
    if not np.array_equal(np.asarray(cloud.stable_ids), np.asarray(proposal.source_ids)):
        raise ValueError("The proposal was not made from this cloud's stable IDs.")
    points = np.asarray(cloud.points, dtype=np.float64)
    targets = np.asarray(proposal.target_points, dtype=np.float64)
    neighborhood = MeshfreeNeighborhoodPlan(
        points,
        cloud.plan.neighbors,
        maximum_candidates=_candidate_capacity(points.shape[0], cloud.plan.neighbors),
        targets=targets,
        source_ids=np.asarray(cloud.stable_ids),
    ).prepare()
    value = MeshfreeFunctional(((0,) * points.shape[1],), (1.0,), name="value")
    stencils = prepare_local_stencils(
        neighborhood, points, targets, (value,), cloud.plan.stencil
    )
    return PointTransferPlan.from_stencils(
        stencils,
        cloud.quadrature_weights,
        proposal.target_measures,
        request=PointTransferRequest(
            "joint", moment_degree=moment_degree, nonnegative=nonnegative
        ),
        linear_policy=linear_policy,
        conic_policy=conic_policy,
    ).prepare()


@final
class MeshfreeAdaptationAcceptance(StrictModule, NonTrainableState):
    """The explicit boundary decision of one adaptation transaction.

    ``accepted`` is the host Boolean passed to the epoch commit. Every failed
    criterion is listed in ``refusals``. Spectral admission of the target
    operator is required only when its owner's stability evidence is supplied,
    and indicator reduction only when both indicators are supplied.
    """

    accepted: bool = eqx.field(static=True)
    refusals: tuple[MeshfreeAdaptationRefusal, ...] = eqx.field(static=True)
    proposal_status: MeshfreeAdaptationStatus = eqx.field(static=True)
    transfer_status: PointTransferStatus = eqx.field(static=True)
    refused_rows: int = eqx.field(static=True)
    solve_successful: bool = eqx.field(static=True)
    stability_outcome: PointStabilityOutcome | None = eqx.field(static=True)
    indicator_before: float | None = eqx.field(static=True)
    indicator_after: float | None = eqx.field(static=True)


def adaptation_acceptance(
    proposal: MeshfreeAdaptationProposal,
    transfer: PreparedPointTransfer,
    report: LocalStencilReport,
    /,
    *,
    solve_successful: bool,
    stability: PointCollocationStability | None = None,
    indicator_before: MeshfreeErrorIndicator | None = None,
    indicator_after: MeshfreeErrorIndicator | None = None,
) -> MeshfreeAdaptationAcceptance:
    """Accept only an admitted proposal, admitted transfer, fully admitted
    target stencils, a successful target solve, (when supplied) a spectrally
    admitted target operator, and (when supplied) a reduced aggregate
    indicator. A successful solve of a spectrally unstable collocation operator
    can be badly wrong, so supply the owner's stability evidence whenever the
    target is a square collocation."""
    if not isinstance(proposal, MeshfreeAdaptationProposal):
        raise TypeError("proposal must be a MeshfreeAdaptationProposal.")
    if not isinstance(transfer, PreparedPointTransfer):
        raise TypeError("transfer must be a PreparedPointTransfer.")
    if not isinstance(report, LocalStencilReport):
        raise TypeError("report must be the target cloud's LocalStencilReport.")
    if not isinstance(solve_successful, bool):
        raise TypeError("solve_successful must be an explicit host bool.")
    if stability is not None and not isinstance(stability, PointCollocationStability):
        raise TypeError(
            "stability must be the target operator's PointCollocationStability."
        )
    if (indicator_before is None) != (indicator_after is None):
        raise ValueError("Indicator reduction needs both indicators or neither.")
    before = None if indicator_before is None else indicator_before.aggregate
    after = None if indicator_after is None else indicator_after.aggregate
    criteria: tuple[tuple[MeshfreeAdaptationRefusal, bool], ...] = (
        ("proposal-refused", not proposal.admitted),
        ("transfer-refused", not transfer.admitted),
        ("stencil-rows-refused", report.refused_rows > 0),
        ("solve-failed", not solve_successful),
        ("operator-unstable", stability is not None and not stability.admitted),
        (
            "indicator-not-reduced",
            before is not None and after is not None and not after < before,
        ),
    )
    refusals = tuple(name for name, failed in criteria if failed)
    return MeshfreeAdaptationAcceptance(
        accepted=not refusals,
        refusals=refusals,
        proposal_status=proposal.status,
        transfer_status=transfer.evidence.status,
        refused_rows=int(report.refused_rows),
        solve_successful=solve_successful,
        stability_outcome=None if stability is None else stability.outcome,
        indicator_before=before,
        indicator_after=after,
    )


__all__ = [
    "MeshfreeAdaptationAcceptance",
    "MeshfreeAdaptationChange",
    "MeshfreeAdaptationPolicy",
    "MeshfreeAdaptationProposal",
    "MeshfreeAdaptationRefusal",
    "MeshfreeAdaptationStatus",
    "MeshfreeAdaptiveSupport",
    "MeshfreeContainment",
    "MeshfreeErrorIndicator",
    "MeshfreeIndicatorKind",
    "MeshfreeMarking",
    "MeshfreeMarkingPolicy",
    "MeshfreeMarkingStrategy",
    "MeshfreeProbeJet",
    "MeshfreeProjection",
    "MeshfreeStrongResidual",
    "MeshfreeSupportKind",
    "MeshfreeSupportQuality",
    "adaptation_acceptance",
    "degree_difference_indicator",
    "flux_jump_indicator",
    "mark_points",
    "meshfree_fill_measures",
    "prepare_adaptation_transfer",
    "probe_residual_indicator",
    "propose_adaptation",
    "support_quality",
]
