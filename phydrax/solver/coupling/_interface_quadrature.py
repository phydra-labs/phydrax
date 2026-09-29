#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact common refinement of two straight-facet side traces of a 2-D curve.

Two owners discretize one physical curve interface with independent facet
partitions. The common refinement intersects every facet of the first side
with every collinear, overlapping facet of the second side and integrates on
the resulting segments with a Gauss--Legendre rule, so every product of the two
sides' polynomial facet traces and a multiplier basis of declared degree is
integrated exactly. Coverage, gaps, overlaps, and the opposition of the two
outward normals are measured on the host and refused when they fail.

Each side is re-evaluated at the common points from its own prepared trace:
the trace is prepared on Gauss--Lobatto--Legendre facet sites (which include
the facet end points), and the published polynomial trace degree lets a
facet-local Lagrange interpolation evaluate it exactly at any facet
parameter. The resampling is a per-point gather and contraction with an exact
scatter-add transpose; no coefficient-by-point matrix is formed.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...discretization import FacetTraceRule, PreparedTraceAction


def _lagrange_weights(nodes: np.ndarray, parameters: np.ndarray, /) -> np.ndarray:
    """Values of the Lagrange basis on `nodes` at `parameters`, shape (P, Q)."""
    differences = parameters[:, None] - nodes[None, :]
    weights = np.ones((parameters.shape[0], nodes.shape[0]), dtype=np.float64)
    for node in range(nodes.shape[0]):
        for other in range(nodes.shape[0]):
            if other != node:
                weights[:, node] *= differences[:, other] / (nodes[node] - nodes[other])
    return weights


def gauss_legendre_unit(points: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights on the unit interval."""
    from ...integration import GaussLegendreRule, interval_rule_data

    data = interval_rule_data(GaussLegendreRule(positive_integer(points, "points")))
    nodes = np.asarray(data.nodes, dtype=np.float64)
    weights = np.asarray(data.weights, dtype=np.float64)
    return 0.5 * (nodes + 1.0), 0.5 * weights


@final
class InterfaceQuadraturePolicy(StrictModule, NonTrainableState):
    """Geometric tolerances and host work bound of one common refinement.

    `geometry_tolerance` is relative to the total arc length of the interface
    and bounds collinearity, facet straightness, coverage gaps, and overlaps.
    `normal_tolerance` bounds `|n_first + n_second|` of the two outward
    normals. `max_candidate_pairs` bounds the host facet-pair search.
    """

    geometry_tolerance: float = eqx.field(static=True)
    normal_tolerance: float = eqx.field(static=True)
    max_candidate_pairs: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_tolerance: float = 1.0e-10,
        normal_tolerance: float = 1.0e-8,
        max_candidate_pairs: int = 4_000_000,
    ) -> None:
        geometry = positive_finite_float(geometry_tolerance, "geometry_tolerance")
        normal = positive_finite_float(normal_tolerance, "normal_tolerance")
        pairs = positive_integer(max_candidate_pairs, "max_candidate_pairs")
        self.geometry_tolerance = geometry
        self.normal_tolerance = normal
        self.max_candidate_pairs = pairs
        self.policy_id = canonical_fingerprint(
            {
                "kind": "interface-quadrature-policy",
                "geometry_tolerance": geometry,
                "normal_tolerance": normal,
                "max_candidate_pairs": pairs,
            }
        )


@final
class InterfaceCoverageEvidence(StrictModule, NonTrainableState):
    """Measured coverage of the common refinement of two facet partitions.

    `first_coverage`/`second_coverage` are the covered fractions (measure of
    the union of common segments) of every facet in trace order;
    `maximum_gap` is the largest uncovered plus multiply covered arc length of
    any facet; `maximum_normal_defect` the largest `|n_first + n_second|` over
    the common segments.
    """

    first_measure: float = eqx.field(static=True)
    second_measure: float = eqx.field(static=True)
    common_measure: float = eqx.field(static=True)
    maximum_gap: float = eqx.field(static=True)
    maximum_normal_defect: float = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    first_coverage: Array
    second_coverage: Array

    def __init__(
        self,
        *,
        first_measure: float,
        second_measure: float,
        common_measure: float,
        maximum_gap: float,
        maximum_normal_defect: float,
        segment_count: int,
        first_coverage: np.ndarray,
        second_coverage: np.ndarray,
    ) -> None:
        self.first_measure = float(first_measure)
        self.second_measure = float(second_measure)
        self.common_measure = float(common_measure)
        self.maximum_gap = float(maximum_gap)
        self.maximum_normal_defect = float(maximum_normal_defect)
        self.segment_count = positive_integer(segment_count, "segment_count")
        self.first_coverage = jnp.asarray(first_coverage)
        self.second_coverage = jnp.asarray(second_coverage)


@final
class FacetResampling(StrictModule, NonTrainableState):
    """Exact evaluation of one polynomial facet trace at interface points.

    `facets[p]` is the trace facet (in trace order) containing point `p` and
    `weights[p]` the Lagrange weights of the trace's facet sites at the
    point's facet parameter. `apply` maps trace data `(facets, sites, ...)` to
    point values `(points, ...)`; `transpose` is its exact scatter-add. Both
    evaluate in the operand's precision: the float64 weights are rounded to
    the real dtype of an inexact operand, so float32 and complex64 data are
    never promoted.
    """

    facets: Array
    weights: Array
    facet_count: int = eqx.field(static=True)

    def __init__(
        self, facets: np.ndarray, weights: np.ndarray, /, *, facet_count: int
    ) -> None:
        indices = np.asarray(facets)
        values = np.asarray(weights, dtype=np.float64)
        count = positive_integer(facet_count, "facet_count")
        if indices.ndim != 1 or values.ndim != 2 or values.shape[0] != indices.shape[0]:
            raise ValueError("Resampling needs one facet and one weight row per point.")
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("Resampling facets lie outside the trace facets.")
        self.facets = jnp.asarray(indices.astype(np.int32))
        self.weights = jnp.asarray(values)
        self.facet_count = count

    def _weights_for(self, operand: Array, /) -> Array:
        if jnp.issubdtype(operand.dtype, jnp.inexact):
            return self.weights.astype(jnp.finfo(operand.dtype).dtype)
        return self.weights

    def apply(self, trace_values: Array, /) -> Array:
        gathered = trace_values[self.facets]
        return ein.contract("pq,pq...->p...", self._weights_for(trace_values), gathered)

    def transpose(self, values: Array, /) -> Array:
        payload = ein.contract("pq,p...->pq...", self._weights_for(values), values)
        zeros = jnp.zeros(
            (self.facet_count, self.weights.shape[1], *values.shape[1:]),
            dtype=payload.dtype,
        )
        return zeros.at[self.facets].add(payload)


@final
class InterfaceSideQuadrature(StrictModule, NonTrainableState):
    """One side of a common interface quadrature: its trace and resampling.

    `values` evaluates the side's prepared trace at the common points;
    `pullback` is its exact coordinate transpose onto the owner's full rows.
    `parameters` are the facet parameters of the common points on this side.
    """

    trace: PreparedTraceAction
    resampling: FacetResampling
    parameters: Array

    def __init__(
        self,
        trace: PreparedTraceAction,
        resampling: FacetResampling,
        parameters: np.ndarray,
        /,
    ) -> None:
        if not isinstance(trace, PreparedTraceAction):
            raise TypeError("trace must be a PreparedTraceAction.")
        if not isinstance(resampling, FacetResampling):
            raise TypeError("resampling must be a FacetResampling.")
        values = np.asarray(parameters, dtype=np.float64)
        if (
            resampling.facet_count != trace.output_shape[0]
            or resampling.weights.shape[1] != trace.output_shape[1]
            or values.shape != (resampling.weights.shape[0],)
        ):
            raise ValueError("The resampling does not match the side trace.")
        self.trace = trace
        self.resampling = resampling
        self.parameters = jnp.asarray(values)

    @property
    def facets(self) -> Array:
        """Trace facet (in trace order) of every common point."""
        return self.resampling.facets

    def values(self, coefficients: Array, /) -> Array:
        """Trace values of the owner's full coefficients at the common points."""
        return self.resampling.apply(self.trace.apply(coefficients))

    def pullback(self, covector: Array, /) -> Array:
        """Exact transpose of `values` onto the owner's full residual rows."""
        return self.trace.dual_pullback(self.resampling.transpose(covector))


@final
class InterfaceQuadrature(StrictModule, NonTrainableState):
    """Common quadrature of a two-sided 2-D curve interface.

    `points`, `weights` (physical arc length), and `normals` (outward from
    side 0) live on the common segments of `sides[0]` and `sides[1]`.
    `exact_degree` is the polynomial degree integrated exactly on every
    segment; `evidence` records coverage and normal opposition.
    """

    points: Array
    weights: Array
    normals: Array
    sides: tuple[InterfaceSideQuadrature, InterfaceSideQuadrature]
    exact_degree: int = eqx.field(static=True)
    evidence: InterfaceCoverageEvidence
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        normals: np.ndarray,
        sides: tuple[InterfaceSideQuadrature, InterfaceSideQuadrature],
        /,
        *,
        exact_degree: int,
        evidence: InterfaceCoverageEvidence,
        policy_id: str,
    ) -> None:
        points_ = np.asarray(points, dtype=np.float64)
        weights_ = np.asarray(weights, dtype=np.float64)
        normals_ = np.asarray(normals, dtype=np.float64)
        if (
            points_.ndim != 2
            or weights_.shape != points_.shape[:1]
            or normals_.shape != points_.shape
        ):
            raise ValueError("Interface points, weights, and normals do not agree.")
        if not isinstance(sides, tuple) or len(sides) != 2:
            raise TypeError("sides must be a pair of InterfaceSideQuadrature values.")
        for side in sides:
            if not isinstance(side, InterfaceSideQuadrature):
                raise TypeError("sides must be InterfaceSideQuadrature values.")
            if side.parameters.shape != weights_.shape:
                raise ValueError("Every side must resample every common point.")
        if not isinstance(evidence, InterfaceCoverageEvidence):
            raise TypeError("evidence must be InterfaceCoverageEvidence.")
        if np.any(weights_ <= 0.0):
            raise ValueError("Interface quadrature weights must be positive.")
        self.points = jnp.asarray(points_)
        self.weights = jnp.asarray(weights_)
        self.normals = jnp.asarray(normals_)
        self.sides = sides
        self.exact_degree = positive_integer(exact_degree, "exact_degree")
        self.evidence = evidence
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "interface-quadrature",
                "sides": [side.trace.action_id for side in sides],
                "points": array_tree_fingerprint(points_),
                "weights": array_tree_fingerprint(weights_),
                "exact_degree": exact_degree,
                "policy": policy_id,
            }
        )

    @property
    def point_count(self) -> int:
        return self.weights.shape[0]


def _gll_nodes(trace: PreparedTraceAction, role: str, /) -> np.ndarray:
    """Facet parameters of the trace sites, refusing non-GLL or non-exact traces."""
    descriptor = trace.descriptor
    sites_per_facet = trace.output_shape[1]
    if sites_per_facet < 2:
        raise ValueError(f"The {role} trace needs at least two sites per facet.")
    rule = FacetTraceRule("gauss-lobatto-legendre", points=sites_per_facet)
    if descriptor.rule_id != rule.rule_id:
        raise ValueError(
            f"The {role} trace must be prepared on a Gauss-Lobatto-Legendre facet "
            "rule, whose end points locate the facet exactly."
        )
    if (
        descriptor.quantity != "value"
        or descriptor.representation != "quadrature-values"
        or trace.sites.shape[-1] != 2
    ):
        raise ValueError(
            f"The {role} trace must be a pointwise scalar value trace on a 2-D curve."
        )
    if descriptor.trace_degree is None or descriptor.trace_degree > sites_per_facet - 1:
        raise ValueError(
            f"The {role} trace must publish a polynomial trace degree of at most "
            f"{sites_per_facet - 1} to be re-evaluated exactly between its sites."
        )
    parameters, _ = rule.reference("edge")
    return parameters[:, 0]


def _straight_facets(
    trace: PreparedTraceAction, nodes: np.ndarray, tolerance: float, role: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Facet end points, directions, lengths, and outward normals of one side."""
    sites = np.asarray(trace.sites, dtype=np.float64)
    normals = np.asarray(trace.normals, dtype=np.float64)
    start, stop = sites[:, 0], sites[:, -1]
    direction = stop - start
    length = np.linalg.norm(direction, axis=1)
    if np.any(length <= tolerance):
        raise ValueError(f"The {role} trace has a degenerate facet.")
    affine = start[:, None, :] + nodes[None, :, None] * direction[:, None, :]
    if np.max(np.linalg.norm(sites - affine, axis=-1)) > tolerance:
        raise ValueError(
            f"The {role} trace facets are not straight affine segments; the exact "
            "segment common refinement requires piecewise-linear interfaces."
        )
    facet_normal = normals[:, 0]
    if np.max(np.linalg.norm(normals - facet_normal[:, None, :], axis=-1)) > tolerance:
        raise ValueError(f"The {role} trace normals vary along a straight facet.")
    return start, direction, length, facet_normal


def _candidate_pairs(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    tolerance: float,
    limit: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Facet pairs whose bounding boxes overlap, in canonical (first, second) order."""
    (first_start, first_direction), (second_start, second_direction) = first, second
    if first_start.shape[0] * second_start.shape[0] > limit:
        raise ValueError(
            "The interface facet-pair search exceeds max_candidate_pairs; raise the "
            "declared bound or partition the interface."
        )
    first_low = np.minimum(first_start, first_start + first_direction) - tolerance
    first_high = np.maximum(first_start, first_start + first_direction) + tolerance
    second_low = np.minimum(second_start, second_start + second_direction)
    second_high = np.maximum(second_start, second_start + second_direction)
    overlap = np.all(
        (first_low[:, None, :] <= second_high[None, :, :])
        & (second_low[None, :, :] <= first_high[:, None, :]),
        axis=-1,
    )
    return np.nonzero(overlap)


def _common_segments(
    first: tuple[np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
    pairs: tuple[np.ndarray, np.ndarray],
    tolerance: float,
    /,
) -> np.ndarray:
    """Rows `(first facet, second facet, first low, first high)` of the refinement."""
    first_start, first_direction, first_length = first
    second_start, second_direction, _ = second
    rows, columns = pairs
    origin = first_start[rows]
    direction = first_direction[rows]
    length = first_length[rows]
    ends = (second_start[columns], second_start[columns] + second_direction[columns])
    offsets = tuple(end - origin for end in ends)
    distances = tuple(
        np.abs(direction[:, 0] * offset[:, 1] - direction[:, 1] * offset[:, 0]) / length
        for offset in offsets
    )
    parameters = tuple(
        np.sum(offset * direction, axis=1) / length**2 for offset in offsets
    )
    low = np.maximum(0.0, np.minimum(*parameters))
    high = np.minimum(1.0, np.maximum(*parameters))
    keep = (
        (distances[0] <= tolerance)
        & (distances[1] <= tolerance)
        & ((high - low) * length > tolerance)
    )
    segments = np.stack(
        (rows[keep], columns[keep], low[keep], high[keep]), axis=1
    ).astype(np.float64)
    order = np.lexsort((segments[:, 2], segments[:, 0]))
    return segments[order]


def _projected_parameters(
    segments: np.ndarray,
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Second-facet parameter interval of every common segment."""
    first_start, first_direction = first
    second_start, second_direction, second_length = second
    rows = segments[:, 0].astype(np.int64)
    columns = segments[:, 1].astype(np.int64)
    ends = tuple(
        np.sum(
            (
                first_start[rows]
                + segments[:, column, None] * first_direction[rows]
                - second_start[columns]
            )
            * second_direction[columns],
            axis=1,
        )
        / second_length[columns] ** 2
        for column in (2, 3)
    )
    return np.minimum(*ends), np.maximum(*ends)


def _coverage(
    facets: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    facet_lengths: np.ndarray,
    tolerance: float,
    subject: str,
    role: str,
    /,
) -> tuple[np.ndarray, float]:
    """Covered fraction of every facet and the largest tiling defect.

    The segments `[low, high]` (facet parameters) must tile every facet: their
    union covers `[0, 1]` and no two overlap. A facet's defect is its uncovered
    plus its multiply covered arc length, so a gap and an overlap on the same
    facet add instead of cancelling; a defect above `tolerance` is refused.
    """
    count = facet_lengths.shape[0]
    order = np.lexsort((low, facets))
    facet, start, stop = facets[order], low[order], high[order]
    size = facet.shape[0]
    # Running maximum of `stop` within each facet: ranks of `stop` offset by the
    # facet index keep earlier facets below later ones, exactly in integers.
    by_stop = np.argsort(stop, kind="stable")
    rank = np.empty((size,), dtype=np.int64)
    rank[by_stop] = np.arange(size, dtype=np.int64)
    reach_key = np.maximum.accumulate(facet * size + rank)
    continues = np.zeros((size,), dtype=np.bool_)
    continues[1:] = facet[1:] == facet[:-1]
    previous = np.zeros((size,), dtype=np.int64)
    previous[1:] = np.where(continues[1:], reach_key[:-1] - facet[1:] * size, 0)
    reach = np.where(continues, stop[by_stop][previous], 0.0)
    fresh = np.maximum(
        0.0,
        np.clip(stop, 0.0, 1.0)
        - np.maximum(np.clip(start, 0.0, 1.0), np.clip(reach, 0.0, 1.0)),
    )
    covered = np.bincount(facet, weights=fresh, minlength=count)
    total = np.bincount(facet, weights=stop - start, minlength=count)
    uncovered = (1.0 - covered) * facet_lengths
    repeated = (total - covered) * facet_lengths
    defect = uncovered + repeated
    worst = int(np.argmax(defect))
    if defect[worst] > tolerance:
        raise ValueError(
            f"{subject} do not cover each other: {role} facet "
            f"{worst} has uncovered length {uncovered[worst]:.3e} and multiply "
            f"covered length {repeated[worst]:.3e} (tolerance {tolerance:.3e})."
        )
    return covered, float(defect[worst])


def prepare_interface_quadrature(
    first: PreparedTraceAction,
    second: PreparedTraceAction,
    /,
    *,
    exact_degree: int,
    policy: InterfaceQuadraturePolicy | None = None,
) -> InterfaceQuadrature:
    """Prepare the exact common quadrature of two sides of one curve interface.

    Both traces are scalar value traces on Gauss--Lobatto--Legendre facet
    sites with published polynomial trace degrees. The facets of each side
    must be straight, the two partitions must cover each other completely
    without gaps or overlaps, and the two outward normals must be opposite.
    """
    if not isinstance(first, PreparedTraceAction) or not isinstance(
        second, PreparedTraceAction
    ):
        raise TypeError("Interface quadrature sides must be PreparedTraceAction values.")
    selected = InterfaceQuadraturePolicy() if policy is None else policy
    if not isinstance(selected, InterfaceQuadraturePolicy):
        raise TypeError("policy must be an InterfaceQuadraturePolicy.")
    degree = positive_integer(exact_degree, "exact_degree")
    first_nodes = _gll_nodes(first, "first")
    second_nodes = _gll_nodes(second, "second")
    first_sites = np.asarray(first.sites, dtype=np.float64)
    scale = float(np.sum(np.linalg.norm(first_sites[:, -1] - first_sites[:, 0], axis=1)))
    tolerance = selected.geometry_tolerance * scale
    first_start, first_direction, first_length, first_normal = _straight_facets(
        first, first_nodes, tolerance, "first"
    )
    second_start, second_direction, second_length, second_normal = _straight_facets(
        second, second_nodes, tolerance, "second"
    )
    pairs = _candidate_pairs(
        (first_start, first_direction),
        (second_start, second_direction),
        tolerance,
        selected.max_candidate_pairs,
    )
    segments = _common_segments(
        (first_start, first_direction, first_length),
        (second_start, second_direction, second_length),
        pairs,
        tolerance,
    )
    if segments.shape[0] == 0:
        raise ValueError("The two interface sides share no common segment.")
    first_facets = segments[:, 0].astype(np.int64)
    second_facets = segments[:, 1].astype(np.int64)
    lengths = (segments[:, 3] - segments[:, 2]) * first_length[first_facets]
    second_low, second_high = _projected_parameters(
        segments,
        (first_start, first_direction),
        (second_start, second_direction, second_length),
    )
    first_coverage, first_gap = _coverage(
        first_facets,
        segments[:, 2],
        segments[:, 3],
        first_length,
        tolerance,
        "The interface sides",
        "first",
    )
    second_coverage, second_gap = _coverage(
        second_facets,
        second_low,
        second_high,
        second_length,
        tolerance,
        "The interface sides",
        "second",
    )
    gap = max(first_gap, second_gap)
    normal_defect = float(
        np.max(
            np.linalg.norm(
                first_normal[first_facets] + second_normal[second_facets], axis=1
            )
        )
    )
    if normal_defect > selected.normal_tolerance:
        raise ValueError(
            "The two sides' outward normals are not opposite on the common "
            f"segments (defect {normal_defect:.3e}); a two-sided interface needs "
            "each owner's trace on its own side of the curve."
        )
    rule_points = degree // 2 + 1
    nodes, weights = gauss_legendre_unit(rule_points)
    low, high = segments[:, 2], segments[:, 3]
    first_parameters = (low[:, None] + nodes[None, :] * (high - low)[:, None]).reshape(-1)
    point_facets_first = np.repeat(first_facets, rule_points)
    point_facets_second = np.repeat(second_facets, rule_points)
    points = (
        first_start[point_facets_first]
        + first_parameters[:, None] * first_direction[point_facets_first]
    )
    second_parameters = (
        np.sum(
            (points - second_start[point_facets_second])
            * second_direction[point_facets_second],
            axis=1,
        )
        / second_length[point_facets_second] ** 2
    )
    point_weights = (weights[None, :] * lengths[:, None]).reshape(-1)
    evidence = InterfaceCoverageEvidence(
        first_measure=float(np.sum(first_length)),
        second_measure=float(np.sum(second_length)),
        common_measure=float(np.sum(lengths)),
        maximum_gap=gap,
        maximum_normal_defect=normal_defect,
        segment_count=segments.shape[0],
        first_coverage=first_coverage,
        second_coverage=second_coverage,
    )
    sides = (
        InterfaceSideQuadrature(
            first,
            FacetResampling(
                point_facets_first,
                _lagrange_weights(first_nodes, first_parameters),
                facet_count=first_start.shape[0],
            ),
            first_parameters,
        ),
        InterfaceSideQuadrature(
            second,
            FacetResampling(
                point_facets_second,
                _lagrange_weights(second_nodes, second_parameters),
                facet_count=second_start.shape[0],
            ),
            second_parameters,
        ),
    )
    return InterfaceQuadrature(
        points,
        point_weights,
        first_normal[point_facets_first],
        sides,
        exact_degree=2 * rule_points - 1,
        evidence=evidence,
        policy_id=selected.policy_id,
    )


def _nested_panel_facets(
    segments: np.ndarray, panel_count: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Owner facet of every panel, refusing panels that straddle facet ends."""
    panels = segments[:, 1].astype(np.int64)
    counts = np.bincount(panels, minlength=panel_count)
    straddling = np.flatnonzero(counts > 1)
    if straddling.size:
        raise ValueError(
            f"Boundary panels {straddling[:8].tolist()} span several facets of the "
            "volume trace; a per-panel rule integrates the kinked trace inexactly. "
            "Refine the boundary panels so that each lies inside one volume facet."
        )
    facets = np.zeros((panel_count,), dtype=np.int64)
    facets[panels] = segments[:, 0].astype(np.int64)
    return facets, panels


def prepare_boundary_panel_resampling(
    trace: PreparedTraceAction,
    panels: tuple[np.ndarray, np.ndarray, np.ndarray],
    points: np.ndarray,
    /,
    *,
    policy: InterfaceQuadraturePolicy | None = None,
) -> tuple[InterfaceSideQuadrature, InterfaceCoverageEvidence]:
    """Exact evaluation of a volume side trace at the sample points of boundary panels.

    ``panels`` are the ``(starts, directions, outward normals)`` of a straight
    boundary partition of the same curve, with the normals of the domain on
    the far side, and ``points`` its ``(panel, sample, 2)`` rule points. Every
    panel must lie inside one straight facet of ``trace`` (the panels refine
    the volume facets), the two partitions must cover each other, and the
    outward normals must be opposite. The returned side evaluates the trace at
    the points in ``(panel, sample)`` order with its exact scatter transpose.
    """
    if not isinstance(trace, PreparedTraceAction):
        raise TypeError("trace must be a PreparedTraceAction.")
    selected = InterfaceQuadraturePolicy() if policy is None else policy
    if not isinstance(selected, InterfaceQuadraturePolicy):
        raise TypeError("policy must be an InterfaceQuadraturePolicy.")
    panel_start, panel_direction, panel_normal = (
        np.asarray(value, dtype=np.float64) for value in panels
    )
    samples = np.asarray(points, dtype=np.float64)
    count = panel_start.shape[0]
    if samples.ndim != 3 or samples.shape[0] != count or samples.shape[2] != 2:
        raise ValueError("points must hold (panel, sample, 2) rule points.")
    nodes = _gll_nodes(trace, "volume")
    sites = np.asarray(trace.sites, dtype=np.float64)
    scale = float(np.sum(np.linalg.norm(sites[:, -1] - sites[:, 0], axis=1)))
    tolerance = selected.geometry_tolerance * scale
    start, direction, length, normal = _straight_facets(trace, nodes, tolerance, "volume")
    panel_length = np.linalg.norm(panel_direction, axis=1)
    segments = _common_segments(
        (start, direction, length),
        (panel_start, panel_direction, panel_length),
        _candidate_pairs(
            (start, direction),
            (panel_start, panel_direction),
            tolerance,
            selected.max_candidate_pairs,
        ),
        tolerance,
    )
    if segments.shape[0] == 0:
        raise ValueError("The boundary panels share no segment with the volume trace.")
    facet_of_segment = segments[:, 0].astype(np.int64)
    lengths = (segments[:, 3] - segments[:, 2]) * length[facet_of_segment]
    panel_low, panel_high = _projected_parameters(
        segments, (start, direction), (panel_start, panel_direction, panel_length)
    )
    first_coverage, first_gap = _coverage(
        facet_of_segment,
        segments[:, 2],
        segments[:, 3],
        length,
        tolerance,
        "The boundary panels and the volume trace",
        "volume",
    )
    second_coverage, second_gap = _coverage(
        segments[:, 1].astype(np.int64),
        panel_low,
        panel_high,
        panel_length,
        tolerance,
        "The boundary panels and the volume trace",
        "panel",
    )
    gap = max(first_gap, second_gap)
    facets, panel_of_segment = _nested_panel_facets(segments, count)
    normal_defect = float(
        np.max(
            np.linalg.norm(
                normal[facet_of_segment] + panel_normal[panel_of_segment], axis=1
            )
        )
    )
    if normal_defect > selected.normal_tolerance:
        raise ValueError(
            "The volume trace and the boundary panels do not face each other across "
            f"the curve (normal defect {normal_defect:.3e}); the volume must lie on "
            "the other side of the boundary from the panel owner's domain."
        )
    point_facets = np.repeat(facets, samples.shape[1])
    flat = samples.reshape((-1, 2))
    parameters = (
        np.sum((flat - start[point_facets]) * direction[point_facets], axis=1)
        / length[point_facets] ** 2
    )
    evidence = InterfaceCoverageEvidence(
        first_measure=float(np.sum(length)),
        second_measure=float(np.sum(panel_length)),
        common_measure=float(np.sum(lengths)),
        maximum_gap=gap,
        maximum_normal_defect=normal_defect,
        segment_count=segments.shape[0],
        first_coverage=first_coverage,
        second_coverage=second_coverage,
    )
    side = InterfaceSideQuadrature(
        trace,
        FacetResampling(
            point_facets,
            _lagrange_weights(nodes, parameters),
            facet_count=start.shape[0],
        ),
        parameters,
    )
    return side, evidence
