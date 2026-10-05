# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Surface node measures: authoritative chart cubature, estimates, declarations."""

from __future__ import annotations

from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...ein import contract
from ...geometry._atlas import AbstractBoundaryMap, BoundaryAtlas
from ...geometry._cubature import CubatureAtlas
from ...integration._rules import CubatureRule, GaussLegendreRule
from ...linalg import orthonormal_frame
from ...metrix._embedded import EmbeddedChart
from ...sparse import RowRelation
from ...typing import Bool, ConvertibleToArray, Float, parse, Scalar
from ._neighbors import _integer, MeshfreeNeighborhoodPlan
from ._stencils import fit_chart_stencils, LocalStencilPolicy
from ._surface_geometry import SurfaceGeometryEvaluation, SurfaceNodeDim


SurfaceQuadratureKind: TypeAlias = Literal[
    "supplied", "tangent-voronoi", "normalized-density", "chart-cubature"
]


@final
class SurfaceQuadratureEvidence(StrictModule):
    """Measure evidence; ``reference_area`` is the authoritative atlas integral.

    For estimates and declarations ``reference_area`` and ``transfer_residual``
    are NaN: a supplied or normalized total is not an independent area test.
    """

    __strict_contract__ = True
    positive: Bool[Scalar]
    total_area: Float[Scalar]
    minimum_measure: Float[Scalar]
    maximum_measure: Float[Scalar]
    reference_area: Float[Scalar]
    transfer_residual: Float[Scalar]
    normalized_to_declared_area: bool = eqx.field(static=True)
    geometric_estimate: bool = eqx.field(static=True)
    authoritative: bool = eqx.field(static=True)
    frozen_support: bool = eqx.field(static=True)


def _clip(
    polygon: Array, count: Array, offset: Array, active: Array
) -> tuple[Array, Array]:
    """One bounded Sutherland-Hodgman half-plane clip ``x.o <= |o|^2/2``."""
    capacity = polygon.shape[0]
    index = jnp.arange(capacity)
    following = jnp.where(index + 1 < count, index + 1, 0)
    first, second = polygon, polygon[following]
    level = offset @ offset / 2
    a, b = first @ offset - level, second @ offset - level
    present = index < count
    keep = present & (a <= 0)
    crossing = present & ((a <= 0) != (b <= 0))
    ratio = a / jnp.where(crossing, a - b, 1)
    intersection = first + (second - first) * ratio[:, None]
    candidates = jnp.stack((first, intersection), axis=1).reshape((-1, 2))
    emitted = jnp.stack((keep, crossing), axis=1).reshape(-1)
    slots = jnp.cumsum(emitted) - 1
    clipped = (
        jnp.zeros_like(polygon)
        .at[jnp.where(emitted & (slots < capacity), slots, capacity)]
        .set(candidates, mode="drop")
    )
    new_count = jnp.sum(emitted, dtype=count.dtype)
    return (
        jnp.where(active, clipped, polygon),
        jnp.where(active, new_count, count),
    )


def _voronoi_row(offsets: Array, valid: Array) -> tuple[Array, Array, Array]:
    """Bounded tangent Voronoi cell area of one row, its boundedness and support."""
    dtype = offsets.dtype
    lengths = jnp.linalg.norm(offsets, axis=-1)
    active = valid & (lengths > jnp.finfo(dtype).eps)
    if offsets.shape[-1] == 1:
        coordinate = offsets[:, 0]
        right = jnp.min(jnp.where(active & (coordinate > 0), coordinate, jnp.inf))
        left = jnp.max(jnp.where(active & (coordinate < 0), coordinate, -jnp.inf))
        bounded = jnp.isfinite(right) & jnp.isfinite(left)
        return jnp.where(bounded, (right - left) / 2, 0), bounded, bounded
    bound = 4 * jnp.max(jnp.where(active, lengths, 0))
    capacity = 4 + offsets.shape[0]
    square = jnp.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=dtype) * bound
    polygon = jnp.zeros((capacity, 2), dtype=dtype).at[:4].set(square)

    def step(
        state: tuple[Array, Array], item: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], None]:
        clipped = _clip(state[0], state[1], item[0], item[1])
        return clipped, None

    (polygon, count), _ = jax.lax.scan(
        step, (polygon, jnp.asarray(4, dtype=jnp.int32)), (offsets, active)
    )
    present = jnp.arange(capacity) < count
    following = jnp.where(jnp.arange(capacity) + 1 < count, jnp.arange(capacity) + 1, 0)
    nxt = polygon[following]
    area = (
        jnp.abs(
            jnp.sum(
                jnp.where(
                    present,
                    polygon[:, 0] * nxt[:, 1] - polygon[:, 1] * nxt[:, 0],
                    0,
                )
            )
        )
        / 2
    )
    bounded = jnp.all(
        ~present | (jnp.max(jnp.abs(polygon), axis=-1) < bound * (1 - 1e-8))
    )
    return area, bounded & (count >= 3), jnp.sum(active) >= 3


def tangent_voronoi_measures(
    geometry: SurfaceGeometryEvaluation, relation: RowRelation
) -> tuple[Array, Array, Array]:
    """Compiled bounded local tangent cells: an estimate, not a tessellation."""
    offsets = contract(
        "nka,nai->nki",
        geometry.points[relation.source_indices] - geometry.points[:, None, :],
        geometry.tangent_frames,
    )
    return jax.vmap(_voronoi_row)(offsets, relation.valid)


def _composite_reference(
    dimension: int, order: int, subdivisions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre rule on the unit reference cube."""
    data = GaussLegendreRule(order).data()
    nodes = (np.asarray(data.nodes) + 1) / 2
    weights = np.asarray(data.weights) / 2
    cells = np.arange(subdivisions)
    axis = ((cells[:, None] + nodes[None, :]) / subdivisions).reshape(-1)
    axis_weights = np.tile(weights, subdivisions) / subdivisions
    grids = np.meshgrid(*([axis] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([axis_weights] * dimension), indexing="ij")
    points = np.stack([grid.reshape(-1) for grid in grids], axis=-1)
    return points, np.prod(np.stack([grid.reshape(-1) for grid in weight_grids]), axis=0)


@final
class _EmbeddedChartBoxMap(AbstractBoundaryMap):
    """Unit reference cube -> declared chart box -> embedding (one chart)."""

    chart: EmbeddedChart
    lower: Array
    upper: Array

    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_dimension(self) -> int:
        return self.chart.chart.dimension

    @property
    def ambient_dimension(self) -> int:
        return self.chart.ambient_dimension

    def _coordinates(self, reference: Array) -> Array:
        return self.lower + (self.upper - self.lower) * reference

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return self.chart(self._coordinates(reference))

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return self.chart.volume_density(self._coordinates(reference)) * jnp.prod(
            self.upper - self.lower
        )


def chart_box_atlas(
    chart: EmbeddedChart,
    lower: ConvertibleToArray,
    upper: ConvertibleToArray,
    *,
    source_id: str,
) -> BoundaryAtlas:
    """Authoritative atlas of a box patch of one declared embedded chart.

    The measure scale is the metrix induced-metric volume density times the
    box volume, so atlas cubature integrates the true patch area element.
    """
    if not isinstance(chart, EmbeddedChart):
        raise TypeError("chart must be a metrix EmbeddedChart.")
    low = jnp.asarray(lower, dtype=jnp.float64).reshape(-1)
    high = jnp.asarray(upper, dtype=jnp.float64).reshape(-1)
    if (
        low.shape != (chart.chart.dimension,)
        or high.shape != low.shape
        or bool(jnp.any(high <= low))
    ):
        raise ValueError("Box bounds must be ordered and match the chart dimension.")
    return BoundaryAtlas(
        _EmbeddedChartBoxMap(chart, low, high),
        source_entity_ids=jnp.zeros((1,), dtype=jnp.int32),
        source_id=source_id,
    )


@final
class SurfaceQuadraturePolicy(StrictModule):
    """Node measures from authoritative chart cubature, estimates or declarations.

    ``chart-cubature`` integrates the declared atlas (a ``BoundaryAtlas`` with a
    composite Gauss-Legendre rule, or a ``CubatureAtlas`` with its native rule)
    and transfers the exact chart/Jacobian quadrature to nodes through bounded
    GMLS value reconstruction, so node measures reproduce the atlas integral of
    locally polynomial fields. ``tangent-voronoi`` remains a compiled local
    estimate. ``normalized-density`` consumes an explicit ``total_area``.
    """

    __strict_contract__ = True
    kind: SurfaceQuadratureKind = eqx.field(static=True)
    values: Float[SurfaceNodeDim] | None
    total_area: float | None = eqx.field(static=True)
    atlas: BoundaryAtlas | CubatureAtlas | None
    cubature_rule: CubatureRule | None
    quadrature_order: int = eqx.field(static=True)
    subdivisions: int = eqx.field(static=True)
    reconstruction: LocalStencilPolicy | None
    reconstruction_neighbors: int = eqx.field(static=True)

    def __init__(
        self,
        kind: SurfaceQuadratureKind = "tangent-voronoi",
        *,
        measures: Array | None = None,
        density: Array | None = None,
        total_area: float | None = None,
        atlas: BoundaryAtlas | CubatureAtlas | None = None,
        cubature_rule: CubatureRule | None = None,
        quadrature_order: int = 4,
        subdivisions: int = 8,
        reconstruction: LocalStencilPolicy | None = None,
        reconstruction_neighbors: int = 16,
    ) -> None:
        kind_ = parse(kind, SurfaceQuadratureKind, "kind")
        chart_inputs = (atlas, cubature_rule, reconstruction)
        match kind_:
            case "supplied":
                if measures is None or density is not None or total_area is not None:
                    raise ValueError("supplied requires measures only.")
                values = measures
            case "normalized-density":
                if (
                    density is None
                    or measures is not None
                    or total_area is None
                    or not np.isfinite(total_area)
                    or total_area <= 0
                ):
                    raise ValueError(
                        "normalized-density requires density and explicit positive total_area."
                    )
                values = density
            case "tangent-voronoi":
                if any(value is not None for value in (measures, density, total_area)):
                    raise ValueError(
                        "tangent-voronoi does not accept supplied measures/density/area."
                    )
                values = None
            case "chart-cubature":
                if any(value is not None for value in (measures, density, total_area)):
                    raise ValueError(
                        "chart-cubature integrates its atlas; measures/density/area are refused."
                    )
                if not isinstance(atlas, (BoundaryAtlas, CubatureAtlas)):
                    raise TypeError(
                        "chart-cubature requires a BoundaryAtlas or CubatureAtlas."
                    )
                if isinstance(atlas, CubatureAtlas) != isinstance(
                    cubature_rule, CubatureRule
                ):
                    raise ValueError(
                        "A CubatureAtlas requires its CubatureRule; a BoundaryAtlas uses Gauss-Legendre."
                    )
                if (
                    isinstance(atlas, CubatureAtlas)
                    and cubature_rule is not None
                    and cubature_rule.reference_domain != atlas.reference_domain
                ):
                    raise ValueError("Cubature rule reference does not match the atlas.")
                values = None
            case _:
                assert_never(kind_)
        if kind_ != "chart-cubature" and any(value is not None for value in chart_inputs):
            raise ValueError(
                "Atlas, cubature rule and reconstruction are chart-cubature only."
            )
        order = _integer(quadrature_order, "quadrature_order")
        cells = _integer(subdivisions, "subdivisions")
        neighbors = _integer(reconstruction_neighbors, "reconstruction_neighbors", 2)
        policy = (
            LocalStencilPolicy(polynomial_degree=2, chunk_rows=256)
            if reconstruction is None and kind_ == "chart-cubature"
            else reconstruction
        )
        if policy is not None and not isinstance(policy, LocalStencilPolicy):
            raise TypeError("reconstruction must be a LocalStencilPolicy.")
        values_ = None if values is None else jnp.asarray(values)
        if values_ is not None:
            host = np.asarray(values_)
            if (
                host.ndim != 1
                or host.size == 0
                or not np.issubdtype(host.dtype, np.floating)
                or np.any(~np.isfinite(host))
                or np.any(host <= 0)
            ):
                raise ValueError(
                    "Quadrature values must be a nonempty finite positive floating vector."
                )
        self.kind, self.values, self.total_area = kind_, values_, total_area
        self.atlas, self.cubature_rule = atlas, cubature_rule
        self.quadrature_order, self.subdivisions = order, cells
        self.reconstruction, self.reconstruction_neighbors = policy, neighbors

    def atlas_quadrature(self) -> tuple[Array, Array, Array]:
        """Atlas quadrature points, positive weights and tangent frames (host)."""
        atlas = self.atlas
        if atlas is None:
            raise ValueError("Only chart-cubature declares an atlas quadrature.")
        dimension = atlas.reference_dimension
        if isinstance(atlas, BoundaryAtlas):
            reference, weights = _composite_reference(
                dimension, self.quadrature_order, self.subdivisions
            )
        else:
            rule = self.cubature_rule
            if rule is None:
                raise ValueError("CubatureAtlas quadrature requires its cubature rule.")
            data = rule.materialize()
            reference, weights = np.asarray(data.points), np.asarray(data.weights)
        charts = atlas.num_charts
        reference_ = jnp.broadcast_to(
            jnp.asarray(reference), (charts, *reference.shape)
        ).reshape((-1, dimension))
        chart_indices = jnp.repeat(
            jnp.arange(charts, dtype=jnp.int32), reference.shape[0]
        )
        if isinstance(atlas, BoundaryAtlas):
            points = atlas.map(chart_indices, reference_)
            scale = atlas.jacobian(chart_indices, reference_)
            active = (
                atlas.reference_mask(chart_indices, reference_)
                & atlas.seam_owner[chart_indices]
                & (scale > 0)
            )
        else:
            evaluation = atlas.evaluate(chart_indices, reference_)
            points, scale, active = (
                evaluation.points,
                evaluation.measure_scale,
                evaluation.admissible,
            )
        mapping = atlas.mapping
        jacobian = jax.vmap(
            lambda index, value: jax.jacfwd(
                lambda u: mapping.map(index[None], u[None])[0]
            )(value)
        )(chart_indices, reference_)
        frame = orthonormal_frame(jacobian)
        selected = np.flatnonzero(np.asarray(active & frame.successful))
        if selected.size == 0:
            raise ValueError("Atlas quadrature has no admissible points.")
        point_weights = jnp.tile(jnp.asarray(weights), charts) * scale
        return (
            points[selected],
            point_weights[selected],
            frame.tangents[selected],
        )

    def _chart_cubature(
        self, geometry: SurfaceGeometryEvaluation
    ) -> tuple[Array, Array, Array]:
        atlas, policy = self.atlas, self.reconstruction
        if atlas is None or policy is None:
            raise ValueError("chart-cubature requires its atlas and reconstruction.")
        if (
            atlas.reference_dimension != geometry.intrinsic_dimension
            or atlas.ambient_dimension != geometry.ambient_dimension
        ):
            raise ValueError(
                "Atlas dimensions must match the declared surface embedding."
            )
        points, weights, frames = self.atlas_quadrature()
        nodes = geometry.points
        neighbors = min(self.reconstruction_neighbors, nodes.shape[0])
        relation = (
            MeshfreeNeighborhoodPlan(nodes, neighbors, targets=points).prepare().relation
        )
        offsets = contract(
            "qka,qai->qki", nodes[relation.source_indices] - points[:, None, :], frames
        )
        values, evidence = fit_chart_stencils(
            offsets,
            relation.valid,
            ((0,) * geometry.intrinsic_dimension,),
            jnp.ones((points.shape[0], 1, 1), dtype=offsets.dtype),
            policy,
        )
        stencils, statuses = values[:, 0, :], evidence.status
        if np.any(np.asarray(statuses)):
            raise ValueError(
                "Chart-cubature transfer refused: a quadrature point lacks a unisolvent node support."
            )
        measures = (
            jnp.zeros((nodes.shape[0],), dtype=nodes.dtype)
            .at[relation.source_indices.reshape(-1)]
            .add((weights[:, None] * stencils).reshape(-1))
        )
        # Independent transfer audit: ambient linear moments of the atlas rule.
        monomials = jnp.concatenate((jnp.ones_like(nodes[:, :1]), nodes), axis=1)
        exact = jnp.concatenate((jnp.ones_like(points[:, :1]), points), axis=1)
        transferred = monomials.T @ measures
        reference = exact.T @ weights
        residual = jnp.max(jnp.abs(transferred - reference)) / jnp.sum(weights)
        return measures, jnp.sum(weights), residual

    def prepare(
        self, geometry: SurfaceGeometryEvaluation, relation: RowRelation
    ) -> tuple[Array, SurfaceQuadratureEvidence]:
        count = geometry.points.shape[0]
        dtype = geometry.points.dtype
        reference_area = jnp.asarray(jnp.nan, dtype=dtype)
        transfer_residual = jnp.asarray(jnp.nan, dtype=dtype)
        match self.kind:
            case "tangent-voronoi":
                measures, bounded, supported = tangent_voronoi_measures(
                    geometry, relation
                )
                if not np.all(np.asarray(supported)):
                    raise ValueError(
                        "Tangent Voronoi row has insufficient distinct neighbors."
                    )
                if not np.all(np.asarray(bounded)):
                    raise ValueError(
                        "Unbounded tangent Voronoi cell: open/undersampled surface refused."
                    )
            case "chart-cubature":
                measures, reference_area, transfer_residual = self._chart_cubature(
                    geometry
                )
            case "supplied" | "normalized-density":
                host = np.asarray(self.values, dtype=np.asarray(geometry.points).dtype)
                if (
                    host.shape != (count,)
                    or np.any(~np.isfinite(host))
                    or np.any(host <= 0)
                ):
                    raise ValueError(
                        "Surface quadrature requires one finite positive value per point."
                    )
                if self.kind == "normalized-density":
                    host = 1 / host
                    host = host * (self.total_area / np.sum(host))
                measures = jnp.asarray(host, dtype=dtype)
            case _:
                assert_never(self.kind)
        host_measures = np.asarray(measures)
        if np.any(~np.isfinite(host_measures)) or np.any(host_measures <= 0):
            raise ValueError("Surface measures must be finite and positive.")
        evidence = SurfaceQuadratureEvidence(
            positive=jnp.all(measures > 0),
            total_area=jnp.sum(measures),
            minimum_measure=jnp.min(measures),
            maximum_measure=jnp.max(measures),
            reference_area=reference_area,
            transfer_residual=transfer_residual,
            normalized_to_declared_area=self.kind == "normalized-density",
            geometric_estimate=self.kind == "tangent-voronoi",
            authoritative=self.kind == "chart-cubature",
            frozen_support=True,
        )
        return measures, evidence


__all__ = [
    "SurfaceQuadratureKind",
    "SurfaceQuadratureEvidence",
    "SurfaceQuadraturePolicy",
    "chart_box_atlas",
]
