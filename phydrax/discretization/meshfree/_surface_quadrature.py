# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Positive immutable host-prepared surface integration rules."""

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...sparse import RowRelation
from ...typing import Bool, Float, parse, Scalar
from ._surface_geometry import SurfaceGeometryEvaluation, SurfaceNodeDim


SurfaceQuadratureKind = Literal["supplied", "tangent-voronoi", "normalized-density"]


@final
class SurfaceQuadratureEvidence(StrictModule):
    __strict_contract__ = True
    positive: Bool[Scalar]
    total_area: Float[Scalar]
    minimum_measure: Float[Scalar]
    maximum_measure: Float[Scalar]
    normalized_to_declared_area: bool = eqx.field(static=True)
    geometric_estimate: bool = eqx.field(static=True)
    frozen_support: bool = eqx.field(static=True)


@final
class SurfaceQuadraturePolicy(StrictModule):
    """Supplied areas, tangent Voronoi estimates, or explicitly normalized density.

    Density is a positive *sampling* density: area weights are proportional to
    its reciprocal. Only ``normalized-density`` consumes ``total_area``. Tangent
    Voronoi uses local chart cells, not a claimed exact curved-surface tessellation.
    """

    __strict_contract__ = True
    kind: SurfaceQuadratureKind = eqx.field(static=True)
    values: Float[SurfaceNodeDim] | None
    total_area: float | None = eqx.field(static=True)

    def __init__(
        self,
        kind: SurfaceQuadratureKind = "tangent-voronoi",
        *,
        measures: Array | None = None,
        density: Array | None = None,
        total_area: float | None = None,
    ) -> None:
        kind_ = parse(kind, SurfaceQuadratureKind, "kind")
        if kind == "supplied":
            if measures is None or density is not None or total_area is not None:
                raise ValueError("supplied requires measures only.")
            values = measures
        elif kind == "normalized-density":
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
        else:
            if any(value is not None for value in (measures, density, total_area)):
                raise ValueError(
                    "tangent-voronoi does not accept supplied measures/density/area."
                )
            values = None
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

    def prepare(
        self, geometry: SurfaceGeometryEvaluation, relation: RowRelation
    ) -> tuple[Array, SurfaceQuadratureEvidence]:
        count = geometry.points.shape[0]
        if self.kind == "tangent-voronoi":
            points, frames = (
                np.asarray(geometry.points),
                np.asarray(geometry.tangent_frames),
            )
            routes, valid = (
                np.asarray(relation.source_indices),
                np.asarray(relation.valid),
            )
            areas = np.empty(count, dtype=points.dtype)
            for row in range(count):
                offsets = (points[routes[row, valid[row]]] - points[row]) @ frames[row]
                lengths = np.linalg.norm(offsets, axis=-1)
                offsets = offsets[lengths > np.finfo(points.dtype).eps]
                if offsets.shape[0] < 3:
                    raise ValueError(
                        "Tangent Voronoi row has insufficient distinct neighbors."
                    )
                bound = 4 * float(np.max(np.linalg.norm(offsets, axis=-1)))
                polygon = np.array(
                    [[-bound, -bound], [bound, -bound], [bound, bound], [-bound, bound]],
                    dtype=points.dtype,
                )
                for offset in offsets:
                    level = float(offset @ offset) / 2
                    output: list[np.ndarray] = []
                    for first, second in zip(
                        polygon, np.roll(polygon, -1, axis=0), strict=True
                    ):
                        a, b = (
                            float(first @ offset - level),
                            float(second @ offset - level),
                        )
                        if a <= 0:
                            output.append(first)
                        if (a <= 0) != (b <= 0):
                            output.append(first + (second - first) * (a / (a - b)))
                    polygon = np.asarray(output, dtype=points.dtype).reshape((-1, 2))
                    if polygon.shape[0] == 0:
                        raise ValueError("Empty tangent Voronoi cell.")
                if np.any(np.abs(polygon) >= bound * (1 - 1e-8)):
                    raise ValueError(
                        "Unbounded tangent Voronoi cell: open/undersampled surface refused."
                    )
                areas[row] = (
                    abs(
                        np.sum(
                            polygon[:, 0] * np.roll(polygon[:, 1], -1)
                            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                        )
                    )
                    / 2
                )
        else:
            areas = np.asarray(
                self.values, dtype=np.asarray(geometry.points).dtype
            ).copy()
            if (
                areas.shape != (count,)
                or np.any(~np.isfinite(areas))
                or np.any(areas <= 0)
            ):
                raise ValueError(
                    "Surface quadrature requires one finite positive value per point."
                )
            if self.kind == "normalized-density":
                areas = 1 / areas
                areas *= self.total_area / np.sum(areas)
        if np.any(~np.isfinite(areas)) or np.any(areas <= 0):
            raise ValueError("Surface measures must be finite and positive.")
        measures = jnp.asarray(areas, dtype=geometry.points.dtype)
        evidence = SurfaceQuadratureEvidence(
            positive=jnp.all(measures > 0),
            total_area=jnp.sum(measures),
            minimum_measure=jnp.min(measures),
            maximum_measure=jnp.max(measures),
            normalized_to_declared_area=self.kind == "normalized-density",
            geometric_estimate=self.kind == "tangent-voronoi",
            frozen_support=True,
        )
        return measures, evidence


__all__ = [
    "SurfaceQuadratureKind",
    "SurfaceQuadratureEvidence",
    "SurfaceQuadraturePolicy",
]
