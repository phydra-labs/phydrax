#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-dimensional conservative transport on arbitrary positive measures."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MetricLineConservationEvidence(StrictModule):
    total_amount_rate: Array
    boundary_amount_rate: Array
    source_amount_rate: Array
    closure_residual: Array
    finite: Array
    successful: Array


class PreparedMetricLine(StrictModule, NonTrainableState):
    """Fixed-capacity line geometry with exact telescoping face incidence."""

    coordinate_faces: Array
    coordinate_cells: Array
    cell_measures: Array
    face_measures: Array
    active_cells: Array
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.cell_measures.shape[0])

    @property
    def face_count(self) -> int:
        return self.cell_count + 1

    def _cell_values(self, values: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[:1] != (self.cell_count,):
            raise ValueError(f"{name} must begin with the metric-line cell count.")
        return array

    def _face_values(self, values: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[:1] != (self.face_count,):
            raise ValueError(f"{name} must begin with the metric-line face count.")
        return array

    def total_amount(self, cell_density: ArrayLike, /) -> Array:
        density = self._cell_values(cell_density, "cell_density")
        scale = self.cell_measures.reshape(
            self.cell_measures.shape + (1,) * (density.ndim - 1)
        )
        active = self.active_cells.reshape(
            self.active_cells.shape + (1,) * (density.ndim - 1)
        )
        return jnp.sum(jnp.where(active, density * scale, 0.0), axis=0)

    def integrated_face_flux(self, face_flux_density: ArrayLike, /) -> Array:
        flux = self._face_values(face_flux_density, "face_flux_density")
        scale = self.face_measures.reshape(
            self.face_measures.shape + (1,) * (flux.ndim - 1)
        )
        return flux * scale

    def amount_rate_from_integrated_flux(
        self,
        integrated_face_flux: ArrayLike,
        /,
        *,
        source_density: ArrayLike | None = None,
    ) -> Array:
        integrated = self._face_values(integrated_face_flux, "integrated_face_flux")
        result = integrated[:-1] - integrated[1:]
        if source_density is not None:
            source = self._cell_values(source_density, "source_density")
            scale = self.cell_measures.reshape(
                self.cell_measures.shape + (1,) * (source.ndim - 1)
            )
            result = result + source * scale
        active = self.active_cells.reshape(
            self.active_cells.shape + (1,) * (result.ndim - 1)
        )
        return jnp.where(active, result, 0.0)

    def amount_rate(
        self,
        face_flux_density: ArrayLike,
        /,
        *,
        source_density: ArrayLike | None = None,
    ) -> Array:
        return self.amount_rate_from_integrated_flux(
            self.integrated_face_flux(face_flux_density),
            source_density=source_density,
        )

    def gradient(
        self,
        cell_values: ArrayLike,
        /,
        *,
        lower_boundary_value: ArrayLike,
        upper_boundary_value: ArrayLike,
    ) -> Array:
        values = self._cell_values(cell_values, "cell_values")
        lower = jnp.asarray(lower_boundary_value, dtype=values.dtype)
        upper = jnp.asarray(upper_boundary_value, dtype=values.dtype)
        if lower.shape != values.shape[1:] or upper.shape != values.shape[1:]:
            raise ValueError("Boundary values must match the cell component shape.")
        padded = jnp.concatenate((lower[None, ...], values, upper[None, ...]), axis=0)
        locations = jnp.concatenate(
            (
                self.coordinate_faces[:1],
                self.coordinate_cells,
                self.coordinate_faces[-1:],
            )
        )
        distances = jnp.diff(locations)
        scale = distances.reshape(distances.shape + (1,) * (values.ndim - 1))
        return jnp.diff(padded, axis=0) / scale

    def conservation_evidence(
        self,
        face_flux_density: ArrayLike,
        /,
        *,
        source_density: ArrayLike | None = None,
    ) -> MetricLineConservationEvidence:
        integrated = self.integrated_face_flux(face_flux_density)
        amount_rate = self.amount_rate(face_flux_density, source_density=source_density)
        total = jnp.sum(amount_rate, axis=0)
        boundary = integrated[0] - integrated[-1]
        source = jnp.zeros_like(total)
        if source_density is not None:
            source = self.total_amount(source_density)
        residual = total - boundary - source
        finite = (
            jnp.all(jnp.isfinite(total))
            & jnp.all(jnp.isfinite(boundary))
            & jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(residual))
        )
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(total),
                jnp.maximum(jnp.abs(boundary), jnp.abs(source)),
            ),
        )
        tolerance = 256.0 * jnp.finfo(jnp.result_type(total, 1.0)).eps * scale
        successful = finite & jnp.all(jnp.abs(residual) <= tolerance)
        return MetricLineConservationEvidence(
            total, boundary, source, residual, finite, successful
        )


@dataclass(frozen=True, slots=True)
class MetricLinePlan:
    """Host-side metric-line geometry prepared into fixed-shape JAX arrays."""

    coordinate_faces: np.ndarray
    cell_measures: np.ndarray
    face_measures: np.ndarray
    geometry_id: str
    active_cells: np.ndarray | None = None
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        faces = np.array(self.coordinate_faces, dtype=np.float64, copy=True)
        cells = np.array(self.cell_measures, dtype=np.float64, copy=True)
        face_measures = np.array(self.face_measures, dtype=np.float64, copy=True)
        if faces.ndim != 1 or faces.size < 2 or np.any(~np.isfinite(faces)):
            raise ValueError("coordinate_faces must be a finite rank-one array.")
        if np.any(np.diff(faces) <= 0.0):
            raise ValueError("coordinate_faces must be strictly increasing.")
        count = faces.size - 1
        if cells.shape != (count,) or face_measures.shape != (count + 1,):
            raise ValueError("Metric-line measures do not match the face topology.")
        if np.any(~np.isfinite(cells)) or np.any(cells <= 0.0):
            raise ValueError("cell_measures must be finite and positive.")
        if np.any(~np.isfinite(face_measures)) or np.any(face_measures < 0.0):
            raise ValueError("face_measures must be finite and nonnegative.")
        active = (
            np.ones((count,), dtype=bool)
            if self.active_cells is None
            else np.array(self.active_cells, dtype=bool, copy=True)
        )
        if active.shape != (count,) or not np.any(active):
            raise ValueError("active_cells must match cells and contain an active cell.")
        geometry = str(self.geometry_id).strip()
        if not geometry or geometry != self.geometry_id:
            raise ValueError("geometry_id must be non-empty canonical text.")
        for name, value in (
            ("coordinate_faces", faces),
            ("cell_measures", cells),
            ("face_measures", face_measures),
            ("active_cells", active),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "geometry_id", geometry)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "metric-line-plan",
                    "coordinate_faces": array_tree_fingerprint(faces),
                    "cell_measures": array_tree_fingerprint(cells),
                    "face_measures": array_tree_fingerprint(face_measures),
                    "active_cells": array_tree_fingerprint(active),
                    "geometry": geometry,
                }
            ),
        )

    def prepare(self) -> PreparedMetricLine:
        faces = jnp.asarray(self.coordinate_faces)
        return PreparedMetricLine(
            faces,
            0.5 * (faces[:-1] + faces[1:]),
            jnp.asarray(self.cell_measures),
            jnp.asarray(self.face_measures),
            jnp.asarray(self.active_cells),
            self.geometry_id,
            self.plan_id,
        )


__all__ = [
    "MetricLineConservationEvidence",
    "MetricLinePlan",
    "PreparedMetricLine",
]
