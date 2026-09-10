#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matched X-ray projectors, detector response, and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._interpolation import linear_interpolate

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...measurement import MeasurementAsset, RaySampleSupport


@dataclass(frozen=True, slots=True)
class ProjectionSupport:
    rays: RaySampleSupport
    projection_shape: tuple[int, ...]
    view_ids: tuple[str, ...]
    support_id: str = field(init=False)
    sample_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rays, RaySampleSupport):
            raise TypeError("rays must be RaySampleSupport.")
        shape = tuple(int(value) for value in self.projection_shape)
        if (
            not shape
            or any(value < 1 for value in shape)
            or int(np.prod(shape)) != self.rays.sample_shape[0]
        ):
            raise ValueError("projection_shape must be positive and contain every ray.")
        views = tuple(str(value) for value in self.view_ids)
        if not views or len(views) != shape[0] or len(views) != len(set(views)):
            raise ValueError("view_ids must uniquely label the first projection axis.")
        object.__setattr__(self, "projection_shape", shape)
        object.__setattr__(self, "view_ids", views)
        object.__setattr__(self, "sample_shape", shape)
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "projection-support",
                    "rays": self.rays.support_id,
                    "shape": list(shape),
                    "views": list(views),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ProjectionAsset:
    measurement: MeasurementAsset

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MeasurementAsset) or not isinstance(
            self.measurement.field.support, ProjectionSupport
        ):
            raise TypeError(
                "ProjectionAsset requires a MeasurementAsset on ProjectionSupport."
            )


class XRayTransformEvidence(StrictModule, NonTrainableState):
    route_complete: Array
    path_length: Array
    finite: Array
    successful: Array
    operator_id: str = eqx.field(static=True)


class XRayProjectionResult(StrictModule):
    values: Array
    evidence: XRayTransformEvidence


class VoxelXRayTransformPlan(StrictModule, NonTrainableState):
    voxel_indices: Array
    segment_lengths: Array
    segment_valid: Array
    projection_shape: tuple[int, ...] = eqx.field(static=True)
    volume_shape: tuple[int, int, int] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: ProjectionSupport,
        volume_shape: tuple[int, int, int],
        origin: ArrayLike,
        spacing: ArrayLike,
        /,
    ):
        shape = tuple(int(value) for value in volume_shape)
        origin_ = np.asarray(origin, dtype=float)
        spacing_ = np.asarray(spacing, dtype=float)
        if (
            len(shape) != 3
            or any(value < 1 for value in shape)
            or origin_.shape != (3,)
            or spacing_.shape != (3,)
            or np.any(spacing_ <= 0.0)
        ):
            raise ValueError(
                "Voxel geometry requires positive three-dimensional shape and spacing."
            )
        indices, lengths, valid = _siddon_routes(support.rays, shape, origin_, spacing_)
        self.voxel_indices = jnp.asarray(indices)
        self.segment_lengths = jnp.asarray(lengths)
        self.segment_valid = jnp.asarray(valid)
        self.projection_shape = support.projection_shape
        self.volume_shape = shape
        self.operator_id = canonical_fingerprint(
            {
                "kind": "voxel-xray-transform",
                "support": support.support_id,
                "shape": list(shape),
                "origin": origin_.tolist(),
                "spacing": spacing_.tolist(),
                "routes": array_tree_fingerprint((indices, lengths, valid)),
            }
        )

    def forward(self, attenuation: ArrayLike, /) -> XRayProjectionResult:
        values = jnp.asarray(attenuation)
        if values.shape != self.volume_shape:
            raise ValueError(f"attenuation must have shape {self.volume_shape}.")
        flat = values.reshape((-1,))
        safe = jnp.maximum(self.voxel_indices, 0)
        contributions = jnp.where(
            self.segment_valid, flat[safe] * self.segment_lengths, 0.0
        )
        projection = jnp.sum(contributions, axis=-1).reshape(self.projection_shape)
        finite = jnp.all(jnp.isfinite(projection))
        path_length = jnp.sum(
            jnp.where(self.segment_valid, self.segment_lengths, 0.0), axis=-1
        ).reshape(self.projection_shape)
        route_complete = jnp.asarray(True)
        return XRayProjectionResult(
            projection,
            XRayTransformEvidence(
                route_complete,
                path_length,
                finite,
                finite & route_complete,
                self.operator_id,
            ),
        )

    def transpose(self, projection: ArrayLike, /) -> Array:
        values = jnp.asarray(projection)
        if values.shape != self.projection_shape:
            raise ValueError(f"projection must have shape {self.projection_shape}.")
        repeated = values.reshape((-1, 1)) * self.segment_lengths
        indices = jnp.maximum(self.voxel_indices, 0).reshape((-1,))
        payload = jnp.where(self.segment_valid, repeated, 0.0).reshape((-1,))
        return (
            jnp.zeros((int(np.prod(self.volume_shape)),), dtype=payload.dtype)
            .at[indices]
            .add(payload)
            .reshape(self.volume_shape)
        )


def _siddon_routes(
    rays: RaySampleSupport,
    shape: tuple[int, int, int],
    origin: np.ndarray,
    spacing: np.ndarray,
):
    origins = np.asarray(rays.origins)
    directions = np.asarray(rays.directions)
    active = np.asarray(rays.active_mask)
    near = np.asarray(rays.near)
    far = np.asarray(rays.far)
    maximum = int(sum(shape) + 1)
    indices = np.full((origins.shape[0], maximum), -1, dtype=np.int32)
    lengths = np.zeros((origins.shape[0], maximum), dtype=float)
    valid = np.zeros((origins.shape[0], maximum), dtype=bool)
    bounds_min = origin
    bounds_max = origin + spacing * np.asarray(shape)
    for ray, (position, direction) in enumerate(zip(origins, directions, strict=True)):
        t_lower, t_upper = -np.inf, np.inf
        for axis in range(3):
            if abs(direction[axis]) <= np.finfo(float).eps:
                if position[axis] < bounds_min[axis] or position[axis] > bounds_max[axis]:
                    t_lower, t_upper = 1.0, 0.0
                    break
            else:
                candidates = (
                    (bounds_min[axis] - position[axis]) / direction[axis],
                    (bounds_max[axis] - position[axis]) / direction[axis],
                )
                t_lower = max(t_lower, min(candidates))
                t_upper = min(t_upper, max(candidates))
        t_lower = max(t_lower, float(near[ray]))
        t_upper = min(t_upper, float(far[ray]))
        if not active[ray] or t_upper <= max(t_lower, 0.0):
            continue
        t_lower = max(t_lower, 0.0)
        crossings = [t_lower, t_upper]
        for axis in range(3):
            if abs(direction[axis]) > np.finfo(float).eps:
                planes = origin[axis] + spacing[axis] * np.arange(1, shape[axis])
                times = (planes - position[axis]) / direction[axis]
                crossings.extend(times[(times > t_lower) & (times < t_upper)].tolist())
        crossings = sorted(set(crossings))
        for segment, (left, right) in enumerate(
            zip(crossings[:-1], crossings[1:], strict=True)
        ):
            midpoint = position + 0.5 * (left + right) * direction
            cell = np.floor((midpoint - origin) / spacing).astype(int)
            if np.all((cell >= 0) & (cell < np.asarray(shape))):
                indices[ray, segment] = np.ravel_multi_index(tuple(cell), shape)
                lengths[ray, segment] = right - left
                valid[ray, segment] = True
    return indices, lengths, valid


class TetrahedralXRayTransformPlan(StrictModule, NonTrainableState):
    cell_indices: Array
    segment_lengths: Array
    segment_valid: Array
    projection_shape: tuple[int, ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self, support: ProjectionSupport, vertices: ArrayLike, tetrahedra: ArrayLike, /
    ):
        vertices_ = np.asarray(vertices, dtype=float)
        cells_ = np.asarray(tetrahedra, dtype=np.int32)
        if (
            vertices_.ndim != 2
            or vertices_.shape[1] != 3
            or cells_.ndim != 2
            or cells_.shape[1] != 4
        ):
            raise ValueError("vertices and tetrahedra require shapes (V,3) and (C,4).")
        cells = vertices_[cells_]
        inverse = np.linalg.inv(
            np.stack(
                (
                    cells[:, 1] - cells[:, 0],
                    cells[:, 2] - cells[:, 0],
                    cells[:, 3] - cells[:, 0],
                ),
                axis=-1,
            )
        )
        origins = np.asarray(support.rays.origins)
        directions = np.asarray(support.rays.directions)
        near = np.asarray(support.rays.near)
        far = np.asarray(support.rays.far)
        indices = np.tile(
            np.arange(cells_.shape[0], dtype=np.int32), (origins.shape[0], 1)
        )
        lengths = np.zeros(indices.shape)
        valid = np.zeros(indices.shape, dtype=bool)
        for ray, (origin, direction) in enumerate(zip(origins, directions, strict=True)):
            for cell in range(cells_.shape[0]):
                base = inverse[cell] @ (origin - cells[cell, 0])
                slope = inverse[cell] @ direction
                coefficients = np.concatenate((np.asarray((1.0 - np.sum(base),)), base))
                derivatives = np.concatenate((np.asarray((-np.sum(slope),)), slope))
                lower, upper = float(near[ray]), float(far[ray])
                for coefficient, derivative in zip(
                    coefficients, derivatives, strict=True
                ):
                    if abs(derivative) <= np.finfo(float).eps:
                        if coefficient < 0.0:
                            lower, upper = 1.0, 0.0
                            break
                    elif derivative > 0.0:
                        lower = max(lower, -coefficient / derivative)
                    else:
                        upper = min(upper, -coefficient / derivative)
                if upper > max(lower, 0.0):
                    lengths[ray, cell] = upper - max(lower, 0.0)
                    valid[ray, cell] = True
        self.cell_indices = jnp.asarray(indices)
        self.segment_lengths = jnp.asarray(lengths)
        self.segment_valid = jnp.asarray(valid)
        self.projection_shape = support.projection_shape
        self.cell_count = cells_.shape[0]
        self.operator_id = canonical_fingerprint(
            {
                "kind": "tetrahedral-xray-transform",
                "support": support.support_id,
                "vertices": array_tree_fingerprint(vertices_),
                "cells": array_tree_fingerprint(cells_),
                "routes": array_tree_fingerprint((lengths, valid)),
            }
        )

    def forward(self, attenuation: ArrayLike, /) -> Array:
        values = jnp.asarray(attenuation)
        if values.shape != (self.cell_count,):
            raise ValueError("attenuation must contain one value per tetrahedron.")
        return jnp.sum(
            jnp.where(
                self.segment_valid, values[self.cell_indices] * self.segment_lengths, 0.0
            ),
            axis=-1,
        ).reshape(self.projection_shape)

    def transpose(self, projection: ArrayLike, /) -> Array:
        values = jnp.asarray(projection).reshape((-1, 1))
        payload = jnp.where(self.segment_valid, values * self.segment_lengths, 0.0)
        return (
            jnp.zeros((self.cell_count,), dtype=payload.dtype)
            .at[self.cell_indices.reshape((-1,))]
            .add(payload.reshape((-1,)))
        )


class BeerLambertResult(StrictModule, NonTrainableState):
    expected_signal: Array
    transmitted_signal: Array
    scatter_signal: Array
    saturated: Array
    finite: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class BeerLambertPlan:
    incident_signal: np.ndarray
    dark_signal: float = 0.0
    gain: float = 1.0
    saturation: float = np.inf

    def __post_init__(self) -> None:
        incident = np.array(self.incident_signal, dtype=float, copy=True)
        if (
            not np.all(np.isfinite(incident))
            or np.any(incident < 0.0)
            or self.gain <= 0.0
            or self.saturation <= 0.0
        ):
            raise ValueError(
                "Detector parameters must be finite/nonnegative with positive gain and saturation."
            )
        incident.setflags(write=False)
        object.__setattr__(self, "incident_signal", incident)

    def evaluate(
        self, line_integral: ArrayLike, /, *, scatter_signal: ArrayLike = 0.0
    ) -> BeerLambertResult:
        attenuation = jnp.asarray(line_integral)
        incident = jnp.asarray(self.incident_signal)
        scatter = jnp.broadcast_to(
            jnp.asarray(scatter_signal, dtype=attenuation.dtype), attenuation.shape
        )
        transmitted = incident * jnp.exp(-attenuation)
        expected = self.dark_signal + self.gain * (transmitted + scatter)
        saturated = expected > self.saturation
        output = jnp.minimum(expected, self.saturation)
        finite = (
            jnp.all(jnp.isfinite(output))
            & jnp.all(attenuation >= 0.0)
            & jnp.all(scatter >= 0.0)
        )
        return BeerLambertResult(output, transmitted, scatter, saturated, finite, finite)


class PolychromaticBeerLambertPlan(StrictModule, NonTrainableState):
    spectral_weights: Array
    incident_signal: Array

    def __init__(self, spectral_weights: ArrayLike, incident_signal: ArrayLike, /):
        weights = jnp.asarray(spectral_weights)
        signal = jnp.asarray(incident_signal)
        if weights.ndim != 1 or signal.shape != weights.shape:
            raise ValueError(
                "spectral_weights and incident_signal must share one spectral axis."
            )
        self.spectral_weights = weights / jnp.sum(weights)
        self.incident_signal = signal

    def evaluate(
        self, spectral_line_integrals: ArrayLike, /, *, scatter_signal: ArrayLike = 0.0
    ) -> Array:
        integrals = jnp.asarray(spectral_line_integrals)
        if integrals.shape[-1] != self.spectral_weights.size:
            raise ValueError("The final line-integral axis must be spectral.")
        transmitted = jnp.sum(
            self.spectral_weights * self.incident_signal * jnp.exp(-integrals), axis=-1
        )
        return transmitted + jnp.asarray(scatter_signal)


class FilteredBackprojectionPlan(StrictModule, NonTrainableState):
    angles: Array
    detector_coordinates: Array
    output_x: Array
    output_y: Array

    def __init__(
        self,
        angles: ArrayLike,
        detector_coordinates: ArrayLike,
        output_x: ArrayLike,
        output_y: ArrayLike,
        /,
    ):
        self.angles = jnp.asarray(angles)
        self.detector_coordinates = jnp.asarray(detector_coordinates)
        self.output_x = jnp.asarray(output_x)
        self.output_y = jnp.asarray(output_y)
        if (
            self.angles.ndim != 1
            or self.detector_coordinates.ndim != 1
            or self.output_x.ndim != 1
            or self.output_y.ndim != 1
        ):
            raise ValueError("FBP coordinates must be rank-one arrays.")

    def reconstruct(self, sinogram: ArrayLike, /) -> Array:
        values = jnp.asarray(sinogram)
        expected = (self.angles.size, self.detector_coordinates.size)
        if values.shape != expected:
            raise ValueError(f"sinogram must have shape {expected}.")
        spacing = jnp.mean(jnp.diff(self.detector_coordinates))
        frequency = jnp.fft.fftfreq(self.detector_coordinates.size, d=spacing)
        filtered = jnp.real(
            jnp.fft.ifft(
                jnp.fft.fft(values, axis=1) * jnp.abs(frequency)[None, :], axis=1
            )
        )
        yy, xx = jnp.meshgrid(self.output_y, self.output_x, indexing="ij")
        reconstruction = jnp.zeros_like(xx)
        for index in range(self.angles.size):
            coordinate = xx * jnp.cos(self.angles[index]) + yy * jnp.sin(
                self.angles[index]
            )
            sample = linear_interpolate(
                self.detector_coordinates,
                filtered[index],
                coordinate.reshape((-1,)),
                bounds="fill",
                fill_value=0.0,
            ).values.reshape(xx.shape)
            reconstruction = reconstruction + sample
        return np.pi * reconstruction / (2.0 * self.angles.size)


class IterativeCTResult(StrictModule, NonTrainableState):
    attenuation: Array
    residual_norms: Array
    finite: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class IterativeCTPlan:
    transform: VoxelXRayTransformPlan
    iteration_count: int
    nonnegative: bool = True
    l2_regularization: float = 0.0

    def solve(
        self, projections: ArrayLike, /, *, initial: ArrayLike | None = None
    ) -> IterativeCTResult:
        target = jnp.asarray(projections)
        x = (
            jnp.zeros(self.transform.volume_shape, dtype=target.dtype)
            if initial is None
            else jnp.asarray(initial)
        )
        if x.shape != self.transform.volume_shape:
            raise ValueError("initial has the wrong volume shape.")
        residual = target - self.transform.forward(x).values
        direction = self.transform.transpose(residual) - self.l2_regularization * x
        gamma = jnp.sum(direction * direction)

        def step(carry, _):
            x, residual, direction, gamma = carry
            projected = self.transform.forward(direction).values
            denominator = jnp.sum(
                projected * projected
            ) + self.l2_regularization * jnp.sum(direction * direction)
            alpha = gamma / jnp.maximum(denominator, jnp.finfo(x.dtype).tiny)
            candidate = x + alpha * direction
            candidate = jnp.maximum(candidate, 0.0) if self.nonnegative else candidate
            residual_new = target - self.transform.forward(candidate).values
            gradient = (
                self.transform.transpose(residual_new)
                - self.l2_regularization * candidate
            )
            gamma_new = jnp.sum(gradient * gradient)
            beta = gamma_new / jnp.maximum(gamma, jnp.finfo(x.dtype).tiny)
            direction_new = gradient + beta * direction
            return (candidate, residual_new, direction_new, gamma_new), jnp.sqrt(
                jnp.sum(residual_new * residual_new)
            )

        final, norms = jax.lax.scan(
            step,
            (x, residual, direction, gamma),
            xs=None,
            length=int(self.iteration_count),
        )
        finite = jnp.all(jnp.isfinite(final[0])) & jnp.all(jnp.isfinite(norms))
        return IterativeCTResult(final[0], norms, finite, finite)


__all__ = [
    "BeerLambertPlan",
    "BeerLambertResult",
    "FilteredBackprojectionPlan",
    "IterativeCTPlan",
    "IterativeCTResult",
    "PolychromaticBeerLambertPlan",
    "ProjectionAsset",
    "ProjectionSupport",
    "TetrahedralXRayTransformPlan",
    "VoxelXRayTransformPlan",
    "XRayProjectionResult",
    "XRayTransformEvidence",
]
