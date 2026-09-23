#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._photon_transport import PhotonTransportResult
from ..detector._core import SensitiveHitBank


_LIGHT_SPEED = 299792458.0


class PlanarXRayDetectorResult(StrictModule, NonTrainableState):
    image_by_scatter_class: Array
    standard_error_by_scatter_class: Array
    hit_position: Array
    pixel_index: Array
    hit: Array
    incident_energy: Array
    energy_residual: Array
    finite: Array
    successful: Array
    transport_history_ids: Array
    transport_plan_id: str = eqx.field(static=True)
    detector_id: str = eqx.field(static=True)


class PlanarXRayDetectorPlan(StrictModule, NonTrainableState):
    center: Array
    normal: Array
    horizontal_axis: Array
    vertical_axis: Array
    width: float = eqx.field(static=True)
    height: float = eqx.field(static=True)
    pixel_shape: tuple[int, int] = eqx.field(static=True)
    detector_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        normal: ArrayLike,
        horizontal_axis: ArrayLike,
        /,
        *,
        width: float,
        height: float,
        pixel_shape: tuple[int, int],
        detector_id: str,
    ):
        center_ = np.asarray(center, dtype=np.float64)
        normal_ = np.asarray(normal, dtype=np.float64)
        horizontal = np.asarray(horizontal_axis, dtype=np.float64)
        width_, height_ = float(width), float(height)
        pixels = tuple(pixel_shape)
        identifier = str(detector_id).strip()
        normal_norm = np.linalg.norm(normal_)
        if normal_norm > 0.0:
            normal_ = normal_ / normal_norm
        horizontal = horizontal - np.dot(horizontal, normal_) * normal_
        horizontal_norm = np.linalg.norm(horizontal)
        if horizontal_norm > 0.0:
            horizontal = horizontal / horizontal_norm
        if (
            center_.shape != (3,)
            or normal_.shape != (3,)
            or horizontal.shape != (3,)
            or np.any(~np.isfinite(center_))
            or np.any(~np.isfinite(normal_))
            or np.any(~np.isfinite(horizontal))
            or normal_norm <= 0.0
            or horizontal_norm <= 0.0
            or not np.isfinite(width_)
            or not np.isfinite(height_)
            or width_ <= 0.0
            or height_ <= 0.0
            or len(pixels) != 2
            or any(value <= 0 for value in pixels)
            or not identifier
        ):
            raise ValueError("Planar X-ray detector geometry or identity is invalid.")
        vertical = np.cross(normal_, horizontal)
        self.center = jnp.asarray(center_)
        self.normal = jnp.asarray(normal_)
        self.horizontal_axis = jnp.asarray(horizontal)
        self.vertical_axis = jnp.asarray(vertical)
        self.width = width_
        self.height = height_
        self.pixel_shape = pixels
        self.detector_id = canonical_fingerprint(
            {
                "kind": "planar-xray-detector",
                "declared_id": identifier,
                "arrays": array_tree_fingerprint(
                    {
                        "center": center_,
                        "normal": normal_,
                        "horizontal": horizontal,
                    }
                ),
                "width": width_,
                "height": height_,
                "pixel_shape": pixels,
            }
        )

    def score(self, transport: PhotonTransportResult, /) -> PlanarXRayDetectorResult:
        if not isinstance(transport, PhotonTransportResult):
            raise TypeError("transport must be PhotonTransportResult.")
        position = transport.terminal_position
        direction = transport.terminal_direction
        denominator = jnp.sum(direction * self.normal, axis=-1)
        distance = jnp.sum((self.center - position) * self.normal, axis=-1) / jnp.where(
            denominator != 0.0, denominator, 1.0
        )
        hit_position = position + distance[..., None] * direction
        offset = hit_position - self.center
        horizontal = jnp.sum(offset * self.horizontal_axis, axis=-1)
        vertical = jnp.sum(offset * self.vertical_axis, axis=-1)
        inside = (
            (denominator > 0.0)
            & (distance >= 0.0)
            & (jnp.abs(horizontal) <= 0.5 * self.width)
            & (jnp.abs(vertical) <= 0.5 * self.height)
            & (transport.escaped_energy > 0.0)
        )
        x = jnp.floor((horizontal / self.width + 0.5) * self.pixel_shape[1]).astype(
            jnp.int32
        )
        y = jnp.floor((vertical / self.height + 0.5) * self.pixel_shape[0]).astype(
            jnp.int32
        )
        x = jnp.clip(x, 0, self.pixel_shape[1] - 1)
        y = jnp.clip(y, 0, self.pixel_shape[0] - 1)
        pixel = jnp.stack((y, x), axis=-1)
        history_count = transport.history_ids.size
        per_history = jnp.zeros(
            (history_count, 4, *self.pixel_shape), dtype=transport.escaped_energy.dtype
        )
        rows = jnp.arange(history_count, dtype=jnp.int32)
        per_history = per_history.at[
            rows,
            transport.scatter_class,
            y,
            x,
        ].add(jnp.where(inside, transport.escaped_energy, 0.0))
        image = jnp.sum(per_history, axis=0)
        centered = per_history - jnp.mean(per_history, axis=0)
        standard_error = jnp.where(
            history_count > 1,
            jnp.sqrt(
                jnp.sum(centered**2, axis=0) / (history_count * (history_count - 1))
            ),
            jnp.zeros_like(image),
        )
        scored = jnp.sum(image)
        expected = jnp.sum(jnp.where(inside, transport.escaped_energy, 0.0))
        residual = scored - expected
        finite = (
            jnp.all(jnp.isfinite(image))
            & jnp.all(jnp.isfinite(hit_position))
            & jnp.isfinite(residual)
        )
        tolerance = (
            128.0 * jnp.finfo(image.dtype).eps * jnp.maximum(jnp.abs(expected), 1.0)
        )
        successful = finite & (jnp.abs(residual) <= tolerance)
        return PlanarXRayDetectorResult(
            image,
            standard_error,
            hit_position,
            pixel,
            inside,
            jnp.where(inside, transport.escaped_energy, 0.0),
            residual,
            finite,
            successful,
            transport.history_ids,
            transport.plan_id,
            self.detector_id,
        )

    def to_sensitive_hits(
        self,
        transport: PhotonTransportResult,
        result: PlanarXRayDetectorResult,
        /,
        *,
        conditions_id: str,
    ) -> SensitiveHitBank:
        if (
            result.detector_id != self.detector_id
            or result.transport_plan_id != transport.plan_id
        ):
            raise ValueError(
                "Detector result belongs to another detector or transport plan."
            )
        if result.transport_history_ids.shape != transport.history_ids.shape:
            raise ValueError("Detector result and transport history shapes differ.")
        history_ids = eqx.error_if(
            transport.history_ids,
            jnp.any(result.transport_history_ids != transport.history_ids),
            "Detector result belongs to different transport histories.",
        )
        count = history_ids.size
        flat_pixel = (
            result.pixel_index[:, 0] * self.pixel_shape[1] + result.pixel_index[:, 1]
        )
        travel = jnp.linalg.norm(
            result.hit_position - transport.terminal_position, axis=-1
        )
        return SensitiveHitBank(
            event_ids=history_ids,
            hit_ids=jnp.zeros((count, 1), dtype=jnp.int32),
            detector_element_ids=flat_pixel[:, None],
            channel_ids=flat_pixel[:, None],
            source_step_indices=jnp.zeros((count, 1), dtype=jnp.int32),
            positions=result.hit_position[:, None, :],
            times=(travel / _LIGHT_SPEED)[:, None],
            energies=result.incident_energy[:, None],
            active=result.hit[:, None],
            conditions_id=conditions_id,
        )


__all__ = [
    "PlanarXRayDetectorPlan",
    "PlanarXRayDetectorResult",
]
