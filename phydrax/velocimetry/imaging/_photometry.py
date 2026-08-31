#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..camera._model import project_points
from ..camera._rig import CameraRig
from ._raster import GaussianRasterizer, GaussianRasterResult
from ._types import ImageGeometry2D


class PhotometryEvidence(StrictModule):
    """Pixelwise clipping, saturation, validity, and noise evidence."""

    valid: Array
    low_clipped: Array
    saturated: Array
    nonfinite_input: Array
    noise: Array
    valid_count: Array
    saturated_count: Array
    nonfinite_count: Array


class PhotometryResult(StrictModule):
    """Sensor signal and the ideal/noise terms which produced it."""

    signal: Array
    expected_signal: Array
    evidence: PhotometryEvidence
    stochastic: bool = eqx.field(static=True)
    successful: Array
    response_id: str = eqx.field(static=True)


class PhotometricResponse(StrictModule, NonTrainableState):
    """Explicit affine detector response with optional shot and read noise."""

    gain: float = eqx.field(static=True)
    black_level: float = eqx.field(static=True)
    saturation_level: float = eqx.field(static=True)
    shot_noise: bool = eqx.field(static=True)
    read_noise_std: float = eqx.field(static=True)
    response_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        gain: float = 1.0,
        black_level: float = 0.0,
        saturation_level: float = float("inf"),
        shot_noise: bool = False,
        read_noise_std: float = 0.0,
    ):
        gain_ = float(gain)
        black_ = float(black_level)
        saturation_ = float(saturation_level)
        read_std = float(read_noise_std)
        if not isfinite(gain_) or gain_ <= 0.0:
            raise ValueError("gain must be finite and positive.")
        if not isfinite(black_):
            raise ValueError("black_level must be finite.")
        if (
            saturation_ != float("inf") and not isfinite(saturation_)
        ) or saturation_ <= black_:
            raise ValueError(
                "saturation_level must be finite or positive infinity and exceed black_level."
            )
        if not isfinite(read_std) or read_std < 0.0:
            raise ValueError("read_noise_std must be finite and non-negative.")
        self.gain = gain_
        self.black_level = black_
        self.saturation_level = saturation_
        self.shot_noise = bool(shot_noise)
        self.read_noise_std = read_std
        self.response_id = canonical_fingerprint(
            {
                "kind": "particle-photometric-response",
                "gain": gain_,
                "black_level": black_,
                "saturation_level": (
                    saturation_ if isfinite(saturation_) else "unbounded"
                ),
                "shot_noise": bool(shot_noise),
                "read_noise_std": read_std,
            }
        )

    @property
    def stochastic(self) -> bool:
        return self.shot_noise or self.read_noise_std > 0.0


class ParticleImageFormation(StrictModule, NonTrainableState):
    """Calibrated camera projection followed by Gaussian image formation."""

    rasterizer: GaussianRasterizer
    response: PhotometricResponse
    formation_id: str = eqx.field(static=True)

    def __init__(
        self,
        rasterizer: GaussianRasterizer,
        response: PhotometricResponse | None = None,
        /,
    ):
        if not isinstance(rasterizer, GaussianRasterizer):
            raise TypeError("rasterizer must be GaussianRasterizer.")
        response_ = PhotometricResponse() if response is None else response
        if not isinstance(response_, PhotometricResponse):
            raise TypeError("response must be PhotometricResponse or None.")
        self.rasterizer = rasterizer
        self.response = response_
        self.formation_id = canonical_fingerprint(
            {
                "kind": "camera-particle-image-formation",
                "rasterizer_id": rasterizer.rasterizer_id,
                "response_id": response_.response_id,
            }
        )


class CameraStackRenderResult(StrictModule):
    """Camera-first projected particle images and forward-model evidence."""

    ideal_images: Array
    images: Array
    projection_pixels: Array
    projection_depth: Array
    projection_valid: Array
    projection_status: Array
    rasterizations: tuple[GaussianRasterResult, ...]
    photometry: PhotometryResult
    successful: Array
    formation_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


def apply_photometry(
    response: PhotometricResponse,
    irradiance: ArrayLike,
    *,
    key: PRNGKeyArray | None = None,
    valid_mask: ArrayLike | None = None,
) -> PhotometryResult:
    """Apply an explicit sensor response; stochastic response requires ``key``."""
    if not isinstance(response, PhotometricResponse):
        raise TypeError("response must be PhotometricResponse.")
    values = jnp.asarray(irradiance)
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    valid = (
        jnp.ones(values.shape, dtype=bool)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != values.shape:
        valid = jnp.broadcast_to(valid, values.shape)
    nonfinite = valid & ~jnp.isfinite(values)
    safe_irradiance = jnp.where(
        valid & jnp.isfinite(values), jnp.maximum(values, 0.0), 0.0
    )
    low_clipped = valid & jnp.isfinite(values) & (values < 0.0)
    expected_photoelectrons = response.gain * safe_irradiance
    if response.stochastic and key is None:
        raise ValueError("A PRNG key is required for stochastic photometry.")
    if response.stochastic:
        shot_key, read_key = jr.split(key)
        photoelectrons = (
            jr.poisson(shot_key, expected_photoelectrons).astype(values.dtype)
            if response.shot_noise
            else expected_photoelectrons
        )
        read_noise = response.read_noise_std * jr.normal(
            read_key, values.shape, dtype=values.dtype
        )
    else:
        photoelectrons = expected_photoelectrons
        read_noise = jnp.zeros_like(expected_photoelectrons)
    expected_signal = expected_photoelectrons + response.black_level
    unclipped = photoelectrons + response.black_level + read_noise
    saturated = valid & (unclipped >= response.saturation_level)
    signal = jnp.clip(unclipped, 0.0, response.saturation_level)
    signal = jnp.where(valid, signal, 0.0)
    noise = jnp.where(valid, unclipped - expected_signal, 0.0)
    evidence = PhotometryEvidence(
        valid,
        low_clipped,
        saturated,
        nonfinite,
        noise,
        jnp.sum(valid, dtype=jnp.int32),
        jnp.sum(saturated, dtype=jnp.int32),
        jnp.sum(nonfinite, dtype=jnp.int32),
    )
    return PhotometryResult(
        signal,
        expected_signal,
        evidence,
        response.stochastic,
        ~jnp.any(nonfinite),
        response.response_id,
    )


def render_camera_stack(
    formation: ParticleImageFormation,
    rig: CameraRig,
    geometry: ImageGeometry2D,
    positions_xyz: ArrayLike,
    amplitude: ArrayLike,
    sigma: ArrayLike,
    active: ArrayLike | None = None,
    *,
    key: PRNGKeyArray | None = None,
) -> CameraStackRenderResult:
    """Project a fixed particle support through a rig and form camera-first images."""
    if not isinstance(formation, ParticleImageFormation):
        raise TypeError("formation must be ParticleImageFormation.")
    if not isinstance(rig, CameraRig):
        raise TypeError("rig must be CameraRig.")
    if not isinstance(geometry, ImageGeometry2D):
        raise TypeError("geometry must be ImageGeometry2D.")
    if any(
        camera.intrinsics.image_shape is not None
        and camera.intrinsics.image_shape != geometry.image_shape
        for camera in rig.cameras
    ):
        raise ValueError("Every declared camera image shape must match geometry.")
    positions = jnp.asarray(positions_xyz)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_xyz must have shape (particle_capacity, 3).")
    if not jnp.issubdtype(positions.dtype, jnp.inexact):
        positions = positions.astype(float)
    capacity = int(positions.shape[0])
    active_ = (
        jnp.ones((capacity,), dtype=bool)
        if active is None
        else jnp.asarray(active, dtype=bool)
    )
    if active_.shape != (capacity,):
        raise ValueError("active must have shape (particle_capacity,).")

    projections = tuple(project_points(camera, positions) for camera in rig.cameras)
    rasterizations = tuple(
        formation.rasterizer.render(
            geometry,
            projection.pixels,
            amplitude,
            sigma,
            active_ & rig.camera_valid[index] & projection.valid,
        )
        for index, projection in enumerate(projections)
    )
    ideal = jnp.stack(tuple(result.image for result in rasterizations), axis=0)
    pixel_valid = jnp.broadcast_to(rig.camera_valid[:, None, None], ideal.shape)
    photometry = apply_photometry(
        formation.response,
        ideal,
        key=key,
        valid_mask=pixel_valid,
    )
    projection_pixels = jnp.stack(tuple(result.pixels for result in projections), axis=0)
    projection_depth = jnp.stack(tuple(result.depth for result in projections), axis=0)
    projection_valid = (
        jnp.stack(tuple(result.valid for result in projections), axis=0)
        & rig.camera_valid[:, None]
    )
    projection_status = jnp.stack(tuple(result.status for result in projections), axis=0)
    raster_success = jnp.stack(tuple(result.successful for result in rasterizations))
    return CameraStackRenderResult(
        ideal,
        photometry.signal,
        projection_pixels,
        projection_depth,
        projection_valid,
        projection_status,
        rasterizations,
        photometry,
        photometry.successful & jnp.all(raster_success),
        formation.formation_id,
        geometry.geometry_id,
    )


__all__ = [
    "CameraStackRenderResult",
    "ParticleImageFormation",
    "PhotometricResponse",
    "PhotometryEvidence",
    "PhotometryResult",
    "apply_photometry",
    "render_camera_stack",
]
