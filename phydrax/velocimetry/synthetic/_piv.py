#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..imaging import (
    DenseDisplacementField2D,
    image_coordinates,
    ImageGeometry2D,
    ImagePair2D,
)
from ..imaging._photometry import (
    apply_photometry,
    PhotometricResponse,
    PhotometryResult,
)
from ..imaging._raster import GaussianRasterizer, GaussianRasterResult
from ._common import PIVScenarioKind, SyntheticEvidence


def _finite_tuple(
    value: Sequence[float], length: int, /, *, name: str
) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values.")
    return result


class PIVScenarioPlan(StrictModule, NonTrainableState):
    """Finite deterministic recipe for a native two-frame PIV scenario."""

    kind: PIVScenarioKind = eqx.field(static=True)
    family_id: str = eqx.field(static=True)
    image_shape: tuple[int, int] = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    particle_density: float = eqx.field(static=True)
    displacement_rc: tuple[float, float] = eqx.field(static=True)
    affine_gradient_rc: tuple[float, float, float, float] = eqx.field(static=True)
    shear: float = eqx.field(static=True)
    rotation_radians: float = eqx.field(static=True)
    spatial_amplitude_rc: tuple[float, float] = eqx.field(static=True)
    spatial_frequency_rc: tuple[float, float] = eqx.field(static=True)
    particle_diameter: float = eqx.field(static=True)
    particle_intensity: float = eqx.field(static=True)
    intensity_variation: float = eqx.field(static=True)
    read_noise_std: float = eqx.field(static=True)
    shot_noise: bool = eqx.field(static=True)
    saturation_level: float = eqx.field(static=True)
    dropout_probability: float = eqx.field(static=True)
    mask_fraction: float = eqx.field(static=True)
    boundary_fraction: float = eqx.field(static=True)
    delta_t: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: PIVScenarioKind | str = PIVScenarioKind.TRANSLATION,
        /,
        *,
        family_id: str | None = None,
        image_shape: Sequence[int] = (64, 64),
        particle_capacity: int = 128,
        particle_density: float = 0.01,
        displacement_rc: Sequence[float] = (1.0, 2.0),
        affine_gradient_rc: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        shear: float = 0.02,
        rotation_radians: float = 0.04,
        spatial_amplitude_rc: Sequence[float] = (1.0, 1.0),
        spatial_frequency_rc: Sequence[float] = (2.0, 2.0),
        particle_diameter: float = 2.0,
        particle_intensity: float = 1.0,
        intensity_variation: float = 0.2,
        read_noise_std: float = 0.0,
        shot_noise: bool = False,
        saturation_level: float = 1.0e12,
        dropout_probability: float = 0.0,
        mask_fraction: float = 0.0,
        boundary_fraction: float = 0.0,
        delta_t: float = 1.0,
        seed: int = 0,
    ):
        kind_ = PIVScenarioKind(kind)
        family = kind_.value if family_id is None else str(family_id)
        shape = tuple(int(item) for item in image_shape)
        if len(shape) != 2 or any(item < 4 for item in shape):
            raise ValueError(
                "image_shape must contain two dimensions of at least four pixels."
            )
        capacity = int(particle_capacity)
        density = float(particle_density)
        diameter = float(particle_diameter)
        intensity = float(particle_intensity)
        variation = float(intensity_variation)
        read_std = float(read_noise_std)
        saturation = float(saturation_level)
        dropout = float(dropout_probability)
        mask = float(mask_fraction)
        boundary = float(boundary_fraction)
        delta_t_ = float(delta_t)
        scalar_values = (
            density,
            diameter,
            intensity,
            variation,
            read_std,
            saturation,
            dropout,
            mask,
            boundary,
            delta_t_,
            float(shear),
            float(rotation_radians),
        )
        if not family:
            raise ValueError("family_id must be non-empty.")
        if capacity < 1:
            raise ValueError("particle_capacity must be positive.")
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("PIV scenario scalar parameters must be finite.")
        if density <= 0.0:
            raise ValueError("particle_density must be positive.")
        requested_count = max(1, int(round(density * shape[0] * shape[1])))
        if requested_count > capacity:
            raise ValueError(
                "particle_capacity is insufficient for the requested particle_density."
            )
        if diameter <= 0.0 or intensity <= 0.0 or delta_t_ <= 0.0:
            raise ValueError(
                "Particle diameter, intensity, and delta_t must be positive."
            )
        if not 0.0 <= variation < 1.0:
            raise ValueError("intensity_variation must lie in [0, 1).")
        if read_std < 0.0:
            raise ValueError("read_noise_std must be non-negative.")
        if saturation <= 0.0:
            raise ValueError("saturation_level must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout_probability must lie in [0, 1].")
        if not 0.0 <= mask < 1.0:
            raise ValueError("mask_fraction must lie in [0, 1).")
        if not 0.0 <= boundary <= 1.0:
            raise ValueError("boundary_fraction must lie in [0, 1].")

        displacement = _finite_tuple(displacement_rc, 2, name="displacement_rc")
        affine = _finite_tuple(affine_gradient_rc, 4, name="affine_gradient_rc")
        spatial_amplitude = _finite_tuple(
            spatial_amplitude_rc, 2, name="spatial_amplitude_rc"
        )
        spatial_frequency = _finite_tuple(
            spatial_frequency_rc, 2, name="spatial_frequency_rc"
        )
        if any(value < 0.0 for value in spatial_frequency):
            raise ValueError("spatial_frequency_rc must be non-negative.")

        self.kind = kind_
        self.family_id = family
        self.image_shape = shape
        self.particle_capacity = capacity
        self.particle_density = density
        self.displacement_rc = displacement
        self.affine_gradient_rc = affine
        self.shear = float(shear)
        self.rotation_radians = float(rotation_radians)
        self.spatial_amplitude_rc = spatial_amplitude
        self.spatial_frequency_rc = spatial_frequency
        self.particle_diameter = diameter
        self.particle_intensity = intensity
        self.intensity_variation = variation
        self.read_noise_std = read_std
        self.shot_noise = bool(shot_noise)
        self.saturation_level = saturation
        self.dropout_probability = dropout
        self.mask_fraction = mask
        self.boundary_fraction = boundary
        self.delta_t = delta_t_
        self.seed = int(seed)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "piv-synthetic-scenario-plan",
                "scenario_kind": kind_.value,
                "family": family,
                "image_shape": list(shape),
                "capacity": capacity,
                "density": density,
                "motion": {
                    "translation_rc": list(displacement),
                    "affine_gradient_rc": list(affine),
                    "shear": float(shear),
                    "rotation_radians": float(rotation_radians),
                    "spatial_amplitude_rc": list(spatial_amplitude),
                    "spatial_frequency_rc": list(spatial_frequency),
                },
                "particles": {
                    "diameter": diameter,
                    "intensity": intensity,
                    "intensity_variation": variation,
                },
                "sensor": {
                    "read_noise_std": read_std,
                    "shot_noise": bool(shot_noise),
                    "saturation_level": saturation,
                    "dropout_probability": dropout,
                    "mask_fraction": mask,
                    "boundary_fraction": boundary,
                },
                "delta_t": delta_t_,
                "seed": int(seed),
                "coordinate_convention": "row-down-column-right",
            }
        )

    @property
    def particle_count(self) -> int:
        return max(
            1,
            int(round(self.particle_density * self.image_shape[0] * self.image_shape[1])),
        )


class PIVSyntheticCase(StrictModule, NonTrainableState):
    """Rendered PIV pair and dense truth with no particle identity surface.

    Particle support is deliberately represented only by padded positions and masks.
    Those latent rasterization slots are not tracking identities and cannot leak into
    PTV/STB association outputs.
    """

    image_pair: ImagePair2D
    truth: DenseDisplacementField2D
    first_positions_rc: Array
    second_positions_rc: Array
    first_active: Array
    second_active: Array
    first_rasterization: GaussianRasterResult
    second_rasterization: GaussianRasterResult
    first_photometry: PhotometryResult
    second_photometry: PhotometryResult
    evidence: SyntheticEvidence
    family_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    scenario_id: str = eqx.field(static=True)

    def __init__(
        self,
        image_pair: ImagePair2D,
        truth: DenseDisplacementField2D,
        first_positions_rc: Array,
        second_positions_rc: Array,
        first_active: Array,
        second_active: Array,
        first_rasterization: GaussianRasterResult,
        second_rasterization: GaussianRasterResult,
        first_photometry: PhotometryResult,
        second_photometry: PhotometryResult,
        evidence: SyntheticEvidence,
        /,
        *,
        family_id: str,
        plan_id: str,
        scenario_id: str,
    ):
        capacity = first_positions_rc.shape[0]
        if first_positions_rc.shape != (capacity, 2) or second_positions_rc.shape != (
            capacity,
            2,
        ):
            raise ValueError("PIV particle positions must have shape (capacity, 2).")
        if first_active.shape != (capacity,) or second_active.shape != (capacity,):
            raise ValueError("PIV particle masks must have shape (capacity,).")
        if not isinstance(image_pair, ImagePair2D) or not isinstance(
            truth, DenseDisplacementField2D
        ):
            raise TypeError("PIV cases require native image-pair and displacement types.")
        if not isinstance(first_rasterization, GaussianRasterResult) or not isinstance(
            second_rasterization, GaussianRasterResult
        ):
            raise TypeError("PIV rasterization evidence must use GaussianRasterResult.")
        if not isinstance(first_photometry, PhotometryResult) or not isinstance(
            second_photometry, PhotometryResult
        ):
            raise TypeError("PIV sensor evidence must use PhotometryResult.")
        if truth.geometry_id != image_pair.geometry.geometry_id:
            raise ValueError("PIV truth and image pair must share one geometry.")
        if (
            first_rasterization.geometry_id != image_pair.geometry.geometry_id
            or second_rasterization.geometry_id != image_pair.geometry.geometry_id
        ):
            raise ValueError("PIV rasterization evidence must share the image geometry.")
        if not isinstance(evidence, SyntheticEvidence):
            raise TypeError("evidence must be SyntheticEvidence.")
        identifiers = (str(family_id), str(plan_id), str(scenario_id))
        if any(not value for value in identifiers):
            raise ValueError("PIV case identifiers must be non-empty.")
        self.image_pair = image_pair
        self.truth = truth
        self.first_positions_rc = jnp.asarray(first_positions_rc)
        self.second_positions_rc = jnp.asarray(second_positions_rc)
        self.first_active = jnp.asarray(first_active, dtype=bool)
        self.second_active = jnp.asarray(second_active, dtype=bool)
        self.first_rasterization = first_rasterization
        self.second_rasterization = second_rasterization
        self.first_photometry = first_photometry
        self.second_photometry = second_photometry
        self.evidence = evidence
        self.family_id, self.plan_id, self.scenario_id = identifiers


def _piv_displacement(plan: PIVScenarioPlan, positions_rc: Array, /) -> Array:
    shape = jnp.asarray(plan.image_shape, dtype=positions_rc.dtype)
    center = 0.5 * (shape - 1.0)
    centered = positions_rc - center
    translation = jnp.asarray(plan.displacement_rc, dtype=positions_rc.dtype)
    if plan.kind is PIVScenarioKind.NO_MOTION:
        return jnp.zeros_like(positions_rc)
    if plan.kind is PIVScenarioKind.TRANSLATION:
        return jnp.broadcast_to(translation, positions_rc.shape)
    if plan.kind is PIVScenarioKind.AFFINE:
        gradient = jnp.asarray(plan.affine_gradient_rc, dtype=positions_rc.dtype).reshape(
            (2, 2)
        )
        return centered @ gradient.T + translation
    if plan.kind is PIVScenarioKind.SHEAR:
        shear = jnp.asarray(plan.shear, dtype=positions_rc.dtype)
        return (
            jnp.stack(
                (jnp.zeros_like(centered[..., 0]), shear * centered[..., 0]), axis=-1
            )
            + translation
        )
    if plan.kind is PIVScenarioKind.ROTATION:
        angle = jnp.asarray(plan.rotation_radians, dtype=positions_rc.dtype)
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        rotation = jnp.stack((jnp.stack((cosine, -sine)), jnp.stack((sine, cosine))))
        return (
            centered @ (rotation - jnp.eye(2, dtype=positions_rc.dtype)).T + translation
        )
    normalized = positions_rc / jnp.maximum(shape - 1.0, 1.0)
    amplitude = jnp.asarray(plan.spatial_amplitude_rc, dtype=positions_rc.dtype)
    frequency = jnp.asarray(plan.spatial_frequency_rc, dtype=positions_rc.dtype)
    return amplitude * jnp.sin(2.0 * jnp.pi * frequency * normalized) + translation


def _sensor_mask(plan: PIVScenarioPlan, coordinates_rc: Array, /) -> Array:
    if plan.mask_fraction == 0.0:
        return jnp.ones(plan.image_shape, dtype=bool)
    side_fraction = math.sqrt(plan.mask_fraction)
    center = 0.5 * (jnp.asarray(plan.image_shape, dtype=float) - 1.0)
    half_extent = 0.5 * side_fraction * jnp.asarray(plan.image_shape, dtype=float)
    masked = jnp.all(jnp.abs(coordinates_rc - center) <= half_extent, axis=-1)
    return ~masked


def _particle_positions(plan: PIVScenarioPlan, key: Array, /) -> Array:
    height, width = plan.image_shape
    margin = max(0.5, 1.5 * plan.particle_diameter)
    lower = jnp.asarray((margin, margin), dtype=float)
    upper = jnp.asarray((height - 1.0 - margin, width - 1.0 - margin), dtype=float)
    positions = jr.uniform(
        key,
        (plan.particle_capacity, 2),
        minval=lower,
        maxval=jnp.maximum(upper, lower),
    )
    boundary_count = int(round(plan.boundary_fraction * plan.particle_count))
    if boundary_count == 0:
        return positions
    indices = jnp.arange(boundary_count, dtype=jnp.int32)
    side = indices % 4
    edge_offset = 0.25 * plan.particle_diameter
    boundary_row = jnp.where(
        side == 0,
        edge_offset,
        jnp.where(side == 1, height - 1.0 - edge_offset, positions[:boundary_count, 0]),
    )
    boundary_column = jnp.where(
        side == 2,
        edge_offset,
        jnp.where(side == 3, width - 1.0 - edge_offset, positions[:boundary_count, 1]),
    )
    return positions.at[:boundary_count].set(
        jnp.stack((boundary_row, boundary_column), axis=-1)
    )


def generate_piv_case(
    plan: PIVScenarioPlan,
    /,
    *,
    rasterizer: GaussianRasterizer | None = None,
) -> PIVSyntheticCase:
    """Materialize one deterministic native PIV image pair and dense truth field."""
    if not isinstance(plan, PIVScenarioPlan):
        raise TypeError("plan must be a PIVScenarioPlan.")
    rasterizer_ = GaussianRasterizer() if rasterizer is None else rasterizer
    if not isinstance(rasterizer_, GaussianRasterizer):
        raise TypeError("rasterizer must be GaussianRasterizer or None.")
    geometry = ImageGeometry2D(plan.image_shape)
    response = PhotometricResponse(
        saturation_level=plan.saturation_level,
        shot_noise=plan.shot_noise,
        read_noise_std=plan.read_noise_std,
    )
    scenario_id = canonical_fingerprint(
        {
            "kind": "piv-synthetic-case",
            "plan": plan.plan_id,
            "geometry": geometry.geometry_id,
            "rasterizer": rasterizer_.rasterizer_id,
            "response": response.response_id,
        }
    )
    keys = jr.split(jr.key(plan.seed), 5)
    first_positions = _particle_positions(plan, keys[0])
    displacement = _piv_displacement(plan, first_positions)
    second_positions = first_positions + displacement
    slots = jnp.arange(plan.particle_capacity)
    first_active = slots < plan.particle_count
    dropout_draw = jr.uniform(keys[1], (plan.particle_capacity,))
    second_active = first_active & (dropout_draw >= plan.dropout_probability)
    amplitudes = plan.particle_intensity * (
        1.0
        + plan.intensity_variation
        * (2.0 * jr.uniform(keys[2], (plan.particle_capacity,)) - 1.0)
    )
    sigma = plan.particle_diameter / 2.3548200450309493
    first_raster = rasterizer_.render(
        geometry, first_positions, amplitudes, sigma, first_active
    )
    second_raster = rasterizer_.render(
        geometry, second_positions, amplitudes, sigma, second_active
    )
    coordinates = image_coordinates(geometry)
    sensor_mask = _sensor_mask(plan, coordinates)
    first_photometry = apply_photometry(
        response, first_raster.image, key=keys[3], valid_mask=sensor_mask
    )
    second_photometry = apply_photometry(
        response, second_raster.image, key=keys[4], valid_mask=sensor_mask
    )
    image_pair = ImagePair2D(
        first_photometry.signal,
        second_photometry.signal,
        geometry,
        first_mask=first_photometry.evidence.valid,
        second_mask=second_photometry.evidence.valid,
        delta_t=plan.delta_t,
        pair_id=scenario_id,
        provenance=(plan.plan_id, rasterizer_.rasterizer_id, response.response_id),
    )
    dense_displacement = _piv_displacement(plan, coordinates)
    target = coordinates + dense_displacement
    target_inside = (
        (target[..., 0] >= 0.0)
        & (target[..., 0] <= plan.image_shape[0] - 1)
        & (target[..., 1] >= 0.0)
        & (target[..., 1] <= plan.image_shape[1] - 1)
    )
    truth_valid = sensor_mask & _sensor_mask(plan, target) & target_inside
    truth = DenseDisplacementField2D(
        coordinates,
        dense_displacement,
        truth_valid,
        geometry_id=geometry.geometry_id,
        field_id=canonical_fingerprint(
            {"kind": "piv-synthetic-truth", "scenario": scenario_id}
        ),
        provenance=(scenario_id, plan.kind.value, "row-down-column-right"),
    )
    finite = bool(
        jnp.all(jnp.isfinite(image_pair.first))
        & jnp.all(jnp.isfinite(image_pair.second))
        & jnp.all(jnp.isfinite(truth.displacement_rc))
    )
    raster_success = bool(first_raster.successful & second_raster.successful)
    boundary_truncated = bool(
        jnp.any(first_raster.evidence.truncated)
        | jnp.any(second_raster.evidence.truncated)
    )
    status = (
        f"{plan.kind.value}-raster-warning"
        if not raster_success
        else f"{plan.kind.value}-boundary-truncated"
        if boundary_truncated
        else plan.kind.value
    )
    evidence = SyntheticEvidence(
        plan.particle_capacity,
        plan.particle_count,
        finite=finite,
        status=status,
        source_id=scenario_id,
    )
    return PIVSyntheticCase(
        image_pair,
        truth,
        first_positions,
        second_positions,
        first_active,
        second_active,
        first_raster,
        second_raster,
        first_photometry,
        second_photometry,
        evidence,
        family_id=plan.family_id,
        plan_id=plan.plan_id,
        scenario_id=scenario_id,
    )


__all__ = ["PIVScenarioPlan", "PIVSyntheticCase", "generate_piv_case"]
