#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""A deliberately limited, analytic late-gadolinium-enhancement forward model.

The implementation is a research observation operator, not a scanner sequence
simulator and not a clinical lesion classifier.  Its explicit stages are tissue
parameters, longitudinal relaxation, an analytic inversion-recovery signal,
periodic PSF and slice filtering, a fixed motion/resampling map, complex noise,
and magnitude formation.  Categorical lesion labels have a separate type and
never enter the continuous signal path.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....observation import CoordinateLayout, LinearObservationPlan
from ._metadata import MedicalImageAsset, SpatialAffine


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _finite_positive(value: float, name: str, /) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


def _finite_nonnegative(value: float, name: str, /) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar


def _floating_array(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=jnp.result_type(array.dtype, jnp.float32))


def _volume_shape(value: tuple[int, int, int], /) -> tuple[int, int, int]:
    shape = tuple(int(size) for size in value)
    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError(
            "volume_shape must contain three positive sizes in (z, y, x) order."
        )
    return shape


def _normalized_kernel(
    value: ArrayLike, dimensions: int, name: str, /
) -> tuple[Array, float]:
    host = np.asarray(value, dtype=float)
    if (
        host.ndim != dimensions
        or any(size % 2 != 1 for size in host.shape)
        or np.any(~np.isfinite(host))
        or np.any(host < 0.0)
    ):
        raise ValueError(
            f"{name} must be a finite, non-negative, odd-sized {dimensions}D array."
        )
    mass = float(np.sum(host))
    if mass <= 0.0:
        raise ValueError(f"{name} must have positive mass.")
    normalized = jax.lax.stop_gradient(jnp.asarray(host / mass))
    return normalized, abs(float(np.sum(host / mass)) - 1.0)


def _periodic_filter_3d(volume: Array, kernel: Array, /) -> Array:
    center = tuple(size // 2 for size in kernel.shape)
    result = jnp.zeros_like(volume)
    for iz in range(kernel.shape[0]):
        for iy in range(kernel.shape[1]):
            for ix in range(kernel.shape[2]):
                shift = (iz - center[0], iy - center[1], ix - center[2])
                result = result + kernel[iz, iy, ix] * jnp.roll(
                    volume, shift=shift, axis=(0, 1, 2)
                )
    return result


def _periodic_slice_filter(volume: Array, profile: Array, /) -> Array:
    center = profile.size // 2
    result = jnp.zeros_like(volume)
    for index in range(profile.size):
        result = result + profile[index] * jnp.roll(volume, shift=index - center, axis=0)
    return result


def _matching_affine(
    asset: MedicalImageAsset, affine: SpatialAffine, name: str, /
) -> None:
    if asset.spatial_affine.affine_id != affine.affine_id:
        raise ValueError(f"{name} and LGE plan spatial affines differ.")


class LGETissueState(StrictModule):
    """Continuous tissue fields used by the analytic LGE signal path."""

    native_t1_ms: Array
    contrast_concentration_mmol_per_l: Array
    proton_density_relative: Array
    phase_rad: Array
    spatial_affine: SpatialAffine = eqx.field(static=True)
    source_asset_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        native_t1_ms: ArrayLike,
        contrast_concentration_mmol_per_l: ArrayLike,
        proton_density_relative: ArrayLike,
        phase_rad: ArrayLike,
        spatial_affine: SpatialAffine,
        /,
        *,
        source_asset_ids: tuple[str, ...] = (),
    ):
        native_t1 = _floating_array(native_t1_ms)
        concentration = jnp.asarray(
            contrast_concentration_mmol_per_l, dtype=native_t1.dtype
        )
        density = jnp.asarray(proton_density_relative, dtype=native_t1.dtype)
        phase = jnp.asarray(phase_rad, dtype=native_t1.dtype)
        if (
            native_t1.ndim != 3
            or concentration.shape != native_t1.shape
            or density.shape != native_t1.shape
            or phase.shape != native_t1.shape
        ):
            raise ValueError(
                "Every LGE continuous tissue field must share one rank-three shape."
            )
        if not isinstance(spatial_affine, SpatialAffine):
            raise TypeError("spatial_affine must be a SpatialAffine.")
        identifiers = tuple(
            _identifier(value, "source asset ID") for value in source_asset_ids
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("source_asset_ids must be unique.")
        self.native_t1_ms = native_t1
        self.contrast_concentration_mmol_per_l = concentration
        self.proton_density_relative = density
        self.phase_rad = phase
        self.spatial_affine = spatial_affine
        self.source_asset_ids = identifiers

    @classmethod
    def from_assets(
        cls,
        native_t1: MedicalImageAsset,
        contrast_concentration: MedicalImageAsset,
        proton_density: MedicalImageAsset,
        phase: MedicalImageAsset,
        /,
    ) -> LGETissueState:
        """Admit normalized image assets after quantity/unit/affine checks."""

        expected = (
            (native_t1, "t1-map", "longitudinal_relaxation_time", "ms"),
            (
                contrast_concentration,
                "contrast-concentration-map",
                "species_concentration",
                "mM",
            ),
            (proton_density, "proton-density-map", "relative_proton_density", "1"),
            (phase, "phase-map", "phase", "rad"),
        )
        affine_id = native_t1.spatial_affine.affine_id
        for asset, modality, quantity, unit in expected:
            if (
                asset.modality != modality
                or asset.quantity != quantity
                or asset.unit != unit
            ):
                raise ValueError(
                    f"Asset {asset.asset_id!r} must be {modality}/{quantity}/{unit}."
                )
            if asset.spatial_affine.affine_id != affine_id:
                raise ValueError("LGE tissue-map assets must share one spatial affine.")
            if not np.all(asset.valid_mask):
                raise ValueError(
                    "LGE tissue-map assets must have complete valid support."
                )
        return cls(
            native_t1.values,
            contrast_concentration.values,
            proton_density.values,
            phase.values,
            native_t1.spatial_affine,
            source_asset_ids=tuple(asset.asset_id for asset, _, _, _ in expected),
        )


class CategoricalLesionMap(StrictModule, NonTrainableState):
    """Categorical lesion annotations kept outside the continuous LGE pipeline."""

    labels: Array
    spatial_affine: SpatialAffine = eqx.field(static=True)
    class_names: tuple[str, ...] = eqx.field(static=True)
    annotation_id: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: ArrayLike,
        spatial_affine: SpatialAffine,
        class_names: tuple[str, ...],
        /,
        *,
        annotation_id: str,
    ):
        host = np.asarray(labels)
        classes = tuple(_identifier(value, "class name") for value in class_names)
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("class_names must be non-empty and unique.")
        if host.ndim != 3 or not np.issubdtype(host.dtype, np.integer):
            raise ValueError(
                "Categorical lesion labels must be a rank-three integer array."
            )
        if np.any(host < 0) or np.any(host >= len(classes)):
            raise ValueError("Categorical lesion labels must index class_names.")
        identifier = _identifier(annotation_id, "annotation_id")
        self.labels = jax.lax.stop_gradient(jnp.asarray(host))
        self.spatial_affine = spatial_affine
        self.class_names = classes
        self.annotation_id = identifier
        self.map_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-categorical-lesion-map",
                "annotation_id": identifier,
                "spatial_affine": spatial_affine.affine_id,
                "class_names": list(classes),
                "labels": array_tree_fingerprint(self.labels),
            }
        )

    @classmethod
    def from_asset(
        cls,
        asset: MedicalImageAsset,
        class_names: tuple[str, ...],
        /,
        *,
        annotation_id: str,
    ) -> CategoricalLesionMap:
        if (
            asset.modality != "lge-lesion-label"
            or asset.quantity != "categorical_lesion"
            or asset.unit != "1"
        ):
            raise ValueError(
                "Categorical lesion assets must be lge-lesion-label/categorical_lesion/1."
            )
        if not np.all(asset.valid_mask):
            raise ValueError(
                "Categorical lesion assets must have complete valid support."
            )
        return cls(
            asset.values,
            asset.spatial_affine,
            class_names,
            annotation_id=annotation_id,
        )


class LGERelaxationEvidence(StrictModule):
    """Input-domain and longitudinal-relaxation validity evidence."""

    finite: Array
    positive_native_t1: Array
    nonnegative_concentration: Array
    nonnegative_proton_density: Array
    positive_postcontrast_t1: Array
    successful: Array


class LGEStageEvidence(StrictModule):
    """Evidence for every fixed stage of the limited LGE forward map."""

    relaxation: LGERelaxationEvidence
    psf_mass_error: Array
    slice_mass_error: Array
    motion_row_sum_error: Array
    motion_nonnegative: Array
    finite_analytic_signal: Array
    finite_after_psf: Array
    finite_after_slice: Array
    finite_after_motion: Array
    finite_noisy_complex: Array
    finite_magnitude: Array
    noise_standard_deviation: Array
    fixed_motion_map: Array
    successful: Array


class LGEObservationResult(StrictModule):
    """All continuous stages of one limited LGE magnitude observation."""

    postcontrast_t1_ms: Array
    analytic_signal: Array
    after_psf: Array
    after_slice_profile: Array
    after_motion: Array
    noisy_complex: Array
    magnitude: Array
    evidence: LGEStageEvidence
    spatial_affine: SpatialAffine = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class LGEObservationPlan(StrictModule, NonTrainableState):
    """Fixed-map tissue-to-magnitude LGE research observation plan.

    The analytic signal is a declared inversion-recovery approximation.  PSF
    and slice filters use periodic boundaries so their normalized constant-field
    limit is exact.  Motion is an authored row-stochastic linear resampling map;
    it is never estimated or changed inside differentiation.
    """

    point_spread_function: Array
    slice_profile: Array
    motion_response: LinearObservationPlan
    spatial_affine: SpatialAffine = eqx.field(static=True)
    volume_shape: tuple[int, int, int] = eqx.field(static=True)
    inversion_time_ms: float = eqx.field(static=True)
    repetition_time_ms: float = eqx.field(static=True)
    flip_angle_rad: float = eqx.field(static=True)
    inversion_efficiency: float = eqx.field(static=True)
    relaxivity_l_per_mmol_s: float = eqx.field(static=True)
    receiver_gain: float = eqx.field(static=True)
    noise_standard_deviation: float = eqx.field(static=True)
    psf_mass_error: float = eqx.field(static=True)
    slice_mass_error: float = eqx.field(static=True)
    motion_row_sum_error: float = eqx.field(static=True)
    motion_nonnegative: bool = eqx.field(static=True)
    acquisition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        spatial_affine: SpatialAffine,
        point_spread_function: ArrayLike,
        slice_profile: ArrayLike,
        motion_matrix: ArrayLike,
        /,
        *,
        inversion_time_ms: float,
        repetition_time_ms: float,
        flip_angle_rad: float,
        inversion_efficiency: float,
        relaxivity_l_per_mmol_s: float,
        receiver_gain: float = 1.0,
        noise_standard_deviation: float = 0.0,
        acquisition_id: str,
    ):
        shape = _volume_shape(volume_shape)
        psf, psf_error = _normalized_kernel(
            point_spread_function, 3, "point_spread_function"
        )
        profile, slice_error = _normalized_kernel(slice_profile, 1, "slice_profile")
        voxel_count = int(np.prod(shape))
        motion = np.asarray(motion_matrix, dtype=float)
        if motion.shape != (voxel_count, voxel_count) or np.any(~np.isfinite(motion)):
            raise ValueError("motion_matrix must be a finite square voxel-to-voxel map.")
        motion_nonnegative = bool(np.all(motion >= 0.0))
        row_error = float(np.max(np.abs(np.sum(motion, axis=1) - 1.0)))
        tolerance = 128.0 * np.finfo(motion.dtype).eps
        if not motion_nonnegative or row_error > tolerance:
            raise ValueError("motion_matrix must be non-negative and row-stochastic.")
        inversion_time = _finite_positive(inversion_time_ms, "inversion_time_ms")
        repetition_time = _finite_positive(repetition_time_ms, "repetition_time_ms")
        if repetition_time <= inversion_time:
            raise ValueError("repetition_time_ms must exceed inversion_time_ms.")
        flip_angle = _finite_positive(flip_angle_rad, "flip_angle_rad")
        if flip_angle >= math.pi:
            raise ValueError("flip_angle_rad must be less than pi.")
        efficiency = _finite_positive(inversion_efficiency, "inversion_efficiency")
        if efficiency > 1.0:
            raise ValueError("inversion_efficiency must not exceed one.")
        relaxivity = _finite_positive(relaxivity_l_per_mmol_s, "relaxivity_l_per_mmol_s")
        gain = _finite_positive(receiver_gain, "receiver_gain")
        noise = _finite_nonnegative(noise_standard_deviation, "noise_standard_deviation")
        identifier = _identifier(acquisition_id, "acquisition_id")
        labels = tuple(
            f"voxel:{z}:{y}:{x}"
            for z in range(shape[0])
            for y in range(shape[1])
            for x in range(shape[2])
        )
        layout = CoordinateLayout(labels)
        self.point_spread_function = psf
        self.slice_profile = profile
        self.motion_response = LinearObservationPlan(motion, layout, layout)
        self.spatial_affine = spatial_affine
        self.volume_shape = shape
        self.inversion_time_ms = inversion_time
        self.repetition_time_ms = repetition_time
        self.flip_angle_rad = flip_angle
        self.inversion_efficiency = efficiency
        self.relaxivity_l_per_mmol_s = relaxivity
        self.receiver_gain = gain
        self.noise_standard_deviation = noise
        self.psf_mass_error = psf_error
        self.slice_mass_error = slice_error
        self.motion_row_sum_error = row_error
        self.motion_nonnegative = motion_nonnegative
        self.acquisition_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-limited-lge-observation-plan",
                "acquisition_id": identifier,
                "shape": list(shape),
                "spatial_affine": spatial_affine.affine_id,
                "psf": array_tree_fingerprint(psf),
                "slice_profile": array_tree_fingerprint(profile),
                "motion": self.motion_response.plan_id,
                "inversion_time_ms": inversion_time,
                "repetition_time_ms": repetition_time,
                "flip_angle_rad": flip_angle,
                "inversion_efficiency": efficiency,
                "relaxivity_l_per_mmol_s": relaxivity,
                "receiver_gain": gain,
                "noise_standard_deviation": noise,
            }
        )

    def evaluate(
        self, tissue: LGETissueState, noise_key: PRNGKeyArray, /
    ) -> LGEObservationResult:
        if not isinstance(tissue, LGETissueState):
            raise TypeError("LGEObservationPlan requires an LGETissueState.")
        if tissue.spatial_affine.affine_id != self.spatial_affine.affine_id:
            raise ValueError(
                "LGE tissue and plan spatial affines must match exactly; "
                "convert the tissue maps explicitly before evaluation."
            )
        if tissue.native_t1_ms.shape != self.volume_shape:
            raise ValueError(
                "LGE tissue shape does not match the fixed plan volume shape."
            )
        native_t1 = tissue.native_t1_ms
        concentration = tissue.contrast_concentration_mmol_per_l
        density = tissue.proton_density_relative
        phase = tissue.phase_rad
        finite = (
            jnp.all(jnp.isfinite(native_t1))
            & jnp.all(jnp.isfinite(concentration))
            & jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(phase))
        )
        positive_native_t1 = jnp.all(native_t1 > 0.0)
        nonnegative_concentration = jnp.all(concentration >= 0.0)
        nonnegative_density = jnp.all(density >= 0.0)
        admissible = (
            finite & positive_native_t1 & nonnegative_concentration & nonnegative_density
        )
        safe_native_t1 = jnp.where(native_t1 > 0.0, native_t1, 1.0)
        safe_concentration = jnp.maximum(concentration, 0.0)
        safe_density = jnp.maximum(density, 0.0)
        longitudinal_rate_per_ms = 1.0 / safe_native_t1 + (
            self.relaxivity_l_per_mmol_s * safe_concentration / 1000.0
        )
        postcontrast_t1 = 1.0 / longitudinal_rate_per_ms
        positive_postcontrast_t1 = jnp.all(postcontrast_t1 > 0.0) & jnp.all(
            jnp.isfinite(postcontrast_t1)
        )
        relaxation_successful = admissible & positive_postcontrast_t1
        longitudinal_signal = safe_density * (
            1.0
            - 2.0
            * self.inversion_efficiency
            * jnp.exp(-self.inversion_time_ms / postcontrast_t1)
            + jnp.exp(-self.repetition_time_ms / postcontrast_t1)
        )
        analytic_signal = (
            self.receiver_gain
            * jnp.sin(self.flip_angle_rad)
            * longitudinal_signal
            * jnp.exp(1j * phase)
        )
        after_psf = _periodic_filter_3d(analytic_signal, self.point_spread_function)
        after_slice = _periodic_slice_filter(after_psf, self.slice_profile)
        after_motion = contract(
            "oi,i->o", self.motion_response.matrix, after_slice.reshape((-1,))
        ).reshape(self.volume_shape)
        component_standard_deviation = self.noise_standard_deviation / math.sqrt(2.0)
        noise_components = jr.normal(
            noise_key, (2,) + self.volume_shape, dtype=native_t1.dtype
        )
        complex_noise = component_standard_deviation * (
            noise_components[0] + 1j * noise_components[1]
        )
        noisy_complex = after_motion + complex_noise
        magnitude = jnp.abs(noisy_complex)
        finite_analytic = jnp.all(jnp.isfinite(analytic_signal))
        finite_psf = jnp.all(jnp.isfinite(after_psf))
        finite_slice = jnp.all(jnp.isfinite(after_slice))
        finite_motion = jnp.all(jnp.isfinite(after_motion))
        finite_noisy = jnp.all(jnp.isfinite(noisy_complex))
        finite_magnitude = jnp.all(jnp.isfinite(magnitude))
        dtype_tolerance = 256.0 * jnp.finfo(native_t1.dtype).eps
        stage_successful = (
            relaxation_successful
            & (self.psf_mass_error <= dtype_tolerance)
            & (self.slice_mass_error <= dtype_tolerance)
            & (self.motion_row_sum_error <= dtype_tolerance)
            & self.motion_nonnegative
            & finite_analytic
            & finite_psf
            & finite_slice
            & finite_motion
            & finite_noisy
            & finite_magnitude
        )
        relaxation_evidence = LGERelaxationEvidence(
            finite,
            positive_native_t1,
            nonnegative_concentration,
            nonnegative_density,
            positive_postcontrast_t1,
            relaxation_successful,
        )
        evidence = LGEStageEvidence(
            relaxation_evidence,
            jnp.asarray(self.psf_mass_error, dtype=native_t1.dtype),
            jnp.asarray(self.slice_mass_error, dtype=native_t1.dtype),
            jnp.asarray(self.motion_row_sum_error, dtype=native_t1.dtype),
            jnp.asarray(self.motion_nonnegative),
            finite_analytic,
            finite_psf,
            finite_slice,
            finite_motion,
            finite_noisy,
            finite_magnitude,
            jnp.asarray(self.noise_standard_deviation, dtype=native_t1.dtype),
            jnp.asarray(True),
            stage_successful,
        )

        def safe(value: Array) -> Array:
            return jnp.where(stage_successful, value, jnp.zeros_like(value))

        return LGEObservationResult(
            safe(postcontrast_t1),
            safe(analytic_signal),
            safe(after_psf),
            safe(after_slice),
            safe(after_motion),
            safe(noisy_complex),
            safe(magnitude),
            evidence,
            self.spatial_affine,
            self.plan_id,
        )

    def validate_asset_geometry(self, asset: MedicalImageAsset, /) -> None:
        """Validate an ingress asset against the plan without entering traced math."""

        _matching_affine(asset, self.spatial_affine, "Medical image asset")
        if asset.values.shape[-3:] != self.volume_shape:
            raise ValueError("Medical image asset shape does not match the LGE plan.")


__all__ = [
    "CategoricalLesionMap",
    "LGEObservationPlan",
    "LGEObservationResult",
    "LGERelaxationEvidence",
    "LGEStageEvidence",
    "LGETissueState",
]
