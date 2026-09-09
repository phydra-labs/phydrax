#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complex MRI acquisition support, matched encoding, and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from nufftax import nufft2d1, nufft2d2

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...measurement import MeasurementAsset, SampleTimeAxis


@dataclass(frozen=True, slots=True)
class KSpaceSupport:
    k_vectors: np.ndarray
    coil_ids: tuple[str, ...]
    trajectory_id: str
    frame_id: str
    sample_times: SampleTimeAxis | None = None
    sample_shape: tuple[int, int] = field(init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        vectors = np.array(self.k_vectors, dtype=float, copy=True)
        if (
            vectors.ndim != 2
            or vectors.shape[1] != 2
            or vectors.shape[0] < 1
            or not np.all(np.isfinite(vectors))
        ):
            raise ValueError(
                "k_vectors must be finite normalized-radian coordinates with shape (samples, 2)."
            )
        coils = tuple(str(value) for value in self.coil_ids)
        if (
            not coils
            or any(not value for value in coils)
            or len(coils) != len(set(coils))
        ):
            raise ValueError("coil_ids must be nonempty and unique.")
        if (
            self.sample_times is not None
            and self.sample_times.sample_count != vectors.shape[0]
        ):
            raise ValueError("sample_times must contain one time per k-space sample.")
        vectors.setflags(write=False)
        object.__setattr__(self, "k_vectors", vectors)
        object.__setattr__(self, "coil_ids", coils)
        object.__setattr__(self, "sample_shape", (vectors.shape[0], len(coils)))
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "kspace-support",
                    "vectors": array_tree_fingerprint(vectors),
                    "coils": list(coils),
                    "trajectory": self.trajectory_id,
                    "frame": self.frame_id,
                    "time": None
                    if self.sample_times is None
                    else self.sample_times.time_axis_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class KSpaceAsset:
    measurement: MeasurementAsset

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MeasurementAsset) or not isinstance(
            self.measurement.field.support, KSpaceSupport
        ):
            raise TypeError("KSpaceAsset requires a MeasurementAsset on KSpaceSupport.")
        if not np.issubdtype(self.measurement.field.values.dtype, np.complexfloating):
            raise TypeError("K-space measurements require complex storage.")


class CoilSensitivityField(StrictModule, NonTrainableState):
    values: Array
    coil_ids: tuple[str, ...] = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, coil_ids: tuple[str, ...], /, *, field_id: str):
        values_ = jnp.asarray(values)
        if (
            values_.ndim != 3
            or values_.shape[0] != len(coil_ids)
            or not jnp.issubdtype(values_.dtype, jnp.complexfloating)
        ):
            raise ValueError(
                "Coil sensitivities require complex shape (coils, rows, columns)."
            )
        self.values = values_
        self.coil_ids = tuple(coil_ids)
        self.field_id = canonical_fingerprint(
            {
                "kind": "coil-sensitivity-field",
                "name": field_id,
                "coils": list(coil_ids),
                "shape": list(values_.shape),
            }
        )


class CoilNoiseCovariance(StrictModule, NonTrainableState):
    covariance: Array
    whitening: Array
    coil_ids: tuple[str, ...] = eqx.field(static=True)
    covariance_id: str = eqx.field(static=True)

    def __init__(self, covariance: ArrayLike, coil_ids: tuple[str, ...], /):
        matrix = np.asarray(covariance)
        if matrix.shape != (len(coil_ids), len(coil_ids)) or not np.allclose(
            matrix, matrix.conj().T
        ):
            raise ValueError(
                "Coil covariance must be Hermitian with one row/column per coil."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("Coil covariance must be positive definite.")
        whitening = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T
        self.covariance = jnp.asarray(matrix)
        self.whitening = jnp.asarray(whitening)
        self.coil_ids = tuple(coil_ids)
        self.covariance_id = canonical_fingerprint(
            {
                "kind": "coil-noise-covariance",
                "coils": list(coil_ids),
                "matrix": array_tree_fingerprint(matrix),
            }
        )

    def whiten(self, values: ArrayLike, /) -> Array:
        data = jnp.asarray(values)
        return contract("cd,...d->...c", self.whitening, data)


class MRIEncodingEvidence(StrictModule, NonTrainableState):
    finite: Array
    adjoint_normalization: Array
    successful: Array
    operator_id: str = eqx.field(static=True)


class CartesianMRIEncodingPlan(StrictModule, NonTrainableState):
    coils: CoilSensitivityField
    sampling_mask: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self, coils: CoilSensitivityField, sampling_mask: ArrayLike | None = None, /
    ):
        shape = coils.values.shape[1:]
        mask = (
            jnp.ones(shape, dtype=bool)
            if sampling_mask is None
            else jnp.asarray(sampling_mask, dtype=bool)
        )
        if mask.shape != shape:
            raise ValueError("sampling_mask must match the image shape.")
        self.coils = coils
        self.sampling_mask = mask
        self.operator_id = canonical_fingerprint(
            {
                "kind": "cartesian-mri-encoding",
                "coils": coils.field_id,
                "mask_shape": list(mask.shape),
            }
        )

    def forward(self, image: ArrayLike, /) -> tuple[Array, MRIEncodingEvidence]:
        value = jnp.asarray(image)
        if value.shape != self.coils.values.shape[1:]:
            raise ValueError("image has the wrong shape.")
        coil_images = self.coils.values * value[None]
        kspace = jnp.fft.fft2(coil_images, norm="ortho")
        kspace = jnp.where(self.sampling_mask[None], kspace, 0.0)
        finite = jnp.all(jnp.isfinite(kspace))
        return kspace, MRIEncodingEvidence(
            finite, jnp.asarray(1.0), finite, self.operator_id
        )

    def adjoint(self, kspace: ArrayLike, /) -> Array:
        value = jnp.asarray(kspace)
        expected = self.coils.values.shape
        if value.shape != expected:
            raise ValueError(f"kspace must have shape {expected}.")
        coil_images = jnp.fft.ifft2(
            jnp.where(self.sampling_mask[None], value, 0.0), norm="ortho"
        )
        return jnp.sum(jnp.conj(self.coils.values) * coil_images, axis=0)


class NUFFTMRIEncodingPlan(StrictModule, NonTrainableState):
    coils: CoilSensitivityField
    support: KSpaceSupport = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        coils: CoilSensitivityField,
        support: KSpaceSupport,
        /,
        *,
        tolerance: float = 1.0e-6,
    ):
        if tuple(coils.coil_ids) != tuple(support.coil_ids):
            raise ValueError("Coil sensitivity and k-space coil identities differ.")
        self.coils = coils
        self.support = support
        self.tolerance = float(tolerance)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "nufft-mri-encoding",
                "coils": coils.field_id,
                "support": support.support_id,
                "tolerance": self.tolerance,
            }
        )

    def forward(self, image: ArrayLike, /) -> tuple[Array, MRIEncodingEvidence]:
        value = jnp.asarray(image)
        if value.shape != self.coils.values.shape[1:]:
            raise ValueError("image has the wrong shape.")
        coordinates = jnp.asarray(self.support.k_vectors)
        data = jax.vmap(
            lambda coil: nufft2d2(
                coordinates[:, 1], coordinates[:, 0], coil, eps=self.tolerance, isign=-1
            )
        )(self.coils.values * value[None])
        output = jnp.swapaxes(data, 0, 1)
        finite = jnp.all(jnp.isfinite(output))
        return output, MRIEncodingEvidence(
            finite, jnp.asarray(1.0), finite, self.operator_id
        )

    def adjoint(self, kspace: ArrayLike, /) -> Array:
        data = jnp.asarray(kspace)
        if data.shape != self.support.sample_shape:
            raise ValueError(f"kspace must have shape {self.support.sample_shape}.")
        coordinates = jnp.asarray(self.support.k_vectors)
        shape = self.coils.values.shape[1:]
        coil_images = jax.vmap(
            lambda coil: nufft2d1(
                coordinates[:, 1],
                coordinates[:, 0],
                coil,
                n_modes=(shape[1], shape[0]),
                eps=self.tolerance,
                isign=1,
            )
        )(jnp.swapaxes(data, 0, 1))
        return jnp.sum(jnp.conj(self.coils.values) * coil_images, axis=0)


class MRIReconstructionResult(StrictModule, NonTrainableState):
    image: Array
    residual_norms: Array
    finite: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CGSensePlan:
    encoding: CartesianMRIEncodingPlan | NUFFTMRIEncodingPlan
    iteration_count: int
    l2_regularization: float = 0.0

    def reconstruct(
        self, kspace: ArrayLike, /, *, initial: ArrayLike | None = None
    ) -> MRIReconstructionResult:
        data = jnp.asarray(kspace)
        shape = self.encoding.coils.values.shape[1:]
        x = (
            jnp.zeros(shape, dtype=data.dtype)
            if initial is None
            else jnp.asarray(initial)
        )
        right = self.encoding.adjoint(data)

        def normal(value):
            return (
                self.encoding.adjoint(self.encoding.forward(value)[0])
                + self.l2_regularization * value
            )

        residual = right - normal(x)
        direction = residual
        gamma = jnp.real(jnp.vdot(residual, residual))

        def step(carry, _):
            x, residual, direction, gamma = carry
            action = normal(direction)
            alpha = gamma / jnp.maximum(
                jnp.real(jnp.vdot(direction, action)), jnp.finfo(gamma.dtype).tiny
            )
            x_new = x + alpha * direction
            residual_new = residual - alpha * action
            gamma_new = jnp.real(jnp.vdot(residual_new, residual_new))
            beta = gamma_new / jnp.maximum(gamma, jnp.finfo(gamma.dtype).tiny)
            return (
                x_new,
                residual_new,
                residual_new + beta * direction,
                gamma_new,
            ), jnp.sqrt(gamma_new)

        final, norms = jax.lax.scan(
            step,
            (x, residual, direction, gamma),
            xs=None,
            length=int(self.iteration_count),
        )
        finite = jnp.all(jnp.isfinite(final[0])) & jnp.all(jnp.isfinite(norms))
        return MRIReconstructionResult(final[0], norms, finite, finite)


@dataclass(frozen=True, slots=True)
class RegularizedMRIPlan:
    base: CGSensePlan
    shrinkage: float
    outer_iterations: int = 4

    def reconstruct(self, kspace: ArrayLike, /) -> MRIReconstructionResult:
        image = jnp.zeros(
            self.base.encoding.coils.values.shape[1:], dtype=jnp.asarray(kspace).dtype
        )
        records = []
        for _ in range(int(self.outer_iterations)):
            result = self.base.reconstruct(kspace, initial=image)
            magnitude = jnp.abs(result.image)
            image = jnp.where(
                magnitude > 0.0,
                result.image * jnp.maximum(magnitude - self.shrinkage, 0.0) / magnitude,
                0.0,
            )
            records.append(result.residual_norms[-1])
        residuals = jnp.stack(records)
        finite = jnp.all(jnp.isfinite(image)) & jnp.all(jnp.isfinite(residuals))
        return MRIReconstructionResult(image, residuals, finite, finite)


class OffResonanceMRIEncodingPlan(StrictModule, NonTrainableState):
    base: NUFFTMRIEncodingPlan
    off_resonance_hz: Array
    translations: Array

    def __init__(
        self,
        base: NUFFTMRIEncodingPlan,
        off_resonance_hz: ArrayLike,
        /,
        *,
        translations: ArrayLike | None = None,
    ):
        field = jnp.asarray(off_resonance_hz)
        if (
            field.shape != base.coils.values.shape[1:]
            or base.support.sample_times is None
        ):
            raise ValueError(
                "Off-resonance encoding requires an image-shaped field and k-space sample times."
            )
        translation = (
            jnp.zeros((base.support.sample_shape[0], 2))
            if translations is None
            else jnp.asarray(translations)
        )
        if translation.shape != (base.support.sample_shape[0], 2):
            raise ValueError("translations must have shape (sample_count, 2).")
        self.base = base
        self.off_resonance_hz = field
        self.translations = translation

    def forward(self, image: ArrayLike, /) -> Array:
        value = jnp.asarray(image)
        coordinates = jnp.asarray(self.base.support.k_vectors)
        sample_times = self.base.support.sample_times
        assert sample_times is not None
        times = jnp.asarray(sample_times.sample_times)
        rows, columns = value.shape
        yy, xx = jnp.meshgrid(
            jnp.arange(rows) - rows / 2, jnp.arange(columns) - columns / 2, indexing="ij"
        )
        points = jnp.stack((yy.reshape((-1,)), xx.reshape((-1,))), axis=-1)
        coil_values = self.base.coils.values.reshape(
            (len(self.base.support.coil_ids), -1)
        )
        flat = value.reshape((-1,))

        def sample(k, time, translation):
            phase = jnp.exp(-2j * jnp.pi * self.off_resonance_hz.reshape((-1,)) * time)
            motion = jnp.exp(-1j * jnp.sum(k * translation))
            fourier = jnp.exp(-1j * (points @ k))
            return motion * jnp.sum(coil_values * (flat * phase * fourier)[None], axis=1)

        return jax.vmap(sample)(coordinates, times, self.translations)


@dataclass(frozen=True, slots=True)
class PhaseContrastMRIPlan:
    velocity_encoding: float

    def encode(self, magnitude: ArrayLike, velocity: ArrayLike, /) -> Array:
        magnitude_, velocity_ = jnp.asarray(magnitude), jnp.asarray(velocity)
        if magnitude_.shape != velocity_.shape:
            raise ValueError("magnitude and velocity must have the same shape.")
        return magnitude_ * jnp.exp(1j * jnp.pi * velocity_ / self.velocity_encoding)

    def decode(self, signal: ArrayLike, /) -> Array:
        return self.velocity_encoding * jnp.angle(jnp.asarray(signal)) / jnp.pi


class QuantitativeMRIPlan(StrictModule, NonTrainableState):
    repetition_time: float = eqx.field(static=True)
    echo_time: float = eqx.field(static=True)

    def __init__(self, repetition_time: float, echo_time: float, /):
        self.repetition_time = float(repetition_time)
        self.echo_time = float(echo_time)
        if self.repetition_time <= 0.0 or self.echo_time < 0.0:
            raise ValueError("MRI sequence times are outside their physical domain.")

    def signal(
        self,
        proton_density: ArrayLike,
        t1: ArrayLike,
        t2: ArrayLike,
        phase: ArrayLike = 0.0,
        /,
    ) -> Array:
        density, t1_, t2_ = jnp.asarray(proton_density), jnp.asarray(t1), jnp.asarray(t2)
        return (
            density
            * (1.0 - jnp.exp(-self.repetition_time / t1_))
            * jnp.exp(-self.echo_time / t2_)
            * jnp.exp(1j * jnp.asarray(phase))
        )


class BlochResult(StrictModule, NonTrainableState):
    magnetization: Array
    history: Array
    finite: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class BlochSequencePlan:
    time_step: float
    gyromagnetic_ratio: float
    equilibrium_magnetization: float = 1.0

    def simulate(
        self,
        initial: ArrayLike,
        effective_field: ArrayLike,
        t1: ArrayLike,
        t2: ArrayLike,
        /,
    ) -> BlochResult:
        magnetization = jnp.asarray(initial)
        field = jnp.asarray(effective_field)
        if magnetization.shape[-1] != 3 or field.ndim != 2 or field.shape[1] != 3:
            raise ValueError("Bloch state/field require final dimension three.")
        dt = jnp.asarray(self.time_step, dtype=magnetization.dtype)

        def step(value, applied):
            angle = self.gyromagnetic_ratio * jnp.sqrt(jnp.sum(applied * applied)) * dt
            axis = applied / jnp.maximum(
                jnp.sqrt(jnp.sum(applied * applied)), jnp.finfo(value.dtype).tiny
            )
            cosine, sine = jnp.cos(angle), jnp.sin(angle)
            rotated = (
                value * cosine
                + jnp.cross(axis, value) * sine
                + axis * jnp.sum(axis * value) * (1.0 - cosine)
            )
            relaxed = rotated.at[..., :2].set(
                rotated[..., :2] * jnp.exp(-dt / jnp.asarray(t2))
            )
            relaxed = relaxed.at[..., 2].set(
                self.equilibrium_magnetization
                + (rotated[..., 2] - self.equilibrium_magnetization)
                * jnp.exp(-dt / jnp.asarray(t1))
            )
            return relaxed, relaxed

        final, history = jax.lax.scan(step, magnetization, field)
        finite = jnp.all(jnp.isfinite(final)) & jnp.all(jnp.isfinite(history))
        return BlochResult(final, history, finite, finite)


__all__ = [
    "BlochResult",
    "BlochSequencePlan",
    "CartesianMRIEncodingPlan",
    "CGSensePlan",
    "CoilNoiseCovariance",
    "CoilSensitivityField",
    "KSpaceAsset",
    "KSpaceSupport",
    "MRIEncodingEvidence",
    "MRIReconstructionResult",
    "NUFFTMRIEncodingPlan",
    "OffResonanceMRIEncodingPlan",
    "PhaseContrastMRIPlan",
    "QuantitativeMRIPlan",
    "RegularizedMRIPlan",
]
