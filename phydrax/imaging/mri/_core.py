#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complex MRI acquisition support, matched encoding, and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._spectral._nonuniform_fourier import (
    NonuniformFourierPlan,
    PreparedNonuniformFourier,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...measurement import MeasurementAsset, SampleTimeAxis
from ...units import SECOND


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
        vectors = np.array(self.k_vectors, dtype=np.float64, copy=True)
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


def _coil_identities(coil_ids: tuple[str, ...], /) -> tuple[str, ...]:
    if isinstance(coil_ids, str):
        raise TypeError("coil_ids must be a tuple of identifiers.")
    identifiers = tuple(coil_ids)
    if (
        not identifiers
        or any(
            not isinstance(identifier, str)
            or not identifier
            or identifier != identifier.strip()
            for identifier in identifiers
        )
        or len(identifiers) != len(set(identifiers))
    ):
        raise ValueError("coil_ids must be canonical, nonempty, and unique.")
    return identifiers


class CoilSensitivityField(StrictModule, NonTrainableState):
    values: Array
    coil_ids: tuple[str, ...] = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, coil_ids: tuple[str, ...], /, *, field_id: str):
        identifiers = _coil_identities(coil_ids)
        values_ = np.asarray(values)
        if (
            values_.ndim != 3
            or values_.shape[0] != len(identifiers)
            or not np.issubdtype(values_.dtype, np.complexfloating)
            or not np.all(np.isfinite(values_))
        ):
            raise ValueError(
                "Coil sensitivities require finite complex shape (coils, rows, columns)."
            )
        self.values = jnp.asarray(values_)
        self.coil_ids = identifiers
        self.field_id = canonical_fingerprint(
            {
                "kind": "coil-sensitivity-field",
                "name": field_id,
                "coils": list(identifiers),
                "values": array_tree_fingerprint(values_),
            }
        )


class CoilNoiseCovariance(StrictModule, NonTrainableState):
    covariance: Array
    whitening: Array
    coil_ids: tuple[str, ...] = eqx.field(static=True)
    covariance_id: str = eqx.field(static=True)

    def __init__(self, covariance: ArrayLike, coil_ids: tuple[str, ...], /):
        identifiers = _coil_identities(coil_ids)
        matrix = np.asarray(covariance)
        if (
            matrix.shape != (len(identifiers), len(identifiers))
            or not np.issubdtype(matrix.dtype, np.number)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix, matrix.conj().T)
        ):
            raise ValueError(
                "Coil covariance must be finite Hermitian with one row/column per coil."
            )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
            raise ValueError("Coil covariance must be positive definite.")
        whitening = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T
        self.covariance = jnp.asarray(matrix)
        self.whitening = jnp.asarray(whitening)
        self.coil_ids = identifiers
        self.covariance_id = canonical_fingerprint(
            {
                "kind": "coil-noise-covariance",
                "coils": list(identifiers),
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
            np.ones(shape, dtype=np.bool_)
            if sampling_mask is None
            else np.asarray(sampling_mask, dtype=np.bool_)
        )
        if mask.shape != shape:
            raise ValueError("sampling_mask must match the image shape.")
        self.coils = coils
        self.sampling_mask = jnp.asarray(mask)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "cartesian-mri-encoding",
                "coils": coils.field_id,
                "mask": array_tree_fingerprint(mask),
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


class NonuniformMRIEncodingPlan(StrictModule, NonTrainableState):
    coils: CoilSensitivityField
    support: KSpaceSupport = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        coils: CoilSensitivityField,
        support: KSpaceSupport,
        /,
        *,
        chunk_size: int = 256,
    ):
        if tuple(coils.coil_ids) != tuple(support.coil_ids):
            raise ValueError("Coil sensitivity and k-space coil identities differ.")
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError("chunk_size must be positive.")
        self.coils = coils
        self.support = support
        self.chunk_size = chunk
        self.operator_id = canonical_fingerprint(
            {
                "kind": "nonuniform-mri-encoding",
                "coils": coils.field_id,
                "support": support.support_id,
                "chunk_size": chunk,
            }
        )

    def forward(self, image: ArrayLike, /) -> tuple[Array, MRIEncodingEvidence]:
        value = jnp.asarray(image)
        if value.shape != self.coils.values.shape[1:]:
            raise ValueError("image has the wrong shape.")
        coordinates = jnp.asarray(self.support.k_vectors)
        points = jnp.stack((coordinates[:, 1], coordinates[:, 0]), axis=-1)
        transform = PreparedNonuniformFourier(
            NonuniformFourierPlan(
                self.coils.values.shape[1:],
                2,
                sign=-1,
                centered=True,
                route="chunked",
                chunk_size=self.chunk_size,
            ),
            dtype=points.dtype,
        )
        data = jax.vmap(lambda coil: transform.type2(points, coil))(
            self.coils.values * value[None]
        )
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
        points = jnp.stack((coordinates[:, 1], coordinates[:, 0]), axis=-1)
        shape = self.coils.values.shape[1:]
        transform = PreparedNonuniformFourier(
            NonuniformFourierPlan(
                shape,
                1,
                sign=1,
                centered=True,
                route="chunked",
                chunk_size=self.chunk_size,
            ),
            dtype=points.dtype,
        )
        coil_images = jax.vmap(lambda coil: transform.type1(points, coil))(
            jnp.swapaxes(data, 0, 1)
        )
        return jnp.sum(jnp.conj(self.coils.values) * coil_images, axis=0)


class MRIReconstructionResult(StrictModule, NonTrainableState):
    image: Array
    residual_norms: Array
    finite: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CGSensePlan:
    encoding: CartesianMRIEncodingPlan | NonuniformMRIEncodingPlan
    iteration_count: int
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(
            self.encoding, (CartesianMRIEncodingPlan, NonuniformMRIEncodingPlan)
        ):
            raise TypeError("encoding must be a supported MRI encoding plan.")
        if isinstance(self.iteration_count, bool) or not isinstance(
            self.iteration_count, Integral
        ):
            raise TypeError("iteration_count must be an integer.")
        if self.iteration_count < 1:
            raise ValueError("iteration_count must be positive.")
        regularization = float(self.l2_regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("l2_regularization must be finite and nonnegative.")
        object.__setattr__(self, "iteration_count", int(self.iteration_count))
        object.__setattr__(self, "l2_regularization", regularization)

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

    def __post_init__(self) -> None:
        if not isinstance(self.base, CGSensePlan):
            raise TypeError("base must be CGSensePlan.")
        if isinstance(self.outer_iterations, bool) or not isinstance(
            self.outer_iterations, Integral
        ):
            raise TypeError("outer_iterations must be an integer.")
        shrinkage = float(self.shrinkage)
        if not np.isfinite(shrinkage) or shrinkage < 0.0 or self.outer_iterations < 1:
            raise ValueError(
                "shrinkage must be finite and nonnegative and outer_iterations positive."
            )
        object.__setattr__(self, "shrinkage", shrinkage)
        object.__setattr__(self, "outer_iterations", int(self.outer_iterations))

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
    base: NonuniformMRIEncodingPlan
    off_resonance_hz: Array
    translations: Array
    sample_times_seconds: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: NonuniformMRIEncodingPlan,
        off_resonance_hz: ArrayLike,
        /,
        *,
        translations: ArrayLike | None = None,
    ):
        field = np.asarray(off_resonance_hz)
        if (
            field.shape != base.coils.values.shape[1:]
            or base.support.sample_times is None
            or not np.all(np.isfinite(field))
        ):
            raise ValueError(
                "Off-resonance encoding requires a finite image-shaped field and k-space sample times."
            )
        translation = (
            np.zeros((base.support.sample_shape[0], 2), dtype=np.float64)
            if translations is None
            else np.asarray(translations)
        )
        if translation.shape != (base.support.sample_shape[0], 2) or not np.all(
            np.isfinite(translation)
        ):
            raise ValueError("translations must be finite with shape (sample_count, 2).")
        sample_times = base.support.sample_times
        assert sample_times is not None
        times_seconds = sample_times.values_in(SECOND)
        self.base = base
        self.off_resonance_hz = jnp.asarray(field)
        self.translations = jnp.asarray(translation)
        self.sample_times_seconds = jnp.asarray(times_seconds)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "off-resonance-mri-encoding",
                "base": base.operator_id,
                "field": array_tree_fingerprint(field),
                "translations": array_tree_fingerprint(translation),
                "sample_times_seconds": array_tree_fingerprint(times_seconds),
            }
        )

    def forward(self, image: ArrayLike, /) -> Array:
        value = jnp.asarray(image)
        coordinates = jnp.asarray(self.base.support.k_vectors)
        times = self.sample_times_seconds
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

    def __post_init__(self) -> None:
        encoding = float(self.velocity_encoding)
        if not np.isfinite(encoding) or encoding <= 0.0:
            raise ValueError("velocity_encoding must be finite and positive.")
        object.__setattr__(self, "velocity_encoding", encoding)

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
        repetition = float(repetition_time)
        echo = float(echo_time)
        if (
            not np.isfinite(repetition)
            or not np.isfinite(echo)
            or repetition <= 0.0
            or echo < 0.0
        ):
            raise ValueError("MRI sequence times are outside their physical domain.")
        self.repetition_time = repetition
        self.echo_time = echo

    def signal(
        self,
        proton_density: ArrayLike,
        t1: ArrayLike,
        t2: ArrayLike,
        phase: ArrayLike = 0.0,
        /,
    ) -> Array:
        density = jnp.asarray(proton_density)
        t1_ = jnp.asarray(t1)
        t2_ = jnp.asarray(t2)
        phase_ = jnp.asarray(phase)
        valid = (
            jnp.isfinite(density)
            & (density >= 0.0)
            & jnp.isfinite(t1_)
            & (t1_ > 0.0)
            & jnp.isfinite(t2_)
            & (t2_ > 0.0)
            & jnp.isfinite(phase_)
        )
        safe_t1 = jnp.where(valid, t1_, 1.0)
        safe_t2 = jnp.where(valid, t2_, 1.0)
        value = (
            density
            * (1.0 - jnp.exp(-self.repetition_time / safe_t1))
            * jnp.exp(-self.echo_time / safe_t2)
            * jnp.exp(1j * phase_)
        )
        invalid = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=value.dtype)
        return jnp.where(valid, value, invalid)


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

    def __post_init__(self) -> None:
        time_step = float(self.time_step)
        ratio = float(self.gyromagnetic_ratio)
        equilibrium = float(self.equilibrium_magnetization)
        if (
            not np.isfinite(time_step)
            or time_step <= 0.0
            or not np.isfinite(ratio)
            or not np.isfinite(equilibrium)
        ):
            raise ValueError(
                "Bloch sequence scalars must be finite with positive time_step."
            )
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "gyromagnetic_ratio", ratio)
        object.__setattr__(self, "equilibrium_magnetization", equilibrium)

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
    "NonuniformMRIEncodingPlan",
    "OffResonanceMRIEncodingPlan",
    "PhaseContrastMRIPlan",
    "QuantitativeMRIPlan",
    "RegularizedMRIPlan",
]
