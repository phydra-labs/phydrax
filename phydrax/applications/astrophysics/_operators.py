#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import CoordinateLayout, LinearObservationPlan, TheoryVector
from ._observation_status import AstrophysicsObservationStatus
from ._photometry import ObservationDataProvenance


class SpectralField(StrictModule):
    coordinate: Array
    values: Array
    provenance: ObservationDataProvenance
    coordinate_unit: str = eqx.field(static=True)
    value_unit: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: ArrayLike,
        values: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        coordinate_unit: str,
        value_unit: str,
        field_id: str,
    ):
        coordinate_ = jnp.asarray(coordinate)
        values_ = jnp.asarray(values)
        if coordinate_.ndim != 1 or values_.shape[-1:] != coordinate_.shape:
            raise ValueError("Spectral values must end in the coordinate axis.")
        self.coordinate = coordinate_
        self.values = values_
        self.provenance = provenance
        self.coordinate_unit = str(coordinate_unit)
        self.value_unit = str(value_unit)
        self.field_id = str(field_id)
        if not self.coordinate_unit or not self.value_unit or not self.field_id:
            raise ValueError("Spectral field identifiers and units must be non-empty.")


class BinnedResponseResult(StrictModule):
    predicted: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class BinnedResponsePlan(StrictModule, NonTrainableState):
    response: LinearObservationPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, matrix: ArrayLike, /, *, response_id: str):
        host = np.asarray(matrix, dtype=float)
        if host.ndim != 2 or np.any(~np.isfinite(host)) or np.any(host < 0.0):
            raise ValueError("Binned response must be a finite non-negative matrix.")
        identifier = str(response_id)
        if not identifier:
            raise ValueError("response_id must be non-empty.")
        source = CoordinateLayout(
            tuple(f"{identifier}:source:{index}" for index in range(host.shape[1]))
        )
        target = CoordinateLayout(
            tuple(f"{identifier}:target:{index}" for index in range(host.shape[0]))
        )
        self.response = LinearObservationPlan(host, source, target)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "astrophysics-binned-response-adapter",
                "response_id": identifier,
                "core": self.response.plan_id,
            }
        )

    @property
    def matrix(self) -> Array:
        return self.response.matrix

    def evaluate(self, integrated_source: ArrayLike, /) -> BinnedResponseResult:
        source = jnp.asarray(integrated_source)
        if source.shape[-1:] != (self.matrix.shape[1],):
            raise ValueError("Integrated source axis does not match response input.")
        flat = source.reshape((-1, source.shape[-1]))
        predicted = jax.vmap(
            lambda values: (
                self.response.apply(
                    TheoryVector(values, self.response.source, self.plan_id)
                ).values
            )
        )(flat).reshape(source.shape[:-1] + (self.matrix.shape[0],))
        valid = jnp.all(jnp.isfinite(source), axis=-1) & jnp.all(source >= 0.0, axis=-1)
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
        ).astype(jnp.int32)
        return BinnedResponseResult(
            jnp.where(valid[..., None], predicted, 0.0), valid, status, self.plan_id
        )


class ImageResponseResult(StrictModule):
    image: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class ImageResponsePlan(StrictModule, NonTrainableState):
    point_spread_function: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, point_spread_function: ArrayLike, /, *, response_id: str):
        host = np.asarray(point_spread_function, dtype=float)
        if host.ndim != 2 or np.any(~np.isfinite(host)) or np.any(host < 0.0):
            raise ValueError("Point-spread function must be a finite non-negative image.")
        total = float(np.sum(host))
        if total <= 0.0:
            raise ValueError("Point-spread function must have positive mass.")
        self.point_spread_function = jnp.asarray(host / total)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "image-response",
                "response_id": str(response_id),
                "shape": list(host.shape),
            }
        )

    def evaluate(self, image: ArrayLike, /) -> ImageResponseResult:
        values = jnp.asarray(image)
        if values.shape[-2:] != self.point_spread_function.shape:
            raise ValueError("Image and point-spread function shapes must match.")
        kernel = jnp.fft.fft2(jnp.fft.ifftshift(self.point_spread_function))
        convolved = jnp.fft.ifft2(jnp.fft.fft2(values) * kernel).real
        valid = jnp.all(jnp.isfinite(values), axis=(-2, -1))
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return ImageResponseResult(
            jnp.where(valid[..., None, None], convolved, 0.0), valid, status, self.plan_id
        )


class FrequencyDomainSignal(StrictModule):
    frequency: Array
    plus: Array
    cross: Array
    provenance: ObservationDataProvenance
    frequency_unit: str = eqx.field(static=True)
    strain_unit: str = eqx.field(static=True)
    signal_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: ArrayLike,
        plus: ArrayLike,
        cross: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        frequency_unit: str = "Hz",
        strain_unit: str = "strain",
        signal_id: str,
    ):
        frequency_ = jnp.asarray(frequency)
        plus_ = jnp.asarray(plus)
        cross_ = jnp.asarray(cross, dtype=plus_.dtype)
        if (
            frequency_.ndim != 1
            or plus_.shape != frequency_.shape
            or cross_.shape != frequency_.shape
        ):
            raise ValueError("Frequency signal arrays must be matching vectors.")
        self.frequency = frequency_
        self.plus = plus_
        self.cross = cross_
        self.provenance = provenance
        self.frequency_unit = str(frequency_unit)
        self.strain_unit = str(strain_unit)
        self.signal_id = str(signal_id)


class FrequencyResponseResult(StrictModule):
    detector_signal: Array
    log_likelihood: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class FrequencyResponsePlan(StrictModule, NonTrainableState):
    response_plus: Array
    response_cross: Array
    observed: Array
    power_spectral_density: Array
    frequency_spacing: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        response_plus: ArrayLike,
        response_cross: ArrayLike,
        observed: ArrayLike,
        power_spectral_density: ArrayLike,
        /,
        *,
        frequency_spacing: ArrayLike,
        response_id: str,
    ):
        plus = jnp.asarray(response_plus)
        cross = jnp.asarray(response_cross, dtype=plus.dtype)
        observed_ = jnp.asarray(observed, dtype=plus.dtype)
        psd = jnp.asarray(power_spectral_density)
        if (
            plus.ndim != 1
            or cross.shape != plus.shape
            or observed_.shape != plus.shape
            or psd.shape != plus.shape
        ):
            raise ValueError("Frequency response arrays must be matching vectors.")
        self.response_plus = plus
        self.response_cross = cross
        self.observed = observed_
        self.power_spectral_density = psd
        self.frequency_spacing = jnp.asarray(frequency_spacing).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frequency-response",
                "response_id": str(response_id),
                "size": int(plus.size),
            }
        )

    def evaluate(self, signal: FrequencyDomainSignal, /) -> FrequencyResponseResult:
        if not isinstance(signal, FrequencyDomainSignal):
            raise TypeError("signal must be a FrequencyDomainSignal.")
        if signal.plus.shape != self.response_plus.shape:
            raise ValueError("Signal frequency grid does not match detector response.")
        detector = self.response_plus * signal.plus + self.response_cross * signal.cross
        valid = (
            jnp.all(jnp.isfinite(detector))
            & jnp.all(jnp.isfinite(self.observed))
            & jnp.all(jnp.isfinite(self.power_spectral_density))
            & jnp.all(self.power_spectral_density > 0.0)
            & (self.frequency_spacing > 0.0)
        )
        matched = (
            4.0
            * self.frequency_spacing
            * jnp.real(
                jnp.sum(detector * jnp.conj(self.observed) / self.power_spectral_density)
            )
        )
        norm = (
            4.0
            * self.frequency_spacing
            * jnp.real(
                jnp.sum(detector * jnp.conj(detector) / self.power_spectral_density)
            )
        )
        return FrequencyResponseResult(
            jnp.where(valid, detector, jnp.zeros_like(detector)),
            jnp.where(valid, matched - 0.5 * norm, -jnp.inf),
            valid,
            self.plan_id,
        )


class RayTransferResult(StrictModule):
    intensity: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class RayTransferPlan(StrictModule, NonTrainableState):
    segment_lengths: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, segment_lengths: ArrayLike, /, *, ray_id: str):
        lengths = jnp.asarray(segment_lengths)
        if lengths.ndim != 2:
            raise ValueError("segment_lengths must have shape (rays, samples).")
        self.segment_lengths = lengths
        self.plan_id = canonical_fingerprint(
            {"kind": "ray-transfer", "ray_id": str(ray_id), "shape": list(lengths.shape)}
        )

    def evaluate(
        self, emissivity: ArrayLike, extinction: ArrayLike, /
    ) -> RayTransferResult:
        source = jnp.asarray(emissivity)
        opacity = jnp.asarray(extinction, dtype=source.dtype)
        if source.shape != self.segment_lengths.shape or opacity.shape != source.shape:
            raise ValueError("Ray source/extinction must match segment lengths.")

        def one_ray(lengths, emission, absorption):
            def step(intensity, sample):
                ds, j_value, k_value = sample
                tau = k_value * ds
                transmission = jnp.exp(-tau)
                source_increment = jnp.where(
                    jnp.abs(k_value) > 0.0,
                    j_value / k_value * (1.0 - transmission),
                    j_value * ds,
                )
                next_intensity = intensity * transmission + source_increment
                return next_intensity, None

            final, _ = jax.lax.scan(
                step,
                jnp.asarray(0.0, dtype=emission.dtype),
                (lengths, emission, absorption),
            )
            return final

        intensity = jax.vmap(one_ray)(self.segment_lengths, source, opacity)
        valid = (
            jnp.all(jnp.isfinite(source), axis=-1)
            & jnp.all(jnp.isfinite(opacity), axis=-1)
            & jnp.all(self.segment_lengths >= 0.0, axis=-1)
            & jnp.all(opacity >= 0.0, axis=-1)
        )
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
        ).astype(jnp.int32)
        return RayTransferResult(
            jnp.where(valid, intensity, 0.0), valid, status, self.plan_id
        )


class ComplexFieldState(StrictModule):
    field: Array
    wavelength: Array
    pixel_scale: Array


class StaticFieldOperatorSequence(StrictModule, NonTrainableState):
    operators: tuple[Callable, ...]
    operator_ids: tuple[str, ...] = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)

    def __init__(self, operators: tuple[Callable, ...], operator_ids: tuple[str, ...], /):
        items = tuple(operators)
        identifiers = tuple(str(value) for value in operator_ids)
        if (
            not items
            or len(items) != len(identifiers)
            or any(not callable(item) for item in items)
        ):
            raise ValueError("Static field operator sequence is invalid.")
        self.operators = items
        self.operator_ids = identifiers
        self.sequence_id = canonical_fingerprint(
            {"kind": "static-field-operator-sequence", "operators": list(identifiers)}
        )

    def apply(self, state: ComplexFieldState, /) -> tuple[ComplexFieldState, ...]:
        if not isinstance(state, ComplexFieldState):
            raise TypeError("state must be ComplexFieldState.")
        outputs = [state]
        current = state
        for operator in self.operators:
            current = operator(current)
            if not isinstance(current, ComplexFieldState):
                raise TypeError("Field operators must return ComplexFieldState.")
            outputs.append(current)
        return tuple(outputs)


__all__ = [
    "BinnedResponsePlan",
    "BinnedResponseResult",
    "ComplexFieldState",
    "FrequencyDomainSignal",
    "FrequencyResponsePlan",
    "FrequencyResponseResult",
    "ImageResponsePlan",
    "ImageResponseResult",
    "RayTransferPlan",
    "RayTransferResult",
    "SpectralField",
    "StaticFieldOperatorSequence",
]
