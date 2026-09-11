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
    "ImageResponsePlan",
    "ImageResponseResult",
    "SpectralField",
    "StaticFieldOperatorSequence",
]
