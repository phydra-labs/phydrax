#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from math import prod
from operator import index
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._doc import DOC_KEY0
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import (
    HermitianSpectralCoordinates,
    TensorSpectralDiscretization,
)
from ....discretization.spectral._modal_discovery import PreparedModalSupport
from ....domain import Domain, DomainFunction
from ..._keys import EvalKey
from ...parameters import PositiveTransform, TransformedParameter


DecayAggregation = Literal["sum", "mean"]


def _component_shape(value: Sequence[int], /) -> tuple[int, ...]:
    if any(isinstance(size, bool) for size in value):
        raise TypeError("component_shape dimensions must be integers.")
    shape = tuple(index(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("component_shape dimensions must be positive.")
    return shape


def _inverse_softplus(value: ArrayLike, /) -> Array:
    physical = jnp.asarray(value, dtype=float)
    return jnp.where(physical > 20.0, physical, jnp.log(jnp.expm1(physical)))


def _batched_model(model: Any, inputs: Array, key: EvalKey, /) -> Array:
    if key is None:
        return jax.vmap(lambda point: model(point, key=None))(inputs)
    sites = jnp.arange(inputs.shape[0], dtype=jnp.uint32)
    return jax.vmap(lambda point, site: model(point, key=jr.fold_in(key, site)))(
        inputs,
        sites,
    )


class _ModalCoordinateGrid(StrictModule, NonTrainableState):
    mode_numbers: Array
    normalized_mode_numbers: Array
    mode_scales: Array
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        mode_scales: ArrayLike | None,
        maximum_query_points: int,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        maximum = index(maximum_query_points)
        point_count = math.prod(discretization.modal_shape)
        if maximum < 1:
            raise ValueError("maximum_query_points must be positive.")
        if point_count > maximum:
            raise ValueError(
                f"Modal query grid has {point_count} points, exceeding "
                f"maximum_query_points={maximum}."
            )
        dimension = len(discretization.axes)
        storage_indices = np.indices(discretization.modal_shape, dtype=np.int64).reshape(
            (dimension, -1)
        )
        numbers = np.stack(
            tuple(
                np.asarray(axis.modes.mode_numbers, dtype=float)[
                    storage_indices[axis_index]
                ]
                for axis_index, axis in enumerate(discretization.axes)
            ),
            axis=-1,
        )
        scales = (
            np.ones((dimension,), dtype=float)
            if mode_scales is None
            else np.asarray(mode_scales, dtype=float).reshape((-1,))
        )
        if scales.shape != (dimension,):
            raise ValueError(
                f"mode_scales must have one value per spectral axis ({dimension})."
            )
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("mode_scales must be finite and positive.")
        normalized = numbers / scales
        self.mode_numbers = jnp.asarray(numbers)
        self.normalized_mode_numbers = jnp.asarray(normalized)
        self.mode_scales = jnp.asarray(scales)
        self.modal_shape = discretization.modal_shape
        self.dimension = dimension
        self.point_count = point_count
        self.grid_id = canonical_fingerprint(
            {
                "kind": "implicit-modal-coordinate-grid",
                "discretization": discretization.prepared_id,
                "mode_numbers": array_tree_fingerprint(numbers),
                "mode_scales": array_tree_fingerprint(scales),
            }
        )


class _FixedRates(StrictModule, NonTrainableState):
    values: Array

    def __init__(self, values: ArrayLike, /):
        self.values = jnp.asarray(values, dtype=float)

    def __call__(self) -> Array:
        return self.values


class ExponentialSpectralEnvelope(StrictModule):
    """Positive axis-anisotropic exponential envelope over modal indices.

    The envelope is ``exp(-sum_j rate_j * abs(k_j)))``. ``aggregation='mean'``
    divides the exponent by the number of axes and is an explicit heuristic rather
    than the tensor-product analytic-decay law.
    """

    parameter: TransformedParameter | _FixedRates
    dimension: int = eqx.field(static=True)
    aggregation: DecayAggregation = eqx.field(static=True)
    trainable: bool = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_rates: ArrayLike,
        /,
        *,
        trainable: bool = True,
        minimum_rate: float = 0.0,
        aggregation: DecayAggregation = "sum",
    ):
        rates = np.asarray(initial_rates, dtype=float).reshape((-1,))
        minimum = float(minimum_rate)
        if rates.size == 0 or np.any(~np.isfinite(rates)):
            raise ValueError("initial_rates must contain finite values.")
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_rate must be finite and nonnegative.")
        if np.any(rates < minimum) or (trainable and np.any(rates <= minimum)):
            relation = "exceed" if trainable else "be at least"
            raise ValueError(f"initial_rates must {relation} minimum_rate.")
        if aggregation not in ("sum", "mean"):
            raise ValueError("aggregation must be 'sum' or 'mean'.")
        parameter: TransformedParameter | _FixedRates
        if trainable:
            raw = _inverse_softplus(rates - minimum)
            parameter = TransformedParameter(raw, PositiveTransform(minimum))
        else:
            parameter = _FixedRates(rates)
        self.parameter = parameter
        self.dimension = int(rates.size)
        self.aggregation = aggregation
        self.trainable = bool(trainable)
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "exponential-spectral-envelope",
                "dimension": self.dimension,
                "aggregation": aggregation,
                "trainable": bool(trainable),
                "minimum_rate": minimum,
            }
        )

    @property
    def rates(self) -> Array:
        return self.parameter()

    def __call__(self, mode_numbers: ArrayLike, /) -> Array:
        modes = jnp.asarray(mode_numbers, dtype=self.rates.dtype)
        if modes.ndim != 2 or modes.shape[-1] != self.dimension:
            raise ValueError("mode_numbers must have shape (points, envelope dimension).")
        exponent = contract("pd,d->p", jnp.abs(modes), self.rates)
        if self.aggregation == "mean":
            exponent = exponent / float(self.dimension)
        return jnp.exp(-exponent)


class _ModalFeatureTable(StrictModule, NonTrainableState):
    values: Array
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    coarse_counts: tuple[int, ...] = eqx.field(static=True)
    feature_size: int = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        coarse_counts: Sequence[int],
        /,
        *,
        maximum_feature_bytes: int,
    ):
        counts = tuple(index(value) for value in coarse_counts)
        if len(counts) != len(discretization.axes):
            raise ValueError("coarse_counts must provide one count per spectral axis.")
        if any(count <= 0 for count in counts):
            raise ValueError("coarse_counts must be positive.")
        multi_indices = np.indices(discretization.modal_shape, dtype=np.int64).reshape(
            (len(discretization.axes), -1)
        )
        blocks: list[np.ndarray] = []
        for axis_index, (axis, count) in enumerate(
            zip(discretization.axes, counts, strict=True)
        ):
            if count > axis.physical_count:
                raise ValueError(
                    "A coarse feature count cannot exceed its prepared physical count."
                )
            positions = np.floor(
                np.linspace(0.0, float(axis.physical_count), count, endpoint=False)
            ).astype(np.int64)
            identity = jnp.eye(
                axis.mode_count,
                dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
            )
            basis_values = np.asarray(jax.vmap(axis.synthesize)(identity))[:, positions]
            selected = basis_values[multi_indices[axis_index]]
            blocks.append(np.asarray(selected.real, dtype=float))
            if np.iscomplexobj(selected):
                blocks.append(np.asarray(selected.imag, dtype=float))
        values = np.concatenate(blocks, axis=-1)
        byte_limit = index(maximum_feature_bytes)
        if byte_limit < 1:
            raise ValueError("maximum_feature_bytes must be positive.")
        if values.nbytes > byte_limit:
            raise ValueError(
                f"Basis feature table requires {values.nbytes} bytes, exceeding "
                f"maximum_feature_bytes={byte_limit}."
            )
        self.values = jnp.asarray(values)
        self.modal_shape = discretization.modal_shape
        self.coarse_counts = counts
        self.feature_size = int(values.shape[-1])
        self.table_id = canonical_fingerprint(
            {
                "kind": "spectral-basis-feature-table",
                "discretization": discretization.prepared_id,
                "coarse_counts": list(counts),
                "values": array_tree_fingerprint(values),
            }
        )


class SpectralBasisModulation(StrictModule):
    """Learn one coefficient multiplier from exact prepared basis samples."""

    model: Any
    table: _ModalFeatureTable
    component_shape: tuple[int, ...] = eqx.field(static=True)
    modulation_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Any,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        coarse_counts: Sequence[int],
        component_shape: Sequence[int] = (),
        maximum_feature_bytes: int = 256 * 1024**2,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        components = _component_shape(component_shape)
        table = _ModalFeatureTable(
            discretization,
            coarse_counts,
            maximum_feature_bytes=maximum_feature_bytes,
        )
        self.model = model
        self.table = table
        self.component_shape = components
        self.modulation_id = canonical_fingerprint(
            {
                "kind": "spectral-basis-modulation",
                "table": table.table_id,
                "component_shape": list(components),
                "model_type": type(model).__qualname__,
            }
        )

    @property
    def feature_size(self) -> int:
        return self.table.feature_size

    def __call__(self, *, key: EvalKey = DOC_KEY0) -> Array:
        values = jnp.asarray(_batched_model(self.model, self.table.values, key))
        expected = (self.table.values.shape[0],) + self.component_shape
        if values.shape != expected:
            raise ValueError(
                f"Basis modulation model must return batched shape {expected}; "
                f"got {values.shape}."
            )
        return values.reshape(self.table.modal_shape + self.component_shape)


class ImplicitModalField(StrictModule):
    """Materialize a modal tensor by querying one shared coefficient model.

    The wrapped model receives ``[k_1 / s_1, ..., k_d / s_d, t]`` and must return
    one scalar or ``component_shape`` coefficient. Complex output should be supplied
    directly, for example through ``ComplexOutputModel``.
    """

    model: Any
    discretization: TensorSpectralDiscretization
    grid: _ModalCoordinateGrid
    envelope: ExponentialSpectralEnvelope | None
    basis_modulation: SpectralBasisModulation | None
    hermitian_coordinates: HermitianSpectralCoordinates | None
    component_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    real_field: bool = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Any,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        component_shape: Sequence[int] = (),
        mode_scales: ArrayLike | None = None,
        envelope: ExponentialSpectralEnvelope | None = None,
        basis_modulation: SpectralBasisModulation | None = None,
        real_field: bool = False,
        reality_tolerance: float = 1e-10,
        maximum_query_points: int = 1_000_000,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        components = _component_shape(component_shape)
        grid = _ModalCoordinateGrid(
            discretization,
            mode_scales=mode_scales,
            maximum_query_points=maximum_query_points,
        )
        if envelope is not None:
            if not isinstance(envelope, ExponentialSpectralEnvelope):
                raise TypeError("envelope must be ExponentialSpectralEnvelope or None.")
            if envelope.dimension != grid.dimension:
                raise ValueError("Envelope dimension must match the spectral axes.")
        if basis_modulation is not None:
            if not isinstance(basis_modulation, SpectralBasisModulation):
                raise TypeError(
                    "basis_modulation must be SpectralBasisModulation or None."
                )
            if basis_modulation.table.modal_shape != discretization.modal_shape:
                raise ValueError("Basis modulation modal shape is incompatible.")
            if basis_modulation.component_shape != components:
                raise ValueError("Basis modulation component shape is incompatible.")
        hermitian = (
            HermitianSpectralCoordinates(
                discretization,
                component_shape=components,
                reality_tolerance=reality_tolerance,
            )
            if real_field
            else None
        )
        self.model = model
        self.discretization = discretization
        self.grid = grid
        self.envelope = envelope
        self.basis_modulation = basis_modulation
        self.hermitian_coordinates = hermitian
        self.component_shape = components
        self.state_shape = discretization.modal_shape + components
        self.real_field = bool(real_field)
        self.field_id = canonical_fingerprint(
            {
                "kind": "implicit-modal-field",
                "discretization": discretization.prepared_id,
                "grid": grid.grid_id,
                "component_shape": list(components),
                "real_field": bool(real_field),
                "envelope": None if envelope is None else envelope.envelope_id,
                "basis_modulation": (
                    None if basis_modulation is None else basis_modulation.modulation_id
                ),
                "model_type": type(model).__qualname__,
            }
        )

    @property
    def input_size(self) -> int:
        return self.grid.dimension + 1

    @property
    def mode_numbers(self) -> Array:
        return self.grid.mode_numbers

    def model_inputs(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time)
        if time_.shape != () or jnp.iscomplexobj(time_):
            raise ValueError("time must be one real scalar.")
        column = jnp.broadcast_to(time_, (self.grid.point_count, 1))
        return jnp.concatenate((self.grid.normalized_mode_numbers, column), axis=-1)

    def __call__(
        self,
        time: ArrayLike,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        raw = jnp.asarray(_batched_model(self.model, self.model_inputs(time), key))
        expected = (self.grid.point_count,) + self.component_shape
        if raw.shape != expected:
            raise ValueError(
                f"Coefficient model must return batched shape {expected}; got {raw.shape}."
            )
        coefficient_dtype = jnp.dtype(
            self.discretization.plan.precision.coefficient_dtype
        )
        state = raw.reshape(self.state_shape).astype(coefficient_dtype)
        if self.basis_modulation is not None:
            modulation_key = None if key is None else jr.fold_in(key, 1_000_000_007)
            state = state * self.basis_modulation(key=modulation_key).astype(
                coefficient_dtype
            )
        if self.envelope is not None:
            envelope = self.envelope(self.grid.mode_numbers).reshape(
                self.discretization.modal_shape + (1,) * len(self.component_shape)
            )
            state = state * envelope.astype(coefficient_dtype)
        if self.hermitian_coordinates is not None:
            state = self.hermitian_coordinates.project(state)
        return state

    def query(
        self,
        time: ArrayLike,
        flat_indices: ArrayLike,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        """Return selected storage-order coefficients after all field constraints."""
        indices = jnp.asarray(flat_indices)
        if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("flat_indices must be a rank-one integer array.")
        flat = self(time, key=key).reshape(
            (self.grid.point_count,) + self.component_shape
        )
        return flat[indices]

    def time_tangent(
        self,
        time: ArrayLike,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> tuple[Array, Array]:
        time_ = jnp.asarray(time)
        if time_.shape != () or jnp.iscomplexobj(time_):
            raise ValueError("time must be one real scalar.")
        return jax.jvp(
            lambda value: self(value, key=key),
            (time_,),
            (jnp.ones_like(time_),),
        )

    def physical_values(
        self,
        time: ArrayLike,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        real_output: bool | None = None,
    ) -> Array:
        return self.discretization.reconstruct(
            self(time, key=key),
            real_output=real_output,
        )

    def as_domain_function(
        self,
        domain: Domain,
        /,
        *,
        time_label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DomainFunction:
        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain.")
        label = (
            domain.labels[0]
            if time_label is None and len(domain.labels) == 1
            else None
            if time_label is None
            else str(time_label)
        )
        if label is None or label not in domain.labels:
            raise ValueError("time_label must identify one factor of domain.")
        values = {} if metadata is None else dict(metadata)
        values.update(
            {
                "implicit_modal_field_id": self.field_id,
                "spectral_discretization_id": self.discretization.prepared_id,
                "representation": "modal_coefficient",
            }
        )
        return DomainFunction(
            domain=domain,
            deps=(label,),
            func=self,
            metadata=values,
        )


class SparseImplicitModalField(StrictModule):
    """Fixed-capacity discovered modal support with explicit dense scatter."""

    support: PreparedModalSupport
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    support_epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: PreparedModalSupport,
        modal_shape: Sequence[int],
        /,
    ):
        if not isinstance(support, PreparedModalSupport):
            raise TypeError("support must be PreparedModalSupport.")
        shape = tuple(int(value) for value in modal_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("modal_shape must be positive.")
        if support.multi_indices.shape[-1] != len(shape):
            raise ValueError("Sparse modal indices do not match modal_shape rank.")
        self.support = support
        self.modal_shape = shape
        self.support_epoch_id = support.support_id

    def sparse_coefficients(self) -> Array:
        mask = self.support.active.reshape(
            (1,) * (self.support.coefficients.ndim - 2)
            + (self.support.active.shape[0], 1)
        )
        return jnp.where(mask, self.support.coefficients, 0.0)

    def __call__(self, /, *, support_epoch_id: str | None = None) -> Array:
        if support_epoch_id is not None and support_epoch_id != self.support_epoch_id:
            raise ValueError("Sparse modal support epoch is stale.")
        coefficients = self.sparse_coefficients()
        flat_indices = jnp.ravel_multi_index(
            tuple(
                self.support.multi_indices[:, axis]
                for axis in range(len(self.modal_shape))
            ),
            self.modal_shape,
        )
        leading = coefficients.shape[:-2]
        channels = coefficients.shape[-1]
        dense = jnp.zeros(
            leading + (prod(self.modal_shape), channels),
            dtype=coefficients.dtype,
        )
        dense = dense.at[..., flat_indices, :].add(coefficients)
        return dense.reshape(leading + self.modal_shape + (channels,))


__all__ = [
    "DecayAggregation",
    "ExponentialSpectralEnvelope",
    "ImplicitModalField",
    "SparseImplicitModalField",
    "SpectralBasisModulation",
]
