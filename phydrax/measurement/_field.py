#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host and JAX quantity fields with explicit sampling evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import conversion_factor, UnitDefinition
from ._quantity import canonical_quantity_text, QuantitySpec, ValueKind, ValueLayout
from ._support import SampleSupport
from ._time import TemporalSampling


def _readonly(value: ArrayLike, name: str, /, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use object dtype.")
    array.setflags(write=False)
    return array


class SpatialSamplingKind(StrEnum):
    POINT = "point"
    CELL_AVERAGE = "cell_average"
    CELL_INTEGRAL = "cell_integral"
    SURFACE_AVERAGE = "surface_average"
    PATH_INTEGRAL = "path_integral"
    DETECTOR_BIN = "detector_bin"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class SamplingSemantics:
    """Spatial and temporal operation represented by stored sample values."""

    spatial_kind: SpatialSamplingKind
    temporal: TemporalSampling = field(default_factory=TemporalSampling)
    footprint_id: str | None = None
    normalization: str = "none"
    sampling_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spatial_kind, SpatialSamplingKind):
            raise TypeError("spatial_kind must be SpatialSamplingKind.")
        if not isinstance(self.temporal, TemporalSampling):
            raise TypeError("temporal must be TemporalSampling.")
        footprint = (
            None
            if self.footprint_id is None
            else canonical_quantity_text(self.footprint_id, "footprint_id")
        )
        normalization = canonical_quantity_text(self.normalization, "normalization")
        object.__setattr__(self, "footprint_id", footprint)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(
            self,
            "sampling_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-sampling-semantics",
                    "spatial": self.spatial_kind.value,
                    "temporal": self.temporal.temporal_sampling_id,
                    "footprint": footprint,
                    "normalization": normalization,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class IndependentStandardUncertainty:
    """Independent standard uncertainty, without an implied covariance model."""

    values: np.ndarray
    unit: UnitDefinition
    uncertainty_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        values = _readonly(self.values, "uncertainty values")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("Standard uncertainty requires floating-point storage.")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Standard uncertainty must be finite and non-negative.")
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "uncertainty_id",
            canonical_fingerprint(
                {
                    "kind": "independent-standard-uncertainty",
                    "unit": self.unit.unit_id,
                    "values": array_tree_fingerprint(values),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class QualityFlag:
    """Named sample quality evidence that does not alter validity implicitly."""

    name: str
    mask: np.ndarray
    meaning: str
    flag_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = canonical_quantity_text(self.name, "quality flag name")
        meaning = canonical_quantity_text(self.meaning, "quality flag meaning")
        mask = _readonly(self.mask, "quality flag mask", dtype=bool)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "meaning", meaning)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(
            self,
            "flag_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-quality-flag",
                    "name": name,
                    "meaning": meaning,
                    "mask": array_tree_fingerprint(mask),
                }
            ),
        )


class PreparedQuantityField(StrictModule, NonTrainableState):
    """Fixed-shape JAX quantity field for compiled observation operations."""

    values: Array
    valid_mask: Array
    standard_uncertainty: Array | None
    quantity_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        valid_mask: ArrayLike,
        /,
        *,
        standard_uncertainty: ArrayLike | None,
        quantity_id: str,
        compatibility_id: str,
        layout_id: str,
        support_id: str,
        sampling_id: str,
        unit_id: str,
        field_id: str,
    ) -> None:
        values_ = jnp.asarray(values)
        valid_ = jnp.asarray(valid_mask, dtype=bool)
        if valid_.ndim > values_.ndim or values_.shape[: valid_.ndim] != valid_.shape:
            raise ValueError("valid_mask must match the leading sample shape of values.")
        uncertainty_ = (
            None if standard_uncertainty is None else jnp.asarray(standard_uncertainty)
        )
        if uncertainty_ is not None and uncertainty_.shape not in {
            valid_.shape,
            values_.shape,
        }:
            raise ValueError("standard_uncertainty must have sample or value shape.")
        identifiers = tuple(
            canonical_quantity_text(value, name)
            for value, name in (
                (quantity_id, "quantity_id"),
                (compatibility_id, "compatibility_id"),
                (layout_id, "layout_id"),
                (support_id, "support_id"),
                (sampling_id, "sampling_id"),
                (unit_id, "unit_id"),
                (field_id, "field_id"),
            )
        )
        self.values = values_
        self.valid_mask = valid_
        self.standard_uncertainty = uncertainty_
        (
            self.quantity_id,
            self.compatibility_id,
            self.layout_id,
            self.support_id,
            self.sampling_id,
            self.unit_id,
            self.field_id,
        ) = identifiers
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-quantity-field",
                "field": self.field_id,
                "quantity": self.quantity_id,
                "layout": self.layout_id,
                "support": self.support_id,
                "sampling": self.sampling_id,
                "unit": self.unit_id,
                "shape": list(values_.shape),
                "dtype": str(values_.dtype),
                "uncertainty": None
                if uncertainty_ is None
                else {
                    "shape": list(uncertainty_.shape),
                    "dtype": str(uncertainty_.dtype),
                },
            }
        )

    @property
    def successful(self) -> Array:
        expanded = self.valid_mask.reshape(
            self.valid_mask.shape + (1,) * (self.values.ndim - self.valid_mask.ndim)
        )
        finite = jnp.all(jnp.where(expanded, jnp.isfinite(self.values), True))
        uncertainty_finite = (
            jnp.asarray(True)
            if self.standard_uncertainty is None
            else jnp.all(jnp.isfinite(self.standard_uncertainty))
        )
        return finite & uncertainty_finite


@dataclass(frozen=True, slots=True)
class QuantityField:
    """One immutable physical quantity sampled on an explicit support."""

    field_id: str
    quantity: QuantitySpec
    layout: ValueLayout
    support: SampleSupport
    sampling: SamplingSemantics
    values: np.ndarray
    valid_mask: np.ndarray | None = None
    uncertainty: IndependentStandardUncertainty | None = None
    quality_flags: tuple[QualityFlag, ...] = ()
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        field_id = canonical_quantity_text(self.field_id, "field_id")
        if not isinstance(self.quantity, QuantitySpec):
            raise TypeError("quantity must be QuantitySpec.")
        if not isinstance(self.layout, ValueLayout):
            raise TypeError("layout must be ValueLayout.")
        if not isinstance(self.support, SampleSupport):
            raise TypeError("support must implement SampleSupport.")
        if not isinstance(self.sampling, SamplingSemantics):
            raise TypeError("sampling must be SamplingSemantics.")
        values = _readonly(self.values, "values")
        expected = self.support.sample_shape + self.layout.component_shape
        if values.shape != expected:
            raise ValueError(f"values must have shape {expected}; got {values.shape}.")
        mask = (
            np.ones(self.support.sample_shape, dtype=bool)
            if self.valid_mask is None
            else _readonly(self.valid_mask, "valid_mask", dtype=bool)
        )
        if mask.shape != self.support.sample_shape:
            raise ValueError("valid_mask must have the support sample shape.")
        selected = values[mask]
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(selected)):
            raise ValueError("values must be finite wherever valid_mask is true.")
        if self.layout.kind is ValueKind.REAL_SCALAR and not np.issubdtype(
            values.dtype, np.floating
        ):
            raise TypeError("real_scalar values require floating-point storage.")
        if self.layout.kind is ValueKind.COMPLEX_SCALAR and not np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise TypeError("complex_scalar values require complex storage.")
        if self.layout.kind is ValueKind.CATEGORICAL and not np.issubdtype(
            values.dtype, np.integer
        ):
            raise TypeError("categorical values require integer storage.")
        if self.layout.kind is ValueKind.COUNT:
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError("count values require integer storage.")
            if np.any(selected < 0):
                raise ValueError("count values must be non-negative where valid.")
        if self.layout.kind is ValueKind.PROBABILITY:
            if not np.issubdtype(values.dtype, np.floating):
                raise TypeError("probability values require floating-point storage.")
            tolerance = 128.0 * np.finfo(values.dtype).eps
            if np.any(selected < -tolerance) or not np.allclose(
                np.sum(selected, axis=-1), 1.0, atol=tolerance, rtol=0.0
            ):
                raise ValueError(
                    "probability values must be non-negative and sum to one."
                )
        uncertainty = self.uncertainty
        if uncertainty is not None:
            if not isinstance(uncertainty, IndependentStandardUncertainty):
                raise TypeError(
                    "uncertainty must be IndependentStandardUncertainty or None."
                )
            conversion_factor(uncertainty.unit, self.quantity.unit)
            if uncertainty.values.shape not in {mask.shape, values.shape}:
                raise ValueError("uncertainty must have sample or value shape.")
        flags = tuple(self.quality_flags)
        if any(not isinstance(flag, QualityFlag) for flag in flags):
            raise TypeError("quality_flags must contain QualityFlag values.")
        if any(flag.mask.shape != mask.shape for flag in flags):
            raise ValueError("Every quality flag must have the sample shape.")
        if len({flag.name for flag in flags}) != len(flags):
            raise ValueError("Quality flag names must be unique.")
        mask.setflags(write=False)
        object.__setattr__(self, "field_id", field_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "quality_flags", flags)
        object.__setattr__(
            self,
            "content_id",
            canonical_fingerprint(
                {
                    "kind": "quantity-field",
                    "field": field_id,
                    "quantity": self.quantity.quantity_id,
                    "layout": self.layout.layout_id,
                    "support": self.support.support_id,
                    "sampling": self.sampling.sampling_id,
                    "values": array_tree_fingerprint(values),
                    "valid": array_tree_fingerprint(mask),
                    "uncertainty": None
                    if uncertainty is None
                    else uncertainty.uncertainty_id,
                    "quality": [flag.flag_id for flag in flags],
                }
            ),
        )

    def prepare(
        self, /, *, target_unit: UnitDefinition | None = None
    ) -> PreparedQuantityField:
        unit = self.quantity.unit if target_unit is None else target_unit
        factor = conversion_factor(self.quantity.unit, unit)
        values = jnp.asarray(self.values)
        if factor != 1:
            values = values * jnp.asarray(
                float(factor), dtype=jnp.result_type(values, 1.0)
            )
        uncertainty = None
        if self.uncertainty is not None:
            uncertainty_factor = conversion_factor(self.uncertainty.unit, unit)
            uncertainty = jnp.asarray(self.uncertainty.values)
            if uncertainty_factor != 1:
                uncertainty = uncertainty * jnp.asarray(
                    float(uncertainty_factor), dtype=uncertainty.dtype
                )
        return PreparedQuantityField(
            values,
            self.valid_mask,
            standard_uncertainty=uncertainty,
            quantity_id=self.quantity.quantity_id,
            compatibility_id=self.quantity.compatibility_id,
            layout_id=self.layout.layout_id,
            support_id=self.support.support_id,
            sampling_id=self.sampling.sampling_id,
            unit_id=unit.unit_id,
            field_id=self.field_id,
        )


__all__ = [
    "IndependentStandardUncertainty",
    "PreparedQuantityField",
    "QualityFlag",
    "QuantityField",
    "SamplingSemantics",
    "SpatialSamplingKind",
]
