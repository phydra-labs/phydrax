#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked, ConvertibleToArray
from ...units import ANGLE, DIMENSIONLESS, LENGTH, UnitDefinition


@dataclass(frozen=True, order=True, slots=True)
class ParameterId:
    """Stable identity of one geometry design parameter."""

    feature_id: str
    name: str

    def __post_init__(self) -> None:
        if not self.feature_id:
            raise ValueError(
                f"ParameterId.feature_id must be nonempty; got {self.feature_id!r}"
            )
        if not self.name:
            raise ValueError(f"ParameterId.name must be nonempty; got {self.name!r}")

    def __str__(self) -> str:
        return f"{self.feature_id}.{self.name}"


def _require_parameter_dimension(role: str, unit: UnitDefinition) -> None:
    expected_dimension = None
    if role in {
        "position",
        "position_offset",
        "length",
        "direction_length",
        "blend_width",
        "extrusion_height",
        "tube_radius",
    }:
        expected_dimension = LENGTH
    elif role == "angle":
        expected_dimension = ANGLE
    elif role in {
        "rotation_matrix",
        "orientation",
        "shape",
        "scale",
        "network_weight",
    }:
        expected_dimension = DIMENSIONLESS
    if expected_dimension is not None and unit.dimension != expected_dimension:
        raise ValueError(
            f"ParameterSpec.unit must match its declared role dimension; got {role}: {unit.symbol}"
        )


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Static schema entry for a design-state array."""

    parameter_id: ParameterId
    shape: tuple[int, ...]
    dtype: str
    role: str
    unit: UnitDefinition
    physical_scale: float = 1.0
    bounds: tuple[float | None, float | None] = (None, None)
    trainable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.unit, UnitDefinition):
            raise TypeError(
                f"ParameterSpec.unit must be a UnitDefinition; got {type(self.unit).__name__}"
            )
        _require_parameter_dimension(self.role, self.unit)
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError(
                f"ParameterSpec.shape must contain nonnegative dimensions; got {self.shape}"
            )
        if not self.dtype:
            raise ValueError(f"ParameterSpec.dtype must be nonempty; got {self.dtype!r}")
        if not self.role:
            raise ValueError(f"ParameterSpec.role must be nonempty; got {self.role!r}")
        if not math.isfinite(self.physical_scale) or self.physical_scale <= 0.0:
            raise ValueError(
                f"ParameterSpec.physical_scale must be finite and positive; got {self.physical_scale}"
            )
        lower, upper = self.bounds
        if lower is not None and not math.isfinite(lower):
            raise ValueError(f"ParameterSpec.bounds.lower must be finite; got {lower}")
        if upper is not None and not math.isfinite(upper):
            raise ValueError(f"ParameterSpec.bounds.upper must be finite; got {upper}")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                f"ParameterSpec.bounds.lower must be smaller than upper bound; got {lower}, {upper}"
            )


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """Ordered, immutable schema giving state leaves stable identities."""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        identifiers = tuple(spec.parameter_id for spec in self.specs)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                f"ParameterSchema.parameter_ids must be unique; got {identifiers}"
            )

    @property
    def schema_id(self) -> str:
        """Identity including every parameter's unit declaration."""
        return canonical_fingerprint(
            {
                "kind": "design-parameter-schema",
                "specs": [
                    {
                        "parameter_id": [
                            spec.parameter_id.feature_id,
                            spec.parameter_id.name,
                        ],
                        "shape": spec.shape,
                        "dtype": spec.dtype,
                        "role": spec.role,
                        "unit_id": spec.unit.unit_id,
                        "physical_scale": spec.physical_scale,
                        "bounds": spec.bounds,
                        "trainable": spec.trainable,
                    }
                    for spec in self.specs
                ],
            }
        )

    def __len__(self) -> int:
        return len(self.specs)

    @checked
    def index(self, parameter_id: ParameterId, /) -> int:
        for index, spec in enumerate(self.specs):
            if spec.parameter_id == parameter_id:
                return index
        raise KeyError(
            f"ParameterSchema.parameter_id must identify a schema parameter; got {parameter_id}"
        )

    @checked
    def spec(self, parameter_id: ParameterId, /) -> ParameterSpec:
        return self.specs[self.index(parameter_id)]

    @property
    def parameter_ids(self) -> tuple[ParameterId, ...]:
        return tuple(spec.parameter_id for spec in self.specs)


class DesignState(StrictModule):
    """Array-only geometry state ordered by a static parameter schema."""

    schema: ParameterSchema = eqx.field(static=True)
    values: tuple[Array, ...]

    @checked
    def __init__(
        self,
        schema: ParameterSchema,
        values: Sequence[ConvertibleToArray],
    ) -> None:
        values_ = tuple(jnp.asarray(value) for value in values)
        if len(values_) != len(schema):
            raise ValueError(
                f"DesignState.values must align with schema entries; got {len(values_)} values for {len(schema)} entries"
            )
        for spec, value in zip(schema.specs, values_, strict=True):
            if value.shape != spec.shape:
                raise ValueError(
                    f"DesignState.values must match parameter shape {spec.shape}; got {spec.parameter_id}: {value.shape}"
                )
            if str(value.dtype) != spec.dtype:
                raise ValueError(
                    f"DesignState.values must match parameter dtype {spec.dtype}; got {spec.parameter_id}: {value.dtype}"
                )
        self.schema = schema
        self.values = values_

    @checked
    def value(self, binding: ParameterBinding, /) -> Array:
        return self.values[binding.index]

    def replace_at(self, index: int, value: ConvertibleToArray, /) -> DesignState:
        """Functionally replace one leaf without changing tree structure."""
        values = list(self.values)
        replacement = jnp.asarray(value, dtype=values[index].dtype)
        if replacement.shape != values[index].shape:
            raise ValueError(
                f"DesignState.replacement must match state shape {values[index].shape}; got {replacement.shape}"
            )
        values[index] = replacement
        return DesignState(self.schema, values)

    def updated(
        self,
        updates: Mapping[ParameterId, ConvertibleToArray],
        /,
    ) -> DesignState:
        """Functionally update state values addressed by stable parameter IDs."""
        values = list(self.values)
        for parameter_id, value in updates.items():
            index = self.schema.index(parameter_id)
            replacement = jnp.asarray(value, dtype=values[index].dtype)
            if replacement.shape != self.schema.specs[index].shape:
                raise ValueError(
                    "DesignState.updates must match parameter shape "
                    f"{self.schema.specs[index].shape}; got {parameter_id}: {replacement.shape}"
                )
            values[index] = replacement
        return DesignState(self.schema, values)


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    """Static compiled reference to one dynamic design-state leaf."""

    parameter_id: ParameterId
    index: int

    @checked
    def read(self, state: DesignState, /) -> Array:
        return state.values[self.index]


class ParameterCollector:
    """Host-side compile context building one global schema for an expression DAG."""

    @checked
    def __init__(self, coordinate_unit: UnitDefinition) -> None:
        if coordinate_unit.dimension != LENGTH:
            raise ValueError(
                f"ParameterCollector.coordinate_unit must declare a length unit; got {coordinate_unit.symbol}"
            )
        self.coordinate_unit = coordinate_unit
        self._specs: list[ParameterSpec] = []
        self._values: list[Array] = []
        self._index_by_id: dict[ParameterId, int] = {}

    @checked
    def bind(
        self,
        parameter_id: ParameterId,
        value: ConvertibleToArray,
        /,
        *,
        role: str,
        unit: UnitDefinition,
        physical_scale: float = 1.0,
        bounds: tuple[float | None, float | None] = (None, None),
        trainable: bool = True,
    ) -> ParameterBinding:
        array = jnp.asarray(value, dtype=jnp.float64)
        spec = ParameterSpec(
            parameter_id=parameter_id,
            shape=array.shape,
            dtype=str(array.dtype),
            role=role,
            unit=unit,
            physical_scale=physical_scale,
            bounds=bounds,
            trainable=trainable,
        )
        if parameter_id in self._index_by_id:
            index = self._index_by_id[parameter_id]
            if self._specs[index] != spec:
                raise ValueError(
                    f"ParameterCollector.parameter must retain its shared schema; got conflicting {parameter_id}"
                )
            previous_host, current_host = jax.device_get((self._values[index], array))
            if not np.array_equal(np.asarray(previous_host), np.asarray(current_host)):
                raise ValueError(
                    f"ParameterCollector.parameter must retain its shared initial value; got conflicting {parameter_id}"
                )
            return ParameterBinding(parameter_id, index)

        index = len(self._specs)
        self._index_by_id[parameter_id] = index
        self._specs.append(spec)
        self._values.append(array)
        return ParameterBinding(parameter_id, index)

    def finish(self) -> tuple[ParameterSchema, DesignState]:
        schema = ParameterSchema(tuple(self._specs))
        return schema, DesignState(schema, tuple(self._values))


__all__ = [
    "DesignState",
    "ParameterBinding",
    "ParameterId",
    "ParameterSchema",
    "ParameterSpec",
]
