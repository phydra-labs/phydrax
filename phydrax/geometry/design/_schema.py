#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule


@dataclass(frozen=True, order=True, slots=True)
class ParameterId:
    """Stable identity of one geometry design parameter."""

    feature_id: str
    name: str

    def __post_init__(self):
        if not self.feature_id:
            raise ValueError("ParameterId.feature_id must be non-empty.")
        if not self.name:
            raise ValueError("ParameterId.name must be non-empty.")

    def __str__(self) -> str:
        return f"{self.feature_id}.{self.name}"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Static schema entry for a design-state array."""

    parameter_id: ParameterId
    shape: tuple[int, ...]
    dtype: str
    role: str
    physical_scale: float = 1.0
    bounds: tuple[float | None, float | None] = (None, None)
    trainable: bool = True

    def __post_init__(self):
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("ParameterSpec.shape dimensions must be non-negative.")
        if not self.dtype:
            raise ValueError("ParameterSpec.dtype must be non-empty.")
        if not self.role:
            raise ValueError("ParameterSpec.role must be non-empty.")
        if not math.isfinite(self.physical_scale) or self.physical_scale <= 0.0:
            raise ValueError("ParameterSpec.physical_scale must be finite and positive.")
        lower, upper = self.bounds
        if lower is not None and not math.isfinite(lower):
            raise ValueError("A finite lower parameter bound is required.")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("A finite upper parameter bound is required.")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                "ParameterSpec lower bound must be smaller than upper bound."
            )


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """Ordered, immutable schema giving state leaves stable identities."""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self):
        identifiers = tuple(spec.parameter_id for spec in self.specs)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ParameterSchema parameter IDs must be unique.")

    def __len__(self) -> int:
        return len(self.specs)

    def index(self, parameter_id: ParameterId, /) -> int:
        for index, spec in enumerate(self.specs):
            if spec.parameter_id == parameter_id:
                return index
        raise KeyError(f"Unknown geometry parameter {parameter_id}.")

    def spec(self, parameter_id: ParameterId, /) -> ParameterSpec:
        return self.specs[self.index(parameter_id)]

    @property
    def parameter_ids(self) -> tuple[ParameterId, ...]:
        return tuple(spec.parameter_id for spec in self.specs)


class DesignState(StrictModule):
    """Array-only geometry state ordered by a static parameter schema."""

    schema: ParameterSchema = eqx.field(static=True)
    values: tuple[Array, ...]

    def __init__(
        self,
        schema: ParameterSchema,
        values: Sequence[Any],
    ):
        values_ = tuple(jnp.asarray(value) for value in values)
        if len(values_) != len(schema):
            raise ValueError(
                f"DesignState has {len(values_)} values for {len(schema)} schema entries."
            )
        for spec, value in zip(schema.specs, values_, strict=True):
            if value.shape != spec.shape:
                raise ValueError(
                    f"Parameter {spec.parameter_id} must have shape {spec.shape}, "
                    f"got {value.shape}."
                )
            if str(value.dtype) != spec.dtype:
                raise ValueError(
                    f"Parameter {spec.parameter_id} must have dtype {spec.dtype}, "
                    f"got {value.dtype}."
                )
        self.schema = schema
        self.values = values_

    def value(self, binding: ParameterBinding, /) -> Array:
        return self.values[binding.index]

    def replace_at(self, index: int, value: Any, /) -> DesignState:
        """Functionally replace one leaf without changing tree structure."""
        values = list(self.values)
        replacement = jnp.asarray(value, dtype=values[index].dtype)
        if replacement.shape != values[index].shape:
            raise ValueError(
                f"Replacement shape {replacement.shape} does not match "
                f"state shape {values[index].shape}."
            )
        values[index] = replacement
        return DesignState(self.schema, values)

    def updated(
        self,
        updates: Mapping[ParameterId, Any],
        /,
    ) -> DesignState:
        """Functionally update state values addressed by stable parameter IDs."""
        values = list(self.values)
        for parameter_id, value in updates.items():
            index = self.schema.index(parameter_id)
            replacement = jnp.asarray(value, dtype=values[index].dtype)
            if replacement.shape != self.schema.specs[index].shape:
                raise ValueError(
                    f"Parameter {parameter_id} must have shape "
                    f"{self.schema.specs[index].shape}, got {replacement.shape}."
                )
            values[index] = replacement
        return DesignState(self.schema, values)


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    """Static compiled reference to one dynamic design-state leaf."""

    parameter_id: ParameterId
    index: int

    def read(self, state: DesignState, /) -> Array:
        return state.values[self.index]


class _ParameterCollector:
    """Host-side compile context building one global schema for an expression DAG."""

    def __init__(self):
        self._specs: list[ParameterSpec] = []
        self._values: list[Array] = []
        self._index_by_id: dict[ParameterId, int] = {}

    def bind(
        self,
        parameter_id: ParameterId,
        value: Any,
        /,
        *,
        role: str,
        physical_scale: float = 1.0,
        bounds: tuple[float | None, float | None] = (None, None),
        trainable: bool = True,
    ) -> ParameterBinding:
        array = jnp.asarray(value, dtype=float)
        spec = ParameterSpec(
            parameter_id=parameter_id,
            shape=array.shape,
            dtype=str(array.dtype),
            role=role,
            physical_scale=physical_scale,
            bounds=bounds,
            trainable=trainable,
        )
        if parameter_id in self._index_by_id:
            index = self._index_by_id[parameter_id]
            if self._specs[index] != spec:
                raise ValueError(
                    f"Conflicting schemas for shared parameter {parameter_id}."
                )
            if not np.array_equal(np.asarray(self._values[index]), np.asarray(array)):
                raise ValueError(
                    f"Conflicting initial values for shared parameter {parameter_id}."
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
