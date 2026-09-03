#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Typed parameter coordinates for cardiovascular personalization."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._probability import AbstractProbabilityLaw
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....uq import AbstractBijector, ParameterSpace
from .._quantities import CardiovascularQuantitySpec


_TOKEN = re.compile(r"[a-z][a-z0-9_.-]*\Z")


class CardiacSubsystem(Enum):
    """Physical owner of a parameter; inverse routes admit only selected owners."""

    ELECTROPHYSIOLOGY = "electrophysiology"
    PASSIVE_MECHANICS = "passive_mechanics"
    ACTIVE_MECHANICS = "active_mechanics"
    CIRCULATION = "circulation"
    LOADING = "loading"
    UNLOADED_GEOMETRY = "unloaded_geometry"


class ParameterIdentifiability(Enum):
    """Declared inference role, distinct from subsequently measured local rank."""

    PRIMARY = "primary"
    NUISANCE = "nuisance"
    FIXED = "fixed"
    CONDITIONAL = "conditional"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value != value.strip() or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lower-case token.")
    return value


def _shape(value: Sequence[int], /) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("shape must be a sequence of dimensions.")
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError("Parameter shape entries must be positive.")
    return resolved


class CardiacParameterSupport(StrictModule, NonTrainableState):
    """Closed physical support for one scalar or tensor parameter."""

    lower: Array
    upper: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        shape: Sequence[int] = (),
    ):
        shape_ = _shape(shape)
        lower_ = jax.lax.stop_gradient(
            jnp.broadcast_to(jnp.asarray(lower, dtype=float), shape_)
        )
        upper_ = jax.lax.stop_gradient(
            jnp.broadcast_to(jnp.asarray(upper, dtype=float), shape_)
        )
        if bool(jnp.any(jnp.isnan(lower_))) or bool(jnp.any(jnp.isnan(upper_))):
            raise ValueError("Parameter support bounds cannot be NaN.")
        if bool(jnp.any(lower_ >= upper_)):
            raise ValueError(
                "Every parameter support lower bound must be below its upper bound."
            )
        self.lower = lower_
        self.upper = upper_
        self.shape = shape_
        self.support_id = canonical_fingerprint(
            {
                "kind": "cardiac-parameter-support",
                "shape": list(shape_),
                "lower": array_tree_fingerprint(lower_),
                "upper": array_tree_fingerprint(upper_),
            }
        )

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    def contains(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != self.shape:
            raise ValueError(
                f"Parameter value must have shape {self.shape}; got {array.shape}."
            )
        return jnp.all(
            jnp.isfinite(array) & (array >= self.lower) & (array <= self.upper)
        )

    def validate(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value, dtype=float)
        if array.shape != self.shape:
            raise ValueError(
                f"Parameter value must have shape {self.shape}; got {array.shape}."
            )
        if not bool(self.contains(array)):
            raise ValueError(
                "Parameter value lies outside its declared physical support."
            )
        return array


class CardiacParameterSpec(StrictModule, NonTrainableState):
    """One physical parameter with native UQ transform and prior semantics."""

    name: str = eqx.field(static=True)
    quantity: CardiovascularQuantitySpec = eqx.field(static=True)
    transform: AbstractBijector
    support: CardiacParameterSupport
    prior: AbstractProbabilityLaw
    subsystem: CardiacSubsystem = eqx.field(static=True)
    identifiability: ParameterIdentifiability = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        quantity: CardiovascularQuantitySpec,
        transform: AbstractBijector,
        support: CardiacParameterSupport,
        prior: AbstractProbabilityLaw,
        subsystem: CardiacSubsystem,
        /,
        *,
        identifiability: ParameterIdentifiability = ParameterIdentifiability.PRIMARY,
    ):
        name_ = _identifier(name, "parameter name")
        if not isinstance(quantity, CardiovascularQuantitySpec):
            raise TypeError("quantity must be a CardiovascularQuantitySpec.")
        if not isinstance(transform, AbstractBijector):
            raise TypeError("transform must implement AbstractBijector.")
        if not isinstance(support, CardiacParameterSupport):
            raise TypeError("support must be a CardiacParameterSupport.")
        if not isinstance(prior, AbstractProbabilityLaw):
            raise TypeError("prior must implement AbstractProbabilityLaw.")
        if not isinstance(subsystem, CardiacSubsystem):
            raise TypeError("subsystem must be a CardiacSubsystem.")
        if not isinstance(identifiability, ParameterIdentifiability):
            raise TypeError("identifiability must be a ParameterIdentifiability.")
        shape_ = support.shape
        raw_shape = transform.inverse_shape(shape_)
        if transform.forward_shape(raw_shape) != shape_:
            raise ValueError("Parameter transform shape declarations are inconsistent.")
        prior_shape = tuple(prior.batch_shape) + tuple(prior.event_shape)
        if prior_shape and prior_shape != shape_:
            raise ValueError(
                "Parameter prior batch_shape + event_shape must match physical shape."
            )
        if prior.density_measure_kind != "lebesgue":
            raise ValueError("Deterministic cardiac parameters require a Lebesgue prior.")
        self.name = name_
        self.quantity = quantity
        self.transform = transform
        self.support = support
        self.prior = prior
        self.subsystem = subsystem
        self.identifiability = identifiability
        self.shape = shape_
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "cardiac-parameter-spec",
                "name": name_,
                "quantity": quantity.quantity_id,
                "transform_type": type(transform).__qualname__,
                "transform_arrays": array_tree_fingerprint(transform),
                "support": support.support_id,
                "prior_type": type(prior).__qualname__,
                "prior_arrays": array_tree_fingerprint(prior),
                "subsystem": subsystem.value,
                "identifiability": identifiability.value,
            }
        )

    @property
    def raw_shape(self) -> tuple[int, ...]:
        return self.transform.inverse_shape(self.shape)

    @property
    def size(self) -> int:
        return self.support.size

    def constrain(self, raw_value: ArrayLike, /) -> Array:
        raw = jnp.asarray(raw_value)
        if raw.shape != self.raw_shape:
            raise ValueError(
                f"Raw parameter {self.name!r} must have shape {self.raw_shape}; got {raw.shape}."
            )
        physical = self.transform.forward(raw)
        if physical.shape != self.shape:
            raise ValueError("Parameter transform returned an invalid physical shape.")
        return physical

    def unconstrain(self, physical_value: ArrayLike, /) -> Array:
        physical = self.support.validate(physical_value)
        raw = self.transform.inverse(physical)
        if raw.shape != self.raw_shape:
            raise ValueError("Parameter inverse transform returned an invalid raw shape.")
        return raw

    def log_prior(self, physical_value: ArrayLike, /) -> Array:
        physical = jnp.asarray(physical_value)
        if physical.shape != self.shape:
            raise ValueError(
                f"Physical parameter {self.name!r} must have shape {self.shape}."
            )
        support_ok = self.support.contains(physical)
        prior_ok = jnp.all(self.prior.contains(physical))
        density = jnp.sum(self.prior.log_prob(physical))
        return jnp.where(support_ok & prior_ok, density, -jnp.inf)


class CardiacParameterSchema(StrictModule, NonTrainableState):
    """Ordered, named parameter block used by exactly one inverse route."""

    fields: tuple[CardiacParameterSpec, ...]
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[CardiacParameterSpec],
        /,
        *,
        schema_id: str | None = None,
    ):
        resolved = tuple(fields)
        if not resolved:
            raise ValueError("CardiacParameterSchema requires at least one field.")
        if any(not isinstance(field, CardiacParameterSpec) for field in resolved):
            raise TypeError("Parameter schemas contain CardiacParameterSpec values.")
        names = tuple(field.name for field in resolved)
        if len(names) != len(set(names)):
            raise ValueError("Cardiac parameter names must be unique.")
        derived = canonical_fingerprint(
            {
                "kind": "cardiac-parameter-schema",
                "fields": [field.parameter_id for field in resolved],
            }
        )
        identifier = derived if schema_id is None else _identifier(schema_id, "schema_id")
        self.fields = resolved
        self.schema_id = identifier

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def physical_size(self) -> int:
        return sum(field.size for field in self.fields)

    @property
    def subsystems(self) -> frozenset[CardiacSubsystem]:
        return frozenset(field.subsystem for field in self.fields)

    @property
    def optimization_fields(self) -> tuple[CardiacParameterSpec, ...]:
        """Return only parameters that are permitted to move in an inverse solve."""

        return tuple(
            field
            for field in self.fields
            if field.identifiability is not ParameterIdentifiability.FIXED
        )

    @property
    def optimization_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.optimization_fields)

    @property
    def fixed_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, field in enumerate(self.fields)
            if field.identifiability is ParameterIdentifiability.FIXED
        )

    def validate_physical(self, values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        resolved = tuple(values)
        if len(resolved) != len(self.fields):
            raise ValueError("Physical parameter values must match schema field count.")
        return tuple(
            field.support.validate(value)
            for field, value in zip(self.fields, resolved, strict=True)
        )

    def unconstrain(self, physical_values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        physical = self.validate_physical(physical_values)
        return tuple(
            field.unconstrain(value)
            for field, value in zip(self.fields, physical, strict=True)
        )

    def constrain(self, raw_values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        raw = tuple(raw_values)
        if len(raw) != len(self.fields):
            raise ValueError("Raw parameter values must match schema field count.")
        return tuple(
            field.constrain(value) for field, value in zip(self.fields, raw, strict=True)
        )

    def constrain_optimization(
        self,
        raw_values: Sequence[ArrayLike],
        fixed_physical: Sequence[ArrayLike],
        /,
    ) -> tuple[Array, ...]:
        """Decode optimizer coordinates while injecting declared fixed values unchanged."""

        raw = tuple(raw_values)
        fixed = tuple(fixed_physical)
        expected = len(self.optimization_fields)
        if len(raw) != expected:
            raise ValueError(
                f"Optimizer coordinates must contain {expected} non-fixed fields."
            )
        if len(fixed) != len(self.fields):
            raise ValueError("fixed_physical must match the complete schema field count.")
        physical: list[Array] = []
        raw_index = 0
        for field, fixed_value in zip(self.fields, fixed, strict=True):
            if field.identifiability is ParameterIdentifiability.FIXED:
                value = jnp.asarray(fixed_value)
                if value.shape != field.shape:
                    raise ValueError(
                        f"Fixed parameter {field.name!r} must have shape {field.shape}."
                    )
                physical.append(value)
            else:
                physical.append(field.constrain(raw[raw_index]))
                raw_index += 1
        return tuple(physical)

    def contains(self, physical_values: Sequence[ArrayLike], /) -> Array:
        values = tuple(physical_values)
        if len(values) != len(self.fields):
            raise ValueError("Physical parameter values must match schema field count.")
        valid = jnp.asarray(True)
        for field, value in zip(self.fields, values, strict=True):
            valid = (
                valid
                & field.support.contains(value)
                & jnp.all(field.prior.contains(value))
            )
        return valid

    def log_prior(self, physical_values: Sequence[ArrayLike], /) -> Array:
        values = tuple(physical_values)
        if len(values) != len(self.fields):
            raise ValueError("Physical parameter values must match schema field count.")
        result = jnp.asarray(0.0)
        for field, value in zip(self.fields, values, strict=True):
            result = result + field.log_prior(value)
        return result

    def parameter_space(self, initial_physical: Sequence[ArrayLike], /) -> ParameterSpace:
        """Lower non-fixed coordinates to the authoritative native UQ space."""

        physical = self.validate_physical(initial_physical)
        fields = self.optimization_fields
        if not fields:
            raise ValueError(
                "An inverse schema requires at least one non-fixed parameter."
            )
        raw = tuple(
            field.unconstrain(value)
            for field, value in zip(self.fields, physical, strict=True)
            if field.identifiability is not ParameterIdentifiability.FIXED
        )
        priors = tuple(field.prior for field in fields)
        bijectors = tuple(field.transform for field in fields)
        return ParameterSpace(raw, priors=priors, bijectors=bijectors)

    def physical_vector(self, values: Sequence[ArrayLike], /) -> Array:
        physical = self.validate_physical(values)
        return jnp.concatenate(tuple(value.reshape(-1) for value in physical))


__all__ = [
    "CardiacParameterSchema",
    "CardiacParameterSpec",
    "CardiacParameterSupport",
    "CardiacSubsystem",
    "ParameterIdentifiability",
]
