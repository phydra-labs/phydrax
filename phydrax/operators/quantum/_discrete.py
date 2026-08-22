#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import AbstractAttribute, StrictModule
from ._amplitude import amplitude_ratio, LogAmplitude


class ConnectedConfigurations(StrictModule):
    """Fixed-capacity connected configurations and ``H[current, connected]`` values."""

    configurations: Array
    matrix_elements: Array
    valid: Array
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    max_connections: int = eqx.field(static=True)

    def __init__(
        self,
        configurations: ArrayLike,
        matrix_elements: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        configuration_shape: Sequence[int],
    ):
        shape = tuple(int(size) for size in configuration_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("configuration_shape must contain positive dimensions.")
        configs = jnp.asarray(configurations)
        elements = jnp.asarray(matrix_elements)
        mask = jnp.asarray(valid, dtype=bool)
        if elements.ndim < 1 or int(elements.shape[-1]) < 1:
            raise ValueError(
                "Connected configurations require a nonempty connection axis."
            )
        if mask.shape != elements.shape:
            raise ValueError("valid must match matrix_elements shape.")
        expected = elements.shape + shape
        if configs.shape != expected:
            raise ValueError(
                "configurations must have shape batch + (connection,) + "
                f"configuration_shape; expected {expected}, got {configs.shape}."
            )
        self.configurations = configs
        self.matrix_elements = elements
        self.valid = mask
        self.configuration_shape = shape
        self.max_connections = int(elements.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.matrix_elements.shape[:-1])


class AbstractDiscreteQuantumOperator(StrictModule):
    """Matrix-free discrete operator exposing diagonal and connected configurations."""

    configuration_shape: AbstractAttribute[tuple[int, ...]]
    operator_id: AbstractAttribute[str]

    @abstractmethod
    def diagonal(self, configurations: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def connections(self, configurations: Array, /) -> ConnectedConfigurations:
        raise NotImplementedError


class CallableDiscreteQuantumOperator(AbstractDiscreteQuantumOperator):
    """Validated callable implementation of a connected discrete operator."""

    diagonal_fn: Callable[[Array], ArrayLike] = eqx.field(static=True)
    connections_fn: Callable[[Array], ConnectedConfigurations] = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagonal: Callable[[Array], ArrayLike],
        connections: Callable[[Array], ConnectedConfigurations],
        /,
        *,
        configuration_shape: Sequence[int],
        operator_id: str,
    ):
        if not callable(diagonal) or not callable(connections):
            raise TypeError("diagonal and connections must be callable.")
        shape = tuple(int(size) for size in configuration_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("configuration_shape must contain positive dimensions.")
        if not isinstance(operator_id, str) or not operator_id:
            raise ValueError("operator_id must be non-empty.")
        self.diagonal_fn = diagonal
        self.connections_fn = connections
        self.configuration_shape = shape
        self.operator_id = operator_id

    def _batch_shape(self, configurations: Array, /) -> tuple[int, ...]:
        values = jnp.asarray(configurations)
        rank = len(self.configuration_shape)
        if values.ndim < rank or tuple(values.shape[-rank:]) != self.configuration_shape:
            raise ValueError(
                "configurations must end in shape "
                f"{self.configuration_shape}; got {values.shape}."
            )
        return tuple(int(size) for size in values.shape[:-rank])

    def diagonal(self, configurations: Array, /) -> Array:
        batch_shape = self._batch_shape(configurations)
        values = jnp.asarray(self.diagonal_fn(configurations))
        if values.shape != batch_shape:
            raise ValueError(
                f"diagonal must return shape {batch_shape}; got {values.shape}."
            )
        return values

    def connections(self, configurations: Array, /) -> ConnectedConfigurations:
        batch_shape = self._batch_shape(configurations)
        result = self.connections_fn(configurations)
        if not isinstance(result, ConnectedConfigurations):
            raise TypeError("connections must return ConnectedConfigurations.")
        if result.configuration_shape != self.configuration_shape:
            raise ValueError("Connected configuration shape does not match the operator.")
        if result.batch_shape != batch_shape:
            raise ValueError(
                f"Connected batch shape must be {batch_shape}; got {result.batch_shape}."
            )
        return result


class LocalEstimate(StrictModule):
    """Per-configuration local operator value and connection validity evidence."""

    value: Array
    valid: Array
    active_connections: Array


def _evaluate_amplitudes(model: Callable[[Array], LogAmplitude], configs: Array):
    values = jax.vmap(model)(configs)
    if not isinstance(values, LogAmplitude):
        raise TypeError("The amplitude model must return LogAmplitude values.")
    return values


def local_estimate(
    model: Callable[[Array], LogAmplitude],
    operator: AbstractDiscreteQuantumOperator,
    configurations: ArrayLike,
    /,
) -> LocalEstimate:
    """Evaluate ``sum_x' H[x, x'] psi(x') / psi(x)`` without a dense matrix."""
    if not callable(model):
        raise TypeError("model must be callable.")
    if not isinstance(operator, AbstractDiscreteQuantumOperator):
        raise TypeError("operator must implement AbstractDiscreteQuantumOperator.")
    configs = jnp.asarray(configurations)
    rank = len(operator.configuration_shape)
    if (
        configs.ndim < rank
        or tuple(configs.shape[-rank:]) != operator.configuration_shape
    ):
        raise ValueError(
            "configurations must end in shape "
            f"{operator.configuration_shape}; got {configs.shape}."
        )
    batch_shape = tuple(int(size) for size in configs.shape[:-rank])
    batch_count = prod(batch_shape) if batch_shape else 1
    flat_configs = configs.reshape((batch_count,) + operator.configuration_shape)
    current = _evaluate_amplitudes(model, flat_configs)
    diagonal = operator.diagonal(configs).reshape((batch_count,))
    connected = operator.connections(configs)
    flat_connected = connected.configurations.reshape(
        (batch_count * connected.max_connections,) + operator.configuration_shape
    )
    proposed_flat = _evaluate_amplitudes(model, flat_connected)
    proposed = LogAmplitude(
        proposed_flat.log_abs.reshape((batch_count, connected.max_connections)),
        proposed_flat.phase.reshape((batch_count, connected.max_connections)),
        valid=proposed_flat.valid.reshape((batch_count, connected.max_connections)),
    )
    amplitude_shape = (batch_count, connected.max_connections)
    current_expanded = LogAmplitude(
        jnp.broadcast_to(current.log_abs[:, None], amplitude_shape),
        jnp.broadcast_to(current.phase[:, None], amplitude_shape),
        valid=jnp.broadcast_to(current.valid[:, None], amplitude_shape),
    )
    ratios = amplitude_ratio(proposed, current_expanded)
    matrix_elements = connected.matrix_elements.reshape(
        (batch_count, connected.max_connections)
    )
    connection_mask = connected.valid.reshape((batch_count, connected.max_connections))
    finite_elements = jnp.isfinite(matrix_elements)
    active_valid = ratios.valid & finite_elements
    invalid_active = jnp.any(connection_mask & ~active_valid, axis=-1)
    safe_mask = connection_mask & active_valid
    safe_elements = jnp.where(
        safe_mask, matrix_elements, jnp.zeros((), dtype=matrix_elements.dtype)
    )
    safe_ratios = jnp.where(
        safe_mask, ratios.value, jnp.zeros((), dtype=ratios.value.dtype)
    )
    terms = safe_elements * safe_ratios
    values = diagonal + jnp.sum(terms, axis=-1)
    diagonal_valid = jnp.isfinite(diagonal)
    valid = current.valid & current.nonzero & diagonal_valid & ~invalid_active
    values = jnp.where(valid, values, jnp.asarray(jnp.nan, dtype=values.dtype))
    return LocalEstimate(
        value=values.reshape(batch_shape),
        valid=valid.reshape(batch_shape),
        active_connections=jnp.sum(connection_mask, axis=-1, dtype=jnp.int32).reshape(
            batch_shape
        ),
    )


__all__ = [
    "AbstractDiscreteQuantumOperator",
    "CallableDiscreteQuantumOperator",
    "ConnectedConfigurations",
    "LocalEstimate",
    "local_estimate",
]
