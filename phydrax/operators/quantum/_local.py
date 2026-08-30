#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import AbstractAttribute, StrictModule
from ._amplitude import LogAmplitude


class LocalOperatorStatus(IntEnum):
    """Portable status for a per-configuration local quantum-operator action."""

    SUCCESS = 0
    INVALID_AMPLITUDE = 1
    SINGULAR_CONFIGURATION = 2
    NONFINITE = 3


class LocalOperatorEstimate(StrictModule):
    """Local values with validity, status, work, precision, and method evidence.

    ``work_count`` records the number of operator-defined primitive actions used
    for each configuration. Its meaning is fixed by ``method_id``; it is not a
    dimension-independent cost claim.
    """

    value: Array
    valid: Array
    status: Array
    work_count: Array
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        work_count: ArrayLike,
        /,
        *,
        configuration_shape: Sequence[int],
        operator_id: str,
        method_id: str,
        compute_dtype: str,
    ):
        values = jnp.asarray(value)
        validity = jnp.asarray(valid, dtype=bool)
        statuses = jnp.asarray(status, dtype=jnp.int32)
        work = jnp.asarray(work_count, dtype=jnp.int32)
        if validity.shape != values.shape:
            raise ValueError("valid must match the local value shape.")
        if statuses.shape != values.shape:
            raise ValueError("status must match the local value shape.")
        if work.shape != values.shape:
            raise ValueError("work_count must match the local value shape.")
        shape = tuple(int(size) for size in configuration_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("configuration_shape must contain positive dimensions.")
        identifiers = (str(operator_id), str(method_id), str(compute_dtype))
        if any(not value for value in identifiers):
            raise ValueError("operator_id, method_id, and compute_dtype must be non-empty.")
        self.value = values
        self.valid = validity
        self.status = statuses
        self.work_count = work
        self.configuration_shape = shape
        self.operator_id, self.method_id, self.compute_dtype = identifiers

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(LocalOperatorStatus.SUCCESS))


class AbstractLocalQuantumOperator(StrictModule):
    """Matrix-free quantum operator capable of evaluating ``(H psi) / psi``."""

    configuration_shape: AbstractAttribute[tuple[int, ...]]
    operator_id: AbstractAttribute[str]

    @abstractmethod
    def estimate(
        self,
        model: Callable[[Array], LogAmplitude],
        configurations: Array,
        /,
    ) -> LocalOperatorEstimate:
        raise NotImplementedError


def evaluate_local_operator(
    model: Callable[[Array], LogAmplitude],
    operator: AbstractLocalQuantumOperator,
    configurations: ArrayLike,
    /,
) -> LocalOperatorEstimate:
    """Evaluate one local operator through its configuration-native algorithm."""
    if not callable(model):
        raise TypeError("model must be callable.")
    if not isinstance(operator, AbstractLocalQuantumOperator):
        raise TypeError("operator must implement AbstractLocalQuantumOperator.")
    values = jnp.asarray(configurations)
    rank = len(operator.configuration_shape)
    if values.ndim < rank or tuple(values.shape[-rank:]) != operator.configuration_shape:
        raise ValueError(
            "configurations must end in shape "
            f"{operator.configuration_shape}; got {values.shape}."
        )
    result = operator.estimate(model, values)
    if not isinstance(result, LocalOperatorEstimate):
        raise TypeError("A local quantum operator must return LocalOperatorEstimate.")
    batch_shape = tuple(int(size) for size in values.shape[:-rank])
    if result.value.shape != batch_shape:
        raise ValueError(
            f"Local estimate must have batch shape {batch_shape}; "
            f"got {result.value.shape}."
        )
    if result.configuration_shape != operator.configuration_shape:
        raise ValueError("Local estimate configuration shape does not match its operator.")
    if result.operator_id != operator.operator_id:
        raise ValueError("Local estimate operator identity does not match its operator.")
    return result


__all__ = [
    "AbstractLocalQuantumOperator",
    "LocalOperatorEstimate",
    "LocalOperatorStatus",
    "evaluate_local_operator",
]
