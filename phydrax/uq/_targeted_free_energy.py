#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._posterior import AbstractBijector


class ReducedPotentialEvaluation(StrictModule):
    value: Array
    valid: Array
    potential_id: str = eqx.field(static=True)


class AbstractReducedPotential(StrictModule, NonTrainableState):
    event_shape: AbstractAttribute[tuple[int, ...]]
    potential_id: AbstractAttribute[str]

    @abstractmethod
    def evaluate(self, value: ArrayLike, /) -> ReducedPotentialEvaluation:
        raise NotImplementedError


class CallableReducedPotential(AbstractReducedPotential):
    function: Callable[[Array], ArrayLike] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array], ArrayLike],
        event_shape: tuple[int, ...],
        potential_id: str,
        /,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        shape = tuple(int(size) for size in event_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("event_shape must contain positive dimensions.")
        identifier = str(potential_id).strip()
        if not identifier:
            raise ValueError("potential_id must be non-empty.")
        self.function = function
        self.event_shape = shape
        self.potential_id = identifier

    def evaluate(self, value: ArrayLike, /) -> ReducedPotentialEvaluation:
        array = jnp.asarray(value)
        if array.shape != self.event_shape:
            raise ValueError(f"value must have event shape {self.event_shape}.")
        potential = jnp.asarray(self.function(array)).reshape(())
        valid = jnp.isfinite(potential) & jnp.all(jnp.isfinite(array))
        return ReducedPotentialEvaluation(potential, valid, self.potential_id)


class TargetedMapPlan(StrictModule):
    bijector: AbstractBijector
    event_shape: tuple[int, ...] = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    parameter_state_id: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        bijector: AbstractBijector,
        event_shape: tuple[int, ...],
        /,
        *,
        architecture_id: str,
    ):
        if not isinstance(bijector, AbstractBijector):
            raise TypeError("bijector must implement AbstractBijector.")
        shape = tuple(int(size) for size in event_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("event_shape must contain positive dimensions.")
        if (
            bijector.forward_shape(shape) != shape
            or bijector.inverse_shape(shape) != shape
        ):
            raise ValueError("Targeted maps must preserve one fixed event shape.")
        architecture = str(architecture_id).strip()
        if not architecture:
            raise ValueError("architecture_id must be non-empty.")
        parameter_state = canonical_fingerprint(
            {
                "kind": "targeted-map-parameters",
                "arrays": array_tree_fingerprint(bijector),
            }
        )
        self.bijector = bijector
        self.event_shape = shape
        self.architecture_id = architecture
        self.parameter_state_id = parameter_state
        self.map_id = canonical_fingerprint(
            {
                "kind": "targeted-map",
                "architecture": architecture,
                "parameters": parameter_state,
                "event_shape": list(shape),
                "jacobian": "exact",
            }
        )

    def with_bijector(self, bijector: AbstractBijector, /) -> "TargetedMapPlan":
        return TargetedMapPlan(
            bijector, self.event_shape, architecture_id=self.architecture_id
        )

    def forward(self, value: ArrayLike, /) -> tuple[Array, Array]:
        array = jnp.asarray(value)
        if array.shape != self.event_shape:
            raise ValueError(f"value must have event shape {self.event_shape}.")
        mapped = self.bijector.forward(array)
        logdet = jnp.sum(self.bijector.forward_log_det_jacobian(array))
        return mapped, logdet

    def inverse(self, value: ArrayLike, /) -> tuple[Array, Array]:
        array = jnp.asarray(value)
        if array.shape != self.event_shape:
            raise ValueError(f"value must have event shape {self.event_shape}.")
        source = self.bijector.inverse(array)
        logdet = -jnp.sum(self.bijector.forward_log_det_jacobian(source))
        return source, logdet


class TargetedFreeEnergyProblem(StrictModule, NonTrainableState):
    source: AbstractReducedPotential
    target: AbstractReducedPotential
    mapping: TargetedMapPlan
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AbstractReducedPotential,
        target: AbstractReducedPotential,
        mapping: TargetedMapPlan,
        /,
    ):
        if not isinstance(source, AbstractReducedPotential) or not isinstance(
            target, AbstractReducedPotential
        ):
            raise TypeError("source and target must implement AbstractReducedPotential.")
        if not isinstance(mapping, TargetedMapPlan):
            raise TypeError("mapping must be TargetedMapPlan.")
        if (
            source.event_shape != target.event_shape
            or source.event_shape != mapping.event_shape
        ):
            raise ValueError("Reduced potentials and map must share one event shape.")
        self.source = source
        self.target = target
        self.mapping = mapping
        self.problem_id = canonical_fingerprint(
            {
                "kind": "targeted-free-energy-problem",
                "source": source.potential_id,
                "target": target.potential_id,
                "mapping": mapping.map_id,
            }
        )


class TargetedWorkEvaluation(StrictModule):
    mapped_source: Array
    mapped_target: Array | None
    forward_work: Array
    reverse_work: Array | None
    forward_log_determinant: Array
    reverse_log_determinant: Array | None
    forward_roundtrip_residual: Array
    reverse_roundtrip_residual: Array | None
    forward_valid: Array
    reverse_valid: Array | None
    valid: Array
    problem_id: str = eqx.field(static=True)


def _evaluate_forward(problem: TargetedFreeEnergyProblem, value: Array, /):
    mapped, logdet = problem.mapping.forward(value)
    source = problem.source.evaluate(value)
    target = problem.target.evaluate(mapped)
    reconstructed, _ = problem.mapping.inverse(mapped)
    residual = jnp.sqrt(jnp.sum((reconstructed - value) ** 2))
    work = target.value - source.value - logdet
    valid = (
        source.valid
        & target.valid
        & jnp.isfinite(logdet)
        & jnp.isfinite(work)
        & jnp.isfinite(residual)
    )
    return mapped, work, logdet, residual, valid


def _evaluate_reverse(problem: TargetedFreeEnergyProblem, value: Array, /):
    mapped, logdet = problem.mapping.inverse(value)
    target = problem.target.evaluate(value)
    source = problem.source.evaluate(mapped)
    reconstructed, _ = problem.mapping.forward(mapped)
    residual = jnp.sqrt(jnp.sum((reconstructed - value) ** 2))
    work = source.value - target.value - logdet
    valid = (
        source.valid
        & target.valid
        & jnp.isfinite(logdet)
        & jnp.isfinite(work)
        & jnp.isfinite(residual)
    )
    return mapped, work, logdet, residual, valid


def evaluate_targeted_work(
    problem: TargetedFreeEnergyProblem,
    source_samples: ArrayLike,
    /,
    *,
    target_samples: ArrayLike | None = None,
) -> TargetedWorkEvaluation:
    if not isinstance(problem, TargetedFreeEnergyProblem):
        raise TypeError("problem must be TargetedFreeEnergyProblem.")
    source = jnp.asarray(source_samples)
    if source.ndim < 1 or tuple(source.shape[1:]) != problem.mapping.event_shape:
        raise ValueError("source_samples must have shape (sample,) + event_shape.")
    mapped_source, forward_work, forward_logdet, forward_residual, forward_valid = (
        jax.vmap(lambda value: _evaluate_forward(problem, value))(source)
    )
    if target_samples is None:
        mapped_target = None
        reverse_work = None
        reverse_logdet = None
        reverse_residual = None
        reverse_valid = None
        valid = jnp.all(forward_valid)
    else:
        target = jnp.asarray(target_samples)
        if target.ndim < 1 or tuple(target.shape[1:]) != problem.mapping.event_shape:
            raise ValueError("target_samples must have shape (sample,) + event_shape.")
        mapped_target, reverse_work, reverse_logdet, reverse_residual, reverse_valid = (
            jax.vmap(lambda value: _evaluate_reverse(problem, value))(target)
        )
        valid = jnp.all(forward_valid) & jnp.all(reverse_valid)
    return TargetedWorkEvaluation(
        mapped_source=mapped_source,
        mapped_target=mapped_target,
        forward_work=forward_work,
        reverse_work=reverse_work,
        forward_log_determinant=forward_logdet,
        reverse_log_determinant=reverse_logdet,
        forward_roundtrip_residual=forward_residual,
        reverse_roundtrip_residual=reverse_residual,
        forward_valid=forward_valid,
        reverse_valid=reverse_valid,
        valid=valid,
        problem_id=problem.problem_id,
    )


__all__ = [
    "AbstractReducedPotential",
    "CallableReducedPotential",
    "ReducedPotentialEvaluation",
    "TargetedFreeEnergyProblem",
    "TargetedMapPlan",
    "TargetedWorkEvaluation",
    "evaluate_targeted_work",
]
