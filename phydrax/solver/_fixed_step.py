#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._numerics._ssp_runge_kutta import ssprk33_step, ssprk54_step
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization import DiscretizationBundle
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry


def _canonical_structured_state(state: Any, /) -> PyTree[Array]:
    leaves, treedef = jax.tree.flatten(state)
    if not leaves:
        raise ValueError("Fixed-step initial_state must contain array leaves.")
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError("Every structured fixed-step state leaf must be an array.")
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    if not any(jnp.issubdtype(array.dtype, jnp.inexact) for array in arrays):
        raise TypeError(
            "Structured fixed-step initial_state requires at least one inexact leaf."
        )
    return jax.tree.unflatten(treedef, arrays)


def _state_dtype(state: PyTree[Array], /):
    dtypes = tuple(
        leaf.dtype
        for leaf in jax.tree.leaves(state)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    if not dtypes:
        raise TypeError("Fixed-step state requires at least one inexact leaf.")
    return jnp.result_type(*dtypes)


def _validate_result_state(
    role: str, candidate: PyTree[Array], reference: PyTree[Array], /
) -> None:
    if jax.tree.structure(candidate) != jax.tree.structure(reference):
        raise ValueError(f"Fixed-step {role} must preserve the state PyTree structure.")
    for proposed, current in zip(
        jax.tree.leaves(candidate), jax.tree.leaves(reference), strict=True
    ):
        if not eqx.is_array(proposed):
            raise TypeError(f"Every fixed-step {role} leaf must be an array.")
        if proposed.shape != current.shape or proposed.dtype != current.dtype:
            raise ValueError(
                f"Fixed-step {role} must preserve every state leaf shape and dtype."
            )


def _validate_scalar_result(role: str, value: Any, /, *, boolean: bool = False) -> None:
    if not eqx.is_array(value) or value.shape != ():
        raise TypeError(f"Fixed-step {role} must be a scalar array.")
    if boolean and value.dtype != jnp.dtype(bool):
        raise TypeError(f"Fixed-step {role} must be Boolean.")


def _prepend_initial_state(
    initial: PyTree[Array], states: PyTree[Array], /
) -> PyTree[Array]:
    return jax.tree.map(
        lambda first, rest: jnp.concatenate((first[None, ...], rest), axis=0),
        initial,
        states,
    )


def _take_saved_states(states: PyTree[Array], indices: Array, /) -> PyTree[Array]:
    return jax.tree.map(lambda leaf: leaf[indices], states)


class AcceptedStepTransformResult(StrictModule):
    transformed_state: Array
    applied: Array
    successful: Array
    correction_norm: Array


class AbstractAcceptedStepTransform(StrictModule, NonTrainableState):
    transform_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        raise NotImplementedError


class IdentityAcceptedStepTransform(AbstractAcceptedStepTransform):
    transform_id: str = "accepted-step-transform:identity"

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        del step_index, time, previous_state, args
        return AcceptedStepTransformResult(
            candidate_state,
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.zeros((), dtype=candidate_state.dtype),
        )


class CompositeAcceptedStepTransform(AbstractAcceptedStepTransform):
    transforms: tuple[AbstractAcceptedStepTransform, ...]
    transform_id: str = eqx.field(static=True)

    def __init__(self, transforms: Sequence[AbstractAcceptedStepTransform], /):
        values = tuple(transforms)
        if any(not isinstance(value, AbstractAcceptedStepTransform) for value in values):
            raise TypeError("Every transform must be an AbstractAcceptedStepTransform.")
        self.transforms = values
        self.transform_id = canonical_fingerprint(
            {
                "kind": "composite-accepted-step-transform",
                "transforms": [value.transform_id for value in values],
            }
        )

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        state = candidate_state
        applied = jnp.asarray(False)
        successful = jnp.asarray(True)
        correction = jnp.zeros((), dtype=candidate_state.dtype)
        for transform in self.transforms:
            result = transform.apply(step_index, time, previous_state, state, args)
            state = jnp.where(result.successful, result.transformed_state, state)
            applied = applied | result.applied
            successful = successful & result.successful
            correction = correction + result.correction_norm
        return AcceptedStepTransformResult(state, applied, successful, correction)


class FixedStepResult(StrictModule):
    candidate_state: PyTree[Array]
    accepted_state: PyTree[Array]
    successful: Array
    residual: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array


class AbstractFixedStepMethod(StrictModule, NonTrainableState):
    method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        raise NotImplementedError


class CallableFixedStepMethod(AbstractFixedStepMethod):
    step_function: Callable[[Array, Array, PyTree[Array], Array, Any], FixedStepResult]
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_function: Callable[
            [Array, Array, PyTree[Array], Array, Any], FixedStepResult
        ],
        method_id: str,
        /,
    ):
        if not callable(step_function):
            raise TypeError("step_function must be callable.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.step_function = step_function
        self.method_id = identifier

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.step_function(step_index, time, state, step_size, args)
        if not isinstance(result, FixedStepResult):
            raise TypeError("step_function must return FixedStepResult.")
        return result


class AbstractSSPRKFixedStepMethod(AbstractFixedStepMethod):
    vector_field: Callable[[Array, Array, Any], Array]
    transform: AbstractAcceptedStepTransform
    order: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        order: int,
        transform: AbstractAcceptedStepTransform | None = None,
    ):
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if order not in (3, 4):
            raise ValueError("Fixed-step SSPRK order must be 3 or 4.")
        transform_ = IdentityAcceptedStepTransform() if transform is None else transform
        if not isinstance(transform_, AbstractAcceptedStepTransform):
            raise TypeError("transform must be an AbstractAcceptedStepTransform or None.")
        self.vector_field = vector_field
        self.transform = transform_
        self.order = int(order)
        self.method_id = canonical_fingerprint(
            {
                "kind": "fixed-step-ssprk",
                "order": order,
                "transform": transform_.transform_id,
            }
        )

    @abc.abstractmethod
    def _advance(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> Array:
        raise NotImplementedError

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        candidate = self._advance(time, state, step_size, args)
        transformed = self.transform.apply(
            step_index, time + step_size, state, candidate, args
        )
        successful = transformed.successful & jnp.all(
            jnp.isfinite(transformed.transformed_state)
        )
        accepted = jnp.where(successful, transformed.transformed_state, state)
        return FixedStepResult(
            candidate,
            accepted,
            successful,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(self.order, dtype=jnp.int32),
            transformed.applied,
            transformed.correction_norm,
        )


class SSPRK33FixedStepMethod(AbstractSSPRKFixedStepMethod):
    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        transform: AbstractAcceptedStepTransform | None = None,
    ):
        super().__init__(vector_field, order=3, transform=transform)

    def _advance(self, time, state, step_size, args, /):
        return ssprk33_step(self.vector_field, time, state, step_size, args)


class SSPRK54FixedStepMethod(AbstractSSPRKFixedStepMethod):
    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        transform: AbstractAcceptedStepTransform | None = None,
    ):
        super().__init__(vector_field, order=4, transform=transform)

    def _advance(self, time, state, step_size, args, /):
        return ssprk54_step(self.vector_field, time, state, step_size, args)


class FixedStepProblem(StrictModule, NonTrainableState):
    method: AbstractFixedStepMethod
    initial_state: PyTree[Array]
    args: Any
    state_geometry: AbstractStateGeometry
    discretization_bundle: DiscretizationBundle | None
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        initial_state: Any,
        /,
        *,
        t0: float,
        t1: float,
        step_size: float,
        args: Any = None,
        state_geometry: AbstractStateGeometry | None = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(method, AbstractFixedStepMethod):
            raise TypeError("method must be an AbstractFixedStepMethod.")
        if state_geometry is None:
            initial = jnp.asarray(initial_state)
            if not jnp.issubdtype(initial.dtype, jnp.inexact):
                raise TypeError("Fixed-step initial_state must have an inexact dtype.")
        else:
            initial = _canonical_structured_state(initial_state)
        start = float(t0)
        end = float(t1)
        step = float(step_size)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Fixed-step times require finite t1 > t0.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        raw_steps = (end - start) / step
        count = int(round(raw_steps))
        if count <= 0 or not np.isclose(raw_steps, count, rtol=1e-12, atol=1e-12):
            raise ValueError("Fixed-step interval must contain an integer step count.")
        geometry = EuclideanStateGeometry() if state_geometry is None else state_geometry
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("state_geometry must be an AbstractStateGeometry or None.")
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        state_payload = (
            {
                "state_shape": list(initial.shape),
                "state_dtype": str(initial.dtype),
            }
            if eqx.is_array(initial)
            else {"state_tree": array_tree_signature(initial)}
        )
        generated = canonical_fingerprint(
            {
                "kind": "fixed-step-problem",
                "method": method.method_id,
                **state_payload,
                "t0": start,
                "t1": end,
                "step_size": step,
                "geometry": geometry.geometry_id,
                "bundle": None
                if discretization_bundle is None
                else discretization_bundle.bundle_id,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.method = method
        self.initial_state = initial
        self.args = args
        self.state_geometry = geometry
        self.discretization_bundle = discretization_bundle
        self.t0 = start
        self.t1 = end
        self.step_size = step
        self.step_count = count
        self.problem_id = identifier


class FixedStepSolution(StrictModule, NonTrainableState):
    times: Array
    states: PyTree[Array]
    valid: Array
    successful: Array
    residuals: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)


def solve_fixed_step(
    problem: FixedStepProblem,
    /,
    *,
    save_every: int = 1,
) -> FixedStepSolution:
    """Run one pure fixed-step scan with fail-closed accepted states."""

    if not isinstance(problem, FixedStepProblem):
        raise TypeError("problem must be a FixedStepProblem.")
    stride = int(save_every)
    if stride <= 0:
        raise ValueError("save_every must be positive.")
    state_dtype = _state_dtype(problem.initial_state)
    step_size = jnp.asarray(problem.step_size, dtype=state_dtype)

    def advance(carry, step_index):
        state, previous_success = carry
        time = jnp.asarray(problem.t0, dtype=state_dtype) + step_index * step_size
        result = problem.method.step(step_index, time, state, step_size, problem.args)
        _validate_result_state("candidate_state", result.candidate_state, state)
        _validate_result_state("accepted_state", result.accepted_state, state)
        _validate_scalar_result("successful", result.successful, boolean=True)
        _validate_scalar_result("residual", result.residual)
        _validate_scalar_result("iterations", result.iterations)
        _validate_scalar_result("work", result.work)
        _validate_scalar_result(
            "transform_applied", result.transform_applied, boolean=True
        )
        _validate_scalar_result(
            "transform_correction_norm", result.transform_correction_norm
        )
        accepted = tree_where(previous_success, result.accepted_state, state)
        successful = previous_success & result.successful
        payload = (
            accepted,
            successful,
            result.residual,
            result.iterations,
            result.work,
            result.transform_applied,
            result.transform_correction_norm,
        )
        return (accepted, successful), payload

    indices = jnp.arange(problem.step_count, dtype=jnp.int32)
    (_, final_success), payload = jax.lax.scan(
        advance,
        (problem.initial_state, jnp.asarray(True)),
        indices,
    )
    states, valid, residuals, iterations, work, transformed, correction = payload
    all_states = _prepend_initial_state(problem.initial_state, states)
    all_valid = jnp.concatenate((jnp.asarray([True]), valid), axis=0)
    all_times = jnp.asarray(problem.t0, dtype=step_size.dtype) + step_size * jnp.arange(
        problem.step_count + 1
    )
    save_indices = jnp.arange(0, problem.step_count + 1, stride, dtype=jnp.int32)
    if int(save_indices[-1]) != problem.step_count:
        save_indices = jnp.concatenate(
            (save_indices, jnp.asarray([problem.step_count], dtype=jnp.int32))
        )
    bundle_id = (
        None
        if problem.discretization_bundle is None
        else problem.discretization_bundle.bundle_id
    )
    return FixedStepSolution(
        all_times[save_indices],
        _take_saved_states(all_states, save_indices),
        all_valid[save_indices],
        final_success,
        residuals,
        iterations,
        work,
        transformed,
        correction,
        problem.problem_id,
        problem.method.method_id,
        problem.state_geometry.geometry_id,
        bundle_id,
    )


__all__ = [
    "AbstractAcceptedStepTransform",
    "CallableFixedStepMethod",
    "AbstractFixedStepMethod",
    "AcceptedStepTransformResult",
    "CompositeAcceptedStepTransform",
    "FixedStepProblem",
    "FixedStepResult",
    "FixedStepSolution",
    "IdentityAcceptedStepTransform",
    "SSPRK33FixedStepMethod",
    "SSPRK54FixedStepMethod",
    "solve_fixed_step",
]
