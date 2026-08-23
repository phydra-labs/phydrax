#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import validate_real_inexact_tree
from ..linalg import PyTreeSpace
from ._bounds import ProjectedLBFGS
from ._iterative import (
    AbstractMinimizationMethod,
    MinimizationProblem,
    MinimizationResult,
    OptimizationTermination,
)


StartGenerator: TypeAlias = Literal["uniform-bounds", "normal"]


class MultiStartPolicy(StrictModule):
    local_method: AbstractMinimizationMethod
    count: int = eqx.field(static=True)
    generator: StartGenerator = eqx.field(static=True)
    normal_scale: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        local_method: AbstractMinimizationMethod | None = None,
        count: int = 16,
        generator: StartGenerator = "uniform-bounds",
        normal_scale: float = 1.0,
        seed: int = 0,
    ):
        method = ProjectedLBFGS() if local_method is None else local_method
        if not isinstance(method, AbstractMinimizationMethod):
            raise TypeError("local_method must be AbstractMinimizationMethod or None.")
        count_ = int(count)
        scale = float(normal_scale)
        if count_ < 1:
            raise ValueError("Multi-start count must be positive.")
        if generator not in ("uniform-bounds", "normal"):
            raise ValueError("Unknown multi-start generator.")
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("normal_scale must be finite and positive.")
        self.local_method = method
        self.count = count_
        self.generator = generator
        self.normal_scale = scale
        self.seed = int(seed)


class MultiStartResult(StrictModule):
    best: MinimizationResult
    initial_points: Array
    final_points: Array
    objectives: Array
    statuses: Array
    certified: Array
    best_index: Array
    attempted: Array

    @property
    def successful(self):
        return self.best.successful


def _starts(
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    policy: MultiStartPolicy,
    /,
) -> tuple[PyTreeSpace, Array]:
    initial = validate_real_inexact_tree(initial_parameters, name="initial_parameters")
    space = PyTreeSpace(initial)
    center = space.flatten(initial)
    key = jax.random.PRNGKey(policy.seed)
    if policy.generator == "uniform-bounds":
        if problem.bounds is None:
            raise ValueError("uniform-bounds starts require problem.bounds.")
        lower, upper = problem.bounds.materialize(initial)
        lower_coordinates = space.flatten(lower)
        upper_coordinates = space.flatten(upper)
        if not bool(
            jnp.all(jnp.isfinite(lower_coordinates))
            & jnp.all(jnp.isfinite(upper_coordinates))
        ):
            raise ValueError("uniform-bounds starts require finite bounds.")
        random = jax.random.uniform(
            key,
            (policy.count - 1, space.size),
            dtype=center.dtype,
        )
        generated = lower_coordinates + random * (upper_coordinates - lower_coordinates)
    else:
        generated = center[None, :] + policy.normal_scale * jax.random.normal(
            key,
            (policy.count - 1, space.size),
            dtype=center.dtype,
        )
        if problem.bounds is not None:
            generated = jax.vmap(
                lambda value: space.flatten(
                    problem.bounds.project(space.unflatten(value))
                )
            )(generated)
    return space, jnp.concatenate([center[None, :], generated], axis=0)


def multistart_minimize(
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    policy: MultiStartPolicy | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> MultiStartResult:
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be MinimizationProblem.")
    policy_ = MultiStartPolicy() if policy is None else policy
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(policy_, MultiStartPolicy):
        raise TypeError("policy must be MultiStartPolicy or None.")
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    space, starts = _starts(problem, initial_parameters, policy_)
    per_steps = max(1, termination_.maximum_steps // policy_.count)
    per_evaluations = (
        None
        if termination_.maximum_evaluations is None
        else max(1, termination_.maximum_evaluations // policy_.count)
    )
    local_termination = OptimizationTermination(
        absolute_optimality=termination_.absolute_optimality,
        relative_optimality=termination_.relative_optimality,
        absolute_step=termination_.absolute_step,
        relative_step=termination_.relative_step,
        maximum_steps=per_steps,
        maximum_evaluations=per_evaluations,
    )
    results = tuple(
        policy_.local_method.solve(
            problem,
            space.unflatten(start),
            termination=local_termination,
            args=args,
        )
        for start in starts
    )
    objectives = jnp.stack([result.objective for result in results])
    statuses = jnp.stack([result.status for result in results])
    certified = jnp.stack([result.successful for result in results])
    feasibility = jnp.stack([result.diagnostics.primal_feasibility for result in results])
    score = jnp.where(
        certified,
        objectives,
        objectives + 1e12 * (1.0 + feasibility),
    )
    best_index = jnp.argmin(score)
    best = results[int(best_index)]
    final_points = jnp.stack([space.flatten(result.parameters) for result in results])
    return MultiStartResult(
        best,
        starts,
        final_points,
        objectives,
        statuses,
        certified,
        best_index,
        jnp.asarray(len(results), dtype=jnp.int32),
    )


__all__ = [
    "MultiStartPolicy",
    "MultiStartResult",
    "StartGenerator",
    "multistart_minimize",
]
