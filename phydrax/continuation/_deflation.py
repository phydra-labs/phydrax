#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from .._tree_math import tree_norm, tree_scale, validate_inexact_tree
from ..linalg import AbstractVectorSpace
from ..nonlinear import (
    AbstractNonlinearMethod,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)


class DeflatedRootStatus(IntEnum):
    """Terminal status of a solve against a deflated residual."""

    SUCCESS = 0
    NONLINEAR_SOLVE_FAILED = 1
    KNOWN_ROOT_REJECTED = 2
    ORIGINAL_RESIDUAL_NOT_CONVERGED = 3
    NONFINITE_DISTANCE = 4


class AbstractDeflationMetric(StrictModule):
    """Distance contract used to distinguish roots in a state space."""

    metric_id: AbstractAttribute[str]

    @abc.abstractmethod
    def distance(
        self,
        left: PyTree[Any],
        right: PyTree[Any],
        /,
    ) -> Array:
        raise NotImplementedError


class VectorSpaceDeflationMetric(AbstractDeflationMetric):
    """Metric induced by an explicit vector-space pairing."""

    space: AbstractVectorSpace
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: AbstractVectorSpace,
        /,
        *,
        metric_id: str | None = None,
    ):
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        identifier = (
            f"vector-space:{space.space_id}" if metric_id is None else str(metric_id)
        )
        if not identifier:
            raise ValueError("metric_id must be non-empty.")
        self.space = space
        self.metric_id = identifier

    def distance(
        self,
        left: PyTree[Any],
        right: PyTree[Any],
        /,
    ) -> Array:
        left_ = self.space.validate(left)
        right_ = self.space.validate(right)
        difference = jax.tree.map(lambda x, y: x - y, left_, right_)
        squared = jnp.real(self.space.inner(difference, difference))
        return jnp.sqrt(jnp.maximum(squared, 0.0))


class CallableDeflationMetric(AbstractDeflationMetric):
    """Adapter for a user-defined state-space metric."""

    function: Callable[[PyTree[Any], PyTree[Any]], Any]
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[PyTree[Any], PyTree[Any]], Any],
        /,
        *,
        metric_id: str = "callable-deflation-metric",
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(metric_id)
        if not identifier:
            raise ValueError("metric_id must be non-empty.")
        self.function = function
        self.metric_id = identifier

    def distance(
        self,
        left: PyTree[Any],
        right: PyTree[Any],
        /,
    ) -> Array:
        value = jnp.asarray(self.function(left, right))
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("A deflation metric must return one real scalar array.")
        return value


class DeflationPolicy(StrictModule):
    """Deflation exponent, shift, and root-acceptance tolerances."""

    power: float = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    distance_floor: float = eqx.field(static=True)
    known_root_tolerance: float = eqx.field(static=True)
    original_residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        power: float = 2.0,
        shift: float = 1.0,
        distance_floor: float = 1e-6,
        known_root_tolerance: float = 1e-5,
        original_residual_tolerance: float = 1e-8,
    ):
        power_ = float(power)
        shift_ = float(shift)
        floor_ = float(distance_floor)
        root_tolerance = float(known_root_tolerance)
        residual_tolerance = float(original_residual_tolerance)
        if not isfinite(power_) or power_ <= 0.0:
            raise ValueError("power must be finite and positive.")
        if not isfinite(shift_) or shift_ < 0.0:
            raise ValueError("shift must be finite and non-negative.")
        if not isfinite(floor_) or floor_ <= 0.0:
            raise ValueError("distance_floor must be finite and positive.")
        if not isfinite(root_tolerance) or root_tolerance < 0.0:
            raise ValueError("known_root_tolerance must be finite and non-negative.")
        if not isfinite(residual_tolerance) or residual_tolerance < 0.0:
            raise ValueError(
                "original_residual_tolerance must be finite and non-negative."
            )
        self.power = power_
        self.shift = shift_
        self.distance_floor = floor_
        self.known_root_tolerance = root_tolerance
        self.original_residual_tolerance = residual_tolerance


class DeflationProvenance(StrictModule):
    """Original problem, deflated problem, method, and metric identities."""

    problem_id: str = eqx.field(static=True)
    deflated_problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        deflated_problem_id: str,
        method_id: str,
        metric_id: str,
    ):
        values = tuple(
            str(value)
            for value in (problem_id, deflated_problem_id, method_id, metric_id)
        )
        if any(not value for value in values):
            raise ValueError("Deflation provenance identifiers must be non-empty.")
        (
            self.problem_id,
            self.deflated_problem_id,
            self.method_id,
            self.metric_id,
        ) = values


class DeflatedRootResult(StrictModule):
    """Deflated solve output with acceptance checked against the original problem."""

    state: PyTree[Array]
    original_residual: PyTree[Array]
    nonlinear_result: NonlinearResult
    status: Array
    minimum_known_root_distance: Array
    deflation_factor: Array
    nearest_known_root: Array
    provenance: DeflationProvenance

    def __init__(
        self,
        *,
        state: PyTree[Any],
        original_residual: PyTree[Any],
        nonlinear_result: NonlinearResult,
        status: Any,
        minimum_known_root_distance: Any,
        deflation_factor: Any,
        nearest_known_root: Any,
        provenance: DeflationProvenance,
    ):
        if not isinstance(nonlinear_result, NonlinearResult):
            raise TypeError("nonlinear_result must be a NonlinearResult.")
        if not isinstance(provenance, DeflationProvenance):
            raise TypeError("provenance must be DeflationProvenance.")
        self.state = state
        self.original_residual = original_residual
        self.nonlinear_result = nonlinear_result
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.minimum_known_root_distance = jnp.asarray(minimum_known_root_distance)
        self.deflation_factor = jnp.asarray(deflation_factor)
        self.nearest_known_root = jnp.asarray(nearest_known_root, dtype=jnp.int32)
        self.provenance = provenance

    @property
    def successful(self) -> Array:
        return self.status == int(DeflatedRootStatus.SUCCESS)


class RootDeflation(StrictModule):
    """Metric-aware residual deflation around a nonempty set of known roots."""

    problem: NonlinearSystemProblem
    known_roots: tuple[PyTree[Array], ...]
    metric: AbstractDeflationMetric
    policy: DeflationPolicy
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        known_roots: Sequence[PyTree[Any]],
        /,
        *,
        metric: AbstractDeflationMetric,
        policy: DeflationPolicy | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        roots = tuple(
            validate_inexact_tree(root, name="known deflated root")
            for root in known_roots
        )
        if not roots:
            raise ValueError("known_roots must contain at least one root.")
        reference_structure = jax.tree.structure(roots[0])
        if any(jax.tree.structure(root) != reference_structure for root in roots[1:]):
            raise ValueError("Every known root must have the same PyTree structure.")
        if not isinstance(metric, AbstractDeflationMetric):
            raise TypeError("metric must be an AbstractDeflationMetric.")
        policy_ = DeflationPolicy() if policy is None else policy
        if not isinstance(policy_, DeflationPolicy):
            raise TypeError("policy must be a DeflationPolicy or None.")
        identifier = (
            f"{problem.problem_id}/deflated" if problem_id is None else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.problem = problem
        self.known_roots = roots
        self.metric = metric
        self.policy = policy_
        self.problem_id = identifier

    def distances(self, state: PyTree[Any], /) -> Array:
        values = tuple(self.metric.distance(state, root) for root in self.known_roots)
        distances = jnp.stack(values)
        if not jnp.issubdtype(distances.dtype, jnp.floating):
            raise TypeError("Deflation distances must have real floating-point dtype.")
        return distances

    def factor(self, state: PyTree[Any], /) -> Array:
        distances = self.distances(state)
        safe = jnp.maximum(distances, self.policy.distance_floor)
        log_shift = (
            -jnp.inf
            if self.policy.shift == 0.0
            else jnp.log(jnp.asarray(self.policy.shift, dtype=safe.dtype))
        )
        log_terms = jnp.logaddexp(log_shift, -self.policy.power * jnp.log(safe))
        log_factor = jnp.sum(log_terms)
        maximum_log = jnp.log(jnp.asarray(jnp.finfo(safe.dtype).max, dtype=safe.dtype))
        return jnp.exp(jnp.minimum(log_factor, maximum_log))

    def residual(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        residual = self.problem.residual(state, args)
        return tree_scale(self.factor(state), residual)

    def as_problem(self, /) -> NonlinearSystemProblem:
        return NonlinearSystemProblem(
            lambda state, args: self.residual(state, args),
            problem_id=self.problem_id,
        )

    def solve(
        self,
        method: AbstractNonlinearMethod,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
    ) -> DeflatedRootResult:
        if not isinstance(method, AbstractNonlinearMethod):
            raise TypeError("method must be an AbstractNonlinearMethod.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        nonlinear_result = method.solve(
            self.as_problem(),
            initial_state,
            termination=termination,
            args=args,
        )
        state = nonlinear_result.state
        original_residual = self.problem.residual(state, args)
        original_norm = tree_norm(original_residual)
        distances = self.distances(state)
        nearest = jnp.argmin(distances)
        minimum_distance = distances[nearest]
        finite_distance = jnp.all(jnp.isfinite(distances))
        known = minimum_distance <= self.policy.known_root_tolerance
        residual_converged = jnp.isfinite(original_norm) & (
            original_norm <= self.policy.original_residual_tolerance
        )
        status = jnp.where(
            ~finite_distance,
            int(DeflatedRootStatus.NONFINITE_DISTANCE),
            jnp.where(
                known,
                int(DeflatedRootStatus.KNOWN_ROOT_REJECTED),
                jnp.where(
                    ~nonlinear_result.successful,
                    int(DeflatedRootStatus.NONLINEAR_SOLVE_FAILED),
                    jnp.where(
                        residual_converged,
                        int(DeflatedRootStatus.SUCCESS),
                        int(DeflatedRootStatus.ORIGINAL_RESIDUAL_NOT_CONVERGED),
                    ),
                ),
            ),
        )
        return DeflatedRootResult(
            state=state,
            original_residual=original_residual,
            nonlinear_result=nonlinear_result,
            status=status,
            minimum_known_root_distance=minimum_distance,
            deflation_factor=self.factor(state),
            nearest_known_root=nearest,
            provenance=DeflationProvenance(
                problem_id=self.problem.problem_id,
                deflated_problem_id=self.problem_id,
                method_id=method.method_id,
                metric_id=self.metric.metric_id,
            ),
        )


def solve_deflated(
    deflation: RootDeflation,
    method: AbstractNonlinearMethod,
    initial_state: PyTree[Any],
    /,
    *,
    termination: NonlinearTermination,
    args: Any = None,
) -> DeflatedRootResult:
    """Solve a deflated problem and reject convergence to any registered root."""
    if not isinstance(deflation, RootDeflation):
        raise TypeError("deflation must be a RootDeflation.")
    return deflation.solve(
        method,
        initial_state,
        termination=termination,
        args=args,
    )


__all__ = [
    "AbstractDeflationMetric",
    "CallableDeflationMetric",
    "DeflatedRootResult",
    "DeflatedRootStatus",
    "DeflationPolicy",
    "DeflationProvenance",
    "RootDeflation",
    "VectorSpaceDeflationMetric",
    "solve_deflated",
]
