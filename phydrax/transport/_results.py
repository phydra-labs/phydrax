#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._blocks import apply_plan, coupling_statistics, dense_plan
from ._problem import DiscreteTransportProblem
from ._status import TransportStatus


class TransportProvenance(StrictModule):
    """Static numerical identity for one native transport solve."""

    method: str = eqx.field(static=True)
    cost: str = eqx.field(static=True)
    execution: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        cost: str,
        execution: str,
        differentiation: str,
        source: str,
        target: str,
        /,
        *,
        approximation: str = "exact",
    ):
        self.method = str(method)
        self.cost = str(cost)
        self.execution = str(execution)
        self.differentiation = str(differentiation)
        self.source = str(source)
        self.target = str(target)
        self.approximation = str(approximation)


class AbstractBalancedTransportPlan(StrictModule):
    """Minimal common contract for balanced finite transport plans."""

    regularized_cost: AbstractAttribute[Array]

    @property
    @abstractmethod
    def converged(self) -> Array:
        """Whether the solve satisfies its convergence and validity contracts."""
        raise NotImplementedError

    @abstractmethod
    def regularized_objective(self) -> Array:
        """Return the physical regularized objective consumed by divergence."""
        raise NotImplementedError

    @abstractmethod
    def source_marginal(self) -> Array:
        """Return the physical source marginal induced by the plan."""
        raise NotImplementedError

    @abstractmethod
    def target_marginal(self) -> Array:
        """Return the physical target marginal induced by the plan."""
        raise NotImplementedError

    @abstractmethod
    def apply_source_to_target(self, values: ArrayLike, /) -> Array:
        """Apply the physical coupling to source-indexed payloads."""
        raise NotImplementedError

    @abstractmethod
    def apply_target_to_source(self, values: ArrayLike, /) -> Array:
        """Apply the physical coupling to target-indexed payloads."""
        raise NotImplementedError

    @abstractmethod
    def barycentric_source_to_target(self, values: ArrayLike, /) -> Array:
        """Return target-conditioned barycenters of source payloads."""
        raise NotImplementedError

    @abstractmethod
    def barycentric_target_to_source(self, values: ArrayLike, /) -> Array:
        """Return source-conditioned barycenters of target payloads."""
        raise NotImplementedError

    @abstractmethod
    def dense_plan(self) -> Array:
        """Explicitly materialize the physical transport matrix."""
        raise NotImplementedError


class AbstractBalancedTransportSolver(StrictModule):
    """Solver producing a balanced plan for one finite transport problem."""

    epsilon: AbstractAttribute[Array]

    @abstractmethod
    def __call__(
        self, problem: DiscreteTransportProblem, /
    ) -> AbstractBalancedTransportPlan:
        raise NotImplementedError


class SinkhornDiagnostics(StrictModule):
    """Fixed-structure convergence diagnostics for balanced Sinkhorn."""

    status: Array
    num_iterations: Array
    first_converged_iteration: Array
    normalized_marginal_residual: Array
    physical_marginal_residual: Array
    dual_residual: Array
    primal_dual_gap: Array
    num_checks: Array
    residual_history: Array


class SinkhornResult(AbstractBalancedTransportPlan):
    """Native balanced entropic transport solution and matrix-free plan."""

    problem: DiscreteTransportProblem
    source_potential: Array
    target_potential: Array
    epsilon: Array
    transport_cost: Array
    regularization: Array
    regularized_cost: Array
    dual_cost: Array
    diagnostics: SinkhornDiagnostics
    provenance: TransportProvenance
    block_size: int | None = eqx.field(static=True)

    @property
    def converged(self) -> Array:
        """Whether the final iterate satisfies the convergence contract."""
        return self.diagnostics.status == int(TransportStatus.CONVERGED)

    def regularized_objective(self) -> Array:
        """Return the physical regularized Sinkhorn objective."""
        return self.regularized_cost

    def source_marginal(self) -> Array:
        """Return the physical source marginal induced by the computed plan."""
        source, _, _, _, _, _ = coupling_statistics(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
            block_size=self.block_size,
        )
        return self.problem.mass * source

    def target_marginal(self) -> Array:
        """Return the physical target marginal induced by the computed plan."""
        _, target, _, _, _, _ = coupling_statistics(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
            block_size=self.block_size,
        )
        return self.problem.mass * target

    def apply_source_to_target(self, values: ArrayLike, /) -> Array:
        """Apply the physical coupling from source atoms to target atoms."""
        return apply_plan(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
            jnp.asarray(values),
            direction="source_to_target",
            block_size=self.block_size,
        )

    def apply_target_to_source(self, values: ArrayLike, /) -> Array:
        """Apply the physical coupling from target atoms to source atoms."""
        return apply_plan(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
            jnp.asarray(values),
            direction="target_to_source",
            block_size=self.block_size,
        )

    def barycentric_source_to_target(self, values: ArrayLike, /) -> Array:
        """Return target-conditioned barycenters of source payloads."""
        applied = self.apply_source_to_target(values)
        weights = self.problem.target_weights
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        return jnp.where(denominator > 0.0, applied / safe_denominator, 0.0)

    def barycentric_target_to_source(self, values: ArrayLike, /) -> Array:
        """Return source-conditioned barycenters of target payloads."""
        applied = self.apply_target_to_source(values)
        weights = self.problem.source_weights
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        return jnp.where(denominator > 0.0, applied / safe_denominator, 0.0)

    def dense_plan(self) -> Array:
        """Explicitly materialize the complete physical transport matrix."""
        return dense_plan(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
        )


def require_converged(
    result: AbstractBalancedTransportPlan, /
) -> AbstractBalancedTransportPlan:
    """Raise unless a balanced solve converged, including under JAX transforms."""
    if not isinstance(result, AbstractBalancedTransportPlan):
        raise TypeError("result must implement the balanced transport plan contract.")
    failed = jnp.logical_not(result.converged)
    if not isinstance(failed, jax_core.Tracer):
        if bool(failed):
            raise eqx.EquinoxRuntimeError("Native balanced transport did not converge.")
        return result
    checked = eqx.error_if(
        result.regularized_cost,
        failed,
        "Native balanced transport did not converge.",
    )
    return eqx.tree_at(lambda item: item.regularized_cost, result, checked)


__all__ = [
    "AbstractBalancedTransportPlan",
    "AbstractBalancedTransportSolver",
    "SinkhornDiagnostics",
    "SinkhornResult",
    "TransportProvenance",
    "require_converged",
]
