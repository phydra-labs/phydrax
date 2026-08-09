#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
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

    def __init__(
        self,
        method: str,
        cost: str,
        execution: str,
        differentiation: str,
        source: str,
        target: str,
        /,
    ):
        self.method = str(method)
        self.cost = str(cost)
        self.execution = str(execution)
        self.differentiation = str(differentiation)
        self.source = str(source)
        self.target = str(target)


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


class SinkhornResult(StrictModule):
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
        return jnp.where(denominator > 0.0, applied / denominator, 0.0)

    def barycentric_target_to_source(self, values: ArrayLike, /) -> Array:
        """Return source-conditioned barycenters of target payloads."""
        applied = self.apply_target_to_source(values)
        weights = self.problem.source_weights
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        return jnp.where(denominator > 0.0, applied / denominator, 0.0)

    def dense_plan(self) -> Array:
        """Explicitly materialize the complete physical transport matrix."""
        return dense_plan(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
        )


def require_converged(result: SinkhornResult, /) -> SinkhornResult:
    """Raise a JAX-compatible error unless a Sinkhorn solve converged."""
    if not isinstance(result, SinkhornResult):
        raise TypeError("result must be a SinkhornResult.")
    checked = eqx.error_if(
        result.source_potential,
        ~result.converged,
        "Native Sinkhorn transport did not converge.",
    )
    return eqx.tree_at(lambda item: item.source_potential, result, checked)


__all__ = [
    "SinkhornDiagnostics",
    "SinkhornResult",
    "TransportProvenance",
    "require_converged",
]
