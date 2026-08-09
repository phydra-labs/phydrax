#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._results import TransportProvenance
from ._status import TransportStatus
from ._unbalanced_blocks import apply_plan, coupling_statistics, dense_plan
from ._unbalanced_problem import UnbalancedTransportProblem


class UnbalancedSinkhornDiagnostics(StrictModule):
    """Fixed-structure diagnostics for generalized unbalanced Sinkhorn."""

    status: Array
    num_iterations: Array
    first_converged_iteration: Array
    fixed_point_residual: Array
    source_stationarity_residual: Array
    target_stationarity_residual: Array
    primal_dual_gap: Array
    transported_mass: Array
    mass_collapsed: Array
    num_checks: Array
    residual_history: Array


class UnbalancedSinkhornResult(StrictModule):
    """Unbalanced entropic transport solution with relaxed physical marginals."""

    problem: UnbalancedTransportProblem
    source_potential: Array
    target_potential: Array
    epsilon: Array
    transported_mass: Array
    transport_cost: Array
    entropy_kl: Array
    source_marginal_kl: Array
    target_marginal_kl: Array
    entropy_regularization: Array
    source_marginal_regularization: Array
    target_marginal_regularization: Array
    regularized_cost: Array
    dual_cost: Array
    diagnostics: UnbalancedSinkhornDiagnostics
    provenance: TransportProvenance
    block_size: int | None = eqx.field(static=True)

    @property
    def converged(self) -> Array:
        """Whether the solve converged without numerical mass collapse."""
        return self.diagnostics.status == int(TransportStatus.CONVERGED)

    @property
    def mass_collapsed(self) -> Array:
        """Whether the computed coupling fell below the declared mass threshold."""
        return self.diagnostics.mass_collapsed

    def source_marginal(self) -> Array:
        """Return the relaxed physical source marginal induced by the coupling."""
        source, _, _, _, _, _ = coupling_statistics(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
            block_size=self.block_size,
        )
        return source

    def target_marginal(self) -> Array:
        """Return the relaxed physical target marginal induced by the coupling."""
        _, target, _, _, _, _ = coupling_statistics(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
            block_size=self.block_size,
        )
        return target

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
        """Return target-conditioned barycenters under the relaxed marginal."""
        applied = self.apply_source_to_target(values)
        weights = self.target_marginal()
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        return jnp.where(denominator > 0.0, applied / denominator, 0.0)

    def barycentric_target_to_source(self, values: ArrayLike, /) -> Array:
        """Return source-conditioned barycenters under the relaxed marginal."""
        applied = self.apply_target_to_source(values)
        weights = self.source_marginal()
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        return jnp.where(denominator > 0.0, applied / denominator, 0.0)

    def dense_plan(self) -> Array:
        """Explicitly materialize the complete physical unbalanced plan."""
        return dense_plan(
            self.problem,
            self.source_potential,
            self.target_potential,
            self.epsilon,
        )


def require_unbalanced_converged(
    result: UnbalancedSinkhornResult,
    /,
) -> UnbalancedSinkhornResult:
    """Raise a JAX-compatible error unless unbalanced transport converged."""
    if not isinstance(result, UnbalancedSinkhornResult):
        raise TypeError("result must be an UnbalancedSinkhornResult.")
    checked = eqx.error_if(
        result.source_potential,
        ~result.converged,
        "Native unbalanced Sinkhorn transport did not converge.",
    )
    return eqx.tree_at(lambda item: item.source_potential, result, checked)


__all__ = [
    "UnbalancedSinkhornDiagnostics",
    "UnbalancedSinkhornResult",
    "require_unbalanced_converged",
]
