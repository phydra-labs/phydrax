#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract


class ParquetEvidence(StrictModule, NonTrainableState):
    fixed_point_residual: Array
    channel_residuals: Array
    finite: Array
    converged: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.finite & self.converged & (self.status == 0)


class ParquetResult(StrictModule, NonTrainableState):
    full_vertex: Array
    reducible_channels: Array
    iterations: Array
    evidence: ParquetEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ParquetIterationPlan(StrictModule, NonTrainableState):
    """Immutable finite three-channel parquet iteration and resource policy."""

    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    maximum_vertex_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 512,
        tolerance: float = 1.0e-10,
        damping: float = 0.75,
        maximum_vertex_elements: int = 1_000_000,
    ):
        iterations = int(maximum_iterations)
        tolerance_, damping_ = float(tolerance), float(damping)
        capacity = int(maximum_vertex_elements)
        if iterations <= 0 or capacity <= 0:
            raise ValueError("Parquet iteration and resource limits must be positive.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Parquet tolerance must be finite and positive.")
        if not np.isfinite(damping_) or not 0.0 < damping_ <= 1.0:
            raise ValueError("Parquet damping must lie in (0, 1].")
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.damping = damping_
        self.maximum_vertex_elements = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "parquet-iteration-plan",
                "iterations": iterations,
                "tolerance": tolerance_,
                "damping": damping_,
                "maximum_vertex_elements": capacity,
            }
        )

    def prepare(
        self,
        fully_irreducible_vertex: ArrayLike,
        channel_bubbles: ArrayLike,
        channel_mixing: ArrayLike | None = None,
        /,
    ) -> "PreparedParquetIteration":
        bare = np.asarray(fully_irreducible_vertex, dtype=np.complex128)
        bubbles = np.asarray(channel_bubbles, dtype=np.complex128)
        mixing = (
            np.eye(3, dtype=np.complex128)
            if channel_mixing is None
            else np.asarray(channel_mixing, dtype=np.complex128)
        )
        if bare.ndim != 2 or bare.shape[0] != bare.shape[1] or bare.shape[0] == 0:
            raise ValueError(
                "fully_irreducible_vertex must be a non-empty square matrix."
            )
        if bubbles.shape != (3,) + bare.shape:
            raise ValueError("channel_bubbles must have shape (3, n, n).")
        if mixing.shape != (3, 3):
            raise ValueError("channel_mixing must have shape (3, 3).")
        required = bare.size + bubbles.size + mixing.size + 3 * bare.size
        if required > self.maximum_vertex_elements:
            raise ValueError("Prepared parquet arrays exceed maximum_vertex_elements.")
        if not all(np.all(np.isfinite(value)) for value in (bare, bubbles, mixing)):
            raise ValueError("Parquet inputs must be finite.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-parquet-iteration",
                "plan": self.plan_id,
                "bare": array_tree_fingerprint(bare),
                "bubbles": array_tree_fingerprint(bubbles),
                "mixing": array_tree_fingerprint(mixing),
            }
        )
        return PreparedParquetIteration(
            jnp.asarray(bare),
            jnp.asarray(bubbles),
            jnp.asarray(mixing),
            self.maximum_iterations,
            self.tolerance,
            self.damping,
            self.plan_id,
            prepared_id,
        )


class PreparedParquetIteration(StrictModule, NonTrainableState):
    """Prepared JAX finite-channel parquet fixed-point map."""

    fully_irreducible_vertex: Array
    channel_bubbles: Array
    channel_mixing: Array
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        fully_irreducible_vertex: Array,
        channel_bubbles: Array,
        channel_mixing: Array,
        maximum_iterations: int,
        tolerance: float,
        damping: float,
        plan_id: str,
        prepared_id: str,
        /,
    ):
        self.fully_irreducible_vertex = fully_irreducible_vertex
        self.channel_bubbles = channel_bubbles
        self.channel_mixing = channel_mixing
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.plan_id = str(plan_id)
        self.prepared_id = str(prepared_id)

    def iterate(self, initial_channels: ArrayLike | None = None, /) -> ParquetResult:
        shape = (3,) + self.fully_irreducible_vertex.shape
        channels = (
            jnp.zeros(shape, dtype=self.fully_irreducible_vertex.dtype)
            if initial_channels is None
            else jnp.asarray(initial_channels)
        )
        if channels.shape != shape:
            raise ValueError("initial_channels must have shape (3, n, n).")
        initial_residuals = jnp.full((3,), jnp.inf)

        def step(_, state):
            current, iterations, residuals, converged = state
            full = self.fully_irreducible_vertex + jnp.sum(current, axis=0)
            channel_images = contract(
                "cik,kl,cjl->cij",
                self.channel_bubbles,
                full,
                jnp.conj(self.channel_bubbles),
                backend="jax",
            )
            raw = contract(
                "cd,dij->cij",
                self.channel_mixing,
                channel_images,
                backend="jax",
            )
            candidate = (1.0 - self.damping) * current + self.damping * raw
            scale = jnp.maximum(1.0, jnp.max(jnp.abs(candidate), axis=(-2, -1)))
            new_residuals = jnp.max(jnp.abs(candidate - current), axis=(-2, -1)) / scale
            newly_converged = jnp.max(new_residuals) <= self.tolerance
            active = ~converged
            next_channels = jnp.where(active, candidate, current)
            next_residuals = jnp.where(active, new_residuals, residuals)
            return (
                next_channels,
                iterations + active.astype(jnp.int32),
                next_residuals,
                converged | newly_converged,
            )

        channels, iterations, residuals, converged = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            step,
            (
                channels,
                jnp.asarray(0, dtype=jnp.int32),
                initial_residuals,
                jnp.asarray(False),
            ),
        )
        full = self.fully_irreducible_vertex + jnp.sum(channels, axis=0)
        finite = jnp.all(jnp.isfinite(full)) & jnp.all(jnp.isfinite(channels))
        successful = finite & converged
        evidence = ParquetEvidence(
            jnp.max(residuals),
            residuals,
            finite,
            converged,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return ParquetResult(
            full,
            channels,
            iterations,
            evidence,
            self.plan_id,
            self.prepared_id,
        )


__all__ = [
    "ParquetEvidence",
    "ParquetIterationPlan",
    "ParquetResult",
    "PreparedParquetIteration",
]
