# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded deterministic sample-repair proposals with honest convergence.

Repair inserts probe samples, removes close samples, projects and relaxes the
cloud of one unchanged surface. It is a sampling proposal with explicit sample
lineage, never a physical topology event: split, merge and pinch changes belong
to the committed multiregion surface authority.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import Float64, Int32
from ._capacity import ActivePointDim, MeshfreeCapacityMap, MeshfreeCapacityPolicy
from ._neighbors import MeshfreeNeighborhoodPlan
from ._shifting import ShiftAmbientDim


@final
class SurfaceQualityEvidence(StrictModule):
    fill_distance: float = eqx.field(static=True)
    separation: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    maximum_amplification: float = eqx.field(static=True)
    triggered: bool = eqx.field(static=True)
    # Fill is witnessed on explicit probes; never advertised as global coverage.
    fill_is_certified_global: bool = eqx.field(static=True, default=False)


@final
class SurfaceResamplingResult(StrictModule):
    """One sample-repair proposal of a fixed surface.

    ``source_indices[k]`` is the input sample a proposed sample descends from,
    or ``-1`` for an inserted probe; ``inserted_probes`` and ``removed_sources``
    list the probe and input indices in application order. The proposal is
    committed only by an owner that prepares a conservative epoch transfer.
    """

    __strict_contract__ = True
    points: Float64[ActivePointDim, ShiftAmbientDim]
    source_indices: Int32[ActivePointDim]
    capacity: MeshfreeCapacityMap
    before: SurfaceQualityEvidence
    after: SurfaceQualityEvidence
    converged: bool = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    inserted_probes: tuple[int, ...] = eqx.field(static=True)
    removed_sources: tuple[int, ...] = eqx.field(static=True)
    projection_residual: float = eqx.field(static=True)
    capacity_refused: bool = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)


@final
class SurfaceResamplingPolicy(StrictModule):
    maximum_fill: float = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    amplification_limit: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    projection_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_fill: float,
        minimum_separation: float,
        condition_limit: float = 1e10,
        amplification_limit: float = 1e8,
        maximum_iterations: int = 20,
        projection_tolerance: float = 1e-9,
    ) -> None:
        if (
            not np.all(
                np.isfinite(
                    [
                        maximum_fill,
                        minimum_separation,
                        condition_limit,
                        amplification_limit,
                        projection_tolerance,
                    ]
                )
            )
            or min(
                maximum_fill,
                minimum_separation,
                condition_limit,
                amplification_limit,
                projection_tolerance,
            )
            <= 0
            or maximum_iterations < 1
        ):
            raise ValueError(
                "Surface quality limits and iteration capacity must be positive."
            )
        self.maximum_fill, self.minimum_separation = (
            float(maximum_fill),
            float(minimum_separation),
        )
        self.condition_limit, self.amplification_limit = (
            float(condition_limit),
            float(amplification_limit),
        )
        self.maximum_iterations, self.projection_tolerance = (
            int(maximum_iterations),
            float(projection_tolerance),
        )

    def assess(
        self,
        points: ArrayLike,
        probes: ArrayLike,
        stencil_quality: Callable[[Array], tuple[Array, Array]],
        /,
    ) -> SurfaceQualityEvidence:
        x = np.asarray(points, dtype=np.float64)
        if x.shape[0] < 2:
            raise ValueError("Surface quality needs at least two points.")
        unique = np.unique(x, axis=0)
        query = MeshfreeNeighborhoodPlan(unique, 1, targets=probes).prepare()
        fill = float(np.max(np.asarray(query.distances)))
        if unique.shape[0] != x.shape[0]:
            # Coincident source rows cannot enter native neighbor/stencil fits.
            # Their unresolved separation/conditioning is real defect evidence.
            cond, amp, sep = float("inf"), float("inf"), 0.0
        else:
            neighbors = MeshfreeNeighborhoodPlan(x, 2).prepare()
            nearest = np.asarray(neighbors.distances)[:, 1]
            condition, amplification = stencil_quality(jnp.asarray(x))
            cond, amp = (
                float(np.max(np.asarray(condition))),
                float(np.max(np.asarray(amplification))),
            )
            sep = float(np.min(nearest))
        bad = (
            (not np.all(np.isfinite([fill, sep, cond, amp])))
            or fill > self.maximum_fill
            or sep < self.minimum_separation
            or cond > self.condition_limit
            or amp > self.amplification_limit
        )
        return SurfaceQualityEvidence(fill, sep, cond, amp, bool(bad))

    def repair(
        self,
        points: ArrayLike,
        probes: ArrayLike,
        capacity_policy: MeshfreeCapacityPolicy,
        project: Callable[[Array], Array],
        surface_residual: Callable[[Array], Array],
        stencil_quality: Callable[[Array], tuple[Array, Array]],
        /,
        *,
        relax: Callable[[Array], Array] | None = None,
    ) -> SurfaceResamplingResult:
        """Lexicographic probe insertion and higher-index close-point removal.

        Each iteration applies at most one insertion, one removal or one
        relaxation, so work is bounded by ``maximum_iterations``. Stencil
        defects that geometric repair cannot resolve remain explicit
        nonconvergence. The result is a sample proposal; an owner must prepare
        a conservative epoch transition before committing any fields.
        """
        x, probe = (
            np.asarray(points, dtype=np.float64),
            np.asarray(probes, dtype=np.float64),
        )
        before = self.assess(x, probe, stencil_quality)
        # Lineage of every current sample: input index, or -1 with its probe.
        inputs = x.shape[0]
        source = np.arange(inputs, dtype=np.int64)
        origin = np.full(x.shape[0], -1, dtype=np.int64)
        iterations = 0
        refused = False
        residual = float(np.max(np.abs(np.asarray(surface_residual(jnp.asarray(x))))))
        evidence = before
        for iteration in range(self.maximum_iterations):
            if not evidence.triggered and residual <= self.projection_tolerance:
                break
            step = self._step(x, probe, evidence, capacity_policy, relax)
            if step is None:
                refused = x.shape[0] >= capacity_policy.buckets[-1] and (
                    evidence.fill_distance > self.maximum_fill
                )
                break
            proposed, keep, inserted = step
            projected = np.asarray(project(jnp.asarray(proposed)), dtype=np.float64)
            residual = float(
                np.max(np.abs(np.asarray(surface_residual(jnp.asarray(projected)))))
            )
            iterations = iteration + 1
            if (
                projected.shape != proposed.shape
                or not np.all(np.isfinite(projected))
                or residual > self.projection_tolerance
            ):
                break
            source, origin = source[keep], origin[keep]
            if inserted is not None:
                source = np.append(source, -1)
                origin = np.append(origin, inserted)
            x = projected
            evidence = self.assess(x, probe, stencil_quality)
        capacity = capacity_policy.allocate(x.shape[0])
        converged = (
            not evidence.triggered
            and residual <= self.projection_tolerance
            and not refused
        )
        removed = np.setdiff1d(np.arange(inputs), source)
        identifier = canonical_fingerprint(
            {
                "kind": "surface-sample-repair",
                "points": x,
                "source_indices": source,
                "probes": probe,
                "input_count": inputs,
            }
        )
        return SurfaceResamplingResult(
            jnp.asarray(x),
            jnp.asarray(source, dtype=jnp.int32),
            capacity,
            before,
            evidence,
            converged,
            iterations,
            tuple(int(item) for item in origin[source < 0]),
            tuple(int(item) for item in removed),
            residual,
            refused,
            identifier,
        )

    def _step(
        self,
        x: np.ndarray,
        probe: np.ndarray,
        evidence: SurfaceQualityEvidence,
        capacity_policy: MeshfreeCapacityPolicy,
        relax: Callable[[Array], Array] | None,
    ) -> tuple[np.ndarray, np.ndarray, int | None] | None:
        """One bounded proposal: points, kept rows, and an inserted probe index."""
        rows = np.arange(x.shape[0])
        if evidence.separation < self.minimum_separation and x.shape[0] > 2:
            _unique, first = np.unique(x, axis=0, return_index=True)
            duplicate = np.setdiff1d(rows, first)
            if duplicate.size:
                remove_index = int(duplicate[-1])
            else:
                neighborhood = MeshfreeNeighborhoodPlan(x, 2).prepare()
                distances = np.asarray(neighborhood.distances)[:, 1]
                row = int(np.argmin(distances))
                other = int(np.asarray(neighborhood.relation.source_indices)[row, 1])
                remove_index = max(row, other)
            keep = rows != remove_index
            return x[keep], keep, None
        if evidence.fill_distance > self.maximum_fill:
            if x.shape[0] >= capacity_policy.buckets[-1]:
                return None
            query = MeshfreeNeighborhoodPlan(
                np.unique(x, axis=0), 1, targets=probe
            ).prepare()
            index = int(np.argmax(np.asarray(query.distances)[:, 0]))
            return (
                np.concatenate((x, probe[index : index + 1]), axis=0),
                np.ones(x.shape[0], dtype=bool),
                index,
            )
        if relax is None:
            return None
        proposed = np.asarray(relax(jnp.asarray(x)), dtype=np.float64)
        if proposed.shape != x.shape:
            raise ValueError("Relaxation must preserve compact geometry shape.")
        return proposed, np.ones(x.shape[0], dtype=bool), None


__all__ = ["SurfaceQualityEvidence", "SurfaceResamplingPolicy", "SurfaceResamplingResult"]
