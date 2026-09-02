#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..discretization import TemporalMesh
from ._markov_cubature import MarkovCubaturePlan, MarkovCubatureSolution


class MarkovCubatureRefinementPolicy(StrictModule):
    """Deterministic host refinement for one finite nonuniform mesh epoch."""

    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_intervals: int = eqx.field(static=True)
    marking_fraction: float = eqx.field(static=True)
    embedded_degree: int = eqx.field(static=True)

    def __init__(
        self,
        absolute_tolerance: float,
        relative_tolerance: float,
        maximum_intervals: int,
        /,
        *,
        marking_fraction: float = 0.5,
        embedded_degree: int = 1,
    ):
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        maximum = int(maximum_intervals)
        fraction = float(marking_fraction)
        embedded = int(embedded_degree)
        if not all(isfinite(value) and value >= 0.0 for value in (absolute, relative)):
            raise ValueError("refinement tolerances must be finite and nonnegative.")
        if maximum <= 0 or embedded <= 0:
            raise ValueError("maximum_intervals and embedded_degree must be positive.")
        if not isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("marking_fraction must lie in (0, 1].")
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.maximum_intervals = maximum
        self.marking_fraction = fraction
        self.embedded_degree = embedded


class WeakObservableEnvelope(StrictModule):
    """Caller-supplied Lie-derivative/stability hypotheses for a weak bound."""

    derivative_bounds: Array
    stability_bound: Array
    observable_id: str = eqx.field(static=True)
    norm_kind: str = eqx.field(static=True)

    def __init__(
        self,
        observable_id: str,
        derivative_bounds: ArrayLike,
        stability_bound: ArrayLike,
        /,
        *,
        norm_kind: str = "supremum",
    ):
        bounds = jnp.asarray(derivative_bounds)
        stability = jnp.asarray(stability_bound)
        if bounds.ndim != 1 or bounds.size == 0 or stability.shape != ():
            raise ValueError(
                "derivative_bounds must be nonempty rank one and stability scalar."
            )
        if not isinstance(observable_id, str) or not observable_id or not norm_kind:
            raise ValueError("observable_id and norm_kind must be non-empty strings.")
        self.derivative_bounds = bounds
        self.stability_bound = stability
        self.observable_id = observable_id
        self.norm_kind = str(norm_kind)


class MarkovCubatureErrorEvidence(StrictModule):
    """Separated cubature errors; a rigorous field needs explicit hypotheses."""

    signature: Array
    recombination: Array
    flow: Array
    embedded: Array
    rigorous_upper_bound: Array | None
    valid: Array
    bound_hypotheses: tuple[str, ...] = eqx.field(static=True)
    evidence_kind: str = eqx.field(static=True)


def markov_cubature_error_evidence(
    solution: MarkovCubatureSolution,
    /,
    *,
    signature_residuals: ArrayLike,
    flow_errors: ArrayLike,
    embedded_errors: ArrayLike,
    envelope: WeakObservableEnvelope | None = None,
) -> MarkovCubatureErrorEvidence:
    """Assemble diagnostics and, only with an envelope, a bounded weak error."""
    if not isinstance(solution, MarkovCubatureSolution):
        raise TypeError("solution must be a MarkovCubatureSolution.")
    signature = jnp.asarray(signature_residuals)
    flow = jnp.asarray(flow_errors)
    embedded = jnp.asarray(embedded_errors)
    recombination = jnp.asarray(solution.diagnostics.maximum_moment_error)
    finite = (
        jnp.all(jnp.isfinite(signature))
        & jnp.all(jnp.isfinite(recombination))
        & jnp.all(jnp.isfinite(flow))
        & jnp.all(jnp.isfinite(embedded))
    )
    if envelope is None:
        bound = None
        hypotheses = ()
        kind = "a-posteriori-diagnostic"
    else:
        if not isinstance(envelope, WeakObservableEnvelope):
            raise TypeError("envelope must be WeakObservableEnvelope or None.")
        coefficient = jnp.max(jnp.abs(envelope.derivative_bounds)) * jnp.abs(
            envelope.stability_bound
        )
        bound = coefficient * (
            jnp.max(jnp.abs(signature))
            + jnp.max(jnp.abs(recombination))
            + jnp.max(jnp.abs(flow))
            + jnp.max(jnp.abs(embedded))
        )
        hypotheses = (
            f"observable:{envelope.observable_id}",
            f"norm:{envelope.norm_kind}",
            "caller-certified-iterated-lie-derivatives",
            "caller-certified-stability",
        )
        kind = "hypothesis-backed-upper-bound"
    return MarkovCubatureErrorEvidence(
        signature=signature,
        recombination=recombination,
        flow=flow,
        embedded=embedded,
        rigorous_upper_bound=bound,
        valid=finite & solution.successful,
        bound_hypotheses=hypotheses,
        evidence_kind=kind,
    )


def refine_markov_cubature(
    plan: MarkovCubaturePlan,
    interval_errors: ArrayLike,
    policy: MarkovCubatureRefinementPolicy,
    /,
) -> MarkovCubaturePlan:
    """Split deterministically marked intervals and return a new plan epoch."""
    if not isinstance(plan, MarkovCubaturePlan):
        raise TypeError("plan must be a MarkovCubaturePlan.")
    if not isinstance(policy, MarkovCubatureRefinementPolicy):
        raise TypeError("policy must be a MarkovCubatureRefinementPolicy.")
    nodes = np.asarray(plan.temporal_mesh.nodes, dtype=float)
    active = np.asarray(plan.temporal_mesh.active_intervals, dtype=bool)
    errors = np.asarray(interval_errors, dtype=float)
    if (
        errors.shape != active.shape
        or np.any(~np.isfinite(errors))
        or np.any(errors < 0.0)
    ):
        raise ValueError("interval_errors must be finite, nonnegative, and mesh-aligned.")
    active_errors = np.where(active, errors, -np.inf)
    maximum_error = float(np.max(active_errors))
    scale = max(float(np.max(np.abs(nodes))), 1.0)
    tolerance = policy.absolute_tolerance + policy.relative_tolerance * scale
    if maximum_error <= tolerance:
        return plan
    threshold = policy.marking_fraction * maximum_error
    marked = active & (errors >= threshold)
    new_count = int(active.size + np.count_nonzero(marked))
    if new_count > policy.maximum_intervals:
        raise ValueError("Markov cubature refinement exceeds maximum_intervals.")
    refined_nodes = [nodes[0]]
    for index in range(active.size):
        if marked[index]:
            refined_nodes.append(0.5 * (nodes[index] + nodes[index + 1]))
        refined_nodes.append(nodes[index + 1])
    mesh = TemporalMesh(
        np.asarray(refined_nodes),
        role=plan.temporal_mesh.role,
        mesh_id=f"{plan.temporal_mesh.mesh_id}:refined:{new_count}",
    )
    return MarkovCubaturePlan(
        mesh,
        plan.increment_rule,
        recombination=plan.recombination,
        method=plan.method,
        path=plan.path,
        maximum_expanded_particles=plan.maximum_expanded_particles,
        flow_substeps=plan.flow_substeps,
        collect_history=plan.collect_history,
        throw=plan.throw,
    )


__all__ = [
    "MarkovCubatureErrorEvidence",
    "MarkovCubatureRefinementPolicy",
    "WeakObservableEnvelope",
    "markov_cubature_error_evidence",
    "refine_markov_cubature",
]
