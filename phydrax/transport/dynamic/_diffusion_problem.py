#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule
from ...dynamics import TimeGrid
from ...integration._api import IntegrationRealization
from ...integration._targets import WeightedSampleTarget
from ...stochastic._path_ensemble import StochasticPathEnsemblePlan
from ...stochastic._state_space import AbstractTransitionKernel
from ._solver import SchrodingerBridgeSolver


class DiffusionBridgeProblem(StrictModule):
    """Finite-dimensional diffusion bridge specification before support lowering."""

    initial_law: AbstractProbabilityLaw
    terminal_law: AbstractProbabilityLaw
    reference: AbstractTransitionKernel
    time_grid: TimeGrid
    contexts: tuple[Any, ...]
    state_geometry: Any
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_law: AbstractProbabilityLaw,
        terminal_law: AbstractProbabilityLaw,
        reference: AbstractTransitionKernel,
        time_grid: TimeGrid,
        contexts: tuple[Any, ...],
        /,
        *,
        state_geometry: Any = None,
        problem_id: str | None = None,
    ):
        if not isinstance(initial_law, AbstractProbabilityLaw) or not isinstance(
            terminal_law, AbstractProbabilityLaw
        ):
            raise TypeError("bridge endpoints must be AbstractProbabilityLaw values.")
        if initial_law.event_shape != terminal_law.event_shape:
            raise ValueError("bridge endpoint event shapes differ.")
        if (
            not isinstance(reference, AbstractTransitionKernel)
            or not reference.has_log_density
        ):
            raise TypeError(
                "diffusion bridge reference must expose a normalized log density."
            )
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        selected_contexts = tuple(contexts)
        if len(selected_contexts) != time_grid.num_steps:
            raise ValueError(
                "contexts must contain one fixed transition context per interval."
            )
        resolved_id = problem_id or canonical_fingerprint(
            {
                "kind": "finite-diffusion-bridge-problem-v1",
                "time_grid": time_grid.time_id,
                "reference": reference.process_id,
                "event_shape": initial_law.event_shape,
                "state_geometry": None
                if state_geometry is None
                else state_geometry.geometry_id,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("problem_id must be non-empty.")
        self.initial_law = initial_law
        self.terminal_law = terminal_law
        self.reference = reference
        self.time_grid = time_grid
        self.contexts = selected_contexts
        self.state_geometry = state_geometry
        self.problem_id = resolved_id


class DiffusionBridgePlan(StrictModule):
    """Fixed support/quadrature/IPF/audit capacities for one bridge epoch."""

    proposal_realizations: tuple[IntegrationRealization, ...]
    solver: SchrodingerBridgeSolver
    ensemble_plan: StochasticPathEnsemblePlan | None
    support_capacity: int = eqx.field(static=True)
    transition_block_size: int = eqx.field(static=True)
    audit_capacity: int = eqx.field(static=True)
    minimum_ess: float = eqx.field(static=True)
    maximum_tail_error: float = eqx.field(static=True)

    def __init__(
        self,
        support_capacity: int,
        proposal_realizations: tuple[IntegrationRealization, ...],
        transition_block_size: int,
        /,
        *,
        solver: SchrodingerBridgeSolver | None = None,
        ensemble_plan: StochasticPathEnsemblePlan | None = None,
        audit_capacity: int = 128,
        minimum_ess: float = 2.0,
        maximum_tail_error: float = 0.1,
    ):
        capacity = int(support_capacity)
        block = int(transition_block_size)
        audit = int(audit_capacity)
        proposals = tuple(proposal_realizations)
        if capacity <= 0 or block <= 0 or audit <= 0:
            raise ValueError("bridge capacities must be positive.")
        if not proposals or any(
            not isinstance(item, IntegrationRealization) for item in proposals
        ):
            raise TypeError(
                "proposal_realizations must contain IntegrationRealization values."
            )
        for realization in proposals:
            if not isinstance(realization.target, WeightedSampleTarget):
                raise TypeError(
                    "bridge proposals must use WeightedSampleTarget realizations."
                )
        minimum = float(minimum_ess)
        tail = float(maximum_tail_error)
        if not isfinite(minimum) or minimum <= 0.0 or not isfinite(tail) or tail < 0.0:
            raise ValueError(
                "minimum_ess must be positive and maximum_tail_error nonnegative."
            )
        method = SchrodingerBridgeSolver() if solver is None else solver
        if not isinstance(method, SchrodingerBridgeSolver):
            raise TypeError("solver must be SchrodingerBridgeSolver.")
        if ensemble_plan is not None and not isinstance(
            ensemble_plan, StochasticPathEnsemblePlan
        ):
            raise TypeError("ensemble_plan must be StochasticPathEnsemblePlan or None.")
        self.support_capacity = capacity
        self.proposal_realizations = proposals
        self.transition_block_size = block
        self.solver = method
        self.ensemble_plan = ensemble_plan
        self.audit_capacity = audit
        self.minimum_ess = minimum
        self.maximum_tail_error = tail


def _proposal_arrays(realization: IntegrationRealization, /):
    target = realization.target
    samples = (
        target.samples.data if isinstance(target.samples, cx.Field) else target.samples
    )
    weights = (
        target.log_weights.data
        if isinstance(target.log_weights, cx.Field)
        else target.log_weights
    )
    mask = target.mask.data if isinstance(target.mask, cx.Field) else target.mask
    points = jnp.asarray(samples)
    log_weights = jnp.asarray(weights)
    active = (
        jnp.ones(log_weights.shape, dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if points.shape[0] != log_weights.shape[0] or active.shape != log_weights.shape:
        raise ValueError(
            "proposal points, log weights, and mask must align on axis zero."
        )
    return points, log_weights, active


class PreparedDiffusionBridge(StrictModule):
    problem: DiffusionBridgeProblem
    plan: DiffusionBridgePlan
    supports: Array
    log_proposal_weights: Array
    masks: Array
    log_transitions: Array
    row_normalizer_residuals: Array
    endpoint_probabilities: Array
    effective_sample_sizes: Array
    prepared_id: str = eqx.field(static=True)


__all__ = [
    "DiffusionBridgePlan",
    "DiffusionBridgeProblem",
    "PreparedDiffusionBridge",
]
