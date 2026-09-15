#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Truthful fixed-history, event-aware, and relaxed block-AMR derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cut_transition import MultivaluedCutCellTransition
from ._mapped_geometry import CanonicalMappedGeometryPlan


BlockAMRDerivativeMode = Literal["frozen-history", "event-aware", "relaxed"]


class BlockAMRDerivativePolicy(StrictModule, NonTrainableState):
    """Explicit derivative meaning and validity margins."""

    mode: BlockAMRDerivativeMode = eqx.field(static=True)
    minimum_topology_margin: float = eqx.field(static=True)
    minimum_transversality: float = eqx.field(static=True)
    relaxation_temperature: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: BlockAMRDerivativeMode = "frozen-history",
        /,
        *,
        minimum_topology_margin: float = 1.0e-8,
        minimum_transversality: float = 1.0e-8,
        relaxation_temperature: float = 0.05,
    ):
        topology = float(minimum_topology_margin)
        transversality = float(minimum_transversality)
        temperature = float(relaxation_temperature)
        if mode not in ("frozen-history", "event-aware", "relaxed"):
            raise ValueError("Unknown block-AMR derivative mode.")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in (topology, transversality, temperature)
        ):
            raise ValueError("Block-AMR derivative margins must be positive and finite.")
        self.mode = mode
        self.minimum_topology_margin = topology
        self.minimum_transversality = transversality
        self.relaxation_temperature = temperature
        self.policy_id = canonical_fingerprint(
            {
                "kind": "block-amr-derivative-policy",
                "mode": mode,
                "minimum_topology_margin": topology,
                "minimum_transversality": transversality,
                "relaxation_temperature": temperature,
            }
        )


class BlockAMRDerivativeEvidence(StrictModule, NonTrainableState):
    """Branch margin and event regularity attached to one derivative action."""

    topology_margin: Array
    transversality: Array
    finite: Array
    valid: Array
    mode: BlockAMRDerivativeMode = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: BlockAMRDerivativePolicy,
        topology_margin: ArrayLike,
        transversality: ArrayLike,
        finite: ArrayLike,
        /,
        *,
        reason: str,
    ):
        if not isinstance(policy, BlockAMRDerivativePolicy):
            raise TypeError("Derivative evidence requires BlockAMRDerivativePolicy.")
        margin = jnp.asarray(topology_margin)
        transverse = jnp.asarray(transversality)
        finite_ = jnp.asarray(finite)
        if margin.shape != () or transverse.shape != () or finite_.shape != ():
            raise ValueError("Derivative evidence values must be scalar.")
        event_valid = (policy.mode != "event-aware") | (
            jnp.isfinite(transverse)
            & (jnp.abs(transverse) >= policy.minimum_transversality)
        )
        valid = (
            finite_
            & jnp.isfinite(margin)
            & (margin >= policy.minimum_topology_margin)
            & event_valid
        )
        self.topology_margin = margin
        self.transversality = transverse
        self.finite = finite_
        self.valid = valid
        self.mode = policy.mode
        self.reason = str(reason)
        self.policy_id = policy.policy_id


class FrozenCutCellDerivativeResult(StrictModule):
    """Primal/tangent pair for a fixed common-refinement graph."""

    primal: Array
    tangent: Array
    evidence: BlockAMRDerivativeEvidence
    plan_id: str = eqx.field(static=True)


class FrozenCutCellTransitionDerivativePlan(StrictModule, NonTrainableState):
    """Exact state derivative and transpose with frozen cut overlap routes."""

    transition: MultivaluedCutCellTransition
    policy: BlockAMRDerivativePolicy
    topology_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: MultivaluedCutCellTransition,
        /,
        *,
        topology_margin: float,
        policy: BlockAMRDerivativePolicy | None = None,
    ):
        policy_ = BlockAMRDerivativePolicy("frozen-history") if policy is None else policy
        margin = float(topology_margin)
        if not isinstance(transition, MultivaluedCutCellTransition):
            raise TypeError("Frozen derivatives require MultivaluedCutCellTransition.")
        if (
            not isinstance(policy_, BlockAMRDerivativePolicy)
            or policy_.mode != "frozen-history"
        ):
            raise ValueError(
                "Frozen transition derivative requires frozen-history policy."
            )
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("Frozen topology margin must be finite and nonnegative.")
        self.transition = transition
        self.policy = policy_
        self.topology_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frozen-cut-cell-transition-derivative",
                "transition": transition.transition_id,
                "policy": policy_.policy_id,
                "topology_margin": margin,
            }
        )

    def jvp_content(
        self,
        source_content: ArrayLike,
        source_tangent: ArrayLike,
        /,
    ) -> FrozenCutCellDerivativeResult:
        primal = self.transition.apply_content(source_content).target_content
        tangent = self.transition.apply_content(source_tangent).target_content
        finite = jnp.all(jnp.isfinite(primal)) & jnp.all(jnp.isfinite(tangent))
        evidence = BlockAMRDerivativeEvidence(
            self.policy,
            self.topology_margin,
            jnp.asarray(jnp.inf),
            finite,
            reason="fixed overlap graph and linear extensive transfer",
        )
        poisoned = jnp.where(evidence.valid, tangent, jnp.nan)
        return FrozenCutCellDerivativeResult(
            primal=primal,
            tangent=poisoned,
            evidence=evidence,
            plan_id=self.plan_id,
        )

    def vjp_content(
        self, target_cotangent: ArrayLike, /
    ) -> tuple[Array, BlockAMRDerivativeEvidence]:
        cotangent = self.transition.transpose_content(target_cotangent)
        finite = jnp.all(jnp.isfinite(cotangent))
        evidence = BlockAMRDerivativeEvidence(
            self.policy,
            self.topology_margin,
            jnp.asarray(jnp.inf),
            finite,
            reason="exact algebraic transpose of fixed extensive transfer",
        )
        return jnp.where(evidence.valid, cotangent, jnp.nan), evidence


class EventAwareCutCellDerivativePlan(StrictModule, NonTrainableState):
    """Delegate isolated transverse topology jumps to hybrid saltation actions."""

    event_plan: Any = eqx.field(static=True)
    policy: BlockAMRDerivativePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_plan: Any,
        /,
        *,
        policy: BlockAMRDerivativePolicy | None = None,
    ):
        from ...solver._hybrid_event import HybridEventPlan

        policy_ = BlockAMRDerivativePolicy("event-aware") if policy is None else policy
        if not isinstance(event_plan, HybridEventPlan):
            raise TypeError("Event-aware cut derivatives require HybridEventPlan.")
        if (
            not isinstance(policy_, BlockAMRDerivativePolicy)
            or policy_.mode != "event-aware"
        ):
            raise ValueError("Event-aware derivative requires event-aware policy.")
        self.event_plan = event_plan
        self.policy = policy_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "event-aware-cut-cell-derivative",
                "event_plan": event_plan.plan_id,
                "policy": policy_.policy_id,
            }
        )

    def jvp(
        self,
        event_time: ArrayLike,
        state_before: ArrayLike,
        state_tangent: ArrayLike,
        /,
        *,
        args: Any = None,
        time_tangent: ArrayLike = 0.0,
        args_tangent: Any = None,
    ):
        from ...solver._hybrid_event import hybrid_event_jvp

        return hybrid_event_jvp(
            self.event_plan,
            event_time,
            state_before,
            state_tangent,
            args=args,
            time_tangent=time_tangent,
            args_tangent=args_tangent,
        )

    def vjp(
        self,
        event_time: ArrayLike,
        state_before: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        args: Any = None,
    ):
        from ...solver._hybrid_event import hybrid_event_vjp

        return hybrid_event_vjp(
            self.event_plan,
            event_time,
            state_before,
            cotangent,
            args=args,
        )


class RelaxedHierarchyBlendPlan(StrictModule, NonTrainableState):
    """Explicit smooth multiresolution surrogate; never a hard-topology gradient."""

    policy: BlockAMRDerivativePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, policy: BlockAMRDerivativePolicy | None = None, /):
        policy_ = BlockAMRDerivativePolicy("relaxed") if policy is None else policy
        if not isinstance(policy_, BlockAMRDerivativePolicy) or policy_.mode != "relaxed":
            raise ValueError("Relaxed hierarchy blend requires relaxed policy.")
        self.policy = policy_
        self.plan_id = canonical_fingerprint(
            {"kind": "relaxed-hierarchy-blend", "policy": policy_.policy_id}
        )

    def blend(
        self,
        level_values: Sequence[ArrayLike],
        refinement_indicators: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        values = tuple(jnp.asarray(value) for value in level_values)
        if not values or any(value.shape != values[0].shape for value in values):
            raise ValueError("Relaxed hierarchy levels must have one common shape.")
        indicators = jnp.asarray(refinement_indicators)
        if indicators.shape != (len(values),) + values[0].shape[:-1]:
            raise ValueError(
                "Relaxed indicators must be level-leading over value support."
            )
        weights = jax.nn.softmax(
            indicators / self.policy.relaxation_temperature,
            axis=0,
        )
        stacked = jnp.stack(values, axis=0)
        blend = jnp.sum(weights[..., None] * stacked, axis=0)
        return blend, weights


class MappedGeometryDerivativePlan(StrictModule, NonTrainableState):
    """JVP of traceable mapped metrics with fixed patch/cut topology."""

    geometry: CanonicalMappedGeometryPlan
    policy: BlockAMRDerivativePolicy
    topology_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CanonicalMappedGeometryPlan,
        /,
        *,
        topology_margin: float,
    ):
        margin = float(topology_margin)
        if not isinstance(geometry, CanonicalMappedGeometryPlan):
            raise TypeError("Mapped geometry derivatives require mapped geometry plan.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("Mapped topology margin must be finite and nonnegative.")
        self.geometry = geometry
        self.policy = BlockAMRDerivativePolicy("frozen-history")
        self.topology_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-geometry-derivative-plan",
                "geometry": geometry.plan_id,
                "topology_margin": margin,
            }
        )

    def jvp(self, time: ArrayLike, args: Any, args_tangent: Any, /, *, revision=0):
        primal, tangent = jax.jvp(
            lambda parameters: self.geometry.evaluate(
                time,
                parameters,
                revision=revision,
            ),
            (args,),
            (args_tangent,),
        )
        finite_terms = tuple(
            jax.tree.leaves(
                jax.tree.map(
                    lambda value: (
                        jnp.all(jnp.isfinite(value))
                        if eqx.is_inexact_array(value)
                        else jnp.asarray(True)
                    ),
                    tangent,
                )
            )
        )
        finite = jnp.all(jnp.stack(finite_terms)) if finite_terms else jnp.asarray(True)
        evidence = BlockAMRDerivativeEvidence(
            self.policy,
            self.topology_margin,
            jnp.asarray(jnp.inf),
            finite & primal.evidence.valid,
            reason="fixed mapped patch topology and traceable quadrature",
        )
        return primal, tangent, evidence


__all__ = [
    "BlockAMRDerivativeEvidence",
    "BlockAMRDerivativeMode",
    "BlockAMRDerivativePolicy",
    "EventAwareCutCellDerivativePlan",
    "FrozenCutCellDerivativeResult",
    "FrozenCutCellTransitionDerivativePlan",
    "MappedGeometryDerivativePlan",
    "RelaxedHierarchyBlendPlan",
]
