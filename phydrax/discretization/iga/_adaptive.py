#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...events import DeterministicEventAddress
from ._identity import OverlayCellId
from ._thb import THBBasisCertificate, THBHierarchy


class QoICertificate(StrictModule, NonTrainableState):
    """Well-posedness evidence required before DWR is allowed."""

    qoi_id: str = eqx.field(static=True)
    state_space_id: str = eqx.field(static=True)
    continuity_bound: float = eqx.field(static=True)
    frechet_differentiable: bool = eqx.field(static=True)
    trace_regular: bool = eqx.field(static=True)
    regularized_point_evaluation: bool = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        qoi_id: str,
        state_space_id: str,
        /,
        *,
        continuity_bound: float,
        frechet_differentiable: bool,
        trace_regular: bool,
        point_evaluation: bool = False,
        regularized_point_evaluation: bool = False,
    ):
        qoi = str(qoi_id)
        state = str(state_space_id)
        bound = float(continuity_bound)
        regularized = bool(regularized_point_evaluation)
        if not qoi or not state or not np.isfinite(bound) or bound <= 0.0:
            raise ValueError(
                "QoI certificates require finite positive continuity evidence."
            )
        passed = (
            bool(frechet_differentiable)
            and bool(trace_regular)
            and (not point_evaluation or regularized)
        )
        self.qoi_id = qoi
        self.state_space_id = state
        self.continuity_bound = bound
        self.frechet_differentiable = bool(frechet_differentiable)
        self.trace_regular = bool(trace_regular)
        self.regularized_point_evaluation = regularized
        self.passed = passed
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "qoi-certificate",
                "qoi": qoi,
                "state_space": state,
                "bound": bound,
                "frechet": bool(frechet_differentiable),
                "trace": bool(trace_regular),
                "point": bool(point_evaluation),
                "regularized": regularized,
            }
        )


class DWREstimate(StrictModule, NonTrainableState):
    """Signed cell indicators and complete estimator pollution ledger."""

    cell_ids: tuple[OverlayCellId, ...]
    signed_indicators: Array
    absolute_mass: Array
    pollution: tuple[tuple[str, float], ...] = eqx.field(static=True)
    estimate: float = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_ids: Sequence[OverlayCellId],
        signed_indicators: ArrayLike,
        /,
        *,
        pollution: Sequence[tuple[str, float]],
    ):
        cells = tuple(cell_ids)
        indicators = np.asarray(signed_indicators, dtype=float)
        pollution_ = tuple((str(name), float(value)) for name, value in pollution)
        if (
            not cells
            or indicators.shape != (len(cells),)
            or not np.all(np.isfinite(indicators))
            or any(
                not name or not np.isfinite(value) or value < 0.0
                for name, value in pollution_
            )
        ):
            raise ValueError(
                "DWR estimates require finite cell indicators and pollution."
            )
        if not all(isinstance(cell, OverlayCellId) for cell in cells):
            raise TypeError("cell_ids must contain OverlayCellId values.")
        absolute = np.abs(indicators)
        self.cell_ids = cells
        self.signed_indicators = jnp.asarray(indicators)
        self.absolute_mass = jnp.asarray(absolute)
        self.pollution = pollution_
        self.estimate = float(np.sum(indicators))
        self.estimator_id = canonical_fingerprint(
            {
                "kind": "dwr-estimate",
                "cells": [cell.value for cell in cells],
                "indicators": array_tree_fingerprint(indicators),
                "pollution": list(pollution_),
            }
        )

    def mark_dorfler(self, fraction: float, /) -> tuple[OverlayCellId, ...]:
        theta = float(fraction)
        if not 0.0 < theta <= 1.0:
            raise ValueError("Dorfler marking fraction must lie in (0, 1].")
        mass = np.asarray(self.absolute_mass)
        total = float(np.sum(mass))
        if total == 0.0:
            return ()
        order = np.lexsort((np.arange(mass.size), -mass))
        cumulative = np.cumsum(mass[order])
        count = int(np.searchsorted(cumulative, theta * total, side="left")) + 1
        return tuple(self.cell_ids[index] for index in sorted(order[:count]))


class AdaptiveDesignEpoch(StrictModule, NonTrainableState):
    """Frozen-plan optimization epoch with one explicit Q6 transition boundary."""

    epoch: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)
    accepted_design_id: str = eqx.field(static=True)
    transition_count: int = eqx.field(static=True)
    minimum_iterations: int = eqx.field(static=True)
    event: DeterministicEventAddress
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        epoch: int,
        plan_id: str,
        hierarchy: THBHierarchy,
        accepted_design_id: str,
        /,
        *,
        transition_count: int = 0,
        minimum_iterations: int = 1,
    ):
        epoch_ = int(epoch)
        count = int(transition_count)
        minimum = int(minimum_iterations)
        plan = str(plan_id)
        design = str(accepted_design_id)
        if epoch_ < 0 or count < 0 or minimum <= 0 or not plan or not design:
            raise ValueError("Adaptive design epoch metadata is invalid.")
        event = DeterministicEventAddress("iga-adaptive-design", epoch_, 0, 0)
        self.epoch = epoch_
        self.plan_id = plan
        self.hierarchy_id = hierarchy.hierarchy_id
        self.accepted_design_id = design
        self.transition_count = count
        self.minimum_iterations = minimum
        self.event = event
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "adaptive-design-epoch",
                "epoch": epoch_,
                "plan": plan,
                "hierarchy": hierarchy.hierarchy_id,
                "design": design,
                "transitions": count,
                "minimum_iterations": minimum,
                "event": event.address_id,
            }
        )

    def transition(
        self,
        target_plan_id: str,
        target: THBHierarchy,
        certificate: THBBasisCertificate,
        accepted_design_id: str,
        /,
        *,
        completed_iterations: int,
        maximum_transitions: int,
    ) -> AdaptiveDesignEpoch:
        if completed_iterations < self.minimum_iterations:
            raise ValueError("Adaptive transition violates the frozen-epoch minimum.")
        if self.transition_count >= int(maximum_transitions):
            raise ValueError("Adaptive transition budget is exhausted.")
        if not certificate.passed or target.hierarchy_id == self.hierarchy_id:
            raise ValueError(
                "Adaptive transition requires a new certified THB hierarchy."
            )
        return AdaptiveDesignEpoch(
            self.epoch + 1,
            target_plan_id,
            target,
            accepted_design_id,
            transition_count=self.transition_count + 1,
            minimum_iterations=self.minimum_iterations,
        )


__all__ = ["AdaptiveDesignEpoch", "DWREstimate", "QoICertificate"]
