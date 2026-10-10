#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...events import DeterministicEventAddress
from ...typing import parse
from ._identity import OverlayCellId
from ._thb import THBBasisCertificate, THBHierarchy
from ._transfer import TransferPlan


if TYPE_CHECKING:
    from ...meshing._decision import (
        AdaptationAction,
        PhysicalErrorEvidence,
        SolverAwareDecision,
    )


type DWRPollutionCategory = Literal["field", "geometry", "algebraic", "transfer"]


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
    ) -> None:
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
    qoi_certificate_id: str | None = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_ids: Sequence[OverlayCellId],
        signed_indicators: ArrayLike,
        /,
        *,
        pollution: Sequence[tuple[str, float]],
        qoi_certificate: QoICertificate | None = None,
    ) -> None:
        cells = tuple(cell_ids)
        indicators = np.asarray(signed_indicators, dtype=np.float64)
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
        if qoi_certificate is not None:
            if not isinstance(qoi_certificate, QoICertificate):
                raise TypeError("qoi_certificate must be QoICertificate or None.")
            if not qoi_certificate.passed:
                raise ValueError("DWR QoI binding requires a passed certificate.")
        certificate_id = (
            None if qoi_certificate is None else qoi_certificate.certificate_id
        )
        absolute = np.abs(indicators)
        self.cell_ids = cells
        self.signed_indicators = jnp.asarray(indicators)
        self.absolute_mass = jnp.asarray(absolute)
        self.pollution = pollution_
        self.estimate = float(np.sum(indicators))
        self.qoi_certificate_id = certificate_id
        self.estimator_id = canonical_fingerprint(
            {
                "kind": "dwr-estimate",
                "cells": [cell.value for cell in cells],
                "indicators": array_tree_fingerprint(indicators),
                "pollution": list(pollution_),
                "qoi_certificate": certificate_id,
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

    def physical_error_evidence(
        self,
        revision_id: str,
        certificate: QoICertificate,
        /,
        *,
        qoi_id: str,
        state_space_id: str,
        pollution_categories: Mapping[str, DWRPollutionCategory],
    ) -> PhysicalErrorEvidence:
        """Account for absolute DWR mass and explicitly owned QoI pollution.

        Pollution bounds must already be in the certified QoI's units. A solver
        residual, geometry defect, or transfer norm is not itself a QoI bound.
        An estimate without a construction-time QoI binding is usable for
        marking, not physical admission. This conversion does not certify the
        estimator's reliability.
        """
        from ...meshing._decision import PhysicalErrorEvidence

        if not isinstance(certificate, QoICertificate):
            raise TypeError("certificate must be QoICertificate.")
        if (
            not certificate.passed
            or self.qoi_certificate_id != certificate.certificate_id
            or certificate.qoi_id != qoi_id
            or certificate.state_space_id != state_space_id
        ):
            raise ValueError("DWR evidence requires matching passed QoI certification.")
        names = tuple(name for name, _ in self.pollution)
        if len(set(names)) != len(names) or set(pollution_categories) != set(names):
            raise ValueError("Every DWR pollution entry requires one explicit owner.")
        components = {
            "field": float(np.sum(np.asarray(self.absolute_mass))),
            "geometry": 0.0,
            "algebraic": 0.0,
            "transfer": 0.0,
        }
        ownership = []
        for name, value in self.pollution:
            category = parse(
                pollution_categories[name], DWRPollutionCategory, "pollution category"
            )
            components[category] += value
            ownership.append((name, category))
        estimator_id = canonical_fingerprint(
            {
                "kind": "iga-dwr-physical-evidence",
                "estimator": self.estimator_id,
                "qoi_certificate": certificate.certificate_id,
                "pollution_ownership": sorted(ownership),
            }
        )
        return PhysicalErrorEvidence(
            revision_id,
            certificate.qoi_id,
            estimator_id,
            field_error=components["field"],
            geometry_error=components["geometry"],
            algebraic_error=components["algebraic"],
            transfer_error=components["transfer"],
            quantity="qoi",
            qoi_certificate_id=certificate.certificate_id,
        )


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
    ) -> None:
        epoch_ = int(epoch)
        count = int(transition_count)
        minimum = int(minimum_iterations)
        plan = str(plan_id)
        design = str(accepted_design_id)
        if epoch_ < 0 or count < 0 or minimum <= 0 or not plan or not design:
            raise ValueError("Adaptive design epoch metadata is invalid.")
        event = DeterministicEventAddress("iga-adaptive-design", epoch_, 0, 0, 0, 0, 0)
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

    def transition_candidate_id(
        self,
        target_plan_id: str,
        target: THBHierarchy,
        certificate: THBBasisCertificate,
        accepted_design_id: str,
        transfer: TransferPlan,
        action: AdaptationAction,
        /,
    ) -> str:
        """Identify the exact epoch route, including approximate versus exact transfer."""
        from ...meshing._decision import AdaptationAction

        if not isinstance(transfer, TransferPlan):
            raise TypeError("transfer must be TransferPlan.")
        if not isinstance(action, AdaptationAction):
            raise TypeError("action must be AdaptationAction.")
        if action not in (AdaptationAction.H, AdaptationAction.P):
            raise ValueError("THB epoch transitions require an h or p decision.")
        if (
            transfer.source_plan_id != self.plan_id
            or transfer.target_plan_id != target_plan_id
        ):
            raise ValueError("IGA transition transfer is bound to different plans.")
        if not certificate.passed or certificate.hierarchy_id != target.hierarchy_id:
            raise ValueError("IGA transition requires the target basis certificate.")
        return canonical_fingerprint(
            {
                "kind": "iga-adaptive-transition",
                "source_epoch": self.epoch_id,
                "target_plan": target_plan_id,
                "target_hierarchy": target.hierarchy_id,
                "target_certificate": certificate.certificate_id,
                "accepted_design": accepted_design_id,
                "transfer": transfer.plan_id,
                "transfer_class": transfer.evidence.transfer_class,
                "action": action.value,
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
        decision: SolverAwareDecision | None = None,
        decision_action: AdaptationAction | None = None,
        transfer: TransferPlan | None = None,
        reanalysis: PhysicalErrorEvidence | None = None,
    ) -> AdaptiveDesignEpoch:
        if completed_iterations < self.minimum_iterations:
            raise ValueError("Adaptive transition violates the frozen-epoch minimum.")
        if self.transition_count >= int(maximum_transitions):
            raise ValueError("Adaptive transition budget is exhausted.")
        if (
            not certificate.passed
            or certificate.hierarchy_id != target.hierarchy_id
            or target.hierarchy_id == self.hierarchy_id
        ):
            raise ValueError(
                "Adaptive transition requires a new certified THB hierarchy."
            )
        if decision is None:
            if (
                decision_action is not None
                or transfer is not None
                or reanalysis is not None
            ):
                raise ValueError("Decision transition evidence requires a decision.")
        else:
            from ...meshing._decision import PhysicalErrorEvidence, SolverAwareDecision

            if not isinstance(decision, SolverAwareDecision):
                raise TypeError("decision must be SolverAwareDecision.")
            if decision_action is None or transfer is None or reanalysis is None:
                raise ValueError(
                    "IGA decisions require an action, qualified transfer, and reanalysis."
                )
            if not isinstance(reanalysis, PhysicalErrorEvidence):
                raise TypeError("reanalysis must be PhysicalErrorEvidence.")
            candidate_id = self.transition_candidate_id(
                target_plan_id,
                target,
                certificate,
                accepted_design_id,
                transfer,
                decision_action,
            )
            selected = decision.require_selected(
                transfer.source_revision_id, candidate_id, transfer.target_revision_id
            )
            if selected.action != decision_action:
                raise ValueError("IGA decision selects a different adaptation action.")
            if reanalysis.quantity != "qoi":
                raise ValueError(
                    "IGA decision reanalysis requires certified QoI evidence."
                )
            decision.require_reanalysis(reanalysis)
        return AdaptiveDesignEpoch(
            self.epoch + 1,
            target_plan_id,
            target,
            accepted_design_id,
            transition_count=self.transition_count + 1,
            minimum_iterations=self.minimum_iterations,
        )


__all__ = ["AdaptiveDesignEpoch", "DWREstimate", "QoICertificate"]
