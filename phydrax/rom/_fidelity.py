#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..fidelity import FidelityCaseSpec, FidelityEvaluation, FidelityLevelSpec
from ._affine import PreparedAffineLinearROM


class AffineLinearROMFidelityEvaluator(StrictModule, NonTrainableState):
    """Expose one reduced-only affine model as an honest fidelity level."""

    model: PreparedAffineLinearROM
    level: FidelityLevelSpec
    cost: float = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: PreparedAffineLinearROM,
        level: FidelityLevelSpec,
        /,
        *,
        cost: float,
        observable: str = "state",
        evaluator_id: str | None = None,
    ):
        if not isinstance(model, PreparedAffineLinearROM):
            raise TypeError("model must be a PreparedAffineLinearROM.")
        if not isinstance(level, FidelityLevelSpec):
            raise TypeError("level must be a FidelityLevelSpec.")
        cost_ = float(cost)
        if not np.isfinite(cost_) or cost_ <= 0.0:
            raise ValueError("ROM fidelity cost must be finite and positive.")
        observable_ = str(observable)
        available = {item.name for item in model.observations}
        if observable_ != "state" and observable_ not in available:
            raise ValueError(
                "observable must be 'state' or one prepared observation name."
            )
        if level.model_id != model.model_id:
            raise ValueError(
                "Fidelity level model_id must equal the prepared ROM model_id."
            )
        if level.observable_id != observable_:
            raise ValueError(
                "Fidelity level observable_id must equal the selected ROM observable."
            )
        expected_contract = (
            model.reduction.trial_state_contract_id
            if observable_ == "state"
            else next(
                item.observation_id
                for item in model.observations
                if item.name == observable_
            )
        )
        if level.observable_contract_id != expected_contract:
            raise ValueError(
                "Fidelity level observable contract does not match the prepared ROM output."
            )
        identifier = (
            f"affine-rom:{model.model_id}:{observable_}"
            if evaluator_id is None
            else str(evaluator_id)
        )
        if not identifier:
            raise ValueError("evaluator_id must be non-empty.")
        self.model = model
        self.level = level
        self.cost = cost_
        self.observable = observable_
        self.evaluator_id = identifier

    def __call__(
        self,
        case: FidelityCaseSpec,
        key=None,
        /,
    ) -> FidelityEvaluation:
        del key
        if not isinstance(case, FidelityCaseSpec):
            raise TypeError("case must be a FidelityCaseSpec.")
        state_output = self.observable == "state"
        result = self.model.evaluate(case.inputs, reconstruct=state_output)
        if state_output:
            observable = (
                self.model.reduction.trial.full_space.zeros()
                if result.reconstructed_state is None
                else result.reconstructed_state
            )
        else:
            matching = tuple(
                item for item in result.observations if item.name == self.observable
            )
            if matching:
                observable = matching[0].value
            else:
                prepared = next(
                    item
                    for item in self.model.observations
                    if item.name == self.observable
                )
                observable = prepared.output_space.zeros()
        evidence_ids = tuple(
            dict.fromkeys(
                (
                    self.model.numeric_revision.revision_id,
                    self.model.reduction.reduction_id,
                    self.model.coefficient_map.coefficient_map_id,
                    self.model.coefficient_map.support_id,
                )
            )
        )
        return FidelityEvaluation(
            observable,
            case_id=case.case_id,
            pair_id=case.case_id,
            level_id=self.level.level_id,
            evaluator_id=self.evaluator_id,
            valid=result.valid,
            cost=self.cost,
            cost_unit="relative-cost",
            result=result,
            artifact_id=self.model.model_id,
            evidence_ids=evidence_ids,
        )


__all__ = ["AffineLinearROMFidelityEvaluator"]
