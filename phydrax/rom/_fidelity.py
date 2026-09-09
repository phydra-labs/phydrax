#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..fidelity import FidelityCaseSpec, FidelityEvaluation, FidelityLevelSpec
from ._core import ROMCaseSpec, TruthModel, TruthSample
from ._profiles import (
    LinearCoerciveRBProfile,
    LinearPODProfile,
    ParametricCertifiedProfile,
)
from ._runtime import evaluate, ROMArtifact


class ROMFidelityEvaluator(StrictModule, NonTrainableState):
    """Expose an executable ROM as one fidelity level without truth fallback."""

    artifact: ROMArtifact
    level: FidelityLevelSpec
    truth_model: TruthModel | Callable[[ROMCaseSpec], TruthSample]
    cost: float = eqx.field(static=True)
    observable: Literal["qoi", "state"] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)

    def __init__(
        self,
        artifact: ROMArtifact,
        level: FidelityLevelSpec,
        truth_model: TruthModel | Callable[[ROMCaseSpec], TruthSample],
        /,
        *,
        cost: float,
        observable: Literal["qoi", "state"] = "qoi",
        geometry_id: str = "fixed",
        evaluator_id: str | None = None,
    ):
        if not isinstance(artifact, ROMArtifact):
            raise TypeError("artifact must be a ROMArtifact.")
        if not isinstance(level, FidelityLevelSpec):
            raise TypeError("level must be a FidelityLevelSpec.")
        if not isinstance(
            artifact.profile,
            (LinearPODProfile, LinearCoerciveRBProfile, ParametricCertifiedProfile),
        ):
            raise TypeError(
                "Only ROM profiles with an online reduced solver are fidelities."
            )
        if not callable(truth_model):
            raise TypeError(
                "truth_model must provide ROM operator and right-hand-side data."
            )
        cost_ = float(cost)
        if not jnp.isfinite(cost_) or cost_ <= 0.0:
            raise ValueError("ROM fidelity cost must be finite and positive.")
        if observable not in ("qoi", "state"):
            raise ValueError("observable must be 'qoi' or 'state'.")
        geometry = str(geometry_id)
        if not geometry:
            raise ValueError("geometry_id must be non-empty.")
        identifier = (
            f"rom:{artifact.artifact_id}:{observable}"
            if evaluator_id is None
            else str(evaluator_id)
        )
        if not identifier:
            raise ValueError("evaluator_id must be non-empty.")
        if level.model_id != artifact.artifact_id:
            raise ValueError("Fidelity level model_id must equal the ROM artifact_id.")
        self.artifact = artifact
        self.level = level
        self.truth_model = truth_model
        self.cost = cost_
        self.observable = observable
        self.geometry_id = geometry
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
        if not isinstance(case.inputs, Mapping):
            raise TypeError("ROM fidelity case inputs must be a named parameter mapping.")
        parameters = {str(name): float(value) for name, value in case.inputs.items()}
        rom_case = ROMCaseSpec(case.case_id, parameters, self.geometry_id)
        result = evaluate(
            self.artifact,
            rom_case,
            truth_model=self.truth_model,
            fallback=False,
        )
        if result.source != "rom":
            raise RuntimeError("ROM fidelity evaluation cannot admit truth fallback.")
        if self.observable == "qoi":
            if result.qoi is None:
                raise ValueError(
                    "ROM fidelity requested a QoI that the model did not provide."
                )
            observable = jnp.asarray(result.qoi)
        else:
            observable = jnp.asarray(result.state)
        return FidelityEvaluation(
            observable,
            case_id=case.case_id,
            pair_id=case.case_id,
            level_id=self.level.level_id,
            evaluator_id=self.evaluator_id,
            valid=True,
            cost=self.cost,
            cost_unit="relative-cost",
            result=result,
            artifact_id=self.artifact.artifact_id,
            evidence_ids=(result.lifecycle_revision.revision_id,),
        )


__all__ = ["ROMFidelityEvaluator"]
