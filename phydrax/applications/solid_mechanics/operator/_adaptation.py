#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import equinox as eqx
from jaxtyping import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....nn.operator.training._adaptation import (
    adapt_operator_context,
    BoundedResidualAdaptationPolicy,
    TestTimeAdaptationResult,
)
from ....nn.operator.training._trained_operator import TrainedOperator
from ._cases import MechanicsOperatorCase
from ._parameters import MechanicsParameterSpec
from ._qualification import assess_mechanics_support, MechanicsSupportEvidence


class MechanicsFineTuningPolicy(StrictModule, NonTrainableState):
    """Explicit bounded context adaptation; never an amortized inference policy."""

    context_policy: BoundedResidualAdaptationPolicy
    allowed_observable_ids: tuple[str, ...] = eqx.field(static=True)
    residual_objective_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    policy_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        context_policy: BoundedResidualAdaptationPolicy,
        /,
        *,
        allowed_observable_ids: Sequence[str],
        residual_objective_id: str,
        policy_id: str,
    ):
        if not isinstance(context_policy, BoundedResidualAdaptationPolicy):
            raise TypeError("context_policy must be a BoundedResidualAdaptationPolicy.")
        observables = tuple(str(value) for value in allowed_observable_ids)
        if (
            not observables
            or any(not value for value in observables)
            or len(set(observables)) != len(observables)
        ):
            raise ValueError(
                "Mechanics adaptation observable IDs must be non-empty and unique."
            )
        objective_id = str(residual_objective_id)
        identifier = str(policy_id)
        if not objective_id or not identifier:
            raise ValueError(
                "Mechanics adaptation objective and policy IDs must be non-empty."
            )
        fingerprint = canonical_fingerprint(
            {
                "kind": "mechanics-fine-tuning-policy",
                "policy_id": identifier,
                "residual_objective": objective_id,
                "allowed_observables": list(observables),
                "iterations": context_policy.iterations,
                "learning_rate": context_policy.learning_rate,
                "maximum_update_norm": context_policy.maximum_update_norm,
                "gradient_clip_norm": context_policy.gradient_clip_norm,
            }
        )
        self.context_policy = context_policy
        self.allowed_observable_ids = observables
        self.residual_objective_id = objective_id
        self.policy_id = identifier
        self.policy_fingerprint = fingerprint


class AdaptedMechanicsOperatorResult(StrictModule):
    """Separately typed adapted context retaining an immutable trained base."""

    base_operator: TrainedOperator
    case: MechanicsOperatorCase
    adaptation: TestTimeAdaptationResult
    support: MechanicsSupportEvidence = eqx.field(static=True)
    metadata: frozendict[str, str] = eqx.field(static=True)
    adaptation_id: str = eqx.field(static=True)
    adaptation_kind: Literal["bounded_residual_context"] = eqx.field(static=True)

    @property
    def base_artifact_id(self) -> str:
        return self.base_operator.artifact_id

    @property
    def adapted_context(self):
        return self.adaptation.context


def fine_tune_mechanics_operator(
    base_operator: TrainedOperator,
    case: MechanicsOperatorCase,
    initial_context: ArrayLike,
    residual_objective: Callable,
    /,
    *,
    policy: MechanicsFineTuningPolicy,
    support_spec: MechanicsParameterSpec,
    required_metadata: Mapping[str, str],
    lower_bound: ArrayLike | None = None,
    upper_bound: ArrayLike | None = None,
    jit: bool = True,
) -> AdaptedMechanicsOperatorResult:
    """Adapt only an explicit context while retaining the exact frozen base model."""
    if not isinstance(base_operator, TrainedOperator):
        raise TypeError("base_operator must be a TrainedOperator.")
    if not isinstance(case, MechanicsOperatorCase):
        raise TypeError("case must be a MechanicsOperatorCase.")
    if not isinstance(policy, MechanicsFineTuningPolicy):
        raise TypeError("policy must be a MechanicsFineTuningPolicy.")
    if not isinstance(support_spec, MechanicsParameterSpec):
        raise TypeError("support_spec must be a MechanicsParameterSpec.")
    if not callable(residual_objective):
        raise TypeError("residual_objective must be callable.")
    bindings = frozendict(
        {str(name): str(value) for name, value in required_metadata.items()}
    )
    if not bindings or any(not name or not value for name, value in bindings.items()):
        raise ValueError("Mechanics adaptation requires metadata fingerprint bindings.")
    missing = set(bindings) - set(base_operator.provenance)
    if missing:
        raise ValueError(
            f"Base operator is missing mechanics metadata bindings {sorted(missing)}."
        )
    mismatched = tuple(
        name
        for name, expected in bindings.items()
        if base_operator.provenance[name] != expected
    )
    if mismatched:
        raise ValueError(
            f"Base operator mechanics metadata mismatch for {sorted(mismatched)}."
        )
    support = assess_mechanics_support(
        support_spec,
        case.realization,
        geometry_fingerprint=case.geometry.geometry_fingerprint,
    )
    if not support.supported:
        raise ValueError("Mechanics adaptation is unavailable outside declared support.")
    adapted = adapt_operator_context(
        initial_context,
        residual_objective,
        policy=policy.context_policy,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        jit=jit,
    )
    adaptation_id = canonical_fingerprint(
        {
            "kind": "adapted-mechanics-operator-result",
            "base_artifact": base_operator.artifact_id,
            "base_contract": base_operator.contract_fingerprint,
            "case": case.case_fingerprint,
            "policy": policy.policy_fingerprint,
            "support": support.status,
        }
    )
    metadata = dict(bindings)
    metadata.update(
        {
            "mechanics_adaptation_policy_fingerprint": policy.policy_fingerprint,
            "mechanics_adaptation_case_fingerprint": case.case_fingerprint,
            "mechanics_adaptation_base_contract_fingerprint": (
                base_operator.contract_fingerprint
            ),
        }
    )
    return AdaptedMechanicsOperatorResult(
        base_operator=base_operator,
        case=case,
        adaptation=adapted,
        support=support,
        metadata=frozendict(metadata),
        adaptation_id=adaptation_id,
        adaptation_kind="bounded_residual_context",
    )


__all__ = [
    "AdaptedMechanicsOperatorResult",
    "MechanicsFineTuningPolicy",
    "fine_tune_mechanics_operator",
]
