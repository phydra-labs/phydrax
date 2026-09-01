# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Literal

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import GeneralizedAlphaMethod
from ._fem_dynamics import ImplicitNewmarkMethod
from ._iga_rods import IGARodPlan
from ._iga_shells import IGAKirchhoffLovePlan, IGAReissnerMindlinPlan
from ._iga_solids import _identifier, IGASolidFormulation


IGATimeMethod = Literal["Newmark", "generalized_alpha"]
IGAFormulation = (
    IGAKirchhoffLovePlan | IGAReissnerMindlinPlan | IGARodPlan | IGASolidFormulation
)


class IGADynamicsPlan(StrictModule, NonTrainableState):
    """Lifecycle-bound use of the native Newmark or generalized-alpha integrator."""

    formulation: IGAFormulation
    method: ImplicitNewmarkMethod | GeneralizedAlphaMethod
    lifecycle_plan_id: str = eqx.field(static=True)
    lifecycle_state_id: str = eqx.field(static=True)
    temporal_method: IGATimeMethod = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: IGAFormulation,
        method: ImplicitNewmarkMethod | GeneralizedAlphaMethod,
        /,
        *,
        lifecycle_plan_id: str,
        lifecycle_state_id: str,
    ):
        if not isinstance(
            formulation,
            (
                IGAKirchhoffLovePlan,
                IGAReissnerMindlinPlan,
                IGARodPlan,
                IGASolidFormulation,
            ),
        ):
            raise TypeError("formulation must be a certified IGA mechanics formulation.")
        if isinstance(method, ImplicitNewmarkMethod):
            temporal = "Newmark"
            profile = "IGA.Structures.Dynamics.Newmark"
        elif isinstance(method, GeneralizedAlphaMethod):
            temporal = "generalized_alpha"
            profile = "IGA.Structures.Dynamics.GeneralizedAlpha"
        else:
            raise TypeError(
                "method must be ImplicitNewmarkMethod or GeneralizedAlphaMethod."
            )
        self.formulation = formulation
        self.method = method
        self.lifecycle_plan_id = _identifier(lifecycle_plan_id, "lifecycle_plan_id")
        self.lifecycle_state_id = _identifier(lifecycle_state_id, "lifecycle_state_id")
        self.temporal_method = temporal
        self.profile_id = profile
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-dynamics-plan",
                "profile": profile,
                "formulation": formulation.plan_id,
                "method": method.method_id,
                "lifecycle_plan": self.lifecycle_plan_id,
                "lifecycle_state": self.lifecycle_state_id,
            }
        )


__all__ = ["IGADynamicsPlan", "IGAFormulation", "IGATimeMethod"]
