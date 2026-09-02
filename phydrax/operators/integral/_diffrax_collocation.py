# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any

from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ...integration import (
    ComponentTarget,
    DensityTarget,
    DiffraxCollocationQuadraturePlan,
    integrate_diffrax_collocation,
    IntegrationEstimate,
    IntegrationPrecisionPolicy,
    materialize_diffrax_collocation,
)


class DiffraxCollocationIntegralOperator(StrictModule):
    """Domain-function adapter over the CID-owned fixed collocation realization."""

    target: ComponentTarget | DensityTarget
    plan: DiffraxCollocationQuadraturePlan
    batch: Any

    def __init__(
        self,
        target: ComponentTarget | DensityTarget,
        plan: DiffraxCollocationQuadraturePlan,
        /,
    ):
        if not isinstance(plan, DiffraxCollocationQuadraturePlan):
            raise TypeError("plan must be CID DiffraxCollocationQuadraturePlan.")
        self.target = target
        self.plan = plan
        self.batch = materialize_diffrax_collocation(target, plan)

    def __call__(
        self,
        integrand: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        kwargs: dict[str, Any] | None = None,
        precision: IntegrationPrecisionPolicy | None = None,
    ) -> IntegrationEstimate:
        return integrate_diffrax_collocation(
            integrand,
            self.target,
            self.batch,
            self.plan,
            key=key,
            kwargs=kwargs,
            precision=precision,
        )


__all__ = ["DiffraxCollocationIntegralOperator"]
