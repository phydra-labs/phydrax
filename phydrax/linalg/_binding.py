#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._plans import LinearSolvePlan


class LinearSolveTemplate(StrictModule):
    """Reusable symbolic analysis for one fixed linear-solve structure.

    A template contains no coefficient-dependent factorization. Numerical arrays are
    introduced only by :func:`phydrax.linalg.bind_numeric`.
    """

    plan: LinearSolvePlan
    symbolic_state: Any
    problem_signature: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    device_bindable: bool = eqx.field(static=True)
    rejection_reason: str | None = eqx.field(static=True)
    schema_version: int = eqx.field(static=True)
    template_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LinearSolvePlan,
        symbolic_state: Any,
        /,
        *,
        device_bindable: bool,
        source_space_id: str,
        target_space_id: str,
        batch_shape: tuple[int, ...],
        rejection_reason: str | None = None,
        schema_version: int = 1,
    ):
        if not isinstance(plan, LinearSolvePlan):
            raise TypeError("plan must be a LinearSolvePlan.")
        version = int(schema_version)
        if version < 1:
            raise ValueError("schema_version must be positive.")
        reason = None if rejection_reason is None else str(rejection_reason)
        if reason == "":
            raise ValueError("rejection_reason must be non-empty or None.")
        bindable = bool(device_bindable)
        if bindable and reason is not None:
            raise ValueError("A device-bindable template cannot have a rejection reason.")
        if not bindable and reason is None:
            raise ValueError(
                "A non-device-bindable template requires a rejection reason."
            )
        source_id = str(source_space_id)
        target_id = str(target_space_id)
        if not source_id or not target_id:
            raise ValueError("Template source and target space IDs must be non-empty.")
        batch = tuple(int(size) for size in batch_shape)
        if any(size < 0 for size in batch):
            raise ValueError("Template batch dimensions must be nonnegative.")
        self.plan = plan
        self.symbolic_state = symbolic_state
        self.problem_signature = plan.problem_signature
        self.source_space_id = source_id
        self.target_space_id = target_id
        self.batch_shape = batch
        self.device_bindable = bindable
        self.rejection_reason = reason
        self.schema_version = version
        self.template_id = canonical_fingerprint(
            {
                "kind": "linear-solve-template",
                "plan": plan.plan_id,
                "backend": plan.backend,
                "schema_version": version,
                "device_bindable": bindable,
                "source": source_id,
                "target": target_id,
                "batch_shape": list(batch),
            }
        )


__all__ = ["LinearSolveTemplate"]
