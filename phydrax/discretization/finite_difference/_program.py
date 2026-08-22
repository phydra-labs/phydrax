#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._execution import lower_stencil_operator, PreparedStencilExecutionOperator
from ._plan import PreparedFiniteDifferenceDiscretization


class StencilAssignment(StrictModule, NonTrainableState):
    """One scaled derivative read accumulated into a target field."""

    target: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    operator: str = eqx.field(static=True)
    scale: complex = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: str,
        source: str,
        operator: str,
        /,
        *,
        scale: complex = 1.0,
    ):
        target_ = str(target)
        source_ = str(source)
        operator_ = str(operator)
        scale_ = complex(scale)
        if not target_ or not source_ or not operator_:
            raise ValueError("Stencil assignment names must be non-empty.")
        self.target = target_
        self.source = source_
        self.operator = operator_
        self.scale = scale_
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "stencil-assignment",
                "target": target_,
                "source": source_,
                "operator": operator_,
                "scale": [scale_.real, scale_.imag],
            }
        )


class FDPipelineReport(StrictModule, NonTrainableState):
    """Unique applications, CSE reuse, metadata bytes, and fusion evidence."""

    assignment_count: int = eqx.field(static=True)
    unique_application_count: int = eqx.field(static=True)
    reused_application_count: int = eqx.field(static=True)
    canonical_metadata_bytes: int = eqx.field(static=True)
    lowered_metadata_bytes: int = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        assignment_count: int,
        unique_application_count: int,
        canonical_metadata_bytes: int,
        lowered_metadata_bytes: int,
        plan_id: str,
    ):
        assignments = int(assignment_count)
        unique = int(unique_application_count)
        canonical = int(canonical_metadata_bytes)
        lowered = int(lowered_metadata_bytes)
        self.assignment_count = assignments
        self.unique_application_count = unique
        self.reused_application_count = assignments - unique
        self.canonical_metadata_bytes = canonical
        self.lowered_metadata_bytes = lowered
        self.passed = 0 < unique <= assignments and lowered <= canonical
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-pipeline-report",
                "plan": plan_id,
                "assignment_count": assignments,
                "unique_application_count": unique,
                "canonical_metadata_bytes": canonical,
                "lowered_metadata_bytes": lowered,
            }
        )


class StencilProgramPlan(StrictModule, NonTrainableState):
    """Read/write-aware local stencil program with reusable derivative applications."""

    discretization: PreparedFiniteDifferenceDiscretization
    field_names: tuple[str, ...] = eqx.field(static=True)
    assignments: tuple[StencilAssignment, ...]
    read_fields: tuple[str, ...] = eqx.field(static=True)
    written_fields: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedFiniteDifferenceDiscretization,
        field_names: Sequence[str],
        assignments: Sequence[StencilAssignment],
        /,
        *,
        plan_id: str | None = None,
    ):
        if not isinstance(discretization, PreparedFiniteDifferenceDiscretization):
            raise TypeError(
                "discretization must be PreparedFiniteDifferenceDiscretization."
            )
        fields = tuple(str(name) for name in field_names)
        if (
            not fields
            or any(not name for name in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError("field_names must be unique and non-empty.")
        assignments_ = tuple(assignments)
        if not assignments_ or not all(
            isinstance(value, StencilAssignment) for value in assignments_
        ):
            raise TypeError("assignments must contain StencilAssignment values.")
        known = set(fields)
        if any(
            value.source not in known or value.target not in known
            for value in assignments_
        ):
            raise ValueError("Stencil assignment fields must be declared.")
        for value in assignments_:
            discretization.operator(value.operator)
        reads = tuple(dict.fromkeys(value.source for value in assignments_))
        writes = tuple(dict.fromkeys(value.target for value in assignments_))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "stencil-program-plan",
                    "discretization": discretization.prepared_id,
                    "fields": list(fields),
                    "assignments": [value.assignment_id for value in assignments_],
                    "halo": discretization.halo_plan.halo_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.discretization = discretization
        self.field_names = fields
        self.assignments = assignments_
        self.read_fields = reads
        self.written_fields = writes
        self.plan_id = identifier

    def prepare(self, /) -> "PreparedStencilProgram":
        return PreparedStencilProgram(self)


class PreparedStencilProgram(StrictModule, NonTrainableState):
    """Fused JAX pipeline with compact kernels and derivative CSE."""

    plan: StencilProgramPlan
    execution_operators: tuple[tuple[str, PreparedStencilExecutionOperator], ...]
    report: FDPipelineReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: StencilProgramPlan, /):
        if not isinstance(plan, StencilProgramPlan):
            raise TypeError("plan must be a StencilProgramPlan.")
        operator_names = tuple(
            dict.fromkeys(value.operator for value in plan.assignments)
        )
        execution = tuple(
            (
                name,
                lower_stencil_operator(plan.discretization.operator(name)),
            )
            for name in operator_names
        )
        canonical_bytes = sum(
            value.execution.report.canonical_metadata_bytes for _, value in execution
        )
        lowered_bytes = sum(
            value.execution.report.lowered_metadata_bytes for _, value in execution
        )
        unique_applications = len(
            {(assignment.source, assignment.operator) for assignment in plan.assignments}
        )
        report = FDPipelineReport(
            assignment_count=len(plan.assignments),
            unique_application_count=unique_applications,
            canonical_metadata_bytes=canonical_bytes,
            lowered_metadata_bytes=lowered_bytes,
            plan_id=plan.plan_id,
        )
        if not report.passed:
            raise RuntimeError("Prepared FD pipeline failed its CSE/metadata evidence.")
        self.plan = plan
        self.execution_operators = execution
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-stencil-program",
                "plan": plan.plan_id,
                "executions": [value.operator_id for _, value in execution],
                "report": report.report_id,
            }
        )

    def _operator(self, name: str, /) -> PreparedStencilExecutionOperator:
        selected = tuple(value for key, value in self.execution_operators if key == name)
        if len(selected) != 1:
            raise RuntimeError("Prepared FD pipeline lost a unique execution operator.")
        return selected[0]

    def __call__(
        self,
        state: Mapping[str, ArrayLike],
        /,
    ) -> dict[str, Array]:
        if set(state) != set(self.plan.field_names):
            raise ValueError("Stencil program state fields must match the plan exactly.")
        values = {name: jnp.asarray(state[name]) for name in self.plan.field_names}
        output = {name: jnp.zeros_like(values[name]) for name in self.plan.written_fields}
        derivative_cache: dict[tuple[str, str], Array] = {}
        for assignment in self.plan.assignments:
            cache_key = (assignment.source, assignment.operator)
            if cache_key not in derivative_cache:
                derivative_cache[cache_key] = self._operator(assignment.operator).mv(
                    values[assignment.source]
                )
            contribution = assignment.scale * derivative_cache[cache_key]
            output[assignment.target] = output[assignment.target] + contribution
        return output


__all__ = [
    "FDPipelineReport",
    "PreparedStencilProgram",
    "StencilAssignment",
    "StencilProgramPlan",
]
