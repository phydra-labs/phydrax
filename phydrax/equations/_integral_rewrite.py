# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..integration import IntegrationRealization
from ._ir import PDEExpression, PDEProblemIR


@dataclass(frozen=True, slots=True)
class StrongToIntegralRewriteSpec:
    equation: str
    coordinate: str
    trace_ids: tuple[str, ...]
    form: Literal["temporal", "weak_spatial"] = "temporal"
    test_space_id: str | None = None
    boundary_evidence_id: str | None = None

    def __post_init__(self):
        if not self.equation or not self.coordinate or not self.trace_ids:
            raise ValueError(
                "Integral rewrite requires equation, coordinate, and traces."
            )
        if self.form not in ("temporal", "weak_spatial"):
            raise ValueError("Unknown strong-to-integral rewrite form.")
        if self.form == "weak_spatial" and (
            not self.test_space_id or not self.boundary_evidence_id
        ):
            raise ValueError("Weak spatial rewrite requires test and boundary evidence.")


@dataclass(frozen=True, slots=True)
class IntegralResidualProgram:
    problem: PDEProblemIR
    equation_id: str
    coordinate: str
    derivative_order: int
    trace_ids: tuple[str, ...]
    form: Literal["temporal", "weak_spatial"]
    integration: IntegrationRealization
    integration_identity: str
    strong_norm_equivalent: bool = False


def _derivatives(
    expression: PDEExpression, coordinate: str, /
) -> tuple[PDEExpression, ...]:
    values = []
    if expression.op == "derivative" and expression.coordinate == coordinate:
        values.append(expression)
    for argument in expression.args:
        values.extend(_derivatives(argument, coordinate))
    return tuple(values)


def rewrite_strong_to_integral(
    problem_ir: PDEProblemIR,
    spec: StrongToIntegralRewriteSpec,
    integration_plan: IntegrationRealization,
    /,
) -> IntegralResidualProgram:
    """Validate a typed derivative isolation and emit its integral residual program."""
    if not isinstance(problem_ir, PDEProblemIR):
        raise TypeError("problem_ir must be PDEProblemIR.")
    if not isinstance(spec, StrongToIntegralRewriteSpec):
        raise TypeError("spec must be StrongToIntegralRewriteSpec.")
    if not isinstance(integration_plan, IntegrationRealization):
        raise TypeError("integration_plan must be CID IntegrationRealization.")
    equations = {equation.name: equation for equation in problem_ir.equations}
    if spec.equation not in equations:
        raise KeyError(f"Unknown PDE equation {spec.equation!r}.")
    if spec.coordinate not in {coordinate.name for coordinate in problem_ir.coordinates}:
        raise KeyError(f"Unknown PDE coordinate {spec.coordinate!r}.")
    derivatives = _derivatives(equations[spec.equation].residual, spec.coordinate)
    if len(derivatives) != 1:
        raise ValueError(
            "Rewrite requires exactly one isolated leading total derivative."
        )
    derivative = derivatives[0]
    required_traces = derivative.order
    if len(spec.trace_ids) < required_traces:
        raise ValueError("Integral rewrite is missing required derivative trace data.")
    identity = f"{type(integration_plan.plan).__module__}.{type(integration_plan.plan).__qualname__}"
    return IntegralResidualProgram(
        problem_ir,
        spec.equation,
        spec.coordinate,
        derivative.order,
        spec.trace_ids,
        spec.form,
        integration_plan,
        identity,
        False,
    )


__all__ = [
    "IntegralResidualProgram",
    "StrongToIntegralRewriteSpec",
    "rewrite_strong_to_integral",
]
