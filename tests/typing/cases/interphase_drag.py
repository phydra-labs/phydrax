"""Public interphase-drag selector, law, and evidence annotations."""

from typing import assert_type

from jax import Array

from phydrax.equations import (
    AbstractInterphaseDragLaw,
    InterphaseDragCorrelation,
    InterphaseDragEvaluation,
    InterphaseDragLawPlan,
)


def drag_contract(law: AbstractInterphaseDragLaw) -> None:
    assert_type(law.correlation, InterphaseDragCorrelation)
    assert_type(law.law_id, str)
    result = law.evaluate(0.9, 0.01, 0.001, 0.001, 1000.0, 1.0e-9)
    assert_type(result, InterphaseDragEvaluation)
    assert_type(result.beta, Array)
    assert_type(result.particle_drag_coefficient, Array)
    assert_type(result.reynolds_number, Array)
    assert_type(result.branch_code, Array)
    assert_type(result.branch_margin, Array)
    assert_type(result.derivative_valid, Array)
    assert_type(result.admissible, Array)
    assert_type(result.successful, Array)
    assert_type(result.law_id, str)


stokes = InterphaseDragLawPlan("stokes", maximum_reynolds=0.1)
assert_type(stokes, InterphaseDragLawPlan)
assert_type(stokes.correlation, InterphaseDragCorrelation)
assert_type(stokes.maximum_reynolds, float)
drag_contract(stokes)
drag_contract(InterphaseDragLawPlan("schiller_naumann"))
drag_contract(InterphaseDragLawPlan("di_felice"))
drag_contract(InterphaseDragLawPlan("ergun"))
drag_contract(InterphaseDragLawPlan("wen_yu"))
drag_contract(InterphaseDragLawPlan("gidaspow"))
drag_contract(InterphaseDragLawPlan("huilin_gidaspow"))
drag_contract(InterphaseDragLawPlan("beetstra"))

InterphaseDragLawPlan("invented")  # ty: ignore[invalid-argument-type]
stokes.evaluate("invalid", 0.01, 0.001, 0.001, 1000.0, 1.0e-9)  # ty: ignore[invalid-argument-type]
