from typing import Any

import diffrax as dfx
import jax.numpy as jnp

import phydrax as phx
from phydrax import DerivativeRoute, DerivativeSurface, GradientLevel


SURFACES = (DerivativeSurface.PRIMAL_STATE, DerivativeSurface.PHYSICAL_PARAMETER)


def _evidence(**overrides: Any) -> Any:
    values = {
        "form": "discretize-then-optimize",
        "orientations": ("forward", "reverse"),
        "checkpointing": "none",
        "checkpoint_count": None,
        "decision_semantics": "fixed-grid",
        "event_semantics": "none",
        "stochastic_semantics": "deterministic",
        "implementation_id": "test-derivative",
    }
    values.update(overrides)
    # ty: ignore[invalid-argument-type]
    return phx.solver.TemporalDifferentiationEvidence(**values)


def _levels(contract: Any) -> Any:
    return tuple(contract.level(surface) for surface in SURFACES)


def test_temporal_derivative_contract_scenario_1() -> None:
    for form, route, conditions in (
        ("discretize-then-optimize", DerivativeRoute.UNROLLED, ()),
        (
            "implicit-solution-map",
            DerivativeRoute.IMPLICIT,
            ("steady-state-reached",),
        ),
        (
            "optimize-then-discretize",
            DerivativeRoute.EXTERNAL_ADJOINT,
            ("continuous-adjoint-approximation",),
        ),
    ):
        contract = _evidence(form=form, orientations=("reverse",)).derivative_contract

        assert contract.route is route
        assert _levels(contract) == (GradientLevel.SMOOTH, GradientLevel.SMOOTH)
        assert contract.conditions == conditions
    for overrides, levels, conditions in (
        (
            {"decision_semantics": "frozen-adaptive-schedule"},
            GradientLevel.SMOOTH,
            ("decisions-frozen",),
        ),
        (
            {"event_semantics": "backend-branchwise-unqualified"},
            GradientLevel.ALMOST_EVERYWHERE,
            ("executed-branch",),
        ),
        (
            {"event_semantics": "implicit-event-replay"},
            GradientLevel.ALMOST_EVERYWHERE,
            ("transversal-events",),
        ),
        (
            {"stochastic_semantics": "fixed-realization-pathwise"},
            GradientLevel.SMOOTH,
            ("fixed-realization",),
        ),
        (
            {"verified": False},
            GradientLevel.CONDITIONAL,
            ("derivative-classification-unverified",),
        ),
    ):
        contract = _evidence(**overrides).derivative_contract

        assert contract.route is DerivativeRoute.UNROLLED
        assert _levels(contract) == (levels, levels)
        assert contract.conditions == conditions
    for overrides in (
        {"form": "unknown"},
        {"orientations": ()},
        {"event_semantics": "unsupported"},
        {"event_semantics": "unknown"},
        {"decision_semantics": "backend-defined"},
        {"stochastic_semantics": "unknown"},
    ):
        contract = _evidence(**overrides).derivative_contract

        assert contract.route is DerivativeRoute.STOPPED
        assert contract.supported_surfaces == ()
    problem = phx.solver.DifferentialProblem(
        lambda time, state, rate: -rate * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(1.0),
        problem_id="temporal-contract-decay",
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 1.0]),
        adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=8),
    )
    # ty: ignore[unresolved-attribute]
    contract = solution.temporal_evidence.differentiation.derivative_contract

    assert contract.route is DerivativeRoute.UNROLLED
    assert _levels(contract) == (GradientLevel.SMOOTH, GradientLevel.SMOOTH)
    assert contract.conditions == ("decisions-frozen",)
