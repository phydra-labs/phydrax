from typing import Any

import diffrax as dfx
import jax.numpy as jnp
import pytest

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


@pytest.mark.parametrize(
    ("form", "route", "conditions"),
    (
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
    ),
)
def test_temporal_form_selects_the_derivative_route(
    form: Any, route: Any, conditions: Any
) -> None:
    contract = _evidence(form=form, orientations=("reverse",)).derivative_contract

    assert contract.route is route
    assert _levels(contract) == (GradientLevel.SMOOTH, GradientLevel.SMOOTH)
    assert contract.conditions == conditions


@pytest.mark.parametrize(
    ("overrides", "levels", "conditions"),
    (
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
    ),
)
def test_temporal_semantics_weaken_levels_and_add_conditions(
    overrides: Any, levels: Any, conditions: Any
) -> None:
    contract = _evidence(**overrides).derivative_contract

    assert contract.route is DerivativeRoute.UNROLLED
    assert _levels(contract) == (levels, levels)
    assert contract.conditions == conditions


@pytest.mark.parametrize(
    "overrides",
    (
        {"form": "unknown"},
        {"orientations": ()},
        {"event_semantics": "unsupported"},
        {"event_semantics": "unknown"},
        {"decision_semantics": "backend-defined"},
        {"stochastic_semantics": "unknown"},
    ),
)
def test_unknown_or_unsupported_temporal_derivatives_are_stopped(overrides: Any) -> None:
    contract = _evidence(**overrides).derivative_contract

    assert contract.route is DerivativeRoute.STOPPED
    assert contract.supported_surfaces == ()


def test_adaptive_diffrax_solve_exposes_its_frozen_schedule_contract() -> None:
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
