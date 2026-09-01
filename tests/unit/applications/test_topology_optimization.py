from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax import optim
from phydrax.applications.solid_mechanics._topology import (
    solve_topology_optimization,
    TopologyContinuationSchedule,
    TopologyContinuationStage,
    TopologyMechanicsProblem,
)
from phydrax.applications.solid_mechanics._topology_design import (
    Aggregation,
    DensityTransform,
    hill_mandel_evidence,
    LoadCase,
    MaterialInterpolation,
    PeriodicHomogenizationCase,
)
from phydrax.applications.solid_mechanics._topology_reanalysis import (
    DensityTransferCandidate,
    FiniteElementReanalysisCandidate,
    reanalyse_topology_design,
    TopologyReanalysisPlan,
)
from phydrax.applications.solid_mechanics._topology_state import (
    certify_state_adjoint,
    FiniteElementStateSolver,
    MechanicsBranchGate,
    MechanicsStateCandidate,
    NeuralVariationalStateSolver,
)


def _density_transform(
    count: int,
    /,
    *,
    radius: float = 0.0,
    beta: float = 1.0,
    design_mask=None,
    fixed_density=None,
) -> DensityTransform:
    mask = (
        jnp.ones((count,), dtype=bool)
        if design_mask is None
        else jnp.asarray(design_mask, dtype=bool)
    )
    prepared = optim.DensityTransformPlan(
        optim.ConicDensityFilterPlan(
            jnp.arange(count, dtype=float).reshape((count, 1)),
            radius,
            mask,
            fixed_density,
            jnp.ones((count,)),
        ),
        optim.TanhDensityProjectionPlan(jnp.asarray(0.5)),
    ).prepare()
    return DensityTransform(prepared, beta=beta)


def _diagonal_fe_solver() -> FiniteElementStateSolver:
    def solve(problem, design, initial_state, args):
        del args
        zero = jax.tree.map(jnp.zeros_like, initial_state)
        one = jax.tree.map(jnp.ones_like, initial_state)
        offset = problem.residual(zero, design)
        slope = jax.tree.map(
            lambda at_one, at_zero: at_one - at_zero,
            problem.residual(one, design),
            offset,
        )
        state = jax.tree.map(
            lambda at_zero, diagonal: -at_zero / diagonal,
            offset,
            slope,
        )
        return MechanicsStateCandidate(
            state,
            diagnostics=optim.OptimizationDiagnostics(residual_evaluations=2),
        )

    return FiniteElementStateSolver(solve, solver_id="test-diagonal-fe")


def _mechanics_problem(
    solver,
    /,
    *,
    loads=(1.0,),
    aggregation=None,
    branch_evaluator=None,
    count: int = 2,
) -> TopologyMechanicsProblem:
    transform = _density_transform(count)
    interpolation = MaterialInterpolation(2.0, minimum=0.2, penalty=2.0)
    cases = tuple(
        LoadCase(
            jnp.full((count,), load),
            weight=1.0,
            case_id=f"load-{index}",
        )
        for index, load in enumerate(loads)
    )
    return TopologyMechanicsProblem(
        lambda state, material, case, args: material * state - case.load,
        cases,
        transform,
        interpolation,
        0.8,
        solver,
        aggregation=Aggregation() if aggregation is None else aggregation,
        acceptance_policy=optim.StateAcceptancePolicy(
            state_relative_tolerance=1.0e-9,
            state_absolute_tolerance=1.0e-10,
            adjoint_relative_tolerance=1.0e-9,
            adjoint_absolute_tolerance=1.0e-10,
        ),
        branch_evaluator=branch_evaluator,
        problem_id="test-topology-mechanics",
    )


class _WarmRollbackMethod(optim.AbstractStateDesignMethod):
    def __init__(self):
        self.method_id = "test-warm-rollback"

    def solve(self, problem, initial_state, initial_design, /, *, termination, args):
        del termination
        first_stage = float(initial_design[0]) < 0.5
        design = jnp.full_like(initial_design, 0.6 if first_stage else 0.9)
        state_result = problem.solve_state(design, initial_state, args=args)
        adjoint = state_result.state
        evidence = certify_state_adjoint(
            problem,
            state_result.state,
            design,
            adjoint,
            reference_state=initial_state,
            args=args,
        )
        objective, auxiliary = problem.value(state_result.state, design, args)
        return optim.StateDesignResult(
            state_result.state,
            design,
            objective,
            auxiliary,
            adjoint,
            (
                optim.OptimizationStatus.SUCCESS
                if first_stage
                else optim.OptimizationStatus.MAXIMUM_STEPS_REACHED
            ),
            state_result.diagnostics,
            optim.OptimizationProvenance(
                problem_id=problem.problem_id,
                method=self.method_id,
                backend="phydrax",
                globalization="none",
                matrix_free=True,
            ),
            state_acceptance=evidence.state,
            adjoint_acceptance=evidence.adjoint,
        )


def test_density_filter_projection_and_fixed_regions_remain_differentiable() -> None:
    transform = _density_transform(
        3,
        radius=1.5,
        beta=3.0,
        design_mask=(False, True, False),
        fixed_density=(1.0, 0.0, 0.0),
    )
    raw = jnp.asarray((0.8, 0.4, 0.7))
    physical = transform.apply(raw)
    gradient = jax.grad(lambda density: jnp.sum(transform.apply(density)))(raw)

    assert physical[0] == pytest.approx(1.0)
    assert physical[2] == pytest.approx(0.0)
    assert gradient[0] == pytest.approx(0.0)
    assert gradient[2] == pytest.approx(0.0)
    assert jnp.isfinite(gradient[1]) and gradient[1] > 0.0


def test_material_interpolation_has_exact_endpoints_and_finite_gradient() -> None:
    interpolation = MaterialInterpolation(10.0, minimum=0.1, penalty=3.0)
    values = interpolation(jnp.asarray((0.0, 1.0, 0.5)))

    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(10.0)
    assert jax.grad(lambda density: interpolation(density))(jnp.asarray(0.5)) > 0.0


def test_multi_load_maximum_ties_have_symmetric_sensitivities() -> None:
    aggregation = Aggregation("maximum")
    sensitivities = aggregation.sensitivities(jnp.asarray((3.0, 3.0)))
    problem = _mechanics_problem(
        _diagonal_fe_solver(),
        loads=(1.0, 1.0),
        aggregation=aggregation,
    )
    state_problem = problem.as_state_design_problem()
    root = state_problem.solve_state(
        jnp.full((2,), 0.5),
        (jnp.zeros((2,)), jnp.zeros((2,))),
    )
    objective, values = state_problem.value(root.state, jnp.full((2,), 0.5))

    assert aggregation(jnp.asarray((3.0, 3.0))) == pytest.approx(3.0)
    assert jnp.allclose(sensitivities, jnp.asarray((0.5, 0.5)))
    assert root.acceptance.accepted
    assert objective == pytest.approx(values[0])
    assert values[0] == pytest.approx(values[1])


def test_hill_mandel_evidence_accepts_power_equivalence_and_rejects_periodic_defect() -> (
    None
):
    homogenization = PeriodicHomogenizationCase(
        jnp.asarray((0.5,)),
        case_id="periodic-x",
    )
    load_case = homogenization.as_load_case(
        jnp.asarray((1.0,)),
        lambda state, material, case, args: jnp.sum(
            state * case.context.macroscopic_strain
        ),
    )
    accepted = hill_mandel_evidence(
        jnp.asarray(((2.0,), (2.0,))),
        jnp.asarray(((0.5,), (0.5,))),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((2.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.0, 0.0)),
    )
    rejected = hill_mandel_evidence(
        jnp.asarray(((2.0,), (2.0,))),
        jnp.asarray(((0.5,), (0.5,))),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((2.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((1.0e-3, 0.0)),
    )

    assert accepted.accepted
    assert accepted.power_defect == pytest.approx(0.0)
    assert not rejected.accepted
    assert load_case.context is homogenization


def test_fe_state_root_rejects_a_large_realized_residual() -> None:
    bad_solver = FiniteElementStateSolver(
        lambda problem, design, initial, args: MechanicsStateCandidate(initial),
        solver_id="bad-fe",
    )
    problem = _mechanics_problem(bad_solver)
    state_problem = problem.as_state_design_problem()
    result = state_problem.solve_state(
        jnp.full((2,), 0.5),
        (jnp.zeros((2,)),),
    )

    assert not result.acceptance.accepted
    assert result.acceptance.residual_norm > result.acceptance.threshold


def test_independent_adjoint_evidence_rejects_wrong_transpose_root() -> None:
    problem = _mechanics_problem(_diagonal_fe_solver())
    state_problem = problem.as_state_design_problem()
    design = jnp.full((2,), 0.5)
    initial = (jnp.zeros((2,)),)
    state = state_problem.solve_state(design, initial).state
    evidence = certify_state_adjoint(
        state_problem,
        state,
        design,
        (jnp.zeros((2,)),),
        reference_state=initial,
    )

    assert evidence.state.accepted
    assert not evidence.adjoint.accepted
    assert evidence.adjoint.transpose_defect_norm > evidence.adjoint.threshold


def test_fe_and_neural_variational_roots_give_the_same_reduced_gradient() -> None:
    fe_solver = _diagonal_fe_solver()
    fe_problem = _mechanics_problem(fe_solver)
    neural_solver = NeuralVariationalStateSolver(
        lambda problem, design, initial, args: jax.tree.map(
            lambda value: value + 0.1, initial
        ),
        fe_solver,
        solver_id="test-neural-proposal",
    )
    neural_problem = _mechanics_problem(neural_solver)
    initial = (jnp.zeros((2,)),)

    def reduced(problem, density):
        state_problem = problem.as_state_design_problem()
        state = state_problem.solve_state(density, initial).state
        return state_problem.value(state, density)[0]

    density = jnp.full((2,), 0.5)
    fe_gradient = jax.grad(lambda value: reduced(fe_problem, value))(density)
    neural_gradient = jax.grad(lambda value: reduced(neural_problem, value))(density)

    assert jnp.all(jnp.isfinite(fe_gradient))
    assert jnp.allclose(neural_gradient, fe_gradient, rtol=1.0e-6, atol=1.0e-8)


def test_neural_proposal_rolls_back_exactly_before_mandatory_fe_root() -> None:
    fe_solver = _diagonal_fe_solver()
    neural_solver = NeuralVariationalStateSolver(
        lambda problem, design, initial, args: jax.tree.map(
            lambda value: jnp.full_like(value, jnp.nan), initial
        ),
        fe_solver,
        solver_id="nonfinite-neural-proposal",
    )
    problem = _mechanics_problem(neural_solver)
    state_problem = problem.as_state_design_problem()
    initial = (jnp.full((2,), 0.25),)
    root = neural_solver.solve_root(
        state_problem,
        jnp.full((2,), 0.5),
        initial,
        args=None,
    )

    assert root.proposal.rollback_applied
    assert jnp.allclose(root.proposal.selected_initial_state[0], initial[0])
    assert root.final_fe_reanalysis
    assert root.state_equation.acceptance.accepted


def test_out_of_support_neural_operator_is_not_evaluated() -> None:
    def forbidden_proposal(problem, design, initial, args):
        raise AssertionError("An out-of-support learned operator was evaluated.")

    fe_solver = _diagonal_fe_solver()
    neural_solver = NeuralVariationalStateSolver(
        forbidden_proposal,
        fe_solver,
        support=lambda design, args: False,
        solver_id="unsupported-neural-proposal",
    )
    problem = _mechanics_problem(neural_solver)
    state_problem = problem.as_state_design_problem()
    initial = (jnp.full((2,), 0.25),)
    root = neural_solver.solve_root(
        state_problem,
        jnp.full((2,), 0.5),
        initial,
        args=None,
    )

    assert not root.proposal.supported
    assert root.proposal.rollback_applied
    assert root.state_equation.acceptance.accepted


def test_continuation_failure_rolls_back_to_last_accepted_design() -> None:
    problem = _mechanics_problem(_diagonal_fe_solver(), count=1)
    schedule = TopologyContinuationSchedule(
        (
            TopologyContinuationStage(1.0, penalty=1.0, stage_id="soft"),
            TopologyContinuationStage(2.0, penalty=2.0, stage_id="sharp"),
        )
    )
    result = solve_topology_optimization(
        problem,
        (jnp.zeros((1,)),),
        jnp.asarray((0.4,)),
        schedule=schedule,
        method=_WarmRollbackMethod(),
        termination=optim.OptimizationTermination(maximum_steps=1),
    )

    assert jnp.allclose(result.raw_density, 0.6)
    assert result.continuation_evidence[1].rollback_applied
    assert not result.continuation_completed


def test_nonlinear_branch_gate_is_part_of_state_acceptance() -> None:
    gate = MechanicsBranchGate(("primary",))
    problem = _mechanics_problem(
        _diagonal_fe_solver(),
        branch_evaluator=lambda state, density, case, args: gate.evaluate("secondary"),
    )
    state_problem = problem.as_state_design_problem()
    result = state_problem.solve_state(
        jnp.full((2,), 0.5),
        (jnp.zeros((2,)),),
    )

    assert result.residual_norm <= result.acceptance.threshold
    assert not result.acceptance.admissible
    assert not result.acceptance.accepted


@pytest.mark.parametrize("event", ("contact", "fracture"))
def test_contact_and_fracture_topology_events_are_rejected(event: str) -> None:
    gate = MechanicsBranchGate(("primary",))
    evidence = gate.evaluate(
        "primary",
        contact_event=event == "contact",
        fracture_event=event == "fracture",
    )

    assert not evidence.accepted
    assert evidence.contact_event or evidence.fracture_event


def test_reference_reanalysis_requires_transfer_primal_and_adjoint_evidence() -> None:
    source_problem = _mechanics_problem(_diagonal_fe_solver(), count=1)
    source = solve_topology_optimization(
        source_problem,
        (jnp.zeros((1,)),),
        jnp.asarray((0.4,)),
        method=_WarmRollbackMethod(),
        termination=optim.OptimizationTermination(maximum_steps=1),
    )
    reference_problem = _mechanics_problem(_diagonal_fe_solver(), count=1)

    def fe_reanalysis(state_problem, design, initial_state, args):
        root = state_problem.solve_state(design, initial_state, args=args)
        return FiniteElementReanalysisCandidate(
            root.state,
            root.state,
            solver_id="reference-fe-adjoint",
        )

    plan = TopologyReanalysisPlan(
        reference_problem,
        lambda physical: DensityTransferCandidate(source.raw_density),
        fe_reanalysis,
        proposal=lambda problem, density, initial, args: jax.tree.map(
            jnp.zeros_like, initial
        ),
        uniform_source_objective=2.0,
        uniform_reference_objective=2.0,
    )
    report = reanalyse_topology_design(
        source,
        plan,
        (jnp.zeros((1,)),),
    )

    assert report.evidence.transfer.accepted
    assert report.evidence.mechanics.state.accepted
    assert report.evidence.mechanics.adjoint.accepted
    assert report.evidence.learned_proposal_used
    assert report.evidence.final_fe_reanalysis
    assert report.accepted
    assert report.discretization_ratio == pytest.approx(1.0)
