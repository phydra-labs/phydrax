#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import (
    AbstractStateDesignMethod,
    AbstractStateSolver,
    AbstractStructuredNonlinearMethod,
    Bounds,
    compile_structured_state_design,
    OptimizationDiagnostics,
    OptimizationStatus,
    OptimizationTermination,
    ReducedAdjoint,
    solve_state_design,
    solve_structured_state_design,
    StateDesignConstraint,
    StateDesignProblem,
    StateDesignResult,
    StateEquationResult,
    StructuredStateDesignCompilation,
    StructuredStateDesignResult,
)
from ._force_density import (
    _physical_state,
    _validated_force_densities,
    ForceDensityInputs,
    ForceDensityPlan,
    ForceDensityProblem,
    ForceDensityResult,
    ForceDensityState,
    ForceDensityStatus,
    prepare_force_density,
    PreparedForceDensitySolve,
    solve_force_density,
)


class ForceDensityDesignConstraint(StrictModule, NonTrainableState):
    """Bound-form constraint evaluated on a reconstructed physical state."""

    function: Callable = eqx.field(static=True)
    lower: Any
    upper: Any
    constraint_id: str = eqx.field(static=True)
    depends_on_state: bool = eqx.field(static=True)

    def __init__(
        self,
        function: Callable,
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        constraint_id: str,
        depends_on_state: bool = True,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(constraint_id)
        if not identifier:
            raise ValueError("constraint_id must be nonempty.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.constraint_id = identifier
        self.depends_on_state = bool(depends_on_state)

    def value(
        self,
        state: ForceDensityState,
        design: PyTree[Any],
        args: Any,
        /,
    ):
        return self.function(state, design, args)


class ForceDensityStateSolver(AbstractStateSolver):
    """Exact linear or converged nonlinear force-density state solver."""

    plan: ForceDensityPlan
    decode_inputs: Callable = eqx.field(static=True)

    def __init__(self, plan: ForceDensityPlan, decode_inputs: Callable, /):
        if not isinstance(plan, ForceDensityPlan):
            raise TypeError("plan must be a ForceDensityPlan.")
        if not callable(decode_inputs):
            raise TypeError("decode_inputs must be callable.")
        self.plan = plan
        self.decode_inputs = decode_inputs

    @property
    def method_id(self) -> str:
        route = (
            "nonlinear" if self.plan.problem.load_model.depends_on_positions else "linear"
        )
        return f"force-density-state/{route}"

    def solve(
        self,
        problem: StateDesignProblem,
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> StateEquationResult:
        inputs = self.decode_inputs(design, args)
        if not isinstance(inputs, ForceDensityInputs):
            raise TypeError("decode_inputs must return ForceDensityInputs.")
        structure = self.plan.problem.structure
        initial = jnp.asarray(initial_state)
        if initial.shape != (structure.free_dof_count,):
            raise ValueError("initial_state has the wrong reduced force-density shape.")
        initial_positions = (
            structure.expand(initial, inputs.prescribed_values)
            if self.plan.problem.load_model.depends_on_positions
            else None
        )
        prepared = prepare_force_density(
            self.plan,
            inputs,
            initial_positions=initial_positions,
        )
        result = solve_force_density(prepared)
        reduced = structure.reduce(result.state.positions)
        residual = problem.residual(reduced, design, args)
        linear_solves = jnp.asarray(0, dtype=jnp.int32)
        linear_iterations = jnp.asarray(0, dtype=jnp.int32)
        residual_evaluations = jnp.asarray(1, dtype=jnp.int32)
        jvp_evaluations = jnp.asarray(0, dtype=jnp.int32)
        vjp_evaluations = jnp.asarray(0, dtype=jnp.int32)
        setup_refreshes = jnp.asarray(0, dtype=jnp.int32)
        numeric_refreshes = jnp.asarray(0, dtype=jnp.int32)
        if result.linear_result is not None:
            linear_solves = jnp.asarray(1, dtype=jnp.int32)
            linear_iterations = result.linear_result.diagnostics.iterations
        if result.nonlinear_result is not None:
            evidence = result.nonlinear_result.diagnostics
            linear_solves = evidence.linear_solves
            linear_iterations = evidence.linear_iterations
            residual_evaluations = evidence.residual_evaluations
            jvp_evaluations = evidence.jvp_evaluations
            vjp_evaluations = evidence.vjp_evaluations
            setup_refreshes = evidence.setup_refreshes
            numeric_refreshes = evidence.numeric_refreshes
        status = jnp.where(
            result.status == int(ForceDensityStatus.SUCCESS),
            int(OptimizationStatus.SUCCESS),
            jnp.where(
                result.status == int(ForceDensityStatus.LINEAR_SOLVE_FAILED),
                int(OptimizationStatus.LINEAR_SOLVE_FAILED),
                jnp.where(
                    result.status == int(ForceDensityStatus.NONLINEAR_SOLVE_FAILED),
                    int(OptimizationStatus.BACKEND_FAILED),
                    jnp.where(
                        result.status == int(ForceDensityStatus.NONFINITE_STATE),
                        int(OptimizationStatus.NONFINITE_EVALUATION),
                        jnp.where(
                            (result.status == int(ForceDensityStatus.DEGENERATE_MEMBER))
                            | (
                                result.status
                                == int(ForceDensityStatus.INVALID_LOAD_GEOMETRY)
                            ),
                            int(OptimizationStatus.INFEASIBLE),
                            int(OptimizationStatus.CERTIFICATION_FAILED),
                        ),
                    ),
                ),
            ),
        )
        diagnostics = OptimizationDiagnostics(
            residual_evaluations=residual_evaluations,
            jvp_evaluations=jvp_evaluations,
            vjp_evaluations=vjp_evaluations,
            linear_solves=linear_solves,
            linear_iterations=linear_iterations,
            setup_refreshes=setup_refreshes,
            numeric_refreshes=numeric_refreshes,
            final_optimality_norm=jnp.asarray(
                jnp.nan, dtype=result.diagnostics.free_residual_norm.dtype
            ),
        )
        return StateEquationResult(reduced, residual, status, diagnostics)


class ForceDensityDesignProblem(StrictModule, NonTrainableState):
    """Physical objective and constraints over a prepared force-density plan."""

    plan: ForceDensityPlan
    decode_inputs: Callable = eqx.field(static=True)
    objective: Callable = eqx.field(static=True)
    design_bounds: Bounds | None
    constraints: tuple[ForceDensityDesignConstraint, ...]
    has_aux: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ForceDensityPlan,
        decode_inputs: Callable,
        objective: Callable,
        /,
        *,
        design_bounds: Bounds | None = None,
        constraints: Sequence[ForceDensityDesignConstraint] = (),
        has_aux: bool = False,
        problem_id: str = "force-density-design",
    ):
        if not isinstance(plan, ForceDensityPlan):
            raise TypeError("plan must be a ForceDensityPlan.")
        if not callable(decode_inputs) or not callable(objective):
            raise TypeError("decode_inputs and objective must be callable.")
        if design_bounds is not None and not isinstance(design_bounds, Bounds):
            raise TypeError("design_bounds must be Bounds or None.")
        constraints_ = tuple(constraints)
        if any(
            not isinstance(constraint, ForceDensityDesignConstraint)
            for constraint in constraints_
        ):
            raise TypeError(
                "constraints must contain ForceDensityDesignConstraint values."
            )
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.plan = plan
        self.decode_inputs = decode_inputs
        self.objective = objective
        self.design_bounds = design_bounds
        self.constraints = constraints_
        self.has_aux = bool(has_aux)
        self.problem_id = identifier

    @property
    def equilibrium_problem(self) -> ForceDensityProblem:
        return self.plan.problem

    def inputs(self, design: PyTree[Any], args: Any = None, /) -> ForceDensityInputs:
        inputs = self.decode_inputs(design, args)
        if not isinstance(inputs, ForceDensityInputs):
            raise TypeError("decode_inputs must return ForceDensityInputs.")
        return inputs

    def physical_state(
        self,
        reduced_state: PyTree[Any],
        design: PyTree[Any],
        args: Any = None,
        /,
    ) -> ForceDensityState:
        inputs = self.inputs(design, args)
        force_densities = _validated_force_densities(self.equilibrium_problem, inputs)
        positions = self.equilibrium_problem.structure.expand(
            jnp.asarray(reduced_state), inputs.prescribed_values
        )
        state, _ = _physical_state(
            self.equilibrium_problem,
            inputs,
            force_densities,
            positions,
        )
        return state

    def as_state_design_problem(self, /) -> StateDesignProblem:
        def residual(reduced_state, design, args):
            state = self.physical_state(reduced_state, design, args)
            return self.equilibrium_problem.structure.reduce(
                state.internal_nodal_forces - state.applied_nodal_loads
            )

        def objective(reduced_state, design, args):
            state = self.physical_state(reduced_state, design, args)
            output = self.objective(state, design, args)
            if self.has_aux:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError(
                        "A has_aux force-density objective must return "
                        "(value, auxiliary)."
                    )
                value, auxiliary = output
                return jnp.asarray(value), auxiliary
            return jnp.asarray(output)

        def lower_constraint(
            constraint: ForceDensityDesignConstraint,
        ) -> StateDesignConstraint:
            def function(reduced_state, design, args):
                state = self.physical_state(reduced_state, design, args)
                return constraint.value(state, design, args)

            return StateDesignConstraint(
                function,
                lower=constraint.lower,
                upper=constraint.upper,
                constraint_id=constraint.constraint_id,
                depends_on_state=constraint.depends_on_state,
            )

        return StateDesignProblem(
            residual,
            objective,
            state_solver=ForceDensityStateSolver(self.plan, self.decode_inputs),
            design_bounds=self.design_bounds,
            constraints=tuple(
                lower_constraint(constraint) for constraint in self.constraints
            ),
            has_aux=self.has_aux,
            problem_id=self.problem_id,
        )


class ForceDensityDesignResult(StrictModule):
    """Accepted design, physical equilibrium, and state-design evidence."""

    state_design: StateDesignResult
    equilibrium: ForceDensityResult
    inputs: ForceDensityInputs
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.state_design.successful & self.equilibrium.successful


class StructuredForceDensityDesignCompilation(StrictModule):
    """Compiled all-at-once force-density state/design problem."""

    problem: ForceDensityDesignProblem
    state_design: StructuredStateDesignCompilation
    args: Any


class StructuredForceDensityDesignResult(StrictModule):
    """Structured KKT result paired with final physical equilibrium evidence."""

    state_design: StructuredStateDesignResult
    equilibrium: ForceDensityResult
    inputs: ForceDensityInputs
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.state_design.successful & self.equilibrium.successful


def compile_structured_force_density_design(
    problem: ForceDensityDesignProblem,
    initial_design: PyTree[Any],
    /,
    *,
    initial_positions: Any | None = None,
    sample_args: Any = None,
    exact_hessian: bool = True,
    compiler: Any = "auto",
    chunk_size: int | None = None,
) -> StructuredForceDensityDesignCompilation:
    """Compile physical equilibrium, objective, and constraints all at once."""
    if not isinstance(problem, ForceDensityDesignProblem):
        raise TypeError("problem must be a ForceDensityDesignProblem.")
    inputs = problem.inputs(initial_design, sample_args)
    equilibrium = solve_force_density(
        prepare_force_density(
            problem.plan,
            inputs,
            initial_positions=initial_positions,
        )
    )
    if not bool(equilibrium.successful):
        raise ValueError(
            "Initial force-density equilibrium must be successful before structured compilation."
        )
    initial_state = problem.equilibrium_problem.structure.reduce(
        equilibrium.state.positions
    )
    compiled = compile_structured_state_design(
        problem.as_state_design_problem(),
        initial_state,
        initial_design,
        sample_args=sample_args,
        exact_hessian=exact_hessian,
        compiler=compiler,
        chunk_size=chunk_size,
    )
    return StructuredForceDensityDesignCompilation(problem, compiled, sample_args)


def solve_structured_force_density_design(
    compilation: StructuredForceDensityDesignCompilation,
    /,
    *,
    method: AbstractStructuredNonlinearMethod,
    termination: OptimizationTermination | None = None,
    initial_state: PyTree[Any] | None = None,
    initial_design: PyTree[Any] | None = None,
    warm_start: Any = None,
) -> StructuredForceDensityDesignResult:
    """Solve one compiled force-density design and recertify physical equilibrium."""
    if not isinstance(compilation, StructuredForceDensityDesignCompilation):
        raise TypeError("compilation must be a StructuredForceDensityDesignCompilation.")
    solved = solve_structured_state_design(
        compilation.state_design,
        method=method,
        termination=termination,
        initial_state=initial_state,
        initial_design=initial_design,
        warm_start=warm_start,
    )
    problem = compilation.problem
    inputs = problem.inputs(solved.design, compilation.args)
    positions = problem.equilibrium_problem.structure.expand(
        solved.state, inputs.prescribed_values
    )
    equilibrium = solve_force_density(
        prepare_force_density(
            problem.plan,
            inputs,
            initial_positions=(
                positions
                if problem.equilibrium_problem.load_model.depends_on_positions
                else None
            ),
        )
    )
    return StructuredForceDensityDesignResult(
        solved,
        equilibrium,
        inputs,
        problem.problem_id,
    )


def solve_force_density_design(
    problem: ForceDensityDesignProblem,
    initial_design: PyTree[Any],
    /,
    *,
    initial_positions: Any | None = None,
    method: AbstractStateDesignMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> ForceDensityDesignResult:
    """Solve and physically recertify one force-density inverse design."""
    if not isinstance(problem, ForceDensityDesignProblem):
        raise TypeError("problem must be a ForceDensityDesignProblem.")
    if problem.constraints and method is None:
        raise ValueError(
            "Constrained force-density designs require an explicit ReducedMMA "
            "method or the structured force-density design lifecycle."
        )
    method_ = ReducedAdjoint() if method is None else method
    termination_ = (
        OptimizationTermination(maximum_steps=200) if termination is None else termination
    )
    if not isinstance(method_, AbstractStateDesignMethod):
        raise TypeError("method must be AbstractStateDesignMethod or None.")
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    inputs = problem.inputs(initial_design, args)
    prepared = prepare_force_density(
        problem.plan,
        inputs,
        initial_positions=initial_positions,
    )
    initial_equilibrium = solve_force_density(prepared)
    initial_state = problem.equilibrium_problem.structure.reduce(
        initial_equilibrium.state.positions
    )
    state_design = solve_state_design(
        problem.as_state_design_problem(),
        initial_state,
        initial_design,
        method=method_,
        termination=termination_,
        args=args,
    )
    final_inputs = problem.inputs(state_design.design, args)
    final_positions = problem.equilibrium_problem.structure.expand(
        state_design.state, final_inputs.prescribed_values
    )
    final_prepared: PreparedForceDensitySolve = prepare_force_density(
        problem.plan,
        final_inputs,
        initial_positions=(
            final_positions
            if problem.equilibrium_problem.load_model.depends_on_positions
            else None
        ),
    )
    equilibrium = solve_force_density(final_prepared)
    return ForceDensityDesignResult(
        state_design,
        equilibrium,
        final_inputs,
        problem.problem_id,
    )


__all__ = [
    "ForceDensityDesignConstraint",
    "ForceDensityDesignProblem",
    "ForceDensityDesignResult",
    "ForceDensityStateSolver",
    "StructuredForceDensityDesignCompilation",
    "StructuredForceDensityDesignResult",
    "compile_structured_force_density_design",
    "solve_force_density_design",
    "solve_structured_force_density_design",
]
