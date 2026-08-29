#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp

import phydrax as phx

from .cases import DirectCollocationQualificationSetup
from .contracts import DirectCollocationQualificationRecord


QualificationBackend = Literal["native", "ipopt"]


def _method(backend: QualificationBackend):
    if backend == "native":
        return phx.optim.PrimalDualInteriorPoint(
            mode="dense-filter", max_dense_dimension=512
        )
    if backend == "ipopt":
        return phx.optim.IpoptMinimize(options={"print_level": 0})
    raise ValueError("backend must be 'native' or 'ipopt'.")


def run_qualification_case(
    setup: DirectCollocationQualificationSetup,
    backend: QualificationBackend,
    /,
) -> DirectCollocationQualificationRecord:
    compilation = phx.control.compile_direct_collocation(
        setup.problem,
        setup.plan,
        setup.initial_states,
        setup.initial_controls,
        parameter_guess=setup.parameter_guess,
        duration_guess=setup.duration_guess,
        bounds=setup.bounds,
    )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-7,
        relative_optimality=0.0,
        maximum_steps=200,
    )
    prepared = phx.control.prepare_direct_collocation(
        compilation,
        method=_method(backend),
        termination=termination,
    )
    started = time.perf_counter()
    result = phx.control.solve_prepared_direct_collocation(prepared)
    elapsed = time.perf_counter() - started
    program = compilation.structured_program
    direction = jnp.linspace(0.1, 1.0, program.num_variables)
    sparse_action = program.jacobian_plan.operator(
        compilation.initial_coordinates,
        setup.problem.args,
    ).mv(direction)
    direct_action = jax.jvp(
        lambda coordinates: program.constraints(coordinates, setup.problem.args),
        (compilation.initial_coordinates,),
        (direction,),
    )[1]
    derivative_error = float(jnp.max(jnp.abs(sparse_action - direct_action), initial=0.0))
    reference_error = (
        0.0
        if setup.reference_objective is None
        else abs(float(result.objective) - setup.reference_objective)
    )
    replay_error = 0.0
    if setup.case.replay_required and bool(result.successful):
        replay = phx.control.replay_direct_collocation(
            result,
            phx.control.DirectCollocationReplayPolicy(
                dae_policy=phx.solver.DAESolvePolicy(
                    method=phx.solver.ThetaMethod(1.0, endpoint=True)
                ),
                node_state_tolerance=1.0e-5,
                terminal_state_tolerance=1.0e-5,
                algebraic_constraint_tolerance=1.0e-6,
            ),
        )
        replay_error = float(replay.maximum_node_discrepancy)
    certified = (
        bool(result.successful)
        and float(result.diagnostics.maximum_defect) <= 1.0e-6
        and float(result.diagnostics.maximum_constraint_violation) <= 1.0e-6
        and derivative_error <= 1.0e-8
    )
    false_success = bool(result.successful) and not certified
    false_failure = setup.case.expected_feasible and not bool(result.successful)
    return DirectCollocationQualificationRecord.create(
        case_id=setup.case.case_id,
        backend=backend,
        method_id=result.method_id,
        successful=bool(result.successful),
        backend_status=int(result.optimization_result.status),
        public_status=int(result.status),
        false_success=false_success,
        false_failure=false_failure,
        objective=float(result.objective),
        reference_error=reference_error,
        maximum_defect=float(result.diagnostics.maximum_defect),
        maximum_constraint_violation=float(
            result.diagnostics.maximum_constraint_violation
        ),
        maximum_off_grid_defect=float(result.diagnostics.maximum_off_grid_defect),
        replay_error=replay_error,
        derivative_action_error=derivative_error,
        variables=program.num_variables,
        constraints=program.num_constraints,
        jacobian_nonzeros=program.jacobian_plan.nnz,
        dense_materialized=backend == "native",
        elapsed_seconds=elapsed,
    )


__all__ = ["QualificationBackend", "run_qualification_case"]
