#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Named block coordinates and block preconditioning for a reduced DAE.

Three thermal nodes lose heat through an algebraic radiative-type flux law
``q = k T^2``. The declaration names each variable and equation block, gives the
temperature and flux unequal scales, and passes structural admission. The DAE
coordinate adapter maps a physical named-block linearization onto the native
implicit stages, and a single subspace-correction term with a 2 by 2 block
factorization over the differential and algebraic blocks preconditions every
Newton step. The result is checked against the analytic decay
``T(t) = T0 / (1 + k T0 t / C)`` and against the same native DAE solved without a
named setup.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


jax.config.update("jax_enable_x64", True)

_CAPACITY = np.asarray([2.0, 3.0, 5.0])
_CONDUCTANCE = 1.0e-3
_INITIAL_TEMPERATURE = np.asarray([300.0, 250.0, 400.0])


def _balance(time: Array, jet: Any, args: Any) -> Array:
    del time, args
    return jnp.asarray(_CAPACITY) * jet.value("temperature", 1) + jet.value("flux")


def _law(time: Array, jet: Any, args: Any) -> Array:
    del time, args
    return jet.value("flux") - _CONDUCTANCE * jet.value("temperature") ** 2


def _compilation() -> phx.dynamics.ReducedDAECompilation:
    dynamics = phx.dynamics
    component = dynamics.DAEComponent(
        "nodes",
        (
            dynamics.DAEVariableBlock(
                "temperature",
                (3,),
                1,
                state_scale=jnp.asarray(_INITIAL_TEMPERATURE),
                rate_scale=10.0,
            ),
            dynamics.DAEVariableBlock("flux", (3,), 0, state_scale=100.0),
        ),
        (
            dynamics.DAEEquationBlock(
                "balance",
                _balance,
                (
                    dynamics.DAEDerivativeIncidence("temperature", 1),
                    dynamics.DAEDerivativeIncidence("flux", 0),
                ),
                residual_scale=50.0,
                residual_semantic_id="node-energy-balance",
                residual_numeric_id="node-energy-balance-capacity",
            ),
            dynamics.DAEEquationBlock(
                "law",
                _law,
                (
                    dynamics.DAEDerivativeIncidence("flux", 0),
                    dynamics.DAEDerivativeIncidence("temperature", 0),
                ),
                residual_scale=100.0,
                residual_semantic_id="node-flux-law",
                residual_numeric_id="node-flux-law-conductance",
            ),
        ),
    )
    return dynamics.compile_acausal_dae(
        dynamics.AcausalDAESource((component,)), dynamics.DAEStructuralPolicy(0, 0)
    )


def _linearization(
    adapter: phx.solver.DAECoordinateAdapter,
) -> phx.solver.AutonomousDAEBlockLinearization:
    la = phx.linalg
    variables = adapter.variable_space
    equations = adapter.equation_space
    flux = variables.spaces[variables.names.index("nodes.flux")]
    temperature = variables.spaces[variables.names.index("nodes.temperature")]
    balance = equations.spaces[equations.names.index("nodes.balance")]
    law = equations.spaces[equations.names.index("nodes.law")]

    def linearization(
        time: Array, state: Any, rate: Any, args: Any
    ) -> phx.solver.DAEBlockJacobian:
        del time, rate, args
        values = dict(zip(variables.names, state, strict=True))
        identity = jnp.eye(3)
        state_jacobian = la.assemble_block_operator(
            (
                (
                    ("nodes.balance",),
                    ("nodes.flux",),
                    la.DenseLinearOperator(identity, source=flux, target=balance),
                ),
                (
                    ("nodes.law",),
                    ("nodes.flux",),
                    la.DenseLinearOperator(identity, source=flux, target=law),
                ),
                (
                    ("nodes.law",),
                    ("nodes.temperature",),
                    la.DenseLinearOperator(
                        jnp.diag(-2.0 * _CONDUCTANCE * values["nodes.temperature"]),
                        source=temperature,
                        target=law,
                    ),
                ),
            ),
            source=variables,
            target=equations,
        )
        rate_jacobian = la.assemble_block_operator(
            (
                (
                    ("nodes.balance",),
                    ("nodes.temperature",),
                    la.DenseLinearOperator(
                        jnp.diag(jnp.asarray(_CAPACITY)),
                        source=temperature,
                        target=balance,
                    ),
                ),
            ),
            source=variables,
            target=equations,
        )
        return phx.solver.DAEBlockJacobian(state_jacobian, rate_jacobian)

    return linearization


def _method(
    term: phx.linalg.SubspaceCorrectionTerm | None,
) -> phx.nonlinear.NewtonKrylov:
    la = phx.linalg
    return phx.nonlinear.NewtonKrylov(
        linear_policy=la.LinearSolvePolicy(
            la.FGMRES(restart=6),
            tolerance=la.TolerancePolicy(relative=1e-13, absolute=1e-15, max_steps=6),
            preconditioning=(
                None
                if term is None
                else la.PreconditioningPolicy(
                    la.AdditiveSubspaceCorrectionBuilder((term,))
                )
            ),
            materialization=la.MaterializationPolicy(max_entries=256, max_bytes=8192),
        )
    )


def _policy(
    stage: phx.linalg.SubspaceCorrectionTerm | None,
    initialization: phx.linalg.SubspaceCorrectionTerm | None,
) -> phx.solver.DAESolvePolicy:
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-12, relative_residual=0.0, maximum_steps=20
    )
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_method=_method(stage),
        initialization_method=_method(initialization),
        nonlinear_termination=termination,
        initialization_termination=termination,
    )


def run() -> dict[str, float | int | bool | str]:
    la = phx.linalg
    compilation = _compilation()
    adapter = phx.solver.DAECoordinateAdapter(compilation)
    stage = la.SubspaceCorrectionTerm(
        *adapter.correction_transfers("stage", adapter.role_groups("stage")),
        la.BlockFactorizationPreconditionerBuilder(
            la.DenseInversePreconditionerBuilder(),
            la.DenseInversePreconditionerBuilder(),
            "ldu",
        ),
    )
    initialization = la.SubspaceCorrectionTerm(
        *adapter.correction_transfers(
            "initialization", adapter.role_groups("initialization")
        ),
        la.DenseInversePreconditionerBuilder(),
    )
    initial_state = jnp.concatenate((jnp.zeros(3), jnp.asarray(_INITIAL_TEMPERATURE)))
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 2.0, 41), time_id="named-nodes")
    named = phx.solver.solve_dae(
        phx.solver.DifferentialAlgebraicProblem(
            adapter.bind_linearization(_linearization(adapter)),
            initial_state,
            initialization="structural",
        ),
        grid,
        policy=_policy(stage, initialization),
    )
    native = phx.solver.solve_dae(
        phx.solver.DifferentialAlgebraicProblem(
            compilation, initial_state, initialization="structural"
        ),
        grid,
        policy=_policy(None, None),
    )
    final_flux, final_temperature = adapter.state_view(named.states[-1])
    analytic = _INITIAL_TEMPERATURE / (
        1.0 + _CONDUCTANCE * _INITIAL_TEMPERATURE * 2.0 / _CAPACITY
    )
    history = named.attempt_history
    attempts = int(history.count)
    linear_solves = int(jnp.sum(history.linear_solves[:attempts]))
    linear_iterations = int(jnp.sum(history.linear_iterations[:attempts]))
    return {
        "variable_roles": ", ".join(
            f"{'/'.join(value.path)}={value.role}" for value in adapter.variables
        ),
        "equation_roles": ", ".join(
            f"{'/'.join(value.path)}={value.role}" for value in adapter.rows
        ),
        "successful": bool(named.successful),
        "native_successful": bool(native.successful),
        "max_named_minus_native": float(jnp.max(jnp.abs(named.states - native.states))),
        "final_temperature": np.array2string(np.asarray(final_temperature), precision=6),
        "final_flux_law_defect": float(
            jnp.max(jnp.abs(final_flux - _CONDUCTANCE * final_temperature**2))
        ),
        "max_relative_error_vs_analytic": float(
            np.max(np.abs(np.asarray(final_temperature) - analytic) / analytic)
        ),
        "stage_linear_solves": linear_solves,
        "stage_krylov_iterations": linear_iterations,
    }


if __name__ == "__main__":
    for name, value in run().items():
        print(f"{name}: {value}")
