#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _schrodinger_constraint(time, hamiltonian):
    condition = phx.conditions.Residual(
        "psi",
        time.component(),
        lambda state: phx.operators.schrodinger_residual(state, hamiltonian),
    )
    return phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(32),
        ),
    )


def test_complex_schrodinger_residual_runs_through_functional_solver():
    time = phx.domain.TimeInterval(0.0, 1.0)
    omega = 1.7
    hamiltonian = time.Function()(0.5 * omega * SIGMA_Z)

    @time.Function("t")
    def exact_state(t):
        return jnp.asarray([jnp.exp(-0.5j * omega * t), 0.0j])

    @time.Function("t")
    def perturbed_state(t):
        return jnp.asarray([jnp.exp(-0.3j * omega * t), 0.0j])

    constraint = _schrodinger_constraint(time, hamiltonian)
    exact_solver = phx.solver.FunctionalSolver(
        functions={"psi": exact_state},
        terms=[constraint],
    )
    perturbed_solver = phx.solver.FunctionalSolver(
        functions={"psi": perturbed_state},
        terms=[constraint],
    )

    loss_fn = eqx.filter_jit(lambda solver, key: solver.loss(key=key))
    exact_loss = loss_fn(exact_solver, jr.key(0))
    perturbed_loss = loss_fn(perturbed_solver, jr.key(1))
    assert jnp.isrealobj(exact_loss)
    assert jnp.isrealobj(perturbed_loss)
    assert exact_loss < 1e-20
    assert perturbed_loss > 1e-4


def test_mixed_register_program_runs_through_refresh_channel_and_gradient():
    quantum = phx.operators.quantum
    layout = quantum.HilbertRegisterLayout(("qubit", "qutrit"), (2, 3))
    identity_channel = jnp.eye(3, dtype=jnp.complex128)[None]
    initial_ket = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)
    initial_density = jnp.outer(initial_ket, jnp.conj(initial_ket))
    observable = jnp.kron(
        jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128),
        jnp.eye(3, dtype=jnp.complex128),
    )

    def program(theta):
        half = 0.5 * theta
        rotation = jnp.asarray(
            [
                [jnp.cos(half), -jnp.sin(half)],
                [jnp.sin(half), jnp.cos(half)],
            ],
            dtype=jnp.complex128,
        )
        return quantum.QuantumProgram(
            layout,
            (
                quantum.LocalUnitaryOperation(rotation, ("qubit",)),
                quantum.LocalKrausChannelOperation(identity_channel, ("qutrit",)),
            ),
            state_kind="density-matrix",
        )

    template = phx.solver.prepare_dense_quantum_program(program(0.0))

    def objective(theta):
        prepared = phx.solver.refresh_dense_quantum_program(template, program(theta))
        result = phx.solver.execute_dense_quantum_program(prepared, initial_density)
        expectation = jnp.real(jnp.trace(result.final_state @ observable))
        return expectation, result.diagnostics.successful

    theta = jnp.asarray(0.6)
    expectation, successful = eqx.filter_jit(objective)(theta)
    gradient = jax.grad(lambda value: objective(value)[0])(theta)

    assert successful
    assert jnp.allclose(expectation, jnp.sin(0.5 * theta) ** 2)
    assert jnp.allclose(gradient, 0.5 * jnp.sin(theta))
