#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _schrodinger_constraint(time, hamiltonian):
    return phx.constraints.FunctionalConstraint.from_operator(
        component=time.component(),
        operator=lambda state: phx.operators.schrodinger_residual(state, hamiltonian),
        constraint_vars="psi",
        num_points=32,
        structure=phx.domain.ProductStructure((("t",),)),
        reduction="mean",
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
        constraints=[constraint],
    )
    perturbed_solver = phx.solver.FunctionalSolver(
        functions={"psi": perturbed_state},
        constraints=[constraint],
    )

    loss_fn = eqx.filter_jit(lambda solver, key: solver.loss(key=key))
    exact_loss = loss_fn(exact_solver, jr.key(0))
    perturbed_loss = loss_fn(perturbed_solver, jr.key(1))
    assert jnp.isrealobj(exact_loss)
    assert jnp.isrealobj(perturbed_loss)
    assert exact_loss < 1e-20
    assert perturbed_loss > 1e-4
