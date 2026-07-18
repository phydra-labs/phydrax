#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


LOWERING = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def test_lindblad_residual_runs_through_functional_solver():
    time = phx.domain.TimeInterval(0.0, 2.0)
    rate = 0.8
    hamiltonian = time.Function()(jnp.zeros((2, 2), dtype=complex))
    collapse = time.Function()(jnp.sqrt(rate) * LOWERING)

    @time.Function("t")
    def exact_density(t):
        excited = jnp.exp(-rate * t)
        return jnp.asarray(
            [[1.0 - excited, 0.0], [0.0, excited]],
            dtype=complex,
        )

    @time.Function("t")
    def perturbed_density(t):
        excited = jnp.exp(-0.6 * rate * t)
        return jnp.asarray(
            [[1.0 - excited, 0.0], [0.0, excited]],
            dtype=complex,
        )

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=time.component(),
        operator=lambda density: phx.operators.lindblad_residual(
            density,
            hamiltonian,
            (collapse,),
        ),
        constraint_vars="rho",
        num_points=32,
        structure=phx.domain.ProductStructure((("t",),)),
        reduction="mean",
    )
    exact = phx.solver.FunctionalSolver(
        functions={"rho": exact_density},
        constraints=[constraint],
    )
    perturbed = phx.solver.FunctionalSolver(
        functions={"rho": perturbed_density},
        constraints=[constraint],
    )

    loss = eqx.filter_jit(lambda solver, key: solver.loss(key=key))
    exact_loss = loss(exact, jr.key(0))
    perturbed_loss = loss(perturbed, jr.key(1))

    assert jnp.isrealobj(exact_loss)
    assert jnp.isrealobj(perturbed_loss)
    assert exact_loss < 1e-20
    assert perturbed_loss > 1e-4
