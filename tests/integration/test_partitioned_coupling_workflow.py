#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


cpl = phx.solver.coupling


def test_three_participant_cycle_solves_and_rolls_out_with_certified_residuals():
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="three-way-interface")
    capabilities = cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=True,
        deterministic_replay=True,
        fixed_topology=True,
    )

    def participant(name, forcing):
        input_port = cpl.CouplingPort(
            f"{name}-input", "input", space, reference_scale=4.0
        )
        output_port = cpl.CouplingPort(
            f"{name}-output", "output", space, reference_scale=4.0
        )

        def advance(window, state, inputs, args):
            del window, state, args
            value = 0.25 * inputs[0] + forcing
            return cpl.CouplingSubsystemResult(
                value, (value,), successful=True, status=0, work=1
            )

        return cpl.CallableCouplingSubsystem(
            advance,
            subsystem_id=name,
            input_ports=(input_port,),
            output_ports=(output_port,),
            capabilities=capabilities,
        )

    a = participant("a", 1.0)
    b = participant("b", 2.0)
    c = participant("c", 3.0)
    graph = cpl.CouplingGraph(
        (c, a, b),
        (
            cpl.CouplingExchange("a-to-b", "a-output", "b-input"),
            cpl.CouplingExchange("b-to-c", "b-output", "c-input"),
            cpl.CouplingExchange("c-to-a", "c-output", "a-input"),
        ),
    )
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(
            acceleration=phx.nonlinear.AndersonAcceleration(history=5)
        ),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11,
            relative_residual=0.0,
            maximum_steps=40,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-9),
            cpl.CouplingTolerance("b-input", absolute=1e-9),
            cpl.CouplingTolerance("c-input", absolute=1e-9),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )
    problem = cpl.CouplingProblem(
        graph,
        (jnp.zeros(1),) * 3,
        (jnp.zeros(1),) * 3,
        policy,
        t0=0.0,
        t1=2.0,
        window_size=1.0,
    )

    solution = cpl.solve_coupling(
        problem,
        rollout=cpl.CouplingRolloutPlan(retention="trajectory"),
    )

    assert bool(solution.successful)
    assert solution.retained_valid.tolist() == [True, True, True]
    assert jnp.all(solution.converged)
    assert jnp.max(solution.exchange_residual_norms) < 1e-9
    assert jnp.all(solution.participant_evaluations > 0)
    assert solution.final_state.subsystem_ids == ("a", "b", "c")
