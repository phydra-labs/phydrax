#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


cpl = phx.solver.coupling


def test_nonmatching_dirichlet_neumann_heat_domains_balance_interface_flux():
    interface = phx.linalg.ArraySpace(
        (1,), dtype=jnp.float64, space_id="heat-interface-scalar"
    )
    left_temperature = cpl.CouplingPort(
        "left-interface-temperature",
        "input",
        interface,
        reference_scale=1.0,
    )
    left_flux = cpl.CouplingPort(
        "left-interface-flux",
        "output",
        interface,
        reference_scale=1.0,
    )
    right_flux = cpl.CouplingPort(
        "right-interface-flux",
        "input",
        interface,
        reference_scale=1.0,
    )
    right_temperature = cpl.CouplingPort(
        "right-interface-temperature",
        "output",
        interface,
        reference_scale=1.0,
    )
    capabilities = cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=True,
        deterministic_replay=True,
        fixed_topology=True,
    )

    def solve_left(window, state, inputs, args):
        del window, state, args
        interface_temperature = inputs[0][0]
        profile = jnp.linspace(1.0, interface_temperature, 5)
        outgoing_flux = jnp.asarray([1.0 - interface_temperature])
        return cpl.CouplingSubsystemResult(
            profile,
            (outgoing_flux,),
            successful=True,
            status=0,
            residual_norm=0.0,
            work=5,
        )

    def solve_right(window, state, inputs, args):
        del window, state, args
        incoming_flux = inputs[0][0]
        interface_temperature = incoming_flux
        profile = jnp.linspace(interface_temperature, 0.0, 7)
        return cpl.CouplingSubsystemResult(
            profile,
            (jnp.asarray([interface_temperature]),),
            successful=True,
            status=0,
            residual_norm=0.0,
            work=7,
        )

    left = cpl.CallableCouplingSubsystem(
        solve_left,
        subsystem_id="left-domain",
        input_ports=(left_temperature,),
        output_ports=(left_flux,),
        capabilities=capabilities,
    )
    right = cpl.CallableCouplingSubsystem(
        solve_right,
        subsystem_id="right-domain",
        input_ports=(right_flux,),
        output_ports=(right_temperature,),
        capabilities=capabilities,
    )
    graph = cpl.CouplingGraph(
        (right, left),
        (
            cpl.CouplingExchange(
                "heat-flux", "left-interface-flux", "right-interface-flux"
            ),
            cpl.CouplingExchange(
                "temperature",
                "right-interface-temperature",
                "left-interface-temperature",
            ),
        ),
    )
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(
            acceleration=phx.nonlinear.AndersonAcceleration(history=3)
        ),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-12,
            relative_residual=0.0,
            maximum_steps=30,
        ),
        (
            cpl.CouplingTolerance("left-interface-temperature", absolute=1e-10),
            cpl.CouplingTolerance("right-interface-flux", absolute=1e-10),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )
    prepared = cpl.prepare_coupling(
        graph,
        (jnp.zeros(7), jnp.zeros(5)),
        (jnp.zeros(1), jnp.zeros(1)),
        policy=policy,
        problem_id="nonmatching-conjugate-heat",
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert bool(result.converged)
    heat_flux, interface_temperature = result.accepted_state.exchange_values
    assert float(heat_flux[0]) == pytest.approx(0.5, abs=1e-9)
    assert float(interface_temperature[0]) == pytest.approx(0.5, abs=1e-9)
    left_profile, right_profile = result.accepted_state.participant_states
    assert left_profile.shape == (5,)
    assert right_profile.shape == (7,)
    assert float(left_profile[-1] - right_profile[0]) == pytest.approx(0.0, abs=1e-9)
    assert float((1.0 - left_profile[-1]) - heat_flux[0]) == pytest.approx(0.0, abs=1e-9)
    assert jnp.max(result.diagnostics.exchange_residual_norms) < 1e-10
