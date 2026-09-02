#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


q = phx.operators.quantum
s = phx.solver


def test_circuit_spectrum_fit_composes_mode_tracking_observations_and_optimization():
    basis = q.ChargeBasis(4)
    policy = q.ModeReductionPolicy(3)
    phases = (-0.4, 0.0, 0.35)
    true_charging_rate = jnp.asarray(0.25)
    layout = phx.observation.CoordinateLayout(
        tuple(f"transition:{index}" for index in range(len(phases)))
    )

    def mode_problem(charging_rate, phase):
        return q.transmon_mode_problem(
            q.TransmonParameters(
                charging_rate,
                8.0,
                7.0,
                external_phase=phase,
            ),
            basis,
            problem_id=f"spectrum:{phase}",
        )

    def transition(reduced):
        return reduced.energies[1] - reduced.energies[0]

    target = jnp.stack(
        tuple(
            transition(
                q.prepare_mode_reduction(
                    mode_problem(true_charging_rate, phase),
                    policy=policy,
                )
            )
            for phase in phases
        )
    )
    references = tuple(
        q.prepare_mode_reduction(
            mode_problem(jnp.asarray(0.18), phase),
            policy=policy,
        )
        for phase in phases
    )

    def theory(log_charging_rate):
        charging_rate = jnp.exp(log_charging_rate)
        values = jnp.stack(
            tuple(
                transition(
                    q.refresh_mode_reduction(
                        reference,
                        mode_problem(charging_rate, phase),
                    )
                )
                for reference, phase in zip(references, phases, strict=True)
            )
        )
        return phx.observation.TheoryVector(values, layout, "circuit-spectrum")

    def objective(log_charging_rate, args):
        del args
        residual = theory(log_charging_rate).values - target
        return 0.5 * jnp.sum(residual * residual)

    initial = jnp.log(jnp.asarray(0.18))
    initial_objective = objective(initial, None)
    optimized = phx.optim.minimize(
        objective,
        initial,
        method=phx.optim.NonlinearConjugateGradient(),
        termination=phx.optim.OptimizationTermination(maximum_steps=16),
    )
    fitted = jnp.exp(optimized.parameters)

    assert optimized.objective <= initial_objective
    assert jnp.allclose(fitted, true_charging_rate, rtol=5e-3, atol=5e-4)
    assert all(
        bool(
            q.refresh_mode_reduction(
                reference,
                mode_problem(fitted, phase),
            ).diagnostics.valid
        )
        for reference, phase in zip(references, phases, strict=True)
    )


def test_circuit_gate_control_composes_device_controls_evolution_and_gate_metrics():
    topology = phx.graph.GraphIR(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([0]),
    )
    basis = q.OscillatorBasis(4)
    placement = s.CircuitModePlacement(
        "q",
        "harmonic",
        basis,
        0,
        q.ModeReductionPolicy(2),
    )
    spec = s.CircuitQEDDeviceSpec(
        topology,
        (placement,),
        (),
        drive_ports=(s.CircuitDrivePort(0, "phase", 0),),
    )
    device = s.prepare_circuit_qed_device(
        spec,
        s.CircuitQEDDeviceParameters(
            (q.HarmonicModeParameters(2.0),),
            drive_scales=jnp.asarray([1.0]),
        ),
    )
    time_grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.6, 5),
        time_id="gate-control-grid",
    )
    parameterization = phx.control.PiecewiseConstantControlParameterization(
        time_grid,
        (),
        parameterization_id="gate-amplitude",
    )
    subspace = q.BasisStateSubspace(2, (0, 1))
    initial_unitary_columns = jnp.eye(2, dtype=jnp.complex128)

    def schedule(amplitude):
        line = s.QuantumControlLine(
            parameterization,
            jnp.full((4,), amplitude),
            support_start=0.0,
            support_stop=0.6,
        )
        controls = s.sample_quantum_control_schedule(
            s.QuantumControlSchedule(
                (line,),
                s.LinearQuantumControlTransfer(jnp.asarray([[1.0]])),
            ),
            time_grid.times,
        )
        return s.assemble_circuit_qed_hamiltonian(device, controls)

    target_schedule = schedule(jnp.asarray(0.45))
    target_evolution = s.solve_local_hamiltonian_evolution(
        s.prepare_local_hamiltonian_evolution(target_schedule),
        initial_unitary_columns,
    )
    target = target_evolution.final_state.T
    reference_evolution = s.prepare_local_hamiltonian_evolution(
        schedule(jnp.asarray(0.1))
    )

    def objective(amplitude, args):
        del args
        evolved = s.solve_local_hamiltonian_evolution(
            s.refresh_local_hamiltonian_evolution(
                reference_evolution,
                schedule(amplitude),
            ),
            initial_unitary_columns,
        )
        physical_unitary = evolved.final_state.T
        quality = q.unitary_gate_quality(physical_unitary, target, subspace)
        return 1.0 - quality.average_fidelity

    initial = jnp.asarray(0.1)
    initial_objective = objective(initial, None)
    optimized = phx.optim.minimize(
        objective,
        initial,
        method=phx.optim.NonlinearConjugateGradient(),
        termination=phx.optim.OptimizationTermination(maximum_steps=12),
    )

    assert bool(device.diagnostics.valid)
    assert bool(target_evolution.successful)
    assert optimized.objective < initial_objective
    assert optimized.objective < 0.01 * initial_objective
