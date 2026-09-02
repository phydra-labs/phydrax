# Circuit-QED design and control

This workflow builds two reduced modes, exposes one control line, propagates all
computational basis columns, and evaluates leakage-sensitive gate quality.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

q = phx.operators.quantum
s = phx.solver

# Two retained harmonic modes on a one-edge processor topology.
topology = phx.graph.GraphIR(
    n_node=jnp.asarray([2]),
    n_edge=jnp.asarray([1]),
    senders=jnp.asarray([0]),
    receivers=jnp.asarray([1]),
)
basis = q.OscillatorBasis(8)
reduction = q.ModeReductionPolicy(3, minimum_boundary_gap=1e-3)
placements = (
    s.CircuitModePlacement("q0", "harmonic", basis, 0, reduction),
    s.CircuitModePlacement("q1", "harmonic", basis, 1, reduction),
)
interaction = s.CircuitInteraction(
    (0, 1),
    ("phase", "phase"),
    0,
)
port = s.CircuitDrivePort(0, "charge", 0)
spec = s.CircuitQEDDeviceSpec(
    topology,
    placements,
    (interaction,),
    drive_ports=(port,),
)
parameters = s.CircuitQEDDeviceParameters(
    (
        q.HarmonicModeParameters(4.8),
        q.HarmonicModeParameters(5.1),
    ),
    interaction_strengths=jnp.asarray([0.03]),
    drive_scales=jnp.asarray([1.0]),
)
device = s.prepare_circuit_qed_device(spec, parameters)

# A scalar piecewise-linear I/Q envelope and one direct actuator.
time_grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 12.0, 97),
    time_id="two-mode-control",
)
parameterization = phx.control.PiecewiseLinearControlParameterization(
    time_grid,
    (),
    parameterization_id="q0-envelope",
)
initial_coefficients = jnp.zeros(parameterization.parameter_shape)
transfer = s.LinearQuantumControlTransfer(jnp.asarray([[1.0]]))


def fixed_grid_hamiltonian(coefficients):
    line = s.QuantumControlLine(
        parameterization,
        coefficients,
        quadrature_coefficients=jnp.zeros_like(coefficients),
        carrier=s.QuantumCarrier(angular_rate=4.8),
        support_start=0.0,
        support_stop=12.0,
    )
    sampled = s.sample_quantum_control_schedule(
        s.QuantumControlSchedule((line,), transfer),
        time_grid.times,
    )
    return s.assemble_circuit_qed_hamiltonian(device, sampled)


nominal_schedule = fixed_grid_hamiltonian(initial_coefficients)
prepared_evolution = s.prepare_local_hamiltonian_evolution(
    nominal_schedule,
    policy=s.LocalHamiltonianEvolutionPolicy(
        order=2,
        differentiation="reversible-product-formula",
    ),
)

# q0 and q1 use levels 0 and 1; retained level 2 remains a leakage level.
computational = q.basis_state_subspace(
    device.plan.layout,
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)
logical_target = jnp.eye(4, dtype=jnp.complex128)
input_columns = jnp.eye(device.plan.layout.dimension, dtype=jnp.complex128)


def objective(coefficients, args):
    del args
    schedule = fixed_grid_hamiltonian(coefficients)
    evolved = s.solve_local_hamiltonian_evolution(
        s.refresh_local_hamiltonian_evolution(prepared_evolution, schedule),
        input_columns,
    )
    physical_action = evolved.final_state.T
    quality = q.unitary_gate_quality(
        physical_action,
        logical_target,
        computational,
    )
    # Fidelity already includes leakage; report leakage separately after optimization.
    return 1.0 - quality.average_fidelity


value, gradient = jax.value_and_grad(lambda coefficients: objective(coefficients, None))(
    initial_coefficients
)
result = phx.optim.minimize(
    objective,
    initial_coefficients,
    method=phx.optim.NonlinearConjugateGradient(),
    termination=phx.optim.OptimizationTermination(maximum_steps=32),
)

final_schedule = fixed_grid_hamiltonian(result.parameters)
final_evolution = s.solve_local_hamiltonian_evolution(
    s.refresh_local_hamiltonian_evolution(prepared_evolution, final_schedule),
    input_columns,
)
final_quality = q.unitary_gate_quality(
    final_evolution.final_state.T,
    logical_target,
    computational,
)

print("initial objective:", value)
print("gradient norm:", jnp.sqrt(jnp.sum(gradient * gradient)))
print("final average fidelity:", final_quality.average_fidelity)
print("final leakage:", final_quality.leakage)
print("device valid:", device.diagnostics.valid)
print("evolution valid:", final_evolution.diagnostics.valid)
```

The example intentionally keeps the physical model small. Increase raw basis size,
retained level count, interval count, or tensor-network capacity only together with the
corresponding convergence and resource evidence.
