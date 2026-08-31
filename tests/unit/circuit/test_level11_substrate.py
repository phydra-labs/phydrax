import jax.numpy as jnp

import phydrax as phx


def _block_network():
    reference = phx.circuit.ElectricalWaveReference(50.0)
    coordinates = ("a", "b")
    left = phx.circuit.WavePort(
        "left", (reference, reference), coordinate_ids=coordinates
    )
    right = phx.circuit.WavePort(
        "right", (reference, reference), coordinate_ids=coordinates
    )
    zero = jnp.zeros((2, 2), dtype=jnp.complex128)
    identity = jnp.eye(2, dtype=jnp.complex128)
    through = phx.circuit.MatrixScatteringComponent(
        jnp.block([[zero, identity], [identity, zero]]),
        (left, right),
        component_id="block-through",
    )
    return phx.circuit.ScatteringNetwork(
        (phx.circuit.ScatteringInstance("device", through),),
        (),
        (
            phx.circuit.InstancePort("device", "left"),
            phx.circuit.InstancePort("device", "right"),
        ),
        external_port_ids=("left", "right"),
        network_id="block-network",
    )


def _rc_circuit():
    reference = phx.circuit.ElectricalWaveReference(50.0)
    source = phx.circuit.CircuitElement(
        phx.circuit.IndependentCurrentSourceLaw(1.0),
        element_id="source",
    )
    return phx.circuit.NodalCircuit(
        (
            phx.circuit.CircuitInstance(
                "resistor", phx.circuit.Resistor(1.0), ("n", "0")
            ),
            phx.circuit.CircuitInstance(
                "capacitor", phx.circuit.Capacitor(1.0), ("n", "0")
            ),
            phx.circuit.CircuitInstance("source", source, ("0", "n")),
        ),
        (phx.circuit.NodalPort("port", "n", "0", reference),),
        ground="0",
        circuit_id="rc-circuit",
    )


def test_block_ports_action_runtime_case_batch_and_connection_map():
    network = _block_network()
    assert tuple(port.size for port in network.ports) == (2, 2)
    dense = phx.circuit.full_scattering_matrix(
        phx.circuit.prepare_scattering_network(network, jnp.asarray(1.0))
    )
    action = phx.circuit.prepare_scattering_action(network, jnp.asarray(1.0))
    action_matrix = phx.circuit.scattering_action_submatrix(
        action,
        action.plan.external_channel_ids,
        action.plan.external_channel_ids,
    )
    assert jnp.allclose(action_matrix, dense, atol=1e-10)
    assert action.plan.cost.retained_bytes < 2 * dense.nbytes

    swap = phx.circuit.WaveConnectionMap(jnp.asarray([[0.0, 1.0], [1.0, 0.0]]))
    assert jnp.allclose(swap.reverse @ swap.forward, jnp.eye(2))

    batch = phx.circuit.prepare_scattering_action_case_batch(
        (network, network), (jnp.asarray(1.0), jnp.asarray(2.0)), (2,)
    )
    incident = jnp.eye(4, dtype=jnp.complex128)
    batched = phx.circuit.solve_scattering_action_case_batch(batch, incident)
    assert batched.external_outgoing.shape == (2, 4, 4)
    assert jnp.all(batched.status == 0)


def test_relation_graph_sparse_action_and_bounded_materialization():
    from phydrax.circuit._relation_graph import bind_linear_relation, plan_linear_routes

    plan = plan_linear_routes(2, 2, (0, 1, 0), (0, 1, 1))
    prepared = bind_linear_relation(plan, jnp.asarray([2.0, 3.0, 4.0]))
    assert jnp.allclose(prepared.apply(jnp.asarray([1.0, 2.0])), jnp.asarray([2.0, 10.0]))
    assert jnp.allclose(
        prepared.materialize(maximum_bytes=128), jnp.asarray([[2.0, 0.0], [4.0, 3.0]])
    )


def test_circuit_dae_operating_point_descriptor_and_periodic_contracts():
    prepared = phx.circuit.prepare_circuit_dae(_rc_circuit())
    assert prepared.plan.layout.roles == ("differential",)
    state = prepared.initialize(node_voltages=jnp.asarray([0.0]))
    diagnostics = prepared.diagnostics(0.0, state, jnp.asarray([1.0]))
    assert jnp.allclose(diagnostics.residual, 0.0, atol=1e-12)

    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 11), time_id="rc-grid")
    solved = phx.circuit.solve_circuit_dae(prepared, state, grid)
    assert jnp.all(solved.solution.valid)
    assert jnp.allclose(solved.solution.states[-1, 0], 1.0 - jnp.exp(-1.0), rtol=3e-2)

    operating = phx.circuit.prepare_circuit_operating_point(prepared, jnp.asarray([0.5]))
    root = phx.circuit.solve_circuit_operating_point(operating)
    assert bool(root.nonlinear.successful)
    assert jnp.allclose(root.state, 1.0, atol=1e-9)

    linearized = phx.circuit.linearize_circuit(prepared, root.state)
    response = phx.circuit.circuit_small_signal_response(
        linearized, jnp.asarray([0.0, 1.0])
    )
    assert jnp.allclose(response.response[0, 0, 0], 1.0, atol=1e-10)

    waveform = jnp.ones((5, 1))
    harmonic = phx.circuit.solve_harmonic_balance(prepared, waveform, jnp.asarray(1.0))
    assert harmonic.diagnostics.residual_norm < 1e-9

    floquet = phx.circuit.floquet_multipliers(lambda value: 0.5 * value, jnp.ones((1,)))
    assert jnp.allclose(floquet.multipliers, 0.5)
    assert bool(floquet.stable)


def test_rational_realization_noise_metrology_and_identifiability():
    poles = jnp.asarray([-1.0 + 0.0j, -3.0 + 0.0j])
    residues = jnp.asarray([[[2.0 + 0.0j]], [[0.5 + 0.0j]]])
    model = phx.circuit.RationalMatrixModel(
        poles, residues, jnp.asarray([[0.1]]), jnp.zeros((1, 1))
    )
    omega = jnp.linspace(0.0, 10.0, 41)
    samples = model.evaluate_frequency(omega)
    fit = phx.circuit.fit_rational_matrix(
        omega,
        samples,
        policy=phx.circuit.RationalFitPolicy(pole_count=2, residual_tolerance=1e-8),
        poles=poles,
    )
    assert bool(fit.evidence.accepted)
    descriptor = phx.circuit.realize_rational_model(fit.model)
    descriptor_response = phx.control.descriptor_frequency_response(descriptor, omega)
    assert jnp.allclose(descriptor_response.response, samples, rtol=1e-8, atol=1e-8)
    reduced = phx.circuit.reduce_rational_model(model, 1)
    assert reduced.model.poles.size == 1
    assert not reduced.passivity_preserved
    audit = phx.circuit.audit_rational_scattering(model, omega)
    assert not jnp.all(audit.passive)
    passive_system, certificate = phx.circuit.passive_descriptor_system(
        jnp.eye(1),
        jnp.zeros((1, 1)),
        jnp.eye(1),
        jnp.ones((1, 1)),
        jnp.zeros((1, 1)),
    )
    assert bool(certificate.certified)
    assert passive_system.state_size == 1

    noise = phx.circuit.propagate_descriptor_noise(
        descriptor,
        omega,
        phx.circuit.NoiseSpectralFactor(jnp.ones((1, 1))),
    )
    assert jnp.all(noise.diagnostics.positive_semidefinite)

    references = (
        phx.circuit.ElectricalWaveReference(50.0),
        phx.circuit.ElectricalWaveReference(50.0),
    )
    dut = jnp.asarray([[0.1, 0.7], [0.7, 0.1]], dtype=jnp.complex128)
    identity_abcd = jnp.eye(2, dtype=jnp.complex128)
    error = phx.circuit.VNAErrorModel(identity_abcd, identity_abcd)
    measured = phx.circuit.apply_vna_error_model(dut, error, references)
    deembedded = phx.circuit.deembed_two_port(measured, error, references)
    assert jnp.allclose(deembedded.scattering, dut, atol=1e-9)

    identifiability = phx.circuit.parameter_identifiability(
        lambda parameters, args: jnp.asarray(
            [parameters[0] + 2.0 * parameters[1], parameters[0] - parameters[1]]
        ),
        jnp.asarray([1.0, 2.0]),
    )
    assert bool(identifiability.identifiable)


def test_spice_behavioral_learned_and_electrothermal_adapters():
    reference = phx.circuit.ElectricalWaveReference(50.0)
    imported = phx.circuit.read_spice_netlist(
        "R1 n 0 1k\nC1 n 0 1u\nV1 n 0 1\nG1 n 0 n 0 2",
        (phx.circuit.NodalPort("port", "n", "0", reference),),
        circuit_id="imported",
    )
    assert len(imported.circuit.instances) == 4
    phx.circuit.prepare_circuit_dae(imported.circuit)

    behavioral = phx.circuit.compile_behavioral_current(
        "p_g * v + u_bias", parameters={"g": 2.0}, input_names=("bias",)
    )
    evaluated = behavioral.implicit_law.evaluate(
        jnp.asarray(0.0),
        jnp.asarray([1.0, 0.0]),
        jnp.zeros((2,)),
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        {"bias": 0.5},
        None,
    )
    assert jnp.allclose(evaluated.terminal_currents, jnp.asarray([2.5, -2.5]))

    learned = phx.circuit.learned_conductance_element(lambda voltage: 0.0 * voltage)
    learned_evaluation = learned.implicit_law.evaluate(
        jnp.asarray(0.0),
        jnp.asarray([2.0, 0.0]),
        jnp.zeros((2,)),
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        None,
        None,
    )
    assert learned_evaluation.terminal_currents[0] > 0.0
    assert learned.implicit_law.evidence.passive_by_construction

    circuit = phx.circuit.prepare_circuit_dae(_rc_circuit())
    coupled = phx.circuit.prepare_electrothermal_circuit(circuit, 2.0, 1.0, 300.0)
    coupled_state = coupled.initialize(jnp.asarray([1.0]), jnp.asarray(300.0))
    coupled_rate = jnp.asarray([0.0, 1.0])
    coupled_diagnostics = coupled.diagnostics(
        0.0,
        coupled_state,
        coupled_rate,
        {"heat_power": lambda time, state, temperature, args: jnp.asarray(2.0)},
    )
    assert jnp.allclose(coupled_diagnostics.thermal_residual, 0.0, atol=1e-12)
