#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent uniform-conductor limits and conservative transfer invariants."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications import semiconductor as sc
from phydrax.linalg import GMRES, LinearSolvePolicy, TolerancePolicy


jax.config.update("jax_enable_x64", True)
Q = 1.602176634e-19
LENGTH = 1e-6
AREA = 1e-12


def _uniform_device(nodes=7):
    support = sc.TransportSupport.interval(np.linspace(0.0, LENGTH, nodes), area=AREA)
    plan = sc.DevicePlan(
        support,
        sc.SemiconductorMaterial.silicon(),
        contacts=(
            sc.OhmicContact("left", support.boundary_patch("left")),
            sc.OhmicContact("right", support.boundary_patch("right", side="upper")),
        ),
    )
    return sc.PreparedSemiconductorDevice(plan)


def _conductance_capacitance(device):
    plan = device.plan
    conductance = (
        Q
        * plan.intrinsic_density[0]
        * (plan.electron_mobility[0] + plan.hole_mobility[0])
        * AREA
        / LENGTH
    )
    capacitance = plan.permittivity[0] * AREA / LENGTH
    return conductance, capacitance


def test_uniform_small_signal_matches_distributed_conductor_and_displacement():
    device = _uniform_device()
    point = device.equilibrium()
    frequencies = jnp.asarray([0.0, 1e5, 1e7])
    response = sc.semiconductor_small_signal(device, point, frequencies)
    assert bool(jnp.all(response.evidence.successful))
    conductance, capacitance = _conductance_capacitance(device)
    terminal_pattern = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    expected = (conductance + 1j * frequencies * capacitance)[
        :, None, None
    ] * terminal_pattern
    np.testing.assert_allclose(response.admittance, expected, rtol=2e-7, atol=1e-18)
    # Both terminal KCL and invariance to a common-mode voltage must survive AC.
    np.testing.assert_allclose(jnp.sum(response.admittance, axis=1), 0.0, atol=1e-17)
    np.testing.assert_allclose(jnp.sum(response.admittance, axis=2), 0.0, atol=1e-17)


def test_native_transient_keeps_carriers_and_initial_displacement_consistent():
    device = _uniform_device(5)
    point = device.equilibrium()
    slope = 2e5
    times = jnp.linspace(0.0, 1e-8, 5)
    drive = lambda time: jnp.asarray([slope * time, 0.0])
    result = sc.semiconductor_transient(device, point, times, drive)
    assert bool(jnp.all(result.successful))
    conductance, capacitance = _conductance_capacitance(device)
    expected_conduction = conductance * slope * times
    expected_displacement = capacitance * slope
    np.testing.assert_allclose(
        result.conduction_currents[:, 0], expected_conduction, rtol=2e-6, atol=1e-18
    )
    np.testing.assert_allclose(
        result.displacement_currents[:, 0], expected_displacement, rtol=2e-6, atol=1e-18
    )
    np.testing.assert_allclose(
        jnp.sum(result.terminal_currents, axis=-1), 0.0, atol=1e-17
    )
    n, p = jax.vmap(device.densities)(result.coordinates)
    np.testing.assert_allclose(
        n, jnp.broadcast_to(device.plan.intrinsic_density, n.shape), rtol=2e-7
    )
    np.testing.assert_allclose(
        p, jnp.broadcast_to(device.plan.intrinsic_density, p.shape), rtol=2e-7
    )


def test_depleted_pn_transient_resolves_screening_without_losing_terminal_charge():
    length, area, slope = 4e-6, 1e-12, 5e6
    device = sc.PreparedSemiconductorDevice(sc.pn_junction(21, length=length, area=area))
    equilibrium = device.equilibrium()
    times = jnp.linspace(0.0, 1e-9, 5)
    result = sc.semiconductor_transient(
        device,
        equilibrium,
        times,
        lambda time: jnp.asarray([slope * time, 0.0]),
    )
    assert bool(jnp.all(result.successful))
    geometric_current = device.plan.permittivity[0] * area / length * slope
    # Initially the frozen carrier population gives the geometric capacitance.
    np.testing.assert_allclose(
        result.terminal_currents[0],
        geometric_current * jnp.asarray([1.0, -1.0]),
        rtol=1e-6,
        atol=1e-19,
    )
    # Carrier redistribution screens the neutral bulk and raises capacitance.
    assert float(result.terminal_currents[-1, 0]) > 2 * float(geometric_current)
    np.testing.assert_allclose(
        jnp.sum(result.terminal_currents, axis=-1),
        0.0,
        atol=1e-18,
    )


def test_implicit_bias_material_and_geometry_derivatives_match_ohms_law():
    device = _uniform_device()
    point = device.solve(jnp.asarray([0.005, 0.0]))
    assert bool(point.successful)
    conductance, _ = _conductance_capacitance(device)
    bias = sc.semiconductor_sensitivity(device, point)
    assert bool(jnp.all(bias.evidence.successful))
    np.testing.assert_allclose(
        bias.derivatives,
        conductance * jnp.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        rtol=2e-7,
        atol=1e-18,
    )

    def parameterize(theta):
        mobility, length = theta
        plan = device.plan
        changed = eqx.tree_at(
            lambda p: (
                p.electron_mobility,
                p.hole_mobility,
                p.support.positions,
                p.support.volumes,
                p.support.transmissibility,
            ),
            plan,
            (
                mobility * plan.electron_mobility,
                mobility * plan.hole_mobility,
                length * plan.support.positions,
                length * plan.support.volumes,
                plan.support.transmissibility / length,
            ),
        )
        return eqx.tree_at(lambda d: d.plan, device, changed), point.voltages

    sensitivity = sc.semiconductor_sensitivity(
        device,
        point,
        parameters=jnp.ones(2),
        parameterize=parameterize,
    )
    assert bool(jnp.all(sensitivity.evidence.successful))
    current = conductance * point.voltages[0]
    expected = current * jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(sensitivity.derivatives, expected, rtol=2e-7, atol=1e-18)


def test_circuit_law_preserves_si_signs_and_carrier_storage_dynamics():
    device = _uniform_device()
    law = sc.SemiconductorCircuitLaw(device)
    voltage, slope = 0.003, 7e5
    fraction = 1.0 - device.plan.support.positions[:, 0] / LENGTH
    potential = voltage * fraction / device.plan.thermal_voltage
    potential_rate = slope * fraction / device.plan.thermal_voltage
    coordinates = jnp.stack((potential, -potential, -potential), axis=-1)
    rate = jnp.stack((potential_rate, -potential_rate, -potential_rate), axis=-1)
    value = law.evaluate(
        jnp.asarray(0.0),
        jnp.asarray([voltage, 0.0]),
        jnp.asarray([slope, 0.0]),
        law.initialize(coordinates),
        law.initialize_rate(coordinates, rate),
        None,
        None,
    )
    conductance, capacitance = _conductance_capacitance(device)
    expected = conductance * voltage + capacitance * slope
    np.testing.assert_allclose(
        value.terminal_currents,
        jnp.asarray([expected, -expected]),
        rtol=2e-10,
        atol=1e-20,
    )
    np.testing.assert_allclose(value.auxiliary_residual, 0.0, atol=1e-10)


def test_failed_linear_solve_never_exposes_admittance_as_valid():
    device = _uniform_device(11)
    point = device.equilibrium()
    policy = LinearSolvePolicy(
        GMRES(restart=1),
        tolerance=TolerancePolicy(relative=1e-14, absolute=1e-15, max_steps=1),
    )
    result = sc.semiconductor_small_signal(
        device, point, jnp.asarray([1e7]), linear_policy=policy
    )
    assert not bool(jnp.any(result.evidence.successful))
    assert bool(jnp.all(jnp.isnan(result.admittance)))
    assert bool(
        jnp.all(result.evidence.residual_norm > result.evidence.residual_threshold)
    )


def _triangle_result():
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray([[0.0, 0.0], [1e-6, 0.0], [0.5e-6, np.sqrt(3.0) * 0.5e-6]]),
        np.asarray([[0, 1, 2]], dtype=np.int32),
        vertex_global_ids=np.asarray([11, 23, 37]),
        cell_global_ids=np.asarray([101]),
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _triangle_device(result):
    support = sc.TransportSupport.from_meshing(result, transverse_measure=1e-6)
    patch = phx.meshing.MeshPatch("reservoir", support.node_scope())
    plan = sc.DevicePlan(
        support,
        sc.SemiconductorMaterial.silicon(),
        contacts=(sc.OhmicContact("reservoir", patch),),
    )
    return sc.PreparedSemiconductorDevice(plan)


def _geometry_transition(source_result, scale):
    source = source_result.mesh
    moved = source.with_coordinates(scale * source.coordinates, numeric_version="next")
    target = phx.meshing.certify_cell_mesh(moved, phx.SpatialCoordinateContract.si())
    destination = target.mesh
    ids = source.vertex_global_ids
    source_set, target_set = (
        source.entity_set(0).entity_set_id,
        destination.entity_set(0).entity_set_id,
    )
    vertices = phx.meshing.EntityLineage(
        0,
        source_set,
        target_set,
        ids,
        ids,
        jnp.full(ids.shape, int(phx.meshing.EntityLineageKind.PRESERVED)),
    )
    lineage = phx.meshing.MeshLineage(
        source.topology_id, destination.topology_id, (vertices,)
    )
    stencil = phx.meshing.VertexInterpolationStencil(
        source_set,
        target_set,
        ids,
        ids[:, None],
        jnp.ones((ids.size, 1)),
        jnp.ones((ids.size, 1), dtype=bool),
    )
    return phx.meshing.CellMeshTransition(
        source.mesh_id,
        source.topology_id,
        target,
        lineage,
        phx.meshing.MeshTransitionKind.REMESH,
        vertex_stencil=stencil,
    )


def test_native_transfer_conserves_particles_and_rejects_reservoir_mass_creation_atomically():
    source = _triangle_result()
    device = _triangle_device(source)
    point = device.equilibrium()
    transition = _geometry_transition(source, 1.0)
    target = _triangle_device(transition.target)
    accepted = sc.semiconductor_reprepare(
        device, point, target, transition, source_result=source
    )
    assert bool(accepted.accepted)
    np.testing.assert_allclose(
        accepted.evidence.reinitialized_counts,
        accepted.evidence.source_counts,
        rtol=1e-12,
    )
    np.testing.assert_allclose(accepted.coordinates, point.coordinates, atol=1e-12)

    enlarged = _geometry_transition(source, 2.0)
    enlarged_device = _triangle_device(enlarged.target)
    rejected = sc.semiconductor_reprepare(
        device, point, enlarged_device, enlarged, source_result=source
    )
    assert not bool(rejected.accepted)
    assert rejected.prepared is device
    np.testing.assert_array_equal(rejected.coordinates, point.coordinates)
    # Remapping itself conserves. The reservoir consistency projection would
    # create carriers in the expanded physical volume, so it must not commit.
    np.testing.assert_allclose(
        rejected.evidence.transferred_counts, rejected.evidence.source_counts, rtol=1e-12
    )
    assert not bool(rejected.evidence.conservative)

    with pytest.raises(ValueError, match="revision"):
        sc.semiconductor_reprepare(
            device, point, target, transition, source_result=transition.target
        )


def test_coupled_nanoampere_circuit_dc_and_rc_transient_are_physically_scaled():
    device = _uniform_device(3)
    plan, support = device.plan, device.plan.support
    conductance, capacitance = _conductance_capacitance(device)
    total_conductance = conductance + 1e-9
    final_voltage = 1e-10 / total_conductance
    time_constant = capacitance / total_conductance
    law = sc.SemiconductorCircuitLaw(device)
    circuit = phx.circuit.NodalCircuit(
        (
            phx.circuit.CircuitInstance(
                "device", phx.circuit.CircuitElement(law, element_id="device"), ("n", "0")
            ),
            phx.circuit.CircuitInstance(
                "resistor", phx.circuit.Resistor(1e9), ("n", "0")
            ),
            phx.circuit.CircuitInstance(
                "source",
                phx.circuit.CircuitElement(
                    phx.circuit.IndependentCurrentSourceLaw(1e-10), element_id="source"
                ),
                ("0", "n"),
            ),
        ),
        (
            phx.circuit.NodalPort(
                "port", "n", "0", phx.circuit.ElectricalWaveReference(50.0)
            ),
        ),
        ground="0",
        circuit_id="semiconductor-rc-regression",
    )
    prepared = phx.circuit.prepare_circuit_dae(circuit)
    initial = prepared.initialize(
        node_voltages=jnp.zeros(1),
        auxiliary_state=law.initialize(plan.equilibrium_coordinates()),
    )
    point = sc.semiconductor_circuit_operating_point(
        prepared, initial, current_scale=1e-10
    )
    assert bool(point.nonlinear.successful)
    np.testing.assert_allclose(point.state[0], final_voltage, rtol=1e-8)
    assert abs(float(point.circuit_diagnostics.residual[0])) < 1e-20
    # A uniform conductor carries fixed intrinsic densities throughout charging.
    slope = 1e-10 / capacitance
    potential_rate = (1 - support.positions[:, 0] / LENGTH) * slope / plan.thermal_voltage
    coordinate_rate = jnp.stack(
        (potential_rate, -potential_rate, -potential_rate), axis=-1
    )
    initial_rate = prepared.initialize(
        node_voltages=jnp.array([slope]),
        auxiliary_state=law.initialize_rate(
            plan.equilibrium_coordinates(), coordinate_rate
        ),
    )
    times = jnp.linspace(0, time_constant, 11)
    transient = sc.semiconductor_circuit_transient(
        prepared, initial, initial_rate, times, current_scale=1e-10
    )
    assert bool(jnp.all(transient.solution.valid))
    assert bool(jnp.all(transient.solution.rate_valid))
    expected_voltage = final_voltage * (1 - jnp.exp(-times / time_constant))
    np.testing.assert_allclose(
        transient.solution.states[:, 0], expected_voltage, rtol=0.03, atol=2e-4
    )
    final_coordinates = law.coordinates(transient.solution.states[-1, 1:])
    electrons, holes = device.densities(final_coordinates)
    np.testing.assert_allclose(electrons / plan.intrinsic_density, 1, rtol=1e-6)
    np.testing.assert_allclose(holes / plan.intrinsic_density, 1, rtol=1e-6)


def test_depleted_junction_ac_and_implicit_response_match_biased_device_solves():
    device = sc.PreparedSemiconductorDevice(sc.pn_junction(21))
    volts = jnp.array([0.025, 0.0])
    point = device.solve(volts)
    assert bool(point.successful)
    ac = sc.semiconductor_small_signal(device, point, jnp.array([0.0, 1e6]))
    sensitivity = sc.semiconductor_sensitivity(device, point)
    assert bool(jnp.all(ac.evidence.successful))
    assert bool(jnp.all(sensitivity.evidence.successful))
    step = 1e-4
    perturbation = jnp.array([step, 0.0])
    upper = device.solve(volts + perturbation, initial=point)
    lower = device.solve(volts - perturbation, initial=point)
    assert bool(upper.successful & lower.successful)
    finite_difference = (upper.terminal_currents - lower.terminal_currents) / (2 * step)
    np.testing.assert_allclose(
        sensitivity.derivatives[:, 0], finite_difference, rtol=2e-3, atol=1e-19
    )
    np.testing.assert_allclose(
        ac.admittance[0].real, sensitivity.derivatives, rtol=2e-4, atol=1e-19
    )
    magnitude = np.max(np.abs(ac.admittance), axis=(1, 2))
    assert np.all(
        np.max(np.abs(np.sum(ac.admittance, axis=1)), axis=1) <= 1e-4 * magnitude
    )
    assert np.all(
        np.max(np.abs(np.sum(ac.admittance, axis=2)), axis=1) <= 1e-4 * magnitude
    )
