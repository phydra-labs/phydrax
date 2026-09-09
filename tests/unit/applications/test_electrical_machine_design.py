#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.electrical_machines import (
    MachineAngleStudy,
    optimize_machine_design,
    PlanarMachine,
    polar_machine,
    polar_machine_study,
    scan_machine_angles,
    solve_planar_machine,
)
from phydrax.discretization import CellMesh
from phydrax.optim import Bounds, OptimizationTermination


jax.config.update("jax_enable_x64", True)


def _small_machine(angle=0.23, **options):
    return polar_machine(
        angle,
        sectors=16,
        rotor_layers=1,
        airgap_layers=2,
        winding_layers=1,
        stator_layers=1,
        **options,
    )


def _rebuild(machine, mesh, **overrides):
    arguments = dict(
        rotation_weights=machine.rotation_weights,
        radial_velocity=machine.radial_velocity,
        airgap_cells=machine.airgap_cells,
        reference_radius=machine.reference_radius,
        radius_bounds=machine.radius_bounds,
        reference_angle=machine.reference_angle,
        angle_window=machine.angle_window,
        axial_length=machine.axial_length,
    )
    arguments.update(overrides)
    return PlanarMachine(mesh, machine.cell_regions, machine.regions, **arguments)


def test_field_residual_gauge_and_constitutive_energy():
    machine = _small_machine()
    currents = jnp.asarray((-5.0, 2.0))
    result = solve_planar_machine(machine, currents)
    shifted = solve_planar_machine(machine, currents, boundary_potential=3e-4)
    assert bool(result.accepted) and bool(shifted.accepted)
    assert float(result.relative_residual) < 1e-8
    np.testing.assert_array_equal(result.potential[machine.boundary_nodes], 0.0)
    np.testing.assert_allclose(
        shifted.potential - result.potential, 3e-4, rtol=1e-8, atol=1e-11
    )
    np.testing.assert_allclose(
        shifted.magnetic_field,
        result.magnetic_field,
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        shifted.flux_linkage,
        result.flux_linkage,
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        result.coenergy + result.energy,
        currents @ result.flux_linkage,
        rtol=1e-11,
        atol=1e-11,
    )
    assert float(result.energy) > 0.0


def test_virtual_work_matches_resolved_energy_and_air_maxwell_stress():
    machine = _small_machine(salient=True)
    currents = jnp.asarray((-6.0, 1.0))
    step = 2e-6
    result = solve_planar_machine(machine, currents)
    plus = solve_planar_machine(machine, currents, angle_delta=step)
    minus = solve_planar_machine(machine, currents, angle_delta=-step)
    finite_difference = (plus.coenergy - minus.coenergy) / (2 * step)
    assert abs(float(result.torque)) > 1e-4
    np.testing.assert_allclose(result.torque, finite_difference, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(result.stress_torque, result.torque, rtol=1e-7, atol=1e-9)
    assert bool(result.accepted)


def test_reluctance_torque_is_even_in_current_not_forced_odd():
    machine = _small_machine(salient=True, remanence=0.0)
    positive = solve_planar_machine(machine, (-8.0, 0.0))
    negative = solve_planar_machine(machine, (8.0, 0.0))
    assert abs(float(positive.torque)) > 1e-5
    np.testing.assert_allclose(
        negative.magnetic_field,
        -positive.magnetic_field,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(negative.torque, positive.torque, rtol=1e-10, atol=1e-11)
    zero = solve_planar_machine(machine, (0.0, 0.0))
    np.testing.assert_array_equal(zero.magnetic_field, 0.0)
    assert float(zero.energy) == 0.0 and float(zero.torque) == 0.0


def test_pm_current_odd_torque_matches_circular_magnet_reference():
    # For mu=mu0 everywhere, a uniformly magnetized disk sees a uniform coil
    # field. This independently predicts the PM-current interaction torque;
    # taking its current-odd part cancels finite-mesh magnet self torque.
    angle = 0.21
    machine = polar_machine(
        angle,
        sectors=32,
        rotor_layers=3,
        airgap_layers=4,
        winding_layers=4,
        stator_layers=4,
        rotor_relative_permeability=1.0,
        stator_relative_permeability=1.0,
    )
    positive = solve_planar_machine(machine, (-4.0, 0.0))
    negative = solve_planar_machine(machine, (4.0, 0.0))
    odd_torque = (positive.torque - negative.torque) / 2
    odd_contour = (positive.contour_torque - negative.contour_torque) / 2
    (
        a,
        b,
        outer,
        radius,
        length,
        br,
        turns,
        current,
    ) = (0.04, 0.05, 0.065, 0.03, 0.1, 0.8, 100.0, 4.0)
    field_per_mu = (
        0.5 * turns / (b * b - a * a) * ((b - a) - (b**3 - a**3) / (3 * outer * outer))
    )
    expected = (
        length * np.pi * radius * radius * br * current * field_per_mu * np.cos(angle)
    )
    assert float(odd_torque) > 0.0
    np.testing.assert_allclose(odd_torque, expected, rtol=0.035)
    np.testing.assert_allclose(odd_contour, expected, rtol=0.07)


def test_angle_topology_survives_full_revolutions_and_radius_bounds():
    angle = 0.19
    first = _small_machine(angle)
    revolved = _small_machine(angle + 4 * np.pi)
    reference = solve_planar_machine(first, (-5.0, 1.0))
    periodic = solve_planar_machine(revolved, (-5.0, 1.0))
    shifted_sector = _small_machine(angle + 0.5 * np.pi, salient=True)
    assert (
        shifted_sector.discretization.mesh.topology_id
        == first.discretization.mesh.topology_id
    )
    np.testing.assert_array_equal(
        shifted_sector.discretization.mesh.blocks[0].vertices,
        first.discretization.mesh.blocks[0].vertices,
    )
    np.testing.assert_allclose(periodic.torque, reference.torque, rtol=1e-9, atol=1e-10)
    for radius in first.radius_bounds:
        for delta in (-first.angle_window, first.angle_window):
            result = solve_planar_machine(
                first,
                (-5.0, 1.0),
                design=(radius, 1.0, 1.0),
                angle_delta=delta,
            )
            assert bool(result.accepted)
    study = polar_machine_study(
        (angle, angle + 0.5 * np.pi),
        ((-5.0, 0.0), (0.0, -5.0)),
        weights=(1.0, 3.0),
        sectors=16,
        rotor_layers=1,
        airgap_layers=2,
        winding_layers=1,
        stator_layers=1,
    )
    assert len({model.discretization.mesh.topology_id for model in study.machines}) == 1
    scan = scan_machine_angles(study)
    expected_mean = (scan.torques[0] + 3 * scan.torques[1]) / 4
    expected_ripple = np.sqrt(
        np.sum(
            np.asarray(study.weights) * (np.asarray(scan.torques) - expected_mean) ** 2
        )
    )
    np.testing.assert_allclose(scan.average_torque, expected_mean, atol=1e-12)
    np.testing.assert_allclose(scan.rms_torque_ripple, expected_ripple, atol=1e-12)
    np.testing.assert_allclose(
        scan.peak_to_peak_torque_ripple,
        abs(scan.torques[1] - scan.torques[0]),
        atol=1e-12,
    )
    assert bool(scan.accepted)


def test_implicit_torque_design_derivative_includes_field_response():
    machine = _small_machine()
    design = jnp.asarray((0.029, 0.9, 1.1))
    direction = jnp.asarray((0.001, 0.1, -0.07))

    def torque(parameters):
        return solve_planar_machine(machine, (-7.0, 1.0), design=parameters).torque

    _, tangent = jax.jvp(torque, (design,), (direction,))
    step = 2e-4
    finite_difference = (
        torque(design + step * direction) - torque(design - step * direction)
    ) / (2 * step)
    assert abs(float(tangent)) > 1e-5
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-4, atol=1e-7)


def test_native_design_improves_physical_torque_with_fresh_final_fields():
    study = polar_machine_study(
        (0.2,),
        (-10.0, 0.0),
        sectors=16,
        rotor_layers=1,
        airgap_layers=2,
        winding_layers=1,
        stator_layers=1,
        rotor_relative_permeability=1.0,
        stator_relative_permeability=1.0,
    )
    initial = jnp.asarray((0.028, 0.8, 0.8))
    bounds = Bounds(
        jnp.asarray((0.026, 0.6, 0.6)),
        jnp.asarray((0.032, 1.2, 1.2)),
    )
    baseline = scan_machine_angles(study, initial)
    result = optimize_machine_design(
        study,
        initial,
        bounds,
        termination=OptimizationTermination(maximum_steps=16, absolute_optimality=1e-5),
    )
    assert bool(bounds.contains(result.design))
    assert bool(result.final_evaluation.accepted)
    assert float(result.final_evaluation.average_torque) > 1.1 * float(
        baseline.average_torque
    )
    assert float(result.final_evaluation.fields[0].relative_residual) < 1e-8


def test_rejects_inverted_airgap_source_and_out_of_domain_geometry():
    machine = _small_machine()
    mesh = machine.discretization.mesh
    cells = np.array(mesh.blocks[0].vertices)
    reflected = np.asarray(mesh.coordinates) * np.asarray((1.0, -1.0))
    inverted = CellMesh.from_triangles(reflected, cells)
    with pytest.raises(ValueError, match="counterclockwise"):
        _rebuild(machine, inverted)
    bad_contour_cells = np.asarray(machine.contour_cells).copy()
    bad_contour_cells[0, 1] = bad_contour_cells[0, 0]
    with pytest.raises(ValueError, match="distinct"):
        _rebuild(
            machine,
            mesh,
            contour_edges=machine.contour_edges,
            contour_cells=bad_contour_cells,
        )
    different_topology = polar_machine(
        0.3,
        sectors=32,
        rotor_layers=1,
        airgap_layers=2,
        winding_layers=1,
        stator_layers=1,
    )
    with pytest.raises(ValueError, match="topology"):
        MachineAngleStudy((machine, different_topology), ((-5.0, 1.0),) * 2)
    with pytest.raises(ValueError, match="source-free"):
        _rebuild(machine, mesh, airgap_cells=np.ones(len(cells), dtype=bool))
    with pytest.raises(ValueError, match="airgap"):
        polar_machine(rotor_radius_bounds=(0.024, 0.041))
    with pytest.raises(ValueError, match="too thin"):
        polar_machine(rotor_radius_bounds=(0.024, 0.03999))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="radius"):
        machine.coordinates(0.05)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="Angle"):
        machine.coordinates(machine.reference_radius, 3.0)
