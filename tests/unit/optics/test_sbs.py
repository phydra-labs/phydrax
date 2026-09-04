#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

from phydrax.optics.sbs._sbs import (
    SBSInteractionCoefficients,
    SBSOverlapPlan,
    SBSSharedDomainMap,
    SBSStatus,
    solve_sbs,
)


_EPSILON_0 = 8.8541878128e-12


def _domain(volume_weights, *, boundary_weights=None):
    weights = np.asarray(volume_weights, dtype=float)
    count = weights.size
    identity = np.eye(count)
    if boundary_weights is None:
        boundary = np.zeros((0,), dtype=float)
        to_boundary = np.zeros((0, count), dtype=float)
        normals = np.zeros((0, 2), dtype=float)
    else:
        boundary = np.asarray(boundary_weights, dtype=float)
        assert boundary.size == count
        to_boundary = identity
        normals = np.broadcast_to(np.asarray([0.0, 1.0]), (count, 2))
    return SBSSharedDomainMap(
        weights,
        identity,
        identity,
        identity,
        boundary,
        to_boundary,
        to_boundary,
        to_boundary,
        normals,
        normal_orientation="material-minus-to-plus",
        jump_convention="minus-minus-plus",
    )


def _interaction(*, acoustic_wavenumber=2.0):
    return SBSInteractionCoefficients(
        pump_angular_frequency=10.0,
        stokes_angular_frequency=8.0,
        acoustic_angular_frequency=2.0,
        pump_propagation_constant=5.0,
        stokes_propagation_constant=3.0,
        acoustic_wavenumber=acoustic_wavenumber,
        acoustic_quality_factor=100.0,
        acoustic_group_velocity=2.0,
        interaction_length=1.0,
    )


def _single_point_plan(*, photoelastic_sign=-1.0, pump=1.0, stokes=1.0, acoustic=1.0):
    tensor = np.zeros((2, 2, 2, 2), dtype=complex)
    tensor[0, 0, 0, 0] = photoelastic_sign
    strain = np.zeros((1, 2, 2), dtype=complex)
    strain[0, 0, 0] = acoustic
    return SBSOverlapPlan(
        _domain([1.0], boundary_weights=[1.0]),
        pump_electric=np.asarray([[pump, 0.0]], dtype=complex),
        pump_electric_displacement=np.zeros((1, 2), dtype=complex),
        stokes_electric=np.asarray([[stokes, 0.0]], dtype=complex),
        stokes_electric_displacement=np.zeros((1, 2), dtype=complex),
        acoustic_displacement=np.asarray([[0.0, acoustic]], dtype=complex),
        acoustic_strain=strain,
        relative_permittivity=1.0,
        photoelastic_tensor=tensor,
        relative_permittivity_jump=1.0,
        inverse_relative_permittivity_jump=0.0,
        pump_power=abs(pump) ** 2,
        stokes_power=abs(stokes) ** 2,
        acoustic_energy_per_length=abs(acoustic) ** 2,
    )


def test_pe_mb_cancellation_and_reinforcement_are_complex_before_magnitude():
    cancellation = solve_sbs(
        _single_point_plan(photoelastic_sign=1.0).prepare(), _interaction()
    )
    np.testing.assert_allclose(cancellation.Q_PE, -0.5 * _EPSILON_0)
    np.testing.assert_allclose(cancellation.Q_MB, 0.5 * _EPSILON_0)
    np.testing.assert_allclose(cancellation.Q_total, 0.0, atol=1e-25)
    np.testing.assert_allclose(cancellation.gain, 0.0, atol=1e-25)

    reinforcement = solve_sbs(
        _single_point_plan(photoelastic_sign=-1.0).prepare(), _interaction()
    )
    np.testing.assert_allclose(reinforcement.Q_PE, 0.5 * _EPSILON_0)
    np.testing.assert_allclose(reinforcement.Q_MB, 0.5 * _EPSILON_0)
    np.testing.assert_allclose(reinforcement.Q_total, _EPSILON_0)
    assert reinforcement.gain > 0.0
    assert int(reinforcement.status) == int(SBSStatus.SUCCESS)


def test_overlap_phase_covariance_and_gain_rescaling_invariance():
    baseline = solve_sbs(_single_point_plan().prepare(), _interaction())
    pump_scale = 2.0j
    stokes_scale = -3.0
    acoustic_scale = np.exp(0.4j) * 4.0
    scaled = solve_sbs(
        _single_point_plan(
            pump=pump_scale,
            stokes=stokes_scale,
            acoustic=acoustic_scale,
        ).prepare(),
        _interaction(),
    )
    covariance = pump_scale * np.conj(stokes_scale) * acoustic_scale
    np.testing.assert_allclose(scaled.Q_PE, covariance * baseline.Q_PE, rtol=1e-12)
    np.testing.assert_allclose(scaled.Q_MB, covariance * baseline.Q_MB, rtol=1e-12)
    np.testing.assert_allclose(scaled.Q_total, covariance * baseline.Q_total, rtol=1e-12)
    np.testing.assert_allclose(
        abs(scaled.normalized_Q_total), abs(baseline.normalized_Q_total), rtol=1e-12
    )
    np.testing.assert_allclose(scaled.gain, baseline.gain, rtol=1e-12)


def test_selection_rule_phase_matching_loss_and_units():
    tensor = np.zeros((2, 2, 2, 2))
    tensor[0, 0, 0, 0] = 1.0
    strain = np.zeros((2, 2, 2))
    strain[:, 0, 0] = 1.0
    odd_stokes = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    plan = SBSOverlapPlan(
        _domain([0.5, 0.5]),
        pump_electric=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        pump_electric_displacement=np.zeros((2, 2)),
        stokes_electric=odd_stokes,
        stokes_electric_displacement=np.zeros((2, 2)),
        acoustic_displacement=np.zeros((2, 2)),
        acoustic_strain=strain,
        relative_permittivity=1.0,
        photoelastic_tensor=tensor,
        relative_permittivity_jump=0.0,
        inverse_relative_permittivity_jump=0.0,
        pump_power=1.0,
        stokes_power=1.0,
        acoustic_energy_per_length=1.0,
    )
    forbidden = solve_sbs(plan.prepare(), _interaction())
    np.testing.assert_allclose(forbidden.Q_total, 0.0, atol=1e-25)

    matched = solve_sbs(_single_point_plan().prepare(), _interaction())
    mismatched = solve_sbs(
        _single_point_plan().prepare(),
        _interaction(acoustic_wavenumber=2.0 + 2.0 * np.pi),
    )
    np.testing.assert_allclose(matched.phase_mismatch, 0.0)
    np.testing.assert_allclose(matched.phase_matching_factor, 1.0)
    assert mismatched.phase_matching_factor < 1e-20
    np.testing.assert_allclose(matched.acoustic_linewidth, 0.02)
    np.testing.assert_allclose(matched.acoustic_power_attenuation, 0.01)
    np.testing.assert_allclose(matched.resonant_gain, 1600.0 * _EPSILON_0**2, rtol=1e-12)
    np.testing.assert_allclose(matched.gain, matched.resonant_gain, rtol=1e-12)
    assert matched.overlap_units == "J m^-1"
    assert matched.linewidth_units == "rad s^-1"
    assert matched.attenuation_units == "m^-1"
    assert matched.phase_mismatch_units == "rad m^-1"
    assert matched.gain_units == "W^-1 m^-1"


def _quadrature_result(count):
    points = np.linspace(0.0, 1.0, count)
    spacing = 1.0 / (count - 1)
    weights = np.full((count,), spacing)
    weights[[0, -1]] *= 0.5
    tensor = np.zeros((2, 2, 2, 2))
    tensor[0, 0, 0, 0] = 1.0
    strain = np.zeros((count, 2, 2))
    strain[:, 0, 0] = points**2
    plan = SBSOverlapPlan(
        _domain(weights),
        pump_electric=np.stack((np.ones_like(points), np.zeros_like(points)), axis=1),
        pump_electric_displacement=np.zeros((count, 2)),
        stokes_electric=np.stack((np.ones_like(points), np.zeros_like(points)), axis=1),
        stokes_electric_displacement=np.zeros((count, 2)),
        acoustic_displacement=np.zeros((count, 2)),
        acoustic_strain=strain,
        relative_permittivity=1.0,
        photoelastic_tensor=tensor,
        relative_permittivity_jump=0.0,
        inverse_relative_permittivity_jump=0.0,
        pump_power=1.0,
        stokes_power=1.0,
        acoustic_energy_per_length=1.0,
    )
    return solve_sbs(plan.prepare(), _interaction())


def test_shared_domain_quadrature_converges_to_analytic_overlap():
    exact = -0.5 * _EPSILON_0 / 3.0
    coarse = _quadrature_result(9)
    fine = _quadrature_result(33)
    assert abs(fine.Q_PE - exact) < abs(coarse.Q_PE - exact)
    np.testing.assert_allclose(fine.Q_PE, exact, rtol=1e-3)
