import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._wave_dark_matter import WaveDarkMatterPlan
from phydrax.applications.cosmology._wave_finite_difference import (
    PeriodicWaveFiniteDifferencePlan,
    WaveContactSelfInteractionPlan,
    WaveFiniteDifferencePolicy,
)


def _grid(count):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _spectral(count):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="psi",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))


def _plane_wave(grid, mode=2):
    coordinate = grid.points[:, 0].reshape(grid.shape)
    return jnp.exp(2j * jnp.pi * mode * coordinate)


def test_periodic_fd_plane_wave_converges_to_spectral_kinetic_authority():
    errors = []
    action = 2.0e-4
    for count in (24, 48):
        grid = _grid(count)
        prepared = PeriodicWaveFiniteDifferencePlan(1.0).prepare(grid)
        initial = _plane_wave(grid)
        state = prepared.initialize(initial)
        drifted, solved, residual, _ = prepared.kinetic_drift(
            state,
            2.0 * action,
            end_coordinate_time=2.0 * action,
        )
        spectral = initial * jnp.exp(-1j * action * (4.0 * jnp.pi) ** 2)
        errors.append(float(jnp.max(jnp.abs(drifted.psi - spectral))))
        assert bool(solved.successful)
        assert float(residual) < 1.0e-10

    assert errors[1] < 0.3 * errors[0]


def test_cayley_action_closes_weighted_norm_and_self_adjoint_residual():
    grid = _grid(32)
    prepared = PeriodicWaveFiniteDifferencePlan(1.0).prepare(grid)
    coordinate = grid.points[:, 0].reshape(grid.shape)
    psi = (1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * coordinate)) * jnp.exp(
        4j * jnp.pi * coordinate
    )
    state = prepared.initialize(psi)
    result = prepared.step(
        state,
        jnp.zeros(grid.shape),
        1.0e-4,
        1.0e-4,
        end_coordinate_time=1.0e-4,
    )

    assert bool(result.successful)
    assert result.kinetic_solve.provenance.method == "gmres"
    assert float(result.diagnostics.cayley_relative_residual) < 1.0e-10
    assert float(result.diagnostics.self_adjoint_residual) < 1.0e-11
    assert float(result.diagnostics.norm_relative_error) < 1.0e-9


def test_cayley_solve_honors_configured_iteration_cap_and_rolls_back():
    grid = _grid(24)
    prepared = PeriodicWaveFiniteDifferencePlan(
        1.0,
        policy=WaveFiniteDifferencePolicy(
            solve_relative_tolerance=0.0,
            solve_absolute_tolerance=0.0,
            maximum_solve_steps=1,
        ),
    ).prepare(grid)
    coordinate = grid.points[:, 0].reshape(grid.shape)
    psi = (
        jnp.exp(2j * jnp.pi * coordinate)
        + 0.4 * jnp.exp(6j * jnp.pi * coordinate)
        + 0.2 * jnp.exp(10j * jnp.pi * coordinate)
    )
    state = prepared.initialize(psi)

    result = prepared.step(
        state,
        jnp.zeros(grid.shape),
        0.1,
        0.0,
        end_coordinate_time=0.1,
    )

    assert prepared.solve_policy.tolerance.max_steps == 1
    assert not bool(result.kinetic_solve.successful)
    assert int(result.kinetic_solve.diagnostics.iterations) <= 1
    assert not bool(result.successful)
    np.testing.assert_array_equal(result.state.psi, state.psi)


def test_nonzero_fd_action_requires_advancing_time_and_obeys_phase_gate():
    grid = _grid(24)
    coordinate = grid.points[:, 0].reshape(grid.shape)
    state_psi = jnp.exp(6j * jnp.pi * coordinate)
    prepared = PeriodicWaveFiniteDifferencePlan(
        1.0,
        policy=WaveFiniteDifferencePolicy(maximum_phase_radians=1.0e-6),
    ).prepare(grid)
    state = prepared.initialize(state_psi)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="explicit end_coordinate_time",
    ):
        prepared.kinetic_drift(state, 1.0e-3)

    result = prepared.step(
        state,
        jnp.zeros(grid.shape),
        1.0e-3,
        0.0,
        end_coordinate_time=1.0e-3,
    )
    assert not bool(result.successful)
    assert int(result.diagnostics.status) == 6
    assert result.diagnostics.maximum_kinetic_phase > 1.0e-6
    np.testing.assert_array_equal(result.state.psi, state.psi)


def test_contact_action_is_symmetric_around_cayley_drift():
    grid = _grid(24)
    prepared = PeriodicWaveFiniteDifferencePlan(
        1.0,
        policy=WaveFiniteDifferencePolicy(maximum_phase_radians=2.0),
        contact=WaveContactSelfInteractionPlan(
            0.2,
            maximum_dealiasing_defect=1.0,
            energy_relative_tolerance=1.0,
            maximum_phase_radians=2.0,
        ),
    ).prepare(grid)
    coordinate = grid.points[:, 0].reshape(grid.shape)
    psi = (1.0 + 0.1 * jnp.cos(2.0 * jnp.pi * coordinate)).astype(jnp.complex128)
    state = prepared.initialize(psi)
    result = prepared.step(
        state,
        jnp.zeros(grid.shape),
        1.0e-4,
        0.0,
        end_coordinate_time=1.0e-4,
        contact_action_factor=1.0e-3,
    )

    first = prepared.contact_kick(state, 1.0e-3, fraction=0.5)
    drifted, _, _, _ = prepared.kinetic_drift(
        first.candidate_state,
        1.0e-4,
        end_coordinate_time=1.0e-4,
    )
    second = prepared.contact_kick(drifted, 1.0e-3, fraction=0.5)
    unsplit_drift, _, _, _ = prepared.kinetic_drift(
        state,
        1.0e-4,
        end_coordinate_time=1.0e-4,
    )
    unsplit = prepared.contact_kick(unsplit_drift, 1.0e-3)

    np.testing.assert_allclose(
        result.candidate_state.psi,
        second.candidate_state.psi,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert (
        jnp.max(jnp.abs(result.candidate_state.psi - unsplit.candidate_state.psi))
        > 1.0e-12
    )


def test_contact_alias_gate_uses_pretruncated_wave_before_density_product():
    grid = _grid(24)
    prepared = PeriodicWaveFiniteDifferencePlan(
        1.0,
        contact=WaveContactSelfInteractionPlan(
            0.2,
            maximum_dealiasing_defect=1.0e-3,
            energy_relative_tolerance=1.0,
            maximum_phase_radians=2.0,
        ),
    ).prepare(grid)
    coordinate = grid.points[:, 0].reshape(grid.shape)
    psi = (1.0 + 0.25 * jnp.cos(10.0 * jnp.pi * coordinate)).astype(jnp.complex128)
    action = prepared.contact_kick(prepared.initialize(psi), 1.0e-3)

    assert float(action.input_truncation_defect) < 1.0e-4
    assert float(action.aliasing_defect) > 1.0e-3
    assert not bool(action.successful)


def test_contact_interaction_has_distinct_identity_and_transactional_gates():
    grid = _grid(24)
    accepted = PeriodicWaveFiniteDifferencePlan(
        1.0,
        contact=WaveContactSelfInteractionPlan(
            0.2,
            maximum_dealiasing_defect=1.0e-12,
        ),
    ).prepare(grid)
    uniform = accepted.initialize(jnp.ones(grid.shape, dtype=jnp.complex128))
    result = accepted.step(
        uniform,
        jnp.zeros(grid.shape),
        1.0e-5,
        1.0e-5,
        end_coordinate_time=1.0e-5,
        contact_action_factor=1.0e-3,
    )
    assert bool(result.successful)
    assert accepted.plan.contact.plan_id != accepted.plan.plan_id
    np.testing.assert_allclose(
        result.diagnostics.contact_dealiasing_defect, 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        result.diagnostics.contact_input_truncation_defect,
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.diagnostics.contact_aliasing_defect,
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.diagnostics.contact_energy_relative_error, 0.0, atol=1.0e-12
    )

    rejected = PeriodicWaveFiniteDifferencePlan(
        1.0,
        contact=WaveContactSelfInteractionPlan(
            0.2,
            maximum_dealiasing_defect=1.0e-6,
        ),
    ).prepare(grid)
    checkerboard = 1.0 + 0.25 * (-1.0) ** jnp.arange(grid.shape[0])
    initial = rejected.initialize(checkerboard.astype(jnp.complex128))
    failed = rejected.step(
        initial,
        jnp.zeros(grid.shape),
        1.0e-5,
        1.0e-5,
        end_coordinate_time=1.0e-5,
        contact_action_factor=1.0e-3,
    )
    assert not bool(failed.successful)
    assert (
        failed.diagnostics.contact_input_truncation_defect > 1.0e-6
        or failed.diagnostics.contact_aliasing_defect > 1.0e-6
    )
    assert int(failed.diagnostics.status) == 7
    np.testing.assert_array_equal(failed.state.psi, initial.psi)


def test_spectral_external_actions_obey_endpoint_time_level_contract():
    space = _spectral(12)
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    prepared = WaveDarkMatterPlan(
        2.0,
        (1.0, 1.001),
        reduced_planck_constant=0.5,
    ).prepare(space, background)
    state = prepared.initialize(jnp.ones(space.physical_shape, dtype=jnp.complex128))
    potential = jnp.full(space.physical_shape, 0.25)

    first = prepared.potential_kick(state, potential, 1.0, 1.001, 0.5)
    drifted = prepared.kinetic_drift(first, 1.0, 1.001)
    second = prepared.potential_kick(drifted, potential, 1.0, 1.001, 0.5)

    np.testing.assert_allclose(prepared.density(state), 2.0)
    np.testing.assert_allclose(first.scale_factor, 1.0)
    np.testing.assert_allclose(drifted.scale_factor, 1.001)
    np.testing.assert_allclose(second.scale_factor, 1.001)
    np.testing.assert_allclose(jnp.abs(second.psi), 1.0, rtol=1.0e-12, atol=1.0e-12)
