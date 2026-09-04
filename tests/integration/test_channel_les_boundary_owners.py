#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.incompressible_flow._boundary_turbulence import (
    StochasticTurbulentInflowMACBoundaryState,
    StochasticTurbulentInflowPlan,
    VectorEquilibriumWallStressPlan,
)
from phydrax.equations._channel_les import channel_les_filter, compile_channel_les
from phydrax.solver._channel_flow import CHANNEL_FLOW_EXPLICIT_RESTRICTION


def _space():
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(2),
            phx.discretization.ChebyshevBasisPlan(5),
            phx.discretization.FourierBasisPlan(2),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )


def _channel_les(
    *,
    tangential_boundary="velocity",
    pressure_gradient=(0.0, 0.0),
    route="ultraspherical_banded",
):
    space = _space()
    stokes = phx.discretization.ChannelStokesPlan(
        space,
        0.05,
        tangential_boundary=tangential_boundary,
        mean_constraint=phx.discretization.ChannelMeanConstraint(
            "pressure_gradient", pressure_gradient
        ),
        route=route,
    )
    base = phx.equations.compile_channel_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.05),
        stokes,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    provenance = phx.equations.LESParameterProvenance(
        channel_les_filter(space),
        space.prepared_id,
        "wall-resolved-channel",
        source_kind="user",
        evidence_ids=(),
    )
    model = phx.equations.WALELESPlan(0.15).prepare(provenance)
    return space, compile_channel_les(base, model)


def _parabolic_state(space, dynamics):
    wall_coordinate = space.axes[1].nodes[None, :, None]
    velocity = jnp.zeros(space.physical_shape + (3,))
    velocity = velocity.at[..., 0].set(
        jnp.broadcast_to(0.1 * (1.0 - wall_coordinate**2), space.physical_shape)
    )
    return dynamics.project_state(velocity)


@pytest.mark.parametrize("route", ("ultraspherical_banded", "dense_reference"))
def test_mixed_channel_stokes_enforces_traction_without_tangential_no_slip(route):
    space = _space()
    solver = phx.discretization.ChannelStokesPlan(
        space,
        0.05,
        tangential_boundary="traction",
        route=route,
    ).prepare(10.0)
    physical_shape = (space.physical_shape[0], space.physical_shape[2], 2)
    lower_physical = jnp.zeros(physical_shape).at[..., 0].set(0.02)
    upper_physical = jnp.zeros(physical_shape).at[..., 0].set(-0.01)
    solved = solver.solve(
        jnp.zeros(space.modal_shape + (3,), dtype=complex),
        lower_tangential_traction=solver.project_horizontal_boundary(lower_physical),
        upper_tangential_traction=solver.project_horizontal_boundary(upper_physical),
    )
    physical = space.reconstruct(solved.velocity)

    assert bool(solved.successful)
    assert float(solved.diagnostics.tangential_traction_residual) < 1.0e-9
    np.testing.assert_allclose(physical[:, 0, :, 1], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(physical[:, -1, :, 1], 0.0, atol=1.0e-9)
    assert float(jnp.max(jnp.abs(physical[:, (0, -1), :, 0]))) > 0.0


def test_wall_owned_channel_changes_trajectory_and_closes_boundary_work():
    space_off, off = _channel_les()
    space_on, on = _channel_les(tangential_boundary="traction")
    initial_off = _parabolic_state(space_off, off)
    initial_on = _parabolic_state(space_on, on)
    maximum_step = min(
        float(off.explicit_restriction(initial_off).maximum_step),
        float(on.explicit_restriction(initial_on).maximum_step),
    )
    step = 0.1 * maximum_step

    off_solution = phx.solver.solve_channel_sbdf2(
        off, initial_off, jnp.asarray((0.0, step, 2.0 * step))
    )
    wall_owner = VectorEquilibriumWallStressPlan().prepare_channel(
        on,
        step,
        density=1.0,
        sample_distance=(0.1, 0.1),
    )
    state = wall_owner.initialize(initial_on, 0.0, None)
    np.testing.assert_allclose(
        wall_owner.sample_coordinates,
        jnp.asarray((-0.9, 0.9)),
        atol=0.0,
        rtol=0.0,
    )
    expected_speed = 0.1 * (1.0 - 0.9**2)
    expected_wall = wall_owner.wall_stress.evaluate(
        jnp.asarray((expected_speed, 0.0, 0.0)),
        jnp.asarray((0.0, 1.0, 0.0)),
        0.1,
        1.0,
        0.05,
    )
    expected_traction = jnp.broadcast_to(
        expected_wall.traction,
        state.current_lower.traction.shape,
    )
    np.testing.assert_allclose(
        state.current_lower.traction,
        expected_traction,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.current_upper.traction,
        expected_traction,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert float(jnp.max(state.current_lower.wall_shear_magnitude)) > 0.0
    first = wall_owner.step(0, 0.0, state, step, None)
    second = wall_owner.step(1, step, first.accepted_state, step, None)

    assert bool(off_solution.successful)
    assert bool(first.successful)
    assert bool(second.successful), (
        second.evidence.wall_law_successful,
        second.evidence.boundary_identity_closed,
        second.evidence.dissipative,
        second.evidence.finite,
        second.evidence.stokes.failed,
        second.evidence.energy_boundary_work_defect,
        second.evidence.stokes.boundary_power,
    )
    assert not bool(
        jnp.allclose(
            off_solution.velocity[-1],
            second.accepted_state.channel.current_velocity,
        )
    )
    assert float(second.evidence.energy_boundary_work_defect) < 1.0e-8
    np.testing.assert_allclose(
        second.evidence.energy_ledger.wall_power,
        second.evidence.stokes.boundary_power,
        atol=1.0e-8,
        rtol=1.0e-8,
    )
    assert bool(second.evidence.dissipative)


def _spectral_mac_owner(variance):
    angles = 0.5 * jnp.pi * jnp.arange(4)
    coordinates = jnp.stack((jnp.zeros_like(angles), angles), axis=-1)
    return StochasticTurbulentInflowPlan("spectral").prepare_mac_boundary(
        coordinates,
        jnp.asarray((1.0, 0.0)),
        jnp.ones((4,)),
        jnp.asarray(((variance, 0.0), (0.0, 0.0))),
        axis="x",
        side="lower",
        boundary_shape=(4,),
        spectral_wavevectors=jnp.asarray(((0.0, 1.0),)),
    )


def test_mac_inflow_owner_commits_covariance_and_restarts_exactly():
    owner = _spectral_mac_owner(0.7)
    initial = owner.initialize(
        jax.random.key(19),
        0.0,
        mean_velocity=jnp.asarray((2.0, 0.0)),
    )
    first = owner.advance(
        initial.state,
        0.1,
        mean_velocity=jnp.asarray((2.0, 0.0)),
    )
    restored = StochasticTurbulentInflowMACBoundaryState(
        inflow_state=initial.state.inflow_state,
        velocity=initial.state.velocity,
        scalars=initial.state.scalars,
        time=initial.state.time,
        accepted_steps=initial.state.accepted_steps,
        prepared_id=initial.state.prepared_id,
    )
    replay = owner.advance(
        restored,
        0.1,
        mean_velocity=jnp.asarray((2.0, 0.0)),
    )
    lower_variance_owner = _spectral_mac_owner(0.2)
    lower_variance_initial = lower_variance_owner.initialize(
        jax.random.key(19),
        0.0,
        mean_velocity=jnp.asarray((2.0, 0.0)),
    )
    lower_variance = lower_variance_owner.advance(
        lower_variance_initial.state,
        0.1,
        mean_velocity=jnp.asarray((2.0, 0.0)),
    )

    assert first.boundary.kind == "velocity-inflow"
    assert first.provider.value.shape == (2, 4)
    np.testing.assert_array_equal(first.provider.value, replay.provider.value)
    np.testing.assert_array_equal(first.provider.rate, replay.provider.rate)
    np.testing.assert_array_equal(
        jax.random.key_data(first.state.inflow_state.key),
        jax.random.key_data(replay.state.inflow_state.key),
    )
    assert not bool(jnp.allclose(first.state.velocity, lower_variance.state.velocity))
    assert float(first.evidence.rate_closure_error) < 1.0e-12
    np.testing.assert_allclose(first.evidence.fluctuation_volume_flux, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        first.evidence.maximum_divergence_residual, 0.0, atol=1.0e-12
    )
    assert bool(first.evidence.successful)


def test_complete_channel_restriction_refuses_unsafe_step():
    space, dynamics = _channel_les()
    initial = _parabolic_state(space, dynamics)
    restriction = dynamics.explicit_restriction(initial)
    unsafe_step = 2.0 * float(restriction.maximum_step)
    solution = phx.solver.solve_channel_sbdf2(
        dynamics,
        initial,
        jnp.asarray((0.0, unsafe_step)),
    )

    assert restriction.temporal_method == "channel-sbdf2"
    assert float(restriction.advective_rate) > 0.0
    assert float(restriction.wall_normal_derivative_norm) > 0.0
    assert float(restriction.total_explicit_rate) >= float(restriction.diffusive_rate)
    assert not bool(restriction.permits(unsafe_step))
    assert int(solution.diagnostics.status[0]) == CHANNEL_FLOW_EXPLICIT_RESTRICTION
    np.testing.assert_array_equal(solution.velocity[0], solution.velocity[-1])
    assert not bool(solution.successful)


def test_unsupported_wall_pressure_gradient_and_open_inflow_refuse():
    _, velocity_owned = _channel_les()
    with pytest.raises(ValueError, match="traction-owned"):
        VectorEquilibriumWallStressPlan().prepare_channel(
            velocity_owned,
            1.0e-5,
            density=1.0,
            sample_distance=0.1,
        )
    with pytest.raises(ValueError, match="cannot also prescribe"):
        phx.discretization.ChannelStokesPlan(
            _space(),
            0.05,
            lower_wall_velocity=(1.0, 0.0, 0.0),
            tangential_boundary="traction",
        )

    _, pressure_driven = _channel_les(
        tangential_boundary="traction", pressure_gradient=(1.0, 0.0)
    )
    with pytest.raises(ValueError, match="zero prescribed pressure gradient"):
        VectorEquilibriumWallStressPlan().prepare_channel(
            pressure_driven,
            1.0e-5,
            density=1.0,
            sample_distance=0.1,
        )
    with pytest.raises(ValueError, match="accepted-step boundary owner"):
        phx.solver.solve_channel_sbdf2(
            pressure_driven,
            jnp.zeros(pressure_driven.state_shape, dtype=complex),
            jnp.asarray((0.0, 1.0e-5)),
        )

    coordinates = jnp.asarray(((0.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="represented-divergence compatible"):
        StochasticTurbulentInflowPlan(
            "compact", compact_support_radius=2.0
        ).prepare_mac_boundary(
            coordinates,
            jnp.asarray((1.0, 0.0)),
            jnp.ones((2,)),
            jnp.eye(2),
            axis="x",
            side="lower",
            boundary_shape=(2,),
        )
