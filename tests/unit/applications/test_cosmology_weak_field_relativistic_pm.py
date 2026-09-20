from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.applications.cosmology._scales import CODE_COSMOLOGY_SCALE
from phydrax.applications.cosmology._weak_field_relativistic_pm import (
    WeakFieldRelativisticPMPlan,
    WeakFieldRelativisticPMPolicy,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.discretization.splatting import ParticleGridSplatPlan
from phydrax.metrix import ADMGridGeometry, RelativityConvention


def _units():
    convention = RelativityConvention.canonical()
    scale = RelativityScaleContract(CODE_COSMOLOGY_SCALE, 1, 1, 1, 1)
    return RelativisticUnitContract(scale, convention)


def _frame(spectral, units, *, time, token):
    shape = spectral.physical_shape
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    geometry = ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        snapshot_token=jnp.asarray(token, dtype=jnp.int32),
        chart_id="periodic-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id=spectral.grid.topology.topology_id,
        geometry_lineage_id="weak-field-pm-lineage",
    )
    xyz = spectral.grid.points.reshape(shape + (3,))
    coordinates = jnp.concatenate((jnp.full(shape + (1,), time), xyz), axis=-1)
    return LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        coordinates,
        jnp.asarray(time),
        jnp.asarray(1.0),
        observer_id="eulerian-observer",
        orientation_id="cartesian-right-handed",
    )


def _lattice_case(
    *,
    count=4,
    weights=None,
    momenta=None,
    gravitational_constant=1.0e-4,
    scalar_only=False,
    policy=None,
):
    units = _units()
    spectral = TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="weak-field-metric",
    ).prepare(tuple(AxisDomain.periodic(0.0, 1.0) for _ in range(3)))
    positions = spectral.grid.points
    capacity = positions.shape[0]
    particles = ParticleSetPlan(
        jnp.arange(capacity),
        jnp.ones((capacity,)),
        ambient_dimension=3,
    ).prepare()
    transfer = ParticleGridSplatPlan(spectral.grid).prepare(particles)
    stress = RelativisticStressDepositPlan(
        transfer,
        units,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([1.0]),
        mass_shell_relative_tolerance=1.0e-5,
        conservation_tolerance=1.0e-5,
        frame_momentum_relative_tolerance=1.0e-5,
    )
    selected_policy = (
        WeakFieldRelativisticPMPolicy(
            maximum_scalar_metric_fraction=0.2,
            maximum_vector_metric_fraction=0.2,
            maximum_tensor_metric_norm=0.2,
            maximum_cell_crossing=0.75,
            constraint_relative_tolerance=2.0e-4,
            gauge_absolute_tolerance=2.0e-4,
            force_relative_tolerance=2.0e-4,
            conservation_relative_tolerance=2.0e-3,
            omitted_channel_tolerance=0.0,
        )
        if policy is None
        else policy
    )
    plan = WeakFieldRelativisticPMPlan(
        stress,
        spectral,
        units,
        gravitational_constant=gravitational_constant,
        scalar_only=scalar_only,
        policy=selected_policy,
    )
    frame = _frame(spectral, units, time=0.0, token=1)
    weight = jnp.ones((capacity,)) if weights is None else jnp.asarray(weights)
    momentum = jnp.zeros((capacity, 3)) if momenta is None else jnp.asarray(momenta)
    state = stress.initialize(
        weight,
        positions,
        momentum,
        jnp.ones((capacity,), dtype=jnp.int32),
        frame,
    )
    return plan, frame, state, spectral


def test_uniform_source_has_zero_mean_modes():
    plan, frame, state, _ = _lattice_case()
    result = plan.solve_stress(state, frame)

    assert bool(result.successful)
    np.testing.assert_allclose(result.metric.phi, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.metric.psi, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.metric.shift_vector, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.metric.tensor_metric, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.acceleration, 0.0, atol=2e-6)
    assert result.scalar_residual < 2e-5
    assert result.vector_residual < 2e-5
    assert result.tensor_residual < 2e-5


def test_cold_plane_wave_recovers_the_newtonian_poisson_limit():
    count = 4
    particle_count = count**3
    x = jnp.repeat(jnp.arange(count) / count, count * count)
    amplitude = 0.1
    weights = 1.0 + amplitude * jnp.cos(2.0 * jnp.pi * x)
    plan, frame, state, _ = _lattice_case(
        count=count,
        weights=weights,
        momenta=jnp.zeros((particle_count, 3)),
    )
    result = plan.solve_stress(state, frame)
    wave_number = 2.0 * jnp.pi
    expected = (
        -4.0
        * jnp.pi
        * plan.gravitational_constant
        * particle_count
        * amplitude
        * jnp.cos(wave_number * x)
        / wave_number**2
    ).reshape((count, count, count))

    assert bool(result.successful)
    np.testing.assert_allclose(result.metric.psi, expected, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(result.metric.phi, expected, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(result.metric.shift_vector, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.metric.tensor_metric, 0.0, atol=2e-7)


def test_plane_wave_scalar_source_solves_poisson_and_anisotropic_stress_slip():
    count = 5
    x = jnp.repeat(jnp.arange(count) / count, count * count)
    modulation = 1.0 + 0.2 * jnp.cos(2.0 * jnp.pi * x)
    momenta = jnp.zeros((count**3, 3)).at[:, 0].set(0.35 * jnp.cos(2.0 * jnp.pi * x))
    plan, frame, state, spectral = _lattice_case(
        count=count, weights=modulation, momenta=momenta
    )
    result = plan.solve_stress(state, frame)
    c = float(plan.units.speed_of_light)
    density = result.stress.energy_density
    mean = spectral.integral(density) / spectral.integral(jnp.ones_like(density))
    source = 4.0 * jnp.pi * plan.gravitational_constant * (density - mean) / c**2

    assert bool(result.successful)
    np.testing.assert_allclose(
        spectral.laplacian(result.metric.psi), source, rtol=3e-4, atol=3e-4
    )
    assert jnp.max(jnp.abs(result.metric.phi - result.metric.psi)) > 1.0e-8
    assert result.scalar_residual < 2e-4
    assert result.spectral_support_defect < 2e-4
    assert jnp.max(jnp.abs(result.stress_evidence.anisotropic_stress)) > 0.0


def test_transverse_vector_and_tt_tensor_projections_close_gauge_constraints():
    count = 5
    x = jnp.repeat(jnp.arange(count) / count, count * count)
    momenta = jnp.zeros((count**3, 3)).at[:, 1].set(0.45 * jnp.cos(2.0 * jnp.pi * x))
    plan, frame, state, spectral = _lattice_case(count=count, momenta=momenta)
    result = plan.solve_stress(state, frame)
    divergence_b = spectral.divergence(result.metric.shift_vector)
    trace_h = jnp.trace(result.metric.tensor_metric, axis1=-2, axis2=-1)
    divergence_h = jnp.stack(
        tuple(
            spectral.divergence(result.metric.tensor_metric[..., :, component])
            for component in range(3)
        ),
        axis=-1,
    )

    assert bool(result.successful)
    assert jnp.max(jnp.abs(result.metric.shift_vector)) > 0.0
    assert jnp.max(jnp.abs(result.metric.tensor_metric)) > 0.0
    np.testing.assert_allclose(divergence_b, 0.0, atol=3e-5)
    np.testing.assert_allclose(trace_h, 0.0, atol=3e-5)
    np.testing.assert_allclose(divergence_h, 0.0, atol=3e-5)
    assert result.vector_residual < 2e-4
    assert result.tensor_residual < 2e-4
    assert result.spectral_support_defect < 2e-4


def test_unrepresentable_tensor_nyquist_source_is_reported_and_refused():
    count = 4
    x = jnp.repeat(jnp.arange(count) / count, count * count)
    momenta = jnp.zeros((count**3, 3)).at[:, 0].set(0.35 * jnp.cos(2.0 * jnp.pi * x))
    plan, frame, state, _ = _lattice_case(count=count, momenta=momenta)
    result = plan.solve_stress(state, frame)

    assert result.spectral_support_defect > 0.0
    assert not bool(result.successful)


def test_scalar_only_profile_refuses_unbounded_omitted_vector_tensor_channels():
    count = 5
    x = jnp.repeat(jnp.arange(count) / count, count * count)
    momenta = jnp.zeros((count**3, 3)).at[:, 1].set(0.5 * jnp.cos(2.0 * jnp.pi * x))
    plan, frame, state, _ = _lattice_case(count=count, momenta=momenta, scalar_only=True)
    result = plan.solve_stress(state, frame)

    assert result.omitted_channel_bound > 0.0
    assert result.spectral_support_defect < 2e-4
    assert not bool(result.successful)
    np.testing.assert_allclose(result.metric.shift_vector, 0.0, atol=0.0)
    np.testing.assert_allclose(result.metric.tensor_metric, 0.0, atol=0.0)


def test_relativistic_geodesic_step_and_resource_evidence():
    count = 3
    capacity = count**3
    momenta = jnp.broadcast_to(jnp.asarray([0.6, 0.0, 0.0]), (capacity, 3))
    permissive = WeakFieldRelativisticPMPolicy(
        maximum_scalar_metric_fraction=0.2,
        maximum_vector_metric_fraction=0.2,
        maximum_tensor_metric_norm=0.2,
        maximum_cell_crossing=0.75,
        constraint_relative_tolerance=5e-4,
        gauge_absolute_tolerance=5e-4,
        force_relative_tolerance=5e-4,
        conservation_relative_tolerance=5e-3,
    )
    plan, start_frame, state, spectral = _lattice_case(
        count=count,
        momenta=momenta,
        gravitational_constant=1.0e-10,
        policy=permissive,
    )
    endpoint_frame = _frame(spectral, plan.units, time=0.01, token=2)
    result = plan.step(state, start_frame, endpoint_frame)
    expected_speed = 0.6 / np.sqrt(1.0 + 0.6**2)

    assert bool(result.successful)
    np.testing.assert_allclose(
        jnp.mod(result.state.positions[:, 0] - state.positions[:, 0], 1.0),
        0.01 * expected_speed,
        rtol=3e-4,
        atol=3e-5,
    )
    assert result.diagnostics.momentum_conservation_defect < 5e-5
    assert int(result.diagnostics.status) == 0
    assert plan.grid_points == count**3
    assert plan.workspace_bytes > 0


def test_static_resource_gate_refuses_oversized_grid_before_execution():
    plan, _, _, spectral = _lattice_case(count=3)
    policy = WeakFieldRelativisticPMPolicy(maximum_grid_points=1)

    with pytest.raises(ValueError, match="maximum_grid_points"):
        WeakFieldRelativisticPMPlan(
            plan.stress_transfer,
            spectral,
            plan.units,
            gravitational_constant=plan.gravitational_constant,
            policy=policy,
        )


def test_failed_time_step_rolls_back_particle_and_metric_state_atomically():
    count = 3
    capacity = count**3
    momenta = jnp.broadcast_to(jnp.asarray([0.8, 0.0, 0.0]), (capacity, 3))
    restrictive = WeakFieldRelativisticPMPolicy(
        maximum_scalar_metric_fraction=0.2,
        maximum_vector_metric_fraction=0.2,
        maximum_tensor_metric_norm=0.2,
        maximum_cell_crossing=1.0e-6,
        constraint_relative_tolerance=5e-4,
        gauge_absolute_tolerance=5e-4,
        force_relative_tolerance=5e-4,
        conservation_relative_tolerance=1.0,
    )
    plan, start_frame, state, spectral = _lattice_case(
        count=count,
        momenta=momenta,
        gravitational_constant=1.0e-10,
        policy=restrictive,
    )
    endpoint_frame = _frame(spectral, plan.units, time=0.01, token=2)
    initial = plan.solve_stress(state, start_frame)
    result = plan.step(state, start_frame, endpoint_frame)

    assert not bool(result.successful)
    assert bool(result.diagnostics.rolled_back)
    np.testing.assert_array_equal(result.state.positions, state.positions)
    np.testing.assert_array_equal(result.state.covariant_momenta, state.covariant_momenta)
    np.testing.assert_array_equal(result.state.frame_token, state.frame_token)
    np.testing.assert_array_equal(result.metric.phi, initial.metric.phi)
    np.testing.assert_array_equal(result.metric.psi, initial.metric.psi)
