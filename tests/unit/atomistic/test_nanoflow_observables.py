#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _dynamics():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    cell = phx.discretization.PeriodicCell(4.0 * jnp.eye(3))
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1, 2],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 0],
        cell=cell,
    ).prepare()
    base = phx.discretization.MetricCellListParticleNeighborhoodPlan(1.6, 3, 3, cell)
    neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
        base, 1.4, 0.2
    ).prepare(system.particles)
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [0.8], 1.4, switch_distance=1.2)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.BAOABLangevinPlan(2.0e-4, 0.2),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble="nvt",
        temperature=1.0,
    ).prepare(dynamics)
    positions = cell.cartesian(
        jnp.asarray(((0.1, 0.1, 0.1), (0.35, 0.1, 0.1), (0.6, 0.1, 0.1)))
    )
    initial = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(99),
    )
    return dynamics, thermodynamic, initial


def test_rollout_observers_update_only_accepted_steps_with_final_retention():
    dynamics, thermodynamic, initial = _dynamics()
    frame = phx.geometry.PlanarWallFramePlan(
        jnp.zeros((3,)),
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        gap=4.0,
        cross_section_area=16.0,
        length_unit_id=dynamics.system.plan.units.scale.length_unit.unit_id,
        lower_wall_id="lower",
        upper_wall_id="upper",
    )
    profile = phx.atomistic.PlanarWallProfileObserverPlan(
        frame,
        jnp.ones((1, 3), dtype="bool"),
        ("fluid",),
        bin_count=1,
        minimum_count_per_bin=3,
    )
    correlation = phx.atomistic.MultiOriginCorrelationObserverPlan(
        origin_stride=1,
        origin_capacity=3,
        lag_count=2,
        minimum_origins=2,
    )
    result = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        thermodynamic,
        phx.atomistic.AtomisticTrajectoryPlan(5, retention="final"),
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
        observers=(profile, correlation),
    ).rollout(initial)

    assert bool(result.successful)
    assert result.trajectory.times.shape == (1,)
    assert len(result.observations) == 2
    profile_result, correlation_result = result.observations
    assert int(profile_result.samples) == 5
    np.testing.assert_allclose(profile_result.counts, ((15.0,),))
    assert bool(profile_result.header.globally_eligible)
    np.testing.assert_array_equal(correlation_result.counts, (5, 4))
    assert bool(correlation_result.header.globally_eligible)

    state = profile.initialize(dynamics, initial)
    unchanged = profile.update(state, dynamics, initial, jnp.asarray(False))
    np.testing.assert_array_equal(unchanged.count_sum, state.count_sum)
    assert int(unchanged.samples) == 0


def test_exact_linear_msd_recovers_diffusion_tensor_and_immutable_artifact():
    time = jnp.arange(1.0, 7.0)
    diffusion = jnp.diag(jnp.asarray((0.2, 0.1, 0.05)))
    msd = 2.0 * time[:, None, None] * diffusion
    covariance = jnp.broadcast_to(jnp.eye(9)[None, :, :] * 1.0e-6, (6, 9, 9))
    header = phx.AdmissibilityHeader(
        jnp.ones((6,)),
        jnp.zeros((6,), dtype=jnp.uint32),
        "correlation",
        "correlation-evidence",
    )
    correlation = phx.atomistic.MultiOriginCorrelationResult(
        time,
        msd,
        covariance,
        jnp.zeros_like(msd),
        jnp.full((6,), 10, dtype=jnp.int32),
        header,
        "correlation-observer",
    )
    support = phx.atomistic.NanoflowClosureSupport(
        temperature_interval=(295.0, 305.0),
        confinement_interval=(3.9, 4.1),
        maximum_driving_magnitude=0.0,
        composition_id="single-component",
        lower_wall_id="lower",
        upper_wall_id="upper",
    )
    fit = phx.atomistic.DiffusionTensorFitPlan(
        0, 6, minimum_origins=4, stationarity_tolerance=1.0e-8
    ).evaluate(
        correlation,
        support_id=support.support_id,
        system_id="system",
        force_field_id="force-field",
        rollout_id="rollout",
    )

    assert bool(fit.header.globally_eligible)
    np.testing.assert_allclose(fit.diffusion_tensor, diffusion, atol=1.0e-14)
    artifact = phx.atomistic.diffusion_closure_artifact(
        fit,
        support,
        value_unit_id="length2/time",
        covariance_unit_id="length4/time2",
        system_id="system",
        force_field_id="force-field",
        rollout_id="rollout",
        observer_id="correlation-observer",
    )

    assert artifact.kind is phx.atomistic.NanoflowClosureKind.DIFFUSION_TENSOR
    np.testing.assert_allclose(artifact.value, diffusion)
    assert artifact.support.support_id == support.support_id
    assert artifact.artifact_id


def test_driven_profile_fit_recovers_two_wall_slip_lengths():
    centers = jnp.asarray((0.5, 1.5, 2.5, 3.5))
    velocity = jnp.stack(
        (
            1.0 + 2.0 * centers,
            jnp.zeros_like(centers),
        ),
        axis=-1,
    )[None, ...]
    profile = phx.atomistic.PlanarWallProfileResult(
        centers,
        jnp.ones((1, 4)),
        jnp.ones((1, 4)),
        jnp.zeros((1, 4)),
        velocity,
        jnp.zeros((1, 4, 2, 2)),
        jnp.full((1, 4), 10.0),
        jnp.zeros((1, 4), dtype="bool"),
        jnp.asarray(10, dtype=jnp.int32),
        phx.AdmissibilityHeader(
            jnp.ones((1, 4)),
            jnp.zeros((1, 4), dtype=jnp.uint32),
            "profile",
            "profile-evidence",
        ),
        "profile-observer",
    )
    fit = phx.atomistic.DrivenSlipFitPlan(0, 0, 0, 4).evaluate(
        profile,
        jnp.asarray(0.0),
        jnp.asarray(10.0),
        jnp.asarray(4.0),
        support_id="slip-support",
        system_id="system",
        force_field_id="force-field",
        rollout_id="rollout",
    )

    assert bool(fit.header.globally_eligible)
    np.testing.assert_allclose(fit.velocity_gradient, 2.0)
    np.testing.assert_allclose(fit.slip_lengths, (0.5, 0.5))


def test_exact_wall_force_contract_produces_friction_with_uncertainty():
    force = jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, 100))[:, None]
    result = phx.atomistic.WallForceCorrelationPlan(
        area=2.0,
        temperature=1.0,
        boltzmann_constant=1.0,
        time_step=0.01,
        lag_count=4,
        minimum_pairs=90,
        force_source_id="exact-wall-force-group",
        support_id="friction-support",
        system_id="system",
        force_field_id="force-field",
        rollout_id="rollout",
    ).evaluate(force)

    assert bool(result.header.globally_eligible)
    assert result.friction_coefficient > 0.0
    assert result.covariance.shape == (1, 1)
    assert result.force_source_id == "exact-wall-force-group"
