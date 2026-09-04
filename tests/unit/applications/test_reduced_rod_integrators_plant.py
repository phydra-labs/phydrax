from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_materials import (
    KelvinVoigtRodMaterialPlan,
)
from phydrax.applications.solid_mechanics._rod_plant import (
    prepare_reduced_rod_plant,
    reduced_rod_differential_system,
    ReducedRodMassResponseRevision,
    ReducedRodPlantState,
)
from phydrax.applications.solid_mechanics._rod_reduced_basis import (
    RodStrainBasisPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_dynamics import (
    prepare_reduced_rod_dynamics,
    ReducedRodDenseCholeskyPlan,
    ReducedRodMaterialControl,
)
from phydrax.applications.solid_mechanics._rod_reduced_integrators import (
    initialize_reduced_rod_integration_state,
    integrate_reduced_rod_step,
    ReducedRodImplicitMidpoint,
    ReducedRodSemiImplicitVelocityEuler,
    ReducedRodStepStatus,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)
from phydrax.dynamics import PlantStepContext
from phydrax.nonlinear import NonlinearTermination


def _axial_dynamics(*, viscosity: float = 0.0):
    dtype = jnp.float32
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((1.0, 1.2, 1.5), dtype=dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.2, 0.3, 0.4), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((40.0, 10.0, 10.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.diag(jnp.asarray((4.0, 5.0, 6.0), dtype=dtype))[None, ...],
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        components=("nu_x",),
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    reduction = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    if viscosity == 0.0:
        return prepare_reduced_rod_dynamics(reduction, ReducedRodDenseCholeskyPlan())
    stretch_viscosity = jnp.zeros_like(rod.plan.stretch_shear_stiffness)
    stretch_viscosity = stretch_viscosity.at[:, 0, 0].set(viscosity)
    bend_viscosity = jnp.zeros_like(rod.plan.bend_twist_stiffness)
    stretch = KelvinVoigtRodMaterialPlan(
        rod.plan.stretch_shear_stiffness, stretch_viscosity
    ).prepare(rod.stretch_shear_workset)
    bend = KelvinVoigtRodMaterialPlan(
        rod.plan.bend_twist_stiffness, bend_viscosity
    ).prepare(rod.bend_twist_workset)
    return prepare_reduced_rod_dynamics(
        reduction,
        ReducedRodDenseCholeskyPlan(),
        stretch_shear_material=stretch,
        bend_twist_material=bend,
    )


def _initial_state(dynamics, *, coefficient: float = 0.08, velocity: float = 0.0):
    dtype = dynamics.reduction.reference_coefficients.dtype
    return initialize_reduced_rod_integration_state(
        dynamics,
        ReducedRodState(
            jnp.asarray((coefficient,), dtype=dtype),
            jnp.asarray((velocity,), dtype=dtype),
        ),
    )


def _assert_tree_arrays_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        assert jnp.array_equal(left_leaf, right_leaf)


def _advance(dynamics, policy, source, step_size: float, count: int):
    state = source
    for _ in range(count):
        result = integrate_reduced_rod_step(
            dynamics,
            policy,
            state,
            jnp.asarray(step_size, dtype=source.reduced_state.values.dtype),
        )
        assert bool(result.successful)
        state = result.accepted_state
    return state


def test_velocity_euler_is_velocity_first_and_first_order_time_convergent():
    dynamics = _axial_dynamics()
    source = _initial_state(dynamics)
    step = jnp.asarray(1.0e-3, dtype=source.reduced_state.values.dtype)
    policy = ReducedRodSemiImplicitVelocityEuler(
        maximum_step_size=float(step), energy_balance_tolerance=1.0
    )
    forward = dynamics.forward_dynamics(
        source.reduced_state,
        material_state=source.material_state,
        time=source.time,
        step_size=step,
    )
    first = integrate_reduced_rod_step(dynamics, policy, source, step)
    expected_velocity = (
        source.reduced_state.coefficient_velocities + step * forward.acceleration
    )

    assert policy.route == "semi-implicit-velocity-euler"
    assert jnp.array_equal(
        first.candidate_state.reduced_state.coefficient_velocities,
        expected_velocity,
    )
    assert jnp.array_equal(
        first.candidate_state.reduced_state.coefficients,
        source.reduced_state.coefficients + step * expected_velocity,
    )

    mass = dynamics.mass(jnp.zeros((1,), dtype=step.dtype)).operator.mv(
        jnp.ones((1,), dtype=step.dtype)
    )[0]
    unit = dynamics.energy(
        ReducedRodState(
            jnp.ones((1,), dtype=step.dtype), jnp.zeros((1,), dtype=step.dtype)
        )
    ).stored_energy
    frequency = jnp.sqrt(2.0 * unit / mass)
    final_time = 0.2
    exact = source.reduced_state.coefficients[0] * jnp.cos(frequency * final_time)
    coarse = _advance(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01, energy_balance_tolerance=1.0
        ),
        source,
        0.01,
        20,
    )
    fine = _advance(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.005, energy_balance_tolerance=1.0
        ),
        source,
        0.005,
        40,
    )
    assert jnp.abs(fine.reduced_state.coefficients[0] - exact) < jnp.abs(
        coarse.reduced_state.coefficients[0] - exact
    )


def test_implicit_midpoint_conservative_and_damped_ledgers_close():
    conservative = _axial_dynamics()
    source = _initial_state(conservative, coefficient=0.08, velocity=-0.1)
    policy = ReducedRodImplicitMidpoint(
        maximum_step_size=0.005, energy_balance_tolerance=2.0e-4
    )
    result = integrate_reduced_rod_step(conservative, policy, source, 0.005)
    ledger = result.evidence.ledger

    assert policy.route == "implicit-midpoint"
    assert bool(result.successful)
    assert bool(ledger.balanced)
    assert ledger.external_work == pytest.approx(0.0, abs=2.0e-7)
    assert ledger.viscous_dissipation == pytest.approx(0.0, abs=2.0e-7)
    assert ledger.mechanical_energy_after == pytest.approx(
        float(ledger.mechanical_energy_before), rel=2.0e-4, abs=2.0e-6
    )

    damped = _axial_dynamics(viscosity=0.4)
    damped_source = _initial_state(damped, coefficient=0.08, velocity=-0.1)
    damped_result = integrate_reduced_rod_step(
        damped,
        ReducedRodImplicitMidpoint(
            maximum_step_size=0.005, energy_balance_tolerance=4.0e-4
        ),
        damped_source,
        0.005,
    )
    damped_ledger = damped_result.evidence.ledger
    assert bool(damped_result.successful)
    assert bool(damped_ledger.dissipation_nonnegative)
    assert bool(damped_ledger.balanced)
    assert damped_ledger.viscous_dissipation > 0.0
    assert damped_ledger.mechanical_energy_after < damped_ledger.mechanical_energy_before


def test_material_mass_and_nonlinear_failures_roll_back_every_integration_leaf():
    dynamics = _axial_dynamics()
    source = _initial_state(dynamics, coefficient=0.1, velocity=0.2)
    explicit = ReducedRodSemiImplicitVelocityEuler(
        maximum_step_size=0.01, energy_balance_tolerance=1.0
    )

    passive = dynamics.initialize_material_control()
    invalid_stretch = eqx.tree_at(
        lambda control: control.intrinsic_strain,
        passive.stretch_shear_control,
        jnp.full_like(passive.stretch_shear_control.intrinsic_strain, jnp.nan),
    )
    material_failure = integrate_reduced_rod_step(
        dynamics,
        explicit,
        source,
        0.005,
        material_control=ReducedRodMaterialControl(
            invalid_stretch, passive.bend_twist_control
        ),
    )
    assert not bool(material_failure.successful)
    _assert_tree_arrays_equal(material_failure.accepted_state, source)

    mass_failure_dynamics = eqx.tree_at(
        lambda prepared: prepared.reduction.rod.node_masses,
        dynamics,
        jnp.zeros_like(dynamics.reduction.rod.node_masses),
    )
    mass_failure = integrate_reduced_rod_step(
        mass_failure_dynamics, explicit, source, 0.005
    )
    assert not bool(mass_failure.successful)
    assert int(mass_failure.status) == int(ReducedRodStepStatus.MASS_SOLVE_FAILED)
    _assert_tree_arrays_equal(mass_failure.accepted_state, source)

    nonlinear_failure = integrate_reduced_rod_step(
        dynamics,
        ReducedRodImplicitMidpoint(
            maximum_step_size=0.01,
            nonlinear_termination=NonlinearTermination(
                absolute_residual=0.0,
                relative_residual=0.0,
                absolute_step=0.0,
                relative_step=0.0,
                maximum_steps=1,
                maximum_evaluations=1,
            ),
            energy_balance_tolerance=1.0,
        ),
        source,
        0.005,
    )
    assert not bool(nonlinear_failure.successful)
    assert int(nonlinear_failure.status) == int(
        ReducedRodStepStatus.NONLINEAR_SOLVE_FAILED
    )
    _assert_tree_arrays_equal(nonlinear_failure.accepted_state, source)


def test_passive_plant_reset_step_rollback_checkpoint_replay_and_mass_revision():
    dynamics = _axial_dynamics()
    plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01, energy_balance_tolerance=1.0
        ),
    )
    parameters = plant.bind_parameters()
    key = jax.random.key(17)
    reset = plant.reset(key, parameters)

    assert bool(reset.successful)
    assert isinstance(reset.accepted_state.payload, ReducedRodPlantState)
    assert plant.control_schema is None
    assert plant.state_schema.schema_id == reset.accepted_state.state_schema_id
    assert plant.parameter_schema.schema_id == parameters.schema_id
    assert reset.accepted_state.payload.actuator_state.values.shape == (0,)
    assert reset.accepted_state.payload.contact_state.values.shape == (0,)
    assert reset.accepted_state.payload.sensor_state.values.shape == (0,)

    context = PlantStepContext(
        reset.accepted_state.time,
        reset.accepted_state.time + jnp.asarray(0.005),
        reset.accepted_state.step_index,
    )
    direct = plant.step(context, reset.accepted_state, None, parameters)
    assert bool(direct.successful)
    assert int(direct.status) == int(ReducedRodStepStatus.SUCCESS)

    checkpoint = plant.checkpoint(reset.accepted_state)
    assert plant.verify_checkpoint(checkpoint)
    expected_digest = plant.state_digest(direct.accepted_state)
    replay = plant.replay(
        checkpoint,
        (context,),
        (None,),
        parameters,
        expected_digests=(expected_digest,),
    )
    assert bool(replay.successful)
    assert replay.matched
    assert plant.state_digest(replay.final_state) == expected_digest

    mass_response = plant.mass_response(direct.accepted_state)
    assert isinstance(mass_response, ReducedRodMassResponseRevision)
    assert bool(mass_response.valid)
    assert bool(mass_response.is_current(direct.accepted_state))
    assert not bool(mass_response.is_current(reset.accepted_state))
    impulse = jnp.asarray((0.03,), dtype=mass_response.configuration.dtype)
    increment = mass_response.apply_impulse(impulse)
    assert jnp.allclose(
        mass_response.mass.operator.mv(increment), impulse, rtol=2.0e-5, atol=2.0e-6
    )

    failed_context = PlantStepContext(
        direct.accepted_state.time,
        direct.accepted_state.time + jnp.asarray(0.02),
        direct.accepted_state.step_index,
    )
    failed = plant.step(failed_context, direct.accepted_state, None, parameters)
    assert not bool(failed.successful)
    assert int(failed.status) == int(ReducedRodStepStatus.STEP_OUT_OF_BOUNDS)
    _assert_tree_arrays_equal(failed.accepted_state, direct.accepted_state)
    assert jnp.array_equal(
        jax.random.key_data(failed.accepted_state.key),
        jax.random.key_data(direct.accepted_state.key),
    )
    assert failed.candidate_state.time == failed_context.target_time


def test_plant_material_mass_and_nonlinear_failures_roll_back_payload_clock_and_key():
    dynamics = _axial_dynamics()
    explicit_plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=0.01, energy_balance_tolerance=1.0
        ),
    )
    explicit_parameters = explicit_plant.bind_parameters()
    explicit_reset = explicit_plant.reset(
        jax.random.key(23), explicit_parameters, initial_time=0.0
    )
    explicit_context = PlantStepContext(
        explicit_reset.accepted_state.time,
        explicit_reset.accepted_state.time + jnp.asarray(0.005),
        explicit_reset.accepted_state.step_index,
    )

    invalid_material_control = eqx.tree_at(
        lambda control: control.stretch_shear_control.intrinsic_strain,
        explicit_plant.material_control,
        jnp.full_like(
            explicit_plant.material_control.stretch_shear_control.intrinsic_strain,
            jnp.nan,
        ),
    )
    material_plant = eqx.tree_at(
        lambda prepared: prepared.material_control,
        explicit_plant,
        invalid_material_control,
    )
    material_failure = material_plant.step(
        explicit_context,
        explicit_reset.accepted_state,
        None,
        explicit_parameters,
    )
    assert not bool(material_failure.successful)
    _assert_tree_arrays_equal(
        material_failure.accepted_state, explicit_reset.accepted_state
    )

    mass_plant = eqx.tree_at(
        lambda prepared: prepared.dynamics.reduction.rod.node_masses,
        explicit_plant,
        jnp.zeros_like(dynamics.reduction.rod.node_masses),
    )
    mass_failure = mass_plant.step(
        explicit_context,
        explicit_reset.accepted_state,
        None,
        explicit_parameters,
    )
    assert not bool(mass_failure.successful)
    assert int(mass_failure.status) == int(ReducedRodStepStatus.MASS_SOLVE_FAILED)
    _assert_tree_arrays_equal(mass_failure.accepted_state, explicit_reset.accepted_state)

    dtype = dynamics.reduction.reference_coefficients.dtype
    nonlinear_plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodImplicitMidpoint(
            maximum_step_size=0.01,
            nonlinear_termination=NonlinearTermination(
                absolute_residual=0.0,
                relative_residual=0.0,
                absolute_step=0.0,
                relative_step=0.0,
                maximum_steps=1,
                maximum_evaluations=1,
            ),
            energy_balance_tolerance=1.0,
        ),
        initial_reduced_state=ReducedRodState(
            jnp.asarray((0.1,), dtype=dtype),
            jnp.asarray((0.2,), dtype=dtype),
        ),
    )
    nonlinear_parameters = nonlinear_plant.bind_parameters()
    nonlinear_reset = nonlinear_plant.reset(
        jax.random.key(29), nonlinear_parameters, initial_time=0.0
    )
    nonlinear_context = PlantStepContext(
        nonlinear_reset.accepted_state.time,
        nonlinear_reset.accepted_state.time + jnp.asarray(0.005),
        nonlinear_reset.accepted_state.step_index,
    )
    nonlinear_failure = nonlinear_plant.step(
        nonlinear_context,
        nonlinear_reset.accepted_state,
        None,
        nonlinear_parameters,
    )
    assert not bool(nonlinear_failure.successful)
    assert int(nonlinear_failure.status) == int(
        ReducedRodStepStatus.NONLINEAR_SOLVE_FAILED
    )
    _assert_tree_arrays_equal(
        nonlinear_failure.accepted_state, nonlinear_reset.accepted_state
    )


def test_differential_adapter_is_the_same_smooth_stateless_contact_free_law():
    dynamics = _axial_dynamics(viscosity=0.25)
    plant = prepare_reduced_rod_plant(
        dynamics,
        ReducedRodImplicitMidpoint(maximum_step_size=0.01, energy_balance_tolerance=1.0),
    )
    system = reduced_rod_differential_system(plant)
    dtype = dynamics.reduction.reference_coefficients.dtype
    time = jnp.asarray(0.2, dtype=dtype)
    q = jnp.asarray((0.04,), dtype=dtype)
    v = jnp.asarray((-0.03,), dtype=dtype)
    a = jnp.asarray((0.07,), dtype=dtype)
    observed = system.evaluate(time, q, v, a, None)
    expected = dynamics.inverse_dynamics(
        ReducedRodState(q, v),
        a,
        material_state=dynamics.initialize_material_state(),
        material_control=plant.material_control,
        time=time,
        step_size=jnp.asarray(1.0, dtype=dtype),
    ).residual

    assert system.state_shape == (1,)
    assert jnp.allclose(observed, expected)
    with pytest.raises(TypeError, match="no runtime arguments"):
        system.evaluate(time, q, v, a, {})
