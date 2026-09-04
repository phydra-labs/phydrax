from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_advanced_actuation import (
    AffineMagneticActuationPlan,
    combine_rod_constitutive_controls,
    IntrinsicStrainActuationPlan,
    IntrinsicStrainCommand,
    MagneticCurrentCommand,
    ReducedTubeChamberPlan,
    RegulatedReducedTubePressurePlan,
    RegulatedTubePressureCommand,
    RodTubeStation,
    SealedReducedTubePressurePlan,
    VariableStiffnessActuationPlan,
    VariableStiffnessCommand,
)
from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_reduced_basis import RodStrainBasisPlan
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
    ReducedRodState,
)


def _spatial_reduced_rod():
    dtype = jnp.float32
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.asarray((1.0, 1.0, 1.0), dtype=dtype),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((80.0, 50.0, 40.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((7.0, 8.0, 9.0), dtype=dtype)),
                (1, 3, 3),
            ),
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    return prepare_reduced_rod(rod, ReducedRodPlan(basis))


def _tube_plan():
    dtype = jnp.float32
    return ReducedTubeChamberPlan(
        (
            RodTubeStation(0, 0.0, jnp.asarray((0.0, 0.1, 0.0), dtype=dtype)),
            RodTubeStation(1, 1.0, jnp.asarray((0.0, 0.1, 0.0), dtype=dtype)),
        ),
        jnp.asarray((0.01,), dtype=dtype),
        0.2,
        volume_bounds=(0.21, 0.25),
        ambient_pressure=100.0,
        source_manifest_id="tube-geometry-measurements",
        calibration_id="tube-area-volume-fit",
    )


def _moving_state(reduction):
    velocity = jnp.zeros_like(reduction.reference_coefficients).at[0].set(0.2)
    return ReducedRodState(reduction.reference_coefficients, velocity)


def test_regulated_tube_pressure_has_exact_pa_and_eccentric_dual_moment():
    reduction = _spatial_reduced_rod()
    plan = RegulatedReducedTubePressurePlan(
        _tube_plan(),
        pressure_bounds=(0.0, 1.0e4),
        maximum_rise_rate=1.0e6,
        maximum_fall_rate=1.0e6,
    )
    prepared = plan.prepare(reduction)
    dtype = reduction.reference_coefficients.dtype
    state = prepared.initialize_state(jnp.asarray(0.0, dtype=dtype))
    command = RegulatedTubePressureCommand(jnp.asarray(2000.0, dtype=dtype))
    evaluation = prepared.evaluate(
        _moving_state(reduction), state, command, jnp.asarray(1.0, dtype=dtype)
    )

    expected_force = 2000.0 * 0.01
    expected_moment = expected_force * 0.1
    assert evaluation.native_forces[0, 0] == pytest.approx(-expected_force)
    assert evaluation.native_forces[2, 0] == pytest.approx(expected_force)
    assert evaluation.native_moments[0, 2] == pytest.approx(expected_moment)
    assert evaluation.native_moments[1, 2] == pytest.approx(-expected_moment)
    assert evaluation.mechanical_power == pytest.approx(
        evaluation.applied_gauge_pressure * evaluation.volume_rate
    )
    assert evaluation.native_virtual_work_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.reduced_virtual_work_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.power_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.valid
    assert not evaluation.electrical_power_available


def test_tube_volume_vjp_is_the_exact_reduced_pressure_effort():
    reduction = _spatial_reduced_rod()
    chamber = _tube_plan().prepare(reduction)
    state = _moving_state(reduction)
    unit_effort = chamber.reduced_volume_rate_operator(state).transpose_mv(
        jnp.asarray(1.0, dtype=state.coefficients.dtype)
    )
    energy_gradient = jax.grad(
        lambda coefficients: chamber.volume(
            ReducedRodState(coefficients, state.coefficient_velocities)
        )
    )(state.coefficients)

    assert jnp.allclose(unit_effort, energy_gradient, rtol=2.0e-5, atol=2.0e-5)
    assert chamber.volume(state) == pytest.approx(0.22)


def test_regulator_saturation_is_explicit_and_source_state_is_not_committed():
    reduction = _spatial_reduced_rod()
    prepared = RegulatedReducedTubePressurePlan(
        _tube_plan(),
        pressure_bounds=(0.0, 100.0),
        maximum_rise_rate=10.0,
        maximum_fall_rate=20.0,
    ).prepare(reduction)
    dtype = reduction.reference_coefficients.dtype
    source = prepared.initialize_state(jnp.asarray(5.0, dtype=dtype))
    evaluation = prepared.evaluate(
        reduction.initialize_state(),
        source,
        RegulatedTubePressureCommand(jnp.asarray(1000.0, dtype=dtype)),
        jnp.asarray(0.5, dtype=dtype),
    )

    assert source.gauge_pressure == pytest.approx(5.0)
    assert evaluation.candidate_state.gauge_pressure == pytest.approx(10.0)
    assert evaluation.applied_pressure_rate == pytest.approx(10.0)
    assert evaluation.saturated
    assert evaluation.valid


def test_sealed_tube_gas_has_conservative_polytropic_power_and_no_source_power():
    reduction = _spatial_reduced_rod()
    prepared = SealedReducedTubePressurePlan(
        _tube_plan(),
        220.0,
        0.22,
        exponent=1.4,
    ).prepare(reduction)
    dtype = reduction.reference_coefficients.dtype
    state = prepared.initialize_state(jnp.asarray(1.0, dtype=dtype))
    rod_state = _moving_state(reduction)
    evaluation = prepared.evaluate(rod_state, state)
    energy_gradient = jax.grad(
        lambda coefficients: (
            prepared.evaluate(
                ReducedRodState(coefficients, jnp.zeros_like(coefficients)), state
            ).stored_energy
        )
    )(rod_state.coefficients)

    assert evaluation.candidate_state is state
    assert evaluation.gauge_pressure == pytest.approx(120.0)
    assert evaluation.source_power == pytest.approx(0.0)
    assert evaluation.stored_power == pytest.approx(-evaluation.mechanical_power)
    assert jnp.allclose(
        energy_gradient,
        -evaluation.reduced_effort,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert evaluation.valid


def test_intrinsic_strain_is_subtracted_inside_the_material_trial():
    reduction = _spatial_reduced_rod()
    material = reduction.rod.stretch_shear_material
    dtype = reduction.reference_coefficients.dtype
    modes = jnp.zeros((2, 3, 1), dtype=dtype).at[:, 0, 0].set(0.1)
    prepared = IntrinsicStrainActuationPlan(
        modes,
        maximum_rise_rate=2.0,
        maximum_fall_rate=2.0,
        source_manifest_id="intrinsic-mode-shapes",
        calibration_id="curvature-command-fit",
    ).prepare(material)
    source_state = prepared.initialize_state(jnp.asarray((0.0,), dtype=dtype))
    command = IntrinsicStrainCommand(jnp.asarray((1.0,), dtype=dtype))
    source_strain = material.workset.reference_strains
    candidate_strain = source_strain + modes[..., 0]
    evaluation = prepared.evaluate(
        source_strain,
        candidate_strain,
        modes[..., 0],
        material.initialize_history(),
        source_state,
        command,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
    )

    assert jnp.allclose(
        evaluation.material_result.elastic_strain,
        0.0,
        atol=8.0 * jnp.finfo(dtype).eps,
    )
    assert jnp.allclose(
        evaluation.material_result.resultants,
        0.0,
        atol=8.0 * jnp.finfo(dtype).eps * jnp.max(jnp.abs(material.plan.stiffness)),
    )
    assert evaluation.stored_energy == pytest.approx(0.0)
    assert evaluation.power_residual == pytest.approx(0.0)
    assert jnp.array_equal(source_state.activation, jnp.asarray((0.0,), dtype=dtype))
    assert jnp.array_equal(
        evaluation.candidate.candidate_state.activation,
        jnp.asarray((1.0,), dtype=dtype),
    )
    assert evaluation.valid


def test_zero_intrinsic_command_exactly_reproduces_passive_material():
    reduction = _spatial_reduced_rod()
    material = reduction.rod.stretch_shear_material
    dtype = reduction.reference_coefficients.dtype
    modes = jnp.ones((2, 3, 1), dtype=dtype)
    prepared = IntrinsicStrainActuationPlan(
        modes,
        maximum_rise_rate=1.0,
        maximum_fall_rate=1.0,
        source_manifest_id="intrinsic-modes",
        calibration_id="intrinsic-zero",
    ).prepare(material)
    state = prepared.initialize_state(jnp.asarray((0.0,), dtype=dtype))
    command = IntrinsicStrainCommand(jnp.asarray((0.0,), dtype=dtype))
    source = material.workset.reference_strains
    rate = jnp.asarray(((0.2, -0.1, 0.0), (0.1, 0.0, -0.2)), dtype=dtype)
    step = jnp.asarray(0.5, dtype=dtype)
    candidate = source + step * rate
    controlled = prepared.evaluate(
        source,
        candidate,
        rate,
        material.initialize_history(),
        state,
        command,
        jnp.asarray(0.0, dtype=dtype),
        step,
    ).material_result
    passive = material(
        source,
        candidate,
        rate,
        material.initialize_history(),
        None,
        jnp.asarray(0.0, dtype=dtype),
        step,
    )

    assert jnp.array_equal(controlled.resultants, passive.resultants)
    assert jnp.array_equal(
        controlled.stored_energy_density, passive.stored_energy_density
    )
    assert controlled.control_source_power == pytest.approx(0.0)


def test_variable_stiffness_interpolates_psd_endpoints_and_nonnegative_energy():
    reduction = _spatial_reduced_rod()
    material = reduction.rod.stretch_shear_material
    dtype = reduction.reference_coefficients.dtype
    low = jnp.broadcast_to(jnp.diag(jnp.asarray((2.0, 3.0, 4.0), dtype=dtype)), (2, 3, 3))
    high = jnp.broadcast_to(
        jnp.diag(jnp.asarray((8.0, 9.0, 10.0), dtype=dtype)), (2, 3, 3)
    )
    prepared = VariableStiffnessActuationPlan(
        low,
        high,
        maximum_rise_rate=2.0,
        maximum_fall_rate=2.0,
        source_manifest_id="stiffness-measurements",
        calibration_id="stiffness-endpoints",
    ).prepare(material)
    source = material.workset.reference_strains
    rate = jnp.full_like(source, 0.1)
    step = jnp.asarray(1.0, dtype=dtype)
    candidate = source + step * rate
    result = prepared.evaluate(
        source,
        candidate,
        rate,
        material.initialize_history(),
        prepared.initialize_state(jnp.asarray(0.0, dtype=dtype)),
        VariableStiffnessCommand(jnp.asarray(1.0, dtype=dtype)),
        jnp.asarray(0.0, dtype=dtype),
        step,
    )

    assert jnp.array_equal(result.candidate.effective_stiffness, high)
    assert result.candidate.minimum_eigenvalue >= 0.0
    assert result.material_result.stored_energy >= 0.0
    assert result.material_result.evidence.stiffness_psd
    assert result.power_residual == pytest.approx(0.0, abs=2.0e-6)
    assert result.valid


def test_intrinsic_and_stiffness_controls_compose_into_one_material_evaluation():
    reduction = _spatial_reduced_rod()
    material = reduction.rod.stretch_shear_material
    dtype = reduction.reference_coefficients.dtype
    modes = jnp.zeros((2, 3, 1), dtype=dtype).at[:, 1, 0].set(0.05)
    intrinsic = IntrinsicStrainActuationPlan(
        modes,
        maximum_rise_rate=2.0,
        maximum_fall_rate=2.0,
        source_manifest_id="intrinsic",
        calibration_id="intrinsic-fit",
    ).prepare(material)
    low = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
    high = 3.0 * low
    stiffness = VariableStiffnessActuationPlan(
        low,
        high,
        maximum_rise_rate=2.0,
        maximum_fall_rate=2.0,
        source_manifest_id="stiffness",
        calibration_id="stiffness-fit",
    ).prepare(material)
    intrinsic_candidate = intrinsic.candidate_control(
        intrinsic.initialize_state(jnp.asarray((0.0,), dtype=dtype)),
        IntrinsicStrainCommand(jnp.asarray((1.0,), dtype=dtype)),
        jnp.asarray(1.0, dtype=dtype),
    )
    stiffness_candidate = stiffness.candidate_control(
        stiffness.initialize_state(jnp.asarray(0.0, dtype=dtype)),
        VariableStiffnessCommand(jnp.asarray(1.0, dtype=dtype)),
        jnp.asarray(1.0, dtype=dtype),
    )
    control = combine_rod_constitutive_controls(
        intrinsic_candidate.control, stiffness_candidate.control
    )
    source = material.workset.reference_strains
    physical_rate = jnp.full_like(source, 0.2)
    result = material(
        source,
        source + physical_rate,
        physical_rate,
        material.initialize_history(),
        control,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
    )
    expected = jnp.einsum("sij,sj->si", high, physical_rate - modes[..., 0])

    assert jnp.allclose(result.elastic_resultants, expected)
    assert result.evidence.valid
    with pytest.raises(ValueError, match="Overlapping intrinsic-strain owners"):
        combine_rod_constitutive_controls(
            intrinsic_candidate.control, intrinsic_candidate.control
        )


def test_variable_stiffness_rejects_unordered_psd_endpoints():
    dtype = jnp.float32
    minimum = jnp.asarray((((2.0, 0.0), (0.0, 2.0)),), dtype=dtype)
    maximum = jnp.asarray((((1.0, 0.0), (0.0, 3.0)),), dtype=dtype)
    with pytest.raises(ValueError, match="positive semidefinite"):
        VariableStiffnessActuationPlan(
            minimum,
            maximum,
            maximum_rise_rate=1.0,
            maximum_fall_rate=1.0,
            source_manifest_id="stiffness",
            calibration_id="unordered",
        )


def _magnetic_plan(*, affine: bool):
    dtype = jnp.float32
    gradient = jnp.zeros((1, 3, 3), dtype=dtype)
    uniform = jnp.asarray(((0.0, 2.0, 0.0),), dtype=dtype)
    if affine:
        gradient = jnp.asarray(
            (((3.0, 0.0, 0.0), (0.0, -3.0, 0.0), (0.0, 0.0, 0.0)),), dtype=dtype
        )
        uniform = jnp.zeros((1, 3), dtype=dtype)
    return AffineMagneticActuationPlan(
        uniform,
        gradient,
        jnp.zeros((3,), dtype=dtype),
        jnp.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=dtype),
        position_bounds=(
            jnp.full((3,), -10.0, dtype=dtype),
            jnp.full((3,), 10.0, dtype=dtype),
        ),
        current_bounds=(-2.0, 2.0),
        maximum_rise_rate=10.0,
        maximum_fall_rate=10.0,
        source_manifest_id="field-map-and-dipoles",
        calibration_id="affine-field-fit",
    )


def test_uniform_affine_magnetic_field_has_torque_and_zero_force():
    reduction = _spatial_reduced_rod()
    prepared = _magnetic_plan(affine=False).prepare(reduction)
    dtype = reduction.reference_coefficients.dtype
    evaluation = prepared.evaluate(
        reduction.initialize_state(),
        prepared.initialize_state(jnp.asarray((0.0,), dtype=dtype)),
        MagneticCurrentCommand(jnp.asarray((1.0,), dtype=dtype)),
        jnp.asarray(1.0, dtype=dtype),
    )

    assert jnp.allclose(evaluation.segment_forces_world, 0.0)
    assert jnp.allclose(
        evaluation.segment_torques_world,
        jnp.asarray(((0.0, 0.0, 2.0), (0.0, 0.0, 2.0)), dtype=dtype),
    )
    assert jnp.allclose(
        evaluation.native_moments,
        jnp.asarray(((0.0, 0.0, 2.0), (0.0, 0.0, 2.0)), dtype=dtype),
    )
    assert evaluation.valid
    assert not evaluation.electrical_power_available


def test_affine_magnetic_gradient_gives_analytic_force_and_exact_reduced_power():
    reduction = _spatial_reduced_rod()
    prepared = _magnetic_plan(affine=True).prepare(reduction)
    dtype = reduction.reference_coefficients.dtype
    evaluation = prepared.evaluate(
        _moving_state(reduction),
        prepared.initialize_state(jnp.asarray((0.0,), dtype=dtype)),
        MagneticCurrentCommand(jnp.asarray((1.0,), dtype=dtype)),
        jnp.asarray(1.0, dtype=dtype),
    )

    assert jnp.allclose(
        evaluation.segment_forces_world,
        jnp.asarray(((3.0, 0.0, 0.0), (3.0, 0.0, 0.0)), dtype=dtype),
    )
    assert evaluation.native_virtual_work_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.reduced_virtual_work_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.power_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.valid


def test_unsupported_pressure_constitutive_and_magnetic_capabilities_fail_closed():
    with pytest.raises(ValueError, match="deformable_cross_sections"):
        ReducedTubeChamberPlan(
            _tube_plan().stations,
            jnp.asarray((0.01,), dtype=jnp.float32),
            0.2,
            volume_bounds=(0.21, 0.25),
            ambient_pressure=100.0,
            source_manifest_id="tube",
            calibration_id="tube",
            deformable_cross_sections=True,
        )
    with pytest.raises(ValueError, match="hysteresis"):
        IntrinsicStrainActuationPlan(
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
            maximum_rise_rate=1.0,
            maximum_fall_rate=1.0,
            source_manifest_id="intrinsic",
            calibration_id="intrinsic",
            hysteresis=True,
        )
    with pytest.raises(ValueError, match="jamming"):
        VariableStiffnessActuationPlan(
            jnp.ones((1, 1, 1), dtype=jnp.float32),
            2.0 * jnp.ones((1, 1, 1), dtype=jnp.float32),
            maximum_rise_rate=1.0,
            maximum_fall_rate=1.0,
            source_manifest_id="stiffness",
            calibration_id="stiffness",
            jamming=True,
        )
    with pytest.raises(ValueError, match="maxwell_solves"):
        AffineMagneticActuationPlan(
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, 3, 3), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
            position_bounds=(
                -jnp.ones((3,), dtype=jnp.float32),
                jnp.ones((3,), dtype=jnp.float32),
            ),
            current_bounds=(-1.0, 1.0),
            maximum_rise_rate=1.0,
            maximum_fall_rate=1.0,
            source_manifest_id="magnetic",
            calibration_id="magnetic",
            maxwell_solves=True,
        )
