import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compressible_flow._contracts import CompressibleFlowCaseSpec
from phydrax.applications.compressible_flow._slow_growth import (
    CompressiblePlaneBaseflowPlan,
    SlowGrowthContinuation,
    SpatialSlowGrowthModelPlan,
    TemporalSlowGrowthModelPlan,
)
from phydrax.equations import (
    ChemicalPhaseKind,
    ChemicalSpeciesSchema,
    HomogeneousHelmholtzPlan,
    IdealGasReferenceHelmholtzTerm,
    PolynomialSpeciesThermodynamicsPlan,
    UNIVERSAL_GAS_CONSTANT,
    ZeroResidualHelmholtzTerm,
)


def _model():
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (ChemicalPhaseKind.GAS, ChemicalPhaseKind.GAS),
        jnp.asarray((0.020, 0.030)),
        ("A", "B"),
        jnp.eye(2, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    calorics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((2, 1), 2.5 * UNIVERSAL_GAS_CONSTANT),
        jnp.asarray((1.0e3, 2.0e3)),
        reference_molar_entropy=jnp.asarray((100.0, 110.0)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=1500.0,
    )
    ideal = IdealGasReferenceHelmholtzTerm(schema, calorics)
    return HomogeneousHelmholtzPlan(ideal, ZeroResidualHelmholtzTerm(schema))


def _case() -> CompressibleFlowCaseSpec:
    return CompressibleFlowCaseSpec(
        "slow-growth-test",
        2,
        "navier_stokes",
        "structured-fv",
        _model(),
    )


def _profile(y, *, offset=0.0):
    density = 1.0 + offset + 0.15 * y
    streamwise_velocity = 1.0 - jnp.exp(-4.0 * y) + 0.1
    temperature = 500.0 + 8.0 * y + 4.0 * y * y + 5.0 * offset
    return jnp.stack(
        (
            0.4 * density,
            0.6 * density,
            streamwise_velocity,
            jnp.zeros_like(y),
            temperature,
        ),
        axis=-1,
    )


def _state_and_snapshot(*, streamwise_derivative=None, offset=0.0, sample_index=0):
    case = _case()
    y = jnp.linspace(0.0, 1.0, 9)
    primitive_profile = _profile(y, offset=offset)
    primitive = jnp.broadcast_to(primitive_profile, (4,) + primitive_profile.shape)
    conserved = case.primitive_to_conserved(primitive)
    plan = CompressiblePlaneBaseflowPlan(case, y, wall_normal_axis=1)
    snapshot = plan.evaluate(
        conserved,
        streamwise_base_derivative=streamwise_derivative,
        sample_index=sample_index,
        sample_time=0.25 * sample_index,
        streamwise_location=2.0 + 0.1 * sample_index,
    )
    return case, y, conserved, plan, snapshot


def test_density_weighted_baseflow_and_zero_growth_reduction():
    case = _case()
    y = jnp.linspace(0.0, 1.0, 7)
    x = jnp.arange(3.0)[:, None]
    density = 1.0 + 0.2 * y[None, :] + 0.1 * x
    species_density = jnp.stack((0.4 * density, 0.6 * density), axis=-1)
    velocity_x = 0.25 + y[None, :] + 0.2 * x
    velocity = jnp.stack((velocity_x, jnp.zeros_like(velocity_x)), axis=-1)
    temperature = jnp.full_like(density, 500.0)
    primitive = jnp.concatenate(
        (species_density, velocity, temperature[..., None]), axis=-1
    )
    conserved = case.primitive_to_conserved(primitive)
    snapshot = CompressiblePlaneBaseflowPlan(case, y, wall_normal_axis=1).evaluate(
        conserved
    )
    expected_favre = jnp.mean(density * velocity_x, axis=0) / jnp.mean(density, axis=0)
    np.testing.assert_allclose(snapshot.favre_mean_velocity[:, 0], expected_favre)
    np.testing.assert_allclose(snapshot.base_primitive[:, 2], expected_favre)

    prepared = TemporalSlowGrowthModelPlan(0.0).prepare(snapshot)
    result = prepared.evaluate(conserved)
    np.testing.assert_allclose(result.source.primitive, 0.0)
    np.testing.assert_allclose(result.source.conservative, 0.0)
    np.testing.assert_allclose(result.evidence.zero_source_residual, 0.0)
    assert bool(result.evidence.admissible)


def test_temporal_manufactured_base_uses_only_temporal_dilation_derivative():
    case = _case()
    y = jnp.linspace(0.0, 1.0, 6)
    primitive_profile = jnp.stack(
        (
            0.4 * (1.0 + 0.2 * y),
            0.6 * (1.0 + 0.2 * y),
            0.5 + 0.3 * y,
            -0.1 + 0.05 * y,
            500.0 + 0.4 * y,
        ),
        axis=-1,
    )
    conserved = case.primitive_to_conserved(
        jnp.broadcast_to(primitive_profile, (3,) + primitive_profile.shape)
    )
    snapshot = CompressiblePlaneBaseflowPlan(case, y, wall_normal_axis=1).evaluate(
        conserved
    )
    growth_rate = 0.125
    prepared = TemporalSlowGrowthModelPlan(growth_rate, wall_indices=()).prepare(snapshot)
    expected = -growth_rate * y[:, None] * jnp.asarray((0.08, 0.12, 0.3, 0.05, 0.4))
    np.testing.assert_allclose(prepared.primitive_source_profile, expected, atol=1e-12)
    assert prepared.coordinate == "temporal"


def test_modeled_spatial_requires_and_uses_supplied_streamwise_derivatives():
    _, y, _, _, snapshot_without_derivative = _state_and_snapshot()
    with pytest.raises(ValueError, match="streamwise base derivatives"):
        SpatialSlowGrowthModelPlan().prepare(snapshot_without_derivative)

    derivative = jnp.stack(
        (
            0.008 * jnp.ones_like(y),
            0.012 * jnp.ones_like(y),
            -0.03 * (1.0 + y),
            jnp.zeros_like(y),
            0.01 * y,
        ),
        axis=-1,
    )
    _, _, _, _, snapshot = _state_and_snapshot(streamwise_derivative=derivative)
    prepared = SpatialSlowGrowthModelPlan(2.5, wall_indices=()).prepare(snapshot)
    np.testing.assert_allclose(
        prepared.primitive_source_profile, -2.5 * derivative, atol=1e-12
    )
    assert prepared.coordinate == "modeled-spatial"
    assert not prepared.claims_spatial_dns


def test_primitive_and_conservative_source_forms_are_algebraically_equal():
    case, _, conserved, _, snapshot = _state_and_snapshot()
    prepared = TemporalSlowGrowthModelPlan(0.08, wall_indices=()).prepare(snapshot)
    result = prepared.evaluate(conserved)
    primitive = case.conserved_to_primitive(conserved)
    primitive_source = result.source.primitive
    expected_conservative = jax.jvp(
        case.primitive_to_conserved,
        (primitive,),
        (primitive_source,),
    )[1]
    np.testing.assert_allclose(
        result.source.species_mass,
        expected_conservative[..., : case.species_count],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.source.mass,
        jnp.sum(expected_conservative[..., : case.species_count], axis=-1),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.source.momentum,
        expected_conservative[..., case.species_count : -1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.source.total_energy, expected_conservative[..., -1], atol=1e-6
    )
    primitive_result = prepared.evaluate_primitive(primitive)
    np.testing.assert_allclose(
        primitive_result.source.conservative, result.source.conservative, atol=1e-12
    )


def test_adiabatic_and_isothermal_wall_thermal_constraints_are_distinct():
    _, _, _, _, snapshot = _state_and_snapshot()
    adiabatic = TemporalSlowGrowthModelPlan(0.1, wall_thermal_mode="adiabatic").prepare(
        snapshot
    )
    isothermal = TemporalSlowGrowthModelPlan(0.1, wall_thermal_mode="isothermal").prepare(
        snapshot
    )
    np.testing.assert_allclose(
        adiabatic.wall_temperature_source_profile[0],
        adiabatic.wall_temperature_source_profile[1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        isothermal.wall_temperature_source_profile[0], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(adiabatic.wall_thermal_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(isothermal.wall_thermal_residual, 0.0, atol=1e-12)


def test_displacement_and_momentum_integral_constraints_are_enforced():
    _, _, conserved, _, snapshot = _state_and_snapshot()
    prepared = TemporalSlowGrowthModelPlan(
        0.04,
        wall_indices=(),
        displacement_thickness_rate=0.012,
        momentum_thickness_rate=0.006,
    ).prepare(snapshot)
    result = prepared.evaluate(conserved)
    np.testing.assert_allclose(
        result.budget.displacement_thickness_rate, 0.012, atol=1e-10
    )
    np.testing.assert_allclose(result.budget.momentum_thickness_rate, 0.006, atol=1e-10)
    np.testing.assert_allclose(
        result.evidence.integral_constraint_residual, 0.0, atol=1e-10
    )


def test_all_stages_share_frozen_snapshot_and_acceptance_advances_it():
    case, y, conserved, plan, snapshot = _state_and_snapshot()
    continuation = SlowGrowthContinuation(snapshot, accepted_time=0.0)
    model = TemporalSlowGrowthModelPlan(0.05, wall_indices=())
    prepared = model.prepare(snapshot, continuation=continuation)
    stage_one = prepared.evaluate(conserved)
    stage_state = conserved.at[..., case.species_count].add(0.01)
    stage_two = prepared.evaluate(stage_state)
    np.testing.assert_allclose(
        stage_one.source.primitive, stage_two.source.primitive, atol=0.0
    )
    assert prepared.snapshot.snapshot_id == snapshot.snapshot_id

    next_profile = _profile(y, offset=0.03)
    next_state = case.primitive_to_conserved(
        jnp.broadcast_to(next_profile, (4,) + next_profile.shape)
    )
    next_snapshot = plan.evaluate(
        next_state,
        sample_index=1,
        sample_time=0.25,
        streamwise_location=2.1,
    )
    accepted = continuation.accept(prepared, next_snapshot)
    next_prepared = model.prepare(next_snapshot, continuation=accepted)
    assert accepted.accepted_step == 1
    assert next_prepared.snapshot.snapshot_id == next_snapshot.snapshot_id
    assert next_prepared.prepared_id != prepared.prepared_id


def test_rejected_parent_step_rolls_back_and_records_rejection_evidence():
    _, _, _, _, snapshot = _state_and_snapshot()
    continuation = SlowGrowthContinuation(snapshot, accepted_time=1.0)
    prepared = TemporalSlowGrowthModelPlan(0.03).prepare(
        snapshot, continuation=continuation
    )
    rolled_back = continuation.reject(prepared)
    evidence = continuation.rejection_evidence(prepared)
    assert rolled_back is continuation
    assert rolled_back.snapshot.snapshot_id == snapshot.snapshot_id
    assert rolled_back.accepted_step == 0
    assert not evidence.accepted
    assert not evidence.continuation_advanced
    assert evidence.parent_continuation_id == evidence.resulting_continuation_id


def test_continuation_restart_binds_exact_accepted_snapshot_and_coordinates():
    _, _, _, _, snapshot = _state_and_snapshot(sample_index=3)
    continuation = SlowGrowthContinuation(snapshot, accepted_step=3, accepted_time=0.75)
    restart = continuation.checkpoint()
    restored = SlowGrowthContinuation.from_restart(restart)
    assert restored.continuation_id == continuation.continuation_id
    assert restored.snapshot.snapshot_id == continuation.snapshot.snapshot_id
    assert restored.accepted_step == continuation.accepted_step
    assert restored.accepted_time == continuation.accepted_time


def test_fixed_snapshot_source_jvp_and_vjp_obey_adjoint_identity():
    _, _, conserved, _, snapshot = _state_and_snapshot()
    prepared = TemporalSlowGrowthModelPlan(0.07, wall_indices=()).prepare(snapshot)
    tangent = jnp.linspace(-0.2, 0.3, conserved.size).reshape(conserved.shape)
    cotangent = jnp.linspace(0.4, -0.1, conserved.size).reshape(conserved.shape)
    source_jvp = prepared.jvp(conserved, tangent)
    source_vjp = prepared.vjp(conserved, cotangent)
    np.testing.assert_allclose(
        jnp.sum(source_jvp * cotangent),
        jnp.sum(tangent * source_vjp),
        rtol=1e-11,
        atol=1e-11,
    )


def test_energy_entropy_base_residuals_and_finite_x_admission_keep_labels_separate():
    _, y, conserved, _, _ = _state_and_snapshot()
    derivative = jnp.stack(
        (
            0.004 * jnp.ones_like(y),
            0.006 * jnp.ones_like(y),
            -0.02 * y,
            jnp.zeros_like(y),
            0.015 * y,
        ),
        axis=-1,
    )
    _, _, _, _, snapshot = _state_and_snapshot(streamwise_derivative=derivative)
    prepared = SpatialSlowGrowthModelPlan(1.5, wall_indices=()).prepare(snapshot)
    result = prepared.evaluate(conserved)
    np.testing.assert_allclose(result.evidence.base_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.evidence.energy_identity_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.evidence.entropy_identity_residual, 0.0, atol=1e-12)
    reference = prepared.conservative_source(conserved)
    comparison = prepared.compare_finite_x(
        conserved,
        reference,
        reference_id="finite-x-run-42",
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    assert bool(comparison.admitted)
    assert comparison.reference_label == "finite-x-dns"
    assert comparison.model_label == "modeled-spatial-slow-growth"
    assert not comparison.claims_spatial_dns
    assert not prepared.claims_spatial_dns

    failed = prepared.compare_finite_x(
        conserved,
        reference + 0.1,
        reference_id="finite-x-run-43",
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    assert not bool(failed.admitted)
