#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.equations._favre_les import FavreLESFieldContract, PreparedFavreLESModel
from phydrax.equations._ksgs import KSGSCoefficients, StaticKSGSPlan
from phydrax.equations._les_closures import (
    LESFilterScale,
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._unstructured_les import (
    UnstructuredLowMachLESPlan,
    UnstructuredLowMachLESState,
)
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver._fixed_step import (
    FixedStepProblem,
    solve_fixed_step,
)
from phydrax.solver._unstructured_les import (
    _step_status,
    UNSTRUCTURED_LES_ENERGY_FAILURE,
    UNSTRUCTURED_LES_STEP_RESTRICTION,
    UnstructuredLowMachLESRestartState,
    UnstructuredLowMachLESStepInputs,
)


def _operators():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    x, y, z = vertices.T
    vertices = np.stack(
        (x + 0.13 * y + 0.04 * z, y + 0.09 * z + 0.03 * x, z + 0.07 * x),
        axis=-1,
    )
    tetrahedra = np.asarray(
        (
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (0, 2, 3, 5),
            (0, 1, 3, 6),
            (0, 1, 2, 7),
        ),
        dtype=np.int32,
    )
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=tetrahedra
    ).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    return phx.discretization.PreparedUnstructuredCollocatedOperators(
        discretization, gradient
    )


def _prepared(*, ksgs=False, coefficient=0.04):
    operators = _operators()
    discretization = operators.discretization
    resolved_filter = ResolvedLESFilter(
        "tetrahedral-control-volume",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="unstructured",
        boundary_class="wall-bounded",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "variable-density-low-mach-tetrahedral-fv",
        source_kind="user",
        evidence_ids=(),
    )
    ksgs_plan = (
        StaticKSGSPlan(
            KSGSCoefficients(0.12, 1.0, 0.8, 1.0, 8.0),
            provenance,
        )
        if ksgs
        else None
    )
    favre = PreparedFavreLESModel(
        SmagorinskyLESPlan(coefficient).prepare(provenance),
        LESFilterScale(discretization.directional_control_volume_widths()),
        FavreLESFieldContract("binary-mixture", ("a", "b")),
        0.9,
        (("a", 0.7), ("b", 0.8)),
        10.0,
        isotropic_trace_policy=("provided-sgs-kinetic-energy" if ksgs else "neglected"),
    )
    return UnstructuredLowMachLESPlan(
        favre,
        ksgs_plan=ksgs_plan,
        conservation_tolerance=3.0e-6,
    ).prepare(operators)


def _case(prepared):
    centers = prepared.operators.discretization.cell_centers
    density = jnp.full((centers.shape[0],), 1.3)
    velocity = jnp.stack(
        (
            0.18 + 0.12 * centers[:, 0] - 0.04 * centers[:, 1],
            -0.07 + 0.08 * centers[:, 1] + 0.03 * centers[:, 2],
            0.05 - 0.03 * centers[:, 0] + 0.06 * centers[:, 2],
        ),
        axis=-1,
    )
    fraction_a = 0.45 + 0.04 * centers[:, 0] - 0.02 * centers[:, 2]
    fractions = jnp.stack((fraction_a, 1.0 - fraction_a), axis=-1)
    ksgs_state = (
        prepared.plan.ksgs_plan.initialize_state(0.02 + 0.003 * centers[:, 1])
        if prepared.plan.ksgs_plan is not None
        else None
    )
    state = UnstructuredLowMachLESState(
        density,
        density[:, None] * velocity,
        density[:, None] * fractions,
        ksgs=ksgs_state,
    )
    pressure = 0.2 * centers[:, 0] - 0.13 * centers[:, 1]
    temperature = 295.0 + 2.0 * centers[:, 1] + centers[:, 2]
    inputs = UnstructuredLowMachLESStepInputs(
        temperature,
        1000.0 + 2.0 * centers[:, 0],
        jnp.stack((1005.0 * temperature, 1120.0 * temperature), axis=-1),
        0.009 + 0.001 * centers[:, 2],
        0.03 + 0.003 * centers[:, 0],
        jnp.stack(
            (0.006 + 0.001 * centers[:, 0], 0.005 + 0.001 * centers[:, 1]),
            axis=-1,
        ),
    )
    return state, pressure, inputs


def _advance(method, restart, inputs, count):
    state = restart
    for index in range(count):
        state = method.step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * method.required_step_size),
            state,
            jnp.asarray(method.required_step_size),
            inputs,
        ).accepted_state
    return state


def test_pressure_corrected_step_closes_divergence_and_uses_one_mass_flux():
    prepared = _prepared()
    state, pressure, inputs = _case(prepared)
    method = prepared.prepare_fixed_step(
        2.0e-4, pressure_tolerance=2.0e-8, pressure_iterations=300
    )
    restart = method.initialize(state, pressure, inputs)
    result = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(2.0e-4),
        inputs,
    )

    assert result.fixed_step.successful
    assert result.evidence.pressure_converged
    assert result.evidence.shared_mass_flux
    assert result.evidence.conservative
    assert result.evidence.energy_finite
    assert result.evidence.divergence_after_norm < 2.0e-8
    assert result.evidence.pressure_residual_norm < 2.0e-8
    assert result.evidence.energy_balanced
    assert result.evidence.normalized_resolved_sgs_energy_balance <= 3.0e-6
    np.testing.assert_allclose(
        result.fixed_step.accepted_state.mass_flux,
        result.rate.fluxes.mass_flux,
        rtol=0.0,
        atol=0.0,
    )
    assert jnp.linalg.norm(result.rate.enthalpy_density_rate) > 0.0
    np.testing.assert_allclose(
        result.fixed_step.accepted_state.enthalpy_density,
        restart.enthalpy_density
        + jnp.asarray(2.0e-4) * result.rate.enthalpy_density_rate,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jnp.sum(result.fixed_step.accepted_state.conservative.scalar_densities, axis=-1),
        result.fixed_step.accepted_state.conservative.density,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_static_ksgs_conservative_rate_and_energy_history_are_advanced():
    prepared = _prepared(ksgs=True)
    state, pressure, inputs = _case(prepared)
    centers = prepared.operators.discretization.cell_centers
    velocity = centers @ jnp.diag(jnp.asarray((1.0, 1.0, -2.0))) + jnp.asarray(
        (0.1, -0.03, 0.02)
    )
    state = eqx.tree_at(
        lambda value: value.momentum_density,
        state,
        state.density[:, None] * velocity,
    )
    method = prepared.prepare_fixed_step(
        1.0e-4,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    restart = method.initialize(state, pressure, inputs)
    result = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(1.0e-4),
        inputs,
    )

    assert result.fixed_step.successful
    assert result.rate.ksgs_density_rate is not None
    assert result.rate.evidence.shared_ksgs_mass_flux_residual == 0.0
    assert result.evidence.sgs_kinetic_energy_change is not None
    assert jnp.isfinite(result.evidence.sgs_kinetic_energy_change)
    assert result.evidence.energy_balanced
    assert result.evidence.normalized_resolved_sgs_energy_balance <= 3.0e-6
    assert jnp.isfinite(result.evidence.modeled_transfer_residual)
    assert result.fixed_step.accepted_state.conservative.ksgs is not None
    assert jnp.all(
        result.fixed_step.accepted_state.conservative.ksgs.kinetic_energy >= 0.0
    )


def test_static_ksgs_owns_transport_despite_favre_coefficient_mismatch():
    weak = _prepared(ksgs=True, coefficient=0.005)
    strong = _prepared(ksgs=True, coefficient=0.4)
    weak_state, weak_pressure, inputs = _case(weak)
    strong_state, strong_pressure, _ = _case(strong)
    weak_method = weak.prepare_fixed_step(
        1.0e-4,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    strong_method = strong.prepare_fixed_step(
        1.0e-4,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    weak_result = weak_method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        weak_method.initialize(weak_state, weak_pressure, inputs),
        jnp.asarray(1.0e-4),
        inputs,
    )
    strong_result = strong_method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        strong_method.initialize(strong_state, strong_pressure, inputs),
        jnp.asarray(1.0e-4),
        inputs,
    )

    assert (
        jnp.max(
            jnp.abs(
                weak_result.rate.favre.kinematic_eddy_viscosity
                - strong_result.rate.favre.kinematic_eddy_viscosity
            )
        )
        > 1.0e-4
    )
    np.testing.assert_allclose(
        weak_result.rate.kinematic_eddy_viscosity,
        strong_result.rate.kinematic_eddy_viscosity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        weak_result.rate.kinematic_eddy_viscosity,
        weak_result.rate.ksgs.eddy_viscosity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        weak_result.rate.fluxes.sgs_momentum_flux,
        strong_result.rate.fluxes.sgs_momentum_flux,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        weak_result.rate.fluxes.sgs_scalar_flux,
        strong_result.rate.fluxes.sgs_scalar_flux,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        weak_result.rate.fluxes.sgs_enthalpy_flux,
        strong_result.rate.fluxes.sgs_enthalpy_flux,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        weak_result.rate.ksgs.contributions.production,
        strong_result.rate.ksgs.contributions.production,
        rtol=0.0,
        atol=0.0,
    )
    assert not weak_result.fixed_step.successful
    assert not strong_result.fixed_step.successful
    assert not weak_result.rate.ksgs.evidence.production_nonnegative.all()
    assert not strong_result.rate.ksgs.evidence.production_nonnegative.all()
    assert weak_result.status == UNSTRUCTURED_LES_ENERGY_FAILURE
    assert strong_result.status == UNSTRUCTURED_LES_ENERGY_FAILURE


def test_two_step_ksgs_energy_balance_closes_every_modeled_contribution():
    prepared = _prepared(ksgs=True, coefficient=0.3)
    state, pressure, inputs = _case(prepared)
    centers = prepared.operators.discretization.cell_centers
    velocity = centers @ jnp.diag(jnp.asarray((1.0, 1.0, -2.0))) + jnp.asarray(
        (0.1, -0.03, 0.02)
    )
    state = eqx.tree_at(
        lambda value: value.momentum_density,
        state,
        state.density[:, None] * velocity,
    )
    method = prepared.prepare_fixed_step(
        1.0e-4,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    restart = method.initialize(state, pressure, inputs)
    first = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(1.0e-4),
        inputs,
    )
    assert jnp.linalg.norm(first.rate.fluxes.sgs_deviatoric_momentum_flux) > 0.0
    assert jnp.min(first.rate.ksgs_raw_production_density) > 0.0
    second = method.step_detailed(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1.0e-4),
        first.fixed_step.accepted_state,
        jnp.asarray(1.0e-4),
        inputs,
    )

    assert first.fixed_step.successful
    assert second.fixed_step.successful
    for result, current in (
        (first, restart),
        (second, first.fixed_step.accepted_state),
    ):
        assert result.evidence.energy_balanced
        assert result.evidence.normalized_resolved_sgs_energy_balance <= 3.0e-6
        assert result.evidence.normalized_modeled_transfer_residual <= 3.0e-6
        assert jnp.max(result.rate.ksgs_production_limit_reduction_density) > 0.0
        np.testing.assert_allclose(
            result.rate.modeled_enthalpy_source_density,
            result.rate.ksgs_production_limit_reduction_density,
            rtol=0.0,
            atol=0.0,
        )
        assert result.evidence.production_limit_thermalization_rate > 0.0
        enthalpy_gain = jnp.sum(
            prepared.operators.discretization.cell_volumes
            * (
                result.fixed_step.candidate_state.enthalpy_density
                - current.enthalpy_density
            )
        )
        np.testing.assert_allclose(
            enthalpy_gain,
            jnp.asarray(1.0e-4) * result.evidence.production_limit_thermalization_rate,
            rtol=2.0e-8,
            atol=2.0e-8,
        )
        assert jnp.abs(result.rate.evidence.modeled_energy_split_residual) <= 3.0e-6
        expected_rhs = (
            result.rate.ksgs_production_density / current.conservative.density
            - result.rate.ksgs.contributions.dissipation
            + result.rate.ksgs.contributions.diffusion
            + result.rate.ksgs.contributions.buoyancy
            - result.rate.ksgs.contributions.low_re_dissipation
        )
        np.testing.assert_allclose(
            result.rate.ksgs.contributions.rhs,
            expected_rhs,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        assert jnp.isfinite(result.evidence.pressure_work_rate)
        assert jnp.isfinite(result.evidence.molecular_viscous_work_rate)
        assert jnp.isfinite(result.evidence.sgs_stress_work_rate)
        assert jnp.isfinite(result.evidence.ksgs_source_rate)
        assert jnp.isfinite(result.evidence.temporal_energy_defect)


def test_inconsistent_modeled_transfer_is_rejected_with_energy_status():
    prepared = _prepared(ksgs=True)
    state, pressure, inputs = _case(prepared)
    method = prepared.prepare_fixed_step(
        1.0e-4,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    restart = method.initialize(state, pressure, inputs)
    result = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(1.0e-4),
        inputs,
    )
    inconsistent_rate = eqx.tree_at(
        lambda value: value.ksgs_raw_production_density,
        result.rate,
        1.25 * result.rate.ksgs_raw_production_density,
    )
    evidence = method._evidence(
        restart,
        result.fixed_step.candidate_state,
        inconsistent_rate,
        result.pressure,
        result.restriction,
        jnp.asarray(1.0e-4),
    )
    rolled_back = jax.tree.map(
        lambda candidate, current: jnp.where(evidence.successful, candidate, current),
        result.fixed_step.candidate_state,
        restart,
    )

    assert not evidence.modeled_transfer_balanced
    assert not evidence.energy_balanced
    assert not evidence.successful
    assert _step_status(evidence) == UNSTRUCTURED_LES_ENERGY_FAILURE
    for actual, expected in zip(
        jax.tree.leaves(rolled_back),
        jax.tree.leaves(restart),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    negative_rate = eqx.tree_at(
        lambda value: (
            value.ksgs_raw_production_density,
            value.ksgs.evidence.production_nonnegative,
        ),
        result.rate,
        (
            -jnp.abs(result.rate.ksgs_raw_production_density),
            jnp.zeros_like(
                result.rate.ksgs.evidence.production_nonnegative,
                dtype=bool,
            ),
        ),
    )
    negative_evidence = method._evidence(
        restart,
        result.fixed_step.candidate_state,
        negative_rate,
        result.pressure,
        result.restriction,
        jnp.asarray(1.0e-4),
    )
    assert not negative_evidence.modeled_transfer_balanced
    assert not negative_evidence.successful
    assert _step_status(negative_evidence) == UNSTRUCTURED_LES_ENERGY_FAILURE


def test_noncoercive_algebraic_sgs_work_is_rejected_and_rolled_back():
    prepared = _prepared()
    state, pressure, inputs = _case(prepared)
    method = prepared.prepare_fixed_step(
        1.0e-4,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    restart = method.initialize(state, pressure, inputs)
    result = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(1.0e-4),
        inputs,
    )
    noncoercive_rate = eqx.tree_at(
        lambda value: value.fluxes.sgs_deviatoric_momentum_flux,
        result.rate,
        -result.rate.fluxes.sgs_deviatoric_momentum_flux,
    )
    evidence = method._evidence(
        restart,
        result.fixed_step.candidate_state,
        noncoercive_rate,
        result.pressure,
        result.restriction,
        jnp.asarray(1.0e-4),
    )
    rolled_back = jax.tree.map(
        lambda candidate, current: jnp.where(evidence.successful, candidate, current),
        result.fixed_step.candidate_state,
        restart,
    )

    assert result.fixed_step.successful
    assert result.evidence.sgs_work_dissipative
    assert evidence.normalized_positive_sgs_work > 3.0e-6
    assert not evidence.sgs_work_dissipative
    assert not evidence.successful
    assert _step_status(evidence) == UNSTRUCTURED_LES_ENERGY_FAILURE
    for actual, expected in zip(
        jax.tree.leaves(rolled_back),
        jax.tree.leaves(restart),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_rejected_step_rolls_back_every_restart_leaf_and_preserves_history():
    prepared = _prepared()
    state, pressure, inputs = _case(prepared)
    method = prepared.prepare_fixed_step(
        2.0e-4,
        maximum_courant_number=1.0e-12,
        maximum_diffusion_number=1.0e-12,
        pressure_tolerance=2.0e-8,
        pressure_iterations=300,
    )
    restart = method.initialize(state, pressure, inputs)
    result = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(2.0e-4),
        inputs,
    )

    assert not result.fixed_step.successful
    assert result.status == UNSTRUCTURED_LES_STEP_RESTRICTION
    assert not result.evidence.step_stable
    for accepted, initial in zip(
        jax.tree.leaves(result.fixed_step.accepted_state),
        jax.tree.leaves(restart),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted, initial)
    assert result.fixed_step.candidate_state.accepted_steps == 1
    assert result.fixed_step.accepted_state.accepted_steps == 0


def test_restart_continuation_is_deterministic_and_jittable_with_momentum_jvp():
    prepared = _prepared()
    state, pressure, inputs = _case(prepared)
    method = prepared.prepare_fixed_step(
        1.0e-4, pressure_tolerance=2.0e-8, pressure_iterations=300
    )
    restart = method.initialize(state, pressure, inputs)
    first_detailed = method.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        restart,
        jnp.asarray(1.0e-4),
        inputs,
    )
    first = first_detailed.fixed_step.accepted_state
    reconstructed_rate = method._rate(
        first.conservative,
        first.pressure,
        inputs,
        jnp.asarray(method.required_step_size) / first.conservative.density,
        face_normal_velocity=first.face_normal_velocity,
    )
    np.testing.assert_array_equal(
        reconstructed_rate.fluxes.face_normal_velocity,
        first.face_normal_velocity,
    )
    discretization = prepared.operators.discretization
    first_divergence = prepared.operators.divergence(first.face_normal_velocity)
    assert jnp.sqrt(jnp.sum(discretization.cell_volumes * first_divergence**2)) < 2.0e-8
    reconstructed_restart = UnstructuredLowMachLESRestartState(
        first.conservative,
        first.enthalpy_density,
        first.pressure,
        first.face_normal_velocity,
        first.mass_flux,
        first.pressure_increment,
        first.accepted_steps,
    )
    eager_detailed = method.step_detailed(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1.0e-4),
        first,
        jnp.asarray(1.0e-4),
        inputs,
    )
    eager = eager_detailed.fixed_step
    restarted = method.step(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1.0e-4),
        reconstructed_restart,
        jnp.asarray(1.0e-4),
        inputs,
    )
    compiled = eqx.filter_jit(method.step)(
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1.0e-4),
        first,
        jnp.asarray(1.0e-4),
        inputs,
    )
    assert eager.successful
    assert first_detailed.evidence.energy_balanced
    assert eager_detailed.evidence.energy_balanced
    assert first_detailed.evidence.normalized_resolved_sgs_energy_balance <= 3.0e-6
    assert eager_detailed.evidence.normalized_resolved_sgs_energy_balance <= 3.0e-6
    np.testing.assert_array_equal(
        eager_detailed.predictor_rate.fluxes.face_normal_velocity,
        first.face_normal_velocity,
    )
    assert compiled.successful
    assert eager.accepted_state.accepted_steps == 2
    assert restarted.successful
    second_divergence = prepared.operators.divergence(
        eager.accepted_state.face_normal_velocity
    )
    assert jnp.sqrt(jnp.sum(discretization.cell_volumes * second_divergence**2)) < 2.0e-8
    for actual, expected in zip(
        jax.tree.leaves(restarted.accepted_state),
        jax.tree.leaves(eager.accepted_state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        jax.tree.leaves(compiled.accepted_state),
        jax.tree.leaves(eager.accepted_state),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=3.0e-9, atol=3.0e-9)
    rollout = solve_fixed_step(
        FixedStepProblem(
            method,
            restart,
            t0=0.0,
            t1=2.0e-4,
            step_size=1.0e-4,
            args=inputs,
            state_geometry=EuclideanStateGeometry(),
        )
    )
    assert rollout.successful
    assert rollout.states.accepted_steps[-1] == 2

    def advance_momentum(momentum):
        perturbed = eqx.tree_at(
            lambda value: value.conservative.momentum_density,
            first,
            momentum,
        )
        return method.step(
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1.0e-4),
            perturbed,
            jnp.asarray(1.0e-4),
            inputs,
        ).accepted_state.conservative.momentum_density

    _, tangent = jax.jvp(
        advance_momentum,
        (first.conservative.momentum_density,),
        (jnp.ones_like(first.conservative.momentum_density) * 1.0e-3,),
    )
    assert jnp.all(jnp.isfinite(tangent))


def test_manufactured_temporal_refinement_converges_to_the_same_transition():
    prepared = _prepared()
    state, pressure, inputs = _case(prepared)
    coarse = prepared.prepare_fixed_step(
        2.0e-4, pressure_tolerance=2.0e-8, pressure_iterations=300
    )
    medium = prepared.prepare_fixed_step(
        1.0e-4, pressure_tolerance=2.0e-8, pressure_iterations=300
    )
    reference = prepared.prepare_fixed_step(
        5.0e-5, pressure_tolerance=2.0e-8, pressure_iterations=300
    )
    coarse_state = _advance(coarse, coarse.initialize(state, pressure, inputs), inputs, 1)
    medium_state = _advance(medium, medium.initialize(state, pressure, inputs), inputs, 2)
    reference_state = _advance(
        reference, reference.initialize(state, pressure, inputs), inputs, 4
    )
    coarse_error = jnp.linalg.norm(
        coarse_state.conservative.scalar_densities
        - reference_state.conservative.scalar_densities
    )
    medium_error = jnp.linalg.norm(
        medium_state.conservative.scalar_densities
        - reference_state.conservative.scalar_densities
    )

    assert medium_error < coarse_error
    assert coarse_error / medium_error > 1.4
