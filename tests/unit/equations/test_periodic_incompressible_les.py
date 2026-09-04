import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._periodic_les import (
    PeriodicAlgebraicLESPlan,
    PeriodicFourierGridFilterPlan,
)
from phydrax.solver._etdrk import LESStabilityGuardedETDRKMethod


def _space(count=6, *, dimension=3, lengths=None):
    lengths = (1.0,) * dimension if lengths is None else lengths
    names = ("x", "y", "z")[:dimension]
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=names,
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, length) for length in lengths)
    )


def _resolved_filter(axis_names=("x", "y", "z")):
    return ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=axis_names,
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )


def _les_plan(space, coefficient=0.16, *, oversampling=1.5):
    resolved_filter = _resolved_filter(tuple(space.plan.axis_names))
    provenance = LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    model = SmagorinskyLESPlan(coefficient).prepare(provenance)
    grid_filter = PeriodicFourierGridFilterPlan(resolved_filter)
    closure_method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.OversamplingDealiasingPlan(oversampling)
    )
    return PeriodicAlgebraicLESPlan(
        model,
        grid_filter,
        closure_method,
        energy_tolerance=2e-9,
    )


def _compiled(space, coefficient=0.16, *, forcing=None, forcing_id=None):
    problem = phx.equations.IncompressibleFlowProblem(
        3,
        0.01,
        forcing=forcing,
        forcing_id=forcing_id,
    )
    resolved_method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    return phx.equations.compile_periodic_incompressible_flow(
        problem,
        space,
        resolved_method,
        algebraic_les=_les_plan(space, coefficient),
    )


def _velocity(space, amplitude=1.0):
    x, y, z = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        space.axes[2].nodes,
        indexing="ij",
    )
    lengths = tuple(axis.length for axis in space.axes)
    return amplitude * jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * y / lengths[1]),
            jnp.sin(2.0 * jnp.pi * z / lengths[2]),
            jnp.sin(2.0 * jnp.pi * x / lengths[0]),
        ),
        axis=-1,
    )


def test_periodic_filter_constructor_widths_live_modes_and_identity():
    space = _space(count=6, lengths=(1.0, 2.0, 3.0))
    plan = PeriodicFourierGridFilterPlan(_resolved_filter())
    prepared = plan.prepare(space)
    state = space.project(_velocity(space))
    filtered = prepared.apply(state)
    expected_widths = jnp.asarray((1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0))

    np.testing.assert_allclose(
        prepared.filter_scale.directional_widths,
        expected_widths,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(prepared.apply(filtered), filtered, atol=2e-12)
    nyquist = ~prepared.live_mask
    np.testing.assert_allclose(filtered[nyquist], 0.0, atol=0.0)
    np.testing.assert_allclose(filtered[prepared.live_mask], state[prepared.live_mask])
    assert prepared.plan.resolved_filter.filter_id == plan.resolved_filter.filter_id


def test_periodic_filter_and_algebraic_les_refuse_unsupported_semantics():
    nonperiodic = ResolvedLESFilter(
        "volume grid",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="wall-bounded",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    with pytest.raises(ValueError, match="sharp Fourier projection"):
        PeriodicFourierGridFilterPlan(nonperiodic)

    space = _space()
    resolved_filter = _resolved_filter()
    provenance = LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "periodic",
        source_kind="user",
        evidence_ids=(),
    )
    model = SmagorinskyLESPlan(0.16).prepare(provenance)
    with pytest.raises(ValueError, match="OversamplingDealiasingPlan"):
        PeriodicAlgebraicLESPlan(
            model,
            PeriodicFourierGridFilterPlan(resolved_filter),
            phx.discretization.PseudospectralMethodPlan(
                dealiasing=phx.discretization.PaddingDealiasingPlan(2)
            ),
        )
    with pytest.raises(ValueError, match="at least 1.5"):
        PeriodicAlgebraicLESPlan(
            model,
            PeriodicFourierGridFilterPlan(resolved_filter),
            phx.discretization.PseudospectralMethodPlan(
                dealiasing=phx.discretization.OversamplingDealiasingPlan(1.25)
            ),
        )


def test_periodic_algebraic_les_enforces_three_dimensional_provenance():
    space_2d = _space(dimension=2)
    resolved_filter = _resolved_filter()
    provenance = LESParameterProvenance(
        resolved_filter,
        space_2d.prepared_id,
        "periodic",
        source_kind="user",
        evidence_ids=(),
    )
    plan = PeriodicAlgebraicLESPlan(
        SmagorinskyLESPlan(0.16).prepare(provenance),
        PeriodicFourierGridFilterPlan(resolved_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    with pytest.raises(ValueError, match="only in 3-D"):
        phx.equations.compile_periodic_incompressible_flow(
            phx.equations.IncompressibleFlowProblem(2, 0.01),
            space_2d,
            method,
            algebraic_les=plan,
        )


def test_periodic_les_stress_sign_modal_work_pressure_and_constraints():
    space = _space(count=6)
    compiled = _compiled(space)
    state = compiled.project_state(_velocity(space))
    stage = compiled.stage(0.0, state)
    les = stage.algebraic_les
    assert les is not None

    np.testing.assert_allclose(
        les.model_result.specific_deviatoric_stress,
        np.swapaxes(les.model_result.specific_deviatoric_stress, -1, -2),
        atol=2e-11,
    )
    assert jnp.all(les.model_result.energy_transfer >= -2e-11)
    assert les.modeled_dissipation >= 0.0
    assert les.modal_energy_rate <= 2e-9
    assert jnp.abs(les.energy_identity_defect) < 2e-9
    assert jnp.abs(les.projection_energy_defect) < 2e-9
    assert les.energy_consistent
    assert les.dissipative
    assert compiled.projector.divergence_norm(les.projected_rate) < 2e-10
    np.testing.assert_allclose(les.unprojected_rate[(0, 0, 0)], 0.0, atol=2e-12)
    np.testing.assert_allclose(
        compiled.pressure_coefficients(0.0, state),
        compiled.projector.pressure_from_unconstrained_rhs(
            stage.pressure_driving_unprojected_rate
        ),
        atol=2e-12,
    )
    pressure_without_sgs = compiled.projector.pressure_from_unconstrained_rhs(
        stage.pressure_driving_unprojected_rate - les.unprojected_rate
    )
    assert (
        jnp.linalg.norm(compiled.pressure_coefficients(0.0, state) - pressure_without_sgs)
        > 1e-12
    )

    weights = (
        compiled.algebraic_les.closure_method.dealiasing.evaluation.quadrature_weights
    )
    physical_transfer = jnp.sum(weights * les.model_result.energy_transfer)
    modal_work = jnp.real(jnp.vdot(state, les.projected_rate))
    np.testing.assert_allclose(modal_work, -physical_transfer, atol=2e-9)


def test_periodic_les_preserves_hermitian_nyquist_and_diagnostic_evidence():
    space = _space(count=6)
    compiled = _compiled(space)
    state = compiled.project_state(_velocity(space))
    coordinates = space.real_coordinates(component_shape=(3,))
    stage = compiled.stage(0.0, state)
    diagnostics = compiled.diagnostics(0.0, state)

    assert coordinates.reality_defect(stage.rates.algebraic_les_rate) < 2e-10
    assert compiled.projector.divergence_norm(stage.rates.nonlinear_rate) < 2e-10
    assert (
        jnp.max(
            jnp.abs(
                jnp.where(
                    compiled.projector.admissibility_mask[..., None],
                    0.0,
                    stage.rates.algebraic_les_rate,
                )
            )
        )
        < 2e-12
    )
    assert diagnostics.algebraic_les_available
    assert diagnostics.finite
    assert diagnostics.maximum_eddy_viscosity > 0.0
    assert jnp.abs(diagnostics.algebraic_les_energy_identity_defect) < 2e-9
    assert jnp.abs(diagnostics.energy_balance_defect) < 2e-9


def test_periodic_les_zero_coefficient_matches_no_les_rhs_and_diagnostics():
    space = _space(count=6)
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    problem = phx.equations.IncompressibleFlowProblem(3, 0.01)
    baseline = phx.equations.compile_periodic_incompressible_flow(
        problem,
        space,
        method,
    )
    zero_les = phx.equations.compile_periodic_incompressible_flow(
        problem,
        space,
        method,
        algebraic_les=_les_plan(space, 0.0),
    )
    state = baseline.project_state(_velocity(space))
    baseline_rate = baseline(0.0, state, None)
    les_rate = zero_les(0.0, state, None)
    les_stage = zero_les.algebraic_les_stage(state)

    np.testing.assert_array_equal(les_rate, baseline_rate)
    assert les_stage is not None
    np.testing.assert_array_equal(les_stage.projected_rate, jnp.zeros_like(state))
    np.testing.assert_array_equal(
        zero_les.pressure_coefficients(0.0, state),
        baseline.pressure_coefficients(0.0, state),
    )
    baseline_diagnostics = baseline.diagnostics(0.0, state)
    les_diagnostics = zero_les.diagnostics(0.0, state)
    np.testing.assert_array_equal(
        les_diagnostics.semidiscrete_energy_rate,
        baseline_diagnostics.semidiscrete_energy_rate,
    )
    np.testing.assert_array_equal(les_diagnostics.algebraic_les_dissipation, 0.0)
    coordinates = space.real_coordinates(component_shape=(3,))
    base_plan = phx.solver.ETDRKMethod(2)
    baseline_method = base_plan.prepare(
        baseline.semilinear_drift,
        coordinates=coordinates,
    )
    guarded_method = LESStabilityGuardedETDRKMethod(base_plan, safety_factor=1.0).prepare(
        zero_les, coordinates=coordinates
    )
    dt = 0.1 * zero_les.step_restriction(state).etdrk_selected
    baseline_step = baseline_method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        dt,
        None,
    )
    guarded_step = guarded_method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        dt,
        None,
    )
    np.testing.assert_array_equal(
        guarded_step.accepted_state, baseline_step.accepted_state
    )


def test_periodic_les_oversampling_is_inexact_and_distinct_from_grid_filter():
    space = _space(count=5)
    projector = phx.discretization.PeriodicLerayProjector(space)
    prepared = tuple(
        _les_plan(space, oversampling=factor).prepare(space, projector)
        for factor in (1.5, 2.0, 3.0)
    )
    evaluation_sizes = tuple(
        item.closure_method.dealiasing.evaluation.modal_shape for item in prepared
    )

    assert evaluation_sizes[0] < evaluation_sizes[1] < evaluation_sizes[2]
    assert all(not item.closure_method.dealiasing.report.exact for item in prepared)
    assert all(
        item.closure_method.dealiasing.report.maximum_polynomial_degree is None
        for item in prepared
    )
    assert len({item.closure_method.prepared_id for item in prepared}) == 3
    assert len({item.grid_filter.prepared_id for item in prepared}) == 1
    assert all(
        item.grid_filter.plan.resolved_filter.filter_id
        != item.closure_method.dealiasing.prepared_id
        for item in prepared
    )
    state = projector.project(space.project(_velocity(space)))
    reference = (
        _les_plan(space, oversampling=4.0)
        .prepare(space, projector)
        .evaluate(state)
        .projected_rate
    )
    errors = tuple(
        jnp.linalg.norm(item.evaluate(state).projected_rate - reference)
        for item in prepared
    )
    assert errors[2] <= errors[0]


def test_periodic_les_restriction_and_guard_reject_without_advancing():
    space = _space(count=6)
    compiled = _compiled(space)
    state = compiled.project_state(_velocity(space))
    restriction = compiled.step_restriction(state)
    assert restriction.finite
    assert restriction.advective > 0.0
    assert restriction.algebraic_les_diffusive > 0.0
    assert restriction.molecular_diffusive > 0.0
    assert restriction.etdrk_selected == jnp.minimum(
        restriction.advective, restriction.algebraic_les_diffusive
    )
    assert restriction.fully_explicit_selected <= restriction.molecular_diffusive

    coordinates = space.real_coordinates(component_shape=(3,))
    guard = LESStabilityGuardedETDRKMethod(phx.solver.ETDRKMethod(2), safety_factor=0.5)
    with pytest.raises(TypeError, match="HermitianSpectralCoordinates"):
        guard.prepare(compiled, coordinates=None)
    mismatched = _space(count=6, lengths=(2.0, 2.0, 2.0)).real_coordinates(
        component_shape=(3,)
    )
    with pytest.raises(ValueError, match="compiled spectral discretization"):
        guard.prepare(compiled, coordinates=mismatched)
    method = guard.prepare(compiled, coordinates=coordinates)
    changed_method = guard.prepare(
        _compiled(space, coefficient=0.17),
        coordinates=coordinates,
    )
    assert changed_method.method_id != method.method_id
    rejected = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        2.0 * restriction.etdrk_selected,
        None,
    )
    assert not rejected.successful
    np.testing.assert_array_equal(rejected.accepted_state, state)
    np.testing.assert_array_equal(rejected.candidate_state, state)


def test_periodic_les_guard_reuses_first_nonlinear_stage():
    calls = []

    def forcing(time, state, args):
        del time, args
        calls.append(None)
        return jnp.zeros_like(state)

    space = _space(count=5)
    compiled = _compiled(
        space,
        forcing=forcing,
        forcing_id="counted-zero-forcing",
    )
    state = compiled.project_state(_velocity(space))
    coordinates = space.real_coordinates(component_shape=(3,))
    prepared = LESStabilityGuardedETDRKMethod(
        phx.solver.ETDRKMethod(2), safety_factor=0.5
    ).prepare(compiled, coordinates=coordinates)
    limit = compiled.step_restriction(state).etdrk_selected
    calls.clear()
    result = prepared.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        0.1 * limit,
        None,
    )

    assert result.successful
    assert len(calls) == 2
    assert prepared.method_id != prepared.base_method.method_id
    assert (
        prepared.dynamics.algebraic_les.prepared_id == compiled.algebraic_les.prepared_id
    )


def test_periodic_les_eager_jit_and_jvp_are_finite_and_consistent():
    space = _space(count=5)
    compiled = _compiled(space)
    state = compiled.project_state(_velocity(space))
    tangent = compiled.project_state(_velocity(space, amplitude=0.1))
    eager = compiled(0.0, state, None)
    staged = jax.jit(lambda value: compiled(0.0, value, None))(state)
    primal, derivative = jax.jvp(
        lambda value: compiled(0.0, value, None),
        (state,),
        (tangent,),
    )

    np.testing.assert_allclose(staged, eager, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(primal, eager, rtol=2e-10, atol=2e-10)
    assert jnp.all(jnp.isfinite(derivative))
    assert compiled.projector.divergence_norm(derivative) < 2e-9
