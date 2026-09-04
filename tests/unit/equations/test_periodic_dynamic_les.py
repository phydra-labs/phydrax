#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.equations._dynamic_les import (
    AdditiveDenominatorRegularization,
    AllowSignedBackscatter,
    DynamicLESInputs,
    DynamicLESProvenance,
    DynamicSmagorinskyPlan,
    ExactDenominatorRegularization,
    GlobalDynamicLESAveraging,
    HomogeneousPlaneDynamicLESAveraging,
    LagrangianDynamicLESAveraging,
    LocalKernelDynamicLESAveraging,
    NonnegativeBackscatterClip,
)
from phydrax.equations._les_closures import LESParameterProvenance, ResolvedLESFilter
from phydrax.equations._periodic_dynamic_les import (
    PeriodicDynamicLESPlan,
    PeriodicFourierTestFilterPlan,
)
from phydrax.equations._periodic_les import PeriodicFourierGridFilterPlan


def _space(count, *, lengths=(2.0 * np.pi,) * 3):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, length) for length in lengths)
    )


def _filter(name):
    return ResolvedLESFilter(
        name,
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )


def _prepared(
    resolved,
    test,
    *,
    averaging=None,
    regularization=None,
    backscatter=None,
    ratio=(2.0, 2.0, 2.0),
    oversampling=2.0,
):
    resolved_filter = _filter("resolved retained Fourier projection")
    test_filter = _filter("coarse test Fourier projection")
    parameters = LESParameterProvenance(
        resolved_filter,
        resolved.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    dynamic_provenance = DynamicLESProvenance(parameters, test_filter, ratio)
    model = DynamicSmagorinskyPlan(
        GlobalDynamicLESAveraging() if averaging is None else averaging,
        ExactDenominatorRegularization() if regularization is None else regularization,
        AllowSignedBackscatter() if backscatter is None else backscatter,
    ).prepare(dynamic_provenance)
    plan = PeriodicDynamicLESPlan(
        model,
        PeriodicFourierGridFilterPlan(resolved_filter),
        PeriodicFourierTestFilterPlan(test_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(oversampling)
        ),
        energy_tolerance=2e-8,
    )
    return plan.prepare(
        resolved,
        test,
        phx.discretization.PeriodicLerayProjector(resolved),
    )


def _state(space):
    projector = phx.discretization.PeriodicLerayProjector(space)
    values = jax.random.normal(jax.random.PRNGKey(17), space.physical_shape + (3,))
    return projector.project(space.project(values))


def test_periodic_test_filter_is_exact_distinct_retained_projection():
    resolved = _space(8)
    test = _space(4)
    prepared = _prepared(resolved, test)
    state = _state(resolved)

    grid_filtered = prepared.grid_filter.apply(state)
    test_filtered = prepared.test_filter.apply(grid_filtered)
    transferred = prepared.test_filter.embedding(
        prepared.test_filter.test_grid_filter.apply(
            prepared.test_filter.restriction(grid_filtered)
        )
    )

    np.testing.assert_allclose(test_filtered, transferred, atol=2e-12)
    np.testing.assert_allclose(
        prepared.test_filter.apply(test_filtered), test_filtered, atol=0.0
    )
    assert jnp.linalg.norm(grid_filtered - test_filtered) > 1e-8
    assert prepared.test_filter.prepared_id != prepared.grid_filter.prepared_id
    assert prepared.test_filter.commutation_status == "commuting"
    assert prepared.test_filter.boundary_support == "periodic"


def test_periodic_adapter_recovers_synthetic_coefficient_from_constructed_model_tensor():
    resolved = _space(8)
    prepared = _prepared(resolved, _space(4))
    inputs, _, _, _ = prepared._germano_inputs(
        _state(resolved), accepted_update_mask=True
    )
    coefficient = 0.31
    synthetic = DynamicLESInputs(
        coefficient * inputs.modeled_tensor + 1.7 * jnp.eye(3),
        inputs.modeled_tensor,
        inputs.algebraic_inputs,
        inputs.provenance,
        accepted_update_mask=True,
    )

    result = prepared.dynamic_model.evaluate(synthetic)

    np.testing.assert_allclose(result.coefficient, coefficient, rtol=2e-6, atol=2e-6)
    assert (
        result.evidence.dynamic_provenance_id
        == prepared.dynamic_model.provenance.provenance_id
    )


def test_periodic_dynamic_stage_stress_rate_transfer_and_energy_identity():
    resolved = _space(8)
    prepared = _prepared(
        resolved,
        _space(4),
        regularization=AdditiveDenominatorRegularization(1e-12),
    )
    state = _state(resolved)

    stage = prepared.evaluate(state)

    assert stage.dynamic_result.evidence.finite
    assert stage.algebraic_stage.finite
    assert stage.model_result.specific_deviatoric_stress.shape[-2:] == (3, 3)
    assert stage.projected_rate.shape == state.shape
    assert stage.model_result.energy_transfer.shape == stage.leonard_tensor.shape[:-2]
    assert prepared.projector.divergence_norm(stage.projected_rate) < 2e-9
    np.testing.assert_allclose(
        stage.algebraic_stage.modal_energy_rate,
        -stage.modeled_dissipation,
        rtol=2e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        stage.algebraic_stage.energy_identity_defect, 0.0, atol=2e-8
    )


@pytest.mark.parametrize(
    ("averaging", "expected_shape"),
    (
        (GlobalDynamicLESAveraging(), ()),
        (HomogeneousPlaneDynamicLESAveraging(("x", "y")), (1, 1, 16)),
        (
            LocalKernelDynamicLESAveraging(jnp.ones((3, 3, 3), dtype=jnp.float32)),
            (16, 16, 16),
        ),
    ),
)
def test_periodic_global_plane_and_local_routes(averaging, expected_shape):
    resolved = _space(8)
    prepared = _prepared(resolved, _space(4), averaging=averaging)

    stage = prepared.evaluate(_state(resolved))

    assert stage.dynamic_result.coefficient.shape == expected_shape
    assert stage.continuation_state is None
    assert stage.accepted_update_mask.shape == ()


def test_periodic_history_mask_restart_and_no_hidden_commit():
    resolved = _space(8)
    prepared = _prepared(
        resolved,
        _space(4),
        averaging=LagrangianDynamicLESAveraging(0.25),
    )
    state = _state(resolved)
    accepted = jnp.indices((16, 16, 16)).sum(axis=0) % 2 == 0
    initial = prepared.initial_state(state, accepted_update_mask=accepted)

    first = prepared.evaluate(state, initial, accepted_update_mask=accepted)
    replay = prepared.evaluate(state, initial, accepted_update_mask=accepted)
    restarted = prepared.evaluate(
        state, first.continuation_state, accepted_update_mask=False
    )

    np.testing.assert_allclose(
        first.continuation_state.averaged_numerator,
        replay.continuation_state.averaged_numerator,
    )
    np.testing.assert_allclose(
        restarted.continuation_state.averaged_numerator,
        first.continuation_state.averaged_numerator,
    )
    np.testing.assert_array_equal(first.accepted_update_mask, accepted)
    assert initial.accepted_updates == 0
    assert first.continuation_state.accepted_updates == jnp.sum(accepted)
    assert restarted.dynamic_result.evidence.rejected_update_count == accepted.size


def test_periodic_signed_and_clipped_backscatter_are_policy_visible():
    resolved = _space(8)
    signed = _prepared(resolved, _space(4), backscatter=AllowSignedBackscatter())
    clipped = _prepared(resolved, _space(4), backscatter=NonnegativeBackscatterClip())
    inputs, _, _, _ = signed._germano_inputs(_state(resolved), accepted_update_mask=True)
    synthetic_signed = DynamicLESInputs(
        -0.2 * inputs.modeled_tensor,
        inputs.modeled_tensor,
        inputs.algebraic_inputs,
        inputs.provenance,
        accepted_update_mask=True,
    )
    synthetic_clipped = DynamicLESInputs(
        synthetic_signed.leonard_tensor,
        synthetic_signed.modeled_tensor,
        synthetic_signed.algebraic_inputs,
        clipped.dynamic_model.provenance,
        accepted_update_mask=True,
    )

    signed_result = signed.dynamic_model.evaluate(synthetic_signed)
    clipped_result = clipped.dynamic_model.evaluate(synthetic_clipped)

    np.testing.assert_allclose(signed_result.coefficient, -0.2, atol=2e-7)
    np.testing.assert_allclose(clipped_result.coefficient, 0.0, atol=0.0)
    assert signed_result.prepared_algebraic_stress.energy_transfer.min() < 0.0
    assert clipped_result.prepared_algebraic_stress.energy_transfer.min() == 0.0
    assert clipped_result.evidence.backscatter_activity_count == 1


def test_periodic_adapter_is_jittable_and_has_finite_jvp():
    resolved = _space(8)
    prepared = _prepared(
        resolved,
        _space(4),
        regularization=AdditiveDenominatorRegularization(1e-8),
    )
    state = _state(resolved)

    rate = jax.jit(lambda value: prepared.evaluate(value).projected_rate)(state)
    _, tangent = jax.jvp(
        lambda value: jnp.sum(prepared.evaluate(value).model_result.energy_transfer),
        (state,),
        (0.01 * state,),
    )

    assert rate.shape == state.shape
    assert jnp.all(jnp.isfinite(rate))
    assert jnp.isfinite(tangent)


def test_periodic_dynamic_prepare_refuses_unsupported_filter_routes():
    resolved = _space(8)
    with pytest.raises(ValueError, match="strictly coarser"):
        _prepared(resolved, _space(8), ratio=(1.01, 1.01, 1.01))
    with pytest.raises(ValueError, match="resolution ratio"):
        _prepared(resolved, _space(4), ratio=(2.1, 2.0, 2.0))
    with pytest.raises(ValueError, match="identical lengths"):
        _prepared(resolved, _space(4, lengths=(4.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi)))

    resolved_filter = _filter("resolved")
    test_filter = _filter("test")
    parameters = LESParameterProvenance(
        resolved_filter,
        resolved.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    dynamic = DynamicSmagorinskyPlan(
        GlobalDynamicLESAveraging(),
        ExactDenominatorRegularization(),
        AllowSignedBackscatter(),
    ).prepare(DynamicLESProvenance(parameters, test_filter, 2.0))
    with pytest.raises(ValueError, match="oversampling"):
        PeriodicDynamicLESPlan(
            dynamic,
            PeriodicFourierGridFilterPlan(resolved_filter),
            PeriodicFourierTestFilterPlan(test_filter),
            phx.discretization.PseudospectralMethodPlan(
                dealiasing=phx.discretization.PaddingDealiasingPlan(2)
            ),
        )


def test_periodic_compiler_consumes_dynamic_les_rate_and_evidence():
    resolved = _space(8)
    test = _space(4)
    adapter = _prepared(
        resolved,
        test,
        averaging=GlobalDynamicLESAveraging(),
        regularization=AdditiveDenominatorRegularization(1.0e-8),
        backscatter=NonnegativeBackscatterClip(),
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01),
        resolved,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
        dynamic_les=adapter.plan,
        dynamic_test_discretization=test,
    )
    velocity = _state(resolved)
    stage = dynamics.stage(0.0, velocity)
    diagnostics = dynamics.diagnostics(0.0, velocity)

    assert stage.algebraic_les is None
    assert stage.dynamic_les is not None
    assert stage.dynamic_les.continuation_state is None
    assert dynamics.dynamic_les.prepared_id == stage.dynamic_les.prepared_id
    np.testing.assert_allclose(
        stage.rates.sgs_rate,
        stage.rates.dynamic_les_rate,
        atol=0.0,
    )
    np.testing.assert_allclose(
        stage.rates.total_rate,
        stage.rates.molecular_rate + stage.rates.nonlinear_rate,
        atol=2.0e-11,
    )
    assert bool(diagnostics.dynamic_les_available)
    assert not bool(diagnostics.algebraic_les_available)
    assert diagnostics.dynamic_les_id == dynamics.dynamic_les.prepared_id
    assert bool(diagnostics.dynamic_evidence_finite)
    assert diagnostics.dynamic_regularization_activity_count > 0


def test_periodic_compiler_lagrangian_state_is_explicit_and_rejection_preserves_history():
    resolved = _space(8)
    test = _space(4)
    adapter = _prepared(
        resolved,
        test,
        averaging=LagrangianDynamicLESAveraging(0.25),
        backscatter=NonnegativeBackscatterClip(),
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01),
        resolved,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
        dynamic_les=adapter.plan,
        dynamic_test_discretization=test,
    )
    velocity = _state(resolved)
    continuation = dynamics.dynamic_les.initial_state(velocity)
    accepted = dynamics.stage(
        0.0,
        velocity,
        continuation_state=continuation,
        accepted_update_mask=True,
    ).dynamic_les.continuation_state
    rejected = dynamics.stage(
        0.0,
        1.1 * velocity,
        continuation_state=accepted,
        accepted_update_mask=False,
    ).dynamic_les.continuation_state

    np.testing.assert_array_equal(
        rejected.averaged_numerator, accepted.averaged_numerator
    )
    np.testing.assert_array_equal(
        rejected.averaged_denominator, accepted.averaged_denominator
    )
    np.testing.assert_array_equal(rejected.initialized_mask, accepted.initialized_mask)
    assert int(rejected.accepted_updates) == int(accepted.accepted_updates)
    assert int(rejected.rejected_updates) > int(accepted.rejected_updates)
    with pytest.raises(TypeError, match="explicit initialized continuation"):
        dynamics.stage(0.0, velocity)
