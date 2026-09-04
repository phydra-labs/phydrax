#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.equations._dynamic_les import (
    AdditiveDenominatorRegularization,
    AllowSignedBackscatter,
    BoundedFractionBackscatter,
    DynamicLESInputs,
    DynamicLESProvenance,
    DynamicSmagorinskyPlan,
    ExactDenominatorRegularization,
    GlobalDynamicLESAveraging,
    HomogeneousPlaneDynamicLESAveraging,
    LagrangianDynamicLESAveraging,
    LagrangianDynamicLESState,
    LocalKernelDynamicLESAveraging,
    NonnegativeBackscatterClip,
)
from phydrax.equations._les_closures import (
    AlgebraicLESInputs,
    LESFilterScale,
    LESParameterProvenance,
    ResolvedLESFilter,
)


def _filter(
    name,
    *,
    family="implicit-grid-volume",
    axis_names=("x", "y", "z"),
    topology="tensor-product",
    boundary_class="periodic",
    commutation_status="unmodeled",
    repeated_filter_semantics="unmodeled",
):
    scale_rule = {
        "implicit-grid-volume": "volume-equivalent",
        "explicit-filter": "kernel-equivalent",
        "sharp-fourier-projection": "cutoff-equivalent",
    }[family]
    return ResolvedLESFilter(
        name,
        family=family,
        axis_names=axis_names,
        topology=topology,
        boundary_class=boundary_class,
        scale_rule=scale_rule,
        commutation_status=commutation_status,
        repeated_filter_semantics=repeated_filter_semantics,
    )


def _provenance(
    *,
    resolved_filter=None,
    test_filter=None,
    ratio=(2.0, 2.0, 2.0),
):
    if resolved_filter is None:
        resolved_filter = _filter("resolved")
    if test_filter is None:
        test_filter = _filter(
            "test",
            family="explicit-filter",
            axis_names=resolved_filter.axis_names,
            topology=resolved_filter.topology,
            boundary_class=resolved_filter.boundary_class,
            commutation_status=resolved_filter.commutation_status,
            repeated_filter_semantics="composed",
        )
    parameter_provenance = LESParameterProvenance(
        resolved_filter,
        "grid-a",
        "dynamic-les",
        source_kind="user",
        evidence_ids=(),
    )
    return DynamicLESProvenance(parameter_provenance, test_filter, ratio)


def _inputs(
    leonard,
    modeled,
    *,
    gradient=None,
    widths=(0.2, 0.3, 0.4),
    provenance=None,
    accepted=True,
):
    leonard = jnp.asarray(leonard)
    modeled = jnp.asarray(modeled)
    if gradient is None:
        gradient = jnp.zeros_like(leonard)
    if provenance is None:
        provenance = _provenance()
    algebraic_inputs = AlgebraicLESInputs(
        jnp.asarray(gradient), LESFilterScale(jnp.asarray(widths))
    )
    return DynamicLESInputs(
        leonard,
        modeled,
        algebraic_inputs,
        provenance,
        accepted_update_mask=jnp.asarray(accepted),
    )


def _prepared(
    provenance,
    *,
    averaging=None,
    regularization=None,
    backscatter=None,
):
    if averaging is None:
        averaging = GlobalDynamicLESAveraging()
    if regularization is None:
        regularization = ExactDenominatorRegularization()
    if backscatter is None:
        backscatter = AllowSignedBackscatter()
    return DynamicSmagorinskyPlan(averaging, regularization, backscatter).prepare(
        provenance
    )


def _trace_free_basis():
    return jnp.asarray(((1.0, 0.25, -0.5), (0.25, -0.4, 0.3), (-0.5, 0.3, -0.6)))


def test_exact_global_germano_coefficient_and_ready_stress():
    provenance = _provenance(ratio=(2.0, 2.5, 3.0))
    modeled = jnp.stack((_trace_free_basis(), 2.0 * _trace_free_basis()))
    coefficient = 0.37
    isotropic = jnp.stack((1.7 * jnp.eye(3), -0.4 * jnp.eye(3)))
    leonard = coefficient * modeled + isotropic
    gradient = jnp.asarray(
        (
            ((-0.7, 0.4, 0.1), (-0.2, -0.3, 0.5), (0.2, -0.1, -0.6)),
            ((0.2, -0.1, 0.3), (0.5, -0.4, 0.2), (-0.2, 0.1, 0.1)),
        )
    )
    inputs = _inputs(leonard, modeled, gradient=gradient, provenance=provenance)
    prepared = _prepared(provenance)

    result = prepared.evaluate(inputs)

    np.testing.assert_allclose(result.coefficient, coefficient, rtol=2e-6)
    assert result.evidence.dynamic_provenance_id == provenance.provenance_id
    assert result.evidence.prepared_id == prepared.prepared_id
    assert result.evidence.zero_denominator_count == 0
    assert result.evidence.regularization_activity_count == 0
    assert bool(result.evidence.finite)
    stress_result = result.prepared_algebraic_stress
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    strain_deviatoric = (
        strain - jnp.trace(strain, axis1=-2, axis2=-1)[..., None, None] * jnp.eye(3) / 3.0
    )
    delta = np.cbrt(0.2 * 0.3 * 0.4)
    expected_viscosity = (
        coefficient * delta**2 * jnp.sqrt(2.0 * jnp.sum(strain * strain, axis=(-2, -1)))
    )
    expected_stress = -2.0 * expected_viscosity[..., None, None] * strain_deviatoric
    np.testing.assert_allclose(
        stress_result.kinematic_viscosity, expected_viscosity, rtol=3e-6
    )
    np.testing.assert_allclose(
        stress_result.specific_deviatoric_stress, expected_stress, rtol=3e-6
    )
    np.testing.assert_allclose(
        stress_result.energy_transfer,
        -jnp.sum(expected_stress * strain, axis=(-2, -1)),
        rtol=3e-6,
    )


def test_exact_zero_denominator_is_finite_zero_without_floor():
    provenance = _provenance()
    inputs = _inputs(jnp.eye(3), jnp.zeros((3, 3)), provenance=provenance)

    result = _prepared(provenance).evaluate(inputs)

    assert result.coefficient == 0.0
    assert result.evidence.averaged_denominator == 0.0
    assert result.evidence.effective_denominator == 0.0
    assert result.evidence.zero_denominator_count == 1
    assert result.evidence.regularization_activity_count == 0
    np.testing.assert_array_equal(
        result.prepared_algebraic_stress.specific_deviatoric_stress,
        jnp.zeros((3, 3)),
    )
    assert bool(result.evidence.finite)


def test_additive_regularization_is_explicit_smooth_and_counted():
    provenance = _provenance()
    modeled = jnp.diag(jnp.asarray((1.0, -1.0, 0.0)))
    inputs = _inputs(4.0 * modeled, modeled, provenance=provenance)
    prepared = _prepared(
        provenance,
        regularization=AdditiveDenominatorRegularization(2.0),
    )

    result = prepared.evaluate(inputs)

    # M:M = 2, so (L:M) / (M:M + shift) = 8 / 4.
    np.testing.assert_allclose(result.coefficient, 2.0)
    np.testing.assert_allclose(result.evidence.effective_denominator, 4.0)
    assert result.evidence.regularization_activity_count == 1
    assert result.evidence.regularization == "additive-tikhonov"
    assert result.evidence.differentiation == "smooth"


def test_global_and_homogeneous_axis_averaging_are_ratio_of_averages():
    provenance = _provenance()
    basis = jnp.diag(jnp.asarray((1.0, -1.0, 0.0)))
    modeled = jnp.broadcast_to(basis, (2, 3, 4, 3, 3))
    coefficients = jnp.arange(24.0).reshape(2, 3, 4)
    leonard = coefficients[..., None, None] * modeled
    inputs = _inputs(leonard, modeled, provenance=provenance)

    global_result = _prepared(provenance).evaluate(inputs)
    plane_result = _prepared(
        provenance,
        averaging=HomogeneousPlaneDynamicLESAveraging(("x", "z")),
    ).evaluate(inputs)

    np.testing.assert_allclose(global_result.coefficient, jnp.mean(coefficients))
    assert plane_result.coefficient.shape == (1, 3, 1)
    np.testing.assert_allclose(
        plane_result.coefficient, jnp.mean(coefficients, axis=(0, 2), keepdims=True)
    )


def test_local_fixed_kernel_averaging_is_periodic_and_normalized():
    provenance = _provenance()
    basis = jnp.diag(jnp.asarray((1.0, -1.0, 0.0)))
    modeled = jnp.broadcast_to(basis, (3, 1, 1, 3, 3))
    coefficients = jnp.asarray((0.0, 3.0, 6.0)).reshape(3, 1, 1)
    inputs = _inputs(
        coefficients[..., None, None] * modeled,
        modeled,
        provenance=provenance,
    )
    averaging = LocalKernelDynamicLESAveraging(jnp.ones((3, 1, 1)))

    result = _prepared(provenance, averaging=averaging).evaluate(inputs)

    np.testing.assert_allclose(result.coefficient, jnp.full((3, 1, 1), 3.0))
    np.testing.assert_allclose(np.sum(np.asarray(averaging.kernel_weights)), 1.0)


def test_lagrangian_mask_accepts_rejects_and_blends_immutable_history():
    provenance = _provenance()
    averaging = LagrangianDynamicLESAveraging(0.25)
    prepared = _prepared(provenance, averaging=averaging)
    basis = jnp.diag(jnp.asarray((1.0, -1.0, 0.0)))
    modeled = jnp.broadcast_to(basis, (2, 3, 3))
    first_inputs = _inputs(
        jnp.asarray((2.0, 5.0))[..., None, None] * modeled,
        modeled,
        provenance=provenance,
        accepted=jnp.asarray((True, False)),
    )
    initial = prepared.initial_state(first_inputs)

    first = prepared.evaluate(first_inputs, initial)
    np.testing.assert_allclose(first.coefficient, (2.0, 0.0))
    assert first.evidence.accepted_update_count == 1
    assert first.evidence.rejected_update_count == 1
    assert first.continuation_state.accepted_updates == 1
    assert first.continuation_state.rejected_updates == 1
    np.testing.assert_array_equal(initial.initialized_mask, (False, False))

    second_inputs = _inputs(
        jnp.asarray((9.0, 6.0))[..., None, None] * modeled,
        modeled,
        provenance=provenance,
        accepted=jnp.asarray((False, True)),
    )
    second = prepared.evaluate(second_inputs, first.continuation_state)
    np.testing.assert_allclose(second.coefficient, (2.0, 6.0))

    third_inputs = _inputs(
        jnp.asarray((6.0, 10.0))[..., None, None] * modeled,
        modeled,
        provenance=provenance,
        accepted=jnp.asarray((True, True)),
    )
    third = prepared.evaluate(third_inputs, second.continuation_state)
    np.testing.assert_allclose(third.coefficient, (3.0, 7.0))
    assert third.continuation_state.accepted_updates == 4
    assert third.continuation_state.rejected_updates == 2
    assert third.evidence.differentiation == "branchwise"


def test_lagrangian_restart_state_reconstructs_identical_continuation():
    provenance = _provenance()
    prepared = _prepared(provenance, averaging=LagrangianDynamicLESAveraging(0.4))
    modeled = _trace_free_basis()
    first_inputs = _inputs(2.0 * modeled, modeled, provenance=provenance)
    first = prepared.evaluate(first_inputs, prepared.initial_state(first_inputs))
    state = first.continuation_state
    restarted = LagrangianDynamicLESState(
        state.averaged_numerator,
        state.averaged_denominator,
        state.initialized_mask,
        state.accepted_updates,
        state.rejected_updates,
        state.continuation_id,
    )
    next_inputs = _inputs(5.0 * modeled, modeled, provenance=provenance)

    uninterrupted = prepared.evaluate(next_inputs, state)
    resumed = prepared.evaluate(next_inputs, restarted)

    np.testing.assert_array_equal(resumed.coefficient, uninterrupted.coefficient)
    np.testing.assert_array_equal(
        resumed.continuation_state.averaged_numerator,
        uninterrupted.continuation_state.averaged_numerator,
    )
    assert (
        resumed.continuation_state.continuation_id
        == uninterrupted.continuation_state.continuation_id
    )


def test_signed_clipped_and_bounded_backscatter_are_explicit_and_counted():
    provenance = _provenance()
    modeled = _trace_free_basis()
    inputs = _inputs(
        -2.0 * modeled,
        modeled,
        gradient=jnp.diag(jnp.asarray((1.0, -0.5, -0.5))),
        provenance=provenance,
    )

    signed = _prepared(provenance, backscatter=AllowSignedBackscatter()).evaluate(inputs)
    clipped = _prepared(provenance, backscatter=NonnegativeBackscatterClip()).evaluate(
        inputs
    )
    bounded = _prepared(
        provenance,
        backscatter=BoundedFractionBackscatter(0.25, 4.0),
    ).evaluate(inputs)

    np.testing.assert_allclose(signed.coefficient, -2.0)
    assert signed.prepared_algebraic_stress.kinematic_viscosity < 0.0
    assert signed.prepared_algebraic_stress.energy_transfer < 0.0
    assert signed.evidence.backscatter_activity_count == 1
    assert signed.evidence.backscatter_limit_count == 0
    assert clipped.coefficient == 0.0
    assert clipped.evidence.backscatter_activity_count == 1
    assert clipped.evidence.backscatter_limit_count == 1
    np.testing.assert_allclose(bounded.coefficient, -1.0)
    assert bounded.evidence.backscatter_activity_count == 1
    assert bounded.evidence.backscatter_limit_count == 1


def test_coordinate_permutation_preserves_inference_and_permutes_stress():
    axes = ("x", "y", "z")
    permutation = np.asarray((2, 0, 1))
    provenance = _provenance(ratio=(2.0, 2.5, 3.0))
    modeled = _trace_free_basis()
    leonard = 0.23 * modeled + 0.8 * jnp.eye(3)
    gradient = jnp.asarray(((-0.7, 0.4, 0.1), (-0.2, -0.3, 0.5), (0.2, -0.1, -0.6)))
    original = _prepared(provenance).evaluate(
        _inputs(
            leonard,
            modeled,
            gradient=gradient,
            widths=(0.12, 0.25, 0.4),
            provenance=provenance,
        )
    )

    permuted_axes = tuple(axes[index] for index in permutation)
    permuted_provenance = _provenance(
        resolved_filter=_filter("resolved", axis_names=permuted_axes),
        test_filter=_filter(
            "test",
            family="explicit-filter",
            axis_names=permuted_axes,
            repeated_filter_semantics="composed",
        ),
        ratio=np.asarray((2.0, 2.5, 3.0))[permutation],
    )
    permuted = _prepared(permuted_provenance).evaluate(
        _inputs(
            leonard[permutation][:, permutation],
            modeled[permutation][:, permutation],
            gradient=gradient[permutation][:, permutation],
            widths=np.asarray((0.12, 0.25, 0.4))[permutation],
            provenance=permuted_provenance,
        )
    )

    np.testing.assert_allclose(permuted.coefficient, original.coefficient, rtol=2e-6)
    original_stress = original.prepared_algebraic_stress.specific_deviatoric_stress
    np.testing.assert_allclose(
        permuted.prepared_algebraic_stress.specific_deviatoric_stress,
        original_stress[permutation][:, permutation],
        rtol=3e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        permuted.prepared_algebraic_stress.energy_transfer,
        original.prepared_algebraic_stress.energy_transfer,
        rtol=3e-6,
    )


def test_smooth_path_is_jittable_and_has_expected_jvp():
    provenance = _provenance()
    modeled = _trace_free_basis()
    denominator = jnp.sum(modeled * modeled)
    shift = 0.7
    prepared = _prepared(
        provenance,
        regularization=AdditiveDenominatorRegularization(shift),
        backscatter=AllowSignedBackscatter(),
    )
    algebraic_inputs = AlgebraicLESInputs(
        jnp.diag(jnp.asarray((1.0, -0.5, -0.5))),
        LESFilterScale(jnp.asarray((0.2, 0.3, 0.4))),
    )

    def infer(scale):
        inputs = DynamicLESInputs(
            scale * modeled,
            modeled,
            algebraic_inputs,
            provenance,
            accepted_update_mask=jnp.asarray(True),
        )
        return prepared.evaluate(inputs).coefficient

    expected_derivative = denominator / (denominator + shift)
    compiled = jax.jit(infer)(1.3)
    primal, tangent = jax.jvp(infer, (jnp.asarray(1.3),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(compiled, 1.3 * expected_derivative, rtol=2e-6)
    np.testing.assert_allclose(primal, compiled, rtol=2e-6)
    np.testing.assert_allclose(tangent, expected_derivative, rtol=2e-6)


def test_exact_zero_branch_has_finite_zero_jvp():
    provenance = _provenance()
    modeled = jnp.zeros((3, 3))
    prepared = _prepared(provenance)

    def infer(scale):
        return prepared.evaluate(
            _inputs(scale * jnp.eye(3), modeled, provenance=provenance)
        ).coefficient

    primal, tangent = jax.jvp(infer, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    assert primal == 0.0
    assert tangent == 0.0
    assert jnp.isfinite(tangent)


def test_filter_pair_refuses_identity_axis_and_semantic_mismatches():
    resolved = _filter("resolved")
    parameter_provenance = LESParameterProvenance(
        resolved, "grid-a", "dynamic", source_kind="user", evidence_ids=()
    )
    with pytest.raises(ValueError, match="distinct"):
        DynamicLESProvenance(parameter_provenance, resolved, 2.0)

    incompatible = (
        _filter("test", family="explicit-filter", axis_names=("z", "y", "x")),
        _filter("test", family="explicit-filter", topology="unstructured"),
        _filter("test", family="explicit-filter", boundary_class="wall-bounded"),
        _filter("test", family="explicit-filter", commutation_status="modeled"),
    )
    for test_filter in incompatible:
        with pytest.raises(ValueError):
            DynamicLESProvenance(parameter_provenance, test_filter, 2.0)


@pytest.mark.parametrize(
    "ratio",
    (
        (2.0, 2.0),
        (2.0, 2.0, 2.0, 2.0),
        (1.0, 2.0, 2.0),
        (2.0, -3.0, 2.0),
        (2.0, np.inf, 2.0),
        np.nan,
    ),
)
def test_test_filter_ratio_refuses_invalid_shape_scale_and_finiteness(ratio):
    with pytest.raises(ValueError):
        _provenance(ratio=ratio)


def test_dynamic_inputs_refuse_shape_finite_scale_and_mask_mismatches():
    provenance = _provenance()
    basis = _trace_free_basis()
    algebraic = AlgebraicLESInputs(basis, LESFilterScale(jnp.ones(3)))
    with pytest.raises(ValueError, match="trailing shape"):
        DynamicLESInputs(
            jnp.ones(3),
            jnp.ones(3),
            algebraic,
            provenance,
            accepted_update_mask=True,
        )
    with pytest.raises(ValueError, match="equal shape"):
        DynamicLESInputs(
            basis,
            jnp.broadcast_to(basis, (2, 3, 3)),
            algebraic,
            provenance,
            accepted_update_mask=True,
        )
    with pytest.raises(ValueError, match="velocity-gradient"):
        _inputs(
            jnp.broadcast_to(basis, (2, 3, 3)),
            jnp.broadcast_to(basis, (2, 3, 3)),
            gradient=basis,
            provenance=provenance,
        )
    for field in (
        basis.at[0, 0].set(jnp.nan),
        basis.at[1, 1].set(jnp.inf),
    ):
        with pytest.raises(ValueError, match="finite"):
            _inputs(field, basis, provenance=provenance)
        with pytest.raises(ValueError, match="finite"):
            _inputs(basis, field, provenance=provenance)
        with pytest.raises(ValueError, match="finite"):
            _inputs(basis, basis, gradient=field, provenance=provenance)

    field = jnp.broadcast_to(basis, (2, 3, 3))
    with pytest.raises(TypeError, match="boolean"):
        _inputs(field, field, provenance=provenance, accepted=jnp.ones(2))
    with pytest.raises(ValueError, match="scalar or match"):
        _inputs(field, field, provenance=provenance, accepted=jnp.ones(3, dtype=bool))
    widths = jnp.ones((3, 3))
    with pytest.raises(ValueError, match="does not broadcast"):
        _inputs(basis, basis, provenance=provenance, widths=widths)


def test_policy_constructors_refuse_invalid_averaging_and_bounds():
    for axes in ((), ("x", "x"), ("x", "y", "z", "w")):
        with pytest.raises((TypeError, ValueError)):
            HomogeneousPlaneDynamicLESAveraging(axes)
    for kernel in (
        jnp.ones((3, 3)),
        jnp.ones((2, 3, 3)),
        -jnp.ones((3, 3, 3)),
        jnp.zeros((3, 3, 3)),
        jnp.ones((3, 3, 3)).at[0, 0, 0].set(jnp.inf),
    ):
        with pytest.raises(ValueError):
            LocalKernelDynamicLESAveraging(kernel)
    for relaxation in (0.0, -0.1, 1.1, np.inf):
        with pytest.raises(ValueError):
            LagrangianDynamicLESAveraging(relaxation)
    for shift in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            AdditiveDenominatorRegularization(shift)
    for fraction, reference in ((-0.1, 1.0), (1.1, 1.0), (0.2, 0.0), (0.2, np.inf)):
        with pytest.raises(ValueError):
            BoundedFractionBackscatter(fraction, reference)


def test_averaging_and_restart_refuse_incompatible_runtime_use():
    provenance = _provenance()
    basis = _trace_free_basis()
    scalar_inputs = _inputs(basis, basis, provenance=provenance)
    global_prepared = _prepared(provenance)
    with pytest.raises(TypeError, match="Only Lagrangian"):
        global_prepared.initial_state(scalar_inputs)

    lagrangian = _prepared(provenance, averaging=LagrangianDynamicLESAveraging(0.5))
    with pytest.raises(TypeError, match="explicit initialized"):
        lagrangian.evaluate(scalar_inputs)
    state = lagrangian.initial_state(scalar_inputs)
    with pytest.raises(TypeError, match="does not accept"):
        global_prepared.evaluate(scalar_inputs, state)
    wrong_state = LagrangianDynamicLESState(
        state.averaged_numerator,
        state.averaged_denominator,
        state.initialized_mask,
        state.accepted_updates,
        state.rejected_updates,
        "wrong-continuation",
    )
    with pytest.raises(ValueError, match="incompatible"):
        lagrangian.evaluate(scalar_inputs, wrong_state)

    other_provenance = _provenance(ratio=(3.0, 3.0, 3.0))
    with pytest.raises(ValueError, match="provenance differ"):
        global_prepared.evaluate(_inputs(basis, basis, provenance=other_provenance))


def test_axis_and_boundary_dependent_averaging_refuses_unsupported_fields():
    provenance = _provenance()
    basis = _trace_free_basis()
    inputs = _inputs(basis, basis, provenance=provenance)
    with pytest.raises(ValueError, match="three spatial axes"):
        _prepared(
            provenance,
            averaging=HomogeneousPlaneDynamicLESAveraging(("x", "z")),
        ).evaluate(inputs)
    with pytest.raises(ValueError, match="absent"):
        field = jnp.broadcast_to(basis, (1, 1, 1, 3, 3))
        _prepared(
            provenance,
            averaging=HomogeneousPlaneDynamicLESAveraging(("not-an-axis",)),
        ).evaluate(_inputs(field, field, provenance=provenance))

    wall_resolved = _filter("resolved-wall", boundary_class="wall-bounded")
    wall_test = _filter(
        "test-wall",
        family="explicit-filter",
        boundary_class="wall-bounded",
        repeated_filter_semantics="composed",
    )
    wall_provenance = _provenance(resolved_filter=wall_resolved, test_filter=wall_test)
    wall_field = jnp.broadcast_to(basis, (1, 1, 1, 3, 3))
    with pytest.raises(ValueError, match="requires periodic"):
        _prepared(
            wall_provenance,
            averaging=LocalKernelDynamicLESAveraging(jnp.ones((1, 1, 1))),
        ).evaluate(_inputs(wall_field, wall_field, provenance=wall_provenance))
