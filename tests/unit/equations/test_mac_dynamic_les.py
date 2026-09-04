#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.incompressible_flow._production import (
    MACDynamicLESProductionState,
    PreparedMACDynamicExplicitMethod,
    StructuredMACProductionPlan,
)
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
from phydrax.equations._mac_dynamic_les import (
    MACDynamicLESPlan,
    MACExplicitTestFilterPlan,
)


def _grid(*, counts=(5, 5, 5), axis_specs=None, side_kind=None):
    specs = (
        tuple(phx.discretization.UniformCellAxisSpec(n, periodic=True) for n in counts)
        if axis_specs is None
        else axis_specs
    )
    grid = phx.discretization.TensorGridPlan(specs, axis_names=("x", "y", "z")).prepare(
        jnp.asarray([[0.0, 0.0, 0.0], [2.0 * jnp.pi] * 3])
    )
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    if side_kind is None:
        momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    else:
        boundaries = phx.discretization.MACBoundaryPlan(
            operators,
            (
                phx.discretization.MACBoundarySide("z", "lower", side_kind),
                phx.discretization.MACBoundarySide("z", "upper", side_kind),
            ),
        ).prepare()
        momentum = phx.discretization.MACMomentumPlan(
            operators, boundaries=boundaries
        ).prepare()
    return discretization, operators, momentum


def _filters(*, test_repeated="composed"):
    resolved = ResolvedLESFilter(
        "periodic uniform MAC cell volumes",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="volume-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="unmodeled",
    )
    test = ResolvedLESFilter(
        "periodic binomial MAC test filter",
        family="explicit-filter",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="kernel-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics=test_repeated,
    )
    return resolved, test


def _prepared(
    discretization,
    momentum,
    *,
    averaging=None,
    regularization=None,
    backscatter=None,
    ratio=(2.0, 2.0, 2.0),
    discretization_id=None,
    regime="incompressible-unit-density",
):
    resolved_filter, test_filter = _filters()
    parameters = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id if discretization_id is None else discretization_id,
        regime,
        source_kind="user",
        evidence_ids=(),
    )
    dynamic = DynamicSmagorinskyPlan(
        GlobalDynamicLESAveraging() if averaging is None else averaging,
        ExactDenominatorRegularization() if regularization is None else regularization,
        NonnegativeBackscatterClip() if backscatter is None else backscatter,
    ).prepare(DynamicLESProvenance(parameters, test_filter, ratio))
    return MACDynamicLESPlan(dynamic, MACExplicitTestFilterPlan(test_filter)).prepare(
        momentum
    )


def _velocity(discretization):
    keys = jax.random.split(jax.random.PRNGKey(23), 3)
    return tuple(
        jax.random.normal(key, layout.shape)
        for key, layout in zip(keys, discretization.face_layouts, strict=True)
    )


def test_mac_test_filter_is_fixed_normalized_distinct_and_reports_support():
    discretization, _, momentum = _grid(counts=(6, 6, 6))
    prepared = _prepared(discretization, momentum)
    test_filter = prepared.test_filter
    constant = jnp.ones(discretization.cell_shape)
    alternating = (-1.0) ** jnp.indices(discretization.cell_shape)[0]
    varying = jnp.arange(np.prod(discretization.cell_shape), dtype=float).reshape(
        discretization.cell_shape
    )

    np.testing.assert_allclose(test_filter.apply(constant), constant, atol=0.0)
    assert jnp.linalg.norm(test_filter.apply(alternating) - alternating) > 1.0
    np.testing.assert_allclose(
        jnp.sum(test_filter.apply(varying)), jnp.sum(varying), atol=2e-11
    )
    np.testing.assert_allclose(sum(test_filter.kernel_weights), 1.0, atol=0.0)
    assert test_filter.kernel_weights == (0.25, 0.5, 0.25)
    assert test_filter.test_filter_ratio == (2.0, 2.0, 2.0)
    assert test_filter.commutation_status == "commuting"
    assert test_filter.boundary_support == "periodic-wrap-only"


def test_mac_filter_commutes_with_periodic_uniform_cell_difference():
    discretization, _, momentum = _grid(counts=(6, 6, 6))
    prepared = _prepared(discretization, momentum)
    values = jnp.sin(discretization.cell_centers[..., 0])
    width = momentum.operators.discretization.grid.structured_axes[0].interval_widths[0]

    derivative_then_filter = prepared.test_filter.apply(
        (jnp.roll(values, -1, axis=0) - jnp.roll(values, 1, axis=0)) / (2.0 * width)
    )
    filter_then_derivative = (
        jnp.roll(prepared.test_filter.apply(values), -1, axis=0)
        - jnp.roll(prepared.test_filter.apply(values), 1, axis=0)
    ) / (2.0 * width)

    np.testing.assert_allclose(
        derivative_then_filter, filter_then_derivative, rtol=2e-12, atol=2e-12
    )


def test_mac_adapter_recovers_synthetic_coefficient_at_variational_cells():
    discretization, _, momentum = _grid()
    prepared = _prepared(discretization, momentum, backscatter=AllowSignedBackscatter())
    velocity = _velocity(discretization)
    boundary_stage = momentum.boundaries.homogeneous_stage()
    inputs, _, _, _, _ = prepared._germano_inputs(
        velocity, boundary_stage, accepted_update_mask=True
    )
    coefficient = 0.27
    synthetic = DynamicLESInputs(
        coefficient * inputs.modeled_tensor + 0.8 * jnp.eye(3),
        inputs.modeled_tensor,
        inputs.algebraic_inputs,
        inputs.provenance,
        accepted_update_mask=True,
    )

    result = prepared.dynamic_model.evaluate(synthetic)

    np.testing.assert_allclose(result.coefficient, coefficient, rtol=2e-6, atol=2e-6)
    assert inputs.algebraic_inputs.velocity_gradient.shape == (
        *discretization.cell_shape,
        3,
        3,
    )


def test_mac_dynamic_stage_realizes_stress_rate_transfer_and_energy_evidence():
    discretization, operators, momentum = _grid()
    prepared = _prepared(
        discretization,
        momentum,
        regularization=AdditiveDenominatorRegularization(1e-12),
    )
    velocity = _velocity(discretization)

    stage = prepared.evaluate(velocity, momentum.boundaries.homogeneous_stage())

    assert stage.dynamic_result.evidence.finite
    assert stage.mac_stage.successful
    assert stage.model_result.specific_deviatoric_stress.shape == (
        *discretization.cell_shape,
        3,
        3,
    )
    assert stage.model_result.energy_transfer.shape == discretization.cell_shape
    assert tuple(value.shape for value in stage.physical_rate) == tuple(
        layout.shape for layout in discretization.face_layouts
    )
    variational_work = jnp.real(
        operators.velocity_space.inner(velocity, stage.physical_rate)
    )
    np.testing.assert_allclose(
        stage.integrated_work, variational_work, rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(
        stage.integrated_work,
        -stage.mac_stage.viscosity_result.integrated_dissipation,
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.all(stage.model_result.kinematic_viscosity >= 0.0)
    assert jnp.all(stage.model_result.energy_transfer >= -1e-12)
    assert stage.commutation_status == "commuting"
    assert stage.boundary_support == "periodic-wrap-only"


@pytest.mark.parametrize(
    ("averaging", "expected_shape"),
    (
        (GlobalDynamicLESAveraging(), ()),
        (HomogeneousPlaneDynamicLESAveraging(("x", "y")), (1, 1, 5)),
        (
            LocalKernelDynamicLESAveraging(jnp.ones((3, 3, 3), dtype=jnp.float32)),
            (5, 5, 5),
        ),
    ),
)
def test_mac_global_plane_and_local_routes(averaging, expected_shape):
    discretization, _, momentum = _grid()
    prepared = _prepared(discretization, momentum, averaging=averaging)

    stage = prepared.evaluate(
        _velocity(discretization), momentum.boundaries.homogeneous_stage()
    )

    assert stage.dynamic_result.coefficient.shape == expected_shape
    assert stage.continuation_state is None
    assert stage.accepted_update_mask.shape == ()


def test_mac_history_mask_restart_and_no_hidden_commit():
    discretization, _, momentum = _grid()
    prepared = _prepared(
        discretization,
        momentum,
        averaging=LagrangianDynamicLESAveraging(0.4),
    )
    velocity = _velocity(discretization)
    boundary_stage = momentum.boundaries.homogeneous_stage()
    accepted = jnp.indices(discretization.cell_shape).sum(axis=0) % 2 == 0
    initial = prepared.initial_state(
        velocity, boundary_stage, accepted_update_mask=accepted
    )

    first = prepared.evaluate(
        velocity,
        boundary_stage,
        initial,
        accepted_update_mask=accepted,
    )
    replay = prepared.evaluate(
        velocity,
        boundary_stage,
        initial,
        accepted_update_mask=accepted,
    )
    restarted = prepared.evaluate(
        velocity,
        boundary_stage,
        first.continuation_state,
        accepted_update_mask=False,
    )

    np.testing.assert_allclose(
        first.continuation_state.averaged_denominator,
        replay.continuation_state.averaged_denominator,
    )
    np.testing.assert_allclose(
        restarted.continuation_state.averaged_denominator,
        first.continuation_state.averaged_denominator,
    )
    np.testing.assert_array_equal(first.accepted_update_mask, accepted)
    assert initial.accepted_updates == 0
    assert first.continuation_state.accepted_updates == jnp.sum(accepted)
    assert restarted.dynamic_result.evidence.rejected_update_count == accepted.size


def test_mac_compiler_consumes_dynamic_stage_and_rate():
    discretization, operators, momentum = _grid(counts=(4, 4, 4))
    adapter = _prepared(
        discretization,
        momentum,
        averaging=GlobalDynamicLESAveraging(),
        regularization=AdditiveDenominatorRegularization(1.0e-8),
    )
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01),
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            solve_method="transform",
            tolerance=1.0e-9,
        ),
        dynamic_les=adapter.plan,
    )
    state = dynamics.project_state(_velocity(discretization))
    components = dynamics.rate_components(0.0, state)
    restriction = dynamics.step_restriction(
        0.0,
        state,
        dynamic_les_stage=components.dynamic_les_stage,
    )
    diagnostics = dynamics.diagnostics(0.0, state)

    assert components.les_stage is None
    assert components.dynamic_les_stage is not None
    assert components.dynamic_les_stage.continuation_state is None
    np.testing.assert_allclose(
        components.sgs[0],
        components.dynamic_les_stage.physical_rate[0],
        atol=0.0,
    )
    assert dynamics.dynamic_les.prepared_id == components.dynamic_les_stage.prepared_id
    assert restriction.sgs_supported
    assert restriction.combined > 0.0
    assert bool(diagnostics.dynamic_les_available)
    assert diagnostics.dynamic_les_id == dynamics.dynamic_les.prepared_id
    assert bool(diagnostics.dynamic_evidence_finite)


def test_mac_dynamic_explicit_method_commits_only_successful_outer_step():
    discretization, operators, momentum = _grid(counts=(4, 4, 4))
    adapter = _prepared(
        discretization,
        momentum,
        averaging=LagrangianDynamicLESAveraging(0.4),
        regularization=AdditiveDenominatorRegularization(1.0e-8),
    )
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01),
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            solve_method="transform",
            tolerance=1.0e-9,
        ),
        dynamic_les=adapter.plan,
    )
    velocity = dynamics.project_state(_velocity(discretization))
    faces = dynamics.physical_state(0.0, velocity)
    boundary = dynamics.boundary_stage(0.0)
    continuation = dynamics.dynamic_les.initial_state(faces, boundary)
    state = MACDynamicLESProductionState(velocity, continuation)
    method = PreparedMACDynamicExplicitMethod(dynamics)

    accepted = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-6),
        None,
    )
    initial_components = dynamics.rate_components(
        0.0,
        velocity,
        continuation_state=continuation,
    )
    rejected_step = (
        2.0
        * dynamics.step_restriction(
            0.0,
            velocity,
            dynamic_les_stage=initial_components.dynamic_les_stage,
        ).combined
    )
    rejected = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        rejected_step,
        None,
    )

    assert bool(accepted.successful)
    assert isinstance(accepted.accepted_state, MACDynamicLESProductionState)
    assert int(accepted.accepted_state.continuation_state.accepted_updates) > int(
        state.continuation_state.accepted_updates
    )
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.accepted_state.velocity, state.velocity)
    np.testing.assert_array_equal(
        rejected.accepted_state.continuation_state.averaged_numerator,
        state.continuation_state.averaged_numerator,
    )
    with pytest.raises(ValueError, match="scientifically incompatible"):
        StructuredMACProductionPlan(
            method,
            dynamics,
            None,
            start_time=0.0,
            end_time=1.0e-6,
            step_size=1.0e-6,
            checkpoint_interval=1,
        )


def test_mac_compiler_refuses_signed_viscosity_dynamic_policy():
    discretization, operators, momentum = _grid(counts=(4, 4, 4))
    signed = _prepared(
        discretization,
        momentum,
        backscatter=AllowSignedBackscatter(),
    )
    with pytest.raises(ValueError, match="nonnegative backscatter clipping"):
        phx.equations.compile_mac_incompressible_flow(
            phx.equations.IncompressibleFlowProblem(3, 0.01),
            momentum,
            phx.solver.MACPressureProjectionPlan(
                operators,
                solve_method="transform",
            ),
            dynamic_les=signed.plan,
        )


def test_mac_signed_and_clipped_backscatter_remain_explicit_policies():
    discretization, _, momentum = _grid()
    signed = _prepared(discretization, momentum, backscatter=AllowSignedBackscatter())
    clipped = _prepared(
        discretization, momentum, backscatter=NonnegativeBackscatterClip()
    )
    velocity = _velocity(discretization)
    boundary_stage = momentum.boundaries.homogeneous_stage()
    inputs, _, _, _, _ = signed._germano_inputs(
        velocity, boundary_stage, accepted_update_mask=True
    )
    signed_inputs = DynamicLESInputs(
        -0.15 * inputs.modeled_tensor,
        inputs.modeled_tensor,
        inputs.algebraic_inputs,
        inputs.provenance,
        accepted_update_mask=True,
    )
    clipped_inputs = DynamicLESInputs(
        signed_inputs.leonard_tensor,
        signed_inputs.modeled_tensor,
        signed_inputs.algebraic_inputs,
        clipped.dynamic_model.provenance,
        accepted_update_mask=True,
    )

    signed_result = signed.dynamic_model.evaluate(signed_inputs)
    clipped_result = clipped.dynamic_model.evaluate(clipped_inputs)

    np.testing.assert_allclose(signed_result.coefficient, -0.15, atol=2e-7)
    np.testing.assert_allclose(clipped_result.coefficient, 0.0, atol=0.0)
    assert signed_result.prepared_algebraic_stress.energy_transfer.min() < 0.0
    assert clipped_result.prepared_algebraic_stress.energy_transfer.min() == 0.0
    assert clipped_result.evidence.backscatter_activity_count == 1


def test_mac_dynamic_adapter_is_jittable_and_has_finite_jvp():
    discretization, _, momentum = _grid()
    prepared = _prepared(
        discretization,
        momentum,
        regularization=AdditiveDenominatorRegularization(1e-8),
    )
    velocity = _velocity(discretization)
    boundary_stage = momentum.boundaries.homogeneous_stage()

    rate = jax.jit(lambda value: prepared.evaluate(value, boundary_stage).physical_rate)(
        velocity
    )
    _, tangent = jax.jvp(
        lambda value: jnp.sum(
            prepared.evaluate(value, boundary_stage).model_result.energy_transfer
        ),
        (velocity,),
        (tuple(0.01 * value for value in velocity),),
    )

    assert all(jnp.all(jnp.isfinite(value)) for value in rate)
    assert jnp.isfinite(tangent)


def test_mac_dynamic_prepare_refuses_nonuniform_wall_open_and_mismatched_routes():
    edges = jnp.asarray((0.0, 0.1, 0.35, 0.7, 1.0))
    nonuniform_specs = tuple(
        phx.discretization.NonuniformCellAxisSpec(edges, periodic=True) for _ in range(3)
    )
    nonuniform, _, nonuniform_momentum = _grid(axis_specs=nonuniform_specs)
    with pytest.raises(ValueError, match="periodic uniform"):
        _prepared(nonuniform, nonuniform_momentum)

    wall_specs = (
        phx.discretization.UniformCellAxisSpec(5, periodic=True),
        phx.discretization.UniformCellAxisSpec(5, periodic=True),
        phx.discretization.UniformCellAxisSpec(5),
    )
    for side_kind in ("free-slip", "pressure-outlet"):
        wall, _, wall_momentum = _grid(axis_specs=wall_specs, side_kind=side_kind)
        with pytest.raises(ValueError, match="periodic uniform|periodic-wrap"):
            _prepared(wall, wall_momentum)

    discretization, _, momentum = _grid()
    with pytest.raises(ValueError, match="directional width ratio 2"):
        _prepared(discretization, momentum, ratio=(2.1, 2.0, 2.0))
    with pytest.raises(ValueError, match="discretization"):
        _prepared(
            discretization,
            momentum,
            discretization_id="different-discretization",
        )
    with pytest.raises(ValueError, match="incompressible-unit-density"):
        _prepared(discretization, momentum, regime="variable-density")

    _, invalid_test = _filters(test_repeated="unmodeled")
    with pytest.raises(ValueError, match="composed semantics"):
        MACExplicitTestFilterPlan(invalid_test)
