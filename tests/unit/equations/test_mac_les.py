#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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
from phydrax.equations._mac_incompressible import (
    compile_mac_incompressible_flow,
    MACIncompressibleRateComponents,
    MACLESStepRestriction,
)
from phydrax.equations._mac_les import (
    MACAlgebraicLESPlan,
    MACLESStageResult,
    PreparedMACAlgebraicLES,
)


def _grid(*, counts=(4, 4, 4), axis_specs=None, side_kind=None):
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


def _les_plan(
    discretization,
    *,
    coefficient=0.17,
    boundary_class="periodic",
    discretization_id=None,
    regime="incompressible-unit-density",
    family="implicit-grid-volume",
):
    resolved_filter = ResolvedLESFilter(
        "mac-cell-volume",
        family=family,
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class=boundary_class,
        scale_rule=(
            "volume-equivalent"
            if family == "implicit-grid-volume"
            else "kernel-equivalent"
        ),
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id if discretization_id is None else discretization_id,
        regime,
        source_kind="user",
        evidence_ids=(),
    )
    return MACAlgebraicLESPlan(SmagorinskyLESPlan(coefficient).prepare(provenance))


def _compiled(*, coefficient=0.17, viscosity=0.01, count=4):
    discretization, operators, momentum = _grid(counts=(count,) * 3)
    projection = phx.solver.MACPressureProjectionPlan(operators, solve_method="transform")
    dynamics = compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        momentum,
        projection,
        algebraic_les=_les_plan(discretization, coefficient=coefficient),
    )
    return discretization, operators, dynamics


def _taylor_green(discretization):
    x_faces, y_faces, z_faces = discretization.face_centers
    return (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]) * jnp.cos(x_faces[..., 2]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]) * jnp.cos(y_faces[..., 2]),
        jnp.zeros(z_faces.shape[:-1], dtype=z_faces.dtype),
    )


def test_mac_les_preparation_retains_only_factored_filter_width_metadata():
    edges = jnp.asarray((0.0, 0.1, 0.35, 0.7, 1.0))
    specs = tuple(
        phx.discretization.NonuniformCellAxisSpec(edges, periodic=True) for _ in range(3)
    )
    discretization, _, momentum = _grid(axis_specs=specs)

    prepared = _les_plan(discretization).prepare(momentum)
    scale = prepared.filter_scale()

    assert isinstance(prepared, PreparedMACAlgebraicLES)
    assert tuple(width.shape for width in prepared.filter_axis_widths) == (
        (4,),
        (4,),
        (4,),
    )
    assert all(width.ndim == 1 for width in prepared.filter_axis_widths)
    assert scale.directional_widths.shape == discretization.cell_shape + (3,)
    for axis, width in enumerate(prepared.filter_axis_widths):
        expected = jnp.broadcast_to(
            width.reshape((1,) * axis + (4,) + (1,) * (2 - axis)),
            discretization.cell_shape,
        )
        np.testing.assert_allclose(scale.directional_widths[..., axis], expected)


@pytest.mark.parametrize(
    ("plan_change", "message"),
    (
        ({"discretization_id": "wrong-grid"}, "discretization"),
        ({"regime": "variable-density"}, "incompressible-unit-density"),
        ({"family": "explicit-filter"}, "filter semantics"),
    ),
)
def test_mac_les_refuses_mismatched_prepared_provenance(plan_change, message):
    discretization, _, momentum = _grid()
    with pytest.raises(ValueError, match=message):
        _les_plan(discretization, **plan_change).prepare(momentum)


def test_mac_les_refuses_non_3d_and_unsupported_physical_boundaries():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    discretization_2d = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators_2d = phx.discretization.MACOperatorPlan(discretization_2d).prepare()
    momentum_2d = phx.discretization.MACMomentumPlan(operators_2d).prepare()
    three_dimensional_plan = _les_plan(
        discretization_2d, discretization_id=discretization_2d.prepared_id
    )
    with pytest.raises(ValueError, match="three-dimensional"):
        three_dimensional_plan.prepare(momentum_2d)

    specs = (
        phx.discretization.UniformCellAxisSpec(4, periodic=True),
        phx.discretization.UniformCellAxisSpec(4, periodic=True),
        phx.discretization.UniformCellAxisSpec(4),
    )
    for unsupported_kind in ("no-slip", "velocity-inflow", "pressure-outlet"):
        unsupported_discretization, _, unsupported_momentum = _grid(
            axis_specs=specs, side_kind=unsupported_kind
        )
        with pytest.raises(ValueError, match="no-slip, open, inflow"):
            _les_plan(unsupported_discretization, boundary_class="wall-bounded").prepare(
                unsupported_momentum
            )

    for supported_kind in ("free-slip", "symmetry"):
        wall_discretization, _, wall_momentum = _grid(
            axis_specs=specs, side_kind=supported_kind
        )
        prepared = _les_plan(wall_discretization, boundary_class="wall-bounded").prepare(
            wall_momentum
        )
        zero = tuple(
            jnp.zeros(layout.shape) for layout in wall_discretization.face_layouts
        )
        result = prepared.evaluate(zero, wall_momentum.boundaries.homogeneous_stage())
        assert isinstance(prepared, PreparedMACAlgebraicLES)
        assert result.successful


@pytest.mark.parametrize("coefficient", (0.0, 0.17))
def test_mac_les_typed_preprojection_rate_and_work_identity(coefficient):
    discretization, operators, dynamics = _compiled(
        coefficient=coefficient, count=4 if coefficient == 0.0 else 5
    )
    velocity = _taylor_green(discretization)
    state = dynamics.pack_velocity(velocity)

    components = dynamics.rate_components(0.0, state)
    projected = dynamics.rate_projection(0.0, state)

    assert isinstance(components, MACIncompressibleRateComponents)
    assert isinstance(components.les_stage, MACLESStageResult)
    assert components.les_stage.successful
    assert jnp.max(jnp.abs(projected.divergence_after)) < 2.0e-10
    expected = tuple(
        -advective + molecular + sgs + forcing
        for advective, molecular, sgs, forcing in zip(
            components.convection,
            components.molecular,
            components.sgs,
            components.forcing,
            strict=True,
        )
    )
    for actual, value in zip(components.unconstrained, expected, strict=True):
        np.testing.assert_allclose(actual, value, rtol=2.0e-12, atol=2.0e-12)
    work = jnp.real(operators.velocity_space.inner(velocity, components.sgs))
    np.testing.assert_allclose(
        work, components.les_stage.integrated_work, rtol=2.0e-11, atol=2.0e-11
    )
    np.testing.assert_allclose(
        components.les_stage.integrated_work,
        -components.les_stage.viscosity_result.integrated_dissipation,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    maximum_sgs = max(float(jnp.max(jnp.abs(value))) for value in components.sgs)
    if coefficient == 0.0:
        assert maximum_sgs == 0.0
        assert jnp.all(components.les_stage.model_result.kinematic_viscosity == 0.0)
    else:
        assert maximum_sgs > 0.0
        assert jnp.std(components.les_stage.model_result.kinematic_viscosity) > 0.0


def test_mac_les_restriction_and_diagnostics_are_current_state_dependent():
    discretization, _, dynamics = _compiled(coefficient=0.17, count=5)
    state = dynamics.pack_velocity(_taylor_green(discretization))
    zero = jnp.zeros_like(state)

    active = dynamics.step_restriction(0.0, state)
    quiescent = dynamics.step_restriction(0.0, zero)
    diagnostics = dynamics.diagnostics(0.0, state)

    assert isinstance(active, MACLESStepRestriction)
    assert active.sgs_supported
    assert jnp.isfinite(active.sgs)
    assert jnp.isinf(quiescent.sgs)
    np.testing.assert_allclose(
        active.combined,
        jnp.minimum(jnp.minimum(active.advective, active.molecular), active.sgs),
    )
    assert diagnostics.sgs_dissipation > 0.0
    np.testing.assert_allclose(
        diagnostics.sgs_energy_rate,
        -diagnostics.sgs_dissipation + diagnostics.sgs_boundary_power,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert diagnostics.successful


def test_mac_no_les_rate_preserves_the_original_momentum_formula_exactly():
    discretization, operators, momentum = _grid()
    projection = phx.solver.MACPressureProjectionPlan(operators, solve_method="transform")
    viscosity = jnp.asarray(0.03)
    dynamics = compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, viscosity),
        momentum,
        projection,
    )
    velocity = _taylor_green(discretization)
    state = dynamics.pack_velocity(velocity)
    stage = dynamics.boundary_stage(0.0)
    physical = momentum.boundaries.enforce(velocity, stage)
    convection = momentum.convection(physical, stage=stage)
    diffusion = momentum.laplacian(physical, stage=stage)
    original = momentum.boundaries.enforce_rate(
        tuple(
            -advective + viscosity * viscous + jnp.zeros_like(advective)
            for advective, viscous in zip(convection, diffusion, strict=True)
        ),
        stage,
    )

    components = dynamics.rate_components(0.0, state)
    restriction = dynamics.step_restriction(0.0, state)

    assert components.les_stage is None
    for actual, expected in zip(components.unconstrained, original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert all(jnp.all(value == 0.0) for value in components.sgs)
    assert jnp.isinf(restriction.sgs)
    assert restriction.sgs_supported


def test_mac_les_rate_is_jittable_and_has_state_jvp():
    discretization, operators, dynamics = _compiled(coefficient=0.12, count=3)
    state = dynamics.pack_velocity(_taylor_green(discretization))

    def sgs_rate(coordinates):
        components = dynamics.rate_components(0.0, coordinates)
        return operators.velocity_space.flatten(components.sgs)

    eager = sgs_rate(state)
    compiled = jax.jit(sgs_rate)(state)
    _, tangent = jax.jvp(sgs_rate, (state,), (0.1 * state,))

    np.testing.assert_allclose(compiled, eager, rtol=2.0e-11, atol=2.0e-11)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.linalg.norm(tangent) > 0.0


def test_active_mac_les_selects_the_frozen_implicit_temporal_profile():
    _, _, dynamics = _compiled(coefficient=0.12, count=3)
    method = phx.solver.MACIMEXEulerMethod(dynamics, solve_method="iterative")

    assert method.implicit_les
    assert method.temporal_profile == "mac-frozen-algebraic-les-imex-euler"
    assert method.prepared_les_id == dynamics.algebraic_les.prepared_id
    assert method.capabilities.method_id == method.method_id
    assert method.capabilities.order == 1
