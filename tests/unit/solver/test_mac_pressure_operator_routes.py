#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.solver._mac_immersed_preconditioner import (
    MACImmersedPressureBlockPreconditionerPlan,
)
from phydrax.solver._mac_pressure_operator import (
    MACPressureOperatorSpec,
    MACPressureRobinSide,
)


def _operators(*, periodic=True, dimension=2):
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=periodic)
        for _ in range(dimension)
    )
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(axes, axis_names=names).prepare(
        jnp.asarray((tuple(0.0 for _ in axes), tuple(1.0 for _ in axes)))
    )
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    return finite_volume, phx.discretization.MACOperatorPlan(finite_volume).prepare()


def test_general_beta_pcg_evidence_and_frozen_coefficient_mismatch():
    _, operators = _operators()
    shape = operators.discretization.cell_shape
    beta = 1.0 + 0.25 * jnp.sin(jnp.arange(operators.pressure_space.size)).reshape(shape)
    prepared = MACPressureOperatorSpec(
        operators,
        beta,
        solve_method="auto",
        tolerance=1.0e-6,
        maximum_resource_bytes=2_000_000,
    ).prepare()
    rhs = operators.gauge_project(
        jnp.cos(0.31 * jnp.arange(operators.pressure_space.size)).reshape(shape)
    )
    result = prepared.solve(rhs)

    assert prepared.spec.route == "pcg"
    assert prepared.preparation.coefficient.contrast > 1.0
    assert prepared.preparation.action_defect < 1.0e-8
    assert prepared.preparation.symmetry_defect < 1.0e-8
    assert prepared.preparation.jvp_defect < 1.0e-8
    assert prepared.preparation.vjp_defect < 1.0e-8
    assert prepared.preparation.resource_bytes <= 2_000_000
    assert prepared.preparation.frozen_preparation
    assert result.successful
    assert result.evidence.residual_norm < 1.0e-5
    with pytest.raises(ValueError, match="frozen prepared coefficient"):
        prepared.validate_frozen(2.0 * beta)


def test_robin_eligibility_removes_gauge_and_nonsymmetric_traction_uses_fgmres():
    _, operators = _operators(periodic=False, dimension=1)
    robin = MACPressureRobinSide(0, "lower", 1.0, 0.5, 2.0)
    prepared = MACPressureOperatorSpec(
        operators,
        1.0,
        robin_sides=(robin,),
        solve_method="iterative",
        tolerance=1.0e-6,
    ).prepare()
    flexible = MACPressureOperatorSpec(
        operators,
        1.0,
        nonsymmetric_traction=True,
        solve_method="auto",
        tolerance=1.0e-6,
    ).prepare()

    assert prepared.preparation.robin_eligible
    assert prepared.preparation.gauge_removed
    assert prepared.spec.route == "pcg"
    assert flexible.spec.route == "fgmres"
    assert "nonsymmetric" in flexible.spec.route_reason


def test_explicit_direct_request_never_falls_back_for_general_beta():
    _, operators = _operators()
    beta = 1.0 + 0.1 * jnp.arange(operators.pressure_space.size).reshape(
        operators.discretization.cell_shape
    )
    with pytest.raises(ValueError, match="no certified exact representation"):
        MACPressureOperatorSpec(operators, beta, solve_method="direct")


def test_mixed_dirichlet_neumann_transform_lift_has_no_pressure_gauge():
    finite_volume, operators = _operators(periodic=False)
    side = phx.discretization.MACBoundarySide
    provider = phx.discretization.MACBoundaryProvider
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            side("x", "lower", "pressure-outlet", provider=provider(1.0)),
            side("x", "upper", "no-slip"),
            side("y", "lower", "no-slip"),
            side("y", "upper", "no-slip"),
        ),
    ).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        boundaries=boundaries,
        solve_method="transform",
        tolerance=1.0e-7,
    )
    zero = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    result = projection.project(zero, 1.0)

    assert result.solve_method == "transform"
    assert result.closure.kind == "dirichlet"
    assert result.closure.gauge == "none"
    assert result.gauge_defect == 0.0
    assert result.converged


def test_immersed_pressure_block_composes_without_owning_kkt():
    composition = MACImmersedPressureBlockPreconditionerPlan()
    policy = composition.policy()

    assert not composition.evidence.owns_kkt_operator
    assert composition.evidence.pressure_action_reused
    assert policy.side == "right"
    assert policy.refresh_policy == "numeric"


def test_identity_ale_manufactured_projection_records_epoch_and_metric_evidence():
    finite_volume, _ = _operators(periodic=False)
    plan = phx.solver.MACALEGeometryPlan(
        finite_volume,
        lambda time, point, args: point,
        lambda time, point, args: jnp.zeros_like(point),
        mapping_id="pressure-route-identity",
        tolerance=1.0e-7,
        geometry_epoch=4,
    )
    geometry = plan.evaluate(0.0)
    pressure = geometry.gauge_project(
        jnp.sin(
            0.21
            * jnp.arange(geometry.cell_volumes.size).reshape(geometry.cell_volumes.shape)
        )
    )
    result = plan.project(geometry, geometry.gradient(pressure), 1.0)

    assert result.success
    assert result.pressure_route == "pcg"
    assert result.geometry_epoch == 4
    assert result.preconditioner_refreshed
    assert result.gcl_identity_residual < 1.0e-7
    assert result.mapped_adjoint_residual < 1.0e-7
    assert result.gauge_defect < 1.0e-7
