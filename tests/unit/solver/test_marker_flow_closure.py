#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import prod

import h5py
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _periodic_mac(count=6):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    return finite_volume, operators, boundaries


def _markers(position):
    return phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(position.shape[0]),
        position,
        jnp.full((position.shape[0],), 1.0 / position.shape[0]),
    ).prepare()


@pytest.mark.parametrize(
    "kernel_name",
    ("cubic-bspline", "peskin-four-point", "roma-three-point"),
)
def test_kernel_families_are_adjoint_deterministic_and_fixed_route_differentiable(
    kernel_name,
):
    finite_volume, operators, _ = _periodic_mac()
    position = jnp.asarray([[0.31, 0.37], [0.97, 0.58]])
    markers = _markers(position)
    transfer = phx.discretization.MACMarkerTransferPlan(
        operators,
        markers,
        kernel=phx.discretization.MACMarkerKernelPlan(kernel_name),
        accumulation="compensated",
    ).prepare()
    relation = transfer.relation(position)
    routes = transfer.route_state(relation)
    velocity = tuple(
        jnp.sin(jnp.arange(prod(layout.shape)).reshape(layout.shape))
        for layout in finite_volume.face_layouts
    )
    force = jnp.asarray([[0.2, -0.3], [-0.1, 0.4]])
    first = transfer.spread(relation, force)
    second = transfer.spread(relation, force)
    diagnostics = transfer.diagnostics(relation, velocity, force)

    def observable(value):
        fixed = transfer.relation_on_routes(value, routes)
        return jnp.sum(transfer.gather(fixed, velocity) ** 2)

    _, tangent = jax.jvp(
        observable,
        (position,),
        (jnp.full_like(position, 1.0e-3),),
    )

    assert relation.successful
    assert diagnostics.successful
    assert jnp.any(relation.periodic_image_used)
    assert all(
        jnp.array_equal(left, right) for left, right in zip(first, second, strict=True)
    )
    assert jnp.isfinite(tangent)


def test_nonuniform_cartesian_transfer_reproduces_affine_velocity():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.NonuniformCellAxisSpec(
                jnp.asarray([0.0, 0.12, 0.31, 0.55, 0.78, 1.0])
            ),
            phx.discretization.NonuniformCellAxisSpec(
                jnp.asarray([0.0, 0.18, 0.39, 0.61, 0.84, 1.0])
            ),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    position = jnp.asarray([[0.43, 0.47], [0.66, 0.72]])
    markers = _markers(position)
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    relation = transfer.relation(position)
    velocity = tuple(
        centers[..., 0] + 2.0 * centers[..., 1] + axis
        for axis, centers in enumerate(finite_volume.face_centers)
    )
    gathered = transfer.gather(relation, velocity)
    expected = jnp.stack(
        (
            position[:, 0] + 2.0 * position[:, 1],
            position[:, 0] + 2.0 * position[:, 1] + 1.0,
        ),
        axis=-1,
    )

    assert relation.successful
    assert jnp.allclose(gathered, expected, atol=1.0e-10)


def test_bounded_truncated_support_fails_closed():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(6) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    position = jnp.asarray([[0.01, 0.5]])
    markers = _markers(position)
    relation = (
        phx.discretization.MACMarkerTransferPlan(operators, markers)
        .prepare()
        .relation(position)
    )

    assert not relation.successful
    assert jnp.any(relation.support_truncated)


def test_exact_coupling_honors_inflow_outflow_boundary_descriptor():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(5) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    zero_provider = phx.discretization.MACBoundaryProvider(jnp.zeros((2,)))
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide(
                "x",
                "lower",
                "normal-flux-inflow",
                provider=phx.discretization.MACBoundaryProvider(0.0),
            ),
            phx.discretization.MACBoundarySide(
                "x",
                "upper",
                "pressure-outlet",
                provider=phx.discretization.MACBoundaryProvider(0.0),
            ),
            phx.discretization.MACBoundarySide(
                "y", "lower", "no-slip", provider=zero_provider
            ),
            phx.discretization.MACBoundarySide(
                "y", "upper", "no-slip", provider=zero_provider
            ),
        ),
    ).prepare()
    position = jnp.asarray([[0.5, 0.5]])
    markers = _markers(position)
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, transfer, boundaries=boundaries, tolerance=1.0e-8
    )
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    result = projection.project(
        zero_velocity,
        1.0,
        markers.kinematics(position, jnp.zeros_like(position)),
        boundary_stage=boundaries.evaluate(0.0),
    )
    pressure_projection = phx.solver.MACPressureProjectionPlan(
        operators,
        boundaries=boundaries,
        solve_method="iterative",
        tolerance=1.0e-8,
    )
    divergence_free = phx.solver.MACDivergenceFreeMarkerTransfer(
        transfer,
        pressure_projection,
        require_periodic=False,
    )
    dfib_diagnostics = divergence_free.diagnostics(
        result.relation,
        zero_velocity,
        jnp.zeros((1, 2)),
        boundary_stage=boundaries.evaluate(0.0),
    )

    assert result.converged
    assert result.closure.kind == "dirichlet"
    assert jnp.linalg.norm(result.divergence_after) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8
    assert dfib_diagnostics.successful


@pytest.mark.parametrize("dimension", (2, 3))
def test_nonzero_exact_core_qualifies_in_two_and_three_dimensions(dimension):
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(5, periodic=True)
            for _ in range(dimension)
        ),
        axis_names=names,
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    position = jnp.asarray([[0.31] * dimension, [0.67] * dimension])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(2), position, jnp.full((2,), 0.5)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        transfer,
        boundaries=boundaries,
        tolerance=1.0e-8,
    )
    target = jnp.arange(1, dimension + 1, dtype=position.dtype) * 0.03
    velocity = tuple(
        jnp.full(layout.shape, target[axis])
        for axis, layout in enumerate(finite_volume.face_layouts)
    )
    result = projection.project(
        velocity,
        1.0,
        markers.kinematics(position, jnp.broadcast_to(target, position.shape)),
    )
    assert result.converged
    assert result.marker_rank_certified
    assert jnp.linalg.norm(result.divergence_after) < 1.0e-8
    assert jnp.linalg.norm(result.marker_slip) < 1.0e-8


def test_composite_projection_enforces_compatible_constraint():
    space = phx.linalg.ArraySpace((2,))
    identity = phx.linalg.FunctionLinearOperator(
        lambda value: value,
        source=space,
        target=space,
        transpose_action=lambda value: value,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="composite-identity",
    )
    negative = phx.linalg.FunctionLinearOperator(
        lambda value: -value,
        source=space,
        target=space,
        transpose_action=lambda value: -value,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id="composite-negative-gradient",
    )
    projection = phx.solver.CompositeMACProjectionPlan(
        identity,
        negative,
        identity,
        lambda value: value,
    )
    result = projection.project(jnp.asarray([1.0, -2.0]))

    assert result.accepted
    assert jnp.allclose(result.velocity, 0.0, atol=1.0e-9)
    assert result.divergence_norm < 1.0e-9


def test_mapped_composite_and_distributed_transfers_preserve_virtual_work():
    finite_volume, operators, _ = _periodic_mac()
    position = jnp.asarray([[0.31, 0.37], [0.63, 0.58]])
    markers = _markers(position)
    mapped = phx.discretization.MappedMACGeometryPlan(
        finite_volume,
        lambda point: point,
        mapping_id="marker-transfer-identity",
    ).prepare()
    mapped_transfer = phx.discretization.MappedMACMarkerTransferPlan(
        mapped, markers
    ).prepare()
    mapped_relation = mapped_transfer.relation(position)
    mapped_velocity = tuple(
        jnp.full(layout.shape, float(axis + 1))
        for axis, layout in enumerate(finite_volume.face_layouts)
    )
    force = jnp.asarray([[0.2, -0.3], [-0.1, 0.4]])
    mapped_diagnostics = mapped_transfer.diagnostics(
        mapped_relation, mapped_velocity, force
    )

    measures = ((jnp.ones((4,)), jnp.ones((4,))),)
    composite = phx.discretization.CompositeMACMarkerTransferPlan(markers, measures)
    offsets = jnp.asarray([[-0.08, -0.08], [0.08, -0.08], [0.08, 0.08], [-0.08, 0.08]])
    centers = position[:, None, :] + offsets[None, :, :]
    levels = tuple(jnp.zeros((2, 4), dtype=jnp.int32) for _ in range(2))
    indices = tuple(
        jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (2, 4)) for _ in range(2)
    )
    valid = tuple(jnp.ones((2, 4), dtype=bool) for _ in range(2))
    relation = composite.relation(
        position,
        levels,
        indices,
        (centers, centers),
        valid,
    )
    composite_velocity = ((jnp.full((4,), 2.0), jnp.full((4,), -1.0)),)
    composite_diagnostics = composite.diagnostics(relation, composite_velocity, force)

    local = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    local_relation = local.relation(position)
    ownership = phx.discretization.DistributedMarkerOwnershipPlan(
        jnp.arange(2),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2, 1), dtype=jnp.int32),
        jnp.ones((2, 1), dtype=bool),
        rank_count=1,
    )
    multi_rank = phx.discretization.DistributedMarkerOwnershipPlan(
        jnp.arange(2),
        jnp.asarray([0, 1]),
        jnp.asarray([[0, 1], [0, 1]]),
        jnp.ones((2, 2), dtype=bool),
        rank_count=2,
    )
    distributed = phx.discretization.DistributedMACMarkerTransfer(local, ownership, 0)
    distributed_diagnostics = distributed.diagnostics(
        local_relation,
        mapped_velocity,
        force,
        lambda _operation, value: value,
    )
    coarse_impulse = composite.impulse_ledger(relation, force, 0.0, 0.1)
    fine_impulses = (
        composite.impulse_ledger(relation, force, 0.0, 0.05),
        composite.impulse_ledger(relation, force, 0.05, 0.1),
    )
    reflux = phx.discretization.reflux_composite_marker_impulse(
        coarse_impulse, fine_impulses
    )

    assert mapped_relation.successful
    assert mapped_diagnostics.successful
    assert relation.successful
    assert composite_diagnostics.successful
    assert jnp.allclose(
        composite.gather(relation, composite_velocity),
        jnp.asarray(
            [
                [2.0, -1.0],
                [2.0, -1.0],
            ]
        ),
    )
    assert distributed_diagnostics.successful
    assert reflux.successful
    assert jnp.allclose(reflux.correction_impulse, 0.0)
    assert not jnp.any(multi_rank.owner_mask(0) & multi_rank.owner_mask(1))
    assert jnp.all(multi_rank.owner_mask(0) | multi_rank.owner_mask(1))


def test_geometry_epochs_lubrication_and_qualification_contracts():
    atlas = phx.geometry.circle_boundary_atlas(
        jnp.asarray([0.5, 0.5]),
        jnp.asarray(0.2),
        source_id="qualification-circle",
    )
    quadrature = phx.geometry.ImmersedMarkerQuadraturePlan(
        jnp.arange(4),
        jnp.arange(4),
        jnp.full((4, 1), 0.5),
        jnp.full((4,), 0.25),
    )
    materialized = quadrature.materialize(atlas, 0.0)
    marker_plan = quadrature.marker_plan(materialized).prepare()
    kinematics = materialized.kinematics(marker_plan)

    source = phx.discretization.MarkerEpochPlan(
        jnp.asarray([0, 1]), jnp.asarray([1.0, 1.0])
    )
    target = phx.discretization.MarkerEpochPlan(jnp.arange(4), jnp.full((4,), 0.5))
    epoch = phx.discretization.MarkerEpochTransferPlan(
        source,
        target,
        jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    )
    transitioned = epoch.transition(
        phx.discretization.MarkerEpochState(
            jnp.asarray([2.0, 3.0]),
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            source.epoch_id,
        ),
        jnp.asarray(0.5),
    )

    lubrication = phx.discretization.ResolvedLubricationCorrectionPlan(
        1.0, 0.2, 0.01
    ).evaluate(
        jnp.asarray([0.02, 0.1]),
        jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([0.1, 0.1]),
        resolved_resistance=jnp.asarray([0.1, 0.1]),
    )
    mechanics = phx.discretization.MarkerMechanicsMigrationPlan(epoch, 2).migrate(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([[0.1, 0.0], [0.0, 0.1]]),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    spacing = jnp.asarray([0.5, 0.25, 0.125])
    order = phx.solver.observed_convergence_order(spacing, spacing**2)
    evidence = phx.solver.MarkerFlowQualificationEvidence(
        jnp.asarray(1.0e-10),
        jnp.asarray(1.0e-10),
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        order,
        order,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
    )
    qualified = phx.solver.MarkerFlowQualificationPlan(
        phx.solver.MarkerFlowQualificationProfile(
            "closure",
            require_contact=True,
            require_interface=True,
            require_stochastic=True,
        )
    ).evaluate(evidence)

    assert materialized.finite
    assert jnp.allclose(kinematics.position, materialized.position)
    assert transitioned.successful
    assert jnp.allclose(transitioned.candidate.value, jnp.asarray([2.0, 2.0, 3.0, 3.0]))
    assert mechanics.successful
    assert lubrication.finite
    assert lubrication.resistance[0] > lubrication.resistance[1] >= 0.0
    assert jnp.all(lubrication.dissipation_rate >= 0.0)
    assert jnp.isclose(order, 2.0)
    assert qualified.successful


def test_periodic_dfib_preserves_divergence_free_no_slip_state():
    finite_volume, operators, boundaries = _periodic_mac(count=6)
    position = jnp.asarray([[0.31, 0.37], [0.63, 0.58]])
    markers = _markers(position)
    regular = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    pressure = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=boundaries, solve_method="transform"
    )
    transfer = phx.solver.MACDivergenceFreeMarkerTransfer(regular, pressure)
    relation = regular.relation(position)
    velocity = (
        jnp.full(finite_volume.face_layouts[0].shape, 0.2),
        jnp.zeros(finite_volume.face_layouts[1].shape),
    )
    force = jnp.asarray([[0.2, -0.3], [-0.1, 0.4]])
    diagnostics = transfer.diagnostics(relation, velocity, force)
    plan = phx.solver.MACDFIBProjectionPlan(transfer)
    result = plan.project(
        velocity,
        1.0,
        markers.kinematics(
            position, jnp.broadcast_to(jnp.asarray([0.2, 0.0]), position.shape)
        ),
    )

    assert diagnostics.successful
    assert result.accepted
    assert result.divergence_norm < 1.0e-9
    assert result.slip_norm < 1.0e-9


def test_sharp_projection_and_variable_density_stage_inverse_preserve_zero_state():
    finite_volume, operators, boundaries = _periodic_mac(count=4)
    zero = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    one = tuple(jnp.ones(layout.shape) for layout in finite_volume.face_layouts)
    stage = boundaries.evaluate(jnp.asarray(0.0), None)
    inverse = phx.solver.MACVariableDensityStageInverseMomentum(
        operators,
        boundaries,
        stage,
        tuple(2.0 * value for value in one),
        0.1,
        stage_id="variable-density-stage",
    )
    applied = inverse.apply_inverse(one)
    sharp = phx.solver.MACSharpInterfaceProjectionPlan(
        operators,
        boundaries,
        jnp.ones(finite_volume.cell_shape),
        one,
        jnp.zeros(finite_volume.cell_shape),
        jnp.zeros(finite_volume.cell_shape + (2,)),
        jnp.zeros(finite_volume.cell_shape + (2,)),
        -jnp.ones(finite_volume.cell_shape, dtype=jnp.int32),
    )
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    variable_viscosity = phx.solver.MACVariableViscosityStagePlan(
        momentum,
        one,
        1.0
        + 0.2
        * jnp.arange(prod(finite_volume.cell_shape)).reshape(finite_volume.cell_shape),
        0.01,
        stage_id="variable-viscosity-stage",
    )
    viscous_action = variable_viscosity.momentum_operator.mv(one)
    result = sharp.project(zero, one, stage)
    geometry = phx.solver.MACSharpInterfaceGeometryData(
        jnp.ones(finite_volume.cell_shape),
        one,
        jnp.zeros(finite_volume.cell_shape),
        jnp.zeros(finite_volume.cell_shape + (2,)),
        jnp.zeros(finite_volume.cell_shape + (2,)),
        -jnp.ones(finite_volume.cell_shape, dtype=jnp.int32),
        jnp.zeros(finite_volume.cell_shape),
    )
    moving = phx.solver.MACMovingSharpInterfaceEpochPlan(
        operators,
        boundaries,
        lambda _time, _args: geometry,
        geometry_family_id="stationary-sharp-test",
    )
    epoch = moving.transition(0.0, geometry, 0.1)
    interface = phx.solver.MACImmersedInterfaceProjectionPlan(
        sharp,
        lambda _time, _geometry, _args: jnp.zeros(finite_volume.cell_shape),
        jump_id="zero-jump",
    )
    interface_result = interface.project(0.0, zero, one, stage)
    selector = phx.solver.MACInterfaceMethodSelector("immersed-interface", interface)
    cut_fraction = jnp.ones(finite_volume.cell_shape).at[0, 0].set(0.5)
    cut_area = jnp.zeros(finite_volume.cell_shape).at[0, 0].set(0.1)
    cut_normal = jnp.zeros(finite_volume.cell_shape + (2,)).at[0, 0, 0].set(1.0)
    cut_body = (-jnp.ones(finite_volume.cell_shape, dtype=jnp.int32)).at[0, 0].set(0)
    cut = phx.solver.MACSharpInterfaceProjectionPlan(
        operators,
        boundaries,
        cut_fraction,
        one,
        cut_area,
        jnp.zeros(finite_volume.cell_shape + (2,)),
        cut_normal,
        cut_body,
    )
    traction = cut.force(jnp.ones(finite_volume.cell_shape))

    assert all(jnp.allclose(value, 0.05) for value in applied)
    assert result.accepted
    assert result.divergence_norm < 1.0e-10
    assert jnp.allclose(result.force.force, 0.0)
    assert epoch.accepted
    assert interface_result.accepted
    assert selector.plan is interface
    assert operators.velocity_space.inner(one, viscous_action) > 0.0
    assert jnp.allclose(traction.force, jnp.asarray([-0.1, 0.0]))


def test_overdamped_fib_matches_free_diffusion_covariance():
    space = phx.linalg.ArraySpace((2048,))
    identity = phx.linalg.FunctionLinearOperator(
        lambda value: value,
        source=space,
        target=space,
        transpose_action=lambda value: value,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="free-fib-mobility",
    )
    step = 1.0e-2
    result = phx.solver.FIBOverdampedPlan(
        space,
        lambda _position: identity,
        temperature=1.0,
    ).step(
        jnp.zeros((2048,)),
        jnp.zeros((2048,)),
        step,
        phx.solver.StochasticReplayKey(
            jnp.asarray(19),
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0),
        ),
    )
    variance = jnp.mean(result.brownian_increment**2)

    assert result.accepted
    assert jnp.abs(variance - 2.0 * step) / (2.0 * step) < 0.15


def test_stochastic_replay_checkpoint_and_output_are_reproducible(tmp_path):
    space = phx.linalg.ArraySpace((2,))
    identity = phx.linalg.FunctionLinearOperator(
        lambda value: value,
        source=space,
        target=space,
        transpose_action=lambda value: value,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="marker-flow-identity",
    )
    stress = phx.solver.MACDiscreteStochasticStressPlan(
        identity, identity, stress_id="identity-stress"
    )
    thermal_plan = stress.thermalize(temperature=1.0)
    key = phx.solver.StochasticReplayKey(
        jnp.asarray(7),
        jnp.asarray(3),
        jnp.asarray(1),
        jnp.asarray(0),
    )
    thermal_first = thermal_plan.sample(0.1, key)
    thermal_second = thermal_plan.sample(0.1, key)
    fib = phx.solver.FIBOverdampedPlan(
        space,
        lambda _position: identity,
        temperature=1.0,
    ).step(jnp.zeros((2,)), jnp.zeros((2,)), 0.1, key)
    inertial = phx.solver.MACInertialStochasticStepPlan(
        space, identity, thermal_plan
    ).step(
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        0.1,
        key,
    )

    record = phx.solver.MarkerFlowReplayRecord(
        jnp.asarray([0.1, 0.2]),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([True, True]),
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([3, 4], dtype=jnp.int32),
        jnp.zeros((2, 1)),
        "closure-replay",
    )
    replay = phx.solver.MarkerFlowReplayPlan("closure-replay").replay(
        jnp.asarray(0.0),
        record,
        lambda state, step, _event, _counter, _route: (
            state + step,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )

    empty = jnp.zeros((0,))
    payload = phx.solver.MarkerFlowCheckpointPayload(
        jnp.asarray(0.2),
        jnp.asarray(2, dtype=jnp.int32),
        (jnp.asarray([1.0]),),
        jnp.asarray([0.0]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        key,
        empty,
    )
    checkpoint_plan = phx.solver.MarkerFlowCheckpointPlan(
        method_id="method",
        operator_id="operator",
        boundary_id="boundary",
        transfer_id="transfer",
    )
    checkpoint = tmp_path / "marker-flow.npz"
    phx.solver.write_marker_flow_checkpoint(checkpoint, checkpoint_plan, payload)
    restored = phx.solver.read_marker_flow_checkpoint(
        checkpoint, checkpoint_plan, payload
    )

    finite_volume, _, _ = _periodic_mac(count=4)
    output = phx.solver.MarkerFlowOutputPlan(
        tmp_path / "marker-flow-output.h5", finite_volume, jnp.asarray([0])
    )
    output.initialize(finite_volume)
    output.write_snapshot(
        finite_volume,
        0.2,
        2,
        jnp.asarray([[0.5, 0.5]]),
        eulerian_fields={"pressure": jnp.zeros(finite_volume.cell_shape)},
        marker_fields={"force": jnp.zeros((1, 2))},
        rigid_fields={"position": jnp.asarray([[0.5, 0.5]])},
        diagnostics={"divergence": jnp.asarray(0.0)},
    )
    load = phx.solver.HydrodynamicLoadPlan(jnp.asarray([0]), 2).record(
        0.0,
        0.1,
        jnp.asarray([[0.2, 0.0]]),
        jnp.zeros((1, 1)),
        pressure_force=jnp.zeros((1, 2)),
        pressure_torque=jnp.zeros((1, 1)),
        viscous_force=jnp.zeros((1, 2)),
        viscous_torque=jnp.zeros((1, 1)),
        marker_force=jnp.asarray([[1.0, 0.0]]),
        marker_torque=jnp.zeros((1, 1)),
        lubrication_force=jnp.zeros((1, 2)),
        lubrication_torque=jnp.zeros((1, 1)),
        contact_impulse=jnp.zeros((1, 2)),
        contact_angular_impulse=jnp.zeros((1, 1)),
    )
    restriction = phx.solver.MarkerFlowAdaptiveStepPlan(maximum_step=0.1).restrict(
        advection=0.05,
        diffusion=0.08,
        marker=0.07,
        contact=0.09,
        lubrication=0.06,
        geometry=0.1,
        stochastic=0.1,
    )
    trajectory = phx.solver.MarkerFlowTrajectoryAdapter(
        lambda state, step, _event, _counter, _route: (
            state + step,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        lambda time, state: jnp.stack((time, state)),
        adapter_id="marker-flow-test-trajectory",
    ).rollout(
        jnp.asarray(0.0),
        0.0,
        jnp.asarray([0.1, 0.1]),
        jnp.zeros((2, 1)),
        jnp.asarray([0, 1]),
        jnp.asarray([0, 0]),
    )
    artifact = phx.solver.marker_flow_artifact_reference(
        checkpoint, "checkpoint", checkpoint_plan.checkpoint_id
    )
    export = phx.solver.MarkerFlowCompiledExportPlan(
        payload,
        fixed_routes=True,
        fixed_topology=True,
        fixed_random_schedule=True,
    ).validate(restored)

    assert thermal_first.report.passed
    assert jax.tree.all(
        jax.tree.map(
            jnp.array_equal,
            thermal_first.stochastic_momentum,
            thermal_second.stochastic_momentum,
        )
    )
    assert fib.accepted
    assert inertial.accepted
    assert replay.successful
    assert replay.time_match
    assert jnp.isclose(replay.state, 0.2)
    assert jnp.array_equal(restored.fluid_state[0], payload.fluid_state[0])
    assert load.successful
    assert restriction.successful
    assert restriction.limiter == "advection"
    assert trajectory.successful
    assert artifact.byte_count > 0
    assert export.exportable
    with h5py.File(tmp_path / "marker-flow-output.h5", "r") as handle:
        assert "steps/0000000002/markers/force" in handle
        assert "steps/0000000002/rigid/position" in handle
