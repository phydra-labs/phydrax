#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import prod

import jax.numpy as jnp

import phydrax as phx


def _finite_volume(*, periodic, count=4):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    return phx.discretization.FiniteVolumePlan(grid).prepare()


def test_mac_marker_transfer_is_dual_measure_adjoint():
    finite_volume = _finite_volume(periodic=True)
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, 0.4, 8).prepare()
    relation = transfer.relation(jnp.asarray([[0.25, 0.25], [0.75, 0.75]]))
    velocity = tuple(
        jnp.full(layout.shape, 0.2 * (axis + 1))
        for axis, layout in enumerate(finite_volume.face_layouts)
    )
    marker_force = jnp.asarray([[0.3, -0.1], [-0.2, 0.4]])
    diagnostics = transfer.diagnostics(relation, velocity, marker_force)

    assert relation.successful
    assert diagnostics.successful
    assert jnp.abs(diagnostics.work_adjoint_residual) < 1e-10
    assert jnp.max(jnp.abs(diagnostics.force_residual)) < 1e-10


def test_single_device_distributed_projection_preserves_global_contract():
    finite_volume = _finite_volume(periodic=True)
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    topology = phx.discretization.MACDistributedTopologyPlan.single_device(
        operators
    ).prepare(momentum)
    velocity = tuple(
        jnp.sin(jnp.arange(prod(layout.shape)).reshape(layout.shape))
        for layout in finite_volume.face_layouts
    )
    state = topology.distribute(jnp.zeros(finite_volume.cell_shape), velocity)
    projection = phx.solver.MACDistributedProjectionPlan(
        topology, relative_tolerance=1e-8, absolute_tolerance=1e-8
    ).project(state, 1.0)

    assert topology.status.ready
    assert projection.converged
    assert projection.divergence_norm < 1e-7
    assert projection.gauge_defect < 1e-8


def test_identity_mapped_and_ale_geometries_preserve_free_stream():
    finite_volume = _finite_volume(periodic=False)
    mapped = phx.discretization.MappedMACGeometryPlan(
        finite_volume, lambda points: points, mapping_id="identity-map"
    ).prepare()
    ale = phx.solver.MACALEGeometryPlan(
        finite_volume,
        lambda _time, points, _args: points,
        lambda _time, points, _args: jnp.zeros_like(points),
        mapping_id="stationary-identity-ale",
    )
    stage = ale.evaluate(0.0)
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    result = ale.advance(velocity, 0.0, 0.01, viscosity=0.0)

    assert mapped.report.passed
    assert stage.passed
    assert stage.maximum_gcl_residual < 1e-10
    assert result.success
    assert jnp.linalg.norm(result.divergence_after) < 1e-8


def test_identity_remesh_epoch_preserves_cells_flux_and_momentum():
    finite_volume = _finite_volume(periodic=False)
    mapped = phx.discretization.MappedMACGeometryPlan(
        finite_volume, lambda points: points, mapping_id="identity-remesh-map"
    ).prepare()
    cell_count = mapped.cell_volumes.size
    face_count = sum(prod(layout.shape) for layout in finite_volume.face_layouts)
    remesh = phx.solver.MACRemeshEpochPlan(
        mapped,
        mapped,
        jnp.arange(cell_count + 1),
        jnp.arange(cell_count),
        mapped.cell_volumes.reshape((-1,)),
        jnp.arange(face_count + 1),
        jnp.arange(face_count),
        jnp.ones(face_count),
        jnp.ones(face_count),
    )
    cells = jnp.ones(finite_volume.cell_shape)
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    result = remesh.execute(cells, velocity)

    assert result.success
    assert not result.differentiation_certified
    assert result.maximum_target_coverage_defect < 1e-10
    assert result.maximum_source_coverage_defect < 1e-10
    assert jnp.max(jnp.abs(result.cell_conservation_residual)) < 1e-10
