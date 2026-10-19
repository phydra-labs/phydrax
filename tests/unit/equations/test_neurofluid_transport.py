from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._metric_network import NetworkEvidence
from tests._support import cell_meshes


def _bulk() -> Any:
    return phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.asarray(((0, 1, 2, 3),)),
    )


def _network(contract: Any) -> Any:
    return phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((0.2, 0.2, 0.15), (0.2, 0.2, 0.25))),
        np.asarray(((0, 1),)),
        contract,
        areas=np.asarray((0.01,)),
        perimeters=np.asarray((0.2,)),
        root_vertex_ids=np.asarray((0,)),
        tip_vertex_ids=np.asarray((1,)),
    ).prepare()


def _runtime(
    *,
    boundary_flux: Any = None,
    removal_rate: Any = None,
    inflow: float = 0.0,
    mesh: Any = None,
    bulk_region_ids: tuple[str, ...] | None = None,
) -> Any:
    contract = phx.SpatialCoordinateContract.si()
    bulk = _bulk() if mesh is None else mesh
    count = sum(block.cell_count for block in bulk.blocks)
    network = _network(contract)
    bulk_plan = phx.equations.BulkDGTransportPlan(
        bulk,
        np.full((count,), 0.8, dtype=np.float64),
        np.full((count,), 0.01, dtype=np.float64),
        np.zeros((count, 3), dtype=np.float64),
        boundary_volume_flux=boundary_flux,
        # ty: ignore[invalid-argument-type]
        boundary_inflow_concentration=inflow,
        removal_rate=removal_rate,
    )
    bulk_prepared = bulk_plan.prepare()
    transfer = phx.discretization.EmbeddedMeasureTransferPlan(
        bulk,
        contract,
        np.asarray(network.mesh.coordinates),
        np.asarray(network.node_measures),
        np.asarray(bulk_prepared.mass),
        network.network_id,
        phx.discretization.CircleAverageKernel(0.01, 8),
        phx.discretization.EmbeddedSourceAssociation.CELL,
        np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
    ).prepare()
    return phx.equations.MixedDimensionalTransportPlan(
        bulk_plan,
        phx.equations.NetworkTransportPlan(
            network, np.asarray((0.01,)), np.asarray((0.0,))
        ),
        transfer,
        np.asarray((0.5, 0.5)),
        np.asarray((1,)),
        np.asarray((0.003,)),
        np.asarray((0.2,)),
        bulk_region_ids=bulk_region_ids,
    ).prepare()


def test_neurofluid_transport_scenario_1() -> None:
    runtime = _runtime()
    transfer = runtime.exchange.transfer
    sampled = transfer.average(np.asarray((3.0,)))
    np.testing.assert_allclose(sampled.values, (3.0, 3.0), atol=1e-12)
    assert bool(sampled.evidence.successful)
    bulk, network, _ = runtime.exchange.residuals(
        np.asarray((3.0,)), np.asarray((1.0, 2.0))
    )
    np.testing.assert_allclose(np.sum(bulk) + np.sum(network), 0.0, atol=1e-12)
    runtime = _runtime()
    state = phx.equations.MixedDimensionalTransportState(
        # ty: ignore[invalid-argument-type]
        np.asarray((2.0,)),
        # ty: ignore[invalid-argument-type]
        np.asarray((2.0, 2.0)),
        # ty: ignore[invalid-argument-type]
        np.asarray((2.0,)),
    )
    for step in (
        runtime.step_backward_euler(state, 0.1),
        runtime.step_imex_euler(state, 0.1),
    ):
        np.testing.assert_allclose(step.state.bulk, state.bulk, atol=1e-11)
        np.testing.assert_allclose(step.state.network, state.network, atol=1e-11)
        np.testing.assert_allclose(step.state.reservoirs, state.reservoirs, atol=1e-11)
        assert bool(step.accepted)
        assert bool(step.ledger.successful)
    runtime = _runtime(boundary_flux=0.1, removal_rate=0.2)
    state = phx.equations.MixedDimensionalTransportState(
        # ty: ignore[invalid-argument-type]
        np.asarray((2.0,)),
        # ty: ignore[invalid-argument-type]
        np.asarray((2.0, 2.0)),
        # ty: ignore[invalid-argument-type]
        np.asarray((2.0,)),
    )
    initial = float(runtime.ledger(state).total_mass)
    step = runtime.step_backward_euler(state, 0.05)
    assert bool(step.accepted)
    assert float(step.ledger.total_mass) < initial
    assert float(step.ledger.external_loss_rate) > 0.0
    assert float(step.ledger.balance_defect) < 1.0e-9


def test_network_pressure_flow_balances_interior_nodes() -> None:
    contract = phx.SpatialCoordinateContract.si()
    network = phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        np.asarray(((0, 1), (1, 2))),
        contract,
        areas=np.ones(2),
        perimeters=np.ones(2),
        root_vertex_ids=np.asarray((0,)),
        tip_vertex_ids=np.asarray((2,)),
    ).prepare()
    result = phx.applications.neurofluid.PVSNetworkFlowPlan(
        network,
        np.asarray((2.0, 1.0)),
        np.asarray((0, 2)),
        np.asarray((1.0, 0.0)),
    ).solve()
    np.testing.assert_allclose(result.volume_flow[0], result.volume_flow[1], atol=1e-12)
    assert bool(result.evidence.successful)


def _independent_transport_matrix() -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray((0.8 / 6.0, 0.0005, 0.0005, 0.003), dtype=np.float64)
    stiffness = np.zeros((4, 4), dtype=np.float64)
    for first, second, conductance in (
        (0, 1, 0.00025),
        (0, 2, 0.00025),
        (1, 2, 0.001),
        (2, 3, 0.0006),
    ):
        difference = (
            np.eye(4, dtype=np.float64)[first] - np.eye(4, dtype=np.float64)[second]
        )
        stiffness += conductance * np.outer(difference, difference)
    return mass, stiffness


def test_closed_transport_full_sensitivity_and_inventory() -> None:
    runtime = _runtime()
    mass, stiffness = _independent_transport_matrix()
    initial = jnp.asarray((3.0, 1.0, 1.5, 0.5), dtype=jnp.float64)
    direction = jnp.sin(jnp.arange(4, dtype=jnp.float64) + 1.0)

    def evolve(owner: Any, value: jax.Array) -> jax.Array:
        def advance(current: jax.Array, unused: None) -> tuple[jax.Array, None]:
            del unused
            step = owner.step_backward_euler(owner.unflatten(current), 0.01)
            return owner.flatten(step.state), None

        return jax.lax.scan(advance, value, None, length=5)[0]

    primal, tangent = jax.jit(
        lambda owner, value, vector: jax.jvp(
            lambda current: evolve(owner, current), (value,), (vector,)
        )
    )(runtime, initial, direction)
    matrix = np.diag(mass) + 0.01 * stiffness
    expected = np.asarray(direction)
    for _ in range(5):
        expected = np.linalg.solve(matrix, mass * expected)
    np.testing.assert_allclose(tangent, expected, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(mass @ (primal - initial), 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(mass @ (tangent - direction), 0.0, atol=1e-12, rtol=0.0)
    seed = jnp.asarray((0.3, -0.7, 0.4, 1.2), dtype=jnp.float64)
    gradient = jax.jit(
        lambda owner, value, cotangent: jax.vjp(
            lambda current: evolve(owner, current), value
        )[1](cotangent)[0]
    )(runtime, initial, seed)
    expected_gradient = np.asarray(seed)
    for _ in range(5):
        expected_gradient = mass * np.linalg.solve(matrix.T, expected_gradient)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(seed @ tangent, gradient @ direction, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("imex", (False, True))
@pytest.mark.parametrize("boundary_flux", (0.001, -0.001))
def test_open_transport_source_removal_balance_and_full_ad(
    imex: bool, boundary_flux: float
) -> None:
    runtime = _runtime(boundary_flux=boundary_flux, removal_rate=0.2, inflow=1.3)
    mass, stiffness = _independent_transport_matrix()
    initial = jnp.asarray((3.0, 1.0, 1.5, 0.5), dtype=jnp.float64)
    source = np.asarray((0.03, 0.002, 0.004, 0.001), dtype=np.float64)
    dt = 0.05

    def advance(owner: Any, parameter: jax.Array, value: jax.Array) -> jax.Array:
        prepared = eqx.tree_at(
            lambda current: (
                current.bulk.removal_rate,
                current.bulk.boundary_volume_flux,
            ),
            owner,
            (
                owner.bulk.removal_rate * (1.0 + parameter),
                owner.bulk.boundary_volume_flux * (1.0 + parameter),
            ),
        )
        sources = phx.equations.MixedDimensionalSources(
            jnp.asarray(source[:1]) * (1.0 + parameter),
            jnp.asarray(source[1:3]) * (1.0 + parameter),
            jnp.asarray(source[3:]) * (1.0 + parameter),
        )
        method = prepared.step_imex_euler if imex else prepared.step_backward_euler
        step = method(prepared.unflatten(value), dt, sources=sources)
        return prepared.flatten(step.state)

    parameter = jnp.asarray(0.2, dtype=jnp.float64)
    direction = jnp.asarray((0.4, -0.2, 0.3, -0.1), dtype=jnp.float64)
    seed = jnp.asarray((0.1, -0.4, 0.6, 0.9), dtype=jnp.float64)
    (value, tangent), (parameter_gradient, state_gradient) = jax.jit(
        lambda owner, theta, current, vector, cotangent: (
            jax.jvp(
                lambda p, x: advance(owner, p, x),
                (theta, current),
                (jnp.ones_like(theta), vector),
            ),
            jax.vjp(lambda p, x: advance(owner, p, x), theta, current)[1](cotangent),
        )
    )(runtime, parameter, initial, direction, seed)
    removal = mass[0] * 0.2
    outflow = 4.0 * max(boundary_flux, 0.0)
    inflow = 4.0 * min(boundary_flux, 0.0) * 1.3
    implicit_loss = removal + (0.0 if imex else outflow)
    matrix = np.diag(mass) + dt * stiffness
    matrix[0, 0] += dt * 1.2 * implicit_loss
    matrix_direction = np.zeros((4, 4), dtype=np.float64)
    matrix_direction[0, 0] = dt * implicit_loss
    explicit = np.zeros(4, dtype=np.float64)
    explicit[0] = inflow + (outflow * initial[0] if imex else 0.0)
    right = mass * initial + dt * 1.2 * (source - explicit)
    expected = np.linalg.solve(matrix, right)
    right_direction = mass * np.asarray(direction) + dt * (source - explicit)
    if imex:
        right_direction[0] -= dt * 1.2 * outflow * direction[0]
    tangent_right = right_direction - matrix_direction @ np.asarray(value)
    expected_tangent = np.linalg.solve(matrix, tangent_right)
    np.testing.assert_allclose(value, expected, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(tangent, expected_tangent, atol=1e-10, rtol=0.0)
    loss = removal * value[0] + inflow
    loss += outflow * (initial[0] if imex else value[0])
    np.testing.assert_allclose(
        mass @ (value - initial),
        dt * 1.2 * (source.sum() - loss),
        atol=1e-12,
        rtol=0.0,
    )
    # Loss and source sensitivities are not closed-system invariants.
    loss_direction = removal * tangent[0]
    loss_direction += outflow * (direction[0] if imex else tangent[0])
    np.testing.assert_allclose(
        mass @ (tangent - direction),
        dt * (source.sum() - loss) - dt * 1.2 * loss_direction,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        seed @ tangent,
        parameter_gradient + state_gradient @ direction,
        atol=1e-10,
        rtol=0.0,
    )
    adjoint = np.linalg.solve(matrix.T, seed)
    expected_state_gradient = mass * adjoint
    if imex:
        expected_state_gradient[0] -= dt * 1.2 * outflow * adjoint[0]
    np.testing.assert_allclose(
        state_gradient, expected_state_gradient, atol=1e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        parameter_gradient,
        adjoint @ (dt * (source - explicit) - matrix_direction @ value),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize("storage_refusal", (False, True))
def test_transport_failed_full_solve_rolls_back_every_field_under_jit(
    storage_refusal: bool,
) -> None:
    runtime = _runtime(mesh=cell_meshes.kuhn_cube())
    initial = runtime.unflatten(
        jnp.concatenate(
            (
                3.0 + 0.1 * jnp.arange(runtime.sizes[0], dtype=jnp.float64),
                jnp.asarray((1.0, 1.5, 0.5), dtype=jnp.float64),
            )
        )
    )
    if storage_refusal:
        policy = phx.linalg.LinearSolvePolicy(
            resources=phx.linalg.SolveResourcePolicy(
                factorization_bytes=0,
                workspace_bytes=0,
                krylov_basis_bytes=0,
                preconditioner_bytes=0,
                recycling_state_bytes=0,
            )
        )
    else:
        policy = phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=1),
            tolerance=phx.linalg.TolerancePolicy(max_steps=1),
        )
    step = jax.jit(
        lambda owner, state: owner.step_backward_euler(state, 0.4, policy=policy)
    )(runtime, initial)
    assert not bool(step.accepted)
    assert int(step.linear_status) != int(phx.linalg.LinearSolveStatus.SUCCESS)
    for actual, previous in zip(
        jax.tree.leaves(step.state), jax.tree.leaves(initial), strict=True
    ):
        np.testing.assert_array_equal(actual, previous)
    mass = np.asarray(runtime.mass_vector(jnp.float64))
    candidate = np.asarray(runtime.flatten(step.candidate_state))
    physical, _ = runtime.residual(step.candidate_state)
    residual = mass * (candidate - runtime.flatten(initial)) + 0.4 * runtime.flatten(
        physical
    )
    threshold = 1e-10 + 1e-8 * np.linalg.norm(mass * runtime.flatten(initial))
    assert np.linalg.norm(residual) > threshold


def _single_bulk_cell_runtime() -> Any:
    runtime = _runtime(removal_rate=0.2)
    source = runtime.network.network
    geometry = phx.discretization.CellGeometrySpec.affine(source.mesh)
    seed = cell_meshes.storage(source.mesh, geometry)
    banks = dict(seed.logical_arrays)
    banks["closure/cell_ids"] = jnp.broadcast_to(
        source.mesh.entity_set(1).entity_ids[0], (2, 1)
    )
    banks["closure/cell_valid"] = jnp.asarray(((True,), (False,)), dtype=jnp.bool_)
    banks["closure/cell_owner"] = jnp.zeros((2, 1), dtype=jnp.int32)
    arrays = tuple(sorted(banks.items()))
    projection = phx.discretization.CellGeometryStorageProjection(
        arrays,
        seed.logical_coordinate_geometry_id,
        seed.global_coordinate_count,
        source_elements={
            block.name: element
            for block, element in zip(source.mesh.blocks, geometry.elements, strict=True)
        },
    )
    empty_ids = tuple(np.empty((0,), dtype=np.int64) for _ in seed.global_entity_counts)
    empty_owners = tuple(
        np.empty((0,), dtype=np.int32) for _ in seed.global_entity_counts
    )
    points = np.empty((0, 3), dtype=np.float64)
    storage = phx.discretization.CellMeshStorage(
        seed.global_entity_counts,
        empty_ids,
        empty_owners,
        partition_index=1,
        partition_count=2,
        logical_topology_id=seed.logical_topology_id,
        logical_geometry_id=seed.logical_geometry_id,
        logical_coordinate_geometry_id=seed.logical_coordinate_geometry_id,
        evidence_id=seed.evidence_id,
        logical_arrays=arrays,
        local_coordinates=points,
        local_blocks=(),
        global_coordinate_count=seed.global_coordinate_count,
        coordinate_global_ids=np.empty((0,), dtype=np.int64),
        coordinate_owner=np.empty((0,), dtype=np.int32),
        geometry_projection=projection,
    )
    mesh = phx.discretization.CellMesh(points, (), storage=storage)
    lengths, areas = source.lengths[:0], source.areas[:0]
    finite = jnp.all(jnp.isfinite(lengths)) & jnp.all(jnp.isfinite(areas))
    component_count = jnp.asarray(0, dtype=jnp.int32)
    evidence = NetworkEvidence(
        component_count,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.nan, dtype=lengths.dtype),
        jnp.asarray(jnp.nan, dtype=areas.dtype),
        finite,
        finite & (component_count == 1),
    )
    network = phx.discretization.PreparedMetricNetwork(
        mesh,
        source.coordinate_contract,
        source.senders[:0],
        source.receivers[:0],
        lengths,
        source.tangents[:0],
        areas,
        source.perimeters[:0],
        source.node_measures[:0],
        source.root_mask[:0],
        source.tip_mask[:0],
        evidence,
        f"{source.network_id}:empty-resident-subcomplex",
    )
    prepared_network = phx.equations.PreparedNetworkTransport(
        network,
        runtime.network.diffusivity[:0],
        runtime.network.volume_flow[:0],
        f"{runtime.network.prepared_id}:empty-resident-subcomplex",
    )
    transfer = phx.discretization.EmbeddedMeasureTransferPlan(
        runtime.bulk.mesh,
        source.coordinate_contract,
        points,
        np.empty((0,), dtype=np.float64),
        np.asarray(runtime.bulk.mass),
        network.network_id,
        phx.discretization.CircleAverageKernel(0.01, 8),
        phx.discretization.EmbeddedSourceAssociation.CELL,
        points,
    ).prepare()
    return phx.equations.PreparedMixedDimensionalTransport(
        runtime.bulk,
        prepared_network,
        phx.equations.PermeabilityExchangePlan(
            transfer, np.empty((0,), dtype=np.float64)
        ),
        phx.equations.ReservoirCouplingPlan(
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            network_size=0,
        ),
        jnp.ones((1, 1), dtype=jnp.float64),
        ("bulk",),
        True,
        f"{runtime.runtime_id}:single-bulk-cell",
    )


@pytest.mark.parametrize("imex", (False, True))
def test_single_bulk_cell_without_active_network_or_reservoirs(imex: bool) -> None:
    runtime = _single_bulk_cell_runtime()
    assert runtime.sizes == (1, 0, 0)
    old = jnp.asarray((3.0,), dtype=jnp.float64)
    source = jnp.asarray((0.03,), dtype=jnp.float64)
    empty = jnp.empty((0,), dtype=jnp.float64)
    dt = 0.05

    def response(owner: Any, value: jax.Array, forcing: jax.Array) -> jax.Array:
        state = owner.unflatten(value)
        sources = phx.equations.MixedDimensionalSources(forcing, empty, empty)
        method = owner.step_imex_euler if imex else owner.step_backward_euler
        return owner.flatten(method(state, dt, sources=sources).state)

    (value, tangent), (state_gradient, source_gradient) = eqx.filter_jit(
        lambda owner, current, forcing: (
            jax.jvp(
                lambda x, s: response(owner, x, s),
                (current, forcing),
                (jnp.ones_like(current), jnp.ones_like(forcing)),
            ),
            jax.vjp(lambda x, s: response(owner, x, s), current, forcing)[1](
                jnp.ones_like(current)
            ),
        )
    )(runtime, old, source)
    mass = 0.8 / 6.0
    denominator = mass * (1.0 + dt * 0.2)
    expected = (mass * old + dt * source) / denominator
    np.testing.assert_allclose(value, expected, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(tangent, (mass + dt) / denominator, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(state_gradient, mass / denominator, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(source_gradient, dt / denominator, atol=1e-13, rtol=0.0)

    def take_step(owner: Any, state: Any, forcing: jax.Array) -> Any:
        method = owner.step_imex_euler if imex else owner.step_backward_euler
        return method(
            state,
            dt,
            sources=phx.equations.MixedDimensionalSources(forcing, empty, empty),
        )

    step = eqx.filter_jit(take_step)(runtime, runtime.unflatten(old), source)
    assert bool(step.accepted)
    assert bool(step.ledger.successful)
    assert step.state.network.shape == step.state.reservoirs.shape == (0,)
    np.testing.assert_allclose(
        mass * (value - old),
        dt * (source - 0.2 * mass * value),
        atol=1e-14,
        rtol=0.0,
    )


def test_zero_target_transfer_retains_balance_and_adjoint_evidence() -> None:
    transfer = _single_bulk_cell_runtime().exchange.transfer
    source = jnp.asarray(((2.0 + 1.0j, 0.3 - 0.7j),), dtype=jnp.complex128)
    direction = jnp.ones_like(source)
    cotangent = jnp.empty((0, 2), dtype=jnp.complex128)
    result = eqx.filter_jit(lambda owner, values: owner.average(values))(transfer, source)
    assert result.values.shape == (0, 2)
    assert result.evidence.coverage_fraction.shape == (0,)
    assert bool(result.evidence.finite)
    assert bool(result.evidence.successful)
    for residual in (
        result.evidence.constant_residual,
        result.evidence.dual_pairing_residual,
        result.evidence.weighted_adjoint_residual,
    ):
        np.testing.assert_array_equal(residual, 0.0)
    (value, tangent), gradient = eqx.filter_jit(
        lambda owner, values: (
            jax.jvp(lambda x: owner.average(x).values, (values,), (direction,)),
            jax.vjp(lambda x: owner.average(x).values, values)[1](cotangent)[0],
        )
    )(transfer, source)
    matrix = np.empty((0, 1), dtype=np.float64)
    np.testing.assert_array_equal(value, matrix @ source)
    np.testing.assert_array_equal(tangent, matrix @ direction)
    expected_adjoint = matrix.T @ cotangent
    np.testing.assert_array_equal(gradient, expected_adjoint)
    np.testing.assert_array_equal(transfer.dual_pullback(cotangent), expected_adjoint)
    np.testing.assert_array_equal(transfer.hilbert_adjoint(cotangent), expected_adjoint)
    rejected = transfer.average(source.at[0, 0].set(jnp.nan))
    assert not bool(rejected.evidence.finite)
    assert not bool(rejected.evidence.successful)


def test_nonempty_transfer_retains_constant_and_source_finiteness_gates() -> None:
    transfer = _runtime().exchange.transfer
    source = jnp.asarray((3.0,), dtype=jnp.float64)
    result = eqx.filter_jit(lambda owner, values: owner.average(values))(transfer, source)
    np.testing.assert_allclose(result.values, (3.0, 3.0), atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(result.evidence.coverage_fraction, (1.0, 1.0))
    assert bool(result.evidence.finite)
    assert bool(result.evidence.successful)
    rejected = transfer.average(jnp.asarray((jnp.nan,), dtype=jnp.float64))
    assert bool(jnp.all(jnp.isnan(rejected.values)))
    assert not bool(rejected.evidence.finite)
    assert not bool(rejected.evidence.successful)


@pytest.mark.parametrize("imex", (False, True))
@pytest.mark.parametrize("boundary_flux", (-0.001, 0.001))
def test_inventory_rate_record_and_original_full_system_audit(
    imex: bool, boundary_flux: float
) -> None:
    runtime = _runtime(boundary_flux=boundary_flux, removal_rate=0.2, inflow=1.3)
    old = jnp.asarray((3.0, 1.0, 1.5, 0.5), dtype=jnp.float64)
    source = jnp.asarray((0.03, 0.002, 0.004, 0.001), dtype=jnp.float64)
    forcing = phx.equations.MixedDimensionalSources(source[:1], source[1:3], source[3:])
    dt = 0.05

    def advance(owner: Any, values: jax.Array) -> Any:
        method = owner.step_imex_euler if imex else owner.step_backward_euler
        return method(owner.unflatten(values), dt, sources=forcing)

    step = jax.jit(advance)(runtime, old)
    assert bool(step.accepted)
    assert step.solved_variable == "inventory-rate-increment"
    np.testing.assert_array_equal(step.step_width, dt)
    mass, stiffness = _independent_transport_matrix()
    advective = np.zeros((4, 4), dtype=np.float64)
    advective[0, 0] = 4.0 * max(boundary_flux, 0.0)
    offset = np.zeros((4,), dtype=np.float64)
    offset[0] = 4.0 * min(boundary_flux, 0.0) * 1.3
    stiffness[0, 0] += 0.2 * mass[0]
    implicit = stiffness + (0.0 if imex else advective)
    explicit = advective @ old + offset if imex else np.zeros((4,), dtype=np.float64)
    implicit_offset = np.zeros((4,), dtype=np.float64) if imex else offset
    matrix = np.diag(mass) + dt * implicit
    right = mass * old + dt * (source - explicit - implicit_offset)
    rate_right = source - explicit - (implicit @ old + implicit_offset)
    candidate = np.asarray(runtime.flatten(step.candidate_state))
    rate = np.asarray(step.increment_solve.value)
    original_residual = matrix @ candidate - right
    rate_residual = (matrix / mass[None, :]) @ rate - rate_right
    envelope = (
        512.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            np.linalg.norm(matrix) * np.linalg.norm(candidate) + np.linalg.norm(right),
        )
    )
    np.testing.assert_allclose(candidate, old + dt * rate / mass, atol=envelope, rtol=0.0)
    np.testing.assert_allclose(
        step.original_residual_norm,
        np.linalg.norm(original_residual),
        atol=envelope,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        step.increment_solve.full_residual_norm,
        np.linalg.norm(rate_residual),
        atol=envelope,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        step.original_residual_threshold,
        1e-10 + 1e-8 * np.linalg.norm(right),
        atol=128.0 * np.finfo(np.float64).eps,
        rtol=0.0,
    )
    assert int(step.original_residual_status) == int(phx.linalg.LinearSolveStatus.SUCCESS)
    assert float(step.original_residual_norm) <= float(step.original_residual_threshold)
    assert int(step.increment_solve.aggregate_rank) == len(runtime.inventory_group_ids)
    assert runtime.inventory_group_ids == ("bulk", "network", "reservoirs")


def test_authored_region_equations_hold_even_when_free_solve_refuses() -> None:
    regions = ("first", "second", "first", "second", "first", "second")
    runtime = _runtime(mesh=cell_meshes.kuhn_cube(), bulk_region_ids=regions)
    old = jnp.concatenate(
        (
            2.0 + jnp.sin(jnp.arange(6, dtype=jnp.float64)),
            jnp.asarray((1.0, 1.5, 0.5), dtype=jnp.float64),
        )
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=1),
        tolerance=phx.linalg.TolerancePolicy(max_steps=1),
    )
    step = jax.jit(
        lambda owner, values: owner.step_backward_euler(
            owner.unflatten(values), 0.4, policy=policy
        )
    )(runtime, old)
    assert not bool(step.accepted)
    assert int(step.linear_status) != int(phx.linalg.LinearSolveStatus.SUCCESS)
    np.testing.assert_array_equal(runtime.flatten(step.state), old)
    candidate = runtime.flatten(step.candidate_state)
    residual, _ = runtime.residual(step.candidate_state)
    physical = runtime.mass_vector(jnp.float64) * (
        candidate - old
    ) / 0.4 + runtime.flatten(residual)
    assert runtime.inventory_group_ids == (
        "bulk:first",
        "bulk:second",
        "network",
        "reservoirs",
    )
    row_residual = runtime.inventory_aggregates.T @ physical
    envelope = (
        512.0
        * np.finfo(np.float64).eps
        * len(old)
        * max(1.0, float(jnp.linalg.norm(physical)))
    )
    np.testing.assert_allclose(row_residual, 0.0, atol=envelope, rtol=0.0)
    assert int(step.increment_solve.aggregate_rank) == 4
    assert float(jnp.max(jnp.abs(physical))) > 1e-9
