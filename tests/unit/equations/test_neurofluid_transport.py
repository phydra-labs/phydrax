import numpy as np

import phydrax as phx


def _bulk():
    return phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.asarray(((0, 1, 2, 3),)),
    )


def _network(contract):
    return phx.discretization.MetricNetworkPlan.from_arrays(
        np.asarray(((0.2, 0.2, 0.15), (0.2, 0.2, 0.25))),
        np.asarray(((0, 1),)),
        contract,
        areas=np.asarray((0.01,)),
        perimeters=np.asarray((0.2,)),
        root_vertex_ids=np.asarray((0,)),
        tip_vertex_ids=np.asarray((1,)),
    ).prepare()


def _runtime(*, boundary_flux=None, removal_rate=None):
    contract = phx.SpatialCoordinateContract.si()
    bulk = _bulk()
    network = _network(contract)
    bulk_plan = phx.equations.BulkDGTransportPlan(
        bulk,
        np.asarray((0.8,)),
        np.asarray((0.01,)),
        np.zeros((1, 3)),
        boundary_volume_flux=boundary_flux,
        boundary_inflow_concentration=0.0,
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
    ).prepare()


def test_metric_network_and_embedded_transfer_are_conservative():
    runtime = _runtime()
    transfer = runtime.exchange.transfer
    sampled = transfer.average(np.asarray((3.0,)))
    np.testing.assert_allclose(sampled.values, (3.0, 3.0), atol=1e-12)
    assert bool(sampled.evidence.successful)
    bulk, network, _ = runtime.exchange.residuals(
        np.asarray((3.0,)), np.asarray((1.0, 2.0))
    )
    np.testing.assert_allclose(np.sum(bulk) + np.sum(network), 0.0, atol=1e-12)


def test_backward_euler_and_imex_preserve_constant_closed_state():
    runtime = _runtime()
    state = phx.equations.MixedDimensionalTransportState(
        np.asarray((2.0,)), np.asarray((2.0, 2.0)), np.asarray((2.0,))
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


def test_boundary_outflow_and_removal_close_mass_ledger():
    runtime = _runtime(boundary_flux=0.1, removal_rate=0.2)
    state = phx.equations.MixedDimensionalTransportState(
        np.asarray((2.0,)), np.asarray((2.0, 2.0)), np.asarray((2.0,))
    )
    initial = float(runtime.ledger(state).total_mass)
    step = runtime.step_backward_euler(state, 0.05)
    assert bool(step.accepted)
    assert float(step.ledger.total_mass) < initial
    assert float(step.ledger.external_loss_rate) > 0.0
    assert float(step.ledger.balance_defect) < 1.0e-9


def test_network_pressure_flow_balances_interior_nodes():
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
