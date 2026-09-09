#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Advance a closed 3D--1D--0D tracer system and verify total mass."""

import json

import numpy as np

import phydrax as phx


contract = phx.SpatialCoordinateContract.si()
bulk_mesh = phx.discretization.CellMesh.from_tetrahedra(
    np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    np.asarray(((0, 1, 2, 3),)),
)
network = phx.discretization.MetricNetworkPlan.from_arrays(
    np.asarray(((0.2, 0.2, 0.15), (0.2, 0.2, 0.25))),
    np.asarray(((0, 1),)),
    contract,
    areas=np.asarray((0.01,)),
    perimeters=np.asarray((0.2,)),
    root_vertex_ids=np.asarray((0,)),
    tip_vertex_ids=np.asarray((1,)),
).prepare()
bulk = phx.equations.BulkDGTransportPlan(
    bulk_mesh, np.asarray((0.8,)), np.asarray((0.01,)), np.zeros((1, 3))
)
bulk_prepared = bulk.prepare()
transfer = phx.discretization.EmbeddedMeasureTransferPlan(
    bulk_mesh,
    contract,
    np.asarray(network.mesh.coordinates),
    np.asarray(network.node_measures),
    np.asarray(bulk_prepared.mass),
    network.network_id,
    phx.discretization.CircleAverageKernel(0.01, 8),
    phx.discretization.EmbeddedSourceAssociation.CELL,
    np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
).prepare()
runtime = phx.equations.MixedDimensionalTransportPlan(
    bulk,
    phx.equations.NetworkTransportPlan(network, np.asarray((0.01,)), np.asarray((0.0,))),
    transfer,
    np.asarray((0.5, 0.5)),
    np.asarray((1,)),
    np.asarray((0.003,)),
    np.asarray((0.2,)),
).prepare()
state = phx.equations.MixedDimensionalTransportState(
    np.asarray((3.0,)), np.asarray((1.0, 1.5)), np.asarray((0.5,))
)
initial_mass = float(runtime.ledger(state).total_mass)
for _ in range(5):
    step = runtime.step_backward_euler(state, 0.05)
    if not bool(step.accepted):
        raise RuntimeError("Mixed-dimensional transport step was rejected.")
    state = step.state
final = runtime.ledger(state)
mass_error = abs(float(final.total_mass) - initial_mass)
if mass_error > 1.0e-9 or not bool(final.successful):
    raise RuntimeError("Closed mixed-dimensional transport did not conserve mass.")
print(
    json.dumps(
        {
            "bulk": np.asarray(state.bulk).tolist(),
            "network": np.asarray(state.network).tolist(),
            "reservoirs": np.asarray(state.reservoirs).tolist(),
            "mass_error": mass_error,
            "exchange_defect": float(final.exchange_defect),
        },
        indent=2,
    )
)
