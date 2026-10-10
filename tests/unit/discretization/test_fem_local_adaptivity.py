#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _mesh() -> Any:
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(
        vertices,
        cells,
        vertex_global_ids=jnp.asarray([1, 2, 3, 4, 5]),
        cell_global_ids=jnp.asarray([10, 20, 30, 40]),
    )


def _cell_table(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    vertices = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    rows = np.asarray(mesh.vertex_global_ids)[vertices]
    order = np.argsort(identifiers, kind="stable")
    return identifiers[order], rows[order]


def _bisect(
    source: Any, refine: Any = (), coarsen: Any = (), hierarchy: Any = None
) -> Any:
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(refine, dtype=np.int64),
                np.asarray(coarsen, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )


def test_fem_local_adaptivity_scenario_1() -> None:
    mesh = _mesh()
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    marked = phx.discretization.dorfler_mark(
        jnp.asarray([4.0, 1.0, 1.0, 1.0]),
        0.5,
        cell_global_ids=mesh.blocks[0].global_ids,
    )
    refined = _bisect(source, marked)
    transfer = refined.transfer
    constant = transfer.apply(jnp.ones((5,)))
    linear = transfer.apply(mesh.coordinates)
    children = np.setdiff1d(
        np.concatenate(
            [np.asarray(block.global_ids) for block in refined.target.mesh.blocks]
        ),
        np.asarray(mesh.blocks[0].global_ids),
    )
    restored = _bisect(refined.target, coarsen=children, hierarchy=refined.hierarchy)

    assert jnp.array_equal(marked, jnp.asarray([10]))
    assert refined.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert sum(block.cell_count for block in refined.target.mesh.blocks) == 5
    assert jnp.allclose(constant, 1.0)
    assert jnp.allclose(linear, refined.target.mesh.coordinates)
    assert transfer.preserves_constants and transfer.preserves_linear
    assert transfer.conservative and transfer.positivity_preserving
    assert transfer.source_topology_id == mesh.topology_id
    assert transfer.target_topology_id == refined.target.mesh.topology_id
    assert restored.target.mesh.topology_id == mesh.topology_id
    restored_ids, restored_cells = _cell_table(restored.target.mesh)
    source_ids, source_cells = _cell_table(mesh)
    np.testing.assert_array_equal(restored_ids, source_ids)
    np.testing.assert_array_equal(restored_cells, source_cells)
    np.testing.assert_array_equal(restored.target.mesh.coordinates, mesh.coordinates)
    mesh = _mesh()
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    refined = _bisect(source, (10,))
    transfer = refined.transfer
    count = refined.target.mesh.coordinates.shape[0]
    primal = jnp.stack((jnp.arange(5.0), jnp.arange(5.0) ** 2), axis=1)
    target_dual = jnp.stack(
        (jnp.arange(float(count)), jnp.cos(jnp.arange(float(count)))),
        axis=1,
    )
    left = jnp.vdot(transfer.apply(primal), target_dual)
    right = jnp.vdot(primal, transfer.pullback(target_dual))
    dwr = phx.discretization.local_dual_weighted_residual(
        jnp.asarray([[1.0, -2.0], [3.0, 4.0]]),
        jnp.asarray([[0.5, 1.0], [2.0, -1.0]]),
    )

    assert jnp.allclose(left, right)
    assert jnp.allclose(dwr.signed, jnp.asarray([-1.5, 2.0]))
    assert jnp.allclose(dwr.absolute, jnp.asarray([1.5, 2.0]))
    mesh = _mesh()
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    accepted = phx.solver.FiniteElementAcceptedState(
        (mesh.coordinates[:, 0],),
        0.0,
        0,
        mesh.topology_id,
        "prepared",
        "compiled",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate_mesh, fields, materials, lineage, args: False,
        fields=phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    )
    result = transaction.execute(accepted, source.mesh, _bisect(source, (10,)))

    assert not bool(result.committed)
    assert result.adaptation is None
    assert result.state.accepted_id == accepted.accepted_id
    assert result.mesh.topology_id == mesh.topology_id
    assert jnp.array_equal(result.state.fields[0], accepted.fields[0])


def test_topology_transaction_moves_each_field_by_its_declared_family() -> None:
    mesh = _mesh()
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    fields = (
        phx.discretization.FiniteElementFieldSpec(
            "pressure", phx.discretization.lagrange_element("triangle", 2)
        ),
        phx.discretization.FiniteElementFieldSpec(
            "flux",
            phx.discretization.form_element(
                "triangle", 1, 1, twist="twisted", proxy="flux"
            ),
        ),
    )

    def pressure(points: Any, args: object) -> Any:
        del args
        return 1.0 + points[..., 0] * points[..., 1] - 0.5 * points[..., 1] ** 2

    def flux(points: Any, args: object) -> Any:
        del args
        return jnp.stack((1.0 + points[..., 0], -0.5 + points[..., 1]), axis=-1)

    functions = (pressure, flux)
    discretization = phx.discretization.FiniteElementPlan(mesh, fields).prepare()
    accepted = phx.solver.FiniteElementAcceptedState(
        tuple(
            discretization.project(field.name, function)
            for field, function in zip(fields, functions, strict=True)
        ),
        0.0,
        0,
        mesh.topology_id,
        "prepared",
        "compiled",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate_mesh, values, materials, lineage, args: True, fields=fields
    )

    result = transaction.execute(accepted, mesh, _bisect(source, (10, 30)))
    target = phx.discretization.FiniteElementPlan(result.mesh, fields).prepare()

    assert bool(result.committed) and result.receipt is not None
    assert result.receipt.remapped == ("field/flux", "field/pressure")
    assert [transfer.semantics for transfer in result.transfers] == [
        "nested-interpolation",
        "contravariant-piola",
    ]
    for field, function, value in zip(
        fields, functions, result.state.fields, strict=True
    ):
        np.testing.assert_allclose(
            value, target.project(field.name, function), atol=1e-13
        )
    assert result.state.transition_id == result.receipt.receipt_id


def test_vertex_interpolation_transfer_certifies_its_invariant_claims() -> None:
    source = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 2.0)))
    rows = np.asarray(((0, 0), (1, 0), (0, 1), (1, 2)), dtype=np.int32)
    weights = np.asarray(((1.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.5, 0.5)))
    valid = np.asarray(((True, False), (True, False), (True, True), (True, True)))
    target = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.5, 1.0)))
    transfer = phx.discretization.fem.vertex_interpolation_transfer(
        rows,
        weights,
        valid,
        source_size=3,
        source_topology_id="source",
        target_topology_id="target",
        preserves_linear=True,
        source_coordinates=source,
        target_coordinates=target,
    )
    values = jnp.asarray((2.0, -1.0, 4.0))
    dual = jnp.asarray((0.3, -0.7, 1.1, 2.0))

    assert transfer.preserves_constants and transfer.preserves_linear
    assert not transfer.conservative
    np.testing.assert_allclose(transfer.apply(values), (2.0, -1.0, 0.5, 1.5))
    np.testing.assert_allclose(
        jnp.vdot(transfer.apply(values), dual),
        jnp.vdot(values, transfer.pullback(dual)),
    )
    with pytest.raises(ValueError, match="linear preservation"):
        phx.discretization.fem.vertex_interpolation_transfer(
            rows,
            weights,
            valid,
            source_size=3,
            source_topology_id="source",
            target_topology_id="target",
            preserves_linear=True,
            source_coordinates=source,
            target_coordinates=target + 0.25,
        )
    with pytest.raises(ValueError, match="conservation"):
        phx.discretization.fem.vertex_interpolation_transfer(
            rows,
            weights,
            valid,
            source_size=3,
            source_topology_id="source",
            target_topology_id="target",
            conservative=True,
            source_measures=np.ones((3,)),
            target_measures=np.ones((4,)),
        )
    orphan = valid.copy()
    orphan[0] = False
    with pytest.raises(ValueError, match="valid source route"):
        phx.discretization.fem.vertex_interpolation_transfer(
            rows,
            weights,
            orphan,
            source_size=3,
            source_topology_id="source",
            target_topology_id="target",
        )


@pytest.mark.parametrize("measure_scale", (1.0, 1e-9), ids=("unit", "si-microliter"))
def test_certified_conservative_transfer_forms_an_accepted_epoch_transition(
    measure_scale: float,
) -> None:
    # A near-identity transfer whose measures change by a relative 2e-12: the
    # accuracy an L2 projection with action condition 1e3 certifies. Independent
    # ledger: content changes by 4 * m * 2e-12 * 300 = 2.4e-9 * m exactly.
    count, defect, measure = 4, 2e-12, 0.25 * measure_scale
    relation = phx.sparse.RowRelation(
        np.tile(np.arange(count, dtype=np.int32), (count, 1)),
        source_size=count,
        valid=np.ones((count, count), dtype=np.bool_),
    )
    primal = phx.sparse.SparseLinearMap(
        relation, jnp.asarray(np.eye(count) * (1.0 + defect)), operator_id="near-identity"
    )
    measures = np.full((count,), measure)
    transfer = phx.discretization.FiniteElementTopologyTransfer(
        primal,
        "coarse",
        "graded",
        conservative=True,
        action_condition=1e3,
        source_measures=measures,
        target_measures=measures,
    )
    epoch = phx.discretization.TopologyEpoch
    source_epoch, target_epoch = (
        epoch(0, "g", "coarse", "p"),
        epoch(1, "g", "graded", "p"),
    )
    geometry = phx.discretization.TransferGeometryBinding(
        "g",
        "g",
        "topology-correspondence",
        source_topology_id="coarse",
        target_topology_id="graded",
        coverage_defect=None,
    )

    def space(item: phx.discretization.TopologyEpoch) -> Any:
        return phx.discretization.DiscreteFieldSpace(
            "u",
            item.epoch_id,
            phx.discretization.EntityDofLayout(f"dofs-{item.index}", count, count),
            phx.linalg.ArraySpace((count,)),
            representation="basis_coefficient",
        )

    transition = transfer.epoch_transition(
        space(source_epoch),
        space(target_epoch),
        source_epoch,
        target_epoch,
        measures,
        measures,
        geometry=geometry,
    )
    values = jnp.full((count,), 300.0, dtype=jnp.float64)
    result = transition.apply(values)
    content = count * measure * 300.0

    np.testing.assert_allclose(result.conservation_residual, content * defect, rtol=1e-3)
    assert bool(result.successful)
    # Admitted content drift stays a tiny relative fraction at every measure scale.
    assert float(result.content_tolerance) <= 1e-9 * content
    with pytest.raises(ValueError, match="not conserved by this transfer"):
        transfer.epoch_transition(
            space(source_epoch),
            space(target_epoch),
            source_epoch,
            target_epoch,
            measures,
            measures * 1.001,
            geometry=geometry,
        )


def test_sparse_cell_operators_use_prepared_routes_under_jit() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    prepared = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("triangle", 1),
            component_shape=(2,),
        ),
    ).prepare()
    values = jnp.asarray(((0.2, -0.3), (0.7, 1.1), (-0.4, 0.8)), dtype=jnp.float64)

    @eqx.filter_jit
    def actions(
        discretization: phx.discretization.FiniteElementDiscretization,
        state: Array,
    ) -> tuple[Array, Array]:
        mass, stiffness = discretization.assemble_field_operators(
            "u", discretization.default_runtime
        )
        return mass.mv(state.reshape((-1,))).reshape(state.shape), stiffness.mv(
            state.reshape((-1,))
        ).reshape(state.shape)

    mass_image, stiffness_image = actions(prepared, values)
    local_mass = np.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))) / 24
    gradients = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    np.testing.assert_allclose(mass_image, local_mass @ np.asarray(values), atol=1e-14)
    np.testing.assert_allclose(
        stiffness_image, 0.5 * gradients @ gradients.T @ np.asarray(values), atol=1e-14
    )
