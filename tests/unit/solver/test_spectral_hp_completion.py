#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _epoch():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )
    topology, geometry = phx.discretization.fem.initial_finite_element_hp_topology(
        mesh, 2, 8
    )
    return phx.discretization.fem.prepare_finite_element_hp_epoch(topology, geometry, "u")


def test_newton_condensation_schwarz_and_trace_coarse_plans():
    newton = phx.solver.HPNewtonKrylovBuilder(12, 1.0e-12)
    result = newton.solve(lambda value: value**2 - 2.0, jnp.asarray((1.0,)))
    assert bool(result.converged)
    np.testing.assert_allclose(np.asarray(result.value), np.sqrt(2.0), atol=1.0e-11)

    condensation = phx.solver.NonlinearLocalCondensation(
        2,
        jnp.asarray((0,), dtype=jnp.int32),
    )
    interior = condensation.eliminate(
        lambda value: jnp.asarray((value[0] + value[1], value[1] ** 2 - value[0])),
        jnp.asarray((4.0,)),
        jnp.asarray((1.0,)),
    )
    np.testing.assert_allclose(np.asarray(interior.value), 2.0, atol=1.0e-11)

    restrictions = (
        jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        jnp.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    )
    schwarz = phx.solver.HPRestrictedSchwarz(
        restrictions,
        (jnp.eye(2), jnp.eye(2)),
    )
    correction = schwarz.apply(jnp.asarray((1.0, 2.0, 3.0)))
    assert correction.shape == (3,)
    trace = phx.solver.BDDCFETIDPTracePlan(jnp.asarray(((1.0, 0.0, 1.0),)), jnp.eye(3))
    assert trace.coarse_matrix.shape == (1, 1)


def test_relaxed_marking_uq_cache_fusion_and_memory_plans(tmp_path):
    marking = phx.solver.RelaxedHPMarking(2, 0.2)
    indicators = jnp.asarray((1.0, 3.0, 2.0, 0.5))
    valid = jnp.asarray((True, True, True, False))
    weights = marking.weights(indicators, valid)
    assert jnp.sum(weights) <= 2.0 + 1.0e-12
    selected = marking.safe_project(
        indicators,
        valid,
        jnp.asarray(((0, 1), (0, 2), (0, 3), (0, 4))),
    )
    np.testing.assert_array_equal(np.asarray(selected), (False, True, True, False))

    aggregator = phx.solver.MeshVaryingUQAggregator(3)
    mean, variance = aggregator.aggregate(
        (jnp.asarray((1.0, 2.0)), jnp.asarray((2.0, 4.0, 6.0))),
        (jnp.asarray(((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))), jnp.eye(3)),
        jnp.asarray((0.25, 0.75)),
    )
    assert mean.shape == (3,)
    assert jnp.all(variance >= 0.0)

    cache = phx.discretization.fem.PersistentSemanticCache(tmp_path / "cache")
    path = cache.store("operator", {"matrix": jnp.eye(2)}, {"kind": "dense"})
    arrays, metadata = cache.load("operator")
    assert path.exists()
    np.testing.assert_allclose(np.asarray(arrays["matrix"]), np.eye(2))
    assert metadata["kind"] == "dense"

    transfer = phx.discretization.fem.FusedTensorTransfer(
        (jnp.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))),)
    )
    value = jnp.asarray((2.0, 4.0))
    mapped = transfer.apply(value)
    dual = jnp.asarray((0.1, 0.2, 0.3))
    np.testing.assert_allclose(
        jnp.vdot(mapped, dual), jnp.vdot(value, transfer.pullback(dual))
    )
    precision = phx.discretization.fem.HPMixedPrecisionPolicy(
        "float32", "float64", "float64"
    )
    assert precision.compute_dtype == "float64"
    memory = phx.discretization.fem.HPWorksetMemoryPlan((16, 12), 4, "float64", 4096)
    assert 0 < memory.planned_bytes <= 4096


def test_adaptive_high_order_io_and_mesh_import(tmp_path):
    epoch = _epoch()
    vtk = tmp_path / "epoch.vtk"
    xdmf = tmp_path / "epoch.xdmf"
    forest = tmp_path / "forest.json"
    phx.discretization.fem.write_adaptive_vtk(vtk, epoch)
    phx.discretization.fem.write_adaptive_xdmf(xdmf, epoch)
    phx.discretization.fem.write_hp_forest(forest, epoch)
    assert "UNSTRUCTURED_GRID" in vtk.read_text()
    assert "Quadrilateral" in xdmf.read_text()
    assert epoch.epoch_id in forest.read_text()

    gmsh = tmp_path / "mesh.msh"
    gmsh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n4\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 3 0 1 2 3 4\n$EndElements\n"
    )
    imported = phx.discretization.fem.read_finite_element_mesh(gmsh)
    mesh = imported.mesh
    assert mesh.blocks[0].cell_kind == "quadrilateral"
    exodus = phx.discretization.fem.read_exodus_high_order_arrays(
        mesh.coordinates, mesh.blocks[0].vertices, "quadrilateral"
    )
    np.testing.assert_allclose(
        np.asarray(exodus.coordinates),
        np.asarray(mesh.coordinates),
    )
    np.testing.assert_array_equal(
        np.asarray(exodus.blocks[0].vertices),
        np.asarray(mesh.blocks[0].vertices),
    )
