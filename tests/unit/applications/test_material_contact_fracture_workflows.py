#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_cpfem_identity_update_is_objective_and_admissible():
    cpfem = phx.applications.crystal_plasticity
    model = cpfem.CrystalPlasticityModel(
        (
            cpfem.CrystalSlipSystem(
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([0.0, 1.0, 0.0]),
            ),
        ),
        cpfem.CrystalPlasticityParameters(10.0, 20.0, 0.01, 0.1, 1.0, 2.0),
    )
    result = model.update(jnp.eye(3), model.initial_state(), 0.1)

    assert bool(result.converged)
    assert jnp.linalg.norm(result.first_piola) < 1.0e-12
    assert jnp.allclose(jnp.linalg.det(result.state.plastic_deformation), 1.0)


def test_fracture_history_is_irreversible_and_xfem_classification_is_stable():
    fracture = phx.applications.fracture
    history = fracture.FractureHistoryState(jnp.zeros((2, 1)), jnp.asarray([0.1, 0.2]))
    promoted = history.promote(jnp.asarray([[1.0], [0.5]]), jnp.asarray([0.2, 0.2]))
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(
        vertices, cells, cell_global_ids=jnp.asarray([10, 20])
    )
    enrichment = fracture.classify_crack_cells(
        mesh, fracture.CrackGeometry([0.0, 0.5], [1.0, 0.5])
    )

    assert jnp.all(promoted.history >= history.history)
    assert jnp.all(promoted.accepted_damage >= history.accepted_damage)
    assert jnp.array_equal(enrichment.active_cell_ids, jnp.asarray([10, 20]))
