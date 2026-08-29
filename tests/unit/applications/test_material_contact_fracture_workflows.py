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


def test_contact_pair_ids_persist_through_accepted_commit():
    contact = phx.applications.contact
    pairs = contact.ContactPairState(
        jnp.asarray([11]),
        jnp.asarray([1]),
        jnp.asarray([2]),
        jnp.asarray([0.1]),
        jnp.asarray([[1.0, 0.0]]),
    )
    workflow = contact.ContactWorkflow(contact.FrictionlessContactLaw(100.0), pairs)
    evaluation = workflow.evaluate(jnp.asarray([[-0.2, 0.0]]), jnp.zeros((1, 2)))
    accepted = phx.solver.FiniteElementAcceptedState(
        (jnp.zeros((1, 2)),), 0.0, 0, "topology", "prepared", "compiled"
    )

    def solve_attempt(state, start, end, time_law, args):
        return workflow.attempt((jnp.asarray([[-0.2, 0.0]]),), evaluation, True)

    promoted, diagnostics = phx.solver.FiniteElementAcceptedStepSchedule(
        solve_attempt
    ).advance(accepted, 1.0, phx.solver.TimeLaw.constant(1.0))

    assert bool(evaluation.active[0])
    assert int(evaluation.pair_ids[0]) == 11
    assert bool(diagnostics.accepted)
    assert promoted.materials.states[0].state_version == 1


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
