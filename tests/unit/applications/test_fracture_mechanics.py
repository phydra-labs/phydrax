#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _sharp_state():
    fracture = phx.applications.fracture
    vertices = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )
    cells = jnp.asarray(
        [
            [0, 1, 4],
            [1, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
        ],
        dtype=jnp.int32,
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        vertices,
        cells,
        cell_global_ids=jnp.asarray([10, 20, 30, 40, 50, 60]),
    )
    geometry = fracture.CrackFrontGeometry(
        jnp.asarray([[0.25, 0.4], [2.75, 0.4]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        tip_ids=jnp.asarray([31, 47]),
        crack_id="center-crack",
    )
    topology = fracture.build_sharp_crack_topology(mesh, geometry)
    quadrature = fracture.build_sharp_crack_quadrature(mesh, topology, order=2)
    return mesh, fracture.SharpFractureState(geometry, topology, quadrature)


def test_finite_crack_geometry_does_not_classify_its_infinite_extension():
    fracture = phx.applications.fracture
    vertices = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [3.0, 0.0], [2.0, 1.0]]
    )
    cells = jnp.asarray([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(
        vertices,
        cells,
        cell_global_ids=jnp.asarray([10, 20]),
    )
    geometry = fracture.CrackFrontGeometry(
        jnp.asarray([[0.1, 0.2], [0.4, 0.2]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        tip_ids=jnp.asarray([7, 9]),
    )

    topology = fracture.build_sharp_crack_topology(mesh, geometry)
    projection = geometry.project(jnp.asarray([[0.25, 0.4], [2.5, 0.2]]))

    assert jnp.array_equal(topology.cut_cell_ids, jnp.asarray([10]))
    assert jnp.allclose(projection.points[0], jnp.asarray([0.25, 0.2]))
    assert jnp.allclose(projection.points[1], jnp.asarray([0.4, 0.2]))
    assert jnp.isclose(geometry.length, 0.3)


def test_sharp_plus_minus_tip_and_face_quadrature_conserve_measure():
    _, state = _sharp_state()
    quadrature = state.quadrature
    integrated_area = (
        jnp.sum(quadrature.plus.weights)
        + jnp.sum(quadrature.minus.weights)
        + jnp.sum(quadrature.tips.weights)
    )
    plus_face = jnp.sum(
        jnp.where(quadrature.faces.side == 1, quadrature.faces.weights, 0.0)
    )
    minus_face = jnp.sum(
        jnp.where(quadrature.faces.side == -1, quadrature.faces.weights, 0.0)
    )

    assert quadrature.plus.weights.size > 0
    assert quadrature.minus.weights.size > 0
    assert quadrature.tips.weights.size > 0
    assert jnp.all(quadrature.tips.radii > 0.0)
    assert jnp.allclose(integrated_area, quadrature.evidence.cut_cell_area)
    assert jnp.allclose(plus_face, state.geometry.length)
    assert jnp.allclose(minus_face, state.geometry.length)
    assert quadrature.evidence.relative_area_defect < 1.0e-12


def test_shifted_heaviside_and_williams_enrichment_vanish_at_owning_nodes():
    fracture = phx.applications.fracture
    geometry = fracture.CrackFrontGeometry(
        jnp.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        jnp.asarray([[0, 1]], dtype=jnp.int32),
        tip_ids=jnp.asarray([3, 5]),
    )
    heaviside_nodes = jnp.asarray([[0.0, 0.2], [0.0, -0.2]])
    branch_nodes = jnp.asarray([[0.8, 0.1], [0.8, -0.1]])
    material = fracture.CrackTipMaterial(2.0, 0.25, kinematics="plane_strain")
    basis = fracture.IsotropicWilliamsCrackTipBasis(material)
    enrichment = fracture.ShiftedCrackEnrichment(
        geometry,
        heaviside_nodes,
        branch_nodes,
        basis,
        tip_id=5,
    )

    heaviside = enrichment.evaluate(heaviside_nodes).heaviside
    williams = enrichment.evaluate(branch_nodes).williams
    local = geometry.tip_local_coordinates(jnp.asarray([[1.2, 0.1]]), 5)
    radial, angular = basis.local_derivatives(local)

    assert jnp.allclose(jnp.diag(heaviside), 0.0)
    assert jnp.allclose(williams[jnp.arange(2), jnp.arange(2)], 0.0)
    assert jnp.all(jnp.isfinite(radial))
    assert jnp.all(jnp.isfinite(angular))
    assert jnp.isclose(material.kappa, 2.0)


def test_interaction_integral_reports_path_and_energy_consistent_sif_evidence():
    fracture = phx.applications.fracture
    contour_count = 3
    stress = jnp.broadcast_to(
        jnp.asarray([[[[1.0, 0.0], [0.0, 0.0]]]]),
        (contour_count, 1, 2, 2),
    )
    gradient = jnp.broadcast_to(
        jnp.asarray([[[[1.0, 0.0], [0.0, 0.0]]]]),
        stress.shape,
    )
    zero = jnp.zeros_like(stress)
    q_gradient = jnp.broadcast_to(
        jnp.asarray([[[-1.0, 0.0]]]),
        (contour_count, 1, 2),
    )
    weights = jnp.ones((contour_count, 1))
    material = fracture.CrackTipMaterial(2.0, 0.0, kinematics="plane_stress")

    evidence = fracture.evaluate_interaction_integral(
        stress,
        gradient,
        stress,
        gradient,
        zero,
        zero,
        q_gradient,
        weights,
        jnp.asarray([0.1, 0.2, 0.3]),
        jnp.asarray([1.0, 0.0]),
        material,
        topology_id="topology",
        quadrature_id="quadrature",
        state_version=4,
        qualification_tolerance=1.0e-12,
    )

    assert jnp.allclose(evidence.mode_i_by_contour, 1.0)
    assert jnp.allclose(evidence.mode_ii_by_contour, 0.0)
    assert jnp.allclose(evidence.j_by_contour, 0.5)
    assert bool(evidence.qualified)
    assert evidence.path_independence_defect == 0.0
    assert evidence.energy_consistency_defect == 0.0


def test_rejected_growth_transaction_rolls_back_every_sharp_object():
    fracture = phx.applications.fracture
    mesh, state = _sharp_state()
    proposal = fracture.CrackGrowthProposal(
        47,
        jnp.asarray([1.0, 0.0]),
        0.1,
        0.25,
        1.0,
        0.0,
        criterion="maximum-hoop-stress",
        admissible=False,
        rejection_reasons=("driving-force-below-toughness",),
        base_geometry_id=state.geometry.geometry_id,
        base_topology_id=state.topology.topology_id,
        base_state_version=state.state_version,
    )
    transaction = fracture.prepare_crack_growth_transaction(
        mesh,
        state,
        proposal,
        accepted=True,
    )

    assert transaction.commit(state) is state
    assert transaction.rollback(state) is state
    assert transaction.candidate is state
    assert not transaction.accepted


def test_accepted_growth_transaction_promotes_geometry_topology_and_quadrature():
    fracture = phx.applications.fracture
    mesh, state = _sharp_state()
    proposal = fracture.CrackGrowthProposal(
        47,
        jnp.asarray([1.0, 0.0]),
        0.1,
        1.0,
        0.5,
        0.0,
        criterion="maximum-hoop-stress",
        admissible=True,
        base_geometry_id=state.geometry.geometry_id,
        base_topology_id=state.topology.topology_id,
        base_state_version=state.state_version,
    )
    transaction = fracture.prepare_crack_growth_transaction(
        mesh,
        state,
        proposal,
        accepted=True,
    )
    promoted = transaction.commit(state)

    assert transaction.accepted
    assert promoted.state_version == state.state_version + 1
    assert promoted.topology.topology_version == state.topology.topology_version + 1
    assert jnp.isclose(promoted.geometry.length, state.geometry.length + 0.1)
    assert promoted.quadrature.topology_id == promoted.topology.topology_id


def test_diffuse_fixed_history_neural_block_is_bounded_and_irreversible():
    fracture = phx.applications.fracture
    accepted = fracture.PhaseFieldHistoryState(
        jnp.asarray([[0.2], [0.4]]),
        jnp.asarray([0.1, 0.6]),
    )
    controller = fracture.BoundedNeuralFixedHistoryController(
        lambda feature: feature[0],
        controller_id="bounded-damage",
    )
    block = controller.evaluate(jnp.asarray([[0.0], [1.0]]), accepted)
    transaction = block.transaction(
        jnp.asarray([[0.1], [0.8]]),
        accepted=True,
    )
    promoted = transaction.commit(accepted)
    parameters = fracture.PhaseFieldFractureParameters(
        1.0,
        2.0,
        3.0,
        0.1,
        residual_stiffness=1.0e-6,
    )
    model = fracture.PhaseFieldFractureModel(parameters)
    form = model.form(accepted.history)

    assert jnp.all(block.damage >= accepted.accepted_damage)
    assert jnp.all(block.damage <= 1.0)
    assert jnp.all(promoted.history >= accepted.history)
    assert jnp.all(promoted.accepted_damage >= accepted.accepted_damage)
    assert promoted.state_version == accepted.state_version + 1
    assert jnp.isclose(parameters.degradation(0.0), 1.0)
    assert jnp.isclose(parameters.degradation(1.0), parameters.residual_stiffness)
    assert form.field_names == ("displacement", "damage")


def test_crack_face_contact_mapping_preserves_gap_and_action_reaction():
    fracture = phx.applications.fracture
    contact = phx.applications.contact
    _, state = _sharp_state()
    adapter = fracture.CrackFaceContactAdapter(
        state.topology,
        contact.PenaltyContactLaw(100.0),
    )
    accepted = adapter.accepted_state()
    zero = jnp.zeros_like(state.geometry.vertices)
    closing = jnp.broadcast_to(jnp.asarray([0.0, -0.1]), zero.shape)

    evaluation = adapter.evaluate(accepted, closing, zero)

    assert adapter.topology_id == state.topology.topology_id
    assert evaluation.query.configuration.epoch == state.topology.topology_version
    assert jnp.all(evaluation.gap < 0.0)
    assert jnp.all(evaluation.active)
    assert jnp.allclose(evaluation.action_reaction_defect, 0.0)
    assert jnp.array_equal(adapter.segment_ids, state.geometry.segment_ids)
