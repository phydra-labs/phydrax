#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def _plane_stress():
    sm = phx.applications.solid_mechanics
    law = sm.NeoHookeanLaw(sm.NeoHookeanParameters.from_shear_bulk(3.0, 11.0))
    plan = sm.BlockDiagonalPlaneStressReductionPlan()
    deformation = jnp.asarray(((1.12, 0.06), (0.02, 0.93)))
    response = plan.evaluate(deformation, law, reference_thickness=1.7)
    tangent = jax.jacfwd(
        lambda value: plan.evaluate(value, law, reference_thickness=1.7).first_piola
    )(deformation)
    tangent_error = jnp.max(jnp.abs(tangent - response.condensed_tangent))
    passed = (
        response.successful
        & (jnp.abs(response.residual) < 1.0e-8)
        & (tangent_error < 1.0e-6)
    )
    return {
        "passed": bool(passed),
        "residual": float(jnp.abs(response.residual)),
        "thickness_stretch": float(response.kinematics.thickness_stretch),
        "tangent_error": float(tangent_error),
    }


def _mixed_incompressibility():
    sm = phx.applications.solid_mechanics

    def isochoric(deformation_bar):
        return 1.5 * (jnp.sum(deformation_bar * deformation_bar) - 2.0)

    def constraint(deformation):
        return jnp.log(jnp.linalg.det(deformation))

    exact = sm.MixedHyperelasticLaw(isochoric, constraint, minimum_jacobian=1.0e-8)
    finite = sm.MixedHyperelasticLaw(
        isochoric,
        constraint,
        bulk_modulus=80.0,
        minimum_jacobian=1.0e-8,
    )
    deformation = jnp.asarray(((1.2, 0.1), (0.0, 0.9)))
    pressure = jnp.asarray(2.5)
    exact_response = exact.evaluate(deformation, pressure)
    finite_response = finite.evaluate(deformation, pressure)
    exact_defect = jnp.abs(exact_response.constraint_residual - constraint(deformation))
    finite_defect = jnp.abs(
        finite_response.constraint_residual - (constraint(deformation) - pressure / 80.0)
    )
    al = sm.MixedAugmentedLagrangianPlan(
        sm.MixedHyperelasticLaw(
            isochoric,
            lambda value: jnp.linalg.det(value) - 1.0,
            minimum_jacobian=1.0e-8,
        ),
        initial_penalty=10.0,
        penalty_growth=4.0,
        constraint_reduction=0.25,
    )
    accepted = al.advance(
        al.initialize(jnp.asarray(((1.2, 0.0), (0.0, 1.0)))),
        jnp.asarray(((1.04, 0.0), (0.0, 1.0))),
        inner_successful=True,
    )
    passed = (
        exact_response.evidence.valid
        & finite_response.evidence.valid
        & (exact_defect < 1.0e-10)
        & (finite_defect < 1.0e-10)
        & accepted.evidence.accepted
    )
    return {
        "passed": bool(passed),
        "exact_constraint_defect": float(exact_defect),
        "finite_bulk_constraint_defect": float(finite_defect),
        "augmented_pressure": float(accepted.accepted_state.pressure),
    }


def _follower_loads():
    sm = phx.applications.solid_mechanics
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    deformation = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
    measure = phx.integration.DeformedMeasurePlan(
        "volume", jnp.asarray((0.5, 0.5, 0.5))
    ).evaluate(deformation)
    current = reference + 0.1
    dead = sm.ReferenceDeadBodyForce(jnp.asarray((2.0, -3.0)))
    evaluation = dead.evaluate(reference, current, measure, sm.MechanicalLoadState())
    matrix = jnp.asarray(((0.0, 2.0), (-1.0, 0.5)))

    def follower(reference_coordinates, current_coordinates, measure_state, state, args):
        del reference_coordinates, measure_state, state, args
        return current_coordinates @ matrix.T

    follower_load = sm.GeneralFollowerLoad(
        follower,
        support="body",
        measure_frame="reference",
        load_id="qualification-follower",
    )
    follower_evaluation = follower_load.evaluate(
        reference, current, measure, sm.MechanicalLoadState()
    )
    passed = (
        evaluation.valid
        & follower_evaluation.valid
        & (evaluation.semantics.conservativity == "potential")
        & (follower_evaluation.semantics.conservativity == "virtual_work")
    )
    return {
        "passed": bool(passed),
        "dead_force_norm": float(jnp.linalg.norm(evaluation.total_force_density)),
        "follower_force_norm": float(
            jnp.linalg.norm(follower_evaluation.total_force_density)
        ),
        "follower_is_nonconservative": bool(
            follower_evaluation.semantics.conservativity == "virtual_work"
        ),
    }


def _continuation_bifurcation():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, parameter, args: {"x": state["x"] ** 2 + parameter - 1.0},
        problem_id="qualification-fold",
    )
    plan = phx.continuation.plan_continuation(
        problem,
        num_steps=4,
        method=phx.continuation.PseudoArclengthContinuation(initial_step=0.1),
        plan_id="qualification-fold-plan",
    )
    prepared = phx.continuation.prepare_continuation(
        problem,
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.0),
        plan,
    )
    result = phx.continuation.run_continuation(prepared)
    passed = (result.status == int(phx.continuation.ContinuationStatus.SUCCESS)) & (
        len(result.branch.points) >= 2
    )
    return {
        "passed": bool(passed),
        "status": int(result.status),
        "points": len(result.branch.points),
        "accepted_steps": int(result.diagnostics.accepted_steps),
        "checkpoint_decisions": len(result.checkpoint.accepted_decision_ids),
    }


def _contact():
    contact = phx.applications.contact
    plus = contact.ContactSurface(
        "plus",
        jnp.asarray((10, 11)),
        jnp.asarray(((0.25, -0.1), (0.75, 0.2))),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        jnp.asarray((100,)),
    )
    minus = contact.ContactSurface(
        "minus",
        jnp.asarray((20, 21)),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        jnp.asarray((200,)),
    )
    query = contact.ContactQueryPlan(
        contact.ContactConfiguration(plus, minus, epoch=0, search_radius=1.0)
    ).execute()
    operator = contact.FixedEpochContactOperator(query, contact.PenaltyContactLaw(100.0))
    evaluation = operator.evaluate(operator.accepted_state())
    passed = jnp.all(evaluation.finite) & jnp.all(
        evaluation.action_reaction_defect < 1.0e-12
    )
    return {
        "passed": bool(passed),
        "pairs": len(query.patches.pair_ids),
        "minimum_gap": float(jnp.min(query.patches.gaps)),
        "action_reaction_defect": float(
            jnp.max(jnp.abs(evaluation.action_reaction_defect))
        ),
    }


def _fracture():
    fracture = phx.applications.fracture
    vertices = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    cells = jnp.asarray(((0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(
        vertices, cells, cell_global_ids=jnp.asarray((10, 20, 30, 40))
    )
    geometry = fracture.CrackFrontGeometry(
        jnp.asarray(((0.25, 0.4), (1.75, 0.4))),
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        tip_ids=jnp.asarray((31, 47)),
        crack_id="qualification-crack",
    )
    topology = fracture.build_sharp_crack_topology(mesh, geometry)
    quadrature = fracture.build_sharp_crack_quadrature(mesh, topology, order=2)
    area = (
        jnp.sum(quadrature.plus.weights)
        + jnp.sum(quadrature.minus.weights)
        + jnp.sum(quadrature.tips.weights)
    )
    defect = jnp.abs(area - quadrature.evidence.cut_cell_area)
    passed = (defect < 1.0e-12) & jnp.all(quadrature.tips.radii > 0.0)
    return {
        "passed": bool(passed),
        "cut_cells": int(topology.cut_cell_ids.size),
        "area_defect": float(defect),
        "minimum_tip_radius": float(jnp.min(quadrature.tips.radii)),
    }


def _topology():
    sm = phx.applications.solid_mechanics
    evidence = sm.hill_mandel_evidence(
        jnp.asarray(((2.0,), (2.0,))),
        jnp.asarray(((0.5,), (0.5,))),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((2.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.0, 0.0)),
    )
    aggregation = sm.Aggregation("maximum")
    sensitivity = aggregation.sensitivities(jnp.asarray((3.0, 3.0)), jnp.ones(2))
    passed = evidence.accepted & jnp.allclose(sensitivity, 0.5)
    return {
        "passed": bool(passed),
        "hill_mandel_defect": float(evidence.power_defect),
        "maximum_tie_sensitivity": [float(value) for value in sensitivity],
    }


def _member_network():
    sm = phx.applications.solid_mechanics
    mn = sm.member_network
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        constrained_dofs=jnp.asarray(((True, True), (False, False))),
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    material = mn.LinearElasticMaterial(1_000.0, 400.0, 1.0)
    section = mn.BeamSection(0.1, 0.01, 0.01, 0.005, 0.08, 0.08)
    properties = mn.MemberPropertyMap((material,), (section,), (0,), (0,))
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure,
        rotation_constrained=jnp.asarray(((True,), (False,))),
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock((0,)),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((2, 1)))
    nodal_forces = jnp.asarray(((0.0, 0.0), (0.0, -0.01)))
    inputs = mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        nodal_forces,
        jnp.zeros((2, 1)),
        reference.rest_lengths,
    )
    equilibrium = mn.member_network_equilibrium(problem, inputs, initial)
    reduced = dofs.reduce(
        equilibrium.state.kinematics.positions,
        equilibrium.state.kinematics.rotation_vectors,
    )

    def energy(state):
        kinematics = dofs.expand(
            state,
            inputs.prescribed_positions,
            inputs.prescribed_rotations,
        )
        return assembly.evaluate(definition, kinematics).energy

    generalized_load = jnp.concatenate(
        (
            structure.reduce(nodal_forces),
            inputs.nodal_moments.reshape((-1,))[dofs.free_rotation_indices],
        )
    )
    force_error = jnp.max(jnp.abs(jax.grad(energy)(reduced) - generalized_load))
    tangent_error = jnp.max(
        jnp.abs(jax.jacfwd(jax.grad(energy))(reduced) - jax.hessian(energy)(reduced))
    )
    angle = jnp.asarray(0.4)
    rotation = mn.rotation_vector_matrix(angle.reshape((1,)))
    rigid = mn.MemberKinematics(
        positions @ rotation.T + jnp.asarray((0.3, -0.2)),
        jnp.full((2, 1), angle),
    )
    objectivity_error = jnp.abs(assembly.evaluate(definition, rigid).energy)
    modal = mn.tangent_stability(
        problem,
        equilibrium,
        mass=jnp.eye(dofs.reduced_size),
    )
    ledger = mn.member_energy_work_evidence(
        jnp.asarray((0.5, 0.25)),
        jnp.asarray((0.5, 0.75)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        accepted=jnp.ones((2,), dtype=bool),
        topology_epoch=jnp.zeros((2,), dtype=jnp.int32),
        contact_epoch=jnp.zeros((2,), dtype=jnp.int32),
        fracture_epoch=jnp.zeros((2,), dtype=jnp.int32),
        unilateral_epoch=jnp.zeros((2,), dtype=jnp.int32),
        mode_epoch=jnp.zeros((2,), dtype=jnp.int32),
    )
    passed = (
        equilibrium.successful
        & (force_error < 1.0e-8)
        & (tangent_error < 1.0e-10)
        & (objectivity_error < 1.0e-9)
        & modal.modal_valid
        & ledger.balanced
    )
    return {
        "passed": bool(passed),
        "force_energy_gradient_error": float(force_error),
        "tangent_energy_hessian_error": float(tangent_error),
        "objectivity_error": float(objectivity_error),
        "modal_residual": float(modal.eigen_residual),
        "minimum_mass_eigenvalue": float(modal.minimum_mass_eigenvalue),
        "mass_orthogonality_error": float(modal.mass_orthogonality_error),
        "energy_balance_defect": float(jnp.max(jnp.abs(ledger.algorithmic_defect))),
    }


def _operator_learning():
    reduction = phx.nn.operator.training.MechanicsCaseReduction("cvar", alpha=0.5)
    result = reduction.evaluate(
        jnp.asarray((1.0, 4.0, 4.0)),
        probability_weights=jnp.asarray((0.5, 0.25, 0.25)),
    )
    passed = jnp.isclose(result.value, 4.0) & jnp.isclose(result.tail_mass, 0.5)
    return {
        "passed": bool(passed),
        "cvar": float(result.value),
        "tail_mass": float(result.tail_mass),
        "effective_sample_size": float(result.effective_sample_size),
    }


def qualify():
    sections = {
        "plane_stress": _plane_stress(),
        "mixed_incompressibility": _mixed_incompressibility(),
        "follower_loads": _follower_loads(),
        "continuation_bifurcation": _continuation_bifurcation(),
        "contact": _contact(),
        "fracture": _fracture(),
        "topology": _topology(),
        "member_network": _member_network(),
        "amortized_operator": _operator_learning(),
    }
    return {
        "maturity": "experimental",
        "passed": all(section["passed"] for section in sections.values()),
        "sections": sections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/mechanics_closure_qualification.json"),
    )
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(arguments.output)
    print(payload, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
