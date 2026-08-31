#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network
structure = sm.ForceDensityStructure.from_edges(
    jnp.asarray(((0, 1),), dtype=jnp.int32),
    2,
    2,
    constrained_dofs=jnp.asarray(((True, True), (False, True))),
    node_ids=("support", "tip"),
    member_ids=("tie",),
)
positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
material = mn.LinearElasticMaterial(
    100.0,
    40.0,
    2.0,
    tension_allowable=20.0,
    compression_allowable=20.0,
)
section = mn.BeamSection(1.0, 1.0, 1.0, 0.5, 1.0, 1.0)
properties = mn.MemberPropertyMap((material,), (section,), (0,), (0,))
reference = mn.MemberReferenceState(structure, positions)
dofs = mn.MemberDOFLayout(structure, rotation_constrained=jnp.ones((2, 1), dtype=bool))
definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
assembly = mn.MemberNetworkAssembly((mn.AxialMemberBlock((0,)),))
problem = mn.MemberNetworkProblem(definition, assembly)
initial = mn.MemberKinematics(positions, jnp.zeros((2, 1)))


def inputs(load):
    return mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        jnp.asarray(((0.0, 0.0), (load, 0.0))),
        jnp.zeros((2, 1)),
        jnp.asarray((1.0,)),
    )


install = mn.ConstructionStage(
    problem,
    inputs(0.0),
    (mn.InstallationRule.STRESS_FREE_AT_CURRENT_GEOMETRY,),
    stage_id="install",
)
load = mn.ConstructionStage(
    problem,
    inputs(5.0),
    (mn.InstallationRule.DECLARED_STRESS_FREE_LENGTH,),
    load_operation=mn.LoadOperation.ADD,
    stage_id="service-load",
)
sequence = mn.solve_construction_sequence(
    mn.plan_construction_sequence((install, load), initial), initial
)
final = sequence.stages[-1].equilibrium
buckling = mn.local_euler_buckling(
    definition,
    final.state.assembly.axial_force,
    jnp.ones((1,)),
    jnp.asarray((1.05,)),
)
sizing = mn.evaluate_member_sizing(definition, final, local_buckling=buckling)
verification = mn.verify_member_structure(
    equilibrium=final,
    construction=sequence,
    sizing=sizing,
    local_buckling=buckling,
    required=("equilibrium", "construction", "sizing", "local_buckling"),
)
print("sequence status", bool(sequence.successful))
print("tip", final.state.kinematics.positions[1])
print("mass", sizing.mass)
print("maximum utilization", sizing.maximum_utilization)
print("verification verdict", int(verification.verdict))
