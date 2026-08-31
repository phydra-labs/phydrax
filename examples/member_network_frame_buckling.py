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
    constrained_dofs=jnp.asarray(((True, True), (False, False))),
)
positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
material = mn.LinearElasticMaterial(1000.0, 400.0, 1.0)
section = mn.BeamSection(1.0, 1.0, 1.0, 0.5, 100.0, 100.0)
properties = mn.MemberPropertyMap((material,), (section,), (0,), (0,))
reference = mn.MemberReferenceState(structure, positions)
dofs = mn.MemberDOFLayout(
    structure, rotation_constrained=jnp.asarray(((True,), (False,)))
)
definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock((0,)),))
problem = mn.MemberNetworkProblem(definition, assembly)
initial = mn.MemberKinematics(positions, jnp.zeros((2, 1)))
inputs = mn.MemberNetworkInputs(
    structure.prescribed_values(positions),
    dofs.prescribed_rotations(initial.rotation_vectors),
    jnp.asarray(((0.0, 0.0), (0.0, -1.0))),
    jnp.zeros((2, 1)),
    jnp.asarray((1.0,)),
)
result = mn.member_network_equilibrium(problem, inputs, initial)
stability = mn.tangent_stability(problem, inputs, result.state.kinematics)
local = mn.local_euler_buckling(
    definition,
    result.state.assembly.axial_force,
    jnp.asarray((2.0,)),
    result.state.assembly.switching_margin * 0.0 + 1.0,
)
print("status", int(result.status), result.message)
print("tip", result.state.kinematics.positions[1])
print("bending moment", result.state.assembly.bending_moment[0])
print("minimum tangent eigenvalue", stability.minimum_eigenvalue)
print("Euler critical load", local.critical_load[0])
