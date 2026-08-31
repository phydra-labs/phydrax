#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network
structure = sm.ForceDensityStructure.from_edges(
    jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
    3,
    2,
    fixed_nodes=(0, 2),
)
positions = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
form = sm.force_density_equilibrium(
    sm.ForceDensityProblem(structure, sign_mode="tension"),
    sm.ForceDensityInputs(jnp.ones((2,)), structure.prescribed_values(positions), loads),
)
material = mn.LinearElasticMaterial(100.0, 40.0, 1.0)
properties = mn.MemberPropertyMap((material,), (mn.AxialSection(1.0),), (0, 0), (0, 0))
assembly = mn.MemberNetworkAssembly(
    (mn.AxialMemberBlock((0, 1), mn.TensionOnlyCableLaw()),)
)
target, definition, inputs, initial = mn.member_network_from_force_density(
    form,
    structure,
    properties,
    assembly,
    axial_law=mn.TensionOnlyCableLaw(),
)
constitutive = mn.member_network_equilibrium(
    mn.MemberNetworkProblem(definition, assembly), inputs, initial
)
expected = definition.reference.rest_lengths
fabrication = mn.PrestressFabricationPolicy(
    0.9 * expected,
    1.1 * expected,
    -0.5 * jnp.ones_like(expected),
    0.5 * jnp.ones_like(expected),
    require_stability=False,
    require_sequence=False,
)
realizability = mn.assess_prestress_realizability(
    target,
    definition,
    mn.TensionOnlyCableLaw(),
    fabrication,
    member_roles="tension-only",
)
print("form-finding forces", form.state.axial_forces)
print("stress-free lengths", realizability.rest_lengths)
print("constitutive forces", constitutive.state.assembly.axial_force)
print("slack mask", ~constitutive.state.assembly.active)
print("realizability verdict", int(realizability.verdict))
