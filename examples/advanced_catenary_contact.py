#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


mn = phx.applications.solid_mechanics.member_network
reference = mn.ElasticCatenaryReference(
    10.5,
    1.0e5,
    jnp.asarray((0.0, -2.0, 0.0)),
)
state = mn.solve_elastic_catenary(
    jnp.asarray((0.0, 0.0, 0.0)),
    jnp.asarray((10.0, 0.0, 0.0)),
    reference,
)
contact = mn.NodePlaneContact(
    (0,),
    jnp.asarray(((0.0, -1.0, 0.0),)),
    jnp.asarray(((0.0, 1.0, 0.0),)),
    friction_coefficient=jnp.asarray((0.3,)),
)
contact_state = mn.evaluate_node_plane_contact(
    contact,
    state.centerline[state.centerline.shape[0] // 2][None, :],
    jnp.asarray((1.0,)),
    jnp.asarray(((0.1, 0.0, 0.0),)),
)
print("regime", int(state.regime))
print("sag", state.sag)
print("minimum tension", state.minimum_tension)
print("contact gap", contact_state.gap)
print("contact active", contact_state.active)
