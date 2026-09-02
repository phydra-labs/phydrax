#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_ideal_mhd_entropy_and_oblique_reflection_are_finite_and_involutive():
    system = phx.equations.IdealMHDSystem(2)
    primitive = jnp.asarray((1.0, 0.2, -0.1, 0.05, 1.0, 0.3, -0.2, 0.1))
    state = system.primitive_to_conserved(primitive)
    entropy = phx.equations.ideal_mhd_entropy_pair(system)
    assert jnp.isfinite(entropy.entropy(state))
    assert jnp.all(jnp.isfinite(entropy.entropy_variables(state)))
    normal = jnp.asarray((0.6, 0.8, 0.0))
    reflected = system.reflect_normal_state(state, normal)
    np.testing.assert_allclose(
        system.reflect_normal_state(reflected, normal), state, atol=3.0e-12
    )
    momentum = state[1:4]
    np.testing.assert_allclose(
        jnp.dot(reflected[1:4], normal), -jnp.dot(momentum, normal), atol=3.0e-12
    )


def test_shallow_water_total_energy_is_convex_entropy_pair():
    system = phx.equations.ShallowWaterSystem(2)
    state = jnp.asarray((2.0, 0.4, -0.2))
    pair = phx.equations.shallow_water_energy_pair(system)
    assert pair.admissible(state)
    assert pair.entropy(state) > 0.0
    assert jnp.all(jnp.isfinite(pair.entropy_variables(state)))
    assert jnp.isfinite(pair.entropy_flux(state, 0))


def test_mhd_executes_through_nodal_conservation_compiler():
    system = phx.equations.IdealMHDSystem(1)
    mesh = phx.discretization.CellMesh(
        np.asarray(((0.0,), (1.0,))),
        (
            phx.discretization.CellBlock(
                "cells", "interval", np.asarray(((0, 1),), dtype=np.int32)
            ),
        ),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "state",
            phx.discretization.discontinuous_element("interval", 0),
            component_shape=(system.component_count,),
        ),
    ).prepare()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = phx.discretization.fem.FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, phx.discretization.ExtrapolationBoundary())},
    )
    compiled = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR("mhd", "state", system, boundaries),
        discretization,
        phx.equations.fem.NodalDGConservationMethodPlan(
            phx.discretization.RusanovFluxPlan()
        ),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(
            jnp.asarray((1.0, 0.1, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0))
        ),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-10)
