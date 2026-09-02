#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _system():
    mixture = phx.equations.ReactingMixture(
        ("H2", "O2", "H2O"),
        jnp.asarray((2.0e-3, 32.0e-3, 18.0e-3)),
        jnp.asarray((14300.0, 918.0, 1860.0)),
        jnp.asarray((0.0, 0.0, -1.34e7)),
    )
    reaction = phx.equations.ArrheniusReaction(
        jnp.asarray((-2.0, -1.0, 2.0)),
        jnp.asarray((1.0, 1.0, 0.0)),
        pre_exponential=2.0e5,
        activation_temperature=8000.0,
    )
    return phx.equations.ReactingEulerSystem(mixture, (reaction,), 1)


def test_reacting_mixture_roundtrip_flux_source_and_entropy_are_consistent():
    system = _system()
    primitive = jnp.asarray((1.0, 20.0, 1400.0, 0.2, 0.2, 0.6))
    state = system.primitive_to_conserved(primitive)
    np.testing.assert_allclose(
        system.conserved_to_primitive(state), primitive, rtol=2.0e-11, atol=2.0e-11
    )
    assert system.admissible(state)
    assert system.physical_flux(state, 0).shape == state.shape
    assert system.max_wave_speed(state, state, 0) > 0.0
    source = system.reaction_source(state)
    assert source[0] == 0.0
    assert source[2] == 0.0
    independent = source[3:]
    final = -jnp.sum(independent)
    np.testing.assert_allclose(jnp.sum(independent) + final, 0.0, atol=2.0e-12)
    entropy = phx.equations.reacting_mixture_entropy_pair(system)
    assert jnp.isfinite(entropy.entropy(state))
    assert jnp.all(jnp.isfinite(entropy.entropy_variables(state)))


def test_reacting_system_executes_in_nodal_conservation_compiler():
    system = _system()
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
        phx.equations.ConservationProblemIR(
            "reacting",
            "state",
            system,
            boundaries,
            source=system.reaction_source_term,
            source_id="arrhenius-source",
        ),
        discretization,
        phx.equations.fem.NodalDGConservationMethodPlan(
            phx.discretization.RusanovFluxPlan()
        ),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 1400.0, 0.2, 0.2, 0.6))),
        discretization.field_spaces[0].vector_space.shape,
    )
    rate = compiled(0.0, state)
    assert rate.shape == state.shape
    assert jnp.all(jnp.isfinite(rate))
