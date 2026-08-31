#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_primary_compiler_accepts_prepared_hp_epoch_and_native_constraint():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )
    topology, geometry = phx.discretization.fem.initial_finite_element_hp_topology(
        mesh, 3, 8
    )
    epoch = phx.discretization.fem.prepare_finite_element_hp_epoch(
        topology,
        geometry,
        "u",
    )
    form = phx.equations.FiniteElementForm(
        "native-hp-mass",
        "u",
        (phx.equations.MassAction("u", 1.0),),
    )
    compiled = phx.equations.compile_finite_element_problem(form, epoch)
    state = compiled.state_space.zeros()
    residual = compiled.weak_residual(state)

    assert compiled.constraint_map is not None
    assert residual.shape == state.shape
    assert jnp.allclose(residual, 0.0)


def test_multi_field_hp_epoch_and_physical_mass_projection_are_native():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
            ),
        ),
    )
    topology, geometry = phx.discretization.fem.initial_finite_element_hp_topology(
        mesh, 2, 8
    )
    epoch = phx.discretization.fem.prepare_multi_field_finite_element_hp_epoch(
        topology,
        geometry,
        {
            "u": ("H1", (), (0, 0)),
            "q": ("L2", (2,), (1, 0)),
        },
    )
    assert tuple(space.name for space in epoch.discretization.field_spaces) == ("u", "q")
    assert epoch.discretization.elements[1][0].conformity == "L2"
    assert epoch.discretization.elements[1][0].reference_nodes.shape[0] == 12

    source_basis = jnp.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0)))
    target_basis = jnp.asarray(((1.0,), (1.0,), (1.0,)))
    projection = phx.discretization.fem.physical_mass_projection(
        source_basis,
        target_basis,
        jnp.asarray((1.0, 2.0, 1.0)),
        jnp.asarray((1.0, 1.5, 2.0)),
    )
    assert projection.shape == (1, 2)
    assert jnp.allclose(projection @ jnp.ones((2,)), 1.0)
