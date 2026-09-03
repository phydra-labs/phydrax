from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _discretization():
    grid = phx.discretization.AdaptiveDyadicGridPlan(
        phx.discretization.MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3),
        cell_capacity=32,
    )
    root = grid.prepare()
    refine = jnp.zeros((grid.cell_capacity,), dtype=bool).at[0].set(True)
    topology = grid.adapt(root, refine_mask=refine).accepted
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.DyadicFiniteVolumePlan(
        topology,
        component_names=system.component_names,
    ).prepare()
    return system, discretization


def test_dyadic_finite_volume_faces_close_and_split_interfaces() -> None:
    _, discretization = _discretization()
    assert discretization.cell_count == 4
    assert discretization.face_count == 12
    assert len(discretization.boundary_patch_names) == 4
    np.testing.assert_allclose(jnp.sum(discretization.cell_volumes), 1.0)
    closure = jnp.zeros_like(discretization.cell_centers)
    closure = closure.at[discretization.owner_cells].add(discretization.area_vectors)
    safe_neighbour = jnp.maximum(discretization.neighbour_cells, 0)
    closure = closure.at[safe_neighbour].add(
        jnp.where(
            (discretization.neighbour_cells >= 0)[:, None],
            -discretization.area_vectors,
            0.0,
        )
    )
    np.testing.assert_allclose(closure, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        jnp.sum(discretization.face_quadrature_weights, axis=1),
        discretization.face_measures,
    )


def test_dyadic_finite_volume_executes_existing_conservation_runtime() -> None:
    system, discretization = _discretization()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "dyadic-constant-state",
        "state",
        system,
        boundaries,
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.2, -0.1, 1.0)),
        (discretization.cell_count, system.component_count),
    )
    state = system.primitive_to_conserved(primitive)
    residual = eqx.filter_jit(compiled.dynamics)(jnp.asarray(0.0), state, None)
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)


def test_dyadic_finite_volume_decomposes_coarse_fine_faces_conservatively() -> None:
    grid = phx.discretization.AdaptiveDyadicGridPlan(
        phx.discretization.MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3),
        cell_capacity=64,
    )
    level_one = grid.adapt(
        grid.prepare(),
        refine_mask=jnp.zeros((grid.cell_capacity,), dtype=bool).at[0].set(True),
    ).accepted
    selected = next(
        int(slot)
        for slot in np.flatnonzero(np.asarray(level_one.leaf_active))
        if int(level_one.prefixes[slot]) == 0
    )
    topology = grid.adapt(
        level_one,
        refine_mask=jnp.zeros((grid.cell_capacity,), dtype=bool).at[selected].set(True),
    ).accepted
    discretization = phx.discretization.DyadicFiniteVolumePlan(topology).prepare()
    assert discretization.cell_count == 7
    internal = discretization.neighbour_cells >= 0
    assert bool(jnp.any(jnp.isclose(discretization.face_measures[internal], 0.25)))
    closure = jnp.zeros_like(discretization.cell_centers)
    closure = closure.at[discretization.owner_cells].add(discretization.area_vectors)
    closure = closure.at[jnp.maximum(discretization.neighbour_cells, 0)].add(
        jnp.where(internal[:, None], -discretization.area_vectors, 0.0)
    )
    np.testing.assert_allclose(closure, 0.0, atol=1.0e-14)
