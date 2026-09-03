#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def _model(*slip_systems):
    cp = phx.applications.crystal_plasticity
    return cp.CrystalPlasticityModel(
        slip_systems,
        cp.CrystalPlasticityParameters(8.0, 20.0, 0.1, 1.0, 1.5, 1.0),
    )


def _discretization():
    points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
        )
    )
    blocks = (
        phx.discretization.CellBlock(
            "phase-a",
            "tetrahedron",
            jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
            global_ids=jnp.asarray((10,)),
        ),
        phx.discretization.CellBlock(
            "phase-b",
            "tetrahedron",
            jnp.asarray(((4, 5, 6, 7),), dtype=jnp.int32),
            global_ids=jnp.asarray((20,)),
        ),
    )
    mesh = phx.discretization.CellMesh(points, blocks)
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()


def main():
    cp = phx.applications.crystal_plasticity
    slip_xy = cp.CrystalSlipSystem(
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 1.0, 0.0)),
    )
    slip_yz = cp.CrystalSlipSystem(
        jnp.asarray((0.0, 1.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
    )
    phase_a = _model(slip_xy)
    phase_b = _model(slip_xy, slip_yz)
    quarter_turn = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))

    discretization = _discretization()
    route = cp.CrystalPlasticityRoute(
        discretization,
        "u",
        (
            ("phase-a", phase_a, jnp.eye(3)),
            ("phase-b", phase_b, quarter_turn),
        ),
    )
    accepted_materials = route.initialize()
    form = cp.cpfem_equilibrium_form(
        discretization,
        "u",
        route,
        accepted_materials,
        0.1,
    )
    problem = phx.equations.compile_finite_element_problem(form, discretization)

    deformation = jnp.eye(3).at[0, 1].set(0.25)
    displacement_gradient = deformation - jnp.eye(3)
    displacement = discretization.dof_maps[0].dof_coordinates @ displacement_gradient.T
    residual, auxiliary = problem.residual_with_auxiliary(displacement)
    candidate = auxiliary.trial_state
    promoted = (
        route.commit(candidate) if bool(auxiliary.valid) else route.rollback(candidate)
    )

    print(f"route: {route.route_id}")
    print(f"state widths: {[shape[-1] for shape in route.state_shapes]}")
    print(f"residual norm: {float(jnp.sqrt(jnp.sum(residual**2))):.6e}")
    print(f"accepted: {bool(auxiliary.valid)}")
    print(f"retry requested: {bool(auxiliary.retry_requested)}")
    print(f"material revision: {promoted.states[0].state_version}")


if __name__ == "__main__":
    main()
