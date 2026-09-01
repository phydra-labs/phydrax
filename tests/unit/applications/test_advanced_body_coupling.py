#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _hydrodynamics():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (4.0, 4.0, 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference, jnp.full((4, 4), -1.0), maximum_iterations=100
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        coupling_iterations=4,
        coupling_tolerance=1.0e-7,
    ).prepare()
    state = hydro.initial_state(jnp.zeros((4, 4)))
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    return hydro, continuation


def _body(*, moving=True, modal=False, drag=0.0):
    markers = jnp.asarray(
        (
            (-0.1, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, -0.1, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, -0.1),
            (0.0, 0.0, 0.1),
        )
    )
    normals = markers / jnp.linalg.norm(markers, axis=-1)[:, None]
    basis = jnp.linspace(-1.0, 1.0, markers.shape[0])[:, None] if modal else None
    return phx.applications.hydrodynamics.MappedRigidHydroelasticBodyPlan(
        markers,
        normals,
        jnp.ones((markers.shape[0],)),
        moving=moving,
        viscous_drag=drag,
        modal_basis=basis,
        modal_mass=None if not modal else (1.0,),
        modal_stiffness=None if not modal else (2.0,),
        modal_damping=None if not modal else (0.1,),
        tolerance=1.0e-6,
    )


def test_mapped_marker_transfer_is_adjoint():
    hydro, continuation = _hydrodynamics()
    view = hydro.view(continuation.state)
    body_plan = _body(moving=False)
    body = body_plan.initial_state(position=(2.0, 2.0, -0.5))

    _, evidence = body_plan.gather_normal_velocity(
        hydro, view.geometry, view.velocity, body
    )

    assert bool(evidence.valid)
    assert jnp.abs(evidence.adjoint_defect) <= 1.0e-6


def test_fixed_and_moving_rigid_body_constraints_close():
    hydro, continuation = _hydrodynamics()
    view = hydro.view(continuation.state)
    for moving in (False, True):
        body_plan = _body(moving=moving, drag=0.2)
        body = body_plan.initial_state(
            position=(2.0, 2.0, -0.5),
            linear_velocity=(0.0 if not moving else 1.0e-4, 0.0, 0.0),
        )
        _, _, _, evidence = body_plan.couple(
            hydro,
            view.geometry,
            continuation.state.momentum,
            view.velocity,
            body,
            jnp.asarray(0.01),
        )

        assert bool(evidence.successful)
        assert evidence.constraint_residual <= 1.0e-6
        assert jnp.isfinite(evidence.viscous_dissipation)


def test_modal_hydroelastic_state_advances_with_finite_work():
    hydro, continuation = _hydrodynamics()
    body_plan = _body(moving=True, modal=True)
    body = body_plan.initial_state(position=(2.0, 2.0, -0.5))
    state = phx.applications.hydrodynamics.RigidHydroelasticContinuationState(
        continuation,
        body,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    method = phx.applications.hydrodynamics.RigidHydroelasticALEMethod(
        phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro),
        body_plan,
    )

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.002),
        None,
    )

    assert bool(result.successful)
    assert jnp.all(jnp.isfinite(result.accepted_state.body.modal_coordinates))
    assert jnp.isfinite(result.accepted_state.body_work)
