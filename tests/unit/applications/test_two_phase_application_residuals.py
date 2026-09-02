import jax.numpy as jnp
import numpy as np

import phydrax as phx


two_phase_api = phx.applications.two_phase_flow


def _prepared():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = two_phase_api.TwoPhaseMaterialPlan(contact_angle=np.deg2rad(60.0))
    return two_phase_api.IncompressibleTwoPhaseVOFPlan(discretization, material).prepare()


def test_two_phase_events_expose_wetting_contact_piercing_and_breaking_routes():
    prepared = _prepared()
    previous_alpha = jnp.zeros((4, 4)).at[1:3, 1:3].set(1.0)
    alpha = previous_alpha.at[0, 1].set(0.5).at[1, 1].set(0.5)
    previous = prepared.initial_state(previous_alpha)
    state = prepared.initial_state(alpha)
    body = two_phase_api.TwoPhaseMovingBodyPlan((0.375, 0.375), 0.3, velocity=(0.0, 0.0))
    evidence = two_phase_api.TwoPhaseCapabilityEventPlan(
        maximum_topology_changes=0,
        contact_angle_tolerance=1.0,
        minimum_overturning_normal=0.9,
    ).evaluate(prepared, state, previous_state=previous, body=body)
    assert bool(evidence.wetting_event)
    assert bool(evidence.moving_contact_event)
    assert bool(evidence.surface_piercing_event)
    assert bool(evidence.body_contact_event)
    assert bool(evidence.breaking_or_overturning_event)
    assert evidence.event_code > 0
    assert not bool(evidence.derivative_available)


def test_two_phase_identity_remesh_is_conservative_and_epoch_explicit():
    prepared = _prepared()
    alpha = jnp.linspace(0.0, 1.0, 16).reshape((4, 4))
    velocity = tuple(
        jnp.full(layout.shape, 0.1 * (axis + 1))
        for axis, layout in enumerate(prepared.plan.discretization.face_layouts)
    )
    state = prepared.initial_state(
        alpha,
        velocity,
        {"dye": jnp.linspace(1.0, 2.0, 16).reshape((4, 4))},
    )
    volume = np.asarray(prepared.plan.discretization.cell_volumes).reshape((-1,))
    overlap = np.diag(volume)
    face_transfer = tuple(
        np.eye(int(np.prod(layout.shape)))
        for layout in prepared.plan.discretization.face_layouts
    )
    remesh = two_phase_api.ConservativeTwoPhaseRemeshPlan(
        prepared,
        prepared,
        overlap,
        face_transfer,
    ).transfer(state)
    assert bool(remesh.successful)
    assert bool(remesh.evidence.conservative)
    assert not bool(remesh.evidence.topology_changed)
    assert not bool(remesh.evidence.derivative_available)
    np.testing.assert_allclose(remesh.state.liquid_content, state.liquid_content)
    np.testing.assert_allclose(
        remesh.state.phase_scalar_content["dye"],
        state.phase_scalar_content["dye"],
    )
    for transferred, original in zip(remesh.state.momentum, state.momentum, strict=True):
        np.testing.assert_allclose(transferred, original)
