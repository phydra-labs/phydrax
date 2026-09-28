#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


two_phase_api = phx.applications.two_phase_flow


def _prepared(
    cells: int,
    *,
    viscosity: float = 0.0,
    law: phx.discretization.LinearSurfaceTensionLaw | None = None,
) -> two_phase_api.PreparedIncompressibleTwoPhaseVOF:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(cells, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1.0,
        gas_density=1.0,
        liquid_viscosity=viscosity,
        gas_viscosity=viscosity,
    )
    return two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=1000,
        surface_tension_law=law,
        surface_tension_scalar=None if law is None else "temperature",
    ).prepare()


def _disk(cells: int, center: tuple[float, float], radius: float) -> np.ndarray:
    samples = 4
    lower = np.arange(cells) / cells
    offsets = (np.arange(samples) + 0.5) / (samples * cells)
    content = np.zeros((cells, cells), dtype=np.float64)
    for offset_x in offsets:
        for offset_y in offsets:
            x, y = np.meshgrid(lower + offset_x, lower + offset_y, indexing="ij")
            content += (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius**2
    return content / samples**2


def test_bubbly_step_commits_flux_markers_and_rolls_back_atomically() -> None:
    cells = 12
    two_phase = _prepared(cells)
    first = _disk(cells, (0.34, 0.5), 0.14)
    second = _disk(cells, (0.66, 0.5), 0.14)
    alpha = 1.0 - first - second
    color = np.where(second > first, 1, 0)
    identity = two_phase_api.BubbleComponentPlan(
        two_phase, component_capacity=6, maximum_rounds=64, pair_capacity=16
    )
    markers = two_phase_api.MultiMarkerPlan(
        two_phase,
        marker_capacity=3,
        component_capacity=6,
        pair_capacity=16,
        proximity_radius=3,
    )
    drainage = two_phase_api.FilmDrainageCoalescencePlan(
        regime="immobile",
        geometry="planar",
        pair_capacity=16,
        id_upper_bound=1_000_000,
        continuous_viscosity=1.0e-3,
        surface_tension=0.05,
        initial_film_thickness=1.0e-6,
        critical_thickness=5.0e-8,
    )
    plan = two_phase_api.BubblyFlowPlan(
        two_phase,
        identity,
        markers=markers,
        coalescence=drainage,
        near_contact_pressure=1.0e-3,
    )
    x_faces = two_phase.plan.discretization.face_centers[0][..., 0]
    velocity = (
        jnp.where(x_faces < 0.5, 0.25, -0.25),
        jnp.zeros(two_phase.plan.discretization.face_layouts[1].shape),
    )
    fluid = two_phase.initial_state(jnp.asarray(alpha), velocity)
    bubbles = plan.initial_state(jnp.asarray(alpha), color=jnp.asarray(color))
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase, bubbles=plan)
    initial = method.initial_continuation(fluid, bubbles=bubbles)

    accepted = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        initial,
        jnp.asarray(0.2 / cells),
        None,
    )
    assert bool(accepted.successful)
    assert accepted.accepted_state.bubble_evidence is not None
    assert float(accepted.accepted_state.bubble_evidence.marker_sum_residual) < 1e-12
    np.testing.assert_allclose(
        np.asarray(accepted.accepted_state.fluxes.previous_liquid_content),
        np.asarray(initial.state.liquid_content),
    )
    assert float(accepted.accepted_state.ledger.contact_work) == float(
        accepted.accepted_state.bubble_evidence.near_contact_work
    )
    assert abs(float(accepted.accepted_state.ledger.contact_work)) > 0.0
    assert float(accepted.accepted_state.ledger.pressure_work) == 0.0

    refused = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        initial,
        jnp.asarray(3.0 / cells),
        None,
    )
    assert not bool(refused.successful)
    np.testing.assert_array_equal(
        np.asarray(refused.accepted_state.state.liquid_content),
        np.asarray(initial.state.liquid_content),
    )
    refused_bubbles = refused.accepted_state.bubbles
    initial_bubbles = initial.bubbles
    if (
        refused_bubbles is None
        or refused_bubbles.markers is None
        or initial_bubbles is None
        or initial_bubbles.markers is None
    ):
        raise AssertionError("Bubbly rollback lost its marker state.")
    np.testing.assert_array_equal(
        np.asarray(refused_bubbles.markers.content),
        np.asarray(initial_bubbles.markers.content),
    )


def test_variable_surface_tension_transports_material_content_and_drives_ygb_direction() -> (
    None
):
    cells = 16
    slope = -1.0e-2
    gradient = 1.0
    radius = 0.25
    viscosity = 1.0
    law = phx.discretization.LinearSurfaceTensionLaw(0.05, slope, 0.5)
    _, law_jvp = jax.jvp(
        lambda scalar: law(
            jnp.zeros((2,), dtype=scalar.dtype),
            scalar.reshape((1,)),
        )[0],
        (jnp.asarray(0.5),),
        (jnp.asarray(1.0),),
    )
    np.testing.assert_allclose(np.asarray(law_jvp), slope)
    two_phase = _prepared(cells, viscosity=viscosity, law=law)
    alpha = _disk(cells, (0.5, 0.5), radius)
    x = np.asarray(two_phase.plan.discretization.cell_centers)[..., 0]
    state = two_phase.initial_state(
        jnp.asarray(alpha), material_scalars={"temperature": jnp.asarray(gradient * x)}
    )
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(state)
    initial_content = float(jnp.sum(state.material_scalar_content["temperature"]))

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(1.0e-3),
        None,
    )
    assert bool(result.successful)
    evidence = result.accepted_state.evidence
    assert evidence is not None
    assert float(evidence.material_scalar_residual) < 1e-13
    assert int(evidence.unsupported_face_count) == 0
    assert float(evidence.surface_tension_minimum) > 0.0
    assert float(evidence.surface_tension_maximum) > float(
        evidence.surface_tension_minimum
    )
    assert float(evidence.marangoni_force_norm) > 0.0
    final_content = float(
        jnp.sum(result.accepted_state.state.material_scalar_content["temperature"])
    )
    np.testing.assert_allclose(final_content, initial_content, rtol=1e-13, atol=1e-13)

    velocity = two_phase.velocity(result.accepted_state.state)
    cell_velocity = 0.5 * (velocity[0] + jnp.roll(velocity[0], -1, axis=0))
    measured = float(jnp.sum(cell_velocity * alpha) / jnp.sum(alpha))
    ygb = -slope * gradient * radius / (4.0 * (viscosity + viscosity))
    assert measured * ygb > 0.0
