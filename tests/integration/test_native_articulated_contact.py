#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""End-to-end native articulated contact rollout through the complete plant."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.contact._articulated_step import ArticulatedActuationMode
from phydrax.applications.robotics._articulated_plant import (
    ArticulatedPlantCommand,
    NativeArticulatedPlantPlan,
    prepare_articulated_plant,
)
from phydrax.discretization.particle._reduced_dynamics import reduced_energy
from phydrax.dynamics import PlantStepContext
from tests._support.native_articulated import (
    dof,
    LBVH_TRAVERSAL_VISITS,
    planar_slider_model,
    route_budget,
    standard_gravity,
    step_policy,
)


def _search(model: Any, backend: str) -> dict[str, Any]:
    return {
        "activation_distance": 0.05,
        "search_backend": backend,
        "route_capacities": route_budget(model),
        "maximum_traversal_visits": LBVH_TRAVERSAL_VISITS if backend == "lbvh" else None,
    }


def _rollout(
    plant: Any, state: Any, steps: int, push: float
) -> tuple[list[Any], list[PlantStepContext], list[ArticulatedPlantCommand]]:
    results: list[Any] = []
    contexts: list[PlantStepContext] = []
    commands: list[ArticulatedPlantCommand] = []
    drive = dof(plant.model, "drive")
    for _ in range(steps):
        command = ArticulatedPlantCommand(jnp.zeros((2,)).at[drive].set(push))
        context = PlantStepContext(state.time, state.time + 0.01, state.step_index)
        result = plant.step(context, state, command, plant.bind_parameters())
        results.append(result)
        contexts.append(context)
        commands.append(command)
        state = result.accepted_state
    return results, contexts, commands


def _vector(index: int, value: float) -> np.ndarray:
    vector = np.zeros((2,))
    vector[index] = value
    return vector


def test_dropped_pushed_slider_lands_slides_and_stops_with_bounded_energy() -> None:
    model = planar_slider_model(drive_bounds=(-1.0, 0.03))
    prepared = model.prepare()
    drive, lift = dof(prepared, "drive"), dof(prepared, "lift")
    plan = NativeArticulatedPlantPlan(
        step_policy(),
        gravity=standard_gravity(),
        actuation_mode=ArticulatedActuationMode.GENERALIZED_EFFORT,
        initial_lifted_configuration=_vector(lift, 0.003),
        initial_velocity=_vector(drive, 0.5),
        **_search(model, "lbvh"),
    )
    plant = prepare_articulated_plant(prepared, plan)
    state = plant.reset(jax.random.key(1), plant.bind_parameters()).accepted_state
    results, contexts, commands = _rollout(plant, state, 12, push=1.0)
    gravity = standard_gravity()
    articulation = plant.model.articulation

    for result in results:
        assert bool(result.successful), int(result.status)
        assert float(result.evidence.step.covered_fraction) == 1.0
        ledger = result.evidence.step.ledger
        assert float(ledger.correction_loss) >= -float(ledger.roundoff_allowance)
        assert float(ledger.positive_contact_work) <= 1.0e-8

    energies = [
        float(
            reduced_energy(
                articulation,
                result.accepted_state.payload.configuration,
                result.accepted_state.payload.velocity,
                gravity,
            ).total
        )
        for result in results
    ]
    works = [float(result.evidence.step.ledger.source_work) for result in results]
    for previous, current, work in zip(
        energies[:-1], energies[1:], works[1:], strict=True
    ):
        assert current - previous - work <= 1.0e-8

    final = results[-1].accepted_state.payload
    lifted = float(final.configuration[drive])
    assert lifted <= 0.03 + 1.0e-9
    assert float(final.stop_impulse[1]) > 0.0
    assert float(final.configuration[lift]) >= -1.0e-9
    contact_seen = [
        float(jnp.sum(result.accepted_state.payload.sensors.contact_normal_impulse))
        for result in results
    ]
    assert contact_seen[0] == 0.0 and max(contact_seen) > 0.0

    checkpoint = plant.checkpoint(state)
    replay = plant.replay(
        checkpoint,
        contexts,
        commands,
        plant.bind_parameters(),
        expected_digests=[
            plant.state_digest(result.accepted_state) for result in results
        ],
    )
    assert replay.matched


def test_batched_lbvh_rollout_matches_unbatched_exhaustive_cases() -> None:
    model = planar_slider_model()

    def plant_for(case_ndim: int, backend: str) -> Any:
        plan = NativeArticulatedPlantPlan(
            step_policy(),
            gravity=standard_gravity(),
            actuation_mode=ArticulatedActuationMode.GENERALIZED_EFFORT,
            initial_lifted_configuration=[0.0, 0.0],
            case_ndim=case_ndim,
            **_search(model, backend),
        )
        return prepare_articulated_plant(model, plan)

    batched = plant_for(1, "lbvh")
    single = plant_for(0, "exhaustive")
    drive = dof(single.model, "drive")
    pushes = np.asarray([0.0, 2.0, 40.0])
    state = batched.reset(
        jax.random.split(jax.random.key(2), 3), batched.bind_parameters(), case_shape=(3,)
    ).accepted_state
    command = ArticulatedPlantCommand(
        jnp.zeros((3, 2)).at[:, drive].set(jnp.asarray(pushes))
    )
    context = PlantStepContext(state.time, state.time + 0.01, state.step_index)
    batch = batched.step(context, state, command, batched.bind_parameters())

    for index, push in enumerate(pushes):
        case_state = single.reset(
            jax.random.key(3), single.bind_parameters()
        ).accepted_state
        case_context = PlantStepContext(
            case_state.time, case_state.time + 0.01, case_state.step_index
        )
        case = single.step(
            case_context,
            case_state,
            ArticulatedPlantCommand(jnp.zeros((2,)).at[drive].set(push)),
            single.bind_parameters(),
        )
        assert bool(case.successful) == bool(batch.successful[index])
        assert np.allclose(
            np.asarray(case.accepted_state.payload.velocity),
            np.asarray(batch.accepted_state.payload.velocity[index]),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
    # Static friction holds the unpushed case; a push beyond mu_s m g slides.
    assert abs(float(batch.accepted_state.payload.velocity[0, drive])) <= 1.0e-9
    assert float(batch.accepted_state.payload.velocity[2, drive]) > 0.0
