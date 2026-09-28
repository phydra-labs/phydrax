#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.bubble_dynamics as bubbles


two_phase_api = phx.applications.two_phase_flow
N = 16
P0 = 1.0e5
ENVIRONMENT = bubbles.BubbleEnvironment(P0, 300.0)


@pytest.fixture(scope="module")
def identity() -> two_phase_api.BubbleComponentPlan:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(N, periodic=True),
            phx.discretization.UniformCellAxisSpec(N, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1000.0, gas_density=1.0, liquid_viscosity=0.0, gas_viscosity=0.0
    )
    two_phase = two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization, material
    ).prepare()
    return two_phase_api.BubbleComponentPlan(
        two_phase, component_capacity=6, maximum_rounds=64, pair_capacity=16
    )


def _field(*disks: tuple[float, float, float]) -> np.ndarray:
    centers = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(centers, centers, indexing="ij")
    alpha = np.ones((N, N))
    for cx, cy, radius in disks:
        alpha[(x - cx) ** 2 + (y - cy) ** 2 < radius**2] = 0.0
    return alpha


BUBBLES = ((0.3, 0.5, 0.12), (0.7, 0.5, 0.12))
LAWS = (
    ("isothermal", bubbles.IsothermalIdealBubbleGasLaw(1.4), 1.0),
    ("caloric", bubbles.CaloricIdealBubbleGasLaw(1.4), 1.4),
)


def _bridged() -> np.ndarray:
    centers = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(centers, centers, indexing="ij")
    alpha = _field(*BUBBLES)
    alpha[(y > 0.45) & (y < 0.55) & (x > 0.3) & (x < 0.7)] = 0.0
    return alpha


@pytest.mark.parametrize(("name", "law", "exponent"), LAWS)
def test_compliance_is_the_process_derivative(
    identity: two_phase_api.BubbleComponentPlan,
    name: str,
    law: bubbles.AbstractBubbleCompartmentGasLaw,
    exponent: float,
) -> None:
    plan = two_phase_api.BubbleCompartmentPlan(law, ENVIRONMENT, capacity=4, dimension=2)
    state = plan.initial_state(identity.initial_state(jnp.asarray(_field(*BUBBLES))), P0)
    volume = np.asarray(state.volume)
    evaluation = plan.evaluate(state, state.volume)
    active = np.asarray(state.active)
    np.testing.assert_allclose(np.asarray(evaluation.pressure)[active], P0, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(evaluation.compliance)[active],
        volume[active] / (exponent * P0),
        rtol=1e-12,
    )
    assert bool(np.all(np.asarray(evaluation.admissible)))


@pytest.mark.parametrize(("name", "law", "exponent"), LAWS)
def test_projection_work_closes_the_first_law(
    identity: two_phase_api.BubbleComponentPlan,
    name: str,
    law: bubbles.AbstractBubbleCompartmentGasLaw,
    exponent: float,
) -> None:
    plan = two_phase_api.BubbleCompartmentPlan(law, ENVIRONMENT, capacity=4, dimension=2)
    state = plan.initial_state(identity.initial_state(jnp.asarray(_field(*BUBBLES))), P0)
    dt = 1.0e-3
    rate = jnp.asarray([-2.0e-3, 1.0e-3, 0.0, 0.0])
    pressure = state.pressure * jnp.asarray([1.02, 0.99, 0.0, 0.0])
    updated, work = plan.commit_projection(
        state, state.volume, state.centroid, pressure, rate, dt
    )
    expected_work = np.asarray(pressure * rate * dt)
    np.testing.assert_allclose(np.asarray(work.work), expected_work, rtol=1e-14)
    energy_change = np.asarray(updated.internal_energy - state.internal_energy)
    np.testing.assert_allclose(energy_change, np.asarray(work.energy_change), rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(work.energy_change) - np.asarray(work.heat),
        -expected_work,
        rtol=1e-12,
        atol=1e-18,
    )
    if name == "caloric":
        np.testing.assert_array_equal(np.asarray(work.energy_change), -expected_work)
        np.testing.assert_array_equal(np.asarray(work.heat), 0.0)
    else:
        np.testing.assert_array_equal(np.asarray(work.energy_change), 0.0)
        np.testing.assert_allclose(np.asarray(work.heat), expected_work, rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(updated.amount), np.asarray(state.amount))


@pytest.mark.parametrize(("name", "law", "exponent"), LAWS)
def test_merge_and_split_conserve_amount_and_energy(
    identity: two_phase_api.BubbleComponentPlan,
    name: str,
    law: bubbles.AbstractBubbleCompartmentGasLaw,
    exponent: float,
) -> None:
    plan = two_phase_api.BubbleCompartmentPlan(law, ENVIRONMENT, capacity=4, dimension=2)
    initial = identity.initial_state(jnp.asarray(_field(*BUBBLES)))
    state = plan.initial_state(initial, jnp.asarray([P0, 1.3 * P0, 0, 0, 0, 0]))
    amount, energy = plan.totals(state)
    merge = identity.propose(initial, jnp.asarray(_bridged()))
    records = two_phase_api.transition_records(merge.event, initial)
    merged = plan.transact(state, records, jnp.full((6,), P0), P0, merge.labels.centroid)
    assert merged.committed
    assert int(merged.state.epoch) == 1
    merged_amount, merged_energy = plan.totals(merged.state)
    np.testing.assert_allclose(merged_amount, amount, rtol=1e-14)
    np.testing.assert_allclose(merged_energy, energy, rtol=1e-14)
    assert np.asarray(merged.state.bubble_id).tolist().count(3) == 1
    assert float(merged.ledger.merge_entropy_production) > 0.0

    # A split whose children fill the parent volume produces no entropy under
    # the uniform-intensive policy and leaves every child at one pressure.
    slot = np.asarray(merged.state.bubble_id).tolist().index(3)
    parent_volume = float(merged.state.volume[slot])
    split_record = two_phase_api.BubbleTransitionRecord(
        2,
        "split",
        parent_ids=(3,),
        child_ids=(4, 5),
        parent_volumes=(parent_volume,),
        child_volumes=(0.3 * parent_volume, 0.7 * parent_volume),
        child_slots=(0, 1),
        overlaps=((3, 4, 0.3 * parent_volume), (3, 5, 0.7 * parent_volume)),
    )
    divided = plan.transact(
        merged.state, (split_record,), jnp.full((6,), P0), P0, jnp.zeros((6, 2))
    )
    assert divided.committed
    split_amount, split_energy = plan.totals(divided.state)
    np.testing.assert_allclose(split_amount, amount, rtol=1e-14)
    np.testing.assert_allclose(split_energy, energy, rtol=1e-14)
    np.testing.assert_allclose(
        float(divided.ledger.split_entropy_production), 0.0, atol=1e-12
    )
    active = np.asarray(divided.state.active)
    pressures = np.asarray(divided.state.pressure)[active]
    np.testing.assert_allclose(pressures, float(merged.state.pressure[slot]), rtol=1e-12)


def test_tiny_bubbles_stay_active_and_vanish_through_the_ledger(
    identity: two_phase_api.BubbleComponentPlan,
) -> None:
    law = bubbles.CaloricIdealBubbleGasLaw(1.4)
    plan = two_phase_api.BubbleCompartmentPlan(law, ENVIRONMENT, capacity=4, dimension=2)
    initial = identity.initial_state(jnp.asarray(_field(*BUBBLES)))
    state = plan.initial_state(initial, P0)
    tiny = state.volume * jnp.asarray([1.0e-9, 1.0, 1.0, 1.0])
    evaluation = plan.evaluate(state, tiny)
    assert bool(np.all(np.asarray(evaluation.admissible)))
    assert float(evaluation.pressure[0]) > 1.0e6 * P0
    vanished = identity.propose(initial, jnp.asarray(_field(BUBBLES[1])))
    records = two_phase_api.transition_records(vanished.event, initial)
    transaction = plan.transact(
        state, records, jnp.full((6,), P0), P0, vanished.labels.centroid
    )
    assert transaction.committed
    amount, energy = plan.totals(state)
    remaining_amount, remaining_energy = plan.totals(transaction.state)
    np.testing.assert_allclose(
        remaining_amount + transaction.ledger.vanished_amount, amount, rtol=1e-14
    )
    np.testing.assert_allclose(
        remaining_energy + transaction.ledger.vanished_energy, energy, rtol=1e-14
    )


def test_capacity_refusal_leaves_the_registry_unchanged(
    identity: two_phase_api.BubbleComponentPlan,
) -> None:
    law = bubbles.IsothermalIdealBubbleGasLaw(1.4)
    plan = two_phase_api.BubbleCompartmentPlan(law, ENVIRONMENT, capacity=2, dimension=2)
    initial = identity.initial_state(jnp.asarray(_field(*BUBBLES)))
    state = plan.initial_state(initial, P0)
    created = identity.propose(initial, jnp.asarray(_field(*BUBBLES, (0.5, 0.15, 0.08))))
    records = two_phase_api.transition_records(created.event, initial)
    transaction = plan.transact(
        state, records, jnp.full((6,), P0), P0, created.labels.centroid
    )
    assert transaction.status is two_phase_api.BubbleCompartmentStatus.CAPACITY_EXCEEDED
    np.testing.assert_array_equal(
        np.asarray(transaction.state.bubble_id), np.asarray(state.bubble_id)
    )
    np.testing.assert_array_equal(
        np.asarray(transaction.state.amount), np.asarray(state.amount)
    )
