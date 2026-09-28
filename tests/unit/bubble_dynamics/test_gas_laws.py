#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.bubble_dynamics as bd
from phydrax.equations import IdealGasMaterial, NobleAbelStiffenedGasMaterial


ENVIRONMENT = bd.BubbleEnvironment(101325.0, 293.15)


def _stack(*states: bd.BubbleGasState) -> bd.BubbleGasState:
    energies = [state.internal_energy for state in states]
    assert all(energy is not None for energy in energies)
    return bd.BubbleGasState(
        jnp.stack([state.amount for state in states]),
        jnp.stack([jnp.asarray(energy) for energy in energies]),
        jnp.stack([state.internal for state in states]),
    )


@pytest.mark.parametrize(
    "law",
    (bd.IsothermalIdealBubbleGasLaw(1.4), bd.CaloricIdealBubbleGasLaw(1.4)),
)
def test_compartment_merge_conserves_amount_and_energy_with_nonnegative_mixing_entropy(
    law: bd.AbstractBubbleCompartmentGasLaw,
) -> None:
    volumes = jnp.asarray([2.0e-12, 5.0e-12])
    first = law.initialize(volumes[0], jnp.asarray(2.0e5), ENVIRONMENT)
    second = law.initialize(volumes[1], jnp.asarray(1.0e5), ENVIRONMENT)
    merged = law.merge(_stack(first, second), volumes, jnp.sum(volumes), ENVIRONMENT)
    assert float(merged.amount_residual) == pytest.approx(0.0, abs=1.0e-30)
    assert float(merged.energy_residual) == pytest.approx(0.0, abs=1.0e-24)
    assert float(merged.volume_change) == 0.0
    assert float(merged.entropy_production) > 0.0
    assert bool(merged.admissible)
    total_amount = float(first.amount + second.amount)
    assert float(merged.state.amount) == pytest.approx(total_amount, rel=1.0e-14)
    # Mixing at equal intensive states produces no entropy and no pressure jump.
    equal = law.merge(
        _stack(first, law.initialize(volumes[1], jnp.asarray(2.0e5), ENVIRONMENT)),
        volumes,
        jnp.sum(volumes),
        ENVIRONMENT,
    )
    assert float(equal.entropy_production) == pytest.approx(0.0, abs=1.0e-18)
    assert float(equal.pressure) == pytest.approx(2.0e5, rel=1.0e-12)


@pytest.mark.parametrize(
    "law",
    (bd.IsothermalIdealBubbleGasLaw(1.4), bd.CaloricIdealBubbleGasLaw(1.4)),
)
def test_compartment_split_partitions_extensive_state_without_entropy(
    law: bd.AbstractBubbleCompartmentGasLaw,
) -> None:
    volume = jnp.asarray(6.0e-12)
    state = law.initialize(volume, jnp.asarray(1.5e5), ENVIRONMENT)
    split = law.split(state, volume, jnp.asarray([1.0e-12, 2.0e-12, 3.0e-12]), ENVIRONMENT)
    assert split.policy == "uniform-intensive"
    assert float(split.amount_residual) == pytest.approx(0.0, abs=1.0e-28)
    assert float(split.energy_residual) == pytest.approx(0.0, abs=1.0e-22)
    assert float(split.entropy_production) == pytest.approx(0.0, abs=1.0e-18)
    for index, child_volume in enumerate((1.0e-12, 2.0e-12, 3.0e-12)):
        child = bd.BubbleGasState(
            split.states.amount[index],
            None if split.states.internal_energy is None else split.states.internal_energy[index],
            split.states.internal[index],
        )
        pressure = law.evaluate(jnp.asarray(child_volume), jnp.asarray(0.0), child, ENVIRONMENT).pressure
        assert float(pressure) == pytest.approx(1.5e5, rel=1.0e-12)


def test_caloric_ideal_gas_is_adiabatic_and_matches_the_material_law() -> None:
    gamma = 1.4
    molar_mass = 0.028
    caloric = bd.CaloricIdealBubbleGasLaw(gamma)
    material = bd.MaterialBubbleGasLaw(
        IdealGasMaterial(gamma, bd.MOLAR_GAS_CONSTANT / molar_mass), molar_mass
    )
    volume, pressure = jnp.asarray(1.0e-15), jnp.asarray(2.0e5)
    first = caloric.initialize(volume, pressure, ENVIRONMENT)
    second = material.initialize(volume, pressure, ENVIRONMENT)
    assert float(second.amount) == pytest.approx(float(first.amount), rel=1.0e-10)
    compressed = 0.5 * volume
    rate = jnp.asarray(-3.0e-9)
    left = caloric.evaluate(compressed, rate, first, ENVIRONMENT)
    right = material.evaluate(compressed, rate, second, ENVIRONMENT)
    assert float(right.pressure) == pytest.approx(float(left.pressure), rel=1.0e-10)
    assert float(right.temperature) == pytest.approx(float(left.temperature), rel=1.0e-10)
    assert left.energy_rate is not None and right.energy_rate is not None
    assert float(left.energy_rate) == pytest.approx(-float(left.pressure) * -3.0e-9, rel=1.0e-14)
    assert float(right.energy_rate) == pytest.approx(float(left.energy_rate), rel=1.0e-10)


def test_noble_abel_material_gas_reduces_to_ideal_gas_and_reports_covolume() -> None:
    molar_mass = 0.018
    capacity = bd.MOLAR_GAS_CONSTANT / (molar_mass * 0.4)
    ideal_limit = NobleAbelStiffenedGasMaterial(1.4, 0.0, 0.0, capacity)
    reference = bd.CaloricIdealBubbleGasLaw(1.4)
    law = bd.MaterialBubbleGasLaw(ideal_limit, molar_mass)
    volume, pressure = jnp.asarray(1.0e-15), jnp.asarray(1.0e5)
    state = law.initialize(volume, pressure, ENVIRONMENT)
    expected = reference.initialize(volume, pressure, ENVIRONMENT)
    assert float(state.amount) == pytest.approx(float(expected.amount), rel=1.0e-10)
    covolume = NobleAbelStiffenedGasMaterial(1.4, 0.0, 1.0e-3, capacity)
    dense = bd.MaterialBubbleGasLaw(covolume, molar_mass)
    dense_state = dense.initialize(volume, pressure, ENVIRONMENT)
    tiny = dense_state.amount * molar_mass * 1.0e-3 * 0.5
    evaluation = dense.evaluate(jnp.asarray(tiny), jnp.asarray(0.0), dense_state, ENVIRONMENT)
    assert not bool(evaluation.admissible)


def test_hard_core_and_boundary_layer_laws_report_support_and_heat() -> None:
    volume = jnp.asarray(4.0e-15)
    hard = bd.HardCorePolytropicBubbleGasLaw(1.4, 1.0 / 8.86)
    state = hard.initialize(volume, jnp.asarray(1.0e5), ENVIRONMENT)
    core = (1.0 / 8.86) ** 3 * volume
    near = hard.evaluate(1.001 * core, jnp.asarray(0.0), state, ENVIRONMENT)
    assert float(near.hard_core_margin) == pytest.approx(1.0 - 1.0 / 1.001, rel=1.0e-9)
    inside = hard.evaluate(0.5 * core, jnp.asarray(0.0), state, ENVIRONMENT)
    assert not bool(inside.admissible)

    thermal = bd.BoundaryLayerThermalBubbleGasLaw(1.4, 0.0262)
    thermal_state = thermal.initialize(volume, jnp.asarray(1.0e5), ENVIRONMENT)
    at_rest = thermal.evaluate(volume, jnp.asarray(0.0), thermal_state, ENVIRONMENT)
    assert float(at_rest.temperature) == pytest.approx(293.15, rel=1.0e-14)
    assert at_rest.heat_rate is not None
    assert float(at_rest.heat_rate) == 0.0
    compressed = thermal.evaluate(0.5 * volume, jnp.asarray(-1.0e-9), thermal_state, ENVIRONMENT)
    assert float(compressed.pressure) == pytest.approx(2.0e5, rel=1.0e-12)
    hot = bd.BubbleGasState(
        thermal_state.amount,
        jnp.asarray(1.5) * jnp.asarray(thermal_state.internal_energy),
        thermal_state.internal,
    )
    cooling = thermal.evaluate(volume, jnp.asarray(0.0), hot, ENVIRONMENT)
    radius = float(bd.sphere_radius(volume))
    assert cooling.heat_rate is not None
    expected = 4.0 * np.pi * radius * 0.0262 * np.pi * (293.15 - 1.5 * 293.15)
    assert float(cooling.heat_rate) == pytest.approx(expected, rel=1.0e-12)
