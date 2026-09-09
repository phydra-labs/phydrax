#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._ecm import ThermalEquivalentCircuitParameters
from phydrax.applications.battery._properties import ConstantPropertyLaw
from phydrax.applications.battery._thermal_graph import (
    augment_circuit_thermal_graph,
    ThermalGraphParameters,
    ThermalGraphPlan,
)
from phydrax.circuit import (
    AbstractImplicitCircuitLaw,
    CircuitElement,
    CircuitElementEvaluation,
    CircuitElementStateLayout,
    CircuitInstance,
    ElectricalWaveReference,
    IndependentVoltageSourceLaw,
    NodalCircuit,
    NodalPort,
    prepare_circuit_dae,
    Resistor,
)


class _HeatStore(AbstractImplicitCircuitLaw):
    """Thermal capacity fixture, not a battery or a pack model."""

    def __init__(self):
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout(("differential",))
        self.input_names = ()
        self.law_id = "test-heat-store"

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ):
        del time, terminal_voltages, terminal_voltage_rates, state, inputs
        return CircuitElementEvaluation(
            jnp.zeros(2), args.cells[0].heat_capacity_j_per_k * state_rate
        )


def test_actual_resistor_heat_allocation_and_internal_edge_cancellation():
    store = CircuitElement(_HeatStore(), element_id="heat-store")
    source = CircuitElement(IndependentVoltageSourceLaw(5.0), element_id="voltage-source")
    circuit = NodalCircuit(
        (
            CircuitInstance("left", store, ("v", "0")),
            CircuitInstance("right", store, ("v", "0")),
            CircuitInstance("link", Resistor(2.0), ("v", "0")),
            CircuitInstance("source", source, ("v", "0")),
        ),
        (NodalPort("observation", "v", "0", ElectricalWaveReference(1.0)),),
        ground="0",
        circuit_id="thermal-graph-constitutive-regression",
    )
    base = prepare_circuit_dae(circuit)
    left, right = (
        base.plan.layout.instance_range("left")[0],
        base.plan.layout.instance_range("right")[0],
    )
    graph = ThermalGraphPlan(
        ("left", "right"), link_ids=("link",), heat_allocation=((0.25,), (0.75,))
    )
    system = augment_circuit_thermal_graph(base, graph, (left, right), lambda args: args)
    ocv = ConstantPropertyLaw(
        3.7,
        (0.0, 1.0),
        quantity="reference-open-circuit-voltage",
        coordinate="state_of_charge",
        value_unit="V",
        coordinate_unit="1",
        source_id="test:thermal-graph",
    )
    entropic = ConstantPropertyLaw(
        0.0,
        (0.0, 1.0),
        quantity="entropic-coefficient",
        coordinate="state_of_charge",
        value_unit="V/K",
        coordinate_unit="1",
        source_id="test:thermal-graph",
    )
    p = ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1,)),
        jnp.asarray((10.0,)),
        1000.0,
        100.0,
        0.5,
        300.0,
        300.0,
        ocv,
        entropic,
    )
    parameters = ThermalGraphParameters((p, p), jnp.asarray((1.0,)))
    # Source current is an independent algebraic guess. Link heat must still be
    # the resistor's actual 5 V * (5 V / 2 ohm), not that unrelated guess.
    state = jnp.asarray((5.0, 310.0, 290.0, -100.0))
    rate = jnp.zeros_like(state)
    terms = system.residual.graph_terms(jnp.asarray(0.0), state, rate, parameters)
    np.testing.assert_allclose(terms.resistor_heat_w, (12.5,))
    np.testing.assert_allclose(terms.allocated_heat_w, (3.125, 9.375))
    np.testing.assert_allclose(terms.edge_outflow_w, (20.0, -20.0))
    np.testing.assert_allclose(jnp.sum(terms.residual_addition_w), -12.5)
    residual = system.evaluate(0.0, state, rate, parameters)
    np.testing.assert_allclose(residual[jnp.asarray((left, right))], (21.875, -34.375))
    raised_rate = rate.at[left].set(0.25).at[right].set(-0.5)
    delta = system.evaluate(0.0, state, raised_rate, parameters) - residual
    # Exactly one cell-owned heat capacity, not another graph capacity.
    np.testing.assert_allclose(delta[jnp.asarray((left, right))], (25.0, -50.0))


def test_heat_allocation_refuses_energy_creation_and_negative_weights():
    with pytest.raises(ValueError, match="sum to one"):
        ThermalGraphPlan(
            ("left", "right"), link_ids=("link",), heat_allocation=((0.8,), (0.8,))
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ThermalGraphPlan(
            ("left", "right"), link_ids=("link",), heat_allocation=((-0.1,), (1.1,))
        )
