#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...circuit._dae import _terminal_values, PreparedCircuitDAE
from ...circuit._elements import TwoTerminalConductanceLaw
from ...dynamics import DifferentialAlgebraicSystem
from ._ecm import ThermalEquivalentCircuitParameters


class ThermalGraphPlan(StrictModule, NonTrainableState):
    """Ordered thermal path and immutable, conservative resistor heat allocation.

    Incidence is the adjacent difference ``T[k] - T[k+1]``. The allocation
    matrix is accepted only at preparation and retained as sparse entries, not
    an N-by-N thermal matrix. No thermal storage is introduced by this graph.
    """

    node_ids: tuple[str, ...] = eqx.field(static=True)
    link_ids: tuple[str, ...] = eqx.field(static=True)
    allocation: tuple[tuple[int, int, float], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_ids: Sequence[str],
        /,
        *,
        link_ids: Sequence[str] = (),
        heat_allocation: ArrayLike | None = None,
    ):
        nodes, links = tuple(node_ids), tuple(link_ids)
        for values, owner in ((nodes, "node"), (links, "link")):
            if any(not isinstance(v, str) or not v or v != v.strip() for v in values):
                raise ValueError(f"Thermal graph {owner} IDs must be canonical strings.")
            if len(set(values)) != len(values):
                raise ValueError(f"Thermal graph {owner} IDs must be unique.")
        if not nodes:
            raise ValueError("Thermal graph must have at least one node.")
        if heat_allocation is None:
            if links:
                raise ValueError("Every resistor link requires explicit heat allocation.")
            weights = np.zeros((len(nodes), 0))
        else:
            weights = np.asarray(heat_allocation, dtype=float)
        if weights.shape != (len(nodes), len(links)):
            raise ValueError(
                "Heat allocation must have one row per node and column per link."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Heat allocation must be finite and nonnegative.")
        if not np.allclose(weights.sum(axis=0), 1.0, rtol=0.0, atol=1e-14):
            raise ValueError("Each heat allocation column must sum to one.")
        allocation = tuple(
            (int(row), int(column), float(weights[row, column]))
            for row, column in zip(*np.nonzero(weights), strict=True)
        )
        self.node_ids, self.link_ids, self.allocation = nodes, links, allocation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-thermal-path",
                "nodes": nodes,
                "links": links,
                "allocation": allocation,
                "incidence": "adjacent-left-minus-right",
                "storage_owner": "cell-implicit-law",
            }
        )

    def edge_outflow(self, temperatures: Array, conductance: Array, /) -> Array:
        """Apply B.T G B by differences and scatter; interior heat cancels."""
        if temperatures.shape != (len(self.node_ids),):
            raise ValueError("Thermal temperatures must match ordered nodes.")
        if conductance.shape != (len(self.node_ids) - 1,):
            raise ValueError("Thermal edge conductance must match adjacent edges.")
        flux = conductance * (temperatures[:-1] - temperatures[1:])
        return jnp.zeros_like(temperatures).at[:-1].add(flux).at[1:].add(-flux)

    def allocate(self, link_heat: Array, /) -> Array:
        if link_heat.shape != (len(self.link_ids),):
            raise ValueError("Resistor heat must match ordered link IDs.")
        result = jnp.zeros((len(self.node_ids),), dtype=link_heat.dtype)
        for node, link, weight in self.allocation:
            result = result.at[node].add(weight * link_heat[link])
        return result


class ThermalGraphParameters(StrictModule):
    """References to the cell parameter owners, plus adjacent edge conductance.

    Heat capacity, ambient conductance and ambient temperature are never copied
    into a second parameter schema. The cell law alone consumes heat capacity.
    """

    cells: tuple[ThermalEquivalentCircuitParameters, ...]
    edge_conductance_w_per_k: Array

    def __init__(
        self,
        cells: Sequence[ThermalEquivalentCircuitParameters],
        edge_conductance_w_per_k: ArrayLike,
        /,
    ):
        owners = tuple(cells)
        if not owners or any(
            not isinstance(p, ThermalEquivalentCircuitParameters) for p in owners
        ):
            raise TypeError("Thermal graph cells must be ECM parameter owners.")
        edges = jnp.asarray(edge_conductance_w_per_k, dtype=float)
        if edges.shape != (len(owners) - 1,):
            raise ValueError("Thermal edge conductance must match adjacent edges.")
        self.cells = owners
        self.edge_conductance_w_per_k = eqx.error_if(
            edges,
            jnp.any(~jnp.isfinite(edges)) | jnp.any(edges < 0.0),
            "Thermal edge conductance must be finite and nonnegative.",
        )


class ThermalGraphTerms(StrictModule):
    edge_outflow_w: Array
    ambient_outflow_w: Array
    resistor_heat_w: Array
    allocated_heat_w: Array

    @property
    def residual_addition_w(self) -> Array:
        return self.edge_outflow_w + self.ambient_outflow_w - self.allocated_heat_w


class _ThermalGraphResidual(StrictModule):
    base: PreparedCircuitDAE
    graph: ThermalGraphPlan
    temperature_indices: tuple[int, ...] = eqx.field(static=True)
    link_indices: tuple[int, ...] = eqx.field(static=True)
    parameters: Callable[[Any], ThermalGraphParameters]
    inputs: Callable[[Array, Array, Any], Array] | None

    def graph_terms(
        self, time: Array, state: Array, rate: Array, args: Any, /
    ) -> ThermalGraphTerms:
        parameters = self.parameters(args)
        if len(parameters.cells) != len(self.graph.node_ids):
            raise ValueError("Thermal parameter owners do not match graph nodes.")
        temperatures = state[jnp.asarray(self.temperature_indices)]
        ambient = jnp.stack(
            tuple(
                p.thermal_conductance_w_per_k
                * (temperatures[k] - p.ambient_temperature_k)
                for k, p in enumerate(parameters.cells)
            )
        )
        heat = []
        plan = self.base.plan
        for index in self.link_indices:
            instance, law = plan.circuit.instances[index], plan.laws[index]
            voltages = _terminal_values(
                instance.nodes, plan.circuit.ground, plan.layout, state
            )
            voltage_rates = _terminal_values(
                instance.nodes, plan.circuit.ground, plan.layout, rate
            )
            start, stop = plan.layout.auxiliary_ranges[index]
            evaluation = law.evaluate(
                time,
                voltages,
                voltage_rates,
                state[start:stop],
                rate[start:stop],
                jnp.zeros((0,), dtype=state.dtype),
                args,
            )
            # The actual native resistor constitutive current, never commanded I.
            heat.append((voltages[0] - voltages[1]) * evaluation.terminal_currents[0])
        link_heat = jnp.stack(heat) if heat else jnp.zeros((0,), dtype=state.dtype)
        return ThermalGraphTerms(
            self.graph.edge_outflow(temperatures, parameters.edge_conductance_w_per_k),
            ambient,
            link_heat,
            self.graph.allocate(link_heat),
        )

    def __call__(self, time: Array, state: Array, rate: Array, args: Any, /) -> Array:
        inputs = None if self.inputs is None else self.inputs(time, state, args)
        residual = self.base.system.evaluate(time, state, rate, args, inputs=inputs)
        addition = self.graph_terms(time, state, rate, args).residual_addition_w
        return residual.at[jnp.asarray(self.temperature_indices)].add(addition)


def augment_circuit_thermal_graph(
    base: PreparedCircuitDAE,
    graph: ThermalGraphPlan,
    temperature_indices: Sequence[int],
    parameters: Callable[[Any], ThermalGraphParameters],
    /,
    *,
    inputs: Callable[[Array, Array, Any], Array] | None = None,
    trial_validity: Callable | None = None,
    trial_validity_id: str | None = None,
) -> DifferentialAlgebraicSystem:
    """Construct the sole autonomous DAE owner for a battery circuit and graph.

    ``base`` is topology and cell residual only. Initialization, stepping,
    events, replay and diagnostics must all consume this returned system.
    """
    if not isinstance(base, PreparedCircuitDAE) or not isinstance(
        graph, ThermalGraphPlan
    ):
        raise TypeError(
            "Thermal augmentation requires native prepared circuit and graph."
        )
    indices = tuple(temperature_indices)
    if len(indices) != len(graph.node_ids) or len(set(indices)) != len(indices):
        raise ValueError("Each thermal node must identify a distinct temperature row.")
    if any(
        isinstance(i, bool)
        or not isinstance(i, int)
        or i < 0
        or i >= base.plan.layout.size
        or base.plan.layout.roles[i] != "differential"
        for i in indices
    ):
        raise ValueError("Thermal temperatures must identify differential circuit rows.")
    if (base.system.input_layout is None) != (inputs is None):
        raise ValueError(
            "Thermal augmentation must bind exactly the native circuit inputs."
        )
    links = tuple(base.plan.layout.instance_ids.index(name) for name in graph.link_ids)
    for index in links:
        law = base.plan.laws[index]
        if not isinstance(law, TwoTerminalConductanceLaw) or bool(law.conductance <= 0.0):
            raise ValueError("Thermal links must be actual positive massless resistors.")
    residual = _ThermalGraphResidual(base, graph, indices, links, parameters, inputs)
    return DifferentialAlgebraicSystem(
        residual,
        state_shape=base.system.state_shape,
        structure=base.system.structure,
        state_scale=base.system.state_scale,
        state_rate_scale=base.system.state_rate_scale,
        residual_scale=base.system.residual_scale,
        trial_validity=trial_validity,
        trial_validity_id=trial_validity_id,
        system_id=canonical_fingerprint(
            {
                "kind": "battery-graph-augmented-circuit",
                "base": base.prepared_id,
                "graph": graph.plan_id,
                "temperature_rows": indices,
            }
        ),
    )


__all__ = [
    "ThermalGraphPlan",
    "ThermalGraphParameters",
    "ThermalGraphTerms",
    "augment_circuit_thermal_graph",
]
