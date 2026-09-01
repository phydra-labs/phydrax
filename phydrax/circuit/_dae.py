#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..dynamics._differential_algebraic import DAEStructure, DifferentialAlgebraicSystem
from ._elements import AbstractImplicitCircuitLaw, implicit_law_for
from ._mna import NodalCircuit, NodeId


class CircuitStateLayout(StrictModule):
    node_ids: tuple[NodeId, ...] = eqx.field(static=True)
    auxiliary_ranges: tuple[tuple[int, int], ...] = eqx.field(static=True)
    instance_ids: tuple[str, ...] = eqx.field(static=True)
    roles: tuple[str, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def node_index(self, node: NodeId, /) -> int | None:
        return None if node not in self.node_ids else self.node_ids.index(node)

    def instance_range(self, instance_id: str, /) -> tuple[int, int]:
        if instance_id not in self.instance_ids:
            raise KeyError(f"Unknown circuit instance {instance_id!r}.")
        return self.auxiliary_ranges[self.instance_ids.index(instance_id)]


class CircuitDAEPlan(StrictModule):
    circuit: NodalCircuit
    layout: CircuitStateLayout
    laws: tuple[AbstractImplicitCircuitLaw, ...]
    state_scale: Array
    rate_scale: Array
    residual_scale: Array
    plan_id: str = eqx.field(static=True)


class CircuitDAEDiagnostics(StrictModule):
    residual: Array
    residual_norm: Array
    kcl_norm: Array
    element_residual_norms: Array
    terminal_power: Array
    finite: Array


class PreparedCircuitDAE(StrictModule):
    plan: CircuitDAEPlan
    system: DifferentialAlgebraicSystem
    prepared_id: str = eqx.field(static=True)

    def initialize(
        self,
        /,
        *,
        node_voltages: ArrayLike | None = None,
        auxiliary_state: ArrayLike | None = None,
    ) -> Array:
        nodes = len(self.plan.layout.node_ids)
        auxiliaries = self.plan.layout.size - nodes
        voltages = (
            jnp.zeros((nodes,))
            if node_voltages is None
            else jnp.asarray(node_voltages, dtype=float)
        )
        state = (
            jnp.zeros((auxiliaries,))
            if auxiliary_state is None
            else jnp.asarray(auxiliary_state, dtype=float)
        )
        if voltages.shape != (nodes,) or state.shape != (auxiliaries,):
            raise ValueError("Initial node and auxiliary values have wrong shapes.")
        result = jnp.concatenate((voltages, state))
        if bool(jnp.any(~jnp.isfinite(result))):
            raise ValueError("Initial circuit DAE state must be finite.")
        return result

    def diagnostics(
        self,
        time: ArrayLike,
        state: ArrayLike,
        state_rate: ArrayLike,
        args: Any = None,
        /,
    ) -> CircuitDAEDiagnostics:
        value = jnp.asarray(state)
        rate = jnp.asarray(state_rate)
        residual = self.system.evaluate(time, value, rate, args)
        node_count = len(self.plan.layout.node_ids)
        element_norms = tuple(
            jnp.linalg.norm(residual[start:stop])
            for start, stop in self.plan.layout.auxiliary_ranges
        )
        terminal_power = _terminal_power(
            self.plan.circuit,
            self.plan.layout,
            self.plan.laws,
            jnp.asarray(time),
            value,
            rate,
            args,
        )
        return CircuitDAEDiagnostics(
            residual,
            jnp.linalg.norm(residual),
            jnp.linalg.norm(residual[:node_count]),
            jnp.stack(element_norms) if element_norms else jnp.zeros((0,)),
            terminal_power,
            jnp.all(jnp.isfinite(residual)) & jnp.isfinite(terminal_power),
        )


class CircuitDAERunResult(StrictModule):
    solution: Any
    final_diagnostics: CircuitDAEDiagnostics
    prepared_id: str = eqx.field(static=True)


class _CircuitResidual(StrictModule):
    circuit: NodalCircuit
    layout: CircuitStateLayout
    laws: tuple[AbstractImplicitCircuitLaw, ...]

    def __call__(
        self, time: Array, state: Array, state_rate: Array, args: Any, /
    ) -> Array:
        node_count = len(self.layout.node_ids)
        residual = jnp.zeros(
            (self.layout.size,), dtype=jnp.result_type(state, state_rate)
        )
        inputs = args["inputs"] if isinstance(args, dict) and "inputs" in args else None
        law_args = args["args"] if isinstance(args, dict) and "args" in args else args
        for instance, law, (start, stop) in zip(
            self.circuit.instances,
            self.laws,
            self.layout.auxiliary_ranges,
            strict=True,
        ):
            voltages = _terminal_values(
                instance.nodes, self.circuit.ground, self.layout, state
            )
            voltage_rates = _terminal_values(
                instance.nodes, self.circuit.ground, self.layout, state_rate
            )
            evaluation = law.evaluate(
                time,
                voltages,
                voltage_rates,
                state[start:stop],
                state_rate[start:stop],
                inputs,
                law_args,
            )
            if evaluation.terminal_currents.shape != (law.terminal_count,) or (
                evaluation.auxiliary_residual.shape != (stop - start,)
            ):
                raise ValueError("Circuit element law returned incompatible shapes.")
            for node, current in zip(
                instance.nodes, evaluation.terminal_currents, strict=True
            ):
                index = self.layout.node_index(node)
                if index is not None:
                    residual = residual.at[index].add(current)
            residual = residual.at[start:stop].set(evaluation.auxiliary_residual)
        if residual.shape != (
            node_count
            + sum(stop - start for start, stop in self.layout.auxiliary_ranges),
        ):
            raise ValueError("Circuit DAE residual layout is inconsistent.")
        return residual


def _terminal_values(
    nodes: Sequence[NodeId],
    ground: NodeId | None,
    layout: CircuitStateLayout,
    values: Array,
    /,
) -> Array:
    zero = jnp.asarray(0.0, dtype=values.dtype)
    return jnp.stack(
        tuple(
            zero if node == ground else values[layout.node_ids.index(node)]
            for node in nodes
        )
    )


def _terminal_power(
    circuit: NodalCircuit,
    layout: CircuitStateLayout,
    laws: tuple[AbstractImplicitCircuitLaw, ...],
    time: Array,
    state: Array,
    state_rate: Array,
    args: Any,
    /,
) -> Array:
    inputs = args["inputs"] if isinstance(args, dict) and "inputs" in args else None
    law_args = args["args"] if isinstance(args, dict) and "args" in args else args
    power = jnp.asarray(0.0, dtype=state.dtype)
    for instance, law, (start, stop) in zip(
        circuit.instances, laws, layout.auxiliary_ranges, strict=True
    ):
        voltage = _terminal_values(instance.nodes, circuit.ground, layout, state)
        voltage_rate = _terminal_values(
            instance.nodes, circuit.ground, layout, state_rate
        )
        evaluation = law.evaluate(
            time,
            voltage,
            voltage_rate,
            state[start:stop],
            state_rate[start:stop],
            inputs,
            law_args,
        )
        power = power + jnp.real(jnp.vdot(voltage, evaluation.terminal_currents))
    return power


def plan_circuit_dae(circuit: NodalCircuit, /) -> CircuitDAEPlan:
    if not isinstance(circuit, NodalCircuit):
        raise TypeError("circuit must be NodalCircuit.")
    if circuit.ground is None or circuit.ground not in circuit.nodes:
        raise ValueError("Circuit DAE compilation requires an explicit ground node.")
    node_ids = tuple(node for node in circuit.nodes if node != circuit.ground)
    laws = tuple(implicit_law_for(instance.component) for instance in circuit.instances)
    auxiliary_ranges: list[tuple[int, int]] = []
    cursor = len(node_ids)
    for law in laws:
        start = cursor
        cursor += law.state_layout.size
        auxiliary_ranges.append((start, cursor))
    differential_nodes: set[str] = set()
    for instance, law in zip(circuit.instances, laws, strict=True):
        if law.voltage_rate_dependent:
            differential_nodes.update(
                node for node in instance.nodes if node != circuit.ground
            )
    roles = tuple(
        "differential" if node in differential_nodes else "algebraic" for node in node_ids
    ) + tuple(role for law in laws for role in law.state_layout.roles)
    state_scale = jnp.concatenate(
        (jnp.ones((len(node_ids),)),)
        + tuple(law.state_layout.state_scale for law in laws)
    )
    rate_scale = jnp.concatenate(
        (jnp.ones((len(node_ids),)),) + tuple(law.state_layout.rate_scale for law in laws)
    )
    residual_scale = jnp.concatenate(
        (jnp.ones((len(node_ids),)),)
        + tuple(law.state_layout.residual_scale for law in laws)
    )
    layout_id = canonical_fingerprint(
        {
            "kind": "circuit-state-layout",
            "circuit": circuit.circuit_id,
            "nodes": node_ids,
            "instances": [instance.instance_id for instance in circuit.instances],
            "ranges": auxiliary_ranges,
            "roles": roles,
            "laws": [law.law_id for law in laws],
        }
    )
    layout = CircuitStateLayout(
        node_ids,
        tuple(auxiliary_ranges),
        tuple(instance.instance_id for instance in circuit.instances),
        roles,
        cursor,
        layout_id,
    )
    plan_id = canonical_fingerprint({"kind": "circuit-dae-plan", "layout": layout_id})
    return CircuitDAEPlan(
        circuit, layout, laws, state_scale, rate_scale, residual_scale, plan_id
    )


def prepare_circuit_dae(
    circuit: NodalCircuit,
    plan: CircuitDAEPlan | None = None,
    /,
) -> PreparedCircuitDAE:
    selected = plan_circuit_dae(circuit) if plan is None else plan
    if not isinstance(selected, CircuitDAEPlan):
        raise TypeError("plan must be CircuitDAEPlan or None.")
    if selected.circuit.circuit_id != circuit.circuit_id:
        raise ValueError("Circuit DAE plan belongs to a different circuit.")
    structure = DAEStructure(
        selected.layout.roles,
        equation_roles=selected.layout.roles,
        component_axis=-1,
    )
    system = DifferentialAlgebraicSystem(
        _CircuitResidual(circuit, selected.layout, selected.laws),
        state_shape=(selected.layout.size,),
        structure=structure,
        state_scale=selected.state_scale,
        state_rate_scale=selected.rate_scale,
        residual_scale=selected.residual_scale,
        system_id=f"{circuit.circuit_id}/dae",
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-circuit-dae", "plan": selected.plan_id}
    )
    return PreparedCircuitDAE(selected, system, prepared_id)


def circuit_dae_problem(
    prepared: PreparedCircuitDAE,
    initial_state: ArrayLike,
    /,
    *,
    initial_state_rate: ArrayLike | None = None,
    args: Any = None,
    initialization: Any = None,
):
    if not isinstance(prepared, PreparedCircuitDAE):
        raise TypeError("prepared must be PreparedCircuitDAE.")
    from ..solver import DifferentialAlgebraicProblem

    return DifferentialAlgebraicProblem(
        prepared.system,
        initial_state,
        initial_state_rate=initial_state_rate,
        args=args,
        initialization=initialization,
        problem_id=f"{prepared.plan.circuit.circuit_id}/dae-problem",
    )


def solve_circuit_dae(
    prepared: PreparedCircuitDAE,
    initial_state: ArrayLike,
    time_grid: Any,
    /,
    *,
    initial_state_rate: ArrayLike | None = None,
    args: Any = None,
    initialization: Any = None,
    policy: Any = None,
) -> CircuitDAERunResult:
    from ..solver import solve_dae

    problem = circuit_dae_problem(
        prepared,
        initial_state,
        initial_state_rate=initial_state_rate,
        args=args,
        initialization=initialization,
    )
    solution = solve_dae(problem, time_grid, policy=policy)
    final_state = solution.states[-1]
    final_rate = solution.state_rates[-1]
    final_time = solution.times[-1]
    diagnostics = prepared.diagnostics(final_time, final_state, final_rate, args)
    return CircuitDAERunResult(solution, diagnostics, prepared.prepared_id)


__all__ = [
    "CircuitDAEDiagnostics",
    "CircuitDAEPlan",
    "CircuitDAERunResult",
    "CircuitStateLayout",
    "PreparedCircuitDAE",
    "circuit_dae_problem",
    "plan_circuit_dae",
    "prepare_circuit_dae",
    "solve_circuit_dae",
]
