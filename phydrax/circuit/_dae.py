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
from ..dynamics._layout import InputLayout
from ..dynamics._system import AbstractInputPolicy
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


class CircuitInputBinding(StrictModule):
    instance_id: str = eqx.field(static=True)
    input_names: tuple[str, ...] = eqx.field(static=True)
    indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        instance_id: str,
        input_names: Sequence[str],
        indices: Sequence[int],
        /,
    ):
        identifier = str(instance_id)
        names = tuple(str(name) for name in input_names)
        bound = tuple(int(index) for index in indices)
        if not identifier:
            raise ValueError("Circuit input binding instance_id must be non-empty.")
        if len(names) != len(bound):
            raise ValueError("Circuit input binding names and indices must align.")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Circuit law input names must be unique and non-empty.")
        if any(index < 0 for index in bound):
            raise ValueError("Circuit input binding indices must be nonnegative.")
        self.instance_id = identifier
        self.input_names = names
        self.indices = bound

    def select(self, inputs: Array | None, dtype, /) -> Array:
        if inputs is None:
            if self.indices:
                raise ValueError(
                    f"Circuit law {self.instance_id!r} requires explicit inputs."
                )
            return jnp.zeros((0,), dtype=dtype)
        return jnp.take(inputs, jnp.asarray(self.indices, dtype=int), axis=0)


class CircuitDAEPlan(StrictModule):
    circuit: NodalCircuit
    layout: CircuitStateLayout
    laws: tuple[AbstractImplicitCircuitLaw, ...]
    input_layout: InputLayout | None
    input_bindings: tuple[CircuitInputBinding, ...]
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
        *,
        inputs: ArrayLike | None = None,
    ) -> CircuitDAEDiagnostics:
        value = jnp.asarray(state)
        rate = jnp.asarray(state_rate)
        residual = self.system.evaluate(time, value, rate, args, inputs=inputs)
        node_count = len(self.plan.layout.node_ids)
        element_norms = tuple(
            jnp.linalg.norm(residual[start:stop])
            for start, stop in self.plan.layout.auxiliary_ranges
        )
        terminal_power = _terminal_power(
            self.plan.circuit,
            self.plan.layout,
            self.plan.laws,
            self.plan.input_bindings,
            jnp.asarray(time),
            value,
            rate,
            None if inputs is None else jnp.asarray(inputs),
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


class _CircuitResidualCore(StrictModule):
    circuit: NodalCircuit
    layout: CircuitStateLayout
    laws: tuple[AbstractImplicitCircuitLaw, ...]
    input_bindings: tuple[CircuitInputBinding, ...]

    def evaluate(
        self,
        time: Array,
        state: Array,
        state_rate: Array,
        inputs: Array | None,
        args: Any,
        /,
    ) -> Array:
        node_count = len(self.layout.node_ids)
        residual = jnp.zeros(
            (self.layout.size,), dtype=jnp.result_type(state, state_rate)
        )
        for instance, law, binding, (start, stop) in zip(
            self.circuit.instances,
            self.laws,
            self.input_bindings,
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
                binding.select(inputs, state.dtype),
                args,
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


class _AutonomousCircuitResidual(StrictModule):
    core: _CircuitResidualCore

    def __call__(
        self, time: Array, state: Array, state_rate: Array, args: Any, /
    ) -> Array:
        return self.core.evaluate(time, state, state_rate, None, args)


class _InputCircuitResidual(StrictModule):
    core: _CircuitResidualCore

    def __call__(
        self,
        time: Array,
        state: Array,
        state_rate: Array,
        inputs: Array,
        args: Any,
        /,
    ) -> Array:
        return self.core.evaluate(time, state, state_rate, inputs, args)


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
    input_bindings: tuple[CircuitInputBinding, ...],
    time: Array,
    state: Array,
    state_rate: Array,
    inputs: Array | None,
    args: Any,
    /,
) -> Array:
    power = jnp.asarray(0.0, dtype=state.dtype)
    for instance, law, binding, (start, stop) in zip(
        circuit.instances,
        laws,
        input_bindings,
        layout.auxiliary_ranges,
        strict=True,
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
            binding.select(inputs, state.dtype),
            args,
        )
        power = power + jnp.real(jnp.vdot(voltage, evaluation.terminal_currents))
    return power


def _circuit_input_topology(
    circuit: NodalCircuit,
    laws: tuple[AbstractImplicitCircuitLaw, ...],
    /,
) -> tuple[InputLayout | None, tuple[CircuitInputBinding, ...]]:
    declared = []
    for instance, law in zip(circuit.instances, laws, strict=True):
        names = tuple(law.input_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(
                f"Circuit law {instance.instance_id!r} input names must be non-empty strings."
            )
        if len(set(names)) != len(names):
            raise ValueError(
                f"Circuit law {instance.instance_id!r} has conflicting input names."
            )
        declared.extend(names)
    global_names = tuple(sorted(set(declared)))
    input_layout = (
        None
        if not global_names
        else InputLayout(
            (len(global_names),),
            axes=("circuit_input",),
            component_names=global_names,
            roles="forcing",
        )
    )
    bindings = tuple(
        CircuitInputBinding(
            instance.instance_id,
            law.input_names,
            tuple(global_names.index(name) for name in law.input_names),
        )
        for instance, law in zip(circuit.instances, laws, strict=True)
    )
    return input_layout, bindings


def _validate_scale_vector(value: Array, size: int, owner: str, /) -> None:
    if value.shape != (size,):
        raise ValueError(f"{owner} must have circuit state shape {(size,)}.")
    if jnp.issubdtype(value.dtype, jnp.complexfloating) or bool(
        jnp.any(~jnp.isfinite(value)) | jnp.any(value <= 0.0)
    ):
        raise ValueError(f"{owner} must be a finite positive real vector.")


def _validate_circuit_plan(circuit: NodalCircuit, plan: CircuitDAEPlan, /) -> None:
    if plan.circuit.circuit_id != circuit.circuit_id:
        raise ValueError("Circuit DAE plan belongs to a different circuit.")
    instance_ids = tuple(instance.instance_id for instance in circuit.instances)
    if plan.layout.instance_ids != instance_ids:
        raise ValueError("Circuit DAE plan instance order conflicts with the circuit.")
    if len(plan.laws) != len(circuit.instances):
        raise ValueError("Circuit DAE plan law count conflicts with the circuit.")
    expected_laws = tuple(
        implicit_law_for(instance.component) for instance in circuit.instances
    )
    if tuple((type(law), law.law_id) for law in plan.laws) != tuple(
        (type(law), law.law_id) for law in expected_laws
    ):
        raise ValueError("Circuit DAE plan laws conflict with circuit instances.")
    if any(
        law.terminal_count != len(instance.nodes)
        for instance, law in zip(circuit.instances, plan.laws, strict=True)
    ):
        raise ValueError("Circuit law terminal counts conflict with circuit topology.")
    expected_ranges = []
    cursor = len(plan.layout.node_ids)
    for law in plan.laws:
        expected_ranges.append((cursor, cursor + law.state_layout.size))
        cursor += law.state_layout.size
    differential_nodes = {
        node
        for instance, law in zip(circuit.instances, plan.laws, strict=True)
        if law.voltage_rate_dependent
        for node in instance.nodes
        if node != circuit.ground
    }
    expected_roles = tuple(
        "differential" if node in differential_nodes else "algebraic"
        for node in plan.layout.node_ids
    ) + tuple(role for law in plan.laws for role in law.state_layout.roles)
    if (
        plan.layout.node_ids
        != tuple(node for node in circuit.nodes if node != circuit.ground)
        or plan.layout.auxiliary_ranges != tuple(expected_ranges)
        or plan.layout.size != cursor
        or plan.layout.roles != expected_roles
    ):
        raise ValueError("Circuit DAE state layout conflicts with circuit topology.")
    input_layout, bindings = _circuit_input_topology(circuit, plan.laws)
    if (input_layout is None) != (plan.input_layout is None):
        raise ValueError("Circuit DAE input layout conflicts with declared law inputs.")
    if (
        input_layout is not None
        and plan.input_layout is not None
        and input_layout.layout_id != plan.input_layout.layout_id
    ):
        raise ValueError(
            "Circuit DAE input shape or names conflict with declared law inputs."
        )
    if tuple(
        (value.instance_id, value.input_names, value.indices)
        for value in plan.input_bindings
    ) != tuple(
        (value.instance_id, value.input_names, value.indices) for value in bindings
    ):
        raise ValueError("Circuit DAE per-law input bindings are inconsistent.")
    _validate_scale_vector(plan.state_scale, cursor, "state_scale")
    _validate_scale_vector(plan.rate_scale, cursor, "rate_scale")
    _validate_scale_vector(plan.residual_scale, cursor, "residual_scale")


def _validate_input_policy(
    prepared: PreparedCircuitDAE,
    input_policy: AbstractInputPolicy | None,
    /,
) -> None:
    if prepared.system.input_layout is None:
        if input_policy is not None:
            raise ValueError("An autonomous circuit does not accept input_policy.")
        return
    if input_policy is None:
        raise ValueError("An input-driven circuit requires input_policy.")
    if not isinstance(input_policy, AbstractInputPolicy):
        raise TypeError("input_policy must be an AbstractInputPolicy or None.")
    if input_policy.input_layout.layout_id != prepared.system.input_layout.layout_id:
        raise ValueError(
            "input_policy layout must exactly match the circuit input layout."
        )


def plan_circuit_dae(circuit: NodalCircuit, /) -> CircuitDAEPlan:
    if not isinstance(circuit, NodalCircuit):
        raise TypeError("circuit must be NodalCircuit.")
    if circuit.ground is None or circuit.ground not in circuit.nodes:
        raise ValueError("Circuit DAE compilation requires an explicit ground node.")
    node_ids = tuple(node for node in circuit.nodes if node != circuit.ground)
    laws = tuple(implicit_law_for(instance.component) for instance in circuit.instances)
    input_layout, input_bindings = _circuit_input_topology(circuit, laws)
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
    plan_id = canonical_fingerprint(
        {
            "kind": "circuit-dae-plan",
            "layout": layout_id,
            "input_layout": (None if input_layout is None else input_layout.layout_id),
            "input_bindings": [
                {
                    "instance": binding.instance_id,
                    "names": binding.input_names,
                    "indices": binding.indices,
                }
                for binding in input_bindings
            ],
        }
    )
    return CircuitDAEPlan(
        circuit,
        layout,
        laws,
        input_layout,
        input_bindings,
        state_scale,
        rate_scale,
        residual_scale,
        plan_id,
    )


def prepare_circuit_dae(
    circuit: NodalCircuit,
    plan: CircuitDAEPlan | None = None,
    /,
) -> PreparedCircuitDAE:
    selected = plan_circuit_dae(circuit) if plan is None else plan
    if not isinstance(selected, CircuitDAEPlan):
        raise TypeError("plan must be CircuitDAEPlan or None.")
    _validate_circuit_plan(circuit, selected)
    structure = DAEStructure(
        selected.layout.roles,
        equation_roles=selected.layout.roles,
        component_axis=-1,
    )
    residual_core = _CircuitResidualCore(
        circuit,
        selected.layout,
        selected.laws,
        selected.input_bindings,
    )
    residual = (
        _AutonomousCircuitResidual(residual_core)
        if selected.input_layout is None
        else _InputCircuitResidual(residual_core)
    )
    system = DifferentialAlgebraicSystem(
        residual,
        state_shape=(selected.layout.size,),
        structure=structure,
        input_layout=selected.input_layout,
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
    input_policy: AbstractInputPolicy | None = None,
    initialization: Any = None,
):
    if not isinstance(prepared, PreparedCircuitDAE):
        raise TypeError("prepared must be PreparedCircuitDAE.")
    _validate_input_policy(prepared, input_policy)
    from ..solver import DifferentialAlgebraicProblem

    return DifferentialAlgebraicProblem(
        prepared.system,
        initial_state,
        initial_state_rate=initial_state_rate,
        args=args,
        input_policy=input_policy,
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
    input_policy: AbstractInputPolicy | None = None,
    initialization: Any = None,
    policy: Any = None,
) -> CircuitDAERunResult:
    from ..solver import solve_dae

    problem = circuit_dae_problem(
        prepared,
        initial_state,
        initial_state_rate=initial_state_rate,
        args=args,
        input_policy=input_policy,
        initialization=initialization,
    )
    solution = solve_dae(problem, time_grid, policy=policy)
    final_state = solution.states[-1]
    final_rate = solution.state_rates[-1]
    final_time = solution.times[-1]
    final_inputs = (
        None
        if input_policy is None
        else input_policy.evaluate(final_time, final_state, args)
    )
    diagnostics = prepared.diagnostics(
        final_time, final_state, final_rate, args, inputs=final_inputs
    )
    return CircuitDAERunResult(solution, diagnostics, prepared.prepared_id)


__all__ = [
    "CircuitDAEDiagnostics",
    "CircuitInputBinding",
    "CircuitDAEPlan",
    "CircuitDAERunResult",
    "CircuitStateLayout",
    "PreparedCircuitDAE",
    "circuit_dae_problem",
    "plan_circuit_dae",
    "prepare_circuit_dae",
    "solve_circuit_dae",
]
