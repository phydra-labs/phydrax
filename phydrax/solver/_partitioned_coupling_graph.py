#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..nonlinear import FixedPointIteration
from ._partitioned_coupling_types import (
    AbstractCouplingPolicy,
    AbstractCouplingSubsystem,
    CouplingDifferentiationPolicy,
    CouplingExchange,
    CouplingPort,
    CouplingState,
    CouplingSweep,
    ExplicitCouplingPolicy,
    ImplicitCouplingPolicy,
)
from ._partitioned_coupling_waveform import (
    coupling_signal_structure,
    CouplingWaveform,
    flatten_coupling_signal,
    validate_coupling_signal,
)


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _state_bytes(value: Any, /) -> int:
    return sum(
        prod(leaf.shape) * np.dtype(leaf.dtype).itemsize
        for leaf in jax.tree.leaves(value)
    )


def _shape_tree(value: Any, /) -> Any:
    return jax.eval_shape(lambda tree: tree, value)


def _port_payload(port: CouplingPort, /) -> dict[str, Any]:
    return {
        "id": port.port_id,
        "direction": port.direction,
        "space": port.space.space_id,
        "field_space": (
            None if port.field_space is None else port.field_space.field_space_id
        ),
        "reference_scale": port.reference_scale,
        "sample_grid": None if port.sample_grid is None else port.sample_grid.time_id,
        "temporal_interpolation": port.temporal_interpolation,
    }


def _capability_payload(subsystem: AbstractCouplingSubsystem, /) -> dict[str, Any]:
    capabilities = subsystem.capabilities
    return {
        "jit": capabilities.jit,
        "differentiable": capabilities.differentiable,
        "deterministic_replay": capabilities.deterministic_replay,
        "fixed_topology": capabilities.fixed_topology,
        "supports_endpoint": capabilities.supports_endpoint,
        "supports_waveform": capabilities.supports_waveform,
        "counts_complete": capabilities.counts_complete,
    }


class CouplingGraph(StrictModule, NonTrainableState):
    """Finite participant and exchange graph with explicit semantic identity."""

    subsystems: tuple[AbstractCouplingSubsystem, ...]
    exchanges: tuple[CouplingExchange, ...]
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        subsystems: tuple[AbstractCouplingSubsystem, ...],
        exchanges: tuple[CouplingExchange, ...],
        /,
    ):
        subsystems_ = tuple(subsystems)
        exchanges_ = tuple(exchanges)
        if not subsystems_ or any(
            not isinstance(value, AbstractCouplingSubsystem) for value in subsystems_
        ):
            raise TypeError(
                "Coupling graph subsystems must contain AbstractCouplingSubsystem values."
            )
        if not exchanges_ or any(
            not isinstance(value, CouplingExchange) for value in exchanges_
        ):
            raise TypeError(
                "Coupling graph exchanges must contain CouplingExchange values."
            )
        subsystem_ids = tuple(value.subsystem_id for value in subsystems_)
        exchange_ids = tuple(value.exchange_id for value in exchanges_)
        if len(set(subsystem_ids)) != len(subsystem_ids):
            raise ValueError("Coupling graph subsystem IDs must be unique.")
        if len(set(exchange_ids)) != len(exchange_ids):
            raise ValueError("Coupling graph exchange IDs must be unique.")
        port_ids = tuple(
            port.port_id
            for subsystem in subsystems_
            for port in (*subsystem.input_ports, *subsystem.output_ports)
        )
        if len(set(port_ids)) != len(port_ids):
            raise ValueError("Coupling graph port IDs must be globally unique.")
        payload = {
            "kind": "coupling-graph",
            "subsystems": sorted(
                (
                    {
                        "id": subsystem.subsystem_id,
                        "inputs": sorted(
                            (_port_payload(port) for port in subsystem.input_ports),
                            key=lambda item: item["id"],
                        ),
                        "outputs": sorted(
                            (_port_payload(port) for port in subsystem.output_ports),
                            key=lambda item: item["id"],
                        ),
                        "capabilities": _capability_payload(subsystem),
                        "bundle": subsystem.discretization_bundle_id,
                    }
                    for subsystem in subsystems_
                ),
                key=lambda item: item["id"],
            ),
            "exchanges": sorted(
                (
                    {
                        "id": exchange.exchange_id,
                        "source": exchange.source_port_id,
                        "target": exchange.target_port_id,
                        "transfer": (
                            None
                            if exchange.transfer is None
                            else exchange.transfer.transfer_id
                        ),
                        "adjoint": exchange.use_adjoint,
                        "requirement": (
                            None
                            if exchange.requirement is None
                            else exchange.requirement.requirement_id
                        ),
                    }
                    for exchange in exchanges_
                ),
                key=lambda item: item["id"],
            ),
        }
        identifier = canonical_fingerprint(payload)
        self.subsystems = subsystems_
        self.exchanges = exchanges_
        self.graph_id = identifier


class CouplingStagePlan(StrictModule, NonTrainableState):
    """One strongly connected participant stage in condensation-DAG order."""

    subsystem_indices: tuple[int, ...] = eqx.field(static=True)
    internal_exchange_indices: tuple[int, ...] = eqx.field(static=True)
    incoming_exchange_indices: tuple[int, ...] = eqx.field(static=True)
    outgoing_exchange_indices: tuple[int, ...] = eqx.field(static=True)
    cyclic: bool = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        subsystem_indices: tuple[int, ...],
        internal_exchange_indices: tuple[int, ...],
        incoming_exchange_indices: tuple[int, ...],
        outgoing_exchange_indices: tuple[int, ...],
        /,
        *,
        cyclic: bool,
        subsystem_ids: tuple[str, ...],
        exchange_ids: tuple[str, ...],
    ):
        self.subsystem_indices = tuple(subsystem_indices)
        self.internal_exchange_indices = tuple(internal_exchange_indices)
        self.incoming_exchange_indices = tuple(incoming_exchange_indices)
        self.outgoing_exchange_indices = tuple(outgoing_exchange_indices)
        self.cyclic = bool(cyclic)
        self.stage_id = canonical_fingerprint(
            {
                "kind": "coupling-stage",
                "subsystems": [subsystem_ids[index] for index in subsystem_indices],
                "internal": [exchange_ids[index] for index in internal_exchange_indices],
                "incoming": [exchange_ids[index] for index in incoming_exchange_indices],
                "outgoing": [exchange_ids[index] for index in outgoing_exchange_indices],
                "cyclic": bool(cyclic),
            }
        )


class CouplingResourcePolicy(StrictModule, NonTrainableState):
    """Static resource limits checked before coupling execution."""

    maximum_interface_size: int | None = eqx.field(static=True)
    maximum_state_bytes: int | None = eqx.field(static=True)
    maximum_history_bytes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_interface_size: int | None = None,
        maximum_state_bytes: int | None = None,
        maximum_history_bytes: int | None = None,
    ):
        values = (
            maximum_interface_size,
            maximum_state_bytes,
            maximum_history_bytes,
        )
        normalized = tuple(None if value is None else int(value) for value in values)
        if any(value is not None and value < 0 for value in normalized):
            raise ValueError("Coupling resource limits must be non-negative or None.")
        (
            self.maximum_interface_size,
            self.maximum_state_bytes,
            self.maximum_history_bytes,
        ) = normalized


class CouplingResourceEstimate(StrictModule, NonTrainableState):
    interface_size: int = eqx.field(static=True)
    participant_state_bytes: int = eqx.field(static=True)
    exchange_value_bytes: int = eqx.field(static=True)
    nonlinear_history_bytes: int = eqx.field(static=True)
    complete: bool = eqx.field(static=True)


class CouplingPreparationReport(StrictModule, NonTrainableState):
    """Canonical graph, transformation, and resource evidence."""

    stages: tuple[CouplingStagePlan, ...]
    resources: CouplingResourceEstimate
    subsystem_ids: tuple[str, ...] = eqx.field(static=True)
    port_ids: tuple[str, ...] = eqx.field(static=True)
    exchange_ids: tuple[str, ...] = eqx.field(static=True)
    implicit_exchange_ids: tuple[str, ...] = eqx.field(static=True)
    transfer_ids: tuple[str | None, ...] = eqx.field(static=True)
    bundle_ids: tuple[str | None, ...] = eqx.field(static=True)
    jit_eligible: bool = eqx.field(static=True)
    differentiation_eligible: bool = eqx.field(static=True)
    eligibility_reasons: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PreparedCoupling(StrictModule, NonTrainableState):
    """Prepared participant graph with canonical indices and numeric state."""

    subsystems: tuple[AbstractCouplingSubsystem, ...]
    exchanges: tuple[CouplingExchange, ...]
    policy: AbstractCouplingPolicy
    differentiation: CouplingDifferentiationPolicy
    stages: tuple[CouplingStagePlan, ...]
    reference_state: CouplingState
    report: CouplingPreparationReport
    numeric_version: jax.Array
    input_exchange_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    exchange_source_subsystems: tuple[int, ...] = eqx.field(static=True)
    exchange_target_subsystems: tuple[int, ...] = eqx.field(static=True)
    exchange_source_output_indices: tuple[int, ...] = eqx.field(static=True)
    exchange_target_input_indices: tuple[int, ...] = eqx.field(static=True)
    implicit_exchange_indices: tuple[int, ...] = eqx.field(static=True)
    interface_offsets: tuple[int, ...] = eqx.field(static=True)
    interface_sizes: tuple[int, ...] = eqx.field(static=True)
    coordinate_dtype: np.dtype = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _validate_requirement(exchange: CouplingExchange, /) -> None:
    requirement = exchange.requirement
    if requirement is None:
        return
    if exchange.transfer is None:
        return
    properties = exchange.transfer.properties
    missing: list[str] = []
    if requirement.conservative and not properties.conservative:
        missing.append("conservative")
    if requirement.constant_preserving and not properties.constant_preserving:
        missing.append("constant_preserving")
    if requirement.positivity_preserving and not properties.positivity_preserving:
        missing.append("positivity_preserving")
    if requirement.adjoint_paired and not properties.adjoint_paired:
        missing.append("adjoint_paired")
    degree = requirement.minimum_exactness_degree
    if degree is not None:
        exact = set(properties.exact_on)
        if degree == 0:
            exact_enough = properties.constant_preserving or "constants" in exact
        elif degree == 1:
            exact_enough = "coordinate-affine" in exact
        else:
            exact_enough = f"polynomial-degree-{degree}" in exact
        if not exact_enough:
            missing.append(f"exactness-degree-{degree}")
    if missing:
        raise ValueError(
            f"Coupling exchange {exchange.exchange_id!r} transfer lacks required "
            + ", ".join(missing)
            + "."
        )


def _strongly_connected_components(
    adjacency: tuple[tuple[int, ...], ...],
    /,
) -> tuple[tuple[int, ...], ...]:
    count = len(adjacency)
    index = 0
    indices = [-1] * count
    lowlink = [0] * count
    stack: list[int] = []
    on_stack = [False] * count
    components: list[tuple[int, ...]] = []

    def visit(vertex: int) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack[vertex] = True
        for target in adjacency[vertex]:
            if indices[target] < 0:
                visit(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif on_stack[target]:
                lowlink[vertex] = min(lowlink[vertex], indices[target])
        if lowlink[vertex] == indices[vertex]:
            component: list[int] = []
            while True:
                member = stack.pop()
                on_stack[member] = False
                component.append(member)
                if member == vertex:
                    break
            components.append(tuple(sorted(component)))

    for vertex in range(count):
        if indices[vertex] < 0:
            visit(vertex)
    return tuple(components)


def _ordered_stages(
    components: tuple[tuple[int, ...], ...],
    source_subsystems: tuple[int, ...],
    target_subsystems: tuple[int, ...],
    subsystem_ids: tuple[str, ...],
    exchange_ids: tuple[str, ...],
    /,
) -> tuple[CouplingStagePlan, ...]:
    component_of = [0] * len(subsystem_ids)
    for component_index, component in enumerate(components):
        for subsystem_index in component:
            component_of[subsystem_index] = component_index
    dag = [set() for _ in components]
    indegree = [0] * len(components)
    for source, target in zip(source_subsystems, target_subsystems, strict=True):
        source_component = component_of[source]
        target_component = component_of[target]
        if (
            source_component != target_component
            and target_component not in dag[source_component]
        ):
            dag[source_component].add(target_component)
            indegree[target_component] += 1

    def component_key(component_index: int) -> tuple[str, ...]:
        return tuple(subsystem_ids[index] for index in components[component_index])

    ready = sorted(
        (index for index, degree in enumerate(indegree) if degree == 0),
        key=component_key,
    )
    order: list[int] = []
    while ready:
        component_index = ready.pop(0)
        order.append(component_index)
        for target in sorted(dag[component_index], key=component_key):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=component_key)
    if len(order) != len(components):
        raise RuntimeError("Coupling SCC condensation graph must be acyclic.")

    stages: list[CouplingStagePlan] = []
    for component_index in order:
        members = components[component_index]
        member_set = set(members)
        internal = tuple(
            index
            for index, (source, target) in enumerate(
                zip(source_subsystems, target_subsystems, strict=True)
            )
            if source in member_set and target in member_set
        )
        incoming = tuple(
            index
            for index, (source, target) in enumerate(
                zip(source_subsystems, target_subsystems, strict=True)
            )
            if source not in member_set and target in member_set
        )
        outgoing = tuple(
            index
            for index, (source, target) in enumerate(
                zip(source_subsystems, target_subsystems, strict=True)
            )
            if source in member_set and target not in member_set
        )
        cyclic = len(members) > 1 or any(
            source_subsystems[index] == target_subsystems[index] for index in internal
        )
        stages.append(
            CouplingStagePlan(
                members,
                internal,
                incoming,
                outgoing,
                cyclic=cyclic,
                subsystem_ids=subsystem_ids,
                exchange_ids=exchange_ids,
            )
        )
    return tuple(stages)


def _validate_sweep(sweep: CouplingSweep, subsystem_ids: tuple[str, ...], /) -> None:
    if sweep.kind == "gauss-seidel" and set(sweep.subsystem_order) != set(subsystem_ids):
        raise ValueError(
            "Gauss--Seidel coupling order must contain every subsystem exactly once."
        )


def _shape_validate_subsystems(
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    state: CouplingState,
    input_exchange_indices: tuple[tuple[int, ...], ...],
    window_dtype: Any,
    args: Any,
    /,
) -> None:
    from ._partitioned_coupling_types import CouplingSubsystemResult, CouplingWindow

    window = CouplingWindow(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=window_dtype),
        jnp.asarray(1.0, dtype=window_dtype),
    )
    for subsystem_index, subsystem in enumerate(subsystems):
        inputs = tuple(
            state.exchange_values[exchange_index]
            for exchange_index in input_exchange_indices[subsystem_index]
        )
        state_value = state.participant_states[subsystem_index]
        result = jax.eval_shape(
            lambda current_state, current_inputs, current_args, current_subsystem=subsystem: (
                current_subsystem.advance_window(
                    window, current_state, current_inputs, current_args
                )
            ),
            state_value,
            inputs,
            args,
        )
        if not isinstance(result, CouplingSubsystemResult):
            raise TypeError(
                f"Coupling subsystem {subsystem.subsystem_id!r} must return "
                "CouplingSubsystemResult."
            )
        if eqx.tree_equal(result.candidate_state, _shape_tree(state_value)) is not True:
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} candidate state "
                "must preserve the prepared state structure."
            )
        if len(result.outputs) != len(subsystem.output_ports):
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} returned the wrong "
                "number of output ports."
            )
        for port, output in zip(subsystem.output_ports, result.outputs, strict=True):
            shaped_output = output
            if port.sample_grid is not None:
                if not isinstance(output, CouplingWaveform):
                    raise TypeError(
                        f"Coupling subsystem {subsystem.subsystem_id!r} output "
                        f"{port.port_id!r} must be a CouplingWaveform."
                    )
                shaped_output = output.values
            if eqx.tree_equal(shaped_output, coupling_signal_structure(port)) is not True:
                raise ValueError(
                    f"Coupling subsystem {subsystem.subsystem_id!r} output "
                    f"{port.port_id!r} does not match its declared coupling signal."
                )
        scalar_fields = (
            result.successful,
            result.status,
            result.residual_norm,
            result.iterations,
            result.work,
        )
        if any(value.shape != () for value in scalar_fields):
            raise ValueError(
                "Coupling participant evidence fields must be scalar arrays."
            )


def prepare_coupling(
    graph: CouplingGraph,
    participant_states: tuple[Any, ...],
    exchange_values: tuple[Any, ...],
    /,
    *,
    policy: AbstractCouplingPolicy,
    differentiation: CouplingDifferentiationPolicy | None = None,
    time: Any = 0.0,
    args: Any = None,
    problem_id: str = "partitioned-coupling",
    resources: CouplingResourcePolicy | None = None,
) -> PreparedCoupling:
    """Validate and compile one fixed-topology participant graph."""

    if not isinstance(graph, CouplingGraph):
        raise TypeError("graph must be a CouplingGraph.")
    if not isinstance(policy, AbstractCouplingPolicy):
        raise TypeError("policy must be an AbstractCouplingPolicy.")
    differentiation_ = (
        CouplingDifferentiationPolicy() if differentiation is None else differentiation
    )
    if not isinstance(differentiation_, CouplingDifferentiationPolicy):
        raise TypeError("differentiation must be CouplingDifferentiationPolicy or None.")
    resources_ = CouplingResourcePolicy() if resources is None else resources
    if not isinstance(resources_, CouplingResourcePolicy):
        raise TypeError("resources must be CouplingResourcePolicy or None.")
    states = tuple(participant_states)
    values = tuple(exchange_values)
    if len(states) != len(graph.subsystems):
        raise ValueError("One initial participant state is required per graph subsystem.")
    if len(values) != len(graph.exchanges):
        raise ValueError("One initial target value is required per graph exchange.")

    subsystem_declaration_index = {
        subsystem.subsystem_id: index for index, subsystem in enumerate(graph.subsystems)
    }
    exchange_declaration_index = {
        exchange.exchange_id: index for index, exchange in enumerate(graph.exchanges)
    }
    subsystems = tuple(sorted(graph.subsystems, key=lambda value: value.subsystem_id))
    exchanges = tuple(sorted(graph.exchanges, key=lambda value: value.exchange_id))
    canonical_states = tuple(
        states[subsystem_declaration_index[subsystem.subsystem_id]]
        for subsystem in subsystems
    )
    canonical_values = tuple(
        values[exchange_declaration_index[exchange.exchange_id]] for exchange in exchanges
    )
    subsystem_ids = tuple(value.subsystem_id for value in subsystems)
    exchange_ids = tuple(value.exchange_id for value in exchanges)

    if any(not subsystem.capabilities.jit for subsystem in subsystems):
        raise ValueError("Native coupling requires every participant to be JIT-capable.")
    if any(not subsystem.capabilities.fixed_topology for subsystem in subsystems):
        raise ValueError("Native coupling requires fixed-topology participants.")
    for subsystem in subsystems:
        ports_ = (*subsystem.input_ports, *subsystem.output_ports)
        if any(port.sample_grid is None for port in ports_) and not (
            subsystem.capabilities.supports_endpoint
        ):
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} does not support "
                "its endpoint ports."
            )
        if any(port.sample_grid is not None for port in ports_) and not (
            subsystem.capabilities.supports_waveform
        ):
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} does not support "
                "its waveform ports."
            )

    ports: dict[str, tuple[int, int, CouplingPort]] = {}
    port_ids: list[str] = []
    for subsystem_index, subsystem in enumerate(subsystems):
        for local_index, port in enumerate(subsystem.input_ports):
            ports[port.port_id] = (subsystem_index, local_index, port)
            port_ids.append(port.port_id)
        for local_index, port in enumerate(subsystem.output_ports):
            ports[port.port_id] = (subsystem_index, local_index, port)
            port_ids.append(port.port_id)

    source_subsystems: list[int] = []
    target_subsystems: list[int] = []
    source_output_indices: list[int] = []
    target_input_indices: list[int] = []
    input_drivers: dict[str, int] = {}
    incident = [False] * len(subsystems)
    validated_values: list[Any] = []
    for exchange_index, (exchange, initial_value) in enumerate(
        zip(exchanges, canonical_values, strict=True)
    ):
        if exchange.source_port_id not in ports or exchange.target_port_id not in ports:
            raise ValueError(
                f"Coupling exchange {exchange.exchange_id!r} references an unknown port."
            )
        source_subsystem, source_local, source_port = ports[exchange.source_port_id]
        target_subsystem, target_local, target_port = ports[exchange.target_port_id]
        if source_port.direction != "output" or target_port.direction != "input":
            raise ValueError(
                f"Coupling exchange {exchange.exchange_id!r} must connect output to input."
            )
        if target_port.port_id in input_drivers:
            raise ValueError(
                f"Coupling input port {target_port.port_id!r} has multiple drivers."
            )
        input_drivers[target_port.port_id] = exchange_index
        if exchange.transfer is None:
            if source_port.space.space_id != target_port.space.space_id:
                raise ValueError(
                    f"Direct coupling exchange {exchange.exchange_id!r} requires exact "
                    "source and target vector-space identity."
                )
        elif not exchange.use_adjoint:
            if source_port.field_space is None or target_port.field_space is None:
                raise ValueError(
                    "Field transfers require field-valued source and target ports."
                )
            if (
                source_port.field_space.field_space_id
                != exchange.transfer.source.field_space_id
                or target_port.field_space.field_space_id
                != exchange.transfer.target.field_space_id
            ):
                raise ValueError(
                    f"Coupling exchange {exchange.exchange_id!r} field spaces do not "
                    "match its forward transfer."
                )
        else:
            if exchange.transfer.adjoint_operator is None:
                raise ValueError(
                    f"Coupling exchange {exchange.exchange_id!r} requests an unavailable "
                    "adjoint transfer."
                )
            if source_port.field_space is None or target_port.field_space is None:
                raise ValueError("Adjoint transfers require field-valued ports.")
            if (
                source_port.field_space.field_space_id
                != exchange.transfer.target.field_space_id
                or target_port.field_space.field_space_id
                != exchange.transfer.source.field_space_id
            ):
                raise ValueError(
                    f"Coupling exchange {exchange.exchange_id!r} field spaces do not "
                    "match its adjoint transfer."
                )
        _validate_requirement(exchange)
        validated_values.append(validate_coupling_signal(target_port, initial_value))
        source_subsystems.append(source_subsystem)
        target_subsystems.append(target_subsystem)
        source_output_indices.append(source_local)
        target_input_indices.append(target_local)
        incident[source_subsystem] = True
        incident[target_subsystem] = True

    missing_inputs = sorted(
        port.port_id
        for subsystem in subsystems
        for port in subsystem.input_ports
        if port.port_id not in input_drivers
    )
    if missing_inputs:
        raise ValueError(
            "Coupling input ports require exactly one driver: "
            + ", ".join(missing_inputs)
        )
    if not all(incident):
        isolated = [
            subsystem_ids[index]
            for index, connected in enumerate(incident)
            if not connected
        ]
        raise ValueError(
            "Coupling graph contains isolated subsystems: " + ", ".join(isolated)
        )

    input_exchange_indices = tuple(
        tuple(input_drivers[port.port_id] for port in subsystem.input_ports)
        for subsystem in subsystems
    )
    adjacency_sets = [set() for _ in subsystems]
    for source, target in zip(source_subsystems, target_subsystems, strict=True):
        adjacency_sets[source].add(target)
    adjacency = tuple(tuple(sorted(targets)) for targets in adjacency_sets)
    components = _strongly_connected_components(adjacency)
    stages = _ordered_stages(
        components,
        tuple(source_subsystems),
        tuple(target_subsystems),
        subsystem_ids,
        exchange_ids,
    )
    implicit_exchange_indices = tuple(
        exchange_index
        for stage in stages
        if stage.cyclic
        for exchange_index in stage.internal_exchange_indices
    )

    if isinstance(policy, ExplicitCouplingPolicy):
        _validate_sweep(policy.sweep, subsystem_ids)
    elif isinstance(policy, ImplicitCouplingPolicy):
        if not implicit_exchange_indices:
            raise ValueError(
                "Implicit coupling requires at least one cyclic participant stage."
            )
        if isinstance(policy.method, FixedPointIteration):
            sweep = policy.fixed_point_sweep
            if sweep is None:
                raise RuntimeError("Prepared fixed-point coupling sweep is missing.")
            _validate_sweep(sweep, subsystem_ids)
        cyclic_target_ports = {
            exchanges[index].target_port_id for index in implicit_exchange_indices
        }
        tolerance_ports = {value.port_id for value in policy.tolerances}
        if cyclic_target_ports != tolerance_ports:
            missing = sorted(cyclic_target_ports - tolerance_ports)
            extra = sorted(tolerance_ports - cyclic_target_ports)
            raise ValueError(
                "Implicit coupling tolerances must exactly cover cyclic target ports; "
                f"missing={missing}, extra={extra}."
            )
        cyclic_subsystems = {
            subsystem_index
            for stage in stages
            if stage.cyclic
            for subsystem_index in stage.subsystem_indices
        }
        if any(
            not subsystems[index].capabilities.deterministic_replay
            for index in cyclic_subsystems
        ):
            raise ValueError(
                "Implicit coupling requires deterministic replay for every cyclic participant."
            )
    else:
        raise TypeError("Unsupported coupling policy type.")

    if differentiation_.mode == "algorithmic" and not isinstance(
        policy, ExplicitCouplingPolicy
    ):
        raise ValueError("Algorithmic coupling differentiation is explicit-only.")
    if differentiation_.mode == "implicit":
        if not isinstance(policy, ImplicitCouplingPolicy) or isinstance(
            policy.method, FixedPointIteration
        ):
            raise ValueError(
                "Implicit differentiation requires a general-root implicit policy."
            )
        if any(not subsystem.capabilities.differentiable for subsystem in subsystems):
            raise ValueError(
                "Implicit differentiation requires differentiable participants."
            )
        for exchange in exchanges:
            if (
                exchange.transfer is not None
                and not exchange.transfer.properties.differentiable_geometry
            ):
                raise ValueError(
                    "Implicit differentiation requires differentiable exchange geometry."
                )

    initial_state = CouplingState(
        canonical_states,
        tuple(validated_values),
        time,
        0,
        subsystem_ids=subsystem_ids,
        exchange_ids=exchange_ids,
    )
    time_dtype = initial_state.time.dtype
    _shape_validate_subsystems(
        subsystems, initial_state, input_exchange_indices, time_dtype, args
    )

    interface_offsets: list[int] = []
    interface_sizes: list[int] = []
    offset = 0
    coordinate_dtypes: list[np.dtype] = []
    for exchange_index in implicit_exchange_indices:
        target_port = ports[exchanges[exchange_index].target_port_id][2]
        flattened = flatten_coupling_signal(target_port, validated_values[exchange_index])
        size = int(flattened.size)
        if size <= 0:
            raise ValueError("Implicit coupling interface spaces must be non-empty.")
        interface_offsets.append(offset)
        interface_sizes.append(size)
        offset += size
        coordinate_dtypes.append(np.dtype(flattened.dtype))
    coordinate_dtype = (
        np.dtype(jnp.asarray(0.0).dtype)
        if not coordinate_dtypes
        else np.dtype(jnp.result_type(*coordinate_dtypes))
    )

    participant_state_bytes = sum(_state_bytes(value) for value in canonical_states)
    exchange_value_bytes = sum(_state_bytes(value) for value in validated_values)
    history_bytes = 0
    history_complete = True
    if isinstance(policy, ImplicitCouplingPolicy):
        if isinstance(policy.method, FixedPointIteration):
            acceleration = policy.method.acceleration
            if acceleration is not None:
                history_bytes = (
                    2 * (acceleration.history + 1) * offset * coordinate_dtype.itemsize
                )
        else:
            history_complete = False
    estimate = CouplingResourceEstimate(
        interface_size=offset,
        participant_state_bytes=participant_state_bytes,
        exchange_value_bytes=exchange_value_bytes,
        nonlinear_history_bytes=history_bytes,
        complete=history_complete
        and all(subsystem.capabilities.counts_complete for subsystem in subsystems),
    )
    if (
        resources_.maximum_interface_size is not None
        and estimate.interface_size > resources_.maximum_interface_size
    ):
        raise MemoryError("Coupling interface size exceeds its resource policy.")
    if (
        resources_.maximum_state_bytes is not None
        and participant_state_bytes + exchange_value_bytes
        > resources_.maximum_state_bytes
    ):
        raise MemoryError("Coupling retained state exceeds its resource policy.")
    if (
        resources_.maximum_history_bytes is not None
        and history_bytes > resources_.maximum_history_bytes
    ):
        raise MemoryError("Coupling nonlinear history exceeds its resource policy.")

    reasons: list[str] = []
    differentiable = all(
        subsystem.capabilities.differentiable for subsystem in subsystems
    )
    if not differentiable:
        reasons.append("one or more participants are nondifferentiable")
    fixed_point_without_derivative = isinstance(
        policy, ImplicitCouplingPolicy
    ) and isinstance(policy.method, FixedPointIteration)
    if fixed_point_without_derivative:
        reasons.append("fixed-point coupling has no implicit derivative contract")
    transfer_ids = tuple(
        None if exchange.transfer is None else exchange.transfer.transfer_id
        for exchange in exchanges
    )
    bundle_ids = tuple(subsystem.discretization_bundle_id for subsystem in subsystems)
    report_id = canonical_fingerprint(
        {
            "kind": "coupling-preparation-report",
            "graph": graph.graph_id,
            "policy": policy.policy_id,
            "differentiation": differentiation_.policy_id,
            "stages": [stage.stage_id for stage in stages],
            "subsystems": list(subsystem_ids),
            "ports": sorted(port_ids),
            "exchanges": list(exchange_ids),
            "implicit_exchanges": [
                exchange_ids[index] for index in implicit_exchange_indices
            ],
            "transfers": list(transfer_ids),
            "bundles": list(bundle_ids),
            "interface_size": offset,
        }
    )
    report = CouplingPreparationReport(
        stages=stages,
        resources=estimate,
        subsystem_ids=subsystem_ids,
        port_ids=tuple(sorted(port_ids)),
        exchange_ids=exchange_ids,
        implicit_exchange_ids=tuple(
            exchange_ids[index] for index in implicit_exchange_indices
        ),
        transfer_ids=transfer_ids,
        bundle_ids=bundle_ids,
        jit_eligible=True,
        differentiation_eligible=differentiable and not fixed_point_without_derivative,
        eligibility_reasons=tuple(reasons),
        report_id=report_id,
    )
    problem_id_ = _identifier(problem_id, "Coupling problem_id")
    plan_id = canonical_fingerprint(
        {
            "kind": "prepared-coupling",
            "problem": problem_id_,
            "graph": graph.graph_id,
            "policy": policy.policy_id,
            "differentiation": differentiation_.policy_id,
            "report": report_id,
            "state": [array_tree_signature(value) for value in canonical_states],
            "exchange_values": [
                array_tree_signature(value) for value in validated_values
            ],
        }
    )
    return PreparedCoupling(
        subsystems=subsystems,
        exchanges=exchanges,
        policy=policy,
        differentiation=differentiation_,
        stages=stages,
        reference_state=initial_state,
        report=report,
        numeric_version=jnp.asarray(0, dtype=jnp.int32),
        input_exchange_indices=input_exchange_indices,
        exchange_source_subsystems=tuple(source_subsystems),
        exchange_target_subsystems=tuple(target_subsystems),
        exchange_source_output_indices=tuple(source_output_indices),
        exchange_target_input_indices=tuple(target_input_indices),
        implicit_exchange_indices=implicit_exchange_indices,
        interface_offsets=tuple(interface_offsets),
        interface_sizes=tuple(interface_sizes),
        coordinate_dtype=coordinate_dtype,
        graph_id=graph.graph_id,
        problem_id=problem_id_,
        plan_id=plan_id,
    )


def refresh_coupling(
    prepared: PreparedCoupling,
    graph: CouplingGraph,
    /,
    *,
    args: Any = None,
) -> PreparedCoupling:
    """Refresh numeric participant/transfer leaves without changing structure."""

    if not isinstance(prepared, PreparedCoupling):
        raise TypeError("prepared must be PreparedCoupling.")
    if not isinstance(graph, CouplingGraph):
        raise TypeError("graph must be CouplingGraph.")
    if graph.graph_id != prepared.graph_id:
        raise ValueError("Coupling refresh requires unchanged structural graph identity.")
    subsystem_by_id = {value.subsystem_id: value for value in graph.subsystems}
    exchange_by_id = {value.exchange_id: value for value in graph.exchanges}
    subsystems = tuple(
        subsystem_by_id[subsystem_id] for subsystem_id in prepared.report.subsystem_ids
    )
    exchanges = tuple(
        exchange_by_id[exchange_id] for exchange_id in prepared.report.exchange_ids
    )
    _shape_validate_subsystems(
        subsystems,
        prepared.reference_state,
        prepared.input_exchange_indices,
        prepared.reference_state.time.dtype,
        args,
    )
    return PreparedCoupling(
        subsystems=subsystems,
        exchanges=exchanges,
        policy=prepared.policy,
        differentiation=prepared.differentiation,
        stages=prepared.stages,
        reference_state=prepared.reference_state,
        report=prepared.report,
        numeric_version=prepared.numeric_version + 1,
        input_exchange_indices=prepared.input_exchange_indices,
        exchange_source_subsystems=prepared.exchange_source_subsystems,
        exchange_target_subsystems=prepared.exchange_target_subsystems,
        exchange_source_output_indices=prepared.exchange_source_output_indices,
        exchange_target_input_indices=prepared.exchange_target_input_indices,
        implicit_exchange_indices=prepared.implicit_exchange_indices,
        interface_offsets=prepared.interface_offsets,
        interface_sizes=prepared.interface_sizes,
        coordinate_dtype=prepared.coordinate_dtype,
        graph_id=prepared.graph_id,
        problem_id=prepared.problem_id,
        plan_id=prepared.plan_id,
    )


__all__ = [
    "CouplingGraph",
    "CouplingPreparationReport",
    "CouplingResourceEstimate",
    "CouplingResourcePolicy",
    "CouplingStagePlan",
    "PreparedCoupling",
    "prepare_coupling",
    "refresh_coupling",
]
