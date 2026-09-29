#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
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
from ..units import TIME, UnitDefinition
from ._partitioned_coupling_measurement import CouplingMeasurement
from ._partitioned_coupling_types import (
    AbstractCouplingPolicy,
    AbstractCouplingSubsystem,
    CouplingDifferentiationPolicy,
    CouplingExchange,
    CouplingPort,
    CouplingQuantity,
    CouplingState,
    CouplingSweep,
    ExplicitCouplingPolicy,
    ImplicitCouplingPolicy,
)
from ._partitioned_coupling_waveform import (
    coupling_signal_structure,
    CouplingTemporalConversionKind,
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
        "quantity": None if port.quantity is None else port.quantity.to_dict(),
        "measurement": (
            None if port.measurement is None else port.measurement.measurement_id
        ),
        "temporal_kind": port.temporal_kind,
        "frame": port.frame,
        "waveform_plan": (
            None if port.waveform_plan is None else port.waveform_plan.plan_id
        ),
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


def _subsystem_payload(subsystem: AbstractCouplingSubsystem, /) -> dict[str, Any]:
    return {
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


def _integrated(exchange: CouplingExchange, /) -> bool:
    return exchange.temporal is not None and exchange.temporal.kind == "integrate"


def _exchange_payload(exchange: CouplingExchange, /) -> dict[str, Any]:
    return {
        "id": exchange.exchange_id,
        "source": exchange.source_port_id,
        "target": exchange.target_port_id,
        "transfer": (
            None if exchange.transfer is None else exchange.transfer.transfer_id
        ),
        "adjoint": exchange.use_adjoint,
        "requirement": (
            None if exchange.requirement is None else exchange.requirement.requirement_id
        ),
        "temporal": (
            None if exchange.temporal is None else exchange.temporal.conversion_id
        ),
    }


class CouplingGraph(StrictModule, NonTrainableState):
    """Finite participant and exchange graph with explicit semantic identity.

    `time_unit` is the unit of the one coupling clock: window times and sizes
    are in it. It is required whenever a waveform `"integrate"` conversion books
    a rate times the window size, and it is part of the graph identity.
    """

    subsystems: tuple[AbstractCouplingSubsystem, ...]
    exchanges: tuple[CouplingExchange, ...]
    time_unit: UnitDefinition | None
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        subsystems: tuple[AbstractCouplingSubsystem, ...],
        exchanges: tuple[CouplingExchange, ...],
        /,
        *,
        time_unit: UnitDefinition | None = None,
    ) -> None:
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
        if time_unit is not None:
            if not isinstance(time_unit, UnitDefinition):
                raise TypeError(
                    "Coupling graph time_unit must be UnitDefinition or None."
                )
            if time_unit.dimension != TIME:
                raise ValueError("Coupling graph time_unit must be a unit of time.")
        if time_unit is None and any(_integrated(value) for value in exchanges_):
            raise ValueError(
                "Waveform integration multiplies by the window size; declare the "
                "coupling clock time_unit on the graph."
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
                (_subsystem_payload(subsystem) for subsystem in subsystems_),
                key=lambda item: item["id"],
            ),
            "exchanges": sorted(
                (_exchange_payload(exchange) for exchange in exchanges_),
                key=lambda item: item["id"],
            ),
            "time_unit": None if time_unit is None else time_unit.unit_id,
        }
        identifier = canonical_fingerprint(payload)
        self.subsystems = subsystems_
        self.exchanges = exchanges_
        self.time_unit = time_unit
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
    ) -> None:
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
    ) -> None:
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
    """Prepared participant graph with canonical indices and numeric state.

    `time_unit` is the graph's declared coupling clock unit.
    """

    subsystems: tuple[AbstractCouplingSubsystem, ...]
    exchanges: tuple[CouplingExchange, ...]
    policy: AbstractCouplingPolicy
    differentiation: CouplingDifferentiationPolicy
    stages: tuple[CouplingStagePlan, ...]
    reference_state: CouplingState
    report: CouplingPreparationReport
    numeric_version: jax.Array
    time_unit: UnitDefinition | None
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


def _required_conversion(
    source: CouplingPort, target: CouplingPort, /
) -> CouplingTemporalConversionKind | None:
    """Return the one temporal conversion meaning an exchange must declare."""
    source_waveform = source.waveform_plan is not None
    target_waveform = target.waveform_plan is not None
    source_amount = source.temporal_kind == "interval_integral"
    target_amount = target.temporal_kind == "interval_integral"
    if source_waveform and target_waveform:
        return "interpolate"
    if source_waveform:
        return "integrate" if target_amount else "sample-end"
    if target_waveform:
        if source_amount:
            raise ValueError("A whole-window amount has no waveform history to hold.")
        return "hold"
    if source_amount != target_amount:
        raise ValueError(
            "Instantaneous values and interval integrals cannot be exchanged; an "
            "endpoint value is never multiplied by the window as an exact integral."
        )
    return "window-integral" if target_amount else None


def _validate_temporal_exchange(
    exchange: CouplingExchange, source: CouplingPort, target: CouplingPort, /
) -> None:
    required = _required_conversion(source, target)
    declared = None if exchange.temporal is None else exchange.temporal.kind
    if declared != required:
        raise ValueError(
            f"Coupling exchange {exchange.exchange_id!r} requires temporal conversion "
            f"{required!r} but declares {declared!r}; temporal meaning is never inferred."
        )


def _received_quantity(
    exchange: CouplingExchange,
    source: CouplingPort,
    time_unit: UnitDefinition | None,
    /,
) -> CouplingQuantity | None:
    """Physical quantity delivered after the declared temporal conversion."""
    conversion = exchange.temporal
    if conversion is None or conversion.kind != "integrate":
        return source.quantity
    integrated = conversion.integrated_quantity
    if integrated is None or time_unit is None:
        raise RuntimeError("Prepared waveform integration lacks its declared semantics.")
    rate = source.quantity
    if rate is None:
        raise ValueError("Waveform integration requires a typed source rate quantity.")
    systems = {
        rate.unit.reference_system_id,
        time_unit.reference_system_id,
        integrated.unit.reference_system_id,
    }
    if (
        integrated.unit.dimension != rate.unit.dimension * time_unit.dimension
        or len(systems) != 1
    ):
        raise ValueError(
            "The integrated quantity must have the source rate dimension times time."
        )
    return integrated


def _validate_inventory_semantics(
    received: CouplingQuantity,
    source: CouplingPort,
    target: CouplingPort,
    target_quantity: CouplingQuantity,
    /,
) -> None:
    """Require one physical quantity whose inventories share dimensions.

    Storage may change between a density and an extensive representation, for
    example J/m² cell averages with an area measurement and J cell integrals with
    a counting measurement, only through a certified conservative transfer; the
    quantity kind, reference configuration, sign, reference system, and the
    inventory dimension `quantity × measurement` must agree. Unmeasured ports
    require exactly equal quantity dimensions.
    """
    if (
        received.quantity_kind != target_quantity.quantity_kind
        or received.reference_configuration != target_quantity.reference_configuration
        or received.sign_convention != target_quantity.sign_convention
        or received.unit.reference_system_id != target_quantity.unit.reference_system_id
    ):
        raise ValueError("Coupling quantities have incompatible physical semantics.")
    source_measurement = source.measurement
    target_measurement = target.measurement
    if source_measurement is None or target_measurement is None:
        if received.compatibility_id != target_quantity.compatibility_id:
            raise ValueError("Coupling quantities have incompatible physical semantics.")
        return
    source_unit = source_measurement.unit
    target_unit = target_measurement.unit
    if (
        received.unit.dimension * source_unit.dimension
        != target_quantity.unit.dimension * target_unit.dimension
        or source_unit.reference_system_id != target_unit.reference_system_id
    ):
        raise ValueError(
            "Coupling measurements require compatible inventory dimensions and "
            "reference systems."
        )
    if source_measurement.component_ids != target_measurement.component_ids:
        raise ValueError("Coupling measurements must inventory the same components.")


def _validate_direct_physical(source: CouplingPort, target: CouplingPort, /) -> None:
    if source.frame != target.frame:
        raise ValueError("Changing component frames requires an explicit FieldTransfer.")
    source_field_space = source.field_space
    target_field_space = target.field_space
    if (source_field_space is None) != (target_field_space is None) or (
        source_field_space is not None
        and target_field_space is not None
        and source_field_space.field_space_id != target_field_space.field_space_id
    ):
        raise ValueError("Different physical storage requires an explicit FieldTransfer.")
    source_measurement = source.measurement
    target_measurement = target.measurement
    if (source_measurement is None) != (target_measurement is None) or (
        source_measurement is not None
        and target_measurement is not None
        and source_measurement.measurement_id != target_measurement.measurement_id
    ):
        raise ValueError(
            "Direct physical exchange requires exact measurement-functional identity."
        )


def _certify_conservative_transfer(
    exchange: CouplingExchange,
    source: CouplingMeasurement,
    target: CouplingMeasurement,
    /,
) -> None:
    """Certify `L_target P = L_source` per component with transposed actions.

    Each inventory covector of the target is pulled back through the transpose of
    the transfer action actually applied, so the certificate needs one transposed
    action per component and never a dense transfer matrix. Quantity scales
    cancel against the runtime unit conversion; measurement scales remain.
    """
    transfer = exchange.transfer
    if transfer is None:
        raise RuntimeError("A conservative certificate requires a prepared transfer.")
    operator = (
        transfer.hilbert_adjoint_operator
        if exchange.use_adjoint
        else transfer.primal_operator
    )
    if operator is None:
        raise RuntimeError("Prepared coupling transfer action is unavailable.")
    if not operator.capabilities.transpose:
        raise ValueError(
            "A conservative exchange requires the transposed transfer action."
        )
    source_scale = float(source.unit.scale_to_reference)
    target_scale = float(target.unit.scale_to_reference)
    dtype = target.inventory_dtype
    count = target.component_count
    for component in range(count):
        basis = jnp.zeros((count,), dtype=dtype).at[component].set(1)
        pulled = np.asarray(
            operator.source.flatten(operator.transpose_mv(target.covector(basis)))
        )
        expected = np.asarray(source.source_space.flatten(source.covector(basis)))
        tolerance = 64 * np.finfo(expected.dtype).eps
        if not np.allclose(
            target_scale * pulled,
            source_scale * expected,
            rtol=tolerance,
            atol=tolerance * source_scale * float(np.max(np.abs(expected))),
        ):
            raise ValueError(
                f"FieldTransfer of exchange {exchange.exchange_id!r} fails the "
                "declared physical measurement identity L_target P = L_source."
            )


def _validate_transferred_physical(
    exchange: CouplingExchange,
    source: CouplingPort,
    target: CouplingPort,
    storage_change: bool,
    /,
) -> None:
    requirement = exchange.requirement
    source_measurement = source.measurement
    target_measurement = target.measurement
    if requirement is None or source_measurement is None or target_measurement is None:
        raise ValueError(
            "Physical FieldTransfer requires explicit measurements and transfer semantics."
        )
    if (source.frame == target.frame) != (requirement.frame_action == "preserve"):
        raise ValueError("Transfer frame_action does not match the physical frames.")
    if target.temporal_kind == "interval_integral" and not requirement.conservative:
        raise ValueError("Whole-window amounts require conservative transfers.")
    # Only the certified identity L_target P = L_source carries the measure between
    # density and extensive storage; an uncertified map would relabel values.
    if storage_change and not requirement.conservative:
        raise ValueError(
            f"Coupling exchange {exchange.exchange_id!r} changes the quantity "
            "dimension between density and extensive storage and requires a "
            "certified conservative transfer."
        )
    if requirement.conservative:
        _certify_conservative_transfer(exchange, source_measurement, target_measurement)


def _validate_physical_exchange(
    exchange: CouplingExchange,
    source: CouplingPort,
    target: CouplingPort,
    time_unit: UnitDefinition | None,
    /,
) -> None:
    received = _received_quantity(exchange, source, time_unit)
    if received is None and target.quantity is None:
        return
    if received is None or target.quantity is None:
        raise ValueError(
            "A physically typed exchange requires descriptors at both ports."
        )
    _validate_inventory_semantics(received, source, target, target.quantity)
    if exchange.transfer is None:
        _validate_direct_physical(source, target)
    else:
        _validate_transferred_physical(
            exchange,
            source,
            target,
            received.unit.dimension != target.quantity.unit.dimension,
        )


def _budget_row_ids(
    exchanges: tuple[CouplingExchange, ...],
    ports: dict[str, tuple[int, int, CouplingPort]],
    /,
) -> tuple[str, ...]:
    """One ledger row per scalar exchange and per component of a componentized one."""
    rows: list[str] = []
    for exchange in exchanges:
        target = ports[exchange.target_port_id][2]
        measurement = target.measurement
        if (
            target.temporal_kind == "interval_integral"
            and measurement is not None
            and measurement.component_count > 1
        ):
            rows.extend(
                f"{exchange.exchange_id}[{component}]"
                for component in measurement.component_ids
            )
        else:
            rows.append(exchange.exchange_id)
    return tuple(rows)


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
        if not subsystem.capabilities.jit:
            # Host participants are never traced; route admission lets them reach
            # this point only for the host orchestrator, which checks their signals
            # against live observations instead.
            continue
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
                f"Coupling subsystem {subsystem.subsystem_id!r} must return CouplingSubsystemResult."
            )
        if eqx.tree_equal(result.candidate_state, _shape_tree(state_value)) is not True:
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} candidate state "
                "must preserve the prepared state structure."
            )
        if len(result.outputs) != len(subsystem.output_ports):
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} returned the wrong number of output ports."
            )
        for port, output in zip(subsystem.output_ports, result.outputs, strict=True):
            shaped_output = output
            if port.waveform_plan is not None:
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


@dataclass(frozen=True, slots=True)
class _CanonicalCouplingInputs:
    differentiation: CouplingDifferentiationPolicy
    resources: CouplingResourcePolicy
    subsystems: tuple[AbstractCouplingSubsystem, ...]
    exchanges: tuple[CouplingExchange, ...]
    states: tuple[Any, ...]
    values: tuple[Any, ...]
    subsystem_ids: tuple[str, ...]
    exchange_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CouplingRoutes:
    ports: dict[str, tuple[int, int, CouplingPort]]
    port_ids: tuple[str, ...]
    values: tuple[Any, ...]
    source_subsystems: tuple[int, ...]
    target_subsystems: tuple[int, ...]
    source_output_indices: tuple[int, ...]
    target_input_indices: tuple[int, ...]
    input_exchange_indices: tuple[tuple[int, ...], ...]
    stages: tuple[Any, ...]
    implicit_exchange_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CouplingInterface:
    initial_state: CouplingState
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    coordinate_dtype: np.dtype


def _canonicalize_coupling_inputs(
    graph: CouplingGraph,
    participant_states: tuple[Any, ...],
    exchange_values: tuple[Any, ...],
    policy: AbstractCouplingPolicy,
    differentiation: CouplingDifferentiationPolicy | None,
    resources: CouplingResourcePolicy | None,
    /,
) -> _CanonicalCouplingInputs:
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
    return _CanonicalCouplingInputs(
        differentiation_,
        resources_,
        subsystems,
        exchanges,
        canonical_states,
        canonical_values,
        subsystem_ids,
        exchange_ids,
    )


def _admit_route_participants(
    subsystems: tuple[AbstractCouplingSubsystem, ...], /, *, host_execution: bool
) -> None:
    """Refuse participants the route cannot execute or feed on their ports.

    The native route traces every participant. Participants that are not
    JIT-capable execute only on the explicit host route, whose orchestrator admits
    exactly its own host participants among them.
    """
    host = sorted(
        subsystem.subsystem_id
        for subsystem in subsystems
        if not subsystem.capabilities.jit
    )
    if host and not host_execution:
        raise ValueError(
            "Native coupling requires every participant to be JIT-capable; "
            f"host-executed participants {host} run only through prepare_host_coupling."
        )
    if any(not subsystem.capabilities.fixed_topology for subsystem in subsystems):
        raise ValueError("Native coupling requires fixed-topology participants.")
    for subsystem in subsystems:
        ports_ = (*subsystem.input_ports, *subsystem.output_ports)
        if any(port.waveform_plan is None for port in ports_) and not (
            subsystem.capabilities.supports_endpoint
        ):
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} does not support its endpoint ports."
            )
        if any(port.waveform_plan is not None for port in ports_) and not (
            subsystem.capabilities.supports_waveform
        ):
            raise ValueError(
                f"Coupling subsystem {subsystem.subsystem_id!r} does not support its waveform ports."
            )


def _index_coupling_ports(
    subsystems: tuple[AbstractCouplingSubsystem, ...], /
) -> tuple[dict[str, tuple[int, int, CouplingPort]], tuple[str, ...]]:
    """Map each port to its canonical participant and local port index."""
    ports: dict[str, tuple[int, int, CouplingPort]] = {}
    port_ids: list[str] = []
    for subsystem_index, subsystem in enumerate(subsystems):
        for local_index, port in enumerate(subsystem.input_ports):
            ports[port.port_id] = (subsystem_index, local_index, port)
            port_ids.append(port.port_id)
        for local_index, port in enumerate(subsystem.output_ports):
            ports[port.port_id] = (subsystem_index, local_index, port)
            port_ids.append(port.port_id)
    return ports, tuple(port_ids)


def _resolve_exchange_endpoints(
    exchange: CouplingExchange,
    ports: dict[str, tuple[int, int, CouplingPort]],
    input_drivers: dict[str, int],
    /,
) -> tuple[tuple[int, int, CouplingPort], tuple[int, int, CouplingPort]]:
    """Resolve one known output-to-input route whose input has no other driver."""
    if exchange.source_port_id not in ports or exchange.target_port_id not in ports:
        raise ValueError(
            f"Coupling exchange {exchange.exchange_id!r} references an unknown port."
        )
    source = ports[exchange.source_port_id]
    target = ports[exchange.target_port_id]
    if source[2].direction != "output" or target[2].direction != "input":
        raise ValueError(
            f"Coupling exchange {exchange.exchange_id!r} must connect output to input."
        )
    if target[2].port_id in input_drivers:
        raise ValueError(
            f"Coupling input port {target[2].port_id!r} has multiple drivers."
        )
    return source, target


def _validate_transfer_spaces(
    exchange: CouplingExchange, source_port: CouplingPort, target_port: CouplingPort, /
) -> None:
    """Require the storage the direct, forward, or adjoint action maps between."""
    transfer = exchange.transfer
    if transfer is None:
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
            source_port.field_space.field_space_id != transfer.source.field_space_id
            or target_port.field_space.field_space_id != transfer.target.field_space_id
        ):
            raise ValueError(
                f"Coupling exchange {exchange.exchange_id!r} field spaces do not match its forward transfer."
            )
    else:
        if transfer.hilbert_adjoint_operator is None:
            raise ValueError(
                f"Coupling exchange {exchange.exchange_id!r} requests an unavailable adjoint transfer."
            )
        if source_port.field_space is None or target_port.field_space is None:
            raise ValueError("Adjoint transfers require field-valued ports.")
        if (
            source_port.field_space.field_space_id != transfer.target.field_space_id
            or target_port.field_space.field_space_id != transfer.source.field_space_id
        ):
            raise ValueError(
                f"Coupling exchange {exchange.exchange_id!r} field spaces do not match its adjoint transfer."
            )


def _refuse_whole_window_double_spend(
    exchange_index: int,
    exchanges: tuple[CouplingExchange, ...],
    ports: dict[str, tuple[int, int, CouplingPort]],
    target_port: CouplingPort,
    /,
) -> None:
    """Refuse a source whose whole-window amount an earlier exchange already spends."""
    if target_port.temporal_kind != "interval_integral":
        return
    source_port_id = exchanges[exchange_index].source_port_id
    if any(
        previous.source_port_id == source_port_id
        and ports[previous.target_port_id][2].temporal_kind == "interval_integral"
        for previous in exchanges[:exchange_index]
    ):
        raise ValueError(
            "An authoritative whole-window amount cannot be spent twice; "
            "partition the physical flux into explicit output ports."
        )


def _validate_route_coverage(
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    subsystem_ids: tuple[str, ...],
    input_drivers: dict[str, int],
    incident: frozenset[int],
    /,
) -> None:
    """Require one driver per input port and at least one exchange per participant."""
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
    isolated = [
        subsystem_ids[index] for index in range(len(subsystems)) if index not in incident
    ]
    if isolated:
        raise ValueError(
            "Coupling graph contains isolated subsystems: " + ", ".join(isolated)
        )


def _assemble_coupling_stages(
    source_subsystems: tuple[int, ...],
    target_subsystems: tuple[int, ...],
    subsystem_ids: tuple[str, ...],
    exchange_ids: tuple[str, ...],
    /,
) -> tuple[tuple[CouplingStagePlan, ...], tuple[int, ...]]:
    """Order strongly connected stages and list the exchanges solved implicitly."""
    adjacency_sets: list[set[int]] = [set() for _ in subsystem_ids]
    for source, target in zip(source_subsystems, target_subsystems, strict=True):
        adjacency_sets[source].add(target)
    adjacency = tuple(tuple(sorted(targets)) for targets in adjacency_sets)
    components = _strongly_connected_components(adjacency)
    stages = _ordered_stages(
        components,
        source_subsystems,
        target_subsystems,
        subsystem_ids,
        exchange_ids,
    )
    implicit_exchange_indices = tuple(
        exchange_index
        for stage in stages
        if stage.cyclic
        for exchange_index in stage.internal_exchange_indices
    )
    return stages, implicit_exchange_indices


def _prepare_coupling_routes(
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    exchanges: tuple[CouplingExchange, ...],
    canonical_values: tuple[Any, ...],
    subsystem_ids: tuple[str, ...],
    exchange_ids: tuple[str, ...],
    time_unit: UnitDefinition | None,
    /,
    *,
    host_execution: bool,
) -> _CouplingRoutes:
    _admit_route_participants(subsystems, host_execution=host_execution)
    ports, port_ids = _index_coupling_ports(subsystems)
    source_subsystems: list[int] = []
    target_subsystems: list[int] = []
    source_output_indices: list[int] = []
    target_input_indices: list[int] = []
    input_drivers: dict[str, int] = {}
    validated_values: list[Any] = []
    for exchange_index, (exchange, initial_value) in enumerate(
        zip(exchanges, canonical_values, strict=True)
    ):
        source, target = _resolve_exchange_endpoints(exchange, ports, input_drivers)
        source_subsystem, source_local, source_port = source
        target_subsystem, target_local, target_port = target
        input_drivers[target_port.port_id] = exchange_index
        _validate_transfer_spaces(exchange, source_port, target_port)
        _validate_requirement(exchange)
        _validate_temporal_exchange(exchange, source_port, target_port)
        _validate_physical_exchange(exchange, source_port, target_port, time_unit)
        _refuse_whole_window_double_spend(exchange_index, exchanges, ports, target_port)
        validated_values.append(validate_coupling_signal(target_port, initial_value))
        source_subsystems.append(source_subsystem)
        target_subsystems.append(target_subsystem)
        source_output_indices.append(source_local)
        target_input_indices.append(target_local)

    _validate_route_coverage(
        subsystems,
        subsystem_ids,
        input_drivers,
        frozenset((*source_subsystems, *target_subsystems)),
    )
    input_exchange_indices = tuple(
        tuple(input_drivers[port.port_id] for port in subsystem.input_ports)
        for subsystem in subsystems
    )
    stages, implicit_exchange_indices = _assemble_coupling_stages(
        tuple(source_subsystems),
        tuple(target_subsystems),
        subsystem_ids,
        exchange_ids,
    )
    return _CouplingRoutes(
        ports,
        port_ids,
        tuple(validated_values),
        tuple(source_subsystems),
        tuple(target_subsystems),
        tuple(source_output_indices),
        tuple(target_input_indices),
        input_exchange_indices,
        stages,
        implicit_exchange_indices,
    )


def _validate_coupling_policy(
    policy: AbstractCouplingPolicy,
    differentiation: CouplingDifferentiationPolicy,
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    exchanges: tuple[CouplingExchange, ...],
    subsystem_ids: tuple[str, ...],
    stages: tuple[Any, ...],
    implicit_exchange_indices: tuple[int, ...],
    /,
) -> None:
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

    if differentiation.mode == "algorithmic" and not isinstance(
        policy, ExplicitCouplingPolicy
    ):
        raise ValueError("Algorithmic coupling differentiation is explicit-only.")
    if differentiation.mode == "implicit":
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


def _prepare_coupling_interface(
    graph: CouplingGraph,
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    exchanges: tuple[CouplingExchange, ...],
    canonical_states: tuple[Any, ...],
    validated_values: tuple[Any, ...],
    subsystem_ids: tuple[str, ...],
    exchange_ids: tuple[str, ...],
    input_exchange_indices: tuple[tuple[int, ...], ...],
    implicit_exchange_indices: tuple[int, ...],
    ports: dict[str, tuple[int, int, CouplingPort]],
    time: Any,
    args: Any,
    /,
) -> _CouplingInterface:
    initial_state = CouplingState(
        canonical_states,
        tuple(validated_values),
        time,
        0,
        subsystem_ids=subsystem_ids,
        exchange_ids=exchange_ids,
        budget_row_ids=_budget_row_ids(exchanges, ports),
        graph_id=graph.graph_id,
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
        size = flattened.size
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
    return _CouplingInterface(
        initial_state,
        tuple(interface_offsets),
        tuple(interface_sizes),
        coordinate_dtype,
    )


def _estimate_coupling_resources(
    policy: AbstractCouplingPolicy,
    resources: CouplingResourcePolicy,
    subsystems: tuple[AbstractCouplingSubsystem, ...],
    canonical_states: tuple[Any, ...],
    validated_values: tuple[Any, ...],
    interface_size: int,
    coordinate_dtype: np.dtype,
    /,
) -> CouplingResourceEstimate:
    offset = interface_size
    resources_ = resources
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
    return estimate


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
    """Validate and compile one fixed-topology participant graph.

    Every participant must be JIT-capable; host-executed participants are refused
    and run only through `prepare_host_coupling`.
    """
    return _prepare_coupling_plan(
        graph,
        participant_states,
        exchange_values,
        policy=policy,
        differentiation=differentiation,
        time=time,
        args=args,
        problem_id=problem_id,
        resources=resources,
        host_execution=False,
    )


def _prepare_coupling_plan(
    graph: CouplingGraph,
    participant_states: tuple[Any, ...],
    exchange_values: tuple[Any, ...],
    /,
    *,
    policy: AbstractCouplingPolicy,
    differentiation: CouplingDifferentiationPolicy | None,
    time: Any,
    args: Any,
    problem_id: str,
    resources: CouplingResourcePolicy | None,
    host_execution: bool,
) -> PreparedCoupling:
    """Validate one participant graph for the native or the explicit host route."""

    canonical = _canonicalize_coupling_inputs(
        graph,
        participant_states,
        exchange_values,
        policy,
        differentiation,
        resources,
    )
    differentiation_ = canonical.differentiation
    resources_ = canonical.resources
    subsystems = canonical.subsystems
    exchanges = canonical.exchanges
    canonical_states = canonical.states
    canonical_values = canonical.values
    subsystem_ids = canonical.subsystem_ids
    exchange_ids = canonical.exchange_ids
    routes = _prepare_coupling_routes(
        subsystems,
        exchanges,
        canonical_values,
        subsystem_ids,
        exchange_ids,
        graph.time_unit,
        host_execution=host_execution,
    )
    ports = routes.ports
    port_ids = routes.port_ids
    validated_values = routes.values
    source_subsystems = routes.source_subsystems
    target_subsystems = routes.target_subsystems
    source_output_indices = routes.source_output_indices
    target_input_indices = routes.target_input_indices
    input_exchange_indices = routes.input_exchange_indices
    stages = routes.stages
    implicit_exchange_indices = routes.implicit_exchange_indices
    _validate_coupling_policy(
        policy,
        differentiation_,
        subsystems,
        exchanges,
        subsystem_ids,
        stages,
        implicit_exchange_indices,
    )
    interface = _prepare_coupling_interface(
        graph,
        subsystems,
        exchanges,
        canonical_states,
        validated_values,
        subsystem_ids,
        exchange_ids,
        input_exchange_indices,
        implicit_exchange_indices,
        ports,
        time,
        args,
    )
    initial_state = interface.initial_state
    interface_offsets = interface.offsets
    interface_sizes = interface.sizes
    coordinate_dtype = interface.coordinate_dtype
    offset = sum(interface_sizes)
    estimate = _estimate_coupling_resources(
        policy,
        resources_,
        subsystems,
        canonical_states,
        validated_values,
        offset,
        coordinate_dtype,
    )
    reasons: list[str] = []
    traced = all(subsystem.capabilities.jit for subsystem in subsystems)
    if not traced:
        reasons.append("host participants execute outside JAX transformations")
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
        jit_eligible=traced,
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
        time_unit=graph.time_unit,
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
    if not prepared.report.jit_eligible:
        raise ValueError(
            "A host coupling plan binds live host participants; prepare it again "
            "with prepare_host_coupling instead of refreshing it."
        )
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
        time_unit=prepared.time_unit,
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
