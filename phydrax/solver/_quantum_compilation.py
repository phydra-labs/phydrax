#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic native-gate compilation with caller-visible routing ledgers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._operations import (
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from ..operators.quantum._propagation import (
    kraus_trace_preservation_residual,
    unitarity_residual,
)
from ..operators.quantum._register import HilbertRegisterLayout


RouteStrategy: TypeAlias = Literal["swap", "interval"]


def _swap_matrix(dtype) -> Array:
    return jnp.asarray(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )


class HardwareTopology(StrictModule):
    """Finite physical topology and explicit native gate vocabulary."""

    physical_wire_ids: tuple[str, ...] = eqx.field(static=True)
    couplings: tuple[tuple[str, str], ...] = eqx.field(static=True)
    native_gate_set: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_wire_ids: Sequence[str],
        couplings: Sequence[tuple[str, str]],
        native_gate_set: Sequence[str],
        /,
    ):
        wires = tuple(str(value) for value in physical_wire_ids)
        if (
            not wires
            or any(not value for value in wires)
            or len(set(wires)) != len(wires)
        ):
            raise ValueError("Physical wire IDs must be unique and nonempty.")
        edges: list[tuple[str, str]] = []
        seen: set[frozenset[str]] = set()
        for left, right in couplings:
            pair = (str(left), str(right))
            if pair[0] not in wires or pair[1] not in wires or pair[0] == pair[1]:
                raise ValueError("Hardware couplings must join distinct known wires.")
            key = frozenset(pair)
            if key in seen:
                raise ValueError("Hardware couplings must not contain duplicate edges.")
            seen.add(key)
            edges.append(pair)
        native = tuple(str(value) for value in native_gate_set)
        if (
            not native
            or any(not value for value in native)
            or len(set(native)) != len(native)
        ):
            raise ValueError("native_gate_set must contain unique nonempty gate IDs.")
        self.physical_wire_ids = wires
        self.couplings = tuple(edges)
        self.native_gate_set = native
        self.topology_id = canonical_fingerprint(
            {
                "kind": "hardware-topology",
                "wires": wires,
                "couplings": tuple(edges),
                "native_gates": native,
            }
        )

    def adjacent(self, left: str, right: str, /) -> bool:
        return any(
            (first == left and second == right) or (first == right and second == left)
            for first, second in self.couplings
        )

    def shortest_path(self, source: str, target: str, /) -> tuple[str, ...]:
        if source not in self.physical_wire_ids or target not in self.physical_wire_ids:
            raise KeyError("Routing endpoints must be physical topology wires.")
        queue: deque[tuple[str, ...]] = deque(((source,),))
        visited = {source}
        order = {wire: index for index, wire in enumerate(self.physical_wire_ids)}
        while queue:
            path = queue.popleft()
            if path[-1] == target:
                return path
            neighbors: list[str] = []
            for left, right in self.couplings:
                if left == path[-1]:
                    neighbors.append(right)
                elif right == path[-1]:
                    neighbors.append(left)
            for neighbor in sorted(neighbors, key=order.__getitem__):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + (neighbor,))
        raise ValueError("No hardware route connects the requested logical targets.")


class QuantumCompilationPolicy(StrictModule):
    route_strategy: RouteStrategy = eqx.field(static=True)
    maximum_swaps: int = eqx.field(static=True)
    gate_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        route_strategy: RouteStrategy,
        maximum_swaps: int,
        gate_tolerance: float = 1e-8,
    ):
        swaps = int(maximum_swaps)
        tolerance = float(gate_tolerance)
        if route_strategy not in ("swap", "interval"):
            raise ValueError("route_strategy must be 'swap' or 'interval'.")
        if swaps < 0 or not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Compilation swap/tolerance policy is invalid.")
        self.route_strategy = route_strategy
        self.maximum_swaps = swaps
        self.gate_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quantum-compilation-policy",
                "route_strategy": route_strategy,
                "maximum_swaps": swaps,
                "gate_tolerance": tolerance,
            }
        )


class QuantumDecompositionRecord(StrictModule):
    original_operation: int = eqx.field(static=True)
    emitted_start: int = eqx.field(static=True)
    emitted_count: int = eqx.field(static=True)
    logical_targets: tuple[str, ...] = eqx.field(static=True)
    physical_targets_before: tuple[str, ...] = eqx.field(static=True)
    physical_targets_after: tuple[str, ...] = eqx.field(static=True)
    emitted_native_gates: tuple[str, ...] = eqx.field(static=True)
    swap_edges: tuple[tuple[str, str], ...] = eqx.field(static=True)
    strategy: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)


class QuantumCompilationResult(StrictModule):
    compiled_program: QuantumProgram
    ledger: tuple[QuantumDecompositionRecord, ...]
    final_logical_to_physical: tuple[tuple[str, str], ...] = eqx.field(static=True)
    swap_count: int = eqx.field(static=True)
    all_emitted_gates_native: Array
    no_hidden_swaps: Array
    emitted_structure_residuals: Array
    emitted_operations_valid: Array
    valid: Array
    compilation_id: str = eqx.field(static=True)


def _routing_edges(
    path: tuple[str, ...], strategy: RouteStrategy, /
) -> tuple[tuple[str, str], ...]:
    if len(path) <= 2:
        return ()
    if strategy == "swap":
        return tuple((path[index], path[index + 1]) for index in range(len(path) - 2))
    left = 0
    right = len(path) - 1
    move_left = True
    edges: list[tuple[str, str]] = []
    while right - left > 1:
        if move_left:
            edges.append((path[left], path[left + 1]))
            left += 1
        else:
            edges.append((path[right], path[right - 1]))
            right -= 1
        move_left = not move_left
    return tuple(edges)


def _native_gate_id(
    operation: LocalUnitaryOperation | LocalKrausChannelOperation,
    topology: HardwareTopology,
    /,
) -> str:
    target_count = len(operation.target_wire_ids)
    if isinstance(operation, LocalKrausChannelOperation):
        gate_id = "kraus-channel"
    else:
        gate_id = "unitary-1q" if target_count == 1 else "unitary-2q"
    if gate_id not in topology.native_gate_set:
        raise ValueError(
            f"Operation requires native gate {gate_id!r}; no implicit decomposition exists."
        )
    return gate_id


def compile_quantum_program(
    program: QuantumProgram,
    topology: HardwareTopology,
    policy: QuantumCompilationPolicy,
    /,
) -> QuantumCompilationResult:
    """Compile deterministically; every inserted SWAP is an emitted operation and ledger row."""
    if not isinstance(program, QuantumProgram):
        raise TypeError("program must be a QuantumProgram.")
    if not isinstance(topology, HardwareTopology) or not isinstance(
        policy, QuantumCompilationPolicy
    ):
        raise TypeError("topology/policy types are invalid.")
    if len(program.layout.wire_ids) != len(topology.physical_wire_ids):
        raise ValueError(
            "Compilation requires an explicit one-to-one logical/physical layout; "
            "ancilla insertion is never implicit."
        )
    if any(dimension != 2 for dimension in program.layout.local_dimensions):
        raise ValueError(
            "Hardware compilation currently requires qubit local dimensions."
        )
    logical_to_physical = dict(
        zip(program.layout.wire_ids, topology.physical_wire_ids, strict=True)
    )
    physical_to_logical = {value: key for key, value in logical_to_physical.items()}
    emitted: list[LocalUnitaryOperation | LocalKrausChannelOperation] = []
    ledger: list[QuantumDecompositionRecord] = []
    swap_count = 0
    for operation_index, operation in enumerate(program.operations):
        logical_targets = operation.target_wire_ids
        if len(logical_targets) > 2:
            raise ValueError("Hardware compiler accepts only one/two-qubit operations.")
        physical_before = tuple(logical_to_physical[value] for value in logical_targets)
        swap_edges: tuple[tuple[str, str], ...] = ()
        gate_names: list[str] = []
        emitted_start = len(emitted)
        if len(logical_targets) == 2 and not topology.adjacent(*physical_before):
            if isinstance(operation, LocalKrausChannelOperation):
                raise ValueError(
                    "Nonlocal Kraus channels require caller-provided decomposition."
                )
            if "swap" not in topology.native_gate_set:
                raise ValueError(
                    "Nonlocal routing requires an explicit native 'swap' gate."
                )
            path = topology.shortest_path(*physical_before)
            swap_edges = _routing_edges(path, policy.route_strategy)
            if swap_count + len(swap_edges) > policy.maximum_swaps:
                raise MemoryError("Compilation exceeds maximum_swaps.")
            for left, right in swap_edges:
                emitted.append(
                    LocalUnitaryOperation(
                        _swap_matrix(operation.unitary.dtype), (left, right)
                    )
                )
                gate_names.append("swap")
                left_logical = physical_to_logical[left]
                right_logical = physical_to_logical[right]
                physical_to_logical[left], physical_to_logical[right] = (
                    right_logical,
                    left_logical,
                )
                logical_to_physical[left_logical], logical_to_physical[right_logical] = (
                    right,
                    left,
                )
            swap_count += len(swap_edges)
        physical_after = tuple(logical_to_physical[value] for value in logical_targets)
        if len(physical_after) == 2 and not topology.adjacent(*physical_after):
            raise RuntimeError(
                "Explicit routing ledger did not produce adjacent targets."
            )
        native_id = _native_gate_id(operation, topology)
        if isinstance(operation, LocalUnitaryOperation):
            emitted.append(LocalUnitaryOperation(operation.unitary, physical_after))
        else:
            emitted.append(LocalKrausChannelOperation(operation.kraus, physical_after))
        gate_names.append(native_id)
        record_id = canonical_fingerprint(
            {
                "kind": "quantum-decomposition-record",
                "operation": operation_index,
                "logical_targets": logical_targets,
                "physical_before": physical_before,
                "physical_after": physical_after,
                "native_gates": tuple(gate_names),
                "swaps": swap_edges,
                "strategy": policy.route_strategy,
            }
        )
        ledger.append(
            QuantumDecompositionRecord(
                operation_index,
                emitted_start,
                len(emitted) - emitted_start,
                logical_targets,
                physical_before,
                physical_after,
                tuple(gate_names),
                swap_edges,
                policy.route_strategy,
                record_id,
            )
        )
    physical_layout = HilbertRegisterLayout(
        topology.physical_wire_ids, (2,) * len(topology.physical_wire_ids)
    )
    compiled = QuantumProgram(
        physical_layout, tuple(emitted), state_kind=program.state_kind
    )
    native = jnp.asarray(
        all(
            gate in topology.native_gate_set
            for record in ledger
            for gate in record.emitted_native_gates
        )
    )
    visible = jnp.asarray(
        swap_count
        == sum(len(record.swap_edges) for record in ledger)
        == sum(
            gate == "swap" for record in ledger for gate in record.emitted_native_gates
        )
    )
    residuals = (
        jnp.stack(
            [
                unitarity_residual(operation.unitary)
                if isinstance(operation, LocalUnitaryOperation)
                else kraus_trace_preservation_residual(operation.kraus)
                for operation in emitted
            ]
        )
        if emitted
        else jnp.empty((0,))
    )
    numerical = jnp.all(jnp.isfinite(residuals)) & jnp.all(
        residuals <= policy.gate_tolerance
    )
    identifier = canonical_fingerprint(
        {
            "kind": "quantum-compilation-result",
            "program": program.program_id,
            "compiled": compiled.program_id,
            "topology": topology.topology_id,
            "policy": policy.policy_id,
            "ledger": tuple(record.record_id for record in ledger),
        }
    )
    return QuantumCompilationResult(
        compiled,
        tuple(ledger),
        tuple((wire, logical_to_physical[wire]) for wire in program.layout.wire_ids),
        swap_count,
        native,
        visible,
        residuals,
        numerical,
        native & visible & numerical,
        identifier,
    )


class ControlHamiltonianTerm(StrictModule):
    generator: Array
    target_wire_ids: tuple[str, ...] = eqx.field(static=True)
    hermiticity_residual: Array
    finite: Array
    valid: Array
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        generator: ArrayLike,
        target_wire_ids: Sequence[str],
        /,
        *,
        tolerance: float = 1e-8,
    ):
        value = jnp.asarray(generator)
        targets = tuple(str(target) for target in target_wire_ids)
        tolerance_ = float(tolerance)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or not targets:
            raise ValueError("Control generators require a square matrix and targets.")
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Control generators must use complex coordinates.")
        residual = jnp.max(jnp.abs(value - jnp.conj(value.T)))
        finite = jnp.all(jnp.isfinite(value))
        self.generator = value
        self.target_wire_ids = targets
        self.hermiticity_residual = residual
        self.finite = finite
        self.valid = finite & jnp.isfinite(residual) & (residual <= tolerance_)
        self.term_id = canonical_fingerprint(
            {
                "kind": "control-hamiltonian-term",
                "targets": targets,
                "shape": value.shape,
                "dtype": str(value.dtype),
                "tolerance": tolerance_,
            }
        )


class FixedGridQuantumControl(StrictModule):
    layout: HilbertRegisterLayout
    terms: tuple[ControlHamiltonianTerm, ...]
    time_grid: Array
    amplitudes: Array
    positive_intervals: Array
    finite: Array
    valid: Array
    interval_count: int = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: HilbertRegisterLayout,
        terms: Sequence[ControlHamiltonianTerm],
        time_grid: ArrayLike,
        amplitudes: ArrayLike,
        /,
    ):
        if not isinstance(layout, HilbertRegisterLayout):
            raise TypeError("layout must be HilbertRegisterLayout.")
        selected = tuple(terms)
        times = jnp.asarray(time_grid)
        values = jnp.asarray(amplitudes)
        if not selected or any(
            not isinstance(term, ControlHamiltonianTerm) for term in selected
        ):
            raise ValueError("At least one ControlHamiltonianTerm is required.")
        if times.ndim != 1 or times.shape[0] < 2:
            raise ValueError("time_grid requires fixed shape (intervals + 1,).")
        if values.shape != (times.shape[0] - 1, len(selected)):
            raise ValueError("amplitudes requires shape (intervals, terms).")
        for term in selected:
            if layout.target_dimension(term.target_wire_ids) != term.generator.shape[0]:
                raise ValueError("Control term dimension does not match its targets.")
        intervals = jnp.diff(times)
        positive = jnp.all(intervals > 0.0)
        finite = jnp.all(jnp.isfinite(times)) & jnp.all(jnp.isfinite(values))
        valid = finite & positive & jnp.all(jnp.stack([term.valid for term in selected]))
        self.layout = layout
        self.terms = selected
        self.time_grid = times
        self.amplitudes = values
        self.positive_intervals = positive
        self.finite = finite
        self.valid = valid
        self.interval_count = int(times.shape[0] - 1)
        self.control_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-quantum-control",
                "layout": layout.layout_id,
                "terms": tuple(term.term_id for term in selected),
                "grid_shape": times.shape,
                "amplitude_shape": values.shape,
                "dtype": str(values.dtype),
            }
        )


class FixedGridControlResult(StrictModule):
    program: QuantumProgram
    step_unitarity_residuals: Array
    grid_intervals: Array
    finite: Array
    valid: Array
    control_id: str = eqx.field(static=True)


def discretize_fixed_grid_control(
    control: FixedGridQuantumControl, /
) -> FixedGridControlResult:
    """Emit the exact advertised first-order piecewise-constant product formula."""
    if not isinstance(control, FixedGridQuantumControl):
        raise TypeError("control must be FixedGridQuantumControl.")
    intervals = jnp.diff(control.time_grid)
    operations: list[LocalUnitaryOperation] = []
    residuals: list[Array] = []
    for interval in range(control.interval_count):
        for term_index, term in enumerate(control.terms):
            unitary = jsp.linalg.expm(
                -1j
                * intervals[interval]
                * control.amplitudes[interval, term_index]
                * term.generator
            )
            operations.append(LocalUnitaryOperation(unitary, term.target_wire_ids))
            residuals.append(unitarity_residual(unitary))
    residuals_ = jnp.stack(residuals)
    program = QuantumProgram(control.layout, operations, state_kind="state-vector")
    finite = jnp.all(jnp.isfinite(residuals_))
    return FixedGridControlResult(
        program,
        residuals_,
        intervals,
        finite,
        control.valid & finite,
        control.control_id,
    )


__all__ = [
    "ControlHamiltonianTerm",
    "FixedGridControlResult",
    "FixedGridQuantumControl",
    "HardwareTopology",
    "QuantumCompilationPolicy",
    "QuantumCompilationResult",
    "QuantumDecompositionRecord",
    "RouteStrategy",
    "compile_quantum_program",
    "discretize_fixed_grid_control",
]
