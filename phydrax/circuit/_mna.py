#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from enum import IntEnum
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    FGMRES,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    plan as plan_linear,
    prepare as prepare_linear,
    PreparedLinearSolve,
    refresh as refresh_linear,
    solve as solve_linear,
    TolerancePolicy,
)
from ..sparse import EdgeRelation, SparseLinearMap
from ._ports import ElectricalWaveReference
from ._relation_graph import plan_linear_routes


NodeId: TypeAlias = str


class NodalPort(StrictModule):
    """Ordered positive/negative electrical port with current into the DUT."""

    reference: ElectricalWaveReference
    port_id: str = eqx.field(static=True)
    positive: NodeId = eqx.field(static=True)
    negative: NodeId = eqx.field(static=True)

    def __init__(
        self,
        port_id: str,
        positive: NodeId,
        negative: NodeId,
        reference: ElectricalWaveReference,
        /,
    ):
        values = tuple(str(value) for value in (port_id, positive, negative))
        if any(not value for value in values):
            raise ValueError("Nodal port and node IDs must be non-empty.")
        if values[1] == values[2]:
            raise ValueError("Nodal port positive and negative nodes must differ.")
        if not isinstance(reference, ElectricalWaveReference):
            raise TypeError("Nodal ports require ElectricalWaveReference.")
        self.port_id, self.positive, self.negative = values
        self.reference = reference


class MNAStamp(StrictModule):
    """Local equations ``i=Yv+Bx`` and ``0=Cv+Dx``."""

    y: Array
    b: Array
    c: Array
    d: Array

    def __init__(self, y: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike, /):
        y_, b_, c_, d_ = (jnp.asarray(value) for value in (y, b, c, d))
        if any(value.ndim < 2 for value in (y_, b_, c_, d_)):
            raise ValueError("MNA stamp blocks must have at least two axes.")
        terminals = int(y_.shape[-1])
        auxiliaries = int(d_.shape[-1])
        batch = y_.shape[:-2]
        if (
            y_.shape[-2:] != (terminals, terminals)
            or b_.shape != batch + (terminals, auxiliaries)
            or c_.shape != batch + (auxiliaries, terminals)
            or d_.shape != batch + (auxiliaries, auxiliaries)
        ):
            raise ValueError("MNA blocks have incompatible batch or local dimensions.")
        dtype = jnp.result_type(y_, b_, c_, d_, jnp.complex128)
        self.y, self.b, self.c, self.d = (
            value.astype(dtype) for value in (y_, b_, c_, d_)
        )

    @property
    def terminal_count(self) -> int:
        return int(self.y.shape[-1])

    @property
    def auxiliary_count(self) -> int:
        return int(self.d.shape[-1])


class AbstractMNAComponent(StrictModule):
    """Fixed local stamp shape with frequency-dependent coefficients."""

    @property
    @abstractmethod
    def terminal_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def auxiliary_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, angular_frequency: ArrayLike, /) -> MNAStamp:
        raise NotImplementedError


class CircuitInstance(StrictModule):
    component: AbstractMNAComponent
    instance_id: str = eqx.field(static=True)
    nodes: tuple[NodeId, ...] = eqx.field(static=True)

    def __init__(
        self,
        instance_id: str,
        component: AbstractMNAComponent,
        nodes: Sequence[NodeId],
        /,
    ):
        identifier = str(instance_id)
        node_tuple = tuple(str(node) for node in nodes)
        if not identifier or any(not node for node in node_tuple):
            raise ValueError("MNA instance and node IDs must be non-empty.")
        if not isinstance(component, AbstractMNAComponent):
            raise TypeError("component must be AbstractMNAComponent.")
        if len(node_tuple) != component.terminal_count:
            raise ValueError("MNA instance node count must match component terminals.")
        self.component = component
        self.instance_id = identifier
        self.nodes = node_tuple


class VoltageProbe(StrictModule):
    probe_id: str = eqx.field(static=True)
    positive: NodeId = eqx.field(static=True)
    negative: NodeId = eqx.field(static=True)

    def __init__(self, probe_id: str, positive: NodeId, negative: NodeId, /):
        values = tuple(str(value) for value in (probe_id, positive, negative))
        if any(not value for value in values) or values[1] == values[2]:
            raise ValueError("MNA probe IDs must be non-empty and nodes must differ.")
        self.probe_id, self.positive, self.negative = values


class NodalCircuit(StrictModule):
    instances: tuple[CircuitInstance, ...]
    ports: tuple[NodalPort, ...]
    probes: tuple[VoltageProbe, ...]
    nodes: tuple[NodeId, ...] = eqx.field(static=True)
    ground: NodeId | None = eqx.field(static=True)
    circuit_id: str = eqx.field(static=True)

    def __init__(
        self,
        instances: Sequence[CircuitInstance],
        ports: Sequence[NodalPort],
        /,
        *,
        ground: NodeId | None,
        nodes: Sequence[NodeId] | None = None,
        probes: Sequence[VoltageProbe] = (),
        circuit_id: str = "nodal-circuit",
    ):
        instance_tuple = tuple(instances)
        port_tuple = tuple(ports)
        probe_tuple = tuple(probes)
        if not instance_tuple or any(
            not isinstance(value, CircuitInstance) for value in instance_tuple
        ):
            raise ValueError(
                "instances must be a non-empty sequence of CircuitInstance values."
            )
        if not port_tuple or any(
            not isinstance(value, NodalPort) for value in port_tuple
        ):
            raise ValueError("ports must be a non-empty sequence of NodalPort values.")
        if any(not isinstance(value, VoltageProbe) for value in probe_tuple):
            raise TypeError("probes must contain VoltageProbe values.")
        for ids, owner in (
            (tuple(value.instance_id for value in instance_tuple), "instance"),
            (tuple(value.port_id for value in port_tuple), "port"),
            (tuple(value.probe_id for value in probe_tuple), "probe"),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"MNA {owner} IDs must be unique.")
        inferred: list[str] = []

        def include(node: str) -> None:
            if node not in inferred:
                inferred.append(node)

        if ground is not None:
            include(str(ground))
        for instance in instance_tuple:
            for node in instance.nodes:
                include(node)
        for port in port_tuple:
            include(port.positive)
            include(port.negative)
        for probe in probe_tuple:
            include(probe.positive)
            include(probe.negative)
        node_tuple = (
            tuple(inferred) if nodes is None else tuple(str(node) for node in nodes)
        )
        if any(not node for node in node_tuple) or len(set(node_tuple)) != len(
            node_tuple
        ):
            raise ValueError("nodes must be unique and non-empty.")
        if any(node not in node_tuple for node in inferred):
            raise ValueError("nodes omits a node referenced by circuit topology.")
        identifier = str(circuit_id)
        if not identifier:
            raise ValueError("circuit_id must be non-empty.")
        self.instances = instance_tuple
        self.ports = port_tuple
        self.probes = probe_tuple
        self.nodes = node_tuple
        self.ground = None if ground is None else str(ground)
        self.circuit_id = identifier


class MNASolveStatus(IntEnum):
    SUCCESS = 0
    SINGULAR = 1
    NONFINITE = 2
    RESIDUAL_TOLERANCE_NOT_MET = 3


class MNASolvePolicy(StrictModule):
    linear: LinearSolvePolicy
    assembly: Literal["dense", "sparse"] = eqx.field(static=True)
    maximum_unknowns: int = eqx.field(static=True)
    maximum_matrix_bytes: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        assembly: Literal["dense", "sparse"] = "dense",
        maximum_unknowns: int = 8192,
        maximum_matrix_bytes: int = 2**30,
        residual_tolerance: float = 1e-10,
        linear: LinearSolvePolicy | None = None,
    ):
        if assembly not in ("dense", "sparse"):
            raise ValueError("assembly must be 'dense' or 'sparse'.")
        if maximum_unknowns <= 0 or maximum_matrix_bytes <= 0 or residual_tolerance < 0.0:
            raise ValueError(
                "MNA resource limits must be positive and tolerance non-negative."
            )
        if linear is None:
            if assembly == "dense":
                selected_linear = LinearSolvePolicy(
                    DenseLU(),
                    failure=FailurePolicy("status"),
                )
            else:
                tolerance = max(float(residual_tolerance), 1e-12)
                selected_linear = LinearSolvePolicy(
                    FGMRES(restart=64, stagnation_iterations=64),
                    tolerance=TolerancePolicy(
                        relative=tolerance,
                        absolute=tolerance,
                    ),
                    failure=FailurePolicy("status"),
                )
        else:
            selected_linear = linear
        self.linear = selected_linear
        if not isinstance(self.linear, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if assembly == "sparse" and not isinstance(self.linear.method, FGMRES):
            raise ValueError("Sparse MNA requires the native FGMRES method.")
        self.assembly = assembly
        self.maximum_unknowns = int(maximum_unknowns)
        self.maximum_matrix_bytes = int(maximum_matrix_bytes)
        self.residual_tolerance = float(residual_tolerance)


class MNACostEstimate(StrictModule):
    nodes: int = eqx.field(static=True)
    auxiliary_unknowns: int = eqx.field(static=True)
    port_unknowns: int = eqx.field(static=True)
    total_unknowns: int = eqx.field(static=True)
    structural_entries: int = eqx.field(static=True)
    matrix_bytes: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)


class MNAPlan(StrictModule):
    circuit: NodalCircuit
    policy: MNASolvePolicy
    linear_plan: LinearSolvePlan
    cost: MNACostEstimate
    sparse_relation: EdgeRelation | None
    frequency_shape: tuple[int, ...] = eqx.field(static=True)
    frequency_dtype: str = eqx.field(static=True)
    node_ids: tuple[NodeId, ...] = eqx.field(static=True)
    instance_auxiliary_offsets: tuple[int, ...] = eqx.field(static=True)
    port_current_offset: int = eqx.field(static=True)
    topology_signature: tuple[Any, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedMNA(StrictModule):
    circuit: NodalCircuit
    plan: MNAPlan
    angular_frequency: Array
    matrix: Array | None
    sparse_operator: SparseLinearMap | None
    linear: PreparedLinearSolve
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class MNADiagnostics(StrictModule):
    status: Array
    linear_status: Array
    original_residual: Array
    relative_residual: Array
    kcl_residual: Array
    port_power_balance: Array
    finite: Array

    @property
    def successful(self) -> Array:
        return self.status == int(MNASolveStatus.SUCCESS)


class MNAResult(StrictModule):
    solution: Array
    node_voltages: Array
    port_voltages: Array
    port_currents: Array
    outgoing: Array
    incident: Array
    probe_voltages: tuple[Array, ...]
    diagnostics: MNADiagnostics
    numeric_version: Array
    node_ids: tuple[NodeId, ...] = eqx.field(static=True)
    port_ids: tuple[str, ...] = eqx.field(static=True)
    probe_ids: tuple[str, ...] = eqx.field(static=True)


def _topology_signature(circuit: NodalCircuit) -> tuple[Any, ...]:
    return (
        circuit.ground,
        circuit.nodes,
        tuple(
            (
                value.instance_id,
                value.nodes,
                value.component.terminal_count,
                value.component.auxiliary_count,
            )
            for value in circuit.instances
        ),
        tuple((value.port_id, value.positive, value.negative) for value in circuit.ports),
        tuple(
            (value.probe_id, value.positive, value.negative) for value in circuit.probes
        ),
    )


def _validate_grounding(circuit: NodalCircuit) -> None:
    if circuit.ground is None or circuit.ground not in circuit.nodes:
        raise ValueError("MNA planning requires one explicit ground node.")
    adjacency = {node: set() for node in circuit.nodes}
    groups = [instance.nodes for instance in circuit.instances] + [
        (port.positive, port.negative) for port in circuit.ports
    ]
    for group in groups:
        for first in group:
            adjacency[first].update(node for node in group if node != first)
    reached = {circuit.ground}
    frontier = [circuit.ground]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    floating = tuple(node for node in circuit.nodes if node not in reached)
    if floating:
        raise ValueError(
            f"Disconnected floating MNA node set has no ground path: {floating!r}."
        )


def _mna_sparse_relation(
    circuit: NodalCircuit,
    node_ids: tuple[NodeId, ...],
    auxiliary_offsets: tuple[int, ...],
    port_current_offset: int,
    total_unknowns: int,
    /,
) -> EdgeRelation:
    sources: list[int] = []
    targets: list[int] = []

    def node_index(node: NodeId) -> int | None:
        return None if node == circuit.ground else node_ids.index(node)

    def append(row: int, column: int) -> None:
        sources.append(column)
        targets.append(row)

    for instance, auxiliary_offset in zip(
        circuit.instances, auxiliary_offsets, strict=True
    ):
        terminals = instance.component.terminal_count
        auxiliaries = instance.component.auxiliary_count
        routes = tuple(node_index(node) for node in instance.nodes)
        for local_row in range(terminals):
            global_row = routes[local_row]
            if global_row is None:
                continue
            for local_column in range(terminals):
                global_column = routes[local_column]
                if global_column is not None:
                    append(global_row, global_column)
            for auxiliary in range(auxiliaries):
                append(global_row, auxiliary_offset + auxiliary)
        for auxiliary in range(auxiliaries):
            row = auxiliary_offset + auxiliary
            for local_column in range(terminals):
                global_column = routes[local_column]
                if global_column is not None:
                    append(row, global_column)
            for local_column in range(auxiliaries):
                append(row, auxiliary_offset + local_column)
    for port_index, port in enumerate(circuit.ports):
        current = port_current_offset + port_index
        positive = node_index(port.positive)
        negative = node_index(port.negative)
        if positive is not None:
            append(positive, current)
        if negative is not None:
            append(negative, current)
        if positive is not None:
            append(current, positive)
        if negative is not None:
            append(current, negative)
        append(current, current)
    return plan_linear_routes(
        total_unknowns,
        total_unknowns,
        sources,
        targets,
        plan_id=f"{circuit.circuit_id}/mna-routes",
    ).relation


def plan_mna(
    circuit: NodalCircuit,
    angular_frequency: ArrayLike,
    policy: MNASolvePolicy | None = None,
    /,
) -> MNAPlan:
    """Compile grounded node/auxiliary/port coordinates and preflight resources."""
    if not isinstance(circuit, NodalCircuit):
        raise TypeError("circuit must be NodalCircuit.")
    _validate_grounding(circuit)
    omega = jnp.asarray(angular_frequency)
    if not jnp.issubdtype(omega.dtype, jnp.number):
        raise TypeError("angular_frequency must be numeric.")
    selected = MNASolvePolicy() if policy is None else policy
    if not isinstance(selected, MNASolvePolicy):
        raise TypeError("policy must be MNASolvePolicy or None.")
    if selected.assembly == "sparse" and omega.ndim != 0:
        raise ValueError(
            "Sparse MNA requires one scalar angular frequency per prepared solve."
        )
    nodes = tuple(node for node in circuit.nodes if node != circuit.ground)
    node_count = len(nodes)
    offsets: list[int] = []
    cursor = node_count
    for instance in circuit.instances:
        offsets.append(cursor)
        cursor += instance.component.auxiliary_count
    port_offset = cursor
    cursor += len(circuit.ports)
    if cursor > selected.maximum_unknowns:
        raise MemoryError("MNA system exceeds maximum_unknowns.")
    relation = _mna_sparse_relation(
        circuit,
        nodes,
        tuple(offsets),
        port_offset,
        cursor,
    )
    structural_entries = relation.capacity
    frequencies = prod(omega.shape) if omega.shape else 1
    if selected.assembly == "dense":
        matrix_bytes = frequencies * cursor * cursor * 16
        factor_bytes = matrix_bytes
        sparse_relation = None
    else:
        route_bytes = sum(
            int(value.size * value.dtype.itemsize)
            for value in (
                relation.source_indices,
                relation.target_indices,
                relation.valid,
            )
        )
        matrix_bytes = frequencies * structural_entries * 16 + route_bytes
        factor_bytes = 0
        sparse_relation = relation
    if matrix_bytes + factor_bytes > selected.maximum_matrix_bytes:
        raise MemoryError("MNA system exceeds maximum_matrix_bytes.")
    if selected.assembly == "dense":
        template_operator = DenseLinearOperator(
            jnp.broadcast_to(
                jnp.eye(cursor, dtype=jnp.complex128),
                omega.shape + (cursor, cursor),
            ),
            operator_id=f"{circuit.circuit_id}/mna-matrix",
        )
    else:
        template_operator = SparseLinearMap(
            relation,
            jnp.ones(
                omega.shape + (structural_entries,),
                dtype=jnp.complex128,
            ),
            operator_id=f"{circuit.circuit_id}/mna-sparse-matrix",
        )
    problem = LinearSystem(
        template_operator,
        problem_id=f"{circuit.circuit_id}/mna-system",
    )
    linear_plan = plan_linear(problem, selected.linear)
    if selected.assembly == "sparse" and linear_plan.backend != "native-krylov":
        raise ValueError("Sparse MNA planning selected a materializing linear backend.")
    signature = _topology_signature(circuit)
    plan_id = canonical_fingerprint(
        {
            "kind": "mna-plan",
            "circuit": circuit.circuit_id,
            "topology": signature,
            "frequency_shape": omega.shape,
            "assembly": selected.assembly,
            "linear": linear_plan.plan_id,
        }
    )
    return MNAPlan(
        circuit,
        selected,
        linear_plan,
        MNACostEstimate(
            nodes=node_count,
            auxiliary_unknowns=port_offset - node_count,
            port_unknowns=len(circuit.ports),
            total_unknowns=cursor,
            structural_entries=structural_entries,
            matrix_bytes=matrix_bytes,
            factor_bytes=factor_bytes,
        ),
        sparse_relation,
        tuple(omega.shape),
        str(omega.dtype),
        nodes,
        tuple(offsets),
        port_offset,
        signature,
        plan_id,
    )


def _node_index(plan: MNAPlan, node: NodeId) -> int | None:
    return None if node == plan.circuit.ground else plan.node_ids.index(node)


def _validate_mna_assembly(
    circuit: NodalCircuit,
    omega: Array,
    plan: MNAPlan,
    /,
) -> None:
    if (
        tuple(omega.shape) != plan.frequency_shape
        or str(omega.dtype) != plan.frequency_dtype
    ):
        raise ValueError(
            "Angular-frequency shape or dtype changed; MNA replanning is required."
        )
    if _topology_signature(circuit) != plan.topology_signature:
        raise ValueError("MNA topology or stamp shape changed; replanning is required.")


def _validate_mna_stamp(
    stamp: MNAStamp,
    omega: Array,
    terminals: int,
    auxiliaries: int,
    /,
) -> None:
    if (
        stamp.y.shape != omega.shape + (terminals, terminals)
        or stamp.b.shape != omega.shape + (terminals, auxiliaries)
        or stamp.c.shape != omega.shape + (auxiliaries, terminals)
        or stamp.d.shape != omega.shape + (auxiliaries, auxiliaries)
    ):
        raise ValueError(
            "MNA component returned a different frequency batch or stamp shape."
        )


def _assemble_mna(circuit: NodalCircuit, omega: Array, plan: MNAPlan) -> Array:
    _validate_mna_assembly(circuit, omega, plan)
    size = plan.cost.total_unknowns
    matrix = jnp.zeros(omega.shape + (size, size), dtype=jnp.complex128)
    for instance, auxiliary_offset in zip(
        circuit.instances, plan.instance_auxiliary_offsets, strict=True
    ):
        stamp = instance.component.evaluate(omega)
        terminals = instance.component.terminal_count
        auxiliaries = instance.component.auxiliary_count
        _validate_mna_stamp(stamp, omega, terminals, auxiliaries)
        routes = tuple(_node_index(plan, node) for node in instance.nodes)
        for local_row, global_row in enumerate(routes):
            if global_row is None:
                continue
            for local_column, global_column in enumerate(routes):
                if global_column is not None:
                    matrix = matrix.at[..., global_row, global_column].add(
                        stamp.y[..., local_row, local_column]
                    )
            for auxiliary in range(auxiliaries):
                matrix = matrix.at[..., global_row, auxiliary_offset + auxiliary].add(
                    stamp.b[..., local_row, auxiliary]
                )
        for auxiliary in range(auxiliaries):
            row = auxiliary_offset + auxiliary
            for local_column, global_column in enumerate(routes):
                if global_column is not None:
                    matrix = matrix.at[..., row, global_column].add(
                        stamp.c[..., auxiliary, local_column]
                    )
            for local_column in range(auxiliaries):
                matrix = matrix.at[..., row, auxiliary_offset + local_column].add(
                    stamp.d[..., auxiliary, local_column]
                )
    for port_index, port in enumerate(circuit.ports):
        current_column = plan.port_current_offset + port_index
        positive = _node_index(plan, port.positive)
        negative = _node_index(plan, port.negative)
        if positive is not None:
            matrix = matrix.at[..., positive, current_column].add(-1.0)
        if negative is not None:
            matrix = matrix.at[..., negative, current_column].add(1.0)
        row = current_column
        if positive is not None:
            matrix = matrix.at[..., row, positive].add(1.0)
        if negative is not None:
            matrix = matrix.at[..., row, negative].add(-1.0)
        z0 = port.reference.z0
        if z0.ndim == 0:
            z0 = jnp.broadcast_to(z0, omega.shape)
        elif z0.shape != omega.shape:
            raise ValueError("MNA port reference batch must match angular_frequency.")
        matrix = matrix.at[..., row, current_column].add(z0)
    return matrix


def _assemble_sparse_mna(
    circuit: NodalCircuit,
    omega: Array,
    plan: MNAPlan,
    /,
) -> SparseLinearMap:
    _validate_mna_assembly(circuit, omega, plan)
    if plan.sparse_relation is None:
        raise ValueError("Sparse MNA assembly requires a sparse MNA plan.")
    values: list[Array] = []

    def append(value: ArrayLike) -> None:
        scalar = jnp.asarray(value, dtype=jnp.complex128)
        values.append(jnp.broadcast_to(scalar, omega.shape))

    for instance, auxiliary_offset in zip(
        circuit.instances, plan.instance_auxiliary_offsets, strict=True
    ):
        stamp = instance.component.evaluate(omega)
        terminals = instance.component.terminal_count
        auxiliaries = instance.component.auxiliary_count
        _validate_mna_stamp(stamp, omega, terminals, auxiliaries)
        routes = tuple(_node_index(plan, node) for node in instance.nodes)
        for local_row, global_row in enumerate(routes):
            if global_row is None:
                continue
            for local_column, global_column in enumerate(routes):
                if global_column is not None:
                    append(stamp.y[..., local_row, local_column])
            for auxiliary in range(auxiliaries):
                append(stamp.b[..., local_row, auxiliary])
        for auxiliary in range(auxiliaries):
            for local_column, global_column in enumerate(routes):
                if global_column is not None:
                    append(stamp.c[..., auxiliary, local_column])
            for local_column in range(auxiliaries):
                append(stamp.d[..., auxiliary, local_column])
    for port in circuit.ports:
        positive = _node_index(plan, port.positive)
        negative = _node_index(plan, port.negative)
        if positive is not None:
            append(-1.0)
        if negative is not None:
            append(1.0)
        if positive is not None:
            append(1.0)
        if negative is not None:
            append(-1.0)
        z0 = port.reference.z0
        if z0.ndim == 0:
            z0 = jnp.broadcast_to(z0, omega.shape)
        elif z0.shape != omega.shape:
            raise ValueError("MNA port reference batch must match angular_frequency.")
        append(z0)
    coefficients = jnp.stack(values, axis=-1)
    if coefficients.shape[-1] != plan.sparse_relation.capacity:
        raise ValueError("Sparse MNA stamp routes changed; replanning is required.")
    return SparseLinearMap(
        plan.sparse_relation,
        coefficients,
        operator_id=f"{circuit.circuit_id}/mna-sparse-matrix",
    )


def _prepared_operator(
    circuit: NodalCircuit,
    omega: Array,
    plan: MNAPlan,
    /,
) -> tuple[
    DenseLinearOperator | SparseLinearMap,
    Array | None,
    SparseLinearMap | None,
]:
    if plan.policy.assembly == "dense":
        matrix = _assemble_mna(circuit, omega, plan)
        return (
            DenseLinearOperator(
                matrix,
                operator_id=f"{circuit.circuit_id}/mna-matrix",
            ),
            matrix,
            None,
        )
    sparse = _assemble_sparse_mna(circuit, omega, plan)
    return sparse, None, sparse


def prepare_mna(
    circuit: NodalCircuit,
    angular_frequency: ArrayLike,
    plan_or_policy: MNAPlan | MNASolvePolicy | None = None,
    /,
) -> PreparedMNA:
    omega = jnp.asarray(angular_frequency)
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, MNAPlan)
        else plan_mna(circuit, omega, plan_or_policy)
    )
    operator, matrix, sparse = _prepared_operator(circuit, omega, plan)
    linear = prepare_linear(
        LinearSystem(operator, problem_id=f"{circuit.circuit_id}/mna-system"),
        plan.linear_plan,
    )
    prepared_id = canonical_fingerprint({"kind": "prepared-mna", "plan": plan.plan_id})
    return PreparedMNA(
        circuit,
        plan,
        omega,
        matrix,
        sparse,
        linear,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_mna(
    prepared: PreparedMNA,
    circuit: NodalCircuit,
    angular_frequency: ArrayLike,
    /,
) -> PreparedMNA:
    if not isinstance(prepared, PreparedMNA):
        raise TypeError("prepared must be PreparedMNA.")
    omega = jnp.asarray(angular_frequency)
    operator, matrix, sparse = _prepared_operator(circuit, omega, prepared.plan)
    linear = refresh_linear(
        prepared.linear,
        LinearSystem(operator, problem_id=f"{circuit.circuit_id}/mna-system"),
    )
    return PreparedMNA(
        circuit,
        prepared.plan,
        omega,
        matrix,
        sparse,
        linear,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _ordered_mna_incident(
    prepared: PreparedMNA, incident: Array, port_ids: Sequence[str]
) -> Array:
    count = len(prepared.circuit.ports)
    available = tuple(port.port_id for port in prepared.circuit.ports)
    if port_ids:
        ids = tuple(str(value) for value in port_ids)
        if incident.shape[-2] != len(ids) or any(value not in available for value in ids):
            raise ValueError("MNA excitation port selection is invalid.")
        ordered = jnp.zeros(
            incident.shape[:-2] + (count, incident.shape[-1]), dtype=incident.dtype
        )
        for source, port_id in enumerate(ids):
            ordered = ordered.at[..., available.index(port_id), :].set(
                incident[..., source, :]
            )
        incident = ordered
    elif incident.shape[-2] != count:
        raise ValueError("MNA incident port axis has the wrong size.")
    if incident.shape[:-2] == ():
        incident = jnp.broadcast_to(
            incident, prepared.plan.frequency_shape + incident.shape[-2:]
        )
    elif incident.shape[:-2] != prepared.plan.frequency_shape:
        raise ValueError("MNA incident batch must be scalar or match angular_frequency.")
    return incident


def _prepared_mna_dtype(prepared: PreparedMNA, /) -> jnp.dtype:
    if prepared.matrix is not None:
        return prepared.matrix.dtype
    if prepared.sparse_operator is None:
        raise ValueError("Prepared MNA has no assembled operator.")
    return prepared.sparse_operator.coefficients.dtype


def _apply_prepared_mna(prepared: PreparedMNA, value: Array, /) -> Array:
    if prepared.matrix is not None:
        return prepared.matrix @ value
    if prepared.sparse_operator is None:
        raise ValueError("Prepared MNA has no assembled operator.")
    return prepared.sparse_operator.mv(value)


def solve_mna(
    prepared: PreparedMNA,
    incident: ArrayLike,
    /,
    *,
    port_ids: Sequence[str] = (),
) -> MNAResult:
    """Solve terminated external ports and reconstruct Kurokawa outgoing waves."""
    if not isinstance(prepared, PreparedMNA):
        raise TypeError("prepared must be PreparedMNA.")
    incident_ = jnp.asarray(incident).astype(jnp.result_type(incident, jnp.complex128))
    if incident_.ndim < 2:
        raise ValueError("incident must have (..., ports, rhs) shape.")
    incident_ = _ordered_mna_incident(prepared, incident_, port_ids)
    rhs = jnp.zeros(
        prepared.plan.frequency_shape
        + (prepared.plan.cost.total_unknowns, incident_.shape[-1]),
        dtype=jnp.result_type(_prepared_mna_dtype(prepared), incident_),
    )
    for port_index, port in enumerate(prepared.circuit.ports):
        root = jnp.sqrt(jnp.real(port.reference.z0))
        if root.ndim == 0:
            root = jnp.broadcast_to(root, prepared.plan.frequency_shape)
        rhs = rhs.at[..., prepared.plan.port_current_offset + port_index, :].set(
            2.0 * root[..., None] * incident_[..., port_index, :]
        )
    linear_result = solve_linear(prepared.linear, rhs)
    solution = jnp.asarray(linear_result.value)
    node_voltages = solution[..., : len(prepared.plan.node_ids), :]

    def node_voltage(node: NodeId) -> Array:
        index = _node_index(prepared.plan, node)
        return (
            jnp.zeros(solution.shape[:-2] + (solution.shape[-1],), dtype=solution.dtype)
            if index is None
            else node_voltages[..., index, :]
        )

    port_voltages = jnp.stack(
        [
            node_voltage(port.positive) - node_voltage(port.negative)
            for port in prepared.circuit.ports
        ],
        axis=-2,
    )
    port_currents = solution[..., prepared.plan.port_current_offset :, :]
    outgoing_values = []
    powers = []
    for port_index, port in enumerate(prepared.circuit.ports):
        root = jnp.sqrt(jnp.real(port.reference.z0))
        outgoing = (
            port_voltages[..., port_index, :]
            - jnp.conj(port.reference.z0)[..., None] * port_currents[..., port_index, :]
        ) / (2.0 * root[..., None])
        outgoing_values.append(outgoing)
        powers.append(
            jnp.real(
                port_voltages[..., port_index, :]
                * jnp.conj(port_currents[..., port_index, :])
            )
        )
    outgoing = jnp.stack(outgoing_values, axis=-2)
    residual = _apply_prepared_mna(prepared, solution) - rhs
    original = jnp.linalg.norm(residual, axis=(-2, -1))
    scale = jnp.maximum(
        jnp.linalg.norm(rhs, axis=(-2, -1)) + jnp.linalg.norm(solution, axis=(-2, -1)),
        1.0,
    )
    relative = original / scale
    kcl = jnp.linalg.norm(residual[..., : len(prepared.plan.node_ids), :], axis=(-2, -1))
    power_balance = jnp.abs(
        jnp.sum(jnp.abs(incident_) ** 2 - jnp.abs(outgoing) ** 2, axis=(-2, -1))
        - jnp.sum(jnp.stack(powers, axis=-2), axis=(-2, -1))
    )
    finite = jnp.all(jnp.isfinite(solution), axis=(-2, -1)) & jnp.isfinite(relative)
    linear_success = jnp.all(linear_result.status == int(LinearSolveStatus.SUCCESS))
    status = jnp.where(
        ~linear_success,
        int(MNASolveStatus.SINGULAR),
        jnp.where(
            ~jnp.all(finite),
            int(MNASolveStatus.NONFINITE),
            jnp.where(
                jnp.max(relative) > prepared.plan.policy.residual_tolerance,
                int(MNASolveStatus.RESIDUAL_TOLERANCE_NOT_MET),
                int(MNASolveStatus.SUCCESS),
            ),
        ),
    )
    diagnostics = MNADiagnostics(
        status=jnp.asarray(status, dtype=jnp.int32),
        linear_status=jnp.asarray(linear_result.status, dtype=jnp.int32),
        original_residual=original,
        relative_residual=relative,
        kcl_residual=kcl,
        port_power_balance=power_balance,
        finite=finite,
    )
    return MNAResult(
        solution,
        node_voltages,
        port_voltages,
        port_currents,
        outgoing,
        incident_,
        tuple(
            node_voltage(probe.positive) - node_voltage(probe.negative)
            for probe in prepared.circuit.probes
        ),
        diagnostics,
        prepared.numeric_version,
        prepared.plan.node_ids,
        tuple(port.port_id for port in prepared.circuit.ports),
        tuple(probe.probe_id for probe in prepared.circuit.probes),
    )


def _mna_selection(
    selection: Sequence[str | int], available: tuple[str, ...]
) -> tuple[int, ...]:
    indices = tuple(
        int(value) if isinstance(value, int) else available.index(str(value))
        for value in selection
    )
    if any(index < 0 or index >= len(available) for index in indices) or len(
        set(indices)
    ) != len(indices):
        raise ValueError("MNA port selection is invalid or duplicated.")
    return indices


def mna_scattering_submatrix(
    prepared: PreparedMNA,
    input_ports: Sequence[str | int],
    output_ports: Sequence[str | int],
    /,
) -> Array:
    available = tuple(port.port_id for port in prepared.circuit.ports)
    inputs = _mna_selection(input_ports, available)
    outputs = _mna_selection(output_ports, available)
    basis = jnp.zeros(
        (len(available), len(inputs)),
        dtype=_prepared_mna_dtype(prepared),
    )
    for column, index in enumerate(inputs):
        basis = basis.at[index, column].set(1.0)
    result = solve_mna(prepared, basis)
    return result.outgoing[..., jnp.asarray(outputs), :]


def full_mna_scattering_matrix(prepared: PreparedMNA, /) -> Array:
    ports = tuple(range(len(prepared.circuit.ports)))
    return mna_scattering_submatrix(prepared, ports, ports)


__all__ = [
    "AbstractMNAComponent",
    "MNACostEstimate",
    "MNADiagnostics",
    "CircuitInstance",
    "MNAPlan",
    "VoltageProbe",
    "MNAResult",
    "MNASolvePolicy",
    "MNASolveStatus",
    "MNAStamp",
    "NodeId",
    "NodalCircuit",
    "NodalPort",
    "PreparedMNA",
    "full_mna_scattering_matrix",
    "mna_scattering_submatrix",
    "plan_mna",
    "prepare_mna",
    "refresh_mna",
    "solve_mna",
]
