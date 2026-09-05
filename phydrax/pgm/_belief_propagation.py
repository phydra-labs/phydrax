#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..graph import segment_log_normalize, segment_sum
from ._kernel import (
    FactorExecutionEvidence,
    FactorGraphPrecisionPolicy,
    FactorGraphResourcePolicy,
)
from ._model import (
    BinaryCardinalityFactorGroup,
    DenseTableFactorGroup,
    DiscreteFactorGraph,
    EnumeratedFactorGroup,
    factor_count,
    factor_graph_log_score,
    factor_group_capabilities,
    factor_group_cardinality_signature,
    factor_group_dense_tables,
    factor_kernel_id,
    IsingFactorGroup,
    LogicalFactorGroup,
    pack_evidence,
    PottsFactorGroup,
    VariableStateValues,
)
from ._sparse_factor import enumerated_factor_messages, enumerated_joint_scores
from ._structured_messages import (
    cardinality_factor_messages,
    ising_factor_messages,
    logical_factor_messages,
)
from ._types import (
    BeliefPropagationDiagnostics,
    BeliefPropagationStatus,
    FactorGraphProvenance,
)


class SumProductBeliefPropagation(StrictModule):
    """Normalized sum-product updates with support-safe probability relaxation."""

    maximum_steps: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 200,
        relaxation: float = 1.0,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
    ):
        steps, relaxed, absolute, relative = _method_parameters(
            maximum_steps,
            relaxation,
            absolute_tolerance,
            relative_tolerance,
        )
        self.maximum_steps = steps
        self.relaxation = relaxed
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.method_id = "sum-product"


class MaxProductBeliefPropagation(StrictModule):
    """Normalized max-product updates with deterministic support semantics."""

    maximum_steps: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 200,
        relaxation: float = 1.0,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
    ):
        steps, relaxed, absolute, relative = _method_parameters(
            maximum_steps,
            relaxation,
            absolute_tolerance,
            relative_tolerance,
        )
        self.maximum_steps = steps
        self.relaxation = relaxed
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.method_id = "max-product"


BeliefPropagationMethod: TypeAlias = (
    SumProductBeliefPropagation | MaxProductBeliefPropagation
)


class BeliefPropagationSchedulePolicy(StrictModule):
    """Explicit synchronous, Gauss--Seidel, or directed-forest update order."""

    kind: Literal["synchronous", "asynchronous", "forest"] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["synchronous", "asynchronous", "forest"] = "synchronous",
    ):
        if kind not in ("synchronous", "asynchronous", "forest"):
            raise ValueError("Unknown belief-propagation schedule.")
        self.kind = kind
        self.schedule_id = f"belief-propagation-{kind}"


def _method_parameters(maximum_steps, relaxation, absolute_tolerance, relative_tolerance):
    steps = int(maximum_steps)
    relaxed = float(relaxation)
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    if steps < 1:
        raise ValueError("maximum_steps must be positive.")
    if not isfinite(relaxed) or not 0.0 < relaxed <= 1.0:
        raise ValueError("relaxation must lie in (0, 1].")
    if not isfinite(absolute) or absolute < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative.")
    if not isfinite(relative) or relative < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative.")
    return steps, relaxed, absolute, relative


class BeliefPropagationState(StrictModule):
    """Persistent factor-to-variable messages and dynamic unary evidence."""

    messages: Array
    evidence: VariableStateValues
    step_index: Array

    def __init__(
        self,
        messages: ArrayLike,
        evidence: VariableStateValues,
        /,
        *,
        step_index: int | Array = 0,
    ):
        if not isinstance(evidence, VariableStateValues):
            raise TypeError("evidence must be VariableStateValues.")
        values = jnp.asarray(messages)
        if jnp.iscomplexobj(values):
            raise TypeError("Belief-propagation messages must be real-valued.")
        index = jnp.asarray(step_index, dtype=jnp.int32)
        if index.shape != ():
            raise ValueError("step_index must be scalar.")
        self.messages = values
        self.evidence = evidence
        self.step_index = index


class PreparedBeliefPropagation(StrictModule):
    """Topology-fixed flat-message execution plan with refreshable factor tables."""

    graph: DiscreteFactorGraph
    method: BeliefPropagationMethod
    factor_tables: tuple[Array, ...]
    precision: FactorGraphPrecisionPolicy
    resources: FactorGraphResourcePolicy
    factor_evidence: tuple[FactorExecutionEvidence, ...]
    message_variable_state_indices: Array
    state_variable_indices: Array
    variable_degrees: Array
    message_layout: tuple[tuple[tuple[int, int, int, int], ...], ...] = eqx.field(
        static=True
    )
    variable_incidents: tuple[tuple[tuple[int, int, int], ...], ...] = eqx.field(
        static=True
    )
    forest_roots: tuple[int, ...] = eqx.field(static=True)
    decode_steps: tuple[tuple[int, int, int, tuple[int, ...]], ...] = eqx.field(
        static=True
    )
    forest: bool = eqx.field(static=True)
    forest_steps: int = eqx.field(static=True)
    message_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class SumProductBeliefPropagationResult(StrictModule):
    """Variable/factor beliefs, normalizer evidence, and terminal message state."""

    variable_log_probabilities: VariableStateValues
    factor_probabilities: tuple[Array, ...]
    log_normalizer: Array
    state: BeliefPropagationState
    status: Array
    valid: Array
    converged: Array
    diagnostics: BeliefPropagationDiagnostics
    provenance: FactorGraphProvenance
    marginals_exact: bool = eqx.field(static=True)
    log_normalizer_exact: bool = eqx.field(static=True)
    log_normalizer_kind: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(BeliefPropagationStatus.SUCCESS))


class MaxProductBeliefPropagationResult(StrictModule):
    """Max-marginals, local modes, and exact forest MAP evidence when available."""

    variable_max_marginals: VariableStateValues
    local_modes: Array
    map_assignment: Array
    map_log_score: Array
    state: BeliefPropagationState
    status: Array
    valid: Array
    converged: Array
    optimal: Array
    diagnostics: BeliefPropagationDiagnostics
    provenance: FactorGraphProvenance
    map_available: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(BeliefPropagationStatus.SUCCESS))


BeliefPropagationResult: TypeAlias = (
    SumProductBeliefPropagationResult | MaxProductBeliefPropagationResult
)


def _forest_metadata(graph: DiscreteFactorGraph, /):
    variable_count = graph.num_variables
    factor_total = graph.num_factors
    node_count = variable_count + factor_total
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    factor_lookup: list[tuple[int, int]] = []
    factor_global = 0
    for group_index, scope in enumerate(graph.factor_scopes):
        scope_host = np.asarray(scope, dtype=np.int32)
        for local_factor, variables in enumerate(scope_host):
            factor_node = variable_count + factor_global
            factor_lookup.append((group_index, local_factor))
            for variable in variables:
                adjacency[int(variable)].append(factor_node)
                adjacency[factor_node].append(int(variable))
            factor_global += 1

    parent = list(range(node_count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    forest = True
    for source, neighbors in enumerate(adjacency):
        for target in neighbors:
            if source >= target:
                continue
            root_source, root_target = find(source), find(target)
            if root_source == root_target:
                forest = False
            else:
                parent[root_target] = root_source

    roots: list[int] = []
    decode: list[tuple[int, int, int, tuple[int, ...]]] = []
    visited: set[int] = set()
    max_diameter = 0
    if forest:
        for variable in range(variable_count):
            if variable in visited:
                continue
            component: list[int] = []
            queue = deque([variable])
            visited.add(variable)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            roots.append(variable)
            bfs_parent = {variable: -1}
            queue = deque([variable])
            factor_order: list[int] = []
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor in bfs_parent:
                        continue
                    bfs_parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor >= variable_count:
                        factor_order.append(neighbor)
            for factor_node in factor_order:
                parent_variable = bfs_parent[factor_node]
                children = tuple(
                    node for node in adjacency[factor_node] if node != parent_variable
                )
                group_index, local_factor = factor_lookup[factor_node - variable_count]
                decode.append((group_index, local_factor, parent_variable, children))

            def distances(start):
                result = {start: 0}
                pending = deque([start])
                while pending:
                    node = pending.popleft()
                    for neighbor in adjacency[node]:
                        if neighbor not in result:
                            result[neighbor] = result[node] + 1
                            pending.append(neighbor)
                return result

            if component:
                first = max(distances(component[0]), key=distances(component[0]).get)
                max_diameter = max(max_diameter, max(distances(first).values()))
        for factor_node in range(variable_count, node_count):
            if factor_node not in visited:
                visited.add(factor_node)
    edge_updates = sum(len(adjacency[node]) for node in range(variable_count, node_count))
    return forest, tuple(roots), tuple(decode), max(1, edge_updates)


def prepare_belief_propagation(
    graph: DiscreteFactorGraph,
    method: BeliefPropagationMethod | None = None,
    /,
    *,
    max_factor_configurations: int = 65_536,
    precision: FactorGraphPrecisionPolicy | None = None,
    resources: FactorGraphResourcePolicy | None = None,
) -> PreparedBeliefPropagation:
    """Compile flat message routes and dense factor kernels for one fixed topology."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    selected = SumProductBeliefPropagation() if method is None else method
    if not isinstance(
        selected, (SumProductBeliefPropagation, MaxProductBeliefPropagation)
    ):
        raise TypeError("method must be sum-product or max-product BP.")
    precision_ = FactorGraphPrecisionPolicy() if precision is None else precision
    resources_ = FactorGraphResourcePolicy() if resources is None else resources
    if not isinstance(precision_, FactorGraphPrecisionPolicy):
        raise TypeError("precision must be FactorGraphPrecisionPolicy or None.")
    if not isinstance(resources_, FactorGraphResourcePolicy):
        raise TypeError("resources must be FactorGraphResourcePolicy or None.")
    cap = min(int(max_factor_configurations), resources_.maximum_configurations)
    if cap < 1:
        raise ValueError("max_factor_configurations must be positive.")

    tables: list[Array] = []
    execution_evidence: list[FactorExecutionEvidence] = []
    layouts: list[tuple[tuple[int, int, int, int], ...]] = []
    message_indices: list[np.ndarray] = []
    degrees = np.zeros((graph.num_variables,), dtype=np.int32)
    incidents: list[list[tuple[int, int, int]]] = [[] for _ in range(graph.num_variables)]
    offset = 0
    state_offsets = np.asarray(graph.variable_state_offsets, dtype=np.int32)
    dense_total = 0
    for group_index, scope in enumerate(graph.factor_scopes):
        signature = factor_group_cardinality_signature(graph, group_index)
        group = graph.factor_groups[group_index]
        capabilities = factor_group_capabilities(group)
        if isinstance(selected, SumProductBeliefPropagation) and not (
            capabilities.sum_product and capabilities.factor_beliefs
        ):
            raise ValueError(
                f"Factor group {group_index} does not support sum-product beliefs."
            )
        if isinstance(selected, MaxProductBeliefPropagation) and not (
            capabilities.max_product
        ):
            raise ValueError(f"Factor group {group_index} does not support max-product.")
        dense_configurations = prod(signature)
        represented = (
            int(group.configurations.shape[0])
            if isinstance(group, EnumeratedFactorGroup)
            else dense_configurations
        )
        if represented > cap:
            raise ValueError(
                f"Factor group {group_index} requires {represented} represented "
                f"configurations, exceeding max_factor_configurations={cap}."
            )
        dense_elements = (
            0
            if isinstance(group, EnumeratedFactorGroup)
            else int(scope.shape[0]) * dense_configurations
        )
        dense_total += dense_elements
        if dense_total > resources_.maximum_dense_elements:
            raise ValueError(
                f"Cumulative factor dense size {dense_total} exceeds "
                f"maximum_dense_elements={resources_.maximum_dense_elements}."
            )
        table = precision_.evaluation(
            group.log_potentials
            if isinstance(group, EnumeratedFactorGroup)
            else factor_group_dense_tables(graph, group_index)
        )
        tables.append(table)
        scope_host = np.asarray(scope, dtype=np.int32)
        group_layout: list[tuple[int, int, int, int]] = []
        for position, cardinality in enumerate(signature):
            count = int(scope.shape[0])
            start = offset
            stop = start + count * cardinality
            if stop > resources_.maximum_message_entries:
                raise ValueError(
                    f"Message entries {stop} exceed maximum_message_entries="
                    f"{resources_.maximum_message_entries}."
                )
            group_layout.append((start, stop, count, cardinality))
            if count:
                variables = scope_host[:, position]
                indices = state_offsets[variables, None] + np.arange(cardinality)[None, :]
                message_indices.append(indices.reshape((-1,)).astype(np.int32))
                np.add.at(degrees, variables, 1)
                for local_factor, variable in enumerate(variables):
                    incidents[int(variable)].append((group_index, local_factor, position))
            offset = stop
        layouts.append(tuple(group_layout))
        message_entries = sum(stop - start for start, stop, _count, _card in group_layout)
        execution_evidence.append(
            FactorExecutionEvidence(
                capabilities,
                represented_configurations=represented,
                dense_elements=dense_elements,
                message_entries=message_entries,
                work_estimate=(
                    factor_count(group) * represented * max(len(signature), 1)
                ),
                workspace_elements=max(
                    dense_elements,
                    factor_count(group) * represented,
                ),
                kernel_id=factor_kernel_id(group),
            )
        )
    if offset > resources_.maximum_message_entries:
        raise ValueError(
            f"Message entries {offset} exceed maximum_message_entries="
            f"{resources_.maximum_message_entries}."
        )
    message_state_indices = (
        np.concatenate(message_indices)
        if message_indices
        else np.zeros((0,), dtype=np.int32)
    )
    state_variable_indices = np.repeat(
        np.arange(graph.num_variables, dtype=np.int32),
        np.asarray(graph.cardinalities, dtype=np.int32),
    )
    forest, roots, decode, forest_steps = _forest_metadata(graph)
    plan_id = canonical_fingerprint(
        {
            "kind": "belief-propagation-plan",
            "structure_id": graph.structure_id,
            "method_id": selected.method_id,
            "maximum_steps": selected.maximum_steps,
            "relaxation": selected.relaxation,
            "absolute_tolerance": selected.absolute_tolerance,
            "relative_tolerance": selected.relative_tolerance,
            "max_factor_configurations": cap,
            "message_count": offset,
            "forest": forest,
            "precision": precision_.policy_id,
            "resources": resources_.policy_id,
        }
    )
    return PreparedBeliefPropagation(
        graph=graph,
        method=selected,
        factor_tables=tuple(tables),
        precision=precision_,
        resources=resources_,
        factor_evidence=tuple(execution_evidence),
        message_variable_state_indices=jnp.asarray(message_state_indices),
        state_variable_indices=jnp.asarray(state_variable_indices),
        variable_degrees=jnp.asarray(degrees),
        message_layout=tuple(layouts),
        variable_incidents=tuple(tuple(value) for value in incidents),
        forest_roots=roots,
        decode_steps=decode,
        forest=forest,
        forest_steps=forest_steps,
        message_count=offset,
        plan_id=plan_id,
    )


def refresh_belief_propagation(
    prepared: PreparedBeliefPropagation,
    graph: DiscreteFactorGraph,
    /,
) -> PreparedBeliefPropagation:
    """Replace compatible numeric factor tables without rebuilding topology routes."""
    if not isinstance(prepared, PreparedBeliefPropagation):
        raise TypeError("prepared must be PreparedBeliefPropagation.")
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    if graph.structure_id != prepared.graph.structure_id:
        raise ValueError("Refreshed graph structure does not match the prepared plan.")
    if graph.parameter_signature != prepared.graph.parameter_signature:
        raise ValueError("Refreshed graph parameter signature does not match the plan.")
    tables = tuple(
        prepared.precision.evaluation(
            group.log_potentials
            if isinstance(group, EnumeratedFactorGroup)
            else factor_group_dense_tables(graph, index)
        )
        for index, group in enumerate(graph.factor_groups)
    )
    updated = eqx.tree_at(lambda value: value.graph, prepared, graph)
    return eqx.tree_at(lambda value: value.factor_tables, updated, tables)


def replace_belief_propagation_tables(
    prepared: PreparedBeliefPropagation,
    factor_tables: tuple[Array, ...],
    /,
) -> PreparedBeliefPropagation:
    """JIT-safe numeric replacement on prepared dense/Potts support.

    Shape and dtype checks are static; numeric feasibility is reported by the
    executor. Structured kernels must instead refresh their native parameters.
    No support, identity, routing, or initial message state is rebuilt.
    """
    if len(factor_tables) != len(prepared.factor_tables):
        raise ValueError("Numeric factor table count differs from prepared support.")
    tables = []
    groups = []
    for group, original, value in zip(
        prepared.graph.factor_groups, prepared.factor_tables, factor_tables
    ):
        if not isinstance(group, (DenseTableFactorGroup, PottsFactorGroup)):
            raise TypeError("Numeric table replacement requires dense/Potts factors.")
        array = jnp.asarray(value)
        if array.shape != original.shape or jnp.iscomplexobj(array):
            raise ValueError("Numeric factor tables must retain real prepared shapes.")
        array = prepared.precision.evaluation(array)
        tables.append(array)
        groups.append(eqx.tree_at(lambda item: item.log_potentials, group, array))
    graph = eqx.tree_at(lambda item: item.factor_groups, prepared.graph, tuple(groups))
    return eqx.tree_at(
        lambda item: (item.graph, item.factor_tables),
        prepared,
        (graph, tuple(tables)),
    )


def initialize_belief_propagation(
    prepared: PreparedBeliefPropagation,
    /,
    *,
    evidence: VariableStateValues | ArrayLike | None = None,
    messages: ArrayLike | None = None,
) -> BeliefPropagationState:
    """Initialize zero or caller-supplied messages and validated unary evidence."""
    if not isinstance(prepared, PreparedBeliefPropagation):
        raise TypeError("prepared must be PreparedBeliefPropagation.")
    graph = prepared.graph
    evidence_values = (
        pack_evidence(graph)
        if evidence is None
        else evidence
        if isinstance(evidence, VariableStateValues)
        else pack_evidence(graph, evidence)
    )
    if evidence_values.structure_id != graph.structure_id:
        raise ValueError("Evidence structure does not match the prepared graph.")
    evidence_values = VariableStateValues(
        prepared.precision.evaluation(evidence_values.values),
        structure_id=graph.structure_id,
    )
    dtype = prepared.precision.accumulation_dtype
    initial = (
        jnp.zeros((prepared.message_count,), dtype=dtype)
        if messages is None
        else jnp.asarray(messages, dtype=dtype)
    )
    if initial.shape != (prepared.message_count,):
        raise ValueError(
            f"messages must have shape ({prepared.message_count},); got {initial.shape}."
        )
    if jnp.iscomplexobj(initial):
        raise TypeError("messages must be real-valued.")
    if bool(jnp.any(jnp.isnan(initial) | jnp.isposinf(initial))):
        raise ValueError("messages may contain finite values and -inf only.")
    return BeliefPropagationState(initial, evidence_values)


def _variable_to_factor(
    prepared: PreparedBeliefPropagation,
    messages: Array,
    evidence: Array,
    /,
) -> Array:
    messages = prepared.precision.accumulation(messages)
    evidence = prepared.precision.accumulation(evidence)
    state_count = int(prepared.state_variable_indices.shape[0])
    indices = prepared.message_variable_state_indices
    finite = jnp.isfinite(messages)
    finite_sums = segment_sum(jnp.where(finite, messages, 0.0), indices, state_count)
    impossible = segment_sum((~finite).astype(jnp.int32), indices, state_count)
    evidence_finite = jnp.isfinite(evidence)
    finite_sums = finite_sums + jnp.where(evidence_finite, evidence, 0.0)
    impossible = impossible + (~evidence_finite).astype(jnp.int32)
    excluded_sums = finite_sums[indices] - jnp.where(finite, messages, 0.0)
    excluded_impossible = impossible[indices] - (~finite).astype(jnp.int32)
    return jnp.where(excluded_impossible > 0, -jnp.inf, excluded_sums)


def _broadcast_message(values: Array, position: int, arity: int, /) -> Array:
    shape = [int(values.shape[0])] + [1] * arity
    shape[position + 1] = int(values.shape[1])
    return values.reshape(tuple(shape))


def _factor_update(
    prepared: PreparedBeliefPropagation,
    variable_to_factor: Array,
    /,
) -> tuple[Array, Array]:
    sum_product = isinstance(prepared.method, SumProductBeliefPropagation)
    output = jnp.full_like(variable_to_factor, -jnp.inf)
    all_feasible = jnp.asarray(True)
    for group_index, (table, layout) in enumerate(
        zip(prepared.factor_tables, prepared.message_layout)
    ):
        table = prepared.precision.accumulation(table)
        arity = len(layout)
        incoming = [
            variable_to_factor[start:stop].reshape((count, cardinality))
            for start, stop, count, cardinality in layout
        ]
        group = prepared.graph.factor_groups[group_index]
        if isinstance(group, EnumeratedFactorGroup):
            sparse_outputs = enumerated_factor_messages(
                group,
                table,
                tuple(incoming),
                factor_group_cardinality_signature(prepared.graph, group_index),
                mode="sum" if sum_product else "max",
            )
            for values, (start, stop, _count, _cardinality) in zip(
                sparse_outputs, layout
            ):
                output = output.at[start:stop].set(values.reshape((-1,)))
                all_feasible = all_feasible & jnp.all(
                    jnp.any(jnp.isfinite(values), axis=-1)
                )
            continue
        structured_outputs = None
        if isinstance(group, IsingFactorGroup):
            structured_outputs = ising_factor_messages(
                group,
                tuple(incoming),
                mode="sum" if sum_product else "max",
            )
        elif isinstance(group, LogicalFactorGroup):
            structured_outputs = logical_factor_messages(
                group,
                tuple(incoming),
                mode="sum" if sum_product else "max",
            )
        elif isinstance(group, BinaryCardinalityFactorGroup):
            structured_outputs = cardinality_factor_messages(
                group,
                tuple(incoming),
                mode="sum" if sum_product else "max",
            )
        if structured_outputs is not None:
            for values, (start, stop, _count, _cardinality) in zip(
                structured_outputs, layout
            ):
                output = output.at[start:stop].set(values.reshape((-1,)))
                all_feasible = all_feasible & jnp.all(
                    jnp.any(jnp.isfinite(values), axis=-1)
                )
            continue
        for position, (start, stop, count, cardinality) in enumerate(layout):
            joint = table
            for other, values in enumerate(incoming):
                if other != position:
                    joint = joint + _broadcast_message(values, other, arity)
            axes = tuple(axis for axis in range(1, arity + 1) if axis != position + 1)
            reduced = (
                jsp.special.logsumexp(joint, axis=axes)
                if sum_product and axes
                else jnp.max(joint, axis=axes)
                if axes
                else joint
            )
            maxima = jnp.max(reduced, axis=-1, keepdims=True)
            feasible = jnp.isfinite(maxima)
            normalized = jnp.where(feasible, reduced - maxima, -jnp.inf)
            output = output.at[start:stop].set(normalized.reshape((-1,)))
            all_feasible = all_feasible & jnp.all(feasible)
    return output, all_feasible


def _normalize_flat_messages(
    prepared: PreparedBeliefPropagation,
    messages: Array,
    /,
) -> Array:
    output = messages
    for layout in prepared.message_layout:
        for start, stop, count, cardinality in layout:
            values = output[start:stop].reshape((count, cardinality))
            maxima = jnp.max(values, axis=-1, keepdims=True)
            values = jnp.where(jnp.isfinite(maxima), values - maxima, -jnp.inf)
            output = output.at[start:stop].set(values.reshape((-1,)))
    return output


def _relax_messages(
    prepared: PreparedBeliefPropagation,
    current: Array,
    candidate: Array,
    /,
    *,
    force_full: bool = False,
) -> Array:
    relaxation = 1.0 if force_full else prepared.method.relaxation
    if relaxation == 1.0:
        return candidate
    if isinstance(prepared.method, SumProductBeliefPropagation):
        old_term = jnp.log1p(-relaxation) + current
        new_term = jnp.log(relaxation) + candidate
        mixed = jnp.logaddexp(old_term, new_term)
    else:
        both = jnp.isfinite(current) & jnp.isfinite(candidate)
        interpolated = current + relaxation * (candidate - current)
        mixed = jnp.where(both, interpolated, candidate)
    return _normalize_flat_messages(prepared, mixed)


def _message_residual(current: Array, updated: Array, /) -> tuple[Array, Array]:
    if int(current.shape[0]) == 0:
        return jnp.asarray(0.0), jnp.asarray(False)
    current_support = jnp.isfinite(current)
    updated_support = jnp.isfinite(updated)
    support_changed = jnp.any(current_support != updated_support)
    differences = jnp.where(
        current_support & updated_support,
        jnp.abs(updated - current),
        0.0,
    )
    residual = jnp.max(differences)
    return jnp.where(support_changed, jnp.inf, residual), support_changed


def _bp_step(
    prepared: PreparedBeliefPropagation,
    messages: Array,
    evidence: Array,
    /,
    *,
    force_full: bool = False,
):
    variable_to_factor = _variable_to_factor(prepared, messages, evidence)
    candidate, feasible = _factor_update(prepared, variable_to_factor)
    updated = _relax_messages(
        prepared,
        messages,
        candidate,
        force_full=force_full,
    )
    residual, support_changed = _message_residual(messages, updated)
    finite = ~jnp.any(jnp.isnan(updated) | jnp.isposinf(updated))
    return updated, residual, support_changed, feasible, finite


def _forest_edge_message(
    prepared: PreparedBeliefPropagation,
    messages: Array,
    evidence: Array,
    group_index: int,
    local_factor: int,
    target_position: int,
    /,
) -> tuple[Array, Array, Array]:
    graph = prepared.graph
    scope = np.asarray(graph.factor_scopes[group_index], dtype=np.int32)[local_factor]
    offsets = np.asarray(graph.variable_state_offsets, dtype=np.int32)
    incoming = []
    for position, variable_value in enumerate(scope):
        variable = int(variable_value)
        values = evidence[offsets[variable] : offsets[variable + 1]]
        for (
            incident_group,
            incident_factor,
            incident_position,
        ) in prepared.variable_incidents[variable]:
            if (
                incident_group == group_index
                and incident_factor == local_factor
                and incident_position == position
            ):
                continue
            start, _stop, _count, cardinality = prepared.message_layout[incident_group][
                incident_position
            ]
            edge_start = start + incident_factor * cardinality
            values = values + messages[edge_start : edge_start + cardinality]
        incoming.append(values)

    group = graph.factor_groups[group_index]
    sum_product = isinstance(prepared.method, SumProductBeliefPropagation)
    if isinstance(group, EnumeratedFactorGroup):
        configurations = group.configurations
        joint = prepared.factor_tables[group_index][local_factor]
        for position, values in enumerate(incoming):
            if position != target_position:
                joint = joint + values[configurations[:, position]]
        target_cardinality = int(incoming[target_position].shape[0])
        reduced = []
        for state in range(target_cardinality):
            candidates = jnp.where(
                configurations[:, target_position] == state,
                joint,
                -jnp.inf,
            )
            reduced.append(
                jsp.special.logsumexp(candidates) if sum_product else jnp.max(candidates)
            )
        values = jnp.stack(reduced)
    else:
        joint = prepared.factor_tables[group_index][local_factor]
        arity = len(incoming)
        for position, incoming_values in enumerate(incoming):
            if position != target_position:
                shape = [1] * arity
                shape[position] = int(incoming_values.shape[0])
                joint = joint + incoming_values.reshape(tuple(shape))
        axes = tuple(axis for axis in range(arity) if axis != target_position)
        values = (
            jsp.special.logsumexp(joint, axis=axes)
            if sum_product and axes
            else jnp.max(joint, axis=axes)
            if axes
            else joint
        )
    maximum = jnp.max(values)
    feasible = jnp.isfinite(maximum)
    normalized = jnp.where(feasible, values - maximum, -jnp.inf)
    finite = ~jnp.any(jnp.isnan(normalized) | jnp.isposinf(normalized))
    return normalized, feasible, finite


def _set_forest_edge_message(
    prepared: PreparedBeliefPropagation,
    messages: Array,
    evidence: Array,
    group_index: int,
    local_factor: int,
    target_variable: int,
    /,
) -> tuple[Array, Array, Array]:
    scope = np.asarray(prepared.graph.factor_scopes[group_index], dtype=np.int32)[
        local_factor
    ]
    target_position = int(np.flatnonzero(scope == target_variable)[0])
    values, feasible, finite = _forest_edge_message(
        prepared,
        messages,
        evidence,
        group_index,
        local_factor,
        target_position,
    )
    start, _stop, _count, cardinality = prepared.message_layout[group_index][
        target_position
    ]
    edge_start = start + local_factor * cardinality
    return (
        messages.at[edge_start : edge_start + cardinality].set(values),
        feasible,
        finite,
    )


def _run_forest(prepared, state):
    original = state.messages
    messages = original
    feasible = jnp.asarray(True)
    finite = jnp.asarray(True)
    for group_index, local_factor, parent_variable, _children in reversed(
        prepared.decode_steps
    ):
        messages, edge_feasible, edge_finite = _set_forest_edge_message(
            prepared,
            messages,
            state.evidence.values,
            group_index,
            local_factor,
            parent_variable,
        )
        feasible = feasible & edge_feasible
        finite = finite & edge_finite
    for group_index, local_factor, _parent_variable, children in prepared.decode_steps:
        for child_variable in children:
            messages, edge_feasible, edge_finite = _set_forest_edge_message(
                prepared,
                messages,
                state.evidence.values,
                group_index,
                local_factor,
                child_variable,
            )
            feasible = feasible & edge_feasible
            finite = finite & edge_finite

    offsets = np.asarray(prepared.graph.variable_state_offsets, dtype=np.int32)
    for root in prepared.forest_roots:
        values = state.evidence.values[offsets[root] : offsets[root + 1]]
        for group_index, local_factor, position in prepared.variable_incidents[root]:
            start, _stop, _count, cardinality = prepared.message_layout[group_index][
                position
            ]
            edge_start = start + local_factor * cardinality
            values = values + messages[edge_start : edge_start + cardinality]
        feasible = feasible & jnp.any(jnp.isfinite(values))
    initial_residual, support_changed = _message_residual(original, messages)
    valid = feasible & finite
    status = jnp.where(
        ~finite,
        int(BeliefPropagationStatus.NONFINITE_MESSAGE),
        jnp.where(
            ~feasible,
            int(BeliefPropagationStatus.INFEASIBLE),
            int(BeliefPropagationStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return (
        BeliefPropagationState(
            messages,
            state.evidence,
            step_index=state.step_index + prepared.forest_steps,
        ),
        status,
        valid,
        valid,
        initial_residual,
        jnp.asarray(0.0, dtype=messages.dtype),
        support_changed.astype(jnp.int32),
        jnp.asarray(prepared.forest_steps, dtype=jnp.int32),
    )


def _asynchronous_bp_step(
    prepared: PreparedBeliefPropagation,
    messages: Array,
    evidence: Array,
    /,
):
    original = messages
    feasible = jnp.asarray(True)
    finite = jnp.asarray(True)
    for layout in prepared.message_layout:
        for start, stop, _count, _cardinality in layout:
            variable_to_factor = _variable_to_factor(prepared, messages, evidence)
            candidate, candidate_feasible = _factor_update(
                prepared,
                variable_to_factor,
            )
            relaxed = _relax_messages(prepared, messages, candidate)
            messages = messages.at[start:stop].set(relaxed[start:stop])
            feasible = feasible & candidate_feasible
            finite = finite & ~jnp.any(
                jnp.isnan(relaxed[start:stop]) | jnp.isposinf(relaxed[start:stop])
            )
    residual, support_changed = _message_residual(original, messages)
    return messages, residual, support_changed, feasible, finite


def _run_loopy(prepared, state, /, *, asynchronous: bool = False):
    maximum_steps = prepared.method.maximum_steps

    def body(carry, _):
        messages, active, initial, residual, support_count, iterations, status = carry
        updated, trial_residual, support_changed, feasible, finite = (
            _asynchronous_bp_step(
                prepared,
                messages,
                state.evidence.values,
            )
            if asynchronous
            else _bp_step(
                prepared,
                messages,
                state.evidence.values,
            )
        )
        first = iterations == 0
        next_initial = jnp.where(first & active, trial_residual, initial)
        threshold = (
            prepared.method.absolute_tolerance
            + prepared.method.relative_tolerance
            * jnp.maximum(jnp.where(jnp.isfinite(next_initial), next_initial, 1.0), 1.0)
        )
        converged = feasible & finite & (trial_residual <= threshold)
        next_status = jnp.where(
            ~finite,
            int(BeliefPropagationStatus.NONFINITE_MESSAGE),
            jnp.where(
                ~feasible,
                int(BeliefPropagationStatus.INFEASIBLE),
                jnp.where(
                    converged,
                    int(BeliefPropagationStatus.SUCCESS),
                    int(BeliefPropagationStatus.MAXIMUM_STEPS_REACHED),
                ),
            ),
        ).astype(jnp.int32)
        take = active
        return (
            jnp.where(take, updated, messages),
            active & finite & feasible & ~converged,
            jnp.where(take, next_initial, initial),
            jnp.where(take, trial_residual, residual),
            support_count + (take & support_changed).astype(jnp.int32),
            iterations + take.astype(jnp.int32),
            jnp.where(take, next_status, status),
        ), None

    initial = (
        state.messages,
        jnp.asarray(True),
        jnp.asarray(jnp.inf, dtype=state.messages.dtype),
        jnp.asarray(jnp.inf, dtype=state.messages.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(int(BeliefPropagationStatus.MAXIMUM_STEPS_REACHED), dtype=jnp.int32),
    )
    final, _ = jax.lax.scan(body, initial, xs=None, length=maximum_steps)
    messages, active, initial_residual, residual, support_count, iterations, status = (
        final
    )
    del active
    valid = (status != int(BeliefPropagationStatus.INFEASIBLE)) & (
        status != int(BeliefPropagationStatus.NONFINITE_MESSAGE)
    )
    converged = status == int(BeliefPropagationStatus.SUCCESS)
    return (
        BeliefPropagationState(
            messages,
            state.evidence,
            step_index=state.step_index + iterations,
        ),
        status,
        valid,
        converged,
        initial_residual,
        residual,
        support_count,
        iterations,
    )


def _variable_log_beliefs(prepared, state):
    graph = prepared.graph
    messages = state.messages
    indices = prepared.message_variable_state_indices
    finite = jnp.isfinite(messages)
    sums = segment_sum(
        jnp.where(finite, messages, 0.0),
        indices,
        int(prepared.state_variable_indices.shape[0]),
    )
    impossible = segment_sum(
        (~finite).astype(jnp.int32),
        indices,
        int(prepared.state_variable_indices.shape[0]),
    )
    evidence = state.evidence.values
    evidence_finite = jnp.isfinite(evidence)
    sums = sums + jnp.where(evidence_finite, evidence, 0.0)
    impossible = impossible + (~evidence_finite).astype(jnp.int32)
    raw = jnp.where(impossible > 0, -jnp.inf, sums)
    return segment_log_normalize(
        raw,
        prepared.state_variable_indices,
        graph.num_variables,
    )


def _factor_joint_scores(prepared, state):
    vtof = _variable_to_factor(prepared, state.messages, state.evidence.values)
    outputs: list[Array] = []
    for group_index, (table, layout) in enumerate(
        zip(prepared.factor_tables, prepared.message_layout)
    ):
        incoming = tuple(
            vtof[start:stop].reshape((count, cardinality))
            for start, stop, count, cardinality in layout
        )
        group = prepared.graph.factor_groups[group_index]
        if isinstance(group, EnumeratedFactorGroup):
            outputs.append(enumerated_joint_scores(group, table, incoming))
            continue
        arity = len(layout)
        joint = prepared.precision.accumulation(table)
        for position, values in enumerate(incoming):
            joint = joint + _broadcast_message(values, position, arity)
        outputs.append(joint)
    return tuple(outputs), vtof


def _factor_probabilities(prepared, state):
    joints, _ = _factor_joint_scores(prepared, state)
    outputs: list[Array] = []
    for joint in joints:
        axes = tuple(range(1, joint.ndim))
        normalizer = jsp.special.logsumexp(joint, axis=axes, keepdims=True)
        outputs.append(
            jnp.where(jnp.isfinite(normalizer), jnp.exp(joint - normalizer), 0.0)
        )
    return tuple(outputs)


def _bethe_log_normalizer(
    prepared, state, variable_log_probabilities, factor_probabilities
):
    variable_log_probabilities = prepared.precision.accumulation(
        variable_log_probabilities
    )
    graph = prepared.graph
    variable_probabilities = jnp.exp(variable_log_probabilities)
    variable_entropy_terms = -variable_probabilities * jnp.where(
        variable_probabilities > 0, variable_log_probabilities, 0.0
    )
    variable_entropies = segment_sum(
        variable_entropy_terms,
        prepared.state_variable_indices,
        graph.num_variables,
    )
    evidence = prepared.precision.accumulation(state.evidence.values)
    safe_evidence = jnp.where(jnp.isfinite(evidence), evidence, 0.0)
    expected_evidence = jnp.sum(variable_probabilities * safe_evidence)
    factor_energy = jnp.asarray(0.0, dtype=variable_probabilities.dtype)
    factor_entropy = jnp.asarray(0.0, dtype=variable_probabilities.dtype)
    for table, probabilities in zip(prepared.factor_tables, factor_probabilities):
        table = prepared.precision.accumulation(table)
        safe_table = jnp.where(jnp.isfinite(table), table, 0.0)
        factor_energy = factor_energy + jnp.sum(probabilities * safe_table)
        factor_entropy = factor_entropy - jnp.sum(
            probabilities * jnp.log(jnp.where(probabilities > 0, probabilities, 1.0))
        )
    variable_correction = jnp.sum((1 - prepared.variable_degrees) * variable_entropies)
    return factor_energy + expected_evidence + factor_entropy + variable_correction


def _local_modes(graph, values):
    modes: list[Array] = []
    offsets = np.asarray(graph.variable_state_offsets)
    for variable in range(graph.num_variables):
        modes.append(jnp.argmax(values[offsets[variable] : offsets[variable + 1]]))
    return (
        jnp.stack(modes).astype(jnp.int32) if modes else jnp.zeros((0,), dtype=jnp.int32)
    )


def _decode_forest_map(prepared, state, max_marginals):
    graph = prepared.graph
    assignment = _local_modes(graph, max_marginals)
    joints, _ = _factor_joint_scores(prepared, state)
    for group_index, local_factor, parent_variable, children in prepared.decode_steps:
        scope = np.asarray(graph.factor_scopes[group_index][local_factor], dtype=np.int32)
        parent_position = int(np.nonzero(scope == parent_variable)[0][0])
        group = graph.factor_groups[group_index]
        joint = joints[group_index][local_factor]
        parent_state = assignment[parent_variable]
        if isinstance(group, EnumeratedFactorGroup):
            valid = group.configurations[:, parent_position] == parent_state
            config_index = jnp.argmax(jnp.where(valid, joint, -jnp.inf))
            configuration = group.configurations[config_index]
        else:
            signature = factor_group_cardinality_signature(graph, group_index)
            parent_axis = jnp.arange(signature[parent_position], dtype=jnp.int32)
            mask_shape = [1] * len(signature)
            mask_shape[parent_position] = signature[parent_position]
            mask = (parent_axis == parent_state).reshape(tuple(mask_shape))
            masked = jnp.where(mask, joint, -jnp.inf)
            flat_index = jnp.argmax(masked.reshape((-1,)))
            configuration = jnp.stack(jnp.unravel_index(flat_index, signature)).astype(
                jnp.int32
            )
        for child in children:
            child_position = int(np.nonzero(scope == child)[0][0])
            assignment = assignment.at[child].set(configuration[child_position])
    evidence_indices = graph.variable_state_offsets[:-1] + assignment
    score = factor_graph_log_score(graph, assignment) + jnp.sum(
        state.evidence.values[evidence_indices]
    )
    return assignment, score


def run_belief_propagation(
    prepared: PreparedBeliefPropagation,
    state: BeliefPropagationState,
    /,
    *,
    schedule: BeliefPropagationSchedulePolicy | None = None,
) -> BeliefPropagationResult:
    """Run directed-forest, synchronous, or Gauss--Seidel belief propagation."""
    if not isinstance(prepared, PreparedBeliefPropagation):
        raise TypeError("prepared must be PreparedBeliefPropagation.")
    if not isinstance(state, BeliefPropagationState):
        raise TypeError("state must be BeliefPropagationState.")
    if state.messages.shape != (prepared.message_count,):
        raise ValueError("State message shape does not match the prepared plan.")
    if state.evidence.structure_id != prepared.graph.structure_id:
        raise ValueError("State evidence does not match the prepared graph.")
    selected = (
        BeliefPropagationSchedulePolicy("forest" if prepared.forest else "synchronous")
        if schedule is None
        else schedule
    )
    if not isinstance(selected, BeliefPropagationSchedulePolicy):
        raise TypeError("schedule must be BeliefPropagationSchedulePolicy or None.")
    if selected.kind == "forest" and not prepared.forest:
        raise ValueError("The forest schedule requires an acyclic factor graph.")
    run = (
        _run_forest(prepared, state)
        if selected.kind == "forest"
        else _run_loopy(
            prepared,
            state,
            asynchronous=selected.kind == "asynchronous",
        )
    )
    (
        final_state,
        status,
        valid,
        converged,
        initial,
        residual,
        support_count,
        iterations,
    ) = run
    evaluation_multiplier = (
        1
        if selected.kind == "forest"
        else sum(len(layout) for layout in prepared.message_layout)
        if selected.kind == "asynchronous"
        else prepared.graph.num_factors
    )
    diagnostics = BeliefPropagationDiagnostics(
        initial_residual=initial,
        final_residual=residual,
        iterations=iterations,
        support_changes=support_count,
        factor_evaluations=iterations * evaluation_multiplier,
    )
    provenance = FactorGraphProvenance(
        structure_id=prepared.graph.structure_id,
        plan_id=prepared.plan_id,
        method_id=f"{prepared.method.method_id}:{selected.schedule_id}",
        implementation=f"flat-ragged-jax-{selected.kind}",
        exact=prepared.forest,
        configuration=(
            ("maximum_steps", str(prepared.method.maximum_steps)),
            ("relaxation", str(prepared.method.relaxation)),
            ("schedule", selected.kind),
            ("evaluation_dtype", prepared.precision.evaluation_dtype),
            ("accumulation_dtype", prepared.precision.accumulation_dtype),
            ("decision_dtype", prepared.precision.decision_dtype),
            ("output_dtype", prepared.precision.output_dtype),
        ),
    )
    variable_values = _variable_log_beliefs(prepared, final_state)
    if isinstance(prepared.method, SumProductBeliefPropagation):
        factor_probabilities = _factor_probabilities(prepared, final_state)
        log_normalizer = _bethe_log_normalizer(
            prepared,
            final_state,
            variable_values,
            factor_probabilities,
        )
        return SumProductBeliefPropagationResult(
            variable_log_probabilities=VariableStateValues(
                prepared.precision.output(variable_values),
                structure_id=prepared.graph.structure_id,
            ),
            factor_probabilities=tuple(
                prepared.precision.output(probabilities)
                for probabilities in factor_probabilities
            ),
            log_normalizer=prepared.precision.output(log_normalizer),
            state=final_state,
            status=status,
            valid=valid,
            converged=converged,
            diagnostics=diagnostics,
            provenance=provenance,
            marginals_exact=prepared.forest,
            log_normalizer_exact=prepared.forest,
            log_normalizer_kind="exact" if prepared.forest else "bethe",
        )

    decision_values = prepared.precision.decision(variable_values)
    local_modes = _local_modes(prepared.graph, decision_values)
    if prepared.forest:
        map_assignment, map_score = _decode_forest_map(
            prepared,
            final_state,
            decision_values,
        )
        optimal = valid
    else:
        map_assignment = jnp.zeros_like(local_modes)
        map_score = jnp.asarray(-jnp.inf, dtype=variable_values.dtype)
        optimal = jnp.asarray(False)
    return MaxProductBeliefPropagationResult(
        variable_max_marginals=VariableStateValues(
            prepared.precision.output(variable_values),
            structure_id=prepared.graph.structure_id,
        ),
        local_modes=local_modes,
        map_assignment=map_assignment,
        map_log_score=prepared.precision.output(map_score),
        state=final_state,
        status=status,
        valid=valid,
        converged=converged,
        optimal=optimal,
        diagnostics=diagnostics,
        provenance=provenance,
        map_available=prepared.forest,
    )


__all__ = [
    "BeliefPropagationMethod",
    "BeliefPropagationSchedulePolicy",
    "BeliefPropagationResult",
    "BeliefPropagationState",
    "MaxProductBeliefPropagation",
    "MaxProductBeliefPropagationResult",
    "PreparedBeliefPropagation",
    "SumProductBeliefPropagation",
    "SumProductBeliefPropagationResult",
    "initialize_belief_propagation",
    "prepare_belief_propagation",
    "refresh_belief_propagation",
    "replace_belief_propagation_tables",
    "run_belief_propagation",
]
