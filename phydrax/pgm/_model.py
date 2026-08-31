#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..graph import HypergraphBipartiteGraph, incidence_to_bipartite_graph
from ._kernel import AbstractDiscreteFactorKernel, FactorKernelCapabilities


def _integer_array(name: str, value: Any, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{name} must contain integers.")
    host = np.asarray(array)
    if host.size and (
        host.min() < np.iinfo(np.int32).min or host.max() > np.iinfo(np.int32).max
    ):
        raise ValueError(f"{name} values must fit in int32.")
    return array.astype(jnp.int32)


def _real_array(name: str, value: Any, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    host = np.asarray(array)
    if np.any(np.isnan(host)) or np.any(np.isposinf(host)):
        raise ValueError(f"{name} may contain finite values and -inf only.")
    return array


def _selection_tuple(
    values: Sequence[VariableSelection], /
) -> tuple[VariableSelection, ...]:
    selections = tuple(values)
    if not selections:
        raise ValueError("A factor group must contain at least one variable selection.")
    if any(not isinstance(selection, VariableSelection) for selection in selections):
        raise TypeError("Factor selections must be VariableSelection values.")
    count = selections[0].size
    if any(selection.size != count for selection in selections):
        raise ValueError(
            "Every factor selection must contain the same number of variables."
        )
    return selections


def _factor_id(
    kind: str, selections: tuple[VariableSelection, ...], payload: Mapping[str, Any]
) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "scope": [
                {
                    "group": selection.group_name,
                    "indices": np.asarray(selection.indices).tolist(),
                }
                for selection in selections
            ],
            **payload,
        }
    )


class DiscreteVariableGroup(StrictModule, NonTrainableState):
    """Named tensor-shaped collection of scalar finite-state variables."""

    cardinalities: Array
    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        num_states: int | ArrayLike,
        shape: tuple[int, ...] | None = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("Variable group name must be a non-empty string.")
        raw = jnp.asarray(num_states)
        if shape is None:
            resolved_shape = tuple(int(size) for size in raw.shape) if raw.ndim else ()
        else:
            resolved_shape = tuple(int(size) for size in shape)
            if any(size < 0 for size in resolved_shape):
                raise ValueError("Variable group shape entries must be non-negative.")
        if raw.ndim == 0:
            cardinalities = jnp.full(resolved_shape or (), raw, dtype=raw.dtype)
        else:
            if tuple(raw.shape) != resolved_shape:
                raise ValueError(
                    f"num_states shape must be {resolved_shape}; got {tuple(raw.shape)}."
                )
            cardinalities = raw
        cardinalities = _integer_array("num_states", cardinalities)
        host = np.asarray(cardinalities)
        if np.any(host < 1):
            raise ValueError("Every variable cardinality must be at least one.")
        self.name = name
        self.shape = resolved_shape
        self.cardinalities = cardinalities
        self.group_id = canonical_fingerprint(
            {
                "kind": "discrete-variable-group",
                "name": name,
                "shape": list(resolved_shape),
                "cardinalities": host.reshape(-1).tolist(),
            }
        )

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def flat_cardinalities(self) -> Array:
        return self.cardinalities.reshape((-1,))


class VariableSelection(StrictModule, NonTrainableState):
    """Factor-local selection from one stable variable-group name."""

    indices: Array
    group_name: str = eqx.field(static=True)

    def __init__(
        self,
        group: DiscreteVariableGroup | str,
        indices: ArrayLike,
        /,
    ):
        if isinstance(group, DiscreteVariableGroup):
            name = group.name
        elif isinstance(group, str) and group:
            name = group
        else:
            raise TypeError("group must be a DiscreteVariableGroup or non-empty name.")
        self.group_name = name
        self.indices = _integer_array("selection indices", indices).reshape((-1,))

    @classmethod
    def all(cls, group: DiscreteVariableGroup, /) -> VariableSelection:
        if not isinstance(group, DiscreteVariableGroup):
            raise TypeError("group must be a DiscreteVariableGroup.")
        return cls(group, jnp.arange(group.size, dtype=jnp.int32))

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])


class DenseTableFactorGroup(StrictModule):
    """Batch of equal-signature factors represented by dense log-potential tables."""

    selections: tuple[VariableSelection, ...]
    log_potentials: Array
    factor_id: str = eqx.field(static=True)

    def __init__(
        self,
        selections: Sequence[VariableSelection],
        log_potentials: ArrayLike,
        /,
    ):
        scope = _selection_tuple(selections)
        values = _real_array("log_potentials", log_potentials)
        if values.ndim != len(scope) + 1:
            raise ValueError(
                "Dense log potentials need one factor axis followed by one state axis "
                "per scope position."
            )
        if int(values.shape[0]) != scope[0].size:
            raise ValueError(
                "Dense log-potential factor axis must match the scope batch."
            )
        self.selections = scope
        self.log_potentials = values
        self.factor_id = _factor_id(
            "dense-table",
            scope,
            {"parameter_shape": list(values.shape), "parameter_dtype": str(values.dtype)},
        )


class EnumeratedFactorGroup(StrictModule):
    """Batch of factors with a common explicit set of supported configurations."""

    selections: tuple[VariableSelection, ...]
    configurations: Array
    log_potentials: Array
    factor_id: str = eqx.field(static=True)

    def __init__(
        self,
        selections: Sequence[VariableSelection],
        configurations: ArrayLike,
        log_potentials: ArrayLike,
        /,
    ):
        scope = _selection_tuple(selections)
        configs = _integer_array("configurations", configurations)
        if configs.ndim != 2 or int(configs.shape[1]) != len(scope):
            raise ValueError("configurations must have shape (configuration, arity).")
        host_configs = np.asarray(configs)
        if len({tuple(row) for row in host_configs.tolist()}) != int(configs.shape[0]):
            raise ValueError("Enumerated configurations must be unique.")
        values = _real_array("log_potentials", log_potentials)
        expected = (scope[0].size, int(configs.shape[0]))
        if values.shape != expected:
            raise ValueError(
                f"log_potentials must have shape {expected}; got {values.shape}."
            )
        self.selections = scope
        self.configurations = configs
        self.log_potentials = values
        self.factor_id = _factor_id(
            "enumerated",
            scope,
            {
                "configurations": host_configs.tolist(),
                "parameter_shape": list(values.shape),
                "parameter_dtype": str(values.dtype),
            },
        )


class IsingFactorGroup(StrictModule):
    """Batch of binary spin-product log potentials."""

    selections: tuple[VariableSelection, ...]
    weights: Array
    factor_id: str = eqx.field(static=True)

    def __init__(self, selections: Sequence[VariableSelection], weights: ArrayLike, /):
        scope = _selection_tuple(selections)
        values = _real_array("weights", weights).reshape((-1,))
        if values.shape != (scope[0].size,):
            raise ValueError("Ising weights must have one value per factor.")
        self.selections = scope
        self.weights = values
        self.factor_id = _factor_id(
            "ising",
            scope,
            {"parameter_shape": list(values.shape), "parameter_dtype": str(values.dtype)},
        )


class PottsFactorGroup(StrictModule):
    """Unary or pairwise categorical log-potential tables."""

    selections: tuple[VariableSelection, ...]
    log_potentials: Array
    factor_id: str = eqx.field(static=True)

    def __init__(
        self,
        selections: Sequence[VariableSelection],
        log_potentials: ArrayLike,
        /,
    ):
        scope = _selection_tuple(selections)
        if len(scope) not in (1, 2):
            raise ValueError("Potts factors must be unary or pairwise.")
        values = _real_array("log_potentials", log_potentials)
        if values.ndim != len(scope) + 1 or int(values.shape[0]) != scope[0].size:
            raise ValueError(
                "Potts tables need one factor axis and one state axis per variable."
            )
        self.selections = scope
        self.log_potentials = values
        self.factor_id = _factor_id(
            "potts",
            scope,
            {"parameter_shape": list(values.shape), "parameter_dtype": str(values.dtype)},
        )


class LogicalFactorGroup(StrictModule, NonTrainableState):
    """Batch of exact binary OR or AND relations with a distinguished child."""

    selections: tuple[VariableSelection, ...]
    kind: Literal["or", "and"] = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)

    def __init__(
        self,
        parents: Sequence[VariableSelection],
        child: VariableSelection,
        /,
        *,
        kind: Literal["or", "and"],
    ):
        parent_scope = tuple(parents)
        if not parent_scope:
            raise ValueError("Logical factors require at least one parent.")
        if kind not in ("or", "and"):
            raise ValueError("Logical factor kind must be 'or' or 'and'.")
        scope = _selection_tuple(parent_scope + (child,))
        self.selections = scope
        self.kind = kind
        self.factor_id = _factor_id(f"logical-{kind}", scope, {})


class BinaryCardinalityFactorGroup(StrictModule):
    """Batch of binary factors whose log potential depends only on active count."""

    selections: tuple[VariableSelection, ...]
    log_count_potentials: Array
    factor_id: str = eqx.field(static=True)

    def __init__(
        self,
        selections: Sequence[VariableSelection],
        log_count_potentials: ArrayLike,
        /,
    ):
        scope = _selection_tuple(selections)
        values = _real_array("log_count_potentials", log_count_potentials)
        expected = (scope[0].size, len(scope) + 1)
        if values.shape != expected:
            raise ValueError(
                f"log_count_potentials must have shape {expected}; got {values.shape}."
            )
        self.selections = scope
        self.log_count_potentials = values
        self.factor_id = _factor_id(
            "binary-cardinality",
            scope,
            {"parameter_shape": list(values.shape), "parameter_dtype": str(values.dtype)},
        )


class KernelFactorGroup(StrictModule):
    """Batch of factors driven by one open local-score kernel and parameter PyTree."""

    selections: tuple[VariableSelection, ...]
    kernel: AbstractDiscreteFactorKernel
    parameters: Any
    factor_id: str = eqx.field(static=True)

    def __init__(
        self,
        selections: Sequence[VariableSelection],
        kernel: AbstractDiscreteFactorKernel,
        parameters: Any,
        /,
    ):
        scope = _selection_tuple(selections)
        if not isinstance(kernel, AbstractDiscreteFactorKernel):
            raise TypeError("kernel must implement AbstractDiscreteFactorKernel.")
        leaves = tuple(
            jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(parameters)
        )
        if not leaves:
            raise ValueError("Kernel factor parameters must contain array leaves.")
        if any(jnp.iscomplexobj(leaf) for leaf in leaves):
            raise TypeError("Kernel factor parameters must be real-valued.")
        signature = [
            {"shape": list(leaf.shape), "dtype": str(leaf.dtype)} for leaf in leaves
        ]
        self.selections = scope
        self.kernel = kernel
        self.parameters = parameters
        self.factor_id = _factor_id(
            f"kernel-{kernel.kernel_id}",
            scope,
            {
                "parameter_signature": signature,
                "capability_id": kernel.capabilities.capability_id,
            },
        )


FactorGroup: TypeAlias = (
    DenseTableFactorGroup
    | EnumeratedFactorGroup
    | IsingFactorGroup
    | PottsFactorGroup
    | LogicalFactorGroup
    | BinaryCardinalityFactorGroup
    | KernelFactorGroup
)


def factor_selections(group: FactorGroup, /) -> tuple[VariableSelection, ...]:
    if not isinstance(
        group,
        (
            DenseTableFactorGroup,
            EnumeratedFactorGroup,
            IsingFactorGroup,
            PottsFactorGroup,
            LogicalFactorGroup,
            BinaryCardinalityFactorGroup,
            KernelFactorGroup,
        ),
    ):
        raise TypeError("Unsupported factor group.")
    return group.selections


def factor_count(group: FactorGroup, /) -> int:
    return factor_selections(group)[0].size


def factor_kernel_id(group: FactorGroup, /) -> str:
    if isinstance(group, DenseTableFactorGroup):
        return "dense-table"
    if isinstance(group, EnumeratedFactorGroup):
        return "enumerated"
    if isinstance(group, IsingFactorGroup):
        return "ising"
    if isinstance(group, PottsFactorGroup):
        return "potts"
    if isinstance(group, LogicalFactorGroup):
        return f"logical-{group.kind}"
    if isinstance(group, BinaryCardinalityFactorGroup):
        return "binary-cardinality"
    if isinstance(group, KernelFactorGroup):
        return group.kernel.kernel_id
    raise TypeError("Unsupported factor group.")


def factor_group_capabilities(group: FactorGroup, /) -> FactorKernelCapabilities:
    """Return explicit inference, conditioning, support, and batching capabilities."""
    if isinstance(group, KernelFactorGroup):
        return group.kernel.capabilities
    if not isinstance(
        group,
        (
            DenseTableFactorGroup,
            EnumeratedFactorGroup,
            IsingFactorGroup,
            PottsFactorGroup,
            LogicalFactorGroup,
            BinaryCardinalityFactorGroup,
        ),
    ):
        raise TypeError("Unsupported factor group.")
    hard_constraints = isinstance(
        group,
        (
            DenseTableFactorGroup,
            EnumeratedFactorGroup,
            PottsFactorGroup,
            LogicalFactorGroup,
            BinaryCardinalityFactorGroup,
        ),
    )
    return FactorKernelCapabilities(
        sum_product=True,
        max_product=True,
        factor_beliefs=True,
        scalar_conditional=True,
        joint_conditional=True,
        sparse_support=isinstance(group, EnumeratedFactorGroup),
        hard_constraints=hard_constraints,
        smooth_parameters=not isinstance(group, LogicalFactorGroup),
        prepared_refresh=True,
        batched=True,
        shardable=True,
    )


def _factor_parameter_signature(group: FactorGroup, /) -> tuple[tuple[int, ...], str]:
    if isinstance(
        group, (DenseTableFactorGroup, EnumeratedFactorGroup, PottsFactorGroup)
    ):
        value = group.log_potentials
    elif isinstance(group, IsingFactorGroup):
        value = group.weights
    elif isinstance(group, BinaryCardinalityFactorGroup):
        value = group.log_count_potentials
    elif isinstance(group, LogicalFactorGroup):
        return (), "none"
    elif isinstance(group, KernelFactorGroup):
        leaves = tuple(
            jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(group.parameters)
        )
        return (
            tuple(int(leaf.size) for leaf in leaves),
            "pytree:" + ",".join(str(leaf.dtype) for leaf in leaves),
        )
    else:
        raise TypeError("Unsupported factor group.")
    return tuple(int(size) for size in value.shape), str(value.dtype)


class VariableStateValues(StrictModule):
    """Flat ragged-by-offset values over all states of all graph variables."""

    values: Array
    structure_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, /, *, structure_id: str):
        array = jnp.asarray(values)
        if jnp.iscomplexobj(array):
            raise TypeError("variable-state values must be real-valued.")
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            array = array.astype(float)
        if not isinstance(structure_id, str) or not structure_id:
            raise ValueError("structure_id must be non-empty.")
        self.values = array
        self.structure_id = structure_id


class DiscreteFactorGraph(StrictModule):
    """Immutable finite-discrete factor graph with trainable log-potential parameters."""

    variable_groups: tuple[DiscreteVariableGroup, ...]
    factor_groups: tuple[FactorGroup, ...]
    factor_scopes: tuple[Array, ...]
    cardinalities: Array
    variable_state_offsets: Array
    topology: HypergraphBipartiteGraph
    group_offsets: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    parameter_signature: tuple[tuple[tuple[int, ...], str], ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_groups: Sequence[DiscreteVariableGroup],
        factor_groups: Sequence[FactorGroup] = (),
        /,
    ):
        variables = tuple(variable_groups)
        factors = tuple(factor_groups)
        if any(not isinstance(group, DiscreteVariableGroup) for group in variables):
            raise TypeError("variable_groups must contain DiscreteVariableGroup values.")
        names = [group.name for group in variables]
        if len(set(names)) != len(names):
            raise ValueError("Variable group names must be unique.")
        if any(
            not isinstance(
                group,
                (
                    DenseTableFactorGroup,
                    EnumeratedFactorGroup,
                    IsingFactorGroup,
                    PottsFactorGroup,
                    LogicalFactorGroup,
                    BinaryCardinalityFactorGroup,
                    KernelFactorGroup,
                ),
            )
            for group in factors
        ):
            raise TypeError("factor_groups contains an unsupported factor group.")

        offsets: list[tuple[str, int, int]] = []
        name_to_group: dict[str, DiscreteVariableGroup] = {}
        name_to_offset: dict[str, int] = {}
        cardinality_parts: list[Array] = []
        variable_start = 0
        for group in variables:
            name_to_group[group.name] = group
            name_to_offset[group.name] = variable_start
            variable_stop = variable_start + group.size
            offsets.append((group.name, variable_start, variable_stop))
            cardinality_parts.append(group.flat_cardinalities)
            variable_start = variable_stop
        cardinalities = (
            jnp.concatenate(cardinality_parts)
            if cardinality_parts
            else jnp.zeros((0,), dtype=jnp.int32)
        )
        cardinalities_host = np.asarray(cardinalities, dtype=np.int32)

        scopes: list[Array] = []
        factor_payloads: list[dict[str, Any]] = []
        factor_node_parts: list[np.ndarray] = []
        factor_index_parts: list[np.ndarray] = []
        factor_start = 0
        for group in factors:
            selections = factor_selections(group)
            resolved: list[np.ndarray] = []
            for selection in selections:
                if selection.group_name not in name_to_group:
                    raise ValueError(
                        f"Unknown variable group {selection.group_name!r} in factor scope."
                    )
                variable_group = name_to_group[selection.group_name]
                local = np.asarray(selection.indices, dtype=np.int64)
                if local.size and (local.min() < 0 or local.max() >= variable_group.size):
                    raise ValueError(
                        f"Factor selection for {selection.group_name!r} is out of bounds."
                    )
                resolved.append(local + name_to_offset[selection.group_name])
            scope_host = np.stack(resolved, axis=1).astype(np.int32)
            for row in scope_host:
                if len({int(value) for value in row}) != len(row):
                    raise ValueError("A factor scope cannot repeat a variable.")
            scope = jnp.asarray(scope_host, dtype=jnp.int32)
            signature = tuple(
                int(cardinalities_host[scope_host[0, position]])
                if scope_host.shape[0]
                else _empty_scope_cardinality(group, position)
                for position in range(scope_host.shape[1])
            )
            for position, expected in enumerate(signature):
                if scope_host.shape[0] and np.any(
                    cardinalities_host[scope_host[:, position]] != expected
                ):
                    raise ValueError(
                        "Factors in one group must share one cardinality per scope position."
                    )
            _validate_factor_signature(group, signature)
            scopes.append(scope)
            count = factor_count(group)
            if count:
                factor_node_parts.append(scope_host.reshape((-1,)))
                factor_index_parts.append(
                    np.repeat(
                        np.arange(factor_start, factor_start + count), len(selections)
                    )
                )
            factor_payloads.append(
                {
                    "kernel": factor_kernel_id(group),
                    "factor_id": group.factor_id,
                    "scope": scope_host.tolist(),
                    "signature": list(signature),
                }
            )
            factor_start += count

        node_indices = (
            np.concatenate(factor_node_parts).astype(np.int32)
            if factor_node_parts
            else np.zeros((0,), dtype=np.int32)
        )
        hyperedge_indices = (
            np.concatenate(factor_index_parts).astype(np.int32)
            if factor_index_parts
            else np.zeros((0,), dtype=np.int32)
        )
        topology = incidence_to_bipartite_graph(
            node_indices,
            hyperedge_indices,
            num_nodes=variable_start,
            num_hyperedges=factor_start,
        )
        state_offsets = np.concatenate(
            ([0], np.cumsum(cardinalities_host, dtype=np.int64))
        )
        if state_offsets.size and state_offsets[-1] > np.iinfo(np.int32).max:
            raise ValueError("Total variable-state storage must fit in int32 indexing.")
        structure_id = canonical_fingerprint(
            {
                "kind": "discrete-factor-graph",
                "variables": [
                    {
                        "name": group.name,
                        "shape": list(group.shape),
                        "cardinalities": np.asarray(group.flat_cardinalities).tolist(),
                    }
                    for group in variables
                ],
                "factors": factor_payloads,
            }
        )
        self.variable_groups = variables
        self.factor_groups = factors
        self.factor_scopes = tuple(scopes)
        self.cardinalities = cardinalities
        self.variable_state_offsets = jnp.asarray(state_offsets, dtype=jnp.int32)
        self.topology = topology
        self.group_offsets = tuple(offsets)
        self.structure_id = structure_id
        self.parameter_signature = tuple(
            _factor_parameter_signature(group) for group in factors
        )

    @property
    def num_variables(self) -> int:
        return int(self.cardinalities.shape[0])

    @property
    def num_factors(self) -> int:
        return sum(factor_count(group) for group in self.factor_groups)

    @property
    def num_variable_states(self) -> int:
        return (
            int(self.variable_state_offsets[-1])
            if self.variable_state_offsets.size
            else 0
        )

    def group_offset(self, name: str, /) -> tuple[int, int]:
        for group_name, start, stop in self.group_offsets:
            if group_name == name:
                return start, stop
        raise KeyError(name)


def _empty_scope_cardinality(group: FactorGroup, position: int, /) -> int:
    if isinstance(group, (DenseTableFactorGroup, PottsFactorGroup)):
        return int(group.log_potentials.shape[position + 1])
    if isinstance(group, EnumeratedFactorGroup):
        configs = np.asarray(group.configurations)
        return int(configs[:, position].max()) + 1 if configs.shape[0] else 1
    if isinstance(
        group, (IsingFactorGroup, LogicalFactorGroup, BinaryCardinalityFactorGroup)
    ):
        return 2
    if isinstance(group, KernelFactorGroup):
        raise ValueError("An empty KernelFactorGroup cannot infer scope cardinalities.")
    raise TypeError("Unsupported factor group.")


def _validate_factor_signature(group: FactorGroup, signature: tuple[int, ...], /) -> None:
    if isinstance(group, (DenseTableFactorGroup, PottsFactorGroup)):
        table_shape = tuple(int(size) for size in group.log_potentials.shape[1:])
        if table_shape != signature:
            raise ValueError(
                f"Factor table state shape must be {signature}; got {table_shape}."
            )
    elif isinstance(group, EnumeratedFactorGroup):
        configs = np.asarray(group.configurations)
        for position, cardinality in enumerate(signature):
            if configs.shape[0] and (
                np.any(configs[:, position] < 0)
                or np.any(configs[:, position] >= cardinality)
            ):
                raise ValueError("Enumerated configuration is outside variable support.")
    elif isinstance(
        group, (IsingFactorGroup, LogicalFactorGroup, BinaryCardinalityFactorGroup)
    ):
        if any(cardinality != 2 for cardinality in signature):
            raise ValueError(
                f"{factor_kernel_id(group)} factors require binary variables."
            )
    elif isinstance(group, KernelFactorGroup):
        sample = jax.ShapeDtypeStruct((factor_count(group), len(signature)), jnp.int32)
        output = jax.eval_shape(
            lambda states: group.kernel.log_scores(group.parameters, states),
            sample,
        )
        if output.shape != (factor_count(group),):
            raise ValueError("Kernel factor score shape must be (factor,).")
    else:
        raise TypeError("Unsupported factor group.")


def factor_group_cardinality_signature(
    graph: DiscreteFactorGraph,
    group_index: int,
    /,
) -> tuple[int, ...]:
    scope = np.asarray(graph.factor_scopes[group_index])
    group = graph.factor_groups[group_index]
    if scope.shape[0]:
        cards = np.asarray(graph.cardinalities)
        return tuple(int(cards[scope[0, position]]) for position in range(scope.shape[1]))
    return tuple(
        _empty_scope_cardinality(group, position) for position in range(scope.shape[1])
    )


def factor_group_dense_tables(
    graph: DiscreteFactorGraph,
    group_index: int,
    /,
) -> Array:
    group = graph.factor_groups[group_index]
    signature = factor_group_cardinality_signature(graph, group_index)
    count = factor_count(group)
    if isinstance(group, (DenseTableFactorGroup, PottsFactorGroup)):
        return group.log_potentials
    if isinstance(group, EnumeratedFactorGroup):
        table = jnp.full((count,) + signature, -jnp.inf, dtype=group.log_potentials.dtype)
        configurations = group.configurations
        if int(configurations.shape[0]) == 0:
            return table
        factor_indices = jnp.broadcast_to(
            jnp.arange(count, dtype=jnp.int32)[:, None],
            (count, int(configurations.shape[0])),
        )
        config_indices = tuple(
            jnp.broadcast_to(configurations[:, position][None, :], factor_indices.shape)
            for position in range(len(signature))
        )
        return table.at[(factor_indices,) + config_indices].set(group.log_potentials)

    configurations = _all_configurations(signature)
    if isinstance(group, IsingFactorGroup):
        spins = 2 * configurations.astype(group.weights.dtype) - 1
        values = group.weights[:, None] * jnp.prod(spins, axis=-1)[None, :]
    elif isinstance(group, LogicalFactorGroup):
        parents = configurations[:, :-1].astype(bool)
        child = configurations[:, -1].astype(bool)
        expected = (
            jnp.any(parents, axis=-1) if group.kind == "or" else jnp.all(parents, axis=-1)
        )
        values = jnp.where(expected == child, 0.0, -jnp.inf)
        values = jnp.broadcast_to(values[None, :], (count, int(values.shape[0])))
    elif isinstance(group, BinaryCardinalityFactorGroup):
        counts = jnp.sum(configurations, axis=-1).astype(jnp.int32)
        values = group.log_count_potentials[:, counts]
    elif isinstance(group, KernelFactorGroup):
        states = jnp.broadcast_to(
            configurations[None, :, :],
            (count, int(configurations.shape[0]), len(signature)),
        )
        values = group.kernel.log_scores(group.parameters, states)
    else:
        raise TypeError("Unsupported factor group.")
    return values.reshape((count,) + signature)


def _all_configurations(signature: tuple[int, ...], /) -> Array:
    total = prod(signature)
    indices = jnp.arange(total, dtype=jnp.int64)
    columns: list[Array] = []
    divisor = total
    for cardinality in signature:
        divisor //= cardinality
        columns.append((indices // divisor) % cardinality)
    return jnp.stack(columns, axis=-1).astype(jnp.int32)


def _dense_scores(tables: Array, states: Array, /) -> Array:
    leading_shape = states.shape[:-2]
    factor_count_ = int(states.shape[-2])
    arity = int(states.shape[-1])
    flat_states = states.reshape((-1, factor_count_, arity))

    def one_batch(batch_states):
        def one_factor(table, factor_states):
            return table[tuple(factor_states)]

        return jax.vmap(one_factor)(tables, batch_states)

    values = jax.vmap(one_batch)(flat_states)
    return values.reshape(leading_shape + (factor_count_,))


def factor_group_scores(
    graph: DiscreteFactorGraph,
    group_index: int,
    scope_states: Array,
    /,
) -> Array:
    """Evaluate one factor group without materializing structured dense tables."""
    group = graph.factor_groups[group_index]
    if isinstance(group, (DenseTableFactorGroup, PottsFactorGroup)):
        return _dense_scores(group.log_potentials, scope_states)
    if isinstance(group, EnumeratedFactorGroup):
        matches = jnp.all(
            scope_states[..., :, None, :] == group.configurations,
            axis=-1,
        )
        leading = (1,) * (scope_states.ndim - 2)
        potentials = group.log_potentials.reshape(leading + group.log_potentials.shape)
        return jnp.max(jnp.where(matches, potentials, -jnp.inf), axis=-1)
    if isinstance(group, IsingFactorGroup):
        spins = 2 * scope_states.astype(group.weights.dtype) - 1
        return jnp.prod(spins, axis=-1) * group.weights
    if isinstance(group, LogicalFactorGroup):
        parents = scope_states[..., :-1].astype(bool)
        child = scope_states[..., -1].astype(bool)
        expected = (
            jnp.any(parents, axis=-1) if group.kind == "or" else jnp.all(parents, axis=-1)
        )
        return jnp.where(expected == child, 0.0, -jnp.inf)
    if isinstance(group, BinaryCardinalityFactorGroup):
        counts = jnp.sum(scope_states, axis=-1).astype(jnp.int32)
        factor_count_ = int(counts.shape[-1])
        flat_counts = counts.reshape((-1, factor_count_))

        def one_batch(batch_counts):
            return group.log_count_potentials[
                jnp.arange(factor_count_, dtype=jnp.int32),
                batch_counts,
            ]

        values = jax.vmap(one_batch)(flat_counts)
        return values.reshape(counts.shape)
    if isinstance(group, KernelFactorGroup):
        return group.kernel.log_scores(group.parameters, scope_states)
    raise TypeError("Unsupported factor group.")


def factor_graph_contains(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    /,
) -> Array:
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    states = jnp.asarray(assignments)
    if states.ndim < 1 or int(states.shape[-1]) != graph.num_variables:
        raise ValueError(
            f"assignments must end with variable axis {graph.num_variables}."
        )
    if not jnp.issubdtype(states.dtype, jnp.integer):
        return jnp.zeros(states.shape[:-1], dtype=bool)
    return jnp.all((states >= 0) & (states < graph.cardinalities), axis=-1)


def factor_graph_log_score(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    /,
) -> Array:
    """Evaluate the unnormalized graph log score over any leading batch axes."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    states = jnp.asarray(assignments)
    if states.ndim < 1 or int(states.shape[-1]) != graph.num_variables:
        raise ValueError(
            f"assignments must end with variable axis {graph.num_variables}."
        )
    integer_states = states.astype(jnp.int32)
    score = jnp.zeros(states.shape[:-1], dtype=_graph_score_dtype(graph))
    for group_index, scope in enumerate(graph.factor_scopes):
        scope_states = integer_states[..., scope]
        score = score + jnp.sum(
            factor_group_scores(graph, group_index, scope_states),
            axis=-1,
        )
    return jnp.where(factor_graph_contains(graph, states), score, -jnp.inf)


def _graph_score_dtype(graph: DiscreteFactorGraph, /):
    dtypes = []
    for group in graph.factor_groups:
        if isinstance(
            group, (DenseTableFactorGroup, EnumeratedFactorGroup, PottsFactorGroup)
        ):
            dtypes.append(group.log_potentials.dtype)
        elif isinstance(group, IsingFactorGroup):
            dtypes.append(group.weights.dtype)
        elif isinstance(group, BinaryCardinalityFactorGroup):
            dtypes.append(group.log_count_potentials.dtype)
        elif isinstance(group, KernelFactorGroup):
            dtypes.extend(
                jnp.asarray(leaf).dtype
                for leaf in jax.tree_util.tree_leaves(group.parameters)
            )
    return jnp.result_type(*dtypes) if dtypes else jnp.dtype(float)


def pack_assignments(
    graph: DiscreteFactorGraph,
    values: Mapping[str, ArrayLike] | ArrayLike,
    /,
) -> Array:
    """Pack shaped variable-group assignments into the canonical final variable axis."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    if not isinstance(values, Mapping):
        array = _integer_array("assignments", values)
        if array.ndim < 1 or int(array.shape[-1]) != graph.num_variables:
            raise ValueError("Packed assignments have the wrong final variable axis.")
        return array
    parts: list[Array] = []
    leading_shape: tuple[int, ...] | None = None
    expected_names = {group.name for group in graph.variable_groups}
    if set(values) != expected_names:
        raise ValueError("Assignment mapping keys must equal the variable-group names.")
    for group in graph.variable_groups:
        array = _integer_array(f"assignments[{group.name!r}]", values[group.name])
        if group.shape:
            if tuple(array.shape[-len(group.shape) :]) != group.shape:
                raise ValueError(
                    f"Assignment group {group.name!r} must end with shape {group.shape}."
                )
            current_leading = tuple(array.shape[: -len(group.shape)])
        else:
            current_leading = tuple(array.shape)
        if leading_shape is None:
            leading_shape = current_leading
        elif leading_shape != current_leading:
            raise ValueError("All assignment groups must share leading batch axes.")
        parts.append(array.reshape(current_leading + (group.size,)))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.int32)
    return jnp.concatenate(parts, axis=-1)


def unpack_assignments(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    /,
) -> dict[str, Array]:
    """Restore canonical assignments to their named tensor group shapes."""
    packed = pack_assignments(graph, assignments)
    leading = packed.shape[:-1]
    output: dict[str, Array] = {}
    for group, (_, start, stop) in zip(graph.variable_groups, graph.group_offsets):
        output[group.name] = packed[..., start:stop].reshape(leading + group.shape)
    return output


def pack_evidence(
    graph: DiscreteFactorGraph,
    values: Mapping[str, ArrayLike] | ArrayLike | None = None,
    /,
) -> VariableStateValues:
    """Pack unary log evidence into the flat variable-state layout."""
    if values is None:
        packed = jnp.zeros((graph.num_variable_states,), dtype=_graph_score_dtype(graph))
        return VariableStateValues(packed, structure_id=graph.structure_id)
    if not isinstance(values, Mapping):
        packed = _real_array("evidence", values)
        if packed.ndim < 1 or int(packed.shape[-1]) != graph.num_variable_states:
            raise ValueError("Packed evidence has the wrong final variable-state axis.")
        return VariableStateValues(packed, structure_id=graph.structure_id)

    expected_names = {group.name for group in graph.variable_groups}
    if set(values) != expected_names:
        raise ValueError("Evidence mapping keys must equal the variable-group names.")
    parts: list[Array] = []
    leading_shape: tuple[int, ...] | None = None
    for group, (_, start, stop) in zip(graph.variable_groups, graph.group_offsets):
        array = _real_array(f"evidence[{group.name!r}]", values[group.name])
        cards = np.asarray(group.flat_cardinalities)
        if cards.size and np.all(cards == cards[0]):
            expected_tail = group.shape + (int(cards[0]),)
            if tuple(array.shape[-len(expected_tail) :]) != expected_tail:
                raise ValueError(
                    f"Evidence group {group.name!r} must end with shape {expected_tail}."
                )
            current_leading = tuple(array.shape[: -len(expected_tail)])
            flat = array.reshape(current_leading + (int(cards.sum()),))
        else:
            state_count = int(cards.sum())
            if array.ndim < 1 or int(array.shape[-1]) != state_count:
                raise ValueError(
                    f"Heterogeneous evidence group {group.name!r} must end with "
                    f"flat state axis {state_count}."
                )
            current_leading = tuple(array.shape[:-1])
            flat = array
        if leading_shape is None:
            leading_shape = current_leading
        elif leading_shape != current_leading:
            raise ValueError("All evidence groups must share leading batch axes.")
        parts.append(flat)
    packed = jnp.concatenate(parts, axis=-1) if parts else jnp.zeros((0,), dtype=float)
    return VariableStateValues(packed, structure_id=graph.structure_id)


__all__ = [
    "BinaryCardinalityFactorGroup",
    "DenseTableFactorGroup",
    "DiscreteFactorGraph",
    "DiscreteVariableGroup",
    "EnumeratedFactorGroup",
    "FactorGroup",
    "KernelFactorGroup",
    "IsingFactorGroup",
    "LogicalFactorGroup",
    "PottsFactorGroup",
    "VariableSelection",
    "VariableStateValues",
    "factor_count",
    "factor_graph_contains",
    "factor_graph_log_score",
    "factor_group_cardinality_signature",
    "factor_group_capabilities",
    "factor_group_dense_tables",
    "factor_kernel_id",
    "pack_assignments",
    "pack_evidence",
    "unpack_assignments",
]
