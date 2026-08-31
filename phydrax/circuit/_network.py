#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    plan as plan_linear,
    prepare as prepare_linear,
    PreparedLinearSolve,
    refresh as refresh_linear,
    solve as solve_linear,
)
from ._models import AbstractScatteringComponent
from ._ports import (
    references_compatible,
    transformed_references_compatible,
    WaveReference,
)
from ._relation_graph import (
    bind_block_diagonal_relation,
    LinearRoutePlan,
    plan_block_diagonal_routes,
)
from ._topology import InstancePort, ScatteringNetwork


class ScatteringNetworkStatus(IntEnum):
    SUCCESS = 0
    SINGULAR = 1
    NONFINITE = 2
    RESIDUAL_TOLERANCE_NOT_MET = 3


class ScatteringNetworkPolicy(StrictModule):
    """Dense native correctness policy and explicit resource envelope."""

    linear: LinearSolvePolicy
    maximum_channels: int = eqx.field(static=True)
    maximum_matrix_bytes: int = eqx.field(static=True)
    maximum_rhs_bytes: int = eqx.field(static=True)
    compatibility_rtol: float = eqx.field(static=True)
    compatibility_atol: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_channels: int = 4096,
        maximum_matrix_bytes: int = 2**30,
        maximum_rhs_bytes: int = 2**30,
        compatibility_rtol: float = 1e-10,
        compatibility_atol: float = 1e-12,
        residual_tolerance: float = 1e-10,
        linear: LinearSolvePolicy | None = None,
    ):
        if maximum_channels <= 0 or maximum_matrix_bytes <= 0 or maximum_rhs_bytes <= 0:
            raise ValueError("Resource limits must be positive.")
        if min(compatibility_rtol, compatibility_atol, residual_tolerance) < 0.0:
            raise ValueError("Tolerances must be non-negative.")
        self.linear = (
            LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status"))
            if linear is None
            else linear
        )
        if not isinstance(self.linear, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        self.maximum_channels = int(maximum_channels)
        self.maximum_matrix_bytes = int(maximum_matrix_bytes)
        self.maximum_rhs_bytes = int(maximum_rhs_bytes)
        self.compatibility_rtol = float(compatibility_rtol)
        self.compatibility_atol = float(compatibility_atol)
        self.residual_tolerance = float(residual_tolerance)


class ScatteringNetworkCostEstimate(StrictModule):
    channels: int = eqx.field(static=True)
    frequencies: int = eqx.field(static=True)
    external_ports: int = eqx.field(static=True)
    block_scattering_entries: int = eqx.field(static=True)
    matrix_bytes: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    requested_rhs_bytes: int = eqx.field(static=True)


class ScatteringNetworkPlan(StrictModule):
    network: ScatteringNetwork
    policy: ScatteringNetworkPolicy
    linear_plan: LinearSolvePlan
    scattering_routes: LinearRoutePlan
    cost: ScatteringNetworkCostEstimate
    leaf_components: tuple[AbstractScatteringComponent, ...]
    channel_references: tuple[WaveReference, ...]
    frequency_shape: tuple[int, ...] = eqx.field(static=True)
    frequency_dtype: str = eqx.field(static=True)
    leaf_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    leaf_channel_ranges: tuple[tuple[int, int], ...] = eqx.field(static=True)
    channel_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    connection_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    connection_blocks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = eqx.field(
        static=True
    )
    connection_map_ids: tuple[str, ...] = eqx.field(static=True)
    external_channels: tuple[int, ...] = eqx.field(static=True)
    external_port_ids: tuple[str, ...] = eqx.field(static=True)
    probe_channels: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    probe_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedScatteringNetwork(StrictModule):
    network: ScatteringNetwork
    plan: ScatteringNetworkPlan
    angular_frequency: Array
    scattering: Array
    connection_matrix: Array
    system_matrix: Array
    linear: PreparedLinearSolve
    references: tuple[WaveReference, ...]
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class WaveExcitation(StrictModule):
    """Ordered external incident waves; the final axis is the RHS axis."""

    incident: Array
    port_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        incident: ArrayLike,
        /,
        *,
        port_ids: Sequence[str] = (),
    ):
        value = jnp.asarray(incident)
        if value.ndim < 2 or not jnp.issubdtype(value.dtype, jnp.number):
            raise ValueError(
                "incident must be numeric with (..., external_ports, rhs) shape."
            )
        ids = tuple(str(port_id) for port_id in port_ids)
        if any(not port_id for port_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("port_ids must be unique and non-empty.")
        self.incident = value.astype(jnp.result_type(value, jnp.complex128))
        self.port_ids = ids


class ScatteringNetworkDiagnostics(StrictModule):
    status: Array
    linear_status: Array
    constitutive_residual: Array
    connection_residual: Array
    relative_residual: Array
    finite: Array

    @property
    def successful(self) -> Array:
        return self.status == int(ScatteringNetworkStatus.SUCCESS)


class ScatteringNetworkResult(StrictModule):
    external_outgoing: Array
    incident: Array
    probe_incident: tuple[Array, ...]
    probe_outgoing: tuple[Array, ...]
    diagnostics: ScatteringNetworkDiagnostics
    numeric_version: Array
    external_port_ids: tuple[str, ...] = eqx.field(static=True)
    probe_ids: tuple[str, ...] = eqx.field(static=True)


class _Flattened:
    def __init__(self):
        self.leaves: list[tuple[tuple[str, ...], AbstractScatteringComponent]] = []
        self.connections: list[tuple[tuple[str, ...], tuple[str, ...], Any]] = []
        self.probes: list[tuple[str, tuple[str, ...]]] = []


def _flatten_definition(
    network: ScatteringNetwork,
    path: tuple[str, ...],
    stack: tuple[int, ...],
    flat: _Flattened,
) -> dict[str, tuple[str, ...]]:
    identity = id(network)
    if identity in stack:
        raise ValueError("Recursive scattering-network definition cycle detected.")
    next_stack = stack + (identity,)
    addresses: dict[tuple[str, str], tuple[str, ...]] = {}
    for instance in network.instances:
        instance_path = path + (instance.instance_id,)
        component = instance.component
        if isinstance(component, ScatteringNetwork):
            nested = _flatten_definition(component, instance_path, next_stack, flat)
            for port_id, key in nested.items():
                addresses[(instance.instance_id, port_id)] = key
        else:
            flat.leaves.append((instance_path, component))
            for port in component.ports:
                addresses[(instance.instance_id, port.port_id)] = instance_path + (
                    port.port_id,
                )

    def resolve(address: InstancePort) -> tuple[str, ...]:
        key = (address.instance_id, address.port_id)
        if key not in addresses:
            raise KeyError(
                f"Unknown network address {address.instance_id!r}.{address.port_id!r}."
            )
        return addresses[key]

    for connection in network.connections:
        flat.connections.append(
            (
                resolve(connection.first),
                resolve(connection.second),
                connection.mapping,
            )
        )
    for probe in network.probes:
        qualified = "/".join(path + (probe.probe_id,))
        flat.probes.append((qualified, resolve(probe.port)))
    return {
        external_id: resolve(address)
        for external_id, address in zip(
            network.external_port_ids, network.external_ports, strict=True
        )
    }


def _compile_topology(network: ScatteringNetwork):
    flat = _Flattened()
    external_mapping = _flatten_definition(network, (), (), flat)
    channel_paths: list[tuple[str, ...]] = []
    references: list[WaveReference] = []
    leaf_ranges: list[tuple[int, int]] = []
    block_channels: dict[tuple[str, ...], tuple[int, ...]] = {}
    block_coordinates: dict[tuple[str, ...], tuple[str, ...]] = {}
    for path, component in flat.leaves:
        start = len(channel_paths)
        for port in component.ports:
            block_path = path + (port.port_id,)
            indices = tuple(range(len(channel_paths), len(channel_paths) + port.size))
            block_channels[block_path] = indices
            block_coordinates[block_path] = port.coordinate_ids
            channel_paths.extend(
                block_path + (coordinate,) for coordinate in port.coordinate_ids
            )
            references.extend(port.references)
        leaf_ranges.append((start, len(channel_paths)))
    if len(channel_paths) != len(set(channel_paths)):
        raise ValueError("Flattened leaf channel paths are not unique.")
    connection_pairs: list[tuple[int, int]] = []
    connection_blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    connection_map_ids: list[str] = []
    counts = [0] * len(channel_paths)
    for first_key, second_key, mapping in flat.connections:
        first_block = block_channels[first_key]
        second_block = block_channels[second_key]
        if len(first_block) != len(second_block):
            raise ValueError(
                "Connected wave port blocks must have equal coordinate counts."
            )
        if mapping is not None and mapping.size != len(first_block):
            raise ValueError(
                "Wave connection map size must equal the connected port block size."
            )
        connection_blocks.append((first_block, second_block))
        connection_map_ids.append("" if mapping is None else mapping.map_id)
        for first, second in zip(first_block, second_block, strict=True):
            if first == second:
                raise ValueError(
                    "A wave connection cannot connect a flattened channel to itself."
                )
            counts[first] += 1
            counts[second] += 1
            if counts[first] > 1 or counts[second] > 1:
                raise ValueError(
                    "A wave channel may participate in at most one connection."
                )
            connection_pairs.append((first, second))
    external_channels_list: list[int] = []
    external_channel_ids: list[str] = []
    for external_id, key in external_mapping.items():
        channels = block_channels[key]
        coordinates = block_coordinates[key]
        for channel, coordinate in zip(channels, coordinates, strict=True):
            external_channels_list.append(channel)
            external_channel_ids.append(
                external_id if len(channels) == 1 else f"{external_id}:{coordinate}"
            )
    external_channels = tuple(external_channels_list)
    if len(set(external_channels)) != len(external_channels):
        raise ValueError("A flattened wave channel cannot be exposed more than once.")
    for channel in external_channels:
        if counts[channel]:
            raise ValueError(
                "A connected wave channel cannot also be externally exposed."
            )
        counts[channel] = 1
    dangling = [channel_paths[index] for index, count in enumerate(counts) if count == 0]
    if dangling:
        raise ValueError(
            f"Dangling internal wave channels require connection or exposure: {dangling!r}."
        )
    probe_ids = tuple(identifier for identifier, _ in flat.probes)
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("Flattened probe IDs must be unique.")
    probe_channels = tuple(block_channels[key] for _, key in flat.probes)
    return (
        flat,
        tuple(channel_paths),
        tuple(references),
        tuple(leaf_ranges),
        tuple(connection_pairs),
        tuple(connection_blocks),
        tuple(connection_map_ids),
        external_channels,
        tuple(external_channel_ids),
        probe_channels,
        probe_ids,
    )


def plan_scattering_network(
    network: ScatteringNetwork,
    angular_frequency: ArrayLike,
    policy: ScatteringNetworkPolicy | None = None,
    /,
) -> ScatteringNetworkPlan:
    """Flatten hierarchy and preflight one exact global scattering equation."""
    if not isinstance(network, ScatteringNetwork):
        raise TypeError("network must be ScatteringNetwork.")
    omega = jnp.asarray(angular_frequency)
    if not jnp.issubdtype(omega.dtype, jnp.number):
        raise TypeError("angular_frequency must be numeric.")
    selected = ScatteringNetworkPolicy() if policy is None else policy
    if not isinstance(selected, ScatteringNetworkPolicy):
        raise TypeError("policy must be ScatteringNetworkPolicy or None.")
    (
        flat,
        channel_paths,
        references,
        leaf_ranges,
        connection_pairs,
        connection_blocks,
        connection_map_ids,
        external_channels,
        external_port_ids,
        probe_channels,
        probe_ids,
    ) = _compile_topology(network)
    mapped_pairs = {
        pair
        for blocks, map_id in zip(connection_blocks, connection_map_ids, strict=True)
        if map_id
        for pair in zip(*blocks, strict=True)
    }
    for first, second in connection_pairs:
        compatibility = (
            transformed_references_compatible
            if (first, second) in mapped_pairs
            else references_compatible
        )
        compatible = compatibility(
            references[first],
            references[second],
            rtol=selected.compatibility_rtol,
            atol=selected.compatibility_atol,
        )
        if not bool(compatible):
            raise ValueError(
                f"Incompatible wave references on connection {channel_paths[first]} "
                f"<-> {channel_paths[second]}."
            )
    channels = len(channel_paths)
    frequencies = prod(omega.shape) if omega.shape else 1
    matrix_bytes = frequencies * channels * channels * 16
    factor_bytes = matrix_bytes
    rhs_bytes = 3 * frequencies * channels * max(1, len(external_channels)) * 16
    if channels > selected.maximum_channels:
        raise MemoryError("Scattering network exceeds maximum_channels.")
    if matrix_bytes + factor_bytes > selected.maximum_matrix_bytes:
        raise MemoryError("Scattering network exceeds maximum_matrix_bytes.")
    if rhs_bytes > selected.maximum_rhs_bytes:
        raise MemoryError("Scattering network exceeds maximum_rhs_bytes.")
    template = jnp.broadcast_to(
        jnp.eye(channels, dtype=jnp.complex128),
        omega.shape + (channels, channels),
    )
    linear_problem = LinearSystem(
        DenseLinearOperator(
            template, operator_id=f"{network.network_id}/global-wave-matrix"
        ),
        problem_id=f"{network.network_id}/global-wave-system",
    )
    linear_plan = plan_linear(linear_problem, selected.linear)
    scattering_routes = plan_block_diagonal_routes(
        tuple(stop - start for start, stop in leaf_ranges),
        plan_id=f"{network.network_id}/scattering-routes",
    )
    cost = ScatteringNetworkCostEstimate(
        channels=channels,
        frequencies=frequencies,
        external_ports=len(external_channels),
        block_scattering_entries=sum((stop - start) ** 2 for start, stop in leaf_ranges),
        matrix_bytes=matrix_bytes,
        factor_bytes=factor_bytes,
        requested_rhs_bytes=rhs_bytes,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "scattering-network-plan",
            "network": network.network_id,
            "channels": channel_paths,
            "connections": connection_pairs,
            "connection_maps": connection_map_ids,
            "external": external_port_ids,
            "frequency_shape": omega.shape,
            "frequency_dtype": str(omega.dtype),
            "linear": linear_plan.plan_id,
        }
    )
    return ScatteringNetworkPlan(
        network,
        selected,
        linear_plan,
        scattering_routes,
        cost,
        tuple(component for _, component in flat.leaves),
        references,
        tuple(omega.shape),
        str(omega.dtype),
        tuple(path for path, _ in flat.leaves),
        leaf_ranges,
        channel_paths,
        connection_pairs,
        connection_blocks,
        connection_map_ids,
        external_channels,
        external_port_ids,
        probe_channels,
        probe_ids,
        plan_id,
    )


def _validate_plan_inputs(
    network: ScatteringNetwork, omega: Array, plan: ScatteringNetworkPlan
) -> None:
    if (
        tuple(omega.shape) != plan.frequency_shape
        or str(omega.dtype) != plan.frequency_dtype
    ):
        raise ValueError("Angular-frequency shape or dtype changed; replan is required.")
    compiled = _compile_topology(network)
    if (
        compiled[1] != plan.channel_paths
        or compiled[4] != plan.connection_pairs
        or compiled[5] != plan.connection_blocks
        or compiled[6] != plan.connection_map_ids
        or compiled[8] != plan.external_port_ids
    ):
        raise ValueError(
            "Scattering topology, connection map, or port schema changed; "
            "replan is required."
        )


def _assemble(
    network: ScatteringNetwork,
    omega: Array,
    plan: ScatteringNetworkPlan,
) -> tuple[Array, Array, tuple[WaveReference, ...]]:
    _validate_plan_inputs(network, omega, plan)
    flat = _Flattened()
    _flatten_definition(network, (), (), flat)
    channels = plan.cost.channels
    references: list[WaveReference] = []
    response_blocks: list[Array] = []
    numeric_versions: list[Array] = []
    for (_, component), (start, stop), expected_path in zip(
        flat.leaves, plan.leaf_channel_ranges, plan.leaf_paths, strict=True
    ):
        response = component.evaluate(omega)
        expected = stop - start
        if response.matrix.shape != omega.shape + (expected, expected):
            raise ValueError(
                f"Component at {expected_path!r} returned shape {response.matrix.shape}; "
                f"expected {omega.shape + (expected, expected)}."
            )
        if sum(port.size for port in component.ports) != expected:
            raise ValueError("Component port schema drift requires replanning.")
        response_blocks.append(response.matrix)
        numeric_versions.append(response.numeric_version)
        references.extend(response.references)
    relation = bind_block_diagonal_relation(
        plan.scattering_routes,
        response_blocks,
        numeric_version=jnp.max(jnp.stack(numeric_versions)),
        operator_id=f"{network.network_id}/block-scattering",
    )
    matrix = relation.materialize(maximum_bytes=plan.cost.matrix_bytes)
    connection = jnp.zeros((channels, channels), dtype=matrix.dtype)
    mapped_pairs = {
        pair
        for blocks, map_id in zip(
            plan.connection_blocks, plan.connection_map_ids, strict=True
        )
        if map_id
        for pair in zip(*blocks, strict=True)
    }
    for first, second in plan.connection_pairs:
        compatibility = (
            transformed_references_compatible
            if (first, second) in mapped_pairs
            else references_compatible
        )
        compatible = compatibility(
            references[first],
            references[second],
            rtol=plan.policy.compatibility_rtol,
            atol=plan.policy.compatibility_atol,
        )
        matrix = eqx.error_if(
            matrix,
            ~compatible,
            "Connected wave references became incompatible; explicit "
            "renormalization is required.",
        )
    for (first_block, second_block), (_, _, mapping) in zip(
        plan.connection_blocks, flat.connections, strict=True
    ):
        first_indices = jnp.asarray(first_block)
        second_indices = jnp.asarray(second_block)
        if mapping is None:
            forward = jnp.eye(len(first_block), dtype=matrix.dtype)
            reverse = forward
        else:
            forward = mapping.forward.astype(matrix.dtype)
            reverse = mapping.reverse.astype(matrix.dtype)
        connection = connection.at[first_indices[:, None], second_indices[None, :]].set(
            forward
        )
        connection = connection.at[second_indices[:, None], first_indices[None, :]].set(
            reverse
        )
    return matrix, connection, tuple(references)


def prepare_scattering_network(
    network: ScatteringNetwork,
    angular_frequency: ArrayLike,
    plan_or_policy: ScatteringNetworkPlan | ScatteringNetworkPolicy | None = None,
    /,
) -> PreparedScatteringNetwork:
    """Evaluate leaves, assemble ``I-C S``, and prepare one native solve."""
    omega = jnp.asarray(angular_frequency)
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, ScatteringNetworkPlan)
        else plan_scattering_network(network, omega, plan_or_policy)
    )
    scattering, connection, references = _assemble(network, omega, plan)
    identity = jnp.eye(plan.cost.channels, dtype=scattering.dtype)
    system = identity - connection @ scattering
    problem = LinearSystem(
        DenseLinearOperator(
            system, operator_id=f"{network.network_id}/global-wave-matrix"
        ),
        problem_id=f"{network.network_id}/global-wave-system",
    )
    linear = prepare_linear(problem, plan.linear_plan)
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-scattering-network", "plan": plan.plan_id}
    )
    return PreparedScatteringNetwork(
        network,
        plan,
        omega,
        scattering,
        connection,
        system,
        linear,
        references,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_scattering_network(
    prepared: PreparedScatteringNetwork,
    network: ScatteringNetwork,
    angular_frequency: ArrayLike,
    /,
) -> PreparedScatteringNetwork:
    """Refresh all numeric coefficients while preserving flattened structure."""
    if not isinstance(prepared, PreparedScatteringNetwork):
        raise TypeError("prepared must be PreparedScatteringNetwork.")
    omega = jnp.asarray(angular_frequency)
    scattering, connection, references = _assemble(network, omega, prepared.plan)
    identity = jnp.eye(prepared.plan.cost.channels, dtype=scattering.dtype)
    system = identity - connection @ scattering
    problem = LinearSystem(
        DenseLinearOperator(
            system, operator_id=f"{network.network_id}/global-wave-matrix"
        ),
        problem_id=f"{network.network_id}/global-wave-system",
    )
    linear = refresh_linear(prepared.linear, problem)
    return PreparedScatteringNetwork(
        network,
        prepared.plan,
        omega,
        scattering,
        connection,
        system,
        linear,
        references,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _ordered_incident(
    prepared: PreparedScatteringNetwork, excitation: WaveExcitation
) -> Array:
    incident = excitation.incident
    external_count = len(prepared.plan.external_channels)
    if excitation.port_ids:
        if incident.shape[-2] != len(excitation.port_ids):
            raise ValueError("Excitation port_ids do not match its port axis.")
        if any(
            port_id not in prepared.plan.external_port_ids
            for port_id in excitation.port_ids
        ):
            raise KeyError("Excitation references an unknown external port.")
        ordered = jnp.zeros(
            incident.shape[:-2] + (external_count, incident.shape[-1]),
            dtype=incident.dtype,
        )
        for source, port_id in enumerate(excitation.port_ids):
            target = prepared.plan.external_port_ids.index(port_id)
            ordered = ordered.at[..., target, :].set(incident[..., source, :])
        incident = ordered
    elif incident.shape[-2] != external_count:
        raise ValueError("Excitation external-port axis has the wrong size.")
    expected_batch = prepared.plan.frequency_shape
    if incident.shape[:-2] == ():
        incident = jnp.broadcast_to(incident, expected_batch + incident.shape[-2:])
    elif incident.shape[:-2] != expected_batch:
        raise ValueError("Excitation batch must be scalar or match angular_frequency.")
    return incident


def solve_scattering_network(
    prepared: PreparedScatteringNetwork,
    excitation: WaveExcitation | ArrayLike,
    /,
) -> ScatteringNetworkResult:
    """Solve only the supplied RHSs and report defects of the original equations."""
    if not isinstance(prepared, PreparedScatteringNetwork):
        raise TypeError("prepared must be PreparedScatteringNetwork.")
    excitation_ = (
        excitation
        if isinstance(excitation, WaveExcitation)
        else WaveExcitation(excitation)
    )
    rhs_count = int(excitation_.incident.shape[-1])
    working_rhs_bytes = (
        prepared.plan.cost.frequencies
        * prepared.plan.cost.channels
        * rhs_count
        * jnp.dtype(jnp.complex128).itemsize
        * 3
    )
    if working_rhs_bytes > prepared.plan.policy.maximum_rhs_bytes:
        raise MemoryError("Scattering excitation exceeds maximum_rhs_bytes.")
    external_incident = _ordered_incident(prepared, excitation_)
    rhs = jnp.zeros(
        prepared.plan.frequency_shape
        + (prepared.plan.cost.channels, external_incident.shape[-1]),
        dtype=jnp.result_type(prepared.scattering, external_incident),
    )
    for source, channel in enumerate(prepared.plan.external_channels):
        rhs = rhs.at[..., channel, :].set(external_incident[..., source, :])
    linear_result = solve_linear(prepared.linear, rhs)
    incident = jnp.asarray(linear_result.value)
    outgoing = prepared.scattering @ incident
    external_outgoing = outgoing[..., jnp.asarray(prepared.plan.external_channels), :]
    constitutive = outgoing - prepared.scattering @ incident
    connection = incident - prepared.connection_matrix @ outgoing - rhs
    constitutive_norm = jnp.linalg.norm(constitutive, axis=(-2, -1))
    connection_norm = jnp.linalg.norm(connection, axis=(-2, -1))
    scale = jnp.maximum(
        jnp.linalg.norm(rhs, axis=(-2, -1))
        + jnp.linalg.norm(incident, axis=(-2, -1))
        + jnp.linalg.norm(outgoing, axis=(-2, -1)),
        1.0,
    )
    relative = (constitutive_norm + connection_norm) / scale
    finite = (
        jnp.all(jnp.isfinite(incident), axis=(-2, -1))
        & jnp.all(jnp.isfinite(outgoing), axis=(-2, -1))
        & jnp.isfinite(relative)
    )
    linear_success = jnp.all(linear_result.status == int(LinearSolveStatus.SUCCESS))
    status = jnp.where(
        ~linear_success,
        int(ScatteringNetworkStatus.SINGULAR),
        jnp.where(
            ~jnp.all(finite),
            int(ScatteringNetworkStatus.NONFINITE),
            jnp.where(
                jnp.max(relative) > prepared.plan.policy.residual_tolerance,
                int(ScatteringNetworkStatus.RESIDUAL_TOLERANCE_NOT_MET),
                int(ScatteringNetworkStatus.SUCCESS),
            ),
        ),
    )
    diagnostics = ScatteringNetworkDiagnostics(
        status=jnp.asarray(status, dtype=jnp.int32),
        linear_status=jnp.asarray(linear_result.status, dtype=jnp.int32),
        constitutive_residual=constitutive_norm,
        connection_residual=connection_norm,
        relative_residual=relative,
        finite=finite,
    )
    return ScatteringNetworkResult(
        external_outgoing,
        external_incident,
        tuple(
            incident[..., jnp.asarray(channels), :]
            for channels in prepared.plan.probe_channels
        ),
        tuple(
            outgoing[..., jnp.asarray(channels), :]
            for channels in prepared.plan.probe_channels
        ),
        diagnostics,
        prepared.numeric_version,
        prepared.plan.external_port_ids,
        prepared.plan.probe_ids,
    )


def _selection_indices(
    selection: Sequence[str | int], available: tuple[str, ...]
) -> tuple[int, ...]:
    values: list[int] = []
    for item in selection:
        index = int(item) if isinstance(item, int) else available.index(str(item))
        if index < 0 or index >= len(available):
            raise IndexError("External-port selection lies outside the available range.")
        values.append(index)
    if len(set(values)) != len(values):
        raise ValueError("External-port selections must not contain duplicates.")
    return tuple(values)


def scattering_submatrix(
    prepared: PreparedScatteringNetwork,
    input_ports: Sequence[str | int],
    output_ports: Sequence[str | int],
    /,
) -> Array:
    """Compute exactly the selected input columns and output rows."""
    inputs = _selection_indices(input_ports, prepared.plan.external_port_ids)
    outputs = _selection_indices(output_ports, prepared.plan.external_port_ids)
    basis = jnp.zeros(
        (len(prepared.plan.external_channels), len(inputs)),
        dtype=prepared.scattering.dtype,
    )
    for column, index in enumerate(inputs):
        basis = basis.at[index, column].set(1.0)
    result = solve_scattering_network(prepared, basis)
    return result.external_outgoing[..., jnp.asarray(outputs), :]


def full_scattering_matrix(prepared: PreparedScatteringNetwork, /) -> Array:
    """Explicit opt-in N-RHS full external scattering matrix."""
    ports = tuple(range(len(prepared.plan.external_channels)))
    return scattering_submatrix(prepared, ports, ports)


__all__ = [
    "PreparedScatteringNetwork",
    "ScatteringNetworkCostEstimate",
    "ScatteringNetworkDiagnostics",
    "ScatteringNetworkPlan",
    "ScatteringNetworkPolicy",
    "ScatteringNetworkResult",
    "ScatteringNetworkStatus",
    "WaveExcitation",
    "full_scattering_matrix",
    "plan_scattering_network",
    "prepare_scattering_network",
    "refresh_scattering_network",
    "scattering_submatrix",
    "solve_scattering_network",
]
