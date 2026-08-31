#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    FailurePolicy,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    plan as plan_linear,
    prepare as prepare_linear,
    PreparedLinearSolve,
    RecyclingPolicy,
    refresh as refresh_linear,
    solve as solve_linear,
    TolerancePolicy,
)
from ..sparse import SparseLinearMap
from ._network import (
    _compile_topology,
    _flatten_definition,
    _Flattened,
    ScatteringNetworkDiagnostics,
    ScatteringNetworkResult,
    ScatteringNetworkStatus,
    WaveExcitation,
)
from ._ports import (
    references_compatible,
    transformed_references_compatible,
    WaveReference,
)
from ._relation_graph import (
    bind_block_diagonal_relation,
    bind_linear_relation,
    LinearRoutePlan,
    plan_block_diagonal_routes,
    plan_linear_routes,
)
from ._topology import ScatteringNetwork


class ScatteringActionPolicy(StrictModule):
    """Matrix-free global scattering policy over sparse block actions."""

    linear: LinearSolvePolicy
    maximum_channels: int = eqx.field(static=True)
    maximum_operator_bytes: int = eqx.field(static=True)
    maximum_rhs_bytes: int = eqx.field(static=True)
    compatibility_rtol: float = eqx.field(static=True)
    compatibility_atol: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_channels: int = 1_000_000,
        maximum_operator_bytes: int = 2**30,
        maximum_rhs_bytes: int = 2**30,
        compatibility_rtol: float = 1e-10,
        compatibility_atol: float = 1e-12,
        residual_tolerance: float = 1e-8,
        linear: LinearSolvePolicy | None = None,
    ):
        limits = (
            int(maximum_channels),
            int(maximum_operator_bytes),
            int(maximum_rhs_bytes),
        )
        tolerances = (
            float(compatibility_rtol),
            float(compatibility_atol),
            float(residual_tolerance),
        )
        if any(value <= 0 for value in limits):
            raise ValueError("Scattering action resource limits must be positive.")
        if any(value < 0.0 for value in tolerances):
            raise ValueError("Scattering action tolerances must be nonnegative.")
        selected = (
            LinearSolvePolicy(
                FGMRES(restart=256, stagnation_iterations=256),
                tolerance=TolerancePolicy(
                    relative=max(tolerances[2], 1e-12),
                    absolute=max(tolerances[2], 1e-12),
                ),
                failure=FailurePolicy("status"),
                recycling=RecyclingPolicy(capacity=20),
            )
            if linear is None
            else linear
        )
        if not isinstance(selected, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(selected.method, FGMRES):
            raise ValueError("Scattering action execution requires native FGMRES.")
        self.linear = selected
        self.maximum_channels, self.maximum_operator_bytes, self.maximum_rhs_bytes = (
            limits
        )
        (
            self.compatibility_rtol,
            self.compatibility_atol,
            self.residual_tolerance,
        ) = tolerances


class ScatteringActionCostEstimate(StrictModule):
    channels: int = eqx.field(static=True)
    external_channels: int = eqx.field(static=True)
    scattering_entries: int = eqx.field(static=True)
    connection_entries: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)


class ScatteringActionPlan(StrictModule):
    network: ScatteringNetwork
    policy: ScatteringActionPolicy
    linear_plan: LinearSolvePlan
    scattering_routes: LinearRoutePlan
    connection_routes: LinearRoutePlan
    cost: ScatteringActionCostEstimate
    frequency_dtype: str = eqx.field(static=True)
    leaf_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    leaf_ranges: tuple[tuple[int, int], ...] = eqx.field(static=True)
    channel_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    connection_blocks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = eqx.field(
        static=True
    )
    connection_map_ids: tuple[str, ...] = eqx.field(static=True)
    external_channels: tuple[int, ...] = eqx.field(static=True)
    external_channel_ids: tuple[str, ...] = eqx.field(static=True)
    probe_channels: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    probe_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedScatteringAction(StrictModule):
    network: ScatteringNetwork
    plan: ScatteringActionPlan
    angular_frequency: Array
    scattering: SparseLinearMap
    connection: SparseLinearMap
    system: FunctionLinearOperator
    linear: PreparedLinearSolve
    references: tuple[WaveReference, ...]
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class _ScatteringSystemAction(StrictModule):
    scattering: SparseLinearMap
    connection: SparseLinearMap

    def __call__(self, value: Array, /) -> Array:
        return value - self.connection.mv(self.scattering.mv(value))


def _connection_structure(compiled) -> tuple[list[int], list[int]]:
    sources: list[int] = []
    targets: list[int] = []
    flat, connection_blocks = compiled[0], compiled[5]
    for (first, second), (_, _, mapping) in zip(
        connection_blocks, flat.connections, strict=True
    ):
        if mapping is None:
            for first_index, second_index in zip(first, second, strict=True):
                sources.extend((second_index, first_index))
                targets.extend((first_index, second_index))
        else:
            for first_index in first:
                for second_index in second:
                    sources.append(second_index)
                    targets.append(first_index)
            for second_index in second:
                for first_index in first:
                    sources.append(first_index)
                    targets.append(second_index)
    return sources, targets


def _connection_values(
    connection_blocks: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    flat: _Flattened,
    /,
) -> Array:
    values: list[Array] = []
    for (first, second), (_, _, mapping) in zip(
        connection_blocks, flat.connections, strict=True
    ):
        if mapping is None:
            for _ in zip(first, second, strict=True):
                values.extend((jnp.asarray(1.0 + 0.0j), jnp.asarray(1.0 + 0.0j)))
        else:
            values.extend(mapping.forward.reshape((-1,)))
            values.extend(mapping.reverse.reshape((-1,)))
    return (
        jnp.stack(values).astype(jnp.complex128)
        if values
        else jnp.zeros((0,), dtype=jnp.complex128)
    )


def _system_operator(
    scattering: SparseLinearMap,
    connection: SparseLinearMap,
    channels: int,
    network_id: str,
    /,
) -> FunctionLinearOperator:
    space = ArraySpace((channels,), dtype=jnp.complex128)
    return FunctionLinearOperator(
        _ScatteringSystemAction(scattering, connection),
        source=space,
        target=space,
        operator_id=f"{network_id}/matrix-free-wave-system",
    )


def plan_scattering_action(
    network: ScatteringNetwork,
    angular_frequency: ArrayLike,
    policy: ScatteringActionPolicy | None = None,
    /,
) -> ScatteringActionPlan:
    if not isinstance(network, ScatteringNetwork):
        raise TypeError("network must be ScatteringNetwork.")
    omega = jnp.asarray(angular_frequency)
    if omega.shape != () or not jnp.issubdtype(omega.dtype, jnp.number):
        raise ValueError("Scattering action planning requires one scalar frequency.")
    selected = ScatteringActionPolicy() if policy is None else policy
    if not isinstance(selected, ScatteringActionPolicy):
        raise TypeError("policy must be ScatteringActionPolicy or None.")
    compiled = _compile_topology(network)
    (
        flat,
        channel_paths,
        references,
        leaf_ranges,
        connection_pairs,
        connection_blocks,
        connection_map_ids,
        external_channels,
        external_ids,
        probe_channels,
        probe_ids,
    ) = compiled
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
        if not bool(
            compatibility(
                references[first],
                references[second],
                rtol=selected.compatibility_rtol,
                atol=selected.compatibility_atol,
            )
        ):
            raise ValueError("Matrix-free network contains incompatible references.")
    channels = len(channel_paths)
    if channels > selected.maximum_channels:
        raise MemoryError("Scattering action exceeds maximum_channels.")
    scattering_routes = plan_block_diagonal_routes(
        tuple(stop - start for start, stop in leaf_ranges),
        plan_id=f"{network.network_id}/action-scattering-routes",
    )
    source_indices, target_indices = _connection_structure(compiled)
    connection_routes = plan_linear_routes(
        channels,
        channels,
        source_indices,
        target_indices,
        plan_id=f"{network.network_id}/action-connection-routes",
    )
    retained = (
        scattering_routes.cost.route_bytes
        + scattering_routes.cost.coefficient_bytes
        + connection_routes.cost.route_bytes
        + connection_routes.cost.coefficient_bytes
    )
    if retained > selected.maximum_operator_bytes:
        raise MemoryError("Scattering action exceeds maximum_operator_bytes.")
    scattering_template = bind_block_diagonal_relation(
        scattering_routes,
        tuple(jnp.eye(stop - start, dtype=jnp.complex128) for start, stop in leaf_ranges),
        operator_id=f"{network.network_id}/action-scattering-template",
    ).operator
    connection_template = bind_linear_relation(
        connection_routes,
        _connection_values(connection_blocks, flat),
        operator_id=f"{network.network_id}/action-connection-template",
    ).operator
    template = _system_operator(
        scattering_template, connection_template, channels, network.network_id
    )
    linear_plan = plan_linear(
        LinearSystem(template, problem_id=f"{network.network_id}/action-system"),
        selected.linear,
    )
    if linear_plan.backend != "native-krylov":
        raise ValueError("Scattering action selected a materializing linear backend.")
    cost = ScatteringActionCostEstimate(
        channels,
        len(external_channels),
        scattering_routes.cost.structural_entries,
        connection_routes.cost.structural_entries,
        retained,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "scattering-action-plan",
            "network": network.network_id,
            "channels": channel_paths,
            "connections": connection_blocks,
            "connection_maps": connection_map_ids,
            "frequency_dtype": str(omega.dtype),
            "linear": linear_plan.plan_id,
        }
    )
    return ScatteringActionPlan(
        network,
        selected,
        linear_plan,
        scattering_routes,
        connection_routes,
        cost,
        str(omega.dtype),
        tuple(path for path, _ in flat.leaves),
        leaf_ranges,
        channel_paths,
        connection_blocks,
        connection_map_ids,
        external_channels,
        external_ids,
        probe_channels,
        probe_ids,
        plan_id,
    )


def _prepare_operators(
    network: ScatteringNetwork,
    omega: Array,
    plan: ScatteringActionPlan,
    /,
) -> tuple[SparseLinearMap, SparseLinearMap, tuple[WaveReference, ...], Array]:
    if omega.shape != () or str(omega.dtype) != plan.frequency_dtype:
        raise ValueError(
            "Scattering action frequency shape or dtype requires replanning."
        )
    compiled = _compile_topology(network)
    if (
        compiled[1] != plan.channel_paths
        or compiled[5] != plan.connection_blocks
        or compiled[6] != plan.connection_map_ids
        or compiled[8] != plan.external_channel_ids
    ):
        raise ValueError("Scattering action topology changed; replanning is required.")
    flat = _Flattened()
    _flatten_definition(network, (), (), flat)
    blocks: list[Array] = []
    references: list[WaveReference] = []
    versions: list[Array] = []
    for (_, component), (start, stop), path in zip(
        flat.leaves, plan.leaf_ranges, plan.leaf_paths, strict=True
    ):
        response = component.evaluate(omega)
        expected = stop - start
        if response.matrix.shape != (expected, expected):
            raise ValueError(
                f"Component at {path!r} returned a structurally incompatible response."
            )
        blocks.append(response.matrix)
        references.extend(response.references)
        versions.append(response.numeric_version)
    scattering = bind_block_diagonal_relation(
        plan.scattering_routes,
        blocks,
        numeric_version=jnp.max(jnp.stack(versions)),
        operator_id=f"{network.network_id}/action-scattering",
    ).operator
    connection = bind_linear_relation(
        plan.connection_routes,
        _connection_values(plan.connection_blocks, flat),
        operator_id=f"{network.network_id}/action-connection",
    ).operator
    return scattering, connection, tuple(references), jnp.max(jnp.stack(versions))


def prepare_scattering_action(
    network: ScatteringNetwork,
    angular_frequency: ArrayLike,
    plan_or_policy: ScatteringActionPlan | ScatteringActionPolicy | None = None,
    /,
) -> PreparedScatteringAction:
    omega = jnp.asarray(angular_frequency)
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, ScatteringActionPlan)
        else plan_scattering_action(network, omega, plan_or_policy)
    )
    scattering, connection, references, version = _prepare_operators(network, omega, plan)
    system = _system_operator(
        scattering, connection, plan.cost.channels, network.network_id
    )
    linear = prepare_linear(
        LinearSystem(system, problem_id=f"{network.network_id}/action-system"),
        plan.linear_plan,
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-scattering-action", "plan": plan.plan_id}
    )
    return PreparedScatteringAction(
        network,
        plan,
        omega,
        scattering,
        connection,
        system,
        linear,
        references,
        version,
        prepared_id,
    )


def refresh_scattering_action(
    prepared: PreparedScatteringAction,
    network: ScatteringNetwork,
    angular_frequency: ArrayLike,
    /,
) -> PreparedScatteringAction:
    if not isinstance(prepared, PreparedScatteringAction):
        raise TypeError("prepared must be PreparedScatteringAction.")
    omega = jnp.asarray(angular_frequency)
    scattering, connection, references, version = _prepare_operators(
        network, omega, prepared.plan
    )
    system = _system_operator(
        scattering, connection, prepared.plan.cost.channels, network.network_id
    )
    linear = refresh_linear(
        prepared.linear,
        LinearSystem(system, problem_id=f"{network.network_id}/action-system"),
    )
    return PreparedScatteringAction(
        network,
        prepared.plan,
        omega,
        scattering,
        connection,
        system,
        linear,
        references,
        version,
        prepared.prepared_id,
    )


def _ordered_excitation(
    prepared: PreparedScatteringAction,
    excitation: WaveExcitation,
    /,
) -> Array:
    incident = excitation.incident
    count = len(prepared.plan.external_channels)
    if excitation.port_ids:
        if incident.shape[-2] != len(excitation.port_ids):
            raise ValueError("Excitation IDs do not match its channel axis.")
        ordered = jnp.zeros((count, incident.shape[-1]), dtype=incident.dtype)
        for source, channel_id in enumerate(excitation.port_ids):
            if channel_id not in prepared.plan.external_channel_ids:
                raise KeyError(f"Unknown external channel {channel_id!r}.")
            ordered = ordered.at[
                prepared.plan.external_channel_ids.index(channel_id), :
            ].set(incident[source, :])
        return ordered
    if incident.shape[-2] != count:
        raise ValueError("Excitation channel axis has the wrong size.")
    return incident


def solve_scattering_action(
    prepared: PreparedScatteringAction,
    excitation: WaveExcitation | ArrayLike,
    /,
) -> ScatteringNetworkResult:
    if not isinstance(prepared, PreparedScatteringAction):
        raise TypeError("prepared must be PreparedScatteringAction.")
    excitation_ = (
        excitation
        if isinstance(excitation, WaveExcitation)
        else WaveExcitation(excitation)
    )
    incident_external = _ordered_excitation(prepared, excitation_)
    rhs_count = int(incident_external.shape[-1])
    rhs_bytes = (
        3 * prepared.plan.cost.channels * rhs_count * jnp.dtype(jnp.complex128).itemsize
    )
    if rhs_bytes > prepared.plan.policy.maximum_rhs_bytes:
        raise MemoryError("Scattering action RHS exceeds maximum_rhs_bytes.")
    rhs = (
        jnp.zeros((prepared.plan.cost.channels, rhs_count), dtype=jnp.complex128)
        .at[jnp.asarray(prepared.plan.external_channels), :]
        .set(incident_external)
    )
    linear_result = solve_linear(prepared.linear, rhs)
    incident = jnp.asarray(linear_result.value)
    outgoing = prepared.scattering.mv(incident)
    constitutive = outgoing - prepared.scattering.mv(incident)
    connection = incident - prepared.connection.mv(outgoing) - rhs
    constitutive_norm = jnp.linalg.norm(constitutive)
    connection_norm = jnp.linalg.norm(connection)
    scale = jnp.maximum(
        jnp.linalg.norm(rhs) + jnp.linalg.norm(incident) + jnp.linalg.norm(outgoing),
        1.0,
    )
    relative = (constitutive_norm + connection_norm) / scale
    finite = jnp.all(jnp.isfinite(incident)) & jnp.all(jnp.isfinite(outgoing))
    linear_success = jnp.all(linear_result.status == int(LinearSolveStatus.SUCCESS))
    status = jnp.where(
        ~linear_success,
        int(ScatteringNetworkStatus.SINGULAR),
        jnp.where(
            ~finite,
            int(ScatteringNetworkStatus.NONFINITE),
            jnp.where(
                relative > prepared.plan.policy.residual_tolerance,
                int(ScatteringNetworkStatus.RESIDUAL_TOLERANCE_NOT_MET),
                int(ScatteringNetworkStatus.SUCCESS),
            ),
        ),
    )
    diagnostics = ScatteringNetworkDiagnostics(
        status,
        jnp.asarray(linear_result.status, dtype=jnp.int32),
        constitutive_norm,
        connection_norm,
        relative,
        finite,
    )
    return ScatteringNetworkResult(
        outgoing[jnp.asarray(prepared.plan.external_channels), :],
        incident_external,
        tuple(
            incident[jnp.asarray(channels), :]
            for channels in prepared.plan.probe_channels
        ),
        tuple(
            outgoing[jnp.asarray(channels), :]
            for channels in prepared.plan.probe_channels
        ),
        diagnostics,
        prepared.numeric_version,
        prepared.plan.external_channel_ids,
        prepared.plan.probe_ids,
    )


def scattering_action_submatrix(
    prepared: PreparedScatteringAction,
    input_channels: Sequence[str | int],
    output_channels: Sequence[str | int],
    /,
) -> Array:
    available = prepared.plan.external_channel_ids

    def selection(values: Sequence[str | int]) -> tuple[int, ...]:
        result = tuple(
            int(value) if isinstance(value, int) else available.index(str(value))
            for value in values
        )
        if (
            not result
            or len(set(result)) != len(result)
            or any(value < 0 or value >= len(available) for value in result)
        ):
            raise ValueError("Scattering action selection is invalid.")
        return result

    inputs, outputs = selection(input_channels), selection(output_channels)
    basis = jnp.zeros((len(available), len(inputs)), dtype=jnp.complex128)
    for column, index in enumerate(inputs):
        basis = basis.at[index, column].set(1.0)
    result = solve_scattering_action(prepared, basis)
    return result.external_outgoing[jnp.asarray(outputs), :]


__all__ = [
    "PreparedScatteringAction",
    "ScatteringActionCostEstimate",
    "ScatteringActionPlan",
    "ScatteringActionPolicy",
    "plan_scattering_action",
    "prepare_scattering_action",
    "refresh_scattering_action",
    "scattering_action_submatrix",
    "solve_scattering_action",
]
