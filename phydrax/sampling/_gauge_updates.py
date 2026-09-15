#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from ..graph._gauge_transport import GaugeStaplePlan
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..metrix import SpecialUnitaryGroup, UnitaryGroup


_UPDATE_ADDRESS = SampleAddress(
    "lattice-gauge", "conflict-colored-local-update", target="link", role="transition"
)
_EXCHANGE_ADDRESS = SampleAddress(
    "lattice-gauge", "replica-exchange", target="neighbor-pair", role="transition"
)

GaugeGroupKind = Literal["u1", "su2", "su3"]
GaugeUpdateKind = Literal["heatbath", "overrelaxation", "mixed"]


class GaugeUpdateStatus(IntEnum):
    SUCCESS = 0
    REJECTION_CAPACITY_EXHAUSTED = 1
    MICROCANONICAL_INVARIANCE_FAILURE = 2
    GROUP_MEMBERSHIP_FAILURE = 3
    INACTIVE_EXCHANGE_PAIR = 4


class GaugeUpdatePlan(StrictModule):
    """Finite local-update policy, independent of a lattice and runtime state."""

    coupling: float = eqx.field(static=True)
    group: GaugeGroupKind = eqx.field(static=True)
    update: GaugeUpdateKind = eqx.field(static=True)
    num_sweeps: int = eqx.field(static=True)
    rejection_attempts: int = eqx.field(static=True)
    overrelaxation_passes: int = eqx.field(static=True)
    su3_subgroup_passes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        group: GaugeGroupKind,
        /,
        *,
        coupling: float,
        update: GaugeUpdateKind = "heatbath",
        num_sweeps: int = 1,
        rejection_attempts: int = 64,
        overrelaxation_passes: int = 1,
        su3_subgroup_passes: int = 1,
    ):
        beta = float(coupling)
        sweeps = int(num_sweeps)
        attempts = int(rejection_attempts)
        reflections = int(overrelaxation_passes)
        subgroup_passes = int(su3_subgroup_passes)
        if group not in ("u1", "su2", "su3"):
            raise ValueError("group must be 'u1', 'su2', or 'su3'.")
        if update not in ("heatbath", "overrelaxation", "mixed"):
            raise ValueError("Unsupported gauge update kind.")
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("coupling must be finite and nonnegative.")
        if (
            min(sweeps, attempts, reflections, subgroup_passes) <= 0
            or sweeps > 4096
            or attempts > 65536
            or reflections > 128
            or subgroup_passes > 128
        ):
            raise ValueError(
                "Gauge update resource bounds are positive and capped at "
                "4096 sweeps, 65536 rejection attempts, and 128 local passes."
            )
        self.coupling = beta
        self.group = group
        self.update = update
        self.num_sweeps = sweeps
        self.rejection_attempts = attempts
        self.overrelaxation_passes = reflections
        self.su3_subgroup_passes = subgroup_passes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-conflict-colored-gauge-update",
                "group": group,
                "coupling": beta,
                "update": update,
                "num_sweeps": sweeps,
                "rejection_attempts": attempts,
                "overrelaxation_passes": reflections,
                "su3_subgroup_passes": subgroup_passes,
            }
        )


class PreparedGaugeUpdate(StrictModule):
    """Lattice-bound immutable update executable with a fixed sweep schedule."""

    staple_plan: GaugeStaplePlan
    small_su2_linalg: SmallLinearSolvePlan
    small_group_linalg: SmallLinearSolvePlan
    color_of_link: Array
    conflict_adjacency: Array
    color_classes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    matrix_dimension: int = eqx.field(static=True)
    num_links: int = eqx.field(static=True)
    maximum_local_updates: int = eqx.field(static=True)
    group: GaugeGroupKind = eqx.field(static=True)
    update: GaugeUpdateKind = eqx.field(static=True)
    coupling: float = eqx.field(static=True)
    num_sweeps: int = eqx.field(static=True)
    rejection_attempts: int = eqx.field(static=True)
    overrelaxation_passes: int = eqx.field(static=True)
    su3_subgroup_passes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)


class GaugeUpdateState(StrictModule):
    links: Array
    sweep_index: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)


class GaugeDetailedBalanceEvidence(StrictModule):
    link_index: Array
    accepted: Array
    log_forward_reverse_ratio: Array
    log_target_ratio: Array
    rejection_attempts: Array
    resource_exhausted: Array
    membership_preserved: Array
    exact_target_correction: Array
    status: Array


class GaugeUpdateResult(StrictModule):
    state: GaugeUpdateState
    evidence: GaugeDetailedBalanceEvidence
    root_key: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class GaugeReplicaExchangePlan(StrictModule):
    inverse_temperatures: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, inverse_temperatures: ArrayLike, /):
        beta = jnp.asarray(inverse_temperatures, dtype=float).reshape((-1,))
        if (
            beta.size < 2
            or beta.size > 65536
            or bool(jnp.any(~jnp.isfinite(beta) | (beta <= 0.0)))
        ):
            raise ValueError(
                "inverse_temperatures must contain 2 to 65536 positive values."
            )
        self.inverse_temperatures = beta
        self.plan_id = canonical_fingerprint(
            {"kind": "gauge-replica-exchange", "inverse_temperatures": np.asarray(beta)}
        )


class GaugeReplicaState(StrictModule):
    configurations: Array
    reduced_potentials: Array
    label_at_slot: Array
    step_index: Array
    exchange_parity: Array
    accepted_swaps: Array
    attempted_swaps: Array
    plan_id: str = eqx.field(static=True)


class GaugeReplicaExchangeResult(StrictModule):
    state: GaugeReplicaState
    pair_indices: Array
    attempted: Array
    accepted: Array
    log_target_ratio: Array
    log_forward_reverse_ratio: Array
    detailed_balance_residual: Array
    root_key: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_gauge_update(
    plan: GaugeUpdatePlan,
    staple_plan: GaugeStaplePlan,
    color_of_link: ArrayLike,
    conflict_adjacency: ArrayLike,
    /,
) -> PreparedGaugeUpdate:
    """Bind and validate one complete fixed conflict coloring before execution."""
    if not isinstance(plan, GaugeUpdatePlan):
        raise TypeError("plan must be GaugeUpdatePlan.")
    if not isinstance(staple_plan, GaugeStaplePlan):
        raise TypeError("staple_plan must be GaugeStaplePlan.")
    link_space = staple_plan.link_space
    colors = np.asarray(color_of_link)
    conflicts = np.asarray(conflict_adjacency, dtype=bool)
    if colors.ndim != 1 or colors.size < 1 or not np.issubdtype(colors.dtype, np.integer):
        raise ValueError("color_of_link must be a nonempty integer vector.")
    colors = colors.astype(np.int32)
    if np.any(colors < 0):
        raise ValueError("Gauge conflict colors must be nonnegative.")
    num_links = int(colors.size)
    if int(colors.max()) >= num_links:
        raise ValueError("Gauge colors must be contiguous within the link count.")
    if conflicts.shape != (num_links, num_links):
        raise ValueError("conflict_adjacency must be square with one row per link.")
    if np.any(conflicts != conflicts.T) or np.any(np.diag(conflicts)):
        raise ValueError("conflict_adjacency must be symmetric with a false diagonal.")
    classes = tuple(
        tuple(np.flatnonzero(colors == color).tolist())
        for color in range(int(colors.max()) + 1)
    )
    if any(not indices for indices in classes):
        raise ValueError("Conflict colors must be contiguous and nonempty.")
    for indices in classes:
        block = conflicts[np.ix_(indices, indices)]
        if np.any(block):
            raise ValueError("Links sharing a sweep color conflict.")
    dimension = {"u1": 1, "su2": 2, "su3": 3}[plan.group]
    expected_group = (
        isinstance(link_space.group, UnitaryGroup)
        if plan.group == "u1"
        else isinstance(link_space.group, SpecialUnitaryGroup)
    )
    if (
        not expected_group
        or link_space.group.dimension != dimension
        or link_space.num_edges != num_links
    ):
        raise ValueError(
            "Gauge group, link count, coloring, and staple link space disagree."
        )
    route_edges = np.asarray(staple_plan.complement_edges, dtype=np.int32)
    route_active = (
        np.asarray(staple_plan.complement_valid, dtype=bool)
        & np.asarray(staple_plan.route_valid, dtype=bool)[..., None]
    )
    required_conflicts = np.zeros_like(conflicts)
    for edge in range(num_links):
        dependencies = route_edges[edge][route_active[edge]]
        required_conflicts[edge, dependencies] = True
        required_conflicts[dependencies, edge] = True
    np.fill_diagonal(required_conflicts, False)
    if np.any(required_conflicts & ~conflicts):
        raise ValueError(
            "conflict_adjacency omits a dependency from the prepared staple routes."
        )
    local_factor = 3 * plan.su3_subgroup_passes if plan.group == "su3" else 1
    update_factor = (
        1
        if plan.update == "heatbath"
        else plan.overrelaxation_passes
        if plan.update == "overrelaxation"
        else 1 + plan.overrelaxation_passes
    )
    capacity = plan.num_sweeps * num_links * local_factor * update_factor
    if capacity > 10_000_000:
        raise ValueError(
            "Prepared gauge update exceeds the 10000000-local-update resource cap."
        )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-gauge-update",
            "plan": plan.plan_id,
            "staples": staple_plan.plan_id,
            "colors": colors,
            "conflicts": conflicts,
            "maximum_local_updates": capacity,
        }
    )
    return PreparedGaugeUpdate(
        staple_plan=staple_plan,
        small_su2_linalg=SmallLinearSolvePlan(2),
        small_group_linalg=SmallLinearSolvePlan(dimension),
        color_of_link=jnp.asarray(colors),
        conflict_adjacency=jnp.asarray(conflicts),
        color_classes=classes,
        matrix_dimension=dimension,
        num_links=num_links,
        maximum_local_updates=capacity,
        group=plan.group,
        update=plan.update,
        coupling=plan.coupling,
        num_sweeps=plan.num_sweeps,
        rejection_attempts=plan.rejection_attempts,
        overrelaxation_passes=plan.overrelaxation_passes,
        su3_subgroup_passes=plan.su3_subgroup_passes,
        plan_id=plan.plan_id,
        prepared_id=prepared_id,
        frozen=True,
    )


def _adjoint(matrix: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(matrix), -1, -2)


def _matmul(left: Array, right: Array, /) -> Array:
    return jnp.asarray(ein.contract("ij,jk->ik", left, right))


def _group_contains(
    linear_plan: SmallLinearSolvePlan,
    group: GaugeGroupKind,
    links: Array,
    /,
) -> Array:
    dimension = linear_plan.dimension
    identity = jnp.eye(dimension, dtype=links.dtype)
    gram = jax.vmap(lambda value: _matmul(_adjoint(value), value))(links)
    unitary = jnp.all(jnp.abs(gram - identity) <= 2e-5)
    finite = jnp.all(jnp.isfinite(links))
    determinant = determinant_small_linear(linear_plan, links)
    special = jnp.all(jnp.abs(determinant - 1.0) <= 2e-5)
    return finite & unitary & (jnp.asarray(True) if group == "u1" else special)


def initialize_gauge_update_state(
    prepared: PreparedGaugeUpdate, links: ArrayLike, /
) -> GaugeUpdateState:
    if not isinstance(prepared, PreparedGaugeUpdate):
        raise TypeError("prepared must be PreparedGaugeUpdate.")
    values = jnp.asarray(links)
    expected = (prepared.num_links, prepared.matrix_dimension, prepared.matrix_dimension)
    if values.shape != expected or not jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise ValueError(f"links must be a complex array with shape {expected}.")
    valid = _group_contains(prepared.small_group_linalg, prepared.group, values)
    values = eqx.error_if(
        values, ~valid, "Initial gauge links must lie in the declared group."
    )
    return GaugeUpdateState(
        values, jnp.asarray(0, dtype=jnp.uint32), valid, prepared.prepared_id
    )


def _sample_von_mises(
    key: Key[Array, ""], mean: Array, concentration: Array, attempts: int, /
) -> tuple[Array, Array, Array]:
    """Best--Fisher exact rejection with explicit finite-capacity exhaustion."""
    dtype = jnp.result_type(mean, concentration, float)
    keys = jr.split(key, attempts)
    uniforms = jax.vmap(lambda value: jr.uniform(value, (3,), dtype=dtype))(keys)
    zero_concentration = concentration == 0.0
    safe = jnp.where(zero_concentration, 1.0, concentration)
    tau = 1.0 + jnp.sqrt(1.0 + 4.0 * safe**2)
    rho = (2.0 * safe) / (tau + jnp.sqrt(2.0 * tau))
    ratio = (1.0 + rho**2) / (2.0 * rho)
    z = jnp.cos(jnp.pi * uniforms[:, 0])
    f = (1.0 + ratio * z) / (ratio + z)
    c = safe * (ratio - f)
    accept = (uniforms[:, 1] < c * (2.0 - c)) | (
        jnp.log(jnp.maximum(c / uniforms[:, 1], jnp.finfo(dtype).tiny)) + 1.0 - c >= 0.0
    )
    found = jnp.any(accept)
    index = jnp.argmax(accept)
    sign = jnp.where(uniforms[index, 2] >= 0.5, 1.0, -1.0)
    proposal = mean + sign * jnp.arccos(jnp.clip(f[index], -1.0, 1.0))
    uniform_angle = mean + (2.0 * uniforms[0, 0] - 1.0) * jnp.pi
    value = jnp.where(zero_concentration, uniform_angle, proposal)
    success = zero_concentration | found
    used = jnp.where(zero_concentration, 1, jnp.where(found, index + 1, attempts))
    return value, success, used.astype(jnp.int32)


def _su2_quaternion(matrix: Array, /) -> Array:
    return jnp.stack(
        (
            0.5 * jnp.real(matrix[0, 0] + matrix[1, 1]),
            0.5 * jnp.imag(matrix[0, 1] + matrix[1, 0]),
            0.5 * jnp.real(matrix[0, 1] - matrix[1, 0]),
            0.5 * jnp.imag(matrix[0, 0] - matrix[1, 1]),
        )
    )


def _quaternion_su2(value: Array, /) -> Array:
    a0, a1, a2, a3 = value
    return jnp.asarray([[a0 + 1j * a3, a2 + 1j * a1], [-a2 + 1j * a1, a0 - 1j * a3]])


def _project_su2(plan: SmallLinearSolvePlan, matrix: Array, /) -> tuple[Array, Array]:
    quaternion = _su2_quaternion(matrix)
    projected = _quaternion_su2(quaternion)
    determinant = determinant_small_linear(plan, projected)
    scale = jnp.sqrt(jnp.abs(determinant))
    identity = jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=quaternion.dtype)
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = jnp.where(scale > 0.0, quaternion / safe_scale, identity)
    return _quaternion_su2(normalized), scale


def _sample_su2(
    key: Key[Array, ""], concentration: Array, attempts: int, /
) -> tuple[Array, Array, Array]:
    """Exact SU(2) Haar heatbath rejection for density exp(k a0)."""
    dtype = jnp.result_type(concentration, float)
    keys = jr.split(key, attempts + 1)
    uniforms = jax.vmap(lambda value: jr.uniform(value, (2,), dtype=dtype))(keys[:-1])
    zero_concentration = concentration == 0.0
    safe = jnp.where(zero_concentration, 1.0, concentration)
    exponential_delta = jnp.expm1(-2.0 * safe)
    a0 = 1.0 + jnp.log1p((1.0 - uniforms[:, 0]) * exponential_delta) / safe
    accept = uniforms[:, 1] <= jnp.sqrt(jnp.maximum(1.0 - a0**2, 0.0))
    found = jnp.any(accept)
    index = jnp.argmax(accept)
    direction = jr.normal(keys[-1], (3,), dtype=dtype)
    direction = direction / jnp.linalg.norm(direction)
    selected = a0[index]
    tilted = jnp.concatenate(
        (selected[None], jnp.sqrt(jnp.maximum(1.0 - selected**2, 0.0))[None] * direction)
    )
    haar = jr.normal(keys[-1], (4,), dtype=dtype)
    haar = haar / jnp.linalg.norm(haar)
    quaternion = jnp.where(zero_concentration, haar, tilted)
    success = zero_concentration | found
    used = jnp.where(zero_concentration, 1, jnp.where(found, index + 1, attempts))
    return _quaternion_su2(quaternion), success, used.astype(jnp.int32)


def _u1_update(prepared, link, staple, key, overrelax):
    product = staple[0, 0]
    magnitude = jnp.abs(product)
    mean = -jnp.angle(product)
    current_angle = jnp.angle(link[0, 0])
    if overrelax:
        angle = 2.0 * mean - current_angle
        success, used = jnp.asarray(magnitude > 0.0), jnp.asarray(1, dtype=jnp.int32)
    else:
        angle, success, used = _sample_von_mises(
            key, mean, prepared.coupling * magnitude, prepared.rejection_attempts
        )
    proposal = jnp.exp(1j * angle).reshape((1, 1)).astype(link.dtype)
    return jnp.where(success, proposal, link), success, used


def _su2_update(prepared, link, staple, key, overrelax):
    normalized, scale = _project_su2(prepared.small_su2_linalg, staple)
    if overrelax:
        relative = _matmul(link, normalized)
        proposal = _matmul(_adjoint(relative), _adjoint(normalized))
        return proposal.astype(link.dtype), scale > 0.0, jnp.asarray(1, dtype=jnp.int32)
    sample, success, used = _sample_su2(
        key, prepared.coupling * scale, prepared.rejection_attempts
    )
    proposal = _matmul(sample, _adjoint(normalized)).astype(link.dtype)
    return jnp.where(success, proposal, link), success, used


def _embedded_su2(value: Array, pair: tuple[int, int], dtype, /) -> Array:
    result = jnp.eye(3, dtype=dtype)
    indices = jnp.asarray(pair)
    return result.at[jnp.ix_(indices, indices)].set(value)


def _su3_subgroup_update(prepared, link, staple, key, overrelax, pair):
    local = _matmul(link, staple)
    indices = jnp.asarray(pair)
    block = local[jnp.ix_(indices, indices)]
    projected, scale = _project_su2(prepared.small_su2_linalg, block)
    if overrelax:
        reflected = _adjoint(projected)
        update = _matmul(reflected, reflected)
        success, used = scale > 0.0, jnp.asarray(1, dtype=jnp.int32)
    else:
        sample, success, used = _sample_su2(
            key, 2.0 * prepared.coupling * scale / 3.0, prepared.rejection_attempts
        )
        update = _matmul(sample, _adjoint(projected))
    embedded = _embedded_su2(update, pair, link.dtype)
    proposal = _matmul(embedded, link)
    return jnp.where(success, proposal, link), success, used


def _local_log_weight(group, coupling, link, staple):
    return (
        coupling
        * jnp.real(jnp.trace(_matmul(link, staple)))
        / {"u1": 1.0, "su2": 2.0, "su3": 3.0}[group]
    )


def gauge_update_sweeps(
    prepared: PreparedGaugeUpdate,
    state: GaugeUpdateState,
    /,
    *,
    key: Key[Array, ""],
) -> GaugeUpdateResult:
    """Execute fixed conflict-colored exact-conditionals and reflections."""
    if not isinstance(prepared, PreparedGaugeUpdate):
        raise TypeError("prepared must be PreparedGaugeUpdate.")
    if not isinstance(state, GaugeUpdateState):
        raise TypeError("state must be GaugeUpdateState.")
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("Gauge state belongs to another prepared update.")
    links = state.links
    records: list[
        tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]
    ] = []
    operation = 0
    pairs = ((0, 1), (0, 2), (1, 2))
    for sweep in range(prepared.num_sweeps):
        for color, indices in enumerate(prepared.color_classes):
            for edge in indices:
                modes = (
                    (False,)
                    if prepared.update == "heatbath"
                    else (True,) * prepared.overrelaxation_passes
                    if prepared.update == "overrelaxation"
                    else (False,) + (True,) * prepared.overrelaxation_passes
                )
                for overrelax in modes:
                    subgroup_iterations = (
                        prepared.su3_subgroup_passes if prepared.group == "su3" else 1
                    )
                    for subgroup_pass in range(subgroup_iterations):
                        subgroup_pairs = pairs if prepared.group == "su3" else ((0, 0),)
                        for subgroup_index, pair in enumerate(subgroup_pairs):
                            staple = prepared.staple_plan.staple(links, edge)
                            old = links[edge]
                            local_key = derive_key(
                                key,
                                _UPDATE_ADDRESS,
                                state.sweep_index,
                                sweep,
                                color,
                                edge,
                                int(overrelax),
                                subgroup_pass,
                                subgroup_index,
                            )
                            if prepared.group == "u1":
                                proposed, accepted, used = _u1_update(
                                    prepared, old, staple, local_key, overrelax
                                )
                            elif prepared.group == "su2":
                                proposed, accepted, used = _su2_update(
                                    prepared, old, staple, local_key, overrelax
                                )
                            else:
                                proposed, accepted, used = _su3_subgroup_update(
                                    prepared, old, staple, local_key, overrelax, pair
                                )
                            before = _local_log_weight(
                                prepared.group, prepared.coupling, old, staple
                            )
                            after = _local_log_weight(
                                prepared.group, prepared.coupling, proposed, staple
                            )
                            log_target_ratio = after - before
                            log_proposal_ratio = (
                                -log_target_ratio if not overrelax else jnp.asarray(0.0)
                            )
                            exact_correction = jnp.where(
                                overrelax,
                                jnp.abs(log_target_ratio) <= 2e-5,
                                jnp.asarray(True),
                            )
                            links = links.at[edge].set(jnp.where(accepted, proposed, old))
                            member = _group_contains(
                                prepared.small_group_linalg,
                                prepared.group,
                                links[edge : edge + 1],
                            )
                            exhausted = (~accepted) & ~jnp.asarray(overrelax)
                            status = jnp.where(
                                ~member,
                                GaugeUpdateStatus.GROUP_MEMBERSHIP_FAILURE,
                                jnp.where(
                                    ~exact_correction,
                                    GaugeUpdateStatus.MICROCANONICAL_INVARIANCE_FAILURE,
                                    jnp.where(
                                        exhausted,
                                        GaugeUpdateStatus.REJECTION_CAPACITY_EXHAUSTED,
                                        GaugeUpdateStatus.SUCCESS,
                                    ),
                                ),
                            ).astype(jnp.int32)
                            records.append(
                                (
                                    jnp.asarray(edge, dtype=jnp.int32),
                                    accepted,
                                    log_proposal_ratio,
                                    log_target_ratio,
                                    used,
                                    exhausted,
                                    member,
                                    exact_correction,
                                    status,
                                )
                            )
                            operation += 1
    if operation > prepared.maximum_local_updates:
        raise RuntimeError("Prepared gauge update capacity was exceeded.")
    stacked = tuple(
        jnp.stack(tuple(record[index] for record in records)) for index in range(9)
    )
    valid = (
        state.valid
        & _group_contains(prepared.small_group_linalg, prepared.group, links)
        & jnp.all(stacked[6])
    )
    successor = GaugeUpdateState(
        links,
        state.sweep_index + jnp.asarray(prepared.num_sweeps, dtype=jnp.uint32),
        valid,
        state.prepared_id,
    )
    evidence = GaugeDetailedBalanceEvidence(*stacked)
    return GaugeUpdateResult(
        successor,
        evidence,
        jnp.asarray(key),
        prepared.plan_id,
        prepared.prepared_id,
        "product-haar",
        "finite-exact-conditional-or-microcanonical-conflict-colored-sweep",
    )


def initialize_gauge_replica_state(
    plan: GaugeReplicaExchangePlan,
    configurations: ArrayLike,
    reduced_potentials: ArrayLike,
    /,
) -> GaugeReplicaState:
    if not isinstance(plan, GaugeReplicaExchangePlan):
        raise TypeError("plan must be GaugeReplicaExchangePlan.")
    configurations_ = jnp.asarray(configurations)
    potentials = jnp.asarray(reduced_potentials, dtype=float)
    count = int(plan.inverse_temperatures.size)
    if configurations_.shape[0] != count or potentials.shape != (count, count):
        raise ValueError("Replica configurations and reduced-potential matrix disagree.")
    if not bool(jnp.all(jnp.isfinite(configurations_))) or not bool(
        jnp.all(jnp.isfinite(potentials))
    ):
        raise ValueError("Replica state must be finite.")
    return GaugeReplicaState(
        configurations_,
        potentials,
        jnp.arange(count, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        plan.plan_id,
    )


def gauge_replica_exchange(
    plan: GaugeReplicaExchangePlan,
    state: GaugeReplicaState,
    /,
    *,
    key: Key[Array, ""],
) -> GaugeReplicaExchangeResult:
    """Exchange disjoint neighbors using the exact reduced-potential ratio."""
    if not isinstance(plan, GaugeReplicaExchangePlan):
        raise TypeError("plan must be GaugeReplicaExchangePlan.")
    if not isinstance(state, GaugeReplicaState):
        raise TypeError("state must be GaugeReplicaState.")
    if state.plan_id != plan.plan_id:
        raise ValueError("Replica state belongs to another exchange plan.")
    count = int(plan.inverse_temperatures.size)
    starts = jnp.arange(count - 1, dtype=jnp.int32)
    attempted = starts % 2 == state.exchange_parity
    labels = state.label_at_slot
    left_label, right_label = labels[:-1], labels[1:]
    current = (
        state.reduced_potentials[left_label, starts]
        + state.reduced_potentials[right_label, starts + 1]
    )
    swapped = (
        state.reduced_potentials[left_label, starts + 1]
        + state.reduced_potentials[right_label, starts]
    )
    ratio = current - swapped
    keys = jax.vmap(
        lambda edge: derive_key(key, _EXCHANGE_ADDRESS, state.step_index, edge)
    )(starts)
    uniforms = jax.vmap(lambda subkey: jr.uniform(subkey))(keys)
    accepted = attempted & (jnp.log(uniforms) < jnp.minimum(ratio, 0.0))
    next_labels = labels
    for index in range(count - 1):
        left, right = next_labels[index], next_labels[index + 1]
        next_labels = next_labels.at[index].set(jnp.where(accepted[index], right, left))
        next_labels = next_labels.at[index + 1].set(
            jnp.where(accepted[index], left, right)
        )
    successor = GaugeReplicaState(
        state.configurations,
        state.reduced_potentials,
        next_labels,
        state.step_index + jnp.asarray(1, dtype=jnp.uint32),
        1 - state.exchange_parity,
        state.accepted_swaps + jnp.sum(accepted, dtype=jnp.int32),
        state.attempted_swaps + jnp.sum(attempted, dtype=jnp.int32),
        state.plan_id,
    )
    zeros = jnp.zeros_like(ratio)
    status = jnp.where(
        attempted,
        GaugeUpdateStatus.SUCCESS,
        GaugeUpdateStatus.INACTIVE_EXCHANGE_PAIR,
    ).astype(jnp.int32)
    return GaugeReplicaExchangeResult(
        successor,
        jnp.stack((starts, starts + 1), axis=-1),
        attempted,
        accepted,
        ratio,
        zeros,
        zeros,
        jnp.asarray(key),
        status,
        plan.plan_id,
        "exact-neighbor-replica-exchange",
    )


__all__ = [
    "GaugeDetailedBalanceEvidence",
    "GaugeReplicaExchangePlan",
    "GaugeReplicaExchangeResult",
    "GaugeReplicaState",
    "GaugeUpdatePlan",
    "GaugeUpdateResult",
    "GaugeUpdateState",
    "GaugeUpdateStatus",
    "PreparedGaugeUpdate",
    "gauge_replica_exchange",
    "gauge_update_sweeps",
    "initialize_gauge_replica_state",
    "initialize_gauge_update_state",
    "prepare_gauge_update",
]
