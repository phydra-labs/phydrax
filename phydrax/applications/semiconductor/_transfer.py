#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...meshing import CellMeshingResult, CellMeshTransition
from ...solver import DAESolvePolicy, initialize_dae
from ._analysis import (
    _consistent_rate,
    _coordinates_from_storage,
    _dae_policy,
    _dae_problem,
    _evidence,
    _linear_policy,
    SemiconductorLinearEvidence,
)
from ._continuum import PreparedSemiconductorDevice, SemiconductorOperatingPoint
from ._materials import SemiconductorMaterial


class SemiconductorTransferEvidence(StrictModule):
    """Conserved particle, material-energy and bound-trap inventories."""

    source_counts: Array
    transferred_counts: Array
    reinitialized_counts: Array
    count_error: Array
    count_threshold: Array
    source_charge: Array
    reinitialized_charge: Array
    source_material_energy: Array
    transferred_material_energy: Array
    reinitialized_material_energy: Array
    material_energy_error: Array
    material_energy_threshold: Array
    source_trap_counts: Array
    transferred_trap_counts: Array
    reinitialized_trap_counts: Array
    positive: Array
    conservative: Array
    initialization_valid: Array
    accepted: Array
    initialization: Any
    rate_evidence: SemiconductorLinearEvidence
    transition_id: str = eqx.field(static=True)


class SemiconductorTransferResult(StrictModule):
    """Atomic result: prepared/coordinates retain the source on rejection."""

    prepared: PreparedSemiconductorDevice
    coordinates: Array
    coordinate_rates: Array
    accepted: Array
    evidence: SemiconductorTransferEvidence
    candidate_prepared: PreparedSemiconductorDevice
    candidate_coordinates: Array


class SemiconductorAdaptationIndicators(StrictModule):
    residual: Array
    current_jump: Array
    potential_gradient: Array
    carrier_log_gradient: Array


def _transition_routes(prepared, target, transition, source_result):
    if not isinstance(transition, CellMeshTransition) or not isinstance(
        source_result, CellMeshingResult
    ):
        raise TypeError(
            "A native CellMeshTransition and source CellMeshingResult are required."
        )
    old, new = prepared.plan.support, target.plan.support
    source, destination = source_result.mesh, transition.target.mesh
    if (
        old.source_id != source.mesh_id
        or old.source_revision != source.numeric_version
        or old.source_topology_id != source.topology_id
        or transition.source_mesh_id != source.mesh_id
        or transition.source_topology_id != source.topology_id
        or new.source_id != destination.mesh_id
        or new.source_revision != destination.numeric_version
        or new.source_topology_id != destination.topology_id
    ):
        raise ValueError(
            "Mesh transition source/target identity or numeric revision is stale."
        )
    if prepared.plan.terminal_names != target.plan.terminal_names:
        raise ValueError(
            "Conservative reprepare must preserve terminal identity and order."
        )
    stencil = transition.vertex_stencil
    if stencil is None:
        raise ValueError("Native transition must supply a vertex interpolation stencil.")
    lineage = transition.lineage.entity_lineage(0)
    if (
        lineage.source_entity_set_id != source.entity_set(0).entity_set_id
        or lineage.target_entity_set_id != destination.entity_set(0).entity_set_id
        or stencil.source_entity_set_id != lineage.source_entity_set_id
        or stencil.target_entity_set_id != lineage.target_entity_set_id
    ):
        raise ValueError(
            "Vertex lineage and interpolation entity sets do not match the meshes."
        )
    old_ids, new_ids = np.asarray(old.node_ids), np.asarray(new.node_ids)
    if set(old_ids.tolist()) != set(np.asarray(source.vertex_global_ids).tolist()) or set(
        new_ids.tolist()
    ) != set(np.asarray(destination.vertex_global_ids).tolist()):
        raise ValueError(
            "Transport nodes must cover the native source and target vertices."
        )
    source_lookup = {int(value): index for index, value in enumerate(old_ids)}
    target_lookup = {
        int(value): index
        for index, value in enumerate(np.asarray(stencil.target_global_ids))
    }
    if len(target_lookup) != stencil.target_global_ids.size or set(target_lookup) != set(
        new_ids.tolist()
    ):
        raise ValueError(
            "Native stencil must cover each target transport node exactly once."
        )
    rows = np.asarray([target_lookup[int(value)] for value in new_ids], dtype=np.int32)
    sources = np.asarray(stencil.source_global_ids)[rows]
    valid = np.asarray(stencil.valid)[rows]
    weights = np.where(valid, np.asarray(stencil.weights)[rows], 0.0)
    if np.any(weights < 0) or not stencil.preserves_constants:
        raise ValueError(
            "Positive conservative transfer requires a nonnegative constant-preserving stencil."
        )
    routes = np.zeros(sources.shape, dtype=np.int32)
    for row, column in zip(*np.nonzero(valid), strict=True):
        identifier = int(sources[row, column])
        if identifier not in source_lookup:
            raise ValueError("Native transition references an unavailable source vertex.")
        routes[row, column] = source_lookup[identifier]
    return jnp.asarray(routes), jnp.asarray(weights)


def _physical_fields(prepared, coordinates):
    n, p = prepared.densities(coordinates)
    return jnp.stack(
        (n, p, prepared.plan.donor_density, prepared.plan.acceptor_density), axis=-1
    )


def _counts(prepared, fields):
    return jnp.sum(prepared.plan.support.volumes[:, None] * fields, axis=0)


def _material_keys(plan):
    identities = tuple(
        f"{type(model).__name__}:{model.name}:{model.provenance}"
        for model in plan.material_models
    )
    index = np.asarray(plan.material_index)
    return np.asarray([identities[int(value)] for value in index], dtype=object)


def _stored_material_energy(prepared, coordinates):
    stored = prepared.storage(coordinates) * prepared.storage_scale
    total = jnp.asarray(0.0, dtype=jnp.asarray(coordinates).dtype)
    for name in prepared.layout.bulk_names:
        if name.endswith("_energy"):
            total += jnp.sum(prepared.field(stored, name))
    trap_counts = []
    for index, binding in enumerate(prepared.plan.traps):
        count = jnp.sum(prepared.field(stored, f"trap_{index}"))
        trap_counts.append(count)
        total += binding.trap.trap_energy * count
    counts = (
        jnp.stack(tuple(trap_counts))
        if trap_counts
        else jnp.empty((0,), dtype=total.dtype)
    )
    return total, counts


def semiconductor_reprepare(
    prepared: PreparedSemiconductorDevice,
    point: SemiconductorOperatingPoint,
    target_prepared: PreparedSemiconductorDevice,
    transition: CellMeshTransition,
    /,
    *,
    source_result: CellMeshingResult,
    policy: DAESolvePolicy | None = None,
    relative_tolerance: float = 1e-10,
    absolute_particle_tolerance: float = 1e-12,
    absolute_energy_tolerance: float = 1e-24,
) -> SemiconductorTransferResult:
    """Conservatively transfer physical inventories, then reinitialize constraints.

    Native positive vertex weights are converted to a local, material-preserving
    redistribution of carrier/dopant counts and optional lattice/carrier energy.
    Bulk or surface trap bindings preserve their extensive occupied count by
    binding order and reject insufficient target capacity. Algebraic potentials
    and interface traces are re-solved; no logarithm, temperature, occupancy
    logit or eigenvector phase is interpolated.

    Counts and material energy are checked again after native DAE
    initialization. Any stale lineage, unsupported material route, invalid
    population, or failed consistency rejects the complete transaction.
    Electrostatic field energy is reclosed by Poisson rather than falsely
    conserved across changed geometry.
    """
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance < 0
        or not np.isfinite(absolute_particle_tolerance)
        or absolute_particle_tolerance < 0
        or not np.isfinite(absolute_energy_tolerance)
        or absolute_energy_tolerance < 0
    ):
        raise ValueError("Conservation tolerances must be finite and nonnegative.")
    routes, interpolation = _transition_routes(
        prepared, target_prepared, transition, source_result
    )
    old, new = prepared.plan, target_prepared.plan
    if old.layout.names != new.layout.names:
        raise ValueError("Conservative reprepare requires the same named physics layout.")
    if any(
        isinstance(model, SemiconductorMaterial)
        and model.incomplete_ionization is not None
        for model in (*old.material_models, *new.material_models)
    ):
        raise ValueError(
            "Equilibrium incomplete ionization has no transient bound-population "
            "inventory; use explicit DynamicTrap populations before transfer."
        )
    if len(old.traps) != len(new.traps):
        raise ValueError("Source and target must bind the same trap populations.")
    for source_binding, target_binding in zip(old.traps, new.traps, strict=True):
        source_trap, target_trap = source_binding.trap, target_binding.trap
        if (
            source_trap.population_kind != target_trap.population_kind
            or source_trap.energy_reference != target_trap.energy_reference
            or source_trap.empty_charge_number != target_trap.empty_charge_number
            or not np.array_equal(
                np.asarray(source_trap.trap_energy),
                np.asarray(target_trap.trap_energy),
            )
        ):
            raise ValueError(
                "Trap transfer requires the same physical population identity."
            )
    fields = _physical_fields(prepared, point.coordinates)
    if bool(jnp.any(~jnp.isfinite(fields)) | jnp.any(fields < 0)):
        raise ValueError("Source particle densities must be finite and nonnegative.")
    old_keys, new_keys = _material_keys(old), _material_keys(new)
    same_material = old_keys[routes] == new_keys[:, None]
    allowed = (
        same_material
        & np.asarray(old.semiconductor_mask)[routes]
        & np.asarray(new.semiconductor_mask)[:, None]
    )
    weights = jnp.where(jnp.asarray(allowed), interpolation, 0.0)
    coverage = (
        jnp.zeros_like(old.support.volumes)
        .at[routes]
        .add(new.support.volumes[:, None] * weights)
    )
    if bool(jnp.any(old.semiconductor_mask & (coverage <= 0))):
        raise ValueError(
            "Native transition leaves semiconductor source particles without a target route."
        )
    safe_coverage = jnp.where(old.semiconductor_mask, coverage, 1.0)
    mass_density = old.support.volumes[:, None] * fields / safe_coverage[:, None]
    transferred = jnp.sum(weights[..., None] * mass_density[routes], axis=1)
    if bool(
        jnp.any(
            new.semiconductor_mask & ((transferred[:, 0] <= 0) | (transferred[:, 1] <= 0))
        )
    ):
        raise ValueError(
            "Transfer cannot create positive carriers on an unsupported target region."
        )
    plan = new.with_doping(transferred[:, 2], transferred[:, 3])
    candidate = PreparedSemiconductorDevice(plan)
    # Interpolate physical electrostatic voltage, never its temperature-scaled chart.
    source_potential = old.thermal_voltage * prepared.field(
        point.coordinates, "potential"
    )
    potential = (
        jnp.sum(interpolation * source_potential[routes], axis=1) / plan.thermal_voltage
    )
    safe_n = jnp.where(plan.semiconductor_mask, transferred[:, 0], plan.intrinsic_density)
    safe_p = jnp.where(plan.semiconductor_mask, transferred[:, 1], plan.intrinsic_density)
    guess = plan.equilibrium_coordinates()
    guess = candidate.layout.set(guess, "potential", potential)
    guess = candidate.coordinates_from_densities(guess, safe_n, safe_p)
    source_storage = prepared.storage(point.coordinates) * prepared.storage_scale
    chart = candidate.storage_coordinates(guess)
    material_weights = jnp.where(jnp.asarray(same_material), interpolation, 0.0)
    for name in candidate.layout.bulk_names:
        if not name.endswith("_energy"):
            continue
        source_active = prepared.field(prepared.differential_mask, name)
        target_active = candidate.field(candidate.differential_mask, name)
        weights_for_field = material_weights * (
            target_active[:, None] & source_active[routes]
        )
        coverage_for_field = (
            jnp.zeros_like(old.support.volumes)
            .at[routes]
            .add(new.support.volumes[:, None] * weights_for_field)
        )
        if bool(jnp.any(source_active & (coverage_for_field <= 0))):
            raise ValueError(
                "Native transition leaves differential material energy without "
                "a same-material differential target."
            )
        safe_coverage = jnp.where(source_active, coverage_for_field, 1.0)
        source_extensive = prepared.field(source_storage, name)
        target_density = jnp.sum(
            weights_for_field * (source_extensive / safe_coverage)[routes],
            axis=1,
        )
        target_extensive = new.support.volumes * target_density
        target_scale = candidate.field(candidate.storage_scale, name)
        target_chart = target_extensive / target_scale
        chart = candidate.layout.set(
            chart,
            name,
            jnp.where(
                target_active,
                target_chart,
                candidate.field(chart, name),
            ),
        )
    for index, (source_binding, target_binding) in enumerate(
        zip(old.traps, new.traps, strict=True)
    ):
        source_count = jnp.sum(prepared.field(source_storage, f"trap_{index}"))
        _, target_measure = candidate._trap_geometry(target_binding)
        target_capacity = target_measure * target_binding.trap.density
        occupancy = source_count / target_capacity
        if not bool(jnp.isfinite(occupancy) & (occupancy > 0) & (occupancy < 1)):
            raise ValueError(
                "Target trap measure cannot represent the conserved occupied population."
            )
        chart = candidate.layout.set(chart, f"trap_{index}", occupancy)
    guess = candidate.coordinates_from_storage(chart)
    problem = _dae_problem(candidate, guess, lambda time: point.voltages)
    selected = _dae_policy(policy, guess.size)
    initialization = initialize_dae(problem, jnp.asarray(0.0), policy=selected)
    coordinates = _coordinates_from_storage(
        candidate, initialization.state.reshape(guess.shape)
    )
    rates, rate_record = _consistent_rate(
        candidate,
        coordinates,
        point.voltages,
        jnp.zeros_like(point.voltages),
        _linear_policy(None, coordinates.size),
    )
    rate_evidence = _evidence((rate_record,), initialization.valid, (1,))
    final_fields = _physical_fields(candidate, coordinates)
    source_counts = _counts(prepared, fields)
    transferred_counts = _counts(candidate, transferred)
    final_counts = _counts(candidate, final_fields)
    threshold = absolute_particle_tolerance + relative_tolerance * jnp.abs(source_counts)
    error = final_counts - source_counts
    source_energy, source_traps = _stored_material_energy(prepared, point.coordinates)
    transferred_energy, transferred_traps = _stored_material_energy(candidate, guess)
    final_energy, final_traps = _stored_material_energy(candidate, coordinates)
    energy_threshold = absolute_energy_tolerance + relative_tolerance * jnp.abs(
        source_energy
    )
    energy_error = final_energy - source_energy
    trap_threshold = absolute_particle_tolerance + relative_tolerance * jnp.abs(
        source_traps
    )
    energy_conservative = (
        jnp.abs(transferred_energy - source_energy) <= energy_threshold
    ) & (jnp.abs(energy_error) <= energy_threshold)
    trap_conservative = jnp.all(
        jnp.abs(transferred_traps - source_traps) <= trap_threshold
    ) & jnp.all(jnp.abs(final_traps - source_traps) <= trap_threshold)
    positive = jnp.all(jnp.isfinite(final_fields)) & jnp.all(final_fields >= 0)
    conservative = (
        jnp.all(jnp.abs(error) <= threshold)
        & jnp.all(jnp.abs(transferred_counts - source_counts) <= threshold)
        & energy_conservative
        & trap_conservative
    )
    accepted = (
        jnp.all(point.successful)
        & initialization.valid
        & rate_record[0]
        & positive
        & conservative
    )
    # Elementary charge is exact in SI. Particle-wise conservation is stronger
    # than cancellation-prone conservation of the net dopant/carrier charge.
    signs = jnp.asarray([-1.0, 1.0, 1.0, -1.0])
    q = 1.602176634e-19
    evidence = SemiconductorTransferEvidence(
        source_counts,
        transferred_counts,
        final_counts,
        error,
        threshold,
        q * jnp.sum(signs * source_counts),
        q * jnp.sum(signs * final_counts),
        source_energy,
        transferred_energy,
        final_energy,
        energy_error,
        energy_threshold,
        source_traps,
        transferred_traps,
        final_traps,
        positive,
        conservative,
        initialization.valid,
        accepted,
        initialization,
        rate_evidence,
        transition.transition_id,
    )
    if bool(accepted):
        return SemiconductorTransferResult(
            candidate, coordinates, rates, accepted, evidence, candidate, coordinates
        )
    return SemiconductorTransferResult(
        prepared,
        point.coordinates,
        jnp.zeros_like(point.coordinates),
        accepted,
        evidence,
        candidate,
        coordinates,
    )


def semiconductor_adaptation_indicators(
    prepared: PreparedSemiconductorDevice,
    coordinates: Array,
    voltages: Array,
    /,
) -> SemiconductorAdaptationIndicators:
    """Nodal residual/current imbalance and edge SI/log gradients for meshing.

    These are indicators, not certified error bounds or a topology algorithm.
    Native meshing decides marking, refinement/coarsening and geometric quality.
    """
    support = prepared.plan.support
    u = jnp.asarray(coordinates)
    tail, head = support.tail, support.head
    lengths = prepared.plan.edge_lengths
    bulk_residual = jnp.stack(
        tuple(
            jnp.abs(
                prepared.field(prepared.time_scale * prepared.residual(u, voltages), name)
            )
            for name in prepared.layout.bulk_names
        ),
        axis=-1,
    )
    residual = jnp.max(bulk_residual, axis=-1)
    electrons, holes = prepared.edge_fluxes(u)
    current = 1.602176634e-19 * (holes - electrons)
    divergence = (
        jnp.zeros_like(support.volumes).at[tail].add(-current).at[head].add(current)
    )
    incident = (
        jnp.zeros_like(support.volumes)
        .at[tail]
        .add(jnp.abs(current))
        .at[head]
        .add(jnp.abs(current))
    )
    jump = jnp.where(
        incident > 0, jnp.abs(divergence) / jnp.where(incident > 0, incident, 1.0), 0.0
    )
    potential = prepared.field(u, "potential")
    potential_gradient = (
        prepared.plan.thermal_voltage * (potential[head] - potential[tail]) / lengths
    )
    electron, hole = prepared.densities(u)
    log_n = jnp.log(jnp.where(prepared.plan.semiconductor_mask, electron, 1.0))
    log_p = jnp.log(jnp.where(prepared.plan.semiconductor_mask, hole, 1.0))
    gradients = (
        jnp.stack((log_n[head] - log_n[tail], log_p[head] - log_p[tail]), axis=-1)
        / lengths[:, None]
    )
    active = (
        prepared.plan.semiconductor_mask[head] & prepared.plan.semiconductor_mask[tail]
    )
    return SemiconductorAdaptationIndicators(
        residual, jump, potential_gradient, jnp.where(active[:, None], gradients, 0.0)
    )


__all__ = [
    "SemiconductorTransferEvidence",
    "SemiconductorTransferResult",
    "SemiconductorAdaptationIndicators",
    "semiconductor_reprepare",
    "semiconductor_adaptation_indicators",
]
