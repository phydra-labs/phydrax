#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization import TopologyEpoch
from ...geometry.multiregion_surface import (
    ConservativeFieldTransfer,
    ExtensiveTransferEvidence,
)
from ...lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionTransport,
)
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
from ._device import DevicePlan
from ._materials import SemiconductorMaterial
from ._quantities import ELEMENTARY_CHARGE_SI
from ._support import TransportSupport


# Composition entries of one adapted device; the owner stages all three.
_OWNER = "semiconductor"
_MESH = "semiconductor/mesh"
_DEVICE = "semiconductor/device"
_STATE = "semiconductor/operating-point"
# Extensive particle inventories carried on the control volumes, in order.
_PARTICLES = ("electrons", "holes", "donors", "acceptors")
_CARRIER_ROUTE = "carriers-and-dopants"


class SemiconductorTransferEvidence(StrictModule):
    """Named conserved inventories through redistribution and electrostatic reclosure.

    ``inventories`` names every component of the inventory vectors: electron,
    hole, donor and acceptor counts, each bound trap population by binding order,
    then each differential stored material energy. ``redistribution`` holds the
    shared positive conservative transfer evidence of each route in
    ``redistribution_routes``. Electrostatic field energy is reclosed by Poisson
    and is never an inventory.
    """

    source_inventory: Array
    transferred_inventory: Array
    reinitialized_inventory: Array
    inventory_error: Array
    inventory_tolerance: Array
    redistribution: tuple[ExtensiveTransferEvidence, ...]
    source_charge: Array
    reinitialized_charge: Array
    positive: Array
    conservative: Array
    initialization_valid: Array
    accepted: Array
    initialization: Any
    rate_evidence: SemiconductorLinearEvidence
    inventories: tuple[str, ...] = eqx.field(static=True)
    redistribution_routes: tuple[str, ...] = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)


class SemiconductorTransferResult(StrictModule):
    """Atomic result of one composition rebind of an adapted device.

    ``prepared`` and ``coordinates`` are read from ``receipt.composition``: the
    staged target when the rebind is published, otherwise the exact accepted
    source objects. ``candidate_*`` retain the unpublished staged device and
    reclosed state for diagnosis.
    """

    prepared: PreparedSemiconductorDevice
    coordinates: Array
    coordinate_rates: Array
    accepted: Array
    evidence: SemiconductorTransferEvidence
    candidate_prepared: PreparedSemiconductorDevice
    candidate_coordinates: Array
    receipt: CompositionRebindReceipt


class SemiconductorAdaptationIndicators(StrictModule):
    residual: Array
    current_jump: Array
    potential_gradient: Array
    carrier_log_gradient: Array


def _transition_routes(
    prepared: PreparedSemiconductorDevice,
    target: PreparedSemiconductorDevice,
    transition: CellMeshTransition,
    source_result: CellMeshingResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Host source rows and nonnegative weights of each target transport node."""
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
    weights = np.where(valid, np.asarray(stencil.weights, dtype=np.float64)[rows], 0.0)
    if np.any(weights < 0) or not stencil.preserves_constants:
        raise ValueError(
            "Positive conservative transfer requires a nonnegative constant-preserving stencil."
        )
    routes = np.zeros(sources.shape, dtype=np.int64)
    for row, column in zip(*np.nonzero(valid), strict=True):
        identifier = int(sources[row, column])
        if identifier not in source_lookup:
            raise ValueError("Native transition references an unavailable source vertex.")
        routes[row, column] = source_lookup[identifier]
    return routes, weights


def _material_identity(model: Any, /) -> str:
    """Canonical identity of one complete material model, never its display name."""
    return canonical_fingerprint(
        {
            "kind": "semiconductor-material",
            "family": f"{type(model).__module__}.{type(model).__qualname__}",
            "structure": str(jax.tree_util.tree_structure(model)),
            "arrays": array_tree_fingerprint(model),
        }
    )


def _material_codes(old: DevicePlan, new: DevicePlan, /) -> tuple[np.ndarray, np.ndarray]:
    """Per-node material identity codes shared by the source and target plans."""
    old_identities = [_material_identity(model) for model in old.material_models]
    new_identities = [_material_identity(model) for model in new.material_models]
    catalog = {
        identity: code
        for code, identity in enumerate(sorted({*old_identities, *new_identities}))
    }
    old_codes = np.asarray([catalog[item] for item in old_identities], dtype=np.int64)
    new_codes = np.asarray([catalog[item] for item in new_identities], dtype=np.int64)
    return (
        old_codes[np.asarray(old.material_index)],
        new_codes[np.asarray(new.material_index)],
    )


def _check_physics(old: DevicePlan, new: DevicePlan, /) -> None:
    """Source and target must declare the same named state and bound populations."""
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


def _redistribution(
    routes: np.ndarray,
    weights: np.ndarray,
    admissible: np.ndarray,
    source_active: np.ndarray,
    target_active: np.ndarray,
    target_volumes: np.ndarray,
    inventory: str,
    /,
) -> ConservativeFieldTransfer:
    """Positive partition of every source control-volume inventory over its routes.

    Target volume ``j`` draws on source ``i`` through its admissible stencil
    slots with share ``V_j w_ji / c_i``, where ``c_i`` is the total target volume
    covering ``i``. The shares of every active source sum to one, so the
    inventory is conserved exactly and stays nonnegative; the stencil only
    supplies the locality of the routes, never the transported field space.
    """
    share = np.where(admissible, target_volumes[:, None] * weights, 0.0)
    selected = share > 0
    sources = routes[selected]
    targets = np.broadcast_to(
        np.arange(routes.shape[0], dtype=np.int64)[:, None], routes.shape
    )[selected]
    share = share[selected]
    coverage = np.bincount(sources, weights=share, minlength=source_active.size)
    if np.any(source_active & (coverage <= 0)):
        raise ValueError(
            f"Native transition leaves source {inventory} without a same-material "
            "target route."
        )
    received = np.bincount(targets, weights=share, minlength=target_active.size)
    if np.any(target_active & (received <= 0)):
        raise ValueError(
            f"Transfer cannot create {inventory} on a target control volume without "
            "a same-material source route."
        )
    pairs, inverse = np.unique(
        np.stack((sources, targets), axis=1), axis=0, return_inverse=True
    )
    merged = np.bincount(
        inverse.reshape(-1),
        weights=share / coverage[sources],
        minlength=pairs.shape[0],
    )
    return ConservativeFieldTransfer(
        pairs[:, 0],
        pairs[:, 1],
        merged,
        source_active=source_active,
        target_active=target_active,
    )


def _physical_fields(prepared: PreparedSemiconductorDevice, coordinates: Array) -> Array:
    n, p = prepared.densities(coordinates)
    return jnp.stack(
        (n, p, prepared.plan.donor_density, prepared.plan.acceptor_density), axis=-1
    )


def _energy_names(prepared: PreparedSemiconductorDevice, /) -> tuple[str, ...]:
    return tuple(name for name in prepared.layout.bulk_names if name.endswith("_energy"))


def _inventory_names(prepared: PreparedSemiconductorDevice, /) -> tuple[str, ...]:
    return (
        *_PARTICLES,
        *(f"trap_{index}" for index in range(len(prepared.plan.traps))),
        *_energy_names(prepared),
    )


def _inventory(prepared: PreparedSemiconductorDevice, coordinates: Array, /) -> Array:
    """Extensive inventories in `_inventory_names` order (counts, then joules)."""
    fields = _physical_fields(prepared, coordinates)
    particles = jnp.sum(prepared.plan.support.volumes[:, None] * fields, axis=0)
    stored = prepared.storage(coordinates) * prepared.storage_scale
    bound = tuple(
        jnp.sum(prepared.field(stored, f"trap_{index}"))
        for index in range(len(prepared.plan.traps))
    )
    energies = tuple(
        jnp.sum(prepared.field(stored, name)) for name in _energy_names(prepared)
    )
    return jnp.concatenate(
        (particles, jnp.asarray((*bound, *energies), dtype=particles.dtype))
    )


def _topology_epoch(support: TransportSupport, index: int, /) -> TopologyEpoch:
    """Topology epoch of one meshed device; the control-volume partition is its layout."""
    return TopologyEpoch(
        index,
        canonical_fingerprint(
            {
                "kind": "semiconductor-mesh-geometry",
                "mesh": support.source_id,
                "revision": support.source_revision,
            }
        ),
        support.source_topology_id,
        support.support_id,
    )


def _stage_candidate(
    prepared: PreparedSemiconductorDevice,
    point: SemiconductorOperatingPoint,
    target_prepared: PreparedSemiconductorDevice,
    routes: np.ndarray,
    interpolation: np.ndarray,
    /,
) -> tuple[
    PreparedSemiconductorDevice,
    Array,
    tuple[ConservativeFieldTransfer, ...],
    tuple[ExtensiveTransferEvidence, ...],
    tuple[str, ...],
]:
    """Material-routed positive conservative transfer of every inventory.

    Returns the target device with transferred doping, the unreclosed chart of
    the transferred inventories, and each route's transfer and evidence.
    """
    old, new = prepared.plan, target_prepared.plan
    fields = _physical_fields(prepared, point.coordinates)
    if bool(jnp.any(~jnp.isfinite(fields)) | jnp.any(fields < 0)):
        raise ValueError("Source particle densities must be finite and nonnegative.")
    old_codes, new_codes = _material_codes(old, new)
    same_material = (old_codes[routes] == new_codes[:, None]) & (interpolation > 0)
    target_volumes = np.asarray(new.support.volumes, dtype=np.float64)
    source_semiconductor = np.asarray(old.semiconductor_mask, dtype=np.bool_)
    target_semiconductor = np.asarray(new.semiconductor_mask, dtype=np.bool_)
    carriers = _redistribution(
        routes,
        interpolation,
        same_material & source_semiconductor[routes] & target_semiconductor[:, None],
        source_semiconductor,
        target_semiconductor,
        target_volumes,
        "carriers and dopants",
    )
    source_counts = old.support.volumes[:, None] * fields
    target_counts = carriers.apply(source_counts)
    transfers = [carriers]
    evidence = [carriers.evidence(source_counts, target_counts)]
    transferred = target_counts / new.support.volumes[:, None]
    plan = new.with_doping(transferred[:, 2], transferred[:, 3])
    candidate = PreparedSemiconductorDevice(plan)
    # The interpolated physical potential is only the Newton guess of the
    # electrostatic reclosure, never a transported quantity; the reference
    # temperature chart is never interpolated.
    source_potential = old.thermal_voltage * prepared.field(
        point.coordinates, "potential"
    )
    potential = (
        jnp.sum(
            jnp.asarray(interpolation) * source_potential[jnp.asarray(routes)], axis=1
        )
        / plan.thermal_voltage
    )
    safe_n = jnp.where(plan.semiconductor_mask, transferred[:, 0], plan.intrinsic_density)
    safe_p = jnp.where(plan.semiconductor_mask, transferred[:, 1], plan.intrinsic_density)
    guess = plan.equilibrium_coordinates()
    guess = candidate.layout.set(guess, "potential", potential)
    guess = candidate.coordinates_from_densities(guess, safe_n, safe_p)
    source_storage = prepared.storage(point.coordinates) * prepared.storage_scale
    chart = candidate.storage_coordinates(guess)
    for name in _energy_names(candidate):
        source_active = np.asarray(
            prepared.field(prepared.differential_mask, name), dtype=np.bool_
        )
        target_active = np.asarray(
            candidate.field(candidate.differential_mask, name), dtype=np.bool_
        )
        energy = _redistribution(
            routes,
            interpolation,
            same_material & source_active[routes] & target_active[:, None],
            source_active,
            target_active,
            target_volumes,
            f"differential {name}",
        )
        source_extensive = prepared.field(source_storage, name)
        target_extensive = energy.apply(source_extensive)
        transfers.append(energy)
        evidence.append(energy.evidence(source_extensive, target_extensive))
        chart = candidate.layout.set(
            chart,
            name,
            jnp.where(
                jnp.asarray(target_active),
                target_extensive / candidate.field(candidate.storage_scale, name),
                candidate.field(chart, name),
            ),
        )
    # A bound population keeps its extensive occupied count by binding order at
    # the uniform occupancy the target capacity admits.
    for index, target_binding in enumerate(new.traps):
        source_count = jnp.sum(prepared.field(source_storage, f"trap_{index}"))
        _, target_measure = candidate._trap_geometry(target_binding)
        occupancy = source_count / (target_measure * target_binding.trap.density)
        if not bool(jnp.isfinite(occupancy) & (occupancy > 0) & (occupancy < 1)):
            raise ValueError(
                "Target trap measure cannot represent the conserved occupied population."
            )
        chart = candidate.layout.set(chart, f"trap_{index}", occupancy)
    return (
        candidate,
        candidate.coordinates_from_storage(chart),
        tuple(transfers),
        tuple(evidence),
        (_CARRIER_ROUTE, *_energy_names(candidate)),
    )


def _entries(
    epoch: TopologyEpoch,
    mesh: CellMeshingResult,
    device: PreparedSemiconductorDevice,
    coordinates: Array,
    voltages: Array,
    /,
) -> tuple[CompositionEntry, CompositionEntry, CompositionEntry]:
    """Mesh topology, prepared device and operating point of one device epoch."""
    support, plan = device.plan.support, device.plan
    topology = CompositionEntry(
        mesh,
        entry_id=_MESH,
        role="topology",
        owner_id="meshing",
        structure_id=epoch.epoch_id,
        revision_id=epoch.geometry_id,
        semantics_id="semiconductor-transport-mesh",
    )
    discretization = CompositionEntry(
        device,
        entry_id=_DEVICE,
        role="discretization",
        owner_id=_OWNER,
        structure_id=epoch.epoch_id,
        revision_id=canonical_fingerprint(
            {
                "kind": "prepared-semiconductor-device",
                "support": support.support_id,
                "plan": array_tree_fingerprint(plan),
            }
        ),
        semantics_id=canonical_fingerprint(
            {
                "kind": "semiconductor-device-semantics",
                "layout": list(plan.layout.names),
                "terminals": list(plan.terminal_names),
            }
        ),
        dependencies=(topology.binding("structure"), topology.binding("revision")),
    )
    state = CompositionEntry(
        coordinates,
        entry_id=_STATE,
        role="physical-state",
        owner_id=_OWNER,
        structure_id=epoch.epoch_id,
        revision_id=canonical_fingerprint(
            {
                "kind": "semiconductor-operating-point-revision",
                "coordinates": array_tree_fingerprint(coordinates),
            }
        ),
        semantics_id=canonical_fingerprint(
            {
                "kind": "semiconductor-operating-point",
                "layout": list(plan.layout.names),
                "voltages": array_tree_fingerprint(voltages),
            }
        ),
        dependencies=(
            discretization.binding("structure"),
            discretization.binding("revision"),
        ),
    )
    return topology, discretization, state


def _stage_rebind(
    source: tuple[CompositionEntry, CompositionEntry, CompositionEntry],
    target: tuple[CompositionEntry, CompositionEntry, CompositionEntry],
    transition_id: str,
    transfers: tuple[ConservativeFieldTransfer, ...],
    names: tuple[str, ...],
    successful: Array,
    source_inventory: Array,
    reclosed_inventory: Array,
    tolerance: Array,
    /,
) -> CompositionRebind:
    """Reprepare mesh and device; transport the reclosed state with its inventories."""
    mesh, _, state = source
    transport = CompositionTransport(
        "physical-remap",
        (state.entry_id,),
        (target[2],),
        source_structure_ids=(state.structure_id,),
        route_id=canonical_fingerprint(
            {
                "kind": "semiconductor-inventory-remap",
                "transition": transition_id,
                "transfers": [item.transfer_id for item in transfers],
                "inventories": list(names),
            }
        ),
        successful=successful,
        source_content=source_inventory,
        target_content=reclosed_inventory,
        content_tolerance=tolerance,
    )
    return CompositionRebind(
        Composition(
            source,
            boundary_id=canonical_fingerprint(
                {
                    "kind": "semiconductor-accepted-operating-point",
                    "epoch": mesh.structure_id,
                    "state": state.revision_id,
                }
            ),
        ),
        reprepare=target[:2],
        transports=(transport,),
    )


def semiconductor_reprepare(
    prepared: PreparedSemiconductorDevice,
    point: SemiconductorOperatingPoint,
    target_prepared: PreparedSemiconductorDevice,
    transition: CellMeshTransition,
    /,
    *,
    source_result: CellMeshingResult,
    epoch_index: int,
    policy: DAESolvePolicy | None = None,
    relative_tolerance: float = 1e-10,
    absolute_particle_tolerance: float = 1e-12,
    absolute_energy_tolerance: float = 1e-24,
) -> SemiconductorTransferResult:
    """Rebind an adapted device: conservative inventory transfer, then reclosure.

    ``epoch_index`` is the index of the accepted source topology epoch; the
    adapted device is epoch ``epoch_index + 1``. Material identity is the
    canonical identity of each complete material model. Carrier and dopant
    counts per control volume and each differential stored material energy
    cross only same-material routes through a shared positive conservative
    transfer; bulk or surface trap populations keep their extensive occupied
    count by binding order at the uniform occupancy the target capacity admits.
    No logarithm, temperature, occupancy logit or eigenvector phase is
    interpolated.

    Electrostatic reclosure is the consumer's explicit nonlinear step: algebraic
    potentials and interface traces are re-solved by native DAE initialization
    from the transferred inventories. Electrostatic field energy is reclosed by
    Poisson rather than falsely conserved across changed geometry.

    The mesh (``topology``), prepared device (``discretization``) and operating
    point (``physical-state``) are staged as one `CompositionRebind`. The state
    crosses by one ``physical-remap`` transport reporting every named source
    inventory against its reclosed value with the conservation tolerance, and
    succeeds only for positive redistribution evidence and a positive, finite
    reclosed state. Reclosure success and the source point's success decide the
    accepted boundary. Anything else leaves the receipt unpublished and the
    result on the exact accepted source device and coordinates. Stale lineage,
    mismatched material identity or unsupported populations raise before any
    candidate is staged.
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
    _check_physics(prepared.plan, target_prepared.plan)
    source_epoch = _topology_epoch(prepared.plan.support, epoch_index)
    target_epoch = _topology_epoch(target_prepared.plan.support, epoch_index + 1)
    candidate, guess, transfers, redistribution, route_names = _stage_candidate(
        prepared, point, target_prepared, routes, interpolation
    )
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
    names = _inventory_names(prepared)
    source_inventory = _inventory(prepared, point.coordinates)
    transferred_inventory = _inventory(candidate, guess)
    final_inventory = _inventory(candidate, coordinates)
    particle_count = len(names) - len(_energy_names(prepared))
    tolerance = jnp.where(
        jnp.arange(len(names)) < particle_count,
        absolute_particle_tolerance,
        absolute_energy_tolerance,
    ) + relative_tolerance * jnp.abs(source_inventory)
    error = final_inventory - source_inventory
    final_fields = _physical_fields(candidate, coordinates)
    positive = (
        jnp.all(jnp.isfinite(final_fields))
        & jnp.all(final_fields >= 0)
        & jnp.all(jnp.isfinite(coordinates))
    )
    redistributed = jnp.all(
        jnp.stack(tuple(item.successful for item in redistribution))
    ) & jnp.all(jnp.abs(transferred_inventory - source_inventory) <= tolerance)
    conservative = redistributed & jnp.all(jnp.abs(error) <= tolerance)
    reclosed = initialization.valid & rate_record[0]
    source_entries = _entries(
        source_epoch, source_result, prepared, point.coordinates, point.voltages
    )
    target_entries = _entries(
        target_epoch, transition.target, candidate, coordinates, point.voltages
    )
    rebind = _stage_rebind(
        source_entries,
        target_entries,
        transition.transition_id,
        transfers,
        names,
        redistributed & positive,
        source_inventory,
        final_inventory,
        tolerance,
    )
    receipt = commit_composition_rebind(
        rebind, accepted_boundary=bool(jnp.all(point.successful) & reclosed)
    )
    # Elementary charge is exact in SI. Particle-wise conservation is stronger
    # than cancellation-prone conservation of the net dopant/carrier charge.
    signs = jnp.asarray([-1.0, 1.0, 1.0, -1.0])
    q = ELEMENTARY_CHARGE_SI
    evidence = SemiconductorTransferEvidence(
        source_inventory,
        transferred_inventory,
        final_inventory,
        error,
        tolerance,
        redistribution,
        q * jnp.sum(signs * source_inventory[:4]),
        q * jnp.sum(signs * final_inventory[:4]),
        positive,
        conservative,
        initialization.valid,
        jnp.asarray(receipt.published),
        initialization,
        rate_evidence,
        names,
        route_names,
        transition.transition_id,
    )
    return SemiconductorTransferResult(
        receipt.composition.value(_DEVICE),
        receipt.composition.value(_STATE),
        rates if receipt.published else jnp.zeros_like(point.coordinates),
        jnp.asarray(receipt.published),
        evidence,
        candidate,
        coordinates,
        receipt,
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
    current = ELEMENTARY_CHARGE_SI * (holes - electrons)
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
