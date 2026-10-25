"""Shared fixtures of distributed atomistic domain tests.

The global oracle enumerates every periodic image route on the host by brute
force and evaluates the same native MACE layers on the global graph
(``R = C = N``) with ordinary JAX: an independent single-device native
reference of the same model, never the domain route itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group, ExecutionRuntime
from phydrax._fingerprint import canonical_fingerprint
from phydrax.atomistic import AtomisticScaleContract
from phydrax.atomistic._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionMode,
    AtomisticDomainReductionPolicy,
    AtomisticDomainRolePlan,
)
from phydrax.atomistic._domain_runtime import (
    prepare_atomistic_domain,
    PreparedAtomisticDomainRuntime,
)
from phydrax.atomistic._domain_state import AtomisticDomainState
from phydrax.atomistic._neighborhood_bundle import AtomisticNeighborhoodRole
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.sparse import EdgeRelation, RelationAccumulation
from phydrax.sparse._streamed import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 2.4
SKIN = 0.3
SPECIES = (1, 8)
# A triclinic cell whose shortest lattice translation is below the cutoff, so
# atoms see several images of the same source and of themselves.
CELL = np.array([[2.2, 0.0, 0.0], [0.6, 2.5, 0.0], [0.3, 0.4, 2.8]])
LOCAL_CAPACITY = 8


def mace_model(layers: int, key: int = 0, *, cutoff: float = CUTOFF) -> MACEPotential:
    kinds = ("real-agnostic",) + ("real-agnostic-residual",) * (layers - 1)
    architecture = MACEArchitecture(
        species=SPECIES,
        cutoff=cutoff,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=kinds,
        correlations=(2,) * layers,
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=8.0,
    )
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.array([-0.3, -1.1]),
        key=jax.random.key(key),
    )


def streaming(
    model: MACEPotential, accumulation: RelationAccumulation = "fast"
) -> StreamedRelationPlan:
    return StreamedRelationPlan(
        receiver_tile=4,
        edge_tile=32,
        channel_capacity=model.configuration.message_payload_width(),
        accumulation=accumulation,
    )


def ownership(
    owners: int,
    *,
    devices: bool = False,
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
    local_capacity: int = LOCAL_CAPACITY,
) -> DistributedOwnershipPlan:
    """Lane-reference (one device) or device-collective owner placement."""
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=periodic_axes
    )
    if jax.process_count() != 1:
        pytest.skip("These global-host-ingress cases require a single JAX process.")
    runtime = ExecutionRuntime.current()
    parent = runtime.root_group.spec
    if devices:
        if parent.device_count < owners:
            pytest.skip(f"Requires at least {owners} visible devices.")
        keys = parent.device_keys[:owners]
        group = bind_execution_group(
            ExecutionGroupSpec(
                canonical_fingerprint(
                    {
                        "parent_group_id": parent.group_id,
                        "group_index": 0,
                        "device_keys": [list(key) for key in keys],
                    }
                ),
                parent.process_indices,
                keys,
                mesh_axes=(("device", owners),),
                parent_group_id=parent.group_id,
                process_host_ids=parent.process_host_ids,
            )
        )
        return DistributedOwnershipPlan(address, group, local_capacity)
    group = runtime.child_groups(parent.device_count)[0]
    return DistributedOwnershipPlan(address, group, local_capacity, owner_lanes=owners)


def role_plan(
    model: Any,
    cell: np.ndarray = CELL,
    *,
    edge_capacity: int = 512,
    alias_capacity: int = 512,
    halo_capacity: int = 8,
    accumulation: RelationAccumulation = "fast",
    skin: float = SKIN,
    stream: StreamedRelationPlan | None = None,
) -> AtomisticDomainRolePlan:
    requirement = model.requirements.neighborhood(
        AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
    )
    periodic = PeriodicCell(cell)
    return AtomisticDomainRolePlan(
        requirement,
        periodic.image_stencil(
            requirement.bounded_cutoff() + skin, maximum_image_count=4096
        ),
        streaming=streaming(model, accumulation) if stream is None else stream,
        skin=skin,
        edge_capacity=edge_capacity,
        alias_capacity=alias_capacity,
        halo_capacity=halo_capacity,
    )


def domain_plan(
    model: Any,
    *,
    owners: int = 2,
    devices: bool = False,
    cell: np.ndarray = CELL,
    counts: tuple[int, int, int] | None = None,
    edge_capacity: int = 512,
    alias_capacity: int = 512,
    halo_capacity: int = 8,
    migration_capacity: int = 8,
    message_capacity_bytes: int = 1 << 20,
    streaming_accumulation: RelationAccumulation = "fast",
    reduction: AtomisticDomainReductionMode = "deterministic",
    dynamic_cell: bool = False,
    local_capacity: int = LOCAL_CAPACITY,
    stream: StreamedRelationPlan | None = None,
) -> AtomisticDomainPlan:
    periodic = PeriodicCell(cell)
    return AtomisticDomainPlan(
        ownership(owners, devices=devices, local_capacity=local_capacity),
        FractionalOwnerPartition(periodic, (owners, 1, 1) if counts is None else counts),
        (
            role_plan(
                model,
                cell,
                edge_capacity=edge_capacity,
                alias_capacity=alias_capacity,
                halo_capacity=halo_capacity,
                accumulation=streaming_accumulation,
                stream=stream,
            ),
        ),
        migration_capacity=migration_capacity,
        message_capacity_bytes=message_capacity_bytes,
        reduction=AtomisticDomainReductionPolicy(reduction),
        dynamic_cell=dynamic_cell,
    )


def structure(count: int = 6, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fractional = rng.uniform(0.0, 1.0, (count, 3))
    # Two atoms straddle the owner boundary at fractional x = 1/2.
    fractional[0] = (0.47, 0.31, 0.22)
    fractional[1] = (0.53, 0.33, 0.25)
    positions = fractional @ CELL
    # One stored coordinate starts outside the cell (an unwrapped atom).
    positions[2] += CELL[0]
    return positions, np.asarray(SPECIES, dtype=np.int32)[rng.integers(0, 2, count)]


def image_routes(
    positions: np.ndarray, cell: np.ndarray, cutoff: float = CUTOFF
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force directed routes ``(source, receiver, n)`` with ``|d| < cutoff``."""
    count = positions.shape[0]
    reach = 3
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    senders, receivers, shifts = [], [], []
    for receiver in range(count):
        for sender in range(count):
            displacement = positions[receiver] - positions[sender] + grid @ cell
            near = np.linalg.norm(displacement, axis=1) < cutoff
            if receiver == sender:
                near &= np.any(grid != 0, axis=1)
            for shift in grid[near]:
                senders.append(sender)
                receivers.append(receiver)
                shifts.append(shift)
    return (
        np.asarray(senders, dtype=np.int32),
        np.asarray(receivers, dtype=np.int32),
        np.asarray(shifts, dtype=np.int32),
    )


def global_energy(
    model: MACEPotential,
    positions: jax.Array,
    species_ids: np.ndarray,
    strain: jax.Array,
    routes: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell: np.ndarray = CELL,
) -> jax.Array:
    """Single-device global MACE energy at strain ``F = I + strain``."""
    senders, receivers, shifts = routes
    count = positions.shape[0]
    deformation = jnp.eye(3) + strain
    deformed = positions @ deformation.T
    vectors_cell = jnp.asarray(cell) @ deformation.T
    relation = streaming(model).prepare(
        EdgeRelation(senders, receivers, source_size=count, target_size=count),
        owner_id="global-oracle",
    )
    vectors = (
        deformed[receivers] - deformed[senders] + shifts.astype(np.float64) @ vectors_cell
    )
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = model.species_indices(jnp.asarray(species_ids), mask)
    energies, successful = model.node_energies(species, mask, relation, vectors)
    return jnp.where(successful, jnp.sum(energies), jnp.nan)


def global_reference(
    model: MACEPotential,
    positions: np.ndarray,
    species_ids: np.ndarray,
    cell: np.ndarray = CELL,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Global energy, forces and ``dE/d strain`` of the oracle."""
    routes = image_routes(positions, cell)
    zero = jnp.zeros((3, 3))
    reference = jnp.asarray(positions)
    energy = global_energy(model, reference, species_ids, zero, routes, cell)
    forces = -jax.grad(global_energy, argnums=1)(
        model, reference, species_ids, zero, routes, cell
    )
    strain = jax.grad(global_energy, argnums=3)(
        model, reference, species_ids, zero, routes, cell
    )
    return energy, forces, strain


def ingress(
    runtime: PreparedAtomisticDomainRuntime,
    model: MACEPotential,
    positions: np.ndarray,
    species_ids: np.ndarray,
    **kwargs: Any,
) -> AtomisticDomainState:
    mask = jnp.ones(species_ids.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(species_ids), mask))
    return runtime.ingress_global(positions, species, rng_key=jax.random.key(7), **kwargs)


def domain_training_fixture(
    owner_lanes: int,
) -> tuple[
    PreparedAtomisticDomainRuntime,
    Callable[[np.ndarray, np.ndarray], AtomisticDomainState],
    MACEPotential,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Lane-reference two-layer MACE domain over the triclinic self-image cell.

    Returns ``(runtime, ingress, model, species, positions, cell_vectors)``;
    ``species`` are native model indices in logical order and ``ingress``
    builds an accepted state for given positions and current cell vectors.
    """
    model = mace_model(2)
    plan = domain_plan(model, owners=owner_lanes, dynamic_cell=True)
    runtime = prepare_atomistic_domain(plan, model)
    positions, species_ids = structure()
    mask = jnp.ones(species_ids.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(species_ids), mask))

    def build(coordinates: np.ndarray, cell_vectors: np.ndarray) -> AtomisticDomainState:
        return runtime.ingress_global(
            coordinates, species, rng_key=jax.random.key(7), cell_vectors=cell_vectors
        )

    return runtime, build, model, species, positions, CELL.copy()


__all__ = [
    "CELL",
    "CUTOFF",
    "SKIN",
    "SPECIES",
    "domain_plan",
    "domain_training_fixture",
    "global_energy",
    "global_reference",
    "image_routes",
    "ingress",
    "mace_model",
    "ownership",
    "role_plan",
    "streaming",
    "structure",
]
