"""Replicated-geometry graph partition: topology, exchange adjoints, memory.

Topology cases compare every owner's receiver routes with a host brute-force
enumeration of all periodic image routes inside ``cutoff + skin`` (repeated
images of one source and nonzero self images included). Exchange cases check
the tiled all-gather / reduce-scatter adjoint pair inside owner regions.
Execution cases compare the graph-partition route with the halo route and an
independent single-device evaluation of the same native model.

Lane-reference ownership runs the owner regions as named vmap lanes on one
device with the same collectives as device-collective ownership; forced CPU
devices prove functional collective parity only, not accelerator or
multi-host qualification.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group, ExecutionRuntime
from phydrax._fingerprint import canonical_fingerprint
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    energy_and_forces,
)
from phydrax.atomistic._conditioning import AtomisticConditioning
from phydrax.atomistic._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._domain_state import AtomisticDomainStrategy
from phydrax.atomistic._graph_partition import (
    AtomisticGraphPartitionPlan,
    AtomisticGraphPartitionResourceError,
    graph_partition_memory_evidence,
    prepare_graph_partition_topology,
    ReplicatedColumnExchange,
)
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticClusterTileImageSearch,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from phydrax.nn.atomistic._aimnet2_source import (
    aimnet2_potential_from_source,
    externalize_source_long_range,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.sparse import EdgeRelation
from phydrax.sparse._streamed import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.aimnet2_source import RC_S, source_inventory


CUTOFF = 2.4
SKIN = 0.3
# Absolute image-stencil headroom of a dynamic-cell role (the family fixture's).
DYNAMIC_CELL_MARGIN = 0.6
LOCAL = 8
# A triclinic cell whose shortest lattice translation is below the cutoff, so
# atoms see several images of the same source and of themselves.
CELL = np.array([[2.2, 0.0, 0.0], [0.6, 2.5, 0.0], [0.3, 0.4, 2.8]])
# Non-contiguous stable IDs that include 0, the padding-row ID of host ingress.
STABLE_IDS = np.array([5, 0, 12, 3, 9, 1], dtype=np.int64)


def _ownership(owners: int, *, devices: bool = False) -> DistributedOwnershipPlan:
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
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
        return DistributedOwnershipPlan(address, group, LOCAL)
    group = runtime.child_groups(parent.device_count)[0]
    return DistributedOwnershipPlan(address, group, LOCAL, owner_lanes=owners)


def _role(
    cell: np.ndarray,
    *,
    edge_capacity: int,
    requirement: AtomisticNeighborhoodRequirement | None = None,
    channel_capacity: int = 8,
    stencil_margin: float = 0.0,
) -> AtomisticDomainRolePlan:
    required = (
        AtomisticNeighborhoodRequirement(
            "graph-partition-test", AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH, CUTOFF
        )
        if requirement is None
        else requirement
    )
    return AtomisticDomainRolePlan(
        required,
        PeriodicCell(cell).image_stencil(
            required.bounded_cutoff() + SKIN + stencil_margin, maximum_image_count=4096
        ),
        streaming=StreamedRelationPlan(
            receiver_tile=4, edge_tile=32, channel_capacity=channel_capacity
        ),
        skin=SKIN,
        edge_capacity=edge_capacity,
        alias_capacity=512,
        halo_capacity=8,
    )


def _cell_list() -> AtomisticCellListImageSearch:
    return AtomisticCellListImageSearch(maximum_candidate_slots=1 << 22)


def _cluster_tiles() -> AtomisticClusterTileImageSearch:
    return AtomisticClusterTileImageSearch(
        cluster_width=2,
        maximum_cluster_radius=3.0,
        tile_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=32,
            maximum_edges=4096,
            maximum_degree=4096,
            maximum_images=2048,
            maximum_case_edges=4096,
        ),
        maximum_candidate_slots=1 << 22,
    )


def _plan(
    owners: int,
    *,
    devices: bool = False,
    cell: np.ndarray = CELL,
    edge_capacity: int = 512,
    receiver_edge_capacity: int = 128,
    maximum_particles_per_cell: int = 8,
    replicated_memory_bytes: int = 1 << 20,
    search: Any = None,
    roles: tuple[AtomisticDomainRolePlan, ...] | None = None,
) -> AtomisticDomainPlan:
    periodic = PeriodicCell(cell)
    ownership = _ownership(owners, devices=devices)
    partition = FractionalOwnerPartition(periodic, (owners, 1, 1))
    declared = (_role(cell, edge_capacity=edge_capacity),) if roles is None else roles
    graph = AtomisticGraphPartitionPlan(
        ownership,
        partition,
        declared,
        replicated_memory_bytes=replicated_memory_bytes,
        receiver_edge_capacity=receiver_edge_capacity,
        maximum_particles_per_cell=maximum_particles_per_cell,
        search=_cell_list() if search is None else search,
    )
    return AtomisticDomainPlan(
        ownership,
        partition,
        declared,
        migration_capacity=8,
        message_capacity_bytes=1 << 20,
        strategy=AtomisticDomainStrategy.GRAPH_PARTITION,
        graph_partition=graph,
    )


def _structure(count: int = 6, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fractional = rng.uniform(0.0, 1.0, (count, 3))
    # Two atoms straddle the owner boundary at fractional x = 1/2.
    fractional[0] = (0.47, 0.31, 0.22)
    fractional[1] = (0.53, 0.33, 0.25)
    positions = fractional @ CELL
    # One coordinate starts outside the cell (an unwrapped atom).
    positions[2] += CELL[0]
    return positions


def _ingress(
    plan: AtomisticDomainPlan,
    positions: np.ndarray,
    stable_ids: np.ndarray,
    cell: np.ndarray = CELL,
) -> tuple[DistributedPointLayout, jax.Array, np.ndarray]:
    """Wrap into the cell, own by the fractional partition, distribute rows."""
    fractional = positions @ np.linalg.inv(cell)
    images = np.floor(fractional)
    wrapped = fractional - images
    stored = positions - images @ cell
    owners = np.asarray(plan.partition.owners(jnp.asarray(wrapped)))
    layout = DistributedPointLayout.from_global(
        plan.ownership, wrapped, owners, stable_ids=stable_ids, epoch=0
    )
    return layout, layout.distribute(stored), stored


def _oracle_routes(
    stored: np.ndarray, cell: np.ndarray, radius: float
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Brute-force directed routes ``(source, receiver, n)`` with ``|d| < radius``."""
    reach = 4
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    routes = []
    for receiver in range(stored.shape[0]):
        for source in range(stored.shape[0]):
            displacement = stored[receiver] - stored[source] + grid @ cell
            near = np.sum(displacement * displacement, axis=1) < radius * radius
            if receiver == source:
                near &= np.any(grid != 0, axis=1)
            routes.extend(
                (source, receiver, tuple(int(item) for item in shift))
                for shift in grid[near]
            )
    return routes


def _topology_routes(
    plan: AtomisticDomainPlan, layout: DistributedPointLayout, topology: Any
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Logical ``(source, receiver, n)`` of every valid owner edge."""
    owners = plan.ownership.owner_count
    edges = topology.receivers.shape[0] // owners
    logical = np.asarray(layout.logical_indices)
    valid = np.asarray(topology.valid)
    receivers = np.asarray(topology.receivers)
    columns = np.asarray(topology.columns)
    shifts = np.asarray(topology.shifts)
    owner_of_edge = np.repeat(np.arange(owners), edges)
    rows = owner_of_edge * LOCAL + receivers
    return [
        (
            int(logical[columns[edge]]),
            int(logical[rows[edge]]),
            tuple(int(item) for item in shifts[edge]),
        )
        for edge in np.flatnonzero(valid)
    ]


def _assert_canonical_owner_blocks(
    plan: AtomisticDomainPlan, layout: DistributedPointLayout, topology: Any
) -> None:
    """Valid edges lead every owner block in (receiver id, source id, n) order."""
    owners = plan.ownership.owner_count
    edges = topology.receivers.shape[0] // owners
    ids = np.asarray(layout.stable_ids)
    valid = np.asarray(topology.valid).reshape((owners, edges))
    receivers = np.asarray(topology.receivers).reshape((owners, edges))
    columns = np.asarray(topology.columns).reshape((owners, edges))
    shifts = np.asarray(topology.shifts).reshape((owners, edges, 3))
    source_ids = np.asarray(topology.source_ids).reshape((owners, edges))
    for owner in range(owners):
        count = int(np.sum(valid[owner]))
        assert np.all(valid[owner, :count]) and not np.any(valid[owner, count:])
        rows = owner * LOCAL + receivers[owner, :count]
        assert np.all(np.asarray(layout.active)[rows])
        np.testing.assert_array_equal(
            source_ids[owner, :count], ids[columns[owner, :count]]
        )
        keys = [
            (int(ids[row]), int(ids[column]), tuple(int(item) for item in shift))
            for row, column, shift in zip(
                rows, columns[owner, :count], shifts[owner, :count], strict=True
            )
        ]
        assert keys == sorted(keys)


def _prepare(plan: AtomisticDomainPlan, positions: np.ndarray, ids: np.ndarray) -> Any:
    layout, blocked, stored = _ingress(plan, positions, ids)
    topology = prepare_graph_partition_topology(
        plan,
        plan.roles[0],
        layout,
        blocked,
        jnp.asarray(CELL),
        jnp.zeros((), dtype=jnp.int32),
    )
    return layout, topology, stored


@pytest.mark.parametrize(
    ("owners", "devices", "search"),
    [
        (1, False, "cell-list"),
        (2, False, "cell-list"),
        (3, False, "cell-list"),
        (5, False, "cell-list"),
        (2, False, "cluster-tiles"),
        (2, True, "cell-list"),
        (3, True, "cell-list"),
    ],
    ids=[
        "lanes-1",
        "lanes-2",
        "lanes-3",
        "lanes-5",
        "lanes-2-cluster",
        "devices-2",
        "devices-3",
    ],
)
def test_owner_routes_equal_complete_image_enumeration(
    owners: int, devices: bool, search: str
) -> None:
    plan = _plan(
        owners,
        devices=devices,
        search=_cell_list() if search == "cell-list" else _cluster_tiles(),
    )
    layout, topology, stored = _prepare(plan, _structure(), STABLE_IDS)
    evidence = topology.evidence
    assert bool(evidence.successful)
    assert bool(evidence.capacity_successful) and bool(evidence.scientific_successful)
    assert topology.halo is None
    assert topology.strategy is AtomisticDomainStrategy.GRAPH_PARTITION
    assert topology.column_count == owners * LOCAL
    assert topology.role_id == plan.roles[0].role_id
    columns = np.asarray(topology.columns)
    np.testing.assert_array_equal(np.asarray(topology.route_owners), columns // LOCAL)
    np.testing.assert_array_equal(np.asarray(topology.route_slots), columns % LOCAL)
    observed = _topology_routes(plan, layout, topology)
    expected = _oracle_routes(stored, CELL, CUTOFF + SKIN)
    assert len(observed) == len(set(observed))
    assert sorted(observed) == sorted(expected)
    # Repeated images of one source and nonzero self images are distinct routes.
    pairs = [(source, receiver) for source, receiver, _ in observed]
    assert len(set(pairs)) < len(pairs)
    assert any(source == receiver for source, receiver in pairs)
    _assert_canonical_owner_blocks(plan, layout, topology)
    assert int(evidence.maximum_edge_count) <= int(evidence.edge_capacity)
    assert int(evidence.refused_routes) == 0
    assert int(evidence.halo_capacity) == 0 and bool(evidence.halo_successful)


def test_receiver_partition_assigns_every_route_to_its_receiver_owner() -> None:
    owners = 3
    plan = _plan(owners)
    layout, topology, _ = _prepare(plan, _structure(), STABLE_IDS)
    edges = topology.receivers.shape[0] // owners
    valid = np.asarray(topology.valid).reshape((owners, edges))
    receivers = np.asarray(topology.receivers).reshape((owners, edges))
    active = np.asarray(layout.active).reshape((owners, LOCAL))
    for owner in range(owners):
        rows = receivers[owner][valid[owner]]
        assert np.all(active[owner, rows])
    # Each atom's in-degree is the oracle in-degree, so no receiver is split.
    totals = np.bincount(
        [receiver for _, receiver, _ in _topology_routes(plan, layout, topology)],
        minlength=6,
    )
    _, _, stored = _ingress(plan, _structure(), STABLE_IDS)
    oracle = np.bincount(
        [receiver for _, receiver, _ in _oracle_routes(stored, CELL, CUTOFF + SKIN)],
        minlength=6,
    )
    np.testing.assert_array_equal(totals, oracle)


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_empty_owners_and_a_lone_self_imaged_atom(devices: bool) -> None:
    owners = 3
    plan = _plan(owners, devices=devices)
    positions = np.array([[0.4, 0.7, 1.1]])
    layout, topology, stored = _prepare(plan, positions, np.array([0], dtype=np.int64))
    assert bool(topology.evidence.successful)
    edges = topology.receivers.shape[0] // owners
    per_owner = np.sum(np.asarray(topology.valid).reshape((owners, edges)), axis=1)
    owner = int(np.flatnonzero(np.asarray(layout.active))[0]) // LOCAL
    assert per_owner[owner] > 0
    assert int(np.sum(per_owner)) == int(per_owner[owner])
    observed = _topology_routes(plan, layout, topology)
    assert all(source == 0 and receiver == 0 for source, receiver, _ in observed)
    assert all(any(item != 0 for item in shift) for _, _, shift in observed)
    assert sorted(observed) == sorted(_oracle_routes(stored, CELL, CUTOFF + SKIN))


@pytest.mark.parametrize(
    ("edge_capacity", "receiver_edge_capacity", "maximum_particles_per_cell", "field"),
    [
        (16, 128, 8, "edge"),
        (512, 4, 8, "degree"),
        (512, 128, 2, "cell"),
    ],
    ids=["owner-routes", "receiver-degree", "cell-occupancy"],
)
def test_capacity_failures_are_not_scientific_failures(
    edge_capacity: int,
    receiver_edge_capacity: int,
    maximum_particles_per_cell: int,
    field: str,
) -> None:
    plan = _plan(
        2,
        edge_capacity=edge_capacity,
        receiver_edge_capacity=receiver_edge_capacity,
        maximum_particles_per_cell=maximum_particles_per_cell,
    )
    _, topology, _ = _prepare(plan, _structure(), STABLE_IDS)
    evidence = topology.evidence
    assert not bool(evidence.successful)
    assert not bool(evidence.capacity_successful)
    assert bool(evidence.scientific_successful)
    if field == "edge":
        assert int(evidence.maximum_edge_count) > int(evidence.edge_capacity)
        assert int(evidence.refused_routes) > 0
    elif field == "degree":
        assert int(evidence.maximum_degree) > 4
    else:
        assert int(evidence.maximum_search_load) > int(evidence.search_capacity)


def test_nonfinite_geometry_and_stale_owners_are_scientific_failures() -> None:
    plan = _plan(2)
    positions = _structure()
    layout, blocked, _ = _ingress(plan, positions, STABLE_IDS)
    row = int(np.flatnonzero(np.asarray(layout.active))[0])
    broken = blocked.at[row, 0].set(jnp.nan)
    topology = prepare_graph_partition_topology(
        plan, plan.roles[0], layout, broken, jnp.asarray(CELL), jnp.asarray(1)
    )
    assert not bool(topology.evidence.positions_finite)
    assert not bool(topology.evidence.scientific_successful)
    stale = DistributedPointLayout(
        layout.plan,
        layout.points,
        layout.stable_ids,
        layout.active,
        layout.logical_indices,
        layout.owner_epochs.at[1].add(1),
        layout.stable_ids_unique,
        layout.logical_count,
    )
    topology = prepare_graph_partition_topology(
        plan, plan.roles[0], stale, blocked, jnp.asarray(CELL), jnp.asarray(1)
    )
    assert not bool(topology.evidence.owners_current)
    assert not bool(topology.evidence.successful)
    assert bool(topology.evidence.capacity_successful)


def test_deformed_cell_beyond_the_role_stencil_is_reported() -> None:
    plan = _plan(2)
    layout, blocked, _ = _ingress(plan, _structure(), STABLE_IDS)
    # Shrinking the cell raises the image count needed for cutoff + skin
    # beyond the declared stencil: never silently an incomplete relation.
    squeezed = jnp.asarray(CELL * 0.5)
    topology = prepare_graph_partition_topology(
        plan, plan.roles[0], layout, blocked * 0.5, squeezed, jnp.asarray(1)
    )
    assert not bool(topology.evidence.stencil_complete)
    assert not bool(topology.evidence.successful)
    assert float(np.min(np.asarray(topology.evidence.stencil_margin))) <= 0.0


@pytest.mark.parametrize(
    ("deformation_margin", "admitted"),
    [(0.0, False), (DYNAMIC_CELL_MARGIN / (CUTOFF + SKIN), True)],
    ids=["zero-search-envelope", "covering-search-envelope"],
)
def test_dynamic_cell_compression_inside_the_role_stencil_needs_the_search_envelope(
    deformation_margin: float, admitted: bool
) -> None:
    # A dynamic-cell role stencil carries absolute deformation headroom, but the
    # cell list sizes its neighbor-cell offsets from its own reference envelope
    # ``radius * (1 + deformation_margin)``. A 13% MTK compression (the AIMNet2
    # family trajectory's first step) stays inside the stencil: a covering
    # search envelope rebuilds the complete relation, a zero one reports a
    # scientific stencil failure and never a capacity failure.
    role = _role(CELL, edge_capacity=512, stencil_margin=DYNAMIC_CELL_MARGIN)
    plan = _plan(
        2,
        roles=(role,),
        search=AtomisticCellListImageSearch(
            maximum_candidate_slots=1 << 22, deformation_margin=deformation_margin
        ),
    )
    cell = CELL * 0.87
    layout, blocked, stored = _ingress(plan, _structure() * 0.87, STABLE_IDS, cell)
    topology = prepare_graph_partition_topology(
        plan, role, layout, blocked, jnp.asarray(cell), jnp.asarray(1)
    )
    evidence = topology.evidence
    assert float(np.min(np.asarray(evidence.stencil_margin))) > 0.0
    assert bool(evidence.capacity_successful)
    assert bool(evidence.scientific_successful) is admitted
    assert bool(evidence.stencil_complete) is admitted
    if admitted:
        observed = _topology_routes(plan, layout, topology)
        assert sorted(observed) == sorted(_oracle_routes(stored, cell, CUTOFF + SKIN))


def test_replicated_geometry_beyond_capacity_is_refused_with_exact_bytes() -> None:
    owners = 3
    rows = owners * LOCAL
    # float64 positions (24 B), bool activity (1 B), int64 IDs (8 B) and
    # int32 search ranks (4 B) per replicated row.
    geometry = rows * (3 * 8 + 1 + 8 + 4)
    admitted = _plan(owners, replicated_memory_bytes=geometry)
    _, topology, _ = _prepare(admitted, _structure(), STABLE_IDS)
    assert int(topology.evidence.replicated_bytes) == geometry
    assert int(topology.evidence.replicated_capacity_bytes) == geometry
    refused = _plan(owners, replicated_memory_bytes=geometry - 1)
    with pytest.raises(AtomisticGraphPartitionResourceError) as error:
        _prepare(refused, _structure(), STABLE_IDS)
    assert error.value.evidence.charges == (("search-geometry", geometry),)
    assert error.value.evidence.refused == (("search-geometry", geometry),)


def test_memory_evidence_matches_analytic_bytes_and_exchange_charges() -> None:
    owners = 3
    rows = owners * LOCAL
    plan = _plan(owners, replicated_memory_bytes=2000)
    layers = (
        {
            "vector": jax.ShapeDtypeStruct((LOCAL, 4, 3), jnp.float64),
            "species": jax.ShapeDtypeStruct((LOCAL,), jnp.int32),
        },
        jax.ShapeDtypeStruct((LOCAL, 16), jnp.float32),
    )
    evidence = graph_partition_memory_evidence(
        plan, position_dtype=np.float64, layer_payloads=layers
    )
    expected = (
        ("search-geometry", rows * (3 * 8 + 1 + 8 + 4)),
        ("positions", rows * 3 * 8),
        ("layer-0", rows * (4 * 3 * 8 + 4)),
        ("layer-1", rows * 16 * 4),
    )
    assert evidence.charges == expected
    assert evidence.replicated_rows == rows
    assert evidence.maximum_charge_bytes == rows * 100
    assert not evidence.admitted
    assert evidence.refused == (expected[2],)
    graph = plan.graph_partition
    assert graph is not None  # the graph-partition strategy binds its plan
    exchange = graph.exchange(plan.ownership)
    assert exchange.column_count == rows
    assert exchange.message_bytes(layers[0]) == expected[2][1]
    assert exchange.message_bytes(layers[1]) == expected[3][1]


def _owner_map(
    ownership: DistributedOwnershipPlan, body: Any, inputs: int, outputs: int
) -> Callable[..., Any]:
    """Owner-blocked map; a single output is a bare array, several a tuple."""
    owner = PartitionSpec(ownership.axis_name)
    out_specs = owner if outputs == 1 else (owner,) * outputs
    return ownership.map(body, (owner,) * inputs, out_specs)


@pytest.mark.parametrize(
    ("owners", "devices"),
    [(1, False), (2, False), (4, False), (2, True), (3, True)],
    ids=["lanes-1", "lanes-2", "lanes-4", "devices-2", "devices-3"],
)
def test_exchange_gather_and_transpose_are_exact_adjoints(
    owners: int, devices: bool
) -> None:
    ownership = _ownership(owners, devices=devices)
    exchange = ReplicatedColumnExchange(ownership, replicated_memory_bytes=1 << 20)
    columns = exchange.column_count
    rng = np.random.default_rng(owners)
    owned = jnp.asarray(rng.normal(size=(owners * LOCAL, 2)))
    cotangent = jnp.asarray(rng.normal(size=(owners * columns, 2)))
    species = jnp.asarray(rng.integers(0, 3, owners * LOCAL), dtype=jnp.int32)

    def body(values: jax.Array, contributions: jax.Array, kinds: jax.Array) -> Any:
        gathered = exchange.gather(values)
        returned = exchange.transpose(contributions, values)
        _, pullback = jax.vjp(exchange.gather, values)
        _, tangent = jax.jvp(exchange.gather, (values,), (2.0 * values,))
        mixed = exchange.transpose(
            {
                "x": contributions,
                "s": np.zeros((columns,), dtype=jax.dtypes.float0),
            },
            {"x": values, "s": kinds},
        )
        assert mixed["s"].dtype == jax.dtypes.float0
        np.testing.assert_equal(mixed["s"].shape, kinds.shape)
        return gathered, returned, pullback(contributions)[0], tangent, mixed["x"]

    gathered, returned, vjp, tangent, mixed = _owner_map(ownership, body, 3, 5)(
        owned, cotangent, species
    )
    np.testing.assert_array_equal(
        np.asarray(gathered), np.tile(np.asarray(owned), (owners, 1))
    )
    expected = np.sum(np.asarray(cotangent).reshape((owners, columns, 2)), axis=0)
    np.testing.assert_allclose(np.asarray(returned), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.asarray(vjp), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.asarray(mixed), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(tangent), 2.0 * np.tile(np.asarray(owned), (owners, 1))
    )
    # <gather(x), y> == <x, transpose(y)> summed over owners: no owner-count
    # scaling and every contribution returned exactly once.
    np.testing.assert_allclose(
        float(jnp.vdot(gathered, cotangent)),
        float(jnp.vdot(owned, returned)),
        rtol=1e-13,
    )


def test_exchange_refuses_a_layer_payload_beyond_replicated_memory() -> None:
    owners = 2
    ownership = _ownership(owners)
    rows = owners * LOCAL
    exchange = ReplicatedColumnExchange(ownership, replicated_memory_bytes=rows * 16)
    fitting = jnp.zeros((owners * LOCAL, 2))
    too_wide = jnp.zeros((owners * LOCAL, 3))
    _owner_map(ownership, exchange.gather, 1, 1)(fitting)
    with pytest.raises(AtomisticGraphPartitionResourceError) as error:
        _owner_map(ownership, exchange.gather, 1, 1)(too_wide)
    assert error.value.evidence.charges == (("gather", rows * 24),)
    with pytest.raises(ValueError, match="lead with"):
        _owner_map(ownership, exchange.gather, 1, 1)(jnp.zeros((owners * 3, 2)))


def test_plan_binds_its_ownership_partition_and_roles() -> None:
    plan = _plan(2)
    graph = plan.graph_partition
    assert graph is not None  # the graph-partition strategy binds its plan
    other = _plan(3)
    with pytest.raises(ValueError, match="another ownership"):
        graph.require_compatible(other.ownership, plan.partition, plan.roles)
    with pytest.raises(ValueError, match="another partition"):
        graph.require_compatible(plan.ownership, other.partition, plan.roles)
    wider = _role(CELL, edge_capacity=1024)
    with pytest.raises(ValueError, match="other roles"):
        graph.require_compatible(plan.ownership, plan.partition, (wider,))
    with pytest.raises(ValueError, match="another ownership"):
        graph.exchange(other.ownership)
    with pytest.raises(TypeError, match="dense reference"):
        AtomisticGraphPartitionPlan(
            plan.ownership,
            plan.partition,
            plan.roles,
            replicated_memory_bytes=1 << 20,
            receiver_edge_capacity=64,
            maximum_particles_per_cell=8,
            search=object(),  # ty: ignore[invalid-argument-type]
        )


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
MACE_SPECIES = (1, 8)
MACE_NUMBERS = np.asarray(MACE_SPECIES, dtype=np.int32)[
    np.random.default_rng(4).integers(0, 2, 6)
]


def _mace(layers: int, key: int = 0) -> MACEPotential:
    kinds = ("real-agnostic",) + ("real-agnostic-residual",) * (layers - 1)
    architecture = MACEArchitecture(
        species=MACE_SPECIES,
        cutoff=CUTOFF,
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


def _model_plan(
    model: Any,
    owners: int,
    strategy: AtomisticDomainStrategy,
    *,
    cell: np.ndarray,
    devices: bool = False,
    channel_capacity: int,
) -> AtomisticDomainPlan:
    requirement = model.requirements.neighborhood(
        AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
    )
    role = _role(
        cell,
        edge_capacity=512,
        requirement=requirement,
        channel_capacity=channel_capacity,
    )
    periodic = PeriodicCell(cell)
    ownership = _ownership(owners, devices=devices)
    partition = FractionalOwnerPartition(periodic, (owners, 1, 1))
    graph = (
        AtomisticGraphPartitionPlan(
            ownership,
            partition,
            (role,),
            replicated_memory_bytes=1 << 22,
            receiver_edge_capacity=256,
            maximum_particles_per_cell=16,
            search=_cell_list(),
        )
        if strategy is AtomisticDomainStrategy.GRAPH_PARTITION
        else None
    )
    return AtomisticDomainPlan(
        ownership,
        partition,
        (role,),
        migration_capacity=8,
        message_capacity_bytes=1 << 22,
        strategy=strategy,
        graph_partition=graph,
    )


def _mace_global_energy(
    model: MACEPotential,
    positions: jax.Array,
    species_ids: np.ndarray,
    strain: jax.Array,
    routes: list[tuple[int, int, tuple[int, ...]]],
    cell: np.ndarray = CELL,
) -> jax.Array:
    """Independent single-device MACE energy at strain ``F = I + strain``."""
    senders = np.asarray([source for source, _, _ in routes], dtype=np.int32)
    receivers = np.asarray([receiver for _, receiver, _ in routes], dtype=np.int32)
    shifts = np.asarray([shift for _, _, shift in routes], dtype=np.float64)
    count = positions.shape[0]
    deformation = jnp.eye(3) + strain
    deformed = positions @ deformation.T
    lattice = jnp.asarray(cell) @ deformation.T
    relation = StreamedRelationPlan(
        receiver_tile=4,
        edge_tile=32,
        channel_capacity=model.configuration.message_payload_width(),
    ).prepare(
        EdgeRelation(senders, receivers, source_size=count, target_size=count),
        owner_id="graph-partition-global-oracle",
    )
    vectors = deformed[receivers] - deformed[senders] + shifts @ lattice
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = model.species_indices(jnp.asarray(species_ids), mask)
    energies, successful = model.node_energies(species, mask, relation, vectors)
    return jnp.where(successful, jnp.sum(energies), jnp.nan)


def _mace_runtime(
    model: MACEPotential,
    owners: int,
    strategy: AtomisticDomainStrategy,
    positions: np.ndarray,
    *,
    devices: bool = False,
) -> tuple[Any, Any]:
    plan = _model_plan(
        model,
        owners,
        strategy,
        cell=CELL,
        devices=devices,
        channel_capacity=model.configuration.message_payload_width(),
    )
    runtime = prepare_atomistic_domain(plan, model)
    mask = jnp.ones(MACE_NUMBERS.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(MACE_NUMBERS), mask))
    state = runtime.ingress_global(
        positions, species, rng_key=jax.random.key(7), stable_ids=STABLE_IDS
    )
    return runtime, state


def _assert_mace_parity(
    model: MACEPotential, runtime: Any, state: Any, positions: np.ndarray
) -> Any:
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    routes = _oracle_routes(positions, CELL, CUTOFF)
    zero = jnp.zeros((3, 3))
    reference = jnp.asarray(positions)
    energy = _mace_global_energy(model, reference, MACE_NUMBERS, zero, routes)
    forces = -jax.grad(_mace_global_energy, argnums=1)(
        model, reference, MACE_NUMBERS, zero, routes
    )
    strain = jax.grad(_mace_global_energy, argnums=3)(
        model, reference, MACE_NUMBERS, zero, routes
    )
    np.testing.assert_allclose(float(evaluation.energy), float(energy), rtol=1e-11)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), forces, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.strain_derivative), strain, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.stress),
        0.5 * (strain + strain.T) / PeriodicCell(CELL).volume,
        atol=1e-10,
    )
    return evaluation


@pytest.mark.parametrize(
    ("owners", "devices", "layers"),
    [
        (1, False, 2),
        (2, False, 2),
        (3, False, 3),
        (2, True, 2),
        (3, True, 2),
    ],
    ids=["lanes-1", "lanes-2", "lanes-3-three-layer", "devices-2", "devices-3"],
)
def test_graph_partition_mace_matches_halo_and_single_device(
    owners: int, devices: bool, layers: int
) -> None:
    model = _mace(layers)
    positions = _structure()
    graph_runtime, graph_state = _mace_runtime(
        model,
        owners,
        AtomisticDomainStrategy.GRAPH_PARTITION,
        positions,
        devices=devices,
    )
    for topology in graph_state.topologies:
        assert topology.strategy is AtomisticDomainStrategy.GRAPH_PARTITION
        assert topology.halo is None
    # Stored coordinates are wrapped at ingress; the oracle uses the same
    # stored representation through the layout's logical egress.
    stored = np.asarray(graph_state.layout.collect(graph_state.positions))
    graph = _assert_mace_parity(model, graph_runtime, graph_state, stored)
    halo_runtime, halo_state = _mace_runtime(
        model, owners, AtomisticDomainStrategy.HALO, positions, devices=devices
    )
    halo = halo_runtime.evaluate(halo_state)
    assert bool(halo.successful)
    np.testing.assert_allclose(float(graph.energy), float(halo.energy), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(graph_state.layout.collect(graph.forces)),
        np.asarray(halo_state.layout.collect(halo.forces)),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(graph.strain_derivative),
        np.asarray(halo.strain_derivative),
        atol=1e-11,
    )


def test_graph_partition_lone_self_imaged_atom_has_zero_force_and_stress() -> None:
    model = _mace(2)
    plan = _model_plan(
        model,
        3,
        AtomisticDomainStrategy.GRAPH_PARTITION,
        cell=CELL,
        channel_capacity=model.configuration.message_payload_width(),
    )
    runtime = prepare_atomistic_domain(plan, model)
    positions = np.array([[0.4, 0.7, 1.1]])
    numbers = np.array([8], dtype=np.int32)
    species = np.asarray(
        model.species_indices(jnp.asarray(numbers), jnp.ones((1,), dtype=jnp.bool_))
    )
    state = runtime.ingress_global(positions, species, rng_key=jax.random.key(7))
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), 0.0, atol=1e-11
    )
    routes = _oracle_routes(positions, CELL, CUTOFF)
    strain = jax.grad(_mace_global_energy, argnums=3)(
        model, jnp.asarray(positions), numbers, jnp.zeros((3, 3)), routes
    )
    assert np.max(np.abs(strain)) > 1e-6
    np.testing.assert_allclose(
        np.asarray(evaluation.strain_derivative), strain, atol=1e-10
    )


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_graph_partition_force_loss_parameter_gradient_matches_single_device(
    devices: bool,
) -> None:
    model = _mace(2)
    owners = 2
    runtime, state = _mace_runtime(
        model,
        owners,
        AtomisticDomainStrategy.GRAPH_PARTITION,
        _structure(),
        devices=devices,
    )
    stored = np.asarray(state.layout.collect(state.positions))
    reference_forces = np.random.default_rng(11).normal(size=stored.shape)
    result = runtime.loss_gradient(
        state,
        -2.0,
        state.layout.distribute(reference_forces),
        energy_weight=0.5,
        force_weight=2.0,
    )
    assert bool(result.successful)
    routes = _oracle_routes(stored, CELL, CUTOFF)
    lanes = partition_parameters(model)

    def loss(parameters: Any) -> jax.Array:
        candidate = combine_parameters(parameters, lanes[1], lanes[2])
        reference = jnp.asarray(stored)
        zero = jnp.zeros((3, 3))
        energy = _mace_global_energy(candidate, reference, MACE_NUMBERS, zero, routes)
        forces = -jax.grad(_mace_global_energy, argnums=1)(
            candidate, reference, MACE_NUMBERS, zero, routes
        )
        return 0.5 * (energy + 2.0) ** 2 + 2.0 * jnp.sum((forces - reference_forces) ** 2)

    expected = jax.grad(loss)(lanes[0])
    np.testing.assert_allclose(float(result.loss), float(loss(lanes[0])), rtol=1e-11)
    for observed, oracle in zip(
        jax.tree.leaves(result.parameter_gradient),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(oracle), atol=1e-9)


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_graph_partition_rebuild_migrates_and_keeps_parity(devices: bool) -> None:
    model = _mace(2)
    owners = 2
    runtime, state = _mace_runtime(
        model,
        owners,
        AtomisticDomainStrategy.GRAPH_PARTITION,
        _structure(),
        devices=devices,
    )
    shift = np.broadcast_to(CELL[0] / owners, (6, 3))
    moved = _with_positions(state, state.positions + state.layout.distribute(shift))
    transition = runtime.rebuild(moved)
    assert bool(transition.committed)
    rebuilt = transition.state
    assert int(rebuilt.topologies[0].epoch) == int(state.topologies[0].epoch) + 1
    assert rebuilt.topologies[0].strategy is AtomisticDomainStrategy.GRAPH_PARTITION
    stored = np.asarray(rebuilt.layout.collect(rebuilt.positions))
    _assert_mace_parity(model, runtime, rebuilt, stored)


def _with_positions(state: Any, positions: jax.Array) -> Any:
    """Replace stored positions within the accepted topology epoch."""
    padding = ~state.layout.active[:, None]
    return eqx.tree_at(
        lambda item: item.positions, state, jnp.where(padding, 0.0, positions)
    )


AIMNET2_CELL = np.asarray([[3.6, 0.0, 0.0], [0.3, 3.7, 0.0], [0.2, 0.1, 3.8]])
AIMNET2_NUMBERS = np.asarray([8, 1, 1, 6])
AIMNET2_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.3, 0.9, 0.1], [0.4, -0.7, 1.2]]
) + np.asarray([1.8, 1.9, 1.9])


def _aimnet2() -> Any:
    declaration, tensors = source_inventory(3)
    core, state, _ = externalize_source_long_range(declaration, tensors)
    return aimnet2_potential_from_source(
        SCALE, core, state, neighbor_cutoff=RC_S
    ).potential


def _aimnet2_conditioning(model: Any, charge: float, atoms: int) -> AtomisticConditioning:
    return AtomisticConditioning(
        model.conditioning_spec,
        {"charge": np.asarray([charge], dtype=np.float64)},
        case_count=1,
        atom_capacity=atoms,
    )


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
@pytest.mark.parametrize("charge", [0.0, 1.0], ids=["neutral", "cation"])
def test_graph_partition_aimnet2_charge_normalization_matches_reference(
    devices: bool, charge: float
) -> None:
    """Per-pass charge normalization is a global condition: graph partition
    reduces it through the executor's system stages exactly once."""
    model = _aimnet2()
    owners = 2
    count = AIMNET2_NUMBERS.shape[0]
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(AIMNET2_NUMBERS), mask))
    evaluations = []
    for strategy in (
        AtomisticDomainStrategy.GRAPH_PARTITION,
        AtomisticDomainStrategy.HALO,
    ):
        plan = _model_plan(
            model,
            owners,
            strategy,
            cell=AIMNET2_CELL,
            devices=devices,
            channel_capacity=64,
        )
        runtime = prepare_atomistic_domain(plan, model)
        # Domain conditioning is bound to the owner-local atom capacity.
        state = runtime.ingress_global(
            AIMNET2_POSITIONS,
            species,
            rng_key=jax.random.key(3),
            conditioning=_aimnet2_conditioning(
                model, charge, plan.ownership.local_capacity
            ),
        )
        evaluation = runtime.evaluate(state)
        assert bool(evaluation.successful)
        evaluations.append((state, evaluation))
    (graph_state, graph), (halo_state, halo) = evaluations
    batch = AtomisticBatch.from_structure(
        AtomicStructure(
            AIMNET2_NUMBERS,
            AIMNET2_POSITIONS,
            np.ones(count),
            SCALE,
            cell=AIMNET2_CELL,
            periodic_axes=np.asarray([True, True, True]),
        )
    )
    reference = energy_and_forces(
        model,
        batch,
        AtomisticGraphExecutionPlan(
            256,
            backend="particle",
            image_capacity=ParticleImageCapacity(
                maximum_particles_per_cell=8,
                maximum_edges=8192,
                maximum_degree=256,
                maximum_images=343,
            ),
        ),
        # The single-device reference binds the batch's own atom capacity.
        conditioning=_aimnet2_conditioning(model, charge, batch.atom_capacity),
    )
    assert bool(reference.valid[0])
    np.testing.assert_allclose(
        float(graph.energy), float(np.asarray(reference.energy)[0]), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(graph_state.layout.collect(graph.forces)),
        np.asarray(reference.forces)[0, :count],
        atol=1e-9,
    )
    np.testing.assert_allclose(float(graph.energy), float(halo.energy), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(graph_state.layout.collect(graph.forces)),
        np.asarray(halo_state.layout.collect(halo.forces)),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(graph.strain_derivative),
        np.asarray(halo.strain_derivative),
        atol=1e-10,
    )
