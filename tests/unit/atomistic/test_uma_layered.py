"""Layered owner-region UMA (``AtomisticLayeredModel`` hooks) vs single-device UMA.

The lane harness here is a functional oracle of the hooks only: owners are
``vmap`` lanes over one named axis, every system stage combines owner partials
through the domain's `owner_system_sum`, and source columns are the tiled
owner-major ``all_gather`` of owned payloads (the replicated-column exchange).
Edges are an independent brute-force periodic route list partitioned by
receiver owner. The reference is the unchanged single-device
`atomistic_energy_derivatives` route of the SAME model.
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.sharding import PartitionSpec

import phydrax as phx
from phydrax._execution_runtime import ExecutionRuntime
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphInputs,
    AtomisticScaleContract,
)
from phydrax.atomistic._feature_execution import (
    owner_layer_gradients,
    owner_system_sum,
    OwnerConditioning,
    OwnerLayerTopology,
)
from phydrax.atomistic._graph import AtomisticGraphLayout
from phydrax.atomistic._graph_partition import ReplicatedColumnExchange
from phydrax.discretization import ParticleImageCapacity
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.nn.atomistic._uma import (
    UMABalancePartials,
    UMACompositionPartials,
    UMALayerContext,
    UMAPotential,
    UMAPropertyModel,
)
from phydrax.nn.atomistic._uma_conditioning import (
    apply_channel_balance,
    balance_channels,
    case_atom_counts,
    case_reduce,
    case_sums,
    channel_balance_correction,
    species_counts,
)
from phydrax.nn.atomistic._uma_layers import UMACaseLinear
from phydrax.nn.atomistic._uma_so3 import UMAEdgeRollGauge
from phydrax.nn.atomistic._uma_source import (
    parse_uma_source_config,
    uma_initial_source_tensors,
    uma_model_from_source,
)
from phydrax.sparse import EdgeRelation, RelationExecutionPlan, StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.uma_source import uma_config


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
AXIS = "owner"
CUTOFF = 2.5
# One cell vector shorter than the cutoff: every atom sees its own images and
# several images of each partner (repeated and self images).
CELL = np.asarray([[2.3, 0.0, 0.0], [0.4, 3.3, 0.0], [0.2, 0.3, 3.5]])
NUMBERS = np.asarray([8, 1, 1, 8, 1], dtype=np.int32)
FRACTIONAL = np.asarray(
    [
        [0.10, 0.20, 0.15],
        [0.45, 0.30, 0.20],
        [0.20, 0.55, 0.40],
        [0.70, 0.75, 0.60],
        [0.85, 0.40, 0.85],
    ]
)
POSITIONS = FRACTIONAL @ CELL
POSITIONS[3] += CELL[1]  # one stored coordinate outside the cell (unwrapped)
MASSES = np.where(NUMBERS == 8, 15.999, 1.008)
EXECUTION = AtomisticGraphExecutionPlan(
    64,
    backend="particle",
    image_capacity=ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=1024,
        maximum_degree=64,
        maximum_images=125,
    ),
    search_cell=CELL,
)
# Atom ownership per lane; the last assignment has an empty owner and splits
# both oxygens, so no owner holds the case composition.
ASSIGNMENTS = (
    ((0, 1, 2, 3, 4),),
    ((0, 1), (2, 3, 4)),
    ((0, 4), (), (1, 2, 3)),
)


def _model(
    seed: int = 3,
    roll_gauge: UMAEdgeRollGauge = "two-chart-nlerp-zero-roll",
    **config: Any,
) -> Any:
    model_config, tasks = uma_config(**config)
    declaration = parse_uma_source_config(model_config, tasks, roll_gauge=roll_gauge)
    tensors = uma_initial_source_tensors(declaration, key=jr.key(seed))
    model, residual = uma_model_from_source(
        SCALE,
        declaration,
        tensors,
        streaming=StreamedRelationPlan(
            receiver_tile=2, edge_tile=4, channel_capacity=512
        ),
        node_block=2,
    )
    assert residual < 1e-6
    return model


def _condition(
    model: UMAPotential, task: str = "omol", charge: int = -1, spin: int = 3
) -> Any:
    return model.schema.conditioning(
        [task], atom_capacity=NUMBERS.size, charges=[charge], spins=[spin]
    )


def _routes(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force directed routes ``(source, receiver, n)`` with ``0 < |d| < cutoff``."""
    reach = 3
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    senders, receivers, shifts = [], [], []
    for receiver in range(positions.shape[0]):
        for sender in range(positions.shape[0]):
            displacement = positions[receiver] - positions[sender] + grid @ CELL
            near = np.linalg.norm(displacement, axis=1) < CUTOFF
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


class _Domain(NamedTuple):
    rows: np.ndarray
    receivers: np.ndarray
    columns: np.ndarray
    shifts: np.ndarray
    valid: np.ndarray
    routes: np.ndarray  # brute-force route index of each owner edge (-1: padding)


def _domain(
    assignment: tuple[tuple[int, ...], ...], capacity: int | None = None
) -> _Domain:
    """Owner-blocked rows and receiver-owned edges into owner-major columns."""
    owners = len(assignment)
    if capacity is None:
        capacity = max(len(atoms) for atoms in assignment) + 1
    rows = np.full((owners, capacity), -1, dtype=np.int32)
    slot: dict[int, tuple[int, int]] = {}
    for owner, atoms in enumerate(assignment):
        for index, atom in enumerate(atoms):
            rows[owner, index] = atom
            slot[atom] = (owner, index)
    senders, receivers, shifts = _routes(POSITIONS)
    per_owner: list[list[tuple[int, int, np.ndarray, int]]] = [[] for _ in range(owners)]
    for route, (sender, receiver, shift) in enumerate(
        zip(senders, receivers, shifts, strict=True)
    ):
        owner, local = slot[int(receiver)]
        source_owner, source_slot = slot[int(sender)]
        per_owner[owner].append(
            (local, source_owner * capacity + source_slot, shift, route)
        )
    edges = max(len(items) for items in per_owner) + 3
    local_receivers = np.zeros((owners, edges), dtype=np.int32)
    columns = np.zeros((owners, edges), dtype=np.int32)
    edge_shifts = np.zeros((owners, edges, 3), dtype=np.int32)
    valid = np.zeros((owners, edges), dtype=np.bool_)
    routes = np.full((owners, edges), -1, dtype=np.int32)
    for owner, items in enumerate(per_owner):
        for index, (local, column, shift, route) in enumerate(items):
            local_receivers[owner, index] = local
            columns[owner, index] = column
            edge_shifts[owner, index] = shift
            valid[owner, index] = True
            routes[owner, index] = route
    return _Domain(rows, local_receivers, columns, edge_shifts, valid, routes)


class _OwnerResult(NamedTuple):
    energy: jax.Array
    system_energy: jax.Array
    successful: jax.Array
    context: UMALayerContext
    features: jax.Array


def _lanes(
    model: UMAPotential,
    domain: _Domain,
    positions: jax.Array,
    strain: jax.Array,
    condition: Any,
    key: Any = None,
    identity: tuple[np.ndarray, ...] = (),
) -> _OwnerResult:
    """Every owner lane's layered evaluation at strain ``F = I + strain``.

    ``identity`` are the owner-blocked ``(owners, edges)`` canonical route
    identity fields of every owner edge; ``key`` is the evaluation's roll key.
    """
    owners, capacity = domain.rows.shape
    present = domain.rows >= 0
    rows = np.where(present, domain.rows, 0)
    deformation = jnp.eye(3, dtype=positions.dtype) + strain
    cell = jnp.asarray(CELL) @ deformation.T
    blocked = jnp.where(present[..., None], positions[rows], 0.0) @ deformation.T
    species = jnp.asarray(np.where(present, NUMBERS[rows], 0), dtype=jnp.int32)
    streaming = model.network.streaming
    assert streaming is not None
    stages = model.system_stages

    def staged(
        after: int, features: Any, context: UMALayerContext, owned: Any, mask: Any
    ) -> tuple[Any, UMALayerContext]:
        for index, stage in enumerate(stages):
            if stage.after_interaction == after:
                partials = model.system_partials(index, features, owned, mask, context)
                total = owner_system_sum(partials, AXIS, "deterministic")
                features, context = model.apply_system_context(
                    index, features, total, owned, mask, context
                )
        return features, context

    def owner(
        x: Any,
        owned: Any,
        mask: Any,
        receivers: Any,
        columns: Any,
        shifts: Any,
        valid: Any,
        fields: tuple[Any, ...],
    ) -> _OwnerResult:
        context = model.layer_context(
            condition, jnp.zeros(mask.shape, dtype=jnp.int32), owned, mask, key=key
        )
        _, context = staged(-1, None, context, owned, mask)
        features = model.init_features(owned, mask, context)
        gathered = jax.lax.all_gather(x, AXIS, tiled=True)
        raw = x[receivers] - gathered[columns] + shifts.astype(x.dtype) @ cell
        sentinel = jnp.zeros((3,), dtype=x.dtype).at[0].set(0.5 * CUTOFF)
        vectors = jnp.where(valid[:, None], raw, sentinel)
        active = valid & (jnp.sum(vectors * vectors, axis=-1) < CUTOFF * CUTOFF)
        relation = streaming.prepare(
            EdgeRelation(
                columns,
                receivers,
                source_size=owners * capacity,
                target_size=capacity,
                valid=valid,
            ),
            owner_id="uma-layered-oracle",
            receiver_valid=mask,
        )
        successful = model.admitted(owned, mask, context)
        energies = []
        for index in range(model.interaction_count):
            sources = model.prepare_sources(index, features, owned, mask, context)
            exchanged = jax.tree.map(
                lambda leaf: jax.lax.all_gather(leaf, AXIS, tiled=True), sources
            )
            update = model.update_receivers(
                index,
                relation,
                vectors,
                exchanged,
                features,
                owned,
                mask,
                context,
                edge_active=active,
                route_identity=fields or None,
            )
            features = update.features
            energies.append(update.energy)
            successful = successful & update.evidence.successful
            features, context = staged(index, features, context, owned, mask)
        assert model.global_payload(features, owned, mask, context) is None
        atom = model.atom_energies(owned, mask, jnp.stack(energies), features, context)
        return _OwnerResult(
            jnp.sum(jnp.where(mask, atom, 0.0)),
            model.system_energy(context),
            successful,
            context,
            features,
        )

    return jax.vmap(owner, axis_name=AXIS)(
        blocked,
        species,
        jnp.asarray(present),
        jnp.asarray(domain.receivers),
        jnp.asarray(domain.columns),
        jnp.asarray(domain.shifts),
        jnp.asarray(domain.valid),
        tuple(jnp.asarray(field) for field in identity),
    )


def _layered_energy(
    model: UMAPotential,
    domain: _Domain,
    positions: jax.Array,
    strain: jax.Array,
    condition: Any,
    key: Any = None,
    identity: tuple[np.ndarray, ...] = (),
) -> jax.Array:
    """Owner-summed atom energies plus the replicated system energy, counted once."""
    lanes = _lanes(model, domain, positions, strain, condition, key, identity)
    total = jnp.sum(lanes.energy) + lanes.system_energy[0]
    return jnp.where(jnp.all(lanes.successful), total, jnp.nan)


def _batch() -> AtomisticBatch:
    return AtomisticBatch(
        NUMBERS[None],
        POSITIONS[None],
        MASSES[None],
        SCALE,
        cells=CELL[None],
        periodic_axes=np.ones((1, 3), dtype=np.bool_),
    )


def _reference(model: UMAPotential, condition: Any, **kwargs: Any) -> Any:
    batch = _batch()
    payload = AtomisticGraphLayout.for_batch(batch, EXECUTION, cutoff=CUTOFF).prepare(
        batch
    )
    return atomistic_energy_derivatives(
        model,
        payload,
        batch.positions,
        conditioning=condition,
        **kwargs,
    )


def _volume() -> float:
    return float(abs(np.linalg.det(CELL)))


# --- schedule ------------------------------------------------------------


def test_layered_schedule_declares_composition_routing_and_balance_stages() -> None:
    model = _model()
    configuration = model.configuration
    assert model.interaction_count == configuration.layer_count + 1
    stages = model.system_stages
    assert [stage.after_interaction for stage in stages] == [-1] + [
        block + 1 for block in range(configuration.layer_count)
    ]
    assert stages[0].name == "uma-composition-routing"
    plain = _model(experts=0, charge_balanced_channels=[], spin_balanced_channels=[])
    assert plain.system_stages == ()
    # A property model has no potential: it exposes no layered energy hooks.
    for name in ("atom_energies", "system_energy", "update_receivers"):
        assert not hasattr(UMAPropertyModel, name)


def test_route_oracle_has_repeated_and_self_images_within_the_selection() -> None:
    senders, receivers, shifts = _routes(POSITIONS)
    assert np.any((senders == receivers) & np.any(shifts != 0, axis=1))
    pairs = list(zip(senders.tolist(), receivers.tolist(), strict=True))
    assert len(set(pairs)) < len(pairs)
    # The declared nearest-neighbour selection keeps every in-cutoff route of
    # this fixture, so the brute-force route set is the selected graph.
    degree = np.bincount(receivers, minlength=NUMBERS.size)
    assert int(degree.max()) < _model().configuration.max_neighbors


# --- scientific parity ---------------------------------------------------


@pytest.mark.parametrize("assignment", ASSIGNMENTS, ids=("one", "two", "empty-owner"))
def test_lane_owners_match_single_device_energy_forces_and_stress(
    assignment: tuple[tuple[int, ...], ...],
) -> None:
    model = _model()
    condition = _condition(model)
    reference = _reference(model, condition, compute_stress=True)
    assert bool(reference.successful[0])
    domain = _domain(assignment)
    positions = jnp.asarray(POSITIONS)
    zero = jnp.zeros((3, 3))
    energy = _layered_energy(model, domain, positions, zero, condition)
    gradient, strain = jax.grad(_layered_energy, argnums=(2, 3))(
        model, domain, positions, zero, condition
    )
    np.testing.assert_allclose(float(energy), float(reference.energy[0]), rtol=1e-11)
    np.testing.assert_allclose(-gradient, reference.forces[0], atol=1e-10)
    np.testing.assert_allclose(
        0.5 * (strain + strain.T) / _volume(), reference.stress[0], atol=1e-10
    )


@pytest.mark.parametrize("assignment", ASSIGNMENTS, ids=("one", "two", "empty-owner"))
def test_lane_owner_parameter_gradient_matches_single_device(
    assignment: tuple[tuple[int, ...], ...],
) -> None:
    """MoLE routing, composition and condition embeddings receive each use once."""
    model = _model()
    condition = _condition(model)
    lane, state, fixed = partition_parameters(model)
    domain = _domain(assignment)
    positions = jnp.asarray(POSITIONS)
    zero = jnp.zeros((3, 3))

    def layered(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        return _layered_energy(rebuilt, domain, positions, zero, condition)

    def single(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        return _reference(rebuilt, condition, compute_forces=False).energy[0]

    observed = jax.grad(layered)(lane)
    expected = jax.grad(single)(lane)
    for left, right in zip(
        jax.tree_util.tree_leaves(observed),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-11)
    router = observed.network.router
    assert router is not None and router.composition is not None
    # Context cotangents reach the router and composition table: nonzero and
    # equal to the single-device response (never scaled by the owner count).
    np.testing.assert_allclose(
        router.composition, expected.network.router.composition, rtol=1e-9, atol=1e-11
    )
    assert float(jnp.max(jnp.abs(router.composition))) > 0.0
    for left, right in zip(router.linears, expected.network.router.linears, strict=True):
        np.testing.assert_allclose(left.weight, right.weight, rtol=1e-9, atol=1e-11)
        assert float(jnp.max(jnp.abs(left.weight))) > 0.0


def test_force_loss_parameter_gradient_matches_single_device() -> None:
    """Mixed parameter-coordinate derivatives through every owner collective."""
    model = _model()
    condition = _condition(model)
    lane, state, fixed = partition_parameters(model)
    domain = _domain(ASSIGNMENTS[2])
    positions = jnp.asarray(POSITIONS)
    zero = jnp.zeros((3, 3))
    target = jnp.asarray(np.random.default_rng(5).normal(size=POSITIONS.shape))

    def layered(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        forces = -jax.grad(_layered_energy, argnums=2)(
            rebuilt, domain, positions, zero, condition
        )
        return jnp.sum((forces - target) ** 2)

    def single(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        return jnp.sum((_reference(rebuilt, condition).forces[0] - target) ** 2)

    for left, right in zip(
        jax.tree_util.tree_leaves(jax.grad(layered)(lane)),
        jax.tree_util.tree_leaves(jax.grad(single)(lane)),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-10)


# --- system stages -------------------------------------------------------


def test_routing_reads_the_complete_case_composition_on_every_owner() -> None:
    model = _model()
    condition = _condition(model)
    lanes = _lanes(
        model,
        _domain(ASSIGNMENTS[2]),
        jnp.asarray(POSITIONS),
        jnp.zeros((3, 3)),
        condition,
    )
    network = model.network
    mask = jnp.ones((NUMBERS.size,), dtype=jnp.bool_)
    charge, spin, task = (
        jnp.asarray(condition.value(name))
        for name in ("total_charge", "uma_spin_index", "uma_task")
    )
    system = network.system(charge, spin, task)
    counts = species_counts(
        jnp.asarray(NUMBERS), mask, 1, NUMBERS.size, model.configuration.max_elements
    )
    expected = network.realized_blocks(
        network.mixture_coefficients(
            system, counts, case_atom_counts(mask, 1, NUMBERS.size)
        )
    )[0].edgewise.first.m0.weights
    blocks = lanes.context.blocks
    assert blocks is not None
    realized = blocks[0].edgewise.first.m0
    assert isinstance(realized, UMACaseLinear)
    observed = realized.weights
    for owner in range(3):
        np.testing.assert_allclose(observed[owner], expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(lanes.context.atom_counts, np.full((3, 1), 5))
    # Owner-local routing would differ: the composition is a global sum.
    local = species_counts(
        jnp.asarray(NUMBERS[[0, 4]]), mask[:2], 1, 2, model.configuration.max_elements
    )
    local_weights = network.realized_blocks(
        network.mixture_coefficients(system, local, jnp.asarray([2]))
    )[0].edgewise.first.m0.weights
    assert not np.allclose(local_weights, expected)


def test_channel_balance_reaches_case_targets_across_owners() -> None:
    model = _model()
    configuration = model.configuration
    charge, spin = -1, 3
    domain = _domain(ASSIGNMENTS[1])
    lanes = _lanes(
        model,
        domain,
        jnp.asarray(POSITIONS),
        jnp.zeros((3, 3)),
        _condition(model, charge=charge, spin=spin),
    )
    present = jnp.asarray(domain.rows >= 0)
    features = jnp.where(present[..., None, None], lanes.features, 0.0)
    start, end = configuration.charge_channels
    np.testing.assert_allclose(jnp.sum(features[:, :, 0, start:end]), charge, atol=1e-12)
    start, end = configuration.spin_channels
    np.testing.assert_allclose(
        jnp.sum(features[:, :, 0, start:end]), spin - 1, atol=1e-12
    )


def test_owner_partials_are_per_case_integer_and_channel_sums() -> None:
    model = _model()
    condition = _condition(model)
    mask = jnp.asarray([True, False, True])
    species = jnp.asarray([8, 1, 1], dtype=jnp.int32)
    context = model.layer_context(condition, jnp.zeros((3,), jnp.int32), species, mask)
    partials = model.system_partials(0, None, species, mask, context)
    assert isinstance(partials, UMACompositionPartials)
    assert partials.atom_counts.dtype == jnp.int32
    np.testing.assert_array_equal(partials.atom_counts, [2])
    assert partials.species_counts is not None
    assert (
        int(partials.species_counts[0, 8]) == 1
        and int(partials.species_counts[0, 1]) == 1
    )
    _, routed = model.apply_system_context(0, None, partials, species, mask, context)
    features = model.init_features(species, mask, routed)
    balance = model.system_partials(1, features, species, mask, routed)
    assert isinstance(balance, UMABalancePartials)
    start, end = model.configuration.charge_channels
    np.testing.assert_allclose(
        balance.charge,
        jnp.sum(jnp.where(mask[:, None], features[:, 0, start:end], 0.0), axis=0)[None],
    )


def test_system_energy_counts_the_task_mean_once_per_case() -> None:
    model = _model()
    shifted = eqx.tree_at(
        lambda m: m.network.references.energy_mean,
        model,
        jnp.asarray([0.37, -1.25]),
    )
    condition = _condition(shifted)
    reference = _reference(shifted, condition, compute_forces=False)
    unshifted = _reference(model, condition, compute_forces=False)
    assert abs(float(reference.energy[0] - unshifted.energy[0])) > 0.1
    for assignment in ASSIGNMENTS:
        energy = _layered_energy(
            shifted,
            _domain(assignment),
            jnp.asarray(POSITIONS),
            jnp.zeros((3, 3)),
            condition,
        )
        np.testing.assert_allclose(float(energy), float(reference.energy[0]), rtol=1e-11)


# --- product executor ----------------------------------------------------


LOCAL_CAPACITY = 8


def _lane_ownership(owners: int) -> DistributedOwnershipPlan:
    if jax.process_count() != 1:
        pytest.skip("Reference owner lanes run in one JAX process.")
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
    )
    runtime = ExecutionRuntime.current()
    group = runtime.child_groups(runtime.root_group.spec.device_count)[0]
    return DistributedOwnershipPlan(address, group, LOCAL_CAPACITY, owner_lanes=owners)


@pytest.mark.parametrize("assignment", ASSIGNMENTS[1:], ids=("two", "empty-owner"))
def test_owner_layer_gradients_match_single_device_with_replicated_columns(
    assignment: tuple[tuple[int, ...], ...],
) -> None:
    """The domain executor's explicit reverse sweep on UMA's layered hooks."""
    model = _model()
    condition = _condition(model)
    reference = _reference(model, condition, compute_stress=True)
    lane, state, fixed = partition_parameters(model)

    def single(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        return _reference(rebuilt, condition, compute_forces=False).energy[0]

    expected_parameters = jax.grad(single)(lane)
    owners = len(assignment)
    ownership = _lane_ownership(owners)
    exchange = ReplicatedColumnExchange(ownership, replicated_memory_bytes=1 << 26)
    domain = _domain(assignment, LOCAL_CAPACITY)
    present = domain.rows >= 0
    rows = np.where(present, domain.rows, 0)
    positions = np.where(present[..., None], POSITIONS[rows], 0.0)
    raw_species = np.where(present, NUMBERS[rows], 0).astype(np.int32)
    arrays, structure = eqx.partition(model, eqx.is_array)
    cell = jnp.asarray(CELL)
    edges = domain.receivers.shape[1]

    def body(
        x: Any,
        raw: Any,
        mask: Any,
        receivers: Any,
        columns: Any,
        shifts: Any,
        valid: Any,
        parameters: Any,
    ) -> Any:
        local = eqx.combine(parameters, structure)
        relation = EdgeRelation(
            columns,
            receivers,
            source_size=exchange.column_count,
            target_size=LOCAL_CAPACITY,
            valid=valid,
        )
        reduction = RelationExecutionPlan()
        topology = OwnerLayerTopology(
            relation=local.network.streaming.prepare(
                relation, owner_id="uma-layered-executor", receiver_valid=mask
            ),
            receiver_reduction=reduction.prepare(relation),
            source_reduction=reduction.prepare(relation.transpose()),
            receivers=receivers,
            columns=columns,
            shifts=shifts,
            valid=valid,
        )
        result = owner_layer_gradients(
            local,
            exchange,
            topology,
            x,
            local.species_indices(raw, mask),
            mask,
            cell,
            CUTOFF,
            context=OwnerConditioning(condition, jnp.zeros(mask.shape, jnp.int32)),
            reduction_mode="deterministic",
            axis_name=ownership.axis_name,
            accumulation="deterministic",
            parameters=True,
        )
        energy = (
            jnp.sum(jnp.where(mask, result.atom_energies, 0.0)) + result.global_energy
        )
        return (
            result.gradient,
            energy[None],
            result.strain_gradient[None],
            result.relation_successful[None],
            jax.tree.map(lambda leaf: leaf[None], result.parameter_gradient),
        )

    owner = PartitionSpec(ownership.axis_name)
    gradient, energies, strains, successful, parameters = ownership.map(
        body, (owner,) * 7 + (PartitionSpec(),), (owner,) * 5
    )(
        jnp.asarray(positions.reshape((-1, 3))),
        jnp.asarray(raw_species.reshape(-1)),
        jnp.asarray(present.reshape(-1)),
        jnp.asarray(domain.receivers.reshape(-1)),
        jnp.asarray(domain.columns.reshape(-1)),
        jnp.asarray(domain.shifts.reshape((owners * edges, 3))),
        jnp.asarray(domain.valid.reshape(-1)),
        arrays,
    )
    assert bool(jnp.all(successful))
    np.testing.assert_allclose(
        float(jnp.sum(energies)), float(reference.energy[0]), rtol=1e-11
    )
    forces = np.zeros_like(POSITIONS)
    blocked = np.asarray(gradient).reshape((owners, LOCAL_CAPACITY, 3))
    forces[domain.rows[present]] = -blocked[present]
    np.testing.assert_allclose(forces, reference.forces[0], atol=1e-10)
    strain = jnp.sum(strains, axis=0)
    np.testing.assert_allclose(
        0.5 * (strain + strain.T) / _volume(), reference.stress[0], atol=1e-10
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(
            jax.tree.map(lambda leaf: jnp.sum(leaf, 0), parameters)
        ),
        jax.tree_util.tree_leaves(expected_parameters),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-11)


# --- admission and refusals ----------------------------------------------


def test_admission_reports_out_of_table_species_and_tasks_without_energy() -> None:
    model = _model()
    condition = _condition(model)
    mask = jnp.asarray([True, True, False])
    cases = jnp.zeros((3,), dtype=jnp.int32)
    species = jnp.asarray([8, 1, 99], dtype=jnp.int32)
    context = model.layer_context(condition, cases, species, mask)
    assert bool(model.admitted(species, mask, context))
    outside = species.at[1].set(99)
    assert not bool(model.admitted(outside, mask, context))
    unavailable = eqx.tree_at(
        lambda m: m.network.references.energy_available,
        model,
        jnp.asarray([True, False]),
    )
    omol = unavailable.layer_context(condition, cases, species, mask)
    assert not bool(unavailable.admitted(species, mask, omol))


def test_keyed_roll_gauge_needs_an_explicit_key() -> None:
    keyed = _model(act_type="s2", roll_gauge="keyed-route-uniform-roll")
    condition = keyed.schema.conditioning(["omol"], atom_capacity=2)
    cases = jnp.zeros((2,), dtype=jnp.int32)
    species = jnp.asarray([8, 1], dtype=jnp.int32)
    mask = jnp.ones((2,), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="explicit PRNG key"):
        keyed.layer_context(condition, cases, species, mask)
    context = keyed.layer_context(condition, cases, species, mask, key=jr.key(4))
    assert context.key is not None
    # Every other gauge ignores the key; its context carries none.
    plain = _model(act_type="s2")
    assert plain.layer_context(condition, cases, species, mask, key=jr.key(4)).key is None


def _route_identity(
    model: UMAPotential, condition: Any, key: Any
) -> tuple[np.ndarray, ...]:
    """Each brute-force route's canonical identity fields, read from the single graph.

    Routes match by sender, receiver and displacement; every selected
    single-graph route matches exactly one brute-force route.
    """
    batch = _batch()
    payload = AtomisticGraphLayout.for_batch(batch, EXECUTION, cutoff=CUTOFF).prepare(
        batch
    )
    graph = payload.graph(cutoff=CUTOFF, selection=model.configuration.selection_policy)
    inputs = AtomisticGraphInputs(
        graph,
        SCALE,
        species_kind=model.capabilities.species_kind,
        conditioning=condition,
        key=key,
    )
    fields = [np.asarray(field) for field in inputs.route_identity_fields]
    bound = graph.topology
    senders = np.asarray(bound.senders)
    receivers = np.asarray(bound.receivers)
    displacement = np.asarray(graph.graph.edges["displacement"])
    active = np.asarray(graph.graph.edge_mask, dtype=np.bool_)
    routes = _routes(POSITIONS)
    index = np.empty((routes[0].size,), dtype=np.int64)
    for route, (sender, receiver, shift) in enumerate(zip(*routes, strict=True)):
        vector = POSITIONS[receiver] - POSITIONS[sender] + shift @ CELL
        hits = np.flatnonzero(
            active
            & (senders == sender)
            & (receivers == receiver)
            & (np.linalg.norm(displacement - vector, axis=1) < 1e-9)
        )
        assert hits.size == 1
        index[route] = hits[0]
    assert int(np.sum(active)) == routes[0].size
    return tuple(field[index] for field in fields)


def _owner_identity(
    domain: _Domain, routes: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, ...]:
    """Owner-blocked identity fields of every owner edge (padding: any route)."""
    index = np.maximum(domain.routes, 0)
    return tuple(field[index] for field in routes)


@pytest.mark.parametrize("assignment", ASSIGNMENTS, ids=("one", "two", "empty-owner"))
def test_keyed_roll_lane_owners_match_single_device(
    assignment: tuple[tuple[int, ...], ...],
) -> None:
    """Same key: every route draws the same roll on every owner layout."""
    model = _model(act_type="s2", roll_gauge="keyed-route-uniform-roll")
    condition = _condition(model)
    key = jr.key(17)
    reference = _reference(model, condition, compute_stress=True, key=key)
    assert bool(reference.successful[0])
    domain = _domain(assignment)
    identity = _owner_identity(domain, _route_identity(model, condition, key))
    positions = jnp.asarray(POSITIONS)
    zero = jnp.zeros((3, 3))
    energy = _layered_energy(model, domain, positions, zero, condition, key, identity)
    gradient, strain = jax.grad(_layered_energy, argnums=(2, 3))(
        model, domain, positions, zero, condition, key, identity
    )
    np.testing.assert_allclose(float(energy), float(reference.energy[0]), rtol=1e-11)
    np.testing.assert_allclose(-gradient, reference.forces[0], atol=1e-10)
    np.testing.assert_allclose(
        0.5 * (strain + strain.T) / _volume(), reference.stress[0], atol=1e-10
    )
    # The rolls are the key's: another key is another realization.
    other = _layered_energy(
        model, domain, positions, zero, condition, jr.key(18), identity
    )
    assert float(other) != float(energy)


def test_conditioning_is_required_by_the_layer_context() -> None:
    model = _model()
    with pytest.raises(ValueError, match="conditioning"):
        model.layer_context(
            None,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.asarray([8, 1], dtype=jnp.int32),
            jnp.ones((2,), dtype=jnp.bool_),
        )


# --- shared single-graph decomposition -----------------------------------


def test_indexed_case_sums_and_balance_decomposition_match_case_major_rows() -> None:
    rng = np.random.default_rng(11)
    values = jnp.asarray(rng.normal(size=(6, 4, 3)))
    mask = jnp.asarray([True, True, False, True, True, True])
    target = jnp.asarray([2, -1])
    cases = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
    np.testing.assert_allclose(
        case_sums(values[:, 0, :], mask, cases, 2),
        case_reduce(values[:, 0, :], mask, 2, 3),
        atol=1e-14,
    )
    sums = case_sums(values[:, 0, 0:2], mask, cases, 2)
    counts = case_sums(mask.astype(jnp.int32), mask, cases, 2)
    correction = channel_balance_correction(sums, counts, target, 0.0)
    split = apply_channel_balance(values, correction[cases], mask, (0, 2))
    np.testing.assert_allclose(
        split, balance_channels(values, target, mask, 2, 3, (0, 2), 0.0), atol=1e-14
    )
