"""Owner-region layered execution with system reductions and global epilogues.

Every parity case maps owner regions as named lanes (the functional oracle of
device-collective execution), builds each owner's halo columns and owner
relation from a brute-force route set, and runs the explicit reverse sweep of
``phydrax.atomistic._feature_execution``. The independent references are the
SAME models evaluated single-device: AIMNet2 through its bound-graph
``evaluate_graph`` (per-case charge equilibration over all atoms at once),
MACE through ``node_energies`` on the global route relation, MACELES through
its single-device latent charges. Owner assignments include an empty owner
and split bonded partners, so charges and features cross owners in every pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    AtomicStructure,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
)
from phydrax.atomistic._conditioning import AtomisticConditioning
from phydrax.atomistic._feature_execution import (
    AtomisticExchangeCapacityError,
    HaloColumnExchange,
    owner_layer_gradients,
    owner_system_sum,
    OwnerConditioning,
    OwnerLayerTopology,
    OwnerReductionMode,
)
from phydrax.atomistic._graph import AtomisticGraphLayout
from phydrax.atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldResources,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import (
    DistributedHaloPlan,
    DistributedOwnershipPlan,
)
from phydrax.nn.atomistic._aimnet2_source import (
    aimnet2_potential_from_source,
    externalize_source_long_range,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_embedding import MACEEmbeddingFeature
from phydrax.nn.atomistic._mace_les import MACELESPotential
from phydrax.sparse import EdgeRelation, RelationExecutionPlan
from phydrax.sparse._streamed import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.aimnet2_source import RC_S, source_inventory


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
STREAMING = StreamedRelationPlan(receiver_tile=4, edge_tile=32, channel_capacity=4096)

# --- AIMNet2 (finite, charged) -------------------------------------------
AIMNET2_NUMBERS = np.asarray([8, 1, 1, 6, 1, 1], dtype=np.int32)
AIMNET2_POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.95, 0.1, 0.0],
        [-0.3, 0.9, 0.1],
        [0.4, -0.7, 1.2],
        [1.5, -0.4, 1.7],
        [-0.6, -1.3, 1.6],
    ]
)
AIMNET2_EXECUTION = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=8)
# One owner, split bonded partners, and an empty owner between two others.
AIMNET2_ASSIGNMENTS = (
    ((0, 1, 2, 3, 4, 5),),
    ((0, 3, 5), (1, 2, 4)),
    ((0, 4), (), (1, 2), (3, 5)),
)
AIMNET2_IDS = ("one-owner", "two-owners", "empty-owner")

# --- MACE (periodic triclinic: repeated and self images) -----------------
MACE_CUTOFF = 2.4
MACE_CELL = np.asarray([[2.2, 0.0, 0.0], [0.6, 2.5, 0.0], [0.3, 0.4, 2.8]])
MACE_NUMBERS = np.asarray([8, 1, 1, 8, 1], dtype=np.int32)
MACE_POSITIONS = (
    np.asarray(
        [
            [0.12, 0.21, 0.33],
            [0.47, 0.31, 0.22],
            [0.53, 0.33, 0.25],
            [0.81, 0.66, 0.58],
            [0.29, 0.84, 0.71],
        ]
    )
    @ MACE_CELL
)
MACE_ASSIGNMENTS = (((0, 2), (1, 3, 4)), ((3,), (), (0, 1, 2, 4)))
MACE_IDS = ("two-owners", "empty-owner")
CASE_FEATURES = (
    MACEEmbeddingFeature(
        "total_charge", "categorical", level="case", width=3, classes=5, offset=2
    ),
    MACEEmbeddingFeature(
        "total_spin", "categorical", level="case", width=2, classes=3, offset=-1
    ),
    MACEEmbeddingFeature("temperature", "continuous", level="case", width=2, bias=False),
)


# --- shared owner-region machinery -----------------------------------------


def _lane_ownership(owners: int, capacity: int) -> DistributedOwnershipPlan:
    if jax.process_count() != 1:
        pytest.skip("Reference owner lanes run in one JAX process.")
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
    )
    runtime = ExecutionRuntime.current()
    group = runtime.child_groups(runtime.root_group.spec.device_count)[0]
    return DistributedOwnershipPlan(address, group, capacity, owner_lanes=owners)


type _Routes = tuple[np.ndarray, np.ndarray, np.ndarray]


def _finite_routes(positions: np.ndarray, cutoff: float) -> _Routes:
    """Every ordered pair ``(source, receiver)`` with ``|x_r - x_s| < cutoff``."""
    senders, receivers = [], []
    for receiver in range(positions.shape[0]):
        for sender in range(positions.shape[0]):
            distance = np.linalg.norm(positions[receiver] - positions[sender])
            if sender != receiver and distance < cutoff:
                senders.append(sender)
                receivers.append(receiver)
    count = len(senders)
    return (
        np.asarray(senders, dtype=np.int32),
        np.asarray(receivers, dtype=np.int32),
        np.zeros((count, 3), dtype=np.int32),
    )


def _image_routes(positions: np.ndarray, cell: np.ndarray, cutoff: float) -> _Routes:
    """Brute-force directed image routes ``(source, receiver, n)`` within cutoff."""
    reach = 3
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    senders, receivers, shifts = [], [], []
    for receiver in range(positions.shape[0]):
        for sender in range(positions.shape[0]):
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


class _Domain(NamedTuple):
    ownership: DistributedOwnershipPlan
    halo: DistributedHaloPlan
    rows: np.ndarray
    receivers: np.ndarray
    columns: np.ndarray
    shifts: np.ndarray
    valid: np.ndarray


def _domain(
    assignment: Sequence[Sequence[int]], routes: _Routes, capacity: int
) -> _Domain:
    owners = len(assignment)
    ownership = _lane_ownership(owners, capacity)
    rows = np.full((owners, capacity), -1, dtype=np.int64)
    owner_of: dict[int, int] = {}
    slot_of: dict[int, int] = {}
    for owner, atoms in enumerate(assignment):
        for slot, atom in enumerate(atoms):
            rows[owner, slot] = atom
            owner_of[atom] = owner
            slot_of[atom] = slot
    senders, receivers, shifts = routes
    edges = max(1, max(int(np.sum(np.isin(receivers, atoms))) for atoms in assignment))
    local_receivers = np.zeros((owners, edges), dtype=np.int32)
    route_owners = np.zeros((owners, edges), dtype=np.int32)
    route_slots = np.zeros((owners, edges), dtype=np.int32)
    local_shifts = np.zeros((owners, edges, 3), dtype=np.int32)
    valid = np.zeros((owners, edges), dtype=np.bool_)
    fill = [0] * owners
    for sender, receiver, shift in zip(senders, receivers, shifts, strict=True):
        owner = owner_of[int(receiver)]
        edge = fill[owner]
        fill[owner] += 1
        local_receivers[owner, edge] = slot_of[int(receiver)]
        route_owners[owner, edge] = owner_of[int(sender)]
        route_slots[owner, edge] = slot_of[int(sender)]
        local_shifts[owner, edge] = shift
        valid[owner, edge] = True
    halo = DistributedHaloPlan(
        ownership,
        route_owners.reshape(-1),
        route_slots.reshape(-1),
        valid.reshape(-1),
        halo_capacity=capacity,
    )
    assert bool(halo.evidence.successful)
    columns = np.asarray(halo.route_columns).reshape((owners, edges))
    np.testing.assert_array_equal(
        np.asarray(halo.route_valid).reshape(valid.shape), valid
    )
    return _Domain(ownership, halo, rows, local_receivers, columns, local_shifts, valid)


class _OwnerResult(NamedTuple):
    energy: float
    gradient: np.ndarray
    strain: np.ndarray
    parameters: Any
    successful: bool


type _TermFactory = Callable[[str], Any]


def _region_inputs(
    domain: _Domain, positions: np.ndarray, numbers: np.ndarray
) -> tuple[Any, ...]:
    present = domain.rows >= 0
    rows = np.where(present, domain.rows, 0)
    owners, capacity = domain.rows.shape
    edges = domain.receivers.shape[1]
    return (
        jnp.asarray(np.where(present[..., None], positions[rows], 0.0).reshape((-1, 3))),
        jnp.asarray(np.where(present, numbers[rows], 0).astype(np.int32).reshape(-1)),
        jnp.asarray(present.reshape(-1)),
        jnp.asarray(domain.receivers.reshape(-1)),
        jnp.asarray(domain.columns.reshape(-1)),
        jnp.asarray(domain.shifts.reshape((owners * edges, 3))),
        jnp.asarray(domain.valid.reshape(-1)),
        domain.halo.send_slots,
        domain.halo.send_valid,
    )


def _owner_topology(
    domain: _Domain, receivers: Any, columns: Any, shifts: Any, valid: Any, mask: Any
) -> OwnerLayerTopology:
    relation = EdgeRelation(
        columns,
        receivers,
        source_size=domain.halo.column_count,
        target_size=domain.ownership.local_capacity,
        valid=valid,
    )
    reduction = RelationExecutionPlan()
    return OwnerLayerTopology(
        STREAMING.prepare(relation, owner_id="layered-test", receiver_valid=mask),
        reduction.prepare(relation),
        reduction.prepare(relation.transpose()),
        receivers,
        columns,
        shifts,
        valid,
    )


def _run_owners(
    model: Any,
    domain: _Domain,
    positions: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    conditioning: AtomisticConditioning | None,
    *,
    mode: OwnerReductionMode = "deterministic",
    term: _TermFactory | None = None,
    message_capacity_bytes: int = 1 << 20,
) -> _OwnerResult:
    axis = domain.ownership.axis_name
    arrays, structure = eqx.partition(model, eqx.is_array)
    vectors = jnp.asarray(cell)

    def body(
        x: Any,
        raw: Any,
        mask: Any,
        receivers: Any,
        columns: Any,
        shifts: Any,
        valid: Any,
        send_slots: Any,
        send_valid: Any,
        parameters: Any,
    ) -> Any:
        local = eqx.combine(parameters, structure)
        result = owner_layer_gradients(
            local,
            HaloColumnExchange(
                domain.halo, send_slots, send_valid, message_capacity_bytes
            ),
            _owner_topology(domain, receivers, columns, shifts, valid, mask),
            x,
            local.species_indices(raw, mask),
            mask,
            vectors,
            cutoff,
            context=OwnerConditioning(conditioning, jnp.zeros(mask.shape, jnp.int32)),
            reduction_mode=mode,
            axis_name=axis,
            accumulation="fast",
            parameters=True,
            global_term=None if term is None else term(axis),
        )
        energy = jnp.sum(result.atom_energies) + result.global_energy
        successful = result.relation_successful & result.model_successful
        return (
            result.gradient,
            energy[None],
            result.strain_gradient[None],
            successful[None],
            jax.tree.map(lambda leaf: leaf[None], result.parameter_gradient),
        )

    owner = PartitionSpec(axis)
    gradient, energies, strains, successful, parameters = domain.ownership.map(
        body, (owner,) * 9 + (PartitionSpec(),), (owner,) * 5
    )(*_region_inputs(domain, positions, numbers), arrays)
    owners, capacity = domain.rows.shape
    present = domain.rows >= 0
    blocked = np.asarray(gradient).reshape((owners, capacity, 3))
    global_gradient = np.zeros_like(positions)
    global_gradient[domain.rows[present]] = blocked[present]
    return _OwnerResult(
        float(jnp.sum(energies)),
        global_gradient,
        np.asarray(jnp.sum(strains, axis=0)),
        jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), parameters),
        bool(jnp.all(successful)),
    )


def _assert_trees_close(left: Any, right: Any, *, rtol: float, atol: float) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for observed, expected in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)


# --- AIMNet2 ---------------------------------------------------------------


def _aimnet2(**options: Any) -> Any:
    declaration, tensors = source_inventory(3, **options)
    core, state, _ = externalize_source_long_range(declaration, tensors)
    return aimnet2_potential_from_source(
        SCALE,
        core,
        state,
        neighbor_cutoff=RC_S,
        ts_pair_cutoff=6.0 if options.get("dispersion") else None,
    ).potential


def _aimnet2_batch(positions: np.ndarray = AIMNET2_POSITIONS) -> AtomisticBatch:
    return AtomisticBatch.from_structure(
        AtomicStructure(AIMNET2_NUMBERS, positions, np.ones(len(AIMNET2_NUMBERS)), SCALE)
    )


def _aimnet2_conditioning(model: Any, charge: float, **values: float) -> Any:
    batch = _aimnet2_batch()
    fields = {"charge": np.asarray([charge], dtype=np.float64)}
    fields.update({name: np.asarray([value]) for name, value in values.items()})
    return AtomisticConditioning(
        model.conditioning_spec,
        fields,
        case_count=batch.case_count,
        atom_capacity=batch.atom_capacity,
    )


def _aimnet2_reference(
    model: Any, conditioning: Any, positions: Any = None, **kwargs: Any
) -> Any:
    batch = _aimnet2_batch()
    payload = AtomisticGraphLayout.for_batch(
        batch, AIMNET2_EXECUTION, cutoff=model.configuration.cutoff
    ).prepare(batch)
    return atomistic_energy_derivatives(
        model,
        payload,
        payload.positions if positions is None else positions,
        conditioning=conditioning,
        **kwargs,
    )


def _parameter_reference(model: Any, energy: Callable[[Any], Any]) -> Any:
    lane, state, fixed = partition_parameters(model)
    return jax.grad(lambda values: energy(combine_parameters(values, state, fixed)))(lane)


@pytest.mark.parametrize("assignment", AIMNET2_ASSIGNMENTS, ids=AIMNET2_IDS)
@pytest.mark.parametrize(
    ("options", "charges"),
    [
        pytest.param({}, {"charge": 1.0}, id="cation"),
        pytest.param(
            {"dispersion": True, "repulsion": True},
            {"charge": -1.0},
            id="anion-dispersion-repulsion",
        ),
        pytest.param(
            {"channels": 2}, {"charge": 0.0, "multiplicity": 3.0}, id="nse-triplet"
        ),
    ],
)
def test_aimnet2_owner_execution_matches_single_device_charge_equilibration(
    assignment: tuple[tuple[int, ...], ...],
    options: dict[str, Any],
    charges: dict[str, float],
) -> None:
    """Per-pass owner sums of q and f equilibrate exactly like the single graph."""
    model = _aimnet2(**options)
    conditioning = _aimnet2_conditioning(model, **charges)
    reference = _aimnet2_reference(model, conditioning)
    expected_parameters = _parameter_reference(
        model,
        lambda rebuilt: _aimnet2_reference(
            rebuilt, conditioning, compute_forces=False
        ).energy[0],
    )
    cutoff = model.configuration.cutoff
    domain = _domain(assignment, _finite_routes(AIMNET2_POSITIONS, cutoff), capacity=6)
    assert model.interaction_count == model.configuration.pass_count + 1
    assert len(model.system_stages) == model.configuration.pass_count - 1
    result = _run_owners(
        model,
        domain,
        AIMNET2_POSITIONS,
        AIMNET2_NUMBERS,
        np.zeros((3, 3)),
        cutoff,
        conditioning,
    )
    assert result.successful
    np.testing.assert_allclose(result.energy, float(reference.energy[0]), rtol=1e-11)
    np.testing.assert_allclose(
        -result.gradient, np.asarray(reference.forces[0]), rtol=1e-9, atol=1e-11
    )
    # Finite and translation invariant: dE/dstrain = sum_i (dE/dx_i) x_i^T.
    expected_strain = np.einsum(
        "ia,ib->ab", -np.asarray(reference.forces[0]), AIMNET2_POSITIONS
    )
    np.testing.assert_allclose(result.strain, expected_strain, rtol=1e-9, atol=1e-11)
    _assert_trees_close(result.parameters, expected_parameters, rtol=1e-9, atol=1e-11)


def test_aimnet2_parameter_gradient_is_invariant_to_the_owner_count() -> None:
    """Exactly-once context cotangents: scaling owners leaves gradients fixed."""
    model = _aimnet2()
    conditioning = _aimnet2_conditioning(model, charge=1.0)
    cutoff = model.configuration.cutoff
    routes = _finite_routes(AIMNET2_POSITIONS, cutoff)
    results = [
        _run_owners(
            model,
            _domain(assignment, routes, capacity=6),
            AIMNET2_POSITIONS,
            AIMNET2_NUMBERS,
            np.zeros((3, 3)),
            cutoff,
            conditioning,
        )
        for assignment in AIMNET2_ASSIGNMENTS
    ]
    first = results[0]
    for other in results[1:]:
        np.testing.assert_allclose(other.energy, first.energy, rtol=1e-12)
        np.testing.assert_allclose(other.gradient, first.gradient, atol=1e-11)
        _assert_trees_close(other.parameters, first.parameters, rtol=1e-10, atol=1e-12)


def test_aimnet2_unsupported_species_is_a_model_failure_not_a_capacity_failure() -> None:
    model = _aimnet2()
    conditioning = _aimnet2_conditioning(model, charge=0.0)
    numbers = AIMNET2_NUMBERS.copy()
    numbers[4] = 9  # fluorine is outside the fixture's implemented species
    cutoff = model.configuration.cutoff
    result = _run_owners(
        model,
        _domain(AIMNET2_ASSIGNMENTS[2], _finite_routes(AIMNET2_POSITIONS, cutoff), 4),
        AIMNET2_POSITIONS,
        numbers,
        np.zeros((3, 3)),
        cutoff,
        conditioning,
    )
    assert not result.successful


def test_aimnet2_mixed_parameter_force_derivative_matches_single_device() -> None:
    """Reverse-over-reverse through stages: d/dtheta of sum |F|^2."""
    model = _aimnet2()
    conditioning = _aimnet2_conditioning(model, charge=1.0)
    cutoff = model.configuration.cutoff
    domain = _domain(
        AIMNET2_ASSIGNMENTS[2], _finite_routes(AIMNET2_POSITIONS, cutoff), capacity=4
    )
    expected = _parameter_reference(
        model,
        lambda rebuilt: jnp.sum(_aimnet2_reference(rebuilt, conditioning).forces[0] ** 2),
    )
    axis = domain.ownership.axis_name
    arrays, structure = eqx.partition(model, eqx.is_array)

    def body(
        x: Any,
        raw: Any,
        mask: Any,
        receivers: Any,
        columns: Any,
        shifts: Any,
        valid: Any,
        send_slots: Any,
        send_valid: Any,
        parameters: Any,
    ) -> Any:
        lane, state, fixed = partition_parameters(eqx.combine(parameters, structure))

        def surrogate(values: Any) -> Any:
            local = combine_parameters(values, state, fixed)
            result = owner_layer_gradients(
                local,
                HaloColumnExchange(domain.halo, send_slots, send_valid, 1 << 20),
                _owner_topology(domain, receivers, columns, shifts, valid, mask),
                x,
                local.species_indices(raw, mask),
                mask,
                jnp.zeros((3, 3)),
                cutoff,
                context=OwnerConditioning(conditioning, jnp.zeros(mask.shape, jnp.int32)),
                reduction_mode="deterministic",
                axis_name=axis,
                accumulation="fast",
                parameters=False,
            )
            return jnp.sum(result.gradient**2)

        return jax.tree.map(lambda leaf: leaf[None], jax.grad(surrogate)(lane))

    owner = PartitionSpec(axis)
    partials = domain.ownership.map(body, (owner,) * 9 + (PartitionSpec(),), owner)(
        *_region_inputs(domain, AIMNET2_POSITIONS, AIMNET2_NUMBERS), arrays
    )
    _assert_trees_close(
        jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), partials),
        expected,
        rtol=1e-8,
        atol=1e-10,
    )


# --- MACE ------------------------------------------------------------------


def _mace(*, conditioned: bool) -> MACEPotential:
    declaration: dict[str, Any] = {
        "species": (1, 8),
        "cutoff": MACE_CUTOFF,
        "radial_basis_count": 4,
        "cutoff_power": 5,
        "channel_count": 3,
        "hidden_degree": 1,
        "edge_degree": 2,
        "interactions": ("real-agnostic-density", "real-agnostic-density-residual"),
        "correlations": (2, 2),
        "radial_widths": (4,),
        "readout_width": 3,
        "average_neighbor_count": 6.0,
        "energy_scaling": "scale-shift",
    }
    if conditioned:
        declaration.update(embedding_features=CASE_FEATURES, embedding_readout=True)
    return MACEPotential(
        SCALE,
        MACEArchitecture(**declaration),
        atomic_energies=np.asarray([[-1.0, -2.0]]),
        energy_scale=np.asarray([1.3]),
        energy_shift=np.asarray([0.2]),
        key=jr.key(7),
    )


def _mace_conditioning(model: MACEPotential) -> AtomisticConditioning | None:
    if model.joint_embedding is None:
        return None
    return AtomisticConditioning(
        model.conditioning,
        {
            "total_charge": np.asarray([1], dtype=np.int64),
            "total_spin": np.asarray([2], dtype=np.int64),
            "temperature": np.asarray([0.3]),
        },
        case_count=1,
        atom_capacity=len(MACE_NUMBERS),
    )


def _mace_states(
    model: MACEPotential,
    positions: Any,
    cell: Any,
    routes: _Routes,
    conditioning: AtomisticConditioning | None,
) -> Any:
    """Single-device node states on the global route relation."""
    senders, receivers, shifts = routes
    count = positions.shape[0]
    relation = STREAMING.prepare(
        EdgeRelation(senders, receivers, source_size=count, target_size=count),
        owner_id="global-reference",
    )
    vectors = positions[receivers] - positions[senders] + shifts.astype(np.float64) @ cell
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = model.species_indices(jnp.asarray(MACE_NUMBERS), mask)
    joint = model.joint_features(conditioning, jnp.zeros((count,), jnp.int32), mask)
    return species, model.node_states(species, mask, relation, vectors, joint=joint)


def _strained(strain: Any) -> tuple[Any, Any]:
    deformation = jnp.eye(3) + strain
    return (
        jnp.asarray(MACE_POSITIONS) @ deformation.T,
        jnp.asarray(MACE_CELL) @ deformation.T,
    )


@pytest.mark.parametrize("assignment", MACE_ASSIGNMENTS, ids=MACE_IDS)
@pytest.mark.parametrize("conditioned", [False, True], ids=("plain", "joint-embedding"))
@pytest.mark.parametrize("mode", ["fast", "deterministic", "compensated"])
def test_mace_owner_execution_matches_single_device_with_images(
    assignment: tuple[tuple[int, ...], ...], conditioned: bool, mode: OwnerReductionMode
) -> None:
    """Conditioned MACE carries its joint addend and readout energy in context."""
    model = _mace(conditioned=conditioned)
    conditioning = _mace_conditioning(model)
    routes = _image_routes(MACE_POSITIONS, MACE_CELL, MACE_CUTOFF)

    def energy(candidate: MACEPotential, strain: Any) -> Any:
        positions, cell = _strained(strain)
        _, states = _mace_states(candidate, positions, cell, routes, conditioning)
        return jnp.sum(states.atom_energy)

    zero = jnp.zeros((3, 3))
    expected_energy = energy(model, zero)
    expected_gradient = jax.grad(
        lambda x: _position_energy(model, x, routes, conditioning)
    )(jnp.asarray(MACE_POSITIONS))
    expected_strain = jax.grad(lambda strain: energy(model, strain))(zero)
    expected_parameters = _parameter_reference(
        model, lambda rebuilt: energy(rebuilt, zero)
    )
    result = _run_owners(
        model,
        _domain(assignment, routes, capacity=4),
        MACE_POSITIONS,
        MACE_NUMBERS,
        MACE_CELL,
        MACE_CUTOFF,
        conditioning,
        mode=mode,
    )
    assert result.successful
    np.testing.assert_allclose(result.energy, float(expected_energy), rtol=1e-11)
    np.testing.assert_allclose(result.gradient, expected_gradient, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(result.strain, expected_strain, rtol=1e-9, atol=1e-11)
    _assert_trees_close(result.parameters, expected_parameters, rtol=1e-9, atol=1e-11)


def _position_energy(
    model: MACEPotential,
    positions: Any,
    routes: _Routes,
    conditioning: AtomisticConditioning | None,
) -> Any:
    _, states = _mace_states(
        model, positions, jnp.asarray(MACE_CELL), routes, conditioning
    )
    return jnp.sum(states.atom_energy)


def test_unconditioned_mace_layer_context_is_empty_and_node_states_unchanged() -> None:
    """No system stage, no payload: the layered hooks are the plain MACE layers."""
    model = _mace(conditioned=False)
    mask = jnp.ones((len(MACE_NUMBERS),), dtype=jnp.bool_)
    species = model.species_indices(jnp.asarray(MACE_NUMBERS), mask)
    context = model.layer_context(None, jnp.zeros(mask.shape, jnp.int32), species, mask)
    assert context.joint is None and context.embedding_energy is None
    assert model.system_stages == ()
    routes = _image_routes(MACE_POSITIONS, MACE_CELL, MACE_CUTOFF)
    _, states = _mace_states(
        model, jnp.asarray(MACE_POSITIONS), jnp.asarray(MACE_CELL), routes, None
    )
    features = model.init_features(species, mask, context)
    assert model.global_payload(features, species, mask, context) is None
    np.testing.assert_array_equal(
        np.asarray(features),
        np.asarray(model.embedding[species] / jnp.sqrt(model.embedding.shape[0])),
    )
    layered = model.atom_energies(
        species,
        mask,
        jnp.zeros((model.interaction_count, mask.shape[0])),
        states.states[-1],
        context,
    )
    np.testing.assert_array_equal(
        np.asarray(layered),
        np.asarray(model.energy_reference.atom_energy(species, jnp.zeros(mask.shape))),
    )


def test_halo_message_over_capacity_is_refused_before_communication() -> None:
    model = _mace(conditioned=False)
    routes = _image_routes(MACE_POSITIONS, MACE_CELL, MACE_CUTOFF)
    with pytest.raises(AtomisticExchangeCapacityError, match="message_capacity_bytes"):
        _run_owners(
            model,
            _domain(MACE_ASSIGNMENTS[0], routes, capacity=4),
            MACE_POSITIONS,
            MACE_NUMBERS,
            MACE_CELL,
            MACE_CUTOFF,
            None,
            message_capacity_bytes=16,
        )


# --- MACELES (nonlocal: global payload epilogue) ---------------------------


def _screened_pair_energy(positions: Any, charges: Any) -> Any:
    """Dense ``1/2 sum_{i != j} q_i q_j / sqrt(r_ij^2 + 1)`` test global energy."""
    difference = positions[:, None, :] - positions[None, :, :]
    kernel = 1.0 / jnp.sqrt(jnp.sum(difference * difference, axis=-1) + 1.0)
    off = 1.0 - jnp.eye(positions.shape[0])
    return 0.5 * jnp.sum(charges[:, None] * charges[None, :] * kernel * off)


def _gathered_pair_term(axis: str) -> Callable[..., Any]:
    """An owner global term with real collectives: this owner's row share."""

    def term(positions: Any, payload: Any, atom_mask: Any, cell: Any) -> Any:
        del cell
        local = positions.shape[0]
        owned = jnp.where(atom_mask, payload, 0.0)
        everyone = jax.lax.all_gather(positions, axis, tiled=True)
        charges = jax.lax.all_gather(owned, axis, tiled=True)
        first = jax.lax.axis_index(axis) * local
        difference = positions[:, None, :] - everyone[None, :, :]
        kernel = 1.0 / jnp.sqrt(jnp.sum(difference * difference, axis=-1) + 1.0)
        rows = first + jnp.arange(local)
        off = rows[:, None] != jnp.arange(everyone.shape[0])[None, :]
        share = 0.5 * jnp.sum(owned[:, None] * charges[None, :] * kernel * off)
        return share, jnp.asarray(True)

    return term


def _maceles() -> MACELESPotential:
    return MACELESPotential.initialize(
        _mace(conditioned=True),
        LatentEwaldDefinition.from_source_arguments({}),
        atom_capacity=len(MACE_NUMBERS),
        resources=LatentEwaldResources(mode_extents=None),
        key=jr.key(13),
    )


def test_maceles_latent_payload_pulls_back_through_the_reverse_sweep() -> None:
    """Global-term position, payload (latent charge) and strain cotangents."""
    model = _maceles()
    conditioning = _mace_conditioning(model.mace)
    routes = _image_routes(MACE_POSITIONS, MACE_CELL, MACE_CUTOFF)

    def energy(candidate: MACELESPotential, strain: Any) -> Any:
        positions, cell = _strained(strain)
        species, states = _mace_states(
            candidate.mace, positions, cell, routes, conditioning
        )
        charges = candidate.latent_charges(species, states)
        return jnp.sum(states.atom_energy) + _screened_pair_energy(positions, charges)

    zero = jnp.zeros((3, 3))

    def position_energy(positions: Any) -> Any:
        species, states = _mace_states(
            model.mace, positions, jnp.asarray(MACE_CELL), routes, conditioning
        )
        charges = model.latent_charges(species, states)
        return jnp.sum(states.atom_energy) + _screened_pair_energy(positions, charges)

    result = _run_owners(
        model,
        _domain(MACE_ASSIGNMENTS[1], routes, capacity=4),
        MACE_POSITIONS,
        MACE_NUMBERS,
        MACE_CELL,
        MACE_CUTOFF,
        conditioning,
        term=_gathered_pair_term,
    )
    assert result.successful
    np.testing.assert_allclose(result.energy, float(energy(model, zero)), rtol=1e-11)
    np.testing.assert_allclose(
        result.gradient,
        jax.grad(position_energy)(jnp.asarray(MACE_POSITIONS)),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        result.strain,
        jax.grad(lambda strain: energy(model, strain))(zero),
        rtol=1e-9,
        atol=1e-11,
    )
    _assert_trees_close(
        result.parameters,
        _parameter_reference(model, lambda rebuilt: energy(rebuilt, zero)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_maceles_is_refused_without_its_global_term() -> None:
    model = _maceles()
    routes = _image_routes(MACE_POSITIONS, MACE_CELL, MACE_CUTOFF)
    with pytest.raises(ValueError, match="nonlocal"):
        _run_owners(
            model,
            _domain(MACE_ASSIGNMENTS[0], routes, capacity=4),
            MACE_POSITIONS,
            MACE_NUMBERS,
            MACE_CELL,
            MACE_CUTOFF,
            _mace_conditioning(model.mace),
        )


# --- forward mode: Hessian actions through owner regions --------------------


def _owner_hessian_action(
    model: Any,
    domain: _Domain,
    positions: np.ndarray,
    numbers: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    conditioning: AtomisticConditioning | None,
    direction: np.ndarray,
    *,
    mode: OwnerReductionMode = "deterministic",
) -> np.ndarray:
    """``H v = d(dE/dx)[v]`` by ``jax.linearize`` of the owner gradients.

    The tangent ``v`` is owner blocked like the coordinates; it crosses the
    halo as the same exchange and the stage owner sums as the same sums.
    """
    axis = domain.ownership.axis_name
    arrays, structure = eqx.partition(model, eqx.is_array)
    vectors = jnp.asarray(cell)
    present = domain.rows >= 0
    rows = np.where(present, domain.rows, 0)
    tangent = np.where(present[..., None], direction[rows], 0.0).reshape((-1, 3))

    def body(
        x: Any,
        raw: Any,
        mask: Any,
        receivers: Any,
        columns: Any,
        shifts: Any,
        valid: Any,
        send_slots: Any,
        send_valid: Any,
        v: Any,
        parameters: Any,
    ) -> Any:
        local = eqx.combine(parameters, structure)
        exchange = HaloColumnExchange(domain.halo, send_slots, send_valid, 1 << 20)
        topology = _owner_topology(domain, receivers, columns, shifts, valid, mask)
        species = local.species_indices(raw, mask)

        def gradient(coordinates: Any) -> Any:
            return owner_layer_gradients(
                local,
                exchange,
                topology,
                coordinates,
                species,
                mask,
                vectors,
                cutoff,
                context=OwnerConditioning(conditioning, jnp.zeros(mask.shape, jnp.int32)),
                reduction_mode=mode,
                axis_name=axis,
                accumulation="fast",
                parameters=False,
            ).gradient

        _, linear = jax.linearize(gradient, x)
        return linear(v)

    owner = PartitionSpec(axis)
    action = domain.ownership.map(body, (owner,) * 10 + (PartitionSpec(),), owner)(
        *_region_inputs(domain, positions, numbers), jnp.asarray(tangent), arrays
    )
    owners, capacity = domain.rows.shape
    blocked = np.asarray(action).reshape((owners, capacity, 3))
    result = np.zeros_like(positions)
    result[domain.rows[present]] = blocked[present]
    return result


def _direction(count: int) -> np.ndarray:
    return np.random.default_rng(23).normal(size=(count, 3))


@pytest.mark.parametrize("assignment", MACE_ASSIGNMENTS, ids=MACE_IDS)
def test_mace_domain_hessian_action_matches_single_device(
    assignment: tuple[tuple[int, ...], ...],
) -> None:
    """Conditioned periodic MACE: forward-over-reverse through every exchange."""
    model = _mace(conditioned=True)
    conditioning = _mace_conditioning(model)
    routes = _image_routes(MACE_POSITIONS, MACE_CELL, MACE_CUTOFF)
    direction = _direction(len(MACE_NUMBERS))
    _, expected = jax.jvp(
        jax.grad(lambda x: _position_energy(model, x, routes, conditioning)),
        (jnp.asarray(MACE_POSITIONS),),
        (jnp.asarray(direction),),
    )
    observed = _owner_hessian_action(
        model,
        _domain(assignment, routes, capacity=4),
        MACE_POSITIONS,
        MACE_NUMBERS,
        MACE_CELL,
        MACE_CUTOFF,
        conditioning,
        direction,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("mode", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize("assignment", AIMNET2_ASSIGNMENTS[1:], ids=AIMNET2_IDS[1:])
def test_aimnet2_domain_hessian_action_matches_single_device(
    assignment: tuple[tuple[int, ...], ...], mode: OwnerReductionMode
) -> None:
    """Charged AIMNet2: stage owner sums carry tangents and transposes once."""
    model = _aimnet2()
    conditioning = _aimnet2_conditioning(model, charge=1.0)
    cutoff = model.configuration.cutoff
    direction = _direction(len(AIMNET2_NUMBERS))
    _, expected = jax.jvp(
        lambda x: -_aimnet2_reference(model, conditioning, x[None]).forces[0],
        (jnp.asarray(AIMNET2_POSITIONS),),
        (jnp.asarray(direction),),
    )
    observed = _owner_hessian_action(
        model,
        _domain(assignment, _finite_routes(AIMNET2_POSITIONS, cutoff), capacity=6),
        AIMNET2_POSITIONS,
        AIMNET2_NUMBERS,
        np.zeros((3, 3)),
        cutoff,
        conditioning,
        direction,
        mode=mode,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-10)


# --- owner_system_sum ------------------------------------------------------


@pytest.mark.parametrize("mode", ["fast", "deterministic", "compensated"])
def test_owner_system_sum_is_replicated_exact_for_counts_and_self_adjoint(
    mode: OwnerReductionMode,
) -> None:
    ownership = _lane_ownership(3, 1)
    axis = ownership.axis_name
    values = jnp.asarray([1.5, -0.25, 2.0])
    counts = jnp.asarray([2, 0, 5], dtype=jnp.int32)
    weights = jnp.asarray([0.5, 3.0, -1.0])

    def body(value: Any, count: Any, weight: Any) -> Any:
        total, number = owner_system_sum((value[0], count[0]), axis, mode)
        # Per-owner local surrogate L_o = w_o S^2 with S the owner sum: each
        # owner's VJP receives the complete dL/dS = 2 S sum_o w_o exactly once.
        gradient = jax.grad(
            lambda own: weight[0] * owner_system_sum(own, axis, mode) ** 2
        )(value[0])
        # Forward mode: the tangent of the owner sum is the owner sum of the
        # tangents; JAX's transpose of the linearized sum returns sum_o w_o to
        # every owner once (an owner-count-scaled transpose would differ).
        _, tangent = jax.jvp(
            lambda own: owner_system_sum(own, axis, mode),
            (value[0],),
            (weight[0],),
        )
        _, linear = jax.linearize(
            lambda own: owner_system_sum(own, axis, mode),
            value[0],
        )
        (transposed,) = jax.linear_transpose(linear, value[0])(weight[0])
        return (
            total[None],
            number[None],
            gradient[None],
            tangent[None],
            transposed[None],
        )

    owner = PartitionSpec(axis)
    totals, numbers, gradients, tangents, transposed = ownership.map(body, owner, owner)(
        values, counts, weights
    )
    total = float(jnp.sum(values))
    weight = float(jnp.sum(weights))
    np.testing.assert_allclose(np.asarray(totals), total, rtol=1e-15)
    assert numbers.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(numbers), 7)
    np.testing.assert_allclose(np.asarray(gradients), 2.0 * total * weight, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(tangents), weight, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(transposed), weight, rtol=1e-15)


def test_owner_system_sum_refuses_boolean_partials() -> None:
    ownership = _lane_ownership(2, 1)
    axis = ownership.axis_name

    def body(flag: Any) -> Any:
        return owner_system_sum(flag, axis, "fast")[None]

    owner = PartitionSpec(axis)
    with pytest.raises(TypeError, match="int32"):
        ownership.map(body, owner, owner)(jnp.asarray([True, False]))
