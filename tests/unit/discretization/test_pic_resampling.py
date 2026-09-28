#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
RELATIVITY = PIC.PIC_CODE_RELATIVITY
_NO_PARENT = np.uint32(0xFFFFFFFF)


def _species(capacity: int, sign: float, name: str, dimension: int, offset: int) -> Any:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + capacity),
        jnp.ones((capacity,)),
        ambient_dimension=dimension,
    ).prepare()
    return PIC.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        PIC.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


def _state(plan: Any, position: Any, velocity: Any, mass: Any, active: Any) -> Any:
    return plan.initialize(
        jnp.asarray(position),
        jnp.asarray(velocity),
        active_mask=jnp.asarray(active),
        masses=jnp.where(jnp.asarray(active), jnp.asarray(mass), 0.0),
    )


def _apply(process: Any, plans: tuple[Any, ...], states: tuple[Any, ...]) -> Any:
    capacity = states[0].population.active.shape[0]
    zero = jnp.zeros((capacity, 3))
    context = PIC.PICProcessContext(
        states,
        (zero,) * len(states),
        (zero,) * len(states),
        jnp.asarray(0.0),
        jnp.asarray(0.1),
        jnp.asarray(0, dtype=jnp.int32),
        None,
    )
    return process.apply(plans, context)


_apply_jit = eqx.filter_jit(_apply)


def _identity(population: Any) -> np.ndarray:
    return (np.asarray(population.id_hi).astype(np.uint64) << np.uint64(32)) | np.asarray(
        population.id_lo
    ).astype(np.uint64)


def _totals(plan: Any, state: Any) -> dict[str, np.ndarray]:
    """Independent float64 totals of the active population (c = 1)."""
    active = np.asarray(state.population.active)
    mass = np.where(active, np.asarray(state.population.mass), 0.0)
    velocity = np.asarray(state.particles.proper_velocity)[active]
    position = np.asarray(state.particles.position)[active]
    weights = mass[active]
    gamma = np.sqrt(1.0 + np.sum(velocity**2, axis=-1))
    return {
        "charge": np.sum(np.asarray(plan.macrocharge(state))),
        "mass": np.sum(weights),
        "momentum": np.sum(weights[:, None] * velocity, axis=0),
        "energy": np.sum(weights * gamma),
        "dipole": np.sum(weights[:, None] * position, axis=0),
    }


def _assert_conserved(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> None:
    for name in ("charge", "mass", "energy"):
        np.testing.assert_allclose(after[name], before[name], rtol=1e-13, err_msg=name)
    for name in ("momentum", "dipole"):
        np.testing.assert_allclose(after[name], before[name], rtol=1e-13, atol=1e-13)


def test_vranic_pairs_reproduce_packet_weight_momentum_energy_and_center() -> None:
    # One crowded cell and one momentum cell: identities 0-3 and 4-7 form the
    # two packets of four in identity order.
    rng = np.random.default_rng(7)
    plan = _species(12, -1.0, "electrons", 1, 0)
    active = np.arange(12) < 8
    position = rng.uniform(0.05, 0.45, (12, 1))
    velocity = rng.normal(0.0, 0.8, (12, 3))
    mass = rng.uniform(0.5, 2.0, 12)
    state = _state(plan, position, velocity, mass, active)
    merge = PIC.ParticleMergePlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (2,), (True,)),
        RELATIVITY,
        species=(0,),
        maximum_per_cell=4,
        momentum_bins=(1, 1, 1),
        minimum_packet_size=3,
        maximum_packet_size=4,
    )
    result = _apply(merge, (plan,), (state,))
    merged = result.species[0]
    (evidence,) = result.evidence
    assert int(evidence.events) == 2
    assert int(evidence.removed) == 8 and int(evidence.created) == 4
    assert int(evidence.status) == int(PIC.ParticleResamplingStatus.NONE)
    assert result.ledger.successful

    population = merged.population
    alive = np.asarray(population.active)
    ids = _identity(population)[alive]
    parents = np.asarray(population.parent_lo)[alive]
    # Fresh identities continue the counter; parents are the lowest constituents.
    np.testing.assert_array_equal(np.sort(ids), np.arange(8, 12, dtype=np.uint64))
    order = np.argsort(ids)
    np.testing.assert_array_equal(parents[order], [0, 0, 4, 4])
    product_mass = np.asarray(population.mass)[alive][order]
    product_velocity = np.asarray(merged.particles.proper_velocity)[alive][order]
    product_position = np.asarray(merged.particles.position)[alive][order]
    for pair, members in enumerate((np.arange(4), np.arange(4, 8))):
        weight = np.sum(mass[members])
        momentum = np.sum(mass[members, None] * velocity[members], axis=0)
        kinetic = np.sum(
            mass[members] * (np.sqrt(1.0 + np.sum(velocity[members] ** 2, -1)) - 1.0)
        )
        center = np.sum(mass[members, None] * position[members], axis=0) / weight
        products = slice(2 * pair, 2 * pair + 2)
        gamma = np.sqrt(1.0 + np.sum(product_velocity[products] ** 2, axis=-1))
        np.testing.assert_allclose(product_mass[products], 0.5 * weight, rtol=1e-15)
        np.testing.assert_allclose(
            0.5 * weight * np.sum(product_velocity[products], axis=0),
            momentum,
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(gamma, 1.0 + kinetic / weight, rtol=1e-13)
        np.testing.assert_allclose(
            product_position[products], [center, center], rtol=1e-14
        )


def test_identical_momenta_merge_without_momentum_distortion() -> None:
    plan = _species(8, 1.0, "ions", 2, 0)
    active = np.ones(8, dtype=bool)
    position = np.random.default_rng(3).uniform(0.1, 0.4, (8, 2))
    velocity = np.tile([0.3, -0.2, 0.5], (8, 1))
    state = _state(plan, position, velocity, np.full(8, 1.5), active)
    merge = PIC.ParticleMergePlan(
        PIC.PICCellBinningPlan((0.0, 0.0), (1.0, 1.0), (2, 2), (True, True)),
        RELATIVITY,
        species=(0,),
        maximum_per_cell=2,
        momentum_bins=(2, 2, 2),
        maximum_packet_size=8,
    )
    result = _apply(merge, (plan,), (state,))
    merged = result.species[0]
    (evidence,) = result.evidence
    alive = np.asarray(merged.population.active)
    assert int(np.sum(alive)) == 2
    # cos ω = 1 - O(ε) opens the pair by ω = O(√ε); the moment change is O(ε).
    np.testing.assert_allclose(
        np.asarray(merged.particles.proper_velocity)[alive], velocity[:2], rtol=1e-7
    )
    assert float(evidence.momentum_second_moment_distortion) < 1e-13
    # Collapsing the packet to its center removes its central second moment,
    # reported relative to (total mass) x (cell width)².
    offset = position - position.mean(axis=0)
    central = 1.5 * offset.T @ offset
    np.testing.assert_allclose(
        float(evidence.spatial_second_moment_distortion),
        np.linalg.norm(central) / (12.0 * 0.5**2),
        rtol=1e-12,
    )
    _assert_conserved(_totals(plan, state), _totals(plan, merged))


def _permuted(state: Any, permutation: np.ndarray) -> Any:
    capacity = permutation.size
    return jax.tree.map(
        lambda value: (
            value[permutation]
            if value.ndim >= 1 and value.shape[0] == capacity
            else value
        ),
        state,
    )


def _by_identity(state: Any) -> tuple[np.ndarray, ...]:
    population = state.population
    alive = np.asarray(population.active)
    identity = _identity(population)[alive]
    order = np.argsort(identity)
    return (
        identity[order],
        np.asarray(population.parent_lo)[alive][order],
        np.asarray(population.mass)[alive][order],
        np.asarray(state.particles.position)[alive][order],
        np.asarray(state.particles.proper_velocity)[alive][order],
        np.asarray(population.next_id_lo),
    )


@pytest.mark.parametrize("kind", ["merge", "split"])
def test_resampling_is_invariant_to_storage_slot_order(kind: str) -> None:
    rng = np.random.default_rng(11)
    capacity = 48
    plan = _species(capacity, -1.0, "electrons", 2, 0)
    active = np.arange(capacity) < 30
    state = _state(
        plan,
        rng.uniform(0.0, 1.0, (capacity, 2)),
        rng.normal(0.0, 0.5, (capacity, 3)),
        rng.uniform(0.5, 2.0, capacity),
        active,
    )
    if kind == "merge":
        process: Any = PIC.ParticleMergePlan(
            PIC.PICCellBinningPlan((0.0, 0.0), (1.0, 1.0), (2, 2), (True, True)),
            RELATIVITY,
            species=(0,),
            maximum_per_cell=4,
            momentum_bins=(1, 1, 2),
            minimum_packet_size=3,
            maximum_packet_size=3,
        )
    else:
        process = PIC.ParticleSplitPlan(
            PIC.PICCellBinningPlan((0.0, 0.0), (1.0, 1.0), (8, 8), (True, True)),
            RELATIVITY,
            species=(0,),
            minimum_per_cell=2,
            minimum_child_mass=0.1,
            maximum_splits=16,
        )
    permutation = rng.permutation(capacity)
    reference = _apply_jit(process, (plan,), (state,))
    shuffled = _apply_jit(process, (plan,), (_permuted(state, permutation),))
    assert int(reference.evidence[0].events) > 0
    assert int(shuffled.evidence[0].events) == int(reference.evidence[0].events)
    assert int(shuffled.evidence[0].refused) == int(reference.evidence[0].refused)
    for expected, actual in zip(
        _by_identity(reference.species[0]), _by_identity(shuffled.species[0]), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def _inject(plan: Any, state: Any, position: Any, velocity: Any, mass: Any) -> Any:
    """Allocate new particles through the population, as a creating process does."""
    capacity = state.population.active.shape[0]
    width = mass.shape[0]
    allocation = plan.population.allocate(
        state.population,
        D.ParticleAllocationRequest(
            jnp.arange(width), jnp.asarray(mass), jnp.ones((width,), dtype=jnp.bool_)
        ),
    )
    slots = jnp.where(allocation.allocated, allocation.slots, capacity)
    charge = state.charge
    injected = PIC.PICSpeciesState(
        PIC.PICParticleState(
            state.particles.position.at[slots].set(jnp.asarray(position), mode="drop"),
            state.particles.proper_velocity.at[slots].set(
                jnp.asarray(velocity), mode="drop"
            ),
        ),
        allocation.accepted_state,
        PIC.PICChargeState(
            charge.charge_number.at[slots].set(1, mode="drop"),
            charge.transition_count,
            charge.last_transition_step,
        ),
    )
    return injected, allocation.successful


def test_merging_bounds_the_population_under_sustained_injection() -> None:
    rng = np.random.default_rng(5)
    capacity, burst = 64, 8
    plan = _species(capacity, -1.0, "electrons", 1, 0)
    state = _state(
        plan,
        np.zeros((capacity, 1)),
        np.zeros((capacity, 3)),
        np.zeros(capacity),
        np.zeros(capacity, dtype=bool),
    )
    merge = PIC.ParticleMergePlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (4,), (True,)),
        RELATIVITY,
        species=(0,),
        maximum_per_cell=burst,
        momentum_bins=(1, 1, 1),
        minimum_packet_size=3,
        maximum_packet_size=4,
    )
    injected_charge = 0.0
    for _ in range(30):
        mass = rng.uniform(0.5, 1.5, burst)
        state, allocated = _inject(
            plan,
            state,
            rng.uniform(0.0, 0.25, (burst, 1)),
            rng.normal(0.0, 0.4, (burst, 3)),
            mass,
        )
        assert allocated
        injected_charge -= float(np.sum(mass))
        result = _apply_jit(merge, (plan,), (state,))
        state = result.species[0]
        (evidence,) = result.evidence
        assert int(evidence.maximum_cell_count_after) <= burst
        assert int(evidence.active_after) <= burst
    # 240 particles entered a 64-slot population; their charge is all retained.
    np.testing.assert_allclose(
        float(jnp.sum(plan.macrocharge(state))), injected_charge, rtol=1e-12
    )


def test_split_children_conserve_moments_and_record_their_parent() -> None:
    plan = _species(10, -1.0, "electrons", 1, 0)
    position = np.asarray([0.1, 0.3, 0.4, 0.55, 0.6, 0.7, 0.99, 0.0, 0.0, 0.0])[:, None]
    velocity = np.random.default_rng(2).normal(0.0, 0.6, (10, 3))
    mass = np.asarray([4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0])
    state = _state(plan, position, velocity, mass, mass > 0.0)
    split = PIC.ParticleSplitPlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (4,), (False,)),
        RELATIVITY,
        species=(0,),
        minimum_per_cell=3,
        minimum_child_mass=0.1,
        maximum_splits=4,
    )
    result = _apply(split, (plan,), (state,))
    child_state = result.species[0]
    (evidence,) = result.evidence
    # Cells 0 and 1 are sparse; the particle at 0.99 would put a child past the
    # nonperiodic wall and is not split.
    assert int(evidence.events) == 2 and int(evidence.unsupported) == 1
    assert int(evidence.created) == 4 and int(evidence.refused) == 0
    assert int(evidence.status) == int(PIC.ParticleResamplingStatus.UNSUPPORTED)
    identity, parent, child_mass, child_position, child_velocity, _ = _by_identity(
        child_state
    )
    children = identity >= 7
    np.testing.assert_array_equal(identity[children], [7, 8, 9, 10])
    np.testing.assert_array_equal(parent[children], [0, 0, 1, 1])
    np.testing.assert_allclose(child_mass[children], [2.0, 2.0, 1.0, 1.0])
    delta = 0.25 * 0.25
    np.testing.assert_allclose(
        child_position[children, 0], [0.1 - delta, 0.1 + delta, 0.3 - delta, 0.3 + delta]
    )
    np.testing.assert_array_equal(child_velocity[children], velocity[[0, 0, 1, 1]])
    # Each split of mass m adds m δ² to Σ m x², reported per (Σ m) h².
    np.testing.assert_allclose(
        float(evidence.spatial_second_moment_distortion),
        (4.0 + 2.0) * delta**2 / (np.sum(mass) * 0.25**2),
        rtol=1e-12,
    )
    _assert_conserved(_totals(plan, state), _totals(plan, child_state))


def test_splitting_beyond_free_capacity_is_refused_in_canonical_order() -> None:
    plan = _species(6, -1.0, "electrons", 1, 0)
    position = np.asarray([0.1, 0.35, 0.6, 0.85, 0.0, 0.0])[:, None]
    active = np.arange(6) < 4
    state = _state(plan, position, np.zeros((6, 3)), np.ones(6), active)
    split = PIC.ParticleSplitPlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (4,), (True,)),
        RELATIVITY,
        species=(0,),
        minimum_per_cell=2,
        minimum_child_mass=0.1,
        maximum_splits=4,
    )
    result = _apply(split, (plan,), (state,))
    (evidence,) = result.evidence
    # Two free slots admit two splits (each frees its parent's slot); the two
    # later cells are refused rather than truncated.
    assert int(evidence.events) == 2 and int(evidence.refused) == 2
    assert int(evidence.status) == int(PIC.ParticleResamplingStatus.CAPACITY_REFUSED)
    assert int(evidence.active_after) == 6
    _, parent, *_ = _by_identity(result.species[0])
    np.testing.assert_array_equal(np.sort(parent), [0, 0, 1, 1, 2**32 - 1, 2**32 - 1])


def _neutral_species(dimension: int, count: int) -> tuple[Any, Any]:
    return (
        _species(count, -1.0, "electrons", dimension, 0),
        _species(count, 1.0, "ions", dimension, 100),
    )


def _merge(dimension: int, lower: float, upper: float, cells: int, periodic: bool) -> Any:
    return PIC.ParticleMergePlan(
        PIC.PICCellBinningPlan(
            (lower,) * dimension,
            (upper,) * dimension,
            (cells,) * dimension,
            (periodic,) * dimension,
        ),
        RELATIVITY,
        species=(0,),
        maximum_per_cell=3,
        momentum_bins=(1, 1, 1),
        minimum_packet_size=3,
        maximum_packet_size=3,
    )


def _reduced_run(dimension: int) -> tuple[Any, Any, Any]:
    count = 6
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(8, periodic=True) for _ in range(dimension)),
        axis_names=("x", "y")[:dimension],
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    field = (
        phx.solver.CompatibleMaxwell1DPlan(grid)
        if dimension == 1
        else phx.solver.CompatibleMaxwell2DPlan(grid)
    )
    solver = phx.solver.ReducedMaxwellPICFieldSolver(
        field, PIC.ReducedPICTransferPlan(grid)
    )
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=_neutral_species(dimension, count),
        processes=(_merge(dimension, 0.0, 1.0, 2, True),),
    )
    rng = np.random.default_rng(dimension)
    # In-plane velocities: the charge redistribution is the only Gauss source.
    velocity = np.zeros((count, 3))
    velocity[:, :dimension] = rng.normal(0.0, 0.05, (count, dimension))
    dt = 0.2 * solver.field.stable_dt
    state = pic.initialize(
        (
            rng.uniform(0.05, 0.45, (count, dimension)),
            rng.uniform(0.0, 1.0, (count, dimension)),
        ),
        (velocity, np.zeros((count, 3))),
        dt,
    )
    return pic, state, dt


def _cochain_run() -> tuple[Any, Any, Any]:
    count = 6
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(3, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = D.StructuredCochainBridge(grid)
    species, transfers = [], []
    for offset, sign, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        particles = D.ParticleSetPlan(
            jnp.arange(offset, offset + count), jnp.ones((count,)), ambient_dimension=3
        ).prepare()
        charged = D.ChargedParticlePlan(sign * jnp.ones((count,)), name).prepare(
            particles
        )
        transfers.append(PIC.PICParticleCochainTransferPlan(bridge).prepare(charged))
        species.append(
            PIC.PICSpeciesPlan(
                D.ParticlePopulationPlan(particles),
                PIC.PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="test-resampling-maxwell",
    ).prepare()
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        phx.solver.CochainElectrostaticPlan(
            bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        tuple(transfers),
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )
    pic = phx.solver.ElectromagneticPICPlan(
        solver, species=tuple(species), processes=(_merge(3, 0.0, 1.0, 3, True),)
    )
    rng = np.random.default_rng(3)
    dt = 0.01 * maxwell.stable_dt
    state = pic.initialize(
        (rng.uniform(0.4, 0.6, (count, 3)), rng.uniform(0.0, 1.0, (count, 3))),
        (rng.normal(0.0, 0.05, (count, 3)), np.zeros((count, 3))),
        dt,
    )
    return pic, state, dt


def _tetrahedral_run() -> tuple[Any, Any, Any]:
    count = 6
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        + ((0.25, 0.25, 0.25),)
    )
    cells = jnp.asarray(((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)))
    mesh = D.CellMesh(coordinates, (D.CellBlock("tet", "tetrahedron", cells),))
    element = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1))
    ).prepare()
    locator = D.PreparedSimplicialCellLocator(
        D.fem.prepare_finite_element_cell_map(element, 0),
        element.default_runtime.coordinates,
        D.SimplicialLocationPolicy(4, 8, 4),
    )
    hodge = phx.solver.maxwell.tetrahedral_maxwell_hodge(mesh.coordinates, locator.cells)
    maxwell = phx.solver.maxwell.UnstructuredMaxwellPlan(
        hodge.cochain, phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(), 100.0
    ).prepare()
    solver = phx.solver.UnstructuredMaxwellPICFieldSolver(
        maxwell, PIC.UnstructuredWhitneyCurrentPlan(locator, maximum_segments=4)
    )
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=_neutral_species(3, count),
        processes=(_merge(3, 0.0, 0.5, 1, False),),
    )
    # Whitney-0 charge is affine inside one tetrahedron, where merging to the
    # center of mass moves no nodal charge; identity-ordered packets here
    # alternate between the centroids of two tetrahedra, away from their faces.
    centroids = np.tile([[0.0625, 0.3125, 0.3125], [0.3125, 0.0625, 0.3125]], (3, 1))
    rng = np.random.default_rng(4)
    velocity = np.zeros((count, 3))
    velocity[:, 0] = 0.01
    dt = 0.5 * float(solver.maxwell.stable_dt)
    state = pic.initialize(
        (
            centroids + rng.uniform(-0.01, 0.01, (count, 3)),
            centroids + rng.uniform(-0.01, 0.01, (count, 3)),
        ),
        (velocity, np.zeros((count, 3))),
        dt,
    )
    return pic, state, dt


@pytest.mark.parametrize(
    ("build", "route"),
    [
        pytest.param(lambda: _reduced_run(1), "cochain-poisson", id="reduced-1d"),
        pytest.param(lambda: _reduced_run(2), "spectral-poisson", id="reduced-2d"),
        pytest.param(_cochain_run, "cochain-poisson", id="cochain-3d"),
        pytest.param(_tetrahedral_run, "cochain-poisson", id="tetrahedral"),
    ],
)
def test_merging_in_a_pic_step_gauss_projects_the_field(build: Any, route: str) -> None:
    pic, state, dt = build()
    result = pic.step_detailed(state, dt)
    assert result.successful
    diagnostics = result.diagnostics
    (evidence,) = diagnostics.process_evidence[0]
    assert int(evidence.events) == 2
    projection = diagnostics.gauss_projection
    assert projection.route == route
    # The merge moved charge on the grid; the projection restores Gauss.
    assert float(projection.divergence_before) > 1e-3
    assert float(projection.divergence_after) < 1e-10
    assert float(diagnostics.electric_constraint) < 1e-10
    # The next step starts from a field whose Gauss charge is the resampled
    # particles' own deposit.
    follow = pic.step_detailed(result.accepted_state, dt)
    assert follow.successful
    assert float(follow.diagnostics.particle_field_charge_defect) < 1e-10


@pytest.mark.parametrize(
    ("occupancy", "merged"), [(0.75, False), (0.5, True)], ids=["below", "at"]
)
def test_occupancy_trigger_gates_merging_of_a_crowded_cell(
    occupancy: float, merged: bool
) -> None:
    # Eight of sixteen slots are active (occupancy 0.5) in one crowded cell.
    plan = _species(16, -1.0, "electrons", 1, 0)
    active = np.arange(16) < 8
    rng = np.random.default_rng(11)
    state = _state(
        plan,
        rng.uniform(0.05, 0.45, (16, 1)),
        rng.normal(0.0, 0.5, (16, 3)),
        np.ones(16),
        active,
    )
    merge = PIC.ParticleMergePlan(
        PIC.PICCellBinningPlan((0.0,), (1.0,), (2,), (True,)),
        RELATIVITY,
        species=(0,),
        maximum_per_cell=4,
        momentum_bins=(1, 1, 1),
        minimum_packet_size=3,
        maximum_packet_size=4,
        minimum_occupancy=occupancy,
    )
    result = _apply(merge, (plan,), (state,))
    (evidence,) = result.evidence
    assert result.ledger.successful
    assert int(evidence.events) == (2 if merged else 0)
    assert int(evidence.active_after) == (4 if merged else 8)
