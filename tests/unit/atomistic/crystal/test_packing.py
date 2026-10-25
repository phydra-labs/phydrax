#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native molecular-crystal packing: contact penalty, relief, budgets and restart.

Independent controls: the overlap penalty is recomputed by an explicit NumPy
sum over every directed lattice image ``n in [-4, 4]^3`` of the unwrapped
expansion, halved, excluding only same-copy zero-shift pairs; its gradient
(force, torque through the rotation retraction, metric-chart pullback) is
compared with central finite differences of that scalar; acceptance is
re-verified on the expanded structure.  A strain-capped finite anisotropic cell
step is checked against ``sqrt(det G)`` of the chart's linear metric (exactly
the pre-step volume, or its declared compression), and acceptance is checked on
both sides of the reference-volume tolerance.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from phydrax._identity import NumericRevision
from phydrax.atomistic._generation import (
    AtomisticGenerationStopReason,
    AtomisticGenerationStreamPlan,
    AtomisticMemoryGenerationSink,
    AtomisticProposalStatus,
)
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodBundleState,
    AtomisticRoleCapacity,
)
from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitState,
    ChiralityPolicy,
    ConformerPool,
    expand_asymmetric_unit,
    metric_from_cell_parameters,
    MolecularCrystalInput,
    PreparedCrystalGroupPacker,
    PreparedCrystalPacker,
    standard_space_group_setting,
    VolumePolicy,
)
from phydrax.atomistic.crystal._packing import (
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalReliefMethod,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.metrix._quaternion_state_geometry import ScalarFirstQuaternionStateGeometry
from tests._support.crystal_geometry_molecules import (
    argon,
    chlorofluorobromomethane,
    trans_dichloroethene,
)


CONTACT = 3.0
_ROTATIONS = ScalarFirstQuaternionStateGeometry(convention="spatial")


def _resources(
    lanes: int, *, headroom: int = 1, skin: float = 0.5
) -> CrystalContactResources:
    """Contact resources: per-case capacities fixed, global allocations scaled."""
    width = 8192
    return CrystalContactResources(
        AtomisticDenseImageSearch(maximum_dense_routes=headroom * lanes * 200_000),
        search_capacities=(
            ParticleImageCapacity(
                maximum_particles_per_cell=64,
                maximum_edges=headroom * lanes * width,
                maximum_degree=width,
                maximum_images=4096,
                maximum_case_edges=width,
            ),
        ),
        role_capacities=(
            AtomisticRoleCapacity(
                maximum_routes=headroom * lanes * width, maximum_degree=width
            ),
        ),
        skin=skin,
    )


def _input(pool: ConformerPool, *, volume: float) -> MolecularCrystalInput:
    atoms = pool.component.atom_count
    return MolecularCrystalInput(
        (pool,),
        contact=np.full((atoms, atoms), CONTACT),
        volume=VolumePolicy(formula_unit_volume=volume),
    )


def _plan(
    pool: ConformerPool | None = None,
    *,
    numbers: Sequence[int] = (1,),
    volume: float = 60.0,
    relief: CrystalReliefPolicy | None = None,
    lanes: int = 1,
    chirality: ChiralityPolicy = "enantiopure",
    resources: CrystalContactResources | None = None,
) -> PreparedCrystalPacker:
    pool = trans_dichloroethene() if pool is None else pool
    settings = [standard_space_group_setting(number) for number in numbers]
    groups = (
        CrystalSpaceGroupPolicy.fixed(settings[0])
        if len(settings) == 1
        else CrystalSpaceGroupPolicy(
            settings, weights=[1.0] * len(settings), provenance="uniform test prior"
        )
    )
    plan = CrystalPackingPlan(
        _input(pool, volume=volume),
        groups,
        z_prime=1,
        chirality=chirality,
        resources=_resources(lanes) if resources is None else resources,
        cell=CrystalCellPolicy(minimum_height=2.6, minimum_height_floor=2.5),
        relief=CrystalReliefPolicy() if relief is None else relief,
    )
    return plan.prepare(lanes=lanes)


def _state(
    plan: AsymmetricUnitPlan,
    center: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    lengths: tuple[float, float, float],
    angles: tuple[float, float, float],
) -> AsymmetricUnitState:
    metric = metric_from_cell_parameters(
        jnp.asarray(lengths, dtype=jnp.float64), jnp.asarray(angles, dtype=jnp.float64)
    )
    q = np.asarray(quaternion, dtype=np.float64)
    return AsymmetricUnitState(
        jnp.asarray([center], dtype=jnp.float64),
        jnp.asarray([q / np.linalg.norm(q)], dtype=jnp.float64),
        plan.chart.coefficients(metric),
        jnp.zeros((1,), dtype=jnp.int32),
    )


def _bruteforce(plan: AsymmetricUnitPlan, state: AsymmetricUnitState) -> float:
    expansion = expand_asymmetric_unit(plan, state)
    positions = np.asarray(expansion.positions)
    cell = np.asarray(expansion.cell.vectors)
    copy = np.asarray(plan.atom_copy)
    total = 0.0
    for n in itertools.product(range(-4, 5), repeat=3):
        shift = np.asarray(n, dtype=np.float64) @ cell
        zero = not any(n)
        for i in range(positions.shape[0]):
            for j in range(positions.shape[0]):
                if zero and copy[i] == copy[j]:
                    continue
                distance = np.linalg.norm(positions[j] - positions[i] + shift)
                overlap = max(CONTACT - distance, 0.0)
                total += 0.25 * overlap * overlap
    return total


def _packer_energy(
    member: PreparedCrystalGroupPacker, state: AsymmetricUnitState
) -> tuple[
    PreparedCrystalGroupPacker,
    tuple[Array, Array, Array, Array],
    Array,
    AtomisticNeighborhoodBundleState,
]:
    lanes = (
        state.centers[None],
        state.orientations[None],
        state.cell_coefficients[None],
        state.conformer_indices[None],
    )
    _, fractional, vectors, _, certified = member._expand(*lanes)
    geometry, counts = member._geometry(fractional, vectors)
    resources = member.bundle.initialize(geometry)
    assert bool(resources.successful[0]) and bool(certified[0])
    return member, lanes, counts, resources


@pytest.mark.parametrize(
    ("center", "lengths", "angles"),
    [
        ((0.31, 0.52, 0.47), (4.6, 5.1, 4.8), (78.0, 101.0, 113.0)),
        # Molecule across a cell face: atoms stay on one whole-molecule branch.
        ((0.985, 0.02, 0.51), (5.4, 4.9, 5.6), (95.0, 84.0, 103.0)),
    ],
)
def test_overlap_penalty_counts_every_pair_image_once(
    center: tuple[float, float, float],
    lengths: tuple[float, float, float],
    angles: tuple[float, float, float],
) -> None:
    packer = _plan()
    member, unit = packer.members[0], packer.units[0]
    state = _state(unit, center, (0.8, 0.1, -0.4, 0.3), lengths, angles)
    member, lanes, counts, resources = _packer_energy(member, state)
    positions, _, vectors, _, _ = member._expand(*lanes)
    energy, _, _ = member._overlaps(positions, vectors, counts, resources)
    reference = _bruteforce(unit, state)
    assert reference > 0.0  # skew small cell: self-images overlap
    np.testing.assert_allclose(float(energy[0]), reference, rtol=1e-10, atol=1e-12)


def test_force_torque_and_cell_pullbacks_match_finite_differences() -> None:
    packer = _plan()
    member, unit = packer.members[0], packer.units[0]
    state = _state(
        unit,
        (0.4, 0.45, 0.5),
        (0.7, 0.2, -0.5, 0.4),
        (4.7, 5.0, 4.9),
        (82.0, 97.0, 108.0),
    )
    member, lanes, counts, resources = _packer_energy(member, state)
    centers, orientations, coefficients, conformers = lanes
    rotation = jnp.zeros_like(centers)
    gradients = jax.grad(
        lambda c, w, k: member._energy_terms(
            c, w, k, orientations, conformers, counts, resources
        )[0],
        argnums=(0, 1, 2),
    )(centers, rotation, coefficients)
    step = 1e-6

    def energy_of(
        center: Array | None = None,
        turn: Array | None = None,
        coefficient: Array | None = None,
    ) -> float:
        c = state.centers if center is None else center
        q = (
            state.orientations
            if turn is None
            else jax.vmap(_ROTATIONS.retract)(state.orientations, turn)
        )
        k = state.cell_coefficients if coefficient is None else coefficient
        return _bruteforce(unit, AsymmetricUnitState(c, q, k, state.conformer_indices))

    for axis in range(3):
        delta = jnp.zeros((1, 3)).at[0, axis].set(step)
        center_fd = (
            energy_of(center=state.centers + delta)
            - energy_of(center=state.centers - delta)
        ) / (2 * step)
        np.testing.assert_allclose(
            float(gradients[0][0, 0, axis]), center_fd, rtol=1e-5, atol=1e-8
        )
        torque_fd = (energy_of(turn=delta) - energy_of(turn=-delta)) / (2 * step)
        np.testing.assert_allclose(
            float(gradients[1][0, 0, axis]), torque_fd, rtol=1e-5, atol=1e-8
        )
    for index in range(unit.chart.dimension):
        delta = jnp.zeros_like(state.cell_coefficients).at[index].set(step)
        cell_fd = (
            energy_of(coefficient=state.cell_coefficients + delta)
            - energy_of(coefficient=state.cell_coefficients - delta)
        ) / (2 * step)
        np.testing.assert_allclose(
            float(gradients[2][0, index]), cell_fd, rtol=1e-5, atol=1e-8
        )


@pytest.mark.parametrize(
    ("target", "budget", "reason"), [(0, 4, "ZERO_TARGET"), (2, 0, "ZERO_BUDGET")]
)
def test_zero_work_packs_nothing(target: int, budget: int, reason: str) -> None:
    packer = _plan(pool=argon(), volume=30.0)
    result, state = packer.pack(
        jr.key(3), campaign_id="zero", target=target, attempt_budget=budget
    )
    assert result.attempts == () and result.stop_reason == reason
    assert int(state.cursor.transitions) == 0 and int(state.cursor.next_ordinal) == 0


def test_vacant_and_terminal_lanes_remain_unchanged_during_relief() -> None:
    member = _plan(lanes=3).members[0]
    vacant = member.vacant(3)
    advance = eqx.filter_jit(lambda state: member.advance(state, 2))
    untouched = advance(vacant)
    for before, after in zip(
        jax.tree.leaves(vacant), jax.tree.leaves(untouched), strict=True
    ):
        np.testing.assert_array_equal(after, before)

    state = member.admit(
        vacant,
        jnp.asarray([True, True, False]),
        jr.split(jr.key(19), 3),
        jnp.arange(3, dtype=jnp.int64),
        None,
    )
    state = eqx.tree_at(
        lambda value: value.terminal, state, state.terminal.at[1].set(True)
    )
    advanced = advance(state)
    np.testing.assert_array_equal(advanced.active, state.active)
    np.testing.assert_array_equal(advanced.terminal[1:], state.terminal[1:])
    retained = member.select_method(
        jnp.asarray([False, True, True]), advanced.method, state.method
    )
    for before, after in zip(
        jax.tree.leaves(state.method), jax.tree.leaves(retained), strict=True
    ):
        np.testing.assert_array_equal(after, before)
    assert int(advanced.method.steps[0]) > int(state.method.steps[0])
    assert np.all(np.isfinite(np.asarray(advanced.method.orientations[0])))


def test_budget_shortfall_is_explicit_and_attempts_are_retained() -> None:
    relief = CrystalReliefPolicy(
        maximum_steps=2, check_interval=1, overlap_tolerance=1e-9
    )
    packer = _plan(relief=relief, lanes=2)
    result, state = packer.pack(
        jr.key(1), campaign_id="shortfall", target=3, attempt_budget=4
    )
    assert len(result.attempts) == 4
    assert result.accepted == ()
    assert result.stop_reason == "BUDGET_EXHAUSTED" and result.shortfall == 3
    for attempt in result.attempts:
        assert attempt.status & (
            int(AtomisticProposalStatus.STEP_BUDGET_EXHAUSTED)
            | int(AtomisticProposalStatus.CELL_INVALID)
        )


def test_relief_accepts_atom_packings_within_two_sided_volume_tolerance() -> None:
    relief = CrystalReliefPolicy(
        maximum_steps=300, check_interval=5, overlap_tolerance=0.05
    )
    packer = _plan(pool=argon(), volume=32.0, relief=relief, lanes=2)
    result, _ = packer.pack(jr.key(11), campaign_id="argon", target=2, attempt_budget=6)
    assert len(result.accepted) >= 1
    for attempt in result.accepted:
        assert attempt.maximum_overlap <= 0.05
        vectors = np.asarray(attempt.expansion.cell.vectors)
        volume = abs(float(np.linalg.det(vectors)))
        assert abs(
            volume - attempt.reference_volume
        ) <= relief.volume_tolerance * attempt.reference_volume * (1.0 + 1e-12)
        shortest = min(
            np.linalg.norm(np.asarray(n) @ vectors)
            for n in itertools.product(range(-3, 4), repeat=3)
            if any(n)
        )
        assert shortest >= CONTACT - 0.05 - 1e-9


def _relief_method(
    member: PreparedCrystalGroupPacker,
    state: AsymmetricUnitState,
    reference_volume: float,
) -> CrystalReliefMethod:
    """One live relief lane at ``state`` with a declared reference volume."""
    lanes = (
        state.centers[None],
        state.orientations[None],
        state.cell_coefficients[None],
        state.conformer_indices[None],
    )
    _, fractional, vectors, _, certified = member._expand(*lanes)
    assert bool(certified[0])
    geometry, _ = member._geometry(fractional, vectors)
    nan = jnp.full((1,), jnp.nan, dtype=jnp.float64)
    return CrystalReliefMethod(
        centers=lanes[0],
        orientations=lanes[1],
        coefficients=lanes[2],
        conformers=lanes[3],
        reference_volume=jnp.asarray([reference_volume], dtype=jnp.float64),
        steps=jnp.zeros((1,), dtype=jnp.int32),
        status=jnp.zeros((1,), dtype=jnp.int32),
        accepted=jnp.zeros((1,), dtype=jnp.bool_),
        maximum_overlap=nan,
        overlap_energy=nan,
        cell_refusals=jnp.zeros((1,), dtype=jnp.int32),
        resources=member.bundle.initialize(geometry),
    )


def _metric_volume(
    unit: AsymmetricUnitPlan, coefficients: Array
) -> tuple[np.ndarray, float]:
    """Independent ``(G, sqrt(det G))`` from the chart's linear metric basis."""
    metric = np.asarray(unit.chart.metric(coefficients), dtype=np.float64)
    return metric, float(np.sqrt(np.linalg.det(metric)))


@pytest.mark.parametrize("reference_fraction", [1.0, 0.8])
def test_finite_anisotropic_cell_step_retracts_to_its_pre_step_volume(
    reference_fraction: float,
) -> None:
    # A strain cap of 0.05 makes the capped deviatoric metric step |dG| = 0.1|G|:
    # its first-order volume-preserving projection loses volume at second order
    # (``diag(d, -d, 0)`` scales V by sqrt(1 - d^2)), far above round-off.
    compression = 0.5
    relief = CrystalReliefPolicy(
        cell_step_scale=1.0e3,
        maximum_cell_strain=0.05,
        volume_compression=compression,
        overlap_tolerance=1e-9,
    )
    packer = _plan(relief=relief)
    member, unit = packer.members[0], packer.units[0]
    state = _state(
        unit,
        (0.4, 0.45, 0.5),
        (0.7, 0.2, -0.5, 0.4),
        (4.7, 5.0, 4.9),
        (82.0, 97.0, 108.0),
    )
    before_metric, before = _metric_volume(unit, state.cell_coefficients)
    reference = reference_fraction * before
    method = _relief_method(member, state, reference)
    stepped = eqx.filter_jit(lambda value: member._step(value, jnp.asarray([True])))(
        method
    )
    assert int(stepped.status[0]) == 0 and not bool(stepped.accepted[0])
    assert int(stepped.steps[0]) == 1 and int(stepped.cell_refusals[0]) == 0
    after_metric, after = _metric_volume(unit, stepped.coefficients[0])
    # Nonzero finite shear: the volume-normalized metric changed by O(strain cap).
    shape_before = before_metric / before ** (2.0 / 3.0)
    shape_after = after_metric / after ** (2.0 / 3.0)
    shear = np.linalg.norm(shape_after - shape_before) / np.linalg.norm(shape_before)
    assert shear > 1e-2
    expected = before - compression * max(before - reference, 0.0)
    np.testing.assert_allclose(after, expected, rtol=1e-10)
    assert after >= reference * (1.0 - 1e-12)


@pytest.mark.parametrize(
    ("volume_ratio", "accepted"),
    [(0.95, False), (0.99, True), (1.0, True), (1.01, True), (1.05, False)],
)
def test_relief_acceptance_requires_volume_within_both_tolerance_sides(
    volume_ratio: float, accepted: bool
) -> None:
    relief = CrystalReliefPolicy(volume_tolerance=0.02)
    packer = _plan(pool=argon(), volume=32.0, relief=relief)
    member, unit = packer.members[0], packer.units[0]
    # One argon atom in a 4 angstrom cube: nearest image 4 > contact 3, no overlap.
    state = _state(
        unit, (0.5, 0.5, 0.5), (1.0, 0.0, 0.0, 0.0), (4.0, 4.0, 4.0), (90.0, 90.0, 90.0)
    )
    _, volume = _metric_volume(unit, state.cell_coefficients)
    np.testing.assert_allclose(volume, 64.0, rtol=1e-12)
    method = _relief_method(member, state, volume / volume_ratio)
    stepped = eqx.filter_jit(lambda value: member._step(value, jnp.asarray([True])))(
        method
    )
    assert float(stepped.maximum_overlap[0]) == 0.0
    assert bool(stepped.accepted[0]) is accepted
    assert int(stepped.status[0]) == 0


_IDENTITY_RELIEF = CrystalReliefPolicy(maximum_steps=40, check_interval=5)
# Executable layouts of one packing law: (lanes, global headroom, skin).
_LAYOUTS = {"lanes": (3, 1, 0.5), "headroom": (1, 2, 0.5), "skin": (1, 1, 0.4)}


def _argon_packer(
    lanes: int = 1, headroom: int = 1, skin: float = 0.5
) -> PreparedCrystalPacker:
    return _plan(
        pool=argon(),
        volume=32.0,
        relief=_IDENTITY_RELIEF,
        lanes=lanes,
        resources=_resources(lanes, headroom=headroom, skin=skin),
    )


def _member_identities(packer: PreparedCrystalPacker) -> tuple[list[str], ...]:
    members = packer.members
    return (
        [member.sampler_id for member in members],
        [member.support.support_id for member in members],
        [member.numeric_revision().revision_id for member in members],
    )


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_executable_layout_keeps_scientific_identities_and_refuses_restore(
    layout: str, tmp_path: Path
) -> None:
    """Identity-only: no relief transition is compiled or run."""
    lanes, headroom, skin = _LAYOUTS[layout]
    reference, other = _argon_packer(), _argon_packer(lanes, headroom, skin)
    assert other.plan.law_id == reference.plan.law_id
    assert other.plan.plan_id != reference.plan.plan_id
    assert _member_identities(other) == _member_identities(reference)
    assert other.family.family_id == reference.family.family_id
    assert [m.signature_facts() for m in other.members] != [
        m.signature_facts() for m in reference.members
    ]
    runtime = reference.stream("lanes").prepare()
    candidate = other.stream("lanes").prepare()
    expected, actual = runtime.identity(), candidate.identity()
    assert actual["samplers"] == expected["samplers"]
    assert actual["revisions"] == expected["revisions"]
    assert actual["executables"] != expected["executables"]
    # Only the lane count enters the stream profile; capacity layouts are
    # distinguished by the executable binding alone.
    assert (candidate.stream_id != runtime.stream_id) == (lanes != 1)
    path = tmp_path / "layout.npz"
    runtime.write_checkpoint(path, runtime.start(jr.key(5), target=10, attempt_budget=4))
    with pytest.raises(ValueError):
        candidate.read_checkpoint(path)


def test_own_layout_checkpoint_restores_the_written_continuation(
    tmp_path: Path,
) -> None:
    runtime = _argon_packer().stream("lanes").prepare()
    state = runtime.start(jr.key(5), target=10, attempt_budget=4)
    path = tmp_path / "layout.npz"
    runtime.write_checkpoint(path, state)
    restored = runtime.read_checkpoint(path)
    assert jax.tree.structure(restored) == jax.tree.structure(state)
    assert (
        NumericRevision(runtime.plan.stream_id, restored).revision_id
        == NumericRevision(runtime.plan.stream_id, state).revision_id
    )


def test_relief_policy_change_alters_packing_and_member_laws() -> None:
    reference = _argon_packer()
    changed = _plan(
        pool=argon(),
        volume=32.0,
        relief=CrystalReliefPolicy(maximum_steps=40, check_interval=10),
    )
    assert changed.plan.law_id != reference.plan.law_id
    assert changed.members[0].sampler_id != reference.members[0].sampler_id


def test_family_slot_is_not_part_of_the_conditional_member_law(tmp_path: Path) -> None:
    alone = _plan(pool=argon(), volume=32.0, numbers=(2,))
    routed = _plan(pool=argon(), volume=32.0, numbers=(1, 2))
    # The group policies differ, so the whole packing laws and families differ.
    assert alone.plan.law_id != routed.plan.law_id
    assert alone.family.family_id != routed.family.family_id
    # The P-1 member keeps its law at another family slot.
    member, slotted = alone.members[0], routed.members[1]
    assert (member.group_index, slotted.group_index) == (0, 1)
    assert member.sampler_id == slotted.sampler_id
    assert member.support.support_id == slotted.support.support_id
    source = AtomisticGenerationStreamPlan(
        slotted,
        campaign_id="member-slot",
        lanes_per_member=1,
        pending_capacity=1,
    ).prepare()
    destination = AtomisticGenerationStreamPlan(
        member,
        campaign_id="member-slot",
        lanes_per_member=1,
        pending_capacity=1,
    ).prepare()
    assert source.stream_id == destination.stream_id
    assert source.identity()["executables"] != destination.identity()["executables"]
    path = tmp_path / "member-slot.npz"
    source.write_checkpoint(path, source.start(jr.key(5), target=10, attempt_budget=4))
    with pytest.raises(ValueError):
        destination.read_checkpoint(path)


def test_results_do_not_depend_on_lane_count_or_global_allocations() -> None:
    relief = CrystalReliefPolicy(maximum_steps=40, check_interval=5)
    runs = []
    for lanes, headroom in ((1, 1), (3, 1), (1, 2)):
        packer = _plan(
            pool=argon(),
            volume=32.0,
            relief=relief,
            lanes=lanes,
            resources=_resources(lanes, headroom=headroom),
        )
        result, _ = packer.pack(
            jr.key(5), campaign_id="lanes", target=10, attempt_budget=4
        )
        runs.append(result.attempts)
    reference = runs[0]
    assert [item.ordinal for item in reference] == [0, 1, 2, 3]
    for run in runs[1:]:
        for left, right in zip(reference, run, strict=True):
            assert left.ordinal == right.ordinal
            assert left.proposal_id == right.proposal_id
            assert left.status == right.status
            np.testing.assert_allclose(
                np.asarray(left.expansion.positions),
                np.asarray(right.expansion.positions),
                rtol=1e-9,
                atol=1e-9,
            )


def test_interrupted_packing_resumes_exactly(tmp_path: Path) -> None:
    relief = CrystalReliefPolicy(maximum_steps=30, check_interval=5)
    packer = _plan(relief=relief, lanes=2)
    reference_sink = AtomisticMemoryGenerationSink("reference")
    reference, _ = packer.pack(
        jr.key(2),
        campaign_id="restart",
        target=10,
        attempt_budget=5,
        sink=reference_sink,
    )
    runtime = packer.stream("restart").prepare()
    state = runtime.start(jr.key(2), target=10, attempt_budget=5)
    sink = AtomisticMemoryGenerationSink("restart")
    state = runtime.run(state, sink, maximum_transitions=2)
    path = tmp_path / "packing.npz"
    runtime.write_checkpoint(path, state)
    resumed = runtime.run(runtime.read_checkpoint(path), sink, maximum_transitions=200)
    assert resumed.stop_reason is AtomisticGenerationStopReason.BUDGET_EXHAUSTED
    assert [item.proposal_id for item in sink.records] == [
        item.proposal_id for item in reference.attempts
    ]
    assert (
        NumericRevision(runtime.plan.stream_id, sink.records).revision_id
        == NumericRevision(runtime.plan.stream_id, reference_sink.records).revision_id
    )
    # PackedCrystal re-expands its ASU payload. Compare that same observation
    # on both paths, not eager re-expansion against compiled emission rounding.
    for item, attempt in zip(sink.records, reference.attempts, strict=True):
        np.testing.assert_array_equal(
            np.asarray(packer.packed(item).expansion.positions),
            np.asarray(attempt.expansion.positions),
        )


def test_weighted_groups_and_chirality_admission() -> None:
    packer = _plan(pool=argon(), volume=32.0, numbers=(1, 2))
    family = packer.family
    groups = np.asarray(family.route.members(jr.key(4), jnp.arange(32, dtype=jnp.int64)))
    assert set(groups.tolist()) == {0, 1}
    assert [unit.copy_count for unit in packer.units] == [1, 2]
    with pytest.raises(ValueError, match="enantiomer"):
        _plan(pool=chlorofluorobromomethane(), numbers=(2,))
    policy = CrystalSpaceGroupPolicy(
        [standard_space_group_setting(number) for number in (1, 2, 4)],
        weights=(1.0, 2.0, 3.0),
        provenance="uniform test prior",
    )
    sohncke = policy.filter(sohncke_only=True)
    assert [setting.international_number for setting in sohncke.settings] == [1, 4]
    np.testing.assert_allclose(sohncke.weights, (0.25, 0.75))
    assert [s.international_number for s in policy.filter(z=2).settings] == [2, 4]


def test_store_records_keep_asu_and_expansion_provenance(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    from phydrax.atomistic._generation import AtomisticStructureStoreGenerationSink
    from phydrax.atomistic._structure_store import (
        AtomisticStoreLayout,
        AtomisticStoreLimits,
        AtomisticStructureStoreWriter,
        open_atomistic_structure_store,
    )
    from phydrax.atomistic.crystal._packing import crystal_structure_schema

    limits = AtomisticStoreLimits(
        max_commit_bytes=1 << 20,
        max_member_bytes=1 << 22,
        max_shard_bytes=1 << 24,
        max_shard_members=8192,
        max_chunk_bytes=1 << 20,
        max_index_bytes=1 << 20,
        max_decoded_bytes=1 << 24,
        max_read_bytes=1 << 24,
        max_records=100,
    )
    writer = AtomisticStructureStoreWriter(
        tmp_path / "crystals",
        layout=AtomisticStoreLayout(
            records_per_shard=4, target_chunk_bytes=4096, compression="none"
        ),
        limits=limits,
    )
    writer.create(crystal_structure_schema(), name="packed-crystals")
    relief = CrystalReliefPolicy(maximum_steps=10, check_interval=5)
    packer = _plan(relief=relief, numbers=(2,), lanes=2)
    sink = AtomisticStructureStoreGenerationSink(
        tmp_path / "crystals", writer, packer.record_builder, limits=limits
    )
    result, _ = packer.pack(
        jr.key(9), campaign_id="store", target=10, attempt_budget=3, sink=sink
    )
    source = open_atomistic_structure_store(tmp_path / "crystals", limits=limits)
    assert source.record_ids == tuple(item.proposal_id for item in result.attempts)
    batch = source.read(np.arange(3))
    plan = result.attempts[0].plan
    np.testing.assert_array_equal(
        batch.values["source_copy"][: plan.atom_count], plan.atom_copy
    )
    np.testing.assert_array_equal(
        batch.values["source_operation"][: plan.atom_count], plan.atom_operation
    )
    np.testing.assert_array_equal(
        batch.values["source_enantiomer"][: plan.atom_count],
        plan.copy_enantiomer[plan.atom_copy].astype(np.int8),
    )
    np.testing.assert_array_equal(batch.values["z"], [2, 2, 2])
    np.testing.assert_allclose(
        batch.values["positions"][: plan.atom_count],
        np.asarray(result.attempts[0].expansion.positions),
    )
    assert batch.lineage_ids == (packer.plan.law_id,) * 3
