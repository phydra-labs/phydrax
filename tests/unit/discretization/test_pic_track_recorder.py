#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Identity-addressed PIC track recorder.

References are independent of the recorder: expected tracks are rebuilt from
the raw species states by NumPy identity lookup, with the leapfrog convention
that a state holds ``x^k`` and ``u^{k−1/2}`` so sample ``k`` carries
``(u^{k−1/2} + u^{k+1/2}) / 2`` (Birdsall and Langdon, *Plasma Physics via
Computer Simulation*, §4-3). Streamed radiation is checked against the offline
trajectory-radiation evaluation of the stored tracks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax import ElectromagneticScaleContract
from phydrax.discretization import ParticleAllocationRequest
from phydrax.discretization.pic import (
    ExternalFieldSample,
    PIC_CODE_RELATIVITY,
    PICParticleState,
    PICSpeciesPlan,
    PICSpeciesState,
    PICTrackRecorder,
    PICTrackRecorderState,
)
from phydrax.electromagnetics import (
    PreparedTrajectoryRadiation,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
    TrajectoryRadiationRoute,
)
from phydrax.units import CHARGE, UnitDefinition


def _code_scale(speed_of_light: int = 1) -> ElectromagneticScaleContract:
    return ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=speed_of_light,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="pic-track-recorder-test",
    )


def _species_plan(
    count: int, dimension: int, specific: float, name: str, *, maximum_charge: int = 1
) -> PICSpeciesPlan:
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=dimension
    ).prepare()
    return PICSpeciesPlan(
        phx.discretization.ParticlePopulationPlan(support),
        phx.discretization.pic.PICChargeModelPlan(
            specific,
            name,
            minimum_charge_number=1,
            maximum_charge_number=maximum_charge,
            initial_charge_number=1,
        ),
    )


def _with_kinematics(
    state: PICSpeciesState, position: np.ndarray, velocity: np.ndarray
) -> PICSpeciesState:
    return PICSpeciesState(
        PICParticleState(jnp.asarray(position), jnp.asarray(velocity)),
        state.population,
        state.charge,
    )


def _identities(*words: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([hi for hi, _ in words], dtype=np.uint32),
        np.asarray([lo for _, lo in words], dtype=np.uint32),
    )


def _run(
    recorder: PICTrackRecorder,
    states: Sequence[tuple[PICSpeciesState, ...]],
    times: np.ndarray,
) -> PICTrackRecorderState:
    record = recorder.initialize(states[0], jnp.asarray(times[0]))
    # Compiled like a PIC step: one trace serves every accepted step.
    advance = eqx.filter_jit(recorder.record)
    for step, species in enumerate(states[1:], start=1):
        record = advance(
            record, species, jnp.asarray(times[step]), jnp.asarray(step, jnp.int32)
        )
    return record


def _expected_lane(
    states: Sequence[tuple[PICSpeciesState, ...]],
    species_index: int,
    identity: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples ``k < len(states) − 1`` of one identity by NumPy slot lookup."""

    def find(species: tuple[PICSpeciesState, ...]) -> tuple[bool, np.ndarray, np.ndarray]:
        population = species[species_index].population
        hits = np.flatnonzero(
            np.asarray(population.active)
            & (np.asarray(population.id_hi) == identity[0])
            & (np.asarray(population.id_lo) == identity[1])
        )
        if hits.size == 0:
            return False, np.zeros(3), np.zeros(3)
        particles = species[species_index].particles
        position = np.zeros(3)
        resolved = np.asarray(particles.position[hits[0]], dtype=np.float64)
        position[: resolved.size] = resolved
        return True, position, np.asarray(particles.proper_velocity[hits[0]], np.float64)

    found = [find(value) for value in states]
    active, positions, velocities = [], [], []
    for (here, x, u_back), (there, _, u_forward) in zip(found[:-1], found[1:]):
        active.append(here)
        positions.append(x)
        velocities.append(0.5 * (u_back + u_forward) if here and there else u_back)
    return np.asarray(active), np.asarray(positions), np.asarray(velocities)


def _synthetic_run(
    plan: PICSpeciesPlan, count: int, steps: int, dimension: int
) -> tuple[list[tuple[PICSpeciesState, ...]], np.ndarray]:
    rng = np.random.default_rng(7)
    base = plan.initialize(np.zeros((count, dimension)), np.zeros((count, 3)))
    times = 0.1 * np.arange(steps + 1, dtype=np.float64)
    states = []
    for _ in times:
        states.append(
            (
                _with_kinematics(
                    base,
                    rng.uniform(-1.0, 1.0, (count, dimension)),
                    rng.uniform(-0.5, 0.5, (count, 3)),
                ),
            )
        )
    return states, times


def _permuted(state: PICSpeciesState, permutation: np.ndarray) -> PICSpeciesState:
    return jax.tree.map(lambda leaf: leaf[permutation] if leaf.ndim else leaf, state)


def test_recorder_samples_time_centered_tracks_of_identities() -> None:
    plan = _species_plan(4, 3, -1.0, "electrons")
    states, times = _synthetic_run(plan, 4, 6, 3)
    lanes = _identities((0, 2), (0, 0), (0, 9))
    recorder = PICTrackRecorder(
        (plan,), [0, 0, 0], lanes, relativity=PIC_CODE_RELATIVITY, sample_capacity=6
    )
    record = _run(recorder, states, times)
    trajectory = recorder.to_charged_trajectory(record, _code_scale())

    np.testing.assert_array_equal(trajectory.times[:, 0], times[:-1])
    for lane, identity in enumerate(((0, 2), (0, 0), (0, 9))):
        active, positions, velocities = _expected_lane(states, 0, identity)
        np.testing.assert_array_equal(trajectory.active[:, lane], active)
        np.testing.assert_allclose(
            np.asarray(trajectory.positions[:, lane])[active], positions[active], 0, 1e-15
        )
        np.testing.assert_allclose(
            np.asarray(trajectory.proper_velocities[:, lane])[active],
            velocities[active],
            rtol=0,
            atol=1e-15,
        )
    np.testing.assert_array_equal(trajectory.id_lo, [2, 0, 9])
    # Unit-mass electrons: charge −e of multiplicity one reproduces the macrocharge.
    np.testing.assert_array_equal(trajectory.charges, [-1.0, -1.0, 0.0])
    np.testing.assert_array_equal(trajectory.multiplicities, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(record.seen, [True, True, False])


def test_identity_tracks_are_invariant_under_slot_migration() -> None:
    plan = _species_plan(5, 3, 1.0, "ions")
    states, times = _synthetic_run(plan, 5, 5, 3)
    rng = np.random.default_rng(3)
    permutations = [rng.permutation(5) for _ in states]
    migrated = [
        (_permuted(value[0], permutation),)
        for value, permutation in zip(states, permutations, strict=True)
    ]
    lanes = _identities((0, 4), (0, 1))
    radiation = _radiation(_code_scale(), "segment-exact")
    recorder = PICTrackRecorder(
        (plan,),
        [0, 0],
        lanes,
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=5,
        radiation=radiation,
    )
    reference = _run(recorder, states, times)
    moved = _run(recorder, migrated, times)

    assert reference.buffer is not None and moved.buffer is not None
    np.testing.assert_array_equal(moved.buffer.positions, reference.buffer.positions)
    np.testing.assert_array_equal(
        moved.buffer.proper_velocities, reference.buffer.proper_velocities
    )
    np.testing.assert_array_equal(moved.buffer.active, reference.buffer.active)
    # Slots follow the migration: the slot holding identity i is inverse[i].
    for step in range(5):
        inverse = np.argsort(permutations[step])
        np.testing.assert_array_equal(moved.buffer.slots[step], inverse[[4, 1]])
    np.testing.assert_array_equal(moved.activations, [1, 1])
    streamed = recorder.finalize_radiation(moved)
    np.testing.assert_array_equal(
        streamed.field_spectrum, recorder.finalize_radiation(reference).field_spectrum
    )
    offline = radiation.evaluate(recorder.to_charged_trajectory(moved, _code_scale()))
    np.testing.assert_allclose(
        streamed.field_spectrum, offline.field_spectrum, rtol=1e-10, atol=0
    )


def test_slot_reuse_ends_the_old_track_and_births_the_new_identity() -> None:
    plan = _species_plan(3, 3, -1.0, "electrons")
    population_plan = plan.population
    initial = plan.initialize(
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        active_mask=np.asarray([True, True, False]),
        masses=np.asarray([1.0, 1.0, 0.0]),
    )
    velocity = np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    killed = population_plan.deactivate(
        initial.population, np.asarray([True, False, False])
    ).accepted_state
    reused = population_plan.allocate(
        killed,
        ParticleAllocationRequest(
            np.asarray([0]),
            np.asarray([2.0]),
            np.asarray([True]),
            parents=(np.asarray([0], np.uint32), np.asarray([0], np.uint32)),
        ),
    ).accepted_state
    populations = [initial.population] * 2 + [killed] + [reused] * 2
    states = [
        (
            PICSpeciesState(
                PICParticleState(jnp.full((3, 3), 0.1 * step), jnp.asarray(velocity)),
                population,
                initial.charge,
            ),
        )
        for step, population in enumerate(populations)
    ]
    times = 0.5 * np.arange(len(states), dtype=np.float64)
    # Lanes: the particle that dies, the identity born into its slot, and one that
    # is never created.
    lanes = _identities((0, 0), (0, 2), (0, 7))
    recorder = PICTrackRecorder(
        (plan,), [0, 0, 0], lanes, relativity=PIC_CODE_RELATIVITY, sample_capacity=4
    )
    record = _run(recorder, states, times)
    assert record.buffer is not None

    np.testing.assert_array_equal(
        record.buffer.active,
        [[True, False, False], [True, False, False], [False, False, False]]
        + [[False, True, False]],
    )
    np.testing.assert_array_equal(record.buffer.slots[:, 1], [-1, -1, -1, 0])
    np.testing.assert_array_equal(record.buffer.incarnations[3], [0, 2, 0])
    # The dying particle's last sample keeps its own half-step velocity.
    np.testing.assert_array_equal(record.buffer.proper_velocities[1, 0], velocity[0])
    np.testing.assert_array_equal(record.first_step, [0, 3, -1])
    np.testing.assert_array_equal(record.last_step, [1, 4, -1])
    np.testing.assert_array_equal(record.parent_lo, [0xFFFFFFFF, 0, 0xFFFFFFFF])
    np.testing.assert_array_equal(record.mass, [1.0, 2.0, 0.0])
    trajectory = recorder.to_charged_trajectory(record, _code_scale())
    np.testing.assert_array_equal(trajectory.multiplicities, [1.0, 2.0, 0.0])


def test_charge_transition_ends_the_lane_with_evidence() -> None:
    plan = _species_plan(2, 3, 1.0, "ions", maximum_charge=2)
    states, times = _synthetic_run(plan, 2, 4, 3)
    ionized = plan.charge_model.transition(
        states[2][0].population, states[2][0].charge, np.asarray([1, 0]), 2
    ).candidate_state
    states = states[:2] + [
        (PICSpeciesState(value[0].particles, value[0].population, ionized),)
        for value in states[2:]
    ]
    recorder = PICTrackRecorder(
        (plan,),
        [0, 0],
        _identities((0, 0), (0, 1)),
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=4,
    )
    record = _run(recorder, states, times)
    assert record.buffer is not None

    np.testing.assert_array_equal(record.buffer.active[:, 0], [True, True, False, False])
    np.testing.assert_array_equal(record.buffer.active[:, 1], [True] * 4)
    np.testing.assert_array_equal(record.property_transition, [True, False])
    np.testing.assert_array_equal(record.charge_number, [1, 1])


def test_reduced_geometry_drifts_unresolved_coordinates() -> None:
    plan = _species_plan(2, 1, -1.0, "electrons")
    states, times = _synthetic_run(plan, 2, 5, 1)
    recorder = PICTrackRecorder(
        (plan,),
        [0],
        _identities((0, 1)),
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=5,
    )
    record = _run(recorder, states, times)
    assert record.buffer is not None

    velocities = np.stack(
        [np.asarray(value[0].particles.proper_velocity[1]) for value in states]
    )
    drift = velocities / np.sqrt(1.0 + np.sum(velocities**2, axis=-1, keepdims=True))
    expected = np.zeros((5, 2))
    for step in range(1, 5):
        expected[step] = expected[step - 1] + 0.1 * drift[step, 1:]
    np.testing.assert_allclose(record.buffer.positions[:, 0, 1:], expected, 0, 1e-15)
    np.testing.assert_array_equal(
        record.buffer.positions[:, 0, 0],
        [np.asarray(value[0].particles.position[1, 0]) for value in states[:-1]],
    )


def test_ring_overflow_is_refused_or_keeps_the_latest_samples() -> None:
    plan = _species_plan(2, 3, -1.0, "electrons")
    states, times = _synthetic_run(plan, 2, 5, 3)
    lanes = _identities((0, 0))
    latest = PICTrackRecorder(
        (plan,),
        [0],
        lanes,
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=3,
        overflow="keep-latest",
    )
    record = _run(latest, states, times)
    trajectory = latest.to_charged_trajectory(record, _code_scale())
    _, positions, _ = _expected_lane(states, 0, (0, 0))

    assert int(latest.dropped_samples(record)) == 2
    np.testing.assert_array_equal(trajectory.times[:, 0], times[2:5])
    np.testing.assert_array_equal(trajectory.positions[:, 0], positions[2:5])

    refusing = PICTrackRecorder(
        (plan,), [0], lanes, relativity=PIC_CODE_RELATIVITY, sample_capacity=3
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="ring overflowed"):
        refusing.to_charged_trajectory(_run(refusing, states, times), _code_scale())

    roomy = PICTrackRecorder(
        (plan,), [0], lanes, relativity=PIC_CODE_RELATIVITY, sample_capacity=8
    )
    padded = roomy.to_charged_trajectory(_run(roomy, states, times), _code_scale())
    np.testing.assert_array_equal(padded.active[5:, 0], [False] * 3)
    np.testing.assert_array_equal(padded.times[5:, 0], [times[4]] * 3)


@pytest.mark.parametrize(
    ("lanes", "keywords", "error"),
    [
        (_identities((0, 1), (0, 1)), {"sample_capacity": 2}, "distinct"),
        (
            _identities((0xFFFFFFFF, 0xFFFFFFFF)),
            {"sample_capacity": 2},
            "reserved",
        ),
        (_identities((0, 1)), {}, "sample_capacity, radiation"),
        (_identities((0, 1)), {"sample_capacity": 2, "overflow": "wrap"}, "overflow"),
    ],
    ids=["duplicate", "reserved", "nothing-recorded", "overflow-selector"],
)
def test_recorder_refuses_invalid_declarations(
    lanes: tuple[np.ndarray, np.ndarray], keywords: dict[str, Any], error: str
) -> None:
    plan = _species_plan(2, 3, -1.0, "electrons")
    with pytest.raises(ValueError, match=error):
        PICTrackRecorder(
            (plan,),
            np.zeros(lanes[0].shape, dtype=np.int32),
            lanes,
            relativity=PIC_CODE_RELATIVITY,
            **keywords,
        )


def _radiation(
    scale: ElectromagneticScaleContract, route: TrajectoryRadiationRoute
) -> PreparedTrajectoryRadiation:
    observers = RadiationObserverPlan(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]]),
        np.asarray([0.0, 1.0, 0.0]),
    )
    return TrajectoryRadiationPlan(
        scale,
        observers,
        np.asarray([5.0, 20.0, 60.0, 150.0]),
        coherence="incoherent",
        route=route,
        emission="truncated",
    ).prepare()


def test_radiation_streaming_refuses_mismatched_scale_and_hermite_route() -> None:
    plan = _species_plan(2, 3, -1.0, "electrons")
    lanes = _identities((0, 1))
    with pytest.raises(ValueError, match="speed of light"):
        PICTrackRecorder(
            (plan,),
            [0],
            lanes,
            relativity=PIC_CODE_RELATIVITY,
            radiation=_radiation(_code_scale(2), "segment-exact"),
        )
    with pytest.raises(ValueError, match="accelerations"):
        PICTrackRecorder(
            (plan,),
            [0],
            lanes,
            relativity=PIC_CODE_RELATIVITY,
            radiation=_radiation(_code_scale(), "segment-hermite"),
        )


class _UniformMagneticField(phx.StrictModule):
    magnetic: Array

    @property
    def source_id(self) -> str:
        return "uniform-axial-test-field"

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        del times
        return ExternalFieldSample(
            jnp.zeros_like(positions),
            jnp.broadcast_to(self.magnetic, positions.shape),
            jnp.ones(positions.shape[:1], dtype=jnp.bool_),
        )


def _gyrating_pic(
    recorders: Sequence[PICTrackRecorder], species: tuple[PICSpeciesPlan, ...]
) -> tuple[Any, Any]:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    transfer_plan = phx.discretization.pic.PICParticleCochainTransferPlan(bridge)
    transfers = []
    for plan in species:
        charged = phx.discretization.ChargedParticlePlan(
            plan.charge_model.base_specific_charge * jnp.ones((4,)), plan.species_id
        ).prepare(plan.population.particles)
        transfers.append(transfer_plan.prepare(charged))
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="track-recorder-maxwell",
    ).prepare()
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        phx.solver.CochainElectrostaticPlan(
            bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        tuple(transfers),
        tuple(
            phx.discretization.pic.ChargeConservingCurrentPlan(value)
            for value in transfers
        ),
    )
    pic = phx.solver.ElectromagneticPICPlan(
        solver,
        species=species,
        recorders=recorders,
        ownership="resolved-field",
        external_fields=(_UniformMagneticField(jnp.asarray([0.0, 0.0, 40.0])),),
    )
    return pic, maxwell


def _gyrating_species() -> tuple[PICSpeciesPlan, ...]:
    species = []
    for offset, sign, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        support = phx.discretization.ParticleSetPlan(
            jnp.arange(offset, offset + 4), jnp.ones((4,)), ambient_dimension=3
        ).prepare()
        species.append(
            PICSpeciesPlan(
                phx.discretization.ParticlePopulationPlan(support),
                phx.discretization.pic.PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    return tuple(species)


_LANE_SPECIES = [0, 1, 0, 0]
_LANE_IDENTITIES = ((0, 0), (0, 3), (0, 2), (0, 40))


def _gyrating_run(
    recorders: Sequence[PICTrackRecorder], species: tuple[PICSpeciesPlan, ...], steps: int
) -> tuple[Any, list[Any], Any]:
    pic, maxwell = _gyrating_pic(recorders, species)
    position = jnp.asarray(
        [[0.20, 0.20, 0.20], [0.35, 0.45, 0.55], [0.60, 0.30, 0.70], [0.80, 0.75, 0.40]]
    )
    velocity = jnp.asarray(
        [[0.5, 0.0, 0.1], [0.0, 0.4, 0.0], [0.3, 0.3, 0.0], [0.0, 0.0, 0.2]]
    )
    dt = 0.05 * maxwell.stable_dt
    advance = pic.step_detailed
    state = pic.initialize((position, position + 0.01), (velocity, 0.1 * velocity), dt)
    history = [state]
    for _ in range(steps):
        result = advance(state, dt)
        assert bool(result.successful)
        state = result.accepted_state
        history.append(state)
    return pic, history, (advance, dt)


def test_recorder_inside_pic_matches_offline_tracks_and_streamed_radiation() -> None:
    species = _gyrating_species()
    scale = _code_scale()
    lanes = _identities(*_LANE_IDENTITIES)
    steps = 8
    tracks = PICTrackRecorder(
        species,
        _LANE_SPECIES,
        lanes,
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=steps,
    )
    radiation = _radiation(scale, "segment-exact")
    # Streaming only: no stored tracks, the spectrum accumulates inside the run.
    streaming = PICTrackRecorder(
        species, _LANE_SPECIES, lanes, relativity=PIC_CODE_RELATIVITY, radiation=radiation
    )
    _, history, _ = _gyrating_run((tracks, streaming), species, steps)
    final = history[-1]

    trajectory = tracks.to_charged_trajectory(final.recorders[0], scale)
    np.testing.assert_array_equal(
        trajectory.times[:, 0], [float(value.time) for value in history[:steps]]
    )
    for lane, (index, identity) in enumerate(
        zip(_LANE_SPECIES, _LANE_IDENTITIES, strict=True)
    ):
        active, positions, velocities = _expected_lane(
            [value.species for value in history], index, identity
        )
        np.testing.assert_array_equal(trajectory.active[:, lane], active)
        np.testing.assert_allclose(
            trajectory.positions[:, lane][active], positions[active], rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            trajectory.proper_velocities[:, lane][active],
            velocities[active],
            rtol=0,
            atol=1e-15,
        )
    np.testing.assert_array_equal(trajectory.active[:, 3], [False] * steps)
    np.testing.assert_array_equal(
        trajectory.charges * trajectory.multiplicities, [-1.0, 1.0, -1.0, 0.0]
    )

    offline = radiation.evaluate(trajectory)
    streamed = streaming.finalize_radiation(final.recorders[1])
    assert float(jnp.max(offline.spectral_energy)) > 0.0
    np.testing.assert_allclose(
        streamed.field_spectrum, offline.field_spectrum, rtol=1e-10, atol=0
    )
    np.testing.assert_allclose(
        streamed.spectral_energy, offline.spectral_energy, rtol=1e-10, atol=0
    )
    assert int(streamed.evidence.status) == int(offline.evidence.status)
    assert int(streamed.evidence.segments_used) == int(offline.evidence.segments_used)


def test_recorder_state_round_trips_through_pic_restart() -> None:
    species = _gyrating_species()
    lanes = _identities(*_LANE_IDENTITIES)
    recorders = (
        PICTrackRecorder(
            species,
            _LANE_SPECIES,
            lanes,
            relativity=PIC_CODE_RELATIVITY,
            sample_capacity=4,
        ),
        PICTrackRecorder(
            species,
            _LANE_SPECIES,
            lanes,
            relativity=PIC_CODE_RELATIVITY,
            radiation=_radiation(_code_scale(), "segment-exact"),
        ),
    )
    pic, history, (advance, dt) = _gyrating_run(recorders, species, 3)
    restored = pic.restore(pic.checkpoint(history[-1]))
    for value, expected in zip(restored.recorders, history[-1].recorders, strict=True):
        jax.tree.map(np.testing.assert_array_equal, value, expected)

    continued, resumed = history[-1], restored
    for _ in range(2):
        continued = advance(continued, dt).accepted_state
        resumed = advance(resumed, dt).accepted_state
    for value, expected in zip(resumed.recorders, continued.recorders, strict=True):
        jax.tree.map(np.testing.assert_array_equal, value, expected)
    assert int(recorders[0].dropped_samples(resumed.recorders[0])) == 1


def _reduced_pic(
    recorders: Sequence[PICTrackRecorder], species: tuple[PICSpeciesPlan, ...]
) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    solver = phx.solver.ReducedMaxwellPICFieldSolver(
        phx.solver.CompatibleMaxwell1DPlan(grid),
        phx.discretization.pic.ReducedPICTransferPlan(grid),
    )
    return phx.solver.ElectromagneticPICPlan(
        solver, species=species, recorders=recorders
    ), solver


def _reduced_pair() -> tuple[PICSpeciesPlan, ...]:
    return (
        _species_plan(2, 1, -1.0, "electrons"),
        _species_plan(2, 1, 1.0, "ions"),
    )


def test_pic_plan_refuses_recorders_of_other_species_or_units() -> None:
    species = _reduced_pair()
    lanes = _identities((0, 0))
    other_units = phx.RelativityScaleContract(
        PIC_CODE_RELATIVITY.dimensional_scale,
        1,
        2,
        1,
        1,
        quantum_constants_explicit=False,
    )
    with pytest.raises(ValueError, match="relativity scale differs"):
        _reduced_pic(
            (
                PICTrackRecorder(
                    species, [0], lanes, relativity=other_units, sample_capacity=2
                ),
            ),
            species,
        )
    with pytest.raises(ValueError, match="other species plans"):
        _reduced_pic(
            (
                PICTrackRecorder(
                    (_species_plan(3, 1, -1.0, "electrons"), species[1]),
                    [0],
                    lanes,
                    relativity=PIC_CODE_RELATIVITY,
                    sample_capacity=2,
                ),
            ),
            species,
        )


def test_moving_window_tracks_stay_in_the_lab_frame() -> None:
    species = _reduced_pair()
    scale = _code_scale()
    radiation = _radiation(scale, "segment-exact")
    steps = 12
    recorder = PICTrackRecorder(
        species,
        [0],
        _identities((0, 0)),
        relativity=PIC_CODE_RELATIVITY,
        sample_capacity=steps,
        radiation=radiation,
    )
    pic, solver = _reduced_pic((recorder,), species)
    window = phx.solver.PICMovingWindowPlan(pic, 0)
    dt = 0.5 * solver.field.stable_dt
    # A co-located electron–ion pair deposits no net current: it drifts uniformly
    # at β = 0.6 while the window moves one cell every third step.
    position = jnp.asarray([[0.5], [0.8]])
    velocity = jnp.asarray([[0.6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    state = window.initialize(
        pic.initialize((position, position), (velocity, velocity), dt)
    )
    for step in range(steps):
        result = pic.step_detailed(state.pic, dt)
        assert bool(result.successful)
        state = dataclasses.replace(state, pic=result.accepted_state)
        if step % 3 == 2:
            shifted = window.shift(state)
            assert bool(shifted.shifted)
            state = shifted.accepted_state
    assert int(state.cumulative_cells) == 4

    record = state.pic.recorders[0]
    trajectory = recorder.to_charged_trajectory(record, scale)
    times = np.asarray(trajectory.times[:, 0])
    np.testing.assert_allclose(
        trajectory.positions[:, 0, 0], 0.5 + 0.6 * times, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(record.frame_offset, [4.0 / 16.0, 0.0, 0.0])
    # Uniform motion has no far field: the velocity-form sum telescopes against
    # the window-edge terms. The scale is one edge term, |q| β_⊥ / (4π ε₀ c κ),
    # at the most oblique observer n = (0.6, 0, 0.8): β_⊥ = 0.48, κ = 1 − 0.36.
    streamed = recorder.finalize_radiation(record)
    edge = 0.48 / (4.0 * np.pi * 0.64)
    assert float(jnp.max(jnp.abs(streamed.field_spectrum))) <= 1e-12 * edge
