"""Enhanced-sampling integration scenarios with independent physical controls.

Each test is one separately collected scientific scenario: a periodic volume
umbrella with MTK cell response, a well-tempered history, fixed-correspondence
RMSD exploration, pair-distance ABF, umbrella-window exchange, shared/walker/
state history ownership, freeze-to-analysis and fresh-process restart.
Reduced units throughout (``k_B = 1``).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._jax_config import default_device
from phydrax.atomistic._cell_dynamics import MTKBarostatPlan, MTKNPTPlan
from phydrax.atomistic._classical import HarmonicBondPotential
from phydrax.atomistic._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticDynamicsState,
    PreparedAtomisticDynamics,
    VelocityVerletPlan,
)
from phydrax.atomistic._term_state import (
    AbstractAtomisticTermState,
    AtomisticTermStateBindings,
)
from phydrax.atomistic._thermal import BAOABLangevinPlan, NoseHooverChainParameters
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.atomistic.sampling import (
    abf_profile,
    AtomisticABFForcePort,
    AtomisticABFPlan,
    AtomisticBiasHistoryPlan,
    AtomisticBiasHistoryState,
    AtomisticBiasRowBinding,
    AtomisticBiasTerm,
    AtomisticCanonicalSamplingQualification,
    AtomisticMultistateCheckpointPlan,
    AtomisticMultistateIteration,
    AtomisticMultistatePlan,
    AtomisticMultistateSegmentPlan,
    AtomisticMultistateState,
    AtomisticReplicaExchangePlan,
    AtomisticThermodynamicStatePlan,
    BiasHistoryLabel,
    BiasHistoryOwnership,
    CollectiveVariableGeometry,
    CollectiveVariableKind,
    FrozenBiasHistory,
    HarmonicBiasForm,
    HillHistoryBiasForm,
    HillKernel,
    IntegratedProfileBiasForm,
    PreparedAtomisticBiasTerm,
    PreparedAtomisticMultistate,
    PreparedBiasedDynamics,
    PreparedThermodynamicStateTable,
    with_bias_history,
    write_atomistic_multistate_checkpoint,
    write_biased_dynamics_checkpoint,
)
from phydrax.atomistic.sampling._bias_history import bias_history_digest
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    measure,
    periodic_cell,
    prepare_potential,
    row,
    step_detailed,
    table,
)
from tests._support.atomistic_sampling_fixtures import (
    bias_term,
    biased_dynamics,
    cv_program,
    kabsch_msd,
    nve_table,
    start,
    zero_field,
)
from tests.integration.atomistic import _enhanced_sampling_restart_worker as worker


REPOSITORY = Path(__file__).resolve().parents[3]
WORKER = Path(worker.__file__).resolve()


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


@eqx.filter_jit
def _iterate(
    runtime: PreparedAtomisticMultistate, state: AtomisticMultistateState
) -> AtomisticMultistateIteration:
    return runtime.iterate(state)


def _qualified_multistate(
    dynamics: PreparedAtomisticDynamics,
    thermodynamic: PreparedThermodynamicStateTable,
    run_id: str,
    /,
    *,
    history_label: BiasHistoryLabel = BiasHistoryLabel.ORIGIN,
) -> PreparedAtomisticMultistate:
    qualification = AtomisticCanonicalSamplingQualification(
        dynamics,
        thermodynamic,
        f"{run_id}-qualification",
        sampling_exact=False,
        sampling_bias_bound=1.0,
    )
    return AtomisticMultistatePlan(
        thermodynamic,
        np.arange(thermodynamic.state_count),
        qualification=qualification,
        exchange=AtomisticReplicaExchangePlan(1),
        run_id=run_id,
        history_label=history_label,
    ).prepare(dynamics)


def _lane[T](tree: T, lane: int, /) -> T:
    return jax.tree.map(lambda leaf: leaf[lane], tree)


def _history(
    term: PreparedAtomisticBiasTerm, state: AbstractAtomisticTermState, /
) -> AtomisticBiasHistoryState:
    """The validated history of one unbatched history-bias term state."""
    history = term.require_state(state).history
    if history is None:
        raise AssertionError(f"Bias term {term.name!r} binds no history.")
    return history


def _pair_distances(positions: object, /) -> np.ndarray:
    q = np.asarray(positions)
    return np.linalg.norm(q[..., 1, :] - q[..., 0, :], axis=-1)


def _gaussian(height: float, width: float, delta: np.ndarray, /) -> np.ndarray:
    return height * np.exp(-0.5 * (np.asarray(delta) / width) ** 2)


def _window_reduced(
    distance: np.ndarray, centers: np.ndarray, beta: float, /
) -> np.ndarray:
    """Independent ``beta (U_bond + w_k)`` of every window ``k`` (rows) at each replica."""
    bond = 0.5 * worker.BOND_STIFFNESS * (distance - worker.BOND_LENGTH) ** 2
    umbrella = 0.5 * worker.WINDOW_STIFFNESS * (distance[None, :] - centers[:, None]) ** 2
    return beta * (bond[None, :] + umbrella)


# --------------------------------------------------------------------------
# 1. Periodic volume umbrella with MTK cell response.

VOLUME_STIFFNESS, VOLUME_CENTER, EXTERNAL_PRESSURE = 0.01, 50.0, 0.01
CELL_POSITIONS = jnp.asarray([[0.5, 1.0, 1.0], [2.5, 1.0, 1.0]])
MTK_STEPS = 150
# Declared ledger policy: relative extended-energy excursion of the
# deterministic MTK realization at omega dt ~ 0.03 stays below 1e-3.
MTK_RELATIVE_LEDGER_BOUND = 1.0e-3


def test_volume_umbrella_enters_mtk_cell_response_with_tensile_sign() -> None:
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(3.0 * np.eye(3)))
    variables = cv_program(
        system, CollectiveVariableKind.CELL_VOLUME, np.zeros((0,), np.int64)
    )
    term = bias_term(
        system, variables, HarmonicBiasForm(np.asarray([VOLUME_CENTER]), VOLUME_STIFFNESS)
    )
    integrator = MTKNPTPlan(
        0.002,
        MTKBarostatPlan(0.5),
        NoseHooverChainParameters(0.5),
        NoseHooverChainParameters(0.5),
    )
    dynamics = biased_dynamics(system, term, integrator)
    thermodynamic = table(
        dynamics,
        row(measure(system), "npt", temperature=1.0, pressure=EXTERNAL_PRESSURE),
    )
    state = start(dynamics, thermodynamic, CELL_POSITIONS)

    def analytic(volume: float) -> tuple[float, float]:
        # V_b = kappa (V - V0)^2 / 2; tension-positive stress sym(dV_b/d eps)/V.
        return (
            0.5 * VOLUME_STIFFNESS * (volume - VOLUME_CENTER) ** 2,
            VOLUME_STIFFNESS * (volume - VOLUME_CENTER),
        )

    volume = float(np.linalg.det(np.asarray(state.cell_vectors)))
    np.testing.assert_allclose(volume, 27.0, rtol=1e-14)
    energy, tension = analytic(volume)
    # Start state: V < V0 so the umbrella stress is compressive (negative)
    # and its pressure contribution -kappa (V - V0) = +0.23 exceeds P_ext.
    np.testing.assert_allclose(
        np.asarray(state.force.stress), tension * np.eye(3), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(float(state.force.potential_energy), energy, rtol=1e-12)
    initial_extended = float(state.energy.extended_energy)
    volumes = [volume]
    excursion = 0.0
    for _ in range(MTK_STEPS):
        evaluation = step_detailed(dynamics, state, thermodynamic)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
        volume = float(np.linalg.det(np.asarray(state.cell_vectors)))
        volumes.append(volume)
        energy, tension = analytic(volume)
        np.testing.assert_allclose(
            np.asarray(state.force.stress), tension * np.eye(3), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            float(state.energy.potential_energy), energy, rtol=1e-10
        )
        np.testing.assert_allclose(
            float(state.energy.pressure_volume), EXTERNAL_PRESSURE * volume, rtol=1e-12
        )
        # Ideal gas at rest: no particle force, the momenta stay exactly zero.
        np.testing.assert_array_equal(np.asarray(state.kinematics.momenta), 0.0)
        excursion = max(
            excursion, abs(float(state.energy.extended_energy) - initial_extended)
        )
    # The tensile sign drives expansion from V = 27 toward V0 - P/kappa = 49;
    # a swapped stress/pressure sign would contract the cell instead.
    assert volumes[10] > volumes[0]
    assert max(volumes) > 30.0
    assert excursion <= MTK_RELATIVE_LEDGER_BOUND * abs(initial_extended)


# --------------------------------------------------------------------------
# 2. Well-tempered metadynamics history.

WT_HEIGHT, WT_WIDTH, WT_FACTOR, WT_TEMPERATURE, WT_STEPS = 0.2, 0.1, 4.0, 1.0, 10


def test_well_tempered_heights_follow_the_pre_deposit_history() -> None:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    history = AtomisticBiasHistoryPlan(
        1,
        height=WT_HEIGHT,
        widths=WT_WIDTH,
        capacity=16,
        bias_factor=WT_FACTOR,
        temperature=WT_TEMPERATURE,
    )
    term = bias_term(system, variables, HillHistoryBiasForm(history))
    dynamics = biased_dynamics(system, term, VelocityVerletPlan(0.05))
    thermodynamic = nve_table(dynamics)
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 2.0, 0.0]])
    velocity = jnp.asarray([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    state = runtime.initialize(
        dynamics.initialize_state(
            positions, thermodynamic, velocity=velocity, key=jax.random.key(0)
        )
    )
    damping = (
        float(system.plan.units.boltzmann_constant) * WT_TEMPERATURE * (WT_FACTOR - 1.0)
    )
    centers: list[float] = []
    heights: list[float] = []
    for _ in range(WT_STEPS):
        evaluation = worker.biased_step(runtime, state)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
        s = float(_pair_distances(state.dynamics.kinematics.positions))
        # Independent recursion: height against the frozen pre-event history.
        pre = float(
            np.sum(
                _gaussian(1.0, WT_WIDTH, s - np.asarray(centers)) * np.asarray(heights)
            )
        )
        height = WT_HEIGHT * math.exp(-pre / damping)
        # The new kernel sits at the committed CV: its history work is its height.
        np.testing.assert_allclose(float(evaluation.history_work), height, rtol=1e-12)
        centers.append(s)
        heights.append(height)
    stored = state.dynamics.term_states[runtime.bias_terms[0].name].history
    assert int(stored.written) == WT_STEPS and int(state.deposition_events) == WT_STEPS
    np.testing.assert_array_equal(np.asarray(stored.valid)[:WT_STEPS], True)
    np.testing.assert_allclose(
        np.asarray(stored.centers)[:WT_STEPS, 0], centers, rtol=1e-13
    )
    np.testing.assert_allclose(np.asarray(stored.heights)[:WT_STEPS], heights, rtol=1e-12)
    np.testing.assert_array_equal(
        np.asarray(stored.deposited_at)[:WT_STEPS], np.arange(WT_STEPS)
    )
    # Non-vacuous: the overlapping history damped later heights.
    assert min(heights) < 0.95 * WT_HEIGHT
    np.testing.assert_allclose(float(state.history_work), sum(heights), rtol=1e-12)


# --------------------------------------------------------------------------
# 3. Fixed-correspondence RMSD (aligned-MSD kernel) exploration.

RMSD_HEIGHT, RMSD_WIDTH, RMSD_STEPS = 0.1, 0.05, 4
RMSD_POSITIONS = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0], [0.1, 0.2, 1.3]]
)


def _proper_rotation(seed: int, /) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))[None, :]
    return q if np.linalg.det(q) > 0.0 else q @ np.diag([1.0, 1.0, -1.0])


def test_aligned_msd_history_is_alignment_invariant_and_matches_kabsch() -> None:
    system = atomistic_system([0, 1, 2, 3], [1.0] * 4)
    variables = cv_program(system, CollectiveVariableKind.CONFIGURATION, [0, 1, 2, 3])
    history = AtomisticBiasHistoryPlan(
        12,
        height=RMSD_HEIGHT,
        widths=RMSD_WIDTH,
        capacity=8,
        kernel=HillKernel.ALIGNED_MSD,
    )
    term = bias_term(system, variables, HillHistoryBiasForm(history))
    dynamics = biased_dynamics(system, term, VelocityVerletPlan(0.05))
    thermodynamic = nve_table(dynamics)
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    velocity = jnp.asarray(np.random.default_rng(3).normal(scale=1.0, size=(4, 3)))
    state = runtime.initialize(
        dynamics.initialize_state(
            RMSD_POSITIONS, thermodynamic, velocity=velocity, key=jax.random.key(1)
        )
    )
    prepared = runtime.bias_terms[0]
    rotation = _proper_rotation(11)
    translation = np.asarray([0.7, -1.3, 2.1])

    def history_energy(configuration: np.ndarray) -> float:
        energy, _, successful = prepared.bias(
            CollectiveVariableGeometry(jnp.asarray(configuration)),
            state.dynamics.term_states[prepared.name],
        )
        assert bool(successful)
        return float(energy)

    deposits: list[np.ndarray] = []
    for step in range(RMSD_STEPS):
        evaluation = worker.biased_step(runtime, state)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
        deposits.append(np.asarray(state.dynamics.kinematics.positions).copy())
        if step == 0:
            moved = deposits[0] @ rotation.T + translation
            # One kernel: the rigidly moved deposit has zero aligned MSD.
            np.testing.assert_allclose(history_energy(moved), RMSD_HEIGHT, rtol=1e-9)
            # Fixed correspondence without superposition would not see it.
            naive = np.mean(np.sum((moved - deposits[0]) ** 2, axis=-1))
            assert (
                RMSD_HEIGHT * math.exp(-naive / (2.0 * RMSD_WIDTH**2)) < 0.5 * RMSD_HEIGHT
            )
    stored = state.dynamics.term_states[prepared.name].history
    assert int(stored.written) == RMSD_STEPS
    # Kernels store the committed configuration in fixed atom order.
    np.testing.assert_allclose(
        np.asarray(stored.centers)[:RMSD_STEPS],
        np.stack([deposit.reshape((-1,)) for deposit in deposits]),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(np.asarray(stored.heights)[:RMSD_STEPS], RMSD_HEIGHT)
    weights = []
    for deposit in deposits:
        moved = deposit @ rotation.T + translation
        msd = np.asarray([kabsch_msd(moved, center) for center in deposits])
        weights.append(np.exp(-msd / (2.0 * RMSD_WIDTH**2)))
        np.testing.assert_allclose(
            history_energy(moved),
            RMSD_HEIGHT * float(np.sum(weights[-1])),
            rtol=1e-9,
            atol=1e-12,
        )
    # Non-vacuous exploration: the deposits are distinct configurations with
    # partially overlapping kernels.
    overlap = np.asarray(weights)[~np.eye(RMSD_STEPS, dtype=bool)]
    assert np.all(overlap < 0.999) and np.any(overlap > 1.0e-3)


# --------------------------------------------------------------------------
# 4. Pair-distance ABF: force-only port, then integrated conservative bias.

ABF_KT, ABF_STIFFNESS, ABF_LENGTH = 0.5, 20.0, 1.0
ABF_MINIMUM, ABF_MAXIMUM, ABF_BINS, ABF_MIN_SAMPLES = 0.85, 1.25, 32, 10
ABF_STEPS = 4000


def _abf_mean_force(r: np.ndarray, /) -> np.ndarray:
    # Independent radial reference A(r) = U(r) - 2 kT log r.
    return ABF_STIFFNESS * (r - ABF_LENGTH) - 2.0 * ABF_KT / r


def _abf_free_energy(r: np.ndarray, /) -> np.ndarray:
    return 0.5 * ABF_STIFFNESS * (r - ABF_LENGTH) ** 2 - 2.0 * ABF_KT * np.log(r)


def test_pair_distance_abf_port_refuses_energy_consumers_then_integrates() -> None:
    system = atomistic_system(
        [0, 1], [1.0, 1.0], topology=MolecularTopologyPlan(bonds=[[0, 1]])
    )
    bond = HarmonicBondPotential(np.asarray([ABF_STIFFNESS]), np.asarray([ABF_LENGTH]))
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    abf = AtomisticABFPlan(
        variables,
        minimum=ABF_MINIMUM,
        maximum=ABF_MAXIMUM,
        bins=ABF_BINS,
        temperature=ABF_KT,
        min_samples=ABF_MIN_SAMPLES,
        full_samples=4 * ABF_MIN_SAMPLES,
    ).prepare(system)
    port = AtomisticABFForcePort(abf, name="pair-distance")
    sampling = AtomisticDynamicsPlan(
        system,
        prepare_potential(system, (bond,)),
        BAOABLangevinPlan(0.02, 1.0),
        force_ports=(port,),
    ).prepare()
    phase = measure(system)
    single = table(sampling, row(phase, "nvt", temperature=ABF_KT))
    # While sampling, the force-only port has no energy: every consumer of
    # the scalar Hamiltonian refuses it.
    with pytest.raises(ValueError, match="complete scalar Hamiltonian"):
        table(
            sampling,
            row(phase, "nvt", temperature=ABF_KT),
            row(phase, "nvt", temperature=2.0 * ABF_KT),
        )
    # No canonical target exists without a complete scalar Hamiltonian, even
    # for the one-state table the port samples: the canonical qualification
    # (required by every multistate runtime and switching plan) refuses it.
    with pytest.raises(ValueError, match="complete scalar Hamiltonian"):
        AtomisticCanonicalSamplingQualification(
            sampling,
            single,
            "abf-force-port-qualification",
            sampling_exact=False,
            sampling_bias_bound=1.0,
        )
    runtime = PreparedBiasedDynamics(sampling, single, physical_force_groups=(0,))
    state = runtime.initialize(
        sampling.initialize_state(
            worker.window_positions(1.05),
            single,
            velocity=jnp.zeros((2, 3)),
            key=jax.random.key(91),
        )
    )
    for _ in range(ABF_STEPS):
        evaluation = worker.biased_step(runtime, state)
        assert bool(evaluation.successful)
        state = evaluation.accepted_state
    estimator = state.dynamics.term_states[port.name]
    counts = np.asarray(estimator.counts)
    assert int(estimator.irregular) == 0
    assert int(counts.sum()) == int(state.abf_samples)
    assert int(counts.sum()) + int(estimator.out_of_range) == ABF_STEPS
    assert int(counts.min()) >= ABF_MIN_SAMPLES

    edges = np.linspace(ABF_MINIMUM, ABF_MAXIMUM, ABF_BINS + 1)
    lower, upper = _abf_mean_force(edges[:-1]), _abf_mean_force(edges[1:])
    means = np.asarray(abf.mean_force(estimator))[:, 0]
    # Every pair sample is exactly A'(r) at its r (U depends on r only), and
    # A' is increasing: each half-open bin mean lies in [A'(left), A'(right)].
    slack = 1.0e-12 * float(np.max(np.abs(np.concatenate((lower, upper)))))
    assert np.all(means >= lower - slack) and np.all(means <= upper + slack)

    profile = abf_profile(abf, estimator)
    assert profile.formula_admitted and profile.connected
    assert bool(np.all(np.asarray(profile.support)))
    nodes = np.asarray(profile.nodes[0])
    free_energy = np.asarray(profile.free_energy)
    reference = _abf_free_energy(nodes) - _abf_free_energy(nodes[0])
    # Declared deterministic tolerance: node mean-force error is bracketed by
    # the bin, plus the trapezoid error w^3/12 max|A'''| on each interval.
    width = (ABF_MAXIMUM - ABF_MINIMUM) / ABF_BINS
    node_error = np.maximum(
        _abf_mean_force(nodes) - lower, upper - _abf_mean_force(nodes)
    )
    interval = 0.5 * width * (node_error[:-1] + node_error[1:]) + width**3 / 12.0 * (
        4.0 * ABF_KT / nodes[:-1] ** 3
    )
    bound = np.concatenate(([0.0], np.cumsum(interval)))
    np.testing.assert_array_less(
        np.abs(free_energy - free_energy[0] - reference), bound + 1.0e-12
    )
    assert float(np.ptp(reference)) > 3.0 * float(bound[-1])

    # The integrated realization of the frozen profile is an ordinary
    # conservative term: multi-state cross-evaluation admits it.
    integrated = AtomisticBiasTerm(
        variables,
        IntegratedProfileBiasForm(profile),
        name="integrated-abf",
        force_group=1,
    )
    energy_dynamics = AtomisticDynamicsPlan(
        system,
        prepare_potential(system, (bond, integrated)),
        BAOABLangevinPlan(0.02, 1.0),
    ).prepare()
    ladder = table(
        energy_dynamics,
        row(phase, "nvt", temperature=ABF_KT),
        row(phase, "nvt", temperature=2.0 * ABF_KT),
    )

    def total_energy(index: int) -> float:
        # V_bias = -A at a node, the physical bond energy is analytic.
        r = nodes[index]
        return 0.5 * ABF_STIFFNESS * (r - ABF_LENGTH) ** 2 - free_energy[index]

    # Constant extension has a kink at an outer node with nonzero slope.
    # Force comparisons use differentiable interior nodes, not an arbitrary
    # autodiff subgradient at that kink.
    probe_indices = (1, ABF_BINS // 2, ABF_BINS - 2)
    lanes = []
    for index in probe_indices:
        lane = energy_dynamics.initialize_state(
            worker.window_positions(float(nodes[index])),
            ladder,
            state_index=0,
            velocity=jnp.zeros((2, 3)),
            key=jax.random.key(index),
        )
        np.testing.assert_allclose(
            float(lane.force.potential_energy),
            total_energy(index),
            rtol=1e-10,
            atol=1e-12,
        )
        # Hermite node slope is the measured mean force: dE/dr = U'(r) - dA/dr.
        slope = ABF_STIFFNESS * (nodes[index] - ABF_LENGTH) - means[index]
        forces = np.asarray(lane.force.forces)
        np.testing.assert_allclose(forces[1], [-slope, 0.0, 0.0], rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(forces[0], [slope, 0.0, 0.0], rtol=1e-9, atol=1e-11)
        lanes.append(lane)
    runtime_ladder = _qualified_multistate(
        energy_dynamics, ladder, "integrated-abf-ladder"
    )
    pair = (
        lanes[0],
        energy_dynamics.initialize_state(
            lanes[2].kinematics.positions,
            ladder,
            state_index=1,
            velocity=jnp.zeros((2, 3)),
            key=jax.random.key(7),
        ),
    )
    ladder_state = runtime_ladder.initialize(pair, np.asarray([0, 1]), jax.random.key(8))
    reduced = runtime_ladder.reduced_potentials(ladder_state.dynamics)
    energies = np.asarray(
        [total_energy(probe_indices[0]), total_energy(probe_indices[-1])]
    )
    beta = 1.0 / np.asarray([ABF_KT, 2.0 * ABF_KT])
    np.testing.assert_array_equal(np.asarray(reduced.successful), True)
    np.testing.assert_allclose(
        np.asarray(reduced.values),
        beta[:, None] * energies[None, :],
        rtol=1e-10,
        atol=1e-12,
    )


# --------------------------------------------------------------------------
# 5. Umbrella-window replica exchange through segments.

EXCHANGE_CENTERS = (0.9, 1.0, 1.1)
EXCHANGE_CAPACITY = 3


def test_umbrella_window_exchange_records_complete_per_draw_evidence() -> None:
    _, dynamics, umbrella, _ = worker.window_hamiltonian(history=False)
    thermodynamic = worker.window_table(dynamics, umbrella, None, EXCHANGE_CENTERS, None)
    runtime = worker.window_runtime(dynamics, thermodynamic, "window-exchange")
    initial = runtime.initialize(
        worker.window_lanes(dynamics, thermodynamic, EXCHANGE_CENTERS),
        [0, 1, 2],
        jax.random.key(71),
    )
    first = AtomisticMultistateSegmentPlan(
        runtime, EXCHANGE_CAPACITY, 0, 0, runtime.initial_continuation_id
    ).run(initial)
    second = AtomisticMultistateSegmentPlan(
        runtime, EXCHANGE_CAPACITY, EXCHANGE_CAPACITY, 1, first.segment_id
    ).run(first.successor_state)
    # Independent replay of the same kernel, one iteration at a time, gives
    # the configuration each draw's cross-evaluation saw.
    replay = initial
    configurations = []
    replay_accepted = np.zeros((2,), dtype=np.int64)
    for _ in range(2 * EXCHANGE_CAPACITY):
        iteration = _iterate(runtime, replay)
        assert bool(iteration.successful)
        replay = iteration.state
        replay_accepted += np.asarray(iteration.exchange_accepted, dtype=np.int64)
        configurations.append(np.asarray(replay.dynamics.kinematics.positions))

    centers = np.asarray(EXCHANGE_CENTERS)
    beta = 1.0 / worker.WINDOW_TEMPERATURE
    labels = np.arange(3)
    attempted_total = np.zeros((2,), dtype=np.int64)
    accepted_total = np.zeros((2,), dtype=np.int64)
    draw = 0
    for segment in (first, second):
        assert bool(segment.successful) and int(segment.count) == EXCHANGE_CAPACITY
        np.testing.assert_array_equal(np.asarray(segment.iteration_valid), True)
        for t in range(EXCHANGE_CAPACITY):
            expected = _window_reduced(
                _pair_distances(configurations[draw]), centers, beta
            )
            np.testing.assert_allclose(
                np.asarray(segment.reduced_potentials[t]), expected, rtol=1e-9, atol=1e-12
            )
            np.testing.assert_array_equal(np.asarray(segment.sample_active[t]), True)
            np.testing.assert_array_equal(np.asarray(segment.coverage[t]), True)
            np.testing.assert_array_equal(np.asarray(segment.origin_state[t]), labels)
            np.testing.assert_array_equal(np.asarray(segment.draw_index[t]), draw)
            np.testing.assert_array_equal(np.asarray(segment.chain_index[t]), [0, 1, 2])
            np.testing.assert_array_equal(np.asarray(segment.repeat_index[t]), 0)
            np.testing.assert_array_equal(
                np.asarray(segment.dependence_group_index[t]), 0
            )
            np.testing.assert_array_equal(
                np.asarray(segment.pair_indices[t]), [[0, 1], [1, 2]]
            )
            attempted = np.asarray(segment.exchange_attempted[t])
            accepted = np.asarray(segment.exchange_accepted[t])
            log_acceptance = np.asarray(segment.exchange_log_acceptance[t])
            # Odd-even schedule: pair (0,1) on even draws, (1,2) on odd draws.
            np.testing.assert_array_equal(attempted, [draw % 2 == 0, draw % 2 == 1])
            assert not np.any(accepted & ~attempted)
            after = labels.copy()
            for start_slot in range(2):
                if not attempted[start_slot]:
                    assert log_acceptance[start_slot] == 0.0
                    continue
                a, b = labels[start_slot], labels[start_slot + 1]
                reference = (
                    expected[a, start_slot]
                    + expected[b, start_slot + 1]
                    - expected[b, start_slot]
                    - expected[a, start_slot + 1]
                )
                np.testing.assert_allclose(
                    log_acceptance[start_slot], reference, rtol=1e-8, atol=1e-10
                )
                # Metropolis: a non-negative log acceptance is always accepted.
                if reference > 1e-9:
                    assert accepted[start_slot]
                if accepted[start_slot]:
                    after[start_slot], after[start_slot + 1] = b, a
            np.testing.assert_array_equal(np.asarray(segment.state_at_replica[t]), after)
            attempted_total += attempted
            accepted_total += accepted
            labels = after
            draw += 1
        np.testing.assert_array_equal(np.asarray(segment.bias_adapting), False)
        assert np.asarray(segment.bias_deposited).shape == (EXCHANGE_CAPACITY, 0)
        assert np.asarray(segment.bias_history_revision).shape == (EXCHANGE_CAPACITY, 0)
        assert segment.bias_term_ids == (umbrella.term_id,)
        np.testing.assert_array_equal(np.asarray(segment.bias_owner_keys), 0)
    # Acceptance statistics: every scheduled attempt is booked once.
    np.testing.assert_array_equal(attempted_total, [3, 3])
    np.testing.assert_array_equal(accepted_total, replay_accepted)
    assert int(second.successor_state.exchange_action_counter) == 2 * EXCHANGE_CAPACITY
    np.testing.assert_array_equal(
        np.asarray(second.successor_state.state_at_replica), labels
    )
    np.testing.assert_array_equal(np.asarray(replay.state_at_replica), labels)
    # Fresh forces after the label moves: each lane's cache is its new window.
    final = second.successor_state.dynamics
    distance = _pair_distances(final.kinematics.positions)
    np.testing.assert_allclose(
        distance, _pair_distances(configurations[-1]), rtol=1e-10, atol=1e-12
    )
    energy = 0.5 * worker.BOND_STIFFNESS * (distance - worker.BOND_LENGTH) ** 2 + (
        0.5 * worker.WINDOW_STIFFNESS * (distance - centers[labels]) ** 2
    )
    np.testing.assert_allclose(
        np.asarray(final.force.potential_energy), energy, rtol=1e-10
    )
    dataset = phx.uq.reduced_potential_dataset_from_multistate(second)
    np.testing.assert_array_equal(np.asarray(dataset.sample_active), True)


# --------------------------------------------------------------------------
# 6. Shared / walker / state history ownership in multistate.

OWNER_HEIGHT, OWNER_WIDTH = 0.3, 0.2
OWNER_MAP = (1, 0)  # row 0 evaluates state owner 1, row 1 owner 0


def _owner_runtime(label: BiasHistoryLabel, owner_map: tuple[int, int], /) -> tuple:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    histories = {
        "shared": AtomisticBiasHistoryPlan(
            1, height=OWNER_HEIGHT, widths=OWNER_WIDTH, capacity=4
        ),
        "walker": AtomisticBiasHistoryPlan(
            1,
            height=OWNER_HEIGHT,
            widths=OWNER_WIDTH,
            capacity=4,
            ownership=BiasHistoryOwnership.WALKER,
            owner_count=2,
        ),
        "state": AtomisticBiasHistoryPlan(
            1,
            height=OWNER_HEIGHT,
            widths=OWNER_WIDTH,
            capacity=4,
            ownership=BiasHistoryOwnership.STATE,
            owner_count=2,
        ),
    }
    terms = {
        name: AtomisticBiasTerm(
            variables, HillHistoryBiasForm(plan), name=f"{name}-history", force_group=1
        )
        for name, plan in histories.items()
    }
    potential = prepare_potential(system, (zero_field(), *terms.values()))
    dynamics = AtomisticDynamicsPlan(
        system, potential, BAOABLangevinPlan(0.01, 1.0)
    ).prepare()

    def ladder(keys: tuple[int, int]) -> PreparedThermodynamicStateTable:
        rows = tuple(
            AtomisticThermodynamicStatePlan(
                measure(system),
                ensemble="nvt",
                temperature=1.0,
                bias=AtomisticBiasRowBinding(
                    {},
                    owner_keys={terms["state"].name: keys[index]},
                    histories={term.name: "adaptive" for term in terms.values()},
                ),
                state_id=f"owner-row-{index}",
            )
            for index in range(2)
        )
        return PreparedThermodynamicStateTable(dynamics, rows)

    thermodynamic = ladder(owner_map)
    runtime = _qualified_multistate(
        dynamics, thermodynamic, f"owner-map-{label.value}", history_label=label
    )
    return system, dynamics, terms, ladder, thermodynamic, runtime


@pytest.mark.parametrize("label", [BiasHistoryLabel.ORIGIN, BiasHistoryLabel.TARGET])
def test_shared_walker_state_history_owners_follow_the_declared_owner_map(
    label: BiasHistoryLabel,
) -> None:
    _, dynamics, terms, ladder, thermodynamic, runtime = _owner_runtime(label, OWNER_MAP)
    # A state owner key outside the declared owner count is refused.
    with pytest.raises(ValueError):
        _qualified_multistate(dynamics, ladder((0, 2)), "owner-map-overflow")
    lanes = tuple(
        dynamics.initialize_state(
            jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            thermodynamic,
            state_index=index,
            velocity=jnp.zeros((3, 3)),
            key=jax.random.key(80 + index),
        )
        for index, distance in enumerate((1.0, 1.3))
    )
    iteration = _iterate(
        runtime, runtime.initialize(lanes, np.asarray([0, 1]), jax.random.key(83))
    )
    assert bool(iteration.successful)
    # Empty histories make both rows the same Hamiltonian: the swap has log
    # acceptance exactly zero and log(u) < 0 accepts it.
    assert float(iteration.exchange_log_acceptance[0]) == 0.0
    assert bool(iteration.exchange_accepted[0])
    origin, target = np.asarray([0, 1]), np.asarray([1, 0])
    np.testing.assert_array_equal(np.asarray(iteration.origin_state), origin)
    np.testing.assert_array_equal(np.asarray(iteration.state_at_replica_after), target)
    tagged = origin if label is BiasHistoryLabel.ORIGIN else target
    expected_tags = {
        "shared": np.asarray([0, 0]),
        "walker": np.asarray([0, 1]),
        "state": np.asarray([OWNER_MAP[int(state)] for state in tagged]),
    }
    committed = iteration.state.dynamics
    s = _pair_distances(committed.kinematics.positions)
    owners = {}
    prepared = {term.name: term for term in runtime.bias_terms}
    for name, term in terms.items():
        history = _history(
            prepared[term.name], _lane(committed.term_states[term.name], 0)
        )
        assert int(history.written) == 2
        np.testing.assert_array_equal(np.asarray(history.owners)[:2], expected_tags[name])
        np.testing.assert_allclose(np.asarray(history.centers)[:2, 0], s, rtol=1e-13)
        np.testing.assert_allclose(np.asarray(history.heights)[:2], OWNER_HEIGHT)
        owners[name] = np.asarray(history.owners)[:2]
    # Evaluated owner keys after the assignment move (rows at target labels).
    column = {
        name: runtime.bias_term_ids.index(term.term_id) for name, term in terms.items()
    }
    keys = np.asarray(iteration.bias_owner_keys)
    np.testing.assert_array_equal(keys[:, column["shared"]], [0, 0])
    np.testing.assert_array_equal(keys[:, column["walker"]], [0, 1])
    np.testing.assert_array_equal(
        keys[:, column["state"]], [OWNER_MAP[int(state)] for state in target]
    )
    np.testing.assert_array_equal(np.asarray(iteration.bias_history_revision), [2, 2, 2])
    np.testing.assert_array_equal(np.asarray(iteration.bias_deposited), True)
    assert bool(iteration.bias_adapting) and not bool(iteration.equilibrium_sample)
    # Fresh forces: each lane feels exactly the kernels its owner keys select.
    for lane in range(2):
        kernels = _gaussian(OWNER_HEIGHT, OWNER_WIDTH, s[lane] - s)
        selected = (
            np.ones((2,), dtype=bool),
            owners["walker"] == lane,
            owners["state"] == OWNER_MAP[int(target[lane])],
        )
        expected = sum(float(np.sum(kernels[mask])) for mask in selected)
        np.testing.assert_allclose(
            float(committed.force.potential_energy[lane]),
            expected,
            rtol=1e-12,
            atol=1e-14,
        )


# --------------------------------------------------------------------------
# 7. Freeze-to-analysis: adaptive deposition, freeze, frozen segment, MBAR.

ADAPTIVE_ITERATIONS, ANALYSIS_ITERATIONS = 4, 16


def _mbar_reference(values: np.ndarray, origin: np.ndarray, /) -> np.ndarray:
    """Independent dense MBAR fixed point, gauge ``f_0 = 0``."""
    counts = np.bincount(origin.astype(np.int64), minlength=values.shape[0]).astype(
        np.float64
    )
    free = np.zeros((values.shape[0],))
    for _ in range(200_000):
        exponent = (
            free[:, None] - values + np.log(np.where(counts > 0, counts, 1.0))[:, None]
        )
        exponent = np.where(counts[:, None] > 0, exponent, -np.inf)
        peak = np.max(exponent, axis=0)
        log_denominator = peak + np.log(np.sum(np.exp(exponent - peak), axis=0))
        terms = -values - log_denominator[None, :]
        peak_terms = np.max(terms, axis=1)
        updated = -(
            peak_terms + np.log(np.sum(np.exp(terms - peak_terms[:, None]), axis=1))
        )
        updated = updated - updated[0]
        if np.max(np.abs(updated - free)) < 1e-12:
            return updated
        free = updated
    raise AssertionError("Reference MBAR did not converge.")


def test_freeze_to_analysis_authenticates_the_history_and_feeds_mbar() -> None:
    _, dynamics, umbrella, hill = worker.window_hamiltonian(history=True)
    centers = worker.FREEZE_CENTERS
    adaptive = worker.window_runtime(
        dynamics,
        worker.window_table(dynamics, umbrella, hill, centers, "adaptive"),
        "freeze-adaptive",
    )
    learned = AtomisticMultistateSegmentPlan(
        adaptive, ADAPTIVE_ITERATIONS, 0, 0, adaptive.initial_continuation_id
    ).run(
        adaptive.initialize(
            worker.window_lanes(dynamics, adaptive.thermodynamic, centers),
            [0, 1],
            jax.random.key(31),
        )
    )
    assert bool(learned.successful) and int(learned.count) == ADAPTIVE_ITERATIONS
    # Draws under an advancing history are never samples.
    np.testing.assert_array_equal(np.asarray(learned.sample_active), False)
    np.testing.assert_array_equal(np.asarray(learned.coverage), False)
    np.testing.assert_array_equal(np.asarray(learned.bias_adapting), True)
    np.testing.assert_array_equal(np.asarray(learned.bias_deposited), True)
    np.testing.assert_array_equal(
        np.asarray(learned.bias_history_revision)[:, 0],
        2 * np.arange(1, ADAPTIVE_ITERATIONS + 1),
    )
    (learned_digest,) = learned.bias_history_digests()

    bindings = learned.successor_state.dynamics.term_states
    hill_term = adaptive.bias_terms[adaptive.bias_term_slots.index(hill.name)]
    frozen = FrozenBiasHistory(
        hill.form.history, _history(hill_term, _lane(bindings[hill.name], 0))
    )
    assert frozen.digest == learned_digest
    assert not frozen.exhausted and not frozen.fifo_overwritten
    analysis_table = worker.window_table(
        dynamics, umbrella, hill, centers, frozen.declaration
    )
    analysis = worker.window_runtime(dynamics, analysis_table, "freeze-analysis")
    assert analysis.bias_history_declarations == (frozen.declaration,)
    labels = np.asarray(learned.successor_state.state_at_replica)

    def continued(
        lane_bindings: AtomisticTermStateBindings, lane: int, /
    ) -> AtomisticDynamicsState:
        kinematics = learned.successor_state.dynamics.kinematics
        return dynamics.initialize_state(
            kinematics.positions[lane],
            analysis_table,
            state_index=int(labels[lane]),
            momentum=kinematics.momenta[lane],
            key=jax.random.key(100 + lane),
            term_states=lane_bindings,
        )

    lane_bindings = [_lane(bindings, lane) for lane in range(2)]
    # Label-only authentication is refused: altered content under the same
    # declaration does not initialize.
    term_state = hill_term.require_state(lane_bindings[0][hill.name])
    history = _history(hill_term, term_state)
    tampered = with_bias_history(
        term_state,
        eqx.tree_at(lambda value: value.heights, history, 1.5 * history.heights),
    )
    altered = AtomisticTermStateBindings(
        {
            **dict(zip(lane_bindings[0].slots, lane_bindings[0].values, strict=True)),
            hill.name: tampered,
        }
    )
    with pytest.raises(ValueError, match="authenticate"):
        analysis.initialize(
            (continued(altered, 0), continued(lane_bindings[1], 1)),
            labels,
            jax.random.key(36),
        )
    sampled = AtomisticMultistateSegmentPlan(
        analysis, ANALYSIS_ITERATIONS, 0, 0, analysis.initial_continuation_id
    ).run(
        analysis.initialize(
            tuple(continued(lane_bindings[lane], lane) for lane in range(2)),
            labels,
            jax.random.key(37),
        )
    )
    assert bool(sampled.successful) and int(sampled.count) == ANALYSIS_ITERATIONS
    np.testing.assert_array_equal(np.asarray(sampled.sample_active), True)
    np.testing.assert_array_equal(np.asarray(sampled.coverage), True)
    np.testing.assert_array_equal(np.asarray(sampled.bias_adapting), False)
    np.testing.assert_array_equal(np.asarray(sampled.bias_deposited), False)
    np.testing.assert_array_equal(
        np.asarray(sampled.bias_history_revision), 2 * ADAPTIVE_ITERATIONS
    )
    assert sampled.bias_history_digests() == (frozen.digest,)
    final_history = _history(
        hill_term, _lane(sampled.successor_state.dynamics.term_states[hill.name], 1)
    )
    assert bias_history_digest(final_history) == frozen.digest

    # Every row's cross-evaluation carries the frozen history (independent
    # Gaussian sum) at the last draw's configuration.
    kept = np.asarray(frozen.state.valid)
    history_centers = np.asarray(frozen.state.centers)[kept, 0]
    history_heights = np.asarray(frozen.state.heights)[kept]
    distance = _pair_distances(sampled.successor_state.dynamics.kinematics.positions)
    hills = np.asarray(
        [
            np.sum(
                history_heights * _gaussian(1.0, worker.HILL_WIDTH, r - history_centers)
            )
            for r in distance
        ]
    )
    beta = 1.0 / worker.WINDOW_TEMPERATURE
    expected = (
        _window_reduced(distance, np.asarray(centers), beta) + beta * hills[None, :]
    )
    np.testing.assert_allclose(
        np.asarray(sampled.reduced_potentials[ANALYSIS_ITERATIONS - 1]),
        expected,
        rtol=1e-9,
        atol=1e-12,
    )

    dataset = phx.uq.reduced_potential_dataset_from_multistate(sampled)
    active = np.asarray(dataset.sample_active)
    assert int(active.sum()) == 2 * ANALYSIS_ITERATIONS
    estimate = phx.uq.multistate_bennett_acceptance_ratio(
        dataset, phx.uq.FreeEnergySelectionPlan(block_length=1)
    )
    assert int(estimate.numerical_status) == int(phx.uq.FreeEnergyStatus.SUCCESS)
    # A finite-step kernel is never self-certified as exact.
    assert int(estimate.statistical_status) & int(
        phx.uq.FreeEnergyStatus.UNQUALIFIED_KERNEL
    )
    reference = _mbar_reference(
        np.asarray(dataset.values)[:, active], np.asarray(dataset.origin_state)[active]
    )
    np.testing.assert_allclose(
        np.asarray(estimate.free_energies), reference, rtol=0.0, atol=1e-7
    )


# --------------------------------------------------------------------------
# 8. Fresh-process restart of biased dynamics and multistate.


def test_fresh_process_restart_continues_bitwise(tmp_path: Path) -> None:
    biased_path = tmp_path / "biased.phx"
    multistate_path = tmp_path / "multistate.phx"
    output = tmp_path / "continued.npz"
    with default_device(jax.devices("cpu")[0]):
        runtime, state = worker.metadynamics_runtime()
        state, successes = worker.advance_biased(
            runtime, state, worker.SAVED_BIASED_STEPS
        )
        assert all(successes)
        biased_written = write_biased_dynamics_checkpoint(biased_path, runtime, state)
        continued, successes = worker.advance_biased(
            runtime, state, worker.CONTINUED_BIASED_STEPS
        )
        assert all(successes)

        ms_runtime, initial = worker.restart_multistate_runtime()
        first = AtomisticMultistateSegmentPlan(
            ms_runtime, worker.SEGMENT_CAPACITY, 0, 0, ms_runtime.initial_continuation_id
        ).run(initial)
        assert bool(first.successful)
        ms_written = write_atomistic_multistate_checkpoint(
            multistate_path,
            AtomisticMultistateCheckpointPlan(ms_runtime, worker.SEGMENT_CAPACITY),
            first,
        )
        segment = worker.continue_multistate(ms_runtime, first)
        assert bool(segment.successful)
        expected = worker.host_leaves((continued, segment))

    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            str(biased_path),
            str(multistate_path),
            str(output),
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY), "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output) as archive:
        assert str(archive["biased_payload"]) == biased_written.payload_id
        assert str(archive["multistate_payload"]) == ms_written.payload_id
        names = sorted(name for name in archive.files if name.startswith("leaf_"))
        observed = [archive[name] for name in names]
    assert len(observed) == len(expected)
    for left, right in zip(observed, expected, strict=True):
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left, right)

    # The continuation is non-trivial: it moved, deposited and advanced revisions.
    term_id = runtime.bias_terms[0].name
    history = continued.dynamics.term_states[term_id].history
    assert (
        int(history.written) == worker.SAVED_BIASED_STEPS + worker.CONTINUED_BIASED_STEPS
    )
    assert int(continued.dynamics.step_index) == int(history.written)
    assert not np.array_equal(
        np.asarray(continued.dynamics.kinematics.positions),
        np.asarray(state.dynamics.kinematics.positions),
    )
    revisions = np.asarray(segment.bias_history_revision)[:, 0]
    np.testing.assert_array_equal(
        revisions,
        2 * np.arange(worker.SEGMENT_CAPACITY + 1, 2 * worker.SEGMENT_CAPACITY + 1),
    )
