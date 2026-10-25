"""Bias-aware multistate execution: window exchange, frozen-history MBAR, adaptivity.

Every scenario uses one tethered radial coordinate: atom 0 is immobile at the
origin, atom 1 is mobile under the isotropic tether ``U0 = (k/2)|x1|^2`` and
the distance CV is ``r = |x1|``.  Umbrella windows ``V_w = (kappa/2)(r - c_w)^2``
and Gaussian hill histories are bias terms of the same scalar program, so
every expected energy, force, acceptance and free energy below is recomputed
independently from ``r`` (analytic expressions or 1-D radial quadrature).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax.atomistic._dynamics import AtomisticDynamicsPlan
from phydrax.atomistic._term_state import AtomisticTermStateBindings
from phydrax.atomistic._thermal import (
    BAOABLangevinPlan,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
)
from phydrax.atomistic.sampling import (
    AtomisticBiasHistoryPlan,
    AtomisticBiasHistoryState,
    AtomisticBiasRowBinding,
    AtomisticBiasTerm,
    AtomisticBiasTermState,
    AtomisticCanonicalSamplingQualification,
    AtomisticDrawConfigurationCapture,
    AtomisticMultistateCheckpointPlan,
    AtomisticMultistateEquilibriumSamples,
    AtomisticMultistatePlan,
    AtomisticMultistateSegmentPlan,
    AtomisticReplicaExchangePlan,
    AtomisticThermodynamicStatePlan,
    BiasHistoryOwnership,
    CollectiveVariableKind,
    deposit_bias_history,
    FrozenBiasHistory,
    HarmonicBiasForm,
    HillHistoryBiasForm,
    PreparedAtomisticMultistate,
    PreparedThermodynamicStateTable,
    read_atomistic_multistate_checkpoint,
    with_bias_history,
    write_atomistic_multistate_checkpoint,
)
from phydrax.atomistic.sampling._bias_history import bias_history_digest
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    ExternalFieldTerm,
    measure,
    prepare_potential,
)
from tests._support.atomistic_sampling_fixtures import cv_program


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


TETHER = 1.0
# BAOAB is not an exact canonical kernel: its O(dt^2) configurational bias is
# declared as the qualification bound and is the absolute floor of the MBAR
# comparison below.
SAMPLING_BIAS_BOUND = 0.03


# ----------------------------------------------------------------- fixtures


def _system() -> Any:
    return atomistic_system([0, 1], [1.0, 1.0], mobile_mask=[False, True])


def _distance(system: Any) -> Any:
    return cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])


def _umbrella(variables: Any, stiffness: float) -> AtomisticBiasTerm:
    return AtomisticBiasTerm(
        variables,
        HarmonicBiasForm(np.asarray([1.0]), stiffness),
        name="umbrella",
        force_group=1,
    )


def _hill_plan(
    *,
    height: float,
    capacity: int,
    widths: float = 0.3,
    interval: int = 1,
    ownership: BiasHistoryOwnership = BiasHistoryOwnership.SHARED,
    owner_count: int = 1,
) -> AtomisticBiasHistoryPlan:
    return AtomisticBiasHistoryPlan(
        1,
        height=height,
        widths=widths,
        capacity=capacity,
        interval=interval,
        ownership=ownership,
        owner_count=owner_count,
    )


def _hill(variables: Any, plan: AtomisticBiasHistoryPlan) -> AtomisticBiasTerm:
    return AtomisticBiasTerm(
        variables, HillHistoryBiasForm(plan), name="hill", force_group=2
    )


def _dynamics(system: Any, terms: tuple[Any, ...], integrator: Any) -> Any:
    tether = ExternalFieldTerm("tether", "harmonic", [0.0, 0.0, 0.0], stiffness=TETHER)
    potential = prepare_potential(system, (tether, *terms))
    return AtomisticDynamicsPlan(system, potential, integrator).prepare()


def _quasistatic() -> Any:
    # A canonical NVT method whose 1e-7 step leaves the kinematics unchanged
    # to rounding: the iteration's moves act on the declared configurations.
    return NoseHooverChainPlan(1.0e-7, NoseHooverChainParameters(0.5))


def _table(dynamics: Any, bindings: tuple[AtomisticBiasRowBinding, ...]) -> Any:
    phase_space = measure(dynamics.system)
    rows = tuple(
        AtomisticThermodynamicStatePlan(
            phase_space,
            ensemble="nvt",
            temperature=1.0,
            bias=binding,
            state_id=f"row-{index}",
        )
        for index, binding in enumerate(bindings)
    )
    return PreparedThermodynamicStateTable(dynamics, rows)


def _plan(
    dynamics: Any, table: Any, replicas: int, *, exchange: bool, run_id: str
) -> AtomisticMultistatePlan:
    qualification = AtomisticCanonicalSamplingQualification(
        dynamics,
        table,
        "tethered-radial-bias-multistate-kernel",
        sampling_exact=False,
        sampling_bias_bound=SAMPLING_BIAS_BOUND,
    )
    return AtomisticMultistatePlan(
        table,
        np.arange(replicas) + 100,
        qualification=qualification,
        exchange=AtomisticReplicaExchangePlan(1) if exchange else None,
        run_id=run_id,
    )


def _lanes(
    dynamics: Any,
    table: Any,
    radii: tuple[float, ...],
    *,
    term_states: AtomisticTermStateBindings | None = None,
    seed: int = 0,
) -> tuple[Any, ...]:
    lanes = []
    for index, radius in enumerate(radii):
        position = jnp.asarray([[0.0, 0.0, 0.0], [radius, 0.0, 0.0]], dtype=jnp.float64)
        lanes.append(
            dynamics.initialize_state(
                position,
                table,
                state_index=min(index, table.state_count - 1),
                velocity=jnp.zeros_like(position),
                key=jax.random.key(seed + index),
                term_states=term_states,
            )
        )
    return tuple(lanes)


def _frozen(
    plan: AtomisticBiasHistoryPlan,
    variables: Any,
    centers: tuple[float, ...],
    owners: tuple[int, ...] | None = None,
) -> FrozenBiasHistory:
    """Explicitly deposited kernels, frozen for analysis."""
    count = len(centers)
    deposition = deposit_bias_history(
        plan,
        plan.initialize(jnp.float64),
        jnp.asarray(centers, dtype=jnp.float64).reshape((count, 1)),
        jnp.zeros((count,), dtype=jnp.int32)
        if owners is None
        else jnp.asarray(owners, dtype=jnp.int32),
        jnp.ones((count,), dtype=jnp.bool_),
        variables.metrics,
        boltzmann_constant=1.0,
    )
    assert bool(deposition.inserted)
    return FrozenBiasHistory(plan, deposition.state)


def _bind_history(dynamics: Any, slot: str, history: Any) -> AtomisticTermStateBindings:
    initial = dynamics.initial_term_states(jnp.float64)
    values = dict(zip(initial.slots, initial.values, strict=True))
    values[slot] = with_bias_history(values[slot], history)
    return AtomisticTermStateBindings(values)


def _lane(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _lane_bias_state(
    runtime: PreparedAtomisticMultistate,
    bindings: AtomisticTermStateBindings,
    slot: str,
    index: int,
) -> AtomisticBiasTermState:
    """One lane's bias term state, validated by its prepared bias term."""
    term = next(term for term in runtime.bias_terms if term.name == slot)
    return term.require_state(_lane(bindings[slot], index))


def _lane_history(
    runtime: PreparedAtomisticMultistate,
    bindings: AtomisticTermStateBindings,
    slot: str,
    index: int,
) -> AtomisticBiasHistoryState:
    history = _lane_bias_state(runtime, bindings, slot, index).history
    if history is None:
        raise AssertionError(f"Bias term {slot!r} binds no history.")
    return history


def _radii(positions: Any) -> np.ndarray:
    host = np.asarray(positions)
    return np.linalg.norm(host[:, 1] - host[:, 0], axis=-1)


def _hill_energy(
    radius: np.ndarray, centers: Any, height: float, width: float
) -> np.ndarray:
    offsets = (np.asarray(radius)[..., None] - np.asarray(centers)[None, :]) / width
    return height * np.sum(np.exp(-0.5 * offsets**2), axis=-1)


def _radial_log_partition(energy: np.ndarray, radius: np.ndarray, beta: float) -> float:
    """``ln int r^2 exp(-beta E) dr`` by the composite trapezoid rule."""
    shift = float(np.min(beta * energy))
    integrand = radius**2 * np.exp(-(beta * energy - shift))
    step = radius[1] - radius[0]
    total = step * (np.sum(integrand) - 0.5 * (integrand[0] + integrand[-1]))
    return float(np.log(total)) - shift


def _assert_fresh_forces(dynamics: Any, table: Any, batched: Any) -> None:
    """Every lane's force cache equals a fresh evaluation at its current label."""
    for index in range(batched.thermodynamic_state_index.shape[0]):
        lane = _lane(batched, index)
        fresh = dynamics.evaluate_state(lane, table)
        assert bool(fresh.successful)
        np.testing.assert_allclose(
            np.asarray(lane.force.forces),
            np.asarray(fresh.forces),
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            float(lane.force.potential_energy),
            float(fresh.energy),
            rtol=1e-13,
            atol=1e-14,
        )
        controls = table.state_at_replica(lane.thermodynamic_state_index).controls
        assert bool(dynamics.force_cache_valid(lane, controls))


# ------------------------------------------------- 1. harmonic-window exchange


def test_harmonic_window_exchange_matches_recomputed_acceptance_and_rebinds_forces() -> (
    None
):
    system = _system()
    variables = _distance(system)
    stiffness = 5.0
    umbrella = _umbrella(variables, stiffness)
    dynamics = _dynamics(system, (umbrella,), _quasistatic())
    centers = np.asarray([1.0, 1.5])
    table = _table(
        dynamics,
        tuple(
            AtomisticBiasRowBinding({umbrella.name: np.asarray([center])})
            for center in centers
        ),
    )
    runtime = _plan(dynamics, table, 2, exchange=True, run_id="window-exchange").prepare(
        dynamics
    )
    # Each replica starts at the other window's center, so the swap lowers
    # both window energies and is accepted deterministically.
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.5, 1.0)),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(5),
    )
    iteration = runtime.iterate(initial)

    assert bool(iteration.successful)
    assert bool(iteration.exchange_attempted[0])
    radius = _radii(iteration.state.dynamics.kinematics.positions)
    beta = float(np.asarray(table.beta)[0])
    np.testing.assert_array_equal(np.asarray(table.beta), [beta, beta])

    def window(label: int, value: float) -> float:
        return 0.5 * stiffness * (value - centers[label]) ** 2

    # Independent reduced potentials u_k(x_r) = beta (U0 + V_k) of the
    # configurations the exchange saw.
    expected_reduced = np.asarray(
        [[beta * (0.5 * TETHER * r**2 + window(k, r)) for r in radius] for k in range(2)]
    )
    np.testing.assert_allclose(
        np.asarray(iteration.reduced_potentials), expected_reduced, rtol=1e-12, atol=1e-12
    )
    # Replica 0 holds window a = 0, replica 1 window b = 1.
    expected = -beta * (
        (window(0, radius[1]) + window(1, radius[0]))
        - (window(0, radius[0]) + window(1, radius[1]))
    )
    assert expected > 0.0
    np.testing.assert_allclose(
        float(iteration.exchange_log_acceptance[0]), expected, rtol=1e-10, atol=1e-12
    )
    assert bool(iteration.exchange_accepted[0])

    state = iteration.state
    labels = np.asarray(state.state_at_replica)
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_array_equal(np.asarray(iteration.state_at_replica_after), [1, 0])
    np.testing.assert_array_equal(
        np.asarray(state.dynamics.thermodynamic_state_index), labels
    )
    # The umbrella control each lane binds is its new window's center.
    np.testing.assert_allclose(
        np.asarray(
            [
                _lane_bias_state(
                    runtime, state.dynamics.term_states, umbrella.name, lane
                ).controls
                for lane in range(2)
            ]
        )[:, 0],
        centers[labels],
    )
    # No history term: owners stay at the static key; no revision evidence.
    assert runtime.bias_term_ids == (umbrella.term_id,)
    assert runtime.bias_history_term_ids == ()
    np.testing.assert_array_equal(np.asarray(iteration.bias_owner_keys), np.zeros((2, 1)))
    assert iteration.bias_history_revision.shape == (0,)
    assert iteration.bias_deposited.shape == (0,)

    _assert_fresh_forces(dynamics, table, state.dynamics)
    positions = np.asarray(state.dynamics.kinematics.positions)
    for lane in range(2):
        x1 = positions[lane, 1] - positions[lane, 0]
        r = float(np.linalg.norm(x1))
        center = centers[labels[lane]]
        analytic = -TETHER * positions[lane, 1] - stiffness * (r - center) * x1 / r
        np.testing.assert_allclose(
            np.asarray(state.dynamics.force.forces)[lane, 1],
            analytic,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            float(np.asarray(state.dynamics.force.potential_energy)[lane]),
            0.5 * TETHER * r**2 + 0.5 * stiffness * (r - center) ** 2,
            rtol=1e-12,
        )


# ----------------------------------- 2. frozen-history MBAR, zero-sample target


def test_frozen_history_mbar_matches_radial_quadrature_at_unsampled_target() -> None:
    system = _system()
    variables = _distance(system)
    stiffness = 2.5
    height = 0.8
    width = 0.3
    hill_centers = (1.3, 1.9)
    umbrella = _umbrella(variables, stiffness)
    hill_plan = _hill_plan(height=height, widths=width, capacity=4)
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (umbrella, hill), BAOABLangevinPlan(0.1, 2.0))
    frozen = _frozen(hill_plan, variables, hill_centers)
    windows = (1.1, 2.0)
    declaration = {hill.name: frozen.declaration}
    rows = tuple(
        AtomisticBiasRowBinding(
            {umbrella.name: np.asarray([center])}, histories=declaration
        )
        for center in windows
    ) + (
        # Unbiased target: same terms at coupling zero, never sampled.
        AtomisticBiasRowBinding(
            {umbrella.name: np.asarray([windows[0]])},
            couplings={umbrella.name: 0.0, hill.name: 0.0},
            histories=declaration,
        ),
    )
    table = _table(dynamics, rows)
    runtime = _plan(dynamics, table, 2, exchange=False, run_id="frozen-mbar").prepare(
        dynamics
    )
    bindings = _bind_history(dynamics, hill.name, frozen.state)
    initial = runtime.initialize(
        _lanes(dynamics, table, windows, term_states=bindings, seed=21),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(22),
    )
    capacity = 4000
    result = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    ).run(initial)

    assert bool(result.successful) and int(result.count) == capacity
    assert bool(np.all(np.asarray(result.sample_active)))
    assert not bool(np.any(np.asarray(result.bias_adapting)))
    assert not bool(np.any(np.asarray(result.bias_deposited)))
    np.testing.assert_array_equal(np.asarray(result.bias_history_revision), 2)
    assert result.bias_history_declarations == (frozen.declaration,)
    assert result.bias_history_digests() == (frozen.digest,)

    dataset = phx.uq.reduced_potential_dataset_from_multistate(result)
    np.testing.assert_array_equal(
        np.asarray(dataset.state_counts), [capacity, capacity, 0]
    )
    # Canonical draw-major flattening of the committed segment's own evidence.
    reduced = np.asarray(result.reduced_potentials)
    np.testing.assert_array_equal(
        np.asarray(dataset.values),
        reduced.transpose((1, 0, 2)).reshape((reduced.shape[1], -1)),
    )
    np.testing.assert_array_equal(
        np.asarray(dataset.origin_state), np.asarray(result.origin_state).reshape((-1,))
    )
    assert dataset.producer_id == result.producer_id
    assert dataset.bias_ids == result.bias_ids
    assert dataset.unit_system_id == result.units.unit_system_id
    estimate = phx.uq.multistate_bennett_acceptance_ratio(
        dataset, phx.uq.FreeEnergySelectionPlan(burn_in=200), reference_state=2
    )
    assert int(estimate.numerical_status) == int(phx.uq.FreeEnergyStatus.SUCCESS)
    statistical = int(estimate.statistical_status)
    assert not statistical & int(phx.uq.FreeEnergyStatus.DISCONNECTED)
    assert statistical & int(phx.uq.FreeEnergyStatus.UNQUALIFIED_KERNEL)

    # Independent reference: Delta f_w = -ln[Z_w / Z_target] with the 3-D
    # configuration integral reduced to r^2 dr (momenta cancel at equal T).
    beta = float(np.asarray(table.beta)[0])
    radius = np.linspace(0.0, 12.0, 240_001)
    base = 0.5 * TETHER * radius**2
    hills = _hill_energy(radius, hill_centers, height, width)
    log_target = _radial_log_partition(base, radius, beta)
    # Tolerance policy: 5 x the MBAR-reported (correlation-inflated) standard
    # error of f_w - f_target plus the declared kernel-bias floor.
    for index, center in enumerate(windows):
        umbrella_energy = 0.5 * stiffness * (radius - center) ** 2
        reference = log_target - _radial_log_partition(
            base + umbrella_energy + hills, radius, beta
        )
        without_hills = log_target - _radial_log_partition(
            base + umbrella_energy, radius, beta
        )
        error = float(np.asarray(estimate.standard_errors)[2, index])
        tolerance = 5.0 * error + SAMPLING_BIAS_BOUND
        assert np.isfinite(error) and error > 0.0
        # The comparison must resolve the frozen hills' contribution.
        assert tolerance < abs(reference - without_hills)
        assert (
            abs(float(np.asarray(estimate.free_energies)[index]) - reference) <= tolerance
        )


# ------------------------------------------------------ 3. counterfeit refusal


def _frozen_setup() -> tuple[Any, ...]:
    system = _system()
    variables = _distance(system)
    umbrella = _umbrella(variables, 2.5)
    hill_plan = _hill_plan(height=0.5, capacity=4)
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (umbrella, hill), BAOABLangevinPlan(0.05, 1.0))
    genuine = _frozen(hill_plan, variables, (1.2, 1.8))
    other = _frozen(hill_plan, variables, (1.2, 1.9))
    return variables, umbrella, hill, hill_plan, dynamics, genuine, other


def _windows(
    umbrella: Any, histories: tuple[dict[str, str] | None, ...]
) -> tuple[AtomisticBiasRowBinding, ...]:
    return tuple(
        AtomisticBiasRowBinding({umbrella.name: np.asarray([center])}, histories=history)
        for center, history in zip((1.1, 2.0), histories, strict=True)
    )


def test_counterfeit_frozen_declarations_are_refused_at_initialize() -> None:
    _, umbrella, hill, _, dynamics, genuine, other = _frozen_setup()
    genuine_bindings = _bind_history(dynamics, hill.name, genuine.state)
    other_bindings = _bind_history(dynamics, hill.name, other.state)
    assert genuine.digest != other.digest

    def runtime(declaration: str, run_id: str) -> tuple[Any, Any]:
        table = _table(dynamics, _windows(umbrella, ({hill.name: declaration},) * 2))
        return table, _plan(dynamics, table, 2, exchange=False, run_id=run_id).prepare(
            dynamics
        )

    # Control: the genuine content authenticates its own declaration.
    table, prepared = runtime(genuine.declaration, "genuine")
    prepared.initialize(
        _lanes(dynamics, table, (1.1, 2.0), term_states=genuine_bindings),
        [0, 1],
        jax.random.key(1),
    )
    # A fabricated digest label is refused against the bound content.
    table, prepared = runtime("frozen:" + "0" * 64, "fabricated-label")
    with pytest.raises(ValueError, match="does not authenticate its frozen declaration"):
        prepared.initialize(
            _lanes(dynamics, table, (1.1, 2.0), term_states=genuine_bindings),
            [0, 1],
            jax.random.key(1),
        )
    # A genuine declaration bound to different history content is refused.
    table, prepared = runtime(genuine.declaration, "swapped-content")
    with pytest.raises(ValueError, match="does not authenticate its frozen declaration"):
        prepared.initialize(
            _lanes(dynamics, table, (1.1, 2.0), term_states=other_bindings),
            [0, 1],
            jax.random.key(1),
        )
    # An adaptive history must be one content replicated on every lane.
    table, prepared = runtime("adaptive", "split-adaptive-content")
    lanes = _lanes(dynamics, table, (1.1,), term_states=genuine_bindings) + _lanes(
        dynamics, table, (2.0,), term_states=other_bindings, seed=1
    )
    with pytest.raises(ValueError, match="same content of history"):
        prepared.initialize(lanes, [0, 1], jax.random.key(1))


def test_misdeclared_history_rows_are_refused_at_prepare() -> None:
    _, umbrella, hill, _, dynamics, genuine, _ = _frozen_setup()
    frozen = {hill.name: genuine.declaration}

    def prepare(rows: tuple[AtomisticBiasRowBinding, ...], run_id: str) -> Any:
        table = _table(dynamics, rows)
        return _plan(dynamics, table, 2, exchange=False, run_id=run_id).prepare(dynamics)

    # Control: one common frozen declaration prepares.
    prepared = prepare(_windows(umbrella, (frozen, frozen)), "common-frozen")
    assert prepared.bias_history_term_ids == (hill.term_id,)
    assert prepared.bias_history_declarations == (genuine.declaration,)
    assert not prepared.history_adaptive

    with pytest.raises(ValueError, match="Only adaptive-history bias terms declare"):
        prepare(
            _windows(
                umbrella,
                ({**frozen, umbrella.name: genuine.declaration},) * 2,
            ),
            "declared-umbrella",
        )
    with pytest.raises(ValueError, match="one common declaration"):
        prepare(
            _windows(umbrella, ({hill.name: "adaptive"}, frozen)), "mixed-declarations"
        )
    missing = AtomisticBiasRowBinding(
        {umbrella.name: np.asarray([2.0])}, couplings={hill.name: 1.0}
    )
    with pytest.raises(ValueError, match="one common declaration"):
        prepare((_windows(umbrella, (frozen, frozen))[0], missing), "missing-declaration")
    with pytest.raises(ValueError, match="must be bound by the table rows"):
        prepare(
            tuple(
                AtomisticBiasRowBinding({umbrella.name: np.asarray([c])})
                for c in (1.1, 2.0)
            ),
            "unbound-history",
        )


def test_tampered_checkpoint_history_or_digest_is_refused_at_read(tmp_path: Path) -> None:
    _, umbrella, hill, _, dynamics, genuine, other = _frozen_setup()
    table = _table(dynamics, _windows(umbrella, ({hill.name: genuine.declaration},) * 2))
    runtime = _plan(
        dynamics, table, 2, exchange=False, run_id="frozen-checkpoint"
    ).prepare(dynamics)
    bindings = _bind_history(dynamics, hill.name, genuine.state)
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.1, 2.0), term_states=bindings),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(3),
    )
    result = AtomisticMultistateSegmentPlan(
        runtime, 2, 0, 0, runtime.initial_continuation_id
    ).run(initial)
    assert bool(result.successful)
    checkpoint_plan = AtomisticMultistateCheckpointPlan(runtime, 2)
    path = tmp_path / "frozen.phx"
    written = write_atomistic_multistate_checkpoint(path, checkpoint_plan, result)
    # Control: the untampered checkpoint authenticates.
    restored = read_atomistic_multistate_checkpoint(path, checkpoint_plan, initial)
    assert restored.payload_id == written.payload_id
    assert restored.segment.bias_history_digests() == (genuine.digest,)

    manifest, arrays = read_array_archive(path)
    assert manifest["bias_history_digests"] == [genuine.digest]
    manifest = dict(manifest)
    manifest.pop("arrays")
    paths = manifest["result"]["paths"]
    heights = [
        name
        for name, leaf_path in zip(manifest["result"]["arrays"], paths, strict=True)
        if leaf_path.endswith(".history.heights")
    ]
    assert len(heights) == 1

    # Same replicated content on every lane, kernel heights doubled.
    tampered = dict(arrays)
    tampered[heights[0]] = 2.0 * np.asarray(arrays[heights[0]])
    history_path = tmp_path / "tampered-history.phx"
    write_array_archive(history_path, manifest=manifest, arrays=tampered)
    with pytest.raises(ValueError, match="does not authenticate its frozen declaration"):
        read_atomistic_multistate_checkpoint(history_path, checkpoint_plan, initial)

    # Untouched content, counterfeit manifest digest.
    digest_path = tmp_path / "tampered-digest.phx"
    write_array_archive(
        digest_path,
        manifest={**manifest, "bias_history_digests": [other.digest]},
        arrays=arrays,
    )
    with pytest.raises(ValueError, match="does not match its digests"):
        read_atomistic_multistate_checkpoint(digest_path, checkpoint_plan, initial)


# ------------------------------------- 4. adaptive evidence through analysis


def test_adaptive_history_evidence_survives_segment_checkpoint_and_analysis(
    tmp_path: Path,
) -> None:
    system = _system()
    variables = _distance(system)
    umbrella = _umbrella(variables, 2.5)
    hill_plan = _hill_plan(height=0.2, capacity=16, interval=1)
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (umbrella, hill), BAOABLangevinPlan(0.05, 1.0))
    adaptive = {hill.name: "adaptive"}
    table = _table(dynamics, _windows(umbrella, (adaptive, adaptive)))
    runtime = _plan(
        dynamics, table, 2, exchange=False, run_id="adaptive-evidence"
    ).prepare(dynamics)
    assert runtime.history_adaptive
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.1, 2.0), seed=7),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(8),
    )
    capacity = 4
    replicas = 2
    result = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    ).run(initial)

    assert bool(result.successful) and int(result.count) == capacity
    assert result.bias_history_term_ids == (hill.term_id,)
    np.testing.assert_array_equal(
        np.asarray(result.bias_deposited), np.ones((capacity, 1))
    )
    # Interval 1: one kernel per walker per draw.
    np.testing.assert_array_equal(
        np.asarray(result.bias_history_revision)[:, 0],
        replicas * np.arange(1, capacity + 1),
    )
    assert bool(np.all(np.asarray(result.bias_adapting)))
    assert not bool(np.any(np.asarray(result.sample_active)))
    assert not bool(np.any(np.asarray(result.coverage)))
    assert not bool(np.any(np.asarray(result.bias_exhausted)))
    np.testing.assert_array_equal(np.asarray(result.bias_owner_keys), 0)

    # The history content: the last event deposited each walker's final r.
    successor = result.successor_state.dynamics
    history = _lane_history(runtime, successor.term_states, hill.name, 0)
    assert int(history.written) == replicas * capacity
    assert int(history.deposits) == capacity
    np.testing.assert_array_equal(np.asarray(history.valid)[: replicas * capacity], True)
    np.testing.assert_array_equal(np.asarray(history.owners)[: replicas * capacity], 0)
    np.testing.assert_allclose(
        np.asarray(history.centers)[replicas * (capacity - 1) : replicas * capacity, 0],
        _radii(successor.kinematics.positions),
        rtol=1e-12,
    )
    # The history changed at the last boundary: the refreshed caches are fresh.
    _assert_fresh_forces(dynamics, table, successor)

    digests = result.bias_history_digests()
    assert digests == (bias_history_digest(history),)
    checkpoint_plan = AtomisticMultistateCheckpointPlan(runtime, capacity)
    path = tmp_path / "adaptive.phx"
    written = write_atomistic_multistate_checkpoint(path, checkpoint_plan, result)
    restored = read_atomistic_multistate_checkpoint(path, checkpoint_plan, initial)
    assert restored.payload_id == written.payload_id
    for name in (
        "bias_adapting",
        "bias_deposited",
        "bias_exhausted",
        "bias_owner_keys",
        "bias_history_revision",
        "sample_active",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(restored.segment, name)), np.asarray(getattr(result, name))
        )
    assert restored.segment.bias_history_declarations == ("adaptive",)
    assert restored.segment.bias_history_digests() == digests

    # Adaptive-only segments retain evidence but supply no equilibrium sample;
    # the analysis owner must refuse an empty free-energy dataset.
    with pytest.raises(ValueError):
        phx.uq.reduced_potential_dataset_from_multistate(restored.segment)

    flipped = eqx.tree_at(
        lambda value: value.sample_active, result, jnp.ones_like(result.sample_active)
    )
    with pytest.raises(ValueError, match="not equilibrium samples"):
        phx.uq.reduced_potential_dataset_from_multistate(flipped)
    with pytest.raises(ValueError, match="padding is invalid"):
        write_atomistic_multistate_checkpoint(
            tmp_path / "flipped.phx", checkpoint_plan, flipped
        )


# ------------------------------------------------ 5. terminal exhaustion


def test_terminal_history_exhaustion_ends_the_segment_at_the_first_due_event(
    tmp_path: Path,
) -> None:
    system = _system()
    variables = _distance(system)
    umbrella = _umbrella(variables, 2.5)
    replicas = 2
    # One slot cannot hold the all-or-nothing event of two walkers.
    hill_plan = _hill_plan(height=0.2, capacity=1, interval=2)
    assert hill_plan.capacity < replicas
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (umbrella, hill), BAOABLangevinPlan(0.05, 1.0))
    adaptive = {hill.name: "adaptive"}
    table = _table(dynamics, _windows(umbrella, (adaptive, adaptive)))
    runtime = _plan(
        dynamics, table, replicas, exchange=False, run_id="exhaustion"
    ).prepare(dynamics)
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.1, 2.0), seed=9),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(10),
    )
    capacity = 4
    result = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    ).run(initial)

    # Committed iteration n (1-based) is due when n % interval == 0; draw
    # index n - 1 is the first refused attempt.
    first_due = next(n for n in range(1, capacity + 1) if bool(hill_plan.due(n))) - 1
    assert first_due == 1
    expected_exhausted = np.arange(capacity) == first_due
    np.testing.assert_array_equal(np.asarray(result.bias_exhausted), expected_exhausted)
    np.testing.assert_array_equal(
        np.asarray(result.iteration_valid), np.arange(capacity) < first_due
    )
    assert int(result.count) == first_due
    assert not bool(result.successful)
    assert int(result.successor_state.iteration_index) == first_due
    assert not bool(np.any(np.asarray(result.bias_deposited)))
    np.testing.assert_array_equal(
        np.asarray(result.bias_history_revision)[:, 0],
        np.where(np.arange(capacity) < first_due, 0, -1),
    )
    history = _lane_history(
        runtime, result.successor_state.dynamics.term_states, hill.name, 0
    )
    assert int(history.written) == 0
    assert not bool(np.any(np.asarray(history.valid)))
    assert not bool(history.exhausted)

    checkpoint_plan = AtomisticMultistateCheckpointPlan(runtime, capacity)
    path = tmp_path / "exhausted.phx"
    write_atomistic_multistate_checkpoint(path, checkpoint_plan, result)
    restored = read_atomistic_multistate_checkpoint(path, checkpoint_plan, initial)
    np.testing.assert_array_equal(
        np.asarray(restored.segment.bias_exhausted), expected_exhausted
    )
    assert int(restored.segment.count) == first_due


# ------------------------------------------------------------- 6. ownership


def _hill_rows(
    hill: Any,
    declaration: str,
    owner_keys: tuple[int, ...],
) -> tuple[AtomisticBiasRowBinding, ...]:
    return tuple(
        AtomisticBiasRowBinding(
            {}, owner_keys={hill.name: key}, histories={hill.name: declaration}
        )
        for key in owner_keys
    )


def _swap_labels(labels: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    updated = labels.copy()
    for start in np.flatnonzero(accepted):
        updated[[start, start + 1]] = updated[[start + 1, start]]
    return updated


def test_walker_owned_history_owner_keys_are_lane_indices() -> None:
    system = _system()
    variables = _distance(system)
    replicas = 3
    hill_plan = _hill_plan(
        height=0.2,
        capacity=32,
        ownership=BiasHistoryOwnership.WALKER,
        owner_count=replicas,
    )
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (hill,), BAOABLangevinPlan(0.05, 1.0))
    # Identical rows: a walker-owned history makes every row's reduced
    # potential of a replica identical, so every attempted swap has
    # log-acceptance exactly zero and is accepted.
    table = _table(dynamics, _hill_rows(hill, "adaptive", (0, 0, 0)))
    runtime = _plan(
        dynamics, table, replicas, exchange=True, run_id="walker-owners"
    ).prepare(dynamics)
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.0, 1.4, 1.8), seed=12),
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jax.random.key(13),
    )
    capacity = 4
    result = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    ).run(initial)

    assert bool(result.successful)
    attempted = np.asarray(result.exchange_attempted)
    np.testing.assert_array_equal(np.asarray(result.exchange_accepted), attempted)
    np.testing.assert_array_equal(np.asarray(result.exchange_log_acceptance), 0.0)
    labels = np.arange(replicas)
    for draw in range(capacity):
        labels = _swap_labels(labels, attempted[draw])
    np.testing.assert_array_equal(
        np.asarray(result.successor_state.state_at_replica), labels
    )
    assert not np.array_equal(labels, np.arange(replicas))
    position = runtime.bias_term_ids.index(hill.term_id)
    keys = np.asarray(result.bias_owner_keys)[:, :, position]
    np.testing.assert_array_equal(keys, np.broadcast_to(np.arange(replicas), keys.shape))
    # Deposits are tagged with the depositing lane, in stable lane order.
    history = _lane_history(
        runtime, result.successor_state.dynamics.term_states, hill.name, 0
    )
    written = int(history.written)
    assert written == replicas * capacity
    np.testing.assert_array_equal(
        np.asarray(history.owners)[:written], np.tile(np.arange(replicas), capacity)
    )


def test_state_owned_history_owner_keys_follow_the_row_owner_of_each_label() -> None:
    system = _system()
    variables = _distance(system)
    replicas = 3
    owner_of_row = np.asarray([2, 0, 1])  # a derangement: key != label always
    hill_plan = _hill_plan(
        height=1.0e-3,
        capacity=8,
        ownership=BiasHistoryOwnership.STATE,
        owner_count=replicas,
    )
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (hill,), BAOABLangevinPlan(0.05, 1.0))
    frozen = _frozen(hill_plan, variables, (1.0, 1.4, 1.8), owners=(0, 1, 2))
    table = _table(
        dynamics,
        _hill_rows(hill, frozen.declaration, tuple(int(key) for key in owner_of_row)),
    )
    runtime = _plan(
        dynamics, table, replicas, exchange=True, run_id="state-owners"
    ).prepare(dynamics)
    bindings = _bind_history(dynamics, hill.name, frozen.state)
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.0, 1.4, 1.8), term_states=bindings, seed=14),
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jax.random.key(15),
    )
    capacity = 6
    result = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    ).run(initial)

    assert bool(result.successful)
    assert bool(np.all(np.asarray(result.sample_active)))
    # Near-degenerate rows (1e-3 hills): swaps change the labels.
    assert bool(np.any(np.asarray(result.exchange_accepted)))
    labels = np.asarray(result.state_at_replica)
    assert not bool(np.all(labels == np.arange(replicas)))
    position = runtime.bias_term_ids.index(hill.term_id)
    keys = np.asarray(result.bias_owner_keys)[:, :, position]
    np.testing.assert_array_equal(keys, owner_of_row[labels])
    assert bool(np.all(keys != labels))
    assert bool(np.any(keys != np.arange(replicas)[None, :]))
    successor_bindings = result.successor_state.dynamics.term_states
    np.testing.assert_array_equal(
        np.asarray(
            [
                _lane_bias_state(runtime, successor_bindings, hill.name, lane).owner_key
                for lane in range(replicas)
            ]
        ),
        owner_of_row[np.asarray(result.successor_state.state_at_replica)],
    )


def test_state_owned_deposits_are_felt_by_the_depositing_row() -> None:
    system = _system()
    variables = _distance(system)
    replicas = 3
    height = 0.4
    owner_of_row = np.asarray([2, 0, 1])
    hill_plan = _hill_plan(
        height=height,
        capacity=8,
        ownership=BiasHistoryOwnership.STATE,
        owner_count=replicas,
    )
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (hill,), _quasistatic())
    table = _table(
        dynamics, _hill_rows(hill, "adaptive", tuple(int(key) for key in owner_of_row))
    )
    runtime = _plan(
        dynamics, table, replicas, exchange=False, run_id="state-deposits"
    ).prepare(dynamics)
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.0, 2.0, 3.0), seed=16),
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jax.random.key(17),
    )
    iteration = runtime.iterate(initial)

    assert bool(iteration.successful)
    np.testing.assert_array_equal(np.asarray(iteration.bias_deposited), [True])
    labels = np.asarray(iteration.origin_state)
    np.testing.assert_array_equal(labels, [0, 1, 2])
    history = _lane_history(runtime, iteration.state.dynamics.term_states, hill.name, 0)
    # ORIGIN labelling: each kernel carries the owner key of the row that sampled it.
    np.testing.assert_array_equal(
        np.asarray(history.owners)[:replicas], owner_of_row[labels]
    )
    # Widely separated walkers: each row feels exactly its own new kernel,
    # deposited at the walker's own r, so U = U0(r) + height.
    radius = _radii(iteration.state.dynamics.kinematics.positions)
    np.testing.assert_allclose(
        np.asarray(iteration.state.dynamics.force.potential_energy),
        0.5 * TETHER * radius**2 + height,
        rtol=1e-10,
    )
    _assert_fresh_forces(dynamics, table, iteration.state.dynamics)


# ------------------------------------- 7. owner-resolved analysis admission


def test_frozen_walker_history_rows_evaluate_their_declared_owner(
    tmp_path: Path,
) -> None:
    system = _system()
    variables = _distance(system)
    replicas = 2
    height, width = 0.5, 0.3
    # Row k declares owner owner_of_row[k]; a derangement, so a lane-owned
    # evaluation (owner = replica lane) would differ at every column.
    owner_of_row = np.asarray([1, 0], dtype=np.int32)
    hill_plan = _hill_plan(
        height=height,
        widths=width,
        capacity=4,
        ownership=BiasHistoryOwnership.WALKER,
        owner_count=replicas,
    )
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (hill,), _quasistatic())
    kernel_centers = np.asarray([1.0, 2.0])
    frozen = _frozen(hill_plan, variables, tuple(kernel_centers), owners=(0, 1))
    table = _table(
        dynamics,
        _hill_rows(hill, frozen.declaration, tuple(int(key) for key in owner_of_row)),
    )
    runtime = _plan(
        dynamics, table, replicas, exchange=True, run_id="frozen-walker-rows"
    ).prepare(dynamics)
    assert runtime.qualification.fixed_row_hamiltonians
    position = runtime.bias_term_ids.index(hill.term_id)
    assert runtime.bias_owner_resolutions[position] == "row"
    bindings = _bind_history(dynamics, hill.name, frozen.state)
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.0, 2.0), term_states=bindings, seed=31),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(32),
    )
    iteration = runtime.iterate(initial)

    assert bool(iteration.successful) and bool(iteration.equilibrium_sample)
    # Independent reduced potentials: row k is U0 + the frozen kernels of its
    # declared owner, at every replica's configuration.
    radius = _radii(iteration.state.dynamics.kinematics.positions)
    beta = float(np.asarray(table.beta)[0])
    expected = np.asarray(
        [
            beta
            * (
                0.5 * TETHER * radius**2
                + _hill_energy(radius, kernel_centers[[owner]], height, width)
            )
            for owner in owner_of_row
        ]
    )
    np.testing.assert_allclose(
        np.asarray(iteration.reduced_potentials), expected, rtol=1e-12, atol=1e-14
    )
    labels = np.asarray(iteration.state_at_replica_after)
    np.testing.assert_array_equal(
        np.asarray(iteration.bias_owner_keys)[:, position], owner_of_row[labels]
    )
    _assert_fresh_forces(dynamics, table, iteration.state.dynamics)

    capacity = 3
    result = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    ).run(initial)
    assert bool(result.successful) and bool(np.all(np.asarray(result.sample_active)))
    np.testing.assert_array_equal(
        np.asarray(result.bias_owner_keys)[:, :, position],
        owner_of_row[np.asarray(result.state_at_replica)],
    )
    samples = AtomisticMultistateEquilibriumSamples(result)
    assert samples.bias_history_digests == (frozen.digest,)
    assert samples.qualification_id == runtime.qualification.qualification_id
    dataset = phx.uq.reduced_potential_dataset_from_multistate(result)
    np.testing.assert_array_equal(
        np.asarray(dataset.values), np.asarray(samples.reduced_potentials)
    )
    assert dataset.potential_ids == table.potential_ids

    # Owner evidence that is not the row's declared owner is refused by the
    # admission, the UQ conversion and the checkpoint.
    swapped = eqx.tree_at(
        lambda value: value.bias_owner_keys,
        result,
        1 - jnp.asarray(result.bias_owner_keys),
    )
    with pytest.raises(ValueError, match="row's declared owner key"):
        AtomisticMultistateEquilibriumSamples(swapped)
    with pytest.raises(ValueError, match="row's declared owner key"):
        phx.uq.reduced_potential_dataset_from_multistate(swapped)
    with pytest.raises(ValueError, match="row's declared owner key"):
        write_atomistic_multistate_checkpoint(
            tmp_path / "wrong-owner.phx",
            AtomisticMultistateCheckpointPlan(runtime, capacity),
            swapped,
        )
    assert not (tmp_path / "wrong-owner.phx").exists()


def test_adaptive_walker_history_rows_are_never_analysis_rows() -> None:
    system = _system()
    variables = _distance(system)
    replicas = 2
    hill_plan = _hill_plan(
        height=0.2,
        capacity=16,
        ownership=BiasHistoryOwnership.WALKER,
        owner_count=replicas,
    )
    hill = _hill(variables, hill_plan)
    dynamics = _dynamics(system, (hill,), BAOABLangevinPlan(0.05, 1.0))
    # A lane-owned history ignores row owner keys: declaring one is refused.
    selecting = _table(dynamics, _hill_rows(hill, "adaptive", (0, 1)))
    with pytest.raises(ValueError, match="cannot select an owner key"):
        _plan(
            dynamics, selecting, replicas, exchange=False, run_id="walker-select"
        ).prepare(dynamics)

    table = _table(dynamics, _hill_rows(hill, "adaptive", (0, 0)))
    plan = _plan(dynamics, table, replicas, exchange=False, run_id="walker-adaptive")
    assert not plan.qualification.fixed_row_hamiltonians
    with pytest.raises(ValueError, match="fixed-Hamiltonian rows"):
        plan.qualification.require_fixed_row_hamiltonians("A distribution objective")
    runtime = plan.prepare(dynamics)
    position = runtime.bias_term_ids.index(hill.term_id)
    assert runtime.bias_owner_resolutions[position] == "lane"
    result = AtomisticMultistateSegmentPlan(
        runtime, 2, 0, 0, runtime.initial_continuation_id
    ).run(
        runtime.initialize(
            _lanes(dynamics, table, (1.0, 1.6), seed=41),
            jnp.asarray([0, 1], dtype=jnp.int32),
            jax.random.key(42),
        )
    )
    assert bool(result.successful)
    assert not bool(np.any(np.asarray(result.sample_active)))
    np.testing.assert_array_equal(
        np.asarray(result.bias_owner_keys)[:, :, position], [[0, 1], [0, 1]]
    )
    # Even with its adaptation evidence removed, a lane-owned history makes
    # each row's bias depend on the sample column: no MBAR row is admitted.
    relabeled = eqx.tree_at(
        lambda value: (value.sample_active, value.bias_adapting),
        result,
        (
            jnp.ones_like(result.sample_active),
            jnp.zeros_like(result.bias_adapting),
        ),
    )
    with pytest.raises(ValueError, match="walker-owned and adaptive"):
        AtomisticMultistateEquilibriumSamples(relabeled)
    with pytest.raises(ValueError, match="walker-owned and adaptive"):
        phx.uq.reduced_potential_dataset_from_multistate(relabeled)


def test_single_state_draws_are_admitted_with_their_fixed_reduced_potentials() -> None:
    system = _system()
    variables = _distance(system)
    stiffness, center = 3.0, 1.2
    umbrella = _umbrella(variables, stiffness)
    dynamics = _dynamics(system, (umbrella,), _quasistatic())
    table = _table(
        dynamics, (AtomisticBiasRowBinding({umbrella.name: np.asarray([center])}),)
    )
    runtime = _plan(dynamics, table, 2, exchange=False, run_id="single-state").prepare(
        dynamics
    )
    initial = runtime.initialize(
        _lanes(dynamics, table, (1.0, 1.5), seed=51),
        jnp.asarray([0, 0], dtype=jnp.int32),
        jax.random.key(52),
    )
    result = AtomisticMultistateSegmentPlan(
        runtime, 1, 0, 0, runtime.initial_continuation_id
    ).run(initial)
    assert bool(result.successful)
    samples = AtomisticMultistateEquilibriumSamples(result)
    radius = _radii(result.successor_state.dynamics.kinematics.positions)
    beta = float(np.asarray(table.beta)[0])
    np.testing.assert_allclose(
        np.asarray(samples.reduced_potentials),
        [beta * (0.5 * TETHER * radius**2 + 0.5 * stiffness * (radius - center) ** 2)],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(np.asarray(samples.sample_active), True)
    np.testing.assert_array_equal(np.asarray(samples.origin_state), [0, 0])
    assert samples.state_ids == table.state_ids
    assert samples.producer_id == runtime.prepared_id
    # MBAR itself needs two states; the admission does not.
    with pytest.raises(ValueError, match="at least two states"):
        phx.uq.reduced_potential_dataset_from_multistate(result)


def test_captured_draw_configurations_reproduce_their_reduced_potentials(
    tmp_path: Path,
) -> None:
    system = _system()
    variables = _distance(system)
    stiffness = 4.0
    centers = np.asarray([1.1, 1.6])
    umbrella = _umbrella(variables, stiffness)
    dynamics = _dynamics(system, (umbrella,), BAOABLangevinPlan(0.05, 1.0))
    table = _table(
        dynamics,
        tuple(
            AtomisticBiasRowBinding({umbrella.name: np.asarray([center])})
            for center in centers
        ),
    )
    runtime = _plan(dynamics, table, 2, exchange=True, run_id="captured-draws").prepare(
        dynamics
    )
    capacity = 5
    capture = AtomisticDrawConfigurationCapture(max_bytes=1 << 20)
    # Two atoms, no cell: per draw 2*3*(8+4) + 2*4 bytes, R = 2.
    assert capture.required_bytes(runtime, capacity) == capacity * 2 * (72 + 8)
    with pytest.raises(ValueError, match="max_bytes"):
        AtomisticMultistateSegmentPlan(
            runtime,
            capacity,
            0,
            0,
            runtime.initial_continuation_id,
            configurations=AtomisticDrawConfigurationCapture(max_bytes=64),
        )
    initial = runtime.initialize(
        _lanes(dynamics, table, tuple(centers), seed=61),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jax.random.key(62),
    )
    captured_plan = AtomisticMultistateSegmentPlan(
        runtime,
        capacity,
        0,
        0,
        runtime.initial_continuation_id,
        configurations=capture,
    )
    plain_plan = AtomisticMultistateSegmentPlan(
        runtime, capacity, 0, 0, runtime.initial_continuation_id
    )
    # The capture policy is part of the segment identity.
    assert captured_plan.segment_id != plain_plan.segment_id
    result = captured_plan.run(initial)
    plain = plain_plan.run(initial)
    assert bool(result.successful) and bool(plain.successful)
    assert plain.draw_positions is None and plain.draw_species is None
    # Capture does not perturb the run.
    np.testing.assert_array_equal(
        np.asarray(result.reduced_potentials), np.asarray(plain.reduced_potentials)
    )
    positions = np.asarray(result.draw_positions)
    assert positions.shape == (capacity, 2, 2, 3)
    assert result.draw_cell_vectors is None
    np.testing.assert_array_equal(np.asarray(result.draw_image_counts), 0)
    np.testing.assert_array_equal(
        np.asarray(result.draw_species),
        np.broadcast_to(np.asarray(system.plan.atom_type_ids), (capacity, 2, 2)),
    )
    # Independent control: every draw's reduced potentials recomputed from its
    # captured configuration, u_k = beta (U0 + V_k)(r).
    radius = np.linalg.norm(positions[:, :, 1] - positions[:, :, 0], axis=-1)
    assert float(np.ptp(radius)) > 0.0
    beta = float(np.asarray(table.beta)[0])
    expected = beta * (
        0.5 * TETHER * radius[:, None, :] ** 2
        + 0.5 * stiffness * (radius[:, None, :] - centers[None, :, None]) ** 2
    )
    np.testing.assert_allclose(
        np.asarray(result.reduced_potentials), expected, rtol=1e-12, atol=1e-12
    )
    samples = AtomisticMultistateEquilibriumSamples(result)
    np.testing.assert_array_equal(
        np.asarray(samples.draw_positions), positions.reshape((capacity * 2, 2, 3))
    )
    assert samples.draw_cell_vectors is None
    assert AtomisticMultistateEquilibriumSamples(plain).draw_positions is None

    capturing = AtomisticMultistateCheckpointPlan(
        runtime, capacity, configurations=capture
    )
    path = tmp_path / "captured.phx"
    written = write_atomistic_multistate_checkpoint(path, capturing, result)
    restored = read_atomistic_multistate_checkpoint(path, capturing, initial)
    assert restored.payload_id == written.payload_id
    np.testing.assert_array_equal(np.asarray(restored.segment.draw_positions), positions)
    # A layout without the policy cannot archive or restore captured draws.
    with pytest.raises(ValueError, match="does not declare"):
        write_atomistic_multistate_checkpoint(
            tmp_path / "undeclared.phx",
            AtomisticMultistateCheckpointPlan(runtime, capacity),
            result,
        )
    with pytest.raises(ValueError):
        read_atomistic_multistate_checkpoint(
            path, AtomisticMultistateCheckpointPlan(runtime, capacity), initial
        )
    # Captured positions are archived evidence authenticated by the payload
    # identity: a genuine manifest over altered positions is refused.
    shifted = eqx.tree_at(
        lambda value: value.draw_positions,
        result,
        jnp.asarray(result.draw_positions).at[0, 0, 1, 0].add(0.1),
    )
    shifted_path = tmp_path / "shifted.phx"
    write_atomistic_multistate_checkpoint(shifted_path, capturing, shifted)
    manifest, _ = read_array_archive(path)
    _, shifted_arrays = read_array_archive(shifted_path)
    manifest = dict(manifest)
    manifest.pop("arrays")
    tampered = tmp_path / "tampered.phx"
    write_array_archive(tampered, manifest=manifest, arrays=shifted_arrays)
    with pytest.raises(ValueError, match="payload identity is corrupt"):
        read_atomistic_multistate_checkpoint(tampered, capturing, initial)
