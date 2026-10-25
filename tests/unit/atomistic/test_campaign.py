#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Staged heterogeneous atomistic campaigns over the native execution owners.

Every scenario uses tiny analytic periodic Hamiltonians (the soft C3 pair
repulsion ``10 (1 - d/r_c)^4`` of the shared dynamics fixtures) so physical
checks are independent of the campaign: relaxed pairs leave the cutoff
(zero force), shared labels equal an independent native evaluation at the
published configuration, and rebucketing reproduces the step and key of the
refused attempt.  Interruption tests abandon the in-memory state at each
transaction boundary and resume from the durable envelope with a freshly
constructed plan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._identity import IdentityWords
from phydrax.atomistic import _campaign as campaign_module
from phydrax.atomistic._campaign import (
    AbstractAtomisticCampaignSink,
    atomistic_hamiltonian_revision,
    AtomisticCampaignPlan,
    AtomisticCampaignReason,
    AtomisticCampaignStore,
    AtomisticCampaignTask,
    AtomisticContinuationPolicy,
    AtomisticDirectorySink,
    AtomisticDynamicsContinuation,
    AtomisticDynamicsStageOperation,
    AtomisticEntryPolicy,
    AtomisticEvaluationKey,
    AtomisticLabelStageOperation,
    AtomisticOutputReceipt,
    AtomisticRelaxationStageOperation,
    AtomisticRevisionPolicy,
    AtomisticStageOutcome,
    AtomisticStagePlan,
    AtomisticStageRoute,
    AtomisticStageRouting,
    AtomisticTaskPacket,
    AtomisticTaskStatus,
)
from phydrax.atomistic._cell_dynamics import MTKBarostatPlan, MTKNPTPlan
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
)
from phydrax.atomistic._potential import (
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from phydrax.atomistic._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticTermEvaluation,
)
from phydrax.atomistic._relaxation import (
    AtomisticCartesianChart,
    AtomisticRelaxationConvergence,
    AtomisticRelaxationPlan,
)
from phydrax.atomistic._thermal import BAOABLangevinPlan, NoseHooverChainParameters
from phydrax.optim import FIRE
from tests._support.atomistic_bundles import bundle_state, role_capacity
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    DirectedPairTerm,
    measure,
    observed_role_storage,
    periodic_cell,
    prepare_dynamics,
    prepare_potential,
    ROLE_CAPACITIES,
    row,
    table,
)


CUTOFF = 1.5
FRICTION = 0.1
TEMPERATURE = 0.01


def _term() -> DirectedPairTerm:
    return DirectedPairTerm("soft", CUTOFF, profile="smooth", strength=10.0)


def _system(atoms: int, box: float) -> Any:
    return atomistic_system(
        list(range(atoms)), [1.0] * atoms, cell=periodic_cell(box * np.eye(3))
    )


def _nvt(system: Any, *, roles: tuple[Any, ...] = ROLE_CAPACITIES, steps: int = 5) -> Any:
    dynamics = prepare_dynamics(
        system, (_term(),), BAOABLangevinPlan(0.01, FRICTION), role_capacities=roles
    )
    return AtomisticDynamicsStageOperation(
        dynamics,
        (row(measure(system), "nvt", temperature=TEMPERATURE),),
        steps_per_round=steps,
    )


def _npt(system: Any, *, pressure: float = 0.2) -> Any:
    integrator = MTKNPTPlan(
        0.005,
        MTKBarostatPlan(0.5),
        NoseHooverChainParameters(0.5),
        NoseHooverChainParameters(0.5),
    )
    dynamics = prepare_dynamics(system, (_term(),), integrator)
    return AtomisticDynamicsStageOperation(
        dynamics,
        (row(measure(system), "npt", temperature=TEMPERATURE, pressure=pressure),),
        steps_per_round=5,
    )


def _relax(system: Any, lanes: int) -> Any:
    program = prepare_potential(system, (_term(),))
    plan = AtomisticRelaxationPlan(
        program,
        AtomisticCartesianChart(),
        method=FIRE(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-8, maximum_evaluations=400
        ),
        group_count=lanes,
    )
    return AtomisticRelaxationStageOperation(plan.prepare(), rounds_per_segment=25)


def _label(system: Any) -> Any:
    return AtomisticLabelStageOperation(
        prepare_potential(system, (_term(),)), stress=True
    )


def _packet(positions: Any, *, momenta: Any = None) -> AtomisticTaskPacket:
    return AtomisticTaskPacket(
        positions=np.asarray(positions, dtype=np.float64),
        momenta=None if momenta is None else np.asarray(momenta, dtype=np.float64),
    )


def _tight_positions() -> tuple[np.ndarray, np.ndarray]:
    """A pair 2.2 apart closing at unit relative speed (cubic L = 6)."""
    return (
        np.asarray([[2.0, 3.0, 3.0], [4.2, 3.0, 3.0]]),
        np.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
    )


def _exact_roles(system: Any) -> Any:
    """Role storage holding exactly the initial (route-free) geometry."""
    positions, momenta = _tight_positions()
    probe = prepare_dynamics(system, (_term(),), BAOABLangevinPlan(0.01, FRICTION))
    rows = (row(measure(system), "nvt", temperature=TEMPERATURE),)
    state = probe.initialize_state(
        jnp.asarray(positions),
        table(probe, *rows),
        momentum=jnp.asarray(momenta),
        key=jr.key(0),
    )
    return observed_role_storage(state)


def _outputs(sink: AtomisticDirectorySink) -> dict[tuple[str, str], Any]:
    return {(output.task_id, output.stage): output for output in sink.outputs().values()}


def _run_rounds(
    prepared: Any, state: Any, sink: Any, store: Any, rounds: int
) -> tuple[Any, list[Any]]:
    evidence = []
    for _ in range(rounds):
        if state.complete:
            break
        state = prepared.run_round(state, sink, store)
        evidence.append(state.evidence)
    return state, evidence


def _assert_exactly_once(state: Any, sink: AtomisticDirectorySink, expected: int) -> None:
    keys = [receipt.semantic_key for receipt in state.receipts]
    assert len(keys) == len(set(keys)) == expected
    assert set(sink.outputs()) == set(keys)


# -- the mixed relax -> NVT -> NPT -> label campaign ------------------------------------


def _mixed_plan() -> tuple[AtomisticCampaignPlan, dict[str, Any]]:
    pair = _system(2, 4.0)
    quad = _system(4, 5.0)
    tight = _system(2, 6.0)
    refuse = _system(2, 6.0)
    exact = _exact_roles(tight)
    systems = {"pair": pair, "quad": quad, "tight": tight, "refuse": refuse}
    stages = (
        AtomisticStagePlan(
            "relax",
            {"pair": _relax(pair, 2), "quad": _relax(quad, 2)},
            routing=AtomisticStageRouting(
                converged=AtomisticStageRoute("nvt", publish=True),
                exhausted=AtomisticStageRoute("nvt", publish=True),
            ),
            lane_capacity=2,
        ),
        AtomisticStagePlan(
            "nvt",
            {
                "pair": _nvt(pair),
                "quad": _nvt(quad),
                "tight": _nvt(tight, roles=(exact, role_capacity(4096, 512))),
                "refuse": _nvt(refuse, roles=(exact,)),
            },
            budget=100,
            routing=AtomisticStageRouting(
                completed=AtomisticStageRoute("npt", publish=True)
            ),
            lane_capacity=2,
            queue_capacity=2,
        ),
        AtomisticStagePlan(
            "npt",
            {name: _npt(system) for name, system in systems.items()},
            budget=10,
            lane_capacity=2,
            queue_capacity=2,
            # A changed integrator and ensemble: an explicit new run.
            continuation=AtomisticContinuationPolicy.RESTART,
        ),
        AtomisticStagePlan(
            "label",
            {name: _label(system) for name, system in systems.items()},
            lane_capacity=2,
            queue_capacity=1,
            entry=AtomisticEntryPolicy.SHARE,
        ),
    )
    tight_positions, tight_momenta = _tight_positions()
    quad_positions = np.asarray(
        [[2.0, 2.0, 2.0], [2.6, 2.0, 2.0], [2.0, 2.7, 2.0], [2.3, 2.3, 2.8]]
    )
    tasks = (
        AtomisticCampaignTask(
            "pair-a",
            "pair",
            _packet([[1.6, 2.0, 2.0], [2.4, 2.0, 2.0]]),
            material_id="pair",
        ),
        AtomisticCampaignTask(
            "pair-b",
            "pair",
            _packet([[2.0, 1.5, 2.0], [2.0, 2.3, 2.1]]),
            material_id="pair",
        ),
        AtomisticCampaignTask(
            "pair-fail",
            "pair",
            _packet([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            material_id="pair",
        ),
        AtomisticCampaignTask(
            "quad-a", "quad", _packet(quad_positions), material_id="quad"
        ),
        AtomisticCampaignTask(
            "tight",
            "tight",
            _packet(tight_positions, momenta=tight_momenta),
            material_id="pair-wide",
            entry_stage="nvt",
        ),
        AtomisticCampaignTask(
            "refuse",
            "refuse",
            _packet(tight_positions, momenta=tight_momenta),
            material_id="pair-wide",
            entry_stage="nvt",
        ),
    )
    plan = AtomisticCampaignPlan(
        "mixed-campaign", stages, tasks, max_pending_publications=2
    )
    return plan, systems


def test_mixed_relax_nvt_npt_label_campaign_end_to_end(tmp_path: Path) -> None:
    plan, systems = _mixed_plan()
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out", accept_per_round=1)
    store = AtomisticCampaignStore(tmp_path / "campaign.ckpt")
    state, evidence = _run_rounds(prepared, prepared.start(jr.key(11)), sink, store, 600)
    assert state.complete
    outputs = _outputs(sink)
    # Coincident atoms: a scientific relaxation failure, terminal and published.
    failed = outputs[("pair-fail", "relax")]
    assert failed.outcome is AtomisticStageOutcome.FAILED
    assert failed.evaluation_key is None
    # An exhausted declared role ladder is a capacity refusal, not a failure.
    refused = outputs[("refuse", "nvt")]
    assert refused.outcome is AtomisticStageOutcome.CAPACITY_REFUSED
    assert refused.campaign_reasons & int(AtomisticCampaignReason.LADDER_EXHAUSTED)
    assert refused.campaign_reasons & int(AtomisticCampaignReason.ROLE_CAPACITY)
    # The capacity-limited task was rebucketed and completed its full budget.
    tight_nvt = outputs[("tight", "nvt")]
    assert tight_nvt.outcome is AtomisticStageOutcome.COMPLETED
    assert tight_nvt.counters["consumed"] == 100
    assert tight_nvt.campaign_reasons & int(AtomisticCampaignReason.ROLE_CAPACITY)
    assert any(item.capacity_transfers for item in evidence)
    families = {"pair-a": "pair", "pair-b": "pair", "quad-a": "quad", "tight": "tight"}
    walkers = prepared.terminal_packets(state)
    npt = plan.stages[2].operations
    for task, family in families.items():
        assert outputs[(task, "nvt")].outcome is AtomisticStageOutcome.COMPLETED
        label = outputs[(task, "label")]
        assert label.outcome is AtomisticStageOutcome.COMPLETED
        # The NPT evaluation served the label request (same Hamiltonian,
        # revision, conditions and quantities): no second evaluation.
        assert label.campaign_reasons & int(AtomisticCampaignReason.SHARED_EVALUATION)
        _assert_label_is_native_evaluation(systems[family], label)
        # Labeling observed the NPT walker without ending its dynamics: the
        # terminal walker is the complete MTK continuation of the NPT run.
        walker = walkers[task].continuation
        assert isinstance(walker, AtomisticDynamicsContinuation)
        assert walker.source_binding_id == npt[family].binding_id
        assert walker.state.controllers.cell is not None
        assert walker.state.controllers.particle_chain is not None
        assert int(walker.state.step_index) == 10
        np.testing.assert_array_equal(label.ports["time"], walker.state.time)
        np.testing.assert_array_equal(
            label.ports["momenta"], walker.state.kinematics.momenta
        )
    # Failed or refused visits before any dynamics have no walker continuation.
    assert "pair-fail" not in walkers
    # Relaxed pairs left the repulsive range: the analytic minimum set of the
    # soft repulsion (zero energy and force at every separation >= r_c).
    for task in ("pair-a", "pair-b"):
        relaxed = outputs[(task, "relax")]
        assert relaxed.outcome is AtomisticStageOutcome.CONVERGED
        separation = relaxed.ports["positions"][1] - relaxed.ports["positions"][0]
        separation -= 4.0 * np.round(separation / 4.0)
        assert np.linalg.norm(separation) >= CUTOFF - 1e-9
        np.testing.assert_allclose(relaxed.ports["energy"], 0.0, atol=1e-12)
        np.testing.assert_allclose(relaxed.ports["forces"], 0.0, atol=1e-12)
    assert outputs[("quad-a", "relax")].outcome in (
        AtomisticStageOutcome.CONVERGED,
        AtomisticStageOutcome.EXHAUSTED,
    )
    # Backpressure parked upstream lanes and the bounded pending bank held.
    assert any(item.backpressured for item in evidence)
    assert any(item.parked for item in evidence)
    assert max(item.pending for item in evidence) <= plan.max_pending_publications
    _assert_exactly_once(state, sink, expected=13)


def _assert_label_is_native_evaluation(system: Any, label: Any) -> None:
    program = prepare_potential(system, (_term(),))
    positions = np.asarray(label.ports["positions"])
    cell = np.asarray(label.ports["cell_vectors"])
    images = np.floor(positions @ np.linalg.inv(cell)).astype(np.int32)
    wrapped = positions - images @ cell
    state = bundle_state(program, wrapped, cell_vectors=cell, image_counts=images)
    evaluation = program.evaluate(
        jnp.asarray(wrapped),
        state,
        compute_stress=True,
        cell=system.cell,
        cell_vectors=jnp.asarray(cell),
        image_counts=jnp.asarray(images),
        unwrapped_positions=jnp.asarray(positions),
    )
    np.testing.assert_allclose(
        label.ports["energy"], evaluation.energy, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        label.ports["forces"], evaluation.forces, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        label.ports["stress"], evaluation.stress, rtol=1e-9, atol=1e-10
    )


# -- interruption at every transaction boundary ----------------------------------------------


def _small_plan(
    *, lanes: int = 2, order: tuple[int, ...] = (0, 1, 2)
) -> AtomisticCampaignPlan:
    system = _system(2, 4.0)
    starts = (
        [[1.4, 2.0, 2.0], [2.6, 2.0, 2.0]],
        [[2.0, 1.2, 2.0], [2.0, 2.4, 2.2]],
        [[1.0, 1.0, 1.0], [2.2, 1.4, 1.0]],
    )
    momenta = np.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    tasks = tuple(
        AtomisticCampaignTask(
            f"walker-{index}",
            "pair",
            _packet(starts[index], momenta=momenta),
            material_id="pair",
        )
        for index in order
    )
    stages = (
        AtomisticStagePlan(
            "nvt",
            {"pair": _nvt(system)},
            budget=10,
            lane_capacity=lanes,
            queue_capacity=1,
        ),
        AtomisticStagePlan(
            "label",
            {"pair": _label(system)},
            lane_capacity=1,
            queue_capacity=1,
            entry=AtomisticEntryPolicy.SHARE,
        ),
    )
    return AtomisticCampaignPlan(
        "small-campaign", stages, tasks, max_pending_publications=1
    )


def _reference(tmp_path: Path) -> dict[tuple[str, str], Any]:
    prepared = _small_plan().prepare()
    sink = AtomisticDirectorySink(tmp_path / "reference", accept_per_round=1)
    state, _ = _run_rounds(
        prepared,
        prepared.start(jr.key(3)),
        sink,
        AtomisticCampaignStore(tmp_path / "reference.ckpt"),
        200,
    )
    assert state.complete
    return _outputs(sink)


class _CrashingSink(AbstractAtomisticCampaignSink):
    """Makes one record durable, then fails before returning its receipt."""

    def __init__(self, inner: AtomisticDirectorySink, /) -> None:
        self.inner = inner
        self.sink_id = inner.sink_id

    def receipt(self, semantic_key: str, /) -> AtomisticOutputReceipt | None:
        return self.inner.receipt(semantic_key)

    def publish(
        self, record: Any, /, *, round_index: int
    ) -> AtomisticOutputReceipt | None:
        self.inner.publish(record, round_index=round_index)
        raise RuntimeError("process lost after the durable write")


@pytest.mark.parametrize(
    "boundary",
    ["evaluate", "stage", "publish", "sink-crash", "commit-write", "commit", "refill"],
)
def test_interruption_at_every_boundary_resumes_without_duplicates(
    tmp_path: Path, boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference(tmp_path)
    prepared = _small_plan().prepare()
    directory = tmp_path / "outputs"
    sink = AtomisticDirectorySink(directory, accept_per_round=1)
    store = AtomisticCampaignStore(tmp_path / "campaign.ckpt")
    state, _ = _run_rounds(prepared, prepared.start(jr.key(3)), sink, store, 4)
    assert not state.complete
    durable_round = state.round_index - 1
    evaluated = prepared.evaluate(state)
    if boundary != "evaluate":
        staged = prepared.stage_outputs(evaluated)
        if boundary == "sink-crash" and staged.pending:
            with pytest.raises(RuntimeError, match="durable write"):
                prepared.publish(staged, _CrashingSink(sink))
        elif boundary not in ("stage", "sink-crash"):
            published = prepared.publish(staged, sink)
            if boundary == "commit-write":

                def refuse(*args: Any, **kwargs: Any) -> None:
                    raise OSError("disk lost during the envelope write")

                with monkeypatch.context() as patch:
                    patch.setattr(campaign_module, "write_array_archive", refuse)
                    with pytest.raises(OSError):
                        prepared.commit(published, store)
            elif boundary in ("commit", "refill"):
                committed = prepared.commit(published, store)
                durable_round = committed.round_index
                if boundary == "refill":
                    prepared.refill(committed)
    # Fresh process: a newly built plan, sink object and store reader.
    fresh = _small_plan().prepare()
    resumed = fresh.resume(AtomisticCampaignStore(tmp_path / "campaign.ckpt"))
    assert resumed.round_index == durable_round + 1
    resumed_sink = AtomisticDirectorySink(directory, accept_per_round=1)
    final, _ = _run_rounds(
        fresh,
        resumed,
        resumed_sink,
        AtomisticCampaignStore(tmp_path / "campaign.ckpt"),
        200,
    )
    assert final.complete
    outputs = _outputs(resumed_sink)
    assert set(outputs) == set(reference)
    for key, output in outputs.items():
        expected = reference[key]
        assert output.semantic_key == expected.semantic_key
        assert output.outcome is expected.outcome
        np.testing.assert_array_equal(output.rng_words, expected.rng_words)
        assert set(output.ports) == set(expected.ports)
        for name, value in output.ports.items():
            np.testing.assert_allclose(
                value, expected.ports[name], rtol=1e-12, atol=1e-12
            )
    assert len(list(directory.glob("*.output"))) == len(reference)


def test_corrupt_or_foreign_envelope_is_refused(tmp_path: Path) -> None:
    prepared = _small_plan().prepare()
    store = AtomisticCampaignStore(tmp_path / "campaign.ckpt")
    _run_rounds(
        prepared,
        prepared.start(jr.key(3)),
        AtomisticDirectorySink(tmp_path / "out"),
        store,
        2,
    )
    other = _small_plan(order=(2, 1, 0)).prepare()
    with pytest.raises(ValueError, match="another campaign plan"):
        other.resume(store)
    data = bytearray((tmp_path / "campaign.ckpt").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (tmp_path / "campaign.ckpt").write_bytes(bytes(data))
    with pytest.raises(Exception):
        prepared.resume(store)


# -- RNG identity ------------------------------------------------------------------------------


def test_rng_keys_follow_task_identity_not_lanes_rounds_or_order(tmp_path: Path) -> None:
    results = []
    for lanes, order in ((1, (0, 1, 2)), (3, (2, 0, 1))):
        prepared = _small_plan(lanes=lanes, order=order).prepare()
        sink = AtomisticDirectorySink(tmp_path / f"out-{lanes}")
        state, _ = _run_rounds(
            prepared,
            prepared.start(jr.key(3)),
            sink,
            AtomisticCampaignStore(tmp_path / f"c-{lanes}"),
            200,
        )
        assert state.complete
        results.append(
            {key[0]: value for key, value in _outputs(sink).items() if key[1] == "label"}
        )
    first, second = results
    assert set(first) == set(second)
    for task, output in first.items():
        other = second[task]
        np.testing.assert_array_equal(output.rng_words, other.rng_words)
        np.testing.assert_allclose(
            output.ports["positions"], other.ports["positions"], atol=1e-12
        )
        assert output.outcome is other.outcome
    words = {task: tuple(output.rng_words) for task, output in first.items()}
    assert len(set(words.values())) == len(words)


# -- capacity rebucketing and reprime ------------------------------------------------------------


def test_capacity_rebucket_reprimes_without_physical_or_rng_advance(
    tmp_path: Path,
) -> None:
    system = _system(2, 6.0)
    exact = _exact_roles(system)
    positions, momenta = _tight_positions()
    plan = AtomisticCampaignPlan(
        "rebucket",
        (
            AtomisticStagePlan(
                "nvt",
                {
                    "tight": _nvt(
                        system, roles=(exact, role_capacity(4096, 512)), steps=1
                    ),
                    "refuse": _nvt(system, roles=(exact,), steps=1),
                },
                budget=100,
                lane_capacity=1,
            ),
        ),
        (
            AtomisticCampaignTask(
                "tight", "tight", _packet(positions, momenta=momenta), material_id="pair"
            ),
            AtomisticCampaignTask(
                "refuse",
                "refuse",
                _packet(positions, momenta=momenta),
                material_id="pair",
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.run_round(prepared.start(jr.key(5)), sink, store)
    transfer_seen = False
    for _ in range(200):
        bucket = state.task("tight").bucket
        lane = state.task("tight").lane
        before = None
        if state.task("tight").status is AtomisticTaskStatus.ACTIVE:
            assert bucket is not None
            bank = state.bank(bucket)
            before = jax.tree.map(lambda leaf: np.asarray(leaf[lane]), bank.lanes.state)
        evaluated = prepared.evaluate(state)
        record = evaluated.task("tight")
        if before is not None and record.status is AtomisticTaskStatus.QUEUED:
            transfer_seen = True
            assert record.packet is not None and record.bucket is not None
            carried = record.packet.continuation
            assert isinstance(carried, AtomisticDynamicsContinuation)
            continuation = carried.state
            # Reprime under grown capacities: no step, time, revision or key event.
            assert int(continuation.step_index) == int(before.step_index)
            assert float(continuation.time) == float(before.time)
            assert int(continuation.configuration_revision) == int(
                before.configuration_revision
            )
            np.testing.assert_array_equal(continuation.random_key, before.random_key)
            np.testing.assert_array_equal(
                continuation.kinematics.positions, before.kinematics.positions
            )
            target = evaluated.bank(record.bucket)
            assert isinstance(target.operation, AtomisticDynamicsStageOperation)
            assert carried.source_binding_id == target.operation.binding_id
            assert (
                continuation.force.bundle_id
                == target.operation.dynamics.bundle.prepared_id
            )
            assert target.operation.dynamics.bundle.role_levels == (1,)
            assert record.consumed == int(before.step_index)
            assert record.campaign_reasons & int(AtomisticCampaignReason.ROLE_CAPACITY)
        state = prepared.refill(
            prepared.commit(
                prepared.publish(prepared.stage_outputs(evaluated), sink), store
            )
        )
        if state.complete:
            break
    assert transfer_seen and state.complete
    outputs = _outputs(sink)
    assert outputs[("tight", "nvt")].outcome is AtomisticStageOutcome.COMPLETED
    assert outputs[("tight", "nvt")].counters["consumed"] == 100
    refused = outputs[("refuse", "nvt")]
    assert refused.outcome is AtomisticStageOutcome.CAPACITY_REFUSED
    # The refused attempt was never consumed: its accepted state is pre-overflow.
    assert refused.counters["consumed"] < 100


def test_admission_capacity_transfer_keeps_the_visit_key_counter(tmp_path: Path) -> None:
    """Capacity found at admission rebuckets with the visit's own semantic key.

    Regression of the relaxation-campaign example: the re-admission of an
    admission-time transfer must reuse counter ``entries - 1`` of a visit
    that was counted, across a fresh-process resume as well.
    """
    system = _system(2, 6.0)
    exact = _exact_roles(system)
    close = np.asarray([[2.0, 3.0, 3.0], [3.0, 3.0, 3.0]])

    def plan(roles: tuple[Any, ...]) -> AtomisticCampaignPlan:
        return AtomisticCampaignPlan(
            "admission-transfer",
            (
                AtomisticStagePlan(
                    "nvt",
                    {"pair": _nvt(system, roles=roles, steps=1)},
                    budget=3,
                    lane_capacity=1,
                ),
            ),
            (
                AtomisticCampaignTask(
                    "close",
                    "pair",
                    _packet(close, momenta=np.zeros((2, 3))),
                    material_id="pair",
                ),
            ),
        )

    ladder = (exact, role_capacity(4096, 512))
    prepared = plan(ladder).prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.run_round(prepared.start(jr.key(9)), sink, store)
    record = state.task("close")
    # The admission found the role capacity exceeded: queued for the grown bucket.
    assert record.status is AtomisticTaskStatus.QUEUED
    assert record.bucket is not None
    assert record.bucket != prepared.plan.stages[0].operations["pair"].binding_id
    assert record.entries == (1,)
    # Commit the queued transfer, then lose the process before admitting it.
    prepared.commit(
        prepared.publish(prepared.stage_outputs(prepared.evaluate(state)), sink), store
    )
    fresh = plan(ladder).prepare()
    state, _ = _run_rounds(fresh, fresh.resume(store), sink, store, 20)
    assert state.complete
    assert state.task("close").entries == (1,)
    output = _outputs(sink)[("close", "nvt")]
    assert output.outcome is AtomisticStageOutcome.COMPLETED
    assert output.stage_entry == 0 and output.counters["consumed"] == 3
    # The same visit key as a campaign that never needed the transfer.
    reference_sink = AtomisticDirectorySink(tmp_path / "reference")
    reference = plan((role_capacity(4096, 512),)).prepare()
    done, _ = _run_rounds(
        reference,
        reference.start(jr.key(9)),
        reference_sink,
        AtomisticCampaignStore(tmp_path / "rc"),
        20,
    )
    assert done.complete
    np.testing.assert_array_equal(
        output.rng_words, _outputs(reference_sink)[("close", "nvt")].rng_words
    )


# -- dynamics continuation across stages -----------------------------------------------------------


_WALKER_POSITIONS = np.asarray([[1.4, 2.0, 2.0], [2.5, 2.0, 2.0]])
_WALKER_MOMENTA = np.asarray([[0.3, 0.0, 0.0], [-0.3, 0.1, 0.0]])


def _walker_campaign(
    campaign_id: str, stages: tuple[AtomisticStagePlan, ...], /
) -> AtomisticCampaignPlan:
    return AtomisticCampaignPlan(
        campaign_id,
        stages,
        (
            AtomisticCampaignTask(
                "walker",
                "pair",
                _packet(_WALKER_POSITIONS, momenta=_WALKER_MOMENTA),
                material_id="pair",
            ),
        ),
    )


def _cadence_plan(
    operation: Any, label: Any, cadence: tuple[int, ...], /
) -> AtomisticCampaignPlan:
    """One walker observed every ``cadence[i]`` steps (optionally labeled between)."""
    stages: list[AtomisticStagePlan] = []
    for index, budget in enumerate(cadence):
        stages.append(
            AtomisticStagePlan(
                f"run-{index}", {"pair": operation}, budget=budget, lane_capacity=1
            )
        )
        if label is not None and index + 1 < len(cadence):
            stages.append(
                AtomisticStagePlan(f"label-{index}", {"pair": label}, lane_capacity=1)
            )
    return _walker_campaign("cadence", tuple(stages))


def _final_walker(
    plan: AtomisticCampaignPlan, directory: Path, /
) -> AtomisticDynamicsContinuation:
    prepared = plan.prepare()
    state, _ = _run_rounds(
        prepared,
        prepared.start(jr.key(7)),
        AtomisticDirectorySink(directory / "out"),
        AtomisticCampaignStore(directory / "c"),
        200,
    )
    assert state.complete
    assert state.task("walker").outcome == int(AtomisticStageOutcome.COMPLETED)
    walker = prepared.terminal_packets(state)["walker"].continuation
    assert isinstance(walker, AtomisticDynamicsContinuation)
    return walker


def _assert_same_state(left: Any, right: Any) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for first, second in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


@pytest.mark.parametrize("method", ["mtk-npt", "baoab-nvt"])
@pytest.mark.parametrize("observed", [False, True])
def test_observation_cadence_does_not_change_the_dynamics(
    tmp_path: Path, method: str, observed: bool
) -> None:
    """20 steps observed once, or every 5 steps (optionally labeled), are one run.

    MTK NPT carries Nose-Hoover chains and the barostat; BAOAB carries its
    stochastic stream.  A stage boundary that re-initialized controllers or
    re-keyed the walker would change the trajectory.
    """
    system = _system(2, 4.0)
    # The high-pressure compression fixture leaves its admitted cell domain
    # before step 20; this continuation control uses zero external pressure.
    operation = _npt(system, pressure=0.0) if method == "mtk-npt" else _nvt(system)
    label = _label(system) if observed else None
    single = _final_walker(_cadence_plan(operation, None, (20,)), tmp_path / "single")
    split = _final_walker(
        _cadence_plan(operation, label, (5, 5, 5, 5)), tmp_path / "split"
    )
    assert int(single.state.step_index) == int(split.state.step_index) == 20
    _assert_same_state(split.state, single.state)
    if method == "mtk-npt":
        # The thermostat chain evolved: a reset at any boundary would show.
        initial = operation.dynamics.initial_controllers(
            jnp.asarray(split.state.cell_vectors),
            split.state.kinematics.positions.dtype,
        )
        assert split.state.controllers.cell is not None
        assert any(
            not np.array_equal(np.asarray(start), np.asarray(end))
            for start, end in zip(
                jax.tree_util.tree_leaves(initial.particle_chain),
                jax.tree_util.tree_leaves(split.state.controllers.particle_chain),
                strict=True,
            )
        )


def test_dynamics_continuation_survives_a_fresh_process_between_stages(
    tmp_path: Path,
) -> None:
    system = _system(2, 4.0)
    reference = _final_walker(
        _cadence_plan(_npt(system, pressure=0.0), None, (10, 10)), tmp_path / "reference"
    )
    prepared = _cadence_plan(_npt(system, pressure=0.0), None, (10, 10)).prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.start(jr.key(7))
    for _ in range(100):
        state = prepared.run_round(state, sink, store)
        if state.task("walker").stage == 1:
            break
    # The durable envelope holds the walker queued at the second stage with
    # its complete continuation; a fresh process rebuilds every operation.
    assert state.task("walker").stage == 1
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import sys
from phydrax.atomistic import AtomisticCampaignStore, AtomisticDirectorySink
from tests.unit.atomistic.test_campaign import _cadence_plan, _npt, _run_rounds, _system

prepared = _cadence_plan(_npt(_system(2, 4.0), pressure=0.0), None, (10, 10)).prepare()
store = AtomisticCampaignStore(Path(sys.argv[1]))
sink = AtomisticDirectorySink(Path(sys.argv[2]))
final, _ = _run_rounds(prepared, prepared.resume(store), sink, store, 100)
if not final.complete:
    raise RuntimeError("The restored campaign did not finish.")
""",
            str(tmp_path / "c"),
            str(tmp_path / "out"),
        ],
        check=True,
        timeout=300,
    )
    fresh = _cadence_plan(_npt(system, pressure=0.0), None, (10, 10)).prepare()
    assert fresh.plan_id == prepared.plan_id
    state = fresh.resume(store)
    assert state.complete
    _assert_same_state(
        fresh.terminal_packets(state)["walker"].continuation.state, reference.state
    )
    # The terminal walker is itself durable.
    again = _cadence_plan(_npt(system, pressure=0.0), None, (10, 10)).prepare()
    restored = again.resume(store)
    _assert_same_state(
        again.terminal_packets(restored)["walker"].continuation.state, reference.state
    )


def test_stage_hamiltonian_change_continues_by_the_owner_transition(
    tmp_path: Path,
) -> None:
    system = _system(2, 4.0)
    old, new = _scaled_nvt(system, 10.0), _scaled_nvt(system, 12.0)
    plan = _walker_campaign(
        "transition",
        (
            AtomisticStagePlan("old", {"pair": old}, budget=4, lane_capacity=1),
            AtomisticStagePlan("new", {"pair": new}, budget=4, lane_capacity=1),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.start(jr.key(4))
    queued = None
    for _ in range(50):
        staged = prepared.stage_outputs(prepared.evaluate(state))
        record = staged.task("walker")
        if record.stage == 1 and record.status is AtomisticTaskStatus.QUEUED:
            queued = record.packet
        state = prepared.refill(prepared.commit(prepared.publish(staged, sink), store))
        if queued is not None:
            break
    assert queued is not None
    leaving = queued.continuation
    assert isinstance(leaving, AtomisticDynamicsContinuation)
    assert leaving.source_binding_id == old.binding_id
    record = state.task("walker")
    assert record.bucket == new.binding_id and len(record.lineage) == 1
    entered = jax.tree.map(
        lambda leaf: np.asarray(leaf[record.lane]), state.bank(record.bucket).lanes.state
    )
    # Physical, stochastic and controller continuation are the owner's.
    for name in ("step_index", "time", "configuration_revision", "random_key"):
        np.testing.assert_array_equal(
            getattr(entered, name), getattr(leaving.state, name)
        )
    _assert_same_state(entered.kinematics, leaving.state.kinematics)
    _assert_same_state(entered.controllers, leaving.state.controllers)
    assert entered.prepared_dynamics_id == new.dynamics.prepared_id
    # The re-primed energy is the new Hamiltonian's; the jump is external work.
    before = float(leaving.state.force.potential_energy)
    assert before > 0.0
    jump = float(entered.force.potential_energy) - before
    np.testing.assert_allclose(jump, 0.2 * before, rtol=1e-12)
    np.testing.assert_allclose(
        float(entered.energy.external_work - leaving.state.energy.external_work),
        jump,
        rtol=1e-12,
    )
    # The host promotion path is the same owner transition.
    promoted = old.transition_packet(queued, new)
    assert isinstance(promoted.continuation, AtomisticDynamicsContinuation)
    assert promoted.continuation.lineage == record.lineage
    _assert_same_state(promoted.continuation.state, entered)
    state, _ = _run_rounds(prepared, state, sink, store, 50)
    assert state.complete
    output = _outputs(sink)[("walker", "new")]
    assert output.outcome is AtomisticStageOutcome.COMPLETED
    assert output.hamiltonian_lineage == record.lineage
    assert output.numeric_revision_ids == new.numeric_revision_ids
    final = prepared.terminal_packets(state)["walker"].continuation
    assert int(final.state.step_index) == 8
    # A continuation whose source runtime is unavailable is refused, never
    # silently re-initialized under the new model.
    orphan = AtomisticCampaignPlan(
        "orphan",
        (AtomisticStagePlan("new", {"pair": new}, budget=4, lane_capacity=1),),
        (AtomisticCampaignTask("walker", "pair", queued, material_id="pair"),),
    ).prepare()
    orphan_sink = AtomisticDirectorySink(tmp_path / "orphan")
    done, _ = _run_rounds(
        orphan,
        orphan.start(jr.key(4)),
        orphan_sink,
        AtomisticCampaignStore(tmp_path / "oc"),
        10,
    )
    assert done.complete
    refused = _outputs(orphan_sink)[("walker", "new")]
    assert refused.outcome is AtomisticStageOutcome.FAILED
    assert refused.campaign_reasons & int(AtomisticCampaignReason.IDENTITY_MISMATCH)
    kept = orphan.terminal_packets(done)["walker"].continuation
    _assert_same_state(kept.state, leaving.state)
    np.testing.assert_array_equal(refused.rng_words, leaving.state.random_key)
    # A packet whose walker continues across campaigns adopts bitwise.
    adopted = AtomisticCampaignPlan(
        "adopted",
        (AtomisticStagePlan("new", {"pair": new}, budget=4, lane_capacity=1),),
        (AtomisticCampaignTask("walker", "pair", promoted, material_id="pair"),),
    ).prepare()
    done, _ = _run_rounds(
        adopted,
        adopted.start(jr.key(99)),
        AtomisticDirectorySink(tmp_path / "adopted"),
        AtomisticCampaignStore(tmp_path / "ac"),
        10,
    )
    assert done.complete
    _assert_same_state(
        adopted.terminal_packets(done)["walker"].continuation.state, final.state
    )


@pytest.mark.parametrize(
    "policy", [AtomisticContinuationPolicy.CONTINUE, AtomisticContinuationPolicy.RESTART]
)
def test_integrator_change_needs_an_explicit_restart(
    tmp_path: Path, policy: AtomisticContinuationPolicy
) -> None:
    system = _system(2, 4.0)
    nvt, npt = _nvt(system), _npt(system)
    plan = _walker_campaign(
        f"ensemble-{policy.value}",
        (
            AtomisticStagePlan("nvt", {"pair": nvt}, budget=5, lane_capacity=1),
            AtomisticStagePlan(
                "npt", {"pair": npt}, budget=5, lane_capacity=1, continuation=policy
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    state, _ = _run_rounds(
        prepared,
        prepared.start(jr.key(1)),
        sink,
        AtomisticCampaignStore(tmp_path / "c"),
        50,
    )
    assert state.complete
    output = _outputs(sink)[("walker", "npt")]
    walker = prepared.terminal_packets(state)["walker"].continuation
    if policy is AtomisticContinuationPolicy.CONTINUE:
        assert output.outcome is AtomisticStageOutcome.FAILED
        assert output.campaign_reasons & int(AtomisticCampaignReason.CONTINUATION_REFUSED)
        # The refused walker keeps its NVT continuation as evidence.
        assert walker.source_binding_id == nvt.binding_id
        assert int(walker.state.step_index) == 5
        np.testing.assert_array_equal(output.rng_words, walker.state.random_key)
    else:
        assert output.outcome is AtomisticStageOutcome.COMPLETED
        assert output.campaign_reasons & int(AtomisticCampaignReason.DYNAMICS_RESTARTED)
        assert walker.source_binding_id == npt.binding_id
        assert int(walker.state.step_index) == 5
        # The new run starts at the NVT walker's time (5 x 0.01).
        np.testing.assert_allclose(
            float(walker.state.time), 5 * 0.01 + 5 * 0.005, rtol=1e-12
        )


# -- groups, parking and revisions ---------------------------------------------------------------


def test_group_waits_for_every_member_and_publishes_together(tmp_path: Path) -> None:
    system = _system(2, 4.0)
    momenta = np.zeros((2, 3))
    good = _packet([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]], momenta=momenta)
    broken = AtomisticTaskPacket(positions=np.full((2, 3), np.nan))
    plan = AtomisticCampaignPlan(
        "group",
        (
            AtomisticStagePlan(
                "nvt", {"pair": _nvt(system, steps=5)}, budget=10, lane_capacity=2
            ),
        ),
        (
            AtomisticCampaignTask(
                "member-ok", "pair", good, material_id="pair", group_id="replicas"
            ),
            AtomisticCampaignTask(
                "member-bad", "pair", broken, material_id="pair", group_id="replicas"
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.run_round(prepared.start(jr.key(1)), sink, store)
    # Both members were admitted atomically into the bank's two lanes.
    assert {state.task("member-ok").status, state.task("member-bad").status} == {
        AtomisticTaskStatus.ACTIVE
    }
    state = prepared.run_round(state, sink, store)
    # The refused member is terminal, but its group is incomplete: nothing moves.
    assert state.evidence.group_waiting == 1
    assert not state.receipts
    state = prepared.run_round(state, sink, store)
    # The whole group left the stage in one transaction.
    assert state.complete
    outputs = _outputs(sink)
    assert outputs[("member-bad", "nvt")].outcome is AtomisticStageOutcome.FAILED
    assert outputs[("member-bad", "nvt")].campaign_reasons & int(
        AtomisticCampaignReason.INVALID_PACKET
    )
    assert outputs[("member-ok", "nvt")].outcome is AtomisticStageOutcome.COMPLETED
    assert outputs[("member-ok", "nvt")].counters["consumed"] == 10
    assert {receipt.semantic_key for receipt in state.receipts} == {
        outputs[("member-bad", "nvt")].semantic_key,
        outputs[("member-ok", "nvt")].semantic_key,
    }


def test_parked_lane_keeps_its_accepted_state_and_budget(tmp_path: Path) -> None:
    prepared = _small_plan(lanes=2).prepare()
    sink = AtomisticDirectorySink(tmp_path / "out", accept_per_round=0)
    store = AtomisticCampaignStore(tmp_path / "c")
    state, evidence = _run_rounds(prepared, prepared.start(jr.key(3)), sink, store, 6)
    assert any(item.parked for item in evidence)
    parked = [
        (bank.bucket_id, slot)
        for bank in state.banks
        for slot, task in enumerate(bank.occupant)
        if task is not None
        and int(bank.outcome[slot]) == int(AtomisticStageOutcome.COMPLETED)
    ]
    assert parked
    later = prepared.run_round(state, sink, store)
    for bucket, slot in parked:
        before, after = state.bank(bucket), later.bank(bucket)
        assert after.occupant[slot] == before.occupant[slot]
        assert int(after.consumed[slot]) == int(before.consumed[slot])
        for left, right in zip(
            jax.tree.leaves(before.lanes), jax.tree.leaves(after.lanes), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(left)[slot], np.asarray(right)[slot])
    # No sink capacity: nothing is durable and the pending bank stays bounded.
    assert not later.receipts and len(later.pending) == 1
    assert later.evidence.backpressured


class _ScaledPairTerm(AbstractAtomisticEnergyTerm):
    """Soft pair repulsion with its strength as a numeric array leaf.

    The strength is part of the scientific term identity (a different
    strength is a different Hamiltonian), while the structural signature —
    term type, role, requirements and leaf shapes — is unchanged, like two
    weight revisions of one learned architecture.
    """

    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    strength: jax.Array
    requirement: AtomisticNeighborhoodRequirement
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, strength: float, /) -> None:
        self.name = "scaled"
        self.force_group = 0
        self.term_identity = IdentityWords.of(f"scaled-soft-pair-{float(strength)!r}")
        self.strength = jnp.asarray(strength, dtype=jnp.float64)
        self.requirement = AtomisticNeighborhoodRequirement(
            "scaled", AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH, CUTOFF
        )
        self.capabilities = AtomisticPotentialCapabilities(
            orthorhombic_periodic=True, triclinic_periodic=True, cell_derivative=True
        )
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=(self.requirement,)
        )

    def prepare(self, system: Any, /) -> _PreparedScaledPairTerm:
        del system
        return _PreparedScaledPairTerm(self)


class _PreparedScaledPairTerm(AbstractPreparedAtomisticEnergyTerm):
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    strength: jax.Array
    requirement: AtomisticNeighborhoodRequirement
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: _ScaledPairTerm, /) -> None:
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.prepared_identity = IdentityWords.of(f"prepared-{plan.term_id}")
        self.strength = plan.strength
        self.requirement = plan.requirement
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements

    def energy(self, context: Any, /) -> AtomisticTermEvaluation:
        view = context.neighborhood(self.requirement)
        positions = context.positions
        vectors = jnp.asarray(context.cell_vectors).astype(positions.dtype)
        displacement = view.displacement(positions, vectors)
        inside = view.cutoff_mask(positions, vectors)
        distance = jnp.sqrt(jnp.where(inside, jnp.sum(displacement**2, axis=-1), 1.0))
        edge = self.strength * jnp.where(
            inside, 0.5 * (1.0 - distance / CUTOFF) ** 4, 0.0
        )
        atom_energy = (
            jnp.zeros((positions.shape[0],), dtype=positions.dtype)
            .at[view.relation.receiver_indices]
            .add(edge)
        )
        return AtomisticTermEvaluation(
            jnp.sum(edge), atom_energy, jnp.all(view.evidence.successful)
        )


def _scaled_nvt(system: Any, strength: float) -> AtomisticDynamicsStageOperation:
    dynamics = prepare_dynamics(
        system, (_ScaledPairTerm(strength),), BAOABLangevinPlan(0.01, FRICTION)
    )
    return AtomisticDynamicsStageOperation(
        dynamics,
        (row(measure(system), "nvt", temperature=TEMPERATURE),),
        steps_per_round=2,
    )


def test_model_revision_rebinding_drains_in_flight_lanes(tmp_path: Path) -> None:
    system = _system(2, 4.0)
    old, new = _scaled_nvt(system, 10.0), _scaled_nvt(system, 12.0)
    # One structural signature (grouping), two scientific Hamiltonians.
    assert old.signature(1).signature_id == new.signature(1).signature_id
    assert old.numeric_revision_ids != new.numeric_revision_ids
    assert old.evaluation_key().hamiltonian_id != new.evaluation_key().hamiltonian_id
    momenta = np.zeros((2, 3))
    tasks = (
        AtomisticCampaignTask(
            "first",
            "pair",
            _packet([[1.4, 2.0, 2.0], [2.5, 2.0, 2.0]], momenta=momenta),
            material_id="pair",
        ),
        AtomisticCampaignTask(
            "second",
            "pair",
            _packet([[1.2, 2.0, 2.0], [2.5, 2.0, 2.0]], momenta=momenta),
            material_id="pair",
        ),
    )
    plan = AtomisticCampaignPlan(
        "revision",
        (AtomisticStagePlan("nvt", {"pair": old}, budget=6, lane_capacity=1),),
        tasks,
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.run_round(
        prepared.run_round(prepared.start(jr.key(2)), sink, store), sink, store
    )
    record = state.task("first")
    assert record.status is AtomisticTaskStatus.ACTIVE
    old_bank = record.bucket
    assert old_bank is not None
    rebound = prepared.rebind_operation(state, "nvt", "pair", new)
    # The admitted lane keeps its Hamiltonian; the stage now admits under `new`.
    assert rebound.task("first").bucket == old_bank
    assert rebound.bank(old_bank).retired
    assert rebound.bank(new.binding_id).operation is new
    state, _ = _run_rounds(prepared, rebound, sink, store, 50)
    assert state.complete
    outputs = _outputs(sink)
    first, second = outputs[("first", "nvt")], outputs[("second", "nvt")]
    assert first.numeric_revision_ids == old.numeric_revision_ids
    assert second.numeric_revision_ids == new.numeric_revision_ids
    assert first.evaluation_key.hamiltonian_id == old.evaluation_key().hamiltonian_id
    assert second.evaluation_key.hamiltonian_id == new.evaluation_key().hamiltonian_id
    # A rebound operation not in the plan must be supplied to a fresh process.
    fresh = AtomisticCampaignPlan(plan.campaign_id, plan.stages, plan.tasks).prepare()
    assert fresh.plan_id == prepared.plan_id
    with pytest.raises(ValueError, match="supply the rebound operation"):
        fresh.resume(store)
    fresh.resume(store, operations=(new,))


def test_model_revision_reprime_transitions_in_flight_dynamics(tmp_path: Path) -> None:
    system = _system(2, 4.0)
    old, new = _scaled_nvt(system, 10.0), _scaled_nvt(system, 12.0)
    momenta = np.zeros((2, 3))
    plan = AtomisticCampaignPlan(
        "reprime",
        (
            AtomisticStagePlan(
                "nvt",
                {"pair": old},
                budget=6,
                lane_capacity=1,
                revision=AtomisticRevisionPolicy.REPRIME,
            ),
        ),
        (
            AtomisticCampaignTask(
                "first",
                "pair",
                _packet([[1.4, 2.0, 2.0], [2.5, 2.0, 2.0]], momenta=momenta),
                material_id="pair",
            ),
            AtomisticCampaignTask(
                "second",
                "pair",
                _packet([[1.2, 2.0, 2.0], [2.5, 2.0, 2.0]], momenta=momenta),
                material_id="pair",
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.run_round(
        prepared.run_round(prepared.start(jr.key(2)), sink, store), sink, store
    )
    record = state.task("first")
    assert record.bucket is not None
    before = jax.tree.map(
        lambda leaf: np.asarray(leaf[record.lane]), state.bank(record.bucket).lanes.state
    )
    rebound = prepared.rebind_operation(state, "nvt", "pair", new)
    moved = rebound.task("first")
    assert moved.bucket == new.binding_id and len(moved.lineage) == 1
    after = jax.tree.map(
        lambda leaf: np.asarray(leaf[moved.lane]),
        rebound.bank(new.binding_id).lanes.state,
    )
    # The owner's transition preserves the physical and stochastic continuation.
    for name in ("step_index", "time", "configuration_revision", "random_key"):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    np.testing.assert_array_equal(after.kinematics.positions, before.kinematics.positions)
    np.testing.assert_array_equal(after.kinematics.momenta, before.kinematics.momenta)
    assert after.prepared_dynamics_id == new.dynamics.prepared_id
    assert after.force.program_id == new.dynamics.potential.prepared_id
    # The re-primed energy is the new Hamiltonian's; the jump is external work.
    jump = float(after.force.potential_energy - before.force.potential_energy)
    np.testing.assert_allclose(
        jump, 0.2 * float(before.force.potential_energy), rtol=1e-12
    )
    np.testing.assert_allclose(
        float(after.energy.external_work - before.energy.external_work), jump, rtol=1e-12
    )
    state, _ = _run_rounds(prepared, rebound, sink, store, 50)
    assert state.complete
    outputs = _outputs(sink)
    first, second = outputs[("first", "nvt")], outputs[("second", "nvt")]
    assert first.numeric_revision_ids == new.numeric_revision_ids
    assert first.hamiltonian_lineage == moved.lineage
    assert first.counters["consumed"] == 6
    assert second.hamiltonian_lineage == ()
    fresh = AtomisticCampaignPlan(plan.campaign_id, plan.stages, plan.tasks).prepare()
    fresh.resume(store, operations=(new,))


def test_relaxation_reprime_uses_the_owner_transition(tmp_path: Path) -> None:
    system = _system(2, 4.0)

    def relaxation(strength: float) -> AtomisticRelaxationStageOperation:
        program = prepare_potential(system, (_ScaledPairTerm(strength),))
        plan = AtomisticRelaxationPlan(
            program,
            AtomisticCartesianChart(),
            method=FIRE(maximum_step=0.02),
            convergence=AtomisticRelaxationConvergence(
                projected_tolerance=1e-8, maximum_evaluations=400
            ),
            group_count=1,
        )
        return AtomisticRelaxationStageOperation(plan.prepare(), rounds_per_segment=2)

    old, new = relaxation(10.0), relaxation(12.0)
    plan = AtomisticCampaignPlan(
        "relax-reprime",
        (
            AtomisticStagePlan(
                "relax",
                {"pair": old},
                lane_capacity=1,
                revision=AtomisticRevisionPolicy.REPRIME,
            ),
        ),
        (
            AtomisticCampaignTask(
                "pair",
                "pair",
                _packet([[1.6, 2.0, 2.0], [2.4, 2.0, 2.0]]),
                material_id="pair",
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.run_round(
        prepared.run_round(prepared.start(jr.key(4)), sink, store), sink, store
    )
    assert state.task("pair").status is AtomisticTaskStatus.ACTIVE
    rebound = prepared.rebind_operation(state, "relax", "pair", new)
    bucket = rebound.task("pair").bucket
    assert bucket is not None
    bank = rebound.bank(bucket)
    assert bank.operation.hamiltonian_lineage == bank.lanes.hamiltonian_lineage
    assert len(bank.operation.hamiltonian_lineage) == 1
    state, _ = _run_rounds(prepared, rebound, sink, store, 400)
    assert state.complete
    output = _outputs(sink)[("pair", "relax")]
    assert output.outcome is AtomisticStageOutcome.CONVERGED
    assert output.numeric_revision_ids == new.numeric_revision_ids
    assert output.hamiltonian_lineage == bank.operation.hamiltonian_lineage
    separation = output.ports["positions"][1] - output.ports["positions"][0]
    assert np.linalg.norm(separation) >= CUTOFF - 1e-9
    # The transitioned bank (lineage-bound binding) is rebuilt on resume.
    AtomisticCampaignPlan(plan.campaign_id, plan.stages, plan.tasks).prepare().resume(
        store, operations=(new,)
    )


def test_label_reprime_relabels_unlabeled_lanes_with_the_new_teacher(
    tmp_path: Path,
) -> None:
    system = _system(2, 4.0)
    old = AtomisticLabelStageOperation(
        prepare_potential(system, (_ScaledPairTerm(10.0),)), stress=True
    )
    new = AtomisticLabelStageOperation(
        prepare_potential(system, (_ScaledPairTerm(12.0),)), stress=True
    )
    plan = AtomisticCampaignPlan(
        "teacher",
        (
            AtomisticStagePlan(
                "label",
                {"pair": old},
                lane_capacity=2,
                revision=AtomisticRevisionPolicy.REPRIME,
            ),
        ),
        tuple(
            AtomisticCampaignTask(name, "pair", _packet(start), material_id="pair")
            for name, start in (
                ("near", [[1.4, 2.0, 2.0], [2.5, 2.0, 2.0]]),
                ("far", [[1.0, 2.0, 2.0], [2.2, 2.3, 2.0]]),
            )
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    admitted = prepared.run_round(prepared.start(jr.key(6)), sink, store)
    rebound = prepared.rebind_operation(admitted, "label", "pair", new)
    state, _ = _run_rounds(prepared, rebound, sink, store, 10)
    assert state.complete
    for output in _outputs(sink).values():
        assert output.evaluation_key.hamiltonian_id == new.hamiltonian_id
        assert output.numeric_revision_ids == new.numeric_revision_ids
        assert len(output.hamiltonian_lineage) == 1
        _assert_native_label(new.program, output)


def test_generation_reprime_refuses_finished_proposals(tmp_path: Path) -> None:
    plan, _, _ = _generation_plan(revision=AtomisticRevisionPolicy.REPRIME)
    other, _, _ = _generation_plan(sites=_SITES + 0.05)
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    admitted = prepared.run_round(
        prepared.start(jr.key(8)), sink, AtomisticCampaignStore(tmp_path / "c")
    )
    target = other.stages[0].operations["proposals"]
    current = plan.stages[0].operations["proposals"]
    assert target.signature(2).signature_id == current.signature(2).signature_id
    assert target.numeric_revision_ids != current.numeric_revision_ids
    # One-shot proposals are complete at admission: they belong to the sampler
    # that drew them, so the transition refuses and the state is untouched.
    with pytest.raises(ValueError, match="previous sampler"):
        prepared.rebind_operation(admitted, "generate", "proposals", target)


# -- evaluation identity -----------------------------------------------------------------------


def test_dynamics_admission_adopts_packet_cells_and_returns_owner_refusals(
    tmp_path: Path,
) -> None:
    """An evolved (NPT) cell is adopted; an inadmissible initial state is FAILED, not a crash."""
    from phydrax.atomistic._dynamics import AtomisticStepRejectionReason

    system = _system(2, 4.0)
    evolved = 4.2 * np.eye(3)
    momenta = np.zeros((2, 3))
    plan = AtomisticCampaignPlan(
        "admission",
        (
            AtomisticStagePlan(
                "nvt", {"pair": _nvt(system, steps=2)}, budget=2, lane_capacity=2
            ),
        ),
        (
            AtomisticCampaignTask(
                "evolved",
                "pair",
                AtomisticTaskPacket(
                    positions=np.asarray([[1.4, 2.0, 2.0], [2.6, 2.1, 2.0]]),
                    momenta=momenta,
                    cell_vectors=evolved,
                ),
                material_id="pair",
            ),
            AtomisticCampaignTask(
                "coincident",
                "pair",
                _packet([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], momenta=momenta),
                material_id="pair",
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    state, _ = _run_rounds(
        prepared,
        prepared.start(jr.key(3)),
        sink,
        AtomisticCampaignStore(tmp_path / "c"),
        10,
    )
    assert state.complete
    outputs = _outputs(sink)
    adopted = outputs[("evolved", "nvt")]
    assert adopted.outcome is AtomisticStageOutcome.COMPLETED
    # NVT keeps the adopted cell fixed: the published cell is the packet's.
    np.testing.assert_array_equal(adopted.ports["cell_vectors"], evolved)
    refused = outputs[("coincident", "nvt")]
    assert refused.outcome is AtomisticStageOutcome.FAILED
    assert refused.campaign_reasons & int(AtomisticCampaignReason.ADMISSION_REFUSED)
    assert refused.reasons & int(
        AtomisticStepRejectionReason.POTENTIAL | AtomisticStepRejectionReason.NONFINITE
    )
    assert not refused.reasons & int(
        AtomisticStepRejectionReason.SOURCE_CAPACITY
        | AtomisticStepRejectionReason.ROUTE_CAPACITY
    )


def test_stages_sharing_one_operation_evaluate_in_one_bank(tmp_path: Path) -> None:
    """Two label stages with one teacher binding share lanes and batched evaluations."""
    system = _system(2, 4.0)
    teacher = _label(system)
    momenta = np.zeros((2, 3))
    plan = AtomisticCampaignPlan(
        "shared-teacher",
        (
            AtomisticStagePlan(
                "label-start",
                {"pair": teacher},
                routing=AtomisticStageRouting(
                    completed=AtomisticStageRoute(None, publish=True)
                ),
                lane_capacity=2,
            ),
            AtomisticStagePlan(
                "nvt",
                {"pair": _nvt(system, steps=1)},
                budget=1,
                routing=AtomisticStageRouting(
                    completed=AtomisticStageRoute("label-end", publish=False)
                ),
                lane_capacity=1,
            ),
            AtomisticStagePlan("label-end", {"pair": teacher}, lane_capacity=2),
        ),
        (
            AtomisticCampaignTask(
                "early",
                "pair",
                _packet([[1.4, 2.0, 2.0], [2.5, 2.0, 2.0]]),
                material_id="pair",
            ),
            AtomisticCampaignTask(
                "late",
                "pair",
                _packet([[1.0, 2.0, 2.0], [2.2, 2.3, 2.0]], momenta=momenta),
                material_id="pair",
                entry_stage="nvt",
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state = prepared.start(jr.key(1))
    label_banks = {bucket for position, _, bucket in state.routes if position in (0, 2)}
    assert label_banks == {teacher.binding_id}
    stages_seen: set[int] = set()
    while not state.complete:
        state = prepared.run_round(state, sink, store)
        bank = state.bank(teacher.binding_id)
        stages_seen |= {int(stage) for stage in bank.lane_stage if stage >= 0}
    outputs = _outputs(sink)
    assert set(outputs) == {("early", "label-start"), ("late", "label-end")}
    for output in outputs.values():
        _assert_native_label(teacher.program, output)
    # Lanes of both stages lived in the one teacher bank (one batched call per round).
    assert stages_seen == {0, 2}


def test_evaluation_key_sharing_requires_identical_hamiltonian_context() -> None:
    system = _system(2, 4.0)
    program = prepare_potential(system, (_term(),))
    revision = atomistic_hamiltonian_revision(program).revision_id
    available = AtomisticEvaluationKey(
        "h",
        numeric_revision_ids=(revision,),
        conditions_id="c",
        quantities=("energy", "forces", "stress"),
    )
    request = available.restricted(("energy", "forces"))
    assert request.served_by(available)
    assert not available.served_by(request)
    for changed in (
        AtomisticEvaluationKey(
            "h2",
            numeric_revision_ids=(revision,),
            conditions_id="c",
            quantities=("energy",),
        ),
        AtomisticEvaluationKey(
            "h",
            numeric_revision_ids=("other",),
            conditions_id="c",
            quantities=("energy",),
        ),
        AtomisticEvaluationKey(
            "h",
            numeric_revision_ids=(revision,),
            conditions_id="c2",
            quantities=("energy",),
        ),
        AtomisticEvaluationKey(
            "h",
            numeric_revision_ids=(revision,),
            conditions_id="c",
            bias_id="b",
            quantities=("energy",),
        ),
    ):
        assert not changed.served_by(available)
    assert (
        AtomisticEvaluationKey.from_record(available.to_record()).key_id
        == available.key_id
    )


# -- grouped reaction paths and generation ------------------------------------------------------


_WINDOW = np.asarray([[3.0, 4.0, 4.0], [5.0, 4.0, 4.0], [4.0, 4.0, 2.9], [4.0, 4.0, 5.1]])


def test_reaction_path_stage_moves_whole_bands_and_fails_them_atomically(
    tmp_path: Path,
) -> None:
    """A mobile LJ atom crossing a window of four fixed atoms (saddle at the centre)."""
    import phydrax as phx
    from phydrax.atomistic._campaign import AtomisticReactionPathStageOperation
    from phydrax.atomistic._neighborhood_bundle import (
        AtomisticCellListImageSearch,
        AtomisticNeighborhoodBundlePlan,
        AtomisticNeighborhoodSource,
        AtomisticRoleCapacity,
        AtomisticRoleCapacityLadder,
    )
    from phydrax.atomistic._sites import AtomisticSiteDomain
    from phydrax.chemistry.reaction import AtomisticNudgedElasticBandPlan
    from phydrax.discretization import (
        ParticleImageCapacity,
        ParticleImageCapacityLadder,
        PeriodicCell,
    )

    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(1, 6),
        np.ones((5,), dtype=np.int32),
        np.ones((5,)),
        units,
        atom_type_ids=np.zeros((5,), dtype=np.int32),
        mobile_mask=np.asarray([False, False, False, False, True]),
        cell=PeriodicCell(8.0 * np.eye(3)),
    ).prepare()
    program_plan = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential(np.asarray([1.0]), np.asarray([1.0]), 2.5)]
    )
    (requirement,) = program_plan.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticCellListImageSearch(
            maximum_candidate_slots=10**5, deformation_margin=1.0
        ),
        ParticleImageCapacityLadder(
            (
                ParticleImageCapacity(
                    maximum_particles_per_cell=16,
                    maximum_edges=256,
                    maximum_degree=64,
                    maximum_images=512,
                    maximum_case_edges=256,
                ),
            )
        ),
        skin=0.4,
    )
    ladder = AtomisticRoleCapacityLadder(
        requirement, (AtomisticRoleCapacity(maximum_routes=256, maximum_degree=64),)
    )
    program = program_plan.prepare(
        system,
        neighborhood_bundle=AtomisticNeighborhoodBundlePlan(
            (requirement,), (source,), (ladder,)
        ),
    )
    surface = phx.chemistry.AtomisticPotentialEnergySurface(system, program)
    runtime = AtomisticNudgedElasticBandPlan(
        surface,
        image_count=7,
        spring_constant=1.0,
        force_tolerance=1e-6,
        maximum_evaluations=4000,
        climbing_start=0,
        group_count=2,
        method=FIRE(maximum_step=0.05),
    ).prepare()
    fractions = np.linspace(0.0, 1.0, 7)
    reactant = np.concatenate((_WINDOW, [[4.0, 5.0, 4.0]]))
    product = np.concatenate((_WINDOW, [[4.0, 3.0, 4.0]]))
    band = reactant[None] + fractions[:, None, None] * (product - reactant)[None]
    band[:, 4, 0] += 0.15 * np.sin(np.pi * fractions)
    broken = band.copy()
    broken[3, 4] = _WINDOW[0]
    plan = AtomisticCampaignPlan(
        "paths",
        (
            AtomisticStagePlan(
                "neb",
                {
                    "window": AtomisticReactionPathStageOperation(
                        runtime, rounds_per_segment=500
                    )
                },
                lane_capacity=2,
            ),
        ),
        (
            AtomisticCampaignTask(
                "band", "window", AtomisticTaskPacket(path=band), material_id="window"
            ),
            AtomisticCampaignTask(
                "broken", "window", AtomisticTaskPacket(path=broken), material_id="window"
            ),
        ),
    )
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    state, _ = _run_rounds(
        prepared,
        prepared.start(jr.key(0)),
        sink,
        AtomisticCampaignStore(tmp_path / "c"),
        40,
    )
    assert state.complete
    outputs = _outputs(sink)
    converged = outputs[("band", "neb")]
    assert converged.outcome is AtomisticStageOutcome.CONVERGED
    path = converged.ports["path"]
    assert path.shape == (7, 5, 3)
    # Fixed endpoints need not be minima: export the interior climbing image,
    # not the more energetic endpoint, as the downstream transition state.
    np.testing.assert_array_equal(path[0], reactant)
    np.testing.assert_array_equal(path[-1], product)
    np.testing.assert_array_equal(path[:, :4], np.broadcast_to(_WINDOW, (7, 4, 3)))
    np.testing.assert_allclose(
        converged.ports["positions"][4], [4.0, 4.0, 4.0], atol=1e-5
    )
    energies = converged.ports["path_energies"]
    assert energies[0] > energies[3]
    failed = outputs[("broken", "neb")]
    assert failed.outcome is AtomisticStageOutcome.FAILED
    # Band-atomic failure: the whole accepted band is the unmoved input band.
    np.testing.assert_array_equal(failed.ports["path"], broken)


_SKEW_CELL = np.asarray([[4.0, 0.0, 0.0], [0.8, 3.8, 0.0], [0.3, 0.4, 4.2]])
_SITES = np.asarray([[0.1, 0.2, 0.95], [0.6, 0.5, 0.45], [0.0, 0.0, 0.0]])
_INITIAL_SCALE = 0.01
_LOG_RATIO = float(np.log(1.0 / _INITIAL_SCALE))


class _WrappedSiteScore(eqx.Module):
    """Exact marginal score of wrapped normals at fixed fractional sites (VE noise)."""

    sites: jax.Array

    def __call__(self, state: jax.Array, time: jax.Array) -> jax.Array:
        variance = (_INITIAL_SCALE * jnp.exp(_LOG_RATIO * time)) ** 2
        delta = state - self.sites
        delta = delta - jnp.round(delta)
        images = jnp.arange(-6, 7, dtype=state.dtype)
        shifted = delta[:, None] + images[None, :]
        weights = jax.nn.softmax(-0.5 * shifted**2 / variance, axis=-1)
        return -jnp.sum(weights * shifted, axis=-1) / variance


def _diffusion_sampler(sites: np.ndarray = _SITES) -> Any:
    """Native periodic reverse-diffusion sampler of a fixed-cell SiO pair."""
    from phydrax.atomistic._diffusion import (
        AtomisticPeriodicCoordinateDiffusion,
        PreparedPeriodicDiffusionSampler,
    )
    from phydrax.atomistic._generation import (
        AtomisticCellBounds,
        AtomisticGenerationSupport,
    )
    from phydrax.domain import HyperRectangle, TimeInterval
    from phydrax.stochastic._gaussian_diffusion import VarianceExplodingDiffusion
    from phydrax.units import ANGSTROM

    support = AtomisticGenerationSupport(
        atom_capacity=3,
        species_vocabulary=(8, 14),
        vocabulary_masses=(15.999, 28.085),
        mass_provenance="IUPAC standard atomic weights (test fixture)",
        chemistry="fixed",
        length_unit=ANGSTROM,
        fixed_species=(14, 8, 0),
        fixed_mask=(True, True, False),
        cell_admission="fixed-cell",
        cell=_SKEW_CELL,
        cell_bounds=AtomisticCellBounds(
            minimum_height=2.0,
            maximum_condition=10.0,
            minimum_volume=10.0,
            maximum_volume=200.0,
        ),
    )
    domain = HyperRectangle(
        jnp.full((9,), -1e3), jnp.full((9,), 1e3), label="x"
    ) @ TimeInterval(0.0, 1.0)
    score = domain.Function("x", "t")(
        _WrappedSiteScore(jnp.asarray(np.asarray(sites).reshape(-1)))
    )
    return PreparedPeriodicDiffusionSampler(
        support,
        AtomisticPeriodicCoordinateDiffusion(
            3,
            VarianceExplodingDiffusion(
                9, initial_scale=_INITIAL_SCALE, terminal_scale=1.0
            ),
            image_extent=4,
        ),
        score,
        score_id="analytic-wrapped-sites",
        dt0=0.01,
        wiener_tolerance=1e-4,
    )


def _silica_family() -> tuple[Any, Any]:
    """eV-angstrom SiO system in the sampler's cell and a native image-graph label term.

    The directed-image pair term counts explicit lattice images, so its 3.5 A
    cutoff reaches the generated Si-O separation in this small skew cell.
    """
    import phydrax as phx
    from phydrax.discretization import PeriodicCell

    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(2),
        np.asarray([14, 8], dtype=np.int32),
        np.asarray([28.085, 15.999]),
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=np.zeros((2,), dtype=np.int32),
        cell=PeriodicCell(_SKEW_CELL),
    ).prepare()
    program = prepare_potential(
        system, (DirectedPairTerm("silica-soft", 3.5, profile="smooth", strength=1.0),)
    )
    return system, program


def _generation_plan(
    filters: tuple[Any, ...] = (),
    *,
    sites: np.ndarray = _SITES,
    revision: AtomisticRevisionPolicy = AtomisticRevisionPolicy.DRAIN,
) -> tuple[AtomisticCampaignPlan, Any, Any]:
    from phydrax.atomistic._campaign import AtomisticGenerationStageOperation

    sampler = _diffusion_sampler(sites)
    system, program = _silica_family()
    plan = AtomisticCampaignPlan(
        "generate",
        (
            AtomisticStagePlan(
                "generate",
                {
                    "proposals": AtomisticGenerationStageOperation(
                        sampler,
                        {"silica": system},
                        campaign_id="generate",
                        filters=filters,
                    )
                },
                budget=1,
                routing=AtomisticStageRouting(
                    completed=AtomisticStageRoute("label", publish=True)
                ),
                lane_capacity=2,
                revision=revision,
            ),
            AtomisticStagePlan(
                "label",
                {"silica": AtomisticLabelStageOperation(program, stress=True)},
                lane_capacity=2,
            ),
        ),
        tuple(
            AtomisticCampaignTask(
                f"attempt-{ordinal}",
                "proposals",
                AtomisticTaskPacket(),
                material_id="silica",
                ordinal=ordinal,
            )
            for ordinal in range(3)
        ),
    )
    return plan, sampler, program


def test_native_generation_spawns_routed_tasks_into_downstream_stages(
    tmp_path: Path,
) -> None:
    from phydrax.atomistic._generation import atomistic_proposal_id

    plan, sampler, program = _generation_plan()
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    store = AtomisticCampaignStore(tmp_path / "c")
    state, _ = _run_rounds(prepared, prepared.start(jr.key(8)), sink, store, 3)
    assert not state.complete
    # Fresh-process continuation through the generated child tasks.
    fresh = _generation_plan()[0].prepare()
    state, _ = _run_rounds(fresh, fresh.resume(store), sink, store, 20)
    assert state.complete
    outputs = _outputs(sink)
    children = []
    for ordinal in range(3):
        parent = outputs[(f"attempt-{ordinal}", "generate")]
        assert parent.outcome is AtomisticStageOutcome.COMPLETED
        child = atomistic_proposal_id(
            "generate", sampler.support.support_id, sampler.sampler_id, ordinal
        )
        label = outputs[(child, "label")]
        assert label.parent_id == f"attempt-{ordinal}"
        assert label.outcome is AtomisticStageOutcome.COMPLETED
        fractional = label.ports["positions"] @ np.linalg.inv(_SKEW_CELL)
        delta = fractional - _SITES[:2]
        assert np.max(np.abs(delta - np.round(delta))) < 0.05
        np.testing.assert_array_equal(label.ports["species"], [14, 8])
        np.testing.assert_array_equal(label.ports["cell_vectors"], _SKEW_CELL)
        _assert_native_label(program, label)
        assert float(label.ports["energy"]) > 0.0
        children.append(label.ports["positions"])
    # Proposals are keyed by task identity: distinct ordinals, distinct draws.
    assert not np.allclose(children[0], children[1])


def test_filtered_generation_publishes_rejections_without_children(
    tmp_path: Path,
) -> None:
    from phydrax.atomistic._generation import (
        AtomisticMinimumDistanceFilter,
        AtomisticProposalStatus,
    )

    plan, _, _ = _generation_plan((AtomisticMinimumDistanceFilter(10.0),))
    prepared = plan.prepare()
    sink = AtomisticDirectorySink(tmp_path / "out")
    state, _ = _run_rounds(
        prepared,
        prepared.start(jr.key(8)),
        sink,
        AtomisticCampaignStore(tmp_path / "c"),
        10,
    )
    assert state.complete
    outputs = _outputs(sink)
    assert {key[1] for key in outputs} == {"generate"}
    for output in outputs.values():
        assert output.outcome is AtomisticStageOutcome.FAILED
        assert output.reasons & int(AtomisticProposalStatus.FILTER_REJECTED)


def _assert_native_label(program: Any, label: Any) -> None:
    positions = np.asarray(label.ports["positions"])
    cell = np.asarray(label.ports["cell_vectors"])
    images = np.floor(positions @ np.linalg.inv(cell)).astype(np.int32)
    wrapped = positions - images @ cell
    state = bundle_state(program, wrapped, cell_vectors=cell, image_counts=images)
    evaluation = program.evaluate(
        jnp.asarray(wrapped),
        state,
        compute_stress=True,
        cell=program.system.cell,
        cell_vectors=jnp.asarray(cell),
        image_counts=jnp.asarray(images),
        unwrapped_positions=jnp.asarray(positions),
    )
    np.testing.assert_allclose(
        label.ports["energy"], evaluation.energy, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        label.ports["forces"], evaluation.forces, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        label.ports["stress"], evaluation.stress, rtol=1e-9, atol=1e-10
    )
