# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Staged atomistic campaigns and grouped periodic NEB path cost.

Scenarios (``--scenarios``, one driver):

``campaign``
    Each case runs a native relax (FIRE) -> NVT (BAOAB) -> label (shared E/F/S)
    campaign of Lennard-Jones cluster tasks at one atom capacity and one lane
    capacity.  Fixture: atoms start on the boustrophedon-ordered sites of the
    ``k^3`` periodic lattice of the gas box (``k`` the smallest integer with
    ``k^3 >= atoms``, site spacing ``s = box / k``); consecutive sites form
    dimers (for an odd atom count the last three sites form a trimer around
    the middle site) and every cluster is contracted along its lattice bonds
    to ``FIXTURE_BOND`` before the seeded per-coordinate perturbation of at
    most ``FIXTURE_PERTURBATION``.  For every atom count ``>= 2`` at the gas
    density, ``s >= 1.88 > FIXTURE_BOND``; every atom then has a bonded
    partner within the cutoff and no pair is closer than
    ``FIXTURE_MINIMUM_SEPARATION`` (both checked per task before the campaign
    runs).  Whether relaxation converges within its evaluation budget is
    measured, not assumed.  Recorded per case: the fixture definition, its
    per-task minimum-image interaction geometry and the native initial
    energy/forces; campaign preparation, the first round (lowering and
    compiling every bank), warm round durations, terminal-task and
    successfully-labeled-task throughput, active-lane utilization,
    admissions/refills, parked lanes and publication backlog per round,
    capacity transfers, durable envelope bytes, the lowered and compiled
    dynamics bank advance (XLA cost/memory analysis) and process peak RSS;
    every terminal stage visit (outcome, consumption, method and campaign
    reason bits, and for relaxation its status, evaluations, residuals,
    energy and displacement), observed between ``evaluate`` and
    ``stage_outputs`` of each round (``run_round`` is executed as exactly
    those phases; the observation is excluded from round time); and every
    task's terminal stage, named outcome, reasons and stage-entry counts.
    Complete means every task is terminal, not that it succeeded.  Outputs
    are published to a throttled directory sink so backpressure is part of
    the measured workload.

``neb``
    Native fixed-cell periodic climbing-image NEB
    (`phydrax.chemistry.reaction.AtomisticNudgedElasticBandPlan`) of one mobile
    Lennard-Jones atom crossing the window of four fixed atoms, swept over
    image count and path-group capacity.  The endpoint minima are the roots of
    the exact one-dimensional crossing force (host bisection, independent of
    the native surface); the window centre is the saddle by symmetry, and the
    analytic barrier is the difference of the unswitched pair energies (every
    mobile pair distance at those points lies below the switch distance).
    Every group starts from a differently bowed band (bows cycle through
    ``PATH_BOWS``, so a larger group capacity adds groups and keeps the smaller
    capacity's groups), so measured executions perform real pending work.
    Recorded per case: host preparation, the ``init`` transaction (resource
    initialization and endpoint evaluation), lowering, compilation, first and
    warm executions of the single compiled band ``run`` entrypoint that
    ``solve`` uses (each execution restarts from the same freshly initialized
    state and runs until no group iterates), the actual active groups and
    per-group evaluations of every execution, compiler cost/memory analysis,
    logical retained state/resource/optimizer/runtime bytes, the public
    ``solve`` result and its agreement with the measured executable, per-group
    status, residual, barrier, transition-state position error and HVP
    saddle-inertia admission, plus log-log byte scaling exponents over group
    capacity and image count.  Refusals (``ValueError``) and runtime failures
    (``JaxRuntimeError``) are recorded with every phase reached before them;
    nothing is retried.

Smoke of the path scenario (explicitly unattested; the record identity is not
captured)::

    .venv/bin/python -c "from benchmarks.atomistic_dynamics_campaigns import \\
        _path_case; print(_path_case(5, 2, maximum_evaluations=2000, \\
        force_tolerance=1e-5, repeats=2))"

Full attested record::

    .venv/bin/python -m benchmarks.atomistic_dynamics_campaigns \\
        --scenarios campaign neb --label <label> --output <path>.json
"""

from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from enum import IntFlag
from functools import partial
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import fitted_scaling_exponent, process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.atomistic import _campaign as campaign
from phydrax.atomistic._relaxation import (
    AtomisticCartesianChart,
    AtomisticRelaxationConvergence,
    AtomisticRelaxationPlan,
    AtomisticRelaxationState,
    AtomisticRelaxationStatus,
    AtomisticSensitivityStatus,
    AtomisticTrialOutcome,
)
from phydrax.chemistry.reaction import (
    _path as reaction_path,
    AtomisticNudgedElasticBandPlan,
    AtomisticReactionPathState,
    PreparedAtomisticNudgedElasticBand,
    ReactionPathBranch,
)
from phydrax.optim import FIRE


ATOMS = phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS
UNITS = phx.atomistic.AtomisticUnitSystem.reduced()
DENSITY = 0.05
LJ_CUTOFF = 2.0
LJ_SWITCH = 1.8
# Campaign fixture (see the module docstring): bonded cluster length, the
# seeded per-coordinate perturbation amplitude, and the separation bound
# every task is checked against (the bond shortened by two perturbations).
FIXTURE_BOND = 1.3
FIXTURE_PERTURBATION = 0.1
FIXTURE_MINIMUM_SEPARATION = FIXTURE_BOND - 2.0 * FIXTURE_PERTURBATION

# Path scenario: four fixed atoms span a window in the x/z plane; the mobile
# atom crosses it along y.  Pair distances to the window are sqrt(1 + h^2) and
# sqrt(1.21 + h^2) at crossing height h.
PATH_BOX = 5.0
PATH_CENTER = np.full((3,), 0.5 * PATH_BOX)
PATH_WINDOW = PATH_CENTER + np.asarray(
    [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.1], [0.0, 0.0, 1.1]]
)
PATH_ATOMS = PATH_WINDOW.shape[0] + 1
PATH_BOWS = (0.15, 0.10, 0.05, 0.12)
PATH_CLIMBING_START = 0
PATH_MAXIMUM_STEP = 0.05
PATH_SPRING_CONSTANT = 1.0
PATH_MAXIMUM_IMAGES = 33
PATH_MAXIMUM_GROUPS = 64
PATH_MAXIMUM_EVALUATIONS = 20_000
PATH_MAXIMUM_REPEATS = 20


def _box(atoms: int, /) -> float:
    return max(5.0, float((atoms / DENSITY) ** (1.0 / 3.0)))


def _system(atoms: int, /) -> phx.atomistic.PreparedAtomisticSystem:
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(atoms),
        np.ones((atoms,), dtype=np.int32),
        np.ones((atoms,)),
        UNITS,
        atom_type_ids=np.zeros((atoms,), dtype=np.int32),
        cell=phx.discretization.PeriodicCell(_box(atoms) * np.eye(3)),
    ).prepare()


def _potential(
    system: phx.atomistic.PreparedAtomisticSystem, atoms: int, /
) -> phx.atomistic.PreparedAtomisticPotentialProgram:
    program = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([1.0], dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
                LJ_CUTOFF,
                switch_distance=LJ_SWITCH,
            )
        ]
    )
    (pairs,) = program.requirements.neighborhoods
    routes = max(16, 8 * atoms)
    search = phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=2 * routes,
        maximum_degree=32,
        maximum_images=27,
        maximum_case_edges=2 * routes,
    )
    bundle = phx.atomistic.AtomisticNeighborhoodBundlePlan(
        (pairs,),
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "pairs",
                ATOMS,
                (pairs,),
                phx.atomistic.AtomisticCellListImageSearch(
                    maximum_candidate_slots=10_000_000, deformation_margin=0.2
                ),
                phx.discretization.ParticleImageCapacityLadder((search,)),
                skin=0.3,
            ),
        ),
        (
            phx.atomistic.AtomisticRoleCapacityLadder(
                pairs,
                (
                    phx.atomistic.AtomisticRoleCapacity(
                        maximum_routes=routes, maximum_degree=32
                    ),
                ),
            ),
        ),
    )
    return program.prepare(system, neighborhood_bundle=bundle)


def _plan(
    atoms: int, lanes: int, tasks: int, steps: int, /
) -> campaign.AtomisticCampaignPlan:
    system = _system(atoms)
    relaxation = AtomisticRelaxationPlan(
        _potential(system, atoms),
        AtomisticCartesianChart(),
        method=FIRE(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-5, maximum_evaluations=300
        ),
        group_count=lanes,
    ).prepare()
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, _potential(system, atoms), phx.atomistic.BAOABLangevinPlan(2e-3, 1.0)
    ).prepare()
    rows = (
        phx.atomistic.AtomisticThermodynamicStatePlan(
            phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
            ensemble="nvt",
            temperature=0.5,
        ),
    )
    stages = (
        campaign.AtomisticStagePlan(
            "relax",
            {
                "gas": campaign.AtomisticRelaxationStageOperation(
                    relaxation, rounds_per_segment=50
                )
            },
            routing=campaign.AtomisticStageRouting(
                exhausted=campaign.AtomisticStageRoute("nvt", publish=False)
            ),
            lane_capacity=lanes,
        ),
        campaign.AtomisticStagePlan(
            "nvt",
            {
                "gas": campaign.AtomisticDynamicsStageOperation(
                    dynamics, rows, steps_per_round=10
                )
            },
            budget=steps,
            lane_capacity=lanes,
            queue_capacity=lanes,
        ),
        campaign.AtomisticStagePlan(
            "label",
            {"gas": campaign.AtomisticLabelStageOperation(_potential(system, atoms))},
            lane_capacity=lanes,
            queue_capacity=lanes,
            entry=campaign.AtomisticEntryPolicy.SHARE,
        ),
    )
    work = tuple(
        campaign.AtomisticCampaignTask(
            f"gas-{index}",
            "gas",
            campaign.AtomisticTaskPacket(positions=positions),
            material_id="lj-gas",
        )
        for index, positions in enumerate(_fixture(atoms, tasks))
    )
    return campaign.AtomisticCampaignPlan(
        f"benchmark-{atoms}-{lanes}", stages, work, max_pending_publications=lanes
    )


# -- campaign fixture -------------------------------------------------------------


def _sites_per_axis(atoms: int, /) -> int:
    """Smallest ``k`` with ``k^3 >= atoms`` (exact integer arithmetic)."""
    per_axis = 1
    while per_axis**3 < atoms:
        per_axis += 1
    return per_axis


def _snake_sites(per_axis: int, /) -> np.ndarray:
    """Lattice indices in boustrophedon order; consecutive sites are lattice neighbours."""
    order: list[tuple[int, int, int]] = []
    row = 0
    for x in range(per_axis):
        ys = range(per_axis) if x % 2 == 0 else range(per_axis - 1, -1, -1)
        for y in ys:
            zs = range(per_axis) if row % 2 == 0 else range(per_axis - 1, -1, -1)
            order.extend((x, y, z) for z in zs)
            row += 1
    return np.asarray(order, dtype=np.int64)


def _fixture_clusters(atoms: int, /) -> tuple[tuple[int, ...], ...]:
    """Consecutive-site dimers; an odd count makes the last three sites a trimer."""
    clusters: list[tuple[int, ...]] = [
        (2 * index, 2 * index + 1) for index in range(atoms // 2)
    ]
    if atoms % 2:
        clusters[-1] = (atoms - 3, atoms - 2, atoms - 1)
    return tuple(clusters)


def _fixture_sites(atoms: int, /) -> np.ndarray:
    """Unperturbed positions: every cluster contracted along its lattice bonds.

    A dimer keeps its midpoint; a trimer keeps its middle site.  Each atom
    moves only toward an adjacent lattice site of its own cluster, so atoms of
    different clusters stay at least the site spacing apart.
    """
    per_axis = _sites_per_axis(atoms)
    spacing = _box(atoms) / per_axis
    if spacing <= FIXTURE_BOND:
        raise ValueError("The campaign fixture needs a site spacing above its bond.")
    sites = (_snake_sites(per_axis)[:atoms] + 0.5) * spacing
    positions = sites.copy()
    for cluster in _fixture_clusters(atoms):
        if len(cluster) == 2:
            first, second = cluster
            middle = 0.5 * (sites[first] + sites[second])
            axis = (sites[second] - sites[first]) / spacing
            positions[first] = middle - 0.5 * FIXTURE_BOND * axis
            positions[second] = middle + 0.5 * FIXTURE_BOND * axis
        else:
            first, centre, last = cluster
            for end in (first, last):
                positions[end] = (
                    sites[centre] + FIXTURE_BOND * (sites[end] - sites[centre]) / spacing
                )
    return positions


def _fixture_seed(atoms: int, tasks: int, /) -> int:
    return atoms * 1000 + tasks


def _fixture(atoms: int, tasks: int, /) -> np.ndarray:
    """``(tasks, atoms, 3)`` wrapped initial positions, one seeded draw per task."""
    rng = np.random.default_rng(_fixture_seed(atoms, tasks))
    base = _fixture_sites(atoms)
    return np.stack(
        [
            base
            + rng.uniform(-FIXTURE_PERTURBATION, FIXTURE_PERTURBATION, size=(atoms, 3))
            for _ in range(tasks)
        ]
    )


def _fixture_geometry(atoms: int, positions: np.ndarray, /) -> dict[str, Any]:
    """Minimum-image pair geometry of one task against the LJ cutoff.

    The box exceeds twice the cutoff, so only the nearest image interacts.
    """
    box = _box(atoms)
    owner = np.empty((atoms,), dtype=np.int64)
    for index, members in enumerate(_fixture_clusters(atoms)):
        owner[list(members)] = index
    upper = np.triu(np.ones((atoms, atoms), dtype=np.bool_), k=1)
    same = owner[:, None] == owner[None, :]
    separation = positions[:, None, :] - positions[None, :, :]
    separation -= box * np.round(separation / box)
    distance = np.linalg.norm(separation, axis=-1)
    within = (distance < LJ_CUTOFF) & ~np.eye(atoms, dtype=np.bool_)
    intra = distance[upper & same]
    inter = distance[upper & ~same]
    return {
        "interacting_pairs": int(np.sum(within & upper)),
        "intra_cluster_interacting_pairs": int(np.sum(within & upper & same)),
        "inter_cluster_interacting_pairs": int(np.sum(within & upper & ~same)),
        "atoms_without_partner": int(np.sum(~np.any(within, axis=1))),
        "minimum_distance": float(np.min(distance[upper])),
        "minimum_intra_cluster_distance": float(np.min(intra)),
        "maximum_intra_cluster_distance": float(np.max(intra)),
        "minimum_inter_cluster_distance": (float(np.min(inter)) if inter.size else None),
    }


def _initial_evaluation(atoms: int, positions: np.ndarray, /) -> list[dict[str, Any]]:
    """Native energy and forces of every initial task configuration.

    The campaign's own Lennard-Jones program definition, neighborhoods
    initialized at each configuration; evaluated after the campaign rounds so
    it never enters their timing.
    """
    potential = _potential(_system(atoms), atoms)
    bundle = potential.neighborhood_bundle

    def evaluate(configuration: jax.Array) -> tuple[jax.Array, ...]:
        geometry = phx.atomistic.bind_atomistic_bundle_geometry(
            bundle, potential.system, configuration
        )
        result = potential.evaluate(configuration, bundle.initialize(geometry))
        return (
            result.energy,
            result.forces,
            jnp.all(result.successful),
            jnp.all(result.neighborhood_successful),
        )

    energy, forces, successful, neighborhoods = (
        np.asarray(value) for value in jax.jit(jax.vmap(evaluate))(jnp.asarray(positions))
    )
    norms = np.linalg.norm(forces, axis=-1)
    return [
        {
            "energy": _reported(energy[index]),
            "maximum_force": _reported(np.max(norms[index])),
            "rms_force": _reported(np.sqrt(np.mean(norms[index] ** 2))),
            "successful": bool(successful[index]),
            "neighborhood_successful": bool(neighborhoods[index]),
        }
        for index in range(positions.shape[0])
    ]


def _fixture_record(
    atoms: int, tasks: int, initial: Mapping[str, np.ndarray], /
) -> dict[str, Any]:
    """Fixture definition and per-task geometry; refuses a violated declared bound."""
    per_axis = _sites_per_axis(atoms)
    geometry = [_fixture_geometry(atoms, positions) for positions in initial.values()]
    for task_id, row in zip(initial, geometry, strict=True):
        row["task_id"] = task_id
        if (
            row["atoms_without_partner"]
            or row["minimum_distance"] < FIXTURE_MINIMUM_SEPARATION
        ):
            raise ValueError(
                f"Campaign fixture task {row['task_id']} violates its declared "
                "partner or separation bound."
            )
    return {
        "definition": (
            "boustrophedon k^3 lattice sites, consecutive-site dimers (odd: final "
            "trimer) contracted to the bond length, seeded uniform perturbation"
        ),
        "seed": _fixture_seed(atoms, tasks),
        "box_length": _box(atoms),
        "lattice_sites_per_axis": per_axis,
        "site_spacing": _box(atoms) / per_axis,
        "bond_length": FIXTURE_BOND,
        "perturbation": FIXTURE_PERTURBATION,
        "minimum_separation_bound": FIXTURE_MINIMUM_SEPARATION,
        "lj_cutoff": LJ_CUTOFF,
        "clusters": [list(cluster) for cluster in _fixture_clusters(atoms)],
        "tasks": geometry,
    }


# -- campaign evidence ------------------------------------------------------------


def _reported(value: Any, /) -> float | dict[str, str]:
    """Finite float, or the explicit nonfinite record of an exceptional value."""
    number = float(value)
    if math.isfinite(number):
        return number
    return {"kind": "nonfinite", "value": repr(number)}


def _flag_names(flags: type[IntFlag], value: int, /) -> list[str]:
    return [str(member.name) for member in flags(value)]


def _reasons(
    operation: campaign.AbstractAtomisticStageOperation,
    reasons: int,
    campaign_reasons: int,
    /,
) -> dict[str, Any]:
    """Raw and named method-owned and campaign reason bits of one stage visit."""
    if isinstance(operation, campaign.AtomisticRelaxationStageOperation):
        names = [AtomisticTrialOutcome(reasons).name]
    elif isinstance(operation, campaign.AtomisticDynamicsStageOperation):
        names = _flag_names(phx.atomistic.AtomisticStepRejectionReason, reasons)
    elif isinstance(operation, campaign.AtomisticLabelStageOperation):
        names = _flag_names(campaign.AtomisticCampaignReason, reasons)
    else:
        raise TypeError(f"Unexpected campaign operation {type(operation).__name__}.")
    return {
        "reasons": reasons,
        "reason_names": names,
        "campaign_reasons": campaign_reasons,
        "campaign_reason_names": _flag_names(
            campaign.AtomisticCampaignReason, campaign_reasons
        ),
    }


def _relaxation_work(
    lanes: AtomisticRelaxationState, slot: int, initial: np.ndarray, box: float, /
) -> dict[str, Any]:
    """Status, work, residuals and minimum-image displacement of one relaxed lane."""
    displacement = np.asarray(lanes.positions[slot], dtype=np.float64) - initial
    displacement -= box * np.round(displacement / box)
    return {
        "status": AtomisticRelaxationStatus(int(lanes.status[slot])).name,
        "evaluations": int(lanes.evaluations[slot]),
        "iterations": int(lanes.iterations[slot]),
        "rejected_trials": int(lanes.rejected_trials[slot]),
        "capacity_replays": int(lanes.capacity_replays[slot]),
        "energy": _reported(lanes.energy[slot]),
        "projected_residual": _reported(lanes.projected_residual[slot]),
        "full_force_residual": _reported(lanes.full_force_residual[slot]),
        "maximum_displacement": _reported(np.max(np.linalg.norm(displacement, axis=-1))),
    }


_LIVE = frozenset(
    {
        campaign.AtomisticStageOutcome.ACTIVE,
        campaign.AtomisticStageOutcome.CAPACITY_PENDING,
    }
)


def _visit_key(
    state: campaign.AtomisticCampaignState, task_id: str, stage: int, /
) -> tuple[str, int, int]:
    return task_id, stage, int(state.task(task_id).entries[stage])


def _runnable_visits(
    state: campaign.AtomisticCampaignState, /
) -> frozenset[tuple[str, int, int]]:
    """Visits ``evaluate`` advances from ``state`` (occupied, not held, active)."""
    return frozenset(
        _visit_key(state, task_id, int(bank.lane_stage[slot]))
        for bank in state.banks
        for slot, task_id in enumerate(bank.occupant)
        if task_id is not None
        and bank.held[slot] is None
        and int(bank.outcome[slot]) == campaign.AtomisticStageOutcome.ACTIVE
    )


def _observe_visits(
    prepared: campaign.PreparedAtomisticCampaign,
    state: campaign.AtomisticCampaignState,
    advanced: frozenset[tuple[str, int, int]],
    visits: dict[tuple[str, int, int], dict[str, Any]],
    initial: Mapping[str, np.ndarray],
    box: float,
    /,
) -> None:
    """Record every newly terminal lane of an evaluated state before staging vacates it.

    A visit is keyed by task, stage and stage entry, so a lane parked by
    backpressure is recorded once.  ``advanced`` are the visits this
    ``evaluate`` advanced; a lane refused at admission never advanced, so it
    carries no relaxation work (its lane state belongs to no run of its own).
    Relaxation displacement is measured from the task's initial packet (the
    relaxation stage is the entry stage).
    """
    for bank in state.banks:
        for slot, task_id in enumerate(bank.occupant):
            if task_id is None:
                continue
            outcome = campaign.AtomisticStageOutcome(int(bank.outcome[slot]))
            if outcome in _LIVE:
                continue
            stage = int(bank.lane_stage[slot])
            key = _visit_key(state, task_id, stage)
            if key in visits:
                continue
            visit: dict[str, Any] = {
                "task_id": task_id,
                "stage": prepared.plan.stages[stage].name,
                "entry": key[2],
                "round_index": state.round_index,
                "advanced": key in advanced,
                "outcome": outcome.name,
                "consumed": int(bank.consumed[slot]),
                **_reasons(
                    bank.operation,
                    int(bank.reasons[slot]),
                    int(bank.campaign_reasons[slot]),
                ),
            }
            if key in advanced and isinstance(bank.lanes, AtomisticRelaxationState):
                visit["relaxation"] = _relaxation_work(
                    bank.lanes, slot, initial[task_id], box
                )
            visits[key] = visit


def _task_outcome(
    prepared: campaign.PreparedAtomisticCampaign,
    record: campaign.AtomisticTaskRecord,
    /,
) -> dict[str, Any]:
    """Terminal stage, named outcome, reasons and stage-entry counts of one task."""
    stages = prepared.plan.stages
    stage = stages[record.stage]
    return {
        "task_id": record.task.task_id,
        "status": record.status.value,
        "stage": stage.name,
        "outcome": campaign.AtomisticStageOutcome(record.outcome).name,
        "consumed": record.consumed,
        **_reasons(
            stage.operations[record.task.family],
            record.reasons,
            record.campaign_reasons,
        ),
        "stage_entries": {
            plan.name: int(count)
            for plan, count in zip(stages, record.entries, strict=True)
        },
    }


def _finish_round(
    prepared: campaign.PreparedAtomisticCampaign,
    state: campaign.AtomisticCampaignState,
    sink: campaign.AtomisticDirectorySink,
    store: campaign.AtomisticCampaignStore,
    /,
) -> campaign.AtomisticCampaignState:
    """The ``run_round`` phases after ``evaluate``: stage, publish, commit, refill."""
    return prepared.refill(
        prepared.commit(prepared.publish(prepared.stage_outputs(state), sink), store)
    )


def _case(
    atoms: int, lanes: int, tasks: int, steps: int, throttle: int, /
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared, prepare_seconds = measure_host(
            lambda: _plan(atoms, lanes, tasks, steps).prepare()
        )
        initial = {
            task.task_id: np.asarray(task.packet.positions, dtype=np.float64)
            for task in prepared.plan.tasks
        }
        fixture = _fixture_record(atoms, tasks, initial)
        box = _box(atoms)
        sink = campaign.AtomisticDirectorySink(
            root / "outputs", accept_per_round=throttle
        )
        store = campaign.AtomisticCampaignStore(root / "campaign.ckpt")
        state = prepared.start(jax.random.key(atoms))
        rounds: list[dict[str, Any]] = []
        durations: list[float] = []
        visits: dict[tuple[str, int, int], dict[str, Any]] = {}
        compile_record: dict[str, Any] | None = None
        while not state.complete and len(rounds) < 10_000:
            advanced = _runnable_visits(state)
            evaluated, evaluate_seconds = measure_host(partial(prepared.evaluate, state))
            _observe_visits(prepared, evaluated, advanced, visits, initial, box)
            state, finish_seconds = measure_host(
                partial(_finish_round, prepared, evaluated, sink, store)
            )
            seconds = evaluate_seconds + finish_seconds
            durations.append(seconds)
            evidence = state.evidence
            rounds.append(
                {
                    **asdict(evidence),
                    "seconds": seconds,
                    "envelope_bytes": (root / "campaign.ckpt").stat().st_size,
                }
            )
            if compile_record is None:
                compile_record = _dynamics_compile(state)
        if not state.complete:
            raise RuntimeError("The benchmark campaign did not complete.")
        peak_rss = process_peak_rss_bytes()
        total = float(np.sum(durations))
        warm = durations[1:] if len(durations) > 1 else durations
        capacity = np.asarray(
            [item["lane_capacity"] for item in rounds], dtype=np.float64
        )
        occupied = np.asarray(
            [item["occupied_lanes"] for item in rounds], dtype=np.float64
        )
        outcomes = [_task_outcome(prepared, record) for record in state.tasks]
        terminal = sum(
            record.status is campaign.AtomisticTaskStatus.TERMINAL
            for record in state.tasks
        )
        labeled = sum(
            item["stage"] == "label"
            and item["outcome"] == campaign.AtomisticStageOutcome.COMPLETED.name
            for item in outcomes
        )
        evaluation = _initial_evaluation(atoms, np.stack(list(initial.values())))
        for row, values in zip(fixture["tasks"], evaluation, strict=True):
            row["initial_evaluation"] = values
        return {
            "atoms": atoms,
            "lanes": lanes,
            "tasks": tasks,
            "nvt_steps": steps,
            "sink_accept_per_round": throttle,
            "fixture": fixture,
            "prepare_seconds": prepare_seconds,
            "first_round_seconds": durations[0],
            "warm_round_seconds": {
                "median": float(np.median(warm)),
                "maximum": float(np.max(warm)),
                "count": len(warm),
            },
            "total_seconds": total,
            "terminal_tasks": terminal,
            "labeled_tasks": labeled,
            "terminal_tasks_per_second": terminal / total,
            "labeled_tasks_per_second": labeled / total,
            "rounds": len(rounds),
            "mean_active_lane_utilization": float(
                np.mean(
                    np.divide(
                        occupied,
                        capacity,
                        out=np.zeros_like(occupied),
                        where=capacity > 0,
                    )
                )
            ),
            "admissions": int(sum(item["admitted"] for item in rounds)),
            "parked_lane_rounds": int(sum(item["parked"] for item in rounds)),
            "maximum_pending": int(max(item["pending"] for item in rounds)),
            "capacity_transfers": int(sum(item["capacity_transfers"] for item in rounds)),
            "capacity_growths": int(sum(item["capacity_growths"] for item in rounds)),
            "shared_evaluations": int(sum(item["shared_evaluations"] for item in rounds)),
            "published": len(state.receipts),
            "maximum_envelope_bytes": int(max(item["envelope_bytes"] for item in rounds)),
            "terminal_outcomes": dict(
                Counter(f"{item['stage']}:{item['outcome']}" for item in outcomes)
            ),
            "visit_outcomes": dict(
                Counter(f"{item['stage']}:{item['outcome']}" for item in visits.values())
            ),
            "task_outcomes": outcomes,
            "stage_visits": list(visits.values()),
            "dynamics_bank": compile_record,
            "per_round": rounds,
            "process_peak_rss_bytes": peak_rss,
        }


def _dynamics_compile(state: campaign.AtomisticCampaignState, /) -> dict[str, Any] | None:
    for bank in state.banks:
        if bank.lanes is None or not isinstance(
            bank.operation, campaign.AtomisticDynamicsStageOperation
        ):
            continue
        active = jnp.asarray(bank.occupied)
        arguments = (
            bank.operation,
            bank.lanes,
            active,
            jnp.asarray(bank.budget),
            jnp.asarray(bank.consumed),
        )
        # Equinox's annotation erases its documented compiled .lower interface.
        lowering = campaign._ADVANCE.lower  # ty: ignore[unresolved-attribute]
        executable, phases = measure_lower_and_compile(
            partial(lowering, *arguments), methodcaller("compile")
        )
        evidence = compiler_evidence(
            executable.compiled.cost_analysis(),
            executable.compiled.memory_analysis(),
            source="xla",
        )
        return {
            "signature_id": bank.operation.signature(bank.lane_count).signature_id,
            "phases": asdict(phases),
            "compiler": asdict(evidence),
            "logical_bank_bytes": logical_array_bytes(bank.lanes),
        }
    return None


# -- grouped periodic NEB path scenario -------------------------------------------


_PATH_SCALED_BYTES = (
    "logical_state_bytes",
    "logical_resources_bytes",
    "logical_optimizer_bytes",
    "logical_runtime_bytes",
    "compiler_argument_bytes",
    "compiler_output_bytes",
    "compiler_temporary_bytes",
    "compiler_generated_code_bytes",
)


def _pair_energy(distance: float, /) -> float:
    """Unswitched reduced Lennard-Jones pair energy (epsilon = sigma = 1)."""
    return 4.0 * (distance**-12 - distance**-6)


def _pair_derivative(distance: float, /) -> float:
    return 4.0 * (-12.0 * distance**-13 + 6.0 * distance**-7)


def _window_distances(height: float, /) -> tuple[float, float]:
    return math.hypot(1.0, height), math.hypot(1.1, height)


def _window_energy(height: float, /) -> float:
    """Mobile-atom energy at crossing height ``height``; fixed pairs are constant."""
    first, second = _window_distances(height)
    return 2.0 * (_pair_energy(first) + _pair_energy(second))


def _crossing_force(height: float, /) -> float:
    """Exact derivative of `_window_energy` along the crossing axis."""
    first, second = _window_distances(height)
    return (
        2.0
        * height
        * (_pair_derivative(first) / first + _pair_derivative(second) / second)
    )


def _window_minimum_height() -> float:
    """Endpoint crossing height: the bisected root of the exact crossing force.

    The analytic control is exact for the switched native potential only while
    every mobile pair distance at the endpoints and at the saddle lies below
    the switch distance; that is checked, not assumed.
    """
    low, high = 0.05, 1.0
    if not _crossing_force(low) < 0.0 < _crossing_force(high):
        raise ValueError("The crossing-force root is not bracketed.")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _crossing_force(middle) < 0.0:
            low = middle
        else:
            high = middle
    height = 0.5 * (low + high)
    if max(*_window_distances(height), *_window_distances(0.0)) >= LJ_SWITCH:
        raise ValueError("The analytic control needs every mobile pair below the switch.")
    return height


def _finite(value: float, /) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_values(values: jax.Array | np.ndarray, /) -> list[float | None]:
    return [_finite(value) for value in np.ravel(np.asarray(values, dtype=np.float64))]


def _require_path_controls(
    images: int,
    groups: int,
    maximum_evaluations: int,
    force_tolerance: float,
    repeats: int,
    /,
) -> None:
    if not 3 <= images <= PATH_MAXIMUM_IMAGES:
        raise ValueError(f"NEB image counts must lie in [3, {PATH_MAXIMUM_IMAGES}].")
    if not 1 <= groups <= PATH_MAXIMUM_GROUPS:
        raise ValueError(f"NEB group capacities must lie in [1, {PATH_MAXIMUM_GROUPS}].")
    if not 1 <= maximum_evaluations <= PATH_MAXIMUM_EVALUATIONS:
        raise ValueError(
            f"NEB maximum evaluations must lie in [1, {PATH_MAXIMUM_EVALUATIONS}]."
        )
    if not math.isfinite(force_tolerance) or force_tolerance <= 0.0:
        raise ValueError("The NEB force tolerance must be positive and finite.")
    if not 1 <= repeats <= PATH_MAXIMUM_REPEATS:
        raise ValueError(f"NEB repeats must lie in [1, {PATH_MAXIMUM_REPEATS}].")


def _path_runtime(
    images: int, groups: int, maximum_evaluations: int, force_tolerance: float, /
) -> PreparedAtomisticNudgedElasticBand:
    """Window system, explicit-capacity LJ program and the prepared band plan."""
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(PATH_ATOMS),
        np.ones((PATH_ATOMS,), dtype=np.int32),
        np.ones((PATH_ATOMS,)),
        UNITS,
        atom_type_ids=np.zeros((PATH_ATOMS,), dtype=np.int32),
        mobile_mask=np.arange(PATH_ATOMS) == PATH_ATOMS - 1,
        cell=phx.discretization.PeriodicCell(PATH_BOX * np.eye(3)),
    ).prepare()
    surface = phx.chemistry.AtomisticPotentialEnergySurface(
        system, _potential(system, PATH_ATOMS)
    )
    return AtomisticNudgedElasticBandPlan(
        surface,
        image_count=images,
        spring_constant=PATH_SPRING_CONSTANT,
        force_tolerance=force_tolerance,
        maximum_evaluations=maximum_evaluations,
        climbing_start=PATH_CLIMBING_START,
        group_count=groups,
        method=FIRE(maximum_step=PATH_MAXIMUM_STEP),
    ).prepare()


def _path_structure(positions: np.ndarray, /) -> phx.atomistic.AtomicStructure:
    return phx.atomistic.AtomicStructure(
        np.ones((PATH_ATOMS,), dtype=np.int32),
        positions,
        np.ones((PATH_ATOMS,)),
        UNITS.scale,
        particle_ids=np.arange(PATH_ATOMS),
        cell=PATH_BOX * np.eye(3),
        periodic_axes=np.ones((3,), dtype=np.bool_),
    )


def _path_bands(
    images: int, groups: int, height: float, /
) -> tuple[
    list[tuple[phx.atomistic.AtomicStructure, phx.atomistic.AtomicStructure]],
    np.ndarray,
]:
    """Shared analytic endpoints and one bowed initial band per group."""
    offset = np.asarray([0.0, height, 0.0])
    reactant = np.concatenate((PATH_WINDOW, (PATH_CENTER + offset)[None]))
    product = np.concatenate((PATH_WINDOW, (PATH_CENTER - offset)[None]))
    fractions = np.linspace(0.0, 1.0, images)
    band = reactant[None] + fractions[:, None, None] * (product - reactant)[None]
    bands = np.repeat(band[None], groups, axis=0)
    bows = np.asarray([PATH_BOWS[group % len(PATH_BOWS)] for group in range(groups)])
    bands[:, 1:-1, -1, 0] += bows[:, None] * np.sin(np.pi * fractions[1:-1])[None]
    pair = (_path_structure(reactant), _path_structure(product))
    return [pair] * groups, bands


def _path_work(state: AtomisticReactionPathState, images: int, /) -> dict[str, Any]:
    """Actual group activity and consumed band evaluations of one state.

    ``loop_rounds`` is derived: every round either consumes one evaluation of
    an iterating group or parks it once as capacity-pending.
    """
    evaluations, iterations, replays, rejected, status, active = jax.device_get(
        (
            state.evaluations,
            state.iterations,
            state.capacity_replays,
            state.rejected_trials,
            state.status,
            state.active,
        )
    )
    consumed = int(np.sum(evaluations))
    rounds = int(np.max(evaluations + replays))
    return {
        "active_groups": int(np.count_nonzero(active)),
        "iterating_groups": int(
            np.count_nonzero(status == int(AtomisticRelaxationStatus.ITERATING))
        ),
        "status": [AtomisticRelaxationStatus(int(value)).name for value in status],
        "evaluations": np.asarray(evaluations).tolist(),
        "iterations": np.asarray(iterations).tolist(),
        "capacity_replays": np.asarray(replays).tolist(),
        "rejected_trials": np.asarray(rejected).tolist(),
        "consumed_band_evaluations": consumed,
        "interior_image_evaluations": consumed * (images - 2),
        "loop_rounds": rounds,
        "group_round_utilization": (
            None if rounds == 0 else consumed / (len(evaluations) * rounds)
        ),
    }


def _path_group(
    runtime: PreparedAtomisticNudgedElasticBand,
    state: AtomisticReactionPathState,
    group: int,
    barrier: float,
    /,
) -> dict[str, Any]:
    """Accepted-band observation and HVP saddle inertia of one group."""
    path = runtime.path_result(state, group)
    inertia = runtime.transition_state_inertia(state, group)
    energies = np.asarray(path.energies, dtype=np.float64)
    climbing = path.climbing_image
    observed = _finite(energies[climbing] - energies[0])
    return {
        "group": group,
        "status": AtomisticRelaxationStatus(int(path.status)).name,
        "successful": bool(path.successful),
        "iterations": int(path.iterations),
        "provider_image_evaluations": int(path.provider_evaluations),
        "climbing_active": path.climbing_active,
        "climbing_image": climbing,
        "maximum_interior_residual": _finite(
            np.max(np.asarray(path.image_residual)[1:-1])
        ),
        "barrier": observed,
        "barrier_error": None if observed is None else observed - barrier,
        "transition_state_error": _finite(
            np.linalg.norm(np.asarray(path.transition_state_positions)[-1] - PATH_CENTER)
        ),
        "path_result_id": path.result_id,
        "saddle": {
            "status": AtomisticSensitivityStatus(int(inertia.status)).name,
            "negative_count": int(inertia.negative_count),
            "curvatures": _finite_values(inertia.curvatures),
            "stationarity": _finite_values(inertia.stationarity),
            "gauge_residual": _finite_values(inertia.gauge_residual),
            "solve_status": np.asarray(inertia.solve_status).tolist(),
        },
    }


def _measure_path(
    record: dict[str, Any],
    images: int,
    groups: int,
    maximum_evaluations: int,
    force_tolerance: float,
    repeats: int,
    /,
) -> None:
    """Fill ``record`` phase by phase so a refusal keeps every reached phase."""
    height, analytic_seconds = measure_host(_window_minimum_height)
    barrier = _window_energy(0.0) - _window_energy(height)
    record["analytic"] = {
        "endpoint_height": height,
        "saddle_position": PATH_CENTER.tolist(),
        "barrier": barrier,
        "seconds": analytic_seconds,
    }
    prepared, prepare_seconds = measure_host(
        partial(_path_runtime, images, groups, maximum_evaluations, force_tolerance)
    )
    record["prepare_seconds"] = prepare_seconds
    record["plan_id"] = prepared.plan.plan_id
    endpoints, initial = _path_bands(images, groups, height)
    (runtime, state, certificates), init_seconds = measure_synchronized(
        partial(
            prepared.init,
            endpoints,
            branch=ReactionPathBranch.EXPLICIT,
            initial_images=initial,
        )
    )
    record["init_seconds"] = init_seconds
    record["init_rebound_capacity"] = runtime.prepared_id != prepared.prepared_id
    record["prepared_id"] = runtime.prepared_id
    record["layout_id"] = runtime.layout_id
    record["branch"] = {
        "policy": ReactionPathBranch.EXPLICIT.value,
        "certificate_ids": [certificate.certificate_id for certificate in certificates],
    }
    memory: dict[str, int | None] = {
        "logical_state_bytes": logical_array_bytes(state),
        "logical_resources_bytes": logical_array_bytes(state.resources),
        "logical_optimizer_bytes": logical_array_bytes(state.optimizer),
        "logical_runtime_bytes": logical_array_bytes(runtime),
    }
    record["memory"] = memory
    record["start_work"] = _path_work(state, images)
    # The single compiled run entrypoint of `solve`; every execution below
    # restarts from the same initialized state and runs until no group iterates.
    arguments = (runtime, state, maximum_evaluations)
    # Equinox's annotation erases its documented compiled .lower interface.
    lowering = reaction_path._RUN_BAND_ROUNDS.lower  # ty: ignore[unresolved-attribute]
    executable, compilation = measure_lower_and_compile(
        partial(lowering, *arguments), methodcaller("compile")
    )
    record["lowering_seconds"] = compilation.lowering_seconds
    record["compilation_seconds"] = compilation.compilation_seconds
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="xla",
    )
    record["compiler"] = asdict(evidence)
    memory.update(
        compiler_argument_bytes=evidence.argument_bytes,
        compiler_output_bytes=evidence.output_bytes,
        compiler_temporary_bytes=evidence.temporary_bytes,
        compiler_generated_code_bytes=evidence.generated_code_bytes,
    )
    first, first_seconds = measure_synchronized(partial(executable, *arguments))
    record["first_seconds"] = first_seconds
    record["first_work"] = _path_work(first, images)
    last, warm = measure_repeated(
        partial(executable, *arguments), warmup=0, repeats=repeats
    )
    record["warm"] = warm.to_seconds_dict()
    record["warm_work"] = _path_work(last, images)
    median = warm.median_seconds
    record["warm_interior_image_evaluations_per_second"] = (
        None if not median else record["warm_work"]["interior_image_evaluations"] / median
    )
    (solved, final), solve_seconds = measure_synchronized(
        partial(runtime.solve, state, maximum_rounds=maximum_evaluations)
    )
    record["solve"] = {
        "seconds": solve_seconds,
        "seconds_scope": (
            "public host driver: its own dispatch of the run entrypoint, including "
            "any compilation that dispatch performs, and every grow transaction"
        ),
        "rebound_capacity": solved.prepared_id != runtime.prepared_id,
        "agrees_with_measured_executable": bool(
            np.array_equal(np.asarray(final.status), np.asarray(first.status))
            and np.array_equal(
                np.asarray(final.evaluations), np.asarray(first.evaluations)
            )
            and np.array_equal(np.asarray(final.images), np.asarray(first.images))
        ),
        "work": _path_work(final, images),
    }
    observed, observe_seconds = measure_synchronized(
        lambda: [_path_group(solved, final, group, barrier) for group in range(groups)]
    )
    record["observation_seconds"] = observe_seconds
    record["groups_observed"] = observed
    converged = AtomisticRelaxationStatus.CONVERGED.name
    admitted = AtomisticSensitivityStatus.ADMITTED.name
    record["saddle_admitted_groups"] = sum(
        item["saddle"]["status"] == admitted and item["saddle"]["negative_count"] == 1
        for item in observed
    )
    record["status"] = (
        "converged"
        if all(item["status"] == converged for item in observed)
        else "unconverged"
    )


def _path_case(
    images: int,
    groups: int,
    /,
    *,
    maximum_evaluations: int,
    force_tolerance: float,
    repeats: int,
) -> dict[str, Any]:
    """One grouped periodic NEB case; refusals and failures are rows, never retried."""
    _require_path_controls(images, groups, maximum_evaluations, force_tolerance, repeats)
    record: dict[str, Any] = {
        "images": images,
        "groups": groups,
        "atoms": PATH_ATOMS,
        "mobile_atoms": 1,
        "bows": [PATH_BOWS[group % len(PATH_BOWS)] for group in range(groups)],
        "maximum_evaluations": maximum_evaluations,
        "force_tolerance": force_tolerance,
        "spring_constant": PATH_SPRING_CONSTANT,
        "climbing_start": PATH_CLIMBING_START,
        "method": {"kind": "FIRE", "maximum_step": PATH_MAXIMUM_STEP},
        "repeats": repeats,
        "status": "incomplete",
    }
    try:
        _measure_path(
            record, images, groups, maximum_evaluations, force_tolerance, repeats
        )
    except ValueError as error:
        record["status"] = "refused"
        record["error"] = {"type": type(error).__qualname__, "message": str(error)}
    except jax.errors.JaxRuntimeError as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__qualname__, "message": str(error)}
    record["process_peak_rss_bytes"] = process_peak_rss_bytes()
    return record


def _path_scaling(cases: list[dict[str, Any]], /) -> dict[str, Any]:
    """Log-log byte exponents over group capacity and over image count.

    An exponent is ``None`` when any case of its series lacks a positive
    measured value (refused, failed or compiler-unavailable rows included).
    """
    summary: dict[str, Any] = {}
    for axis, fixed in (("groups", "images"), ("images", "groups")):
        series: dict[int, list[dict[str, Any]]] = {}
        for case in cases:
            series.setdefault(case[fixed], []).append(case)
        rows: list[dict[str, Any]] = []
        for value, members in sorted(series.items()):
            ordered = sorted(members, key=itemgetter(axis))
            if len(ordered) < 2:
                continue
            capacities = [case[axis] for case in ordered]
            exponents: dict[str, float | None] = {}
            for key in _PATH_SCALED_BYTES:
                values = [case.get("memory", {}).get(key) for case in ordered]
                exponents[key] = (
                    fitted_scaling_exponent(capacities, values)
                    if all(isinstance(item, int) and item > 0 for item in values)
                    else None
                )
            rows.append({fixed: value, axis: capacities, "exponents": exponents})
        summary[f"over_{axis}"] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("campaign", "neb"),
        default=["campaign", "neb"],
    )
    parser.add_argument("--atoms", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--lanes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sink-accept-per-round", type=int, default=2)
    parser.add_argument("--neb-images", nargs="+", type=int, default=[5, 9])
    parser.add_argument("--neb-groups", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--neb-maximum-evaluations", type=int, default=4000)
    parser.add_argument("--neb-force-tolerance", type=float, default=1e-5)
    parser.add_argument("--neb-repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenarios = sorted(set(args.scenarios))
    if (
        any(count < 2 for count in args.atoms)
        or any(count < 1 for count in args.lanes)
        or args.tasks < 1
        or args.steps < 1
        or args.sink_accept_per_round < 1
    ):
        raise ValueError(
            "Atom counts >= 2; lanes, tasks, steps and sink throughput >= 1."
        )
    path_images = sorted(set(args.neb_images))
    path_groups = sorted(set(args.neb_groups))
    for images in path_images:
        for groups in path_groups:
            _require_path_controls(
                images,
                groups,
                args.neb_maximum_evaluations,
                args.neb_force_tolerance,
                args.neb_repeats,
            )
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/atomistic/_campaign.py",
        "phydrax/atomistic/_dynamics.py",
        "phydrax/atomistic/_relaxation.py",
        "phydrax/_execution_pool.py",
        "phydrax/_execution_workset.py",
        "phydrax/chemistry/_atomistic.py",
        "phydrax/chemistry/reaction/_path.py",
        "phydrax/optim/_fire.py",
    )
    record: dict[str, Any] = {
        "benchmark": "atomistic-dynamics-campaigns",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            (
                "cases",
                "environment",
                "neb_cases",
                "neb_scaling",
                "scenarios",
                "source_sha256",
            ),
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "scenarios": scenarios,
        "cases": (
            [
                _case(atoms, lanes, args.tasks, args.steps, args.sink_accept_per_round)
                for atoms in args.atoms
                for lanes in args.lanes
            ]
            if "campaign" in scenarios
            else []
        ),
        "neb_cases": (
            [
                _path_case(
                    images,
                    groups,
                    maximum_evaluations=args.neb_maximum_evaluations,
                    force_tolerance=args.neb_force_tolerance,
                    repeats=args.neb_repeats,
                )
                for images in path_images
                for groups in path_groups
            ]
            if "neb" in scenarios
            else []
        ),
    }
    record["neb_scaling"] = _path_scaling(record["neb_cases"])
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
