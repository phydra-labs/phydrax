#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heterogeneous relax -> NVT -> NPT -> label campaign with a fresh-process restart.

Two Lennard-Jones families (dimers and tetramers in periodic boxes) flow through
four native stages: FIRE relaxation, BAOAB NVT, MTK NPT and E/F/S labeling.  The
NPT stage changes integrator and ensemble, so it declares an explicit new run
(`AtomisticContinuationPolicy.RESTART`) from the NVT phase point instead of
pretending to continue the Langevin trajectory; the label stage observes the
NPT walker's configuration and keeps its complete dynamics continuation.  The
label stage reuses the NPT force evaluation when its Hamiltonian, revision,
conditions and requested quantities agree.  One task fails scientifically
(coincident atoms), one is capacity limited (its NVT bank declares a role
capacity too small for a relaxed tetramer and the task is rebucketed to the next
declared capacity), and the output sink accepts one record per round so lanes
park behind the bounded pending-publication bank.

Run without arguments: a first process executes several rounds and exits
after publishing but before committing its final round. A second, fresh
process resumes from the last durable commit and finishes. The driver checks
that every output was published exactly once; this is not a SIGKILL test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np

import phydrax as phx
from phydrax.atomistic import (
    AtomisticCampaignPlan,
    AtomisticCampaignReason,
    AtomisticCampaignStore,
    AtomisticCampaignTask,
    AtomisticCartesianChart,
    AtomisticContinuationPolicy,
    AtomisticDirectorySink,
    AtomisticDynamicsStageOperation,
    AtomisticEntryPolicy,
    AtomisticLabelStageOperation,
    AtomisticRelaxationConvergence,
    AtomisticRelaxationPlan,
    AtomisticRelaxationStageOperation,
    AtomisticStagePlan,
    AtomisticStageRoute,
    AtomisticStageRouting,
    AtomisticTaskPacket,
)
from phydrax.optim import FIRE


ATOMS = phx.atomistic.AtomisticSiteDomain.PHYSICAL_ATOMS
UNITS = phx.atomistic.AtomisticUnitSystem.reduced()
BOX = 5.0
LANES = 2
SEARCH = phx.discretization.ParticleImageCapacity(
    maximum_particles_per_cell=16,
    maximum_edges=256,
    maximum_degree=32,
    maximum_images=27,
    maximum_case_edges=256,
)


def _role(routes: int) -> phx.atomistic.AtomisticRoleCapacity:
    return phx.atomistic.AtomisticRoleCapacity(
        maximum_routes=routes, maximum_degree=routes
    )


def _system(atoms: int) -> phx.atomistic.PreparedAtomisticSystem:
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(atoms),
        np.ones((atoms,), dtype=np.int32),
        np.ones((atoms,)),
        UNITS,
        atom_type_ids=np.zeros((atoms,), dtype=np.int32),
        cell=phx.discretization.PeriodicCell(BOX * np.eye(3)),
    ).prepare()


def _potential(
    system: phx.atomistic.PreparedAtomisticSystem, roles: tuple[int, ...] = (256,)
) -> phx.atomistic.PreparedAtomisticPotentialProgram:
    program = phx.atomistic.AtomisticPotentialProgram(
        [
            phx.atomistic.LennardJonesPotential(
                np.asarray([1.0]), np.asarray([1.0]), 2.0, switch_distance=1.8
            )
        ]
    )
    (pairs,) = program.requirements.neighborhoods
    plan = phx.atomistic.AtomisticNeighborhoodBundlePlan(
        (pairs,),
        (
            phx.atomistic.AtomisticNeighborhoodSource(
                "lennard-jones-pairs",
                ATOMS,
                (pairs,),
                phx.atomistic.AtomisticCellListImageSearch(
                    maximum_candidate_slots=1_000_000, deformation_margin=0.2
                ),
                phx.discretization.ParticleImageCapacityLadder((SEARCH,)),
                skin=0.3,
            ),
        ),
        (
            phx.atomistic.AtomisticRoleCapacityLadder(
                pairs, tuple(_role(value) for value in roles)
            ),
        ),
    )
    return program.prepare(system, neighborhood_bundle=plan)


def _measure(
    system: phx.atomistic.PreparedAtomisticSystem,
) -> phx.atomistic.AtomisticPhaseSpaceMeasurePlan:
    return phx.atomistic.AtomisticPhaseSpaceMeasurePlan(
        system, volume_coordinate_convention="atomic"
    )


def _relax(
    system: phx.atomistic.PreparedAtomisticSystem,
) -> AtomisticRelaxationStageOperation:
    plan = AtomisticRelaxationPlan(
        _potential(system),
        AtomisticCartesianChart(),
        method=FIRE(maximum_step=0.05),
        convergence=AtomisticRelaxationConvergence(
            projected_tolerance=1e-6, maximum_evaluations=500
        ),
        group_count=LANES,
    )
    return AtomisticRelaxationStageOperation(plan.prepare(), rounds_per_segment=50)


def _nvt(
    system: phx.atomistic.PreparedAtomisticSystem, roles: tuple[int, ...] = (256,)
) -> AtomisticDynamicsStageOperation:
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, _potential(system, roles), phx.atomistic.BAOABLangevinPlan(2e-3, 1.0)
    ).prepare()
    rows = (
        phx.atomistic.AtomisticThermodynamicStatePlan(
            _measure(system), ensemble="nvt", temperature=0.2
        ),
    )
    return AtomisticDynamicsStageOperation(dynamics, rows, steps_per_round=10)


def _npt(
    system: phx.atomistic.PreparedAtomisticSystem,
) -> AtomisticDynamicsStageOperation:
    integrator = phx.atomistic.MTKNPTPlan(
        1e-3,
        phx.atomistic.MTKBarostatPlan(1.0),
        phx.atomistic.NoseHooverChainParameters(0.1),
        phx.atomistic.NoseHooverChainParameters(1.0),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, _potential(system), integrator
    ).prepare()
    rows = (
        phx.atomistic.AtomisticThermodynamicStatePlan(
            _measure(system), ensemble="npt", temperature=0.2, pressure=0.05
        ),
    )
    return AtomisticDynamicsStageOperation(dynamics, rows, steps_per_round=10)


def build_plan() -> AtomisticCampaignPlan:
    dimer, tetramer = _system(2), _system(4)
    stages = (
        AtomisticStagePlan(
            "relax",
            {"dimer": _relax(dimer), "tetramer": _relax(tetramer)},
            routing=AtomisticStageRouting(
                exhausted=AtomisticStageRoute("nvt", publish=True)
            ),
            lane_capacity=LANES,
        ),
        AtomisticStagePlan(
            "nvt",
            # Two directed-half routes hold a dimer but not a relaxed tetramer:
            # tetramers are rebucketed to the next declared role capacity.
            {"dimer": _nvt(dimer, (2, 64)), "tetramer": _nvt(tetramer, (2, 64))},
            budget=40,
            lane_capacity=LANES,
            queue_capacity=2,
        ),
        AtomisticStagePlan(
            "npt",
            {"dimer": _npt(dimer), "tetramer": _npt(tetramer)},
            budget=20,
            lane_capacity=LANES,
            queue_capacity=2,
            continuation=AtomisticContinuationPolicy.RESTART,
        ),
        AtomisticStagePlan(
            "label",
            {
                "dimer": AtomisticLabelStageOperation(_potential(dimer), stress=True),
                "tetramer": AtomisticLabelStageOperation(
                    _potential(tetramer), stress=True
                ),
            },
            lane_capacity=LANES,
            queue_capacity=1,
            entry=AtomisticEntryPolicy.SHARE,
        ),
    )
    rng = np.random.default_rng(4)
    tasks = []
    for index in range(3):
        start = rng.uniform(1.0, 4.0, size=(1, 3))
        positions = np.concatenate((start, start + [[1.3, 0.1 * index, 0.0]]))
        tasks.append(
            AtomisticCampaignTask(
                f"dimer-{index}",
                "dimer",
                AtomisticTaskPacket(positions=positions),
                material_id="dimer",
            )
        )
    tetrahedron = np.asarray(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0], [0.6, 0.35, 1.0]]
    )
    for index in range(2):
        tasks.append(
            AtomisticCampaignTask(
                f"tetramer-{index}",
                "tetramer",
                AtomisticTaskPacket(positions=tetrahedron + 1.5 + 0.2 * index),
                material_id="tetramer",
            )
        )
    tasks.append(
        AtomisticCampaignTask(
            "dimer-coincident",
            "dimer",
            AtomisticTaskPacket(positions=np.asarray([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])),
            material_id="dimer",
        )
    )
    return AtomisticCampaignPlan(
        "lj-relaxation-campaign", stages, tuple(tasks), max_pending_publications=2
    )


def _process(work: Path, phase: str, rounds: int) -> None:
    prepared = build_plan().prepare()
    store = AtomisticCampaignStore(work / "campaign.ckpt")
    sink = AtomisticDirectorySink(work / "outputs", accept_per_round=1)
    if phase == "interrupt":
        state = prepared.start(jax.random.key(2026))
        for _ in range(rounds):
            state = prepared.run_round(state, sink, store)
        # Lose the process after publishing, before the commit is durable.
        prepared.publish(prepared.stage_outputs(prepared.evaluate(state)), sink)
        print(json.dumps({"phase": phase, "durable_round": state.round_index - 1}))
        return
    state = prepared.resume(store)
    parked = backpressured = 0
    for _ in range(2000):
        if state.complete:
            break
        state = prepared.run_round(state, sink, store)
        parked += state.evidence.parked
        backpressured += int(state.evidence.backpressured)
    if not state.complete:
        raise RuntimeError("The campaign did not finish within its round bound.")
    outcomes = {f"{record.task.task_id}": record.outcome for record in state.tasks}
    print(
        json.dumps(
            {
                "phase": phase,
                "rounds": state.round_index,
                "receipts": len(state.receipts),
                "parked_lane_rounds": parked,
                "backpressured_rounds": backpressured,
                "final_outcomes": outcomes,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--phase", choices=("interrupt", "resume"))
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.phase is not None:
        if args.work is None:
            raise ValueError("--phase requires --work.")
        _process(args.work, args.phase, args.rounds)
        return
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for phase in ("interrupt", "resume"):
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--work",
                    str(work),
                    "--phase",
                    phase,
                    "--rounds",
                    str(args.rounds),
                ],
                check=True,
            )
        outputs = AtomisticDirectorySink(work / "outputs").outputs()
        keys = [output.semantic_key for output in outputs.values()]
        if len(keys) != len(set(keys)):
            raise RuntimeError("A campaign output was published twice.")
        summary = {}
        for output in sorted(
            outputs.values(), key=lambda item: (item.task_id, item.stage)
        ):
            summary[f"{output.task_id}/{output.stage}"] = {
                "outcome": output.outcome.name,
                "shared_evaluation": bool(
                    output.campaign_reasons
                    & int(AtomisticCampaignReason.SHARED_EVALUATION)
                ),
                "dynamics_restarted": bool(
                    output.campaign_reasons
                    & int(AtomisticCampaignReason.DYNAMICS_RESTARTED)
                ),
                "energy": None
                if "energy" not in output.ports
                else float(output.ports["energy"]),
            }
        print(json.dumps({"outputs": len(keys), "records": summary}, indent=2))


if __name__ == "__main__":
    main()
