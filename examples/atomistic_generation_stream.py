# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded, resumable native generation stream with exact restart.

A stateful sampler family (one overlap-relief packer per admitted space-group
setting, routed per attempt ordinal by a declared prior) feeds an
`AtomisticGenerationStreamPlan`.  The example shows

* zero-target and zero-budget calls that draw nothing;
* a finite segment that is checkpointed mid-flight through the typed
  array-artifact owner, restored in a fresh stream object and continued to
  the identical ordinal-ordered output;
* an explicit continuation segment (`extend`) instead of an unbounded loop;
* typed rejections (step budget, filters) published with their reasons and
  the explicit shortfall.

Proposal keys and identities follow (campaign, support, sampler, ordinal),
never lane or completion order.  Packing acceptance is geometric overlap
relief, not physical qualification.

Run: ``python examples/atomistic_generation_stream.py``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import jax.random as jr
import numpy as np

from phydrax import NumericRevision
from phydrax.atomistic import (
    AtomisticDenseImageSearch,
    AtomisticGeneratedProposal,
    AtomisticMemoryGenerationSink,
    AtomisticMinimumDistanceFilter,
    AtomisticProposalStatus,
    AtomisticRoleCapacity,
)
from phydrax.atomistic.crystal import (
    ConformerPool,
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
    MolecularComponent,
    MolecularConformer,
    MolecularCrystalInput,
    PreparedCrystalPacker,
    standard_space_group_setting,
    VolumePolicy,
)
from phydrax.discretization import ParticleImageCapacity


def _argon() -> ConformerPool:
    component = MolecularComponent(
        (18,),
        (),
        bond_orders=(),
        masses=(39.948,),
        spin_multiplicity=1,
        valence="unchecked",
    )
    return ConformerPool(
        component,
        (MolecularConformer(component, np.zeros((1, 3)), provenance="single atom"),),
    )


def _packer(lanes: int) -> PreparedCrystalPacker:
    width = 4096
    plan = CrystalPackingPlan(
        MolecularCrystalInput(
            (_argon(),),
            contact=np.full((1, 1), 3.4),
            volume=VolumePolicy(formula_unit_volume=40.0),
        ),
        CrystalSpaceGroupPolicy(
            [standard_space_group_setting(number) for number in (1, 2)],
            weights=(0.5, 0.5),
            provenance="uniform demonstration prior",
        ),
        z_prime=1,
        chirality="enantiopure",
        resources=CrystalContactResources(
            AtomisticDenseImageSearch(maximum_dense_routes=lanes * 200_000),
            search_capacities=(
                ParticleImageCapacity(
                    maximum_particles_per_cell=32,
                    maximum_edges=lanes * width,
                    maximum_degree=width,
                    maximum_images=4096,
                    maximum_case_edges=width,
                ),
            ),
            role_capacities=(
                AtomisticRoleCapacity(maximum_routes=lanes * width, maximum_degree=width),
            ),
            skin=0.5,
        ),
        cell=CrystalCellPolicy(minimum_height=3.0, minimum_height_floor=2.5),
        relief=CrystalReliefPolicy(maximum_steps=200, check_interval=5),
    )
    return plan.prepare(lanes=lanes)


def _describe(records: Sequence[AtomisticGeneratedProposal]) -> list[str]:
    lines = []
    for record in records:
        reasons = (
            "accepted"
            if record.status == 0
            else "|".join(
                f.name for f in AtomisticProposalStatus if f and record.status & f
            )
        )
        lines.append(
            f"  ordinal {record.ordinal:2d} group {int(record.value('group_index'))}: {reasons}"
        )
    return lines


def main() -> None:
    packer = _packer(lanes=2)
    stream = packer.stream(
        "generation-stream-example", filters=(AtomisticMinimumDistanceFilter(3.3),)
    ).prepare()

    for target, budget in ((0, 4), (3, 0)):
        state = stream.start(jr.key(0), target=target, attempt_budget=budget)
        sink = AtomisticMemoryGenerationSink("zero-work")
        state = stream.run(state, sink, maximum_transitions=10)
        print(
            f"target={target} budget={budget}: {state.stop_reason.name}, {len(sink.records)} records"
        )

    # Uninterrupted reference segment.
    reference = AtomisticMemoryGenerationSink("reference")
    stream.run(
        stream.start(jr.key(1), target=4, attempt_budget=6),
        reference,
        maximum_transitions=500,
    )

    # The same segment interrupted, checkpointed and restored in a fresh stream.
    sink = AtomisticMemoryGenerationSink("restarted")
    state = stream.run(
        stream.start(jr.key(1), target=4, attempt_budget=6), sink, maximum_transitions=3
    )
    path = Path(tempfile.mkdtemp()) / "stream.npz"
    stream.write_checkpoint(path, state)
    fresh = (
        _packer(lanes=2)
        .stream(
            "generation-stream-example", filters=(AtomisticMinimumDistanceFilter(3.3),)
        )
        .prepare()
    )
    state = fresh.run(fresh.read_checkpoint(path), sink, maximum_transitions=500)
    # Proposal IDs identify addresses, not numerical payloads or failure evidence.
    reference_revision = NumericRevision(stream.plan.stream_id, reference.records)
    restarted_revision = NumericRevision(fresh.plan.stream_id, sink.records)
    identical = reference_revision.revision_id == restarted_revision.revision_id
    if not identical:
        raise RuntimeError("Restart changed proposal metadata or numerical payloads.")
    print(f"restart reproduces the reference: {identical}")
    print("\n".join(_describe(sink.records)))
    print(f"segment: {state.stop_reason.name}, shortfall {fresh.shortfall(state)}")

    # An explicit continuation segment raises the limits; nothing is redone.
    state = fresh.extend(state, target=8, attempt_budget=12)
    state = fresh.run(state, sink, maximum_transitions=500)
    print(
        f"after extend: {state.stop_reason.name}, {len(sink.records)} published, shortfall {fresh.shortfall(state)}"
    )


if __name__ == "__main__":
    main()
