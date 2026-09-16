#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax._fingerprint import canonical_fingerprint
from phydrax.lifecycle._event_graph_repository import (
    EventGraphRepository,
    GlobalWorkItem,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._dark_sector_epoch_runtime import (
    admit_dark_sector_work,
    DarkSectorEpochPlan,
    DarkSectorRunCoordinator,
    finalize_dark_sector_epoch,
)


def _digest(name):
    return canonical_fingerprint({"integration": name})


def test_unbounded_committed_epoch_chain_uses_tiny_fixed_resident_pools(tmp_path):
    profile = HPCFilesystemProfile(
        "unbounded-dark-sector-posix",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    artifacts = POSIXArtifactRepository(
        tmp_path / "repository",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=512,
            maximum_metadata_bytes=64 * 1024,
        ),
    )
    graph = EventGraphRepository(artifacts, maximum_record_bytes=1024 * 1024)
    plan = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=1,
        product_capacity=1,
        radiation_capacity=1,
        work_capacity=1,
        frontier_capacity=1,
        packet_width=4,
        event_width=2,
        product_width=4,
        radiation_width=4,
        work_width=2,
        frontier_width=2,
        species_revision_id=_digest("species"),
        topology_revision_id=_digest("topology"),
    )
    coordinator = DarkSectorRunCoordinator(graph, plan, "long-run", worker_id="worker-a")
    parent_work = None
    epoch_count = 32
    for epoch in range(epoch_count):
        resume = coordinator.resume()
        state = resume.state
        new_count = 2 if epoch == 0 else 1
        new_work = []
        values = []
        for local in range(new_count):
            work = GlobalWorkItem(
                "dark-cascade",
                (),
                _digest("matrix-element"),
                partition_key=f"shard-{epoch % 2}",
                priority=epoch * 2 + local,
                parent_work_id=None if parent_work is None else parent_work.work_id,
            )
            parent_work = work
            new_work.append(work)
            values.append((float(epoch), float(local)))
        admission = admit_dark_sector_work(
            state,
            tuple(work.work_id for work in new_work),
            jnp.asarray(values),
        )
        assert not bool(admission.refused)
        assert bool(admission.backpressured)
        result = finalize_dark_sector_epoch(
            state,
            admission.state,
            complete=True,
            backpressured=admission.backpressured,
            evidence_ids=(f"epoch-{epoch}-conservation",),
        )
        receipt = coordinator.commit_epoch(
            result,
            work_items=tuple(new_work),
            matrix_element_revision_id=_digest("matrix-element"),
            committed_at=epoch + 1,
        )
        assert receipt.tip.epoch_sequence == epoch
        assert max(map(int, result.resident_high_water)) <= 1

    tip = graph.run_tip("long-run")
    assert tip.epoch_sequence == epoch_count - 1
    lineage = []
    current = tip.epoch_manifest_id
    while current is not None:
        persisted = graph.load_epoch(current)
        lineage.append(persisted.manifest.epoch_sequence)
        current = persisted.manifest.parent_manifest_id
    assert lineage == list(reversed(range(epoch_count)))
    final_resume = coordinator.resume()
    assert final_resume.state.epoch_sequence == epoch_count
    assert int(jnp.sum(final_resume.state.work_mask)) == 1
    assert int(jnp.sum(final_resume.state.frontier_mask)) == 0
