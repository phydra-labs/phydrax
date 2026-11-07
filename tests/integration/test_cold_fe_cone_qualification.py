"""Fresh-process owner-local physical FE restart qualification."""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_native_cold_owner_local_physical_cone_continuation() -> None:
    from tools.meshing_qualification.scenarios.cold_owner_local import (
        native_cold_owner_local,
    )

    result = native_cold_owner_local()
    assert result["status"] == "passed"
    assert result["producer_processes_exited_before_each_reader"]
    assert result["profiles"]
    for profile in result["profiles"]:
        assert profile["status"] == "passed"
        assert profile["producer_processes_exited_before_reader"]
        assert profile["fresh_processes"]
        archives = profile["physical_cone_archive_bytes_by_rank"]
        assert len(archives) == 3 and all(size > 0 for size in archives)
        evidence = profile["physical_cone_continuation_by_rank"]
        assert len(evidence) == 3
        writer_ids = profile["physical_cone_writer_manifest_ids_by_rank"]
        assert len(writer_ids) == 3
        assert writer_ids == [item["source_manifest_id"] for item in evidence]
        global_cells = evidence[0]["cells"]
        assert global_cells > 0
        assert all(item["cells"] == global_cells for item in evidence)
        owned_cells = [set(item["owned_cell_digests"]) for item in evidence]
        assert sum(map(len, owned_cells)) == len(set.union(*owned_cells)) == global_cells
        for item in evidence:
            assert item
            assert all(
                item[name]
                for name in (
                    "changed_placement",
                    "independent_reference",
                    "newmark",
                    "linear_momentum",
                    "material_commit_trial_version",
                    "rollback",
                    "history_velocity_acceleration_timestep_sensitive",
                )
            )
            assert all(
                item[name]
                for name in (
                    "source_manifest_id",
                    "adapted_manifest_id",
                    "target_manifest_id",
                )
            )
