#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path
from typing import Any

import pytest


def test_native_owner_local_fv_archive_cold_repartitions_and_continues(
    tmp_path: Any,
) -> None:
    """Cold-read typed state in a new three-owner group before graph migration."""
    import json
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    from phydrax import config as pxconfig
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchError,
        LocalProcessLaunchPlan,
    )

    jax = sys.modules.get("jax")
    if jax is not None and jax.distributed.is_initialized() and jax.process_count() != 1:
        pytest.skip("The cold FV coordinator must run outside a live process group.")
    if os.name != "posix":
        pytest.skip("The cold FV process group requires POSIX supervision.")
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        pytest.skip("The three-owner cold FV qualification runs outside xdist.")
    repository = Path(__file__).resolve().parents[3]
    artifact_root = Path(tmp_path) / "owner-local-fv-cold"
    artifact_root.mkdir()
    base_environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        base_environment.pop(name, None)
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=base_environment,
    )
    environment.update(
        PYTHONPATH=str(repository),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )

    def launch_phase(root: Path, phase: str) -> list[dict[str, Any]]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        try:
            results = launch_local_processes(
                (
                    sys.executable,
                    "-m",
                    "tools.meshing_qualification.scenarios.cold_owner_local_fv",
                    str(root),
                    phase,
                ),
                LocalProcessLaunchPlan(3, f"127.0.0.1:{port}", devices_per_process=1),
                environment=environment,
                cwd=str(repository),
                timeout_seconds=900.0,
            )
        except LocalProcessLaunchError as error:
            output_directory = root / "rank-failure-logs" / phase
            output_directory.mkdir(parents=True, exist_ok=True)
            for result in error.results:
                rank_directory = output_directory / f"rank-{result.process_id}"
                rank_directory.mkdir(exist_ok=True)
                (rank_directory / "stdout.txt").write_bytes(result.stdout)
                (rank_directory / "stderr.txt").write_bytes(result.stderr)
                (rank_directory / "status.txt").write_text(
                    f"returncode={result.returncode}\n"
                    f"output_truncated={result.output_truncated}\n",
                    encoding="utf-8",
                )
            error.add_note(f"Full per-rank output saved under {output_directory}.")
            raise
        reports = [
            json.loads(result.stdout.decode().strip().splitlines()[-1])
            for result in results
        ]
        assert [item["rank"] for item in reports] == [0, 1, 2]
        assert all(item["phase"] == phase for item in reports)
        return reports

    launch_phase(artifact_root, "write")
    for rank in range(3):
        assert (artifact_root / f"owner-local-fv-state-{rank}.phx").is_file()
    launch_phase(artifact_root, "read")

    tampered_root = Path(tmp_path) / "owner-local-fv-tampered"
    subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            str(artifact_root),
            str(tampered_root),
        ),
        cwd=str(repository),
        env=environment,
        check=True,
    )

    with pytest.raises(LocalProcessLaunchError) as refused:
        launch_phase(tampered_root, "read")
    assert any(
        "Restored source state identity changed" in result.stderr.decode(errors="replace")
        for result in refused.value.results
    )


def _tamper_archive(artifact_root: Path, tampered_root: Path) -> None:
    import shutil

    from phydrax._model._structure import FiniteVolumeRuntimeStateArchiveRecipe
    from phydrax.meshing._model_codec import read_model_archive, write_model_archive

    state_identity_index = FiniteVolumeRuntimeStateArchiveRecipe._fields.index(
        "expected_state_id"
    )
    tampered_root.mkdir()
    for rank in range(3):
        source = artifact_root / f"owner-local-fv-state-{rank}.phx"
        destination = tampered_root / source.name
        if rank != 0:
            shutil.copyfile(source, destination)
            continue
        manifest, arrays = read_model_archive(source, limits=None)

        def tamper_state_identity(node: Any) -> bool:
            if isinstance(node, dict):
                if str(node.get("type", "")).endswith(
                    "FiniteVolumeRuntimeStateArchiveRecipe"
                ):
                    fields = node.get("fields")
                    items = node.get("items")
                    identity = (
                        fields.get("expected_state_id")
                        if isinstance(fields, dict)
                        else items[state_identity_index]
                        if isinstance(items, list) and len(items) > state_identity_index
                        else None
                    )
                    if isinstance(identity, dict) and isinstance(
                        identity.get("value"), str
                    ):
                        current = identity["value"]
                        identity["value"] = "0" * 64 if current != "0" * 64 else "1" * 64
                        return True
                return any(tamper_state_identity(child) for child in node.values())
            if isinstance(node, list):
                return any(tamper_state_identity(child) for child in node)
            return False

        assert tamper_state_identity(manifest["model_recipe"])
        write_model_archive(destination, manifest=manifest, arrays=arrays, limits=None)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "run":
        test_native_owner_local_fv_archive_cold_repartitions_and_continues(
            Path(sys.argv[2])
        )
    else:
        _tamper_archive(Path(sys.argv[1]), Path(sys.argv[2]))
